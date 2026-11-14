import {
  BandRuntime,
  BandSpec,
  DustPuff,
  PigmentKey,
  RubberBallDiagnostics,
  SEGMENTS_PER_BAND,
} from './types';

const TOTAL_BANDS = 180;
const NUM_LAYERS = 12;
const CORE_RADIUS = 0.28;
const CORE_CLEARANCE = 0.012;
const OUTER_RADIUS = 0.95;
const MAX_BASE_RADIUS = 0.908;
const MAX_DUST_PUFFS = 384;
const MAX_CASCADE_QUEUE = 16;
const RINGS_PER_BAND = SEGMENTS_PER_BAND + 1;

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const TWO_PI = Math.PI * 2;
const STRAIN_PER_UNIT_PULL = 0.62;
const TWANG_FREQ = 68.0;
const TWANG_DAMP = 5.4;
const GRAVITY = 5.6;

const BAND_PIGMENTS: readonly PigmentKey[] = [
  'rawUmber',
  'yellowOchre',
  'boneCalcite',
  'cinnabarVermilion',
  'burntSienna',
  'celadonGlauconite',
  'azuritePrussian',
];

/**
 * Deterministic PRNG (Mulberry32) so ball wrapping and unit tests are reproducible.
 */
function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shortestArcDist(a: number, b: number): number {
  let d = Math.abs((a % 1) - (b % 1));
  if (d > 0.5) d = 1 - d;
  return d;
}

interface CascadeEntry {
  active: boolean;
  bandId: number;
  delaySec: number;
  uHit: number;
}

export class RubberBandBallSim {
  private readonly specs: BandSpec[] = [];
  private readonly runtimes: BandRuntime[] = [];
  private readonly dustPuffs: DustPuff[] = [];
  private readonly cascadeQueue: CascadeEntry[] = [];

  private readonly initialOuterLoad = new Float32Array(TOTAL_BANDS);
  private readonly targetRadius = new Float32Array(TOTAL_BANDS);
  private readonly effectiveRadius = new Float32Array(TOTAL_BANDS);
  private readonly unitCirclePts = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND * 3);
  private readonly radialProfile = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND);
  private readonly maxBandRadius = new Float32Array(TOTAL_BANDS);

  // Internal snap-kinematics state (no types.ts churn required)
  private readonly snapAgeSec = new Float32Array(TOTAL_BANDS);
  private readonly snapPeakPull = new Float32Array(TOTAL_BANDS);
  private readonly snapAsym = new Float32Array(TOTAL_BANDS);
  private readonly snapLaunchDir = new Float32Array(TOTAL_BANDS * 3);

  private activePullBandId: number | null = null;
  private snapCount = 0;
  private simTimeSec = 0;
  private rng = createRng(20261114);
  private readonly seed: number;

  // Damped spring recoil of the whole ball on each band pop
  private readonly recoilPos: [number, number, number] = [0, 0, 0];
  private readonly recoilVel: [number, number, number] = [0, 0, 0];

  // Pre-allocated scratch for zero-GC raycast & pull tracking
  private readonly scratchPoint: [number, number, number] = [0, 0, 0];
  private readonly pullBaseNormal: [number, number, number] = [0, 0, 1];
  private readonly pullDragVec: [number, number, number] = [0, 0, 0];

  constructor(seed = 20261114) {
    this.seed = seed;
    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      this.dustPuffs.push({
        active: false,
        pos: [0, 0, 0],
        vel: [0, 0, 0],
        ageSec: 0,
        maxAgeSec: 0.6,
        size: 0.03,
      });
    }
    for (let i = 0; i < MAX_CASCADE_QUEUE; i++) {
      this.cascadeQueue.push({
        active: false,
        bandId: -1,
        delaySec: 0,
        uHit: 0,
      });
    }
    this.reset();
  }

  public reset(): void {
    this.rng = createRng(this.seed);
    this.specs.length = 0;
    this.runtimes.length = 0;
    this.activePullBandId = null;
    this.snapCount = 0;

    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      this.dustPuffs[i].active = false;
    }
    for (let i = 0; i < MAX_CASCADE_QUEUE; i++) {
      this.cascadeQueue[i].active = false;
    }

    const bandsPerLayer = Math.ceil(TOTAL_BANDS / NUM_LAYERS);

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const t = i / (TOTAL_BANDS - 1);
      const layer = Math.min(NUM_LAYERS - 1, Math.floor(t * NUM_LAYERS));
      const indexInLayer = i % bandsPerLayer;

      // Stratified Fibonacci-hemisphere normal with organic jitter
      const yBase = 1 - (indexInLayer + 0.5) / bandsPerLayer;
      const y = Math.max(0.04, Math.min(0.995, yBase + (this.rng() - 0.5) * 0.06));
      const rSin = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = indexInLayer * GOLDEN_ANGLE + layer * 1.17 + (this.rng() - 0.5) * 0.14;

      let nx = rSin * Math.cos(theta);
      let ny = y;
      let nz = rSin * Math.sin(theta);
      const nLen = Math.hypot(nx, ny, nz) || 1;
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      // Orthonormal basis (tangentU, tangentV) rotated by random phase around normal
      const refX = Math.abs(ny) < 0.9 ? 0 : 1;
      const refY = Math.abs(ny) < 0.9 ? 1 : 0;
      const refZ = 0;

      // ref x normal
      let ux = refY * nz - refZ * ny;
      let uy = refZ * nx - refX * nz;
      let uz = refX * ny - refY * nx;
      const uLen = Math.hypot(ux, uy, uz) || 1;
      ux /= uLen;
      uy /= uLen;
      uz /= uLen;

      // normal x tangentU
      const vx0 = ny * uz - nz * uy;
      const vy0 = nz * ux - nx * uz;
      const vz0 = nx * uy - ny * ux;

      const phase = this.rng() * TWO_PI;
      const cp = Math.cos(phase);
      const sp = Math.sin(phase);
      const txU: [number, number, number] = [
        ux * cp + vx0 * sp,
        uy * cp + vy0 * sp,
        uz * cp + vz0 * sp,
      ];
      const txV: [number, number, number] = [
        vx0 * cp - ux * sp,
        vy0 * cp - uy * sp,
        vz0 * cp - uz * sp,
      ];

      const thickness = 0.008 + this.rng() * 0.007;
      const minBaseRadius = CORE_RADIUS + CORE_CLEARANCE + thickness * 0.5 + 0.004;
      const baseRadius = minBaseRadius + t * (MAX_BASE_RADIUS - minBaseRadius);
      const width = (0.034 + this.rng() * 0.036) * 1.22;

      // Stratified pigment selection
      const pigment = this.pickStratifiedPigment(layer);

      // Inner layers are compressed at lower restStrain; outer layers carry higher preload
      const restStrain = 0.18 + t * 0.34 + (this.rng() - 0.5) * 0.08;
      const breakStrain = 0.78 + this.rng() * 0.28;

      this.specs.push({
        id: i,
        layer,
        baseRadius,
        width,
        thickness,
        pigment,
        restStrain: Math.max(0.1, Math.min(0.66, restStrain)),
        breakStrain: Math.max(restStrain + 0.16, breakStrain),
        normal: [nx, ny, nz],
        tangentU: txU,
        tangentV: txV,
      });

      this.runtimes.push({
        id: i,
        status: 'intact',
        currentStrain: Math.max(0.1, Math.min(0.66, restStrain)),
        pullU: 0,
        pullAmount: 0,
        pullDir: [0, 0, 0],
        twangPhase: 0,
        twangAmp: 0,
        snapU: 0,
        leftArc: 0.5,
        rightArc: 0.5,
        leftVel: 0,
        rightVel: 0,
        curlAmount: 0,
        fallOffset: [0, 0, 0],
        fallVel: [0, 0, 0],
        opacity: 1,
      });

      this.targetRadius[i] = baseRadius;
      this.effectiveRadius[i] = baseRadius;
      this.snapAgeSec[i] = 0;
      this.snapPeakPull[i] = 0;
      this.snapAsym[i] = 1;

      const ringOffset = i * RINGS_PER_BAND * 3;
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const theta = (k / SEGMENTS_PER_BAND) * TWO_PI;
        const c = Math.cos(theta);
        const s = Math.sin(theta);
        const idx = ringOffset + k * 3;
        this.unitCirclePts[idx] = c * txU[0] + s * txV[0];
        this.unitCirclePts[idx + 1] = c * txU[1] + s * txV[1];
        this.unitCirclePts[idx + 2] = c * txU[2] + s * txV[2];
      }
    }

    // Compute initial compressive preload exerted by outer layers on inner layers
    for (let i = 0; i < TOTAL_BANDS; i++) {
      this.initialOuterLoad[i] = this.computeOuterCompressionLoad(i);
    }

    // Build layered non-intersecting crossing-wrap profiles
    this.rebuildRadialProfiles();
  }

  private pickStratifiedPigment(layer: number): PigmentKey {
    const r = this.rng();
    if (layer <= 2) {
      if (r < 0.4) return 'rawUmber';
      if (r < 0.7) return 'boneCalcite';
      if (r < 0.9) return 'yellowOchre';
      return 'burntSienna';
    }
    if (layer <= 6) {
      if (r < 0.25) return 'burntSienna';
      if (r < 0.5) return 'yellowOchre';
      if (r < 0.72) return 'cinnabarVermilion';
      if (r < 0.88) return 'boneCalcite';
      return 'rawUmber';
    }
    if (r < 0.22) return 'cinnabarVermilion';
    if (r < 0.42) return 'celadonGlauconite';
    if (r < 0.6) return 'azuritePrussian';
    if (r < 0.78) return 'yellowOchre';
    if (r < 0.92) return 'boneCalcite';
    return BAND_PIGMENTS[Math.floor(this.rng() * BAND_PIGMENTS.length)];
  }

  private computeOuterCompressionLoad(bandIdx: number): number {
    const layerI = this.specs[bandIdx].layer;
    let load = 0;
    for (let j = bandIdx + 1; j < TOTAL_BANDS; j++) {
      const rtJ = this.runtimes[j];
      if (rtJ.status !== 'intact' && rtJ.status !== 'pulled') continue;
      const dLayer = this.specs[j].layer - layerI;
      if (dLayer <= 0) continue;
      load += (0.045 + 0.065 * rtJ.currentStrain) / (1 + 0.35 * dLayer);
    }
    return load;
  }

  private rebuildRadialProfiles(): void {
    const coreMinCenter = CORE_RADIUS + CORE_CLEARANCE;
    const shoulderWidth = 0.008;
    const contactGap = 0.0025;

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rtI = this.runtimes[i];
      if (rtI.status !== 'intact' && rtI.status !== 'pulled') continue;
      const specI = this.specs[i];
      const rBase = this.targetRadius[i];
      const halfHI = specI.thickness * 0.5;
      const minCoreR = coreMinCenter + halfHI;
      const profOffset = i * RINGS_PER_BAND;
      const ptsOffsetI = profOffset * 3;

      // Initialize profile at tensioned base radius (clamped outside core)
      const initR = Math.max(rBase, minCoreR);
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        this.radialProfile[profOffset + k] = initR;
      }

      // Resolve physical crossing wrap over intact/pulled underlying bands j < i
      for (let j = 0; j < i; j++) {
        const rtJ = this.runtimes[j];
        if (rtJ.status !== 'intact' && rtJ.status !== 'pulled') continue;

        const specJ = this.specs[j];
        const halfHJ = specJ.thickness * 0.5;
        // Quick radial cull: skip j if its highest point is well below i's base
        if (this.maxBandRadius[j] + halfHJ + halfHI + contactGap <= initR) continue;

        const nj = specJ.normal;
        const uj = specJ.tangentU;
        const vj = specJ.tangentV;
        const halfWJ = specJ.width * 0.5;
        const crossLimit = halfWJ + shoulderWidth;
        const profOffsetJ = j * RINGS_PER_BAND;

        for (let k = 0; k < RINGS_PER_BAND; k++) {
          const pIdx = ptsOffsetI + k * 3;
          const ux = this.unitCirclePts[pIdx];
          const uy = this.unitCirclePts[pIdx + 1];
          const uz = this.unitCirclePts[pIdx + 2];

          const latDot = Math.abs(ux * nj[0] + uy * nj[1] + uz * nj[2]);
          const latDist = latDot * initR;
          if (latDist >= crossLimit) continue;

          // Sample band j's actual wrapped radius at the closest point along j
          const pu = ux * uj[0] + uy * uj[1] + uz * uj[2];
          const pv = ux * vj[0] + uy * vj[1] + uz * vj[2];
          let angleJ = Math.atan2(pv, pu);
          if (angleJ < 0) angleJ += TWO_PI;
          const kJFloat = (angleJ / TWO_PI) * SEGMENTS_PER_BAND;
          const kJ0 = Math.floor(kJFloat) % SEGMENTS_PER_BAND;
          const kJ1 = (kJ0 + 1) % SEGMENTS_PER_BAND;
          const rJ = Math.max(
            this.radialProfile[profOffsetJ + kJ0],
            this.radialProfile[profOffsetJ + kJ1]
          );

          const minClearR = rJ + halfHJ + halfHI + contactGap;
          if (minClearR <= this.radialProfile[profOffset + k]) continue;

          const ramp =
            latDist <= halfWJ
              ? 1.0
              : 0.5 + 0.5 * Math.cos((Math.PI * (latDist - halfWJ)) / shoulderWidth);
          const wrappedR = initR + (minClearR - initR) * ramp;
          if (wrappedR > this.radialProfile[profOffset + k]) {
            this.radialProfile[profOffset + k] = wrappedR;
          }
        }
      }

      let maxR = initR;
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const rk = this.radialProfile[profOffset + k];
        if (rk > maxR) maxR = rk;
      }
      this.maxBandRadius[i] = maxR;
    }
  }

  private recomputeCompression(triggerCascadeFromId: number | null): void {
    let cascadeScheduled = 0;
    const srcSpec = triggerCascadeFromId !== null ? this.specs[triggerCascadeFromId] : null;

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      if (rt.status !== 'intact' && rt.status !== 'pulled') continue;

      const spec = this.specs[i];
      const currentLoad = this.computeOuterCompressionLoad(i);
      const releasedLoad = Math.max(0, this.initialOuterLoad[i] - currentLoad);

      // Subtle breathing expansion as outer compression is relieved
      const breathDelta = Math.min(0.055, releasedLoad * 0.022);
      this.targetRadius[i] = spec.baseRadius + breathDelta;

      // Released clamping slightly redistributes hoop tension onto remaining intact bands
      if (rt.status === 'intact') {
        const tensionReleaseBump = Math.min(0.08, releasedLoad * 0.024);
        rt.currentStrain = Math.min(spec.breakStrain * 1.02, spec.restStrain + tensionReleaseBump);
      }

      // Sympathetic cascade check: adjacent over-strained bands can pop shortly after a neighbor snaps
      if (
        srcSpec !== null &&
        i !== srcSpec.id &&
        rt.status === 'intact' &&
        cascadeScheduled < 2 &&
        Math.abs(spec.layer - srcSpec.layer) <= 2
      ) {
        const align = Math.abs(
          spec.normal[0] * srcSpec.normal[0] +
            spec.normal[1] * srcSpec.normal[1] +
            spec.normal[2] * srcSpec.normal[2]
        );
        if (align > 0.6 && rt.currentStrain >= spec.breakStrain * 0.88) {
          this.enqueueCascade(i, 0.04 + this.rng() * 0.06, this.rng());
          cascadeScheduled++;
        }
      }
    }

    this.rebuildRadialProfiles();
  }

  private enqueueCascade(bandId: number, delaySec: number, uHit: number): void {
    for (let i = 0; i < MAX_CASCADE_QUEUE; i++) {
      const entry = this.cascadeQueue[i];
      if (entry.active && entry.bandId === bandId) return;
    }
    for (let i = 0; i < MAX_CASCADE_QUEUE; i++) {
      const entry = this.cascadeQueue[i];
      if (!entry.active) {
        entry.active = true;
        entry.bandId = bandId;
        entry.delaySec = delaySec;
        entry.uHit = uHit;
        return;
      }
    }
  }

  public startPull(bandId: number, uHit: number, pullDirWorld: [number, number, number]): void {
    if (bandId < 0 || bandId >= TOTAL_BANDS) return;
    const rt = this.runtimes[bandId];
    if (rt.status !== 'intact') return;

    if (this.activePullBandId !== null && this.activePullBandId !== bandId) {
      this.releasePull();
    }

    const uNorm = ((uHit % 1) + 1) % 1;
    let dx = pullDirWorld[0];
    let dy = pullDirWorld[1];
    let dz = pullDirWorld[2];
    const len = Math.hypot(dx, dy, dz);
    if (len > 1e-5) {
      dx /= len;
      dy /= len;
      dz /= len;
    } else {
      this.evalCirclePoint(bandId, uNorm, this.scratchPoint);
      const rLen = Math.hypot(this.scratchPoint[0], this.scratchPoint[1], this.scratchPoint[2]) || 1;
      dx = this.scratchPoint[0] / rLen;
      dy = this.scratchPoint[1] / rLen;
      dz = this.scratchPoint[2] / rLen;
    }

    rt.status = 'pulled';
    rt.pullU = uNorm;
    rt.pullAmount = 0;
    rt.pullDir[0] = dx;
    rt.pullDir[1] = dy;
    rt.pullDir[2] = dz;
    this.pullBaseNormal[0] = dx;
    this.pullBaseNormal[1] = dy;
    this.pullBaseNormal[2] = dz;
    this.pullDragVec[0] = 0;
    this.pullDragVec[1] = 0;
    this.pullDragVec[2] = 0;
    rt.twangAmp = 0;
    this.activePullBandId = bandId;
  }

  public updatePull(worldDelta: [number, number, number]): boolean {
    if (this.activePullBandId === null) return false;
    const id = this.activePullBandId;
    const rt = this.runtimes[id];
    const spec = this.specs[id];
    if (rt.status !== 'pulled') {
      this.activePullBandId = null;
      return false;
    }

    this.pullDragVec[0] += worldDelta[0];
    this.pullDragVec[1] += worldDelta[1];
    this.pullDragVec[2] += worldDelta[2];

    const vx = this.pullDragVec[0];
    const vy = this.pullDragVec[1];
    const vz = this.pullDragVec[2];
    const dragLen = Math.hypot(vx, vy, vz);

    const nx = this.pullBaseNormal[0];
    const ny = this.pullBaseNormal[1];
    const nz = this.pullBaseNormal[2];
    const radialProj = vx * nx + vy * ny + vz * nz;

    rt.pullAmount = Math.max(0, Math.max(radialProj, dragLen * 0.85));
    rt.currentStrain = spec.restStrain + rt.pullAmount * STRAIN_PER_UNIT_PULL;

    if (dragLen > 1e-4) {
      const invLen = 1 / dragLen;
      let px = nx * 0.55 + vx * invLen * 0.85;
      let py = ny * 0.55 + vy * invLen * 0.85;
      let pz = nz * 0.55 + vz * invLen * 0.85;
      const pLen = Math.hypot(px, py, pz) || 1;
      rt.pullDir[0] = px / pLen;
      rt.pullDir[1] = py / pLen;
      rt.pullDir[2] = pz / pLen;
    }

    if (rt.currentStrain >= spec.breakStrain) {
      this.activePullBandId = null;
      this.severBand(id, rt.pullU, true);
      return true;
    }

    return false;
  }

  public releasePull(): void {
    if (this.activePullBandId === null) return;
    const id = this.activePullBandId;
    const rt = this.runtimes[id];
    const spec = this.specs[id];
    this.activePullBandId = null;

    if (rt.status !== 'pulled') return;

    rt.status = 'intact';
    rt.twangAmp = rt.pullAmount;
    rt.twangPhase = 0;
    rt.pullAmount = 0;
    rt.currentStrain = spec.restStrain;
  }

  public nickBand(bandId: number, uHit: number): void {
    if (bandId < 0 || bandId >= TOTAL_BANDS) return;
    const rt = this.runtimes[bandId];
    if (rt.status !== 'intact' && rt.status !== 'pulled') return;

    if (this.activePullBandId === bandId) {
      this.activePullBandId = null;
    }

    this.severBand(bandId, uHit, false);
  }

  private severBand(bandId: number, uHit: number, fromPull: boolean): void {
    const rt = this.runtimes[bandId];
    const spec = this.specs[bandId];
    if (rt.status === 'snapping' || rt.status === 'gone') return;

    const uNorm = ((uHit % 1) + 1) % 1;
    const strainFactor = Math.max(0.5, rt.currentStrain);
    const peakPull = fromPull ? rt.pullAmount : 0;

    this.snapAgeSec[bandId] = 0;
    this.snapPeakPull[bandId] = peakPull;
    const asym = 0.86 + this.rng() * 0.28;
    this.snapAsym[bandId] = asym;
    if (fromPull) {
      this.snapLaunchDir[bandId * 3] = rt.pullDir[0];
      this.snapLaunchDir[bandId * 3 + 1] = rt.pullDir[1];
      this.snapLaunchDir[bandId * 3 + 2] = rt.pullDir[2];
    }

    rt.status = 'snapping';
    rt.snapU = uNorm;
    rt.leftArc = 0.5;
    rt.rightArc = 0.5;
    const baseWhipVel = (3.6 + strainFactor * 2.4) * (fromPull ? 1.18 : 1.0);
    rt.leftVel = baseWhipVel * asym;
    rt.rightVel = baseWhipVel / asym;
    rt.curlAmount = 0;
    rt.pullAmount = 0;
    rt.twangAmp = 0;
    rt.opacity = 1;

    // Outward radial vector at the snap point
    this.evalCirclePoint(bandId, uNorm, this.scratchPoint);
    const pLen = Math.hypot(this.scratchPoint[0], this.scratchPoint[1], this.scratchPoint[2]) || 1;
    const radX = this.scratchPoint[0] / pLen;
    const radY = this.scratchPoint[1] / pLen;
    const radZ = this.scratchPoint[2] / pLen;

    if (!fromPull) {
      this.snapLaunchDir[bandId * 3] = radX;
      this.snapLaunchDir[bandId * 3 + 1] = radY;
      this.snapLaunchDir[bandId * 3 + 2] = radZ;
    }

    // Lateral tangent at snap point so the band peels & flings off the side rather than tunneling through the ball
    const thetaSnap = uNorm * TWO_PI;
    const tx = -Math.sin(thetaSnap) * spec.tangentU[0] + Math.cos(thetaSnap) * spec.tangentV[0];
    const ty = -Math.sin(thetaSnap) * spec.tangentU[1] + Math.cos(thetaSnap) * spec.tangentV[1];
    const tz = -Math.sin(thetaSnap) * spec.tangentU[2] + Math.cos(thetaSnap) * spec.tangentV[2];
    const latSign = this.rng() < 0.5 ? -1 : 1;

    const outwardKick = fromPull ? 0.55 : 0.35;
    const sideKick = (fromPull ? 1.15 : 0.78) * latSign;
    rt.fallOffset[0] = 0;
    rt.fallOffset[1] = 0;
    rt.fallOffset[2] = 0;
    rt.fallVel[0] = radX * outwardKick + tx * sideKick + (this.rng() - 0.5) * 0.45;
    rt.fallVel[1] = radY * outwardKick + ty * sideKick + (fromPull ? 1.15 : 0.85);
    rt.fallVel[2] = radZ * outwardKick + tz * sideKick + (this.rng() - 0.5) * 0.45;

    // Crisp recoil jolt on the whole ball opposite the snap point
    const recoilImp = (fromPull ? 1.15 : 0.72) * (0.75 + strainFactor * 0.55);
    this.recoilVel[0] -= radX * recoilImp;
    this.recoilVel[1] -= radY * recoilImp;
    this.recoilVel[2] -= radZ * recoilImp;

    // Primary talc burst at snap point + secondary talc wake along whipping arc
    this.spawnDustPuffs(
      this.scratchPoint[0],
      this.scratchPoint[1],
      this.scratchPoint[2],
      radX,
      radY,
      radZ,
      spec.width,
      30,
      1.0 + strainFactor * 0.45
    );
    const wakeOffsets = [-0.1, 0.1] as const;
    for (let w = 0; w < wakeOffsets.length; w++) {
      const uWake = ((uNorm + wakeOffsets[w]) % 1 + 1) % 1;
      this.evalCirclePoint(bandId, uWake, this.scratchPoint);
      const wLen = Math.hypot(this.scratchPoint[0], this.scratchPoint[1], this.scratchPoint[2]) || 1;
      this.spawnDustPuffs(
        this.scratchPoint[0],
        this.scratchPoint[1],
        this.scratchPoint[2],
        this.scratchPoint[0] / wLen,
        this.scratchPoint[1] / wLen,
        this.scratchPoint[2] / wLen,
        spec.width,
        9,
        0.75 + strainFactor * 0.3
      );
    }

    // Sympathetic elastic shockwave twang across nearby intact bands
    for (let i = 0; i < TOTAL_BANDS; i++) {
      if (i === bandId) continue;
      const nrt = this.runtimes[i];
      if (nrt.status !== 'intact') continue;
      const nspec = this.specs[i];
      if (Math.abs(nspec.layer - spec.layer) > 4) continue;
      const align = Math.abs(
        nspec.normal[0] * spec.normal[0] +
          nspec.normal[1] * spec.normal[1] +
          nspec.normal[2] * spec.normal[2]
      );
      if (align > 0.25) {
        nrt.twangAmp = Math.max(nrt.twangAmp, 0.055 + 0.05 * strainFactor * align);
        nrt.twangPhase = 0;
        nrt.pullU = uNorm;
        nrt.pullDir[0] = radX;
        nrt.pullDir[1] = radY;
        nrt.pullDir[2] = radZ;
      }
    }

    this.snapCount++;
    this.recomputeCompression(bandId);
  }

  private spawnDustPuffs(
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
    bandWidth: number,
    targetCount = 30,
    energyScale = 1.0
  ): void {
    let spawned = 0;
    const cloudCount = Math.floor(targetCount * 0.65);
    for (let i = 0; i < MAX_DUST_PUFFS && spawned < targetCount; i++) {
      const puff = this.dustPuffs[i];
      if (puff.active) continue;

      const isFineSpeck = spawned >= cloudCount;
      puff.active = true;
      const spread = bandWidth * (isFineSpeck ? 1.1 : 1.6);
      puff.pos[0] = px + (this.rng() - 0.5) * spread;
      puff.pos[1] = py + (this.rng() - 0.5) * spread;
      puff.pos[2] = pz + (this.rng() - 0.5) * spread;

      const speed = (isFineSpeck ? 1.5 + this.rng() * 1.7 : 0.65 + this.rng() * 1.15) * energyScale;
      const jitter = (isFineSpeck ? 1.35 : 0.85) * energyScale;
      puff.vel[0] = nx * speed + (this.rng() - 0.5) * jitter;
      puff.vel[1] = ny * speed + (this.rng() - 0.5) * jitter + 0.18;
      puff.vel[2] = nz * speed + (this.rng() - 0.5) * jitter;

      puff.ageSec = 0;
      puff.maxAgeSec = isFineSpeck ? 0.24 + this.rng() * 0.24 : 0.48 + this.rng() * 0.42;
      puff.size = isFineSpeck ? 0.008 + this.rng() * 0.008 : 0.024 + this.rng() * 0.028;
      spawned++;
    }
  }

  public step(dtSec: number): void {
    const dt = Math.min(Math.max(dtSec, 0), 0.05);
    if (dt <= 0) return;
    this.simTimeSec += dt;

    // Damped spring recoil of the ball (leans toward active pull, snaps back on release/pop)
    const kSpring = 240.0;
    const kDamp = 14.0;
    const activePull =
      this.activePullBandId !== null ? this.runtimes[this.activePullBandId] : null;
    const leanAmt =
      activePull && activePull.status === 'pulled'
        ? Math.min(0.09, activePull.pullAmount * 0.085)
        : 0;
    for (let a = 0; a < 3; a++) {
      const targetPos = activePull ? activePull.pullDir[a] * leanAmt : 0;
      const acc = -kSpring * (this.recoilPos[a] - targetPos) - kDamp * this.recoilVel[a];
      this.recoilVel[a] += acc * dt;
      this.recoilPos[a] += this.recoilVel[a] * dt;
    }

    // 1. Process sympathetic cascade pops
    for (let i = 0; i < MAX_CASCADE_QUEUE; i++) {
      const entry = this.cascadeQueue[i];
      if (!entry.active) continue;
      entry.delaySec -= dt;
      if (entry.delaySec <= 0) {
        entry.active = false;
        const rt = this.runtimes[entry.bandId];
        if (rt && rt.status === 'intact') {
          this.severBand(entry.bandId, entry.uHit, false);
        }
      }
    }

    // 2. Update band physics (breathing, twang, whip-back & C-curl)
    const breathBlend = 1 - Math.exp(-12.0 * dt);
    const twangDecay = Math.exp(-TWANG_DAMP * dt);

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      const spec = this.specs[i];

      // Smooth breathing expansion
      this.effectiveRadius[i] += (this.targetRadius[i] - this.effectiveRadius[i]) * breathBlend;

      if (rt.status === 'intact') {
        if (rt.twangAmp > 1e-4) {
          rt.twangPhase += dt * TWANG_FREQ;
          rt.twangAmp *= twangDecay;
          if (rt.twangAmp < 1e-4) {
            rt.twangAmp = 0;
            rt.currentStrain = spec.restStrain;
          } else {
            rt.currentStrain =
              spec.restStrain + Math.abs(Math.sin(rt.twangPhase)) * rt.twangAmp * 0.4;
          }
        }
      } else if (rt.status === 'snapping') {
        this.snapAgeSec[i] += dt;
        const arcDamp = Math.exp(-2.6 * dt);
        rt.leftVel *= arcDamp;
        rt.rightVel *= arcDamp;
        rt.leftArc = Math.max(0.055, rt.leftArc - rt.leftVel * dt);
        rt.rightArc = Math.max(0.055, rt.rightArc - rt.rightVel * dt);
        rt.curlAmount = Math.min(1, rt.curlAmount + dt * 6.5);

        rt.fallVel[1] -= GRAVITY * dt;
        rt.fallOffset[0] += rt.fallVel[0] * dt;
        rt.fallOffset[1] += rt.fallVel[1] * dt;
        rt.fallOffset[2] += rt.fallVel[2] * dt;

        rt.currentStrain = Math.max(0, rt.currentStrain - dt * 5.0);

        // Keep coiled scrap visible while whipping & tumbling, then dissolve cleanly
        if (this.snapAgeSec[i] > 0.42 || rt.fallOffset[1] < -0.95) {
          rt.opacity = Math.max(0, rt.opacity - dt * 2.6);
        }

        if (rt.opacity <= 0.005) {
          rt.status = 'gone';
          rt.opacity = 0;
          rt.leftArc = 0;
          rt.rightArc = 0;
        }
      }
    }

    // 3. Update talcum dust puffs (strong air drag so fine powder bursts then hangs)
    const drag = Math.exp(-6.5 * dt);
    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      const puff = this.dustPuffs[i];
      if (!puff.active) continue;

      puff.ageSec += dt;
      if (puff.ageSec >= puff.maxAgeSec) {
        puff.active = false;
        continue;
      }

      puff.vel[0] *= drag;
      puff.vel[1] = puff.vel[1] * drag + 0.08 * dt;
      puff.vel[2] *= drag;

      puff.pos[0] += puff.vel[0] * dt;
      puff.pos[1] += puff.vel[1] * dt;
      puff.pos[2] += puff.vel[2] * dt;
    }
  }

  private evalCirclePoint(bandIndex: number, u: number, out: [number, number, number]): void {
    const spec = this.specs[bandIndex];
    const r = this.effectiveRadius[bandIndex];
    const theta = u * TWO_PI;
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    out[0] = r * (c * spec.tangentU[0] + s * spec.tangentV[0]);
    out[1] = r * (c * spec.tangentU[1] + s * spec.tangentV[1]);
    out[2] = r * (c * spec.tangentU[2] + s * spec.tangentV[2]);
  }

  public sampleBandCenterline(
    bandIndex: number,
    outPositions: Float32Array,
    outStrains: Float32Array
  ): number {
    if (bandIndex < 0 || bandIndex >= TOTAL_BANDS) return 0;
    const rt = this.runtimes[bandIndex];
    if (rt.status === 'gone') return 0;

    const spec = this.specs[bandIndex];
    const r = this.effectiveRadius[bandIndex];
    const tu = spec.tangentU;
    const tv = spec.tangentV;

    if (rt.status === 'snapping') {
      const totalArc = rt.leftArc + rt.rightArc;
      if (totalArc < 0.015) return 0;

      const activeSegs = Math.max(2, Math.min(SEGMENTS_PER_BAND, Math.round(totalArc * SEGMENTS_PER_BAND)));
      const antipodeU = (rt.snapU + 0.5) % 1;
      const curl = rt.curlAmount;
      const age = this.snapAgeSec[bandIndex];
      const fx = rt.fallOffset[0];
      const fy = rt.fallOffset[1];
      const fz = rt.fallOffset[2];

      // Rapidly whip back from pulled apex if snapped under tension
      const peakPull = this.snapPeakPull[bandIndex];
      const whipPeak = peakPull > 0 ? peakPull * Math.exp(-32.0 * age) : 0;
      const lx = this.snapLaunchDir[bandIndex * 3];
      const ly = this.snapLaunchDir[bandIndex * 3 + 1];
      const lz = this.snapLaunchDir[bandIndex * 3 + 2];

      // Minimum clearance sphere so snapping band peels off the ball instead of tunneling through it
      const minClearR = Math.max(
        CORE_RADIUS + CORE_CLEARANCE + spec.thickness * 0.5,
        r + spec.thickness * 0.5
      );
      const lashEnv = Math.exp(-4.5 * age);

      for (let k = 0; k <= activeSegs; k++) {
        const frac = k / activeSegs;
        const uOffset = -rt.leftArc + frac * totalArc;
        const theta = (antipodeU + uOffset) * TWO_PI;
        const c = Math.cos(theta);
        const s = Math.sin(theta);

        // Free ends whip radially outward off the sphere with an immediate elastic lash wave & C-curl
        const endDist = Math.abs(frac - 0.5) * 2; // 0 at center, 1 at tips
        const whipWave = Math.sin(endDist * 8.5 - age * 48.0) * 0.16 * endDist * lashEnv;
        const curlLift = curl * endDist * endDist * 0.62 + Math.max(0, whipWave);
        const rad = r + curlLift;

        let px = rad * (c * tu[0] + s * tv[0]) + fx;
        let py = rad * (c * tu[1] + s * tv[1]) + fy;
        let pz = rad * (c * tu[2] + s * tv[2]) + fz;

        if (whipPeak > 1e-4) {
          const tipLift = whipPeak * endDist * endDist;
          px += lx * tipLift;
          py += ly * tipLift;
          pz += lz * tipLift;
        }

        // Enforce non-penetration against the ball sphere
        const pDist = Math.hypot(px, py, pz);
        if (pDist < minClearR && pDist > 1e-5) {
          const scale = minClearR / pDist;
          px *= scale;
          py *= scale;
          pz *= scale;
        }

        const base = k * 3;
        outPositions[base] = px;
        outPositions[base + 1] = py;
        outPositions[base + 2] = pz;
        outStrains[k] = rt.currentStrain * (1 - endDist * 0.5);
      }

      return activeSegs;
    }

    // Intact or pulled
    const isPulled = rt.status === 'pulled' && rt.pullAmount > 0;
    const hasTwang = rt.twangAmp > 1e-4;
    const twangSin = hasTwang ? Math.sin(rt.twangPhase) * rt.twangAmp : 0;
    const nearBreak = isPulled
      ? Math.max(0, (rt.currentStrain - spec.breakStrain * 0.68) / (spec.breakStrain * 0.32 + 1e-4))
      : 0;
    const tremble = nearBreak > 0
      ? Math.sin(this.simTimeSec * 72.0 + bandIndex * 2.1) * 0.015 * nearBreak
      : 0;

    const profOffset = bandIndex * RINGS_PER_BAND;
    const breathShift = r - this.targetRadius[bandIndex];
    const minCoreR = CORE_RADIUS + CORE_CLEARANCE + spec.thickness * 0.5;
    const pullHalfArc = isPulled
      ? Math.min(0.22, Math.max(0.11, Math.acos(Math.min(0.98, r / (r + rt.pullAmount + 0.01))) / TWO_PI))
      : 0.22;

    for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
      const u = k / SEGMENTS_PER_BAND;
      const theta = u * TWO_PI;
      const c = Math.cos(theta);
      const s = Math.sin(theta);

      // Physical layered crossing-wrap radius (zero intersection with core or underlying bands)
      const rLayer = Math.max(minCoreR, this.radialProfile[profOffset + k] + breathShift);

      let px = rLayer * (c * tu[0] + s * tv[0]);
      let py = rLayer * (c * tu[1] + s * tv[1]);
      let pz = rLayer * (c * tu[2] + s * tv[2]);
      let strain = rt.currentStrain;

      if (isPulled) {
        const d = shortestArcDist(u, rt.pullU);
        const tent = Math.max(0, 1 - d / pullHalfArc);
        const vShape = Math.pow(tent, 1.08);
        const disp = Math.max(0, (rt.pullAmount + tremble) * vShape);
        px += rt.pullDir[0] * disp;
        py += rt.pullDir[1] * disp;
        pz += rt.pullDir[2] * disp;
        strain = spec.restStrain + (rt.currentStrain - spec.restStrain) * (0.35 + 0.65 * vShape);

        // Enforce radial wrapping constraint so pulled flanks never clip into the ball
        const pDist = Math.hypot(px, py, pz);
        if (pDist < rLayer && pDist > 1e-5) {
          const scale = rLayer / pDist;
          px *= scale;
          py *= scale;
          pz *= scale;
        }
      } else if (hasTwang) {
        const d = shortestArcDist(u, rt.pullU);
        const env = Math.exp(-d * d * 18.0);
        // Elastic rebound off the ball surface (cannot dip inside underlying layers)
        const disp = Math.max(0, twangSin * env);
        px += rt.pullDir[0] * disp;
        py += rt.pullDir[1] * disp;
        pz += rt.pullDir[2] * disp;

        const pDist = Math.hypot(px, py, pz);
        if (pDist < rLayer && pDist > 1e-5) {
          const scale = rLayer / pDist;
          px *= scale;
          py *= scale;
          pz *= scale;
        }
      }

      const base = k * 3;
      outPositions[base] = px;
      outPositions[base + 1] = py;
      outPositions[base + 2] = pz;
      outStrains[k] = strain;
    }

    return SEGMENTS_PER_BAND;
  }

  public raycastBand(
    rayOrigin: [number, number, number],
    rayDir: [number, number, number]
  ): { bandId: number; uHit: number; hitPoint: [number, number, number] } | null {
    const ox = rayOrigin[0];
    const oy = rayOrigin[1];
    const oz = rayOrigin[2];

    let dx = rayDir[0];
    let dy = rayDir[1];
    let dz = rayDir[2];
    const dLen = Math.hypot(dx, dy, dz);
    if (dLen < 1e-8) return null;
    dx /= dLen;
    dy /= dLen;
    dz /= dLen;

    const bHalf = ox * dx + oy * dy + oz * dz;
    const oDotO = ox * ox + oy * oy + oz * oz;

    // Quick bounding-sphere reject
    const maxR = OUTER_RADIUS + 0.12;
    if (bHalf * bHalf - (oDotO - maxR * maxR) < 0) {
      return null;
    }

    let bestT = Infinity;
    let bestBandId = -1;
    let bestU = 0;
    let bestPx = 0;
    let bestPy = 0;
    let bestPz = 0;

    const hitSlop = 0.008;

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      if (rt.status !== 'intact') continue;

      const r = this.effectiveRadius[i];
      const disc = bHalf * bHalf - (oDotO - r * r);
      if (disc < 0) continue;

      const t = -bHalf - Math.sqrt(disc);
      if (t <= 0 || t >= bestT) continue;

      const px = ox + t * dx;
      const py = oy + t * dy;
      const pz = oz + t * dz;

      const spec = this.specs[i];
      const latDist = Math.abs(
        px * spec.normal[0] + py * spec.normal[1] + pz * spec.normal[2]
      );
      if (latDist > spec.width * 0.5 + hitSlop) continue;

      const pu = px * spec.tangentU[0] + py * spec.tangentU[1] + pz * spec.tangentU[2];
      const pv = px * spec.tangentV[0] + py * spec.tangentV[1] + pz * spec.tangentV[2];
      const angle = Math.atan2(pv, pu);
      const uHit = (angle >= 0 ? angle : angle + TWO_PI) / TWO_PI;

      bestT = t;
      bestBandId = i;
      bestU = uHit;
      bestPx = px;
      bestPy = py;
      bestPz = pz;
    }

    if (bestBandId === -1) return null;

    return {
      bandId: bestBandId,
      uHit: bestU,
      hitPoint: [bestPx, bestPy, bestPz],
    };
  }

  public getBandSpecs(): readonly BandSpec[] {
    return this.specs;
  }

  public getBandRuntimes(): readonly BandRuntime[] {
    return this.runtimes;
  }

  public getDustPuffs(): readonly DustPuff[] {
    return this.dustPuffs;
  }

  public getBallRecoil(): readonly [number, number, number] {
    return this.recoilPos;
  }

  public getDiagnostics(): RubberBallDiagnostics {
    let intactCount = 0;
    let snappingCount = 0;
    let goneCount = 0;
    let strainSum = 0;
    let activeCount = 0;
    let maxRadius = CORE_RADIUS;

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      if (rt.status === 'intact' || rt.status === 'pulled') {
        intactCount++;
        strainSum += rt.currentStrain;
        activeCount++;
        const bandOuter = this.maxBandRadius[i] + (this.effectiveRadius[i] - this.targetRadius[i]);
        if (bandOuter > maxRadius) {
          maxRadius = bandOuter;
        }
      } else if (rt.status === 'snapping') {
        snappingCount++;
      } else {
        goneCount++;
      }
    }

    return {
      totalBands: TOTAL_BANDS,
      intactCount,
      snappingCount,
      goneCount,
      avgStrain: activeCount > 0 ? strainSum / activeCount : 0,
      ballRadius: maxRadius,
      activePullBandId: this.activePullBandId,
      snapCount: this.snapCount,
    };
  }
}
