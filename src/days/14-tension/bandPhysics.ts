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
const OUTER_RADIUS = 0.95;
const MAX_DUST_PUFFS = 384;
const MAX_CASCADE_QUEUE = 16;

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const TWO_PI = Math.PI * 2;
const STRAIN_PER_UNIT_PULL = 0.62;
const TWANG_FREQ = 54.0;
const TWANG_DAMP = 6.2;
const GRAVITY = 4.8;

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

      const baseRadius =
        CORE_RADIUS + 0.018 + t * (OUTER_RADIUS - CORE_RADIUS - 0.018) + (this.rng() - 0.5) * 0.008;
      const width = (0.034 + this.rng() * 0.036) * 1.22;
      const thickness = 0.009 + this.rng() * 0.009;

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
    }

    // Compute initial compressive preload exerted by outer layers on inner layers
    for (let i = 0; i < TOTAL_BANDS; i++) {
      this.initialOuterLoad[i] = this.computeOuterCompressionLoad(i);
    }
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
          this.enqueueCascade(i, 0.05 + this.rng() * 0.07, this.rng());
          cascadeScheduled++;
        }
      }
    }
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

    rt.status = 'snapping';
    rt.snapU = uNorm;
    rt.leftArc = 0.5;
    rt.rightArc = 0.5;
    rt.leftVel = (1.55 + strainFactor * 1.1) * (0.9 + this.rng() * 0.25);
    rt.rightVel = (1.55 + strainFactor * 1.1) * (0.9 + this.rng() * 0.25);
    rt.curlAmount = 0;
    rt.pullAmount = 0;
    rt.twangAmp = 0;
    rt.opacity = 1;

    // Outward kick from the sphere at the antipode direction
    this.evalCirclePoint(bandId, uNorm, this.scratchPoint);
    const pLen = Math.hypot(this.scratchPoint[0], this.scratchPoint[1], this.scratchPoint[2]) || 1;
    const radX = this.scratchPoint[0] / pLen;
    const radY = this.scratchPoint[1] / pLen;
    const radZ = this.scratchPoint[2] / pLen;

    const kick = fromPull ? 1.45 : 0.95;
    rt.fallOffset[0] = 0;
    rt.fallOffset[1] = 0;
    rt.fallOffset[2] = 0;
    rt.fallVel[0] = -radX * kick + (this.rng() - 0.5) * 0.55;
    rt.fallVel[1] = -radY * kick + 0.65;
    rt.fallVel[2] = -radZ * kick + (this.rng() - 0.5) * 0.55;

    // Crisp recoil jolt on the whole ball opposite the snap point
    const recoilImp = (fromPull ? 0.34 : 0.22) * (0.7 + strainFactor * 0.5);
    this.recoilVel[0] -= radX * recoilImp;
    this.recoilVel[1] -= radY * recoilImp;
    this.recoilVel[2] -= radZ * recoilImp;

    this.spawnDustPuffs(
      this.scratchPoint[0],
      this.scratchPoint[1],
      this.scratchPoint[2],
      radX,
      radY,
      radZ,
      spec.width
    );

    // Sympathetic elastic shockwave twang across nearby intact bands
    for (let i = 0; i < TOTAL_BANDS; i++) {
      if (i === bandId) continue;
      const nrt = this.runtimes[i];
      if (nrt.status !== 'intact') continue;
      const nspec = this.specs[i];
      if (Math.abs(nspec.layer - spec.layer) > 3) continue;
      const align = Math.abs(
        nspec.normal[0] * spec.normal[0] +
          nspec.normal[1] * spec.normal[1] +
          nspec.normal[2] * spec.normal[2]
      );
      if (align > 0.35) {
        nrt.twangAmp = Math.max(nrt.twangAmp, 0.032 + 0.028 * strainFactor * align);
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
    bandWidth: number
  ): void {
    let spawned = 0;
    const targetCount = 24;
    for (let i = 0; i < MAX_DUST_PUFFS && spawned < targetCount; i++) {
      const puff = this.dustPuffs[i];
      if (puff.active) continue;

      const isFineSpeck = spawned >= 16;
      puff.active = true;
      const spread = bandWidth * (isFineSpeck ? 0.9 : 1.4);
      puff.pos[0] = px + (this.rng() - 0.5) * spread;
      puff.pos[1] = py + (this.rng() - 0.5) * spread;
      puff.pos[2] = pz + (this.rng() - 0.5) * spread;

      const speed = isFineSpeck ? 0.9 + this.rng() * 1.1 : 0.35 + this.rng() * 0.85;
      const jitter = isFineSpeck ? 0.95 : 0.65;
      puff.vel[0] = nx * speed + (this.rng() - 0.5) * jitter;
      puff.vel[1] = ny * speed + (this.rng() - 0.5) * jitter + 0.12;
      puff.vel[2] = nz * speed + (this.rng() - 0.5) * jitter;

      puff.ageSec = 0;
      puff.maxAgeSec = isFineSpeck ? 0.28 + this.rng() * 0.25 : 0.55 + this.rng() * 0.45;
      puff.size = isFineSpeck ? 0.005 + this.rng() * 0.006 : 0.016 + this.rng() * 0.022;
      spawned++;
    }
  }

  public step(dtSec: number): void {
    const dt = Math.min(Math.max(dtSec, 0), 0.05);
    if (dt <= 0) return;
    this.simTimeSec += dt;

    // Damped spring recoil of the ball (leans toward active pull, snaps back on release/pop)
    const kSpring = 180.0;
    const kDamp = 15.0;
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
        rt.leftArc = Math.max(0, rt.leftArc - rt.leftVel * dt);
        rt.rightArc = Math.max(0, rt.rightArc - rt.rightVel * dt);
        rt.curlAmount = Math.min(1, rt.curlAmount + dt * 2.4);

        rt.fallVel[1] -= GRAVITY * dt;
        rt.fallOffset[0] += rt.fallVel[0] * dt;
        rt.fallOffset[1] += rt.fallVel[1] * dt;
        rt.fallOffset[2] += rt.fallVel[2] * dt;

        rt.currentStrain = Math.max(0, rt.currentStrain - dt * 2.5);

        if (rt.leftArc <= 0.08 && rt.rightArc <= 0.08) {
          rt.opacity = Math.max(0, rt.opacity - dt * 2.2);
        } else if (rt.fallOffset[1] < -1.2) {
          rt.opacity = Math.max(0, rt.opacity - dt * 1.5);
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
      const fx = rt.fallOffset[0];
      const fy = rt.fallOffset[1];
      const fz = rt.fallOffset[2];

      for (let k = 0; k <= activeSegs; k++) {
        const frac = k / activeSegs;
        const uOffset = -rt.leftArc + frac * totalArc;
        const theta = (antipodeU + uOffset) * TWO_PI;
        const c = Math.cos(theta);
        const s = Math.sin(theta);

        // Free ends whip radially outward off the sphere with a traveling elastic lash wave
        const endDist = Math.abs(frac - 0.5) * 2; // 0 at center, 1 at tips
        const whipWave = Math.sin(endDist * 7.0 - curl * 18.0) * 0.12 * endDist * (1 - curl * 0.45);
        const curlLift = curl * endDist * endDist * 0.55 + Math.max(0, whipWave);
        const rad = r + curlLift;

        const base = k * 3;
        outPositions[base] = rad * (c * tu[0] + s * tv[0]) + fx;
        outPositions[base + 1] = rad * (c * tu[1] + s * tv[1]) + fy;
        outPositions[base + 2] = rad * (c * tu[2] + s * tv[2]) + fz;
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

    for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
      const u = k / SEGMENTS_PER_BAND;
      const theta = u * TWO_PI;
      const c = Math.cos(theta);
      const s = Math.sin(theta);

      // Subtle crossing-bite indentation so bands visibly cinch into underlying layers
      const bite = 1.0 - 0.016 * (0.5 + 0.5 * Math.sin(theta * 5.0 + bandIndex * 1.7)) * Math.cos(theta * 3.0 - bandIndex * 0.9);
      const rBite = r * bite;

      let px = rBite * (c * tu[0] + s * tv[0]);
      let py = rBite * (c * tu[1] + s * tv[1]);
      let pz = rBite * (c * tu[2] + s * tv[2]);
      let strain = rt.currentStrain;

      if (isPulled) {
        const d = shortestArcDist(u, rt.pullU);
        const tent = Math.max(0, 1 - d / 0.34);
        const vShape = Math.pow(tent, 1.08);
        const disp = (rt.pullAmount + tremble) * vShape;
        px += rt.pullDir[0] * disp;
        py += rt.pullDir[1] * disp;
        pz += rt.pullDir[2] * disp;
        strain = spec.restStrain + (rt.currentStrain - spec.restStrain) * (0.35 + 0.65 * vShape);
      } else if (hasTwang) {
        const d = shortestArcDist(u, rt.pullU);
        const env = Math.exp(-d * d * 18.0);
        const disp = twangSin * env;
        px += rt.pullDir[0] * disp;
        py += rt.pullDir[1] * disp;
        pz += rt.pullDir[2] * disp;
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
        if (this.effectiveRadius[i] > maxRadius) {
          maxRadius = this.effectiveRadius[i];
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
