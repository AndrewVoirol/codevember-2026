import {
  BandRuntime,
  BandSpec,
  CORE_RADIUS,
  DustPuff,
  PigmentKey,
  RubberBallDiagnostics,
  SEGMENTS_PER_BAND,
  SnapImpulseInfo,
  TOTAL_BANDS,
} from './types';

const NUM_LAYERS = 14;
const CORE_CLEARANCE = 0.0075;
const CONTACT_GAP = 0.0008;
const SHOULDER_MARGIN = 0.0045;
const MAX_DUST_PUFFS = 384;
const MAX_CASCADE_QUEUE = 16;
const RINGS_PER_BAND = SEGMENTS_PER_BAND + 1;

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const TWO_PI = Math.PI * 2;
const DTHETA = TWO_PI / SEGMENTS_PER_BAND;
const COS_HALF_DTHETA = Math.cos(DTHETA * 0.5);
const HALF_DTHETA_SQ = 0.5 * DTHETA * DTHETA;

const STRAIN_PER_UNIT_PULL = 1.05;
const TWANG_FREQ = 68.0;
const TWANG_DAMP = 5.4;
const GRAVITY = 5.8;

const LAT_FRACS = [-1.0, 0.0, 1.0] as const;
const CORNER_R = [1.0, 1.0, 0.52, -0.52, -1.0, -1.0, -0.52, 0.52] as const;
const CORNER_B = [-0.78, 0.78, 1.0, 1.0, 0.78, -0.78, -1.0, -1.0] as const;

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

/**
 * C^1 elastic pull arch: smooth rounded crown at s=0 (d=0) and tangent touchdown at s=1 (d=halfArc).
 */
function evalPullArch(d: number, halfArc: number): number {
  if (d >= halfArc) return 0;
  const tent = 1 - d / halfArc;
  const s = Math.sin(0.5 * Math.PI * tent);
  return Math.pow(s * s, 0.85);
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
  private readonly targetBreath = new Float32Array(TOTAL_BANDS);
  private readonly currentBreath = new Float32Array(TOTAL_BANDS);
  private readonly restingMinRadius = new Float32Array(TOTAL_BANDS);
  private readonly unitCirclePts = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND * 3);
  private readonly unitHalfPts = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly radialProfile = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND);
  private readonly minBandRadius = new Float32Array(TOTAL_BANDS);
  private readonly maxBandRadius = new Float32Array(TOTAL_BANDS);
  private readonly floorScratch = new Float32Array(RINGS_PER_BAND);
  private readonly dynEnvScratch = new Float32Array(RINGS_PER_BAND);

  // Precomputed 3D segment OBBs for resting bands + zero-GC scratch for exact clearance
  private readonly restSegCenter = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly restSegT = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly restSegB = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly restSegR = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly restSegHalfL = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly restSegHalfW = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly restSegHalfH = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly dynSegCenter = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly dynSegT = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly dynSegB = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly dynSegR = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND * 3);
  private readonly dynSegHalfL = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly dynSegHalfW = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly dynSegHalfH = new Float32Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly scratchPtsI = new Float32Array(RINGS_PER_BAND * 3);
  private readonly scratchRingR = new Float32Array(RINGS_PER_BAND * 3);
  private readonly scratchRingB = new Float32Array(RINGS_PER_BAND * 3);
  private readonly candSegs = new Uint8Array(SEGMENTS_PER_BAND);
  private readonly candOverBands = new Uint8Array(TOTAL_BANDS);
  private readonly topBowIds = new Int32Array(6);
  private readonly topBowAmps = new Float32Array(6);
  private readonly topBowKs = new Int32Array(6);
  private readonly dynRingMoved = new Uint8Array(TOTAL_BANDS * RINGS_PER_BAND);
  private readonly dynDirtyBox = new Float32Array(TOTAL_BANDS * 6);
  private readonly restBandBox = new Float32Array(TOTAL_BANDS * 6);
  private readonly dynBandBox = new Float32Array(TOTAL_BANDS * 6);
  private readonly dirtyRingIdx = new Uint16Array(SEGMENTS_PER_BAND);
  private readonly dirtyRingBox = new Float32Array(6);
  private readonly iFullBox = new Float32Array(6);
  private dirtyRingCount = 0;
  private readonly dynMaxBandRadius = new Float32Array(TOTAL_BANDS);
  private readonly dynMaxPlaneOff = new Float32Array(TOTAL_BANDS);
  private readonly dynMovedSegIdx = new Uint8Array(TOTAL_BANDS * SEGMENTS_PER_BAND);
  private readonly dynMovedSegCount = new Uint8Array(TOTAL_BANDS);
  private readonly winActive = new Uint8Array(TOTAL_BANDS);
  private readonly winJMax0 = new Float32Array(TOTAL_BANDS);
  private readonly winJMax1 = new Float32Array(TOTAL_BANDS);
  private readonly winStart0Arr = new Uint8Array(TOTAL_BANDS);
  private readonly winStart1Arr = new Uint8Array(TOTAL_BANDS);
  private readonly winCountArr = new Uint8Array(TOTAL_BANDS);

  // Dynamic pull/twang coupling buffers (zero-GC)
  private readonly dynRadii = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND);
  private readonly dynPts = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND * 3);
  private readonly dynStrains = new Float32Array(TOTAL_BANDS * RINGS_PER_BAND);
  private readonly dynActive = new Uint8Array(TOTAL_BANDS);
  private readonly prevDynActive = new Uint8Array(TOTAL_BANDS);
  private readonly bandVersion = new Uint32Array(TOTAL_BANDS);

  // Internal snap-kinematics state
  private readonly snapAgeSec = new Float32Array(TOTAL_BANDS);
  private readonly snapPeakPull = new Float32Array(TOTAL_BANDS);
  private readonly snapHalfArc = new Float32Array(TOTAL_BANDS);
  private readonly snapAsym = new Float32Array(TOTAL_BANDS);
  private readonly snapSideSign = new Float32Array(TOTAL_BANDS);
  private readonly snapLaunchDir = new Float32Array(TOTAL_BANDS * 3);

  private activePullBandId: number | null = null;
  private snapCount = 0;
  private simTimeSec = 0;
  private ballMaxOuterRadius = CORE_RADIUS + 0.1;
  private rng = createRng(20261114);
  private readonly seed: number;

  // Damped spring recoil of the whole ball on pull & band pop
  private readonly recoilPos: [number, number, number] = [0, 0, 0];
  private readonly recoilVel: [number, number, number] = [0, 0, 0];

  // Last snap directional impulse for visual torque jolt & audio in main.ts
  private hasLastSnap = false;
  private readonly lastSnapInfo: SnapImpulseInfo = {
    snapId: 0,
    bandId: -1,
    hitPos: [0, 0, 0],
    pullDir: [0, 0, 1],
    normal: [0, 0, 1],
    intensity: 0,
  };

  // Pre-allocated scratch for zero-GC raycast & pull tracking
  private readonly scratchPoint: [number, number, number] = [0, 0, 0];
  private readonly pullBaseNormal: [number, number, number] = [0, 0, 1];
  private readonly pullDragVec: [number, number, number] = [0, 0, 0];

  constructor(seed = 20261114) {
    this.seed = seed;
    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      this.dustPuffs.push({
        active: false,
        isSpeck: true,
        pos: [0, 0, 0],
        vel: [0, 0, 0],
        ageSec: 0,
        maxAgeSec: 0.28,
        size: 0.006,
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
    this.simTimeSec = 0;
    this.hasLastSnap = false;
    this.recoilPos[0] = 0;
    this.recoilPos[1] = 0;
    this.recoilPos[2] = 0;
    this.recoilVel[0] = 0;
    this.recoilVel[1] = 0;
    this.recoilVel[2] = 0;

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

      // Isotropic golden-ratio stratified hemisphere normal (prevents multi-layer latitude pile-up)
      const layerPhase = (layer * 0.6180339887498949) % 1;
      const rawFrac = (indexInLayer + layerPhase + (this.rng() - 0.5) * 0.35) / bandsPerLayer;
      const wrappedFrac = ((rawFrac % 1) + 1) % 1;
      const y = Math.max(0.03, Math.min(0.992, 0.03 + wrappedFrac * 0.96));
      const rSin = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = i * GOLDEN_ANGLE + (this.rng() - 0.5) * 0.18;

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

      // Realistic thick rubber bands tightly wrapped onto the core/inner layers
      const thickness = 0.0132 + this.rng() * 0.004;
      const width = 0.038 + this.rng() * 0.022;
      const minCoreR = CORE_RADIUS + CORE_CLEARANCE + thickness * 0.5;
      // Tight unpopped hoop tension radius: grows gently with layer and pulls taut onto inner bands
      const baseRadius = minCoreR + t * 0.06;

      const pigment = this.pickStratifiedPigment(layer);

      // Tuned so ~0.35-0.50 world units of drag cleanly crosses breakStrain
      const restStrain = Math.max(0.12, Math.min(0.48, 0.18 + t * 0.25 + (this.rng() - 0.5) * 0.06));
      const breakStrain = restStrain + 0.36 + this.rng() * 0.15;

      this.specs.push({
        id: i,
        layer,
        baseRadius,
        width,
        thickness,
        pigment,
        restStrain,
        breakStrain,
        normal: [nx, ny, nz],
        tangentU: txU,
        tangentV: txV,
      });

      this.runtimes.push({
        id: i,
        status: 'intact',
        currentStrain: restStrain,
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

      this.targetBreath[i] = 0;
      this.currentBreath[i] = 0;
      this.restingMinRadius[i] = baseRadius;
      this.dynActive[i] = 0;
      this.prevDynActive[i] = 0;
      this.bandVersion[i]++;
      this.snapAgeSec[i] = 0;
      this.snapPeakPull[i] = 0;
      this.snapHalfArc[i] = 0.18;
      this.snapAsym[i] = 1;
      this.snapSideSign[i] = 1;

      const ringOffset = i * RINGS_PER_BAND * 3;
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const thetaK = k === SEGMENTS_PER_BAND ? 0 : (k / SEGMENTS_PER_BAND) * TWO_PI;
        const c = Math.cos(thetaK);
        const s = Math.sin(thetaK);
        const idx = ringOffset + k * 3;
        this.unitCirclePts[idx] = c * txU[0] + s * txV[0];
        this.unitCirclePts[idx + 1] = c * txU[1] + s * txV[1];
        this.unitCirclePts[idx + 2] = c * txU[2] + s * txV[2];
      }

      const halfOffset = i * SEGMENTS_PER_BAND * 3;
      for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
        const thetaMid = ((k + 0.5) / SEGMENTS_PER_BAND) * TWO_PI;
        const c = Math.cos(thetaMid);
        const s = Math.sin(thetaMid);
        const idx = halfOffset + k * 3;
        this.unitHalfPts[idx] = c * txU[0] + s * txV[0];
        this.unitHalfPts[idx + 1] = c * txU[1] + s * txV[1];
        this.unitHalfPts[idx + 2] = c * txU[2] + s * txV[2];
      }
    }

    // Compute initial compressive preload exerted by outer layers on inner layers
    for (let i = 0; i < TOTAL_BANDS; i++) {
      this.initialOuterLoad[i] = this.computeOuterCompressionLoad(i);
    }

    // Build tightly packed, zero-intersection wrapped profiles
    this.rebuildRadialProfiles(true);
  }

  private pickStratifiedPigment(layer: number): PigmentKey {
    const r = this.rng();
    if (layer <= 2) {
      if (r < 0.4) return 'rawUmber';
      if (r < 0.7) return 'boneCalcite';
      if (r < 0.9) return 'yellowOchre';
      return 'burntSienna';
    }
    if (layer <= 7) {
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

  /**
   * Computes minimum non-intersecting radius for band i at sample unit vector (ux, uy, uz)
   * across full ribbon width ±halfWI against underlying band j.
   */
  private evalRibbonPointClearance(
    ux: number,
    uy: number,
    uz: number,
    ni: readonly [number, number, number],
    halfWI: number,
    halfHI: number,
    rFloorMin: number,
    j: number
  ): number {
    const specJ = this.specs[j];
    const nj = specJ.normal;
    const uj = specJ.tangentU;
    const vj = specJ.tangentV;
    const halfWJ = specJ.width * 0.5;
    const halfHJ = specJ.thickness * 0.5;
    const rMaxNeeded = this.maxBandRadius[j] + halfHJ + halfHI + CONTACT_GAP + 0.01;
    if (rFloorMin >= rMaxNeeded) return 0;

    const profOffsetJ = j * RINGS_PER_BAND;
    const shoulderPad = Math.max(SHOULDER_MARGIN, halfHI * 0.95 + 0.003);

    // Conservative angular width cull across band i's full ribbon width + slope tilt guard
    const latCenter = Math.abs(ux * nj[0] + uy * nj[1] + uz * nj[2]);
    const scaleRatio = this.maxBandRadius[j] / Math.max(CORE_RADIUS, rFloorMin);
    if (latCenter * this.minBandRadius[j] > halfWJ + halfWI * scaleRatio + shoulderPad) {
      return 0;
    }

    const rEstInit = Math.max(rFloorMin, 0.5 * (this.minBandRadius[j] + this.maxBandRadius[j]));
    let bestClear = 0;

    for (let li = 0; li < LAT_FRACS.length; li++) {
      const latOff = LAT_FRACS[li] * halfWI;
      let rEst = rEstInit;
      let rJ = 0;
      let pn = 0;
      for (let refine = 0; refine < 1; refine++) {
        const qx = rEst * ux + latOff * ni[0];
        const qy = rEst * uy + latOff * ni[1];
        const qz = rEst * uz + latOff * ni[2];
        const qLen = Math.hypot(qx, qy, qz) || 1;
        const uqx = qx / qLen;
        const uqy = qy / qLen;
        const uqz = qz / qLen;

        pn = Math.abs(uqx * nj[0] + uqy * nj[1] + uqz * nj[2]);
        const pu = uqx * uj[0] + uqy * uj[1] + uqz * uj[2];
        const pv = uqx * vj[0] + uqy * vj[1] + uqz * vj[2];

        let angleJ = Math.atan2(pv, pu);
        if (angleJ < 0) angleJ += TWO_PI;
        const kJFloat = (angleJ / TWO_PI) * SEGMENTS_PER_BAND;
        const kJ0 = Math.floor(kJFloat) % SEGMENTS_PER_BAND;
        const kJ1 = (kJ0 + 1) % SEGMENTS_PER_BAND;
        rJ = Math.max(
          this.radialProfile[profOffsetJ + kJ0],
          this.radialProfile[profOffsetJ + kJ1]
        );
        if (Math.abs(rEst - (rJ + halfHJ + halfHI)) <= 0.01) break;
        rEst = rJ + halfHJ + halfHI;
      }

      const latDistJ = pn * rJ;
      if (latDistJ > halfWJ + shoulderPad) continue;

      const clampLatJ = Math.min(latDistJ, halfWJ);
      const rJOuter = Math.hypot(rJ + halfHJ, clampLatJ);
      const targetNorm = rJOuter + CONTACT_GAP;
      const reqInner =
        targetNorm > Math.abs(latOff)
          ? Math.sqrt(targetNorm * targetNorm - latOff * latOff)
          : 0;
      const minClearCenter = reqInner + halfHI;
      if (minClearCenter > bestClear) {
        bestClear = minClearCenter;
      }
    }

    return bestClear;
  }

  /**
   * Builds the per-ring scaled cross-section axes (R_r * ht, B_r * hw) matching rubberOptics.ts.
   */
  private buildBandVerts(
    bandIdx: number,
    pts: Float32Array,
    ptsOffset: number,
    strains: Float32Array | null,
    strainOffset: number
  ): void {
    const spec = this.specs[bandIdx];
    const halfWBase = spec.width * 0.5;
    const halfTBase = spec.thickness * 0.5;
    let prevBx = spec.normal[0];
    let prevBy = spec.normal[1];
    let prevBz = spec.normal[2];

    for (let r = 0; r < SEGMENTS_PER_BAND; r++) {
      const rIdx = ptsOffset + r * 3;
      const px = pts[rIdx];
      const py = pts[rIdx + 1];
      const pz = pts[rIdx + 2];

      const prevIdx = ptsOffset + (r > 0 ? r - 1 : SEGMENTS_PER_BAND - 1) * 3;
      const nextIdx = ptsOffset + (r < SEGMENTS_PER_BAND ? r + 1 : 1) * 3;
      let tx = pts[nextIdx] - pts[prevIdx];
      let ty = pts[nextIdx + 1] - pts[prevIdx + 1];
      let tz = pts[nextIdx + 2] - pts[prevIdx + 2];
      const tLen = Math.hypot(tx, ty, tz) || 1;
      tx /= tLen;
      ty /= tLen;
      tz /= tLen;

      // Anchor binormal B to wrap-plane normal (parallel-transport fallback -> P x T) matching rubberOptics
      const nDotT = spec.normal[0] * tx + spec.normal[1] * ty + spec.normal[2] * tz;
      let bx = spec.normal[0] - nDotT * tx;
      let by = spec.normal[1] - nDotT * ty;
      let bz = spec.normal[2] - nDotT * tz;
      let bLen = Math.hypot(bx, by, bz);

      if (bLen < 0.15) {
        const pDotT = prevBx * tx + prevBy * ty + prevBz * tz;
        bx = prevBx - pDotT * tx;
        by = prevBy - pDotT * ty;
        bz = prevBz - pDotT * tz;
        bLen = Math.hypot(bx, by, bz);
      }
      if (bLen < 1e-4) {
        bx = py * tz - pz * ty;
        by = pz * tx - px * tz;
        bz = px * ty - py * tx;
        bLen = Math.hypot(bx, by, bz) || 1;
      }
      bx /= bLen;
      by /= bLen;
      bz /= bLen;
      prevBx = bx;
      prevBy = by;
      prevBz = bz;

      let rx = ty * bz - tz * by;
      let ry = tz * bx - tx * bz;
      let rz = tx * by - ty * bx;
      const rOrtho = Math.hypot(rx, ry, rz) || 1;
      rx /= rOrtho;
      ry /= rOrtho;
      rz /= rOrtho;

      const strain = strains !== null ? strains[strainOffset + r] : spec.restStrain;
      const extraStrain = strain > spec.restStrain ? strain - spec.restStrain : 0;
      const hw = halfWBase * Math.max(0.32, 1 / Math.sqrt(1 + extraStrain * 2.1));
      const ht = halfTBase * Math.max(0.45, 1 / Math.sqrt(1 + extraStrain * 1.2));

      const o3 = r * 3;
      this.scratchRingR[o3] = rx * ht;
      this.scratchRingR[o3 + 1] = ry * ht;
      this.scratchRingR[o3 + 2] = rz * ht;
      this.scratchRingB[o3] = bx * hw;
      this.scratchRingB[o3 + 1] = by * hw;
      this.scratchRingB[o3 + 2] = bz * hw;
    }

    // Closed seam ring SEGMENTS_PER_BAND mirrors ring 0
    const seamOff = SEGMENTS_PER_BAND * 3;
    this.scratchRingR[seamOff] = this.scratchRingR[0];
    this.scratchRingR[seamOff + 1] = this.scratchRingR[1];
    this.scratchRingR[seamOff + 2] = this.scratchRingR[2];
    this.scratchRingB[seamOff] = this.scratchRingB[0];
    this.scratchRingB[seamOff + 1] = this.scratchRingB[1];
    this.scratchRingB[seamOff + 2] = this.scratchRingB[2];
  }

  /**
   * Builds per-segment 3D OBB frames for band `bandIdx` into `outOffset` (= bandIdx * SEGMENTS_PER_BAND).
   */
  private buildBandOBB(
    bandIdx: number,
    pts: Float32Array,
    ptsOffset: number,
    strains: Float32Array | null,
    strainOffset: number,
    segCenter: Float32Array,
    segT: Float32Array,
    segB: Float32Array,
    segR: Float32Array,
    segHalfL: Float32Array,
    segHalfW: Float32Array,
    segHalfH: Float32Array,
    outOffset: number
  ): void {
    const spec = this.specs[bandIdx];
    const halfWBase = spec.width * 0.5;
    const halfTBase = spec.thickness * 0.5;
    let prevBx = spec.normal[0];
    let prevBy = spec.normal[1];
    let prevBz = spec.normal[2];

    for (let s = 0; s < SEGMENTS_PER_BAND; s++) {
      const a3 = ptsOffset + s * 3;
      const b3 = ptsOffset + (s + 1) * 3;
      const mx = 0.5 * (pts[a3] + pts[b3]);
      const my = 0.5 * (pts[a3 + 1] + pts[b3 + 1]);
      const mz = 0.5 * (pts[a3 + 2] + pts[b3 + 2]);

      const d3 = (outOffset + s) * 3;
      segCenter[d3] = mx;
      segCenter[d3 + 1] = my;
      segCenter[d3 + 2] = mz;

      let tx = pts[b3] - pts[a3];
      let ty = pts[b3 + 1] - pts[a3 + 1];
      let tz = pts[b3 + 2] - pts[a3 + 2];
      const segLen = Math.hypot(tx, ty, tz) || 1;
      tx /= segLen;
      ty /= segLen;
      tz /= segLen;
      segT[d3] = tx;
      segT[d3 + 1] = ty;
      segT[d3 + 2] = tz;

      const nDotT = spec.normal[0] * tx + spec.normal[1] * ty + spec.normal[2] * tz;
      let bx = spec.normal[0] - nDotT * tx;
      let by = spec.normal[1] - nDotT * ty;
      let bz = spec.normal[2] - nDotT * tz;
      let bLen = Math.hypot(bx, by, bz);

      if (bLen < 0.15) {
        const pDotT = prevBx * tx + prevBy * ty + prevBz * tz;
        bx = prevBx - pDotT * tx;
        by = prevBy - pDotT * ty;
        bz = prevBz - pDotT * tz;
        bLen = Math.hypot(bx, by, bz);
      }
      if (bLen < 1e-4) {
        bx = my * tz - mz * ty;
        by = mz * tx - mx * tz;
        bz = mx * ty - my * tx;
        bLen = Math.hypot(bx, by, bz) || 1;
      }
      bx /= bLen;
      by /= bLen;
      bz /= bLen;
      prevBx = bx;
      prevBy = by;
      prevBz = bz;
      segB[d3] = bx;
      segB[d3 + 1] = by;
      segB[d3 + 2] = bz;

      let rx = ty * bz - tz * by;
      let ry = tz * bx - tx * bz;
      let rz = tx * by - ty * bx;
      const rLen = Math.hypot(rx, ry, rz) || 1;
      rx /= rLen;
      ry /= rLen;
      rz /= rLen;
      segR[d3] = rx;
      segR[d3 + 1] = ry;
      segR[d3 + 2] = rz;

      const avgStrain =
        strains !== null
          ? 0.5 * (strains[strainOffset + s] + strains[strainOffset + s + 1])
          : spec.restStrain;
      const extraStrain = avgStrain > spec.restStrain ? avgStrain - spec.restStrain : 0;
      const dst = outOffset + s;
      segHalfL[dst] = segLen * 0.5;
      segHalfW[dst] = halfWBase * Math.max(0.32, 1 / Math.sqrt(1 + extraStrain * 2.1));
      segHalfH[dst] = halfTBase * Math.max(0.45, 1 / Math.sqrt(1 + extraStrain * 1.2));
    }
  }

  /**
   * Accumulates exact radial lifts into `floorScratch` (initialized to current radii of `ptsI`)
   * wherever any vertex or chord midpoint of `bandI` (`vertsI`) intersects `bandJ`'s OBB.
   */
  private accumulateOBBLiftFromBandJ(
    ptsI: Float32Array,
    ptsOffsetI: number,
    specI: BandSpec,
    bandJ: number,
    segCenter: Float32Array,
    segT: Float32Array,
    segB: Float32Array,
    segR: Float32Array,
    segHalfL: Float32Array,
    segHalfW: Float32Array,
    segHalfH: Float32Array,
    useAzimuthWindowJ: boolean,
    gap: number,
    ringDirtyI: Uint8Array | null,
    ringDirtyOffI: number,
    dirtyBoxJ: Float32Array | null,
    boxOffJ: number,
    minRHint = CORE_RADIUS,
    isPlanarI = true
  ): boolean {
    const specJ = this.specs[bandJ];
    const nj = specJ.normal;
    const uj = specJ.tangentU;
    const vj = specJ.tangentV;
    const halfWJ = specJ.width * 0.5;
    const halfWI = specI.width * 0.5;
    const jOff = bandJ * SEGMENTS_PER_BAND;
    const slabThresh = halfWJ + halfWI + 0.045;
    let anyLift = false;

    const hasBoxJ = dirtyBoxJ !== null;
    const slabThreshEff = slabThresh + (hasBoxJ ? this.dynMaxPlaneOff[bandJ] : 0);
    const baseMaxRJ = hasBoxJ ? this.dynMaxBandRadius[bandJ] : this.maxBandRadius[bandJ];
    const rMaxCheck =
      baseMaxRJ +
      (ringDirtyI === null || isPlanarI
        ? (specJ.thickness + specI.thickness) * 0.5 + 0.018
        : specJ.thickness + specI.thickness + 0.08);
    const rMaxCheckSq = rMaxCheck * rMaxCheck;

    let winStart0 = 0;
    let winCount = SEGMENTS_PER_BAND;
    let twoWindows = false;
    let winStart1 = 0;

    if (isPlanarI && (useAzimuthWindowJ || hasBoxJ)) {
      const tui = specI.tangentU;
      const tvi = specI.tangentV;
      const a = tui[0] * nj[0] + tui[1] * nj[1] + tui[2] * nj[2];
      const b = tvi[0] * nj[0] + tvi[1] * nj[1] + tvi[2] * nj[2];
      const amp = Math.hypot(a, b);
      const tau = (slabThreshEff + 0.01) / (Math.max(CORE_RADIUS, minRHint) * Math.max(1e-4, amp));
      if (amp > 1e-4 && tau < 1.0) {
        const halfWinRad = Math.asin(tau) + DTHETA * 1.5;
        const halfWinSegs = Math.min(SEGMENTS_PER_BAND >> 1, Math.ceil(halfWinRad / DTHETA));
        let theta0 = Math.atan2(-a, b);
        if (theta0 < 0) theta0 += TWO_PI;
        const kCross0 = Math.round((theta0 / TWO_PI) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
        const kCross1 = (kCross0 + (SEGMENTS_PER_BAND >> 1)) % SEGMENTS_PER_BAND;
        const count = halfWinSegs * 2 + 1;
        if (count * 2 < SEGMENTS_PER_BAND) {
          twoWindows = true;
          winCount = count;
          winStart0 = (kCross0 - halfWinSegs + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          winStart1 = (kCross1 - halfWinSegs + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
        }
      }
    }

    const jMinX = hasBoxJ ? dirtyBoxJ[boxOffJ] : -1e9;
    const jMaxX = hasBoxJ ? dirtyBoxJ[boxOffJ + 1] : 1e9;
    const jMinY = hasBoxJ ? dirtyBoxJ[boxOffJ + 2] : -1e9;
    const jMaxY = hasBoxJ ? dirtyBoxJ[boxOffJ + 3] : 1e9;
    const jMinZ = hasBoxJ ? dirtyBoxJ[boxOffJ + 4] : -1e9;
    const jMaxZ = hasBoxJ ? dirtyBoxJ[boxOffJ + 5] : 1e9;

    const useDirtyList = ringDirtyI !== null && !hasBoxJ;
    let useBoxArc = false;
    if (!useDirtyList && !twoWindows && hasBoxJ && isPlanarI && ringDirtyI !== null) {
      const mcx = 0.5 * (jMinX + jMaxX);
      const mcy = 0.5 * (jMinY + jMaxY);
      const mcz = 0.5 * (jMinZ + jMaxZ);
      const rBox = 0.5 * Math.hypot(jMaxX - jMinX, jMaxY - jMinY, jMaxZ - jMinZ);
      const tui = specI.tangentU;
      const tvi = specI.tangentV;
      const pu = mcx * tui[0] + mcy * tui[1] + mcz * tui[2];
      const pv = mcx * tvi[0] + mcy * tvi[1] + mcz * tvi[2];
      const pLen = Math.hypot(pu, pv);
      if (pLen > rBox + 1e-4) {
        const halfBoxSegs = Math.min(
          SEGMENTS_PER_BAND >> 1,
          Math.ceil((Math.asin(rBox / pLen) + DTHETA * 1.5) / DTHETA)
        );
        const boxCount = halfBoxSegs * 2 + 1;
        if (boxCount + this.dirtyRingCount < SEGMENTS_PER_BAND) {
          let thetaBox = Math.atan2(pv, pu);
          if (thetaBox < 0) thetaBox += TWO_PI;
          const kBoxCenter = Math.round((thetaBox / TWO_PI) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          useBoxArc = true;
          winStart0 = (kBoxCenter - halfBoxSegs + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          winCount = boxCount;
        }
      }
    }

    const totalSteps = useDirtyList
      ? this.dirtyRingCount
      : twoWindows
        ? winCount * 2
        : useBoxArc
          ? winCount + this.dirtyRingCount
          : winCount;
    for (let stepIdx = 0; stepIdx < totalSteps; stepIdx++) {
      let r = 0;
      if (useDirtyList) {
        r = this.dirtyRingIdx[stepIdx];
      } else if (useBoxArc) {
        if (stepIdx < winCount) {
          r = (winStart0 + stepIdx) % SEGMENTS_PER_BAND;
        } else {
          r = this.dirtyRingIdx[stepIdx - winCount];
          if (((r - winStart0 + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND) < winCount) continue;
        }
      } else {
        r =
          stepIdx < winCount
            ? (winStart0 + stepIdx) % SEGMENTS_PER_BAND
            : (winStart1 + (stepIdx - winCount)) % SEGMENTS_PER_BAND;
      }
      const r3 = ptsOffsetI + r * 3;
      const cx = ptsI[r3];
      const cy = ptsI[r3 + 1];
      const cz = ptsI[r3 + 2];

      let nearBoxJ = false;
      if (ringDirtyI !== null) {
        const selfDirty = ringDirtyI[ringDirtyOffI + r] !== 0;
        nearBoxJ =
          hasBoxJ &&
          cx >= jMinX &&
          cx <= jMaxX &&
          cy >= jMinY &&
          cy <= jMaxY &&
          cz >= jMinZ &&
          cz <= jMaxZ;
        if (!selfDirty && !nearBoxJ) continue;
      }

      const inPlaneWindow = useAzimuthWindowJ || !nearBoxJ;
      const cLenSq = cx * cx + cy * cy + cz * cz;
      if (cLenSq > rMaxCheckSq) continue;

      const slabDist = Math.abs(cx * nj[0] + cy * nj[1] + cz * nj[2]);
      if (slabDist > slabThreshEff) continue;
      if (inPlaneWindow && slabDist > slabThresh) continue;

      const cLen = Math.sqrt(cLenSq) || 1;
      const ux = cx / cLen;
      const uy = cy / cLen;
      const uz = cz / cLen;

      // Pre-filter candidate segments ONCE per ring r (instead of 16x inside mode/vertex loops)
      let candCount = 0;
      const distThreshSq = ringDirtyI === null ? 0.011 : 0.024;
      if (inPlaneWindow) {
        let ang = Math.atan2(
          ux * vj[0] + uy * vj[1] + uz * vj[2],
          ux * uj[0] + uy * uj[1] + uz * uj[2]
        );
        if (ang < 0) ang += TWO_PI;
        const kJCenter = Math.floor((ang / TWO_PI) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
        const dsHalf = cLen > 0.65 ? 4 : 6;
        for (let ds = -dsHalf; ds <= dsHalf; ds++) {
          const s = (kJCenter + ds + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          const s3 = (jOff + s) * 3;
          const dcx = cx - segCenter[s3];
          const dcy = cy - segCenter[s3 + 1];
          const dcz = cz - segCenter[s3 + 2];
          if (dcx * dcx + dcy * dcy + dcz * dcz <= distThreshSq) {
            this.candSegs[candCount++] = s;
          }
        }
      } else {
        const mCount = this.dynMovedSegCount[bandJ];
        for (let mi = 0; mi < mCount; mi++) {
          const s = this.dynMovedSegIdx[jOff + mi];
          const s3 = (jOff + s) * 3;
          const dcx = cx - segCenter[s3];
          const dcy = cy - segCenter[s3 + 1];
          const dcz = cz - segCenter[s3 + 2];
          if (dcx * dcx + dcy * dcy + dcz * dcz <= distThreshSq) {
            this.candSegs[candCount++] = s;
          }
        }
        if (slabDist <= slabThresh) {
          let ang = Math.atan2(
            ux * vj[0] + uy * vj[1] + uz * vj[2],
            ux * uj[0] + uy * uj[1] + uz * uj[2]
          );
          if (ang < 0) ang += TWO_PI;
          const kJCenter = Math.floor((ang / TWO_PI) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          const dsHalf = cLen > 0.65 ? 4 : 6;
          const jProfOff = bandJ * RINGS_PER_BAND;
          for (let ds = -dsHalf; ds <= dsHalf; ds++) {
            const s = (kJCenter + ds + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
            const sNext = s + 1 === SEGMENTS_PER_BAND ? 0 : s + 1;
            if (
              this.dynRingMoved[jProfOff + s] !== 0 ||
              this.dynRingMoved[jProfOff + sNext] !== 0
            ) {
              continue;
            }
            const s3 = (jOff + s) * 3;
            const dcx = cx - segCenter[s3];
            const dcy = cy - segCenter[s3 + 1];
            const dcz = cz - segCenter[s3 + 2];
            if (dcx * dcx + dcy * dcy + dcz * dcz <= distThreshSq) {
              this.candSegs[candCount++] = s;
            }
          }
        }
      }

      if (candCount === 0) continue;

      const rNext = r + 1;
      const ring3 = r * 3;
      const rn3 = rNext * 3;
      const pNext3 = ptsOffsetI + rn3;
      const rRx = this.scratchRingR[ring3];
      const rRy = this.scratchRingR[ring3 + 1];
      const rRz = this.scratchRingR[ring3 + 2];
      const bRx = this.scratchRingB[ring3];
      const bRy = this.scratchRingB[ring3 + 1];
      const bRz = this.scratchRingB[ring3 + 2];
      const nrRx = this.scratchRingR[rn3];
      const nrRy = this.scratchRingR[rn3 + 1];
      const nrRz = this.scratchRingR[rn3 + 2];
      const nbRx = this.scratchRingB[rn3];
      const nbRy = this.scratchRingB[rn3 + 1];
      const nbRz = this.scratchRingB[rn3 + 2];
      const nx = ptsI[pNext3];
      const ny = ptsI[pNext3 + 1];
      const nz = ptsI[pNext3 + 2];
      const mcx = 0.5 * (cx + nx);
      const mcy = 0.5 * (cy + ny);
      const mcz = 0.5 * (cz + nz);
      const mRx = 0.5 * (rRx + nrRx);
      const mRy = 0.5 * (rRy + nrRy);
      const mRz = 0.5 * (rRz + nrRz);
      const mBx = 0.5 * (bRx + nbRx);
      const mBy = 0.5 * (bRy + nbRy);
      const mBz = 0.5 * (bRz + nbRz);

      const padVert = ringDirtyI === null ? 0.065 : 0.105;

      for (let ci = 0; ci < candCount; ci++) {
        const s = this.candSegs[ci];
        const sOff = jOff + s;
        const s3 = sOff * 3;
        const mx = segCenter[s3];
        const my = segCenter[s3 + 1];
        const mz = segCenter[s3 + 2];
        const halfHJ = segHalfH[sOff];
        const stX = segT[s3];
        const stY = segT[s3 + 1];
        const stZ = segT[s3 + 2];
        const sbX = segB[s3];
        const sbY = segB[s3 + 1];
        const sbZ = segB[s3 + 2];
        const srX = segR[s3];
        const srY = segR[s3 + 1];
        const srZ = segR[s3 + 2];
        const halfL = segHalfL[sOff];
        const halfW = segHalfW[sOff];

        const cdx = cx - mx;
        const cdy = cy - my;
        const cdz = cz - mz;
        const cLocR = cdx * srX + cdy * srY + cdz * srZ;
        if (cLocR > halfHJ + gap + padVert || cLocR < -halfHJ - 0.03 - padVert) continue;
        const cLocT = cdx * stX + cdy * stY + cdz * stZ;
        if (Math.abs(cLocT) > halfL + padVert) continue;
        const cLocB = cdx * sbX + cdy * sbY + cdz * sbZ;
        if (Math.abs(cLocB) > halfW + padVert) continue;

        const mRad = Math.hypot(mx, my, mz);
        if (
          (ringDirtyI === null || isPlanarI) &&
          cLen - specI.thickness * 0.5 - 0.005 > mRad + halfHJ + 0.012
        ) {
          continue;
        }

        const rrR = rRx * srX + rRy * srY + rRz * srZ;
        const brR = bRx * srX + bRy * srY + bRz * srZ;
        const maxOffR0 = Math.abs(rrR) + Math.abs(brR);
        const canHit0 = cLocR - maxOffR0 < halfHJ + gap && cLocR + maxOffR0 > -halfHJ - 0.03;

        const mdx = mcx - mx;
        const mdy = mcy - my;
        const mdz = mcz - mz;
        const mLocR = mdx * srX + mdy * srY + mdz * srZ;
        const mRrR = mRx * srX + mRy * srY + mRz * srZ;
        const mBrR = mBx * srX + mBy * srY + mBz * srZ;
        const maxOffR1 = Math.abs(mRrR) + Math.abs(mBrR);
        const canHit1 = mLocR - maxOffR1 < halfHJ + gap && mLocR + maxOffR1 > -halfHJ - 0.03;

        if (!canHit0 && !canHit1) continue;

        if (canHit0) {
          const rrT = rRx * stX + rRy * stY + rRz * stZ;
          const brT = bRx * stX + bRy * stY + bRz * stZ;
          if (Math.abs(cLocT) - (Math.abs(rrT) + Math.abs(brT)) < halfL) {
            const rrB = rRx * sbX + rRy * sbY + rRz * sbZ;
            const brB = bRx * sbX + bRy * sbY + bRz * sbZ;
            if (Math.abs(cLocB) - (Math.abs(rrB) + Math.abs(brB)) < halfW) {
              for (let v = 0; v < 8; v++) {
                const cr = CORNER_R[v];
                const cb = CORNER_B[v];
                const locR = cLocR + cr * rrR + cb * brR;
                if (locR >= halfHJ + gap || locR <= -halfHJ - 0.03) continue;
                const locT = cLocT + cr * rrT + cb * brT;
                if (Math.abs(locT) >= halfL) continue;
                const locB = cLocB + cr * rrB + cb * brB;
                if (Math.abs(locB) >= halfW) continue;

                const px = cx + cr * rRx + cb * bRx;
                const py = cy + cr * rRy + cb * bRy;
                const pz = cz + cr * rRz + cb * bRz;
                const dx = px - mx;
                const dy = py - my;
                const dz = pz - mz;
                if (dx * dx + dy * dy + dz * dz > 0.013) continue;

                const pRad = Math.hypot(px, py, pz);
                if (pRad > mRad + halfHJ + 0.012) continue;

                const needR = halfHJ + gap - locR;
                const uDotR = Math.max(0.4, ux * srX + uy * srY + uz * srZ);
                const dR = needR / uDotR + 0.00012;
                const targetR = cLen + dR;
                if (targetR > this.floorScratch[r]) {
                  this.floorScratch[r] = targetR;
                  anyLift = true;
                }
              }
            }
          }
        }

        if (canHit1) {
          const mLocT = mdx * stX + mdy * stY + mdz * stZ;
          const mRrT = mRx * stX + mRy * stY + mRz * stZ;
          const mBrT = mBx * stX + mBy * stY + mBz * stZ;
          if (Math.abs(mLocT) - (Math.abs(mRrT) + Math.abs(mBrT)) < halfL) {
            const mLocB = mdx * sbX + mdy * sbY + mdz * sbZ;
            const mRrB = mRx * sbX + mRy * sbY + mRz * sbZ;
            const mBrB = mBx * sbX + mBy * sbY + mBz * sbZ;
            if (Math.abs(mLocB) - (Math.abs(mRrB) + Math.abs(mBrB)) < halfW) {
              for (let v = 0; v < 8; v++) {
                const cr = CORNER_R[v];
                const cb = CORNER_B[v];
                const locR = mLocR + cr * mRrR + cb * mBrR;
                if (locR >= halfHJ + gap || locR <= -halfHJ - 0.03) continue;
                const locT = mLocT + cr * mRrT + cb * mBrT;
                if (Math.abs(locT) >= halfL) continue;
                const locB = mLocB + cr * mRrB + cb * mBrB;
                if (Math.abs(locB) >= halfW) continue;

                const px = mcx + cr * mRx + cb * mBx;
                const py = mcy + cr * mRy + cb * mBy;
                const pz = mcz + cr * mRz + cb * mBz;
                const dx = px - mx;
                const dy = py - my;
                const dz = pz - mz;
                if (dx * dx + dy * dy + dz * dz > 0.013) continue;

                const pRad = Math.hypot(px, py, pz);
                if (pRad > mRad + halfHJ + 0.012) continue;

                const needR = halfHJ + gap - locR;
                const uDotR = Math.max(0.4, ux * srX + uy * srY + uz * srZ);
                const dR = needR / (uDotR * COS_HALF_DTHETA) + 0.00012;
                const nLen = Math.hypot(nx, ny, nz) || cLen;
                const tR0 = cLen + dR;
                const tR1 = nLen + dR;
                if (tR0 > this.floorScratch[r]) {
                  this.floorScratch[r] = tR0;
                  anyLift = true;
                }
                const nextRing = rNext === SEGMENTS_PER_BAND ? 0 : rNext;
                if (tR1 > this.floorScratch[nextRing]) {
                  this.floorScratch[nextRing] = tR1;
                  anyLift = true;
                }
              }
            }
          }
        }
      }
    }

    if (anyLift) {
      this.floorScratch[SEGMENTS_PER_BAND] = this.floorScratch[0];
    }
    return anyLift;
  }

  /**
   * Smooths `floorScratch` in-place via taut elastic tension-envelope sweeps:
   * r[k] = max(floor[k], 0.5*(r[k-1] + r[k+1]) - maxSag).
   */
  private smoothElasticEnvelope(
    profileOut: Float32Array,
    outOffset: number,
    sweeps: number,
    sagSlack = 0.0004
  ): void {
    const floor = this.floorScratch;
    for (let k = 0; k < RINGS_PER_BAND; k++) {
      profileOut[outOffset + k] = floor[k];
    }

    for (let pass = 0; pass < sweeps; pass++) {
      // Forward pass
      for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
        const kPrev = k === 0 ? SEGMENTS_PER_BAND - 1 : k - 1;
        const kNext = k + 1 === SEGMENTS_PER_BAND ? 0 : k + 1;
        const rCur = profileOut[outOffset + k];
        const maxSag = HALF_DTHETA_SQ * rCur + sagSlack;
        const avg =
          0.5 * (profileOut[outOffset + kPrev] + profileOut[outOffset + kNext]) - maxSag;
        profileOut[outOffset + k] = avg > floor[k] ? avg : floor[k];
      }
      profileOut[outOffset + SEGMENTS_PER_BAND] = profileOut[outOffset];

      // Backward pass for fast symmetric propagation
      for (let k = SEGMENTS_PER_BAND - 1; k >= 0; k--) {
        const kPrev = k === 0 ? SEGMENTS_PER_BAND - 1 : k - 1;
        const kNext = k + 1 === SEGMENTS_PER_BAND ? 0 : k + 1;
        const rCur = profileOut[outOffset + k];
        const maxSag = HALF_DTHETA_SQ * rCur + sagSlack;
        const avg =
          0.5 * (profileOut[outOffset + kPrev] + profileOut[outOffset + kNext]) - maxSag;
        profileOut[outOffset + k] = avg > floor[k] ? avg : floor[k];
      }
      profileOut[outOffset + SEGMENTS_PER_BAND] = profileOut[outOffset];
    }
  }

  private rebuildRadialProfiles(captureRestingMin = false): void {
    const coreMinCenter = CORE_RADIUS + CORE_CLEARANCE;
    let maxBallR = CORE_RADIUS + CORE_CLEARANCE;

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rtI = this.runtimes[i];
      if (rtI.status !== 'intact' && rtI.status !== 'pulled') continue;

      const specI = this.specs[i];
      const halfHI = specI.thickness * 0.5;
      const halfWI = specI.width * 0.5;
      const minCoreR = coreMinCenter + halfHI;
      const baseFloor = captureRestingMin ? specI.baseRadius : this.restingMinRadius[i];
      const initR = Math.max(minCoreR, baseFloor + this.targetBreath[i]);
      const profOffset = i * RINGS_PER_BAND;
      const ptsOffsetI = profOffset * 3;
      const halfOffsetI = i * SEGMENTS_PER_BAND * 3;
      const ni = specI.normal;
      const tui = specI.tangentU;
      const tvi = specI.tangentV;

      for (let k = 0; k < RINGS_PER_BAND; k++) {
        this.floorScratch[k] = initR;
      }

      // Wrap inside-out over every underlying intact/pulled band j < i (outer-to-inner to maximize early-reject)
      for (let j = i - 1; j >= 0; j--) {
        const rtJ = this.runtimes[j];
        if (rtJ.status !== 'intact' && rtJ.status !== 'pulled') {
          this.winActive[j] = 0;
          continue;
        }

        const specJ = this.specs[j];
        const halfHJ = specJ.thickness * 0.5;
        const halfWJ = specJ.width * 0.5;
        const jMaxOuter = Math.hypot(this.maxBandRadius[j] + halfHJ, halfWJ);
        if (jMaxOuter + halfHI + CONTACT_GAP <= initR) {
          this.winActive[j] = 0;
          continue;
        }

        // Analytic great-circle crossing window: only inspect segments near the 2 crossing zones
        const nj = specJ.normal;
        const a = tui[0] * nj[0] + tui[1] * nj[1] + tui[2] * nj[2];
        const b = tvi[0] * nj[0] + tvi[1] * nj[1] + tvi[2] * nj[2];
        const amp = Math.hypot(a, b);
        const minRCheck = Math.max(CORE_RADIUS, Math.min(initR, this.minBandRadius[j]));
        const tau =
          (halfWJ + SHOULDER_MARGIN + halfWI * (this.maxBandRadius[j] / minRCheck) + halfHI + 0.012) /
          minRCheck;

        let winStart0 = 0;
        let winCount = SEGMENTS_PER_BAND;
        let twoWindows = false;
        let winStart1 = 0;
        let jMax0 = this.maxBandRadius[j];
        let jMax1 = jMax0;
        this.winActive[j] = 2;

        if (amp > tau && amp > 1e-4) {
          const ratio = Math.min(1, tau / amp);
          const halfWinRad = Math.asin(ratio) + DTHETA * 1.5;
          const halfWinSegs = Math.min(SEGMENTS_PER_BAND >> 1, Math.ceil(halfWinRad / DTHETA));
          // Zero of a*cos(theta) + b*sin(theta) occurs at theta0 = atan2(-a, b) and theta0 + PI
          let theta0 = Math.atan2(-a, b);
          if (theta0 < 0) theta0 += TWO_PI;
          const kCross0 = Math.round((theta0 / TWO_PI) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          const kCross1 = (kCross0 + (SEGMENTS_PER_BAND >> 1)) % SEGMENTS_PER_BAND;
          winCount = halfWinSegs * 2 + 1;
          if (winCount * 2 < SEGMENTS_PER_BAND) {
            twoWindows = true;
            winStart0 = (kCross0 - halfWinSegs + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
            winStart1 = (kCross1 - halfWinSegs + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;

            const invAmp = 1 / amp;
            const cos0 = b * invAmp;
            const sin0 = -a * invAmp;
            const c0x = cos0 * tui[0] + sin0 * tvi[0];
            const c0y = cos0 * tui[1] + sin0 * tvi[1];
            const c0z = cos0 * tui[2] + sin0 * tvi[2];
            const uj = specJ.tangentU;
            const vj = specJ.tangentV;
            let phi0 = Math.atan2(
              c0x * vj[0] + c0y * vj[1] + c0z * vj[2],
              c0x * uj[0] + c0y * uj[1] + c0z * uj[2]
            );
            if (phi0 < 0) phi0 += TWO_PI;
            const sCross0 = Math.round((phi0 / TWO_PI) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
            const sCross1 = (sCross0 + (SEGMENTS_PER_BAND >> 1)) % SEGMENTS_PER_BAND;

            const tauJ =
              (halfWJ + halfWI + halfHJ + halfHI + 0.055) /
              (Math.max(CORE_RADIUS, this.minBandRadius[j]) * amp);
            if (tauJ < 1.0) {
              const halfWinJSegs = Math.min(
                SEGMENTS_PER_BAND >> 1,
                Math.ceil((Math.asin(tauJ) + DTHETA * 3) / DTHETA)
              );
              const jProf = j * RINGS_PER_BAND;
              if (halfWinJSegs * 2 + 1 < SEGMENTS_PER_BAND >> 1) {
                let m0 = 0;
                let m1 = 0;
                for (let ds = -halfWinJSegs; ds <= halfWinJSegs; ds++) {
                  const s0 = (sCross0 + ds + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
                  const s1 = (sCross1 + ds + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
                  const r0 = this.radialProfile[jProf + s0];
                  const r1 = this.radialProfile[jProf + s1];
                  if (r0 > m0) m0 = r0;
                  if (r1 > m1) m1 = r1;
                }
                jMax0 = m0;
                jMax1 = m1;
              }
            }

            const accTau = (halfWJ + halfWI + 0.055) / (Math.max(CORE_RADIUS, initR) * amp);
            if (accTau < 1.0) {
              const accHalf = Math.min(
                SEGMENTS_PER_BAND >> 1,
                Math.ceil((Math.asin(accTau) + DTHETA * 1.5) / DTHETA)
              );
              const accCount = accHalf * 2 + 1;
              if (accCount * 2 < SEGMENTS_PER_BAND) {
                this.winActive[j] = 1;
                this.winJMax0[j] = jMax0;
                this.winJMax1[j] = jMax1;
                this.winStart0Arr[j] = (kCross0 - accHalf + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
                this.winStart1Arr[j] = (kCross1 - accHalf + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
                this.winCountArr[j] = accCount;
              }
            }
          } else {
            winCount = SEGMENTS_PER_BAND;
          }
        }

        const numWindows = twoWindows ? 2 : 1;
        for (let w = 0; w < numWindows; w++) {
          const startK = w === 0 ? winStart0 : winStart1;
          const rMaxNeedJ =
            (twoWindows ? (w === 0 ? jMax0 : jMax1) : this.maxBandRadius[j]) +
            halfHJ +
            halfHI +
            CONTACT_GAP +
            0.01;
          for (let step = 0; step < winCount; step++) {
            const k = (startK + step) % SEGMENTS_PER_BAND;
            const kNext = k + 1 === SEGMENTS_PER_BAND ? 0 : k + 1;
            const curFloorK = this.floorScratch[k];
            if (curFloorK < rMaxNeedJ) {
              const pIdx = ptsOffsetI + k * 3;
              const clearR = this.evalRibbonPointClearance(
                this.unitCirclePts[pIdx],
                this.unitCirclePts[pIdx + 1],
                this.unitCirclePts[pIdx + 2],
                ni,
                halfWI,
                halfHI,
                curFloorK,
                j
              );
              if (clearR > curFloorK) {
                this.floorScratch[k] = clearR;
              }
            }

            const curFloorMid =
              Math.min(this.floorScratch[k], this.floorScratch[kNext]) * COS_HALF_DTHETA;
            if (curFloorMid < rMaxNeedJ) {
              const hIdx = halfOffsetI + k * 3;
              const clearMid = this.evalRibbonPointClearance(
                this.unitHalfPts[hIdx],
                this.unitHalfPts[hIdx + 1],
                this.unitHalfPts[hIdx + 2],
                ni,
                halfWI,
                halfHI,
                curFloorMid,
                j
              );
              if (clearMid > 0) {
                const reqEndpoint = clearMid / COS_HALF_DTHETA;
                if (reqEndpoint > this.floorScratch[k]) this.floorScratch[k] = reqEndpoint;
                if (reqEndpoint > this.floorScratch[kNext]) this.floorScratch[kNext] = reqEndpoint;
              }
            }
          }
        }
      }

      this.floorScratch[SEGMENTS_PER_BAND] = this.floorScratch[0];

      // Smooth taut elastic bridging between crossings (convex tension, no concave dips)
      this.smoothElasticEnvelope(this.radialProfile, profOffset, 32, 0.0004);

      // Exact 3D chamfered-vertex & chord-midpoint verification against resting OBBs j < i
      for (let pass = 0; pass < 4; pass++) {
        for (let k = 0; k < RINGS_PER_BAND; k++) {
          const rk = this.radialProfile[profOffset + k];
          this.floorScratch[k] = rk;
          const src3 = ptsOffsetI + k * 3;
          const dst3 = k * 3;
          this.scratchPtsI[dst3] = rk * this.unitCirclePts[src3];
          this.scratchPtsI[dst3 + 1] = rk * this.unitCirclePts[src3 + 1];
          this.scratchPtsI[dst3 + 2] = rk * this.unitCirclePts[src3 + 2];
        }
        this.buildBandVerts(i, this.scratchPtsI, 0, null, 0);

        let curMinR = Infinity;
        for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
          const rk = this.radialProfile[profOffset + k];
          if (rk < curMinR) curMinR = rk;
        }

        let anyPassLift = false;
        for (let j = 0; j < i; j++) {
          const wAct = this.winActive[j];
          if (wAct === 0) continue;
          const rtJ = this.runtimes[j];
          if (rtJ.status !== 'intact' && rtJ.status !== 'pulled') continue;
          const specJ = this.specs[j];
          if (curMinR - specI.thickness > this.maxBandRadius[j] + specJ.thickness + 0.016) continue;

          if (pass === 0 && wAct === 1) {
            const w0 = this.winStart0Arr[j];
            const w1 = this.winStart1Arr[j];
            const wc = this.winCountArr[j];
            let minI0 = Infinity;
            let minI1 = Infinity;
            for (let s = 0; s < wc; s++) {
              const r0 = this.radialProfile[profOffset + ((w0 + s) & (SEGMENTS_PER_BAND - 1))];
              const r1 = this.radialProfile[profOffset + ((w1 + s) & (SEGMENTS_PER_BAND - 1))];
              if (r0 < minI0) minI0 = r0;
              if (r1 < minI1) minI1 = r1;
            }
            if (
              minI0 - halfHI > this.winJMax0[j] + specJ.thickness * 0.5 + 0.015 &&
              minI1 - halfHI > this.winJMax1[j] + specJ.thickness * 0.5 + 0.015
            ) {
              continue;
            }
          }

          if (
            this.accumulateOBBLiftFromBandJ(
              this.scratchPtsI,
              0,
              specI,
              j,
              this.restSegCenter,
              this.restSegT,
              this.restSegB,
              this.restSegR,
              this.restSegHalfL,
              this.restSegHalfW,
              this.restSegHalfH,
              true,
              0.00028,
              null,
              0,
              null,
              0,
              curMinR,
              true
            )
          ) {
            anyPassLift = true;
          }
        }

        if (!anyPassLift) break;
        // Re-smooth lifted elbows back into taut convex bridges (clamped >= floorScratch) before re-verifying
        this.floorScratch[SEGMENTS_PER_BAND] = this.floorScratch[0];
        this.smoothElasticEnvelope(this.radialProfile, profOffset, 16, 0.00045);
      }

      // Build finalized 3D resting centerlines & segment OBB for band i
      let minR = Infinity;
      let maxR = initR;
      let rx0 = 1e9;
      let rx1 = -1e9;
      let ry0 = 1e9;
      let ry1 = -1e9;
      let rz0 = 1e9;
      let rz1 = -1e9;
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const rk = this.radialProfile[profOffset + k];
        if (k < SEGMENTS_PER_BAND) {
          if (rk < minR) minR = rk;
          if (rk > maxR) maxR = rk;
        }
        const src3 = ptsOffsetI + k * 3;
        const dst3 = k * 3;
        const px = rk * this.unitCirclePts[src3];
        const py = rk * this.unitCirclePts[src3 + 1];
        const pz = rk * this.unitCirclePts[src3 + 2];
        this.scratchPtsI[dst3] = px;
        this.scratchPtsI[dst3 + 1] = py;
        this.scratchPtsI[dst3 + 2] = pz;
        if (k < SEGMENTS_PER_BAND) {
          if (px < rx0) rx0 = px;
          if (px > rx1) rx1 = px;
          if (py < ry0) ry0 = py;
          if (py > ry1) ry1 = py;
          if (pz < rz0) rz0 = pz;
          if (pz > rz1) rz1 = pz;
        }
      }
      const rbOff = i * 6;
      this.restBandBox[rbOff] = rx0;
      this.restBandBox[rbOff + 1] = rx1;
      this.restBandBox[rbOff + 2] = ry0;
      this.restBandBox[rbOff + 3] = ry1;
      this.restBandBox[rbOff + 4] = rz0;
      this.restBandBox[rbOff + 5] = rz1;
      this.buildBandOBB(
        i,
        this.scratchPtsI,
        0,
        null,
        0,
        this.restSegCenter,
        this.restSegT,
        this.restSegB,
        this.restSegR,
        this.restSegHalfL,
        this.restSegHalfW,
        this.restSegHalfH,
        i * SEGMENTS_PER_BAND
      );

      this.minBandRadius[i] = minR;
      this.maxBandRadius[i] = maxR;
      if (captureRestingMin) {
        this.restingMinRadius[i] = minR;
      }
      if (maxR + halfHI > maxBallR) {
        maxBallR = maxR + halfHI;
      }
      this.bandVersion[i]++;
    }

    this.ballMaxOuterRadius = maxBallR;
  }

  /**
   * Computes non-intersecting 3D centerlines for all active bands whenever any band is pulled
   * or twanging. Underlying pulls/twangs physically bow overlying cross-bands outward so no
   * ribbon ever ghosts through an overlying band.
   */
  private recomputeDynamicDeformation(): void {
    let anyActive = false;
    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      if (
        (rt.status === 'pulled' && rt.pullAmount > 0) ||
        (rt.status === 'intact' && rt.twangAmp > 1e-4)
      ) {
        anyActive = true;
        break;
      }
    }

    if (!anyActive) {
      for (let i = 0; i < TOTAL_BANDS; i++) {
        if (this.prevDynActive[i]) {
          this.prevDynActive[i] = 0;
          this.dynActive[i] = 0;
          this.bandVersion[i]++;
        }
      }
      return;
    }

    const coreMinCenter = CORE_RADIUS + CORE_CLEARANCE;

    // Seed dynamic radii from resting wrapped profile + current breathing state
    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      this.dynActive[i] =
        (rt.status === 'pulled' && rt.pullAmount > 0) ||
        (rt.status === 'intact' && rt.twangAmp > 1e-4)
          ? 1
          : 0;
      if (rt.status !== 'intact' && rt.status !== 'pulled') continue;

      const breathShift = this.currentBreath[i] - this.targetBreath[i];
      const minCoreR = coreMinCenter + this.specs[i].thickness * 0.5;
      const profOffset = i * RINGS_PER_BAND;
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const r = this.radialProfile[profOffset + k] + breathShift;
        this.dynRadii[profOffset + k] = r > minCoreR ? r : minCoreR;
      }
    }

    // Inside-out propagation: evaluate band i, then lift overlying bands j > i wherever i's lifted arch crosses j
    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      if (rt.status !== 'intact' && rt.status !== 'pulled') continue;

      const isPulled = rt.status === 'pulled' && rt.pullAmount > 0;
      const hasTwang = !isPulled && rt.twangAmp > 1e-4;
      const isSelfDriven = isPulled || hasTwang;
      const isCoupledLift = this.dynActive[i] === 1 && !isSelfDriven;

      const spec = this.specs[i];
      const profOffset = i * RINGS_PER_BAND;
      const ptsOffset = profOffset * 3;
      const breathShift = this.currentBreath[i] - this.targetBreath[i];
      const minCoreR = coreMinCenter + spec.thickness * 0.5;

      // Smooth bowed overlying & twanging bands into elastic arches (never jagged sawtooth zigzags)
      if (this.dynActive[i] === 1 && !isPulled) {
        for (let k = 0; k < RINGS_PER_BAND; k++) {
          this.floorScratch[k] = this.dynRadii[profOffset + k];
        }
        this.smoothElasticEnvelope(this.dynRadii, profOffset, 10, 0.0005);
      }

      // Construct 3D centerline points & strains for band i
      if (this.dynActive[i] === 1) {
        const twangSin = hasTwang ? Math.sin(rt.twangPhase) * rt.twangAmp : 0;
        const nearBreak = isPulled
          ? Math.max(
              0,
              (rt.currentStrain - spec.breakStrain * 0.68) / (spec.breakStrain * 0.32 + 1e-4)
            )
          : 0;
        const tremble =
          nearBreak > 0 ? Math.sin(this.simTimeSec * 72.0 + i * 2.1) * 0.012 * nearBreak : 0;

        let avgR = 0;
        for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
          avgR += this.dynRadii[profOffset + k];
        }
        avgR /= SEGMENTS_PER_BAND;

        const pullHalfArc = isPulled
          ? Math.min(
              0.24,
              Math.max(0.13, Math.acos(Math.min(0.98, avgR / (avgR + rt.pullAmount + 0.03))) / TWO_PI)
            )
          : 0.22;

        let maxLift = 0;
        for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
          const u = k === SEGMENTS_PER_BAND ? 0 : k / SEGMENTS_PER_BAND;
          const pIdx = ptsOffset + k * 3;
          const rLayer = this.dynRadii[profOffset + k];
          let px = rLayer * this.unitCirclePts[pIdx];
          let py = rLayer * this.unitCirclePts[pIdx + 1];
          let pz = rLayer * this.unitCirclePts[pIdx + 2];
          let strain = rt.currentStrain;
          let ringMoved = Math.abs(rLayer - (this.radialProfile[profOffset + k] + breathShift)) > 1e-5;

          if (isPulled) {
            const d = shortestArcDist(u, rt.pullU);
            const vShape = evalPullArch(d, pullHalfArc);
            const disp = Math.max(0, (rt.pullAmount + tremble) * vShape);
            if (disp > 1e-5) ringMoved = true;
            px += rt.pullDir[0] * disp;
            py += rt.pullDir[1] * disp;
            pz += rt.pullDir[2] * disp;
            strain = spec.restStrain + (rt.currentStrain - spec.restStrain) * (0.35 + 0.65 * vShape);

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
            const disp = Math.max(0, twangSin * env);
            if (disp > 1e-5) ringMoved = true;
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
          } else {
            // Bowed overlying band: strain bumps gently proportional to outward bow
            const restR = Math.max(minCoreR, this.radialProfile[profOffset + k] + breathShift);
            const bowLift = Math.max(0, rLayer - restR);
            if (bowLift > maxLift) maxLift = bowLift;
            strain = Math.min(spec.breakStrain * 0.96, spec.restStrain + bowLift * 0.65);
          }

          this.dynPts[pIdx] = px;
          this.dynPts[pIdx + 1] = py;
          this.dynPts[pIdx + 2] = pz;
          this.dynStrains[profOffset + k] = strain;
          this.dynRingMoved[profOffset + k] = ringMoved ? 1 : 0;
        }

        // Enforce exact 3D OBB clearance against all underlying bands m < i
        if (isSelfDriven || isCoupledLift) {
          this.enforceExactDynClearanceAgainstLower(i, isSelfDriven);
        }

        // Compute tight dirty AABB of moved segments for fast downstream OBB filtering
        let bx0 = 1e9;
        let bx1 = -1e9;
        let by0 = 1e9;
        let by1 = -1e9;
        let bz0 = 1e9;
        let bz1 = -1e9;
        let anyDirty = false;
        for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
          if (this.dynRingMoved[profOffset + k] === 0) continue;
          anyDirty = true;
          const p3 = ptsOffset + k * 3;
          const x = this.dynPts[p3];
          const y = this.dynPts[p3 + 1];
          const z = this.dynPts[p3 + 2];
          if (x < bx0) bx0 = x;
          if (x > bx1) bx1 = x;
          if (y < by0) by0 = y;
          if (y > by1) by1 = y;
          if (z < bz0) bz0 = z;
          if (z > bz1) bz1 = z;
        }
        const boxOff = i * 6;
        if (anyDirty) {
          const pad = 0.19;
          this.dynDirtyBox[boxOff] = bx0 - pad;
          this.dynDirtyBox[boxOff + 1] = bx1 + pad;
          this.dynDirtyBox[boxOff + 2] = by0 - pad;
          this.dynDirtyBox[boxOff + 3] = by1 + pad;
          this.dynDirtyBox[boxOff + 4] = bz0 - pad;
          this.dynDirtyBox[boxOff + 5] = bz1 + pad;
        } else {
          this.dynDirtyBox[boxOff] = 1;
          this.dynDirtyBox[boxOff + 1] = -1;
          this.dynDirtyBox[boxOff + 2] = 1;
          this.dynDirtyBox[boxOff + 3] = -1;
          this.dynDirtyBox[boxOff + 4] = 1;
          this.dynDirtyBox[boxOff + 5] = -1;
        }

        // Cache finalized 3D dynamic AABB, maxR, moved-seg list, & OBB for band i
        let dbx0 = 1e9;
        let dbx1 = -1e9;
        let dby0 = 1e9;
        let dby1 = -1e9;
        let dbz0 = 1e9;
        let dbz1 = -1e9;
        let dMaxR = 0;
        let dMaxPlaneOff = 0;
        const niDyn0 = spec.normal[0];
        const niDyn1 = spec.normal[1];
        const niDyn2 = spec.normal[2];
        let mSegCount = 0;
        const mSegBase = i * SEGMENTS_PER_BAND;
        for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
          const p3 = ptsOffset + k * 3;
          const x = this.dynPts[p3];
          const y = this.dynPts[p3 + 1];
          const z = this.dynPts[p3 + 2];
          const rk = Math.hypot(x, y, z);
          if (rk > dMaxR) dMaxR = rk;
          const pOff = Math.abs(x * niDyn0 + y * niDyn1 + z * niDyn2);
          if (pOff > dMaxPlaneOff) dMaxPlaneOff = pOff;
          if (x < dbx0) dbx0 = x;
          if (x > dbx1) dbx1 = x;
          if (y < dby0) dby0 = y;
          if (y > dby1) dby1 = y;
          if (z < dbz0) dbz0 = z;
          if (z > dbz1) dbz1 = z;
          const kNext = k + 1 === SEGMENTS_PER_BAND ? 0 : k + 1;
          if (
            this.dynRingMoved[profOffset + k] !== 0 ||
            this.dynRingMoved[profOffset + kNext] !== 0
          ) {
            this.dynMovedSegIdx[mSegBase + mSegCount++] = k;
          }
        }
        this.dynMaxBandRadius[i] = dMaxR;
        this.dynMaxPlaneOff[i] = dMaxPlaneOff;
        this.dynMovedSegCount[i] = mSegCount;
        this.dynBandBox[boxOff] = dbx0;
        this.dynBandBox[boxOff + 1] = dbx1;
        this.dynBandBox[boxOff + 2] = dby0;
        this.dynBandBox[boxOff + 3] = dby1;
        this.dynBandBox[boxOff + 4] = dbz0;
        this.dynBandBox[boxOff + 5] = dbz1;

        this.buildBandOBB(
          i,
          this.dynPts,
          ptsOffset,
          this.dynStrains,
          profOffset,
          this.dynSegCenter,
          this.dynSegT,
          this.dynSegB,
          this.dynSegR,
          this.dynSegHalfL,
          this.dynSegHalfW,
          this.dynSegHalfH,
          i * SEGMENTS_PER_BAND
        );

        // Propagate lifted segments outward onto overlying bands j > i
        const halfHI = spec.thickness * 0.5;
        const halfWI = spec.width * 0.5;
        const ni = spec.normal;

        for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
          const p0Idx = ptsOffset + k * 3;
          const p1Idx = ptsOffset + (k + 1) * 3;
          const r0 = Math.hypot(this.dynPts[p0Idx], this.dynPts[p0Idx + 1], this.dynPts[p0Idx + 2]);
          const r1 = Math.hypot(this.dynPts[p1Idx], this.dynPts[p1Idx + 1], this.dynPts[p1Idx + 2]);
          const restR0 = Math.max(minCoreR, this.radialProfile[profOffset + k] + breathShift);
          const restR1 = Math.max(minCoreR, this.radialProfile[profOffset + k + 1] + breathShift);
          if (r0 <= restR0 + 0.00022 && r1 <= restR1 + 0.00022) continue;

          // Compute exact local binormal B matching rubberOptics.ts frame
          const prevIdx = ptsOffset + (k > 0 ? k - 1 : SEGMENTS_PER_BAND - 1) * 3;
          const nextIdx = ptsOffset + (k + 1) * 3;
          const tx = this.dynPts[nextIdx] - this.dynPts[prevIdx];
          const ty = this.dynPts[nextIdx + 1] - this.dynPts[prevIdx + 1];
          const tz = this.dynPts[nextIdx + 2] - this.dynPts[prevIdx + 2];

          const px0 = this.dynPts[p0Idx];
          const py0 = this.dynPts[p0Idx + 1];
          const pz0 = this.dynPts[p0Idx + 2];
          let bx = py0 * tz - pz0 * ty;
          let by = pz0 * tx - px0 * tz;
          let bz = px0 * ty - py0 * tx;
          let bLen = Math.hypot(bx, by, bz);
          if (bLen < 1e-5) {
            bx = ni[0];
            by = ni[1];
            bz = ni[2];
            bLen = Math.hypot(bx, by, bz) || 1;
          }
          bx /= bLen;
          by /= bLen;
          bz /= bLen;

          const mx = 0.5 * (px0 + this.dynPts[p1Idx]);
          const my = 0.5 * (py0 + this.dynPts[p1Idx + 1]);
          const mz = 0.5 * (pz0 + this.dynPts[p1Idx + 2]);
          const rSegMax = Math.max(r0, r1);

          // Pre-filter overlying bands j > i ONCE per lifted segment k before sampling ribbon width
          let overCount = 0;
          const mRadBase = rSegMax + halfHI + halfWI + CONTACT_GAP + 0.015;
          const slabBase = halfWI + halfHI + 0.045;
          for (let j = i + 1; j < TOTAL_BANDS; j++) {
            const rtJ = this.runtimes[j];
            if (rtJ.status !== 'intact' && rtJ.status !== 'pulled') continue;
            const specJ = this.specs[j];
            const halfHJ = specJ.thickness * 0.5;
            if (mRadBase + halfHJ <= this.minBandRadius[j]) continue;
            const nj = specJ.normal;
            const slabM = Math.abs(mx * nj[0] + my * nj[1] + mz * nj[2]);
            if (slabM > specJ.width * 0.5 + slabBase + halfHJ) continue;
            this.candOverBands[overCount++] = j;
          }
          if (overCount === 0) continue;

          // Check both vertex k and midpoint k + 0.5 across band i's ribbon width
          for (let sub = 0; sub < 2; sub++) {
            const cx = sub === 0 ? px0 : mx;
            const cy = sub === 0 ? py0 : my;
            const cz = sub === 0 ? pz0 : mz;
            const cRadius = Math.hypot(cx, cy, cz);

            for (let li = 0; li < LAT_FRACS.length; li++) {
              const latOff = LAT_FRACS[li] * halfWI;
              const qx = cx + latOff * bx;
              const qy = cy + latOff * by;
              const qz = cz + latOff * bz;
              const qRadius = Math.hypot(qx, qy, qz);
              const invQ = 1 / (qRadius || 1);
              const uqx = qx * invQ;
              const uqy = qy * invQ;
              const uqz = qz * invQ;

              for (let oi = 0; oi < overCount; oi++) {
                const j = this.candOverBands[oi];
                const specJ = this.specs[j];
                const halfHJ = specJ.thickness * 0.5;
                const halfWJ = specJ.width * 0.5;
                if (qRadius + halfHI + halfHJ + CONTACT_GAP + 0.008 <= this.minBandRadius[j]) continue;

                const nj = specJ.normal;
                const pn = Math.abs(uqx * nj[0] + uqy * nj[1] + uqz * nj[2]);
                const padJ = halfWJ + Math.max(SHOULDER_MARGIN, halfHI + halfHJ + 0.005);
                if (pn * Math.max(qRadius, cRadius) > padJ + 0.01) continue;

                const uj = specJ.tangentU;
                const vj = specJ.tangentV;
                const pu = uqx * uj[0] + uqy * uj[1] + uqz * uj[2];
                const pv = uqx * vj[0] + uqy * vj[1] + uqz * vj[2];

                let angleJ = Math.atan2(pv, pu);
                if (angleJ < 0) angleJ += TWO_PI;
                const kJFloat = (angleJ / TWO_PI) * SEGMENTS_PER_BAND;
                const kJ0 = Math.floor(kJFloat) % SEGMENTS_PER_BAND;
                const kJ1 = (kJ0 + 1) % SEGMENTS_PER_BAND;
                const profJ = j * RINGS_PER_BAND;

                const rJCur = Math.max(this.dynRadii[profJ + kJ0], this.dynRadii[profJ + kJ1]);
                const latDistJ = pn * Math.max(rJCur, qRadius);
                if (latDistJ > padJ) continue;

                // Ensure j's inner surface clears i's outer surface (exact OBB pass refines corners afterward)
                const minReqJ = (qRadius + halfHI + halfHJ + CONTACT_GAP + 0.0005) / COS_HALF_DTHETA;
                if (minReqJ > this.dynRadii[profJ + kJ0]) {
                  this.dynRadii[profJ + kJ0] = minReqJ;
                  this.dynActive[j] = 1;
                }
                if (minReqJ > this.dynRadii[profJ + kJ1]) {
                  this.dynRadii[profJ + kJ1] = minReqJ;
                  this.dynActive[j] = 1;
                }
                if (kJ0 === 0 || kJ1 === 0) {
                  this.dynRadii[profJ + SEGMENTS_PER_BAND] = this.dynRadii[profJ];
                }
              }
            }
          }
        }

        this.bandVersion[i]++;
      } else if (this.prevDynActive[i]) {
        this.bandVersion[i]++;
      }
    }

    for (let i = 0; i < TOTAL_BANDS; i++) {
      this.prevDynActive[i] = this.dynActive[i];
    }
  }

  /**
   * Ensures dynamic band i (whether pulled, twanging, or bowed overlying) clears every
   * underlying band m < i across all 8 chamfered vertices and segment midpoints.
   */
  private enforceExactDynClearanceAgainstLower(bandIdx: number, isSelfDrivenI: boolean): void {
    const specI = this.specs[bandIdx];
    const profI = bandIdx * RINGS_PER_BAND;
    const ptsOffsetI = profI * 3;
    const breathI = Math.abs(this.currentBreath[bandIdx] - this.targetBreath[bandIdx]);

    for (let pass = 0; pass < 6; pass++) {
      let minRI = Infinity;
      let maxRI = 0;
      let minRDirtyI = Infinity;
      let dCount = 0;
      let dx0 = 1e9;
      let dx1 = -1e9;
      let dy0 = 1e9;
      let dy1 = -1e9;
      let dz0 = 1e9;
      let dz1 = -1e9;
      let fx0 = 1e9;
      let fx1 = -1e9;
      let fy0 = 1e9;
      let fy1 = -1e9;
      let fz0 = 1e9;
      let fz1 = -1e9;
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const p3 = ptsOffsetI + k * 3;
        const px = this.dynPts[p3];
        const py = this.dynPts[p3 + 1];
        const pz = this.dynPts[p3 + 2];
        const rk = Math.hypot(px, py, pz);
        this.floorScratch[k] = rk;
        if (k < SEGMENTS_PER_BAND) {
          if (rk < minRI) minRI = rk;
          if (rk > maxRI) maxRI = rk;
          if (px < fx0) fx0 = px;
          if (px > fx1) fx1 = px;
          if (py < fy0) fy0 = py;
          if (py > fy1) fy1 = py;
          if (pz < fz0) fz0 = pz;
          if (pz > fz1) fz1 = pz;
          if (this.dynRingMoved[profI + k] !== 0) {
            this.dirtyRingIdx[dCount++] = k;
            if (rk < minRDirtyI) minRDirtyI = rk;
            if (px < dx0) dx0 = px;
            if (px > dx1) dx1 = px;
            if (py < dy0) dy0 = py;
            if (py > dy1) dy1 = py;
            if (pz < dz0) dz0 = pz;
            if (pz > dz1) dz1 = pz;
          }
        }
      }
      this.dirtyRingCount = dCount;
      this.dirtyRingBox[0] = dx0;
      this.dirtyRingBox[1] = dx1;
      this.dirtyRingBox[2] = dy0;
      this.dirtyRingBox[3] = dy1;
      this.dirtyRingBox[4] = dz0;
      this.dirtyRingBox[5] = dz1;
      this.iFullBox[0] = fx0;
      this.iFullBox[1] = fx1;
      this.iFullBox[2] = fy0;
      this.iFullBox[3] = fy1;
      this.iFullBox[4] = fz0;
      this.iFullBox[5] = fz1;
      const cullMinR =
        breathI <= 1e-4 && minRDirtyI < Infinity ? minRDirtyI : minRI;

      let builtVerts = false;
      let anyLift = false;
      const dCx = 0.5 * (dx0 + dx1);
      const dCy = 0.5 * (dy0 + dy1);
      const dCz = 0.5 * (dz0 + dz1);
      const dHx = 0.5 * (dx1 - dx0);
      const dHy = 0.5 * (dy1 - dy0);
      const dHz = 0.5 * (dz1 - dz0);
      const ni = specI.normal;
      const ani0 = Math.abs(ni[0]);
      const ani1 = Math.abs(ni[1]);
      const ani2 = Math.abs(ni[2]);
      const halfWI = specI.width * 0.5;

      for (let m = 0; m < bandIdx; m++) {
        const rtM = this.runtimes[m];
        if (rtM.status !== 'intact' && rtM.status !== 'pulled') continue;
        const isDynM = this.dynActive[m] === 1;
        const breathM = this.currentBreath[m] - this.targetBreath[m];
        if (!isSelfDrivenI && !isDynM && breathI + Math.abs(breathM) <= 1e-4) continue;
        if (!isDynM && dCount === 0) continue;

        const specM = this.specs[m];
        const mMaxR = isDynM ? this.dynMaxBandRadius[m] : this.maxBandRadius[m];
        const rCheckI = isDynM ? minRI : cullMinR;
        if (rCheckI - specI.thickness > mMaxR + specM.thickness + 0.018) {
          continue;
        }

        const isSelfDrivenM =
          isDynM &&
          ((rtM.status === 'pulled' && rtM.pullAmount > 0) ||
            (rtM.status === 'intact' && rtM.twangAmp > 1e-4));

        const mOff6 = m * 6;
        if (!isDynM) {
          if (
            dx1 + 0.17 < this.restBandBox[mOff6] ||
            dx0 - 0.17 > this.restBandBox[mOff6 + 1] ||
            dy1 + 0.17 < this.restBandBox[mOff6 + 2] ||
            dy0 - 0.17 > this.restBandBox[mOff6 + 3] ||
            dz1 + 0.17 < this.restBandBox[mOff6 + 4] ||
            dz0 - 0.17 > this.restBandBox[mOff6 + 5]
          ) {
            continue;
          }
          const nm = specM.normal;
          const slabThreshM = specM.width * 0.5 + halfWI + 0.045;
          const cDist = Math.abs(dCx * nm[0] + dCy * nm[1] + dCz * nm[2]);
          const pRad = dHx * Math.abs(nm[0]) + dHy * Math.abs(nm[1]) + dHz * Math.abs(nm[2]);
          if (cDist > pRad + slabThreshM) continue;
        } else {
          let dirtyHitsM =
            dCount > 0 &&
            dx1 + 0.17 >= this.dynBandBox[mOff6] &&
            dx0 - 0.17 <= this.dynBandBox[mOff6 + 1] &&
            dy1 + 0.17 >= this.dynBandBox[mOff6 + 2] &&
            dy0 - 0.17 <= this.dynBandBox[mOff6 + 3] &&
            dz1 + 0.17 >= this.dynBandBox[mOff6 + 4] &&
            dz0 - 0.17 <= this.dynBandBox[mOff6 + 5];
          if (dirtyHitsM) {
            const nm = specM.normal;
            const slabThreshM =
              specM.width * 0.5 +
              halfWI +
              0.045 +
              (isSelfDrivenM ? this.dynMaxPlaneOff[m] : 0);
            const cDist = Math.abs(dCx * nm[0] + dCy * nm[1] + dCz * nm[2]);
            const pRad = dHx * Math.abs(nm[0]) + dHy * Math.abs(nm[1]) + dHz * Math.abs(nm[2]);
            if (cDist > pRad + slabThreshM) dirtyHitsM = false;
          }

          const mbx0 = this.dynDirtyBox[mOff6];
          const mbx1 = this.dynDirtyBox[mOff6 + 1];
          const mby0 = this.dynDirtyBox[mOff6 + 2];
          const mby1 = this.dynDirtyBox[mOff6 + 3];
          const mbz0 = this.dynDirtyBox[mOff6 + 4];
          const mbz1 = this.dynDirtyBox[mOff6 + 5];
          let fullHitsDirtyM =
            fx1 + 0.005 >= mbx0 &&
            fx0 - 0.005 <= mbx1 &&
            fy1 + 0.005 >= mby0 &&
            fy0 - 0.005 <= mby1 &&
            fz1 + 0.005 >= mbz0 &&
            fz0 - 0.005 <= mbz1 &&
            fx1 + 0.17 >= this.dynBandBox[mOff6] &&
            fx0 - 0.17 <= this.dynBandBox[mOff6 + 1] &&
            fy1 + 0.17 >= this.dynBandBox[mOff6 + 2] &&
            fy0 - 0.17 <= this.dynBandBox[mOff6 + 3] &&
            fz1 + 0.17 >= this.dynBandBox[mOff6 + 4] &&
            fz0 - 0.17 <= this.dynBandBox[mOff6 + 5];
          if (fullHitsDirtyM && !isSelfDrivenI) {
            const mcx = 0.5 * (mbx0 + mbx1);
            const mcy = 0.5 * (mby0 + mby1);
            const mcz = 0.5 * (mbz0 + mbz1);
            const mhx = 0.5 * (mbx1 - mbx0);
            const mhy = 0.5 * (mby1 - mby0);
            const mhz = 0.5 * (mbz1 - mbz0);
            const mcDotN = mcx * ni[0] + mcy * ni[1] + mcz * ni[2];
            const cDistI = Math.abs(mcDotN);
            const pRadI = mhx * ani0 + mhy * ani1 + mhz * ani2;
            if (cDistI > pRadI + 0.002) {
              fullHitsDirtyM = false;
            } else {
              const rBox = Math.hypot(mhx, mhy, mhz);
              const pPlane = Math.hypot(
                mcx - mcDotN * ni[0],
                mcy - mcDotN * ni[1],
                mcz - mcDotN * ni[2]
              );
              if (pPlane + rBox < minRI - 0.005 || pPlane - rBox > maxRI + 0.005) {
                fullHitsDirtyM = false;
              }
            }
          }
          if (!dirtyHitsM && !fullHitsDirtyM) continue;
        }

        if (!builtVerts) {
          this.buildBandVerts(
            bandIdx,
            this.dynPts,
            ptsOffsetI,
            this.dynStrains,
            profI
          );
          builtVerts = true;
        }

        const gap =
          CONTACT_GAP * 0.35 + (isDynM ? 0 : Math.max(0, breathM));
        if (
          this.accumulateOBBLiftFromBandJ(
            this.dynPts,
            ptsOffsetI,
            specI,
            m,
            isDynM ? this.dynSegCenter : this.restSegCenter,
            isDynM ? this.dynSegT : this.restSegT,
            isDynM ? this.dynSegB : this.restSegB,
            isDynM ? this.dynSegR : this.restSegR,
            isDynM ? this.dynSegHalfL : this.restSegHalfL,
            isDynM ? this.dynSegHalfW : this.restSegHalfW,
            isDynM ? this.dynSegHalfH : this.restSegHalfH,
            !isSelfDrivenM,
            gap,
            this.dynRingMoved,
            profI,
            isDynM ? this.dynDirtyBox : null,
            mOff6,
            minRI,
            !isSelfDrivenI
          )
        ) {
          anyLift = true;
        }
      }

      if (!anyLift) break;
      this.floorScratch[SEGMENTS_PER_BAND] = this.floorScratch[0];
      const smoothSweeps = isSelfDrivenI ? 4 : 8;
      this.smoothElasticEnvelope(this.dynEnvScratch, 0, smoothSweeps, 0.0005);
      for (let k = 0; k < RINGS_PER_BAND; k++) {
        const p3 = ptsOffsetI + k * 3;
        const curR = Math.hypot(this.dynPts[p3], this.dynPts[p3 + 1], this.dynPts[p3 + 2]);
        const targetR = this.dynEnvScratch[k];
        if (targetR > curR && curR > 1e-5) {
          const scale = targetR / curR;
          this.dynPts[p3] *= scale;
          this.dynPts[p3 + 1] *= scale;
          this.dynPts[p3 + 2] *= scale;
          if (!isSelfDrivenI) {
            this.dynRadii[profI + k] = targetR;
          }
          this.dynRingMoved[profI + k] = 1;
        }
      }
      // Bit-identical closed-loop seam
      const seamDst = ptsOffsetI + SEGMENTS_PER_BAND * 3;
      this.dynPts[seamDst] = this.dynPts[ptsOffsetI];
      this.dynPts[seamDst + 1] = this.dynPts[ptsOffsetI + 1];
      this.dynPts[seamDst + 2] = this.dynPts[ptsOffsetI + 2];
      this.dynRadii[profI + SEGMENTS_PER_BAND] = this.dynRadii[profI];
    }
  }

  private recomputeCompression(triggerCascadeFromId: number | null): void {
    let cascadeScheduled = 0;
    const srcSpec = triggerCascadeFromId !== null ? this.specs[triggerCascadeFromId] : null;

    // Enforce monotonically non-increasing targetBreath[i] so inner layers never overtake outer layers during breathing
    let prevBreath = 0.055;
    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      if (rt.status !== 'intact' && rt.status !== 'pulled') continue;

      const spec = this.specs[i];
      const currentLoad = this.computeOuterCompressionLoad(i);
      const releasedLoad = Math.max(0, this.initialOuterLoad[i] - currentLoad);

      const rawBreath = Math.min(0.05, releasedLoad * 0.02);
      const monotonicBreath = Math.min(prevBreath, rawBreath);
      prevBreath = monotonicBreath;
      this.targetBreath[i] = monotonicBreath;

      if (rt.status === 'intact') {
        const tensionReleaseBump = Math.min(0.075, releasedLoad * 0.022);
        rt.currentStrain = Math.min(spec.breakStrain * 0.98, spec.restStrain + tensionReleaseBump);
      }

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

    this.rebuildRadialProfiles(false);
    this.recomputeDynamicDeformation();
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
      this.evalWrappedPoint(bandId, uNorm, this.scratchPoint);
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
    this.bandVersion[bandId]++;
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

    rt.pullAmount = Math.max(0, Math.max(radialProj, dragLen * 0.92));
    rt.currentStrain = spec.restStrain + rt.pullAmount * STRAIN_PER_UNIT_PULL;

    if (dragLen > 1e-4) {
      // Clamp inward drag so pullDir never points back through the ball core
      const clampIn = radialProj < 0 ? radialProj : 0;
      const cx = vx - nx * clampIn;
      const cy = vy - ny * clampIn;
      const cz = vz - nz * clampIn;
      const cLen = Math.hypot(cx, cy, cz);
      let px = nx * 0.42;
      let py = ny * 0.42;
      let pz = nz * 0.42;
      if (cLen > 1e-5) {
        const invC = 1 / cLen;
        px += cx * invC * 0.88;
        py += cy * invC * 0.88;
        pz += cz * invC * 0.88;
      }
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

    this.recomputeDynamicDeformation();
    return false;
  }

  public releasePull(): void {
    if (this.activePullBandId === null) return;
    const id = this.activePullBandId;
    const rt = this.runtimes[id];
    const spec = this.specs[id];
    this.activePullBandId = null;

    if (rt.status !== 'pulled') return;

    // Give at most the 6 most-bowed overlying bands a spring-back twang so they settle organically within budget
    let topCount = 0;
    for (let j = id + 1; j < TOTAL_BANDS; j++) {
      if (!this.prevDynActive[j]) continue;
      const rj = this.runtimes[j];
      if (rj.status !== 'intact') continue;
      const profJ = j * RINGS_PER_BAND;
      const breathJ = this.currentBreath[j] - this.targetBreath[j];
      let maxBow = 0;
      let maxK = 0;
      for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
        const bow = this.dynRadii[profJ + k] - (this.radialProfile[profJ + k] + breathJ);
        if (bow > maxBow) {
          maxBow = bow;
          maxK = k;
        }
      }
      if (maxBow > 0.018) {
        let ins = topCount;
        if (topCount < 6) {
          topCount++;
        } else if (maxBow <= this.topBowAmps[5]) {
          continue;
        } else {
          ins = 5;
        }
        while (ins > 0 && maxBow > this.topBowAmps[ins - 1]) {
          if (ins < 6) {
            this.topBowIds[ins] = this.topBowIds[ins - 1];
            this.topBowAmps[ins] = this.topBowAmps[ins - 1];
            this.topBowKs[ins] = this.topBowKs[ins - 1];
          }
          ins--;
        }
        this.topBowIds[ins] = j;
        this.topBowAmps[ins] = maxBow;
        this.topBowKs[ins] = maxK;
      }
    }
    for (let ti = 0; ti < topCount; ti++) {
      const j = this.topBowIds[ti];
      const rj = this.runtimes[j];
      const maxBow = this.topBowAmps[ti];
      const maxK = this.topBowKs[ti];
      rj.twangAmp = Math.max(rj.twangAmp, maxBow * 0.9);
      rj.twangPhase = Math.PI * 0.5;
      rj.pullU = maxK / SEGMENTS_PER_BAND;
      this.evalWrappedPoint(j, rj.pullU, this.scratchPoint);
      const len = Math.hypot(this.scratchPoint[0], this.scratchPoint[1], this.scratchPoint[2]) || 1;
      rj.pullDir[0] = this.scratchPoint[0] / len;
      rj.pullDir[1] = this.scratchPoint[1] / len;
      rj.pullDir[2] = this.scratchPoint[2] / len;
    }

    rt.status = 'intact';
    rt.twangAmp = rt.pullAmount;
    rt.twangPhase = 0;
    rt.pullAmount = 0;
    rt.currentStrain = spec.restStrain;
    this.recomputeDynamicDeformation();
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

    // Wrapped surface point at snap site
    this.evalWrappedPoint(bandId, uNorm, this.scratchPoint);
    const hitX = this.scratchPoint[0];
    const hitY = this.scratchPoint[1];
    const hitZ = this.scratchPoint[2];
    const pLen = Math.hypot(hitX, hitY, hitZ) || 1;
    const radX = hitX / pLen;
    const radY = hitY / pLen;
    const radZ = hitZ / pLen;

    const avgR = 0.5 * (this.minBandRadius[bandId] + this.maxBandRadius[bandId]);
    this.snapAgeSec[bandId] = 0;
    this.snapPeakPull[bandId] = peakPull;
    this.snapHalfArc[bandId] = fromPull
      ? Math.min(0.24, Math.max(0.13, Math.acos(Math.min(0.98, avgR / (avgR + peakPull + 0.03))) / TWO_PI))
      : 0.16;

    const asym = 0.9 + this.rng() * 0.2;
    this.snapAsym[bandId] = asym;
    const latSign = this.rng() < 0.5 ? -1 : 1;
    this.snapSideSign[bandId] = latSign;

    if (fromPull) {
      this.snapLaunchDir[bandId * 3] = rt.pullDir[0];
      this.snapLaunchDir[bandId * 3 + 1] = rt.pullDir[1];
      this.snapLaunchDir[bandId * 3 + 2] = rt.pullDir[2];
    } else {
      this.snapLaunchDir[bandId * 3] = radX;
      this.snapLaunchDir[bandId * 3 + 1] = radY;
      this.snapLaunchDir[bandId * 3 + 2] = radZ;
    }

    rt.status = 'snapping';
    rt.snapU = uNorm;
    rt.leftArc = 0.5;
    rt.rightArc = 0.5;

    // Readable high-speed whip retraction (~2.0-2.8 rev/s) over ~150-250ms
    const baseWhipVel = (2.05 + strainFactor * 0.5) * (fromPull ? 1.12 : 0.98);
    rt.leftVel = baseWhipVel * asym;
    rt.rightVel = baseWhipVel / asym;
    rt.curlAmount = 0;
    rt.pullAmount = 0;
    rt.twangAmp = 0;
    rt.opacity = 1;

    // Kick the coiled scrap outward from its antipode center and sideways clear of the ball (no 6 o'clock slide)
    const thetaSnap = uNorm * TWO_PI;
    const tx = -Math.sin(thetaSnap) * spec.tangentU[0] + Math.cos(thetaSnap) * spec.tangentV[0];
    const ty = -Math.sin(thetaSnap) * spec.tangentU[1] + Math.cos(thetaSnap) * spec.tangentV[1];
    const tz = -Math.sin(thetaSnap) * spec.tangentU[2] + Math.cos(thetaSnap) * spec.tangentV[2];

    const antipodeRadX = -radX;
    const antipodeRadY = -radY;
    const antipodeRadZ = -radZ;
    const outwardKick = fromPull ? 0.55 : 0.45;
    const sideKick = (fromPull ? 1.55 : 1.3) * latSign;
    const upKick = (fromPull ? 0.85 : 0.65) + Math.max(0, -antipodeRadY) * 0.45;
    rt.fallOffset[0] = 0;
    rt.fallOffset[1] = 0;
    rt.fallOffset[2] = 0;
    rt.fallVel[0] =
      antipodeRadX * outwardKick + spec.normal[0] * sideKick - tx * latSign * 0.28;
    rt.fallVel[1] =
      antipodeRadY * outwardKick + spec.normal[1] * sideKick - ty * latSign * 0.28 + upKick;
    rt.fallVel[2] =
      antipodeRadZ * outwardKick + spec.normal[2] * sideKick - tz * latSign * 0.28;

    // Punchy ball recoil opposite the snap launch vector
    const lx = this.snapLaunchDir[bandId * 3];
    const ly = this.snapLaunchDir[bandId * 3 + 1];
    const lz = this.snapLaunchDir[bandId * 3 + 2];
    const recoilImp = (fromPull ? 2.15 : 1.18) * (0.78 + strainFactor * 0.52);
    this.recoilVel[0] -= lx * recoilImp;
    this.recoilVel[1] -= ly * recoilImp;
    this.recoilVel[2] -= lz * recoilImp;

    // Record physical snap impulse info for directional torque jolt & audio in main.ts
    this.snapCount++;
    this.hasLastSnap = true;
    this.lastSnapInfo.snapId = this.snapCount;
    this.lastSnapInfo.bandId = bandId;
    this.lastSnapInfo.hitPos[0] = hitX;
    this.lastSnapInfo.hitPos[1] = hitY;
    this.lastSnapInfo.hitPos[2] = hitZ;
    this.lastSnapInfo.pullDir[0] = lx;
    this.lastSnapInfo.pullDir[1] = ly;
    this.lastSnapInfo.pullDir[2] = lz;
    this.lastSnapInfo.normal[0] = spec.normal[0];
    this.lastSnapInfo.normal[1] = spec.normal[1];
    this.lastSnapInfo.normal[2] = spec.normal[2];
    this.lastSnapInfo.intensity = Math.min(1.0, (fromPull ? 0.72 : 0.48) + strainFactor * 0.35);

    // Crisp micro-speck talc burst (~44 fast bright chalk specks + ~8 quick-pop burst wisps <= 0.28s)
    const burstScale = 0.95 + strainFactor * 0.45;
    this.spawnDustPuffs(hitX, hitY, hitZ, radX, radY, radZ, tx, ty, tz, spec.width, 32, 6, burstScale);
    const wakeOffsets = [-0.075, 0.075] as const;
    for (let w = 0; w < wakeOffsets.length; w++) {
      const uWake = (((uNorm + wakeOffsets[w]) % 1) + 1) % 1;
      this.evalWrappedPoint(bandId, uWake, this.scratchPoint);
      const wLen = Math.hypot(this.scratchPoint[0], this.scratchPoint[1], this.scratchPoint[2]) || 1;
      const thW = uWake * TWO_PI;
      const wtx = -Math.sin(thW) * spec.tangentU[0] + Math.cos(thW) * spec.tangentV[0];
      const wty = -Math.sin(thW) * spec.tangentU[1] + Math.cos(thW) * spec.tangentV[1];
      const wtz = -Math.sin(thW) * spec.tangentU[2] + Math.cos(thW) * spec.tangentV[2];
      this.spawnDustPuffs(
        this.scratchPoint[0],
        this.scratchPoint[1],
        this.scratchPoint[2],
        this.scratchPoint[0] / wLen,
        this.scratchPoint[1] / wLen,
        this.scratchPoint[2] / wLen,
        wtx,
        wty,
        wtz,
        spec.width,
        6,
        1,
        0.85 * burstScale
      );
    }

    // Sympathetic elastic shockwave twang across local crossing bands near the snap site
    const maxWake = fromPull ? 6 : 3;
    let wokeCount = 0;
    for (let i = 0; i < TOTAL_BANDS && wokeCount < maxWake; i++) {
      if (i === bandId) continue;
      const nrt = this.runtimes[i];
      if (nrt.status !== 'intact') continue;
      const nspec = this.specs[i];
      const wasDyn = this.prevDynActive[i] === 1;
      if (Math.abs(nspec.layer - spec.layer) > 2 && !wasDyn) continue;
      const hitPlaneDist = Math.abs(
        hitX * nspec.normal[0] + hitY * nspec.normal[1] + hitZ * nspec.normal[2]
      );
      if (hitPlaneDist > nspec.width * 0.5 + (wasDyn ? 0.12 : 0.085)) continue;

      const align = Math.abs(
        nspec.normal[0] * spec.normal[0] +
          nspec.normal[1] * spec.normal[1] +
          nspec.normal[2] * spec.normal[2]
      );
      if (align > 0.2 || wasDyn) {
        wokeCount++;
        const amp = wasDyn
          ? Math.max(0.05, 0.075 * strainFactor)
          : (fromPull ? 0.04 + 0.04 * strainFactor : 0.025 + 0.025 * strainFactor) *
            Math.max(0.25, align);
        nrt.twangAmp = Math.max(nrt.twangAmp, amp);
        nrt.twangPhase = 0;
        const nu =
          hitX * nspec.tangentU[0] + hitY * nspec.tangentU[1] + hitZ * nspec.tangentU[2];
        const nv =
          hitX * nspec.tangentV[0] + hitY * nspec.tangentV[1] + hitZ * nspec.tangentV[2];
        let nAngle = Math.atan2(nv, nu);
        if (nAngle < 0) nAngle += TWO_PI;
        nrt.pullU = nAngle / TWO_PI;
        nrt.pullDir[0] = radX;
        nrt.pullDir[1] = radY;
        nrt.pullDir[2] = radZ;
      }
    }

    this.bandVersion[bandId]++;
    this.recomputeCompression(bandId);
  }

  private spawnDustPuffs(
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
    tx: number,
    ty: number,
    tz: number,
    bandWidth: number,
    speckCount: number,
    hazeCount: number,
    energyScale = 1.0
  ): void {
    const totalWanted = speckCount + hazeCount;
    let spawned = 0;
    for (let i = 0; i < MAX_DUST_PUFFS && spawned < totalWanted; i++) {
      const puff = this.dustPuffs[i];
      if (puff.active) continue;

      const isSpeck = spawned < speckCount;
      puff.active = true;
      puff.isSpeck = isSpeck;

      const spread = bandWidth * (isSpeck ? 0.9 : 1.25);
      puff.pos[0] = px + (this.rng() - 0.5) * spread;
      puff.pos[1] = py + (this.rng() - 0.5) * spread;
      puff.pos[2] = pz + (this.rng() - 0.5) * spread;

      const speed = (isSpeck ? 1.55 + this.rng() * 1.65 : 0.95 + this.rng() * 0.75) * energyScale;
      const tangSpray = (this.rng() - 0.5) * (isSpeck ? 1.9 : 1.1) * energyScale;
      const jitter = (isSpeck ? 1.25 : 0.85) * energyScale;
      puff.vel[0] = nx * speed + tx * tangSpray + (this.rng() - 0.5) * jitter;
      puff.vel[1] = ny * speed + ty * tangSpray + (this.rng() - 0.5) * jitter + 0.1;
      puff.vel[2] = nz * speed + tz * tangSpray + (this.rng() - 0.5) * jitter;

      puff.ageSec = 0;
      puff.maxAgeSec = isSpeck ? 0.18 + this.rng() * 0.10 : 0.12 + this.rng() * 0.08;
      puff.size = isSpeck ? 0.007 + this.rng() * 0.007 : 0.02 + this.rng() * 0.012;
      spawned++;
    }
  }

  public step(dtSec: number): void {
    const dt = Math.min(Math.max(dtSec, 0), 0.05);
    if (dt <= 0) return;
    this.simTimeSec += dt;

    // Damped spring recoil of the ball (leans toward active pull, recoils briskly on pop)
    const kSpring = 175.0;
    const kDamp = 11.5;
    const activePull =
      this.activePullBandId !== null ? this.runtimes[this.activePullBandId] : null;
    const leanAmt =
      activePull && activePull.status === 'pulled'
        ? Math.min(0.11, activePull.pullAmount * 0.16)
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

    // 2. Update band physics (breathing, twang, readable whip-back & spiral C-curl)
    const breathBlend = 1 - Math.exp(-12.0 * dt);
    const twangDecay = Math.exp(-TWANG_DAMP * dt);
    const arcDamp = Math.exp(-3.6 * dt);
    let needDynamicUpdate = false;

    for (let i = 0; i < TOTAL_BANDS; i++) {
      const rt = this.runtimes[i];
      const spec = this.specs[i];

      const breathDelta = this.targetBreath[i] - this.currentBreath[i];
      if (Math.abs(breathDelta) > 1e-5) {
        this.currentBreath[i] += breathDelta * breathBlend;
        if (Math.abs(this.targetBreath[i] - this.currentBreath[i]) <= 1e-5) {
          this.currentBreath[i] = this.targetBreath[i];
        }
        this.bandVersion[i]++;
      }

      if (rt.status === 'pulled') {
        needDynamicUpdate = true;
      } else if (rt.status === 'intact') {
        if (rt.twangAmp > 1e-4) {
          needDynamicUpdate = true;
          rt.twangPhase += dt * TWANG_FREQ;
          rt.twangAmp *= twangDecay;
          if (rt.twangAmp < 1e-4) {
            rt.twangAmp = 0;
            rt.currentStrain = spec.restStrain;
          } else {
            rt.currentStrain =
              spec.restStrain + Math.abs(Math.sin(rt.twangPhase)) * rt.twangAmp * 0.4;
          }
          this.bandVersion[i]++;
        }
      } else if (rt.status === 'snapping') {
        this.snapAgeSec[i] += dt;
        rt.leftVel *= arcDamp;
        rt.rightVel *= arcDamp;
        rt.leftArc = Math.max(0.085, rt.leftArc - rt.leftVel * dt);
        rt.rightArc = Math.max(0.085, rt.rightArc - rt.rightVel * dt);
        rt.curlAmount = Math.min(1, rt.curlAmount + dt * 9.0);

        if (this.snapAgeSec[i] > 0.04) {
          rt.fallVel[1] -= GRAVITY * dt;
        }
        rt.fallOffset[0] += rt.fallVel[0] * dt;
        rt.fallOffset[1] += rt.fallVel[1] * dt;
        rt.fallOffset[2] += rt.fallVel[2] * dt;

        rt.currentStrain = Math.max(0, rt.currentStrain - dt * 4.5);

        if (this.snapAgeSec[i] > 0.24 || rt.fallOffset[1] < -0.9) {
          rt.opacity = Math.max(0, rt.opacity - dt * 6.5);
        }

        if (rt.opacity <= 0.005) {
          rt.status = 'gone';
          rt.opacity = 0;
          rt.leftArc = 0;
          rt.rightArc = 0;
        }
        this.bandVersion[i]++;
      }
    }

    if (needDynamicUpdate) {
      this.recomputeDynamicDeformation();
    } else {
      // Clear prevDynActive on the settle tick when all pulls/twangs have stopped
      for (let i = 0; i < TOTAL_BANDS; i++) {
        if (this.prevDynActive[i]) {
          this.prevDynActive[i] = 0;
          this.dynActive[i] = 0;
          this.bandVersion[i]++;
        }
      }
    }

    // 3. Update talc specks & brief haze (crisp deceleration, rapid fade <= 0.28s)
    const speckDrag = Math.exp(-4.8 * dt);
    const hazeDrag = Math.exp(-8.5 * dt);
    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      const puff = this.dustPuffs[i];
      if (!puff.active) continue;

      puff.ageSec += dt;
      if (puff.ageSec >= puff.maxAgeSec) {
        puff.active = false;
        continue;
      }

      const d = puff.isSpeck ? speckDrag : hazeDrag;
      puff.vel[0] *= d;
      puff.vel[1] = puff.vel[1] * d - (puff.isSpeck ? 0.55 * dt : 0);
      puff.vel[2] *= d;

      puff.pos[0] += puff.vel[0] * dt;
      puff.pos[1] += puff.vel[1] * dt;
      puff.pos[2] += puff.vel[2] * dt;
    }
  }

  private evalWrappedPoint(bandIndex: number, u: number, out: [number, number, number]): void {
    const spec = this.specs[bandIndex];
    const uWrapped = ((u % 1) + 1) % 1;
    const kFloat = uWrapped * SEGMENTS_PER_BAND;
    const k0 = Math.floor(kFloat) % SEGMENTS_PER_BAND;
    const k1 = (k0 + 1) % SEGMENTS_PER_BAND;
    const frac = kFloat - Math.floor(kFloat);
    const profOffset = bandIndex * RINGS_PER_BAND;
    const breathShift = this.currentBreath[bandIndex] - this.targetBreath[bandIndex];
    const r =
      this.radialProfile[profOffset + k0] * (1 - frac) +
      this.radialProfile[profOffset + k1] * frac +
      breathShift;

    const theta = uWrapped * TWO_PI;
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
    const tu = spec.tangentU;
    const tv = spec.tangentV;
    const nj = spec.normal;
    const profOffset = bandIndex * RINGS_PER_BAND;
    const breathShift = this.currentBreath[bandIndex] - this.targetBreath[bandIndex];
    const minCoreR = CORE_RADIUS + CORE_CLEARANCE + spec.thickness * 0.5;

    if (rt.status === 'snapping') {
      const totalArc = rt.leftArc + rt.rightArc;
      if (totalArc < 0.015) return 0;

      const activeSegs = Math.max(4, Math.min(SEGMENTS_PER_BAND, Math.round(totalArc * SEGMENTS_PER_BAND)));
      const antipodeU = (rt.snapU + 0.5) % 1;
      const curl = rt.curlAmount;
      const age = this.snapAgeSec[bandIndex];
      const fx = rt.fallOffset[0];
      const fy = rt.fallOffset[1];
      const fz = rt.fallOffset[2];

      const peakPull = this.snapPeakPull[bandIndex];
      const crownEnv = peakPull > 0 ? peakPull * Math.exp(-15.0 * age) : 0;
      const crownHalfArc = this.snapHalfArc[bandIndex];
      const lx = this.snapLaunchDir[bandIndex * 3];
      const ly = this.snapLaunchDir[bandIndex * 3 + 1];
      const lz = this.snapLaunchDir[bandIndex * 3 + 2];

      const rAvg = 0.5 * (this.minBandRadius[bandIndex] + this.maxBandRadius[bandIndex]) + breathShift;
      const minClearR = Math.max(minCoreR, spec.baseRadius + spec.thickness * 0.5);
      const lashEnv = Math.exp(-7.2 * age);

      // Fast outward peel & 3D oblique tumble once severed so the coiled scrap springs clear of the ball
      const peelBase = Math.min(0.24, age * 2.2);
      const tumbleBlend = age > 0.045 ? Math.min(1, (age - 0.045) / 0.16) : 0;
      const tumbleAngle = this.snapSideSign[bandIndex] * tumbleBlend * Math.max(0, age - 0.045) * 9.5;
      const cosT = Math.cos(tumbleAngle);
      const sinT = Math.sin(tumbleAngle);

      const thAnt = antipodeU * TWO_PI;
      const cAnt = Math.cos(thAnt);
      const sAnt = Math.sin(thAnt);
      const antRx = cAnt * tu[0] + sAnt * tv[0];
      const antRy = cAnt * tu[1] + sAnt * tv[1];
      const antRz = cAnt * tu[2] + sAnt * tv[2];
      const antCenterR = rAvg + peelBase * 0.38;
      const antCx = antCenterR * antRx;
      const antCy = antCenterR * antRy;
      const antCz = antCenterR * antRz;

      const tax = antRx;
      const tay = antRy;
      const taz = antRz;

      const hingeDist = 0.56;
      const uHingeLeft = antipodeU - rt.leftArc * hingeDist;
      const uHingeRight = antipodeU + rt.rightArc * hingeDist;
      const thHL = uHingeLeft * TWO_PI;
      const thHR = uHingeRight * TWO_PI;
      const cHL = Math.cos(thHL);
      const sHL = Math.sin(thHL);
      const cHR = Math.cos(thHR);
      const sHR = Math.sin(thHR);

      // Evaluate exact wrapped radius at both hinge anchors for C^0 continuous spiral attachment
      const uHLW = ((uHingeLeft % 1) + 1) % 1;
      const kHLF = uHLW * SEGMENTS_PER_BAND;
      const kHL0 = Math.floor(kHLF) % SEGMENTS_PER_BAND;
      const kHL1 = (kHL0 + 1) % SEGMENTS_PER_BAND;
      const fHL = kHLF - Math.floor(kHLF);
      const rWrapHL =
        this.radialProfile[profOffset + kHL0] * (1 - fHL) +
        this.radialProfile[profOffset + kHL1] * fHL +
        breathShift;

      const uHRW = ((uHingeRight % 1) + 1) % 1;
      const kHRF = uHRW * SEGMENTS_PER_BAND;
      const kHR0 = Math.floor(kHRF) % SEGMENTS_PER_BAND;
      const kHR1 = (kHR0 + 1) % SEGMENTS_PER_BAND;
      const fHR = kHRF - Math.floor(kHRF);
      const rWrapHR =
        this.radialProfile[profOffset + kHR0] * (1 - fHR) +
        this.radialProfile[profOffset + kHR1] * fHR +
        breathShift;

      const hingeLiftFrac = 0.38 + 0.62 * hingeDist * hingeDist;
      const hingeWhip =
        0.11 *
        hingeDist *
        lashEnv *
        (0.4 + 0.6 * Math.max(0, Math.sin(hingeDist * 8.5 - age * 38.0)));
      const rHL = Math.max(minClearR, rWrapHL) + peelBase * hingeLiftFrac + hingeWhip;
      const rHR = Math.max(minClearR, rWrapHR) + peelBase * hingeLiftFrac + hingeWhip;

      const erLx = cHL * tu[0] + sHL * tv[0];
      const erLy = cHL * tu[1] + sHL * tv[1];
      const erLz = cHL * tu[2] + sHL * tv[2];
      const etLx = -(-sHL * tu[0] + cHL * tv[0]);
      const etLy = -(-sHL * tu[1] + cHL * tv[1]);
      const etLz = -(-sHL * tu[2] + cHL * tv[2]);

      const erRx = cHR * tu[0] + sHR * tv[0];
      const erRy = cHR * tu[1] + sHR * tv[1];
      const erRz = cHR * tu[2] + sHR * tv[2];
      const etRx = -sHR * tu[0] + cHR * tv[0];
      const etRy = -sHR * tu[1] + cHR * tv[1];
      const etRz = -sHR * tu[2] + cHR * tv[2];

      const totalPhi = 4.4; // ~252 deg tight 3D helical C-loop
      const spiralR0 = 0.042;

      for (let k = 0; k <= activeSegs; k++) {
        const frac = k / activeSegs;
        const uOffset = -rt.leftArc + frac * totalArc;
        const u = antipodeU + uOffset;
        const uWrapped = ((u % 1) + 1) % 1;
        const theta = u * TWO_PI;
        const c = Math.cos(theta);
        const s = Math.sin(theta);

        // Interpolate wrapped base radius along u so the middle hugs the ball accurately
        const kF = uWrapped * SEGMENTS_PER_BAND;
        const k0 = Math.floor(kF) % SEGMENTS_PER_BAND;
        const k1 = (k0 + 1) % SEGMENTS_PER_BAND;
        const kf = kF - Math.floor(kF);
        const rWrap =
          this.radialProfile[profOffset + k0] * (1 - kf) +
          this.radialProfile[profOffset + k1] * kf +
          breathShift;

        const endDist = Math.abs(frac - 0.5) * 2;
        const whipLash =
          0.11 *
          endDist *
          lashEnv *
          (0.4 + 0.6 * Math.max(0, Math.sin(endDist * 8.5 - age * 38.0)));
        const liftFrac = 0.38 + 0.62 * endDist * endDist;
        const radCircle = Math.max(minClearR, rWrap) + peelBase * liftFrac + whipLash;

        let cx = radCircle * (c * tu[0] + s * tv[0]);
        let cy = radCircle * (c * tu[1] + s * tv[1]);
        let cz = radCircle * (c * tu[2] + s * tv[2]);

        // Smoothly curl free tips (> 56% toward tip) into a tight 3D helical C-loop spiral
        if (endDist > hingeDist && curl > 1e-4) {
          const tipS = (endDist - hingeDist) / (1 - hingeDist);
          const phi = totalPhi * Math.pow(tipS, 1.08);
          const rho = spiralR0 * (1 - 0.32 * tipS);
          const dsTan = rho * Math.sin(phi);
          const dsRad = rho * (1 - Math.cos(phi));
          const sideSign = frac >= 0.5 ? 1 : -1;
          const dsLat = sideSign * 0.036 * Math.sin(phi * 1.15);

          const isRight = frac >= 0.5;
          const hx = isRight ? rHR * erRx : rHL * erLx;
          const hy = isRight ? rHR * erRy : rHL * erLy;
          const hz = isRight ? rHR * erRz : rHL * erLz;

          const erx = isRight ? erRx : erLx;
          const ery = isRight ? erRy : erLy;
          const erz = isRight ? erRz : erLz;
          const etx = isRight ? etRx : etLx;
          const ety = isRight ? etRy : etLy;
          const etz = isRight ? etRz : etLz;

          const coilX = hx + dsTan * etx + dsRad * erx + dsLat * nj[0];
          const coilY = hy + dsTan * ety + dsRad * ery + dsLat * nj[1];
          const coilZ = hz + dsTan * etz + dsRad * erz + dsLat * nj[2];

          cx = cx + (coilX - cx) * curl;
          cy = cy + (coilY - cy) * curl;
          cz = cz + (coilZ - cz) * curl;
        }

        // Preserve stretched crown into initial whip-down so break starts right at cursor apex
        if (crownEnv > 1e-4) {
          const dSnap = shortestArcDist(uWrapped, rt.snapU);
          const crownWeight = evalPullArch(dSnap, crownHalfArc);
          const crownLift = crownEnv * crownWeight;
          cx += lx * crownLift;
          cy += ly * crownLift;
          cz += lz * crownLift;
        }

        // 3D oblique tumble around the coiled scrap's own center while springing away from the ball
        if (tumbleBlend > 1e-4) {
          const dx = cx - antCx;
          const dy = cy - antCy;
          const dz = cz - antCz;
          const aDotD = (tax * dx + tay * dy + taz * dz) * (1 - cosT);
          const crx = tay * dz - taz * dy;
          const cry = taz * dx - tax * dz;
          const crz = tax * dy - tay * dx;
          cx = antCx + dx * cosT + crx * sinT + tax * aDotD;
          cy = antCy + dy * cosT + cry * sinT + tay * aDotD;
          cz = antCz + dz * cosT + crz * sinT + taz * aDotD;
        }

        let px = cx + fx;
        let py = cy + fy;
        let pz = cz + fz;

        // Enforce non-penetration against ball sphere
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
        outStrains[k] = rt.currentStrain * (1 - endDist * 0.55);
      }

      return activeSegs;
    }

    // Intact or pulled: read directly from coupled dynamic buffers if active
    if (this.dynActive[bandIndex] === 1) {
      const ptsOffset = profOffset * 3;
      outPositions.set(this.dynPts.subarray(ptsOffset, ptsOffset + RINGS_PER_BAND * 3));
      outStrains.set(this.dynStrains.subarray(profOffset, profOffset + RINGS_PER_BAND));
      return SEGMENTS_PER_BAND;
    }

    // Fast path for stationary intact band
    const ptsOffset = profOffset * 3;
    const strain = rt.currentStrain;
    for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
      const rLayer = Math.max(minCoreR, this.radialProfile[profOffset + k] + breathShift);
      const pIdx = ptsOffset + k * 3;
      const base = k * 3;
      outPositions[base] = rLayer * this.unitCirclePts[pIdx];
      outPositions[base + 1] = rLayer * this.unitCirclePts[pIdx + 1];
      outPositions[base + 2] = rLayer * this.unitCirclePts[pIdx + 2];
      outStrains[k] = strain;
    }
    return SEGMENTS_PER_BAND;
  }

  /**
   * Raycasts against actual 3D wrapped radialProfile ribbon geometry (centerlines + width + thickness).
   * Pass 1 tests exact physical strap bounds so clicks always grab the true topmost visible band;
   * Pass 2 adds small silhouette edge tolerance only if pass 1 missed.
   */
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
    const boundR = this.ballMaxOuterRadius + 0.08;
    if (bHalf * bHalf - (oDotO - boundR * boundR) < 0) {
      return null;
    }

    const exactHit = this.raycastPass(ox, oy, oz, dx, dy, dz, bHalf, oDotO, 0.0);
    if (exactHit !== null) return exactHit;

    return this.raycastPass(ox, oy, oz, dx, dy, dz, bHalf, oDotO, 0.0055);
  }

  private raycastPass(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    bHalf: number,
    oDotO: number,
    hitSlop: number
  ): { bandId: number; uHit: number; hitPoint: [number, number, number] } | null {
    let bestT = Infinity;
    let bestBandId = -1;
    let bestU = 0;

    for (let i = TOTAL_BANDS - 1; i >= 0; i--) {
      const rt = this.runtimes[i];
      if (rt.status !== 'intact') continue;

      const spec = this.specs[i];
      const halfW = spec.width * 0.5 + hitSlop;
      const halfH = spec.thickness * 0.5 + hitSlop * 0.3;
      const breathShift = this.currentBreath[i] - this.targetBreath[i];
      const maxOuterR = Math.hypot(this.maxBandRadius[i] + breathShift + halfH, halfW);

      // 1. Band bounding-sphere interval [tS0, tS1]
      const disc = bHalf * bHalf - (oDotO - maxOuterR * maxOuterR);
      if (disc < 0) continue;
      const sqrtDisc = Math.sqrt(disc);
      const tS0 = -bHalf - sqrtDisc;
      const tS1 = -bHalf + sqrtDisc;
      if (tS1 <= 0 || tS0 >= bestT) continue;

      // 2. Band plane-slab interval |((o + t*d) · n)| <= halfW
      const nj = spec.normal;
      const on = ox * nj[0] + oy * nj[1] + oz * nj[2];
      const dn = dx * nj[0] + dy * nj[1] + dz * nj[2];
      let tEnter = tS0 > 0 ? tS0 : 0;
      let tExit = tS1 < bestT ? tS1 : bestT;
      if (Math.abs(dn) > 1e-6) {
        const tA = (-halfW - on) / dn;
        const tB = (halfW - on) / dn;
        const tN0 = tA < tB ? tA : tB;
        const tN1 = tA < tB ? tB : tA;
        if (tN0 > tEnter) tEnter = tN0;
        if (tN1 < tExit) tExit = tN1;
      } else if (Math.abs(on) > halfW) {
        continue;
      }
      if (tEnter > tExit) continue;

      // 3. Ray-OBB test against wrapped ribbon segments along radialProfile[i]
      const profOffset = i * RINGS_PER_BAND;
      const ptsOffset = profOffset * 3;
      const useDyn = this.dynActive[i] === 1;
      let prevBx = nj[0];
      let prevBy = nj[1];
      let prevBz = nj[2];

      for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
        let ax: number;
        let ay: number;
        let az: number;
        let bx: number;
        let by: number;
        let bz: number;
        if (useDyn) {
          const p0 = ptsOffset + k * 3;
          const p1 = ptsOffset + (k + 1) * 3;
          ax = this.dynPts[p0];
          ay = this.dynPts[p0 + 1];
          az = this.dynPts[p0 + 2];
          bx = this.dynPts[p1];
          by = this.dynPts[p1 + 1];
          bz = this.dynPts[p1 + 2];
        } else {
          const r0 = this.radialProfile[profOffset + k] + breathShift;
          const r1 = this.radialProfile[profOffset + k + 1] + breathShift;
          const p0 = ptsOffset + k * 3;
          const p1 = ptsOffset + (k + 1) * 3;
          ax = r0 * this.unitCirclePts[p0];
          ay = r0 * this.unitCirclePts[p0 + 1];
          az = r0 * this.unitCirclePts[p0 + 2];
          bx = r1 * this.unitCirclePts[p1];
          by = r1 * this.unitCirclePts[p1 + 1];
          bz = r1 * this.unitCirclePts[p1 + 2];
        }

        const mx = 0.5 * (ax + bx);
        const my = 0.5 * (ay + by);
        const mz = 0.5 * (az + bz);

        // Segment tangent axis T
        let tx = bx - ax;
        let ty = by - ay;
        let tz = bz - az;
        const segLen = Math.hypot(tx, ty, tz);
        if (segLen < 1e-6) continue;
        const halfL = segLen * 0.5 + 0.0005;
        tx /= segLen;
        ty /= segLen;
        tz /= segLen;

        // Wrap-plane-anchored binormal matching rubberOptics / buildBandOBB
        const nDotT = nj[0] * tx + nj[1] * ty + nj[2] * tz;
        let binX = nj[0] - nDotT * tx;
        let binY = nj[1] - nDotT * ty;
        let binZ = nj[2] - nDotT * tz;
        let binLen = Math.hypot(binX, binY, binZ);
        if (binLen < 0.15) {
          const pDotT = prevBx * tx + prevBy * ty + prevBz * tz;
          binX = prevBx - pDotT * tx;
          binY = prevBy - pDotT * ty;
          binZ = prevBz - pDotT * tz;
          binLen = Math.hypot(binX, binY, binZ);
        }
        if (binLen < 1e-4) {
          binX = my * tz - mz * ty;
          binY = mz * tx - mx * tz;
          binZ = mx * ty - my * tx;
          binLen = Math.hypot(binX, binY, binZ) || 1;
        }
        binX /= binLen;
        binY /= binLen;
        binZ /= binLen;
        prevBx = binX;
        prevBy = binY;
        prevBz = binZ;

        // Radial axis R = T x B
        let radX = ty * binZ - tz * binY;
        let radY = tz * binX - tx * binZ;
        let radZ = tx * binY - ty * binX;
        const radLen = Math.hypot(radX, radY, radZ) || 1;
        radX /= radLen;
        radY /= radLen;
        radZ /= radLen;

        const omx = ox - mx;
        const omy = oy - my;
        const omz = oz - mz;

        let tMin = 0;
        let tMax = bestT;

        // Slab along T
        const oT = omx * tx + omy * ty + omz * tz;
        const dT = dx * tx + dy * ty + dz * tz;
        if (Math.abs(dT) < 1e-8) {
          if (Math.abs(oT) > halfL) continue;
        } else {
          let t1 = (-halfL - oT) / dT;
          let t2 = (halfL - oT) / dT;
          if (t1 > t2) {
            const tmp = t1;
            t1 = t2;
            t2 = tmp;
          }
          if (t1 > tMin) tMin = t1;
          if (t2 < tMax) tMax = t2;
          if (tMin > tMax) continue;
        }

        // Slab along B
        const oB = omx * binX + omy * binY + omz * binZ;
        const dB = dx * binX + dy * binY + dz * binZ;
        if (Math.abs(dB) < 1e-8) {
          if (Math.abs(oB) > halfW) continue;
        } else {
          let t1 = (-halfW - oB) / dB;
          let t2 = (halfW - oB) / dB;
          if (t1 > t2) {
            const tmp = t1;
            t1 = t2;
            t2 = tmp;
          }
          if (t1 > tMin) tMin = t1;
          if (t2 < tMax) tMax = t2;
          if (tMin > tMax) continue;
        }

        // Slab along R
        const oR = omx * radX + omy * radY + omz * radZ;
        const dR = dx * radX + dy * radY + dz * radZ;
        if (Math.abs(dR) < 1e-8) {
          if (Math.abs(oR) > halfH) continue;
        } else {
          let t1 = (-halfH - oR) / dR;
          let t2 = (halfH - oR) / dR;
          if (t1 > t2) {
            const tmp = t1;
            t1 = t2;
            t2 = tmp;
          }
          if (t1 > tMin) tMin = t1;
          if (t2 < tMax) tMax = t2;
          if (tMin > tMax) continue;
        }

        if (tMin > 0 && tMin < bestT) {
          bestT = tMin;
          bestBandId = i;
          const hx = ox + tMin * dx;
          const hy = oy + tMin * dy;
          const hz = oz + tMin * dz;
          const pu = hx * spec.tangentU[0] + hy * spec.tangentU[1] + hz * spec.tangentU[2];
          const pv = hx * spec.tangentV[0] + hy * spec.tangentV[1] + hz * spec.tangentV[2];
          const angle = Math.atan2(pv, pu);
          bestU = (angle >= 0 ? angle : angle + TWO_PI) / TWO_PI;
        }
      }
    }

    if (bestBandId === -1) return null;

    return {
      bandId: bestBandId,
      uHit: bestU,
      hitPoint: [ox + bestT * dx, oy + bestT * dy, oz + bestT * dz],
    };
  }

  public getBandVersion(bandIndex: number): number {
    if (bandIndex < 0 || bandIndex >= TOTAL_BANDS) return 0;
    return this.bandVersion[bandIndex];
  }

  public getLastSnapInfo(): Readonly<SnapImpulseInfo> | null {
    return this.hasLastSnap ? this.lastSnapInfo : null;
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
        const breathShift = this.currentBreath[i] - this.targetBreath[i];
        const bandOuter = this.maxBandRadius[i] + breathShift;
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
