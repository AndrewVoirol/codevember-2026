import * as THREE from 'three/webgpu';
import { MINERAL_PALETTE, type PigmentKey } from '../../core/palette';

/**
 * Day 30 — Dissolve:
 * Pure TypeScript physical simulation for effervescent mineral-pigment cubes dissolving
 * inside a suspended thick-walled borosilicate laboratory beaker.
 *
 * Physical Systems Coupled Here (Zero Per-Frame Heap Allocations):
 *  1. Curvature-Weighted Crystal Dissolution & Exact Volume-Conserving Corner Rounding
 *     (Gibbs-Thomson / Ostwald-Freundlich corner exposure + 3D etching pits + buoyant wafer sliver).
 *  2. Exact Solute Mass Conservation (`totalSolidMass + bathSoluteMass === totalInjectedMass`)
 *     with per-pigment Beer-Lambert optical extinction & reflectance mixing across up to 4 cubes.
 *  3. Rankine/Lamb-Oseen Vortex Swirl Field (`omega(r)`), Secondary Ekman Circulation,
 *     and Slender Borosilicate Glass Stirring Rod Coupling.
 *  4. 2D Circular Gravity-Capillary Free-Surface Wave PDE (`48x48` HalfFloat DataTexture)
 *     driven by cube entry splashes, stirring rod wakes, and popping effervescent bubbles.
 *  5. 2,000-Instance Rising Effervescent Silver Micro-Bubble Pool nucleating directly off
 *     the live eroded facets of submerged dissolving crystals.
 */

// ---------------------------------------------------------------------------
// Shared Physical Apparatus Constants (Griffin Low-Form Laboratory Beaker)
// ---------------------------------------------------------------------------
export const BEAKER_OUTER_RADIUS = 0.78;
export const BEAKER_WALL_THICKNESS = 0.042;
export const BEAKER_INNER_RADIUS = BEAKER_OUTER_RADIUS - BEAKER_WALL_THICKNESS; // 0.738
export const BEAKER_BOTTOM_OUTER_Y = -0.84;
export const BEAKER_BASE_THICKNESS = 0.115;
export const BEAKER_INNER_FLOOR_Y = BEAKER_BOTTOM_OUTER_Y + BEAKER_BASE_THICKNESS; // -0.725
export const BEAKER_TOP_Y = 0.90;

export const WATER_FILL_FRACTION = 0.78;
export const WATER_REST_Y =
  BEAKER_INNER_FLOOR_Y + (BEAKER_TOP_Y - BEAKER_INNER_FLOOR_Y) * WATER_FILL_FRACTION; // +0.5425
export const WATER_RADIUS = BEAKER_INNER_RADIUS - 0.004; // 0.734

export const CUBE_INITIAL_HALF = 0.178;
export const CUBE_UNIT_MASS = 1.0;
export const MAX_ACTIVE_CUBES = 4;
export const MAX_SWIRL_OMEGA = 10.5;

export const BUBBLE_COUNT = 2000;
export const CRUMB_COUNT = 512;
export const WAVE_RES = 48;
const WAVE_CELLS = WAVE_RES * WAVE_RES;

export const DISSOLVE_PIGMENT_CYCLE: readonly PigmentKey[] = [
  'celadonGlauconite',
  'realgarAmber',
  'azuritePrussian',
  'cinnabarVermilion',
  'yellowOchre',
] as const;

const SQRT2 = Math.SQRT2;
const SQRT3 = Math.sqrt(3);
const FILLET_FRAC_INITIAL = 0.055;
const FILLET_FRAC_FINAL = 0.96;

/** Exact rounded-cube dimensionless volume coefficient V / H^3 as a function of fillet fraction f_c. */
function roundedBoxVolumeCoeff(fc: number): number {
  const omf = 1.0 - fc;
  return (
    8.0 * omf * omf * omf +
    24.0 * omf * omf * fc +
    6.0 * Math.PI * omf * fc * fc +
    (4.0 / 3.0) * Math.PI * fc * fc * fc
  );
}

const PHI_INITIAL = roundedBoxVolumeCoeff(FILLET_FRAC_INITIAL);

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / Math.max(1e-6, edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Deterministic hash & 3D irrational-angle FBM (matching TSL shader contract; zero star harmonics)
function fractCpu(x: number): number {
  return x - Math.floor(x);
}

function hash31Cpu(x: number, y: number, z: number): number {
  let px = fractCpu(x * 0.1031 + 1.0);
  let py = fractCpu(y * 0.103 + 1.0);
  let pz = fractCpu(z * 0.0973 + 1.0);
  const d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33);
  px += d;
  py += d;
  pz += d;
  return fractCpu((px + py) * pz);
}

function valueNoise3DCpu(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3.0 - 2.0 * fx);
  const uy = fy * fy * (3.0 - 2.0 * fy);
  const uz = fz * fz * (3.0 - 2.0 * fz);

  const n000 = hash31Cpu(ix, iy, iz);
  const n100 = hash31Cpu(ix + 1, iy, iz);
  const n010 = hash31Cpu(ix, iy + 1, iz);
  const n110 = hash31Cpu(ix + 1, iy + 1, iz);
  const n001 = hash31Cpu(ix, iy, iz + 1);
  const n101 = hash31Cpu(ix + 1, iy, iz + 1);
  const n011 = hash31Cpu(ix, iy + 1, iz + 1);
  const n111 = hash31Cpu(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

export function mineralPittingFbmCpu(x: number, y: number, z: number): number {
  const c = 0.80901699;
  const s = 0.58778525;
  const v0 = valueNoise3DCpu(x, y, z);
  const x1 = (x * c - z * s) * 2.03;
  const y1 = (y * c + x * s) * 2.03;
  const z1 = (z * c + y * s) * 2.03;
  const v1 = valueNoise3DCpu(x1, y1, z1);
  const x2 = (y1 * c - x1 * s) * 2.07;
  const y2 = (z1 * c + y1 * s) * 2.07;
  const z2 = (x1 * c + z1 * s) * 2.07;
  const v2 = valueNoise3DCpu(x2, y2, z2);
  return v0 * 0.57 + v1 * 0.29 + v2 * 0.14;
}

// ---------------------------------------------------------------------------
// Curvature-Weighted Crystal Erosion Model (Shared Analytically with TSL)
// ---------------------------------------------------------------------------
export interface CrystalErosionMetrics {
  mass: number;
  progress: number;
  halfExtent: number;
  coreHalf: number;
  filletRadius: number;
  filletFraction: number;
  scaleY: number;
  scaleXZ: number;
  pitAmp: number;
  /** Surface radius at flat face center (1, 0, 0) */
  faceRadius: number;
  /** Surface radius at edge midpoint (1, 1, 0) */
  edgeRadius: number;
  /** Surface radius at sharp corner (1, 1, 1) */
  cornerRadius: number;
  /** Absolute inward erosion at flat face center since m = 1 */
  faceErosion: number;
  /** Absolute inward erosion at corner since m = 1 */
  cornerErosion: number;
  /** Fractional erosion [0..1] at flat face center */
  faceErosionFraction: number;
  /** Fractional erosion [0..1] at corner (strictly > faceErosionFraction for all 0 < m < 1) */
  cornerErosionFraction: number;
  /** Ratio cornerRadius / faceRadius (drops strictly from ~1.692 toward ~1.02) */
  cornerToFaceRatio: number;
}

const INITIAL_FACE_RADIUS = CUBE_INITIAL_HALF;
const INITIAL_CORNER_RADIUS =
  CUBE_INITIAL_HALF * (SQRT3 * (1.0 - FILLET_FRAC_INITIAL) + FILLET_FRAC_INITIAL);

export function evalCrystalErosionMetrics(massInput: number): CrystalErosionMetrics {
  const mass = clamp(massInput, 0.0, 1.0);
  const progress = 1.0 - mass;

  if (mass <= 1e-6) {
    return {
      mass: 0,
      progress: 1,
      halfExtent: 0,
      coreHalf: 0,
      filletRadius: 0,
      filletFraction: FILLET_FRAC_FINAL,
      scaleY: 0.68,
      scaleXZ: 1.0 / Math.sqrt(0.68),
      pitAmp: 0,
      faceRadius: 0,
      edgeRadius: 0,
      cornerRadius: 0,
      faceErosion: INITIAL_FACE_RADIUS,
      cornerErosion: INITIAL_CORNER_RADIUS,
      faceErosionFraction: 1,
      cornerErosionFraction: 1,
      cornerToFaceRatio: 1,
    };
  }

  // Corners have maximum local curvature exposure -> round rapidly early in dissolution
  const roundRamp = 1.0 - Math.pow(Math.max(0.0, 1.0 - progress), 1.65);
  const filletFraction =
    FILLET_FRAC_INITIAL + (FILLET_FRAC_FINAL - FILLET_FRAC_INITIAL) * roundRamp;
  const phi = roundedBoxVolumeCoeff(filletFraction);

  // Exact volume-conserving half-extent H(m) so geometric volume tracks remaining crystal mass
  const halfExtent = CUBE_INITIAL_HALF * Math.cbrt((mass * PHI_INITIAL) / phi);
  const filletRadius = halfExtent * filletFraction;
  const coreHalf = halfExtent - filletRadius;

  // Late-stage gentle buoyant sliver flattening (strictly volume-preserving: scaleXZ^2 * scaleY == 1)
  const sliverT = smoothstepCpu(0.62, 0.98, progress);
  const scaleY = 1.0 - 0.30 * sliverT;
  const scaleXZ = 1.0 / Math.sqrt(scaleY);

  // Dissolution etching pits deepen sharply across early/mid-dissolution so notches & cavities bite visibly
  const pitEnv = Math.sin(Math.PI * clamp(progress / 0.96, 0.0, 1.0));
  const pitAmp = 0.175 * pitEnv * halfExtent;

  // Radial extents before anisotropic sliver flattening (for clean isotropic corner-vs-face measure)
  const faceRadius = halfExtent;
  const edgeRadius = SQRT2 * coreHalf + filletRadius;
  const cornerRadius = SQRT3 * coreHalf + filletRadius;

  const faceErosion = INITIAL_FACE_RADIUS - faceRadius;
  const cornerErosion = INITIAL_CORNER_RADIUS - cornerRadius;
  const faceErosionFraction = faceErosion / INITIAL_FACE_RADIUS;
  const cornerErosionFraction = cornerErosion / INITIAL_CORNER_RADIUS;

  return {
    mass,
    progress,
    halfExtent,
    coreHalf,
    filletRadius,
    filletFraction,
    scaleY,
    scaleXZ,
    pitAmp,
    faceRadius,
    edgeRadius,
    cornerRadius,
    faceErosion,
    cornerErosion,
    faceErosionFraction,
    cornerErosionFraction,
    cornerToFaceRatio: cornerRadius / Math.max(1e-6, faceRadius),
  };
}

/**
 * Evaluates a deformed 3D crystal surface point & outward direction for a unit-cube vertex
 * `(ux, uy, uz) in [-1, 1]^3`, matching the exact TSL GPU vertex morph.
 */
export function evalCrystalSurfacePoint(
  ux: number,
  uy: number,
  uz: number,
  mass: number,
  seed: number,
  outPos: THREE.Vector3,
  outNormal?: THREE.Vector3
): THREE.Vector3 {
  const m = evalCrystalErosionMetrics(mass);
  if (m.halfExtent <= 1e-6) {
    outPos.set(0, 0, 0);
    if (outNormal) outNormal.set(0, 1, 0);
    return outPos;
  }

  const px = ux * m.halfExtent;
  const py = uy * m.halfExtent;
  const pz = uz * m.halfExtent;

  const qx = clamp(px, -m.coreHalf, m.coreHalf);
  const qy = clamp(py, -m.coreHalf, m.coreHalf);
  const qz = clamp(pz, -m.coreHalf, m.coreHalf);

  let dx = px - qx;
  let dy = py - qy;
  let dz = pz - qz;
  const dLen = Math.hypot(dx, dy, dz);

  let rx = px;
  let ry = py;
  let rz = pz;
  let nx = 0;
  let ny = 1;
  let nz = 0;

  if (dLen > 1e-6) {
    const inv = 1.0 / dLen;
    nx = dx * inv;
    ny = dy * inv;
    nz = dz * inv;
    rx = qx + nx * m.filletRadius;
    ry = qy + ny * m.filletRadius;
    rz = qz + nz * m.filletRadius;
  } else {
    const ax = Math.abs(ux);
    const ay = Math.abs(uy);
    const az = Math.abs(uz);
    if (ax >= ay && ax >= az) {
      nx = Math.sign(ux) || 1;
      ny = 0;
      nz = 0;
    } else if (ay >= ax && ay >= az) {
      nx = 0;
      ny = Math.sign(uy) || 1;
      nz = 0;
    } else {
      nx = 0;
      ny = 0;
      nz = Math.sign(uz) || 1;
    }
  }

  // High-curvature corner/edge exposure weighting for deeper dissolution pitting along edges
  const uLen = Math.hypot(ux, uy, uz);
  const curvatureExposure = clamp((uLen - 1.0) / (SQRT3 - 1.0), 0.0, 1.0);
  const fbm = mineralPittingFbmCpu(
    ux * 3.1 + seed * 4.7,
    uy * 3.1 + seed * 2.3,
    uz * 3.1 + seed * 6.1
  );
  const pitDepth = m.pitAmp * smoothstepCpu(0.32, 0.78, fbm) * (0.55 + 0.45 * curvatureExposure);

  rx = (rx - nx * pitDepth) * m.scaleXZ;
  ry = (ry - ny * pitDepth) * m.scaleY;
  rz = (rz - nz * pitDepth) * m.scaleXZ;

  outPos.set(rx, ry, rz);
  if (outNormal) {
    outNormal.set(nx / m.scaleXZ, ny / m.scaleY, nz / m.scaleXZ).normalize();
  }
  return outPos;
}

// ---------------------------------------------------------------------------
// Data Interfaces
// ---------------------------------------------------------------------------
export interface DissolvingCubeState {
  id: number;
  slotIndex: number;
  active: boolean;
  pigmentIndex: number;
  pigmentKey: PigmentKey;
  seed: number;
  initialMass: number;
  mass: number;
  dissolvedMass: number;
  submerged: boolean;
  splashTriggered: boolean;
  ageSec: number;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  quat: THREE.Quaternion;
  angVel: THREE.Vector3;
  buoyantLift: number;
  lastDissolveRate: number;
}

export interface StirringRodState {
  active: boolean;
  /** Smooth immersion blend [0 = lifted above rim, 1 = dipped inside water column] */
  dip: number;
  x: number;
  z: number;
  vx: number;
  vz: number;
  tiltX: number;
  tiltZ: number;
}

export interface Day30TelemetryState {
  activeCubeCount: number;
  totalDissolvedCount: number;
  meanCubeMass: number;
  activeBubbles: number;
  activeCrumbs: number;
  swirlAngularVel: number;
  bathConcentration: number;
}

// Module-scope scratch math objects (strictly zero per-frame allocations)
const _tmpQuat = new THREE.Quaternion();
const _tmpAxis = new THREE.Vector3();
const _tmpLocalPt = new THREE.Vector3();
const _tmpLocalNorm = new THREE.Vector3();
const _tmpWorldPt = new THREE.Vector3();
const _tmpWorldNorm = new THREE.Vector3();
const _tmpUp = new THREE.Vector3(0, 1, 0);
const _tmpBodyAxis = new THREE.Vector3();

// ---------------------------------------------------------------------------
// Main Pure Physical Dissolution, Swirl, Wave & Bubble Simulation Class
// ---------------------------------------------------------------------------
export class BeakerDissolveSim {
  readonly cubes: DissolvingCubeState[] = [];
  readonly rod: StirringRodState = {
    active: false,
    dip: 0,
    x: 0.24,
    z: 0.14,
    vx: 0,
    vz: 0,
    tiltX: 0.18,
    tiltZ: -0.14,
  };

  // Exact solute mass conservation ledger
  totalInjectedMass = 0;
  bathSoluteMass = 0;
  totalDissolvedCount = 0;
  readonly pigmentSoluteMass = new Float64Array(DISSOLVE_PIGMENT_CYCLE.length);
  readonly bathExtinctionRgb = new THREE.Vector3(0.08, 0.06, 0.05);
  readonly bathTintRgb = new THREE.Vector3(0.76, 0.84, 0.86);

  // Hydrodynamic vortex & secondary Ekman circulation
  swirlAngularVel = 0;
  ekmanStrength = 0;
  simTime = 0;

  // 2D Circular Gravity-Capillary Free-Surface Wave PDE (`48x48` HalfFloat)
  readonly waveCurr = new Float32Array(WAVE_CELLS);
  readonly wavePrev = new Float32Array(WAVE_CELLS);
  readonly waveNext = new Float32Array(WAVE_CELLS);
  readonly waveInside = new Uint8Array(WAVE_CELLS);
  readonly waveTexData = new Uint16Array(WAVE_CELLS * 4);
  readonly waveTexture: THREE.DataTexture;

  // Rising Effervescent Silver Micro-Bubble Pool (SoA, zero GC)
  readonly bubbleAlive = new Uint8Array(BUBBLE_COUNT);
  readonly bubblePosX = new Float32Array(BUBBLE_COUNT);
  readonly bubblePosY = new Float32Array(BUBBLE_COUNT);
  readonly bubblePosZ = new Float32Array(BUBBLE_COUNT);
  readonly bubbleVelX = new Float32Array(BUBBLE_COUNT);
  readonly bubbleVelY = new Float32Array(BUBBLE_COUNT);
  readonly bubbleVelZ = new Float32Array(BUBBLE_COUNT);
  readonly bubbleRadius = new Float32Array(BUBBLE_COUNT);
  readonly bubblePhase = new Float32Array(BUBBLE_COUNT);
  readonly bubbleCubeSlot = new Int8Array(BUBBLE_COUNT);
  activeBubbleCount = 0;

  // Detaching Solid Mineral Crumb / Granule Pool (SoA, zero GC)
  readonly crumbAlive = new Uint8Array(CRUMB_COUNT);
  readonly crumbPosX = new Float32Array(CRUMB_COUNT);
  readonly crumbPosY = new Float32Array(CRUMB_COUNT);
  readonly crumbPosZ = new Float32Array(CRUMB_COUNT);
  readonly crumbVelX = new Float32Array(CRUMB_COUNT);
  readonly crumbVelY = new Float32Array(CRUMB_COUNT);
  readonly crumbVelZ = new Float32Array(CRUMB_COUNT);
  readonly crumbRotX = new Float32Array(CRUMB_COUNT);
  readonly crumbRotY = new Float32Array(CRUMB_COUNT);
  readonly crumbRotZ = new Float32Array(CRUMB_COUNT);
  readonly crumbSpin = new Float32Array(CRUMB_COUNT);
  readonly crumbScale = new Float32Array(CRUMB_COUNT);
  readonly crumbInitScale = new Float32Array(CRUMB_COUNT);
  readonly crumbLife = new Float32Array(CRUMB_COUNT);
  readonly crumbMaxLife = new Float32Array(CRUMB_COUNT);
  readonly crumbPigmentIndex = new Uint8Array(CRUMB_COUNT);
  activeCrumbCount = 0;

  private nextCubeId = 1;
  private pigmentCursor = 0;
  private bubbleSpawnCursor = 0;
  private bubbleEmitAccum = 0;
  private crumbSpawnCursor = 0;
  private crumbEmitAccum = 0;
  private popRippleBudget = 0;
  private waveAmbientTimer = 0;
  private rngState = 1337911;

  constructor(seedStarter = true) {
    // Initialize circular beaker wave mask
    for (let iy = 0; iy < WAVE_RES; iy++) {
      const nz = ((iy + 0.5) / WAVE_RES) * 2.0 - 1.0;
      for (let ix = 0; ix < WAVE_RES; ix++) {
        const nx = ((ix + 0.5) / WAVE_RES) * 2.0 - 1.0;
        this.waveInside[iy * WAVE_RES + ix] = nx * nx + nz * nz < 0.965 ? 1 : 0;
      }
    }

    this.waveTexture = new THREE.DataTexture(
      this.waveTexData,
      WAVE_RES,
      WAVE_RES,
      THREE.RGBAFormat,
      THREE.HalfFloatType
    );
    this.waveTexture.minFilter = THREE.LinearFilter;
    this.waveTexture.magFilter = THREE.LinearFilter;
    this.waveTexture.wrapS = THREE.ClampToEdgeWrapping;
    this.waveTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.waveTexture.generateMipmaps = false;

    // Pre-allocate MAX_ACTIVE_CUBES fixed slots
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      this.cubes.push({
        id: 0,
        slotIndex: i,
        active: false,
        pigmentIndex: 0,
        pigmentKey: DISSOLVE_PIGMENT_CYCLE[0],
        seed: (i + 1) * 1.731,
        initialMass: CUBE_UNIT_MASS,
        mass: 0,
        dissolvedMass: 0,
        submerged: false,
        splashTriggered: false,
        ageSec: 0,
        pos: new THREE.Vector3(0, BEAKER_INNER_FLOOR_Y + CUBE_INITIAL_HALF, 0),
        vel: new THREE.Vector3(),
        quat: new THREE.Quaternion(),
        angVel: new THREE.Vector3(),
        buoyantLift: 0,
        lastDissolveRate: 0,
      });
    }

    if (seedStarter) {
      this.seedStarterCrystal();
    } else {
      this.packWaveTexture();
    }
  }

  private nextRand(): number {
    // Deterministic Mulberry32 step (zero allocations)
    let t = (this.rngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Flushes the beaker clean with clear water and drops one crisp starter mineral cube.
   */
  flushBeaker(dropStarter = true): void {
    this.totalInjectedMass = 0;
    this.bathSoluteMass = 0;
    this.pigmentSoluteMass.fill(0);
    this.swirlAngularVel = 0;
    this.ekmanStrength = 0;
    this.waveCurr.fill(0);
    this.wavePrev.fill(0);
    this.waveNext.fill(0);
    this.bubbleAlive.fill(0);
    this.activeBubbleCount = 0;
    this.bubbleEmitAccum = 0;
    this.crumbAlive.fill(0);
    this.activeCrumbCount = 0;
    this.crumbEmitAccum = 0;

    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      const c = this.cubes[i];
      c.active = false;
      c.mass = 0;
      c.dissolvedMass = 0;
      c.submerged = false;
      c.splashTriggered = false;
      c.lastDissolveRate = 0;
    }

    this.updateBathOptics();
    if (dropStarter) {
      this.seedStarterCrystal();
    } else {
      this.packWaveTexture();
    }
  }

  /**
   * Drops the hero boot crystal already plunging cleanly through the meniscus so Frame 1
   * showcases crisp bevels, splash ripples, rising micro-bubbles, and peeling solute ribbons.
   */
  seedStarterCrystal(): DissolvingCubeState {
    const slot = this.cubes[0];
    const pigIdx = this.pigmentCursor % DISSOLVE_PIGMENT_CYCLE.length;
    this.pigmentCursor = (this.pigmentCursor + 1) % DISSOLVE_PIGMENT_CYCLE.length;

    slot.id = this.nextCubeId++;
    slot.active = true;
    slot.pigmentIndex = pigIdx;
    slot.pigmentKey = DISSOLVE_PIGMENT_CYCLE[pigIdx];
    slot.seed = 2.417 + slot.id * 0.713;
    slot.initialMass = CUBE_UNIT_MASS;
    slot.mass = CUBE_UNIT_MASS;
    slot.dissolvedMass = 0;
    slot.submerged = true;
    slot.splashTriggered = true;
    slot.ageSec = 0;
    slot.pos.set(0.04, BEAKER_INNER_FLOOR_Y + CUBE_INITIAL_HALF + 0.24, 0.03);
    slot.vel.set(-0.04, -0.46, 0.03);
    slot.quat.setFromEuler(new THREE.Euler(0.24, 0.46, -0.18));
    slot.angVel.set(0.55, 0.42, -0.36);
    slot.buoyantLift = 0;
    slot.lastDissolveRate = 0.045;

    this.totalInjectedMass += CUBE_UNIT_MASS;
    // Live entry splash on the meniscus + gentle natural swirling convective motion
    this.injectSplash(0.04, 0.03, -0.034, 0.24);
    this.injectSplash(-0.06, -0.05, 0.014, 0.32);
    this.swirlAngularVel = 0.42;

    // Pre-seed effervescent micro-bubble stream & detaching mineral crumbs off the plunging starter cube
    this.seedInitialBubbleColumn(slot, 280);
    this.seedInitialCrumbs(slot, 54);
    this.updateBathOptics();
    this.packWaveTexture();
    return slot;
  }

  /**
   * Drops a fresh crisp crystalline mineral cube into the beaker from just above the pouring rim.
   * Cycles cleanly through the 5 mineral pigments; up to 4 concurrent cubes can dissolve at once.
   */
  dropCube(worldX?: number, worldZ?: number, pigmentOverride?: PigmentKey): DissolvingCubeState | null {
    // Pick an inactive slot, or recycle the smallest nearly-gone cube if all 4 slots are full
    let slot: DissolvingCubeState | null = null;
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      if (!this.cubes[i].active) {
        slot = this.cubes[i];
        break;
      }
    }
    if (!slot) {
      let smallestIdx = 0;
      let minMass = Infinity;
      for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
        if (this.cubes[i].mass < minMass) {
          minMass = this.cubes[i].mass;
          smallestIdx = i;
        }
      }
      slot = this.cubes[smallestIdx];
      // Conserve exact mass of displaced tiny remnant by dissolving it cleanly into the bath
      if (slot.mass > 0) {
        this.bathSoluteMass += slot.mass;
        this.pigmentSoluteMass[slot.pigmentIndex] += slot.mass;
        slot.dissolvedMass += slot.mass;
        slot.mass = 0;
        this.totalDissolvedCount++;
      }
    }

    let pigIdx: number;
    if (pigmentOverride !== undefined) {
      const found = DISSOLVE_PIGMENT_CYCLE.indexOf(pigmentOverride);
      pigIdx = found >= 0 ? found : this.pigmentCursor % DISSOLVE_PIGMENT_CYCLE.length;
    } else {
      pigIdx = this.pigmentCursor % DISSOLVE_PIGMENT_CYCLE.length;
      this.pigmentCursor = (this.pigmentCursor + 1) % DISSOLVE_PIGMENT_CYCLE.length;
    }

    const maxSpawnR = WATER_RADIUS * 0.56;
    let sx = worldX !== undefined ? worldX : (this.nextRand() - 0.5) * maxSpawnR * 1.4;
    let sz = worldZ !== undefined ? worldZ : (this.nextRand() - 0.5) * maxSpawnR * 1.4;
    const sr = Math.hypot(sx, sz);
    if (sr > maxSpawnR) {
      sx = (sx / sr) * maxSpawnR;
      sz = (sz / sr) * maxSpawnR;
    }

    slot.id = this.nextCubeId++;
    slot.active = true;
    slot.pigmentIndex = pigIdx;
    slot.pigmentKey = DISSOLVE_PIGMENT_CYCLE[pigIdx];
    slot.seed = 1.13 + slot.id * 1.6180339;
    slot.initialMass = CUBE_UNIT_MASS;
    slot.mass = CUBE_UNIT_MASS;
    slot.dissolvedMass = 0;
    slot.submerged = false;
    slot.splashTriggered = false;
    slot.ageSec = 0;

    // Drop cleanly above the water surface so user sees the plunge & splash ripple
    slot.pos.set(sx, BEAKER_TOP_Y + 0.16, sz);
    slot.vel.set((this.nextRand() - 0.5) * 0.08, -0.55, (this.nextRand() - 0.5) * 0.08);
    slot.quat.setFromEuler(
      new THREE.Euler(
        (this.nextRand() - 0.5) * 0.9,
        this.nextRand() * Math.PI * 2.0,
        (this.nextRand() - 0.5) * 0.9
      )
    );
    slot.angVel.set(
      (this.nextRand() - 0.5) * 3.2,
      (this.nextRand() - 0.5) * 2.6,
      (this.nextRand() - 0.5) * 3.2
    );
    slot.buoyantLift = 0;
    slot.lastDissolveRate = 0;

    this.totalInjectedMass += CUBE_UNIT_MASS;
    this.updateBathOptics();
    return slot;
  }

  /**
   * Updates the borosilicate glass stirring rod pointer target and imparts real tangential
   * vortex torque (`r x v`) + meniscus capillary wake into the water column.
   */
  setStirringRodTarget(active: boolean, targetX: number, targetZ: number, dtSec: number): void {
    const dt = clamp(dtSec, 0.001, 0.05);
    this.rod.active = active;

    if (!active) {
      this.rod.vx *= Math.exp(-10.0 * dt);
      this.rod.vz *= Math.exp(-10.0 * dt);
      return;
    }

    const maxR = WATER_RADIUS - 0.055;
    let tx = targetX;
    let tz = targetZ;
    const r = Math.hypot(tx, tz);
    if (r > maxR) {
      tx = (tx / r) * maxR;
      tz = (tz / r) * maxR;
    }

    const prevX = this.rod.x;
    const prevZ = this.rod.z;
    const follow = 1.0 - Math.exp(-22.0 * dt);
    this.rod.x += (tx - this.rod.x) * follow;
    this.rod.z += (tz - this.rod.z) * follow;

    const vx = (this.rod.x - prevX) / dt;
    const vz = (this.rod.z - prevZ) / dt;
    this.rod.vx = this.rod.vx * 0.62 + vx * 0.38;
    this.rod.vz = this.rod.vz * 0.62 + vz * 0.38;

    // Gentle laboratory wrist tilt opposing motion
    this.rod.tiltX = clamp(0.16 - this.rod.vz * 0.08, -0.28, 0.36);
    this.rod.tiltZ = clamp(-0.14 + this.rod.vx * 0.08, -0.36, 0.28);

    if (this.rod.dip > 0.25) {
      this.applyStirImpulse(this.rod.x, this.rod.z, this.rod.vx, this.rod.vz, dt);
    }
  }

  /**
   * Direct physical tangential torque impulse from a stirring stroke at `(rx, rz)` with velocity `(vx, vz)`.
   * Right-handed around +Y: `tau_y = z * vx - x * vz`.
   */
  applyStirImpulse(rx: number, rz: number, vx: number, vz: number, dtSec: number): void {
    const dt = clamp(dtSec, 0.001, 0.05);
    const speed = Math.hypot(vx, vz);
    if (!Number.isFinite(speed) || speed < 1e-4) return;

    // Angular momentum transfer tau_y = (r x v)_y = z * vx - x * vz
    const tauY = rz * vx - rx * vz;
    const dOmega = clamp(tauY * 13.5 * dt, -3.5, 3.5);
    this.swirlAngularVel = clamp(this.swirlAngularVel + dOmega, -MAX_SWIRL_OMEGA, MAX_SWIRL_OMEGA);

    // Inject crisp rod-wake capillary disturbance on meniscus
    const wakeStrength = clamp(speed * 0.0045, 0.001, 0.016);
    this.injectSplash(rx, rz, -wakeStrength, 0.14);
  }

  /**
   * Evaluates physical Rankine / Lamb-Oseen tangential fluid velocity at `(x, z)` inside the beaker.
   * Respects no-slip damping near the inner borosilicate wall (`r -> WATER_RADIUS`).
   */
  evalFluidVelocityXZ(x: number, z: number, out: { vx: number; vz: number }): void {
    const r = Math.hypot(x, z);
    if (r < 1e-5 || Math.abs(this.swirlAngularVel) < 1e-5) {
      out.vx = 0;
      out.vz = 0;
      return;
    }
    const rNorm = clamp(r / WATER_RADIUS, 0.0, 1.0);
    const r2 = rNorm * rNorm;
    const wallNoSlip = Math.max(0.0, 1.0 - r2 * r2);
    const omegaLocal = this.swirlAngularVel * (0.28 + 0.72 * wallNoSlip);
    // Right-handed +Y rotation: omega_vec x r = (+omega * z, 0, -omega * x)
    out.vx = omegaLocal * z;
    out.vz = -omegaLocal * x;
  }

  /**
   * Injects a zero-mean volume-conserving capillary splash dimple + collar into the 2D meniscus PDE.
   */
  injectSplash(worldX: number, worldZ: number, strength: number, radiusNorm = 0.22): void {
    const normX = clamp(worldX / WATER_RADIUS, -0.92, 0.92);
    const normZ = clamp(worldZ / WATER_RADIUS, -0.92, 0.92);
    const invSigma2 = 1.0 / Math.max(0.003, radiusNorm * radiusNorm);
    const cutoff = 5.5 * radiusNorm * radiusNorm;

    for (let iy = 0; iy < WAVE_RES; iy++) {
      const gz = ((iy + 0.5) / WAVE_RES) * 2.0 - 1.0;
      const dz = gz - normZ;
      const row = iy * WAVE_RES;
      for (let ix = 0; ix < WAVE_RES; ix++) {
        const idx = row + ix;
        if (this.waveInside[idx] === 0) continue;
        const gx = ((ix + 0.5) / WAVE_RES) * 2.0 - 1.0;
        const dx = gx - normX;
        const d2 = dx * dx + dz * dz;
        if (d2 > cutoff) continue;
        const g = Math.exp(-d2 * invSigma2) * (1.0 - 0.46 * d2 * invSigma2);
        this.waveCurr[idx] = clamp(this.waveCurr[idx] + strength * g, -0.055, 0.055);
      }
    }
  }

  /**
   * Samples current interpolated wave height (plus parabolic vortex depression) at world `(x, z)`.
   */
  sampleMeniscusOffsetY(worldX: number, worldZ: number): number {
    const nx = clamp(worldX / WATER_RADIUS, -1.0, 1.0);
    const nz = clamp(worldZ / WATER_RADIUS, -1.0, 1.0);

    const fx = clamp((nx * 0.5 + 0.5) * (WAVE_RES - 1), 0.0, WAVE_RES - 1);
    const fz = clamp((nz * 0.5 + 0.5) * (WAVE_RES - 1), 0.0, WAVE_RES - 1);
    const x0 = Math.floor(fx);
    const z0 = Math.floor(fz);
    const x1 = Math.min(WAVE_RES - 1, x0 + 1);
    const z1 = Math.min(WAVE_RES - 1, z0 + 1);
    const tx = fx - x0;
    const tz = fz - z0;

    const h00 = this.waveCurr[z0 * WAVE_RES + x0];
    const h10 = this.waveCurr[z0 * WAVE_RES + x1];
    const h01 = this.waveCurr[z1 * WAVE_RES + x0];
    const h11 = this.waveCurr[z1 * WAVE_RES + x1];
    const waveH = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;

    // Volume-conserving parabolic free-surface vortex funnel: dips in center, climbs at wall!
    const r2 = Math.min(1.0, nx * nx + nz * nz);
    const omegaSq = Math.min(85.0, this.swirlAngularVel * this.swirlAngularVel);
    const vortexDip = 0.00145 * omegaSq * (2.0 * r2 - 1.0);

    return clamp(waveH + vortexDip, -0.095, 0.095);
  }

  /**
   * Advances the complete coupled beaker dissolution simulation by `dtSec`.
   */
  step(dtSec: number): void {
    const dt = clamp(dtSec, 0.001, 0.04);
    this.simTime += dt;
    this.popRippleBudget = 3;

    // 1. Smoothly lower / lift borosilicate stirring rod
    const dipGoal = this.rod.active ? 1.0 : 0.0;
    this.rod.dip += (dipGoal - this.rod.dip) * (1.0 - Math.exp(-11.0 * dt));

    // 2. Viscous decay of bulk vortex swirl & secondary Ekman tea-leaf circulation
    const linDecay = Math.exp(-0.44 * dt);
    const quadDecay = 1.0 / (1.0 + 0.08 * Math.abs(this.swirlAngularVel) * dt);
    this.swirlAngularVel = clamp(
      this.swirlAngularVel * linDecay * quadDecay,
      -MAX_SWIRL_OMEGA,
      MAX_SWIRL_OMEGA
    );
    if (Math.abs(this.swirlAngularVel) < 1e-4) this.swirlAngularVel = 0;

    const targetEkman = Math.min(1.0, (this.swirlAngularVel * this.swirlAngularVel) / 18.0);
    this.ekmanStrength += (targetEkman - this.ekmanStrength) * (1.0 - Math.exp(-3.5 * dt));

    // 3. Step dissolving mineral cubes (kinematics, buoyancy, curvature dissolution, solute conservation)
    this.stepCubes(dt);

    // 4. Step 2,000-instance rising effervescent silver micro-bubble pool
    this.stepBubbles(dt);

    // 4b. Step 512-instance detaching solid mineral crumbs flaking off the active eroding crust
    this.stepCrumbs(dt);

    // 5. Step 2D circular gravity-capillary meniscus wave PDE & pack HalfFloat DataTexture
    this.stepWavePde(dt);

    // 6. Update accumulated multi-pigment Beer-Lambert bath optics
    this.updateBathOptics();
  }

  private stepCubes(dt: number): void {
    const fluidVel = { vx: 0, vz: 0 };
    const swirlMag = Math.abs(this.swirlAngularVel);
    const rodSpeed = this.rod.dip > 0.25 ? Math.hypot(this.rod.vx, this.rod.vz) : 0;

    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      const c = this.cubes[i];
      if (!c.active) continue;

      c.ageSec += dt;
      const metrics = evalCrystalErosionMetrics(c.mass);
      const effHalfY = Math.max(0.014, metrics.halfExtent * metrics.scaleY);
      const effRadXZ = Math.max(0.018, metrics.halfExtent * metrics.scaleXZ * 1.15);

      // Check water-surface entry for newly dropped cubes
      const surfY = WATER_REST_Y + this.sampleMeniscusOffsetY(c.pos.x, c.pos.z);
      if (!c.submerged) {
        c.vel.y -= 6.8 * dt;
        c.pos.addScaledVector(c.vel, dt);
        this.integrateCubeRotation(c, dt);

        if (c.pos.y - effHalfY * 0.45 <= surfY) {
          c.submerged = true;
          if (!c.splashTriggered) {
            c.splashTriggered = true;
            this.injectSplash(c.pos.x, c.pos.z, -0.044, 0.25);
            this.spawnSplashBubbles(c, 55);
            this.spawnSplashCrumbs(c, 24);
          }
          // Immediate entry drag decelerates air plunge into smooth underwater sink
          c.vel.y *= 0.48;
          c.vel.x *= 0.65;
          c.vel.z *= 0.65;
        }
        continue;
      }

      // ---- Submerged Hydrodynamics & Effervescent Bubble-Jet Rocking/Lift ----
      this.evalFluidVelocityXZ(c.pos.x, c.pos.z, fluidVel);

      // Secondary Ekman tea-leaf inflow near bottom draws cubes smoothly toward central spiral
      const rXZ = Math.hypot(c.pos.x, c.pos.z);
      const nearFloor = smoothstepCpu(
        BEAKER_INNER_FLOOR_Y + 0.45,
        BEAKER_INNER_FLOOR_Y + 0.05,
        c.pos.y
      );
      let ekmanFx = 0;
      let ekmanFz = 0;
      if (rXZ > 0.03 && this.ekmanStrength > 0.02) {
        const pull = -this.ekmanStrength * nearFloor * 0.42;
        ekmanFx = (c.pos.x / rXZ) * pull;
        ekmanFz = (c.pos.z / rXZ) * pull;
      }

      // Direct rod-to-cube gentle nudge when sweeping stirring rod brushes past a cube
      let rodPushX = 0;
      let rodPushZ = 0;
      if (this.rod.dip > 0.35) {
        const dxR = c.pos.x - this.rod.x;
        const dzR = c.pos.z - this.rod.z;
        const dR = Math.hypot(dxR, dzR);
        const minSep = effRadXZ + 0.045;
        if (dR < minSep && dR > 1e-4) {
          const push = ((minSep - dR) / minSep) * 3.8;
          rodPushX = (dxR / dR) * push + this.rod.vx * 0.45;
          rodPushZ = (dzR / dR) * push + this.rod.vz * 0.45;
        }
      }

      // Uneven effervescent bubble-jet lift & buoyant wafer levitation as crystal shrinks
      const sliverLift = smoothstepCpu(0.40, 0.05, c.mass);
      const pulseA = Math.sin(this.simTime * 4.4 + c.seed * 6.3);
      const pulseB = Math.cos(this.simTime * 6.1 + c.seed * 3.7);
      const netGravityY = -(1.55 + 2.35 * c.mass);
      const bubbleLiftY =
        (0.45 + 2.65 * sliverLift) * (0.78 + 0.34 * pulseA) +
        this.ekmanStrength * (0.55 + 1.45 * sliverLift) * Math.exp(-rXZ * rXZ * 6.0);
      c.buoyantLift = clamp(bubbleLiftY / Math.max(0.5, Math.abs(netGravityY)), 0.0, 2.0);

      // Fluid drag relaxes cube velocity toward swirling liquid velocity
      const dragRate = 3.4 + 2.6 * (1.0 - c.mass);
      c.vel.x += ((fluidVel.vx - c.vel.x) * dragRate + ekmanFx + rodPushX) * dt;
      c.vel.z += ((fluidVel.vz - c.vel.z) * dragRate + ekmanFz + rodPushZ) * dt;
      c.vel.y += (netGravityY + bubbleLiftY - c.vel.y * (3.2 + 2.4 * sliverLift)) * dt;

      c.pos.addScaledVector(c.vel, dt);

      // Floor collision & bubble-jet micro-rocking torque
      const floorLimitY = BEAKER_INNER_FLOOR_Y + effHalfY * 0.96;
      const onFloor = c.pos.y <= floorLimitY + 0.008;
      if (c.pos.y < floorLimitY) {
        c.pos.y = floorLimitY;
        if (c.vel.y < -0.05) {
          c.vel.y = -c.vel.y * 0.14;
        } else if (c.vel.y < 0) {
          c.vel.y = 0;
        }
        // Subtle floor friction still permits lively vortex orbiting
        const floorDrag = Math.exp(-1.5 * dt);
        c.vel.x *= floorDrag;
        c.vel.z *= floorDrag;
      }

      // Keep lifted buoyant slivers safely below the water meniscus
      const ceilingY = surfY - effHalfY * 1.15;
      if (c.pos.y > ceilingY) {
        c.pos.y = ceilingY;
        if (c.vel.y > 0) c.vel.y *= -0.2;
      }

      // Smooth cylindrical glass inner wall containment
      const maxCubeR = WATER_RADIUS - effRadXZ * 0.96;
      const curR = Math.hypot(c.pos.x, c.pos.z);
      if (curR > maxCubeR && curR > 1e-5) {
        const nx = c.pos.x / curR;
        const nz = c.pos.z / curR;
        c.pos.x = nx * maxCubeR;
        c.pos.z = nz * maxCubeR;
        const vRad = c.vel.x * nx + c.vel.z * nz;
        if (vRad > 0) {
          c.vel.x -= 1.25 * vRad * nx;
          c.vel.z -= 1.25 * vRad * nz;
        }
      }

      // Uneven bubble-jet torque + vortex spin + gentle floor face-settling when heavy
      const rockTorqueAmp = 0.65 + 2.2 * sliverLift + 0.18 * swirlMag;
      c.angVel.x += (pulseA * rockTorqueAmp - c.angVel.x * 3.1) * dt;
      c.angVel.z += (pulseB * rockTorqueAmp - c.angVel.z * 3.1) * dt;
      const targetSpinY =
        this.swirlAngularVel * (0.65 + 0.45 * sliverLift) +
        Math.sin(this.simTime * 2.1 + c.seed) * (0.28 + 0.85 * sliverLift);
      c.angVel.y += ((targetSpinY - c.angVel.y) * 2.8) * dt;

      // Subtle gravity settling of heavy square cube faces toward flat floor contact
      if (onFloor && c.mass > 0.45 && swirlMag < 1.8) {
        // Pull nearest local cube axis gently toward +Y
        _tmpBodyAxis.set(0, 1, 0).applyQuaternion(c.quat);
        const ax = Math.abs(_tmpBodyAxis.x);
        const ay = Math.abs(_tmpBodyAxis.y);
        const az = Math.abs(_tmpBodyAxis.z);
        if (ay < 0.995 && ay >= ax && ay >= az) {
          const signY = _tmpBodyAxis.y >= 0 ? 1 : -1;
          _tmpAxis.crossVectors(_tmpBodyAxis, _tmpUp).multiplyScalar(signY);
          c.angVel.addScaledVector(_tmpAxis, 2.2 * c.mass * dt);
        }
      }

      this.integrateCubeRotation(c, dt);

      // ---- Curvature-Enhanced & Convective Sherwood Dissolution Rate ----
      // Base rate ~16-18s at rest so surface erosion, corner biting & crumb shedding read clearly
      // within the first 2-4 seconds, accelerating up to ~2.8x under active glass-rod stirring.
      const areaExposure = 0.24 + 0.76 * Math.pow(Math.max(0.0, c.mass), 0.52);
      const convectiveSherwood = 1.0 + 0.25 * swirlMag + 0.34 * Math.min(2.5, rodSpeed);
      const dissolveRate = 0.068 * areaExposure * convectiveSherwood;
      c.lastDissolveRate = dissolveRate;

      let dm = dissolveRate * dt;
      if (c.mass - dm <= 1e-5) {
        dm = c.mass;
        c.mass = 0;
        c.dissolvedMass += dm;
        c.active = false;
        c.lastDissolveRate = 0;
        this.totalDissolvedCount++;
      } else {
        c.mass -= dm;
        c.dissolvedMass += dm;
      }

      // Exact solute mass conservation transfer: deltaSolid + deltaBath === 0
      this.bathSoluteMass += dm;
      this.pigmentSoluteMass[c.pigmentIndex] += dm;
    }

    // Soft-sphere separation between concurrent dissolving cubes
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      const a = this.cubes[i];
      if (!a.active || !a.submerged) continue;
      const rA = evalCrystalErosionMetrics(a.mass).halfExtent * 1.15;
      for (let j = i + 1; j < MAX_ACTIVE_CUBES; j++) {
        const b = this.cubes[j];
        if (!b.active || !b.submerged) continue;
        const rB = evalCrystalErosionMetrics(b.mass).halfExtent * 1.15;
        const dx = b.pos.x - a.pos.x;
        const dy = b.pos.y - a.pos.y;
        const dz = b.pos.z - a.pos.z;
        const dist = Math.hypot(dx, dy, dz);
        const minDist = rA + rB;
        if (dist < minDist && dist > 1e-5) {
          const overlap = (minDist - dist) * 0.5;
          const nx = dx / dist;
          const ny = dy / dist;
          const nz = dz / dist;
          a.pos.x -= nx * overlap;
          a.pos.y -= ny * overlap * 0.35;
          a.pos.z -= nz * overlap;
          b.pos.x += nx * overlap;
          b.pos.y += ny * overlap * 0.35;
          b.pos.z += nz * overlap;
          a.vel.x -= nx * overlap * 4.0;
          a.vel.z -= nz * overlap * 4.0;
          b.vel.x += nx * overlap * 4.0;
          b.vel.z += nz * overlap * 4.0;
        }
      }
    }
  }

  private integrateCubeRotation(c: DissolvingCubeState, dt: number): void {
    const wLen = c.angVel.length();
    if (wLen > 1e-5) {
      _tmpAxis.copy(c.angVel).divideScalar(wLen);
      _tmpQuat.setFromAxisAngle(_tmpAxis, wLen * dt);
      c.quat.premultiply(_tmpQuat).normalize();
    }
  }

  // -------------------------------------------------------------------------
  // Rising Effervescent Silver Micro-Bubble Streams (Nucleated on Eroded Facets)
  // -------------------------------------------------------------------------
  private spawnBubbleOnCube(cube: DissolvingCubeState, heightProgress = 0): void {
    const idx = this.bubbleSpawnCursor;
    this.bubbleSpawnCursor = (this.bubbleSpawnCursor + 1) % BUBBLE_COUNT;

    // Sample a point on one of the 6 unit cube faces, slightly biased toward pitted edges/corners
    const face = Math.floor(this.nextRand() * 6);
    const s1 = this.nextRand() * 2.0 - 1.0;
    const s2 = this.nextRand() * 2.0 - 1.0;
    // Gently push distribution toward edges where dissolution effervescence is fiercest
    const u1 = Math.sign(s1) * Math.pow(Math.abs(s1), 0.72);
    const u2 = Math.sign(s2) * Math.pow(Math.abs(s2), 0.72);

    let ux = 0;
    let uy = 0;
    let uz = 0;
    if (face === 0) {
      ux = 1;
      uy = u1;
      uz = u2;
    } else if (face === 1) {
      ux = -1;
      uy = u1;
      uz = u2;
    } else if (face === 2) {
      ux = u1;
      uy = 1;
      uz = u2;
    } else if (face === 3) {
      ux = u1;
      uy = -1;
      uz = u2;
    } else if (face === 4) {
      ux = u1;
      uy = u2;
      uz = 1;
    } else {
      ux = u1;
      uy = u2;
      uz = -1;
    }

    evalCrystalSurfacePoint(ux, uy, uz, cube.mass, cube.seed, _tmpLocalPt, _tmpLocalNorm);
    _tmpWorldPt.copy(_tmpLocalPt).applyQuaternion(cube.quat).add(cube.pos);
    _tmpWorldNorm.copy(_tmpLocalNorm).applyQuaternion(cube.quat);

    const r = 0.0042 + Math.pow(this.nextRand(), 1.9) * 0.0115;
    const riseSpan = Math.max(0.05, WATER_REST_Y - _tmpWorldPt.y);

    this.bubbleAlive[idx] = 1;
    this.bubblePosX[idx] = _tmpWorldPt.x + _tmpWorldNorm.x * (r + 0.003);
    this.bubblePosY[idx] =
      _tmpWorldPt.y + _tmpWorldNorm.y * (r + 0.003) + heightProgress * riseSpan * 0.88;
    this.bubblePosZ[idx] = _tmpWorldPt.z + _tmpWorldNorm.z * (r + 0.003);

    this.bubbleVelX[idx] = _tmpWorldNorm.x * 0.065 + (this.nextRand() - 0.5) * 0.04;
    this.bubbleVelY[idx] = 0.24 + r * 32.0 + this.nextRand() * 0.14;
    this.bubbleVelZ[idx] = _tmpWorldNorm.z * 0.065 + (this.nextRand() - 0.5) * 0.04;
    this.bubbleRadius[idx] = r;
    this.bubblePhase[idx] = this.nextRand() * Math.PI * 2.0;
    this.bubbleCubeSlot[idx] = cube.slotIndex;
  }

  private spawnSplashBubbles(cube: DissolvingCubeState, count: number): void {
    for (let k = 0; k < count; k++) {
      const idx = this.bubbleSpawnCursor;
      this.bubbleSpawnCursor = (this.bubbleSpawnCursor + 1) % BUBBLE_COUNT;
      const ang = this.nextRand() * Math.PI * 2.0;
      const dist = 0.04 + this.nextRand() * 0.20;
      const r = 0.0055 + Math.pow(this.nextRand(), 1.6) * 0.013;

      this.bubbleAlive[idx] = 1;
      this.bubblePosX[idx] = clamp(cube.pos.x + Math.cos(ang) * dist, -WATER_RADIUS * 0.9, WATER_RADIUS * 0.9);
      this.bubblePosY[idx] = WATER_REST_Y - 0.04 - this.nextRand() * 0.28;
      this.bubblePosZ[idx] = clamp(cube.pos.z + Math.sin(ang) * dist, -WATER_RADIUS * 0.9, WATER_RADIUS * 0.9);
      this.bubbleVelX[idx] = Math.cos(ang) * (0.14 + this.nextRand() * 0.18);
      this.bubbleVelY[idx] = -0.22 - this.nextRand() * 0.25;
      this.bubbleVelZ[idx] = Math.sin(ang) * (0.14 + this.nextRand() * 0.18);
      this.bubbleRadius[idx] = r;
      this.bubblePhase[idx] = this.nextRand() * Math.PI * 2.0;
      this.bubbleCubeSlot[idx] = cube.slotIndex;
    }
  }

  private seedInitialBubbleColumn(cube: DissolvingCubeState, count: number): void {
    for (let k = 0; k < count; k++) {
      const hFrac = this.nextRand();
      this.spawnBubbleOnCube(cube, hFrac);
    }
    this.activeBubbleCount = count;
  }

  private stepBubbles(dt: number): void {
    // Emit fresh micro-bubbles directly from active submerged dissolving crystals
    let totalEmitRate = 0;
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      const c = this.cubes[i];
      if (!c.active || !c.submerged || c.mass <= 0) continue;
      // Rich continuous effervescent streams (~240..420 bubbles/sec per dissolving cube!)
      const cubeRate = (260.0 + 180.0 * Math.pow(c.mass, 0.45)) * (1.0 + 0.12 * Math.abs(this.swirlAngularVel));
      totalEmitRate += cubeRate;
    }

    this.bubbleEmitAccum += totalEmitRate * dt;
    const toSpawn = Math.min(48, Math.floor(this.bubbleEmitAccum));
    this.bubbleEmitAccum -= toSpawn;

    if (toSpawn > 0) {
      let spawned = 0;
      for (let pass = 0; pass < MAX_ACTIVE_CUBES && spawned < toSpawn; pass++) {
        // Round-robin distribute across active submerged cubes
        for (let i = 0; i < MAX_ACTIVE_CUBES && spawned < toSpawn; i++) {
          const c = this.cubes[i];
          if (!c.active || !c.submerged || c.mass <= 0) continue;
          const share = Math.max(1, Math.ceil(toSpawn / MAX_ACTIVE_CUBES));
          for (let s = 0; s < share && spawned < toSpawn; s++) {
            this.spawnBubbleOnCube(c, 0);
            spawned++;
          }
        }
      }
    }

    // Advect all live bubbles: buoyant Stokes rise + helical micro-wobble + vortex spiral + meniscus pop
    const fluidVel = { vx: 0, vz: 0 };
    const maxR = WATER_RADIUS - 0.018;
    const floorY = BEAKER_INNER_FLOOR_Y + 0.014;
    const t = this.simTime;
    const omegaAbs = Math.abs(this.swirlAngularVel);
    let aliveCount = 0;

    for (let i = 0; i < BUBBLE_COUNT; i++) {
      if (this.bubbleAlive[i] === 0) continue;

      let x = this.bubblePosX[i];
      let y = this.bubblePosY[i];
      let z = this.bubblePosZ[i];
      const rad = this.bubbleRadius[i];
      const ph = this.bubblePhase[i];

      this.evalFluidVelocityXZ(x, z, fluidVel);

      // Rising buoyant terminal velocity scales with bubble radius
      const targetVy = 0.34 + rad * 36.0;
      // Organic helical wobble (irrational frequencies, zero star harmonics)
      const wobbleFreq = 6.2 + (i % 7) * 0.53;
      const wobbleAmp = 0.055 + rad * 4.5;
      const wobX = Math.sin(t * wobbleFreq + ph) * wobbleAmp;
      const wobZ = Math.cos(t * wobbleFreq * 0.87 + ph * 1.31) * wobbleAmp;

      // Swirling bubbles get drawn gently inward toward the glittering central vortex funnel!
      let vortexInX = 0;
      let vortexInZ = 0;
      if (omegaAbs > 0.35) {
        const rHoriz = Math.hypot(x, z);
        if (rHoriz > 0.04) {
          const centripetal = -Math.min(0.42, omegaAbs * 0.065);
          vortexInX = (x / rHoriz) * centripetal;
          vortexInZ = (z / rHoriz) * centripetal;
        }
      }

      const relax = 1.0 - Math.exp(-7.2 * dt);
      this.bubbleVelX[i] += (fluidVel.vx + wobX + vortexInX - this.bubbleVelX[i]) * relax;
      this.bubbleVelY[i] += (targetVy - this.bubbleVelY[i]) * (1.0 - Math.exp(-5.4 * dt));
      this.bubbleVelZ[i] += (fluidVel.vz + wobZ + vortexInZ - this.bubbleVelZ[i]) * relax;

      x += this.bubbleVelX[i] * dt;
      y += this.bubbleVelY[i] * dt;
      z += this.bubbleVelZ[i] * dt;

      if (y < floorY) y = floorY;

      const rNew = Math.hypot(x, z);
      if (rNew > maxR && rNew > 1e-5) {
        const scale = maxR / rNew;
        x *= scale;
        z *= scale;
      }

      // Pop cleanly right at the deformable water meniscus, spawning subtle capillary pop ripples!
      const surfY = WATER_REST_Y + this.sampleMeniscusOffsetY(x, z) - 0.008;
      if (y >= surfY) {
        this.bubbleAlive[i] = 0;
        if (this.popRippleBudget > 0 && rad > 0.0075) {
          this.popRippleBudget--;
          this.injectSplash(x, z, 0.0042 + rad * 0.16, 0.095);
        }
        continue;
      }

      this.bubblePosX[i] = x;
      this.bubblePosY[i] = y;
      this.bubblePosZ[i] = z;
      aliveCount++;
    }

    this.activeBubbleCount = aliveCount;
  }

  // -------------------------------------------------------------------------
  // Detaching Solid Mineral Crumbs / Granules Flaking Off the Eroding Crust
  // -------------------------------------------------------------------------
  private spawnCrumbOnCube(cube: DissolvingCubeState, preAgeFrac = 0): void {
    const idx = this.crumbSpawnCursor;
    this.crumbSpawnCursor = (this.crumbSpawnCursor + 1) % CRUMB_COUNT;

    // Strongly bias detachment toward high-curvature corners & edges where the crust clips away first
    const face = Math.floor(this.nextRand() * 6);
    const s1 = this.nextRand() * 2.0 - 1.0;
    const s2 = this.nextRand() * 2.0 - 1.0;
    const u1 = Math.sign(s1) * Math.pow(Math.abs(s1), 0.42);
    const u2 = Math.sign(s2) * Math.pow(Math.abs(s2), 0.42);

    let ux = 0;
    let uy = 0;
    let uz = 0;
    if (face === 0) {
      ux = 1;
      uy = u1;
      uz = u2;
    } else if (face === 1) {
      ux = -1;
      uy = u1;
      uz = u2;
    } else if (face === 2) {
      ux = u1;
      uy = 1;
      uz = u2;
    } else if (face === 3) {
      ux = u1;
      uy = -1;
      uz = u2;
    } else if (face === 4) {
      ux = u1;
      uy = u2;
      uz = 1;
    } else {
      ux = u1;
      uy = u2;
      uz = -1;
    }

    evalCrystalSurfacePoint(ux, uy, uz, cube.mass, cube.seed, _tmpLocalPt, _tmpLocalNorm);
    _tmpWorldPt.copy(_tmpLocalPt).applyQuaternion(cube.quat).add(cube.pos);
    _tmpWorldNorm.copy(_tmpLocalNorm).applyQuaternion(cube.quat);

    // Chunky faceted mineral grains (larger early-mid dissolution, finer salt crystals near end)
    const massScale = 0.55 + 0.45 * Math.pow(Math.max(0.02, cube.mass), 0.4);
    const initScale = (0.0068 + Math.pow(this.nextRand(), 1.55) * 0.0175) * massScale;
    const maxLife = 2.0 + this.nextRand() * 2.8;
    const life = clamp(1.0 - preAgeFrac, 0.12, 1.0);

    const peelSpeed = 0.065 + this.nextRand() * 0.11;
    // Slight bubble-jostle scatter on detachment; heavier chunks sink decisively toward glass floor
    const isTinyFleck = initScale < 0.011;
    const initVy = isTinyFleck
      ? _tmpWorldNorm.y * peelSpeed + 0.04 + (this.nextRand() - 0.5) * 0.12
      : _tmpWorldNorm.y * peelSpeed * 0.45 - (0.09 + this.nextRand() * 0.14);

    const floorY = BEAKER_INNER_FLOOR_Y + initScale * 0.9;
    let px = _tmpWorldPt.x + _tmpWorldNorm.x * (initScale * 0.75 + 0.004);
    let py = _tmpWorldPt.y + _tmpWorldNorm.y * (initScale * 0.75 + 0.004);
    let pz = _tmpWorldPt.z + _tmpWorldNorm.z * (initScale * 0.75 + 0.004);

    if (preAgeFrac > 0) {
      const tPre = preAgeFrac * maxLife * 0.42;
      px += (_tmpWorldNorm.x * peelSpeed + (this.nextRand() - 0.5) * 0.08) * tPre;
      py += initVy * tPre - 0.08 * tPre * tPre;
      pz += (_tmpWorldNorm.z * peelSpeed + (this.nextRand() - 0.5) * 0.08) * tPre;
    }

    py = clamp(py, floorY, WATER_REST_Y - 0.03);
    const rHoriz = Math.hypot(px, pz);
    const maxR = WATER_RADIUS - 0.025;
    if (rHoriz > maxR && rHoriz > 1e-5) {
      px = (px / rHoriz) * maxR;
      pz = (pz / rHoriz) * maxR;
    }

    this.crumbAlive[idx] = 1;
    this.crumbPosX[idx] = px;
    this.crumbPosY[idx] = py;
    this.crumbPosZ[idx] = pz;
    this.crumbVelX[idx] = _tmpWorldNorm.x * peelSpeed + (this.nextRand() - 0.5) * 0.05;
    this.crumbVelY[idx] = initVy;
    this.crumbVelZ[idx] = _tmpWorldNorm.z * peelSpeed + (this.nextRand() - 0.5) * 0.05;
    this.crumbRotX[idx] = this.nextRand() * Math.PI * 2.0;
    this.crumbRotY[idx] = this.nextRand() * Math.PI * 2.0;
    this.crumbRotZ[idx] = this.nextRand() * Math.PI * 2.0;
    this.crumbSpin[idx] = (this.nextRand() - 0.5) * 5.2;
    this.crumbInitScale[idx] = initScale;
    this.crumbScale[idx] = initScale * Math.pow(life, 0.72);
    this.crumbLife[idx] = life;
    this.crumbMaxLife[idx] = maxLife;
    this.crumbPigmentIndex[idx] = cube.pigmentIndex;
  }

  private spawnSplashCrumbs(cube: DissolvingCubeState, count: number): void {
    for (let k = 0; k < count; k++) {
      this.spawnCrumbOnCube(cube, 0);
    }
  }

  private seedInitialCrumbs(cube: DissolvingCubeState, count: number): void {
    for (let k = 0; k < count; k++) {
      this.spawnCrumbOnCube(cube, this.nextRand() * 0.72);
    }
    this.activeCrumbCount = count;
  }

  private stepCrumbs(dt: number): void {
    let totalEmitRate = 0;
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      const c = this.cubes[i];
      if (!c.active || !c.submerged || c.mass <= 0) continue;
      // Steady visible crumbling (~52..108 solid mineral chips/sec per dissolving cube!)
      const cubeRate =
        (52.0 + 46.0 * Math.pow(c.mass, 0.35)) *
        (1.0 + 0.18 * Math.abs(this.swirlAngularVel));
      totalEmitRate += cubeRate;
    }

    this.crumbEmitAccum += totalEmitRate * dt;
    const toSpawn = Math.min(18, Math.floor(this.crumbEmitAccum));
    this.crumbEmitAccum -= toSpawn;

    if (toSpawn > 0) {
      let spawned = 0;
      for (let pass = 0; pass < MAX_ACTIVE_CUBES && spawned < toSpawn; pass++) {
        for (let i = 0; i < MAX_ACTIVE_CUBES && spawned < toSpawn; i++) {
          const c = this.cubes[i];
          if (!c.active || !c.submerged || c.mass <= 0) continue;
          const share = Math.max(1, Math.ceil(toSpawn / MAX_ACTIVE_CUBES));
          for (let s = 0; s < share && spawned < toSpawn; s++) {
            this.spawnCrumbOnCube(c, 0);
            spawned++;
          }
        }
      }
    }

    const fluidVel = { vx: 0, vz: 0 };
    const maxR = WATER_RADIUS - 0.022;
    const topY = WATER_REST_Y - 0.026;
    const swirlAbs = Math.abs(this.swirlAngularVel);
    const t = this.simTime;
    let aliveCount = 0;

    for (let i = 0; i < CRUMB_COUNT; i++) {
      if (this.crumbAlive[i] === 0) continue;

      const nextLife = this.crumbLife[i] - dt / this.crumbMaxLife[i];
      if (nextLife <= 0.005) {
        this.crumbAlive[i] = 0;
        this.crumbScale[i] = 0;
        continue;
      }

      this.crumbLife[i] = nextLife;
      const sc = this.crumbInitScale[i] * Math.pow(nextLife, 0.72);
      this.crumbScale[i] = sc;

      let x = this.crumbPosX[i];
      let y = this.crumbPosY[i];
      let z = this.crumbPosZ[i];

      this.evalFluidVelocityXZ(x, z, fluidVel);

      // Heavier mineral granules sink toward the glass floor; fine flecks dance in bubble wakes & Ekman spiral
      const rHoriz = Math.hypot(x, z);
      const nearFloor = smoothstepCpu(BEAKER_INNER_FLOOR_Y + 0.16, BEAKER_INNER_FLOOR_Y + 0.01, y);
      let ekX = 0;
      let ekZ = 0;
      if (rHoriz > 0.03 && this.ekmanStrength > 0.02) {
        const pull = -this.ekmanStrength * nearFloor * 0.36;
        ekX = (x / rHoriz) * pull;
        ekZ = (z / rHoriz) * pull;
      }

      const sinkTerminal = -(0.095 + sc * 13.5) + this.ekmanStrength * 0.18 * Math.exp(-rHoriz * rHoriz * 5.0);
      const flutter = Math.sin(t * (4.3 + (i % 9) * 0.41) + i * 1.7) * 0.045;
      const relaxXZ = 1.0 - Math.exp(-(4.2 + swirlAbs * 0.35) * dt);
      const relaxY = 1.0 - Math.exp(-3.8 * dt);

      this.crumbVelX[i] += (fluidVel.vx + ekX + flutter - this.crumbVelX[i]) * relaxXZ;
      this.crumbVelZ[i] += (fluidVel.vz + ekZ - flutter * 0.8 - this.crumbVelZ[i]) * relaxXZ;
      this.crumbVelY[i] += (sinkTerminal - this.crumbVelY[i]) * relaxY;

      x += this.crumbVelX[i] * dt;
      y += this.crumbVelY[i] * dt;
      z += this.crumbVelZ[i] * dt;

      const floorY = BEAKER_INNER_FLOOR_Y + sc * 0.85;
      if (y < floorY) {
        y = floorY;
        this.crumbVelY[i] = Math.max(0, -this.crumbVelY[i] * 0.08);
        const fDrag = Math.exp(-3.4 * dt);
        this.crumbVelX[i] *= fDrag;
        this.crumbVelZ[i] *= fDrag;
      } else if (y > topY) {
        y = topY;
        if (this.crumbVelY[i] > 0) this.crumbVelY[i] *= -0.15;
      }

      const rNew = Math.hypot(x, z);
      if (rNew > maxR && rNew > 1e-5) {
        const s = maxR / rNew;
        x *= s;
        z *= s;
      }

      const spinRate = this.crumbSpin[i] * (y <= floorY + 0.004 ? 0.25 : 1.0);
      this.crumbRotX[i] += spinRate * dt;
      this.crumbRotY[i] += spinRate * 0.78 * dt;
      this.crumbRotZ[i] += spinRate * 0.54 * dt;

      this.crumbPosX[i] = x;
      this.crumbPosY[i] = y;
      this.crumbPosZ[i] = z;
      aliveCount++;
    }

    this.activeCrumbCount = aliveCount;
  }

  // -------------------------------------------------------------------------
  // 2D Circular Gravity-Capillary Free-Surface Wave PDE
  // -------------------------------------------------------------------------
  private stepWavePde(dt: number): void {
    // Gentle non-repeating capillary micro-shimmer from effervescent surface tension
    this.waveAmbientTimer += dt;
    if (this.waveAmbientTimer > 0.45 && this.activeBubbleCount > 20) {
      this.waveAmbientTimer = 0;
      const ang = this.simTime * 2.3999632;
      const rr = (0.12 + 0.46 * fractCpu(this.simTime * 0.6180339)) * WATER_RADIUS;
      this.injectSplash(Math.cos(ang) * rr, Math.sin(ang) * rr, 0.0028, 0.12);
    }

    const substeps = 2;
    const c2 = 0.21; // CFL-safe (< 0.5)
    const atten = Math.exp(-dt * 2.15);

    for (let s = 0; s < substeps; s++) {
      for (let iy = 0; iy < WAVE_RES; iy++) {
        const row = iy * WAVE_RES;
        const yUp = Math.max(0, iy - 1) * WAVE_RES;
        const yDn = Math.min(WAVE_RES - 1, iy + 1) * WAVE_RES;

        for (let ix = 0; ix < WAVE_RES; ix++) {
          const idx = row + ix;
          if (this.waveInside[idx] === 0) {
            this.waveNext[idx] = 0;
            continue;
          }
          const hC = this.waveCurr[idx];
          const xL = Math.max(0, ix - 1);
          const xR = Math.min(WAVE_RES - 1, ix + 1);

          // Neumann dH/dn = 0 reflection at circular borosilicate glass wall
          const hL = this.waveInside[row + xL] ? this.waveCurr[row + xL] : hC;
          const hR = this.waveInside[row + xR] ? this.waveCurr[row + xR] : hC;
          const hU = this.waveInside[yUp + ix] ? this.waveCurr[yUp + ix] : hC;
          const hD = this.waveInside[yDn + ix] ? this.waveCurr[yDn + ix] : hC;

          const lap = hL + hR + hU + hD - 4.0 * hC;
          const vel = (hC - this.wavePrev[idx]) * atten;
          const next = hC + vel + c2 * lap - 0.011 * hC;
          this.waveNext[idx] = clamp(next, -0.055, 0.055);
        }
      }

      this.wavePrev.set(this.waveCurr);
      this.waveCurr.set(this.waveNext);
    }

    this.packWaveTexture();
  }

  private packWaveTexture(): void {
    const worldCellSize = (2.0 * WATER_RADIUS) / WAVE_RES;
    const invTwoDx = 1.0 / (2.0 * worldCellSize);
    const toHalf = THREE.DataUtils.toHalfFloat;

    for (let iy = 0; iy < WAVE_RES; iy++) {
      const row = iy * WAVE_RES;
      const yUp = Math.max(0, iy - 1) * WAVE_RES;
      const yDn = Math.min(WAVE_RES - 1, iy + 1) * WAVE_RES;

      for (let ix = 0; ix < WAVE_RES; ix++) {
        const idx = row + ix;
        const inside = this.waveInside[idx];
        const hC = inside ? this.waveCurr[idx] : 0;

        const xL = Math.max(0, ix - 1);
        const xR = Math.min(WAVE_RES - 1, ix + 1);
        const hL = this.waveInside[row + xL] ? this.waveCurr[row + xL] : hC;
        const hR = this.waveInside[row + xR] ? this.waveCurr[row + xR] : hC;
        const hU = this.waveInside[yUp + ix] ? this.waveCurr[yUp + ix] : hC;
        const hD = this.waveInside[yDn + ix] ? this.waveCurr[yDn + ix] : hC;

        const dHdx = inside ? (hR - hL) * invTwoDx : 0;
        const dHdz = inside ? (hD - hU) * invTwoDx : 0;

        const base = idx * 4;
        this.waveTexData[base] = toHalf(hC);
        this.waveTexData[base + 1] = toHalf(dHdx);
        this.waveTexData[base + 2] = toHalf(dHdz);
        this.waveTexData[base + 3] = toHalf(inside ? 1.0 : 0.0);
      }
    }
    this.waveTexture.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Multi-Pigment Beer-Lambert Bath Optics & Conservation Telemetry
  // -------------------------------------------------------------------------
  private updateBathOptics(): void {
    // Pure laboratory distilled water baseline extinction (very faint cool mineral clarity)
    let extR = 0.095;
    let extG = 0.062;
    let extB = 0.052;

    let tintR = 0;
    let tintG = 0;
    let tintB = 0;
    let weightSum = 0;

    for (let k = 0; k < DISSOLVE_PIGMENT_CYCLE.length; k++) {
      const m = this.pigmentSoluteMass[k];
      if (m <= 1e-7) continue;
      const p = MINERAL_PALETTE[DISSOLVE_PIGMENT_CYCLE[k]];
      // Beer-Lambert additive molar extinction sigma_total = sum(c_k * sigma_k)
      const cScale = m * 0.42;
      extR += p.extinctionRgb[0] * cScale;
      extG += p.extinctionRgb[1] * cScale;
      extB += p.extinctionRgb[2] * cScale;

      tintR += p.rInfLinear[0] * m;
      tintG += p.rInfLinear[1] * m;
      tintB += p.rInfLinear[2] * m;
      weightSum += m;
    }

    this.bathExtinctionRgb.set(extR, extG, extB);
    if (weightSum > 1e-6) {
      this.bathTintRgb.set(tintR / weightSum, tintG / weightSum, tintB / weightSum);
    } else {
      // Clear borosilicate water baseline (faint calcite-celadon hint)
      this.bathTintRgb.set(0.68, 0.78, 0.80);
    }
  }

  getTotalSolidMass(): number {
    let sum = 0;
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      if (this.cubes[i].active) sum += this.cubes[i].mass;
    }
    return sum;
  }

  /** Bath concentration in cube-equivalent solute mass per unit beaker bath volume. */
  getBathConcentration(): number {
    return this.bathSoluteMass;
  }

  getTelemetry(): Day30TelemetryState {
    let activeCubeCount = 0;
    let solidSum = 0;
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      if (this.cubes[i].active && this.cubes[i].mass > 0) {
        activeCubeCount++;
        solidSum += this.cubes[i].mass;
      }
    }
    return {
      activeCubeCount,
      totalDissolvedCount: this.totalDissolvedCount,
      meanCubeMass: activeCubeCount > 0 ? Number((solidSum / activeCubeCount).toFixed(4)) : 0,
      activeBubbles: this.activeBubbleCount,
      activeCrumbs: this.activeCrumbCount,
      swirlAngularVel: Number(this.swirlAngularVel.toFixed(4)),
      bathConcentration: Number(this.bathSoluteMass.toFixed(4)),
    };
  }

  dispose(): void {
    this.waveTexture.dispose();
  }
}
