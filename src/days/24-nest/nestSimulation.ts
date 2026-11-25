import type RAPIER_NS from '@dimforge/rapier3d-compat';
import {
  DOWEL_CHAMFER_FRAC,
  buildOvoidPebbleHullPoints,
  generateReciprocalDomeLayout,
  type CeladonEggSpec,
  type ReciprocalDomeLayout,
  type ReciprocalRodSpec,
} from './reciprocalDome';

export type RapierModule = typeof RAPIER_NS;

export interface NestSimOptions {
  /** Wood-on-wood Coulomb friction coefficient (default 0.68) */
  woodFriction?: number;
  /** Plinth stone/brass friction coefficient (default 0.76) */
  plinthFriction?: number;
  /** Egg-on-wood friction coefficient (default 0.68) */
  eggFriction?: number;
  /** Timber density (default 0.85) */
  woodDensity?: number;
  /** Celadon stone egg density (default 0.90) */
  eggDensity?: number;
  /** Substeps per 60 Hz frame (default 3 -> 180 Hz internal integration) */
  substepsPer60Hz?: number;
  /** Rapier PGS solver iterations (default 28 for crisp sub-mm hardwood contact) */
  solverIterations?: number;
  /** Rapier internal PGS iterations (default 4) */
  internalPgsIterations?: number;
  /** Hardwood contact natural frequency in Hz (default 120 Hz -> stiff timber, zero squash) */
  contactNaturalFrequency?: number;
  /** Hardwood contact penetration allowance in meters (default 0.0001 m) */
  allowedLinearError?: number;
  /** Mild aerodynamic linear damping (default 0.18) */
  linearDamping?: number;
  /** Mild aerodynamic angular damping (default 0.25) */
  angularDamping?: number;
  /** Rod indices to omit on spawn (for adversarial keystone-removal tests) */
  removedRodIds?: readonly number[];
}

export interface Day24TelemetryState {
  intactRodCount: number;
  fallenRodCount: number;
  collapsed: boolean;
  maxRodDrift: number;
  activeGrabId: number | null;
  simTime: number;
  reweaving?: boolean;
  reweaveProgress?: number;
  dustActiveCount?: number;
}

export interface RodBodyRecord {
  readonly spec: ReciprocalRodSpec;
  readonly body: RAPIER_NS.RigidBody;
  readonly collider: RAPIER_NS.Collider;
  removed: boolean;
  abyssCulled: boolean;
  fallen: boolean;
  drift: number;
  drop: number;
}

export interface EggBodyRecord {
  readonly spec: CeladonEggSpec;
  readonly body: RAPIER_NS.RigidBody;
  readonly collider: RAPIER_NS.Collider;
  abyssCulled: boolean;
  drop: number;
}

export const REWEAVE_DURATION_SEC = 0.85;
export const MAX_DUST_MOTES = 64;

const GRAVITY_Y = -9.81;
const FALLEN_DROP_THRESHOLD = 0.16;
const FALLEN_DRIFT_THRESHOLD = 0.26;
const FALLEN_Y_FLOOR = -0.18;
const ABYSS_SLEEP_Y = -8.0;
const COLLAPSED_ROD_THRESHOLD = 6;

const DEFAULT_SIM_OPTIONS: Required<Omit<NestSimOptions, 'removedRodIds'>> = {
  woodFriction: 0.68,
  plinthFriction: 0.76,
  eggFriction: 0.68,
  woodDensity: 0.85,
  eggDensity: 0.90,
  substepsPer60Hz: 3,
  solverIterations: 28,
  internalPgsIterations: 4,
  contactNaturalFrequency: 120,
  allowedLinearError: 0.0001,
  linearDamping: 0.18,
  angularDamping: 0.25,
};

/**
 * Pure quaternion-vector rotation without allocating: writes q * v * q^-1 into `out`.
 */
function rotateVecByQuatOut(
  vx: number,
  vy: number,
  vz: number,
  qx: number,
  qy: number,
  qz: number,
  qw: number,
  out: [number, number, number]
): void {
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
}

/**
 * Zero-allocation shortest-arc quaternion slerp written directly into `out` at `offset`.
 */
function slerpQuatInto(
  ax: number,
  ay: number,
  az: number,
  aw: number,
  bx: number,
  by: number,
  bz: number,
  bw: number,
  t: number,
  out: Float32Array,
  offset: number
): void {
  let dot = ax * bx + ay * by + az * bz + aw * bw;
  let tx = bx;
  let ty = by;
  let tz = bz;
  let tw = bw;
  if (dot < 0) {
    dot = -dot;
    tx = -tx;
    ty = -ty;
    tz = -tz;
    tw = -tw;
  }

  let s0: number;
  let s1: number;
  if (dot > 0.9995) {
    s0 = 1 - t;
    s1 = t;
  } else {
    const theta = Math.acos(Math.min(1, dot));
    const sinTheta = Math.sin(theta);
    s0 = Math.sin((1 - t) * theta) / sinTheta;
    s1 = Math.sin(t * theta) / sinTheta;
  }

  const rx = ax * s0 + tx * s1;
  const ry = ay * s0 + ty * s1;
  const rz = az * s0 + tz * s1;
  const rw = aw * s0 + tw * s1;
  const invLen = 1 / (Math.hypot(rx, ry, rz, rw) || 1);
  out[offset + 0] = rx * invLen;
  out[offset + 1] = ry * invLen;
  out[offset + 2] = rz * invLen;
  out[offset + 3] = rw * invLen;
}

/**
 * Quintic C^2 smootherstep: zero velocity and zero acceleration at both u = 0 and u = 1.
 */
function quinticEase(u: number): number {
  const c = u <= 0 ? 0 : u >= 1 ? 1 : u;
  return c * c * c * (10 + c * (-15 + 6 * c));
}

/**
 * Pure, zero-allocation choreographed spiral re-weave controller (`~0.85 s`).
 * Restores Tier 0 -> Tier 4 dowels along bottom-to-top arched spiral paths followed by
 * the 3 river-pebble eggs settling gently into the birch oculus, handing off to Rapier
 * at exact zero-velocity analytical equilibrium on completion.
 */
export class NestReweaveController {
  readonly durationSec: number;
  readonly rodCount: number;
  readonly eggCount: number;

  readonly rodCurrentPos: Float32Array;
  readonly rodCurrentQuat: Float32Array;
  readonly eggCurrentPos: Float32Array;
  readonly eggCurrentQuat: Float32Array;

  private readonly rodStartPos: Float32Array;
  private readonly rodStartQuat: Float32Array;
  private readonly rodTargetPos: Float32Array;
  private readonly rodTargetQuat: Float32Array;
  private readonly rodArcLift: Float32Array;
  private readonly rodRadialBreath: Float32Array;
  private readonly rodStartTime: Float32Array;
  private readonly rodEndTime: Float32Array;

  private readonly eggStartPos: Float32Array;
  private readonly eggStartQuat: Float32Array;
  private readonly eggTargetPos: Float32Array;
  private readonly eggTargetQuat: Float32Array;
  private readonly eggArcLift: Float32Array;
  private readonly eggStartTime: Float32Array;
  private readonly eggEndTime: Float32Array;

  private active = false;
  private elapsedSec = 0;

  constructor(layout: ReciprocalDomeLayout, durationSec = REWEAVE_DURATION_SEC) {
    this.durationSec = durationSec;
    this.rodCount = layout.rods.length;
    this.eggCount = layout.eggs.length;

    this.rodStartPos = new Float32Array(this.rodCount * 3);
    this.rodStartQuat = new Float32Array(this.rodCount * 4);
    this.rodTargetPos = new Float32Array(this.rodCount * 3);
    this.rodTargetQuat = new Float32Array(this.rodCount * 4);
    this.rodCurrentPos = new Float32Array(this.rodCount * 3);
    this.rodCurrentQuat = new Float32Array(this.rodCount * 4);
    this.rodArcLift = new Float32Array(this.rodCount);
    this.rodRadialBreath = new Float32Array(this.rodCount);
    this.rodStartTime = new Float32Array(this.rodCount);
    this.rodEndTime = new Float32Array(this.rodCount);

    this.eggStartPos = new Float32Array(this.eggCount * 3);
    this.eggStartQuat = new Float32Array(this.eggCount * 4);
    this.eggTargetPos = new Float32Array(this.eggCount * 3);
    this.eggTargetQuat = new Float32Array(this.eggCount * 4);
    this.eggCurrentPos = new Float32Array(this.eggCount * 3);
    this.eggCurrentQuat = new Float32Array(this.eggCount * 4);
    this.eggArcLift = new Float32Array(this.eggCount);
    this.eggStartTime = new Float32Array(this.eggCount);
    this.eggEndTime = new Float32Array(this.eggCount);

    // Precompute staggered bottom-to-top spiral schedule normalized to `durationSec` (0.85 s)
    const tierBaseStart = [0.0, 0.11, 0.23, 0.33, 0.42];
    const tierAzimuthSpan = [0.15, 0.14, 0.11, 0.10, 0.10];
    const tierMoveDur = [0.40, 0.39, 0.37, 0.36, 0.34];
    const scale = durationSec / 0.85;

    for (let i = 0; i < this.rodCount; i++) {
      const spec = layout.rods[i];
      const t = Math.min(4, spec.tier);
      const tierCount = layout.tiers[t]?.count ?? 8;
      const frac = spec.indexInTier / Math.max(1, tierCount);
      const tStart = (tierBaseStart[t] + frac * tierAzimuthSpan[t]) * scale;
      const tEnd = Math.min(durationSec * 0.94, tStart + tierMoveDur[t] * scale);

      this.rodStartTime[i] = tStart;
      this.rodEndTime[i] = tEnd;

      this.rodTargetPos[i * 3 + 0] = spec.center[0];
      this.rodTargetPos[i * 3 + 1] = spec.center[1];
      this.rodTargetPos[i * 3 + 2] = spec.center[2];
      this.rodTargetQuat[i * 4 + 0] = spec.quat.x;
      this.rodTargetQuat[i * 4 + 1] = spec.quat.y;
      this.rodTargetQuat[i * 4 + 2] = spec.quat.z;
      this.rodTargetQuat[i * 4 + 3] = spec.quat.w;

      this.rodCurrentPos[i * 3 + 0] = spec.center[0];
      this.rodCurrentPos[i * 3 + 1] = spec.center[1];
      this.rodCurrentPos[i * 3 + 2] = spec.center[2];
      this.rodCurrentQuat[i * 4 + 0] = spec.quat.x;
      this.rodCurrentQuat[i * 4 + 1] = spec.quat.y;
      this.rodCurrentQuat[i * 4 + 2] = spec.quat.z;
      this.rodCurrentQuat[i * 4 + 3] = spec.quat.w;
    }

    for (let e = 0; e < this.eggCount; e++) {
      const spec = layout.eggs[e];
      const tStart = (0.48 + e * 0.045) * scale;
      const tEnd = Math.min(durationSec, tStart + 0.28 * scale);
      this.eggStartTime[e] = tStart;
      this.eggEndTime[e] = tEnd;

      this.eggTargetPos[e * 3 + 0] = spec.pos[0];
      this.eggTargetPos[e * 3 + 1] = spec.pos[1];
      this.eggTargetPos[e * 3 + 2] = spec.pos[2];
      this.eggTargetQuat[e * 4 + 0] = spec.rot.x;
      this.eggTargetQuat[e * 4 + 1] = spec.rot.y;
      this.eggTargetQuat[e * 4 + 2] = spec.rot.z;
      this.eggTargetQuat[e * 4 + 3] = spec.rot.w;

      this.eggCurrentPos[e * 3 + 0] = spec.pos[0];
      this.eggCurrentPos[e * 3 + 1] = spec.pos[1];
      this.eggCurrentPos[e * 3 + 2] = spec.pos[2];
      this.eggCurrentQuat[e * 4 + 0] = spec.rot.x;
      this.eggCurrentQuat[e * 4 + 1] = spec.rot.y;
      this.eggCurrentQuat[e * 4 + 2] = spec.rot.z;
      this.eggCurrentQuat[e * 4 + 3] = spec.rot.w;
    }
  }

  isActive(): boolean {
    return this.active;
  }

  getProgress(): number {
    if (!this.active) return 0;
    return Math.min(1, this.elapsedSec / Math.max(1e-4, this.durationSec));
  }

  /**
   * Captures current world poses from `sim` and starts the staggered spiral re-weave.
   * Rods or eggs that tumbled far into the abyss stage cleanly from just below the hollow bore
   * so they rise gracefully into place without high-velocity vertical streaks.
   */
  beginFromSim(sim: ReciprocalNestSim): void {
    this.active = true;
    this.elapsedSec = 0;

    for (let i = 0; i < this.rodCount; i++) {
      const rec = sim.rods[i];
      const tx = this.rodTargetPos[i * 3 + 0];
      const ty = this.rodTargetPos[i * 3 + 1];
      const tz = this.rodTargetPos[i * 3 + 2];

      let sx: number;
      let sy: number;
      let sz: number;
      let qx: number;
      let qy: number;
      let qz: number;
      let qw: number;

      if (rec.removed) {
        sx = tx * 1.24;
        sy = -0.46 - rec.spec.tier * 0.05;
        sz = tz * 1.24;
        qx = rec.spec.quat.x;
        qy = rec.spec.quat.y;
        qz = rec.spec.quat.z;
        qw = rec.spec.quat.w;
      } else {
        const tr = rec.body.translation();
        const rq = rec.body.rotation();
        if (tr.y < -1.25 || Math.hypot(tr.x, tr.z) > 3.2) {
          // Stage deep-fallen rods cleanly below the hollow ring with preserved tumble angle
          const rCur = Math.hypot(tr.x, tr.z);
          const dirX = rCur > 0.05 ? tr.x / rCur : tx / (Math.hypot(tx, tz) || 1);
          const dirZ = rCur > 0.05 ? tr.z / rCur : tz / (Math.hypot(tx, tz) || 1);
          const stageR = 0.65 + (i % 7) * 0.085;
          sx = dirX * stageR;
          sy = -0.52 - rec.spec.tier * 0.055;
          sz = dirZ * stageR;
        } else {
          sx = tr.x;
          sy = tr.y;
          sz = tr.z;
        }
        qx = rq.x;
        qy = rq.y;
        qz = rq.z;
        qw = rq.w;
      }

      this.rodStartPos[i * 3 + 0] = sx;
      this.rodStartPos[i * 3 + 1] = sy;
      this.rodStartPos[i * 3 + 2] = sz;
      this.rodStartQuat[i * 4 + 0] = qx;
      this.rodStartQuat[i * 4 + 1] = qy;
      this.rodStartQuat[i * 4 + 2] = qz;
      this.rodStartQuat[i * 4 + 3] = qw;

      this.rodCurrentPos[i * 3 + 0] = sx;
      this.rodCurrentPos[i * 3 + 1] = sy;
      this.rodCurrentPos[i * 3 + 2] = sz;
      this.rodCurrentQuat[i * 4 + 0] = qx;
      this.rodCurrentQuat[i * 4 + 1] = qy;
      this.rodCurrentQuat[i * 4 + 2] = qz;
      this.rodCurrentQuat[i * 4 + 3] = qw;

      const dist = Math.hypot(tx - sx, ty - sy, tz - sz);
      if (dist < 0.03) {
        // Already near rest: graceful subtle spiral breath-lift ripple
        this.rodArcLift[i] = 0.048;
        this.rodRadialBreath[i] = 0.028;
      } else {
        // Displaced/fallen: smooth arched sweep clearing the outer brass lip
        this.rodArcLift[i] = Math.min(0.42, 0.13 + dist * 0.22);
        this.rodRadialBreath[i] = Math.min(0.14, 0.04 + dist * 0.06);
      }
    }

    for (let e = 0; e < this.eggCount; e++) {
      const rec = sim.eggs[e];
      const tx = this.eggTargetPos[e * 3 + 0];
      const ty = this.eggTargetPos[e * 3 + 1];
      const tz = this.eggTargetPos[e * 3 + 2];
      const tr = rec.body.translation();
      const rq = rec.body.rotation();

      let sx = tr.x;
      let sy = tr.y;
      let sz = tr.z;
      if (sy < -1.0 || Math.hypot(sx, sz) > 2.6) {
        sx = tx * 1.45;
        sy = -0.38 - e * 0.07;
        sz = tz * 1.45;
      }

      this.eggStartPos[e * 3 + 0] = sx;
      this.eggStartPos[e * 3 + 1] = sy;
      this.eggStartPos[e * 3 + 2] = sz;
      this.eggStartQuat[e * 4 + 0] = rq.x;
      this.eggStartQuat[e * 4 + 1] = rq.y;
      this.eggStartQuat[e * 4 + 2] = rq.z;
      this.eggStartQuat[e * 4 + 3] = rq.w;

      this.eggCurrentPos[e * 3 + 0] = sx;
      this.eggCurrentPos[e * 3 + 1] = sy;
      this.eggCurrentPos[e * 3 + 2] = sz;
      this.eggCurrentQuat[e * 4 + 0] = rq.x;
      this.eggCurrentQuat[e * 4 + 1] = rq.y;
      this.eggCurrentQuat[e * 4 + 2] = rq.z;
      this.eggCurrentQuat[e * 4 + 3] = rq.w;

      const dist = Math.hypot(tx - sx, ty - sy, tz - sz);
      this.eggArcLift[e] = dist < 0.03 ? 0.065 : Math.min(0.38, 0.16 + dist * 0.20);
    }
  }

  /**
   * Advances the choreography by `dtSec`. Returns `true` on the exact frame the
   * transition finishes (signaling caller to commit clean zero-velocity Rapier reset).
   */
  step(dtSec: number): boolean {
    if (!this.active) return false;
    this.elapsedSec += Math.max(0, dtSec);
    const done = this.elapsedSec >= this.durationSec;
    const tNow = done ? this.durationSec : this.elapsedSec;

    for (let i = 0; i < this.rodCount; i++) {
      const t0 = this.rodStartTime[i];
      const t1 = this.rodEndTime[i];
      const rawU = done ? 1 : tNow <= t0 ? 0 : tNow >= t1 ? 1 : (tNow - t0) / Math.max(1e-4, t1 - t0);
      const e = quinticEase(rawU);
      const arc = Math.sin(Math.PI * e);

      const pOff = i * 3;
      const sx = this.rodStartPos[pOff + 0];
      const sy = this.rodStartPos[pOff + 1];
      const sz = this.rodStartPos[pOff + 2];
      const tx = this.rodTargetPos[pOff + 0];
      const ty = this.rodTargetPos[pOff + 1];
      const tz = this.rodTargetPos[pOff + 2];

      const rTar = Math.hypot(tx, tz) || 1;
      const radialOut = arc * this.rodRadialBreath[i];

      this.rodCurrentPos[pOff + 0] = sx + (tx - sx) * e + (tx / rTar) * radialOut;
      this.rodCurrentPos[pOff + 1] = sy + (ty - sy) * e + arc * this.rodArcLift[i];
      this.rodCurrentPos[pOff + 2] = sz + (tz - sz) * e + (tz / rTar) * radialOut;

      const qOff = i * 4;
      slerpQuatInto(
        this.rodStartQuat[qOff + 0],
        this.rodStartQuat[qOff + 1],
        this.rodStartQuat[qOff + 2],
        this.rodStartQuat[qOff + 3],
        this.rodTargetQuat[qOff + 0],
        this.rodTargetQuat[qOff + 1],
        this.rodTargetQuat[qOff + 2],
        this.rodTargetQuat[qOff + 3],
        e,
        this.rodCurrentQuat,
        qOff
      );
    }

    for (let j = 0; j < this.eggCount; j++) {
      const t0 = this.eggStartTime[j];
      const t1 = this.eggEndTime[j];
      const rawU = done ? 1 : tNow <= t0 ? 0 : tNow >= t1 ? 1 : (tNow - t0) / Math.max(1e-4, t1 - t0);
      const e = quinticEase(rawU);
      const arc = Math.sin(Math.PI * e);

      const pOff = j * 3;
      const sx = this.eggStartPos[pOff + 0];
      const sy = this.eggStartPos[pOff + 1];
      const sz = this.eggStartPos[pOff + 2];
      const tx = this.eggTargetPos[pOff + 0];
      const ty = this.eggTargetPos[pOff + 1];
      const tz = this.eggTargetPos[pOff + 2];

      this.eggCurrentPos[pOff + 0] = sx + (tx - sx) * e;
      this.eggCurrentPos[pOff + 1] = sy + (ty - sy) * e + arc * this.eggArcLift[j];
      this.eggCurrentPos[pOff + 2] = sz + (tz - sz) * e;

      const qOff = j * 4;
      slerpQuatInto(
        this.eggStartQuat[qOff + 0],
        this.eggStartQuat[qOff + 1],
        this.eggStartQuat[qOff + 2],
        this.eggStartQuat[qOff + 3],
        this.eggTargetQuat[qOff + 0],
        this.eggTargetQuat[qOff + 1],
        this.eggTargetQuat[qOff + 2],
        this.eggTargetQuat[qOff + 3],
        e,
        this.eggCurrentQuat,
        qOff
      );
    }

    if (done) {
      this.active = false;
      return true;
    }
    return false;
  }

  /**
   * Computes current maximum Euclidean distance across all rods & eggs from exact target rest pose.
   */
  getMaxErrorToRest(): number {
    let maxErr = 0;
    for (let i = 0; i < this.rodCount; i++) {
      const o = i * 3;
      const d = Math.hypot(
        this.rodCurrentPos[o + 0] - this.rodTargetPos[o + 0],
        this.rodCurrentPos[o + 1] - this.rodTargetPos[o + 1],
        this.rodCurrentPos[o + 2] - this.rodTargetPos[o + 2]
      );
      if (d > maxErr) maxErr = d;
    }
    for (let j = 0; j < this.eggCount; j++) {
      const o = j * 3;
      const d = Math.hypot(
        this.eggCurrentPos[o + 0] - this.eggTargetPos[o + 0],
        this.eggCurrentPos[o + 1] - this.eggTargetPos[o + 1],
        this.eggCurrentPos[o + 2] - this.eggTargetPos[o + 2]
      );
      if (d > maxErr) maxErr = d;
    }
    return maxErr;
  }
}

/**
 * Zero-allocation pre-allocated pool of fine dry wood-dust & limestone-chalk motes
 * released on tap-test wiggles, pinned-dowel extraction scrapes, and stone plinth impacts.
 */
export class NestDustPool {
  readonly capacity: number;
  readonly pos: Float32Array;
  readonly vel: Float32Array;
  readonly rot: Float32Array;
  readonly angVel: Float32Array;
  readonly age: Float32Array;
  readonly life: Float32Array;
  readonly size: Float32Array;
  readonly tone: Float32Array;
  readonly opacity: Float32Array;
  readonly active: Uint8Array;

  private activeCount = 0;
  private cursor = 0;
  private rngState = 0x51a7c3d9;

  constructor(capacity = MAX_DUST_MOTES) {
    this.capacity = capacity;
    this.pos = new Float32Array(capacity * 3);
    this.vel = new Float32Array(capacity * 3);
    this.rot = new Float32Array(capacity * 3);
    this.angVel = new Float32Array(capacity * 3);
    this.age = new Float32Array(capacity);
    this.life = new Float32Array(capacity);
    this.size = new Float32Array(capacity);
    this.tone = new Float32Array(capacity);
    this.opacity = new Float32Array(capacity);
    this.active = new Uint8Array(capacity);
  }

  private nextRand(): number {
    // Deterministic xorshift32 in [0, 1) — zero allocations
    let x = this.rngState | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x;
    return ((x >>> 0) & 0x0fffffff) / 0x10000000;
  }

  getActiveCount(): number {
    return this.activeCount;
  }

  clear(): void {
    this.active.fill(0);
    this.opacity.fill(0);
    this.activeCount = 0;
  }

  /**
   * Emits a subtle localized puff of dry wood-dust or limestone-chalk motes.
   * `materialTone`: 0.0 = dark walnut dust, 0.45 = cedar/birch sawdust, 0.95 = limestone chalk.
   */
  emitBurst(
    originX: number,
    originY: number,
    originZ: number,
    count: number,
    speed: number,
    materialTone: number,
    spreadRadius = 0.025
  ): void {
    for (let n = 0; n < count; n++) {
      const idx = this.cursor;
      this.cursor = (this.cursor + 1) % this.capacity;

      const u1 = this.nextRand();
      const u2 = this.nextRand();
      const u3 = this.nextRand();
      const theta = u1 * Math.PI * 2;
      const elev = (u2 - 0.35) * 1.15;
      const cElev = Math.cos(elev);
      const sElev = Math.sin(elev);
      const rOff = u3 * spreadRadius;

      const pOff = idx * 3;
      this.pos[pOff + 0] = originX + Math.cos(theta) * cElev * rOff;
      this.pos[pOff + 1] = originY + Math.abs(sElev) * rOff * 0.65;
      this.pos[pOff + 2] = originZ + Math.sin(theta) * cElev * rOff;

      const spd = speed * (0.45 + this.nextRand() * 0.85);
      this.vel[pOff + 0] = Math.cos(theta) * cElev * spd;
      this.vel[pOff + 1] = Math.max(0.03, (0.28 + this.nextRand() * 0.72) * spd);
      this.vel[pOff + 2] = Math.sin(theta) * cElev * spd;

      this.rot[pOff + 0] = this.nextRand() * Math.PI * 2;
      this.rot[pOff + 1] = this.nextRand() * Math.PI * 2;
      this.rot[pOff + 2] = this.nextRand() * Math.PI * 2;

      this.angVel[pOff + 0] = (this.nextRand() - 0.5) * 7.5;
      this.angVel[pOff + 1] = (this.nextRand() - 0.5) * 7.5;
      this.angVel[pOff + 2] = (this.nextRand() - 0.5) * 7.5;

      this.age[idx] = 0;
      this.life[idx] = 0.55 + this.nextRand() * 0.65;
      this.size[idx] = 0.58 + this.nextRand() * 0.85;
      this.tone[idx] = Math.max(0, Math.min(1, materialTone + (this.nextRand() - 0.5) * 0.12));
      this.opacity[idx] = 0;
      this.active[idx] = 1;
    }
    this.recountActive();
  }

  private recountActive(): void {
    let cnt = 0;
    for (let i = 0; i < this.capacity; i++) {
      if (this.active[i]) cnt++;
    }
    this.activeCount = cnt;
  }

  /**
   * Advances all active motes with gentle aerodynamic drag, soft gravity settling,
   * and smooth fade-in / fade-out (strictly zero allocations).
   */
  step(dtSec: number): void {
    if (this.activeCount === 0 || dtSec <= 0) return;
    const dt = Math.min(0.05, dtSec);
    const drag = Math.exp(-3.1 * dt);
    let alive = 0;

    for (let i = 0; i < this.capacity; i++) {
      if (!this.active[i]) continue;
      const nextAge = this.age[i] + dt;
      const life = this.life[i];
      if (nextAge >= life) {
        this.active[i] = 0;
        this.opacity[i] = 0;
        continue;
      }
      this.age[i] = nextAge;
      const u = nextAge / life;

      const pOff = i * 3;
      const driftX = Math.sin(nextAge * 4.5 + i * 1.9) * 0.032;
      const driftZ = Math.cos(nextAge * 3.9 + i * 2.3) * 0.032;

      this.vel[pOff + 0] = this.vel[pOff + 0] * drag + driftX * dt;
      this.vel[pOff + 1] = this.vel[pOff + 1] * drag - 0.92 * dt;
      this.vel[pOff + 2] = this.vel[pOff + 2] * drag + driftZ * dt;

      this.pos[pOff + 0] += this.vel[pOff + 0] * dt;
      this.pos[pOff + 1] += this.vel[pOff + 1] * dt;
      this.pos[pOff + 2] += this.vel[pOff + 2] * dt;

      this.rot[pOff + 0] += this.angVel[pOff + 0] * dt;
      this.rot[pOff + 1] += this.angVel[pOff + 1] * dt;
      this.rot[pOff + 2] += this.angVel[pOff + 2] * dt;

      // Crisp soft dust fade-in (first 10%) and gentle airy dissolve (last 55%)
      const fadeIn = u < 0.1 ? u / 0.1 : 1.0;
      const fadeOut = u > 0.45 ? 1.0 - (u - 0.45) / 0.55 : 1.0;
      this.opacity[i] = 0.72 * fadeIn * fadeOut * fadeOut;
      alive++;
    }

    this.activeCount = alive;
  }
}

/**
 * True 3D rigid-body simulation of the 56-dowel reciprocal timber dome + 3 ovoid
 * Celadon & Bone-Calcite river-pebble eggs on a hollow circular plinth ring using
 * `@dimforge/rapier3d-compat` 0.21 (`roundCylinder` dowels + `convexHull` ovoid eggs).
 *
 * Strictly zero artificial weld joints, zero fake freeze latches, zero velocity clamps.
 */
export class ReciprocalNestSim {
  readonly rapier: RapierModule;
  readonly layout: ReciprocalDomeLayout;
  readonly options: Required<Omit<NestSimOptions, 'removedRodIds'>>;

  world!: RAPIER_NS.World;
  rods: RodBodyRecord[] = [];
  eggs: EggBodyRecord[] = [];

  private simTimeSec = 0;
  private accumulatorSec = 0;
  private fixedDt = 1 / 180;
  private isDisposed = false;

  // Active 3D grab-spring state (zero per-frame allocations)
  private activeGrabRodId: number | null = null;
  private readonly grabLocalPt: [number, number, number] = [0, 0, 0];
  private readonly grabTargetWorld: [number, number, number] = [0, 0, 0];
  private readonly scratchRot: [number, number, number] = [0, 0, 0];
  private readonly scratchAxis: [number, number, number] = [0, 1, 0];
  private readonly scratchWorldPt: [number, number, number] = [0, 0, 0];
  private readonly scratchArm: [number, number, number] = [0, 0, 0];
  private readonly scratchPtVel: [number, number, number] = [0, 0, 0];
  private readonly scratchImpulse: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };
  private readonly scratchPoint: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };
  private readonly scratchZeroVec: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };

  private readonly colliderHandleToRodId = new Map<number, number>();
  private readonly eggColliderHandles = new Set<number>();

  // Cached telemetry updated after every step
  private intactRodCount = 0;
  private fallenRodCount = 0;
  private collapsed = false;
  private maxRodDrift = 0;
  private maxRodDrop = 0;

  constructor(
    rapier: RapierModule,
    layout: ReciprocalDomeLayout = generateReciprocalDomeLayout(),
    options: NestSimOptions = {}
  ) {
    this.rapier = rapier;
    this.layout = layout;
    this.options = {
      ...DEFAULT_SIM_OPTIONS,
      ...options,
    };
    this.fixedDt = 1 / (60 * Math.max(1, this.options.substepsPer60Hz));
    this.buildWorld(options.removedRodIds ?? []);
  }

  private buildWorld(removedRodIds: readonly number[]): void {
    const R = this.rapier;
    const o = this.options;
    const world = new R.World({ x: 0, y: GRAVITY_Y, z: 0 });
    world.timestep = this.fixedDt;
    world.integrationParameters.numSolverIterations = o.solverIterations;
    world.integrationParameters.numInternalPgsIterations = o.internalPgsIterations;
    world.integrationParameters.normalizedAllowedLinearError = o.allowedLinearError;
    world.integrationParameters.normalizedPredictionDistance = Math.max(0.0008, o.allowedLinearError * 6);
    // Calibrate Rapier contact stiffness for hardwood-on-hardwood & stone, scaled safely to the
    // active substep rate so single-substep (60 Hz) ablated runs stay strictly within Nyquist stability
    const ip = world.integrationParameters as RAPIER_NS.IntegrationParameters & {
      contact_natural_frequency?: number;
    };
    if (typeof ip.contact_natural_frequency !== 'undefined' || 'contact_natural_frequency' in Object.getPrototypeOf(ip)) {
      const safeMaxFreq = o.substepsPer60Hz <= 1 ? 30 : Math.round(o.substepsPer60Hz * 40);
      ip.contact_natural_frequency = Math.min(o.contactNaturalFrequency, safeMaxFreq);
    }

    // 1. Hollow circular Honed-Limestone & Dark-Brass plinth ring collider (48 harmonic segments)
    //    Center bore `r < plinth.innerRadius (1.12 m)` is 100% open void so collapsing rods & eggs plummet cleanly!
    const plinth = this.layout.plinth;
    const M = plinth.colliderSegments;
    const shelfOuterR = plinth.lipOuterRadius;
    const shelfMidR = 0.5 * (plinth.innerRadius + shelfOuterR);
    const shelfRadWidth = shelfOuterR - plinth.innerRadius;
    const shelfThickness = Math.abs(plinth.baseBottomY);
    const segArcLen = ((2 * Math.PI * shelfOuterR) / M) * 1.06;

    const plinthBody = world.createRigidBody(R.RigidBodyDesc.fixed());
    const lipThick = Math.max(0.055, plinth.lipOuterRadius - plinth.seatOuterRadius);
    const lipMidR = plinth.seatOuterRadius + lipThick * 0.5;
    const lipHalfH = (plinth.lipTopY + 0.01) * 0.5;
    const lipCenterY = plinth.lipTopY - lipHalfH;

    for (let m = 0; m < M; m++) {
      const th = (m / M) * Math.PI * 2;
      const cs = Math.cos(th);
      const sn = Math.sin(th);
      // Map local +X to radial (cos(th), 0, sin(th)) via rotation by -th around +Y
      const q = { x: 0, y: Math.sin(-th * 0.5), z: 0, w: Math.cos(-th * 0.5) };

      const shelfDesc = R.ColliderDesc.cuboid(
        shelfRadWidth * 0.5,
        shelfThickness * 0.5,
        segArcLen * 0.5
      )
        .setTranslation(shelfMidR * cs, -shelfThickness * 0.5, shelfMidR * sn)
        .setRotation(q)
        .setFriction(o.plinthFriction)
        .setRestitution(0.02);
      world.createCollider(shelfDesc, plinthBody);

      const lipDesc = R.ColliderDesc.cuboid(lipThick * 0.5, lipHalfH, segArcLen * 0.55)
        .setTranslation(lipMidR * cs, lipCenterY, lipMidR * sn)
        .setRotation(q)
        .setFriction(o.plinthFriction)
        .setRestitution(0.02);
      world.createCollider(lipDesc, plinthBody);
    }

    // 2. Dynamic Chamfered RoundCylinder Rigid Bodies for all 56 Timber Dowels
    //    Matches `buildDowelGeometry` flat chamfered end-grain profile 1:1 (0.0 mm gap/clip)
    this.colliderHandleToRodId.clear();
    this.eggColliderHandles.clear();
    const removedSet = new Set(removedRodIds);
    const rods: RodBodyRecord[] = [];
    for (let i = 0; i < this.layout.rods.length; i++) {
      const spec = this.layout.rods[i];
      const isRemoved = removedSet.has(spec.id);
      const bodyDesc = R.RigidBodyDesc.dynamic()
        .setTranslation(
          isRemoved ? 0 : spec.center[0],
          isRemoved ? -50 - i : spec.center[1],
          isRemoved ? 0 : spec.center[2]
        )
        .setRotation(spec.quat)
        .setLinearDamping(o.linearDamping)
        .setAngularDamping(o.angularDamping);
      const body = world.createRigidBody(bodyDesc);
      if (isRemoved) {
        body.setEnabled(false);
      }

      const bevelR = spec.radius * DOWEL_CHAMFER_FRAC;
      const colliderDesc = R.ColliderDesc.roundCylinder(
        spec.halfHeight,
        spec.radius - bevelR,
        bevelR
      )
        .setDensity(o.woodDensity)
        .setFriction(o.woodFriction)
        .setFrictionCombineRule(R.CoefficientCombineRule.Average)
        .setRestitution(0.02);
      const collider = world.createCollider(colliderDesc, body);
      this.colliderHandleToRodId.set(collider.handle, spec.id);

      rods.push({
        spec,
        body,
        collider,
        removed: isRemoved,
        abyssCulled: false,
        fallen: false,
        drift: 0,
        drop: 0,
      });
    }

    // 3. Dynamic Convex-Hull Ovoid Rigid Bodies for the 3 Celadon & Bone-Calcite River-Pebble Eggs
    //    Built from exact `buildOvoidPebbleHullPoints` matching `buildPebbleEggGeometry` to 0.0 mm
    const eggs: EggBodyRecord[] = [];
    for (let i = 0; i < this.layout.eggs.length; i++) {
      const spec = this.layout.eggs[i];
      const bodyDesc = R.RigidBodyDesc.dynamic()
        .setTranslation(spec.pos[0], spec.pos[1], spec.pos[2])
        .setRotation(spec.rot)
        .setLinearDamping(o.linearDamping)
        .setAngularDamping(o.angularDamping);
      const body = world.createRigidBody(bodyDesc);

      const hullPoints = buildOvoidPebbleHullPoints(spec.radius, spec.visualScale, spec.seed);
      const hullDesc = R.ColliderDesc.convexHull(hullPoints);
      const colliderDesc = (hullDesc ?? R.ColliderDesc.ball(spec.radius))
        .setDensity(o.eggDensity)
        .setFriction(o.eggFriction)
        .setRestitution(0.02);
      const collider = world.createCollider(colliderDesc, body);
      this.eggColliderHandles.add(collider.handle);

      eggs.push({ spec, body, collider, abyssCulled: false, drop: 0 });
    }

    this.world = world;
    this.rods = rods;
    this.eggs = eggs;
    this.simTimeSec = 0;
    this.accumulatorSec = 0;
    this.activeGrabRodId = null;
    this.refreshTelemetry();
  }

  /**
   * Estimates how tightly pinned/load-bearing rod `rodId` currently is in [0..1]
   * by querying live Rapier contact manifolds and normal impulses (0 = free/unloaded,
   * 1 = heavily pinned beneath upper tiers). Zero per-call allocations.
   */
  estimateRodPinFactor(rodId: number): number {
    const rec = this.rods[rodId];
    if (!rec || rec.removed || rec.fallen || rec.abyssCulled) return 0;

    let totalContacts = 0;
    let upperCount = 0;
    let lowerCount = 0;
    let plinthCount = 0;
    let totalImpulse = 0;

    this.world.contactPairsWith(rec.collider, (other) => {
      let hasActive = false;
      this.world.contactPair(rec.collider, other, (manifold) => {
        const np = manifold.numContacts();
        if (np > 0) hasActive = true;
        for (let k = 0; k < np; k++) {
          totalImpulse += Math.abs(manifold.contactImpulse(k));
        }
      });
      if (!hasActive) return;

      totalContacts++;
      const otherRodId = this.colliderHandleToRodId.get(other.handle);
      if (otherRodId !== undefined) {
        const otherTier = this.rods[otherRodId].spec.tier;
        if (otherTier > rec.spec.tier) upperCount++;
        else if (otherTier < rec.spec.tier) lowerCount++;
      } else if (!this.eggColliderHandles.has(other.handle)) {
        plinthCount++;
      }
    });

    if (totalContacts === 0) return 0;

    const hasLowerSupport = lowerCount > 0 || plinthCount > 0;
    const upperLoadScore =
      upperCount > 0 ? Math.min(1, upperCount * 0.75 + (hasLowerSupport ? 0.25 : 0)) : 0;
    const weightImpulse = Math.max(1e-6, rec.body.mass() * 9.81 * this.fixedDt);
    const excessLoad = Math.max(0, totalImpulse / weightImpulse - 1.2);
    const forceWeight = upperCount > 0 ? 1.0 : 0.22;
    const forceScore = Math.min(1, excessLoad / 3.5) * forceWeight;
    const tierClampBonus = plinthCount > 0 && upperCount > 0 ? 0.15 : 0;
    const driftFactor = Math.max(0, 1.0 - rec.drift / 0.14);

    return Math.max(
      0,
      Math.min(1, (upperLoadScore * 0.62 + forceScore * 0.35 + tierClampBonus) * driftFactor)
    );
  }

  /**
   * Starts a 3D spring-tether grab on rod `rodId` anchored at its local-space point `localPt`.
   */
  startGrab(
    rodId: number,
    localPt: readonly [number, number, number],
    targetWorld: readonly [number, number, number]
  ): boolean {
    const rec = this.rods[rodId];
    if (!rec || rec.removed || rec.abyssCulled) return false;
    this.activeGrabRodId = rodId;
    this.grabLocalPt[0] = localPt[0];
    this.grabLocalPt[1] = localPt[1];
    this.grabLocalPt[2] = localPt[2];
    this.grabTargetWorld[0] = targetWorld[0];
    this.grabTargetWorld[1] = targetWorld[1];
    this.grabTargetWorld[2] = targetWorld[2];
    rec.body.wakeUp();
    return true;
  }

  updateGrabTarget(targetWorld: readonly [number, number, number]): void {
    this.grabTargetWorld[0] = targetWorld[0];
    this.grabTargetWorld[1] = targetWorld[1];
    this.grabTargetWorld[2] = targetWorld[2];
  }

  endGrab(): void {
    this.activeGrabRodId = null;
  }

  /**
   * Applies a crisp tap/poke impulse at a world point on a rod so the user can test
   * whether a timber dowel is loose or tightly locked under structural compression.
   */
  pokeRod(
    rodId: number,
    worldPt: readonly [number, number, number],
    dir: readonly [number, number, number],
    strength = 0.0085
  ): boolean {
    const rec = this.rods[rodId];
    if (!rec || rec.removed || rec.abyssCulled) return false;
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const mass = rec.body.mass();
    const scale = (strength * Math.max(0.001, mass)) / 0.003;
    this.scratchImpulse.x = (dir[0] / len) * scale;
    this.scratchImpulse.y = (dir[1] / len) * scale;
    this.scratchImpulse.z = (dir[2] / len) * scale;
    this.scratchPoint.x = worldPt[0];
    this.scratchPoint.y = worldPt[1];
    this.scratchPoint.z = worldPt[2];
    rec.body.wakeUp();
    rec.body.applyImpulseAtPoint(this.scratchImpulse, this.scratchPoint, true);
    return true;
  }

  /**
   * Removes specified rod IDs cleanly from the simulation (e.g., adversarial keystone extraction).
   */
  removeRods(rodIds: readonly number[]): void {
    for (let i = 0; i < rodIds.length; i++) {
      const id = rodIds[i];
      const rec = this.rods[id];
      if (!rec || rec.removed) continue;
      rec.removed = true;
      rec.body.setEnabled(false);
      if (this.activeGrabRodId === id) {
        this.activeGrabRodId = null;
      }
    }
    // Wake all remaining bodies so support loss propagates immediately with zero stale sleepers
    for (let i = 0; i < this.rods.length; i++) {
      if (!this.rods[i].removed) this.rods[i].body.wakeUp();
    }
    for (let i = 0; i < this.eggs.length; i++) {
      this.eggs[i].body.wakeUp();
    }
    this.refreshTelemetry();
  }

  /**
   * Applies the active 3D damped spring-tether impulse to the currently grabbed dowel,
   * gently biasing pull responsiveness along the dowel's natural longitudinal extraction
   * axis blended with the camera drag plane so Jenga pulls glide smoothly without binding.
   */
  private applyGrabSpringSubstep(dt: number): void {
    if (this.activeGrabRodId === null) return;
    const rec = this.rods[this.activeGrabRodId];
    if (!rec || rec.removed) {
      this.activeGrabRodId = null;
      return;
    }

    const body = rec.body;
    const tr = body.translation();
    const q = body.rotation();
    const lv = body.linvel();
    const av = body.angvel();

    rotateVecByQuatOut(
      this.grabLocalPt[0],
      this.grabLocalPt[1],
      this.grabLocalPt[2],
      q.x,
      q.y,
      q.z,
      q.w,
      this.scratchRot
    );
    // Live longitudinal axis of the grabbed dowel (local +Y rotated into world space)
    rotateVecByQuatOut(0, 1, 0, q.x, q.y, q.z, q.w, this.scratchAxis);

    const wx = tr.x + this.scratchRot[0];
    const wy = tr.y + this.scratchRot[1];
    const wz = tr.z + this.scratchRot[2];
    this.scratchWorldPt[0] = wx;
    this.scratchWorldPt[1] = wy;
    this.scratchWorldPt[2] = wz;

    // Velocity at the world grab point: v_pt = v_cm + omega x r
    this.scratchArm[0] = this.scratchRot[0];
    this.scratchArm[1] = this.scratchRot[1];
    this.scratchArm[2] = this.scratchRot[2];
    this.scratchPtVel[0] = lv.x + (av.y * this.scratchArm[2] - av.z * this.scratchArm[1]);
    this.scratchPtVel[1] = lv.y + (av.z * this.scratchArm[0] - av.x * this.scratchArm[2]);
    this.scratchPtVel[2] = lv.z + (av.x * this.scratchArm[1] - av.y * this.scratchArm[0]);

    let dx = this.grabTargetWorld[0] - wx;
    let dy = this.grabTargetWorld[1] - wy;
    let dz = this.grabTargetWorld[2] - wz;

    // Smoothly cap tether extension so sudden cursor flicks feel weighted rather than explosive
    const dist = Math.hypot(dx, dy, dz);
    const maxStretch = 0.85;
    if (dist > maxStretch) {
      const s = maxStretch / dist;
      dx *= s;
      dy *= s;
      dz *= s;
    }

    // Natural fingertip lift when pulling a Tier-0 rod radially outward over the 4.6 cm brass lip:
    // prevents downward camera-plane drag angles from jamming the dowel butt into the stone shelf or brass wall.
    if (rec.spec.tier === 0 && wy < 0.26) {
      const rLen = Math.hypot(tr.x, tr.z) || 1;
      const outwardPull = (dx * tr.x + dz * tr.z) / rLen;
      if (outwardPull > 0.015) {
        const minLipClearDy = 0.115 - wy + outwardPull * 0.40;
        if (dy < minLipClearDy) {
          dy = minLipClearDy;
        }
      }
    }

    // Decompose pull into axial (along the dowel's natural pull-out path) + transverse camera-plane
    // so sliding sticks out Jenga-style feels silky, weighted, and responsive under full dome load
    const ax = this.scratchAxis[0];
    const ay = this.scratchAxis[1];
    const az = this.scratchAxis[2];
    const dAxial = dx * ax + dy * ay + dz * az;
    const dTransX = dx - dAxial * ax;
    const dTransY = dy - dAxial * ay;
    const dTransZ = dz - dAxial * az;

    const vAxial = this.scratchPtVel[0] * ax + this.scratchPtVel[1] * ay + this.scratchPtVel[2] * az;
    const vTransX = this.scratchPtVel[0] - vAxial * ax;
    const vTransY = this.scratchPtVel[1] - vAxial * ay;
    const vTransZ = this.scratchPtVel[2] - vAxial * az;

    const mass = Math.max(0.0005, body.mass());
    const kAxial = 440.0 * mass;
    const cAxial = 26.0 * mass;
    const kTrans = 310.0 * mass;
    const cTrans = 26.5 * mass;

    const fAxial = dAxial * kAxial - vAxial * cAxial;
    const fx = fAxial * ax + dTransX * kTrans - vTransX * cTrans;
    const fy = fAxial * ay + dTransY * kTrans - vTransY * cTrans + mass * 9.81 * 0.90;
    const fz = fAxial * az + dTransZ * kTrans - vTransZ * cTrans;

    // Clamp max force so heavy pinned sticks slide out cleanly without launching explosively
    const fMag = Math.hypot(fx, fy, fz);
    const maxForce = 385.0 * mass;
    const fScale = fMag > maxForce ? maxForce / fMag : 1.0;

    this.scratchImpulse.x = fx * fScale * dt;
    this.scratchImpulse.y = fy * fScale * dt;
    this.scratchImpulse.z = fz * fScale * dt;
    this.scratchPoint.x = wx;
    this.scratchPoint.y = wy;
    this.scratchPoint.z = wz;

    body.wakeUp();
    body.applyImpulseAtPoint(this.scratchImpulse, this.scratchPoint, true);
  }

  /**
   * Advances physics by exactly one 60 Hz frame (`substepsPer60Hz` fixed substeps).
   */
  stepFixedFrame(): void {
    if (this.isDisposed) return;
    const substeps = Math.max(1, this.options.substepsPer60Hz);
    for (let s = 0; s < substeps; s++) {
      this.applyGrabSpringSubstep(this.fixedDt);
      this.world.step();
      this.simTimeSec += this.fixedDt;
    }
    this.refreshTelemetry();
  }

  /**
   * Advances physics using a fixed-timestep accumulator suitable for variable-refresh displays.
   */
  step(dtSec: number): void {
    if (this.isDisposed) return;
    const clampedDt = Math.min(0.05, Math.max(0, dtSec));
    this.accumulatorSec += clampedDt;
    const maxSubsteps = Math.max(6, this.options.substepsPer60Hz * 3);
    let stepsRun = 0;
    while (this.accumulatorSec >= this.fixedDt && stepsRun < maxSubsteps) {
      this.applyGrabSpringSubstep(this.fixedDt);
      this.world.step();
      this.simTimeSec += this.fixedDt;
      this.accumulatorSec -= this.fixedDt;
      stepsRun++;
    }
    if (stepsRun > 0) {
      this.refreshTelemetry();
    }
  }

  private refreshTelemetry(): void {
    let intact = 0;
    let fallen = 0;
    let maxDrift = 0;
    let maxDrop = 0;

    for (let i = 0; i < this.rods.length; i++) {
      const rec = this.rods[i];
      if (rec.removed) continue;
      const tr = rec.body.translation();
      const dx = tr.x - rec.spec.center[0];
      const dy = tr.y - rec.spec.center[1];
      const dz = tr.z - rec.spec.center[2];
      const drift = Math.hypot(dx, dy, dz);
      const drop = rec.spec.center[1] - tr.y;
      rec.drift = drift;
      rec.drop = drop;

      if (drift > maxDrift) maxDrift = drift;
      if (drop > maxDrop) maxDrop = drop;

      const isFallen =
        drop > FALLEN_DROP_THRESHOLD ||
        drift > FALLEN_DRIFT_THRESHOLD ||
        tr.y < FALLEN_Y_FLOOR;
      rec.fallen = isFallen;
      if (isFallen) fallen++;
      else intact++;

      if (!rec.abyssCulled && this.activeGrabRodId !== rec.spec.id && tr.y < ABYSS_SLEEP_Y) {
        rec.body.setLinvel(this.scratchZeroVec, false);
        rec.body.setAngvel(this.scratchZeroVec, false);
        rec.body.setEnabled(false);
        rec.abyssCulled = true;
      }
    }

    for (let i = 0; i < this.eggs.length; i++) {
      const egg = this.eggs[i];
      const tr = egg.body.translation();
      egg.drop = egg.spec.pos[1] - tr.y;
      if (!egg.abyssCulled && tr.y < ABYSS_SLEEP_Y) {
        egg.body.setLinvel(this.scratchZeroVec, false);
        egg.body.setAngvel(this.scratchZeroVec, false);
        egg.body.setEnabled(false);
        egg.abyssCulled = true;
      }
    }

    this.intactRodCount = intact;
    this.fallenRodCount = fallen;
    this.collapsed = fallen >= COLLAPSED_ROD_THRESHOLD;
    this.maxRodDrift = maxDrift;
    this.maxRodDrop = maxDrop;
  }

  /**
   * Immediately re-weaves and resets the intact reciprocal dome + ovoid Celadon river-pebble eggs
   * (used directly by unit tests and committed cleanly at the end of `NestReweaveController`).
   */
  reset(): void {
    if (this.isDisposed) return;
    this.world.free();
    this.buildWorld([]);
  }

  getMaxRodDrop(): number {
    return this.maxRodDrop;
  }

  getEggDrops(): number[] {
    return this.eggs.map((e) => e.drop);
  }

  getEggPositionsY(): number[] {
    return this.eggs.map((e) => e.body.translation().y);
  }

  /**
   * Canonical read-only telemetry matching `window.__DAY24__.getState()`.
   */
  getState(): Day24TelemetryState {
    return {
      intactRodCount: this.intactRodCount,
      fallenRodCount: this.fallenRodCount,
      collapsed: this.collapsed,
      maxRodDrift: Number(this.maxRodDrift.toFixed(4)),
      activeGrabId: this.activeGrabRodId,
      simTime: Number(this.simTimeSec.toFixed(3)),
    };
  }

  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    this.activeGrabRodId = null;
    this.world.free();
  }
}
