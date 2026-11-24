import * as THREE from 'three/webgpu';

export type VeilApertureMode = 'planar' | 'curved-s' | 'cone';

export const APERTURE_MODES: readonly VeilApertureMode[] = ['planar', 'curved-s', 'cone'] as const;

/** Exact vertical exit plane of the overhead machined brass optical slit jaws. */
export const VEIL_Y_TOP = 1.16;
/** Lower studio feathering extent of the suspended light-and-mist curtain. */
export const VEIL_Y_BOT = -1.48;
export const VEIL_HEIGHT = VEIL_Y_TOP - VEIL_Y_BOT;

/** Mode 0 (Flat Planar Curtain) collimated half-width flare from top slit to bottom. */
export const PLANAR_HALF_W_TOP = 0.86;
export const PLANAR_HALF_W_BOT = 1.28;

/** Mode 1 (Graceful Curved S-Veil) half-width & sinusoidal Z-sweep flare. */
export const CURVED_S_HALF_W_TOP = 0.84;
export const CURVED_S_HALF_W_BOT = 1.22;
export const CURVED_S_AMP_TOP = 0.052;
export const CURVED_S_AMP_BOT = 0.48;

/** Mode 2 (Hollow Elliptical Light Cone) annular top iris to wide elliptical veil base. */
export const CONE_RX_TOP = 0.21;
export const CONE_RZ_TOP = 0.15;
export const CONE_RX_BOT = 1.12;
export const CONE_RZ_BOT = 0.82;

/** Optical half-thickness of the razor-thin collimated light sheet (flares slightly downward). */
export const BEAM_HALF_THICK_TOP = 0.013;
export const BEAM_HALF_THICK_BOT = 0.048;

export interface ModeWeights {
  planar: number;
  curvedS: number;
  cone: number;
}

export interface VeilRayHit {
  u: number;
  v: number;
  distance: number;
  point: THREE.Vector3;
}

const TWO_PI = Math.PI * 2;

/**
 * Evaluates exact 3D point, unit surface normal, and tangent basis on the morphing
 * volumetric light-veil surface for normalized curtain coordinates `(u, v) in [0, 1]^2`.
 * Shared identically between CPU ray-curtain intersection, GPU veil deformation, and dust motes.
 */
export function evalVeilSurface(
  u: number,
  v: number,
  weights: ModeWeights,
  outPos: THREE.Vector3,
  outNormal?: THREE.Vector3,
  outTanU?: THREE.Vector3,
  outTanV?: THREE.Vector3
): void {
  const s = u - 0.5;
  const theta = TWO_PI * s;
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);

  const y = VEIL_Y_TOP - v * VEIL_HEIGHT;

  // Mode 0: Planar collimated curtain
  const hwP = PLANAR_HALF_W_TOP + (PLANAR_HALF_W_BOT - PLANAR_HALF_W_TOP) * v;
  const dHwP = PLANAR_HALF_W_BOT - PLANAR_HALF_W_TOP;
  const xP = 2 * s * hwP;
  const zP = 0;

  // Mode 1: Graceful Curved S-Veil (flush inside lamp throat at v=0, sweeping S-ribbon below)
  const hwS = CURVED_S_HALF_W_TOP + (CURVED_S_HALF_W_BOT - CURVED_S_HALF_W_TOP) * v;
  const dHwS = CURVED_S_HALF_W_BOT - CURVED_S_HALF_W_TOP;
  const ampS = CURVED_S_AMP_TOP + (CURVED_S_AMP_BOT - CURVED_S_AMP_TOP) * v;
  const dAmpS = CURVED_S_AMP_BOT - CURVED_S_AMP_TOP;
  const xS = 2 * s * hwS;
  const zS = ampS * sinT;

  // Mode 2: Hollow Elliptical Light Cone (u=0.5 front +Z, u=0 & u=1 seamless at rear -Z)
  const rxC = CONE_RX_TOP + (CONE_RX_BOT - CONE_RX_TOP) * v;
  const rzC = CONE_RZ_TOP + (CONE_RZ_BOT - CONE_RZ_TOP) * v;
  const dRxC = CONE_RX_BOT - CONE_RX_TOP;
  const dRzC = CONE_RZ_BOT - CONE_RZ_TOP;
  const xC = rxC * sinT;
  const zC = rzC * cosT;

  const w0 = weights.planar;
  const w1 = weights.curvedS;
  const w2 = weights.cone;

  const x = w0 * xP + w1 * xS + w2 * xC;
  const z = w0 * zP + w1 * zS + w2 * zC;
  outPos.set(x, y, z);

  if (!outNormal && !outTanU && !outTanV) return;

  // Exact partial derivatives dP/du and dP/dv
  const dxdu = w0 * (2 * hwP) + w1 * (2 * hwS) + w2 * (rxC * TWO_PI * cosT);
  const dydu = 0;
  const dzdu = w1 * (ampS * TWO_PI * cosT) + w2 * (-rzC * TWO_PI * sinT);

  const dxdv = w0 * (2 * s * dHwP) + w1 * (2 * s * dHwS) + w2 * (dRxC * sinT);
  const dydv = -VEIL_HEIGHT;
  const dzdv = w1 * (dAmpS * sinT) + w2 * (dRzC * cosT);

  if (outTanU) {
    outTanU.set(dxdu, dydu, dzdu);
    const lenU = outTanU.length();
    if (lenU > 1e-6) outTanU.divideScalar(lenU);
    else outTanU.set(1, 0, 0);
  }

  if (outTanV) {
    outTanV.set(dxdv, dydv, dzdv);
    const lenV = outTanV.length();
    if (lenV > 1e-6) outTanV.divideScalar(lenV);
    else outTanV.set(0, -1, 0);
  }

  if (outNormal) {
    // tanV x tanU points strictly outward toward +Z at front (u = 0.5)
    const nx = dydv * dzdu - dzdv * dydu;
    const ny = dzdv * dxdu - dxdv * dzdu;
    const nz = dxdv * dydu - dydv * dxdu;
    const nLen = Math.hypot(nx, ny, nz);
    if (nLen > 1e-6) {
      outNormal.set(nx / nLen, ny / nLen, nz / nLen);
    } else {
      outNormal.set(0, 0, 1);
    }
  }
}

/**
 * Smoothly manages transitions between the 3 optical aperture geometries
 * (`planar`, `curved-s`, `cone`) and exposes normalized blend weights `[w0, w1, w2]`.
 */
export class ApertureModeController {
  private modeIndex = 0;
  readonly weights: ModeWeights = { planar: 1, curvedS: 0, cone: 0 };
  /** Cumulative mechanical focus-drum rotation angle (radians) driven by mode switches. */
  drumAngleRad = 0;
  private targetDrumAngleRad = 0;
  /** Transient 0..1 optical pulse when aperture blades cycle. */
  transitionPulse = 0;

  constructor(initialMode: VeilApertureMode = 'planar') {
    const idx = APERTURE_MODES.indexOf(initialMode);
    this.modeIndex = idx >= 0 ? idx : 0;
    this.snapToCurrent();
  }

  get mode(): VeilApertureMode {
    return APERTURE_MODES[this.modeIndex];
  }

  get index(): number {
    return this.modeIndex;
  }

  setMode(mode: VeilApertureMode): void {
    const idx = APERTURE_MODES.indexOf(mode);
    if (idx === -1 || idx === this.modeIndex) return;
    this.modeIndex = idx;
    this.targetDrumAngleRad += Math.PI * 0.85;
    this.transitionPulse = 1.0;
  }

  cycleMode(): VeilApertureMode {
    this.modeIndex = (this.modeIndex + 1) % APERTURE_MODES.length;
    this.targetDrumAngleRad += Math.PI * 0.85;
    this.transitionPulse = 1.0;
    return this.mode;
  }

  snapToCurrent(): void {
    this.weights.planar = this.modeIndex === 0 ? 1 : 0;
    this.weights.curvedS = this.modeIndex === 1 ? 1 : 0;
    this.weights.cone = this.modeIndex === 2 ? 1 : 0;
    this.drumAngleRad = this.targetDrumAngleRad;
    this.transitionPulse = 0;
  }

  step(dtSec: number): void {
    const dt = Math.min(0.05, Math.max(0.001, dtSec));
    const t0 = this.modeIndex === 0 ? 1 : 0;
    const t1 = this.modeIndex === 1 ? 1 : 0;
    const t2 = this.modeIndex === 2 ? 1 : 0;

    const alpha = 1 - Math.exp(-dt * 6.5);
    this.weights.planar += (t0 - this.weights.planar) * alpha;
    this.weights.curvedS += (t1 - this.weights.curvedS) * alpha;
    this.weights.cone += (t2 - this.weights.cone) * alpha;

    // Normalize strictly so weights always form an exact convex combination (sum === 1)
    const sum = this.weights.planar + this.weights.curvedS + this.weights.cone;
    if (sum > 1e-6) {
      const inv = 1 / sum;
      this.weights.planar *= inv;
      this.weights.curvedS *= inv;
      this.weights.cone *= inv;
    }

    // Snap cleanly to exact target weights once converged (< 0.0025 residual) so Cone mode
    // closes its u = 0 | 1 seam 100% watertight without an asymptotic dark slit!
    const targetWeight =
      this.modeIndex === 0
        ? this.weights.planar
        : this.modeIndex === 1
          ? this.weights.curvedS
          : this.weights.cone;
    if (targetWeight > 0.9975) {
      this.weights.planar = t0;
      this.weights.curvedS = t1;
      this.weights.cone = t2;
    }

    const drumAlpha = 1 - Math.exp(-dt * 8.0);
    this.drumAngleRad += (this.targetDrumAngleRad - this.drumAngleRad) * drumAlpha;
    this.transitionPulse *= Math.exp(-dt * 3.8);
    if (this.transitionPulse < 1e-4) this.transitionPulse = 0;
  }
}

// Pre-allocated zero-GC scratch vectors for Möller-Trumbore ray-curtain intersection
const HIT_U_STEPS = 28;
const HIT_V_STEPS = 20;
const _gridVerts: THREE.Vector3[] = Array.from(
  { length: (HIT_U_STEPS + 1) * (HIT_V_STEPS + 1) },
  () => new THREE.Vector3()
);
const _e1 = new THREE.Vector3();
const _e2 = new THREE.Vector3();
const _h = new THREE.Vector3();
const _sVec = new THREE.Vector3();
const _q = new THREE.Vector3();
const _triHit = { t: 0, b1: 0, b2: 0 };

function intersectTriTwoSided(
  orig: THREE.Vector3,
  dir: THREE.Vector3,
  v0: THREE.Vector3,
  v1: THREE.Vector3,
  v2: THREE.Vector3
): boolean {
  _e1.subVectors(v1, v0);
  _e2.subVectors(v2, v0);
  _h.crossVectors(dir, _e2);
  const a = _e1.dot(_h);
  if (Math.abs(a) < 1e-7) return false;
  const f = 1 / a;
  _sVec.subVectors(orig, v0);
  const b1 = f * _sVec.dot(_h);
  if (b1 < 0 || b1 > 1) return false;
  _q.crossVectors(_sVec, _e1);
  const b2 = f * dir.dot(_q);
  if (b2 < 0 || b1 + b2 > 1) return false;
  const t = f * _e2.dot(_q);
  if (t <= 1e-4) return false;
  _triHit.t = t;
  _triHit.b1 = b1;
  _triHit.b2 = b2;
  return true;
}

/**
 * Exact, zero-GC ray-curtain intersection across all 3 aperture modes and mid-morph states.
 * Returns the nearest front-most hit `(u, v, distance, point)` along `ray`.
 */
export function intersectVeilRay(
  ray: THREE.Ray,
  weights: ModeWeights,
  outHit: VeilRayHit
): boolean {
  // Fast analytical path when strictly in Flat Planar mode (w0 > 0.999)
  if (weights.planar > 0.999) {
    const dz = ray.direction.z;
    if (Math.abs(dz) < 1e-6) return false;
    const t = -ray.origin.z / dz;
    if (t <= 1e-4) return false;
    const hx = ray.origin.x + ray.direction.x * t;
    const hy = ray.origin.y + ray.direction.y * t;
    const v = (VEIL_Y_TOP - hy) / VEIL_HEIGHT;
    if (v < 0 || v > 1) return false;
    const halfW = PLANAR_HALF_W_TOP + (PLANAR_HALF_W_BOT - PLANAR_HALF_W_TOP) * v;
    const u = 0.5 + hx / (2 * halfW);
    if (u < 0 || u > 1) return false;
    outHit.u = u;
    outHit.v = v;
    outHit.distance = t;
    outHit.point.set(hx, hy, 0);
    return true;
  }

  // Coarse bounding-sphere pre-cull (veil sits strictly within radius 2.25 around origin)
  const ox = ray.origin.x;
  const oy = ray.origin.y;
  const oz = ray.origin.z;
  const dx = ray.direction.x;
  const dy = ray.direction.y;
  const dz = ray.direction.z;
  const b = ox * dx + oy * dy + oz * dz;
  const c = ox * ox + oy * oy + oz * oz - 2.35 * 2.35;
  if (b * b - c < 0) return false;

  const stride = HIT_U_STEPS + 1;
  for (let iv = 0; iv <= HIT_V_STEPS; iv++) {
    const v = iv / HIT_V_STEPS;
    for (let iu = 0; iu <= HIT_U_STEPS; iu++) {
      const u = iu / HIT_U_STEPS;
      evalVeilSurface(u, v, weights, _gridVerts[iv * stride + iu]);
    }
  }

  let bestT = Infinity;
  let bestU = 0;
  let bestV = 0;
  const duStep = 1 / HIT_U_STEPS;
  const dvStep = 1 / HIT_V_STEPS;

  for (let iv = 0; iv < HIT_V_STEPS; iv++) {
    const v0Coord = iv * dvStep;
    const v1Coord = (iv + 1) * dvStep;
    const row0 = iv * stride;
    const row1 = (iv + 1) * stride;
    for (let iu = 0; iu < HIT_U_STEPS; iu++) {
      const u0Coord = iu * duStep;
      const u1Coord = (iu + 1) * duStep;
      const p00 = _gridVerts[row0 + iu];
      const p10 = _gridVerts[row0 + iu + 1];
      const p01 = _gridVerts[row1 + iu];
      const p11 = _gridVerts[row1 + iu + 1];

      if (intersectTriTwoSided(ray.origin, ray.direction, p00, p10, p11) && _triHit.t < bestT) {
        bestT = _triHit.t;
        const w0 = 1 - _triHit.b1 - _triHit.b2;
        bestU = w0 * u0Coord + _triHit.b1 * u1Coord + _triHit.b2 * u1Coord;
        bestV = w0 * v0Coord + _triHit.b1 * v0Coord + _triHit.b2 * v1Coord;
      }

      if (intersectTriTwoSided(ray.origin, ray.direction, p00, p11, p01) && _triHit.t < bestT) {
        bestT = _triHit.t;
        const w0 = 1 - _triHit.b1 - _triHit.b2;
        bestU = w0 * u0Coord + _triHit.b1 * u1Coord + _triHit.b2 * u0Coord;
        bestV = w0 * v0Coord + _triHit.b1 * v1Coord + _triHit.b2 * v1Coord;
      }
    }
  }

  if (!Number.isFinite(bestT)) return false;
  outHit.u = Math.max(0, Math.min(1, bestU));
  outHit.v = Math.max(0, Math.min(1, bestV));
  outHit.distance = bestT;
  outHit.point.copy(ray.origin).addScaledVector(ray.direction, bestT);
  return true;
}
