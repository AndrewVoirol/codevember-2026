import type {
  ArchLayerSpec,
  MirageCameraState,
  MirageDiagnostics,
  MothRuntime,
  MothSpec,
  Vec3Tuple,
} from './types';

export const BOOK_HALF_W = 0.74;
export const BOOK_HALF_H = 0.59;
export const BACK_SANCTUM_Z = -0.58;

export const FIG_ORIGIN: Vec3Tuple = [0.09, -0.245, -0.45];
export const FIG_YAW_DEG = 28;
const FIG_YAW_RAD = (FIG_YAW_DEG * Math.PI) / 180;
export const FIG_FWD_X = -Math.cos(FIG_YAW_RAD);
export const FIG_FWD_Z = Math.sin(FIG_YAW_RAD);
export const FIG_LEFT_X = Math.sin(FIG_YAW_RAD);
export const FIG_LEFT_Z = Math.cos(FIG_YAW_RAD);

export const CAM0_POS: Vec3Tuple = [0.0, -0.01, 3.12];
export const CAM0_TARGET: Vec3Tuple = [0.0, -0.02, -0.20];
export const CAM0_FOV = 34;

export const CAM1_POS: Vec3Tuple = [-0.52, -0.10, 0.24];
export const CAM1_TARGET: Vec3Tuple = [-0.01, 0.005, -0.41];
export const CAM1_FOV = 43;

export const ARCH_LAYERS: readonly ArchLayerSpec[] = [
  {
    index: 0,
    z: 0.16,
    rx: 0.50,
    yTop: 0.40,
    yBot: -0.34,
    bevelWidth: 0.0055,
    pigmentMode: 'frontCalcite',
    hatchDensity: 0.04,
    hatchAngle: 0.42,
  },
  {
    index: 1,
    z: -0.02,
    rx: 0.48,
    yTop: 0.35,
    yBot: -0.315,
    bevelWidth: 0.006,
    pigmentMode: 'graphiteHatch',
    hatchDensity: 0.85,
    hatchAngle: 0.58,
  },
  {
    index: 2,
    z: -0.072,
    rx: 0.435,
    yTop: 0.315,
    yBot: -0.295,
    bevelWidth: 0.0058,
    pigmentMode: 'rawUmber',
    hatchDensity: 0.56,
    hatchAngle: -0.48,
  },
  {
    index: 3,
    z: -0.124,
    rx: 0.395,
    yTop: 0.285,
    yBot: -0.28,
    bevelWidth: 0.0062,
    pigmentMode: 'lapisUltramarine',
    hatchDensity: 0.24,
    hatchAngle: 0.82,
  },
  {
    index: 4,
    z: -0.176,
    rx: 0.36,
    yTop: 0.255,
    yBot: -0.265,
    bevelWidth: 0.0055,
    pigmentMode: 'charcoalSlate',
    hatchDensity: 0.68,
    hatchAngle: -0.66,
  },
  {
    index: 5,
    z: -0.228,
    rx: 0.328,
    yTop: 0.228,
    yBot: -0.24,
    bevelWidth: 0.0055,
    pigmentMode: 'warmKaolin',
    hatchDensity: 0.16,
    hatchAngle: 0.35,
  },
  {
    index: 6,
    z: -0.28,
    rx: 0.298,
    yTop: 0.20,
    yBot: -0.22,
    bevelWidth: 0.0055,
    pigmentMode: 'innerCalcite',
    hatchDensity: 0.08,
    hatchAngle: -0.25,
  },
];

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / Math.max(1e-6, edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function figToWorld(fwd: number, up: number, latLeft: number, out?: Vec3Tuple): Vec3Tuple {
  const target: Vec3Tuple = out ?? [0, 0, 0];
  target[0] = FIG_ORIGIN[0] + fwd * FIG_FWD_X + latLeft * FIG_LEFT_X;
  target[1] = FIG_ORIGIN[1] + up;
  target[2] = FIG_ORIGIN[2] + fwd * FIG_FWD_Z + latLeft * FIG_LEFT_Z;
  return target;
}

export const FLAME_FLOOR_3D: Vec3Tuple = figToWorld(0.11, 0.022, 0.0);
export const BACKLIGHT_REST_3D: Vec3Tuple = figToWorld(-0.22, 0.26, -0.04);
export const HANDS_HOLLOW_3D: Vec3Tuple = figToWorld(0.172, 0.212, 0.0);
export const HEAD_WORLD_3D: Vec3Tuple = figToWorld(0.065, 0.305, 0.0);
export const HAND_L_WORLD_3D: Vec3Tuple = figToWorld(0.172, 0.206, 0.038);
export const HAND_R_WORLD_3D: Vec3Tuple = figToWorld(0.172, 0.206, -0.038);

// Face normal tilted gently down (-0.38 Y) toward cupped palms
const _fnLen = Math.hypot(FIG_FWD_X * 0.925, -0.38, FIG_FWD_Z * 0.925);
export const FACE_NORMAL_3D: Vec3Tuple = [
  (FIG_FWD_X * 0.925) / _fnLen,
  -0.38 / _fnLen,
  (FIG_FWD_Z * 0.925) / _fnLen,
];

/**
 * Computes exact perspective-ray-locked perch along the ray from CAM0_POS to pSil at z = perchZ.
 */
export function computeRayLockedPerch(
  pSil: readonly [number, number, number],
  perchZ: number,
  camPos: readonly [number, number, number] = CAM0_POS
): { perchPos: Vec3Tuple; perchScale: number } {
  const denom = pSil[2] - camPos[2];
  const s = Math.abs(denom) < 1e-6 ? 1 : (perchZ - camPos[2]) / denom;
  return {
    perchPos: [
      camPos[0] + s * (pSil[0] - camPos[0]),
      camPos[1] + s * (pSil[1] - camPos[1]),
      perchZ,
    ],
    perchScale: s,
  };
}

const RAW_MOTH_SEEDS: ReadonlyArray<{
  perchZ: number;
  fwd: number;
  up: number;
  lat: number;
  phase: number;
  orbitRadius: number;
  orbitHeight: number;
  flapHz: number;
  flightSize: number;
  pigmentBias: number;
  wingTilt: number;
  perchRoll: number;
}> = [
  { perchZ: -0.35, fwd: 0.02, up: 0.315, lat: 0.0, phase: 0.2, orbitRadius: 0.048, orbitHeight: 0.045, flapHz: 8.2, flightSize: 0.026, pigmentBias: 0.15, wingTilt: 0.25, perchRoll: -0.25 },
  { perchZ: -0.26, fwd: -0.025, up: 0.285, lat: 0.0, phase: 1.15, orbitRadius: 0.062, orbitHeight: 0.095, flapHz: 9.4, flightSize: 0.031, pigmentBias: 0.72, wingTilt: -0.2, perchRoll: -0.65 },
  { perchZ: -0.18, fwd: -0.055, up: 0.225, lat: 0.0, phase: 2.05, orbitRadius: 0.068, orbitHeight: 0.145, flapHz: 7.8, flightSize: 0.033, pigmentBias: 0.45, wingTilt: 0.18, perchRoll: -1.05 },
  { perchZ: -0.31, fwd: -0.065, up: 0.145, lat: 0.0, phase: 2.85, orbitRadius: 0.052, orbitHeight: 0.02, flapHz: 10.1, flightSize: 0.022, pigmentBias: 0.88, wingTilt: -0.12, perchRoll: -1.35 },
  { perchZ: -0.12, fwd: 0.055, up: 0.225, lat: 0.01, phase: 3.7, orbitRadius: 0.072, orbitHeight: 0.115, flapHz: 8.7, flightSize: 0.028, pigmentBias: 0.28, wingTilt: 0.32, perchRoll: 0.35 },
  { perchZ: -0.06, fwd: 0.115, up: 0.185, lat: 0.015, phase: 4.55, orbitRadius: 0.056, orbitHeight: 0.068, flapHz: 9.1, flightSize: 0.025, pigmentBias: 0.62, wingTilt: -0.28, perchRoll: 0.65 },
  { perchZ: -0.23, fwd: 0.140, up: 0.145, lat: -0.01, phase: 5.2, orbitRadius: 0.082, orbitHeight: 0.155, flapHz: 7.6, flightSize: 0.030, pigmentBias: 0.92, wingTilt: 0.14, perchRoll: 0.95 },
  { perchZ: 0.01, fwd: 0.155, up: 0.085, lat: 0.0, phase: 0.72, orbitRadius: 0.048, orbitHeight: -0.025, flapHz: 9.8, flightSize: 0.023, pigmentBias: 0.12, wingTilt: -0.22, perchRoll: 1.25 },
  { perchZ: 0.06, fwd: 0.115, up: 0.055, lat: 0.0, phase: 1.68, orbitRadius: 0.042, orbitHeight: 0.012, flapHz: 10.4, flightSize: 0.020, pigmentBias: 0.55, wingTilt: 0.28, perchRoll: 0.1 },
];

export const MOTH_SPECS: readonly MothSpec[] = RAW_MOTH_SEEDS.map((seed, id) => {
  const pSil = figToWorld(seed.fwd, seed.up, seed.lat);
  const { perchPos, perchScale } = computeRayLockedPerch(pSil, seed.perchZ, CAM0_POS);
  return {
    id,
    perchZ: seed.perchZ,
    pSil,
    perchPos,
    perchScale,
    phase: seed.phase,
    orbitRadius: seed.orbitRadius,
    orbitHeight: seed.orbitHeight,
    flapHz: seed.flapHz,
    flightSize: seed.flightSize,
    pigmentBias: seed.pigmentBias,
    wingTilt: seed.wingTilt,
    perchRoll: seed.perchRoll,
  };
});

export function evaluateCameraState(
  uReveal: number,
  smoothPtrX: number,
  smoothPtrY: number,
  out: MirageCameraState
): MirageCameraState {
  const u = clamp01(uReveal);
  const e = u * u * (3 - 2 * u);
  const parScale = 1.0 - 0.52 * e;

  // Smooth lateral bow around the outer arch threshold so the camera sweeps cleanly inside
  const bowX = -0.14 * Math.sin(Math.PI * e);
  const baseX = CAM0_POS[0] * (1 - e) + CAM1_POS[0] * e + bowX;
  const baseY = CAM0_POS[1] * (1 - e) + CAM1_POS[1] * e;
  const baseZ = CAM0_POS[2] * (1 - e) + CAM1_POS[2] * e;

  out.pos[0] = baseX + smoothPtrX * 0.19 * parScale;
  out.pos[1] = baseY + smoothPtrY * 0.13 * parScale;
  out.pos[2] = baseZ;

  out.target[0] = CAM0_TARGET[0] * (1 - e) + CAM1_TARGET[0] * e;
  out.target[1] = CAM0_TARGET[1] * (1 - e) + CAM1_TARGET[1] * e;
  out.target[2] = CAM0_TARGET[2] * (1 - e) + CAM1_TARGET[2] * e;

  out.fovDeg = CAM0_FOV * (1 - e) + CAM1_FOV * e;
  out.easedReveal = e;

  const ang0 = Math.atan2(CAM0_POS[0] - FIG_ORIGIN[0], CAM0_POS[2] - FIG_ORIGIN[2]);
  const angNow = Math.atan2(baseX - FIG_ORIGIN[0], baseZ - FIG_ORIGIN[2]);
  out.arcAngleDeg = Math.abs((angNow - ang0) * (180 / Math.PI));

  return out;
}

export function evaluateMothRuntime(
  spec: MothSpec,
  effectiveAwake: number,
  elapsedSec: number,
  pointerSpeed: number,
  hoverBias: number,
  out: MothRuntime
): MothRuntime {
  const localAwake = clamp01(effectiveAwake * 1.32 - spec.id * 0.085);
  const t = localAwake * localAwake * (3 - 2 * localAwake);
  out.id = spec.id;
  out.awake = t;

  // Flight anchor spirals cleanly in the luminous arch space forward & above her cupped hands
  // (clear of her eyes and forehead)
  const swirlAngle = spec.phase + elapsedSec * (0.95 + (spec.id % 3) * 0.22);
  const swirlCenter: Vec3Tuple = figToWorld(
    0.268 + Math.cos(swirlAngle) * spec.orbitRadius * 0.68,
    0.255 + spec.orbitHeight + Math.sin(elapsedSec * 1.85 + spec.phase) * 0.016,
    0.022 + Math.sin(swirlAngle) * spec.orbitRadius * 1.08
  );

  // Midpoint Bezier lift peels through the warm updraft channel in front of her knees
  const midLift: Vec3Tuple = figToWorld(
    0.15,
    0.13 + spec.orbitHeight * 0.35,
    -0.015 + Math.sin(spec.phase) * 0.025
  );

  const omt = 1 - t;
  const b0 = omt * omt;
  const b1 = 2 * omt * t;
  const b2 = t * t;

  out.pos[0] = b0 * spec.perchPos[0] + b1 * midLift[0] + b2 * swirlCenter[0];
  out.pos[1] = b0 * spec.perchPos[1] + b1 * midLift[1] + b2 * swirlCenter[1];
  out.pos[2] = b0 * spec.perchPos[2] + b1 * midLift[2] + b2 * swirlCenter[2];

  // Tangent along flight orbit (or silhouette contour at rest)
  const tanX = -Math.sin(swirlAngle) * FIG_FWD_X * 0.7 + Math.cos(swirlAngle) * FIG_LEFT_X;
  const tanY = 0.28 + 0.16 * Math.cos(elapsedSec * 1.85 + spec.phase);
  const tanZ = -Math.sin(swirlAngle) * FIG_FWD_Z * 0.7 + Math.cos(swirlAngle) * FIG_LEFT_Z;
  const tl = Math.hypot(tanX, tanY, tanZ) || 1;
  out.tangent[0] = tanX / tl;
  out.tangent[1] = tanY / tl;
  out.tangent[2] = tanZ / tl;

  const restSize = 0.014 * spec.perchScale;
  out.scale = restSize * (1 - t) + spec.flightSize * t;

  // Subtle breathing/parallax rustle while perched; crisp origami flutter in flight
  const rustle = (1 - t) * clamp01(pointerSpeed * 0.18 + hoverBias * 0.25) * 0.06 * Math.sin(elapsedSec * 9.0 + spec.phase);
  const flap = Math.sin(elapsedSec * spec.flapHz + spec.phase * 1.7);
  out.wingFlapRad = (0.04 + rustle) * (1 - t) + (0.52 + 0.44 * flap) * t;
  out.foldProgress = t;
  out.hoverGlow = (1 - t) * clamp01(pointerSpeed * 0.32 + hoverBias * 0.35);

  return out;
}

export function computeRayLockAngularError(
  perchPos: readonly [number, number, number],
  pSil: readonly [number, number, number],
  camPos: readonly [number, number, number] = CAM0_POS
): number {
  const ax = perchPos[0] - camPos[0];
  const ay = perchPos[1] - camPos[1];
  const az = perchPos[2] - camPos[2];
  const bx = pSil[0] - camPos[0];
  const by = pSil[1] - camPos[1];
  const bz = pSil[2] - camPos[2];
  const la = Math.hypot(ax, ay, az) || 1;
  const lb = Math.hypot(bx, by, bz) || 1;
  const cosAng = clamp((ax * bx + ay * by + az * bz) / (la * lb), -1, 1);
  return Math.acos(cosAng);
}

export function computeDiagnosticsMetrics(
  cam: MirageCameraState,
  lightPos: readonly [number, number, number],
  emberPos: readonly [number, number, number],
  uReveal: number,
  uAwake: number,
  effectiveAwake: number,
  lightLift: number,
  sparkPulse: number,
  activeMothCount: number
): MirageDiagnostics {
  const normDot = (a: readonly [number, number, number], b: readonly [number, number, number]) => {
    const la = Math.hypot(a[0], a[1], a[2]) || 1;
    const lb = Math.hypot(b[0], b[1], b[2]) || 1;
    return clamp((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb), -1, 1);
  };

  const toCam: Vec3Tuple = [
    cam.pos[0] - HEAD_WORLD_3D[0],
    cam.pos[1] - HEAD_WORLD_3D[1],
    cam.pos[2] - HEAD_WORLD_3D[2],
  ];
  const toLight: Vec3Tuple = [
    lightPos[0] - HEAD_WORLD_3D[0],
    lightPos[1] - HEAD_WORLD_3D[1],
    lightPos[2] - HEAD_WORLD_3D[2],
  ];
  const toRestLight: Vec3Tuple = [
    BACKLIGHT_REST_3D[0] - HEAD_WORLD_3D[0],
    BACKLIGHT_REST_3D[1] - HEAD_WORLD_3D[1],
    BACKLIGHT_REST_3D[2] - HEAD_WORLD_3D[2],
  ];

  const faceDotCam = normDot([FIG_FWD_X, 0, FIG_FWD_Z], toCam);
  const faceDotLight = normDot(FACE_NORMAL_3D, toLight);
  const faceLightSwingDeg = Math.acos(normDot(toRestLight, toLight)) * (180 / Math.PI);

  const projSepX = (cPos: readonly [number, number, number], cTgt: readonly [number, number, number], fov: number) => {
    const fx = cTgt[0] - cPos[0];
    const fy = cTgt[1] - cPos[1];
    const fz = cTgt[2] - cPos[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    const f0 = fx / fl;
    const f1 = fy / fl;
    const f2 = fz / fl;
    const rl = Math.hypot(-f2, f0) || 1;
    const rx = -f2 / rl;
    const rz = f0 / rl;
    const tFov = Math.tan((fov * Math.PI) / 360) * 1.6;
    const px = (p: readonly [number, number, number]) => {
      const dx = p[0] - cPos[0];
      const dy = p[1] - cPos[1];
      const dz = p[2] - cPos[2];
      const zc = dx * f0 + dy * f1 + dz * f2;
      const xc = dx * rx + dz * rz;
      return xc / (Math.max(0.05, zc) * tFov);
    };
    return Math.abs(px(HAND_L_WORLD_3D) - px(HAND_R_WORLD_3D));
  };

  const sep0 = Math.max(1e-4, projSepX(CAM0_POS, CAM0_TARGET, CAM0_FOV));
  const sepNow = projSepX(cam.pos, cam.target, cam.fovDeg);

  let maxRayErr = 0;
  for (let i = 0; i < MOTH_SPECS.length; i++) {
    const err = computeRayLockAngularError(MOTH_SPECS[i].perchPos, MOTH_SPECS[i].pSil, CAM0_POS);
    if (err > maxRayErr) maxRayErr = err;
  }

  return {
    uReveal,
    uAwake,
    effectiveAwake,
    lightLift,
    sparkPulse,
    activeMothCount,
    cameraPos: [cam.pos[0], cam.pos[1], cam.pos[2]],
    cameraTarget: [cam.target[0], cam.target[1], cam.target[2]],
    fovDeg: cam.fovDeg,
    lightPos: [lightPos[0], lightPos[1], lightPos[2]],
    emberPos: [emberPos[0], emberPos[1], emberPos[2]],
    faceDotCam,
    faceDotLight,
    faceLightSwingDeg,
    handSeparationPxRatio: sepNow / sep0,
    maxRayLockErrorAtRest: maxRayErr,
    arcAngleDeg: cam.arcAngleDeg,
  };
}

/**
 * Pure interactive state machine supporting continuous wheel, pointer drag without quadrant flips,
 * click waking (< 6 px travel), and 100% -> 0% reversibility.
 */
export class MirageController {
  targetReveal = 0;
  uReveal = 0;
  targetAwake = 0;
  uAwake = 0;
  sparkPulse = 0;
  smoothPtrX = 0;
  smoothPtrY = 0;

  private isPointerDown = false;
  private downNdcX = 0;
  private downNdcY = 0;
  private downClientX = 0;
  private downClientY = 0;
  private downReveal = 0;
  private maxTravelPx = 0;

  readonly lightPos: Vec3Tuple = [...BACKLIGHT_REST_3D];
  readonly emberPos: Vec3Tuple = [...FLAME_FLOOR_3D];
  readonly cameraState: MirageCameraState = {
    pos: [...CAM0_POS],
    target: [...CAM0_TARGET],
    fovDeg: CAM0_FOV,
    easedReveal: 0,
    arcAngleDeg: 0,
  };
  readonly moths: MothRuntime[] = MOTH_SPECS.map((m) => ({
    id: m.id,
    awake: 0,
    pos: [...m.perchPos],
    tangent: [1, 0, 0],
    scale: 0.014 * m.perchScale,
    wingFlapRad: 0.04,
    foldProgress: 0,
    hoverGlow: 0,
  }));

  onWheel(deltaY: number): void {
    const nextReveal = clamp01(this.targetReveal + deltaY * 0.00105);
    this.targetReveal = nextReveal;
    if (deltaY < 0) {
      this.targetAwake = Math.min(this.targetAwake, nextReveal * 1.15);
    }
  }

  onPointerDown(ndcX: number, ndcY: number, clientX: number, clientY: number): void {
    this.isPointerDown = true;
    this.downNdcX = ndcX;
    this.downNdcY = ndcY;
    this.downClientX = clientX;
    this.downClientY = clientY;
    this.downReveal = this.targetReveal;
    this.maxTravelPx = 0;
  }

  onPointerMove(ndcX: number, ndcY: number, clientX: number, clientY: number): void {
    if (!this.isPointerDown) return;
    const travel = Math.hypot(clientX - this.downClientX, clientY - this.downClientY);
    if (travel > this.maxTravelPx) this.maxTravelPx = travel;
    if (this.maxTravelPx >= 6) {
      const deltaReveal = -(ndcY - this.downNdcY) * 0.62 + (ndcX - this.downNdcX) * 0.38;
      const nextReveal = clamp01(this.downReveal + deltaReveal);
      if (deltaReveal < 0) {
        this.targetAwake = Math.min(this.targetAwake, nextReveal * 1.15);
      }
      this.targetReveal = nextReveal;
    }
  }

  /**
   * Returns true if pointerup resolved as a discrete click (< 6px drag travel).
   */
  onPointerUp(): boolean {
    if (!this.isPointerDown) return false;
    this.isPointerDown = false;
    if (this.maxTravelPx < 6) {
      this.triggerClickWake();
      return true;
    }
    return false;
  }

  triggerClickWake(): void {
    this.targetAwake = clamp01(this.targetAwake + 0.34);
    this.sparkPulse = Math.min(1.4, this.sparkPulse + 0.75);
    if (this.targetReveal >= 0.08 && this.targetReveal < 0.92) {
      this.targetReveal = clamp01(this.targetReveal + 0.14);
    }
  }

  resetToRest(): void {
    this.targetReveal = 0;
    this.targetAwake = 0;
    this.sparkPulse = 0;
  }

  get effectiveAwake(): number {
    return clamp01(Math.max(this.uAwake, smoothstep(0.08, 0.68, this.uReveal)));
  }

  get lightLift(): number {
    return smoothstep(0.0, 1.0, clamp01(Math.max(this.uReveal * 1.08, this.uAwake * 0.92)));
  }

  step(
    dtSec: number,
    elapsedSec: number,
    ptrNdcX: number,
    ptrNdcY: number,
    ptrSpeed: number,
    ptrDwellSec: number
  ): void {
    const dt = clamp(dtSec, 0.001, 0.1);
    const revealAlpha = 1 - Math.exp(-dt * 8.0);
    const awakeAlpha = 1 - Math.exp(-dt * 4.8);
    const ptrAlpha = 1 - Math.exp(-dt * 7.5);

    this.uReveal += (this.targetReveal - this.uReveal) * revealAlpha;
    this.uAwake += (this.targetAwake - this.uAwake) * awakeAlpha;
    this.smoothPtrX += (ptrNdcX - this.smoothPtrX) * ptrAlpha;
    this.smoothPtrY += (ptrNdcY - this.smoothPtrY) * ptrAlpha;
    this.sparkPulse = Math.max(0, this.sparkPulse - dt * 0.45);

    if (this.uReveal < 0.002 && this.targetReveal === 0) this.uReveal = 0;
    if (this.uAwake < 0.002 && this.targetAwake === 0) this.uAwake = 0;

    evaluateCameraState(this.uReveal, this.smoothPtrX, this.smoothPtrY, this.cameraState);

    const ll = this.lightLift;
    this.lightPos[0] = BACKLIGHT_REST_3D[0] + (HANDS_HOLLOW_3D[0] - BACKLIGHT_REST_3D[0]) * ll;
    this.lightPos[1] = BACKLIGHT_REST_3D[1] + (HANDS_HOLLOW_3D[1] - BACKLIGHT_REST_3D[1]) * ll;
    this.lightPos[2] = BACKLIGHT_REST_3D[2] + (HANDS_HOLLOW_3D[2] - BACKLIGHT_REST_3D[2]) * ll;

    const effAwake = this.effectiveAwake;
    const hoverBias = clamp01(Math.hypot(this.smoothPtrX, this.smoothPtrY) * 0.55 + ptrDwellSec * 0.12);
    for (let i = 0; i < MOTH_SPECS.length; i++) {
      evaluateMothRuntime(MOTH_SPECS[i], effAwake, elapsedSec, ptrSpeed, hoverBias, this.moths[i]);
    }
  }

  getDiagnostics(): MirageDiagnostics {
    let activeMoths = 0;
    for (let i = 0; i < this.moths.length; i++) {
      if (this.moths[i].awake > 0.05) activeMoths++;
    }
    return computeDiagnosticsMetrics(
      this.cameraState,
      this.lightPos,
      this.emberPos,
      this.uReveal,
      this.uAwake,
      this.effectiveAwake,
      this.lightLift,
      this.sparkPulse,
      activeMoths
    );
  }
}
