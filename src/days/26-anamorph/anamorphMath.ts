import type { PigmentKey } from '../../core/palette';

export type Vec3Tuple = [number, number, number];

export type ArchitecturalMaterialClass = 0 | 1 | 2;

/**
 * 0 = Board-formed pale Bone-Calcite architectural concrete (walls, piers, beams, portal)
 * 1 = Honed Warm-Kaolin Roman travertine (cantilevered stair treads, terrace dais, landing)
 * 2 = Dark Raw-Umber honed stone plinth & recessed structural spine
 */
export interface ArchitecturalSlab {
  readonly id: string;
  readonly kind: 'plinth' | 'terrace' | 'portal' | 'stair' | 'wall' | 'colonnade' | 'beam' | 'pier';
  readonly materialClass: ArchitecturalMaterialClass;
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly bevel: number;
}

export interface StencilDefinition {
  readonly index: number;
  readonly id: string;
  readonly name: string;
  readonly primaryPigment: PigmentKey;
  readonly secondaryPigment: PigmentKey;
}

export interface VantageBasis {
  readonly pos: Vec3Tuple;
  readonly target: Vec3Tuple;
  readonly forward: Vec3Tuple;
  readonly right: Vec3Tuple;
  readonly up: Vec3Tuple;
  readonly fovDeg: number;
  readonly tanHalfFov: number;
  readonly distance: number;
  readonly stencilScale: number;
}

export interface ProjectiveUvResult {
  readonly u: number;
  readonly v: number;
  readonly zCam: number;
  readonly inFront: boolean;
}

export interface StencilEvaluation {
  readonly primary: number;
  readonly secondary: number;
  readonly total: number;
}

export interface AlignmentMetrics {
  readonly angularErrorDeg: number;
  readonly radialError: number;
  readonly effectiveErrorDeg: number;
  readonly isLockedIn: boolean;
}

export interface Day26Diagnostics {
  readonly alignmentErrorDeg: number;
  readonly isLockedIn: boolean;
  readonly activeStencilIndex: number;
  readonly projectionCount: number;
  readonly vantagePos: Vec3Tuple;
  readonly cameraPos: Vec3Tuple;
}

export interface ProjectionSnapshot {
  readonly vantagePos: Vec3Tuple;
  readonly vantageTarget: Vec3Tuple;
  readonly stencilIndex: number;
  readonly projectionCount: number;
}

export const STAGE_FOV_DEG = 35;
export const VANTAGE_TARGET: Vec3Tuple = [0.0, -0.03, 0.0];

/**
 * Default optical sweet-spot vantage (front-right 3/4 elevated view looking cleanly
 * through the foreground portal opening, across all 6 cantilevered stairs, past the
 * midground baffle & colonnade piers, up to the overhead sky-beams and rear monolith).
 */
export const DEFAULT_VANTAGE_RADIUS = 5.35;
export const DEFAULT_VANTAGE_AZIMUTH_RAD = 0.50; // ~28.6 deg right of +Z
export const DEFAULT_VANTAGE_ELEVATION_RAD = 0.30; // ~17.2 deg above horizon

export function sphericalToCartesian(
  target: readonly [number, number, number],
  radius: number,
  azimuthRad: number,
  elevationRad: number
): Vec3Tuple {
  const cosElev = Math.cos(elevationRad);
  return [
    target[0] + radius * cosElev * Math.sin(azimuthRad),
    target[1] + radius * Math.sin(elevationRad),
    target[2] + radius * cosElev * Math.cos(azimuthRad),
  ];
}

export const DEFAULT_VANTAGE_POS: Vec3Tuple = sphericalToCartesian(
  VANTAGE_TARGET,
  DEFAULT_VANTAGE_RADIUS,
  DEFAULT_VANTAGE_AZIMUTH_RAD,
  DEFAULT_VANTAGE_ELEVATION_RAD
);

/**
 * Initial oblique architectural 3/4 view (~22.6 deg off the sweet spot) so frame 0
 * immediately reveals the fractured Felice Varini pigment bands splayed across
 * separated concrete planes alongside the hovering brass surveyor's sight-ring.
 */
export const INITIAL_CAMERA_POS: Vec3Tuple = sphericalToCartesian(
  VANTAGE_TARGET,
  5.42,
  DEFAULT_VANTAGE_AZIMUTH_RAD - 0.39,
  DEFAULT_VANTAGE_ELEVATION_RAD + 0.13
);

/** Standoff radius of the machined brass optical surveyor's sight-ring along vantageDir (clears max corner 1.82 from every 360 deg angle). */
export const SIGHT_RING_RADIUS_FROM_TARGET = 2.05;

/** Analytical ray-box self-intersection guard bias (exceeds max slab bevel 0.006, well below min slab thickness 0.085). */
export const OCCLUSION_SELF_BIAS = 0.018;
export const OCCLUSION_MIN_T = 0.001;
export const OCCLUSION_FRONT_DOT_MIN = 0.025;

/** Magnetic sweet-spot detent & lock-in thresholds. */
export const LOCK_IN_ANGULAR_DEG = 1.65;
export const LOCK_IN_RADIAL_TOL = 0.18;
export const DETENT_CAPTURE_DEG = 3.6;

export const STENCIL_CONFIGS: readonly StencilDefinition[] = [
  {
    index: 0,
    id: 'concentric-diamond',
    name: 'Concentric Rings & Diamond',
    primaryPigment: 'cinnabarVermilion',
    secondaryPigment: 'yellowOchre',
  },
  {
    index: 1,
    id: 'penrose-tribar',
    name: 'Impossible Penrose Tribar',
    primaryPigment: 'yellowOchre',
    secondaryPigment: 'azuritePrussian',
  },
  {
    index: 2,
    id: 'isometric-cube',
    name: 'Nested Isometric Cube',
    primaryPigment: 'azuritePrussian',
    secondaryPigment: 'realgarAmber',
  },
  {
    index: 3,
    id: 'varini-squares',
    name: 'Architectural Squares & Meridian',
    primaryPigment: 'cinnabarVermilion',
    secondaryPigment: 'azuritePrussian',
  },
] as const;

export const STENCIL_COUNT = STENCIL_CONFIGS.length;

/**
 * Single source of truth for the 30-slab Suspended Brutalist Architectural Courtyard Maquette.
 * Every slab is flush-seated (zero hovering blocks, zero unintentional volume overlaps)
 * and has minimum dimension >= 0.085 for 100% watertight analytical occlusion.
 */
export const ARCHITECTURAL_SLABS: readonly ArchitecturalSlab[] = [
  // 1-2. Chamfered two-tier museum plinth base
  {
    id: 'plinth-sub',
    kind: 'plinth',
    materialClass: 2,
    min: [-1.08, -0.70, -0.98],
    max: [1.08, -0.60, 0.98],
    bevel: 0.006,
  },
  {
    id: 'plinth-deck',
    kind: 'plinth',
    materialClass: 0,
    min: [-1.24, -0.60, -1.12],
    max: [1.24, -0.46, 1.12],
    bevel: 0.006,
  },
  // 3. Raised inner travertine courtyard dais
  {
    id: 'terrace-step',
    kind: 'terrace',
    materialClass: 1,
    min: [-0.94, -0.46, -0.96],
    max: [0.84, -0.36, 0.02],
    bevel: 0.005,
  },
  // 4. Low front courtyard threshold curb / bench
  {
    id: 'curb-front',
    kind: 'terrace',
    materialClass: 1,
    min: [0.08, -0.46, 0.76],
    max: [0.84, -0.30, 0.96],
    bevel: 0.005,
  },
  // 5-7. Foreground-left freestanding brutalist portal archway (wide open doorway)
  {
    id: 'portal-left-pier',
    kind: 'portal',
    materialClass: 0,
    min: [-1.14, -0.46, 0.64],
    max: [-0.94, 0.42, 0.86],
    bevel: 0.005,
  },
  {
    id: 'portal-right-pier',
    kind: 'portal',
    materialClass: 0,
    min: [-0.44, -0.46, 0.64],
    max: [-0.26, 0.42, 0.86],
    bevel: 0.005,
  },
  {
    id: 'portal-lintel',
    kind: 'portal',
    materialClass: 0,
    min: [-1.18, 0.42, 0.62],
    max: [-0.22, 0.58, 0.88],
    bevel: 0.006,
  },
  // 8. Recessed dark-mineral concrete staircase spine wall
  {
    id: 'stair-spine',
    kind: 'stair',
    materialClass: 2,
    min: [-1.12, -0.46, -0.96],
    max: [-0.94, 0.28, 0.42],
    bevel: 0.004,
  },
  // 9-14. Six cantilevered honed-travertine steps ascending front-to-rear
  {
    id: 'stair-step-1',
    kind: 'stair',
    materialClass: 1,
    min: [-0.94, -0.40, 0.24],
    max: [-0.20, -0.315, 0.42],
    bevel: 0.004,
  },
  {
    id: 'stair-step-2',
    kind: 'stair',
    materialClass: 1,
    min: [-0.94, -0.28, 0.10],
    max: [-0.20, -0.195, 0.28],
    bevel: 0.004,
  },
  {
    id: 'stair-step-3',
    kind: 'stair',
    materialClass: 1,
    min: [-0.94, -0.16, -0.04],
    max: [-0.20, -0.075, 0.14],
    bevel: 0.004,
  },
  {
    id: 'stair-step-4',
    kind: 'stair',
    materialClass: 1,
    min: [-0.94, -0.04, -0.18],
    max: [-0.20, 0.045, 0.00],
    bevel: 0.004,
  },
  {
    id: 'stair-step-5',
    kind: 'stair',
    materialClass: 1,
    min: [-0.94, 0.08, -0.32],
    max: [-0.20, 0.165, -0.14],
    bevel: 0.004,
  },
  {
    id: 'stair-step-6',
    kind: 'stair',
    materialClass: 1,
    min: [-0.94, 0.20, -0.46],
    max: [-0.20, 0.285, -0.28],
    bevel: 0.004,
  },
  // 15-16. Upper stair landing terrace & sculptural left parapet wall
  {
    id: 'stair-landing',
    kind: 'terrace',
    materialClass: 1,
    min: [-1.12, 0.28, -0.96],
    max: [-0.20, 0.39, -0.46],
    bevel: 0.005,
  },
  {
    id: 'landing-parapet',
    kind: 'wall',
    materialClass: 0,
    min: [-1.12, 0.39, -0.96],
    max: [-0.96, 0.74, -0.46],
    bevel: 0.005,
  },
  // 17. Offset freestanding midground baffle monolith (supports front sky-beam)
  {
    id: 'baffle-mid-wall',
    kind: 'wall',
    materialClass: 0,
    min: [0.04, -0.46, 0.06],
    max: [0.20, 0.64, 0.54],
    bevel: 0.005,
  },
  // 18-22. Right-flank recessed colonnade loggia (3 piers, entablature lintel, rear screen)
  {
    id: 'colonnade-pier-1',
    kind: 'colonnade',
    materialClass: 0,
    min: [0.84, -0.46, 0.48],
    max: [1.02, 0.62, 0.66],
    bevel: 0.005,
  },
  {
    id: 'colonnade-pier-2',
    kind: 'colonnade',
    materialClass: 0,
    min: [0.84, -0.46, -0.02],
    max: [1.02, 0.62, 0.16],
    bevel: 0.005,
  },
  {
    id: 'colonnade-pier-3',
    kind: 'colonnade',
    materialClass: 0,
    min: [0.84, -0.46, -0.52],
    max: [1.02, 0.62, -0.34],
    bevel: 0.005,
  },
  {
    id: 'colonnade-lintel',
    kind: 'colonnade',
    materialClass: 0,
    min: [0.78, 0.62, -0.96],
    max: [1.08, 0.78, 0.78],
    bevel: 0.006,
  },
  {
    id: 'colonnade-screen',
    kind: 'colonnade',
    materialClass: 0,
    min: [1.06, -0.46, -0.94],
    max: [1.18, 0.62, 0.22],
    bevel: 0.005,
  },
  // 23-26. Tall rear brutalist sanctuary monolith with framed rectangular sky-window
  {
    id: 'rear-wall-lower',
    kind: 'wall',
    materialClass: 0,
    min: [-0.18, -0.36, -0.96],
    max: [0.84, 0.14, -0.80],
    bevel: 0.005,
  },
  {
    id: 'rear-wall-left',
    kind: 'wall',
    materialClass: 0,
    min: [-0.18, 0.14, -0.96],
    max: [0.14, 0.74, -0.80],
    bevel: 0.005,
  },
  {
    id: 'rear-wall-right',
    kind: 'wall',
    materialClass: 0,
    min: [0.56, 0.14, -0.96],
    max: [0.78, 0.74, -0.80],
    bevel: 0.005,
  },
  {
    id: 'rear-wall-crown',
    kind: 'wall',
    materialClass: 0,
    min: [-0.20, 0.74, -0.98],
    max: [0.78, 0.92, -0.78],
    bevel: 0.006,
  },
  // 27-28. Two cantilevered overhead concrete courtyard sky-beams (brise-soleil)
  {
    id: 'sky-beam-front',
    kind: 'beam',
    materialClass: 0,
    min: [-0.68, 0.64, 0.18],
    max: [0.78, 0.76, 0.32],
    bevel: 0.005,
  },
  {
    id: 'sky-beam-rear',
    kind: 'beam',
    materialClass: 0,
    min: [-0.96, 0.62, -0.62],
    max: [0.78, 0.72, -0.48],
    bevel: 0.005,
  },
  // 29. Front-right corner monolithic sentinel pier
  {
    id: 'sentinel-pier',
    kind: 'pier',
    materialClass: 0,
    min: [0.88, -0.46, 0.82],
    max: [1.10, 0.22, 1.00],
    bevel: 0.005,
  },
  // 30. Proud vertical architectural pilaster fin projecting from rear monolith
  {
    id: 'rear-pilaster',
    kind: 'pier',
    materialClass: 0,
    min: [-0.20, -0.36, -0.80],
    max: [-0.06, 0.74, -0.62],
    bevel: 0.005,
  },
] as const;

export const SLAB_COUNT = ARCHITECTURAL_SLABS.length;

function normalize3(x: number, y: number, z: number): Vec3Tuple {
  const len = Math.hypot(x, y, z);
  if (len < 1e-8) return [0, 0, -1];
  return [x / len, y / len, z / len];
}

function cross3(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number
): Vec3Tuple {
  return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / Math.max(1e-7, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Computes the exact orthonormal projective camera basis for a given 3D vantage point.
 * Matches Three.js `PerspectiveCamera.lookAt(target)` with worldUp = (0, 1, 0)
 * (falling back gracefully near zenith/nadir).
 */
export function computeVantageBasis(
  vantagePos: readonly [number, number, number],
  target: readonly [number, number, number] = VANTAGE_TARGET,
  fovDeg: number = STAGE_FOV_DEG
): VantageBasis {
  const dx = target[0] - vantagePos[0];
  const dy = target[1] - vantagePos[1];
  const dz = target[2] - vantagePos[2];
  const distance = Math.max(0.25, Math.hypot(dx, dy, dz));
  const forward = normalize3(dx, dy, dz);

  // World up = (0, 1, 0), fallback to (0, 0, -1) if viewing nearly vertically along Y
  const upRefY = Math.abs(forward[1]) > 0.995 ? 0 : 1;
  const upRefZ = Math.abs(forward[1]) > 0.995 ? -1 : 0;

  const rawRight = cross3(forward[0], forward[1], forward[2], 0, upRefY, upRefZ);
  const right = normalize3(rawRight[0], rawRight[1], rawRight[2]);
  const rawUp = cross3(right[0], right[1], right[2], forward[0], forward[1], forward[2]);
  const up = normalize3(rawUp[0], rawUp[1], rawUp[2]);

  const tanHalfFov = Math.tan((fovDeg * Math.PI) / 360);
  const stencilScale = clamp(DEFAULT_VANTAGE_RADIUS / distance, 0.72, 1.45);

  return {
    pos: [vantagePos[0], vantagePos[1], vantagePos[2]],
    target: [target[0], target[1], target[2]],
    forward,
    right,
    up,
    fovDeg,
    tanHalfFov,
    distance,
    stencilScale,
  };
}

/**
 * Projects any 3D world point into isotropic angular projective stencil coordinates (u, v).
 * When the viewer camera sits at `basis.pos` looking at `basis.target` with FOV `basis.fovDeg`,
 * (u * stencilScale, v * stencilScale) === (ndcX * aspect, ndcY) identically for EVERY depth zCam!
 */
export function projectWorldToStencilUv(
  worldPos: readonly [number, number, number],
  basis: VantageBasis
): ProjectiveUvResult {
  const rx = worldPos[0] - basis.pos[0];
  const ry = worldPos[1] - basis.pos[1];
  const rz = worldPos[2] - basis.pos[2];

  const zCam = rx * basis.forward[0] + ry * basis.forward[1] + rz * basis.forward[2];
  if (zCam <= 0.05) {
    return { u: 100, v: 100, zCam, inFront: false };
  }

  const xCam = rx * basis.right[0] + ry * basis.right[1] + rz * basis.right[2];
  const yCam = rx * basis.up[0] + ry * basis.up[1] + rz * basis.up[2];
  const denom = zCam * basis.tanHalfFov * basis.stencilScale;

  return {
    u: xCam / denom,
    v: yCam / denom,
    zCam,
    inFront: true,
  };
}

/**
 * Guarded slab-method ray-AABB intersection along segment t in (OCCLUSION_MIN_T, maxT).
 * Mirrors the exact zero-branch TSL GPU intersection routine bit-for-bit.
 */
export function rayBoxIntersectSegment(
  origin: readonly [number, number, number],
  dir: readonly [number, number, number],
  maxT: number,
  slab: ArchitecturalSlab
): boolean {
  const invX = 1 / (dir[0] < 0 ? Math.min(dir[0], -1e-6) : Math.max(dir[0], 1e-6));
  const invY = 1 / (dir[1] < 0 ? Math.min(dir[1], -1e-6) : Math.max(dir[1], 1e-6));
  const invZ = 1 / (dir[2] < 0 ? Math.min(dir[2], -1e-6) : Math.max(dir[2], 1e-6));

  const tx1 = (slab.min[0] - origin[0]) * invX;
  const tx2 = (slab.max[0] - origin[0]) * invX;
  const tMinX = Math.min(tx1, tx2);
  const tMaxX = Math.max(tx1, tx2);

  const ty1 = (slab.min[1] - origin[1]) * invY;
  const ty2 = (slab.max[1] - origin[1]) * invY;
  const tMinY = Math.min(ty1, ty2);
  const tMaxY = Math.max(ty1, ty2);

  const tz1 = (slab.min[2] - origin[2]) * invZ;
  const tz2 = (slab.max[2] - origin[2]) * invZ;
  const tMinZ = Math.min(tz1, tz2);
  const tMaxZ = Math.max(tz1, tz2);

  const tEnter = Math.max(tMinX, Math.max(tMinY, tMinZ));
  const tExit = Math.min(tMaxX, Math.min(tMaxY, tMaxZ));

  const clampedEnter = Math.max(tEnter, OCCLUSION_MIN_T);
  return tExit > clampedEnter + 0.0005 && tEnter < maxT;
}

/**
 * Exact line-of-sight visibility test from `vantagePos` to a surface point `(worldPos, worldNormal)`.
 * Rejects back-facing surfaces (`dot(N, dirToVantage) <= OCCLUSION_FRONT_DOT_MIN`)
 * and any point shadowed behind any of the 30 architectural slabs.
 */
export function isPointVisibleFromVantage(
  worldPos: readonly [number, number, number],
  worldNormal: readonly [number, number, number],
  vantagePos: readonly [number, number, number],
  slabs: readonly ArchitecturalSlab[] = ARCHITECTURAL_SLABS
): boolean {
  const vx = vantagePos[0] - worldPos[0];
  const vy = vantagePos[1] - worldPos[1];
  const vz = vantagePos[2] - worldPos[2];
  const dist = Math.hypot(vx, vy, vz);
  if (dist < 0.05) return false;

  const invDist = 1 / dist;
  const toVantageX = vx * invDist;
  const toVantageY = vy * invDist;
  const toVantageZ = vz * invDist;

  // 1. Strict front-facing gate (back walls / faces turned away from projector receive zero paint)
  const nDotV =
    worldNormal[0] * toVantageX +
    worldNormal[1] * toVantageY +
    worldNormal[2] * toVantageZ;
  if (nDotV <= OCCLUSION_FRONT_DOT_MIN) {
    return false;
  }

  // 2. Ray from vantagePos toward worldPos, checking all slabs strictly before `dist - OCCLUSION_SELF_BIAS`
  const rayDir: Vec3Tuple = [-toVantageX, -toVantageY, -toVantageZ];
  const maxT = dist - OCCLUSION_SELF_BIAS;
  for (let i = 0; i < slabs.length; i++) {
    if (rayBoxIntersectSegment(vantagePos, rayDir, maxT, slabs[i])) {
      return false;
    }
  }
  return true;
}

/**
 * Exact 2D Euclidean distance from point (px, py) to line segment (ax, ay)-(bx, by).
 */
export function sdSegment2D(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number
): number {
  const pax = px - ax;
  const pay = py - ay;
  const bax = bx - ax;
  const bay = by - ay;
  const denom = Math.max(1e-8, bax * bax + bay * bay);
  const h = clamp((pax * bax + pay * bay) / denom, 0, 1);
  return Math.hypot(pax - bax * h, pay - bay * h);
}

/**
 * Helper converting L=7 Penrose triangular lattice coordinates (i, j) to centered 2D stencil coordinates.
 */
function penroseLatticeToUv(i: number, j: number): [number, number] {
  const scale = 0.132;
  const x = (i - 7 / 3 + 0.5 * (j - 7 / 3)) * scale;
  const y = (j - 7 / 3) * (Math.sqrt(3) * 0.5) * scale;
  return [x, y];
}

// Precomputed exact Penrose tribar segments (3 arms x 3 segments, closed L=7 impossible topology)
const PENROSE_OUTER_SEGMENTS: ReadonlyArray<readonly [number, number, number, number]> = (() => {
  const segs: Array<[number, number, number, number]> = [];
  const rot120 = (i: number, j: number): [number, number] => [7 - i - j, i];
  const baseOuter: Array<[[number, number], [number, number]]> = [
    [[0, 0], [7, 0]],
    [[0, 1], [5, 1]],
  ];
  for (const [a0, b0] of baseOuter) {
    let a = a0;
    let b = b0;
    for (let k = 0; k < 3; k++) {
      const [ax, ay] = penroseLatticeToUv(a[0], a[1]);
      const [bx, by] = penroseLatticeToUv(b[0], b[1]);
      segs.push([ax, ay, bx, by]);
      a = rot120(a[0], a[1]);
      b = rot120(b[0], b[1]);
    }
  }
  return segs;
})();

const PENROSE_INNER_SEGMENTS: ReadonlyArray<readonly [number, number, number, number]> = (() => {
  const segs: Array<[number, number, number, number]> = [];
  const rot120 = (i: number, j: number): [number, number] => [7 - i - j, i];
  let a: [number, number] = [1, 2];
  let b: [number, number] = [3, 2];
  for (let k = 0; k < 3; k++) {
    const [ax, ay] = penroseLatticeToUv(a[0], a[1]);
    const [bx, by] = penroseLatticeToUv(b[0], b[1]);
    segs.push([ax, ay, bx, by]);
    a = rot120(a[0], a[1]);
    b = rot120(b[0], b[1]);
  }
  return segs;
})();

export const PENROSE_PRIMARY_SEGS = PENROSE_OUTER_SEGMENTS;
export const PENROSE_SECONDARY_SEGS = PENROSE_INNER_SEGMENTS;

/**
 * Evaluates the active Felice Varini geometric stencil in isotropic projective (u, v) space.
 * Returns { primary, secondary, total } in [0, 1], matching the TSL fragment evaluator.
 */
export function evaluateStencilCpu(
  u: number,
  v: number,
  stencilIndex: number,
  aaWidth = 0.003
): StencilEvaluation {
  const idx = ((Math.round(stencilIndex) % STENCIL_COUNT) + STENCIL_COUNT) % STENCIL_COUNT;
  const edge = Math.max(1e-4, aaWidth);
  const strokeMask = (dist: number, halfWidth: number): number =>
    1 - smoothstep(halfWidth - edge, halfWidth + edge, dist);

  let primary = 0;
  let secondary = 0;

  if (idx === 0) {
    // Stencil 0: Concentric Rings (Cinnabar) & Rotated Diamond Frame (Yellow Ochre)
    const r = Math.hypot(u, v);
    const dRing = Math.min(Math.abs(r - 0.43), Math.min(Math.abs(r - 0.285), Math.abs(r - 0.14)));
    primary = strokeMask(dRing, 0.021);

    const dDiamond = Math.abs((Math.abs(u) + Math.abs(v)) * Math.SQRT1_2 - 0.295);
    const dTickX = sdSegment2D(Math.abs(u), Math.abs(v), 0.465, 0, 0.535, 0);
    const dTickY = sdSegment2D(Math.abs(u), Math.abs(v), 0, 0.465, 0, 0.535);
    const dSec = Math.min(dDiamond, Math.min(dTickX, dTickY));
    secondary = strokeMask(dSec, 0.013);
  } else if (idx === 1) {
    // Stencil 1: Impossible Penrose Tribar (Yellow Ochre + Azurite Prussian)
    let dPri = 1e9;
    for (let s = 0; s < PENROSE_OUTER_SEGMENTS.length; s++) {
      const seg = PENROSE_OUTER_SEGMENTS[s];
      dPri = Math.min(dPri, sdSegment2D(u, v, seg[0], seg[1], seg[2], seg[3]));
    }
    let dSec = 1e9;
    for (let s = 0; s < PENROSE_INNER_SEGMENTS.length; s++) {
      const seg = PENROSE_INNER_SEGMENTS[s];
      dSec = Math.min(dSec, sdSegment2D(u, v, seg[0], seg[1], seg[2], seg[3]));
    }
    primary = strokeMask(dPri, 0.017);
    secondary = strokeMask(dSec, 0.015);
  } else if (idx === 2) {
    // Stencil 2: Nested Isometric Cube (Azurite Prussian + Realgar Amber)
    const R = 0.44;
    const hX = R * 0.86602540378; // cos(30 deg)
    const hY = R * 0.5; // sin(30 deg)
    const hexVerts: readonly [number, number][] = [
      [0, R],
      [hX, hY],
      [hX, -hY],
      [0, -R],
      [-hX, -hY],
      [-hX, hY],
    ];
    let dPri = 1e9;
    for (let k = 0; k < 6; k++) {
      const a = hexVerts[k];
      const b = hexVerts[(k + 1) % 6];
      dPri = Math.min(dPri, sdSegment2D(u, v, a[0], a[1], b[0], b[1]));
    }
    // 3 interior Y-spokes creating the 3D isometric cube vertex
    dPri = Math.min(dPri, sdSegment2D(u, v, 0, 0, 0, -R));
    dPri = Math.min(dPri, sdSegment2D(u, v, 0, 0, hX, hY));
    dPri = Math.min(dPri, sdSegment2D(u, v, 0, 0, -hX, hY));
    primary = strokeMask(dPri, 0.018);

    // Inner nested isometric hexagon + opposite center spokes in Realgar Amber
    const rIn = 0.22;
    const ix = rIn * 0.86602540378;
    const iy = rIn * 0.5;
    const innerVerts: readonly [number, number][] = [
      [0, rIn],
      [ix, iy],
      [ix, -iy],
      [0, -rIn],
      [-ix, -iy],
      [-ix, iy],
    ];
    let dSec = 1e9;
    for (let k = 0; k < 6; k++) {
      const a = innerVerts[k];
      const b = innerVerts[(k + 1) % 6];
      dSec = Math.min(dSec, sdSegment2D(u, v, a[0], a[1], b[0], b[1]));
    }
    secondary = strokeMask(dSec, 0.013);
  } else {
    // Stencil 3: Architectural Nested Squares & Surveyor Meridian (Cinnabar + Azurite)
    const ax = Math.abs(u);
    const ay = Math.abs(v);
    const dSqOuter = Math.abs(Math.max(ax, ay) - 0.40);
    const dSqInner = Math.abs(Math.max(ax, ay) - 0.19);
    const dSqRot = Math.abs((ax + ay) * Math.SQRT1_2 - 0.30);
    const dPri = Math.min(dSqOuter, Math.min(dSqInner, dSqRot));
    primary = strokeMask(dPri, 0.018);

    const dMerX1 = sdSegment2D(ax, ay, 0.44, 0, 0.54, 0);
    const dMerY1 = sdSegment2D(ax, ay, 0, 0.44, 0, 0.54);
    const dCenterX = sdSegment2D(ax, ay, 0, 0, 0.11, 0);
    const dCenterY = sdSegment2D(ax, ay, 0, 0, 0, 0.11);
    const dSec = Math.min(Math.min(dMerX1, dMerY1), Math.min(dCenterX, dCenterY));
    secondary = strokeMask(dSec, 0.012);
  }

  // Primary takes clean precedence over secondary at crossings
  const cleanSecondary = secondary * (1 - primary);
  const total = clamp(primary + cleanSecondary, 0, 1);
  return { primary, secondary: cleanSecondary, total };
}

/**
 * Full CPU projective paint-mask evaluator combining front-facing test,
 * exact 30-slab analytical ray-box occlusion, and projective stencil SDF.
 */
export function evaluateProjectedPaintCpu(
  worldPos: readonly [number, number, number],
  worldNormal: readonly [number, number, number],
  basis: VantageBasis,
  stencilIndex: number,
  slabs: readonly ArchitecturalSlab[] = ARCHITECTURAL_SLABS
): StencilEvaluation {
  if (!isPointVisibleFromVantage(worldPos, worldNormal, basis.pos, slabs)) {
    return { primary: 0, secondary: 0, total: 0 };
  }
  const proj = projectWorldToStencilUv(worldPos, basis);
  if (!proj.inFront) {
    return { primary: 0, secondary: 0, total: 0 };
  }
  return evaluateStencilCpu(proj.u, proj.v, stencilIndex);
}

/**
 * Computes exact angular & radial alignment error between the live viewer camera
 * and the active projective sweet-spot vantage point.
 */
export function computeAlignmentMetrics(
  cameraPos: readonly [number, number, number],
  vantagePos: readonly [number, number, number],
  target: readonly [number, number, number] = VANTAGE_TARGET
): AlignmentMetrics {
  const cx = cameraPos[0] - target[0];
  const cy = cameraPos[1] - target[1];
  const cz = cameraPos[2] - target[2];
  const cLen = Math.max(1e-6, Math.hypot(cx, cy, cz));

  const vx = vantagePos[0] - target[0];
  const vy = vantagePos[1] - target[1];
  const vz = vantagePos[2] - target[2];
  const vLen = Math.max(1e-6, Math.hypot(vx, vy, vz));

  const cosAng = clamp((cx * vx + cy * vy + cz * vz) / (cLen * vLen), -1, 1);
  const angularErrorDeg = (Math.acos(cosAng) * 180) / Math.PI;
  const radialError = Math.abs(cLen - vLen);
  const radialEquivDeg = (Math.atan2(radialError * 0.25, vLen) * 180) / Math.PI;
  const effectiveErrorDeg = Math.hypot(angularErrorDeg, radialEquivDeg);
  const isLockedIn =
    angularErrorDeg <= LOCK_IN_ANGULAR_DEG && radialError <= LOCK_IN_RADIAL_TOL;

  return {
    angularErrorDeg,
    radialError,
    effectiveErrorDeg,
    isLockedIn,
  };
}

/**
 * Computes the 3D position and outward axis of the machined brass surveyor's sight-ring
 * along the active optical sightline just outside the architectural maquette.
 */
export function computeSightRingPose(basis: VantageBasis): {
  position: Vec3Tuple;
  lookTarget: Vec3Tuple;
} {
  // Ray from target toward vantagePos is -basis.forward
  const px = basis.target[0] - basis.forward[0] * SIGHT_RING_RADIUS_FROM_TARGET;
  const py = basis.target[1] - basis.forward[1] * SIGHT_RING_RADIUS_FROM_TARGET;
  const pz = basis.target[2] - basis.forward[2] * SIGHT_RING_RADIUS_FROM_TARGET;
  return {
    position: [px, py, pz],
    lookTarget: [basis.pos[0], basis.pos[1], basis.pos[2]],
  };
}

/**
 * Pure deterministic controller for Day 26 - Anamorph:
 * manages projective vantage state, live re-projection history (for zero-latency click
 * + clean double-click rollback), magnetic detent cushion, lock-in glide, and scan flash.
 */
export class AnamorphController {
  vantageBasis: VantageBasis;
  activeStencilIndex = 0;
  projectionCount = 1;

  cameraPos: Vec3Tuple = [...INITIAL_CAMERA_POS];
  cameraTarget: Vec3Tuple = [...VANTAGE_TARGET];

  alignmentErrorDeg = 0;
  radialError = 0;
  isLockedIn = false;

  /** Optical scan-flash progress in [0, 1]; 1 = settled, < 1 = actively sweeping. */
  flashProgress = 1;
  /** Visibility factor [0, 1] for the brass optical sight-ring (fades cleanly at lock-in). */
  sightRingAlpha = 1;

  isDragging = false;
  glideActive = false;
  private glideStartPos: Vec3Tuple = [...INITIAL_CAMERA_POS];
  private glideElapsedSec = 0;
  private readonly glideDurationSec = 0.72;
  private wheelSuppressSec = 0;

  private prevProjection: ProjectionSnapshot | null = null;
  private lastProjectionTimeMs = -1e9;

  constructor() {
    this.vantageBasis = computeVantageBasis(DEFAULT_VANTAGE_POS, VANTAGE_TARGET, STAGE_FOV_DEG);
    this.syncAlignment();
  }

  private syncAlignment(): void {
    const m = computeAlignmentMetrics(
      this.cameraPos,
      this.vantageBasis.pos,
      this.vantageBasis.target
    );
    this.alignmentErrorDeg = m.angularErrorDeg;
    this.radialError = m.radialError;
    this.isLockedIn = m.isLockedIn;
    // Brass sight-ring smoothly dissolves as viewer enters the optical eyepiece
    this.sightRingAlpha = smoothstep(1.2, 7.5, m.angularErrorDeg);
  }

  setCameraPos(x: number, y: number, z: number): void {
    this.cameraPos[0] = x;
    this.cameraPos[1] = y;
    this.cameraPos[2] = z;
    this.syncAlignment();
  }

  notifyDragStart(): void {
    this.isDragging = true;
    this.glideActive = false;
  }

  notifyDragEnd(): void {
    this.isDragging = false;
  }

  notifyWheel(): void {
    this.glideActive = false;
    this.wheelSuppressSec = 1.25;
  }

  /**
   * Projects and bakes a fresh Felice Varini mineral stencil from `fromCameraPos`,
   * advancing to the next geometric stencil and triggering the optical projection scan-flash.
   */
  projectFromViewpoint(
    fromCameraPos: readonly [number, number, number],
    nowMs = 0,
    explicitStencilIndex?: number
  ): void {
    this.prevProjection = {
      vantagePos: [...this.vantageBasis.pos],
      vantageTarget: [...this.vantageBasis.target],
      stencilIndex: this.activeStencilIndex,
      projectionCount: this.projectionCount,
    };
    this.lastProjectionTimeMs = nowMs;

    this.activeStencilIndex =
      explicitStencilIndex !== undefined
        ? ((explicitStencilIndex % STENCIL_COUNT) + STENCIL_COUNT) % STENCIL_COUNT
        : (this.activeStencilIndex + 1) % STENCIL_COUNT;
    this.projectionCount += 1;
    this.vantageBasis = computeVantageBasis(fromCameraPos, this.cameraTarget, STAGE_FOV_DEG);
    this.flashProgress = 0;
    this.glideActive = false;

    this.cameraPos[0] = fromCameraPos[0];
    this.cameraPos[1] = fromCameraPos[1];
    this.cameraPos[2] = fromCameraPos[2];
    this.syncAlignment();
  }

  /**
   * Restores the previous stencil projection if a double-click occurred immediately
   * after a speculative single-click projection on the building.
   */
  rollbackRecentProjectionIfWithin(nowMs: number, windowMs = 360): boolean {
    if (!this.prevProjection || nowMs - this.lastProjectionTimeMs > windowMs) {
      return false;
    }
    const snap = this.prevProjection;
    this.vantageBasis = computeVantageBasis(snap.vantagePos, snap.vantageTarget, STAGE_FOV_DEG);
    this.activeStencilIndex = snap.stencilIndex;
    this.projectionCount = snap.projectionCount;
    this.flashProgress = 1;
    this.prevProjection = null;
    this.syncAlignment();
    return true;
  }

  /**
   * Starts a smooth spherical + radial camera glide directly into exact optical lock-in.
   */
  triggerLockInGlide(fromCameraPos?: readonly [number, number, number]): void {
    if (fromCameraPos) {
      this.cameraPos[0] = fromCameraPos[0];
      this.cameraPos[1] = fromCameraPos[1];
      this.cameraPos[2] = fromCameraPos[2];
    }
    this.glideStartPos = [this.cameraPos[0], this.cameraPos[1], this.cameraPos[2]];
    this.glideElapsedSec = 0;
    this.glideActive = true;
    this.wheelSuppressSec = 0;
  }

  /**
   * Advances scan-flash, smooth lock-in glide, and gentle magnetic detent cushioning.
   * Returns `true` if `outCameraPos` was modified by glide or magnetic detent.
   */
  step(
    dtSec: number,
    liveCameraPos: readonly [number, number, number],
    outCameraPos: Vec3Tuple
  ): boolean {
    const dt = Math.max(0, Math.min(0.1, dtSec));
    if (this.flashProgress < 1) {
      this.flashProgress = Math.min(1, this.flashProgress + dt / 0.52);
    }
    if (this.wheelSuppressSec > 0) {
      this.wheelSuppressSec = Math.max(0, this.wheelSuppressSec - dt);
    }

    // 1. Active smooth lock-in glide (spherical slerp-like direction + radius interpolation)
    if (this.glideActive) {
      this.glideElapsedSec += dt;
      const rawT = clamp(this.glideElapsedSec / this.glideDurationSec, 0, 1);
      // Cubic ease-in-out for crisp optical glide without abrupt snap
      const ease = rawT < 0.5 ? 4 * rawT * rawT * rawT : 1 - Math.pow(-2 * rawT + 2, 3) / 2;

      const t = this.vantageBasis.target;
      const sx = this.glideStartPos[0] - t[0];
      const sy = this.glideStartPos[1] - t[1];
      const sz = this.glideStartPos[2] - t[2];
      const sRad = Math.max(0.25, Math.hypot(sx, sy, sz));
      const sDir = [sx / sRad, sy / sRad, sz / sRad];

      const vx = this.vantageBasis.pos[0] - t[0];
      const vy = this.vantageBasis.pos[1] - t[1];
      const vz = this.vantageBasis.pos[2] - t[2];
      const vRad = Math.max(0.25, Math.hypot(vx, vy, vz));
      const vDir = [vx / vRad, vy / vRad, vz / vRad];

      // Spherical interpolation along unit sphere (with safe antipodal midpoint bias)
      let dotP = clamp(sDir[0] * vDir[0] + sDir[1] * vDir[1] + sDir[2] * vDir[2], -1, 1);
      let dx: number;
      let dy: number;
      let dz: number;
      if (dotP > 0.9999) {
        dx = vDir[0];
        dy = vDir[1];
        dz = vDir[2];
      } else if (dotP < -0.995) {
        // Near-antipodal: arc gently over the right/upper side so we never pass through origin
        const midX = sDir[0] * (1 - ease) + vDir[0] * ease + Math.sin(ease * Math.PI) * 0.65;
        const midY = sDir[1] * (1 - ease) + vDir[1] * ease + Math.sin(ease * Math.PI) * 0.35;
        const midZ = sDir[2] * (1 - ease) + vDir[2] * ease;
        const mLen = Math.max(1e-6, Math.hypot(midX, midY, midZ));
        dx = midX / mLen;
        dy = midY / mLen;
        dz = midZ / mLen;
      } else {
        const omega = Math.acos(dotP);
        const sinO = Math.sin(omega);
        const w0 = Math.sin((1 - ease) * omega) / sinO;
        const w1 = Math.sin(ease * omega) / sinO;
        dx = sDir[0] * w0 + vDir[0] * w1;
        dy = sDir[1] * w0 + vDir[1] * w1;
        dz = sDir[2] * w0 + vDir[2] * w1;
      }

      const rad = sRad + (vRad - sRad) * ease;
      outCameraPos[0] = t[0] + dx * rad;
      outCameraPos[1] = t[1] + dy * rad;
      outCameraPos[2] = t[2] + dz * rad;

      if (rawT >= 1) {
        outCameraPos[0] = this.vantageBasis.pos[0];
        outCameraPos[1] = this.vantageBasis.pos[1];
        outCameraPos[2] = this.vantageBasis.pos[2];
        this.glideActive = false;
      }

      this.setCameraPos(outCameraPos[0], outCameraPos[1], outCameraPos[2]);
      return true;
    }

    this.setCameraPos(liveCameraPos[0], liveCameraPos[1], liveCameraPos[2]);

    // 2. Gentle magnetic optical detent when released inside DETENT_CAPTURE_DEG (never fights drag!)
    if (
      !this.isDragging &&
      this.alignmentErrorDeg <= DETENT_CAPTURE_DEG &&
      (this.alignmentErrorDeg > 0.008 || (this.wheelSuppressSec <= 0 && this.radialError > 0.004))
    ) {
      const pullStrength = 1 - Math.exp(-dt * 7.5);
      const t = this.vantageBasis.target;
      const cx = liveCameraPos[0] - t[0];
      const cy = liveCameraPos[1] - t[1];
      const cz = liveCameraPos[2] - t[2];
      const cRad = Math.max(0.25, Math.hypot(cx, cy, cz));

      const vx = this.vantageBasis.pos[0] - t[0];
      const vy = this.vantageBasis.pos[1] - t[1];
      const vz = this.vantageBasis.pos[2] - t[2];
      const vRad = Math.max(0.25, Math.hypot(vx, vy, vz));

      const nx = (cx / cRad) * (1 - pullStrength) + (vx / vRad) * pullStrength;
      const ny = (cy / cRad) * (1 - pullStrength) + (vy / vRad) * pullStrength;
      const nz = (cz / cRad) * (1 - pullStrength) + (vz / vRad) * pullStrength;
      const nLen = Math.max(1e-6, Math.hypot(nx, ny, nz));

      const targetRad =
        this.wheelSuppressSec > 0 ? cRad : cRad + (vRad - cRad) * (1 - Math.exp(-dt * 5.0));

      outCameraPos[0] = t[0] + (nx / nLen) * targetRad;
      outCameraPos[1] = t[1] + (ny / nLen) * targetRad;
      outCameraPos[2] = t[2] + (nz / nLen) * targetRad;
      this.setCameraPos(outCameraPos[0], outCameraPos[1], outCameraPos[2]);
      return true;
    }

    outCameraPos[0] = liveCameraPos[0];
    outCameraPos[1] = liveCameraPos[1];
    outCameraPos[2] = liveCameraPos[2];
    return false;
  }

  getState(): Day26Diagnostics {
    return {
      alignmentErrorDeg: Number(this.alignmentErrorDeg.toFixed(3)),
      isLockedIn: this.isLockedIn,
      activeStencilIndex: this.activeStencilIndex,
      projectionCount: this.projectionCount,
      vantagePos: [
        Number(this.vantageBasis.pos[0].toFixed(4)),
        Number(this.vantageBasis.pos[1].toFixed(4)),
        Number(this.vantageBasis.pos[2].toFixed(4)),
      ],
      cameraPos: [
        Number(this.cameraPos[0].toFixed(4)),
        Number(this.cameraPos[1].toFixed(4)),
        Number(this.cameraPos[2].toFixed(4)),
      ],
    };
  }
}
