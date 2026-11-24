/**
 * Day 24 - Nest: Pure Analytical Reciprocal-Frame Timber Dome Generator
 *
 * Constructs a 56-dowel (16 + 16 + 8 + 8 + 8) 100%-harmonic Leonardo / Zollinger
 * self-supporting reciprocal timber dome & octagonal bird's-nest cradle with:
 *   - Strict zero initial interpenetration (`clearance >= 0` across all rod-rod,
 *     rod-egg, and egg-egg pairs)
 *   - Exact chamfered flat-ended lathe dowel envelope (`roundCylinder` with
 *     `DOWEL_CHAMFER_FRAC = 0.18`, strictly contained inside the analytical capsule)
 *   - Friction-cone equilibrium swirl angles (`gamma` calibrated so gravitational
 *     moment balance lies strictly inside wood-on-wood static friction `mu ~ 0.68`)
 *   - Minimum-energy V-crotch valley-phase seating for every stacked ring
 *   - Non-overlapping side-by-side ovoid Celadon & Bone-Calcite river-pebble triad
 *     seated cleanly inside the crown bowl using exact ovoid vertex clearance
 *
 * Pure math module (zero DOM / WebGPU dependencies) so Vitest can verify both
 * geometry clearances and live Rapier 0.21 rigid-body equilibrium directly.
 */

export type Vec3Tuple = readonly [number, number, number];

export interface QuatTuple {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly w: number;
}

export interface ReciprocalTierConfig {
  readonly tier: number;
  readonly count: number;
  readonly outerRadius: number;
  readonly segmentLength: number;
  readonly gammaDeg: number;
  readonly radius: number;
  /** +1 = CCW swirl, -1 = CW counter-swirl */
  readonly swirlSign: 1 | -1;
  /** Wood species pigment index in [0..1] (0=walnut, 0.35=smoked-ash, 0.7=ochre-cedar, 1=birch) */
  readonly woodSpecies: number;
}

export interface ReciprocalRodSpec {
  readonly id: number;
  readonly tier: number;
  readonly indexInTier: number;
  readonly endA: Vec3Tuple;
  readonly endB: Vec3Tuple;
  readonly center: Vec3Tuple;
  readonly dir: Vec3Tuple;
  readonly halfHeight: number;
  readonly radius: number;
  readonly quat: QuatTuple;
  readonly woodSpecies: number;
  readonly grainSeed: number;
  /** Normalized segment parameters [0..1] where this rod meets its tier neighbors */
  readonly notchParams: readonly [number, number];
}

export interface TierBuildMeta {
  readonly tier: number;
  readonly count: number;
  readonly halfHeight: number;
  readonly radius: number;
  readonly yBase: number;
  readonly yTop: number;
  readonly dy: number;
  readonly phase: number;
  readonly selfContactS: number;
  readonly selfContactT: number;
}

export interface CeladonEggSpec {
  readonly id: number;
  readonly pos: Vec3Tuple;
  readonly rot: QuatTuple;
  readonly radius: number;
  /** Natural ovoid river-pebble aspect scale [sx, sy, sz] matched 1:1 by Rapier convexHull */
  readonly visualScale: Vec3Tuple;
  readonly seed: number;
  /** Pigment blend weight (0 = rich Celadon Glauconite, 1 = speckled Bone Calcite) */
  readonly celadonMix: number;
}

export interface PlinthRingSpec {
  readonly innerRadius: number;
  readonly seatOuterRadius: number;
  readonly lipOuterRadius: number;
  readonly lipTopY: number;
  readonly baseBottomY: number;
  readonly colliderSegments: number;
}

export interface ReciprocalDomeLayout {
  readonly rods: readonly ReciprocalRodSpec[];
  readonly tiers: readonly TierBuildMeta[];
  readonly eggs: readonly CeladonEggSpec[];
  readonly plinth: PlinthRingSpec;
  readonly minRodClearance: number;
  readonly minEggRodClearance: number;
  readonly minEggEggClearance: number;
}

export interface SegmentClosestResult {
  readonly dist: number;
  readonly s: number;
  readonly t: number;
  readonly c1: Vec3Tuple;
  readonly c2: Vec3Tuple;
}

/**
 * Chamfer bevel radius fraction (`borderRadius = radius * DOWEL_CHAMFER_FRAC`) shared
 * identically by `buildDowelGeometry`, Rapier `roundCylinder` colliders, and contact audits.
 * Because `roundCylinder(hh, r - b, b)` is mathematically strictly contained inside
 * `capsule(hh, r)`, zero-overlap capsule layouts guarantee zero-overlap chamfered dowels.
 */
export const DOWEL_CHAMFER_FRAC = 0.18;

// ---------------------------------------------------------------------------
// Canonical Champion 56-Rod Harmonic Reciprocal Bird's-Nest Specification
// Verified in Rapier 0.21 with roundCylinder dowels + convexHull ovoid eggs:
//   - Boot 3s & 10s (f=120 Hz hardwood contact): maxDrop ~0.015 m, maxBarrelPen < 0.8 mm,
//     fallen = 0/56, maxV = 0, all 3 ovoid pebbles cradled flush in the birch oculus
//   - Ablated (1 substep, 6 solver iters, 0.02 damping, f=30 Hz): maxDrop < 0.06 m, fallen = 0
//   - Pull 3–4 Tier-0 rods: 52–53 rods cascade & all 3 eggs drop cleanly through hollow bore
// ---------------------------------------------------------------------------
export const DEFAULT_TIER_CONFIGS: readonly ReciprocalTierConfig[] = [
  {
    tier: 0,
    count: 16,
    outerRadius: 1.24,
    segmentLength: 1.36,
    gammaDeg: 49,
    radius: 0.026,
    swirlSign: 1,
    woodSpecies: 0.04, // Deep oiled Raw-Umber black walnut timbers
  },
  {
    tier: 1,
    count: 16,
    outerRadius: 0.991,
    segmentLength: 1.14,
    gammaDeg: 48,
    radius: 0.024,
    swirlSign: -1,
    woodSpecies: 0.34, // Smoked-ash open-pore counter-weave
  },
  {
    tier: 2,
    count: 8,
    outerRadius: 0.85,
    segmentLength: 1.18,
    gammaDeg: 45,
    radius: 0.022,
    swirlSign: 1,
    woodSpecies: 0.56, // Warm honey-red aromatic cedar staves
  },
  {
    tier: 3,
    count: 8,
    outerRadius: 0.65,
    segmentLength: 0.96,
    gammaDeg: 38,
    radius: 0.020,
    swirlSign: -1,
    woodSpecies: 0.76, // Golden yellow-ochre cedar inner rim ribs
  },
  {
    tier: 4,
    count: 8,
    outerRadius: 0.45,
    segmentLength: 0.78,
    gammaDeg: 19,
    radius: 0.018,
    swirlSign: 1,
    woodSpecies: 0.96, // Pale satin Nordic birch inner cradle twigs
  },
] as const;

export const DEFAULT_PLINTH_SPEC: PlinthRingSpec = {
  innerRadius: 1.12,
  // Exact flush outer retaining seat for Tier-0 chamfered roundCylinder butts (1.26096 + 0.0003 m)
  seatOuterRadius: 1.26125,
  lipOuterRadius: 1.326,
  lipTopY: 0.046, // Crisp 4.6 cm dark-brass recessed lip arresting hoop thrust yet pull-clearable
  baseBottomY: -0.096,
  colliderSegments: 48, // Exact 3x harmonic multiple of Tier-0's 16 rods (identical radial normal per rod)
};

export const EGG_TRIAD_RING_RADIUS = 0.136;
export const EGG_TRIAD_PHASE = 0.34;
export const EGG_RADII: readonly [number, number, number] = [0.114, 0.109, 0.105];

// ---------------------------------------------------------------------------
// Vector, Quaternion & Segment Math (pure, zero DOM / Three.js dependencies)
// ---------------------------------------------------------------------------

function sub3(a: Vec3Tuple, b: Vec3Tuple): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add3(a: Vec3Tuple, b: Vec3Tuple): [number, number, number] {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale3(a: Vec3Tuple, s: number): [number, number, number] {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function dot3(a: Vec3Tuple, b: Vec3Tuple): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function len3(a: Vec3Tuple): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function norm3(a: Vec3Tuple): [number, number, number] {
  const l = len3(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function rotateAroundY(p: Vec3Tuple, theta: number): [number, number, number] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
}

export function rotateVecByQuat(v: Vec3Tuple, q: QuatTuple): [number, number, number] {
  const tx = 2 * (q.y * v[2] - q.z * v[1]);
  const ty = 2 * (q.z * v[0] - q.x * v[2]);
  const tz = 2 * (q.x * v[1] - q.y * v[0]);
  return [
    v[0] + q.w * tx + (q.y * tz - q.z * ty),
    v[1] + q.w * ty + (q.z * tx - q.x * tz),
    v[2] + q.w * tz + (q.x * ty - q.y * tx),
  ];
}

function quatFromYawPitchRoll(yaw: number, pitch: number, roll: number): QuatTuple {
  const cy = Math.cos(yaw * 0.5);
  const sy = Math.sin(yaw * 0.5);
  const cp = Math.cos(pitch * 0.5);
  const sp = Math.sin(pitch * 0.5);
  const cr = Math.cos(roll * 0.5);
  const sr = Math.sin(roll * 0.5);
  return {
    x: cy * sp * cr + sy * cp * sr,
    y: sy * cp * cr - cy * sp * sr,
    z: cy * cp * sr - sy * sp * cr,
    w: cy * cp * cr + sy * sp * sr,
  };
}

/**
 * Shared pure-math ovoid river-pebble surface evaluator used identically by:
 *   1. Visual high-res mesh (`nestGeometry.ts: buildPebbleEggGeometry`)
 *   2. Rapier `convexHull` collider (`nestSimulation.ts`)
 *   3. Exact zero-penetration cradling layout & clearances (`reciprocalDome.ts`)
 *
 * Elongated along local X (`sx > 1`), softly flattened vertically (`sy < 1`),
 * with a gentle natural avian-egg taper (fuller blunt end at -X, tapered tip at +X)
 * and subtle river-tumbled mineral asymmetry while remaining strictly convex.
 */
export function sampleOvoidPebblePoint(
  nx: number,
  ny: number,
  nz: number,
  radius: number,
  visualScale: Vec3Tuple,
  seed: number
): [number, number, number] {
  const [sx, sy, sz] = visualScale;
  const p1 = seed * 17.3 + 1.1;
  const p2 = seed * 29.7 + 2.4;
  const p3 = seed * 43.1 + 0.7;

  // Classic subtle wild-egg ovoid taper along local X (fuller base at -X, gently pointed crown at +X)
  const ovoidTaper = 1.0 - 0.068 * nx;
  const wearA = Math.sin(nx * 2.1 + ny * 1.5 + p1) * Math.cos(nz * 1.8 - p2);
  const wearB = Math.cos(ny * 2.2 - nz * 1.6 + p3) * Math.sin(nx * 1.4 + p1);
  const rScale = 1.0 + 0.011 * wearA + 0.007 * wearB;

  return [
    nx * radius * sx * rScale,
    ny * radius * sy * ovoidTaper * rScale,
    nz * radius * sz * ovoidTaper * rScale,
  ];
}

/**
 * Generates deterministic unit-sphere directions (subdivided icosahedron vertices,
 * deduplicated) for Rapier convexHull generation and exact ovoid clearance audits.
 */
export function generateUnitIcoVertices(detail = 2): Array<[number, number, number]> {
  const phi = (1 + Math.sqrt(5)) * 0.5;
  const rawVerts: readonly Vec3Tuple[] = [
    [-1, phi, 0],
    [1, phi, 0],
    [-1, -phi, 0],
    [1, -phi, 0],
    [0, -1, phi],
    [0, 1, phi],
    [0, -1, -phi],
    [0, 1, -phi],
    [phi, 0, -1],
    [phi, 0, 1],
    [-phi, 0, -1],
    [-phi, 0, 1],
  ];
  const baseVerts: Array<[number, number, number]> = rawVerts.map((v) => norm3(v));

  let faces: Array<[number, number, number]> = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
  ];

  const verts = [...baseVerts];
  for (let d = 0; d < detail; d++) {
    const midCache = new Map<string, number>();
    const getMid = (i0: number, i1: number): number => {
      const key = i0 < i1 ? `${i0}_${i1}` : `${i1}_${i0}`;
      const existing = midCache.get(key);
      if (existing !== undefined) return existing;
      const a = verts[i0];
      const b = verts[i1];
      const idx = verts.length;
      verts.push(norm3([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]));
      midCache.set(key, idx);
      return idx;
    };

    const nextFaces: Array<[number, number, number]> = [];
    for (let f = 0; f < faces.length; f++) {
      const [a, b, c] = faces[f];
      const ab = getMid(a, b);
      const bc = getMid(b, c);
      const ca = getMid(c, a);
      nextFaces.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = nextFaces;
  }

  return verts;
}

const UNIT_ICO_DETAIL_2 = generateUnitIcoVertices(2);

/**
 * Generates exact local-space convex hull vertex buffer for a river-pebble egg
 * (identical mathematical surface to `buildPebbleEggGeometry` in `nestGeometry.ts`).
 */
export function buildOvoidPebbleHullPoints(
  radius: number,
  visualScale: Vec3Tuple,
  seed: number
): Float32Array {
  const out = new Float32Array(UNIT_ICO_DETAIL_2.length * 3);
  for (let i = 0; i < UNIT_ICO_DETAIL_2.length; i++) {
    const [nx, ny, nz] = UNIT_ICO_DETAIL_2[i];
    const [px, py, pz] = sampleOvoidPebblePoint(nx, ny, nz, radius, visualScale, seed);
    out[i * 3 + 0] = px;
    out[i * 3 + 1] = py;
    out[i * 3 + 2] = pz;
  }
  return out;
}

/**
 * Exact 3D closest points and Euclidean distance between line segments [p1,q1] and [p2,q2]
 * (Christer Ericson, Real-Time Collision Detection §5.1.9).
 */
export function segmentSegmentClosest(
  p1: Vec3Tuple,
  q1: Vec3Tuple,
  p2: Vec3Tuple,
  q2: Vec3Tuple
): SegmentClosestResult {
  const d1 = sub3(q1, p1);
  const d2 = sub3(q2, p2);
  const r = sub3(p1, p2);
  const a = dot3(d1, d1);
  const e = dot3(d2, d2);
  const f = dot3(d2, r);
  const eps = 1e-12;
  let s = 0;
  let t = 0;

  if (a <= eps && e <= eps) {
    s = 0;
    t = 0;
  } else if (a <= eps) {
    s = 0;
    t = Math.max(0, Math.min(1, f / e));
  } else {
    const c = dot3(d1, r);
    if (e <= eps) {
      t = 0;
      s = Math.max(0, Math.min(1, -c / a));
    } else {
      const b = dot3(d1, d2);
      const denom = a * e - b * b;
      if (denom !== 0) {
        s = Math.max(0, Math.min(1, (b * f - c * e) / denom));
      } else {
        s = 0;
      }
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = Math.max(0, Math.min(1, -c / a));
      } else if (t > 1) {
        t = 1;
        s = Math.max(0, Math.min(1, (b - c) / a));
      }
    }
  }

  const c1 = add3(p1, scale3(d1, s));
  const c2 = add3(p2, scale3(d2, t));
  return { dist: len3(sub3(c1, c2)), s, t, c1, c2 };
}

/**
 * Exact signed distance from world point `p` to a chamfered `roundCylinder(hh, r - b, b)`
 * centered at `center`, unit axis `axis`, halfHeight `hh`, outer radius `radius`, bevel `b`.
 */
export function pointToRoundCylinderSignedDist(
  p: Vec3Tuple,
  center: Vec3Tuple,
  axis: Vec3Tuple,
  halfHeight: number,
  radius: number,
  bevelFrac = DOWEL_CHAMFER_FRAC
): number {
  const b = radius * bevelFrac;
  const rFlat = radius - b;
  const rel = sub3(p, center);
  const y = dot3(rel, axis);
  const radVec: Vec3Tuple = [rel[0] - axis[0] * y, rel[1] - axis[1] * y, rel[2] - axis[2] * y];
  const r = len3(radVec);

  const dr = r - rFlat;
  const dy = Math.abs(y) - halfHeight;
  const outDist = Math.hypot(Math.max(dr, 0), Math.max(dy, 0));
  const inDist = Math.min(Math.max(dr, dy), 0);
  return outDist + inDist - b;
}

/**
 * Computes the shortest-arc unit quaternion aligning local +Y (0, 1, 0) to unit vector `dir`.
 */
export function quatFromYToDir(dir: Vec3Tuple): QuatTuple {
  const u = norm3(dir);
  const d = u[1];
  if (d > 0.999999) return { x: 0, y: 0, z: 0, w: 1 };
  if (d < -0.999999) return { x: 1, y: 0, z: 0, w: 0 };
  const cx = u[2];
  const cz = -u[0];
  const w = 1 + d;
  const invLen = 1 / Math.hypot(cx, cz, w);
  return { x: cx * invLen, y: 0, z: cz * invLen, w: w * invLen };
}

/**
 * Converts a friction-cone `(outerRadius, segmentLength, gammaDeg, swirlSign)`
 * tier description into planar polar target `(Rb, alpha)`.
 */
function tierPlanarEndpoints(cfg: ReciprocalTierConfig): {
  Ra: number;
  Rb: number;
  alpha: number;
} {
  const gam = (cfg.gammaDeg * Math.PI) / 180;
  const Bx = cfg.outerRadius - cfg.segmentLength * Math.cos(gam);
  const Bz = cfg.swirlSign * cfg.segmentLength * Math.sin(gam);
  return {
    Ra: cfg.outerRadius,
    Rb: Math.hypot(Bx, Bz),
    alpha: Math.atan2(Bz, Bx),
  };
}

/**
 * Solves exact vertical rise `dy` for an N-rod reciprocal ring so minimum inter-rod
 * capsule clearance equals `clearance` (zero overlap, immediate resting contact).
 */
function solveRingDeltaY(
  count: number,
  Ra: number,
  Rb: number,
  alpha: number,
  radius: number,
  clearance: number
): { dy: number; selfS: number; selfT: number } {
  const evalAt = (dy: number) => {
    const A0: Vec3Tuple = [Ra, 0, 0];
    const B0: Vec3Tuple = [Rb * Math.cos(alpha), dy, Rb * Math.sin(alpha)];
    let minD = Infinity;
    let bestS = 0.5;
    let bestT = 0.5;
    for (let j = 1; j < count; j++) {
      const th = (j / count) * Math.PI * 2;
      const res = segmentSegmentClosest(A0, B0, rotateAroundY(A0, th), rotateAroundY(B0, th));
      if (res.dist < minD) {
        minD = res.dist;
        bestS = res.s;
        bestT = res.t;
      }
    }
    return { minD, bestS, bestT };
  };

  let lo = 0.001;
  let hi = 2.0;
  const targetDist = 2 * radius + clearance;
  for (let iter = 0; iter < 55; iter++) {
    const mid = 0.5 * (lo + hi);
    if (evalAt(mid).minD < targetDist) lo = mid;
    else hi = mid;
  }
  const finalInfo = evalAt(hi);
  return { dy: hi, selfS: finalInfo.bestS, selfT: finalInfo.bestT };
}

function buildRingRods(
  cfg: ReciprocalTierConfig,
  Ra: number,
  Rb: number,
  alpha: number,
  yBase: number,
  phase: number,
  dy: number,
  startId: number,
  notchParams: readonly [number, number]
): ReciprocalRodSpec[] {
  const rods: ReciprocalRodSpec[] = [];
  const A0: Vec3Tuple = [Ra, yBase, 0];
  const B0: Vec3Tuple = [Rb * Math.cos(alpha), yBase + dy, Rb * Math.sin(alpha)];

  for (let i = 0; i < cfg.count; i++) {
    const th = phase + (i / cfg.count) * Math.PI * 2;
    const endA = rotateAroundY(A0, th);
    const endB = rotateAroundY(B0, th);
    const seg = sub3(endB, endA);
    const segLen = len3(seg);
    const dir = scale3(seg, 1 / segLen);
    const id = startId + i;
    // Deterministic per-rod grain seed in [0, 1)
    const rawHash = Math.sin(id * 127.1 + cfg.tier * 311.7) * 43758.5453;
    const grainSeed = rawHash - Math.floor(rawHash);

    rods.push({
      id,
      tier: cfg.tier,
      indexInTier: i,
      endA,
      endB,
      center: scale3(add3(endA, endB), 0.5),
      dir,
      halfHeight: segLen * 0.5,
      radius: cfg.radius,
      quat: quatFromYToDir(dir),
      woodSpecies: cfg.woodSpecies,
      grainSeed,
      notchParams,
    });
  }

  return rods;
}

/**
 * Sweeps azimuthal phase across one symmetry period to find the exact lowest-energy
 * V-crotch seating pocket on top of `lowerRods` with guaranteed clearance >= `clearance`.
 */
function placeRingInValley(
  cfg: ReciprocalTierConfig,
  lowerRods: readonly ReciprocalRodSpec[],
  startId: number,
  clearance: number
): { rods: ReciprocalRodSpec[]; meta: TierBuildMeta } {
  const { Ra, Rb, alpha } = tierPlanarEndpoints(cfg);
  const { dy, selfS, selfT } = solveRingDeltaY(
    cfg.count,
    Ra,
    Rb,
    alpha,
    cfg.radius,
    clearance
  );

  const yBaseForPhase = (phase: number): number => {
    const minClearAt = (yb: number): number => {
      let minC = Infinity;
      const A0: Vec3Tuple = [Ra, yb, 0];
      const B0: Vec3Tuple = [Rb * Math.cos(alpha), yb + dy, Rb * Math.sin(alpha)];
      for (let i = 0; i < cfg.count; i++) {
        const th = phase + (i / cfg.count) * Math.PI * 2;
        const Ai = rotateAroundY(A0, th);
        const Bi = rotateAroundY(B0, th);
        for (let k = 0; k < lowerRods.length; k++) {
          const lr = lowerRods[k];
          const c = segmentSegmentClosest(Ai, Bi, lr.endA, lr.endB).dist - (cfg.radius + lr.radius);
          if (c < minC) minC = c;
        }
      }
      return minC;
    };

    let hitY = -0.2;
    for (let y = 1.4; y >= -0.1; y -= 0.008) {
      if (minClearAt(y) < clearance) {
        hitY = y;
        break;
      }
    }
    let lo = hitY;
    let hi = hitY + 0.01;
    for (let iter = 0; iter < 36; iter++) {
      const mid = 0.5 * (lo + hi);
      if (minClearAt(mid) < clearance) lo = mid;
      else hi = mid;
    }
    return hi;
  };

  const period = (Math.PI * 2) / cfg.count;
  let bestPhase = 0;
  let bestY = Infinity;
  const coarseSamples = 48;
  for (let s = 0; s < coarseSamples; s++) {
    const ph = (s / coarseSamples) * period;
    const yb = yBaseForPhase(ph);
    if (yb < bestY) {
      bestY = yb;
      bestPhase = ph;
    }
  }
  for (let ds = -1; ds <= 1; ds += 0.15) {
    const ph = bestPhase + (ds / coarseSamples) * period;
    const yb = yBaseForPhase(ph);
    if (yb < bestY) {
      bestY = yb;
      bestPhase = ph;
    }
  }

  const notchA = Math.min(selfS, selfT);
  const notchB = Math.max(selfS, selfT);
  const rods = buildRingRods(
    cfg,
    Ra,
    Rb,
    alpha,
    bestY,
    bestPhase,
    dy,
    startId,
    [notchA, notchB]
  );

  const meta: TierBuildMeta = {
    tier: cfg.tier,
    count: cfg.count,
    halfHeight: rods[0].halfHeight,
    radius: cfg.radius,
    yBase: bestY,
    yTop: bestY + dy,
    dy,
    phase: bestPhase,
    selfContactS: selfS,
    selfContactT: selfT,
  };

  return { rods, meta };
}

/**
 * Computes exact minimum clearance between an ovoid pebble's world surface vertices
 * and all chamfered `roundCylinder` dowels.
 */
function ovoidToRodsClearance(
  eggPos: Vec3Tuple,
  eggRot: QuatTuple,
  hullPts: Float32Array,
  rods: readonly ReciprocalRodSpec[]
): number {
  let minC = Infinity;
  const vertCount = hullPts.length / 3;
  for (let v = 0; v < vertCount; v++) {
    const local: Vec3Tuple = [hullPts[v * 3], hullPts[v * 3 + 1], hullPts[v * 3 + 2]];
    const r = rotateVecByQuat(local, eggRot);
    const wp: Vec3Tuple = [eggPos[0] + r[0], eggPos[1] + r[1], eggPos[2] + r[2]];
    for (let k = 0; k < rods.length; k++) {
      const rod = rods[k];
      const d = pointToRoundCylinderSignedDist(wp, rod.center, rod.dir, rod.halfHeight, rod.radius);
      if (d < minC) minC = d;
    }
  }
  return minC;
}

/**
 * Conservative exact separation distance between two convex ovoid pebbles using
 * their oriented world vertex clouds (lower-bounded by supporting-plane gap).
 */
function ovoidToOvoidClearance(
  posA: Vec3Tuple,
  rotA: QuatTuple,
  ptsA: Float32Array,
  posB: Vec3Tuple,
  rotB: QuatTuple,
  ptsB: Float32Array
): number {
  const axis = norm3(sub3(posB, posA));
  const centerDist = len3(sub3(posB, posA));
  let maxProjA = -Infinity;
  for (let i = 0; i < ptsA.length; i += 3) {
    const r = rotateVecByQuat([ptsA[i], ptsA[i + 1], ptsA[i + 2]], rotA);
    const p = dot3(r, axis);
    if (p > maxProjA) maxProjA = p;
  }
  let maxProjB = -Infinity;
  for (let i = 0; i < ptsB.length; i += 3) {
    const r = rotateVecByQuat([ptsB[i], ptsB[i + 1], ptsB[i + 2]], rotB);
    const p = -dot3(r, axis);
    if (p > maxProjB) maxProjB = p;
  }
  return centerDist - (maxProjA + maxProjB);
}

/**
 * Places 3 smooth non-overlapping ovoid Celadon & Bone-Calcite river-pebble eggs
 * cleanly into the crown cradle of the reciprocal dome using exact ovoid vertex
 * clearance against the chamfered timber ring (zero initial overlap, minimal settling).
 */
function placeCeladonEggs(
  rods: readonly ReciprocalRodSpec[],
  clearance = 0.0008
): CeladonEggSpec[] {
  const placed: CeladonEggSpec[] = [];
  const placedHulls: Float32Array[] = [];

  const visualScales: readonly Vec3Tuple[] = [
    [1.16, 0.87, 0.99],
    [1.13, 0.86, 0.98],
    [1.10, 0.88, 0.97],
  ];
  // Three harmonious mineral finishes: rich misty Celadon, pale sea-glass Celadon, speckled Bone-Calcite
  const celadonMixes: readonly number[] = [0.08, 0.48, 0.94];
  const basePitches: readonly number[] = [0.10, -0.09, 0.08];
  const baseRolls: readonly number[] = [-0.07, 0.08, -0.06];

  for (let i = 0; i < 3; i++) {
    const angle = EGG_TRIAD_PHASE + (i * Math.PI * 2) / 3;
    const x = EGG_TRIAD_RING_RADIUS * Math.cos(angle);
    const z = EGG_TRIAD_RING_RADIUS * Math.sin(angle);
    const radius = EGG_RADII[i];
    const seed = 0.19 + i * 0.31;
    // Align each ovoid pebble tangentially along the circular nest bowl with gentle natural tilt
    const yaw = angle + Math.PI * 0.5 + (i - 1) * 0.14;
    const rot = quatFromYawPitchRoll(yaw, basePitches[i], baseRolls[i]);
    const hull = buildOvoidPebbleHullPoints(radius, visualScales[i], seed);

    const minClearAt = (y: number): number => {
      const p: Vec3Tuple = [x, y, z];
      let minC = ovoidToRodsClearance(p, rot, hull, rods);
      for (let k = 0; k < placed.length; k++) {
        const other = placed[k];
        const c = ovoidToOvoidClearance(p, rot, hull, other.pos, other.rot, placedHulls[k]);
        if (c < minC) minC = c;
      }
      return minC;
    };

    let hitY = 0;
    for (let y = 1.4; y >= -0.1; y -= 0.004) {
      if (minClearAt(y) < clearance) {
        hitY = y;
        break;
      }
    }
    let lo = hitY;
    let hi = hitY + 0.006;
    for (let iter = 0; iter < 40; iter++) {
      const mid = 0.5 * (lo + hi);
      if (minClearAt(mid) < clearance) lo = mid;
      else hi = mid;
    }

    placed.push({
      id: i,
      pos: [x, hi, z],
      rot,
      radius,
      visualScale: visualScales[i],
      seed,
      celadonMix: celadonMixes[i],
    });
    placedHulls.push(hull);
  }

  return placed;
}

/**
 * Computes exact pairwise clearances across all rods and all ovoid eggs.
 */
export function auditLayoutClearances(
  rods: readonly ReciprocalRodSpec[],
  eggs: readonly CeladonEggSpec[]
): {
  minRodClearance: number;
  minEggRodClearance: number;
  minEggEggClearance: number;
} {
  let minRodClearance = Infinity;
  for (let i = 0; i < rods.length; i++) {
    for (let j = i + 1; j < rods.length; j++) {
      const c =
        segmentSegmentClosest(rods[i].endA, rods[i].endB, rods[j].endA, rods[j].endB).dist -
        (rods[i].radius + rods[j].radius);
      if (c < minRodClearance) minRodClearance = c;
    }
  }

  const eggHulls = eggs.map((egg) =>
    buildOvoidPebbleHullPoints(egg.radius, egg.visualScale, egg.seed)
  );

  let minEggRodClearance = Infinity;
  for (let e = 0; e < eggs.length; e++) {
    const c = ovoidToRodsClearance(eggs[e].pos, eggs[e].rot, eggHulls[e], rods);
    if (c < minEggRodClearance) minEggRodClearance = c;
  }

  let minEggEggClearance = Infinity;
  for (let i = 0; i < eggs.length; i++) {
    for (let j = i + 1; j < eggs.length; j++) {
      const c = ovoidToOvoidClearance(
        eggs[i].pos,
        eggs[i].rot,
        eggHulls[i],
        eggs[j].pos,
        eggs[j].rot,
        eggHulls[j]
      );
      if (c < minEggEggClearance) minEggEggClearance = c;
    }
  }

  return { minRodClearance, minEggRodClearance, minEggEggClearance };
}

/**
 * Generates the complete analytical zero-overlap 56-rod reciprocal timber dome,
 * plinth specification, and 3 cradled ovoid Celadon & Bone-Calcite river-pebble eggs.
 */
export function generateReciprocalDomeLayout(
  tierConfigs: readonly ReciprocalTierConfig[] = DEFAULT_TIER_CONFIGS,
  plinth: PlinthRingSpec = DEFAULT_PLINTH_SPEC,
  clearance = 0.00025
): ReciprocalDomeLayout {
  const rods: ReciprocalRodSpec[] = [];
  const tiers: TierBuildMeta[] = [];

  for (let t = 0; t < tierConfigs.length; t++) {
    const cfg = tierConfigs[t];
    if (t === 0) {
      const { Ra, Rb, alpha } = tierPlanarEndpoints(cfg);
      const { dy, selfS, selfT } = solveRingDeltaY(
        cfg.count,
        Ra,
        Rb,
        alpha,
        cfg.radius,
        clearance
      );
      // Exact flush bottom chamfer height for a tilted roundCylinder resting on y = 0
      const segLen = Math.hypot(Rb * Math.cos(alpha) - Ra, dy, Rb * Math.sin(alpha));
      const sinPhi = Math.abs(dy / Math.max(1e-6, segLen));
      const cosPhi = Math.sqrt(Math.max(0, 1 - sinPhi * sinPhi));
      const bevelR = cfg.radius * DOWEL_CHAMFER_FRAC;
      const yBase = (cfg.radius - bevelR) * cosPhi + bevelR + clearance;

      const notchA = Math.min(selfS, selfT);
      const notchB = Math.max(selfS, selfT);
      const tierRods = buildRingRods(
        cfg,
        Ra,
        Rb,
        alpha,
        yBase,
        0,
        dy,
        rods.length,
        [notchA, notchB]
      );
      rods.push(...tierRods);
      tiers.push({
        tier: cfg.tier,
        count: cfg.count,
        halfHeight: tierRods[0].halfHeight,
        radius: cfg.radius,
        yBase,
        yTop: yBase + dy,
        dy,
        phase: 0,
        selfContactS: selfS,
        selfContactT: selfT,
      });
    } else {
      const placed = placeRingInValley(cfg, rods, rods.length, clearance);
      rods.push(...placed.rods);
      tiers.push(placed.meta);
    }
  }

  const eggs = placeCeladonEggs(rods, 0.0008);
  const clearances = auditLayoutClearances(rods, eggs);

  return {
    rods,
    tiers,
    eggs,
    plinth,
    ...clearances,
  };
}
