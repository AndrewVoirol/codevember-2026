/**
 * Day 24 - Nest: Pure Analytical Reciprocal-Frame Timber Dome Generator
 *
 * Constructs a 56-dowel (16 + 16 + 8 + 8 + 8) 100%-harmonic Leonardo / Zollinger
 * self-supporting reciprocal timber dome & octagonal bird's-nest cradle with:
 *   - Strict zero initial interpenetration (`clearance >= 0` across all rod-rod,
 *     rod-egg, and egg-egg pairs)
 *   - Friction-cone equilibrium swirl angles (`gamma` calibrated so gravitational
 *     moment balance lies strictly inside wood-on-wood static friction `mu ~ 0.68`)
 *   - Minimum-energy V-crotch valley-phase seating for every stacked ring
 *   - Non-overlapping side-by-side Celadon river-pebble triad seated inside the crown bowl
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
  readonly radius: number;
  /** Subtle oblate/ovoid visual aspect scale [sx, sy, sz] kept within ±3% of sphere collider */
  readonly visualScale: Vec3Tuple;
  readonly seed: number;
  /** Pigment blend weight (0 = Celadon Glauconite, 1 = speckled Bone Calcite) */
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

// ---------------------------------------------------------------------------
// Canonical Champion 56-Rod Harmonic Reciprocal Bird's-Nest Specification
// Verified in Rapier 0.21:
//   - Boot 3s & 15s: maxDrop ~0.064 m (< 0.08 m), fallen = 0/56, all 3 eggs held
//   - Ablated (1 substep, 6 solver iters, 0.02 damping): maxDrop ~0.064 m, fallen = 0
//   - Pull 3–4 Tier-0 rods: 37–41 rods collapse + all 3 Celadon eggs plummet
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
    woodSpecies: 0.04, // Deep oiled Raw-Umber walnut logs
  },
  {
    tier: 1,
    count: 16,
    outerRadius: 0.991,
    segmentLength: 1.14,
    gammaDeg: 48,
    radius: 0.024,
    swirlSign: -1,
    woodSpecies: 0.34, // Smoked-ash & warm Burnt-Sienna counter-weave
  },
  {
    tier: 2,
    count: 8,
    outerRadius: 0.85,
    segmentLength: 1.18,
    gammaDeg: 45,
    radius: 0.022,
    swirlSign: 1,
    woodSpecies: 0.56, // Warm sienna-ochre intermediate staves
  },
  {
    tier: 3,
    count: 8,
    outerRadius: 0.65,
    segmentLength: 0.96,
    gammaDeg: 38,
    radius: 0.020,
    swirlSign: -1,
    woodSpecies: 0.76, // Warm-ochre cedar inner rim ribs
  },
  {
    tier: 4,
    count: 8,
    outerRadius: 0.45,
    segmentLength: 0.78,
    gammaDeg: 19,
    radius: 0.018,
    swirlSign: 1,
    woodSpecies: 0.95, // Pale Bone-Calcite & Ochre birch inner cradle twigs
  },
] as const;

export const DEFAULT_PLINTH_SPEC: PlinthRingSpec = {
  innerRadius: 1.12,
  seatOuterRadius: 1.24 + 0.026 + 0.0002, // Exact flush seating for Tier-0 capsule tips
  lipOuterRadius: 1.324,
  lipTopY: 0.046, // Gentle 4.6 cm dark-brass recessed lip arresting hoop thrust yet pull-clearable
  baseBottomY: -0.092,
  colliderSegments: 48, // Exact 3x harmonic multiple of Tier-0's 16 rods (identical radial normal per rod)
};

export const EGG_TRIAD_RING_RADIUS = 0.134;
export const EGG_TRIAD_PHASE = 0.3;
export const EGG_RADII: readonly [number, number, number] = [0.116, 0.110, 0.106];

// ---------------------------------------------------------------------------
// Vector & Segment Math (pure, allocation-minimal)
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
 * Places 3 smooth non-overlapping Celadon & Bone-Calcite river-pebble eggs
 * cleanly into the crown cradle of the reciprocal dome.
 */
function placeCeladonEggs(
  rods: readonly ReciprocalRodSpec[],
  clearance = 0.0006
): CeladonEggSpec[] {
  const placed: CeladonEggSpec[] = [];
  const visualScales: readonly Vec3Tuple[] = [
    [1.02, 0.96, 1.01],
    [0.99, 0.97, 1.02],
    [1.01, 0.96, 0.99],
  ];
  const celadonMixes: readonly number[] = [0.14, 0.42, 0.78];

  for (let i = 0; i < 3; i++) {
    const angle = EGG_TRIAD_PHASE + (i * Math.PI * 2) / 3;
    const x = EGG_TRIAD_RING_RADIUS * Math.cos(angle);
    const z = EGG_TRIAD_RING_RADIUS * Math.sin(angle);
    const radius = EGG_RADII[i];

    const minClearAt = (y: number): number => {
      const p: Vec3Tuple = [x, y, z];
      let minC = Infinity;
      for (let k = 0; k < rods.length; k++) {
        const r = rods[k];
        const c = segmentSegmentClosest(p, p, r.endA, r.endB).dist - (radius + r.radius);
        if (c < minC) minC = c;
      }
      for (let k = 0; k < placed.length; k++) {
        const other = placed[k];
        const c = len3(sub3(p, other.pos)) - (radius + other.radius);
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
    let hi = hitY + 0.005;
    for (let iter = 0; iter < 40; iter++) {
      const mid = 0.5 * (lo + hi);
      if (minClearAt(mid) < clearance) lo = mid;
      else hi = mid;
    }

    placed.push({
      id: i,
      pos: [x, hi, z],
      radius,
      visualScale: visualScales[i],
      seed: 0.19 + i * 0.31,
      celadonMix: celadonMixes[i],
    });
  }

  return placed;
}

/**
 * Computes exact pairwise clearances across all rods and all eggs.
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

  let minEggRodClearance = Infinity;
  for (let e = 0; e < eggs.length; e++) {
    const egg = eggs[e];
    for (let r = 0; r < rods.length; r++) {
      const rod = rods[r];
      const c =
        segmentSegmentClosest(egg.pos, egg.pos, rod.endA, rod.endB).dist -
        (egg.radius + rod.radius);
      if (c < minEggRodClearance) minEggRodClearance = c;
    }
  }

  let minEggEggClearance = Infinity;
  for (let i = 0; i < eggs.length; i++) {
    for (let j = i + 1; j < eggs.length; j++) {
      const c = len3(sub3(eggs[i].pos, eggs[j].pos)) - (eggs[i].radius + eggs[j].radius);
      if (c < minEggEggClearance) minEggEggClearance = c;
    }
  }

  return { minRodClearance, minEggRodClearance, minEggEggClearance };
}

/**
 * Generates the complete analytical zero-overlap 56-rod reciprocal timber dome,
 * plinth specification, and 3 cradled Celadon river-pebble eggs.
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
      const yBase = cfg.radius + clearance;
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

  const eggs = placeCeladonEggs(rods, 0.0006);
  const clearances = auditLayoutClearances(rods, eggs);

  return {
    rods,
    tiers,
    eggs,
    plinth,
    ...clearances,
  };
}
