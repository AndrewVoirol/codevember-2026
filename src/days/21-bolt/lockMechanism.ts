/**
 * Day 21 - Bolt: Deterministic 19th-Century Chubb-Style 5-Lever Mortise Deadbolt Physics
 *
 * Pure mechanical solver (zero Three.js rendering dependencies, zero per-frame allocations)
 * modeling exact geometric contact between:
 *  - 5 stacked independent brass lever tumblers pivoting around (PIVOT_X, PIVOT_Y)
 *  - Shared single-source-of-truth concentric Chubb belly curves (exact 0-clip, 0-gap contact)
 *  - 5 preloaded curved phosphor-bronze leaf springs with restoring torque
 *  - 5 uniquely bitted key teeth + rear bolt-talon drive bit rotating about (KEY_X, KEY_Y)
 *  - Heavy square blued-steel deadbolt, U-shaped bolt talon, and riveted square bolt stump
 *  - Exact bilateral bolt-stump <-> lever-gate collision blocking & lever trapping
 */

export const LEVER_COUNT = 5;

// Master mechanical datum coordinates (lock-local XY plane, 1 unit ~ 10 cm)
export const PIVOT_X = -0.58;
export const PIVOT_Y = -0.02;

export const KEY_X = 0.04;
export const KEY_Y = -0.22;
export const KEY_SHAFT_RADIUS = 0.048;

// Deadbolt & stump geometry (exact outer dimensions INCLUDING chamfer bevels)
export const STUMP_Y = 0.16;
export const STUMP_X_UNLOCKED = -0.08;
export const BOLT_TRAVEL = 0.34;
export const STUMP_X_LOCKED = STUMP_X_UNLOCKED + BOLT_TRAVEL; // +0.26
export const STUMP_WIDTH = 0.082;
export const STUMP_HEIGHT = 0.072;

// Lever Z-stack layout (front-to-back parallax, zero interpenetration)
export const LEVER_Z_START = -0.04;
export const LEVER_Z_PITCH = 0.044;
export const LEVER_THICKNESS = 0.024;
export const LEVER_BEVEL = 0.0022;

export function getLeverZ(index: number): number {
  return LEVER_Z_START + index * LEVER_Z_PITCH;
}

// Bolt tail sits behind Lever 0; bolt stump spans through all 5 levers
export const BOLT_TAIL_Z_BACK = -0.168;
export const BOLT_TAIL_Z_FRONT = -0.072;
export const BOLT_HEAD_Z_HALF = 0.168;
export const STUMP_Z_BACK = BOLT_TAIL_Z_FRONT;
export const STUMP_Z_FRONT = 0.174;

// Lock case & oak specimen bounds
export const CASE_INNER_MIN_X = -0.80;
export const CASE_INNER_MAX_X = 0.88;
export const CASE_INNER_MIN_Y = -0.565;
export const CASE_INNER_MAX_Y = 0.585;
export const FOREND_X = 0.94;
export const FOREND_THICKNESS = 0.065;

// Spring abutment & bottom rest stop posts
export const SPRING_POST_X = -0.46;
export const SPRING_POST_Y = 0.455;
export const REST_PIN_X = 0.405;
export const REST_PIN_Y = -0.27;
export const REST_PIN_RADIUS = 0.032;
// Lever local Y of bottom rest heel flush on top of REST_PIN (-0.27 + 0.032 - PIVOT_Y = -0.218)
export const LEVER_HEEL_LOCAL_Y = REST_PIN_Y + REST_PIN_RADIUS - PIVOT_Y;
export const GUIDE_PIN_X = -0.34;
export const GUIDE_PIN_Y = STUMP_Y;

// Lever belly datum (in lever-local coords at theta = 0, directly above KEY_X)
export const LEVER_BELLY_LOCAL_X = KEY_X - PIVOT_X; // +0.62
export const LEVER_BELLY_LOCAL_Y = -0.115;

// Unique Chubb 5-lever gate alignment angles (classic high-low-low-high-mid bitting [2, 5, 1, 4, 3])
export const LEVER_ALIGN_DEG: readonly number[] = [8.5, 16.5, 6.0, 14.0, 11.2];
export const LEVER_ALIGN_RAD: readonly number[] = LEVER_ALIGN_DEG.map((d) => (d * Math.PI) / 180);
// Physical upper pick stop: clears bolt stump cleanly inside deep right H-gate pocket up to 21.0 deg
export const LEVER_MAX_RAD = (21.0 * Math.PI) / 180;

// Exact key bitting step radii derived from rigid-body contact at phi = PI (12 o'clock)
export const KEY_BIT_RADII: readonly number[] = LEVER_ALIGN_RAD.map((theta) => {
  const yBellyWorld =
    PIVOT_Y + LEVER_BELLY_LOCAL_X * Math.sin(theta) + LEVER_BELLY_LOCAL_Y * Math.cos(theta);
  return Number((yBellyWorld - KEY_Y).toFixed(5));
});

// Key bitting tooth half-width (outer crown width including bevels)
export const KEY_BIT_HALF_WIDTH = 0.017;

// Rear bolt-talon drive bit radius and exact talon engagement half-angle
export const KEY_TALON_RADIUS = 0.305;
export const TALON_HALF_ANGLE_RAD = Math.asin((BOLT_TRAVEL * 0.5) / KEY_TALON_RADIUS); // ~0.5912 rad (33.87 deg)
export const TALON_ENGAGE_START_RAD = Math.PI - TALON_HALF_ANGLE_RAD; // ~146.13 deg
export const TALON_ENGAGE_END_RAD = Math.PI + TALON_HALF_ANGLE_RAD; // ~213.87 deg

// Concentric Chubb belly dwell half-angle (strictly wider than talon stroke so gates align BEFORE talon pushes)
export const CAM_DWELL_HALF_RAD = TALON_HALF_ANGLE_RAD + 0.055; // ~0.6462 rad (37.0 deg)

// Gate clearance & locking pocket play (exact physical clearance before stump contacts solid brass cheek)
export const GATE_HALF_HEIGHT = 0.043; // stump half-height = 0.036 -> 0.007 crisp precision daylight per side
export const GATE_ALIGN_TOL_RAD = (1.65 * Math.PI) / 180; // ~0.0288 rad exact mechanical tolerance
export const POCKET_PLAY_01 = 0.022; // 0.022 * 0.34 = 0.00748 physical play before flush cheek stop (zero clipping)

// H-Gate pocket X bounds in lever-local coordinates (actual physical brass wall positions)
export const GATE_POCKET_LEFT_OUTER_X = 0.432;
export const GATE_POCKET_LEFT_INNER_X = 0.554; // at theta=0, stump right edge 0.541 + play 0.0075 = 0.5485 < 0.554
export const GATE_POCKET_RIGHT_INNER_X = 0.788; // at theta=0, stump left edge 0.799 - play 0.0075 = 0.7915 > 0.788
export const GATE_POCKET_RIGHT_OUTER_X = 0.924; // clears rotated locked stump corner (<= 0.908) across 0..21 deg
export const GATE_POCKET_TOP_Y = 0.232; // clears stump top (0.216) at rest by 0.016, leaves 0.032 top bar
export const GATE_POCKET_BOT_R_Y = -0.184; // clears rotated locked stump bottom corner (-0.180 at 21 deg), leaves 0.034 lower bar

// Per-lever spring & inertia constants
const LEVER_INERTIA = 0.0014;
const LEVER_SPRING_PRELOAD = 0.052;
const LEVER_SPRING_STIFFNESS = 0.42;
const LEVER_DAMPING = 0.022;

// Key & bolt inertia constants
const KEY_AUTO_SPEED_RAD_S = 3.85; // crisp, weighty ~1.63s full 360-deg lock throw
const KEY_DETENT_STIFFNESS = 42.0;
const KEY_DAMPING = 11.5;
const BOLT_FRICTION = 16.0;
const BOLT_TALON_STIFFNESS = 4200.0;
const BOLT_TALON_DAMPING = 92.0;

export interface Point2D {
  readonly x: number;
  readonly y: number;
}

export interface LeverGateGeometry {
  index: number;
  alignRad: number;
  alignDeg: number;
  bitRadius: number;
  /** Keyhole center (KEY_X, KEY_Y) mapped into lever-local coordinates when lever is at alignRad */
  keyCenterLocalX: number;
  keyCenterLocalY: number;
  /** Stump center in lever-local coordinates when unlocked (boltThrow01 = 0) at alignRad */
  gateUnlockedLocalX: number;
  gateUnlockedLocalY: number;
  /** Stump center in lever-local coordinates when locked (boltThrow01 = 1) at alignRad */
  gateLockedLocalX: number;
  gateLockedLocalY: number;
  /** Stump center in lever-local coordinates at rest (theta = 0) */
  restUnlockedLocalX: number;
  restLockedLocalX: number;
  restLocalY: number;
}

/**
 * Computes exact lever-local coordinates of the bolt stump at both endpoints of travel
 * when lever `index` is lifted to its exact alignment angle `alignRad`.
 * Because the gate tunnel is milled along the line from (gateUnlockedLocal) to (gateLockedLocal),
 * rotating the lever by +alignRad brings the gate dead-level horizontally at world Y = STUMP_Y!
 */
export function computeLeverGateGeometry(index: number): LeverGateGeometry {
  const alignRad = LEVER_ALIGN_RAD[index];
  const cosA = Math.cos(alignRad);
  const sinA = Math.sin(alignRad);

  const dxU = STUMP_X_UNLOCKED - PIVOT_X;
  const dxL = STUMP_X_LOCKED - PIVOT_X;
  const dy = STUMP_Y - PIVOT_Y;

  const kdx = KEY_X - PIVOT_X;
  const kdy = KEY_Y - PIVOT_Y;

  return {
    index,
    alignRad,
    alignDeg: LEVER_ALIGN_DEG[index],
    bitRadius: KEY_BIT_RADII[index],
    keyCenterLocalX: kdx * cosA + kdy * sinA,
    keyCenterLocalY: -kdx * sinA + kdy * cosA,
    gateUnlockedLocalX: dxU * cosA + dy * sinA,
    gateUnlockedLocalY: -dxU * sinA + dy * cosA,
    gateLockedLocalX: dxL * cosA + dy * sinA,
    gateLockedLocalY: -dxL * sinA + dy * cosA,
    restUnlockedLocalX: dxU,
    restLockedLocalX: dxL,
    restLocalY: dy,
  };
}

export const LEVER_GATE_SPECS: readonly LeverGateGeometry[] = Array.from(
  { length: LEVER_COUNT },
  (_, i) => computeLeverGateGeometry(i)
);

/**
 * Single source of truth for lever `index`'s lower cam-belly polyline in lever-local coordinates
 * (left-to-right from pivot throat to lower-right rest heel).
 *
 * The middle section is an EXACT Chubb circular arc centered on `(spec.keyCenterLocalX, spec.keyCenterLocalY)`
 * with radius `spec.bitRadius` across `d ∈ [-CAM_DWELL_HALF_RAD, +CAM_DWELL_HALF_RAD]`,
 * flanked by C1-continuous flared entry/exit cam shoulders that stay strictly outside the dwell circle
 * so the rotating key bit maintains exact flush contact (0 clip, 0 air gap) across 360 degrees.
 */
function buildLeverBellyPolyline(index: number): readonly Point2D[] {
  const spec = LEVER_GATE_SPECS[index];
  const R = spec.bitRadius;
  const a = spec.alignRad;
  const pts: Point2D[] = [];

  // Arc point helper: d = PI - phi (+d is left of 12 o'clock, -d is right of 12 o'clock)
  const arcPt = (d: number, rad = R): Point2D => ({
    x: spec.keyCenterLocalX + rad * Math.sin(a - d),
    y: spec.keyCenterLocalY + rad * Math.cos(a - d),
  });

  // Left throat anchor (connects smoothly to pivot ear at bottom)
  const throatStart: Point2D = { x: 0.22, y: -0.145 };
  const arcLeft = arcPt(CAM_DWELL_HALF_RAD);
  // Tangent along the circle going left (increasing d): (-cos(a - d), sin(a - d))
  const leftAng = a - CAM_DWELL_HALF_RAD;
  const leftTanX = -Math.cos(leftAng);
  const leftTanY = Math.sin(leftAng);

  // Entry shoulder (cubic Hermite from throatStart to arcLeft, arriving tangent and flaring gently outward)
  const entryLen = Math.hypot(arcLeft.x - throatStart.x, arcLeft.y - throatStart.y);
  const c1x = throatStart.x + entryLen * 0.34;
  const c1y = throatStart.y - 0.012;
  const c2x = arcLeft.x + leftTanX * entryLen * 0.34;
  const c2y = arcLeft.y + leftTanY * entryLen * 0.34;

  const entrySegs = 10;
  for (let s = 0; s < entrySegs; s++) {
    const t = s / entrySegs;
    const it = 1 - t;
    const x =
      it * it * it * throatStart.x +
      3 * it * it * t * c1x +
      3 * it * t * t * c2x +
      t * t * t * arcLeft.x;
    const y =
      it * it * it * throatStart.y +
      3 * it * it * t * c1y +
      3 * it * t * t * c2y +
      t * t * t * arcLeft.y;
    pts.push({ x, y });
  }

  // Exact concentric Chubb circular dwell arc from +CAM_DWELL_HALF_RAD (left) to -CAM_DWELL_HALF_RAD (right)
  const arcSegs = 24;
  for (let s = 0; s <= arcSegs; s++) {
    const d = CAM_DWELL_HALF_RAD - (2 * CAM_DWELL_HALF_RAD * s) / arcSegs;
    pts.push(arcPt(d));
  }

  // Exit shoulder (cubic Hermite from arcRight to rest-heel start, departing tangent and sloping to rest heel)
  const arcRight = arcPt(-CAM_DWELL_HALF_RAD);
  const rightAng = a + CAM_DWELL_HALF_RAD;
  // Tangent along the circle going right (decreasing d): (+cos(a + d), -sin(a + d))
  const rightTanX = Math.cos(rightAng);
  const rightTanY = -Math.sin(rightAng);

  const heelStartX = Math.max(arcRight.x + 0.068, 0.815);
  const heelStart: Point2D = { x: heelStartX, y: LEVER_HEEL_LOCAL_Y };
  const exitLen = Math.hypot(heelStart.x - arcRight.x, heelStart.y - arcRight.y);
  const e1x = arcRight.x + rightTanX * Math.min(exitLen * 0.24, 0.024);
  const e1y = Math.max(LEVER_HEEL_LOCAL_Y + 0.002, arcRight.y + rightTanY * Math.min(exitLen * 0.24, 0.024));
  const e2x = heelStart.x - exitLen * 0.42;
  const e2y = LEVER_HEEL_LOCAL_Y;

  const exitSegs = 8;
  for (let s = 1; s <= exitSegs; s++) {
    const t = s / exitSegs;
    const it = 1 - t;
    const x =
      it * it * it * arcRight.x +
      3 * it * it * t * e1x +
      3 * it * t * t * e2x +
      t * t * t * heelStart.x;
    const y =
      it * it * it * arcRight.y +
      3 * it * it * t * e1y +
      3 * it * t * t * e2y +
      t * t * t * heelStart.y;
    pts.push({ x, y });
  }

  // Flat lower-right rest heel seated flush on brass rest stop pin
  pts.push({ x: 0.995, y: LEVER_HEEL_LOCAL_Y });
  return pts;
}

export const LEVER_BELLY_POLYLINES: readonly (readonly Point2D[])[] = Array.from(
  { length: LEVER_COUNT },
  (_, i) => buildLeverBellyPolyline(i)
);

/**
 * Evaluates the exact lever-local Y of lever `index`'s belly polyline at a given lever-local `x`,
 * or returns `null` if `x` is outside the polyline X-span.
 */
export function sampleLeverBellyLocalY(index: number, localX: number): number | null {
  const poly = LEVER_BELLY_POLYLINES[index];
  if (localX < poly[0].x || localX > poly[poly.length - 1].x) return null;
  for (let s = 0; s < poly.length - 1; s++) {
    const p0 = poly[s];
    const p1 = poly[s + 1];
    if (localX >= p0.x && localX <= p1.x) {
      const span = p1.x - p0.x;
      if (span < 1e-9) return Math.min(p0.y, p1.y);
      const t = (localX - p0.x) / span;
      return p0.y + (p1.y - p0.y) * t;
    }
  }
  return null;
}

/**
 * Wraps any angle in radians into [0, 2*PI).
 */
export function normalizeAngle2Pi(rad: number): number {
  const twoPi = Math.PI * 2;
  const m = rad % twoPi;
  return m < 0 ? m + twoPi : m;
}

/**
 * Exact rigid-body unilateral contact solver: finds the exact minimum lift angle `theta >= 0`
 * for lever `index` so its physical bottom belly polyline `LEVER_BELLY_POLYLINES[index]` sits
 * strictly on or above the rotating key bitting tooth `index` at phase angle `phiRad`.
 *
 * Uses exact circle-segment intersection around the lever pivot `(PIVOT_X, PIVOT_Y)` (zero allocations):
 * rotating a lever around its pivot preserves radial distance `d` from the pivot, so any belly point
 * `P(s)` at radius `d` moves from local angle `atan2(Py, Px)` to `atan2(Py, Px) + theta`.
 *
 * Guarantees:
 *  - ZERO clipping of key tooth into solid brass lever at every angle phi in [0, 2*PI)
 *  - ZERO floating air gap while lifted by the key
 *  - Exact `LEVER_ALIGN_RAD[index]` dwell across [PI - CAM_DWELL_HALF_RAD, PI + CAM_DWELL_HALF_RAD]
 */
const TOOTH_CROWN_SAMPLES: readonly number[] = [-1.0, -0.5, 0.0, 0.5, 1.0];

export function computeKeyCamLiftRad(index: number, phiRad: number): number {
  const phi = normalizeAngle2Pi(phiRad);
  const R = KEY_BIT_RADII[index];
  const poly = LEVER_BELLY_POLYLINES[index];

  // Key tooth rotates clockwise by `phi`: radial unit dir u = (-sin(phi), -cos(phi)),
  // tangential unit dir t = (cos(phi), -sin(phi)).
  // Tooth crown is machined as a concentric arc of exact radius R across half-width KEY_BIT_HALF_WIDTH
  // (spanning +/- halfAng around the tooth centerline) so its entire crown glides flush along the Chubb belly!
  const halfAng = Math.asin(Math.min(0.95, KEY_BIT_HALF_WIDTH / R));
  let maxTheta = 0;

  for (let k = 0; k < TOOTH_CROWN_SAMPLES.length; k++) {
    const samplePhi = phi + TOOTH_CROWN_SAMPLES[k] * halfAng;
    const wx = KEY_X - R * Math.sin(samplePhi);
    const wy = KEY_Y - R * Math.cos(samplePhi);

    const dx = wx - PIVOT_X;
    const dy = wy - PIVOT_Y;
    const dSq = dx * dx + dy * dy;
    const alpha = Math.atan2(dy, dx);

    // Quick reject when sample point is below lowest possible contact sweep
    if (wy < PIVOT_Y - 0.32) continue;

    // Intersect circle of radius sqrt(dSq) centered at lever pivot (0,0) with each belly segment
    for (let s = 0; s < poly.length - 1; s++) {
      const p0 = poly[s];
      const p1 = poly[s + 1];
      const vx = p1.x - p0.x;
      const vy = p1.y - p0.y;
      const a = vx * vx + vy * vy;
      if (a < 1e-12) continue;
      const b = 2 * (p0.x * vx + p0.y * vy);
      const c = p0.x * p0.x + p0.y * p0.y - dSq;
      const disc = b * b - 4 * a * c;
      if (disc < 0) continue;

      const sqrtDisc = Math.sqrt(disc);
      const inv2a = 0.5 / a;
      for (let root = 0; root < 2; root++) {
        const t = root === 0 ? (-b - sqrtDisc) * inv2a : (-b + sqrtDisc) * inv2a;
        if (t < -1e-6 || t > 1 + 1e-6) continue;
        const tc = t < 0 ? 0 : t > 1 ? 1 : t;
        const lx = p0.x + vx * tc;
        const ly = p0.y + vy * tc;
        const beta = Math.atan2(ly, lx);
        const reqTheta = alpha - beta;
        if (reqTheta > maxTheta && reqTheta < 0.75) {
          maxTheta = reqTheta;
        }
      }
    }
  }

  // Exact numerical snap inside concentric dwell arc eliminates floating-point polyline chord sagitta (< 0.0002 rad)
  const distFromTop = Math.abs(phi - Math.PI);
  if (distFromTop <= CAM_DWELL_HALF_RAD - halfAng && Math.abs(maxTheta - LEVER_ALIGN_RAD[index]) < 0.004) {
    return LEVER_ALIGN_RAD[index];
  }

  return Math.min(LEVER_MAX_RAD, maxTheta);
}

/**
 * Evaluates world-space (x, y) tip of the key's rear bolt-talon drive bit at angle `phiRad`.
 * Uses clockwise convention: phi=0 is (0, -R) down, phi=PI/2 is (-R, 0) left, phi=PI is (0, +R) up.
 */
export function computeTalonBitTip(phiRad: number, out: { x: number; y: number }): void {
  out.x = KEY_X - KEY_TALON_RADIUS * Math.sin(phiRad);
  out.y = KEY_Y - KEY_TALON_RADIUS * Math.cos(phiRad);
}

/**
 * Evaluates world-space (x, y) center of the bolt talon's U-notch for a given `boltThrow01`.
 */
export function computeTalonNotchX(boltThrow01: number): number {
  return KEY_X - BOLT_TRAVEL * 0.5 + boltThrow01 * BOLT_TRAVEL;
}

/**
 * Computes the exact physical upper angle limit for lever `index` at current `boltThrow01`
 * so manually picking a lever can never drive its bottom H-gate pocket floor into the steel bolt stump.
 */
export function computeLeverPhysicalMaxRad(index: number, boltThrow01: number): number {
  // Check locked pocket floor clearance (right pocket is the tightest at high lift)
  if (boltThrow01 > 0.5) {
    const dx = STUMP_X_UNLOCKED + boltThrow01 * BOLT_TRAVEL - PIVOT_X;
    const dy = STUMP_Y - PIVOT_Y;
    // Ensure lowest rotated corner of bolt stump stays >= GATE_POCKET_BOT_R_Y
    const hw = STUMP_WIDTH * 0.5;
    const hh = STUMP_HEIGHT * 0.5;
    // Monotonic decreasing with theta; check LEVER_MAX_RAD first (fast path)
    const testCornerY = (th: number) => {
      const c = Math.cos(th);
      const s = Math.sin(th);
      return -dx * s + dy * c - (hw * s + hh * c);
    };
    if (testCornerY(LEVER_MAX_RAD) >= GATE_POCKET_BOT_R_Y) {
      return LEVER_MAX_RAD;
    }
    let lo = LEVER_ALIGN_RAD[index];
    let hi = LEVER_MAX_RAD;
    for (let it = 0; it < 12; it++) {
      const mid = (lo + hi) * 0.5;
      if (testCornerY(mid) >= GATE_POCKET_BOT_R_Y) lo = mid;
      else hi = mid;
    }
    return lo;
  }
  return LEVER_MAX_RAD;
}

export interface LockDiagnosticsState {
  keyAngleRad: number;
  boltThrow01: number;
  isLocked: boolean;
  leverAnglesDeg: number[];
  gatesAligned: boolean;
  boltBlocked: boolean;
  bindingLeverIndex: number;
  autoTurning: boolean;
  gearAnglesRad: {
    upperPinionU1: number;
    upperDetentU2: number;
    upperPawl: number;
    curtainB1: number;
    curtainIdlerB2: number;
    curtainCompoundB3: number;
    keyHomeWheelB4: number;
    lowerPawl: number;
  };
}

// =============================================================================
// Horological Geared Deadbolt Drive Train & Positive Detent Escapement Specs
// Single source of truth shared between physics, 3D involute geometry & Vitest!
// =============================================================================

export interface SpurGearSpec {
  readonly name: string;
  readonly cx: number;
  readonly cy: number;
  readonly module: number;
  readonly teeth: number;
  readonly pitchRadius: number;
  readonly baseRadius: number;
  readonly tipRadius: number;
  readonly rootRadius: number;
  readonly zBack: number;
  readonly zFront: number;
}

export interface RackSpec {
  readonly module: number;
  readonly pitch: number;
  readonly toothCount: number;
  readonly firstToothLocalX: number;
  readonly pitchLineY: number;
  readonly tipY: number;
  readonly rootY: number;
  readonly barBaseY: number;
  readonly barMinLocalX: number;
  readonly barMaxLocalX: number;
  readonly zBack: number;
  readonly zFront: number;
}

export interface DetentCamSpec {
  readonly cx: number;
  readonly cy: number;
  readonly outerRadius: number;
  readonly rootRadius: number;
  readonly notchHalfWidthRad: number;
  readonly notchLocalAngles: readonly number[];
  readonly rippleCount: number;
  readonly rippleAmp: number;
  readonly pivotX: number;
  readonly pivotY: number;
  readonly pawlArmLength: number;
  readonly beakRadius: number;
  readonly pawlSign: 1 | -1;
  readonly springPinX: number;
  readonly springPinY: number;
  readonly zBack: number;
  readonly zFront: number;
}

export const GEAR_PRESSURE_ANGLE_RAD = (20 * Math.PI) / 180;
export const GEAR_ADDENDUM_FACTOR = 0.85;
export const GEAR_DEDENDUM_FACTOR = 1.15;
export const GEAR_TOOTH_THICKNESS_RATIO = 0.46; // crisp running backlash -> zero flank clipping

function makeSpurGearSpec(
  name: string,
  cx: number,
  cy: number,
  module: number,
  teeth: number,
  zBack: number,
  zFront: number
): SpurGearSpec {
  const pitchRadius = (module * teeth) * 0.5;
  return {
    name,
    cx,
    cy,
    module,
    teeth,
    pitchRadius,
    baseRadius: pitchRadius * Math.cos(GEAR_PRESSURE_ANGLE_RAD),
    tipRadius: pitchRadius + GEAR_ADDENDUM_FACTOR * module,
    rootRadius: pitchRadius - GEAR_DEDENDUM_FACTOR * module,
    zBack,
    zFront,
  };
}

// --- TRAIN A (Upper-Right): Deadbolt Upper Rack -> U1A/U1B -> U2A/U2B Detent Wheel + Upper Detent Pawl ---
export const UPPER_RACK_MODULE = 0.014;
export const UPPER_RACK_PITCH = Math.PI * UPPER_RACK_MODULE;
export const UPPER_RACK_SPEC: RackSpec = {
  module: UPPER_RACK_MODULE,
  pitch: UPPER_RACK_PITCH,
  toothCount: 12,
  firstToothLocalX: 0.008,
  pitchLineY: 0.348,
  tipY: 0.348 + GEAR_ADDENDUM_FACTOR * UPPER_RACK_MODULE,
  rootY: 0.348 - GEAR_DEDENDUM_FACTOR * UPPER_RACK_MODULE,
  barBaseY: 0.240,
  barMinLocalX: -0.014,
  barMaxLocalX: 0.518,
  zBack: -0.156,
  zFront: -0.114,
};

const U1_TEETH_RACK = 12;
const U1_RACK_PITCH_R = (UPPER_RACK_MODULE * U1_TEETH_RACK) * 0.5; // 0.084
export const ARBOR_U1_X = 0.425;
export const ARBOR_U1_Y = UPPER_RACK_SPEC.pitchLineY + U1_RACK_PITCH_R; // 0.432

export const UPPER_STAGE_MODULE = 0.0115;
const U1B_TEETH = 16;
const U2A_TEETH = 18;
const U12_DIST = (UPPER_STAGE_MODULE * (U1B_TEETH + U2A_TEETH)) * 0.5; // 0.1955
export const UPPER_U12_ANGLE_RAD = 0.095;
export const ARBOR_U2_X = ARBOR_U1_X + U12_DIST * Math.cos(UPPER_U12_ANGLE_RAD);
export const ARBOR_U2_Y = ARBOR_U1_Y + U12_DIST * Math.sin(UPPER_U12_ANGLE_RAD);

export const GEAR_U1A_PINION = makeSpurGearSpec(
  'RackThrowPinion_U1A',
  ARBOR_U1_X,
  ARBOR_U1_Y,
  UPPER_RACK_MODULE,
  U1_TEETH_RACK,
  -0.156,
  -0.114
);

export const GEAR_U1B_SPUR = makeSpurGearSpec(
  'CompoundThrowSpur_U1B',
  ARBOR_U1_X,
  ARBOR_U1_Y,
  UPPER_STAGE_MODULE,
  U1B_TEETH,
  -0.106,
  -0.068
);

export const GEAR_U2A_DRIVEN = makeSpurGearSpec(
  'DetentDrivenWheel_U2A',
  ARBOR_U2_X,
  ARBOR_U2_Y,
  UPPER_STAGE_MODULE,
  U2A_TEETH,
  -0.106,
  -0.068
);

// --- TRAIN B (Lower-Right & Key Hub): B1 Curtain -> B2 Bridge Idler -> B3f/B3r Compound -> B4 Key-Home Wheel ---
// Exact compound gear ratio: (24/20)*(20/18)*(15/20) = (4/3)*(3/4) = 1.000000 (exact 1:1 opposite rotation with key!)
export const CURTAIN_STAGE_MODULE = 0.0105;
const B1_TEETH = 24;
const B2_TEETH = 20;
const B3F_TEETH = 18;
const B12_DIST = (CURTAIN_STAGE_MODULE * (B1_TEETH + B2_TEETH)) * 0.5; // 0.231
export const LOWER_B12_ANGLE_RAD = -0.74;
export const ARBOR_B2_X = KEY_X + B12_DIST * Math.cos(LOWER_B12_ANGLE_RAD);
export const ARBOR_B2_Y = KEY_Y + B12_DIST * Math.sin(LOWER_B12_ANGLE_RAD);

const B23_DIST = (CURTAIN_STAGE_MODULE * (B2_TEETH + B3F_TEETH)) * 0.5; // 0.1995
export const LOWER_B23_ANGLE_RAD = -0.30;
export const ARBOR_B3_X = ARBOR_B2_X + B23_DIST * Math.cos(LOWER_B23_ANGLE_RAD);
export const ARBOR_B3_Y = ARBOR_B2_Y + B23_DIST * Math.sin(LOWER_B23_ANGLE_RAD);

export const KEYHOME_STAGE_MODULE = 0.0135;
const B3R_TEETH = 15;
const B4A_TEETH = 20;
const B34_DIST = (KEYHOME_STAGE_MODULE * (B3R_TEETH + B4A_TEETH)) * 0.5; // 0.23625
export const LOWER_B34_ANGLE_RAD = 0.52;
export const ARBOR_B4_X = ARBOR_B3_X + B34_DIST * Math.cos(LOWER_B34_ANGLE_RAD);
export const ARBOR_B4_Y = ARBOR_B3_Y + B34_DIST * Math.sin(LOWER_B34_ANGLE_RAD);

export const GEAR_B1_CURTAIN = makeSpurGearSpec(
  'KeyholeCurtainGear_B1',
  KEY_X,
  KEY_Y,
  CURTAIN_STAGE_MODULE,
  B1_TEETH,
  0.156,
  0.180
);

export const GEAR_B2_IDLER = makeSpurGearSpec(
  'CurtainTransferIdler_B2',
  ARBOR_B2_X,
  ARBOR_B2_Y,
  CURTAIN_STAGE_MODULE,
  B2_TEETH,
  0.156,
  0.180
);

export const GEAR_B3F_STEP = makeSpurGearSpec(
  'CurtainStepGear_B3F',
  ARBOR_B3_X,
  ARBOR_B3_Y,
  CURTAIN_STAGE_MODULE,
  B3F_TEETH,
  0.156,
  0.180
);

export const GEAR_B3R_PINION = makeSpurGearSpec(
  'CompoundDrivePinion_B3R',
  ARBOR_B3_X,
  ARBOR_B3_Y,
  KEYHOME_STAGE_MODULE,
  B3R_TEETH,
  0.048,
  0.104
);

export const GEAR_B4A_KEYHOME = makeSpurGearSpec(
  'SpokedKeyHomeWheel_B4A',
  ARBOR_B4_X,
  ARBOR_B4_Y,
  KEYHOME_STAGE_MODULE,
  B4A_TEETH,
  0.048,
  0.104
);

/**
 * Computes exact Three.js right-handed CCW rotation angles for an externally meshed driven gear
 * so tooth-center of driver meets exact valley-center of driven gear along line-of-centers `beta`.
 */
export function computeMeshedGearAngle(
  driverAngleRad: number,
  driverTeeth: number,
  drivenTeeth: number,
  lineOfCentersRad: number
): number {
  return (
    lineOfCentersRad +
    Math.PI -
    Math.PI / drivenTeeth -
    (driverTeeth / drivenTeeth) * (driverAngleRad - lineOfCentersRad)
  );
}

/**
 * Exact right-handed CCW angle of Pinion U1A meshing cleanly on top of DeadboltUpperRack at `boltThrow01`.
 */
export function computeUpperPinionAngleRad(boltThrow01: number): number {
  const boltX = boltThrow01 * BOLT_TRAVEL;
  return (
    -Math.PI * 0.5 +
    Math.PI / GEAR_U1A_PINION.teeth +
    (boltX + UPPER_RACK_SPEC.firstToothLocalX - GEAR_U1A_PINION.cx) / GEAR_U1A_PINION.pitchRadius
  );
}

export function computeUpperDetentWheelAngleRad(boltThrow01: number): number {
  const u1 = computeUpperPinionAngleRad(boltThrow01);
  return computeMeshedGearAngle(
    u1,
    GEAR_U1B_SPUR.teeth,
    GEAR_U2A_DRIVEN.teeth,
    UPPER_U12_ANGLE_RAD
  );
}

export function computeLowerTrainAnglesRad(keyAngleRad: number): {
  b1: number;
  b2: number;
  b3: number;
  b4: number;
} {
  // Key rotates clockwise (+keyAngleRad) -> right-handed Three.js rotation.z = -keyAngleRad
  const b1 = -keyAngleRad;
  const b2 = computeMeshedGearAngle(b1, GEAR_B1_CURTAIN.teeth, GEAR_B2_IDLER.teeth, LOWER_B12_ANGLE_RAD);
  const b3 = computeMeshedGearAngle(b2, GEAR_B2_IDLER.teeth, GEAR_B3F_STEP.teeth, LOWER_B23_ANGLE_RAD);
  const b4 = computeMeshedGearAngle(b3, GEAR_B3R_PINION.teeth, GEAR_B4A_KEYHOME.teeth, LOWER_B34_ANGLE_RAD);
  return { b1, b2, b3, b4 };
}

function wrapAnglePi(a: number): number {
  let x = a % (Math.PI * 2);
  if (x > Math.PI) x -= Math.PI * 2;
  if (x < -Math.PI) x += Math.PI * 2;
  return x;
}

/**
 * Computes exact seated world contact angle (from cam center toward beak center) at a given cam radius `rc`.
 */
function computePawlSeatedContactWorldAngle(
  cx: number,
  cy: number,
  pivotX: number,
  pivotY: number,
  armLength: number,
  rc: number,
  sign: 1 | -1
): number {
  const dx = pivotX - cx;
  const dy = pivotY - cy;
  const d = Math.hypot(dx, dy);
  const cosGammaC = Math.max(-1, Math.min(1, (d * d + rc * rc - armLength * armLength) / (2 * d * rc)));
  return Math.atan2(dy, dx) + sign * Math.acos(cosGammaC);
}

// Compute exact V-notch local angles so the pawls drop DEAD-FLUSH at unlocked(0), locked(1), and keyHome(0)!
const UPPER_PAWL_PIVOT_X = 0.788;
const UPPER_PAWL_PIVOT_Y = 0.402;
const UPPER_PAWL_ARM_LEN = 0.128;
const UPPER_CAM_OUTER_R = 0.094;
const UPPER_CAM_ROOT_R = 0.074;
const PAWL_BEAK_R = 0.010;

const UPPER_SEATED_WORLD_ANG = computePawlSeatedContactWorldAngle(
  ARBOR_U2_X,
  ARBOR_U2_Y,
  UPPER_PAWL_PIVOT_X,
  UPPER_PAWL_PIVOT_Y,
  UPPER_PAWL_ARM_LEN,
  UPPER_CAM_ROOT_R + PAWL_BEAK_R,
  1
);

export const UPPER_DETENT_CAM_SPEC: DetentCamSpec = {
  cx: ARBOR_U2_X,
  cy: ARBOR_U2_Y,
  outerRadius: UPPER_CAM_OUTER_R,
  rootRadius: UPPER_CAM_ROOT_R,
  notchHalfWidthRad: 0.25,
  notchLocalAngles: [
    wrapAnglePi(UPPER_SEATED_WORLD_ANG - computeUpperDetentWheelAngleRad(0)),
    wrapAnglePi(UPPER_SEATED_WORLD_ANG - computeUpperDetentWheelAngleRad(1)),
  ],
  rippleCount: 18,
  rippleAmp: 0.0022,
  pivotX: UPPER_PAWL_PIVOT_X,
  pivotY: UPPER_PAWL_PIVOT_Y,
  pawlArmLength: UPPER_PAWL_ARM_LEN,
  beakRadius: PAWL_BEAK_R,
  pawlSign: 1,
  springPinX: 0.818,
  springPinY: 0.518,
  zBack: 0.018,
  zFront: 0.088,
};

const LOWER_PAWL_PIVOT_X = 0.774;
const LOWER_PAWL_PIVOT_Y = -0.365;
const LOWER_PAWL_ARM_LEN = 0.128;
const LOWER_CAM_OUTER_R = 0.094;
const LOWER_CAM_ROOT_R = 0.074;

const LOWER_SEATED_WORLD_ANG = computePawlSeatedContactWorldAngle(
  ARBOR_B4_X,
  ARBOR_B4_Y,
  LOWER_PAWL_PIVOT_X,
  LOWER_PAWL_PIVOT_Y,
  LOWER_PAWL_ARM_LEN,
  LOWER_CAM_ROOT_R + PAWL_BEAK_R,
  1
);

export const LOWER_KEYHOME_CAM_SPEC: DetentCamSpec = {
  cx: ARBOR_B4_X,
  cy: ARBOR_B4_Y,
  outerRadius: LOWER_CAM_OUTER_R,
  rootRadius: LOWER_CAM_ROOT_R,
  notchHalfWidthRad: 0.28,
  notchLocalAngles: [
    wrapAnglePi(LOWER_SEATED_WORLD_ANG - computeLowerTrainAnglesRad(0).b4),
  ],
  rippleCount: 0,
  rippleAmp: 0,
  pivotX: LOWER_PAWL_PIVOT_X,
  pivotY: LOWER_PAWL_PIVOT_Y,
  pawlArmLength: LOWER_PAWL_ARM_LEN,
  beakRadius: PAWL_BEAK_R,
  pawlSign: 1,
  springPinX: 0.806,
  springPinY: -0.248,
  zBack: 0.114,
  zFront: 0.160,
};

/**
 * Evaluates the exact physical outer radius of a detent cam at wheel-local angle `localAngleRad`.
 * Shared identically by the 3D cam mesh builder and the real-time pawl contact solver (0 clip, 0 gap!).
 */
export function evalDetentCamRadius(spec: DetentCamSpec, localAngleRad: number): number {
  let minDist = Infinity;
  for (let i = 0; i < spec.notchLocalAngles.length; i++) {
    const d = Math.abs(wrapAnglePi(localAngleRad - spec.notchLocalAngles[i]));
    if (d < minDist) minDist = d;
  }
  const hw = spec.notchHalfWidthRad;
  if (minDist < hw) {
    const u = minDist / hw;
    // Smooth V-cradle profile matching rounded beak radius cleanly at bottom & tangent at shoulders
    const s = u * u * (3 - 2 * u);
    return spec.rootRadius + (spec.outerRadius - spec.rootRadius) * s;
  }
  if (spec.rippleCount > 0 && spec.rippleAmp > 0) {
    // Feather ripples gently out near V-notch shoulders for crisp V-entry lips
    const shoulderFade = Math.min(1, (minDist - hw) / 0.14);
    const wave = 0.5 - 0.5 * Math.cos(localAngleRad * spec.rippleCount);
    return spec.outerRadius - spec.rippleAmp * wave * shoulderFade;
  }
  return spec.outerRadius;
}

export interface PawlContactSolution {
  pawlWorldAngleRad: number;
  beakWorldX: number;
  beakWorldY: number;
  contactWorldAngleRad: number;
  camRadiusAtContact: number;
  lift01: number;
}

/**
 * Analytical circle-cam contact solver: computes exact pawl rotation angle and beak center
 * so the rounded steel pawl beak rides strictly flush along the detent cam perimeter & drops into V-notches!
 */
export function solveDetentPawlContact(
  spec: DetentCamSpec,
  camWorldAngleRad: number,
  out: PawlContactSolution
): void {
  const dx = spec.pivotX - spec.cx;
  const dy = spec.pivotY - spec.cy;
  const d = Math.hypot(dx, dy);
  const angCtoP = Math.atan2(dy, dx);
  const angPtoC = Math.atan2(-dy, -dx);

  let rc = spec.outerRadius + spec.beakRadius;
  let contactWorldAngle = angCtoP;
  let camR = spec.outerRadius;

  for (let iter = 0; iter < 6; iter++) {
    const cosGammaC = Math.max(
      -1,
      Math.min(1, (d * d + rc * rc - spec.pawlArmLength * spec.pawlArmLength) / (2 * d * rc))
    );
    contactWorldAngle = angCtoP + spec.pawlSign * Math.acos(cosGammaC);
    const localAng = contactWorldAngle - camWorldAngleRad;
    camR = evalDetentCamRadius(spec, localAng);
    rc = camR + spec.beakRadius;
  }

  const cosGammaP = Math.max(
    -1,
    Math.min(1, (d * d + spec.pawlArmLength * spec.pawlArmLength - rc * rc) / (2 * d * spec.pawlArmLength))
  );
  const pawlWorldAngle = angPtoC - spec.pawlSign * Math.acos(cosGammaP);

  out.pawlWorldAngleRad = pawlWorldAngle;
  out.beakWorldX = spec.pivotX + Math.cos(pawlWorldAngle) * spec.pawlArmLength;
  out.beakWorldY = spec.pivotY + Math.sin(pawlWorldAngle) * spec.pawlArmLength;
  out.contactWorldAngleRad = contactWorldAngle;
  out.camRadiusAtContact = camR;
  out.lift01 = Math.max(
    0,
    Math.min(1, (camR - spec.rootRadius) / Math.max(1e-5, spec.outerRadius - spec.rootRadius))
  );
}

/**
 * Generates exact CCW 2D involute spur-gear outer contour points in gear-local XY (center at (0, 0),
 * tooth #0 centered on +X at angle 0). Shared directly by 3D geometry & Vitest tooth-overlap proofs!
 */
export function buildInvoluteGearPolygon2D(spec: SpurGearSpec, flankSegs = 4): Point2D[] {
  const Z = spec.teeth;
  const rb = spec.baseRadius;
  const ra = spec.tipRadius;
  const rf = spec.rootRadius;
  const step = (Math.PI * 2) / Z;

  const invAlphaP = Math.tan(GEAR_PRESSURE_ANGLE_RAD) - GEAR_PRESSURE_ANGLE_RAD;
  const halfPitchAng = (GEAR_TOOTH_THICKNESS_RATIO * Math.PI) / Z;
  const psiBase = halfPitchAng + invAlphaP;

  const rStart = Math.max(rb, rf);
  const tStart = rStart > rb ? Math.sqrt(Math.max(0, (rStart / rb) * (rStart / rb) - 1)) : 0;
  const tMax = Math.sqrt(Math.max(0, (ra / rb) * (ra / rb) - 1));
  const phiStart = tStart - Math.atan(tStart);
  const rootFlankHalfAng = psiBase - phiStart;

  const pts: Point2D[] = [];
  for (let k = 0; k < Z; k++) {
    const centerA = k * step;

    // Root valley leading into right (lower) flank of tooth k
    const aRootRight = centerA - rootFlankHalfAng;
    const prevValleyMid = centerA - step * 0.5;
    pts.push({ x: Math.cos(prevValleyMid) * rf, y: Math.sin(prevValleyMid) * rf });
    if (rf < rb - 1e-6) {
      pts.push({ x: Math.cos(aRootRight) * rf, y: Math.sin(aRootRight) * rf });
    }

    // Right involute flank ascending from rStart -> ra
    for (let s = 0; s <= flankSegs; s++) {
      const t = tStart + (tMax - tStart) * (s / flankSegs);
      const r = rb * Math.sqrt(1 + t * t);
      const phi = t - Math.atan(t);
      const a = centerA - (psiBase - phi);
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }

    // Left involute flank descending from ra -> rStart
    for (let s = flankSegs; s >= 0; s--) {
      const t = tStart + (tMax - tStart) * (s / flankSegs);
      const r = rb * Math.sqrt(1 + t * t);
      const phi = t - Math.atan(t);
      const a = centerA + (psiBase - phi);
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }

    // Root fillet down from rb to rf on left (upper) flank of tooth k
    if (rf < rb - 1e-6) {
      const aRootLeft = centerA + rootFlankHalfAng;
      pts.push({ x: Math.cos(aRootLeft) * rf, y: Math.sin(aRootLeft) * rf });
    }
  }
  return pts;
}

/**
 * Generates exact CCW 2D involute linear rack outer contour in bolt-local XY (teeth pointing +Y).
 * Shared directly by DeadboltUpperRack 3D mesh & Vitest rack-to-pinion zero-clip proofs!
 */
export function buildInvoluteRackPolygon2D(spec: RackSpec): Point2D[] {
  const tanAlpha = Math.tan(GEAR_PRESSURE_ANGLE_RAD);
  const hwPitch = 0.5 * GEAR_TOOTH_THICKNESS_RATIO * spec.pitch;
  const ha = spec.tipY - spec.pitchLineY;
  const hf = spec.pitchLineY - spec.rootY;
  const hwTip = Math.max(0.001, hwPitch - ha * tanAlpha);
  const hwRoot = Math.min(spec.pitch * 0.48, hwPitch + hf * tanAlpha);

  const pts: Point2D[] = [
    { x: spec.barMinLocalX, y: spec.barBaseY },
    { x: spec.barMaxLocalX, y: spec.barBaseY },
    { x: spec.barMaxLocalX, y: spec.rootY },
  ];

  // Traverse teeth right-to-left along top (+Y) to maintain strict CCW winding
  for (let k = spec.toothCount - 1; k >= 0; k--) {
    const cx = spec.firstToothLocalX + k * spec.pitch;
    pts.push({ x: cx + hwRoot, y: spec.rootY });
    pts.push({ x: cx + hwTip, y: spec.tipY });
    pts.push({ x: cx - hwTip, y: spec.tipY });
    pts.push({ x: cx - hwRoot, y: spec.rootY });
  }

  pts.push({ x: spec.barMinLocalX, y: spec.rootY });
  return pts;
}

export class LockMechanism {
  // Primary physical states (boots cleanly LOCKED so the massive blued-steel deadbolt is proudly extended!)
  keyAngleRad = 0;
  keyAngularVel = 0;
  boltThrow01 = 1.0;
  boltVel = 0;

  readonly leverAnglesRad = new Float64Array(LEVER_COUNT);
  readonly leverAngVels = new Float64Array(LEVER_COUNT);
  readonly leverSpringFlex01 = new Float64Array(LEVER_COUNT);
  readonly leverBinding01 = new Float64Array(LEVER_COUNT);

  gatesAligned = false;
  boltBlocked = false;
  bindingLeverIndex = -1;
  detentPulse = 0;
  caseRecoilX = 0;
  caseRecoilVelX = 0;

  // Coupled horological gear-train & detent-pawl kinematic states (zero allocations)
  upperPinionAngleRad = 0;
  upperDetentAngleRad = 0;
  upperPawlAngleRad = 0;
  upperPawlLift01 = 0;
  curtainB1AngleRad = 0;
  curtainB2AngleRad = 0;
  curtainB3AngleRad = 0;
  keyHomeB4AngleRad = 0;
  lowerPawlAngleRad = 0;
  lowerPawlLift01 = 0;

  // Interactive control states
  private isDraggingKey = false;
  private keyDragTargetRad = 0;

  private draggedLeverIndex = -1;
  private draggedLeverTargetRad = 0;

  private isDraggingBolt = false;
  private boltDragTarget01 = 1.0;

  // Motorized full 360-degree click/Space turn state
  private autoTurnActive = false;
  private autoTurnTargetRad = 0;

  // Scratch objects (strictly zero per-frame allocations)
  private readonly _talonTip = { x: 0, y: 0 };
  private readonly _upperPawlSol: PawlContactSolution = {
    pawlWorldAngleRad: 0,
    beakWorldX: 0,
    beakWorldY: 0,
    contactWorldAngleRad: 0,
    camRadiusAtContact: UPPER_DETENT_CAM_SPEC.rootRadius,
    lift01: 0,
  };
  private readonly _lowerPawlSol: PawlContactSolution = {
    pawlWorldAngleRad: 0,
    beakWorldX: 0,
    beakWorldY: 0,
    contactWorldAngleRad: 0,
    camRadiusAtContact: LOWER_KEYHOME_CAM_SPEC.rootRadius,
    lift01: 0,
  };

  constructor(initiallyLocked = true) {
    this.boltThrow01 = initiallyLocked ? 1.0 : 0.0;
    this.boltDragTarget01 = this.boltThrow01;
    this.refreshAlignmentFlags();
    this.syncGearTrainStates();
  }

  /**
   * Synchronizes all coupled gear-train angles & spring-loaded detent-pawl contact poses
   * from the exact physical bolt position and key rotation (strictly zero allocations).
   */
  syncGearTrainStates(): void {
    this.upperPinionAngleRad = computeUpperPinionAngleRad(this.boltThrow01);
    this.upperDetentAngleRad = computeMeshedGearAngle(
      this.upperPinionAngleRad,
      GEAR_U1B_SPUR.teeth,
      GEAR_U2A_DRIVEN.teeth,
      UPPER_U12_ANGLE_RAD
    );
    solveDetentPawlContact(UPPER_DETENT_CAM_SPEC, this.upperDetentAngleRad, this._upperPawlSol);
    this.upperPawlAngleRad = this._upperPawlSol.pawlWorldAngleRad;
    this.upperPawlLift01 = this._upperPawlSol.lift01;

    this.curtainB1AngleRad = -this.keyAngleRad;
    this.curtainB2AngleRad = computeMeshedGearAngle(
      this.curtainB1AngleRad,
      GEAR_B1_CURTAIN.teeth,
      GEAR_B2_IDLER.teeth,
      LOWER_B12_ANGLE_RAD
    );
    this.curtainB3AngleRad = computeMeshedGearAngle(
      this.curtainB2AngleRad,
      GEAR_B2_IDLER.teeth,
      GEAR_B3F_STEP.teeth,
      LOWER_B23_ANGLE_RAD
    );
    this.keyHomeB4AngleRad = computeMeshedGearAngle(
      this.curtainB3AngleRad,
      GEAR_B3R_PINION.teeth,
      GEAR_B4A_KEYHOME.teeth,
      LOWER_B34_ANGLE_RAD
    );
    solveDetentPawlContact(LOWER_KEYHOME_CAM_SPEC, this.keyHomeB4AngleRad, this._lowerPawlSol);
    this.lowerPawlAngleRad = this._lowerPawlSol.pawlWorldAngleRad;
    this.lowerPawlLift01 = this._lowerPawlSol.lift01;
  }

  get isLocked(): boolean {
    return this.boltThrow01 >= 0.5;
  }

  get isAutoTurning(): boolean {
    return this.autoTurnActive;
  }

  /**
   * Starts a smooth motorized 360-degree key revolution in the mechanically valid direction:
   *  - When unlocked: turns clockwise (+2*PI) to lock/extend the deadbolt.
   *  - When locked: turns counter-clockwise (-2*PI) to unlock/retract the deadbolt.
   */
  triggerToggleTurn(): void {
    const twoPi = Math.PI * 2;
    const baseTurn = Math.round(this.keyAngleRad / twoPi) * twoPi;
    const dir = this.isLocked ? -1 : 1;
    this.autoTurnTargetRad = baseTurn + dir * twoPi;
    this.autoTurnActive = true;
    this.isDraggingKey = false;
    this.draggedLeverIndex = -1;
    this.isDraggingBolt = false;
  }

  /**
   * Begins manual circular key dragging.
   */
  beginKeyDrag(): void {
    this.isDraggingKey = true;
    this.autoTurnActive = false;
    this.keyDragTargetRad = this.keyAngleRad;
  }

  /**
   * Applies an incremental circular drag delta (radians, + = clockwise) while dragging the key.
   */
  applyKeyDragDelta(deltaRad: number): void {
    if (!this.isDraggingKey) {
      this.beginKeyDrag();
    }
    this.keyDragTargetRad += deltaRad;
  }

  /**
   * Releases manual key drag; key settles back toward resting vertical keyhole if near 0/2PI,
   * or holds cleanly wherever released if parked mid-turn for gate inspection!
   */
  endKeyDrag(): void {
    this.isDraggingKey = false;
    const phi = normalizeAngle2Pi(this.keyAngleRad);
    const distRest = Math.min(phi, Math.PI * 2 - phi);
    if (distRest < 0.38) {
      const twoPi = Math.PI * 2;
      this.keyDragTargetRad = Math.round(this.keyAngleRad / twoPi) * twoPi;
    }
  }

  /**
   * Applies a discrete scroll-wheel nudge to key rotation.
   */
  nudgeKeyAngle(deltaRad: number): void {
    this.autoTurnActive = false;
    this.isDraggingKey = false;
    this.keyAngularVel += deltaRad * 6.5;
  }

  /**
   * Begins direct manual lifting of a single brass lever (interactive lock-picking / inspection).
   */
  beginLeverDrag(index: number, targetRad?: number): void {
    if (index < 0 || index >= LEVER_COUNT) return;
    this.autoTurnActive = false;
    this.draggedLeverIndex = index;
    const maxLift = computeLeverPhysicalMaxRad(index, this.boltThrow01);
    this.draggedLeverTargetRad = Math.max(
      0,
      Math.min(maxLift, targetRad ?? this.leverAnglesRad[index])
    );
  }

  setLeverDragTarget(targetRad: number): void {
    if (this.draggedLeverIndex < 0) return;
    const maxLift = computeLeverPhysicalMaxRad(this.draggedLeverIndex, this.boltThrow01);
    this.draggedLeverTargetRad = Math.max(0, Math.min(maxLift, targetRad));
  }

  endLeverDrag(): void {
    this.draggedLeverIndex = -1;
  }

  /**
   * Begins direct manual push/pull on the deadbolt head or bolt stump to test locking security.
   */
  beginBoltDrag(): void {
    this.autoTurnActive = false;
    this.isDraggingBolt = true;
    this.boltDragTarget01 = this.boltThrow01;
  }

  setBoltDragTarget(target01: number): void {
    if (!this.isDraggingBolt) {
      this.beginBoltDrag();
    }
    this.boltDragTarget01 = Math.max(-0.08, Math.min(1.08, target01));
  }

  endBoltDrag(): void {
    this.isDraggingBolt = false;
  }

  /**
   * Directly sets a lever angle (useful for deterministic unit tests & inspection).
   */
  setLeverAngleImmediate(index: number, rad: number): void {
    if (index < 0 || index >= LEVER_COUNT) return;
    const maxLift = computeLeverPhysicalMaxRad(index, this.boltThrow01);
    this.leverAnglesRad[index] = Math.max(0, Math.min(maxLift, rad));
    this.leverAngVels[index] = 0;
    this.leverSpringFlex01[index] = this.leverAnglesRad[index] / LEVER_MAX_RAD;
    this.refreshAlignmentFlags();
  }

  /**
   * Checks whether all 5 lever gates are simultaneously aligned with the bolt stump
   * within physical machining clearance (`GATE_ALIGN_TOL_RAD`).
   */
  checkAllGatesAligned(): boolean {
    for (let i = 0; i < LEVER_COUNT; i++) {
      if (Math.abs(this.leverAnglesRad[i] - LEVER_ALIGN_RAD[i]) > GATE_ALIGN_TOL_RAD) {
        return false;
      }
    }
    return true;
  }

  /**
   * Evaluates restoring spring torque (N*m scaled) exerted downward on lever `index` by its
   * preloaded curved phosphor-bronze leaf spring.
   */
  getLeafSpringRestoringTorque(index: number): number {
    const theta = Math.max(0, this.leverAnglesRad[index]);
    return LEVER_SPRING_PRELOAD + LEVER_SPRING_STIFFNESS * theta;
  }

  private refreshAlignmentFlags(): void {
    let aligned = true;
    let worstIdx = -1;
    let worstErr = 0;
    for (let i = 0; i < LEVER_COUNT; i++) {
      const err = Math.abs(this.leverAnglesRad[i] - LEVER_ALIGN_RAD[i]);
      if (err > GATE_ALIGN_TOL_RAD) {
        aligned = false;
        if (err > worstErr) {
          worstErr = err;
          worstIdx = i;
        }
      }
    }
    this.gatesAligned = aligned;
    if (aligned) {
      this.bindingLeverIndex = -1;
    } else if (this.boltBlocked) {
      this.bindingLeverIndex = worstIdx;
    }
  }

  /**
   * Advances the entire coupled 5-lever lock mechanism with fixed-timestep substeps
   * for rock-solid contact stability at any frame rate (60Hz..240Hz).
   */
  step(dtSec: number): void {
    const clampedDt = Math.min(0.05, Math.max(0.0005, dtSec));
    const substeps = Math.max(2, Math.ceil(clampedDt / (1 / 240)));
    const h = clampedDt / substeps;

    for (let s = 0; s < substeps; s++) {
      this.stepSub(h);
    }
  }

  private stepSub(h: number): void {
    const prevKeyAngle = this.keyAngleRad;
    let blockedThisSub = false;

    // 1. Key angular actuation (motorized 360 turn, manual circular drag, or free detent settle)
    if (this.autoTurnActive) {
      const rem = this.autoTurnTargetRad - this.keyAngleRad;
      const dist = Math.abs(rem);
      if (dist <= 0.004) {
        this.keyAngleRad = this.autoTurnTargetRad;
        this.keyAngularVel = 0;
        this.autoTurnActive = false;
      } else {
        const phi = normalizeAngle2Pi(this.keyAngleRad);
        const inThrow = phi > TALON_ENGAGE_START_RAD && phi < TALON_ENGAGE_END_RAD;
        const speedProfile = inThrow ? 0.72 : 1.0;
        const maxStep = KEY_AUTO_SPEED_RAD_S * speedProfile;
        const desiredVel = Math.sign(rem) * Math.min(maxStep, Math.max(0.65, dist * 7.5));
        this.keyAngularVel += (desiredVel - this.keyAngularVel) * Math.min(1, h * 18);
        this.keyAngleRad += this.keyAngularVel * h;
      }
    } else if (this.isDraggingKey) {
      const err = this.keyDragTargetRad - this.keyAngleRad;
      const phi = normalizeAngle2Pi(this.keyAngleRad);
      const liftLoad = Math.abs(phi - Math.PI) < CAM_DWELL_HALF_RAD + 0.95 ? 0.78 : 1.0;
      const stiffness = 145.0 * liftLoad;
      const accel = err * stiffness - this.keyAngularVel * 20.0;
      this.keyAngularVel += accel * h;
      this.keyAngleRad += this.keyAngularVel * h;
    } else {
      const phi = normalizeAngle2Pi(this.keyAngleRad);
      const distRest = Math.min(phi, Math.PI * 2 - phi);
      let detentTorque = 0;
      if (distRest < 0.34) {
        const twoPi = Math.PI * 2;
        const nearestRest = Math.round(this.keyAngleRad / twoPi) * twoPi;
        detentTorque = (nearestRest - this.keyAngleRad) * KEY_DETENT_STIFFNESS;
      }
      this.keyAngularVel += (detentTorque - this.keyAngularVel * KEY_DAMPING) * h;
      this.keyAngleRad += this.keyAngularVel * h;
      if (Math.abs(this.keyAngularVel) < 0.002 && Math.abs(detentTorque) < 0.02) {
        this.keyAngularVel = 0;
      }
    }

    const phi = normalizeAngle2Pi(this.keyAngleRad);
    const stumpInGateTunnel =
      this.boltThrow01 > POCKET_PLAY_01 + 0.004 && this.boltThrow01 < 1.0 - POCKET_PLAY_01 - 0.004;

    // 2. Integrate all 5 brass levers under phosphor-bronze leaf spring torque, exact belly cam contact, and manual pick drag
    for (let i = 0; i < LEVER_COUNT; i++) {
      const prevAngle = this.leverAnglesRad[i];
      const springTorque = -this.getLeafSpringRestoringTorque(i);
      let totalTorque = springTorque - this.leverAngVels[i] * LEVER_DAMPING;

      if (this.draggedLeverIndex === i) {
        const dragErr = this.draggedLeverTargetRad - prevAngle;
        totalTorque += dragErr * 4.2 - this.leverAngVels[i] * 0.14;
      }

      let nextVel = this.leverAngVels[i] + (totalTorque / LEVER_INERTIA) * h;
      let nextAngle = prevAngle + nextVel * h;

      // Exact unilateral rigid-body contact with stepped key bitting tooth i
      const camLift = computeKeyCamLiftRad(i, phi);
      if (nextAngle < camLift) {
        nextAngle = camLift;
        if (nextVel < 0) nextVel = 0;
      }

      // If the square steel bolt stump is physically inside the horizontal gate tunnel,
      // the stump mechanically traps each lever within its gate vertical clearance!
      if (stumpInGateTunnel) {
        const minGate = LEVER_ALIGN_RAD[i] - GATE_ALIGN_TOL_RAD * 0.65;
        const maxGate = LEVER_ALIGN_RAD[i] + GATE_ALIGN_TOL_RAD * 0.65;
        if (nextAngle < minGate) {
          nextAngle = minGate;
          if (nextVel < 0) nextVel = 0;
        } else if (nextAngle > maxGate) {
          nextAngle = maxGate;
          if (nextVel > 0) nextVel = 0;
        }
      }

      // Hard physical rest stop on bottom brass rest pin (theta = 0) & stump-in-pocket upper stop
      const physicalMax = computeLeverPhysicalMaxRad(i, this.boltThrow01);
      if (nextAngle <= 0) {
        if (prevAngle > 0.015 && nextVel < -0.5) {
          this.detentPulse = Math.min(1.0, this.detentPulse + Math.min(0.32, Math.abs(nextVel) * 0.08));
        }
        nextAngle = 0;
        nextVel = 0;
      } else if (nextAngle >= physicalMax) {
        nextAngle = physicalMax;
        if (nextVel > 0) nextVel = 0;
      }

      this.leverAnglesRad[i] = nextAngle;
      this.leverAngVels[i] = nextVel;
      this.leverSpringFlex01[i] = nextAngle / LEVER_MAX_RAD;
    }

    // 3. Evaluate gate alignment across all 5 levers
    this.gatesAligned = this.checkAllGatesAligned();

    // 4. Compute deadbolt driving force (from key talon drive bit OR manual bolt drag)
    let boltForce = -this.boltVel * BOLT_FRICTION;
    let talonEngaged = false;
    let talonBlockedByStop = false;

    const inTalonArc = phi >= TALON_ENGAGE_START_RAD - 0.04 && phi <= TALON_ENGAGE_END_RAD + 0.04;
    if (inTalonArc) {
      computeTalonBitTip(phi, this._talonTip);
      const notchX = computeTalonNotchX(this.boltThrow01);
      const talonHalfMouth = 0.046;
      const dxTalon = this._talonTip.x - notchX;

      if (Math.abs(dxTalon) <= 0.145) {
        talonEngaged = true;
        if (this.isDraggingBolt && !this.isDraggingKey && !this.autoTurnActive) {
          const twoPi = Math.PI * 2;
          const turnBase = this.keyAngleRad - phi;
          const desiredNotchX = computeTalonNotchX(this.boltThrow01);
          const sinTarget = Math.max(-0.92, Math.min(0.92, (KEY_X - desiredNotchX) / KEY_TALON_RADIUS));
          const targetPhi = Math.PI - Math.asin(sinTarget);
          const targetKeyRad = turnBase + targetPhi;
          this.keyAngleRad += (targetKeyRad - this.keyAngleRad) * Math.min(1, h * 36);
          this.keyAngularVel = 0;
          if (Math.abs(this.keyAngleRad - turnBase - Math.PI) > twoPi) {
            this.keyAngleRad = targetKeyRad;
          }
        } else {
          const effectiveHalfTravel = BOLT_TRAVEL * 0.475;
          const targetThrow01 = Math.max(
            0,
            Math.min(1, 0.5 + (this._talonTip.x - KEY_X) / (effectiveHalfTravel * 2))
          );
          const tipVelX = -KEY_TALON_RADIUS * Math.cos(phi) * this.keyAngularVel;
          const targetBoltVel =
            targetThrow01 <= 0.0001 || targetThrow01 >= 0.9999
              ? 0
              : tipVelX / (effectiveHalfTravel * 2);
          boltForce +=
            (targetThrow01 - this.boltThrow01) * BOLT_TALON_STIFFNESS +
            (targetBoltVel - this.boltVel) * BOLT_TALON_DAMPING;
        }
      } else if (
        (this.boltThrow01 >= 1.0 - POCKET_PLAY_01 && this._talonTip.x < notchX - talonHalfMouth) ||
        (this.boltThrow01 <= POCKET_PLAY_01 && this._talonTip.x > notchX + talonHalfMouth)
      ) {
        talonBlockedByStop = true;
      }
    }

    if (this.isDraggingBolt) {
      boltForce += (this.boltDragTarget01 - this.boltThrow01) * 260.0;
    }

    // Physical spring-loaded upper detent pawl seating force near unlocked (0) & locked (1) V-notches
    if (!talonEngaged && !this.isDraggingBolt) {
      if (this.boltThrow01 < 0.065) {
        boltForce -= this.boltThrow01 * 55.0;
      } else if (this.boltThrow01 > 0.935) {
        boltForce += (1.0 - this.boltThrow01) * 55.0;
      }
    }

    // 5. Integrate deadbolt linear motion and enforce exact bolt-stump <-> 5-lever collision
    const prevBolt = this.boltThrow01;
    this.boltVel += boltForce * h;
    let nextBolt = prevBolt + this.boltVel * h;

    if (nextBolt < 0) {
      if (prevBolt > 0.02 && this.boltVel < -0.3) {
        this.detentPulse = Math.min(1.0, this.detentPulse + 0.45);
        this.caseRecoilVelX -= 0.035;
      }
      nextBolt = 0;
      if (this.boltVel < 0) this.boltVel = 0;
    } else if (nextBolt > 1) {
      if (prevBolt < 0.98 && this.boltVel > 0.3) {
        this.detentPulse = Math.min(1.0, this.detentPulse + 0.45);
        this.caseRecoilVelX += 0.035;
      }
      nextBolt = 1;
      if (this.boltVel > 0) this.boltVel = 0;
    }

    // BILATERAL BOLT-STUMP BLOCKING BY MISALIGNED LEVERS:
    // If ANY 1 of the 5 levers is not aligned with the gate, the square steel stump
    // stops flush against that lever's solid brass locking cheek (zero interpenetration!).
    if (!this.gatesAligned) {
      if (prevBolt <= POCKET_PLAY_01 + 0.001 && nextBolt > POCKET_PLAY_01) {
        nextBolt = POCKET_PLAY_01;
        if (this.boltVel > 0) {
          this.caseRecoilVelX += Math.min(0.025, this.boltVel * 0.02);
          this.boltVel = 0;
        }
        blockedThisSub = true;
      } else if (prevBolt >= 1.0 - POCKET_PLAY_01 - 0.001 && nextBolt < 1.0 - POCKET_PLAY_01) {
        nextBolt = 1.0 - POCKET_PLAY_01;
        if (this.boltVel < 0) {
          this.caseRecoilVelX -= Math.min(0.025, Math.abs(this.boltVel) * 0.02);
          this.boltVel = 0;
        }
        blockedThisSub = true;
      }
    }

    // Prevent key from clipping through a jammed bolt or wrong-way stop; allow instant smooth retreat!
    if (blockedThisSub && talonEngaged) {
      this.keyAngleRad = prevKeyAngle + (this.keyAngleRad - prevKeyAngle) * 0.08;
      this.keyAngularVel *= 0.25;
      if (this.autoTurnActive) {
        this.autoTurnActive = false;
      }
    } else if (talonBlockedByStop && !this.autoTurnActive) {
      const deltaKey = this.keyAngleRad - prevKeyAngle;
      // Locked (bolt=1): wrong direction is CW (+deltaKey); Unlocked (bolt=0): wrong direction is CCW (-deltaKey)
      const pushingIntoStop =
        (this.boltThrow01 > 0.5 && deltaKey > 0) || (this.boltThrow01 <= 0.5 && deltaKey < 0);
      if (pushingIntoStop) {
        this.keyAngleRad = prevKeyAngle;
        this.keyAngularVel = 0;
        if (this.isDraggingKey) {
          this.keyDragTargetRad = prevKeyAngle;
        }
        this.caseRecoilVelX += (this.boltThrow01 > 0.5 ? 1 : -1) * 0.012;
      }
    }

    // Settle tiny play back cleanly toward 0 or 1 once released inside locking pocket
    if (!talonEngaged && !this.isDraggingBolt && !blockedThisSub) {
      if (nextBolt <= POCKET_PLAY_01 && Math.abs(this.boltVel) < 0.05) {
        nextBolt += (0 - nextBolt) * Math.min(1, h * 14);
      } else if (nextBolt >= 1.0 - POCKET_PLAY_01 && Math.abs(this.boltVel) < 0.05) {
        nextBolt += (1 - nextBolt) * Math.min(1, h * 14);
      }
    }

    this.boltThrow01 = nextBolt;
    this.boltBlocked = blockedThisSub;

    // Update per-lever binding shimmer intensity (only misaligned levers when stump pushes against them)
    let firstBinder = -1;
    for (let i = 0; i < LEVER_COUNT; i++) {
      const mis = Math.abs(this.leverAnglesRad[i] - LEVER_ALIGN_RAD[i]) > GATE_ALIGN_TOL_RAD;
      const targetBind = blockedThisSub && mis ? 1.0 : 0.0;
      if (blockedThisSub && mis && firstBinder === -1) firstBinder = i;
      this.leverBinding01[i] += (targetBind - this.leverBinding01[i]) * Math.min(1, h * 16);
    }
    this.bindingLeverIndex = firstBinder;

    // Subtle damped mechanical case recoil & detent pulse decay
    this.caseRecoilVelX += (-this.caseRecoilX * 380.0 - this.caseRecoilVelX * 26.0) * h;
    this.caseRecoilX = Math.max(-0.012, Math.min(0.012, this.caseRecoilX + this.caseRecoilVelX * h));
    this.detentPulse *= Math.exp(-h * 12.0);

    // Sync coupled horological gear-train & detent-pawl poses strictly with live bolt/key state
    this.syncGearTrainStates();
  }

  getState(): LockDiagnosticsState {
    const leverAnglesDeg: number[] = [];
    for (let i = 0; i < LEVER_COUNT; i++) {
      leverAnglesDeg.push(Number(((this.leverAnglesRad[i] * 180) / Math.PI).toFixed(2)));
    }
    return {
      keyAngleRad: Number(this.keyAngleRad.toFixed(4)),
      boltThrow01: Number(this.boltThrow01.toFixed(4)),
      isLocked: this.isLocked,
      leverAnglesDeg,
      gatesAligned: this.gatesAligned,
      boltBlocked: this.boltBlocked,
      bindingLeverIndex: this.bindingLeverIndex,
      autoTurning: this.autoTurnActive,
      gearAnglesRad: {
        upperPinionU1: Number(this.upperPinionAngleRad.toFixed(4)),
        upperDetentU2: Number(this.upperDetentAngleRad.toFixed(4)),
        upperPawl: Number(this.upperPawlAngleRad.toFixed(4)),
        curtainB1: Number(this.curtainB1AngleRad.toFixed(4)),
        curtainIdlerB2: Number(this.curtainB2AngleRad.toFixed(4)),
        curtainCompoundB3: Number(this.curtainB3AngleRad.toFixed(4)),
        keyHomeWheelB4: Number(this.keyHomeB4AngleRad.toFixed(4)),
        lowerPawl: Number(this.lowerPawlAngleRad.toFixed(4)),
      },
    };
  }
}
