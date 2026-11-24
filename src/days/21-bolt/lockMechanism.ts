/**
 * Day 21 - Bolt: Deterministic 19th-Century Geared 5-Lever Chubb Thumb-Turn Mortise Deadbolt Physics
 *
 * Pure mechanical solver (zero Three.js rendering dependencies, zero per-frame allocations)
 * modeling exact geometric contact between:
 *  - A front quarter-turn brass deadbolt thumb-turn knob (0 deg Vertical = Unlocked <-> 90 deg Horizontal = Locked)
 *  - A load-bearing 3.000:1 coaxial step-up horological gear train (B1 28T -> B2a 14T / B2b 24T -> C1 16T)
 *    driving an internal 5-stepped cam barrel smoothly across 45 deg .. 315 deg (270 deg active sweep, near-zero dead zone!)
 *  - A coupled lower dual-notch detent escapement (B2b -> B3f 16T / B3r 14T -> B4a 20T + blued-steel V-notch cam & pawl)
 *  - 5 stacked independent brass lever tumblers pivoting around (PIVOT_X, PIVOT_Y) with exact concentric Chubb belly arcs
 *  - 5 preloaded staggered phosphor-bronze leaf springs clamped between abutment & reaction pins
 *  - Heavy square blued-steel deadbolt, U-shaped bolt talon, riveted square bolt stump, and upper involute rack-and-pinion
 *    detent escapement (Rack -> U1A 12T / U1B 16T -> U2A 18T + upper dual-notch cam & pawl)
 *  - Exact bilateral bolt-stump <-> lever-gate collision blocking, lever trapping, and manual deadbolt back-driving
 */

export const LEVER_COUNT = 5;

// Master mechanical datum coordinates (lock-local XY plane, 1 unit ~ 10 cm)
export const PIVOT_X = -0.58;
export const PIVOT_Y = -0.02;

export const KEY_X = 0.04;
export const KEY_Y = -0.22;
export const KEY_SHAFT_RADIUS = 0.046;

// Quarter-turn thumb-turn knob & exact 3.000:1 coaxial step-up internal cam barrel contract
// knobAngleRad = 0 (Vertical) -> Unlocked (boltThrow01 = 0), cam = 45 deg (PI/4)
// knobAngleRad = PI/2 (Horizontal) -> Locked (boltThrow01 = 1), cam = 315 deg (7*PI/4)
export const KNOB_MIN_RAD = 0;
export const KNOB_MAX_RAD = Math.PI * 0.5;
export const KNOB_TO_CAM_RATIO = 3.0;
export const CAM_PHASE_OFFSET_RAD = Math.PI * 0.25;

export function knobToCamAngleRad(knobRad: number): number {
  return CAM_PHASE_OFFSET_RAD + KNOB_TO_CAM_RATIO * knobRad;
}

export function camToKnobAngleRad(camRad: number): number {
  return (camRad - CAM_PHASE_OFFSET_RAD) / KNOB_TO_CAM_RATIO;
}

// Deadbolt & stump geometry (exact outer dimensions INCLUDING chamfer bevels)
export const STUMP_Y = 0.16;
export const STUMP_X_UNLOCKED = -0.08;
export const BOLT_TRAVEL = 0.34;
export const STUMP_X_LOCKED = STUMP_X_UNLOCKED + BOLT_TRAVEL; // +0.26
export const STUMP_WIDTH = 0.082;
export const STUMP_HEIGHT = 0.072;

// Lever Z-stack layout (front-to-back parallax, zero interpenetration, generous foreground depth for 2-deck 3:1 gear train + glass!)
export const LEVER_Z_START = -0.048;
export const LEVER_Z_PITCH = 0.041;
export const LEVER_THICKNESS = 0.022;
export const LEVER_BEVEL = 0.002;

export function getLeverZ(index: number): number {
  return LEVER_Z_START + index * LEVER_Z_PITCH;
}

// Bolt tail sits behind Lever 0; bolt stump spans cleanly through all 5 levers
export const BOLT_TAIL_Z_BACK = -0.168;
export const BOLT_TAIL_Z_FRONT = -0.074;
export const BOLT_HEAD_Z_HALF = 0.168;
export const STUMP_Z_BACK = BOLT_TAIL_Z_FRONT;
export const STUMP_Z_FRONT = 0.144;

// Lock case, cutaway glass window aperture & sealed crystal pane bounds
export const CASE_INNER_MIN_X = -0.80;
export const CASE_INNER_MAX_X = 0.88;
export const CASE_INNER_MIN_Y = -0.585;
export const CASE_INNER_MAX_Y = 0.605;
export const FOREND_X = 0.94;
export const FOREND_THICKNESS = 0.065;

// Visible museum-cutaway window opening (100% of every gear, spring & pin fits inside!)
export const CUTAWAY_WIN_MIN_X = -0.755;
export const CUTAWAY_WIN_MAX_X = 0.855;
export const CUTAWAY_WIN_MIN_Y = -0.548;
export const CUTAWAY_WIN_MAX_Y = 0.566;
export const CUTAWAY_FRONT_Z = 0.208;

// Spring abutment post, flush cantilever fulcrum reaction pin & bottom rest stop pin
export const SPRING_POST_X = -0.46;
export const SPRING_POST_Y = 0.415;
export const SPRING_REACTION_PIN_X = -0.374;
export const SPRING_REACTION_PIN_Y = 0.431;
export const SPRING_REACTION_PIN_R = 0.017;

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

// Exact internal cam-lobe radii derived from rigid-body contact at phi = PI (12 o'clock)
export const KEY_BIT_RADII: readonly number[] = LEVER_ALIGN_RAD.map((theta) => {
  const yBellyWorld =
    PIVOT_Y + LEVER_BELLY_LOCAL_X * Math.sin(theta) + LEVER_BELLY_LOCAL_Y * Math.cos(theta);
  return Number((yBellyWorld - KEY_Y).toFixed(5));
});

// Cam-lobe tooth half-width (outer crown width including bevels)
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
export const GATE_POCKET_LEFT_INNER_X = 0.554;
export const GATE_POCKET_RIGHT_INNER_X = 0.788;
export const GATE_POCKET_RIGHT_OUTER_X = 0.924;
export const GATE_POCKET_TOP_Y = 0.232;
export const GATE_POCKET_BOT_R_Y = -0.184;

// Per-lever spring & inertia constants
const LEVER_INERTIA = 0.0014;
const LEVER_SPRING_PRELOAD = 0.052;
const LEVER_SPRING_STIFFNESS = 0.42;
const LEVER_DAMPING = 0.022;

// Quarter-turn knob & bolt inertia constants (weighty ~1.18s mechanical toggle throw)
const KNOB_AUTO_SPEED_RAD_S = 1.48;
const KNOB_DETENT_STIFFNESS = 52.0;
const KNOB_DAMPING = 13.5;
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
  /** Cam center (KEY_X, KEY_Y) mapped into lever-local coordinates when lever is at alignRad */
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
 * Rotating lever `index` by +alignRad brings the milled H-gate dead-level horizontally at world Y = STUMP_Y!
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
 */
function buildLeverBellyPolyline(index: number): readonly Point2D[] {
  const spec = LEVER_GATE_SPECS[index];
  const R = spec.bitRadius;
  const a = spec.alignRad;
  const pts: Point2D[] = [];

  const arcPt = (d: number, rad = R): Point2D => ({
    x: spec.keyCenterLocalX + rad * Math.sin(a - d),
    y: spec.keyCenterLocalY + rad * Math.cos(a - d),
  });

  const throatStart: Point2D = { x: 0.22, y: -0.145 };
  const arcLeft = arcPt(CAM_DWELL_HALF_RAD);
  const leftAng = a - CAM_DWELL_HALF_RAD;
  const leftTanX = -Math.cos(leftAng);
  const leftTanY = Math.sin(leftAng);

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

  const arcSegs = 24;
  for (let s = 0; s <= arcSegs; s++) {
    const d = CAM_DWELL_HALF_RAD - (2 * CAM_DWELL_HALF_RAD * s) / arcSegs;
    pts.push(arcPt(d));
  }

  const arcRight = arcPt(-CAM_DWELL_HALF_RAD);
  const rightAng = a + CAM_DWELL_HALF_RAD;
  const rightTanX = Math.cos(rightAng);
  const rightTanY = -Math.sin(rightAng);

  const heelStartX = Math.max(arcRight.x + 0.068, 0.815);
  const heelStart: Point2D = { x: heelStartX, y: LEVER_HEEL_LOCAL_Y };
  const exitLen = Math.hypot(heelStart.x - arcRight.x, heelStart.y - arcRight.y);
  const e1x = arcRight.x + rightTanX * Math.min(exitLen * 0.24, 0.024);
  const e1y = Math.max(
    LEVER_HEEL_LOCAL_Y + 0.002,
    arcRight.y + rightTanY * Math.min(exitLen * 0.24, 0.024)
  );
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

export function normalizeAngle2Pi(rad: number): number {
  const twoPi = Math.PI * 2;
  const m = rad % twoPi;
  return m < 0 ? m + twoPi : m;
}

/**
 * Exact rigid-body unilateral contact solver: finds the exact minimum lift angle `theta >= 0`
 * for lever `index` so its physical bottom belly polyline sits strictly on or above
 * rotating internal cam lobe `index` at phase angle `phiRad`.
 */
const TOOTH_CROWN_SAMPLES: readonly number[] = [-1.0, -0.5, 0.0, 0.5, 1.0];

export function computeKeyCamLiftRad(index: number, phiRad: number): number {
  const phi = normalizeAngle2Pi(phiRad);
  const R = KEY_BIT_RADII[index];
  const poly = LEVER_BELLY_POLYLINES[index];

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

    if (wy < PIVOT_Y - 0.32) continue;

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

  const distFromTop = Math.abs(phi - Math.PI);
  if (distFromTop <= CAM_DWELL_HALF_RAD - halfAng && Math.abs(maxTheta - LEVER_ALIGN_RAD[index]) < 0.004) {
    return LEVER_ALIGN_RAD[index];
  }

  return Math.min(LEVER_MAX_RAD, maxTheta);
}

/**
 * Evaluates world-space (x, y) tip of the internal cam barrel's rear bolt-talon drive arm at angle `phiRad`.
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
 * Computes exact physical upper angle limit for lever `index` at current `boltThrow01`
 * so manually lifting a lever can never drive its bottom H-gate pocket floor into the steel bolt stump.
 */
export function computeLeverPhysicalMaxRad(index: number, boltThrow01: number): number {
  if (boltThrow01 > 0.5) {
    const dx = STUMP_X_UNLOCKED + boltThrow01 * BOLT_TRAVEL - PIVOT_X;
    const dy = STUMP_Y - PIVOT_Y;
    const hw = STUMP_WIDTH * 0.5;
    const hh = STUMP_HEIGHT * 0.5;
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
  knobAngleRad: number;
  knobAngleDeg: number;
  camAngleRad: number;
  camAngleDeg: number;
  /** Alias for camAngleRad for backwards-compatible telemetry inspection */
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
    knobDriveB1: number;
    stepUpCountershaftB2: number;
    camDrivenC1: number;
    transferArborB3: number;
    lowerDetentWheelB4: number;
    lowerPawl: number;
  };
}

// =============================================================================
// Horological 3:1 Coaxial Geared Thumb-Turn Transmission & Dual Detent Escapements
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
  const pitchRadius = module * teeth * 0.5;
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
// Carefully proportioned so 100% of every gear tooth, pawl, spring pin & leaf spring sits inside GLASS_WIN!
export const UPPER_RACK_MODULE = 0.0138;
export const UPPER_RACK_PITCH = Math.PI * UPPER_RACK_MODULE;
export const UPPER_RACK_SPEC: RackSpec = {
  module: UPPER_RACK_MODULE,
  pitch: UPPER_RACK_PITCH,
  toothCount: 12,
  firstToothLocalX: 0.010,
  pitchLineY: 0.348,
  tipY: 0.348 + GEAR_ADDENDUM_FACTOR * UPPER_RACK_MODULE,
  rootY: 0.348 - GEAR_DEDENDUM_FACTOR * UPPER_RACK_MODULE,
  barBaseY: 0.240,
  barMinLocalX: -0.012,
  barMaxLocalX: 0.514,
  zBack: -0.156,
  zFront: -0.116,
};

const U1_TEETH_RACK = 12;
const U1_RACK_PITCH_R = UPPER_RACK_MODULE * U1_TEETH_RACK * 0.5; // 0.0828
export const ARBOR_U1_X = 0.422;
export const ARBOR_U1_Y = UPPER_RACK_SPEC.pitchLineY + U1_RACK_PITCH_R; // 0.4308

export const UPPER_STAGE_MODULE = 0.0110;
const U1B_TEETH = 16;
const U2A_TEETH = 18;
const U12_DIST = UPPER_STAGE_MODULE * (U1B_TEETH + U2A_TEETH) * 0.5; // 0.1870
export const UPPER_U12_ANGLE_RAD = 0.05;
export const ARBOR_U2_X = ARBOR_U1_X + U12_DIST * Math.cos(UPPER_U12_ANGLE_RAD);
export const ARBOR_U2_Y = ARBOR_U1_Y + U12_DIST * Math.sin(UPPER_U12_ANGLE_RAD);

export const GEAR_U1A_PINION = makeSpurGearSpec(
  'RackThrowPinion_U1A',
  ARBOR_U1_X,
  ARBOR_U1_Y,
  UPPER_RACK_MODULE,
  U1_TEETH_RACK,
  -0.156,
  -0.116
);

export const GEAR_U1B_SPUR = makeSpurGearSpec(
  'CompoundThrowSpur_U1B',
  ARBOR_U1_X,
  ARBOR_U1_Y,
  UPPER_STAGE_MODULE,
  U1B_TEETH,
  -0.108,
  -0.072
);

export const GEAR_U2A_DRIVEN = makeSpurGearSpec(
  'DetentDrivenWheel_U2A',
  ARBOR_U2_X,
  ARBOR_U2_Y,
  UPPER_STAGE_MODULE,
  U2A_TEETH,
  -0.108,
  -0.072
);

// --- TRAIN B (Load-Bearing 3.000:1 Coaxial Thumb-Turn Transmission + Lower Dual-Notch Detent) ---
// Stage 1 (Front Deck 2, z: 0.162..0.178): B1 (28T Knob Drive) -> B2a (14T Countershaft Pinion)  [2.000:1]
// Stage 2 (Front Deck 1, z: 0.138..0.155): B2b (24T Step Wheel) -> C1 (16T Inner Cam Pinion)      [1.500:1]
//   -> Exact Coaxial Speed-Up Ratio: (28/14) * (24/16) = 3.000000:1 SAME DIRECTION (90 deg knob -> 270 deg cam)!
// Detent Tap (Deck 1 -> Mid-Rear): B2b (24T) -> B3f (16T) / B3r (14T) -> B4a (20T Detent Wheel)  [189 deg sweep]
export const STAGE1_MODULE = 0.0106;
const B1_TEETH = 28;
const B2A_TEETH = 14;
const B12_DIST = STAGE1_MODULE * (B1_TEETH + B2A_TEETH) * 0.5; // 0.2226

const B2B_TEETH = 24;
const C1_TEETH = 16;
export const STAGE2_MODULE = (2 * B12_DIST) / (B2B_TEETH + C1_TEETH); // 0.01113 (EXACT shared center distance!)

export const LOWER_B12_ANGLE_RAD = -0.76;
export const ARBOR_B2_X = KEY_X + B12_DIST * Math.cos(LOWER_B12_ANGLE_RAD);
export const ARBOR_B2_Y = KEY_Y + B12_DIST * Math.sin(LOWER_B12_ANGLE_RAD);

const B3F_TEETH = 16;
const B23_DIST = STAGE2_MODULE * (B2B_TEETH + B3F_TEETH) * 0.5; // 0.2226
export const LOWER_B23_ANGLE_RAD = -0.22;
export const ARBOR_B3_X = ARBOR_B2_X + B23_DIST * Math.cos(LOWER_B23_ANGLE_RAD);
export const ARBOR_B3_Y = ARBOR_B2_Y + B23_DIST * Math.sin(LOWER_B23_ANGLE_RAD);

export const STAGE3_MODULE = 0.0118;
const B3R_TEETH = 14;
const B4A_TEETH = 20;
const B34_DIST = STAGE3_MODULE * (B3R_TEETH + B4A_TEETH) * 0.5; // 0.2006
export const LOWER_B34_ANGLE_RAD = 0.44;
export const ARBOR_B4_X = ARBOR_B3_X + B34_DIST * Math.cos(LOWER_B34_ANGLE_RAD);
export const ARBOR_B4_Y = ARBOR_B3_Y + B34_DIST * Math.sin(LOWER_B34_ANGLE_RAD);

export const GEAR_B1_KNOB_DRIVE = makeSpurGearSpec(
  'ThumbTurnDriveWheel_B1',
  KEY_X,
  KEY_Y,
  STAGE1_MODULE,
  B1_TEETH,
  0.162,
  0.178
);

export const GEAR_B2A_PINION = makeSpurGearSpec(
  'StepUpInputPinion_B2A',
  ARBOR_B2_X,
  ARBOR_B2_Y,
  STAGE1_MODULE,
  B2A_TEETH,
  0.162,
  0.178
);

export const GEAR_B2B_STEP = makeSpurGearSpec(
  'StepUpOutputWheel_B2B',
  ARBOR_B2_X,
  ARBOR_B2_Y,
  STAGE2_MODULE,
  B2B_TEETH,
  0.138,
  0.155
);

export const GEAR_C1_CAM_DRIVEN = makeSpurGearSpec(
  'CamBarrelDrivenPinion_C1',
  KEY_X,
  KEY_Y,
  STAGE2_MODULE,
  C1_TEETH,
  0.138,
  0.155
);

export const GEAR_B3F_TRANSFER = makeSpurGearSpec(
  'DetentTransferGear_B3F',
  ARBOR_B3_X,
  ARBOR_B3_Y,
  STAGE2_MODULE,
  B3F_TEETH,
  0.138,
  0.155
);

export const GEAR_B3R_PINION = makeSpurGearSpec(
  'CompoundDetentPinion_B3R',
  ARBOR_B3_X,
  ARBOR_B3_Y,
  STAGE3_MODULE,
  B3R_TEETH,
  0.048,
  0.096
);

export const GEAR_B4A_DETENT = makeSpurGearSpec(
  'SpokedLowerDetentWheel_B4A',
  ARBOR_B4_X,
  ARBOR_B4_Y,
  STAGE3_MODULE,
  B4A_TEETH,
  0.048,
  0.096
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

export interface LowerTransmissionAngles {
  b1: number;
  b2: number;
  c1: number;
  b3: number;
  b4: number;
}

/**
 * Computes exact Three.js right-handed Z rotation angles for the entire 3:1 coaxial thumb-turn
 * gear transmission & lower detent wheel from `knobAngleRad`.
 */
export function computeLowerTrainAnglesRad(knobAngleRad: number): LowerTransmissionAngles {
  // Knob turns CW (+knobAngleRad) -> right-handed Three.js rotation.z = -knobAngleRad
  const b1 = -knobAngleRad;
  const b2 = computeMeshedGearAngle(
    b1,
    GEAR_B1_KNOB_DRIVE.teeth,
    GEAR_B2A_PINION.teeth,
    LOWER_B12_ANGLE_RAD
  );
  // B2b drives coaxial inner cam pinion C1 back along (LOWER_B12_ANGLE_RAD + PI) at exact 3.000:1 CW!
  const c1 = computeMeshedGearAngle(
    b2,
    GEAR_B2B_STEP.teeth,
    GEAR_C1_CAM_DRIVEN.teeth,
    LOWER_B12_ANGLE_RAD + Math.PI
  );
  // B2b simultaneously drives detent transfer arbor B3 -> grand lower detent wheel B4
  const b3 = computeMeshedGearAngle(
    b2,
    GEAR_B2B_STEP.teeth,
    GEAR_B3F_TRANSFER.teeth,
    LOWER_B23_ANGLE_RAD
  );
  const b4 = computeMeshedGearAngle(
    b3,
    GEAR_B3R_PINION.teeth,
    GEAR_B4A_DETENT.teeth,
    LOWER_B34_ANGLE_RAD
  );
  return { b1, b2, c1, b3, b4 };
}

function wrapAnglePi(a: number): number {
  let x = a % (Math.PI * 2);
  if (x > Math.PI) x -= Math.PI * 2;
  if (x < -Math.PI) x += Math.PI * 2;
  return x;
}

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

// Compute exact V-notch local angles so BOTH pawls drop DEAD-FLUSH at Unlocked (0) and Locked (1)!
const UPPER_PAWL_PIVOT_X = 0.766;
const UPPER_PAWL_PIVOT_Y = 0.394;
const UPPER_PAWL_ARM_LEN = 0.122;
const UPPER_CAM_OUTER_R = 0.088;
const UPPER_CAM_ROOT_R = 0.068;
const PAWL_BEAK_R = 0.0105;

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
  rippleAmp: 0.0020,
  pivotX: UPPER_PAWL_PIVOT_X,
  pivotY: UPPER_PAWL_PIVOT_Y,
  pawlArmLength: UPPER_PAWL_ARM_LEN,
  beakRadius: PAWL_BEAK_R,
  pawlSign: 1,
  springPinX: 0.796,
  springPinY: 0.506,
  zBack: 0.020,
  zFront: 0.086,
};

const LOWER_PAWL_PIVOT_X = 0.758;
const LOWER_PAWL_PIVOT_Y = -0.382;
const LOWER_PAWL_ARM_LEN = 0.122;
const LOWER_CAM_OUTER_R = 0.088;
const LOWER_CAM_ROOT_R = 0.068;

const LOWER_SEATED_WORLD_ANG = computePawlSeatedContactWorldAngle(
  ARBOR_B4_X,
  ARBOR_B4_Y,
  LOWER_PAWL_PIVOT_X,
  LOWER_PAWL_PIVOT_Y,
  LOWER_PAWL_ARM_LEN,
  LOWER_CAM_ROOT_R + PAWL_BEAK_R,
  1
);

export const LOWER_DETENT_CAM_SPEC: DetentCamSpec = {
  cx: ARBOR_B4_X,
  cy: ARBOR_B4_Y,
  outerRadius: LOWER_CAM_OUTER_R,
  rootRadius: LOWER_CAM_ROOT_R,
  notchHalfWidthRad: 0.26,
  // Two opposing V-notches separated by exact 189 deg: one for Vertical Unlocked (0), one for Horizontal Locked (PI/2)!
  notchLocalAngles: [
    wrapAnglePi(LOWER_SEATED_WORLD_ANG - computeLowerTrainAnglesRad(KNOB_MIN_RAD).b4),
    wrapAnglePi(LOWER_SEATED_WORLD_ANG - computeLowerTrainAnglesRad(KNOB_MAX_RAD).b4),
  ],
  rippleCount: 16,
  rippleAmp: 0.0018,
  pivotX: LOWER_PAWL_PIVOT_X,
  pivotY: LOWER_PAWL_PIVOT_Y,
  pawlArmLength: LOWER_PAWL_ARM_LEN,
  beakRadius: PAWL_BEAK_R,
  pawlSign: 1,
  springPinX: 0.788,
  springPinY: -0.268,
  zBack: 0.106,
  zFront: 0.150,
};

/**
 * Evaluates exact physical outer radius of a detent cam at wheel-local angle `localAngleRad`.
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
    const s = u * u * (3 - 2 * u);
    return spec.rootRadius + (spec.outerRadius - spec.rootRadius) * s;
  }
  if (spec.rippleCount > 0 && spec.rippleAmp > 0) {
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
    Math.min(
      1,
      (d * d + spec.pawlArmLength * spec.pawlArmLength - rc * rc) / (2 * d * spec.pawlArmLength)
    )
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

    const aRootRight = centerA - rootFlankHalfAng;
    const prevValleyMid = centerA - step * 0.5;
    pts.push({ x: Math.cos(prevValleyMid) * rf, y: Math.sin(prevValleyMid) * rf });
    if (rf < rb - 1e-6) {
      pts.push({ x: Math.cos(aRootRight) * rf, y: Math.sin(aRootRight) * rf });
    }

    for (let s = 0; s <= flankSegs; s++) {
      const t = tStart + (tMax - tStart) * (s / flankSegs);
      const r = rb * Math.sqrt(1 + t * t);
      const phi = t - Math.atan(t);
      const a = centerA - (psiBase - phi);
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }

    for (let s = flankSegs; s >= 0; s--) {
      const t = tStart + (tMax - tStart) * (s / flankSegs);
      const r = rb * Math.sqrt(1 + t * t);
      const phi = t - Math.atan(t);
      const a = centerA + (psiBase - phi);
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }

    if (rf < rb - 1e-6) {
      const aRootLeft = centerA + rootFlankHalfAng;
      pts.push({ x: Math.cos(aRootLeft) * rf, y: Math.sin(aRootLeft) * rf });
    }
  }
  return pts;
}

/**
 * Generates exact CCW 2D involute linear rack outer contour in bolt-local XY (teeth pointing +Y).
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
  // Primary physical states (boots cleanly LOCKED: Horizontal 90 deg knob, massive blued deadbolt extended!)
  knobAngleRad = KNOB_MAX_RAD;
  knobAngularVel = 0;
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
  knobDriveB1AngleRad = 0;
  stepUpB2AngleRad = 0;
  camDrivenC1AngleRad = 0;
  transferB3AngleRad = 0;
  lowerDetentB4AngleRad = 0;
  lowerPawlAngleRad = 0;
  lowerPawlLift01 = 0;

  // Interactive control states
  private isDraggingKnob = false;
  private knobDragTargetRad = KNOB_MAX_RAD;

  private draggedLeverIndex = -1;
  private draggedLeverTargetRad = 0;

  private isDraggingBolt = false;
  private boltDragTarget01 = 1.0;

  // Motorized quarter-turn click/Space throw state
  private autoTurnActive = false;
  private autoTurnTargetRad = KNOB_MAX_RAD;

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
    camRadiusAtContact: LOWER_DETENT_CAM_SPEC.rootRadius,
    lift01: 0,
  };

  constructor(initiallyLocked = true) {
    this.boltThrow01 = initiallyLocked ? 1.0 : 0.0;
    this.boltDragTarget01 = this.boltThrow01;
    this.knobAngleRad = initiallyLocked ? KNOB_MAX_RAD : KNOB_MIN_RAD;
    this.knobDragTargetRad = this.knobAngleRad;
    this.autoTurnTargetRad = this.knobAngleRad;
    this.refreshAlignmentFlags();
    this.syncGearTrainStates();
  }

  /** Internal 5-stepped cam barrel angle driven 3:1 by the front thumb-turn knob (45 deg .. 315 deg). */
  get camAngleRad(): number {
    return knobToCamAngleRad(this.knobAngleRad);
  }

  set camAngleRad(rad: number) {
    this.knobAngleRad = Math.max(KNOB_MIN_RAD, Math.min(KNOB_MAX_RAD, camToKnobAngleRad(rad)));
  }

  /** Synchronizes all coupled gear-train angles & spring-loaded detent-pawl contact poses (zero allocations). */
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

    this.knobDriveB1AngleRad = -this.knobAngleRad;
    this.stepUpB2AngleRad = computeMeshedGearAngle(
      this.knobDriveB1AngleRad,
      GEAR_B1_KNOB_DRIVE.teeth,
      GEAR_B2A_PINION.teeth,
      LOWER_B12_ANGLE_RAD
    );
    this.camDrivenC1AngleRad = computeMeshedGearAngle(
      this.stepUpB2AngleRad,
      GEAR_B2B_STEP.teeth,
      GEAR_C1_CAM_DRIVEN.teeth,
      LOWER_B12_ANGLE_RAD + Math.PI
    );
    this.transferB3AngleRad = computeMeshedGearAngle(
      this.stepUpB2AngleRad,
      GEAR_B2B_STEP.teeth,
      GEAR_B3F_TRANSFER.teeth,
      LOWER_B23_ANGLE_RAD
    );
    this.lowerDetentB4AngleRad = computeMeshedGearAngle(
      this.transferB3AngleRad,
      GEAR_B3R_PINION.teeth,
      GEAR_B4A_DETENT.teeth,
      LOWER_B34_ANGLE_RAD
    );
    solveDetentPawlContact(LOWER_DETENT_CAM_SPEC, this.lowerDetentB4AngleRad, this._lowerPawlSol);
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
   * Starts a smooth motorized 90-degree deadbolt thumb-turn throw:
   *  - When unlocked (0 deg Vertical): turns CW to +90 deg Horizontal -> extends/locks deadbolt!
   *  - When locked (+90 deg Horizontal): turns CCW to 0 deg Vertical -> retracts/unlocks deadbolt!
   */
  triggerToggleTurn(): void {
    this.autoTurnTargetRad = this.isLocked ? KNOB_MIN_RAD : KNOB_MAX_RAD;
    this.autoTurnActive = true;
    this.isDraggingKnob = false;
    this.draggedLeverIndex = -1;
    this.isDraggingBolt = false;
  }

  beginKnobDrag(): void {
    this.isDraggingKnob = true;
    this.autoTurnActive = false;
    this.knobDragTargetRad = this.knobAngleRad;
  }

  applyKnobDragDelta(deltaRad: number): void {
    if (!this.isDraggingKnob) {
      this.beginKnobDrag();
    }
    // Allow tiny tactile overswing buffer (+/- 0.08 rad) so pushing against 0/90 stop registers crisp recoil
    this.knobDragTargetRad = Math.max(
      KNOB_MIN_RAD - 0.08,
      Math.min(KNOB_MAX_RAD + 0.08, this.knobDragTargetRad + deltaRad)
    );
  }

  /**
   * Releases manual knob drag: snaps positively into Vertical (0) or Horizontal (90) if within the
   * detent V-notch shoulder zone, or holds cleanly mid-throw for live cutaway gate & spring inspection!
   */
  endKnobDrag(): void {
    this.isDraggingKnob = false;
    if (this.knobAngleRad < 0.22) {
      this.knobDragTargetRad = KNOB_MIN_RAD;
    } else if (this.knobAngleRad > KNOB_MAX_RAD - 0.22) {
      this.knobDragTargetRad = KNOB_MAX_RAD;
    } else {
      this.knobDragTargetRad = this.knobAngleRad;
    }
  }

  nudgeKnobAngle(deltaRad: number): void {
    this.autoTurnActive = false;
    this.isDraggingKnob = false;
    this.knobAngularVel += deltaRad * 5.2;
  }

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

  setLeverAngleImmediate(index: number, rad: number): void {
    if (index < 0 || index >= LEVER_COUNT) return;
    const maxLift = computeLeverPhysicalMaxRad(index, this.boltThrow01);
    this.leverAnglesRad[index] = Math.max(0, Math.min(maxLift, rad));
    this.leverAngVels[index] = 0;
    this.leverSpringFlex01[index] = this.leverAnglesRad[index] / LEVER_MAX_RAD;
    this.refreshAlignmentFlags();
  }

  checkAllGatesAligned(): boolean {
    for (let i = 0; i < LEVER_COUNT; i++) {
      if (Math.abs(this.leverAnglesRad[i] - LEVER_ALIGN_RAD[i]) > GATE_ALIGN_TOL_RAD) {
        return false;
      }
    }
    return true;
  }

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

  step(dtSec: number): void {
    const clampedDt = Math.min(0.05, Math.max(0.0005, dtSec));
    const substeps = Math.max(2, Math.ceil(clampedDt / (1 / 240)));
    const h = clampedDt / substeps;

    for (let s = 0; s < substeps; s++) {
      this.stepSub(h);
    }
  }

  private stepSub(h: number): void {
    const prevKnobAngle = this.knobAngleRad;
    let blockedThisSub = false;

    // 1. Quarter-turn brass thumb-turn knob actuation (0 deg Vertical Unlocked <-> 90 deg Horizontal Locked)
    if (this.autoTurnActive) {
      const rem = this.autoTurnTargetRad - this.knobAngleRad;
      const dist = Math.abs(rem);
      if (dist <= 0.0025) {
        this.knobAngleRad = this.autoTurnTargetRad;
        this.knobAngularVel = 0;
        this.autoTurnActive = false;
      } else {
        const phiCur = knobToCamAngleRad(this.knobAngleRad);
        const inThrow = phiCur > TALON_ENGAGE_START_RAD && phiCur < TALON_ENGAGE_END_RAD;
        const speedProfile = inThrow ? 0.74 : 1.0;
        const maxStep = KNOB_AUTO_SPEED_RAD_S * speedProfile;
        const desiredVel = Math.sign(rem) * Math.min(maxStep, Math.max(0.32, dist * 6.8));
        this.knobAngularVel += (desiredVel - this.knobAngularVel) * Math.min(1, h * 18);
        this.knobAngleRad += this.knobAngularVel * h;
      }
    } else if (this.isDraggingKnob) {
      const err = this.knobDragTargetRad - this.knobAngleRad;
      const phiCur = knobToCamAngleRad(this.knobAngleRad);
      const liftLoad = Math.abs(phiCur - Math.PI) < CAM_DWELL_HALF_RAD + 0.95 ? 0.80 : 1.0;
      const stiffness = 175.0 * liftLoad;
      const accel = err * stiffness - this.knobAngularVel * 22.0;
      this.knobAngularVel += accel * h;
      this.knobAngleRad += this.knobAngularVel * h;
    } else {
      // Positive spring-loaded V-notch detent pull near Vertical Unlocked (0) and Horizontal Locked (PI/2)
      let detentTorque = 0;
      if (this.knobAngleRad < 0.24) {
        detentTorque = (KNOB_MIN_RAD - this.knobAngleRad) * KNOB_DETENT_STIFFNESS;
      } else if (this.knobAngleRad > KNOB_MAX_RAD - 0.24) {
        detentTorque = (KNOB_MAX_RAD - this.knobAngleRad) * KNOB_DETENT_STIFFNESS;
      }
      this.knobAngularVel += (detentTorque - this.knobAngularVel * KNOB_DAMPING) * h;
      this.knobAngleRad += this.knobAngularVel * h;
      if (Math.abs(this.knobAngularVel) < 0.0015 && Math.abs(detentTorque) < 0.02) {
        this.knobAngularVel = 0;
      }
    }

    // Crisp quarter-turn mechanical travel stops at 0 deg (Vertical) & 90 deg (Horizontal)
    if (this.knobAngleRad < KNOB_MIN_RAD) {
      if (prevKnobAngle > KNOB_MIN_RAD + 0.01 && this.knobAngularVel < -0.25) {
        this.detentPulse = Math.min(1.0, this.detentPulse + 0.35);
        this.caseRecoilVelX -= 0.022;
      } else if (this.isDraggingKnob && this.knobDragTargetRad < KNOB_MIN_RAD - 0.015) {
        this.caseRecoilVelX -= 0.008;
        this.knobDragTargetRad = KNOB_MIN_RAD;
      }
      this.knobAngleRad = KNOB_MIN_RAD;
      if (this.knobAngularVel < 0) this.knobAngularVel = 0;
    } else if (this.knobAngleRad > KNOB_MAX_RAD) {
      if (prevKnobAngle < KNOB_MAX_RAD - 0.01 && this.knobAngularVel > 0.25) {
        this.detentPulse = Math.min(1.0, this.detentPulse + 0.35);
        this.caseRecoilVelX += 0.022;
      } else if (this.isDraggingKnob && this.knobDragTargetRad > KNOB_MAX_RAD + 0.015) {
        this.caseRecoilVelX += 0.008;
        this.knobDragTargetRad = KNOB_MAX_RAD;
      }
      this.knobAngleRad = KNOB_MAX_RAD;
      if (this.knobAngularVel > 0) this.knobAngularVel = 0;
    }

    const phi = normalizeAngle2Pi(knobToCamAngleRad(this.knobAngleRad));
    const camAngularVel = KNOB_TO_CAM_RATIO * this.knobAngularVel;
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

      // Exact unilateral rigid-body contact with stepped internal cam lobe i
      const camLift = computeKeyCamLiftRad(i, phi);
      if (nextAngle < camLift) {
        nextAngle = camLift;
        if (nextVel < 0) nextVel = 0;
      }

      // Square steel bolt stump mechanically traps each lever inside its gate tunnel while traversing
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

    // 4. Compute deadbolt driving force (from internal cam talon arm OR manual bolt drag)
    let boltForce = -this.boltVel * BOLT_FRICTION;
    let talonEngaged = false;

    const inTalonArc = phi >= TALON_ENGAGE_START_RAD - 0.04 && phi <= TALON_ENGAGE_END_RAD + 0.04;
    if (inTalonArc) {
      computeTalonBitTip(phi, this._talonTip);
      const notchX = computeTalonNotchX(this.boltThrow01);
      const dxTalon = this._talonTip.x - notchX;

      if (Math.abs(dxTalon) <= 0.145) {
        talonEngaged = true;
        if (this.isDraggingBolt && !this.isDraggingKnob && !this.autoTurnActive) {
          // Back-drive the internal cam barrel, 3:1 gear train & front thumb-turn knob from manual bolt motion!
          const desiredNotchX = computeTalonNotchX(this.boltThrow01);
          const sinTarget = Math.max(-0.92, Math.min(0.92, (KEY_X - desiredNotchX) / KEY_TALON_RADIUS));
          const targetPhi = Math.PI - Math.asin(sinTarget);
          const targetKnobRad = Math.max(KNOB_MIN_RAD, Math.min(KNOB_MAX_RAD, camToKnobAngleRad(targetPhi)));
          this.knobAngleRad += (targetKnobRad - this.knobAngleRad) * Math.min(1, h * 36);
          this.knobAngularVel = 0;
        } else {
          const effectiveHalfTravel = BOLT_TRAVEL * 0.475;
          const targetThrow01 = Math.max(
            0,
            Math.min(1, 0.5 + (this._talonTip.x - KEY_X) / (effectiveHalfTravel * 2))
          );
          const tipVelX = -KEY_TALON_RADIUS * Math.cos(phi) * camAngularVel;
          const targetBoltVel =
            targetThrow01 <= 0.0001 || targetThrow01 >= 0.9999
              ? 0
              : tipVelX / (effectiveHalfTravel * 2);
          boltForce +=
            (targetThrow01 - this.boltThrow01) * BOLT_TALON_STIFFNESS +
            (targetBoltVel - this.boltVel) * BOLT_TALON_DAMPING;
        }
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

    // BILATERAL BOLT-STUMP BLOCKING BY MISALIGNED LEVERS
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

    // Prevent internal cam from clipping through a jammed bolt if someone manually misaligned a lever mid-turn
    if (blockedThisSub && talonEngaged) {
      this.knobAngleRad = prevKnobAngle + (this.knobAngleRad - prevKnobAngle) * 0.08;
      this.knobAngularVel *= 0.25;
      if (this.autoTurnActive) {
        this.autoTurnActive = false;
      }
    }

    // Settle tiny pocket play cleanly back to 0 or 1 once released
    if (!talonEngaged && !this.isDraggingBolt && !blockedThisSub) {
      if (nextBolt <= POCKET_PLAY_01 && Math.abs(this.boltVel) < 0.05) {
        nextBolt += (0 - nextBolt) * Math.min(1, h * 14);
      } else if (nextBolt >= 1.0 - POCKET_PLAY_01 && Math.abs(this.boltVel) < 0.05) {
        nextBolt += (1 - nextBolt) * Math.min(1, h * 14);
      }
    }

    this.boltThrow01 = nextBolt;
    this.boltBlocked = blockedThisSub;

    // Update per-lever binding shimmer intensity
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

    // Sync coupled horological gear-train & detent-pawl poses strictly with live bolt/knob state
    this.syncGearTrainStates();
  }

  getState(): LockDiagnosticsState {
    const leverAnglesDeg: number[] = [];
    for (let i = 0; i < LEVER_COUNT; i++) {
      leverAnglesDeg.push(Number(((this.leverAnglesRad[i] * 180) / Math.PI).toFixed(2)));
    }
    const camRad = this.camAngleRad;
    return {
      knobAngleRad: Number(this.knobAngleRad.toFixed(4)),
      knobAngleDeg: Number(((this.knobAngleRad * 180) / Math.PI).toFixed(2)),
      camAngleRad: Number(camRad.toFixed(4)),
      camAngleDeg: Number(((camRad * 180) / Math.PI).toFixed(2)),
      keyAngleRad: Number(camRad.toFixed(4)),
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
        knobDriveB1: Number(this.knobDriveB1AngleRad.toFixed(4)),
        stepUpCountershaftB2: Number(this.stepUpB2AngleRad.toFixed(4)),
        camDrivenC1: Number(this.camDrivenC1AngleRad.toFixed(4)),
        transferArborB3: Number(this.transferB3AngleRad.toFixed(4)),
        lowerDetentWheelB4: Number(this.lowerDetentB4AngleRad.toFixed(4)),
        lowerPawl: Number(this.lowerPawlAngleRad.toFixed(4)),
      },
    };
  }
}
