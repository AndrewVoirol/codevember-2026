/**
 * Day 21 - Bolt: Deterministic 19th-Century Chubb-Style 5-Lever Mortise Deadbolt Physics
 *
 * Pure mechanical solver (zero Three.js rendering dependencies, zero per-frame allocations)
 * modeling true geometric contact between:
 *  - 5 stacked independent brass lever tumblers pivoting around (PIVOT_X, PIVOT_Y)
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

// Deadbolt & stump geometry
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
export const REST_PIN_Y = -0.215;
export const GUIDE_PIN_X = -0.34;
export const GUIDE_PIN_Y = STUMP_Y;

// Lever belly datum (in lever-local coords at theta = 0, directly above KEY_X)
export const LEVER_BELLY_LOCAL_X = KEY_X - PIVOT_X; // +0.62
export const LEVER_BELLY_LOCAL_Y = -0.115;

// Unique Chubb 5-lever gate alignment angles (classic high-low-low-high-mid bitting [2, 5, 1, 4, 3])
export const LEVER_ALIGN_DEG: readonly number[] = [8.5, 16.5, 6.0, 14.0, 11.2];
export const LEVER_ALIGN_RAD: readonly number[] = LEVER_ALIGN_DEG.map((d) => (d * Math.PI) / 180);
export const LEVER_MAX_RAD = (22.0 * Math.PI) / 180;

// Exact key bitting step radii derived from rigid-body contact at phi = PI (12 o'clock)
export const KEY_BIT_RADII: readonly number[] = LEVER_ALIGN_RAD.map((theta) => {
  const yBellyWorld =
    PIVOT_Y + LEVER_BELLY_LOCAL_X * Math.sin(theta) + LEVER_BELLY_LOCAL_Y * Math.cos(theta);
  return Number((yBellyWorld - KEY_Y).toFixed(5));
});

// Rear bolt-talon drive bit radius and exact talon engagement half-angle
export const KEY_TALON_RADIUS = 0.305;
export const TALON_HALF_ANGLE_RAD = Math.asin((BOLT_TRAVEL * 0.5) / KEY_TALON_RADIUS); // ~0.5912 rad (33.87 deg)
export const TALON_ENGAGE_START_RAD = Math.PI - TALON_HALF_ANGLE_RAD; // ~146.13 deg
export const TALON_ENGAGE_END_RAD = Math.PI + TALON_HALF_ANGLE_RAD; // ~213.87 deg

// Cam lift ramp margins outside the concentric Chubb belly dwell arc
export const CAM_DWELL_HALF_RAD = TALON_HALF_ANGLE_RAD + 0.045; // slightly wider than talon stroke so gates align BEFORE talon pushes
export const CAM_RAMP_SPAN_RAD = 1.02; // ~58.4 deg smooth cam ramp on entry and exit

// Gate clearance & locking pocket play
export const GATE_HALF_HEIGHT = 0.043; // stump half-height = 0.036 -> 0.007 precision clearance per side
export const GATE_ALIGN_TOL_RAD = (1.65 * Math.PI) / 180; // ~0.0288 rad exact mechanical tolerance
export const POCKET_PLAY_01 = 0.048; // realistic tactile mechanical play inside locking pocket before stump hits brass cheek

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

export interface LeverGateGeometry {
  index: number;
  alignRad: number;
  alignDeg: number;
  bitRadius: number;
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

  return {
    index,
    alignRad,
    alignDeg: LEVER_ALIGN_DEG[index],
    bitRadius: KEY_BIT_RADII[index],
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
 * Wraps any angle in radians into [0, 2*PI).
 */
export function normalizeAngle2Pi(rad: number): number {
  const twoPi = Math.PI * 2;
  const m = rad % twoPi;
  return m < 0 ? m + twoPi : m;
}

/**
 * Computes the exact physical contact lift angle (radians) imparted onto lever `index`
 * by key bitting step `index` at key phase angle `phiRad` (0 = 6 o'clock down, PI = 12 o'clock up).
 *
 * Uses exact circular-arc Chubb belly dwell across [PI - CAM_DWELL_HALF_RAD, PI + CAM_DWELL_HALF_RAD]
 * and smooth C1 cam lift ramps whose contact onset scales naturally with each step's physical radius.
 */
export function computeKeyCamLiftRad(index: number, phiRad: number): number {
  const phi = normalizeAngle2Pi(phiRad);
  const distFromTop = Math.abs(phi - Math.PI);
  const alignRad = LEVER_ALIGN_RAD[index];

  // Concentric Chubb belly arc: holds lever at exact gate-alignment height throughout talon stroke
  if (distFromTop <= CAM_DWELL_HALF_RAD) {
    return alignRad;
  }

  // Taller bitting teeth make contact earlier on the curved entry shoulder than shorter teeth
  const radiusFactor = 0.84 + 0.34 * ((KEY_BIT_RADII[index] - 0.15) / 0.12);
  const rampSpan = CAM_RAMP_SPAN_RAD * radiusFactor;
  const excess = distFromTop - CAM_DWELL_HALF_RAD;
  if (excess >= rampSpan) {
    return 0;
  }

  const u = 1.0 - excess / rampSpan;
  // Smooth C1 cam profile (cubic Hermite rising monotonically with contact geometry)
  const liftFactor = u * u * (3.0 - 2.0 * u);
  return alignRad * liftFactor;
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

export interface LockDiagnosticsState {
  keyAngleRad: number;
  boltThrow01: number;
  isLocked: boolean;
  leverAnglesDeg: number[];
  gatesAligned: boolean;
  boltBlocked: boolean;
  bindingLeverIndex: number;
  autoTurning: boolean;
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

  // Scratch object (zero per-frame allocations)
  private readonly _talonTip = { x: 0, y: 0 };

  constructor(initiallyLocked = true) {
    this.boltThrow01 = initiallyLocked ? 1.0 : 0.0;
    this.boltDragTarget01 = this.boltThrow01;
    this.refreshAlignmentFlags();
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
    // Snap base to nearest full turn
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
    // If released within 22 deg of vertical rest, settle smoothly into the vertical keyhole detent
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
    this.draggedLeverTargetRad = Math.max(
      0,
      Math.min(LEVER_MAX_RAD, targetRad ?? this.leverAnglesRad[index])
    );
  }

  setLeverDragTarget(targetRad: number): void {
    if (this.draggedLeverIndex < 0) return;
    this.draggedLeverTargetRad = Math.max(0, Math.min(LEVER_MAX_RAD, targetRad));
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
    this.leverAnglesRad[index] = Math.max(0, Math.min(LEVER_MAX_RAD, rad));
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
        // Weighty acceleration + slight slowdown across heavy bolt throw for tactile mechanical cadence
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
      // Add realistic tactile resistance while lifting all 5 springs against leaf-spring preload
      const phi = normalizeAngle2Pi(this.keyAngleRad);
      const liftLoad = Math.abs(phi - Math.PI) < CAM_DWELL_HALF_RAD + CAM_RAMP_SPAN_RAD ? 0.78 : 1.0;
      const stiffness = 145.0 * liftLoad;
      const accel = err * stiffness - this.keyAngularVel * 20.0;
      this.keyAngularVel += accel * h;
      this.keyAngleRad += this.keyAngularVel * h;
    } else {
      // Gentle keyhole vertical detent pull only when close to vertical 0/2PI rest
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

    // 2. Integrate all 5 brass levers under phosphor-bronze leaf spring torque, key cam contact, and manual pick drag
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

      // Unilateral upward contact with stepped key bitting cam
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

      // Hard physical rest stop on bottom brass-collared rest pin (theta = 0) & top case ceiling
      if (nextAngle <= 0) {
        if (prevAngle > 0.015 && nextVel < -0.5) {
          // Crisp spring-driven metallic clack as lever drops back into locking pocket
          this.detentPulse = Math.min(1.0, this.detentPulse + Math.min(0.32, Math.abs(nextVel) * 0.08));
        }
        nextAngle = 0;
        nextVel = 0;
      } else if (nextAngle >= LEVER_MAX_RAD) {
        nextAngle = LEVER_MAX_RAD;
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

    // Check key rear drive-bit contact with the U-shaped bolt talon
    const inTalonArc = phi >= TALON_ENGAGE_START_RAD - 0.04 && phi <= TALON_ENGAGE_END_RAD + 0.04;
    if (inTalonArc) {
      computeTalonBitTip(phi, this._talonTip);
      const notchX = computeTalonNotchX(this.boltThrow01);
      const talonHalfMouth = 0.044; // realistic clearance inside the U-shaped talon jaws
      const dxTalon = this._talonTip.x - notchX;

      if (Math.abs(dxTalon) <= 0.145) {
        // Key drive bit is physically inside the talon U-notch!
        talonEngaged = true;
        if (this.isDraggingBolt && !this.isDraggingKey && !this.autoTurnActive) {
          // User is directly dragging the deadbolt while the key rests in the talon:
          // the talon jaws smoothly back-drive the key along its engagement arc!
          const twoPi = Math.PI * 2;
          const turnBase = this.keyAngleRad - phi;
          const desiredNotchX = computeTalonNotchX(this.boltThrow01);
          const sinTarget = Math.max(-0.92, Math.min(0.92, (KEY_X - desiredNotchX) / KEY_TALON_RADIUS));
          const targetPhi = Math.PI - Math.asin(sinTarget);
          const targetKeyRad = turnBase + targetPhi;
          this.keyAngleRad += (targetKeyRad - this.keyAngleRad) * Math.min(1, h * 36);
          this.keyAngularVel = 0;
          // Ensure wrapped key stays in same revolution
          if (Math.abs(this.keyAngleRad - turnBase - Math.PI) > twoPi) {
            this.keyAngleRad = targetKeyRad;
          }
        } else {
          // Stiff steel-on-steel key-to-talon drive
          // Scale effective travel span slightly inward (0.95) so bolt seats cleanly at 0/1 before bit exits talon
          const effectiveHalfTravel = BOLT_TRAVEL * 0.475;
          const targetThrow01 = Math.max(
            0,
            Math.min(1, 0.5 + (this._talonTip.x - KEY_X) / (effectiveHalfTravel * 2))
          );
          // Exact analytical kinematic velocity of the talon bit tip along X: d(x)/dt = -R * cos(phi) * d(phi)/dt
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
        // User turned key in the wrong direction when already at travel limit -> hits solid outer cheek of bolt tail!
        talonBlockedByStop = true;
      }
    }

    if (this.isDraggingBolt) {
      boltForce += (this.boltDragTarget01 - this.boltThrow01) * 260.0;
    }

    // 5. Integrate deadbolt linear motion and enforce exact bolt-stump <-> 5-lever collision
    const prevBolt = this.boltThrow01;
    this.boltVel += boltForce * h;
    let nextBolt = prevBolt + this.boltVel * h;

    // Physical hard stops of mortise forend & case backwall
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
    // collides physically with that lever's solid brass locking cheek!
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

    // If the bolt was blocked by misaligned levers while the key talon was actively pushing it,
    // resist the key rotation so the key can NEVER clip through a jammed bolt!
    if (blockedThisSub && talonEngaged) {
      this.keyAngleRad = prevKeyAngle + (this.keyAngleRad - prevKeyAngle) * 0.08;
      this.keyAngularVel *= 0.25;
      if (this.autoTurnActive) {
        this.autoTurnActive = false;
      }
    } else if (talonBlockedByStop && !this.autoTurnActive) {
      // Gentle physical stop when turning the wrong way against an already-thrown deadbolt
      const stopAngle = this.boltThrow01 > 0.5 ? TALON_ENGAGE_START_RAD - 0.02 : TALON_ENGAGE_END_RAD + 0.02;
      const dPrev = Math.abs(normalizeAngle2Pi(prevKeyAngle) - stopAngle);
      const dCur = Math.abs(phi - stopAngle);
      if (dCur < dPrev + 0.08) {
        this.keyAngleRad = prevKeyAngle;
        this.keyAngularVel *= -0.15;
        if (this.isDraggingKey) {
          this.keyDragTargetRad = prevKeyAngle + (this.keyDragTargetRad - prevKeyAngle) * 0.25;
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
    };
  }
}
