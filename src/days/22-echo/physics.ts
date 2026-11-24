import * as THREE from 'three/webgpu';
import {
  CAVITY_HALF,
  MAX_MIRROR_TILT_RAD,
  MIRROR_PRESETS,
  computeMirrorPlaneNormals,
} from './optics';

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

/** Top interior brass ceiling suspension anchor Y coordinate. */
export const CORE_ANCHOR_Y = CAVITY_HALF - 0.014;
/** Taut brass suspension wire length so core rests dead-center at y = -0.02. */
export const CORE_WIRE_LENGTH = 0.674;
/** Resting Y coordinate of the suspended amber filament & calcite prism core. */
export const CORE_REST_Y = CORE_ANCHOR_Y - CORE_WIRE_LENGTH;
/** Gravitational acceleration constant for the suspended interior pendulum. */
export const CORE_GRAVITY = 4.4;
/** Maximum pendulum swing angle (radians) keeping safe clearance from all 6 mirror panes. */
export const CORE_MAX_ANGLE_RAD = 0.54;
/** Micrometer screw gear ratio (radians of thumbscrew twist per radian of mirror tilt). */
export const THUMBSCREW_GEAR_RATIO = 28.0;

/**
 * Pure TypeScript damped dynamics controller for Day 22 - Echo:
 * 1. Critically damped spring-damper for the micrometer-tilted interior mirror planes
 *    and coupled knurled brass thumbscrew rotation angles.
 * 2. Exact 3D spherical pendulum simulation for the suspended glowing Realgar-Amber
 *    filament & faceted Bone-Calcite prism core inside the mirror cube.
 *
 * Strictly zero per-frame allocations inside `step(dtSec)`.
 */
export class EchoSimulation {
  /** Current mirror tilt angles in radians. */
  tiltXRad: number;
  tiltYRad: number;
  /** Target mirror tilt angles in radians. */
  targetTiltXRad: number;
  targetTiltYRad: number;
  /** Angular velocities of mirror tilt spring damper (rad/s). */
  tiltVelX = 0;
  tiltVelY = 0;

  /** Continuous rotation angles of the knurled brass micrometer thumbscrews (radians). */
  screwAngleX: number;
  screwAngleY: number;

  /** Active canonical preset index (`0..MIRROR_PRESETS.length-1`). */
  presetIndex = 0;
  /** True when user has manually adjusted mirror tilts away from the active preset. */
  isCustomTilt = false;

  /** Pre-allocated 6 inward-pointing unit mirror normals synced every tick. */
  readonly mirrorNormals: readonly THREE.Vector3[] = [
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, -1),
    new THREE.Vector3(0, 0, 1),
  ];

  /** Suspended glowing filament/prism core world/local position inside cavity. */
  readonly corePos = new THREE.Vector3(0, CORE_REST_Y, 0);
  /** Suspended core 3D linear velocity inside cavity. */
  readonly coreVel = new THREE.Vector3(0.14, 0, 0.085);
  /** Unit axis pointing from core center up toward ceiling brass anchor (coaxial with taut wire). */
  readonly coreAxis = new THREE.Vector3(0, 1, 0);
  /** Slow axial rotation angle of the faceted Bone-Calcite prism around `coreAxis`. */
  coreSpinAngle = 0.35;
  /** Transient warm filament surge [0..1] on click/Space pluck. */
  pluckFlash = 0;
  /** Monotonically increasing physical simulation time (seconds). */
  simTime = 0;

  // Scratch vectors (zero per-frame heap allocations)
  private readonly _offset = new THREE.Vector3();
  private readonly _radial = new THREE.Vector3();

  constructor(initialPresetIndex = 0) {
    const safeIdx =
      ((initialPresetIndex % MIRROR_PRESETS.length) + MIRROR_PRESETS.length) %
      MIRROR_PRESETS.length;
    this.presetIndex = safeIdx;
    const preset = MIRROR_PRESETS[safeIdx];
    this.tiltXRad = preset.tiltXDeg * DEG_TO_RAD;
    this.tiltYRad = preset.tiltYDeg * DEG_TO_RAD;
    this.targetTiltXRad = this.tiltXRad;
    this.targetTiltYRad = this.tiltYRad;
    this.screwAngleX = this.tiltXRad * THUMBSCREW_GEAR_RATIO;
    this.screwAngleY = this.tiltYRad * THUMBSCREW_GEAR_RATIO;
    computeMirrorPlaneNormals(this.tiltXRad, this.tiltYRad, this.mirrorNormals);
    this.projectCoreConstraint();
  }

  getTiltDegrees(): [number, number] {
    return [this.tiltXRad * RAD_TO_DEG, this.tiltYRad * RAD_TO_DEG];
  }

  /**
   * Adjusts target mirror micro-tilt angles incrementally from a pointer drag gesture.
   */
  adjustMirrorTilt(deltaXRad: number, deltaYRad: number): void {
    this.targetTiltXRad = Math.max(
      -MAX_MIRROR_TILT_RAD,
      Math.min(MAX_MIRROR_TILT_RAD, this.targetTiltXRad + deltaXRad)
    );
    this.targetTiltYRad = Math.max(
      -MAX_MIRROR_TILT_RAD,
      Math.min(MAX_MIRROR_TILT_RAD, this.targetTiltYRad + deltaYRad)
    );
    this.isCustomTilt = true;
  }

  /**
   * Sets target mirror tilt directly in degrees (clamped to optical convex bounds).
   */
  setMirrorTiltDeg(tiltXDeg: number, tiltYDeg: number, markCustom = true): void {
    this.targetTiltXRad = Math.max(
      -MAX_MIRROR_TILT_RAD,
      Math.min(MAX_MIRROR_TILT_RAD, tiltXDeg * DEG_TO_RAD)
    );
    this.targetTiltYRad = Math.max(
      -MAX_MIRROR_TILT_RAD,
      Math.min(MAX_MIRROR_TILT_RAD, tiltYDeg * DEG_TO_RAD)
    );
    this.isCustomTilt = markCustom;
  }

  /**
   * Snaps to a specific canonical preset by index.
   */
  setPreset(index: number): void {
    const safeIdx = ((index % MIRROR_PRESETS.length) + MIRROR_PRESETS.length) % MIRROR_PRESETS.length;
    this.presetIndex = safeIdx;
    const preset = MIRROR_PRESETS[safeIdx];
    this.targetTiltXRad = preset.tiltXDeg * DEG_TO_RAD;
    this.targetTiltYRad = preset.tiltYDeg * DEG_TO_RAD;
    this.isCustomTilt = false;
  }

  /**
   * Double-click behavior:
   * - If the user manually dragged tilts away from parallel, cleanly snaps back to
   *   Parallel Hall (presetIndex = 1, 0° / 0°).
   * - Otherwise cycles through canonical mirror geometries (Sweeping Arc -> Parallel Hall ->
   *   Helical Vortex -> Kaleidoscope Corner).
   */
  cyclePreset(): number {
    const parallelIdx = 1;
    const isNearParallel =
      Math.abs(this.targetTiltXRad) < 0.0025 && Math.abs(this.targetTiltYRad) < 0.0025;
    if (this.isCustomTilt && !isNearParallel) {
      this.setPreset(parallelIdx);
    } else {
      this.setPreset(this.presetIndex + 1);
    }
    return this.presetIndex;
  }

  /**
   * Plucks the suspended glowing amber filament & calcite prism core on its 3D pendulum.
   */
  pluckCore(impulseX = 0.72, impulseZ = -0.52): void {
    this.coreVel.x += impulseX;
    this.coreVel.z += impulseZ;

    // Remove any radial velocity component along wire and cap max tangential speed safely
    this._radial.set(this.corePos.x, this.corePos.y - CORE_ANCHOR_Y, this.corePos.z).normalize();
    const vRad = this.coreVel.dot(this._radial);
    this.coreVel.addScaledVector(this._radial, -vRad);

    const maxSpeed = 1.45;
    const speedSq = this.coreVel.lengthSq();
    if (speedSq > maxSpeed * maxSpeed) {
      this.coreVel.multiplyScalar(maxSpeed / Math.sqrt(speedSq));
    }

    this.pluckFlash = 1.0;
  }

  /**
   * Exact specific mechanical energy (kinetic + gravitational potential relative to bottom rest).
   */
  computeCoreMechanicalEnergy(): number {
    const ke = 0.5 * this.coreVel.lengthSq();
    const pe = CORE_GRAVITY * Math.max(0, this.corePos.y - CORE_REST_Y);
    return ke + pe;
  }

  /**
   * Projects core position & velocity strictly onto the spherical tether constraint
   * around ceiling anchor `(0, CORE_ANCHOR_Y, 0)` with length `CORE_WIRE_LENGTH`.
   */
  private projectCoreConstraint(): void {
    this._offset.set(this.corePos.x, this.corePos.y - CORE_ANCHOR_Y, this.corePos.z);
    let dist = this._offset.length();
    if (dist < 1e-6) {
      this._offset.set(0, -CORE_WIRE_LENGTH, 0);
      dist = CORE_WIRE_LENGTH;
    }
    this._radial.copy(this._offset).divideScalar(dist);

    // Clamp polar angle from -Y within CORE_MAX_ANGLE_RAD so prism never clips mirror glass
    const cosTheta = Math.max(-1, Math.min(1, -this._radial.y));
    const minCos = Math.cos(CORE_MAX_ANGLE_RAD);
    if (cosTheta < minCos) {
      const horizLen = Math.hypot(this._radial.x, this._radial.z);
      const maxSin = Math.sin(CORE_MAX_ANGLE_RAD);
      if (horizLen > 1e-6) {
        this._radial.x = (this._radial.x / horizLen) * maxSin;
        this._radial.z = (this._radial.z / horizLen) * maxSin;
      } else {
        this._radial.x = maxSin;
        this._radial.z = 0;
      }
      this._radial.y = -minCos;
      // Gently damp outward swing velocity at angle bumper
      const outwardHoriz = this.coreVel.x * this._radial.x + this.coreVel.z * this._radial.z;
      if (outwardHoriz > 0) {
        this.coreVel.x -= this._radial.x * outwardHoriz * 1.15;
        this.coreVel.z -= this._radial.z * outwardHoriz * 1.15;
      }
    }

    this.corePos.set(
      this._radial.x * CORE_WIRE_LENGTH,
      CORE_ANCHOR_Y + this._radial.y * CORE_WIRE_LENGTH,
      this._radial.z * CORE_WIRE_LENGTH
    );

    // Strip radial velocity so motion stays strictly tangent to the wire sphere
    const vDotR = this.coreVel.dot(this._radial);
    this.coreVel.addScaledVector(this._radial, -vDotR);

    // Core axis points upward from core center along taut wire toward ceiling anchor
    this.coreAxis.copy(this._radial).negate();
  }

  /**
   * Advances mirror tilt spring-damper and suspended core 3D pendulum by `dtSec`.
   */
  step(dtSec: number): void {
    const dt = Math.max(0.0005, Math.min(0.05, dtSec));
    this.simTime += dt;

    // 1. Damped second-order micrometer spring for mirror tilt angles
    const omega = 13.5;
    const zeta = 0.88;
    const kSpring = omega * omega;
    const cDamp = 2 * zeta * omega;

    const substeps = 4;
    const h = dt / substeps;

    for (let s = 0; s < substeps; s++) {
      const accX = kSpring * (this.targetTiltXRad - this.tiltXRad) - cDamp * this.tiltVelX;
      const accY = kSpring * (this.targetTiltYRad - this.tiltYRad) - cDamp * this.tiltVelY;
      this.tiltVelX += accX * h;
      this.tiltVelY += accY * h;
      this.tiltXRad += this.tiltVelX * h;
      this.tiltYRad += this.tiltVelY * h;

      if (this.tiltXRad < -MAX_MIRROR_TILT_RAD) {
        this.tiltXRad = -MAX_MIRROR_TILT_RAD;
        this.tiltVelX = 0;
      } else if (this.tiltXRad > MAX_MIRROR_TILT_RAD) {
        this.tiltXRad = MAX_MIRROR_TILT_RAD;
        this.tiltVelX = 0;
      }
      if (this.tiltYRad < -MAX_MIRROR_TILT_RAD) {
        this.tiltYRad = -MAX_MIRROR_TILT_RAD;
        this.tiltVelY = 0;
      } else if (this.tiltYRad > MAX_MIRROR_TILT_RAD) {
        this.tiltYRad = MAX_MIRROR_TILT_RAD;
        this.tiltVelY = 0;
      }

      // 2. 3D spherical pendulum symplectic substep for suspended filament/prism core
      this.coreVel.y -= CORE_GRAVITY * h;

      // Strip radial component before position update
      this._radial.set(this.corePos.x, this.corePos.y - CORE_ANCHOR_Y, this.corePos.z).normalize();
      const vDotR = this.coreVel.dot(this._radial);
      this.coreVel.addScaledVector(this._radial, -vDotR);

      // Physical linear viscous air damping (~0.24 s^-1 -> smooth ~10s graceful decay)
      const airDrag = Math.exp(-0.24 * h);
      this.coreVel.multiplyScalar(airDrag);

      this.corePos.addScaledVector(this.coreVel, h);
      this.projectCoreConstraint();
    }

    // Update continuous micrometer thumbscrew rotation angles & 6 mirror plane normals
    this.screwAngleX = this.tiltXRad * THUMBSCREW_GEAR_RATIO;
    this.screwAngleY = this.tiltYRad * THUMBSCREW_GEAR_RATIO;
    computeMirrorPlaneNormals(this.tiltXRad, this.tiltYRad, this.mirrorNormals);

    // Slow faceted prism axial rotation & filament pluck flash decay
    const swingSpeed = this.coreVel.length();
    this.coreSpinAngle += dt * (0.28 + swingSpeed * 0.65);
    this.pluckFlash *= Math.exp(-dt * 3.4);
    if (this.pluckFlash < 1e-4) this.pluckFlash = 0;
  }
}
