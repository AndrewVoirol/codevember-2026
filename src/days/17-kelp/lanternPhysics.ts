import {
  DEFAULT_LANTERN_CONFIG,
  FILAMENT_TARGETS,
  type FilamentMode,
  type LanternConfig,
  type LanternPhysicsState,
} from './types';

const MODE_ORDER: readonly FilamentMode[] = ['warm', 'bright', 'ember'];
const MAX_SWING_RAD = 0.54;
const MAX_BULB_SWING_RAD = 0.24;

export function createLanternPhysicsState(): LanternPhysicsState {
  return {
    swingX: 0,
    swingZ: 0,
    swingVelX: 0,
    swingVelZ: 0,
    yaw: 0,
    yawVel: 0,
    bulbSwingX: 0,
    bulbSwingZ: 0,
    bulbVelX: 0,
    bulbVelZ: 0,
    pokeLocal: [0, 0, 1.0],
    pokeBay: -1,
    pokeDepth: 0,
    pokeTargetDepth: 0,
    pokeVel: 0,
    rippleAmp: 0,
    ripplePhase: 0,
    grabActive: false,
    grabTorqueX: 0,
    grabTorqueZ: 0,
    grabTorqueYaw: 0,
    filamentMode: 'warm',
    filamentTarget: FILAMENT_TARGETS.warm,
    filamentGlow: FILAMENT_TARGETS.warm,
    timeSec: 0,
  };
}

/**
 * Applies an instantaneous tangential swing + torsional impulse to the suspended lantern.
 * Inputs are world-space lateral velocity impulses (impulseX toward +X, impulseZ toward +Z)
 * plus torsional angular impulse around the hanging cord axis.
 *
 * Note: For backwards-compatible callers passing screen-plane `(dx, dy, yaw)`,
 * `impulseX` tilts around +Z (moving the hanging lantern along +X) and
 * `impulseZOrY` tilts around -X (moving the hanging lantern along -Z).
 */
export function applyLanternImpulse(
  state: LanternPhysicsState,
  impulseX: number,
  impulseZOrY: number,
  leverYaw: number
): void {
  const clampedX = Math.max(-2.5, Math.min(2.5, impulseX));
  const clampedZ = Math.max(-2.5, Math.min(2.5, impulseZOrY));
  const clampedYaw = Math.max(-3.5, Math.min(3.5, leverYaw));

  // Hanging along -Y: +Z rotation swings bottom toward +X; -X rotation swings bottom toward +Z
  state.swingVelZ += clampedX * 0.52;
  state.swingVelX -= clampedZ * 0.42;
  state.yawVel += clampedYaw * 0.85;
}

/**
 * Applies a world-space 3D force impulse (fx, fy, fz) at a world-space lever arm
 * relative to the ceiling pivot, plus torsional yaw torque around the vertical cord axis.
 * Works identically from any 360-degree camera orbit angle.
 */
export function applyWorldLanternImpulse(
  state: LanternPhysicsState,
  worldForceX: number,
  worldForceZ: number,
  yawImpulse: number
): void {
  const fx = Math.max(-3.2, Math.min(3.2, worldForceX));
  const fz = Math.max(-3.2, Math.min(3.2, worldForceZ));
  const ty = Math.max(-4.0, Math.min(4.0, yawImpulse));

  // Hanging lantern at (0, -L, 0):
  // +Z rotation moves bottom (+X), so torqueZ ~ +fx
  // -X rotation moves bottom (+Z), so torqueX ~ -fz
  state.swingVelZ += fx * 0.48;
  state.swingVelX -= fz * 0.48;
  state.yawVel += ty * 0.78;
}

/**
 * Sets continuous spring-tether grab torques while the user holds and drags the lantern.
 * Pass `active = false` on release to let momentum carry cleanly into free pendulum swing.
 */
export function setLanternGrabTorque(
  state: LanternPhysicsState,
  torqueX: number,
  torqueZ: number,
  torqueYaw: number,
  active: boolean
): void {
  state.grabActive = active;
  if (!active) {
    state.grabTorqueX = 0;
    state.grabTorqueZ = 0;
    state.grabTorqueYaw = 0;
    return;
  }
  state.grabTorqueX = Math.max(-18.0, Math.min(18.0, torqueX));
  state.grabTorqueZ = Math.max(-18.0, Math.min(18.0, torqueZ));
  state.grabTorqueYaw = Math.max(-22.0, Math.min(22.0, torqueYaw));
}

/**
 * Updates the local kelp-parchment poke target on the lantern skin.
 */
export function setParchmentPoke(
  state: LanternPhysicsState,
  localX: number,
  localY: number,
  localZ: number,
  targetDepth: number,
  bayIdx = -1
): void {
  state.pokeLocal[0] = localX;
  state.pokeLocal[1] = localY;
  state.pokeLocal[2] = localZ;
  state.pokeBay = bayIdx;
  state.pokeTargetDepth = Math.max(0, Math.min(0.16, targetDepth));
}

/**
 * Plucks the kelp parchment membrane, sending a damped drumhead ripple across the skin.
 */
export function pluckParchment(
  state: LanternPhysicsState,
  localX: number,
  localY: number,
  localZ: number,
  strength = 1.0,
  bayIdx = -1
): void {
  state.pokeLocal[0] = localX;
  state.pokeLocal[1] = localY;
  state.pokeLocal[2] = localZ;
  if (bayIdx >= 0) {
    state.pokeBay = bayIdx;
  }
  const s = Math.max(0.2, Math.min(1.6, strength));
  state.pokeVel += 1.55 * s;
  state.rippleAmp = Math.min(1.0, state.rippleAmp + 0.82 * s);
  state.ripplePhase = 0;
}

/**
 * Cycles the internal filament mode (warm -> bright -> ember -> warm).
 */
export function cycleFilamentMode(state: LanternPhysicsState): FilamentMode {
  const idx = MODE_ORDER.indexOf(state.filamentMode);
  const next = MODE_ORDER[(idx + 1) % MODE_ORDER.length];
  state.filamentMode = next;
  state.filamentTarget = FILAMENT_TARGETS[next];
  return next;
}

/**
 * Advances the coupled world-space spherical pendulum, independent torsional hemp-cord yaw,
 * yaw-coupled internal bulb lag, and taut kelp-parchment membrane physics using sub-stepped
 * semi-implicit Euler.
 */
export function stepLanternPhysics(
  state: LanternPhysicsState,
  dtSec: number,
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): void {
  const safeDt = Math.min(0.05, Math.max(0.0005, dtSec));
  const substeps = 4;
  const h = safeDt / substeps;

  // Gravitational angular frequency squared: g / L
  const omegaSq = 9.81 / Math.max(0.5, config.pendulumLength);
  // Extra critical damping while hand-held so the lantern tracks the cursor smoothly without jitter
  const swingDamp = state.grabActive ? 5.6 : 0.72;
  const yawSpring = state.grabActive ? 0.35 : 0.88;
  const yawDamp = state.grabActive ? 5.2 : 1.05;

  // Internal bulb short-pendulum parameters
  const bulbOmegaSq = 20.5;
  const bulbDamp = 2.75;

  // Kelp-leather parchment drumhead spring-damper
  const pokeStiffness = 235.0;
  const pokeDamp = 15.0;

  for (let i = 0; i < substeps; i++) {
    state.timeSec += h;
    const t = state.timeSec;

    // Gentle thermal convection draft so the hanging lantern breathes subtly at rest
    const draftScale = state.grabActive ? 0.15 : 1.0;
    const ambientTorqueX = (Math.sin(t * 0.67) * 0.011 + Math.cos(t * 1.13) * 0.006) * draftScale;
    const ambientTorqueZ = (Math.cos(t * 0.59 + 0.8) * 0.014 + Math.sin(t * 1.01) * 0.007) * draftScale;
    const ambientYaw = Math.sin(t * 0.37) * 0.009 * draftScale;

    // World-space spherical pendulum restoring torque (isotropic around ceiling anchor)
    const swingMag = Math.hypot(state.swingX, state.swingZ);
    const restoreScale = swingMag > 1e-5 ? (omegaSq * Math.sin(swingMag)) / swingMag : omegaSq;

    const accX =
      -restoreScale * state.swingX -
      swingDamp * state.swingVelX +
      ambientTorqueX +
      (state.grabActive ? state.grabTorqueX : 0);
    const accZ =
      -restoreScale * state.swingZ -
      swingDamp * state.swingVelZ +
      ambientTorqueZ +
      (state.grabActive ? state.grabTorqueZ : 0);

    state.swingVelX += accX * h;
    state.swingVelZ += accZ * h;
    state.swingX += state.swingVelX * h;
    state.swingZ += state.swingVelZ * h;

    // Isotropic circular angular ceiling so corners don't snag asymmetrically
    const nextMag = Math.hypot(state.swingX, state.swingZ);
    if (nextMag > MAX_SWING_RAD) {
      const nx = state.swingX / nextMag;
      const nz = state.swingZ / nextMag;
      state.swingX = nx * MAX_SWING_RAD;
      state.swingZ = nz * MAX_SWING_RAD;
      const radialVel = state.swingVelX * nx + state.swingVelZ * nz;
      if (radialVel > 0) {
        state.swingVelX -= radialVel * 1.35 * nx;
        state.swingVelZ -= radialVel * 1.35 * nz;
      }
    }

    // Independent torsional hemp-cord twist around hanging axis
    const yawAcc =
      -yawSpring * state.yaw -
      yawDamp * state.yawVel +
      ambientYaw +
      (state.grabActive ? state.grabTorqueYaw : 0);
    state.yawVel += yawAcc * h;
    state.yaw += state.yawVel * h;
    // Soft cord-twist limit (±1.75 full turns)
    const maxYaw = Math.PI * 3.5;
    if (state.yaw > maxYaw) {
      state.yaw = maxYaw;
      state.yawVel *= -0.3;
    } else if (state.yaw < -maxYaw) {
      state.yaw = -maxYaw;
      state.yawVel *= -0.3;
    }

    // Rotate world pendulum acceleration (accX, accZ) into lantern's yaw-spun local frame
    // so the internal Edison bulb ALWAYS lags opposite true world acceleration even after spinning 180 deg!
    const cosY = Math.cos(state.yaw);
    const sinY = Math.sin(state.yaw);
    const localAccX = accX * cosY - accZ * sinY;
    const localAccZ = accX * sinY + accZ * cosY;

    const bulbAccX = -bulbOmegaSq * state.bulbSwingX - bulbDamp * state.bulbVelX - localAccX * 0.44;
    const bulbAccZ = -bulbOmegaSq * state.bulbSwingZ - bulbDamp * state.bulbVelZ - localAccZ * 0.44;
    state.bulbVelX += bulbAccX * h;
    state.bulbVelZ += bulbAccZ * h;
    state.bulbSwingX = Math.max(
      -MAX_BULB_SWING_RAD,
      Math.min(MAX_BULB_SWING_RAD, state.bulbSwingX + state.bulbVelX * h)
    );
    state.bulbSwingZ = Math.max(
      -MAX_BULB_SWING_RAD,
      Math.min(MAX_BULB_SWING_RAD, state.bulbSwingZ + state.bulbVelZ * h)
    );

    // Taut kelp-parchment skin indentation spring
    const pokeAcc =
      pokeStiffness * (state.pokeTargetDepth - state.pokeDepth) - pokeDamp * state.pokeVel;
    state.pokeVel += pokeAcc * h;
    state.pokeDepth = Math.max(-0.04, Math.min(0.2, state.pokeDepth + state.pokeVel * h));

    // Damped drumhead ripple wave
    if (state.rippleAmp > 1e-4) {
      state.ripplePhase += h * 26.0;
      state.rippleAmp *= Math.exp(-3.1 * h);
      if (state.rippleAmp < 1e-4) state.rippleAmp = 0;
    }

    // Smooth tungsten filament thermal inertia + subtle organic thermal breath
    const flicker =
      1.0 +
      (Math.sin(t * 7.3) * 0.011 + Math.sin(t * 13.7 + 1.4) * 0.007) *
        Math.min(1.0, state.filamentGlow);
    const desired = state.filamentTarget * flicker;
    state.filamentGlow += (desired - state.filamentGlow) * (1.0 - Math.exp(-6.5 * h));
  }
}
