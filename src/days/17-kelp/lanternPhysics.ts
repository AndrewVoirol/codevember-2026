import {
  DEFAULT_LANTERN_CONFIG,
  FILAMENT_TARGETS,
  type FilamentMode,
  type LanternConfig,
  type LanternPhysicsState,
} from './types';

const MODE_ORDER: readonly FilamentMode[] = ['warm', 'bright', 'ember'];
const MAX_SWING_RAD = 0.58;
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
    pokeDepth: 0,
    pokeTargetDepth: 0,
    pokeVel: 0,
    rippleAmp: 0,
    ripplePhase: 0,
    filamentMode: 'warm',
    filamentTarget: FILAMENT_TARGETS.warm,
    filamentGlow: FILAMENT_TARGETS.warm,
    timeSec: 0,
  };
}

/**
 * Applies a tangential swing + torsional impulse to the suspended lantern.
 * Inputs are in world-ish screen-plane deltas (dx right, dy up).
 */
export function applyLanternImpulse(
  state: LanternPhysicsState,
  impulseX: number,
  impulseY: number,
  leverYaw: number
): void {
  const clampedX = Math.max(-2.5, Math.min(2.5, impulseX));
  const clampedY = Math.max(-2.5, Math.min(2.5, impulseY));
  const clampedYaw = Math.max(-3.5, Math.min(3.5, leverYaw));

  // Pushing right (+X) tilts pendulum positive around +Z (since hanging down -Y, +Z rotation moves bottom +X)
  state.swingVelZ += clampedX * 0.52;
  state.swingVelX -= clampedY * 0.34;
  state.yawVel += clampedYaw * 0.85;
}

/**
 * Updates the local kelp-parchment poke target on the lantern skin.
 */
export function setParchmentPoke(
  state: LanternPhysicsState,
  localX: number,
  localY: number,
  localZ: number,
  targetDepth: number
): void {
  state.pokeLocal[0] = localX;
  state.pokeLocal[1] = localY;
  state.pokeLocal[2] = localZ;
  state.pokeTargetDepth = Math.max(0, Math.min(0.18, targetDepth));
}

/**
 * Plucks the kelp parchment membrane, sending a damped drumhead ripple across the skin.
 */
export function pluckParchment(
  state: LanternPhysicsState,
  localX: number,
  localY: number,
  localZ: number,
  strength = 1.0
): void {
  state.pokeLocal[0] = localX;
  state.pokeLocal[1] = localY;
  state.pokeLocal[2] = localZ;
  const s = Math.max(0.2, Math.min(1.6, strength));
  state.pokeVel += 1.45 * s;
  state.rippleAmp = Math.min(1.0, state.rippleAmp + 0.72 * s);
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
 * Advances the coupled pendulum, torsional hemp-cord yaw, internal bulb lag,
 * and taut kelp-parchment membrane physics using sub-stepped semi-implicit Euler.
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
  const swingDamp = 0.78;
  const yawSpring = 0.92;
  const yawDamp = 1.15;

  // Internal bulb short-pendulum parameters
  const bulbOmegaSq = 18.5;
  const bulbDamp = 2.6;

  // Kelp-leather parchment drumhead spring-damper
  const pokeStiffness = 220.0;
  const pokeDamp = 14.5;

  for (let i = 0; i < substeps; i++) {
    state.timeSec += h;
    const t = state.timeSec;

    // Subtle thermal convection air-current breath so the hanging lantern is alive at rest
    const ambientTorqueX = Math.sin(t * 0.73) * 0.014 + Math.cos(t * 1.19) * 0.008;
    const ambientTorqueZ = Math.cos(t * 0.61 + 0.8) * 0.018 + Math.sin(t * 1.07) * 0.009;
    const ambientYaw = Math.sin(t * 0.41) * 0.012;

    // Primary spherical pendulum (small-to-moderate angle sin(theta) restoring torque)
    const accX = -omegaSq * Math.sin(state.swingX) - swingDamp * state.swingVelX + ambientTorqueX;
    const accZ = -omegaSq * Math.sin(state.swingZ) - swingDamp * state.swingVelZ + ambientTorqueZ;

    state.swingVelX += accX * h;
    state.swingVelZ += accZ * h;
    state.swingX += state.swingVelX * h;
    state.swingZ += state.swingVelZ * h;

    // Soft angular ceiling to avoid unrealistic inversion
    if (state.swingX > MAX_SWING_RAD) {
      state.swingX = MAX_SWING_RAD;
      state.swingVelX *= -0.35;
    } else if (state.swingX < -MAX_SWING_RAD) {
      state.swingX = -MAX_SWING_RAD;
      state.swingVelX *= -0.35;
    }
    if (state.swingZ > MAX_SWING_RAD) {
      state.swingZ = MAX_SWING_RAD;
      state.swingVelZ *= -0.35;
    } else if (state.swingZ < -MAX_SWING_RAD) {
      state.swingZ = -MAX_SWING_RAD;
      state.swingVelZ *= -0.35;
    }

    // Torsional hemp-cord twist
    const yawAcc = -yawSpring * state.yaw - yawDamp * state.yawVel + ambientYaw;
    state.yawVel += yawAcc * h;
    state.yaw += state.yawVel * h;

    // Coupled internal bulb pendulum driven by lantern frame acceleration
    const bulbAccX = -bulbOmegaSq * state.bulbSwingX - bulbDamp * state.bulbVelX - accX * 0.42;
    const bulbAccZ = -bulbOmegaSq * state.bulbSwingZ - bulbDamp * state.bulbVelZ - accZ * 0.42;
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
    state.pokeDepth = Math.max(-0.04, Math.min(0.22, state.pokeDepth + state.pokeVel * h));

    // Damped drumhead ripple wave
    if (state.rippleAmp > 1e-4) {
      state.ripplePhase += h * 24.0;
      state.rippleAmp *= Math.exp(-3.4 * h);
      if (state.rippleAmp < 1e-4) state.rippleAmp = 0;
    }

    // Smooth tungsten filament thermal inertia + gentle natural flicker
    const flicker =
      1.0 +
      (Math.sin(t * 7.3) * 0.012 + Math.sin(t * 13.7 + 1.4) * 0.008) *
        Math.min(1.0, state.filamentGlow);
    const desired = state.filamentTarget * flicker;
    state.filamentGlow += (desired - state.filamentGlow) * (1.0 - Math.exp(-6.5 * h));
  }
}
