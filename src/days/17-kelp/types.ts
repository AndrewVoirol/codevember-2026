export type FilamentMode = 'warm' | 'bright' | 'ember';

export interface LanternPhysicsState {
  /** World-space pendulum tilt angles (radians) around X (pitch) and Z (roll) from top ceiling pivot */
  swingX: number;
  swingZ: number;
  swingVelX: number;
  swingVelZ: number;
  /** Torsional rotation (radians) around the lantern's hanging hemp-cord axis */
  yaw: number;
  yawVel: number;
  /** Secondary coupled sway of the internal filament bulb inside the lantern's local frame */
  bulbSwingX: number;
  bulbSwingZ: number;
  bulbVelX: number;
  bulbVelZ: number;
  /** Tactile kelp-parchment membrane poke in lantern-local coordinates */
  pokeLocal: [number, number, number];
  /** Active poked bay index (0..bayCount-1), or -1 when idle */
  pokeBay: number;
  pokeDepth: number;
  pokeTargetDepth: number;
  pokeVel: number;
  rippleAmp: number;
  ripplePhase: number;
  /** Active spring-tether grab torques while the user holds and pulls the lantern */
  grabActive: boolean;
  grabTorqueX: number;
  grabTorqueZ: number;
  grabTorqueYaw: number;
  /** Filament thermal glow state */
  filamentMode: FilamentMode;
  filamentTarget: number;
  filamentGlow: number;
  /** Elapsed simulation time (seconds) */
  timeSec: number;
}

export interface LanternConfig {
  /** Height of the ceiling anchor above the lantern center (world units) */
  pivotY: number;
  /** Effective pendulum length from ceiling anchor to lantern center of mass */
  pendulumLength: number;
  /** Number of bent walnut meridian ribs / kelp bays */
  bayCount: number;
  /** Lantern vertical half-height */
  halfHeight: number;
  /** Maximum equatorial radius of the lantern belly */
  maxRadius: number;
}

export const DEFAULT_LANTERN_CONFIG: Readonly<LanternConfig> = {
  pivotY: 3.65,
  pendulumLength: 3.65,
  bayCount: 6,
  halfHeight: 0.96,
  maxRadius: 1.02,
};

export const FILAMENT_TARGETS: Readonly<Record<FilamentMode, number>> = {
  warm: 1.0,
  bright: 1.44,
  ember: 0.42,
};
