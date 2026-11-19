export type FilamentMode = 'warm' | 'bright' | 'ember';

export interface LanternPhysicsState {
  /** Pendulum tilt angles (radians) around X (pitch) and Z (roll) from top ceiling pivot */
  swingX: number;
  swingZ: number;
  swingVelX: number;
  swingVelZ: number;
  /** Torsional rotation (radians) around vertical hemp cord axis */
  yaw: number;
  yawVel: number;
  /** Secondary coupled sway of the internal filament bulb inside the lantern */
  bulbSwingX: number;
  bulbSwingZ: number;
  bulbVelX: number;
  bulbVelZ: number;
  /** Tactile kelp-parchment membrane poke in lantern-local coordinates */
  pokeLocal: [number, number, number];
  pokeDepth: number;
  pokeTargetDepth: number;
  pokeVel: number;
  rippleAmp: number;
  ripplePhase: number;
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
  pivotY: 3.15,
  pendulumLength: 3.15,
  bayCount: 6,
  halfHeight: 0.96,
  maxRadius: 1.02,
};

export const FILAMENT_TARGETS: Readonly<Record<FilamentMode, number>> = {
  warm: 1.0,
  bright: 1.42,
  ember: 0.42,
};
