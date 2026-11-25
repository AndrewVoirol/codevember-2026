import * as THREE from 'three/webgpu';

export interface BloomPetalSpec {
  id: number;
  /** 0 = outer torn rice-paper sepals, 1 = middle sheer silk petals, 2 = inner warm parchment petals */
  whorl: 0 | 1 | 2;
  /** Phyllotactic azimuth around Y (rad) */
  azimuth: number;
  /** Total arc length along petal spine */
  length: number;
  /** Half-width at belly */
  maxWidth: number;
  /** Transverse concave spoon cupping depth */
  cupDepth: number;
  /** Radial attachment distance inside calyx receptacle */
  baseRadius: number;
  /** Tight vertical bud tilt when bloom = 0 (rad; slightly negative leans inward) */
  closedTilt: number;
  /** Poised upward-and-outward blooming tilt at bloom = 1 (rad) */
  openTilt: number;
  /** Staggered cascading bloom delay in [0, 0.42] */
  openDelay: number;
  /** Subtle outward tip curl along upper half of petal */
  tipRecurve: number;
  /** Per-petal paper density / opacity multiplier (sheer rice paper vs denser parchment) */
  opacityBias: number;
  /** Deterministic organic noise seed */
  deckleSeed: number;
}

export interface OrigamiMothState {
  id: number;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  orbitRadius: number;
  orbitHeight: number;
  orbitPhase: number;
  orbitSpeed: number;
  heading: number;
  pitch: number;
  roll: number;
  wingPhase: number;
  wingSpeed: number;
  /** 0 = tucked inside closed heart, 1 = flying aloft */
  liftProgress: number;
  scale: number;
  /** Temporary upward buoyant boost timer (s) on click / burst */
  burstTimer: number;
  active: boolean;
}

export interface CocoonSimConfig {
  /** Initial bloom opening (0 = sealed candled cocoon, 0.74 = poised rice-paper lotus bloom, 1 = full blossom) */
  initialBloom: number;
  /** Harmonic spring stiffness for bloom opening */
  bloomStiffness: number;
  /** Harmonic spring damping for bloom opening */
  bloomDamping: number;
  /** Number of curl-gliding origami moths */
  mothCount: number;
  /** Number of soft backlit curl-advected pollen/silk motes */
  pollenCount: number;
  /** Number of delicate curved stamens in the floral heart */
  stamenCount: number;
}

export const DEFAULT_KOZO_CONFIG: CocoonSimConfig = {
  initialBloom: 0.74,
  bloomStiffness: 16.0,
  bloomDamping: 6.4,
  mothCount: 12,
  pollenCount: 280,
  stamenCount: 18,
};

export interface Day20TelemetryState {
  bloom: number;
  targetBloom: number;
  petalCount: number;
  activeMothCount: number;
  activePollenCount: number;
  emberGlow: number;
}
