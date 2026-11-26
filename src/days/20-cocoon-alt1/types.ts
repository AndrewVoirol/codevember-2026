export type ChrysalisStage = 'bound' | 'unwinding' | 'emerged' | 'buoyant';

export interface SilkLoopState {
  index: number;
  anchorAngle: number;
  anchorV: number;
  peelAmount: number;
  swayX: number;
  swayY: number;
  swayZ: number;
  phase: number;
}

export interface ButterflyState {
  index: number;
  pos: [number, number, number];
  vel: [number, number, number];
  yaw: number;
  pitch: number;
  roll: number;
  wingPhase: number;
  wingFreq: number;
  scale: number;
  warmTint: number;
  emerged: boolean;
  orbitSeed: number;
  active: boolean;
}

export interface EmberMoteState {
  pos: [number, number, number];
  vel: [number, number, number];
  life: number;
  maxLife: number;
  size: number;
  seed: number;
}

export interface SilkLanternSimState {
  stage: ChrysalisStage;
  unravel: number;
  breathExpansion: number;
  ignition: number;
  emberPulse: number;
  lanternSwingX: number;
  lanternSwingZ: number;
  lanternYaw: number;
  lanternLiftY: number;
  butterfliesEmerged: number;
  activeButterflies: number;
  time: number;
}

export interface SilkLanternConfig {
  butterflyCount: number;
  emberCount: number;
  silkLoopCount: number;
  autoAdvance: boolean;
}

export const DEFAULT_SILK_CONFIG: SilkLanternConfig = {
  butterflyCount: 14,
  emberCount: 160,
  silkLoopCount: 28,
  autoAdvance: true,
};

export interface Day20Alt1Telemetry {
  readonly stage: ChrysalisStage;
  readonly unravel: number;
  readonly breathExpansion: number;
  readonly ignition: number;
  readonly emberPulse: number;
  readonly lanternSwingX: number;
  readonly lanternSwingZ: number;
  readonly lanternYaw: number;
  readonly lanternLiftY: number;
  readonly butterfliesEmerged: number;
  readonly activeButterflies: number;
  readonly time: number;
  readonly cameraAzimuth: number;
  readonly cameraPitch: number;
  readonly cameraDistance: number;
}
