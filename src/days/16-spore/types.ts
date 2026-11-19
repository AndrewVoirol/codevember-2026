export type Vec3Tuple = [number, number, number];

export interface VortexRingState {
  slot: number;
  active: boolean;
  generation: number;
  ageSec: number;
  maxAgeSec: number;
  center: Vec3Tuple;
  axis: Vec3Tuple;
  speed: number;
  majorRadius: number;
  coreRadius: number;
  circulation: number;
  swirlPhase: number;
  burstStrength: number;
  seed: number;
}

export interface PuffSpawnEvent {
  ringSlot: number;
  particleCount: number;
  burstStrength: number;
  nozzlePos: Vec3Tuple;
  axis: Vec3Tuple;
  ringRadius: number;
  seed: number;
}

export interface BellowsStepResult {
  puffEvents: PuffSpawnEvent[];
  continuousEmitCount: number;
  nozzlePos: Vec3Tuple;
  nozzleAxis: Vec3Tuple;
}

export interface BellowsSnapshot {
  compression: number;
  indentStrength: number;
  indentDir: Vec3Tuple;
  pressure: number;
  peristomeFlare: number;
  petalRecoil: number;
  isSqueezing: boolean;
  totalPuffs: number;
  activeRingCount: number;
  rings: readonly VortexRingState[];
}

export interface SporeQaApi {
  getState: () => {
    totalPuffs: number;
    activeRings: number;
    compression: number;
    pressure: number;
    peristomeFlare: number;
    petalRecoil: number;
    particleCount: number;
    ringHeights: number[];
    ringRadii: number[];
  };
  triggerPuff: (strength?: number, pokeX?: number, pokeZ?: number) => void;
}

declare global {
  interface Window {
    __SPORE__?: SporeQaApi;
  }
}
