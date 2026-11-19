export type Vec3Tuple = [number, number, number];

export type ArchPigmentMode =
  | 'frontCalcite'
  | 'graphiteHatch'
  | 'rawUmber'
  | 'lapisUltramarine'
  | 'charcoalSlate'
  | 'warmKaolin'
  | 'innerCalcite'
  | 'backSanctum';

export interface ArchLayerSpec {
  index: number;
  z: number;
  rx: number;
  yTop: number;
  yBot: number;
  bevelWidth: number;
  pigmentMode: ArchPigmentMode;
  hatchDensity: number;
  hatchAngle: number;
}

export interface MothSpec {
  id: number;
  perchZ: number;
  pSil: Vec3Tuple;
  perchPos: Vec3Tuple;
  perchScale: number;
  phase: number;
  orbitRadius: number;
  orbitHeight: number;
  flapHz: number;
  flightSize: number;
  pigmentBias: number;
  wingTilt: number;
  perchRoll: number;
}

export interface MothRuntime {
  id: number;
  awake: number;
  pos: Vec3Tuple;
  tangent: Vec3Tuple;
  scale: number;
  wingFlapRad: number;
  foldProgress: number;
  hoverGlow: number;
}

export interface MirageCameraState {
  pos: Vec3Tuple;
  target: Vec3Tuple;
  fovDeg: number;
  easedReveal: number;
  arcAngleDeg: number;
}

export interface MirageDiagnostics {
  uReveal: number;
  uAwake: number;
  effectiveAwake: number;
  lightLift: number;
  sparkPulse: number;
  activeMothCount: number;
  cameraPos: Vec3Tuple;
  cameraTarget: Vec3Tuple;
  fovDeg: number;
  lightPos: Vec3Tuple;
  emberPos: Vec3Tuple;
  faceDotCam: number;
  faceDotLight: number;
  faceLightSwingDeg: number;
  handSeparationPxRatio: number;
  maxRayLockErrorAtRest: number;
  arcAngleDeg: number;
}
