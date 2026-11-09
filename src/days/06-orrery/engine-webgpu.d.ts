import { type OrreryGeometry } from './geometry-3d';

export interface OrreryGroupInfo {
  id: number;
  code: string;
  name: string;
  summary: string;
}

export interface OrreryCameraPreset {
  label: string;
  az: number;
  el: number;
  dist: number;
  target: [number, number, number] | string;
  fov: number;
}

export interface OrreryMaterialDef {
  rgb: [number, number, number];
  id: number;
  rough: number;
  metal: number;
}

export interface OrreryKinematics {
  day: number;
  dWrap: number;
  season: string;
  zodiacSign: string;
  lonDeg: number;
  declDeg: number;
  eotMin: number;
  angE: number;
  angV: number;
  angM: number;
  angMars: number;
  angJup: number;
  angSat: number;
  angMoon: number;
  angNode: number;
  angApsidal: number;
  eclipseStrength: number;
  marsRetrograde: boolean;
  crownRecoilActive: boolean;
  [key: string]: unknown;
}

export interface OrreryEngineState {
  preset: string;
  az: number;
  el: number;
  dist: number;
  target: [number, number, number];
  fov: number;
  goalAz: number;
  goalEl: number;
  goalDist: number;
  goalTarget: [number, number, number];
  goalFov: number;
  day: number;
  goalDay: number;
  playing: boolean;
  speedDaysPerSec: number;
  cutaway: number;
  goalCutaway: number;
  explode: number;
  goalExplode: number;
  hoveredGroup: number;
  selectedGroup: number;
  isLightTheme: boolean;
  lastFrameMs: number;
  avgFrameMs: number;
}

export interface OrreryInspectedPart {
  meshName: string;
  tx: number;
  ty: number;
  tz: number;
  rx: number;
  ry: number;
  rz: number;
  sx: number;
  sy: number;
  sz: number;
  mat: OrreryMaterialDef;
  groupId: number;
  cutawayFlag: number;
  matrix: Float32Array;
  normalMatrix: Float32Array;
  rodEndpoints?: [[number, number, number], [number, number, number]];
}

export interface OrrerySceneInspection {
  kin: OrreryKinematics;
  parts: OrreryInspectedPart[];
  pickSpheres: Array<{ x: number; y: number; z: number; r: number; groupId: number }>;
  overlayVertCount: number;
  earthPos: [number, number, number];
  moonPos: [number, number, number];
}

export interface OrreryWebGPUEngine {
  state: OrreryEngineState;
  renderFrame: (nowMs: number, dtSec: number) => OrreryKinematics;
  pickAtNormalized: (nx: number, ny: number) => number;
  projectWorldToViewBox: (
    x: number,
    y: number,
    z: number,
    viewW?: number,
    viewH?: number
  ) => [number, number];
  setPreset: (name: string) => void;
  dispose: () => void;
  getActiveDraws: () => number;
  getViewProj: () => Float32Array;
}

export const GROUP_INFO: Record<number, OrreryGroupInfo>;
export const CAMERA_PRESETS: Record<string, OrreryCameraPreset>;
export const MATS: Record<string, OrreryMaterialDef>;

export const T_EARTH: number;
export const T_VENUS: number;
export const T_MERC: number;
export const T_MARS: number;
export const T_JUP: number;
export const T_SAT: number;
export const T_MOON: number;
export const T_SUN_ROT: number;

export function mat4Identity(out: Float32Array): Float32Array;
export function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array;
export function composeTRS(
  out: Float32Array,
  tx: number,
  ty: number,
  tz: number,
  rx: number,
  ry: number,
  rz: number,
  sx: number,
  sy: number,
  sz: number
): Float32Array;
export function mat4LookAt(
  out: Float32Array,
  eye: readonly number[],
  target: readonly number[],
  up: readonly number[]
): Float32Array;
export function mat4PerspectiveZO(
  out: Float32Array,
  fovYRad: number,
  aspect: number,
  near: number,
  far: number
): Float32Array;
export function mat4OrthoZO(
  out: Float32Array,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number
): Float32Array;
export function mat4Invert(out: Float32Array, m: Float32Array): Float32Array;

export function evaluateKinematics(day: number): OrreryKinematics;
export function formatTelemetryForGroup(
  groupId: number,
  kin: OrreryKinematics
): { code: string; name: string; metrics: string; summary: string } | null;

export function inspectOrrerySceneAtDay(
  day: number,
  geometry: OrreryGeometry,
  lightTheme?: boolean,
  cutawayMode?: number,
  explodeMode?: number
): OrrerySceneInspection;

export function initOrreryWebGPU(options: {
  canvas: HTMLCanvasElement;
  shaderSource: string;
  geometry: OrreryGeometry;
  isLightTheme?: boolean;
  onTelemetryChange?: (kin: OrreryKinematics, state: OrreryEngineState) => void;
}): Promise<OrreryWebGPUEngine>;
