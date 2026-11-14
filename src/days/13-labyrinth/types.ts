export type CubeFaceId = 'px' | 'nx' | 'py' | 'ny' | 'pz' | 'nz';

export const CUBE_FACES: readonly CubeFaceId[] = ['px', 'nx', 'py', 'ny', 'pz', 'nz'];

export const WET_ATLAS_FACE_SIZE = 256;
export const WET_ATLAS_WIDTH = WET_ATLAS_FACE_SIZE * CUBE_FACES.length; // 1536
export const WET_ATLAS_HEIGHT = WET_ATLAS_FACE_SIZE; // 256
export const EDGE_SAMPLE_COUNT = 9;
export const GRID_PITCH = 0.215;
export const CHECKPOINT_SOL_INDICES = [15, 31, 45] as const;

export interface LabyrinthNode {
  id: number;
  face: CubeFaceId;
  u: number;
  v: number;
  tier: 0 | 1;
  kind: 'corridor' | 'start' | 'goal' | 'portal' | 'underpass';
  localPos: readonly [number, number, number];
  localNormal: readonly [number, number, number];
}

export interface LabyrinthEdge {
  id: number;
  from: number;
  to: number;
  kind: 'surface' | 'bevel-wrap' | 'ramp' | 'underpass';
  length: number;
  samplesPos: Float32Array;
  samplesNormal: Float32Array;
}

export interface StoneBridgeSpec {
  face: CubeFaceId;
  u: number;
  v: number;
  overAxis: 'u' | 'v';
}

export interface LabyrinthTopology {
  cubeHalfExtent: number;
  bevelRadius: number;
  beadRadius: number;
  gridSize: number;
  channelWidth: number;
  surfaceDepth: number;
  underpassDepth: number;
  nodes: readonly LabyrinthNode[];
  edges: readonly LabyrinthEdge[];
  adjacency: ReadonlyArray<readonly number[]>;
  startNodeId: number;
  goalNodeId: number;
  solutionNodePath: readonly number[];
  solutionEdgePath: readonly number[];
  bridges: readonly StoneBridgeSpec[];
}

export interface BeadState {
  edgeId: number;
  t: number;
  speed: number;
  localPos: [number, number, number];
  localNormal: [number, number, number];
  localTangent: [number, number, number];
  impactImpulse: number;
  solved: boolean;
  solveProgress01: number;
  checkpointsUnlocked: number;
  dowsingHint01: number;
}
