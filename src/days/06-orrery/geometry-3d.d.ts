export interface OrreryMeshEntry {
  firstIndex: number;
  indexCount: number;
  vertexOffset: number;
  vertexCount: number;
}

export interface OrreryGeometry {
  vertexData: Float32Array;
  indexData: Uint32Array;
  meshes: Record<string, OrreryMeshEntry>;
}

export function buildOrreryGeometry(): OrreryGeometry;
