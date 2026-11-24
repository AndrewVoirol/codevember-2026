import * as THREE from 'three/webgpu';

export type CoreMaterialRole = 'brass' | 'basalt' | 'calcite' | 'verdigris' | 'amber';

export interface SdfSpherePrimitive {
  kind: 'sphere';
  cx: number;
  cy: number;
  cz: number;
  radius: number;
}

export interface SdfCapsulePrimitive {
  kind: 'capsule';
  ax: number;
  ay: number;
  az: number;
  bx: number;
  by: number;
  bz: number;
  radiusA: number;
  radiusB: number;
}

export interface SdfTorusPrimitive {
  kind: 'torus';
  cx: number;
  cy: number;
  cz: number;
  /** Orthonormal frame: local ring lies in (ux,uy,uz)-(vx,vy,vz) plane, axis along (nx,ny,nz) */
  ux: number;
  uy: number;
  uz: number;
  vx: number;
  vy: number;
  vz: number;
  nx: number;
  ny: number;
  nz: number;
  majorRadius: number;
  minorRadius: number;
}

export interface SdfOrientedBoxPrimitive {
  kind: 'obox';
  cx: number;
  cy: number;
  cz: number;
  /** Orthonormal basis rows/axes for local X, Y, Z */
  ux: number;
  uy: number;
  uz: number;
  vx: number;
  vy: number;
  vz: number;
  wx: number;
  wy: number;
  wz: number;
  hx: number;
  hy: number;
  hz: number;
  bevel: number;
}

export type CoreSdfPrimitive =
  | SdfSpherePrimitive
  | SdfCapsulePrimitive
  | SdfTorusPrimitive
  | SdfOrientedBoxPrimitive;

export interface CoreAssemblyDescriptor {
  index: number;
  name: string;
  primitives: readonly CoreSdfPrimitive[];
  /** Contiguous flat packed SDF buffer (16 floats/primitive) for zero-alloc SIMD-friendly evaluation */
  packedData: Float32Array;
  /** Flat AABB per primitive (6 floats: minX,maxX,minY,maxY,minZ,maxZ) for exact early-out culling */
  packedBounds: Float32Array;
  /** Primitive kind codes: 0=sphere, 1=capsule, 2=torus, 3=obox */
  packedKinds: Uint8Array;
  primCount: number;
  boundingRadius: number;
}

export interface BuiltCoreAssembly {
  descriptor: CoreAssemblyDescriptor;
  group: THREE.Group;
  amberHeartMesh: THREE.Mesh;
  dispose: () => void;
}

export interface CocoonPhysicsConfig {
  /** Icosphere subdivision level (4 -> 2,562 vertices, 5,120 triangles) */
  subdivisions: number;
  /** Base loose cocoon radial envelope */
  looseRadius: number;
  /** Vertical elongation factor of the loose sleeve */
  looseElongationY: number;
  /** Maximum fraction of rest-length contraction at shrink = 1.0 */
  maxShrinkRatio: number;
  /** Minimum offset between membrane centerline and exact analytical core SDF */
  skinClearance: number;
  /** Substeps per animation frame */
  substeps: number;
  /** Constraint Gauss-Seidel relaxation iterations per substep */
  constraintIterations: number;
}

export const DEFAULT_COCOON_CONFIG: CocoonPhysicsConfig = {
  subdivisions: 4,
  looseRadius: 1.44,
  looseElongationY: 1.11,
  maxShrinkRatio: 0.58,
  skinClearance: 0.018,
  substeps: 3,
  constraintIterations: 3,
};

export interface Day20TelemetryState {
  meanShrink: number;
  maxTension: number;
  tearFraction: number;
  coreIndex: number;
  vertexCount: number;
}
