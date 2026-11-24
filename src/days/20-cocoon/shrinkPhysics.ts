import * as THREE from 'three/webgpu';
import { evalCoreSdf, evalCoreSdfGrad } from './coreAssemblies';
import {
  DEFAULT_COCOON_CONFIG,
  type CocoonPhysicsConfig,
  type CoreAssemblyDescriptor,
} from './types';

// ============================================================================
// Deterministic Irrational-Angle 3D Value Noise & FBM (Zero Polar Harmonics)
// ============================================================================

function hash3Int(ix: number, iy: number, iz: number): number {
  let h =
    (Math.imul(ix | 0, 374761393) +
      Math.imul(iy | 0, 668265263) +
      Math.imul(iz | 0, 2147483647)) |
    0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}

function valueNoise3D(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = hash3Int(ix, iy, iz);
  const n100 = hash3Int(ix + 1, iy, iz);
  const n010 = hash3Int(ix, iy + 1, iz);
  const n110 = hash3Int(ix + 1, iy + 1, iz);
  const n001 = hash3Int(ix, iy, iz + 1);
  const n101 = hash3Int(ix + 1, iy, iz + 1);
  const n011 = hash3Int(ix, iy + 1, iz + 1);
  const n111 = hash3Int(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

function crinkleFbm3D(x: number, y: number, z: number): number {
  const c = 0.80901699;
  const s = 0.58778525;
  const v0 = valueNoise3D(x, y, z);

  const x1 = (x * c - z * s) * 2.03 + 7.13;
  const y1 = (y * c + x * s) * 2.03 + 3.41;
  const z1 = (z * c + y * s) * 2.03 + 5.79;
  const v1 = valueNoise3D(x1, y1, z1);

  const x2 = (y1 * c - x1 * s) * 2.07 + 11.2;
  const y2 = (z1 * c + y1 * s) * 2.07 + 2.9;
  const z2 = (x1 * c + z1 * s) * 2.07 + 8.6;
  const v2 = valueNoise3D(x2, y2, z2);

  return v0 * 0.57 + v1 * 0.29 + v2 * 0.14;
}

function smoothFbm2Oct3D(x: number, y: number, z: number): number {
  const c = 0.80901699;
  const s = 0.58778525;
  const v0 = valueNoise3D(x, y, z);
  const x1 = (x * c - z * s) * 1.97 + 5.41;
  const y1 = (y * c + x * s) * 1.97 + 2.83;
  const z1 = (z * c + y * s) * 1.97 + 8.17;
  const v1 = valueNoise3D(x1, y1, z1);
  return v0 * 0.68 + v1 * 0.32;
}

// ============================================================================
// Subdivided Outward-Wound Organic-Relaxed Icosphere Topology Builder
// ============================================================================

interface BaseMeshTopology {
  vertexCount: number;
  triCount: number;
  edgeCount: number;
  unitDirs: Float32Array;
  indices: Uint32Array;
  edgeA: Int32Array;
  edgeB: Int32Array;
  bendA: Int32Array;
  bendB: Int32Array;
  bendSharedA: Int32Array;
  bendSharedB: Int32Array;
  bendCount: number;
}

function buildIcosphereTopology(subdivisions: number): BaseMeshTopology {
  const t = (1 + Math.sqrt(5)) * 0.5;
  const rawVerts: number[] = [
    -1, t, 0,
    1, t, 0,
    -1, -t, 0,
    1, -t, 0,
    0, -1, t,
    0, 1, t,
    0, -1, -t,
    0, 1, -t,
    t, 0, -1,
    t, 0, 1,
    -t, 0, -1,
    -t, 0, 1,
  ];

  // Normalize initial icosahedron vertices onto unit sphere
  for (let i = 0; i < rawVerts.length; i += 3) {
    const len = Math.hypot(rawVerts[i], rawVerts[i + 1], rawVerts[i + 2]);
    rawVerts[i] /= len;
    rawVerts[i + 1] /= len;
    rawVerts[i + 2] /= len;
  }

  let tris: number[] = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];

  for (let s = 0; s < subdivisions; s++) {
    const midpointCache = new Map<number, number>();
    const nextTris: number[] = [];

    const getMidpoint = (i0: number, i1: number): number => {
      const lo = i0 < i1 ? i0 : i1;
      const hi = i0 < i1 ? i1 : i0;
      const key = lo * 65536 + hi;
      const cached = midpointCache.get(key);
      if (cached !== undefined) return cached;

      const mx = 0.5 * (rawVerts[lo * 3] + rawVerts[hi * 3]);
      const my = 0.5 * (rawVerts[lo * 3 + 1] + rawVerts[hi * 3 + 1]);
      const mz = 0.5 * (rawVerts[lo * 3 + 2] + rawVerts[hi * 3 + 2]);
      const len = Math.hypot(mx, my, mz);
      const idx = rawVerts.length / 3;
      rawVerts.push(mx / len, my / len, mz / len);
      midpointCache.set(key, idx);
      return idx;
    };

    for (let i = 0; i < tris.length; i += 3) {
      const a = tris[i];
      const b = tris[i + 1];
      const c = tris[i + 2];
      const ab = getMidpoint(a, b);
      const bc = getMidpoint(b, c);
      const ca = getMidpoint(c, a);
      nextTris.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    tris = nextTris;
  }

  const vertexCount = rawVerts.length / 3;
  const triCount = tris.length / 3;
  const unitDirs = new Float32Array(rawVerts);
  const indices = new Uint32Array(tris);

  // Enforce strict outward CCW winding on every triangle before organic relaxation
  for (let ti = 0; ti < triCount; ti++) {
    const i0 = indices[ti * 3];
    const i1 = indices[ti * 3 + 1];
    const i2 = indices[ti * 3 + 2];
    const ax = unitDirs[i0 * 3];
    const ay = unitDirs[i0 * 3 + 1];
    const az = unitDirs[i0 * 3 + 2];
    const bx = unitDirs[i1 * 3];
    const by = unitDirs[i1 * 3 + 1];
    const bz = unitDirs[i1 * 3 + 2];
    const cx = unitDirs[i2 * 3];
    const cy = unitDirs[i2 * 3 + 1];
    const cz = unitDirs[i2 * 3 + 2];

    const e1x = bx - ax;
    const e1y = by - ay;
    const e1z = bz - az;
    const e2x = cx - ax;
    const e2y = cy - ay;
    const e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const dotOut = nx * (ax + bx + cx) + ny * (ay + by + cy) + nz * (az + bz + cz);
    if (dotOut < 0) {
      indices[ti * 3 + 1] = i2;
      indices[ti * 3 + 2] = i1;
    }
  }

  // Deterministic organic tangential vertex jitter on S^2:
  // Breaks the regular 60-degree geodesic icosphere lattice so neither shrink pleats nor
  // tear paths ever fall onto rigid geometric grid lines, while strictly preserving outward
  // CCW winding and well-conditioned triangle aspect ratios (zero slivers/inversions).
  const vertTris: number[][] = Array.from({ length: vertexCount }, () => []);
  for (let ti = 0; ti < triCount; ti++) {
    vertTris[indices[ti * 3]].push(ti);
    vertTris[indices[ti * 3 + 1]].push(ti);
    vertTris[indices[ti * 3 + 2]].push(ti);
  }

  // Measure nominal angular edge length at this subdivision level
  const refA = indices[0] * 3;
  const refB = indices[1] * 3;
  const hEdge = Math.hypot(
    unitDirs[refB] - unitDirs[refA],
    unitDirs[refB + 1] - unitDirs[refA + 1],
    unitDirs[refB + 2] - unitDirs[refA + 2]
  );
  const minTriArea = 0.24 * hEdge * hEdge;

  for (let v = 0; v < vertexCount; v++) {
    const v3 = v * 3;
    const ux = unitDirs[v3];
    const uy = unitDirs[v3 + 1];
    const uz = unitDirs[v3 + 2];

    // Orthonormal tangent basis (t1, t2) on unit sphere at u
    let rx = Math.abs(uy) < 0.85 ? 0 : 1;
    let ry = Math.abs(uy) < 0.85 ? 1 : 0;
    let rz = 0;
    let t1x = uy * rz - uz * ry;
    let t1y = uz * rx - ux * rz;
    let t1z = ux * ry - uy * rx;
    const t1Len = Math.hypot(t1x, t1y, t1z) + 1e-8;
    t1x /= t1Len;
    t1y /= t1Len;
    t1z /= t1Len;
    const t2x = uy * t1z - uz * t1y;
    const t2y = uz * t1x - ux * t1z;
    const t2z = ux * t1y - uy * t1x;

    // Combine gentle organic domain swirl with per-vertex irregular jitter (±20% edge length)
    const swirlA = smoothFbm2Oct3D(ux * 3.1 + 1.4, uy * 3.1 + 4.7, uz * 3.1 + 2.3) - 0.5;
    const swirlB = smoothFbm2Oct3D(ux * 3.1 + 8.9, uy * 3.1 + 1.2, uz * 3.1 + 6.5) - 0.5;
    const jA = hash3Int(v, 19, 37) - 0.5;
    const jB = hash3Int(v, 61, 103) - 0.5;
    let d1 = (swirlA * 0.22 + jA * 0.28) * hEdge;
    let d2 = (swirlB * 0.22 + jB * 0.28) * hEdge;

    const inc = vertTris[v];
    for (let attempt = 0; attempt < 4; attempt++) {
      let cx = ux + t1x * d1 + t2x * d2;
      let cy = uy + t1y * d1 + t2y * d2;
      let cz = uz + t1z * d1 + t2z * d2;
      const cLen = Math.hypot(cx, cy, cz);
      cx /= cLen;
      cy /= cLen;
      cz /= cLen;

      unitDirs[v3] = cx;
      unitDirs[v3 + 1] = cy;
      unitDirs[v3 + 2] = cz;

      let valid = true;
      for (let k = 0; k < inc.length; k++) {
        const ti = inc[k];
        const i0 = indices[ti * 3] * 3;
        const i1 = indices[ti * 3 + 1] * 3;
        const i2 = indices[ti * 3 + 2] * 3;
        const ax = unitDirs[i0];
        const ay = unitDirs[i0 + 1];
        const az = unitDirs[i0 + 2];
        const e1x = unitDirs[i1] - ax;
        const e1y = unitDirs[i1 + 1] - ay;
        const e1z = unitDirs[i1 + 2] - az;
        const e2x = unitDirs[i2] - ax;
        const e2y = unitDirs[i2 + 1] - ay;
        const e2z = unitDirs[i2 + 2] - az;
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        const area = 0.5 * Math.hypot(nx, ny, nz);
        const dotOut =
          nx * (ax + unitDirs[i1] + unitDirs[i2]) +
          ny * (ay + unitDirs[i1 + 1] + unitDirs[i2 + 1]) +
          nz * (az + unitDirs[i1 + 2] + unitDirs[i2 + 2]);
        if (dotOut <= 0.0005 || area < minTriArea) {
          valid = false;
          break;
        }
      }

      if (valid) break;
      d1 *= 0.45;
      d2 *= 0.45;
      if (attempt === 3) {
        unitDirs[v3] = ux;
        unitDirs[v3 + 1] = uy;
        unitDirs[v3 + 2] = uz;
      }
    }
  }

  // Extract unique undirected structural edges & opposite-vertex dihedral bending pairs
  const edgeMap = new Map<number, { a: number; b: number; opp0: number; opp1: number }>();
  const registerEdge = (u: number, v: number, opp: number) => {
    const lo = u < v ? u : v;
    const hi = u < v ? v : u;
    const key = lo * 65536 + hi;
    const existing = edgeMap.get(key);
    if (!existing) {
      edgeMap.set(key, { a: lo, b: hi, opp0: opp, opp1: -1 });
    } else if (existing.opp1 === -1) {
      existing.opp1 = opp;
    }
  };

  for (let ti = 0; ti < triCount; ti++) {
    const i0 = indices[ti * 3];
    const i1 = indices[ti * 3 + 1];
    const i2 = indices[ti * 3 + 2];
    registerEdge(i0, i1, i2);
    registerEdge(i1, i2, i0);
    registerEdge(i2, i0, i1);
  }

  const edgeCount = edgeMap.size;
  const edgeA = new Int32Array(edgeCount);
  const edgeB = new Int32Array(edgeCount);
  const bendATemp: number[] = [];
  const bendBTemp: number[] = [];
  const bendSharedATemp: number[] = [];
  const bendSharedBTemp: number[] = [];

  let eIdx = 0;
  for (const entry of edgeMap.values()) {
    edgeA[eIdx] = entry.a;
    edgeB[eIdx] = entry.b;
    if (entry.opp0 !== -1 && entry.opp1 !== -1) {
      bendATemp.push(entry.opp0);
      bendBTemp.push(entry.opp1);
      bendSharedATemp.push(entry.a);
      bendSharedBTemp.push(entry.b);
    }
    eIdx++;
  }

  return {
    vertexCount,
    triCount,
    edgeCount,
    unitDirs,
    indices,
    edgeA,
    edgeB,
    bendA: new Int32Array(bendATemp),
    bendB: new Int32Array(bendBTemp),
    bendSharedA: new Int32Array(bendSharedATemp),
    bendSharedB: new Int32Array(bendSharedBTemp),
    bendCount: bendATemp.length,
  };
}

// ============================================================================
// Crack-Tip Propagation State & Fibrous Silk Tether Pool Constants
// ============================================================================

const MAX_TETHERS = 72;
const TETHER_SEGS = 6;
const TETHER_VERTS_PER_STRAND = (TETHER_SEGS + 1) * 2;
const TETHER_INDICES_PER_STRAND = TETHER_SEGS * 6;

interface ActiveCrackTip {
  tipVertex: number;
  prevA: number;
  prevB: number;
  dirX: number;
  dirY: number;
  dirZ: number;
  remainingSteps: number;
  cooldownSec: number;
  meanderPhase: number;
  meanderRate: number;
  branchBudget: number;
}

interface IncidentTriInfo {
  triIndex: number;
  prevVert: number; // vertex after center in CCW? Let's define CCW triple (center, nextV, prevV): outgoing edge center->nextV, incoming edge prevV->center
  nextVert: number;
}

// ============================================================================
// Shrink-Film XPBD Solver & Topological Seam-Tear Engine
// ============================================================================

export class ShrinkFilmSimulation {
  readonly config: CocoonPhysicsConfig;
  readonly baseVertexCount: number;
  readonly triCount: number;
  readonly maxVertices: number;
  readonly maxEdges: number;

  readonly geometry: THREE.BufferGeometry;
  readonly posAttr: THREE.BufferAttribute;
  readonly normalAttr: THREE.BufferAttribute;
  readonly restDirAttr: THREE.BufferAttribute;
  readonly stateAttr: THREE.BufferAttribute;
  readonly tearEdgeAttr: THREE.BufferAttribute;
  readonly organicAttr: THREE.BufferAttribute;
  readonly indexAttr: THREE.BufferAttribute;

  // Delicate stretched fibrous silk strands spanning freshly opened tear gaps (zero-GC pre-allocated pool)
  readonly tetherGeometry: THREE.BufferGeometry;
  private readonly tetherPos: Float32Array;
  private readonly tetherNorm: Float32Array;
  private readonly tetherUv: Float32Array;
  private readonly tetherStrain: Float32Array; // vec2: [strain 0..1, visibilityAlpha 0..1]
  private readonly tetherPosAttr: THREE.BufferAttribute;
  private readonly tetherNormAttr: THREE.BufferAttribute;
  private readonly tetherStrainAttr: THREE.BufferAttribute;
  private readonly tetherA = new Int32Array(MAX_TETHERS);
  private readonly tetherB = new Int32Array(MAX_TETHERS);
  private readonly tetherMaxSpan = new Float32Array(MAX_TETHERS);
  private readonly tetherBreakProgress = new Float32Array(MAX_TETHERS);
  private readonly tetherPhase = new Float32Array(MAX_TETHERS);
  private readonly tetherActive = new Uint8Array(MAX_TETHERS); // 0=unused, 1=stretched, 2=snapping
  private tetherCursor = 0;
  activeTetherCount = 0;

  // Exposed typed arrays (backing GPU attributes directly — zero per-frame copy/alloc!)
  readonly pos: Float32Array;
  readonly normals: Float32Array;
  readonly restDirs: Float32Array;
  readonly stateData: Float32Array; // vec4: [shrink, tension, heat, contact]
  readonly tearEdgeData: Float32Array;
  readonly organicData: Float32Array; // vec4: [wallThickness, lipOpenness, fiberPhase, edgeTaper]
  readonly indices: Uint32Array;

  // Internal simulation state
  readonly shrink: Float32Array;
  readonly heat: Float32Array;
  readonly tension: Float32Array;
  readonly contact: Float32Array;
  readonly sdfDist: Float32Array;
  readonly wallThickness: Float32Array;
  readonly fiberPhase: Float32Array;

  private readonly prevPos: Float32Array;
  private readonly vel: Float32Array;
  private readonly looseHomePos: Float32Array;
  private readonly heatScratch: Float32Array;
  private readonly tensionAccum: Float32Array;
  private readonly tensionCount: Uint16Array;
  private readonly flapPeel: Float32Array;
  private readonly flapPeelScratch: Float32Array;
  private readonly flapRecoilEnergy: Float32Array;
  private readonly flapRecoilScratch: Float32Array;
  private readonly seamRecoilDir: Float32Array;
  private readonly tearPairTwin: Int32Array;
  private readonly lipOpenness: Float32Array;
  private readonly lipOpenScratch: Float32Array;
  private readonly edgeTaper: Float32Array;
  private readonly edgeTaperScratch: Float32Array;
  private readonly suctionAccel: Float32Array;
  private readonly nearCoreMask: Uint8Array;
  private readonly surfaceMinGap: Float32Array;
  private readonly lipSmoothAccum: Float32Array;
  private readonly lipSmoothCount: Uint16Array;
  private readonly candEdges: Int32Array;
  private readonly candTris: Int32Array;
  private readonly gradScratch = new Float32Array(3);

  // Dynamic edge & bending tables (pre-allocated pools)
  private readonly edgeA: Int32Array;
  private readonly edgeB: Int32Array;
  private readonly edgeBaseRest: Float32Array;
  private readonly edgeActive: Uint8Array;

  private readonly bendA: Int32Array;
  private readonly bendB: Int32Array;
  private readonly bendBaseRest: Float32Array;
  private readonly bendActive: Uint8Array;

  // Immutable base topology for instant zero-GC re-bagging reset
  private readonly topo: BaseMeshTopology;

  activeVertexCount: number;
  activeEdgeCount: number;
  splitCount = 0;
  simTime = 0;
  rewrapAnimProgress = 1.0;

  // Aggregate telemetry updated every step
  meanShrink = 0;
  maxTension = 0;
  tearFraction = 0;

  // Active crack tip queue (short, bounded array)
  private readonly crackTips: ActiveCrackTip[] = [];
  private core: CoreAssemblyDescriptor;
  private rewrapSeed = 0;

  // Reusable scratch collections for topological vertex-split (zero steady-state alloc)
  private readonly splitIncTris: IncidentTriInfo[] = [];
  private readonly splitSectorBTris: number[] = [];
  private readonly splitOnlyBVerts = new Set<number>();
  private readonly fanUsedScratch = new Uint8Array(64);

  constructor(
    initialCore: CoreAssemblyDescriptor,
    config: CocoonPhysicsConfig = DEFAULT_COCOON_CONFIG
  ) {
    this.config = { ...config };
    this.core = initialCore;
    this.topo = buildIcosphereTopology(this.config.subdivisions);

    this.baseVertexCount = this.topo.vertexCount;
    this.triCount = this.topo.triCount;
    this.maxVertices = this.baseVertexCount + 1400;
    this.maxEdges = this.topo.edgeCount + 2800;

    this.pos = new Float32Array(this.maxVertices * 3);
    this.prevPos = new Float32Array(this.maxVertices * 3);
    this.vel = new Float32Array(this.maxVertices * 3);
    this.looseHomePos = new Float32Array(this.maxVertices * 3);
    this.normals = new Float32Array(this.maxVertices * 3);
    this.restDirs = new Float32Array(this.maxVertices * 3);
    this.stateData = new Float32Array(this.maxVertices * 4);
    this.tearEdgeData = new Float32Array(this.maxVertices);
    this.organicData = new Float32Array(this.maxVertices * 4);
    this.indices = new Uint32Array(this.triCount * 3);

    this.shrink = new Float32Array(this.maxVertices);
    this.heat = new Float32Array(this.maxVertices);
    this.heatScratch = new Float32Array(this.maxVertices);
    this.tension = new Float32Array(this.maxVertices);
    this.tensionAccum = new Float32Array(this.maxVertices);
    this.tensionCount = new Uint16Array(this.maxVertices);
    this.contact = new Float32Array(this.maxVertices);
    this.sdfDist = new Float32Array(this.maxVertices);
    this.wallThickness = new Float32Array(this.maxVertices);
    this.fiberPhase = new Float32Array(this.maxVertices);
    this.flapPeel = new Float32Array(this.maxVertices);
    this.flapPeelScratch = new Float32Array(this.maxVertices);
    this.flapRecoilEnergy = new Float32Array(this.maxVertices);
    this.flapRecoilScratch = new Float32Array(this.maxVertices);
    this.seamRecoilDir = new Float32Array(this.maxVertices * 3);
    this.tearPairTwin = new Int32Array(this.maxVertices);
    this.lipOpenness = new Float32Array(this.maxVertices);
    this.lipOpenScratch = new Float32Array(this.maxVertices);
    this.edgeTaper = new Float32Array(this.maxVertices);
    this.edgeTaperScratch = new Float32Array(this.maxVertices);
    this.suctionAccel = new Float32Array(this.maxVertices * 3);
    this.nearCoreMask = new Uint8Array(this.maxVertices);
    this.surfaceMinGap = new Float32Array(this.maxVertices);
    this.lipSmoothAccum = new Float32Array(this.maxVertices * 3);
    this.lipSmoothCount = new Uint16Array(this.maxVertices);
    this.candEdges = new Int32Array(this.maxEdges);
    this.candTris = new Int32Array(this.triCount);

    this.edgeA = new Int32Array(this.maxEdges);
    this.edgeB = new Int32Array(this.maxEdges);
    this.edgeBaseRest = new Float32Array(this.maxEdges);
    this.edgeActive = new Uint8Array(this.maxEdges);

    this.bendA = new Int32Array(this.topo.bendCount);
    this.bendB = new Int32Array(this.topo.bendCount);
    this.bendBaseRest = new Float32Array(this.topo.bendCount);
    this.bendActive = new Uint8Array(this.topo.bendCount);

    this.activeVertexCount = this.baseVertexCount;
    this.activeEdgeCount = this.topo.edgeCount;

    // Build shared BufferGeometry backed directly by our pre-allocated typed arrays
    this.geometry = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.pos, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.normalAttr = new THREE.BufferAttribute(this.normals, 3);
    this.normalAttr.setUsage(THREE.DynamicDrawUsage);
    this.restDirAttr = new THREE.BufferAttribute(this.restDirs, 3);
    this.restDirAttr.setUsage(THREE.DynamicDrawUsage);
    this.stateAttr = new THREE.BufferAttribute(this.stateData, 4);
    this.stateAttr.setUsage(THREE.DynamicDrawUsage);
    this.tearEdgeAttr = new THREE.BufferAttribute(this.tearEdgeData, 1);
    this.tearEdgeAttr.setUsage(THREE.DynamicDrawUsage);
    this.organicAttr = new THREE.BufferAttribute(this.organicData, 4);
    this.organicAttr.setUsage(THREE.DynamicDrawUsage);
    this.indexAttr = new THREE.BufferAttribute(this.indices, 1);
    this.indexAttr.setUsage(THREE.DynamicDrawUsage);

    this.geometry.setAttribute('position', this.posAttr);
    this.geometry.setAttribute('normal', this.normalAttr);
    this.geometry.setAttribute('aRestDir', this.restDirAttr);
    this.geometry.setAttribute('aState', this.stateAttr);
    this.geometry.setAttribute('aTearEdge', this.tearEdgeAttr);
    this.geometry.setAttribute('aOrganic', this.organicAttr);
    this.geometry.setIndex(this.indexAttr);

    // Pre-set generous fixed bounding sphere so frustum culling never clips curled flaps
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.0);

    // Pre-allocate fibrous silk tether strand pool (zero per-frame allocations)
    const totalTetherVerts = MAX_TETHERS * TETHER_VERTS_PER_STRAND;
    const totalTetherIndices = MAX_TETHERS * TETHER_INDICES_PER_STRAND;
    this.tetherPos = new Float32Array(totalTetherVerts * 3);
    this.tetherNorm = new Float32Array(totalTetherVerts * 3);
    this.tetherUv = new Float32Array(totalTetherVerts * 2);
    this.tetherStrain = new Float32Array(totalTetherVerts * 2);
    const tetherIdx = new Uint16Array(totalTetherIndices);

    for (let s = 0; s < MAX_TETHERS; s++) {
      const vBase = s * TETHER_VERTS_PER_STRAND;
      const iBase = s * TETHER_INDICES_PER_STRAND;
      for (let k = 0; k <= TETHER_SEGS; k++) {
        const u = k / TETHER_SEGS;
        const v0 = vBase + k * 2;
        this.tetherUv[v0 * 2] = u;
        this.tetherUv[v0 * 2 + 1] = 0;
        this.tetherUv[(v0 + 1) * 2] = u;
        this.tetherUv[(v0 + 1) * 2 + 1] = 1;
        this.tetherNorm[v0 * 3 + 2] = 1;
        this.tetherNorm[(v0 + 1) * 3 + 2] = 1;
      }
      for (let k = 0; k < TETHER_SEGS; k++) {
        const a = vBase + k * 2;
        const b = a + 1;
        const c = a + 2;
        const d = a + 3;
        const o = iBase + k * 6;
        tetherIdx[o] = a;
        tetherIdx[o + 1] = c;
        tetherIdx[o + 2] = b;
        tetherIdx[o + 3] = b;
        tetherIdx[o + 4] = c;
        tetherIdx[o + 5] = d;
      }
    }

    this.tetherGeometry = new THREE.BufferGeometry();
    this.tetherPosAttr = new THREE.BufferAttribute(this.tetherPos, 3);
    this.tetherPosAttr.setUsage(THREE.DynamicDrawUsage);
    this.tetherNormAttr = new THREE.BufferAttribute(this.tetherNorm, 3);
    this.tetherNormAttr.setUsage(THREE.DynamicDrawUsage);
    this.tetherStrainAttr = new THREE.BufferAttribute(this.tetherStrain, 2);
    this.tetherStrainAttr.setUsage(THREE.DynamicDrawUsage);
    this.tetherGeometry.setAttribute('position', this.tetherPosAttr);
    this.tetherGeometry.setAttribute('normal', this.tetherNormAttr);
    this.tetherGeometry.setAttribute('uv', new THREE.BufferAttribute(this.tetherUv, 2));
    this.tetherGeometry.setAttribute('aTetherStrain', this.tetherStrainAttr);
    this.tetherGeometry.setIndex(new THREE.BufferAttribute(tetherIdx, 1));
    this.tetherGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.0);
    this.tetherGeometry.setDrawRange(0, 0);

    this.reset(initialCore, false);
  }

  getCore(): CoreAssemblyDescriptor {
    return this.core;
  }

  /**
   * Resets/re-wraps a fresh organic silkworm-parchment cocoon around `nextCore`.
   */
  reset(nextCore: CoreAssemblyDescriptor, animateDrop = true): void {
    this.core = nextCore;
    this.rewrapSeed += 1;
    this.activeVertexCount = this.baseVertexCount;
    this.activeEdgeCount = this.topo.edgeCount;
    this.splitCount = 0;
    this.crackTips.length = 0;
    this.meanShrink = 0;
    this.maxTension = 0;
    this.tearFraction = 0;
    this.rewrapAnimProgress = animateDrop ? 0.0 : 1.0;

    // Reset silk tether pool cleanly
    this.tetherActive.fill(0);
    this.tetherBreakProgress.fill(0);
    this.tetherStrain.fill(0);
    this.tetherPos.fill(0);
    this.tetherCursor = 0;
    this.activeTetherCount = 0;
    this.tetherGeometry.setDrawRange(0, 0);
    this.tetherPosAttr.needsUpdate = true;
    this.tetherStrainAttr.needsUpdate = true;

    // Restore immutable triangle index topology
    this.indices.set(this.topo.indices);
    this.tearPairTwin.fill(-1);
    this.lipOpenness.fill(0);
    this.edgeTaper.fill(0);

    const seedShift = (this.rewrapSeed * 3.71) % 37.0;
    const baseR = this.config.looseRadius;
    const elongY = this.config.looseElongationY;
    const expandStart = animateDrop ? 1.18 : 1.0;

    for (let i = 0; i < this.baseVertexCount; i++) {
      const i3 = i * 3;
      const ux = this.topo.unitDirs[i3];
      const uy = this.topo.unitDirs[i3 + 1];
      const uz = this.topo.unitDirs[i3 + 2];

      // Smooth mesh-resolved organic silkworm-cocoon & botanical husk lofting (freq 1.4..2.6):
      // Deliberately avoids Nyquist-aliased vertex noise so silhouettes stay silky-smooth & organic!
      const wX = smoothFbm2Oct3D(ux * 1.45 + seedShift, uy * 1.45 + 2.1, uz * 1.45 + 5.7) - 0.5;
      const wY = smoothFbm2Oct3D(ux * 1.45 + 6.4 + seedShift, uy * 1.45 + 1.3, uz * 1.45 + 3.2) - 0.5;
      const wZ = smoothFbm2Oct3D(ux * 1.45 + 3.8 + seedShift, uy * 1.45 + 7.5, uz * 1.45 + 0.9) - 0.5;

      // Domain-warped gentle silk-wrap sculptural contours + organic shallow folds
      const qx = ux + wX * 0.34;
      const qy = uy + wY * 0.34;
      const qz = uz + wZ * 0.34;
      const fMacro = smoothFbm2Oct3D(qx * 1.95 + seedShift, qy * 1.75 + 1.7, qz * 1.95 + 4.3) - 0.5;
      const fFold = smoothFbm2Oct3D(qx * 2.75 + 9.2 + seedShift, qy * 2.15 + 2.8, qz * 2.75 + 6.1) - 0.5;

      // Natural silkworm-cocoon ovoid profile: gentle mid-equatorial waisting + subtle asymmetric fullness
      const cocoonWaist = 1.0 - 0.052 * Math.exp(-uy * uy * 5.6);
      const botanicalTaper = 1.0 + 0.038 * uy - 0.022 * uy * uy + 0.018 * (ux * 0.7 - uz * 0.7);
      // C1 soft organic drape crease (broad enough to resolve cleanly across 6-8 triangles)
      const softRidge = 0.5 - Math.sqrt(fMacro * fMacro * 3.2 + 0.09);
      const r =
        baseR *
        cocoonWaist *
        botanicalTaper *
        (1.0 + fMacro * 0.072 + fFold * 0.032 + softRidge * 0.022);

      const hx = ux * r;
      const hy = uy * r * elongY;
      const hz = uz * r;

      this.looseHomePos[i3] = hx;
      this.looseHomePos[i3 + 1] = hy;
      this.looseHomePos[i3 + 2] = hz;

      // Uneven organic wall thickness (luminous thin honey-kaolin windows vs denser silk veins)
      const thickWarp = crinkleFbm3D(qx * 2.8 + 4.1 + seedShift, qy * 3.6 + 1.9, qz * 2.8 + 7.3);
      const silkBand = Math.sin((qy * 5.2 + qx * 1.9 - qz * 1.4 + thickWarp * 2.4) * Math.PI) * 0.5 + 0.5;
      const thickVal = Math.max(0.20, Math.min(1.0, thickWarp * 0.72 + silkBand * 0.28));
      this.wallThickness[i] = thickVal;

      // Per-vertex organic curl/stiffness phase for asymmetric natural flap peeling
      const phaseVal = smoothFbm2Oct3D(ux * 3.4 + 12.1 + seedShift, uy * 3.4 + 5.3, uz * 3.4 + 8.7);
      this.fiberPhase[i] = phaseVal;

      // Ensure initial loose sleeve strictly clears core SDF
      const px = hx * expandStart;
      const py = hy * expandStart + (animateDrop ? 0.16 : 0);
      const pz = hz * expandStart;

      this.pos[i3] = px;
      this.pos[i3 + 1] = py;
      this.pos[i3 + 2] = pz;
      this.prevPos[i3] = px;
      this.prevPos[i3 + 1] = py;
      this.prevPos[i3 + 2] = pz;
      this.vel[i3] = 0;
      this.vel[i3 + 1] = 0;
      this.vel[i3 + 2] = 0;

      this.restDirs[i3] = ux;
      this.restDirs[i3 + 1] = uy;
      this.restDirs[i3 + 2] = uz;

      this.shrink[i] = 0;
      this.heat[i] = 0;
      this.tension[i] = 0;
      this.flapPeel[i] = 0;
      this.flapRecoilEnergy[i] = 0;
      this.surfaceMinGap[i] = 1.0;
      this.tearEdgeData[i] = 0;
      this.seamRecoilDir[i3] = 0;
      this.seamRecoilDir[i3 + 1] = 0;
      this.seamRecoilDir[i3 + 2] = 0;

      const i4 = i * 4;
      this.organicData[i4] = thickVal;
      this.organicData[i4 + 1] = 0;
      this.organicData[i4 + 2] = phaseVal;
      this.organicData[i4 + 3] = 0;
    }

    // Enforce core clearance at rest (vertices + continuous triangle surface)
    this.projectCoreSdf(false, 2);

    // Initialize structural edges & rest lengths from looseHomePos (not transient drop scale)
    this.edgeActive.fill(0);
    for (let e = 0; e < this.topo.edgeCount; e++) {
      const a = this.topo.edgeA[e];
      const b = this.topo.edgeB[e];
      this.edgeA[e] = a;
      this.edgeB[e] = b;
      const a3 = a * 3;
      const b3 = b * 3;
      const len = Math.hypot(
        this.looseHomePos[b3] - this.looseHomePos[a3],
        this.looseHomePos[b3 + 1] - this.looseHomePos[a3 + 1],
        this.looseHomePos[b3 + 2] - this.looseHomePos[a3 + 2]
      );
      this.edgeBaseRest[e] = len;
      this.edgeActive[e] = 1;
    }

    // Initialize bending pairs
    for (let bIdx = 0; bIdx < this.topo.bendCount; bIdx++) {
      const a = this.topo.bendA[bIdx];
      const b = this.topo.bendB[bIdx];
      this.bendA[bIdx] = a;
      this.bendB[bIdx] = b;
      const a3 = a * 3;
      const b3 = b * 3;
      const len = Math.hypot(
        this.looseHomePos[b3] - this.looseHomePos[a3],
        this.looseHomePos[b3 + 1] - this.looseHomePos[a3 + 1],
        this.looseHomePos[b3 + 2] - this.looseHomePos[a3 + 2]
      );
      this.bendBaseRest[bIdx] = len;
      this.bendActive[bIdx] = 1;
    }

    this.recomputeNormalsAndAttributes();
    this.indexAttr.needsUpdate = true;
    this.restDirAttr.needsUpdate = true;
    this.tearEdgeAttr.needsUpdate = true;
    this.organicAttr.needsUpdate = true;
  }

  /**
   * Projects a warm thermal beam onto the sleeve around world point `(hx, hy, hz)`,
   * heating the polymer front and monotonically advancing local shrink activation.
   */
  applyHeatBeam(
    hx: number,
    hy: number,
    hz: number,
    camX: number,
    camY: number,
    camZ: number,
    radius: number,
    power: number,
    dtSec: number
  ): void {
    const dt = Math.max(0.001, Math.min(0.05, dtSec));
    const invTwoSigmaSq = 1.0 / Math.max(1e-4, 2.0 * radius * radius * 0.52);
    const maxR2 = radius * radius * 3.0;

    for (let i = 0; i < this.activeVertexCount; i++) {
      const i3 = i * 3;
      const vx = this.pos[i3];
      const vy = this.pos[i3 + 1];
      const vz = this.pos[i3 + 2];

      const dx = vx - hx;
      const dy = vy - hy;
      const dz = vz - hz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > maxR2) continue;

      // Favor camera-facing side while letting gentle thermal wrap warm the grazing rim cleanly
      const toCamX = camX - vx;
      const toCamY = camY - vy;
      const toCamZ = camZ - vz;
      const toCamLen = Math.hypot(toCamX, toCamY, toCamZ) + 1e-6;
      const ndotv =
        (this.normals[i3] * toCamX +
          this.normals[i3 + 1] * toCamY +
          this.normals[i3 + 2] * toCamZ) /
        toCamLen;
      const facing = Math.max(0.28, Math.min(1.0, ndotv * 0.68 + 0.48));

      const gauss = Math.exp(-d2 * invTwoSigmaSq);
      const flux = power * gauss * facing;

      const nextHeat = Math.min(1.0, this.heat[i] + flux * dt * 3.6);
      this.heat[i] = nextHeat;

      // Monotonic irreversible thermoplastic rest-length shrinkage once above glass transition
      // Thinner parchment pockets draw taut slightly ahead of dense silk veins -> natural non-geometric folds
      if (nextHeat > 0.05) {
        const organicRate = 0.84 + 0.32 * (1.0 - this.wallThickness[i]);
        const shrinkDelta = (nextHeat - 0.05) * Math.min(2.6, flux) * dt * 1.45 * organicRate;
        if (shrinkDelta > 0) {
          this.shrink[i] = Math.min(1.0, this.shrink[i] + shrinkDelta);
        }
      }
    }
  }

  /**
   * Gentle physical tap/poke on an unheated loose patch (adds mild thermal warmth + local inward ripple).
   */
  applySoftPoke(px: number, py: number, pz: number, strength = 0.65): void {
    const r2Max = 0.36;
    for (let i = 0; i < this.activeVertexCount; i++) {
      const i3 = i * 3;
      const dx = this.pos[i3] - px;
      const dy = this.pos[i3 + 1] - py;
      const dz = this.pos[i3 + 2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2Max) continue;
      const w = Math.exp(-d2 * 14.0);
      // Push gently inward along local normal
      this.vel[i3] -= this.normals[i3] * strength * w;
      this.vel[i3 + 1] -= this.normals[i3 + 1] * strength * w;
      this.vel[i3 + 2] -= this.normals[i3 + 2] * strength * w;
      this.heat[i] = Math.min(1.0, this.heat[i] + 0.32 * w);
      this.shrink[i] = Math.min(1.0, this.shrink[i] + 0.14 * w);
    }
  }

  /**
   * Finds the closest active mesh vertex to world point `(px, py, pz)` and returns its index + local tautness.
   */
  findNearestVertex(px: number, py: number, pz: number): {
    vertex: number;
    distSq: number;
    localTension: number;
    localShrink: number;
  } {
    let bestV = -1;
    let bestD2 = Infinity;
    for (let i = 0; i < this.activeVertexCount; i++) {
      const i3 = i * 3;
      const dx = this.pos[i3] - px;
      const dy = this.pos[i3 + 1] - py;
      const dz = this.pos[i3 + 2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestV = i;
      }
    }

    if (bestV === -1) {
      return { vertex: -1, distSq: Infinity, localTension: 0, localShrink: 0 };
    }

    // Aggregate peak tautness within a small ~0.22 neighborhood around the clicked point
    let peakT = this.tension[bestV];
    let peakS = this.shrink[bestV];
    const vx = this.pos[bestV * 3];
    const vy = this.pos[bestV * 3 + 1];
    const vz = this.pos[bestV * 3 + 2];
    for (let i = 0; i < this.activeVertexCount; i++) {
      const i3 = i * 3;
      const dx = this.pos[i3] - vx;
      const dy = this.pos[i3 + 1] - vy;
      const dz = this.pos[i3 + 2] - vz;
      if (dx * dx + dy * dy + dz * dz < 0.05) {
        if (this.tension[i] > peakT) peakT = this.tension[i];
        if (this.shrink[i] > peakS) peakS = this.shrink[i];
      }
    }

    return {
      vertex: bestV,
      distSq: bestD2,
      localTension: peakT,
      localShrink: peakS,
    };
  }

  /**
   * Attempts to rupture a taut membrane span near `(px, py, pz)`.
   * Returns true if a seam tear was initiated.
   */
  triggerRuptureAtPoint(px: number, py: number, pz: number, force = false): boolean {
    const hit = this.findNearestVertex(px, py, pz);
    if (hit.vertex === -1) return false;
    if (!force && hit.localTension < 0.10 && hit.localShrink < 0.22) {
      return false;
    }

    const v0 = hit.vertex;
    // Compute Mode-I crack direction perpendicular to peak tensile edge at v0
    const tearDir = this.computePrincipalTearDirection(v0);
    const steps = Math.round(9 + 11 * Math.max(hit.localTension, hit.localShrink));
    const opened = this.initiatePunctureAtVertex(v0, tearDir[0], tearDir[1], tearDir[2], steps);
    if (opened) {
      // Immediately advance 2 steps on the click frame for instant snappy visual split!
      this.advanceCrackTips(0.05, 2);
      this.diffuseFlapPeelWeights();
      this.smoothTornLips(2);
      this.projectCoreSdf(false, 2);
      this.updateSilkTethers(0.016);
      this.recomputeNormalsAndAttributes();
      this.indexAttr.needsUpdate = true;
      this.restDirAttr.needsUpdate = true;
      this.tearEdgeAttr.needsUpdate = true;
      this.organicAttr.needsUpdate = true;
    }
    return opened;
  }

  /**
   * Advances the XPBD shrink-film physics, thermal conduction/cooling, SDF collision,
   * crack propagation, and GPU attribute buffers. Strictly zero allocations inside hot loop.
   */
  step(dtSec: number): void {
    const clampedDt = Math.max(0.001, Math.min(0.045, dtSec));
    this.simTime += clampedDt;

    if (this.rewrapAnimProgress < 1.0) {
      this.rewrapAnimProgress = Math.min(1.0, this.rewrapAnimProgress + clampedDt * 2.2);
    }

    // 1. Propagate active seam crack tips smoothly over ~0.3s
    if (this.crackTips.length > 0) {
      const didSplit = this.advanceCrackTips(clampedDt, 2);
      if (didSplit) {
        this.diffuseFlapPeelWeights();
        this.smoothTornLips(2);
        this.indexAttr.needsUpdate = true;
        this.restDirAttr.needsUpdate = true;
        this.tearEdgeAttr.needsUpdate = true;
      }
    }

    // Decay transient elastic recoil energy on ruptured flaps smoothly into a settled wide curled opening
    if (this.splitCount > 0) {
      const recoilDecay = Math.exp(-clampedDt * 1.25);
      for (let i = 0; i < this.activeVertexCount; i++) {
        const re = this.flapRecoilEnergy[i] * recoilDecay;
        this.flapRecoilEnergy[i] = re > 0.002 ? re : 0;
      }
    }

    // 2. Mild thermal conduction across intact edges + radiative cooling of heat glow
    this.stepThermalDiffusion(clampedDt);

    // 3. Precompute smooth vacuum-suction acceleration ONCE per frame (instead of per substep)
    const vacuumSeal = Math.max(0, 1.0 - this.tearFraction * 3.8);
    const clearance = this.config.skinClearance;
    for (let i = 0; i < this.activeVertexCount; i++) {
      const i3 = i * 3;
      const sh = this.shrink[i];
      const peel = this.flapPeel[i];
      if (sh > 0.05 && vacuumSeal > 0.01 && peel < 0.72) {
        const d = evalCoreSdfGrad(
          this.core,
          this.pos[i3],
          this.pos[i3 + 1],
          this.pos[i3 + 2],
          this.gradScratch,
          0
        );
        const gap = Math.max(0, d - clearance);
        const suction = 3.85 * sh * vacuumSeal * (1.0 - peel) * Math.min(1.0, gap * 4.2 + 0.14);
        this.suctionAccel[i3] = -this.gradScratch[0] * suction;
        this.suctionAccel[i3 + 1] = -this.gradScratch[1] * suction;
        this.suctionAccel[i3 + 2] = -this.gradScratch[2] * suction;
      } else {
        this.suctionAccel[i3] = 0;
        this.suctionAccel[i3 + 1] = 0;
        this.suctionAccel[i3 + 2] = 0;
      }
    }

    // 4. XPBD substeps
    const substeps = this.config.substeps;
    const dtSub = clampedDt / substeps;
    const dtSubSq = dtSub * dtSub;
    const invDtSub = 1.0 / dtSub;
    const velDamp = Math.exp((-clampedDt * 3.6) / substeps);
    const maxShrinkRatio = this.config.maxShrinkRatio;

    for (let s = 0; s < substeps; s++) {
      // Predict unconstrained positions
      for (let i = 0; i < this.activeVertexCount; i++) {
        const i3 = i * 3;
        const px = this.pos[i3];
        const py = this.pos[i3 + 1];
        const pz = this.pos[i3 + 2];

        this.prevPos[i3] = px;
        this.prevPos[i3 + 1] = py;
        this.prevPos[i3 + 2] = pz;

        let vx = this.vel[i3] * velDamp;
        let vy = this.vel[i3 + 1] * velDamp;
        let vz = this.vel[i3 + 2] * velDamp;

        const sh = this.shrink[i];
        const peel = this.flapPeel[i];
        let ax = this.suctionAccel[i3];
        let ay = this.suctionAccel[i3 + 1];
        let az = this.suctionAccel[i3 + 2];

        // Soft studio-air suspension tether for unheated loose cocoon bag:
        // Releases smoothly as the cocoon shrinks so the unheated perimeter glides inward naturally
        // instead of fighting the shrunk front and buckling into a pie-crust ruffle!
        const globalLooseRelax = Math.max(0.08, 1.0 - this.meanShrink * 2.1);
        const looseFactor = Math.max(0, 1.0 - sh * 2.6) * (1.0 - peel) * globalLooseRelax;
        if (looseFactor > 0.01) {
          const phase = this.simTime * 0.95 + this.restDirs[i3] * 2.4 + this.restDirs[i3 + 1] * 1.9;
          const breath = Math.sin(phase) * 0.012;
          const homeScale = 1.0 - this.meanShrink * 0.22;
          const hx = this.looseHomePos[i3] * homeScale * (1.0 + breath);
          const hy = this.looseHomePos[i3 + 1] * homeScale * (1.0 + breath);
          const hz = this.looseHomePos[i3 + 2] * homeScale * (1.0 + breath);
          const kHome = 4.2 * looseFactor;
          ax += (hx - px) * kHome;
          ay += (hy - py) * kHome;
          az += (hz - pz) * kHome;
        }

        // Bounded asymmetric organic peel-back & dried-vellum curl on torn seam flaps:
        // Variable stiffness (wallThickness) + organic wave phase (fiberPhase) make petals curl
        // unevenly like real dried mulberry parchment/bark while keeping lift strictly bounded.
        if (peel > 0.02) {
          const stiffMod = 0.72 + 0.56 * (1.0 - this.wallThickness[i]);
          const curlWave =
            0.74 +
            0.40 *
              Math.sin(
                this.fiberPhase[i] * 6.28318 +
                  this.restDirs[i3 + 1] * 3.8 +
                  this.restDirs[i3] * 2.6
              );
          const liftGap = Math.max(0, this.sdfDist[i] - clearance);
          const maxRadialGap = 0.20 + 0.06 * curlWave;
          const radialBound = Math.max(0, 1.0 - liftGap / maxRadialGap);
          const tangentBound = Math.max(0.18, 1.0 - liftGap / 0.42);
          const recoilAmp = 0.34 + 0.66 * this.flapRecoilEnergy[i];
          const baseForce = (2.0 + 2.3 * sh) * peel * recoilAmp * stiffMod;
          const rLen = Math.hypot(px, py, pz) + 1e-5;
          const outX = px / rLen;
          const outY = py / rLen;
          const outZ = pz / rLen;
          const radPull = 0.32 * curlWave * radialBound * baseForce;
          const tanPull = (1.02 + 0.30 * curlWave) * tangentBound * baseForce;
          ax += outX * radPull + this.seamRecoilDir[i3] * tanPull;
          ay += outY * radPull + this.seamRecoilDir[i3 + 1] * tanPull;
          az += outZ * radPull + this.seamRecoilDir[i3 + 2] * tanPull;
        }

        vx += ax * dtSub;
        vy += ay * dtSub;
        vz += az * dtSub;

        this.pos[i3] = px + vx * dtSub;
        this.pos[i3 + 1] = py + vy * dtSub;
        this.pos[i3 + 2] = pz + vz * dtSub;
      }

      // XPBD Structural Edge Constraints (calibrated stretch/compress asymmetry: crisp draws, zero pie-crust scallops)
      const alphaStretch = 0.00012 / dtSubSq;
      const alphaCompress = 0.00065 / dtSubSq;

      for (let iter = 0; iter < this.config.constraintIterations; iter++) {
        for (let e = 0; e < this.activeEdgeCount; e++) {
          if (this.edgeActive[e] === 0) continue;
          const a = this.edgeA[e];
          const b = this.edgeB[e];
          const a3 = a * 3;
          const b3 = b * 3;

          const dx = this.pos[b3] - this.pos[a3];
          const dy = this.pos[b3 + 1] - this.pos[a3 + 1];
          const dz = this.pos[b3 + 2] - this.pos[a3 + 2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < 1e-7) continue;

          const sEdge = 0.5 * (this.shrink[a] + this.shrink[b]);
          const targetLen = this.edgeBaseRest[e] * (1.0 - maxShrinkRatio * sEdge);
          const diff = dist - targetLen;

          const alpha = diff > 0 ? alphaStretch : alphaCompress;
          const corr = (diff / (2.0 + alpha)) / dist;

          const cx = dx * corr;
          const cy = dy * corr;
          const cz = dz * corr;

          this.pos[a3] += cx;
          this.pos[a3 + 1] += cy;
          this.pos[a3 + 2] += cz;
          this.pos[b3] -= cx;
          this.pos[b3 + 1] -= cy;
          this.pos[b3 + 2] -= cz;
        }

        // Cross-edge bending constraint: enforces smooth organic parchment curvature across the shrink transition
        // so partially-heated sleeves drape cleanly without geometric accordion pie-crust spokes!
        if (iter === 0) {
          const alphaBend = 0.0036 / dtSubSq;
          const bendCount = this.topo.bendCount;
          for (let bi = 0; bi < bendCount; bi++) {
            if (this.bendActive[bi] === 0) continue;
            const a = this.bendA[bi];
            const b = this.bendB[bi];
            const a3 = a * 3;
            const b3 = b * 3;

            const dx = this.pos[b3] - this.pos[a3];
            const dy = this.pos[b3 + 1] - this.pos[a3 + 1];
            const dz = this.pos[b3 + 2] - this.pos[a3 + 2];
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < 1e-7) continue;

            const sBend = 0.5 * (this.shrink[a] + this.shrink[b]);
            const targetLen = this.bendBaseRest[bi] * (1.0 - maxShrinkRatio * 0.52 * sBend);
            const diff = dist - targetLen;
            // Resist inward radial accordion buckling strongly while permitting clean outward ridge draping
            const localAlpha = diff < 0 ? alphaBend * 0.40 : alphaBend * (1.5 + sBend * 1.3);
            const corr = (diff / (2.0 + localAlpha)) / dist;

            const cx = dx * corr;
            const cy = dy * corr;
            const cz = dz * corr;
            this.pos[a3] += cx;
            this.pos[a3 + 1] += cy;
            this.pos[a3 + 2] += cz;
            this.pos[b3] -= cx;
            this.pos[b3 + 1] -= cy;
            this.pos[b3 + 2] -= cz;
          }
        }
      }

      // Fast vertex SDF Non-Penetration Projection per substep (continuous face pass runs cleanly at end of step)
      this.projectCoreSdf(true, 0);

      // Derive XPBD velocities with stability clamp & contact friction
      for (let i = 0; i < this.activeVertexCount; i++) {
        const i3 = i * 3;
        let vx = (this.pos[i3] - this.prevPos[i3]) * invDtSub;
        let vy = (this.pos[i3 + 1] - this.prevPos[i3 + 1]) * invDtSub;
        let vz = (this.pos[i3 + 2] - this.prevPos[i3 + 2]) * invDtSub;

        // Tangential friction when vacuum-seated tight on the core surface
        const gap = Math.max(0, this.sdfDist[i] - this.config.skinClearance);
        if (gap < 0.018) {
          const friction = 0.60;
          vx *= friction;
          vy *= friction;
          vz *= friction;
        }

        const speedSq = vx * vx + vy * vy + vz * vz;
        if (speedSq > 28.0) {
          const inv = Math.sqrt(28.0 / speedSq);
          vx *= inv;
          vy *= inv;
          vz *= inv;
        }

        this.vel[i3] = vx;
        this.vel[i3 + 1] = vy;
        this.vel[i3 + 2] = vz;
      }
    }

    // 4. Final hard continuous-surface SDF projection (vertices + active-set edge/triangle projection)
    if (this.splitCount > 0) {
      this.smoothTornLips(1);
    }
    this.projectCoreSdf(false, 4);
    this.computeStrainAndTelemetry();
    this.recomputeNormalsAndAttributes();
    if (this.splitCount > 0 || this.activeTetherCount > 0) {
      this.updateSilkTethers(clampedDt);
    }
  }

  private stepThermalDiffusion(dt: number): void {
    const cool = Math.exp(-dt * 1.75);
    for (let i = 0; i < this.activeVertexCount; i++) {
      this.heatScratch[i] = this.heat[i] * cool;
    }

    // Light diffusive ring smoothing so heat fronts read as cohesive glowing infrared waves,
    // plus monotonic warm-front shrink feathering (eliminates abrupt buckling cliff at boundary)
    const diffRate = Math.min(0.18, dt * 3.2);
    const shrinkFeather = Math.min(0.12, dt * 1.8);
    for (let e = 0; e < this.activeEdgeCount; e++) {
      if (this.edgeActive[e] === 0) continue;
      const a = this.edgeA[e];
      const b = this.edgeB[e];
      const ha = this.heat[a];
      const hb = this.heat[b];
      const delta = (hb - ha) * diffRate * 0.25;
      this.heatScratch[a] += delta;
      this.heatScratch[b] -= delta;

      if (ha > 0.08 || hb > 0.08) {
        const sa = this.shrink[a];
        const sb = this.shrink[b];
        if (sb > sa) {
          this.shrink[a] = Math.min(1.0, sa + (sb - sa) * shrinkFeather * Math.max(ha, hb));
        } else if (sa > sb) {
          this.shrink[b] = Math.min(1.0, sb + (sa - sb) * shrinkFeather * Math.max(ha, hb));
        }
      }
    }

    for (let i = 0; i < this.activeVertexCount; i++) {
      const h = this.heatScratch[i];
      const nextH = h > 0.002 ? Math.min(1.0, h) : 0;
      this.heat[i] = nextH;
      // Warm residual heat softly advances the creeping thermoplastic shrink front
      if (nextH > 0.24) {
        this.shrink[i] = Math.min(1.0, this.shrink[i] + (nextH - 0.24) * dt * 0.28);
      }
    }
  }

  /**
   * 3-phase continuous-surface analytical SDF projection with exact 1-Lipschitz culling:
   *   Phase 1: Enforce exact vertex clearance `evalCoreSdf(v) >= skinClearance`
   *   Phase 2: Exact 1-Lipschitz-culled edge-midpoint, triangle-centroid, and corner-barycentric
   *            projection so coarse triangle faces can NEVER chord-slice through sharp crystal tips
   *            or brass rings (`< 0.5 ms` overhead via `max(dA, dB) - 0.5*L >= faceClearance` skip!).
   *   Phase 3: Final fast vertex polish so every vertex stays strictly >= skinClearance.
   */
  private projectCoreSdf(applyTangentialGrip: boolean, facePasses = 0): void {
    const clearance = this.config.skinClearance;
    const faceClearance = Math.max(0.015, clearance * 0.85);
    const boundLimit = this.core.boundingRadius + clearance + 0.16;
    const boundLimitSq = boundLimit * boundLimit;
    const nearThresh = 0.16;

    // Phase 1: Vertex projection & near-core active mask
    for (let i = 0; i < this.activeVertexCount; i++) {
      const i3 = i * 3;
      const px = this.pos[i3];
      const py = this.pos[i3 + 1];
      const pz = this.pos[i3 + 2];

      const rSq = px * px + py * py + pz * pz;
      if (rSq > boundLimitSq) {
        const approx = Math.sqrt(rSq) - this.core.boundingRadius;
        this.sdfDist[i] = approx;
        this.surfaceMinGap[i] = Math.max(0, approx - clearance);
        this.nearCoreMask[i] = 0;
        continue;
      }

      const d = evalCoreSdf(this.core, px, py, pz);
      if (d < clearance) {
        evalCoreSdfGrad(this.core, px, py, pz, this.gradScratch, 0);
        const push = clearance - d;
        const gx = this.gradScratch[0];
        const gy = this.gradScratch[1];
        const gz = this.gradScratch[2];

        this.pos[i3] = px + gx * push;
        this.pos[i3 + 1] = py + gy * push;
        this.pos[i3 + 2] = pz + gz * push;

        if (applyTangentialGrip) {
          // Damp tangential slip against core surface so stretched film locks cleanly onto bevels
          const dx = this.pos[i3] - this.prevPos[i3];
          const dy = this.pos[i3 + 1] - this.prevPos[i3 + 1];
          const dz = this.pos[i3 + 2] - this.prevPos[i3 + 2];
          const vn = dx * gx + dy * gy + dz * gz;
          const tx = dx - gx * vn;
          const ty = dy - gy * vn;
          const tz = dz - gz * vn;
          this.pos[i3] -= tx * 0.35;
          this.pos[i3 + 1] -= ty * 0.35;
          this.pos[i3 + 2] -= tz * 0.35;
        }
        this.sdfDist[i] = clearance;
        this.surfaceMinGap[i] = 0;
        this.nearCoreMask[i] = 1;
      } else {
        this.sdfDist[i] = d;
        this.surfaceMinGap[i] = d - clearance;
        this.nearCoreMask[i] = d < nearThresh ? 1 : 0;
      }
    }

    if (facePasses <= 0) return;

    // Phase 2: Active-set 1-Lipschitz-culled deduplicated edge-midpoint & triangle-interior projection
    // Pass 0 filters the full mesh via exact 1-Lipschitz bounds into tiny active candidate lists;
    // Passes 1..N-1 iterate strictly over those ~80-180 touching edges/triangles (<0.4 ms total!).
    const core = this.core;
    const grad = this.gradScratch;
    const pos = this.pos;
    const sdfDist = this.sdfDist;
    const indices = this.indices;
    const nearMask = this.nearCoreMask;
    const minGap = this.surfaceMinGap;
    const candEdges = this.candEdges;
    const candTris = this.candTris;

    let candEdgeCount = 0;
    let candTriCount = 0;

    for (let pass = 0; pass < facePasses; pass++) {
      const liftScale = pass === facePasses - 1 ? 1.06 : 0.92;
      let anyLift = false;

      // 2A. Active edge midpoints
      const edgeLoopCount = pass === 0 ? this.activeEdgeCount : candEdgeCount;
      for (let ei = 0; ei < edgeLoopCount; ei++) {
        const e = pass === 0 ? ei : candEdges[ei];
        if (this.edgeActive[e] === 0) continue;
        const a = this.edgeA[e];
        const b = this.edgeB[e];
        if (pass === 0 && (nearMask[a] | nearMask[b]) === 0) continue;

        const a3 = a * 3;
        const b3 = b * 3;
        if (pass === 0) {
          const dx = pos[b3] - pos[a3];
          const dy = pos[b3 + 1] - pos[a3 + 1];
          const dz = pos[b3 + 2] - pos[a3 + 2];
          const halfLen = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
          const maxD = sdfDist[a] > sdfDist[b] ? sdfDist[a] : sdfDist[b];
          if (maxD - halfLen >= faceClearance + 0.002) continue;
        }

        const mx = 0.5 * (pos[a3] + pos[b3]);
        const my = 0.5 * (pos[a3 + 1] + pos[b3 + 1]);
        const mz = 0.5 * (pos[a3 + 2] + pos[b3 + 2]);
        const dm = evalCoreSdf(core, mx, my, mz);
        const mGap = Math.max(0, dm - faceClearance);
        if (mGap < minGap[a]) minGap[a] = mGap;
        if (mGap < minGap[b]) minGap[b] = mGap;

        if (dm < faceClearance) {
          if (pass === 0) candEdges[candEdgeCount++] = e;
          evalCoreSdfGrad(core, mx, my, mz, grad, 0);
          const push = (faceClearance - dm) * liftScale;
          const px = grad[0] * push;
          const py = grad[1] * push;
          const pz = grad[2] * push;
          pos[a3] += px;
          pos[a3 + 1] += py;
          pos[a3 + 2] += pz;
          pos[b3] += px;
          pos[b3 + 1] += py;
          pos[b3 + 2] += pz;
          nearMask[a] = 1;
          nearMask[b] = 1;
          anyLift = true;
        } else if (pass === 0 && dm < faceClearance + 0.014) {
          candEdges[candEdgeCount++] = e;
        }
      }

      // 2B. Active triangle centroids & vertex-biased corner interior samples
      const triLoopCount = pass === 0 ? this.triCount : candTriCount;
      for (let ti = 0; ti < triLoopCount; ti++) {
        const t = pass === 0 ? ti : candTris[ti];
        const t3 = t * 3;
        const i0 = indices[t3];
        const i1 = indices[t3 + 1];
        const i2 = indices[t3 + 2];
        if (pass === 0 && (nearMask[i0] | nearMask[i1] | nearMask[i2]) === 0) continue;

        const a3 = i0 * 3;
        const b3 = i1 * 3;
        const c3 = i2 * 3;
        const cx = (pos[a3] + pos[b3] + pos[c3]) * (1 / 3);
        const cy = (pos[a3 + 1] + pos[b3 + 1] + pos[c3 + 1]) * (1 / 3);
        const cz = (pos[a3 + 2] + pos[b3 + 2] + pos[c3 + 2]) * (1 / 3);

        if (pass === 0) {
          const r0 = Math.hypot(cx - pos[a3], cy - pos[a3 + 1], cz - pos[a3 + 2]);
          const r1 = Math.hypot(cx - pos[b3], cy - pos[b3 + 1], cz - pos[b3 + 2]);
          const r2 = Math.hypot(cx - pos[c3], cy - pos[c3 + 1], cz - pos[c3 + 2]);
          const cenLB = Math.max(sdfDist[i0] - r0, sdfDist[i1] - r1, sdfDist[i2] - r2);
          if (cenLB >= faceClearance + 0.04) continue;
        }

        const dc = evalCoreSdf(core, cx, cy, cz);
        const cGap = Math.max(0, dc - faceClearance);
        if (cGap < minGap[i0]) minGap[i0] = cGap;
        if (cGap < minGap[i1]) minGap[i1] = cGap;
        if (cGap < minGap[i2]) minGap[i2] = cGap;

        let triHit = false;
        if (dc < faceClearance) {
          triHit = true;
          evalCoreSdfGrad(core, cx, cy, cz, grad, 0);
          const push = (faceClearance - dc) * liftScale;
          const px = grad[0] * push;
          const py = grad[1] * push;
          const pz = grad[2] * push;
          pos[a3] += px;
          pos[a3 + 1] += py;
          pos[a3 + 2] += pz;
          pos[b3] += px;
          pos[b3 + 1] += py;
          pos[b3 + 2] += pz;
          pos[c3] += px;
          pos[c3 + 1] += py;
          pos[c3 + 2] += pz;
          nearMask[i0] = 1;
          nearMask[i1] = 1;
          nearMask[i2] = 1;
          anyLift = true;
        }

        // Test 3 vertex-biased interior barycentric points whenever triangle sits near a sharp spire/ring tip
        if (dc < faceClearance + 0.075) {
          for (let cIdx = 0; cIdx < 3; cIdx++) {
            const p0 = cIdx === 0 ? a3 : cIdx === 1 ? b3 : c3;
            const p1 = cIdx === 0 ? b3 : cIdx === 1 ? c3 : a3;
            const p2 = cIdx === 0 ? c3 : cIdx === 1 ? a3 : b3;
            const bx = 0.5 * pos[p0] + 0.25 * (pos[p1] + pos[p2]);
            const by = 0.5 * pos[p0 + 1] + 0.25 * (pos[p1 + 1] + pos[p2 + 1]);
            const bz = 0.5 * pos[p0 + 2] + 0.25 * (pos[p1 + 2] + pos[p2 + 2]);
            const db = evalCoreSdf(core, bx, by, bz);
            if (db < faceClearance) {
              triHit = true;
              evalCoreSdfGrad(core, bx, by, bz, grad, 0);
              const push = (faceClearance - db) * liftScale;
              const px = grad[0] * push;
              const py = grad[1] * push;
              const pz = grad[2] * push;
              pos[p0] += px * 1.15;
              pos[p0 + 1] += py * 1.15;
              pos[p0 + 2] += pz * 1.15;
              pos[p1] += px * 0.80;
              pos[p1 + 1] += py * 0.80;
              pos[p1 + 2] += pz * 0.80;
              pos[p2] += px * 0.80;
              pos[p2 + 1] += py * 0.80;
              pos[p2 + 2] += pz * 0.80;
              nearMask[i0] = 1;
              nearMask[i1] = 1;
              nearMask[i2] = 1;
              anyLift = true;
            }
          }
        }

        if (pass === 0 && (triHit || dc < faceClearance + 0.018)) {
          candTris[candTriCount++] = t;
        }
      }

      if (!anyLift) break;
    }

    // Phase 3: Final 2-iteration vertex polish (resolves concave ring/spoke junctions to 100% exact clearance!)
    for (let i = 0; i < this.activeVertexCount; i++) {
      if (nearMask[i] === 0) continue;
      const i3 = i * 3;
      let px = pos[i3];
      let py = pos[i3 + 1];
      let pz = pos[i3 + 2];
      let d = evalCoreSdf(core, px, py, pz);
      for (let k = 0; k < 2 && d < clearance; k++) {
        evalCoreSdfGrad(core, px, py, pz, grad, 0);
        const push = clearance - d + 0.0005;
        px += grad[0] * push;
        py += grad[1] * push;
        pz += grad[2] * push;
        d = evalCoreSdf(core, px, py, pz);
      }
      pos[i3] = px;
      pos[i3 + 1] = py;
      pos[i3 + 2] = pz;
      this.sdfDist[i] = d;
      const g = Math.max(0, d - clearance);
      if (g < minGap[i]) minGap[i] = g;
    }
  }

  private computeStrainAndTelemetry(): void {
    const maxShrinkRatio = this.config.maxShrinkRatio;
    for (let i = 0; i < this.activeVertexCount; i++) {
      this.tensionAccum[i] = 0;
      this.tensionCount[i] = 0;
    }

    for (let e = 0; e < this.activeEdgeCount; e++) {
      if (this.edgeActive[e] === 0) continue;
      const a = this.edgeA[e];
      const b = this.edgeB[e];
      const a3 = a * 3;
      const b3 = b * 3;

      const dist = Math.hypot(
        this.pos[b3] - this.pos[a3],
        this.pos[b3 + 1] - this.pos[a3 + 1],
        this.pos[b3 + 2] - this.pos[a3 + 2]
      );
      const sEdge = 0.5 * (this.shrink[a] + this.shrink[b]);
      const targetLen = Math.max(1e-5, this.edgeBaseRest[e] * (1.0 - maxShrinkRatio * sEdge));
      const strain = Math.max(0, (dist - targetLen) / targetLen);

      // Normalize visible strain: ~0.45 engineering strain over core spokes maps to tension = 1.0
      const normT = Math.min(1.0, strain * 2.35);
      this.tensionAccum[a] += normT;
      this.tensionCount[a] += 1;
      this.tensionAccum[b] += normT;
      this.tensionCount[b] += 1;
    }

    let shrinkSum = 0;
    let peakTension = 0;
    const clearance = this.config.skinClearance;

    for (let i = 0; i < this.activeVertexCount; i++) {
      const cnt = this.tensionCount[i];
      const tVal = cnt > 0 ? Math.min(1.0, this.tensionAccum[i] / cnt) : 0;
      // Smooth slight temporal filter so specular tension creases never shimmer frame-to-frame
      const smoothT = this.tension[i] + (tVal - this.tension[i]) * 0.45;
      this.tension[i] = smoothT;
      if (smoothT > peakTension) peakTension = smoothT;

      const sh = this.shrink[i];
      shrinkSum += sh;

      // Optical contact-clarity: rises sharply as vertex OR touching triangle surface hugs within ~6cm of core SDF
      const gap = Math.min(Math.max(0, this.sdfDist[i] - clearance), this.surfaceMinGap[i]);
      const rawContact = Math.exp(-gap / 0.062);
      // Unheated loose film stays frosted even near ridges; heated/tensioned contact turns crystal clear
      this.contact[i] = rawContact * Math.min(1.0, sh * 1.45 + smoothT * 0.48);
    }

    this.meanShrink = this.activeVertexCount > 0 ? shrinkSum / this.activeVertexCount : 0;
    this.maxTension = peakTension;
    this.tearFraction = Math.min(1.0, this.splitCount / 64);
  }

  private recomputeNormalsAndAttributes(): void {
    const vCount = this.activeVertexCount;
    for (let i = 0; i < vCount * 3; i++) {
      this.normals[i] = 0;
    }

    for (let t = 0; t < this.triCount; t++) {
      const i0 = this.indices[t * 3];
      const i1 = this.indices[t * 3 + 1];
      const i2 = this.indices[t * 3 + 2];

      const a3 = i0 * 3;
      const b3 = i1 * 3;
      const c3 = i2 * 3;

      const ax = this.pos[a3];
      const ay = this.pos[a3 + 1];
      const az = this.pos[a3 + 2];
      const e1x = this.pos[b3] - ax;
      const e1y = this.pos[b3 + 1] - ay;
      const e1z = this.pos[b3 + 2] - az;
      const e2x = this.pos[c3] - ax;
      const e2y = this.pos[c3 + 1] - ay;
      const e2z = this.pos[c3 + 2] - az;

      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;

      this.normals[a3] += nx;
      this.normals[a3 + 1] += ny;
      this.normals[a3 + 2] += nz;
      this.normals[b3] += nx;
      this.normals[b3 + 1] += ny;
      this.normals[b3 + 2] += nz;
      this.normals[c3] += nx;
      this.normals[c3 + 1] += ny;
      this.normals[c3 + 2] += nz;
    }

    // Compute continuous torn-lip openness & organic seam taper for fragment-level deckle feathering
    if (this.splitCount > 0) {
      for (let i = 0; i < vCount; i++) {
        this.lipOpenness[i] = 0;
        this.edgeTaper[i] = 0;
      }
      for (let i = 0; i < vCount; i++) {
        if (this.tearEdgeData[i] <= 0.5) continue;
        const twin = this.tearPairTwin[i];
        let openVal = 0.55;
        if (twin >= 0 && twin < vCount) {
          const i3 = i * 3;
          const t3 = twin * 3;
          const sep = Math.hypot(
            this.pos[i3] - this.pos[t3],
            this.pos[i3 + 1] - this.pos[t3 + 1],
            this.pos[i3 + 2] - this.pos[t3 + 2]
          );
          openVal = Math.max(0.0, Math.min(1.0, (sep - 0.012) / 0.11));
        }
        this.lipOpenness[i] = openVal;
        this.edgeTaper[i] = Math.min(1.0, 0.35 + openVal * 0.65);
      }

      // Propagate 1 ring inward so every triangle bordering an open lip carries consistent edge activation
      for (let i = 0; i < vCount; i++) {
        this.lipOpenScratch[i] = this.lipOpenness[i];
        this.edgeTaperScratch[i] = this.edgeTaper[i];
      }
      for (let e = 0; e < this.activeEdgeCount; e++) {
        if (this.edgeActive[e] === 0) continue;
        const a = this.edgeA[e];
        const b = this.edgeB[e];
        const oa = this.lipOpenness[a] * 0.90;
        const ob = this.lipOpenness[b] * 0.90;
        if (oa > this.lipOpenScratch[b]) this.lipOpenScratch[b] = oa;
        if (ob > this.lipOpenScratch[a]) this.lipOpenScratch[a] = ob;
        const ta = this.edgeTaper[a] * 0.90;
        const tb = this.edgeTaper[b] * 0.90;
        if (ta > this.edgeTaperScratch[b]) this.edgeTaperScratch[b] = ta;
        if (tb > this.edgeTaperScratch[a]) this.edgeTaperScratch[a] = tb;
      }
      for (let i = 0; i < vCount; i++) {
        this.lipOpenness[i] = this.lipOpenScratch[i];
        this.edgeTaper[i] = this.edgeTaperScratch[i];
      }
    }

    for (let i = 0; i < vCount; i++) {
      const i3 = i * 3;
      const len = Math.hypot(this.normals[i3], this.normals[i3 + 1], this.normals[i3 + 2]);
      if (len > 1e-8) {
        const inv = 1.0 / len;
        this.normals[i3] *= inv;
        this.normals[i3 + 1] *= inv;
        this.normals[i3 + 2] *= inv;
      } else {
        this.normals[i3] = this.restDirs[i3];
        this.normals[i3 + 1] = this.restDirs[i3 + 1];
        this.normals[i3 + 2] = this.restDirs[i3 + 2];
      }

      const i4 = i * 4;
      this.stateData[i4] = this.shrink[i];
      this.stateData[i4 + 1] = this.tension[i];
      this.stateData[i4 + 2] = this.heat[i];
      this.stateData[i4 + 3] = this.contact[i];

      this.organicData[i4] = this.wallThickness[i];
      this.organicData[i4 + 1] = this.lipOpenness[i];
      this.organicData[i4 + 2] = this.fiberPhase[i];
      this.organicData[i4 + 3] = this.edgeTaper[i];
    }

    this.posAttr.needsUpdate = true;
    this.normalAttr.needsUpdate = true;
    this.stateAttr.needsUpdate = true;
    this.organicAttr.needsUpdate = true;
  }

  // ==========================================================================
  // Topological Vertex Fan-Split Seam Rupture Engine
  // ==========================================================================

  private computePrincipalTearDirection(v: number): [number, number, number] {
    const v3 = v * 3;
    const vx = this.pos[v3];
    const vy = this.pos[v3 + 1];
    const vz = this.pos[v3 + 2];
    const nx = this.normals[v3];
    const ny = this.normals[v3 + 1];
    const nz = this.normals[v3 + 2];

    let bestStrain = -1;
    let pullX = 1;
    let pullY = 0;
    let pullZ = 0;

    for (let e = 0; e < this.activeEdgeCount; e++) {
      if (this.edgeActive[e] === 0) continue;
      const a = this.edgeA[e];
      const b = this.edgeB[e];
      if (a !== v && b !== v) continue;
      const o = a === v ? b : a;
      const o3 = o * 3;
      const dx = this.pos[o3] - vx;
      const dy = this.pos[o3 + 1] - vy;
      const dz = this.pos[o3 + 2] - vz;
      const dist = Math.hypot(dx, dy, dz);
      if (dist < 1e-6) continue;
      const sEdge = 0.5 * (this.shrink[a] + this.shrink[b]);
      const targetLen = Math.max(1e-5, this.edgeBaseRest[e] * (1.0 - this.config.maxShrinkRatio * sEdge));
      const strain = (dist - targetLen) / targetLen;
      if (strain > bestStrain) {
        bestStrain = strain;
        pullX = dx / dist;
        pullY = dy / dist;
        pullZ = dz / dist;
      }
    }

    // Mode-I tensile tear runs perpendicular to the peak stretched edge (n x pullDir)
    let tx = ny * pullZ - nz * pullY;
    let ty = nz * pullX - nx * pullZ;
    let tz = nx * pullY - ny * pullX;
    let len = Math.hypot(tx, ty, tz);
    if (len < 1e-5) {
      // Fallback tangent orthogonal to normal
      const refX = Math.abs(ny) < 0.9 ? 0 : 1;
      const refY = Math.abs(ny) < 0.9 ? 1 : 0;
      const refZ = 0;
      tx = ny * refZ - nz * refY;
      ty = nz * refX - nx * refZ;
      tz = nx * refY - ny * refX;
      len = Math.hypot(tx, ty, tz) + 1e-6;
    }
    return [tx / len, ty / len, tz / len];
  }

  private cloneVertex(src: number): number {
    if (this.activeVertexCount >= this.maxVertices) return -1;
    const dst = this.activeVertexCount++;
    const s3 = src * 3;
    const d3 = dst * 3;

    this.pos[d3] = this.pos[s3];
    this.pos[d3 + 1] = this.pos[s3 + 1];
    this.pos[d3 + 2] = this.pos[s3 + 2];
    this.prevPos[d3] = this.prevPos[s3];
    this.prevPos[d3 + 1] = this.prevPos[s3 + 1];
    this.prevPos[d3 + 2] = this.prevPos[s3 + 2];
    this.vel[d3] = this.vel[s3];
    this.vel[d3 + 1] = this.vel[s3 + 1];
    this.vel[d3 + 2] = this.vel[s3 + 2];
    this.looseHomePos[d3] = this.looseHomePos[s3];
    this.looseHomePos[d3 + 1] = this.looseHomePos[s3 + 1];
    this.looseHomePos[d3 + 2] = this.looseHomePos[s3 + 2];
    this.normals[d3] = this.normals[s3];
    this.normals[d3 + 1] = this.normals[s3 + 1];
    this.normals[d3 + 2] = this.normals[s3 + 2];
    this.restDirs[d3] = this.restDirs[s3];
    this.restDirs[d3 + 1] = this.restDirs[s3 + 1];
    this.restDirs[d3 + 2] = this.restDirs[s3 + 2];

    this.shrink[dst] = this.shrink[src];
    this.heat[dst] = this.heat[src];
    this.tension[dst] = this.tension[src];
    this.contact[dst] = this.contact[src];
    this.sdfDist[dst] = this.sdfDist[src];
    this.surfaceMinGap[dst] = this.surfaceMinGap[src];
    this.wallThickness[dst] = this.wallThickness[src];
    this.fiberPhase[dst] = (this.fiberPhase[src] + 0.37) % 1.0;
    this.flapPeel[dst] = 1.0;
    this.flapRecoilEnergy[dst] = 1.0;
    this.tearEdgeData[dst] = 1.0;
    this.tearPairTwin[dst] = src;
    if (this.tearPairTwin[src] === -1) {
      this.tearPairTwin[src] = dst;
    }

    return dst;
  }

  private gatherIncidentTris(v: number): void {
    this.splitIncTris.length = 0;
    for (let t = 0; t < this.triCount; t++) {
      const i0 = this.indices[t * 3];
      const i1 = this.indices[t * 3 + 1];
      const i2 = this.indices[t * 3 + 2];
      if (i0 === v) {
        // CCW triple (v, i1, i2) -> starts on spoke i1, advances CCW to spoke i2
        this.splitIncTris.push({ triIndex: t, prevVert: i1, nextVert: i2 });
      } else if (i1 === v) {
        this.splitIncTris.push({ triIndex: t, prevVert: i2, nextVert: i0 });
      } else if (i2 === v) {
        this.splitIncTris.push({ triIndex: t, prevVert: i0, nextVert: i1 });
      }
    }
  }

  /**
   * Orders incident triangles around vertex `v` into a single CCW fan chain.
   * Returns `{ ordered, isClosed }` where `ordered[k].nextVert === ordered[k+1].prevVert`.
   */
  private orderFanAroundVertex(v: number): { ordered: IncidentTriInfo[]; isClosed: boolean } | null {
    this.gatherIncidentTris(v);
    const inc = this.splitIncTris;
    if (inc.length < 2) return null;

    // Identify if there is a boundary start spoke (a prevVert that is never a nextVert)
    let startIdx = -1;
    for (let i = 0; i < inc.length; i++) {
      const candidate = inc[i].prevVert;
      let hasPredecessor = false;
      for (let j = 0; j < inc.length; j++) {
        if (inc[j].nextVert === candidate) {
          hasPredecessor = true;
          break;
        }
      }
      if (!hasPredecessor) {
        startIdx = i;
        break;
      }
    }

    const isClosed = startIdx === -1;
    const ordered: IncidentTriInfo[] = [];
    if (inc.length > this.fanUsedScratch.length) return null;
    this.fanUsedScratch.fill(0, 0, inc.length);

    let curIdx = isClosed ? 0 : startIdx;
    while (curIdx !== -1 && ordered.length < inc.length) {
      this.fanUsedScratch[curIdx] = 1;
      const cur = inc[curIdx];
      ordered.push(cur);
      let nextIdx = -1;
      for (let j = 0; j < inc.length; j++) {
        if (this.fanUsedScratch[j] === 0 && inc[j].prevVert === cur.nextVert) {
          nextIdx = j;
          break;
        }
      }
      curIdx = nextIdx;
    }

    if (ordered.length !== inc.length) return null;
    return { ordered, isClosed };
  }

  private applySplitRebinding(
    v: number,
    vNew: number,
    sectorBTris: readonly number[],
    onlyBVerts: Set<number>,
    cutSpokes: readonly number[],
    dirAX: number,
    dirAY: number,
    dirAZ: number,
    dirBX: number,
    dirBY: number,
    dirBZ: number
  ): void {
    // 1. Rewrite triangle indices on Sector B to use vNew
    for (let k = 0; k < sectorBTris.length; k++) {
      const t3 = sectorBTris[k] * 3;
      if (this.indices[t3] === v) this.indices[t3] = vNew;
      if (this.indices[t3 + 1] === v) this.indices[t3 + 1] = vNew;
      if (this.indices[t3 + 2] === v) this.indices[t3 + 2] = vNew;
    }

    // 2. Rebind structural edges touching v (move Sector B edges to vNew, duplicate cut-spoke edges)
    const curEdgeCount = this.activeEdgeCount;
    for (let e = 0; e < curEdgeCount; e++) {
      if (this.edgeActive[e] === 0) continue;
      const a = this.edgeA[e];
      const b = this.edgeB[e];
      if (a !== v && b !== v) continue;
      const other = a === v ? b : a;

      let isCutSpoke = false;
      for (let c = 0; c < cutSpokes.length; c++) {
        if (cutSpokes[c] === other) {
          isCutSpoke = true;
          break;
        }
      }

      if (isCutSpoke) {
        // Keep original edge on Side A (v, other) and allocate twin edge on Side B (vNew, other)
        if (this.activeEdgeCount < this.maxEdges) {
          const twin = this.activeEdgeCount++;
          this.edgeA[twin] = vNew;
          this.edgeB[twin] = other;
          this.edgeBaseRest[twin] = this.edgeBaseRest[e];
          this.edgeActive[twin] = 1;
        }
      } else if (onlyBVerts.has(other)) {
        if (a === v) this.edgeA[e] = vNew;
        else this.edgeB[e] = vNew;
      }
    }

    // 3. Sever bending constraints crossing the torn seam vertex so flaps curl freely
    const bendCount = this.topo.bendCount;
    for (let bi = 0; bi < bendCount; bi++) {
      if (this.bendActive[bi] === 0) continue;
      const ba = this.bendA[bi];
      const bb = this.bendB[bi];
      const sa = this.topo.bendSharedA[bi];
      const sb = this.topo.bendSharedB[bi];
      if (ba === v || bb === v || sa === v || sb === v) {
        this.bendActive[bi] = 0;
      }
    }

    // 4. Mark torn lip attributes and inject immediate elastic snap-back impulse
    this.tearEdgeData[v] = 1.0;
    this.tearEdgeData[vNew] = 1.0;
    this.flapPeel[v] = 1.0;
    this.flapPeel[vNew] = 1.0;
    this.flapRecoilEnergy[v] = 1.0;
    this.flapRecoilEnergy[vNew] = 1.0;

    const v3 = v * 3;
    const n3 = vNew * 3;
    this.seamRecoilDir[v3] = dirAX;
    this.seamRecoilDir[v3 + 1] = dirAY;
    this.seamRecoilDir[v3 + 2] = dirAZ;
    this.seamRecoilDir[n3] = dirBX;
    this.seamRecoilDir[n3 + 1] = dirBY;
    this.seamRecoilDir[n3 + 2] = dirBZ;

    const snapSpeed = 1.15 + 1.85 * Math.max(this.tension[v], this.shrink[v] * 0.65);
    const nx = this.normals[v3];
    const ny = this.normals[v3 + 1];
    const nz = this.normals[v3 + 2];

    this.vel[v3] += (dirAX * 1.05 + nx * 0.30) * snapSpeed;
    this.vel[v3 + 1] += (dirAY * 1.05 + ny * 0.30) * snapSpeed;
    this.vel[v3 + 2] += (dirAZ * 1.05 + nz * 0.30) * snapSpeed;

    this.vel[n3] += (dirBX * 1.05 + nx * 0.30) * snapSpeed;
    this.vel[n3 + 1] += (dirBY * 1.05 + ny * 0.30) * snapSpeed;
    this.vel[n3 + 2] += (dirBZ * 1.05 + nz * 0.30) * snapSpeed;

    // Immediate micro-separation so zero-frame puncture reads crisply
    this.pos[v3] += (dirAX * 0.012 + nx * 0.008);
    this.pos[v3 + 1] += (dirAY * 0.012 + ny * 0.008);
    this.pos[v3 + 2] += (dirAZ * 0.012 + nz * 0.008);

    this.pos[n3] += (dirBX * 0.012 + nx * 0.008);
    this.pos[n3 + 1] += (dirBY * 0.012 + ny * 0.008);
    this.pos[n3 + 2] += (dirBZ * 0.012 + nz * 0.008);

    // Spawn delicate gossamer stretched silk tether filaments across selected freshly split lips
    // (slightly irregular diagonal pairing so fibers form an organic web, never a parallel harp!)
    if (hash3Int(v, vNew, this.splitCount + 3) > 0.34) {
      this.spawnSilkTether(v, vNew, 0.24 + 0.26 * hash3Int(v, vNew, this.splitCount + 7));
    }
    if (cutSpokes.length > 0 && hash3Int(vNew, cutSpokes[0], 13) > 0.68) {
      this.spawnSilkTether(
        vNew,
        cutSpokes[0],
        0.22 + 0.22 * hash3Int(cutSpokes[0], v, this.splitCount + 29)
      );
    }

    this.splitCount += 1;
    this.tearFraction = Math.min(1.0, this.splitCount / 64);
  }

  private spawnSilkTether(vertA: number, vertB: number, maxSpan: number): void {
    const slot = this.tetherCursor % MAX_TETHERS;
    this.tetherCursor++;
    this.tetherA[slot] = vertA;
    this.tetherB[slot] = vertB;
    this.tetherMaxSpan[slot] = maxSpan;
    this.tetherBreakProgress[slot] = 0;
    this.tetherPhase[slot] = hash3Int(vertA, vertB, slot + 101);
    this.tetherActive[slot] = 1;
  }

  /**
   * Updates pre-allocated stretched silk fiber ribbons spanning freshly torn seam gaps.
   * Filaments bow gently clear of the inner core SDF, thin under tensile strain, and snap
   * cleanly with brief elastic recoil once stretched beyond their breaking length.
   */
  private updateSilkTethers(dtSec: number): void {
    const clearance = this.config.skinClearance + 0.006;
    const pos = this.pos;
    const normals = this.normals;
    const tPos = this.tetherPos;
    const tNorm = this.tetherNorm;
    const tStrain = this.tetherStrain;
    let aliveCount = 0;
    let highestSlot = -1;

    for (let s = 0; s < MAX_TETHERS; s++) {
      const state = this.tetherActive[s];
      if (state === 0) continue;

      const a = this.tetherA[s];
      const b = this.tetherB[s];
      if (a < 0 || b < 0 || a >= this.activeVertexCount || b >= this.activeVertexCount) {
        this.tetherActive[s] = 0;
        continue;
      }

      const a3 = a * 3;
      const b3 = b * 3;
      // Anchor each silk fiber slightly inward along the torn flap's recoil direction so ends
      // emerge seamlessly out of the alpha-feathered deckle parchment (zero detached floating tabs!)
      const ax = pos[a3] + this.seamRecoilDir[a3] * 0.036;
      const ay = pos[a3 + 1] + this.seamRecoilDir[a3 + 1] * 0.036;
      const az = pos[a3 + 2] + this.seamRecoilDir[a3 + 2] * 0.036;
      let bx = pos[b3] + this.seamRecoilDir[b3] * 0.036;
      let by = pos[b3 + 1] + this.seamRecoilDir[b3 + 1] * 0.036;
      let bz = pos[b3 + 2] + this.seamRecoilDir[b3 + 2] * 0.036;

      let dx = bx - ax;
      let dy = by - ay;
      let dz = bz - az;
      let span = Math.hypot(dx, dy, dz);
      const maxSpan = this.tetherMaxSpan[s];

      let visAlpha = 0;
      let strain = 0;

      if (state === 1) {
        if (span > maxSpan) {
          this.tetherActive[s] = 2;
          this.tetherBreakProgress[s] = 0.01;
        }
        strain = Math.max(0, Math.min(1, span / Math.max(1e-4, maxSpan)));
        // Only reveal once lips actually part slightly (> 42 mm incl. inward tuck) and fade smoothly at break
        const openFade = Math.max(0, Math.min(1, (span - 0.042) / 0.055));
        const preSnapFade = 1.0 - Math.max(0, (strain - 0.82) / 0.18) * 0.55;
        visAlpha = openFade * preSnapFade;
      }

      if (this.tetherActive[s] === 2) {
        const nextP = this.tetherBreakProgress[s] + dtSec * 6.5;
        if (nextP >= 1.0) {
          this.tetherActive[s] = 0;
          const vBase = s * TETHER_VERTS_PER_STRAND;
          for (let k = 0; k < TETHER_VERTS_PER_STRAND; k++) {
            const p3 = (vBase + k) * 3;
            const s2 = (vBase + k) * 2;
            tPos[p3] = 0;
            tPos[p3 + 1] = 0;
            tPos[p3 + 2] = 0;
            tStrain[s2] = 0;
            tStrain[s2 + 1] = 0;
          }
          continue;
        }
        this.tetherBreakProgress[s] = nextP;
        // Free snapped wisp recoils back cleanly into the deckled lip while vanishing swiftly
        const recoilFrac = 1.0 - nextP * 0.82;
        bx = ax + dx * recoilFrac;
        by = ay + dy * recoilFrac;
        bz = az + dz * recoilFrac;
        dx = bx - ax;
        dy = by - ay;
        dz = bz - az;
        span = Math.hypot(dx, dy, dz);
        strain = 1.0;
        visAlpha = (1.0 - nextP) * (1.0 - nextP) * 0.52;
      }

      aliveCount++;
      highestSlot = s;

      // Average surface normal along the span for gentle outward bow & ribbon framing
      let nx = normals[a3] + normals[b3];
      let ny = normals[a3 + 1] + normals[b3 + 1];
      let nz = normals[a3 + 2] + normals[b3 + 2];
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen > 1e-5) {
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;
      } else {
        nx = 0;
        ny = 1;
        nz = 0;
      }

      // Compute ribbon side vector perpendicular to strand tangent & outward normal
      const invSpan = span > 1e-5 ? 1.0 / span : 0;
      const tx = dx * invSpan;
      const ty = dy * invSpan;
      const tz = dz * invSpan;
      let sideX = ty * nz - tz * ny;
      let sideY = tz * nx - tx * nz;
      let sideZ = tx * ny - ty * nx;
      let sideLen = Math.hypot(sideX, sideY, sideZ);
      if (sideLen < 1e-5) {
        sideX = 1;
        sideY = 0;
        sideZ = 0;
        sideLen = 1;
      }
      sideX /= sideLen;
      sideY /= sideLen;
      sideZ /= sideLen;

      // Subtle organic catenary / curl bow + guaranteed clearance above core SDF
      const phase = this.tetherPhase[s];
      let bowOut = 0.012 + 0.022 * (1.0 - strain) * (0.5 + 0.5 * Math.sin(phase * 12.56));
      const midX = 0.5 * (ax + bx) + nx * bowOut;
      const midY = 0.5 * (ay + by) + ny * bowOut;
      const midZ = 0.5 * (az + bz) + nz * bowOut;
      const dMid = evalCoreSdf(this.core, midX, midY, midZ);
      if (dMid < clearance) {
        bowOut += clearance - dMid + 0.008;
      }
      const lateralWave = (phase - 0.5) * 0.022 * (1.0 - strain * 0.7);
      // Truly gossamer hair-thin spun silk strand width (0.9..1.7 mm half-width, thinning under stretch!)
      const snapThin = this.tetherActive[s] === 2 ? 0.55 : 1.0;
      const baseHalfW = (0.0009 + phase * 0.0008) * (1.0 - 0.38 * strain) * snapThin;

      const vBase = s * TETHER_VERTS_PER_STRAND;
      for (let k = 0; k <= TETHER_SEGS; k++) {
        const u = k / TETHER_SEGS;
        const arch = 4.0 * u * (1.0 - u);
        // Gossamer filament profile: gently flared fibrous root -> ultra-fine taut silk thread at center
        const taperW = baseHalfW * (0.48 + 0.72 * (1.0 - arch));
        const px = ax + dx * u + nx * (bowOut * arch) + sideX * (lateralWave * arch);
        const py = ay + dy * u + ny * (bowOut * arch) + sideY * (lateralWave * arch);
        const pz = az + dz * u + nz * (bowOut * arch) + sideZ * (lateralWave * arch);

        const v0 = vBase + k * 2;
        const v1 = v0 + 1;
        const p0 = v0 * 3;
        const p1 = v1 * 3;
        tPos[p0] = px - sideX * taperW;
        tPos[p0 + 1] = py - sideY * taperW;
        tPos[p0 + 2] = pz - sideZ * taperW;
        tPos[p1] = px + sideX * taperW;
        tPos[p1 + 1] = py + sideY * taperW;
        tPos[p1 + 2] = pz + sideZ * taperW;

        tNorm[p0] = nx;
        tNorm[p0 + 1] = ny;
        tNorm[p0 + 2] = nz;
        tNorm[p1] = nx;
        tNorm[p1 + 1] = ny;
        tNorm[p1 + 2] = nz;

        const st0 = v0 * 2;
        const st1 = v1 * 2;
        tStrain[st0] = strain;
        tStrain[st0 + 1] = visAlpha;
        tStrain[st1] = strain;
        tStrain[st1 + 1] = visAlpha;
      }
    }

    this.activeTetherCount = aliveCount;
    this.tetherGeometry.setDrawRange(0, (highestSlot + 1) * TETHER_INDICES_PER_STRAND);
    this.tetherPosAttr.needsUpdate = true;
    this.tetherNormAttr.needsUpdate = true;
    this.tetherStrainAttr.needsUpdate = true;
  }

  private initiatePunctureAtVertex(
    v0: number,
    dirX: number,
    dirY: number,
    dirZ: number,
    steps: number
  ): boolean {
    const fan = this.orderFanAroundVertex(v0);
    if (!fan || fan.ordered.length < 3) return false;

    const { ordered, isClosed } = fan;
    const v3 = v0 * 3;
    const vx = this.pos[v3];
    const vy = this.pos[v3 + 1];
    const vz = this.pos[v3 + 2];

    if (!isClosed) {
      // Already on an existing open boundary -> propagate tear inward along best interior spoke!
      const startSpoke = ordered[0].prevVert;
      const endSpoke = ordered[ordered.length - 1].nextVert;
      return this.splitOpenFanTip(
        v0,
        startSpoke,
        endSpoke,
        dirX,
        dirY,
        dirZ,
        steps,
        this.fiberPhase[v0] * 6.28318,
        0.55,
        1
      );
    }

    const deg = ordered.length;
    let kA = 0;
    let bestPosDot = -Infinity;
    let kB = Math.floor(deg / 2);
    let bestNegDot = -Infinity;

    for (let k = 0; k < deg; k++) {
      const w = ordered[k].prevVert;
      const w3 = w * 3;
      const dx = this.pos[w3] - vx;
      const dy = this.pos[w3 + 1] - vy;
      const dz = this.pos[w3 + 2] - vz;
      const len = Math.hypot(dx, dy, dz) + 1e-6;
      const dPos = (dx * dirX + dy * dirY + dz * dirZ) / len;
      if (dPos > bestPosDot) {
        bestPosDot = dPos;
        kA = k;
      }
    }

    for (let k = 0; k < deg; k++) {
      if (k === kA) continue;
      const w = ordered[k].prevVert;
      const w3 = w * 3;
      const dx = this.pos[w3] - vx;
      const dy = this.pos[w3 + 1] - vy;
      const dz = this.pos[w3 + 2] - vz;
      const len = Math.hypot(dx, dy, dz) + 1e-6;
      const dNeg = -(dx * dirX + dy * dirY + dz * dirZ) / len;
      if (dNeg > bestNegDot) {
        bestNegDot = dNeg;
        kB = k;
      }
    }

    if (kA === kB) return false;
    const vNew = this.cloneVertex(v0);
    if (vNew === -1) return false;

    const wA = ordered[kA].prevVert;
    const wB = ordered[kB].prevVert;

    this.splitSectorBTris.length = 0;
    this.splitOnlyBVerts.clear();

    // Sector A: CCW from kA to kB (exclusive of kB)
    let axSum = 0;
    let aySum = 0;
    let azSum = 0;
    for (let k = kA; k !== kB; k = (k + 1) % deg) {
      const w = ordered[k].nextVert;
      if (w !== wA && w !== wB) {
        axSum += this.pos[w * 3] - vx;
        aySum += this.pos[w * 3 + 1] - vy;
        azSum += this.pos[w * 3 + 2] - vz;
      }
    }

    // Sector B: CCW from kB to kA (exclusive of kA)
    let bxSum = 0;
    let bySum = 0;
    let bzSum = 0;
    for (let k = kB; k !== kA; k = (k + 1) % deg) {
      this.splitSectorBTris.push(ordered[k].triIndex);
      const w = ordered[k].nextVert;
      if (w !== wA && w !== wB) {
        this.splitOnlyBVerts.add(w);
        bxSum += this.pos[w * 3] - vx;
        bySum += this.pos[w * 3 + 1] - vy;
        bzSum += this.pos[w * 3 + 2] - vz;
      }
    }

    // Fallback perpendicular recoil if a sector spans only 1 triangle
    let aLen = Math.hypot(axSum, aySum, azSum);
    let bLen = Math.hypot(bxSum, bySum, bzSum);
    if (aLen < 1e-4 || bLen < 1e-4) {
      const nx = this.normals[v3];
      const ny = this.normals[v3 + 1];
      const nz = this.normals[v3 + 2];
      const px = ny * dirZ - nz * dirY;
      const py = nz * dirX - nx * dirZ;
      const pz = nx * dirY - ny * dirX;
      const pLen = Math.hypot(px, py, pz) + 1e-6;
      if (aLen < 1e-4) {
        axSum = px / pLen;
        aySum = py / pLen;
        azSum = pz / pLen;
        aLen = 1;
      }
      if (bLen < 1e-4) {
        bxSum = -px / pLen;
        bySum = -py / pLen;
        bzSum = -pz / pLen;
        bLen = 1;
      }
    }

    this.applySplitRebinding(
      v0,
      vNew,
      this.splitSectorBTris,
      this.splitOnlyBVerts,
      [wA, wB],
      axSum / aLen,
      aySum / aLen,
      azSum / aLen,
      bxSum / bLen,
      bySum / bLen,
      bzSum / bLen
    );

    const halfSteps = Math.max(3, Math.floor(steps * 0.5));
    const seedPhase = this.fiberPhase[v0] * 6.28318;
    this.crackTips.push({
      tipVertex: wA,
      prevA: v0,
      prevB: vNew,
      dirX,
      dirY,
      dirZ,
      remainingSteps: halfSteps,
      cooldownSec: 0,
      meanderPhase: seedPhase,
      meanderRate: 0.58,
      branchBudget: 1,
    });
    this.crackTips.push({
      tipVertex: wB,
      prevA: v0,
      prevB: vNew,
      dirX: -dirX,
      dirY: -dirY,
      dirZ: -dirZ,
      remainingSteps: halfSteps,
      cooldownSec: 0,
      meanderPhase: seedPhase + 2.35,
      meanderRate: -0.64,
      branchBudget: 1,
    });

    return true;
  }

  private splitOpenFanTip(
    w: number,
    prevA: number,
    prevB: number,
    dirX: number,
    dirY: number,
    dirZ: number,
    remainingSteps: number,
    meanderPhase: number,
    meanderRate: number,
    branchBudget: number
  ): boolean {
    const fan = this.orderFanAroundVertex(w);
    if (!fan || fan.isClosed || fan.ordered.length < 2) return false;

    const { ordered } = fan;
    const startSpoke = ordered[0].prevVert;
    const endSpoke = ordered[ordered.length - 1].nextVert;

    // Confirm this open fan matches the incoming seam boundary (or any open edge if user clicked a boundary)
    const matchesBoundary =
      (startSpoke === prevA && endSpoke === prevB) ||
      (startSpoke === prevB && endSpoke === prevA);
    if (!matchesBoundary) return false;

    const w3 = w * 3;
    const wx = this.pos[w3];
    const wy = this.pos[w3 + 1];
    const wz = this.pos[w3 + 2];

    // Organic curved meander: rotate propagation target in local surface tangent plane using
    // smooth domain-warped fiber-grain flow + slow curvature meander (eliminates straight chords!)
    const nx = this.normals[w3];
    const ny = this.normals[w3 + 1];
    const nz = this.normals[w3 + 2];
    let perpX = ny * dirZ - nz * dirY;
    let perpY = nz * dirX - nx * dirZ;
    let perpZ = nx * dirY - ny * dirX;
    const pLen = Math.hypot(perpX, perpY, perpZ) + 1e-6;
    perpX /= pLen;
    perpY /= pLen;
    perpZ /= pLen;

    const grainFlow =
      crinkleFbm3D(
        this.restDirs[w3] * 3.8 + 4.2,
        this.restDirs[w3 + 1] * 3.8 + 1.9,
        this.restDirs[w3 + 2] * 3.8 + 7.4
      ) - 0.5;
    const meanderAngle = grainFlow * 0.72 + Math.sin(meanderPhase) * 0.28;
    const cosM = Math.cos(meanderAngle);
    const sinM = Math.sin(meanderAngle);
    const steerX = dirX * cosM + perpX * sinM;
    const steerY = dirY * cosM + perpY * sinM;
    const steerZ = dirZ * cosM + perpZ * sinM;

    // Interior spokes are ordered[k].prevVert for k = 1 .. ordered.length - 1
    let bestSplitIdx = -1;
    let bestScore = -Infinity;
    let bestSpokeDirX = steerX;
    let bestSpokeDirY = steerY;
    let bestSpokeDirZ = steerZ;

    for (let k = 1; k < ordered.length; k++) {
      const u = ordered[k].prevVert;
      const u3 = u * 3;
      const dx = this.pos[u3] - wx;
      const dy = this.pos[u3 + 1] - wy;
      const dz = this.pos[u3 + 2] - wz;
      const len = Math.hypot(dx, dy, dz) + 1e-6;
      const ux = dx / len;
      const uy = dy / len;
      const uz = dz / len;
      const align = ux * steerX + uy * steerY + uz * steerZ;
      // Steer cracks through thinner parchment pockets & high-tension spans around dense silk veins
      const thinWeakness = 1.0 - this.wallThickness[u];
      const score = align + this.tension[u] * 0.28 + thinWeakness * 0.36;
      if (score > bestScore) {
        bestScore = score;
        bestSplitIdx = k;
        bestSpokeDirX = ux;
        bestSpokeDirY = uy;
        bestSpokeDirZ = uz;
      }
    }

    if (bestSplitIdx === -1) return false;
    const wNew = this.cloneVertex(w);
    if (wNew === -1) return false;

    const nextSpoke = ordered[bestSplitIdx].prevVert;
    this.splitSectorBTris.length = 0;
    this.splitOnlyBVerts.clear();

    // First half (0 .. bestSplitIdx - 1) stays on w; Second half (bestSplitIdx .. end) moves to wNew
    let axSum = 0;
    let aySum = 0;
    let azSum = 0;
    for (let k = 0; k < bestSplitIdx; k++) {
      const sp = ordered[k].prevVert;
      axSum += this.pos[sp * 3] - wx;
      aySum += this.pos[sp * 3 + 1] - wy;
      azSum += this.pos[sp * 3 + 2] - wz;
    }

    let bxSum = 0;
    let bySum = 0;
    let bzSum = 0;
    for (let k = bestSplitIdx; k < ordered.length; k++) {
      this.splitSectorBTris.push(ordered[k].triIndex);
      const sp = ordered[k].nextVert;
      if (sp !== nextSpoke) {
        this.splitOnlyBVerts.add(sp);
        bxSum += this.pos[sp * 3] - wx;
        bySum += this.pos[sp * 3 + 1] - wy;
        bzSum += this.pos[sp * 3 + 2] - wz;
      }
    }

    const aLen = Math.hypot(axSum, aySum, azSum) + 1e-6;
    const bLen = Math.hypot(bxSum, bySum, bzSum) + 1e-6;

    this.applySplitRebinding(
      w,
      wNew,
      this.splitSectorBTris,
      this.splitOnlyBVerts,
      [nextSpoke],
      axSum / aLen,
      aySum / aLen,
      azSum / aLen,
      bxSum / bLen,
      bySum / bLen,
      bzSum / bLen
    );

    if (remainingSteps > 1) {
      // Smoothly blend propagation direction along the organically curved surface tangent
      const ndx = steerX * 0.48 + bestSpokeDirX * 0.52;
      const ndy = steerY * 0.48 + bestSpokeDirY * 0.52;
      const ndz = steerZ * 0.48 + bestSpokeDirZ * 0.52;
      const nLen = Math.hypot(ndx, ndy, ndz) + 1e-6;

      this.crackTips.push({
        tipVertex: nextSpoke,
        prevA: w,
        prevB: wNew,
        dirX: ndx / nLen,
        dirY: ndy / nLen,
        dirZ: ndz / nLen,
        remainingSteps: remainingSteps - 1,
        cooldownSec: 0.018,
        meanderPhase: meanderPhase + meanderRate,
        meanderRate,
        branchBudget,
      });

      // Natural secondary Y-fork micro-branching along high-tension mid-seam tears:
      // Angles ~55 deg into whichever side still has >= 2 open-fan triangles to create an organic split fork!
      if (
        branchBudget > 0 &&
        remainingSteps >= 3 &&
        remainingSteps <= 5 &&
        this.crackTips.length < 6 &&
        this.tension[w] > 0.18
      ) {
        const sectorATris = bestSplitIdx;
        const sectorBTris = ordered.length - bestSplitIdx;
        if (sectorBTris >= 2) {
          const branchX = bestSpokeDirX * 0.52 + (bxSum / bLen) * 0.85;
          const branchY = bestSpokeDirY * 0.52 + (bySum / bLen) * 0.85;
          const branchZ = bestSpokeDirZ * 0.52 + (bzSum / bLen) * 0.85;
          const bNorm = Math.hypot(branchX, branchY, branchZ) + 1e-6;
          this.crackTips.push({
            tipVertex: wNew,
            prevA: nextSpoke,
            prevB: endSpoke,
            dirX: branchX / bNorm,
            dirY: branchY / bNorm,
            dirZ: branchZ / bNorm,
            remainingSteps: 2,
            cooldownSec: 0.042,
            meanderPhase: meanderPhase + 1.4,
            meanderRate: -meanderRate * 0.8,
            branchBudget: 0,
          });
        } else if (sectorATris >= 2) {
          const branchX = bestSpokeDirX * 0.52 + (axSum / aLen) * 0.85;
          const branchY = bestSpokeDirY * 0.52 + (aySum / aLen) * 0.85;
          const branchZ = bestSpokeDirZ * 0.52 + (azSum / aLen) * 0.85;
          const aNorm = Math.hypot(branchX, branchY, branchZ) + 1e-6;
          this.crackTips.push({
            tipVertex: w,
            prevA: startSpoke,
            prevB: nextSpoke,
            dirX: branchX / aNorm,
            dirY: branchY / aNorm,
            dirZ: branchZ / aNorm,
            remainingSteps: 2,
            cooldownSec: 0.042,
            meanderPhase: meanderPhase - 1.4,
            meanderRate: -meanderRate * 0.8,
            branchBudget: 0,
          });
        }
      }
    }

    return true;
  }

  private advanceCrackTips(dtSec: number, maxSplitsThisTick: number): boolean {
    let splitsDone = 0;
    let i = 0;
    while (i < this.crackTips.length && splitsDone < maxSplitsThisTick) {
      const tip = this.crackTips[i];
      tip.cooldownSec -= dtSec;
      if (tip.cooldownSec > 0) {
        i++;
        continue;
      }
      // Remove current tip; splitOpenFanTip will enqueue the next vertex if steps remain
      this.crackTips.splice(i, 1);
      if (tip.remainingSteps <= 0) continue;

      const splitOk = this.splitOpenFanTip(
        tip.tipVertex,
        tip.prevA,
        tip.prevB,
        tip.dirX,
        tip.dirY,
        tip.dirZ,
        tip.remainingSteps,
        tip.meanderPhase,
        tip.meanderRate,
        tip.branchBudget
      );
      if (splitOk) {
        splitsDone++;
      }
    }
    return splitsDone > 0;
  }

  /**
   * Diffuses flap peel-back influence & transient recoil energy 3 rings inward from torn seam lips
   * so peeled flaps curl smoothly into graceful parchment petals.
   */
  private diffuseFlapPeelWeights(): void {
    const vCount = this.activeVertexCount;
    for (let i = 0; i < vCount; i++) {
      this.flapPeel[i] = this.tearEdgeData[i] > 0.5 ? 1.0 : 0.0;
    }

    for (let pass = 0; pass < 4; pass++) {
      for (let i = 0; i < vCount; i++) {
        this.flapPeelScratch[i] = this.flapPeel[i];
        this.flapRecoilScratch[i] = this.flapRecoilEnergy[i];
      }
      for (let e = 0; e < this.activeEdgeCount; e++) {
        if (this.edgeActive[e] === 0) continue;
        const a = this.edgeA[e];
        const b = this.edgeB[e];
        const pa = this.flapPeel[a] * 0.74;
        const pb = this.flapPeel[b] * 0.74;
        const ra = this.flapRecoilEnergy[a] * 0.82;
        const rb = this.flapRecoilEnergy[b] * 0.82;
        if (ra > this.flapRecoilScratch[b]) this.flapRecoilScratch[b] = ra;
        if (rb > this.flapRecoilScratch[a]) this.flapRecoilScratch[a] = rb;

        if (pa > this.flapPeelScratch[b]) {
          this.flapPeelScratch[b] = pa;
          const a3 = a * 3;
          const b3 = b * 3;
          this.seamRecoilDir[b3] = this.seamRecoilDir[a3];
          this.seamRecoilDir[b3 + 1] = this.seamRecoilDir[a3 + 1];
          this.seamRecoilDir[b3 + 2] = this.seamRecoilDir[a3 + 2];
        }
        if (pb > this.flapPeelScratch[a]) {
          this.flapPeelScratch[a] = pb;
          const a3 = a * 3;
          const b3 = b * 3;
          this.seamRecoilDir[a3] = this.seamRecoilDir[b3];
          this.seamRecoilDir[a3 + 1] = this.seamRecoilDir[b3 + 1];
          this.seamRecoilDir[a3 + 2] = this.seamRecoilDir[b3 + 2];
        }
      }
      for (let i = 0; i < vCount; i++) {
        this.flapPeel[i] = this.flapPeelScratch[i];
        this.flapRecoilEnergy[i] = this.flapRecoilScratch[i];
      }
    }
  }

  /**
   * Gentle tangential Laplacian fairing strictly along torn seam boundary chains (`tearEdgeData > 0.5`).
   * Turns coarse icosphere sawtooth vertices into clean curved Mode-I parchment tear arcs without
   * moving interior vertices or violating core clearance.
   */
  private smoothTornLips(iterations: number): void {
    const vCount = this.activeVertexCount;
    for (let it = 0; it < iterations; it++) {
      for (let i = 0; i < vCount; i++) {
        this.lipSmoothCount[i] = 0;
        const i3 = i * 3;
        this.lipSmoothAccum[i3] = 0;
        this.lipSmoothAccum[i3 + 1] = 0;
        this.lipSmoothAccum[i3 + 2] = 0;
      }

      for (let e = 0; e < this.activeEdgeCount; e++) {
        if (this.edgeActive[e] === 0) continue;
        const a = this.edgeA[e];
        const b = this.edgeB[e];
        if (this.tearEdgeData[a] > 0.5 && this.tearEdgeData[b] > 0.5) {
          const a3 = a * 3;
          const b3 = b * 3;
          this.lipSmoothAccum[a3] += this.pos[b3];
          this.lipSmoothAccum[a3 + 1] += this.pos[b3 + 1];
          this.lipSmoothAccum[a3 + 2] += this.pos[b3 + 2];
          this.lipSmoothCount[a] += 1;

          this.lipSmoothAccum[b3] += this.pos[a3];
          this.lipSmoothAccum[b3 + 1] += this.pos[a3 + 1];
          this.lipSmoothAccum[b3 + 2] += this.pos[a3 + 2];
          this.lipSmoothCount[b] += 1;
        }
      }

      const blend = 0.22;
      for (let i = 0; i < vCount; i++) {
        const cnt = this.lipSmoothCount[i];
        if (cnt < 2) continue;
        const inv = 1.0 / cnt;
        const i3 = i * 3;
        const avgX = this.lipSmoothAccum[i3] * inv;
        const avgY = this.lipSmoothAccum[i3 + 1] * inv;
        const avgZ = this.lipSmoothAccum[i3 + 2] * inv;
        this.pos[i3] += (avgX - this.pos[i3]) * blend;
        this.pos[i3 + 1] += (avgY - this.pos[i3 + 1]) * blend;
        this.pos[i3 + 2] += (avgZ - this.pos[i3 + 2]) * blend;
      }
    }
  }

  dispose(): void {
    this.geometry.dispose();
    this.tetherGeometry.dispose();
  }
}
