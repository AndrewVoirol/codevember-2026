import * as THREE from 'three/webgpu';
import {
  CUBE_FACES,
  WET_ATLAS_FACE_SIZE,
  type CubeFaceId,
  type LabyrinthEdge,
  type LabyrinthNode,
  type LabyrinthTopology,
  type StoneBridgeSpec,
} from './types';

/**
 * Day 13 — Labyrinth
 * Procedural Watertight Carved Stone Monolith, Split Vault Octants, Inlays & Mercury Bead Geometry
 *
 * Uses Sharp-Feature Hermite Dual Contouring (regularized QEF) on an exact
 * 1-Lipschitz 3D Signed Distance Field built directly from `LabyrinthTopology`:
 * - 45-degree chamfered cube edges and 3-way mitered corners (`H = 0.88, R = 0.095`)
 * - Crisp chiseled `tier = 0` corridor trenches, smooth C1 `bevel-wrap` edge troughs,
 *   and sloped `ramp` chutes descending to `tier = 1`
 * - Hollowed `tier = 1` vaulted stone underpass arches beneath intact `tier = 0` bridge decks
 * - Safe island-gated stepped dark vesicular basalt core pockets + regional basalt outcrops
 * - UV seams split cleanly per dominant cube face into the `6x1` (`1536x256`) atlas strip
 * - Crease-aware CSR attribute smoothing (`aMaterialZone`), `aStoneDetail` (`[floorMask, edgeWear]`),
 *   and single-pass shifted-octant Dual Contouring for solid-walled 8-piece vault separation.
 */

export interface LabyrinthMeshesResult {
  monolithGeometry: THREE.BufferGeometry;
  sanctumInlayGeometry: THREE.BufferGeometry;
  startSpringGeometry: THREE.BufferGeometry;
  beadGeometry: THREE.BufferGeometry;
}

export interface VaultOctantsResult {
  octantGeometries: THREE.BufferGeometry[];
  sanctumHalfPosZGeometry: THREE.BufferGeometry;
  sanctumHalfNegZGeometry: THREE.BufferGeometry;
}

export interface RawStoneMeshBuffers {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  materialZone: Float32Array;
  stoneDetail: Float32Array;
}

export interface RawSanctumHalfBuffers {
  positions: Float32Array;
  normals: Float32Array;
  sanctumZone: Float32Array;
}

export interface VaultOctantsRawPayload {
  octants: RawStoneMeshBuffers[];
  sanctumPosZ: RawSanctumHalfBuffers;
  sanctumNegZ: RawSanctumHalfBuffers;
}

interface FaceBasis {
  id: CubeFaceId;
  index: number;
  nx: number;
  ny: number;
  nz: number;
  ux: number;
  uy: number;
  uz: number;
  vx: number;
  vy: number;
  vz: number;
}

const FACE_BASES: readonly FaceBasis[] = [
  { id: 'px', index: 0, nx: 1, ny: 0, nz: 0, ux: 0, uy: 0, uz: -1, vx: 0, vy: 1, vz: 0 },
  { id: 'nx', index: 1, nx: -1, ny: 0, nz: 0, ux: 0, uy: 0, uz: 1, vx: 0, vy: 1, vz: 0 },
  { id: 'py', index: 2, nx: 0, ny: 1, nz: 0, ux: 1, uy: 0, uz: 0, vx: 0, vy: 0, vz: -1 },
  { id: 'ny', index: 3, nx: 0, ny: -1, nz: 0, ux: 1, uy: 0, uz: 0, vx: 0, vy: 0, vz: 1 },
  { id: 'pz', index: 4, nx: 0, ny: 0, nz: 1, ux: 1, uy: 0, uz: 0, vx: 0, vy: 1, vz: 0 },
  { id: 'nz', index: 5, nx: 0, ny: 0, nz: -1, ux: -1, uy: 0, uz: 0, vx: 0, vy: 1, vz: 0 },
];

const FACE_BY_ID: Record<CubeFaceId, FaceBasis> = {
  px: FACE_BASES[0],
  nx: FACE_BASES[1],
  py: FACE_BASES[2],
  ny: FACE_BASES[3],
  pz: FACE_BASES[4],
  nz: FACE_BASES[5],
};

const INV_SQRT2 = Math.SQRT1_2;
const INV_SQRT3 = 1 / Math.sqrt(3);

// Dual Contouring voxel grid (`N = 84`, default bound = `0.92`; octant pass uses `0.95` + `halfSlot = 0.024`)
const DC_GRID_N = 84;
const DC_BOUND = 0.92;
const DC_OCTANT_BOUND = 0.95;
const OCTANT_HALF_SLOT = 0.024;
const DC_MIN = -DC_BOUND;
const DC_SPAN = DC_BOUND * 2;
const DC_STEP = DC_SPAN / DC_GRID_N;
const DC_VERTS = DC_GRID_N + 1;

// Pocket clearance grid resolution per face
const POCKET_N = 41;
const POCKET_BOUND = 0.56;
const POCKET_STEP = (POCKET_BOUND * 2) / (POCKET_N - 1);

const ARCH_SPRING_H = 0.055;
const ARCH_CROWN_H = 0.104; // 0.012 clearance above 0.092 bead diameter; 0.056 stone arch crown
const BRIDGE_HALF_SPAN = 0.130;

// Inner faceted geode vault extents (strictly >= 0.135 rock below tier=1 underpass floors)
const VAULT_R_BOX = 0.435;
const VAULT_R_EDGE = 0.465;
const VAULT_R_CORNER = 0.505;

// Coarse spatial bin grid (`16^3`) for fast exact analytical SDF gradient & floor lookups
const EDGE_BIN_N = 16;
const EDGE_BIN_TOTAL = EDGE_BIN_N * EDGE_BIN_N * EDGE_BIN_N;
const sBinCounts = new Int32Array(EDGE_BIN_TOTAL);
const sBinOffsets = new Int32Array(EDGE_BIN_TOTAL + 1);
const sBinEdgeList = new Int32Array(32768);
const sNearEdgeIdx = new Int32Array(64);
const sNearSocketIdx = new Int32Array(8);

// Pre-allocated scratch buffers for zero-GC startup and lazy octant construction
const SCRATCH_VERT_TOTAL = DC_VERTS * DC_VERTS * DC_VERTS;
const SCRATCH_CELL_TOTAL = DC_GRID_N * DC_GRID_N * DC_GRID_N;
const SCRATCH_MAX_EDGES = 260000;
const sSdfGrid = new Float32Array(SCRATCH_VERT_TOTAL);
const sCellToActive = new Int32Array(SCRATCH_CELL_TOTAL);
const sEdgeAxis = new Uint8Array(SCRATCH_MAX_EDGES);
const sEdgeSignDir = new Int8Array(SCRATCH_MAX_EDGES);
const sEdgeC0 = new Int32Array(SCRATCH_MAX_EDGES);
const sEdgeC1 = new Int32Array(SCRATCH_MAX_EDGES);
const sEdgeC2 = new Int32Array(SCRATCH_MAX_EDGES);
const sEdgeC3 = new Int32Array(SCRATCH_MAX_EDGES);
const sEdgeIx = new Uint8Array(SCRATCH_MAX_EDGES);
const sEdgeIy = new Uint8Array(SCRATCH_MAX_EDGES);
const sEdgeIz = new Uint8Array(SCRATCH_MAX_EDGES);
const sEdgeT0 = new Float32Array(SCRATCH_MAX_EDGES);
const sCellBoxCoords = new Int32Array(SCRATCH_MAX_EDGES * 3);
const sQefBuf = new Float32Array(SCRATCH_MAX_EDGES * 13);
const sCellVertPos = new Float32Array(SCRATCH_MAX_EDGES * 3);
const sTriCellIndices = new Int32Array(SCRATCH_MAX_EDGES * 6);
const sTriFaceNormals = new Float32Array(SCRATCH_MAX_EDGES * 6);
const sVertTriCounts = new Int32Array(SCRATCH_MAX_EDGES);
const sAvgCellNorm = new Float32Array(SCRATCH_MAX_EDGES * 3);
const sCellIsSharp = new Uint8Array(SCRATCH_MAX_EDGES);
const sVertTriOffsets = new Int32Array(SCRATCH_MAX_EDGES + 1);
const sVertTriList = new Int32Array(SCRATCH_MAX_EDGES * 6);
const sCellAo = new Float32Array(SCRATCH_MAX_EDGES);
const sCellBasalt = new Float32Array(SCRATCH_MAX_EDGES);
const sCellPatina = new Float32Array(SCRATCH_MAX_EDGES);
const sCellFloorMask = new Float32Array(SCRATCH_MAX_EDGES);
const sCellEdgeWear = new Float32Array(SCRATCH_MAX_EDGES);
const sCellAoTmp = new Float32Array(SCRATCH_MAX_EDGES);
const sCellBasaltTmp = new Float32Array(SCRATCH_MAX_EDGES);
const sCellPatinaTmp = new Float32Array(SCRATCH_MAX_EDGES);
const sCellEdgeWearTmp = new Float32Array(SCRATCH_MAX_EDGES);
const sTriOctantId = new Uint8Array(SCRATCH_MAX_EDGES * 2);
const sOctantTriCounts = new Int32Array(8);
const sOctantCursors = new Int32Array(8);

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function organicField3D(x: number, y: number, z: number): number {
  const s1 = Math.sin(x * 7.3 + y * 3.1 - z * 5.7);
  const s2 = Math.sin(y * 13.7 - z * 8.9 + x * 6.1);
  const s3 = Math.cos(z * 21.1 + x * 14.3 - y * 11.7);
  return 0.5 + 0.25 * s1 + 0.15 * s2 + 0.10 * s3;
}

function smin(a: number, b: number, k: number): number {
  const h = clamp(0.5 + (0.5 * (b - a)) / k, 0, 1);
  return b + (a - b) * h - k * h * (1 - h);
}

interface CarveEdgeData {
  kind: LabyrinthEdge['kind'];
  segCount: number;
  pos: Float32Array;
  norm: Float32Array;
  segTx: Float32Array;
  segTy: Float32Array;
  segTz: Float32Array;
  segLen: Float32Array;
  segBx: Float32Array;
  segBy: Float32Array;
  segBz: Float32Array;
  extStart: number;
  extEnd: number;
  colinearStart: boolean;
  colinearEnd: boolean;
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
  bridgeIndices: number[];
}

interface BridgeVolume {
  face: FaceBasis;
  cx: number;
  cy: number;
  cz: number;
  overX: number;
  overY: number;
  overZ: number;
  underX: number;
  underY: number;
  underZ: number;
  halfSpan: number;
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

interface SocketRecess {
  cx: number;
  cy: number;
  cz: number;
  nx: number;
  ny: number;
  nz: number;
  radius: number;
  flareStartH: number;
  coneSlope: number;
  depthOffset: number;
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

function dominantFaceBasis(x: number, y: number, z: number): FaceBasis {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  if (ax >= ay && ax >= az) {
    return x >= 0 ? FACE_BASES[0] : FACE_BASES[1];
  }
  if (ay >= az) {
    return y >= 0 ? FACE_BASES[2] : FACE_BASES[3];
  }
  return z >= 0 ? FACE_BASES[4] : FACE_BASES[5];
}

function resolveGridPitch(topology: LabyrinthTopology): number {
  for (let i = 0; i < topology.nodes.length; i++) {
    const n = topology.nodes[i];
    if (n.tier !== 0 || n.kind === 'portal') continue;
    const basis = FACE_BY_ID[n.face];
    const du = n.u - (topology.gridSize - 1) * 0.5;
    if (Math.abs(du) >= 1) {
      const uCoord = n.localPos[0] * basis.ux + n.localPos[1] * basis.uy + n.localPos[2] * basis.uz;
      const candidate = Math.abs(uCoord / du);
      if (candidate > 0.1 && candidate < 0.4) return candidate;
    }
    const dv = n.v - (topology.gridSize - 1) * 0.5;
    if (Math.abs(dv) >= 1) {
      const vCoord = n.localPos[0] * basis.vx + n.localPos[1] * basis.vy + n.localPos[2] * basis.vz;
      const candidate = Math.abs(vCoord / dv);
      if (candidate > 0.1 && candidate < 0.4) return candidate;
    }
  }
  return 0.215;
}

function buildBridgeVolumes(topology: LabyrinthTopology, gridPitch: number): BridgeVolume[] {
  const halfGrid = (topology.gridSize - 1) * 0.5;
  const H = topology.cubeHalfExtent;
  const floorR = H - topology.underpassDepth;
  const result: BridgeVolume[] = [];

  for (let i = 0; i < topology.bridges.length; i++) {
    const b: StoneBridgeSpec = topology.bridges[i];
    const face = FACE_BY_ID[b.face];
    let cx = 0;
    let cy = 0;
    let cz = 0;
    let found = false;
    for (let ni = 0; ni < topology.nodes.length; ni++) {
      const node = topology.nodes[ni];
      if (node.face === b.face && node.u === b.u && node.v === b.v && node.tier === 1) {
        cx = node.localPos[0];
        cy = node.localPos[1];
        cz = node.localPos[2];
        found = true;
        break;
      }
    }
    if (!found) {
      const uVal = (b.u - halfGrid) * gridPitch;
      const vVal = (b.v - halfGrid) * gridPitch;
      cx = face.nx * floorR + face.ux * uVal + face.vx * vVal;
      cy = face.ny * floorR + face.uy * uVal + face.vy * vVal;
      cz = face.nz * floorR + face.uz * uVal + face.vz * vVal;
    }

    const overIsU = b.overAxis === 'u';
    const overX = overIsU ? face.ux : face.vx;
    const overY = overIsU ? face.uy : face.vy;
    const overZ = overIsU ? face.uz : face.vz;
    const underX = overIsU ? face.vx : face.ux;
    const underY = overIsU ? face.vy : face.uy;
    const underZ = overIsU ? face.vz : face.uz;

    const pad = BRIDGE_HALF_SPAN + 0.16;
    result.push({
      face,
      cx,
      cy,
      cz,
      overX,
      overY,
      overZ,
      underX,
      underY,
      underZ,
      halfSpan: BRIDGE_HALF_SPAN,
      minX: cx - pad,
      minY: cy - pad,
      minZ: cz - pad,
      maxX: cx + pad,
      maxY: cy + pad,
      maxZ: cz + pad,
    });
  }
  return result;
}

function getEdgeSampleTangentAtNode(
  edge: LabyrinthEdge,
  nodeId: number,
  nodeNormal: readonly [number, number, number],
  out: [number, number, number]
): boolean {
  const kCount = Math.floor(edge.samplesPos.length / 3);
  if (kCount < 2) return false;
  let dx = 0;
  let dy = 0;
  let dz = 0;
  if (edge.from === nodeId) {
    dx = edge.samplesPos[3] - edge.samplesPos[0];
    dy = edge.samplesPos[4] - edge.samplesPos[1];
    dz = edge.samplesPos[5] - edge.samplesPos[2];
  } else if (edge.to === nodeId) {
    const i0 = (kCount - 1) * 3;
    const i1 = (kCount - 2) * 3;
    dx = edge.samplesPos[i1] - edge.samplesPos[i0];
    dy = edge.samplesPos[i1 + 1] - edge.samplesPos[i0 + 1];
    dz = edge.samplesPos[i1 + 2] - edge.samplesPos[i0 + 2];
  } else {
    return false;
  }
  const dn = dx * nodeNormal[0] + dy * nodeNormal[1] + dz * nodeNormal[2];
  dx -= dn * nodeNormal[0];
  dy -= dn * nodeNormal[1];
  dz -= dn * nodeNormal[2];
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-6) return false;
  out[0] = dx / len;
  out[1] = dy / len;
  out[2] = dz / len;
  return true;
}

function buildCarveEdges(
  topology: LabyrinthTopology,
  bridges: readonly BridgeVolume[]
): CarveEdgeData[] {
  const H = topology.cubeHalfExtent;
  const nodeIncident: LabyrinthEdge[][] = Array.from({ length: topology.nodes.length }, () => []);
  const seenPairs = new Set<string>();
  const uniqueEdges: LabyrinthEdge[] = [];

  for (let i = 0; i < topology.edges.length; i++) {
    const e = topology.edges[i];
    nodeIncident[e.from]?.push(e);
    nodeIncident[e.to]?.push(e);
    const lo = e.from < e.to ? e.from : e.to;
    const hi = e.from < e.to ? e.to : e.from;
    const key = `${lo}:${hi}:${e.kind}`;
    if (!seenPairs.has(key)) {
      seenPairs.add(key);
      uniqueEdges.push(e);
    }
  }

  const hw = topology.channelWidth * 0.5;
  const padRadius = hw + 0.032;
  const tSelf: [number, number, number] = [0, 0, 0];
  const tOther: [number, number, number] = [0, 0, 0];
  const result: CarveEdgeData[] = [];

  for (let ei = 0; ei < uniqueEdges.length; ei++) {
    const e = uniqueEdges[ei];
    const kCount = Math.floor(e.samplesPos.length / 3);
    if (kCount < 2) continue;

    let extStart = hw;
    let extEnd = hw;
    let colinearStart = false;
    let colinearEnd = false;
    const fromNormal = topology.nodes[e.from].localNormal;
    if (getEdgeSampleTangentAtNode(e, e.from, fromNormal, tSelf)) {
      const inc = nodeIncident[e.from];
      for (let j = 0; j < inc.length; j++) {
        const other = inc[j];
        if (other.id === e.id || (other.from === e.to && other.to === e.from)) continue;
        if (getEdgeSampleTangentAtNode(other, e.from, fromNormal, tOther)) {
          if (tSelf[0] * tOther[0] + tSelf[1] * tOther[1] + tSelf[2] * tOther[2] < -0.82) {
            colinearStart = true;
            extStart = 0.026;
            break;
          }
        }
      }
    }
    const toNormal = topology.nodes[e.to].localNormal;
    if (getEdgeSampleTangentAtNode(e, e.to, toNormal, tSelf)) {
      const inc = nodeIncident[e.to];
      for (let j = 0; j < inc.length; j++) {
        const other = inc[j];
        if (other.id === e.id || (other.from === e.to && other.to === e.from)) continue;
        if (getEdgeSampleTangentAtNode(other, e.to, toNormal, tOther)) {
          if (tSelf[0] * tOther[0] + tSelf[1] * tOther[1] + tSelf[2] * tOther[2] < -0.82) {
            colinearEnd = true;
            extEnd = 0.026;
            break;
          }
        }
      }
    }

    const p0x = e.samplesPos[0];
    const p0y = e.samplesPos[1];
    const p0z = e.samplesPos[2];
    const lastIdx = (kCount - 1) * 3;
    const p1x = e.samplesPos[lastIdx];
    const p1y = e.samplesPos[lastIdx + 1];
    const p1z = e.samplesPos[lastIdx + 2];
    const n0x = e.samplesNormal[0];
    const n0y = e.samplesNormal[1];
    const n0z = e.samplesNormal[2];
    const n1x = e.samplesNormal[lastIdx];
    const n1y = e.samplesNormal[lastIdx + 1];
    const n1z = e.samplesNormal[lastIdx + 2];

    const spanX = p1x - p0x;
    const spanY = p1y - p0y;
    const spanZ = p1z - p0z;
    const spanLenSq = spanX * spanX + spanY * spanY + spanZ * spanZ;
    let isStraight =
      e.kind !== 'bevel-wrap' &&
      e.kind !== 'ramp' &&
      n0x * n1x + n0y * n1y + n0z * n1z > 0.999 &&
      spanLenSq > 1e-8;

    if (isStraight) {
      for (let k = 1; k < kCount - 1; k++) {
        const vx = e.samplesPos[k * 3] - p0x;
        const vy = e.samplesPos[k * 3 + 1] - p0y;
        const vz = e.samplesPos[k * 3 + 2] - p0z;
        const proj = clamp((vx * spanX + vy * spanY + vz * spanZ) / spanLenSq, 0, 1);
        const rx = vx - proj * spanX;
        const ry = vy - proj * spanY;
        const rz = vz - proj * spanZ;
        if (rx * rx + ry * ry + rz * rz > 1e-6) {
          isStraight = false;
          break;
        }
      }
    }

    const pos = isStraight ? new Float32Array([p0x, p0y, p0z, p1x, p1y, p1z]) : e.samplesPos;
    const norm = isStraight ? new Float32Array([n0x, n0y, n0z, n1x, n1y, n1z]) : e.samplesNormal;

    const segCount = Math.floor(pos.length / 3) - 1;
    const segTx = new Float32Array(segCount);
    const segTy = new Float32Array(segCount);
    const segTz = new Float32Array(segCount);
    const segLen = new Float32Array(segCount);
    const segBx = new Float32Array(segCount);
    const segBy = new Float32Array(segCount);
    const segBz = new Float32Array(segCount);

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (let s = 0; s < segCount; s++) {
      const a = s * 3;
      const b = (s + 1) * 3;
      const dx = pos[b] - pos[a];
      const dy = pos[b + 1] - pos[a + 1];
      const dz = pos[b + 2] - pos[a + 2];

      const nx = e.kind === 'ramp' ? n0x : 0.5 * (norm[a] + norm[b]);
      const ny = e.kind === 'ramp' ? n0y : 0.5 * (norm[a + 1] + norm[b + 1]);
      const nz = e.kind === 'ramp' ? n0z : 0.5 * (norm[a + 2] + norm[b + 2]);
      let pdx = dx;
      let pdy = dy;
      let pdz = dz;
      if (e.kind === 'ramp') {
        const dn = pdx * nx + pdy * ny + pdz * nz;
        pdx -= dn * nx;
        pdy -= dn * ny;
        pdz -= dn * nz;
      }
      const len = Math.max(1e-6, Math.hypot(pdx, pdy, pdz));
      const tx = pdx / len;
      const ty = pdy / len;
      const tz = pdz / len;
      segTx[s] = tx;
      segTy[s] = ty;
      segTz[s] = tz;
      segLen[s] = len;

      const bvx = ny * tz - nz * ty;
      const bvy = nz * tx - nx * tz;
      const bvz = nx * ty - ny * tx;
      const bl = Math.max(1e-6, Math.sqrt(bvx * bvx + bvy * bvy + bvz * bvz));
      segBx[s] = bvx / bl;
      segBy[s] = bvy / bl;
      segBz[s] = bvz / bl;
    }

    const totalK = segCount + 1;
    const reachOut =
      e.kind === 'surface' || e.kind === 'bevel-wrap'
        ? topology.surfaceDepth + 0.025
        : topology.underpassDepth + 0.025;
    for (let k = 0; k < totalK; k++) {
      const px = pos[k * 3];
      const py = pos[k * 3 + 1];
      const pz = pos[k * 3 + 2];
      const nx = norm[k * 3];
      const ny = norm[k * 3 + 1];
      const nz = norm[k * 3 + 2];
      const topX = px + nx * reachOut;
      const topY = py + ny * reachOut;
      const topZ = pz + nz * reachOut;
      minX = Math.min(minX, px, topX);
      minY = Math.min(minY, py, topY);
      minZ = Math.min(minZ, pz, topZ);
      maxX = Math.max(maxX, px, topX);
      maxY = Math.max(maxY, py, topY);
      maxZ = Math.max(maxZ, pz, topZ);
    }

    const cubeLim = H + DC_STEP;
    minX = Math.max(-cubeLim, minX - padRadius);
    minY = Math.max(-cubeLim, minY - padRadius);
    minZ = Math.max(-cubeLim, minZ - padRadius);
    maxX = Math.min(cubeLim, maxX + padRadius);
    maxY = Math.min(cubeLim, maxY + padRadius);
    maxZ = Math.min(cubeLim, maxZ + padRadius);

    const bridgeIndices: number[] = [];
    if (e.kind === 'underpass') {
      for (let bi = 0; bi < bridges.length; bi++) {
        const br = bridges[bi];
        if (
          br.maxX >= minX &&
          br.minX <= maxX &&
          br.maxY >= minY &&
          br.minY <= maxY &&
          br.maxZ >= minZ &&
          br.minZ <= maxZ
        ) {
          bridgeIndices.push(bi);
        }
      }
    }

    result.push({
      kind: e.kind,
      segCount,
      pos,
      norm,
      segTx,
      segTy,
      segTz,
      segLen,
      segBx,
      segBy,
      segBz,
      extStart,
      extEnd,
      colinearStart,
      colinearEnd,
      minX,
      minY,
      minZ,
      maxX,
      maxY,
      maxZ,
      bridgeIndices,
    });
  }

  return result;
}

function evalCarveEdgeAirAndCoords(
  edge: CarveEdgeData,
  px: number,
  py: number,
  pz: number,
  hw: number,
  H: number,
  bridges: readonly BridgeVolume[]
): { air: number; h: number; flatDist: number } {
  const segCount = edge.segCount;
  let bestSeg = 0;
  let bestT = 0;

  if (segCount === 1) {
    const ax = edge.pos[0];
    const ay = edge.pos[1];
    const az = edge.pos[2];
    const tx = edge.segTx[0];
    const ty = edge.segTy[0];
    const tz = edge.segTz[0];
    const len = edge.segLen[0];
    bestT = clamp((px - ax) * tx + (py - ay) * ty + (pz - az) * tz, 0, len) / len;
  } else {
    let bestDistSq = Infinity;
    for (let s = 0; s < segCount; s++) {
      const a = s * 3;
      const ax = edge.pos[a];
      const ay = edge.pos[a + 1];
      const az = edge.pos[a + 2];
      const tx = edge.segTx[s];
      const ty = edge.segTy[s];
      const tz = edge.segTz[s];
      const len = edge.segLen[s];
      const vx = px - ax;
      const vy = py - ay;
      const vz = pz - az;
      const sAlong = clamp(vx * tx + vy * ty + vz * tz, 0, len);
      let rx = vx - sAlong * tx;
      let ry = vy - sAlong * ty;
      let rz = vz - sAlong * tz;
      if (edge.kind === 'ramp') {
        const rn = rx * edge.norm[0] + ry * edge.norm[1] + rz * edge.norm[2];
        rx -= rn * edge.norm[0];
        ry -= rn * edge.norm[1];
        rz -= rn * edge.norm[2];
      }
      const dSq = rx * rx + ry * ry + rz * rz;
      if (dSq < bestDistSq) {
        bestDistSq = dSq;
        bestSeg = s;
        bestT = sAlong / len;
      }
    }
  }

  const i0 = bestSeg * 3;
  const i1 = (bestSeg + 1) * 3;
  const cx = edge.pos[i0] + (edge.pos[i1] - edge.pos[i0]) * bestT;
  const cy = edge.pos[i0 + 1] + (edge.pos[i1 + 1] - edge.pos[i0 + 1]) * bestT;
  const cz = edge.pos[i0 + 2] + (edge.pos[i1 + 2] - edge.pos[i0 + 2]) * bestT;

  let nx = edge.norm[i0] + (edge.norm[i1] - edge.norm[i0]) * bestT;
  let ny = edge.norm[i0 + 1] + (edge.norm[i1 + 1] - edge.norm[i0 + 1]) * bestT;
  let nz = edge.norm[i0 + 2] + (edge.norm[i1 + 2] - edge.norm[i0 + 2]) * bestT;
  const nLen = Math.max(1e-6, Math.sqrt(nx * nx + ny * ny + nz * nz));
  const invNLen = 1 / nLen;
  nx *= invNLen;
  ny *= invNLen;
  nz *= invNLen;

  const dx = px - cx;
  const dy = py - cy;
  const dz = pz - cz;

  let h: number;
  let hWall: number;
  if (edge.kind === 'ramp') {
    const fnx = edge.norm[0];
    const fny = edge.norm[1];
    const fnz = edge.norm[2];
    const zRel = dx * fnx + dy * fny + dz * fnz;
    const cosSlope = Math.max(0.65, nx * fnx + ny * fny + nz * fnz);
    h = zRel * cosSlope;
    hWall = Math.max(0, zRel);
  } else {
    h = dx * nx + dy * ny + dz * nz;
    hWall = Math.max(0, h);
  }
  const w = Math.abs(dx * edge.segBx[bestSeg] + dy * edge.segBy[bestSeg] + dz * edge.segBz[bestSeg]);

  let sOvershoot = 0;
  let extAllowed = 1.0;
  let overshootColinear = false;
  if (bestSeg === 0 && bestT <= 0.0001) {
    const proj = -(dx * edge.segTx[0] + dy * edge.segTy[0] + dz * edge.segTz[0]);
    if (proj > 0) {
      sOvershoot = proj;
      extAllowed = edge.extStart;
      overshootColinear = edge.colinearStart;
    }
  } else if (bestSeg === segCount - 1 && bestT >= 0.9999) {
    const last = segCount - 1;
    const proj = dx * edge.segTx[last] + dy * edge.segTy[last] + dz * edge.segTz[last];
    if (proj > 0) {
      sOvershoot = proj;
      extAllowed = edge.extEnd;
      overshootColinear = edge.colinearEnd;
    }
  }

  const flatDist = Math.hypot(w, sOvershoot);
  const maxAbs = Math.max(Math.abs(px), Math.abs(py), Math.abs(pz));
  const nearOuterRim = maxAbs - (H - 0.015);
  const lipFlare = nearOuterRim > 0 ? Math.min(0.003, nearOuterRim * 0.45) : 0;
  const hwFloor = hw - 0.003;
  const hwProfile = Math.min(hw + 0.003, hwFloor + 0.015 * hWall + lipFlare);

  if (edge.kind === 'underpass') {
    const hwArch = hw - 0.003;
    const wNorm = Math.min(1.0, w / hwArch);
    const hCeil = ARCH_SPRING_H + (ARCH_CROWN_H - ARCH_SPRING_H) * Math.sqrt(Math.max(0, 1 - wNorm * wNorm));

    const latAir = hwProfile - w;
    const endProfileU =
      extAllowed > 0.03
        ? Math.min(extAllowed + 0.003, extAllowed - 0.003 + 0.015 * hWall + lipFlare)
        : extAllowed + 0.015 * hWall + lipFlare;
    const endAir = extAllowed === 1.0 ? 1.0 : endProfileU - sOvershoot;
    const wallAir = sOvershoot > 0 && extAllowed < 0.99 ? Math.min(latAir, endAir) : latAir;
    const openTrenchAir = smin(wallAir, h, 0.004);

    if (edge.bridgeIndices.length === 0) {
      return { air: openTrenchAir, h, flatDist };
    }

    const tunnelAir = smin(Math.min(hwArch - w, hCeil - h), h, 0.004);

    let minDistInsideBridge = Infinity;
    for (let bi = 0; bi < edge.bridgeIndices.length; bi++) {
      const br = bridges[edge.bridgeIndices[bi]];
      const rx = px - br.cx;
      const ry = py - br.cy;
      const rz = pz - br.cz;
      const alongUnder = Math.abs(rx * br.underX + ry * br.underY + rz * br.underZ);
      const alongOver = Math.abs(rx * br.overX + ry * br.overY + rz * br.overZ);
      if (alongOver <= hw + 0.04) {
        const dPortal = alongUnder - br.halfSpan;
        if (dPortal < minDistInsideBridge) minDistInsideBridge = dPortal;
      }
    }

    const air =
      minDistInsideBridge === Infinity
        ? openTrenchAir
        : Math.max(tunnelAir, Math.min(openTrenchAir, minDistInsideBridge));
    return { air, h, flatDist };
  }

  const dLat = hwProfile - w;
  let dHoriz = dLat;
  if (sOvershoot > 0) {
    if (overshootColinear) {
      const endProfile = extAllowed + 0.015 * hWall + lipFlare;
      dHoriz = Math.min(dLat, endProfile - sOvershoot);
    } else {
      const endProfile =
        extAllowed > 0.03
          ? Math.min(extAllowed + 0.003, extAllowed - 0.003 + 0.015 * hWall + lipFlare)
          : extAllowed;
      const dEnd = endProfile - sOvershoot;
      if (dLat < 0 && dEnd < 0) {
        dHoriz = -Math.sqrt(dLat * dLat + dEnd * dEnd);
      } else {
        const rCorner = 0.005;
        const qx = -dLat - rCorner;
        const qy = -dEnd - rCorner;
        const qxp = qx > 0 ? qx : 0;
        const qyp = qy > 0 ? qy : 0;
        dHoriz = -(Math.sqrt(qxp * qxp + qyp * qyp) + Math.min(Math.max(qx, qy), 0) - rCorner);
      }
    }
  }

  return { air: smin(dHoriz, h, 0.004), h, flatDist };
}

function evalCarveEdgeAir(
  edge: CarveEdgeData,
  px: number,
  py: number,
  pz: number,
  hw: number,
  H: number,
  bridges: readonly BridgeVolume[]
): number {
  return evalCarveEdgeAirAndCoords(edge, px, py, pz, hw, H, bridges).air;
}

function buildFacePocketGrids(
  topology: LabyrinthTopology,
  edges: readonly CarveEdgeData[],
  bridges: readonly BridgeVolume[]
): Float32Array[] {
  const grids: Float32Array[] = [];
  const startNode = topology.nodes[topology.startNodeId];
  const goalNode = topology.nodes[topology.goalNodeId];

  const faceSegments: number[][] = Array.from({ length: 6 }, () => []);
  for (let ei = 0; ei < edges.length; ei++) {
    const e = edges[ei];
    const totalK = e.segCount + 1;
    for (let s = 0; s < e.segCount; s++) {
      const a = s * 3;
      const b = (s + 1) * 3;
      const mx = 0.5 * (e.pos[a] + e.pos[b]);
      const my = 0.5 * (e.pos[a + 1] + e.pos[b + 1]);
      const mz = 0.5 * (e.pos[a + 2] + e.pos[b + 2]);
      const basis = dominantFaceBasis(mx, my, mz);
      faceSegments[basis.index].push(
        e.pos[a] * basis.ux + e.pos[a + 1] * basis.uy + e.pos[a + 2] * basis.uz,
        e.pos[a] * basis.vx + e.pos[a + 1] * basis.vy + e.pos[a + 2] * basis.vz,
        e.pos[b] * basis.ux + e.pos[b + 1] * basis.uy + e.pos[b + 2] * basis.uz,
        e.pos[b] * basis.vx + e.pos[b + 1] * basis.vy + e.pos[b + 2] * basis.vz
      );
    }
    for (let k = 0; k < totalK; k += Math.max(1, totalK - 1)) {
      const px = e.pos[k * 3];
      const py = e.pos[k * 3 + 1];
      const pz = e.pos[k * 3 + 2];
      const basis = dominantFaceBasis(px, py, pz);
      const u = px * basis.ux + py * basis.uy + pz * basis.uz;
      const v = px * basis.vx + py * basis.vy + pz * basis.vz;
      faceSegments[basis.index].push(u, v, u, v);
    }
  }

  const safeMargin = topology.channelWidth * 0.5 + 0.115;

  for (let fi = 0; fi < 6; fi++) {
    const basis = FACE_BASES[fi];
    const segs = faceSegments[fi];
    const grid = new Float32Array(POCKET_N * POCKET_N);

    const obsU: number[] = [];
    const obsV: number[] = [];
    const obsR: number[] = [];
    for (let bi = 0; bi < bridges.length; bi++) {
      const br = bridges[bi];
      if (br.face.index === fi) {
        obsU.push(br.cx * basis.ux + br.cy * basis.uy + br.cz * basis.uz);
        obsV.push(br.cx * basis.vx + br.cy * basis.vy + br.cz * basis.vz);
        obsR.push(0.14);
      }
    }
    if (startNode.face === basis.id) {
      const [sx, sy, sz] = startNode.localPos;
      obsU.push(sx * basis.ux + sy * basis.uy + sz * basis.uz);
      obsV.push(sx * basis.vx + sy * basis.vy + sz * basis.vz);
      obsR.push(0.13);
    }
    if (goalNode.face === basis.id) {
      const [gx, gy, gz] = goalNode.localPos;
      obsU.push(gx * basis.ux + gy * basis.uy + gz * basis.uz);
      obsV.push(gx * basis.vx + gy * basis.vy + gz * basis.vz);
      obsR.push(0.15);
    }

    const segAx = new Float32Array(segs.length >> 2);
    const segAy = new Float32Array(segs.length >> 2);
    const segDx = new Float32Array(segs.length >> 2);
    const segDy = new Float32Array(segs.length >> 2);
    const segInvLenSq = new Float32Array(segs.length >> 2);
    const segCount2d = segs.length >> 2;
    for (let s = 0; s < segCount2d; s++) {
      const s4 = s * 4;
      const ax = segs[s4];
      const ay = segs[s4 + 1];
      const dx = segs[s4 + 2] - ax;
      const dy = segs[s4 + 3] - ay;
      const lenSq = dx * dx + dy * dy;
      segAx[s] = ax;
      segAy[s] = ay;
      segDx[s] = dx;
      segDy[s] = dy;
      segInvLenSq[s] = lenSq > 1e-9 ? 1 / lenSq : 0;
    }

    for (let j = 0; j < POCKET_N; j++) {
      const vCoord = -POCKET_BOUND + j * POCKET_STEP;
      const edgeDistV = POCKET_BOUND - Math.abs(vCoord);
      for (let i = 0; i < POCKET_N; i++) {
        const uCoord = -POCKET_BOUND + i * POCKET_STEP;
        const edgeDistU = POCKET_BOUND - Math.abs(uCoord);
        let minDist = edgeDistU < edgeDistV ? edgeDistU : edgeDistV;

        let minSegDistSq = Infinity;
        for (let s = 0; s < segCount2d; s++) {
          const ux = uCoord - segAx[s];
          const vy = vCoord - segAy[s];
          const dx = segDx[s];
          const dy = segDy[s];
          const invL = segInvLenSq[s];
          const t = invL > 0 ? clamp((ux * dx + vy * dy) * invL, 0, 1) : 0;
          const rx = ux - t * dx;
          const ry = vy - t * dy;
          const dSq = rx * rx + ry * ry;
          if (dSq < minSegDistSq) minSegDistSq = dSq;
        }
        const segD = Math.sqrt(minSegDistSq) - safeMargin;
        if (segD < minDist) minDist = segD;

        for (let oi = 0; oi < obsU.length; oi++) {
          const du = uCoord - obsU[oi];
          const dv = vCoord - obsV[oi];
          const d = Math.sqrt(du * du + dv * dv) - (obsR[oi] + 0.055);
          if (d < minDist) minDist = d;
        }

        grid[j * POCKET_N + i] = minDist;
      }
    }
    grids.push(grid);
  }

  return grids;
}

function samplePocketClearance(grid: Float32Array, uCoord: number, vCoord: number): number {
  if (Math.abs(uCoord) >= POCKET_BOUND || Math.abs(vCoord) >= POCKET_BOUND) {
    return -0.05;
  }
  const fu = (uCoord + POCKET_BOUND) / POCKET_STEP;
  const fv = (vCoord + POCKET_BOUND) / POCKET_STEP;
  const iu = clamp(Math.floor(fu), 0, POCKET_N - 2);
  const iv = clamp(Math.floor(fv), 0, POCKET_N - 2);
  const tu = fu - iu;
  const tv = fv - iv;
  const row0 = iv * POCKET_N + iu;
  const row1 = row0 + POCKET_N;
  return (
    (grid[row0] + (grid[row0 + 1] - grid[row0]) * tu) * (1 - tv) +
    (grid[row1] + (grid[row1 + 1] - grid[row1]) * tu) * tv
  );
}

function evalPocketCarveAirForBasis(
  basis: FaceBasis,
  px: number,
  py: number,
  pz: number,
  H: number,
  grid: Float32Array
): number {
  const uCoord = px * basis.ux + py * basis.uy + pz * basis.uz;
  const vCoord = px * basis.vx + py * basis.vy + pz * basis.vz;
  const dPocket = samplePocketClearance(grid, uCoord, vCoord) - 0.056;
  if (dPocket <= -0.025) return -1.0;

  const faceCoord = px * basis.nx + py * basis.ny + pz * basis.nz;
  const depthFromSurface = H - faceCoord;
  const shelfAir = smin(dPocket, 0.075 - depthFromSurface, 0.005);
  const wellAir = smin(dPocket - 0.036, 0.135 - depthFromSurface, 0.005);
  return Math.max(shelfAir, wellAir);
}

function solveQef3x3(
  axx: number,
  axy: number,
  axz: number,
  ayy: number,
  ayz: number,
  azz: number,
  bx: number,
  by: number,
  bz: number,
  mx: number,
  my: number,
  mz: number,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
  count: number,
  out: Float32Array,
  outOffset: number
): void {
  const reg = 0.028 * Math.max(1, count);
  const m00 = axx + reg;
  const m01 = axy;
  const m02 = axz;
  const m11 = ayy + reg;
  const m12 = ayz;
  const m22 = azz + reg;

  const rx = bx - (axx * mx + axy * my + axz * mz);
  const ry = by - (axy * mx + ayy * my + ayz * mz);
  const rz = bz - (axz * mx + ayz * my + azz * mz);

  const c00 = m11 * m22 - m12 * m12;
  const c01 = m02 * m12 - m01 * m22;
  const c02 = m01 * m12 - m02 * m11;
  const c11 = m00 * m22 - m02 * m02;
  const c12 = m02 * m01 - m00 * m12;
  const c22 = m00 * m11 - m01 * m01;

  const det = m00 * c00 + m01 * c01 + m02 * c02;
  let vx = mx;
  let vy = my;
  let vz = mz;
  if (Math.abs(det) > 1e-10) {
    const invDet = 1 / det;
    vx = mx + (c00 * rx + c01 * ry + c02 * rz) * invDet;
    vy = my + (c01 * rx + c11 * ry + c12 * rz) * invDet;
    vz = mz + (c02 * rx + c12 * ry + c22 * rz) * invDet;
  }

  out[outOffset] = clamp(vx, minX, maxX);
  out[outOffset + 1] = clamp(vy, minY, maxY);
  out[outOffset + 2] = clamp(vz, minZ, maxZ);
}

let sLastActiveCells = 0;
let sLastCrossingEdges = 0;

function sampleGridSdfFast(
  vStrideZ: number,
  dcMin: number,
  invDcStep: number,
  x: number,
  y: number,
  z: number
): number {
  const fx = clamp((x - dcMin) * invDcStep, 0, DC_GRID_N - 0.001);
  const fy = clamp((y - dcMin) * invDcStep, 0, DC_GRID_N - 0.001);
  const fz = clamp((z - dcMin) * invDcStep, 0, DC_GRID_N - 0.001);
  const ix = fx | 0;
  const iy = fy | 0;
  const iz = fz | 0;
  const tx = fx - ix;
  const ty = fy - iy;
  const tz = fz - iz;
  const i000 = iz * vStrideZ + iy * DC_VERTS + ix;
  const i010 = i000 + DC_VERTS;
  const i001 = i000 + vStrideZ;
  const i011 = i001 + DC_VERTS;
  const c00 = sSdfGrid[i000] + (sSdfGrid[i000 + 1] - sSdfGrid[i000]) * tx;
  const c10 = sSdfGrid[i010] + (sSdfGrid[i010 + 1] - sSdfGrid[i010]) * tx;
  const c01 = sSdfGrid[i001] + (sSdfGrid[i001 + 1] - sSdfGrid[i001]) * tx;
  const c11 = sSdfGrid[i011] + (sSdfGrid[i011 + 1] - sSdfGrid[i011]) * tx;
  const c0 = c00 + (c10 - c00) * ty;
  const c1 = c01 + (c11 - c01) * ty;
  return c0 + (c1 - c0) * tz;
}

function evalInnerVaultAir(qx: number, qy: number, qz: number): number {
  const ax = Math.abs(qx);
  const ay = Math.abs(qy);
  const az = Math.abs(qz);
  const maxYZ = ay > az ? ay : az;
  const dBox = VAULT_R_BOX - (ax > maxYZ ? ax : maxYZ);
  const dXY = VAULT_R_EDGE - (ax + ay) * INV_SQRT2;
  const dXZ = VAULT_R_EDGE - (ax + az) * INV_SQRT2;
  const dYZ = VAULT_R_EDGE - (ay + az) * INV_SQRT2;
  const dEdges = dXY < dXZ ? (dXY < dYZ ? dXY : dYZ) : dXZ < dYZ ? dXZ : dYZ;
  const dCorner = VAULT_R_CORNER - (ax + ay + az) * INV_SQRT3;
  return dBox < dEdges ? (dBox < dCorner ? dBox : dCorner) : dEdges < dCorner ? dEdges : dCorner;
}

function buildCarveEdgeSpatialBins(carveEdges: readonly CarveEdgeData[]): void {
  sBinCounts.fill(0);
  const pad = 0.018;
  const n2 = EDGE_BIN_N * EDGE_BIN_N;
  const invBinStep = EDGE_BIN_N / DC_SPAN;

  for (let ei = 0; ei < carveEdges.length; ei++) {
    const e = carveEdges[ei];
    const x0 = clamp(Math.floor((e.minX - pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const y0 = clamp(Math.floor((e.minY - pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const z0 = clamp(Math.floor((e.minZ - pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const x1 = clamp(Math.floor((e.maxX + pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const y1 = clamp(Math.floor((e.maxY + pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const z1 = clamp(Math.floor((e.maxZ + pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    for (let bz = z0; bz <= z1; bz++) {
      const zOff = bz * n2;
      for (let by = y0; by <= y1; by++) {
        const yzOff = zOff + by * EDGE_BIN_N;
        for (let bx = x0; bx <= x1; bx++) {
          sBinCounts[yzOff + bx]++;
        }
      }
    }
  }

  sBinOffsets[0] = 0;
  for (let i = 0; i < EDGE_BIN_TOTAL; i++) {
    sBinOffsets[i + 1] = sBinOffsets[i] + sBinCounts[i];
  }
  sBinCounts.fill(0);

  for (let ei = 0; ei < carveEdges.length; ei++) {
    const e = carveEdges[ei];
    const x0 = clamp(Math.floor((e.minX - pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const y0 = clamp(Math.floor((e.minY - pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const z0 = clamp(Math.floor((e.minZ - pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const x1 = clamp(Math.floor((e.maxX + pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const y1 = clamp(Math.floor((e.maxY + pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const z1 = clamp(Math.floor((e.maxZ + pad - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    for (let bz = z0; bz <= z1; bz++) {
      const zOff = bz * n2;
      for (let by = y0; by <= y1; by++) {
        const yzOff = zOff + by * EDGE_BIN_N;
        for (let bx = x0; bx <= x1; bx++) {
          const cid = yzOff + bx;
          const idx = sBinOffsets[cid] + sBinCounts[cid]++;
          if (idx < sBinEdgeList.length) {
            sBinEdgeList[idx] = ei;
          }
        }
      }
    }
  }
}

function evalOuterCubeSdf(
  x: number,
  y: number,
  z: number,
  H: number,
  edgeLimit: number,
  cornerLimit: number
): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  const maxYZ = ay > az ? ay : az;
  const dFaces = (ax > maxYZ ? ax : maxYZ) - H;
  const dXY = (ax + ay) * INV_SQRT2 - edgeLimit;
  const dXZ = (ax + az) * INV_SQRT2 - edgeLimit;
  const dYZ = (ay + az) * INV_SQRT2 - edgeLimit;
  const dEdges = dXY > dXZ ? (dXY > dYZ ? dXY : dYZ) : dXZ > dYZ ? dXZ : dYZ;
  const dCorners = (ax + ay + az) * INV_SQRT3 - cornerLimit;
  return dFaces > dEdges
    ? dFaces > dCorners
      ? dFaces
      : dCorners
    : dEdges > dCorners
      ? dEdges
      : dCorners;
}

function evalExactLocalFiltered(
  px: number,
  py: number,
  pz: number,
  H: number,
  edgeLimit: number,
  cornerLimit: number,
  hw: number,
  halfSlot: number,
  bridges: readonly BridgeVolume[],
  carveEdges: readonly CarveEdgeData[],
  pocketGrids: readonly Float32Array[],
  sockets: readonly SocketRecess[],
  basisX: FaceBasis | null,
  basisY: FaceBasis | null,
  basisZ: FaceBasis | null,
  nearEdgeCount: number,
  nearSocketCount: number
): number {
  let x = px;
  let y = py;
  let z = pz;
  let gapAir = -1.0;

  if (halfSlot > 0) {
    const gx = halfSlot - Math.abs(px);
    const gy = halfSlot - Math.abs(py);
    const gz = halfSlot - Math.abs(pz);
    gapAir = gx > gy ? (gx > gz ? gx : gz) : gy > gz ? gy : gz;
    x = px >= 0 ? px - halfSlot : px + halfSlot;
    y = py >= 0 ? py - halfSlot : py + halfSlot;
    z = pz >= 0 ? pz - halfSlot : pz + halfSlot;
  }

  let sdf = evalOuterCubeSdf(x, y, z, H, edgeLimit, cornerLimit);
  if (halfSlot > 0) {
    if (gapAir > sdf) sdf = gapAir;
    const vaultAir = evalInnerVaultAir(x, y, z);
    if (vaultAir > sdf) sdf = vaultAir;
  }
  if (sdf > 0.025) return sdf;

  if (basisX !== null) {
    const pAir = evalPocketCarveAirForBasis(basisX, x, y, z, H, pocketGrids[basisX.index]);
    if (pAir > sdf) sdf = pAir;
  }
  if (basisY !== null) {
    const pAir = evalPocketCarveAirForBasis(basisY, x, y, z, H, pocketGrids[basisY.index]);
    if (pAir > sdf) sdf = pAir;
  }
  if (basisZ !== null) {
    const pAir = evalPocketCarveAirForBasis(basisZ, x, y, z, H, pocketGrids[basisZ.index]);
    if (pAir > sdf) sdf = pAir;
  }

  for (let ni = 0; ni < nearEdgeCount; ni++) {
    const carveAir = evalCarveEdgeAir(carveEdges[sNearEdgeIdx[ni]], x, y, z, hw, H, bridges);
    if (carveAir > sdf) sdf = carveAir;
  }

  for (let si = 0; si < nearSocketCount; si++) {
    const s = sockets[sNearSocketIdx[si]];
    const dx = x - s.cx;
    const dy = y - s.cy;
    const dz = z - s.cz;
    const h = dx * s.nx + dy * s.ny + dz * s.nz;
    const r = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - h * h));
    const air = Math.min(s.radius + Math.max(0, h - s.flareStartH) * s.coneSlope - r, h - s.depthOffset);
    const smaxAir = -smin(-sdf, -air, 0.009);
    if (smaxAir > sdf) sdf = smaxAir;
  }

  return sdf;
}

function bakeMonolithSdf(
  dcMin: number,
  dcStep: number,
  invDcStep: number,
  halfSlot: number,
  H: number,
  edgeLimit: number,
  cornerLimit: number,
  hw: number,
  bridges: readonly BridgeVolume[],
  carveEdges: readonly CarveEdgeData[],
  pocketGrids: readonly Float32Array[],
  sockets: readonly SocketRecess[]
): void {
  const sdfGrid = sSdfGrid;
  sdfGrid.fill(-0.38);
  const skipCore = halfSlot === 0;
  const coreVertLo = Math.floor(((-0.52) - dcMin) * invDcStep);
  const coreVertHi = Math.ceil((0.52 - dcMin) * invDcStep);
  const vStrideZ = DC_VERTS * DC_VERTS;

  for (let iz = 0; iz < DC_VERTS; iz++) {
    const zInCore = skipCore && iz > coreVertLo && iz < coreVertHi;
    const z = dcMin + iz * dcStep;
    const zQ = halfSlot > 0 ? (z >= 0 ? z - halfSlot : z + halfSlot) : z;
    const az = Math.abs(zQ);
    const gapZ = halfSlot - Math.abs(z);
    const zOff = iz * vStrideZ;
    for (let iy = 0; iy < DC_VERTS; iy++) {
      const yzInCore = zInCore && iy > coreVertLo && iy < coreVertHi;
      const y = dcMin + iy * dcStep;
      const yQ = halfSlot > 0 ? (y >= 0 ? y - halfSlot : y + halfSlot) : y;
      const ay = Math.abs(yQ);
      const gapY = halfSlot - Math.abs(y);
      const gapYZ = gapY > gapZ ? gapY : gapZ;
      const yzOff = zOff + iy * DC_VERTS;
      const dYZ = (ay + az) * INV_SQRT2 - edgeLimit;
      const maxYZ = ay > az ? ay : az;
      for (let ix = 0; ix < DC_VERTS; ix++) {
        if (yzInCore && ix === coreVertLo) {
          ix = coreVertHi - 1;
          continue;
        }
        const x = dcMin + ix * dcStep;
        const xQ = halfSlot > 0 ? (x >= 0 ? x - halfSlot : x + halfSlot) : x;
        const ax = Math.abs(xQ);
        const dFaces = (ax > maxYZ ? ax : maxYZ) - H;
        const dXY = (ax + ay) * INV_SQRT2 - edgeLimit;
        const dXZ = (ax + az) * INV_SQRT2 - edgeLimit;
        const dEdges = dXY > dXZ ? (dXY > dYZ ? dXY : dYZ) : dXZ > dYZ ? dXZ : dYZ;
        const dCorners = (ax + ay + az) * INV_SQRT3 - cornerLimit;
        let sdf =
          dFaces > dEdges ? (dFaces > dCorners ? dFaces : dCorners) : dEdges > dCorners ? dEdges : dCorners;

        if (halfSlot > 0) {
          const gapX = halfSlot - Math.abs(x);
          const gapAir = gapX > gapYZ ? gapX : gapYZ;
          if (gapAir > sdf) sdf = gapAir;
          const vaultAir = evalInnerVaultAir(xQ, yQ, zQ);
          if (vaultAir > sdf) sdf = vaultAir;
        }
        sdfGrid[yzOff + ix] = sdf;
      }
    }
  }

  for (let fi = 0; fi < 6; fi++) {
    const basis = FACE_BASES[fi];
    const grid = pocketGrids[fi];
    const minN = H - 0.17;
    const maxN = H + dcStep + halfSlot;
    const uLim = POCKET_BOUND + dcStep + halfSlot;
    const iu0 = clamp(Math.floor((-uLim - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iu1 = clamp(Math.ceil((uLim - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const in0 =
      basis.nx + basis.ny + basis.nz > 0
        ? clamp(Math.floor((minN - dcMin) * invDcStep), 1, DC_GRID_N - 1)
        : clamp(Math.floor((-maxN - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const in1 =
      basis.nx + basis.ny + basis.nz > 0
        ? clamp(Math.ceil((maxN - dcMin) * invDcStep), 1, DC_GRID_N - 1)
        : clamp(Math.ceil((-minN - dcMin) * invDcStep), 1, DC_GRID_N - 1);

    const uSign = basis.ux + basis.uy + basis.uz;
    const vSign = basis.vx + basis.vy + basis.vz;
    const nSign = basis.nx + basis.ny + basis.nz;
    const sU = basis.ux !== 0 ? 1 : basis.uy !== 0 ? DC_VERTS : vStrideZ;
    const sV = basis.vx !== 0 ? 1 : basis.vy !== 0 ? DC_VERTS : vStrideZ;
    const sN = basis.nx !== 0 ? 1 : basis.ny !== 0 ? DC_VERTS : vStrideZ;

    for (let iv = iu0; iv <= iu1; iv++) {
      const vcRaw = (dcMin + iv * dcStep) * vSign;
      const vCoord = halfSlot > 0 ? (vcRaw >= 0 ? vcRaw - halfSlot : vcRaw + halfSlot) : vcRaw;
      const vOff = iv * sV;
      for (let iu = iu0; iu <= iu1; iu++) {
        const ucRaw = (dcMin + iu * dcStep) * uSign;
        const uCoord = halfSlot > 0 ? (ucRaw >= 0 ? ucRaw - halfSlot : ucRaw + halfSlot) : ucRaw;
        const dClear = samplePocketClearance(grid, uCoord, vCoord);
        if (dClear <= -0.025) continue;
        const baseOff = vOff + iu * sU;
        const dClearWell = dClear - 0.038;
        for (let iDepth = in0; iDepth <= in1; iDepth++) {
          const idx = baseOff + iDepth * sN;
          if (sdfGrid[idx] > 0.025) continue;
          const fcRaw = (dcMin + iDepth * dcStep) * nSign;
          const faceCoord = halfSlot > 0 ? (fcRaw >= 0 ? fcRaw - halfSlot : fcRaw + halfSlot) : fcRaw;
          const depthFromSurface = H - faceCoord;
          const shelfAir = dClear < 0.075 - depthFromSurface ? dClear : 0.075 - depthFromSurface;
          const wellAir = dClearWell < 0.135 - depthFromSurface ? dClearWell : 0.135 - depthFromSurface;
          const pAir = shelfAir > wellAir ? shelfAir : wellAir;
          if (pAir > sdfGrid[idx]) sdfGrid[idx] = pAir;
        }
      }
    }
  }

  for (let ei = 0; ei < carveEdges.length; ei++) {
    const edge = carveEdges[ei];
    const ix0 = clamp(Math.floor((edge.minX - halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iy0 = clamp(Math.floor((edge.minY - halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iz0 = clamp(Math.floor((edge.minZ - halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const ix1 = clamp(Math.ceil((edge.maxX + halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iy1 = clamp(Math.ceil((edge.maxY + halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iz1 = clamp(Math.ceil((edge.maxZ + halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);

    for (let iz = iz0; iz <= iz1; iz++) {
      const zRaw = dcMin + iz * dcStep;
      const z = halfSlot > 0 ? (zRaw >= 0 ? zRaw - halfSlot : zRaw + halfSlot) : zRaw;
      if (z < edge.minZ || z > edge.maxZ) continue;
      const zOff = iz * vStrideZ;
      for (let iy = iy0; iy <= iy1; iy++) {
        const yRaw = dcMin + iy * dcStep;
        const y = halfSlot > 0 ? (yRaw >= 0 ? yRaw - halfSlot : yRaw + halfSlot) : yRaw;
        if (y < edge.minY || y > edge.maxY) continue;
        const yzOff = zOff + iy * DC_VERTS;
        for (let ix = ix0; ix <= ix1; ix++) {
          const idx = yzOff + ix;
          if (sdfGrid[idx] > 0.025) continue;
          const xRaw = dcMin + ix * dcStep;
          const x = halfSlot > 0 ? (xRaw >= 0 ? xRaw - halfSlot : xRaw + halfSlot) : xRaw;
          if (x < edge.minX || x > edge.maxX) continue;
          const carveAir = evalCarveEdgeAir(edge, x, y, z, hw, H, bridges);
          if (carveAir > sdfGrid[idx]) sdfGrid[idx] = carveAir;
        }
      }
    }
  }

  for (let si = 0; si < sockets.length; si++) {
    const s = sockets[si];
    const ix0 = clamp(Math.floor((s.minX - halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iy0 = clamp(Math.floor((s.minY - halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iz0 = clamp(Math.floor((s.minZ - halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const ix1 = clamp(Math.ceil((s.maxX + halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iy1 = clamp(Math.ceil((s.maxY + halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);
    const iz1 = clamp(Math.ceil((s.maxZ + halfSlot - dcMin) * invDcStep), 1, DC_GRID_N - 1);

    for (let iz = iz0; iz <= iz1; iz++) {
      const zRaw = dcMin + iz * dcStep;
      const z = halfSlot > 0 ? (zRaw >= 0 ? zRaw - halfSlot : zRaw + halfSlot) : zRaw;
      const dz = z - s.cz;
      const zOff = iz * vStrideZ;
      for (let iy = iy0; iy <= iy1; iy++) {
        const yRaw = dcMin + iy * dcStep;
        const y = halfSlot > 0 ? (yRaw >= 0 ? yRaw - halfSlot : yRaw + halfSlot) : yRaw;
        const dy = y - s.cy;
        const yzOff = zOff + iy * DC_VERTS;
        for (let ix = ix0; ix <= ix1; ix++) {
          const idx = yzOff + ix;
          if (sdfGrid[idx] > 0.025) continue;
          const xRaw = dcMin + ix * dcStep;
          const x = halfSlot > 0 ? (xRaw >= 0 ? xRaw - halfSlot : xRaw + halfSlot) : xRaw;
          const dx = x - s.cx;
          const h = dx * s.nx + dy * s.ny + dz * s.nz;
          const r = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - h * h));
          const air = Math.min(s.radius + Math.max(0, h - s.flareStartH) * s.coneSlope - r, h - s.depthOffset);
          const smaxAir = -smin(-sdfGrid[idx], -air, 0.009);
          if (smaxAir > sdfGrid[idx]) sdfGrid[idx] = smaxAir;
        }
      }
    }
  }
}

const CREASE_COS = 0.829;

function extractCrossingEdges(dcMin: number, invDcStep: number, skipCore: boolean): void {
  const sdfGrid = sSdfGrid;
  sCellToActive.fill(-1);
  let activeCellCount = 0;
  let crossingEdgeCount = 0;
  const n2 = DC_GRID_N * DC_GRID_N;
  const vStrideZ = DC_VERTS * DC_VERTS;
  const coreLo = Math.floor(((-0.52) - dcMin) * invDcStep);
  const coreHi = Math.ceil((0.52 - dcMin) * invDcStep);

  for (let iz = 1; iz < DC_GRID_N; iz++) {
    const zInCore = skipCore && iz > coreLo && iz < coreHi;
    const zOff = iz * vStrideZ;
    for (let iy = 1; iy < DC_GRID_N; iy++) {
      const yzInCore = zInCore && iy > coreLo && iy < coreHi;
      const rowOff = zOff + iy * DC_VERTS;
      for (let ix = 1; ix < DC_GRID_N; ix++) {
        if (yzInCore && ix === coreLo) {
          ix = coreHi - 1;
          continue;
        }
        const v0 = rowOff + ix;
        const s0 = sdfGrid[v0];
        const neg0 = s0 < 0;

        if (ix < DC_GRID_N - 1) {
          const sx = sdfGrid[v0 + 1];
          if (neg0 !== (sx < 0)) {
            const ei = crossingEdgeCount++;
            sEdgeAxis[ei] = 0;
            sEdgeSignDir[ei] = neg0 ? 1 : -1;
            sEdgeIx[ei] = ix;
            sEdgeIy[ei] = iy;
            sEdgeIz[ei] = iz;
            sEdgeT0[ei] = s0 / (s0 - sx);

            const cid0 = (iz - 1) * n2 + (iy - 1) * DC_GRID_N + ix;
            let a0 = sCellToActive[cid0];
            if (a0 === -1) { a0 = activeCellCount++; sCellToActive[cid0] = a0; const b = a0 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy - 1; sCellBoxCoords[b + 2] = iz - 1; }
            const cid1 = (iz - 1) * n2 + iy * DC_GRID_N + ix;
            let a1 = sCellToActive[cid1];
            if (a1 === -1) { a1 = activeCellCount++; sCellToActive[cid1] = a1; const b = a1 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz - 1; }
            const cid2 = iz * n2 + iy * DC_GRID_N + ix;
            let a2 = sCellToActive[cid2];
            if (a2 === -1) { a2 = activeCellCount++; sCellToActive[cid2] = a2; const b = a2 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz; }
            const cid3 = iz * n2 + (iy - 1) * DC_GRID_N + ix;
            let a3 = sCellToActive[cid3];
            if (a3 === -1) { a3 = activeCellCount++; sCellToActive[cid3] = a3; const b = a3 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy - 1; sCellBoxCoords[b + 2] = iz; }

            sEdgeC0[ei] = a0; sEdgeC1[ei] = a1; sEdgeC2[ei] = a2; sEdgeC3[ei] = a3;
          }
        }

        if (iy < DC_GRID_N - 1) {
          const sy = sdfGrid[v0 + DC_VERTS];
          if (neg0 !== (sy < 0)) {
            const ei = crossingEdgeCount++;
            sEdgeAxis[ei] = 1;
            sEdgeSignDir[ei] = neg0 ? 1 : -1;
            sEdgeIx[ei] = ix;
            sEdgeIy[ei] = iy;
            sEdgeIz[ei] = iz;
            sEdgeT0[ei] = s0 / (s0 - sy);

            const cid0 = (iz - 1) * n2 + iy * DC_GRID_N + (ix - 1);
            let a0 = sCellToActive[cid0];
            if (a0 === -1) { a0 = activeCellCount++; sCellToActive[cid0] = a0; const b = a0 * 3; sCellBoxCoords[b] = ix - 1; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz - 1; }
            const cid1 = iz * n2 + iy * DC_GRID_N + (ix - 1);
            let a1 = sCellToActive[cid1];
            if (a1 === -1) { a1 = activeCellCount++; sCellToActive[cid1] = a1; const b = a1 * 3; sCellBoxCoords[b] = ix - 1; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz; }
            const cid2 = iz * n2 + iy * DC_GRID_N + ix;
            let a2 = sCellToActive[cid2];
            if (a2 === -1) { a2 = activeCellCount++; sCellToActive[cid2] = a2; const b = a2 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz; }
            const cid3 = (iz - 1) * n2 + iy * DC_GRID_N + ix;
            let a3 = sCellToActive[cid3];
            if (a3 === -1) { a3 = activeCellCount++; sCellToActive[cid3] = a3; const b = a3 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz - 1; }

            sEdgeC0[ei] = a0; sEdgeC1[ei] = a1; sEdgeC2[ei] = a2; sEdgeC3[ei] = a3;
          }
        }

        if (iz < DC_GRID_N - 1) {
          const sz = sdfGrid[v0 + vStrideZ];
          if (neg0 !== (sz < 0)) {
            const ei = crossingEdgeCount++;
            sEdgeAxis[ei] = 2;
            sEdgeSignDir[ei] = neg0 ? 1 : -1;
            sEdgeIx[ei] = ix;
            sEdgeIy[ei] = iy;
            sEdgeIz[ei] = iz;
            sEdgeT0[ei] = s0 / (s0 - sz);

            const cid0 = iz * n2 + (iy - 1) * DC_GRID_N + (ix - 1);
            let a0 = sCellToActive[cid0];
            if (a0 === -1) { a0 = activeCellCount++; sCellToActive[cid0] = a0; const b = a0 * 3; sCellBoxCoords[b] = ix - 1; sCellBoxCoords[b + 1] = iy - 1; sCellBoxCoords[b + 2] = iz; }
            const cid1 = iz * n2 + (iy - 1) * DC_GRID_N + ix;
            let a1 = sCellToActive[cid1];
            if (a1 === -1) { a1 = activeCellCount++; sCellToActive[cid1] = a1; const b = a1 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy - 1; sCellBoxCoords[b + 2] = iz; }
            const cid2 = iz * n2 + iy * DC_GRID_N + ix;
            let a2 = sCellToActive[cid2];
            if (a2 === -1) { a2 = activeCellCount++; sCellToActive[cid2] = a2; const b = a2 * 3; sCellBoxCoords[b] = ix; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz; }
            const cid3 = iz * n2 + iy * DC_GRID_N + (ix - 1);
            let a3 = sCellToActive[cid3];
            if (a3 === -1) { a3 = activeCellCount++; sCellToActive[cid3] = a3; const b = a3 * 3; sCellBoxCoords[b] = ix - 1; sCellBoxCoords[b + 1] = iy; sCellBoxCoords[b + 2] = iz; }

            sEdgeC0[ei] = a0; sEdgeC1[ei] = a1; sEdgeC2[ei] = a2; sEdgeC3[ei] = a3;
          }
        }
      }
    }
  }

  sLastActiveCells = activeCellCount;
  sLastCrossingEdges = crossingEdgeCount;
}

function solveQefCells(
  dcMin: number,
  dcStep: number,
  halfSlot: number,
  activeCellCount: number,
  crossingEdgeCount: number,
  H: number,
  edgeLimit: number,
  cornerLimit: number,
  hw: number,
  bridges: readonly BridgeVolume[],
  carveEdges: readonly CarveEdgeData[],
  pocketGrids: readonly Float32Array[],
  sockets: readonly SocketRecess[]
): void {
  const qefBuf = sQefBuf;
  qefBuf.fill(0, 0, activeCellCount * 13);
  const n2 = EDGE_BIN_N * EDGE_BIN_N;
  const invBinStep = EDGE_BIN_N / DC_SPAN;
  const cellPad = dcStep * 0.12;
  const h = 0.002 * INV_SQRT3;
  const hPad = h + 0.0002;
  const inv4h = 0.25 / h;
  const pocketFaceThresh = H - 0.165 - hPad;
  const pocketUvMax = POCKET_BOUND + 0.01 + hPad;

  for (let ei = 0; ei < crossingEdgeCount; ei++) {
    const ax = sEdgeAxis[ei];
    const ix = sEdgeIx[ei];
    const iy = sEdgeIy[ei];
    const iz = sEdgeIz[ei];
    const t = sEdgeT0[ei];

    const px = dcMin + (ix + (ax === 0 ? t : 0)) * dcStep;
    const py = dcMin + (iy + (ax === 1 ? t : 0)) * dcStep;
    const pz = dcMin + (iz + (ax === 2 ? t : 0)) * dcStep;

    const qx = halfSlot > 0 ? (px >= 0 ? px - halfSlot : px + halfSlot) : px;
    const qy = halfSlot > 0 ? (py >= 0 ? py - halfSlot : py + halfSlot) : py;
    const qz = halfSlot > 0 ? (pz >= 0 ? pz - halfSlot : pz + halfSlot) : pz;

    const bxIdx = clamp(Math.floor((qx - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const byIdx = clamp(Math.floor((qy - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const bzIdx = clamp(Math.floor((qz - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
    const binId = bzIdx * n2 + byIdx * EDGE_BIN_N + bxIdx;
    const binStart = sBinOffsets[binId];
    const binEnd = sBinOffsets[binId + 1];

    let nearEdgeCount = 0;
    for (let bi = binStart; bi < binEnd; bi++) {
      const eIdx = sBinEdgeList[bi];
      const edge = carveEdges[eIdx];
      if (
        qx >= edge.minX - hPad &&
        qx <= edge.maxX + hPad &&
        qy >= edge.minY - hPad &&
        qy <= edge.maxY + hPad &&
        qz >= edge.minZ - hPad &&
        qz <= edge.maxZ + hPad
      ) {
        if (nearEdgeCount < 64) sNearEdgeIdx[nearEdgeCount++] = eIdx;
      }
    }

    let nearSocketCount = 0;
    for (let si = 0; si < sockets.length; si++) {
      const s = sockets[si];
      if (
        qx >= s.minX - hPad &&
        qx <= s.maxX + hPad &&
        qy >= s.minY - hPad &&
        qy <= s.maxY + hPad &&
        qz >= s.minZ - hPad &&
        qz <= s.maxZ + hPad
      ) {
        if (nearSocketCount < 8) sNearSocketIdx[nearSocketCount++] = si;
      }
    }

    const apx = Math.abs(qx);
    const apy = Math.abs(qy);
    const apz = Math.abs(qz);
    const basisX =
      apx >= pocketFaceThresh && apy <= pocketUvMax && apz <= pocketUvMax
        ? qx >= 0
          ? FACE_BASES[0]
          : FACE_BASES[1]
        : null;
    const basisY =
      apy >= pocketFaceThresh && apx <= pocketUvMax && apz <= pocketUvMax
        ? qy >= 0
          ? FACE_BASES[2]
          : FACE_BASES[3]
        : null;
    const basisZ =
      apz >= pocketFaceThresh && apx <= pocketUvMax && apy <= pocketUvMax
        ? qz >= 0
          ? FACE_BASES[4]
          : FACE_BASES[5]
        : null;

    let f0: number;
    let f1: number;
    let f2: number;
    let f3: number;
    if (
      halfSlot === 0 &&
      nearEdgeCount === 0 &&
      nearSocketCount === 0 &&
      basisX === null &&
      basisY === null &&
      basisZ === null
    ) {
      f0 = evalOuterCubeSdf(px + h, py + h, pz + h, H, edgeLimit, cornerLimit);
      f1 = evalOuterCubeSdf(px + h, py - h, pz - h, H, edgeLimit, cornerLimit);
      f2 = evalOuterCubeSdf(px - h, py + h, pz - h, H, edgeLimit, cornerLimit);
      f3 = evalOuterCubeSdf(px - h, py - h, pz + h, H, edgeLimit, cornerLimit);
    } else {
      f0 = evalExactLocalFiltered(
        px + h, py + h, pz + h,
        H, edgeLimit, cornerLimit, hw, halfSlot, bridges, carveEdges, pocketGrids, sockets,
        basisX, basisY, basisZ, nearEdgeCount, nearSocketCount
      );
      f1 = evalExactLocalFiltered(
        px + h, py - h, pz - h,
        H, edgeLimit, cornerLimit, hw, halfSlot, bridges, carveEdges, pocketGrids, sockets,
        basisX, basisY, basisZ, nearEdgeCount, nearSocketCount
      );
      f2 = evalExactLocalFiltered(
        px - h, py + h, pz - h,
        H, edgeLimit, cornerLimit, hw, halfSlot, bridges, carveEdges, pocketGrids, sockets,
        basisX, basisY, basisZ, nearEdgeCount, nearSocketCount
      );
      f3 = evalExactLocalFiltered(
        px - h, py - h, pz + h,
        H, edgeLimit, cornerLimit, hw, halfSlot, bridges, carveEdges, pocketGrids, sockets,
        basisX, basisY, basisZ, nearEdgeCount, nearSocketCount
      );
    }

    let gx = (f0 + f1 - f2 - f3) * inv4h;
    let gy = (f0 - f1 + f2 - f3) * inv4h;
    let gz = (f0 - f1 - f2 + f3) * inv4h;
    const gLen = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (gLen > 1e-7) {
      const invG = 1 / gLen;
      gx *= invG;
      gy *= invG;
      gz *= invG;
    } else {
      gx = ax === 0 ? sEdgeSignDir[ei] : 0;
      gy = ax === 1 ? sEdgeSignDir[ei] : 0;
      gz = ax === 2 ? sEdgeSignDir[ei] : 0;
    }

    const dDot = gx * px + gy * py + gz * pz;
    const gxx = gx * gx;
    const gxy = gx * gy;
    const gxz = gx * gz;
    const gyy = gy * gy;
    const gyz = gy * gz;
    const gzz = gz * gz;
    const bx = gx * dDot;
    const by = gy * dDot;
    const bz = gz * dDot;

    let q = sEdgeC0[ei] * 13;
    qefBuf[q] += gxx; qefBuf[q + 1] += gxy; qefBuf[q + 2] += gxz;
    qefBuf[q + 3] += gyy; qefBuf[q + 4] += gyz; qefBuf[q + 5] += gzz;
    qefBuf[q + 6] += bx;  qefBuf[q + 7] += by;  qefBuf[q + 8] += bz;
    qefBuf[q + 9] += px;  qefBuf[q + 10] += py; qefBuf[q + 11] += pz;
    qefBuf[q + 12] += 1;

    q = sEdgeC1[ei] * 13;
    qefBuf[q] += gxx; qefBuf[q + 1] += gxy; qefBuf[q + 2] += gxz;
    qefBuf[q + 3] += gyy; qefBuf[q + 4] += gyz; qefBuf[q + 5] += gzz;
    qefBuf[q + 6] += bx;  qefBuf[q + 7] += by;  qefBuf[q + 8] += bz;
    qefBuf[q + 9] += px;  qefBuf[q + 10] += py; qefBuf[q + 11] += pz;
    qefBuf[q + 12] += 1;

    q = sEdgeC2[ei] * 13;
    qefBuf[q] += gxx; qefBuf[q + 1] += gxy; qefBuf[q + 2] += gxz;
    qefBuf[q + 3] += gyy; qefBuf[q + 4] += gyz; qefBuf[q + 5] += gzz;
    qefBuf[q + 6] += bx;  qefBuf[q + 7] += by;  qefBuf[q + 8] += bz;
    qefBuf[q + 9] += px;  qefBuf[q + 10] += py; qefBuf[q + 11] += pz;
    qefBuf[q + 12] += 1;

    q = sEdgeC3[ei] * 13;
    qefBuf[q] += gxx; qefBuf[q + 1] += gxy; qefBuf[q + 2] += gxz;
    qefBuf[q + 3] += gyy; qefBuf[q + 4] += gyz; qefBuf[q + 5] += gzz;
    qefBuf[q + 6] += bx;  qefBuf[q + 7] += by;  qefBuf[q + 8] += bz;
    qefBuf[q + 9] += px;  qefBuf[q + 10] += py; qefBuf[q + 11] += pz;
    qefBuf[q + 12] += 1;
  }

  const cellVertPos = sCellVertPos;
  for (let aid = 0; aid < activeCellCount; aid++) {
    const q = aid * 13;
    const count = Math.max(1, qefBuf[q + 12]);
    const invC = 1 / count;
    const cb = aid * 3;
    const minX = dcMin + sCellBoxCoords[cb] * dcStep;
    const minY = dcMin + sCellBoxCoords[cb + 1] * dcStep;
    const minZ = dcMin + sCellBoxCoords[cb + 2] * dcStep;
    solveQef3x3(
      qefBuf[q], qefBuf[q + 1], qefBuf[q + 2],
      qefBuf[q + 3], qefBuf[q + 4], qefBuf[q + 5],
      qefBuf[q + 6], qefBuf[q + 7], qefBuf[q + 8],
      qefBuf[q + 9] * invC, qefBuf[q + 10] * invC, qefBuf[q + 11] * invC,
      minX + cellPad, minY + cellPad, minZ + cellPad,
      minX + dcStep - cellPad, minY + dcStep - cellPad, minZ + dcStep - cellPad,
      count, cellVertPos, aid * 3
    );
  }
}

function emitDualQuadsAndCsr(activeCellCount: number, crossingEdgeCount: number): number {
  const cellVertPos = sCellVertPos;
  const triCellIndices = sTriCellIndices;
  const triFaceNormals = sTriFaceNormals;
  let triCount = 0;
  sVertTriCounts.fill(0, 0, activeCellCount);
  sAvgCellNorm.fill(0, 0, activeCellCount * 3);

  for (let ei = 0; ei < crossingEdgeCount; ei++) {
    const c0 = sEdgeC0[ei];
    const c1 = sEdgeC1[ei];
    const c2 = sEdgeC2[ei];
    const c3 = sEdgeC3[ei];
    const dir = sEdgeSignDir[ei];

    const qa = c0;
    const qb = dir > 0 ? c1 : c3;
    const qc = c2;
    const qd = dir > 0 ? c3 : c1;

    const p0x = cellVertPos[qa * 3], p0y = cellVertPos[qa * 3 + 1], p0z = cellVertPos[qa * 3 + 2];
    const p1x = cellVertPos[qb * 3], p1y = cellVertPos[qb * 3 + 1], p1z = cellVertPos[qb * 3 + 2];
    const p2x = cellVertPos[qc * 3], p2y = cellVertPos[qc * 3 + 1], p2z = cellVertPos[qc * 3 + 2];
    const p3x = cellVertPos[qd * 3], p3y = cellVertPos[qd * 3 + 1], p3z = cellVertPos[qd * 3 + 2];

    const d02 = (p2x - p0x) * (p2x - p0x) + (p2y - p0y) * (p2y - p0y) + (p2z - p0z) * (p2z - p0z);
    const d13 = (p3x - p1x) * (p3x - p1x) + (p3y - p1y) * (p3y - p1y) + (p3z - p1z) * (p3z - p1z);

    let t0a = qa, t0b = qb, t0c = qc;
    let t1a = qa, t1b = qc, t1c = qd;
    if (d02 > d13) {
      t0a = qa; t0b = qb; t0c = qd;
      t1a = qb; t1b = qc; t1c = qd;
    }

    {
      const ax = cellVertPos[t0a * 3], ay = cellVertPos[t0a * 3 + 1], az = cellVertPos[t0a * 3 + 2];
      const e1x = cellVertPos[t0b * 3] - ax, e1y = cellVertPos[t0b * 3 + 1] - ay, e1z = cellVertPos[t0b * 3 + 2] - az;
      const e2x = cellVertPos[t0c * 3] - ax, e2y = cellVertPos[t0c * 3 + 1] - ay, e2z = cellVertPos[t0c * 3 + 2] - az;
      const cx = e1y * e2z - e1z * e2y;
      const cy = e1z * e2x - e1x * e2z;
      const cz = e1x * e2y - e1y * e2x;
      const len = Math.sqrt(cx * cx + cy * cy + cz * cz);
      if (len > 1e-12) {
        const invL = 1 / len;
        const fnx = cx * invL, fny = cy * invL, fnz = cz * invL;
        const ti = triCount++;
        const t3 = ti * 3;
        triCellIndices[t3] = t0a; triCellIndices[t3 + 1] = t0b; triCellIndices[t3 + 2] = t0c;
        triFaceNormals[t3] = fnx; triFaceNormals[t3 + 1] = fny; triFaceNormals[t3 + 2] = fnz;
        sVertTriCounts[t0a]++; sVertTriCounts[t0b]++; sVertTriCounts[t0c]++;
        sAvgCellNorm[t0a * 3] += fnx; sAvgCellNorm[t0a * 3 + 1] += fny; sAvgCellNorm[t0a * 3 + 2] += fnz;
        sAvgCellNorm[t0b * 3] += fnx; sAvgCellNorm[t0b * 3 + 1] += fny; sAvgCellNorm[t0b * 3 + 2] += fnz;
        sAvgCellNorm[t0c * 3] += fnx; sAvgCellNorm[t0c * 3 + 1] += fny; sAvgCellNorm[t0c * 3 + 2] += fnz;
      }
    }

    {
      const ax = cellVertPos[t1a * 3], ay = cellVertPos[t1a * 3 + 1], az = cellVertPos[t1a * 3 + 2];
      const e1x = cellVertPos[t1b * 3] - ax, e1y = cellVertPos[t1b * 3 + 1] - ay, e1z = cellVertPos[t1b * 3 + 2] - az;
      const e2x = cellVertPos[t1c * 3] - ax, e2y = cellVertPos[t1c * 3 + 1] - ay, e2z = cellVertPos[t1c * 3 + 2] - az;
      const cx = e1y * e2z - e1z * e2y;
      const cy = e1z * e2x - e1x * e2z;
      const cz = e1x * e2y - e1y * e2x;
      const len = Math.sqrt(cx * cx + cy * cy + cz * cz);
      if (len > 1e-12) {
        const invL = 1 / len;
        const fnx = cx * invL, fny = cy * invL, fnz = cz * invL;
        const ti = triCount++;
        const t3 = ti * 3;
        triCellIndices[t3] = t1a; triCellIndices[t3 + 1] = t1b; triCellIndices[t3 + 2] = t1c;
        triFaceNormals[t3] = fnx; triFaceNormals[t3 + 1] = fny; triFaceNormals[t3 + 2] = fnz;
        sVertTriCounts[t1a]++; sVertTriCounts[t1b]++; sVertTriCounts[t1c]++;
        sAvgCellNorm[t1a * 3] += fnx; sAvgCellNorm[t1a * 3 + 1] += fny; sAvgCellNorm[t1a * 3 + 2] += fnz;
        sAvgCellNorm[t1b * 3] += fnx; sAvgCellNorm[t1b * 3 + 1] += fny; sAvgCellNorm[t1b * 3 + 2] += fnz;
        sAvgCellNorm[t1c * 3] += fnx; sAvgCellNorm[t1c * 3 + 1] += fny; sAvgCellNorm[t1c * 3 + 2] += fnz;
      }
    }
  }

  sVertTriOffsets[0] = 0;
  for (let i = 0; i < activeCellCount; i++) {
    sVertTriOffsets[i + 1] = sVertTriOffsets[i] + sVertTriCounts[i];
  }
  sVertTriCounts.fill(0, 0, activeCellCount);
  for (let ti = 0; ti < triCount; ti++) {
    const t3 = ti * 3;
    const a = triCellIndices[t3];
    const b = triCellIndices[t3 + 1];
    const c = triCellIndices[t3 + 2];
    sVertTriList[sVertTriOffsets[a] + sVertTriCounts[a]++] = ti;
    sVertTriList[sVertTriOffsets[b] + sVertTriCounts[b]++] = ti;
    sVertTriList[sVertTriOffsets[c] + sVertTriCounts[c]++] = ti;
  }

  return triCount;
}

function computeCellAoAndZones(
  dcMin: number,
  invDcStep: number,
  halfSlot: number,
  activeCellCount: number,
  H: number,
  surfaceDepth: number,
  underpassDepth: number,
  hw: number,
  edgeLimit: number,
  cornerLimit: number,
  pocketGrids: readonly Float32Array[],
  bridges: readonly BridgeVolume[],
  carveEdges: readonly CarveEdgeData[]
): void {
  const cellVertPos = sCellVertPos;
  const triFaceNormals = sTriFaceNormals;
  const vStrideZ = DC_VERTS * DC_VERTS;
  const bridgeCount = bridges.length;
  const n2 = EDGE_BIN_N * EDGE_BIN_N;
  const invBinStep = EDGE_BIN_N / DC_SPAN;
  sCellIsSharp.fill(0, 0, activeCellCount);

  for (let aid = 0; aid < activeCellCount; aid++) {
    const p3 = aid * 3;
    const gpx = cellVertPos[p3];
    const gpy = cellVertPos[p3 + 1];
    const gpz = cellVertPos[p3 + 2];

    const px = halfSlot > 0 ? (gpx >= 0 ? gpx - halfSlot : gpx + halfSlot) : gpx;
    const py = halfSlot > 0 ? (gpy >= 0 ? gpy - halfSlot : gpy + halfSlot) : gpy;
    const pz = halfSlot > 0 ? (gpz >= 0 ? gpz - halfSlot : gpz + halfSlot) : gpz;

    let nx = sAvgCellNorm[p3];
    let ny = sAvgCellNorm[p3 + 1];
    let nz = sAvgCellNorm[p3 + 2];
    const nl = Math.max(1e-6, Math.sqrt(nx * nx + ny * ny + nz * nz));
    const invNl = 1 / nl;
    nx *= invNl;
    ny *= invNl;
    nz *= invNl;
    sAvgCellNorm[p3] = nx;
    sAvgCellNorm[p3 + 1] = ny;
    sAvgCellNorm[p3 + 2] = nz;

    const start = sVertTriOffsets[aid];
    const end = sVertTriOffsets[aid + 1];
    for (let j = start; j < end; j++) {
      const n3 = sVertTriList[j] * 3;
      if (nx * triFaceNormals[n3] + ny * triFaceNormals[n3 + 1] + nz * triFaceNormals[n3 + 2] < CREASE_COS) {
        sCellIsSharp[aid] = 1;
        break;
      }
    }

    const dNear = sampleGridSdfFast(vStrideZ, dcMin, invDcStep, gpx + nx * 0.022, gpy + ny * 0.022, gpz + nz * 0.022);
    const d1 = sampleGridSdfFast(vStrideZ, dcMin, invDcStep, gpx + nx * 0.045, gpy + ny * 0.045, gpz + nz * 0.045);
    const d2 = sampleGridSdfFast(vStrideZ, dcMin, invDcStep, gpx + nx * 0.125, gpy + ny * 0.125, gpz + nz * 0.125);
    const occ =
      Math.max(0, 0.022 - dNear) * 20.0 +
      Math.max(0, 0.045 - d1) * 12.0 +
      Math.max(0, 0.125 - d2) * 4.2;
    const ao = clamp(1.0 - occ * 0.78, 0.10, 1.0);
    sCellAo[aid] = ao;

    // Convex edge thickness taps for chiseled limestone ridge & bevel arris wear
    const tIn1 = sampleGridSdfFast(vStrideZ, dcMin, invDcStep, gpx - nx * 0.024, gpy - ny * 0.024, gpz - nz * 0.024);
    const tIn2 = sampleGridSdfFast(vStrideZ, dcMin, invDcStep, gpx - nx * 0.046, gpy - ny * 0.046, gpz - nz * 0.046);
    const thin1 = clamp((tIn1 + 0.021) * 48.0, 0, 1);
    const thin2 = clamp((tIn2 + 0.041) * 34.0, 0, 1);
    sCellEdgeWear[aid] = clamp((thin1 * 0.5 + thin2 * 0.5) * (ao > 0.45 ? 1.0 : 0.25), 0, 1);

    const ax = Math.abs(px);
    const ay = Math.abs(py);
    const az = Math.abs(pz);
    const dFaces = Math.max(ax, ay, az) - H;
    const dEdges =
      Math.max((ax + ay) * INV_SQRT2, (ay + az) * INV_SQRT2, (ax + az) * INV_SQRT2) - edgeLimit;
    const dCorners = (ax + ay + az) * INV_SQRT3 - cornerLimit;
    const depthIn = -Math.max(dFaces, dEdges, dCorners);

    const isCleavagePlane = halfSlot > 0 && Math.min(ax, ay, az) < 0.015;
    const innerVaultAir = halfSlot > 0 ? evalInnerVaultAir(px, py, pz) : -1.0;
    const isInnerVault = halfSlot > 0 && innerVaultAir > -0.028;

    let basalt = 0;
    let patina = 0;
    let floorMask = 0;
    const fNoise = organicField3D(px, py, pz);

    if (isCleavagePlane || isInnerVault) {
      basalt = 0.88 + 0.12 * fNoise;
      if (isInnerVault && fNoise > 0.48) {
        patina = clamp((fNoise - 0.48) * 1.9, 0.2, 0.85);
      }
      sCellBasalt[aid] = clamp(basalt, 0, 1);
      sCellPatina[aid] = clamp(patina, 0, 1);
      sCellFloorMask[aid] = 0;
      continue;
    }

    if (depthIn > 0.04) {
      const domBasis = dominantFaceBasis(px, py, pz);
      const uC = px * domBasis.ux + py * domBasis.uy + pz * domBasis.uz;
      const vC = px * domBasis.vx + py * domBasis.vy + pz * domBasis.vz;
      const pocketClr = samplePocketClearance(pocketGrids[domBasis.index], uC, vC);

      let isBridgeArchBody = false;
      let isUnderpassSoffit = false;
      for (let bi = 0; bi < bridgeCount; bi++) {
        const br = bridges[bi];
        const rx = px - br.cx;
        const ry = py - br.cy;
        const rz = pz - br.cz;
        const aUnder = Math.abs(rx * br.underX + ry * br.underY + rz * br.underZ);
        const aOver = Math.abs(rx * br.overX + ry * br.overY + rz * br.overZ);
        if (aUnder <= br.halfSpan + 0.02 && aOver <= hw + 0.06) {
          const dotFace = nx * br.face.nx + ny * br.face.ny + nz * br.face.nz;
          if (depthIn < underpassDepth - 0.035) {
            isBridgeArchBody = true;
            if (dotFace < -0.55 && aUnder <= br.halfSpan - 0.012 && aOver <= hw + 0.02) {
              isUnderpassSoffit = true;
            }
          }
        }
      }

      const pocketD = pocketClr - 0.056;
      if (pocketD > -0.004 && depthIn > 0.042) {
        const depthFade = clamp((depthIn - 0.040) / 0.014, 0, 1);
        const clrFade = clamp((pocketD + 0.004) / 0.014, 0, 1);
        basalt = Math.max(basalt, depthFade * clrFade);
      }
      if (!isBridgeArchBody && depthIn > surfaceDepth + 0.055) {
        const s = clamp((depthIn - (surfaceDepth + 0.048) + (fNoise - 0.5) * 0.018) / 0.032, 0, 1);
        basalt = Math.max(basalt, s);
      }

      // Compute strict corridor floorMask first so wet Yellow Ochre trail stays off walls and floor suppresses cyan patina
      if (depthIn >= 0.085 && !isUnderpassSoffit) {
        const bxIdx = clamp(Math.floor((px - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
        const byIdx = clamp(Math.floor((py - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
        const bzIdx = clamp(Math.floor((pz - DC_MIN) * invBinStep), 0, EDGE_BIN_N - 1);
        const binId = bzIdx * n2 + byIdx * EDGE_BIN_N + bxIdx;
        const binStart = sBinOffsets[binId];
        const binEnd = sBinOffsets[binId + 1];
        for (let bi = binStart; bi < binEnd; bi++) {
          const edge = carveEdges[sBinEdgeList[bi]];
          if (
            px < edge.minX || px > edge.maxX ||
            py < edge.minY || py > edge.maxY ||
            pz < edge.minZ || pz > edge.maxZ
          ) {
            continue;
          }
          const c = evalCarveEdgeAirAndCoords(edge, px, py, pz, hw, H, bridges);
          if (c.h >= -0.025 && c.h <= 0.028 && c.flatDist <= hw + 0.015) {
            const hMask = clamp(1.0 - (c.h - 0.015) / 0.013, 0, 1);
            const wMask = clamp(1.0 - (c.flatDist - hw) / 0.015, 0, 1);
            const fM = hMask * wMask;
            if (fM > floorMask) floorMask = fM;
          }
        }
      }

      const floorSuppress = 1.0 - clamp(floorMask * 1.35, 0, 1);
      const recessFactor = clamp((depthIn - 0.075) / 0.065, 0, 1);
      const cornerAoMask = clamp((0.46 - ao) / 0.20, 0, 1);
      const seamMask = ao < 0.54 ? clamp((0.018 - dNear) / 0.007, 0, 1) * 0.65 : 0;
      const organicVein = clamp(fNoise * 1.45 - 0.22, 0, 1);
      patina = Math.max(cornerAoMask, seamMask) * recessFactor * organicVein * 0.52 * floorSuppress;
      if (ao >= 0.54 && !isUnderpassSoffit) {
        patina = Math.min(patina, 0.03);
      }
      if (isUnderpassSoffit) {
        patina = Math.max(patina, 0.16 + 0.22 * organicVein);
      }
    }

    sCellBasalt[aid] = clamp(basalt, 0, 1);
    sCellPatina[aid] = clamp(patina, 0, 1);
    sCellFloorMask[aid] = clamp(floorMask, 0, 1);
  }

  // Two passes of crease-aware CSR neighbor averaging eliminate voxel diagonal stair-steps
  for (let pass = 0; pass < 2; pass++) {
    for (let aid = 0; aid < activeCellCount; aid++) {
      const start = sVertTriOffsets[aid];
      const end = sVertTriOffsets[aid + 1];
      const nx = sAvgCellNorm[aid * 3];
      const ny = sAvgCellNorm[aid * 3 + 1];
      const nz = sAvgCellNorm[aid * 3 + 2];

      let wSum = 2.6;
      let aoSum = sCellAo[aid] * 2.6;
      let basSum = sCellBasalt[aid] * 2.6;
      let patSum = sCellPatina[aid] * 2.6;
      let wearSum = sCellEdgeWear[aid] * 2.6;

      for (let j = start; j < end; j++) {
        const t3 = sVertTriList[j] * 3;
        for (let k = 0; k < 3; k++) {
          const nb = sTriCellIndices[t3 + k];
          if (nb === aid) continue;
          const dotN =
            nx * sAvgCellNorm[nb * 3] +
            ny * sAvgCellNorm[nb * 3 + 1] +
            nz * sAvgCellNorm[nb * 3 + 2];
          const w = dotN > 0.65 ? dotN : 0.10;
          wSum += w;
          aoSum += sCellAo[nb] * w;
          basSum += sCellBasalt[nb] * w;
          patSum += sCellPatina[nb] * w;
          wearSum += sCellEdgeWear[nb] * w;
        }
      }
      const inv = 1 / wSum;
      sCellAoTmp[aid] = aoSum * inv;
      sCellBasaltTmp[aid] = basSum * inv;
      sCellPatinaTmp[aid] = patSum * inv;
      sCellEdgeWearTmp[aid] = wearSum * inv;
    }
    sCellAo.set(sCellAoTmp.subarray(0, activeCellCount), 0);
    sCellBasalt.set(sCellBasaltTmp.subarray(0, activeCellCount), 0);
    sCellPatina.set(sCellPatinaTmp.subarray(0, activeCellCount), 0);
    sCellEdgeWear.set(sCellEdgeWearTmp.subarray(0, activeCellCount), 0);
  }
}

function writeVertexToBuffers(
  vIdx: number,
  aid: number,
  fnx: number,
  fny: number,
  fnz: number,
  ti: number,
  halfSlot: number,
  faceIdx: number,
  bux: number,
  buy: number,
  buz: number,
  bvx: number,
  bvy: number,
  bvz: number,
  uvPadLo: number,
  uvPadHi: number,
  inv2H: number,
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array,
  materialZone: Float32Array,
  stoneDetail: Float32Array
): void {
  const gx = sCellVertPos[aid * 3];
  const gy = sCellVertPos[aid * 3 + 1];
  const gz = sCellVertPos[aid * 3 + 2];
  const px = halfSlot > 0 ? (gx >= 0 ? gx - halfSlot : gx + halfSlot) : gx;
  const py = halfSlot > 0 ? (gy >= 0 ? gy - halfSlot : gy + halfSlot) : gy;
  const pz = halfSlot > 0 ? (gz >= 0 ? gz - halfSlot : gz + halfSlot) : gz;

  positions[vIdx * 3] = px;
  positions[vIdx * 3 + 1] = py;
  positions[vIdx * 3 + 2] = pz;

  if (sCellIsSharp[aid] === 0) {
    normals[vIdx * 3] = sAvgCellNorm[aid * 3];
    normals[vIdx * 3 + 1] = sAvgCellNorm[aid * 3 + 1];
    normals[vIdx * 3 + 2] = sAvgCellNorm[aid * 3 + 2];
  } else {
    let nx = fnx;
    let ny = fny;
    let nz = fnz;
    const start = sVertTriOffsets[aid];
    const end = sVertTriOffsets[aid + 1];
    for (let j = start; j < end; j++) {
      const nti = sVertTriList[j];
      if (nti === ti) continue;
      const n3 = nti * 3;
      const onx = sTriFaceNormals[n3];
      const ony = sTriFaceNormals[n3 + 1];
      const onz = sTriFaceNormals[n3 + 2];
      if (fnx * onx + fny * ony + fnz * onz >= CREASE_COS) {
        nx += onx;
        ny += ony;
        nz += onz;
      }
    }
    const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nLen > 1e-6 && nx * fnx + ny * fny + nz * fnz > 0) {
      const invL = 1 / nLen;
      normals[vIdx * 3] = nx * invL;
      normals[vIdx * 3 + 1] = ny * invL;
      normals[vIdx * 3 + 2] = nz * invL;
    } else {
      normals[vIdx * 3] = fnx;
      normals[vIdx * 3 + 1] = fny;
      normals[vIdx * 3 + 2] = fnz;
    }
  }

  const u01 = clamp(0.5 + (px * bux + py * buy + pz * buz) * inv2H, uvPadLo, uvPadHi);
  const v01 = clamp(0.5 + (px * bvx + py * bvy + pz * bvz) * inv2H, uvPadLo, uvPadHi);
  uvs[vIdx * 2] = (faceIdx + u01) / 6;
  uvs[vIdx * 2 + 1] = v01;

  const bWeight = sCellBasalt[aid];
  materialZone[vIdx * 4] = 1 - bWeight;
  materialZone[vIdx * 4 + 1] = bWeight;
  materialZone[vIdx * 4 + 2] = sCellAo[aid];
  materialZone[vIdx * 4 + 3] = sCellPatina[aid];

  stoneDetail[vIdx * 2] = sCellFloorMask[aid];
  stoneDetail[vIdx * 2 + 1] = sCellEdgeWear[aid];
}

function finalizeGeometryBounds(geom: THREE.BufferGeometry, positions: Float32Array): THREE.BufferGeometry {
  let bMinX = Infinity, bMinY = Infinity, bMinZ = Infinity;
  let bMaxX = -Infinity, bMaxY = -Infinity, bMaxZ = -Infinity;
  let maxRSq = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < bMinX) bMinX = x;
    if (y < bMinY) bMinY = y;
    if (z < bMinZ) bMinZ = z;
    if (x > bMaxX) bMaxX = x;
    if (y > bMaxY) bMaxY = y;
    if (z > bMaxZ) bMaxZ = z;
    const rSq = x * x + y * y + z * z;
    if (rSq > maxRSq) maxRSq = rSq;
  }
  geom.boundingBox = new THREE.Box3(
    new THREE.Vector3(bMinX, bMinY, bMinZ),
    new THREE.Vector3(bMaxX, bMaxY, bMaxZ)
  );
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), Math.sqrt(maxRSq));
  return geom;
}

function emitFinalMonolithBuffers(
  triCount: number,
  H: number
): THREE.BufferGeometry {
  const cellVertPos = sCellVertPos;
  const triCellIndices = sTriCellIndices;
  const triFaceNormals = sTriFaceNormals;
  const outVerts = triCount * 3;
  const positions = new Float32Array(outVerts * 3);
  const normals = new Float32Array(outVerts * 3);
  const uvs = new Float32Array(outVerts * 2);
  const materialZone = new Float32Array(outVerts * 4);
  const stoneDetail = new Float32Array(outVerts * 2);

  const uvPadLo = 0.5 / WET_ATLAS_FACE_SIZE;
  const uvPadHi = (WET_ATLAS_FACE_SIZE - 0.5) / WET_ATLAS_FACE_SIZE;
  const inv2H = 0.5 / H;

  for (let ti = 0; ti < triCount; ti++) {
    const t3 = ti * 3;
    const cA = triCellIndices[t3];
    const cB = triCellIndices[t3 + 1];
    const cC = triCellIndices[t3 + 2];

    const fnx = triFaceNormals[t3];
    const fny = triFaceNormals[t3 + 1];
    const fnz = triFaceNormals[t3 + 2];

    const cx = (cellVertPos[cA * 3] + cellVertPos[cB * 3] + cellVertPos[cC * 3]) * (1 / 3);
    const cy = (cellVertPos[cA * 3 + 1] + cellVertPos[cB * 3 + 1] + cellVertPos[cC * 3 + 1]) * (1 / 3);
    const cz = (cellVertPos[cA * 3 + 2] + cellVertPos[cB * 3 + 2] + cellVertPos[cC * 3 + 2]) * (1 / 3);
    const basis = dominantFaceBasis(cx, cy, cz);

    for (let k = 0; k < 3; k++) {
      const aid = k === 0 ? cA : k === 1 ? cB : cC;
      writeVertexToBuffers(
        t3 + k,
        aid,
        fnx,
        fny,
        fnz,
        ti,
        0,
        basis.index,
        basis.ux,
        basis.uy,
        basis.uz,
        basis.vx,
        basis.vy,
        basis.vz,
        uvPadLo,
        uvPadHi,
        inv2H,
        positions,
        normals,
        uvs,
        materialZone,
        stoneDetail
      );
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aMaterialZone', new THREE.BufferAttribute(materialZone, 4));
  geom.setAttribute('aStoneDetail', new THREE.BufferAttribute(stoneDetail, 2));
  return finalizeGeometryBounds(geom, positions);
}

function emitOctantMonolithRawBuffers(
  triCount: number,
  H: number,
  halfSlot: number
): RawStoneMeshBuffers[] {
  const cellVertPos = sCellVertPos;
  const triCellIndices = sTriCellIndices;
  const triFaceNormals = sTriFaceNormals;
  sOctantTriCounts.fill(0);

  for (let ti = 0; ti < triCount; ti++) {
    const t3 = ti * 3;
    const cA = triCellIndices[t3];
    const cB = triCellIndices[t3 + 1];
    const cC = triCellIndices[t3 + 2];
    const cx = cellVertPos[cA * 3] + cellVertPos[cB * 3] + cellVertPos[cC * 3];
    const cy = cellVertPos[cA * 3 + 1] + cellVertPos[cB * 3 + 1] + cellVertPos[cC * 3 + 1];
    const cz = cellVertPos[cA * 3 + 2] + cellVertPos[cB * 3 + 2] + cellVertPos[cC * 3 + 2];
    const oi = (cx >= 0 ? 4 : 0) | (cy >= 0 ? 2 : 0) | (cz >= 0 ? 1 : 0);
    sTriOctantId[ti] = oi;
    sOctantTriCounts[oi]++;
  }

  const posArr: Float32Array[] = [];
  const normArr: Float32Array[] = [];
  const uvArr: Float32Array[] = [];
  const matArr: Float32Array[] = [];
  const detArr: Float32Array[] = [];

  for (let oi = 0; oi < 8; oi++) {
    const vCount = sOctantTriCounts[oi] * 3;
    posArr.push(new Float32Array(vCount * 3));
    normArr.push(new Float32Array(vCount * 3));
    uvArr.push(new Float32Array(vCount * 2));
    matArr.push(new Float32Array(vCount * 4));
    detArr.push(new Float32Array(vCount * 2));
  }
  sOctantCursors.fill(0);

  const uvPadLo = 0.5 / WET_ATLAS_FACE_SIZE;
  const uvPadHi = (WET_ATLAS_FACE_SIZE - 0.5) / WET_ATLAS_FACE_SIZE;
  const inv2H = 0.5 / H;

  for (let ti = 0; ti < triCount; ti++) {
    const oi = sTriOctantId[ti];
    const t3 = ti * 3;
    const cA = triCellIndices[t3];
    const cB = triCellIndices[t3 + 1];
    const cC = triCellIndices[t3 + 2];
    const fnx = triFaceNormals[t3];
    const fny = triFaceNormals[t3 + 1];
    const fnz = triFaceNormals[t3 + 2];

    const gxC = (cellVertPos[cA * 3] + cellVertPos[cB * 3] + cellVertPos[cC * 3]) * (1 / 3);
    const gyC = (cellVertPos[cA * 3 + 1] + cellVertPos[cB * 3 + 1] + cellVertPos[cC * 3 + 1]) * (1 / 3);
    const gzC = (cellVertPos[cA * 3 + 2] + cellVertPos[cB * 3 + 2] + cellVertPos[cC * 3 + 2]) * (1 / 3);
    const qxC = gxC >= 0 ? gxC - halfSlot : gxC + halfSlot;
    const qyC = gyC >= 0 ? gyC - halfSlot : gyC + halfSlot;
    const qzC = gzC >= 0 ? gzC - halfSlot : gzC + halfSlot;
    const basis = dominantFaceBasis(qxC, qyC, qzC);

    const outTri = sOctantCursors[oi]++;
    const outBase = outTri * 3;
    for (let k = 0; k < 3; k++) {
      const aid = k === 0 ? cA : k === 1 ? cB : cC;
      writeVertexToBuffers(
        outBase + k,
        aid,
        fnx,
        fny,
        fnz,
        ti,
        halfSlot,
        basis.index,
        basis.ux,
        basis.uy,
        basis.uz,
        basis.vx,
        basis.vy,
        basis.vz,
        uvPadLo,
        uvPadHi,
        inv2H,
        posArr[oi],
        normArr[oi],
        uvArr[oi],
        matArr[oi],
        detArr[oi]
      );
    }
  }

  const outRaw: RawStoneMeshBuffers[] = [];
  for (let oi = 0; oi < 8; oi++) {
    outRaw.push({
      positions: posArr[oi],
      normals: normArr[oi],
      uvs: uvArr[oi],
      materialZone: matArr[oi],
      stoneDetail: detArr[oi],
    });
  }
  return outRaw;
}

function buildSockets(topology: LabyrinthTopology): SocketRecess[] {
  const goalNode = topology.nodes[topology.goalNodeId];
  const startNode = topology.nodes[topology.startNodeId];
  return [
    {
      cx: goalNode.localPos[0],
      cy: goalNode.localPos[1],
      cz: goalNode.localPos[2],
      nx: goalNode.localNormal[0],
      ny: goalNode.localNormal[1],
      nz: goalNode.localNormal[2],
      radius: 0.082,
      flareStartH: 0.045,
      coneSlope: 0.10,
      depthOffset: -0.024,
      minX: goalNode.localPos[0] - 0.180,
      minY: goalNode.localPos[1] - 0.180,
      minZ: goalNode.localPos[2] - 0.180,
      maxX: goalNode.localPos[0] + 0.180,
      maxY: goalNode.localPos[1] + 0.180,
      maxZ: goalNode.localPos[2] + 0.180,
    },
    {
      cx: startNode.localPos[0],
      cy: startNode.localPos[1],
      cz: startNode.localPos[2],
      nx: startNode.localNormal[0],
      ny: startNode.localNormal[1],
      nz: startNode.localNormal[2],
      radius: 0.074,
      flareStartH: 0.010,
      coneSlope: 0.08,
      depthOffset: -0.020,
      minX: startNode.localPos[0] - 0.120,
      minY: startNode.localPos[1] - 0.120,
      minZ: startNode.localPos[2] - 0.120,
      maxX: startNode.localPos[0] + 0.120,
      maxY: startNode.localPos[1] + 0.120,
      maxZ: startNode.localPos[2] + 0.120,
    },
  ];
}

function runDualContouringPipeline(
  topology: LabyrinthTopology,
  dcBound: number,
  halfSlot: number
): { triCount: number; activeCellCount: number; H: number } {
  const H = topology.cubeHalfExtent;
  const R = topology.bevelRadius;
  const hw = topology.channelWidth * 0.5;
  const surfaceDepth = topology.surfaceDepth;
  const edgeLimit = (2 * H - R) * INV_SQRT2;
  const cornerLimit = (3 * H - 2 * R) * INV_SQRT3;

  const dcMin = -dcBound;
  const dcStep = (dcBound * 2) / DC_GRID_N;
  const invDcStep = 1 / dcStep;

  const gridPitch = resolveGridPitch(topology);
  const bridges = buildBridgeVolumes(topology, gridPitch);
  const carveEdges = buildCarveEdges(topology, bridges);
  const pocketGrids = buildFacePocketGrids(topology, carveEdges, bridges);
  const sockets = buildSockets(topology);

  buildCarveEdgeSpatialBins(carveEdges);
  bakeMonolithSdf(
    dcMin,
    dcStep,
    invDcStep,
    halfSlot,
    H,
    edgeLimit,
    cornerLimit,
    hw,
    bridges,
    carveEdges,
    pocketGrids,
    sockets
  );

  extractCrossingEdges(dcMin, invDcStep, halfSlot === 0);
  const activeCellCount = sLastActiveCells;
  const crossingEdgeCount = sLastCrossingEdges;

  solveQefCells(
    dcMin,
    dcStep,
    halfSlot,
    activeCellCount,
    crossingEdgeCount,
    H,
    edgeLimit,
    cornerLimit,
    hw,
    bridges,
    carveEdges,
    pocketGrids,
    sockets
  );

  const triCount = emitDualQuadsAndCsr(activeCellCount, crossingEdgeCount);
  computeCellAoAndZones(
    dcMin,
    invDcStep,
    halfSlot,
    activeCellCount,
    H,
    surfaceDepth,
    topology.underpassDepth,
    hw,
    edgeLimit,
    cornerLimit,
    pocketGrids,
    bridges,
    carveEdges
  );

  return { triCount, activeCellCount, H };
}

function buildMonolithGeometry(topology: LabyrinthTopology): THREE.BufferGeometry {
  const { triCount, H } = runDualContouringPipeline(topology, DC_BOUND, 0);
  return emitFinalMonolithBuffers(triCount, H);
}

function buildNodeTangentFrame(
  node: LabyrinthNode,
  edges: readonly LabyrinthEdge[]
): { t1: THREE.Vector3; t2: THREE.Vector3; n: THREE.Vector3 } {
  const n = new THREE.Vector3(node.localNormal[0], node.localNormal[1], node.localNormal[2]).normalize();
  const t1 = new THREE.Vector3();
  let foundEdge = false;
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    if (e.from === node.id || e.to === node.id) {
      const kCount = Math.floor(e.samplesPos.length / 3);
      if (kCount >= 2) {
        if (e.from === node.id) {
          t1.set(
            e.samplesPos[3] - e.samplesPos[0],
            e.samplesPos[4] - e.samplesPos[1],
            e.samplesPos[5] - e.samplesPos[2]
          );
        } else {
          const i0 = (kCount - 1) * 3;
          const i1 = (kCount - 2) * 3;
          t1.set(
            e.samplesPos[i1] - e.samplesPos[i0],
            e.samplesPos[i1 + 1] - e.samplesPos[i0 + 1],
            e.samplesPos[i1 + 2] - e.samplesPos[i0 + 2]
          );
        }
        t1.addScaledVector(n, -t1.dot(n));
        if (t1.lengthSq() > 1e-6) {
          t1.normalize();
          foundEdge = true;
          break;
        }
      }
    }
  }
  if (!foundEdge) {
    const basis = FACE_BY_ID[node.face];
    t1.set(basis.ux, basis.uy, basis.uz);
    t1.addScaledVector(n, -t1.dot(n)).normalize();
  }
  const t2 = new THREE.Vector3().crossVectors(n, t1).normalize();
  return { t1, t2, n };
}

function buildSanctumInlayGeometry(topology: LabyrinthTopology): THREE.BufferGeometry {
  const goalNode = topology.nodes[topology.goalNodeId];
  const { t1, t2, n } = buildNodeTangentFrame(goalNode, topology.edges);
  const origin = new THREE.Vector3(goalNode.localPos[0], goalNode.localPos[1], goalNode.localPos[2]);

  const spillwayAngles: number[] = [];
  const tmpDir = new THREE.Vector3();
  for (let i = 0; i < topology.edges.length; i++) {
    const e = topology.edges[i];
    if (e.from !== goalNode.id && e.to !== goalNode.id) continue;
    const kCount = Math.floor(e.samplesPos.length / 3);
    if (kCount < 2) continue;
    if (e.from === goalNode.id) {
      tmpDir.set(
        e.samplesPos[3] - e.samplesPos[0],
        e.samplesPos[4] - e.samplesPos[1],
        e.samplesPos[5] - e.samplesPos[2]
      );
    } else {
      const i0 = (kCount - 1) * 3;
      const i1 = (kCount - 2) * 3;
      tmpDir.set(
        e.samplesPos[i1] - e.samplesPos[i0],
        e.samplesPos[i1 + 1] - e.samplesPos[i0 + 1],
        e.samplesPos[i1 + 2] - e.samplesPos[i0 + 2]
      );
    }
    tmpDir.addScaledVector(n, -tmpDir.dot(n));
    if (tmpDir.lengthSq() > 1e-6) {
      tmpDir.normalize();
      spillwayAngles.push(Math.atan2(tmpDir.dot(t2), tmpDir.dot(t1)));
    }
  }

  const profile: Array<readonly [number, number, number, number]> = [
    [0.0, 0.010, 0.0, 1.0],
    [0.016, 0.013, 0.0, 1.0],
    [0.032, 0.022, 0.0, 1.0],
    [0.044, 0.036, 0.0, 1.0],
    [0.050, 0.044, 0.15, 0.85],
    [0.055, 0.049, 1.0, 0.0],
    [0.066, 0.052, 1.0, 0.0],
    [0.078, 0.050, 1.0, 0.0],
    [0.087, 0.035, 1.0, 0.0],
    [0.091, 0.012, 1.0, 0.0],
    [0.092, -0.028, 1.0, 0.0],
    [0.0, -0.028, 0.0, 1.0],
  ];

  const radialSegs = 56;
  const pCount = profile.length;
  const vertCount = radialSegs * pCount;
  const positions = new Float32Array(vertCount * 3);
  const sanctumZone = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  for (let i = 0; i < radialSegs; i++) {
    const theta = (i / radialSegs) * Math.PI * 2;
    const ct = Math.cos(theta);
    const st = Math.sin(theta);

    let notchFactor = 0;
    for (let si = 0; si < spillwayAngles.length; si++) {
      let dAng = Math.abs(theta - spillwayAngles[si]);
      if (dAng > Math.PI) dAng = Math.PI * 2 - dAng;
      if (dAng < 0.42) {
        const u = 1 - dAng / 0.42;
        notchFactor = Math.max(notchFactor, u * u * (3 - 2 * u));
      }
    }

    for (let j = 0; j < pCount; j++) {
      const [r, baseH, calW, cinW] = profile[j];
      const h = j >= 5 && j <= 9 ? baseH - notchFactor * 0.036 : baseH;
      const vi = i * pCount + j;
      const lx = r * ct;
      const ly = r * st;
      positions[vi * 3] = origin.x + t1.x * lx + t2.x * ly + n.x * h;
      positions[vi * 3 + 1] = origin.y + t1.y * lx + t2.y * ly + n.y * h;
      positions[vi * 3 + 2] = origin.z + t1.z * lx + t2.z * ly + n.z * h;
      sanctumZone[vi * 2] = calW;
      sanctumZone[vi * 2 + 1] = cinW;
    }
  }

  for (let i = 0; i < radialSegs; i++) {
    const iNext = (i + 1) % radialSegs;
    for (let j = 0; j < pCount - 1; j++) {
      const a = i * pCount + j;
      const b = iNext * pCount + j;
      const c = iNext * pCount + (j + 1);
      const d = i * pCount + (j + 1);
      if (j === 0) {
        indices.push(a, d, c);
      } else if (j === pCount - 2) {
        indices.push(a, c, b);
      } else {
        indices.push(a, c, b);
        indices.push(a, d, c);
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('aSanctumZone', new THREE.BufferAttribute(sanctumZone, 2));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}

function buildStartSpringGeometry(topology: LabyrinthTopology): THREE.BufferGeometry {
  const startNode = topology.nodes[topology.startNodeId];
  const { t1, t2, n } = buildNodeTangentFrame(startNode, topology.edges);
  const origin = new THREE.Vector3(startNode.localPos[0], startNode.localPos[1], startNode.localPos[2]);

  const exitAngles: number[] = [];
  const tmpDir = new THREE.Vector3();
  for (let i = 0; i < topology.edges.length; i++) {
    const e = topology.edges[i];
    if (e.from !== startNode.id && e.to !== startNode.id) continue;
    const kCount = Math.floor(e.samplesPos.length / 3);
    if (kCount < 2) continue;
    if (e.from === startNode.id) {
      tmpDir.set(
        e.samplesPos[3] - e.samplesPos[0],
        e.samplesPos[4] - e.samplesPos[1],
        e.samplesPos[5] - e.samplesPos[2]
      );
    } else {
      const i0 = (kCount - 1) * 3;
      const i1 = (kCount - 2) * 3;
      tmpDir.set(
        e.samplesPos[i1] - e.samplesPos[i0],
        e.samplesPos[i1 + 1] - e.samplesPos[i0 + 1],
        e.samplesPos[i1 + 2] - e.samplesPos[i0 + 2]
      );
    }
    tmpDir.addScaledVector(n, -tmpDir.dot(n));
    if (tmpDir.lengthSq() > 1e-6) {
      tmpDir.normalize();
      exitAngles.push(Math.atan2(tmpDir.dot(t2), tmpDir.dot(t1)));
    }
  }

  const profile: Array<readonly [number, number, number, number]> = [
    [0.0, -0.001, 0.0, 1.0],
    [0.024, 0.002, 0.0, 1.0],
    [0.040, 0.011, 0.15, 0.85],
    [0.046, 0.018, 0.35, 0.65],
    [0.052, 0.022, 0.85, 0.15],
    [0.066, 0.021, 1.0, 0.0],
    [0.076, 0.009, 1.0, 0.0],
    [0.079, -0.024, 1.0, 0.0],
    [0.0, -0.024, 1.0, 0.0],
  ];

  const radialSegs = 44;
  const pCount = profile.length;
  const vertCount = radialSegs * pCount;
  const positions = new Float32Array(vertCount * 3);
  const startZone = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  for (let i = 0; i < radialSegs; i++) {
    const theta = (i / radialSegs) * Math.PI * 2;
    const ct = Math.cos(theta);
    const st = Math.sin(theta);

    let notchFactor = 0;
    for (let si = 0; si < exitAngles.length; si++) {
      let dAng = Math.abs(theta - exitAngles[si]);
      if (dAng > Math.PI) dAng = Math.PI * 2 - dAng;
      if (dAng < 0.48) {
        const u = 1 - dAng / 0.48;
        notchFactor = Math.max(notchFactor, u * u * (3 - 2 * u));
      }
    }

    for (let j = 0; j < pCount; j++) {
      const [r, baseH, basW, ambW] = profile[j];
      const h = j >= 3 && j <= 6 ? baseH - notchFactor * 0.016 : baseH;
      const vi = i * pCount + j;
      const lx = r * ct;
      const ly = r * st;
      positions[vi * 3] = origin.x + t1.x * lx + t2.x * ly + n.x * h;
      positions[vi * 3 + 1] = origin.y + t1.y * lx + t2.y * ly + n.y * h;
      positions[vi * 3 + 2] = origin.z + t1.z * lx + t2.z * ly + n.z * h;
      startZone[vi * 2] = basW;
      startZone[vi * 2 + 1] = ambW;
    }
  }

  for (let i = 0; i < radialSegs; i++) {
    const iNext = (i + 1) % radialSegs;
    for (let j = 0; j < pCount - 1; j++) {
      const a = i * pCount + j;
      const b = iNext * pCount + j;
      const c = iNext * pCount + (j + 1);
      const d = i * pCount + (j + 1);
      if (j === 0) {
        indices.push(a, d, c);
      } else if (j === pCount - 2) {
        indices.push(a, c, b);
      } else {
        indices.push(a, c, b);
        indices.push(a, d, c);
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('aStartZone', new THREE.BufferAttribute(startZone, 2));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}

function buildBeadGeometry(beadRadius: number): THREE.BufferGeometry {
  const t = (1 + Math.sqrt(5)) * 0.5;
  const baseVerts: Array<[number, number, number]> = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ];

  const verts: number[] = [];
  for (let i = 0; i < baseVerts.length; i++) {
    const [x, y, z] = baseVerts[i];
    const l = Math.hypot(x, y, z);
    verts.push(x / l, y / l, z / l);
  }

  let tris: number[] = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];

  for (let level = 0; level < 4; level++) {
    const midCache = new Map<number, number>();
    const nextTris: number[] = [];

    const getMid = (a: number, b: number): number => {
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      const key = lo * 65536 + hi;
      const hit = midCache.get(key);
      if (hit !== undefined) return hit;
      const mx = 0.5 * (verts[a * 3] + verts[b * 3]);
      const my = 0.5 * (verts[a * 3 + 1] + verts[b * 3 + 1]);
      const mz = 0.5 * (verts[a * 3 + 2] + verts[b * 3 + 2]);
      const invL = 1 / Math.hypot(mx, my, mz);
      const idx = verts.length / 3;
      verts.push(mx * invL, my * invL, mz * invL);
      midCache.set(key, idx);
      return idx;
    };

    for (let i = 0; i < tris.length; i += 3) {
      const a = tris[i];
      const b = tris[i + 1];
      const c = tris[i + 2];
      const ab = getMid(a, b);
      const bc = getMid(b, c);
      const ca = getMid(c, a);
      nextTris.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    tris = nextTris;
  }

  const vCount = verts.length / 3;
  const positions = new Float32Array(vCount * 3);
  const normals = new Float32Array(vCount * 3);
  for (let i = 0; i < vCount; i++) {
    const nx = verts[i * 3];
    const ny = verts[i * 3 + 1];
    const nz = verts[i * 3 + 2];
    normals[i * 3] = nx;
    normals[i * 3 + 1] = ny;
    normals[i * 3 + 2] = nz;
    positions[i * 3] = nx * beadRadius;
    positions[i * 3 + 1] = ny * beadRadius;
    positions[i * 3 + 2] = nz * beadRadius;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setIndex(tris);
  geom.computeBoundingSphere();
  return geom;
}

function clipSanctumHalfByZRaw(
  srcGeom: THREE.BufferGeometry,
  topology: LabyrinthTopology,
  signZ: 1 | -1
): RawSanctumHalfBuffers {
  const posAttr = srcGeom.getAttribute('position') as THREE.BufferAttribute;
  const normAttr = srcGeom.getAttribute('normal') as THREE.BufferAttribute;
  const zoneAttr = srcGeom.getAttribute('aSanctumZone') as THREE.BufferAttribute;
  const index = srcGeom.getIndex()!;

  const outPos: number[] = [];
  const outNorm: number[] = [];
  const outZone: number[] = [];

  interface ClipV {
    x: number;
    y: number;
    z: number;
    nx: number;
    ny: number;
    nz: number;
    cal: number;
    cin: number;
    sd: number;
  }

  const readV = (vi: number): ClipV => {
    const z = posAttr.getZ(vi);
    return {
      x: posAttr.getX(vi),
      y: posAttr.getY(vi),
      z,
      nx: normAttr.getX(vi),
      ny: normAttr.getY(vi),
      nz: normAttr.getZ(vi),
      cal: zoneAttr.getX(vi),
      cin: zoneAttr.getY(vi),
      sd: signZ * z,
    };
  };

  const lerpV = (a: ClipV, b: ClipV): ClipV => {
    const t = a.sd / (a.sd - b.sd);
    let nx = a.nx + (b.nx - a.nx) * t;
    let ny = a.ny + (b.ny - a.ny) * t;
    let nz = a.nz + (b.nz - a.nz) * t;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl;
    ny /= nl;
    nz /= nl;
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      z: 0,
      nx,
      ny,
      nz,
      cal: a.cal + (b.cal - a.cal) * t,
      cin: a.cin + (b.cin - a.cin) * t,
      sd: 0,
    };
  };

  const pushTri = (a: ClipV, b: ClipV, c: ClipV): void => {
    outPos.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    outNorm.push(a.nx, a.ny, a.nz, b.nx, b.ny, b.nz, c.nx, c.ny, c.nz);
    outZone.push(a.cal, a.cin, b.cal, b.cin, c.cal, c.cin);
  };

  const eps = 1e-6;
  const poly: ClipV[] = [];
  for (let i = 0; i < index.count; i += 3) {
    const v0 = readV(index.getX(i));
    const v1 = readV(index.getX(i + 1));
    const v2 = readV(index.getX(i + 2));
    const src = [v0, v1, v2];
    poly.length = 0;
    for (let k = 0; k < 3; k++) {
      const cur = src[k];
      const nxt = src[(k + 1) % 3];
      const curIn = cur.sd >= -eps;
      const nxtIn = nxt.sd >= -eps;
      if (curIn) {
        poly.push(cur);
        if (!nxtIn) poly.push(lerpV(cur, nxt));
      } else if (nxtIn) {
        poly.push(lerpV(cur, nxt));
      }
    }
    if (poly.length >= 3) {
      pushTri(poly[0], poly[1], poly[2]);
      if (poly.length === 4) {
        pushTri(poly[0], poly[2], poly[3]);
      }
    }
  }

  // Cap the two z=0 cross-sections of the carved mortar cup so split halves appear solid
  const goalNode = topology.nodes[topology.goalNodeId];
  const { t1, t2, n } = buildNodeTangentFrame(goalNode, topology.edges);
  const spillwayAngles: number[] = [];
  const tmpDir = new THREE.Vector3();
  for (let i = 0; i < topology.edges.length; i++) {
    const e = topology.edges[i];
    if (e.from !== goalNode.id && e.to !== goalNode.id) continue;
    const kCount = Math.floor(e.samplesPos.length / 3);
    if (kCount < 2) continue;
    if (e.from === goalNode.id) {
      tmpDir.set(
        e.samplesPos[3] - e.samplesPos[0],
        e.samplesPos[4] - e.samplesPos[1],
        e.samplesPos[5] - e.samplesPos[2]
      );
    } else {
      const i0 = (kCount - 1) * 3;
      const i1 = (kCount - 2) * 3;
      tmpDir.set(
        e.samplesPos[i1] - e.samplesPos[i0],
        e.samplesPos[i1 + 1] - e.samplesPos[i0 + 1],
        e.samplesPos[i1 + 2] - e.samplesPos[i0 + 2]
      );
    }
    tmpDir.addScaledVector(n, -tmpDir.dot(n));
    if (tmpDir.lengthSq() > 1e-6) {
      tmpDir.normalize();
      spillwayAngles.push(Math.atan2(tmpDir.dot(t2), tmpDir.dot(t1)));
    }
  }

  const profile: Array<readonly [number, number, number, number]> = [
    [0.0, 0.010, 0.0, 1.0],
    [0.016, 0.013, 0.0, 1.0],
    [0.032, 0.022, 0.0, 1.0],
    [0.044, 0.036, 0.0, 1.0],
    [0.050, 0.044, 0.15, 0.85],
    [0.055, 0.049, 1.0, 0.0],
    [0.066, 0.052, 1.0, 0.0],
    [0.078, 0.050, 1.0, 0.0],
    [0.087, 0.035, 1.0, 0.0],
    [0.091, 0.012, 1.0, 0.0],
    [0.092, -0.028, 1.0, 0.0],
    [0.0, -0.028, 0.0, 1.0],
  ];

  // Cross-section along z=0 lies along +X (dir = (+1, 0, 0)) and -X (dir = (-1, 0, 0))
  const capNormZ = -signZ;
  const ox = goalNode.localPos[0];
  const oy = goalNode.localPos[1];
  const capTris: Array<readonly [number, number, number]> = [
    [11, 0, 1],
    [11, 1, 2],
    [11, 2, 3],
    [11, 3, 4],
    [11, 4, 10],
    [10, 4, 5],
    [10, 5, 6],
    [10, 6, 7],
    [10, 7, 8],
    [10, 8, 9],
  ];

  for (const xArm of [-1, 1] as const) {
    const theta = Math.atan2(xArm * t2.x, xArm * t1.x);
    let notchFactor = 0;
    for (let si = 0; si < spillwayAngles.length; si++) {
      let dAng = Math.abs(theta - spillwayAngles[si]);
      if (dAng > Math.PI) dAng = Math.PI * 2 - dAng;
      if (dAng < 0.42) {
        const u = 1 - dAng / 0.42;
        notchFactor = Math.max(notchFactor, u * u * (3 - 2 * u));
      }
    }
    const pts = profile.map(([r, baseH, cal, cin], j) => {
      const h = j >= 5 && j <= 9 ? baseH - notchFactor * 0.036 : baseH;
      return {
        x: ox + xArm * r,
        y: oy + h,
        z: 0,
        nx: 0,
        ny: 0,
        nz: capNormZ,
        cal,
        cin,
        sd: 0,
      };
    });

    for (const [ia, ib, ic] of capTris) {
      const a = pts[ia], b = pts[ib], c = pts[ic];
      const e1x = b.x - a.x, e1y = b.y - a.y;
      const e2x = c.x - a.x, e2y = c.y - a.y;
      const cz = e1x * e2y - e1y * e2x;
      if (cz * capNormZ >= 0) {
        pushTri(a, b, c);
      } else {
        pushTri(a, c, b);
      }
    }
  }

  return {
    positions: new Float32Array(outPos),
    normals: new Float32Array(outNorm),
    sanctumZone: new Float32Array(outZone),
  };
}

export function buildVaultOctantRawBuffers(topology: LabyrinthTopology): VaultOctantsRawPayload {
  const { triCount, H } = runDualContouringPipeline(topology, DC_OCTANT_BOUND, OCTANT_HALF_SLOT);
  const octants = emitOctantMonolithRawBuffers(triCount, H, OCTANT_HALF_SLOT);
  const fullSanctum = buildSanctumInlayGeometry(topology);
  const sanctumPosZ = clipSanctumHalfByZRaw(fullSanctum, topology, 1);
  const sanctumNegZ = clipSanctumHalfByZRaw(fullSanctum, topology, -1);
  fullSanctum.dispose();

  return {
    octants,
    sanctumPosZ,
    sanctumNegZ,
  };
}

export function hydrateVaultOctantGeometries(payload: VaultOctantsRawPayload): VaultOctantsResult {
  const octantGeometries: THREE.BufferGeometry[] = [];
  for (let oi = 0; oi < payload.octants.length; oi++) {
    const raw = payload.octants[oi];
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(raw.positions, 3));
    geom.setAttribute('normal', new THREE.BufferAttribute(raw.normals, 3));
    geom.setAttribute('uv', new THREE.BufferAttribute(raw.uvs, 2));
    geom.setAttribute('aMaterialZone', new THREE.BufferAttribute(raw.materialZone, 4));
    geom.setAttribute('aStoneDetail', new THREE.BufferAttribute(raw.stoneDetail, 2));
    octantGeometries.push(finalizeGeometryBounds(geom, raw.positions));
  }

  const hydrateSanctumHalf = (raw: RawSanctumHalfBuffers): THREE.BufferGeometry => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(raw.positions, 3));
    geom.setAttribute('normal', new THREE.BufferAttribute(raw.normals, 3));
    geom.setAttribute('aSanctumZone', new THREE.BufferAttribute(raw.sanctumZone, 2));
    return finalizeGeometryBounds(geom, raw.positions);
  };

  return {
    octantGeometries,
    sanctumHalfPosZGeometry: hydrateSanctumHalf(payload.sanctumPosZ),
    sanctumHalfNegZGeometry: hydrateSanctumHalf(payload.sanctumNegZ),
  };
}

export function buildVaultOctantGeometries(topology: LabyrinthTopology): VaultOctantsResult {
  return hydrateVaultOctantGeometries(buildVaultOctantRawBuffers(topology));
}

export function createLabyrinthGeometries(topology: LabyrinthTopology): LabyrinthMeshesResult {
  void CUBE_FACES;
  const monolithGeometry = buildMonolithGeometry(topology);
  const sanctumInlayGeometry = buildSanctumInlayGeometry(topology);
  const startSpringGeometry = buildStartSpringGeometry(topology);
  const beadGeometry = buildBeadGeometry(topology.beadRadius);

  return {
    monolithGeometry,
    sanctumInlayGeometry,
    startSpringGeometry,
    beadGeometry,
  };
}
