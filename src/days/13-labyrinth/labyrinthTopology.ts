import {
  CUBE_FACES,
  EDGE_SAMPLE_COUNT,
  GRID_PITCH,
  WET_ATLAS_FACE_SIZE,
  type CubeFaceId,
  type LabyrinthEdge,
  type LabyrinthNode,
  type LabyrinthTopology,
  type StoneBridgeSpec,
} from './types';

export const LABYRINTH_CONSTANTS = {
  cubeHalfExtent: 0.88,
  bevelRadius: 0.095,
  beadRadius: 0.046,
  gridSize: 7,
  channelWidth: 0.135,
  surfaceDepth: 0.145,
  underpassDepth: 0.305,
} as const;

export interface FaceBasis {
  readonly face: CubeFaceId;
  readonly index: number;
  readonly normal: readonly [number, number, number];
  readonly uAxis: readonly [number, number, number];
  readonly vAxis: readonly [number, number, number];
}

// Strictly right-handed face frames (uAxis x vAxis === normal)
export const FACE_BASES: readonly FaceBasis[] = [
  { face: 'px', index: 0, normal: [1, 0, 0], uAxis: [0, 0, -1], vAxis: [0, 1, 0] },
  { face: 'nx', index: 1, normal: [-1, 0, 0], uAxis: [0, 0, 1], vAxis: [0, 1, 0] },
  { face: 'py', index: 2, normal: [0, 1, 0], uAxis: [1, 0, 0], vAxis: [0, 0, -1] },
  { face: 'ny', index: 3, normal: [0, -1, 0], uAxis: [1, 0, 0], vAxis: [0, 0, 1] },
  { face: 'pz', index: 4, normal: [0, 0, 1], uAxis: [1, 0, 0], vAxis: [0, 1, 0] },
  { face: 'nz', index: 5, normal: [0, 0, -1], uAxis: [-1, 0, 0], vAxis: [0, 1, 0] },
];

const FACE_BASIS_BY_ID: Record<CubeFaceId, FaceBasis> = {
  px: FACE_BASES[0],
  nx: FACE_BASES[1],
  py: FACE_BASES[2],
  ny: FACE_BASES[3],
  pz: FACE_BASES[4],
  nz: FACE_BASES[5],
};

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

function dot3(
  a: readonly [number, number, number],
  b: readonly [number, number, number]
): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function getFaceBasis(face: CubeFaceId): FaceBasis {
  return FACE_BASIS_BY_ID[face];
}

export function faceUvToAtlasUv(
  face: CubeFaceId,
  u01: number,
  v01: number,
  out?: [number, number]
): [number, number] {
  const basis = FACE_BASIS_BY_ID[face];
  const halfTexel = 0.5 / WET_ATLAS_FACE_SIZE;
  const maxTexel = 1.0 - halfTexel;
  const uClamped = clamp(u01, halfTexel, maxTexel);
  const vClamped = clamp(v01, halfTexel, maxTexel);
  const uAtlas = (basis.index + uClamped) / CUBE_FACES.length;
  const vAtlas = vClamped;
  if (out) {
    out[0] = uAtlas;
    out[1] = vAtlas;
    return out;
  }
  return [uAtlas, vAtlas];
}

export function localPointToFaceUv(
  localPos: readonly [number, number, number],
  cubeHalfExtent: number
): { face: CubeFaceId; faceIndex: number; u01: number; v01: number } {
  const x = localPos[0];
  const y = localPos[1];
  const z = localPos[2];
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);

  let basis: FaceBasis;
  if (ax >= ay && ax >= az) {
    basis = x >= 0 ? FACE_BASES[0] : FACE_BASES[1];
  } else if (ay >= az) {
    basis = y >= 0 ? FACE_BASES[2] : FACE_BASES[3];
  } else {
    basis = z >= 0 ? FACE_BASES[4] : FACE_BASES[5];
  }

  const inv2H = 0.5 / cubeHalfExtent;
  const u01 = clamp(0.5 + dot3(localPos, basis.uAxis) * inv2H, 0.0, 1.0);
  const v01 = clamp(0.5 + dot3(localPos, basis.vAxis) * inv2H, 0.0, 1.0);

  return {
    face: basis.face,
    faceIndex: basis.index,
    u01,
    v01,
  };
}

export function projectPointOntoFaceUv(
  localPos: readonly [number, number, number],
  face: CubeFaceId,
  cubeHalfExtent: number,
  out: [number, number]
): [number, number] {
  const basis = FACE_BASIS_BY_ID[face];
  const inv2H = 0.5 / cubeHalfExtent;
  out[0] = clamp(0.5 + dot3(localPos, basis.uAxis) * inv2H, 0.0, 1.0);
  out[1] = clamp(0.5 + dot3(localPos, basis.vAxis) * inv2H, 0.0, 1.0);
  return out;
}

type CellTuple = readonly [u: number, v: number, tier: 0 | 1];

interface FaceCorridorSpec {
  readonly face: CubeFaceId;
  readonly chains: ReadonlyArray<readonly CellTuple[]>;
}

const BRIDGES: readonly StoneBridgeSpec[] = [
  { face: 'pz', u: 3, v: 2, overAxis: 'u' },
  { face: 'py', u: 2, v: 3, overAxis: 'v' },
  { face: 'nx', u: 3, v: 3, overAxis: 'u' },
  { face: 'nz', u: 3, v: 3, overAxis: 'v' },
];

// 12 chamfered portals — one across each of the 12 edges of the cube.
const PORTAL_PAIRS: ReadonlyArray<
  readonly [
    readonly [CubeFaceId, number, number],
    readonly [CubeFaceId, number, number],
  ]
> = [
  [['pz', 5, 6], ['py', 5, 0]], // +Z / +Y
  [['pz', 6, 2], ['px', 0, 2]], // +Z / +X
  [['pz', 0, 4], ['nx', 6, 4]], // +Z / -X
  [['pz', 3, 0], ['ny', 3, 6]], // +Z / -Y
  [['ny', 0, 2], ['nx', 2, 0]], // -Y / -X
  [['ny', 6, 4], ['px', 2, 0]], // -Y / +X
  [['ny', 3, 0], ['nz', 3, 0]], // -Y / -Z
  [['nz', 6, 3], ['nx', 0, 3]], // -Z / -X
  [['nz', 0, 3], ['px', 6, 3]], // -Z / +X
  [['nz', 3, 6], ['py', 3, 6]], // -Z / +Y
  [['nx', 3, 6], ['py', 0, 3]], // -X / +Y
  [['px', 3, 6], ['py', 6, 3]], // +X / +Y
];

const FACE_CORRIDORS: readonly FaceCorridorSpec[] = [
  {
    face: 'pz',
    chains: [
      // Start Spring (1,5) multi-turn decoy alleys facing Goal ridge across +Y/+Z
      [
        [1, 5, 0],
        [1, 6, 0],
        [2, 6, 0],
        [3, 6, 0],
        [4, 6, 0],
      ],
      [
        [1, 5, 0],
        [2, 5, 0],
        [3, 5, 0],
        [4, 5, 0],
        [4, 4, 0],
      ],
      // Main exit from Start dives through Bridge 0 underpass to (3,0)->ny
      [
        [1, 5, 0],
        [1, 4, 0],
        [2, 4, 0],
        [3, 4, 0],
        [3, 3, 1],
        [3, 2, 1],
        [3, 1, 1],
        [3, 0, 0],
      ],
      // Upper Bridge 0 overpass wing connecting nx(6,4) <-> px(0,2) <-> py(5,0)
      [
        [0, 4, 0],
        [0, 3, 0],
        [0, 2, 0],
        [1, 2, 0],
        [2, 2, 0],
        [3, 2, 0],
        [4, 2, 0],
        [5, 2, 0],
        [6, 2, 0],
      ],
      [
        [0, 2, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
      ],
      [
        [5, 2, 0],
        [5, 3, 0],
        [5, 4, 0],
        [5, 5, 0],
        [5, 6, 0],
      ],
    ],
  },
  {
    face: 'ny',
    chains: [
      // East wing: pz(3,0)->ny(3,6) -> px(2,0)
      [
        [3, 6, 0],
        [4, 6, 0],
        [4, 5, 0],
        [5, 5, 0],
        [5, 4, 0],
        [6, 4, 0],
      ],
      [
        [4, 5, 0],
        [3, 5, 0],
        [2, 5, 0],
        [2, 4, 0],
        [3, 4, 0],
      ],
      // West wing: nx(2,0)->ny(0,2) <-> nz(3,0)
      [
        [0, 2, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
      ],
      [
        [1, 1, 0],
        [1, 2, 0],
        [1, 3, 0],
        [2, 3, 0],
      ],
    ],
  },
  {
    face: 'px',
    chains: [
      // Primary transit across +X connecting ny(6,4) -> nz(0,3) + branches to pz & py
      [
        [2, 0, 0],
        [2, 1, 0],
        [3, 1, 0],
        [4, 1, 0],
        [5, 1, 0],
        [5, 2, 0],
        [6, 2, 0],
        [6, 3, 0],
      ],
      [
        [2, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
        [0, 2, 0],
      ],
      [
        [5, 2, 0],
        [5, 3, 0],
        [5, 4, 0],
        [4, 4, 0],
        [4, 5, 0],
        [3, 5, 0],
        [3, 6, 0],
      ],
      [
        [4, 4, 0],
        [3, 4, 0],
        [3, 3, 0],
        [2, 3, 0],
        [2, 4, 0],
      ],
      [
        [5, 4, 0],
        [5, 5, 0],
        [6, 5, 0],
      ],
    ],
  },
  {
    face: 'nz',
    chains: [
      // Underpass wing: px(6,3)->nz(0,3) through Bridge 3 underpass -> nx(0,3)
      [
        [0, 3, 0],
        [1, 3, 0],
        [2, 3, 1],
        [3, 3, 1],
        [4, 3, 1],
        [5, 3, 0],
        [6, 3, 0],
      ],
      // Overpass wing: ny(3,0)->nz(3,0) over Bridge 3 -> py(3,6)
      [
        [3, 0, 0],
        [3, 1, 0],
        [3, 2, 0],
        [3, 3, 0],
        [3, 4, 0],
        [3, 5, 0],
        [3, 6, 0],
      ],
      [
        [3, 1, 0],
        [2, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
      ],
      [
        [3, 5, 0],
        [4, 5, 0],
        [5, 5, 0],
        [5, 4, 0],
      ],
    ],
  },
  {
    face: 'nx',
    chains: [
      // Underpass wing: nz(6,3)->nx(0,3) through Bridge 2 underpass -> py(0,3)
      [
        [0, 3, 0],
        [0, 2, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
        [3, 1, 0],
        [3, 2, 1],
        [3, 3, 1],
        [3, 4, 1],
        [3, 5, 0],
        [3, 6, 0],
      ],
      // Overpass wing: ny(0,2)->nx(2,0) over Bridge 2 -> pz(0,4)
      [
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
        [4, 1, 0],
        [4, 2, 0],
        [4, 3, 0],
        [3, 3, 0],
        [2, 3, 0],
        [2, 4, 0],
        [1, 4, 0],
        [1, 5, 0],
      ],
      [
        [4, 3, 0],
        [5, 3, 0],
        [5, 4, 0],
        [6, 4, 0],
      ],
    ],
  },
  {
    face: 'py',
    chains: [
      // Zone A (Goal Sanctum wing): nx(3,6)->py(0,3) through Bridge 1 underpass -> Goal (5,3)
      [
        [0, 3, 0],
        [1, 3, 1],
        [2, 3, 1],
        [3, 3, 1],
        [4, 3, 0],
        [5, 3, 0],
      ],
      // Zone B (Overpass scenic wing): pz(5,6)->py(5,0) over Bridge 1 -> nz(3,6) & px(3,6)
      [
        [5, 0, 0],
        [4, 0, 0],
        [4, 1, 0],
        [3, 1, 0],
        [2, 1, 0],
        [2, 2, 0],
        [2, 3, 0],
        [2, 4, 0],
        [2, 5, 0],
        [3, 5, 0],
        [3, 6, 0],
      ],
      [
        [4, 1, 0],
        [5, 1, 0],
        [5, 2, 0],
        [6, 2, 0],
        [6, 3, 0],
      ],
      [
        [2, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
      ],
      [
        [2, 5, 0],
        [1, 5, 0],
        [1, 6, 0],
      ],
      [
        [6, 3, 0],
        [6, 4, 0],
        [6, 5, 0],
        [5, 5, 0],
        [4, 5, 0],
        [4, 4, 0],
      ],
    ],
  },
];

function nodeKey(face: CubeFaceId, u: number, v: number, tier: 0 | 1): string {
  return `${face}:${u}:${v}:${tier}`;
}

function computeFloorPosition(
  face: CubeFaceId,
  u: number,
  v: number,
  tier: 0 | 1
): [number, number, number] {
  const basis = FACE_BASIS_BY_ID[face];
  const depth =
    tier === 0 ? LABYRINTH_CONSTANTS.surfaceDepth : LABYRINTH_CONSTANTS.underpassDepth;
  const floorHeight = LABYRINTH_CONSTANTS.cubeHalfExtent - depth;
  const uCoord = (u - 3) * GRID_PITCH;
  const vCoord = (v - 3) * GRID_PITCH;
  return [
    basis.normal[0] * floorHeight + basis.uAxis[0] * uCoord + basis.vAxis[0] * vCoord,
    basis.normal[1] * floorHeight + basis.uAxis[1] * uCoord + basis.vAxis[1] * vCoord,
    basis.normal[2] * floorHeight + basis.uAxis[2] * uCoord + basis.vAxis[2] * vCoord,
  ];
}

function buildEdgeSamples(
  fromNode: LabyrinthNode,
  toNode: LabyrinthNode,
  kind: LabyrinthEdge['kind']
): { length: number; samplesPos: Float32Array; samplesNormal: Float32Array } {
  const samplesPos = new Float32Array(EDGE_SAMPLE_COUNT * 3);
  const samplesNormal = new Float32Array(EDGE_SAMPLE_COUNT * 3);
  const p0 = fromNode.localPos;
  const p1 = toNode.localPos;
  const n0 = fromNode.localNormal;
  const n1 = toNode.localNormal;

  if (kind === 'surface' || kind === 'underpass') {
    for (let k = 0; k < EDGE_SAMPLE_COUNT; k++) {
      const t = k / (EDGE_SAMPLE_COUNT - 1);
      const idx = k * 3;
      samplesPos[idx + 0] = p0[0] + (p1[0] - p0[0]) * t;
      samplesPos[idx + 1] = p0[1] + (p1[1] - p0[1]) * t;
      samplesPos[idx + 2] = p0[2] + (p1[2] - p0[2]) * t;
      samplesNormal[idx + 0] = n0[0];
      samplesNormal[idx + 1] = n0[1];
      samplesNormal[idx + 2] = n0[2];
    }
    const dx = p1[0] - p0[0];
    const dy = p1[1] - p0[1];
    const dz = p1[2] - p0[2];
    return {
      length: Math.hypot(dx, dy, dz),
      samplesPos,
      samplesNormal,
    };
  }

  if (kind === 'ramp') {
    // C1 cubic S-curve elevation between tier 0 and tier 1 so surface<->ramp<->underpass joints have continuous normals
    const dn0 = dot3(p0, n0);
    const dn1 = dot3(p1, n0);
    const deltaH = dn1 - dn0;
    const h0x = p0[0] - n0[0] * dn0;
    const h0y = p0[1] - n0[1] * dn0;
    const h0z = p0[2] - n0[2] * dn0;
    const h1x = p1[0] - n0[0] * dn1;
    const h1y = p1[1] - n0[1] * dn1;
    const h1z = p1[2] - n0[2] * dn1;
    const dHorizX = h1x - h0x;
    const dHorizY = h1y - h0y;
    const dHorizZ = h1z - h0z;
    const horizLen = Math.hypot(dHorizX, dHorizY, dHorizZ);
    const dirX = dHorizX / horizLen;
    const dirY = dHorizY / horizLen;
    const dirZ = dHorizZ / horizLen;

    // Binormal = dir_horiz x n0 (constant across the ramp)
    const bx = dirY * n0[2] - dirZ * n0[1];
    const by = dirZ * n0[0] - dirX * n0[2];
    const bz = dirX * n0[1] - dirY * n0[0];

    for (let k = 0; k < EDGE_SAMPLE_COUNT; k++) {
      const s = k / (EDGE_SAMPLE_COUNT - 1);
      const hermite = s * s * (3 - 2 * s);
      const dhds = 6 * s * (1 - s) * deltaH;
      const height = dn0 + deltaH * hermite;

      const idx = k * 3;
      samplesPos[idx + 0] = h0x + dHorizX * s + n0[0] * height;
      samplesPos[idx + 1] = h0y + dHorizY * s + n0[1] * height;
      samplesPos[idx + 2] = h0z + dHorizZ * s + n0[2] * height;

      // Tangent = dHoriz + dhds * n0; normal = normalize(binormal x tangent)
      const tx = dHorizX + n0[0] * dhds;
      const ty = dHorizY + n0[1] * dhds;
      const tz = dHorizZ + n0[2] * dhds;
      const nx = by * tz - bz * ty;
      const ny = bz * tx - bx * tz;
      const nz = bx * ty - by * tx;
      const nLen = Math.hypot(nx, ny, nz) || 1;
      samplesNormal[idx + 0] = nx / nLen;
      samplesNormal[idx + 1] = ny / nLen;
      samplesNormal[idx + 2] = nz / nLen;
    }

    // Integrate exact arc length with fine Simpson/polyline quadrature
    let arcLen = 0;
    const STEPS = 64;
    let prevX = p0[0];
    let prevY = p0[1];
    let prevZ = p0[2];
    for (let i = 1; i <= STEPS; i++) {
      const s = i / STEPS;
      const height = dn0 + deltaH * (s * s * (3 - 2 * s));
      const cx = h0x + dHorizX * s + n0[0] * height;
      const cy = h0y + dHorizY * s + n0[1] * height;
      const cz = h0z + dHorizZ * s + n0[2] * height;
      arcLen += Math.hypot(cx - prevX, cy - prevY, cz - prevZ);
      prevX = cx;
      prevY = cy;
      prevZ = cz;
    }

    return {
      length: arcLen,
      samplesPos,
      samplesNormal,
    };
  }

  // 'bevel-wrap': exact C1 90-degree quarter-circle arc around (3 * GRID_PITCH) edge core
  const floorHeight =
    LABYRINTH_CONSTANTS.cubeHalfExtent - LABYRINTH_CONSTANTS.surfaceDepth;
  const edgeCore = 3 * GRID_PITCH; // 0.645
  const arcRadius = floorHeight - edgeCore; // 0.090
  // Shared-axis component orthogonal to both n0 and n1
  const p0n0 = dot3(p0, n0);
  const p0n1 = dot3(p0, n1);
  const sharedX = p0[0] - p0n0 * n0[0] - p0n1 * n1[0];
  const sharedY = p0[1] - p0n0 * n0[1] - p0n1 * n1[1];
  const sharedZ = p0[2] - p0n0 * n0[2] - p0n1 * n1[2];

  const centerX = sharedX + edgeCore * (n0[0] + n1[0]);
  const centerY = sharedY + edgeCore * (n0[1] + n1[1]);
  const centerZ = sharedZ + edgeCore * (n0[2] + n1[2]);

  for (let k = 0; k < EDGE_SAMPLE_COUNT; k++) {
    const theta = (k / (EDGE_SAMPLE_COUNT - 1)) * (Math.PI * 0.5);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const nx = cosT * n0[0] + sinT * n1[0];
    const ny = cosT * n0[1] + sinT * n1[1];
    const nz = cosT * n0[2] + sinT * n1[2];
    const idx = k * 3;
    samplesPos[idx + 0] = centerX + arcRadius * nx;
    samplesPos[idx + 1] = centerY + arcRadius * ny;
    samplesPos[idx + 2] = centerZ + arcRadius * nz;
    samplesNormal[idx + 0] = nx;
    samplesNormal[idx + 1] = ny;
    samplesNormal[idx + 2] = nz;
  }

  return {
    length: (Math.PI * 0.5) * arcRadius,
    samplesPos,
    samplesNormal,
  };
}

function computeShortestPathBfs(
  nodes: readonly LabyrinthNode[],
  edges: readonly LabyrinthEdge[],
  adjacency: ReadonlyArray<readonly number[]>,
  startNodeId: number,
  goalNodeId: number
): { nodePath: number[]; edgePath: number[] } {
  const n = nodes.length;
  const visited = new Uint8Array(n);
  const parentNode = new Int32Array(n).fill(-1);
  const parentEdge = new Int32Array(n).fill(-1);
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;

  visited[startNodeId] = 1;
  queue[tail++] = startNodeId;

  while (head < tail) {
    const curr = queue[head++];
    if (curr === goalNodeId) break;
    const outEdges = adjacency[curr];
    for (let i = 0; i < outEdges.length; i++) {
      const e = edges[outEdges[i]];
      const next = e.to;
      if (visited[next] === 0) {
        visited[next] = 1;
        parentNode[next] = curr;
        parentEdge[next] = e.id;
        queue[tail++] = next;
      }
    }
  }

  if (visited[goalNodeId] === 0) {
    throw new Error('Labyrinth topology is disconnected between startNodeId and goalNodeId');
  }

  const nodePath: number[] = [];
  const edgePath: number[] = [];
  let cursor = goalNodeId;
  while (cursor !== -1) {
    nodePath.push(cursor);
    const eId = parentEdge[cursor];
    if (eId !== -1) {
      edgePath.push(eId);
    }
    cursor = parentNode[cursor];
  }
  nodePath.reverse();
  edgePath.reverse();
  return { nodePath, edgePath };
}

export function createLabyrinthTopology(): LabyrinthTopology {
  const nodes: LabyrinthNode[] = [];
  const nodeIndexByKey = new Map<string, number>();

  const portalKeySet = new Set<string>();
  for (const [a, b] of PORTAL_PAIRS) {
    portalKeySet.add(nodeKey(a[0], a[1], a[2], 0));
    portalKeySet.add(nodeKey(b[0], b[1], b[2], 0));
  }

  const ensureNode = (face: CubeFaceId, u: number, v: number, tier: 0 | 1): number => {
    const key = nodeKey(face, u, v, tier);
    const existing = nodeIndexByKey.get(key);
    if (existing !== undefined) return existing;

    let kind: LabyrinthNode['kind'] = 'corridor';
    if (face === 'pz' && u === 1 && v === 5 && tier === 0) {
      kind = 'start';
    } else if (face === 'py' && u === 5 && v === 3 && tier === 0) {
      kind = 'goal';
    } else if (tier === 1) {
      kind = 'underpass';
    } else if (portalKeySet.has(key)) {
      kind = 'portal';
    }

    const basis = FACE_BASIS_BY_ID[face];
    const id = nodes.length;
    nodes.push({
      id,
      face,
      u,
      v,
      tier,
      kind,
      localPos: computeFloorPosition(face, u, v, tier),
      localNormal: [basis.normal[0], basis.normal[1], basis.normal[2]],
    });
    nodeIndexByKey.set(key, id);
    return id;
  };

  const edges: LabyrinthEdge[] = [];
  const adjacency: number[][] = [];
  const undirectedEdgeSet = new Set<string>();

  const addBidirectionalEdge = (
    aId: number,
    bId: number,
    kind: LabyrinthEdge['kind']
  ): void => {
    if (aId === bId) return;
    const minId = aId < bId ? aId : bId;
    const maxId = aId < bId ? bId : aId;
    const pairKey = `${minId}->${maxId}`;
    if (undirectedEdgeSet.has(pairKey)) return;
    undirectedEdgeSet.add(pairKey);

    while (adjacency.length < nodes.length) {
      adjacency.push([]);
    }

    const nodeA = nodes[aId];
    const nodeB = nodes[bId];
    const forwardSample = buildEdgeSamples(nodeA, nodeB, kind);

    // Reverse samples exactly mirror forward samples for strict twin symmetry
    const revPos = new Float32Array(EDGE_SAMPLE_COUNT * 3);
    const revNormal = new Float32Array(EDGE_SAMPLE_COUNT * 3);
    for (let k = 0; k < EDGE_SAMPLE_COUNT; k++) {
      const src = (EDGE_SAMPLE_COUNT - 1 - k) * 3;
      const dst = k * 3;
      revPos[dst + 0] = forwardSample.samplesPos[src + 0];
      revPos[dst + 1] = forwardSample.samplesPos[src + 1];
      revPos[dst + 2] = forwardSample.samplesPos[src + 2];
      revNormal[dst + 0] = forwardSample.samplesNormal[src + 0];
      revNormal[dst + 1] = forwardSample.samplesNormal[src + 1];
      revNormal[dst + 2] = forwardSample.samplesNormal[src + 2];
    }

    const fwdId = edges.length;
    edges.push({
      id: fwdId,
      from: aId,
      to: bId,
      kind,
      length: forwardSample.length,
      samplesPos: forwardSample.samplesPos,
      samplesNormal: forwardSample.samplesNormal,
    });
    adjacency[aId].push(fwdId);

    const revId = edges.length;
    edges.push({
      id: revId,
      from: bId,
      to: aId,
      kind,
      length: forwardSample.length,
      samplesPos: revPos,
      samplesNormal: revNormal,
    });
    adjacency[bId].push(revId);
  };

  for (const spec of FACE_CORRIDORS) {
    for (const chain of spec.chains) {
      for (let i = 0; i < chain.length; i++) {
        const [u, v, tier] = chain[i];
        ensureNode(spec.face, u, v, tier);
        if (i > 0) {
          const [pu, pv, ptier] = chain[i - 1];
          const prevId = ensureNode(spec.face, pu, pv, ptier);
          const currId = ensureNode(spec.face, u, v, tier);
          let kind: LabyrinthEdge['kind'] = 'surface';
          if (ptier === 1 && tier === 1) {
            kind = 'underpass';
          } else if (ptier !== tier) {
            kind = 'ramp';
          }
          addBidirectionalEdge(prevId, currId, kind);
        }
      }
    }
  }

  for (const [a, b] of PORTAL_PAIRS) {
    const aId = ensureNode(a[0], a[1], a[2], 0);
    const bId = ensureNode(b[0], b[1], b[2], 0);
    addBidirectionalEdge(aId, bId, 'bevel-wrap');
  }

  const startNodeId = nodeIndexByKey.get(nodeKey('pz', 1, 5, 0));
  const goalNodeId = nodeIndexByKey.get(nodeKey('py', 5, 3, 0));
  if (startNodeId === undefined || goalNodeId === undefined) {
    throw new Error('Missing required start or goal node in labyrinth topology');
  }

  const { nodePath, edgePath } = computeShortestPathBfs(
    nodes,
    edges,
    adjacency,
    startNodeId,
    goalNodeId
  );

  const topology: LabyrinthTopology = {
    cubeHalfExtent: LABYRINTH_CONSTANTS.cubeHalfExtent,
    bevelRadius: LABYRINTH_CONSTANTS.bevelRadius,
    beadRadius: LABYRINTH_CONSTANTS.beadRadius,
    gridSize: LABYRINTH_CONSTANTS.gridSize,
    channelWidth: LABYRINTH_CONSTANTS.channelWidth,
    surfaceDepth: LABYRINTH_CONSTANTS.surfaceDepth,
    underpassDepth: LABYRINTH_CONSTANTS.underpassDepth,
    nodes,
    edges,
    adjacency,
    startNodeId,
    goalNodeId,
    solutionNodePath: nodePath,
    solutionEdgePath: edgePath,
    bridges: BRIDGES,
  };

  verifyTopologyInvariants(topology);
  return topology;
}

export interface TopologyVerificationReport {
  nodeCount: number;
  edgeCount: number;
  portalPairCount: number;
  bridgeCount: number;
  solutionEdgeCount: number;
  solutionFacesCount: number;
  solutionUnderpassCount: number;
}

export function verifyTopologyInvariants(
  topology: LabyrinthTopology
): TopologyVerificationReport {
  if (
    topology.cubeHalfExtent !== 0.88 ||
    topology.bevelRadius !== 0.095 ||
    topology.beadRadius !== 0.046 ||
    topology.gridSize !== 7 ||
    topology.channelWidth !== 0.135 ||
    topology.surfaceDepth !== 0.145 ||
    topology.underpassDepth !== 0.305
  ) {
    throw new Error('Topology scalar constants do not match specification');
  }

  const { nodes, edges, adjacency, startNodeId, goalNodeId, bridges } = topology;
  const startNode = nodes[startNodeId];
  const goalNode = nodes[goalNodeId];
  if (!startNode || startNode.face !== 'pz' || startNode.u !== 1 || startNode.v !== 5 || startNode.tier !== 0) {
    throw new Error('Start Spring must be at (pz, 1, 5, tier=0)');
  }
  if (!goalNode || goalNode.face !== 'py' || goalNode.u !== 5 || goalNode.v !== 3 || goalNode.tier !== 0) {
    throw new Error('Goal Sanctum must be at (py, 5, 3, tier=0)');
  }

  if (bridges.length !== 4) {
    throw new Error(`Expected 4 stone bridges, got ${bridges.length}`);
  }

  const heroFaces = new Set<CubeFaceId>(['py', 'pz', 'px']);
  let heroBridges = 0;
  let backBridges = 0;
  for (const bridge of bridges) {
    if (heroFaces.has(bridge.face)) heroBridges++;
    else backBridges++;

    const hasOver = nodes.some(
      (n) => n.face === bridge.face && n.u === bridge.u && n.v === bridge.v && n.tier === 0
    );
    const hasUnder = nodes.some(
      (n) => n.face === bridge.face && n.u === bridge.u && n.v === bridge.v && n.tier === 1
    );
    if (!hasOver || !hasUnder) {
      throw new Error(`Bridge at ${bridge.face}(${bridge.u},${bridge.v}) lacks tier 0/1 pair`);
    }
  }
  if (heroBridges < 2 || backBridges < 2) {
    throw new Error(`Bridge distribution mismatch: hero=${heroBridges}, back=${backBridges}`);
  }

  // Verify edges, K=9 samples, and strict bidirectional twins
  const edgePairSet = new Set<string>();
  const cubeEdgeCrossings = new Set<string>();
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    if (e.id !== i) throw new Error(`Edge id mismatch at ${i}`);
    if (e.samplesPos.length !== EDGE_SAMPLE_COUNT * 3 || e.samplesNormal.length !== EDGE_SAMPLE_COUNT * 3) {
      throw new Error(`Edge ${e.id} must have K=${EDGE_SAMPLE_COUNT} samples`);
    }
    if (!Number.isFinite(e.length) || e.length <= 0) {
      throw new Error(`Edge ${e.id} has invalid length ${e.length}`);
    }
    for (let k = 0; k < EDGE_SAMPLE_COUNT; k++) {
      const idx = k * 3;
      const nx = e.samplesNormal[idx + 0];
      const ny = e.samplesNormal[idx + 1];
      const nz = e.samplesNormal[idx + 2];
      const nMag = Math.hypot(nx, ny, nz);
      if (!Number.isFinite(nMag) || Math.abs(nMag - 1) > 1e-3) {
        throw new Error(`Edge ${e.id} sample ${k} normal not unit length (${nMag})`);
      }
      if (
        !Number.isFinite(e.samplesPos[idx + 0]) ||
        !Number.isFinite(e.samplesPos[idx + 1]) ||
        !Number.isFinite(e.samplesPos[idx + 2])
      ) {
        throw new Error(`Edge ${e.id} sample ${k} contains NaN`);
      }
    }
    edgePairSet.add(`${e.from}->${e.to}`);
    if (e.kind === 'bevel-wrap') {
      const fA = nodes[e.from].face;
      const fB = nodes[e.to].face;
      const pairKey = fA < fB ? `${fA}:${fB}` : `${fB}:${fA}`;
      cubeEdgeCrossings.add(pairKey);
    }
  }

  for (const e of edges) {
    if (!edgePairSet.has(`${e.to}->${e.from}`)) {
      throw new Error(`One-way trap detected on edge ${e.id} (${e.from} -> ${e.to})`);
    }
  }

  if (cubeEdgeCrossings.size !== 12) {
    throw new Error(`Expected 12 cube-edge chamfer portals, found ${cubeEdgeCrossings.size}`);
  }

  // Verify 100% strong connectivity (all nodes reachable from start & can reach goal)
  const visitedFwd = new Uint8Array(nodes.length);
  const queue: number[] = [startNodeId];
  visitedFwd[startNodeId] = 1;
  for (let head = 0; head < queue.length; head++) {
    const curr = queue[head];
    for (const eId of adjacency[curr]) {
      const nxt = edges[eId].to;
      if (visitedFwd[nxt] === 0) {
        visitedFwd[nxt] = 1;
        queue.push(nxt);
      }
    }
  }
  if (queue.length !== nodes.length) {
    throw new Error(`Graph not strongly connected: reached ${queue.length}/${nodes.length} nodes`);
  }

  // Verify shortest geodesic constraints
  const { nodePath, edgePath } = topology.solutionNodePath.length
    ? { nodePath: topology.solutionNodePath, edgePath: topology.solutionEdgePath }
    : computeShortestPathBfs(nodes, edges, adjacency, startNodeId, goalNodeId);

  if (edgePath.length < 22) {
    throw new Error(`Shortest geodesic too short: ${edgePath.length} < 22 edges`);
  }

  const solFaces = new Set<CubeFaceId>();
  for (const nId of nodePath) {
    solFaces.add(nodes[nId].face);
  }
  if (solFaces.size < 4) {
    throw new Error(`Shortest geodesic wraps only ${solFaces.size} < 4 faces`);
  }

  // Count distinct underpasses threaded along the solution path
  let underpassRuns = 0;
  let inUnderpass = false;
  for (const eId of edgePath) {
    const isUnder = edges[eId].kind === 'underpass';
    if (isUnder && !inUnderpass) {
      underpassRuns++;
      inUnderpass = true;
    } else if (!isUnder) {
      inUnderpass = false;
    }
  }
  if (underpassRuns < 2) {
    throw new Error(`Shortest geodesic threads ${underpassRuns} < 2 underpasses`);
  }

  return {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    portalPairCount: cubeEdgeCrossings.size,
    bridgeCount: bridges.length,
    solutionEdgeCount: edgePath.length,
    solutionFacesCount: solFaces.size,
    solutionUnderpassCount: underpassRuns,
  };
}
