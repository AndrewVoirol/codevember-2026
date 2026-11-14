import {
  CHECKPOINT_SOL_INDICES,
  EDGE_SAMPLE_COUNT,
  WET_ATLAS_FACE_SIZE,
  WET_ATLAS_HEIGHT,
  WET_ATLAS_WIDTH,
  type BeadState,
  type CubeFaceId,
  type LabyrinthTopology,
} from './types';
import {
  getFaceBasis,
  localPointToFaceUv,
  projectPointOntoFaceUv,
} from './labyrinthTopology';

export const ROLLING_GRAVITY_FACTOR = 5 / 7;
export const DEAD_END_RESTITUTION = 0.28;
export const MAX_SUBSTEP_DT = 1 / 180;
export const SOLVER_GRAVITY_MAGNITUDE = 9.81;

const ROLLING_LINEAR_DRAG = 0.34;
const ROLLING_QUAD_DRAG = 2.35;
const WALL_RUB_FRICTION = 0.14;
const CORNER_RETENTION_BASE = 0.82;
const BRUSH_TEXEL_RADIUS = 6.0;
const RAMP_CLIMB_RESISTANCE_ATTEN = 0.46;
const JUNCTION_AXIAL_STEER_DT = 0.045;
const JUNCTION_MOUTH_CAPTURE_NUM = 0.50;
const JUNCTION_LATERAL_STEER_MIN_DT = 0.30;
const JUNCTION_LATERAL_STEER_MAX_DT = 1.05;
const IGNITE_SUBSAMPLES_PER_EDGE = 8;

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

function hash2D01(x: number, y: number): number {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263);
  h = (h ^ (h >>> 13)) | 0;
  h = Math.imul(h, 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

export class LabyrinthBeadSimulator {
  readonly topology: LabyrinthTopology;
  readonly state: BeadState;
  readonly wetAtlasData: Uint8Array;
  wetAtlasDirty = true;

  private readonly edgeCount: number;
  private readonly nodeCount: number;
  private readonly twinEdgeId: Int32Array;
  private readonly edgeBeadPos: Float32Array;
  private readonly edgeBeadLen: Float32Array;
  private readonly edgeStartTan: Float32Array;
  private readonly edgeEndTan: Float32Array;

  private readonly distToGoal: Float64Array;
  private readonly nextEdgeToGoal: Int32Array;
  private readonly maxDistToGoal: number;
  private readonly solutionNodeIndex: Int32Array;
  readonly checkpointNodeIds: readonly [number, number, number];
  private readonly checkpointBackEdgeIds: Int32Array;

  // High-precision wet atlas accumulators & active-texel sparse index for zero-GC O(painted) decay
  private readonly wetnessF32: Float32Array;
  private readonly sheenF32: Float32Array;
  private readonly thicknessF32: Float32Array;
  private readonly verdigrisF32: Float32Array;
  private readonly activeTexels: Int32Array;
  private readonly activeFlag: Uint8Array;
  private activeTexelCount = 0;
  private lastIgniteSample = -1;
  private decayAccumSec = 0;

  // Preallocated scratch buffers for strictly zero-allocation step() and solver queries
  private readonly scratchUv: [number, number] = [0, 0];
  private readonly scratchPos: [number, number, number] = [0, 0, 0];
  private readonly scratchNorm: [number, number, number] = [0, 0, 0];
  private readonly scratchTan: [number, number, number] = [0, 0, 0];
  private readonly scratchGravityOut: [number, number, number] = [0, 0, 0];

  constructor(topology: LabyrinthTopology) {
    this.topology = topology;
    this.edgeCount = topology.edges.length;
    this.nodeCount = topology.nodes.length;

    const texelCount = WET_ATLAS_WIDTH * WET_ATLAS_HEIGHT;
    this.wetAtlasData = new Uint8Array(texelCount * 4);
    this.wetnessF32 = new Float32Array(texelCount);
    this.sheenF32 = new Float32Array(texelCount);
    this.thicknessF32 = new Float32Array(texelCount);
    this.verdigrisF32 = new Float32Array(texelCount);
    this.activeTexels = new Int32Array(texelCount);
    this.activeFlag = new Uint8Array(texelCount);

    this.twinEdgeId = new Int32Array(this.edgeCount).fill(-1);
    this.edgeBeadPos = new Float32Array(this.edgeCount * EDGE_SAMPLE_COUNT * 3);
    this.edgeBeadLen = new Float32Array(this.edgeCount);
    this.edgeStartTan = new Float32Array(this.edgeCount * 3);
    this.edgeEndTan = new Float32Array(this.edgeCount * 3);

    this.state = {
      edgeId: 0,
      t: 0,
      speed: 0,
      localPos: [0, 0, 0],
      localNormal: [0, 0, 1],
      localTangent: [1, 0, 0],
      impactImpulse: 0,
      solved: false,
      solveProgress01: 0,
      checkpointsUnlocked: 0,
      dowsingHint01: 0,
    };

    this.precomputeEdgeCurves();

    this.solutionNodeIndex = new Int32Array(this.nodeCount).fill(-1);
    for (let i = 0; i < topology.solutionNodePath.length; i++) {
      this.solutionNodeIndex[topology.solutionNodePath[i]] = i;
    }

    const cp0 = topology.solutionNodePath[CHECKPOINT_SOL_INDICES[0]];
    const cp1 = topology.solutionNodePath[CHECKPOINT_SOL_INDICES[1]];
    const cp2 = topology.solutionNodePath[CHECKPOINT_SOL_INDICES[2]];
    this.checkpointNodeIds = [cp0, cp1, cp2];
    this.checkpointBackEdgeIds = new Int32Array(CHECKPOINT_SOL_INDICES.length).fill(-1);
    for (let k = 0; k < CHECKPOINT_SOL_INDICES.length; k++) {
      const solIdx = CHECKPOINT_SOL_INDICES[k];
      const cpNode = topology.solutionNodePath[solIdx];
      const prevNode = topology.solutionNodePath[solIdx - 1];
      const outEdges = topology.adjacency[cpNode];
      for (let ei = 0; ei < outEdges.length; ei++) {
        const candId = outEdges[ei];
        if (topology.edges[candId].to === prevNode) {
          this.checkpointBackEdgeIds[k] = candId;
          break;
        }
      }
    }

    this.distToGoal = new Float64Array(this.nodeCount);
    this.nextEdgeToGoal = new Int32Array(this.nodeCount).fill(-1);
    this.precomputeShortestPaths();
    this.maxDistToGoal = Math.max(1e-6, this.distToGoal[topology.startNodeId]);

    this.reset();
  }

  private precomputeEdgeCurves(): void {
    const { edges, adjacency, beadRadius } = this.topology;
    const K = EDGE_SAMPLE_COUNT;

    for (let eId = 0; eId < this.edgeCount; eId++) {
      const e = edges[eId];

      // Find twin reverse edge
      const backEdges = adjacency[e.to];
      for (let i = 0; i < backEdges.length; i++) {
        const candId = backEdges[i];
        if (edges[candId].to === e.from) {
          this.twinEdgeId[eId] = candId;
          break;
        }
      }

      // Compute bead-center sample positions = floorPos + beadRadius * normal
      const baseOffset = eId * K * 3;
      for (let k = 0; k < K; k++) {
        const src = k * 3;
        const dst = baseOffset + src;
        this.edgeBeadPos[dst + 0] = e.samplesPos[src + 0] + beadRadius * e.samplesNormal[src + 0];
        this.edgeBeadPos[dst + 1] = e.samplesPos[src + 1] + beadRadius * e.samplesNormal[src + 1];
        this.edgeBeadPos[dst + 2] = e.samplesPos[src + 2] + beadRadius * e.samplesNormal[src + 2];
      }

      // Compute bead-center spline arc length across 32 subdivisions
      const SUBS = 32;
      let totalLen = 0;
      let prevX = this.edgeBeadPos[baseOffset + 0];
      let prevY = this.edgeBeadPos[baseOffset + 1];
      let prevZ = this.edgeBeadPos[baseOffset + 2];
      for (let s = 1; s <= SUBS; s++) {
        const u = s / SUBS;
        this.evalEdgeBeadFrame(eId, u, this.state.localPos, this.state.localNormal, this.scratchTan);
        const cx = this.state.localPos[0];
        const cy = this.state.localPos[1];
        const cz = this.state.localPos[2];
        totalLen += Math.hypot(cx - prevX, cy - prevY, cz - prevZ);
        prevX = cx;
        prevY = cy;
        prevZ = cz;
      }
      this.edgeBeadLen[eId] = Math.max(1e-5, totalLen);

      // Endpoint unit tangents
      this.evalEdgeBeadFrame(eId, 0, this.state.localPos, this.state.localNormal, this.scratchTan);
      this.edgeStartTan[eId * 3 + 0] = this.scratchTan[0];
      this.edgeStartTan[eId * 3 + 1] = this.scratchTan[1];
      this.edgeStartTan[eId * 3 + 2] = this.scratchTan[2];

      this.evalEdgeBeadFrame(eId, 1, this.state.localPos, this.state.localNormal, this.scratchTan);
      this.edgeEndTan[eId * 3 + 0] = this.scratchTan[0];
      this.edgeEndTan[eId * 3 + 1] = this.scratchTan[1];
      this.edgeEndTan[eId * 3 + 2] = this.scratchTan[2];
    }
  }

  private precomputeShortestPaths(): void {
    // Reverse hop-priority BFS (matches topology.solutionEdgePath hop optimality)
    const { edges, adjacency, goalNodeId } = this.topology;
    this.distToGoal.fill(Infinity);
    this.nextEdgeToGoal.fill(-1);

    const queue = new Int32Array(this.nodeCount);
    let head = 0;
    let tail = 0;

    this.distToGoal[goalNodeId] = 0;
    queue[tail++] = goalNodeId;

    while (head < tail) {
      const targetNode = queue[head++];
      const nextDist = this.distToGoal[targetNode] + 1;
      const incEdges = adjacency[targetNode];
      for (let i = 0; i < incEdges.length; i++) {
        const revEdge = edges[incEdges[i]];
        const prevNode = revEdge.to;
        if (nextDist < this.distToGoal[prevNode]) {
          this.distToGoal[prevNode] = nextDist;
          this.nextEdgeToGoal[prevNode] = this.twinEdgeId[revEdge.id];
          queue[tail++] = prevNode;
        }
      }
    }
  }

  private evalEdgeBeadFrame(
    edgeId: number,
    t: number,
    outPos: [number, number, number],
    outNormal: [number, number, number],
    outTangent: [number, number, number]
  ): void {
    const K = EDGE_SAMPLE_COUNT;
    const segCount = K - 1;
    const clampedT = clamp(t, 0, 1);
    const scaled = clampedT * segCount;
    const seg = Math.min(segCount - 1, Math.floor(scaled));
    const u = scaled - seg;
    const u2 = u * u;
    const u3 = u2 * u;

    const basePos = edgeId * K * 3;
    const samplesNormal = this.topology.edges[edgeId].samplesNormal;

    const i1 = seg;
    const i2 = seg + 1;
    const hasLeft = i1 > 0;
    const hasRight = i2 < K - 1;

    for (let c = 0; c < 3; c++) {
      const p1 = this.edgeBeadPos[basePos + i1 * 3 + c];
      const p2 = this.edgeBeadPos[basePos + i2 * 3 + c];
      // 2nd-order quadratic ghost extrapolation keeps endpoint tangents smooth on circular arcs
      const p0 = hasLeft
        ? this.edgeBeadPos[basePos + (i1 - 1) * 3 + c]
        : 3 * p1 - 3 * p2 + this.edgeBeadPos[basePos + (i2 + 1) * 3 + c];
      const p3 = hasRight
        ? this.edgeBeadPos[basePos + (i2 + 1) * 3 + c]
        : 3 * p2 - 3 * p1 + this.edgeBeadPos[basePos + (i1 - 1) * 3 + c];

      const a = 2 * p1;
      const b = -p0 + p2;
      const cc = 2 * p0 - 5 * p1 + 4 * p2 - p3;
      const d = -p0 + 3 * p1 - 3 * p2 + p3;

      outPos[c] = 0.5 * (a + b * u + cc * u2 + d * u3);
      outTangent[c] = 0.5 * (b + 2 * cc * u + 3 * d * u2);

      const n1 = samplesNormal[i1 * 3 + c];
      const n2 = samplesNormal[i2 * 3 + c];
      outNormal[c] = n1 + (n2 - n1) * u;
    }

    const nLen = Math.hypot(outNormal[0], outNormal[1], outNormal[2]) || 1;
    outNormal[0] /= nLen;
    outNormal[1] /= nLen;
    outNormal[2] /= nLen;

    const tLen = Math.hypot(outTangent[0], outTangent[1], outTangent[2]) || 1;
    outTangent[0] /= tLen;
    outTangent[1] /= tLen;
    outTangent[2] /= tLen;
  }

  private clearWetAtlas(): void {
    const data = this.wetAtlasData;
    for (let i = 0; i < this.activeTexelCount; i++) {
      const texelIdx = this.activeTexels[i];
      this.wetnessF32[texelIdx] = 0;
      this.sheenF32[texelIdx] = 0;
      this.thicknessF32[texelIdx] = 0;
      this.verdigrisF32[texelIdx] = 0;
      this.activeFlag[texelIdx] = 0;
      const byteIdx = texelIdx * 4;
      data[byteIdx + 0] = 0;
      data[byteIdx + 1] = 0;
      data[byteIdx + 2] = 0;
      data[byteIdx + 3] = 0;
    }
    this.activeTexelCount = 0;
    this.lastIgniteSample = -1;
    this.decayAccumSec = 0;
    this.wetAtlasDirty = true;
  }

  private stampFaceBrush(
    face: CubeFaceId,
    u01: number,
    v01: number,
    wetStrength: number,
    sheenStrength: number,
    thicknessStrength: number,
    verdigrisStrength = 0
  ): void {
    const faceIndex = getFaceBasis(face).index;
    const faceSize = WET_ATLAS_FACE_SIZE;
    const cx = clamp(u01, 0, 1) * (faceSize - 1);
    const cy = clamp(v01, 0, 1) * (faceSize - 1);
    const rad = BRUSH_TEXEL_RADIUS;
    const radSq = rad * rad;
    const invRad = 1 / rad;

    const minX = Math.max(0, Math.floor(cx - rad));
    const maxX = Math.min(faceSize - 1, Math.ceil(cx + rad));
    const minY = Math.max(0, Math.floor(cy - rad));
    const maxY = Math.min(faceSize - 1, Math.ceil(cy + rad));
    const faceXOffset = faceIndex * faceSize;

    for (let ly = minY; ly <= maxY; ly++) {
      const dy = ly - cy;
      const dy2 = dy * dy;
      const rowOffset = ly * WET_ATLAS_WIDTH;
      for (let lx = minX; lx <= maxX; lx++) {
        const dx = lx - cx;
        const d2 = dx * dx + dy2;
        if (d2 >= radSq) continue;

        const w = 1 - d2 / radSq;
        const rNorm = Math.sqrt(d2) * invRad;
        // Organic capillary edge wicking near outer contact perimeter without sharp transitions
        const rimBlend = clamp((rNorm - 0.35) / 0.65, 0, 1);
        const smoothRim = rimBlend * rimBlend * (3 - 2 * rimBlend);
        const wick = 1.0 - smoothRim * 0.14 * (1.0 - hash2D01(faceXOffset + lx, ly));
        const falloff = w * Math.sqrt(w) * wick;
        const texelIdx = rowOffset + faceXOffset + lx;

        const nextWet = wetStrength * falloff;
        const nextSheen = sheenStrength * falloff;
        const nextThick = thicknessStrength * falloff;
        const nextVerdigris = verdigrisStrength * falloff;

        let changed = false;
        if (nextWet > this.wetnessF32[texelIdx]) {
          this.wetnessF32[texelIdx] = nextWet;
          changed = true;
        }
        if (nextSheen > this.sheenF32[texelIdx]) {
          this.sheenF32[texelIdx] = nextSheen;
          changed = true;
        }
        if (nextThick > this.thicknessF32[texelIdx]) {
          this.thicknessF32[texelIdx] = nextThick;
          changed = true;
        }
        if (nextVerdigris > this.verdigrisF32[texelIdx]) {
          this.verdigrisF32[texelIdx] = nextVerdigris;
          changed = true;
        }

        if (changed) {
          if (this.activeFlag[texelIdx] === 0) {
            this.activeFlag[texelIdx] = 1;
            this.activeTexels[this.activeTexelCount++] = texelIdx;
          }
          const byteIdx = texelIdx * 4;
          this.wetAtlasData[byteIdx + 0] = Math.min(255, Math.round(this.wetnessF32[texelIdx] * 255));
          this.wetAtlasData[byteIdx + 1] = Math.min(255, Math.round(this.sheenF32[texelIdx] * 255));
          this.wetAtlasData[byteIdx + 2] = Math.min(255, Math.round(this.thicknessF32[texelIdx] * 255));
          this.wetAtlasData[byteIdx + 3] = Math.min(255, Math.round(this.verdigrisF32[texelIdx] * 255));
          this.wetAtlasDirty = true;
        }
      }
    }
  }

  private stampPositionWithEdge(
    pos: readonly [number, number, number],
    edgeId: number,
    wetStrength: number,
    sheenStrength: number,
    thicknessStrength: number,
    verdigrisStrength = 0
  ): void {
    const H = this.topology.cubeHalfExtent;
    const { face, u01, v01 } = localPointToFaceUv(pos, H);
    this.stampFaceBrush(
      face,
      u01,
      v01,
      wetStrength,
      sheenStrength,
      thicknessStrength,
      verdigrisStrength
    );

    // Dual-stamp adjacent cube faces across bevel-wrap portals and rim transitions for seamless corners
    const edge = this.topology.edges[edgeId];
    if (edge.kind === 'bevel-wrap') {
      const fromFace = this.topology.nodes[edge.from].face;
      const toFace = this.topology.nodes[edge.to].face;
      if (fromFace !== face) {
        projectPointOntoFaceUv(pos, fromFace, H, this.scratchUv);
        this.stampFaceBrush(
          fromFace,
          this.scratchUv[0],
          this.scratchUv[1],
          wetStrength,
          sheenStrength,
          thicknessStrength,
          verdigrisStrength
        );
      }
      if (toFace !== face) {
        projectPointOntoFaceUv(pos, toFace, H, this.scratchUv);
        this.stampFaceBrush(
          toFace,
          this.scratchUv[0],
          this.scratchUv[1],
          wetStrength,
          sheenStrength,
          thicknessStrength,
          verdigrisStrength
        );
      }
    }
  }

  private stampCurrentTrail(
    wetStrength: number,
    sheenStrength: number,
    thicknessStrength: number,
    verdigrisStrength = 0
  ): void {
    this.stampPositionWithEdge(
      this.state.localPos,
      this.state.edgeId,
      wetStrength,
      sheenStrength,
      thicknessStrength,
      verdigrisStrength
    );
  }

  private isEdgeBlocked(edgeId: number): boolean {
    const unlocked = this.state.checkpointsUnlocked;
    for (let k = 0; k < unlocked; k++) {
      if (edgeId === this.checkpointBackEdgeIds[k]) {
        return true;
      }
    }
    return false;
  }

  private crystallizeSolutionEdgeRange(startSolEdgeIdx: number, endSolEdgeIdxExclusive: number): void {
    const solEdges = this.topology.solutionEdgePath;
    const lo = Math.max(0, startSolEdgeIdx);
    const hi = Math.min(solEdges.length, endSolEdgeIdxExclusive);
    const lockSubsamples = 20;
    for (let pIdx = lo; pIdx < hi; pIdx++) {
      const edgeId = solEdges[pIdx];
      for (let subStep = 0; subStep <= lockSubsamples; subStep++) {
        const tEdge = subStep / lockSubsamples;
        this.evalEdgeBeadFrame(edgeId, tEdge, this.scratchPos, this.scratchNorm, this.scratchTan);
        this.stampPositionWithEdge(this.scratchPos, edgeId, 0.94, 0.88, 1.0, 1.0);
      }
    }
  }

  private unlockCheckpointsUpToNode(nodeId: number): void {
    while (
      this.state.checkpointsUnlocked < 3 &&
      nodeId === this.checkpointNodeIds[this.state.checkpointsUnlocked]
    ) {
      const k = this.state.checkpointsUnlocked;
      this.state.checkpointsUnlocked = k + 1;
      this.state.impactImpulse = Math.min(1.5, Math.max(this.state.impactImpulse, 0.72));
      const startIdx = k === 0 ? 0 : CHECKPOINT_SOL_INDICES[k - 1];
      const endIdx = CHECKPOINT_SOL_INDICES[k];
      this.crystallizeSolutionEdgeRange(startIdx, endIdx);
    }
  }

  /**
   * Progressively floods a flowing Verdigris Malachite & Celadon capillary river
   * backward from Goal Sanctum to Start Spring along the 47-edge solution geodesic.
   */
  igniteSolutionPathWave(progress01: number): void {
    const solEdges = this.topology.solutionEdgePath;
    if (solEdges.length === 0) return;

    this.state.checkpointsUnlocked = 3;

    const totalSamples = solEdges.length * IGNITE_SUBSAMPLES_PER_EDGE;
    const targetSample = Math.min(
      totalSamples - 1,
      Math.floor(clamp(progress01, 0, 1) * totalSamples)
    );
    if (targetSample <= this.lastIgniteSample) return;

    for (let sIdx = this.lastIgniteSample + 1; sIdx <= targetSample; sIdx++) {
      const revEdgeOffset = Math.floor(sIdx / IGNITE_SUBSAMPLES_PER_EDGE);
      const pathIdx = solEdges.length - 1 - revEdgeOffset;
      const edgeId = solEdges[pathIdx];
      const subStep = sIdx % IGNITE_SUBSAMPLES_PER_EDGE;
      const tEdge = 1.0 - subStep / IGNITE_SUBSAMPLES_PER_EDGE;
      this.evalEdgeBeadFrame(edgeId, tEdge, this.scratchPos, this.scratchNorm, this.scratchTan);
      this.stampPositionWithEdge(this.scratchPos, edgeId, 1.0, 1.0, 1.0, 1.0);
    }

    this.lastIgniteSample = targetSample;
  }

  private decayWetAtlas(dt: number): void {
    if (this.activeTexelCount === 0) return;

    const wetDecay = Math.exp(-0.035 * dt);
    const sheenDecay = Math.exp(-0.32 * dt);
    const thickDecay = Math.exp(-0.008 * dt);
    const dowsingDecay = Math.exp(-0.40 * dt);

    let anyChanged = false;
    for (let k = this.activeTexelCount - 1; k >= 0; k--) {
      const i = this.activeTexels[k];
      const w = this.wetnessF32[i];
      const sh = this.sheenF32[i];
      const th = this.thicknessF32[i];
      const vd = this.verdigrisF32[i];

      const isLockedVerdigris = vd >= th * 0.45 && vd >= 0.004;
      const nw = w * wetDecay;
      const nsh = sh * sheenDecay;
      // Preserve permanent mineral yellow-ochre or smooth feathered locked verdigris floor
      const nth = isLockedVerdigris ? th : Math.max(th * 0.65, th * thickDecay);
      const nvd = isLockedVerdigris ? vd : vd * dowsingDecay;

      const fw = nw < 0.002 ? 0 : nw;
      const fsh = nsh < 0.002 ? 0 : nsh;
      const fth = nth < 0.002 ? 0 : nth;
      const fvd = nvd < 0.002 ? 0 : nvd;

      this.wetnessF32[i] = fw;
      this.sheenF32[i] = fsh;
      this.thicknessF32[i] = fth;
      this.verdigrisF32[i] = fvd;

      const byteIdx = i * 4;
      const r = Math.min(255, Math.round(fw * 255));
      const g = Math.min(255, Math.round(fsh * 255));
      const b = Math.min(255, Math.round(fth * 255));
      const a = Math.min(255, Math.round(fvd * 255));
      if (
        this.wetAtlasData[byteIdx + 0] !== r ||
        this.wetAtlasData[byteIdx + 1] !== g ||
        this.wetAtlasData[byteIdx + 2] !== b ||
        this.wetAtlasData[byteIdx + 3] !== a
      ) {
        this.wetAtlasData[byteIdx + 0] = r;
        this.wetAtlasData[byteIdx + 1] = g;
        this.wetAtlasData[byteIdx + 2] = b;
        this.wetAtlasData[byteIdx + 3] = a;
        anyChanged = true;
      }

      if (fw === 0 && fsh === 0 && fth === 0 && fvd === 0) {
        this.activeFlag[i] = 0;
        const lastIdx = --this.activeTexelCount;
        if (k !== lastIdx) {
          this.activeTexels[k] = this.activeTexels[lastIdx];
        }
      }
    }
    if (anyChanged) {
      this.wetAtlasDirty = true;
    }
  }

  reset(): void {
    const { startNodeId, adjacency, solutionEdgePath } = this.topology;
    const startEdgeId =
      solutionEdgePath.length > 0
        ? solutionEdgePath[0]
        : adjacency[startNodeId][0];

    this.state.edgeId = startEdgeId;
    this.state.t = 0;
    this.state.speed = 0;
    this.state.impactImpulse = 0;
    this.state.solved = false;
    this.state.solveProgress01 = 0;
    this.state.checkpointsUnlocked = 0;
    this.state.dowsingHint01 = 0;

    this.evalEdgeBeadFrame(
      startEdgeId,
      0,
      this.state.localPos,
      this.state.localNormal,
      this.state.localTangent
    );

    this.clearWetAtlas();
    this.stampCurrentTrail(0.95, 0.9, 0.9, 0);
  }

  step(dtSec: number, localGravity: readonly [number, number, number]): void {
    if (!Number.isFinite(dtSec) || dtSec <= 0) return;
    const gx = Number.isFinite(localGravity[0]) ? localGravity[0] : 0;
    const gy = Number.isFinite(localGravity[1]) ? localGravity[1] : 0;
    const gz = Number.isFinite(localGravity[2]) ? localGravity[2] : 0;

    const clampedDt = Math.min(dtSec, 0.1);
    const substeps = Math.max(1, Math.ceil(clampedDt / MAX_SUBSTEP_DT));
    const dtSub = clampedDt / substeps;

    const { edges, adjacency, goalNodeId } = this.topology;

    if (this.state.solved) {
      this.state.speed *= Math.exp(-14 * clampedDt);
      this.state.impactImpulse *= Math.exp(-9 * clampedDt);
      if (this.state.impactImpulse < 1e-5) this.state.impactImpulse = 0;
      this.state.solveProgress01 = 1.0;
      this.state.checkpointsUnlocked = 3;
      this.state.dowsingHint01 = 0;
      return;
    }

    for (let s = 0; s < substeps; s++) {
      this.state.impactImpulse *= Math.exp(-9.0 * dtSub);
      if (this.state.impactImpulse < 1e-5) this.state.impactImpulse = 0;

      let edgeId = this.state.edgeId;
      let t = this.state.t;
      let speed = this.state.speed;

      // Resting-junction launch selection: align with localGravity if resting near edge.from
      if (t <= 1e-4 && speed < 1e-3) {
        const fromNode = edges[edgeId].from;
        this.unlockCheckpointsUpToNode(fromNode);
        const outEdges = adjacency[fromNode];
        let bestOut = -1;
        let bestDot = -Infinity;
        for (let i = 0; i < outEdges.length; i++) {
          const candId = outEdges[i];
          if (this.isEdgeBlocked(candId)) continue;
          const off = candId * 3;
          const d =
            this.edgeStartTan[off + 0] * gx +
            this.edgeStartTan[off + 1] * gy +
            this.edgeStartTan[off + 2] * gz;
          if (d > bestDot) {
            bestDot = d;
            bestOut = candId;
          }
        }
        if (bestOut !== -1 && (this.isEdgeBlocked(edgeId) || (bestDot > 0.02 && bestOut !== edgeId))) {
          edgeId = bestOut;
          t = 0;
          speed = 0;
        }
      }

      this.evalEdgeBeadFrame(
        edgeId,
        t,
        this.state.localPos,
        this.state.localNormal,
        this.state.localTangent
      );

      const tx = this.state.localTangent[0];
      const ty = this.state.localTangent[1];
      const tz = this.state.localTangent[2];
      let gTangential = gx * tx + gy * ty + gz * tz;
      const nx = this.state.localNormal[0];
      const ny = this.state.localNormal[1];
      const nz = this.state.localNormal[2];
      const gNormal = gx * nx + gy * ny + gz * nz;
      const gLat = Math.hypot(
        gx - gNormal * nx - gTangential * tx,
        gy - gNormal * ny - gTangential * ty,
        gz - gNormal * nz - gTangential * tz
      );

      // Underpass exit ramp climb ergonomics: attenuate upward face-normal resistance so
      // a comfortable 24°-35° camera tilt exits stone underpasses while keeping 0° flat rollback
      const stepEdge = edges[edgeId];
      if (stepEdge.kind === 'ramp') {
        const faceN = this.topology.nodes[stepEdge.from].localNormal;
        const fnx = faceN[0];
        const fny = faceN[1];
        const fnz = faceN[2];
        const tDotN = tx * fnx + ty * fny + tz * fnz;
        const gDotN = gx * fnx + gy * fny + gz * fnz;
        if (tDotN > 0 && gDotN < 0) {
          gTangential -= (1.0 - RAMP_CLIMB_RESISTANCE_ATTEN) * gDotN * tDotN;
        }
      }

      // Static threshold prevents micro-creep when cube is virtually level;
      // tangent-plane side-wall load adds Coulomb rubbing grip when banking laterally into a turn
      let accel = 0;
      if (Math.abs(speed) >= 0.008 || Math.abs(gTangential) >= 0.03) {
        const drive = ROLLING_GRAVITY_FACTOR * gTangential;
        const wallRubDecel = WALL_RUB_FRICTION * gLat;
        let netDrive = drive;
        if (speed > 0.02) {
          netDrive -= wallRubDecel;
        } else if (Math.abs(drive) > wallRubDecel * 0.30) {
          netDrive -= Math.sign(drive) * wallRubDecel * 0.30;
        }
        accel =
          netDrive -
          ROLLING_LINEAR_DRAG * speed -
          ROLLING_QUAD_DRAG * speed * Math.abs(speed);
      }

      let newSpeed = speed + accel * dtSub;

      // Velocity reversal mid-edge flips cleanly onto twin reverse edge unless gated by ratchet
      if (newSpeed < 0) {
        const twinId = this.twinEdgeId[edgeId];
        if (twinId !== -1 && !this.isEdgeBlocked(twinId)) {
          edgeId = twinId;
          t = 1.0 - t;
          newSpeed = -newSpeed;
        } else {
          newSpeed = 0;
        }
      }

      let edgeLen = this.edgeBeadLen[edgeId];
      let tNext = t + (newSpeed * dtSub) / edgeLen;

      // Check immediate Goal Sanctum capture
      const curEdge = edges[edgeId];
      if (
        (curEdge.to === goalNodeId && tNext >= 0.92) ||
        (curEdge.from === goalNodeId && tNext <= 0.08)
      ) {
        const goalSideT = curEdge.to === goalNodeId ? 1.0 : 0.0;
        this.state.edgeId = edgeId;
        this.state.t = goalSideT;
        this.state.impactImpulse = Math.min(1.5, this.state.impactImpulse + newSpeed * 0.45);
        this.state.speed = 0;
        this.state.solved = true;
        this.state.solveProgress01 = 1.0;
        this.state.checkpointsUnlocked = 3;
        this.state.dowsingHint01 = 0;
        this.evalEdgeBeadFrame(
          edgeId,
          goalSideT,
          this.state.localPos,
          this.state.localNormal,
          this.state.localTangent
        );
        this.stampCurrentTrail(1.0, 1.0, 1.0, 0.96);
        return;
      }

      // Junction transitions, catch-basin stops & dead-end bounces (cap hops per substep)
      let hopGuard = 0;
      while (tNext >= 1.0 && hopGuard < 8) {
        hopGuard++;
        const overshootDist = (tNext - 1.0) * edgeLen;
        const arrivedNode = edges[edgeId].to;

        if (arrivedNode === goalNodeId) {
          this.state.edgeId = edgeId;
          this.state.t = 1.0;
          this.state.impactImpulse = Math.min(1.5, this.state.impactImpulse + newSpeed * 0.45);
          this.state.speed = 0;
          this.state.solved = true;
          this.state.solveProgress01 = 1.0;
          this.state.checkpointsUnlocked = 3;
          this.state.dowsingHint01 = 0;
          this.evalEdgeBeadFrame(
            edgeId,
            1.0,
            this.state.localPos,
            this.state.localNormal,
            this.state.localTangent
          );
          this.stampCurrentTrail(1.0, 1.0, 1.0, 0.96);
          return;
        }

        this.unlockCheckpointsUpToNode(arrivedNode);

        const outEdges = adjacency[arrivedNode];
        const twinId = this.twinEdgeId[edgeId];
        const inOff = edgeId * 3;
        const inDirX = this.edgeEndTan[inOff + 0];
        const inDirY = this.edgeEndTan[inOff + 1];
        const inDirZ = this.edgeEndTan[inOff + 2];

        let unblockedCount = 0;
        let onlyUnblockedId = -1;
        for (let i = 0; i < outEdges.length; i++) {
          const candId = outEdges[i];
          if (!this.isEdgeBlocked(candId)) {
            unblockedCount++;
            onlyUnblockedId = candId;
          }
        }

        if (outEdges.length === 1) {
          // Physical dead-end impact bounce with restitution e = 0.28
          const bouncedSpeed = newSpeed * DEAD_END_RESTITUTION;
          this.state.impactImpulse = Math.min(
            1.5,
            this.state.impactImpulse + newSpeed * (1.0 + DEAD_END_RESTITUTION) * 0.5
          );
          edgeId = outEdges[0];
          newSpeed = bouncedSpeed;
          edgeLen = this.edgeBeadLen[edgeId];
          tNext = Math.min(0.99, overshootDist / edgeLen);
        } else if (unblockedCount === 1 && onlyUnblockedId !== -1) {
          // Degree-2 one-way checkpoint catch-basin: free forward passage, cushioned stop if reversing
          const bOff = onlyUnblockedId * 3;
          const cosTurn = clamp(
            inDirX * this.edgeStartTan[bOff + 0] +
              inDirY * this.edgeStartTan[bOff + 1] +
              inDirZ * this.edgeStartTan[bOff + 2],
            -1,
            1
          );
          edgeId = onlyUnblockedId;
          edgeLen = this.edgeBeadLen[edgeId];
          if (onlyUnblockedId === twinId) {
            if (newSpeed > 0.08) {
              this.state.impactImpulse = Math.min(1.5, this.state.impactImpulse + newSpeed * 0.35);
            }
            newSpeed = 0;
            tNext = 0;
          } else {
            const cornerFactor = CORNER_RETENTION_BASE + (1.0 - CORNER_RETENTION_BASE) * Math.max(0, cosTurn);
            newSpeed *= cornerFactor;
            tNext = overshootDist / edgeLen;
          }
        } else {
          // Junction routing: project gravity onto corridor face tangent plane (excluding floor normal).
          // Intentional diagonal banking scales lateral mouth capture inversely with approach speed
          // while high-speed forward sprints and shallow axial tilts roll cleanly past 90° mouths.
          const faceN = this.topology.nodes[arrivedNode].localNormal;
          const gDotFaceN = gx * faceN[0] + gy * faceN[1] + gz * faceN[2];
          const gPlaneX = gx - gDotFaceN * faceN[0];
          const gPlaneY = gy - gDotFaceN * faceN[1];
          const gPlaneZ = gz - gDotFaceN * faceN[2];
          const gAxial = gPlaneX * inDirX + gPlaneY * inDirY + gPlaneZ * inDirZ;
          const gLatX = gPlaneX - gAxial * inDirX;
          const gLatY = gPlaneY - gAxial * inDirY;
          const gLatZ = gPlaneZ - gAxial * inDirZ;
          const gLatMag = Math.hypot(gLatX, gLatY, gLatZ);

          const axialSteerDt = Math.max(dtSub, JUNCTION_AXIAL_STEER_DT);
          const intentionalBank = gLatMag >= Math.max(0.38, gAxial * 0.52);
          const fastForwardFlyby =
            (newSpeed > 1.12 && gAxial > gLatMag) ||
            (gDotFaceN > 0.5 && gAxial > gLatMag);
          const vEff = Math.max(0.26, newSpeed);
          const lateralSteerDt =
            intentionalBank && !fastForwardFlyby
              ? clamp(
                  JUNCTION_MOUTH_CAPTURE_NUM / vEff,
                  JUNCTION_LATERAL_STEER_MIN_DT,
                  JUNCTION_LATERAL_STEER_MAX_DT
                )
              : axialSteerDt;

          const axialMag = Math.max(0, newSpeed + axialSteerDt * gAxial);
          const momX = axialMag * inDirX + lateralSteerDt * gLatX;
          const momY = axialMag * inDirY + lateralSteerDt * gLatY;
          const momZ = axialMag * inDirZ + lateralSteerDt * gLatZ;

          let bestOutId = -1;
          let bestScore = -Infinity;
          let bestBlockedScore = -Infinity;
          for (let i = 0; i < outEdges.length; i++) {
            const candId = outEdges[i];
            const cOff = candId * 3;
            const ux = this.edgeStartTan[cOff + 0];
            const uy = this.edgeStartTan[cOff + 1];
            const uz = this.edgeStartTan[cOff + 2];
            let score = ux * momX + uy * momY + uz * momZ;
            const candEdge = edges[candId];
            if (
              candEdge.kind === 'ramp' &&
              this.topology.nodes[candEdge.to].tier >
                this.topology.nodes[candEdge.from].tier &&
              gDotFaceN < -2.0 &&
              gAxial > 0.15
            ) {
              const alignIn = inDirX * ux + inDirY * uy + inDirZ * uz;
              if (alignIn > 0.7) {
                score += -gDotFaceN * 0.22;
              }
            }
            if (this.isEdgeBlocked(candId)) {
              if (score > bestBlockedScore) bestBlockedScore = score;
              continue;
            }
            if (candId !== twinId) score += 1e-6;
            if (score > bestScore) {
              bestScore = score;
              bestOutId = candId;
            }
          }
          if (bestOutId === -1) {
            bestOutId = outEdges[0];
          }

          if (unblockedCount < outEdges.length && bestBlockedScore > bestScore && bestScore <= 0.005) {
            // Momentum presses directly into the locked catch-basin stop -> park smoothly at basin lip
            if (newSpeed > 0.08) {
              this.state.impactImpulse = Math.min(1.5, this.state.impactImpulse + newSpeed * 0.35);
            }
            edgeId = bestOutId;
            edgeLen = this.edgeBeadLen[edgeId];
            newSpeed = 0;
            tNext = 0;
            break;
          }

          const bOff = bestOutId * 3;
          const cosTurn = clamp(
            inDirX * this.edgeStartTan[bOff + 0] +
              inDirY * this.edgeStartTan[bOff + 1] +
              inDirZ * this.edgeStartTan[bOff + 2],
            -1,
            1
          );

          if (bestOutId === twinId) {
            if (newSpeed > 0.05) {
              this.state.impactImpulse = Math.min(1.5, this.state.impactImpulse + newSpeed * 0.55);
              newSpeed *= DEAD_END_RESTITUTION;
            } else {
              newSpeed = 0;
            }
          } else {
            const cornerFactor = CORNER_RETENTION_BASE + (1.0 - CORNER_RETENTION_BASE) * Math.max(0, cosTurn);
            if (cosTurn < 0.35 && newSpeed > 0.25) {
              this.state.impactImpulse = Math.min(
                1.5,
                this.state.impactImpulse + newSpeed * (1.0 - Math.max(0, cosTurn)) * 0.22
              );
            }
            newSpeed *= cornerFactor;
          }

          edgeId = bestOutId;
          edgeLen = this.edgeBeadLen[edgeId];
          tNext = overshootDist / edgeLen;
        }
      }

      this.state.edgeId = edgeId;
      this.state.t = clamp(tNext, 0, 1);
      this.state.speed = newSpeed;

      this.evalEdgeBeadFrame(
        edgeId,
        this.state.t,
        this.state.localPos,
        this.state.localNormal,
        this.state.localTangent
      );

      const postEdge = edges[edgeId];
      const idxFrom = this.solutionNodeIndex[postEdge.from];
      const idxTo = this.solutionNodeIndex[postEdge.to];
      let dowsingHint = 0;
      if (idxFrom >= 0 && idxTo >= 0 && Math.abs(idxTo - idxFrom) === 1) {
        const curSolPos = idxFrom + (idxTo - idxFrom) * this.state.t;
        const targetSolPos =
          this.state.checkpointsUnlocked < 3
            ? CHECKPOINT_SOL_INDICES[this.state.checkpointsUnlocked]
            : this.topology.solutionNodePath.length - 1;
        const hopsRemaining = targetSolPos - curSolPos;
        if (hopsRemaining >= 0 && hopsRemaining <= 4.5) {
          const raw = 1 - hopsRemaining / 4.5;
          dowsingHint = raw * raw;
        }
      }
      this.state.dowsingHint01 = dowsingHint;

      this.stampCurrentTrail(1.0, 1.0, 0.96, dowsingHint * 0.20);
    }

    const activeEdge = edges[this.state.edgeId];
    const dFrom = this.distToGoal[activeEdge.from];
    const dTo = this.distToGoal[activeEdge.to];
    const curDist = dFrom + (dTo - dFrom) * this.state.t;
    this.state.solveProgress01 = clamp(1.0 - curDist / this.maxDistToGoal, 0, 1);

    this.decayAccumSec += clampedDt;
    if (this.decayAccumSec >= 0.05) {
      this.decayWetAtlas(this.decayAccumSec);
      this.decayAccumSec = 0;
    }
  }

  computeDownhillGravityForNextStep(
    out?: [number, number, number]
  ): [number, number, number] {
    const target = out ?? this.scratchGravityOut;
    const { edges, goalNodeId } = this.topology;
    const curEdge = edges[this.state.edgeId];

    if (this.state.solved) {
      const gn = this.topology.nodes[goalNodeId];
      target[0] = -gn.localNormal[0] * SOLVER_GRAVITY_MAGNITUDE;
      target[1] = -gn.localNormal[1] * SOLVER_GRAVITY_MAGNITUDE;
      target[2] = -gn.localNormal[2] * SOLVER_GRAVITY_MAGNITUDE;
      return target;
    }

    const dFrom = this.distToGoal[curEdge.from];
    const dTo = this.distToGoal[curEdge.to];
    const headingTowardGoal =
      this.state.edgeId === this.nextEdgeToGoal[curEdge.from] || dTo < dFrom;

    let gx = 0;
    let gy = 0;
    let gz = 0;

    if (this.state.t <= 1e-4 && this.state.speed < 1e-3) {
      const idealFromEdge = this.nextEdgeToGoal[curEdge.from];
      if (idealFromEdge !== -1) {
        const off = idealFromEdge * 3;
        gx = this.edgeStartTan[off + 0];
        gy = this.edgeStartTan[off + 1];
        gz = this.edgeStartTan[off + 2];
      }
    }

    if (gx === 0 && gy === 0 && gz === 0) {
      if (headingTowardGoal) {
        const tx = this.state.localTangent[0];
        const ty = this.state.localTangent[1];
        const tz = this.state.localTangent[2];
        gx = tx;
        gy = ty;
        gz = tz;

        // Look ahead 1-2 edges for T-junction / multi-way turns with smooth corner banking
        const isClimbingRamp =
          curEdge.kind === 'ramp' &&
          this.topology.nodes[curEdge.from].tier > this.topology.nodes[curEdge.to].tier;

        if (curEdge.to !== goalNodeId) {
          const nextEdgeId = this.nextEdgeToGoal[curEdge.to];
          if (nextEdgeId !== -1) {
            const nOff = nextEdgeId * 3;
            const nx = this.edgeStartTan[nOff + 0];
            const ny = this.edgeStartTan[nOff + 1];
            const nz = this.edgeStartTan[nOff + 2];
            const endOff = this.state.edgeId * 3;
            const cosTurn =
              this.edgeEndTan[endOff + 0] * nx +
              this.edgeEndTan[endOff + 1] * ny +
              this.edgeEndTan[endOff + 2] * nz;

            const deg = this.topology.adjacency[curEdge.to].length;
            if (deg >= 3 && cosTurn < 0.65) {
              if (isClimbingRamp) {
                if (this.state.t > 0.55) {
                  gx = tx * 0.35 + nx * 0.95;
                  gy = ty * 0.35 + ny * 0.95;
                  gz = tz * 0.35 + nz * 0.95;
                }
              } else {
                const checkAxial = this.state.speed > 0.65 ? -0.35 : 0.30;
                gx = tx * checkAxial + nx * 0.95;
                gy = ty * checkAxial + ny * 0.95;
                gz = tz * checkAxial + nz * 0.95;
              }
            } else if (!isClimbingRamp) {
              const nextEdge = edges[nextEdgeId];
              if (nextEdge.to !== goalNodeId) {
                const next2Id = this.nextEdgeToGoal[nextEdge.to];
                if (next2Id !== -1 && this.topology.adjacency[nextEdge.to].length >= 3) {
                  const n2Off = next2Id * 3;
                  const n1EndOff = nextEdgeId * 3;
                  const cosTurn2 =
                    this.edgeEndTan[n1EndOff + 0] * this.edgeStartTan[n2Off + 0] +
                    this.edgeEndTan[n1EndOff + 1] * this.edgeStartTan[n2Off + 1] +
                    this.edgeEndTan[n1EndOff + 2] * this.edgeStartTan[n2Off + 2];
                  if (cosTurn2 < 0.65 && this.state.t > 0.40) {
                    const preW = ((this.state.t - 0.40) / 0.60) * 0.55;
                    gx = tx * (1 - preW * 0.4) + this.edgeStartTan[n2Off + 0] * preW;
                    gy = ty * (1 - preW * 0.4) + this.edgeStartTan[n2Off + 1] * preW;
                    gz = tz * (1 - preW * 0.4) + this.edgeStartTan[n2Off + 2] * preW;
                  }
                }
              }
            } else if (this.state.t > 0.65) {
              const blend = (this.state.t - 0.65) / 0.35;
              const wNext = 0.35 + 0.65 * blend;
              gx = tx * (1 - wNext * 0.45) + nx * wNext;
              gy = ty * (1 - wNext * 0.45) + ny * wNext;
              gz = tz * (1 - wNext * 0.45) + nz * wNext;
            }
          }
        }
      } else {
        // Heading away from goal (e.g. inside a decoy branch): reverse toward `from`
        gx = -this.state.localTangent[0];
        gy = -this.state.localTangent[1];
        gz = -this.state.localTangent[2];
      }
    }

    const mag = Math.hypot(gx, gy, gz) || 1;
    target[0] = (gx / mag) * SOLVER_GRAVITY_MAGNITUDE;
    target[1] = (gy / mag) * SOLVER_GRAVITY_MAGNITUDE;
    target[2] = (gz / mag) * SOLVER_GRAVITY_MAGNITUDE;
    return target;
  }
}
