import {
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

const ROLLING_LINEAR_DRAG = 0.16;
const ROLLING_QUAD_DRAG = 0.05;
const CORNER_RETENTION_BASE = 0.82;
const BRUSH_TEXEL_RADIUS = 1.85;

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
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

  // High-precision wet atlas accumulators to prevent 8-bit quantization stall during decay
  private readonly wetnessF32: Float32Array;
  private readonly sheenF32: Float32Array;
  private readonly thicknessF32: Float32Array;
  private decayAccumSec = 0;

  // Preallocated scratch buffers for strictly zero-allocation step() and solver queries
  private readonly scratchUv: [number, number] = [0, 0];
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
    };

    this.precomputeEdgeCurves();

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
    this.wetnessF32.fill(0);
    this.sheenF32.fill(0);
    this.thicknessF32.fill(0);
    const data = this.wetAtlasData;
    for (let i = 0; i < data.length; i += 4) {
      data[i + 0] = 0;
      data[i + 1] = 0;
      data[i + 2] = 0;
      data[i + 3] = 255;
    }
    this.decayAccumSec = 0;
    this.wetAtlasDirty = true;
  }

  private stampFaceBrush(
    face: CubeFaceId,
    u01: number,
    v01: number,
    wetStrength: number,
    sheenStrength: number,
    thicknessStrength: number
  ): void {
    const faceIndex = getFaceBasis(face).index;
    const faceSize = WET_ATLAS_FACE_SIZE;
    const cx = clamp(u01, 0, 1) * (faceSize - 1);
    const cy = clamp(v01, 0, 1) * (faceSize - 1);
    const rad = BRUSH_TEXEL_RADIUS;
    const radSq = rad * rad;

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
        const falloff = w * Math.sqrt(w);
        const texelIdx = rowOffset + faceXOffset + lx;

        const nextWet = wetStrength * falloff;
        const nextSheen = sheenStrength * falloff;
        const nextThick = thicknessStrength * falloff;

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

        if (changed) {
          const byteIdx = texelIdx * 4;
          this.wetAtlasData[byteIdx + 0] = Math.min(255, Math.round(this.wetnessF32[texelIdx] * 255));
          this.wetAtlasData[byteIdx + 1] = Math.min(255, Math.round(this.sheenF32[texelIdx] * 255));
          this.wetAtlasData[byteIdx + 2] = Math.min(255, Math.round(this.thicknessF32[texelIdx] * 255));
          this.wetAtlasData[byteIdx + 3] = 255;
          this.wetAtlasDirty = true;
        }
      }
    }
  }

  private stampCurrentTrail(wetStrength: number, sheenStrength: number, thicknessStrength: number): void {
    const H = this.topology.cubeHalfExtent;
    const { face, u01, v01 } = localPointToFaceUv(this.state.localPos, H);
    this.stampFaceBrush(face, u01, v01, wetStrength, sheenStrength, thicknessStrength);

    // Dual-stamp adjacent cube faces across bevel-wrap portals and rim transitions for seamless corners
    const edge = this.topology.edges[this.state.edgeId];
    if (edge.kind === 'bevel-wrap') {
      const fromFace = this.topology.nodes[edge.from].face;
      const toFace = this.topology.nodes[edge.to].face;
      if (fromFace !== face) {
        projectPointOntoFaceUv(this.state.localPos, fromFace, H, this.scratchUv);
        this.stampFaceBrush(
          fromFace,
          this.scratchUv[0],
          this.scratchUv[1],
          wetStrength,
          sheenStrength,
          thicknessStrength
        );
      }
      if (toFace !== face) {
        projectPointOntoFaceUv(this.state.localPos, toFace, H, this.scratchUv);
        this.stampFaceBrush(
          toFace,
          this.scratchUv[0],
          this.scratchUv[1],
          wetStrength,
          sheenStrength,
          thicknessStrength
        );
      }
    }
  }

  private decayWetAtlas(dt: number): void {
    const wetDecay = Math.exp(-0.035 * dt);
    const sheenDecay = Math.exp(-0.32 * dt);
    const thickDecay = Math.exp(-0.008 * dt);
    const texelCount = WET_ATLAS_WIDTH * WET_ATLAS_HEIGHT;

    let anyChanged = false;
    for (let i = 0; i < texelCount; i++) {
      const w = this.wetnessF32[i];
      const sh = this.sheenF32[i];
      const th = this.thicknessF32[i];
      if (w <= 0 && sh <= 0 && th <= 0) continue;

      const nw = w * wetDecay;
      const nsh = sh * sheenDecay;
      // Preserve permanent mineral yellow-ochre residue floor at 65% of peak deposit
      const nth = Math.max(th * 0.65, th * thickDecay);

      this.wetnessF32[i] = nw < 0.002 ? 0 : nw;
      this.sheenF32[i] = nsh < 0.002 ? 0 : nsh;
      this.thicknessF32[i] = nth < 0.002 ? 0 : nth;

      const byteIdx = i * 4;
      const r = Math.min(255, Math.round(this.wetnessF32[i] * 255));
      const g = Math.min(255, Math.round(this.sheenF32[i] * 255));
      const b = Math.min(255, Math.round(this.thicknessF32[i] * 255));
      if (
        this.wetAtlasData[byteIdx + 0] !== r ||
        this.wetAtlasData[byteIdx + 1] !== g ||
        this.wetAtlasData[byteIdx + 2] !== b
      ) {
        this.wetAtlasData[byteIdx + 0] = r;
        this.wetAtlasData[byteIdx + 1] = g;
        this.wetAtlasData[byteIdx + 2] = b;
        anyChanged = true;
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

    this.evalEdgeBeadFrame(
      startEdgeId,
      0,
      this.state.localPos,
      this.state.localNormal,
      this.state.localTangent
    );

    this.clearWetAtlas();
    this.stampCurrentTrail(0.95, 0.9, 0.9);
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
        const outEdges = adjacency[fromNode];
        let bestOut = edgeId;
        let bestDot = -Infinity;
        for (let i = 0; i < outEdges.length; i++) {
          const candId = outEdges[i];
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
        if (bestDot > 0.02 && bestOut !== edgeId) {
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
      const gTangential = gx * tx + gy * ty + gz * tz;

      // Static threshold prevents micro-creep when cube is virtually level
      let accel = 0;
      if (Math.abs(speed) >= 0.008 || Math.abs(gTangential) >= 0.03) {
        accel =
          ROLLING_GRAVITY_FACTOR * gTangential -
          ROLLING_LINEAR_DRAG * speed -
          ROLLING_QUAD_DRAG * speed * Math.abs(speed);
      }

      let newSpeed = speed + accel * dtSub;

      // Velocity reversal mid-edge flips cleanly onto twin reverse edge
      if (newSpeed < 0) {
        const twinId = this.twinEdgeId[edgeId];
        if (twinId !== -1) {
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
        this.evalEdgeBeadFrame(
          edgeId,
          goalSideT,
          this.state.localPos,
          this.state.localNormal,
          this.state.localTangent
        );
        this.stampCurrentTrail(1.0, 1.0, 1.0);
        return;
      }

      // Junction transitions & dead-end bounces (cap hops per substep)
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
          this.evalEdgeBeadFrame(
            edgeId,
            1.0,
            this.state.localPos,
            this.state.localNormal,
            this.state.localTangent
          );
          this.stampCurrentTrail(1.0, 1.0, 1.0);
          return;
        }

        const outEdges = adjacency[arrivedNode];
        const twinId = this.twinEdgeId[edgeId];
        const inOff = edgeId * 3;
        const inDirX = this.edgeEndTan[inOff + 0];
        const inDirY = this.edgeEndTan[inOff + 1];
        const inDirZ = this.edgeEndTan[inOff + 2];

        if (outEdges.length === 1) {
          // Dead-end impact bounce with restitution e = 0.28
          const bouncedSpeed = newSpeed * DEAD_END_RESTITUTION;
          this.state.impactImpulse = Math.min(
            1.5,
            this.state.impactImpulse + newSpeed * (1.0 + DEAD_END_RESTITUTION) * 0.5
          );
          edgeId = outEdges[0];
          newSpeed = bouncedSpeed;
          edgeLen = this.edgeBeadLen[edgeId];
          tNext = Math.min(0.99, overshootDist / edgeLen);
        } else {
          // Junction routing: score = dot(unitDir, incomingMomentum + dt * g_local)
          // Use a physical intersection contact window (min 0.035s) so steering is frame-rate independent
          const steerDt = Math.max(dtSub, 0.035);
          const momX = newSpeed * inDirX + steerDt * gx;
          const momY = newSpeed * inDirY + steerDt * gy;
          const momZ = newSpeed * inDirZ + steerDt * gz;

          let bestOutId = outEdges[0];
          let bestScore = -Infinity;
          for (let i = 0; i < outEdges.length; i++) {
            const candId = outEdges[i];
            const cOff = candId * 3;
            const ux = this.edgeStartTan[cOff + 0];
            const uy = this.edgeStartTan[cOff + 1];
            const uz = this.edgeStartTan[cOff + 2];
            let score = ux * momX + uy * momY + uz * momZ;
            if (candId !== twinId) score += 1e-6;
            if (score > bestScore) {
              bestScore = score;
              bestOutId = candId;
            }
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

      this.stampCurrentTrail(1.0, 1.0, 0.96);
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

        // Look ahead 1-2 edges for T-junction / multi-way turns so approach speed stays controllable
        let needsBrake = false;
        let turnX = tx;
        let turnY = ty;
        let turnZ = tz;

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
              // Multi-way junction turn: keep speed <= 0.24 so steerDt * g dominates straight decoys
              turnX = nx;
              turnY = ny;
              turnZ = nz;
              const distLeft = (1.0 - this.state.t) * this.edgeBeadLen[this.state.edgeId];
              const stopDist = (this.state.speed * this.state.speed) / (2 * ROLLING_GRAVITY_FACTOR * SOLVER_GRAVITY_MAGNITUDE);
              if (this.state.speed > 0.24 && (stopDist * 1.35 >= distLeft || this.state.t > 0.35)) {
                needsBrake = true;
              }
            } else {
              // Check one node further ahead in case high speed needs braking before entering the edge before a T-turn
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
                  if (cosTurn2 < 0.65 && this.state.speed > 1.25) {
                    needsBrake = true;
                  }
                }
              }
            }

            if (needsBrake) {
              gx = -tx + turnX * 0.25;
              gy = -ty + turnY * 0.25;
              gz = -tz + turnZ * 0.25;
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
