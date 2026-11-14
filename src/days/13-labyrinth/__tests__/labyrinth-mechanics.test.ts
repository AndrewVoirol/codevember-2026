import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { float } from 'three/tsl';
import { type SpecimenStage } from '../../../core/stage/SpecimenStage';
import {
  CHECKPOINT_SOL_INDICES,
  CUBE_FACES,
  EDGE_SAMPLE_COUNT,
  WET_ATLAS_FACE_SIZE,
  WET_ATLAS_HEIGHT,
  WET_ATLAS_WIDTH,
  type CubeFaceId,
} from '../types';
import {
  createLabyrinthTopology,
  faceUvToAtlasUv,
  localPointToFaceUv,
  verifyTopologyInvariants,
} from '../labyrinthTopology';
import {
  DEAD_END_RESTITUTION,
  GROOVE_MAX_LAT_OFFSET,
  LabyrinthBeadSimulator,
  ROLLING_GRAVITY_FACTOR,
} from '../labyrinthPhysics';
import { createLabyrinthGeometries } from '../labyrinthGeometry';
import { createLabyrinthOptics } from '../labyrinthOptics';

describe('Day 13 — Labyrinth: Multi-Face Topology & Geodesic Invariants', () => {
  const topology = createLabyrinthTopology();
  const report = verifyTopologyInvariants(topology);

  it('satisfies exact specification constants, 12 portals, 4 bridges, and 100% bidirectional reachability', () => {
    expect(topology.cubeHalfExtent).toBe(0.88);
    expect(topology.bevelRadius).toBe(0.095);
    expect(topology.beadRadius).toBe(0.046);
    expect(topology.gridSize).toBe(7);
    expect(topology.channelWidth).toBe(0.135);
    expect(topology.surfaceDepth).toBe(0.145);
    expect(topology.underpassDepth).toBe(0.305);

    expect(report.portalPairCount).toBe(12);
    expect(report.bridgeCount).toBe(4);

    // Every bridge has both tier 0 overpass and tier 1 underpass; 2 hero + 2 back
    const heroFaces = new Set<CubeFaceId>(['py', 'pz', 'px']);
    let heroCount = 0;
    let backCount = 0;
    for (const b of topology.bridges) {
      if (heroFaces.has(b.face)) heroCount++;
      else backCount++;
    }
    expect(heroCount).toBe(2);
    expect(backCount).toBe(2);

    // Start & Goal placement
    const startNode = topology.nodes[topology.startNodeId];
    const goalNode = topology.nodes[topology.goalNodeId];
    expect(startNode.face).toBe('pz');
    expect(startNode.u).toBe(1);
    expect(startNode.v).toBe(5);
    expect(startNode.tier).toBe(0);
    expect(startNode.kind).toBe('start');

    expect(goalNode.face).toBe('py');
    expect(goalNode.u).toBe(5);
    expect(goalNode.v).toBe(3);
    expect(goalNode.tier).toBe(0);
    expect(goalNode.kind).toBe('goal');

    // Every directed edge has K=9 finite C1 samples and an exact reverse twin
    for (const edge of topology.edges) {
      expect(edge.samplesPos.length).toBe(EDGE_SAMPLE_COUNT * 3);
      expect(edge.samplesNormal.length).toBe(EDGE_SAMPLE_COUNT * 3);
      for (let i = 0; i < edge.samplesPos.length; i++) {
        expect(Number.isFinite(edge.samplesPos[i])).toBe(true);
        expect(Number.isFinite(edge.samplesNormal[i])).toBe(true);
      }
      const hasTwin = topology.adjacency[edge.to].some(
        (id) => topology.edges[id].to === edge.from
      );
      expect(hasTwin).toBe(true);
    }

    // 100% strong connectivity from every node to every other node (zero traps)
    const visited = new Uint8Array(topology.nodes.length);
    const queue = [topology.startNodeId];
    visited[topology.startNodeId] = 1;
    for (let h = 0; h < queue.length; h++) {
      const curr = queue[h];
      for (const eId of topology.adjacency[curr]) {
        const nxt = topology.edges[eId].to;
        if (!visited[nxt]) {
          visited[nxt] = 1;
          queue.push(nxt);
        }
      }
    }
    expect(queue.length).toBe(topology.nodes.length);
  });

  it('requires >= 22 edges, >= 4 cube faces, >= 2 underpasses, and >= 15 decision junctions across all acts', () => {
    expect(topology.solutionEdgePath.length).toBeGreaterThanOrEqual(22);
    expect(report.solutionFacesCount).toBeGreaterThanOrEqual(4);
    expect(report.solutionUnderpassCount).toBeGreaterThanOrEqual(2);
    expect(report.solutionDecisionJunctions).toBeGreaterThanOrEqual(15);

    const mid = Math.floor(topology.solutionNodePath.length / 2);
    let secondHalfJunctions = 0;
    let pyJunctions = 0;
    for (let i = 0; i < topology.solutionNodePath.length - 1; i++) {
      const nId = topology.solutionNodePath[i];
      if (topology.adjacency[nId].length >= 3) {
        if (i >= mid) secondHalfJunctions++;
        if (topology.nodes[nId].face === 'py') pyJunctions++;
      }
    }
    expect(secondHalfJunctions).toBeGreaterThanOrEqual(6);
    expect(pyJunctions).toBeGreaterThanOrEqual(3);

    // Verify solutionNodePath & solutionEdgePath consistency
    expect(topology.solutionNodePath[0]).toBe(topology.startNodeId);
    expect(topology.solutionNodePath[topology.solutionNodePath.length - 1]).toBe(
      topology.goalNodeId
    );
    expect(topology.solutionNodePath.length).toBe(topology.solutionEdgePath.length + 1);
    for (let i = 0; i < topology.solutionEdgePath.length; i++) {
      const e = topology.edges[topology.solutionEdgePath[i]];
      expect(e.from).toBe(topology.solutionNodePath[i]);
      expect(e.to).toBe(topology.solutionNodePath[i + 1]);
    }
  });

  it('maps 6-face UVs and high-res wet-atlas UVs consistently without cross-tile bleed', () => {
    for (const face of CUBE_FACES) {
      const node = topology.nodes.find((n) => n.face === face && n.tier === 0);
      expect(node).toBeDefined();
      if (!node) continue;

      const proj = localPointToFaceUv(node.localPos, topology.cubeHalfExtent);
      expect(proj.face).toBe(face);
      expect(proj.u01).toBeGreaterThan(0.05);
      expect(proj.u01).toBeLessThan(0.95);
      expect(proj.v01).toBeGreaterThan(0.05);
      expect(proj.v01).toBeLessThan(0.95);

      const out: [number, number] = [0, 0];
      const atlas = faceUvToAtlasUv(face, proj.u01, proj.v01, out);
      expect(atlas).toBe(out);
      expect(atlas[0]).toBeGreaterThan(proj.faceIndex / 6);
      expect(atlas[0]).toBeLessThan((proj.faceIndex + 1) / 6);
      expect(atlas[1]).toBeCloseTo(proj.v01, 5);
    }
  });
});

describe('Day 13 — Labyrinth: Physics, Spline Integration & Automated Solve', () => {
  const topology = createLabyrinthTopology();

  it('exhibits true (5/7)*g tangential free-roll acceleration parity along straight surface channels', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    // Edge 0 from Start (pz,1,5) -> (pz,1,6) runs straight in +Y along pz face
    // Find the main downward edge (pz,1,5)->(pz,1,4) which runs straight in -Y
    const downEdgeId = topology.adjacency[topology.startNodeId].find((id) => {
      const nTo = topology.nodes[topology.edges[id].to];
      return nTo.face === 'pz' && nTo.u === 1 && nTo.v === 4;
    });
    expect(downEdgeId).toBeDefined();

    const gMag = 9.81;
    const gLocal: [number, number, number] = [0, -gMag, 0];
    const dt = 0.09; // within single straight segment & single-frame clamp (0.10 s)
    sim.step(dt, gLocal);

    expect(sim.state.edgeId).toBe(downEdgeId);
    const expectedSpeedNoDrag = ROLLING_GRAVITY_FACTOR * gMag * dt;
    expect(sim.state.speed).toBeGreaterThan(expectedSpeedNoDrag * 0.94);
    expect(sim.state.speed).toBeLessThan(expectedSpeedNoDrag * 1.02);
  });

  it('produces zero NaNs across 10,000 substeps and paints wet Yellow Ochre RGBA trail', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    expect(sim.wetAtlasData.length).toBe(WET_ATLAS_WIDTH * WET_ATLAS_HEIGHT * 4);

    // 10,000 substeps: 10000 * (1/180) ≈ 55.6 s via 3,334 frames at dt = 1/60
    const g: [number, number, number] = [0, -9.81, 0];
    for (let f = 0; f < 3334; f++) {
      const phase = f * 0.03;
      g[0] = Math.sin(phase) * 7.5;
      g[1] = Math.cos(phase * 0.7) * 6.5;
      g[2] = Math.sin(phase * 1.3) * 5.0;
      sim.step(1 / 60, g);

      expect(Number.isFinite(sim.state.t)).toBe(true);
      expect(Number.isFinite(sim.state.speed)).toBe(true);
      expect(Number.isFinite(sim.state.impactImpulse)).toBe(true);
      expect(Number.isFinite(sim.state.solveProgress01)).toBe(true);
      for (let k = 0; k < 3; k++) {
        expect(Number.isFinite(sim.state.localPos[k])).toBe(true);
        expect(Number.isFinite(sim.state.localNormal[k])).toBe(true);
        expect(Number.isFinite(sim.state.localTangent[k])).toBe(true);
      }
    }

    // Verify wet Yellow Ochre trail painted into atlas with valid scalar verdigris alpha channel [0..255]
    let paintedTexels = 0;
    for (let i = 0; i < sim.wetAtlasData.length; i += 4) {
      expect(sim.wetAtlasData[i + 3]).toBeGreaterThanOrEqual(0);
      expect(sim.wetAtlasData[i + 3]).toBeLessThanOrEqual(255);
      if (sim.wetAtlasData[i + 0] > 30 && sim.wetAtlasData[i + 2] > 30) {
        paintedTexels++;
      }
    }
    expect(paintedTexels).toBeGreaterThan(20);
  });

  it('bounces off dead-ends with restitution e = 0.28 and monotonic bounce decay', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    // Drive bead into the +Y decoy dead-end (pz,1,5)->(pz,1,6)->(pz,2,6)->(pz,3,6)->(pz,4,6)
    // First move up (+Y) to (1,6), then strongly right (+X) toward (4,6)
    for (let i = 0; i < 45; i++) {
      sim.step(1 / 60, [0.5, 9.81, 0]);
    }
    // Now drive hard in +X into dead end (4,6) and observe speed peaks across bounces
    let maxApproachSpeed = 0;
    let firstBouncePeak = 0;
    let firstBounceImpact = 0;
    let sawReversal = false;
    let prevEdge = sim.state.edgeId;
    const bouncePeaks: number[] = [];
    let wasHeadingToDeadEnd = false;

    for (let i = 0; i < 180; i++) {
      sim.step(1 / 60, [9.81, 0.2, 0]);
      const curNodeTo = topology.nodes[topology.edges[sim.state.edgeId].to];
      const curNodeFrom = topology.nodes[topology.edges[sim.state.edgeId].from];

      if (curNodeTo.face === 'pz' && curNodeTo.u === 4 && curNodeTo.v === 6) {
        if (sim.state.speed > maxApproachSpeed) {
          maxApproachSpeed = sim.state.speed;
        }
      }
      if (
        prevEdge !== sim.state.edgeId &&
        curNodeFrom.face === 'pz' &&
        curNodeFrom.u === 4 &&
        curNodeFrom.v === 6 &&
        !sawReversal
      ) {
        sawReversal = true;
        firstBouncePeak = sim.state.speed;
        firstBounceImpact = sim.state.impactImpulse;
      }
      const headingIn = curNodeTo.face === 'pz' && curNodeTo.u === 4 && curNodeTo.v === 6;
      const bouncingOut = curNodeFrom.face === 'pz' && curNodeFrom.u === 4 && curNodeFrom.v === 6;
      if (wasHeadingToDeadEnd && bouncingOut) {
        bouncePeaks.push(sim.state.speed);
      }
      wasHeadingToDeadEnd = headingIn;
      prevEdge = sim.state.edgeId;
    }

    expect(sawReversal).toBe(true);
    expect(maxApproachSpeed).toBeGreaterThan(0.5);
    expect(firstBouncePeak).toBeGreaterThan(0);
    // Immediately after dead-end bounce, rebound speed <= ~e * approachSpeed (with slight drag/accel over 1/60s)
    expect(firstBouncePeak).toBeLessThanOrEqual(maxApproachSpeed * (DEAD_END_RESTITUTION + 0.08));
    expect(firstBounceImpact).toBeGreaterThan(0);
    expect(bouncePeaks.length).toBeGreaterThanOrEqual(2);
    for (let k = 1; k < bouncePeaks.length; k++) {
      expect(bouncePeaks[k]).toBeLessThanOrEqual(bouncePeaks[k - 1] + 1e-4);
    }
    // After bouncing finishes, dead-end settles cleanly with zero chatter or residual impact
    expect(sim.state.speed).toBe(0);
    expect(sim.state.impactImpulse).toBe(0);
  });

  it('completes an automated end-to-end geodesic solve from startNodeId to goalNodeId', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    const gOut: [number, number, number] = [0, 0, 0];

    let steps = 0;
    const maxSteps = 2400; // up to 40s sim time at 60Hz
    while (!sim.state.solved && steps < maxSteps) {
      sim.computeDownhillGravityForNextStep(gOut);
      sim.step(1 / 60, gOut);
      steps++;
    }

    expect(sim.state.solved).toBe(true);
    expect(sim.state.solveProgress01).toBe(1);

    const endEdge = topology.edges[sim.state.edgeId];
    const arrivedAtGoal =
      (endEdge.to === topology.goalNodeId && sim.state.t >= 0.9) ||
      (endEdge.from === topology.goalNodeId && sim.state.t <= 0.1);
    expect(arrivedAtGoal).toBe(true);

    // Reset returns cleanly to startNodeId with solved === false
    sim.reset();
    expect(sim.state.solved).toBe(false);
    expect(sim.state.solveProgress01).toBe(0);
    expect(topology.edges[sim.state.edgeId].from).toBe(topology.startNodeId);
    expect(sim.state.t).toBe(0);
  });

  it('recovers and solves from deep decoy wandering, preserves zero-alloc object identity, and paints bevel-wrap atlases seamlessly', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    const posRef = sim.state.localPos;
    const normRef = sim.state.localNormal;
    const tanRef = sim.state.localTangent;
    const gOut: [number, number, number] = [0, 0, 0];

    // Push bead into a decoy/arbitrary state with fluctuating gravity for 3 seconds
    for (let i = 0; i < 180; i++) {
      sim.step(1 / 60, [Math.cos(i * 0.05) * 9.81, Math.sin(i * 0.07) * 9.81, 4.0]);
    }
    expect(sim.state.localPos).toBe(posRef);
    expect(sim.state.localNormal).toBe(normRef);
    expect(sim.state.localTangent).toBe(tanRef);

    // Now run automated solver from that dirty off-path state
    let steps = 0;
    while (!sim.state.solved && steps < 3600) {
      const ret = sim.computeDownhillGravityForNextStep(gOut);
      expect(ret).toBe(gOut);
      sim.step(1 / 60, gOut);
      steps++;
    }
    expect(sim.state.solved).toBe(true);

    // Check that both portal faces on bevel-wrap edges in the solution path received yellow-ochre paint
    let facesWithPaint = 0;
    const faceStride = WET_ATLAS_FACE_SIZE;
    for (let f = 0; f < CUBE_FACES.length; f++) {
      let maxR = 0;
      const xStart = f * faceStride;
      for (let y = 0; y < WET_ATLAS_HEIGHT; y++) {
        for (let x = xStart; x < xStart + faceStride; x++) {
          const r = sim.wetAtlasData[(y * WET_ATLAS_WIDTH + x) * 4];
          if (r > maxR) maxR = r;
        }
      }
      if (maxR > 100) facesWithPaint++;
    }
    expect(facesWithPaint).toBeGreaterThanOrEqual(4);
  });

  it('is genuinely solvable by smooth human-rate SO(3) quaternion rotation under fixed world -Y gravity', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    const dt = 1 / 60;
    const maxAngularRate = 2.2; // rad/s realistic smooth hand trackball drag (strictly <= 2.4 rad/s)
    const maxAngleStep = maxAngularRate * dt;

    // Start from the default hero isometric quaternion
    const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -0.62);
    const pitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.7);
    const roll = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -0.52);
    const cubeQuat = new THREE.Quaternion().copy(roll).multiply(pitch).multiply(yaw).normalize();

    const worldGravity = new THREE.Vector3(0, -9.81, 0);
    const worldDown = new THREE.Vector3(0, -1, 0);
    const desiredLocalVec = new THREE.Vector3();
    const currentWorldDesired = new THREE.Vector3();
    const rotAxis = new THREE.Vector3();
    const deltaQuat = new THREE.Quaternion();
    const invQuat = new THREE.Quaternion();
    const localGravVec = new THREE.Vector3();
    const desiredGravOut: [number, number, number] = [0, 0, 0];
    const appliedLocalGrav: [number, number, number] = [0, -9.81, 0];

    const visitedUnderpasses = new Set<number>();
    let steps = 0;
    const maxSteps = 4000; // <= 66.7 s real play time

    while (!sim.state.solved && steps < maxSteps) {
      const curEdge = topology.edges[sim.state.edgeId];
      if (curEdge.kind === 'underpass') {
        visitedUnderpasses.add(curEdge.from);
      }

      sim.computeDownhillGravityForNextStep(desiredGravOut);
      desiredLocalVec.set(desiredGravOut[0], desiredGravOut[1], desiredGravOut[2]).normalize();
      currentWorldDesired.copy(desiredLocalVec).applyQuaternion(cubeQuat).normalize();

      const cosAngle = Math.max(-1, Math.min(1, currentWorldDesired.dot(worldDown)));
      const fullAngle = Math.acos(cosAngle);
      if (fullAngle > 1e-4) {
        rotAxis.crossVectors(currentWorldDesired, worldDown);
        if (rotAxis.lengthSq() < 1e-8) {
          rotAxis.set(1, 0, 0);
        } else {
          rotAxis.normalize();
        }
        const stepAngle = Math.min(fullAngle, maxAngleStep);
        expect(stepAngle / dt).toBeLessThanOrEqual(2.4);
        deltaQuat.setFromAxisAngle(rotAxis, stepAngle);
        cubeQuat.premultiply(deltaQuat).normalize();
      }

      invQuat.copy(cubeQuat).invert();
      localGravVec.copy(worldGravity).applyQuaternion(invQuat);
      appliedLocalGrav[0] = localGravVec.x;
      appliedLocalGrav[1] = localGravVec.y;
      appliedLocalGrav[2] = localGravVec.z;

      sim.step(dt, appliedLocalGrav);
      steps++;
    }

    expect(sim.state.solved).toBe(true);
    expect(visitedUnderpasses.size).toBeGreaterThanOrEqual(4);

    // Verify solution ignition paints all 6 faces cleanly
    sim.igniteSolutionPathWave(1.0);
    let paintedFaces = 0;
    for (let f = 0; f < CUBE_FACES.length; f++) {
      let maxR = 0;
      const xStart = f * WET_ATLAS_FACE_SIZE;
      for (let y = 0; y < WET_ATLAS_HEIGHT; y++) {
        for (let x = xStart; x < xStart + WET_ATLAS_FACE_SIZE; x++) {
          const r = sim.wetAtlasData[(y * WET_ATLAS_WIDTH + x) * 4];
          if (r > maxR) maxR = r;
        }
      }
      if (maxR > 180) paintedFaces++;
    }
    expect(paintedFaces).toBe(6);
  });

  it('diverts passive boot gravity and pure -Y drop into decoy loops rather than auto-solving', () => {
    const simHero = new LabyrinthBeadSimulator(topology);
    const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -0.62);
    const pitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.7);
    const roll = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -0.52);
    const heroInv = new THREE.Quaternion().copy(roll).multiply(pitch).multiply(yaw).normalize().invert();
    const heroLocalG = new THREE.Vector3(0, -9.81, 0).applyQuaternion(heroInv);
    const heroGrav: [number, number, number] = [heroLocalG.x, heroLocalG.y, heroLocalG.z];

    for (let i = 0; i < 180; i++) {
      simHero.step(1 / 60, heroGrav);
    }
    expect(simHero.state.solved).toBe(false);
    expect(simHero.state.solveProgress01).toBeLessThan(0.12);

    const simDown = new LabyrinthBeadSimulator(topology);
    for (let i = 0; i < 180; i++) {
      simDown.step(1 / 60, [0, -9.81, 0]);
    }
    expect(simDown.state.solved).toBe(false);
    expect(simDown.state.solveProgress01).toBeLessThan(0.12);
  });

  it('maintains C1 tangent and normal continuity across surface, bevel-wrap, ramp, and underpass joints', () => {
    const normalize = (a: number, b: number, c: number): [number, number, number] => {
      const len = Math.hypot(a, b, c) || 1;
      return [a / len, b / len, c / len];
    };

    const getStartEndTangents = (edgeIdx: number) => {
      const e = topology.edges[edgeIdx];
      const p = e.samplesPos;
      // 2nd-order finite-difference tangent at k=0 and k=8
      const st = normalize(
        -3 * p[0] + 4 * p[3] - p[6],
        -3 * p[1] + 4 * p[4] - p[7],
        -3 * p[2] + 4 * p[5] - p[8]
      );
      const n = EDGE_SAMPLE_COUNT - 1;
      const i0 = (n - 2) * 3;
      const i1 = (n - 1) * 3;
      const i2 = n * 3;
      const et = normalize(
        3 * p[i2 + 0] - 4 * p[i1 + 0] + p[i0 + 0],
        3 * p[i2 + 1] - 4 * p[i1 + 1] + p[i0 + 1],
        3 * p[i2 + 2] - 4 * p[i1 + 2] + p[i0 + 2]
      );
      return { st, et };
    };

    // Check normal continuity across ALL joints, horizontal tangent flushness at ramp/bevel endpoints,
    // and collinear C1 tangent continuity across ramp<->underpass transitions
    for (let i = 0; i < topology.solutionEdgePath.length - 1; i++) {
      const eA = topology.edges[topology.solutionEdgePath[i]];
      const eB = topology.edges[topology.solutionEdgePath[i + 1]];
      const nLast = (EDGE_SAMPLE_COUNT - 1) * 3;
      const dotNorm =
        eA.samplesNormal[nLast + 0] * eB.samplesNormal[0] +
        eA.samplesNormal[nLast + 1] * eB.samplesNormal[1] +
        eA.samplesNormal[nLast + 2] * eB.samplesNormal[2];
      expect(dotNorm).toBeGreaterThan(0.999);

      const tA = getStartEndTangents(eA.id).et;
      const tB = getStartEndTangents(eB.id).st;
      const endNormalDot =
        tA[0] * eA.samplesNormal[nLast + 0] +
        tA[1] * eA.samplesNormal[nLast + 1] +
        tA[2] * eA.samplesNormal[nLast + 2];
      const startNormalDot =
        tB[0] * eB.samplesNormal[0] +
        tB[1] * eB.samplesNormal[1] +
        tB[2] * eB.samplesNormal[2];
      expect(Math.abs(endNormalDot)).toBeLessThan(0.06);
      expect(Math.abs(startNormalDot)).toBeLessThan(0.06);

      const dotTan = tA[0] * tB[0] + tA[1] * tB[1] + tA[2] * tB[2];
      if (
        (eA.kind === 'ramp' && eB.kind === 'underpass') ||
        (eA.kind === 'underpass' && eB.kind === 'ramp')
      ) {
        expect(dotTan).toBeGreaterThan(0.99);
      }
    }
  });

  it('unlocks 3 exterior one-way checkpoints sequentially, locks completed acts in verdigris, and guarantees zero downstream deadlocks', () => {
    const sim = new LabyrinthBeadSimulator(topology);

    // Every checkpoint must sit on an exterior (tier = 0) corridor floor
    for (let k = 0; k < CHECKPOINT_SOL_INDICES.length; k++) {
      const cpNode = topology.nodes[topology.solutionNodePath[CHECKPOINT_SOL_INDICES[k]]];
      expect(cpNode.tier).toBe(0);
    }

    // Verify graph-theoretic deadlock-freedom when backward cut-edges are locked at stages 1, 2, 3
    const cpBackEdges = CHECKPOINT_SOL_INDICES.map((solIdx) => {
      const cpId = topology.solutionNodePath[solIdx];
      const prevId = topology.solutionNodePath[solIdx - 1];
      return topology.adjacency[cpId].find((eid) => topology.edges[eid].to === prevId)!;
    });
    for (let stage = 1; stage <= 3; stage++) {
      const blocked = new Set(cpBackEdges.slice(0, stage));
      const cpStart = topology.solutionNodePath[CHECKPOINT_SOL_INDICES[stage - 1]];
      const reach = new Set<number>([cpStart]);
      const q = [cpStart];
      for (let h = 0; h < q.length; h++) {
        const u = q[h];
        for (const eid of topology.adjacency[u]) {
          if (blocked.has(eid)) continue;
          const v = topology.edges[eid].to;
          if (!reach.has(v)) {
            reach.add(v);
            q.push(v);
          }
        }
      }
      const canReachGoal = new Set<number>([topology.goalNodeId]);
      const rq = [topology.goalNodeId];
      for (let h = 0; h < rq.length; h++) {
        const v = rq[h];
        for (const revEid of topology.adjacency[v]) {
          const u = topology.edges[revEid].to;
          const fwd = topology.adjacency[u].find((eid) => topology.edges[eid].to === v)!;
          if (blocked.has(fwd)) continue;
          if (!canReachGoal.has(u)) {
            canReachGoal.add(u);
            rq.push(u);
          }
        }
      }
      for (const n of reach) {
        expect(canReachGoal.has(n)).toBe(true);
      }
      expect(reach.has(topology.solutionNodePath[CHECKPOINT_SOL_INDICES[stage - 1] - 1])).toBe(false);
    }

    // Drive forward past Checkpoint 0 and verify sequential unlock + act crystallization + ratchet stop
    const gOut: [number, number, number] = [0, 0, 0];
    let guard = 0;
    while (sim.state.checkpointsUnlocked < 1 && guard < 1200) {
      sim.computeDownhillGravityForNextStep(gOut);
      sim.step(1 / 60, gOut);
      guard++;
    }
    expect(sim.state.checkpointsUnlocked).toBe(1);

    // At least part of Act 1 is crystallized with strong verdigris alpha (>= 220)
    let act1VerdigrisCount = 0;
    for (let i = 3; i < sim.wetAtlasData.length; i += 4) {
      if (sim.wetAtlasData[i] >= 220) act1VerdigrisCount++;
    }
    expect(act1VerdigrisCount).toBeGreaterThan(80);

    // Push strongly backward toward Checkpoint 0's locked previous node; must NOT cross back into Act 1
    const cp0Node = topology.nodes[sim.checkpointNodeIds[0]];
    const prev0Node = topology.nodes[topology.solutionNodePath[CHECKPOINT_SOL_INDICES[0] - 1]];
    const backGrav: [number, number, number] = [
      (prev0Node.localPos[0] - cp0Node.localPos[0]) * 45,
      (prev0Node.localPos[1] - cp0Node.localPos[1]) * 45,
      (prev0Node.localPos[2] - cp0Node.localPos[2]) * 45,
    ];
    for (let i = 0; i < 180; i++) {
      sim.step(1 / 60, backGrav);
      const curFrom = topology.edges[sim.state.edgeId].from;
      const curTo = topology.edges[sim.state.edgeId].to;
      expect(curFrom).not.toBe(prev0Node.id);
      expect(curTo).not.toBe(prev0Node.id);
    }

    // Continue solving to completion; checkpoints 2 and 3 must unlock strictly in order
    const unlockSeq = [sim.state.checkpointsUnlocked];
    while (!sim.state.solved && guard < 3600) {
      const prev = sim.state.checkpointsUnlocked;
      sim.computeDownhillGravityForNextStep(gOut);
      sim.step(1 / 60, gOut);
      if (sim.state.checkpointsUnlocked !== prev) {
        unlockSeq.push(sim.state.checkpointsUnlocked);
      }
      guard++;
    }
    expect(sim.state.solved).toBe(true);
    expect(unlockSeq).toEqual([1, 2, 3]);
  });

  it('keeps dowsing hint strictly zero on decoys and subtly bounded near checkpoints', () => {
    const sim = new LabyrinthBeadSimulator(topology);

    // Drive directly into decoy branch (pz,1,5) -> (1,6) -> (4,6)
    for (let i = 0; i < 60; i++) {
      sim.step(1 / 60, [0.5, 9.81, 0]);
      expect(sim.state.dowsingHint01).toBe(0);
    }
    for (let i = 0; i < 90; i++) {
      sim.step(1 / 60, [9.81, 0.2, 0]);
      expect(sim.state.dowsingHint01).toBe(0);
    }
    for (let i = 3; i < sim.wetAtlasData.length; i += 4) {
      expect(sim.wetAtlasData[i]).toBe(0);
    }

    // Reset and advance along solution toward Checkpoint 0, verifying subtle bounded alpha before crossing
    sim.reset();
    const gOut: [number, number, number] = [0, 0, 0];
    let maxDowsingBeforeCp = 0;
    let maxAlphaWhileDowsing = 0;
    let steps = 0;
    while (sim.state.checkpointsUnlocked === 0 && steps < 900) {
      sim.computeDownhillGravityForNextStep(gOut);
      sim.step(1 / 60, gOut);
      if (sim.state.checkpointsUnlocked === 0) {
        if (sim.state.dowsingHint01 > maxDowsingBeforeCp) {
          maxDowsingBeforeCp = sim.state.dowsingHint01;
        }
        if (sim.state.dowsingHint01 > 0.35) {
          for (let i = 3; i < sim.wetAtlasData.length; i += 4) {
            if (sim.wetAtlasData[i] > maxAlphaWhileDowsing) {
              maxAlphaWhileDowsing = sim.wetAtlasData[i];
            }
          }
        }
      }
      steps++;
    }
    expect(maxDowsingBeforeCp).toBeGreaterThan(0.15);
    expect(maxDowsingBeforeCp).toBeLessThanOrEqual(1);
    expect(maxAlphaWhileDowsing).toBeGreaterThan(0);
    expect(maxAlphaWhileDowsing).toBeLessThanOrEqual(60);
  });

  it('transmutes solution river to full verdigris while preserving decoy ochre and resets cleanly', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    // Paint a decoy trail on pz(4,6) dead-end first
    for (let i = 0; i < 45; i++) sim.step(1 / 60, [0.5, 9.81, 0]);
    for (let i = 0; i < 90; i++) sim.step(1 / 60, [9.81, 0.2, 0]);

    const deadEndNode = topology.nodes.find((n) => n.face === 'pz' && n.u === 4 && n.v === 6)!;
    const decoyProj = localPointToFaceUv(deadEndNode.localPos, topology.cubeHalfExtent);
    const dx = Math.round(decoyProj.u01 * (WET_ATLAS_FACE_SIZE - 1));
    const dy = Math.round(decoyProj.v01 * (WET_ATLAS_FACE_SIZE - 1));
    const decoyByteIdx =
      (dy * WET_ATLAS_WIDTH + decoyProj.faceIndex * WET_ATLAS_FACE_SIZE + dx) * 4;
    expect(sim.wetAtlasData[decoyByteIdx + 0]).toBeGreaterThan(60); // painted ochre/wetness
    expect(sim.wetAtlasData[decoyByteIdx + 3]).toBe(0); // zero verdigris on decoy

    // Ignite solution wave -> stamps full verdigris across all 47 solution edges, decoy stays non-green
    sim.igniteSolutionPathWave(1.0);
    let maxSolAlpha = 0;
    for (let i = 3; i < sim.wetAtlasData.length; i += 4) {
      if (sim.wetAtlasData[i] > maxSolAlpha) maxSolAlpha = sim.wetAtlasData[i];
    }
    expect(maxSolAlpha).toBeGreaterThanOrEqual(240);
    expect(sim.wetAtlasData[decoyByteIdx + 3]).toBe(0);

    // Reset clears checkpoints, dowsing, and every verdigris texel back to 0
    sim.reset();
    expect(sim.state.checkpointsUnlocked).toBe(0);
    expect(sim.state.dowsingHint01).toBe(0);
    for (let i = 3; i < sim.wetAtlasData.length; i += 4) {
      expect(sim.wetAtlasData[i]).toBe(0);
    }
  });

  it('captures intentional diagonal camera-facing tilts into 90-degree pass-through channels while rolling straight on shallow tilts', () => {
    const tiltRad = (24 * Math.PI) / 180;
    const gMag = 9.81 * Math.sin(tiltRad);
    const gZ = -9.81 * Math.cos(tiltRad);

    // 1. Shallow 18° forward-down tilt from Start rolls straight past pz[3,5] into pz[4,5] -> pz[4,4] decoy
    const simShallow = new LabyrinthBeadSimulator(topology);
    const r18 = (18 * Math.PI) / 180;
    let shallowEnteredCenter = false;
    let shallowPassedToPz45 = false;
    for (let i = 0; i < 180; i++) {
      simShallow.step(1 / 60, [gMag * Math.cos(r18), -gMag * Math.sin(r18), gZ]);
      const nTo = topology.nodes[topology.edges[simShallow.state.edgeId].to];
      if (nTo.face === 'pz' && nTo.u === 4 && nTo.v === 5) shallowPassedToPz45 = true;
      if (nTo.face === 'pz' && nTo.u === 3 && nTo.v <= 4) shallowEnteredCenter = true;
    }
    const shallowTo = topology.nodes[topology.edges[simShallow.state.edgeId].to];
    expect(shallowEnteredCenter).toBe(false);
    expect(shallowPassedToPz45).toBe(true);
    expect(shallowTo.face).toBe('pz');
    expect(shallowTo.u).toBe(4);
    expect(shallowTo.v).toBe(4);

    // 2. Intentional 32° and 40° diagonal tilts from Start enter the pz[3,5]->pz[3,4]->pz[3,3] pass-through underpass
    for (const angDeg of [32, 40]) {
      const simDiag = new LabyrinthBeadSimulator(topology);
      const rad = (angDeg * Math.PI) / 180;
      let enteredPassThrough = false;
      for (let i = 0; i < 180; i++) {
        simDiag.step(1 / 60, [gMag * Math.cos(rad), -gMag * Math.sin(rad), gZ]);
        const nTo = topology.nodes[topology.edges[simDiag.state.edgeId].to];
        if (nTo.face === 'pz' && nTo.u === 3 && nTo.v <= 3 && nTo.tier === 1) {
          enteredPassThrough = true;
        }
      }
      expect(enteredPassThrough).toBe(true);
    }

    // 3. Steep 55° careless sprint at 30° diagonal vaults past pz[3,5] rim onto pz[4,5] instead of magnetic snap
    const tilt55 = (55 * Math.PI) / 180;
    const g55 = 9.81 * Math.sin(tilt55);
    const gz55 = -9.81 * Math.cos(tilt55);
    const d30 = (30 * Math.PI) / 180;
    const simSprint = new LabyrinthBeadSimulator(topology);
    let skippedToPz45 = false;
    for (let i = 0; i < 90; i++) {
      simSprint.step(1 / 60, [g55 * Math.cos(d30), -g55 * Math.sin(d30), gz55]);
      const nTo = topology.nodes[topology.edges[simSprint.state.edgeId].to];
      if (nTo.face === 'pz' && nTo.u >= 4 && nTo.v === 5) {
        skippedToPz45 = true;
      }
    }
    expect(skippedToPz45).toBe(true);

    // 4. Backing out from pz[4,5] at 36° down-left enters pz[3,5]->pz[3,4], then steering down enters pz[3,3] underpass
    const simBack = new LabyrinthBeadSimulator(topology);
    const n28 = topology.nodes.find((n) => n.face === 'pz' && n.u === 4 && n.v === 5 && n.tier === 0)!;
    const n22 = topology.nodes.find((n) => n.face === 'pz' && n.u === 3 && n.v === 5 && n.tier === 0)!;
    const backEdgeId = topology.adjacency[n28.id].find((id) => topology.edges[id].to === n22.id)!;
    simBack.state.edgeId = backEdgeId;
    simBack.state.t = 0;
    simBack.state.speed = 0;

    const r36 = (36 * Math.PI) / 180;
    let enteredCenterChannel = false;
    let enteredFromBack = false;
    for (let i = 0; i < 180; i++) {
      const eCur = topology.edges[simBack.state.edgeId];
      const nFrom = topology.nodes[eCur.from];
      const curTo = topology.nodes[eCur.to];
      if (nFrom.face === 'pz' && nFrom.u === 3 && nFrom.v === 5 && curTo.u === 3 && curTo.v === 4) {
        enteredCenterChannel = true;
      }
      if (enteredCenterChannel) {
        simBack.step(1 / 60, [0, -gMag, gZ]);
      } else {
        simBack.step(1 / 60, [-gMag * Math.cos(r36), -gMag * Math.sin(r36), gZ]);
      }
      const nTo = topology.nodes[topology.edges[simBack.state.edgeId].to];
      if (nTo.face === 'pz' && nTo.u === 3 && nTo.v <= 3 && nTo.tier === 1) {
        enteredFromBack = true;
      }
    }
    expect(enteredCenterChannel).toBe(true);
    expect(enteredFromBack).toBe(true);
  });

  it('preserves corridor floorMask across antipodal bridge projections (px[3,3], ny[2,3], nz[3,2])', () => {
    const geo = createLabyrinthGeometries(topology).monolithGeometry;
    const pos = geo.getAttribute('position');
    const norm = geo.getAttribute('normal');
    const detail = geo.getAttribute('aStoneDetail');

    const measureFloorRatio = (n: (typeof topology.nodes)[number]) => {
      let total = 0;
      let painted = 0;
      for (let i = 0; i < pos.count; i++) {
        const dx = pos.getX(i) - n.localPos[0];
        const dy = pos.getY(i) - n.localPos[1];
        const dz = pos.getZ(i) - n.localPos[2];
        if (dx * dx + dy * dy + dz * dz < 0.08 * 0.08) {
          const dot =
            norm.getX(i) * n.localNormal[0] +
            norm.getY(i) * n.localNormal[1] +
            norm.getZ(i) * n.localNormal[2];
          if (dot > 0.75) {
            total++;
            if (detail.getX(i) >= 0.75) painted++;
          }
        }
      }
      return { total, ratio: total > 0 ? painted / total : 0 };
    };

    let zeroedNodes = 0;
    for (const n of topology.nodes) {
      const m = measureFloorRatio(n);
      if (m.total === 0 || m.ratio < 0.8) zeroedNodes++;
    }
    expect(zeroedNodes).toBe(0);

    for (const [f, u, v, minRatio] of [
      ['px', 3, 3, 0.9],
      ['ny', 2, 3, 0.8],
      ['nz', 3, 2, 0.95],
    ] as const) {
      const node = topology.nodes.find((x) => x.face === f && x.u === u && x.v === v && x.tier === 0)!;
      const m = measureFloorRatio(node);
      expect(m.total).toBeGreaterThan(100);
      expect(m.ratio).toBeGreaterThanOrEqual(minRatio);
    }
  });

  it('isolates tier-0 overpass bridge decks (row 0) from tier-1 underpass floors (row 1) in 1536x512 atlas', () => {
    expect(WET_ATLAS_HEIGHT).toBe(WET_ATLAS_FACE_SIZE * 2);
    const sim = new LabyrinthBeadSimulator(topology);
    sim.igniteSolutionPathWave(1.0);

    for (const br of topology.bridges) {
      const n0 = topology.nodes.find(
        (n) => n.face === br.face && n.u === br.u && n.v === br.v && n.tier === 0
      )!;
      const proj = localPointToFaceUv(n0.localPos, topology.cubeHalfExtent);
      const dx = Math.round(proj.u01 * (WET_ATLAS_FACE_SIZE - 1));
      const dy = Math.round(proj.v01 * (WET_ATLAS_FACE_SIZE - 1));
      const idxRow0 = (dy * WET_ATLAS_WIDTH + proj.faceIndex * WET_ATLAS_FACE_SIZE + dx) * 4;
      const idxRow1 =
        ((WET_ATLAS_FACE_SIZE + dy) * WET_ATLAS_WIDTH + proj.faceIndex * WET_ATLAS_FACE_SIZE + dx) * 4;

      expect(sim.wetAtlasData[idxRow0 + 0]).toBe(0);
      expect(sim.wetAtlasData[idxRow0 + 3]).toBe(0);
      expect(sim.wetAtlasData[idxRow1 + 0]).toBeGreaterThanOrEqual(220);
      expect(sim.wetAtlasData[idxRow1 + 3]).toBeGreaterThanOrEqual(220);
    }
  });

  it('simulates 2-DOF wall hugging, no-slip rollQuat spin, and 90-degree L-corner jamb deflection', () => {
    const simLat = new LabyrinthBeadSimulator(topology);
    let maxAbsLat = 0;
    for (let i = 0; i < 30; i++) {
      simLat.step(1 / 60, [4.5, -4.0, -7.5]);
      if (Math.abs(simLat.state.latOffset) > maxAbsLat) {
        maxAbsLat = Math.abs(simLat.state.latOffset);
      }
    }
    expect(maxAbsLat).toBeGreaterThan(0.014);
    expect(maxAbsLat).toBeLessThanOrEqual(GROOVE_MAX_LAT_OFFSET + 1e-6);

    const rq = simLat.state.rollQuat;
    expect(Math.hypot(rq[0], rq[1], rq[2], rq[3])).toBeCloseTo(1, 5);
    expect(Math.abs(rq[3])).toBeLessThan(0.999);

    const simL = new LabyrinthBeadSimulator(topology);
    let preCornerSpeed = 0;
    let postCornerSpeed = 0;
    let turnedCorner = false;
    let prevEdge = simL.state.edgeId;

    for (let i = 0; i < 60; i++) {
      const before = simL.state.speed;
      simL.step(1 / 60, [3.2, 9.2, 0]);
      if (simL.state.edgeId !== prevEdge) {
        const e = topology.edges[simL.state.edgeId];
        const fn = topology.nodes[e.from];
        const tn = topology.nodes[e.to];
        if (fn.u === 1 && fn.v === 6 && tn.u === 2 && tn.v === 6) {
          turnedCorner = true;
          preCornerSpeed = before;
          postCornerSpeed = simL.state.speed;
          break;
        }
        prevEdge = simL.state.edgeId;
      }
    }
    expect(turnedCorner).toBe(true);
    expect(preCornerSpeed).toBeGreaterThan(0.8);
    expect(postCornerSpeed).toBeLessThan(preCornerSpeed * 0.55);
    expect(simL.state.impactImpulse).toBeGreaterThan(0);
  });

  it('settles with zero position or spin stutter in degree-2 L-corner and T-junction V-pockets (live-traced nx[2,3] regression)', () => {
    const dt = 1 / 60;
    const cornerNode = topology.nodes.find(
      (n) => n.face === 'nx' && n.u === 2 && n.v === 3 && n.tier === 0
    )!;
    const outFromCorner = topology.adjacency[cornerNode.id];
    expect(outFromCorner.length).toBe(2);

    // Pick incoming edge from nx[2,4] -> nx[2,3] so gravity localG pulls downhill into the corner V-pocket
    const incomingEdge = topology.edges.find(
      (e) =>
        e.to === cornerNode.id &&
        topology.nodes[e.from].face === 'nx' &&
        topology.nodes[e.from].u === 2 &&
        topology.nodes[e.from].v === 4
    )!;
    const exitEdge = topology.edges.find(
      (e) =>
        e.from === cornerNode.id &&
        topology.nodes[e.to].face === 'nx' &&
        topology.nodes[e.to].u === 3 &&
        topology.nodes[e.to].v === 3
    )!;

    const sim = new LabyrinthBeadSimulator(topology);
    (sim.state as { edgeId: number; t: number; speed: number }).edgeId = incomingEdge.id;
    (sim.state as { t: number }).t = 0.4;
    (sim.state as { speed: number }).speed = 0;

    // Exact stationary live-traced cube gravity pointing into the nx[2,3] 90-degree V-pocket
    const pocketGrav: [number, number, number] = [2.19805, -6.14270, -7.32611];
    for (let i = 0; i < 90; i++) {
      sim.step(dt, pocketGrav);
    }

    // Over 120 subsequent frames at rest, zero edge flips, zero position jitter, zero spin jitter
    let maxPosDelta = 0;
    let maxQuatDelta = 0;
    let edgeFlips = 0;
    let lastEdgeId = sim.state.edgeId;
    let lastPos = [...sim.state.localPos];
    let lastQuat = [...sim.state.rollQuat];

    for (let i = 0; i < 120; i++) {
      sim.step(dt, pocketGrav);
      if (sim.state.edgeId !== lastEdgeId) edgeFlips++;
      const dp = Math.hypot(
        sim.state.localPos[0] - lastPos[0],
        sim.state.localPos[1] - lastPos[1],
        sim.state.localPos[2] - lastPos[2]
      );
      const dq = Math.hypot(
        sim.state.rollQuat[0] - lastQuat[0],
        sim.state.rollQuat[1] - lastQuat[1],
        sim.state.rollQuat[2] - lastQuat[2],
        sim.state.rollQuat[3] - lastQuat[3]
      );
      if (dp > maxPosDelta) maxPosDelta = dp;
      if (dq > maxQuatDelta) maxQuatDelta = dq;
      lastEdgeId = sim.state.edgeId;
      lastPos = [...sim.state.localPos];
      lastQuat = [...sim.state.rollQuat];
    }

    expect(edgeFlips).toBe(0);
    expect(sim.state.speed).toBe(0);
    expect(sim.state.latSpeed).toBe(0);
    expect(maxPosDelta).toBeLessThan(1e-7);
    expect(maxQuatDelta).toBeLessThan(1e-7);

    // Tilting downhill along the +Z arm launches without any 23mm orthogonal binormal snap
    const posBeforeLaunch = [...sim.state.localPos];
    const launchGrav: [number, number, number] = [2.2, -2.0, 7.5];
    sim.step(dt, launchGrav);
    const firstFrameJump = Math.hypot(
      sim.state.localPos[0] - posBeforeLaunch[0],
      sim.state.localPos[1] - posBeforeLaunch[1],
      sim.state.localPos[2] - posBeforeLaunch[2]
    );
    expect(firstFrameJump).toBeLessThan(0.005);

    for (let i = 1; i < 30; i++) {
      sim.step(dt, launchGrav);
    }
    expect(sim.state.edgeId === exitEdge.id || sim.state.speed > 0.2).toBe(true);
    expect(sim.state.speed).toBeGreaterThan(0.2);

    // Stem-into-bar degree-3 T-junction held head-on against the crossbar settles with zero jitter
    const tNode = topology.nodes.find((n) => {
      if (n.tier !== 0 || n.id === topology.goalNodeId) return false;
      const out = topology.adjacency[n.id];
      if (out.length !== 3) return false;
      return out.some((aid, idxA) =>
        out.some((bid, idxB) => {
          if (idxA === idxB) return false;
          const pa = topology.edges[aid].samplesPos;
          const pb = topology.edges[bid].samplesPos;
          const dA = Math.hypot(pa[3] - pa[0], pa[4] - pa[1], pa[5] - pa[2]) || 1;
          const dB = Math.hypot(pb[3] - pb[0], pb[4] - pb[1], pb[5] - pb[2]) || 1;
          const dot =
            ((pa[3] - pa[0]) * (pb[3] - pb[0]) +
              (pa[4] - pa[1]) * (pb[4] - pb[1]) +
              (pa[5] - pa[2]) * (pb[5] - pb[2])) /
            (dA * dB);
          return dot < -0.95;
        })
      );
    })!;
    expect(tNode).toBeDefined();

    // Find the orthogonal stem outgoing edge and its incoming twin
    const tOut = topology.adjacency[tNode.id];
    const stemOutId = tOut.find((candId) => {
      let oppDots = 0;
      const pc = topology.edges[candId].samplesPos;
      const dc = Math.hypot(pc[3] - pc[0], pc[4] - pc[1], pc[5] - pc[2]) || 1;
      for (const otherId of tOut) {
        if (otherId === candId) continue;
        const po = topology.edges[otherId].samplesPos;
        const doLen = Math.hypot(po[3] - po[0], po[4] - po[1], po[5] - po[2]) || 1;
        const dot =
          ((pc[3] - pc[0]) * (po[3] - po[0]) +
            (pc[4] - pc[1]) * (po[4] - po[1]) +
            (pc[5] - pc[2]) * (po[5] - po[2])) /
          (dc * doLen);
        if (Math.abs(dot) < 0.2) oppDots++;
      }
      return oppDots === 2;
    })!;
    const stemOutEdge = topology.edges[stemOutId];
    const stemInEdge = topology.edges.find(
      (e) => e.from === stemOutEdge.to && e.to === tNode.id
    )!;
    const pS = stemOutEdge.samplesPos;
    const dS = Math.hypot(pS[3] - pS[0], pS[4] - pS[1], pS[5] - pS[2]) || 1;
    const stemInDir: [number, number, number] = [
      -(pS[3] - pS[0]) / dS,
      -(pS[4] - pS[1]) / dS,
      -(pS[5] - pS[2]) / dS,
    ];
    const tWallGrav: [number, number, number] = [
      stemInDir[0] * 8.5 - tNode.localNormal[0] * 4.8,
      stemInDir[1] * 8.5 - tNode.localNormal[1] * 4.8,
      stemInDir[2] * 8.5 - tNode.localNormal[2] * 4.8,
    ];

    const simT = new LabyrinthBeadSimulator(topology);
    (simT.state as { edgeId: number; t: number; speed: number }).edgeId = stemInEdge.id;
    (simT.state as { t: number }).t = 0.998;
    (simT.state as { speed: number }).speed = 0.02;
    for (let i = 0; i < 60; i++) {
      simT.step(dt, tWallGrav);
    }
    let maxTPosDelta = 0;
    let tEdgeFlips = 0;
    let tLastEdge = simT.state.edgeId;
    let tLastPos = [...simT.state.localPos];
    for (let i = 0; i < 60; i++) {
      simT.step(dt, tWallGrav);
      if (simT.state.edgeId !== tLastEdge) tEdgeFlips++;
      const dp = Math.hypot(
        simT.state.localPos[0] - tLastPos[0],
        simT.state.localPos[1] - tLastPos[1],
        simT.state.localPos[2] - tLastPos[2]
      );
      if (dp > maxTPosDelta) maxTPosDelta = dp;
      tLastEdge = simT.state.edgeId;
      tLastPos = [...simT.state.localPos];
    }
    expect(tEdgeFlips).toBe(0);
    expect(simT.state.speed).toBe(0);
    expect(maxTPosDelta).toBeLessThan(1e-7);

    // Locked one-way checkpoint catch-basin held under reverse gravity rests cleanly with zero chatter
    const simCp = new LabyrinthBeadSimulator(topology);
    const cpNodeId = simCp.checkpointNodeIds[0];
    const cpNode = topology.nodes[cpNodeId];
    const cpSolIdx = CHECKPOINT_SOL_INDICES[0];
    const nextSolNodeId = topology.solutionNodePath[cpSolIdx + 1];
    const prevSolNodeId = topology.solutionNodePath[cpSolIdx - 1];
    const cpFwdEdge = topology.edges.find((e) => e.from === cpNodeId && e.to === nextSolNodeId)!;
    const pCp = cpFwdEdge.samplesPos;
    const dCp = Math.hypot(pCp[3] - pCp[0], pCp[4] - pCp[1], pCp[5] - pCp[2]) || 1;
    const cpFwdDir: [number, number, number] = [
      (pCp[3] - pCp[0]) / dCp,
      (pCp[4] - pCp[1]) / dCp,
      (pCp[5] - pCp[2]) / dCp,
    ];
    // Push directly backward against the locked checkpoint while forcing any unblocked side arm neutral/uphill
    const cpRevGrav: [number, number, number] = [
      -cpFwdDir[0] * 8.0 - cpNode.localNormal[0] * 4.8,
      -cpFwdDir[1] * 8.0 - cpNode.localNormal[1] * 4.8,
      -cpFwdDir[2] * 8.0 - cpNode.localNormal[2] * 4.8,
    ];
    expect(topology.adjacency[cpNodeId].filter((eid) => topology.edges[eid].to !== prevSolNodeId).length).toBeGreaterThanOrEqual(1);
    simCp.state.checkpointsUnlocked = 1;
    (simCp.state as { edgeId: number; t: number; speed: number }).edgeId = cpFwdEdge.id;
    (simCp.state as { t: number }).t = 0.25;
    (simCp.state as { speed: number }).speed = 0;
    for (let i = 0; i < 90; i++) {
      simCp.step(dt, cpRevGrav);
    }
    let cpEdgeFlips = 0;
    let maxCpPosDelta = 0;
    let cpLastEdge = simCp.state.edgeId;
    let cpLastPos = [...simCp.state.localPos];
    for (let i = 0; i < 60; i++) {
      simCp.step(dt, cpRevGrav);
      if (simCp.state.edgeId !== cpLastEdge) cpEdgeFlips++;
      const dp = Math.hypot(
        simCp.state.localPos[0] - cpLastPos[0],
        simCp.state.localPos[1] - cpLastPos[1],
        simCp.state.localPos[2] - cpLastPos[2]
      );
      if (dp > maxCpPosDelta) maxCpPosDelta = dp;
      cpLastEdge = simCp.state.edgeId;
      cpLastPos = [...simCp.state.localPos];
    }
    expect(cpEdgeFlips).toBe(0);
    expect(maxCpPosDelta).toBeLessThan(1e-7);
  });

  it('preserves C0 3D position continuity across every 90-degree corner and mouth turn during full solve', () => {
    const sim = new LabyrinthBeadSimulator(topology);
    const gOut: [number, number, number] = [0, 0, 0];
    const dt = 1 / 60;

    let maxExcessJump = 0;
    let steps = 0;
    while (!sim.state.solved && steps < 2400) {
      const prevEdge = sim.state.edgeId;
      const prevPos = [...sim.state.localPos];
      const prevTotalVel = Math.hypot(sim.state.speed, sim.state.latSpeed);
      sim.computeDownhillGravityForNextStep(gOut);
      sim.step(dt, gOut);
      if (sim.state.solved) break;
      const curTotalVel = Math.hypot(sim.state.speed, sim.state.latSpeed);
      const posDelta = Math.hypot(
        sim.state.localPos[0] - prevPos[0],
        sim.state.localPos[1] - prevPos[1],
        sim.state.localPos[2] - prevPos[2]
      );
      const isRamp =
        topology.edges[prevEdge].kind === 'ramp' ||
        topology.edges[sim.state.edgeId].kind === 'ramp';
      const isBevel =
        topology.edges[prevEdge].kind === 'bevel-wrap' ||
        topology.edges[sim.state.edgeId].kind === 'bevel-wrap';
      const velScale = isRamp ? 2.2 : isBevel ? 1.48 : 1.12;
      const allowed = Math.max(prevTotalVel, curTotalVel) * dt * velScale + 0.006;
      const excess = posDelta - allowed;
      if (excess > maxExcessJump) maxExcessJump = excess;
      expect(posDelta).toBeLessThanOrEqual(allowed);
      steps++;
    }
    expect(sim.state.solved).toBe(true);
    expect(maxExcessJump).toBeLessThanOrEqual(0);
  });

  it('eliminates phantom sideways ramp suction past py[4,3] when tilted purely along the corridor', () => {
    const fromNode = topology.nodes.find(
      (n) => n.face === 'py' && n.u === 4 && n.v === 4 && n.tier === 0
    )!;
    const midNode = topology.nodes.find(
      (n) => n.face === 'py' && n.u === 4 && n.v === 3 && n.tier === 0
    )!;
    const approachEdge = topology.edges.find(
      (e) => e.from === fromNode.id && e.to === midNode.id
    )!;
    expect(approachEdge).toBeDefined();

    const dx = midNode.localPos[0] - fromNode.localPos[0];
    const dy = midNode.localPos[1] - fromNode.localPos[1];
    const dz = midNode.localPos[2] - fromNode.localPos[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    const fn = midNode.localNormal;
    const gPureAlong: [number, number, number] = [
      (dx / len) * 2.4 - fn[0] * 9.5,
      (dy / len) * 2.4 - fn[1] * 9.5,
      (dz / len) * 2.4 - fn[2] * 9.5,
    ];

    const sim = new LabyrinthBeadSimulator(topology);
    (sim.state as { edgeId: number; t: number; speed: number }).edgeId = approachEdge.id;
    (sim.state as { t: number }).t = 0.1;
    (sim.state as { speed: number }).speed = 0.45;

    let enteredRamp = false;
    let rolledPastJunction = false;
    for (let i = 0; i < 60; i++) {
      sim.step(1 / 60, gPureAlong);
      const curEdge = topology.edges[sim.state.edgeId];
      if (curEdge.kind === 'ramp') enteredRamp = true;
      if (curEdge.from === midNode.id && curEdge.kind === 'surface') {
        rolledPastJunction = true;
      }
    }
    expect(enteredRamp).toBe(false);
    expect(rolledPastJunction).toBe(true);
  });

  it('glides freely on flat corridors without excessive low-speed drag before settling', () => {
    const startEdge = topology.edges.find(
      (e) =>
        topology.nodes[e.from].face === 'pz' &&
        topology.nodes[e.from].u === 1 &&
        topology.nodes[e.from].v === 5 &&
        topology.nodes[e.to].face === 'pz' &&
        topology.nodes[e.to].u === 2 &&
        topology.nodes[e.to].v === 5
    )!;
    const sim = new LabyrinthBeadSimulator(topology);
    (sim.state as { edgeId: number; t: number; speed: number }).edgeId = startEdge.id;
    (sim.state as { t: number }).t = 0;
    (sim.state as { speed: number }).speed = 0.42;

    let totalDist = 0;
    let prevPos: [number, number, number] = [...sim.state.localPos];
    for (let i = 0; i < 600; i++) {
      sim.step(1 / 60, [0, 0, -9.81]);
      totalDist += Math.hypot(
        sim.state.localPos[0] - prevPos[0],
        sim.state.localPos[1] - prevPos[1],
        sim.state.localPos[2] - prevPos[2]
      );
      prevPos = [...sim.state.localPos];
    }
    expect(totalDist).toBeGreaterThan(0.55);
    expect(sim.state.speed).toBeLessThan(0.02);
  });

  it('produces diminishing multi-bounce dead-end settle and true head-on axial T-wall rebound', () => {
    const dt = 1 / 60;
    const deadEndEdge = topology.edges.find(
      (e) => topology.adjacency[e.to].length === 1 && e.kind === 'surface'
    )!;
    expect(deadEndEdge).toBeDefined();

    const fromNode = topology.nodes[deadEndEdge.from];
    const toNode = topology.nodes[deadEndEdge.to];
    const dx = toNode.localPos[0] - fromNode.localPos[0];
    const dy = toNode.localPos[1] - fromNode.localPos[1];
    const dz = toNode.localPos[2] - fromNode.localPos[2];
    const dl = Math.hypot(dx, dy, dz) || 1;
    const fn = toNode.localNormal;
    const gDead: [number, number, number] = [
      (dx / dl) * 3.1 - fn[0] * 9.3,
      (dy / dl) * 3.1 - fn[1] * 9.3,
      (dz / dl) * 3.1 - fn[2] * 9.3,
    ];

    const simDead = new LabyrinthBeadSimulator(topology);
    (simDead.state as { edgeId: number; t: number; speed: number }).edgeId = deadEndEdge.id;
    (simDead.state as { t: number }).t = 0.15;
    (simDead.state as { speed: number }).speed = 0.55;

    let deadBounces = 0;
    let prevDeadEdge = deadEndEdge.id;
    const bouncePeaks: number[] = [];
    for (let i = 0; i < 150; i++) {
      simDead.step(dt, gDead);
      if (prevDeadEdge === deadEndEdge.id && simDead.state.edgeId !== deadEndEdge.id && simDead.state.speed > 0) {
        deadBounces++;
        bouncePeaks.push(simDead.state.speed);
      }
      prevDeadEdge = simDead.state.edgeId;
    }
    expect(deadBounces).toBeGreaterThanOrEqual(3);
    expect(bouncePeaks[0]).toBeGreaterThan(bouncePeaks[1]);
    expect(bouncePeaks[1]).toBeGreaterThan(bouncePeaks[2]);
    expect(simDead.state.speed).toBe(0);

    // Stem-into-bar flat T-wall head-on hit rebounds axially onto twin reverse edge
    const tNode = topology.nodes.find((n) => {
      if (n.tier !== 0 || n.id === topology.goalNodeId) return false;
      const out = topology.adjacency[n.id];
      if (out.length !== 3) return false;
      return out.some((aid, idxA) =>
        out.some((bid, idxB) => {
          if (idxA === idxB) return false;
          const pa = topology.edges[aid].samplesPos;
          const pb = topology.edges[bid].samplesPos;
          const dA = Math.hypot(pa[3] - pa[0], pa[4] - pa[1], pa[5] - pa[2]) || 1;
          const dB = Math.hypot(pb[3] - pb[0], pb[4] - pb[1], pb[5] - pb[2]) || 1;
          const dot =
            ((pa[3] - pa[0]) * (pb[3] - pb[0]) +
              (pa[4] - pa[1]) * (pb[4] - pb[1]) +
              (pa[5] - pa[2]) * (pb[5] - pb[2])) /
            (dA * dB);
          return dot < -0.95;
        })
      );
    })!;
    const tOut = topology.adjacency[tNode.id];
    const stemOutId = tOut.find((candId) => {
      let oppDots = 0;
      const pc = topology.edges[candId].samplesPos;
      const dc = Math.hypot(pc[3] - pc[0], pc[4] - pc[1], pc[5] - pc[2]) || 1;
      for (const otherId of tOut) {
        if (otherId === candId) continue;
        const po = topology.edges[otherId].samplesPos;
        const doLen = Math.hypot(po[3] - po[0], po[4] - po[1], po[5] - po[2]) || 1;
        const dot =
          ((pc[3] - pc[0]) * (po[3] - po[0]) +
            (pc[4] - pc[1]) * (po[4] - po[1]) +
            (pc[5] - pc[2]) * (po[5] - po[2])) /
          (dc * doLen);
        if (Math.abs(dot) < 0.2) oppDots++;
      }
      return oppDots === 2;
    })!;
    const stemOutEdge = topology.edges[stemOutId];
    const stemInEdge = topology.edges.find((e) => e.from === stemOutEdge.to && e.to === tNode.id)!;
    const pS = stemOutEdge.samplesPos;
    const dS = Math.hypot(pS[3] - pS[0], pS[4] - pS[1], pS[5] - pS[2]) || 1;
    const stemInDir: [number, number, number] = [
      -(pS[3] - pS[0]) / dS,
      -(pS[4] - pS[1]) / dS,
      -(pS[5] - pS[2]) / dS,
    ];
    const tHeadGrav: [number, number, number] = [
      stemInDir[0] * 3.0 - tNode.localNormal[0] * 9.3,
      stemInDir[1] * 3.0 - tNode.localNormal[1] * 9.3,
      stemInDir[2] * 3.0 - tNode.localNormal[2] * 9.3,
    ];

    const simT = new LabyrinthBeadSimulator(topology);
    (simT.state as { edgeId: number; t: number; speed: number }).edgeId = stemInEdge.id;
    (simT.state as { t: number }).t = 0.25;
    (simT.state as { speed: number }).speed = 0.55;

    let hitAxialRebound = false;
    for (let i = 0; i < 120; i++) {
      const prevEdgeId = simT.state.edgeId;
      simT.step(dt, tHeadGrav);
      if (prevEdgeId === stemInEdge.id && simT.state.edgeId === stemOutId && simT.state.speed > 0.05) {
        hitAxialRebound = true;
      }
    }
    expect(hitAxialRebound).toBe(true);
    expect(simT.state.speed).toBe(0);
  });

  it('excites and cleanly damps liquid-mercury dome slosh on abrupt stops', () => {
    const wetAtlas = new Uint8Array(WET_ATLAS_WIDTH * WET_ATLAS_HEIGHT * 4);
    const optics = createLabyrinthOptics({ uTime: float(0) } as unknown as SpecimenStage, wetAtlas);
    try {
      const sloshOut: [number, number, number] = [0, 0, 0];
      const dt = 1 / 60;
      optics.updateBeadUniforms([1.0, 0, 0], [0, 1, 0], 0, 0, dt);

      let maxAbsSlosh = 0;
      let minSloshX = Infinity;
      let magAt06s = Infinity;
      for (let i = 1; i <= 60; i++) {
        optics.updateBeadUniforms([0, 0, 0], [0, 1, 0], 0, 0, dt);
        optics.getBeadSloshLocal(sloshOut);
        const mag = Math.hypot(sloshOut[0], sloshOut[1], sloshOut[2]);
        if (mag > maxAbsSlosh) maxAbsSlosh = mag;
        if (sloshOut[0] < minSloshX) minSloshX = sloshOut[0];
        if (i === 36) magAt06s = mag;
        expect(Math.abs(sloshOut[1])).toBeLessThan(1e-6);
      }
      expect(maxAbsSlosh).toBeGreaterThan(0.02);
      expect(maxAbsSlosh).toBeLessThanOrEqual(0.22);
      expect(minSloshX).toBeLessThan(-0.005);
      expect(magAt06s).toBeLessThan(0.003);
      optics.getBeadSloshLocal(sloshOut);
      expect(Math.hypot(sloshOut[0], sloshOut[1], sloshOut[2])).toBe(0);
    } finally {
      optics.dispose();
    }
  });
});
