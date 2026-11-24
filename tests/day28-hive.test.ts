import { describe, expect, it } from 'vitest';
import {
  HiveShimmerSimulation,
  UNIT_STATE_FIRING,
  UNIT_STATE_REFRACTORY,
  UNIT_STATE_RESTING,
  buildHiveClusterGraph,
} from '../src/days/28-hive/shimmerGraph';
import {
  buildBeeAbdomenGeometry,
  buildBeeThoraxGeometry,
  buildBeeWingsGeometry,
  buildBranchGeometry,
  buildCurtainShellGeometry,
  computeSignedVolume,
} from '../src/days/28-hive/hiveGeometry';
import { createHiveMaterials } from '../src/days/28-hive/hiveMaterials';

describe('Day 28 - Hive: Apis dorsata Shimmering Defense-Wave Cluster', () => {
  const sharedGraph = buildHiveClusterGraph(202628);

  it('1. Graph Connectivity, Zero Degenerate Overlaps & Outward Surface Normals / Winding', () => {
    const g = sharedGraph;

    // 2,400..3,000 tightly overlapping living bee units
    expect(g.unitCount).toBeGreaterThanOrEqual(2400);
    expect(g.unitCount).toBeLessThanOrEqual(3000);

    // Every unit must have >= 4 valid local neighbors on the 3D teardrop surface
    expect(g.minDegree).toBeGreaterThanOrEqual(4);
    expect(g.maxDegree).toBeLessThanOrEqual(10);

    let minNeighborDist = Infinity;
    for (let i = 0; i < g.unitCount; i++) {
      const start = g.neighborOffsets[i];
      const end = g.neighborOffsets[i + 1];
      const deg = end - start;
      expect(deg).toBeGreaterThanOrEqual(4);

      let weightSum = 0;
      for (let e = start; e < end; e++) {
        const j = g.neighborIds[e];
        expect(j).toBeGreaterThanOrEqual(0);
        expect(j).toBeLessThan(g.unitCount);
        expect(j).not.toBe(i);
        const d = g.neighborDists[e];
        expect(d).toBeGreaterThan(0.02);
        if (d < minNeighborDist) minNeighborDist = d;
        weightSum += g.neighborWeights[e];
      }
      expect(weightSum).toBeCloseTo(1.0, 3);

      // Verify unit-length outward surface normal & exact right-handed orthonormal frame (T x U = N)
      const i3 = i * 3;
      const px = g.positions[i3];
      const py = g.positions[i3 + 1];
      const pz = g.positions[i3 + 2];
      const nx = g.normals[i3];
      const ny = g.normals[i3 + 1];
      const nz = g.normals[i3 + 2];
      const ux = g.upDirs[i3];
      const uy = g.upDirs[i3 + 1];
      const uz = g.upDirs[i3 + 2];
      const tx = g.tangents[i3];
      const ty = g.tangents[i3 + 1];
      const tz = g.tangents[i3 + 2];

      expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1.0, 4);
      expect(Math.hypot(ux, uy, uz)).toBeCloseTo(1.0, 4);
      expect(Math.hypot(tx, ty, tz)).toBeCloseTo(1.0, 4);

      // Outward relative to interior teardrop centroid (0, -0.52, 0)
      const outwardDot = nx * px + ny * (py + 0.52) + nz * pz;
      expect(outwardDot).toBeGreaterThan(0);

      // Right-handed frame determinant det(T, U, N) = (T x U) . N === +1
      const cx = ty * uz - tz * uy;
      const cy = tz * ux - tx * uz;
      const cz = tx * uy - ty * ux;
      expect(cx * nx + cy * ny + cz * nz).toBeCloseTo(1.0, 4);
    }

    // Zero overlapping degenerate unit positions
    expect(minNeighborDist).toBeGreaterThan(0.025);

    // Verify 100% single connected graph component wrapping front, flanks, back & bottom crest
    const visited = new Uint8Array(g.unitCount);
    const queue = new Int32Array(g.unitCount);
    let head = 0;
    let tail = 0;
    visited[0] = 1;
    queue[tail++] = 0;
    while (head < tail) {
      const cur = queue[head++];
      const start = g.neighborOffsets[cur];
      const end = g.neighborOffsets[cur + 1];
      for (let e = start; e < end; e++) {
        const nb = g.neighborIds[e];
        if (!visited[nb]) {
          visited[nb] = 1;
          queue[tail++] = nb;
        }
      }
    }
    expect(tail).toBe(g.unitCount);

    // Every closed solid geometry must have strictly positive signed volume (outward CCW winding)
    const branchGeo = buildBranchGeometry();
    const innerCurtainGeo = buildCurtainShellGeometry(0.945);
    const raycastHullGeo = buildCurtainShellGeometry(1.075);
    const thoraxGeo = buildBeeThoraxGeometry();
    const abdomenGeo = buildBeeAbdomenGeometry();
    const wingsGeo = buildBeeWingsGeometry();

    expect(computeSignedVolume(branchGeo)).toBeGreaterThan(0);
    expect(computeSignedVolume(innerCurtainGeo)).toBeGreaterThan(0);
    expect(computeSignedVolume(raycastHullGeo)).toBeGreaterThan(0);
    expect(computeSignedVolume(thoraxGeo)).toBeGreaterThan(0);
    expect(computeSignedVolume(abdomenGeo)).toBeGreaterThan(0);

    // Both left and right gossamer wings must have outward dorsal (+Z) normals
    const wingNorm = wingsGeo.getAttribute('normal');
    const wingUv = wingsGeo.getAttribute('aWingUv');
    let leftNzSum = 0;
    let leftCount = 0;
    let rightNzSum = 0;
    let rightCount = 0;
    for (let i = 0; i < wingNorm.count; i++) {
      const side = wingUv.getZ(i);
      const nz = wingNorm.getZ(i);
      if (side < 0) {
        leftNzSum += nz;
        leftCount++;
      } else {
        rightNzSum += nz;
        rightCount++;
      }
    }
    expect(leftNzSum / Math.max(1, leftCount)).toBeGreaterThan(0.5);
    expect(rightNzSum / Math.max(1, rightCount)).toBeGreaterThan(0.5);

    branchGeo.dispose();
    innerCurtainGeo.dispose();
    raycastHullGeo.dispose();
    thoraxGeo.dispose();
    abdomenGeo.dispose();
    wingsGeo.dispose();

    // TSL mineral NodeMaterials construct & dispose cleanly
    const mats = createHiveMaterials();
    expect(mats.branchMat.colorNode).toBeDefined();
    expect(mats.branchMat.normalNode).toBeDefined();
    expect(mats.innerCurtainMat.colorNode).toBeDefined();
    expect(mats.thoraxMat.colorNode).toBeDefined();
    expect(mats.abdomenMat.colorNode).toBeDefined();
    expect(mats.abdomenMat.normalNode).toBeDefined();
    expect(mats.wingMat.colorNode).toBeDefined();
    expect(mats.wingMat.opacityNode).toBeDefined();
    mats.dispose();
  });

  it('2. Wave Propagation Speed & Reach: single seed propagates a finite-speed wave across >70% of cluster within 1.5 s', () => {
    const sim = new HiveShimmerSimulation(sharedGraph);
    sim.clearVisited();

    // Ignite ONE single seed unit on the front cheek of the hanging bivouac
    const seedIdx = sim.findNearestUnit(0.0, -0.45, 0.42, true);
    expect(seedIdx).toBeGreaterThanOrEqual(0);
    expect(sim.igniteUnit(seedIdx, 1.0, false)).toBe(true);

    // Verify finite wave propagation (not instantaneous teleportation at t = 0.15 s)
    for (let step = 0; step < 9; step++) {
      sim.step(1 / 60);
    }
    const earlyReach = sim.getVisitedFraction();
    expect(earlyReach).toBeGreaterThan(0.002);
    expect(earlyReach).toBeLessThan(0.45);

    // Step to t = 1.5 s total (81 more 60 Hz frames) and track peak flip angle & sway
    let peakFlipSeen = 0;
    let peakSwaySeen = 0;
    for (let step = 0; step < 81; step++) {
      sim.step(1 / 60);
      const t = sim.getTelemetry();
      if (t.clusterSwayRad > peakSwaySeen) peakSwaySeen = t.clusterSwayRad;
      for (let i = 0; i < sim.graph.unitCount; i++) {
        if (sim.flipAngle[i] > peakFlipSeen) peakFlipSeen = sim.flipAngle[i];
      }
    }

    const reachAt1500ms = sim.getVisitedFraction();
    const telem = sim.getTelemetry();

    // Must cross > 70% of the entire 3D teardrop within 1.5 s at true biological speed (~0.4-1.3 m/s)
    expect(reachAt1500ms).toBeGreaterThan(0.70);
    expect(telem.peakWaveSpeed).toBeGreaterThan(0.35);
    expect(telem.peakWaveSpeed).toBeLessThan(1.50);
    // Articulated abdomen snap reaches ~55-75 deg (>= 0.95 rad)
    expect(peakFlipSeen).toBeGreaterThan(0.95);
    // Wave front imparts subtle physical pendulum recoil to the hanging bivouac
    expect(peakSwaySeen).toBeGreaterThan(0.0005);
  });

  it('3. Refractory Annihilation Invariant: single pulse & twin opposing collision settle cleanly to 0 firing, 0 refractory', () => {
    // A. Single pulse crosses full cluster, enters refractory cooldown, and settles cleanly to rest
    const sim = new HiveShimmerSimulation(sharedGraph);
    const seedIdx = sim.findNearestUnit(-0.52, -0.35, 0.34, true);
    expect(seedIdx).toBeGreaterThanOrEqual(0);
    sim.igniteUnit(seedIdx);

    let sawRefractory = false;
    let maxFiring = 0;
    for (let f = 0; f < 240; f++) {
      sim.step(1 / 60);
      if (sim.refractoryCount > 100) sawRefractory = true;
      if (sim.activeFiringCount > maxFiring) maxFiring = sim.activeFiringCount;
    }

    expect(sawRefractory).toBe(true);
    expect(maxFiring).toBeGreaterThan(150);

    // By t = 4.0 s, the wave has exited completely with zero runaway re-ignition
    expect(sim.activeFiringCount).toBe(0);
    expect(sim.refractoryCount).toBe(0);
    for (let i = 0; i < sim.graph.unitCount; i++) {
      expect(sim.state[i]).toBe(UNIT_STATE_RESTING);
      expect(Math.abs(sim.flipAngle[i])).toBeLessThan(0.02);
    }

    // B. Twin opposing-pole collision waves collide and annihilate cleanly down the midline
    const twinIgnited = sim.triggerTwinCollision();
    expect(twinIgnited).toBeGreaterThan(10);
    expect(sim.totalWavesTriggered).toBeGreaterThanOrEqual(2);

    let sawMidCollision = false;
    for (let f = 0; f < 210; f++) {
      sim.step(1 / 60);
      if (sim.activeFiringCount > 250 && sim.refractoryCount > 100) {
        sawMidCollision = true;
      }
    }
    expect(sawMidCollision).toBe(true);
    expect(sim.activeFiringCount).toBe(0);
    expect(sim.refractoryCount).toBe(0);

    // C. Broken wavefront spiral seed curls into sustained activity, and quietAll() calms it cleanly
    const spiralSeedCount = sim.triggerSpiralSeed();
    expect(spiralSeedCount).toBeGreaterThan(10);
    let hasFiring = false;
    let hasRefractory = false;
    for (let f = 0; f < 90; f++) {
      sim.step(1 / 60);
      for (let i = 0; i < sim.graph.unitCount; i++) {
        if (sim.state[i] === UNIT_STATE_FIRING) hasFiring = true;
        if (sim.state[i] === UNIT_STATE_REFRACTORY) hasRefractory = true;
      }
    }
    expect(hasFiring).toBe(true);
    expect(hasRefractory).toBe(true);

    sim.quietAll();
    expect(sim.activeFiringCount).toBe(0);
    expect(sim.refractoryCount).toBe(0);
  });
});
