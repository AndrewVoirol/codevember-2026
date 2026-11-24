import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  CORE_COUNT,
  buildCoreAssembly,
  evalCoreSdf,
  evalCoreSdfGrad,
  type CoreMaterialSet,
} from '../src/days/20-cocoon/coreAssemblies';
import { ShrinkFilmSimulation } from '../src/days/20-cocoon/shrinkPhysics';
import { DEFAULT_COCOON_CONFIG } from '../src/days/20-cocoon/types';

function makeDummyCoreMaterials(): { mats: CoreMaterialSet; dispose: () => void } {
  const mat = new THREE.MeshBasicMaterial();
  return {
    mats: {
      brass: mat,
      basalt: mat,
      calcite: mat,
      verdigris: mat,
      amber: mat,
    },
    dispose: () => mat.dispose(),
  };
}

function measureOutwardWinding(geo: THREE.BufferGeometry): {
  signedVol: number;
  outwardDot: number;
  triCount: number;
} {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const idx = geo.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  geo.computeBoundingBox();
  const center = new THREE.Vector3();
  geo.boundingBox!.getCenter(center);

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const n = new THREE.Vector3();
  const cen = new THREE.Vector3();
  const bc = new THREE.Vector3();

  let signedVol = 0;
  let outwardDot = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);

    bc.crossVectors(b, c);
    signedVol += a.dot(bc) / 6;

    ab.subVectors(b, a);
    ac.subVectors(c, a);
    n.crossVectors(ab, ac);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(center);
    outwardDot += n.dot(cen);
  }

  return { signedVol, outwardDot, triCount };
}

describe('Day 20 - Cocoon: Heat-Shrink Vacuum-Form Skin & Sculptural Core SDFs', () => {
  it('enforces outward triangle winding and exact analytical SDF enclosure across all 3 sculptural cores', () => {
    const { mats, dispose } = makeDummyCoreMaterials();
    const vWorld = new THREE.Vector3();
    const grad = new Float32Array(3);

    for (let coreIdx = 0; coreIdx < CORE_COUNT; coreIdx++) {
      const built = buildCoreAssembly(coreIdx, mats);
      built.group.updateMatrixWorld(true);

      let meshCount = 0;
      let worstSdf = -Infinity;

      built.group.traverse((obj) => {
        if (!(obj instanceof THREE.Mesh)) return;
        meshCount++;
        const geo = obj.geometry;
        const { signedVol, outwardDot, triCount } = measureOutwardWinding(geo);
        expect(triCount, `core ${coreIdx}/${obj.name} triCount`).toBeGreaterThan(0);
        expect(signedVol, `core ${coreIdx}/${obj.name} signedVol`).toBeGreaterThan(0);
        expect(outwardDot, `core ${coreIdx}/${obj.name} outwardDot`).toBeGreaterThan(0);

        // Verify analytical SDF strictly encloses every visual mesh vertex (zero skin Z-fighting/clipping)
        const pos = geo.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < pos.count; i++) {
          vWorld.fromBufferAttribute(pos, i).applyMatrix4(obj.matrixWorld);
          const d = evalCoreSdf(built.descriptor, vWorld.x, vWorld.y, vWorld.z);
          if (d > worstSdf) worstSdf = d;
        }
      });

      expect(meshCount).toBeGreaterThanOrEqual(4);
      expect(worstSdf, `core ${coreIdx} SDF visual enclosure`).toBeLessThanOrEqual(0.004);

      // Verify SDF gradient is unit-length and points outward outside the core
      const dOut = evalCoreSdfGrad(built.descriptor, 1.5, 0.2, 0.1, grad, 0);
      expect(dOut).toBeGreaterThan(0.1);
      expect(Math.hypot(grad[0], grad[1], grad[2])).toBeCloseTo(1.0, 4);
      expect(grad[0]).toBeGreaterThan(0.5);

      built.dispose();
    }

    dispose();
  });

  it('builds an outward-wound loose crinkled cocoon mesh with zero initial SDF penetration', () => {
    const { mats, dispose } = makeDummyCoreMaterials();
    const core0 = buildCoreAssembly(0, mats);
    const sim = new ShrinkFilmSimulation(core0.descriptor, DEFAULT_COCOON_CONFIG);

    const { signedVol, outwardDot, triCount } = measureOutwardWinding(sim.geometry);
    expect(sim.baseVertexCount).toBe(2562);
    expect(triCount).toBe(5120);
    expect(signedVol).toBeGreaterThan(10.0);
    expect(outwardDot).toBeGreaterThan(60.0);
    expect(sim.meanShrink).toBe(0);
    expect(sim.tearFraction).toBe(0);

    // Every individual triangle must face outward (0 inward-wound faces)
    let inwardTris = 0;
    for (let t = 0; t < sim.triCount; t++) {
      const i0 = sim.indices[t * 3] * 3;
      const i1 = sim.indices[t * 3 + 1] * 3;
      const i2 = sim.indices[t * 3 + 2] * 3;
      const ax = sim.pos[i0];
      const ay = sim.pos[i0 + 1];
      const az = sim.pos[i0 + 2];
      const e1x = sim.pos[i1] - ax;
      const e1y = sim.pos[i1 + 1] - ay;
      const e1z = sim.pos[i1 + 2] - az;
      const e2x = sim.pos[i2] - ax;
      const e2y = sim.pos[i2 + 1] - ay;
      const e2z = sim.pos[i2 + 2] - az;
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;
      if (nx * ax + ny * ay + nz * az <= 0) inwardTris++;
    }
    expect(inwardTris).toBe(0);

    // Every initial loose vertex must clear the core SDF
    for (let i = 0; i < sim.activeVertexCount; i++) {
      const i3 = i * 3;
      const d = evalCoreSdf(core0.descriptor, sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
      expect(d).toBeGreaterThanOrEqual(DEFAULT_COCOON_CONFIG.skinClearance - 1e-4);
    }

    sim.dispose();
    core0.dispose();
    dispose();
  });

  it('produces monotonic rest-length shrinkage under heat input and cools glow cleanly afterward', () => {
    const { mats, dispose } = makeDummyCoreMaterials();
    const core0 = buildCoreAssembly(0, mats);
    const sim = new ShrinkFilmSimulation(core0.descriptor, DEFAULT_COCOON_CONFIG);

    const prevShrink = new Float32Array(sim.baseVertexCount);
    let prevMeanShrink = 0;

    // Sweep heat beam on front face (+Z) for 1.0 second
    for (let f = 0; f < 60; f++) {
      sim.applyHeatBeam(0.1, 0.1, 1.25, 0, 0.2, 4.9, 0.86, 2.8, 1 / 60);
      sim.step(1 / 60);

      for (let i = 0; i < sim.activeVertexCount; i++) {
        expect(sim.shrink[i]).toBeGreaterThanOrEqual(prevShrink[i] - 1e-6);
        prevShrink[i] = sim.shrink[i];
      }
      expect(sim.meanShrink).toBeGreaterThanOrEqual(prevMeanShrink - 1e-6);
      prevMeanShrink = sim.meanShrink;
    }

    expect(sim.meanShrink).toBeGreaterThan(0.12);

    // Measure peak active heat, then remove beam and confirm infrared heat cools cleanly while shrink stays monotonic
    let peakHeat = 0;
    for (let i = 0; i < sim.activeVertexCount; i++) {
      if (sim.heat[i] > peakHeat) peakHeat = sim.heat[i];
    }
    expect(peakHeat).toBeGreaterThan(0.45);

    const shrinkAfterHeat = sim.meanShrink;
    for (let f = 0; f < 180; f++) {
      sim.step(1 / 60);
    }

    let cooledPeakHeat = 0;
    for (let i = 0; i < sim.activeVertexCount; i++) {
      if (sim.heat[i] > cooledPeakHeat) cooledPeakHeat = sim.heat[i];
    }
    expect(cooledPeakHeat).toBeLessThan(0.05);
    expect(sim.meanShrink).toBeGreaterThanOrEqual(shrinkAfterHeat);

    sim.dispose();
    core0.dispose();
    dispose();
  });

  it('enforces non-penetration against exact core SDF and builds realistic radial tension under full vacuum-shrinking across all 3 cores', () => {
    const { mats, dispose } = makeDummyCoreMaterials();

    const heatDirs: ReadonlyArray<readonly [number, number, number]> = [
      [0, 0, 1.05],
      [0, 0, -1.05],
      [1.05, 0, 0],
      [-1.05, 0, 0],
      [0, 1.15, 0],
      [0, -1.15, 0],
    ];

    for (let coreIdx = 0; coreIdx < CORE_COUNT; coreIdx++) {
      const built = buildCoreAssembly(coreIdx, mats);
      const sim = new ShrinkFilmSimulation(built.descriptor, DEFAULT_COCOON_CONFIG);

      // Heat all 6 sides so the skin vacuum-forms completely onto the sculptural core
      for (let f = 0; f < 150; f++) {
        const d = heatDirs[f % heatDirs.length];
        sim.applyHeatBeam(d[0], d[1], d[2], d[0] * 4.2, d[1] * 4.2, d[2] * 4.2, 1.15, 3.0, 1 / 60);
        sim.step(1 / 60);
      }

      // Settle for 30 frames
      for (let f = 0; f < 30; f++) {
        sim.step(1 / 60);
      }

      expect(sim.meanShrink, `core ${coreIdx} meanShrink`).toBeGreaterThan(0.55);
      expect(sim.maxTension, `core ${coreIdx} maxTension`).toBeGreaterThan(0.20);

      let minSdf = Infinity;
      let seatedCount = 0;
      let maxContact = 0;
      for (let i = 0; i < sim.activeVertexCount; i++) {
        const i3 = i * 3;
        const px = sim.pos[i3];
        const py = sim.pos[i3 + 1];
        const pz = sim.pos[i3 + 2];
        expect(Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)).toBe(true);

        const dist = evalCoreSdf(built.descriptor, px, py, pz);
        if (dist < minSdf) minSdf = dist;
        if (dist <= DEFAULT_COCOON_CONFIG.skinClearance + 0.025) seatedCount++;
        if (sim.contact[i] > maxContact) maxContact = sim.contact[i];
      }

      // Strict non-penetration invariant: zero vertices collapse inside core
      expect(minSdf, `core ${coreIdx} minSdf`).toBeGreaterThanOrEqual(
        DEFAULT_COCOON_CONFIG.skinClearance - 1e-4
      );
      // Continuous-surface invariant: zero triangle edge midpoints or centroids chord-slice core
      let minMidSdf = Infinity;
      let minCenSdf = Infinity;
      for (let t = 0; t < sim.triCount; t++) {
        const i0 = sim.indices[t * 3] * 3;
        const i1 = sim.indices[t * 3 + 1] * 3;
        const i2 = sim.indices[t * 3 + 2] * 3;
        const cx = (sim.pos[i0] + sim.pos[i1] + sim.pos[i2]) / 3;
        const cy = (sim.pos[i0 + 1] + sim.pos[i1 + 1] + sim.pos[i2 + 1]) / 3;
        const cz = (sim.pos[i0 + 2] + sim.pos[i1 + 2] + sim.pos[i2 + 2]) / 3;
        const dc = evalCoreSdf(built.descriptor, cx, cy, cz);
        if (dc < minCenSdf) minCenSdf = dc;

        const m01 = evalCoreSdf(
          built.descriptor,
          0.5 * (sim.pos[i0] + sim.pos[i1]),
          0.5 * (sim.pos[i0 + 1] + sim.pos[i1 + 1]),
          0.5 * (sim.pos[i0 + 2] + sim.pos[i1 + 2])
        );
        const m12 = evalCoreSdf(
          built.descriptor,
          0.5 * (sim.pos[i1] + sim.pos[i2]),
          0.5 * (sim.pos[i1 + 1] + sim.pos[i2 + 1]),
          0.5 * (sim.pos[i1 + 2] + sim.pos[i2 + 2])
        );
        const m20 = evalCoreSdf(
          built.descriptor,
          0.5 * (sim.pos[i2] + sim.pos[i0]),
          0.5 * (sim.pos[i2 + 1] + sim.pos[i0 + 1]),
          0.5 * (sim.pos[i2 + 2] + sim.pos[i0 + 2])
        );
        const mMin = Math.min(m01, m12, m20);
        if (mMin < minMidSdf) minMidSdf = mMin;
      }
      expect(minMidSdf, `core ${coreIdx} minEdgeMidpointSdf`).toBeGreaterThanOrEqual(0.0);
      expect(minCenSdf, `core ${coreIdx} minTriCentroidSdf`).toBeGreaterThanOrEqual(0.0);

      // Meaningful vacuum-formed seating onto core ridges and spokes
      expect(seatedCount, `core ${coreIdx} seatedCount`).toBeGreaterThan(120);
      expect(maxContact, `core ${coreIdx} maxContact`).toBeGreaterThan(0.70);

      sim.dispose();
      built.dispose();
    }

    dispose();
  });

  it('propagates seam rupture along taut zones, separates elastic flaps without penetrating core, and resets cleanly', () => {
    const { mats, dispose } = makeDummyCoreMaterials();
    const core0 = buildCoreAssembly(0, mats);
    const core1 = buildCoreAssembly(1, mats);
    const sim = new ShrinkFilmSimulation(core0.descriptor, DEFAULT_COCOON_CONFIG);

    // Unheated loose zone should ignore non-forced rupture click
    expect(sim.triggerRuptureAtPoint(0, 0, 1.45, false)).toBe(false);
    expect(sim.tearFraction).toBe(0);

    // Heat front face strongly so membrane vacuum-hugs and builds high tension
    for (let f = 0; f < 90; f++) {
      sim.applyHeatBeam(0.05, 0.08, 1.25, 0, 0.2, 4.9, 0.85, 2.8, 1 / 60);
      sim.step(1 / 60);
    }
    expect(sim.maxTension).toBeGreaterThan(0.15);

    // Click the taut front membrane to initiate a propagating seam tear
    const ruptured = sim.triggerRuptureAtPoint(0.05, 0.08, 1.0, false);
    expect(ruptured).toBe(true);
    const initialSplits = sim.splitCount;
    expect(initialSplits).toBeGreaterThanOrEqual(2);

    // Step 45 frames while crack tips unzip and elastic flaps peel back
    for (let f = 0; f < 45; f++) {
      sim.step(1 / 60);
    }

    // Verify tear propagated beyond initial puncture and opened a real geometric seam gap
    expect(sim.splitCount).toBeGreaterThan(initialSplits);
    expect(sim.activeVertexCount).toBe(sim.baseVertexCount + sim.splitCount);
    expect(sim.tearFraction).toBeGreaterThan(0.10);

    // Measure separation among torn-edge vertices (flaps must physically peel apart)
    let tornVertexCount = 0;
    let maxTearRadius = 0;
    for (let i = 0; i < sim.activeVertexCount; i++) {
      const i3 = i * 3;
      const d = evalCoreSdf(core0.descriptor, sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
      expect(d).toBeGreaterThanOrEqual(DEFAULT_COCOON_CONFIG.skinClearance - 1e-4);

      if (sim.tearEdgeData[i] > 0.5) {
        tornVertexCount++;
        const r = Math.hypot(sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
        if (r > maxTearRadius) maxTearRadius = r;
      }
    }
    expect(tornVertexCount).toBeGreaterThanOrEqual(sim.splitCount * 2);
    expect(maxTearRadius).toBeGreaterThan(0.6);

    // Verify triangle indices stay 100% valid within [0, activeVertexCount)
    for (let k = 0; k < sim.indices.length; k++) {
      const idx = sim.indices[k];
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(sim.activeVertexCount);
    }

    // Verify re-bagging reset restores base vertex count, zero tearFraction, and switches core cleanly
    sim.reset(core1.descriptor, true);
    expect(sim.getCore().index).toBe(1);
    expect(sim.activeVertexCount).toBe(sim.baseVertexCount);
    expect(sim.splitCount).toBe(0);
    expect(sim.tearFraction).toBe(0);
    expect(sim.meanShrink).toBe(0);
    expect(sim.maxTension).toBe(0);

    // Step after reset to confirm smooth drop-settle without NaNs or penetration
    for (let f = 0; f < 40; f++) {
      sim.step(1 / 60);
    }
    for (let i = 0; i < sim.activeVertexCount; i++) {
      const i3 = i * 3;
      const d = evalCoreSdf(core1.descriptor, sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
      expect(d).toBeGreaterThanOrEqual(DEFAULT_COCOON_CONFIG.skinClearance - 1e-4);
    }

    sim.dispose();
    core0.dispose();
    core1.dispose();
    dispose();
  });

  it('handles repeated multi-seam ruptures cleanly within real-time CPU step budget', () => {
    const { mats, dispose } = makeDummyCoreMaterials();
    const core2 = buildCoreAssembly(2, mats);
    const sim = new ShrinkFilmSimulation(core2.descriptor, DEFAULT_COCOON_CONFIG);

    // Heat front and sides aggressively
    for (let f = 0; f < 60; f++) {
      sim.applyHeatBeam(0.2, 0.1, 1.1, 0, 0.2, 5.0, 1.0, 3.0, 1 / 60);
      sim.step(1 / 60);
    }

    // Trigger 3 distinct seam tears across the taut shell (including forced slice)
    expect(sim.triggerRuptureAtPoint(0.2, 0.2, 1.0, false)).toBe(true);
    for (let f = 0; f < 20; f++) sim.step(1 / 60);

    expect(sim.triggerRuptureAtPoint(-0.4, -0.2, 0.95, true)).toBe(true);
    for (let f = 0; f < 20; f++) sim.step(1 / 60);

    expect(sim.triggerRuptureAtPoint(0.0, 0.65, 0.85, true)).toBe(true);

    // Measure average wall-clock step time over 45 active frames
    const t0 = performance.now();
    for (let f = 0; f < 45; f++) {
      sim.step(1 / 60);
    }
    const avgMs = (performance.now() - t0) / 45;

    expect(avgMs).toBeLessThan(6.5);
    expect(sim.activeVertexCount).toBeLessThanOrEqual(sim.maxVertices);
    expect(sim.tearFraction).toBeGreaterThan(0.25);

    // Validate finite positions, valid indices, and strict SDF non-penetration across all torn flaps
    for (let i = 0; i < sim.activeVertexCount; i++) {
      const i3 = i * 3;
      expect(Number.isFinite(sim.pos[i3])).toBe(true);
      expect(Number.isFinite(sim.pos[i3 + 1])).toBe(true);
      expect(Number.isFinite(sim.pos[i3 + 2])).toBe(true);
      const d = evalCoreSdf(core2.descriptor, sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
      expect(d).toBeGreaterThanOrEqual(DEFAULT_COCOON_CONFIG.skinClearance - 1e-4);
    }
    for (let k = 0; k < sim.indices.length; k++) {
      expect(sim.indices[k]).toBeLessThan(sim.activeVertexCount);
    }

    sim.dispose();
    core2.dispose();
    dispose();
  });

  it('enforces split-hub recessed amber seating, non-overlapping concentric rings, and bounded damped flap curl equilibrium', () => {
    const { mats, dispose } = makeDummyCoreMaterials();
    const vWorld = new THREE.Vector3();

    // 1. Verify Core 0 & Core 1 split equatorial basalt hub + recessed amber heart (strictly inside bezels)
    for (const ci of [0, 1]) {
      const built = buildCoreAssembly(ci, mats);
      built.group.updateMatrixWorld(true);

      let heartMaxRadial = 0;
      let bezelMinOuterR = Infinity;
      let upperBasaltMinY = Infinity;
      let lowerBasaltMaxY = -Infinity;

      built.group.traverse((obj) => {
        if (!(obj instanceof THREE.Mesh)) return;
        const pos = obj.geometry.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < pos.count; i++) {
          vWorld.fromBufferAttribute(pos, i).applyMatrix4(obj.matrixWorld);
          const rXZ = Math.hypot(vWorld.x, vWorld.z);
          if (obj.name === 'AmberCavityHeart' && rXZ > heartMaxRadial) {
            heartMaxRadial = rXZ;
          } else if (obj.name === 'HubBezelUpper' && rXZ < bezelMinOuterR) {
            bezelMinOuterR = rXZ;
          } else if (obj.name === 'BasaltCoreHubUpper' && vWorld.y < upperBasaltMinY) {
            upperBasaltMinY = vWorld.y;
          } else if (obj.name === 'BasaltCoreHubLower' && vWorld.y > lowerBasaltMaxY) {
            lowerBasaltMaxY = vWorld.y;
          }
        }
      });

      // Upper & lower basalt caps must leave a true open equatorial aperture around the recessed amber heart
      expect(upperBasaltMinY, `core ${ci} upperBasaltMinY`).toBeGreaterThan(0.08);
      expect(lowerBasaltMaxY, `core ${ci} lowerBasaltMaxY`).toBeLessThan(-0.08);
      expect(heartMaxRadial, `core ${ci} heart recessed inside brass bezel`).toBeLessThan(
        bezelMinOuterR + 0.005
      );

      // Verify caliper spokes / spires do not clip through outer brass rings
      const outerRings = built.descriptor.primitives.filter(
        (p) => p.kind === 'torus' && p.majorRadius > 0.6
      );
      built.group.traverse((obj) => {
        if (!(obj instanceof THREE.Mesh)) return;
        if (!obj.name.includes('CaliperAxis') && !obj.name.includes('Spire_')) return;
        const pos = obj.geometry.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < pos.count; i++) {
          vWorld.fromBufferAttribute(pos, i).applyMatrix4(obj.matrixWorld);
          for (const r of outerRings) {
            if (r.kind !== 'torus') continue;
            const lu = vWorld.x * r.ux + vWorld.y * r.uy + vWorld.z * r.uz;
            const lv = vWorld.x * r.vx + vWorld.y * r.vy + vWorld.z * r.vz;
            const ln = vWorld.x * r.nx + vWorld.y * r.ny + vWorld.z * r.nz;
            const qx = Math.hypot(lu, lv) - r.majorRadius;
            const d = Math.hypot(qx, ln) - r.minorRadius;
            expect(d, `core ${ci} ${obj.name} clears outer ring`).toBeGreaterThan(0.0);
          }
        }
      });

      built.dispose();
    }

    // 2. Verify Mode-I ruptured flaps curl open smoothly and settle into bounded equilibrium (no runaway spikes!)
    const core0 = buildCoreAssembly(0, mats);
    const sim = new ShrinkFilmSimulation(core0.descriptor, DEFAULT_COCOON_CONFIG);
    for (let f = 0; f < 90; f++) {
      sim.applyHeatBeam(0.05, 0.08, 1.25, 0, 0.2, 4.9, 0.85, 2.8, 1 / 60);
      sim.step(1 / 60);
    }
    expect(sim.triggerRuptureAtPoint(0.05, 0.08, 1.0, false)).toBe(true);

    // Run 150 frames (2.5 s) post-rupture so transient elastic recoil settles cleanly
    for (let f = 0; f < 150; f++) {
      sim.step(1 / 60);
    }

    let maxRadius = 0;
    let maxFlapGap = 0;
    let degenTris = 0;
    for (let i = 0; i < sim.activeVertexCount; i++) {
      const i3 = i * 3;
      const r = Math.hypot(sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
      if (r > maxRadius) maxRadius = r;
      if (sim.tearEdgeData[i] > 0.5) {
        const gap = evalCoreSdf(core0.descriptor, sim.pos[i3], sim.pos[i3 + 1], sim.pos[i3 + 2]);
        if (gap > maxFlapGap) maxFlapGap = gap;
      }
    }
    for (let t = 0; t < sim.triCount; t++) {
      const a = sim.indices[t * 3] * 3;
      const b = sim.indices[t * 3 + 1] * 3;
      const c = sim.indices[t * 3 + 2] * 3;
      const e1x = sim.pos[b] - sim.pos[a];
      const e1y = sim.pos[b + 1] - sim.pos[a + 1];
      const e1z = sim.pos[b + 2] - sim.pos[a + 2];
      const e2x = sim.pos[c] - sim.pos[a];
      const e2y = sim.pos[c + 1] - sim.pos[a + 1];
      const e2z = sim.pos[c + 2] - sim.pos[a + 2];
      const area =
        0.5 *
        Math.hypot(e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x);
      if (area < 1e-5) degenTris++;
    }

    // Bounded flap lift (< 0.45 off core SDF, total radius < 1.95) + zero collapsed/degenerate triangles
    expect(maxRadius).toBeLessThan(1.95);
    expect(maxFlapGap).toBeLessThan(0.45);
    expect(degenTris).toBe(0);

    sim.dispose();
    core0.dispose();
    dispose();
  });
});
