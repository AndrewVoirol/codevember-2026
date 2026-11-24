import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  BEAKER_INNER_FLOOR_Y,
  BEAKER_OUTER_RADIUS,
  BEAKER_TOP_Y,
  CUBE_UNIT_MASS,
  DISSOLVE_PIGMENT_CYCLE,
  MAX_ACTIVE_CUBES,
  MAX_SWIRL_OMEGA,
  WATER_RADIUS,
  BeakerDissolveSim,
  evalCrystalErosionMetrics,
  evalCrystalSurfacePoint,
} from '../src/days/30-dissolve/dissolveSim';
import {
  DeformableWaterCylinder,
  buildBaseCausticDiscGeometry,
  buildBorosilicateBeakerGeometry,
  buildGraduatedMarkingsGeometry,
  buildMineralCrumbGeometry,
  buildStirringRodGeometry,
  buildSubdividedUnitCubeGeometry,
  createGraduatedMarkingsTexture,
} from '../src/days/30-dissolve/beakerGeometry';

function computeSignedVolume(geom: THREE.BufferGeometry): number {
  const pos = geom.attributes.position as THREE.BufferAttribute;
  const idx = geom.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const cross = new THREE.Vector3();
  let vol = 0;
  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);
    cross.crossVectors(b, c);
    vol += a.dot(cross) / 6.0;
  }
  return vol;
}

function countInwardTrianglesConvex(geom: THREE.BufferGeometry, center: THREE.Vector3): number {
  const pos = geom.attributes.position as THREE.BufferAttribute;
  const idx = geom.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const n = new THREE.Vector3();
  const centroid = new THREE.Vector3();
  let inward = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);
    ab.subVectors(b, a);
    ac.subVectors(c, a);
    n.crossVectors(ab, ac);
    if (n.lengthSq() < 1e-12) continue;
    centroid.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(center);
    if (n.dot(centroid) < 0) inward++;
  }
  return inward;
}

function countOpenBoundaryEdges(geom: THREE.BufferGeometry): number {
  const idx = geom.index;
  if (!idx) throw new Error('Expected indexed geometry for manifold edge audit');
  const directedEdges = new Map<string, number>();
  const triCount = idx.count / 3;

  const addEdge = (u: number, v: number) => {
    const key = `${u}->${v}`;
    directedEdges.set(key, (directedEdges.get(key) ?? 0) + 1);
  };

  for (let t = 0; t < triCount; t++) {
    const i0 = idx.getX(t * 3);
    const i1 = idx.getX(t * 3 + 1);
    const i2 = idx.getX(t * 3 + 2);
    addEdge(i0, i1);
    addEdge(i1, i2);
    addEdge(i2, i0);
  }

  let unmatched = 0;
  for (const [edge, count] of directedEdges.entries()) {
    const [uStr, vStr] = edge.split('->');
    const twinCount = directedEdges.get(`${vStr}->${uStr}`) ?? 0;
    if (count !== 1 || twinCount !== 1) {
      unmatched++;
    }
  }
  return unmatched;
}

describe('Day 30 - Dissolve: Effervescent Mineral Cubes in a Borosilicate Glass Beaker', () => {
  it('1. Curvature-Weighted Corner Rounding: corners erode strictly faster than flat face centers as mass decreases', () => {
    const masses = [1.0, 0.85, 0.65, 0.45, 0.25, 0.08];
    let prevRatio = Infinity;
    let prevCornerFrac = -1;
    let prevFaceFrac = -1;

    const pFace0 = new THREE.Vector3();
    const pCorner0 = new THREE.Vector3();
    evalCrystalSurfacePoint(1, 0, 0, 1.0, 2.5, pFace0);
    evalCrystalSurfacePoint(1, 1, 1, 1.0, 2.5, pCorner0);
    const initFaceLen = pFace0.length();
    const initCornerLen = pCorner0.length();

    const pFace = new THREE.Vector3();
    const pCorner = new THREE.Vector3();

    for (const m of masses) {
      const metrics = evalCrystalErosionMetrics(m);
      expect(metrics.cornerToFaceRatio).toBeLessThanOrEqual(prevRatio);
      if (m < 1.0) {
        // Strict curvature weighting: both absolute AND fractional corner erosion outpace flat face centers
        expect(metrics.cornerErosion).toBeGreaterThan(metrics.faceErosion);
        expect(metrics.cornerErosionFraction).toBeGreaterThan(metrics.faceErosionFraction);
        expect(metrics.cornerToFaceRatio).toBeLessThan(prevRatio);
        expect(metrics.cornerErosionFraction).toBeGreaterThan(prevCornerFrac);
        expect(metrics.faceErosionFraction).toBeGreaterThan(prevFaceFrac);

        // Also verify actual 3D deformed crystal surface points (including pitting & rounding)
        evalCrystalSurfacePoint(1, 0, 0, m, 2.5, pFace);
        evalCrystalSurfacePoint(1, 1, 1, m, 2.5, pCorner);
        const surfFaceFrac = (initFaceLen - pFace.length()) / initFaceLen;
        const surfCornerFrac = (initCornerLen - pCorner.length()) / initCornerLen;
        expect(surfCornerFrac).toBeGreaterThan(surfFaceFrac);
      }

      prevRatio = metrics.cornerToFaceRatio;
      prevCornerFrac = metrics.cornerErosionFraction;
      prevFaceFrac = metrics.faceErosionFraction;
    }

    // Crisp start (~1.69 corner/face ratio) rounds cleanly down toward spherical pebble (< 1.08)
    expect(evalCrystalErosionMetrics(1.0).cornerToFaceRatio).toBeGreaterThan(1.64);
    expect(evalCrystalErosionMetrics(0.08).cornerToFaceRatio).toBeLessThan(1.08);
    expect(evalCrystalErosionMetrics(0.0).halfExtent).toBe(0);
  });

  it('2. Solute Mass Conservation: lost solid crystal mass transfers monotonically into dissolved bath concentration until mass === 0', () => {
    const sim = new BeakerDissolveSim(true);
    const starter = sim.cubes[0];
    expect(starter.active).toBe(true);
    expect(starter.mass).toBeCloseTo(CUBE_UNIT_MASS, 8);
    expect(sim.getBathConcentration()).toBe(0);

    let prevSolid = sim.getTotalSolidMass();
    let prevBath = sim.getBathConcentration();

    // Run until the starter crystal completely dissolves (mass === 0)
    const maxSteps = 2600; // ~43 seconds at 60 Hz (rest dissolution completes around ~25-30s)
    let completed = false;
    for (let step = 0; step < maxSteps; step++) {
      sim.step(1 / 60);
      const solid = sim.getTotalSolidMass();
      const bath = sim.getBathConcentration();
      const deltaSolid = solid - prevSolid;
      const deltaBath = bath - prevBath;

      // Exact per-step conservation: deltaSolid + deltaBath === 0
      expect(deltaSolid + deltaBath).toBeCloseTo(0, 9);
      expect(solid).toBeLessThanOrEqual(prevSolid + 1e-12);
      expect(bath).toBeGreaterThanOrEqual(prevBath - 1e-12);
      expect(solid + bath).toBeCloseTo(sim.totalInjectedMass, 9);

      prevSolid = solid;
      prevBath = bath;

      if (!starter.active && starter.mass === 0) {
        completed = true;
        break;
      }
    }

    expect(completed).toBe(true);
    expect(starter.mass).toBe(0);
    expect(sim.getTotalSolidMass()).toBe(0);
    expect(sim.getBathConcentration()).toBeCloseTo(CUBE_UNIT_MASS, 8);
    expect(sim.totalDissolvedCount).toBe(1);

    // Multi-cube pigment cycling & concurrent mass conservation (4 cubes)
    const droppedPigments = new Set<string>();
    for (let k = 0; k < MAX_ACTIVE_CUBES; k++) {
      const c = sim.dropCube();
      expect(c).not.toBeNull();
      if (c) droppedPigments.add(c.pigmentKey);
    }
    expect(droppedPigments.size).toBe(MAX_ACTIVE_CUBES);
    for (const p of droppedPigments) {
      expect(DISSOLVE_PIGMENT_CYCLE).toContain(p);
    }

    for (let s = 0; s < 300; s++) {
      sim.step(1 / 60);
      expect(sim.getTotalSolidMass() + sim.getBathConcentration()).toBeCloseTo(
        sim.totalInjectedMass,
        8
      );
    }

    // Flush beaker resets bath cleanly and drops one crisp starter cube
    sim.flushBeaker(true);
    expect(sim.getBathConcentration()).toBe(0);
    expect(sim.getTotalSolidMass()).toBeCloseTo(CUBE_UNIT_MASS, 8);
    expect(sim.getTelemetry().activeCubeCount).toBe(1);

    sim.dispose();
  });

  it('3. Swirl Vortex & Splash Coupling: stirring imparts bounded swirl velocity that decays smoothly with zero NaN over 600 steps', () => {
    const sim = new BeakerDissolveSim(true);

    // Drop a second cube from above the rim and verify meniscus splash excitation on crossing
    const dropped = sim.dropCube(0.18, -0.14);
    expect(dropped).not.toBeNull();
    expect(dropped!.submerged).toBe(false);

    let splashDetected = false;
    for (let i = 0; i < 120; i++) {
      sim.step(1 / 60);
      if (dropped!.submerged) {
        let waveEnergy = 0;
        for (let w = 0; w < sim.waveCurr.length; w++) {
          waveEnergy += Math.abs(sim.waveCurr[w]);
        }
        if (waveEnergy > 0.01) splashDetected = true;
        break;
      }
    }
    expect(dropped!.submerged).toBe(true);
    expect(splashDetected).toBe(true);

    // Circularly stir the glass rod inside the beaker to spin up a lively vortex
    sim.swirlAngularVel = 0;
    const stirR = 0.36;
    for (let i = 0; i < 90; i++) {
      const ang = i * 0.14;
      const rx = Math.cos(ang) * stirR;
      const rz = Math.sin(ang) * stirR;
      sim.setStirringRodTarget(true, rx, rz, 1 / 60);
      sim.step(1 / 60);
    }

    const peakOmega = Math.abs(sim.swirlAngularVel);
    expect(peakOmega).toBeGreaterThan(1.0);
    expect(peakOmega).toBeLessThanOrEqual(MAX_SWIRL_OMEGA);

    // Release stirring rod and step 600 frames (~10 s): verify smooth monotonic viscous decay & zero NaN
    sim.setStirringRodTarget(false, 0, 0, 1 / 60);
    let prevAbsOmega = Math.abs(sim.swirlAngularVel);
    let maxBubblesSeen = 0;

    for (let i = 0; i < 600; i++) {
      sim.step(1 / 60);
      const curAbsOmega = Math.abs(sim.swirlAngularVel);
      expect(Number.isFinite(curAbsOmega)).toBe(true);
      expect(curAbsOmega).toBeLessThanOrEqual(prevAbsOmega + 1e-9);
      expect(curAbsOmega).toBeLessThanOrEqual(MAX_SWIRL_OMEGA);
      prevAbsOmega = curAbsOmega;

      const tele = sim.getTelemetry();
      expect(Number.isFinite(tele.meanCubeMass)).toBe(true);
      expect(Number.isFinite(tele.bathConcentration)).toBe(true);
      expect(Number.isFinite(tele.swirlAngularVel)).toBe(true);
      if (tele.activeBubbles > maxBubblesSeen) maxBubblesSeen = tele.activeBubbles;

      for (const c of sim.cubes) {
        if (!c.active) continue;
        expect(Number.isFinite(c.pos.x)).toBe(true);
        expect(Number.isFinite(c.pos.y)).toBe(true);
        expect(Number.isFinite(c.pos.z)).toBe(true);
        expect(Math.hypot(c.pos.x, c.pos.z)).toBeLessThan(WATER_RADIUS + 0.02);
        expect(c.pos.y).toBeGreaterThanOrEqual(BEAKER_INNER_FLOOR_Y - 0.01);
        expect(c.pos.y).toBeLessThan(BEAKER_TOP_Y + 0.5);
      }
    }

    // Swirl velocity decayed smoothly under viscosity while effervescent bubbles nucleated actively
    expect(Math.abs(sim.swirlAngularVel)).toBeLessThan(peakOmega * 0.25);
    expect(maxBubblesSeen).toBeGreaterThan(150);

    sim.dispose();
  });

  it('4. Outward Winding & Watertight Manifold Integrity: beaker lathe, water cylinder, crystal cube, rod & markings', () => {
    // A. Thick-walled Borosilicate Beaker Lathe (with flared pouring spout lip)
    const beakerGeom = buildBorosilicateBeakerGeometry(96);
    const beakerVol = computeSignedVolume(beakerGeom);
    // Solid thick-walled glass shell has strictly positive physical volume (> 0.35 cubic units)
    expect(beakerVol).toBeGreaterThan(0.35);

    // Audit every exterior cylindrical wall & underside base triangle: 0 inward-facing exterior triangles
    const bPos = beakerGeom.attributes.position as THREE.BufferAttribute;
    const bMeta = beakerGeom.attributes.aGlassMeta as THREE.BufferAttribute;
    const bIdx = beakerGeom.index!;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();
    const triN = new THREE.Vector3();
    const outDir = new THREE.Vector3();

    let exteriorWallTrisChecked = 0;
    let exteriorInwardTris = 0;
    for (let t = 0; t < bIdx.count / 3; t++) {
      const i0 = bIdx.getX(t * 3);
      const i1 = bIdx.getX(t * 3 + 1);
      const i2 = bIdx.getX(t * 3 + 2);
      const isExt =
        bMeta.getX(i0) > 0.5 && bMeta.getX(i1) > 0.5 && bMeta.getX(i2) > 0.5;
      if (!isExt) continue;

      a.fromBufferAttribute(bPos, i0);
      b.fromBufferAttribute(bPos, i1);
      c.fromBufferAttribute(bPos, i2);
      ab.subVectors(b, a);
      ac.subVectors(c, a);
      triN.crossVectors(ab, ac);
      if (triN.lengthSq() < 1e-12) continue;

      const cy = (a.y + b.y + c.y) / 3;
      const cx = (a.x + b.x + c.x) / 3;
      const cz = (a.z + b.z + c.z) / 3;
      const rXZ = Math.hypot(cx, cz);

      // Exterior cylindrical side wall & heel/base checks
      if (cy < BEAKER_TOP_Y - 0.14 && cy > BEAKER_INNER_FLOOR_Y && rXZ > BEAKER_OUTER_RADIUS * 0.92) {
        outDir.set(cx, 0, cz).normalize();
        exteriorWallTrisChecked++;
        if (triN.dot(outDir) <= 0) exteriorInwardTris++;
      } else if (cy < BEAKER_INNER_FLOOR_Y - 0.08 && rXZ < BEAKER_OUTER_RADIUS * 0.65) {
        outDir.set(0, -1, 0);
        exteriorWallTrisChecked++;
        if (triN.dot(outDir) <= 0) exteriorInwardTris++;
      }
    }
    expect(exteriorWallTrisChecked).toBeGreaterThan(2000);
    expect(exteriorInwardTris).toBe(0);
    beakerGeom.dispose();

    // B. Watertight Deformable Cylindrical Water Volume (shared perimeter waterline vertices!)
    const dummyMat = new THREE.MeshBasicMaterial();
    const waterCyl = new DeformableWaterCylinder(dummyMat);
    const sim = new BeakerDissolveSim(true);
    sim.applyStirImpulse(0.35, 0, 0, -2.5, 0.25);
    sim.injectSplash(0.15, -0.1, -0.035, 0.22);
    sim.step(1 / 60);
    waterCyl.updateFromSim(sim);

    const waterVol = computeSignedVolume(waterCyl.geometry);
    const expectedCylVol = Math.PI * WATER_RADIUS * WATER_RADIUS * 1.26;
    expect(waterVol).toBeGreaterThan(expectedCylVol * 0.85);
    expect(waterVol).toBeLessThan(expectedCylVol * 1.15);

    const waterBox = new THREE.Box3().setFromBufferAttribute(
      waterCyl.geometry.attributes.position as THREE.BufferAttribute
    );
    const waterCenter = waterBox.getCenter(new THREE.Vector3());
    expect(countInwardTrianglesConvex(waterCyl.geometry, waterCenter)).toBe(0);
    // Exact topological manifold guarantee: ZERO open boundary edges (top meniscus & side wall share perimeter indices!)
    expect(countOpenBoundaryEdges(waterCyl.geometry)).toBe(0);
    waterCyl.dispose();
    sim.dispose();

    // C. Welded High-Subdivision Unit Crystal Cube
    const cubeGeom = buildSubdividedUnitCubeGeometry(18);
    expect(computeSignedVolume(cubeGeom)).toBeCloseTo(8.0, 4);
    expect(countInwardTrianglesConvex(cubeGeom, new THREE.Vector3(0, 0, 0))).toBe(0);
    expect(countOpenBoundaryEdges(cubeGeom)).toBe(0);
    cubeGeom.dispose();

    // D. Borosilicate Stirring Rod
    const rodGeom = buildStirringRodGeometry(0.024, 1.92);
    expect(computeSignedVolume(rodGeom)).toBeGreaterThan(0.0025);
    expect(countInwardTrianglesConvex(rodGeom, new THREE.Vector3(0, 0.96, 0))).toBe(0);
    rodGeom.dispose();

    // E. Graduated Etched Volumetric Markings Strip & Texture + Base Caustic Disc
    const markGeom = buildGraduatedMarkingsGeometry();
    expect(countInwardTrianglesConvex(markGeom, new THREE.Vector3(0, 0.1, 0))).toBe(0);
    markGeom.dispose();

    const markTex = createGraduatedMarkingsTexture();
    const texBytes = markTex.image.data as Uint8Array;
    let etchedPixels = 0;
    for (let i = 3; i < texBytes.length; i += 4) {
      if (texBytes[i] > 100) etchedPixels++;
    }
    expect(etchedPixels).toBeGreaterThan(1500);
    markTex.dispose();

    const causticGeom = buildBaseCausticDiscGeometry();
    const cNorm = causticGeom.attributes.normal as THREE.BufferAttribute;
    expect(cNorm.getY(0)).toBeGreaterThan(0.99);
    causticGeom.dispose();

    // F. Faceted Cleaved Mineral Crumb Geometry (Outward Winding & Positive Volume)
    const crumbGeom = buildMineralCrumbGeometry();
    expect(computeSignedVolume(crumbGeom)).toBeGreaterThan(1.2);
    expect(countInwardTrianglesConvex(crumbGeom, new THREE.Vector3(0, 0, 0))).toBe(0);
    crumbGeom.dispose();

    dummyMat.dispose();
  });

  it('5. Detaching Mineral Crumbs: flakes spawn off active eroding cubes, stay inside the beaker, and dissolve monotonically', () => {
    const sim = new BeakerDissolveSim(true);
    expect(sim.activeCrumbCount).toBeGreaterThan(20);
    expect(sim.getTelemetry().activeCrumbs).toBe(sim.activeCrumbCount);

    let maxCrumbs = sim.activeCrumbCount;
    for (let s = 0; s < 180; s++) {
      sim.step(1 / 60);
      if (sim.activeCrumbCount > maxCrumbs) maxCrumbs = sim.activeCrumbCount;
      for (let i = 0; i < sim.crumbAlive.length; i++) {
        if (sim.crumbAlive[i] === 0) continue;
        expect(Number.isFinite(sim.crumbPosX[i])).toBe(true);
        expect(Number.isFinite(sim.crumbPosY[i])).toBe(true);
        expect(Number.isFinite(sim.crumbPosZ[i])).toBe(true);
        expect(Math.hypot(sim.crumbPosX[i], sim.crumbPosZ[i])).toBeLessThan(WATER_RADIUS);
        expect(sim.crumbPosY[i]).toBeGreaterThanOrEqual(BEAKER_INNER_FLOOR_Y - 0.005);
        expect(sim.crumbScale[i]).toBeGreaterThan(0);
      }
    }
    expect(maxCrumbs).toBeGreaterThan(50);

    // Flush clears crumbs & re-seeds starter crumbs cleanly
    sim.flushBeaker(false);
    expect(sim.activeCrumbCount).toBe(0);
    sim.dispose();
  });
});
