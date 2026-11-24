import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  FLUID_CELLS,
  FLUID_H,
  FLUID_W,
  FluidVeilSolver,
  encodeHalfFloat,
} from '../src/days/25-veil/fluidVeil';
import {
  APERTURE_MODES,
  ApertureModeController,
  VEIL_Y_BOT,
  VEIL_Y_TOP,
  evalVeilSurface,
  intersectVeilRay,
  type VeilRayHit,
} from '../src/days/25-veil/veilSurface';
import {
  buildGrazingCrossSectionGeometry,
  buildSlitLampAssembly,
  buildVolumetricVeilGeometry,
  createDiamondKnurledDrumGeometry,
} from '../src/days/25-veil/slitLampGeometry';
import { createVeilMaterials } from '../src/days/25-veil/veilMaterials';
import { VEIL_MOTE_COUNT, createVeilMotesSystem } from '../src/days/25-veil/motesCompute';

describe('Day 25 - Veil (Studio Slit-Lamp Volumetric Light-and-Haze Curtain)', () => {
  it('injects pointer momentum and generates counter-rotating Kelvin-Helmholtz vorticity pairs', () => {
    const solver = new FluidVeilSolver();
    // Settle baseline briefly
    for (let i = 0; i < 15; i++) solver.step(1 / 60, false);
    const baseVort = solver.maxVorticity;

    // Sweep rapidly across the middle of the curtain from left to right
    solver.injectPointerSweep(0.28, 0.48, 0.72, 0.50, 2.8, 0.12, true);
    const tel = solver.step(1 / 60, false);

    expect(tel.maxVorticity).toBeGreaterThan(Math.max(8.0, baseVort * 2.5));

    // Verify genuine opposing counter-rotating vortex flanks (+curl on one flank, -curl on the other)
    let maxPosCurl = 0;
    let minNegCurl = 0;
    for (let i = 0; i < FLUID_CELLS; i++) {
      const c = solver.curl[i];
      if (c > maxPosCurl) maxPosCurl = c;
      if (c < minNegCurl) minNegCurl = c;
    }
    expect(maxPosCurl).toBeGreaterThan(6.0);
    expect(minNegCurl).toBeLessThan(-6.0);

    // Verify incompressible pressure projection keeps interior divergence small
    let sumAbsDiv = 0;
    let count = 0;
    for (let iy = 2; iy < FLUID_H - 2; iy++) {
      for (let ix = 2; ix < FLUID_W - 2; ix++) {
        const idx = iy * FLUID_W + ix;
        const div =
          0.5 *
          (solver.u[idx + 1] -
            solver.u[idx - 1] +
            (solver.v[idx + FLUID_W] - solver.v[idx - FLUID_W]));
        sumAbsDiv += Math.abs(div);
        count++;
      }
    }
    expect(sumAbsDiv / count).toBeLessThan(0.025);

    solver.dispose();
  });

  it('maintains strict mass/density bounds [0, 1] with zero NaN or blowup across 300 turbulent steps', () => {
    const solver = new FluidVeilSolver();

    for (let stepIdx = 0; stepIdx < 300; stepIdx++) {
      // Alternate open-curtain and periodic-cone boundary regimes under repeated strong slicing
      solver.periodicX = stepIdx >= 150;
      if (stepIdx % 12 === 0) {
        const u0 = (stepIdx * 0.17) % 1;
        const u1 = (u0 + 0.28) % 1;
        const v0 = 0.2 + ((stepIdx * 0.13) % 0.6);
        const v1 = Math.min(0.88, v0 + 0.14);
        solver.injectPointerSweep(u0, v0, u1, v1, 6.5, -4.2, true);
      }
      if (stepIdx === 100 || stepIdx === 200) {
        solver.injectAperturePulse();
      }

      const tel = solver.step(1 / 60, stepIdx % 50 === 0);
      expect(Number.isFinite(tel.maxVorticity)).toBe(true);
      expect(Number.isFinite(tel.meanDensity)).toBe(true);
      expect(Number.isFinite(tel.tearEnergy)).toBe(true);
      expect(tel.meanDensity).toBeGreaterThanOrEqual(0.05);
      expect(tel.meanDensity).toBeLessThanOrEqual(0.95);
    }

    // Audit every single cell in density, u, v, curl and packed RGBA16F texture
    solver.packToTexture();
    for (let i = 0; i < FLUID_CELLS; i++) {
      const d = solver.density[i];
      const u = solver.u[i];
      const v = solver.v[i];
      const c = solver.curl[i];
      expect(Number.isFinite(d)).toBe(true);
      expect(Number.isFinite(u)).toBe(true);
      expect(Number.isFinite(v)).toBe(true);
      expect(Number.isFinite(c)).toBe(true);
      expect(d).toBeGreaterThanOrEqual(0);
      expect(d).toBeLessThanOrEqual(1);
      expect(Math.hypot(u, v)).toBeLessThanOrEqual(5.25);
    }

    // Half-float encoder never outputs NaN/Inf exponent pattern (0x7c00..0x7fff)
    const testVals = [0, 0.5, 1.0, -3.25, 42.5, 1e9, -1e9, Number.NaN];
    for (const val of testVals) {
      const hf = encodeHalfFloat(val);
      expect((hf & 0x7c00) === 0x7c00).toBe(false);
    }

    solver.dispose();
  });

  it('carves a dark turbulent channel on drag and heals cleanly back toward laminar stillness', () => {
    const solver = new FluidVeilSolver();
    // Let initial boot curl settle
    for (let i = 0; i < 60; i++) solver.step(1 / 60, false);

    const cx = Math.floor(FLUID_W * 0.5);
    const cy = Math.floor(FLUID_H * 0.5);
    const centerIdx = cy * FLUID_W + cx;
    const preCutDensity = solver.density[centerIdx];

    // Repeated heavy hold-drag strokes carving across the exact center
    for (let k = 0; k < 5; k++) {
      solver.injectPointerSweep(0.32, 0.50, 0.68, 0.50, 3.2, 0.05, true);
      solver.step(1 / 60, false);
    }

    const postCutDensity = solver.density[centerIdx];
    const peakVort = solver.maxVorticity;
    const peakTear = solver.tearEnergy;

    expect(postCutDensity).toBeLessThan(preCutDensity * 0.55);
    expect(peakVort).toBeGreaterThan(10.0);
    expect(peakTear).toBeGreaterThan(0.08);

    // Step undisturbed for 420 frames (~7 seconds) -> must heal back toward laminar baseline
    for (let f = 0; f < 420; f++) {
      solver.step(1 / 60, false);
    }

    const healedDensity = solver.density[centerIdx];
    const targetDensity = solver.evalBaselineDensity(
      (cx + 0.5) / FLUID_W,
      (cy + 0.5) / FLUID_H,
      solver.simTime
    );

    expect(Math.abs(healedDensity - targetDensity)).toBeLessThan(0.14);
    expect(solver.maxVorticity).toBeLessThan(peakVort * 0.3);
    expect(solver.tearEnergy).toBeLessThan(peakTear * 0.3);

    solver.dispose();
  });

  it('cycles seamlessly across Planar, Curved S-Veil, and Hollow Cone modes with accurate ray-curtain hits', () => {
    const ctrl = new ApertureModeController('planar');
    expect(ctrl.mode).toBe('planar');
    expect(APERTURE_MODES).toEqual(['planar', 'curved-s', 'cone']);

    const posA = new THREE.Vector3();
    const posB = new THREE.Vector3();
    const norm = new THREE.Vector3();
    const hit: VeilRayHit = { u: 0, v: 0, distance: 0, point: new THREE.Vector3() };

    // Test all 3 modes in sequence and verify wrap-around back to planar
    const expectedModes = ['planar', 'curved-s', 'cone', 'planar'] as const;
    for (let m = 0; m < expectedModes.length; m++) {
      expect(ctrl.mode).toBe(expectedModes[m]);

      // Advance transition smoothly and check convex weight conservation at every substep
      for (let f = 0; f < 120; f++) {
        ctrl.step(1 / 60);
        const wSum = ctrl.weights.planar + ctrl.weights.curvedS + ctrl.weights.cone;
        expect(wSum).toBeCloseTo(1.0, 5);
        expect(ctrl.weights.planar).toBeGreaterThanOrEqual(0);
        expect(ctrl.weights.curvedS).toBeGreaterThanOrEqual(0);
        expect(ctrl.weights.cone).toBeGreaterThanOrEqual(0);
      }

      // Ray aimed straight at center (0, -0.16, 0) from front camera (0, -0.16, 5)
      const ray = new THREE.Ray(new THREE.Vector3(0, -0.16, 5), new THREE.Vector3(0, 0, -1));
      const didHit = intersectVeilRay(ray, ctrl.weights, hit);
      expect(didHit).toBe(true);
      expect(hit.v).toBeGreaterThan(0.35);
      expect(hit.v).toBeLessThan(0.65);

      // Verify hit lies accurately on the parametric surface
      evalVeilSurface(hit.u, hit.v, ctrl.weights, posA, norm);
      expect(posA.distanceTo(hit.point)).toBeLessThan(0.025);
      expect(norm.length()).toBeCloseTo(1.0, 4);

      // In cone mode, verify u=0 and u=1 close seamlessly at the rear seam (-Z)
      if (ctrl.mode === 'cone') {
        evalVeilSurface(0, 0.5, ctrl.weights, posA);
        evalVeilSurface(1, 0.5, ctrl.weights, posB);
        expect(posA.distanceTo(posB)).toBeLessThan(1e-5);
        // Front hit should hit the +Z wall (closer to z = +5 camera), not the back wall
        expect(hit.point.z).toBeGreaterThan(0.35);
      }

      // Top slit edge always connects flush at VEIL_Y_TOP and bottom at VEIL_Y_BOT
      evalVeilSurface(0.5, 0, ctrl.weights, posA);
      evalVeilSurface(0.5, 1, ctrl.weights, posB);
      expect(posA.y).toBeCloseTo(VEIL_Y_TOP, 5);
      expect(posB.y).toBeCloseTo(VEIL_Y_BOT, 5);

      if (m < expectedModes.length - 1) {
        const prevDrum = ctrl.drumAngleRad;
        ctrl.cycleMode();
        for (let f = 0; f < 60; f++) ctrl.step(1 / 60);
        expect(ctrl.drumAngleRad).toBeGreaterThan(prevDrum + 1.0);
      }
    }
  });

  it('enforces strict outward winding (signedVol > 0 and outwardDot > 0) and zero-overhang clearance across all slit-lamp geometry', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const assembly = buildSlitLampAssembly({
      forgedIron: dummyMat,
      satinBrass: dummyMat,
      vernierBrass: dummyMat,
      knurledBrass: dummyMat,
      bluedSteel: dummyMat,
      honedSlate: dummyMat,
      ceramicInsulator: dummyMat,
      opticalGlass: dummyMat,
      opticalBaffle: dummyMat,
      linearFilament: dummyMat,
      coneFilament: dummyMat,
    });

    const assertOutward = (label: string, geo: THREE.BufferGeometry) => {
      const pos = geo.attributes.position as THREE.BufferAttribute;
      const idx = geo.index;
      const triCount = idx ? idx.count / 3 : pos.count / 3;
      expect(triCount, `${label} triCount`).toBeGreaterThan(0);

      let signedVol = 0;
      let outwardDot = 0;
      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      const c = new THREE.Vector3();
      const ab = new THREE.Vector3();
      const ac = new THREE.Vector3();
      const n = new THREE.Vector3();
      const cen = new THREE.Vector3();
      geo.computeBoundingBox();
      const boxCenter = new THREE.Vector3();
      geo.boundingBox!.getCenter(boxCenter);

      for (let t = 0; t < triCount; t++) {
        const i0 = idx ? idx.getX(t * 3) : t * 3;
        const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
        const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
        a.fromBufferAttribute(pos, i0);
        b.fromBufferAttribute(pos, i1);
        c.fromBufferAttribute(pos, i2);
        signedVol += a.dot(b.clone().cross(c)) / 6;
        ab.subVectors(b, a);
        ac.subVectors(c, a);
        n.crossVectors(ab, ac);
        cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(boxCenter);
        outwardDot += n.dot(cen);
      }

      expect(signedVol, `${label} signedVol`).toBeGreaterThan(0);
      expect(outwardDot, `${label} outwardDot`).toBeGreaterThan(0);
    };

    const checkedGeos = new Set<THREE.BufferGeometry>();
    let solidMeshCount = 0;
    const byName = new Map<string, THREE.Mesh>();
    assembly.group.traverse((obj: any) => {
      if (!obj.isMesh) return;
      solidMeshCount++;
      byName.set(obj.name, obj);
      if (!checkedGeos.has(obj.geometry)) {
        checkedGeos.add(obj.geometry);
        assertOutward(obj.name || obj.geometry.type, obj.geometry);
      }
    });

    expect(solidMeshCount).toBeGreaterThanOrEqual(80);
    expect(assembly.hitMeshes.length).toBeGreaterThanOrEqual(70);

    // Verify mechanical articulation moves jaws & annular iris cleanly without rail/end-cap clipping!
    const boxOf = (name: string) => {
      const m = byName.get(name);
      expect(m, `missing mesh ${name}`).toBeDefined();
      return new THREE.Box3().setFromObject(m!);
    };

    // 1. Planar mode: retracted annular iris stays 100% inside CentralCondenserChest Z footprint!
    assembly.updateArticulation({ planar: 1, curvedS: 0, cone: 0 }, 0);
    assembly.group.updateMatrixWorld(true);
    const jawPlanarZ = assembly.jaws.frontLeft.position.z;
    const irisPlanarY = assembly.coneTurretGroup.position.y;
    const chestBox = boxOf('CentralCondenserChest');
    const turretBox = boxOf('CentralCondenserTurret');
    const irisPlanarBox = boxOf('AnnularIrisOuterBezel');
    // Turret seats flush within chest Z bounds (zero pancake air overhang)
    expect(turretBox.max.z).toBeLessThanOrEqual(chestBox.max.z + 0.002);
    // Retracted annular iris fits cleanly inside central condenser chest Z bounds
    expect(irisPlanarBox.max.z).toBeLessThanOrEqual(chestBox.max.z - 0.01);

    // 2. Curved-S mode: jaws articulate cleanly without punching through LowerBaffleRail (`Z = +-0.112`)
    assembly.updateArticulation({ planar: 0, curvedS: 1, cone: 0 }, 1.2);
    assembly.group.updateMatrixWorld(true);
    expect(assembly.jaws.frontLeft.position.z).not.toBeCloseTo(jawPlanarZ, 3);
    expect(Math.abs(assembly.jaws.frontLeft.rotation.y)).toBeGreaterThan(0.035);
    const jawFRBoxS = boxOf('BrassSlitJaw_FR');
    const railFrontBox = boxOf('LowerBaffleRail_Front_R');
    expect(jawFRBoxS.max.z).toBeLessThan(railFrontBox.min.z + 0.002);

    // 3. Cone mode: jaws stay strictly inside end-caps (`|X| <= 0.935`) and clear deployed annular iris!
    assembly.updateArticulation({ planar: 0, curvedS: 0, cone: 1 }, 2.4);
    assembly.group.updateMatrixWorld(true);
    expect(assembly.coneTurretGroup.position.y).toBeLessThan(irisPlanarY - 0.05);
    const jawFRBoxCone = boxOf('BrassSlitJaw_FR');
    const irisConeBox = boxOf('AnnularIrisOuterBezel');
    const endCapRBox = boxOf('BrassEndCap_R');
    expect(jawFRBoxCone.max.x).toBeLessThanOrEqual(endCapRBox.min.x + 0.002);
    expect(jawFRBoxCone.min.x).toBeGreaterThan(irisConeBox.max.x + 0.004);

    // Verify diamond-knurled drum vertex valley attribute
    const knurlGeo = createDiamondKnurledDrumGeometry();
    assertOutward('DiamondKnurledDrum', knurlGeo);
    const kvAttr = knurlGeo.attributes.aKnurlValley as THREE.BufferAttribute;
    expect(kvAttr).toBeDefined();
    let minKv = Infinity;
    let maxKv = -Infinity;
    for (let i = 0; i < kvAttr.count; i++) {
      const v = kvAttr.getX(i);
      if (v < minKv) minKv = v;
      if (v > maxKv) maxKv = v;
    }
    expect(minKv).toBeGreaterThanOrEqual(0);
    expect(maxKv).toBeLessThanOrEqual(1);
    expect(maxKv - minKv).toBeGreaterThan(0.5);

    // Verify volumetric veil & grazing cross-section geometries carry exact `aStratum` attributes
    const veilGeo = buildVolumetricVeilGeometry(48, 36, 5);
    expect(veilGeo.attributes.aStratum).toBeDefined();
    expect(veilGeo.index).toBeDefined();

    const grazingGeo = buildGrazingCrossSectionGeometry(32);
    expect(grazingGeo.attributes.aStratum).toBeDefined();
    expect(grazingGeo.index).toBeDefined();

    knurlGeo.dispose();
    veilGeo.dispose();
    grazingGeo.dispose();
    assembly.dispose();
    dummyMat.dispose();
  });

  it('instantiates all TSL materials and GPU compute sunlit dust motes cleanly without runtime errors', () => {
    const fluid = new FluidVeilSolver();
    const mats = createVeilMaterials(fluid.dataTexture);
    const motes = createVeilMotesSystem(fluid.dataTexture);

    expect(mats.forgedIron.colorNode).toBeTruthy();
    expect(mats.satinBrass.colorNode).toBeTruthy();
    expect(mats.vernierBrass.colorNode).toBeTruthy();
    expect(mats.knurledBrass.colorNode).toBeTruthy();
    expect(mats.bluedSteel.colorNode).toBeTruthy();
    expect(mats.honedSlate.colorNode).toBeTruthy();
    expect(mats.ceramicInsulator.colorNode).toBeTruthy();
    expect(mats.opticalGlass.colorNode).toBeTruthy();
    expect(mats.opticalBaffle.colorNode).toBeTruthy();
    expect(mats.veilMaterial.positionNode).toBeTruthy();
    expect(mats.veilMaterial.colorNode).toBeTruthy();
    expect(mats.grazingMaterial.colorNode).toBeTruthy();
    expect(motes.moteCount).toBe(VEIL_MOTE_COUNT);
    expect(motes.sprite.count).toBe(VEIL_MOTE_COUNT);

    motes.dispose();
    mats.dispose();
    fluid.dispose();
  });
});
