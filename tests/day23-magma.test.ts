import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  COLUMN_BOUNDARIES_X,
  CORE_HALF_D,
  CORE_HALF_W,
  MAX_HEAVE_DISPLACEMENT,
  MAX_PISTON_TRAVEL,
  OVERBURDEN_HEIGHT,
  SLAB_COLUMNS,
  SLAB_COUNT,
  STRATA_TIERS,
  TectonicMagmaSolver,
  computeSillFraction,
  evalMagmaBlackbody,
} from '../src/days/23-magma/tectonicSolver';
import {
  buildHydraulicPistonAssembly,
  buildLateralClampAssemblies,
  buildMagmaSystemAssembly,
  buildPressFrameAssembly,
  buildRockCoreAssembly,
} from '../src/days/23-magma/geometry';

describe('Day 23 - Pressurized Tectonic Magma-Intrusion Block', () => {
  it('switches geomechanical regime between extensional vertical dikes and compressional horizontal sills + dome heave', () => {
    // 1. Extensional stress regime (lateral clamps pulled outward)
    const extSolver = new TectonicMagmaSolver({
      initialPressure: 0.90,
      initialCompression: -0.88,
      preWarmSec: 0,
    });
    for (let i = 0; i < 240; i++) {
      extSolver.step(1 / 60);
    }
    const extState = extSolver.getState();

    // 2. Compressional stress regime (lateral clamps cranked tightly inward)
    const compSolver = new TectonicMagmaSolver({
      initialPressure: 0.90,
      initialCompression: 0.88,
      preWarmSec: 0,
    });
    for (let i = 0; i < 240; i++) {
      compSolver.step(1 / 60);
    }
    const compState = compSolver.getState();

    // Regime partitioning assertions
    expect(extState.stressRegime).toBe('extensional');
    expect(compState.stressRegime).toBe('compressional');
    expect(extState.sillFraction).toBeLessThan(0.10);
    expect(compState.sillFraction).toBeGreaterThan(0.90);

    // Extensional regime drives sharp vertical dikes through the top crest & erupts lobes
    expect(extState.activeDikeHeight).toBeGreaterThan(OVERBURDEN_HEIGHT * 0.88);
    expect(extSolver.dikes[0].breachedCrest).toBe(true);
    expect(extSolver.dikes[0].aperture).toBeGreaterThan(0.05);
    expect(extState.extrudedLobes).toBeGreaterThanOrEqual(2);

    // Compressional regime suppresses vertical dike ascent & redirects melt into thick horizontal sills
    expect(compState.activeDikeHeight).toBeLessThan(extState.activeDikeHeight * 0.65);
    expect(compSolver.sills[0].halfWidth).toBeGreaterThan(0.55);
    expect(compSolver.sills[0].thickness).toBeGreaterThan(0.065);
    expect(compSolver.sills[0].thickness).toBeGreaterThan(extSolver.sills[0].thickness * 4.0);

    // Compressional sills physically heave & dome overburden slabs significantly higher than extensional dikes
    expect(compState.heaveDisplacement).toBeGreaterThan(0.12);
    expect(compState.heaveDisplacement).toBeGreaterThan(extState.heaveDisplacement * 1.8);

    // Live regime switch: cranking clamps inward mid-run transfers intrusion to horizontal sills cleanly
    extSolver.setTargetCompression(0.92);
    for (let i = 0; i < 240; i++) {
      extSolver.step(1 / 60);
    }
    const switched = extSolver.getState();
    expect(switched.stressRegime).toBe('compressional');
    expect(switched.sillFraction).toBeGreaterThan(0.90);
    expect(switched.heaveDisplacement).toBeGreaterThan(0.11);
    expect(switched.maxSillThickness).toBeGreaterThan(0.06);
  });

  it('maintains symmetric laccolith dome heave balance and enforces strict steel platen & press bounding clamps', () => {
    const solver = new TectonicMagmaSolver({
      initialPressure: 0.95,
      initialCompression: 0.92,
      preWarmSec: 0,
    });

    // Sweep pressure & compression across full physical envelope
    const checkpoints: Array<[number, number]> = [
      [1.0, 1.0],
      [1.0, -1.0],
      [0.85, 0.75],
      [0.20, 1.0],
      [0.95, -0.95],
      [0.0, 0.0],
    ];

    for (const [p, c] of checkpoints) {
      solver.setTargetPressure(p);
      solver.setTargetCompression(c);
      for (let step = 0; step < 150; step++) {
        solver.step(1 / 60);

        const maxOuterEdgeX = CORE_HALF_W + solver.platenOffset + 1e-4;
        expect(solver.heaveDisplacement).toBeGreaterThanOrEqual(0);
        expect(solver.heaveDisplacement).toBeLessThanOrEqual(MAX_HEAVE_DISPLACEMENT + 1e-6);
        expect(solver.pistonTravel).toBeGreaterThanOrEqual(0);
        expect(solver.pistonTravel).toBeLessThanOrEqual(MAX_PISTON_TRAVEL + 1e-6);

        for (let i = 0; i < solver.slabs.length; i++) {
          const s = solver.slabs[i];
          expect(Number.isFinite(s.dx)).toBe(true);
          expect(Number.isFinite(s.dy)).toBe(true);
          expect(Number.isFinite(s.rotZ)).toBe(true);
          expect(s.dy).toBeGreaterThanOrEqual(0);
          expect(s.dy).toBeLessThanOrEqual(MAX_HEAVE_DISPLACEMENT + 1e-6);

          // Leftmost & rightmost slabs must never clip outside lateral forged-iron platens
          if (s.col === 0) {
            expect(COLUMN_BOUNDARIES_X[0] + s.dx).toBeGreaterThanOrEqual(-maxOuterEdgeX);
          }
          if (s.col === SLAB_COLUMNS - 1) {
            expect(COLUMN_BOUNDARIES_X[SLAB_COLUMNS] + s.dx).toBeLessThanOrEqual(maxOuterEdgeX);
          }
        }
      }
    }

    // Verify anticlinal laccolith dome symmetry under peak compression
    solver.setTargetPressure(0.95);
    solver.setTargetCompression(0.90);
    for (let i = 0; i < 220; i++) {
      solver.step(1 / 60);
    }
    const topLeftOuter = solver.slabs[2 * SLAB_COLUMNS + 0];
    const topLeftInner = solver.slabs[2 * SLAB_COLUMNS + 1];
    const topRightInner = solver.slabs[2 * SLAB_COLUMNS + 2];
    const topRightOuter = solver.slabs[2 * SLAB_COLUMNS + 3];

    // Central crown domes higher than outer shoulders
    expect(topLeftInner.dy).toBeGreaterThan(topLeftOuter.dy);
    expect(topRightInner.dy).toBeGreaterThan(topRightOuter.dy);

    // Exact left-right symmetry in heave, lateral offset & outward anticlinal roll
    expect(topLeftInner.dy).toBeCloseTo(topRightInner.dy, 3);
    expect(topLeftOuter.dy).toBeCloseTo(topRightOuter.dy, 3);
    expect(topLeftOuter.dx).toBeCloseTo(-topRightOuter.dx, 3);
    expect(topLeftOuter.rotZ).toBeGreaterThan(0.04);
    expect(topRightOuter.rotZ).toBeLessThan(-0.04);
    expect(topLeftOuter.rotZ).toBeCloseTo(-topRightOuter.rotZ, 3);
  });

  it('enforces monotonic blackbody thermal cooling from incandescent pale-gold to glossy black obsidian', () => {
    let prevLum = -1;
    let prevTempC = -1;
    const samples = 50;

    for (let i = 0; i <= samples; i++) {
      const t = i / samples;
      const res = evalMagmaBlackbody(t);

      expect(Number.isFinite(res.luminance)).toBe(true);
      expect(res.luminance).toBeGreaterThan(prevLum);
      expect(res.temperatureC).toBeGreaterThan(prevTempC);

      // Strictly warm mineral blackbody spectrum (zero digital neon blue/cyan/magenta spikes)
      if (t >= 0.14) {
        const [r, g, b] = res.albedoLinear;
        expect(r).toBeGreaterThanOrEqual(g);
        expect(g).toBeGreaterThanOrEqual(b);
      }

      prevLum = res.luminance;
      prevTempC = res.temperatureC;
    }

    // Quenched obsidian glass (t = 0): jet-black, zero emission, glossy specular roughness (~0.14)
    const obsidian = evalMagmaBlackbody(0);
    expect(obsidian.emissiveLinear[0]).toBe(0);
    expect(obsidian.emissiveLinear[1]).toBe(0);
    expect(obsidian.emissiveLinear[2]).toBe(0);
    expect(Math.max(...obsidian.albedoLinear)).toBeLessThan(0.025);
    expect(obsidian.roughness).toBeGreaterThanOrEqual(0.13);
    expect(obsidian.roughness).toBeLessThanOrEqual(0.16);

    // Blistering dark-red crust transition (t ~ 0.26) has elevated rough clinker texture
    const blistering = evalMagmaBlackbody(0.26);
    expect(blistering.roughness).toBeGreaterThan(0.45);

    // Incandescent core (t = 1.0): ~1180°C high HDR pale-gold radiance
    const core = evalMagmaBlackbody(1.0);
    expect(core.temperatureC).toBeCloseTo(1180, 0);
    expect(core.emissiveLinear[0]).toBeGreaterThan(1.5);
    expect(core.roughness).toBeLessThan(blistering.roughness);

    // Sill fraction monotonicity across compression [-1, +1]
    let prevSill = 0;
    for (let c = -1; c <= 1.001; c += 0.1) {
      const sf = computeSillFraction(c);
      expect(sf).toBeGreaterThanOrEqual(prevSill);
      prevSill = sf;
    }
  });

  it('cools extruded pahoehoe lobes into obsidian, re-cracks them on pressure surges, and quenches cleanly on reset', () => {
    const solver = new TectonicMagmaSolver({
      initialPressure: 0.92,
      initialCompression: -0.78,
      preWarmSec: 0,
    });

    // Pump long enough to extrude multiple hot crest lobes
    for (let i = 0; i < 240; i++) {
      solver.step(1 / 60);
    }
    const activeLobes = solver.lobes.filter((l) => l.active);
    expect(activeLobes.length).toBeGreaterThanOrEqual(2);

    const probeLobe = activeLobes[0];
    const hotCrustTemp = probeLobe.crustTemp;
    expect(hotCrustTemp).toBeGreaterThan(0.22);

    // Depressurize gently and let outer skin cool into glossy dark obsidian over ~8 seconds
    solver.setTargetPressure(0.12);
    for (let i = 0; i < 480; i++) {
      solver.step(1 / 60);
    }
    const cooledCrust = probeLobe.crustTemp;
    const cooledCrack = probeLobe.crackIntensity;
    expect(cooledCrust).toBeLessThan(0.22);
    expect(cooledCrust).toBeLessThan(hotCrustTemp);
    expect(probeLobe.coreTemp).toBeGreaterThanOrEqual(cooledCrust);

    // Fresh hydraulic pressure surge fractures the cooled obsidian skin open with glowing cinnabar seams!
    solver.setTargetPressure(0.98);
    let peakReCrack = cooledCrack;
    for (let i = 0; i < 120; i++) {
      solver.step(1 / 60);
      if (probeLobe.crackIntensity > peakReCrack) {
        peakReCrack = probeLobe.crackIntensity;
      }
    }
    expect(peakReCrack).toBeGreaterThan(cooledCrack + 0.12);

    // Quench & Reset releases pressure, chills veins/lobes, and relaxes overburden plates cleanly
    solver.quenchAndReset(false);
    for (let i = 0; i < 240; i++) {
      solver.step(1 / 60);
    }
    const resetState = solver.getState();
    expect(resetState.chamberPressure).toBeLessThan(0.02);
    expect(resetState.heaveDisplacement).toBeLessThan(0.015);
    expect(resetState.quenchedObsidianFraction).toBeGreaterThan(0.8);
  });

  it('enforces strict outward triangle winding (signedVol > 0 and outwardDot > 0) across all procedural meshes', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      forgedPressIron: dummyMat,
      machinedThreadSteel: dummyMat,
      satinBrass: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      emberSpark: dummyMat,
    };

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
      const bCrossC = new THREE.Vector3();

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

        bCrossC.crossVectors(b, c);
        signedVol += a.dot(bCrossC) / 6;

        ab.subVectors(b, a);
        ac.subVectors(c, a);
        n.crossVectors(ab, ac);
        cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(boxCenter);
        outwardDot += n.dot(cen);
      }

      expect(Number.isFinite(signedVol), `${label} finite signedVol`).toBe(true);
      expect(signedVol, `${label} signedVol`).toBeGreaterThan(0);
      expect(outwardDot, `${label} outwardDot`).toBeGreaterThan(0);
    };

    const checkedGeos = new Set<THREE.BufferGeometry>();
    const verifyAssembly = (prefix: string, root: THREE.Object3D) => {
      let meshCount = 0;
      root.traverse((obj: any) => {
        if (!obj.isMesh) return;
        if (typeof obj.name === 'string' && obj.name.includes('HitProxy')) return;
        meshCount++;
        if (!checkedGeos.has(obj.geometry)) {
          checkedGeos.add(obj.geometry);
          assertOutward(`${prefix}/${obj.name || obj.geometry.type}`, obj.geometry);
        }
      });
      expect(meshCount, `${prefix} meshCount`).toBeGreaterThan(0);
    };

    const frame = buildPressFrameAssembly(mats);
    verifyAssembly('pressFrame', frame.group);
    frame.dispose();

    const piston = buildHydraulicPistonAssembly(mats);
    verifyAssembly('piston', piston.group);
    piston.dispose();

    const clamps = buildLateralClampAssemblies(mats);
    verifyAssembly('lateralClamps', clamps.group);
    clamps.dispose();

    const rock = buildRockCoreAssembly(mats);
    expect(rock.slabMeshes.length).toBe(SLAB_COUNT);
    expect(rock.slabMeshes.length).toBe(STRATA_TIERS * SLAB_COLUMNS);
    verifyAssembly('rockCore', rock.group);

    // Verify 3D object-space petrography attributes exist with finite values on all 12 rock slabs
    for (let i = 0; i < rock.slabMeshes.length; i++) {
      const geo = rock.slabMeshes[i].geometry;
      const restPos = geo.attributes.aRestPos as THREE.BufferAttribute;
      const stratum = geo.attributes.aStratum as THREE.BufferAttribute;
      const fracMask = geo.attributes.aFractureMask as THREE.BufferAttribute;
      const slabIdx = geo.attributes.aSlabIdx as THREE.BufferAttribute;
      expect(restPos).toBeDefined();
      expect(stratum).toBeDefined();
      expect(fracMask).toBeDefined();
      expect(slabIdx).toBeDefined();
      expect(restPos.count).toBe(geo.attributes.position.count);
      for (let v = 0; v < restPos.count; v++) {
        expect(Number.isFinite(restPos.getX(v))).toBe(true);
        expect(Number.isFinite(restPos.getY(v))).toBe(true);
        expect(Number.isFinite(restPos.getZ(v))).toBe(true);
      }
    }
    rock.dispose();

    const magma = buildMagmaSystemAssembly(mats);
    verifyAssembly('magmaSystem', magma.group);
    magma.dispose();

    dummyMat.dispose();
  });

  it('maintains watertight press clearances across full piston & lateral screw-clamp travel and exposes valid default boot telemetry', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      forgedPressIron: dummyMat,
      machinedThreadSteel: dummyMat,
      satinBrass: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      emberSpark: dummyMat,
    };

    const frame = buildPressFrameAssembly(mats);
    const piston = buildHydraulicPistonAssembly(mats);
    const clamps = buildLateralClampAssemblies(mats);

    // Compute world bounds at extreme extensional tension (platenOffset = +MAX_PLATEN_EXTENSION)
    // and full hydraulic piston stroke (pistonTravel = MAX_PISTON_TRAVEL)
    const extSolver = new TectonicMagmaSolver({
      initialPressure: 1.0,
      initialCompression: -1.0,
      preWarmSec: 3.0,
    });
    piston.ramGroup.position.y = extSolver.pistonTravel;
    clamps.rightPlatenGroup.position.x = extSolver.platenOffset;
    clamps.rightScrewWheelGroup.position.x = extSolver.platenOffset;

    frame.group.updateMatrixWorld(true);
    piston.group.updateMatrixWorld(true);
    clamps.group.updateMatrixWorld(true);

    const boxOf = (root: THREE.Object3D, name: string): THREE.Box3 => {
      const obj = root.getObjectByName(name);
      expect(obj, `mesh ${name} exists`).toBeDefined();
      return new THREE.Box3().setFromObject(obj!);
    };

    // 1. Sliding right lateral platen shoe clears the right side trunnion bracket at max tension
    const rightShoeBox = boxOf(clamps.group, 'LateralPlatenShoe_Right');
    const rightTrunnionBox = boxOf(frame.group, 'SideTrunnionBracket_Right');
    expect(rightShoeBox.max.x).toBeLessThan(rightTrunnionBox.min.x - 0.08);

    // 2. ACME threaded lead-screw continuously bridges swivel socket -> trunnion bushing -> capstan hub
    const rightScrewBox = boxOf(clamps.group, 'LateralAcmeScrew_Right');
    const rightBushingBox = boxOf(frame.group, 'SideTrunnionBushing_Right');
    expect(rightScrewBox.min.x).toBeLessThan(rightBushingBox.min.x);
    expect(rightScrewBox.max.x).toBeGreaterThan(rightBushingBox.max.x);

    // 3. Lower brass hydraulic foot collar clears stationary brass gland even at 100% upward stroke
    const glandBox = boxOf(piston.group, 'HydraulicBrassGland');
    const footBox = boxOf(piston.group, 'HydraulicFootBrassCollar');
    expect(footBox.max.y).toBeLessThan(glandBox.min.y - 0.02);

    // 4. Upper press bridge beam clears maximum overburden dome heave cleanly
    const upperBeamBox = boxOf(frame.group, 'UpperBridgeFrontBeam');
    expect(0.65 + MAX_HEAVE_DISPLACEMENT).toBeLessThan(upperBeamBox.min.y - 0.03);

    // 4b. Dark forged-iron plunger forms top face above narrow perimeter brass seal ring
    const plungerBox = boxOf(piston.group, 'ChamberInternalPlunger');
    const sealFrontBox = boxOf(piston.group, 'ChamberPlungerBrassSeal_F');
    expect(plungerBox.max.y).toBeGreaterThan(sealFrontBox.max.y + 0.015);

    // 4c. Vertical dikes & horizontal sills stay recessed inside [-CORE_HALF_D, +CORE_HALF_D]
    const magma = buildMagmaSystemAssembly(mats);
    magma.group.updateMatrixWorld(true);
    for (let i = 0; i < magma.dikeMeshes.length; i++) {
      const db = new THREE.Box3().setFromObject(magma.dikeMeshes[i]);
      expect(db.max.z).toBeLessThan(CORE_HALF_D);
      expect(db.min.z).toBeGreaterThan(-CORE_HALF_D);
    }
    for (let i = 0; i < magma.sillMeshes.length; i++) {
      const sb = new THREE.Box3().setFromObject(magma.sillMeshes[i]);
      expect(sb.max.z).toBeLessThan(CORE_HALF_D);
      expect(sb.min.z).toBeGreaterThan(-CORE_HALF_D);
    }
    magma.dispose();

    // 5. Default boot specimen opens visibly pressurized with valid telemetry contract
    const bootSolver = new TectonicMagmaSolver({
      initialPressure: 0.58,
      initialCompression: -0.34,
      preWarmSec: 2.6,
    });
    const st = bootSolver.getState();
    expect(st.chamberPressure).toBeGreaterThan(0.5);
    expect(st.tectonicCompression).toBeLessThan(-0.2);
    expect(st.activeDikeHeight).toBeGreaterThan(0.5);
    expect(st.sillFraction).toBeGreaterThan(0);
    expect(st.sillFraction).toBeLessThan(1);
    expect(st.heaveDisplacement).toBeGreaterThanOrEqual(0);
    expect(st.extrudedLobes).toBeGreaterThanOrEqual(0);

    frame.dispose();
    piston.dispose();
    clamps.dispose();
    dummyMat.dispose();
  });
});
