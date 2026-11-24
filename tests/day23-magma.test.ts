import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  ACME_THREAD_PITCH,
  CHAMBER_BOTTOM_Y,
  CHAMBER_ROOF_Y,
  COLUMN_BOUNDARIES_X,
  CORE_HALF_D,
  CORE_HALF_W,
  EMBER_COUNT,
  MAX_HEAVE_DISPLACEMENT,
  MAX_PISTON_TRAVEL,
  OVERBURDEN_BOTTOM_Y,
  OVERBURDEN_HEIGHT,
  OVERBURDEN_TOP_Y,
  ROCK_CHIP_COUNT,
  SLAB_COLUMNS,
  SLAB_COUNT,
  STRATA_TIERS,
  TIER_BOUNDARIES_Y,
  TectonicMagmaSolver,
  computeSillFraction,
  evalColumnSeamX,
  evalMagmaBlackbody,
} from '../src/days/23-magma/tectonicSolver';
import {
  buildHydraulicPistonAssembly,
  buildLateralClampAssemblies,
  buildMagmaSystemAssembly,
  buildPressFrameAssembly,
  buildRockCoreAssembly,
  evalAcmeThreadRadius,
  updateDikeGeometry,
  updateSillGeometry,
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
      gaugeDial: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      rockChip: dummyMat,
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
      gaugeDial: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      rockChip: dummyMat,
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

  it('enforces rotated-corner slab-to-platen and Tier-0 chamber-roof clearances under peak dome heave and tension', () => {
    const solver = new TectonicMagmaSolver({
      initialPressure: 1.0,
      initialCompression: 1.0,
      preWarmSec: 0,
    });

    const sweeps: Array<[number, number]> = [
      [1.0, 1.0],
      [1.0, -1.0],
      [0.95, 0.85],
      [0.95, -0.85],
      [0.5, 0.0],
    ];

    for (const [p, c] of sweeps) {
      solver.setTargetPressure(p);
      solver.setTargetCompression(c);
      for (let step = 0; step < 180; step++) {
        solver.step(1 / 60);
        // Lateral platen inner face sits at ±(CORE_HALF_W + platenOffset + 0.004)
        const platenInnerX = CORE_HALF_W + solver.platenOffset + 0.004;

        for (let i = 0; i < solver.slabs.length; i++) {
          const s = solver.slabs[i];
          const halfW = 0.5 * (COLUMN_BOUNDARIES_X[s.col + 1] - COLUMN_BOUNDARIES_X[s.col]);
          const halfH = 0.5 * (TIER_BOUNDARIES_Y[s.tier + 1] - TIER_BOUNDARIES_Y[s.tier]);
          const rotReachX = halfW * Math.cos(s.rotZ) + (halfH + 0.009) * Math.abs(Math.sin(s.rotZ));

          if (s.col === 0) {
            const leftMostCornerX = s.restCenterX + s.dx - rotReachX;
            expect(leftMostCornerX).toBeGreaterThanOrEqual(-platenInnerX - 1e-4);
          }
          if (s.col === SLAB_COLUMNS - 1) {
            const rightMostCornerX = s.restCenterX + s.dx + rotReachX;
            expect(rightMostCornerX).toBeLessThanOrEqual(platenInnerX + 1e-4);
          }
          if (s.tier === 0) {
            const lowestCornerY =
              s.restCenterY + s.dy - (halfH * Math.cos(s.rotZ) + halfW * Math.abs(Math.sin(s.rotZ)));
            expect(lowestCornerY).toBeGreaterThanOrEqual(CHAMBER_ROOF_Y - 0.002);
          }
        }
      }
    }
  });

  it('couples true helical ACME screw threads coherently to platen travel with zero slip at the stationary bushing', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      forgedPressIron: dummyMat,
      machinedThreadSteel: dummyMat,
      satinBrass: dummyMat,
      gaugeDial: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      rockChip: dummyMat,
      emberSpark: dummyMat,
    };

    const clamps = buildLateralClampAssemblies(mats);
    const screwMesh = clamps.group.getObjectByName('LateralAcmeScrew_Right') as THREE.Mesh;
    expect(screwMesh).toBeDefined();

    // Verify screw geometry is truly helical (radius varies with angle at fixed axial y near center)
    const pos = screwMesh.geometry.attributes.position as THREE.BufferAttribute;
    let rMinAtMidY = Infinity;
    let rMaxAtMidY = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      if (Math.abs(y) < 0.003) {
        const r = Math.hypot(pos.getX(i), pos.getZ(i));
        if (r < rMinAtMidY) rMinAtMidY = r;
        if (r > rMaxAtMidY) rMaxAtMidY = r;
      }
    }
    expect(rMaxAtMidY - rMinAtMidY).toBeGreaterThan(0.012);

    // Verify exact helical phase invariance at stationary bushing x = +1.24 across platen travel
    const solver = new TectonicMagmaSolver({ initialPressure: 0.5, initialCompression: -0.9, preWarmSec: 1.5 });
    const evalBushingThreadRadius = (platenOffset: number, screwAngle: number, phiWorld: number): number => {
      // Right screw local y at stationary world x = 1.24 is yLocal = (1.24 - 1.28) - platenOffset
      const yLocal = -0.04 - platenOffset;
      // Right screw rotated -PI/2 around Z then +screwAngle around X maps world angle phiWorld to theta = phiWorld + PI - screwAngle
      const thetaLocal = phiWorld + Math.PI - screwAngle;
      const phase = yLocal / ACME_THREAD_PITCH + thetaLocal / (Math.PI * 2);
      return evalAcmeThreadRadius(0.041, 0.059, phase);
    };

    const rAtTension = evalBushingThreadRadius(solver.platenOffset, solver.clampScrewAngle, 0.4);
    solver.setTargetCompression(0.9);
    for (let i = 0; i < 180; i++) solver.step(1 / 60);
    const rAtCompression = evalBushingThreadRadius(solver.platenOffset, solver.clampScrewAngle, 0.4);
    expect(Math.abs(rAtTension - rAtCompression)).toBeLessThan(1e-5);

    clamps.dispose();
    dummyMat.dispose();
  });

  it('conforms deformable dikes to jagged fracture seams and arches sills cleanly inside outer rock flanks', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      forgedPressIron: dummyMat,
      machinedThreadSteel: dummyMat,
      satinBrass: dummyMat,
      gaugeDial: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      rockChip: dummyMat,
      emberSpark: dummyMat,
    };
    const magma = buildMagmaSystemAssembly(mats);

    // 1. Extensional dike tracks full jagged evalColumnSeamX contour across Y (never flattened to straight plane!)
    const extSolver = new TectonicMagmaSolver({ initialPressure: 0.95, initialCompression: -0.9, preWarmSec: 3.0 });
    const d0 = extSolver.dikes[0];
    const dikeGeo = magma.dikeMeshes[0].geometry;
    updateDikeGeometry(dikeGeo, 0, d0.height, Math.max(d0.aperture * 1.15, 0.02), extSolver.slabs);
    const dPos = dikeGeo.attributes.position as THREE.BufferAttribute;

    let minCenterX = Infinity;
    let maxCenterX = -Infinity;
    let maxDikeY = -Infinity;
    for (let i = 0; i < dPos.count; i++) {
      const y = dPos.getY(i);
      const z = dPos.getZ(i);
      if (y > maxDikeY) maxDikeY = y;
      if (Math.abs(z) < 0.08 && y > OVERBURDEN_BOTTOM_Y + 0.05 && y < OVERBURDEN_BOTTOM_Y + OVERBURDEN_HEIGHT - 0.08) {
        const x = dPos.getX(i);
        if (x < minCenterX) minCenterX = x;
        if (x > maxCenterX) maxCenterX = x;
      }
    }
    // Seam contour varies by +/-0.022m along Y (total swing > 0.04m); dike must preserve that jag!
    const expectedSeamMin = Math.min(evalColumnSeamX(2, 0.0, 0), evalColumnSeamX(2, 0.3, 0), evalColumnSeamX(2, 0.5, 0));
    const expectedSeamMax = Math.max(evalColumnSeamX(2, 0.0, 0), evalColumnSeamX(2, 0.3, 0), evalColumnSeamX(2, 0.5, 0));
    expect(maxCenterX - minCenterX).toBeGreaterThan((expectedSeamMax - expectedSeamMin) * 0.95);
    // Dike top stays capped cleanly below upper basalt crest (zero shark-fin overshoot)
    expect(maxDikeY).toBeLessThan(OVERBURDEN_BOTTOM_Y + OVERBURDEN_HEIGHT + extSolver.heaveDisplacement + 0.025);

    // 2. Compressional sill arches higher at center than flanks AND stays strictly inside outer rock edges
    const compSolver = new TectonicMagmaSolver({ initialPressure: 0.95, initialCompression: 0.95, preWarmSec: 3.0 });
    const s0 = compSolver.sills[0];
    const sillGeo = magma.sillMeshes[0].geometry;
    updateSillGeometry(sillGeo, 0, s0.halfWidth, s0.thickness * 1.12, compSolver.slabs);
    const sPos = sillGeo.attributes.position as THREE.BufferAttribute;

    let centerYAvg = 0;
    let centerCount = 0;
    let flankYAvg = 0;
    let flankCount = 0;
    let maxAbsSillX = 0;
    for (let i = 0; i < sPos.count; i++) {
      const x = sPos.getX(i);
      const y = sPos.getY(i);
      if (Math.abs(x) > maxAbsSillX) maxAbsSillX = Math.abs(x);
      if (Math.abs(x) < 0.12) {
        centerYAvg += y;
        centerCount++;
      } else if (Math.abs(x) > 0.50) {
        flankYAvg += y;
        flankCount++;
      }
    }
    centerYAvg /= Math.max(1, centerCount);
    flankYAvg /= Math.max(1, flankCount);
    expect(centerYAvg).toBeGreaterThan(flankYAvg + 0.015);
    expect(maxAbsSillX).toBeLessThan(CORE_HALF_W + compSolver.platenOffset);

    magma.dispose();
    dummyMat.dispose();
  });

  it('verifies watertight hardware craft: full-skirted plunger, non-overlapping bezels, recessed trunnion bushings, visible xenoliths, Bourdon gauge & welded obsidian lobes', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      forgedPressIron: dummyMat,
      machinedThreadSteel: dummyMat,
      satinBrass: dummyMat,
      gaugeDial: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      rockChip: dummyMat,
      emberSpark: dummyMat,
    };

    const frame = buildPressFrameAssembly(mats);
    const piston = buildHydraulicPistonAssembly(mats);
    const clamps = buildLateralClampAssemblies(mats);
    const magma = buildMagmaSystemAssembly(mats);

    // Move plunger to 100% upward stroke
    piston.ramGroup.position.y = MAX_PISTON_TRAVEL;
    frame.group.updateMatrixWorld(true);
    piston.group.updateMatrixWorld(true);
    clamps.group.updateMatrixWorld(true);
    magma.group.updateMatrixWorld(true);

    const boxOf = (root: THREE.Object3D, name: string): THREE.Box3 => {
      const obj = root.getObjectByName(name);
      expect(obj, `mesh ${name} exists`).toBeDefined();
      return new THREE.Box3().setFromObject(obj!);
    };

    // 1. Full-skirted hydraulic plunger stays seated inside base bore even at 100% upward stroke (zero floating gap)
    const plungerMaxBox = boxOf(piston.group, 'ChamberInternalPlunger');
    expect(plungerMaxBox.min.y).toBeLessThanOrEqual(CHAMBER_BOTTOM_Y + 0.005);

    // 2. Side trunnion brass bushing sits cleanly inside forged iron trunnion bar vertical bounds (zero top/bot bleed)
    const trunnionBar = boxOf(frame.group, 'SideTrunnionBracket_Right');
    const trunnionBushing = boxOf(frame.group, 'SideTrunnionBushing_Right');
    expect(trunnionBushing.max.y).toBeLessThan(trunnionBar.max.y - 0.005);
    expect(trunnionBushing.min.y).toBeGreaterThan(trunnionBar.min.y + 0.005);

    // 3. Quartz brass bezel vertical rails butt cleanly between top & bottom rails (zero coplanar corner z-fighting)
    const bezelTop = boxOf(frame.group, 'QuartzBezelTop_Front');
    const bezelBot = boxOf(frame.group, 'QuartzBezelBot_Front');
    const bezelL = boxOf(frame.group, 'QuartzBezelL_Front');
    expect(bezelL.max.y).toBeLessThanOrEqual(bezelTop.min.y + 0.001);
    expect(bezelL.min.y).toBeGreaterThanOrEqual(bezelBot.max.y - 0.001);

    // 4. Diegetic Bourdon pressure gauge clears lateral platen shoe, lower base platen & right tie-rod cleanly
    const gaugeCase = boxOf(frame.group, 'BourdonGaugeCase');
    const rightPlaten = boxOf(clamps.group, 'LateralPlatenShoe_Right');
    const rightTieRod = boxOf(frame.group, 'TieRodShaft_R_F');
    expect(gaugeCase.max.y).toBeLessThan(rightPlaten.min.y - 0.05);
    expect(gaugeCase.min.y).toBeGreaterThan(CHAMBER_BOTTOM_Y + 0.05);
    expect(gaugeCase.max.x).toBeLessThan(rightTieRod.min.x - 0.05);

    // 5. Front xenolith emerges past opaque chamberMeltMesh front face while clearing quartz pane behind it
    const meltBox = boxOf(magma.group, 'MagmaChamberMeltVolume');
    const frontXeno = boxOf(magma.group, 'ChamberXenolith_0');
    const frontGlass = boxOf(frame.group, 'QuartzCutawayGlass_Front');
    expect(frontXeno.max.z).toBeGreaterThan(meltBox.max.z + 0.01);
    expect(frontXeno.max.z).toBeLessThan(frontGlass.min.z);

    // 6. Pahoehoe lobe geometry is indexed (welded vertices -> silky smooth obsidian normals, zero flat facets)
    expect(magma.lobeInstancedMesh.geometry.index).not.toBeNull();

    frame.dispose();
    piston.dispose();
    clamps.dispose();
    magma.dispose();
    dummyMat.dispose();
  });

  it('verifies branching hairline crack pressure response, multi-vent lobe surface anchoring, press clearance, and zero-allocation particle updates', () => {
    // 1. Hairline fracture openness & inside-out thermal illumination respond cleanly to pressure & quench
    const solver = new TectonicMagmaSolver({
      initialPressure: 0.08,
      initialCompression: -0.72,
      preWarmSec: 0.5,
    });
    const lowOpen = solver.hairlineCrackOpenness;
    const lowHeat = solver.hairlineCrackHeat;

    // Capture pre-allocated object identities for strict zero-allocation verification
    const chipRefs = solver.rockChips.slice();
    const emberRefs = solver.embers.slice();
    const lobeRefs = solver.lobes.slice();
    expect(chipRefs.length).toBe(ROCK_CHIP_COUNT);
    expect(emberRefs.length).toBe(EMBER_COUNT);

    // Pump sharply into high-pressure extensional rifting
    solver.setTargetPressure(0.94);
    let sawDislodgedChips = false;
    let sawRisingEmbers = false;
    for (let i = 0; i < 240; i++) {
      solver.step(1 / 60);
      const st = solver.getState();
      if (st.activeRockChips > 0) sawDislodgedChips = true;
      if (st.activeEmbers > 0) sawRisingEmbers = true;
    }

    const highOpen = solver.hairlineCrackOpenness;
    const highHeat = solver.hairlineCrackHeat;
    expect(highOpen).toBeGreaterThan(lowOpen + 0.45);
    expect(highHeat).toBeGreaterThan(lowHeat + 0.45);
    expect(sawDislodgedChips).toBe(true);
    expect(sawRisingEmbers).toBe(true);

    // Verify both crest-dike AND face-dike oozing pillows spawn and stay anchored inside press bounds
    let crestCount = 0;
    let faceCount = 0;
    const extMaxPressX = CORE_HALF_W + solver.platenOffset;
    for (let i = 0; i < solver.lobes.length; i++) {
      const l = solver.lobes[i];
      expect(l).toBe(lobeRefs[i]);
      if (!l.active) continue;
      if (l.ventType === 'crest-dike') crestCount++;
      if (l.ventType === 'face-dike') faceCount++;
      expect(Math.abs(l.posX)).toBeLessThanOrEqual(extMaxPressX);
      expect(l.posY).toBeGreaterThanOrEqual(CHAMBER_ROOF_Y + 0.05);
      expect(l.posY).toBeLessThanOrEqual(OVERBURDEN_TOP_Y + MAX_HEAVE_DISPLACEMENT + 0.032);
      expect(Math.abs(l.posZ)).toBeLessThanOrEqual(CORE_HALF_D + 0.025);
    }
    expect(crestCount).toBeGreaterThanOrEqual(1);
    expect(faceCount).toBeGreaterThanOrEqual(1);

    // Verify compressional sill switching spawns flank-sill oozing beads cleanly inside tightened platens
    solver.setTargetCompression(0.92);
    for (let i = 0; i < 240; i++) {
      solver.step(1 / 60);
    }
    let flankCount = 0;
    const compMaxPressX = CORE_HALF_W + solver.platenOffset;
    for (let i = 0; i < solver.lobes.length; i++) {
      const l = solver.lobes[i];
      if (!l.active) continue;
      if (l.ventType === 'flank-sill') flankCount++;
      expect(Math.abs(l.posX)).toBeLessThanOrEqual(compMaxPressX);
      expect(l.posY).toBeLessThanOrEqual(OVERBURDEN_TOP_Y + MAX_HEAVE_DISPLACEMENT + 0.032);
    }
    expect(flankCount).toBeGreaterThanOrEqual(1);

    // Verify zero object replacement across all particle pools
    for (let i = 0; i < ROCK_CHIP_COUNT; i++) {
      expect(solver.rockChips[i]).toBe(chipRefs[i]);
    }
    for (let i = 0; i < EMBER_COUNT; i++) {
      expect(solver.embers[i]).toBe(emberRefs[i]);
    }

    // Quench & settle: hairline heat drops sharply, permanent dark micro-crack scar persists, embers extinguish!
    solver.quenchAndReset(false);
    for (let i = 0; i < 180; i++) {
      solver.step(1 / 60);
    }
    expect(solver.hairlineCrackHeat).toBeLessThan(0.14);
    expect(solver.hairlineCrackOpenness).toBeGreaterThan(0.18);
    expect(solver.getState().activeEmbers).toBe(0);
    expect(Math.abs(solver.gaugeNeedleAngle)).toBeLessThanOrEqual(2.31);

    // Verify rock slabs expose finite aSlipMask (vertical slickenside fault-wall mask) & outward-wound rock chips
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      forgedPressIron: dummyMat,
      machinedThreadSteel: dummyMat,
      satinBrass: dummyMat,
      gaugeDial: dummyMat,
      quartzGlass: dummyMat,
      stratifiedRockCore: dummyMat,
      xenolithBasalt: dummyMat,
      chamberMagma: dummyMat,
      dikeMagma: dummyMat,
      sillMagma: dummyMat,
      pahoehoeLobe: dummyMat,
      rockChip: dummyMat,
      emberSpark: dummyMat,
    };
    const rock = buildRockCoreAssembly(mats);
    let totalSlipVerts = 0;
    for (let i = 0; i < rock.slabMeshes.length; i++) {
      const slipAttr = rock.slabMeshes[i].geometry.attributes.aSlipMask as THREE.BufferAttribute;
      expect(slipAttr).toBeDefined();
      for (let v = 0; v < slipAttr.count; v++) {
        const val = slipAttr.getX(v);
        expect(Number.isFinite(val)).toBe(true);
        if (val > 0.5) totalSlipVerts++;
      }
    }
    expect(totalSlipVerts).toBeGreaterThan(100);

    // Pressurized dike meniscus swells cleanly proud of recessed rest plane while capping below basalt crest
    const magma = buildMagmaSystemAssembly(mats);
    expect(magma.rockChipInstancedMesh.count).toBe(ROCK_CHIP_COUNT);
    const ext2 = new TectonicMagmaSolver({ initialPressure: 0.92, initialCompression: -0.85, preWarmSec: 2.5 });
    const d0 = ext2.dikes[0];
    updateDikeGeometry(magma.dikeMeshes[0].geometry, 0, d0.height, Math.max(d0.aperture * 1.18, 0.02), ext2.slabs, 1.0, 1.0);
    const dPos = magma.dikeMeshes[0].geometry.attributes.position as THREE.BufferAttribute;
    let maxZ = -Infinity;
    for (let v = 0; v < dPos.count; v++) {
      const z = dPos.getZ(v);
      if (z > maxZ) maxZ = z;
    }
    expect(maxZ).toBeGreaterThan(CORE_HALF_D - 0.008);
    expect(maxZ).toBeLessThan(CORE_HALF_D + 0.022);

    rock.dispose();
    magma.dispose();
    dummyMat.dispose();
  });
});
