import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  AIR_CAVITY,
  AIR_SKY,
  CELL_EMPTY,
  CELL_LOOSE_SAND,
  CELL_PACKED_SAND,
  CELL_PEBBLE,
  DIGGER_COUNT,
  GRID_H,
  GRID_W,
} from '../src/days/27-burrow/constants';
import { SandSimulation } from '../src/days/27-burrow/sandSim';
import { buildTerrariumAssembly } from '../src/days/27-burrow/geometry';
import { createBurrowMaterials } from '../src/days/27-burrow/materials';

describe('Day 27 - Burrow (Glass-Walled Stratified Sand-Slice Terrarium)', () => {
  it('holds >90% of a narrow arched tunnel open after 120 physics steps (Narrow Arch Stability Gate)', () => {
    const sim = new SandSimulation(42);
    sim.reset(42, false); // Clean undisturbed stratified bed (no starter burrow interference)

    // Carve a narrow vaulted tunnel (radius 5.5 -> ~11 cells wide <= ARCH_SPAN_LIMIT=18)
    // deep beneath the top horizon (y = 92) along a gentle arch span
    const cx0 = 150;
    const cy0 = 90;
    const cx1 = 178;
    const cy1 = 98;
    const removed = sim.excavateCapsule(cx0, cy0, cx1, cy1, 5.4, false);
    expect(removed).toBeGreaterThan(220);

    // Snapshot the exact carved empty cells of our narrow arched tunnel
    const carvedCells: number[] = [];
    for (let y = 76; y <= 112; y++) {
      for (let x = 138; x <= 192; x++) {
        const idx = sim.idx(x, y);
        if (sim.cellType[idx] === CELL_EMPTY) {
          carvedCells.push(idx);
        }
      }
    }
    expect(carvedCells.length).toBeGreaterThan(200);

    // Step 120 granular physics ticks (with diggers disabled so we test pure arch stability)
    for (let t = 0; t < 120; t++) {
      sim.step({ enableDiggers: false, enableArchCheck: true });
    }

    let stillOpen = 0;
    for (let i = 0; i < carvedCells.length; i++) {
      if (sim.cellType[carvedCells[i]] === CELL_EMPTY) stillOpen++;
    }
    const openFraction = stillOpen / carvedCells.length;

    expect(sim.collapseEvents).toBe(0);
    expect(openFraction).toBeGreaterThan(0.9);
  });

  it('triggers gravitational shear-fault ceiling collapse when excavating an over-wide flat chamber (Cave-In Gate)', () => {
    const sim = new SandSimulation(17);
    sim.reset(17, false);

    // Excavate a wide flat unsupported void (44 cells wide x 18 cells tall -> exceeds ARCH_SPAN_LIMIT)
    const xMin = 128;
    const xMax = 172;
    const yMin = 76;
    const yMax = 94;
    const removed = sim.excavateBox(xMin, yMin, xMax, yMax);
    expect(removed).toBeGreaterThan(600);

    // Record pale upper Bone-Calcite/Kaolin & Burnt-Sienna strata mass originally above y = 104
    let initialUpperRowSum = 0;
    let initialUpperCount = 0;
    for (let y = yMax + 1; y < GRID_H - 10; y++) {
      for (let x = xMin; x <= xMax; x++) {
        const idx = sim.idx(x, y);
        if (sim.cellType[idx] === CELL_PACKED_SAND) {
          initialUpperRowSum += y;
          initialUpperCount++;
        }
      }
    }
    const initialUpperCentroidY = initialUpperRowSum / Math.max(1, initialUpperCount);

    // Run 180 physics steps -> shear-fault slip wedge must propagate upward and slump colored strata down!
    let peakFalling = 0;
    let midBedSkyVoids = 0;
    let checkedFallingCells = 0;

    for (let t = 0; t < 180; t++) {
      sim.step({ enableDiggers: false, enableArchCheck: true });
      if (sim.activeFallingGrains > peakFalling) peakFalling = sim.activeFallingGrains;

      // Regression gates on active mid-bed collapse ticks:
      // (a) Zero AIR_SKY holes punched inside active subterranean collapse zone (y <= 145)
      // (b) Falling LOOSE_SAND grains strictly carry fieldTex.a === 0 (never darkened by stressCrack!)
      if (t === 15 || t === 35 || t === 65 || t === 110) {
        sim.syncOpticalTextures();
        for (let y = yMin; y <= 145; y++) {
          for (let x = xMin; x <= xMax; x++) {
            const idx = sim.idx(x, y);
            if (sim.cellType[idx] === CELL_EMPTY && sim.airClass[idx] === AIR_SKY) {
              midBedSkyVoids++;
            }
            if (sim.cellType[idx] === CELL_LOOSE_SAND && sim.velY[idx] > 0.12) {
              checkedFallingCells++;
              expect(sim.fieldTexData[(idx << 2) + 3]).toBe(0);
              expect(sim.colorTexData[(idx << 2) + 3]).toBe(255);
            }
          }
        }
      }
    }

    expect(midBedSkyVoids).toBe(0);
    expect(checkedFallingCells).toBeGreaterThan(20);

    // Count how many fallen grains landed inside the originally excavated wide chamber
    let filledChamberCells = 0;
    const chamberTotal = (xMax - xMin + 1) * (yMax - yMin + 1);
    for (let y = yMin; y <= yMax; y++) {
      for (let x = xMin; x <= xMax; x++) {
        const ct = sim.cellType[sim.idx(x, y)];
        if (ct === CELL_LOOSE_SAND || ct === CELL_PACKED_SAND) {
          filledChamberCells++;
        }
      }
    }

    // Recompute mean Y of all sand in the vertical column [xMin..xMax] above yMin
    let postRowSum = 0;
    let postCount = 0;
    for (let y = yMin; y < GRID_H - 10; y++) {
      for (let x = xMin; x <= xMax; x++) {
        const ct = sim.cellType[sim.idx(x, y)];
        if (ct === CELL_PACKED_SAND || ct === CELL_LOOSE_SAND) {
          postRowSum += y;
          postCount++;
        }
      }
    }
    const postCentroidY = postRowSum / Math.max(1, postCount);

    expect(sim.collapseEvents).toBeGreaterThan(0);
    expect(peakFalling).toBeGreaterThan(25);
    expect(filledChamberCells / chamberTotal).toBeGreaterThan(0.32);
    // Overlying strata must visibly drop downward in mean elevation!
    expect(postCentroidY).toBeLessThan(initialUpperCentroidY - 4.0);
  });

  it('conserves exact solid sand grain count across gravity settling, avalanches, and digger hauling (Mass Conservation Gate)', () => {
    const sim = new SandSimulation(9);
    sim.reset(9, false);

    // 1. Non-excavating gravity settling + dramatic cave-in (diggers off): exact grid grain conservation!
    sim.excavateBox(115, 72, 160, 90);
    sim.pourSandAt(140, GRID_H - 12, 60);
    sim.pourSandAt(195, GRID_H - 10, 45);

    const exactGridStart = sim.countGridSandGrains();
    const exactPebblesStart = sim.countPebbleCells();
    expect(exactGridStart).toBeGreaterThan(30000);

    for (let t = 0; t < 200; t++) {
      sim.step({ enableDiggers: false, enableArchCheck: true });
      expect(sim.countGridSandGrains(), `step ${t} grid sand count`).toBe(exactGridStart);
      expect(sim.countPebbleCells(), `step ${t} pebble count`).toBe(exactPebblesStart);
    }

    // 2. Active autonomous diggers hauling grains: total mass (grid + carried in mandibles) is strictly invariant!
    sim.reset(11, true);
    const exactTotalMassStart = sim.countTotalSandMass();
    for (let t = 0; t < 260; t++) {
      sim.step({ enableDiggers: true, enableArchCheck: true });
      expect(sim.countTotalSandMass(), `digger step ${t} total mass`).toBe(exactTotalMassStart);
    }
  });

  it('drives autonomous burrowing diggers to excavate at depth and deposit tailings near the top surface', () => {
    const sim = new SandSimulation(3);
    const initialExcavated = sim.excavatedCells;
    expect(sim.diggers.length).toBe(DIGGER_COUNT);
    expect(sim.diggers.length).toBeGreaterThanOrEqual(3);

    let sawDeepExcavation = false;
    let sawCarryingClimb = false;

    for (let t = 0; t < 480; t++) {
      sim.step({ enableDiggers: true, enableArchCheck: true });
      for (let i = 0; i < sim.diggers.length; i++) {
        const d = sim.diggers[i];
        if (d.lastExcavatedY < 135 && d.carryingPigment >= 0) {
          sawDeepExcavation = true;
        }
        if (d.state === 'haul_up' && d.carryingPigment >= 0) {
          sawCarryingClimb = true;
        }
      }
    }

    const metrics = sim.getMetrics();
    expect(metrics.diggerCount).toBe(DIGGER_COUNT);
    expect(metrics.stepCount).toBe(480);
    expect(metrics.excavatedCells).toBeGreaterThan(initialExcavated);
    expect(metrics.tailingsDeposited).toBeGreaterThanOrEqual(3);
    expect(sawDeepExcavation).toBe(true);
    expect(sawCarryingClimb).toBe(true);

    // Verify top airspace classifies cleanly as AIR_SKY while deep starter tunnels classify as AIR_CAVITY
    sim.syncOpticalTextures();
    expect(sim.airClass[sim.idx(150, GRID_H - 4)]).toBe(AIR_SKY);
    let cavityCount = 0;
    for (let i = 0; i < sim.cellCount; i++) {
      if (sim.airClass[i] === AIR_CAVITY) cavityCount++;
    }
    expect(cavityCount).toBeGreaterThan(500);
  });

  it('keeps embedded river pebbles immovable and pours fresh loose sand cleanly from top airspace', () => {
    const sim = new SandSimulation(5);
    const pebbleCountBefore = sim.countPebbleCells();
    expect(pebbleCountBefore).toBeGreaterThan(400);
    expect(sim.pebbles.length).toBeGreaterThanOrEqual(6);

    // Attempt to excavate directly across the first river pebble -> pebble must remain intact!
    const p = sim.pebbles[0];
    sim.excavateCapsule(p.cx - 4, p.cy, p.cx + 4, p.cy, 6, false);
    expect(sim.countPebbleCells()).toBe(pebbleCountBefore);

    // Pour fresh sand into open top airspace
    const pourX = 210;
    const pourY = GRID_H - 8;
    expect(sim.isInAirPourZone(pourX, pourY)).toBe(true);
    const beforeMass = sim.countGridSandGrains();
    const spawned = sim.pourSandAt(pourX, pourY, 16);
    expect(spawned).toBeGreaterThan(0);
    expect(sim.countGridSandGrains()).toBe(beforeMass + spawned);

    // Step gravity and verify grains fall cleanly downward while conserving mass
    sim.step({ enableDiggers: false, enableArchCheck: false });
    expect(sim.activeFallingGrains).toBeGreaterThan(0);
    expect(sim.countGridSandGrains()).toBe(beforeMass + spawned);

    // Verify airborne poured grains have zero stressCrack darkening & zero cohesive parallax mask
    sim.syncOpticalTextures();
    let airborneChecked = 0;
    for (let y = GRID_H - 22; y < GRID_H - 2; y++) {
      for (let x = pourX - 8; x <= pourX + 8; x++) {
        const idx = sim.idx(x, y);
        if (sim.cellType[idx] === CELL_LOOSE_SAND && sim.velY[idx] > 0.12) {
          airborneChecked++;
          expect(sim.fieldTexData[idx << 2]).toBe(0); // zero parallax wall extrusion!
          expect(sim.fieldTexData[(idx << 2) + 3]).toBe(0); // zero stressCrack darkening!
          expect(sim.colorTexData[(idx << 2) + 3]).toBe(255);
        }
      }
    }
    expect(airborneChecked).toBeGreaterThan(0);
  });

  it('enforces outward triangle winding across all custom walnut, brass, glass, crest, and beetle geometry', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const assembly = buildTerrariumAssembly({
      oiledWalnut: dummyMat,
      satinBrass: dummyMat,
      agedBronze: dummyMat,
      beveledGlass: dummyMat,
      sandForeground: dummyMat,
      sandBackwall: dummyMat,
      sandCrestTop: dummyMat,
      beetleChitin: dummyMat,
      beetleAccent: dummyMat,
    });

    const openPlaneNames = new Set([
      'SandCrossSectionFront',
      'SandCrossSectionBack',
      'TunnelBackwallFront',
      'TunnelBackwallBack',
      'SandCrestTopRibbon',
    ]);

    const assertClosedOutward = (label: string, geo: THREE.BufferGeometry) => {
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

    const assertOpenSurfaceNormal = (
      label: string,
      geo: THREE.BufferGeometry,
      expectedDir: THREE.Vector3
    ) => {
      const pos = geo.attributes.position as THREE.BufferAttribute;
      const idx = geo.index;
      const triCount = idx ? idx.count / 3 : pos.count / 3;
      expect(triCount, `${label} open triCount`).toBeGreaterThan(0);

      let inwardTri = 0;
      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      const c = new THREE.Vector3();
      const ab = new THREE.Vector3();
      const ac = new THREE.Vector3();
      const n = new THREE.Vector3();
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
        if (n.dot(expectedDir) <= 0) inwardTri++;
      }
      expect(inwardTri, `${label} inwardTri`).toBe(0);
    };

    const checkedClosedGeos = new Set<THREE.BufferGeometry>();
    let visibleMeshCount = 0;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    assembly.group.traverse((obj: any) => {
      if (!obj.isMesh || !obj.material.visible) return;
      visibleMeshCount++;

      if (openPlaneNames.has(obj.name)) {
        const dir =
          obj.name === 'SandCrestTopRibbon'
            ? new THREE.Vector3(0, 1, 0)
            : obj.name === 'SandCrossSectionFront' || obj.name === 'TunnelBackwallFront'
              ? new THREE.Vector3(0, 0, 1)
              : new THREE.Vector3(0, 0, -1);
        assertOpenSurfaceNormal(obj.name, obj.geometry, dir);
        return;
      }

      if (!checkedClosedGeos.has(obj.geometry)) {
        checkedClosedGeos.add(obj.geometry);
        assertClosedOutward(obj.name || obj.geometry.type, obj.geometry);
      }
    });

    expect(visibleMeshCount).toBeGreaterThanOrEqual(75);
    expect(assembly.diggers.length).toBe(DIGGER_COUNT);

    // Regression Gate: Bottom brass corner brackets, side wraps & bottom corner screws must
    // clear the top shelf of WalnutBasePlinth cleanly (zero clipping into plinth!)
    assembly.group.updateMatrixWorld(true);
    const plinthObj = assembly.group.getObjectByName('WalnutBasePlinth');
    expect(plinthObj).toBeDefined();
    const plinthBox = new THREE.Box3().setFromObject(plinthObj!);
    const plinthTopY = plinthBox.max.y;

    let checkedBottomBrackets = 0;
    let checkedBottomWraps = 0;
    let checkedBottomScrews = 0;
    for (const child of assembly.group.children) {
      if (child.name.startsWith('BrassCornerBracket_') && child.name.split('_')[2] === '-1') {
        const box = new THREE.Box3().setFromObject(child);
        expect(box.min.y, `${child.name} clears plinth top`).toBeGreaterThanOrEqual(plinthTopY);
        checkedBottomBrackets++;
      } else if (child.name.startsWith('BrassSideWrap_') && child.name.endsWith('_-1')) {
        const box = new THREE.Box3().setFromObject(child);
        expect(box.min.y, `${child.name} clears plinth top`).toBeGreaterThanOrEqual(plinthTopY);
        checkedBottomWraps++;
      } else if (child.name.startsWith('CornerScrew') || child.name.startsWith('SideScrew')) {
        const box = new THREE.Box3().setFromObject(child);
        if (box.min.y < 0) {
          expect(box.min.y, `${child.name} clears plinth top`).toBeGreaterThanOrEqual(plinthTopY);
          checkedBottomScrews++;
        }
      }
    }
    expect(checkedBottomBrackets).toBe(4);
    expect(checkedBottomWraps).toBe(2);
    expect(checkedBottomScrews).toBe(12);

    // Exercise live 3D crest ribbon update with real sim data
    const sim = new SandSimulation(1);
    assembly.updateSandCrest(sim.surfaceY, sim.surfaceRgb);
    assertOpenSurfaceNormal('SandCrestTopRibbonAfterUpdate', assembly.sandCrestGeo, new THREE.Vector3(0, 1, 0));

    assembly.dispose();
    dummyMat.dispose();
  });

  it('instantiates all TSL WebGPU materials and DataTexture bindings cleanly without runtime errors', () => {
    const sim = new SandSimulation(1);
    const colorTex = new THREE.DataTexture(
      sim.colorTexData,
      GRID_W,
      GRID_H,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    );
    const fieldTex = new THREE.DataTexture(
      sim.fieldTexData,
      GRID_W,
      GRID_H,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    );

    const mats = createBurrowMaterials(colorTex, fieldTex);
    mats.syncPebbles(sim.pebbles);
    expect(mats.oiledWalnut.colorNode).toBeTruthy();
    expect(mats.satinBrass.colorNode).toBeTruthy();
    expect(mats.agedBronze.colorNode).toBeTruthy();
    expect(mats.beveledGlass.colorNode).toBeTruthy();
    expect(mats.beveledGlass.opacityNode).toBeTruthy();
    expect(mats.sandForeground.colorNode).toBeTruthy();
    expect(mats.sandForeground.opacityNode).toBeTruthy();
    expect(mats.sandBackwall.colorNode).toBeTruthy();
    expect(mats.sandBackwall.opacityNode).toBeTruthy();
    expect(mats.sandCrestTop.colorNode).toBeTruthy();
    expect(mats.beetleChitin.colorNode).toBeTruthy();

    mats.dispose();
    colorTex.dispose();
    fieldTex.dispose();
  });
});
