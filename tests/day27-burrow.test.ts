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

    // Exercise live 3D crest ribbon, worker ants & 3D granule sync with real sim data
    const sim = new SandSimulation(1);
    assembly.updateSandCrest(sim.surfaceY, sim.surfaceRgb);
    assembly.updateWorkerAnts(sim.diggers, 0.5);
    assembly.updateParticles3D(sim.grains3D, sim.dust3D, sim.pigmentRgbLinear);
    assertOpenSurfaceNormal('SandCrestTopRibbonAfterUpdate', assembly.sandCrestGeo, new THREE.Vector3(0, 1, 0));

    assembly.dispose();
    dummyMat.dispose();
  });

  it('enforces zero solid-wall phasing across 1,000 live colony ticks, user carvings, pours, and cave-ins', () => {
    const sim = new SandSimulation(27);
    const startMass = sim.countTotalSandMass();

    for (let t = 0; t < 1000; t++) {
      // Simulate real player carving a winding side tunnel & triggering a mid-bed cave-in!
      if (t === 120) {
        sim.excavateCapsule(160, 115, 220, 102, 5.8, true);
      }
      if (t === 280) {
        sim.pourSandAt(120, GRID_H - 12, 24);
      }
      if (t === 450) {
        sim.excavateBox(168, 66, 208, 84); // wide flat cave-in trigger!
      }

      sim.step({ enableDiggers: true, enableArchCheck: true });

      // Every single non-buried worker ant must strictly occupy CELL_EMPTY (NEVER packed sand or pebble!)
      for (let i = 0; i < sim.diggers.length; i++) {
        const a = sim.diggers[i];
        const gx = Math.max(0, Math.min(GRID_W - 1, Math.round(a.x)));
        const gy = Math.max(0, Math.min(GRID_H - 1, Math.round(a.y)));
        const ct = sim.cellType[sim.idx(gx, gy)];
        if (a.state !== 'buried') {
          expect(ct, `tick ${t} ant ${a.id} (${a.state}) must not phase into solid cell`).toBe(
            CELL_EMPTY
          );
        }
      }
    }

    expect(sim.colony.solidWallPenetrations).toBe(0);
    // Exact mass conservation holds after non-excavating/non-pouring phases; check colony stats live!
    expect(sim.diggers.length).toBe(DIGGER_COUNT);
    expect(sim.tailingsDeposited).toBeGreaterThanOrEqual(4);
    expect(startMass).toBeGreaterThan(25000);
  });

  it('routes grain-hauling ants strictly through open winding corridors rather than phasing vertically through solid sand', () => {
    const sim = new SandSimulation(77);
    sim.reset(77, false);

    // Carve an explicit L-shaped winding open tunnel:
    // Vertical shaft at x = 90 down to y = 92, then horizontal corridor right to x = 165!
    const topY = sim.topSolidY[90] || 175;
    sim.excavateCapsule(90, topY + 3, 90, 92, 5.2, false);
    sim.excavateCapsule(90, 92, 165, 92, 5.2, false);
    sim.recomputeSurfaceAndOpticalFields();
    sim.colony.rebuildNavigationFields(sim);

    // Place Ant 0 carrying a Burnt Sienna grain at the far right end (162, 92) — solid packed sand sits directly above it!
    const ant = sim.diggers[0];
    ant.x = 162;
    ant.y = 92;
    ant.state = 'haul_up';
    ant.carryingPigment = 3;
    ant.homeShaftX = 90;
    ant.stateTimer = 0;

    // Confirm thick solid impassable overburden (packed sand or pebble) exists directly above (162, 92)
    expect(sim.cellType[sim.idx(162, 120)]).not.toBe(CELL_EMPTY);
    expect(sim.cellType[sim.idx(162, 152)]).toBe(CELL_PACKED_SAND);

    let passedLeftCorner = false;
    let reachedSurface = false;

    for (let t = 0; t < 420; t++) {
      sim.step({ enableDiggers: true, enableArchCheck: false });
      const gx = Math.round(ant.x);
      const gy = Math.round(ant.y);
      expect(sim.cellType[sim.idx(gx, gy)], `tick ${t} ant must stay in open corridor`).toBe(
        CELL_EMPTY
      );
      if (ant.x < 104 && ant.y < 110) passedLeftCorner = true;
      if (ant.state === 'deposit' || ant.y >= topY - 4) {
        reachedSurface = true;
        break;
      }
    }

    expect(passedLeftCorner).toBe(true);
    expect(reachedSurface).toBe(true);
    expect(sim.colony.solidWallPenetrations).toBe(0);
  });

  it('detects sealed trapped chambers after cave-ins and initiates rescue digging while conserving sand mass', () => {
    const sim = new SandSimulation(55);
    sim.reset(55, false);

    // Carve an isolated deep oval chamber completely cut off from the surface (y = 80..92, x = 140..158)
    sim.excavateCapsule(142, 86, 156, 86, 5.2, false);
    sim.recomputeSurfaceAndOpticalFields();
    sim.colony.rebuildNavigationFields(sim);

    // Place Ant 0 strictly inside this sealed subterranean chamber
    const ant = sim.diggers[0];
    ant.x = 149;
    ant.y = 86;
    ant.state = 'patrol_down';
    ant.carryingPigment = -1;
    ant.stateTimer = 0;

    const exactMassStart = sim.countTotalSandMass();
    const initialExcavated = sim.excavatedCells;
    let enteredRescueOrBackfill = false;

    for (let t = 0; t < 180; t++) {
      sim.step({ enableDiggers: true, enableArchCheck: true });
      if (ant.state === 'rescue_dig' || ant.state === 'backfill') {
        enteredRescueOrBackfill = true;
      }
      expect(sim.countTotalSandMass(), `rescue tick ${t} exact sand mass`).toBe(exactMassStart);
    }

    expect(enteredRescueOrBackfill).toBe(true);
    expect(sim.colony.rescuesTriggered).toBeGreaterThan(0);
    expect(sim.excavatedCells).toBeGreaterThan(initialExcavated);
  });

  it('applies gravity tumble and clean landing recovery when floor support disappears beneath an ant', () => {
    const sim = new SandSimulation(61);
    sim.reset(61, false);

    // Carve a tall vertical cavern (24 cells tall) and drop Ant 0 near mid-air center
    sim.excavateBox(136, 70, 152, 108);
    sim.recomputeSurfaceAndOpticalFields();
    sim.colony.rebuildNavigationFields(sim);

    const ant = sim.diggers[0];
    ant.x = 144;
    ant.y = 95;
    ant.state = 'patrol_down';
    ant.carryingPigment = -1;

    let sawFalling = false;
    let recovered = false;

    for (let t = 0; t < 90; t++) {
      sim.step({ enableDiggers: true, enableArchCheck: false });
      if (ant.state === 'falling') sawFalling = true;
      if (sawFalling && ant.state !== 'falling') {
        recovered = true;
        break;
      }
    }

    expect(sawFalling).toBe(true);
    expect(recovered).toBe(true);
    expect(ant.y).toBeLessThan(86);
  });

  it('fans poured sand laterally off river pebbles and spawns live 3D tumbling granules and dust', () => {
    const sim = new SandSimulation(88);
    const p = sim.pebbles[1]; // central-upper river pebble
    // Excavate airspace above the pebble and pour a brisk stream right onto its crown
    sim.excavateBox(p.cx - 22, p.cy + 4, p.cx + 22, GRID_H - 6);
    sim.pourSandAt(p.cx, Math.min(GRID_H - 6, p.cy + 34), 24);

    let peak3DGrains = 0;
    let sawLateralMomentum = false;
    for (let t = 0; t < 80; t++) {
      if (t < 15) sim.pourSandAt(p.cx, Math.min(GRID_H - 6, p.cy + 34), 10);
      sim.step({ enableDiggers: false, enableArchCheck: false });
      if (sim.active3DGrains > peak3DGrains) peak3DGrains = sim.active3DGrains;
      for (let i = 0; i < sim.cellCount; i++) {
        if (sim.cellType[i] === CELL_LOOSE_SAND && Math.abs(sim.velX[i]) > 0.25) {
          sawLateralMomentum = true;
          break;
        }
      }
    }

    expect(peak3DGrains).toBeGreaterThan(10);
    expect(sawLateralMomentum).toBe(true);
  });

  it('starts with 0 trapped ants in the connected starter burrow and keeps every worker moving without corner stalls', () => {
    const sim = new SandSimulation(1);
    const bootStats = sim.colony.getStats();
    expect(bootStats.trappedCount).toBe(0);

    const startPos = sim.diggers.map((a) => ({ x: a.x, y: a.y }));
    const maxMove = new Float32Array(sim.diggers.length);

    for (let t = 0; t < 320; t++) {
      sim.step({ enableDiggers: true, enableArchCheck: true });
      for (let i = 0; i < sim.diggers.length; i++) {
        const d = Math.hypot(sim.diggers[i].x - startPos[i].x, sim.diggers[i].y - startPos[i].y);
        if (d > maxMove[i]) maxMove[i] = d;
      }
    }

    for (let i = 0; i < sim.diggers.length; i++) {
      expect(maxMove[i], `ant ${i} should travel smoothly without freezing`).toBeGreaterThan(6.0);
    }
    expect(sim.colony.solidWallPenetrations).toBe(0);
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
    expect(mats.antGaster.colorNode).toBeTruthy();
    expect(mats.antEye.colorNode).toBeTruthy();
    expect(mats.mineralGrain3D.colorNode).toBeTruthy();
    expect(mats.dustMote3D.colorNode).toBeTruthy();

    mats.dispose();
    colorTex.dispose();
    fieldTex.dispose();
  });

  it('slumps tall 90-degree vertical surface cuts along active Rankine slip wedges instead of freezing like concrete', () => {
    const sim = new SandSimulation(10);
    sim.reset(10, false);

    // Carve a steep 95-cell-tall vertical trench open to the top sky (width 14 <= ARCH_SPAN_LIMIT)
    sim.excavateBox(140, 80, 153, 185);
    for (let t = 0; t < 180; t++) {
      sim.step({ enableDiggers: false, enableArchCheck: true });
    }

    let slumpedInTrench = 0;
    for (let y = 80; y <= 176; y++) {
      for (let x = 140; x <= 153; x++) {
        if (sim.cellType[sim.idx(x, y)] !== CELL_EMPTY) {
          slumpedInTrench++;
        }
      }
    }

    expect(sim.collapseEvents).toBeGreaterThan(0);
    expect(slumpedInTrench).toBeGreaterThan(450);
  });

  it('keeps poured surface dunes permanently loose so undercutting the toe triggers a full repose cascade', () => {
    const sim = new SandSimulation(21);
    sim.reset(21, false);
    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        sim.cellType[sim.idx(x, y)] = y <= 40 ? CELL_PACKED_SAND : CELL_EMPTY;
      }
    }
    sim.recomputeSurfaceAndOpticalFields();

    // Pour a tall loose sand cone and let it sit well past 200 ticks
    for (let t = 0; t < 240; t++) {
      if (t < 150) sim.pourSandAt(150, 140, 10);
      sim.step({ enableDiggers: false, enableArchCheck: false });
    }

    // Undercut the entire left half of the rested sand cone
    for (let y = 41; y <= 95; y++) {
      for (let x = 90; x <= 146; x++) {
        sim.cellType[sim.idx(x, y)] = CELL_EMPTY;
      }
    }

    // Step 90 frames -> entire steep loose face must peel from crest down to natural repose (zero frozen cliff!)
    for (let t = 0; t < 90; t++) {
      sim.step({ enableDiggers: false, enableArchCheck: false });
    }

    let maxStep = 0;
    for (let x = 110; x <= 185; x++) {
      let h0 = 40;
      let h1 = 40;
      for (let y = 120; y > 40; y--) {
        if (sim.cellType[sim.idx(x, y)] !== CELL_EMPTY) {
          h0 = y;
          break;
        }
      }
      for (let y = 120; y > 40; y--) {
        if (sim.cellType[sim.idx(x + 1, y)] !== CELL_EMPTY) {
          h1 = y;
          break;
        }
      }
      maxStep = Math.max(maxStep, Math.abs(h1 - h0));
    }

    expect(maxStep).toBeLessThanOrEqual(2);
  });

  it('prevents airborne pour streams from tenting SandCrestTopRibbon or darkening open sky into AIR_CAVITY', () => {
    const sim = new SandSimulation(1);
    const baselineCrestY = sim.surfaceY[180];

    for (let t = 0; t < 6; t++) {
      sim.pourSandAt(180, GRID_H - 8, 12);
      sim.step({ enableDiggers: false, enableArchCheck: false });
    }
    sim.syncOpticalTextures();

    // Airborne grains still falling well above the dune must NOT pull `surfaceY` up into mid-air
    expect(Math.abs(sim.surfaceY[180] - baselineCrestY)).toBeLessThan(0.5);

    let skyTurnedToCavity = 0;
    for (let y = Math.ceil(baselineCrestY) + 5; y < GRID_H - 5; y++) {
      for (let x = 168; x <= 192; x++) {
        const idx = sim.idx(x, y);
        if (sim.cellType[idx] === CELL_EMPTY && sim.airClass[idx] === AIR_CAVITY) {
          skyTurnedToCavity++;
        }
      }
    }
    expect(skyTurnedToCavity).toBe(0);
  });
});
