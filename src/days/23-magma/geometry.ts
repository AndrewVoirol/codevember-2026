import * as THREE from 'three/webgpu';
import {
  CHAMBER_BOTTOM_Y,
  CHAMBER_HEIGHT,
  CHAMBER_ROOF_Y,
  COLUMN_BOUNDARIES_X,
  CORE_DEPTH,
  CORE_HALF_D,
  CORE_HALF_W,
  CORE_WIDTH,
  DIKE_SEAM_X,
  MAX_LOBES,
  OVERBURDEN_BOTTOM_Y,
  OVERBURDEN_HEIGHT,
  SILL_BED_Y,
  SLAB_COLUMNS,
  STRATA_TIERS,
  TIER_BOUNDARIES_Y,
  evalBeddingPlaneY,
  evalColumnSeamX,
} from './tectonicSolver';
import { type MagmaMaterials } from './materials';

export const EMBER_COUNT = 36;

export interface PressFrameAssembly {
  group: THREE.Group;
  heaveGaugeRods: THREE.Mesh[];
  chamberHitMesh: THREE.Mesh;
  dispose: () => void;
}

export interface HydraulicPistonAssembly {
  group: THREE.Group;
  ramGroup: THREE.Group;
  hitMesh: THREE.Mesh;
  dispose: () => void;
}

export interface LateralClampAssembly {
  group: THREE.Group;
  leftPlatenGroup: THREE.Group;
  leftScrewWheelGroup: THREE.Group;
  rightPlatenGroup: THREE.Group;
  rightScrewWheelGroup: THREE.Group;
  leftHitMesh: THREE.Mesh;
  rightHitMesh: THREE.Mesh;
  dispose: () => void;
}

export interface RockCoreAssembly {
  group: THREE.Group;
  slabMeshes: THREE.Mesh[];
  coreHitMesh: THREE.Mesh;
  dispose: () => void;
}

export interface MagmaSystemAssembly {
  group: THREE.Group;
  chamberMeltMesh: THREE.Mesh;
  xenolithMeshes: THREE.Mesh[];
  dikeMeshes: THREE.Mesh[];
  sillMeshes: THREE.Mesh[];
  lobeInstancedMesh: THREE.InstancedMesh;
  lobeThermalAttr: THREE.InstancedBufferAttribute;
  emberInstancedMesh: THREE.InstancedMesh;
  emberHeatAttr: THREE.InstancedBufferAttribute;
  dispose: () => void;
}

// ---------------------------------------------------------------------------
// Geometry Helpers (Strict Watertight Outward Winding)
// ---------------------------------------------------------------------------

/**
 * Creates a watertight LatheGeometry from bottom-to-top (r, y) profile points.
 * Strictly orders y ascending so Three.js generates outward-facing CCW normals
 * (signedVolume > 0, zero inside-out shells).
 */
function createClosedLathe(
  points: ReadonlyArray<readonly [number, number]>,
  segments = 24
): THREE.LatheGeometry {
  const vecs: THREE.Vector2[] = [];
  const firstY = points[0][1];
  const lastY = points[points.length - 1][1];
  if (points[0][0] > 1e-4) {
    vecs.push(new THREE.Vector2(0, firstY));
  }
  for (let i = 0; i < points.length; i++) {
    vecs.push(new THREE.Vector2(Math.max(0, points[i][0]), points[i][1]));
  }
  if (points[points.length - 1][0] > 1e-4) {
    vecs.push(new THREE.Vector2(0, lastY));
  }
  const geo = new THREE.LatheGeometry(vecs, segments);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Deterministic 3D mineral-cleavage pseudo-noise (zero polar harmonics).
 */
function rockHash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
  return s - Math.floor(s);
}

function rockSmoothNoise(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = rockHash3(ix, iy, iz);
  const n100 = rockHash3(ix + 1, iy, iz);
  const n010 = rockHash3(ix, iy + 1, iz);
  const n110 = rockHash3(ix + 1, iy + 1, iz);
  const n001 = rockHash3(ix, iy, iz + 1);
  const n101 = rockHash3(ix + 1, iy, iz + 1);
  const n011 = rockHash3(ix, iy + 1, iz + 1);
  const n111 = rockHash3(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;
  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

/**
 * Creates a chamfered rectangular steel beam/platen via ExtrudeGeometry centered at origin.
 * Guaranteed watertight outward winding.
 */
function createChamferedBlock(
  w: number,
  h: number,
  d: number,
  bevel = 0.012
): THREE.BufferGeometry {
  const b = Math.min(bevel, w * 0.2, h * 0.2, d * 0.2);
  const iw = w - b * 2;
  const ih = h - b * 2;
  const shape = new THREE.Shape();
  shape.moveTo(-iw * 0.5, -ih * 0.5);
  shape.lineTo(iw * 0.5, -ih * 0.5);
  shape.lineTo(iw * 0.5, ih * 0.5);
  shape.lineTo(-iw * 0.5, ih * 0.5);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.004, d - b * 2),
    bevelEnabled: true,
    bevelThickness: b,
    bevelSize: b,
    bevelSegments: 2,
    curveSegments: 1,
  });
  geo.center();
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a precision ACME-threaded steel rod along local +Y using bottom-to-top LatheGeometry.
 */
function createThreadedRodGeometry(
  coreRadius: number,
  crestRadius: number,
  length: number,
  threadCount: number,
  radialSegments = 20
): THREE.LatheGeometry {
  const pts: Array<[number, number]> = [];
  const y0 = -length * 0.5;
  const pitch = length / threadCount;
  pts.push([0, y0]);
  pts.push([coreRadius * 0.85, y0]);
  pts.push([coreRadius, y0 + pitch * 0.12]);

  for (let i = 0; i < threadCount; i++) {
    const base = y0 + i * pitch;
    pts.push([coreRadius, base + pitch * 0.18]);
    pts.push([crestRadius, base + pitch * 0.36]);
    pts.push([crestRadius, base + pitch * 0.64]);
    pts.push([coreRadius, base + pitch * 0.82]);
  }

  const y1 = length * 0.5;
  pts.push([coreRadius, y1 - pitch * 0.08]);
  pts.push([coreRadius * 0.85, y1]);
  pts.push([0, y1]);
  return createClosedLathe(pts, radialSegments);
}

// ---------------------------------------------------------------------------
// 1. Heavy Steel Rock-Mechanics Geology Press Frame & Quartz Cutaway Windows
// ---------------------------------------------------------------------------

export function buildPressFrameAssembly(materials: MagmaMaterials): PressFrameAssembly {
  const group = new THREE.Group();
  group.name = 'GeologyPressFrame';
  const disposables: THREE.BufferGeometry[] = [];
  const heaveGaugeRods: THREE.Mesh[] = [];

  const track = <T extends THREE.BufferGeometry>(g: T): T => {
    disposables.push(g);
    return g;
  };

  const addMesh = (
    name: string,
    geo: THREE.BufferGeometry,
    mat: THREE.Material,
    pos: [number, number, number],
    rot?: [number, number, number],
    castShadow = true,
    receiveShadow = true
  ): THREE.Mesh => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = name;
    mesh.position.set(pos[0], pos[1], pos[2]);
    if (rot) mesh.rotation.set(rot[0], rot[1], rot[2]);
    mesh.castShadow = castShadow;
    mesh.receiveShadow = receiveShadow;
    group.add(mesh);
    return mesh;
  };

  // A. Heavy Lower Base Crosshead Platen (top flush at CHAMBER_BOTTOM_Y = -0.65)
  const basePlatenH = 0.16;
  const basePlatenY = CHAMBER_BOTTOM_Y - basePlatenH * 0.5; // -0.73
  const basePlatenGeo = track(createChamferedBlock(2.80, basePlatenH, 1.34, 0.018));
  addMesh('LowerBasePlaten', basePlatenGeo, materials.forgedPressIron, [0, basePlatenY, 0]);

  // Lower brass trim collar ringing the chamber footprint cleanly
  const baseRimGeo = track(createChamferedBlock(1.74, 0.028, 1.20, 0.006));
  addMesh('ChamberBaseTrimCollar', baseRimGeo, materials.satinBrass, [0, CHAMBER_BOTTOM_Y + 0.014, 0]);

  // B. Sealed Lower Magma Chamber Steel Casing (y in [-0.65, -0.22])
  const sideWallGeo = track(createChamferedBlock(0.11, CHAMBER_HEIGHT, 1.16, 0.012));
  addMesh('ChamberLeftWall', sideWallGeo, materials.forgedPressIron, [-0.805, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0]);
  addMesh('ChamberRightWall', sideWallGeo, materials.forgedPressIron, [0.805, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0]);

  // Front (+Z) & Back (-Z) lower sill & upper lintel bars framing the wide quartz windows
  const sillBarGeo = track(createChamferedBlock(1.54, 0.072, 0.072, 0.008));
  const lintelBarGeo = track(createChamferedBlock(1.66, 0.068, 0.082, 0.008));
  const cornerPostGeo = track(createChamferedBlock(0.088, CHAMBER_HEIGHT, 0.088, 0.008));

  for (const zSign of [-1, 1] as const) {
    const zFrame = zSign * (CORE_HALF_D + 0.015);
    addMesh(
      `ChamberLowerBar_${zSign > 0 ? 'F' : 'B'}`,
      sillBarGeo,
      materials.forgedPressIron,
      [0, CHAMBER_BOTTOM_Y + 0.036, zFrame]
    );
    addMesh(
      `ChamberUpperLintel_${zSign > 0 ? 'F' : 'B'}`,
      lintelBarGeo,
      materials.forgedPressIron,
      [0, CHAMBER_ROOF_Y - 0.034, zFrame]
    );
    for (const xSign of [-1, 1] as const) {
      addMesh(
        `ChamberCornerMullion_${zSign}_${xSign}`,
        cornerPostGeo,
        materials.forgedPressIron,
        [xSign * 0.745, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, zFrame]
      );
    }
  }

  // C. Bolted Satin-Brass Quartz Window Bezels + 12 Cap Bolts + Beveled Quartz Glass Panes (Front & Back)
  const bezelHGeo = track(createChamferedBlock(1.42, 0.032, 0.034, 0.005));
  const bezelVGeo = track(createChamferedBlock(0.034, 0.305, 0.034, 0.005));
  const boltGeo = track(new THREE.CylinderGeometry(0.013, 0.013, 0.022, 6));
  const quartzGeo = track(new THREE.BoxGeometry(1.36, 0.285, 0.020));
  const windowCenterY = 0.5 * (CHAMBER_BOTTOM_Y + 0.072 + (CHAMBER_ROOF_Y - 0.068)); // -0.433

  for (const zSign of [-1, 1] as const) {
    const zBezel = zSign * (CORE_HALF_D + 0.048);
    const zGlass = zSign * (CORE_HALF_D + 0.026);
    const tag = zSign > 0 ? 'Front' : 'Back';

    addMesh(`QuartzBezelTop_${tag}`, bezelHGeo, materials.satinBrass, [0, windowCenterY + 0.148, zBezel]);
    addMesh(`QuartzBezelBot_${tag}`, bezelHGeo, materials.satinBrass, [0, windowCenterY - 0.148, zBezel]);
    addMesh(`QuartzBezelL_${tag}`, bezelVGeo, materials.satinBrass, [-0.695, windowCenterY, zBezel]);
    addMesh(`QuartzBezelR_${tag}`, bezelVGeo, materials.satinBrass, [0.695, windowCenterY, zBezel]);

    // 10 hex cap bolts seated flush on top & bottom bezel rails + 2 side bolts = 12 bolts per window
    const boltXs = [-0.62, -0.31, 0.0, 0.31, 0.62];
    for (let bi = 0; bi < boltXs.length; bi++) {
      addMesh(
        `BezelBoltTop_${tag}_${bi}`,
        boltGeo,
        materials.machinedThreadSteel,
        [boltXs[bi], windowCenterY + 0.148, zBezel + zSign * 0.014],
        [Math.PI * 0.5, 0, 0]
      );
      addMesh(
        `BezelBoltBot_${tag}_${bi}`,
        boltGeo,
        materials.machinedThreadSteel,
        [boltXs[bi], windowCenterY - 0.148, zBezel + zSign * 0.014],
        [Math.PI * 0.5, 0, 0]
      );
    }
    for (const xSide of [-0.695, 0.695]) {
      addMesh(
        `BezelBoltSide_${tag}_${xSide > 0 ? 'R' : 'L'}`,
        boltGeo,
        materials.machinedThreadSteel,
        [xSide, windowCenterY, zBezel + zSign * 0.014],
        [Math.PI * 0.5, 0, 0]
      );
    }

    // Translucent parallax laboratory quartz cutaway pane
    const glassMesh = addMesh(
      `QuartzCutawayGlass_${tag}`,
      quartzGeo,
      materials.quartzGlass,
      [0, windowCenterY, zGlass],
      undefined,
      false,
      false
    );
    glassMesh.renderOrder = 4;
  }

  // D. 4 Corner Heavy Machined Steel Tie-Rods (x = ±1.24, z = ±0.48) with Threads, Hex Nuts & Brass Collars
  //    Wide tieX = 1.24 guarantees full lateral clearance for sliding platen shoes across compression [-1, +1]
  const tieX = 1.24;
  const tieZ = 0.48;
  const rodShaftGeo = track(new THREE.CylinderGeometry(0.045, 0.045, 1.72, 20));
  const rodTopThreadGeo = track(createThreadedRodGeometry(0.042, 0.052, 0.28, 10, 18));
  const rodBotThreadGeo = track(createThreadedRodGeometry(0.042, 0.052, 0.18, 6, 18));
  const hexNutGeo = track(new THREE.CylinderGeometry(0.088, 0.088, 0.068, 6));
  const washerGeo = track(new THREE.CylinderGeometry(0.096, 0.096, 0.018, 20));
  const brassCollarGeo = track(
    createClosedLathe(
      [
        [0.046, -0.036],
        [0.080, -0.036],
        [0.086, -0.022],
        [0.086, 0.022],
        [0.080, 0.036],
        [0.046, 0.036],
      ],
      22
    )
  );

  for (const sx of [-1, 1] as const) {
    for (const sz of [-1, 1] as const) {
      const rx = sx * tieX;
      const rz = sz * tieZ;
      const id = `${sx > 0 ? 'R' : 'L'}_${sz > 0 ? 'F' : 'B'}`;

      addMesh(`TieRodShaft_${id}`, rodShaftGeo, materials.machinedThreadSteel, [rx, 0.08, rz]);
      addMesh(`TieRodTopThread_${id}`, rodTopThreadGeo, materials.machinedThreadSteel, [rx, 1.03, rz]);
      addMesh(`TieRodBotThread_${id}`, rodBotThreadGeo, materials.machinedThreadSteel, [rx, -0.84, rz]);

      // Lower nuts & brass collar seated flush on LowerBasePlaten
      addMesh(`TieRodBotWasher_${id}`, washerGeo, materials.satinBrass, [rx, -0.819, rz]);
      addMesh(`TieRodBotNut_${id}`, hexNutGeo, materials.forgedPressIron, [rx, -0.862, rz]);
      addMesh(`TieRodBaseCollar_${id}`, brassCollarGeo, materials.satinBrass, [rx, CHAMBER_BOTTOM_Y + 0.036, rz]);

      // Upper nuts & washers clamping the Open-Bridge Upper Reaction Crosshead (y in [0.94, 1.06])
      addMesh(`TieRodTopWasherLow_${id}`, washerGeo, materials.satinBrass, [rx, 0.931, rz]);
      addMesh(`TieRodTopNutLow_${id}`, hexNutGeo, materials.forgedPressIron, [rx, 0.888, rz]);
      addMesh(`TieRodTopWasherHigh_${id}`, washerGeo, materials.satinBrass, [rx, 1.069, rz]);
      addMesh(`TieRodTopNutHigh_${id}`, hexNutGeo, materials.forgedPressIron, [rx, 1.112, rz]);
    }
  }

  // E. Open-Bridge Upper Reaction Crosshead (Twin front/back beams leaving open central viewport!)
  const upperBeamGeo = track(createChamferedBlock(2.78, 0.12, 0.15, 0.015));
  addMesh('UpperBridgeFrontBeam', upperBeamGeo, materials.forgedPressIron, [0, 1.00, tieZ]);
  addMesh('UpperBridgeBackBeam', upperBeamGeo, materials.forgedPressIron, [0, 1.00, -tieZ]);

  const upperYokeGeo = track(createChamferedBlock(0.24, 0.12, 1.16, 0.014));
  addMesh('UpperBridgeLeftYoke', upperYokeGeo, materials.forgedPressIron, [-tieX, 1.00, 0]);
  addMesh('UpperBridgeRightYoke', upperYokeGeo, materials.forgedPressIron, [tieX, 1.00, 0]);

  // Diegetic satin-brass overburden heave gauge posts riding on the upper front/back beams
  const gaugeSleeveGeo = track(
    createClosedLathe(
      [
        [0.018, -0.035],
        [0.034, -0.035],
        [0.034, 0.035],
        [0.022, 0.044],
      ],
      16
    )
  );
  const gaugeRodGeo = track(new THREE.CylinderGeometry(0.012, 0.012, 0.22, 14));
  for (const sx of [-1, 1] as const) {
    addMesh(`HeaveGaugeSleeve_${sx > 0 ? 'R' : 'L'}`, gaugeSleeveGeo, materials.satinBrass, [sx * 0.56, 1.065, tieZ]);
    const rod = addMesh(
      `HeaveGaugeRod_${sx > 0 ? 'R' : 'L'}`,
      gaugeRodGeo,
      materials.machinedThreadSteel,
      [sx * 0.56, 1.04, tieZ]
    );
    heaveGaugeRods.push(rod);
  }

  // F. Left & Right Side Trunnion Brackets (clamped across front/back tie-rods at x = ±1.24, y = +0.20)
  const trunnionY = 0.20;
  const trunnionBarGeo = track(createChamferedBlock(0.18, 0.24, 1.18, 0.014));
  const trunnionBushingGeo = track(
    createClosedLathe(
      [
        [0.068, -0.120],
        [0.105, -0.120],
        [0.105, -0.088],
        [0.126, -0.076],
        [0.126, 0.076],
        [0.105, 0.088],
        [0.105, 0.120],
        [0.068, 0.120],
      ],
      24
    )
  );

  for (const sx of [-1, 1] as const) {
    const tag = sx > 0 ? 'Right' : 'Left';
    addMesh(`SideTrunnionBracket_${tag}`, trunnionBarGeo, materials.forgedPressIron, [sx * tieX, trunnionY, 0]);
    addMesh(
      `SideTrunnionBushing_${tag}`,
      trunnionBushingGeo,
      materials.satinBrass,
      [sx * tieX, trunnionY, 0],
      [0, 0, Math.PI * 0.5]
    );
  }

  // Invisible hit proxy covering the cutaway chamber for click-hold hydraulic pumping
  const chamberHitGeo = track(new THREE.BoxGeometry(1.56, CHAMBER_HEIGHT, 1.18));
  const chamberHitMesh = new THREE.Mesh(chamberHitGeo);
  chamberHitMesh.name = 'MagmaChamberHitProxy';
  chamberHitMesh.position.set(0, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0);
  chamberHitMesh.visible = false;
  group.add(chamberHitMesh);

  return {
    group,
    heaveGaugeRods,
    chamberHitMesh,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// 2. Vertical Brass & Steel Hydraulic Pressure Piston Assembly
// ---------------------------------------------------------------------------

export function buildHydraulicPistonAssembly(materials: MagmaMaterials): HydraulicPistonAssembly {
  const group = new THREE.Group();
  group.name = 'HydraulicPistonAssembly';
  const ramGroup = new THREE.Group();
  ramGroup.name = 'HydraulicPistonRamGroup';
  group.add(ramGroup);

  const disposables: THREE.BufferGeometry[] = [];
  const track = <T extends THREE.BufferGeometry>(g: T): T => {
    disposables.push(g);
    return g;
  };

  // Stationary ribbed cast-iron hydraulic cylinder barrel bolted flush beneath LowerBasePlaten (y = -0.81)
  const barrelTopY = -0.81;
  const barrelGeo = track(
    createClosedLathe(
      [
        [0.165, -0.39],
        [0.215, -0.39],
        [0.235, -0.36],
        [0.235, -0.30],
        [0.268, -0.29],
        [0.268, -0.25],
        [0.235, -0.24],
        [0.235, -0.18],
        [0.268, -0.17],
        [0.268, -0.13],
        [0.235, -0.12],
        [0.235, -0.06],
        [0.315, -0.05],
        [0.315, 0.0],
      ],
      28
    )
  );
  const barrelMesh = new THREE.Mesh(barrelGeo, materials.forgedPressIron);
  barrelMesh.name = 'HydraulicCylinderBarrel';
  barrelMesh.position.set(0, barrelTopY, 0);
  barrelMesh.castShadow = true;
  barrelMesh.receiveShadow = true;
  group.add(barrelMesh);

  // Satin-brass hydraulic packing gland collar at bottom of cylinder barrel (y = -1.20..-1.28)
  const glandGeo = track(
    createClosedLathe(
      [
        [0.122, -0.080],
        [0.180, -0.080],
        [0.195, -0.064],
        [0.195, -0.020],
        [0.220, -0.015],
        [0.220, 0.0],
      ],
      24
    )
  );
  const glandMesh = new THREE.Mesh(glandGeo, materials.satinBrass);
  glandMesh.name = 'HydraulicBrassGland';
  glandMesh.position.set(0, barrelTopY - 0.39, 0);
  glandMesh.castShadow = true;
  glandMesh.receiveShadow = true;
  group.add(glandMesh);

  // 6 hex flange bolts fastening cylinder barrel to lower base platen
  const flangeBoltGeo = track(new THREE.CylinderGeometry(0.018, 0.018, 0.032, 6));
  for (let i = 0; i < 6; i++) {
    const ang = (i / 6) * Math.PI * 2;
    const bolt = new THREE.Mesh(flangeBoltGeo, materials.machinedThreadSteel);
    bolt.name = `HydraulicFlangeBolt_${i}`;
    bolt.position.set(Math.cos(ang) * 0.275, barrelTopY - 0.062, Math.sin(ang) * 0.275);
    bolt.castShadow = true;
    group.add(bolt);
  }

  // Translating Hydraulic Ram Group (translates vertically by +pistonTravel in [0, 0.21])
  // 1. Internal dark forged-iron magma-chamber plunger plate framed by a narrow outer brass wiper ring
  //    (keeps top face dark machined steel compressing the glowing melt rather than a shiny brass floor)
  const plungerGeo = track(createChamferedBlock(1.26, 0.084, 0.88, 0.012));
  const plungerMesh = new THREE.Mesh(plungerGeo, materials.forgedPressIron);
  plungerMesh.name = 'ChamberInternalPlunger';
  plungerMesh.position.set(0, CHAMBER_BOTTOM_Y + 0.044, 0);
  ramGroup.add(plungerMesh);

  const sealRailFBGeo = track(createChamferedBlock(1.29, 0.022, 0.024, 0.004));
  const sealRailLRGeo = track(createChamferedBlock(0.024, 0.022, 0.88, 0.004));
  const sealY = CHAMBER_BOTTOM_Y + 0.044;
  for (const sz of [-1, 1] as const) {
    const rail = new THREE.Mesh(sealRailFBGeo, materials.satinBrass);
    rail.name = `ChamberPlungerBrassSeal_${sz > 0 ? 'F' : 'B'}`;
    rail.position.set(0, sealY, sz * 0.442);
    ramGroup.add(rail);
  }
  for (const sx of [-1, 1] as const) {
    const rail = new THREE.Mesh(sealRailLRGeo, materials.satinBrass);
    rail.name = `ChamberPlungerBrassSeal_${sx > 0 ? 'R' : 'L'}`;
    rail.position.set(sx * 0.635, sealY, 0);
    ramGroup.add(rail);
  }

  // 2. Machined steel vertical piston rod descending cleanly out through brass gland
  const pistonRodGeo = track(new THREE.CylinderGeometry(0.105, 0.105, 1.00, 24));
  const pistonRodMesh = new THREE.Mesh(pistonRodGeo, materials.machinedThreadSteel);
  pistonRodMesh.name = 'HydraulicPistonRod';
  pistonRodMesh.position.set(0, -1.11, 0);
  pistonRodMesh.castShadow = true;
  pistonRodMesh.receiveShadow = true;
  ramGroup.add(pistonRodMesh);

  // 3. Lower Satin-Brass Pressure Collar & Ergonomic Crossbar Pump Handle
  //    Top of collar at -1.525 rises to -1.315 at max stroke (clears -1.28 gland by 3.5cm!)
  const footCollarGeo = track(
    createClosedLathe(
      [
        [0.0, -0.10],
        [0.135, -0.10],
        [0.155, -0.075],
        [0.155, 0.055],
        [0.128, 0.075],
        [0.105, 0.075],
      ],
      24
    )
  );
  const footCollar = new THREE.Mesh(footCollarGeo, materials.satinBrass);
  footCollar.name = 'HydraulicFootBrassCollar';
  footCollar.position.set(0, -1.60, 0);
  footCollar.castShadow = true;
  ramGroup.add(footCollar);

  const crossbarGeo = track(new THREE.CylinderGeometry(0.028, 0.028, 0.62, 18));
  const crossbar = new THREE.Mesh(crossbarGeo, materials.machinedThreadSteel);
  crossbar.name = 'HydraulicPumpCrossbar';
  crossbar.position.set(0, -1.60, 0);
  crossbar.rotation.z = Math.PI * 0.5;
  crossbar.castShadow = true;
  ramGroup.add(crossbar);

  const knobGeo = track(
    createClosedLathe(
      [
        [0.0, -0.065],
        [0.036, -0.065],
        [0.048, -0.045],
        [0.048, 0.045],
        [0.034, 0.065],
        [0.0, 0.065],
      ],
      18
    )
  );
  for (const sx of [-1, 1] as const) {
    const knob = new THREE.Mesh(knobGeo, materials.satinBrass);
    knob.name = `HydraulicHandleKnob_${sx > 0 ? 'R' : 'L'}`;
    knob.position.set(sx * 0.34, -1.60, 0);
    knob.rotation.z = Math.PI * 0.5;
    knob.castShadow = true;
    ramGroup.add(knob);
  }

  // Generous invisible raycast proxy covering barrel + moving T-handle
  const hitGeo = track(new THREE.CylinderGeometry(0.38, 0.38, 1.00, 14));
  const hitMesh = new THREE.Mesh(hitGeo);
  hitMesh.name = 'HydraulicPistonHitProxy';
  hitMesh.position.set(0, -1.28, 0);
  hitMesh.visible = false;
  group.add(hitMesh);

  return {
    group,
    ramGroup,
    hitMesh,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// 3. Left & Right Lateral Threaded Tectonic Screw-Clamps & Capstan Handwheels
// ---------------------------------------------------------------------------

export function buildLateralClampAssemblies(materials: MagmaMaterials): LateralClampAssembly {
  const group = new THREE.Group();
  group.name = 'LateralTectonicClamps';

  const disposables: THREE.BufferGeometry[] = [];
  const track = <T extends THREE.BufferGeometry>(g: T): T => {
    disposables.push(g);
    return g;
  };

  const clampY = 0.20;

  // Shared geometries across both left & right screw clamps
  const platenShoeGeo = track(createChamferedBlock(0.072, 0.82, 0.86, 0.012));
  const gussetGeo = track(createChamferedBlock(0.046, 0.072, 0.74, 0.008));
  const swivelSocketGeo = track(
    createClosedLathe(
      [
        [0.0, -0.038],
        [0.098, -0.038],
        [0.098, 0.008],
        [0.080, 0.038],
        [0.0, 0.038],
      ],
      20
    )
  );
  // Longer 0.82m ACME lead-screw bridges seamlessly from platen socket through x=±1.24 trunnion to x=±1.68 capstan
  const acmeScrewGeo = track(createThreadedRodGeometry(0.043, 0.060, 0.82, 20, 20));
  const wheelHubGeo = track(
    createClosedLathe(
      [
        [0.0, -0.055],
        [0.095, -0.055],
        [0.105, -0.032],
        [0.105, 0.032],
        [0.078, 0.060],
        [0.0, 0.060],
      ],
      22
    )
  );
  const wheelRimGeo = track(new THREE.TorusGeometry(0.245, 0.028, 16, 36));
  const wheelSpokeGeo = track(new THREE.CylinderGeometry(0.017, 0.022, 0.22, 12));
  const gripHandleGeo = track(
    createClosedLathe(
      [
        [0.0, -0.052],
        [0.022, -0.052],
        [0.030, -0.018],
        [0.030, 0.032],
        [0.020, 0.052],
        [0.0, 0.052],
      ],
      14
    )
  );
  const hitBoxGeo = track(new THREE.BoxGeometry(0.96, 0.88, 1.00));

  function buildOneSide(side: -1 | 1): {
    platenGroup: THREE.Group;
    screwWheelGroup: THREE.Group;
    hitMesh: THREE.Mesh;
  } {
    const tag = side < 0 ? 'Left' : 'Right';
    const platenGroup = new THREE.Group();
    platenGroup.name = `LateralPlatenGroup_${tag}`;
    group.add(platenGroup);

    // Platen shoe rests flush against rock core flank (inner face at side * (CORE_HALF_W + 0.004))
    const shoeCenterX = side * (CORE_HALF_W + 0.040);
    const shoe = new THREE.Mesh(platenShoeGeo, materials.forgedPressIron);
    shoe.name = `LateralPlatenShoe_${tag}`;
    shoe.position.set(shoeCenterX, clampY, 0);
    shoe.castShadow = true;
    shoe.receiveShadow = true;
    platenGroup.add(shoe);

    for (const dy of [-0.26, 0.26]) {
      const rib = new THREE.Mesh(gussetGeo, materials.forgedPressIron);
      rib.name = `LateralPlatenRib_${tag}_${dy > 0 ? 'Top' : 'Bot'}`;
      rib.position.set(shoeCenterX + side * 0.040, clampY + dy, 0);
      rib.castShadow = true;
      platenGroup.add(rib);
    }

    const socket = new THREE.Mesh(swivelSocketGeo, materials.satinBrass);
    socket.name = `LateralSwivelSocket_${tag}`;
    socket.position.set(shoeCenterX + side * 0.048, clampY, 0);
    socket.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
    socket.castShadow = true;
    platenGroup.add(socket);

    // Translating & Rotating Screw + 4-Spoke Brass/Iron Capstan Handwheel
    const screwWheelGroup = new THREE.Group();
    screwWheelGroup.name = `LateralScrewWheelGroup_${tag}`;
    screwWheelGroup.position.set(0, clampY, 0);
    group.add(screwWheelGroup);

    // Horizontal ACME threaded lead-screw coaxial with x = ±1.24 trunnion bushing
    const screw = new THREE.Mesh(acmeScrewGeo, materials.machinedThreadSteel);
    screw.name = `LateralAcmeScrew_${tag}`;
    screw.position.set(side * 1.28, 0, 0);
    screw.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
    screw.castShadow = true;
    screw.receiveShadow = true;
    screwWheelGroup.add(screw);

    // Outer Capstan Handwheel at x = side * 1.68
    const wheelX = side * 1.68;
    const hub = new THREE.Mesh(wheelHubGeo, materials.satinBrass);
    hub.name = `CapstanHub_${tag}`;
    hub.position.set(wheelX, 0, 0);
    hub.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
    hub.castShadow = true;
    screwWheelGroup.add(hub);

    const rim = new THREE.Mesh(wheelRimGeo, materials.forgedPressIron);
    rim.name = `CapstanRim_${tag}`;
    rim.position.set(wheelX, 0, 0);
    rim.rotation.y = Math.PI * 0.5;
    rim.castShadow = true;
    rim.receiveShadow = true;
    screwWheelGroup.add(rim);

    for (let s = 0; s < 4; s++) {
      const ang = (s / 4) * Math.PI * 2 + Math.PI * 0.25;
      const sy = Math.cos(ang);
      const sz = Math.sin(ang);

      const spoke = new THREE.Mesh(wheelSpokeGeo, materials.satinBrass);
      spoke.name = `CapstanSpoke_${tag}_${s}`;
      spoke.position.set(wheelX, sy * 0.135, sz * 0.135);
      spoke.rotation.x = ang;
      spoke.castShadow = true;
      screwWheelGroup.add(spoke);

      const grip = new THREE.Mesh(gripHandleGeo, materials.satinBrass);
      grip.name = `CapstanGrip_${tag}_${s}`;
      grip.position.set(wheelX + side * 0.048, sy * 0.245, sz * 0.245);
      grip.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
      grip.castShadow = true;
      screwWheelGroup.add(grip);
    }

    const hitMesh = new THREE.Mesh(hitBoxGeo);
    hitMesh.name = `LateralClampHitProxy_${tag}`;
    hitMesh.position.set(side * 1.38, clampY, 0);
    hitMesh.visible = false;
    platenGroup.add(hitMesh);

    return { platenGroup, screwWheelGroup, hitMesh };
  }

  const left = buildOneSide(-1);
  const right = buildOneSide(1);

  return {
    group,
    leftPlatenGroup: left.platenGroup,
    leftScrewWheelGroup: left.screwWheelGroup,
    rightPlatenGroup: right.platenGroup,
    rightScrewWheelGroup: right.screwWheelGroup,
    leftHitMesh: left.hitMesh,
    rightHitMesh: right.hitMesh,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// 4. 12 Interlocking Stratified Basalt & Ochre-Tuff Rock Slabs
// ---------------------------------------------------------------------------

/**
 * Generates a single watertight chiseled geological rock slab whose left/right/top/bottom
 * walls follow the exact shared fracture profiles (evalColumnSeamX, evalBeddingPlaneY).
 * Strictly preserves outward CCW winding (signedVolume > 0) via smooth bijective cube mapping.
 */
function createFacetedRockSlabGeometry(tier: number, col: number, slabIndex: number): THREE.BufferGeometry {
  const geo = new THREE.BoxGeometry(1, 1, 1, 8, 6, 8);
  const posAttr = geo.attributes.position as THREE.BufferAttribute;
  const count = posAttr.count;

  const restPosArr = new Float32Array(count * 3);
  const stratumArr = new Float32Array(count);
  const fractureMaskArr = new Float32Array(count);
  const slabIdxArr = new Float32Array(count);

  const restCenterX = 0.5 * (COLUMN_BOUNDARIES_X[col] + COLUMN_BOUNDARIES_X[col + 1]);
  const restCenterY = 0.5 * (TIER_BOUNDARIES_Y[tier] + TIER_BOUNDARIES_Y[tier + 1]);

  const hasLeftFracture = col > 0;
  const hasRightFracture = col < SLAB_COLUMNS - 1;
  const hasBottomBedding = tier > 0;
  const hasTopBedding = tier < STRATA_TIERS - 1;

  for (let i = 0; i < count; i++) {
    // Unit box coordinates in [-0.5, +0.5]
    const ux = posAttr.getX(i);
    const uy = posAttr.getY(i);
    const uz = posAttr.getZ(i);

    const tx = ux + 0.5; // [0, 1] left -> right
    const ty = uy + 0.5; // [0, 1] bottom -> top

    const zBase = uz * CORE_DEPTH;

    // Initial linear estimate for boundary evaluation
    const xApprox =
      COLUMN_BOUNDARIES_X[col] + tx * (COLUMN_BOUNDARIES_X[col + 1] - COLUMN_BOUNDARIES_X[col]);
    const yBottom = evalBeddingPlaneY(tier, xApprox, zBase);
    const yTop = evalBeddingPlaneY(tier + 1, xApprox, zBase);
    let yWorld = yBottom + ty * (yTop - yBottom);

    const xLeft = evalColumnSeamX(col, yWorld, zBase);
    const xRight = evalColumnSeamX(col + 1, yWorld, zBase);
    let xWorld = xLeft + tx * (xRight - xLeft);
    let zWorld = zBase;

    // Micro-chamfer along slab seam edges so dark mineral joints read crisply and never Z-fight
    const gapInsetX = 0.0032;
    const gapInsetY = 0.0028;
    const edgeProximityX = Math.max(0, Math.abs(ux) * 2 - 0.85) / 0.15;
    const edgeProximityY = Math.max(0, Math.abs(uy) * 2 - 0.85) / 0.15;
    const edgeProximityZ = Math.max(0, Math.abs(uz) * 2 - 0.85) / 0.15;
    const cornerBevel =
      (edgeProximityX * edgeProximityY + edgeProximityY * edgeProximityZ + edgeProximityX * edgeProximityZ) *
      0.0055;

    if (hasLeftFracture && tx < 0.02) xWorld += gapInsetX;
    if (hasRightFracture && tx > 0.98) xWorld -= gapInsetX;
    if (hasBottomBedding && ty < 0.02) yWorld += gapInsetY;
    if (hasTopBedding && ty > 0.98) yWorld -= gapInsetY;

    // Subtle organic chiseled rock cleavage relief on exposed front/back/top rock faces
    const isOuterZ = Math.abs(uz) > 0.48;
    const isTopCrest = tier === STRATA_TIERS - 1 && ty > 0.96;
    const n1 = rockSmoothNoise(xWorld * 5.2 + 1.3, yWorld * 5.8 + 2.7, zWorld * 5.2 + 0.9) - 0.5;
    const n2 = rockSmoothNoise(xWorld * 12.5 + 4.1, yWorld * 11.2 + 0.6, zWorld * 12.5 + 3.3) - 0.5;
    const cleave = n1 * 0.022 + n2 * 0.009;

    // Taper relief cleanly near shared interior edges to keep neighboring slabs non-overlapping
    const interiorFade =
      (1 - 0.55 * edgeProximityX) * (1 - 0.55 * edgeProximityY);

    if (isOuterZ) {
      zWorld += Math.sign(uz) * cleave * interiorFade;
    }
    if (isTopCrest) {
      yWorld += cleave * 1.15 * interiorFade;
    }

    // Apply subtle corner bevel pulling strictly toward slab centroid (strictly preserves outward normals)
    const lx = xWorld - restCenterX;
    const ly = yWorld - restCenterY;
    const lz = zWorld;
    const shrink = 1.0 - cornerBevel;

    posAttr.setXYZ(i, lx * shrink, ly * shrink, lz * shrink);

    // Store continuous core rest-coordinates for 3D object-space petrography
    restPosArr[i * 3] = xWorld;
    restPosArr[i * 3 + 1] = yWorld;
    restPosArr[i * 3 + 2] = zWorld;
    stratumArr[i] = tier;
    slabIdxArr[i] = slabIndex;

    // Mark internal fracture faces (vertical dike seams & horizontal bedding planes) for magma glow
    const onLeftFrac = hasLeftFracture && tx < 0.08 ? 1.0 : 0.0;
    const onRightFrac = hasRightFracture && tx > 0.92 ? 1.0 : 0.0;
    const onBotBed = hasBottomBedding && ty < 0.08 ? 1.0 : 0.0;
    const onTopBed = hasTopBedding && ty > 0.92 ? 1.0 : 0.0;
    const onChamberRoof = tier === 0 && ty < 0.08 ? 0.85 : 0.0;
    fractureMaskArr[i] = Math.max(onLeftFrac, onRightFrac, onBotBed, onTopBed, onChamberRoof);
  }

  geo.setAttribute('aRestPos', new THREE.BufferAttribute(restPosArr, 3));
  geo.setAttribute('aStratum', new THREE.BufferAttribute(stratumArr, 1));
  geo.setAttribute('aFractureMask', new THREE.BufferAttribute(fractureMaskArr, 1));
  geo.setAttribute('aSlabIdx', new THREE.BufferAttribute(slabIdxArr, 1));
  geo.computeVertexNormals();
  return geo;
}

export function buildRockCoreAssembly(materials: MagmaMaterials): RockCoreAssembly {
  const group = new THREE.Group();
  group.name = 'StratifiedRockCore';
  const slabMeshes: THREE.Mesh[] = [];
  const disposables: THREE.BufferGeometry[] = [];

  for (let tier = 0; tier < STRATA_TIERS; tier++) {
    const restCenterY = 0.5 * (TIER_BOUNDARIES_Y[tier] + TIER_BOUNDARIES_Y[tier + 1]);
    for (let col = 0; col < SLAB_COLUMNS; col++) {
      const idx = tier * SLAB_COLUMNS + col;
      const restCenterX = 0.5 * (COLUMN_BOUNDARIES_X[col] + COLUMN_BOUNDARIES_X[col + 1]);
      const geo = createFacetedRockSlabGeometry(tier, col, idx);
      disposables.push(geo);

      const mesh = new THREE.Mesh(geo, materials.stratifiedRockCore);
      mesh.name = `RockSlab_T${tier}_C${col}`;
      mesh.position.set(restCenterX, restCenterY, 0);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      slabMeshes.push(mesh);
    }
  }

  // Invisible unified hit proxy over the overburden rock core for easy vertical pressure drag
  const coreHitGeo = new THREE.BoxGeometry(CORE_WIDTH + 0.06, OVERBURDEN_HEIGHT + 0.14, CORE_DEPTH + 0.06);
  disposables.push(coreHitGeo);
  const coreHitMesh = new THREE.Mesh(coreHitGeo);
  coreHitMesh.name = 'RockCoreHitProxy';
  coreHitMesh.position.set(0, OVERBURDEN_BOTTOM_Y + OVERBURDEN_HEIGHT * 0.5, 0);
  coreHitMesh.visible = false;
  group.add(coreHitMesh);

  return {
    group,
    slabMeshes,
    coreHitMesh,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// 5. Magma Chamber, Xenoliths, Vertical Dikes, Horizontal Sills & Pahoehoe Lobes
// ---------------------------------------------------------------------------

/**
 * Creates a watertight tapered vertical magma dike blade anchored at y = 0 (chamber roof)
 * extending up to y = 1 (scaled dynamically by dike.height in onTick), following the exact
 * jagged vertical fracture contour evalColumnSeamX.
 */
function createDikeWedgeGeometry(seamIdx: number): THREE.BufferGeometry {
  const geo = new THREE.BoxGeometry(1, 1, 1, 4, 18, 10);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const baseX = DIKE_SEAM_X[seamIdx];
  // Recessed cleanly inside the front/back rock faces so dikes gleam from inside opened seams
  const depthSpan = CORE_DEPTH - 0.014;

  for (let i = 0; i < pos.count; i++) {
    const ux = pos.getX(i); // [-0.5, +0.5]
    const uy = pos.getY(i) + 0.5; // [0, 1] bottom -> tip
    const uz = pos.getZ(i); // [-0.5, +0.5]

    const zWorld = uz * depthSpan;
    const yRef = OVERBURDEN_BOTTOM_Y + uy * OVERBURDEN_HEIGHT;
    // Slight natural fracture jag relative to baseX
    const seamOffset = (evalColumnSeamX(seamIdx === 0 ? 2 : seamIdx === 1 ? 1 : 3, yRef, zWorld) - baseX) * 0.85;

    // Taper thickness gently toward sharp hydraulic-fracture tip (uy -> 1) while remaining positive volume
    const tipTaper = 1.0 - 0.58 * uy * uy;
    const widthX = ux * tipTaper + seamOffset * 0.15;

    // Keep Z strictly inside CORE_HALF_D (gentle inward taper toward tip, zero proud ribbon bulge)
    const zRecessed = zWorld * (1.0 - 0.012 * uy);
    pos.setXYZ(i, widthX, uy, zRecessed);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Creates a watertight tapered horizontal laccolith sill lens centered at (0, 0, 0)
 * spanning unit width [-1, +1] in X and unit thickness [-0.5, +0.5] in Y.
 */
function createSillLensGeometry(): THREE.BufferGeometry {
  // Recessed cleanly inside the front/back bedding-plane cavity so sills gleam inside the opened strata seam
  const geo = new THREE.BoxGeometry(2, 1, CORE_DEPTH - 0.012, 20, 4, 10);
  const pos = geo.attributes.position as THREE.BufferAttribute;

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i); // [-1, +1]
    const y = pos.getY(i); // [-0.5, +0.5]
    const z = pos.getZ(i);

    // Classic lenticular laccolith profile: thickest at central feeder, smoothly pinched toward outer tips
    const r = Math.min(1, Math.abs(x));
    const lensFactor = 0.24 + 0.76 * Math.cos(r * Math.PI * 0.46);
    const zEdgeTaper = 0.86 + 0.14 * Math.cos((z / CORE_HALF_D) * Math.PI * 0.45);
    pos.setXYZ(i, x, y * lensFactor * zEdgeTaper, z);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Creates a watertight sculpted viscoplastic pahoehoe lava pillow / tongue geometry
 * using smooth 3D multi-octave noise folds (strictly zero polar star harmonics!).
 */
function createPahoehoeLobeGeometry(): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(1.0, 3);
  const pos = geo.attributes.position as THREE.BufferAttribute;

  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    let z = pos.getZ(i);

    // Flatten underside slightly so erupted lobes hug the rock crest/ledge naturally
    if (y < 0) {
      y *= 0.42;
    } else {
      y = Math.pow(y, 0.88);
    }

    // Organic viscous ropy pahoehoe billow displacement using 3D smooth noise
    const n1 = rockSmoothNoise(x * 2.1 + 1.4, y * 3.0 + 2.2, z * 2.1 + 0.7) - 0.5;
    const n2 = rockSmoothNoise(x * 4.5 + 3.8, y * 4.8 + 1.1, z * 4.5 + 5.2) - 0.5;
    const disp = 1.0 + n1 * 0.26 + n2 * 0.10;

    // Slight forward tongue elongation along +Z
    z *= 1.18;
    pos.setXYZ(i, x * disp, y * disp, z * disp);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Creates a cleaved angular dark basalt xenolith rock chunk floating inside the chamber.
 */
function createXenolithGeometry(seed: number): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(0.088, 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const n = rockSmoothNoise(x * 16 + seed, y * 16 + seed * 1.7, z * 16 + seed * 0.6);
    const sc = 0.78 + n * 0.42;
    pos.setXYZ(i, x * sc * 1.25, y * sc * 0.76, z * sc * 0.95);
  }
  geo.computeVertexNormals();
  return geo;
}

export function buildMagmaSystemAssembly(materials: MagmaMaterials): MagmaSystemAssembly {
  const group = new THREE.Group();
  group.name = 'MagmaIntrusionSystem';
  const disposables: THREE.BufferGeometry[] = [];
  const track = <T extends THREE.BufferGeometry>(g: T): T => {
    disposables.push(g);
    return g;
  };

  // 1. Churning Lower Magma Chamber Volume (behind front & back quartz windows)
  const chamberGeo = track(new THREE.BoxGeometry(1.48, CHAMBER_HEIGHT - 0.025, CORE_DEPTH - 0.02, 16, 6, 12));
  const chamberMeltMesh = new THREE.Mesh(chamberGeo, materials.chamberMagma);
  chamberMeltMesh.name = 'MagmaChamberMeltVolume';
  chamberMeltMesh.position.set(0, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0);
  group.add(chamberMeltMesh);

  // 2. 4 Drifting Dark Basalt Xenolith Roof-Clasts suspended inside the glowing chamber
  const xenolithMeshes: THREE.Mesh[] = [];
  const xenoOffsets: ReadonlyArray<readonly [number, number, number]> = [
    [-0.36, CHAMBER_ROOF_Y - 0.14, 0.24],
    [0.28, CHAMBER_ROOF_Y - 0.18, 0.18],
    [-0.08, CHAMBER_ROOF_Y - 0.22, -0.22],
    [0.42, CHAMBER_ROOF_Y - 0.13, -0.18],
  ];
  for (let i = 0; i < xenoOffsets.length; i++) {
    const xGeo = track(createXenolithGeometry(i * 4.3 + 1.2));
    const mesh = new THREE.Mesh(xGeo, materials.xenolithBasalt);
    mesh.name = `ChamberXenolith_${i}`;
    mesh.position.set(xenoOffsets[i][0], xenoOffsets[i][1], xenoOffsets[i][2]);
    group.add(mesh);
    xenolithMeshes.push(mesh);
  }

  // 3. 3 Dynamic Vertical Magma Dike Conduits (central primary + left/right flanks)
  const dikeMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < DIKE_SEAM_X.length; i++) {
    const dGeo = track(createDikeWedgeGeometry(i));
    const mesh = new THREE.Mesh(dGeo, materials.dikeMagma);
    mesh.name = `VerticalMagmaDike_${i}`;
    mesh.position.set(DIKE_SEAM_X[i], CHAMBER_ROOF_Y - 0.02, 0);
    group.add(mesh);
    dikeMeshes.push(mesh);
  }

  // 4. 2 Dynamic Horizontal Magma Sill Lenses (lower Basalt/Tuff & upper Tuff/Cap bedding planes)
  const sillMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < SILL_BED_Y.length; i++) {
    const sGeo = track(createSillLensGeometry());
    const mesh = new THREE.Mesh(sGeo, materials.sillMagma);
    mesh.name = `HorizontalMagmaSill_${i}`;
    mesh.position.set(0, SILL_BED_Y[i], 0);
    group.add(mesh);
    sillMeshes.push(mesh);
  }

  // 5. 24 Instanced Viscoplastic Cooling/Re-Cracking Pahoehoe Magma Lobes (1 draw call!)
  const lobeGeo = track(createPahoehoeLobeGeometry());
  const lobeThermalData = new Float32Array(MAX_LOBES * 4);
  const lobeThermalAttr = new THREE.InstancedBufferAttribute(lobeThermalData, 4);
  lobeGeo.setAttribute('aLobeThermal', lobeThermalAttr);

  const lobeInstancedMesh = new THREE.InstancedMesh(lobeGeo, materials.pahoehoeLobe, MAX_LOBES);
  lobeInstancedMesh.name = 'PahoehoeMagmaLobes';
  lobeInstancedMesh.castShadow = true;
  lobeInstancedMesh.receiveShadow = true;
  lobeInstancedMesh.frustumCulled = false;

  const hiddenMat = new THREE.Matrix4().makeScale(0, 0, 0);
  for (let i = 0; i < MAX_LOBES; i++) {
    lobeInstancedMesh.setMatrixAt(i, hiddenMat);
  }
  lobeInstancedMesh.instanceMatrix.needsUpdate = true;
  group.add(lobeInstancedMesh);

  // 6. 36 Restrained Rising Eruptive Volcanic Ash / Ember Sparks
  const emberGeo = track(new THREE.OctahedronGeometry(0.011, 0));
  const emberHeatData = new Float32Array(EMBER_COUNT);
  const emberHeatAttr = new THREE.InstancedBufferAttribute(emberHeatData, 1);
  emberGeo.setAttribute('aEmberHeat', emberHeatAttr);

  const emberInstancedMesh = new THREE.InstancedMesh(emberGeo, materials.emberSpark, EMBER_COUNT);
  emberInstancedMesh.name = 'EruptiveEmberSparks';
  emberInstancedMesh.frustumCulled = false;
  for (let i = 0; i < EMBER_COUNT; i++) {
    emberInstancedMesh.setMatrixAt(i, hiddenMat);
  }
  emberInstancedMesh.instanceMatrix.needsUpdate = true;
  group.add(emberInstancedMesh);

  return {
    group,
    chamberMeltMesh,
    xenolithMeshes,
    dikeMeshes,
    sillMeshes,
    lobeInstancedMesh,
    lobeThermalAttr,
    emberInstancedMesh,
    emberHeatAttr,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}
