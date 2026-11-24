import * as THREE from 'three/webgpu';
import {
  ACME_THREAD_PITCH,
  CHAMBER_BOTTOM_Y,
  CHAMBER_HEIGHT,
  CHAMBER_ROOF_Y,
  COLUMN_BOUNDARIES_X,
  CORE_DEPTH,
  CORE_HALF_D,
  CORE_HALF_W,
  CORE_WIDTH,
  DIKE_SEAM_X,
  EMBER_COUNT,
  MAX_LOBES,
  OVERBURDEN_BOTTOM_Y,
  OVERBURDEN_HEIGHT,
  ROCK_CHIP_COUNT,
  SILL_BED_Y,
  SLAB_COLUMNS,
  STRATA_TIERS,
  TIER_BOUNDARIES_Y,
  evalBeddingPlaneY,
  evalColumnSeamX,
  type RockSlabState,
} from './tectonicSolver';
import { type MagmaMaterials } from './materials';

export { EMBER_COUNT, ROCK_CHIP_COUNT };

export interface PressFrameAssembly {
  group: THREE.Group;
  heaveGaugeRods: THREE.Object3D[];
  gaugeNeedleGroup: THREE.Group;
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
  chamberFeederMesh: THREE.Mesh;
  xenolithMeshes: THREE.Mesh[];
  dikeMeshes: THREE.Mesh[];
  sillMeshes: THREE.Mesh[];
  lobeInstancedMesh: THREE.InstancedMesh;
  lobeThermalAttr: THREE.InstancedBufferAttribute;
  rockChipInstancedMesh: THREE.InstancedMesh;
  rockChipToneAttr: THREE.InstancedBufferAttribute;
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
 * Evaluates trapezoidal ACME thread radius profile for helical phase in [0, 1).
 */
export function evalAcmeThreadRadius(
  coreRadius: number,
  crestRadius: number,
  phaseRaw: number
): number {
  const phase = phaseRaw - Math.floor(phaseRaw);
  if (phase < 0.16) return coreRadius;
  if (phase < 0.34) {
    const t = (phase - 0.16) / 0.18;
    return coreRadius + (crestRadius - coreRadius) * t;
  }
  if (phase < 0.64) return crestRadius;
  if (phase < 0.82) {
    const t = (phase - 0.64) / 0.18;
    return crestRadius - (crestRadius - coreRadius) * t;
  }
  return coreRadius;
}

/**
 * Builds a true watertight right-handed helical ACME trapezoidal threaded lead-screw
 * along local +Y with exact pitch matching `ACME_THREAD_PITCH` (0.048 m/rev).
 * Because the thread is genuinely helical (`phase = y / pitch + theta / (2*PI)`),
 * rotating `screwWheelGroup.rotation.x` in lockstep with `platenOffset` visibly drives
 * the spiral crests smoothly through the stationary brass trunnion bushings with zero slip!
 */
export function createHelicalAcmeScrewGeometry(
  coreRadius: number,
  crestRadius: number,
  length: number,
  pitch = ACME_THREAD_PITCH,
  radialSegments = 24,
  axialRingsPerPitch = 10
): THREE.BufferGeometry {
  const turns = length / pitch;
  const ringsY = Math.max(16, Math.round(turns * axialRingsPerPitch));
  const cols = radialSegments + 1;

  // Side vertices + bottom cap center/ring + top cap center/ring
  const sideVertCount = (ringsY + 1) * cols;
  const capVertCount = 2 * (1 + radialSegments);
  const positions = new Float32Array((sideVertCount + capVertCount) * 3);
  const indices: number[] = [];

  const halfL = length * 0.5;
  const endFade = Math.min(pitch * 0.65, length * 0.08);

  // 1. Helical side wall (x = r*cos(theta), z = -r*sin(theta) -> right-handed CCW outward normals)
  for (let iy = 0; iy <= ringsY; iy++) {
    const v = iy / ringsY;
    const y = -halfL + v * length;
    const distFromEnd = Math.min(y + halfL, halfL - y);
    const threadEnv = Math.min(1, Math.max(0, (distFromEnd - 0.008) / endFade));

    for (let it = 0; it <= radialSegments; it++) {
      const uTheta = it / radialSegments;
      const theta = uTheta * Math.PI * 2;
      const helicalPhase = y / pitch + uTheta;
      const rThread = evalAcmeThreadRadius(coreRadius, crestRadius, helicalPhase);
      const r = coreRadius + (rThread - coreRadius) * threadEnv;

      const idx = (iy * cols + it) * 3;
      positions[idx] = r * Math.cos(theta);
      positions[idx + 1] = y;
      positions[idx + 2] = -r * Math.sin(theta);
    }
  }

  for (let iy = 0; iy < ringsY; iy++) {
    for (let it = 0; it < radialSegments; it++) {
      const a = iy * cols + it;
      const b = a + 1;
      const c = (iy + 1) * cols + it;
      const d = c + 1;
      // Outward CCW winding: (a -> b -> d) and (a -> d -> c)
      indices.push(a, b, d);
      indices.push(a, d, c);
    }
  }

  // 2. Watertight bottom cap (y = -halfL, normal pointing -Y)
  const botCenterIdx = sideVertCount;
  positions[botCenterIdx * 3] = 0;
  positions[botCenterIdx * 3 + 1] = -halfL;
  positions[botCenterIdx * 3 + 2] = 0;
  const botRingStart = botCenterIdx + 1;
  for (let it = 0; it < radialSegments; it++) {
    const theta = (it / radialSegments) * Math.PI * 2;
    const vi = (botRingStart + it) * 3;
    positions[vi] = coreRadius * Math.cos(theta);
    positions[vi + 1] = -halfL;
    positions[vi + 2] = -coreRadius * Math.sin(theta);
  }
  for (let it = 0; it < radialSegments; it++) {
    const next = (it + 1) % radialSegments;
    // Looking from -Y: (center, next, cur) yields outward -Y normal
    indices.push(botCenterIdx, botRingStart + next, botRingStart + it);
  }

  // 3. Watertight top cap (y = +halfL, normal pointing +Y)
  const topCenterIdx = botRingStart + radialSegments;
  positions[topCenterIdx * 3] = 0;
  positions[topCenterIdx * 3 + 1] = halfL;
  positions[topCenterIdx * 3 + 2] = 0;
  const topRingStart = topCenterIdx + 1;
  for (let it = 0; it < radialSegments; it++) {
    const theta = (it / radialSegments) * Math.PI * 2;
    const vi = (topRingStart + it) * 3;
    positions[vi] = coreRadius * Math.cos(theta);
    positions[vi + 1] = halfL;
    positions[vi + 2] = -coreRadius * Math.sin(theta);
  }
  for (let it = 0; it < radialSegments; it++) {
    const next = (it + 1) % radialSegments;
    // Looking from +Y: (center, cur, next) yields outward +Y normal
    indices.push(topCenterIdx, topRingStart + it, topRingStart + next);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

// ---------------------------------------------------------------------------
// 1. Heavy Steel Rock-Mechanics Geology Press Frame, Quartz Windows & Bourdon Gauge
// ---------------------------------------------------------------------------

export function buildPressFrameAssembly(materials: MagmaMaterials): PressFrameAssembly {
  const group = new THREE.Group();
  group.name = 'GeologyPressFrame';
  const disposables: THREE.BufferGeometry[] = [];
  const heaveGaugeRods: THREE.Object3D[] = [];

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
    receiveShadow = true,
    parent: THREE.Object3D = group
  ): THREE.Mesh => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = name;
    mesh.position.set(pos[0], pos[1], pos[2]);
    if (rot) mesh.rotation.set(rot[0], rot[1], rot[2]);
    mesh.castShadow = castShadow;
    mesh.receiveShadow = receiveShadow;
    parent.add(mesh);
    return mesh;
  };

  // A. Heavy Planed-Steel Lower Base Crosshead Platen (top flush at CHAMBER_BOTTOM_Y = -0.65)
  const basePlatenH = 0.16;
  const basePlatenY = CHAMBER_BOTTOM_Y - basePlatenH * 0.5; // -0.73
  const basePlatenGeo = track(createChamferedBlock(2.84, basePlatenH, 1.38, 0.018));
  addMesh('LowerBasePlaten', basePlatenGeo, materials.forgedPressIron, [0, basePlatenY, 0]);

  // Machined front/back longitudinal stiffening ribs & tie-rod boss pads on top of LowerBasePlaten
  const deckRailGeo = track(createChamferedBlock(2.74, 0.016, 0.044, 0.004));
  addMesh('BaseDeckRail_F', deckRailGeo, materials.forgedPressIron, [0, CHAMBER_BOTTOM_Y + 0.008, 0.635]);
  addMesh('BaseDeckRail_B', deckRailGeo, materials.forgedPressIron, [0, CHAMBER_BOTTOM_Y + 0.008, -0.635]);

  const baseBossPadGeo = track(new THREE.CylinderGeometry(0.128, 0.138, 0.014, 22));
  for (const sx of [-1, 1] as const) {
    for (const sz of [-1, 1] as const) {
      addMesh(
        `BaseTieRodBoss_${sx > 0 ? 'R' : 'L'}_${sz > 0 ? 'F' : 'B'}`,
        baseBossPadGeo,
        materials.forgedPressIron,
        [sx * 1.24, CHAMBER_BOTTOM_Y + 0.007, sz * 0.48]
      );
    }
  }

  // Non-coplanar perimeter brass mounting flange rails + 8 hex anchor bolts seating the chamber onto the bed
  const flangeFBGeo = track(createChamferedBlock(1.792, 0.026, 0.042, 0.005));
  const flangeLRGeo = track(createChamferedBlock(0.042, 0.026, 1.140, 0.005));
  const anchorBoltGeo = track(new THREE.CylinderGeometry(0.016, 0.016, 0.022, 6));
  const flangeY = CHAMBER_BOTTOM_Y + 0.014;
  for (const sz of [-1, 1] as const) {
    addMesh(`ChamberBaseFlange_${sz > 0 ? 'F' : 'B'}`, flangeFBGeo, materials.satinBrass, [0, flangeY, sz * 0.592]);
    for (const bx of [-0.66, -0.22, 0.22, 0.66]) {
      addMesh(
        `ChamberAnchorBolt_${sz > 0 ? 'F' : 'B'}_${bx}`,
        anchorBoltGeo,
        materials.machinedThreadSteel,
        [bx, flangeY + 0.016, sz * 0.602]
      );
    }
  }
  for (const sx of [-1, 1] as const) {
    addMesh(`ChamberBaseFlange_${sx > 0 ? 'R' : 'L'}`, flangeLRGeo, materials.satinBrass, [sx * 0.875, flangeY, 0]);
  }

  // B. Sealed Lower Magma Chamber Steel Casing (y in [-0.65, -0.22], strictly non-coplanar joins)
  const sideWallGeo = track(createChamferedBlock(0.11, CHAMBER_HEIGHT - 0.008, 1.04, 0.010));
  addMesh('ChamberLeftWall', sideWallGeo, materials.forgedPressIron, [-0.805, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0]);
  addMesh('ChamberRightWall', sideWallGeo, materials.forgedPressIron, [0.805, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0]);

  const sillBarGeo = track(createChamferedBlock(1.40, 0.070, 0.074, 0.007));
  const lintelBarGeo = track(createChamferedBlock(1.68, 0.064, 0.088, 0.008));
  const cornerPostGeo = track(createChamferedBlock(0.092, CHAMBER_HEIGHT - 0.002, 0.090, 0.008));

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
      [0, CHAMBER_ROOF_Y - 0.032, zFrame]
    );
    for (const xSign of [-1, 1] as const) {
      addMesh(
        `ChamberCornerMullion_${zSign}_${xSign}`,
        cornerPostGeo,
        materials.forgedPressIron,
        [xSign * 0.746, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, zFrame]
      );
    }
  }

  // C. Watertight Rabbeted Satin-Brass Quartz Window Bezels + 12 Cap Bolts + Quartz Panes (Front & Back)
  //    Vertical bezel rails butt cleanly between top/bottom rails (ZERO coplanar corner z-fighting!)
  const bezelHGeo = track(createChamferedBlock(1.43, 0.030, 0.034, 0.005));
  const bezelVGeo = track(createChamferedBlock(0.034, 0.264, 0.034, 0.005));
  const rabbetHGeo = track(createChamferedBlock(1.39, 0.016, 0.016, 0.002));
  const rabbetVGeo = track(createChamferedBlock(0.016, 0.278, 0.016, 0.002));
  const boltGeo = track(new THREE.CylinderGeometry(0.013, 0.013, 0.022, 6));
  const quartzGeo = track(new THREE.BoxGeometry(1.375, 0.284, 0.018));
  const windowCenterY = 0.5 * (CHAMBER_BOTTOM_Y + 0.070 + (CHAMBER_ROOF_Y - 0.064)); // -0.432

  for (const zSign of [-1, 1] as const) {
    const zBezel = zSign * (CORE_HALF_D + 0.050);
    const zGlass = zSign * (CORE_HALF_D + 0.028);
    const zRabbet = zSign * (CORE_HALF_D + 0.015);
    const tag = zSign > 0 ? 'Front' : 'Back';

    // Dark forged-iron inner rabbet gasket frame seating the quartz pane watertight
    addMesh(`QuartzRabbetTop_${tag}`, rabbetHGeo, materials.forgedPressIron, [0, windowCenterY + 0.144, zRabbet]);
    addMesh(`QuartzRabbetBot_${tag}`, rabbetHGeo, materials.forgedPressIron, [0, windowCenterY - 0.144, zRabbet]);
    addMesh(`QuartzRabbetL_${tag}`, rabbetVGeo, materials.forgedPressIron, [-0.688, windowCenterY, zRabbet]);
    addMesh(`QuartzRabbetR_${tag}`, rabbetVGeo, materials.forgedPressIron, [0.688, windowCenterY, zRabbet]);

    // Non-overlapping outer satin-brass compression bezel rails
    addMesh(`QuartzBezelTop_${tag}`, bezelHGeo, materials.satinBrass, [0, windowCenterY + 0.148, zBezel]);
    addMesh(`QuartzBezelBot_${tag}`, bezelHGeo, materials.satinBrass, [0, windowCenterY - 0.148, zBezel]);
    addMesh(`QuartzBezelL_${tag}`, bezelVGeo, materials.satinBrass, [-0.698, windowCenterY, zBezel]);
    addMesh(`QuartzBezelR_${tag}`, bezelVGeo, materials.satinBrass, [0.698, windowCenterY, zBezel]);

    // 12 machined hex cap bolts per quartz window
    const boltXs = [-0.62, -0.31, 0.0, 0.31, 0.62];
    for (let bi = 0; bi < boltXs.length; bi++) {
      addMesh(
        `BezelBoltTop_${tag}_${bi}`,
        boltGeo,
        materials.machinedThreadSteel,
        [boltXs[bi], windowCenterY + 0.148, zBezel + zSign * 0.015],
        [Math.PI * 0.5, 0, 0]
      );
      addMesh(
        `BezelBoltBot_${tag}_${bi}`,
        boltGeo,
        materials.machinedThreadSteel,
        [boltXs[bi], windowCenterY - 0.148, zBezel + zSign * 0.015],
        [Math.PI * 0.5, 0, 0]
      );
    }
    for (const xSide of [-0.698, 0.698]) {
      addMesh(
        `BezelBoltSide_${tag}_${xSide > 0 ? 'R' : 'L'}`,
        boltGeo,
        materials.machinedThreadSteel,
        [xSide, windowCenterY, zBezel + zSign * 0.015],
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

  // D. 4 Corner Heavy Machined Steel Tie-Rods (x = ±1.24, z = ±0.48) with Helical Threads, Hex Nuts & Brass Collars
  const tieX = 1.24;
  const tieZ = 0.48;
  const rodShaftGeo = track(new THREE.CylinderGeometry(0.045, 0.045, 1.72, 22));
  const rodTopThreadGeo = track(createHelicalAcmeScrewGeometry(0.042, 0.052, 0.30, 0.028, 20, 8));
  const rodBotThreadGeo = track(createHelicalAcmeScrewGeometry(0.042, 0.052, 0.19, 0.028, 20, 8));
  const hexNutGeo = track(new THREE.CylinderGeometry(0.088, 0.088, 0.068, 6));
  const washerGeo = track(new THREE.CylinderGeometry(0.098, 0.098, 0.018, 22));
  const brassCollarGeo = track(
    createClosedLathe(
      [
        [0.046, -0.036],
        [0.082, -0.036],
        [0.090, -0.022],
        [0.090, 0.022],
        [0.082, 0.036],
        [0.046, 0.036],
      ],
      24
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

      // Lower nuts & brass collar seated flush on BaseTieRodBoss
      addMesh(`TieRodBotWasher_${id}`, washerGeo, materials.satinBrass, [rx, -0.819, rz]);
      addMesh(`TieRodBotNut_${id}`, hexNutGeo, materials.forgedPressIron, [rx, -0.862, rz]);
      addMesh(`TieRodBaseCollar_${id}`, brassCollarGeo, materials.satinBrass, [rx, CHAMBER_BOTTOM_Y + 0.048, rz]);

      // Upper nuts & washers clamping the stepped Upper Reaction Yokes (y in [0.934, 1.066])
      addMesh(`TieRodTopWasherLow_${id}`, washerGeo, materials.satinBrass, [rx, 0.925, rz]);
      addMesh(`TieRodTopNutLow_${id}`, hexNutGeo, materials.forgedPressIron, [rx, 0.882, rz]);
      addMesh(`TieRodTopWasherHigh_${id}`, washerGeo, materials.satinBrass, [rx, 1.075, rz]);
      addMesh(`TieRodTopNutHigh_${id}`, hexNutGeo, materials.forgedPressIron, [rx, 1.118, rz]);
    }
  }

  // E. Open-Bridge Upper Reaction Crosshead (Stepped end yokes embrace longitudinal beams cleanly — ZERO z-fighting!)
  const upperBeamGeo = track(createChamferedBlock(2.64, 0.112, 0.148, 0.014));
  addMesh('UpperBridgeFrontBeam', upperBeamGeo, materials.forgedPressIron, [0, 1.00, tieZ]);
  addMesh('UpperBridgeBackBeam', upperBeamGeo, materials.forgedPressIron, [0, 1.00, -tieZ]);

  const upperYokeGeo = track(createChamferedBlock(0.25, 0.132, 1.18, 0.015));
  addMesh('UpperBridgeLeftYoke', upperYokeGeo, materials.forgedPressIron, [-tieX, 1.00, 0]);
  addMesh('UpperBridgeRightYoke', upperYokeGeo, materials.forgedPressIron, [tieX, 1.00, 0]);

  // 4 hex cross-bolts securing upper yokes to longitudinal bridge beams
  const yokeBoltGeo = track(new THREE.CylinderGeometry(0.019, 0.019, 0.026, 6));
  for (const sx of [-1, 1] as const) {
    for (const sz of [-1, 1] as const) {
      addMesh(
        `UpperYokeBolt_${sx > 0 ? 'R' : 'L'}_${sz > 0 ? 'F' : 'B'}`,
        yokeBoltGeo,
        materials.machinedThreadSteel,
        [sx * 1.10, 1.00, sz * 0.562],
        [Math.PI * 0.5, 0, 0]
      );
    }
  }

  // E2. Mechanical Overburden Heave Follower Gauges (Cantilevered from upper front beam directly onto rock crown!)
  //     Replaces floating pins with real chamfered brass contact pads riding on the upper basalt slabs
  const followerArmGeo = track(createChamferedBlock(0.072, 0.062, 0.25, 0.008));
  const gaugeSleeveGeo = track(
    createClosedLathe(
      [
        [0.016, -0.055],
        [0.032, -0.055],
        [0.036, -0.040],
        [0.036, 0.040],
        [0.026, 0.052],
        [0.016, 0.052],
      ],
      18
    )
  );
  const followerShaftGeo = track(new THREE.CylinderGeometry(0.011, 0.011, 0.48, 14));
  const followerFootGeo = track(
    createClosedLathe(
      [
        [0.0, 0.0],
        [0.036, 0.0],
        [0.036, 0.012],
        [0.022, 0.024],
        [0.011, 0.024],
      ],
      18
    )
  );
  const followerFinialGeo = track(
    createClosedLathe(
      [
        [0.011, -0.010],
        [0.020, -0.006],
        [0.020, 0.012],
        [0.010, 0.020],
        [0.0, 0.020],
      ],
      14
    )
  );

  for (const sx of [-1, 1] as const) {
    const tag = sx > 0 ? 'R' : 'L';
    const gx = sx * 0.25;
    const gz = 0.28; // Rides squarely atop inner basalt cap slabs (T2_C1 & T2_C2), clear of central crest lobes

    // Cantilevered steel bracket joining front upper beam (z=0.48) to follower guide sleeve (z=0.28)
    addMesh(`HeaveFollowerArm_${tag}`, followerArmGeo, materials.forgedPressIron, [gx, 1.00, 0.375]);
    addMesh(`HeaveGaugeSleeve_${tag}`, gaugeSleeveGeo, materials.satinBrass, [gx, 1.00, gz]);

    // Translating follower rod + rock-contact foot + upper graduated brass finial
    // Positioned at y = slabTopY each tick so foot bottom (local y = 0) rests flush on basalt!
    const followerGroup = new THREE.Group();
    followerGroup.name = `HeaveGaugeFollower_${tag}`;
    followerGroup.position.set(gx, 0.65, gz);
    group.add(followerGroup);

    addMesh(`HeaveGaugeFoot_${tag}`, followerFootGeo, materials.satinBrass, [0, 0, 0], undefined, true, true, followerGroup);
    addMesh(`HeaveGaugeRod_${tag}`, followerShaftGeo, materials.machinedThreadSteel, [0, 0.26, 0], undefined, true, true, followerGroup);
    addMesh(`HeaveGaugeFinial_${tag}`, followerFinialGeo, materials.satinBrass, [0, 0.505, 0], undefined, true, true, followerGroup);

    heaveGaugeRods.push(followerGroup);
  }

  // F. Left & Right Side Trunnion Brackets (Tall 0.29m forged blocks with split-clamp bosses — ZERO bushing bleed!)
  const trunnionY = 0.20;
  const trunnionBarGeo = track(createChamferedBlock(0.18, 0.290, 1.22, 0.014));
  const trunnionClampBossGeo = track(new THREE.CylinderGeometry(0.084, 0.084, 0.294, 20));
  const trunnionBoltGeo = track(new THREE.CylinderGeometry(0.017, 0.017, 0.026, 6));
  const trunnionBushingGeo = track(
    createClosedLathe(
      [
        [0.068, -0.122],
        [0.104, -0.122],
        [0.104, -0.090],
        [0.122, -0.078],
        [0.122, 0.078],
        [0.104, 0.090],
        [0.104, 0.122],
        [0.068, 0.122],
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
    for (const sz of [-1, 1] as const) {
      addMesh(
        `TrunnionTieBoss_${tag}_${sz > 0 ? 'F' : 'B'}`,
        trunnionClampBossGeo,
        materials.forgedPressIron,
        [sx * tieX, trunnionY, sz * tieZ]
      );
      for (const dy of [-0.075, 0.075]) {
        addMesh(
          `TrunnionLockBolt_${tag}_${sz > 0 ? 'F' : 'B'}_${dy > 0 ? 'T' : 'B'}`,
          trunnionBoltGeo,
          materials.machinedThreadSteel,
          [sx * (tieX + 0.098), trunnionY + dy, sz * (tieZ + 0.02)],
          [0, 0, Math.PI * 0.5]
        );
      }
    }
  }

  // G. Diegetic Machined Satin-Brass Bourdon Pressure Gauge (Mounted on Right Lower Chamber Manifold)
  //    Clears right platen shoe (y >= -0.215), base platen (y <= -0.65), and front-right tie-rod (x=1.24, z=0.48)
  const gaugeGroup = new THREE.Group();
  gaugeGroup.name = 'BourdonPressureGaugeAssembly';
  gaugeGroup.position.set(0.96, -0.42, 0.34);
  gaugeGroup.rotation.set(0, 0.32, 0);
  group.add(gaugeGroup);

  // Brass manifold pipe elbow joining ChamberRightWall (x=0.86) cleanly into Bourdon stem
  const manifoldUnionGeo = track(new THREE.CylinderGeometry(0.026, 0.026, 0.14, 16));
  const manifoldNutGeo = track(new THREE.CylinderGeometry(0.036, 0.036, 0.026, 6));
  const gaugeStemGeo = track(new THREE.CylinderGeometry(0.017, 0.020, 0.12, 16));
  addMesh('GaugeManifoldPipe', manifoldUnionGeo, materials.satinBrass, [0.90, -0.51, 0.34], [0, 0, Math.PI * 0.5]);
  addMesh('GaugeManifoldHexNut', manifoldNutGeo, materials.satinBrass, [0.87, -0.51, 0.34], [0, 0, Math.PI * 0.5]);
  addMesh('GaugeVerticalStem', gaugeStemGeo, materials.satinBrass, [0.96, -0.49, 0.34]);

  // Heavy turned brass Bourdon gauge housing & beveled front bezel ring (facing local +Z)
  const gaugeCaseGeo = track(
    createClosedLathe(
      [
        [0.0, -0.026],
        [0.082, -0.026],
        [0.092, -0.018],
        [0.092, 0.014],
        [0.088, 0.022],
        [0.074, 0.022],
        [0.074, 0.008],
        [0.0, 0.008],
      ],
      28
    )
  );
  addMesh('BourdonGaugeCase', gaugeCaseGeo, materials.satinBrass, [0, 0, 0], [Math.PI * 0.5, 0, 0], true, true, gaugeGroup);

  // Mineral-enamel calibrated dial disk (closed solid cylinder -> guaranteed signedVol > 0)
  const dialFaceGeo = track(new THREE.CylinderGeometry(0.073, 0.073, 0.006, 28));
  const dialMat = materials.gaugeDial ?? materials.satinBrass;
  addMesh('BourdonGaugeDial', dialFaceGeo, dialMat, [0, 0, 0.011], [Math.PI * 0.5, 0, 0], false, true, gaugeGroup);

  // Pivoting blued-iron Bourdon pressure needle (rotated around local Z in main.ts!)
  const gaugeNeedleGroup = new THREE.Group();
  gaugeNeedleGroup.name = 'BourdonGaugeNeedleGroup';
  gaugeNeedleGroup.position.set(0, 0, 0.017);
  gaugeGroup.add(gaugeNeedleGroup);

  const needleBladeGeo = track(createChamferedBlock(0.007, 0.068, 0.003, 0.001));
  addMesh('BourdonNeedleBlade', needleBladeGeo, materials.forgedPressIron, [0, 0.020, 0], undefined, false, false, gaugeNeedleGroup);
  const needleHubGeo = track(new THREE.CylinderGeometry(0.011, 0.012, 0.007, 14));
  addMesh('BourdonNeedleHub', needleHubGeo, materials.satinBrass, [0, 0, 0.002], [Math.PI * 0.5, 0, 0], false, false, gaugeNeedleGroup);

  // Protective quartz gauge crystal lens seated cleanly inside brass bezel
  const gaugeLensGeo = track(new THREE.CylinderGeometry(0.074, 0.074, 0.005, 24));
  const gaugeLensMesh = addMesh(
    'BourdonGaugeCrystal',
    gaugeLensGeo,
    materials.quartzGlass,
    [0, 0, 0.022],
    [Math.PI * 0.5, 0, 0],
    false,
    false,
    gaugeGroup
  );
  gaugeLensMesh.renderOrder = 5;

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
    gaugeNeedleGroup,
    chamberHitMesh,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// 2. Vertical Brass & Steel Hydraulic Pressure Piston Assembly (Full-Skirted Plunger!)
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

  // Translating Hydraulic Ram Group (translates vertically by +pistonTravel in [0, 0.155])
  // 1. Full-depth, full-skirted forged-iron chamber plunger (width 1.44, depth 1.05, height 0.24)
  //    Top crown at -0.562 rises to -0.407 at max stroke; solid skirt extends down to -0.802 so
  //    even at 100% stroke (-0.647) it stays seated inside the base bore with ZERO floating underside void!
  const plungerH = 0.24;
  const plungerTopY = CHAMBER_BOTTOM_Y + 0.088; // -0.562
  const plungerGeo = track(createChamferedBlock(1.42, plungerH, 1.03, 0.012));
  const plungerMesh = new THREE.Mesh(plungerGeo, materials.forgedPressIron);
  plungerMesh.name = 'ChamberInternalPlunger';
  plungerMesh.position.set(0, plungerTopY - plungerH * 0.5, 0); // -0.682
  plungerMesh.receiveShadow = true;
  ramGroup.add(plungerMesh);

  // Snug satin-brass perimeter wiper seal band riding flush behind the quartz cutaway windows
  // Top of seal at -0.581 keeps plunger crown proud by 19 mm (> 15 mm clearance contract)
  const sealRailFBGeo = track(createChamferedBlock(1.44, 0.022, 0.022, 0.004));
  const sealRailLRGeo = track(createChamferedBlock(0.022, 0.022, 1.03, 0.004));
  const sealY = plungerTopY - 0.030; // -0.592
  for (const sz of [-1, 1] as const) {
    const rail = new THREE.Mesh(sealRailFBGeo, materials.satinBrass);
    rail.name = `ChamberPlungerBrassSeal_${sz > 0 ? 'F' : 'B'}`;
    rail.position.set(0, sealY, sz * 0.520);
    ramGroup.add(rail);
  }
  for (const sx of [-1, 1] as const) {
    const rail = new THREE.Mesh(sealRailLRGeo, materials.satinBrass);
    rail.name = `ChamberPlungerBrassSeal_${sx > 0 ? 'R' : 'L'}`;
    rail.position.set(sx * 0.715, sealY, 0);
    ramGroup.add(rail);
  }

  // Secondary lower steel scraper ring visible on the plunger skirt as it rises
  const lowerRingFBGeo = track(createChamferedBlock(1.43, 0.014, 0.016, 0.003));
  for (const sz of [-1, 1] as const) {
    const ring = new THREE.Mesh(lowerRingFBGeo, materials.machinedThreadSteel);
    ring.name = `ChamberPlungerScraper_${sz > 0 ? 'F' : 'B'}`;
    ring.position.set(0, sealY - 0.055, sz * 0.518);
    ramGroup.add(ring);
  }

  // 2. Machined steel vertical piston rod descending cleanly out through brass gland
  const pistonRodGeo = track(new THREE.CylinderGeometry(0.105, 0.105, 0.96, 24));
  const pistonRodMesh = new THREE.Mesh(pistonRodGeo, materials.machinedThreadSteel);
  pistonRodMesh.name = 'HydraulicPistonRod';
  pistonRodMesh.position.set(0, -1.14, 0);
  pistonRodMesh.castShadow = true;
  pistonRodMesh.receiveShadow = true;
  ramGroup.add(pistonRodMesh);

  // 3. Lower Satin-Brass Pressure Collar & Ergonomic Crossbar Pump Handle
  //    Top of collar at -1.525 rises to -1.370 at max stroke (clears -1.28 gland by 9.0 cm!)
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
// 3. Left & Right Lateral Helical ACME Screw-Clamps & Tall Sculpted Platens
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
  // Platen shoe spans y in [-0.215, +0.725]: clears chamber side wall (-0.220) cleanly & cradles heaved basalt cap!
  const platenCenterY = 0.255;

  const platenShoeGeo = track(createChamferedBlock(0.072, 0.94, 0.92, 0.012));
  const gussetGeo = track(createChamferedBlock(0.048, 0.068, 0.78, 0.008));
  const vertRibGeo = track(createChamferedBlock(0.042, 0.80, 0.056, 0.007));
  const swivelSocketGeo = track(
    createClosedLathe(
      [
        [0.0, -0.038],
        [0.100, -0.038],
        [0.100, 0.008],
        [0.080, 0.038],
        [0.0, 0.038],
      ],
      22
    )
  );
  // True right-handed helical ACME lead-screw (exact 0.048 m/rev pitch matching solver rotation!)
  const acmeScrewGeo = track(createHelicalAcmeScrewGeometry(0.041, 0.059, 0.82, ACME_THREAD_PITCH, 24, 10));
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
  const hitBoxGeo = track(new THREE.BoxGeometry(0.96, 0.94, 1.00));

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
    shoe.position.set(shoeCenterX, platenCenterY, 0);
    shoe.castShadow = true;
    shoe.receiveShadow = true;
    platenGroup.add(shoe);

    // Horizontal & vertical forged backing ribs
    for (const dy of [-0.30, 0.30]) {
      const rib = new THREE.Mesh(gussetGeo, materials.forgedPressIron);
      rib.name = `LateralPlatenRib_${tag}_${dy > 0 ? 'Top' : 'Bot'}`;
      rib.position.set(shoeCenterX + side * 0.042, platenCenterY + dy, 0);
      rib.castShadow = true;
      platenGroup.add(rib);
    }
    for (const dz of [-0.32, 0.32]) {
      const vRib = new THREE.Mesh(vertRibGeo, materials.forgedPressIron);
      vRib.name = `LateralPlatenVRib_${tag}_${dz > 0 ? 'F' : 'B'}`;
      vRib.position.set(shoeCenterX + side * 0.040, platenCenterY, dz);
      vRib.castShadow = true;
      platenGroup.add(vRib);
    }

    // Swivel socket sits coaxial with the ACME screw axis (y = clampY = 0.20)
    const socket = new THREE.Mesh(swivelSocketGeo, materials.satinBrass);
    socket.name = `LateralSwivelSocket_${tag}`;
    socket.position.set(shoeCenterX + side * 0.048, clampY, 0);
    socket.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
    socket.castShadow = true;
    platenGroup.add(socket);

    // Translating & Rotating Helical Screw + 4-Spoke Brass/Iron Capstan Handwheel
    const screwWheelGroup = new THREE.Group();
    screwWheelGroup.name = `LateralScrewWheelGroup_${tag}`;
    screwWheelGroup.position.set(0, clampY, 0);
    group.add(screwWheelGroup);

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
// 4. 12 Interlocking Chiseled Vesicular Basalt & Banded Ochre-Tuff Rock Slabs
// ---------------------------------------------------------------------------

/**
 * Generates a watertight chiseled geological rock slab whose fracture/bedding walls
 * follow exact shared profiles (evalColumnSeamX, evalBeddingPlaneY) with crisp mineral
 * cleavage edges (zero mushy pillow rounding) and 3D object-space petrography attributes.
 */
function createFacetedRockSlabGeometry(tier: number, col: number, slabIndex: number): THREE.BufferGeometry {
  // Higher tessellation (18x12x14) yields crisp chiseled fracture lips, bedding ledges & basalt cleavage facets
  const geo = new THREE.BoxGeometry(1, 1, 1, 18, 12, 14);
  const posAttr = geo.attributes.position as THREE.BufferAttribute;
  const count = posAttr.count;

  const restPosArr = new Float32Array(count * 3);
  const stratumArr = new Float32Array(count);
  const fractureMaskArr = new Float32Array(count);
  const slipMaskArr = new Float32Array(count);
  const slabIdxArr = new Float32Array(count);

  const restCenterX = 0.5 * (COLUMN_BOUNDARIES_X[col] + COLUMN_BOUNDARIES_X[col + 1]);
  const restCenterY = 0.5 * (TIER_BOUNDARIES_Y[tier] + TIER_BOUNDARIES_Y[tier + 1]);

  const hasLeftFracture = col > 0;
  const hasRightFracture = col < SLAB_COLUMNS - 1;
  const hasBottomBedding = tier > 0;
  const hasTopBedding = tier < STRATA_TIERS - 1;
  const isTuff = tier === 1;

  for (let i = 0; i < count; i++) {
    const ux = posAttr.getX(i); // [-0.5, +0.5]
    const uy = posAttr.getY(i);
    const uz = posAttr.getZ(i);

    const tx = ux + 0.5;
    const ty = uy + 0.5;
    const zBase = uz * CORE_DEPTH;

    const xApprox =
      COLUMN_BOUNDARIES_X[col] + tx * (COLUMN_BOUNDARIES_X[col + 1] - COLUMN_BOUNDARIES_X[col]);
    const yBottom = evalBeddingPlaneY(tier, xApprox, zBase);
    const yTop = evalBeddingPlaneY(tier + 1, xApprox, zBase);
    let yWorld = yBottom + ty * (yTop - yBottom);

    const xLeft = evalColumnSeamX(col, yWorld, zBase);
    const xRight = evalColumnSeamX(col + 1, yWorld, zBase);
    let xWorld = xLeft + tx * (xRight - xLeft);
    let zWorld = zBase;

    // Tight crisp hairline seam inset & subtle mineral edge chamfer (preserves sharp cleaved corners!)
    const gapInsetX = 0.0022;
    const gapInsetY = 0.0020;
    const edgeProximityX = Math.max(0, Math.abs(ux) * 2 - 0.90) / 0.10;
    const edgeProximityY = Math.max(0, Math.abs(uy) * 2 - 0.90) / 0.10;
    const edgeProximityZ = Math.max(0, Math.abs(uz) * 2 - 0.90) / 0.10;
    const cornerBevel =
      (edgeProximityX * edgeProximityY + edgeProximityY * edgeProximityZ + edgeProximityX * edgeProximityZ) *
      0.0028;

    if (hasLeftFracture && tx < 0.015) xWorld += gapInsetX;
    if (hasRightFracture && tx > 0.985) xWorld -= gapInsetX;
    if (hasBottomBedding && ty < 0.015) yWorld += gapInsetY;
    if (hasTopBedding && ty > 0.985) yWorld -= gapInsetY;

    // Strict press-contact boundary guards so outer X platen seats & Tier-0 chamber-roof seating stay invariant!
    const onLeftPressEdge = !hasLeftFracture ? Math.max(0, 1 - tx / 0.10) : 0;
    const onRightPressEdge = !hasRightFracture ? Math.max(0, 1 - (1 - tx) / 0.10) : 0;
    const onChamberFloorEdge = tier === 0 ? Math.max(0, 1 - ty / 0.10) : 0;
    const pressSafeFade = (1 - onLeftPressEdge) * (1 - onRightPressEdge) * (1 - onChamberFloorEdge);

    // Chiseled fracture-lip chip notches right along internal dike & sill mouths (never boxy extruded edges!)
    const nearLeftFrac = hasLeftFracture ? Math.max(0, 1 - tx / 0.12) : 0;
    const nearRightFrac = hasRightFracture ? Math.max(0, 1 - (1 - tx) / 0.12) : 0;
    const nearBotBed = hasBottomBedding ? Math.max(0, 1 - ty / 0.14) : 0;
    const nearTopBed = hasTopBedding ? Math.max(0, 1 - (1 - ty) / 0.14) : 0;
    const nearInternalLip = Math.max(nearLeftFrac, nearRightFrac, nearBotBed, nearTopBed);

    const isOuterZ = Math.abs(uz) > 0.49;
    const isTopCrest = tier === STRATA_TIERS - 1 && ty > 0.98;
    const n1 = rockSmoothNoise(xWorld * 6.4 + 1.3, yWorld * 6.8 + 2.7, zWorld * 6.4 + 0.9) - 0.5;
    const n2 = rockSmoothNoise(xWorld * 16.5 + 4.1, yWorld * 15.2 + 0.6, zWorld * 16.5 + 3.3) - 0.5;
    const nChip = rockSmoothNoise(xWorld * 24.0 + 7.2, yWorld * 22.0 + 3.1, zWorld * 24.0 + 5.8);

    // Angular volcanic basalt columnar jointing terraces vs. laminated pyroclastic tuff bedding ledges
    const colPrism = isTuff
      ? (rockSmoothNoise(xWorld * 3.8 + 2.1, yWorld * 28.0 + 1.4, zWorld * 3.8 + 4.2) - 0.5) * 0.011
      : (rockSmoothNoise(xWorld * 11.5 + 3.7, yWorld * 3.6 + 1.1, zWorld * 11.5 + 2.9) - 0.5) * 0.014;
    const rawCleave = n1 * 0.022 + n2 * 0.010 + colPrism;
    const cleave = rawCleave * 0.58 + (Math.round(rawCleave * 85) / 85) * 0.42;

    // Micro-chipped irregular recession right on internal fault/bedding lips (crisp spalled edge bevel!)
    const lipChipRecess = nearInternalLip * Math.max(0, nChip - 0.42) * 0.018;

    if (isOuterZ) {
      // Allow crisp chiseled fracture relief right up to internal seam lips; fade only at outer press platen edges!
      const zShift = (cleave - lipChipRecess) * pressSafeFade;
      zWorld += Math.sign(uz) * zShift;
    }
    if (isTopCrest) {
      const yShift = (cleave * 0.98 - lipChipRecess * 0.72) * (1 - onLeftPressEdge) * (1 - onRightPressEdge);
      yWorld += yShift;
    }
    // Slight inward chip recess along internal vertical dike walls so parted rock silhouettes look ragged & chiseled
    if (hasLeftFracture && tx < 0.03) {
      xWorld += Math.max(0, nChip - 0.46) * 0.009 * (1 - onChamberFloorEdge);
    }
    if (hasRightFracture && tx > 0.97) {
      xWorld -= Math.max(0, nChip - 0.46) * 0.009 * (1 - onChamberFloorEdge);
    }

    const lx = xWorld - restCenterX;
    const ly = yWorld - restCenterY;
    const lz = zWorld;
    const shrink = 1.0 - cornerBevel;

    posAttr.setXYZ(i, lx * shrink, ly * shrink, lz * shrink);

    // Continuous 3D core rest coordinates for zero-stretch object-space petrography & branching fractures
    restPosArr[i * 3] = xWorld;
    restPosArr[i * 3 + 1] = yWorld;
    restPosArr[i * 3 + 2] = zWorld;
    stratumArr[i] = tier;
    slabIdxArr[i] = slabIndex;

    // Mark internal fracture faces (vertical dike seams & horizontal bedding planes) for magma wall glow
    const onLeftFrac = hasLeftFracture && tx < 0.05 ? 1.0 : 0.0;
    const onRightFrac = hasRightFracture && tx > 0.95 ? 1.0 : 0.0;
    const onBotBed = hasBottomBedding && ty < 0.05 ? 1.0 : 0.0;
    const onTopBed = hasTopBedding && ty > 0.95 ? 1.0 : 0.0;
    const onChamberRoof = tier === 0 && ty < 0.06 ? 0.85 : 0.0;
    fractureMaskArr[i] = Math.max(onLeftFrac, onRightFrac, onBotBed, onTopBed, onChamberRoof);
    // Dedicated vertical fault-wall slickenside mask (strictly parted vertical dike walls!)
    slipMaskArr[i] = Math.max(onLeftFrac, onRightFrac);
  }

  geo.setAttribute('aRestPos', new THREE.BufferAttribute(restPosArr, 3));
  geo.setAttribute('aStratum', new THREE.BufferAttribute(stratumArr, 1));
  geo.setAttribute('aFractureMask', new THREE.BufferAttribute(fractureMaskArr, 1));
  geo.setAttribute('aSlipMask', new THREE.BufferAttribute(slipMaskArr, 1));
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
// 5. Seam-Conforming Vertical Dikes, Arched Horizontal Sills, Xenoliths & Lobes
// ---------------------------------------------------------------------------

/**
 * Evaluates smooth column displacement `(dx, dy, rotZ)` interpolated across vertical tiers
 * so deformable dike vertices hug the exact jagged fracture walls of parting slabs!
 */
function sampleColumnWallX(
  slabs: ReadonlyArray<RockSlabState>,
  col: number,
  boundaryIdx: number,
  yWorld: number,
  zWorld: number
): number {
  const s0 = slabs[0 * SLAB_COLUMNS + col];
  const s1 = slabs[1 * SLAB_COLUMNS + col];
  const s2 = slabs[2 * SLAB_COLUMNS + col];

  const y0 = s0.restCenterY + s0.dy;
  const y1 = s1.restCenterY + s1.dy;
  const y2 = s2.restCenterY + s2.dy;

  let dx: number;
  let dy: number;
  let rotZ: number;
  let restCy: number;
  if (yWorld <= y1) {
    const t = Math.max(0, Math.min(1, (yWorld - y0) / Math.max(1e-4, y1 - y0)));
    dx = s0.dx + (s1.dx - s0.dx) * t;
    dy = s0.dy + (s1.dy - s0.dy) * t;
    rotZ = s0.rotZ + (s1.rotZ - s0.rotZ) * t;
    restCy = s0.restCenterY + (s1.restCenterY - s0.restCenterY) * t;
  } else {
    const t = Math.max(0, Math.min(1, (yWorld - y1) / Math.max(1e-4, y2 - y1)));
    dx = s1.dx + (s2.dx - s1.dx) * t;
    dy = s1.dy + (s2.dy - s1.dy) * t;
    rotZ = s1.rotZ + (s2.rotZ - s1.rotZ) * t;
    restCy = s1.restCenterY + (s2.restCenterY - s1.restCenterY) * t;
  }

  const yRest = Math.max(OVERBURDEN_BOTTOM_Y, Math.min(OVERBURDEN_BOTTOM_Y + OVERBURDEN_HEIGHT, yWorld - dy));
  const xSeamRest = evalColumnSeamX(boundaryIdx, yRest, zWorld);
  const restCx = s0.restCenterX;
  const lx = xSeamRest - restCx;
  const ly = yRest - restCy;
  // Exact 2D slab rotation in XY plane
  return restCx + dx + lx * Math.cos(rotZ) - ly * Math.sin(rotZ);
}

/**
 * Evaluates smooth arched bedding elevation along a tier of 4 slabs at world X.
 */
function sampleTierArchY(
  slabs: ReadonlyArray<RockSlabState>,
  tier: number,
  xWorld: number
): number {
  const s0 = slabs[tier * SLAB_COLUMNS + 0];
  const s1 = slabs[tier * SLAB_COLUMNS + 1];
  const s2 = slabs[tier * SLAB_COLUMNS + 2];
  const s3 = slabs[tier * SLAB_COLUMNS + 3];

  const xs = [s0.restCenterX + s0.dx, s1.restCenterX + s1.dx, s2.restCenterX + s2.dx, s3.restCenterX + s3.dx];
  const ys = [s0.dy, s1.dy, s2.dy, s3.dy];
  const rs = [s0.rotZ, s1.rotZ, s2.rotZ, s3.rotZ];

  if (xWorld <= xs[0]) return ys[0] + Math.sin(rs[0]) * (xWorld - xs[0]);
  if (xWorld >= xs[3]) return ys[3] + Math.sin(rs[3]) * (xWorld - xs[3]);

  for (let i = 0; i < 3; i++) {
    if (xWorld <= xs[i + 1]) {
      const t = (xWorld - xs[i]) / Math.max(1e-4, xs[i + 1] - xs[i]);
      const yA = ys[i] + Math.sin(rs[i]) * (xWorld - xs[i]);
      const yB = ys[i + 1] + Math.sin(rs[i + 1]) * (xWorld - xs[i + 1]);
      // Smooth cubic blend across slab joints so the magma lens arches continuously
      const st = t * t * (3 - 2 * t);
      return yA + (yB - yA) * st;
    }
  }
  return ys[1];
}

/**
 * Creates a watertight deformable vertical dike geometry with saved unit `(uX, uY, uZ)`
 * coordinates in `userData.unitCoords` for zero-allocation per-frame fracture conformation.
 */
function createDikeWedgeGeometry(seamIdx: number): THREE.BufferGeometry {
  // Higher tessellation (6x24x14) supports convex viscous magma meniscus pillowing across open cleft mouths!
  const geo = new THREE.BoxGeometry(1, 1, 1, 6, 24, 14);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const unitCoords = new Float32Array(pos.count * 3);
  const colBoundaryIdx = seamIdx === 0 ? 2 : seamIdx === 1 ? 1 : 3;
  const depthSpan = CORE_DEPTH - 0.026;

  for (let i = 0; i < pos.count; i++) {
    const ux = pos.getX(i); // [-0.5, +0.5]
    const uy = pos.getY(i) + 0.5; // [0, 1] bottom -> tip
    const uz = pos.getZ(i); // [-0.5, +0.5]
    unitCoords[i * 3] = ux;
    unitCoords[i * 3 + 1] = uy;
    unitCoords[i * 3 + 2] = uz;

    // Initial rest pose strictly inside (-CORE_HALF_D, +CORE_HALF_D) following jagged evalColumnSeamX!
    const zWorld = uz * depthSpan;
    const yWorld = CHAMBER_ROOF_Y - 0.02 + uy * (OVERBURDEN_HEIGHT * 0.55);
    const xSeam = evalColumnSeamX(colBoundaryIdx, Math.max(OVERBURDEN_BOTTOM_Y, yWorld), zWorld);
    const tipTaper = 1.0 - 0.52 * uy * uy;
    const halfThick = 0.014 * tipTaper;
    pos.setXYZ(i, xSeam + ux * 2 * halfThick, yWorld, zWorld * (1.0 - 0.01 * uy));
  }
  geo.userData.unitCoords = unitCoords;
  geo.computeVertexNormals();
  return geo;
}

/**
 * Zero-allocation per-frame update conforming a vertical magma dike mesh directly to the
 * exact jagged `evalColumnSeamX` walls & live parted positions of its flanking rock slabs,
 * swelling a plump viscous convex lava meniscus proud of exposed cleft mouths when pressurized!
 */
export function updateDikeGeometry(
  geo: THREE.BufferGeometry,
  seamIdx: number,
  height: number,
  veinThickness: number,
  slabs: ReadonlyArray<RockSlabState>,
  oozeBulge01 = 0,
  flowTimeSec = 0
): void {
  const unitCoords = geo.userData.unitCoords as Float32Array | undefined;
  if (!unitCoords) return;

  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colBoundaryIdx = seamIdx === 0 ? 2 : seamIdx === 1 ? 1 : 3;
  const leftCol = colBoundaryIdx - 1;
  const rightCol = colBoundaryIdx;
  const baseX = DIKE_SEAM_X[seamIdx];

  const sL2 = slabs[2 * SLAB_COLUMNS + leftCol];
  const sR2 = slabs[2 * SLAB_COLUMNS + rightCol];
  const crestDy = 0.5 * (sL2.dy + sR2.dy);

  const depthSpan = CORE_DEPTH - 0.022;
  const yStart = CHAMBER_ROOF_Y - 0.028;

  for (let i = 0; i < pos.count; i++) {
    const ux = unitCoords[i * 3]; // [-0.5, +0.5]
    const uy = unitCoords[i * 3 + 1]; // [0, 1]
    const uz = unitCoords[i * 3 + 2]; // [-0.5, +0.5]

    const zRecessed = uz * depthSpan * (1.0 - 0.010 * uy);
    // Cap dike top cleanly at the local basalt crest surface (zero midair shark-fin spikes!)
    const localCrestTop = evalBeddingPlaneY(STRATA_TIERS, baseX, zRecessed) + crestDy - 0.002;
    const yEnd = Math.min(localCrestTop, CHAMBER_ROOF_Y + height + crestDy * 0.45);
    const yWorld = yStart + uy * Math.max(0.02, yEnd - yStart);

    // Exact rotated & jagged left/right slab fracture wall coordinates at (yWorld, zRecessed)
    const xWallL = sampleColumnWallX(slabs, leftCol, colBoundaryIdx, yWorld, zRecessed);
    const xWallR = sampleColumnWallX(slabs, rightCol, colBoundaryIdx, yWorld, zRecessed);
    const xMid = 0.5 * (xWallL + xWallR);
    const actualGap = Math.max(0.004, xWallR - xWallL);

    // Fill the full real fracture gap + 4mm wall-lap seal (zero black gaps!) with hydraulic tip taper
    const tipTaper = uy > 0.86 ? 1.0 - 0.54 * ((uy - 0.86) / 0.14) : 1.0;
    const halfWidth = Math.max(actualGap * 0.5 + 0.005, veinThickness * 0.48) * tipTaper;

    // Convex viscous lava meniscus bulging organically proud of the exposed front/back fissure mouth
    // Strictly tapers to 0 at rock walls (|ux|=0.5) & near chamber floor/tip so zero floating edges!
    const archX = Math.max(0, Math.cos(ux * Math.PI));
    const endMaskZ = Math.max(0, (Math.abs(uz) - 0.36) / 0.14);
    const verticalEnv = Math.min(1, uy * 4.5) * tipTaper;
    const ropySwell = 0.72 + 0.28 * Math.sin(uy * 16.5 - flowTimeSec * 0.85 + seamIdx * 2.1);
    const meniscusOut =
      endMaskZ *
      archX *
      verticalEnv *
      ropySwell *
      Math.min(0.022, actualGap * 0.36 + 0.009 * oozeBulge01);
    const zFinal = zRecessed + Math.sign(uz) * meniscusOut;

    // Slight top crest pillow crown when dike approaches/breaches the upper basalt lip
    const crestCrown =
      uy > 0.94 && oozeBulge01 > 0.1
        ? archX * ((uy - 0.94) / 0.06) * Math.min(0.012, actualGap * 0.22)
        : 0;

    pos.setXYZ(i, xMid + ux * 2.0 * halfWidth, yWorld + crestCrown, zFinal);
  }

  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

/**
 * Creates a watertight deformable horizontal laccolith sill lens geometry with saved
 * unit `(uX, uY, uZ)` coordinates for zero-allocation arched bedding conformation.
 */
function createSillLensGeometry(sillIdx = 0): THREE.BufferGeometry {
  // Higher vertical & lateral resolution (28x6x14) for plump pillowing sill lips along bedding planes
  const geo = new THREE.BoxGeometry(2, 1, CORE_DEPTH - 0.026, 28, 6, 14);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const unitCoords = new Float32Array(pos.count * 3);
  const depthSpan = CORE_DEPTH - 0.026;
  const halfDepth = depthSpan * 0.5;

  for (let i = 0; i < pos.count; i++) {
    const ux = pos.getX(i); // [-1, +1]
    const uy = pos.getY(i); // [-0.5, +0.5]
    const uz = pos.getZ(i) / depthSpan; // [-0.5, +0.5]
    unitCoords[i * 3] = ux;
    unitCoords[i * 3 + 1] = uy;
    unitCoords[i * 3 + 2] = uz;

    const x = ux * 0.45;
    const z = uz * depthSpan;
    const bedY = evalBeddingPlaneY(sillIdx + 1, x, z);
    const r = Math.min(1, Math.abs(ux));
    const lensFactor = 0.22 + 0.78 * Math.cos(r * Math.PI * 0.46);
    const zEdgeTaper = 0.88 + 0.12 * Math.cos((z / halfDepth) * Math.PI * 0.45);
    pos.setXYZ(i, x, bedY + uy * 0.028 * lensFactor * zEdgeTaper, z);
  }
  geo.userData.unitCoords = unitCoords;
  geo.computeVertexNormals();
  return geo;
}

/**
 * Zero-allocation per-frame update conforming a horizontal magma sill lens strictly to:
 *  - Undulating `evalBeddingPlaneY` stratigraphy
 *  - Live anticlinal dome arch & slab tilt between lower & upper strata tiers
 *  - Actual outer rock flank boundaries (zero rigid flat tongues poking out the sides!)
 *  - Viscous convex lava meniscus swelling out of exposed front/back bedding mouths!
 */
export function updateSillGeometry(
  geo: THREE.BufferGeometry,
  sillIdx: number,
  halfWidth: number,
  sillThickness: number,
  slabs: ReadonlyArray<RockSlabState>,
  oozeBulge01 = 0,
  flowTimeSec = 0
): void {
  const unitCoords = geo.userData.unitCoords as Float32Array | undefined;
  if (!unitCoords) return;

  const pos = geo.attributes.position as THREE.BufferAttribute;
  const tierBelow = sillIdx;
  const tierAbove = sillIdx + 1;

  // Clamp sill lateral span strictly inside the outer rock slab edges so it never sticks out in midair
  const leftBelow = slabs[tierBelow * SLAB_COLUMNS + 0];
  const leftAbove = slabs[tierAbove * SLAB_COLUMNS + 0];
  const rightBelow = slabs[tierBelow * SLAB_COLUMNS + 3];
  const rightAbove = slabs[tierAbove * SLAB_COLUMNS + 3];
  const minRockX = Math.max(COLUMN_BOUNDARIES_X[0] + leftBelow.dx, COLUMN_BOUNDARIES_X[0] + leftAbove.dx) + 0.022;
  const maxRockX = Math.min(COLUMN_BOUNDARIES_X[4] + rightBelow.dx, COLUMN_BOUNDARIES_X[4] + rightAbove.dx) - 0.022;
  const safeHalfW = Math.max(0.04, Math.min(halfWidth, Math.min(-minRockX, maxRockX)));

  const depthSpan = CORE_DEPTH - 0.022;
  const halfDepth = depthSpan * 0.5;

  for (let i = 0; i < pos.count; i++) {
    const ux = unitCoords[i * 3]; // [-1, +1]
    const uy = unitCoords[i * 3 + 1]; // [-0.5, +0.5]
    const uz = unitCoords[i * 3 + 2]; // [-0.5, +0.5]

    const xWorld = ux * safeHalfW;
    const zWorld = uz * depthSpan;
    const bedY = evalBeddingPlaneY(tierAbove, xWorld, zWorld);

    const dyBelow = sampleTierArchY(slabs, tierBelow, xWorld);
    const dyAbove = sampleTierArchY(slabs, tierAbove, xWorld);
    const actualBedGap = Math.max(0.006, dyAbove - dyBelow);

    const r = Math.min(1, Math.abs(ux));
    const lensFactor = 0.24 + 0.76 * Math.cos(r * Math.PI * 0.46);
    const zEdgeTaper = 0.88 + 0.12 * Math.cos((zWorld / halfDepth) * Math.PI * 0.45);

    const midY = bedY + 0.5 * (dyBelow + dyAbove);
    // Fill actual arched cavity + 5mm bedding-wall lap seal, pinched cleanly at tips
    const localThickness = Math.max(actualBedGap + 0.008, sillThickness) * lensFactor * zEdgeTaper;

    // Convex viscous meniscus pillowing slightly proud across open front/back sill mouth
    // Rounded smoothly along both vertical arch (archY) and lateral tips (latRound) -> zero sharp box corners!
    const archY = Math.max(0, Math.cos(uy * Math.PI));
    const latRound = Math.max(0, Math.cos(r * Math.PI * 0.49));
    const endMaskZ = Math.max(0, (Math.abs(uz) - 0.34) / 0.16);
    const ropyBead = 0.72 + 0.28 * Math.sin(ux * 10.5 - flowTimeSec * 0.7 + sillIdx * 1.9);
    const meniscusOut =
      endMaskZ *
      archY *
      latRound *
      lensFactor *
      ropyBead *
      Math.min(0.022, localThickness * 0.32 + 0.008 * oozeBulge01);
    // Gentle round-edge pinch along top/bottom front sill rim so exposed lava looks plump & viscous
    const rimBevelZ = endMaskZ * (1.0 - archY) * Math.min(0.014, localThickness * 0.24);
    // Subtle gravitational sag right at the protruding lip
    const lipSag = endMaskZ * archY * latRound * Math.min(0.007, meniscusOut * 0.35);

    pos.setXYZ(
      i,
      xWorld,
      midY + uy * localThickness - lipSag,
      zWorld + Math.sign(uz) * (meniscusOut - rimBevelZ)
    );
  }

  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

/**
 * Welds duplicate vertices on a non-indexed BufferGeometry (e.g., Three.js IcosahedronGeometry)
 * into a watertight indexed mesh so `computeVertexNormals()` produces silky smooth interpolated
 * volcanic-glass normals instead of flat per-triangle facets!
 */
function weldIndexedGeometry(rawGeo: THREE.BufferGeometry): THREE.BufferGeometry {
  const pos = rawGeo.attributes.position as THREE.BufferAttribute;
  const vertMap = new Map<string, number>();
  const uniquePositions: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const key = `${Math.round(x * 1e5)}_${Math.round(y * 1e5)}_${Math.round(z * 1e5)}`;
    let idx = vertMap.get(key);
    if (idx === undefined) {
      idx = uniquePositions.length / 3;
      vertMap.set(key, idx);
      uniquePositions.push(x, y, z);
    }
    indices.push(idx);
  }

  const welded = new THREE.BufferGeometry();
  welded.setAttribute('position', new THREE.Float32BufferAttribute(uniquePositions, 3));
  welded.setIndex(indices);
  rawGeo.dispose();
  return welded;
}

/**
 * Creates a smooth watertight sculpted viscoplastic pahoehoe lava pillow / braided tongue geometry
 * (welded indexed subdivision-4 icosahedron -> silky-smooth glossy obsidian glass normals,
 * slumped rock-hugging contact skirt, bulbous advancing toe & arcuate transverse ropy folds).
 */
function createPahoehoeLobeGeometry(): THREE.BufferGeometry {
  const geo = weldIndexedGeometry(new THREE.IcosahedronGeometry(1.0, 4));
  const pos = geo.attributes.position as THREE.BufferAttribute;

  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    let z = pos.getZ(i);

    // Flattened rock-hugging base with subtly flared viscous contact skirt so pillows slump onto rock
    const baseSkirt = Math.exp(-Math.pow((y + 0.08) / 0.28, 2)) * 0.10;
    if (y < 0) {
      y *= 0.30;
    } else {
      y = Math.pow(y, 0.86);
    }

    // Organic viscous billow + deep arcuate braided transverse pahoehoe cord folds (zero polar harmonics!)
    const n1 = rockSmoothNoise(x * 1.7 + 1.4, y * 2.2 + 2.2, z * 1.7 + 0.7) - 0.5;
    const upperMask = Math.max(0, Math.min(1, (y + 0.15) * 1.15));
    const arcCoord = z * 9.4 + x * x * 4.4 + n1 * 2.8;
    const ropePrimary = Math.sin(arcCoord) * 0.044;
    const ropeBraided = Math.sin(arcCoord * 1.78 + x * 2.1) * 0.018;
    // Plump bulbous advancing lava toe swell toward +Z front rim
    const toeSwell = Math.max(0, z) * 0.14 * (1 - Math.abs(y) * 0.45);
    const disp = 1.0 + n1 * 0.18 + (ropePrimary + ropeBraided) * upperMask + baseSkirt;

    // Gentle viscous tongue elongation along +Z
    z = z * (1.26 + toeSwell * 0.45);
    x = x * (1.0 + toeSwell * 0.32);
    pos.setXYZ(i, x * disp, y * disp, z * disp);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Creates a cleaved angular dark basalt xenolith rock chunk floating inside the chamber.
 */
function createXenolithGeometry(seed: number): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(0.080, 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const n1 = rockSmoothNoise(x * 14 + seed, y * 14 + seed * 1.7, z * 14 + seed * 0.6);
    const n2 = rockSmoothNoise(x * 29 + seed * 2.3, y * 29 + seed * 0.9, z * 29 + seed * 1.4);
    const raw = 0.76 + n1 * 0.36 + (n2 - 0.5) * 0.14;
    // Slight angular cleavage quantization for sharp submerged basalt roof-clasts
    const sc = raw * 0.70 + (Math.round(raw * 14) / 14) * 0.30;
    pos.setXYZ(i, x * sc * 1.30, y * sc * 0.70, z * sc * 0.88);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Creates a watertight angular cleaved volcanic rock spall chip geometry (signedVol > 0)
 * for instanced basalt/tuff fault-lip debris tumbling off widening fractures.
 */
function createRockChipGeometry(): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(1.0, 1);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const n = rockSmoothNoise(x * 4.2 + 2.3, y * 4.2 + 5.1, z * 4.2 + 1.7);
    const sc = 0.72 + n * 0.52;
    pos.setXYZ(i, x * sc * 1.32, y * sc * 0.64, z * sc * 0.95);
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

  // 1. Churning Lower Magma Chamber Volume (recessed at z=±0.485 so drifting basalt xenoliths emerge visibly behind quartz!)
  const chamberGeo = track(new THREE.BoxGeometry(1.42, CHAMBER_HEIGHT - 0.025, CORE_DEPTH - 0.13, 18, 8, 12));
  const chamberMeltMesh = new THREE.Mesh(chamberGeo, materials.chamberMagma);
  chamberMeltMesh.name = 'MagmaChamberMeltVolume';
  chamberMeltMesh.position.set(0, CHAMBER_BOTTOM_Y + CHAMBER_HEIGHT * 0.5, 0);
  group.add(chamberMeltMesh);

  // 1b. Basal Chamber-Roof Magma Feeder Conduit (bridges chamber ceiling to Tier 0 slabs cleanly during basalLift!)
  const feederGeo = track(new THREE.BoxGeometry(1.44, 0.065, CORE_DEPTH - 0.04, 10, 2, 8));
  const chamberFeederMesh = new THREE.Mesh(feederGeo, materials.chamberMagma);
  chamberFeederMesh.name = 'ChamberRoofFeederMelt';
  chamberFeederMesh.position.set(0, CHAMBER_ROOF_Y - 0.025, 0);
  group.add(chamberFeederMesh);

  // 2. 6 Drifting Dark Vesicular Basalt Xenolith Roof-Clasts emerging right behind front & back quartz windows!
  const xenolithMeshes: THREE.Mesh[] = [];
  const xenoOffsets: ReadonlyArray<readonly [number, number, number]> = [
    [-0.42, CHAMBER_ROOF_Y - 0.13, 0.47],
    [0.14, CHAMBER_ROOF_Y - 0.15, 0.48],
    [0.50, CHAMBER_ROOF_Y - 0.12, 0.46],
    [-0.36, CHAMBER_ROOF_Y - 0.14, -0.47],
    [0.08, CHAMBER_ROOF_Y - 0.16, -0.48],
    [0.44, CHAMBER_ROOF_Y - 0.12, -0.46],
  ];
  for (let i = 0; i < xenoOffsets.length; i++) {
    const xGeo = track(createXenolithGeometry(i * 4.3 + 1.2));
    const mesh = new THREE.Mesh(xGeo, materials.xenolithBasalt);
    mesh.name = `ChamberXenolith_${i}`;
    mesh.position.set(xenoOffsets[i][0], xenoOffsets[i][1], xenoOffsets[i][2]);
    group.add(mesh);
    xenolithMeshes.push(mesh);
  }

  // 3. 3 Dynamic Seam-Conforming Vertical Magma Dike Conduits (world-space deformed in onTick)
  const dikeMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < DIKE_SEAM_X.length; i++) {
    const dGeo = track(createDikeWedgeGeometry(i));
    const mesh = new THREE.Mesh(dGeo, materials.dikeMagma);
    mesh.name = `VerticalMagmaDike_${i}`;
    mesh.frustumCulled = false;
    group.add(mesh);
    dikeMeshes.push(mesh);
  }

  // 4. 2 Dynamic Arched Horizontal Magma Sill Lenses (world-space deformed in onTick)
  const sillMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < SILL_BED_Y.length; i++) {
    const sGeo = track(createSillLensGeometry(i));
    const mesh = new THREE.Mesh(sGeo, materials.sillMagma);
    mesh.name = `HorizontalMagmaSill_${i}`;
    mesh.frustumCulled = false;
    group.add(mesh);
    sillMeshes.push(mesh);
  }

  // 5. MAX_LOBES Instanced Smooth Braided Ropy Viscoplastic Pahoehoe Magma Lobes (1 draw call!)
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

  // 5b. ROCK_CHIP_COUNT Pre-Allocated Dislodged Angular Basalt & Yellow-Ochre Tuff Fault-Lip Chips (1 draw call!)
  const rockChipGeo = track(createRockChipGeometry());
  const rockChipToneData = new Float32Array(ROCK_CHIP_COUNT);
  const rockChipToneAttr = new THREE.InstancedBufferAttribute(rockChipToneData, 1);
  rockChipGeo.setAttribute('aChipTone', rockChipToneAttr);

  const rockChipInstancedMesh = new THREE.InstancedMesh(
    rockChipGeo,
    materials.rockChip ?? materials.stratifiedRockCore,
    ROCK_CHIP_COUNT
  );
  rockChipInstancedMesh.name = 'FaultLipRockChips';
  rockChipInstancedMesh.castShadow = true;
  rockChipInstancedMesh.receiveShadow = true;
  rockChipInstancedMesh.frustumCulled = false;
  for (let i = 0; i < ROCK_CHIP_COUNT; i++) {
    rockChipInstancedMesh.setMatrixAt(i, hiddenMat);
  }
  rockChipInstancedMesh.instanceMatrix.needsUpdate = true;
  group.add(rockChipInstancedMesh);

  // 6. EMBER_COUNT Restrained Rising Eruptive Volcanic Ash / Ember Sparks (1 draw call!)
  const emberGeo = track(new THREE.OctahedronGeometry(0.010, 0));
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
    chamberFeederMesh,
    xenolithMeshes,
    dikeMeshes,
    sillMeshes,
    lobeInstancedMesh,
    lobeThermalAttr,
    rockChipInstancedMesh,
    rockChipToneAttr,
    emberInstancedMesh,
    emberHeatAttr,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}
