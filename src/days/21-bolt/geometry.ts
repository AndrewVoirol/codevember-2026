import * as THREE from 'three/webgpu';
import {
  BOLT_HEAD_Z_HALF,
  BOLT_TAIL_Z_BACK,
  BOLT_TAIL_Z_FRONT,
  CASE_INNER_MAX_X,
  CASE_INNER_MAX_Y,
  CASE_INNER_MIN_X,
  CASE_INNER_MIN_Y,
  FOREND_THICKNESS,
  FOREND_X,
  GATE_HALF_HEIGHT,
  GUIDE_PIN_X,
  GUIDE_PIN_Y,
  KEY_BIT_RADII,
  KEY_SHAFT_RADIUS,
  KEY_TALON_RADIUS,
  KEY_X,
  KEY_Y,
  LEVER_BELLY_LOCAL_X,
  LEVER_BELLY_LOCAL_Y,
  LEVER_BEVEL,
  LEVER_COUNT,
  LEVER_GATE_SPECS,
  LEVER_THICKNESS,
  PIVOT_X,
  PIVOT_Y,
  REST_PIN_X,
  REST_PIN_Y,
  SPRING_POST_X,
  SPRING_POST_Y,
  STUMP_HEIGHT,
  STUMP_WIDTH,
  STUMP_X_UNLOCKED,
  STUMP_Y,
  STUMP_Z_BACK,
  STUMP_Z_FRONT,
  getLeverZ,
} from './lockMechanism';
import type { BoltMaterials } from './materials';

/**
 * Ensures 2D polygon points are counter-clockwise (required for outward normals on ExtrudeGeometry).
 */
function ensureCCW(pts: THREE.Vector2[]): THREE.Vector2[] {
  if (THREE.ShapeUtils.isClockWise(pts)) {
    return [...pts].reverse();
  }
  return pts;
}

/**
 * Ensures 2D hole contour points are clockwise (required for outward normals on extruded holes).
 */
function ensureCW(pts: THREE.Vector2[]): THREE.Vector2[] {
  if (!THREE.ShapeUtils.isClockWise(pts)) {
    return [...pts].reverse();
  }
  return pts;
}

/**
 * Builds a rounded rectangle point list in CCW order.
 */
function makeRoundedRectPts(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  radius: number,
  segs = 5
): THREE.Vector2[] {
  const r = Math.min(radius, (maxX - minX) * 0.45, (maxY - minY) * 0.45);
  const pts: THREE.Vector2[] = [];
  const corners: Array<[number, number, number, number]> = [
    [maxX - r, minY + r, -Math.PI / 2, 0],
    [maxX - r, maxY - r, 0, Math.PI / 2],
    [minX + r, maxY - r, Math.PI / 2, Math.PI],
    [minX + r, minY + r, Math.PI, Math.PI * 1.5],
  ];
  for (const [cx, cy, a0, a1] of corners) {
    for (let s = 0; s <= segs; s++) {
      const a = a0 + (a1 - a0) * (s / segs);
      pts.push(new THREE.Vector2(cx + Math.cos(a) * r, cy + Math.sin(a) * r));
    }
  }
  return ensureCCW(pts);
}

/**
 * Builds a circular hole path in CW order.
 */
function makeCircleHole(cx: number, cy: number, radius: number, segs = 28): THREE.Path {
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i < segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    pts.push(new THREE.Vector2(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius));
  }
  const cw = ensureCW(pts);
  const path = new THREE.Path();
  path.moveTo(cw[0].x, cw[0].y);
  for (let i = 1; i < cw.length; i++) path.lineTo(cw[i].x, cw[i].y);
  path.closePath();
  return path;
}

/**
 * Builds a watertight closed LatheGeometry along Y (ordered strictly bottom-to-top for outward normals),
 * optionally rotated to lie along Z.
 */
function makeClosedLatheZ(
  profileZR: Array<[number, number]>,
  radialSegments = 28
): THREE.BufferGeometry {
  // LatheGeometry revolves around +Y with y strictly increasing from bottom to top for outward normals.
  // After Lathe, rotateX(+PI/2) maps +Y -> +Z (and +Z -> -Y, preserving right-handed outward winding!).
  const pts: THREE.Vector2[] = [];
  const sorted = [...profileZR].sort((a, b) => a[0] - b[0]);
  if (sorted[0][1] > 1e-4) {
    pts.push(new THREE.Vector2(0, sorted[0][0]));
  }
  for (const [z, r] of sorted) {
    pts.push(new THREE.Vector2(Math.max(0, r), z));
  }
  const last = sorted[sorted.length - 1];
  if (last[1] > 1e-4) {
    pts.push(new THREE.Vector2(0, last[0]));
  }
  const geo = new THREE.LatheGeometry(pts, radialSegments);
  geo.rotateX(Math.PI / 2);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Creates a countersunk slotted flathead brass wood-screw oriented with its face pointing along +X
 * (or +Z when rotateToZ is true). Built from two outward-wound beveled D-caps flanking a crisp
 * recessed screwdriver slot + solid recessed floor disc.
 */
function buildSlottedBrassScrew(
  radius: number,
  headDepth: number,
  slotWidth: number,
  slotDepth: number,
  mat: THREE.Material,
  disposables: THREE.BufferGeometry[]
): THREE.Group {
  const grp = new THREE.Group();
  grp.name = 'SlottedBrassScrew';

  // Backing countersunk base disc behind the slot
  const baseDepth = Math.max(0.008, headDepth - slotDepth);
  const baseShape = new THREE.Shape();
  baseShape.absarc(0, 0, radius, 0, Math.PI * 2, false);
  const baseGeo = new THREE.ExtrudeGeometry(baseShape, {
    depth: baseDepth,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.002,
    bevelSegments: 2,
    curveSegments: 24,
  });
  baseGeo.translate(0, 0, -headDepth);
  disposables.push(baseGeo);
  const baseMesh = new THREE.Mesh(baseGeo, mat);
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  grp.add(baseMesh);

  // Two opposing D-lobes forming the genuine recessed screwdriver slot
  const halfSlot = slotWidth * 0.5;
  const theta0 = Math.asin(Math.min(0.85, halfSlot / radius));
  for (const sign of [1, -1]) {
    const pts: THREE.Vector2[] = [];
    const segs = 16;
    const startA = sign > 0 ? theta0 : Math.PI + theta0;
    const endA = sign > 0 ? Math.PI - theta0 : Math.PI * 2 - theta0;
    for (let s = 0; s <= segs; s++) {
      const a = startA + (endA - startA) * (s / segs);
      pts.push(new THREE.Vector2(Math.cos(a) * radius, Math.sin(a) * radius));
    }
    const ccw = ensureCCW(pts);
    const lobeShape = new THREE.Shape(ccw);
    const lobeGeo = new THREE.ExtrudeGeometry(lobeShape, {
      depth: slotDepth,
      bevelEnabled: true,
      bevelThickness: 0.0025,
      bevelSize: 0.002,
      bevelSegments: 2,
    });
    lobeGeo.translate(0, 0, -slotDepth);
    disposables.push(lobeGeo);
    const lobeMesh = new THREE.Mesh(lobeGeo, mat);
    lobeMesh.castShadow = true;
    lobeMesh.receiveShadow = true;
    grp.add(lobeMesh);
  }

  return grp;
}

// =============================================================================
// 1. Oiled Charred-Oak Cutaway Specimen Slab
// =============================================================================

export interface OakSpecimenAssembly {
  group: THREE.Group;
  dispose: () => void;
}

export function buildOakSpecimenBlock(materials: BoltMaterials): OakSpecimenAssembly {
  const group = new THREE.Group();
  group.name = 'CharredOakSpecimenBlock';
  const disposables: THREE.BufferGeometry[] = [];

  const outerMinX = -1.14;
  const outerMaxX = FOREND_X; // +0.94 (meets mortised brass forend cleanly)
  const outerMinY = -0.80;
  const outerMaxY = 0.82;

  const pocketMinX = CASE_INNER_MIN_X - 0.045; // -0.845
  const pocketMinY = CASE_INNER_MIN_Y - 0.04; // -0.605
  const pocketMaxY = CASE_INNER_MAX_Y + 0.04; // +0.625

  // 1A. Solid Rear Charred-Oak Backing Timber Slab (z: -0.355 .. -0.221, zero overlap with front C-frame)
  const rearPts = makeRoundedRectPts(outerMinX, outerMinY, outerMaxX, outerMaxY, 0.036, 5);
  const rearShape = new THREE.Shape(rearPts);
  const rearGeo = new THREE.ExtrudeGeometry(rearShape, {
    depth: 0.094,
    bevelEnabled: true,
    bevelThickness: 0.020,
    bevelSize: 0.020,
    bevelSegments: 3,
  });
  rearGeo.translate(0, 0, -0.335);
  disposables.push(rearGeo);
  const rearMesh = new THREE.Mesh(rearGeo, materials.charredOak);
  rearMesh.name = 'OakRearBacking';
  rearMesh.castShadow = true;
  rearMesh.receiveShadow = true;
  group.add(rearMesh);

  // 1B. Sculpted Front C-Frame Oak Cutaway Surround (z: -0.220 .. +0.252)
  // Leaves the mortise cavity and right brass forend cleanly exposed with generous chamfered oak bevels
  const cPts = ensureCCW([
    new THREE.Vector2(outerMinX + 0.03, outerMinY),
    new THREE.Vector2(outerMaxX - 0.005, outerMinY),
    new THREE.Vector2(outerMaxX - 0.005, pocketMinY),
    new THREE.Vector2(pocketMinX + 0.025, pocketMinY),
    new THREE.Vector2(pocketMinX, pocketMinY + 0.025),
    new THREE.Vector2(pocketMinX, pocketMaxY - 0.025),
    new THREE.Vector2(pocketMinX + 0.025, pocketMaxY),
    new THREE.Vector2(outerMaxX - 0.005, pocketMaxY),
    new THREE.Vector2(outerMaxX - 0.005, outerMaxY),
    new THREE.Vector2(outerMinX + 0.03, outerMaxY),
    new THREE.Vector2(outerMinX, outerMaxY - 0.03),
    new THREE.Vector2(outerMinX, outerMinY + 0.03),
  ]);
  const cShape = new THREE.Shape(cPts);
  const cGeo = new THREE.ExtrudeGeometry(cShape, {
    depth: 0.428,
    bevelEnabled: true,
    bevelThickness: 0.022,
    bevelSize: 0.022,
    bevelSegments: 3,
  });
  cGeo.translate(0, 0, -0.198);
  disposables.push(cGeo);
  const cMesh = new THREE.Mesh(cGeo, materials.charredOak);
  cMesh.name = 'OakCutawayFrame';
  cMesh.castShadow = true;
  cMesh.receiveShadow = true;
  group.add(cMesh);

  return {
    group,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 2. Mortised Satin-Brass Forend Faceplate & Countersunk Slotted Screws
// =============================================================================

export interface ForendAssembly {
  group: THREE.Group;
  dispose: () => void;
}

export function buildForendAssembly(materials: BoltMaterials): ForendAssembly {
  const group = new THREE.Group();
  group.name = 'ForendBrassAssembly';
  const disposables: THREE.BufferGeometry[] = [];

  // Built in local XY (where local X -> world -Z, local Y -> world Y, extrude +Z -> world +X)
  // Forend spans world Z in [-0.245, +0.245], world Y in [-0.76, +0.78]
  const halfZ = 0.245;
  const minY = -0.76;
  const maxY = 0.78;

  const outerPts = makeRoundedRectPts(-halfZ, minY, halfZ, maxY, 0.038, 6);
  const plateShape = new THREE.Shape(outerPts);

  // Precision rectangular through-slot matching the blued-steel deadbolt head (+0.008 clearance)
  const slotHalfZ = BOLT_HEAD_Z_HALF + 0.01;
  const slotMinY = STUMP_Y - 0.178;
  const slotMaxY = STUMP_Y + 0.178;
  const slotHolePts = ensureCW([
    new THREE.Vector2(-slotHalfZ, slotMinY),
    new THREE.Vector2(slotHalfZ, slotMinY),
    new THREE.Vector2(slotHalfZ, slotMaxY),
    new THREE.Vector2(-slotHalfZ, slotMaxY),
  ]);
  const slotHole = new THREE.Path(slotHolePts);
  plateShape.holes.push(slotHole);

  const plateGeo = new THREE.ExtrudeGeometry(plateShape, {
    depth: FOREND_THICKNESS,
    bevelEnabled: true,
    bevelThickness: 0.008,
    bevelSize: 0.008,
    bevelSegments: 2,
  });
  // Rotate +PI/2 around Y maps local +Z -> world +X, local +X -> world -Z
  plateGeo.rotateY(Math.PI / 2);
  plateGeo.translate(FOREND_X, 0, 0);
  disposables.push(plateGeo);

  const plateMesh = new THREE.Mesh(plateGeo, materials.forendBrass);
  plateMesh.name = 'ForendFaceplate';
  plateMesh.castShadow = true;
  plateMesh.receiveShadow = true;
  group.add(plateMesh);

  // Two countersunk slotted brass wood-screws seated cleanly top & bottom
  const screwPositions: Array<[number, number]> = [
    [0.675, 0.56], // top screw rotated +32 deg
    [-0.655, -0.82], // bottom screw rotated -47 deg
  ];
  for (const [sy, rotAngle] of screwPositions) {
    const screw = buildSlottedBrassScrew(0.056, 0.024, 0.015, 0.013, materials.forendBrass, disposables);
    screw.rotation.y = Math.PI / 2;
    screw.rotation.x = rotAngle;
    screw.position.set(FOREND_X + FOREND_THICKNESS + 0.011, sy, 0);
    group.add(screw);
  }

  return {
    group,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 3. Skeletonized Gunmetal Lock Case, Cutaway Front Rim & Hardened Internal Pins
// =============================================================================

export interface LockCaseAssembly {
  group: THREE.Group;
  dispose: () => void;
}

export function buildLockCaseAssembly(materials: BoltMaterials): LockCaseAssembly {
  const group = new THREE.Group();
  group.name = 'LockCaseAssembly';
  const disposables: THREE.BufferGeometry[] = [];

  const outerMinX = CASE_INNER_MIN_X - 0.04; // -0.84
  const outerMaxX = FOREND_X; // +0.94
  const outerMinY = CASE_INNER_MIN_Y - 0.035; // -0.60
  const outerMaxY = CASE_INNER_MAX_Y + 0.035; // +0.62

  // 3A. Gunmetal Lock Case Backplate (z: -0.22 .. -0.18)
  const backShape = new THREE.Shape(
    makeRoundedRectPts(outerMinX, outerMinY, outerMaxX, outerMaxY, 0.02, 4)
  );
  const backGeo = new THREE.ExtrudeGeometry(backShape, {
    depth: 0.036,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
  });
  backGeo.translate(0, 0, -0.22);
  disposables.push(backGeo);
  const backMesh = new THREE.Mesh(backGeo, materials.gunmetalCase);
  backMesh.name = 'LockCaseBackplate';
  backMesh.castShadow = true;
  backMesh.receiveShadow = true;
  group.add(backMesh);

  // 3B. Perimeter Steel Case Walls (top, bottom, rear left; z: -0.180 .. +0.188, cleanly abutted)
  const wallDepth = 0.360;
  const wallPts = ensureCCW([
    new THREE.Vector2(outerMinX, outerMinY),
    new THREE.Vector2(outerMaxX, outerMinY),
    new THREE.Vector2(outerMaxX, CASE_INNER_MIN_Y),
    new THREE.Vector2(CASE_INNER_MIN_X, CASE_INNER_MIN_Y),
    new THREE.Vector2(CASE_INNER_MIN_X, CASE_INNER_MAX_Y),
    new THREE.Vector2(outerMaxX, CASE_INNER_MAX_Y),
    new THREE.Vector2(outerMaxX, outerMaxY),
    new THREE.Vector2(outerMinX, outerMaxY),
  ]);
  const wallGeo = new THREE.ExtrudeGeometry(new THREE.Shape(wallPts), {
    depth: wallDepth,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
  });
  wallGeo.translate(0, 0, -0.176);
  disposables.push(wallGeo);
  const wallMesh = new THREE.Mesh(wallGeo, materials.gunmetalCase);
  wallMesh.name = 'LockCaseWalls';
  wallMesh.castShadow = true;
  wallMesh.receiveShadow = true;
  group.add(wallMesh);

  // Right inner throat walls strictly inside CASE_INNER_MIN_Y..CASE_INNER_MAX_Y (zero overlap with case walls)
  const throatW = FOREND_X - CASE_INNER_MAX_X;
  const throatCenterX = (CASE_INNER_MAX_X + FOREND_X) * 0.5;
  const throatTopH = Math.max(0.02, CASE_INNER_MAX_Y - (STUMP_Y + 0.178));
  const throatGeoTop = new THREE.BoxGeometry(throatW, throatTopH, wallDepth);
  throatGeoTop.translate(throatCenterX, STUMP_Y + 0.178 + throatTopH * 0.5, 0.004);
  disposables.push(throatGeoTop);
  const throatTop = new THREE.Mesh(throatGeoTop, materials.gunmetalCase);
  throatTop.name = 'CaseThroatTop';
  throatTop.castShadow = true;
  throatTop.receiveShadow = true;
  group.add(throatTop);

  const throatBotH = Math.max(0.02, STUMP_Y - 0.178 - CASE_INNER_MIN_Y);
  const throatGeoBot = new THREE.BoxGeometry(throatW, throatBotH, wallDepth);
  throatGeoBot.translate(throatCenterX, CASE_INNER_MIN_Y + throatBotH * 0.5, 0.004);
  disposables.push(throatGeoBot);
  const throatBot = new THREE.Mesh(throatGeoBot, materials.gunmetalCase);
  throatBot.name = 'CaseThroatBottom';
  throatBot.castShadow = true;
  throatBot.receiveShadow = true;
  group.add(throatBot);

  // 3C. Skeletonized Museum-Cutaway Front Inspection Rim (z: +0.188 .. +0.220)
  // Frames the outer steel case perimeter cleanly while keeping the entire lever pack,
  // bolt stump, talon, pivot post, and phosphor-bronze springs 100% open to inspection!
  const bezelOuter = makeRoundedRectPts(outerMinX, outerMinY, outerMaxX, outerMaxY, 0.02, 4);
  const bezelShape = new THREE.Shape(bezelOuter);

  // Wide museum-cutaway viewing window cutout (leaves subtle corner tabs & bottom keyhole bridge)
  const winHolePts = ensureCW([
    new THREE.Vector2(-0.74, -0.49),
    new THREE.Vector2(-0.18, -0.49),
    new THREE.Vector2(-0.14, -0.43),
    new THREE.Vector2(-0.14, -0.35),
    new THREE.Vector2(0.22, -0.35),
    new THREE.Vector2(0.22, -0.43),
    new THREE.Vector2(0.26, -0.49),
    new THREE.Vector2(0.84, -0.49),
    new THREE.Vector2(0.84, 0.51),
    new THREE.Vector2(-0.74, 0.51),
  ]);
  bezelShape.holes.push(new THREE.Path(winHolePts));

  const bezelGeo = new THREE.ExtrudeGeometry(bezelShape, {
    depth: 0.024,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
  });
  bezelGeo.translate(0, 0, 0.192);
  disposables.push(bezelGeo);
  const bezelMesh = new THREE.Mesh(bezelGeo, materials.gunmetalCase);
  bezelMesh.name = 'CutawayFrontRim';
  bezelMesh.castShadow = true;
  bezelMesh.receiveShadow = true;
  group.add(bezelMesh);

  // 3D. Recessed Satin-Brass Keyhole Escutcheon Ring (mounted proud at z: +0.197..+0.237, zero z-fighting)
  // Classic keyhole collar with warded downward slot
  const escOuter: THREE.Vector2[] = [];
  const escSegs = 32;
  for (let i = 0; i < escSegs; i++) {
    const a = (i / escSegs) * Math.PI * 2;
    escOuter.push(new THREE.Vector2(Math.cos(a) * 0.118, Math.sin(a) * 0.118));
  }
  const escShape = new THREE.Shape(ensureCCW(escOuter));
  // Keyhole ward aperture: round barrel bore + downward rectangular key-bit slot
  const khPts: THREE.Vector2[] = [];
  const barrelR = KEY_SHAFT_RADIUS + 0.014;
  const slotHalfW = 0.022;
  const slotAngle = Math.asin(slotHalfW / barrelR);
  // Arc from bottom-right (-PI/2 + slotAngle) CCW around top to bottom-left (3PI/2 - slotAngle)
  const khArcSegs = 22;
  for (let s = 0; s <= khArcSegs; s++) {
    const a = -Math.PI / 2 + slotAngle + (Math.PI * 2 - 2 * slotAngle) * (s / khArcSegs);
    khPts.push(new THREE.Vector2(Math.cos(a) * barrelR, Math.sin(a) * barrelR));
  }
  khPts.push(new THREE.Vector2(-slotHalfW, -0.102));
  khPts.push(new THREE.Vector2(slotHalfW, -0.102));
  escShape.holes.push(new THREE.Path(ensureCW(khPts)));

  const escGeo = new THREE.ExtrudeGeometry(escShape, {
    depth: 0.030,
    bevelEnabled: true,
    bevelThickness: 0.005,
    bevelSize: 0.005,
    bevelSegments: 2,
  });
  escGeo.translate(KEY_X, KEY_Y, 0.202);
  disposables.push(escGeo);
  const escMesh = new THREE.Mesh(escGeo, materials.forendBrass);
  escMesh.name = 'KeyholeEscutcheon';
  escMesh.castShadow = true;
  escMesh.receiveShadow = true;
  group.add(escMesh);

  // Small gunmetal mounting bridge tying brass keyhole escutcheon cleanly to lower cutaway pedestal
  const bridgeGeo = new THREE.BoxGeometry(0.16, 0.075, 0.018);
  bridgeGeo.translate(KEY_X, -0.365, 0.197);
  disposables.push(bridgeGeo);
  const bridgeMesh = new THREE.Mesh(bridgeGeo, materials.gunmetalCase);
  bridgeMesh.name = 'KeyholeBridge';
  bridgeMesh.castShadow = true;
  bridgeMesh.receiveShadow = true;
  group.add(bridgeMesh);

  // 3E. Rear Bronze Bolt Wear-Rail (horizontal machined slide pad behind bolt tail)
  const railGeo = new THREE.BoxGeometry(1.52, 0.14, 0.012);
  railGeo.translate(0.06, STUMP_Y, -0.174);
  disposables.push(railGeo);
  const railMesh = new THREE.Mesh(railGeo, materials.forendBrass);
  railMesh.name = 'RearWearRail';
  railMesh.receiveShadow = true;
  group.add(railMesh);

  // 3F. Hardened Steel Lever Pivot Post + 4 Satin-Brass Inter-Lever Spacer Washers + Front Nut
  const pivotPinGeo = makeClosedLatheZ([
    [-0.182, 0.056],
    [0.158, 0.056],
    [0.162, 0.072],
    [0.186, 0.072],
    [0.194, 0.052],
    [0.198, 0.036],
  ]);
  pivotPinGeo.translate(PIVOT_X, PIVOT_Y, 0);
  disposables.push(pivotPinGeo);
  const pivotMesh = new THREE.Mesh(pivotPinGeo, materials.hardenedSteel);
  pivotMesh.name = 'LeverPivotPost';
  pivotMesh.castShadow = true;
  pivotMesh.receiveShadow = true;
  group.add(pivotMesh);

  // Rear thrust washer + 4 inter-lever brass spacer washers (visible in 3/4 parallax!)
  const washerOuterR = 0.098;
  const washerInnerR = 0.057;
  const washerShape = new THREE.Shape();
  washerShape.absarc(0, 0, washerOuterR, 0, Math.PI * 2, false);
  washerShape.holes.push(makeCircleHole(0, 0, washerInnerR, 24));

  for (let w = 0; w < LEVER_COUNT - 1; w++) {
    const zMid = (getLeverZ(w) + getLeverZ(w + 1)) * 0.5;
    const wThickness = 0.014;
    const wGeo = new THREE.ExtrudeGeometry(washerShape, {
      depth: wThickness,
      bevelEnabled: true,
      bevelThickness: 0.0012,
      bevelSize: 0.0012,
      bevelSegments: 1,
      curveSegments: 24,
    });
    wGeo.translate(PIVOT_X, PIVOT_Y, zMid - wThickness * 0.5);
    disposables.push(wGeo);
    const wMesh = new THREE.Mesh(wGeo, materials.forendBrass);
    wMesh.name = `LeverSpacerWasher_${w}`;
    wMesh.castShadow = true;
    wMesh.receiveShadow = true;
    group.add(wMesh);
  }

  // Front chamfered brass pivot retaining collar
  const capWasherGeo = new THREE.ExtrudeGeometry(washerShape, {
    depth: 0.012,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.002,
    bevelSegments: 2,
    curveSegments: 24,
  });
  capWasherGeo.translate(PIVOT_X, PIVOT_Y, getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.004);
  disposables.push(capWasherGeo);
  const capWasherMesh = new THREE.Mesh(capWasherGeo, materials.forendBrass);
  capWasherMesh.name = 'PivotFrontWasher';
  capWasherMesh.castShadow = true;
  capWasherMesh.receiveShadow = true;
  group.add(capWasherMesh);

  // 3G. Hardened Steel Spring Abutment Post (anchors all 5 phosphor-bronze leaf springs!)
  const springPostGeo = makeClosedLatheZ([
    [-0.182, 0.044],
    [0.168, 0.044],
    [0.178, 0.054],
    [0.190, 0.046],
  ]);
  springPostGeo.translate(SPRING_POST_X, SPRING_POST_Y, 0);
  disposables.push(springPostGeo);
  const springPostMesh = new THREE.Mesh(springPostGeo, materials.hardenedSteel);
  springPostMesh.name = 'SpringAbutmentPost';
  springPostMesh.castShadow = true;
  springPostMesh.receiveShadow = true;
  group.add(springPostMesh);

  // 3H. Lever Bottom Rest Stop Pin with Satin-Brass Sleeve (outside key sweep circle)
  const restPinGeo = makeClosedLatheZ([
    [-0.182, 0.032],
    [0.162, 0.032],
    [0.172, 0.025],
  ]);
  restPinGeo.translate(REST_PIN_X, REST_PIN_Y, 0);
  disposables.push(restPinGeo);
  const restPinMesh = new THREE.Mesh(restPinGeo, materials.forendBrass);
  restPinMesh.name = 'LeverRestStopPin';
  restPinMesh.castShadow = true;
  restPinMesh.receiveShadow = true;
  group.add(restPinMesh);

  // 3I. Hardened Bolt-Tail Guide Pin (rides inside the horizontal guide slot of the deadbolt tail)
  const guidePinGeo = makeClosedLatheZ([
    [-0.182, 0.030],
    [-0.068, 0.030],
    [-0.060, 0.040],
    [-0.052, 0.034],
  ]);
  guidePinGeo.translate(GUIDE_PIN_X, GUIDE_PIN_Y, 0);
  disposables.push(guidePinGeo);
  const guidePinMesh = new THREE.Mesh(guidePinGeo, materials.hardenedSteel);
  guidePinMesh.name = 'BoltGuidePin';
  guidePinMesh.castShadow = true;
  guidePinMesh.receiveShadow = true;
  group.add(guidePinMesh);

  // 3J. Hardened Key Guide Drill-Pin (projects forward from backplate into hollow key pipe)
  const drillPinGeo = makeClosedLatheZ([
    [-0.182, 0.026],
    [0.025, 0.026],
    [0.040, 0.018],
  ]);
  drillPinGeo.translate(KEY_X, KEY_Y, 0);
  disposables.push(drillPinGeo);
  const drillPinMesh = new THREE.Mesh(drillPinGeo, materials.hardenedSteel);
  drillPinMesh.name = 'KeyDrillPin';
  drillPinMesh.castShadow = true;
  group.add(drillPinMesh);

  // 3K. 4 Corner Case Pillars with Slotted Brass Cover Screws
  const cornerCoords: Array<[number, number, number]> = [
    [-0.785, 0.555, 0.42],
    [-0.785, -0.535, -0.68],
    [0.885, 0.555, 1.15],
    [0.885, -0.535, -0.25],
  ];
  for (let c = 0; c < cornerCoords.length; c++) {
    const [cx, cy, ang] = cornerCoords[c];
    const screw = buildSlottedBrassScrew(0.028, 0.014, 0.008, 0.007, materials.forendBrass, disposables);
    screw.rotation.z = ang;
    screw.position.set(cx, cy, 0.224);
    group.add(screw);
  }

  return {
    group,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 4. Heavy Square Blued-Steel Deadbolt, Slotted Tail, U-Talon & Hardened Stump
// =============================================================================

export interface DeadboltAssembly {
  group: THREE.Group;
  hitMeshes: THREE.Mesh[];
  dispose: () => void;
}

export function buildDeadboltAssembly(materials: BoltMaterials): DeadboltAssembly {
  const group = new THREE.Group();
  group.name = 'DeadboltAssembly';
  const disposables: THREE.BufferGeometry[] = [];
  const hitMeshes: THREE.Mesh[] = [];

  // 4A. Massive Chamfered Square Blued-Steel Deadbolt Head
  // Local coords at boltThrow01 = 0 (unlocked): x in [+0.51, +1.045], y in [STUMP_Y - 0.168, STUMP_Y + 0.168]
  const headMinX = 0.51;
  const headMaxX = 1.045;
  const headMinY = STUMP_Y - 0.164;
  const headMaxY = STUMP_Y + 0.164;
  const headDepth = BOLT_HEAD_Z_HALF * 2 - 0.024; // plus 2 * 0.012 bevel = exact BOLT_HEAD_Z_HALF * 2

  const headPts = ensureCCW([
    new THREE.Vector2(headMinX, headMinY + 0.012),
    new THREE.Vector2(headMinX + 0.012, headMinY),
    new THREE.Vector2(headMaxX - 0.024, headMinY),
    new THREE.Vector2(headMaxX, headMinY + 0.024),
    new THREE.Vector2(headMaxX, headMaxY - 0.024),
    new THREE.Vector2(headMaxX - 0.024, headMaxY),
    new THREE.Vector2(headMinX + 0.012, headMaxY),
    new THREE.Vector2(headMinX, headMaxY - 0.012),
  ]);
  const headGeo = new THREE.ExtrudeGeometry(new THREE.Shape(headPts), {
    depth: headDepth,
    bevelEnabled: true,
    bevelThickness: 0.012,
    bevelSize: 0.012,
    bevelSegments: 2,
  });
  headGeo.translate(0, 0, -headDepth * 0.5);
  disposables.push(headGeo);
  const headMesh = new THREE.Mesh(headGeo, materials.bluedSteelBolt);
  headMesh.name = 'DeadboltHead';
  headMesh.castShadow = true;
  headMesh.receiveShadow = true;
  group.add(headMesh);
  hitMeshes.push(headMesh);

  // Subtle hardened anti-saw steel roller inlays visible on the deadbolt front nose face
  for (const dy of [-0.075, 0.075]) {
    const rollerGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.24, 18);
    rollerGeo.rotateX(Math.PI / 2);
    rollerGeo.translate(headMaxX - 0.008, STUMP_Y + dy, 0);
    disposables.push(rollerGeo);
    const rollerMesh = new THREE.Mesh(rollerGeo, materials.hardenedSteel);
    rollerMesh.name = 'AntiSawRollerPin';
    rollerMesh.castShadow = true;
    group.add(rollerMesh);
  }

  // 4B. Machined Blued-Steel Bolt Tail (Shank) with Horizontal Guide Slot & U-Shaped Bolt Talon
  // Centered cleanly in Z behind Lever 0: [BOLT_TAIL_Z_BACK, BOLT_TAIL_Z_FRONT]
  const tailDepth = BOLT_TAIL_Z_FRONT - BOLT_TAIL_Z_BACK - 0.008;
  const tailZMid = (BOLT_TAIL_Z_BACK + BOLT_TAIL_Z_FRONT) * 0.5;

  // Bolt talon U-notch centered at x_talon_local = KEY_X - BOLT_TRAVEL * 0.5 = -0.13
  const talonCenterX = KEY_X - 0.17; // -0.13
  const notchHalfW = 0.048;
  const jawW = 0.082;
  const spineMinY = STUMP_Y - 0.085; // +0.075
  const spineMaxY = STUMP_Y + 0.085; // +0.245
  const jawBottomY = -0.018;
  const notchTopY = 0.102;

  const tailPts = ensureCCW([
    new THREE.Vector2(-0.765, spineMinY + 0.012),
    new THREE.Vector2(-0.753, spineMinY),
    // Left talon jaw outer edge
    new THREE.Vector2(talonCenterX - notchHalfW - jawW - 0.02, spineMinY),
    new THREE.Vector2(talonCenterX - notchHalfW - jawW, jawBottomY + 0.018),
    new THREE.Vector2(talonCenterX - notchHalfW - jawW + 0.018, jawBottomY),
    new THREE.Vector2(talonCenterX - notchHalfW - 0.012, jawBottomY),
    // U-notch left jaw inner wall
    new THREE.Vector2(talonCenterX - notchHalfW, jawBottomY + 0.014),
    new THREE.Vector2(talonCenterX - notchHalfW, notchTopY - 0.014),
    new THREE.Vector2(talonCenterX - notchHalfW + 0.014, notchTopY),
    // U-notch ceiling & right jaw inner wall
    new THREE.Vector2(talonCenterX + notchHalfW - 0.014, notchTopY),
    new THREE.Vector2(talonCenterX + notchHalfW, notchTopY - 0.014),
    new THREE.Vector2(talonCenterX + notchHalfW, jawBottomY + 0.014),
    // Right talon jaw bottom & outer edge
    new THREE.Vector2(talonCenterX + notchHalfW + 0.012, jawBottomY),
    new THREE.Vector2(talonCenterX + notchHalfW + jawW - 0.018, jawBottomY),
    new THREE.Vector2(talonCenterX + notchHalfW + jawW, jawBottomY + 0.018),
    new THREE.Vector2(talonCenterX + notchHalfW + jawW + 0.02, spineMinY),
    // Meet bolt head shoulder
    new THREE.Vector2(headMinX + 0.03, spineMinY),
    new THREE.Vector2(headMinX + 0.03, spineMaxY),
    new THREE.Vector2(-0.753, spineMaxY),
    new THREE.Vector2(-0.765, spineMaxY - 0.012),
  ]);

  const tailShape = new THREE.Shape(tailPts);

  // Horizontal guide slot riding over fixed steel guide pin (GUIDE_PIN_X = -0.34)
  // At boltThrow=0 pin is at local -0.34; at boltThrow=1 pin is at local -0.68.
  const slotPts = makeRoundedRectPts(-0.718, STUMP_Y - 0.034, -0.302, STUMP_Y + 0.034, 0.032, 6);
  tailShape.holes.push(new THREE.Path(ensureCW(slotPts)));

  const tailGeo = new THREE.ExtrudeGeometry(tailShape, {
    depth: tailDepth,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
  });
  tailGeo.translate(0, 0, tailZMid - tailDepth * 0.5);
  disposables.push(tailGeo);
  const tailMesh = new THREE.Mesh(tailGeo, materials.bluedSteelBolt);
  tailMesh.name = 'DeadboltTailAndTalon';
  tailMesh.castShadow = true;
  tailMesh.receiveShadow = true;
  group.add(tailMesh);
  hitMeshes.push(tailMesh);

  // 4C. Hardened Square Steel Bolt Stump (riveted to bolt tail, projecting through ALL 5 levers!)
  const stumpDepth = STUMP_Z_FRONT - STUMP_Z_BACK - 0.008;
  const stumpPts = makeRoundedRectPts(
    STUMP_X_UNLOCKED - STUMP_WIDTH * 0.5,
    STUMP_Y - STUMP_HEIGHT * 0.5,
    STUMP_X_UNLOCKED + STUMP_WIDTH * 0.5,
    STUMP_Y + STUMP_HEIGHT * 0.5,
    0.008,
    3
  );
  const stumpGeo = new THREE.ExtrudeGeometry(new THREE.Shape(stumpPts), {
    depth: stumpDepth,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
  });
  stumpGeo.translate(0, 0, STUMP_Z_BACK + 0.004);
  disposables.push(stumpGeo);
  const stumpMesh = new THREE.Mesh(stumpGeo, materials.hardenedSteel);
  stumpMesh.name = 'BoltStump';
  stumpMesh.castShadow = true;
  stumpMesh.receiveShadow = true;
  group.add(stumpMesh);
  hitMeshes.push(stumpMesh);

  // Brass peened rivet boss on front face of hardened bolt stump
  const rivetGeo = makeClosedLatheZ([
    [STUMP_Z_FRONT - 0.002, 0.022],
    [STUMP_Z_FRONT + 0.008, 0.020],
    [STUMP_Z_FRONT + 0.013, 0.012],
  ]);
  rivetGeo.translate(STUMP_X_UNLOCKED, STUMP_Y, 0);
  disposables.push(rivetGeo);
  const rivetMesh = new THREE.Mesh(rivetGeo, materials.forendBrass);
  rivetMesh.name = 'BoltStumpRivet';
  rivetMesh.castShadow = true;
  group.add(rivetMesh);
  hitMeshes.push(rivetMesh);

  return {
    group,
    hitMeshes,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 5. Five Stacked Precision Brass Lever Tumblers (with H-Gates & Anti-Pick Serrations)
// =============================================================================

export interface LeverPackAssembly {
  group: THREE.Group;
  leverPivots: THREE.Group[];
  leverMeshes: THREE.Mesh[];
  dispose: () => void;
}

/**
 * Constructs one precision-machined Chubb brass lever plate in lever-local coordinates
 * (pivot bore centered at local (0, 0), plate thickness centered at Z = 0).
 */
export function createLeverGeometry(index: number): THREE.BufferGeometry {
  const spec = LEVER_GATE_SPECS[index];

  // Outer lever contour (CCW):
  // - Rounded pivot ear around (0, 0) of radius 0.125
  // - Sculpted upper spring-cam shoulder rising toward the gating head
  // - Solid right locking cheek & rest heel sitting on REST_PIN
  // - Concave Chubb circular-arc key belly directly above KEY_X
  const outerPts: THREE.Vector2[] = [];

  // Left semicircular pivot ear: angle from +PI/2 (top) around PI (left) to -PI/2 (bottom)
  const earR = 0.122;
  const earSegs = 14;
  for (let s = 0; s <= earSegs; s++) {
    const a = Math.PI * 0.5 + Math.PI * (s / earSegs);
    outerPts.push(new THREE.Vector2(Math.cos(a) * earR, Math.sin(a) * earR));
  }

  // Bottom throat leading into concave Chubb key-belly cam arch
  outerPts.push(new THREE.Vector2(0.24, -0.132));
  outerPts.push(new THREE.Vector2(0.36, -0.158));

  // Concave key belly arch peaking at (LEVER_BELLY_LOCAL_X = 0.62, LEVER_BELLY_LOCAL_Y = -0.115)
  const bellySegs = 12;
  for (let s = 0; s <= bellySegs; s++) {
    const t = s / bellySegs; // 0..1 across x in [0.43, 0.81]
    const bx = 0.43 + t * 0.38;
    const u = (bx - LEVER_BELLY_LOCAL_X) / 0.19; // -1..+1
    const by = LEVER_BELLY_LOCAL_Y - 0.044 * u * u;
    outerPts.push(new THREE.Vector2(bx, by));
  }

  // Lower-right rest heel seated cleanly on brass rest stop pin at local (0.985, -0.163)
  outerPts.push(new THREE.Vector2(0.89, -0.163));
  outerPts.push(new THREE.Vector2(0.995, -0.163));
  outerPts.push(new THREE.Vector2(1.015, -0.142));

  // Right vertical locking cheek edge (clears bolt head at +0.51 -> local +1.09 by 0.075!)
  outerPts.push(new THREE.Vector2(1.015, 0.236));
  outerPts.push(new THREE.Vector2(0.988, 0.262));

  // Sculpted upper spring-cam crest with subtle locksmith assembly step per lever (0..4)
  // so all 5 stacked brass lever plates are visibly distinct even at rest
  const shoulderStepX = 0.24 + index * 0.036;
  outerPts.push(new THREE.Vector2(0.42, 0.262));
  outerPts.push(new THREE.Vector2(shoulderStepX + 0.035, 0.232));
  outerPts.push(new THREE.Vector2(shoulderStepX, 0.198));
  outerPts.push(new THREE.Vector2(0.11, 0.152));

  const shape = new THREE.Shape(ensureCCW(outerPts));

  // Hole 1: Precision circular pivot bore
  shape.holes.push(makeCircleHole(0, 0, 0.058, 24));

  // Hole 2: Authentic Chubb H-Gate aperture with exact sloped gate slot & anti-pick false serrations
  const xLOut = 0.442;
  const xLIn = 0.562;
  const xRIn = 0.782;
  const xROut = 0.896;

  // Interpolate exact tilted gate center Y at left & right gate bridge mouths
  const spanX = Math.max(1e-4, spec.gateLockedLocalX - spec.gateUnlockedLocalX);
  const slopeY = (spec.gateLockedLocalY - spec.gateUnlockedLocalY) / spanX;
  const yGateL = spec.gateUnlockedLocalY + (xLIn - spec.gateUnlockedLocalX) * slopeY;
  const yGateR = spec.gateUnlockedLocalY + (xRIn - spec.gateUnlockedLocalX) * slopeY;

  const yTopPocket = 0.224;
  const yBotPocketL = Math.max(-0.088, Math.min(yGateL - GATE_HALF_HEIGHT - 0.014, 0.01));
  const yBotPocketR = Math.max(-0.120, Math.min(yGateR - GATE_HALF_HEIGHT - 0.012, -0.02));

  const gateTopL = yGateL + GATE_HALF_HEIGHT;
  const gateBotL = yGateL - GATE_HALF_HEIGHT;
  const gateTopR = yGateR + GATE_HALF_HEIGHT;
  const gateBotR = yGateR - GATE_HALF_HEIGHT;

  // Trace H-gate cavity perimeter in strict CCW first, then convert to CW hole cleanly
  const hPts: THREE.Vector2[] = [
    new THREE.Vector2(xLOut, yTopPocket),
    new THREE.Vector2(xLOut, yBotPocketL),
    new THREE.Vector2(xLIn, yBotPocketL),
    new THREE.Vector2(xLIn, gateBotL),
    new THREE.Vector2(xRIn, gateBotR),
    new THREE.Vector2(xRIn, yBotPocketR),
    new THREE.Vector2(xROut, yBotPocketR),
    new THREE.Vector2(xROut, yTopPocket),
    new THREE.Vector2(xRIn, yTopPocket),
  ];

  // Add a crisp machined anti-pick false serration tooth on the upper right locking cheek if vertical margin allows
  if (yTopPocket - gateTopR > 0.068) {
    const midY = gateTopR + (yTopPocket - gateTopR) * 0.52;
    hPts.push(new THREE.Vector2(xRIn, midY + 0.012));
    hPts.push(new THREE.Vector2(xRIn - 0.020, midY + 0.012));
    hPts.push(new THREE.Vector2(xRIn - 0.020, midY - 0.012));
    hPts.push(new THREE.Vector2(xRIn, midY - 0.012));
  }

  hPts.push(new THREE.Vector2(xRIn, gateTopR));
  hPts.push(new THREE.Vector2(xLIn, gateTopL));

  // Add a matching anti-pick false serration tooth on the upper left locking cheek if vertical margin allows
  if (yTopPocket - gateTopL > 0.068) {
    const midY = gateTopL + (yTopPocket - gateTopL) * 0.52;
    hPts.push(new THREE.Vector2(xLIn, midY - 0.012));
    hPts.push(new THREE.Vector2(xLIn + 0.020, midY - 0.012));
    hPts.push(new THREE.Vector2(xLIn + 0.020, midY + 0.012));
    hPts.push(new THREE.Vector2(xLIn, midY + 0.012));
  }

  hPts.push(new THREE.Vector2(xLIn, yTopPocket));

  const cwHole = ensureCW(hPts);
  const gateHolePath = new THREE.Path();
  gateHolePath.moveTo(cwHole[0].x, cwHole[0].y);
  for (let i = 1; i < cwHole.length; i++) gateHolePath.lineTo(cwHole[i].x, cwHole[i].y);
  gateHolePath.closePath();
  shape.holes.push(gateHolePath);

  const coreDepth = LEVER_THICKNESS - LEVER_BEVEL * 2;
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: LEVER_BEVEL,
    bevelSize: LEVER_BEVEL,
    bevelSegments: 2,
    curveSegments: 20,
  });
  geo.translate(0, 0, -coreDepth * 0.5);
  geo.computeVertexNormals();
  return geo;
}

export function buildLeverPackAssembly(materials: BoltMaterials): LeverPackAssembly {
  const group = new THREE.Group();
  group.name = 'LeverPackAssembly';
  const leverPivots: THREE.Group[] = [];
  const leverMeshes: THREE.Mesh[] = [];
  const disposables: THREE.BufferGeometry[] = [];

  for (let i = 0; i < LEVER_COUNT; i++) {
    const pivotGroup = new THREE.Group();
    pivotGroup.name = `LeverPivot_${i}`;
    pivotGroup.position.set(PIVOT_X, PIVOT_Y, getLeverZ(i));

    const geo = createLeverGeometry(i);
    disposables.push(geo);

    const mesh = new THREE.Mesh(geo, materials.leverMaterials[i]);
    mesh.name = `BrassLever_${i}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.leverIndex = i;

    pivotGroup.add(mesh);
    group.add(pivotGroup);

    leverPivots.push(pivotGroup);
    leverMeshes.push(mesh);
  }

  return {
    group,
    leverPivots,
    leverMeshes,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 6. Five Dynamically Flexing Curved Phosphor-Bronze Leaf Springs (Zero Per-Frame GC)
// =============================================================================

const SPRING_SEGMENTS = 22;
const SPRING_HALF_THICK_Z = 0.0085;
const SPRING_HALF_HEIGHT_N = 0.0085;
// Contact point on top edge of lever in lever-local coordinates
const SPRING_CONTACT_LOCAL_X = 0.56;
const SPRING_CONTACT_LOCAL_Y = 0.264;

export class LeafSpringMesh {
  readonly index: number;
  readonly mesh: THREE.Mesh;
  readonly geometry: THREE.BufferGeometry;

  private readonly positions: Float32Array;
  private readonly normals: Float32Array;
  private readonly cx = new Float32Array(SPRING_SEGMENTS + 1);
  private readonly cy = new Float32Array(SPRING_SEGMENTS + 1);
  private readonly nx = new Float32Array(SPRING_SEGMENTS + 1);
  private readonly ny = new Float32Array(SPRING_SEGMENTS + 1);
  private lastAngle = -999;

  constructor(index: number, material: THREE.Material) {
    this.index = index;
    this.geometry = new THREE.BufferGeometry();

    // Closed watertight extruded rectangular spring blade:
    //  - 4 longitudinal faces * SPRING_SEGMENTS * 2 tris = 8 * SPRING_SEGMENTS tris
    //  - 2 end caps * 2 tris = 4 tris
    // Use non-indexed hard-faceted strip (crisp beveled metallic ribbon shading & exact normals!)
    const totalTris = SPRING_SEGMENTS * 8 + 4;
    this.positions = new Float32Array(totalTris * 9);
    this.normals = new Float32Array(totalTris * 9);

    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3));

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.name = `PhosphorBronzeLeafSpring_${index}`;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;

    this.update(0);
  }

  /**
   * Recomputes the Euler-Bernoulli curved phosphor-bronze leaf spring in-place (zero allocations)
   * anchored at the fixed steel spring abutment post and pressing tangentially onto lever `index`.
   */
  update(leverAngleRad: number): void {
    if (Math.abs(leverAngleRad - this.lastAngle) < 1e-5) return;
    this.lastAngle = leverAngleRad;

    const zCenter = getLeverZ(this.index);
    const zBack = zCenter - SPRING_HALF_THICK_Z;
    const zFront = zCenter + SPRING_HALF_THICK_Z;

    // Anchor root wraps cleanly off the right flank of the fixed steel spring abutment post
    const p0x = SPRING_POST_X + 0.022;
    const p0y = SPRING_POST_Y + 0.036;

    // Sliding contact foot on top shoulder of lever `index`
    const cosA = Math.cos(leverAngleRad);
    const sinA = Math.sin(leverAngleRad);
    const p3x = PIVOT_X + SPRING_CONTACT_LOCAL_X * cosA - SPRING_CONTACT_LOCAL_Y * sinA;
    const p3y = PIVOT_Y + SPRING_CONTACT_LOCAL_X * sinA + SPRING_CONTACT_LOCAL_Y * cosA + SPRING_HALF_HEIGHT_N;

    // Cubic cantilever deflection control points (bows gracefully upward as lever compresses spring!)
    const liftRatio = Math.max(0, Math.min(1, leverAngleRad / 0.35));
    const bowLift = 0.045 + liftRatio * 0.088;
    const p1x = p0x + 0.22;
    const p1y = p0y + bowLift;
    const p2x = p3x - 0.16;
    const p2y = p3y + 0.068 + liftRatio * 0.025;

    // Sample centerline & unit 2D upward normals
    for (let s = 0; s <= SPRING_SEGMENTS; s++) {
      const t = s / SPRING_SEGMENTS;
      const it = 1 - t;
      const b0 = it * it * it;
      const b1 = 3 * it * it * t;
      const b2 = 3 * it * t * t;
      const b3 = t * t * t;

      this.cx[s] = b0 * p0x + b1 * p1x + b2 * p2x + b3 * p3x;
      this.cy[s] = b0 * p0y + b1 * p1y + b2 * p2y + b3 * p3y;

      const tx = 3 * it * it * (p1x - p0x) + 6 * it * t * (p2x - p1x) + 3 * t * t * (p3x - p2x);
      const ty = 3 * it * it * (p1y - p0y) + 6 * it * t * (p2y - p1y) + 3 * t * t * (p3y - p2y);
      const len = Math.max(1e-5, Math.hypot(tx, ty));
      // Rotate tangent (+tx, +ty) by +90 deg CCW -> (-ty, +tx) points UPWARD (+Y when tx > 0)
      this.nx[s] = -ty / len;
      this.ny[s] = tx / len;
    }

    let ptr = 0;
    const writeQuad = (
      ax: number,
      ay: number,
      az: number,
      bx: number,
      by: number,
      bz: number,
      cx: number,
      cy: number,
      cz: number,
      dx: number,
      dy: number,
      dz: number,
      fnx: number,
      fny: number,
      fnz: number
    ) => {
      // Tri 1: a -> b -> c (outward CCW)
      this.positions[ptr] = ax;
      this.positions[ptr + 1] = ay;
      this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = bx;
      this.positions[ptr + 4] = by;
      this.positions[ptr + 5] = bz;
      this.positions[ptr + 6] = cx;
      this.positions[ptr + 7] = cy;
      this.positions[ptr + 8] = cz;

      this.normals[ptr] = fnx;
      this.normals[ptr + 1] = fny;
      this.normals[ptr + 2] = fnz;
      this.normals[ptr + 3] = fnx;
      this.normals[ptr + 4] = fny;
      this.normals[ptr + 5] = fnz;
      this.normals[ptr + 6] = fnx;
      this.normals[ptr + 7] = fny;
      this.normals[ptr + 8] = fnz;
      ptr += 9;

      // Tri 2: a -> c -> d
      this.positions[ptr] = ax;
      this.positions[ptr + 1] = ay;
      this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = cx;
      this.positions[ptr + 4] = cy;
      this.positions[ptr + 5] = cz;
      this.positions[ptr + 6] = dx;
      this.positions[ptr + 7] = dy;
      this.positions[ptr + 8] = dz;

      this.normals[ptr] = fnx;
      this.normals[ptr + 1] = fny;
      this.normals[ptr + 2] = fnz;
      this.normals[ptr + 3] = fnx;
      this.normals[ptr + 4] = fny;
      this.normals[ptr + 5] = fnz;
      this.normals[ptr + 6] = fnx;
      this.normals[ptr + 7] = fny;
      this.normals[ptr + 8] = fnz;
      ptr += 9;
    };

    for (let s = 0; s < SPRING_SEGMENTS; s++) {
      // Slight elegant taper from thick root to nimble contact foot
      const h0 = SPRING_HALF_HEIGHT_N * (1.15 - 0.30 * (s / SPRING_SEGMENTS));
      const h1 = SPRING_HALF_HEIGHT_N * (1.15 - 0.30 * ((s + 1) / SPRING_SEGMENTS));

      const top0x = this.cx[s] + this.nx[s] * h0;
      const top0y = this.cy[s] + this.ny[s] * h0;
      const bot0x = this.cx[s] - this.nx[s] * h0;
      const bot0y = this.cy[s] - this.ny[s] * h0;

      const top1x = this.cx[s + 1] + this.nx[s + 1] * h1;
      const top1y = this.cy[s + 1] + this.ny[s + 1] * h1;
      const bot1x = this.cx[s + 1] - this.nx[s + 1] * h1;
      const bot1y = this.cy[s + 1] - this.ny[s + 1] * h1;

      const mnx = (this.nx[s] + this.nx[s + 1]) * 0.5;
      const mny = (this.ny[s] + this.ny[s + 1]) * 0.5;
      const mnLen = Math.max(1e-5, Math.hypot(mnx, mny));
      const unx = mnx / mnLen;
      const uny = mny / mnLen;

      // Front face (+Z outward): bot0 -> bot1 -> top1 -> top0
      writeQuad(
        bot0x, bot0y, zFront,
        bot1x, bot1y, zFront,
        top1x, top1y, zFront,
        top0x, top0y, zFront,
        0, 0, 1
      );
      // Back face (-Z outward): bot1 -> bot0 -> top0 -> top1
      writeQuad(
        bot1x, bot1y, zBack,
        bot0x, bot0y, zBack,
        top0x, top0y, zBack,
        top1x, top1y, zBack,
        0, 0, -1
      );
      // Top face (+N outward): top0F -> top1F -> top1B -> top0B
      writeQuad(
        top0x, top0y, zFront,
        top1x, top1y, zFront,
        top1x, top1y, zBack,
        top0x, top0y, zBack,
        unx, uny, 0
      );
      // Bottom face (-N outward): bot0B -> bot1B -> bot1F -> bot0F
      writeQuad(
        bot0x, bot0y, zBack,
        bot1x, bot1y, zBack,
        bot1x, bot1y, zFront,
        bot0x, bot0y, zFront,
        -unx, -uny, 0
      );
    }

    // Start cap (s = 0, facing -tangent)
    const hStart = SPRING_HALF_HEIGHT_N * 1.15;
    const t0x = this.cx[0] + this.nx[0] * hStart;
    const t0y = this.cy[0] + this.ny[0] * hStart;
    const b0x = this.cx[0] - this.nx[0] * hStart;
    const b0y = this.cy[0] - this.ny[0] * hStart;
    writeQuad(
      b0x, b0y, zBack,
      b0x, b0y, zFront,
      t0x, t0y, zFront,
      t0x, t0y, zBack,
      -this.ny[0], this.nx[0], 0
    );

    // End cap (s = SPRING_SEGMENTS, facing +tangent)
    const hEnd = SPRING_HALF_HEIGHT_N * 0.85;
    const e = SPRING_SEGMENTS;
    const tEx = this.cx[e] + this.nx[e] * hEnd;
    const tEy = this.cy[e] + this.ny[e] * hEnd;
    const bEx = this.cx[e] - this.nx[e] * hEnd;
    const bEy = this.cy[e] - this.ny[e] * hEnd;
    writeQuad(
      bEx, bEy, zFront,
      bEx, bEy, zBack,
      tEx, tEy, zBack,
      tEx, tEy, zFront,
      this.ny[e], -this.nx[e], 0
    );

    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.normal as THREE.BufferAttribute).needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

export interface SpringPackAssembly {
  group: THREE.Group;
  springs: LeafSpringMesh[];
  update: (leverAnglesRad: Float64Array) => void;
  dispose: () => void;
}

export function buildSpringPackAssembly(materials: BoltMaterials): SpringPackAssembly {
  const group = new THREE.Group();
  group.name = 'LeafSpringPackAssembly';
  const springs: LeafSpringMesh[] = [];
  const disposables: THREE.BufferGeometry[] = [];

  // Curled bronze retaining anchor sleeves wrapped around the steel spring post
  const collarShape = new THREE.Shape();
  collarShape.absarc(0, 0, 0.062, 0, Math.PI * 2, false);
  collarShape.holes.push(makeCircleHole(0, 0, 0.045, 20));

  for (let i = 0; i < LEVER_COUNT; i++) {
    const spring = new LeafSpringMesh(i, materials.phosphorBronze);
    springs.push(spring);
    group.add(spring.mesh);

    const cDepth = SPRING_HALF_THICK_Z * 2;
    const cGeo = new THREE.ExtrudeGeometry(collarShape, {
      depth: cDepth,
      bevelEnabled: true,
      bevelThickness: 0.001,
      bevelSize: 0.001,
      bevelSegments: 1,
      curveSegments: 20,
    });
    cGeo.translate(SPRING_POST_X, SPRING_POST_Y, getLeverZ(i) - cDepth * 0.5);
    disposables.push(cGeo);
    const cMesh = new THREE.Mesh(cGeo, materials.phosphorBronze);
    cMesh.name = `SpringAnchorEye_${i}`;
    cMesh.castShadow = true;
    group.add(cMesh);
  }

  return {
    group,
    springs,
    update: (leverAnglesRad: Float64Array) => {
      for (let i = 0; i < LEVER_COUNT; i++) {
        springs[i].update(leverAnglesRad[i]);
      }
    },
    dispose: () => {
      for (const s of springs) s.dispose();
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 7. Ornate 19th-Century Victorian 5-Bitted Warded Mortise Key
// =============================================================================

export interface KeyAssembly {
  group: THREE.Group;
  hitMeshes: THREE.Mesh[];
  dispose: () => void;
}

export function buildKeyAssembly(materials: BoltMaterials): KeyAssembly {
  const group = new THREE.Group();
  group.name = 'VictorianBitKeyAssembly';
  group.position.set(KEY_X, KEY_Y, 0);
  const disposables: THREE.BufferGeometry[] = [];
  const hitMeshes: THREE.Mesh[] = [];

  // 7A. Turned Hardened-Steel Pipe Barrel with Victorian Collar Mouldings (z: -0.17 .. +0.58)
  const shaftGeo = makeClosedLatheZ(
    [
      [-0.170, 0.044],
      [0.236, 0.044],
      [0.246, 0.060],
      [0.268, 0.068],
      [0.288, 0.068],
      [0.302, 0.052],
      [0.316, 0.045],
      [0.455, 0.045],
      [0.478, 0.062],
      [0.496, 0.066],
      [0.514, 0.052],
      [0.545, 0.048],
      [0.575, 0.064],
    ],
    28
  );
  disposables.push(shaftGeo);
  const shaftMesh = new THREE.Mesh(shaftGeo, materials.hardenedSteel);
  shaftMesh.name = 'KeyPipeShaft';
  shaftMesh.castShadow = true;
  shaftMesh.receiveShadow = true;
  group.add(shaftMesh);
  hitMeshes.push(shaftMesh);

  // 7B. Precision 6-Step Key Bit extending radially downward (-Y at phi = 0)
  // Rear Talon Drive Step (z: -0.152 .. -0.082, aligned with bolt tail talon!)
  const talonBitW = 0.036;
  const talonBitDepth = 0.064;
  const talonShape = new THREE.Shape(
    makeRoundedRectPts(-talonBitW * 0.5, -KEY_TALON_RADIUS, talonBitW * 0.5, -0.032, 0.008, 3)
  );
  const talonGeo = new THREE.ExtrudeGeometry(talonShape, {
    depth: talonBitDepth,
    bevelEnabled: true,
    bevelThickness: 0.003,
    bevelSize: 0.003,
    bevelSegments: 2,
  });
  talonGeo.translate(0, 0, -0.118 - talonBitDepth * 0.5);
  disposables.push(talonGeo);
  const talonMesh = new THREE.Mesh(talonGeo, materials.hardenedSteel);
  talonMesh.name = 'KeyTalonDriveBit';
  talonMesh.castShadow = true;
  talonMesh.receiveShadow = true;
  group.add(talonMesh);
  hitMeshes.push(talonMesh);

  // 5 Stepped Lever Bitting Teeth (exact KEY_BIT_RADII[0..4] aligned 1:1 with lever[0..4] Z-planes!)
  const bitW = 0.034;
  const stepZWidth = 0.024;
  for (let i = 0; i < LEVER_COUNT; i++) {
    const rBit = KEY_BIT_RADII[i];
    const stepShape = new THREE.Shape(
      makeRoundedRectPts(-bitW * 0.5, -rBit, bitW * 0.5, -0.034, 0.006, 3)
    );
    const stepGeo = new THREE.ExtrudeGeometry(stepShape, {
      depth: stepZWidth,
      bevelEnabled: true,
      bevelThickness: 0.0022,
      bevelSize: 0.0022,
      bevelSegments: 2,
    });
    stepGeo.translate(0, 0, getLeverZ(i) - stepZWidth * 0.5);
    disposables.push(stepGeo);
    const stepMesh = new THREE.Mesh(stepGeo, materials.keyBrass);
    stepMesh.name = `KeyBittingStep_${i}`;
    stepMesh.castShadow = true;
    stepMesh.receiveShadow = true;
    group.add(stepMesh);
    hitMeshes.push(stepMesh);
  }

  // Connecting warded key-bit bridge web along Z just below the shaft (narrower than brass steps -> zero z-fighting)
  const webGeo = new THREE.BoxGeometry(0.022, 0.082, 0.285);
  webGeo.translate(0, -0.072, 0.006);
  disposables.push(webGeo);
  const webMesh = new THREE.Mesh(webGeo, materials.hardenedSteel);
  webMesh.name = 'KeyBitWeb';
  webMesh.castShadow = true;
  group.add(webMesh);
  hitMeshes.push(webMesh);

  // 7C. Ornate Sculpted Victorian Satin-Brass Oval Bow (Handle)
  // Built cleanly with architectural cubic Bezier curves in (u -> Z, v -> Y), extruded in X
  const bowShape = new THREE.Shape();
  bowShape.moveTo(0.56, -0.068);
  bowShape.bezierCurveTo(0.61, -0.175, 0.72, -0.245, 0.825, -0.232);
  bowShape.bezierCurveTo(0.935, -0.218, 1.005, -0.125, 1.015, 0.0);
  bowShape.bezierCurveTo(1.005, 0.125, 0.935, 0.218, 0.825, 0.232);
  bowShape.bezierCurveTo(0.72, 0.245, 0.61, 0.175, 0.56, 0.068);
  bowShape.closePath();

  // Large central oval thumb aperture inside the brass bow
  const bowHolePts: THREE.Vector2[] = [];
  for (let s = 0; s < 36; s++) {
    const a = (s / 36) * Math.PI * 2;
    const u = 0.805 + Math.cos(a) * 0.132;
    const v = Math.sin(a) * 0.138;
    bowHolePts.push(new THREE.Vector2(u, v));
  }
  bowShape.holes.push(new THREE.Path(ensureCW(bowHolePts)));

  // Flanking decorative upper & lower Victorian eyelet piercings near the bow shoulder
  bowShape.holes.push(makeCircleHole(0.648, 0.092, 0.028, 16));
  bowShape.holes.push(makeCircleHole(0.648, -0.092, 0.028, 16));

  const bowThickX = 0.038;
  const bowGeo = new THREE.ExtrudeGeometry(bowShape, {
    depth: bowThickX,
    bevelEnabled: true,
    bevelThickness: 0.012,
    bevelSize: 0.011,
    bevelSegments: 3,
    curveSegments: 24,
  });
  // Center extrusion in Z, then rotate -PI/2 around Y so local +X(u) -> +Z, local +Z(thick) -> +X!
  bowGeo.translate(0, 0, -bowThickX * 0.5);
  bowGeo.rotateY(-Math.PI / 2);
  bowGeo.computeVertexNormals();
  disposables.push(bowGeo);

  const bowMesh = new THREE.Mesh(bowGeo, materials.keyBrass);
  bowMesh.name = 'KeyOrnateBow';
  bowMesh.castShadow = true;
  bowMesh.receiveShadow = true;
  group.add(bowMesh);
  hitMeshes.push(bowMesh);

  // Decorative brass collar band where the ornate bow joins the turned steel stem
  const neckCollarGeo = makeClosedLatheZ([
    [0.525, 0.056],
    [0.552, 0.074],
    [0.580, 0.068],
  ]);
  disposables.push(neckCollarGeo);
  const neckCollarMesh = new THREE.Mesh(neckCollarGeo, materials.keyBrass);
  neckCollarMesh.name = 'KeyBowNeckCollar';
  neckCollarMesh.castShadow = true;
  group.add(neckCollarMesh);
  hitMeshes.push(neckCollarMesh);

  // Generous invisible grab-target proxy around the ornate key bow so dragging/clicking feels effortless
  const hitProxyGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.54, 16);
  hitProxyGeo.rotateX(Math.PI / 2);
  hitProxyGeo.translate(0, 0, 0.72);
  disposables.push(hitProxyGeo);
  const hitProxyMat = new THREE.MeshBasicMaterial({ visible: false });
  const hitProxyMesh = new THREE.Mesh(hitProxyGeo, hitProxyMat);
  hitProxyMesh.name = 'KeyBowHitProxy';
  group.add(hitProxyMesh);
  hitMeshes.push(hitProxyMesh);

  return {
    group,
    hitMeshes,
    dispose: () => {
      for (const g of disposables) g.dispose();
      hitProxyMat.dispose();
    },
  };
}
