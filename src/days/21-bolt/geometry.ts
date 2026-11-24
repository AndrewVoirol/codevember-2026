import * as THREE from 'three/webgpu';
import {
  BOLT_HEAD_Z_HALF,
  BOLT_TAIL_Z_BACK,
  BOLT_TAIL_Z_FRONT,
  CASE_INNER_MAX_X,
  CASE_INNER_MAX_Y,
  CASE_INNER_MIN_X,
  CASE_INNER_MIN_Y,
  CUTAWAY_FRONT_Z,
  CUTAWAY_WIN_MAX_X,
  CUTAWAY_WIN_MAX_Y,
  CUTAWAY_WIN_MIN_X,
  CUTAWAY_WIN_MIN_Y,
  FOREND_THICKNESS,
  FOREND_X,
  GATE_HALF_HEIGHT,
  GATE_POCKET_BOT_R_Y,
  GATE_POCKET_LEFT_INNER_X,
  GATE_POCKET_LEFT_OUTER_X,
  GATE_POCKET_RIGHT_INNER_X,
  GATE_POCKET_RIGHT_OUTER_X,
  GATE_POCKET_TOP_Y,
  GUIDE_PIN_X,
  GUIDE_PIN_Y,
  KEY_BIT_HALF_WIDTH,
  KEY_BIT_RADII,
  KEY_TALON_RADIUS,
  KEY_X,
  KEY_Y,
  LEVER_BELLY_POLYLINES,
  LEVER_BEVEL,
  LEVER_COUNT,
  LEVER_GATE_SPECS,
  LEVER_MAX_RAD,
  LEVER_THICKNESS,
  PIVOT_X,
  PIVOT_Y,
  REST_PIN_RADIUS,
  REST_PIN_X,
  REST_PIN_Y,
  SPRING_POST_X,
  SPRING_POST_Y,
  SPRING_REACTION_PIN_R,
  SPRING_REACTION_PIN_X,
  SPRING_REACTION_PIN_Y,
  STUMP_HEIGHT,
  STUMP_WIDTH,
  STUMP_X_UNLOCKED,
  STUMP_Y,
  STUMP_Z_BACK,
  STUMP_Z_FRONT,
  UPPER_RACK_SPEC,
  buildInvoluteRackPolygon2D,
  getLeverZ,
} from './lockMechanism';
import type { BoltMaterials } from './materials';

const CREASE_ANGLE_RAD = (34 * Math.PI) / 180;

export function ensureCCW(pts: THREE.Vector2[]): THREE.Vector2[] {
  if (THREE.ShapeUtils.isClockWise(pts)) {
    return [...pts].reverse();
  }
  return pts;
}

export function ensureCW(pts: THREE.Vector2[]): THREE.Vector2[] {
  if (!THREE.ShapeUtils.isClockWise(pts)) {
    return [...pts].reverse();
  }
  return pts;
}

/**
 * Computes precision machined normals for extruded parts (before any final orientation rotation):
 *  - Planar front/back extrusion cap faces (|nz| > 0.996) receive razor-flat (0, 0, ±1) normals
 *  - Curved bevels/walls smooth across adjacent non-cap faces within `creaseAngleRad`
 */
export function applyMachinedNormals(
  rawGeo: THREE.BufferGeometry,
  creaseAngleRad = CREASE_ANGLE_RAD
): THREE.BufferGeometry {
  const geo = rawGeo.index ? rawGeo.toNonIndexed() : rawGeo;
  if (geo !== rawGeo) rawGeo.dispose();

  const posAttr = geo.attributes.position as THREE.BufferAttribute;
  const vertCount = posAttr.count;
  const triCount = Math.floor(vertCount / 3);
  const normals = new Float32Array(vertCount * 3);

  const faceNx = new Float32Array(triCount);
  const faceNy = new Float32Array(triCount);
  const faceNz = new Float32Array(triCount);
  const faceArea = new Float32Array(triCount);
  const isCap = new Uint8Array(triCount);

  const vertBuckets = new Map<string, number[]>();
  const quantKey = (x: number, y: number, z: number) =>
    `${Math.round(x * 1e4)},${Math.round(y * 1e4)},${Math.round(z * 1e4)}`;

  for (let t = 0; t < triCount; t++) {
    const i0 = t * 3;
    const ax = posAttr.getX(i0);
    const ay = posAttr.getY(i0);
    const az = posAttr.getZ(i0);
    const bx = posAttr.getX(i0 + 1);
    const by = posAttr.getY(i0 + 1);
    const bz = posAttr.getZ(i0 + 1);
    const cx = posAttr.getX(i0 + 2);
    const cy = posAttr.getY(i0 + 2);
    const cz = posAttr.getZ(i0 + 2);

    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const acx = cx - ax;
    const acy = cy - ay;
    const acz = cz - az;

    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const len = Math.hypot(nx, ny, nz);
    const invLen = len > 1e-12 ? 1 / len : 0;
    const unx = nx * invLen;
    const uny = ny * invLen;
    const unz = nz * invLen;

    faceNx[t] = unx;
    faceNy[t] = uny;
    faceNz[t] = unz;
    faceArea[t] = len * 0.5;

    if (Math.abs(unz) > 0.996) {
      isCap[t] = unz > 0 ? 1 : 2;
      const capZ = unz > 0 ? 1 : -1;
      for (let k = 0; k < 3; k++) {
        const off = (i0 + k) * 3;
        normals[off] = 0;
        normals[off + 1] = 0;
        normals[off + 2] = capZ;
      }
    } else {
      for (let k = 0; k < 3; k++) {
        const vi = i0 + k;
        const key = quantKey(posAttr.getX(vi), posAttr.getY(vi), posAttr.getZ(vi));
        let bucket = vertBuckets.get(key);
        if (!bucket) {
          bucket = [];
          vertBuckets.set(key, bucket);
        }
        bucket.push(t);
      }
    }
  }

  const cosThreshold = Math.cos(creaseAngleRad);
  for (let t = 0; t < triCount; t++) {
    if (isCap[t] !== 0) continue;
    const fnx = faceNx[t];
    const fny = faceNy[t];
    const fnz = faceNz[t];
    const i0 = t * 3;

    for (let k = 0; k < 3; k++) {
      const vi = i0 + k;
      const key = quantKey(posAttr.getX(vi), posAttr.getY(vi), posAttr.getZ(vi));
      const neighbors = vertBuckets.get(key)!;
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let m = 0; m < neighbors.length; m++) {
        const nt = neighbors[m];
        const dot = fnx * faceNx[nt] + fny * faceNy[nt] + fnz * faceNz[nt];
        if (dot >= cosThreshold) {
          const w = Math.max(1e-8, faceArea[nt]);
          sx += faceNx[nt] * w;
          sy += faceNy[nt] * w;
          sz += faceNz[nt] * w;
        }
      }
      const sLen = Math.hypot(sx, sy, sz);
      const off = vi * 3;
      if (sLen > 1e-12) {
        normals[off] = sx / sLen;
        normals[off + 1] = sy / sLen;
        normals[off + 2] = sz / sLen;
      } else {
        normals[off] = fnx;
        normals[off + 1] = fny;
        normals[off + 2] = fnz;
      }
    }
  }

  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return geo;
}

function makeRoundedRectPts(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  radius: number,
  segs = 6
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

export function makeCircleHole(cx: number, cy: number, radius: number, segs = 28): THREE.Path {
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

export function offsetClosedPolygonCCW(pts: THREE.Vector2[], offset: number): THREE.Vector2[] {
  const n = pts.length;
  const out: THREE.Vector2[] = [];
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const curr = pts[i];
    const next = pts[(i + 1) % n];

    const e1x = curr.x - prev.x;
    const e1y = curr.y - prev.y;
    const l1 = Math.max(1e-7, Math.hypot(e1x, e1y));
    const n1x = e1y / l1;
    const n1y = -e1x / l1;

    const e2x = next.x - curr.x;
    const e2y = next.y - curr.y;
    const l2 = Math.max(1e-7, Math.hypot(e2x, e2y));
    const n2x = e2y / l2;
    const n2y = -e2x / l2;

    let bx = n1x + n2x;
    let by = n1y + n2y;
    const bLen = Math.hypot(bx, by);
    if (bLen < 1e-5) {
      out.push(new THREE.Vector2(curr.x + n1x * offset, curr.y + n1y * offset));
    } else {
      bx /= bLen;
      by /= bLen;
      const denom = Math.max(0.45, bx * n1x + by * n1y);
      const scale = Math.min(2.2, 1 / denom) * offset;
      out.push(new THREE.Vector2(curr.x + bx * scale, curr.y + by * scale));
    }
  }
  return out;
}

function buildOneSidedBevelPrism(
  outerCCW: THREE.Vector2[],
  zBack: number,
  zFront: number,
  backBevel: number,
  frontBevel: number
): THREE.BufferGeometry {
  const cleanOuter = ensureCCW(outerCCW);
  const n = cleanOuter.length;

  interface Ring {
    pts: THREE.Vector2[];
    z: number;
  }

  const rings: Ring[] = [];
  if (backBevel > 1e-5) {
    rings.push({ pts: offsetClosedPolygonCCW(cleanOuter, -backBevel), z: zBack });
    rings.push({ pts: cleanOuter, z: zBack + backBevel });
  } else {
    rings.push({ pts: cleanOuter, z: zBack });
  }

  if (frontBevel > 1e-5) {
    rings.push({ pts: cleanOuter, z: zFront - frontBevel });
    rings.push({ pts: offsetClosedPolygonCCW(cleanOuter, -frontBevel), z: zFront });
  } else {
    rings.push({ pts: cleanOuter, z: zFront });
  }

  const positions: number[] = [];
  const pushTri = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number
  ) => {
    positions.push(ax, ay, az, bx, by, bz, cx, cy, cz);
  };

  for (let r = 0; r < rings.length - 1; r++) {
    const r0 = rings[r];
    const r1 = rings[r + 1];
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const a0 = r0.pts[i];
      const b0 = r0.pts[j];
      const b1 = r1.pts[j];
      const a1 = r1.pts[i];
      pushTri(a0.x, a0.y, r0.z, b0.x, b0.y, r0.z, b1.x, b1.y, r1.z);
      pushTri(a0.x, a0.y, r0.z, b1.x, b1.y, r1.z, a1.x, a1.y, r1.z);
    }
  }

  const frontRing = rings[rings.length - 1];
  const frontFaces = THREE.ShapeUtils.triangulateShape(frontRing.pts, []);
  for (const [i0, i1, i2] of frontFaces) {
    const p0 = frontRing.pts[i0];
    const p1 = frontRing.pts[i1];
    const p2 = frontRing.pts[i2];
    pushTri(p0.x, p0.y, frontRing.z, p1.x, p1.y, frontRing.z, p2.x, p2.y, frontRing.z);
  }

  const backRing = rings[0];
  const backFaces = THREE.ShapeUtils.triangulateShape(backRing.pts, []);
  for (const [i0, i1, i2] of backFaces) {
    const p0 = backRing.pts[i0];
    const p1 = backRing.pts[i1];
    const p2 = backRing.pts[i2];
    pushTri(p0.x, p0.y, backRing.z, p2.x, p2.y, backRing.z, p1.x, p1.y, backRing.z);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return applyMachinedNormals(geo, (28 * Math.PI) / 180);
}

export function makeClosedLatheZ(
  profileZR: Array<[number, number]>,
  radialSegments = 32,
  sortByZ = true
): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [];
  const ordered = sortByZ ? [...profileZR].sort((a, b) => a[0] - b[0]) : profileZR;
  if (ordered[0][1] > 1e-4) {
    pts.push(new THREE.Vector2(0, ordered[0][0]));
  }
  for (const [z, r] of ordered) {
    pts.push(new THREE.Vector2(Math.max(0, r), z));
  }
  const last = ordered[ordered.length - 1];
  if (last[1] > 1e-4) {
    pts.push(new THREE.Vector2(0, last[0]));
  }
  const geo = new THREE.LatheGeometry(pts, radialSegments);
  geo.rotateX(Math.PI / 2);
  geo.computeVertexNormals();
  return geo;
}

export function buildSlottedBrassScrew(
  radius: number,
  headDepth: number,
  slotWidth: number,
  slotDepth: number,
  mat: THREE.Material,
  disposables: THREE.BufferGeometry[]
): THREE.Group {
  const grp = new THREE.Group();
  grp.name = 'SlottedBrassScrew';

  const bevel = 0.0018;
  const baseR = Math.max(0.004, radius - bevel);
  const baseDepth = Math.max(0.005, headDepth - slotDepth - bevel);
  const baseShape = new THREE.Shape();
  baseShape.absarc(0, 0, baseR, 0, Math.PI * 2, false);
  const baseRaw = new THREE.ExtrudeGeometry(baseShape, {
    depth: baseDepth,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
    curveSegments: 28,
  });
  baseRaw.translate(0, 0, -headDepth + bevel);
  const baseGeo = applyMachinedNormals(baseRaw);
  disposables.push(baseGeo);
  const baseMesh = new THREE.Mesh(baseGeo, mat);
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  grp.add(baseMesh);

  const halfSlot = slotWidth * 0.5 + bevel;
  const theta0 = Math.asin(Math.min(0.85, halfSlot / baseR));
  for (const sign of [1, -1]) {
    const pts: THREE.Vector2[] = [];
    const segs = 18;
    const startA = sign > 0 ? theta0 : Math.PI + theta0;
    const endA = sign > 0 ? Math.PI - theta0 : Math.PI * 2 - theta0;
    for (let s = 0; s <= segs; s++) {
      const a = startA + (endA - startA) * (s / segs);
      pts.push(new THREE.Vector2(Math.cos(a) * baseR, Math.sin(a) * baseR));
    }
    const ccw = ensureCCW(pts);
    const lobeShape = new THREE.Shape(ccw);
    const lobeRaw = new THREE.ExtrudeGeometry(lobeShape, {
      depth: Math.max(0.003, slotDepth - bevel * 2),
      bevelEnabled: true,
      bevelThickness: bevel,
      bevelSize: bevel,
      bevelSegments: 2,
    });
    lobeRaw.translate(0, 0, -slotDepth + bevel);
    const lobeGeo = applyMachinedNormals(lobeRaw);
    disposables.push(lobeGeo);
    const lobeMesh = new THREE.Mesh(lobeGeo, mat);
    lobeMesh.castShadow = true;
    lobeMesh.receiveShadow = true;
    grp.add(lobeMesh);
  }

  return grp;
}

// =============================================================================
// 1. Seamless Monolithic Oiled Charred-Oak Door-Stile Specimen with Mortise Rebate
// =============================================================================

export interface OakSpecimenAssembly {
  group: THREE.Group;
  dispose: () => void;
}

export const OAK_OUTER_MIN_X = -1.14;
export const OAK_OUTER_MAX_X = FOREND_X + FOREND_THICKNESS + 0.003; // +1.008
export const OAK_OUTER_MIN_Y = -0.865;
export const OAK_OUTER_MAX_Y = 0.885;
export const OAK_Z_BACK = -0.340;
export const OAK_Z_FRONT = 0.252;
export const OAK_REBATE_X = FOREND_X - 0.002; // +0.938
export const OAK_REBATE_MIN_Y = -0.788;
export const OAK_REBATE_MAX_Y = 0.808;
export const OAK_REBATE_Z_BACK = -0.250;

export function buildOakSpecimenBlock(materials: BoltMaterials): OakSpecimenAssembly {
  const group = new THREE.Group();
  group.name = 'CharredOakSpecimenBlock';
  const disposables: THREE.BufferGeometry[] = [];

  const pocketMinX = CASE_INNER_MIN_X - 0.044;
  const pocketMinY = CASE_INNER_MIN_Y - 0.039;
  const pocketMaxY = CASE_INNER_MAX_Y + 0.039;
  const caseBackZ = -0.224;

  const fullOuterPts = ensureCCW([
    new THREE.Vector2(OAK_OUTER_MIN_X + 0.018, OAK_OUTER_MIN_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_OUTER_MIN_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_OUTER_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MIN_X + 0.018, OAK_OUTER_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MIN_X, OAK_OUTER_MAX_Y - 0.018),
    new THREE.Vector2(OAK_OUTER_MIN_X, OAK_OUTER_MIN_Y + 0.018),
  ]);
  const rearStileGeo = buildOneSidedBevelPrism(
    fullOuterPts,
    OAK_Z_BACK,
    OAK_REBATE_Z_BACK,
    0.016,
    0
  );
  disposables.push(rearStileGeo);
  const rearStileMesh = new THREE.Mesh(rearStileGeo, materials.charredOak);
  rearStileMesh.name = 'OakRearStile';
  rearStileMesh.castShadow = true;
  rearStileMesh.receiveShadow = true;
  group.add(rearStileMesh);

  const rebateBackPts = ensureCCW([
    new THREE.Vector2(OAK_OUTER_MIN_X + 0.018, OAK_OUTER_MIN_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_OUTER_MIN_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_REBATE_MIN_Y),
    new THREE.Vector2(OAK_REBATE_X, OAK_REBATE_MIN_Y),
    new THREE.Vector2(OAK_REBATE_X, OAK_REBATE_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_REBATE_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_OUTER_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MIN_X + 0.018, OAK_OUTER_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MIN_X, OAK_OUTER_MAX_Y - 0.018),
    new THREE.Vector2(OAK_OUTER_MIN_X, OAK_OUTER_MIN_Y + 0.018),
  ]);
  const rebateBackGeo = buildOneSidedBevelPrism(rebateBackPts, OAK_REBATE_Z_BACK, caseBackZ, 0, 0);
  disposables.push(rebateBackGeo);
  const rebateBackMesh = new THREE.Mesh(rebateBackGeo, materials.charredOak);
  rebateBackMesh.name = 'OakMortiseFloor';
  rebateBackMesh.castShadow = true;
  rebateBackMesh.receiveShadow = true;
  group.add(rebateBackMesh);

  const cFramePts = ensureCCW([
    new THREE.Vector2(OAK_OUTER_MIN_X + 0.018, OAK_OUTER_MIN_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_OUTER_MIN_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_REBATE_MIN_Y),
    new THREE.Vector2(OAK_REBATE_X, OAK_REBATE_MIN_Y),
    new THREE.Vector2(OAK_REBATE_X, pocketMinY),
    new THREE.Vector2(pocketMinX + 0.02, pocketMinY),
    new THREE.Vector2(pocketMinX, pocketMinY + 0.02),
    new THREE.Vector2(pocketMinX, pocketMaxY - 0.02),
    new THREE.Vector2(pocketMinX + 0.02, pocketMaxY),
    new THREE.Vector2(OAK_REBATE_X, pocketMaxY),
    new THREE.Vector2(OAK_REBATE_X, OAK_REBATE_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_REBATE_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MAX_X, OAK_OUTER_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MIN_X + 0.018, OAK_OUTER_MAX_Y),
    new THREE.Vector2(OAK_OUTER_MIN_X, OAK_OUTER_MAX_Y - 0.018),
    new THREE.Vector2(OAK_OUTER_MIN_X, OAK_OUTER_MIN_Y + 0.018),
  ]);
  const cFrameGeo = buildOneSidedBevelPrism(cFramePts, caseBackZ, OAK_Z_FRONT, 0, 0.015);
  disposables.push(cFrameGeo);
  const cFrameMesh = new THREE.Mesh(cFrameGeo, materials.charredOak);
  cFrameMesh.name = 'OakCutawayFrame';
  cFrameMesh.castShadow = true;
  cFrameMesh.receiveShadow = true;
  group.add(cFrameMesh);

  return {
    group,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 2. Mortised Satin-Brass Forend Faceplate & Countersunk Slotted Wood-Screws
// =============================================================================

export interface ForendAssembly {
  group: THREE.Group;
  dispose: () => void;
}

export const FOREND_BEVEL = 0.004;
export const FOREND_MIN_Y = -0.784;
export const FOREND_MAX_Y = 0.804;
export const FOREND_HALF_Z = 0.246;
export const FOREND_OUTER_FACE_X = FOREND_X + FOREND_THICKNESS + 0.003; // +1.008
export const FOREND_SLOT_HALF_Y = 0.174;
export const FOREND_SLOT_HALF_Z = BOLT_HEAD_Z_HALF + 0.01;

export function buildForendAssembly(materials: BoltMaterials): ForendAssembly {
  const group = new THREE.Group();
  group.name = 'ForendBrassAssembly';
  const disposables: THREE.BufferGeometry[] = [];

  const halfZ = FOREND_HALF_Z - FOREND_BEVEL;
  const minY = FOREND_MIN_Y + FOREND_BEVEL;
  const maxY = FOREND_MAX_Y - FOREND_BEVEL;

  const outerPts = makeRoundedRectPts(-halfZ, minY, halfZ, maxY, 0.022, 6);
  const plateShape = new THREE.Shape(outerPts);

  const slotHalfZ = FOREND_SLOT_HALF_Z + FOREND_BEVEL;
  const slotMinY = STUMP_Y - FOREND_SLOT_HALF_Y - FOREND_BEVEL;
  const slotMaxY = STUMP_Y + FOREND_SLOT_HALF_Y + FOREND_BEVEL;
  const slotHolePts = ensureCW([
    new THREE.Vector2(-slotHalfZ, slotMinY),
    new THREE.Vector2(slotHalfZ, slotMinY),
    new THREE.Vector2(slotHalfZ, slotMaxY),
    new THREE.Vector2(-slotHalfZ, slotMaxY),
  ]);
  plateShape.holes.push(new THREE.Path(slotHolePts));

  const screwRadius = 0.054;
  const screwPositions: Array<[number, number]> = [
    [0.696, 0.54],
    [-0.676, -0.80],
  ];
  for (const [sy] of screwPositions) {
    plateShape.holes.push(makeCircleHole(0, sy, screwRadius + FOREND_BEVEL + 0.001, 28));
  }

  const coreDepth = FOREND_OUTER_FACE_X - FOREND_X - FOREND_BEVEL * 2;
  const plateRaw = new THREE.ExtrudeGeometry(plateShape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: FOREND_BEVEL,
    bevelSize: FOREND_BEVEL,
    bevelSegments: 2,
  });
  const plateGeo = applyMachinedNormals(plateRaw);
  plateGeo.rotateY(Math.PI / 2);
  plateGeo.translate(FOREND_X + FOREND_BEVEL, 0, 0);
  disposables.push(plateGeo);

  const plateMesh = new THREE.Mesh(plateGeo, materials.forendBrass);
  plateMesh.name = 'ForendFaceplate';
  plateMesh.castShadow = true;
  plateMesh.receiveShadow = true;
  group.add(plateMesh);

  for (const [sy, rotAngle] of screwPositions) {
    const screw = buildSlottedBrassScrew(
      screwRadius,
      0.032,
      0.015,
      0.014,
      materials.forendBrass,
      disposables
    );
    screw.rotation.y = Math.PI / 2;
    screw.rotation.x = rotAngle;
    screw.position.set(FOREND_OUTER_FACE_X + 0.0015, sy, 0);
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
// 3. Skeletonized Gunmetal Lock Case, Glass Rabbet Rim & Hardened Internal Pins
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
  const outerMaxX = FOREND_X - 0.001; // +0.939
  const outerMinY = CASE_INNER_MIN_Y - 0.035; // -0.620
  const outerMaxY = CASE_INNER_MAX_Y + 0.035; // +0.640

  // 3A. Gunmetal Lock Case Backplate (z: -0.224 .. -0.180)
  const backBevel = 0.004;
  const backShape = new THREE.Shape(
    makeRoundedRectPts(
      outerMinX + backBevel,
      outerMinY + backBevel,
      outerMaxX - backBevel,
      outerMaxY - backBevel,
      0.018,
      4
    )
  );
  const backRaw = new THREE.ExtrudeGeometry(backShape, {
    depth: 0.036,
    bevelEnabled: true,
    bevelThickness: backBevel,
    bevelSize: backBevel,
    bevelSegments: 2,
  });
  backRaw.translate(0, 0, -0.220);
  const backGeo = applyMachinedNormals(backRaw);
  disposables.push(backGeo);
  const backMesh = new THREE.Mesh(backGeo, materials.gunmetalCase);
  backMesh.name = 'LockCaseBackplate';
  backMesh.castShadow = true;
  backMesh.receiveShadow = true;
  group.add(backMesh);

  // 3B. Perimeter Steel Case Walls (top, bottom, rear left; z: -0.180 .. +0.204)
  const wallDepth = 0.376;
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
  const wallRaw = new THREE.ExtrudeGeometry(new THREE.Shape(wallPts), {
    depth: wallDepth,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
  });
  wallRaw.translate(0, 0, -0.176);
  const wallGeo = applyMachinedNormals(wallRaw);
  disposables.push(wallGeo);
  const wallMesh = new THREE.Mesh(wallGeo, materials.gunmetalCase);
  wallMesh.name = 'LockCaseWalls';
  wallMesh.castShadow = true;
  wallMesh.receiveShadow = true;
  group.add(wallMesh);

  // Right inner throat walls clearing deadbolt head cleanly
  const throatW = outerMaxX - CASE_INNER_MAX_X;
  const throatCenterX = (CASE_INNER_MAX_X + outerMaxX) * 0.5;
  const throatSlotHalfY = FOREND_SLOT_HALF_Y + 0.004;
  const throatTopH = Math.max(0.02, CASE_INNER_MAX_Y - (STUMP_Y + throatSlotHalfY));
  const throatGeoTop = new THREE.BoxGeometry(throatW, throatTopH, wallDepth);
  throatGeoTop.translate(throatCenterX, STUMP_Y + throatSlotHalfY + throatTopH * 0.5, 0.012);
  disposables.push(throatGeoTop);
  const throatTop = new THREE.Mesh(throatGeoTop, materials.gunmetalCase);
  throatTop.name = 'CaseThroatTop';
  throatTop.castShadow = true;
  throatTop.receiveShadow = true;
  group.add(throatTop);

  const throatBotH = Math.max(0.02, STUMP_Y - throatSlotHalfY - CASE_INNER_MIN_Y);
  const throatGeoBot = new THREE.BoxGeometry(throatW, throatBotH, wallDepth);
  throatGeoBot.translate(throatCenterX, CASE_INNER_MIN_Y + throatBotH * 0.5, 0.012);
  disposables.push(throatGeoBot);
  const throatBot = new THREE.Mesh(throatGeoBot, materials.gunmetalCase);
  throatBot.name = 'CaseThroatBottom';
  throatBot.castShadow = true;
  throatBot.receiveShadow = true;
  group.add(throatBot);

  // 3C. Skeletonized Museum-Cutaway Gunmetal Front Inspection Frame & Brass Inner Bezel Trim
  // Clean unobstructed rectangular viewing window [CUTAWAY_WIN_MIN_X..MAX_X] x [CUTAWAY_WIN_MIN_Y..MAX_Y]
  const rimBevel = 0.0035;
  const bezelOuter = makeRoundedRectPts(
    outerMinX + rimBevel,
    outerMinY + rimBevel,
    outerMaxX - rimBevel,
    outerMaxY - rimBevel,
    0.018,
    4
  );
  const bezelShape = new THREE.Shape(bezelOuter);

  const winHolePts = ensureCW(
    makeRoundedRectPts(
      CUTAWAY_WIN_MIN_X - rimBevel,
      CUTAWAY_WIN_MIN_Y - rimBevel,
      CUTAWAY_WIN_MAX_X + rimBevel,
      CUTAWAY_WIN_MAX_Y + rimBevel,
      0.018,
      5
    )
  );
  bezelShape.holes.push(new THREE.Path(winHolePts));

  const cornerCoords: Array<[number, number, number]> = [
    [-0.798, 0.602, 0.42],
    [-0.798, -0.582, -0.68],
    [0.896, 0.602, 1.15],
    [0.896, -0.582, -0.25],
  ];
  const cornerScrewR = 0.024;
  for (const [cx, cy] of cornerCoords) {
    bezelShape.holes.push(makeCircleHole(cx, cy, cornerScrewR + rimBevel + 0.001, 24));
  }

  const bezelRaw = new THREE.ExtrudeGeometry(bezelShape, {
    depth: 0.022,
    bevelEnabled: true,
    bevelThickness: rimBevel,
    bevelSize: rimBevel,
    bevelSegments: 2,
  });
  bezelRaw.translate(0, 0, CUTAWAY_FRONT_Z - 0.012);
  const bezelGeo = applyMachinedNormals(bezelRaw);
  disposables.push(bezelGeo);
  const bezelMesh = new THREE.Mesh(bezelGeo, materials.gunmetalCase);
  bezelMesh.name = 'CutawayFrontRim';
  bezelMesh.castShadow = true;
  bezelMesh.receiveShadow = true;
  group.add(bezelMesh);

  // Slim chamfered satin-brass picture-frame inner cutaway trim bordering the open inspection window
  const trimBevel = 0.0022;
  const trimOuter = makeRoundedRectPts(
    CUTAWAY_WIN_MIN_X - 0.016 + trimBevel,
    CUTAWAY_WIN_MIN_Y - 0.016 + trimBevel,
    CUTAWAY_WIN_MAX_X + 0.016 - trimBevel,
    CUTAWAY_WIN_MAX_Y + 0.016 - trimBevel,
    0.018,
    5
  );
  const trimShape = new THREE.Shape(trimOuter);
  const trimInner = ensureCW(
    makeRoundedRectPts(
      CUTAWAY_WIN_MIN_X + 0.008 - trimBevel,
      CUTAWAY_WIN_MIN_Y + 0.008 - trimBevel,
      CUTAWAY_WIN_MAX_X - 0.008 + trimBevel,
      CUTAWAY_WIN_MAX_Y - 0.008 + trimBevel,
      0.012,
      5
    )
  );
  trimShape.holes.push(new THREE.Path(trimInner));

  const trimRaw = new THREE.ExtrudeGeometry(trimShape, {
    depth: 0.006,
    bevelEnabled: true,
    bevelThickness: trimBevel,
    bevelSize: trimBevel,
    bevelSegments: 2,
  });
  trimRaw.translate(0, 0, CUTAWAY_FRONT_Z + 0.014);
  const trimGeo = applyMachinedNormals(trimRaw);
  disposables.push(trimGeo);
  const trimMesh = new THREE.Mesh(trimGeo, materials.forendBrass);
  trimMesh.name = 'CutawayBrassTrim';
  trimMesh.castShadow = true;
  trimMesh.receiveShadow = true;
  group.add(trimMesh);

  // 4 Countersunk Corner Case Cover Screws seated flush inside rim holes
  for (const [cx, cy, ang] of cornerCoords) {
    const screw = buildSlottedBrassScrew(
      cornerScrewR,
      0.016,
      0.0075,
      0.007,
      materials.forendBrass,
      disposables
    );
    screw.rotation.z = ang;
    screw.position.set(cx, cy, CUTAWAY_FRONT_Z + 0.015);
    group.add(screw);
  }

  // 3D. Beveled Rear Brass Bolt Wear-Rail + 2 Turned Steel Rivets (z: -0.180 .. -0.168)
  const railShape = new THREE.Shape(
    makeRoundedRectPts(-0.695, STUMP_Y - 0.064, 0.815, STUMP_Y + 0.064, 0.016, 4)
  );
  const railRaw = new THREE.ExtrudeGeometry(railShape, {
    depth: 0.008,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.002,
    bevelSegments: 1,
  });
  railRaw.translate(0, 0, -0.178);
  const railGeo = applyMachinedNormals(railRaw);
  disposables.push(railGeo);
  const railMesh = new THREE.Mesh(railGeo, materials.forendBrass);
  railMesh.name = 'RearWearRail';
  railMesh.receiveShadow = true;
  group.add(railMesh);

  for (const rx of [-0.625, 0.745]) {
    const rPinGeo = makeClosedLatheZ(
      [
        [-0.174, 0.016],
        [-0.169, 0.016],
        [-0.166, 0.011],
      ],
      18
    );
    rPinGeo.translate(rx, STUMP_Y, 0);
    disposables.push(rPinGeo);
    const rPinMesh = new THREE.Mesh(rPinGeo, materials.hardenedSteel);
    rPinMesh.name = 'WearRailRivet';
    group.add(rPinMesh);
  }

  // 3E. Hardened Steel Lever Pivot Post + Thrust Bushing + 4 Inter-Lever Brass Washers + Front Retaining Collar
  // Cleanly capped at z = +0.176 (strictly recessed behind sealed glass plane GLASS_Z_BACK = +0.204!)
  const pivotPinGeo = makeClosedLatheZ([
    [-0.182, 0.056],
    [0.144, 0.056],
    [0.147, 0.072],
    [0.166, 0.072],
    [0.173, 0.052],
    [0.177, 0.034],
  ]);
  pivotPinGeo.translate(PIVOT_X, PIVOT_Y, 0);
  disposables.push(pivotPinGeo);
  const pivotMesh = new THREE.Mesh(pivotPinGeo, materials.hardenedSteel);
  pivotMesh.name = 'LeverPivotPost';
  pivotMesh.castShadow = true;
  pivotMesh.receiveShadow = true;
  group.add(pivotMesh);

  const washerOuterR = 0.098;
  const washerInnerR = 0.0575;
  const washerShape = new THREE.Shape();
  washerShape.absarc(0, 0, washerOuterR, 0, Math.PI * 2, false);
  washerShape.holes.push(makeCircleHole(0, 0, washerInnerR, 28));

  // Rear solid brass thrust stand-off bushing supporting Lever 0 off the backplate
  const rearBushingGeo = makeClosedLatheZ(
    [
      [-0.180, 0.057],
      [-0.180, 0.098],
      [-0.064, 0.098],
      [-0.061, 0.092],
      [-0.061, 0.057],
    ],
    28,
    false
  );
  rearBushingGeo.translate(PIVOT_X, PIVOT_Y, 0);
  disposables.push(rearBushingGeo);
  const rearBushingMesh = new THREE.Mesh(rearBushingGeo, materials.forendBrass);
  rearBushingMesh.name = 'PivotRearThrustBushing';
  rearBushingMesh.castShadow = true;
  rearBushingMesh.receiveShadow = true;
  group.add(rearBushingMesh);

  // 4 Precision Inter-Lever Brass Spacer Washers (exact 0.0025 daylight gap per side, zero overlap)
  for (let w = 0; w < LEVER_COUNT - 1; w++) {
    const zMid = (getLeverZ(w) + getLeverZ(w + 1)) * 0.5;
    const wThickness = 0.012;
    const wRaw = new THREE.ExtrudeGeometry(washerShape, {
      depth: wThickness,
      bevelEnabled: true,
      bevelThickness: 0.001,
      bevelSize: 0.001,
      bevelSegments: 1,
      curveSegments: 28,
    });
    wRaw.translate(PIVOT_X, PIVOT_Y, zMid - wThickness * 0.5);
    const wGeo = applyMachinedNormals(wRaw);
    disposables.push(wGeo);
    const wMesh = new THREE.Mesh(wGeo, materials.forendBrass);
    wMesh.name = `LeverSpacerWasher_${w}`;
    wMesh.castShadow = true;
    wMesh.receiveShadow = true;
    group.add(wMesh);
  }

  // Front chamfered brass pivot retaining collar seated flush in front of Lever 4 (z: +0.130 .. +0.144)
  const capWasherRaw = new THREE.ExtrudeGeometry(washerShape, {
    depth: 0.010,
    bevelEnabled: true,
    bevelThickness: 0.0018,
    bevelSize: 0.0018,
    bevelSegments: 2,
    curveSegments: 28,
  });
  capWasherRaw.translate(
    PIVOT_X,
    PIVOT_Y,
    getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.004
  );
  const capWasherGeo = applyMachinedNormals(capWasherRaw);
  disposables.push(capWasherGeo);
  const capWasherMesh = new THREE.Mesh(capWasherGeo, materials.forendBrass);
  capWasherMesh.name = 'PivotFrontWasher';
  capWasherMesh.castShadow = true;
  capWasherMesh.receiveShadow = true;
  group.add(capWasherMesh);

  // 3F. Hardened Steel Spring Abutment Post & Anti-Rotation Reaction Fulcrum Pin (capped cleanly at z <= +0.168)
  const springPostGeo = makeClosedLatheZ([
    [-0.182, 0.043],
    [0.144, 0.043],
    [0.152, 0.054],
    [0.164, 0.046],
    [0.168, 0.030],
  ]);
  springPostGeo.translate(SPRING_POST_X, SPRING_POST_Y, 0);
  disposables.push(springPostGeo);
  const springPostMesh = new THREE.Mesh(springPostGeo, materials.hardenedSteel);
  springPostMesh.name = 'SpringAbutmentPost';
  springPostMesh.castShadow = true;
  springPostMesh.receiveShadow = true;
  group.add(springPostMesh);

  // Hardened-steel anti-rotation reaction fulcrum pin bracing the 5 phosphor-bronze spring shanks against torque!
  const reactionPinGeo = makeClosedLatheZ([
    [-0.182, SPRING_REACTION_PIN_R],
    [0.144, SPRING_REACTION_PIN_R],
    [0.150, SPRING_REACTION_PIN_R + 0.005],
    [0.158, SPRING_REACTION_PIN_R - 0.004],
  ]);
  reactionPinGeo.translate(SPRING_REACTION_PIN_X, SPRING_REACTION_PIN_Y, 0);
  disposables.push(reactionPinGeo);
  const reactionPinMesh = new THREE.Mesh(reactionPinGeo, materials.hardenedSteel);
  reactionPinMesh.name = 'SpringReactionPin';
  reactionPinMesh.castShadow = true;
  reactionPinMesh.receiveShadow = true;
  group.add(reactionPinMesh);

  // 3G. Lever Bottom Rest Stop Pin with Satin-Brass Sleeve (capped cleanly at z = +0.154)
  const restPinGeo = makeClosedLatheZ([
    [-0.182, REST_PIN_RADIUS],
    [0.134, REST_PIN_RADIUS],
    [0.140, REST_PIN_RADIUS + 0.005],
    [0.150, REST_PIN_RADIUS + 0.005],
    [0.156, REST_PIN_RADIUS - 0.007],
  ]);
  restPinGeo.translate(REST_PIN_X, REST_PIN_Y, 0);
  disposables.push(restPinGeo);
  const restPinMesh = new THREE.Mesh(restPinGeo, materials.forendBrass);
  restPinMesh.name = 'LeverRestStopPin';
  restPinMesh.castShadow = true;
  restPinMesh.receiveShadow = true;
  group.add(restPinMesh);

  // 3H. Hardened Bolt-Tail Guide Pin
  const guidePinGeo = makeClosedLatheZ([
    [-0.182, 0.030],
    [-0.070, 0.030],
    [-0.062, 0.040],
    [-0.056, 0.034],
  ]);
  guidePinGeo.translate(GUIDE_PIN_X, GUIDE_PIN_Y, 0);
  disposables.push(guidePinGeo);
  const guidePinMesh = new THREE.Mesh(guidePinGeo, materials.hardenedSteel);
  guidePinMesh.name = 'BoltGuidePin';
  guidePinMesh.castShadow = true;
  guidePinMesh.receiveShadow = true;
  group.add(guidePinMesh);

  // 3I. Hardened Central Cam-Barrel Guide Drill-Pin (projects forward from backplate into hollow inner cam barrel!)
  const drillPinGeo = makeClosedLatheZ([
    [-0.182, 0.019],
    [0.036, 0.019],
    [0.050, 0.012],
  ]);
  drillPinGeo.translate(KEY_X, KEY_Y, 0);
  disposables.push(drillPinGeo);
  const drillPinMesh = new THREE.Mesh(drillPinGeo, materials.hardenedSteel);
  drillPinMesh.name = 'CamCenterGuidePin';
  drillPinMesh.castShadow = true;
  group.add(drillPinMesh);

  return {
    group,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 4. Heavy Square Blued-Steel Deadbolt, Slotted Tail, Flared U-Talon & Hardened Stump
// =============================================================================

export interface DeadboltAssembly {
  group: THREE.Group;
  hitMeshes: THREE.Mesh[];
  dispose: () => void;
}

export const BOLT_HEAD_HALF_Y = 0.164;
export const BOLT_TALON_JAW_BOTTOM_Y = 0.024;

export function buildDeadboltAssembly(materials: BoltMaterials): DeadboltAssembly {
  const group = new THREE.Group();
  group.name = 'DeadboltAssembly';
  const disposables: THREE.BufferGeometry[] = [];
  const hitMeshes: THREE.Mesh[] = [];

  const headBevel = 0.012;
  const headMinX = 0.510 + headBevel;
  const headMaxX = 1.022 - headBevel;
  const headMinY = STUMP_Y - BOLT_HEAD_HALF_Y + headBevel;
  const headMaxY = STUMP_Y + BOLT_HEAD_HALF_Y - headBevel;
  const headDepth = BOLT_HEAD_Z_HALF * 2 - headBevel * 2;

  const headPts = ensureCCW([
    new THREE.Vector2(headMinX, headMinY + 0.010),
    new THREE.Vector2(headMinX + 0.010, headMinY),
    new THREE.Vector2(headMaxX - 0.018, headMinY),
    new THREE.Vector2(headMaxX, headMinY + 0.018),
    new THREE.Vector2(headMaxX, headMaxY - 0.018),
    new THREE.Vector2(headMaxX - 0.018, headMaxY),
    new THREE.Vector2(headMinX + 0.010, headMaxY),
    new THREE.Vector2(headMinX, headMaxY - 0.010),
  ]);
  const headRaw = new THREE.ExtrudeGeometry(new THREE.Shape(headPts), {
    depth: headDepth,
    bevelEnabled: true,
    bevelThickness: headBevel,
    bevelSize: headBevel,
    bevelSegments: 2,
  });
  headRaw.translate(0, 0, -headDepth * 0.5);
  const headGeo = applyMachinedNormals(headRaw);
  disposables.push(headGeo);
  const headMesh = new THREE.Mesh(headGeo, materials.bluedSteelBolt);
  headMesh.name = 'DeadboltHead';
  headMesh.castShadow = true;
  headMesh.receiveShadow = true;
  group.add(headMesh);
  hitMeshes.push(headMesh);

  // Two flush-machined hardened anti-saw steel drill-pin end plugs seated cleanly on the deadbolt nose face
  const noseFaceX = headMaxX + headBevel;
  for (const dy of [-0.072, 0.072]) {
    const plugGeo = makeClosedLatheZ(
      [
        [-0.008, 0.023],
        [0.001, 0.023],
        [0.004, 0.018],
      ],
      22
    );
    plugGeo.rotateY(Math.PI / 2);
    plugGeo.translate(noseFaceX, STUMP_Y + dy, 0);
    disposables.push(plugGeo);
    const plugMesh = new THREE.Mesh(plugGeo, materials.hardenedSteel);
    plugMesh.name = 'AntiSawDrillPlug';
    plugMesh.castShadow = true;
    group.add(plugMesh);
  }

  // 4B. Blued-Steel Bolt Tail (Shank) with Horizontal Guide Slot & Flared U-Shaped Bolt Talon
  const tailBevel = 0.004;
  const tailDepth = BOLT_TAIL_Z_FRONT - BOLT_TAIL_Z_BACK - tailBevel * 2;
  const tailZMid = (BOLT_TAIL_Z_BACK + BOLT_TAIL_Z_FRONT) * 0.5;

  const talonCenterX = KEY_X - 0.17;
  const notchHalfW = 0.046 + tailBevel;
  const mouthFlareW = 0.076 + tailBevel;
  const jawW = 0.064;
  const spineMinY = STUMP_Y - 0.084;
  const spineMaxY = STUMP_Y + 0.084;
  const jawBottomY = BOLT_TALON_JAW_BOTTOM_Y + tailBevel;
  const flareTopY = 0.048;
  const notchTopY = 0.104 + tailBevel;

  const tailPts = ensureCCW([
    new THREE.Vector2(-0.765, spineMinY + 0.012),
    new THREE.Vector2(-0.753, spineMinY),
    new THREE.Vector2(talonCenterX - mouthFlareW - jawW - 0.018, spineMinY),
    new THREE.Vector2(talonCenterX - mouthFlareW - jawW, jawBottomY + 0.016),
    new THREE.Vector2(talonCenterX - mouthFlareW - jawW + 0.014, jawBottomY),
    new THREE.Vector2(talonCenterX - mouthFlareW, jawBottomY),
    new THREE.Vector2(talonCenterX - notchHalfW, flareTopY),
    new THREE.Vector2(talonCenterX - notchHalfW, notchTopY - 0.012),
    new THREE.Vector2(talonCenterX - notchHalfW + 0.012, notchTopY),
    new THREE.Vector2(talonCenterX + notchHalfW - 0.012, notchTopY),
    new THREE.Vector2(talonCenterX + notchHalfW, notchTopY - 0.012),
    new THREE.Vector2(talonCenterX + notchHalfW, flareTopY),
    new THREE.Vector2(talonCenterX + mouthFlareW, jawBottomY),
    new THREE.Vector2(talonCenterX + mouthFlareW + jawW - 0.014, jawBottomY),
    new THREE.Vector2(talonCenterX + mouthFlareW + jawW, jawBottomY + 0.016),
    new THREE.Vector2(talonCenterX + mouthFlareW + jawW + 0.018, spineMinY),
    new THREE.Vector2(headMinX + 0.024, spineMinY),
    new THREE.Vector2(headMinX + 0.024, spineMaxY),
    new THREE.Vector2(-0.753, spineMaxY),
    new THREE.Vector2(-0.765, spineMaxY - 0.012),
  ]);

  const tailShape = new THREE.Shape(tailPts);
  const slotPts = makeRoundedRectPts(-0.722, STUMP_Y - 0.037, -0.298, STUMP_Y + 0.037, 0.035, 6);
  tailShape.holes.push(new THREE.Path(ensureCW(slotPts)));

  const tailRaw = new THREE.ExtrudeGeometry(tailShape, {
    depth: tailDepth,
    bevelEnabled: true,
    bevelThickness: tailBevel,
    bevelSize: tailBevel,
    bevelSegments: 2,
  });
  tailRaw.translate(0, 0, tailZMid - tailDepth * 0.5);
  const tailGeo = applyMachinedNormals(tailRaw);
  disposables.push(tailGeo);
  const tailMesh = new THREE.Mesh(tailGeo, materials.bluedSteelBolt);
  tailMesh.name = 'DeadboltTailAndTalon';
  tailMesh.castShadow = true;
  tailMesh.receiveShadow = true;
  group.add(tailMesh);
  hitMeshes.push(tailMesh);

  // 4C. Hardened Square Steel Bolt Stump (exact STUMP_WIDTH x STUMP_HEIGHT, cleanly recessed behind glass!)
  const stumpBevel = 0.004;
  const stumpDepth = STUMP_Z_FRONT - STUMP_Z_BACK - stumpBevel * 2;
  const stumpHalfW = STUMP_WIDTH * 0.5 - stumpBevel;
  const stumpHalfH = STUMP_HEIGHT * 0.5 - stumpBevel;
  const stumpPts = makeRoundedRectPts(
    STUMP_X_UNLOCKED - stumpHalfW,
    STUMP_Y - stumpHalfH,
    STUMP_X_UNLOCKED + stumpHalfW,
    STUMP_Y + stumpHalfH,
    0.006,
    3
  );
  const stumpRaw = new THREE.ExtrudeGeometry(new THREE.Shape(stumpPts), {
    depth: stumpDepth,
    bevelEnabled: true,
    bevelThickness: stumpBevel,
    bevelSize: stumpBevel,
    bevelSegments: 2,
  });
  stumpRaw.translate(0, 0, STUMP_Z_BACK + stumpBevel);
  const stumpGeo = applyMachinedNormals(stumpRaw);
  disposables.push(stumpGeo);
  const stumpMesh = new THREE.Mesh(stumpGeo, materials.hardenedSteel);
  stumpMesh.name = 'BoltStump';
  stumpMesh.castShadow = true;
  stumpMesh.receiveShadow = true;
  group.add(stumpMesh);
  hitMeshes.push(stumpMesh);

  const rivetGeo = makeClosedLatheZ([
    [STUMP_Z_FRONT - 0.002, 0.022],
    [STUMP_Z_FRONT + 0.006, 0.020],
    [STUMP_Z_FRONT + 0.011, 0.012],
  ]);
  rivetGeo.translate(STUMP_X_UNLOCKED, STUMP_Y, 0);
  disposables.push(rivetGeo);
  const rivetMesh = new THREE.Mesh(rivetGeo, materials.forendBrass);
  rivetMesh.name = 'BoltStumpRivet';
  rivetMesh.castShadow = true;
  group.add(rivetMesh);
  hitMeshes.push(rivetMesh);

  // 4D. Precision-Milled Tempered Bronze Upper Drive Rack riveted onto Deadbolt Tail Spine
  const rackBevel = 0.0012;
  const rackRawPts = buildInvoluteRackPolygon2D(UPPER_RACK_SPEC).map(
    (p) => new THREE.Vector2(p.x, p.y)
  );
  const rackInset = offsetClosedPolygonCCW(ensureCCW(rackRawPts), -rackBevel);
  const rackDepth = UPPER_RACK_SPEC.zFront - UPPER_RACK_SPEC.zBack - rackBevel * 2;
  const rackRawGeo = new THREE.ExtrudeGeometry(new THREE.Shape(ensureCCW(rackInset)), {
    depth: rackDepth,
    bevelEnabled: true,
    bevelThickness: rackBevel,
    bevelSize: rackBevel,
    bevelSegments: 2,
  });
  rackRawGeo.translate(0, 0, UPPER_RACK_SPEC.zBack + rackBevel);
  const rackGeo = applyMachinedNormals(rackRawGeo);
  disposables.push(rackGeo);
  const rackMesh = new THREE.Mesh(rackGeo, materials.machinedBronzeRack);
  rackMesh.name = 'DeadboltUpperRack';
  rackMesh.castShadow = true;
  rackMesh.receiveShadow = true;
  group.add(rackMesh);
  hitMeshes.push(rackMesh);

  for (const pinX of [0.055, 0.255, 0.455]) {
    const pinGeo = makeClosedLatheZ(
      [
        [UPPER_RACK_SPEC.zFront - 0.002, 0.013],
        [UPPER_RACK_SPEC.zFront + 0.005, 0.013],
        [UPPER_RACK_SPEC.zFront + 0.009, 0.0085],
      ],
      20
    );
    pinGeo.translate(pinX, 0.285, 0);
    disposables.push(pinGeo);
    const pinMesh = new THREE.Mesh(pinGeo, materials.hardenedSteel);
    pinMesh.name = 'RackDovetailPin';
    pinMesh.castShadow = true;
    group.add(pinMesh);
    hitMeshes.push(pinMesh);
  }

  return {
    group,
    hitMeshes,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 5. Five Stacked Precision Brass Lever Tumblers (Concentric Chubb Bellies & Anti-Pick)
// =============================================================================

export interface LeverPackAssembly {
  group: THREE.Group;
  leverPivots: THREE.Group[];
  leverMeshes: THREE.Mesh[];
  dispose: () => void;
}

export const LEVER_TOP_CREST_LOCAL_Y = 0.264;

export function createLeverGeometry(index: number): THREE.BufferGeometry {
  const spec = LEVER_GATE_SPECS[index];
  const belly = LEVER_BELLY_POLYLINES[index];

  const nominalOuter: THREE.Vector2[] = [];

  const earR = 0.124;
  const earSegs = 16;
  for (let s = 0; s <= earSegs; s++) {
    const a = Math.PI * 0.5 + Math.PI * (s / earSegs);
    nominalOuter.push(new THREE.Vector2(Math.cos(a) * earR, Math.sin(a) * earR));
  }

  for (let i = 0; i < belly.length; i++) {
    nominalOuter.push(new THREE.Vector2(belly[i].x, belly[i].y));
  }

  const lastBellyY = belly[belly.length - 1].y;
  nominalOuter.push(new THREE.Vector2(1.018, lastBellyY + 0.022));
  nominalOuter.push(new THREE.Vector2(1.018, LEVER_TOP_CREST_LOCAL_Y - 0.026));
  nominalOuter.push(new THREE.Vector2(0.992, LEVER_TOP_CREST_LOCAL_Y));

  const shoulderStepX = 0.24 + index * 0.036;
  nominalOuter.push(new THREE.Vector2(0.42, LEVER_TOP_CREST_LOCAL_Y));
  nominalOuter.push(new THREE.Vector2(shoulderStepX + 0.035, 0.234));
  nominalOuter.push(new THREE.Vector2(shoulderStepX, 0.200));
  nominalOuter.push(new THREE.Vector2(0.11, 0.154));

  const insetOuter = offsetClosedPolygonCCW(ensureCCW(nominalOuter), -LEVER_BEVEL);
  const shape = new THREE.Shape(ensureCCW(insetOuter));

  shape.holes.push(makeCircleHole(0, 0, 0.058 + LEVER_BEVEL, 28));

  const b = LEVER_BEVEL;
  const xLOut = GATE_POCKET_LEFT_OUTER_X - b;
  const xLIn = GATE_POCKET_LEFT_INNER_X + b;
  const xRIn = GATE_POCKET_RIGHT_INNER_X - b;
  const xROut = GATE_POCKET_RIGHT_OUTER_X + b;

  const spanX = Math.max(1e-4, spec.gateLockedLocalX - spec.gateUnlockedLocalX);
  const slopeY = (spec.gateLockedLocalY - spec.gateUnlockedLocalY) / spanX;
  const yGateL = spec.gateUnlockedLocalY + (xLIn - spec.gateUnlockedLocalX) * slopeY;
  const yGateR = spec.gateUnlockedLocalY + (xRIn - spec.gateUnlockedLocalX) * slopeY;

  const vertGateHalfH = (GATE_HALF_HEIGHT + b) / Math.cos(spec.alignRad);
  const gateTopL = yGateL + vertGateHalfH;
  const gateBotL = yGateL - vertGateHalfH;
  const gateTopR = yGateR + vertGateHalfH;
  const gateBotR = yGateR - vertGateHalfH;

  const yTopPocket = GATE_POCKET_TOP_Y + b;
  const yBotPocketL = Math.max(-0.088, Math.min(gateBotL - 0.016, -0.015)) - b;
  const yBotPocketR = GATE_POCKET_BOT_R_Y - b;

  const addAntiPickTeeth = (
    pts: THREE.Vector2[],
    xWall: number,
    yStart: number,
    yEnd: number,
    notchDir: -1 | 1
  ) => {
    const dir = yEnd > yStart ? 1 : -1;
    const span = Math.abs(yEnd - yStart);
    const margin = 0.014;
    const usable = span - margin * 2;
    if (usable < 0.022) return;
    const pitch = 0.032;
    const toothCount = Math.min(3, Math.max(1, Math.floor(usable / pitch)));
    const step = usable / toothCount;
    const halfNotch = 0.0075;
    const notchDepth = 0.016;
    for (let t = 0; t < toothCount; t++) {
      const yc = yStart + dir * (margin + step * (t + 0.5));
      pts.push(new THREE.Vector2(xWall, yc - dir * halfNotch));
      pts.push(new THREE.Vector2(xWall + notchDir * notchDepth, yc - dir * halfNotch));
      pts.push(new THREE.Vector2(xWall + notchDir * notchDepth, yc + dir * halfNotch));
      pts.push(new THREE.Vector2(xWall, yc + dir * halfNotch));
    }
  };

  const hPts: THREE.Vector2[] = [
    new THREE.Vector2(xLOut, yTopPocket),
    new THREE.Vector2(xLOut, yBotPocketL),
    new THREE.Vector2(xLIn, yBotPocketL),
  ];

  addAntiPickTeeth(hPts, xLIn, yBotPocketL, gateBotL, 1);
  hPts.push(new THREE.Vector2(xLIn, gateBotL));
  hPts.push(new THREE.Vector2(xRIn, gateBotR));

  addAntiPickTeeth(hPts, xRIn, gateBotR, yBotPocketR, -1);
  hPts.push(new THREE.Vector2(xRIn, yBotPocketR));
  hPts.push(new THREE.Vector2(xROut, yBotPocketR));
  hPts.push(new THREE.Vector2(xROut, yTopPocket));
  hPts.push(new THREE.Vector2(xRIn, yTopPocket));

  addAntiPickTeeth(hPts, xRIn, yTopPocket, gateTopR, -1);
  hPts.push(new THREE.Vector2(xRIn, gateTopR));
  hPts.push(new THREE.Vector2(xLIn, gateTopL));

  addAntiPickTeeth(hPts, xLIn, gateTopL, yTopPocket, 1);
  hPts.push(new THREE.Vector2(xLIn, yTopPocket));

  const cwHole = ensureCW(hPts);
  const gateHolePath = new THREE.Path();
  gateHolePath.moveTo(cwHole[0].x, cwHole[0].y);
  for (let i = 1; i < cwHole.length; i++) gateHolePath.lineTo(cwHole[i].x, cwHole[i].y);
  gateHolePath.closePath();
  shape.holes.push(gateHolePath);

  const coreDepth = LEVER_THICKNESS - LEVER_BEVEL * 2;
  const rawGeo = new THREE.ExtrudeGeometry(shape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: LEVER_BEVEL,
    bevelSize: LEVER_BEVEL,
    bevelSegments: 2,
    curveSegments: 24,
  });
  rawGeo.translate(0, 0, -coreDepth * 0.5);
  return applyMachinedNormals(rawGeo);
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
// 6. Five Staggered Dynamically Flexing Phosphor-Bronze Cantilever Leaf Springs
// =============================================================================

const SPRING_SEGMENTS = 30;
const SPRING_HALF_THICK_Z = 0.0076;
const SPRING_HALF_HEIGHT_N = 0.0078;
export const SPRING_COLLAR_OUTER_R = 0.059;
export const SPRING_COLLAR_INNER_R = 0.044;

export interface LeafSpringDiagnostics {
  rootX: number;
  rootY: number;
  rootDistFromPost: number;
  tipBottomX: number;
  tipBottomY: number;
  leverContactX: number;
  leverContactY: number;
  tipContactGap: number;
  maxArcY: number;
  minClearanceAboveLever: number;
  arcLength: number;
}

/**
 * Pure analytical cantilever leaf-spring curve evaluator shared by LeafSpringMesh and Vitest:
 *  - Anchors tangentially off the bronze anchor eye (braced cleanly over SpringReactionPin!)
 *  - Staggers contact feet across lever[0..4] so all 5 phosphor-bronze blades fan visibly like watch springs
 *  - Sweeps in a single graceful elastic arch (zero S-kink!), arriving strictly tangent & flush onto each lever crest
 *  - Stays strictly inside GLASS_WIN_MAX_Y - 0.030 across the full 0..21 deg pick stroke!
 */
export function evaluateLeafSpringCurve(
  leverAngleRad: number,
  outP0: { x: number; y: number },
  outP1: { x: number; y: number },
  outP2: { x: number; y: number },
  outP3: { x: number; y: number },
  outContact?: { x: number; y: number },
  leverIndex = 0
): void {
  const cosA = Math.cos(leverAngleRad);
  const sinA = Math.sin(leverAngleRad);
  const liftRatio = Math.max(0, Math.min(1, leverAngleRad / LEVER_MAX_RAD));

  // Anchor tangentially off upper-right outer perimeter of bronze anchor collar (radius ~0.0535)
  outP0.x = SPRING_POST_X + 0.022;
  outP0.y = SPRING_POST_Y + 0.049;

  // Staggered sliding contact point glides rightward along each lever's flat top crest [0.455..0.81]
  // preserving near-constant blade arc length (< 2% delta) while visibly fanning all 5 springs!
  const contactLocalX = 0.455 + leverIndex * 0.032 + liftRatio * 0.218;
  const contactLocalY = LEVER_TOP_CREST_LOCAL_Y;
  const cx = PIVOT_X + contactLocalX * cosA - contactLocalY * sinA;
  const cy = PIVOT_Y + contactLocalX * sinA + contactLocalY * cosA;
  if (outContact) {
    outContact.x = cx;
    outContact.y = cy;
  }

  const lnx = -sinA;
  const lny = cosA;
  const ltx = cosA;
  const lty = sinA;

  const hEnd = SPRING_HALF_HEIGHT_N * 0.85;
  outP3.x = cx + lnx * hEnd;
  outP3.y = cy + lny * hEnd;

  // Smooth tangential approach parallel to rotated lever top edge with tiny upward clearance (+0.007 * N)
  const tanLead = 0.155 + leverIndex * 0.012 + liftRatio * 0.026;
  outP2.x = outP3.x - ltx * tanLead + lnx * 0.007;
  outP2.y = outP3.y - lty * tanLead + lny * 0.007;

  // Graceful single-bow cantilever arch staying well inside CUTAWAY_WIN_MAX_Y (0.566)
  const archTargetY = Math.min(
    CUTAWAY_WIN_MAX_Y - 0.044,
    outP0.y + 0.012 + leverIndex * 0.002 + liftRatio * 0.038
  );
  outP1.x = outP0.x + 0.168 + leverIndex * 0.014 + liftRatio * 0.040;
  outP1.y = archTargetY;
}

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
  private readonly _p0 = { x: 0, y: 0 };
  private readonly _p1 = { x: 0, y: 0 };
  private readonly _p2 = { x: 0, y: 0 };
  private readonly _p3 = { x: 0, y: 0 };
  private readonly _contact = { x: 0, y: 0 };
  private lastAngle = -999;

  constructor(index: number, material: THREE.Material) {
    this.index = index;
    this.geometry = new THREE.BufferGeometry();

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

  inspectAtAngle(leverAngleRad: number): LeafSpringDiagnostics {
    this.update(leverAngleRad);
    const cosA = Math.cos(leverAngleRad);
    const sinA = Math.sin(leverAngleRad);
    const lnx = -sinA;
    const lny = cosA;

    const e = SPRING_SEGMENTS;
    const hEnd = SPRING_HALF_HEIGHT_N * 0.85;
    const tipBottomX = this.cx[e] - this.nx[e] * hEnd;
    const tipBottomY = this.cy[e] - this.ny[e] * hEnd;
    const tipContactGap = Math.hypot(tipBottomX - this._contact.x, tipBottomY - this._contact.y);

    let maxArcY = -Infinity;
    let minClearanceAboveLever = Infinity;
    let arcLength = 0;

    for (let s = 0; s <= SPRING_SEGMENTS; s++) {
      const h = SPRING_HALF_HEIGHT_N * (1.12 - 0.27 * (s / SPRING_SEGMENTS));
      const topY = this.cy[s] + this.ny[s] * h;
      if (topY > maxArcY) maxArcY = topY;

      const botX = this.cx[s] - this.nx[s] * h;
      const botY = this.cy[s] - this.ny[s] * h;
      const dx = botX - PIVOT_X;
      const dy = botY - PIVOT_Y;
      const localX = dx * cosA + dy * sinA;
      if (localX >= 0.42 && localX <= 0.992) {
        const signedAbove = (botX - this._contact.x) * lnx + (botY - this._contact.y) * lny;
        if (signedAbove < minClearanceAboveLever) minClearanceAboveLever = signedAbove;
      }

      if (s > 0) {
        arcLength += Math.hypot(this.cx[s] - this.cx[s - 1], this.cy[s] - this.cy[s - 1]);
      }
    }

    return {
      rootX: this._p0.x,
      rootY: this._p0.y,
      rootDistFromPost: Math.hypot(this._p0.x - SPRING_POST_X, this._p0.y - SPRING_POST_Y),
      tipBottomX,
      tipBottomY,
      leverContactX: this._contact.x,
      leverContactY: this._contact.y,
      tipContactGap,
      maxArcY,
      minClearanceAboveLever,
      arcLength,
    };
  }

  update(leverAngleRad: number): void {
    if (Math.abs(leverAngleRad - this.lastAngle) < 1e-5) return;
    this.lastAngle = leverAngleRad;

    const zCenter = getLeverZ(this.index);
    const zBack = zCenter - SPRING_HALF_THICK_Z;
    const zFront = zCenter + SPRING_HALF_THICK_Z;

    evaluateLeafSpringCurve(
      leverAngleRad,
      this._p0,
      this._p1,
      this._p2,
      this._p3,
      this._contact,
      this.index
    );

    const p0x = this._p0.x;
    const p0y = this._p0.y;
    const p1x = this._p1.x;
    const p1y = this._p1.y;
    const p2x = this._p2.x;
    const p2y = this._p2.y;
    const p3x = this._p3.x;
    const p3y = this._p3.y;

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
      this.nx[s] = -ty / len;
      this.ny[s] = tx / len;
    }

    let ptr = 0;
    const writeQuadSmooth = (
      ax: number, ay: number, az: number, nax: number, nay: number, naz: number,
      bx: number, by: number, bz: number, nbx: number, nby: number, nbz: number,
      cx: number, cy: number, cz: number, ncx: number, ncy: number, ncz: number,
      dx: number, dy: number, dz: number, ndx: number, ndy: number, ndz: number
    ) => {
      this.positions[ptr] = ax; this.positions[ptr + 1] = ay; this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = bx; this.positions[ptr + 4] = by; this.positions[ptr + 5] = bz;
      this.positions[ptr + 6] = cx; this.positions[ptr + 7] = cy; this.positions[ptr + 8] = cz;
      this.normals[ptr] = nax; this.normals[ptr + 1] = nay; this.normals[ptr + 2] = naz;
      this.normals[ptr + 3] = nbx; this.normals[ptr + 4] = nby; this.normals[ptr + 5] = nbz;
      this.normals[ptr + 6] = ncx; this.normals[ptr + 7] = ncy; this.normals[ptr + 8] = ncz;
      ptr += 9;

      this.positions[ptr] = ax; this.positions[ptr + 1] = ay; this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = cx; this.positions[ptr + 4] = cy; this.positions[ptr + 5] = cz;
      this.positions[ptr + 6] = dx; this.positions[ptr + 7] = dy; this.positions[ptr + 8] = dz;
      this.normals[ptr] = nax; this.normals[ptr + 1] = nay; this.normals[ptr + 2] = naz;
      this.normals[ptr + 3] = ncx; this.normals[ptr + 4] = ncy; this.normals[ptr + 5] = ncz;
      this.normals[ptr + 6] = ndx; this.normals[ptr + 7] = ndy; this.normals[ptr + 8] = ndz;
      ptr += 9;
    };

    for (let s = 0; s < SPRING_SEGMENTS; s++) {
      const h0 = SPRING_HALF_HEIGHT_N * (1.12 - 0.27 * (s / SPRING_SEGMENTS));
      const h1 = SPRING_HALF_HEIGHT_N * (1.12 - 0.27 * ((s + 1) / SPRING_SEGMENTS));

      const top0x = this.cx[s] + this.nx[s] * h0;
      const top0y = this.cy[s] + this.ny[s] * h0;
      const bot0x = this.cx[s] - this.nx[s] * h0;
      const bot0y = this.cy[s] - this.ny[s] * h0;

      const top1x = this.cx[s + 1] + this.nx[s + 1] * h1;
      const top1y = this.cy[s + 1] + this.ny[s + 1] * h1;
      const bot1x = this.cx[s + 1] - this.nx[s + 1] * h1;
      const bot1y = this.cy[s + 1] - this.ny[s + 1] * h1;

      const n0x = this.nx[s];
      const n0y = this.ny[s];
      const n1x = this.nx[s + 1];
      const n1y = this.ny[s + 1];

      writeQuadSmooth(
        bot0x, bot0y, zFront, 0, 0, 1,
        bot1x, bot1y, zFront, 0, 0, 1,
        top1x, top1y, zFront, 0, 0, 1,
        top0x, top0y, zFront, 0, 0, 1
      );
      writeQuadSmooth(
        bot1x, bot1y, zBack, 0, 0, -1,
        bot0x, bot0y, zBack, 0, 0, -1,
        top0x, top0y, zBack, 0, 0, -1,
        top1x, top1y, zBack, 0, 0, -1
      );
      writeQuadSmooth(
        top0x, top0y, zFront, n0x, n0y, 0,
        top1x, top1y, zFront, n1x, n1y, 0,
        top1x, top1y, zBack, n1x, n1y, 0,
        top0x, top0y, zBack, n0x, n0y, 0
      );
      writeQuadSmooth(
        bot0x, bot0y, zBack, -n0x, -n0y, 0,
        bot1x, bot1y, zBack, -n1x, -n1y, 0,
        bot1x, bot1y, zFront, -n1x, -n1y, 0,
        bot0x, bot0y, zFront, -n0x, -n0y, 0
      );
    }

    const hStart = SPRING_HALF_HEIGHT_N * 1.12;
    writeQuadSmooth(
      this.cx[0] - this.nx[0] * hStart, this.cy[0] - this.ny[0] * hStart, zBack, -this.ny[0], this.nx[0], 0,
      this.cx[0] - this.nx[0] * hStart, this.cy[0] - this.ny[0] * hStart, zFront, -this.ny[0], this.nx[0], 0,
      this.cx[0] + this.nx[0] * hStart, this.cy[0] + this.ny[0] * hStart, zFront, -this.ny[0], this.nx[0], 0,
      this.cx[0] + this.nx[0] * hStart, this.cy[0] + this.ny[0] * hStart, zBack, -this.ny[0], this.nx[0], 0
    );

    const hEnd = SPRING_HALF_HEIGHT_N * 0.85;
    const e = SPRING_SEGMENTS;
    writeQuadSmooth(
      this.cx[e] - this.nx[e] * hEnd, this.cy[e] - this.ny[e] * hEnd, zFront, this.ny[e], -this.nx[e], 0,
      this.cx[e] - this.nx[e] * hEnd, this.cy[e] - this.ny[e] * hEnd, zBack, this.ny[e], -this.nx[e], 0,
      this.cx[e] + this.nx[e] * hEnd, this.cy[e] + this.ny[e] * hEnd, zBack, this.ny[e], -this.nx[e], 0,
      this.cx[e] + this.nx[e] * hEnd, this.cy[e] + this.ny[e] * hEnd, zFront, this.ny[e], -this.nx[e], 0
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

  const collarShape = new THREE.Shape();
  collarShape.absarc(0, 0, SPRING_COLLAR_OUTER_R, 0, Math.PI * 2, false);
  collarShape.holes.push(makeCircleHole(0, 0, SPRING_COLLAR_INNER_R, 24));

  // Brass spacer rings separating the 5 spring eyes cleanly on SpringAbutmentPost
  const spacerShape = new THREE.Shape();
  spacerShape.absarc(0, 0, 0.056, 0, Math.PI * 2, false);
  spacerShape.holes.push(makeCircleHole(0, 0, 0.044, 24));

  for (let i = 0; i < LEVER_COUNT; i++) {
    const spring = new LeafSpringMesh(i, materials.phosphorBronze);
    springs.push(spring);
    group.add(spring.mesh);

    const cDepth = SPRING_HALF_THICK_Z * 2;
    const cRaw = new THREE.ExtrudeGeometry(collarShape, {
      depth: cDepth,
      bevelEnabled: true,
      bevelThickness: 0.001,
      bevelSize: 0.001,
      bevelSegments: 1,
      curveSegments: 24,
    });
    cRaw.translate(SPRING_POST_X, SPRING_POST_Y, getLeverZ(i) - cDepth * 0.5);
    const cGeo = applyMachinedNormals(cRaw);
    disposables.push(cGeo);
    const cMesh = new THREE.Mesh(cGeo, materials.phosphorBronze);
    cMesh.name = `SpringAnchorEye_${i}`;
    cMesh.castShadow = true;
    group.add(cMesh);

    if (i < LEVER_COUNT - 1) {
      const zMid = (getLeverZ(i) + getLeverZ(i + 1)) * 0.5;
      const sDepth = 0.014;
      const sRaw = new THREE.ExtrudeGeometry(spacerShape, {
        depth: sDepth,
        bevelEnabled: true,
        bevelThickness: 0.001,
        bevelSize: 0.001,
        bevelSegments: 1,
        curveSegments: 22,
      });
      sRaw.translate(SPRING_POST_X, SPRING_POST_Y, zMid - sDepth * 0.5);
      const sGeo = applyMachinedNormals(sRaw);
      disposables.push(sGeo);
      const sMesh = new THREE.Mesh(sGeo, materials.forendBrass);
      sMesh.name = `SpringPostSpacer_${i}`;
      group.add(sMesh);
    }
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
// 7. Internal Coaxial 5-Stepped Cam Barrel & Bolt-Talon Arm (Driven 3:1 by Gear C1!)
// =============================================================================

export interface CamBarrelAssembly {
  group: THREE.Group;
  dispose: () => void;
}

export function buildCamBarrelAssembly(materials: BoltMaterials): CamBarrelAssembly {
  const group = new THREE.Group();
  group.name = 'InternalCamBarrelAssembly';
  group.position.set(KEY_X, KEY_Y, 0);
  const disposables: THREE.BufferGeometry[] = [];

  // 7A. Inner Hardened-Steel Hollow Cam Shaft (z: -0.168 .. +0.162, sleeves over CamCenterGuidePin r=0.019!)
  // Keyed rigidly to CamBarrelDrivenPinion_C1 (z: +0.138..+0.155) and rotates 3x faster inside outer knob sleeve!
  const barrelGeo = makeClosedLatheZ(
    [
      [0.056, 0.0],
      [0.056, 0.023],
      [-0.164, 0.023],
      [-0.168, 0.027],
      [-0.168, 0.031],
      [-0.163, 0.032],
      [0.160, 0.032],
      [0.164, 0.026],
      [0.164, 0.0],
    ],
    32,
    false
  );
  disposables.push(barrelGeo);
  const barrelMesh = new THREE.Mesh(barrelGeo, materials.hardenedSteel);
  barrelMesh.name = 'InternalCamShaft';
  barrelMesh.castShadow = true;
  barrelMesh.receiveShadow = true;
  group.add(barrelMesh);

  // 7B. Rear Hardened-Steel Bolt-Talon Drive Arm (z: -0.152 .. -0.084, engages Deadbolt U-Talon!)
  const talonBevel = 0.003;
  const talonBitW = 0.034 - talonBevel * 2;
  const talonBitDepth = 0.064 - talonBevel * 2;
  const talonR = KEY_TALON_RADIUS - talonBevel;
  const talonShape = new THREE.Shape(
    makeRoundedRectPts(-talonBitW * 0.5, -talonR, talonBitW * 0.5, -0.024, 0.007, 4)
  );
  const talonRaw = new THREE.ExtrudeGeometry(talonShape, {
    depth: talonBitDepth,
    bevelEnabled: true,
    bevelThickness: talonBevel,
    bevelSize: talonBevel,
    bevelSegments: 2,
  });
  talonRaw.translate(0, 0, -0.118 - talonBitDepth * 0.5);
  const talonGeo = applyMachinedNormals(talonRaw);
  disposables.push(talonGeo);
  const talonMesh = new THREE.Mesh(talonGeo, materials.hardenedSteel);
  talonMesh.name = 'CamTalonDriveArm';
  talonMesh.castShadow = true;
  talonMesh.receiveShadow = true;
  group.add(talonMesh);

  // 7C. Five Stepped Concentric-Crown Brass Lever-Lifting Cam Lobes matching exact KEY_BIT_RADII[0..4]!
  const bitBevel = 0.0020;
  const bitHalfW = KEY_BIT_HALF_WIDTH - bitBevel;
  const stepZCore = 0.022 - bitBevel * 2;
  for (let i = 0; i < LEVER_COUNT; i++) {
    const rOuter = KEY_BIT_RADII[i] - bitBevel;
    const halfAng = Math.asin(Math.min(0.92, bitHalfW / rOuter));
    const toothPts: THREE.Vector2[] = [];
    const crownSegs = 8;
    for (let s = 0; s <= crownSegs; s++) {
      const a = -halfAng + (2 * halfAng * s) / crownSegs;
      toothPts.push(new THREE.Vector2(Math.sin(a) * rOuter, -Math.cos(a) * rOuter));
    }
    toothPts.push(new THREE.Vector2(bitHalfW, -0.026));
    toothPts.push(new THREE.Vector2(-bitHalfW, -0.026));

    const stepRaw = new THREE.ExtrudeGeometry(new THREE.Shape(ensureCCW(toothPts)), {
      depth: stepZCore,
      bevelEnabled: true,
      bevelThickness: bitBevel,
      bevelSize: bitBevel,
      bevelSegments: 2,
    });
    stepRaw.translate(0, 0, getLeverZ(i) - stepZCore * 0.5);
    const stepGeo = applyMachinedNormals(stepRaw);
    disposables.push(stepGeo);
    const stepMesh = new THREE.Mesh(stepGeo, materials.keyBrass);
    stepMesh.name = `CamLeverLobe_${i}`;
    stepMesh.castShadow = true;
    stepMesh.receiveShadow = true;
    group.add(stepMesh);
  }

  // Connecting cam spine web along Z joining talon & all 5 cam lobes solidly
  const webGeo = new THREE.BoxGeometry(0.018, 0.068, 0.256);
  webGeo.translate(0, -0.060, -0.008);
  disposables.push(webGeo);
  const webMesh = new THREE.Mesh(webGeo, materials.hardenedSteel);
  webMesh.name = 'CamBarrelSpineWeb';
  webMesh.castShadow = true;
  group.add(webMesh);

  return {
    group,
    dispose: () => {
      for (const g of disposables) g.dispose();
    },
  };
}

// =============================================================================
// 8. Heavy Architectural Satin-Brass Quarter-Turn Deadbolt Thumb-Turn Knob
// =============================================================================

export interface ThumbTurnAssembly {
  group: THREE.Group;
  hitMeshes: THREE.Mesh[];
  dispose: () => void;
}

export function buildThumbTurnAssembly(materials: BoltMaterials): ThumbTurnAssembly {
  const group = new THREE.Group();
  group.name = 'DeadboltThumbTurnAssembly';
  group.position.set(KEY_X, KEY_Y, 0);
  const disposables: THREE.BufferGeometry[] = [];
  const hitMeshes: THREE.Mesh[] = [];

  // 8A. Outer Hollow Brass/Steel Coaxial Spindle Sleeve (z: +0.162 .. +0.258)
  // Carries Gear B1 at z: 0.162..0.178, passes cleanly through ThumbTurnSpindleBridgeCock bearing,
  // and flares into a stepped architectural neck boss right proud of the bearing collar!
  const sleeveGeo = makeClosedLatheZ(
    [
      [0.204, 0.0], // solid cap front of inner cam shaft
      [0.204, 0.035], // internal clearance bore around coaxial inner steel cam shaft (r=0.032)
      [0.162, 0.035],
      [0.162, 0.048], // outer journal keying into Gear B1 (bore r=0.048)
      [0.208, 0.048],
      [0.216, 0.062], // stepped architectural neck collar proud of bridge bearing
      [0.236, 0.070],
      [0.252, 0.066],
      [0.258, 0.056],
      [0.258, 0.0],
    ],
    36,
    false
  );
  disposables.push(sleeveGeo);
  const sleeveMesh = new THREE.Mesh(sleeveGeo, materials.hardenedSteel);
  sleeveMesh.name = 'ThumbTurnSpindleSleeve';
  sleeveMesh.castShadow = true;
  sleeveMesh.receiveShadow = true;
  group.add(sleeveMesh);
  hitMeshes.push(sleeveMesh);

  // Turned brass neck accent ring seated cleanly around spindle neck
  const neckRingGeo = makeClosedLatheZ(
    [
      [0.210, 0.049],
      [0.210, 0.066],
      [0.232, 0.075],
      [0.246, 0.075],
      [0.254, 0.064],
      [0.254, 0.049],
    ],
    36,
    false
  );
  disposables.push(neckRingGeo);
  const neckRingMesh = new THREE.Mesh(neckRingGeo, materials.keyBrass);
  neckRingMesh.name = 'ThumbTurnNeckCollar';
  neckRingMesh.castShadow = true;
  neckRingMesh.receiveShadow = true;
  group.add(neckRingMesh);
  hitMeshes.push(neckRingMesh);

  // 8B. Sculpted Heavy Chamfered Satin-Brass Deadbolt Thumb-Turn Paddle Bar (z: +0.246 .. +0.398)
  // At knobAngleRad = 0 (Unlocked): stands VERTICAL (pointer prow at +Y 12 o'clock)
  // At knobAngleRad = PI/2 (Locked): swings 90 deg CW to HORIZONTAL (pointer prow at +X 3 o'clock inline with bolt!)
  const paddleShape = new THREE.Shape();
  paddleShape.moveTo(0.0, 0.188);
  paddleShape.bezierCurveTo(0.036, 0.188, 0.048, 0.145, 0.052, 0.088);
  paddleShape.bezierCurveTo(0.064, 0.036, 0.064, -0.036, 0.054, -0.088);
  paddleShape.bezierCurveTo(0.048, -0.145, 0.038, -0.178, 0.0, -0.178);
  paddleShape.bezierCurveTo(-0.038, -0.178, -0.048, -0.145, -0.054, -0.088);
  paddleShape.bezierCurveTo(-0.064, -0.036, -0.064, 0.036, -0.052, 0.088);
  paddleShape.bezierCurveTo(-0.048, 0.145, -0.036, 0.188, 0.0, 0.188);
  paddleShape.closePath();

  const paddleCoreZ = 0.096;
  const paddleRaw = new THREE.ExtrudeGeometry(paddleShape, {
    depth: paddleCoreZ,
    bevelEnabled: true,
    bevelThickness: 0.022,
    bevelSize: 0.016,
    bevelSegments: 4,
    curveSegments: 36,
  });
  paddleRaw.translate(0, 0, 0.262);
  const paddleGeo = applyMachinedNormals(paddleRaw, (42 * Math.PI) / 180);
  disposables.push(paddleGeo);
  const paddleMesh = new THREE.Mesh(paddleGeo, materials.keyBrass);
  paddleMesh.name = 'ThumbTurnBrassPaddle';
  paddleMesh.castShadow = true;
  paddleMesh.receiveShadow = true;
  group.add(paddleMesh);
  hitMeshes.push(paddleMesh);

  // Subtle raised ergonomic thumb-ridge crest running along the front face of the paddle
  const ridgeShape = new THREE.Shape();
  ridgeShape.moveTo(0.0, 0.158);
  ridgeShape.bezierCurveTo(0.020, 0.158, 0.026, 0.105, 0.028, 0.040);
  ridgeShape.bezierCurveTo(0.032, 0.0, 0.032, -0.035, 0.028, -0.082);
  ridgeShape.bezierCurveTo(0.025, -0.130, 0.019, -0.148, 0.0, -0.148);
  ridgeShape.bezierCurveTo(-0.019, -0.148, -0.025, -0.130, -0.028, -0.082);
  ridgeShape.bezierCurveTo(-0.032, -0.035, -0.032, 0.0, -0.028, 0.040);
  ridgeShape.bezierCurveTo(-0.026, 0.105, -0.020, 0.158, 0.0, 0.158);
  ridgeShape.closePath();

  const ridgeRaw = new THREE.ExtrudeGeometry(ridgeShape, {
    depth: 0.018,
    bevelEnabled: true,
    bevelThickness: 0.010,
    bevelSize: 0.008,
    bevelSegments: 3,
    curveSegments: 28,
  });
  ridgeRaw.translate(0, 0, 0.372);
  const ridgeGeo = applyMachinedNormals(ridgeRaw, (40 * Math.PI) / 180);
  disposables.push(ridgeGeo);
  const ridgeMesh = new THREE.Mesh(ridgeGeo, materials.keyBrass);
  ridgeMesh.name = 'ThumbTurnErgonomicCrest';
  ridgeMesh.castShadow = true;
  ridgeMesh.receiveShadow = true;
  group.add(ridgeMesh);
  hitMeshes.push(ridgeMesh);

  // Precision machined lateral grip ribs on both thumb flanks of the paddle
  for (const sideSign of [-1, 1]) {
    for (const ry of [-0.115, -0.075, 0.075, 0.115]) {
      const ribGeo = makeClosedLatheZ(
        [
          [0.272, 0.0065],
          [0.360, 0.0065],
          [0.366, 0.0035],
        ],
        12
      );
      ribGeo.translate(sideSign * 0.064, ry, 0);
      disposables.push(ribGeo);
      const ribMesh = new THREE.Mesh(ribGeo, materials.forendBrass);
      ribMesh.name = 'ThumbTurnGripRib';
      ribMesh.castShadow = true;
      group.add(ribMesh);
      hitMeshes.push(ribMesh);
    }
  }

  // Central blued-steel horological hub medallion & slotted set-screw on front of thumb-turn
  const hubCapGeo = makeClosedLatheZ(
    [
      [0.386, 0.036],
      [0.402, 0.036],
      [0.408, 0.030],
      [0.410, 0.0],
    ],
    28
  );
  disposables.push(hubCapGeo);
  const hubCapMesh = new THREE.Mesh(hubCapGeo, materials.bluedSteelBolt);
  hubCapMesh.name = 'ThumbTurnCenterCap';
  hubCapMesh.castShadow = true;
  group.add(hubCapMesh);
  hitMeshes.push(hubCapMesh);

  const hubScrew = buildSlottedBrassScrew(
    0.022,
    0.012,
    0.0065,
    0.006,
    materials.forendBrass,
    disposables
  );
  hubScrew.position.set(0, 0, 0.411);
  group.add(hubScrew);

  // Inlaid blued-steel directional pointer jewel near the +Y prow (points at 12 o'clock Unlocked / 3 o'clock Locked!)
  const prowPipGeo = makeClosedLatheZ(
    [
      [0.388, 0.013],
      [0.402, 0.013],
      [0.408, 0.008],
      [0.410, 0.0],
    ],
    20
  );
  prowPipGeo.translate(0, 0.132, 0);
  disposables.push(prowPipGeo);
  const prowPipMesh = new THREE.Mesh(prowPipGeo, materials.bluedSteelBolt);
  prowPipMesh.name = 'ThumbTurnDirectionPip';
  prowPipMesh.castShadow = true;
  group.add(prowPipMesh);
  hitMeshes.push(prowPipMesh);

  // Generous invisible grab-target proxy around the thumb-turn knob so dragging/clicking/scrolling feels effortless
  const hitProxyGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.26, 18);
  hitProxyGeo.rotateX(Math.PI / 2);
  hitProxyGeo.translate(0, 0, 0.31);
  disposables.push(hitProxyGeo);
  const hitProxyMat = new THREE.MeshBasicMaterial({ visible: false });
  const hitProxyMesh = new THREE.Mesh(hitProxyGeo, hitProxyMat);
  hitProxyMesh.name = 'KnobHitProxy';
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
