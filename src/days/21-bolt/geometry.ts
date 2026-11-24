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
  KEY_SHAFT_RADIUS,
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
  STUMP_HEIGHT,
  STUMP_WIDTH,
  STUMP_X_UNLOCKED,
  STUMP_Y,
  STUMP_Z_BACK,
  STUMP_Z_FRONT,
  getLeverZ,
} from './lockMechanism';
import type { BoltMaterials } from './materials';

const CREASE_ANGLE_RAD = (34 * Math.PI) / 180;

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
 * Computes precision machined normals for extruded parts (before any final orientation rotation):
 *  - All planar front/back extrusion cap faces (|nz| > 0.996) receive exact razor-flat (0, 0, ±1) normals
 *    so bevel rings NEVER bleed curved Gouraud triangle-fan streaks across flat metal plates or wood faces!
 *  - Non-cap faces (curved rims, rounded corners, bores, bevels) smoothly average normals across adjacent
 *    non-cap faces within `creaseAngleRad` so curved edges gleam smoothly without polygonal faceting.
 */
function applyMachinedNormals(
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
  const isCap = new Uint8Array(triCount); // 0 = bevel/wall, 1 = +Z cap, 2 = -Z cap

  // Bucket non-cap vertices by quantized 3D position for fast crease-angle smoothing
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

/**
 * Builds a rounded rectangle point list in CCW order.
 */
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
 * Computes an inward-offset version of a CCW open or closed 2D polygon by `inset` along averaged edge normals.
 * Used to place the extruded bevel outer skin of precision components onto exact mechanical coordinates.
 */
function offsetClosedPolygonCCW(pts: THREE.Vector2[], offset: number): THREE.Vector2[] {
  const n = pts.length;
  const out: THREE.Vector2[] = [];
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const curr = pts[i];
    const next = pts[(i + 1) % n];

    const e1x = curr.x - prev.x;
    const e1y = curr.y - prev.y;
    const l1 = Math.max(1e-7, Math.hypot(e1x, e1y));
    // Outward normal for CCW contour is (+dy, -dx)
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

/**
 * Builds a watertight outward-wound extruded prism along Z from `zBack` to `zFront` with
 * optional independent front (+Z) and back (-Z) bevels so flush-mated timber blocks have
 * ZERO perimeter V-groove seams where their interior faces join!
 */
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
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    cx: number,
    cy: number,
    cz: number
  ) => {
    positions.push(ax, ay, az, bx, by, bz, cx, cy, cz);
  };

  // Side walls between consecutive rings (r -> r+1 in increasing Z, CCW contour -> outward normal)
  for (let r = 0; r < rings.length - 1; r++) {
    const r0 = rings[r];
    const r1 = rings[r + 1];
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const a0 = r0.pts[i];
      const b0 = r0.pts[j];
      const b1 = r1.pts[j];
      const a1 = r1.pts[i];
      // Outward CCW when viewed from outside: a0 -> b0 -> b1 and a0 -> b1 -> a1
      pushTri(a0.x, a0.y, r0.z, b0.x, b0.y, r0.z, b1.x, b1.y, r1.z);
      pushTri(a0.x, a0.y, r0.z, b1.x, b1.y, r1.z, a1.x, a1.y, r1.z);
    }
  }

  // Front cap (+Z outward): triangulate last ring (CCW -> +Z normal)
  const frontRing = rings[rings.length - 1];
  const frontFaces = THREE.ShapeUtils.triangulateShape(frontRing.pts, []);
  for (const [i0, i1, i2] of frontFaces) {
    const p0 = frontRing.pts[i0];
    const p1 = frontRing.pts[i1];
    const p2 = frontRing.pts[i2];
    pushTri(p0.x, p0.y, frontRing.z, p1.x, p1.y, frontRing.z, p2.x, p2.y, frontRing.z);
  }

  // Back cap (-Z outward): triangulate first ring, reversed winding (CW -> -Z normal)
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

/**
 * Builds a watertight closed LatheGeometry along Z.
 * Profile points `[z, r]` are given in strict CCW order around the positive-R cross-section
 * (allowing both solid shafts and genuine hollow-bore pipe barrels with outward normals!).
 */
function makeClosedLatheZ(
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

/**
 * Creates a countersunk slotted flathead brass wood-screw seated cleanly inside a matching circular hole.
 * Built from two outward-wound beveled D-caps flanking a genuine empty screwdriver slot + recessed floor disc.
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

  // Backing countersunk base disc behind the slot (inset slightly in radius for clean bevel)
  const bevel = 0.0018;
  const baseR = Math.max(0.004, radius - bevel);
  const baseDepth = Math.max(0.006, headDepth - slotDepth - bevel);
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

  // Two opposing D-lobes forming the genuine recessed screwdriver slot (clean empty channel!)
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

// Exported spatial constants so unit tests can assert zero forend-to-oak overlap & flush seating
export const OAK_OUTER_MIN_X = -1.14;
export const OAK_OUTER_MAX_X = FOREND_X + FOREND_THICKNESS + 0.003; // +1.008 (exact flush match with brass forend front face!)
export const OAK_OUTER_MIN_Y = -0.845;
export const OAK_OUTER_MAX_Y = 0.865;
export const OAK_Z_BACK = -0.340;
export const OAK_Z_FRONT = 0.252;
export const OAK_REBATE_X = FOREND_X - 0.002; // +0.938 (0.002 crisp reveal behind forend back x = 0.940)
export const OAK_REBATE_MIN_Y = -0.768;
export const OAK_REBATE_MAX_Y = 0.788;
export const OAK_REBATE_Z_BACK = -0.250; // 0.004 crisp reveal behind forend rear face z = -0.246

export function buildOakSpecimenBlock(materials: BoltMaterials): OakSpecimenAssembly {
  const group = new THREE.Group();
  group.name = 'CharredOakSpecimenBlock';
  const disposables: THREE.BufferGeometry[] = [];

  const pocketMinX = CASE_INNER_MIN_X - 0.044; // -0.844
  const pocketMinY = CASE_INNER_MIN_Y - 0.039; // -0.604
  const pocketMaxY = CASE_INNER_MAX_Y + 0.039; // +0.624
  const caseBackZ = -0.224;

  // 1A. Full-Height Rear Oak Door-Stile Backing (z: -0.340 .. -0.250)
  // Chamfered ONLY on the exterior rear face (-Z); razor-flat at z = -0.250 so perimeter is 100% seamless!
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

  // 1B. Mid-Depth Mortise Floor & Forend Rebate Wall (z: -0.250 .. -0.224)
  // Extends flush to OAK_OUTER_MAX_X above/below forend, stepped cleanly back to OAK_REBATE_X behind the forend!
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

  // 1C. Front Cutaway Surround with Chiseled Mortise Pocket & Flush Top/Bottom Forend Shoulders (z: -0.224 .. +0.252)
  // Razor-flat at z = -0.224 (zero seam!) and crisply chamfered on exterior front cutaway face (+Z)!
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
export const FOREND_MIN_Y = -0.764;
export const FOREND_MAX_Y = 0.784;
export const FOREND_HALF_Z = 0.246;
export const FOREND_OUTER_FACE_X = FOREND_X + FOREND_THICKNESS + 0.003; // +1.008 (flush with oak!)
export const FOREND_SLOT_HALF_Y = 0.174; // clears deadbolt head (+/- 0.164) by 0.010 top & bottom
export const FOREND_SLOT_HALF_Z = BOLT_HEAD_Z_HALF + 0.01; // +/- 0.178 (clears deadbolt +/- 0.168 by 0.010)

export function buildForendAssembly(materials: BoltMaterials): ForendAssembly {
  const group = new THREE.Group();
  group.name = 'ForendBrassAssembly';
  const disposables: THREE.BufferGeometry[] = [];

  // Built in local XY (local +X -> world -Z, local +Y -> world +Y, extrude +Z -> world +X)
  // Inset outer XY by FOREND_BEVEL so total beveled dimensions match FOREND_HALF_Z / FOREND_MIN_Y..MAX_Y exactly!
  const halfZ = FOREND_HALF_Z - FOREND_BEVEL;
  const minY = FOREND_MIN_Y + FOREND_BEVEL;
  const maxY = FOREND_MAX_Y - FOREND_BEVEL;

  const outerPts = makeRoundedRectPts(-halfZ, minY, halfZ, maxY, 0.022, 6);
  const plateShape = new THREE.Shape(outerPts);

  // Precision rectangular through-slot expanded outward by FOREND_BEVEL so actual beveled opening
  // equals exact FOREND_SLOT_HALF_Z x FOREND_SLOT_HALF_Y (0.010 uniform clearance around deadbolt!)
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

  // Punch two real countersunk circular screw holes directly in the faceplate so screw slots are crisp & hollow!
  const screwRadius = 0.054;
  const screwPositions: Array<[number, number]> = [
    [0.678, 0.54], // top screw rotated +31 deg
    [-0.658, -0.80], // bottom screw rotated -46 deg
  ];
  for (const [sy] of screwPositions) {
    plateShape.holes.push(makeCircleHole(0, sy, screwRadius + FOREND_BEVEL + 0.001, 28));
  }

  // Exact thickness from x = FOREND_X (0.940) to x = FOREND_OUTER_FACE_X (1.008) including 2 * FOREND_BEVEL
  const coreDepth = FOREND_OUTER_FACE_X - FOREND_X - FOREND_BEVEL * 2;
  const plateRaw = new THREE.ExtrudeGeometry(plateShape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: FOREND_BEVEL,
    bevelSize: FOREND_BEVEL,
    bevelSegments: 2,
  });
  // Apply flat-cap machined normals BEFORE rotating Y so the brass forend front/back faces stay 100% planar!
  const plateGeo = applyMachinedNormals(plateRaw);
  plateGeo.rotateY(Math.PI / 2);
  plateGeo.translate(FOREND_X + FOREND_BEVEL, 0, 0);
  disposables.push(plateGeo);

  const plateMesh = new THREE.Mesh(plateGeo, materials.forendBrass);
  plateMesh.name = 'ForendFaceplate';
  plateMesh.castShadow = true;
  plateMesh.receiveShadow = true;
  group.add(plateMesh);

  // Two countersunk slotted brass wood-screws seated flush inside their machined holes!
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
    // Outer face of screw lobes sits cleanly at FOREND_OUTER_FACE_X + 0.0015 (flush countersunk!)
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
  const outerMaxX = FOREND_X - 0.001; // +0.939 (cleanly meets forend backplate at 0.940, zero overlap!)
  const outerMinY = CASE_INNER_MIN_Y - 0.035; // -0.60
  const outerMaxY = CASE_INNER_MAX_Y + 0.035; // +0.62

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

  // 3B. Perimeter Steel Case Walls (top, bottom, rear left; z: -0.180 .. +0.188)
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

  // Right inner throat walls strictly inside CASE_INNER_MIN_Y..CASE_INNER_MAX_Y (clears deadbolt +/- 0.176)
  const throatW = outerMaxX - CASE_INNER_MAX_X;
  const throatCenterX = (CASE_INNER_MAX_X + outerMaxX) * 0.5;
  const throatSlotHalfY = FOREND_SLOT_HALF_Y + 0.004;
  const throatTopH = Math.max(0.02, CASE_INNER_MAX_Y - (STUMP_Y + throatSlotHalfY));
  const throatGeoTop = new THREE.BoxGeometry(throatW, throatTopH, wallDepth);
  throatGeoTop.translate(throatCenterX, STUMP_Y + throatSlotHalfY + throatTopH * 0.5, 0.004);
  disposables.push(throatGeoTop);
  const throatTop = new THREE.Mesh(throatGeoTop, materials.gunmetalCase);
  throatTop.name = 'CaseThroatTop';
  throatTop.castShadow = true;
  throatTop.receiveShadow = true;
  group.add(throatTop);

  const throatBotH = Math.max(0.02, STUMP_Y - throatSlotHalfY - CASE_INNER_MIN_Y);
  const throatGeoBot = new THREE.BoxGeometry(throatW, throatBotH, wallDepth);
  throatGeoBot.translate(throatCenterX, CASE_INNER_MIN_Y + throatBotH * 0.5, 0.004);
  disposables.push(throatGeoBot);
  const throatBot = new THREE.Mesh(throatGeoBot, materials.gunmetalCase);
  throatBot.name = 'CaseThroatBottom';
  throatBot.castShadow = true;
  throatBot.receiveShadow = true;
  group.add(throatBot);

  // 3C. Skeletonized Museum-Cutaway Front Inspection Rim (z: +0.188 .. +0.220)
  // Integrates the lower keyhole mounting pedestal directly into the rim (zero coplanar bridge overlap!)
  // and punches 4 real countersunk circular holes for the corner brass cover screws.
  const rimBevel = 0.004;
  const bezelOuter = makeRoundedRectPts(
    outerMinX + rimBevel,
    outerMinY + rimBevel,
    outerMaxX - rimBevel,
    outerMaxY - rimBevel,
    0.018,
    4
  );
  const bezelShape = new THREE.Shape(bezelOuter);

  const winHolePts = ensureCW([
    new THREE.Vector2(-0.74, -0.49),
    new THREE.Vector2(-0.18, -0.49),
    new THREE.Vector2(-0.13, -0.42),
    new THREE.Vector2(-0.08, -0.325),
    new THREE.Vector2(0.16, -0.325),
    new THREE.Vector2(0.21, -0.42),
    new THREE.Vector2(0.26, -0.49),
    new THREE.Vector2(0.84, -0.49),
    new THREE.Vector2(0.84, 0.51),
    new THREE.Vector2(-0.74, 0.51),
  ]);
  bezelShape.holes.push(new THREE.Path(winHolePts));

  const cornerCoords: Array<[number, number, number]> = [
    [-0.785, 0.555, 0.42],
    [-0.785, -0.535, -0.68],
    [0.885, 0.555, 1.15],
    [0.885, -0.535, -0.25],
  ];
  const cornerScrewR = 0.028;
  for (const [cx, cy] of cornerCoords) {
    bezelShape.holes.push(makeCircleHole(cx, cy, cornerScrewR + rimBevel + 0.001, 24));
  }

  const bezelRaw = new THREE.ExtrudeGeometry(bezelShape, {
    depth: 0.024,
    bevelEnabled: true,
    bevelThickness: rimBevel,
    bevelSize: rimBevel,
    bevelSegments: 2,
  });
  bezelRaw.translate(0, 0, 0.192);
  const bezelGeo = applyMachinedNormals(bezelRaw);
  disposables.push(bezelGeo);
  const bezelMesh = new THREE.Mesh(bezelGeo, materials.gunmetalCase);
  bezelMesh.name = 'CutawayFrontRim';
  bezelMesh.castShadow = true;
  bezelMesh.receiveShadow = true;
  group.add(bezelMesh);

  // 4 Corner Case Cover Screws seated flush inside their punched countersunk rim holes!
  for (const [cx, cy, ang] of cornerCoords) {
    const screw = buildSlottedBrassScrew(
      cornerScrewR,
      0.018,
      0.0085,
      0.008,
      materials.forendBrass,
      disposables
    );
    screw.rotation.z = ang;
    screw.position.set(cx, cy, 0.221);
    group.add(screw);
  }

  // 3D. Recessed Satin-Brass Keyhole Escutcheon Ring (mounted cleanly on pedestal at z: +0.200..+0.238)
  const escOuter: THREE.Vector2[] = [];
  const escSegs = 40;
  for (let i = 0; i < escSegs; i++) {
    const a = (i / escSegs) * Math.PI * 2;
    escOuter.push(new THREE.Vector2(Math.cos(a) * 0.114, Math.sin(a) * 0.114));
  }
  const escShape = new THREE.Shape(ensureCCW(escOuter));
  const khPts: THREE.Vector2[] = [];
  const barrelR = KEY_SHAFT_RADIUS + 0.016;
  const slotHalfW = 0.024;
  const slotAngle = Math.asin(slotHalfW / barrelR);
  const khArcSegs = 26;
  for (let s = 0; s <= khArcSegs; s++) {
    const a = -Math.PI / 2 + slotAngle + (Math.PI * 2 - 2 * slotAngle) * (s / khArcSegs);
    khPts.push(new THREE.Vector2(Math.cos(a) * barrelR, Math.sin(a) * barrelR));
  }
  khPts.push(new THREE.Vector2(-slotHalfW, -0.100));
  khPts.push(new THREE.Vector2(slotHalfW, -0.100));
  escShape.holes.push(new THREE.Path(ensureCW(khPts)));

  const escRaw = new THREE.ExtrudeGeometry(escShape, {
    depth: 0.028,
    bevelEnabled: true,
    bevelThickness: 0.005,
    bevelSize: 0.005,
    bevelSegments: 2,
  });
  escRaw.translate(KEY_X, KEY_Y, 0.205);
  const escGeo = applyMachinedNormals(escRaw);
  disposables.push(escGeo);
  const escMesh = new THREE.Mesh(escGeo, materials.forendBrass);
  escMesh.name = 'KeyholeEscutcheon';
  escMesh.castShadow = true;
  escMesh.receiveShadow = true;
  group.add(escMesh);

  // 3E. Beveled Rear Brass Bolt Wear-Rail + 2 Turned Steel Rivets (z: -0.180 .. -0.168)
  const railShape = new THREE.Shape(makeRoundedRectPts(-0.695, STUMP_Y - 0.064, 0.815, STUMP_Y + 0.064, 0.016, 4));
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

  // 3F. Hardened Steel Lever Pivot Post + Rear Thrust Bushing + 4 Inter-Lever Brass Spacer Washers + Front Collar
  const pivotPinGeo = makeClosedLatheZ([
    [-0.182, 0.056],
    [0.164, 0.056],
    [0.167, 0.074],
    [0.188, 0.074],
    [0.195, 0.054],
    [0.199, 0.036],
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

  // Rear solid brass thrust stand-off bushing supporting Lever 0 off the case backplate (z: -0.180 .. -0.054)
  const rearBushingGeo = makeClosedLatheZ(
    [
      [-0.180, 0.057],
      [-0.180, 0.098],
      [-0.058, 0.098],
      [-0.054, 0.092],
      [-0.054, 0.057],
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

  // 4 Precision Inter-Lever Brass Spacer Washers (exact 0.0018 daylight gap per side, zero overlap)
  for (let w = 0; w < LEVER_COUNT - 1; w++) {
    const zMid = (getLeverZ(w) + getLeverZ(w + 1)) * 0.5;
    const wThickness = 0.014;
    const wRaw = new THREE.ExtrudeGeometry(washerShape, {
      depth: wThickness,
      bevelEnabled: true,
      bevelThickness: 0.0012,
      bevelSize: 0.0012,
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

  // Front chamfered brass pivot retaining collar seated flush against Lever 4 and Pivot Post flange (z: +0.150 .. +0.164)
  const capWasherRaw = new THREE.ExtrudeGeometry(washerShape, {
    depth: 0.010,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.002,
    bevelSegments: 2,
    curveSegments: 28,
  });
  capWasherRaw.translate(PIVOT_X, PIVOT_Y, getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.004);
  const capWasherGeo = applyMachinedNormals(capWasherRaw);
  disposables.push(capWasherGeo);
  const capWasherMesh = new THREE.Mesh(capWasherGeo, materials.forendBrass);
  capWasherMesh.name = 'PivotFrontWasher';
  capWasherMesh.castShadow = true;
  capWasherMesh.receiveShadow = true;
  group.add(capWasherMesh);

  // 3G. Hardened Steel Spring Abutment Post (anchors all 5 phosphor-bronze leaf spring eyes!)
  const springPostGeo = makeClosedLatheZ([
    [-0.182, 0.044],
    [0.168, 0.044],
    [0.178, 0.055],
    [0.190, 0.047],
    [0.194, 0.032],
  ]);
  springPostGeo.translate(SPRING_POST_X, SPRING_POST_Y, 0);
  disposables.push(springPostGeo);
  const springPostMesh = new THREE.Mesh(springPostGeo, materials.hardenedSteel);
  springPostMesh.name = 'SpringAbutmentPost';
  springPostMesh.castShadow = true;
  springPostMesh.receiveShadow = true;
  group.add(springPostMesh);

  // 3H. Lever Bottom Rest Stop Pin with Machined Satin-Brass Sleeve & Chamfered Tip
  const restPinGeo = makeClosedLatheZ([
    [-0.182, REST_PIN_RADIUS],
    [0.154, REST_PIN_RADIUS],
    [0.160, REST_PIN_RADIUS + 0.005],
    [0.172, REST_PIN_RADIUS + 0.005],
    [0.178, REST_PIN_RADIUS - 0.007],
  ]);
  restPinGeo.translate(REST_PIN_X, REST_PIN_Y, 0);
  disposables.push(restPinGeo);
  const restPinMesh = new THREE.Mesh(restPinGeo, materials.forendBrass);
  restPinMesh.name = 'LeverRestStopPin';
  restPinMesh.castShadow = true;
  restPinMesh.receiveShadow = true;
  group.add(restPinMesh);

  // 3I. Hardened Bolt-Tail Guide Pin (rides cleanly inside the horizontal guide slot of the deadbolt tail)
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

  // 3J. Hardened Key Guide Drill-Pin (projects forward from backplate into the hollow bore of the pipe key!)
  const drillPinGeo = makeClosedLatheZ([
    [-0.182, 0.025],
    [0.026, 0.025],
    [0.042, 0.017],
  ]);
  drillPinGeo.translate(KEY_X, KEY_Y, 0);
  disposables.push(drillPinGeo);
  const drillPinMesh = new THREE.Mesh(drillPinGeo, materials.hardenedSteel);
  drillPinMesh.name = 'KeyDrillPin';
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

export const BOLT_HEAD_HALF_Y = 0.164; // Exact outer half-height INCLUDING bevel (clears forend 0.174 by 0.010!)
export const BOLT_TALON_JAW_BOTTOM_Y = 0.024; // Clears rotating KeyTalonDriveBit across full 360° with zero clip!

export function buildDeadboltAssembly(materials: BoltMaterials): DeadboltAssembly {
  const group = new THREE.Group();
  group.name = 'DeadboltAssembly';
  const disposables: THREE.BufferGeometry[] = [];
  const hitMeshes: THREE.Mesh[] = [];

  // 4A. Massive Chamfered Square Blued-Steel Deadbolt Head
  // Inset shape by headBevel so outer beveled extents are EXACTLY [0.510, 1.022] x [STUMP_Y ± 0.164] x [±0.168]!
  const headBevel = 0.012;
  const headMinX = 0.510 + headBevel;
  const headMaxX = 1.022 - headBevel; // At unlocked (0), nose protrudes 0.014 chamfer cleanly past forend 1.008; locked -> 1.362!
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

  // 4B. Machined Blued-Steel Bolt Tail (Shank) with Horizontal Guide Slot & Flared U-Shaped Bolt Talon
  // Centered cleanly in Z behind Lever 0: [BOLT_TAIL_Z_BACK, BOLT_TAIL_Z_FRONT]
  const tailBevel = 0.004;
  const tailDepth = BOLT_TAIL_Z_FRONT - BOLT_TAIL_Z_BACK - tailBevel * 2;
  const tailZMid = (BOLT_TAIL_Z_BACK + BOLT_TAIL_Z_FRONT) * 0.5;

  // Bolt talon U-notch centered at x_talon_local = KEY_X - BOLT_TRAVEL * 0.5 = -0.13
  // Flared entry mouth from y = +0.024..+0.048 clears the sweeping arc of KeyTalonDriveBit with 0 clipping!
  const talonCenterX = KEY_X - 0.17; // -0.13
  const notchHalfW = 0.046 + tailBevel; // actual beveled inner U-wall = 0.046
  const mouthFlareW = 0.076 + tailBevel; // wide chamfered entry mouth clears R=0.308 arc at y=0.024
  const jawW = 0.064;
  const spineMinY = STUMP_Y - 0.084; // +0.076
  const spineMaxY = STUMP_Y + 0.084; // +0.244
  const jawBottomY = BOLT_TALON_JAW_BOTTOM_Y + tailBevel; // actual beveled bottom = +0.024
  const flareTopY = 0.048;
  const notchTopY = 0.104 + tailBevel; // actual beveled ceiling = +0.104 (clears bit tip +0.088 by 0.016!)

  const tailPts = ensureCCW([
    new THREE.Vector2(-0.765, spineMinY + 0.012),
    new THREE.Vector2(-0.753, spineMinY),
    // Left talon jaw outer cheek & bottom shoulder
    new THREE.Vector2(talonCenterX - mouthFlareW - jawW - 0.018, spineMinY),
    new THREE.Vector2(talonCenterX - mouthFlareW - jawW, jawBottomY + 0.016),
    new THREE.Vector2(talonCenterX - mouthFlareW - jawW + 0.014, jawBottomY),
    new THREE.Vector2(talonCenterX - mouthFlareW, jawBottomY),
    // Flared left U-notch mouth rising into vertical left jaw face
    new THREE.Vector2(talonCenterX - notchHalfW, flareTopY),
    new THREE.Vector2(talonCenterX - notchHalfW, notchTopY - 0.012),
    new THREE.Vector2(talonCenterX - notchHalfW + 0.012, notchTopY),
    // U-notch ceiling & vertical right jaw face
    new THREE.Vector2(talonCenterX + notchHalfW - 0.012, notchTopY),
    new THREE.Vector2(talonCenterX + notchHalfW, notchTopY - 0.012),
    new THREE.Vector2(talonCenterX + notchHalfW, flareTopY),
    // Flared right U-notch mouth & right talon jaw outer cheek
    new THREE.Vector2(talonCenterX + mouthFlareW, jawBottomY),
    new THREE.Vector2(talonCenterX + mouthFlareW + jawW - 0.014, jawBottomY),
    new THREE.Vector2(talonCenterX + mouthFlareW + jawW, jawBottomY + 0.016),
    new THREE.Vector2(talonCenterX + mouthFlareW + jawW + 0.018, spineMinY),
    // Meet bolt head shoulder cleanly inside headMinX
    new THREE.Vector2(headMinX + 0.024, spineMinY),
    new THREE.Vector2(headMinX + 0.024, spineMaxY),
    new THREE.Vector2(-0.753, spineMaxY),
    new THREE.Vector2(-0.765, spineMaxY - 0.012),
  ]);

  const tailShape = new THREE.Shape(tailPts);

  // Horizontal guide slot riding cleanly over fixed steel guide pin (GUIDE_PIN_X = -0.34, r = 0.030)
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

  // 4C. Hardened Square Steel Bolt Stump (inset by stumpBevel so exact outer bounds == STUMP_WIDTH x STUMP_HEIGHT!)
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

  // Brass peened rivet boss on front face of hardened bolt stump
  const rivetGeo = makeClosedLatheZ([
    [STUMP_Z_FRONT - 0.002, 0.022],
    [STUMP_Z_FRONT + 0.007, 0.020],
    [STUMP_Z_FRONT + 0.012, 0.012],
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
// 5. Five Stacked Precision Brass Lever Tumblers (Concentric Chubb Bellies & Multi-Tooth Anti-Pick)
// =============================================================================

export interface LeverPackAssembly {
  group: THREE.Group;
  leverPivots: THREE.Group[];
  leverMeshes: THREE.Mesh[];
  dispose: () => void;
}

export const LEVER_TOP_CREST_LOCAL_Y = 0.264; // Exact outer beveled top surface of all 5 levers

/**
 * Constructs one precision-machined Chubb brass lever plate in lever-local coordinates
 * (pivot bore centered at local (0, 0), plate thickness centered at Z = 0).
 *
 * Uses `LEVER_BELLY_POLYLINES[index]` directly as the lower contour (inset by LEVER_BEVEL so the
 * extruded beveled brass skin sits at exact physics contact coordinates!) and mills an exact
 * perpendicular-compensated sloped H-gate with multi-tooth anti-pick false serrations on both cheeks.
 */
export function createLeverGeometry(index: number): THREE.BufferGeometry {
  const spec = LEVER_GATE_SPECS[index];
  const belly = LEVER_BELLY_POLYLINES[index];

  // Outer contour in CCW order, inset by LEVER_BEVEL along outward normals so actual beveled outer skin
  // lands cleanly on nominal physics dimensions (top = +0.264, heel = LEVER_HEEL_LOCAL_Y, belly = exact arc!)
  const nominalOuter: THREE.Vector2[] = [];

  // Left semicircular pivot ear: angle from +PI/2 (top) around PI (left) to -PI/2 (bottom)
  const earR = 0.124;
  const earSegs = 16;
  for (let s = 0; s <= earSegs; s++) {
    const a = Math.PI * 0.5 + Math.PI * (s / earSegs);
    nominalOuter.push(new THREE.Vector2(Math.cos(a) * earR, Math.sin(a) * earR));
  }

  // Exact single-source-of-truth Chubb belly polyline (entry shoulder -> concentric dwell arc -> exit -> rest heel)
  for (let i = 0; i < belly.length; i++) {
    nominalOuter.push(new THREE.Vector2(belly[i].x, belly[i].y));
  }

  // Lower-right chamfered corner & right vertical locking cheek edge (clears bolt head +0.51 -> local +1.09!)
  const lastBellyY = belly[belly.length - 1].y;
  nominalOuter.push(new THREE.Vector2(1.018, lastBellyY + 0.022));
  nominalOuter.push(new THREE.Vector2(1.018, LEVER_TOP_CREST_LOCAL_Y - 0.026));
  nominalOuter.push(new THREE.Vector2(0.992, LEVER_TOP_CREST_LOCAL_Y));

  // Flat upper spring-cam crest (x in [0.42, 0.992]) with locksmith assembly step per lever (0..4)
  const shoulderStepX = 0.24 + index * 0.036;
  nominalOuter.push(new THREE.Vector2(0.42, LEVER_TOP_CREST_LOCAL_Y));
  nominalOuter.push(new THREE.Vector2(shoulderStepX + 0.035, 0.234));
  nominalOuter.push(new THREE.Vector2(shoulderStepX, 0.200));
  nominalOuter.push(new THREE.Vector2(0.11, 0.154));

  const insetOuter = offsetClosedPolygonCCW(ensureCCW(nominalOuter), -LEVER_BEVEL);
  const shape = new THREE.Shape(ensureCCW(insetOuter));

  // Hole 1: Precision circular pivot bore (expanded by LEVER_BEVEL so actual beveled bore = 0.058 > post 0.056)
  shape.holes.push(makeCircleHole(0, 0, 0.058 + LEVER_BEVEL, 28));

  // Hole 2: Authentic Chubb H-Gate aperture with perpendicular-compensated sloped gate & multi-tooth anti-pick!
  // Because ExtrudeGeometry bevelSize shrinks hole interiors inward by LEVER_BEVEL, expand every hole wall
  // outward by LEVER_BEVEL so the rendered brass surfaces sit at exact GATE_POCKET_* coordinates!
  const b = LEVER_BEVEL;
  const xLOut = GATE_POCKET_LEFT_OUTER_X - b;
  const xLIn = GATE_POCKET_LEFT_INNER_X + b;
  const xRIn = GATE_POCKET_RIGHT_INNER_X - b;
  const xROut = GATE_POCKET_RIGHT_OUTER_X + b;

  // Exact tilted gate centerline at left & right gate bridge mouths
  const spanX = Math.max(1e-4, spec.gateLockedLocalX - spec.gateUnlockedLocalX);
  const slopeY = (spec.gateLockedLocalY - spec.gateUnlockedLocalY) / spanX;
  const yGateL = spec.gateUnlockedLocalY + (xLIn - spec.gateUnlockedLocalX) * slopeY;
  const yGateR = spec.gateUnlockedLocalY + (xRIn - spec.gateUnlockedLocalX) * slopeY;

  // Exact perpendicular gate half-height compensated by 1/cos(alignRad) + LEVER_BEVEL!
  const vertGateHalfH = (GATE_HALF_HEIGHT + b) / Math.cos(spec.alignRad);
  const gateTopL = yGateL + vertGateHalfH;
  const gateBotL = yGateL - vertGateHalfH;
  const gateTopR = yGateR + vertGateHalfH;
  const gateBotR = yGateR - vertGateHalfH;

  const yTopPocket = GATE_POCKET_TOP_Y + b;
  const yBotPocketL = Math.max(-0.088, Math.min(gateBotL - 0.016, -0.015)) - b;
  const yBotPocketR = GATE_POCKET_BOT_R_Y - b;

  // Helper to append crisp multi-tooth Chubb anti-pick false serrations along a vertical cheek edge
  // (goes strictly monotonically from yStart -> yEnd; notchDir = -1 bites left into right cheek, +1 bites right into left cheek)
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

  // Trace H-gate perimeter cleanly in CCW order, then convert to CW hole path
  const hPts: THREE.Vector2[] = [
    new THREE.Vector2(xLOut, yTopPocket),
    new THREE.Vector2(xLOut, yBotPocketL),
    new THREE.Vector2(xLIn, yBotPocketL),
  ];

  // Lower-left cheek anti-pick serrations (yBotPocketL -> gateBotL)
  addAntiPickTeeth(hPts, xLIn, yBotPocketL, gateBotL, 1);
  hPts.push(new THREE.Vector2(xLIn, gateBotL));
  hPts.push(new THREE.Vector2(xRIn, gateBotR));

  // Lower-right cheek anti-pick serrations (gateBotR -> yBotPocketR)
  addAntiPickTeeth(hPts, xRIn, gateBotR, yBotPocketR, -1);
  hPts.push(new THREE.Vector2(xRIn, yBotPocketR));
  hPts.push(new THREE.Vector2(xROut, yBotPocketR));
  hPts.push(new THREE.Vector2(xROut, yTopPocket));
  hPts.push(new THREE.Vector2(xRIn, yTopPocket));

  // Upper-right cheek anti-pick serrations (yTopPocket -> gateTopR)
  addAntiPickTeeth(hPts, xRIn, yTopPocket, gateTopR, -1);
  hPts.push(new THREE.Vector2(xRIn, gateTopR));
  hPts.push(new THREE.Vector2(xLIn, gateTopL));

  // Upper-left cheek anti-pick serrations (gateTopL -> yTopPocket)
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
// 6. Five Dynamically Flexing Curved Phosphor-Bronze Leaf Springs (Zero Per-Frame GC)
// =============================================================================

const SPRING_SEGMENTS = 30;
const SPRING_HALF_THICK_Z = 0.0082;
const SPRING_HALF_HEIGHT_N = 0.0082;
export const SPRING_COLLAR_OUTER_R = 0.062;
export const SPRING_COLLAR_INNER_R = 0.045;

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
 * Pure analytical spring curve evaluator shared by LeafSpringMesh and unit tests:
 *  - Anchors tangentially off the outer bronze collar (r = 0.056 > steel post 0.044)
 *  - Slides smoothly outward along the lever top flat crest as `leverAngleRad` increases (near-constant arc length!)
 *  - Arrives strictly TANGENT to the rotated lever top shoulder with flush 0.000 contact
 *  - Keeps the entire deflected arch cleanly below CASE_INNER_MAX_Y - 0.025
 */
export function evaluateLeafSpringCurve(
  leverAngleRad: number,
  outP0: { x: number; y: number },
  outP1: { x: number; y: number },
  outP2: { x: number; y: number },
  outP3: { x: number; y: number },
  outContact?: { x: number; y: number }
): void {
  const cosA = Math.cos(leverAngleRad);
  const sinA = Math.sin(leverAngleRad);
  const liftRatio = Math.max(0, Math.min(1, leverAngleRad / LEVER_MAX_RAD));

  // Anchor smoothly off upper-right outer perimeter of bronze anchor collar (radius ~ 0.056 > post 0.044)
  outP0.x = SPRING_POST_X + 0.018;
  outP0.y = SPRING_POST_Y + 0.052;

  // Sliding contact point glides smoothly rightward along lever top flat crest [0.54 .. 0.78]
  // to preserve physical phosphor-bronze blade arc length (< 3% delta across full 0..21 deg lift!)
  const contactLocalX = 0.54 + liftRatio * 0.245;
  const contactLocalY = LEVER_TOP_CREST_LOCAL_Y;
  const cx = PIVOT_X + contactLocalX * cosA - contactLocalY * sinA;
  const cy = PIVOT_Y + contactLocalX * sinA + contactLocalY * cosA;
  if (outContact) {
    outContact.x = cx;
    outContact.y = cy;
  }

  // Rotated lever unit normal (-sinA, cosA) and unit tangent (+cosA, +sinA)
  const lnx = -sinA;
  const lny = cosA;
  const ltx = cosA;
  const lty = sinA;

  // Tip centerline offset along lever normal by exact end half-height so bottom face kisses flush (0 gap, 0 clip!)
  const hEnd = SPRING_HALF_HEIGHT_N * 0.85;
  outP3.x = cx + lnx * hEnd;
  outP3.y = cy + lny * hEnd;

  // Tangential arrival control point: strictly parallel to lever top edge with tiny upward clearance (+0.006 * N)
  // so the blade sweeps cleanly ABOVE the lever crest for all t < 1 and lands flush tangent at t = 1!
  const tanLead = 0.175 + liftRatio * 0.035;
  outP2.x = outP3.x - ltx * tanLead + lnx * 0.006;
  outP2.y = outP3.y - lty * tanLead + lny * 0.006;

  // Gentle cantilever arch bowing smoothly upward with compression while staying well under CASE_INNER_MAX_Y (0.585)
  const archTargetY = Math.min(CASE_INNER_MAX_Y - 0.032, outP0.y + 0.018 + liftRatio * 0.044);
  outP1.x = outP0.x + 0.215 + liftRatio * 0.055;
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

  /**
   * Computes exact diagnostic clearance/contact metrics for unit tests without allocations.
   */
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
      // Signed height of spring bottom above lever's rotated top plane (only over lever top crest x >= 0.42)
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

  /**
   * Recomputes the curved phosphor-bronze leaf spring in-place (zero allocations)
   * with smooth curved longitudinal normals and crisp hard side normals.
   */
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
      this._contact
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
      // Tri 1: a -> b -> c
      this.positions[ptr] = ax; this.positions[ptr + 1] = ay; this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = bx; this.positions[ptr + 4] = by; this.positions[ptr + 5] = bz;
      this.positions[ptr + 6] = cx; this.positions[ptr + 7] = cy; this.positions[ptr + 8] = cz;
      this.normals[ptr] = nax; this.normals[ptr + 1] = nay; this.normals[ptr + 2] = naz;
      this.normals[ptr + 3] = nbx; this.normals[ptr + 4] = nby; this.normals[ptr + 5] = nbz;
      this.normals[ptr + 6] = ncx; this.normals[ptr + 7] = ncy; this.normals[ptr + 8] = ncz;
      ptr += 9;

      // Tri 2: a -> c -> d
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

      // Front face (+Z outward)
      writeQuadSmooth(
        bot0x, bot0y, zFront, 0, 0, 1,
        bot1x, bot1y, zFront, 0, 0, 1,
        top1x, top1y, zFront, 0, 0, 1,
        top0x, top0y, zFront, 0, 0, 1
      );
      // Back face (-Z outward)
      writeQuadSmooth(
        bot1x, bot1y, zBack, 0, 0, -1,
        bot0x, bot0y, zBack, 0, 0, -1,
        top0x, top0y, zBack, 0, 0, -1,
        top1x, top1y, zBack, 0, 0, -1
      );
      // Top curved face (+N outward, smooth vertex normals along spring arc!)
      writeQuadSmooth(
        top0x, top0y, zFront, n0x, n0y, 0,
        top1x, top1y, zFront, n1x, n1y, 0,
        top1x, top1y, zBack, n1x, n1y, 0,
        top0x, top0y, zBack, n0x, n0y, 0
      );
      // Bottom curved face (-N outward, smooth vertex normals along spring arc!)
      writeQuadSmooth(
        bot0x, bot0y, zBack, -n0x, -n0y, 0,
        bot1x, bot1y, zBack, -n1x, -n1y, 0,
        bot1x, bot1y, zFront, -n1x, -n1y, 0,
        bot0x, bot0y, zFront, -n0x, -n0y, 0
      );
    }

    // Start cap (s = 0, facing -tangent)
    const hStart = SPRING_HALF_HEIGHT_N * 1.12;
    const t0x = this.cx[0] + this.nx[0] * hStart;
    const t0y = this.cy[0] + this.ny[0] * hStart;
    const b0x = this.cx[0] - this.nx[0] * hStart;
    const b0y = this.cy[0] - this.ny[0] * hStart;
    writeQuadSmooth(
      b0x, b0y, zBack, -this.ny[0], this.nx[0], 0,
      b0x, b0y, zFront, -this.ny[0], this.nx[0], 0,
      t0x, t0y, zFront, -this.ny[0], this.nx[0], 0,
      t0x, t0y, zBack, -this.ny[0], this.nx[0], 0
    );

    // End cap (s = SPRING_SEGMENTS, facing +tangent)
    const hEnd = SPRING_HALF_HEIGHT_N * 0.85;
    const e = SPRING_SEGMENTS;
    const tEx = this.cx[e] + this.nx[e] * hEnd;
    const tEy = this.cy[e] + this.ny[e] * hEnd;
    const bEx = this.cx[e] - this.nx[e] * hEnd;
    const bEy = this.cy[e] - this.ny[e] * hEnd;
    writeQuadSmooth(
      bEx, bEy, zFront, this.ny[e], -this.nx[e], 0,
      bEx, bEy, zBack, this.ny[e], -this.nx[e], 0,
      tEx, tEy, zBack, this.ny[e], -this.nx[e], 0,
      tEx, tEy, zFront, this.ny[e], -this.nx[e], 0
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

  // Curled bronze retaining anchor eyes wrapped cleanly around the steel spring abutment post
  const collarShape = new THREE.Shape();
  collarShape.absarc(0, 0, SPRING_COLLAR_OUTER_R, 0, Math.PI * 2, false);
  collarShape.holes.push(makeCircleHole(0, 0, SPRING_COLLAR_INNER_R, 24));

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
// 7. Ornate 19th-Century Victorian 5-Bitted Hollow-Pipe Warded Mortise Key
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

  // 7A. Genuine Hollow-Pipe Turned Hardened-Steel Barrel (bore r = 0.030 cleanly sleeves over KeyDrillPin r = 0.025!)
  // Profile ordered strictly along interior bore -> rear lip -> exterior turned Victorian mouldings -> front neck!
  const shaftGeo = makeClosedLatheZ(
    [
      [0.056, 0.0], // solid internal bore ceiling forward of drill pin
      [0.056, 0.030], // hollow pipe inner wall (clears drill pin r=0.025 by 0.005!)
      [-0.166, 0.030],
      [-0.170, 0.034], // crisp chamfered rear pipe lip
      [-0.170, 0.041],
      [-0.165, 0.044], // outer pipe barrel
      [0.236, 0.044],
      [0.246, 0.060], // Victorian stepped ward collar
      [0.268, 0.068],
      [0.288, 0.068],
      [0.302, 0.052],
      [0.316, 0.045],
      [0.455, 0.045],
      [0.478, 0.062], // turned neck moulding ring
      [0.496, 0.066],
      [0.514, 0.052],
      [0.545, 0.048],
      [0.575, 0.064],
      [0.575, 0.0],
    ],
    36,
    false
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
  const talonBevel = 0.003;
  const talonBitW = 0.034 - talonBevel * 2;
  const talonBitDepth = 0.064 - talonBevel * 2;
  const talonR = KEY_TALON_RADIUS - talonBevel;
  const talonShape = new THREE.Shape(
    makeRoundedRectPts(-talonBitW * 0.5, -talonR, talonBitW * 0.5, -0.034, 0.007, 4)
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
  talonMesh.name = 'KeyTalonDriveBit';
  talonMesh.castShadow = true;
  talonMesh.receiveShadow = true;
  group.add(talonMesh);
  hitMeshes.push(talonMesh);

  // 5 Stepped Lever Bitting Teeth with true concentric-arc crowns matching exact KEY_BIT_RADII[0..4]!
  const bitBevel = 0.0022;
  const bitHalfW = KEY_BIT_HALF_WIDTH - bitBevel;
  const stepZCore = 0.024 - bitBevel * 2;
  for (let i = 0; i < LEVER_COUNT; i++) {
    const rOuter = KEY_BIT_RADII[i] - bitBevel; // outer beveled crown sits at exact KEY_BIT_RADII[i]!
    const halfAng = Math.asin(Math.min(0.92, bitHalfW / rOuter));
    const toothPts: THREE.Vector2[] = [];
    // Bottom concentric circular arc crown from left (-halfAng) to right (+halfAng) around -Y
    const crownSegs = 8;
    for (let s = 0; s <= crownSegs; s++) {
      const a = -halfAng + (2 * halfAng * s) / crownSegs;
      toothPts.push(new THREE.Vector2(Math.sin(a) * rOuter, -Math.cos(a) * rOuter));
    }
    toothPts.push(new THREE.Vector2(bitHalfW, -0.036));
    toothPts.push(new THREE.Vector2(-bitHalfW, -0.036));

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
    stepMesh.name = `KeyBittingStep_${i}`;
    stepMesh.castShadow = true;
    stepMesh.receiveShadow = true;
    group.add(stepMesh);
    hitMeshes.push(stepMesh);
  }

  // Connecting warded key-bit bridge web along Z just below the shaft (narrower than brass steps -> zero z-fighting)
  const webGeo = new THREE.BoxGeometry(0.020, 0.078, 0.282);
  webGeo.translate(0, -0.070, 0.006);
  disposables.push(webGeo);
  const webMesh = new THREE.Mesh(webGeo, materials.hardenedSteel);
  webMesh.name = 'KeyBitWeb';
  webMesh.castShadow = true;
  group.add(webMesh);
  hitMeshes.push(webMesh);

  // 7C. High-Resolution Sculpted Victorian Satin-Brass Oval Bow (Handle)
  // Built cleanly with high curve resolution + crease-angle normals for liquid-smooth turned gold rims!
  const bowShape = new THREE.Shape();
  bowShape.moveTo(0.56, -0.068);
  bowShape.bezierCurveTo(0.61, -0.175, 0.72, -0.245, 0.825, -0.232);
  bowShape.bezierCurveTo(0.935, -0.218, 1.005, -0.125, 1.015, 0.0);
  bowShape.bezierCurveTo(1.005, 0.125, 0.935, 0.218, 0.825, 0.232);
  bowShape.bezierCurveTo(0.72, 0.245, 0.61, 0.175, 0.56, 0.068);
  bowShape.closePath();

  // Large smooth central oval thumb aperture inside the brass bow
  const bowHolePts: THREE.Vector2[] = [];
  const bowHoleSegs = 56;
  for (let s = 0; s < bowHoleSegs; s++) {
    const a = (s / bowHoleSegs) * Math.PI * 2;
    const u = 0.805 + Math.cos(a) * 0.132;
    const v = Math.sin(a) * 0.138;
    bowHolePts.push(new THREE.Vector2(u, v));
  }
  bowShape.holes.push(new THREE.Path(ensureCW(bowHolePts)));

  // Flanking decorative upper & lower Victorian eyelet piercings near the bow shoulder
  bowShape.holes.push(makeCircleHole(0.648, 0.092, 0.028, 24));
  bowShape.holes.push(makeCircleHole(0.648, -0.092, 0.028, 24));

  const bowThickX = 0.038;
  const bowRaw = new THREE.ExtrudeGeometry(bowShape, {
    depth: bowThickX,
    bevelEnabled: true,
    bevelThickness: 0.012,
    bevelSize: 0.011,
    bevelSegments: 4,
    curveSegments: 42,
  });
  // Apply flat-cap machined normals BEFORE rotating Y so the bow front/back faces stay 100% planar
  // while the curved Victorian oval bevels remain liquid-smooth!
  const bowGeo = applyMachinedNormals(bowRaw, (42 * Math.PI) / 180);
  bowGeo.translate(0, 0, -bowThickX * 0.5);
  bowGeo.rotateY(-Math.PI / 2);
  disposables.push(bowGeo);

  const bowMesh = new THREE.Mesh(bowGeo, materials.keyBrass);
  bowMesh.name = 'KeyOrnateBow';
  bowMesh.castShadow = true;
  bowMesh.receiveShadow = true;
  group.add(bowMesh);
  hitMeshes.push(bowMesh);

  // Decorative brass collar band where the ornate bow joins the turned steel stem
  const neckCollarGeo = makeClosedLatheZ(
    [
      [0.525, 0.056],
      [0.552, 0.074],
      [0.580, 0.068],
    ],
    32
  );
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
