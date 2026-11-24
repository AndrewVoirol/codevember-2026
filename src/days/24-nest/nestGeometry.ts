import * as THREE from 'three/webgpu';
import {
  DOWEL_CHAMFER_FRAC,
  sampleOvoidPebblePoint,
  type PlinthRingSpec,
} from './reciprocalDome';

/**
 * Computes the signed enclosed volume of a closed triangle mesh (`> 0` proves
 * outward CCW triangle winding; `< 0` catches inverted inside-out lathes).
 */
export function computeSignedVolume(geometry: THREE.BufferGeometry): number {
  const pos = geometry.getAttribute('position');
  if (!pos) return 0;
  const idx = geometry.getIndex();
  const triCount = idx ? idx.count / 3 : pos.count / 3;

  let vol6 = 0;
  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;

    const ax = pos.getX(i0);
    const ay = pos.getY(i0);
    const az = pos.getZ(i0);
    const bx = pos.getX(i1);
    const by = pos.getY(i1);
    const bz = pos.getZ(i1);
    const cx = pos.getX(i2);
    const cy = pos.getY(i2);
    const cz = pos.getZ(i2);

    // Scalar triple product a . (b x c)
    const crossX = by * cz - bz * cy;
    const crossY = bz * cx - bx * cz;
    const crossZ = bx * cy - by * cx;
    vol6 += ax * crossX + ay * crossY + az * crossZ;
  }

  return vol6 / 6;
}

/**
 * Builds a precision lathe-turned hardwood dowel geometry with crisp chamfered flat
 * cut end-grain faces matching Rapier's `ColliderDesc.roundCylinder(halfHeight, radius - b, b)`
 * to exact millimeter parity (`0.0 mm` crossing air gap, zero hemispherical pill ends).
 *
 * Profile runs strictly bottom-to-top with edge-holding normal support rings so flat
 * end-grain caps get pure `(0, ±1, 0)` normals, chamfers catch crisp specular highlights,
 * and `LatheGeometry` produces strictly outward CCW winding (`signedVolume > 0`).
 *
 * Bakes vertex attribute `aNotchMask` (`float`, `0..1`) at dense 96-step resolution
 * (`du = 0.0104 < notchWidth / 2`, zero aliasing) for dark burnished joinery saddle wear.
 */
export function buildDowelGeometry(
  halfHeight: number,
  radius: number,
  notchParams: readonly [number, number],
  radialSegments = 24
): THREE.BufferGeometry {
  const segLen = halfHeight * 2;
  const bevelR = radius * DOWEL_CHAMFER_FRAC;
  const rFlat = radius - bevelR;
  const yEnd = halfHeight + bevelR;
  const points: THREE.Vector2[] = [];
  const notchValues: number[] = [];

  const pushPt = (r: number, y: number, notch: number) => {
    points.push(new THREE.Vector2(Math.max(0, r), y));
    notchValues.push(notch);
  };

  // Evaluate burnished joinery contact band along the full-radius cylindrical shaft
  // (keeps outer radius >= 99.65% of collider radius so touching dowels sit flush wood-to-wood!)
  const notchWidth = 0.026;
  const shaftProfileAt = (u: number): { r: number; notch: number } => {
    let notch = 0;
    for (let k = 0; k < 2; k++) {
      const center = notchParams[k];
      const dist = Math.abs(u - center);
      if (dist < notchWidth) {
        const t = 1 - dist / notchWidth;
        const smooth = t * t * (3 - 2 * t);
        if (smooth > notch) notch = smooth;
      }
    }
    const r = radius * (1.0 - 0.0035 * notch);
    return { r, notch };
  };

  const holdInset = Math.min(0.0012, rFlat * 0.08);
  const shaftHoldY = Math.min(0.0022, halfHeight * 0.012);
  const bevelSteps = 3;

  // 1. Bottom flat circular end-grain cap (strictly outward normal (0, -1, 0))
  pushPt(0, -yEnd, 0);
  pushPt(rFlat * 0.48, -yEnd, 0);
  pushPt(rFlat - holdInset, -yEnd, 0);
  pushPt(rFlat, -yEnd, 0);

  // 2. Bottom crisp 3-step lathe chamfer bevel matching roundCylinder quarter-circle arc
  for (let i = 1; i <= bevelSteps; i++) {
    const ang = -Math.PI * 0.5 + (i / bevelSteps) * (Math.PI * 0.5);
    const r = rFlat + Math.cos(ang) * bevelR;
    const y = -halfHeight + Math.sin(ang) * bevelR;
    pushPt(r, y, 0);
  }
  // Edge-holding ring right onto the cylindrical barrel for razor-crisp bevel break
  pushPt(radius, -halfHeight + shaftHoldY, 0);

  // 3. Adaptive full-radius cylindrical shaft: lean on straight stretches, dense across both
  //    joinery notch bands (exact peak & shoulders -> zero notch aliasing at 3x fewer tris!)
  const uMin = shaftHoldY / segLen;
  const uMax = 1.0 - uMin;
  const uSamples: number[] = [];
  for (let i = 1; i < 16; i++) {
    uSamples.push(uMin + (i / 16) * (uMax - uMin));
  }
  const notchOffsets = [-1.0, -0.62, -0.28, 0.0, 0.28, 0.62, 1.0];
  for (let k = 0; k < 2; k++) {
    const c = notchParams[k];
    for (let j = 0; j < notchOffsets.length; j++) {
      const u = c + notchOffsets[j] * notchWidth;
      if (u > uMin + 0.002 && u < uMax - 0.002) {
        uSamples.push(u);
      }
    }
  }
  uSamples.sort((a, b) => a - b);
  let lastU = uMin;
  for (let i = 0; i < uSamples.length; i++) {
    const u = uSamples[i];
    if (u - lastU < 0.002) continue;
    lastU = u;
    const y = -halfHeight + u * segLen;
    const { r, notch } = shaftProfileAt(u);
    pushPt(r, y, notch);
  }

  // Top cylindrical edge-holding ring
  pushPt(radius, halfHeight - shaftHoldY, 0);

  // 4. Top crisp 3-step lathe chamfer bevel
  for (let i = 0; i <= bevelSteps; i++) {
    const ang = (i / bevelSteps) * (Math.PI * 0.5);
    const r = rFlat + Math.cos(ang) * bevelR;
    const y = halfHeight + Math.sin(ang) * bevelR;
    pushPt(r, y, 0);
  }

  // 5. Top flat circular end-grain cap (strictly outward normal (0, +1, 0))
  pushPt(rFlat - holdInset, yEnd, 0);
  pushPt(rFlat * 0.48, yEnd, 0);
  pushPt(0, yEnd, 0);

  const geom = new THREE.LatheGeometry(points, radialSegments);

  // Assign `aNotchMask` and cleanly normalize longitudinal `uv.y` in [0..1] along dowel length
  const pos = geom.getAttribute('position');
  const uvAttr = geom.getAttribute('uv');
  const count = pos.count;
  const notchArr = new Float32Array(count);
  const ringProfileCount = points.length;

  for (let i = 0; i < count; i++) {
    const profileIdx = i % ringProfileCount;
    notchArr[i] = notchValues[profileIdx];
    const y = pos.getY(i);
    const vNorm = Math.max(0, Math.min(1, (y + yEnd) / (2 * yEnd)));
    uvAttr.setY(i, vNorm);
  }
  geom.setAttribute('aNotchMask', new THREE.BufferAttribute(notchArr, 1));
  uvAttr.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Inserts tight edge-holding support points `inset` distance away from every corner of
 * a closed 2D `(r, y)` cross-section so `LatheGeometry.computeVertexNormals()` preserves
 * razor-flat horizontal/vertical architectural faces while keeping crisp chamfer speculars.
 */
function reinforceProfileNormals(
  profile: Array<[number, number]>,
  inset = 0.0012
): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const n = profile.length;
  for (let i = 0; i < n; i++) {
    const prev = profile[(i - 1 + n) % n];
    const curr = profile[i];
    const next = profile[(i + 1) % n];

    const dPrev = Math.hypot(curr[0] - prev[0], curr[1] - prev[1]);
    const dNext = Math.hypot(next[0] - curr[0], next[1] - curr[1]);

    if (dPrev > inset * 3.2) {
      const t = inset / dPrev;
      out.push([curr[0] + (prev[0] - curr[0]) * t, curr[1] + (prev[1] - curr[1]) * t]);
    }
    out.push(curr);
    if (dNext > inset * 3.2) {
      const t = inset / dNext;
      out.push([curr[0] + (next[0] - curr[0]) * t, curr[1] + (next[1] - curr[1]) * t]);
    }
  }
  return out;
}

/**
 * Builds a closed-solid rotational ring `LatheGeometry` from a CCW 2D `(r, y)` cross-section
 * with reinforced flat-face normals and guaranteed outward CCW winding (`signedVolume > 0`).
 */
function buildClosedRingLathe(
  profile: Array<[number, number]>,
  segments = 96,
  normalHoldInset = 0.0012
): THREE.BufferGeometry {
  const crisp = reinforceProfileNormals(profile, normalHoldInset);
  const pts = crisp.map(([r, y]) => new THREE.Vector2(r, y));
  // Close the profile loop explicitly
  const first = crisp[0];
  const last = crisp[crisp.length - 1];
  if (Math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-6) {
    pts.push(new THREE.Vector2(first[0], first[1]));
  }
  const geom = new THREE.LatheGeometry(pts, segments);
  if (computeSignedVolume(geom) < 0) {
    const idx = geom.getIndex();
    if (idx) {
      const arr = idx.array as Uint16Array | Uint32Array;
      for (let i = 0; i < arr.length; i += 3) {
        const tmp = arr[i + 1];
        arr[i + 1] = arr[i + 2];
        arr[i + 2] = tmp;
      }
      idx.needsUpdate = true;
    }
  }
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Builds the suspended Honed-Limestone circular plinth ring (hollow open center!).
 *
 * Non-overlapping architectural rabbet joinery:
 *   - Top flat timber-seating shelf sits at exact `y = 0.0` spanning
 *     `r ∈ [innerRadius + 0.0155, seatOuterRadius - 0.0005]` (zero overlap with inner/outer brass!)
 *   - Stepped lower stone ledge (`y ∈ [-0.096, -0.0445]`) cradles the outer dark-brass collar
 *     cleanly from underneath with a crisp 0.5 mm architectural reveal line.
 */
export function buildPlinthLimestoneGeometry(
  plinth: PlinthRingSpec,
  segments = 96
): THREE.BufferGeometry {
  const rInnerJoint = plinth.innerRadius + 0.0155; // Exact non-overlapping butt against inner brass bezel
  const rOuterJoint = plinth.seatOuterRadius - 0.0005; // Exact non-overlapping butt against outer brass lip
  const rStoneSkirt = plinth.lipOuterRadius - 0.004; // Architectural stone base carrying the brass collar
  const yRabbet = -0.0445;
  const yBot = plinth.baseBottomY;

  // CCW loop in (r, y)
  const profile: Array<[number, number]> = [
    [rStoneSkirt - 0.016, yBot],
    [rStoneSkirt, yBot + 0.014],
    [rStoneSkirt, yRabbet - 0.003],
    [rStoneSkirt - 0.003, yRabbet],
    [rOuterJoint, yRabbet],
    [rOuterJoint, -0.0025],
    [rOuterJoint - 0.0025, 0.0],
    [rInnerJoint + 0.0025, 0.0],
    [rInnerJoint, -0.0025],
    [rInnerJoint, yBot + 0.018],
    [rInnerJoint + 0.010, yBot + 0.006],
    [rInnerJoint + 0.022, yBot],
  ];

  return buildClosedRingLathe(profile, segments, 0.0012);
}

/**
 * Builds the Dark-Brass recessed outer retaining collar seated cleanly inside the
 * limestone outer rabbet (`y >= -0.044`, `r >= seatOuterRadius`) with ZERO overlap
 * against the limestone mesh. Inner vertical wall sits at exact `plinth.seatOuterRadius`,
 * matching the Rapier retaining lip collider to 0.0 mm.
 */
export function buildPlinthBrassOuterLipGeometry(
  plinth: PlinthRingSpec,
  segments = 96
): THREE.BufferGeometry {
  const rInnerWall = plinth.seatOuterRadius;
  const rOuterWall = plinth.lipOuterRadius;
  const yTop = plinth.lipTopY;
  const ySeat = -0.044;

  const profile: Array<[number, number]> = [
    [rOuterWall - 0.004, ySeat],
    [rOuterWall, ySeat + 0.004],
    [rOuterWall, yTop - 0.008],
    [rOuterWall - 0.003, yTop - 0.0025],
    [rOuterWall - 0.008, yTop],
    [rInnerWall + 0.007, yTop],
    [rInnerWall + 0.002, yTop - 0.0025],
    [rInnerWall, yTop - 0.007],
    [rInnerWall, ySeat + 0.002],
    [rInnerWall + 0.002, ySeat],
  ];

  return buildClosedRingLathe(profile, segments, 0.0009);
}

/**
 * Builds the precision Dark-Brass inner oculus bezel framing the hollow open center
 * (`r ∈ [innerRadius, innerRadius + 0.015]`). Seated strictly FLUSH at `y = 0.0`
 * (never proud above `y = 0.0`) so Tier-0 dowels clear it cleanly with zero clipping,
 * and matching the Rapier hollow bore `plinth.innerRadius` to 0.0 mm.
 */
export function buildPlinthBrassInnerBezelGeometry(
  plinth: PlinthRingSpec,
  segments = 96
): THREE.BufferGeometry {
  const rIn = plinth.innerRadius;
  const rOut = plinth.innerRadius + 0.015;
  const yBot = plinth.baseBottomY + 0.016;

  const profile: Array<[number, number]> = [
    [rOut - 0.002, yBot],
    [rOut, yBot + 0.002],
    [rOut, -0.002],
    [rOut - 0.002, 0.0],
    [rIn + 0.0035, 0.0],
    [rIn, -0.0035],
    [rIn, yBot + 0.006],
    [rIn + 0.006, yBot],
  ];

  return buildClosedRingLathe(profile, segments, 0.0008);
}

/**
 * Welds coincident vertices of a non-indexed triangle geometry into an indexed
 * shared-vertex mesh so `computeVertexNormals()` yields silky smooth normals
 * with zero flat-triangle faceting and zero seam creases.
 */
function weldPositionsToIndexed(src: THREE.BufferGeometry): THREE.BufferGeometry {
  const srcPos = src.getAttribute('position');
  const uniquePositions: number[] = [];
  const indices: number[] = [];
  const vertMap = new Map<string, number>();

  for (let i = 0; i < srcPos.count; i++) {
    const x = srcPos.getX(i);
    const y = srcPos.getY(i);
    const z = srcPos.getZ(i);
    const key = `${Math.round(x * 1e5)}_${Math.round(y * 1e5)}_${Math.round(z * 1e5)}`;
    let idx = vertMap.get(key);
    if (idx === undefined) {
      idx = uniquePositions.length / 3;
      uniquePositions.push(x, y, z);
      vertMap.set(key, idx);
    }
    indices.push(idx);
  }

  const welded = new THREE.BufferGeometry();
  welded.setAttribute('position', new THREE.Float32BufferAttribute(uniquePositions, 3));
  welded.setIndex(indices);
  src.dispose();
  return welded;
}

/**
 * Builds a smooth river-worn Celadon & Bone-Calcite ovoid pebble "egg" mesh using
 * the exact shared `sampleOvoidPebblePoint` equation used for Rapier's `convexHull`
 * collider (0.0 mm collider-to-visual parity — zero hovering, zero clipping).
 * Welds icosahedral vertices before smooth deformation so normals are 100% silky
 * smooth with zero flat facets and zero polar pinch.
 */
export function buildPebbleEggGeometry(
  radius: number,
  visualScale: readonly [number, number, number],
  seed: number,
  detail = 4
): THREE.BufferGeometry {
  const rawIco = new THREE.IcosahedronGeometry(radius, detail);
  const geom = weldPositionsToIndexed(rawIco);
  const pos = geom.getAttribute('position');

  for (let i = 0; i < pos.count; i++) {
    const nx = pos.getX(i) / radius;
    const ny = pos.getY(i) / radius;
    const nz = pos.getZ(i) / radius;

    const [px, py, pz] = sampleOvoidPebblePoint(nx, ny, nz, radius, visualScale, seed);
    pos.setXYZ(i, px, py, pz);
  }

  pos.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}
