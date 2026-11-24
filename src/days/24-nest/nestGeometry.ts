import * as THREE from 'three/webgpu';
import type { PlinthRingSpec } from './reciprocalDome';

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
 * Builds a chamfered, subtly tapered hardwood timber dowel geometry matching
 * Rapier's `ColliderDesc.capsule(halfHeight, radius)` outer envelope, complete
 * with subtle carved joinery notch rings at the analytical reciprocal crossings
 * `notchParams = [sA, sB]` in `[0..1]`.
 *
 * Profile runs strictly bottom-to-top so Three.js `LatheGeometry` produces
 * strictly outward-facing normals (`signedVolume > 0`).
 * Bakes vertex attribute `aNotchMask` (`float`, `0..1`) for dark burnished
 * joinery-ring oil shading in TSL.
 */
export function buildDowelGeometry(
  halfHeight: number,
  radius: number,
  notchParams: readonly [number, number],
  radialSegments = 24
): THREE.BufferGeometry {
  const segLen = halfHeight * 2;
  const totalHalf = halfHeight + radius;
  const points: THREE.Vector2[] = [];
  const notchValues: number[] = [];

  const pushPt = (r: number, y: number, notch: number) => {
    points.push(new THREE.Vector2(Math.max(0.0001, r), y));
    notchValues.push(notch);
  };

  // Evaluate subtle organic hand-turned taper + carved joinery notch indentation
  const shaftRadiusAt = (u: number): { r: number; notch: number } => {
    // Gentle 4% organic butt-to-tip taper (always <= collider radius so zero visual overlap)
    const taper = 1.0 - 0.045 * u;
    let notch = 0;
    const notchWidth = 0.022;
    for (let k = 0; k < 2; k++) {
      const center = notchParams[k];
      const dist = Math.abs(u - center);
      if (dist < notchWidth) {
        const t = 1 - dist / notchWidth;
        const smooth = t * t * (3 - 2 * t);
        if (smooth > notch) notch = smooth;
      }
    }
    // Crisp 7% recessed lathe-carved notch groove at each crossing band
    const r = radius * taper * (1.0 - 0.075 * notch);
    return { r, notch };
  };

  // 1. Bottom pole & chamfered hemispherical end cap (strictly bottom-to-top)
  const capSteps = 7;
  const rButt = shaftRadiusAt(0).r;
  points.push(new THREE.Vector2(0, -totalHalf));
  notchValues.push(0);
  for (let i = 1; i <= capSteps; i++) {
    const frac = i / capSteps;
    const ang = -Math.PI * 0.5 + frac * (Math.PI * 0.5);
    // Crisp wooden bevel shoulder near cap equator
    const bevel = 1.0 - 0.025 * Math.sin(frac * Math.PI);
    const r = Math.cos(ang) * rButt * bevel;
    const y = -halfHeight + Math.sin(ang) * radius;
    pushPt(r, y, 0);
  }

  // 2. Cylindrical tapered shaft with dense sampling around carved joinery notch rings
  const shaftSteps = 44;
  for (let i = 1; i < shaftSteps; i++) {
    const u = i / shaftSteps;
    const y = -halfHeight + u * segLen;
    const { r, notch } = shaftRadiusAt(u);
    pushPt(r, y, notch);
  }

  // 3. Top chamfered hemispherical end cap & top pole
  const rTip = shaftRadiusAt(1).r;
  for (let i = 0; i < capSteps; i++) {
    const frac = i / capSteps;
    const ang = frac * (Math.PI * 0.5);
    const bevel = 1.0 - 0.025 * Math.sin(frac * Math.PI);
    const r = Math.cos(ang) * rTip * bevel;
    const y = halfHeight + Math.sin(ang) * radius;
    pushPt(r, y, 0);
  }
  points.push(new THREE.Vector2(0, totalHalf));
  notchValues.push(0);

  const geom = new THREE.LatheGeometry(points, radialSegments);

  // Assign `aNotchMask` and cleanly normalize longitudinal `uv.y` in [0..1] along rod length
  const pos = geom.getAttribute('position');
  const uvAttr = geom.getAttribute('uv');
  const count = pos.count;
  const notchArr = new Float32Array(count);
  const ringProfileCount = points.length;

  for (let i = 0; i < count; i++) {
    const profileIdx = i % ringProfileCount;
    notchArr[i] = notchValues[profileIdx];
    const y = pos.getY(i);
    const vNorm = Math.max(0, Math.min(1, (y + totalHalf) / (2 * totalHalf)));
    uvAttr.setY(i, vNorm);
  }
  geom.setAttribute('aNotchMask', new THREE.BufferAttribute(notchArr, 1));
  uvAttr.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Builds a closed-solid rotational ring `LatheGeometry` from a CCW 2D `(r, y)` cross-section.
 * Reverses winding automatically only if `signedVolume < 0` so outward normals are guaranteed.
 */
function buildClosedRingLathe(
  profile: Array<[number, number]>,
  segments = 96
): THREE.BufferGeometry {
  const pts = profile.map(([r, y]) => new THREE.Vector2(r, y));
  // Close the profile loop explicitly
  const first = profile[0];
  const last = profile[profile.length - 1];
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
 * Builds the suspended Honed-Limestone circular plinth ring shelf (hollow open center!)
 * whose flat top seating plane sits at exact `y = 0.0` matching the Rapier shelf collider.
 */
export function buildPlinthLimestoneGeometry(
  plinth: PlinthRingSpec,
  segments = 96
): THREE.BufferGeometry {
  const rIn = plinth.innerRadius + 0.008; // Leaves room for flush inner dark-brass bezel
  const rOut = plinth.seatOuterRadius + 0.044;
  const yBot = plinth.baseBottomY;

  // CCW loop in (r, y): outer wall bottom->top, top shelf outer->inner, inner wall top->bottom, bottom inner->outer
  const profile: Array<[number, number]> = [
    [rOut - 0.016, yBot],
    [rOut, yBot + 0.016],
    [rOut, -0.006],
    [rOut - 0.006, 0.0],
    [rIn + 0.006, 0.0],
    [rIn, -0.007],
    [rIn, yBot + 0.022],
    [rIn + 0.014, yBot + 0.008],
    [rIn + 0.032, yBot],
  ];

  return buildClosedRingLathe(profile, segments);
}

/**
 * Builds the Dark-Brass recessed outer retaining lip collar (inner vertical face at
 * exact `plinth.seatOuterRadius`, rising to `plinth.lipTopY = 0.046 m`) that arrests
 * Tier-0's outward hoop thrust with a crisp chamfered museum-grade profile.
 */
export function buildPlinthBrassOuterLipGeometry(
  plinth: PlinthRingSpec,
  segments = 96
): THREE.BufferGeometry {
  const rInnerWall = plinth.seatOuterRadius;
  const rOuterWall = plinth.lipOuterRadius;
  const yTop = plinth.lipTopY;
  const yMid = -0.028;

  const profile: Array<[number, number]> = [
    [rOuterWall - 0.010, yMid - 0.012],
    [rOuterWall, yMid],
    [rOuterWall, yTop - 0.010],
    [rOuterWall - 0.009, yTop],
    [rInnerWall + 0.009, yTop],
    [rInnerWall, yTop - 0.007],
    [rInnerWall, -0.002],
    [rInnerWall + 0.014, -0.004],
    [rOuterWall - 0.024, yMid - 0.012],
  ];

  return buildClosedRingLathe(profile, segments);
}

/**
 * Builds the delicate Dark-Brass inner oculus inlay collar framing the hollow open
 * center of the suspended plinth ring (`r = innerRadius - 0.010 .. innerRadius + 0.012`).
 */
export function buildPlinthBrassInnerBezelGeometry(
  plinth: PlinthRingSpec,
  segments = 96
): THREE.BufferGeometry {
  const rIn = plinth.innerRadius - 0.008;
  const rOut = plinth.innerRadius + 0.014;
  const yBot = plinth.baseBottomY + 0.018;

  const profile: Array<[number, number]> = [
    [rOut, yBot],
    [rOut, -0.002],
    [rOut - 0.004, 0.003],
    [rIn + 0.005, 0.003],
    [rIn, -0.004],
    [rIn, yBot + 0.008],
    [rIn + 0.008, yBot],
  ];

  return buildClosedRingLathe(profile, segments);
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
 * Builds a smooth river-worn Celadon & Bone-Calcite mineral pebble "egg" mesh
 * whose outer envelope stays tightly matched (±3%) to its Rapier sphere collider
 * so visual contact with the timber cradle is flush with zero hovering or clipping.
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
  const [sx, sy, sz] = visualScale;

  // Deterministic organic river-pebble phase offsets
  const p1 = seed * 17.3 + 1.1;
  const p2 = seed * 29.7 + 2.4;
  const p3 = seed * 43.1 + 0.7;

  for (let i = 0; i < pos.count; i++) {
    const nx = pos.getX(i) / radius;
    const ny = pos.getY(i) / radius;
    const nz = pos.getZ(i) / radius;

    // Gentle ovoid taper (slightly fuller base, softly tapered crown) + broad river-worn asymmetry
    const ovoidTaper = 1.0 - 0.025 * ny;
    const wearA = Math.sin(nx * 2.1 + ny * 1.6 + p1) * Math.cos(nz * 1.9 - p2);
    const wearB = Math.cos(ny * 2.3 - nz * 1.7 + p3) * Math.sin(nx * 1.5 + p1);
    const rScale = 1.0 + 0.016 * wearA + 0.010 * wearB;

    pos.setXYZ(
      i,
      nx * radius * sx * ovoidTaper * rScale,
      ny * radius * sy * rScale,
      nz * radius * sz * ovoidTaper * rScale
    );
  }

  pos.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}
