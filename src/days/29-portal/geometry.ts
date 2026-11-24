import * as THREE from 'three/webgpu';
import {
  APERTURE_PLANE_OFFSET,
  ARCH_HALF_WIDTH,
  ARCH_RADIUS,
  ARCH_SILL_Y,
  ARCH_SPRING_Y,
  CUBE_HALF,
  PORTAL_FACES,
  PORTAL_FACE_ORDER,
  type PortalFaceId,
} from './portalWorlds';

export interface PortalMaterialsContract {
  matteBasalt: THREE.Material;
  basaltAccent: THREE.Material;
  satinBrass: THREE.Material;
  darkBrassFastener: THREE.Material;
  portalMaterials: Record<PortalFaceId, THREE.Material>;
}

export interface PortalCubeAssembly {
  group: THREE.Group;
  portalMeshes: Record<PortalFaceId, THREE.Mesh>;
  portalHitMeshes: THREE.Mesh[];
  exteriorHitMeshes: THREE.Mesh[];
  solidMeshes: THREE.Mesh[];
  dispose: () => void;
}

interface MeshBuilder {
  positions: number[];
  normals: number[];
  uvs: number[];
}

function createBuilder(): MeshBuilder {
  return { positions: [], normals: [], uvs: [] };
}

function pushTri(
  b: MeshBuilder,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  n: THREE.Vector3
): void {
  b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  b.normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z);
  b.uvs.push(0, 0, 1, 0, 0, 1);
}

/**
 * Emits a convex polygon sorted counter-clockwise around `outwardNormal`,
 * ensuring exact outward winding (`inwardTri === 0`) and flat architectural bevel normals.
 */
function pushConvexFace(
  b: MeshBuilder,
  verts: THREE.Vector3[],
  outwardNormal: THREE.Vector3
): void {
  if (verts.length < 3) return;
  const n = outwardNormal.clone().normalize();
  const center = new THREE.Vector3();
  for (let i = 0; i < verts.length; i++) center.add(verts[i]);
  center.divideScalar(verts.length);

  const up = Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const tangent = new THREE.Vector3().crossVectors(up, n).normalize();
  const bitangent = new THREE.Vector3().crossVectors(n, tangent).normalize();

  const sorted = verts
    .map((v) => {
      const rel = v.clone().sub(center);
      return { v, ang: Math.atan2(rel.dot(bitangent), rel.dot(tangent)) };
    })
    .sort((a, bItem) => a.ang - bItem.ang);

  for (let i = 1; i < sorted.length - 1; i++) {
    pushTri(b, sorted[0].v, sorted[i].v, sorted[i + 1].v, n);
  }
}

/**
 * Appends a watertight 26-face chamfered rectangular block centered at `(cx, cy, cz)`
 * directly into `b` (optionally rotated around Y by `rotY`).
 */
function appendChamferedBlock(
  b: MeshBuilder,
  cx: number,
  cy: number,
  cz: number,
  hx: number,
  hy: number,
  hz: number,
  bevel: number,
  rotY = 0
): void {
  const bx = Math.min(bevel, hx * 0.45);
  const by = Math.min(bevel, hy * 0.45);
  const bz = Math.min(bevel, hz * 0.45);
  const ix = hx - bx;
  const iy = hy - by;
  const iz = hz - bz;

  const cosR = Math.cos(rotY);
  const sinR = Math.sin(rotY);

  const xformPt = (x: number, y: number, z: number): THREE.Vector3 => {
    const rx = x * cosR + z * sinR;
    const rz = -x * sinR + z * cosR;
    return new THREE.Vector3(cx + rx, cy + y, cz + rz);
  };
  const xformDir = (x: number, y: number, z: number): THREE.Vector3 => {
    const rx = x * cosR + z * sinR;
    const rz = -x * sinR + z * cosR;
    return new THREE.Vector3(rx, y, rz).normalize();
  };

  const vx = (sx: number, sy: number, sz: number) => xformPt(sx * hx, sy * iy, sz * iz);
  const vy = (sx: number, sy: number, sz: number) => xformPt(sx * ix, sy * hy, sz * iz);
  const vz = (sx: number, sy: number, sz: number) => xformPt(sx * ix, sy * iy, sz * hz);

  for (const s of [-1, 1]) {
    pushConvexFace(b, [vx(s, -1, -1), vx(s, 1, -1), vx(s, 1, 1), vx(s, -1, 1)], xformDir(s, 0, 0));
    pushConvexFace(b, [vy(-1, s, -1), vy(1, s, -1), vy(1, s, 1), vy(-1, s, 1)], xformDir(0, s, 0));
    pushConvexFace(b, [vz(-1, -1, s), vz(1, -1, s), vz(1, 1, s), vz(-1, 1, s)], xformDir(0, 0, s));
  }

  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      pushConvexFace(
        b,
        [vx(s1, s2, -1), vx(s1, s2, 1), vy(s1, s2, 1), vy(s1, s2, -1)],
        xformDir(s1, s2, 0)
      );
      pushConvexFace(
        b,
        [vx(s1, -1, s2), vx(s1, 1, s2), vz(s1, 1, s2), vz(s1, -1, s2)],
        xformDir(s1, 0, s2)
      );
      pushConvexFace(
        b,
        [vy(-1, s1, s2), vy(1, s1, s2), vz(1, s1, s2), vz(-1, s1, s2)],
        xformDir(0, s1, s2)
      );
    }
  }

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        pushConvexFace(b, [vx(sx, sy, sz), vy(sx, sy, sz), vz(sx, sy, sz)], xformDir(sx, sy, sz));
      }
    }
  }
}

/**
 * Appends a watertight convex 2D polygon extruded along a cardinal face's outward normal
 * from `dBack` to `dFront`, with an optional slight chamfer inset on the front face.
 */
function appendFaceExtrudedQuad(
  b: MeshBuilder,
  faceId: PortalFaceId,
  quadUv: ReadonlyArray<readonly [number, number]>,
  dBack: number,
  dFront: number,
  frontBevelInset = 0
): void {
  const f = PORTAL_FACES[faceId];
  const nVec = new THREE.Vector3(f.normal[0], f.normal[1], f.normal[2]);
  const rVec = new THREE.Vector3(f.right[0], f.right[1], f.right[2]);
  const uVec = new THREE.Vector3(f.up[0], f.up[1], f.up[2]);

  const toWorld = (u: number, v: number, d: number): THREE.Vector3 =>
    new THREE.Vector3()
      .addScaledVector(rVec, u)
      .addScaledVector(uVec, v)
      .addScaledVector(nVec, d);

  // Ensure 2D polygon is strictly CCW in (u, v)
  let area2 = 0;
  for (let i = 0; i < quadUv.length; i++) {
    const [x0, y0] = quadUv[i];
    const [x1, y1] = quadUv[(i + 1) % quadUv.length];
    area2 += x0 * y1 - x1 * y0;
  }
  const poly = area2 >= 0 ? [...quadUv] : [...quadUv].reverse();

  const backVerts = poly.map(([u, v]) => toWorld(u, v, dBack));
  const frontShoulderVerts = poly.map(([u, v]) => toWorld(u, v, dFront - frontBevelInset));

  // Centroid for optional front chamfer cap
  let cu = 0;
  let cv = 0;
  for (let i = 0; i < poly.length; i++) {
    cu += poly[i][0];
    cv += poly[i][1];
  }
  cu /= poly.length;
  cv /= poly.length;

  const hasBevel = frontBevelInset > 1e-5;
  const frontCapVerts = hasBevel
    ? poly.map(([u, v]) => {
        const du = u - cu;
        const dv = v - cv;
        const len = Math.max(1e-5, Math.hypot(du, dv));
        const shrink = Math.min(0.35, frontBevelInset / len);
        return toWorld(u - du * shrink, v - dv * shrink, dFront);
      })
    : poly.map(([u, v]) => toWorld(u, v, dFront));

  // Front & back caps
  pushConvexFace(b, frontCapVerts, nVec);
  pushConvexFace(b, backVerts, nVec.clone().negate());

  // Side walls (and 45-deg bevel collar if beveled)
  const count = poly.length;
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    const du = poly[j][0] - poly[i][0];
    const dv = poly[j][1] - poly[i][1];
    // For CCW 2D polygon, outward edge normal is (+dv, -du)
    const outSide = new THREE.Vector3()
      .addScaledVector(rVec, dv)
      .addScaledVector(uVec, -du)
      .normalize();

    const upperEdge = hasBevel ? frontShoulderVerts : frontCapVerts;
    pushConvexFace(b, [backVerts[i], backVerts[j], upperEdge[j], upperEdge[i]], outSide);

    if (hasBevel) {
      const bevelN = outSide.clone().add(nVec).normalize();
      pushConvexFace(
        b,
        [frontShoulderVerts[i], frontShoulderVerts[j], frontCapVerts[j], frontCapVerts[i]],
        bevelN
      );
    }
  }
}

/**
 * Appends a watertight closed beveled cylinder / bolt boss along axis `axisDir`
 * centered at `center`, with radius `radius` and total length `length`.
 */
function appendChamferedCylinder(
  b: MeshBuilder,
  center: THREE.Vector3,
  axisDir: THREE.Vector3,
  radius: number,
  length: number,
  segments = 14,
  bevel = 0.004
): void {
  const n = axisDir.clone().normalize();
  const up = Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const t1 = new THREE.Vector3().crossVectors(up, n).normalize();
  const t2 = new THREE.Vector3().crossVectors(n, t1).normalize();

  const halfL = length * 0.5;
  const bv = Math.min(bevel, radius * 0.35, halfL * 0.4);
  const rCap = radius - bv;

  const ringPts = (zOff: number, r: number): THREE.Vector3[] => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      const ca = Math.cos(a) * r;
      const sa = Math.sin(a) * r;
      pts.push(
        center
          .clone()
          .addScaledVector(n, zOff)
          .addScaledVector(t1, ca)
          .addScaledVector(t2, sa)
      );
    }
    return pts;
  };

  const botCap = ringPts(-halfL, rCap);
  const botShoulder = ringPts(-halfL + bv, radius);
  const topShoulder = ringPts(halfL - bv, radius);
  const topCap = ringPts(halfL, rCap);

  pushConvexFace(b, topCap, n);
  pushConvexFace(b, botCap, n.clone().negate());

  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % segments;
    const midA = ((i + 0.5) / segments) * Math.PI * 2;
    const radial = t1
      .clone()
      .multiplyScalar(Math.cos(midA))
      .addScaledVector(t2, Math.sin(midA))
      .normalize();

    pushConvexFace(
      b,
      [botShoulder[i], botShoulder[j], topShoulder[j], topShoulder[i]],
      radial
    );
    pushConvexFace(
      b,
      [topShoulder[i], topShoulder[j], topCap[j], topCap[i]],
      radial.clone().add(n).normalize()
    );
    pushConvexFace(
      b,
      [botCap[i], botCap[j], botShoulder[j], botShoulder[i]],
      radial.clone().sub(n).normalize()
    );
  }
}

/**
 * Appends a watertight flat-extruded 2D polygon on a horizontal plane (`y = yBase` to `yBase + height`,
 * supporting both top `+Y` and bottom `-Y` inlays).
 */
function appendHorizontalExtrudedPoly(
  b: MeshBuilder,
  ptsXZ: ReadonlyArray<readonly [number, number]>,
  yMin: number,
  yMax: number,
  topBevel = 0
): void {
  // Ensure CCW in (x, -z) so +Y is outward top normal
  let area2 = 0;
  for (let i = 0; i < ptsXZ.length; i++) {
    const [x0, z0] = ptsXZ[i];
    const [x1, z1] = ptsXZ[(i + 1) % ptsXZ.length];
    // (x, -z) signed area
    area2 += x0 * -z1 - x1 * -z0;
  }
  const poly = area2 >= 0 ? [...ptsXZ] : [...ptsXZ].reverse();

  let cx = 0;
  let cz = 0;
  for (let i = 0; i < poly.length; i++) {
    cx += poly[i][0];
    cz += poly[i][1];
  }
  cx /= poly.length;
  cz /= poly.length;

  const hasTopBevel = topBevel > 1e-5;
  const botVerts = poly.map(([x, z]) => new THREE.Vector3(x, yMin, z));
  const shoulderY = hasTopBevel ? yMax - topBevel : yMax;
  const shoulderVerts = poly.map(([x, z]) => new THREE.Vector3(x, shoulderY, z));
  const topVerts = hasTopBevel
    ? poly.map(([x, z]) => {
        const dx = x - cx;
        const dz = z - cz;
        const len = Math.max(1e-5, Math.hypot(dx, dz));
        const shrink = Math.min(0.32, topBevel / len);
        return new THREE.Vector3(x - dx * shrink, yMax, z - dz * shrink);
      })
    : shoulderVerts;

  pushConvexFace(b, topVerts, new THREE.Vector3(0, 1, 0));
  pushConvexFace(b, botVerts, new THREE.Vector3(0, -1, 0));

  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const dx = poly[j][0] - poly[i][0];
    const dz = poly[j][1] - poly[i][1];
    // Edge tangent is (dx, 0, dz). Outward normal = tangent × (0, 1, 0) = (-dz, 0, dx)
    const outN = new THREE.Vector3(-dz, 0, dx).normalize();
    pushConvexFace(b, [botVerts[i], botVerts[j], shoulderVerts[j], shoulderVerts[i]], outN);
    if (hasTopBevel) {
      pushConvexFace(
        b,
        [shoulderVerts[i], shoulderVerts[j], topVerts[j], topVerts[i]],
        outN.clone().add(new THREE.Vector3(0, 1, 0)).normalize()
      );
    }
  }
}

function finalizeGeometry(b: MeshBuilder): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(b.positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(b.normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uvs, 2));
  return geo;
}

/**
 * Generates a smooth angular partition `[0..PI]` containing exact spandrel box corner angles
 * so the continuous basalt spandrel and satin-brass arch molding share identical chord steps
 * with zero radial notches or basalt-vs-brass chord interpenetration.
 */
function buildSpandrelAngles(halfW: number, topH: number): number[] {
  const cornerR = Math.atan2(topH, halfW);
  const cornerL = Math.PI - cornerR;
  const angles: number[] = [];

  const sideSteps = 10;
  const crownSteps = 28;

  for (let i = 0; i <= sideSteps; i++) {
    angles.push((i / sideSteps) * cornerR);
  }
  for (let i = 1; i <= crownSteps; i++) {
    angles.push(cornerR + (i / crownSteps) * (cornerL - cornerR));
  }
  for (let i = 1; i <= sideSteps; i++) {
    angles.push(cornerL + (i / sideSteps) * (Math.PI - cornerL));
  }
  return angles;
}

/**
 * Emits a continuous, watertight, outward-wound semicircular basalt arch spandrel around
 * `innerR` out to the rectangular frame `[±spandrelHalfW, ARCH_SPRING_Y..spandrelTopY]`,
 * with a seamless 45° inner-arch chamfer and ZERO radial joint notches.
 */
function appendContinuousBasaltSpandrel(
  b: MeshBuilder,
  faceId: PortalFaceId,
  angles: readonly number[],
  innerR: number,
  spandrelHalfW: number,
  spandrelTopH: number,
  dBack: number,
  dFront: number,
  innerLipBevel: number
): void {
  const f = PORTAL_FACES[faceId];
  const nVec = new THREE.Vector3(f.normal[0], f.normal[1], f.normal[2]);
  const rVec = new THREE.Vector3(f.right[0], f.right[1], f.right[2]);
  const uVec = new THREE.Vector3(f.up[0], f.up[1], f.up[2]);
  const negN = nVec.clone().negate();

  const toWorld = (u: number, v: number, d: number): THREE.Vector3 =>
    new THREE.Vector3()
      .addScaledVector(rVec, u)
      .addScaledVector(uVec, v)
      .addScaledVector(nVec, d);

  const outerPt = (theta: number): [number, number] => {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const tx = Math.abs(c) > 1e-6 ? spandrelHalfW / Math.abs(c) : 1e6;
    const ty = s > 1e-6 ? spandrelTopH / s : 1e6;
    const t = Math.min(tx, ty);
    return [t * c, ARCH_SPRING_Y + t * s];
  };

  const inBack: THREE.Vector3[] = [];
  const inShoulder: THREE.Vector3[] = [];
  const inFront: THREE.Vector3[] = [];
  const outBack: THREE.Vector3[] = [];
  const outFront: THREE.Vector3[] = [];

  const bv = Math.min(innerLipBevel, (dFront - dBack) * 0.35);
  const frontR = innerR + bv;

  for (let i = 0; i < angles.length; i++) {
    const th = angles[i];
    const c = Math.cos(th);
    const s = Math.sin(th);
    const iu = c * innerR;
    const iv = ARCH_SPRING_Y + s * innerR;
    const fu = c * frontR;
    const fv = ARCH_SPRING_Y + s * frontR;
    const [ou, ov] = outerPt(th);

    inBack.push(toWorld(iu, iv, dBack));
    inShoulder.push(toWorld(iu, iv, dFront - bv));
    inFront.push(toWorld(fu, fv, dFront));
    outBack.push(toWorld(ou, ov, dBack));
    outFront.push(toWorld(ou, ov, dFront));
  }

  for (let i = 0; i < angles.length - 1; i++) {
    const thMid = (angles[i] + angles[i + 1]) * 0.5;
    const radialIn = new THREE.Vector3()
      .addScaledVector(rVec, -Math.cos(thMid))
      .addScaledVector(uVec, -Math.sin(thMid))
      .normalize();

    // Seamless front flat basalt spandrel face
    pushConvexFace(b, [inFront[i], outFront[i], outFront[i + 1], inFront[i + 1]], nVec);
    // Seamless back face
    pushConvexFace(b, [inBack[i], inBack[i + 1], outBack[i + 1], outBack[i]], negN);
    // Continuous 45° stone inner-arch reveal chamfer
    pushConvexFace(
      b,
      [inShoulder[i], inFront[i], inFront[i + 1], inShoulder[i + 1]],
      radialIn.clone().add(nVec).normalize()
    );
    // Continuous inner cylindrical stone arch intrados
    pushConvexFace(b, [inBack[i], inShoulder[i], inShoulder[i + 1], inBack[i + 1]], radialIn);

    // Outer rectangular perimeter wall (east pier edge, top lintel edge, or west pier edge)
    const edgeVec = outBack[i + 1].clone().sub(outBack[i]);
    const outSideN = new THREE.Vector3().crossVectors(edgeVec, nVec).normalize();
    pushConvexFace(b, [outBack[i], outBack[i + 1], outFront[i + 1], outFront[i]], outSideN);
  }

  // Watertight bottom end-caps at theta = 0 (right springline) and theta = PI (left springline)
  const downN = uVec.clone().negate();
  pushConvexFace(b, [inBack[0], inShoulder[0], inFront[0], outFront[0], outBack[0]], downN);
  const last = angles.length - 1;
  pushConvexFace(
    b,
    [inBack[last], outBack[last], outFront[last], inFront[last], inShoulder[last]],
    downN
  );
}

/**
 * Emits a continuous, watertight, outward-wound semicircular satin-brass arch molding
 * with seamless inner & outer 45° rim bevels and ZERO radial joint notches.
 */
function appendContinuousBrassArchMolding(
  b: MeshBuilder,
  faceId: PortalFaceId,
  angles: readonly number[],
  innerR: number,
  outerR: number,
  dBack: number,
  dFront: number,
  rimBevel: number
): void {
  const f = PORTAL_FACES[faceId];
  const nVec = new THREE.Vector3(f.normal[0], f.normal[1], f.normal[2]);
  const rVec = new THREE.Vector3(f.right[0], f.right[1], f.right[2]);
  const uVec = new THREE.Vector3(f.up[0], f.up[1], f.up[2]);
  const negN = nVec.clone().negate();

  const toWorld = (u: number, v: number, d: number): THREE.Vector3 =>
    new THREE.Vector3()
      .addScaledVector(rVec, u)
      .addScaledVector(uVec, v)
      .addScaledVector(nVec, d);

  const bv = Math.min(rimBevel, (outerR - innerR) * 0.28, (dFront - dBack) * 0.35);

  const inBack: THREE.Vector3[] = [];
  const inSh: THREE.Vector3[] = [];
  const inCap: THREE.Vector3[] = [];
  const outCap: THREE.Vector3[] = [];
  const outSh: THREE.Vector3[] = [];
  const outBack: THREE.Vector3[] = [];

  for (let i = 0; i < angles.length; i++) {
    const th = angles[i];
    const c = Math.cos(th);
    const s = Math.sin(th);
    inBack.push(toWorld(c * innerR, ARCH_SPRING_Y + s * innerR, dBack));
    inSh.push(toWorld(c * innerR, ARCH_SPRING_Y + s * innerR, dFront - bv));
    inCap.push(toWorld(c * (innerR + bv), ARCH_SPRING_Y + s * (innerR + bv), dFront));
    outCap.push(toWorld(c * (outerR - bv), ARCH_SPRING_Y + s * (outerR - bv), dFront));
    outSh.push(toWorld(c * outerR, ARCH_SPRING_Y + s * outerR, dFront - bv));
    outBack.push(toWorld(c * outerR, ARCH_SPRING_Y + s * outerR, dBack));
  }

  for (let i = 0; i < angles.length - 1; i++) {
    const thMid = (angles[i] + angles[i + 1]) * 0.5;
    const radOut = new THREE.Vector3()
      .addScaledVector(rVec, Math.cos(thMid))
      .addScaledVector(uVec, Math.sin(thMid))
      .normalize();
    const radIn = radOut.clone().negate();

    // Front brushed-brass plateau
    pushConvexFace(b, [inCap[i], outCap[i], outCap[i + 1], inCap[i + 1]], nVec);
    // Inner 45° beveled rim
    pushConvexFace(
      b,
      [inSh[i], inCap[i], inCap[i + 1], inSh[i + 1]],
      radIn.clone().add(nVec).normalize()
    );
    // Outer 45° beveled rim
    pushConvexFace(
      b,
      [outCap[i], outSh[i], outSh[i + 1], outCap[i + 1]],
      radOut.clone().add(nVec).normalize()
    );
    // Inner aperture bore wall
    pushConvexFace(b, [inBack[i], inSh[i], inSh[i + 1], inBack[i + 1]], radIn);
    // Outer cylindrical seating wall
    pushConvexFace(b, [outSh[i], outBack[i], outBack[i + 1], outSh[i + 1]], radOut);
    // Back face
    pushConvexFace(b, [outBack[i], inBack[i], inBack[i + 1], outBack[i + 1]], negN);
  }

  // Watertight flat end-caps at theta = 0 and theta = PI meeting vertical brass jambs
  const downN = uVec.clone().negate();
  pushConvexFace(
    b,
    [inBack[0], inSh[0], inCap[0], outCap[0], outSh[0], outBack[0]],
    downN
  );
  const last = angles.length - 1;
  pushConvexFace(
    b,
    [inBack[last], outBack[last], outSh[last], outCap[last], inCap[last], inSh[last]],
    downN
  );
}

/**
 * Emits a continuous 360° watertight horizontal brass azimuth ring (top or bottom cap inlay)
 * with smooth circular rim chamfers and ZERO radial segment notches.
 */
function appendContinuousHorizontalRing(
  b: MeshBuilder,
  rIn: number,
  rOut: number,
  yMin: number,
  yMax: number,
  bevel: number,
  bevelOnTop: boolean,
  segments = 48
): void {
  const bv = Math.min(bevel, (rOut - rIn) * 0.28, (yMax - yMin) * 0.42);
  const upN = new THREE.Vector3(0, 1, 0);
  const downN = new THREE.Vector3(0, -1, 0);

  const pt = (ang: number, r: number, y: number): THREE.Vector3 =>
    new THREE.Vector3(Math.sin(ang) * r, y, Math.cos(ang) * r);

  const yCap = bevelOnTop ? yMax : yMin;
  const ySh = bevelOnTop ? yMax - bv : yMin + bv;
  const yFlat = bevelOnTop ? yMin : yMax;
  const capN = bevelOnTop ? upN : downN;
  const flatN = bevelOnTop ? downN : upN;

  const inFlat: THREE.Vector3[] = [];
  const inSh: THREE.Vector3[] = [];
  const inCap: THREE.Vector3[] = [];
  const outCap: THREE.Vector3[] = [];
  const outSh: THREE.Vector3[] = [];
  const outFlat: THREE.Vector3[] = [];

  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    inFlat.push(pt(a, rIn, yFlat));
    inSh.push(pt(a, rIn, ySh));
    inCap.push(pt(a, rIn + bv, yCap));
    outCap.push(pt(a, rOut - bv, yCap));
    outSh.push(pt(a, rOut, ySh));
    outFlat.push(pt(a, rOut, yFlat));
  }

  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % segments;
    const midA = ((i + 0.5) / segments) * Math.PI * 2;
    const radOut = new THREE.Vector3(Math.sin(midA), 0, Math.cos(midA)).normalize();
    const radIn = radOut.clone().negate();

    pushConvexFace(b, [inCap[i], outCap[i], outCap[j], inCap[j]], capN);
    pushConvexFace(b, [inFlat[i], inFlat[j], outFlat[j], outFlat[i]], flatN);
    pushConvexFace(
      b,
      [inSh[i], inCap[i], inCap[j], inSh[j]],
      radIn.clone().add(capN).normalize()
    );
    pushConvexFace(
      b,
      [outCap[i], outSh[i], outSh[j], outCap[j]],
      radOut.clone().add(capN).normalize()
    );
    pushConvexFace(b, [inFlat[i], inSh[i], inSh[j], inFlat[j]], radIn);
    pushConvexFace(b, [outSh[i], outFlat[i], outFlat[j], outSh[j]], radOut);
  }
}

/**
 * Creates a standalone watertight chamfered basalt block geometry (exported for unit tests).
 */
export function createChamferedBasaltBlockGeometry(
  hx: number,
  hy: number,
  hz: number,
  bevel: number
): THREE.BufferGeometry {
  const b = createBuilder();
  appendChamferedBlock(b, 0, 0, 0, hx, hy, hz, bevel);
  return finalizeGeometry(b);
}

/**
 * Builds the recessed arched portal aperture mesh for a cardinal face directly in cube-local space.
 * Strictly outward-wound (`FrontSide`) with outer rim tucked slightly inside the opaque brass frame.
 */
export function createPortalApertureGeometry(faceId: PortalFaceId): THREE.BufferGeometry {
  const f = PORTAL_FACES[faceId];
  const nVec = new THREE.Vector3(f.normal[0], f.normal[1], f.normal[2]);
  const rVec = new THREE.Vector3(f.right[0], f.right[1], f.right[2]);
  const uVec = new THREE.Vector3(f.up[0], f.up[1], f.up[2]);

  const toWorld = (u: number, v: number): THREE.Vector3 =>
    new THREE.Vector3()
      .addScaledVector(nVec, APERTURE_PLANE_OFFSET)
      .addScaledVector(rVec, u)
      .addScaledVector(uVec, v);

  // Terminates squarely inside the solid recessed satin-brass casing [0.415..0.438]
  // so zero hairline gaps occur and portal edge never peeks into the 0.438..0.439 brass/basalt rabbet
  const hw = ARCH_HALF_WIDTH - 0.004; // 0.431
  const r = ARCH_RADIUS - 0.004; // 0.431
  const sillY = ARCH_SILL_Y + 0.004;

  // Outline strictly CCW in (u, v): bottom-left -> bottom-right -> right spring -> semicircular arc -> left spring
  const perimeter: THREE.Vector3[] = [];
  perimeter.push(toWorld(-hw, sillY));
  perimeter.push(toWorld(hw, sillY));

  const arcSegments = 48;
  for (let i = 0; i <= arcSegments; i++) {
    const theta = (i / arcSegments) * Math.PI;
    const u = Math.cos(theta) * r;
    const v = ARCH_SPRING_Y + Math.sin(theta) * r;
    perimeter.push(toWorld(u, v));
  }

  const center = toWorld(0, (sillY + ARCH_SPRING_Y + r) * 0.45);
  const b = createBuilder();

  for (let i = 0; i < perimeter.length; i++) {
    const p1 = perimeter[i];
    const p2 = perimeter[(i + 1) % perimeter.length];
    pushTri(b, center, p1, p2, nVec);
  }

  return finalizeGeometry(b);
}

/**
 * Constructs the full suspended Matte-Basalt & Machined-Brass Four-Portal Architectural Cube:
 * - 4 stacked ashlar corner basalt piers with crisp beveled courses
 * - Stepped chamfered basalt cornice cap & heavy plinth base
 * - 4 recessed arched stone masonry spandrels (24 beveled voussoir blocks per face) & sills
 * - 4 recessed chamfered satin-brass arch window casings with crown keystone clasps
 * - 8 corner satin-brass joinery L-brackets with countersunk dark-brass rivets
 * - Solid cap & base inlaid satin-brass compass-rose azimuth rings & cardinal/45° ticks
 * - 4 recessed arched portal aperture meshes facing +Z, +X, -Z, -X
 */
export function buildPortalCubeAssembly(
  materials: PortalMaterialsContract
): PortalCubeAssembly {
  const group = new THREE.Group();
  group.name = 'NonEuclideanPortalCube';

  const disposables: THREE.BufferGeometry[] = [];
  const solidMeshes: THREE.Mesh[] = [];
  const exteriorHitMeshes: THREE.Mesh[] = [];

  const registerSolidMesh = (
    name: string,
    builder: MeshBuilder,
    material: THREE.Material,
    castShadow = true,
    receiveShadow = true
  ): THREE.Mesh => {
    const geo = finalizeGeometry(builder);
    disposables.push(geo);
    const mesh = new THREE.Mesh(geo, material);
    mesh.name = name;
    mesh.castShadow = castShadow;
    mesh.receiveShadow = receiveShadow;
    group.add(mesh);
    solidMeshes.push(mesh);
    exteriorHitMeshes.push(mesh);
    return mesh;
  };

  // ---------------------------------------------------------------------------
  // 1. Matte Basalt Primary Masonry (Corner Ashlar Piers + Top Cap + Bottom Plinth)
  // ---------------------------------------------------------------------------
  const basaltMain = createBuilder();

  // Top cornice cap: lower lintel collar + upper chamfered crown slab
  appendChamferedBlock(basaltMain, 0, 0.582, 0, 0.744, 0.062, 0.744, 0.014);
  appendChamferedBlock(basaltMain, 0, 0.701, 0, CUBE_HALF, 0.059, CUBE_HALF, 0.022);

  // Bottom plinth base: upper sill collar + heavy chamfered lower base slab
  appendChamferedBlock(basaltMain, 0, -0.576, 0, 0.744, 0.066, 0.744, 0.014);
  appendChamferedBlock(basaltMain, 0, -0.701, 0, CUBE_HALF, 0.059, CUBE_HALF, 0.022);

  // 4 Heavy Corner Ashlar Piers (4 stacked dressed basalt courses per corner)
  // Start strictly at basaltArchInnerR (0.439) so corner piers never overlap the 0.438 satin-brass jambs
  const basaltArchInnerR = ARCH_RADIUS + 0.004; // 0.439
  const pierCenter = (basaltArchInnerR + CUBE_HALF) * 0.5; // 0.5995
  const pierHalf = (CUBE_HALF - basaltArchInnerR) * 0.5; // 0.1605
  const pierYMin = ARCH_SILL_Y - 0.004;
  const pierYMax = 0.526;
  const courseCount = 4;
  const courseH = (pierYMax - pierYMin) / courseCount;

  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      for (let c = 0; c < courseCount; c++) {
        const cy = pierYMin + (c + 0.5) * courseH;
        appendChamferedBlock(
          basaltMain,
          sx * pierCenter,
          cy,
          sz * pierCenter,
          pierHalf,
          courseH * 0.5,
          pierHalf,
          0.011
        );
      }
    }
  }

  // Watertight inner backing core deep behind the four recessed apertures (0.40 < 0.645)
  appendChamferedBlock(basaltMain, 0, 0, 0, 0.39, 0.46, 0.39, 0.02);

  // 4 Continuous Monolithic Arched Basalt Spandrels (48-segment seamless arch lip) + proud stone sills
  // Inner basalt radius (ARCH_RADIUS + 0.004 = 0.439) clears the satin-brass outer molding (0.438) cleanly -> zero z-fighting!
  const spandrelHalfW = ARCH_HALF_WIDTH + 0.036; // 0.471
  const spandrelTopY = 0.552;
  const spandrelTopH = spandrelTopY - ARCH_SPRING_Y; // 0.467
  const archAngles = buildSpandrelAngles(spandrelHalfW, spandrelTopH);

  for (const faceId of PORTAL_FACE_ORDER) {
    appendContinuousBasaltSpandrel(
      basaltMain,
      faceId,
      archAngles,
      basaltArchInnerR,
      spandrelHalfW,
      spandrelTopH,
      APERTURE_PLANE_OFFSET - 0.008,
      CUBE_HALF - 0.005,
      0.012
    );

    // Sculpted proud outer stone archivolt hood-molding crowning the basalt arch reveal
    // Gives each portal doorway a classical architectural frame line with zero radial notches
    appendContinuousBrassArchMolding(
      basaltMain,
      faceId,
      archAngles,
      basaltArchInnerR + 0.014,
      basaltArchInnerR + 0.052,
      CUBE_HALF - 0.008,
      CUBE_HALF + 0.009,
      0.0055
    );

    // Proud chamfered basalt windowsill ledge beneath each arched opening + lower corbel apron
    appendFaceExtrudedQuad(
      basaltMain,
      faceId,
      [
        [-spandrelHalfW - 0.024, ARCH_SILL_Y - 0.036],
        [spandrelHalfW + 0.024, ARCH_SILL_Y - 0.036],
        [spandrelHalfW + 0.024, ARCH_SILL_Y - 0.001],
        [-spandrelHalfW - 0.024, ARCH_SILL_Y - 0.001],
      ],
      APERTURE_PLANE_OFFSET - 0.008,
      CUBE_HALF + 0.012,
      0.008
    );

    // Matching architectural stone springline impost blocks left & right where arch meets piers
    for (const side of [-1, 1]) {
      const impIn = side * (basaltArchInnerR + 0.014);
      const impOut = side * (basaltArchInnerR + 0.072);
      const u0 = Math.min(impIn, impOut);
      const u1 = Math.max(impIn, impOut);
      appendFaceExtrudedQuad(
        basaltMain,
        faceId,
        [
          [u0, ARCH_SPRING_Y - 0.024],
          [u1, ARCH_SPRING_Y - 0.024],
          [u1, ARCH_SPRING_Y + 0.008],
          [u0, ARCH_SPRING_Y + 0.008],
        ],
        CUBE_HALF - 0.01,
        CUBE_HALF + 0.012,
        0.005
      );
    }
  }

  registerSolidMesh('BasaltArchitecturalCube', basaltMain, materials.matteBasalt, true, true);

  // ---------------------------------------------------------------------------
  // 2. Honed Dark Basalt Accent Reveals (Cornice Undercut, Mortar Grooves & Window Jamb Liners)
  // ---------------------------------------------------------------------------
  const basaltAccent = createBuilder();

  // Crisp cornice & plinth horizontal shadow-reveal collars
  appendChamferedBlock(basaltAccent, 0, 0.642, 0, 0.735, 0.014, 0.735, 0.005);
  appendChamferedBlock(basaltAccent, 0, -0.642, 0, 0.735, 0.014, 0.735, 0.005);

  // Recessed honed dark-basalt circular compass-bed plates on top & bottom caps (frames inlaid brass rose cleanly)
  for (const signY of [1, -1] as const) {
    appendChamferedCylinder(
      basaltAccent,
      new THREE.Vector3(0, signY * (CUBE_HALF - 0.0015), 0),
      new THREE.Vector3(0, signY, 0),
      0.648,
      0.006,
      48,
      0.002
    );
  }

  // Recessed chamfered stone jamb cheek sleeves strictly outside the brass jambs (|u| >= 0.439)
  // plus dark stone rabbet backing rings strictly behind the aperture plane (d in [0.630, 0.640])
  for (const faceId of PORTAL_FACE_ORDER) {
    for (const side of [-1, 1]) {
      const uInner = side * basaltArchInnerR; // 0.439 (matches spandrel inner radius, clears 0.438 brass!)
      const uOuter = side * (ARCH_HALF_WIDTH + 0.035); // 0.470
      const u0 = Math.min(uInner, uOuter);
      const u1 = Math.max(uInner, uOuter);
      appendFaceExtrudedQuad(
        basaltAccent,
        faceId,
        [
          [u0, ARCH_SILL_Y],
          [u1, ARCH_SILL_Y],
          [u1, ARCH_SPRING_Y + 0.001],
          [u0, ARCH_SPRING_Y + 0.001],
        ],
        APERTURE_PLANE_OFFSET - 0.008,
        CUBE_HALF - 0.016,
        0.007
      );
    }

    // Dark stone backing rabbet sealing the 1 mm architectural shadow reveal behind brass & spandrel
    appendContinuousBrassArchMolding(
      basaltAccent,
      faceId,
      archAngles,
      ARCH_RADIUS - 0.015,
      ARCH_RADIUS + 0.018,
      APERTURE_PLANE_OFFSET - 0.016,
      APERTURE_PLANE_OFFSET - 0.005,
      0.0
    );
  }

  registerSolidMesh('BasaltRecessedReveals', basaltAccent, materials.basaltAccent, true, true);

  // ---------------------------------------------------------------------------
  // 3. Machined Satin-Brass Hardware: Seamless Arched Window Casings, Keystones & Brackets
  // ---------------------------------------------------------------------------
  const brassFrames = createBuilder();

  const brassInnerR = ARCH_RADIUS - 0.02; // 0.415
  const brassOuterR = ARCH_RADIUS + 0.003; // 0.438 (strictly inside basaltArchInnerR = 0.439!)
  const brassD0 = APERTURE_PLANE_OFFSET - 0.003; // 0.642
  const brassD1 = APERTURE_PLANE_OFFSET + 0.058; // 0.703
  const brassRimBevel = 0.0055;

  for (const faceId of PORTAL_FACE_ORDER) {
    const f = PORTAL_FACES[faceId];
    const nVec = new THREE.Vector3(f.normal[0], f.normal[1], f.normal[2]);
    const rVec = new THREE.Vector3(f.right[0], f.right[1], f.right[2]);
    const uVec = new THREE.Vector3(f.up[0], f.up[1], f.up[2]);
    const negN = nVec.clone().negate();
    const toWorld = (u: number, v: number, d: number): THREE.Vector3 =>
      new THREE.Vector3()
        .addScaledVector(rVec, u)
        .addScaledVector(uVec, v)
        .addScaledVector(nVec, d);

    // Seamless 48-segment semicircular satin-brass arch molding (zero radial notches or z-fighting!)
    appendContinuousBrassArchMolding(
      brassFrames,
      faceId,
      archAngles,
      brassInnerR,
      brassOuterR,
      brassD0,
      brassD1,
      brassRimBevel
    );

    // Subtle stepped inner fine brass glazing bead right at the threshold aperture plane
    appendContinuousBrassArchMolding(
      brassFrames,
      faceId,
      archAngles,
      brassInnerR - 0.008,
      brassInnerR + 0.004,
      brassD0,
      brassD0 + 0.022,
      0.0028
    );

    // Left & right vertical recessed satin-brass jamb casings meeting the arch springline seamlessly
    // (beveled along vertical front edges only so the top joint mates flush into the arch ring!)
    for (const side of [-1, 1]) {
      const uMin = side > 0 ? brassInnerR : -brassOuterR;
      const uMax = side > 0 ? brassOuterR : -brassInnerR;
      const v0 = ARCH_SILL_Y + 0.019;
      const v1 = ARCH_SPRING_Y;
      const bv = brassRimBevel;

      const bL0 = toWorld(uMin, v0, brassD0);
      const bR0 = toWorld(uMax, v0, brassD0);
      const bL1 = toWorld(uMin, v1, brassD0);
      const bR1 = toWorld(uMax, v1, brassD0);

      const sL0 = toWorld(uMin, v0, brassD1 - bv);
      const sR0 = toWorld(uMax, v0, brassD1 - bv);
      const sL1 = toWorld(uMin, v1, brassD1 - bv);
      const sR1 = toWorld(uMax, v1, brassD1 - bv);

      const cL0 = toWorld(uMin + bv, v0, brassD1);
      const cR0 = toWorld(uMax - bv, v0, brassD1);
      const cL1 = toWorld(uMin + bv, v1, brassD1);
      const cR1 = toWorld(uMax - bv, v1, brassD1);

      const leftN = rVec.clone().negate();
      const rightN = rVec.clone();
      const upN = uVec.clone();
      const downN = uVec.clone().negate();

      // Front flat face, back face, left/right 45° vertical chamfers, left/right side walls, top/bottom caps
      pushConvexFace(brassFrames, [cL0, cR0, cR1, cL1], nVec);
      pushConvexFace(brassFrames, [bL0, bL1, bR1, bR0], negN);
      pushConvexFace(brassFrames, [sL0, cL0, cL1, sL1], leftN.clone().add(nVec).normalize());
      pushConvexFace(brassFrames, [cR0, sR0, sR1, cR1], rightN.clone().add(nVec).normalize());
      pushConvexFace(brassFrames, [bL0, sL0, sL1, bL1], leftN);
      pushConvexFace(brassFrames, [sR0, bR0, bR1, sR1], rightN);
      pushConvexFace(brassFrames, [bL1, sL1, cL1, cR1, sR1, bR1], upN);
      pushConvexFace(brassFrames, [bL0, bR0, sR0, cR0, cL0, sL0], downN);

      // Subtle brass springline astragal collar marking transition from vertical jamb to semicircular arch
      const astIn = side > 0 ? brassInnerR - 0.005 : -brassOuterR - 0.001;
      const astOut = side > 0 ? brassOuterR + 0.001 : -(brassInnerR - 0.005);
      appendFaceExtrudedQuad(
        brassFrames,
        faceId,
        [
          [astIn, ARCH_SPRING_Y - 0.011],
          [astOut, ARCH_SPRING_Y - 0.011],
          [astOut, ARCH_SPRING_Y + 0.009],
          [astIn, ARCH_SPRING_Y + 0.009],
        ],
        brassD0 + 0.004,
        brassD1 + 0.006,
        0.003
      );
    }

    // Horizontal satin-brass threshold trim bar along the bottom sill (1 mm clear above plinth collar)
    appendFaceExtrudedQuad(
      brassFrames,
      faceId,
      [
        [-brassOuterR, ARCH_SILL_Y + 0.001],
        [brassOuterR, ARCH_SILL_Y + 0.001],
        [brassOuterR, ARCH_SILL_Y + 0.019],
        [-brassOuterR, ARCH_SILL_Y + 0.019],
      ],
      brassD0,
      brassD1 + 0.006,
      0.005
    );

    // Sculpted two-tier satin-brass crown keystone agrafe seated cleanly into the basalt archivolt
    appendFaceExtrudedQuad(
      brassFrames,
      faceId,
      [
        [-0.038, ARCH_SPRING_Y + basaltArchInnerR + 0.002],
        [0.038, ARCH_SPRING_Y + basaltArchInnerR + 0.002],
        [0.056, ARCH_SPRING_Y + basaltArchInnerR + 0.074],
        [-0.056, ARCH_SPRING_Y + basaltArchInnerR + 0.074],
      ],
      brassD1 + 0.006,
      CUBE_HALF + 0.013,
      0.0055
    );
    // Crisp stepped inner keystone cap boss catching overhead gallery key light
    appendFaceExtrudedQuad(
      brassFrames,
      faceId,
      [
        [-0.022, ARCH_SPRING_Y + basaltArchInnerR + 0.014],
        [0.022, ARCH_SPRING_Y + basaltArchInnerR + 0.014],
        [0.034, ARCH_SPRING_Y + basaltArchInnerR + 0.064],
        [-0.034, ARCH_SPRING_Y + basaltArchInnerR + 0.064],
      ],
      CUBE_HALF + 0.011,
      CUBE_HALF + 0.020,
      0.0035
    );
  }

  // 8 Machined Satin-Brass Corner Joinery Angle-Brackets (4 top corners + 4 bottom corners)
  const brassBrackets = createBuilder();
  const brassFasteners = createBuilder();

  const bracketArmLen = 0.21;
  const bracketBandH = 0.056;
  const bracketThick = 0.016;
  const bracketFaceOut = CUBE_HALF + bracketThick * 0.5 - 0.002;
  const cornerYPositions = [0.696, -0.696];

  for (const cy of cornerYPositions) {
    const sy = cy > 0 ? 1 : -1;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        // Arm wrapping along X on ±Z face
        appendChamferedBlock(
          brassBrackets,
          sx * (CUBE_HALF - bracketArmLen * 0.5 + 0.006),
          cy,
          sz * bracketFaceOut,
          bracketArmLen * 0.5,
          bracketBandH * 0.5,
          bracketThick * 0.5,
          0.004
        );
        // Arm wrapping along Z on ±X face
        appendChamferedBlock(
          brassBrackets,
          sx * bracketFaceOut,
          cy,
          sz * (CUBE_HALF - bracketArmLen * 0.5 + 0.006),
          bracketThick * 0.5,
          bracketBandH * 0.5,
          bracketArmLen * 0.5,
          0.004
        );
        // Vertical corner return clasping the stone corner pier right below/above cornice joint
        appendChamferedBlock(
          brassBrackets,
          sx * (CUBE_HALF - 0.022),
          cy - sy * 0.105,
          sz * bracketFaceOut,
          0.032,
          0.088,
          bracketThick * 0.5,
          0.004
        );
        appendChamferedBlock(
          brassBrackets,
          sx * bracketFaceOut,
          cy - sy * 0.105,
          sz * (CUBE_HALF - 0.022),
          bracketThick * 0.5,
          0.088,
          0.032,
          0.004
        );

        // Countersunk dark-brass rivets on each joinery bracket arm
        const rivetOut = CUBE_HALF + bracketThick - 0.001;
        appendChamferedCylinder(
          brassFasteners,
          new THREE.Vector3(sx * (CUBE_HALF - 0.145), cy, sz * rivetOut),
          new THREE.Vector3(0, 0, sz),
          0.012,
          0.008,
          10,
          0.002
        );
        appendChamferedCylinder(
          brassFasteners,
          new THREE.Vector3(sx * rivetOut, cy, sz * (CUBE_HALF - 0.145)),
          new THREE.Vector3(sx, 0, 0),
          0.012,
          0.008,
          10,
          0.002
        );
        appendChamferedCylinder(
          brassFasteners,
          new THREE.Vector3(sx * (CUBE_HALF - 0.022), cy - sy * 0.148, sz * rivetOut),
          new THREE.Vector3(0, 0, sz),
          0.011,
          0.008,
          10,
          0.002
        );
        appendChamferedCylinder(
          brassFasteners,
          new THREE.Vector3(sx * rivetOut, cy - sy * 0.148, sz * (CUBE_HALF - 0.022)),
          new THREE.Vector3(sx, 0, 0),
          0.011,
          0.008,
          10,
          0.002
        );
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 4. Solid Cap & Base Inlaid Satin-Brass Compass-Rose Cardinal Ticks
  // ---------------------------------------------------------------------------
  const brassCompass = createBuilder();

  const buildCompassInlayOnCap = (signY: 1 | -1) => {
    const yBase = signY > 0 ? CUBE_HALF - 0.002 : -CUBE_HALF - 0.006;
    const yTop = signY > 0 ? CUBE_HALF + 0.006 : -CUBE_HALF + 0.002;
    const bevel = 0.0024;
    const bevelOnTop = signY > 0;

    // Continuous 48-segment satin-brass azimuth rings + precision outer bezel ring framing dark basalt bed
    appendContinuousHorizontalRing(brassCompass, 0.636, 0.664, yBase, yTop, bevel, bevelOnTop, 48);
    appendContinuousHorizontalRing(brassCompass, 0.455, 0.482, yBase, yTop, bevel, bevelOnTop, 48);
    appendContinuousHorizontalRing(brassCompass, 0.185, 0.205, yBase, yTop, bevel, bevelOnTop, 48);

    // 4 Cardinal compass-rose pointers (stepped 3 mm proud of azimuth ring -> zero coplanar overlap!)
    const ptrMin = signY > 0 ? yBase : yBase - 0.003;
    const ptrMax = signY > 0 ? yTop + 0.003 : yTop;
    for (let k = 0; k < 4; k++) {
      const ang = (k * Math.PI) / 2;
      const sx = Math.sin(ang);
      const sz = Math.cos(ang);
      const tx = Math.cos(ang);
      const tz = -Math.sin(ang);
      const rBase = 0.212;
      const rMid = 0.34;
      const rTip = k === 0 ? 0.635 : 0.605; // North pointer slightly longer for orientation clarity
      const halfW = k === 0 ? 0.044 : 0.036;

      appendHorizontalExtrudedPoly(
        brassCompass,
        [
          [sx * rBase, sz * rBase],
          [sx * rMid + tx * halfW, sz * rMid + tz * halfW],
          [sx * rTip, sz * rTip],
          [sx * rMid - tx * halfW, sz * rMid - tz * halfW],
        ],
        ptrMin,
        ptrMax,
        bevelOnTop ? bevel : 0
      );
    }

    // 4 Intercardinal 45° split-corner ticks (strictly outside 0.482 outer ring -> zero z-fighting)
    for (let k = 0; k < 4; k++) {
      const ang = ((k * 90 + 45) * Math.PI) / 180;
      const sx = Math.sin(ang);
      const sz = Math.cos(ang);
      const tx = Math.cos(ang);
      const tz = -Math.sin(ang);
      const r0 = 0.486;
      const r1 = 0.605;
      const hw = 0.016;
      appendHorizontalExtrudedPoly(
        brassCompass,
        [
          [sx * r0 - tx * hw, sz * r0 - tz * hw],
          [sx * r0 + tx * hw, sz * r0 + tz * hw],
          [sx * r1 + tx * (hw * 0.45), sz * r1 + tz * (hw * 0.45)],
          [sx * r1 - tx * (hw * 0.45), sz * r1 - tz * (hw * 0.45)],
        ],
        yBase,
        yTop,
        bevelOnTop ? bevel * 0.8 : 0
      );
    }

    // 8 Fine 22.5° vernier ticks around the outer azimuth ring (strictly outside 0.482)
    for (let k = 0; k < 16; k++) {
      if (k % 2 === 0) continue; // skip 0/45/90/... already marked
      const ang = (k * Math.PI) / 8;
      const sx = Math.sin(ang);
      const sz = Math.cos(ang);
      const tx = Math.cos(ang);
      const tz = -Math.sin(ang);
      const r0 = 0.486;
      const r1 = 0.54;
      const hw = 0.008;
      appendHorizontalExtrudedPoly(
        brassCompass,
        [
          [sx * r0 - tx * hw, sz * r0 - tz * hw],
          [sx * r0 + tx * hw, sz * r0 + tz * hw],
          [sx * r1 + tx * hw, sz * r1 + tz * hw],
          [sx * r1 - tx * hw, sz * r1 - tz * hw],
        ],
        yBase,
        yTop,
        bevelOnTop ? bevel * 0.6 : 0
      );
    }

    // Central turned brass compass-rose medallion + dark-brass center pin
    appendChamferedCylinder(
      brassCompass,
      new THREE.Vector3(0, signY * (CUBE_HALF + 0.003), 0),
      new THREE.Vector3(0, signY, 0),
      0.082,
      0.011,
      20,
      0.003
    );
    appendChamferedCylinder(
      brassFasteners,
      new THREE.Vector3(0, signY * (CUBE_HALF + 0.007), 0),
      new THREE.Vector3(0, signY, 0),
      0.032,
      0.009,
      14,
      0.002
    );
  };

  buildCompassInlayOnCap(1);
  buildCompassInlayOnCap(-1);

  registerSolidMesh('BrassArchWindowFrames', brassFrames, materials.satinBrass, true, true);
  registerSolidMesh('BrassCornerBrackets', brassBrackets, materials.satinBrass, true, true);
  registerSolidMesh('BrassCompassRoseInlay', brassCompass, materials.satinBrass, true, true);
  registerSolidMesh('DarkBrassFasteners', brassFasteners, materials.darkBrassFastener, false, true);

  // ---------------------------------------------------------------------------
  // 5. Four Recessed Arched Non-Euclidean Portal Aperture Meshes (+Z, +X, -Z, -X)
  // ---------------------------------------------------------------------------
  const portalMeshes = {} as Record<PortalFaceId, THREE.Mesh>;
  const portalHitMeshes: THREE.Mesh[] = [];

  for (const faceId of PORTAL_FACE_ORDER) {
    const geo = createPortalApertureGeometry(faceId);
    disposables.push(geo);
    const mesh = new THREE.Mesh(geo, materials.portalMaterials[faceId]);
    mesh.name = `PortalAperture_${faceId}`;
    mesh.userData.portalFace = faceId;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    group.add(mesh);
    portalMeshes[faceId] = mesh;
    portalHitMeshes.push(mesh);
  }

  return {
    group,
    portalMeshes,
    portalHitMeshes,
    exteriorHitMeshes,
    solidMeshes,
    dispose: () => {
      for (let i = 0; i < disposables.length; i++) {
        disposables[i].dispose();
      }
    },
  };
}
