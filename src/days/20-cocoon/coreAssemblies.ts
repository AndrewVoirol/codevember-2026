import * as THREE from 'three/webgpu';
import {
  type BuiltCoreAssembly,
  type CoreAssemblyDescriptor,
  type CoreMaterialRole,
  type CoreSdfPrimitive,
  type SdfCapsulePrimitive,
  type SdfOrientedBoxPrimitive,
  type SdfSpherePrimitive,
  type SdfTorusPrimitive,
} from './types';

export interface CoreMaterialSet {
  brass: THREE.Material;
  basalt: THREE.Material;
  calcite: THREE.Material;
  verdigris: THREE.Material;
  amber: THREE.Material;
}

// ============================================================================
// Pure Zero-Allocation Flat-Packed Analytical Core SDF + Analytical Gradient
// ============================================================================

const PRIM_STRIDE = 16;

function packPrimitives(primitives: readonly CoreSdfPrimitive[]): {
  packedData: Float32Array;
  packedKinds: Uint8Array;
} {
  const count = primitives.length;
  const packedData = new Float32Array(count * PRIM_STRIDE);
  const packedKinds = new Uint8Array(count);

  for (let i = 0; i < count; i++) {
    const p = primitives[i];
    const o = i * PRIM_STRIDE;
    if (p.kind === 'sphere') {
      packedKinds[i] = 0;
      packedData[o] = p.cx;
      packedData[o + 1] = p.cy;
      packedData[o + 2] = p.cz;
      packedData[o + 3] = p.radius;
    } else if (p.kind === 'capsule') {
      packedKinds[i] = 1;
      const bax = p.bx - p.ax;
      const bay = p.by - p.ay;
      const baz = p.bz - p.az;
      const baLenSq = bax * bax + bay * bay + baz * baz;
      packedData[o] = p.ax;
      packedData[o + 1] = p.ay;
      packedData[o + 2] = p.az;
      packedData[o + 3] = bax;
      packedData[o + 4] = bay;
      packedData[o + 5] = baz;
      packedData[o + 6] = baLenSq > 1e-9 ? 1.0 / baLenSq : 0;
      packedData[o + 7] = p.radiusA;
      packedData[o + 8] = p.radiusB - p.radiusA;
      packedData[o + 9] = Math.sqrt(baLenSq);
    } else if (p.kind === 'torus') {
      packedKinds[i] = 2;
      packedData[o] = p.cx;
      packedData[o + 1] = p.cy;
      packedData[o + 2] = p.cz;
      packedData[o + 3] = p.ux;
      packedData[o + 4] = p.uy;
      packedData[o + 5] = p.uz;
      packedData[o + 6] = p.vx;
      packedData[o + 7] = p.vy;
      packedData[o + 8] = p.vz;
      packedData[o + 9] = p.nx;
      packedData[o + 10] = p.ny;
      packedData[o + 11] = p.nz;
      packedData[o + 12] = p.majorRadius;
      packedData[o + 13] = p.minorRadius;
    } else {
      packedKinds[i] = 3;
      packedData[o] = p.cx;
      packedData[o + 1] = p.cy;
      packedData[o + 2] = p.cz;
      packedData[o + 3] = p.ux;
      packedData[o + 4] = p.uy;
      packedData[o + 5] = p.uz;
      packedData[o + 6] = p.vx;
      packedData[o + 7] = p.vy;
      packedData[o + 8] = p.vz;
      packedData[o + 9] = p.wx;
      packedData[o + 10] = p.wy;
      packedData[o + 11] = p.wz;
      packedData[o + 12] = p.hx;
      packedData[o + 13] = p.hy;
      packedData[o + 14] = p.hz;
      packedData[o + 15] = p.bevel;
    }
  }

  return { packedData, packedKinds };
}

/**
 * Evaluates the exact signed distance from world point (px, py, pz) to the core assembly.
 * Flat contiguous Float32Array loop with zero allocations.
 */
export function evalCoreSdf(
  core: CoreAssemblyDescriptor,
  px: number,
  py: number,
  pz: number
): number {
  const data = core.packedData;
  const kinds = core.packedKinds;
  const count = core.primCount;
  let minDist = 1e9;

  for (let i = 0; i < count; i++) {
    const o = i * PRIM_STRIDE;
    const kind = kinds[i];
    let d: number;

    if (kind === 1) {
      // Tapered capsule (most common primitive)
      const pax = px - data[o];
      const pay = py - data[o + 1];
      const paz = pz - data[o + 2];
      const bax = data[o + 3];
      const bay = data[o + 4];
      const baz = data[o + 5];
      let h = (pax * bax + pay * bay + paz * baz) * data[o + 6];
      if (h < 0) h = 0;
      else if (h > 1) h = 1;
      const dx = pax - bax * h;
      const dy = pay - bay * h;
      const dz = paz - baz * h;
      d = Math.sqrt(dx * dx + dy * dy + dz * dz) - (data[o + 7] + data[o + 8] * h);
    } else if (kind === 2) {
      // Oriented torus
      const rx = px - data[o];
      const ry = py - data[o + 1];
      const rz = pz - data[o + 2];
      const lu = rx * data[o + 3] + ry * data[o + 4] + rz * data[o + 5];
      const lv = rx * data[o + 6] + ry * data[o + 7] + rz * data[o + 8];
      const ln = rx * data[o + 9] + ry * data[o + 10] + rz * data[o + 11];
      const qx = Math.sqrt(lu * lu + lv * lv) - data[o + 12];
      d = Math.sqrt(qx * qx + ln * ln) - data[o + 13];
    } else if (kind === 0) {
      // Sphere
      const dx = px - data[o];
      const dy = py - data[o + 1];
      const dz = pz - data[o + 2];
      d = Math.sqrt(dx * dx + dy * dy + dz * dz) - data[o + 3];
    } else {
      // Oriented rounded box
      const rx = px - data[o];
      const ry = py - data[o + 1];
      const rz = pz - data[o + 2];
      const lx = Math.abs(rx * data[o + 3] + ry * data[o + 4] + rz * data[o + 5]) - data[o + 12];
      const ly = Math.abs(rx * data[o + 6] + ry * data[o + 7] + rz * data[o + 8]) - data[o + 13];
      const lz = Math.abs(rx * data[o + 9] + ry * data[o + 10] + rz * data[o + 11]) - data[o + 14];
      const ox = lx > 0 ? lx : 0;
      const oy = ly > 0 ? ly : 0;
      const oz = lz > 0 ? lz : 0;
      const inside = Math.min(Math.max(lx, Math.max(ly, lz)), 0);
      d = Math.sqrt(ox * ox + oy * oy + oz * oz) + inside - data[o + 15];
    }

    if (d < minDist) minDist = d;
  }

  return minDist;
}

/**
 * Single-pass exact analytical SDF + exact analytical unit outward gradient.
 * Computes the winning primitive and its closed-form surface normal directly into
 * `out[offset..offset+2]` in ONE pass (5x faster than 4-tap finite differences!).
 */
export function evalCoreSdfGrad(
  core: CoreAssemblyDescriptor,
  px: number,
  py: number,
  pz: number,
  out: Float32Array,
  offset: number
): number {
  const data = core.packedData;
  const kinds = core.packedKinds;
  const count = core.primCount;
  let minDist = 1e9;
  let winIdx = -1;

  for (let i = 0; i < count; i++) {
    const o = i * PRIM_STRIDE;
    const kind = kinds[i];
    let d: number;

    if (kind === 1) {
      const pax = px - data[o];
      const pay = py - data[o + 1];
      const paz = pz - data[o + 2];
      const bax = data[o + 3];
      const bay = data[o + 4];
      const baz = data[o + 5];
      let h = (pax * bax + pay * bay + paz * baz) * data[o + 6];
      if (h < 0) h = 0;
      else if (h > 1) h = 1;
      const dx = pax - bax * h;
      const dy = pay - bay * h;
      const dz = paz - baz * h;
      d = Math.sqrt(dx * dx + dy * dy + dz * dz) - (data[o + 7] + data[o + 8] * h);
    } else if (kind === 2) {
      const rx = px - data[o];
      const ry = py - data[o + 1];
      const rz = pz - data[o + 2];
      const lu = rx * data[o + 3] + ry * data[o + 4] + rz * data[o + 5];
      const lv = rx * data[o + 6] + ry * data[o + 7] + rz * data[o + 8];
      const ln = rx * data[o + 9] + ry * data[o + 10] + rz * data[o + 11];
      const qx = Math.sqrt(lu * lu + lv * lv) - data[o + 12];
      d = Math.sqrt(qx * qx + ln * ln) - data[o + 13];
    } else if (kind === 0) {
      const dx = px - data[o];
      const dy = py - data[o + 1];
      const dz = pz - data[o + 2];
      d = Math.sqrt(dx * dx + dy * dy + dz * dz) - data[o + 3];
    } else {
      const rx = px - data[o];
      const ry = py - data[o + 1];
      const rz = pz - data[o + 2];
      const lx = Math.abs(rx * data[o + 3] + ry * data[o + 4] + rz * data[o + 5]) - data[o + 12];
      const ly = Math.abs(rx * data[o + 6] + ry * data[o + 7] + rz * data[o + 8]) - data[o + 13];
      const lz = Math.abs(rx * data[o + 9] + ry * data[o + 10] + rz * data[o + 11]) - data[o + 14];
      const ox = lx > 0 ? lx : 0;
      const oy = ly > 0 ? ly : 0;
      const oz = lz > 0 ? lz : 0;
      const inside = Math.min(Math.max(lx, Math.max(ly, lz)), 0);
      d = Math.sqrt(ox * ox + oy * oy + oz * oz) + inside - data[o + 15];
    }

    if (d < minDist) {
      minDist = d;
      winIdx = i;
    }
  }

  let gx = 0;
  let gy = 1;
  let gz = 0;

  if (winIdx !== -1) {
    const o = winIdx * PRIM_STRIDE;
    const kind = kinds[winIdx];

    if (kind === 1) {
      const pax = px - data[o];
      const pay = py - data[o + 1];
      const paz = pz - data[o + 2];
      const bax = data[o + 3];
      const bay = data[o + 4];
      const baz = data[o + 5];
      let h = (pax * bax + pay * bay + paz * baz) * data[o + 6];
      const interior = h > 0 && h < 1;
      if (h < 0) h = 0;
      else if (h > 1) h = 1;
      const dx = pax - bax * h;
      const dy = pay - bay * h;
      const dz = paz - baz * h;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 1e-8) {
        const inv = 1.0 / len;
        gx = dx * inv;
        gy = dy * inv;
        gz = dz * inv;
        if (interior && data[o + 9] > 1e-6) {
          const slope = data[o + 8] * data[o + 6];
          gx -= bax * slope;
          gy -= bay * slope;
          gz -= baz * slope;
        }
      }
    } else if (kind === 2) {
      const rx = px - data[o];
      const ry = py - data[o + 1];
      const rz = pz - data[o + 2];
      const ux = data[o + 3];
      const uy = data[o + 4];
      const uz = data[o + 5];
      const vx = data[o + 6];
      const vy = data[o + 7];
      const vz = data[o + 8];
      const nx = data[o + 9];
      const ny = data[o + 10];
      const nz = data[o + 11];
      const lu = rx * ux + ry * uy + rz * uz;
      const lv = rx * vx + ry * vy + rz * vz;
      const ln = rx * nx + ry * ny + rz * nz;
      const rPlane = Math.sqrt(lu * lu + lv * lv);
      const qx = rPlane - data[o + 12];
      if (rPlane > 1e-7) {
        const invRP = 1.0 / rPlane;
        const rux = (lu * ux + lv * vx) * invRP;
        const ruy = (lu * uy + lv * vy) * invRP;
        const ruz = (lu * uz + lv * vz) * invRP;
        gx = qx * rux + ln * nx;
        gy = qx * ruy + ln * ny;
        gz = qx * ruz + ln * nz;
      } else {
        gx = nx;
        gy = ny;
        gz = nz;
      }
    } else if (kind === 0) {
      gx = px - data[o];
      gy = py - data[o + 1];
      gz = pz - data[o + 2];
    } else {
      const rx = px - data[o];
      const ry = py - data[o + 1];
      const rz = pz - data[o + 2];
      const ux = data[o + 3];
      const uy = data[o + 4];
      const uz = data[o + 5];
      const vx = data[o + 6];
      const vy = data[o + 7];
      const vz = data[o + 8];
      const wx = data[o + 9];
      const wy = data[o + 10];
      const wz = data[o + 11];
      const sx = rx * ux + ry * uy + rz * uz;
      const sy = rx * vx + ry * vy + rz * vz;
      const sz = rx * wx + ry * wy + rz * wz;
      const lx = Math.abs(sx) - data[o + 12];
      const ly = Math.abs(sy) - data[o + 13];
      const lz = Math.abs(sz) - data[o + 14];
      const ox = lx > 0 ? lx : 0;
      const oy = ly > 0 ? ly : 0;
      const oz = lz > 0 ? lz : 0;
      let glx = 0;
      let gly = 0;
      let glz = 0;
      if (ox > 0 || oy > 0 || oz > 0) {
        glx = sx >= 0 ? ox : -ox;
        gly = sy >= 0 ? oy : -oy;
        glz = sz >= 0 ? oz : -oz;
      } else if (lx >= ly && lx >= lz) {
        glx = sx >= 0 ? 1 : -1;
      } else if (ly >= lz) {
        gly = sy >= 0 ? 1 : -1;
      } else {
        glz = sz >= 0 ? 1 : -1;
      }
      gx = glx * ux + gly * vx + glz * wx;
      gy = glx * uy + gly * vy + glz * wy;
      gz = glx * uz + gly * vz + glz * wz;
    }
  }

  const gLen = Math.sqrt(gx * gx + gy * gy + gz * gz);
  if (gLen > 1e-8) {
    const inv = 1.0 / gLen;
    out[offset] = gx * inv;
    out[offset + 1] = gy * inv;
    out[offset + 2] = gz * inv;
  } else {
    const rLen = Math.sqrt(px * px + py * py + pz * pz);
    if (rLen > 1e-6) {
      out[offset] = px / rLen;
      out[offset + 1] = py / rLen;
      out[offset + 2] = pz / rLen;
    } else {
      out[offset] = 0;
      out[offset + 1] = 1;
      out[offset + 2] = 0;
    }
  }
  return minDist;
}

// ============================================================================
// Guaranteed Outward-Wound Procedural Geometry Builders
// ============================================================================

/**
 * Builds an outward-wound closed polygonal ring-stack solid along the +Y axis.
 * Each ring defines [y, radius]. Bottom and top rings are capped cleanly.
 */
function buildPolygonalStackGeometry(
  rings: ReadonlyArray<readonly [number, number]>,
  radialSegments: number,
  faceted = true
): THREE.BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const ringCount = rings.length;
  const stride = radialSegments + 1;

  for (let j = 0; j < ringCount; j++) {
    const [y, r] = rings[j];
    const vCoord = j / Math.max(1, ringCount - 1);
    for (let i = 0; i <= radialSegments; i++) {
      const phi = (i / radialSegments) * Math.PI * 2;
      positions.push(Math.cos(phi) * r, y, Math.sin(phi) * r);
      uvs.push(i / radialSegments, vCoord);
    }
  }

  // Side quads: (a, d, c) and (a, c, b) produce outward normals for p = (r cos phi, y, r sin phi)
  for (let j = 0; j < ringCount - 1; j++) {
    const row0 = j * stride;
    const row1 = (j + 1) * stride;
    for (let i = 0; i < radialSegments; i++) {
      const a = row0 + i;
      const b = row0 + i + 1;
      const c = row1 + i + 1;
      const d = row1 + i;
      indices.push(a, d, c, a, c, b);
    }
  }

  // Bottom cap (outward normal = -Y)
  const botCenterIdx = positions.length / 3;
  const [yBot, rBot] = rings[0];
  positions.push(0, yBot, 0);
  uvs.push(0.5, 0);
  if (rBot > 1e-5) {
    for (let i = 0; i < radialSegments; i++) {
      indices.push(botCenterIdx, i, i + 1);
    }
  }

  // Top cap (outward normal = +Y)
  const topCenterIdx = positions.length / 3;
  const [yTop, rTop] = rings[ringCount - 1];
  const topRow = (ringCount - 1) * stride;
  positions.push(0, yTop, 0);
  uvs.push(0.5, 1);
  if (rTop > 1e-5) {
    for (let i = 0; i < radialSegments; i++) {
      indices.push(topCenterIdx, topRow + i + 1, topRow + i);
    }
  }

  let geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);

  if (faceted) {
    const nonIdx = geo.toNonIndexed();
    geo.dispose();
    geo = nonIdx;
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a razor-crisp octagonal-chamfered horological brass ring lying in the XY plane
 * (axis along +Z) with strictly outward winding (dP/dphi x dP/dtheta = +N).
 */
function buildChamferedRingGeometry(
  majorRadius: number,
  halfWidthRadial: number,
  halfWidthAxial: number,
  chamferFrac = 0.28,
  majorSegments = 56
): THREE.BufferGeometry {
  const wr = halfWidthRadial;
  const wa = halfWidthAxial;
  const cr = wr * (1 - chamferFrac);
  const ca = wa * (1 - chamferFrac);

  // 8-point CCW chamfered octagonal cross section in (eR, eZ) plane
  const profile: ReadonlyArray<readonly [number, number]> = [
    [wr, -ca],
    [wr, ca],
    [cr, wa],
    [-cr, wa],
    [-wr, ca],
    [-wr, -ca],
    [-cr, -wa],
    [cr, -wa],
  ];

  const profCount = profile.length;
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i <= majorSegments; i++) {
    const u = i / majorSegments;
    const phi = u * Math.PI * 2;
    const c = Math.cos(phi);
    const s = Math.sin(phi);
    for (let k = 0; k <= profCount; k++) {
      const [pr, pz] = profile[k % profCount];
      const rad = majorRadius + pr;
      positions.push(rad * c, rad * s, pz);
      uvs.push(u * 4, k / profCount);
    }
  }

  const stride = profCount + 1;
  for (let i = 0; i < majorSegments; i++) {
    const r0 = i * stride;
    const r1 = (i + 1) * stride;
    for (let k = 0; k < profCount; k++) {
      const p00 = r0 + k;
      const p10 = r1 + k;
      const p11 = r1 + k + 1;
      const p01 = r0 + k + 1;
      indices.push(p00, p10, p11, p00, p11, p01);
    }
  }

  const indexed = new THREE.BufferGeometry();
  indexed.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  indexed.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  indexed.setIndex(indices);
  const faceted = indexed.toNonIndexed();
  indexed.dispose();
  faceted.computeVertexNormals();
  return faceted;
}

/**
 * Builds a chamfered octahedral / double-beveled basalt polyhedral core centered at origin.
 */
function buildChamferedHubGeometry(radius: number, radialSides = 8): THREE.BufferGeometry {
  const rings: Array<readonly [number, number]> = [
    [-radius * 0.96, radius * 0.14],
    [-radius * 0.78, radius * 0.54],
    [-radius * 0.34, radius * 0.94],
    [-radius * 0.14, radius * 1.0],
    [radius * 0.14, radius * 1.0],
    [radius * 0.34, radius * 0.94],
    [radius * 0.78, radius * 0.54],
    [radius * 0.96, radius * 0.14],
  ];
  return buildPolygonalStackGeometry(rings, radialSides, true);
}

/**
 * Builds an elongated 6-sided faceted mineral crystal prism with machined collar seat & sharp pyramidal apex.
 */
function buildCrystalPrismGeometry(
  length: number,
  baseRadius: number,
  shoulderRadius: number,
  sides = 6
): THREE.BufferGeometry {
  const rings: Array<readonly [number, number]> = [
    [0.0, baseRadius * 0.86],
    [length * 0.08, baseRadius],
    [length * 0.72, shoulderRadius],
    [length * 0.88, shoulderRadius * 0.56],
    [length, Math.max(0.008, shoulderRadius * 0.09)],
  ];
  return buildPolygonalStackGeometry(rings, sides, true);
}

/**
 * Builds a stepped horological brass caliper spoke / ferrule shaft from y = 0 to y = length.
 */
function buildCaliperSpokeGeometry(
  length: number,
  rootRadius: number,
  tipRadius: number
): THREE.BufferGeometry {
  const rings: Array<readonly [number, number]> = [
    [0.0, rootRadius * 1.15],
    [length * 0.10, rootRadius * 1.18],
    [length * 0.16, rootRadius * 0.82],
    [length * 0.76, tipRadius * 0.88],
    [length * 0.82, tipRadius * 1.24],
    [length * 0.94, tipRadius * 1.20],
    [length, tipRadius * 0.95],
  ];
  return buildPolygonalStackGeometry(rings, 12, true);
}

/**
 * Builds a chamfered boss / inlaid fastener turret from y = 0 to y = height.
 */
function buildInlayBossGeometry(radius: number, height: number, sides = 8): THREE.BufferGeometry {
  const rings: Array<readonly [number, number]> = [
    [0.0, radius * 1.0],
    [height * 0.65, radius * 0.96],
    [height * 0.88, radius * 0.74],
    [height, radius * 0.48],
  ];
  return buildPolygonalStackGeometry(rings, sides, true);
}

/**
 * Builds a glowing faceted realgar-amber bipyramidal core heart centered at origin.
 */
function buildAmberHeartGeometry(radius: number, height: number): THREE.BufferGeometry {
  const rings: Array<readonly [number, number]> = [
    [-height * 0.5, radius * 0.08],
    [-height * 0.24, radius * 0.78],
    [0.0, radius],
    [height * 0.24, radius * 0.78],
    [height * 0.5, radius * 0.08],
  ];
  return buildPolygonalStackGeometry(rings, 8, true);
}

// ============================================================================
// Helper: Construct Oriented Mesh & Matching Analytical SDF Primitives
// ============================================================================

const _upAxis = new THREE.Vector3(0, 1, 0);
const _tmpQuat = new THREE.Quaternion();
const _tmpMat4 = new THREE.Matrix4();
const _tmpBasisX = new THREE.Vector3();
const _tmpBasisY = new THREE.Vector3();
const _tmpBasisZ = new THREE.Vector3();
const _tmpPosA = new THREE.Vector3();
const _tmpPosB = new THREE.Vector3();

interface AssemblyBuilder {
  group: THREE.Group;
  primitives: CoreSdfPrimitive[];
  geometries: THREE.BufferGeometry[];
  amberHeartMesh: THREE.Mesh | null;
  materials: CoreMaterialSet;
}

function getMat(materials: CoreMaterialSet, role: CoreMaterialRole): THREE.Material {
  return materials[role];
}

function addOrientedRing(
  b: AssemblyBuilder,
  majorRadius: number,
  halfWidthRadial: number,
  halfWidthAxial: number,
  eulerX: number,
  eulerY: number,
  eulerZ: number,
  role: CoreMaterialRole,
  name: string
): void {
  const geo = buildChamferedRingGeometry(majorRadius, halfWidthRadial, halfWidthAxial);
  b.geometries.push(geo);
  const mesh = new THREE.Mesh(geo, getMat(b.materials, role));
  mesh.name = name;
  mesh.rotation.set(eulerX, eulerY, eulerZ);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  b.group.add(mesh);

  mesh.updateMatrix();
  _tmpMat4.copy(mesh.matrix);
  _tmpMat4.extractBasis(_tmpBasisX, _tmpBasisY, _tmpBasisZ);

  // Circumscribed cross-section radius ensures SDF strictly contains every chamfer corner
  const minorBound =
    Math.max(
      Math.hypot(halfWidthRadial, halfWidthAxial * 0.72),
      Math.hypot(halfWidthRadial * 0.72, halfWidthAxial)
    ) + 0.005;
  const prim: SdfTorusPrimitive = {
    kind: 'torus',
    cx: 0,
    cy: 0,
    cz: 0,
    ux: _tmpBasisX.x,
    uy: _tmpBasisX.y,
    uz: _tmpBasisX.z,
    vx: _tmpBasisY.x,
    vy: _tmpBasisY.y,
    vz: _tmpBasisY.z,
    nx: _tmpBasisZ.x,
    ny: _tmpBasisZ.y,
    nz: _tmpBasisZ.z,
    majorRadius,
    minorRadius: minorBound,
  };
  b.primitives.push(prim);
}

function addDirectionalSpokeWithCrystal(
  b: AssemblyBuilder,
  dirIn: readonly [number, number, number],
  startDist: number,
  spokeLen: number,
  crystalLen: number,
  spokeRootR: number,
  spokeTipR: number,
  crystalBaseR: number,
  crystalShoulderR: number,
  crystalRole: CoreMaterialRole,
  name: string,
  crystalSides = 6
): void {
  const dir = new THREE.Vector3(dirIn[0], dirIn[1], dirIn[2]).normalize();
  _tmpQuat.setFromUnitVectors(_upAxis, dir);

  // Brass caliper spoke shaft
  if (spokeLen > 0.01) {
    const spokeGeo = buildCaliperSpokeGeometry(spokeLen, spokeRootR, spokeTipR);
    b.geometries.push(spokeGeo);
    const spokeMesh = new THREE.Mesh(spokeGeo, b.materials.brass);
    spokeMesh.name = `${name}_Spoke`;
    spokeMesh.position.copy(dir).multiplyScalar(startDist);
    spokeMesh.quaternion.copy(_tmpQuat);
    spokeMesh.castShadow = true;
    spokeMesh.receiveShadow = true;
    b.group.add(spokeMesh);

    const rA = spokeRootR * 1.2 + 0.005;
    const rB = spokeTipR * 1.26 + 0.005;
    _tmpPosA.copy(dir).multiplyScalar(startDist + rA * 0.25);
    _tmpPosB.copy(dir).multiplyScalar(startDist + spokeLen - rB * 0.15);
    const prim: SdfCapsulePrimitive = {
      kind: 'capsule',
      ax: _tmpPosA.x,
      ay: _tmpPosA.y,
      az: _tmpPosA.z,
      bx: _tmpPosB.x,
      by: _tmpPosB.y,
      bz: _tmpPosB.z,
      radiusA: rA,
      radiusB: rB,
    };
    b.primitives.push(prim);
  }

  // Faceted mineral crystal prism crowning the spoke
  if (crystalLen > 0.01) {
    const crystalStart = startDist + spokeLen - 0.015;
    const crystalGeo = buildCrystalPrismGeometry(
      crystalLen,
      crystalBaseR,
      crystalShoulderR,
      crystalSides
    );
    b.geometries.push(crystalGeo);
    const crystalMesh = new THREE.Mesh(crystalGeo, getMat(b.materials, crystalRole));
    crystalMesh.name = `${name}_Crystal`;
    crystalMesh.position.copy(dir).multiplyScalar(crystalStart);
    crystalMesh.quaternion.copy(_tmpQuat);
    crystalMesh.castShadow = true;
    crystalMesh.receiveShadow = true;
    b.group.add(crystalMesh);

    // Two tightly fitted tapered capsules (shaft + pyramidal tip) strictly bound the crystal silhouette
    const rBodyA = crystalBaseR + 0.005;
    const rBodyB = crystalShoulderR + 0.005;
    const shoulderDist = crystalStart + crystalLen * 0.72;
    _tmpPosA.copy(dir).multiplyScalar(crystalStart);
    _tmpPosB.copy(dir).multiplyScalar(shoulderDist);
    b.primitives.push({
      kind: 'capsule',
      ax: _tmpPosA.x,
      ay: _tmpPosA.y,
      az: _tmpPosA.z,
      bx: _tmpPosB.x,
      by: _tmpPosB.y,
      bz: _tmpPosB.z,
      radiusA: rBodyA,
      radiusB: rBodyB,
    });

    const rTip = Math.max(0.024, crystalShoulderR * 0.26);
    _tmpPosA.copy(dir).multiplyScalar(shoulderDist);
    _tmpPosB.copy(dir).multiplyScalar(crystalStart + crystalLen - rTip * 0.85);
    b.primitives.push({
      kind: 'capsule',
      ax: _tmpPosA.x,
      ay: _tmpPosA.y,
      az: _tmpPosA.z,
      bx: _tmpPosB.x,
      by: _tmpPosB.y,
      bz: _tmpPosB.z,
      radiusA: rBodyB,
      radiusB: rTip,
    });
  }
}

function addDirectionalBoss(
  b: AssemblyBuilder,
  dirIn: readonly [number, number, number],
  startDist: number,
  radius: number,
  height: number,
  role: CoreMaterialRole,
  name: string
): void {
  const dir = new THREE.Vector3(dirIn[0], dirIn[1], dirIn[2]).normalize();
  _tmpQuat.setFromUnitVectors(_upAxis, dir);

  const geo = buildInlayBossGeometry(radius, height, 8);
  b.geometries.push(geo);
  const mesh = new THREE.Mesh(geo, getMat(b.materials, role));
  mesh.name = name;
  mesh.position.copy(dir).multiplyScalar(startDist);
  mesh.quaternion.copy(_tmpQuat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  b.group.add(mesh);

  _tmpPosA.copy(dir).multiplyScalar(startDist + height * 0.12);
  _tmpPosB.copy(dir).multiplyScalar(startDist + height * 0.68);
  b.primitives.push({
    kind: 'capsule',
    ax: _tmpPosA.x,
    ay: _tmpPosA.y,
    az: _tmpPosA.z,
    bx: _tmpPosB.x,
    by: _tmpPosB.y,
    bz: _tmpPosB.z,
    radiusA: radius + 0.005,
    radiusB: radius * 0.96 + 0.005,
  });

  // Top bevel cap of boss
  _tmpPosA.copy(dir).multiplyScalar(startDist + height * 0.68);
  _tmpPosB.copy(dir).multiplyScalar(startDist + height - Math.min(height * 0.28, radius * 0.45));
  b.primitives.push({
    kind: 'capsule',
    ax: _tmpPosA.x,
    ay: _tmpPosA.y,
    az: _tmpPosA.z,
    bx: _tmpPosB.x,
    by: _tmpPosB.y,
    bz: _tmpPosB.z,
    radiusA: radius * 0.96 + 0.005,
    radiusB: radius * 0.56 + 0.005,
  });
}

function addChamferedHubAndHeart(
  b: AssemblyBuilder,
  hubRadius: number,
  heartRadius: number,
  heartHeight: number,
  hubSides = 8
): void {
  // Chamfered basalt outer shell split into upper & lower caliper hemispheres so the glowing amber cavity gleams through the waist!
  const hubGeo = buildChamferedHubGeometry(hubRadius, hubSides);
  b.geometries.push(hubGeo);
  const hubMesh = new THREE.Mesh(hubGeo, b.materials.basalt);
  hubMesh.name = 'BasaltCoreHub';
  hubMesh.castShadow = true;
  hubMesh.receiveShadow = true;
  b.group.add(hubMesh);

  // Brass equatorial waist collar framing the hub
  addOrientedRing(b, hubRadius * 0.99, 0.036, 0.042, Math.PI * 0.5, 0, 0, 'brass', 'HubCollarRing');

  // Glowing Realgar-Amber crystal heart gleaming inside the equatorial aperture
  const heartGeo = buildAmberHeartGeometry(heartRadius, heartHeight);
  b.geometries.push(heartGeo);
  const heartMesh = new THREE.Mesh(heartGeo, b.materials.amber);
  heartMesh.name = 'AmberCavityHeart';
  b.group.add(heartMesh);
  b.amberHeartMesh = heartMesh;

  const spherePrim: SdfSpherePrimitive = {
    kind: 'sphere',
    cx: 0,
    cy: 0,
    cz: 0,
    radius: Math.max(hubRadius, heartRadius) + 0.006,
  };
  b.primitives.push(spherePrim);
}

// ============================================================================
// The 3 Distinct Sculptural Hidden Inner Cores
// ============================================================================

export const CORE_COUNT = 3;

export function buildCoreAssembly(
  coreIndex: number,
  materials: CoreMaterialSet
): BuiltCoreAssembly {
  const idx = ((coreIndex % CORE_COUNT) + CORE_COUNT) % CORE_COUNT;
  const b: AssemblyBuilder = {
    group: new THREE.Group(),
    primitives: [],
    geometries: [],
    amberHeartMesh: null,
    materials,
  };

  let name = '';
  let boundingRadius = 1.2;

  if (idx === 0) {
    // -------------------------------------------------------------------------
    // Core 0: Astrolabe Caliper & Bone-Calcite Prism Reliquary
    // -------------------------------------------------------------------------
    name = 'Astrolabe Caliper & Calcite Reliquary';
    boundingRadius = 1.18;

    addChamferedHubAndHeart(b, 0.42, 0.47, 0.24, 8);

    // Two dramatic intersecting cantilevered satin-brass meridian rings
    addOrientedRing(b, 0.85, 0.048, 0.036, 0.56, 0.24, 0.18, 'brass', 'AstrolabeRingA');
    addOrientedRing(b, 0.83, 0.045, 0.034, -0.62, 1.38, -0.22, 'brass', 'AstrolabeRingB');

    // Six projecting chamfered brass caliper spokes crowned with faceted Bone-Calcite prisms
    const spokes: ReadonlyArray<{
      dir: readonly [number, number, number];
      spokeLen: number;
      crystalLen: number;
      baseR: number;
      role: CoreMaterialRole;
    }> = [
      { dir: [0.16, 0.98, 0.10], spokeLen: 0.34, crystalLen: 0.44, baseR: 0.115, role: 'calcite' },
      { dir: [-0.14, -0.98, -0.10], spokeLen: 0.32, crystalLen: 0.40, baseR: 0.108, role: 'calcite' },
      { dir: [0.96, 0.18, 0.22], spokeLen: 0.34, crystalLen: 0.38, baseR: 0.105, role: 'calcite' },
      { dir: [-0.96, -0.15, -0.22], spokeLen: 0.34, crystalLen: 0.38, baseR: 0.105, role: 'calcite' },
      { dir: [-0.18, 0.16, 0.97], spokeLen: 0.32, crystalLen: 0.41, baseR: 0.112, role: 'calcite' },
      { dir: [0.18, -0.18, -0.97], spokeLen: 0.30, crystalLen: 0.37, baseR: 0.105, role: 'calcite' },
    ];

    for (let i = 0; i < spokes.length; i++) {
      const s = spokes[i];
      addDirectionalSpokeWithCrystal(
        b,
        s.dir,
        0.38,
        s.spokeLen,
        s.crystalLen,
        0.056,
        0.068,
        s.baseR,
        s.baseR * 0.78,
        s.role,
        `CaliperAxis_${i}`,
        6
      );
    }

    // Four diagonal Verdigris Malachite inlay bosses emerging between the primary axes
    const bosses: ReadonlyArray<readonly [number, number, number]> = [
      [0.64, 0.56, 0.52],
      [-0.62, 0.58, -0.52],
      [-0.58, -0.54, 0.61],
      [0.60, -0.56, -0.57],
    ];
    for (let i = 0; i < bosses.length; i++) {
      addDirectionalBoss(b, bosses[i], 0.39, 0.11, 0.27, 'verdigris', `VerdigrisBoss_${i}`);
    }
  } else if (idx === 1) {
    // -------------------------------------------------------------------------
    // Core 1: Triaxial Mineral Geode Caliper & Prismatic Spire Cluster
    // -------------------------------------------------------------------------
    name = 'Prismatic Calcite & Verdigris Spire Cluster';
    boundingRadius = 1.20;

    addChamferedHubAndHeart(b, 0.46, 0.51, 0.28, 10);

    // Equatorial brass vernier caliper girdle
    addOrientedRing(b, 0.78, 0.052, 0.038, 0.22, 0.0, -0.18, 'brass', 'GeodeVernierBand');

    // Eight asymmetric faceted Bone-Calcite & Verdigris crystal spires radiating from hub
    const spires: ReadonlyArray<{
      dir: readonly [number, number, number];
      spokeLen: number;
      crystalLen: number;
      baseR: number;
      role: CoreMaterialRole;
    }> = [
      { dir: [0.08, 0.99, 0.12], spokeLen: 0.14, crystalLen: 0.64, baseR: 0.145, role: 'calcite' },
      { dir: [-0.28, 0.76, 0.58], spokeLen: 0.12, crystalLen: 0.58, baseR: 0.125, role: 'calcite' },
      { dir: [0.68, 0.52, -0.51], spokeLen: 0.12, crystalLen: 0.56, baseR: 0.125, role: 'verdigris' },
      { dir: [-0.82, 0.24, -0.52], spokeLen: 0.12, crystalLen: 0.55, baseR: 0.120, role: 'calcite' },
      { dir: [0.85, -0.22, 0.47], spokeLen: 0.12, crystalLen: 0.58, baseR: 0.130, role: 'calcite' },
      { dir: [-0.56, -0.62, 0.55], spokeLen: 0.12, crystalLen: 0.52, baseR: 0.122, role: 'verdigris' },
      { dir: [0.24, -0.92, -0.31], spokeLen: 0.14, crystalLen: 0.60, baseR: 0.135, role: 'calcite' },
      { dir: [-0.14, -0.32, -0.94], spokeLen: 0.12, crystalLen: 0.54, baseR: 0.118, role: 'verdigris' },
    ];

    for (let i = 0; i < spires.length; i++) {
      const sp = spires[i];
      addDirectionalSpokeWithCrystal(
        b,
        sp.dir,
        0.42,
        sp.spokeLen,
        sp.crystalLen,
        0.072,
        0.088,
        sp.baseR,
        sp.baseR * 0.74,
        sp.role,
        `Spire_${i}`,
        6
      );
    }
  } else {
    // -------------------------------------------------------------------------
    // Core 2: Helical Brass Orrery Gyro-Armature & Amber Heart
    // -------------------------------------------------------------------------
    name = 'Tri-Axis Brass Gyro-Armature';
    boundingRadius = 1.18;

    // Open suspended glowing amber core heart framed inside three nested chamfered brass rings
    const heartGeo = buildAmberHeartGeometry(0.36, 0.58);
    b.geometries.push(heartGeo);
    const heartMesh = new THREE.Mesh(heartGeo, b.materials.amber);
    heartMesh.name = 'AmberCavityHeart';
    b.group.add(heartMesh);
    b.amberHeartMesh = heartMesh;

    // Oriented rounded box for the faceted amber bipyramid core
    const obox: SdfOrientedBoxPrimitive = {
      kind: 'obox',
      cx: 0,
      cy: 0,
      cz: 0,
      ux: 1,
      uy: 0,
      uz: 0,
      vx: 0,
      vy: 1,
      vz: 0,
      wx: 0,
      wy: 0,
      wz: 1,
      hx: 0.28,
      hy: 0.25,
      hz: 0.28,
      bevel: 0.085,
    };
    b.primitives.push(obox);

    // Three nested tilted chamfered brass & basalt gyro rings
    addOrientedRing(b, 0.96, 0.052, 0.042, 0.34, 0.18, 0.0, 'brass', 'OuterGyroRing');
    addOrientedRing(b, 0.81, 0.046, 0.036, -0.78, 1.18, 0.26, 'basalt', 'MidGyroRing');
    addOrientedRing(b, 0.64, 0.042, 0.034, 1.12, -0.62, 0.44, 'brass', 'InnerGyroRing');

    // Four projecting machined trunnion finials & Bone-Calcite tips jutting outside the outer ring
    const trunnions: ReadonlyArray<readonly [number, number, number]> = [
      [0.0, 0.94, 0.34],
      [0.0, -0.94, -0.34],
      [0.98, 0.06, -0.18],
      [-0.98, -0.06, 0.18],
    ];
    for (let i = 0; i < trunnions.length; i++) {
      addDirectionalSpokeWithCrystal(
        b,
        trunnions[i],
        0.62,
        0.34,
        0.24,
        0.048,
        0.062,
        0.088,
        0.062,
        i % 2 === 0 ? 'calcite' : 'verdigris',
        `Trunnion_${i}`,
        8
      );
    }
  }

  b.group.name = `CocoonInnerCore_${idx}`;

  const { packedData, packedKinds } = packPrimitives(b.primitives);
  const descriptor: CoreAssemblyDescriptor = {
    index: idx,
    name,
    primitives: b.primitives,
    packedData,
    packedKinds,
    primCount: b.primitives.length,
    boundingRadius,
  };

  return {
    descriptor,
    group: b.group,
    amberHeartMesh: b.amberHeartMesh!,
    dispose: () => {
      for (let i = 0; i < b.geometries.length; i++) {
        b.geometries[i].dispose();
      }
    },
  };
}
