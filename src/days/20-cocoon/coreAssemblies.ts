import * as THREE from 'three/webgpu';
import {
  type BuiltCoreAssembly,
  type CoreAssemblyDescriptor,
  type CoreMaterialRole,
  type CoreSdfPrimitive,
  type SdfCapsulePrimitive,
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
  packedBounds: Float32Array;
  packedKinds: Uint8Array;
} {
  const count = primitives.length;
  const packedData = new Float32Array(count * PRIM_STRIDE);
  const packedBounds = new Float32Array(count * 6);
  const packedKinds = new Uint8Array(count);

  for (let i = 0; i < count; i++) {
    const p = primitives[i];
    const o = i * PRIM_STRIDE;
    const b = i * 6;
    if (p.kind === 'sphere') {
      packedKinds[i] = 0;
      packedData[o] = p.cx;
      packedData[o + 1] = p.cy;
      packedData[o + 2] = p.cz;
      packedData[o + 3] = p.radius;
      packedBounds[b] = p.cx - p.radius;
      packedBounds[b + 1] = p.cx + p.radius;
      packedBounds[b + 2] = p.cy - p.radius;
      packedBounds[b + 3] = p.cy + p.radius;
      packedBounds[b + 4] = p.cz - p.radius;
      packedBounds[b + 5] = p.cz + p.radius;
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
      const rMax = Math.max(p.radiusA, p.radiusB);
      packedBounds[b] = Math.min(p.ax, p.bx) - rMax;
      packedBounds[b + 1] = Math.max(p.ax, p.bx) + rMax;
      packedBounds[b + 2] = Math.min(p.ay, p.by) - rMax;
      packedBounds[b + 3] = Math.max(p.ay, p.by) + rMax;
      packedBounds[b + 4] = Math.min(p.az, p.bz) - rMax;
      packedBounds[b + 5] = Math.max(p.az, p.bz) + rMax;
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
      // Exact oriented torus AABB extents: majorRadius * sqrt(1 - n_axis^2) + minorRadius
      const ex = p.majorRadius * Math.sqrt(Math.max(0, 1 - p.nx * p.nx)) + p.minorRadius;
      const ey = p.majorRadius * Math.sqrt(Math.max(0, 1 - p.ny * p.ny)) + p.minorRadius;
      const ez = p.majorRadius * Math.sqrt(Math.max(0, 1 - p.nz * p.nz)) + p.minorRadius;
      packedBounds[b] = p.cx - ex;
      packedBounds[b + 1] = p.cx + ex;
      packedBounds[b + 2] = p.cy - ey;
      packedBounds[b + 3] = p.cy + ey;
      packedBounds[b + 4] = p.cz - ez;
      packedBounds[b + 5] = p.cz + ez;
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
      const ex = Math.abs(p.ux) * p.hx + Math.abs(p.vx) * p.hy + Math.abs(p.wx) * p.hz + p.bevel;
      const ey = Math.abs(p.uy) * p.hx + Math.abs(p.vy) * p.hy + Math.abs(p.wy) * p.hz + p.bevel;
      const ez = Math.abs(p.uz) * p.hx + Math.abs(p.vz) * p.hy + Math.abs(p.wz) * p.hz + p.bevel;
      packedBounds[b] = p.cx - ex;
      packedBounds[b + 1] = p.cx + ex;
      packedBounds[b + 2] = p.cy - ey;
      packedBounds[b + 3] = p.cy + ey;
      packedBounds[b + 4] = p.cz - ez;
      packedBounds[b + 5] = p.cz + ez;
    }
  }

  return { packedData, packedBounds, packedKinds };
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
  const bounds = core.packedBounds;
  const kinds = core.packedKinds;
  const count = core.primCount;
  let minDist = 1e9;

  for (let i = 0; i < count; i++) {
    // Exact AABB lower-bound skip (zero sqrt when point is farther than current minDist!)
    if (minDist > 0) {
      const b = i * 6;
      if (
        px < bounds[b] - minDist ||
        px > bounds[b + 1] + minDist ||
        py < bounds[b + 2] - minDist ||
        py > bounds[b + 3] + minDist ||
        pz < bounds[b + 4] - minDist ||
        pz > bounds[b + 5] + minDist
      ) {
        continue;
      }
    }
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
  const bounds = core.packedBounds;
  const kinds = core.packedKinds;
  const count = core.primCount;
  let minDist = 1e9;
  let winIdx = -1;

  for (let i = 0; i < count; i++) {
    if (minDist > 0) {
      const b = i * 6;
      if (
        px < bounds[b] - minDist ||
        px > bounds[b + 1] + minDist ||
        py < bounds[b + 2] - minDist ||
        py > bounds[b + 3] + minDist ||
        pz < bounds[b + 4] - minDist ||
        pz > bounds[b + 5] + minDist
      ) {
        continue;
      }
    }
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
 * Builds a razor-crisp octagonal-chamfered horological brass/basalt ring lying in the XY plane
 * (axis along +Z) with strictly outward winding (dP/dphi x dP/dtheta = +N).
 */
function buildChamferedRingGeometry(
  majorRadius: number,
  halfWidthRadial: number,
  halfWidthAxial: number,
  chamferFrac = 0.28,
  majorSegments = 64
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
 * Builds one hemisphere cap of the split equatorial basalt hub (`sign = +1` upper, `-1` lower),
 * leaving a clean open gallery around `|y| < gapHalf` for the recessed Realgar-Amber cavity heart.
 */
function buildSplitHubCapGeometry(
  radius: number,
  gapHalf: number,
  sign: 1 | -1,
  radialSides = 8
): THREE.BufferGeometry {
  const rawRings: Array<readonly [number, number]> = [
    [gapHalf, radius * 0.90],
    [gapHalf + radius * 0.14, radius * 0.95],
    [radius * 0.44, radius * 0.88],
    [radius * 0.78, radius * 0.55],
    [radius * 0.96, radius * 0.16],
  ];
  const rings: Array<readonly [number, number]> =
    sign > 0
      ? rawRings
      : rawRings
          .slice()
          .reverse()
          .map(([y, r]) => [-y, r] as const);
  return buildPolygonalStackGeometry(rings, radialSides, true);
}

/**
 * Builds a precision horological brass gimbal axle pin / strut from y = 0 to y = length.
 */
function buildGimbalPinGeometry(length: number, radius: number, sides = 10): THREE.BufferGeometry {
  const rings: Array<readonly [number, number]> = [
    [0.0, radius * 1.32],
    [length * 0.14, radius * 1.32],
    [length * 0.22, radius * 0.94],
    [length * 0.78, radius * 0.94],
    [length * 0.86, radius * 1.32],
    [length, radius * 1.32],
  ];
  return buildPolygonalStackGeometry(rings, sides, true);
}

/**
 * Builds an elongated 6-sided faceted mineral crystal prism with machined collar seat & chamfered apex.
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
    [length, Math.max(0.014, shoulderRadius * 0.14)],
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
    [0.0, rootRadius * 1.18],
    [length * 0.12, rootRadius * 1.22],
    [length * 0.18, rootRadius * 0.82],
    [length * 0.76, tipRadius * 0.88],
    [length * 0.82, tipRadius * 1.26],
    [length * 0.94, tipRadius * 1.20],
    [length, tipRadius * 0.95],
  ];
  return buildPolygonalStackGeometry(rings, 12, true);
}

/**
 * Builds a chamfered boss / inlaid fastener turret with brass seating collar from y = 0 to y = height.
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
    [-height * 0.5, radius * 0.12],
    [-height * 0.25, radius * 0.80],
    [0.0, radius],
    [height * 0.25, radius * 0.80],
    [height * 0.5, radius * 0.12],
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
const _tmpPinDir = new THREE.Vector3();

interface AssemblyBuilder {
  group: THREE.Group;
  primitives: CoreSdfPrimitive[];
  geometries: THREE.BufferGeometry[];
  amberHeartMesh: THREE.Mesh | null;
  materials: CoreMaterialSet;
}

interface BuiltRingBasis {
  majorRadius: number;
  halfWidthRadial: number;
  halfWidthAxial: number;
  u: THREE.Vector3;
  v: THREE.Vector3;
  n: THREE.Vector3;
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
  name: string,
  offsetY = 0
): BuiltRingBasis {
  const geo = buildChamferedRingGeometry(majorRadius, halfWidthRadial, halfWidthAxial);
  b.geometries.push(geo);
  const mesh = new THREE.Mesh(geo, getMat(b.materials, role));
  mesh.name = name;
  mesh.position.set(0, offsetY, 0);
  mesh.rotation.set(eulerX, eulerY, eulerZ);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  b.group.add(mesh);

  mesh.updateMatrix();
  _tmpMat4.copy(mesh.matrix);
  _tmpMat4.extractBasis(_tmpBasisX, _tmpBasisY, _tmpBasisZ);

  // Circumscribed cross-section radius strictly encloses every octagonal chamfer corner
  const minorBound =
    Math.max(
      Math.hypot(halfWidthRadial, halfWidthAxial * 0.72),
      Math.hypot(halfWidthRadial * 0.72, halfWidthAxial)
    ) + 0.008;
  const prim: SdfTorusPrimitive = {
    kind: 'torus',
    cx: 0,
    cy: offsetY,
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

  return {
    majorRadius,
    halfWidthRadial,
    halfWidthAxial,
    u: _tmpBasisX.clone(),
    v: _tmpBasisY.clone(),
    n: _tmpBasisZ.clone(),
  };
}

/**
 * Adds a pair of opposing radial gimbal pivot pins linking two concentric rings along
 * their exact geometric line of intersection (`nOuter x nInner`).
 */
function addRingBridgePins(
  b: AssemblyBuilder,
  outerRing: BuiltRingBasis,
  innerRing: BuiltRingBasis,
  pinRadius: number,
  namePrefix: string
): void {
  const axis = new THREE.Vector3().crossVectors(outerRing.n, innerRing.n);
  if (axis.lengthSq() < 1e-6) {
    axis.copy(outerRing.u);
  } else {
    axis.normalize();
  }

  const rStart = innerRing.majorRadius + innerRing.halfWidthRadial * 0.45;
  const rEnd = outerRing.majorRadius - outerRing.halfWidthRadial * 0.45;
  const pinLen = rEnd - rStart;
  if (pinLen <= 0.01) return;

  for (let side = 0; side < 2; side++) {
    const sign = side === 0 ? 1 : -1;
    _tmpPinDir.copy(axis).multiplyScalar(sign);
    _tmpQuat.setFromUnitVectors(_upAxis, _tmpPinDir);

    const pinGeo = buildGimbalPinGeometry(pinLen, pinRadius, 10);
    b.geometries.push(pinGeo);
    const pinMesh = new THREE.Mesh(pinGeo, b.materials.brass);
    pinMesh.name = `${namePrefix}_${side}`;
    pinMesh.position.copy(_tmpPinDir).multiplyScalar(rStart);
    pinMesh.quaternion.copy(_tmpQuat);
    pinMesh.castShadow = true;
    pinMesh.receiveShadow = true;
    b.group.add(pinMesh);

    _tmpPosA.copy(_tmpPinDir).multiplyScalar(rStart + pinRadius * 0.4);
    _tmpPosB.copy(_tmpPinDir).multiplyScalar(rEnd - pinRadius * 0.4);
    b.primitives.push({
      kind: 'capsule',
      ax: _tmpPosA.x,
      ay: _tmpPosA.y,
      az: _tmpPosA.z,
      bx: _tmpPosB.x,
      by: _tmpPosB.y,
      bz: _tmpPosB.z,
      radiusA: pinRadius * 1.35 + 0.006,
      radiusB: pinRadius * 1.35 + 0.006,
    });
  }
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

    const rA = spokeRootR * 1.24 + 0.007;
    const rB = spokeTipR * 1.28 + 0.007;
    _tmpPosA.copy(dir).multiplyScalar(startDist + rA * 0.15);
    _tmpPosB.copy(dir).multiplyScalar(startDist + spokeLen - rB * 0.1);
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

    // Three tightly fitted tapered capsules (shaft, bevel crown, apex) strictly enclose
    // every single vertex including the 0.88*L chamfer shoulder and sharp apex!
    const rBodyA = crystalBaseR + 0.008;
    const rBodyB = crystalShoulderR + 0.008;
    const rMid = crystalShoulderR * 0.56 + 0.009;
    const rTip = Math.max(0.032, crystalShoulderR * 0.30);

    const shoulderDist = crystalStart + crystalLen * 0.72;
    const bevelDist = crystalStart + crystalLen * 0.88;
    const apexCenterDist = crystalStart + crystalLen - rTip * 0.65;

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

    _tmpPosA.copy(dir).multiplyScalar(shoulderDist);
    _tmpPosB.copy(dir).multiplyScalar(bevelDist);
    b.primitives.push({
      kind: 'capsule',
      ax: _tmpPosA.x,
      ay: _tmpPosA.y,
      az: _tmpPosA.z,
      bx: _tmpPosB.x,
      by: _tmpPosB.y,
      bz: _tmpPosB.z,
      radiusA: rBodyB,
      radiusB: rMid,
    });

    _tmpPosA.copy(dir).multiplyScalar(bevelDist);
    _tmpPosB.copy(dir).multiplyScalar(Math.max(bevelDist + 0.005, apexCenterDist));
    b.primitives.push({
      kind: 'capsule',
      ax: _tmpPosA.x,
      ay: _tmpPosA.y,
      az: _tmpPosA.z,
      bx: _tmpPosB.x,
      by: _tmpPosB.y,
      bz: _tmpPosB.z,
      radiusA: rMid,
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

  // Machined brass mounting bezel seat anchoring the boss cleanly into the basalt hub
  const seatH = height * 0.24;
  const seatGeo = buildInlayBossGeometry(radius * 1.14, seatH, 8);
  b.geometries.push(seatGeo);
  const seatMesh = new THREE.Mesh(seatGeo, b.materials.brass);
  seatMesh.name = `${name}_Collar`;
  seatMesh.position.copy(dir).multiplyScalar(startDist - 0.02);
  seatMesh.quaternion.copy(_tmpQuat);
  seatMesh.castShadow = true;
  seatMesh.receiveShadow = true;
  b.group.add(seatMesh);

  const geo = buildInlayBossGeometry(radius, height, 8);
  b.geometries.push(geo);
  const mesh = new THREE.Mesh(geo, getMat(b.materials, role));
  mesh.name = name;
  mesh.position.copy(dir).multiplyScalar(startDist);
  mesh.quaternion.copy(_tmpQuat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  b.group.add(mesh);

  _tmpPosA.copy(dir).multiplyScalar(startDist + height * 0.08);
  _tmpPosB.copy(dir).multiplyScalar(startDist + height * 0.68);
  b.primitives.push({
    kind: 'capsule',
    ax: _tmpPosA.x,
    ay: _tmpPosA.y,
    az: _tmpPosA.z,
    bx: _tmpPosB.x,
    by: _tmpPosB.y,
    bz: _tmpPosB.z,
    radiusA: radius * 1.14 + 0.008,
    radiusB: radius * 0.98 + 0.008,
  });

  // Top bevel cap of boss
  _tmpPosA.copy(dir).multiplyScalar(startDist + height * 0.68);
  _tmpPosB.copy(dir).multiplyScalar(startDist + height - Math.min(height * 0.24, radius * 0.42));
  b.primitives.push({
    kind: 'capsule',
    ax: _tmpPosA.x,
    ay: _tmpPosA.y,
    az: _tmpPosA.z,
    bx: _tmpPosB.x,
    by: _tmpPosB.y,
    bz: _tmpPosB.z,
    radiusA: radius * 0.98 + 0.008,
    radiusB: radius * 0.58 + 0.008,
  });
}

/**
 * Builds a true split equatorial caliper basalt hub with twin upper/lower machined brass bezel
 * collars, 4 vertical retaining pillars, and a strictly recessed glowing Realgar-Amber crystal
 * heart gleaming through the open gallery (zero hub/collar/heart clipping!).
 */
function addChamferedHubAndHeart(
  b: AssemblyBuilder,
  hubRadius: number,
  gapHalf: number,
  hubSides = 8
): void {
  // Upper & lower split basalt hemispheres
  const upperGeo = buildSplitHubCapGeometry(hubRadius, gapHalf, 1, hubSides);
  b.geometries.push(upperGeo);
  const upperMesh = new THREE.Mesh(upperGeo, b.materials.basalt);
  upperMesh.name = 'BasaltCoreHubUpper';
  upperMesh.castShadow = true;
  upperMesh.receiveShadow = true;
  b.group.add(upperMesh);

  const lowerGeo = buildSplitHubCapGeometry(hubRadius, gapHalf, -1, hubSides);
  b.geometries.push(lowerGeo);
  const lowerMesh = new THREE.Mesh(lowerGeo, b.materials.basalt);
  lowerMesh.name = 'BasaltCoreHubLower';
  lowerMesh.castShadow = true;
  lowerMesh.receiveShadow = true;
  b.group.add(lowerMesh);

  // Twin machined brass bezel collars framing the upper & lower lips of the equatorial gallery
  const bezelR = hubRadius * 0.95;
  addOrientedRing(b, bezelR, 0.032, 0.022, Math.PI * 0.5, 0, 0, 'brass', 'HubBezelUpper', gapHalf);
  addOrientedRing(b, bezelR, 0.032, 0.022, Math.PI * 0.5, 0, 0, 'brass', 'HubBezelLower', -gapHalf);

  // Four vertical brass retaining mullions bridging the equatorial amber gallery
  const mullionR = hubRadius * 0.87;
  const mullionLen = gapHalf * 2.02;
  for (let i = 0; i < 4; i++) {
    const phi = (i * Math.PI) / 2 + Math.PI * 0.25;
    const mx = Math.cos(phi) * mullionR;
    const mz = Math.sin(phi) * mullionR;
    const pGeo = buildGimbalPinGeometry(mullionLen, 0.020, 8);
    b.geometries.push(pGeo);
    const pMesh = new THREE.Mesh(pGeo, b.materials.brass);
    pMesh.name = `HubGalleryPillar_${i}`;
    pMesh.position.set(mx, -mullionLen * 0.5, mz);
    pMesh.castShadow = true;
    pMesh.receiveShadow = true;
    b.group.add(pMesh);
  }

  // Strictly recessed glowing Realgar-Amber crystal heart nested inside the open gallery
  // heartRadius < bezelR - 0.035 guarantees zero poking through brass collars or basalt caps!
  const heartRadius = hubRadius * 0.80;
  const heartHeight = gapHalf * 2.15;
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
    radius: hubRadius + 0.010,
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
  let boundingRadius = 1.24;

  if (idx === 0) {
    // -------------------------------------------------------------------------
    // Core 0: Astrolabe Caliper & Bone-Calcite Prism Reliquary
    // -------------------------------------------------------------------------
    name = 'Astrolabe Caliper & Calcite Reliquary';
    boundingRadius = 1.20;

    addChamferedHubAndHeart(b, 0.43, 0.095, 8);

    // Two cleanly nested concentric cantilevered satin-brass meridian rings (zero overlap!)
    // Ring A: [0.858, 0.942], Ring B: [0.732, 0.808] -> crisp 50 mm radial clearance!
    const ringA = addOrientedRing(b, 0.90, 0.042, 0.034, 0.56, 0.24, 0.18, 'brass', 'AstrolabeRingA');
    const ringB = addOrientedRing(b, 0.77, 0.038, 0.030, -0.62, 1.38, -0.22, 'brass', 'AstrolabeRingB');

    // Machined brass gimbal pivot pins joining Ring A to Ring B along their true intersection axis
    addRingBridgePins(b, ringA, ringB, 0.022, 'AstrolabeBridgePin');

    // Six projecting chamfered brass caliper spokes crowned with faceted Bone-Calcite prisms
    // Carefully aligned to thread cleanly through the open octants of Ring A & Ring B without clipping!
    const spokes: ReadonlyArray<{
      dir: readonly [number, number, number];
      spokeLen: number;
      crystalLen: number;
      baseR: number;
      role: CoreMaterialRole;
    }> = [
      { dir: [0.22, 0.95, 0.22], spokeLen: 0.34, crystalLen: 0.42, baseR: 0.112, role: 'calcite' },
      { dir: [-0.22, -0.95, -0.22], spokeLen: 0.32, crystalLen: 0.39, baseR: 0.106, role: 'calcite' },
      { dir: [0.95, -0.18, 0.25], spokeLen: 0.34, crystalLen: 0.38, baseR: 0.104, role: 'calcite' },
      { dir: [-0.95, 0.18, -0.25], spokeLen: 0.34, crystalLen: 0.38, baseR: 0.104, role: 'calcite' },
      { dir: [-0.42, 0.28, 0.86], spokeLen: 0.32, crystalLen: 0.40, baseR: 0.110, role: 'calcite' },
      { dir: [0.42, -0.28, -0.86], spokeLen: 0.30, crystalLen: 0.37, baseR: 0.104, role: 'calcite' },
    ];

    for (let i = 0; i < spokes.length; i++) {
      const s = spokes[i];
      addDirectionalSpokeWithCrystal(
        b,
        s.dir,
        0.39,
        s.spokeLen,
        s.crystalLen,
        0.052,
        0.064,
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
      addDirectionalBoss(b, bosses[i], 0.38, 0.102, 0.25, 'verdigris', `VerdigrisBoss_${i}`);
    }
  } else if (idx === 1) {
    // -------------------------------------------------------------------------
    // Core 1: Triaxial Mineral Geode Caliper & Prismatic Spire Cluster
    // -------------------------------------------------------------------------
    name = 'Prismatic Calcite & Verdigris Spire Cluster';
    boundingRadius = 1.22;

    addChamferedHubAndHeart(b, 0.46, 0.105, 10);

    // Tilted equatorial brass vernier caliper girdle (tilted off horizontal so equatorial gallery gleams!)
    const girdle = addOrientedRing(b, 0.80, 0.048, 0.036, 0.28, 0.12, -0.24, 'brass', 'GeodeVernierBand');

    // Two opposing brass support spokes anchoring the girdle to the basalt hub along its plane
    for (let side = 0; side < 2; side++) {
      const sign = side === 0 ? 1 : -1;
      const strutDir = girdle.u.clone().multiplyScalar(sign);
      _tmpQuat.setFromUnitVectors(_upAxis, strutDir);
      const strutLen = 0.80 - 0.43;
      const strutGeo = buildGimbalPinGeometry(strutLen, 0.030, 10);
      b.geometries.push(strutGeo);
      const strutMesh = new THREE.Mesh(strutGeo, b.materials.brass);
      strutMesh.name = `GirdleStrut_${side}`;
      strutMesh.position.copy(strutDir).multiplyScalar(0.41);
      strutMesh.quaternion.copy(_tmpQuat);
      strutMesh.castShadow = true;
      strutMesh.receiveShadow = true;
      b.group.add(strutMesh);

      _tmpPosA.copy(strutDir).multiplyScalar(0.42);
      _tmpPosB.copy(strutDir).multiplyScalar(0.79);
      b.primitives.push({
        kind: 'capsule',
        ax: _tmpPosA.x,
        ay: _tmpPosA.y,
        az: _tmpPosA.z,
        bx: _tmpPosB.x,
        by: _tmpPosB.y,
        bz: _tmpPosB.z,
        radiusA: 0.046,
        radiusB: 0.046,
      });
    }

    // Eight asymmetric faceted Bone-Calcite & Verdigris crystal spires radiating cleanly
    // above and below the vernier girdle plane (zero awkward clipping through the brass ring!)
    const spires: ReadonlyArray<{
      dir: readonly [number, number, number];
      spokeLen: number;
      crystalLen: number;
      baseR: number;
      role: CoreMaterialRole;
    }> = [
      // Front-upper crown (aligned cleanly toward +girdle.n hemisphere)
      { dir: [0.10, 0.82, 0.56], spokeLen: 0.14, crystalLen: 0.62, baseR: 0.138, role: 'calcite' },
      { dir: [-0.52, 0.54, 0.66], spokeLen: 0.12, crystalLen: 0.56, baseR: 0.122, role: 'calcite' },
      { dir: [0.64, 0.34, 0.69], spokeLen: 0.12, crystalLen: 0.54, baseR: 0.120, role: 'verdigris' },
      { dir: [-0.20, -0.58, 0.79], spokeLen: 0.12, crystalLen: 0.54, baseR: 0.124, role: 'verdigris' },
      // Rear-lower crown (aligned cleanly toward -girdle.n hemisphere)
      { dir: [-0.14, -0.86, -0.49], spokeLen: 0.14, crystalLen: 0.60, baseR: 0.134, role: 'calcite' },
      { dir: [0.62, 0.46, -0.63], spokeLen: 0.12, crystalLen: 0.56, baseR: 0.122, role: 'verdigris' },
      { dir: [-0.68, 0.28, -0.68], spokeLen: 0.12, crystalLen: 0.55, baseR: 0.120, role: 'calcite' },
      { dir: [0.46, -0.52, -0.72], spokeLen: 0.12, crystalLen: 0.55, baseR: 0.126, role: 'calcite' },
    ];

    for (let i = 0; i < spires.length; i++) {
      const sp = spires[i];
      addDirectionalSpokeWithCrystal(
        b,
        sp.dir,
        0.42,
        sp.spokeLen,
        sp.crystalLen,
        0.068,
        0.084,
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
    boundingRadius = 1.32;

    // Open suspended glowing faceted amber crystal heart cradled by an equatorial brass bezel ring
    const heartR = 0.35;
    const heartH = 0.56;
    const heartGeo = buildAmberHeartGeometry(heartR, heartH);
    b.geometries.push(heartGeo);
    const heartMesh = new THREE.Mesh(heartGeo, b.materials.amber);
    heartMesh.name = 'AmberCavityHeart';
    b.group.add(heartMesh);
    b.amberHeartMesh = heartMesh;

    // Tight exact SDF for the bipyramidal amber crystal (2 tapered vertical capsules — zero corner bloat!)
    b.primitives.push({
      kind: 'capsule',
      ax: 0,
      ay: 0,
      az: 0,
      bx: 0,
      by: heartH * 0.43,
      bz: 0,
      radiusA: heartR + 0.008,
      radiusB: heartR * 0.20 + 0.008,
    });
    b.primitives.push({
      kind: 'capsule',
      ax: 0,
      ay: 0,
      az: 0,
      bx: 0,
      by: -heartH * 0.43,
      bz: 0,
      radiusA: heartR + 0.008,
      radiusB: heartR * 0.20 + 0.008,
    });

    // Three nested tilted chamfered brass & honed-basalt gyro rings with clean radial clearance
    const outerRing = addOrientedRing(b, 0.94, 0.048, 0.038, 0.34, 0.18, 0.0, 'brass', 'OuterGyroRing');
    const midRing = addOrientedRing(b, 0.80, 0.042, 0.034, -0.78, 1.18, 0.26, 'basalt', 'MidGyroRing');
    const innerRing = addOrientedRing(b, 0.65, 0.038, 0.030, 1.12, -0.62, 0.44, 'brass', 'InnerGyroRing');
    // Delicate machined equatorial cradle collar gripping the amber heart
    const heartCollar = addOrientedRing(b, heartR + 0.014, 0.022, 0.020, Math.PI * 0.5, 0, 0, 'brass', 'HeartCradleCollar');

    // Real mechanical gimbal axle pins linking Outer -> Mid -> Inner -> HeartCradle!
    addRingBridgePins(b, outerRing, midRing, 0.022, 'GyroOuterMidPin');
    addRingBridgePins(b, midRing, innerRing, 0.020, 'GyroMidInnerPin');
    addRingBridgePins(b, innerRing, heartCollar, 0.018, 'GyroInnerHeartPin');

    // Four machined finial trunnions mounted cleanly ON THE OUTER RIM of OuterGyroRing
    // along exact in-plane cardinal directions (zero cutting through inner gyro rings!)
    const trunnionDirs: readonly THREE.Vector3[] = [
      outerRing.v.clone(),
      outerRing.v.clone().negate(),
      outerRing.u.clone(),
      outerRing.u.clone().negate(),
    ];
    for (let i = 0; i < trunnionDirs.length; i++) {
      const td = trunnionDirs[i];
      addDirectionalSpokeWithCrystal(
        b,
        [td.x, td.y, td.z],
        outerRing.majorRadius + outerRing.halfWidthRadial * 0.35,
        0.13,
        0.20,
        0.042,
        0.052,
        0.076,
        0.056,
        i % 2 === 0 ? 'calcite' : 'verdigris',
        `Trunnion_${i}`,
        8
      );
    }
  }

  b.group.name = `CocoonInnerCore_${idx}`;

  const { packedData, packedBounds, packedKinds } = packPrimitives(b.primitives);
  const descriptor: CoreAssemblyDescriptor = {
    index: idx,
    name,
    primitives: b.primitives,
    packedData,
    packedBounds,
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
