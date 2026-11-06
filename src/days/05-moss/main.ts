import * as THREE from 'three/webgpu';
import {
  attribute,
  cameraPosition,
  clamp,
  cos,
  dot,
  exp,
  float,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  sin,
  smoothstep,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import { CHALKY_SLATE_LINEAR } from '../../core/palette';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  hash31Tsl,
  kubelkaMunkGlaze,
  mineralToothFbm,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

// ---------------------------------------------------------------------------
// Module-Scope Scratch Objects (Zero Per-Frame GC Allocations)
// ---------------------------------------------------------------------------
const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _dir = new THREE.Vector3();
const _localPointer = new THREE.Vector3();
const _localVel = new THREE.Vector3();
const _invGroupMatrix = new THREE.Matrix4();
const _invGroupQuat = new THREE.Quaternion();
const _raycaster = new THREE.Raycaster();
const _rayHitPos = new THREE.Vector3();
const _ndcScratch = new THREE.Vector2();

// ---------------------------------------------------------------------------
// Deterministic PRNG & 3D CPU Value Noise
// ---------------------------------------------------------------------------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash3cpu(ix: number, iy: number, iz: number): number {
  let n = ix * 15731 + iy * 789221 + iz * 1376312589;
  n = (n << 13) ^ n;
  return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
}

function noise3cpu(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = hash3cpu(ix, iy, iz);
  const n100 = hash3cpu(ix + 1, iy, iz);
  const n010 = hash3cpu(ix, iy + 1, iz);
  const n110 = hash3cpu(ix + 1, iy + 1, iz);
  const n001 = hash3cpu(ix, iy, iz + 1);
  const n101 = hash3cpu(ix + 1, iy, iz + 1);
  const n011 = hash3cpu(ix, iy + 1, iz + 1);
  const n111 = hash3cpu(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;
  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return (nxy0 + (nxy1 - nxy0) * uz) * 0.5 + 0.5;
}

function fbm3cpu(x: number, y: number, z: number): number {
  const v0 = noise3cpu(x, y, z);
  const c = 0.80901699;
  const s = 0.58778525;
  const x1 = (x * c - z * s) * 2.03;
  const y1 = (y * c + x * s) * 2.03;
  const z1 = (z * c + y * s) * 2.03;
  const v1 = noise3cpu(x1, y1, z1);
  const x2 = (y1 * c - x1 * s) * 2.07;
  const y2 = (z1 * c + y1 * s) * 2.07;
  const z2 = (x1 * c + z1 * s) * 2.07;
  const v2 = noise3cpu(x2, y2, z2);
  return v0 * 0.57 + v1 * 0.29 + v2 * 0.14;
}

/**
 * Evaluates the multi-lobed bryophyte hummock pillow profile `[0..1]` at local position `(wx, wy, wz)`.
 */
function evalHummockProfileCpu(wx: number, wy: number, wz: number, veinSeed: number): number {
  const macro = fbm3cpu(wx * 9.5 + veinSeed, wy * 9.5, wz * 9.5 - veinSeed);
  const meso = fbm3cpu(wx * 21.0 - veinSeed, wy * 21.0 + 1.7, wz * 21.0 + veinSeed);
  // Billowing hummock domes with deep velvety cleavage folds between cushions
  const billow = Math.sin(Math.min(1.0, Math.max(0.0, macro * 1.15 - 0.08)) * Math.PI * 0.5);
  return billow * 0.72 + meso * 0.28;
}

// ---------------------------------------------------------------------------
// 7 Stepped Polished Marble & Dark Stone Pillars (Architectural Cluster)
// ---------------------------------------------------------------------------
export interface MarblePillarSpec {
  id: number;
  cx: number;
  cz: number;
  radius: number;
  topY: number;
  bottomY: number;
  basinDepth: number;
  bevelWidth: number;
  /** Natural slight tilt of the top basin floor so excess liquid gravitates toward lower neighbors */
  slopeX: number;
  slopeZ: number;
  /** 1.0 = creamy Calacatta/Carrara veined marble; 0.16 = dark Nero Marquina/Verde Alpi stone */
  marbleLightness: number;
  veinSeed: number;
  /** Primary downstream pillar ID and landing coordinate when this basin overflows */
  overflowTargetId: number;
  overflowRimAngle: number;
  overflowLandingX: number;
  overflowLandingZ: number;
}

const PILLARS: readonly MarblePillarSpec[] = [
  // Pillar 0: Tall upper-back-left Calacatta Marble Crown Pillar -> cascades into Pillar 2
  {
    id: 0,
    cx: -0.26,
    cz: -0.24,
    radius: 0.44,
    topY: 0.64,
    bottomY: -0.92,
    basinDepth: 0.028,
    bevelWidth: 0.045,
    slopeX: 0.026,
    slopeZ: 0.034,
    marbleLightness: 0.96,
    veinSeed: 1.1,
    overflowTargetId: 2,
    overflowRimAngle: 1.08,
    overflowLandingX: -0.06,
    overflowLandingZ: 0.14,
  },
  // Pillar 1: Upper-right broad Carrara Marble Terrace Pillar -> cascades into Pillar 2
  {
    id: 1,
    cx: 0.34,
    cz: -0.06,
    radius: 0.48,
    topY: 0.36,
    bottomY: -0.98,
    basinDepth: 0.030,
    bevelWidth: 0.048,
    slopeX: -0.032,
    slopeZ: 0.032,
    marbleLightness: 0.91,
    veinSeed: 2.7,
    overflowTargetId: 2,
    overflowRimAngle: 2.42,
    overflowLandingX: 0.10,
    overflowLandingZ: 0.22,
  },
  // Pillar 2: Central Heart Polished Dark Nero Marquina / Serpentinite Rock Pillar -> cascades into Pillar 6
  {
    id: 2,
    cx: -0.04,
    cz: 0.24,
    radius: 0.50,
    topY: 0.08,
    bottomY: -1.06,
    basinDepth: 0.032,
    bevelWidth: 0.050,
    slopeX: 0.018,
    slopeZ: 0.036,
    marbleLightness: 0.16,
    veinSeed: 4.3,
    overflowTargetId: 6,
    overflowRimAngle: 1.18,
    overflowLandingX: 0.10,
    overflowLandingZ: 0.54,
  },
  // Pillar 3: Mid-left Polished White Calacatta Gold Marble Pillar -> cascades into Pillar 4
  {
    id: 3,
    cx: -0.58,
    cz: 0.14,
    radius: 0.42,
    topY: -0.04,
    bottomY: -1.00,
    basinDepth: 0.026,
    bevelWidth: 0.042,
    slopeX: 0.024,
    slopeZ: 0.036,
    marbleLightness: 0.93,
    veinSeed: 5.9,
    overflowTargetId: 4,
    overflowRimAngle: 1.24,
    overflowLandingX: -0.44,
    overflowLandingZ: 0.48,
  },
  // Pillar 4: Lower-left Polished Dark Veined Rock Pillar -> cascades into Pillar 6
  {
    id: 4,
    cx: -0.42,
    cz: 0.56,
    radius: 0.44,
    topY: -0.32,
    bottomY: -1.12,
    basinDepth: 0.028,
    bevelWidth: 0.044,
    slopeX: 0.038,
    slopeZ: 0.016,
    marbleLightness: 0.20,
    veinSeed: 7.4,
    overflowTargetId: 6,
    overflowRimAngle: 0.16,
    overflowLandingX: -0.04,
    overflowLandingZ: 0.60,
  },
  // Pillar 5: Mid-right Flank Polished Carrara Marble Pillar -> cascades into Pillar 6
  {
    id: 5,
    cx: 0.56,
    cz: 0.36,
    radius: 0.42,
    topY: -0.18,
    bottomY: -1.08,
    basinDepth: 0.026,
    bevelWidth: 0.042,
    slopeX: -0.036,
    slopeZ: 0.028,
    marbleLightness: 0.89,
    veinSeed: 8.8,
    overflowTargetId: 6,
    overflowRimAngle: 2.54,
    overflowLandingX: 0.28,
    overflowLandingZ: 0.58,
  },
  // Pillar 6: Lower-front-right Broad Polished Dark Veined Rock & Calcite Sanctuary Pillar
  {
    id: 6,
    cx: 0.16,
    cz: 0.62,
    radius: 0.52,
    topY: -0.54,
    bottomY: -1.18,
    basinDepth: 0.032,
    bevelWidth: 0.050,
    slopeX: 0.010,
    slopeZ: 0.025,
    marbleLightness: 0.14,
    veinSeed: 10.2,
    overflowTargetId: -1,
    overflowRimAngle: 1.57,
    overflowLandingX: 0.16,
    overflowLandingZ: 0.72,
  },
];

/**
 * Evaluates the exact top height of a single polished pillar at offset `(lx, lz)` from its center.
 */
function evalPillarTopY(p: MarblePillarSpec, lx: number, lz: number): number {
  const r = Math.hypot(lx, lz);
  const rNorm = Math.min(1.2, r / p.radius);
  // Shallow concave catchment dish in center (rNorm < 0.86) + polished rounded rim bevel (0.86..1.0)
  const dish = Math.cos(Math.min(1, rNorm / 0.86) * Math.PI * 0.5);
  const rimLip = Math.exp(-Math.pow((rNorm - 0.88) / 0.09, 2.0)) * 0.010;
  const chamferDrop = rNorm > 0.92 ? Math.pow((rNorm - 0.92) / 0.08, 2.0) * 0.018 : 0.0;
  const slopeOffset = -(lx * p.slopeX + lz * p.slopeZ);
  return p.topY - dish * p.basinDepth + rimLip - chamferDrop + slopeOffset;
}

/**
 * Evaluates which pillar rules point `(x, z)` at elevation `maxY`:
 * Returns the highest pillar whose footprint contains `(x, z)` and whose `topY <= maxY + 0.08`.
 */
function findSupportingPillar(x: number, z: number, maxY = 10.0): {
  pillar: MarblePillarSpec | null;
  surfaceY: number;
  rNorm: number;
} {
  let bestPillar: MarblePillarSpec | null = null;
  let bestY = -10.0;
  let bestRNorm = 999;

  for (let i = 0; i < PILLARS.length; i++) {
    const p = PILLARS[i];
    if (p.topY > maxY + 0.12) continue;
    const lx = x - p.cx;
    const lz = z - p.cz;
    const r = Math.hypot(lx, lz);
    if (r <= p.radius * 1.02) {
      const sy = evalPillarTopY(p, lx, lz);
      if (sy > bestY) {
        bestY = sy;
        bestPillar = p;
        bestRNorm = r / p.radius;
      }
    }
  }
  return { pillar: bestPillar, surfaceY: bestY, rNorm: bestRNorm };
}

/**
 * Builds a high-precision, architectural polished marble pillar geometry with a
 * chamfered top rim, shallow concave catchment basin, smooth vertical polished walls,
 * and a tapered quarry-undercut base.
 */
function buildPolishedPillarGeometry(p: MarblePillarSpec): THREE.BufferGeometry {
  const height = p.topY - p.bottomY;
  const radialSegs = 80;
  const heightSegs = 40;
  // Place cylinder seam on the back (-Z, thetaStart = Math.PI) so the front face is 100% seamless
  const geom = new THREE.CylinderGeometry(
    p.radius,
    p.radius,
    height,
    radialSegs,
    heightSegs,
    false,
    Math.PI,
    Math.PI * 2
  );

  const posAttr = geom.attributes.position as THREE.BufferAttribute;
  const vCount = posAttr.count;
  // aPillarMeta: [marbleLightness, isTopSurface, wallDropMeters, veinSeed]
  const meta = new Float32Array(vCount * 4);
  const midY = (p.topY + p.bottomY) * 0.5;

  for (let i = 0; i < vCount; i++) {
    let lx = posAttr.getX(i);
    let y = posAttr.getY(i) + midY;
    let lz = posAttr.getZ(i);

    const vNorm = Math.max(0, Math.min(1, (y - p.bottomY) / Math.max(1e-4, height)));
    const r = Math.hypot(lx, lz);
    const rNorm = r / p.radius;
    const isTopCap = y >= p.topY - 0.01;
    const isBottomCap = y <= p.bottomY + 0.01;

    let isTopSurf = 0.0;
    let wallDropMeters = 0.0;

    if (isTopCap) {
      y = evalPillarTopY(p, lx, lz);
      isTopSurf = 1.0;
      wallDropMeters = 0.0;
    } else if (isBottomCap) {
      const baseTaper = 0.68 * (0.6 + 0.4 * (1.0 - rNorm));
      lx *= baseTaper;
      lz *= baseTaper;
      const crag = (fbm3cpu(lx * 3.5 + p.cx, p.bottomY * 3.0, lz * 3.5 + p.cz) - 0.5) * 0.06;
      y = p.bottomY + crag - (1.0 - rNorm) * 0.08;
      wallDropMeters = height;
    } else {
      const topEdgeY = evalPillarTopY(p, lx, lz);
      if (vNorm < 0.24) {
        const rootT = vNorm / 0.24;
        const smoothRoot = 0.68 + 0.32 * (rootT * rootT * (3 - 2 * rootT));
        lx *= smoothRoot;
        lz *= smoothRoot;
      }
      if (vNorm > 0.92) {
        const bevelT = (vNorm - 0.92) / 0.08;
        y = y * (1.0 - bevelT) + topEdgeY * bevelT;
        const chamferIn = 1.0 - Math.pow(bevelT, 2.0) * 0.024;
        lx *= chamferIn;
        lz *= chamferIn;
        isTopSurf = bevelT * 0.92;
      }
      wallDropMeters = Math.max(0.0, topEdgeY - y);
    }

    posAttr.setXYZ(i, lx + p.cx, y, lz + p.cz);

    meta[i * 4 + 0] = p.marbleLightness * (0.66 + 0.34 * Math.pow(vNorm, 0.28));
    meta[i * 4 + 1] = isTopSurf;
    meta[i * 4 + 2] = wallDropMeters;
    meta[i * 4 + 3] = p.veinSeed;
  }

  geom.setAttribute('aPillarMeta', new THREE.BufferAttribute(meta, 4));
  geom.computeVertexNormals();

  // Weld cylinder seam normals (column 0 and column radialSegs on each height ring) so zero seam line ever appears
  const normAttr = geom.attributes.normal as THREE.BufferAttribute;
  for (let row = 0; row <= heightSegs; row++) {
    const idx0 = row * (radialSegs + 1);
    const idx1 = idx0 + radialSegs;
    const nx = (normAttr.getX(idx0) + normAttr.getX(idx1)) * 0.5;
    const ny = (normAttr.getY(idx0) + normAttr.getY(idx1)) * 0.5;
    const nz = (normAttr.getZ(idx0) + normAttr.getZ(idx1)) * 0.5;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    normAttr.setXYZ(idx0, nx / nLen, ny / nLen, nz / nLen);
    normAttr.setXYZ(idx1, nx / nLen, ny / nLen, nz / nLen);
  }
  return geom;
}

/**
 * Builds a high-precision, macro-inspectable botanical star-bryophyte gametophyte tuft
 * (`Dicranum` / `Leucobryum` / `Bryum` cushion architecture):
 * - Part A: Continuous micro-velvet cushion domelet (`8x3` hemisphere, `r = 0.0155, h = 0.0095`)
 *   so adjacent tufts coalesce into a seamless, volumetric 3D velvet moss pillow with zero bare gaps.
 * - Part B: 24 finely tapered, curved lanceolate phyllid needles arranged across 3 phyllotaxis whorls
 *   (inner upright apical crown spears, mid-tier arching rosettes, and outer creeping basal fronds),
 *   each with a 3D V-folded central midrib (`costa`) normal ridge so close-up zoom inspection reveals
 *   razor-sharp botanical needle detail!
 */
function createMicroBryophyteFloretGeometry(): THREE.BufferGeometry {
  const domeRadial = 8;
  const domeRings = 3;
  const domeVertCount = 1 + domeRings * domeRadial;

  const frondCount = 24;
  const segsY = 4;
  const vertsPerFrond = (segsY + 1) * 2;
  const totalVerts = domeVertCount + frondCount * vertsPerFrond;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const frondMeta = new Float32Array(totalVerts * 4);
  const indices: number[] = [];

  // Part A: Coalescing Velvet Cushion Sub-Pillow (`vIdx = 0 .. domeVertCount - 1`)
  const domeRadius = 0.0155;
  const domeHeight = 0.0092;
  positions[0] = 0;
  positions[1] = domeHeight;
  positions[2] = 0;
  normals[0] = 0;
  normals[1] = 1;
  normals[2] = 0;
  uvs[0] = 0.5;
  uvs[1] = 0.36;
  frondMeta[0] = 0;
  frondMeta[1] = 0;
  frondMeta[2] = 0;
  frondMeta[3] = 0.0;

  for (let r = 1; r <= domeRings; r++) {
    const ringT = r / domeRings;
    const elevAngle = ringT * Math.PI * 0.48;
    const rad = Math.sin(elevAngle) * domeRadius;
    const y = Math.cos(elevAngle) * domeHeight;
    const ny = Math.cos(elevAngle);
    const nHoriz = Math.sin(elevAngle);

    for (let s = 0; s < domeRadial; s++) {
      const phi = (s / domeRadial) * Math.PI * 2;
      const cosP = Math.cos(phi);
      const sinP = Math.sin(phi);
      const vIdx = 1 + (r - 1) * domeRadial + s;

      positions[vIdx * 3 + 0] = cosP * rad;
      positions[vIdx * 3 + 1] = y;
      positions[vIdx * 3 + 2] = sinP * rad;

      normals[vIdx * 3 + 0] = cosP * nHoriz;
      normals[vIdx * 3 + 1] = ny;
      normals[vIdx * 3 + 2] = sinP * nHoriz;

      uvs[vIdx * 2 + 0] = s / domeRadial;
      uvs[vIdx * 2 + 1] = (1.0 - ringT) * 0.36;

      frondMeta[vIdx * 4 + 0] = cosP * ringT * 0.35;
      frondMeta[vIdx * 4 + 1] = sinP * ringT * 0.35;
      frondMeta[vIdx * 4 + 2] = 0;
      frondMeta[vIdx * 4 + 3] = ringT * 0.22;
    }
  }

  for (let s = 0; s < domeRadial; s++) {
    const nextS = (s + 1) % domeRadial;
    indices.push(0, 1 + s, 1 + nextS);
  }
  for (let r = 1; r < domeRings; r++) {
    const ring0 = 1 + (r - 1) * domeRadial;
    const ring1 = 1 + r * domeRadial;
    for (let s = 0; s < domeRadial; s++) {
      const nextS = (s + 1) % domeRadial;
      indices.push(ring0 + s, ring1 + s, ring0 + nextS);
      indices.push(ring0 + nextS, ring1 + s, ring1 + nextS);
    }
  }

  // Part B: 24 Multi-Whorl Lanceolate Bryophyte Phyllids (`vIdx >= domeVertCount`)
  const goldenAngle = 2.399963229728653;

  for (let f = 0; f < frondCount; f++) {
    const tierNorm = f / (frondCount - 1);
    const angle = f * goldenAngle + Math.sin(f * 1.7) * 0.08;
    const dx = Math.cos(angle);
    const dz = Math.sin(angle);
    const px = -dz;
    const pz = dx;

    // Inner whorls stand upright like star-moss crown spears; outer whorls arch and feather over the pillow
    const splay = 0.0055 + Math.pow(tierNorm, 1.02) * 0.0135;
    const archPeakHeight = (0.0152 - tierNorm * 0.0064) * (0.92 + 0.16 * Math.cos(f * 2.3));
    const baseHalfWidth = 0.0024 * (1.08 - tierNorm * 0.22);
    const rootRadius = 0.0014 + tierNorm * 0.0068;
    const rootY = 0.0032 + (1.0 - tierNorm) * 0.0048;
    const falcateCurl = Math.sin(f * 3.1) * 0.0022 * (0.4 + tierNorm * 0.6);

    for (let s = 0; s <= segsY; s++) {
      const t = s / segsY;
      // Sharp acuminate needle taper at tip
      const widthEnvelope = Math.sin(Math.pow(t, 0.52) * Math.PI * 0.96) * Math.pow(1.0 - t * 0.85, 0.62);
      const halfW = Math.max(0.00018, baseHalfWidth * widthEnvelope);

      const radialOut = rootRadius + (t * 0.56 + Math.pow(t, 1.45) * 0.44) * splay;
      const archCurve = Math.sin(t * Math.PI * (0.65 + tierNorm * 0.22));
      const y = rootY + archCurve * archPeakHeight;
      const latOffset = falcateCurl * t * t;
      const cx = dx * radialOut + px * latOffset;
      const cz = dz * radialOut + pz * latOffset;

      for (let side = 0; side < 2; side++) {
        const sign = side === 0 ? -1 : 1;
        const vIdx = domeVertCount + f * vertsPerFrond + s * 2 + side;

        positions[vIdx * 3 + 0] = cx + px * sign * halfW;
        positions[vIdx * 3 + 1] = y + Math.abs(sign) * halfW * 0.22;
        positions[vIdx * 3 + 2] = cz + pz * sign * halfW;

        const nx = dx * 0.36 + px * sign * 0.42;
        const ny = 0.83;
        const nz = dz * 0.36 + pz * sign * 0.42;
        const nLen = Math.hypot(nx, ny, nz) || 1;
        normals[vIdx * 3 + 0] = nx / nLen;
        normals[vIdx * 3 + 1] = ny / nLen;
        normals[vIdx * 3 + 2] = nz / nLen;

        uvs[vIdx * 2 + 0] = side;
        uvs[vIdx * 2 + 1] = 0.20 + t * 0.80;

        frondMeta[vIdx * 4 + 0] = dx;
        frondMeta[vIdx * 4 + 1] = dz;
        frondMeta[vIdx * 4 + 2] = sign;
        frondMeta[vIdx * 4 + 3] = tierNorm;
      }

      if (s < segsY) {
        const row0 = domeVertCount + f * vertsPerFrond + s * 2;
        const row1 = row0 + 2;
        indices.push(row0, row0 + 1, row1);
        indices.push(row0 + 1, row1 + 1, row1);
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aFrondMeta', new THREE.BufferAttribute(frondMeta, 4));
  geom.setIndex(indices);
  return geom;
}

/**
 * Creates a hair-fine bryophyte sporophyte pin stalk (`seta`) crowned with a nodding
 * golden-amber spore capsule (`theca` + `operculum`, `height = 0.048`).
 */
function createMicroPinSporophyteGeometry(): THREE.BufferGeometry {
  const radialSegs = 8;
  const lengthSegs = 14;
  const vertCount = (lengthSegs + 1) * (radialSegs + 1);
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  const totalHeight = 0.048;

  for (let i = 0; i <= lengthSegs; i++) {
    const t = i / lengthSegs;
    let cx = 0;
    let cy = t * totalHeight;
    let tangentX = 0;
    let tangentY = 1;

    if (t > 0.66) {
      const archT = (t - 0.66) / 0.34;
      const archAngle = archT * Math.PI * 0.48;
      cx = (1 - Math.cos(archAngle)) * 0.0058;
      cy = 0.66 * totalHeight + Math.sin(archAngle) * 0.0088;
      tangentX = Math.sin(archAngle);
      tangentY = Math.cos(archAngle);
    } else {
      cx = Math.sin(t * Math.PI) * 0.0012;
      tangentX = Math.cos(t * Math.PI) * 0.0040;
      tangentY = 1.0;
    }

    const tLen = Math.hypot(tangentX, tangentY) || 1;
    tangentX /= tLen;
    tangentY /= tLen;
    const norm2DX = tangentY;
    const norm2DY = -tangentX;

    let radius = 0.00052 * (1.0 - t * 0.16);
    if (t > 0.74) {
      const capT = (t - 0.74) / 0.26;
      const capsuleBell = Math.sin(Math.pow(capT, 0.68) * Math.PI);
      radius = 0.00048 + capsuleBell * 0.0026;
      if (capT > 0.94) radius *= Math.max(0, (1.0 - capT) * 16.0);
    }

    for (let j = 0; j <= radialSegs; j++) {
      const u = j / radialSegs;
      const phi = u * Math.PI * 2;
      const cosP = Math.cos(phi);
      const sinP = Math.sin(phi);

      const idx = i * (radialSegs + 1) + j;
      positions[idx * 3 + 0] = cx + norm2DX * cosP * radius;
      positions[idx * 3 + 1] = cy + norm2DY * cosP * radius;
      positions[idx * 3 + 2] = sinP * radius;

      normals[idx * 3 + 0] = norm2DX * cosP;
      normals[idx * 3 + 1] = norm2DY * cosP;
      normals[idx * 3 + 2] = sinP;

      uvs[idx * 2 + 0] = u;
      uvs[idx * 2 + 1] = t;
    }
  }

  for (let i = 0; i < lengthSegs; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * (radialSegs + 1) + j;
      const b = a + radialSegs + 1;
      indices.push(a, b, a + 1);
      indices.push(a + 1, b, b + 1);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  return geom;
}

/**
 * Builds a dynamic deformable pour-stream tube geometry (`radialSegs = 14`, `lengthSegs = 28`)
 * whose centerline vertices are updated in-place whenever green botanical liquid is poured.
 */
function createDynamicPourStreamGeometry(): THREE.BufferGeometry {
  const radialSegs = 14;
  const lengthSegs = 28;
  const vertCount = (lengthSegs + 1) * (radialSegs + 1);
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  for (let i = 0; i <= lengthSegs; i++) {
    const t = i / lengthSegs;
    for (let j = 0; j <= radialSegs; j++) {
      const u = j / radialSegs;
      const idx = i * (radialSegs + 1) + j;
      uvs[idx * 2 + 0] = u;
      uvs[idx * 2 + 1] = t;
      normals[idx * 3 + 1] = 1;
    }
  }

  for (let i = 0; i < lengthSegs; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * (radialSegs + 1) + j;
      const b = a + radialSegs + 1;
      indices.push(a, b, a + 1);
      indices.push(a + 1, b, b + 1);
    }
  }

  const geom = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  const normAttr = new THREE.BufferAttribute(normals, 3);
  normAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', posAttr);
  geom.setAttribute('normal', normAttr);
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  return geom;
}

function updateDynamicPourStreamMesh(
  geom: THREE.BufferGeometry,
  startPos: THREE.Vector3,
  endPos: THREE.Vector3,
  velLagX: number,
  velLagZ: number,
  elapsedSec: number,
  streamStrength: number
): void {
  const radialSegs = 14;
  const lengthSegs = 28;
  const posAttr = geom.attributes.position as THREE.BufferAttribute;
  const normAttr = geom.attributes.normal as THREE.BufferAttribute;

  const dx = endPos.x - startPos.x;
  const dy = endPos.y - startPos.y;
  const dz = endPos.z - startPos.z;

  // Parabolic vessel pour arc from above-right into a vertical gravitational strike
  const spoutArcX = 0.085;
  const spoutArcZ = 0.055;

  for (let i = 0; i <= lengthSegs; i++) {
    const t = i / lengthSegs;
    // Gravitational acceleration curve t^1.58
    const gravT = Math.pow(t, 1.58);
    const arcEnv = Math.pow(1.0 - t, 1.65);
    const bow = Math.sin(t * Math.PI) * 0.14;
    const waveX = Math.sin(t * 15.0 - elapsedSec * 18.0) * 0.0024 * t;
    const waveZ = Math.cos(t * 13.0 - elapsedSec * 16.0) * 0.0024 * t;

    const cx = startPos.x + spoutArcX * arcEnv + dx * t - velLagX * bow + waveX;
    const cy = startPos.y + dy * gravT;
    const cz = startPos.z + spoutArcZ * arcEnv + dz * t - velLagZ * bow + waveZ;

    // Taper smoothly to 0 at t = 0 (so no chopped cylinder top ever shows in mid-air),
    // thin via Rayleigh-Plateau acceleration in mid-flight, and flare into a meniscus foot on contact!
    const topPinch = Math.sin(Math.min(1.0, t / 0.14) * Math.PI * 0.5);
    const footFlare = Math.pow(Math.max(0, (t - 0.85) / 0.15), 2.1) * 0.024;
    const pulse = 1.0 + 0.09 * Math.sin(t * 19.0 - elapsedSec * 24.0);
    const baseR =
      (0.0135 * topPinch * (1.0 - t * 0.38) + footFlare) * streamStrength * pulse;

    for (let j = 0; j <= radialSegs; j++) {
      const phi = (j / radialSegs) * Math.PI * 2;
      const nx = Math.cos(phi);
      const nz = Math.sin(phi);
      const idx = i * (radialSegs + 1) + j;

      posAttr.setXYZ(idx, cx + nx * baseR, cy, cz + nz * baseR);
      normAttr.setXYZ(idx, nx, 0.18, nz);
    }
  }

  posAttr.needsUpdate = true;
  normAttr.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// High-Precision 256x256 Surface Grid (Liquid -> Mold Color Shift -> Moss Bloom)
// ---------------------------------------------------------------------------
const GRID_RES = 256;
const WORLD_MIN_X = -1.15;
const WORLD_MAX_X = 1.15;
const WORLD_MIN_Z = -0.85;
const WORLD_MAX_Z = 1.25;
const WORLD_SPAN_X = WORLD_MAX_X - WORLD_MIN_X;
const WORLD_SPAN_Z = WORLD_MAX_Z - WORLD_MIN_Z;

function worldToGridX(wx: number): number {
  return ((wx - WORLD_MIN_X) / WORLD_SPAN_X) * (GRID_RES - 1);
}

function worldToGridZ(wz: number): number {
  return ((wz - WORLD_MIN_Z) / WORLD_SPAN_Z) * (GRID_RES - 1);
}

interface MicroFloretNode {
  pillarId: number;
  isWallNode: boolean;
  anchorPos: THREE.Vector3;
  baseNormal: THREE.Vector3;
  baseQuat: THREE.Quaternion;
  maxScale: number;
  /** Height of the underlying 3D moss hummock pillow at this node (`0..0.014`) */
  hummockLift: number;
  /** Grid index into `gridLiquid`, `gridMold`, & `gridMoss` */
  gridIdx: number;
  /** For vertical wall nodes, how far down from the top rim `[0..0.44]` */
  wallDropDist: number;
  /** Organic hummock nucleation threshold in `[0, 0.38]` */
  filigreeThreshold: number;
  bloom: number;
  bendX: number;
  bendZ: number;
  velBendX: number;
  velBendZ: number;
  seedPhase: number;
}

interface PinSporophyteNode {
  floretIndex: number;
  baseQuat: THREE.Quaternion;
  maxScale: number;
  growth: number;
}

type DropletMode = 'falling' | 'splash' | 'wall-drip';

interface AerialDroplet {
  active: boolean;
  mode: DropletMode;
  pillarId: number;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  radius: number;
  age: number;
  maxLife: number;
}

export async function bootDay05Moss(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(5);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.62, 4.15],
    fov: 34,
    enableOrbit: true,
  });

  // Configure OrbitControls so the user can freely zoom right into macro bryophyte detail
  // and orbit around all 7 stepped marble pillars to explore the mossing!
  stage.controls.enableZoom = true;
  stage.controls.zoomSpeed = 1.15;
  stage.controls.rotateSpeed = 0.85;
  stage.controls.minDistance = 0.72;
  stage.controls.maxDistance = 7.5;
  stage.controls.minPolarAngle = 0.08;
  stage.controls.maxPolarAngle = Math.PI * 0.82;
  stage.controls.target.set(0.0, -0.08, 0.14);
  stage.controls.update();

  // Position and orient the pillar cluster once—keep rock-solid so marble veins NEVER swim!
  stage.specimenGroup.position.set(0.02, 0.04, -0.08);
  stage.specimenGroup.rotation.set(0.24, -0.18, 0.0);
  stage.specimenGroup.updateMatrixWorld();

  // -------------------------------------------------------------------------
  // Physical Mineral & Botanical Pigments (`src/core/palette.ts`)
  // -------------------------------------------------------------------------
  const calcitePigment = createPigmentUniforms('boneCalcite');
  const kaolinPigment = createPigmentUniforms('warmKaolin');
  const umberPigment = createPigmentUniforms('rawUmber');
  const azuritePigment = createPigmentUniforms('azuritePrussian');
  const malachitePigment = createPigmentUniforms('verdigrisMalachite');
  const celadonPigment = createPigmentUniforms('celadonGlauconite');
  const mossPigment = createPigmentUniforms('mossUmbrous');
  const ochrePigment = createPigmentUniforms('yellowOchre');
  const sporePigment = createPigmentUniforms('sporeSulfur');
  const siennaPigment = createPigmentUniforms('burntSienna');
  const rustPigment = createPigmentUniforms('hematiteRust');

  // Shared uniform for live concentric pour-impact ripples on the pooled green liquid
  const uPourImpactLocal = uniform(new THREE.Vector4(0.28, PILLARS[1].topY, -0.04, 1.0));

  const rand = mulberry32(20261105);

  // -------------------------------------------------------------------------
  // 1. Live 256x256 4-Stage Surface Field:
  //    R = Liquid Film (`gridLiquid`), G = Mold Color-Shift Biofilm (`gridMold`),
  //    B = Mature Bryophyte Moss (`gridMoss`), A = Rim Cascade Runoff (`gridCascade`)
  // -------------------------------------------------------------------------
  const gridCellCount = GRID_RES * GRID_RES;
  const gridLiquid = new Float32Array(gridCellCount); // Stage 1: Glistening translucent green liquid [0..1]
  const gridLiquidNext = new Float32Array(gridCellCount);
  const gridWetAge = new Float32Array(gridCellCount); // Seconds liquid has dwelled at cell
  const gridMold = new Float32Array(gridCellCount); // Stage 2: Olive-ochre-umber molding patination [0..1]
  const gridMoss = new Float32Array(gridCellCount); // Stage 3: 3D Bryophyte moss colonization [0..1]
  const gridCascade = new Float32Array(gridCellCount); // Active vertical rim-overflow depth [0..1]
  const gridPillarId = new Int16Array(gridCellCount); // -1 if empty space, 0..6 if on pillar
  const gridIsRim = new Uint8Array(gridCellCount); // 1 if near pillar outer rim lip (rNorm > 0.88)
  const gridOrganicBias = new Float32Array(gridCellCount); // Multi-scale FBM organic vein/hummock bias
  const gridWarpX = new Float32Array(gridCellCount); // Domain-warp X offset for non-circular organic pools
  const gridWarpZ = new Float32Array(gridCellCount); // Domain-warp Z offset for non-circular organic pools

  for (let gz = 0; gz < GRID_RES; gz++) {
    const wz = WORLD_MIN_Z + (gz / (GRID_RES - 1)) * WORLD_SPAN_Z;
    for (let gx = 0; gx < GRID_RES; gx++) {
      const wx = WORLD_MIN_X + (gx / (GRID_RES - 1)) * WORLD_SPAN_X;
      const idx = gz * GRID_RES + gx;
      const support = findSupportingPillar(wx, wz, 10.0);
      if (support.pillar) {
        const p = support.pillar;
        gridPillarId[idx] = p.id;
        if (support.rNorm >= 0.88) {
          gridIsRim[idx] = 1;
        }
        const nMacro = fbm3cpu(wx * 8.5 + p.veinSeed, 1.2, wz * 8.5 - p.veinSeed);
        const nMicro = fbm3cpu(wx * 22.0 - p.veinSeed, 3.7, wz * 22.0 + p.veinSeed);
        gridOrganicBias[idx] = nMacro * 0.62 + nMicro * 0.38;
        gridWarpX[idx] = (fbm3cpu(wx * 7.2 + p.veinSeed, 2.1, wz * 7.2) - 0.5) * 0.14;
        gridWarpZ[idx] = (fbm3cpu(wx * 7.2, 4.3, wz * 7.2 - p.veinSeed) - 0.5) * 0.14;
      } else {
        gridPillarId[idx] = -1;
      }
    }
  }

  /**
   * Deposits translucent green botanical liquid onto the marble surface at `(wx, wz)` with
   * 3D domain-warped FBM contours so pools and moss colonies form natural lobed botanical shapes
   * following the stone veins—never synthetic geometric circles!
   */
  function depositGreenLiquid(
    wx: number,
    wz: number,
    radius: number,
    amount: number,
    targetPillarId = -1,
    initialAgeBoost = 0.0,
    allowRimCascade = false
  ): void {
    const padRadius = radius * 1.45;
    const gxCenter = Math.round(worldToGridX(wx));
    const gzCenter = Math.round(worldToGridZ(wz));
    const rCellsX = Math.ceil((padRadius / WORLD_SPAN_X) * GRID_RES);
    const rCellsZ = Math.ceil((padRadius / WORLD_SPAN_Z) * GRID_RES);
    const rSq = radius * radius;

    const minGx = Math.max(0, gxCenter - rCellsX);
    const maxGx = Math.min(GRID_RES - 1, gxCenter + rCellsX);
    const minGz = Math.max(0, gzCenter - rCellsZ);
    const maxGz = Math.min(GRID_RES - 1, gzCenter + rCellsZ);

    for (let gz = minGz; gz <= maxGz; gz++) {
      const cz = WORLD_MIN_Z + (gz / (GRID_RES - 1)) * WORLD_SPAN_Z;
      for (let gx = minGx; gx <= maxGx; gx++) {
        const idx = gz * GRID_RES + gx;
        const cellPillar = gridPillarId[idx];
        if (cellPillar === -1) continue;
        if (targetPillarId !== -1 && cellPillar !== targetPillarId) continue;

        const cx = WORLD_MIN_X + (gx / (GRID_RES - 1)) * WORLD_SPAN_X;
        // Domain-warped coordinates so every pool & moss cushion has organic lobed contours
        const dx = cx + gridWarpX[idx] * radius * 2.2 - wx;
        const dz = cz + gridWarpZ[idx] * radius * 2.2 - wz;
        const dSq = dx * dx + dz * dz;
        if (dSq < rSq) {
          const bias = 0.72 + gridOrganicBias[idx] * 0.56;
          const falloff = Math.exp(-dSq / (rSq * 0.42)) * bias;
          const added = amount * falloff;
          gridLiquid[idx] = Math.min(1.0, gridLiquid[idx] + added);
          if (allowRimCascade && gridIsRim[idx] === 1 && falloff > 0.32) {
            gridCascade[idx] = Math.min(1.0, gridCascade[idx] + added * 0.9);
          }
          if (initialAgeBoost > 0 && falloff > 0.12) {
            gridWetAge[idx] = Math.max(gridWetAge[idx], initialAgeBoost * Math.min(1.1, falloff * 1.15));
          }
        }
      }
    }
  }

  // Curated Multi-Stage Botanical Boot State:
  // 1. Pillar 0 (Upper Calacatta Crown) -> Pillar 2 (Central Dark Nero Marquina Heart):
  //    An organic, multi-lobed moss cushion hugging the front-right spillover lip of Pillar 0
  //    and cascading down into the central dark stone basin of Pillar 2, showing all 3 concentric rings:
  //    - Plush 3D bryophyte hummock core (`age ~ 6.5s`)
  //    - Warm olive-ochre-umber molding transition rim (`age ~ 2.4s`)
  //    - Glistening translucent emerald liquid leading edge (`age ~ 0.6s`)
  // 2. Pillar 1 (Upper-right Carrara Marble Terrace):
  //    Receives the live opening pour stream (`t = 0..2.2s`), starting as pure glistening green liquid
  //    that pools, cascades toward Pillar 2, subtly shifts color into olive-ochre mold, and blooms into moss!
  const bootColonies = [
    // Pillar 0 front-right rim sanctuary (lobed cushion right at the spillover lip!)
    { x: -0.06, z: -0.04, r: 0.25, liq: 0.98, age: 6.8, pId: 0, cascade: true },
    { x: -0.22, z: -0.14, r: 0.23, liq: 0.94, age: 6.2, pId: 0, cascade: false },
    { x: -0.12, z: 0.08, r: 0.19, liq: 0.92, age: 5.4, pId: 0, cascade: true },
    // Pillar 2 central dark rock sanctuary (fed by Pillar 0's cascade + molding outer lobe)
    { x: -0.04, z: 0.16, r: 0.28, liq: 0.96, age: 6.5, pId: 2, cascade: false },
    { x: 0.12, z: 0.28, r: 0.24, liq: 0.90, age: 3.2, pId: 2, cascade: true },
    // Pillar 1 (Carrara Terrace): fresh green liquid pool + nascent olive-ochre mold center under the pour stream
    { x: 0.26, z: -0.02, r: 0.22, liq: 0.88, age: 2.1, pId: 1, cascade: false },
    { x: 0.16, z: 0.10, r: 0.20, liq: 0.84, age: 0.7, pId: 1, cascade: true },
  ];

  for (let s = 0; s < bootColonies.length; s++) {
    const c = bootColonies[s];
    depositGreenLiquid(c.x, c.z, c.r, c.liq, c.pId, c.age, c.cascade);
  }

  // Evaluate initial `gridMold` and `gridMoss` from `gridWetAge`
  for (let i = 0; i < gridCellCount; i++) {
    if (gridPillarId[i] === -1) continue;
    const liq = gridLiquid[i];
    const age = gridWetAge[i];
    const org = gridOrganicBias[i];
    if (liq > 0.06 && age > 0.0) {
      const moldStart = 1.25 + (1.0 - org) * 0.70;
      const moldSpan = 1.55;
      gridMold[i] = Math.max(0.0, Math.min(1.0, ((age - moldStart) / moldSpan) * Math.min(1.0, liq * 1.35)));

      const mossStart = 2.65 + (1.0 - org) * 0.95;
      const mossSpan = 2.10;
      if (gridMold[i] > 0.35) {
        gridMoss[i] = Math.max(0.0, Math.min(1.0, ((age - mossStart) / mossSpan) * gridMold[i]));
      }
    }
  }

  // RGBA DataTexture encoding `[gridLiquid, gridMold, gridMoss, gridCascade]` at 256x256 resolution
  const moistureTexData = new Uint8Array(gridCellCount * 4);
  const moistureDataTex = new THREE.DataTexture(
    moistureTexData,
    GRID_RES,
    GRID_RES,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  moistureDataTex.wrapS = THREE.ClampToEdgeWrapping;
  moistureDataTex.wrapT = THREE.ClampToEdgeWrapping;
  moistureDataTex.magFilter = THREE.LinearFilter;
  moistureDataTex.minFilter = THREE.LinearFilter;

  function syncMoistureDataTexture(): void {
    for (let i = 0; i < gridCellCount; i++) {
      moistureTexData[i * 4 + 0] = Math.min(255, Math.floor(gridLiquid[i] * 255));
      moistureTexData[i * 4 + 1] = Math.min(255, Math.floor(gridMold[i] * 255));
      moistureTexData[i * 4 + 2] = Math.min(255, Math.floor(gridMoss[i] * 255));
      moistureTexData[i * 4 + 3] = Math.min(255, Math.floor(gridCascade[i] * 255));
    }
    moistureDataTex.needsUpdate = true;
  }
  syncMoistureDataTexture();

  // -------------------------------------------------------------------------
  // 2. Polished Calacatta / Carrara Marble + 4-Stage Liquid -> Mold -> 3D Hummock Moss Material
  // -------------------------------------------------------------------------
  const stoneMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.12,
    metalness: 0.03,
  });

  {
    const lPos = positionLocal;
    const lNorm = normalize(normalLocal);
    const wPos = positionWorld;
    const wNorm = normalize(normalWorld);
    const pMeta = attribute('aPillarMeta', 'vec4'); // [marbleLightness, isTopSurface, wallDropMeters, veinSeed]

    const marbleLightness = clamp(pMeta.x, 0.0, 1.0);
    const isTopSurface = pMeta.y;
    const wallDropMeters = pMeta.z;
    const veinSeed = pMeta.w;

    // Sample the live 256x256 4-stage DataTexture in uniform control flow
    const gridUvX = clamp(lPos.x.sub(WORLD_MIN_X).div(WORLD_SPAN_X), 0.0, 1.0);
    const gridUvY = clamp(lPos.z.sub(WORLD_MIN_Z).div(WORLD_SPAN_Z), 0.0, 1.0);
    const surfaceSample = texture(moistureDataTex, vec2(gridUvX, gridUvY));

    // Organic Narrow Vertical Rivulet Channels (No Wide Flat Paint Bands on Cylinder Sides!):
    // On vertical walls (`isTopSurface < 0.5`), liquid & moss only travel down slender,
    // gravity-drawn azimuthal rivulet tendrils where `surfaceSample.a` (rim cascade) is active!
    const rivuletTendrilNoise = mineralToothFbm(
      lPos.mul(vec3(18.0, 3.2, 18.0)).add(veinSeed.mul(2.3)),
      float(2.4)
    );
    const tendrilChannel = smoothstep(0.42, 0.78, rivuletTendrilNoise);
    const maxRivuletReach = surfaceSample.a
      .mul(tendrilChannel)
      .mul(float(0.38))
      .add(float(0.012));
    const wallRivuletEnvelope = smoothstep(
      maxRivuletReach,
      maxRivuletReach.mul(0.22),
      wallDropMeters
    ).mul(smoothstep(0.18, 0.52, surfaceSample.a.mul(tendrilChannel)));

    const surfaceMask = mix(wallRivuletEnvelope, float(1.0), isTopSurface);

    const localLiquid = clamp(surfaceSample.r.mul(surfaceMask), 0.0, 1.0);
    const localMold = clamp(surfaceSample.g.mul(surfaceMask), 0.0, 1.0);
    const localMoss = clamp(surfaceSample.b.mul(surfaceMask), 0.0, 1.0);

    // Physical 3D Volumetric Hummock & Meniscus Relief along the Stone Normal:
    // 1. Fresh Liquid (`localLiquid`): 1.2mm glossy pool with a raised meniscus rim + live concentric pour ripples!
    // 2. Mold Biofilm (`localMold`): 2.0mm coagulated protonema crust
    // 3. Mature Moss (`localMoss`): 11.5mm billowing multi-lobed 3D bryophyte hummock cushions!
    const hummockMacro = mineralToothFbm(lPos.mul(9.5).add(veinSeed), float(2.0));
    const hummockMeso = mineralToothFbm(lPos.mul(21.0).sub(veinSeed), float(2.4));
    const hummockPillow = sin(clamp(hummockMacro.mul(1.15).sub(0.08), 0.0, 1.0).mul(1.570796)).mul(0.72).add(
      hummockMeso.mul(0.28)
    );

    // Live concentric capillary ripples radiating from `uPourImpactLocal.xyz` across fresh liquid pools
    const distToPour = lPos.sub(uPourImpactLocal.xyz).length();
    const rippleWave = sin(distToPour.mul(85.0).sub(stage.uTime.mul(16.0)))
      .mul(exp(distToPour.negate().mul(6.5)))
      .mul(uPourImpactLocal.w);
    const pureLiquidFactor = localLiquid.mul(float(1.0).sub(localMold.mul(0.85)));
    const liquidMeniscusLip = smoothstep(0.04, 0.22, localLiquid).mul(
      float(1.0).sub(smoothstep(0.25, 0.58, localMold))
    );

    const normalDisplacement = liquidMeniscusLip
      .mul(0.0012)
      .add(rippleWave.mul(pureLiquidFactor).mul(0.0008))
      .add(localMold.mul(0.0020).mul(float(0.6).add(hummockMeso.mul(0.5))))
      .add(localMoss.mul(0.0115).mul(hummockPillow).mul(isTopSurface.mul(0.75).add(0.25)));

    stoneMaterial.positionNode = lPos.add(lNorm.mul(normalDisplacement));

    // Multi-layered Italian Calacatta & Carrara Marble Veining (100% static in `lPos`!)
    const marbleCoord = lPos.add(vec3(veinSeed.mul(1.7), veinSeed.mul(0.9), veinSeed.mul(-1.3)));
    const macroWarp = mineralToothFbm(marbleCoord.mul(1.35), float(1.8));
    const microWarp = mineralToothFbm(marbleCoord.mul(3.8).add(macroWarp), float(2.4));

    const warpedP = marbleCoord
      .mul(vec3(2.1, 1.6, 2.1))
      .add(vec3(macroWarp.mul(1.4), microWarp.mul(0.75), macroWarp.mul(-1.25)));
    const veinField = mineralToothFbm(warpedP, float(2.0));
    const fineCrystalline = valueNoise3DTsl(warpedP.mul(5.5));

    const smokyCloud = smoothstep(0.32, 0.68, veinField);
    const sharpGreyVein = exp(veinField.sub(0.51).abs().negate().mul(36.0));
    const goldVein = exp(
      veinField.add(fineCrystalline.mul(0.12)).sub(0.64).abs().negate().mul(48.0)
    );

    const pureCalcite = mix(calcitePigment.rInf, kaolinPigment.rInf, smokyCloud.mul(0.38));
    const greyVeinRgb = mix(
      azuritePigment.rInf.mul(0.52),
      umberPigment.rInf.mul(0.68),
      microWarp
    );
    const goldVeinRgb = mix(ochrePigment.rInf, siennaPigment.rInf, fineCrystalline.mul(0.35));

    const whiteMarbleWithClouds = mix(pureCalcite, greyVeinRgb, smokyCloud.mul(0.22));
    const withGreySeams = mix(whiteMarbleWithClouds, greyVeinRgb, sharpGreyVein.mul(0.52));
    const calacattaMarbleRgb = mix(withGreySeams, goldVeinRgb, goldVein.mul(0.62));

    // Dark polished Nero Marquina / Verde Alpi rock for the dark stone pillars (`Pillars 2, 4, 6`)
    const darkRockMatrix = vec3(
      CHALKY_SLATE_LINEAR[0] * 0.62,
      CHALKY_SLATE_LINEAR[1] * 0.68,
      CHALKY_SLATE_LINEAR[2] * 0.66
    );
    const darkSerpentineCloud = kubelkaMunkGlaze(
      mossPigment.K,
      azuritePigment.S,
      float(0.32).add(macroWarp.mul(0.28)),
      darkRockMatrix
    );
    const darkVeinedStone = mix(
      darkSerpentineCloud,
      mix(calcitePigment.rInf, ochrePigment.rInf, microWarp.mul(0.55)),
      sharpGreyVein.mul(0.48).add(goldVein.mul(0.58))
    );

    const baseMarbleRgb = mix(
      darkVeinedStone,
      calacattaMarbleRgb,
      smoothstep(0.25, 0.72, marbleLightness)
    );

    // -----------------------------------------------------------------------
    // 3-Phase Optical Progression on the Marble Surface:
    // Phase 1 (`localLiquid`): Translucent Emerald-Jade Botanical Elixir
    // Phase 2 (`localMold`): "Molding" Olive-Ochre-Umber Biofilm Color Shift
    // Phase 3 (`localMoss`): Deep Woodland-Moss Velvet Hummock Cushion
    // -----------------------------------------------------------------------
    const cellularFbm = mineralToothFbm(lPos.mul(36.0), float(3.2));
    const microSporeNoise = valueNoise3DTsl(lPos.mul(92.0));

    // Phase 1: Translucent jade-emerald liquid pool over veined marble
    const liquidTransmittance = beerLambertTransmittance(
      malachitePigment.extinction,
      localLiquid.mul(float(1.05).add(cellularFbm.mul(0.35)))
    );
    const emeraldTint = mix(
      malachitePigment.rInf,
      mix(celadonPigment.rInf, sporePigment.rInf, float(0.36)),
      float(0.54)
    );
    const pooledLiquidRgb = mix(
      emeraldTint.mul(0.78),
      baseMarbleRgb.mul(
        mix(vec3(1.0, 1.0, 1.0), emeraldTint.mul(1.22), smoothstep(0.02, 0.26, localLiquid))
      ),
      liquidTransmittance.mul(0.58)
    );
    const stage1Rgb = mix(baseMarbleRgb, pooledLiquidRgb, smoothstep(0.03, 0.18, localLiquid));

    // Phase 2: The "Molding" Color Shift (Earthy Olive-Umber + Yellow Ochre + Spore Sulfur Liesegang Rings)
    const oliveUmberMoldBase = kubelkaMunkGlaze(
      umberPigment.K.add(mossPigment.K.mul(0.65)),
      ochrePigment.S.add(mossPigment.S.mul(0.45)),
      float(0.66).add(cellularFbm.mul(0.48)),
      mix(umberPigment.rInf, ochrePigment.rInf, float(0.54))
    );
    const moldRingPattern = smoothstep(
      0.32,
      0.76,
      cellularFbm.mul(0.65).add(microSporeNoise.mul(0.35))
    );
    const ochreSporeMold = mix(
      ochrePigment.rInf,
      mix(sporePigment.rInf, celadonPigment.rInf, float(0.32)),
      microSporeNoise
    );
    const moldedBiofilmRgb = mix(oliveUmberMoldBase, ochreSporeMold, moldRingPattern.mul(0.62));

    const stage2Rgb = mix(stage1Rgb, moldedBiofilmRgb, smoothstep(0.05, 0.80, localMold));

    // Phase 3: Lush Velvet Bryophyte Hummock Cushion (with deep inter-hummock cleavage shadows!)
    const deepVelvetBase = kubelkaMunkGlaze(
      mossPigment.K,
      mossPigment.S,
      float(0.86).add(cellularFbm.mul(0.45)),
      mix(mossPigment.rInf, malachitePigment.rInf, float(0.36))
    );
    const chartreuseTips = mix(malachitePigment.rInf, sporePigment.rInf, float(0.50));
    const hummockCreviceShade = mix(float(0.58), float(1.12), smoothstep(0.22, 0.82, hummockPillow));
    const matureVelvetRgb = mix(
      deepVelvetBase,
      chartreuseTips,
      smoothstep(0.42, 0.84, cellularFbm.mul(0.55).add(microSporeNoise.mul(0.45))).mul(0.48)
    ).mul(hummockCreviceShade);

    const surfaceRgb = mix(stage2Rgb, matureVelvetRgb, smoothstep(0.06, 0.58, localMoss));

    // High-gloss mirror Fresnel reflection on bare marble & wet liquid pools (fading as it molds & blooms!)
    const viewDir = normalize(cameraPosition.sub(wPos));
    const nDotV = clamp(dot(wNorm, viewDir), 0.0, 1.0);
    const grazing = float(1.0).sub(nDotV);
    const fresnel = float(0.045).add(float(0.48).mul(grazing.mul(grazing).mul(grazing)));

    const subsurfaceCalcite = beerLambertTransmittance(
      kaolinPigment.extinction,
      float(1.05).sub(nDotV.mul(0.65))
    )
      .mul(calcitePigment.rInf)
      .mul(marbleLightness)
      .mul(float(1.0).sub(localMold.mul(0.78)))
      .mul(0.08);

    // Glistening wet meniscus rim + concentric ripple specular highlights on pure liquid pools
    const rippleHighlight = clamp(rippleWave.mul(0.5).add(0.5), 0.0, 1.0).mul(pureLiquidFactor);
    const liquidSpecular = mix(
      calcitePigment.rInf,
      celadonPigment.rInf,
      liquidMeniscusLip.add(rippleHighlight.mul(0.5)).clamp(0.0, 1.0)
    ).mul(
      fresnel.mul(
        float(0.38)
          .add(pureLiquidFactor.mul(0.82))
          .add(rippleHighlight.mul(0.45))
          .sub(localMoss.mul(0.34))
      )
    );

    stoneMaterial.colorNode = surfaceRgb.add(subsurfaceCalcite).add(liquidSpecular);

    // Roughness progression:
    // Bare polished marble (`0.11`) -> Glistening wet liquid pool (`0.025`) ->
    // Coagulating olive-ochre mold (`0.56`) -> Plush matte velvet moss (`0.86`)
    const wetRoughness = mix(
      float(0.11).add(sharpGreyVein.mul(0.04)),
      float(0.025),
      smoothstep(0.04, 0.24, localLiquid)
    );
    const moldedRoughness = mix(wetRoughness, float(0.56), smoothstep(0.10, 0.85, localMold));
    stoneMaterial.roughnessNode = clamp(
      mix(moldedRoughness, float(0.86), smoothstep(0.12, 0.72, localMoss)),
      0.025,
      0.88
    );
  }

  const pillarGroup = new THREE.Group();
  pillarGroup.name = 'PolishedMarblePillars';
  const pillarMeshes: THREE.Mesh[] = [];

  for (let i = 0; i < PILLARS.length; i++) {
    const geom = buildPolishedPillarGeometry(PILLARS[i]);
    const mesh = new THREE.Mesh(geom, stoneMaterial);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    pillarGroup.add(mesh);
    pillarMeshes.push(mesh);
  }
  stage.specimenGroup.add(pillarGroup);

  // -------------------------------------------------------------------------
  // 3. Seed 22,400 High-Precision Multi-Whorl Bryophyte Tufts Rooted on the 3D Hummocks
  // -------------------------------------------------------------------------
  const florets: MicroFloretNode[] = [];

  for (let pIdx = 0; pIdx < PILLARS.length; pIdx++) {
    const p = PILLARS[pIdx];
    // 1. High-density organic hummock carpet across the top basin & chamfered rim (~2,550 tufts/pillar)
    const topFloretCount = 2550;
    for (let j = 0; j < topFloretCount; j++) {
      const frac = (j + 0.5) / topFloretCount;
      const rJitter = (rand() - 0.5) * 0.018;
      const rDisk = Math.max(0.0, Math.min(0.99, Math.sqrt(frac) * 0.985 + rJitter));
      const theta = j * 2.399963229728653 + p.veinSeed + (rand() - 0.5) * 0.04;

      const lx = Math.cos(theta) * rDisk * p.radius;
      const lz = Math.sin(theta) * rDisk * p.radius;
      const wx = p.cx + lx;
      const wz = p.cz + lz;
      const wy = evalPillarTopY(p, lx, lz) - 0.0005;

      const gx = Math.max(0, Math.min(GRID_RES - 1, Math.round(worldToGridX(wx))));
      const gz = Math.max(0, Math.min(GRID_RES - 1, Math.round(worldToGridZ(wz))));
      const gridIdx = gz * GRID_RES + gx;

      const rimTilt = rDisk > 0.86 ? ((rDisk - 0.86) / 0.14) * 0.68 : -0.06 * rDisk;
      const baseNormal = new THREE.Vector3(
        Math.cos(theta) * rimTilt + (rand() - 0.5) * 0.09,
        1.0,
        Math.sin(theta) * rimTilt + (rand() - 0.5) * 0.09
      ).normalize();

      const baseQuat = new THREE.Quaternion().setFromUnitVectors(_up, baseNormal);
      const yawQ = new THREE.Quaternion().setFromAxisAngle(baseNormal, rand() * Math.PI * 2);
      baseQuat.premultiply(yawQ);

      const hummockProfile = evalHummockProfileCpu(wx, wy, wz, p.veinSeed);
      const fbmDetail = fbm3cpu(wx * 22.0 + p.veinSeed, wy * 22.0, wz * 22.0);
      const filigreeThreshold = (1.0 - fbmDetail) * 0.18 + (1.0 - hummockProfile) * 0.10;
      const maxScale = (0.72 + hummockProfile * 0.68 + fbmDetail * 0.30) * (0.88 + rand() * 0.24);
      const hummockLift = 0.0112 * hummockProfile;

      const cellMoss = gridMoss[gridIdx];
      const initBloom = Math.max(
        0,
        Math.min(1, (cellMoss - filigreeThreshold) / Math.max(0.15, 1.0 - filigreeThreshold))
      );

      florets.push({
        pillarId: p.id,
        isWallNode: false,
        anchorPos: new THREE.Vector3(wx, wy, wz),
        baseNormal,
        baseQuat,
        maxScale,
        hummockLift,
        gridIdx,
        wallDropDist: 0,
        filigreeThreshold,
        bloom: initBloom,
        bendX: 0,
        bendZ: 0,
        velBendX: 0,
        velBendZ: 0,
        seedPhase: rand() * Math.PI * 2,
      });
    }

    // 2. Vertical Wall Rivulet Moss Tufts (~650 tufts/pillar clustered along natural overflow tendrils!)
    const wallFloretCount = 650;
    for (let w = 0; w < wallFloretCount; w++) {
      const theta = w * 2.399963229728653 + p.veinSeed * 2.1;
      const wxWall = p.cx + Math.cos(theta) * (p.radius + 0.0015);
      const wzWall = p.cz + Math.sin(theta) * (p.radius + 0.0015);
      const fbmStream = fbm3cpu(wxWall * 12.0 + p.veinSeed, w * 0.015, wzWall * 12.0 - p.veinSeed);
      const maxDrop = 0.04 + Math.pow(fbmStream, 1.8) * 0.32;
      const dropFrac = Math.pow(rand(), 1.35);
      const wallDropDist = dropFrac * maxDrop;

      const rimLx = Math.cos(theta) * p.radius * 0.94;
      const rimLz = Math.sin(theta) * p.radius * 0.94;
      const rimWx = p.cx + rimLx;
      const rimWz = p.cz + rimLz;

      const gx = Math.max(0, Math.min(GRID_RES - 1, Math.round(worldToGridX(rimWx))));
      const gz = Math.max(0, Math.min(GRID_RES - 1, Math.round(worldToGridZ(rimWz))));
      const gridIdx = gz * GRID_RES + gx;

      const wy = evalPillarTopY(p, rimLx, rimLz) - 0.010 - wallDropDist;

      const baseNormal = new THREE.Vector3(
        Math.cos(theta) * 0.78,
        0.62 + (1.0 - dropFrac) * 0.25,
        Math.sin(theta) * 0.78
      ).normalize();

      const baseQuat = new THREE.Quaternion().setFromUnitVectors(_up, baseNormal);
      const yawQ = new THREE.Quaternion().setFromAxisAngle(baseNormal, rand() * Math.PI * 2);
      baseQuat.premultiply(yawQ);

      const filigreeThreshold = 0.12 + wallDropDist * 1.35 + (1.0 - fbmStream) * 0.18;
      const maxScale = (0.66 + fbmStream * 0.56) * (1.0 - dropFrac * 0.36);
      const cellMoss = gridMoss[gridIdx] * Math.min(1.0, gridCascade[gridIdx] * 1.4);
      const initBloom = Math.max(
        0,
        Math.min(1, (cellMoss - filigreeThreshold) / Math.max(0.15, 1.0 - filigreeThreshold))
      );

      florets.push({
        pillarId: p.id,
        isWallNode: true,
        anchorPos: new THREE.Vector3(wxWall, wy, wzWall),
        baseNormal,
        baseQuat,
        maxScale,
        hummockLift: 0.0035 * fbmStream,
        gridIdx,
        wallDropDist,
        filigreeThreshold,
        bloom: initBloom,
        bendX: 0,
        bendZ: 0,
        velBendX: 0,
        velBendZ: 0,
        seedPhase: rand() * Math.PI * 2,
      });
    }
  }

  const floretCount = florets.length;
  const floretGeom = createMicroBryophyteFloretGeometry();
  const floretStateArray = new Float32Array(floretCount * 4);
  const floretStateAttr = new THREE.InstancedBufferAttribute(floretStateArray, 4);
  floretStateAttr.setUsage(THREE.DynamicDrawUsage);
  floretGeom.setAttribute('aFloretState', floretStateAttr);

  const floretMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.58,
    metalness: 0.01,
    side: THREE.DoubleSide,
  });

  {
    const lPos = positionLocal;
    const uvCoord = uv();
    const frondMeta = attribute('aFrondMeta', 'vec4'); // [dirX, dirZ, bladeSign, tierNorm]
    const floretState = attribute('aFloretState', 'vec4'); // [bloom, bendX, bendZ, seedPhase]

    const bloom = clamp(floretState.x, 0.0, 1.0);
    const unbloomed = float(1.0).sub(bloom);
    const bendX = floretState.y;
    const bendZ = floretState.z;
    const seedPhase = floretState.w;
    const hNorm = uvCoord.y;
    const hPow = hNorm.mul(hNorm);

    const radialDir = vec3(frondMeta.x, float(0.0), frondMeta.y);
    const budPinch = radialDir.negate().mul(unbloomed.mul(hNorm).mul(0.008));
    const rosetteSpread = radialDir.mul(bloom.mul(hNorm).mul(0.006));

    const shimmerAngle = stage.uTime.mul(1.5).add(seedPhase).add(frondMeta.w.mul(3.14));
    const microSwayX = sin(shimmerAngle).mul(0.0010).mul(bloom);
    const microSwayZ = cos(shimmerAngle.mul(0.83)).mul(0.0010).mul(bloom);

    const combOffset = vec3(
      bendX.add(microSwayX).mul(hPow),
      bendX.abs().add(bendZ.abs()).negate().mul(hPow).mul(0.16),
      bendZ.add(microSwayZ).mul(hPow)
    );

    const wPos = positionWorld;
    const wNorm = normalize(normalWorld);
    const microCell = mineralToothFbm(positionLocal.add(seedPhase), float(48.0));
    const microHummockLift = vec3(
      float(0.0),
      microCell.sub(0.45).mul(0.0038).mul(bloom),
      float(0.0)
    );

    floretMaterial.positionNode = lPos
      .add(budPinch)
      .add(rosetteSpread)
      .add(combOffset)
      .add(microHummockLift);

    const leafHash = hash31Tsl(vec3(seedPhase, frondMeta.w.mul(19.7), 2.3));
    const costaRidge = exp(uvCoord.x.sub(0.5).abs().negate().mul(12.0)).mul(
      smoothstep(0.15, 0.85, hNorm)
    );

    // Emerging buds (`bloom < 0.35`) carry the warm olive-ochre-sulfur tone of the molding biofilm
    const moldBudRgb = mix(
      mix(umberPigment.rInf, ochrePigment.rInf, float(0.58)),
      mix(sporePigment.rInf, celadonPigment.rInf, float(0.45)),
      hNorm.mul(0.82)
    );

    // Deep rhizoid & lower-stem Kubelka-Munk shadow
    const deepCushionBase = kubelkaMunkGlaze(
      mossPigment.K,
      mossPigment.S,
      clamp(float(0.78).sub(hNorm.mul(0.42)).add(leafHash.mul(0.16)), 0.14, 0.94),
      mix(mossPigment.rInf, malachitePigment.rInf, float(0.42))
    );
    const vibrantChlorophyllBody = mix(
      deepCushionBase,
      mix(malachitePigment.rInf, celadonPigment.rInf, float(0.36)),
      smoothstep(0.16, 0.74, hNorm).mul(0.52)
    );

    // Sunlit chartreuse-gold acuminate needle tips + central costa midrib gleam
    const chartreuseTips = mix(
      malachitePigment.rInf,
      mix(sporePigment.rInf, ochrePigment.rInf, leafHash.mul(0.32)),
      float(0.58).add(leafHash.mul(0.22))
    );
    const matureWithTips = mix(
      vibrantChlorophyllBody,
      chartreuseTips,
      smoothstep(0.48, 0.95, hNorm)
        .mul(smoothstep(0.22, 0.86, bloom))
        .mul(float(0.56).add(leafHash.mul(0.24)))
        .add(costaRidge.mul(0.22))
        .clamp(0.0, 1.0)
    );

    const filamentColor = mix(moldBudRgb, matureWithTips, smoothstep(0.06, 0.48, bloom));

    // Subsurface chlorophyll translucency + velvet rim sheen under studio softbox lights
    const viewDir = normalize(cameraPosition.sub(wPos));
    const grazing = float(1.0).sub(clamp(dot(wNorm, viewDir).abs(), 0.0, 1.0));
    const transmitted = beerLambertTransmittance(
      mossPigment.extinction,
      clamp(float(0.76).sub(hNorm.mul(0.54)).sub(grazing.mul(0.34)), 0.05, 0.95)
    )
      .mul(mix(malachitePigment.rInf, sporePigment.rInf, float(0.54)))
      .mul(bloom)
      .mul(float(0.22).add(grazing.mul(0.38)));

    const rootAO = mix(float(0.68), float(1.06), smoothstep(0.0, 0.56, hNorm));
    floretMaterial.colorNode = filamentColor.add(transmitted).mul(rootAO);
    floretMaterial.roughnessNode = mix(float(0.32), float(0.68), bloom);
  }

  const floretInstancedMesh = new THREE.InstancedMesh(floretGeom, floretMaterial, floretCount);
  floretInstancedMesh.castShadow = true;
  floretInstancedMesh.receiveShadow = true;
  stage.specimenGroup.add(floretInstancedMesh);

  for (let i = 0; i < floretCount; i++) {
    const fl = florets[i];
    const s = fl.bloom > 0.008 ? fl.maxScale * (0.24 + fl.bloom * 0.76) : 0.0001;
    _scale.set(s, s * (0.72 + fl.bloom * 0.38), s);
    _pos.copy(fl.anchorPos).addScaledVector(fl.baseNormal, fl.hummockLift * fl.bloom);
    _mat4.compose(_pos, fl.baseQuat, _scale);
    floretInstancedMesh.setMatrixAt(i, _mat4);

    floretStateArray[i * 4 + 0] = fl.bloom;
    floretStateArray[i * 4 + 1] = 0;
    floretStateArray[i * 4 + 2] = 0;
    floretStateArray[i * 4 + 3] = fl.seedPhase;
  }
  floretInstancedMesh.instanceMatrix.needsUpdate = true;
  floretStateAttr.needsUpdate = true;

  // -------------------------------------------------------------------------
  // 4. Hair-Fine Micro-Pin Sporophyte Capsules (~1,250 Pins on Mature Cushions)
  // -------------------------------------------------------------------------
  const pinSporophytes: PinSporophyteNode[] = [];
  for (let i = 0; i < floretCount; i += 14) {
    const fl = florets[i];
    if (fl.isWallNode) continue;
    const initG = Math.max(0, (fl.bloom - 0.62) / 0.38);
    pinSporophytes.push({
      floretIndex: i,
      baseQuat: fl.baseQuat.clone(),
      maxScale: 0.76 + rand() * 0.44,
      growth: initG,
    });
  }

  const pinCount = pinSporophytes.length;
  const pinGeom = createMicroPinSporophyteGeometry();
  const pinMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.32,
    metalness: 0.04,
  });

  {
    const uvCoord = uv();
    const tStalk = uvCoord.y;
    const isCapsule = smoothstep(0.72, 0.80, tStalk);
    const setaColor = kubelkaMunkGlaze(
      rustPigment.K,
      rustPigment.S,
      float(0.55),
      siennaPigment.rInf
    );
    const capsuleColor = mix(ochrePigment.rInf, sporePigment.rInf, smoothstep(0.75, 0.96, tStalk));
    pinMaterial.colorNode = mix(setaColor, capsuleColor, isCapsule);
    pinMaterial.emissiveNode = sporePigment.rInf.mul(isCapsule).mul(0.18);
  }

  const pinInstancedMesh = new THREE.InstancedMesh(pinGeom, pinMaterial, pinCount);
  pinInstancedMesh.castShadow = true;
  stage.specimenGroup.add(pinInstancedMesh);

  for (let i = 0; i < pinCount; i++) {
    const pin = pinSporophytes[i];
    const hostFl = florets[pin.floretIndex];
    const s = pin.maxScale * pin.growth;
    _scale.set(s, s, s);
    _pos.copy(hostFl.anchorPos).addScaledVector(hostFl.baseNormal, hostFl.hummockLift * hostFl.bloom);
    _mat4.compose(_pos, pin.baseQuat, _scale);
    pinInstancedMesh.setMatrixAt(i, _mat4);
  }
  pinInstancedMesh.instanceMatrix.needsUpdate = true;

  // -------------------------------------------------------------------------
  // 5. Glistening Translucent Emerald Botanical Liquid Material (Pour Stream & Aerial Droplets)
  // -------------------------------------------------------------------------
  const liquidMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.025,
    metalness: 0.02,
  });

  {
    const wPos = positionWorld;
    const wNorm = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(wPos));
    const nDotV = clamp(dot(wNorm, viewDir), 0.0, 1.0);
    const grazing = float(1.0).sub(nDotV);

    const ripple = sin(wPos.x.mul(36.0).add(wPos.y.mul(28.0)).sub(stage.uTime.mul(10.0)));
    const opticalChord = float(0.32).add(nDotV.mul(0.68)).add(ripple.mul(0.05));
    const beerTrans = beerLambertTransmittance(malachitePigment.extinction, opticalChord);

    const deepEmerald = kubelkaMunkGlaze(
      mossPigment.K,
      malachitePigment.S,
      float(0.52),
      malachitePigment.rInf
    );
    const brightJade = mix(celadonPigment.rInf, sporePigment.rInf, float(0.42));
    const liquidRgb = mix(deepEmerald, brightJade, beerTrans.mul(0.82));

    const fresnelSpec = calcitePigment.rInf.mul(grazing.mul(grazing).mul(0.75));
    liquidMaterial.colorNode = liquidRgb.add(fresnelSpec);
    liquidMaterial.emissiveNode = mix(malachitePigment.rInf, celadonPigment.rInf, float(0.45)).mul(
      0.22
    );
  }

  // Dynamic Continuous Pour Stream Mesh (Descends gracefully from high above when pouring)
  const pourStreamGeom = createDynamicPourStreamGeometry();
  const pourStreamMesh = new THREE.Mesh(pourStreamGeom, liquidMaterial);
  pourStreamMesh.castShadow = true;
  pourStreamMesh.visible = false;
  stage.specimenGroup.add(pourStreamMesh);

  // 180 Pure Aerial & Rim-Cascade Droplets (Strictly spherical/teardrop falling & splashing beads—
  // zero flat seed pellets on the horizontal marble surface!)
  const DROPLET_COUNT = 180;
  const droplets: AerialDroplet[] = [];

  for (let i = 0; i < DROPLET_COUNT; i++) {
    droplets.push({
      active: false,
      mode: 'falling',
      pillarId: 0,
      x: 0,
      y: -10,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      radius: 0.006,
      age: 0,
      maxLife: 0.9,
    });
  }

  let nextDropletIdx = 0;
  function spawnAerialDroplet(
    mode: DropletMode,
    srcX: number,
    srcY: number,
    srcZ: number,
    vx: number,
    vy: number,
    vz: number,
    pillarId: number,
    radius: number,
    maxLife: number
  ): void {
    const d = droplets[nextDropletIdx];
    nextDropletIdx = (nextDropletIdx + 1) % DROPLET_COUNT;
    d.active = true;
    d.mode = mode;
    d.pillarId = pillarId;
    d.x = srcX;
    d.y = srcY;
    d.z = srcZ;
    d.vx = vx;
    d.vy = vy;
    d.vz = vz;
    d.radius = radius;
    d.age = 0;
    d.maxLife = maxLife;
  }

  const dropletGeom = new THREE.SphereGeometry(1.0, 10, 8);
  const dropletInstancedMesh = new THREE.InstancedMesh(dropletGeom, liquidMaterial, DROPLET_COUNT);
  stage.specimenGroup.add(dropletInstancedMesh);

  // -------------------------------------------------------------------------
  // 6. Click-to-Pour, Click-More-to-Pour-More & Orbit/Zoom Exploration State Machine
  // -------------------------------------------------------------------------
  const pourTargetLocal = new THREE.Vector3(0.24, PILLARS[1].topY, 0.02);
  const pourEmitterLocal = new THREE.Vector3(0.24, PILLARS[1].topY + 1.18, 0.02);
  const hoverTargetLocal = new THREE.Vector3(0.24, PILLARS[1].topY, 0.02);
  let activePourPillarId = 1;
  let pourReservoir = 1.45; // Opening pour on Pillar 1; each user click adds +0.95 (stacking up to 3.2!)
  let pourStreamStrength = 0.0;

  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTimeMs = 0;
  let pointerMaxDragPx = 0;
  let isPointerPressed = false;
  let isHoldPourLocked = false;
  let downHitPillarId = -1;
  const downHitLocalPos = new THREE.Vector3();

  function raycastPillarAtClientXY(clientX: number, clientY: number): {
    pillar: MarblePillarSpec;
    localPoint: THREE.Vector3;
    directHit: boolean;
  } {
    const rect = stage.shell.canvas.getBoundingClientRect();
    const nx = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    const ny = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
    _ndcScratch.set(nx, ny);

    stage.specimenGroup.updateMatrixWorld();
    _invGroupMatrix.copy(stage.specimenGroup.matrixWorld).invert();

    _raycaster.setFromCamera(_ndcScratch, stage.camera);
    const hits = _raycaster.intersectObjects(pillarMeshes, false);

    if (hits.length > 0) {
      _rayHitPos.copy(hits[0].point).applyMatrix4(_invGroupMatrix);
      const support = findSupportingPillar(_rayHitPos.x, _rayHitPos.z, _rayHitPos.y + 0.25);
      const p = support.pillar ?? PILLARS[0];
      const lx = _rayHitPos.x - p.cx;
      const lz = _rayHitPos.z - p.cz;
      const r = Math.hypot(lx, lz);
      const clampedR = Math.min(p.radius * 0.86, r);
      const ang = Math.atan2(lz, lx);
      const tx = p.cx + Math.cos(ang) * clampedR;
      const tz = p.cz + Math.sin(ang) * clampedR;
      const ty = evalPillarTopY(p, tx - p.cx, tz - p.cz);
      return {
        pillar: p,
        localPoint: _pos.set(tx, ty, tz).clone(),
        directHit: true,
      };
    }

    _localPointer.copy(stage.pointer.worldPos).applyMatrix4(_invGroupMatrix);
    let nearestP = PILLARS[0];
    let minDist = 999;
    for (let i = 0; i < PILLARS.length; i++) {
      const p = PILLARS[i];
      const d = Math.hypot(_localPointer.x - p.cx, _localPointer.z - p.cz) - p.radius;
      if (d < minDist) {
        minDist = d;
        nearestP = p;
      }
    }
    const clampedR = Math.min(
      nearestP.radius * 0.84,
      Math.hypot(_localPointer.x - nearestP.cx, _localPointer.z - nearestP.cz)
    );
    const ang = Math.atan2(_localPointer.z - nearestP.cz, _localPointer.x - nearestP.cx);
    const tx = nearestP.cx + Math.cos(ang) * clampedR;
    const tz = nearestP.cz + Math.sin(ang) * clampedR;
    const ty = evalPillarTopY(nearestP, tx - nearestP.cx, tz - nearestP.cz);
    return {
      pillar: nearestP,
      localPoint: _pos.set(tx, ty, tz).clone(),
      directHit: false,
    };
  }

  const onCanvasPointerDown = (ev: PointerEvent): void => {
    if (ev.button !== 0) return;
    isPointerPressed = true;
    isHoldPourLocked = false;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    pointerDownTimeMs = performance.now();
    pointerMaxDragPx = 0;

    const hit = raycastPillarAtClientXY(ev.clientX, ev.clientY);
    downHitPillarId = hit.pillar.id;
    downHitLocalPos.copy(hit.localPoint);

    if (hit.directHit) {
      activePourPillarId = hit.pillar.id;
      pourTargetLocal.copy(hit.localPoint);
      pourReservoir = Math.max(pourReservoir, 0.42);
    }
  };

  const onCanvasPointerMove = (ev: PointerEvent): void => {
    const hit = raycastPillarAtClientXY(ev.clientX, ev.clientY);
    hoverTargetLocal.copy(hit.localPoint);

    if (!isPointerPressed) return;
    const distPx = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    if (distPx > pointerMaxDragPx) {
      pointerMaxDragPx = distPx;
    }

    if (isHoldPourLocked) {
      activePourPillarId = hit.pillar.id;
      pourTargetLocal.lerp(hit.localPoint, 0.35);
    }
  };

  const onCanvasPointerUp = (): void => {
    if (!isPointerPressed) return;
    const holdDurationMs = performance.now() - pointerDownTimeMs;

    // A click (`pointerMaxDragPx < 8px`) pours green liquid; clicking repeatedly ("clicking pours more")
    // stacks `pourReservoir` up to 3.2 for a heavier pour and rapid multi-pillar cascade!
    if (pointerMaxDragPx < 8 && downHitPillarId !== -1) {
      activePourPillarId = downHitPillarId;
      pourTargetLocal.copy(downHitLocalPos);
      pourReservoir = Math.min(3.2, pourReservoir + (holdDurationMs < 260 ? 1.05 : 0.65));
    }

    isPointerPressed = false;
    isHoldPourLocked = false;
    stage.controls.enabled = true;
  };

  stage.shell.canvas.addEventListener('pointerdown', onCanvasPointerDown, { passive: true });
  stage.shell.canvas.addEventListener('pointermove', onCanvasPointerMove, { passive: true });
  window.addEventListener('pointerup', onCanvasPointerUp, { passive: true });
  stage.shell.registerDispose(() => {
    stage.shell.canvas.removeEventListener('pointerdown', onCanvasPointerDown);
    stage.shell.canvas.removeEventListener('pointermove', onCanvasPointerMove);
    window.removeEventListener('pointerup', onCanvasPointerUp);
  });

  // -------------------------------------------------------------------------
  // 7. Real-Time Simulation Loop:
  //    Click/Hold Pouring -> Basin Pooling & Rim Cascade -> Mold Color Shift -> Precision Moss Bloom
  // -------------------------------------------------------------------------
  stage.onTick(({ dtSec, elapsedSec, pointer }) => {
    const dt = Math.min(0.04, Math.max(0.001, dtSec));

    stage.specimenGroup.updateMatrixWorld();
    _invGroupMatrix.copy(stage.specimenGroup.matrixWorld).invert();
    _invGroupQuat.copy(stage.specimenGroup.quaternion).invert();
    _localVel.copy(pointer.velocity).applyQuaternion(_invGroupQuat);

    if (
      isPointerPressed &&
      !isHoldPourLocked &&
      pointerMaxDragPx < 8 &&
      performance.now() - pointerDownTimeMs > 190
    ) {
      isHoldPourLocked = true;
      stage.controls.enabled = false;
      activePourPillarId = downHitPillarId;
      pourTargetLocal.copy(downHitLocalPos);
    }

    if (isHoldPourLocked) {
      pourReservoir = Math.min(3.2, Math.max(pourReservoir, 1.35) + dt * 1.45);
    }

    const isActivelyPouring = pourReservoir > 0.04;
    if (isActivelyPouring) {
      const drainRate = 0.78 + Math.min(1.2, pourReservoir * 0.35);
      pourReservoir = Math.max(0.0, pourReservoir - dt * drainRate);
    }

    const targetStreamStrength = isActivelyPouring
      ? Math.min(1.60, 0.65 + Math.min(2.2, pourReservoir) * 0.42)
      : 0.0;
    const streamResponse = isActivelyPouring ? 15.0 : 4.5;
    pourStreamStrength += (targetStreamStrength - pourStreamStrength) * (1 - Math.exp(-dt * streamResponse));

    // Sync pour impact coordinates & ripple strength to the stone shader
    uPourImpactLocal.value.set(
      pourTargetLocal.x,
      pourTargetLocal.y,
      pourTargetLocal.z,
      Math.min(1.0, pourStreamStrength)
    );

    pourEmitterLocal.set(
      pourTargetLocal.x,
      pourTargetLocal.y + 1.18,
      pourTargetLocal.z
    );

    if (pourStreamStrength > 0.025) {
      pourStreamMesh.visible = true;
      updateDynamicPourStreamMesh(
        pourStreamGeom,
        pourEmitterLocal,
        pourTargetLocal,
        _localVel.x * 0.06,
        _localVel.z * 0.06,
        elapsedSec,
        pourStreamStrength
      );

      // Deposit fresh translucent green liquid (`initialAgeBoost = 0` so it starts 100% pure liquid!)
      const pourRadius = 0.12 + Math.min(0.11, pourStreamStrength * 0.052);
      depositGreenLiquid(
        pourTargetLocal.x,
        pourTargetLocal.z,
        pourRadius,
        dt * 3.1 * pourStreamStrength,
        activePourPillarId,
        0.0,
        true
      );

      // Spawn fine spherical splash beads bouncing lightly at the impact meniscus
      if (rand() < 0.75) {
        const ang = rand() * Math.PI * 2;
        const outSpd = (0.14 + rand() * 0.18) * Math.min(1.3, pourStreamStrength);
        spawnAerialDroplet(
          'splash',
          pourTargetLocal.x + Math.cos(ang) * 0.015,
          pourTargetLocal.y + 0.008,
          pourTargetLocal.z + Math.sin(ang) * 0.015,
          Math.cos(ang) * outSpd,
          0.22 + rand() * 0.24,
          Math.sin(ang) * outSpd,
          activePourPillarId,
          0.0045 + rand() * 0.0025,
          0.28 + rand() * 0.22
        );
      }

      // If pouring heavily (`pourStreamStrength > 0.85`), also trigger an active overflow rivulet
      // down this pillar's lowest rim lip onto its downstream neighbor!
      const pActive = PILLARS[activePourPillarId];
      if (pourStreamStrength > 0.75 && pActive.overflowTargetId !== -1) {
        const rimAng = pActive.overflowRimAngle + (rand() - 0.5) * 0.36;
        const rimX = pActive.cx + Math.cos(rimAng) * pActive.radius * 0.92;
        const rimZ = pActive.cz + Math.sin(rimAng) * pActive.radius * 0.92;
        depositGreenLiquid(rimX, rimZ, 0.09, dt * 1.85 * pourStreamStrength, pActive.id, 0.0, true);

        if (rand() < 0.55) {
          const rimY = evalPillarTopY(pActive, rimX - pActive.cx, rimZ - pActive.cz);
          spawnAerialDroplet(
            'wall-drip',
            pActive.cx + Math.cos(rimAng) * (pActive.radius + 0.006),
            rimY - 0.015,
            pActive.cz + Math.sin(rimAng) * (pActive.radius + 0.006),
            Math.cos(rimAng) * 0.12,
            -0.18,
            Math.sin(rimAng) * 0.12,
            pActive.id,
            0.006 + rand() * 0.0028,
            0.85 + rand() * 0.45
          );
        }
      }
    } else {
      pourStreamMesh.visible = false;
    }

    // Update Aerial & Rim-Cascade Droplets
    const gravity = 3.8;
    for (let i = 0; i < DROPLET_COUNT; i++) {
      const d = droplets[i];
      if (!d.active) {
        _scale.set(0.0001, 0.0001, 0.0001);
        _pos.set(0, -10, 0);
        _mat4.compose(_pos, _quat, _scale);
        dropletInstancedMesh.setMatrixAt(i, _mat4);
        continue;
      }

      d.age += dt;
      if (d.age >= d.maxLife || d.y < -1.16) {
        d.active = false;
        _scale.set(0.0001, 0.0001, 0.0001);
        _pos.set(0, -10, 0);
        _mat4.compose(_pos, _quat, _scale);
        dropletInstancedMesh.setMatrixAt(i, _mat4);
        continue;
      }

      d.x += d.vx * dt;
      d.y += d.vy * dt;
      d.z += d.vz * dt;
      d.vy -= gravity * dt;

      if (d.mode === 'wall-drip') {
        const hostP = PILLARS[d.pillarId];
        const dropBelowTop = hostP.topY - d.y;
        if (dropBelowTop < 0.22) {
          const lx = d.x - hostP.cx;
          const lz = d.z - hostP.cz;
          const r = Math.hypot(lx, lz) + 1e-4;
          const wallR = hostP.radius + 0.006;
          d.x = hostP.cx + (lx / r) * (r * 0.65 + wallR * 0.35);
          d.z = hostP.cz + (lz / r) * (r * 0.65 + wallR * 0.35);
        }

        // Check if this cascading droplet lands on the lower neighbor pillar!
        const lowerSupport = findSupportingPillar(d.x, d.z, hostP.topY - 0.10);
        if (
          lowerSupport.pillar &&
          lowerSupport.pillar.id !== hostP.id &&
          d.y <= lowerSupport.surfaceY + 0.012
        ) {
          const nextP = lowerSupport.pillar;
          // Deposit a fresh pool of green liquid on the lower pillar (starts at age 0 -> pools -> molds -> blooms!)
          depositGreenLiquid(d.x, d.z, 0.11, 0.32, nextP.id, 0.0, false);
          d.active = false;
          _scale.set(0.0001, 0.0001, 0.0001);
          _pos.set(0, -10, 0);
          _mat4.compose(_pos, _quat, _scale);
          dropletInstancedMesh.setMatrixAt(i, _mat4);
          continue;
        }
      } else if (d.mode === 'splash' && d.vy < 0) {
        const support = findSupportingPillar(d.x, d.z, d.y + 0.05);
        if (support.pillar && d.y <= support.surfaceY + 0.006) {
          depositGreenLiquid(d.x, d.z, 0.065, 0.14, support.pillar.id, 0.0, false);
          d.active = false;
          _scale.set(0.0001, 0.0001, 0.0001);
          _pos.set(0, -10, 0);
          _mat4.compose(_pos, _quat, _scale);
          dropletInstancedMesh.setMatrixAt(i, _mat4);
          continue;
        }
      }

      const speed = Math.hypot(d.vx, d.vy, d.vz) || 1;
      const lifeEnv = Math.sin(Math.min(1, d.age / d.maxLife) * Math.PI);
      const r = d.radius * (0.45 + 0.55 * lifeEnv);
      _dir.set(d.vx / speed, d.vy / speed, d.vz / speed);
      _quat.setFromUnitVectors(_up, _dir);
      _pos.set(d.x, d.y, d.z);
      const stretch = 1.0 + Math.min(1.8, speed * 0.65);
      _scale.set(r * 0.85, r * stretch, r * 0.85);
      _mat4.compose(_pos, _quat, _scale);
      dropletInstancedMesh.setMatrixAt(i, _mat4);
    }
    dropletInstancedMesh.instanceMatrix.needsUpdate = true;

    // -----------------------------------------------------------------------
    // 256x256 Surface Advection + 4-Stage Chronological Lifecycle:
    //   Stage 1 (0.0s .. 1.4s): Pure Glistening Emerald Liquid Pooling & Cascading
    //   Stage 2 (1.4s .. 3.2s): Subtle Color Shift into Olive-Ochre-Umber Mold Biofilm
    //   Stage 3 (3.0s .. 5.6s): Precision 3D Bryophyte Moss Blooming from the Molded Bed
    // -----------------------------------------------------------------------
    const diffRate = 0.22;
    for (let gz = 1; gz < GRID_RES - 1; gz++) {
      const rowOffset = gz * GRID_RES;
      for (let gx = 1; gx < GRID_RES - 1; gx++) {
        const idx = rowOffset + gx;
        const pId = gridPillarId[idx];
        if (pId === -1) {
          gridLiquidNext[idx] = 0;
          continue;
        }

        const wCenter = gridLiquid[idx];
        let neighborSum = 0;
        let neighborCount = 0;
        const nLeft = idx - 1;
        const nRight = idx + 1;
        const nDown = idx - GRID_RES;
        const nUp = idx + GRID_RES;

        if (gridPillarId[nLeft] === pId) {
          neighborSum += gridLiquid[nLeft];
          neighborCount++;
        }
        if (gridPillarId[nRight] === pId) {
          neighborSum += gridLiquid[nRight];
          neighborCount++;
        }
        if (gridPillarId[nDown] === pId) {
          neighborSum += gridLiquid[nDown];
          neighborCount++;
        }
        if (gridPillarId[nUp] === pId) {
          neighborSum += gridLiquid[nUp];
          neighborCount++;
        }

        const avgNeighbor = neighborCount > 0 ? neighborSum / neighborCount : wCenter;
        const spreadGain =
          avgNeighbor > wCenter
            ? diffRate * (0.62 + gridOrganicBias[idx] * 0.58)
            : diffRate * 0.32;
        const diffused = wCenter + (avgNeighbor - wCenter) * spreadGain;
        gridLiquidNext[idx] = Math.max(0, Math.min(1.0, diffused));
      }
    }

    for (let i = 0; i < gridCellCount; i++) {
      const pId = gridPillarId[i];
      if (pId === -1) continue;
      const liq = gridLiquidNext[i];
      gridLiquid[i] = liq;

      if (liq > 0.08) {
        gridWetAge[i] += dt * (0.65 + liq * 0.55);
        const age = gridWetAge[i];
        const org = gridOrganicBias[i];

        if (gridIsRim[i] === 1 && liq > 0.52) {
          gridCascade[i] = Math.min(1.0, gridCascade[i] + (liq - 0.48) * 0.65 * dt);
        }

        // Stage 2: After pooling as pure green liquid for ~1.3s..1.9s, the liquid begins to MOLD
        // (shifting color toward olive-ochre-umber before any 3D moss shoots appear!)
        const moldOnsetSec = 1.30 + (1.0 - org) * 0.70;
        if (age > moldOnsetSec) {
          const moldTarget = Math.min(1.0, (liq - 0.05) * 1.28);
          const moldRate = (0.38 + org * 0.32) * dt;
          if (gridMold[i] < moldTarget) {
            gridMold[i] = Math.min(moldTarget, gridMold[i] + moldRate);
          }
        }

        // Stage 3: Only after the liquid has visibly molded (`gridMold > 0.40` and `age > 2.75s`)
        // does it bloom into 3D bryophyte moss!
        const mossOnsetSec = 2.75 + (1.0 - org) * 0.95;
        if (age > mossOnsetSec && gridMold[i] > 0.40) {
          const mossTarget = Math.min(1.0, gridMold[i] * (0.80 + org * 0.30));
          const mossRate = (0.34 + org * 0.28) * dt;
          if (gridMoss[i] < mossTarget) {
            gridMoss[i] = Math.min(mossTarget, gridMoss[i] + mossRate);
          }
        }
      }
    }

    syncMoistureDataTexture();

    // Update 22,400 Instanced Multi-Whorl Bryophyte Tufts (lifted dynamically onto the 3D hummock pillows!)
    const combCenter = isActivelyPouring ? pourTargetLocal : hoverTargetLocal;
    const combRadiusSq = 0.32 * 0.32;
    for (let i = 0; i < floretCount; i++) {
      const fl = florets[i];
      const cellMoss = gridMoss[fl.gridIdx];
      const cellCascade = gridCascade[fl.gridIdx];

      const effectiveMoss = fl.isWallNode ? cellMoss * Math.min(1.0, cellCascade * 1.45) : cellMoss;

      const targetBloom = Math.max(
        0.0,
        Math.min(
          1.0,
          (effectiveMoss - fl.filigreeThreshold) / Math.max(0.15, 1.0 - fl.filigreeThreshold)
        )
      );
      fl.bloom += (targetBloom - fl.bloom) * (1 - Math.exp(-dt * 6.0));

      if (fl.bloom > 0.015) {
        const dx = fl.anchorPos.x - combCenter.x;
        const dy = fl.anchorPos.y - combCenter.y;
        const dz = fl.anchorPos.z - combCenter.z;
        const dSq = dx * dx + dy * dy + dz * dz;

        if (dSq < combRadiusSq) {
          const prox = Math.exp(-dSq / (combRadiusSq * 0.35));
          const dist = Math.sqrt(dSq) + 1e-4;
          const splashOut = isActivelyPouring ? 0.028 * pourStreamStrength : 0.008;
          const pushX = _localVel.x * 0.009 + (dx / dist) * splashOut;
          const pushZ = _localVel.z * 0.009 + (dz / dist) * splashOut;
          fl.velBendX += pushX * prox * 26.0 * dt;
          fl.velBendZ += pushZ * prox * 26.0 * dt;
        }
      }

      const kSpring = 34.0;
      fl.velBendX = (fl.velBendX - fl.bendX * kSpring * dt) * Math.exp(-dt * 7.2);
      fl.velBendZ = (fl.velBendZ - fl.bendZ * kSpring * dt) * Math.exp(-dt * 7.2);
      fl.bendX = Math.max(-0.034, Math.min(0.034, fl.bendX + fl.velBendX * dt));
      fl.bendZ = Math.max(-0.034, Math.min(0.034, fl.bendZ + fl.velBendZ * dt));

      const s = fl.bloom > 0.008 ? fl.maxScale * (0.24 + fl.bloom * 0.76) : 0.0001;
      _scale.set(s, s * (0.72 + fl.bloom * 0.38), s);
      _pos.copy(fl.anchorPos).addScaledVector(fl.baseNormal, fl.hummockLift * fl.bloom);
      _mat4.compose(_pos, fl.baseQuat, _scale);
      floretInstancedMesh.setMatrixAt(i, _mat4);

      floretStateArray[i * 4 + 0] = fl.bloom;
      floretStateArray[i * 4 + 1] = fl.bendX;
      floretStateArray[i * 4 + 2] = fl.bendZ;
      floretStateArray[i * 4 + 3] = fl.seedPhase;
    }
    floretInstancedMesh.instanceMatrix.needsUpdate = true;
    floretStateAttr.needsUpdate = true;

    // Update Hair-Fine Micro-Pin Sporophyte Capsules on Mature Cushions
    for (let i = 0; i < pinCount; i++) {
      const pin = pinSporophytes[i];
      const hostFloret = florets[pin.floretIndex];
      const targetG = Math.max(0.0, Math.min(1.0, (hostFloret.bloom - 0.62) / 0.38));
      pin.growth += (targetG - pin.growth) * (1 - Math.exp(-dt * 4.5));

      const s = pin.maxScale * pin.growth;
      _scale.set(s, s, s);
      _pos
        .copy(hostFloret.anchorPos)
        .addScaledVector(hostFloret.baseNormal, hostFloret.hummockLift * hostFloret.bloom);
      _mat4.compose(_pos, pin.baseQuat, _scale);
      pinInstancedMesh.setMatrixAt(i, _mat4);
    }
    pinInstancedMesh.instanceMatrix.needsUpdate = true;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay05Moss();
}
