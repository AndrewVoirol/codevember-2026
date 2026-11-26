import * as THREE from 'three/webgpu';
import { type BloomPetalSpec } from './types';

export const COCOON_BASE_Y = -0.68;
export const PETAL_BASE_Y = -0.60;

const WHORL0_OPACITIES = [0.72, 1.05, 0.84, 1.12, 0.76, 0.98] as const;
const WHORL1_OPACITIES = [0.66, 0.94, 0.74, 1.04, 0.68, 0.90] as const;
const WHORL2_OPACITIES = [0.78, 1.08, 0.86, 1.14, 0.82, 1.02] as const;

function buildThreeWhorlSpecs(): BloomPetalSpec[] {
  const specs: BloomPetalSpec[] = [];

  // Whorl 0 (0..5): 6 outer torn rice-paper sepals — offset +PI/6 so two frame left/right gracefully,
  // opening upward-and-outward (68°-75°) in a poised lotus cup without drooping off-screen
  for (let k = 0; k < 6; k++) {
    const az = ((k + 0.5) / 6) * Math.PI * 2 + (k % 2 === 0 ? 0.03 : -0.03);
    specs.push({
      id: specs.length,
      whorl: 0,
      azimuth: az,
      length: 1.88 + (k % 2) * 0.08,
      maxWidth: 0.62,
      cupDepth: 0.24,
      baseRadius: 0.045,
      closedTilt: -0.02,
      openTilt: 1.18 + (k % 3) * 0.06,
      openDelay: 0.0 + (k % 3) * 0.035,
      tipRecurve: 0.22,
      opacityBias: WHORL0_OPACITIES[k],
      deckleSeed: 1.7 + k * 3.41,
    });
  }

  // Whorl 1 (6..11): 6 middle sheer Bone-Calcite rice-paper petals
  for (let k = 0; k < 6; k++) {
    const az = (k / 6) * Math.PI * 2 + ((k % 2) - 0.5) * 0.04;
    specs.push({
      id: specs.length,
      whorl: 1,
      azimuth: az,
      length: 1.66 + (k % 2) * 0.08,
      maxWidth: 0.50,
      cupDepth: 0.20,
      baseRadius: 0.058,
      closedTilt: -0.08,
      openTilt: 0.84 + (k % 3) * 0.06,
      openDelay: 0.16 + (k % 3) * 0.035,
      tipRecurve: 0.16,
      opacityBias: WHORL1_OPACITIES[k],
      deckleSeed: 22.3 + k * 4.17,
    });
  }

  // Whorl 2 (12..17): 6 inner warm translucent parchment heart petals
  for (let k = 0; k < 6; k++) {
    const az = ((k + 0.45) / 6) * Math.PI * 2 + ((k % 2) - 0.5) * 0.05;
    specs.push({
      id: specs.length,
      whorl: 2,
      azimuth: az,
      length: 1.32 + (k % 2) * 0.08,
      maxWidth: 0.40,
      cupDepth: 0.16,
      baseRadius: 0.070,
      closedTilt: -0.14,
      openTilt: 0.46 + (k % 3) * 0.06,
      openDelay: 0.30 + (k % 3) * 0.035,
      tipRecurve: 0.10,
      opacityBias: WHORL2_OPACITIES[k],
      deckleSeed: 49.1 + k * 3.89,
    });
  }

  return specs;
}

export const BLOOM_PETAL_SPECS: readonly BloomPetalSpec[] = buildThreeWhorlSpecs();

export interface BloomPetalBaseData {
  spec: BloomPetalSpec;
  geometry: THREE.BufferGeometry;
  uSegs: number;
  vSegs: number;
  vertexCount: number;
  /** Local transverse half-width coordinate (lx), spine arc parameter (ly), transverse cup (lz) */
  restLx: Float32Array;
  restLy: Float32Array;
  restCupZ: Float32Array;
  restU: Float32Array;
  restV: Float32Array;
  posAttr: THREE.BufferAttribute;
  normAttr: THREE.BufferAttribute;
}

/**
 * Builds one hand-torn lanceolate-ovate rice-paper petal across Whorl 0..2 with subtle
 * organic width asymmetry, full-perimeter torn edge mask, and outward CCW winding.
 */
export function buildBloomPetalBaseData(spec: BloomPetalSpec): BloomPetalBaseData {
  const uSegs = 14;
  const vSegs = 22;
  const cols = uSegs + 1;
  const rows = vSegs + 1;
  const vertexCount = cols * rows;

  const restLx = new Float32Array(vertexCount);
  const restLy = new Float32Array(vertexCount);
  const restCupZ = new Float32Array(vertexCount);
  const restU = new Float32Array(vertexCount);
  const restV = new Float32Array(vertexCount);

  const pos = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const petalMeta = new Float32Array(vertexCount * 4);
  const edgeMask = new Float32Array(vertexCount);
  const fiberSeed = new Float32Array(vertexCount * 3);

  const cosAz = Math.cos(spec.azimuth);
  const sinAz = Math.sin(spec.azimuth);

  for (let iv = 0; iv <= vSegs; iv++) {
    const v = iv / vSegs;
    const bell = Math.sin(Math.pow(v, 0.74) * Math.PI);
    const widthProfile = Math.pow(Math.max(0, bell), 0.76) * (1.0 - 0.14 * v * v);
    const organicAsym = 1.0 + 0.05 * Math.sin(v * 7.0 + spec.deckleSeed);
    const halfW = spec.maxWidth * (0.07 + 0.93 * widthProfile) * organicAsym;
    const ly = v * spec.length;

    for (let iu = 0; iu <= uSegs; iu++) {
      const u = iu / uSegs;
      const idx = iv * cols + iu;
      const s = u * 2 - 1; // -1..+1

      const lx = s * halfW;
      const cupZ = -(1.0 - s * s) * spec.cupDepth * Math.sin(Math.pow(v, 0.72) * Math.PI);

      restLx[idx] = lx;
      restLy[idx] = ly;
      restCupZ[idx] = cupZ;
      restU[idx] = u;
      restV[idx] = v;

      const rZ = spec.baseRadius + cupZ;
      pos[idx * 3] = lx * cosAz + rZ * sinAz;
      pos[idx * 3 + 1] = PETAL_BASE_Y + ly;
      pos[idx * 3 + 2] = -lx * sinAz + rZ * cosAz;

      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;

      // x=whorlNorm, y=v, z=opacityBias, w=deckleSeed
      petalMeta[idx * 4] = spec.whorl / 2;
      petalMeta[idx * 4 + 1] = v;
      petalMeta[idx * 4 + 2] = spec.opacityBias;
      petalMeta[idx * 4 + 3] = spec.deckleSeed;

      // Full-perimeter torn rice-paper deckle mask across ALL 3 whorls (anchored solid at calyx base)
      const baseHold = smoothstepCpu(0.10, 0.24, v);
      const latProx = smoothstepCpu(0.54, 1.0, Math.abs(s));
      const tipProx = smoothstepCpu(0.68, 1.0, v);
      edgeMask[idx] = baseHold * Math.max(latProx, tipProx);

      // Material-space 3D fiber coords (non-swimming)
      fiberSeed[idx * 3] = s * 0.65 + Math.sin(spec.azimuth) * 0.55 + spec.deckleSeed * 0.09;
      fiberSeed[idx * 3 + 1] = (v - 0.5) * 2.1 + spec.whorl * 0.23;
      fiberSeed[idx * 3 + 2] = Math.cos(spec.azimuth) * 0.55 - spec.deckleSeed * 0.07;
    }
  }

  const indices: number[] = [];
  for (let iv = 0; iv < vSegs; iv++) {
    for (let iu = 0; iu < uSegs; iu++) {
      const a = iv * cols + iu;
      const b = a + 1;
      const c = b + cols;
      const d = a + cols;
      indices.push(a, b, c, a, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(pos, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  const normAttr = new THREE.BufferAttribute(normals, 3);
  normAttr.setUsage(THREE.DynamicDrawUsage);

  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('normal', normAttr);
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aPetalMeta', new THREE.BufferAttribute(petalMeta, 4));
  geometry.setAttribute('aEdgeMask', new THREE.BufferAttribute(edgeMask, 1));
  geometry.setAttribute('aFiberSeed', new THREE.BufferAttribute(fiberSeed, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return {
    spec,
    geometry,
    uSegs,
    vSegs,
    vertexCount,
    restLx,
    restLy,
    restCupZ,
    restU,
    restV,
    posAttr,
    normAttr,
  };
}

/**
 * Delicate curved botanical stamen filaments + glowing teardrop anthers rising from the floral heart.
 */
export function buildStamenCrownGeometry(stamenCount = 18): THREE.BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];
  const stamenMeta: number[] = [];
  const indices: number[] = [];

  const steps = 10;
  const radialSegs = 6;
  const _pt = new THREE.Vector3();
  const _tan = new THREE.Vector3();
  const _n = new THREE.Vector3();
  const _b = new THREE.Vector3();
  const _up = new THREE.Vector3(0, 1, 0);

  for (let s = 0; s < stamenCount; s++) {
    const ring = s < 6 ? 0 : 1;
    const ang = (s / stamenCount) * Math.PI * 2 + ring * 0.31;
    const r0 = ring === 0 ? 0.04 : 0.08;
    const r1 = ring === 0 ? 0.14 : 0.24;
    const h = ring === 0 ? 0.42 + (s % 2) * 0.06 : 0.34 + (s % 3) * 0.05;

    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(Math.cos(ang) * r0, PETAL_BASE_Y + 0.04, Math.sin(ang) * r0),
      new THREE.Vector3(Math.cos(ang) * (r0 + r1) * 0.48, PETAL_BASE_Y + 0.04 + h * 0.55, Math.sin(ang) * (r0 + r1) * 0.48),
      new THREE.Vector3(Math.cos(ang) * r1, PETAL_BASE_Y + 0.04 + h, Math.sin(ang) * r1),
    ]);

    const baseVert = positions.length / 3;
    for (let i = 0; i <= steps; i++) {
      const u = i / steps;
      curve.getPoint(u, _pt);
      curve.getTangent(u, _tan).normalize();
      _n.crossVectors(_tan, _up);
      if (_n.lengthSq() < 1e-6) _n.set(1, 0, 0);
      _n.normalize();
      _b.crossVectors(_tan, _n).normalize();

      const isAnther = u > 0.80 ? 1.0 : 0.0;
      const antherSwell = u > 0.78 ? Math.sin(((u - 0.78) / 0.22) * Math.PI) * 0.011 : 0.0;
      const radius = 0.0042 * (1.0 - 0.3 * u) + antherSwell;

      for (let j = 0; j <= radialSegs; j++) {
        const v = j / radialSegs;
        const a = v * Math.PI * 2;
        const cx = Math.cos(a) * radius;
        const cy = Math.sin(a) * radius;
        positions.push(
          _pt.x + _n.x * cx + _b.x * cy,
          _pt.y + _n.y * cx + _b.y * cy,
          _pt.z + _n.z * cx + _b.z * cy
        );
        uvs.push(v, u);
        stamenMeta.push(u, isAnther);
      }
    }

    const stride = radialSegs + 1;
    for (let i = 0; i < steps; i++) {
      for (let j = 0; j < radialSegs; j++) {
        const a = baseVert + i * stride + j;
        const b = a + stride;
        const c = b + 1;
        const d = a + 1;
        indices.push(a, b, c, a, c, d);
      }
    }

    const tipCenterIdx = positions.length / 3;
    curve.getPoint(1, _pt);
    positions.push(_pt.x, _pt.y + 0.004, _pt.z);
    uvs.push(0.5, 1);
    stamenMeta.push(1, 1);
    const tipRing = baseVert + steps * stride;
    for (let j = 0; j < radialSegs; j++) {
      indices.push(tipCenterIdx, tipRing + j, tipRing + j + 1);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('aStamenMeta', new THREE.Float32BufferAttribute(stamenMeta, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Woven conical calyx cup sealing the bottom pole around all 18 petal roots (outward-wound).
 */
export function buildCalyxBaseCapGeometry(): THREE.BufferGeometry {
  const radialSegs = 32;
  const heightSegs = 12;
  const positions: number[] = [];
  const uvs: number[] = [];
  const petalMeta: number[] = [];
  const edgeMask: number[] = [];
  const fiberSeed: number[] = [];
  const indices: number[] = [];

  for (let iy = 0; iy <= heightSegs; iy++) {
    const v = iy / heightSegs;
    const y = COCOON_BASE_Y + v * 0.22;
    const r = 0.008 + Math.pow(v, 0.72) * 0.16;

    for (let ix = 0; ix <= radialSegs; ix++) {
      const u = ix / radialSegs;
      const theta = u * Math.PI * 2;
      const x = Math.sin(theta) * r;
      const z = Math.cos(theta) * r;
      positions.push(x, y, z);
      uvs.push(u, v);
      petalMeta.push(0.0, v * 0.16, 1.0, 0.5);
      edgeMask.push(0.0);
      fiberSeed.push(Math.sin(theta) * 0.65, -0.95 + v * 0.24, Math.cos(theta) * 0.65);
    }
  }

  const cols = radialSegs + 1;
  for (let iy = 0; iy < heightSegs; iy++) {
    for (let ix = 0; ix < radialSegs; ix++) {
      const a = iy * cols + ix;
      const b = a + 1;
      const c = b + cols;
      const d = a + cols;
      indices.push(a, b, c, a, c, d);
    }
  }

  const apexIdx = positions.length / 3;
  positions.push(0, COCOON_BASE_Y - 0.025, 0);
  uvs.push(0.5, 0);
  petalMeta.push(0.0, 0.0, 1.0, 0.5);
  edgeMask.push(0.0);
  fiberSeed.push(0, -1.0, 0);
  for (let ix = 0; ix < radialSegs; ix++) {
    indices.push(apexIdx, ix + 1, ix);
  }

  const topCenterIdx = positions.length / 3;
  const topRing = heightSegs * cols;
  positions.push(0, COCOON_BASE_Y + 0.21, 0);
  uvs.push(0.5, 1);
  petalMeta.push(1.0, 0.0, 1.0, 0.5);
  edgeMask.push(0.0);
  fiberSeed.push(0, -0.7, 0);
  for (let ix = 0; ix < radialSegs; ix++) {
    indices.push(topCenterIdx, topRing + ix, topRing + ix + 1);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('aPetalMeta', new THREE.Float32BufferAttribute(petalMeta, 4));
  geo.setAttribute('aEdgeMask', new THREE.Float32BufferAttribute(edgeMask, 1));
  geo.setAttribute('aFiberSeed', new THREE.Float32BufferAttribute(fiberSeed, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

export function buildEmberPearlGeometry(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(0.082, 22, 16);
  geo.computeVertexNormals();
  return geo;
}

export function buildOrigamiWingGeometry(side: -1 | 1): THREE.BufferGeometry {
  const s = side;
  const verts = new Float32Array([
    0.0, 0.0, 0.40,
    0.0, 0.045, 0.02,
    0.0, -0.02, -0.46,
    s * 1.24, 0.28, 0.68,
    s * 0.80, -0.11, 0.14,
    s * 0.94, 0.13, -0.24,
    s * 0.38, -0.08, -0.74,
  ]);
  const uvs = new Float32Array([
    0.0, 0.82,
    0.0, 0.50,
    0.0, 0.12,
    1.0, 0.96,
    0.72, 0.52,
    0.84, 0.24,
    0.42, 0.02,
  ]);
  const indices =
    side === 1
      ? [0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 6]
      : [0, 4, 3, 0, 1, 4, 1, 5, 4, 1, 2, 5, 2, 6, 5];

  const indexed = new THREE.BufferGeometry();
  indexed.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  indexed.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  indexed.setIndex(indices);
  const geom = indexed.toNonIndexed();
  indexed.dispose();
  geom.computeVertexNormals();
  return geom;
}

export function buildPollenBillboardQuadGeometry(): THREE.BufferGeometry {
  return new THREE.PlaneGeometry(1, 1);
}

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
