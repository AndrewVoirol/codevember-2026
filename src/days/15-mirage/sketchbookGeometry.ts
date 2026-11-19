import * as THREE from 'three/webgpu';
import {
  ARCH_LAYERS,
  BACK_SANCTUM_Z,
  BOOK_HALF_H,
  BOOK_HALF_W,
} from './mirageRig';
import type { ArchLayerSpec } from './types';

function hash11(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

/**
 * High-frequency organic cotton-rag tear along perimeter parameter u in [0, 1).
 * Avoids low-frequency sinusoidal pillows so edges read as crisp torn paper fibers.
 */
function paperTearNoise(u: number, layerIdx: number, amp = 1.0): number {
  const tau = Math.PI * 2;
  const p = u * tau;
  const seed = layerIdx * 5.13;
  const n1 = Math.sin(p * 5.0 + seed) * 0.0065;
  const n2 = Math.cos(p * 13.0 - seed * 1.9) * 0.0048;
  const n3 = Math.sin(p * 29.0 + seed * 4.3) * 0.0036;
  const n4 = Math.cos(p * 61.0 - seed * 7.1) * 0.0024;
  const n5 = (hash11(Math.floor(u * 240) + layerIdx * 137) - 0.5) * 0.0026;
  return (n1 + n2 + n3 + n4 + n5) * amp;
}

/**
 * Evaluates a point along the organic arch cutout loop for parameter u in [0, 1).
 */
function sampleArchInnerPoint(
  layer: ArchLayerSpec,
  u: number,
  out: [number, number]
): void {
  const k = layer.index;
  const sillFrac = 0.28;
  const archSpan = 1.0 - sillFrac;

  if (u < archSpan) {
    const s = u / archSpan;
    const theta = s * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);

    const gothicLift = Math.pow(Math.max(0, sinT), 0.78);
    const shoulderWave = 1.0 + 0.032 * Math.sin(theta * 2.0 + k * 1.65);
    const tear = paperTearNoise(u, k, 1.0);

    const x = cosT * (layer.rx * shoulderWave + tear * 0.9);
    const y =
      layer.yBot +
      gothicLift * (layer.yTop - layer.yBot) +
      tear * (0.7 + 0.4 * sinT);
    out[0] = x;
    out[1] = y;
  } else {
    const s = (u - archSpan) / sillFrac;
    const xBase = -layer.rx + s * (2.0 * layer.rx);
    const spineDip = -0.014 * Math.exp(-(xBase * xBase) / 0.007);
    const tear = paperTearNoise(u, k + 11, 0.95);
    out[0] = xBase;
    out[1] = layer.yBot + spineDip + tear;
  }
}

function sampleBookOuterBoundary(
  ix: number,
  iy: number,
  cy: number,
  layerIdx: number,
  scaleFactor: number,
  out: [number, number]
): void {
  const dx = ix;
  const dy = iy - cy;
  const hw = BOOK_HALF_W * scaleFactor;
  const hh = BOOK_HALF_H * scaleFactor;
  const tx = Math.abs(dx) > 1e-6 ? hw / Math.abs(dx) : 1e6;
  const ty = Math.abs(dy) > 1e-6 ? hh / Math.abs(dy) : 1e6;
  const t = Math.min(tx, ty);

  let bx = dx * t;
  let by = cy + dy * t;

  const perimCoord = (bx + by * 1.37) * 2.5 + layerIdx * 0.7;
  const edgeTear =
    Math.sin(perimCoord * 11.0) * 0.004 +
    Math.cos(perimCoord * 29.0) * 0.0022;

  if (tx < ty) {
    bx += Math.sign(dx) * edgeTear;
  } else {
    by += Math.sign(dy) * edgeTear;
  }

  const spineNotch = 0.014 * Math.exp(-(bx * bx) / 0.01);
  if (by > 0) by -= spineNotch;
  else by -= spineNotch * 0.7;

  out[0] = bx;
  out[1] = by;
}

function spineCreaseZ(x: number, layerIdx: number): number {
  const crease = -0.02 * Math.exp(-(x * x) / 0.018);
  const pageCurl =
    layerIdx === 0
      ? 0.01 * Math.sin((Math.abs(x) / BOOK_HALF_W) * Math.PI)
      : 0.003 * Math.sin((Math.abs(x) / BOOK_HALF_W) * Math.PI);
  return crease + pageCurl;
}

/**
 * Builds a crisp, flat torn-paper arch sheet with SPLIT NORMALS between:
 *   - Cut-edge thickness lip (Rings 0..1: recessed -0.005 Z, inward bevel)
 *   - Flat paper face + narrow torn white cotton fiber bevel collar (Rings 2..5: pure front-facing paper normals)
 * Zero pillowy tube-normal smoothing!
 */
export function buildArchLayerGeometry(layer: ArchLayerSpec): THREE.BufferGeometry {
  const SEGMENTS = 240;
  const RINGS = 6; // [0,1] = cut paper edge lip; [2,3,4,5] = flat paper face
  const vertCount = SEGMENTS * RINGS;

  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const aDeckle = new Float32Array(vertCount);
  const aLayerIdx = new Float32Array(vertCount);
  const aHatchDensity = new Float32Array(vertCount);
  const aHatchAngle = new Float32Array(vertCount);
  const aRadial = new Float32Array(vertCount);

  const innerPt: [number, number] = [0, 0];
  const outerPt: [number, number] = [0, 0];
  const cy = (layer.yTop + layer.yBot) * 0.5;
  const outerScale = layer.index === 0 ? 1.0 : 0.985 - layer.index * 0.004;

  for (let i = 0; i < SEGMENTS; i++) {
    const u = i / SEGMENTS;
    sampleArchInnerPoint(layer, u, innerPt);
    const ix = innerPt[0];
    const iy = innerPt[1];
    sampleBookOuterBoundary(ix, iy, cy, layer.index, outerScale, outerPt);
    const ox = outerPt[0];
    const oy = outerPt[1];

    const dx = ox - ix;
    const dy = oy - iy;
    const dist = Math.hypot(dx, dy) || 1;
    const dirX = dx / dist;
    const dirY = dy / dist;

    // Narrow jagged cotton fiber deckle tear along cut rim (0.004..0.008 wide)
    const fiberVar = 0.65 + 0.7 * hash11(i * 19 + layer.index * 53);
    const bevel = layer.bevelWidth * fiberVar;
    const midFrac = Math.min(0.32, (bevel * 5.0) / dist);
    const zBase = layer.z + spineCreaseZ(ix, layer.index);

    for (let r = 0; r < RINGS; r++) {
      const idx = r * SEGMENTS + i;
      let px = ix;
      let py = iy;
      let pz = zBase;
      let deckle = 0;
      let radial = 0;

      if (r === 0) {
        // Back corner of cut-paper thickness edge
        px = ix;
        py = iy;
        pz = zBase - 0.005;
        deckle = 1.0;
        radial = 0.0;
      } else if (r === 1) {
        // Front corner of cut-paper thickness edge
        px = ix;
        py = iy;
        pz = zBase;
        deckle = 1.0;
        radial = 0.02;
      } else if (r === 2) {
        // Front sheet inner edge (duplicate position to break normal smoothing!)
        px = ix;
        py = iy;
        pz = zBase;
        deckle = 0.95;
        radial = 0.03;
      } else if (r === 3) {
        // Sharp outer edge of white torn cotton bevel
        px = ix + dirX * bevel;
        py = iy + dirY * bevel;
        pz = layer.z + spineCreaseZ(px, layer.index);
        deckle = 0.0;
        radial = 0.08;
      } else if (r === 4) {
        // Mid sheet bridge
        px = ix + dx * midFrac;
        py = iy + dy * midFrac;
        pz = layer.z + spineCreaseZ(px, layer.index);
        deckle = 0.0;
        radial = 0.35;
      } else {
        // Outer deckled book rectangle
        px = ox;
        py = oy;
        pz = layer.z + spineCreaseZ(px, layer.index);
        deckle = 0.0;
        radial = 1.0;
      }

      positions[idx * 3] = px;
      positions[idx * 3 + 1] = py;
      positions[idx * 3 + 2] = pz;

      uvs[idx * 2] = px / (BOOK_HALF_W * 2) + 0.5;
      uvs[idx * 2 + 1] = py / (BOOK_HALF_H * 2) + 0.5;

      aDeckle[idx] = deckle;
      aLayerIdx[idx] = layer.index;
      aHatchDensity[idx] = layer.hatchDensity;
      aHatchAngle[idx] = layer.hatchAngle;
      aRadial[idx] = radial;
    }
  }

  const indices: number[] = [];
  // Lip strip (r = 0 -> 1) and flat sheet strips (r = 2 -> 3 -> 4 -> 5), skipping 1 -> 2 seam!
  const stripPairs = [0, 2, 3, 4];
  for (const r of stripPairs) {
    for (let i = 0; i < SEGMENTS; i++) {
      const next = (i + 1) % SEGMENTS;
      const a = r * SEGMENTS + i;
      const b = (r + 1) * SEGMENTS + i;
      const c = (r + 1) * SEGMENTS + next;
      const d = r * SEGMENTS + next;
      indices.push(a, b, c, a, c, d);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aDeckle', new THREE.BufferAttribute(aDeckle, 1));
  geom.setAttribute('aLayerIdx', new THREE.BufferAttribute(aLayerIdx, 1));
  geom.setAttribute('aHatchDensity', new THREE.BufferAttribute(aHatchDensity, 1));
  geom.setAttribute('aHatchAngle', new THREE.BufferAttribute(aHatchAngle, 1));
  geom.setAttribute('aRadial', new THREE.BufferAttribute(aRadial, 1));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Builds the curved back cotton-rag sanctum backdrop (Layer 7).
 */
export function buildBackSanctumGeometry(): THREE.BufferGeometry {
  const nx = 44;
  const ny = 36;
  const halfW = BOOK_HALF_W * 1.05;
  const halfH = BOOK_HALF_H * 1.04;
  const vertCount = (nx + 1) * (ny + 1);

  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const aDeckle = new Float32Array(vertCount);
  const aLayerIdx = new Float32Array(vertCount);
  const aHatchDensity = new Float32Array(vertCount);
  const aHatchAngle = new Float32Array(vertCount);
  const aRadial = new Float32Array(vertCount);

  let ptr = 0;
  for (let iy = 0; iy <= ny; iy++) {
    const v = iy / ny;
    const y = (v * 2 - 1) * halfH;
    for (let ix = 0; ix <= nx; ix++) {
      const u = ix / nx;
      const x = (u * 2 - 1) * halfW;
      const flankCurve = Math.max(0, Math.abs(x) - 0.34);
      const z = BACK_SANCTUM_Z + flankCurve * flankCurve * 1.35 + spineCreaseZ(x, 7) * 0.5;

      positions[ptr * 3] = x;
      positions[ptr * 3 + 1] = y;
      positions[ptr * 3 + 2] = z;

      uvs[ptr * 2] = u;
      uvs[ptr * 2 + 1] = v;

      aDeckle[ptr] = 0.0;
      aLayerIdx[ptr] = 7.0;
      aHatchDensity[ptr] = 0.04;
      aHatchAngle[ptr] = 0.3;
      aRadial[ptr] = Math.min(1.0, Math.hypot(x / halfW, y / halfH));
      ptr++;
    }
  }

  const indices: number[] = [];
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const a = iy * (nx + 1) + ix;
      const b = a + 1;
      const c = a + (nx + 1) + 1;
      const d = a + (nx + 1);
      indices.push(a, b, c, a, c, d);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aDeckle', new THREE.BufferAttribute(aDeckle, 1));
  geom.setAttribute('aLayerIdx', new THREE.BufferAttribute(aLayerIdx, 1));
  geom.setAttribute('aHatchDensity', new THREE.BufferAttribute(aHatchDensity, 1));
  geom.setAttribute('aHatchAngle', new THREE.BufferAttribute(aHatchAngle, 1));
  geom.setAttribute('aRadial', new THREE.BufferAttribute(aRadial, 1));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Builds stacked riffled deckle-edged page strips along the outer flanks, top, bottom,
 * and cloth binding spine.
 */
export function buildPageBlockSheavesGeometry(): THREE.BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];
  const aDeckle: number[] = [];
  const aLayerIdx: number[] = [];
  const aHatchDensity: number[] = [];
  const aHatchAngle: number[] = [];
  const aRadial: number[] = [];
  const indices: number[] = [];

  const pushBoxStrip = (
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    layerTag: number,
    deckleVal: number
  ) => {
    const base = positions.length / 3;
    const corners: Array<[number, number, number, number, number]> = [
      [x0, y0, z0, 0, 0],
      [x1, y0, z0, 1, 0],
      [x1, y1, z0, 1, 1],
      [x0, y1, z0, 0, 1],
      [x0, y0, z1, 0, 0],
      [x1, y0, z1, 1, 0],
      [x1, y1, z1, 1, 1],
      [x0, y1, z1, 0, 1],
    ];
    for (const [px, py, pz, u, v] of corners) {
      positions.push(px, py, pz);
      uvs.push(u, v);
      aDeckle.push(deckleVal);
      aLayerIdx.push(layerTag);
      aHatchDensity.push(0.16);
      aHatchAngle.push(0.5);
      aRadial.push(1.0);
    }
    const faces = [
      [0, 1, 2, 0, 2, 3],
      [4, 6, 5, 4, 7, 6],
      [0, 4, 5, 0, 5, 1],
      [3, 2, 6, 3, 6, 7],
      [0, 3, 7, 0, 7, 4],
      [1, 5, 6, 1, 6, 2],
    ];
    for (const f of faces) {
      for (const idx of f) indices.push(base + idx);
    }
  };

  const pageCount = 10;
  for (let p = 0; p < pageCount; p++) {
    const t = p / (pageCount - 1);
    const zFront = 0.12 - t * 0.66;
    const zBack = zFront - 0.048;
    const flare = (1 - t) * 0.028 + Math.sin(p * 2.1) * 0.004;
    const hw = BOOK_HALF_W + flare;
    const hh = BOOK_HALF_H - t * 0.015 + Math.cos(p * 1.7) * 0.004;
    const slabThickness = 0.012;
    const deckle = p % 2 === 0 ? 0.35 : 0.15;

    pushBoxStrip(-hw, -hh, zBack, -hw + slabThickness, hh, zFront, 8.0, deckle);
    pushBoxStrip(hw - slabThickness, -hh, zBack, hw, hh, zFront, 8.0, deckle);
    if (p % 2 === 0) {
      pushBoxStrip(-hw + 0.01, hh - 0.01, zBack, hw - 0.01, hh, zFront, 8.0, 0.2);
      pushBoxStrip(-hw + 0.01, -hh, zBack, hw - 0.01, -hh + 0.01, zFront, 8.0, 0.2);
    }
  }

  pushBoxStrip(
    -BOOK_HALF_W - 0.034,
    -BOOK_HALF_H - 0.012,
    BACK_SANCTUM_Z - 0.055,
    BOOK_HALF_W + 0.034,
    BOOK_HALF_H + 0.012,
    BACK_SANCTUM_Z - 0.018,
    9.0,
    0.0
  );

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geom.setAttribute('aDeckle', new THREE.Float32BufferAttribute(aDeckle, 1));
  geom.setAttribute('aLayerIdx', new THREE.Float32BufferAttribute(aLayerIdx, 1));
  geom.setAttribute('aHatchDensity', new THREE.Float32BufferAttribute(aHatchDensity, 1));
  geom.setAttribute('aHatchAngle', new THREE.Float32BufferAttribute(aHatchAngle, 1));
  geom.setAttribute('aRadial', new THREE.Float32BufferAttribute(aRadial, 1));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

export function buildAllArchGeometries(): THREE.BufferGeometry[] {
  return ARCH_LAYERS.map((layer) => buildArchLayerGeometry(layer));
}
