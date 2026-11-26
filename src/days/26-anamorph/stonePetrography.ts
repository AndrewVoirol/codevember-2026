import * as THREE from 'three/webgpu';

export const PETRO_VOLUME_SIZE = 64;
export const DETAIL_TEX_SIZE = 512;

export interface ArchitecturalPetrographyTextures {
  /** Seamless 64^3 3D lithic volume: R=Voronoi clasts, G=3D pores/vesicles, B=mineral sand, A=3D height */
  readonly stoneVolumeTex: THREE.Data3DTexture;
  /** 512^2 tileable board-formed concrete map: R=sand/aggregate grit, G=sharp air bugholes, B=cedar formwork fibers, A=height */
  readonly concreteDetailTex: THREE.DataTexture;
  /** 512^2 tileable Roman travertine map: R=sedimentary strata/veins, G=elongated open vug cavities, B=honed calcite tooth, A=height */
  readonly travertineDetailTex: THREE.DataTexture;
  /** 512^2 tileable architectural mineral paint map: R=roller nap stipple & pass waves, G=capillary edge bleed tooth, B=paint micro-height, A=dry-brush skip flecks */
  readonly paintStippleTex: THREE.DataTexture;
  readonly dispose: () => void;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// ---------------------------------------------------------------------------
// Fast Periodic 2D Value / Gradient Noise (exact tileable wrap across [0, 1))
// ---------------------------------------------------------------------------
interface PeriodicGrid2D {
  readonly period: number;
  readonly values: Float32Array;
}

function createPeriodicGrid2D(period: number, rand: () => number): PeriodicGrid2D {
  const values = new Float32Array(period * period);
  for (let i = 0; i < values.length; i++) {
    values[i] = rand();
  }
  return { period, values };
}

function samplePeriodic2D(grid: PeriodicGrid2D, u: number, v: number): number {
  const p = grid.period;
  const sx = (((u % 1) + 1) % 1) * p;
  const sy = (((v % 1) + 1) % 1) * p;
  const ix0 = Math.floor(sx);
  const iy0 = Math.floor(sy);
  const fx = sx - ix0;
  const fy = sy - iy0;
  const wx = fx * fx * (3 - 2 * fx);
  const wy = fy * fy * (3 - 2 * fy);

  const x0 = ix0 % p;
  const y0 = iy0 % p;
  const x1 = (x0 + 1) % p;
  const y1 = (y0 + 1) % p;

  const vals = grid.values;
  const r0 = y0 * p;
  const r1 = y1 * p;
  const a = vals[r0 + x0];
  const b = vals[r0 + x1];
  const c = vals[r1 + x0];
  const d = vals[r1 + x1];

  const ab = a + (b - a) * wx;
  const cd = c + (d - c) * wx;
  return ab + (cd - ab) * wy;
}

// ---------------------------------------------------------------------------
// Periodic 3D Gradient Noise for Seamless 64^3 Volumetric Lithic Petrography
// ---------------------------------------------------------------------------
const GRAD12: ReadonlyArray<readonly [number, number, number]> = [
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
];

interface PeriodicGradientField3D {
  readonly period: number;
  readonly gx: Float32Array;
  readonly gy: Float32Array;
  readonly gz: Float32Array;
}

function createPeriodicGradientField3D(
  period: number,
  rand: () => number
): PeriodicGradientField3D {
  const total = period * period * period;
  const gx = new Float32Array(total);
  const gy = new Float32Array(total);
  const gz = new Float32Array(total);
  const invSqrt2 = 1 / Math.SQRT2;
  for (let i = 0; i < total; i++) {
    const g = GRAD12[(rand() * 12) | 0];
    gx[i] = g[0] * invSqrt2;
    gy[i] = g[1] * invSqrt2;
    gz[i] = g[2] * invSqrt2;
  }
  return { period, gx, gy, gz };
}

function samplePeriodicGradient3D(
  field: PeriodicGradientField3D,
  ux: number,
  uy: number,
  uz: number
): number {
  const m = field.period;
  const sx = ux * m;
  const sy = uy * m;
  const sz = uz * m;
  const ix0 = Math.floor(sx);
  const iy0 = Math.floor(sy);
  const iz0 = Math.floor(sz);
  const fx = sx - ix0;
  const fy = sy - iy0;
  const fz = sz - iz0;

  const wx = fade(fx);
  const wy = fade(fy);
  const wz = fade(fz);

  const x0 = ((ix0 % m) + m) % m;
  const y0 = ((iy0 % m) + m) % m;
  const z0 = ((iz0 % m) + m) % m;
  const x1 = (x0 + 1) % m;
  const y1 = (y0 + 1) % m;
  const z1 = (z0 + 1) % m;

  const m2 = m * m;
  const { gx, gy, gz } = field;

  const i000 = z0 * m2 + y0 * m + x0;
  const i100 = z0 * m2 + y0 * m + x1;
  const i010 = z0 * m2 + y1 * m + x0;
  const i110 = z0 * m2 + y1 * m + x1;
  const i001 = z1 * m2 + y0 * m + x0;
  const i101 = z1 * m2 + y0 * m + x1;
  const i011 = z1 * m2 + y1 * m + x0;
  const i111 = z1 * m2 + y1 * m + x1;

  const fx1 = fx - 1;
  const fy1 = fy - 1;
  const fz1 = fz - 1;

  const d000 = gx[i000] * fx + gy[i000] * fy + gz[i000] * fz;
  const d100 = gx[i100] * fx1 + gy[i100] * fy + gz[i100] * fz;
  const d010 = gx[i010] * fx + gy[i010] * fy1 + gz[i010] * fz;
  const d110 = gx[i110] * fx1 + gy[i110] * fy1 + gz[i110] * fz;
  const d001 = gx[i001] * fx + gy[i001] * fy + gz[i001] * fz1;
  const d101 = gx[i101] * fx1 + gy[i101] * fy + gz[i101] * fz1;
  const d011 = gx[i011] * fx + gy[i011] * fy1 + gz[i011] * fz1;
  const d111 = gx[i111] * fx1 + gy[i111] * fy1 + gz[i111] * fz1;

  const c00 = d000 + (d100 - d000) * wx;
  const c10 = d010 + (d110 - d010) * wx;
  const c01 = d001 + (d101 - d001) * wx;
  const c11 = d011 + (d111 - d011) * wx;
  const c0 = c00 + (c10 - c00) * wy;
  const c1 = c01 + (c11 - c01) * wy;
  return c0 + (c1 - c0) * wz;
}

/**
 * 1. Seamless 64^3 3D Lithic Volume Texture (aggregate clasts, 3D vesicles, sand tooth, 3D height).
 */
function buildStoneVolumeTexture(): THREE.Data3DTexture {
  const N = PETRO_VOLUME_SIZE;
  const N2 = N * N;
  const totalVoxels = N * N * N;
  const data = new Uint8Array(totalVoxels * 4);
  const rand = mulberry32(0x26a9a301);

  const warpX = createPeriodicGradientField3D(4, rand);
  const warpY = createPeriodicGradientField3D(4, rand);
  const warpZ = createPeriodicGradientField3D(4, rand);
  const grain10 = createPeriodicGradientField3D(10, rand);
  const grain19 = createPeriodicGradientField3D(19, rand);
  const grain31 = createPeriodicGradientField3D(31, rand);

  // Periodic 8^3 Voronoi mineral clast lattice (dark basalt/slag, matrix, bright quartz/calcite)
  const VM = 8;
  const VM2 = VM * VM;
  const vCount = VM * VM * VM;
  const siteX = new Float32Array(vCount);
  const siteY = new Float32Array(vCount);
  const siteZ = new Float32Array(vCount);
  const siteTone = new Float32Array(vCount);
  for (let i = 0; i < vCount; i++) {
    siteX[i] = 0.14 + 0.72 * rand();
    siteY[i] = 0.14 + 0.72 * rand();
    siteZ[i] = 0.14 + 0.72 * rand();
    const roll = rand();
    if (roll < 0.20) {
      siteTone[i] = 0.10 + 0.14 * rand();
    } else if (roll < 0.80) {
      siteTone[i] = 0.40 + 0.22 * rand();
    } else {
      siteTone[i] = 0.80 + 0.18 * rand();
    }
  }

  // Splat 2,400 sharp-rimmed 3D pores/vesicles into periodic buffer
  const poreField = new Float32Array(totalVoxels);
  const mask = N - 1;
  for (let pi = 0; pi < 2400; pi++) {
    const cx = rand() * N;
    const cy = rand() * N;
    const cz = rand() * N;
    const isCoarse = pi >= 1950;
    const radius = isCoarse ? 1.3 + 0.75 * rand() : 0.85 + 0.45 * rand();
    const strength = isCoarse ? 0.85 + 0.15 * rand() : 0.60 + 0.35 * rand();
    const rCeil = Math.ceil(radius);
    const invR2 = 1 / (radius * radius);

    for (let dz = -rCeil; dz <= rCeil; dz++) {
      const wz = Math.floor(cz) + dz;
      const ddz = wz + 0.5 - cz;
      const gz = (wz & mask) * N2;
      for (let dy = -rCeil; dy <= rCeil; dy++) {
        const wy = Math.floor(cy) + dy;
        const ddy = wy + 0.5 - cy;
        const gy = gz + (wy & mask) * N;
        for (let dx = -rCeil; dx <= rCeil; dx++) {
          const wx = Math.floor(cx) + dx;
          const ddx = wx + 0.5 - cx;
          const normSq = (ddx * ddx + ddy * ddy + ddz * ddz) * invR2;
          if (normSq < 1) {
            const profile = Math.pow(1 - normSq, 0.58) * strength;
            const vIdx = gy + (wx & mask);
            if (profile > poreField[vIdx]) {
              poreField[vIdx] = profile;
            }
          }
        }
      }
    }
  }

  const invN = 1 / N;
  const warpAmp = 0.052;
  for (let z = 0; z < N; z++) {
    const uz = (z + 0.5) * invN;
    const zOff = z * N2;
    for (let y = 0; y < N; y++) {
      const uy = (y + 0.5) * invN;
      const yzOff = zOff + y * N;
      for (let x = 0; x < N; x++) {
        const ux = (x + 0.5) * invN;
        const vIdx = yzOff + x;

        const wx = samplePeriodicGradient3D(warpX, ux, uy, uz) * warpAmp;
        const wy = samplePeriodicGradient3D(warpY, ux, uy, uz) * warpAmp;
        const wz = samplePeriodicGradient3D(warpZ, ux, uy, uz) * warpAmp;

        const vx = ((((ux + wx) % 1) + 1) % 1) * VM;
        const vy = ((((uy + wy) % 1) + 1) % 1) * VM;
        const vz = ((((uz + wz) % 1) + 1) % 1) * VM;

        const ix = Math.floor(vx);
        const iy = Math.floor(vy);
        const iz = Math.floor(vz);

        let d1Sq = 999;
        let d2Sq = 999;
        let closestTone = 0.5;

        for (let oz = -1; oz <= 1; oz++) {
          const cz = iz + oz;
          const zBase = ((cz + VM) & 7) * VM2;
          for (let oy = -1; oy <= 1; oy++) {
            const cy = iy + oy;
            const yzBase = zBase + ((cy + VM) & 7) * VM;
            for (let ox = -1; ox <= 1; ox++) {
              const cx = ix + ox;
              const cIdx = yzBase + ((cx + VM) & 7);
              const dx = cx + siteX[cIdx] - vx;
              const dy = cy + siteY[cIdx] - vy;
              const dz = cz + siteZ[cIdx] - vz;
              const dSq = dx * dx + dy * dy + dz * dz;
              if (dSq < d1Sq) {
                d2Sq = d1Sq;
                d1Sq = dSq;
                closestTone = siteTone[cIdx];
              } else if (dSq < d2Sq) {
                d2Sq = dSq;
              }
            }
          }
        }

        const edgeDist = Math.sqrt(d2Sq) - Math.sqrt(d1Sq);
        const hairline = 1 - smoothstep(0.012, 0.044, edgeDist);
        const rVal = clamp01(closestTone * (1 - hairline * 0.72) + 0.09 * hairline);
        const gVal = clamp01(poreField[vIdx]);

        const n10 = samplePeriodicGradient3D(grain10, ux, uy, uz);
        const n19 = samplePeriodicGradient3D(grain19, ux, uy, uz);
        const n31 = samplePeriodicGradient3D(grain31, ux, uy, uz);
        const rawGrain = n10 * 0.36 + n19 * 0.36 + n31 * 0.28;
        const bVal = clamp01(0.5 + rawGrain * 0.92) * 0.92 + 0.04;
        const aVal = clamp01(0.55 + 0.20 * (rVal - 0.5) - 0.58 * gVal + 0.38 * (bVal - 0.5));

        const outIdx = vIdx * 4;
        data[outIdx] = Math.round(rVal * 255);
        data[outIdx + 1] = Math.round(gVal * 255);
        data[outIdx + 2] = Math.round(bVal * 255);
        data[outIdx + 3] = Math.round(aVal * 255);
      }
    }
  }

  const tex = new THREE.Data3DTexture(data, N, N, N);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.wrapR = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

function finalizeDetailTexture(data: Uint8Array, size: number): THREE.DataTexture {
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace;
  tex.generateMipmaps = true;
  tex.anisotropy = 16;
  tex.needsUpdate = true;
  return tex;
}

/**
 * 2. High-Resolution 512x512 Tileable Board-Formed Architectural Concrete Detail Map:
 *    - R: Multi-scale Portland cement sand & sharp mineral aggregate flecks (quartz + dark slag)
 *    - G: Sharp entrapped-air concrete bugholes & micro-pinholes (clustered formwork voids)
 *    - B: Anisotropic rough-sawn cedar shuttering wood-grain fiber & cathedral ring impression
 *    - A: Crisp composite micro-relief heightfield for raking-light normal perturbation
 */
function buildConcreteDetailTexture(): THREE.DataTexture {
  const N = DETAIL_TEX_SIZE;
  const mask = N - 1;
  const total = N * N;
  const rand = mulberry32(0x26c0c7e1);

  // Multi-scale periodic fields for cement laitance, fine silica sand, and cedar wood fibers
  const g8 = createPeriodicGrid2D(8, rand);
  const g24 = createPeriodicGrid2D(24, rand);
  const g64 = createPeriodicGrid2D(64, rand);
  const g128 = createPeriodicGrid2D(128, rand);
  const g256 = createPeriodicGrid2D(256, rand);
  const woodWarp = createPeriodicGrid2D(10, rand);
  const woodFine = createPeriodicGrid2D(96, rand);

  // Splat realistic sharp-rimmed entrapped-air bugholes & vibration pinholes into periodic G buffer
  const bugholeBuf = new Float32Array(total);
  const bugholeRimBuf = new Float32Array(total);

  // Cluster seeds mimic formwork vibration air pockets (refined architectural concrete: sparse natural clusters, never porous pumice!)
  const clusterCount = 12;
  const clusterX = new Float32Array(clusterCount);
  const clusterY = new Float32Array(clusterCount);
  for (let c = 0; c < clusterCount; c++) {
    clusterX[c] = rand() * N;
    clusterY[c] = rand() * N;
  }

  const totalHoles = 285;
  for (let hi = 0; hi < totalHoles; hi++) {
    const clustered = hi < 190;
    let cx: number;
    let cy: number;
    if (clustered) {
      const ci = (rand() * clusterCount) | 0;
      cx = clusterX[ci] + (rand() - 0.5) * 58;
      cy = clusterY[ci] + (rand() - 0.5) * 26;
    } else {
      cx = rand() * N;
      cy = rand() * N;
    }

    const tier = rand();
    // Mostly delicate 0.9-2.2 px pinholes + occasional crisp 2.4-4.8 px entrapped-air bugholes
    const rx = tier < 0.74 ? 0.95 + rand() * 1.25 : tier < 0.94 ? 2.1 + rand() * 1.55 : 3.5 + rand() * 1.65;
    const aspect = 0.68 + rand() * 0.50;
    const ry = rx * aspect;
    const rot = (rand() - 0.5) * 0.55;
    const cosR = Math.cos(rot);
    const sinR = Math.sin(rot);
    const depth = tier < 0.74 ? 0.48 + rand() * 0.30 : 0.72 + rand() * 0.24;
    const irregPhase = rand() * Math.PI * 2;
    const irregAmp = 0.14 + rand() * 0.11;

    const rMax = Math.ceil(Math.max(rx, ry) * 1.45) + 1;
    const invRx2 = 1 / (rx * rx);
    const invRy2 = 1 / (ry * ry);

    for (let dy = -rMax; dy <= rMax; dy++) {
      const wy = Math.floor(cy) + dy;
      const ddy = wy + 0.5 - cy;
      const rowOff = (wy & mask) * N;
      for (let dx = -rMax; dx <= rMax; dx++) {
        const wx = Math.floor(cx) + dx;
        const ddx = wx + 0.5 - cx;
        const lx = ddx * cosR + ddy * sinR;
        const ly = -ddx * sinR + ddy * cosR;
        // Organic irregular air-bubble boundary (non-circular, low-order asymmetry only)
        const ang = Math.atan2(ly, lx);
        const mod = 1 + irregAmp * Math.sin(ang * 2 + irregPhase) + irregAmp * 0.55 * Math.cos(ang + irregPhase * 1.7);
        const r2 = (lx * lx * invRx2 + ly * ly * invRy2) / (mod * mod);
        const idx = rowOff + (wx & mask);
        if (r2 < 1.0) {
          // Steep concave crater walls: stays deep nearly to rim then rises sharply
          const cavity = Math.pow(1 - r2, 0.38) * depth;
          if (cavity > bugholeBuf[idx]) bugholeBuf[idx] = cavity;
        } else if (r2 < 1.75) {
          // Subtle raised cement meniscus lip around larger air voids
          const rim = (1 - (r2 - 1.0) / 0.75) * depth * 0.18;
          if (rim > bugholeRimBuf[idx]) bugholeRimBuf[idx] = rim;
        }
      }
    }
  }

  // Splat ~1,600 crisp fine exposed mineral sand / aggregate specks (dark basalt + pale quartz)
  const aggValBuf = new Float32Array(total);
  const aggHgtBuf = new Float32Array(total);
  for (let ai = 0; ai < 1600; ai++) {
    const cx = rand() * N;
    const cy = rand() * N;
    const rad = 0.85 + rand() * 2.1;
    const isBrightQuartz = rand() > 0.48;
    const sign = isBrightQuartz ? 1.0 : -1.0;
    const amp = (0.28 + rand() * 0.42) * sign;
    const hAmp = (0.14 + rand() * 0.22) * (isBrightQuartz ? 1.0 : 0.55);
    const rCeil = Math.ceil(rad);
    const invR2 = 1 / (rad * rad);
    for (let dy = -rCeil; dy <= rCeil; dy++) {
      const wy = Math.floor(cy) + dy;
      const ddy = wy + 0.5 - cy;
      const rowOff = (wy & mask) * N;
      for (let dx = -rCeil; dx <= rCeil; dx++) {
        const wx = Math.floor(cx) + dx;
        const ddx = wx + 0.5 - cx;
        const r2 = (ddx * ddx + ddy * ddy) * invR2;
        if (r2 < 1.0) {
          const w = 1 - r2;
          const idx = rowOff + (wx & mask);
          aggValBuf[idx] += amp * w * 0.55;
          aggHgtBuf[idx] = Math.max(aggHgtBuf[idx], hAmp * w);
        }
      }
    }
  }

  const data = new Uint8Array(total * 4);
  const invN = 1 / N;
  for (let y = 0; y < N; y++) {
    const v = (y + 0.5) * invN;
    const rowOff = y * N;
    for (let x = 0; x < N; x++) {
      const u = (x + 0.5) * invN;
      const idx = rowOff + x;

      // R: Multi-scale crisp Portland sand & mineral aggregate matrix (sharp 1-4px micro-grain!)
      const n8 = samplePeriodic2D(g8, u, v);
      const n24 = samplePeriodic2D(g24, u, v);
      const n64 = samplePeriodic2D(g64, u, v);
      const n128 = samplePeriodic2D(g128, u, v);
      const n256 = samplePeriodic2D(g256, u, v);
      const pixelSpark = (rand() - 0.5) * 0.16;
      const sandBase =
        n8 * 0.14 + n24 * 0.18 + n64 * 0.22 + n128 * 0.24 + n256 * 0.22 + pixelSpark;
      const rVal = clamp01(sandBase + aggValBuf[idx] * 0.45);

      // G: Crisp entrapped-air bugholes
      const gVal = clamp01(bugholeBuf[idx]);

      // B: Anisotropic rough-sawn cedar shuttering wood-grain fiber & earlywood cathedral rings
      // Horizontal along u, tightly stratified vertically along v
      const wWarp = (samplePeriodic2D(woodWarp, u * 1.0, v * 2.0) - 0.5) * 0.26;
      const ringPhase = (v * 14.0 + wWarp * 3.8 + Math.sin(u * Math.PI * 2.0) * 0.32) * Math.PI * 2.0;
      const ringWave = 0.5 + 0.5 * Math.sin(ringPhase);
      const cathedral = Math.pow(ringWave, 1.85);
      // Fine sharp horizontal cedar fiber striations (anisotropic 1:10 aspect)
      const fiberA = samplePeriodic2D(woodFine, u * 0.25, v * 2.5);
      const fiberB = samplePeriodic2D(g256, u * 0.25, v * 1.0);
      const bVal = clamp01(cathedral * 0.42 + fiberA * 0.36 + fiberB * 0.22);

      // A: Composite crisp heightfield: smooth troweled cement paste + fine sand + woodgrain fibers - crisp sparse bugholes
      const hRaw =
        0.60 +
        (rVal - 0.5) * 0.26 +
        aggHgtBuf[idx] * 0.34 +
        (bVal - 0.5) * 0.24 +
        bugholeRimBuf[idx] * 0.28 -
        gVal * 0.56;
      const aVal = clamp01(hRaw);

      const o = idx * 4;
      data[o] = Math.round(rVal * 255);
      data[o + 1] = Math.round(gVal * 255);
      data[o + 2] = Math.round(bVal * 255);
      data[o + 3] = Math.round(aVal * 255);
    }
  }

  return finalizeDetailTexture(data, N);
}

/**
 * 3. High-Resolution 512x512 Tileable Honed & Open-Pore Roman Travertine Detail Map:
 *    - R: Flowing vein-cut sedimentary limestone bands & thin dark stylolite mineral seams
 *    - G: Signature sharp-walled elongated horizontal travertine cavities (vugs)
 *    - B: Fine honed calcite crystal grain & subtle directional honing tooth
 *    - A: Smooth honed plateau (~0.78) cut by steep-walled recessed vugs (0.04..0.38)
 */
function buildTravertineDetailTexture(): THREE.DataTexture {
  const N = DETAIL_TEX_SIZE;
  const mask = N - 1;
  const total = N * N;
  const rand = mulberry32(0x2674a7e2);

  const warpSlow = createPeriodicGrid2D(6, rand);
  const warpMed = createPeriodicGrid2D(18, rand);
  const veinFine = createPeriodicGrid2D(64, rand);
  const calcite128 = createPeriodicGrid2D(128, rand);
  const calcite256 = createPeriodicGrid2D(256, rand);

  // Splat ~460 signature elongated horizontal open Roman travertine cavities (vugs)
  // clustered preferentially along specific porous sedimentary bedding horizons!
  const vugBuf = new Float32Array(total);
  const vugRimBuf = new Float32Array(total);
  const bedHorizons = [0.11, 0.24, 0.37, 0.52, 0.63, 0.78, 0.89];

  for (let vi = 0; vi < 460; vi++) {
    const onHorizon = vi < 350;
    const cx = rand() * N;
    const baseY = onHorizon
      ? bedHorizons[(rand() * bedHorizons.length) | 0] * N + (rand() - 0.5) * 26
      : rand() * N;
    // Slight wavy bedding displacement matching strata wave below
    const uNorm = (((cx / N) % 1) + 1) % 1;
    const vNorm = (((baseY / N) % 1) + 1) % 1;
    const cy = baseY + (samplePeriodic2D(warpSlow, uNorm, vNorm) - 0.5) * 22;

    const isMajor = rand() > 0.72;
    const rx = isMajor ? 8.5 + rand() * 14.0 : 2.6 + rand() * 5.8;
    const ry = isMajor ? 1.8 + rand() * 2.4 : 1.0 + rand() * 1.45;
    const depth = isMajor ? 0.76 + rand() * 0.24 : 0.48 + rand() * 0.38;
    const phase = rand() * Math.PI * 2;

    const rMaxX = Math.ceil(rx * 1.35) + 1;
    const rMaxY = Math.ceil(ry * 1.55) + 1;
    const invRx2 = 1 / (rx * rx);
    const invRy2 = 1 / (ry * ry);

    for (let dy = -rMaxY; dy <= rMaxY; dy++) {
      const wy = Math.floor(cy) + dy;
      const ddy = wy + 0.5 - cy;
      const rowOff = (wy & mask) * N;
      for (let dx = -rMaxX; dx <= rMaxX; dx++) {
        const wx = Math.floor(cx) + dx;
        const ddx = wx + 0.5 - cx;
        // Organic pinched lenticular travertine pore shape
        const pinch = 1 + 0.22 * Math.sin((ddx / rx) * 3.1 + phase);
        const yWarp = ddy + Math.sin((ddx / rx) * 2.2 + phase) * ry * 0.22;
        const r2 = ddx * ddx * invRx2 + (yWarp * yWarp * invRy2) / (pinch * pinch);
        const idx = rowOff + (wx & mask);
        if (r2 < 1.0) {
          // Steep cut wall so honed flat face transitions crisply into cavernous vug!
          const cav = Math.pow(1 - r2, 0.32) * depth;
          if (cav > vugBuf[idx]) vugBuf[idx] = cav;
        } else if (r2 < 1.55) {
          const rim = (1 - (r2 - 1.0) / 0.55) * depth * 0.14;
          if (rim > vugRimBuf[idx]) vugRimBuf[idx] = rim;
        }
      }
    }
  }

  const data = new Uint8Array(total * 4);
  const invN = 1 / N;
  for (let y = 0; y < N; y++) {
    const v = (y + 0.5) * invN;
    const rowOff = y * N;
    for (let x = 0; x < N; x++) {
      const u = (x + 0.5) * invN;
      const idx = rowOff + x;

      const ws = (samplePeriodic2D(warpSlow, u, v) - 0.5) * 0.085;
      const wm = (samplePeriodic2D(warpMed, u * 1.0, v * 2.0) - 0.5) * 0.025;
      const vBed = v + ws + wm;

      // Flowing sedimentary travertine strata + thin dark mineral stylolite veins
      const band1 = 0.5 + 0.5 * Math.sin(vBed * Math.PI * 2 * 5.0);
      const band2 = 0.5 + 0.5 * Math.sin(vBed * Math.PI * 2 * 13.0 + 1.4);
      const band3 = samplePeriodic2D(veinFine, u * 0.25, vBed * 1.5);
      const stylolite = Math.pow(1 - Math.abs(Math.sin(vBed * Math.PI * 2 * 7.0 + wm * 14.0)), 10.0);
      const rVal = clamp01(band1 * 0.42 + band2 * 0.26 + band3 * 0.32 - stylolite * 0.36);

      // Modulate vug edges slightly with fine mineral grit so cavity walls look fractured, not oval
      const edgeMicro = samplePeriodic2D(calcite128, u, v);
      const gRaw = vugBuf[idx] > 0.02 ? vugBuf[idx] * (0.76 + 0.38 * edgeMicro) : 0;
      const gVal = clamp01(smoothstep(0.14, 0.44, gRaw) * gRaw);

      // Honed calcite crystal tooth (subtle crystalline sheen variation + fine mineral flecks)
      const c128 = samplePeriodic2D(calcite128, u, v);
      const c256 = samplePeriodic2D(calcite256, u, v);
      const sparkle = (rand() - 0.5) * 0.14;
      const bVal = clamp01(c128 * 0.45 + c256 * 0.55 + sparkle);

      // Height: smooth honed stone plateau punctuated by crisp deep open travertine vugs!
      const hRaw =
        0.78 +
        (rVal - 0.5) * 0.06 +
        (bVal - 0.5) * 0.09 +
        vugRimBuf[idx] * 0.16 -
        gVal * 0.74 -
        stylolite * 0.08;
      const aVal = clamp01(hRaw);

      const o = idx * 4;
      data[o] = Math.round(rVal * 255);
      data[o + 1] = Math.round(gVal * 255);
      data[o + 2] = Math.round(bVal * 255);
      data[o + 3] = Math.round(aVal * 255);
    }
  }

  return finalizeDetailTexture(data, N);
}

/**
 * 4. High-Resolution 512x512 Tileable Architectural Mural Paint Stipple & Edge-Tooth Map:
 *    - R: Nap-roller orange-peel stipple + subtle broad hand-rolled coat thickness variation
 *    - G: High-frequency fibrous capillary bleed noise for realistic masonry stencil edges
 *    - B: Fine roller-stipple micro-heightfield catching raking light on painted surfaces
 *    - A: Sparse dry-brush micro-skip specks over rough stone peaks
 */
function buildPaintStippleTexture(): THREE.DataTexture {
  const N = DETAIL_TEX_SIZE;
  const total = N * N;
  const rand = mulberry32(0x269a17e3);

  const passLap = createPeriodicGrid2D(6, rand);
  const rollerMed = createPeriodicGrid2D(48, rand);
  const rollerFine = createPeriodicGrid2D(128, rand);
  const stipple256 = createPeriodicGrid2D(256, rand);
  const bleed64 = createPeriodicGrid2D(64, rand);
  const bleed192 = createPeriodicGrid2D(192, rand);

  const data = new Uint8Array(total * 4);
  const invN = 1 / N;
  for (let y = 0; y < N; y++) {
    const v = (y + 0.5) * invN;
    const rowOff = y * N;
    for (let x = 0; x < N; x++) {
      const u = (x + 0.5) * invN;
      const idx = rowOff + x;

      // Subtle broad vertical/angled roller pass lap variation + crisp fine roller-nap orange peel
      const lap = samplePeriodic2D(passLap, u * 1.0, v * 0.5);
      const rMed = samplePeriodic2D(rollerMed, u, v);
      const rFine = samplePeriodic2D(rollerFine, u, v);
      const s256 = samplePeriodic2D(stipple256, u, v);
      const crispPeel = Math.abs(s256 - 0.5) * 2.0;

      // R: Coat thickness / pigment density modulation (never flat vector fill!)
      const rVal = clamp01(0.24 * lap + 0.24 * rMed + 0.30 * rFine + 0.22 * s256);

      // G: Capillary bleed along masonry pores at stencil masking edge
      const b64 = samplePeriodic2D(bleed64, u, v);
      const b192 = samplePeriodic2D(bleed192, u, v);
      const gVal = clamp01(0.38 * b64 + 0.42 * b192 + 0.20 * s256);

      // B: Fine physical roller-stipple micro-heightfield
      const bVal = clamp01(0.5 + (rFine - 0.5) * 0.42 + (s256 - 0.5) * 0.52 - crispPeel * 0.14);

      // A: Sparse microscopic dry-brush pinholes where stiff pigment skips rough aggregate
      const skipRaw = rFine * 0.52 + s256 * 0.48 + (rand() - 0.5) * 0.12;
      const aVal = clamp01(smoothstep(0.72, 0.90, skipRaw));

      const o = idx * 4;
      data[o] = Math.round(rVal * 255);
      data[o + 1] = Math.round(gVal * 255);
      data[o + 2] = Math.round(bVal * 255);
      data[o + 3] = Math.round(aVal * 255);
    }
  }

  return finalizeDetailTexture(data, N);
}

/**
 * Generates all 4 deterministic procedural petrography & mineral-paint textures for Day 26.
 */
export function createArchitecturalPetrographyTextures(): ArchitecturalPetrographyTextures {
  const stoneVolumeTex = buildStoneVolumeTexture();
  const concreteDetailTex = buildConcreteDetailTexture();
  const travertineDetailTex = buildTravertineDetailTexture();
  const paintStippleTex = buildPaintStippleTexture();

  return {
    stoneVolumeTex,
    concreteDetailTex,
    travertineDetailTex,
    paintStippleTex,
    dispose: () => {
      stoneVolumeTex.dispose();
      concreteDetailTex.dispose();
      travertineDetailTex.dispose();
      paintStippleTex.dispose();
    },
  };
}
