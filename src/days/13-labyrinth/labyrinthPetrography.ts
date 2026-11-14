import * as THREE from 'three/webgpu';

export const PETRO_VOLUME_SIZE = 64;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

const GRAD12: ReadonlyArray<readonly [number, number, number]> = [
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
];

interface PeriodicGradientField {
  period: number;
  gx: Float32Array;
  gy: Float32Array;
  gz: Float32Array;
}

function createPeriodicGradientField(
  period: number,
  rand: () => number
): PeriodicGradientField {
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

function samplePeriodicGradient(
  field: PeriodicGradientField,
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
 * Generates a tileable 64x64x64 RGBA8 3D petrographic volume texture encoding
 * isotropic mineral clasts, true 3D spherical pores, sandstone grain, and relief.
 *
 * Channels (linear NoColorSpace `[0..1]`):
 * - R: Warped 3D Voronoi clasts + pale calcite shell chips + thin fracture hairlines
 * - G: True 3D spherical vesicular pin-pores & coarse basaltic vesicles
 * - B: 3-octave periodic 3D gradient sandstone/chisel grain (`M = 10, 19, 31`)
 * - A: Composite isotropic micro-relief heightfield for 3D normal perturbation
 */
export function createPetrographicVolumeTexture(): THREE.Data3DTexture {
  const N = PETRO_VOLUME_SIZE;
  const N2 = N * N;
  const totalVoxels = N * N * N;
  const data = new Uint8Array(totalVoxels * 4);
  const rand = mulberry32(0x13c0de26);

  // Domain-warp periodic fields (M = 4) + 3-octave sandstone grain fields (M = 10, 19, 31)
  const warpX = createPeriodicGradientField(4, rand);
  const warpY = createPeriodicGradientField(4, rand);
  const warpZ = createPeriodicGradientField(4, rand);
  const grain10 = createPeriodicGradientField(10, rand);
  const grain19 = createPeriodicGradientField(19, rand);
  const grain31 = createPeriodicGradientField(31, rand);

  // Periodic M = 8 Voronoi clast lattice (8^3 = 512 cells)
  const VM = 8;
  const VM2 = VM * VM;
  const vCount = VM * VM * VM;
  const siteX = new Float32Array(vCount);
  const siteY = new Float32Array(vCount);
  const siteZ = new Float32Array(vCount);
  const siteTone = new Float32Array(vCount);
  for (let z = 0; z < VM; z++) {
    for (let y = 0; y < VM; y++) {
      for (let x = 0; x < VM; x++) {
        const idx = z * VM2 + y * VM + x;
        siteX[idx] = 0.14 + 0.72 * rand();
        siteY[idx] = 0.14 + 0.72 * rand();
        siteZ[idx] = 0.14 + 0.72 * rand();
        const roll = rand();
        // ~16% angular pale shell/calcite clasts, remaining warm oolitic matrix
        siteTone[idx] = roll < 0.16 ? 0.78 + 0.18 * rand() : 0.34 + 0.34 * rand();
      }
    }
  }

  // Splat 2,600 true 3D Euclidean spherical pores into periodic G buffer (`& 63` wrap)
  const poreField = new Float32Array(totalVoxels);
  const mask63 = N - 1;
  const totalPores = 2600;
  for (let pi = 0; pi < totalPores; pi++) {
    const cx = rand() * N;
    const cy = rand() * N;
    const cz = rand() * N;
    const isCoarse = pi >= 2100;
    const radius = isCoarse ? 1.45 + 0.90 * rand() : 1.00 + 0.45 * rand();
    const strength = isCoarse ? 0.75 + 0.25 * rand() : 0.56 + 0.36 * rand();
    const rCeil = Math.ceil(radius);
    const invR2 = 1 / (radius * radius);

    for (let dz = -rCeil; dz <= rCeil; dz++) {
      const wz = Math.floor(cz) + dz;
      const ddz = wz + 0.5 - cz;
      const gz = (wz & mask63) * N2;
      for (let dy = -rCeil; dy <= rCeil; dy++) {
        const wy = Math.floor(cy) + dy;
        const ddy = wy + 0.5 - cy;
        const gy = gz + (wy & mask63) * N;
        for (let dx = -rCeil; dx <= rCeil; dx++) {
          const wx = Math.floor(cx) + dx;
          const ddx = wx + 0.5 - cx;
          const normSq = (ddx * ddx + ddy * ddy + ddz * ddz) * invR2;
          if (normSq < 1) {
            const profile = Math.pow(1 - normSq, 1.5) * strength;
            const vIdx = gy + (wx & mask63);
            if (profile > poreField[vIdx]) {
              poreField[vIdx] = profile;
            }
          }
        }
      }
    }
  }

  const invN = 1 / N;
  const warpAmp = 0.055;
  for (let z = 0; z < N; z++) {
    const uz = (z + 0.5) * invN;
    const zOff = z * N2;
    for (let y = 0; y < N; y++) {
      const uy = (y + 0.5) * invN;
      const yzOff = zOff + y * N;
      for (let x = 0; x < N; x++) {
        const ux = (x + 0.5) * invN;
        const vIdx = yzOff + x;

        // Domain-warped periodic 3D Voronoi clasts
        const wx = samplePeriodicGradient(warpX, ux, uy, uz) * warpAmp;
        const wy = samplePeriodicGradient(warpY, ux, uy, uz) * warpAmp;
        const wz = samplePeriodicGradient(warpZ, ux, uy, uz) * warpAmp;

        const vx = ((ux + wx) % 1 + 1) % 1 * VM;
        const vy = ((uy + wy) % 1 + 1) % 1 * VM;
        const vz = ((uz + wz) % 1 + 1) % 1 * VM;

        const ix = Math.floor(vx);
        const iy = Math.floor(vy);
        const iz = Math.floor(vz);

        let d1Sq = 999;
        let d2Sq = 999;
        let closestTone = 0.5;

        for (let oz = -1; oz <= 1; oz++) {
          const cz = iz + oz;
          const czWrap = (cz + VM) & 7;
          const zBase = czWrap * VM2;
          for (let oy = -1; oy <= 1; oy++) {
            const cy = iy + oy;
            const cyWrap = (cy + VM) & 7;
            const yzBase = zBase + cyWrap * VM;
            for (let ox = -1; ox <= 1; ox++) {
              const cx = ix + ox;
              const cxWrap = (cx + VM) & 7;
              const cIdx = yzBase + cxWrap;
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
        const hairlineBlend = 1 - smoothstep(0.018, 0.068, edgeDist);
        const rVal = clamp01(closestTone * (1 - hairlineBlend * 0.72) + 0.10 * hairlineBlend);

        // True 3D spherical pores
        const gVal = clamp01(poreField[vIdx]);

        // 3-octave periodic sandstone grain mapped cleanly to [0.08..0.92]
        const n10 = samplePeriodicGradient(grain10, ux, uy, uz);
        const n19 = samplePeriodicGradient(grain19, ux, uy, uz);
        const n31 = samplePeriodicGradient(grain31, ux, uy, uz);
        const rawGrain = n10 * 0.45 + n19 * 0.35 + n31 * 0.20;
        const bVal = clamp01(0.5 + rawGrain * 0.62) * 0.84 + 0.08;

        // Smooth composite micro-relief height
        const aVal = clamp01(0.52 + 0.15 * (rVal - 0.5) - 0.44 * gVal + 0.26 * (bVal - 0.5));

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
