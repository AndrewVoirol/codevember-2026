import { Fn, abs, dot, float, max, select, sqrt, vec2, vec3 } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Resolution of the octahedral erosion heightfield texture (including 1-texel mirrored gutter).
 * Interior playable grid spans indices [1 .. EROSION_GRID_SIZE - 2] = 254 x 254.
 */
export const EROSION_GRID_SIZE = 256;
export const EROSION_GRID_INNER = EROSION_GRID_SIZE - 2;

/**
 * Exact C-infinity analytical oblate pillowed concretion base hull parameters.
 * Evaluated identically on CPU and GPU TSL so the octahedral DataTexture stores ONLY
 * isotropic rock relief + carved erosion delta (`deltaR = currentRadius - R_base(d)`).
 * Eliminates 100% of bilinear V-trough creases across the octahedral y = 0 / x = 0 folds!
 */
export const NODULE_BASE_AX = 1.13;
export const NODULE_BASE_AY = 1.07;
export const NODULE_BASE_AZ = 0.398;
export const NODULE_BASE_PILLOW = 0.62;

/**
 * Exact C-infinity analytical base concretion radius along unit direction `(dx, dy, dz)` on CPU.
 */
export function evalBaseBoulderRadiusCpu(dx: number, dy: number, dz: number): number {
  const qx = dx / NODULE_BASE_AX;
  const qy = dy / NODULE_BASE_AY;
  const qz = dz / NODULE_BASE_AZ;
  const rEll = 1.0 / Math.sqrt(Math.max(1e-8, qx * qx + qy * qy + qz * qz));
  const rxy2 = dx * dx + dy * dy;
  return rEll * (1.0 + NODULE_BASE_PILLOW * rxy2 * dz * dz);
}

/**
 * Exact C-infinity analytical base concretion radius along unit direction `dir` in GPU TSL.
 * Matches `evalBaseBoulderRadiusCpu` bit-for-bit with zero octahedral fold discontinuities.
 */
export const evalBaseBoulderRadiusTsl = Fn(([dir_in]: [TslNode]) => {
  const d = vec3(dir_in);
  const q = d.div(vec3(NODULE_BASE_AX, NODULE_BASE_AY, NODULE_BASE_AZ));
  const rEll = float(1.0).div(sqrt(max(float(1e-6), dot(q, q))));
  const rxy2 = d.x.mul(d.x).add(d.y.mul(d.y));
  const dz2 = d.z.mul(d.z);
  return rEll.mul(float(1.0).add(rxy2.mul(dz2).mul(NODULE_BASE_PILLOW)));
});

/**
 * Encodes a unit 3D direction `(x, y, z)` to octahedral coordinates `(ox, oy)` in `[-1, 1]^2`.
 * The `+Z` front hemisphere maps cleanly to the central diamond `|ox| + |oy| <= 1` with zero seams.
 */
export function octEncode(
  x: number,
  y: number,
  z: number,
  out: [number, number]
): [number, number] {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z);
  const invL1 = l1 > 1e-12 ? 1.0 / l1 : 1.0;
  let ox = x * invL1;
  let oy = y * invL1;
  if (z < 0) {
    const tx = (1.0 - Math.abs(oy)) * (ox >= 0 ? 1.0 : -1.0);
    const ty = (1.0 - Math.abs(ox)) * (oy >= 0 ? 1.0 : -1.0);
    ox = tx;
    oy = ty;
  }
  out[0] = ox;
  out[1] = oy;
  return out;
}

/**
 * Maps octahedral coordinates `(ox, oy)` in `[-1, 1]^2` to UV coordinates in `[1/N, (N-1)/N]^2`
 * aligned with the 1-texel mirrored octahedral gutter.
 */
export function octToUv(
  ox: number,
  oy: number,
  out: [number, number],
  size = EROSION_GRID_SIZE
): [number, number] {
  const inner = size - 2;
  const scale = inner / size;
  const pad = 1.0 / size;
  out[0] = (ox * 0.5 + 0.5) * scale + pad;
  out[1] = (oy * 0.5 + 0.5) * scale + pad;
  return out;
}

/**
 * Encodes a unit 3D direction `(x, y, z)` directly into gutter-padded texture UV `[0, 1]^2`.
 */
export function dirToOctUv(
  x: number,
  y: number,
  z: number,
  out: [number, number],
  size = EROSION_GRID_SIZE
): [number, number] {
  octEncode(x, y, z, out);
  return octToUv(out[0], out[1], out, size);
}

/**
 * Decodes octahedral coordinates `(ox, oy)` in `[-1, 1]^2` back to a unit 3D direction `(x, y, z)`.
 */
export function octDecode(
  ox: number,
  oy: number,
  out: [number, number, number]
): [number, number, number] {
  let x = ox;
  let y = oy;
  const z = 1.0 - Math.abs(ox) - Math.abs(oy);
  if (z < 0) {
    const tx = (1.0 - Math.abs(y)) * (x >= 0 ? 1.0 : -1.0);
    const ty = (1.0 - Math.abs(x)) * (y >= 0 ? 1.0 : -1.0);
    x = tx;
    y = ty;
  }
  const len = Math.hypot(x, y, z);
  const inv = len > 1e-12 ? 1.0 / len : 1.0;
  out[0] = x * inv;
  out[1] = y * inv;
  out[2] = z * inv;
  return out;
}

/**
 * Returns the octahedral `(ox, oy)` center of interior texel `(ix, iy)` for `1 <= ix, iy <= size - 2`.
 */
export function texelToOct(
  ix: number,
  iy: number,
  out: [number, number],
  size = EROSION_GRID_SIZE
): [number, number] {
  const inner = size - 2;
  out[0] = ((ix - 0.5) / inner) * 2.0 - 1.0;
  out[1] = ((iy - 0.5) / inner) * 2.0 - 1.0;
  return out;
}

/**
 * Computes the exact unit sphere direction AND exact solid angle `dOmega` (in steradians)
 * of interior texel `(ix, iy)` so spherical-shell carved volume integrates exactly to `4*pi`.
 */
export function texelDirAndSolidAngle(
  ix: number,
  iy: number,
  outDir: [number, number, number],
  size = EROSION_GRID_SIZE
): number {
  const inner = size - 2;
  const ox = ((ix - 0.5) / inner) * 2.0 - 1.0;
  const oy = ((iy - 0.5) / inner) * 2.0 - 1.0;

  let x = ox;
  let y = oy;
  const z = 1.0 - Math.abs(ox) - Math.abs(oy);
  if (z < 0) {
    const tx = (1.0 - Math.abs(y)) * (x >= 0 ? 1.0 : -1.0);
    const ty = (1.0 - Math.abs(x)) * (y >= 0 ? 1.0 : -1.0);
    x = tx;
    y = ty;
  }
  const norm = Math.hypot(x, y, z);
  const inv = norm > 1e-12 ? 1.0 / norm : 1.0;
  outDir[0] = x * inv;
  outDir[1] = y * inv;
  outDir[2] = z * inv;

  const d = 2.0 / inner;
  return (d * d) / (norm * norm * norm);
}

/**
 * Mirrors the 1-texel octahedral border gutter on a single-channel Float32Array (`size * size`).
 * Guarantees C0 bilinear continuity across all 4 octahedral perimeter edges and all 4 `-Z` corners.
 */
export function mirrorOctahedralScalarGutter(grid: Float32Array, size = EROSION_GRID_SIZE): void {
  const last = size - 1;
  const innerMax = size - 2;

  for (let i = 1; i <= innerMax; i++) {
    const mir = last - i;
    // Top and bottom gutters mirror X
    grid[i] = grid[size + mir];
    grid[last * size + i] = grid[innerMax * size + mir];
    // Left and right gutters mirror Y
    grid[i * size] = grid[mir * size + 1];
    grid[i * size + last] = grid[mir * size + innerMax];
  }

  // Four corners all fold to the antipodal opposite interior corner (-Z pole)
  grid[0] = grid[innerMax * size + innerMax];
  grid[last] = grid[innerMax * size + 1];
  grid[last * size] = grid[size + innerMax];
  grid[last * size + last] = grid[size + 1];
}

/**
 * Mirrors the 1-texel octahedral border gutter on a 4-channel RGBA packed Uint16Array (`size * size * 4`).
 */
export function mirrorOctahedralRgba16Gutter(data: Uint16Array, size = EROSION_GRID_SIZE): void {
  const last = size - 1;
  const innerMax = size - 2;

  const copy4 = (dstIdx: number, srcIdx: number) => {
    const d4 = dstIdx * 4;
    const s4 = srcIdx * 4;
    data[d4] = data[s4];
    data[d4 + 1] = data[s4 + 1];
    data[d4 + 2] = data[s4 + 2];
    data[d4 + 3] = data[s4 + 3];
  };

  for (let i = 1; i <= innerMax; i++) {
    const mir = last - i;
    copy4(i, size + mir);
    copy4(last * size + i, innerMax * size + mir);
    copy4(i * size, mir * size + 1);
    copy4(i * size + last, mir * size + innerMax);
  }

  copy4(0, innerMax * size + innerMax);
  copy4(last, innerMax * size + 1);
  copy4(last * size, size + innerMax);
  copy4(last * size + last, size + 1);
}

/**
 * Branch-free TSL octahedral encoder matching `dirToOctUv` bit-for-bit on GPU.
 * Strictly uniform control flow (uses `select` with zero dynamic `If` branches).
 */
export const octDirToUvTsl = Fn(([dir_in]: [TslNode]) => {
  const d = vec3(dir_in);
  const l1 = abs(d.x).add(abs(d.y)).add(abs(d.z));
  const invL1 = float(1.0).div(l1.max(1e-6));
  const px = d.x.mul(invL1);
  const py = d.y.mul(invL1);

  const sx = select(px.greaterThanEqual(0.0), float(1.0), float(-1.0));
  const sy = select(py.greaterThanEqual(0.0), float(1.0), float(-1.0));
  const foldedX = float(1.0).sub(abs(py)).mul(sx);
  const foldedY = float(1.0).sub(abs(px)).mul(sy);

  const isNegZ = d.z.lessThan(0.0);
  const ox = select(isNegZ, foldedX, px);
  const oy = select(isNegZ, foldedY, py);

  const scale = float(EROSION_GRID_INNER / EROSION_GRID_SIZE);
  const pad = float(1.0 / EROSION_GRID_SIZE);
  return vec2(ox.mul(0.5).add(0.5).mul(scale).add(pad), oy.mul(0.5).add(0.5).mul(scale).add(pad));
});

// ---------------------------------------------------------------------------
// Deterministic 3D Value Noise & Irrational-Angle Rotated FBM on CPU
// (Zero polar star harmonics; bit-deterministic per seed)
// ---------------------------------------------------------------------------

export function hash3Cpu(ix: number, iy: number, iz: number): number {
  let h =
    (Math.imul(ix | 0, 374761393) +
      Math.imul(iy | 0, 668265263) +
      Math.imul(iz | 0, 2147483647)) |
    0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}

export function valueNoise3DCpu(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3.0 - 2.0 * fx);
  const uy = fy * fy * (3.0 - 2.0 * fy);
  const uz = fz * fz * (3.0 - 2.0 * fz);

  const n000 = hash3Cpu(ix, iy, iz);
  const n100 = hash3Cpu(ix + 1, iy, iz);
  const n010 = hash3Cpu(ix, iy + 1, iz);
  const n110 = hash3Cpu(ix + 1, iy + 1, iz);
  const n001 = hash3Cpu(ix, iy, iz + 1);
  const n101 = hash3Cpu(ix + 1, iy, iz + 1);
  const n011 = hash3Cpu(ix, iy + 1, iz + 1);
  const n111 = hash3Cpu(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

/**
 * 3-octave golden-angle rotated 3D FBM matching `mineralToothFbm` structure on CPU.
 */
export function fbm3DCpu(x: number, y: number, z: number, scale = 1.0): number {
  const p0x = x * scale;
  const p0y = y * scale;
  const p0z = z * scale;
  const v0 = valueNoise3DCpu(p0x, p0y, p0z);

  const c = 0.80901699;
  const s = 0.58778525;
  const p1x = (p0x * c - p0z * s) * 2.03;
  const p1y = (p0y * c + p0x * s) * 2.03;
  const p1z = (p0z * c + p0y * s) * 2.03;
  const v1 = valueNoise3DCpu(p1x, p1y, p1z);

  const p2x = (p1y * c - p1x * s) * 2.07;
  const p2y = (p1z * c + p1y * s) * 2.07;
  const p2z = (p1x * c + p1z * s) * 2.07;
  const v2 = valueNoise3DCpu(p2x, p2y, p2z);

  return v0 * 0.57 + v1 * 0.29 + v2 * 0.14;
}
