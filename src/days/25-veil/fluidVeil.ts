import * as THREE from 'three/webgpu';

export const FLUID_W = 128;
export const FLUID_H = 128;
export const FLUID_CELLS = FLUID_W * FLUID_H;

const TWO_PI = Math.PI * 2;
const MAX_VEL = 5.2;
const MAX_CONF_CURL = 55.0;
const PRESSURE_ITERS = 18;

// Pre-allocated bit-cast views for zero-GC IEEE-754 float32 -> float16 packing
const _f32View = new Float32Array(1);
const _u32View = new Uint32Array(_f32View.buffer);

/**
 * Fast IEEE-754 float32 -> float16 encoder (zero heap allocations).
 */
export function encodeHalfFloat(val: number): number {
  _f32View[0] = Number.isFinite(val) ? val : 0;
  const x = _u32View[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 112;
  let mant = x & 0x007fffff;

  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x00800000) >>> (1 - exp);
    if (mant & 0x00001000) mant += 0x00002000;
    return sign | (mant >>> 13);
  }
  if (exp >= 31) {
    return sign | 0x7bff; // clamp finite max to avoid Inf/NaN in half-float textures
  }
  if (mant & 0x00001000) {
    mant += 0x00002000;
    if (mant & 0x00800000) {
      mant = 0;
      exp += 1;
      if (exp >= 31) return sign | 0x7bff;
    }
  }
  return sign | (exp << 10) | (mant >>> 13);
}

export interface FluidVeilTelemetry {
  maxVorticity: number;
  meanDensity: number;
  tearEnergy: number;
}

/**
 * Pure 2D Incompressible Navier-Stokes + Vorticity Confinement + Clamped MacCormack
 * Haze Density Solver (`128 x 128`) for Day 25 - Veil.
 *
 * Strictly zero per-frame heap allocations. Outputs an `RGBA16F` `DataTexture` packing:
 *   - `R`: optical haze density `d in [0, 1]`
 *   - `G`: horizontal velocity `u` (curtain widths / sec)
 *   - `B`: vertical velocity `v` (curtain heights / sec, +v downward)
 *   - `A`: signed vorticity `curl = dv/du - du/dv` (1 / sec)
 */
export class FluidVeilSolver {
  readonly width = FLUID_W;
  readonly height = FLUID_H;

  readonly u = new Float32Array(FLUID_CELLS);
  readonly v = new Float32Array(FLUID_CELLS);
  readonly density = new Float32Array(FLUID_CELLS);
  readonly curl = new Float32Array(FLUID_CELLS);

  private readonly uTmp = new Float32Array(FLUID_CELLS);
  private readonly vTmp = new Float32Array(FLUID_CELLS);
  private readonly dTmp = new Float32Array(FLUID_CELLS);
  private readonly dAux = new Float32Array(FLUID_CELLS);
  private readonly div = new Float32Array(FLUID_CELLS);
  private readonly p = new Float32Array(FLUID_CELLS);

  /** Packed RGBA16F texture payload (`128 * 128 * 4` uint16 half-floats). */
  readonly packedTextureData = new Uint16Array(FLUID_CELLS * 4);
  readonly dataTexture: THREE.DataTexture;

  /** Enable seamless horizontal periodic boundary wrap (used in Hollow Cone mode). */
  periodicX = false;
  simTime = 0;

  maxVorticity = 0;
  meanDensity = 0.56;
  tearEnergy = 0;

  // Tunable physical parameters
  readonly vorticityConfinement = 0.36; // dimensionless Fedkiw-Stam-Jensen epsilon (multiplied by dx = 1/W)
  readonly velocityDamping = 0.78;
  readonly densityHealRate = 0.36;
  readonly densityDiffusion = 0.00018;

  constructor() {
    this.dataTexture = new THREE.DataTexture(
      this.packedTextureData,
      FLUID_W,
      FLUID_H,
      THREE.RGBAFormat,
      THREE.HalfFloatType
    );
    this.dataTexture.minFilter = THREE.LinearFilter;
    this.dataTexture.magFilter = THREE.LinearFilter;
    this.dataTexture.wrapS = THREE.RepeatWrapping;
    this.dataTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.dataTexture.generateMipmaps = false;

    this.resetToLaminar();
  }

  /**
   * Evaluates smooth drifting laminar studio mist target density at normalized `(u, v)`
   * and simulation time `t`. Periodic in `u` so Cone mode has zero seam at `u = 0 | 1`.
   */
  evalBaselineDensity(uNorm: number, vNorm: number, t: number): number {
    const phi = TWO_PI * uNorm;
    // Smooth multi-scale drifting silk-like mist ribbons (strictly zero polar star harmonics)
    const w1 = Math.sin(phi * 1.0 + vNorm * 2.1 - t * 0.24 + 0.4);
    const w2 = Math.sin(phi * 2.0 - vNorm * 3.1 + t * 0.17 + 1.6);
    const w3 = Math.sin(phi * 3.0 + vNorm * 4.4 - t * 0.31 + 2.9);
    const w4 = Math.sin(phi * 5.0 - vNorm * 2.4 + t * 0.13 + 0.9);
    const ribbon = 0.42 * w1 + 0.28 * w2 + 0.19 * w3 + 0.11 * w4;

    // Gentle vertical optical density envelope (richest beneath the slit lamp, silkiest near bottom)
    const verticalEnv = 0.64 - 0.14 * vNorm + 0.06 * Math.sin(vNorm * Math.PI);
    const d = verticalEnv + 0.16 * ribbon;
    return d < 0.18 ? 0.18 : d > 0.88 ? 0.88 : d;
  }

  /**
   * Initializes the veil in a calm laminar convective state with one subtle graceful curl
   * so Frame 0 immediately reads as living sunlit studio mist.
   */
  resetToLaminar(): void {
    this.simTime = 0;
    this.u.fill(0);
    this.v.fill(0);
    this.p.fill(0);
    this.div.fill(0);
    this.curl.fill(0);

    for (let iy = 0; iy < FLUID_H; iy++) {
      const vNorm = (iy + 0.5) / FLUID_H;
      const row = iy * FLUID_W;
      for (let ix = 0; ix < FLUID_W; ix++) {
        const uNorm = (ix + 0.5) / FLUID_W;
        this.density[row + ix] = this.evalBaselineDensity(uNorm, vNorm, 0);
      }
    }

    // Seed gentle initial studio convective eddies and step 12 frames so wisps look naturally curled
    this.injectPointerSweep(0.24, 0.32, 0.42, 0.36, 0.55, 0.08, false, 0.09);
    this.injectPointerSweep(0.76, 0.58, 0.58, 0.52, -0.48, -0.1, false, 0.09);
    for (let s = 0; s < 12; s++) {
      this.step(1 / 60, false);
    }
    this.packToTexture();
  }

  /**
   * Injects high-vorticity fluid momentum, dipole shear, and mist parting along a swept
   * pointer segment `(u0, v0) -> (u1, v1)` on the curtain surface.
   */
  injectPointerSweep(
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    velU: number,
    velV: number,
    isDrag: boolean,
    radiusOverride?: number
  ): void {
    const cu0 = Math.max(0, Math.min(1, u0));
    const cv0 = Math.max(0, Math.min(1, v0));
    let cu1 = Math.max(0, Math.min(1, u1));
    const cv1 = Math.max(0, Math.min(1, v1));

    // Handle periodic cone-seam wrap cleanly without streaking across the entire curtain
    if (this.periodicX) {
      const duWrap = cu1 - cu0;
      if (duWrap > 0.5) cu1 -= 1;
      else if (duWrap < -0.5) cu1 += 1;
    }

    const segX = cu1 - cu0;
    const segY = cv1 - cv0;
    const segLen = Math.hypot(segX, segY);

    // Clamp injected impulse speed so wild cursor jumps stay physically graceful
    const rawSpeed = Math.hypot(velU, velV);
    const maxInject = isDrag ? 4.4 : 2.6;
    const scale = rawSpeed > maxInject ? maxInject / rawSpeed : 1.0;
    const vxInj = velU * scale;
    const vyInj = velV * scale;
    const speed = rawSpeed * scale;

    if (speed < 0.015 && !isDrag) return;

    const dirX = speed > 1e-4 ? vxInj / speed : segLen > 1e-4 ? segX / segLen : 1;
    const dirY = speed > 1e-4 ? vyInj / speed : segLen > 1e-4 ? segY / segLen : 0;
    // Perpendicular vector for twin counter-rotating Kelvin-Helmholtz shear flanks
    const perpX = -dirY;
    const perpY = dirX;

    const radius = radiusOverride ?? (isDrag ? 0.072 : 0.056);
    const invR2 = 1 / (radius * radius);
    const maxR = radius * 2.6;

    const minV = Math.max(0, Math.floor((Math.min(cv0, cv1) - maxR) * FLUID_H));
    const maxV = Math.min(FLUID_H - 1, Math.ceil((Math.max(cv0, cv1) + maxR) * FLUID_H));

    const minURaw = Math.floor((Math.min(cu0, cu1) - maxR) * FLUID_W);
    const maxURaw = Math.ceil((Math.max(cu0, cu1) + maxR) * FLUID_W);
    const startU = this.periodicX ? minURaw : Math.max(0, minURaw);
    const endU = this.periodicX ? maxURaw : Math.min(FLUID_W - 1, maxURaw);

    const segLenSq = segX * segX + segY * segY;
    const momGain = isDrag ? 0.58 : 0.36;
    const shearGain = (isDrag ? 0.42 : 0.24) * Math.min(2.8, speed + (isDrag ? 0.55 : 0.1));
    const carveGain = isDrag ? 0.52 : Math.min(0.28, speed * 0.14);
    const bankGain = isDrag ? 0.14 : Math.min(0.08, speed * 0.045);

    for (let iy = minV; iy <= maxV; iy++) {
      const py = (iy + 0.5) / FLUID_H;
      const row = iy * FLUID_W;
      for (let ixRaw = startU; ixRaw <= endU; ixRaw++) {
        const px = (ixRaw + 0.5) / FLUID_W;
        let t = 0;
        if (segLenSq > 1e-7) {
          t = ((px - cu0) * segX + (py - cv0) * segY) / segLenSq;
          if (t < 0) t = 0;
          else if (t > 1) t = 1;
        }
        const dx = px - (cu0 + t * segX);
        const dy = py - (cv0 + t * segY);
        const d2 = dx * dx + dy * dy;
        if (d2 > maxR * maxR) continue;

        const w = Math.exp(-d2 * invR2);
        const ix = this.periodicX ? ((ixRaw % FLUID_W) + FLUID_W) % FLUID_W : ixRaw;
        const idx = row + ix;

        // Signed lateral offset across swept wake -> drives twin counter-rotating vortex flanks
        const lat = (dx * perpX + dy * perpY) / radius;
        const flankEnv = Math.exp(-0.55 * lat * lat) * Math.exp(-(d2 * invR2) * 0.45);
        // Outward lateral push + shear backflow at edges of wake
        const shearU = perpX * lat * flankEnv * shearGain * 0.55;
        const shearV = perpY * lat * flankEnv * shearGain * 0.55;

        let nu = this.u[idx] + vxInj * w * momGain + shearU;
        let nv = this.v[idx] + vyInj * w * momGain + shearV;
        const mag = Math.hypot(nu, nv);
        if (mag > MAX_VEL) {
          const clampS = MAX_VEL / mag;
          nu *= clampS;
          nv *= clampS;
        }
        this.u[idx] = nu;
        this.v[idx] = nv;

        // Carve crisp dark opening along swept path core while banking mist into sheared flanks
        const coreMask = Math.exp(-d2 * invR2 * 1.85);
        const ringMask = Math.exp(-Math.pow(Math.abs(lat) - 1.15, 2) * 2.1) * Math.exp(-d2 * invR2 * 0.35);
        let d = this.density[idx];
        d = d * (1 - carveGain * coreMask) + bankGain * ringMask * (1 - d * 0.45);
        this.density[idx] = d < 0 ? 0 : d > 1 ? 1 : d;
      }
    }
  }

  /**
   * Injects a gentle downward optical-shutter ripple along the top slit when aperture modes cycle.
   */
  injectAperturePulse(): void {
    for (let iy = 0; iy < 22; iy++) {
      const vNorm = (iy + 0.5) / FLUID_H;
      const env = Math.exp(-Math.pow((vNorm - 0.055) / 0.055, 2));
      const row = iy * FLUID_W;
      for (let ix = 0; ix < FLUID_W; ix++) {
        const uNorm = (ix + 0.5) / FLUID_W;
        const wave = Math.sin(TWO_PI * uNorm * 3.0 + this.simTime * 1.7);
        const idx = row + ix;
        this.v[idx] += 0.42 * env * (0.7 + 0.3 * wave);
        this.u[idx] += 0.24 * env * Math.cos(TWO_PI * uNorm * 2.0 - this.simTime);
      }
    }
  }

  /**
   * Advances the incompressible Navier-Stokes velocity field, vorticity confinement,
   * pressure projection, and clamped MacCormack haze density by `dtSec`.
   */
  step(dtSec: number, uploadTexture = true): FluidVeilTelemetry {
    const dt = Math.min(0.033, Math.max(0.001, dtSec));
    this.simTime += dt;
    const t = this.simTime;

    // 1. Subtle divergence-free laminar thermal convection & viscous damping
    const damp = Math.exp(-dt * this.velocityDamping);
    const convGain = dt * 0.16;
    for (let iy = 0; iy < FLUID_H; iy++) {
      const vNorm = (iy + 0.5) / FLUID_H;
      const vSin = Math.sin(Math.PI * vNorm);
      const row = iy * FLUID_W;
      for (let ix = 0; ix < FLUID_W; ix++) {
        const uNorm = (ix + 0.5) / FLUID_W;
        const idx = row + ix;
        const phi = TWO_PI * uNorm;

        // Gentle studio thermal updraft + slow lateral convection drift
        const ambU =
          (Math.cos(phi * 1.0 + vNorm * 2.4 - t * 0.28) * 0.55 +
            Math.sin(phi * 2.0 - vNorm * 3.1 + t * 0.19) * 0.32) *
          vSin;
        const ambV =
          (-0.085 -
            0.09 * Math.sin(phi * 2.0 + vNorm * 2.0 - t * 0.22) +
            0.06 * Math.cos(phi * 3.0 - vNorm * 3.6 + t * 0.31)) *
          vSin;

        this.u[idx] = this.u[idx] * damp + ambU * convGain;
        this.v[idx] = this.v[idx] * damp + ambV * convGain;
      }
    }

    // 2. Vorticity calculation & Fedkiw-Stam-Jensen vorticity confinement
    this.computeVorticityAndConfine(dt);

    // 3. Semi-Lagrangian velocity advection (paired with vorticity confinement for smooth silky curls)
    this.advectSemiLagrangian(this.u, this.u, this.v, this.uTmp, dt, -MAX_VEL, MAX_VEL);
    this.advectSemiLagrangian(this.v, this.u, this.v, this.vTmp, dt, -MAX_VEL, MAX_VEL);
    this.u.set(this.uTmp);
    this.v.set(this.vTmp);

    // 4. Incompressible Red-Black Gauss-Seidel pressure projection (div u = 0)
    this.projectIncompressible();

    // 5. Recompute post-projection curl for exact rendering & telemetry
    let maxVort = 0;
    const W = FLUID_W;
    const H = FLUID_H;
    const wrapX = this.periodicX;
    for (let iy = 0; iy < H; iy++) {
      const yU = iy > 0 ? iy - 1 : 0;
      const yD = iy < H - 1 ? iy + 1 : H - 1;
      const row = iy * W;
      const rowU = yU * W;
      const rowD = yD * W;
      for (let ix = 0; ix < W; ix++) {
        const xL = ix > 0 ? ix - 1 : wrapX ? W - 1 : 0;
        const xR = ix < W - 1 ? ix + 1 : wrapX ? 0 : W - 1;
        const wVal = 0.5 * (this.v[row + xR] - this.v[row + xL] - (this.u[rowD + ix] - this.u[rowU + ix])) * W;
        this.curl[row + ix] = wVal;
        const absW = wVal < 0 ? -wVal : wVal;
        if (absW > maxVort) maxVort = absW;
      }
    }

    // 6. Clamped MacCormack scalar haze density advection (strictly preserves [0, 1] mass bounds!)
    this.advectMacCormack(this.density, this.u, this.v, this.dTmp, this.dAux, dt, 0, 1);

    // 7. Gentle laminar mist healing & micro-diffusion + telemetry reduction
    const healAlpha = 1 - Math.exp(-dt * this.densityHealRate);
    const diffAlpha = Math.min(0.18, this.densityDiffusion * dt * FLUID_W * FLUID_H * 0.25);
    let sumD = 0;
    let sumTear = 0;

    for (let iy = 0; iy < H; iy++) {
      const vNorm = (iy + 0.5) / H;
      const yU = iy > 0 ? iy - 1 : 0;
      const yD = iy < H - 1 ? iy + 1 : H - 1;
      const row = iy * W;
      const rowU = yU * W;
      const rowD = yD * W;
      for (let ix = 0; ix < W; ix++) {
        const uNorm = (ix + 0.5) / W;
        const xL = ix > 0 ? ix - 1 : wrapX ? W - 1 : 0;
        const xR = ix < W - 1 ? ix + 1 : wrapX ? 0 : W - 1;
        const idx = row + ix;

        const lap =
          this.dTmp[row + xL] +
          this.dTmp[row + xR] +
          this.dTmp[rowU + ix] +
          this.dTmp[rowD + ix] -
          4 * this.dTmp[idx];
        const targetD = this.evalBaselineDensity(uNorm, vNorm, t);
        let d = this.dTmp[idx] + lap * diffAlpha;
        d = d + (targetD - d) * healAlpha;
        if (d < 0) d = 0;
        else if (d > 1) d = 1;
        this.density[idx] = d;

        sumD += d;
        const diff = d - targetD;
        const ke = this.u[idx] * this.u[idx] + this.v[idx] * this.v[idx];
        sumTear += diff * diff * 4.5 + ke * 0.14;
      }
    }

    this.maxVorticity = maxVort;
    this.meanDensity = sumD / FLUID_CELLS;
    this.tearEnergy = Math.min(10, sumTear / FLUID_CELLS);

    if (uploadTexture) {
      this.packToTexture();
    }

    return {
      maxVorticity: this.maxVorticity,
      meanDensity: this.meanDensity,
      tearEnergy: this.tearEnergy,
    };
  }

  private computeVorticityAndConfine(dt: number): void {
    const W = FLUID_W;
    const H = FLUID_H;
    const wrapX = this.periodicX;

    for (let iy = 0; iy < H; iy++) {
      const yU = iy > 0 ? iy - 1 : 0;
      const yD = iy < H - 1 ? iy + 1 : H - 1;
      const row = iy * W;
      const rowU = yU * W;
      const rowD = yD * W;
      for (let ix = 0; ix < W; ix++) {
        const xL = ix > 0 ? ix - 1 : wrapX ? W - 1 : 0;
        const xR = ix < W - 1 ? ix + 1 : wrapX ? 0 : W - 1;
        this.curl[row + ix] =
          0.5 * (this.v[row + xR] - this.v[row + xL] - (this.u[rowD + ix] - this.u[rowU + ix])) * W;
      }
    }

    // Exact physical Fedkiw-Stam-Jensen scaling: f = eps * dx * (N x omega), where dx = 1 / W
    const eps = (this.vorticityConfinement / W) * dt;
    for (let iy = 1; iy < H - 1; iy++) {
      const row = iy * W;
      const rowU = (iy - 1) * W;
      const rowD = (iy + 1) * W;
      for (let ix = 0; ix < W; ix++) {
        if (!wrapX && (ix === 0 || ix === W - 1)) continue;
        const xL = ix > 0 ? ix - 1 : W - 1;
        const xR = ix < W - 1 ? ix + 1 : 0;
        const idx = row + ix;

        const wL = Math.abs(this.curl[row + xL]);
        const wR = Math.abs(this.curl[row + xR]);
        const wU = Math.abs(this.curl[rowU + ix]);
        const wD = Math.abs(this.curl[rowD + ix]);

        const gx = 0.5 * (wR - wL);
        const gy = 0.5 * (wD - wU);
        const gLen = Math.hypot(gx, gy) + 1e-5;
        const nx = gx / gLen;
        const ny = gy / gLen;

        let wLocal = this.curl[idx];
        if (wLocal > MAX_CONF_CURL) wLocal = MAX_CONF_CURL;
        else if (wLocal < -MAX_CONF_CURL) wLocal = -MAX_CONF_CURL;

        this.u[idx] += eps * ny * wLocal;
        this.v[idx] -= eps * nx * wLocal;
      }
    }
  }

  private sampleBilinear(field: Float32Array, gx: number, gy: number): number {
    const W = FLUID_W;
    const H = FLUID_H;

    let x = gx;
    if (this.periodicX) {
      x = ((x % W) + W) % W;
    } else {
      if (x < 0.5) x = 0.5;
      else if (x > W - 1.5) x = W - 1.5;
    }
    let y = gy;
    if (y < 0.5) y = 0.5;
    else if (y > H - 1.5) y = H - 1.5;

    const ix0 = Math.floor(x);
    const iy0 = Math.floor(y);
    const fx = x - ix0;
    const fy = y - iy0;

    const ix1 = this.periodicX ? (ix0 + 1) % W : Math.min(W - 1, ix0 + 1);
    const iy1 = Math.min(H - 1, iy0 + 1);

    const r0 = iy0 * W;
    const r1 = iy1 * W;
    const v00 = field[r0 + ix0];
    const v10 = field[r0 + ix1];
    const v01 = field[r1 + ix0];
    const v11 = field[r1 + ix1];

    const a = v00 + (v10 - v00) * fx;
    const b = v01 + (v11 - v01) * fx;
    return a + (b - a) * fy;
  }

  private advectSemiLagrangian(
    src: Float32Array,
    velU: Float32Array,
    velV: Float32Array,
    dst: Float32Array,
    dt: number,
    minVal: number,
    maxVal: number
  ): void {
    const W = FLUID_W;
    const H = FLUID_H;
    const scaleX = dt * W;
    const scaleY = dt * H;

    for (let iy = 0; iy < H; iy++) {
      const row = iy * W;
      for (let ix = 0; ix < W; ix++) {
        const idx = row + ix;
        const bx = ix - velU[idx] * scaleX;
        const by = iy - velV[idx] * scaleY;
        let val = this.sampleBilinear(src, bx, by);
        if (val < minVal) val = minVal;
        else if (val > maxVal) val = maxVal;
        dst[idx] = val;
      }
    }
  }

  /**
   * Unconditionally stable, low-diffusion Clamped MacCormack (BFECC) advection.
   * Local 4-neighbor min/max clamping guarantees strict preservation of `[minVal, maxVal]`
   * and zero numerical ringing while keeping smoke vortex filaments razor crisp.
   */
  private advectMacCormack(
    src: Float32Array,
    velU: Float32Array,
    velV: Float32Array,
    dst: Float32Array,
    aux: Float32Array,
    dt: number,
    minVal: number,
    maxVal: number
  ): void {
    const W = FLUID_W;
    const H = FLUID_H;
    const scaleX = dt * W;
    const scaleY = dt * H;
    const wrapX = this.periodicX;

    // Forward Semi-Lagrangian pass: aux = SL(src, -dt)
    for (let iy = 0; iy < H; iy++) {
      const row = iy * W;
      for (let ix = 0; ix < W; ix++) {
        const idx = row + ix;
        const bx = ix - velU[idx] * scaleX;
        const by = iy - velV[idx] * scaleY;
        aux[idx] = this.sampleBilinear(src, bx, by);
      }
    }

    // Backward error-correction pass with strict local min/max clamp
    for (let iy = 0; iy < H; iy++) {
      const row = iy * W;
      for (let ix = 0; ix < W; ix++) {
        const idx = row + ix;
        const bx = ix - velU[idx] * scaleX;
        const by = iy - velV[idx] * scaleY;

        const fxPos = ix + velU[idx] * scaleX;
        const fyPos = iy + velV[idx] * scaleY;
        const backSample = this.sampleBilinear(aux, fxPos, fyPos);
        let corrected = aux[idx] + 0.5 * (src[idx] - backSample);

        // Local 2x2 donor cell extrema clamp around backtrace coordinate `(bx, by)`
        let cx = bx;
        if (wrapX) {
          cx = ((cx % W) + W) % W;
        } else {
          if (cx < 0.5) cx = 0.5;
          else if (cx > W - 1.5) cx = W - 1.5;
        }
        let cy = by;
        if (cy < 0.5) cy = 0.5;
        else if (cy > H - 1.5) cy = H - 1.5;

        const i0 = Math.floor(cx);
        const j0 = Math.floor(cy);
        const i1 = wrapX ? (i0 + 1) % W : Math.min(W - 1, i0 + 1);
        const j1 = Math.min(H - 1, j0 + 1);
        const r0 = j0 * W;
        const r1 = j1 * W;

        const q00 = src[r0 + i0];
        const q10 = src[r0 + i1];
        const q01 = src[r1 + i0];
        const q11 = src[r1 + i1];

        let lo = q00 < q10 ? q00 : q10;
        if (q01 < lo) lo = q01;
        if (q11 < lo) lo = q11;

        let hi = q00 > q10 ? q00 : q10;
        if (q01 > hi) hi = q01;
        if (q11 > hi) hi = q11;

        if (corrected < lo) corrected = lo;
        else if (corrected > hi) corrected = hi;
        if (corrected < minVal) corrected = minVal;
        else if (corrected > maxVal) corrected = maxVal;

        dst[idx] = corrected;
      }
    }
  }

  private projectIncompressible(): void {
    const W = FLUID_W;
    const H = FLUID_H;
    const wrapX = this.periodicX;

    // Divergence & warm-start pressure decay
    for (let iy = 0; iy < H; iy++) {
      const yU = iy > 0 ? iy - 1 : 0;
      const yD = iy < H - 1 ? iy + 1 : H - 1;
      const row = iy * W;
      const rowU = yU * W;
      const rowD = yD * W;
      for (let ix = 0; ix < W; ix++) {
        const xL = ix > 0 ? ix - 1 : wrapX ? W - 1 : 0;
        const xR = ix < W - 1 ? ix + 1 : wrapX ? 0 : W - 1;
        const idx = row + ix;
        this.div[idx] =
          -0.5 * (this.u[row + xR] - this.u[row + xL] + (this.v[rowD + ix] - this.v[rowU + ix]));
        this.p[idx] *= 0.82;
      }
    }

    // Red-Black Gauss-Seidel Poisson solve
    for (let iter = 0; iter < PRESSURE_ITERS; iter++) {
      for (let color = 0; color < 2; color++) {
        for (let iy = 0; iy < H; iy++) {
          const yU = iy > 0 ? iy - 1 : 0;
          const yD = iy < H - 1 ? iy + 1 : H - 1;
          const row = iy * W;
          const rowU = yU * W;
          const rowD = yD * W;
          const startX = (iy + color) & 1;
          for (let ix = startX; ix < W; ix += 2) {
            const xL = ix > 0 ? ix - 1 : wrapX ? W - 1 : 0;
            const xR = ix < W - 1 ? ix + 1 : wrapX ? 0 : W - 1;
            const idx = row + ix;
            this.p[idx] =
              0.25 *
              (this.div[idx] +
                this.p[row + xL] +
                this.p[row + xR] +
                this.p[rowU + ix] +
                this.p[rowD + ix]);
          }
        }
      }
    }

    // Subtract pressure gradient & enforce gentle edge damping
    for (let iy = 0; iy < H; iy++) {
      const yU = iy > 0 ? iy - 1 : 0;
      const yD = iy < H - 1 ? iy + 1 : H - 1;
      const row = iy * W;
      const rowU = yU * W;
      const rowD = yD * W;
      for (let ix = 0; ix < W; ix++) {
        const xL = ix > 0 ? ix - 1 : wrapX ? W - 1 : 0;
        const xR = ix < W - 1 ? ix + 1 : wrapX ? 0 : W - 1;
        const idx = row + ix;
        let nu = this.u[idx] - 0.5 * (this.p[row + xR] - this.p[row + xL]);
        let nv = this.v[idx] - 0.5 * (this.p[rowD + ix] - this.p[rowU + ix]);

        if (iy === 0 || iy === H - 1) {
          nv *= 0.65;
        }
        if (!wrapX && (ix === 0 || ix === W - 1)) {
          nu *= 0.45;
        }

        const mag = Math.hypot(nu, nv);
        if (mag > MAX_VEL) {
          const s = MAX_VEL / mag;
          nu *= s;
          nv *= s;
        }
        this.u[idx] = nu;
        this.v[idx] = nv;
      }
    }
  }

  /**
   * Packs `(density, u, v, curl)` into the persistent RGBA16F `DataTexture` buffer
   * with zero allocations and flags `needsUpdate = true`.
   */
  packToTexture(): void {
    const out = this.packedTextureData;
    for (let i = 0; i < FLUID_CELLS; i++) {
      const o = i * 4;
      out[o] = encodeHalfFloat(this.density[i]);
      out[o + 1] = encodeHalfFloat(this.u[i]);
      out[o + 2] = encodeHalfFloat(this.v[i]);
      out[o + 3] = encodeHalfFloat(this.curl[i]);
    }
    this.dataTexture.needsUpdate = true;
  }

  dispose(): void {
    this.dataTexture.dispose();
  }
}
