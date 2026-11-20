import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  abs,
  clamp,
  cos,
  exp,
  float,
  floor,
  max,
  sin,
  smoothstep,
  sqrt,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export type SeaStateId = 'golden' | 'storm' | 'biolum';

export const WAKE_SLOT_COUNT = 8;

export interface SeaStateConfig {
  id: SeaStateId;
  label: string;
  waveAmpScale: number;
  chopScale: number;
  breakerHeight: number;
  sunAzimuth: number;
  sunElevation: number;
  sunColor: [number, number, number];
  skyZenith: [number, number, number];
  skyHorizon: [number, number, number];
  waterDeep: [number, number, number];
  waterShallow: [number, number, number];
  glintColor: [number, number, number];
  foamColor: [number, number, number];
  biolumGain: number;
  fogDensity: number;
  speedScale: number;
}

/**
 * Three instant zero-rebuild "Possibilities" sea states calibrated in linear sRGB
 * for ACESFilmicToneMapping. 'golden' matches the locked twilight/amber hero & macro renders.
 */
export const SEA_STATE_PRESETS: Readonly<Record<SeaStateId, SeaStateConfig>> = {
  golden: {
    id: 'golden',
    label: 'Golden Swell',
    waveAmpScale: 1.0,
    chopScale: 1.0,
    breakerHeight: 1.06,
    sunAzimuth: 0.0,
    sunElevation: 0.058,
    sunColor: [1.0, 0.76, 0.34],
    skyZenith: [0.006, 0.011, 0.032],
    skyHorizon: [0.62, 0.24, 0.055],
    waterDeep: [0.004, 0.015, 0.044],
    waterShallow: [0.012, 0.46, 0.44],
    glintColor: [1.0, 0.77, 0.34],
    foamColor: [0.86, 0.95, 0.97],
    biolumGain: 0.0,
    fogDensity: 0.0046,
    speedScale: 1.0,
  },
  storm: {
    id: 'storm',
    label: 'Abyssal Storm',
    waveAmpScale: 1.44,
    chopScale: 1.35,
    breakerHeight: 1.38,
    sunAzimuth: -0.28,
    sunElevation: 0.14,
    sunColor: [0.36, 0.44, 0.52],
    skyZenith: [0.004, 0.008, 0.016],
    skyHorizon: [0.075, 0.115, 0.155],
    waterDeep: [0.005, 0.016, 0.029],
    waterShallow: [0.018, 0.44, 0.38],
    glintColor: [0.64, 0.78, 0.88],
    foamColor: [0.84, 0.91, 0.94],
    biolumGain: 0.0,
    fogDensity: 0.0068,
    speedScale: 1.28,
  },
  biolum: {
    id: 'biolum',
    label: 'Bioluminescent',
    waveAmpScale: 0.92,
    chopScale: 0.96,
    breakerHeight: 1.02,
    sunAzimuth: 0.14,
    sunElevation: 0.19,
    sunColor: [0.24, 0.34, 0.52],
    skyZenith: [0.001, 0.003, 0.010],
    skyHorizon: [0.008, 0.022, 0.052],
    waterDeep: [0.002, 0.007, 0.023],
    waterShallow: [0.016, 0.68, 0.88],
    glintColor: [0.46, 0.86, 1.0],
    foamColor: [0.36, 0.96, 1.0],
    biolumGain: 1.0,
    fogDensity: 0.0044,
    speedScale: 0.92,
  },
};

export interface GerstnerWaveSpec {
  readonly dirX: number;
  readonly dirZ: number;
  readonly wavelength: number;
  readonly k: number;
  readonly amp: number;
  readonly steepness: number;
  readonly omega: number;
  readonly phase: number;
}

const GRAVITY = 9.81;
const WAVE_SPEED_SCALE = 0.72;

function makeGerstnerWave(
  angleRad: number,
  wavelength: number,
  amp: number,
  steepness: number,
  phase: number
): GerstnerWaveSpec {
  const k = (Math.PI * 2) / wavelength;
  const omega = Math.sqrt(GRAVITY * k) * WAVE_SPEED_SCALE;
  return {
    dirX: Math.sin(angleRad),
    dirZ: Math.cos(angleRad),
    wavelength,
    k,
    amp,
    steepness,
    omega,
    phase,
  };
}

/**
 * 8 directional Gerstner swells with irrational azimuths (zero grid tiling,
 * zero polar harmonics) spanning ~29.3m rolling horizon swells down to ~3.45m foreground chop.
 * Sum(k * amp * steepness) = 0.388, remaining loop-free (< 0.80) even at storm chop.
 */
export const GERSTNER_WAVES: readonly GerstnerWaveSpec[] = [
  makeGerstnerWave(0.19, 29.3, 0.34, 0.42, 0.40),
  makeGerstnerWave(-0.37, 21.7, 0.26, 0.48, 2.13),
  makeGerstnerWave(0.61, 15.1, 0.21, 0.56, 4.71),
  makeGerstnerWave(-0.94, 11.4, 0.16, 0.62, 1.09),
  makeGerstnerWave(1.28, 8.2, 0.115, 0.68, 3.67),
  makeGerstnerWave(-1.71, 6.1, 0.078, 0.72, 5.42),
  makeGerstnerWave(2.39, 4.6, 0.052, 0.74, 0.88),
  makeGerstnerWave(-2.63, 3.45, 0.034, 0.76, 2.95),
];

export interface HeroSolitonSpec {
  readonly id: string;
  readonly baseX: number;
  readonly baseZ: number;
  readonly dirX: number;
  readonly dirZ: number;
  readonly tanX: number;
  readonly tanZ: number;
  readonly sigmaN: number;
  readonly sigmaT: number;
  readonly bow: number;
  readonly amp: number;
  readonly curlOverhang: number;
  readonly period: number;
  readonly phase0: number;
  readonly travelSpan: number;
}

function makeSoliton(
  id: string,
  baseX: number,
  baseZ: number,
  rawDirX: number,
  rawDirZ: number,
  sigmaN: number,
  sigmaT: number,
  bow: number,
  amp: number,
  curlOverhang: number,
  period: number,
  phase0: number,
  travelSpan: number
): HeroSolitonSpec {
  const len = Math.hypot(rawDirX, rawDirZ);
  const dirX = rawDirX / len;
  const dirZ = rawDirZ / len;
  return {
    id,
    baseX,
    baseZ,
    dirX,
    dirZ,
    tanX: -dirZ,
    tanZ: dirX,
    sigmaN,
    sigmaT,
    bow,
    amp,
    curlOverhang,
    period,
    phase0,
    travelSpan,
  };
}

/**
 * 3 traveling asymmetric curling Hero Breaker Solitons tuned for the hero camera at
 * ~(0, 4.3, 13.6) looking down -Z toward (0, 1.1, -6.5):
 * - [0] foreground-right hero breaker curling left/forward across the right third (active immediately at t = 0..4s)
 * - [1] midground-left secondary breaker curling right/forward across the left third (active immediately at t = 0..4s)
 * - [2] midground-right cycling breaker rising cleanly as [0] subsides so the sea stays continuously alive
 */
export const HERO_SOLITONS: readonly HeroSolitonSpec[] = [
  makeSoliton('fg-right', 6.8, 3.6, -0.86, 0.51, 1.85, 3.9, 0.046, 2.68, 1.24, 18.5, 0.44, 2.4),
  makeSoliton('mg-left', -7.8, -5.0, 0.77, 0.64, 1.95, 4.0, 0.034, 2.38, 1.14, 21.3, 0.47, 2.6),
  makeSoliton('mg-far', 9.2, -13.5, -0.74, 0.672, 2.25, 4.3, 0.026, 2.1, 0.96, 16.7, 0.92, 4.2),
];

const ROGUE_DIR_LEN = Math.hypot(-0.34, 0.94);
const ROGUE_DIR_X = -0.34 / ROGUE_DIR_LEN;
const ROGUE_DIR_Z = 0.94 / ROGUE_DIR_LEN;
const ROGUE_TAN_X = -ROGUE_DIR_Z;
const ROGUE_TAN_Z = ROGUE_DIR_X;
const ROGUE_SIGMA_N = 2.45;
const ROGUE_SIGMA_T = 4.8;
const ROGUE_BOW = 0.038;
const ROGUE_AMP = 2.35;
const ROGUE_CURL_OVERHANG = 0.96;
const ROGUE_SPEED = 4.2;

export type FloatUniformNode = ReturnType<typeof uniform<'float'>>;
export type Vec3UniformNode = ReturnType<typeof uniform<'vec3'>>;
export type Vec4UniformNode = ReturnType<typeof uniform<'vec4'>>;

export interface WaveFieldUniforms {
  uTime: FloatUniformNode;
  uWaveAmpScale: FloatUniformNode;
  uChopScale: FloatUniformNode;
  uBreakerHeight: FloatUniformNode;
  uSunDir: Vec3UniformNode;
  uSunColor: Vec3UniformNode;
  uSkyZenith: Vec3UniformNode;
  uSkyHorizon: Vec3UniformNode;
  uWaterDeep: Vec3UniformNode;
  uWaterShallow: Vec3UniformNode;
  uGlintColor: Vec3UniformNode;
  uFoamColor: Vec3UniformNode;
  uBiolumGain: FloatUniformNode;
  uFogDensity: FloatUniformNode;
  uWakeSlots: ReturnType<typeof uniformArray<'vec4'>>;
  uVortex: Vec4UniformNode;
  uRogueWave: Vec4UniformNode;
  /** Backing Vector4 array for uWakeSlots: mutate in place (x, z, birthTime, strength) with zero allocations. */
  wakeData: THREE.Vector4[];
}

interface WakeTslCache {
  dispAndEnergy: (px: TslNode, pz: TslNode) => TslNode;
  gradAndOrbital: (px: TslNode, pz: TslNode) => TslNode;
}

const _wakeFnCache = new WeakMap<WaveFieldUniforms, WakeTslCache>();

function sunDirectionFromAngles(azimuth: number, elevation: number, out: THREE.Vector3): THREE.Vector3 {
  const cosEl = Math.cos(elevation);
  const sinEl = Math.sin(elevation);
  out.set(Math.sin(azimuth) * cosEl, sinEl, -Math.cos(azimuth) * cosEl).normalize();
  return out;
}

export function createWaveFieldUniforms(initial: SeaStateId = 'golden'): WaveFieldUniforms {
  const preset = SEA_STATE_PRESETS[initial];
  const sunDir = sunDirectionFromAngles(preset.sunAzimuth, preset.sunElevation, new THREE.Vector3());

  const wakeData: THREE.Vector4[] = [];
  for (let i = 0; i < WAKE_SLOT_COUNT; i++) {
    wakeData.push(new THREE.Vector4(0, 0, -100, 0));
  }

  return {
    uTime: uniform(0.0),
    uWaveAmpScale: uniform(preset.waveAmpScale),
    uChopScale: uniform(preset.chopScale),
    uBreakerHeight: uniform(preset.breakerHeight),
    uSunDir: uniform(sunDir),
    uSunColor: uniform(new THREE.Vector3(preset.sunColor[0], preset.sunColor[1], preset.sunColor[2])),
    uSkyZenith: uniform(new THREE.Vector3(preset.skyZenith[0], preset.skyZenith[1], preset.skyZenith[2])),
    uSkyHorizon: uniform(new THREE.Vector3(preset.skyHorizon[0], preset.skyHorizon[1], preset.skyHorizon[2])),
    uWaterDeep: uniform(new THREE.Vector3(preset.waterDeep[0], preset.waterDeep[1], preset.waterDeep[2])),
    uWaterShallow: uniform(
      new THREE.Vector3(preset.waterShallow[0], preset.waterShallow[1], preset.waterShallow[2])
    ),
    uGlintColor: uniform(new THREE.Vector3(preset.glintColor[0], preset.glintColor[1], preset.glintColor[2])),
    uFoamColor: uniform(new THREE.Vector3(preset.foamColor[0], preset.foamColor[1], preset.foamColor[2])),
    uBiolumGain: uniform(preset.biolumGain),
    uFogDensity: uniform(preset.fogDensity),
    uWakeSlots: uniformArray<'vec4'>(wakeData, 'vec4'),
    uVortex: uniform(new THREE.Vector4(0, 0, 0, 0)),
    uRogueWave: uniform(new THREE.Vector4(0, 0, -100, 0)),
    wakeData,
  };
}

function lerpScalar(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpRgb(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  t: number,
  out: [number, number, number]
): void {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
}

export function blendSeaStates(
  a: SeaStateConfig,
  b: SeaStateConfig,
  t: number,
  out: SeaStateConfig
): SeaStateConfig {
  const k = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
  out.id = k < 0.5 ? a.id : b.id;
  out.label = k < 0.5 ? a.label : b.label;
  out.waveAmpScale = lerpScalar(a.waveAmpScale, b.waveAmpScale, k);
  out.chopScale = lerpScalar(a.chopScale, b.chopScale, k);
  out.breakerHeight = lerpScalar(a.breakerHeight, b.breakerHeight, k);
  out.sunAzimuth = lerpScalar(a.sunAzimuth, b.sunAzimuth, k);
  out.sunElevation = lerpScalar(a.sunElevation, b.sunElevation, k);
  lerpRgb(a.sunColor, b.sunColor, k, out.sunColor);
  lerpRgb(a.skyZenith, b.skyZenith, k, out.skyZenith);
  lerpRgb(a.skyHorizon, b.skyHorizon, k, out.skyHorizon);
  lerpRgb(a.waterDeep, b.waterDeep, k, out.waterDeep);
  lerpRgb(a.waterShallow, b.waterShallow, k, out.waterShallow);
  lerpRgb(a.glintColor, b.glintColor, k, out.glintColor);
  lerpRgb(a.foamColor, b.foamColor, k, out.foamColor);
  out.biolumGain = lerpScalar(a.biolumGain, b.biolumGain, k);
  out.fogDensity = lerpScalar(a.fogDensity, b.fogDensity, k);
  out.speedScale = lerpScalar(a.speedScale, b.speedScale, k);
  return out;
}

export function applySeaStateToUniforms(u: WaveFieldUniforms, blended: SeaStateConfig): void {
  u.uWaveAmpScale.value = blended.waveAmpScale;
  u.uChopScale.value = blended.chopScale;
  u.uBreakerHeight.value = blended.breakerHeight;
  sunDirectionFromAngles(blended.sunAzimuth, blended.sunElevation, u.uSunDir.value);
  u.uSunColor.value.set(blended.sunColor[0], blended.sunColor[1], blended.sunColor[2]);
  u.uSkyZenith.value.set(blended.skyZenith[0], blended.skyZenith[1], blended.skyZenith[2]);
  u.uSkyHorizon.value.set(blended.skyHorizon[0], blended.skyHorizon[1], blended.skyHorizon[2]);
  u.uWaterDeep.value.set(blended.waterDeep[0], blended.waterDeep[1], blended.waterDeep[2]);
  u.uWaterShallow.value.set(
    blended.waterShallow[0],
    blended.waterShallow[1],
    blended.waterShallow[2]
  );
  u.uGlintColor.value.set(blended.glintColor[0], blended.glintColor[1], blended.glintColor[2]);
  u.uFoamColor.value.set(blended.foamColor[0], blended.foamColor[1], blended.foamColor[2]);
  u.uBiolumGain.value = blended.biolumGain;
  u.uFogDensity.value = blended.fogDensity;
}

function clampCpu(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
  const t = clampCpu((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

export interface SolitonLiveStateCPU {
  x: number;
  z: number;
  crestX: number;
  crestZ: number;
  riseEnv: number;
  curlLife: number;
  cycle: number;
}

/**
 * Queries the traveling world position & curl lifecycle of one of the 3 Hero Breaker Solitons.
 * Useful for camera framing checks, tests, or anchor placement without allocating.
 */
export function getHeroSolitonStateCPU(
  index: number,
  timeSec: number,
  out: SolitonLiveStateCPU = {
    x: 0,
    z: 0,
    crestX: 0,
    crestZ: 0,
    riseEnv: 0,
    curlLife: 0,
    cycle: 0,
  }
): SolitonLiveStateCPU {
  const s = HERO_SOLITONS[index] ?? HERO_SOLITONS[0];
  const raw = timeSec / s.period + s.phase0;
  const cycle = raw - Math.floor(raw);
  const riseEnv = smoothstepCpu(0.0, 0.34, cycle) * (1.0 - smoothstepCpu(0.70, 1.0, cycle));
  const curlLife = smoothstepCpu(0.20, 0.50, cycle) * (1.0 - smoothstepCpu(0.68, 0.94, cycle));
  const travel = (cycle - 0.48) * s.travelSpan;
  const cx = s.baseX + s.dirX * travel;
  const cz = s.baseZ + s.dirZ * travel;
  // Peak crest ridge sits near u = +0.30 along propagation direction
  const peakOffset = 0.30 * s.sigmaN;
  out.x = cx;
  out.z = cz;
  out.crestX = cx + s.dirX * peakOffset;
  out.crestZ = cz + s.dirZ * peakOffset;
  out.riseEnv = riseEnv;
  out.curlLife = curlLife;
  out.cycle = cycle;
  return out;
}

export interface WaveSampleCPU {
  dx: number;
  dy: number;
  dz: number;
  nx: number;
  ny: number;
  nz: number;
  crestEnergy: number;
  curlFactor: number;
  wakeEnergy: number;
  orbitalPhase: number;
}

/**
 * Exact CPU twin of `evaluateWaveTSL` — zero per-frame allocations when `out` is supplied.
 */
export function evaluateWaveCPU(
  x: number,
  z: number,
  u: WaveFieldUniforms,
  out: WaveSampleCPU = {
    dx: 0,
    dy: 0,
    dz: 0,
    nx: 0,
    ny: 1,
    nz: 0,
    crestEnergy: 0,
    curlFactor: 0,
    wakeEnergy: 0,
    orbitalPhase: 0,
  }
): WaveSampleCPU {
  const t = u.uTime.value;
  const ampScale = u.uWaveAmpScale.value;
  const chopScale = u.uChopScale.value;
  const breakerHeight = u.uBreakerHeight.value;
  const breakerEnergyScale = clampCpu(breakerHeight, 0.2, 1.8);

  let dx = 0;
  let dy = 0;
  let dz = 0;

  let dDxDx = 0;
  let dDxDz = 0;
  let dDyDx = 0;
  let dDyDz = 0;
  let dDzDx = 0;
  let dDzDz = 0;

  let crestAccum = 0;
  let curlAccum = 0;
  let wakeAccum = 0;
  let orbitalPhase = t * 0.85;

  // (a) 8 directional Gerstner swells
  for (let i = 0; i < GERSTNER_WAVES.length; i++) {
    const w = GERSTNER_WAVES[i];
    const amp = w.amp * ampScale;
    const qa = w.steepness * chopScale * amp;
    const theta = w.k * (w.dirX * x + w.dirZ * z) - w.omega * t + w.phase;
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    dx += w.dirX * qa * c;
    dy += amp * s;
    dz += w.dirZ * qa * c;

    const kQaS = w.k * qa * s;
    const kAmpC = w.k * amp * c;
    dDxDx -= w.dirX * w.dirX * kQaS;
    dDxDz -= w.dirX * w.dirZ * kQaS;
    dDyDx += w.dirX * kAmpC;
    dDyDz += w.dirZ * kAmpC;
    dDzDx -= w.dirZ * w.dirX * kQaS;
    dDzDz -= w.dirZ * w.dirZ * kQaS;

    const peakPos = s > 0 ? s : 0;
    crestAccum += peakPos * peakPos * (w.k * amp * 1.15);
    orbitalPhase += s * (w.amp * 1.28);
  }

  // (b) 3 traveling asymmetric curling Hero Breaker Solitons
  for (let k = 0; k < HERO_SOLITONS.length; k++) {
    const sol = HERO_SOLITONS[k];
    const raw = t / sol.period + sol.phase0;
    const cycle = raw - Math.floor(raw);
    const riseEnv = smoothstepCpu(0.0, 0.34, cycle) * (1.0 - smoothstepCpu(0.70, 1.0, cycle));
    const curlLife = smoothstepCpu(0.20, 0.50, cycle) * (1.0 - smoothstepCpu(0.68, 0.94, cycle));
    const travel = (cycle - 0.48) * sol.travelSpan;

    const cx = sol.baseX + sol.dirX * travel;
    const cz = sol.baseZ + sol.dirZ * travel;

    const rx = x - cx;
    const rz = z - cz;
    const nRaw = rx * sol.dirX + rz * sol.dirZ;
    const tDist = rx * sol.tanX + rz * sol.tanZ;
    const nLocal = nRaw - sol.bow * tDist * tDist;

    const uCoord = nLocal / sol.sigmaN;
    const vCoord = tDist / sol.sigmaT;

    const dvDx = sol.tanX / sol.sigmaT;
    const dvDz = sol.tanZ / sol.sigmaT;
    const duDx = (sol.dirX - 2.0 * sol.bow * tDist * sol.tanX) / sol.sigmaN;
    const duDz = (sol.dirZ - 2.0 * sol.bow * tDist * sol.tanZ) / sol.sigmaN;

    const envT = Math.exp(-vCoord * vCoord);
    const hScale = riseEnv * sol.amp * breakerHeight;

    const g1 = Math.exp(-uCoord * uCoord);
    const u2 = uCoord - 0.30;
    const g2 = Math.exp(-2.4 * u2 * u2);
    const u3 = uCoord - 1.12;
    const g3 = Math.exp(-1.15 * u3 * u3);
    const u4 = uCoord - 0.62;
    const g4 = Math.exp(-5.0 * u4 * u4);

    const hProf = 0.68 * g1 + 0.48 * g2 - 0.26 * g3 - 0.16 * curlLife * g4;
    const dhProf =
      -1.36 * uCoord * g1 - 2.304 * u2 * g2 + 0.598 * u3 * g3 + 1.60 * curlLife * u4 * g4;

    const uL = uCoord - 0.34;
    const gL = Math.exp(-3.2 * uL * uL);
    const uB = uCoord + 0.65;
    const gB = Math.exp(-1.4 * uB * uB);

    const cProf = sol.curlOverhang * curlLife * gL - 0.36 * gB;
    const dcProf = -6.4 * sol.curlOverhang * curlLife * uL * gL + 1.008 * uB * gB;

    const dyK = hScale * envT * hProf;
    const hThrustK = hScale * envT * cProf;

    dx += sol.dirX * hThrustK;
    dy += dyK;
    dz += sol.dirZ * hThrustK;

    const dDyDu = hScale * envT * dhProf;
    const dDyDv = -2.0 * vCoord * dyK;
    dDyDx += dDyDu * duDx + dDyDv * dvDx;
    dDyDz += dDyDu * duDz + dDyDv * dvDz;

    const dHDu = hScale * envT * dcProf;
    const dHDv = -2.0 * vCoord * hThrustK;
    const dHDx = dHDu * duDx + dHDv * dvDx;
    const dHDz = dHDu * duDz + dHDv * dvDz;
    dDxDx += sol.dirX * dHDx;
    dDxDz += sol.dirX * dHDz;
    dDzDx += sol.dirZ * dHDx;
    dDzDz += sol.dirZ * dHDz;

    const uBelly = uCoord - 0.25;
    const bellyMask = envT * Math.exp(-1.8 * uBelly * uBelly);
    const lipMask = envT * gL;

    crestAccum += riseEnv * (0.72 * bellyMask + 0.45 * lipMask) * breakerEnergyScale;
    curlAccum += riseEnv * curlLife * (0.45 * bellyMask + 0.78 * lipMask) * breakerEnergyScale;
    orbitalPhase += (uCoord * 2.4 + curlLife * 1.8) * bellyMask * riseEnv;
  }

  // (c) 8-slot interactive wake/ripple ring field (strictly origin-relative to (slot.x, slot.y))
  for (let i = 0; i < WAKE_SLOT_COUNT; i++) {
    const slot = u.wakeData[i];
    const age = t - slot.z;
    const strength = clampCpu(slot.w, 0.0, 2.5);
    const active =
      smoothstepCpu(0.005, 0.055, age) * (1.0 - smoothstepCpu(3.2, 5.2, age)) * strength;

    const rx = x - slot.x;
    const rz = z - slot.y;
    const dist = Math.sqrt(rx * rx + rz * rz + 1e-4);
    const radX = rx / dist;
    const radZ = rz / dist;

    const frontR = age * 3.1;
    const diff = dist - frontR;
    const ringEnv =
      Math.exp(-diff * diff * 0.52) * Math.exp(-dist * 0.14) * Math.exp(-age * 0.52) * active;

    const phase = diff * 4.2;
    const c = Math.cos(phase);
    const s = Math.sin(phase);

    dx += -0.09 * ringEnv * s * radX;
    dy += 0.30 * ringEnv * c;
    dz += -0.09 * ringEnv * s * radZ;

    const dEnvDr = ringEnv * (-1.04 * diff - 0.14);
    const dDyDr = 0.30 * (dEnvDr * c - 4.2 * ringEnv * s);
    dDyDx += dDyDr * radX;
    dDyDz += dDyDr * radZ;

    wakeAccum += ringEnv * (0.78 + 0.32 * Math.abs(c));
    crestAccum += ringEnv * 0.32;
    orbitalPhase += ringEnv * s * 1.4;
  }

  // (d1) Interactive maelstrom vortex (uVortex: x, z, strength, spinPhase)
  {
    const vx = u.uVortex.value.x;
    const vz = u.uVortex.value.y;
    const vs = clampCpu(u.uVortex.value.z, 0.0, 2.0);
    const spinPhase = u.uVortex.value.w;

    const rx = x - vx;
    const rz = z - vz;
    const r2 = rx * rx + rz * rz;
    const r = Math.sqrt(r2 + 0.01);
    const urX = rx / r;
    const urZ = rz / r;
    const utX = -urZ;
    const utZ = urX;

    const eye = Math.exp(-r2 * 0.32);
    const drRing = r - 2.6;
    const ringEnv = Math.exp(-drRing * drRing * 0.42);
    const swirlDecay = Math.exp(-r2 * 0.055);
    const swirlArg = r * 2.1 - spinPhase * 2.4;
    const swirlWave = Math.sin(swirlArg);
    const swirlCos = Math.cos(swirlArg);

    const dyV = vs * (-0.95 * eye + 1.45 * ringEnv + 0.24 * swirlWave * swirlDecay);
    const swirlAmp = vs * 0.92 * Math.exp(-drRing * drRing * 0.28);

    dx += utX * swirlAmp;
    dy += dyV;
    dz += utZ * swirlAmp;

    const dEyeDr = -0.64 * r * eye;
    const dRingDr = -0.84 * drRing * ringEnv;
    const dSwirlDr = (2.1 * swirlCos - 0.11 * r * swirlWave) * swirlDecay;
    const dDyVDr = vs * (-0.95 * dEyeDr + 1.45 * dRingDr + 0.24 * dSwirlDr);

    dDyDx += dDyVDr * urX;
    dDyDz += dDyVDr * urZ;

    crestAccum += vs * ringEnv * 0.85;
    curlAccum += vs * ringEnv * 0.65;
    wakeAccum += vs * (ringEnv * 0.80 + eye * 0.50 + 0.30 * swirlDecay);
    orbitalPhase += vs * (spinPhase * 0.8 + ringEnv * 1.6) * swirlDecay;
  }

  // (d2) Interactive rogue breaker surge (uRogueWave: x, z, birthTime, amplitude)
  {
    const rx0 = u.uRogueWave.value.x;
    const rz0 = u.uRogueWave.value.y;
    const birthTime = u.uRogueWave.value.z;
    const ampR = clampCpu(u.uRogueWave.value.w, 0.0, 2.5);
    const age = t - birthTime;
    const rLife =
      smoothstepCpu(0.02, 0.42, age) * (1.0 - smoothstepCpu(2.8, 5.2, age)) * ampR;
    const rCurl = smoothstepCpu(0.18, 0.72, age) * (1.0 - smoothstepCpu(2.6, 4.8, age));
    const agePos = age > 0 ? age : 0;

    const cx = rx0 + ROGUE_DIR_X * agePos * ROGUE_SPEED;
    const cz = rz0 + ROGUE_DIR_Z * agePos * ROGUE_SPEED;

    const rx = x - cx;
    const rz = z - cz;
    const nRaw = rx * ROGUE_DIR_X + rz * ROGUE_DIR_Z;
    const tDist = rx * ROGUE_TAN_X + rz * ROGUE_TAN_Z;
    const nLocal = nRaw - ROGUE_BOW * tDist * tDist;

    const uCoord = nLocal / ROGUE_SIGMA_N;
    const vCoord = tDist / ROGUE_SIGMA_T;

    const dvDx = ROGUE_TAN_X / ROGUE_SIGMA_T;
    const dvDz = ROGUE_TAN_Z / ROGUE_SIGMA_T;
    const duDx = (ROGUE_DIR_X - 2.0 * ROGUE_BOW * tDist * ROGUE_TAN_X) / ROGUE_SIGMA_N;
    const duDz = (ROGUE_DIR_Z - 2.0 * ROGUE_BOW * tDist * ROGUE_TAN_Z) / ROGUE_SIGMA_N;

    const envT = Math.exp(-vCoord * vCoord);
    const hScale = rLife * ROGUE_AMP;

    const g1 = Math.exp(-uCoord * uCoord);
    const u2 = uCoord - 0.30;
    const g2 = Math.exp(-2.4 * u2 * u2);
    const u3 = uCoord - 1.12;
    const g3 = Math.exp(-1.15 * u3 * u3);
    const u4 = uCoord - 0.62;
    const g4 = Math.exp(-5.0 * u4 * u4);

    const hProf = 0.68 * g1 + 0.48 * g2 - 0.26 * g3 - 0.16 * rCurl * g4;
    const dhProf =
      -1.36 * uCoord * g1 - 2.304 * u2 * g2 + 0.598 * u3 * g3 + 1.60 * rCurl * u4 * g4;

    const uL = uCoord - 0.34;
    const gL = Math.exp(-3.2 * uL * uL);
    const uB = uCoord + 0.65;
    const gB = Math.exp(-1.4 * uB * uB);

    const cProf = ROGUE_CURL_OVERHANG * rCurl * gL - 0.36 * gB;
    const dcProf = -6.4 * ROGUE_CURL_OVERHANG * rCurl * uL * gL + 1.008 * uB * gB;

    const dyR = hScale * envT * hProf;
    const hThrustR = hScale * envT * cProf;

    dx += ROGUE_DIR_X * hThrustR;
    dy += dyR;
    dz += ROGUE_DIR_Z * hThrustR;

    const dDyDu = hScale * envT * dhProf;
    const dDyDv = -2.0 * vCoord * dyR;
    dDyDx += dDyDu * duDx + dDyDv * dvDx;
    dDyDz += dDyDu * duDz + dDyDv * dvDz;

    const dHDu = hScale * envT * dcProf;
    const dHDv = -2.0 * vCoord * hThrustR;
    const dHDx = dHDu * duDx + dHDv * dvDx;
    const dHDz = dHDu * duDz + dHDv * dvDz;
    dDxDx += ROGUE_DIR_X * dHDx;
    dDxDz += ROGUE_DIR_X * dHDz;
    dDzDx += ROGUE_DIR_Z * dHDx;
    dDzDz += ROGUE_DIR_Z * dHDz;

    const uBelly = uCoord - 0.25;
    const bellyMask = envT * Math.exp(-1.8 * uBelly * uBelly);
    const lipMask = envT * gL;

    crestAccum += rLife * (0.75 * bellyMask + 0.48 * lipMask);
    curlAccum += rLife * rCurl * (0.48 * bellyMask + 0.80 * lipMask);
    wakeAccum += rLife * (0.45 * bellyMask + 0.45 * lipMask);
    orbitalPhase += (uCoord * 2.4 + rCurl * 1.8) * bellyMask * rLife;
  }

  // Exact parametric Jacobian normal: N = normalize(dP/dz x dP/dx)
  const txX = 1.0 + dDxDx;
  const txY = dDyDx;
  const txZ = dDzDx;

  const tzX = dDxDz;
  const tzY = dDyDz;
  const tzZ = 1.0 + dDzDz;

  const nxRaw = tzY * txZ - tzZ * txY;
  const nyRaw = tzZ * txX - tzX * txZ;
  const nzRaw = tzX * txY - tzY * txX;
  const nLen = Math.sqrt(nxRaw * nxRaw + nyRaw * nyRaw + nzRaw * nzRaw + 1e-8);

  out.dx = dx;
  out.dy = dy;
  out.dz = dz;
  out.nx = nxRaw / nLen;
  out.ny = nyRaw / nLen;
  out.nz = nzRaw / nLen;
  out.crestEnergy = clampCpu(crestAccum, 0.0, 1.0);
  out.curlFactor = clampCpu(curlAccum, 0.0, 1.0);
  out.wakeEnergy = clampCpu(wakeAccum, 0.0, 1.0);
  out.orbitalPhase = orbitalPhase;
  return out;
}

export interface WaveSampleTSL {
  displacement: TslNode;
  normal: TslNode;
  crestEnergy: TslNode;
  curlFactor: TslNode;
  wakeEnergy: TslNode;
  orbitalPhase: TslNode;
}

function getWakeTslFns(u: WaveFieldUniforms): WakeTslCache {
  const cached = _wakeFnCache.get(u);
  if (cached) return cached;

  // Strictly inside Fn() so Loop() always executes inside an active TSL StackNode
  // and adheres to @types/three@0.187.0 uniformArray<'vec4'> + vec4(element(i)) rules.
  const dispAndEnergy = Fn(([px_in, pz_in]: [TslNode, TslNode]) => {
    const px = float(px_in);
    const pz = float(pz_in);
    const dxSum = float(0.0).toVar();
    const dySum = float(0.0).toVar();
    const dzSum = float(0.0).toVar();
    const wakeSum = float(0.0).toVar();

    Loop(WAKE_SLOT_COUNT, ({ i }) => {
      const slot = vec4(u.uWakeSlots.element(i));
      const age = u.uTime.sub(slot.z);
      const strength = clamp(slot.w, 0.0, 2.5);
      const active = smoothstep(float(0.005), float(0.055), age)
        .mul(float(1.0).sub(smoothstep(float(3.2), float(5.2), age)))
        .mul(strength);

      const rx = px.sub(slot.x);
      const rz = pz.sub(slot.y);
      const dist = sqrt(rx.mul(rx).add(rz.mul(rz)).add(1e-4));
      const radX = rx.div(dist);
      const radZ = rz.div(dist);

      const frontR = age.mul(3.1);
      const diff = dist.sub(frontR);
      const ringEnv = exp(diff.mul(diff).mul(-0.52))
        .mul(exp(dist.mul(-0.14)))
        .mul(exp(age.mul(-0.52)))
        .mul(active);

      const phase = diff.mul(4.2);
      const c = cos(phase);
      const s = sin(phase);

      dxSum.addAssign(ringEnv.mul(s).mul(radX).mul(-0.09));
      dySum.addAssign(ringEnv.mul(c).mul(0.30));
      dzSum.addAssign(ringEnv.mul(s).mul(radZ).mul(-0.09));
      wakeSum.addAssign(ringEnv.mul(float(0.78).add(abs(c).mul(0.32))));
    });

    return vec4(dxSum, dySum, dzSum, wakeSum);
  });

  const gradAndOrbital = Fn(([px_in, pz_in]: [TslNode, TslNode]) => {
    const px = float(px_in);
    const pz = float(pz_in);
    const dDyDxSum = float(0.0).toVar();
    const dDyDzSum = float(0.0).toVar();
    const crestSum = float(0.0).toVar();
    const orbSum = float(0.0).toVar();

    Loop(WAKE_SLOT_COUNT, ({ i }) => {
      const slot = vec4(u.uWakeSlots.element(i));
      const age = u.uTime.sub(slot.z);
      const strength = clamp(slot.w, 0.0, 2.5);
      const active = smoothstep(float(0.005), float(0.055), age)
        .mul(float(1.0).sub(smoothstep(float(3.2), float(5.2), age)))
        .mul(strength);

      const rx = px.sub(slot.x);
      const rz = pz.sub(slot.y);
      const dist = sqrt(rx.mul(rx).add(rz.mul(rz)).add(1e-4));
      const radX = rx.div(dist);
      const radZ = rz.div(dist);

      const frontR = age.mul(3.1);
      const diff = dist.sub(frontR);
      const ringEnv = exp(diff.mul(diff).mul(-0.52))
        .mul(exp(dist.mul(-0.14)))
        .mul(exp(age.mul(-0.52)))
        .mul(active);

      const phase = diff.mul(4.2);
      const c = cos(phase);
      const s = sin(phase);

      const dEnvDr = ringEnv.mul(diff.mul(-1.04).sub(0.14));
      const dDyDr = dEnvDr.mul(c).sub(ringEnv.mul(s).mul(4.2)).mul(0.30);

      dDyDxSum.addAssign(dDyDr.mul(radX));
      dDyDzSum.addAssign(dDyDr.mul(radZ));
      crestSum.addAssign(ringEnv.mul(0.32));
      orbSum.addAssign(ringEnv.mul(s).mul(1.4));
    });

    return vec4(dDyDxSum, dDyDzSum, crestSum, orbSum);
  });

  const entry: WakeTslCache = { dispAndEnergy, gradAndOrbital };
  _wakeFnCache.set(u, entry);
  return entry;
}

/**
 * Pure TSL ocean wave field evaluator matching `evaluateWaveCPU` term-for-term.
 * Safe to invoke either at top-level material creation or inside a caller's `Fn(() => ...)`,
 * with strictly uniform control flow and zero dynamic branches.
 */
export function evaluateWaveTSL(posXZ: TslNode, u: WaveFieldUniforms): WaveSampleTSL {
  const p = vec2(posXZ);
  const px = p.x;
  const pz = p.y;
  const t = u.uTime;
  const ampScale = u.uWaveAmpScale;
  const chopScale = u.uChopScale;
  const breakerHeight = u.uBreakerHeight;
  const breakerEnergyScale = clamp(breakerHeight, 0.2, 1.8);

  let dx: TslNode = float(0.0);
  let dy: TslNode = float(0.0);
  let dz: TslNode = float(0.0);

  let dDxDx: TslNode = float(0.0);
  let dDxDz: TslNode = float(0.0);
  let dDyDx: TslNode = float(0.0);
  let dDyDz: TslNode = float(0.0);
  let dDzDx: TslNode = float(0.0);
  let dDzDz: TslNode = float(0.0);

  let crestAccum: TslNode = float(0.0);
  let curlAccum: TslNode = float(0.0);
  let wakeAccum: TslNode = float(0.0);
  let orbitalPhase: TslNode = t.mul(0.85);

  // (a) 8 directional Gerstner swells
  for (let i = 0; i < GERSTNER_WAVES.length; i++) {
    const w = GERSTNER_WAVES[i];
    const amp = ampScale.mul(w.amp);
    const qa = chopScale.mul(amp).mul(w.steepness);
    const theta = px
      .mul(w.dirX)
      .add(pz.mul(w.dirZ))
      .mul(w.k)
      .sub(t.mul(w.omega))
      .add(w.phase);
    const c = cos(theta);
    const s = sin(theta);

    dx = dx.add(qa.mul(c).mul(w.dirX));
    dy = dy.add(amp.mul(s));
    dz = dz.add(qa.mul(c).mul(w.dirZ));

    const kQaS = qa.mul(s).mul(w.k);
    const kAmpC = amp.mul(c).mul(w.k);
    dDxDx = dDxDx.sub(kQaS.mul(w.dirX * w.dirX));
    dDxDz = dDxDz.sub(kQaS.mul(w.dirX * w.dirZ));
    dDyDx = dDyDx.add(kAmpC.mul(w.dirX));
    dDyDz = dDyDz.add(kAmpC.mul(w.dirZ));
    dDzDx = dDzDx.sub(kQaS.mul(w.dirZ * w.dirX));
    dDzDz = dDzDz.sub(kQaS.mul(w.dirZ * w.dirZ));

    const peakPos = max(float(0.0), s);
    crestAccum = crestAccum.add(peakPos.mul(peakPos).mul(amp.mul(w.k * 1.15)));
    orbitalPhase = orbitalPhase.add(s.mul(w.amp * 1.28));
  }

  // (b) 3 traveling asymmetric curling Hero Breaker Solitons
  for (let k = 0; k < HERO_SOLITONS.length; k++) {
    const sol = HERO_SOLITONS[k];
    const raw = t.div(sol.period).add(sol.phase0);
    const cycle = raw.sub(floor(raw));
    const riseEnv = smoothstep(float(0.0), float(0.34), cycle).mul(
      float(1.0).sub(smoothstep(float(0.70), float(1.0), cycle))
    );
    const curlLife = smoothstep(float(0.20), float(0.50), cycle).mul(
      float(1.0).sub(smoothstep(float(0.68), float(0.94), cycle))
    );
    const travel = cycle.sub(0.48).mul(sol.travelSpan);

    const cx = travel.mul(sol.dirX).add(sol.baseX);
    const cz = travel.mul(sol.dirZ).add(sol.baseZ);

    const rx = px.sub(cx);
    const rz = pz.sub(cz);
    const nRaw = rx.mul(sol.dirX).add(rz.mul(sol.dirZ));
    const tDist = rx.mul(sol.tanX).add(rz.mul(sol.tanZ));
    const nLocal = nRaw.sub(tDist.mul(tDist).mul(sol.bow));

    const invSigmaN = 1.0 / sol.sigmaN;
    const invSigmaT = 1.0 / sol.sigmaT;
    const uCoord = nLocal.mul(invSigmaN);
    const vCoord = tDist.mul(invSigmaT);

    const dvDx = sol.tanX * invSigmaT;
    const dvDz = sol.tanZ * invSigmaT;
    const duDx = float(sol.dirX).sub(tDist.mul(2.0 * sol.bow * sol.tanX)).mul(invSigmaN);
    const duDz = float(sol.dirZ).sub(tDist.mul(2.0 * sol.bow * sol.tanZ)).mul(invSigmaN);

    const envT = exp(vCoord.mul(vCoord).negate());
    const hScale = riseEnv.mul(breakerHeight).mul(sol.amp);

    const g1 = exp(uCoord.mul(uCoord).negate());
    const u2 = uCoord.sub(0.30);
    const g2 = exp(u2.mul(u2).mul(-2.4));
    const u3 = uCoord.sub(1.12);
    const g3 = exp(u3.mul(u3).mul(-1.15));
    const u4 = uCoord.sub(0.62);
    const g4 = exp(u4.mul(u4).mul(-5.0));

    const hProf = g1
      .mul(0.68)
      .add(g2.mul(0.48))
      .sub(g3.mul(0.26))
      .sub(curlLife.mul(g4).mul(0.16));
    const dhProf = uCoord
      .mul(g1)
      .mul(-1.36)
      .sub(u2.mul(g2).mul(2.304))
      .add(u3.mul(g3).mul(0.598))
      .add(curlLife.mul(u4).mul(g4).mul(1.60));

    const uL = uCoord.sub(0.34);
    const gL = exp(uL.mul(uL).mul(-3.2));
    const uB = uCoord.add(0.65);
    const gB = exp(uB.mul(uB).mul(-1.4));

    const cProf = curlLife.mul(gL).mul(sol.curlOverhang).sub(gB.mul(0.36));
    const dcProf = curlLife
      .mul(uL)
      .mul(gL)
      .mul(-6.4 * sol.curlOverhang)
      .add(uB.mul(gB).mul(1.008));

    const hEnvT = hScale.mul(envT);
    const dyK = hEnvT.mul(hProf);
    const hThrustK = hEnvT.mul(cProf);

    dx = dx.add(hThrustK.mul(sol.dirX));
    dy = dy.add(dyK);
    dz = dz.add(hThrustK.mul(sol.dirZ));

    const dDyDu = hEnvT.mul(dhProf);
    const dDyDv = vCoord.mul(dyK).mul(-2.0);
    dDyDx = dDyDx.add(dDyDu.mul(duDx)).add(dDyDv.mul(dvDx));
    dDyDz = dDyDz.add(dDyDu.mul(duDz)).add(dDyDv.mul(dvDz));

    const dHDu = hEnvT.mul(dcProf);
    const dHDv = vCoord.mul(hThrustK).mul(-2.0);
    const dHDx = dHDu.mul(duDx).add(dHDv.mul(dvDx));
    const dHDz = dHDu.mul(duDz).add(dHDv.mul(dvDz));
    dDxDx = dDxDx.add(dHDx.mul(sol.dirX));
    dDxDz = dDxDz.add(dHDz.mul(sol.dirX));
    dDzDx = dDzDx.add(dHDx.mul(sol.dirZ));
    dDzDz = dDzDz.add(dHDz.mul(sol.dirZ));

    const uBelly = uCoord.sub(0.25);
    const bellyMask = envT.mul(exp(uBelly.mul(uBelly).mul(-1.8)));
    const lipMask = envT.mul(gL);

    crestAccum = crestAccum.add(
      riseEnv.mul(bellyMask.mul(0.72).add(lipMask.mul(0.45))).mul(breakerEnergyScale)
    );
    curlAccum = curlAccum.add(
      riseEnv.mul(curlLife).mul(bellyMask.mul(0.45).add(lipMask.mul(0.78))).mul(breakerEnergyScale)
    );
    orbitalPhase = orbitalPhase.add(
      uCoord.mul(2.4).add(curlLife.mul(1.8)).mul(bellyMask).mul(riseEnv)
    );
  }

  // (c) 8-slot interactive wake/ripple ring field via Loop() inside Fn()
  const wakeFns = getWakeTslFns(u);
  const wakeDE = wakeFns.dispAndEnergy(px, pz);
  const wakeGO = wakeFns.gradAndOrbital(px, pz);

  dx = dx.add(wakeDE.x);
  dy = dy.add(wakeDE.y);
  dz = dz.add(wakeDE.z);
  wakeAccum = wakeAccum.add(wakeDE.w);

  dDyDx = dDyDx.add(wakeGO.x);
  dDyDz = dDyDz.add(wakeGO.y);
  crestAccum = crestAccum.add(wakeGO.z);
  orbitalPhase = orbitalPhase.add(wakeGO.w);

  // (d1) Interactive maelstrom vortex
  {
    const vx = u.uVortex.x;
    const vz = u.uVortex.y;
    const vs = clamp(u.uVortex.z, 0.0, 2.0);
    const spinPhase = u.uVortex.w;

    const rx = px.sub(vx);
    const rz = pz.sub(vz);
    const r2 = rx.mul(rx).add(rz.mul(rz));
    const r = sqrt(r2.add(0.01));
    const urX = rx.div(r);
    const urZ = rz.div(r);
    const utX = urZ.negate();
    const utZ = urX;

    const eye = exp(r2.mul(-0.32));
    const drRing = r.sub(2.6);
    const ringEnv = exp(drRing.mul(drRing).mul(-0.42));
    const swirlDecay = exp(r2.mul(-0.055));
    const swirlArg = r.mul(2.1).sub(spinPhase.mul(2.4));
    const swirlWave = sin(swirlArg);
    const swirlCos = cos(swirlArg);

    const dyV = vs.mul(
      eye.mul(-0.95).add(ringEnv.mul(1.45)).add(swirlWave.mul(swirlDecay).mul(0.24))
    );
    const swirlAmp = vs.mul(0.92).mul(exp(drRing.mul(drRing).mul(-0.28)));

    dx = dx.add(utX.mul(swirlAmp));
    dy = dy.add(dyV);
    dz = dz.add(utZ.mul(swirlAmp));

    const dEyeDr = r.mul(eye).mul(-0.64);
    const dRingDr = drRing.mul(ringEnv).mul(-0.84);
    const dSwirlDr = swirlCos.mul(2.1).sub(r.mul(swirlWave).mul(0.11)).mul(swirlDecay);
    const dDyVDr = vs.mul(dEyeDr.mul(-0.95).add(dRingDr.mul(1.45)).add(dSwirlDr.mul(0.24)));

    dDyDx = dDyDx.add(dDyVDr.mul(urX));
    dDyDz = dDyDz.add(dDyVDr.mul(urZ));

    crestAccum = crestAccum.add(vs.mul(ringEnv).mul(0.85));
    curlAccum = curlAccum.add(vs.mul(ringEnv).mul(0.65));
    wakeAccum = wakeAccum.add(
      vs.mul(ringEnv.mul(0.80).add(eye.mul(0.50)).add(swirlDecay.mul(0.30)))
    );
    orbitalPhase = orbitalPhase.add(
      vs.mul(spinPhase.mul(0.8).add(ringEnv.mul(1.6))).mul(swirlDecay)
    );
  }

  // (d2) Interactive rogue breaker surge
  {
    const rx0 = u.uRogueWave.x;
    const rz0 = u.uRogueWave.y;
    const birthTime = u.uRogueWave.z;
    const ampR = clamp(u.uRogueWave.w, 0.0, 2.5);
    const age = t.sub(birthTime);
    const rLife = smoothstep(float(0.02), float(0.42), age)
      .mul(float(1.0).sub(smoothstep(float(2.8), float(5.2), age)))
      .mul(ampR);
    const rCurl = smoothstep(float(0.18), float(0.72), age).mul(
      float(1.0).sub(smoothstep(float(2.6), float(4.8), age))
    );
    const agePos = max(float(0.0), age);

    const cx = rx0.add(agePos.mul(ROGUE_DIR_X * ROGUE_SPEED));
    const cz = rz0.add(agePos.mul(ROGUE_DIR_Z * ROGUE_SPEED));

    const rx = px.sub(cx);
    const rz = pz.sub(cz);
    const nRaw = rx.mul(ROGUE_DIR_X).add(rz.mul(ROGUE_DIR_Z));
    const tDist = rx.mul(ROGUE_TAN_X).add(rz.mul(ROGUE_TAN_Z));
    const nLocal = nRaw.sub(tDist.mul(tDist).mul(ROGUE_BOW));

    const invSigmaN = 1.0 / ROGUE_SIGMA_N;
    const invSigmaT = 1.0 / ROGUE_SIGMA_T;
    const uCoord = nLocal.mul(invSigmaN);
    const vCoord = tDist.mul(invSigmaT);

    const dvDx = ROGUE_TAN_X * invSigmaT;
    const dvDz = ROGUE_TAN_Z * invSigmaT;
    const duDx = float(ROGUE_DIR_X)
      .sub(tDist.mul(2.0 * ROGUE_BOW * ROGUE_TAN_X))
      .mul(invSigmaN);
    const duDz = float(ROGUE_DIR_Z)
      .sub(tDist.mul(2.0 * ROGUE_BOW * ROGUE_TAN_Z))
      .mul(invSigmaN);

    const envT = exp(vCoord.mul(vCoord).negate());
    const hEnvT = rLife.mul(ROGUE_AMP).mul(envT);

    const g1 = exp(uCoord.mul(uCoord).negate());
    const u2 = uCoord.sub(0.30);
    const g2 = exp(u2.mul(u2).mul(-2.4));
    const u3 = uCoord.sub(1.12);
    const g3 = exp(u3.mul(u3).mul(-1.15));
    const u4 = uCoord.sub(0.62);
    const g4 = exp(u4.mul(u4).mul(-5.0));

    const hProf = g1
      .mul(0.68)
      .add(g2.mul(0.48))
      .sub(g3.mul(0.26))
      .sub(rCurl.mul(g4).mul(0.16));
    const dhProf = uCoord
      .mul(g1)
      .mul(-1.36)
      .sub(u2.mul(g2).mul(2.304))
      .add(u3.mul(g3).mul(0.598))
      .add(rCurl.mul(u4).mul(g4).mul(1.60));

    const uL = uCoord.sub(0.34);
    const gL = exp(uL.mul(uL).mul(-3.2));
    const uB = uCoord.add(0.65);
    const gB = exp(uB.mul(uB).mul(-1.4));

    const cProf = rCurl.mul(gL).mul(ROGUE_CURL_OVERHANG).sub(gB.mul(0.36));
    const dcProf = rCurl
      .mul(uL)
      .mul(gL)
      .mul(-6.4 * ROGUE_CURL_OVERHANG)
      .add(uB.mul(gB).mul(1.008));

    const dyR = hEnvT.mul(hProf);
    const hThrustR = hEnvT.mul(cProf);

    dx = dx.add(hThrustR.mul(ROGUE_DIR_X));
    dy = dy.add(dyR);
    dz = dz.add(hThrustR.mul(ROGUE_DIR_Z));

    const dDyDu = hEnvT.mul(dhProf);
    const dDyDv = vCoord.mul(dyR).mul(-2.0);
    dDyDx = dDyDx.add(dDyDu.mul(duDx)).add(dDyDv.mul(dvDx));
    dDyDz = dDyDz.add(dDyDu.mul(duDz)).add(dDyDv.mul(dvDz));

    const dHDu = hEnvT.mul(dcProf);
    const dHDv = vCoord.mul(hThrustR).mul(-2.0);
    const dHDx = dHDu.mul(duDx).add(dHDv.mul(dvDx));
    const dHDz = dHDu.mul(duDz).add(dHDv.mul(dvDz));
    dDxDx = dDxDx.add(dHDx.mul(ROGUE_DIR_X));
    dDxDz = dDxDz.add(dHDz.mul(ROGUE_DIR_X));
    dDzDx = dDzDx.add(dHDx.mul(ROGUE_DIR_Z));
    dDzDz = dDzDz.add(dHDz.mul(ROGUE_DIR_Z));

    const uBelly = uCoord.sub(0.25);
    const bellyMask = envT.mul(exp(uBelly.mul(uBelly).mul(-1.8)));
    const lipMask = envT.mul(gL);

    crestAccum = crestAccum.add(rLife.mul(bellyMask.mul(0.75).add(lipMask.mul(0.48))));
    curlAccum = curlAccum.add(rLife.mul(rCurl).mul(bellyMask.mul(0.48).add(lipMask.mul(0.80))));
    wakeAccum = wakeAccum.add(rLife.mul(bellyMask.mul(0.45).add(lipMask.mul(0.45))));
    orbitalPhase = orbitalPhase.add(
      uCoord.mul(2.4).add(rCurl.mul(1.8)).mul(bellyMask).mul(rLife)
    );
  }

  // Exact parametric Jacobian normal: N = normalize(dP/dz x dP/dx)
  const txX = float(1.0).add(dDxDx);
  const txY = dDyDx;
  const txZ = dDzDx;

  const tzX = dDxDz;
  const tzY = dDyDz;
  const tzZ = float(1.0).add(dDzDz);

  const nxRaw = tzY.mul(txZ).sub(tzZ.mul(txY));
  const nyRaw = tzZ.mul(txX).sub(tzX.mul(txZ));
  const nzRaw = tzX.mul(txY).sub(tzY.mul(txX));
  const nLen = sqrt(nxRaw.mul(nxRaw).add(nyRaw.mul(nyRaw)).add(nzRaw.mul(nzRaw)).add(1e-8));

  return {
    displacement: vec3(dx, dy, dz),
    normal: vec3(nxRaw.div(nLen), nyRaw.div(nLen), nzRaw.div(nLen)),
    crestEnergy: clamp(crestAccum, 0.0, 1.0),
    curlFactor: clamp(curlAccum, 0.0, 1.0),
    wakeEnergy: clamp(wakeAccum, 0.0, 1.0),
    orbitalPhase,
  };
}

/**
 * Convenience TSL helper exposing wave displacement, height, crestEnergy, and curlFactor
 * from the unified wave field for vertex-stage consumers.
 */
export function sampleOceanHeightAndCurl(posXZ: TslNode, u: WaveFieldUniforms) {
  const sample = evaluateWaveTSL(posXZ, u);
  return {
    displacement: sample.displacement,
    height: sample.displacement.y,
    normal: sample.normal,
    crestEnergy: sample.crestEnergy,
    curlFactor: sample.curlFactor,
    wakeEnergy: sample.wakeEnergy,
    orbitalPhase: sample.orbitalPhase,
  };
}
