import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  clamp,
  cos,
  dot,
  exp,
  float,
  floor,
  fract,
  max,
  mix,
  screenUV,
  sin,
  sqrt,
  uniform,
  vec3,
} from 'three/tsl';
import {
  CHALKY_SLATE_LINEAR,
  MINERAL_PALETTE,
  type PigmentKey,
} from './palette';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * 3D Value Noise + Irrational-Angle 3-Octave FBM in TSL.
 * Zero polar star harmonics (cos(m*theta)) and zero hardcoded (0,0) radial ramps.
 */
export const hash31Tsl = Fn(([p_in]: [TslNode]) => {
  const p3 = fract(vec3(p_in).mul(vec3(0.1031, 0.103, 0.0973)).add(1.0));
  const d = dot(p3, p3.yzx.add(33.33));
  const pShift = p3.add(d);
  return fract(pShift.x.add(pShift.y).mul(pShift.z));
});

export const valueNoise3DTsl = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const i = floor(p);
  const f = fract(p.add(1.0));
  const u = f.mul(f).mul(float(3.0).sub(f.mul(2.0)));

  const n000 = hash31Tsl(i.add(vec3(0, 0, 0)));
  const n100 = hash31Tsl(i.add(vec3(1, 0, 0)));
  const n010 = hash31Tsl(i.add(vec3(0, 1, 0)));
  const n110 = hash31Tsl(i.add(vec3(1, 1, 0)));
  const n001 = hash31Tsl(i.add(vec3(0, 0, 1)));
  const n101 = hash31Tsl(i.add(vec3(1, 0, 1)));
  const n011 = hash31Tsl(i.add(vec3(0, 1, 1)));
  const n111 = hash31Tsl(i.add(vec3(1, 1, 1)));

  const nx00 = mix(n000, n100, u.x);
  const nx10 = mix(n010, n110, u.x);
  const nx01 = mix(n001, n101, u.x);
  const nx11 = mix(n011, n111, u.x);

  const nxy0 = mix(nx00, nx10, u.y);
  const nxy1 = mix(nx01, nx11, u.y);

  return mix(nxy0, nxy1, u.z);
});

/**
 * 3-octave irrational-angle rotated 3D FBM for mineral grain & paper/slate tooth.
 */
export const mineralToothFbm = Fn(([pos_in, scale_in]: [TslNode, TslNode]) => {
  const p0 = vec3(pos_in).mul(scale_in);
  const v0 = valueNoise3DTsl(p0);

  // Irrational golden-angle rotation between octaves to avoid axis grid artifacts
  const c = float(0.80901699);
  const s = float(0.58778525);
  const p1 = vec3(
    p0.x.mul(c).sub(p0.z.mul(s)),
    p0.y.mul(c).add(p0.x.mul(s)),
    p0.z.mul(c).add(p0.y.mul(s))
  ).mul(2.03);
  const v1 = valueNoise3DTsl(p1);

  const p2 = vec3(
    p1.y.mul(c).sub(p1.x.mul(s)),
    p1.z.mul(c).add(p1.y.mul(s)),
    p1.x.mul(c).add(p1.z.mul(s))
  ).mul(2.07);
  const v2 = valueNoise3DTsl(p2);

  return v0.mul(0.57).add(v1.mul(0.29)).add(v2.mul(0.14));
});

/**
 * 2-Flux Kubelka-Munk pigment reflectance node in TSL.
 * Thickness `d` enters coth(b * S * d) directly with guarded singularities
 * at (K + S) * d < 1e-4 and |u| < 1e-3.
 */
export const kubelkaMunkGlaze = Fn(
  ([kVec, sVec, thicknessScalar, backingVec]: [TslNode, TslNode, TslNode, TslNode]) => {
    const d = max(float(0.0), float(thicknessScalar));
    const K = max(vec3(1e-5), vec3(kVec));
    const S = max(vec3(1e-5), vec3(sVec));
    const Rg = clamp(vec3(backingVec), 0.0, 1.0);

    const sEff = S.mul(d);
    const a = vec3(1.0).add(K.div(S));
    const b = sqrt(max(vec3(1e-7), a.mul(a).sub(1.0)));
    const u = max(vec3(1e-5), b.mul(sEff));

    // Stable coth(u) = (exp(2u) + 1) / (exp(2u) - 1) with clamped upper exponent
    const uClamped = clamp(u, 1e-4, 12.0);
    const exp2u = exp(uClamped.mul(2.0));
    const cothHyperbolic = exp2u.add(1.0).div(max(vec3(1e-6), exp2u.sub(1.0)));
    const cothSeries = vec3(1.0).div(uClamped).add(uClamped.div(3.0));
    const isTinyU = clamp(u.sub(1e-3).mul(1000.0), 0.0, 1.0);
    const cothU = mix(cothSeries, cothHyperbolic, isTinyU);

    const num = vec3(1.0).sub(Rg.mul(a.sub(b.mul(cothU))));
    const den = max(vec3(1e-5), a.sub(Rg).add(b.mul(cothU)));
    const kmRgb = clamp(num.div(den), 0.0, 1.0);

    const totalOptical = K.add(S).mul(d);
    const blendGate = clamp(totalOptical.x.add(totalOptical.y).add(totalOptical.z).mul(250.0), 0.0, 1.0);
    return mix(Rg, kmRgb, blendGate);
  }
);

/**
 * Beer-Lambert optical chord transmittance T = exp(-extinctionRgb * chordLength).
 */
export const beerLambertTransmittance = Fn(([extinctionVec, chordScalar]: [TslNode, TslNode]) => {
  const sigma = max(vec3(0.0), vec3(extinctionVec));
  const ell = max(float(0.0), float(chordScalar));
  return exp(sigma.negate().mul(ell));
});

/**
 * Canonical Chalky Slate (#212529) backdrop shader node with subtle studio softbox
 * falloff (±2.5% luminance) and sub-pixel mineral tooth dithering so gradients never band.
 */
export const chalkySlateBackdropNode = Fn(() => {
  const uv = screenUV;
  const centered = uv.sub(0.5);
  const vignetteDistSq = dot(centered, centered);

  // Base Chalky Slate (#212529) in linear sRGB
  const baseSlate = vec3(
    CHALKY_SLATE_LINEAR[0],
    CHALKY_SLATE_LINEAR[1],
    CHALKY_SLATE_LINEAR[2]
  );

  // Subtle overhead studio softbox falloff + sub-pixel mineral tooth
  const radialFalloff = clamp(float(1.035).sub(vignetteDistSq.mul(0.11)), 0.86, 1.04);
  const grain = hash31Tsl(vec3(uv.mul(1024.0), 0.37)).sub(0.5).mul(0.0014);

  return max(vec3(0.006), baseSlate.mul(radialFalloff).add(grain));
});

/**
 * Creates pre-allocated Three.js TSL uniform nodes for a given MineralPigment.
 */
export function createPigmentUniforms(key: PigmentKey) {
  const p = MINERAL_PALETTE[key];
  return {
    pigment: p,
    rInf: uniform(new THREE.Vector3(...p.rInfLinear)),
    K: uniform(new THREE.Vector3(...p.absorptionK)),
    S: uniform(new THREE.Vector3(...p.scatteringS)),
    extinction: uniform(new THREE.Vector3(...p.extinctionRgb)),
  };
}

export type PigmentUniformPack = ReturnType<typeof createPigmentUniforms>;

/**
 * Suppress unused import warnings in strict builds while keeping helpers available.
 */
export const _tslMathDeps = { abs, cos, sin };
