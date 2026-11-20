import * as THREE from 'three/webgpu';
import {
  Fn,
  cameraPosition,
  clamp,
  dot,
  float,
  floor,
  length,
  max,
  mix,
  normalize,
  positionWorld,
  pow,
  sin,
  smoothstep,
  sqrt,
  vec2,
  vec3,
} from 'three/tsl';
import type { WaveFieldUniforms } from './waveField';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface SkyDomeRig {
  mesh: THREE.Mesh;
  dispose: () => void;
}

/**
 * Negative-safe fractional part (`x - floor(x)`), immune to Metal fast-math
 * `fract(x)` negative-UV truncation artifacts.
 */
const safeFract3 = Fn(([v_in]: [TslNode]) => {
  const v = vec3(v_in);
  return v.sub(floor(v));
});

const safeFract1 = Fn(([x_in]: [TslNode]) => {
  const x = float(x_in);
  return x.sub(floor(x));
});

/**
 * Fast 3D -> 1D hash in [0, 1) with zero periodic grid alignment.
 */
const hash31 = Fn(([p_in]: [TslNode]) => {
  const p3 = safeFract3(vec3(p_in).mul(vec3(0.1031, 0.103, 0.0973)));
  const d = dot(p3, p3.yzx.add(33.33));
  const shifted = p3.add(d);
  return safeFract1(shifted.x.add(shifted.y).mul(shifted.z));
});

/**
 * 3D -> 3D jitter hash in [0, 1)^3 for pinpoint star placement.
 */
const hash33 = Fn(([p_in]: [TslNode]) => {
  const p3 = safeFract3(vec3(p_in).mul(vec3(0.1031, 0.103, 0.0973)));
  const d = dot(p3, p3.yxz.add(33.33));
  const s = p3.add(d);
  return safeFract3(
    vec3(
      s.x.add(s.y).mul(s.z),
      s.x.add(s.z).mul(s.y),
      s.y.add(s.z).mul(s.x)
    )
  );
});

/**
 * Smooth quintic-interpolated 3D value noise (`C^2` continuous).
 */
const valueNoise3D = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const i = floor(p);
  const f = p.sub(i);
  // Quintic Hermite: 6t^5 - 15t^4 + 10t^3
  const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0));

  const n000 = hash31(i.add(vec3(0.0, 0.0, 0.0)));
  const n100 = hash31(i.add(vec3(1.0, 0.0, 0.0)));
  const n010 = hash31(i.add(vec3(0.0, 1.0, 0.0)));
  const n110 = hash31(i.add(vec3(1.0, 1.0, 0.0)));
  const n001 = hash31(i.add(vec3(0.0, 0.0, 1.0)));
  const n101 = hash31(i.add(vec3(1.0, 0.0, 1.0)));
  const n011 = hash31(i.add(vec3(0.0, 1.0, 1.0)));
  const n111 = hash31(i.add(vec3(1.0, 1.0, 1.0)));

  const nx00 = mix(n000, n100, u.x);
  const nx10 = mix(n010, n110, u.x);
  const nx01 = mix(n001, n101, u.x);
  const nx11 = mix(n011, n111, u.x);

  const nxy0 = mix(nx00, nx10, u.y);
  const nxy1 = mix(nx01, nx11, u.y);

  return mix(nxy0, nxy1, u.z);
});

/**
 * 4-octave irrational-angle rotated 3D FBM for volumetric-feel twilight clouds.
 * Zero polar star harmonics, zero Voronoi bisector cracks.
 */
const skyCloudFbm4 = Fn(([p_in]: [TslNode]) => {
  const p0 = vec3(p_in);
  const v0 = valueNoise3D(p0);

  // Golden-angle rotation between octaves eliminates axis-aligned cloud streaks
  const c = float(0.80901699);
  const s = float(0.58778525);

  const p1 = vec3(
    p0.x.mul(c).sub(p0.z.mul(s)),
    p0.y.mul(c).add(p0.x.mul(s)),
    p0.z.mul(c).add(p0.y.mul(s))
  ).mul(2.03).add(vec3(17.3, -9.1, 11.7));
  const v1 = valueNoise3D(p1);

  const p2 = vec3(
    p1.y.mul(c).sub(p1.x.mul(s)),
    p1.z.mul(c).add(p1.y.mul(s)),
    p1.x.mul(c).add(p1.z.mul(s))
  ).mul(2.07).add(vec3(-13.4, 21.8, -7.2));
  const v2 = valueNoise3D(p2);

  const p3 = vec3(
    p2.z.mul(c).sub(p2.y.mul(s)),
    p2.x.mul(c).add(p2.z.mul(s)),
    p2.y.mul(c).add(p2.x.mul(s))
  ).mul(2.05).add(vec3(6.9, -15.6, 19.4));
  const v3 = valueNoise3D(p3);

  return v0.mul(0.51).add(v1.mul(0.26)).add(v2.mul(0.145)).add(v3.mul(0.085));
});

/**
 * Continuous mode classifiers derived purely from preset uniforms (zero rebuilds):
 * - goldenMask: 1.0 in Golden Swell, 0.0 in Storm/Biolum
 * - stormMask:  1.0 in Abyssal Storm, 0.0 in Golden/Biolum
 * - solarGain:  1.0 in Golden Swell, 0.26 in cool Storm/Biolum modes
 */
function classifySkyMode(sunColor: TslNode, biolumGain: TslNode) {
  const warmth = clamp(sunColor.r.sub(sunColor.b).mul(2.4), float(0.0), float(1.0));
  const nonBiolum = float(1.0).sub(clamp(biolumGain, float(0.0), float(1.0)));
  const goldenMask = warmth.mul(nonBiolum);
  const stormMask = float(1.0).sub(warmth).mul(nonBiolum);
  const solarGain = mix(float(0.26), float(1.0), goldenMask);
  return { goldenMask, stormMask, nonBiolum, solarGain };
}

/**
 * Shared azimuth-aware atmospheric horizon haze color.
 * Evaluated identically by BOTH `skyDome.ts` and `oceanSurface.ts` so the distant
 * sea melts seamlessly into the twilight sun haze while outer horizon flanks stay
 * cool moody slate-indigo (zero muddy brown tint on flanks).
 */
export function evaluateHorizonHazeTSL(rayDirIn: TslNode, u: WaveFieldUniforms): TslNode {
  const rayDir = normalize(vec3(rayDirIn));
  const sunDir = normalize(vec3(u.uSunDir));
  const skyZenith = u.uSkyZenith.rgb;
  const skyHorizon = u.uSkyHorizon.rgb;
  const sunColor = u.uSunColor.rgb;
  const biolumGain = float(u.uBiolumGain);

  const { goldenMask, solarGain } = classifySkyMode(sunColor, biolumGain);

  const rayXZ = normalize(vec2(rayDir.x, rayDir.z).add(vec2(1e-5, 0.0)));
  const sunXZ = normalize(vec2(sunDir.x, sunDir.z).add(vec2(1e-5, 0.0)));
  const azimCos = clamp(dot(rayXZ, sunXZ), float(-1.0), float(1.0));

  // Keep warm golden-apricot horizon tightly centered near the sun azimuth;
  // flanks cool smoothly to smoky twilight slate-indigo matching fused_hero_sea_of_c.
  const azimWarm = pow(
    clamp(azimCos.mul(0.5).add(0.5), float(1e-5), float(1.0)),
    float(6.2)
  );
  const azimCore = pow(max(azimCos, float(1e-5)), float(14.0));

  const coolFlank = mix(
    skyZenith.mul(1.85).add(vec3(0.012, 0.018, 0.036).mul(goldenMask)),
    skyHorizon.mul(0.24),
    float(0.32)
  );
  const sunwardHorizon = mix(skyHorizon.mul(0.94), sunColor.mul(0.96), float(0.24));
  const baseHaze = mix(coolFlank, sunwardHorizon, azimWarm);

  // Subtle forward Mie band right beneath the low solar azimuth
  const sunVis = smoothstep(float(-0.12), float(0.04), sunDir.y);
  const mieBand = sunColor.mul(azimCore.mul(0.18).mul(sunVis).mul(solarGain));

  return baseHaze.add(mieBand);
}

/**
 * Lightweight analytical twilight sky reflection for `oceanSurface.ts`.
 * Reflects cool twilight sapphire-indigo overhead & across flanks (preventing muddy
 * brown reflections on midground swells and panoramic overviews), while preserving
 * warm amber-gold reflection strictly along the low grazing sun corridor.
 */
export function evaluateSkyReflectionTSL(reflDirIn: TslNode, u: WaveFieldUniforms): TslNode {
  const dir = normalize(vec3(reflDirIn));
  const sunDir = normalize(vec3(u.uSunDir));
  const skyZenith = u.uSkyZenith.rgb;
  const skyHorizon = u.uSkyHorizon.rgb;
  const sunColor = u.uSunColor.rgb;
  const biolumGain = float(u.uBiolumGain);

  const { goldenMask, nonBiolum, solarGain } = classifySkyMode(sunColor, biolumGain);
  const elev = clamp(dir.y, float(0.0), float(1.0));
  const cosSun = max(dot(dir, sunDir), float(1e-5));

  // Cool twilight sapphire-slate ambient sky for non-glint rolling waves & panoramic views
  const coolOverhead = skyZenith
    .mul(1.65)
    .add(vec3(0.008, 0.022, 0.056).mul(nonBiolum));
  const coolHorizon = mix(
    coolOverhead.mul(1.25),
    skyHorizon.mul(0.30).add(vec3(0.010, 0.028, 0.064).mul(goldenMask)),
    float(0.35)
  );

  const elevBand = clamp(elev.div(0.16), float(0.0), float(1.0));
  const zenithT = pow(max(elevBand, float(1e-5)), float(0.62));
  const coolSky = mix(coolHorizon, coolOverhead.mul(0.82), zenithT);

  // Warm golden horizon reflection restricted strictly to low grazing rays near sun azimuth
  const sunCorridor = pow(cosSun, float(9.0)).mul(
    float(1.0).sub(smoothstep(float(0.02), float(0.16), elev))
  );
  const horizonCol = evaluateHorizonHazeTSL(dir, u);
  const skyGrad = mix(coolSky, horizonCol, sunCorridor.mul(0.75));

  // Tight low-sun reflection aureole
  const sunVis = smoothstep(float(-0.10), float(0.04), sunDir.y);
  const mieGlow = pow(cosSun, float(14.0))
    .mul(0.14)
    .add(pow(cosSun, float(58.0)).mul(0.28))
    .mul(solarGain);

  const cloudCanopy = smoothstep(float(0.08), float(0.36), elev).mul(0.24);
  const baseRefl = skyGrad.mul(float(1.0).sub(cloudCanopy));

  return baseRefl.add(sunColor.mul(mieGlow.mul(sunVis)));
}

/**
 * Procedural Starfield Layer (high-altitude crisp pinpoints in Bioluminescent mode).
 */
const evaluateStarLayer = Fn(([dir_in, scale_in, time_in]: [TslNode, TslNode, TslNode]) => {
  const p = vec3(dir_in).mul(float(scale_in));
  const cell = floor(p);
  const local = p.sub(cell);

  const rnd = hash33(cell);
  const starPos = vec3(0.22).add(rnd.mul(0.56));
  const dist = length(local.sub(starPos));

  const rarity = hash31(cell.add(vec3(41.7, 19.3, 73.1)));
  const isStar = smoothstep(float(0.958), float(0.994), rarity);
  const core = float(1.0).sub(smoothstep(float(0.016), float(0.115), dist));

  const twinklePhase = float(time_in).mul(float(1.3).add(rnd.x.mul(1.9))).add(rnd.y.mul(6.28318));
  const twinkle = float(0.68).add(sin(twinklePhase).mul(0.32));

  const starTint = mix(vec3(0.76, 0.88, 1.0), vec3(1.0, 0.94, 0.84), rnd.z);
  return starTint.mul(core.mul(isStar).mul(twinkle));
});

/**
 * Creates the infinite atmospheric twilight/golden-hour/storm/biolum Sky Dome.
 * Vertical Rayleigh gradient and stratocumulus cloud deck are calibrated for the
 * 40 deg FOV hero framing (`elev ∈ [0.0, 0.21]`) with textured violet-slate indigo
 * clouds in Golden Swell, overcast thunderheads in Storm, and starlit midnight in Biolum.
 */
export function createSkyDome(u: WaveFieldUniforms): SkyDomeRig {
  const geometry = new THREE.SphereGeometry(265, 48, 32);

  const material = new THREE.MeshBasicNodeMaterial({
    side: THREE.BackSide,
    fog: false,
    depthWrite: false,
    depthTest: true,
  });

  const skyColorFn = Fn(() => {
    // True view ray from camera guarantees exact eye-level horizon at any camera elevation
    const dir = normalize(positionWorld.sub(cameraPosition));
    const sunDir = normalize(vec3(u.uSunDir));
    const skyZenith = u.uSkyZenith.rgb;
    const skyHorizon = u.uSkyHorizon.rgb;
    const sunColor = u.uSunColor.rgb;
    const time = float(u.uTime);
    const biolumGain = float(u.uBiolumGain);

    const { goldenMask, stormMask, solarGain } = classifySkyMode(sunColor, biolumGain);
    const elev = clamp(dir.y, float(0.0), float(1.0));
    const cosSun = dot(dir, sunDir);
    const cosSunPos = max(cosSun, float(1e-5));
    const sunVis = smoothstep(float(-0.10), float(0.04), sunDir.y);

    // 1. Azimuth-aware seamless horizon color (shared identically with oceanSurface.ts)
    const horizonCol = evaluateHorizonHazeTSL(dir, u);

    // 2. Compressed vertical Rayleigh gradient (`elev ∈ [0.0, 0.15]`) matching 40° FOV hero sky:
    //    warm horizon hugs elev 0..0.055, dusty mauve-indigo transitions across 0.05..0.12,
    //    and upper-left/right corners reach textured violet-slate indigo / midnight by elev >= 0.14.
    const elevNorm = clamp(elev.div(0.15), float(0.0), float(1.0));
    const rayleighT = pow(max(elevNorm, float(1e-5)), float(0.66));
    const goldenUpperSky = vec3(0.020, 0.028, 0.056);
    const zenithTarget = mix(skyZenith.mul(0.95), goldenUpperSky, goldenMask);
    const midTwilight = mix(zenithTarget.mul(1.35), skyHorizon.mul(0.42), float(0.30));
    const lowerSky = mix(horizonCol, midTwilight, smoothstep(float(0.0), float(0.46), elevNorm));
    let skyCol = mix(lowerSky, zenithTarget, rayleighT);

    // Tight warm horizon ribbon strictly hugging elev 0..0.09 near sun azimuth
    const ribbonDecay = pow(
      max(float(1.0).sub(clamp(elev.div(0.10), float(0.0), float(1.0))), float(1e-5)),
      float(3.2)
    );
    const sunAzimNarrow = pow(
      clamp(cosSun.mul(0.5).add(0.5), float(1e-5), float(1.0)),
      float(4.5)
    );
    skyCol = skyCol.add(
      mix(skyHorizon, sunColor, float(0.36)).mul(
        ribbonDecay.mul(0.20).mul(sunAzimNarrow).mul(solarGain)
      )
    );

    // 3. Textured moody stratocumulus cloud deck on curved planetary shell
    //    Evaluated BEFORE solar disk so overcast storm clouds naturally occlude/diffuse the sun.
    const tShellMain = sqrt(elev.mul(elev).mul(196.0).add(29.0)).sub(elev.mul(14.0));
    const driftMain = vec2(time.mul(0.014), time.mul(-0.008));
    const cloudXZ = dir.xz.mul(tShellMain).mul(0.84).add(driftMain);

    // Low-frequency domain warp gives organic sweeping dusk cloud banks
    const warpX = valueNoise3D(vec3(cloudXZ.mul(0.52), time.mul(0.010))).sub(0.5);
    const warpZ = valueNoise3D(vec3(cloudXZ.mul(0.52).add(vec2(19.4, -11.2)), time.mul(0.010))).sub(0.5);
    const warpedXZ = cloudXZ.add(vec2(warpX, warpZ).mul(0.72));

    const fbmMain = skyCloudFbm4(vec3(warpedXZ.x, time.mul(0.006), warpedXZ.y));
    // Secondary fine billow detail gives textured slate-indigo cloud relief (no flat dark blobs)
    const fbmDetail = skyCloudFbm4(vec3(warpedXZ.x.mul(2.15), float(2.7), warpedXZ.y.mul(2.15)));

    // Part clouds around the low golden sun gap (closed over in overcast storm!)
    const clearStrength = mix(float(0.12), float(0.72), goldenMask);
    const sunCorridorClear = float(1.0).sub(
      pow(cosSunPos, float(8.0))
        .mul(float(1.0).sub(smoothstep(float(0.02), float(0.14), elev)))
        .mul(clearStrength)
    );

    // Rich textured canopy framing upper corners (heavier overcast deck in storm mode)
    const canopyBoost = smoothstep(float(0.04), float(0.17), elev)
      .mul(float(0.16).add(stormMask.mul(0.18)));
    const rawDensity = fbmMain.mul(0.78).add(fbmDetail.mul(0.22)).add(canopyBoost).mul(sunCorridorClear);
    const cloudMain = smoothstep(float(0.38), float(0.76), rawDensity);

    // Sunward directional thickness gradient -> warm sunlit undersides & silver wisps
    const sunStepXZ = normalize(vec2(sunDir.x, sunDir.z).add(vec2(1e-5, 0.0))).mul(0.24);
    const fbmSun = skyCloudFbm4(vec3(warpedXZ.x.add(sunStepXZ.x), time.mul(0.006), warpedXZ.y.add(sunStepXZ.y)));
    const sunRimGrad = clamp(fbmMain.sub(fbmSun).mul(3.2).add(0.34), float(0.0), float(1.0));

    // Textured moody violet-slate indigo cloud cores in Golden Swell (matching fused_hero_sea_of_c),
    // churning dark storm slate in Storm, and pitch velvety midnight in Bioluminescent mode.
    const forwardScatter = pow(cosSunPos, float(8.5)).mul(sunVis).mul(solarGain);
    const billowRelief = float(0.74).add(fbmDetail.mul(0.56));
    const goldenCloudCore = vec3(0.017, 0.023, 0.046).mul(billowRelief);
    const stormCloudCore = vec3(0.010, 0.016, 0.026).mul(billowRelief);
    const biolumCloudCore = skyZenith.mul(0.72);

    const cloudCoreDark = mix(
      mix(biolumCloudCore, stormCloudCore, stormMask),
      goldenCloudCore,
      goldenMask
    );
    const cloudMidShade = mix(
      cloudCoreDark.mul(1.42),
      mix(vec3(0.045, 0.048, 0.072).mul(goldenMask).add(skyZenith.mul(1.3)), skyHorizon.mul(0.46), float(0.45)),
      float(0.26).add(sunAzimNarrow.mul(0.36))
    );
    const cloudUndersideLit = mix(skyHorizon.mul(0.86), sunColor, float(0.44)).mul(
      float(0.58).add(forwardScatter.mul(0.64))
    );

    const thinEdge = float(1.0).sub(smoothstep(float(0.22), float(0.82), cloudMain));
    const litEdgeWeight = clamp(
      sunRimGrad.mul(0.48).add(forwardScatter.mul(0.52)).mul(float(0.42).add(thinEdge.mul(0.58))),
      float(0.0),
      float(1.0)
    );
    const cloudBaseCol = mix(cloudCoreDark, cloudMidShade, smoothstep(float(0.18), float(0.70), litEdgeWeight));
    const cloudFinalCol = mix(
      cloudBaseCol,
      cloudUndersideLit,
      smoothstep(float(0.48), float(0.95), litEdgeWeight).mul(float(0.28).add(forwardScatter.mul(0.72)))
    );

    // Subtle secondary higher wispy dusk deck
    const tShellHigh = sqrt(elev.mul(elev).mul(400.0).add(41.0)).sub(elev.mul(20.0));
    const highXZ = dir.xz.mul(tShellHigh).mul(0.96).add(vec2(time.mul(-0.009), time.mul(0.005)));
    const fbmHigh = skyCloudFbm4(vec3(highXZ.x, float(4.3), highXZ.y));
    const cloudHigh = smoothstep(float(0.54), float(0.84), fbmHigh).mul(0.30);
    const highTint = mix(
      zenithTarget.mul(1.22),
      mix(skyHorizon.mul(0.58), sunColor.mul(0.70), float(0.36)),
      float(0.18).add(forwardScatter.mul(0.54))
    );

    // Start clouds smoothly above horizon mist; thin gently in biolum for crisp midnight stars
    const horizonCloudFade = smoothstep(float(0.012), float(0.085), elev);
    const biolumCloudThin = mix(float(1.0), float(0.54), clamp(biolumGain, float(0.0), float(1.0)));
    const highAlpha = cloudHigh.mul(horizonCloudFade).mul(biolumCloudThin);
    const mainAlpha = cloudMain.mul(float(0.84).add(stormMask.mul(0.10))).mul(horizonCloudFade).mul(biolumCloudThin);

    // 4. Soft low horizon sun/moon + smooth exponential golden inner corona
    //    Tightened soft disk (`smoothstep(0.99962, 0.99992)`) avoids a hard ping-pong ball;
    //    in Storm mode, overcast clouds suppress the sharp disk into a moody diffused break.
    const aboveHorizonMask = smoothstep(float(-0.014), float(0.012), dir.y);
    const mieOuter = pow(cosSunPos, float(10.0)).mul(0.14).mul(solarGain);
    const mieMid = pow(cosSunPos, float(42.0)).mul(0.24).mul(solarGain);
    const mieInnerCorona = pow(cosSunPos, float(85.0)).mul(0.45).mul(solarGain);
    const mieHotCorona = pow(cosSunPos, float(300.0)).mul(0.70).mul(solarGain);

    const sunDiskRaw = smoothstep(float(0.99962), float(0.99992), cosSun);
    const stormDiskSuppress = mix(
      float(1.0),
      float(1.0).sub(mainAlpha.mul(0.92)).mul(0.14),
      stormMask
    );
    const sunDisk = sunDiskRaw.mul(stormDiskSuppress);
    const diskGain = mix(float(0.52), float(1.38), solarGain);

    // Diffused overcast silver glow in storm clouds without a hard circular edge
    const stormOvercastGlow = sunColor
      .mul(pow(cosSunPos, float(7.5)).mul(0.14).add(pow(cosSunPos, float(24.0)).mul(0.10)))
      .mul(stormMask);

    const solarRadiance = sunColor
      .mul(mieOuter.add(mieMid).add(mieInnerCorona).add(mieHotCorona))
      .add(mix(sunColor, vec3(1.0, 0.97, 0.89), float(0.42)).mul(sunDisk.mul(diskGain)))
      .add(stormOvercastGlow)
      .mul(sunVis)
      .mul(aboveHorizonMask);

    skyCol = skyCol.add(solarRadiance);

    // 5. High-altitude crisp twinkling stars in Bioluminescent mode
    const starModeGain = smoothstep(float(0.25), float(0.85), biolumGain);
    const starElevMask = smoothstep(float(0.028), float(0.11), elev);
    const starsNear = evaluateStarLayer(dir, float(135.0), time);
    const starsFar = evaluateStarLayer(dir.add(vec3(0.17, 0.31, -0.23)), float(230.0), time.mul(1.25));
    const starTotal = starsNear.add(starsFar.mul(0.72)).mul(starModeGain).mul(starElevMask).mul(1.12);

    // Composite stars -> wispy high deck -> main textured stratocumulus deck
    const skyWithStars = skyCol.add(starTotal);
    const skyWithHigh = mix(skyWithStars, highTint, highAlpha);
    let finalSky = mix(skyWithHigh, cloudFinalCol, mainAlpha);

    // 6. Below-horizon seamless continuation into exact horizon haze -> deep twilight floor
    const belowT = float(1.0).sub(smoothstep(float(-0.18), float(0.0), dir.y));
    const belowFloor = mix(horizonCol, u.uWaterDeep.rgb.mul(0.45), belowT);
    finalSky = mix(finalSky, belowFloor, float(1.0).sub(smoothstep(float(-0.025), float(0.004), dir.y)));

    // Sub-LSB dither prevents 8-bit twilight gradient banding
    const dither = hash31(dir.mul(731.3).add(vec3(0.37, 0.71, 0.19))).sub(0.5).mul(0.0018);
    return max(vec3(0.0), finalSky.add(dither));
  });

  material.colorNode = skyColorFn();

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'SeaOfC_SkyDome';
  mesh.frustumCulled = false;
  mesh.renderOrder = -10;

  return {
    mesh,
    dispose: () => {
      geometry.dispose();
      material.dispose();
    },
  };
}
