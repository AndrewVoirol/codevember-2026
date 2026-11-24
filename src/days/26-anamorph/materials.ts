import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  abs,
  attribute,
  clamp,
  dot,
  exp,
  float,
  fract,
  fwidth,
  length,
  max,
  min,
  mix,
  normalLocal,
  normalWorldGeometry,
  normalize,
  positionLocal,
  positionWorld,
  select,
  smoothstep,
  step,
  transformNormalToView,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import {
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import {
  ARCHITECTURAL_SLABS,
  OCCLUSION_FRONT_DOT_MIN,
  OCCLUSION_MIN_T,
  OCCLUSION_SELF_BIAS,
  PENROSE_PRIMARY_SEGS,
  PENROSE_SECONDARY_SEGS,
  SLAB_COUNT,
  STENCIL_CONFIGS,
  type VantageBasis,
} from './anamorphMath';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface AnamorphMaterials {
  readonly courtyardMaterial: THREE.MeshStandardNodeMaterial;
  readonly brassBezelMaterial: THREE.MeshStandardNodeMaterial;
  readonly brassTicksMaterial: THREE.MeshStandardNodeMaterial;
  readonly syncVantageBasis: (basis: VantageBasis) => void;
  readonly syncStencil: (stencilIndex: number) => void;
  readonly uFlashProgress: { value: number };
  readonly uLockGlow: { value: number };
  readonly uRingAlpha: { value: number };
  readonly uRingHover: { value: number };
  readonly uPointerWorld: { value: THREE.Vector3 };
  readonly uPointerActive: { value: number };
  readonly dispose: () => void;
}

/**
 * 2D Euclidean distance from point p to segment (a, b) in TSL.
 */
const sdSegment2DTsl = Fn(([p_in, a_in, b_in]: [TslNode, TslNode, TslNode]) => {
  const p = vec2(p_in);
  const a = vec2(a_in);
  const b = vec2(b_in);
  const pa = p.sub(a);
  const ba = b.sub(a);
  const h = clamp(dot(pa, ba).div(max(float(1e-7), dot(ba, ba))), 0.0, 1.0);
  return length(pa.sub(ba.mul(h)));
});

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space,
 * normalized independent of scale so cast concrete & travertine grain stays crisp
 * without low-frequency crumpled blotches.
 */
const perturbMineralNormal = Fn(
  ([baseNorm_in, pos_in, scaleVec_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in).mul(vec3(scaleVec_in));
    const eps = float(0.085);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), float(1.0));
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), float(1.0));
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), float(1.0));
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), float(1.0));

    const maxScale = max(
      vec3(scaleVec_in).x,
      max(vec3(scaleVec_in).y, max(vec3(scaleVec_in).z, float(1.0)))
    );
    const anisoWeight = vec3(scaleVec_in).div(maxScale);
    const grad = k1
      .mul(f1)
      .add(k2.mul(f2))
      .add(k3.mul(f3))
      .add(k4.mul(f4))
      .div(eps.mul(4.0))
      .mul(anisoWeight);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

export function createAnamorphMaterials(): AnamorphMaterials {
  // Static 30-slab AABB uniform arrays (single source of truth matching anamorphMath.ts)
  const boxMinArr: THREE.Vector4[] = [];
  const boxMaxArr: THREE.Vector4[] = [];
  for (let i = 0; i < SLAB_COUNT; i++) {
    const s = ARCHITECTURAL_SLABS[i];
    boxMinArr.push(new THREE.Vector4(s.min[0], s.min[1], s.min[2], 0));
    boxMaxArr.push(new THREE.Vector4(s.max[0], s.max[1], s.max[2], 0));
  }
  const uBoxMin = uniformArray<'vec4'>(boxMinArr, 'vec4');
  const uBoxMax = uniformArray<'vec4'>(boxMaxArr, 'vec4');

  // Dynamic projective vantage uniforms (updated in <0.01ms on re-projection, zero allocations)
  const uVantagePos = uniform(new THREE.Vector3(2.2, 1.5, 4.0));
  const uVantageForward = uniform(new THREE.Vector3(0, 0, -1));
  const uVantageRight = uniform(new THREE.Vector3(1, 0, 0));
  const uVantageUp = uniform(new THREE.Vector3(0, 1, 0));
  // x = tanHalfFov * stencilScale, y = activeStencilIndex, z = flashProgress, w = lockGlow
  const uVantageScale = uniform(0.3153);
  const uActiveStencil = uniform(0);
  const uFlashProgress = uniform(1.0);
  const uLockGlow = uniform(0.0);

  // Active stencil mineral pigment Kubelka-Munk (K, S) uniforms
  const p0Pri = MINERAL_PALETTE[STENCIL_CONFIGS[0].primaryPigment];
  const p0Sec = MINERAL_PALETTE[STENCIL_CONFIGS[0].secondaryPigment];
  const uPriK = uniform(new THREE.Vector3(...p0Pri.absorptionK));
  const uPriS = uniform(new THREE.Vector3(...p0Pri.scatteringS));
  const uSecK = uniform(new THREE.Vector3(...p0Sec.absorptionK));
  const uSecS = uniform(new THREE.Vector3(...p0Sec.scatteringS));

  // Pointer hover warmth & brass sight-ring uniforms
  const uPointerWorld = uniform(new THREE.Vector3(0, 0, 0));
  const uPointerActive = uniform(0.0);
  const uRingAlpha = uniform(1.0);
  const uRingHover = uniform(0.0);

  // Mineral substrate anchor colors (strictly from MINERAL_PALETTE, zero neons)
  const boneCalciteRgb = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const warmKaolinRgb = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const rawUmberRgb = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const yellowOchreRgb = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const realgarAmberRgb = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);

  /**
   * Combined single-pass 30-slab analytical ray-box occlusion (from vantagePos -> wp)
   * AND 2-radius analytical architectural crevice ambient occlusion.
   * Returns vec3(visibility01, tightCreviceAO, broadCornerAO).
   * Pure ALU, zero derivatives, zero texture taps — 100% uniform control flow.
   */
  const evalOcclusionAndCreviceAO = Fn(([wp_in, gn_in]: [TslNode, TslNode]) => {
    const wp = vec3(wp_in);
    const gn = normalize(vec3(gn_in));

    const toVantage = uVantagePos.sub(wp);
    const dist = max(float(0.05), length(toVantage));
    const dirToVantage = toVantage.div(dist);
    const rayDir = dirToVantage.negate();

    // Strict front-facing gate using unperturbed geometric normal
    const nDotV = dot(gn, dirToVantage);
    const frontMask = smoothstep(
      float(OCCLUSION_FRONT_DOT_MIN),
      float(OCCLUSION_FRONT_DOT_MIN + 0.045),
      nDotV
    );

    // Zero-division-safe inverse ray direction (matches anamorphMath.ts bit-for-bit)
    const safeDx = select(rayDir.x.lessThan(0.0), min(rayDir.x, float(-1e-6)), max(rayDir.x, float(1e-6)));
    const safeDy = select(rayDir.y.lessThan(0.0), min(rayDir.y, float(-1e-6)), max(rayDir.y, float(1e-6)));
    const safeDz = select(rayDir.z.lessThan(0.0), min(rayDir.z, float(-1e-6)), max(rayDir.z, float(1e-6)));
    const invD = vec3(float(1.0).div(safeDx), float(1.0).div(safeDy), float(1.0).div(safeDz));

    const maxT = dist.sub(float(OCCLUSION_SELF_BIAS));

    // Two normal-offset sample probes for crisp analytical architectural crevice & soffit AO
    const aoProbeTight = wp.add(gn.mul(0.065));
    const aoProbeBroad = wp.add(gn.mul(0.20));

    const occSum = float(0.0).toVar();
    const tightAO = float(0.0).toVar();
    const broadAO = float(0.0).toVar();

    Loop(SLAB_COUNT, ({ i }) => {
      const bMin = vec4(uBoxMin.element(i)).xyz;
      const bMax = vec4(uBoxMax.element(i)).xyz;

      // 1. Exact slab-method ray-AABB intersection from uVantagePos -> wp
      const t1 = bMin.sub(uVantagePos).mul(invD);
      const t2 = bMax.sub(uVantagePos).mul(invD);
      const tMinV = min(t1, t2);
      const tMaxV = max(t1, t2);
      const tEnter = max(tMinV.x, max(tMinV.y, tMinV.z));
      const tExit = min(tMaxV.x, min(tMaxV.y, tMaxV.z));
      const clampedEnter = max(tEnter, float(OCCLUSION_MIN_T));

      // Smooth 1-2px optical edge on grazing occluder silhouettes, watertight inside slabs
      const chord = tExit.sub(clampedEnter);
      const hitBeforeSurface = step(tEnter, maxT);
      const hitStrength = smoothstep(float(0.0005), float(0.0038), chord).mul(hitBeforeSurface);
      occSum.assign(max(occSum, hitStrength));

      // 2. Analytical architectural crevice AO (self-slab naturally sits at dist >= 0.058 > 0.054 -> 0 self-AO)
      const dTight = length(max(max(bMin.sub(aoProbeTight), aoProbeTight.sub(bMax)), vec3(0.0)));
      const dBroad = length(max(max(bMin.sub(aoProbeBroad), aoProbeBroad.sub(bMax)), vec3(0.0)));

      const cTight = float(1.0).sub(smoothstep(float(0.0), float(0.054), dTight));
      const cBroad = float(1.0).sub(smoothstep(float(0.0), float(0.175), dBroad));
      tightAO.assign(max(tightAO, cTight));
      broadAO.assign(max(broadAO, cBroad));
    });

    const visibility = clamp(frontMask.mul(float(1.0).sub(occSum)), 0.0, 1.0);
    return vec3(visibility, clamp(tightAO, 0.0, 1.0), clamp(broadAO, 0.0, 1.0));
  });

  /**
   * Evaluates the active Felice Varini geometric stencil SDF in isotropic (u, v) space
   * with precomputed uniform-control-flow `aaWidth`.
   * Returns vec2(primaryMask, secondaryMask).
   */
  const evalStencilTsl = Fn(([uv_in, aa_in]: [TslNode, TslNode]) => {
    const uv = vec2(uv_in);
    const aa = max(float(0.0008), float(aa_in));
    const u = uv.x;
    const v = uv.y;
    const au = abs(u);
    const av = abs(v);

    const priMask = float(0.0).toVar();
    const secMask = float(0.0).toVar();

    const stroke = (dist: TslNode, halfW: number) =>
      float(1.0).sub(smoothstep(float(halfW).sub(aa), float(halfW).add(aa), dist));

    If(uActiveStencil.lessThan(0.5), () => {
      // Stencil 0: Concentric Rings (Cinnabar) & Rotated Diamond + Ticks (Yellow Ochre)
      const r = length(uv);
      const dRing = min(
        abs(r.sub(0.43)),
        min(abs(r.sub(0.285)), abs(r.sub(0.14)))
      );
      priMask.assign(stroke(dRing, 0.021));

      const dDiamond = abs(au.add(av).mul(0.70710678).sub(0.295));
      const absP = vec2(au, av);
      const dTickX = sdSegment2DTsl(absP, vec2(0.465, 0.0), vec2(0.535, 0.0));
      const dTickY = sdSegment2DTsl(absP, vec2(0.0, 0.465), vec2(0.0, 0.535));
      const dSec = min(dDiamond, min(dTickX, dTickY));
      secMask.assign(stroke(dSec, 0.013));
    })
      .ElseIf(uActiveStencil.lessThan(1.5), () => {
        // Stencil 1: Impossible Penrose Tribar (Yellow Ochre + Azurite Prussian)
        let dPri: TslNode = float(10.0);
        for (let s = 0; s < PENROSE_PRIMARY_SEGS.length; s++) {
          const seg = PENROSE_PRIMARY_SEGS[s];
          dPri = min(dPri, sdSegment2DTsl(uv, vec2(seg[0], seg[1]), vec2(seg[2], seg[3])));
        }
        let dSec: TslNode = float(10.0);
        for (let s = 0; s < PENROSE_SECONDARY_SEGS.length; s++) {
          const seg = PENROSE_SECONDARY_SEGS[s];
          dSec = min(dSec, sdSegment2DTsl(uv, vec2(seg[0], seg[1]), vec2(seg[2], seg[3])));
        }
        priMask.assign(stroke(dPri, 0.017));
        secMask.assign(stroke(dSec, 0.015));
      })
      .ElseIf(uActiveStencil.lessThan(2.5), () => {
        // Stencil 2: Nested Isometric Cube (Azurite Prussian + Realgar Amber)
        const R = 0.44;
        const hX = R * 0.86602540378;
        const hY = R * 0.5;
        const v0 = vec2(0.0, R);
        const v1 = vec2(hX, hY);
        const v2 = vec2(hX, -hY);
        const v3 = vec2(0.0, -R);
        const v4 = vec2(-hX, -hY);
        const v5 = vec2(-hX, hY);
        const origin = vec2(0.0, 0.0);

        let dPri = sdSegment2DTsl(uv, v0, v1);
        dPri = min(dPri, sdSegment2DTsl(uv, v1, v2));
        dPri = min(dPri, sdSegment2DTsl(uv, v2, v3));
        dPri = min(dPri, sdSegment2DTsl(uv, v3, v4));
        dPri = min(dPri, sdSegment2DTsl(uv, v4, v5));
        dPri = min(dPri, sdSegment2DTsl(uv, v5, v0));
        dPri = min(dPri, sdSegment2DTsl(uv, origin, v3));
        dPri = min(dPri, sdSegment2DTsl(uv, origin, v1));
        dPri = min(dPri, sdSegment2DTsl(uv, origin, v5));
        priMask.assign(stroke(dPri, 0.018));

        const rIn = 0.22;
        const ix = rIn * 0.86602540378;
        const iy = rIn * 0.5;
        const i0 = vec2(0.0, rIn);
        const i1 = vec2(ix, iy);
        const i2 = vec2(ix, -iy);
        const i3 = vec2(0.0, -rIn);
        const i4 = vec2(-ix, -iy);
        const i5 = vec2(-ix, iy);

        let dSec = sdSegment2DTsl(uv, i0, i1);
        dSec = min(dSec, sdSegment2DTsl(uv, i1, i2));
        dSec = min(dSec, sdSegment2DTsl(uv, i2, i3));
        dSec = min(dSec, sdSegment2DTsl(uv, i3, i4));
        dSec = min(dSec, sdSegment2DTsl(uv, i4, i5));
        dSec = min(dSec, sdSegment2DTsl(uv, i5, i0));
        secMask.assign(stroke(dSec, 0.013));
      })
      .Else(() => {
        // Stencil 3: Architectural Nested Squares & Surveyor Meridian (Cinnabar + Azurite)
        const dSqOuter = abs(max(au, av).sub(0.40));
        const dSqInner = abs(max(au, av).sub(0.19));
        const dSqRot = abs(au.add(av).mul(0.70710678).sub(0.30));
        const dPri = min(dSqOuter, min(dSqInner, dSqRot));
        priMask.assign(stroke(dPri, 0.018));

        const absP = vec2(au, av);
        const dMerX = sdSegment2DTsl(absP, vec2(0.44, 0.0), vec2(0.54, 0.0));
        const dMerY = sdSegment2DTsl(absP, vec2(0.0, 0.44), vec2(0.0, 0.54));
        const dCtrX = sdSegment2DTsl(absP, vec2(0.0, 0.0), vec2(0.11, 0.0));
        const dCtrY = sdSegment2DTsl(absP, vec2(0.0, 0.0), vec2(0.0, 0.11));
        const dSec = min(min(dMerX, dMerY), min(dCtrX, dCtrY));
        secMask.assign(stroke(dSec, 0.012));
      });

    const cleanSec = secMask.mul(float(1.0).sub(priMask));
    return vec2(priMask, cleanSec);
  });

  // =========================================================================
  // 1. Unified Brutalist Architectural Concrete / Travertine + Varini Paint Material
  // =========================================================================
  const courtyardMaterial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.78,
    envMapIntensity: 0.58,
  });

  const aMatClass = attribute('aMatClass', 'float');

  // Shared per-fragment projective evaluation (evaluated cleanly in colorNode & roughnessNode)
  const evalCourtyardSurface = Fn(() => {
    const wp = positionWorld;
    const lp = positionLocal;
    const gn = normalize(normalWorldGeometry);

    // 1. Projective clip-space angular coordinates from uVantagePos (uniform-control-flow fwidth!)
    const rel = wp.sub(uVantagePos);
    const zCam = dot(rel, uVantageForward);
    const xCam = dot(rel, uVantageRight);
    const yCam = dot(rel, uVantageUp);
    const inFront = step(float(0.05), zCam);
    const safeDenom = max(zCam, float(0.15)).mul(uVantageScale);
    const projUv = vec2(xCam, yCam).div(safeDenom);

    // Evaluate fwidth unconditionally at top level before any branches (WGSL uniform control flow!)
    const uvFw = max(fwidth(projUv.x), fwidth(projUv.y));
    const aaWidth = clamp(uvFw.mul(0.82), float(0.0012), float(0.016));

    // 2. Single-pass 30-slab exact line-of-sight occlusion + analytical architectural crevice AO
    const occAndAO = evalOcclusionAndCreviceAO(wp, gn);
    const visibility = occAndAO.x.mul(inFront);
    const tightAO = occAndAO.y;
    const broadAO = occAndAO.z;

    // 3. Evaluate Felice Varini stencil SDF & radial optical projection scan-flash reveal
    const stencilMasks = evalStencilTsl(projUv, aaWidth);
    const rUv = length(projUv);
    const frontR = uFlashProgress.mul(0.84);
    const isFlashing = float(1.0).sub(smoothstep(float(0.96), float(1.0), uFlashProgress));
    const revealWave = mix(
      float(1.0),
      float(1.0).sub(smoothstep(frontR.sub(0.04), frontR.add(0.04), rUv)),
      isFlashing
    );
    const flashBand = exp(rUv.sub(frontR).mul(rUv.sub(frontR)).mul(-420.0))
      .mul(isFlashing)
      .mul(visibility);

    const priPaint = stencilMasks.x.mul(visibility).mul(revealWave);
    const secPaint = stencilMasks.y.mul(visibility).mul(revealWave);
    const totalPaint = clamp(priPaint.add(secPaint), 0.0, 1.0);

    // 4. 3D Object-Space Brutalist Architectural Concrete & Honed Travertine Petrography
    const macroFbm = mineralToothFbm(lp, float(2.8));
    const fineGrain = mineralToothFbm(lp, float(18.5));
    const poreFbm = mineralToothFbm(lp, float(46.0));
    const travertineStrata = mineralToothFbm(lp.mul(vec3(1.6, 11.5, 1.6)), float(2.2));

    // Subtle Tadao Ando horizontal board-formed concrete pour lift seams & tie-rod dimples on vertical walls
    const isVertical = float(1.0).sub(smoothstep(float(0.55), float(0.85), abs(gn.y)));
    const liftPhase = fract(lp.y.mul(4.5).add(10.0));
    const liftSeam = float(1.0)
      .sub(smoothstep(float(0.0), float(0.032), liftPhase))
      .add(smoothstep(float(0.968), float(1.0), liftPhase))
      .mul(isVertical);

    // Subtle tie-rod dimple grid on vertical concrete walls (0.44 x 0.44 rhythm)
    const wallCoordU = select(abs(gn.x).greaterThan(abs(gn.z)), lp.z, lp.x);
    const tieU = abs(fract(wallCoordU.mul(2.25).add(10.5)).sub(0.5));
    const tieV = abs(fract(lp.y.mul(2.25).add(10.5)).sub(0.5));
    const tieDist = length(vec2(tieU, tieV));
    const tieDimple = float(1.0).sub(smoothstep(float(0.018), float(0.036), tieDist)).mul(isVertical);

    // Material Class weights: 0 = Board-Formed Concrete, 1 = Honed Travertine, 2 = Dark Umber Stone
    const isTravertine = smoothstep(float(0.5), float(1.2), aMatClass).mul(
      float(1.0).sub(smoothstep(float(1.5), float(1.8), aMatClass))
    );
    const isDarkStone = smoothstep(float(1.5), float(1.9), aMatClass);

    // Class 0: Pale warm-neutral architectural board-formed concrete
    const concreteWarm = mix(boneCalciteRgb.mul(0.76), warmKaolinRgb.mul(0.68), macroFbm);
    const concreteCoolTint = mix(concreteWarm, vec3(0.52, 0.51, 0.48), float(0.24));
    const concreteColor = concreteCoolTint
      .mul(mix(float(0.92), float(1.05), fineGrain))
      .mul(float(1.0).sub(liftSeam.mul(0.11)))
      .mul(float(1.0).sub(tieDimple.mul(0.22)));

    // Class 1: Warm Roman honed travertine (cantilevered stairs & courtyard dais)
    const travBase = mix(warmKaolinRgb.mul(0.70), boneCalciteRgb.mul(0.74), travertineStrata);
    const travVein = mix(travBase, yellowOchreRgb.mul(0.55), smoothstep(float(0.66), float(0.88), travertineStrata).mul(0.16));
    const travColor = travVein.mul(mix(float(0.93), float(1.04), fineGrain));

    // Class 2: Dark honed Raw-Umber sub-plinth & recessed staircase spine
    const darkStoneColor = mix(rawUmberRgb.mul(0.72), rawUmberRgb.mul(1.12), macroFbm)
      .mul(mix(float(0.90), float(1.08), fineGrain));

    const substrateRaw = mix(mix(concreteColor, travColor, isTravertine), darkStoneColor, isDarkStone);

    // Crisp architectural ambient occlusion (tight 90-deg contact seams + broad courtyard pockets)
    const aoFactor = clamp(
      float(1.0).sub(tightAO.mul(0.42)).sub(broadAO.mul(0.24)),
      0.34,
      1.0
    );

    // Subtle micro-bevel highlight catch on 45-deg stone chamfers (|gn.x|+|gn.y|+|gn.z| > 1.25)
    const bevelCatch = smoothstep(
      float(1.16),
      float(1.36),
      abs(gn.x).add(abs(gn.y)).add(abs(gn.z))
    );
    const substrateLit = substrateRaw
      .mul(aoFactor)
      .mul(float(1.0).add(bevelCatch.mul(0.08)));

    // 5. True 2-Flux Kubelka-Munk Matte Mineral Pigment Glaze on Projected Stencil Bands
    const poreMod = mix(float(0.82), float(1.18), poreFbm);
    const priThick = priPaint.mul(2.15).mul(poreMod);
    const secThick = secPaint.mul(2.05).mul(poreMod);

    const afterPri = kubelkaMunkGlaze(uPriK, uPriS, priThick, substrateLit);
    const afterSec = kubelkaMunkGlaze(uSecK, uSecS, secThick, afterPri);

    // Subtle pointer hover grazing warmth on bare concrete (understated tactile affordance)
    const ptrDist = length(wp.sub(uPointerWorld));
    const ptrWarmth = exp(ptrDist.mul(ptrDist).mul(-3.2))
      .mul(uPointerActive)
      .mul(float(1.0).sub(totalPaint))
      .mul(0.045);

    // Brief warm optical projector scan-flash strictly on unoccluded surfaces + micro lock polish
    const flashTint = mix(boneCalciteRgb, realgarAmberRgb, float(0.32))
      .mul(flashBand.mul(0.55));
    const lockHighlight = totalPaint.mul(uLockGlow).mul(0.045);

    const finalRgb = afterSec
      .add(boneCalciteRgb.mul(ptrWarmth))
      .add(flashTint)
      .mul(float(1.0).add(lockHighlight));

    // Matte mineral fresco paint has chalkier roughness (~0.90) than honed travertine/concrete (~0.64-0.78)
    const baseRough = mix(
      mix(float(0.76), float(0.64), isTravertine),
      float(0.80),
      isDarkStone
    ).add(fineGrain.sub(0.5).mul(0.12));
    const finalRough = clamp(mix(baseRough, float(0.90), totalPaint), 0.48, 0.96);

    return vec4(finalRgb, finalRough);
  });

  const surfaceEval = evalCourtyardSurface();
  courtyardMaterial.colorNode = surfaceEval.xyz;
  courtyardMaterial.roughnessNode = surfaceEval.w;

  // Fine mineral concrete & travertine micro-normal perturbation (zero cloudy blotches)
  courtyardMaterial.normalNode = perturbMineralNormal(
    normalLocal,
    positionLocal,
    vec3(22.0, 28.0, 22.0),
    float(0.022)
  );

  // =========================================================================
  // 2. Machined Satin-Brass Optical Surveyor's Sight-Ring Materials
  // =========================================================================
  const brassBezelMaterial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.28,
    envMapIntensity: 1.24,
    transparent: true,
    depthWrite: true,
  });
  {
    const radialGrain = mineralToothFbm(
      positionLocal.mul(vec3(34.0, 34.0, 12.0)),
      float(1.8)
    );
    const satinBrass = mix(yellowOchreRgb.mul(0.92), boneCalciteRgb.mul(0.88), float(0.34));
    const hoveredBrass = mix(satinBrass, realgarAmberRgb.mul(0.95).add(boneCalciteRgb.mul(0.32)), uRingHover.mul(0.35));
    brassBezelMaterial.colorNode = hoveredBrass.mul(mix(float(0.92), float(1.06), radialGrain));
    brassBezelMaterial.roughnessNode = clamp(
      float(0.28).add(radialGrain.sub(0.5).mul(0.10)).sub(uRingHover.mul(0.07)),
      0.16,
      0.44
    );
    brassBezelMaterial.opacityNode = uRingAlpha;
    brassBezelMaterial.normalNode = perturbMineralNormal(
      normalLocal,
      positionLocal,
      vec3(40.0, 40.0, 14.0),
      float(0.014)
    );
  }

  const brassTicksMaterial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.84,
    roughness: 0.34,
    envMapIntensity: 1.18,
    transparent: true,
    depthWrite: true,
  });
  {
    const darkBrass = mix(yellowOchreRgb.mul(0.76), rawUmberRgb.mul(0.65), float(0.22));
    const hoveredTicks = mix(darkBrass, boneCalciteRgb.mul(0.92), uRingHover.mul(0.40));
    brassTicksMaterial.colorNode = hoveredTicks;
    brassTicksMaterial.opacityNode = uRingAlpha;
  }

  const syncVantageBasis = (basis: VantageBasis): void => {
    uVantagePos.value.set(basis.pos[0], basis.pos[1], basis.pos[2]);
    uVantageForward.value.set(basis.forward[0], basis.forward[1], basis.forward[2]);
    uVantageRight.value.set(basis.right[0], basis.right[1], basis.right[2]);
    uVantageUp.value.set(basis.up[0], basis.up[1], basis.up[2]);
    uVantageScale.value = basis.tanHalfFov * basis.stencilScale;
  };

  const syncStencil = (stencilIndex: number): void => {
    const idx =
      ((Math.round(stencilIndex) % STENCIL_CONFIGS.length) + STENCIL_CONFIGS.length) %
      STENCIL_CONFIGS.length;
    uActiveStencil.value = idx;
    const cfg = STENCIL_CONFIGS[idx];
    const pri = MINERAL_PALETTE[cfg.primaryPigment];
    const sec = MINERAL_PALETTE[cfg.secondaryPigment];
    uPriK.value.set(pri.absorptionK[0], pri.absorptionK[1], pri.absorptionK[2]);
    uPriS.value.set(pri.scatteringS[0], pri.scatteringS[1], pri.scatteringS[2]);
    uSecK.value.set(sec.absorptionK[0], sec.absorptionK[1], sec.absorptionK[2]);
    uSecS.value.set(sec.scatteringS[0], sec.scatteringS[1], sec.scatteringS[2]);
  };

  return {
    courtyardMaterial,
    brassBezelMaterial,
    brassTicksMaterial,
    syncVantageBasis,
    syncStencil,
    uFlashProgress,
    uLockGlow,
    uRingAlpha,
    uRingHover,
    uPointerWorld,
    uPointerActive,
    dispose: () => {
      courtyardMaterial.dispose();
      brassBezelMaterial.dispose();
      brassTicksMaterial.dispose();
    },
  };
}
