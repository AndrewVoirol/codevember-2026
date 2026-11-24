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
  floor,
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
  sin,
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
  hash31Tsl,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import {
  ARCHITECTURAL_SLABS,
  OCCLUSION_FRONT_DOT_MIN,
  OCCLUSION_MIN_T,
  OCCLUSION_OTHER_BIAS,
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
 * Analytical 4-tap tetrahedral 3D FBM micro-normal gradient in object space.
 * Returns the unprojected 3D height gradient scaled by anisotropy so callers can
 * combine multiple architectural relief gradients before tangent-plane projection.
 */
const fbmGradient3D = Fn(([pos_in, scaleVec_in, eps_in]: [TslNode, TslNode, TslNode]) => {
  const sv = vec3(scaleVec_in);
  const p = vec3(pos_in).mul(sv);
  const eps = float(eps_in);

  const k1 = vec3(1.0, -1.0, -1.0);
  const k2 = vec3(-1.0, -1.0, 1.0);
  const k3 = vec3(-1.0, 1.0, -1.0);
  const k4 = vec3(1.0, 1.0, 1.0);

  const f1 = mineralToothFbm(p.add(k1.mul(eps)), float(1.0));
  const f2 = mineralToothFbm(p.add(k2.mul(eps)), float(1.0));
  const f3 = mineralToothFbm(p.add(k3.mul(eps)), float(1.0));
  const f4 = mineralToothFbm(p.add(k4.mul(eps)), float(1.0));

  const maxScale = max(sv.x, max(sv.y, max(sv.z, float(1.0))));
  const anisoWeight = sv.div(maxScale);
  return k1
    .mul(f1)
    .add(k2.mul(f2))
    .add(k3.mul(f3))
    .add(k4.mul(f4))
    .div(eps.mul(4.0))
    .mul(anisoWeight);
});

/**
 * Convenience brass micro-normal helper for the satin-brass sight-ring.
 */
const perturbBrassNormal = Fn(
  ([baseNorm_in, pos_in, scaleVec_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const grad = fbmGradient3D(pos_in, scaleVec_in, float(0.08));
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
   * Single-pass 30-slab analytical ray-box occlusion (from vantagePos -> wp)
   * with exact per-slab exclusion (`aSlabId`) so contact seams tighten to 2 mm
   * without bevel self-shadowing, plus 2-radius analytical architectural crevice AO.
   * Returns vec3(visibility01, tightCreviceAO, broadCornerAO).
   * Pure ALU, zero derivatives inside loop — 100% uniform control flow.
   */
  const evalOcclusionAndCreviceAO = Fn(
    ([wp_in, gn_in, slabId_in, pxWorld_in]: [TslNode, TslNode, TslNode, TslNode]) => {
      const wp = vec3(wp_in);
      const gn = normalize(vec3(gn_in));
      const slabId = float(slabId_in);
      const pxWorld = float(pxWorld_in);

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

      // Tight 2 mm clearance against all OTHER slabs (self-slab excluded cleanly by aSlabId!)
      const maxT = dist.sub(float(OCCLUSION_OTHER_BIAS));
      // Pixel-footprint-scaled occlusion edge AA stays razor-crisp on macro zoom without grazing jaggies
      const chordEdgeHi = clamp(pxWorld.mul(1.65), float(0.0014), float(0.0048));

      // Two normal-offset sample probes for crisp analytical architectural crevice & soffit AO
      const aoProbeTight = wp.add(gn.mul(0.048));
      const aoProbeBroad = wp.add(gn.mul(0.20));

      const occSum = float(0.0).toVar();
      const tightAO = float(0.0).toVar();
      const broadAO = float(0.0).toVar();

      Loop(SLAB_COUNT, ({ i }) => {
        const bMin = vec4(uBoxMin.element(i)).xyz;
        const bMax = vec4(uBoxMax.element(i)).xyz;
        const isOtherSlab = step(float(0.5), abs(float(i).sub(slabId)));

        // 1. Exact slab-method ray-AABB intersection from uVantagePos -> wp
        const t1 = bMin.sub(uVantagePos).mul(invD);
        const t2 = bMax.sub(uVantagePos).mul(invD);
        const tMinV = min(t1, t2);
        const tMaxV = max(t1, t2);
        const tEnter = max(tMinV.x, max(tMinV.y, tMinV.z));
        const tExit = min(tMaxV.x, min(tMaxV.y, tMaxV.z));
        const clampedEnter = max(tEnter, float(OCCLUSION_MIN_T));

        const chord = tExit.sub(clampedEnter);
        const hitBeforeSurface = step(tEnter, maxT);
        const hitStrength = smoothstep(float(0.00035), chordEdgeHi, chord)
          .mul(hitBeforeSurface)
          .mul(isOtherSlab);
        occSum.assign(max(occSum, hitStrength));

        // 2. Analytical architectural crevice AO strictly from neighboring slabs (0 self-AO on bevels)
        const dTight = length(max(max(bMin.sub(aoProbeTight), aoProbeTight.sub(bMax)), vec3(0.0)));
        const dBroad = length(max(max(bMin.sub(aoProbeBroad), aoProbeBroad.sub(bMax)), vec3(0.0)));

        const cTight = float(1.0).sub(smoothstep(float(0.0), float(0.048), dTight)).mul(isOtherSlab);
        const cBroad = float(1.0).sub(smoothstep(float(0.0), float(0.185), dBroad)).mul(isOtherSlab);
        tightAO.assign(max(tightAO, cTight));
        broadAO.assign(max(broadAO, cBroad));
      });

      const visibility = clamp(frontMask.mul(float(1.0).sub(occSum)), 0.0, 1.0);
      return vec3(visibility, clamp(tightAO, 0.0, 1.0), clamp(broadAO, 0.0, 1.0));
    }
  );

  /**
   * Evaluates the active Felice Varini geometric stencil SDF in isotropic (u, v) space
   * with precomputed uniform-control-flow `aaWidth` and optional sub-millimeter
   * close-up mineral capillary tooth perturbation (`edgeJitter`).
   * Returns vec2(primaryMask, secondaryMask).
   */
  const evalStencilTsl = Fn(([uv_in, aa_in, jitter_in]: [TslNode, TslNode, TslNode]) => {
    const uv = vec2(uv_in);
    const aa = max(float(0.0007), float(aa_in));
    const jitter = float(jitter_in);
    const u = uv.x;
    const v = uv.y;
    const au = abs(u);
    const av = abs(v);

    const priMask = float(0.0).toVar();
    const secMask = float(0.0).toVar();

    const stroke = (dist: TslNode, halfW: number) =>
      float(1.0).sub(smoothstep(float(halfW).sub(aa), float(halfW).add(aa), dist.add(jitter)));

    If(uActiveStencil.lessThan(0.5), () => {
      // Stencil 0: Concentric Rings (Cinnabar) & Rotated Diamond + Axis Ticks (Yellow Ochre)
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
        // Stencil 1: Classic Truncated-Corner Impossible Penrose Tribar (Yellow Ochre + Azurite)
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
        secMask.assign(stroke(dSec, 0.014));
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
    roughness: 0.72,
    envMapIntensity: 0.62,
  });

  const aMatClass = attribute('aMatClass', 'float');
  const aSlabId = attribute('aSlabId', 'float');
  const aSlabLocal = attribute('aSlabLocal', 'vec3');
  const aSlabSize = attribute('aSlabSize', 'vec3');
  const aKindFlags = attribute('aKindFlags', 'vec2');

  /**
   * Pure analytical architectural relief evaluator (zero texture taps, pure ALU):
   * computes exact 3D height gradients (xyz) + physical cavity ambient occlusion (w) for:
   *  - Board-formed concrete cedar-shuttering V-grooves (vertical faces)
   *  - Slab-margin-gated Tadao Ando recessed conical tie-rod p-cones (never clipped at corners!)
   *  - Carved 3-flute anti-slip stair nosing grooves + underside drip groove + spine socket reveal
   *  - Honed Roman travertine courtyard ashlar paving joints (running bond)
   */
  const evalArchitecturalMasonryRelief = Fn(
    ([
      lp_in,
      gn_in,
      slabLocal_in,
      slabSize_in,
      kindFlags_in,
      isConcrete_in,
      isTravertine_in,
      pxWorld_in,
    ]: [TslNode, TslNode, TslNode, TslNode, TslNode, TslNode, TslNode, TslNode]) => {
      const lp = vec3(lp_in);
      const gn = normalize(vec3(gn_in));
      const sLoc = vec3(slabLocal_in);
      const sSize = vec3(slabSize_in);
      const isStairTread = vec2(kindFlags_in).x;
      const isTerrace = vec2(kindFlags_in).y;
      const isConcrete = float(isConcrete_in);
      const isTravertine = float(isTravertine_in);
      const pxW = float(pxWorld_in);

      const isVertical = float(1.0).sub(smoothstep(float(0.55), float(0.85), abs(gn.y)));
      const isTopFace = smoothstep(float(0.72), float(0.92), gn.y);
      const isBottomFace = smoothstep(float(0.72), float(0.92), gn.y.negate());

      const grad = vec3(0.0, 0.0, 0.0).toVar();
      const cavityAO = float(1.0).toVar();

      // ---------------------------------------------------------------------
      // A. Board-Formed Concrete Horizontal Shuttering Plank V-Grooves
      // ---------------------------------------------------------------------
      const plankH = float(0.16);
      const plankPhase = fract(lp.y.div(plankH).add(100.0));
      // Signed offset from nearest seam in [-plankH/2, +plankH/2]
      const seamSigned = select(
        plankPhase.greaterThan(0.5),
        plankPhase.sub(1.0),
        plankPhase
      ).mul(plankH);
      const seamAbs = abs(seamSigned);
      const seamHalfW = max(float(0.0017), pxW.mul(0.95));
      const inSeam = float(1.0).sub(smoothstep(seamHalfW.mul(0.25), seamHalfW.mul(1.25), seamAbs));
      // Smooth V-slope normal: below seam tilts +Y (upward catch), above seam tilts -Y (shadow lip)
      const seamSlopeY = clamp(seamSigned.div(max(float(0.001), seamHalfW)), -1.0, 1.0)
        .mul(inSeam)
        .mul(isVertical)
        .mul(isConcrete);
      grad.addAssign(vec3(0.0, seamSlopeY.mul(0.68), 0.0));
      cavityAO.mulAssign(float(1.0).sub(inSeam.mul(isVertical).mul(isConcrete).mul(0.28)));

      // ---------------------------------------------------------------------
      // B. Margin-Gated Recessed Conical Tie-Rod Holes (Tadao Ando P-Cones)
      // ---------------------------------------------------------------------
      const isXFace = abs(gn.x).greaterThan(abs(gn.z));
      const wallSizeU = select(isXFace, sSize.z, sSize.x);
      const wallLocalU = select(isXFace, sLoc.z, sLoc.x);
      const wallDirU = select(isXFace, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0));

      // Strict slab-size & edge-margin gating so p-cones NEVER slice across thin beams/piers/corners
      const marginU = min(wallLocalU, float(1.0).sub(wallLocalU)).mul(wallSizeU);
      const marginY = min(sLoc.y, float(1.0).sub(sLoc.y)).mul(sSize.y);
      const allowCone = isVertical
        .mul(isConcrete)
        .mul(step(float(0.30), wallSizeU))
        .mul(step(float(0.46), sSize.y))
        .mul(smoothstep(float(0.068), float(0.088), marginU))
        .mul(smoothstep(float(0.068), float(0.088), marginY));

      // Symmetric horizontal pitch centered on slab, vertically centered inside every 2nd plank
      const uPitch = float(0.34);
      const yPitch = float(0.32);
      const uCentered = wallLocalU.sub(0.5).mul(wallSizeU);
      const du = fract(uCentered.div(uPitch).add(100.5)).sub(0.5).mul(uPitch);
      const dy = fract(lp.y.sub(0.08).div(yPitch).add(100.5)).sub(0.5).mul(yPitch);
      const rCone = length(vec2(du, dy));
      const coneRad = float(0.0155);
      const coneAa = max(float(0.0014), pxW.mul(0.95));

      // Inside cone r < coneRad: concave radial bevel (grad points outward -> normal points inward!)
      const coneInside = float(1.0).sub(smoothstep(coneRad.sub(coneAa), coneRad.add(coneAa), rCone));
      const coneRimRing = smoothstep(coneRad.mul(0.32), coneRad.mul(0.58), rCone).mul(coneInside);
      const safeR = max(float(0.0015), rCone);
      const coneRadialDir = wallDirU.mul(du.div(safeR)).add(vec3(0.0, dy.div(safeR), 0.0));
      grad.addAssign(coneRadialDir.mul(coneRimRing).mul(allowCone).mul(0.92));

      // Crisp cavity shadow inside recessed cone + faint outer mortar collar
      const coneDepthShadow = float(1.0)
        .sub(smoothstep(float(0.003), coneRad, rCone))
        .mul(0.38)
        .add(coneInside.mul(0.24));
      cavityAO.mulAssign(float(1.0).sub(coneDepthShadow.mul(allowCone)));

      // ---------------------------------------------------------------------
      // C. Cantilevered Travertine Stair Treads: 3-Flute Honed Nosing & Root Reveal
      // ---------------------------------------------------------------------
      const distFromNose = float(1.0).sub(sLoc.z).mul(sSize.z);
      const distFromRoot = sLoc.x.mul(sSize.x);
      const distFromFreeEnd = float(1.0).sub(sLoc.x).mul(sSize.x);
      const insideStairSpan = smoothstep(float(0.026), float(0.036), distFromRoot).mul(
        smoothstep(float(0.022), float(0.032), distFromFreeEnd)
      );

      // Three crisp parallel honed anti-slip flutes along the front top edge of each step
      const f1 = distFromNose.sub(0.022);
      const f2 = distFromNose.sub(0.034);
      const f3 = distFromNose.sub(0.046);
      const fluteSigned = select(
        abs(f1).lessThan(abs(f2)),
        select(abs(f1).lessThan(abs(f3)), f1, f3),
        select(abs(f2).lessThan(abs(f3)), f2, f3)
      );
      const fluteHalfW = max(float(0.0022), pxW.mul(1.05));
      const inFlute = float(1.0)
        .sub(smoothstep(fluteHalfW.mul(0.3), fluteHalfW.mul(1.25), abs(fluteSigned)))
        .mul(isTopFace)
        .mul(isStairTread)
        .mul(insideStairSpan);
      // Since distFromNose = maxZ - z, d(distFromNose)/dz = -1 -> negate sign for world +Z gradient
      const fluteSlopeZ = clamp(fluteSigned.div(max(float(0.0012), fluteHalfW)), -1.0, 1.0)
        .negate()
        .mul(inFlute);
      grad.addAssign(vec3(0.0, 0.0, fluteSlopeZ.mul(0.62)));
      cavityAO.mulAssign(float(1.0).sub(inFlute.mul(0.30)));

      // Subtle architectural drip-groove on the front underside soffit of each cantilevered step
      const dripSigned = distFromNose.sub(0.017);
      const inDrip = float(1.0)
        .sub(smoothstep(fluteHalfW.mul(0.3), fluteHalfW.mul(1.25), abs(dripSigned)))
        .mul(isBottomFace)
        .mul(isStairTread)
        .mul(insideStairSpan);
      const dripSlopeZ = clamp(dripSigned.div(max(float(0.0012), fluteHalfW)), -1.0, 1.0)
        .negate()
        .mul(inDrip);
      grad.addAssign(vec3(0.0, 0.0, dripSlopeZ.mul(0.45)));
      cavityAO.mulAssign(float(1.0).sub(inDrip.mul(0.28)));

      // Cantilever root shadow-gap reveal collar where each travertine tread sockets into dark stair-spine
      const rootGapMask = float(1.0)
        .sub(smoothstep(float(0.006), float(0.013), distFromRoot))
        .mul(isStairTread);
      const rootEdgeSlope = smoothstep(float(0.005), float(0.010), distFromRoot)
        .mul(float(1.0).sub(smoothstep(float(0.010), float(0.015), distFromRoot)))
        .mul(isStairTread);
      grad.addAssign(vec3(rootEdgeSlope.mul(0.55), 0.0, 0.0));
      cavityAO.mulAssign(float(1.0).sub(rootGapMask.mul(0.42)));

      // ---------------------------------------------------------------------
      // D. Honed Roman Travertine Courtyard Dais & Landing Ashlar Paving Joints
      // ---------------------------------------------------------------------
      const tileW = float(0.38);
      const tileD = float(0.32);
      const rowIdx = floor(lp.z.div(tileD).add(100.0));
      const rowOffset = fract(rowIdx.mul(0.5)).mul(tileW);
      const pTileX = fract(lp.x.add(rowOffset).div(tileW).add(100.0));
      const pTileZ = fract(lp.z.div(tileD).add(100.0));
      const dJointX = select(pTileX.greaterThan(0.5), pTileX.sub(1.0), pTileX).mul(tileW);
      const dJointZ = select(pTileZ.greaterThan(0.5), pTileZ.sub(1.0), pTileZ).mul(tileD);
      const jointHalfW = max(float(0.0022), pxW.mul(1.0));
      const inJointX = float(1.0).sub(smoothstep(jointHalfW.mul(0.35), jointHalfW.mul(1.3), abs(dJointX)));
      const inJointZ = float(1.0).sub(smoothstep(jointHalfW.mul(0.35), jointHalfW.mul(1.3), abs(dJointZ)));
      const pavingGate = isTopFace.mul(isTravertine).mul(isTerrace);

      grad.addAssign(
        vec3(
          clamp(dJointX.div(max(float(0.0012), jointHalfW)), -1.0, 1.0).mul(inJointX).mul(pavingGate).mul(0.44),
          0.0,
          clamp(dJointZ.div(max(float(0.0012), jointHalfW)), -1.0, 1.0).mul(inJointZ).mul(pavingGate).mul(0.44)
        )
      );
      cavityAO.mulAssign(float(1.0).sub(max(inJointX, inJointZ).mul(pavingGate).mul(0.22)));

      return vec4(grad, clamp(cavityAO, 0.34, 1.0));
    }
  );

  // =========================================================================
  // Shared Fragment Evaluation (single TSL graph shared across color, roughness, normal!)
  // =========================================================================
  const evalCourtyardGraph = () => {
    const wp = positionWorld;
    const lp = positionLocal;
    const gn = normalize(normalWorldGeometry);

    // 1. Projective clip-space angular coordinates from uVantagePos
    const rel = wp.sub(uVantagePos);
    const zCam = dot(rel, uVantageForward);
    const xCam = dot(rel, uVantageRight);
    const yCam = dot(rel, uVantageUp);
    const inFront = step(float(0.05), zCam);
    const safeDenom = max(zCam, float(0.15)).mul(uVantageScale);
    const projUv = vec2(xCam, yCam).div(safeDenom);

    // Evaluate ALL screen-space derivatives strictly at top-level uniform control flow!
    const uvFw = max(fwidth(projUv.x), fwidth(projUv.y));
    const aaWidth = clamp(uvFw.mul(0.75), float(0.0007), float(0.014));
    const pxWorld = clamp(
      fwidth(wp.x).add(fwidth(wp.y)).add(fwidth(wp.z)).mul(0.55),
      float(0.00035),
      float(0.015)
    );

    // 2. Material Class weights: 0 = Board-Formed Concrete, 1 = Honed Travertine, 2 = Dark Basalt/Stone
    const isTravertine = smoothstep(float(0.5), float(1.2), aMatClass).mul(
      float(1.0).sub(smoothstep(float(1.5), float(1.8), aMatClass))
    );
    const isDarkStone = smoothstep(float(1.5), float(1.9), aMatClass);
    const isConcrete = clamp(float(1.0).sub(isTravertine).sub(isDarkStone), 0.0, 1.0);
    const isVertical = float(1.0).sub(smoothstep(float(0.55), float(0.85), abs(gn.y)));

    // 3. Single-pass 30-slab per-slab-indexed line-of-sight occlusion + crevice AO
    const occAndAO = evalOcclusionAndCreviceAO(wp, gn, aSlabId, pxWorld);
    const visibility = occAndAO.x.mul(inFront);
    const tightAO = occAndAO.y;
    const broadAO = occAndAO.z;

    // 4. Object-Space Multi-Scale Mineral Petrography (Board Concrete, Travertine Vugs, Basalt)
    const macroFbm = mineralToothFbm(lp, float(2.8));
    const fineGrain = mineralToothFbm(lp, float(38.0));
    // High-frequency crisp millimeter cement/sand aggregate & mineral pinholes (stays sharp at 1:1 zoom!)
    const microGrit = mineralToothFbm(lp, float(165.0));
    const pinholeNoise = mineralToothFbm(lp.add(vec3(7.3, 2.1, 4.9)), float(135.0));

    // Cedar shuttering horizontal wood-grain impression on vertical board-formed concrete
    const isXFace = abs(gn.x).greaterThan(abs(gn.z));
    const boardCoord = select(
      isXFace,
      vec3(lp.x.mul(5.0), lp.y.mul(52.0), lp.z.mul(6.5)),
      vec3(lp.x.mul(6.5), lp.y.mul(52.0), lp.z.mul(5.0))
    );
    const cedarGrain = mineralToothFbm(boardCoord, float(1.0));
    const plankBandId = floor(lp.y.div(0.16).add(100.0));
    const plankToneRand = hash31Tsl(vec3(plankBandId, aSlabId.mul(3.17), float(1.37)));

    // Fine millimeter entrapped-air concrete bug-holes (never chunky blobs!) & quartz sand flecks
    const bugHole = smoothstep(float(0.79), float(0.87), pinholeNoise)
      .mul(smoothstep(float(0.58), float(0.76), microGrit))
      .mul(isConcrete);
    const aggregateFleck = smoothstep(float(0.66), float(0.84), microGrit);

    // Roman Travertine wavy sedimentary bedding & crisp fine elongated horizontal mineral vugs
    const strataWave = sin(lp.x.mul(4.5).add(lp.z.mul(3.9)).add(macroFbm.mul(4.2))).mul(0.055);
    const travBedding = mineralToothFbm(
      vec3(lp.x.mul(3.0), lp.y.add(strataWave).mul(20.0), lp.z.mul(3.0)),
      float(1.4)
    );
    const travFineBand = mineralToothFbm(
      vec3(lp.x.mul(4.0), lp.y.add(strataWave.mul(0.7)).mul(48.0), lp.z.mul(4.0)),
      float(1.25)
    );
    const vugRaw = mineralToothFbm(
      vec3(lp.x.mul(24.0), lp.y.mul(92.0), lp.z.mul(24.0)),
      float(1.1)
    );
    const travVug = smoothstep(float(0.68), float(0.79), vugRaw).mul(isTravertine);

    // Honed travertine courtyard paving subtle per-tile tonal variation
    const tileRow = floor(lp.z.div(0.32).add(100.0));
    const tileCol = floor(lp.x.add(fract(tileRow.mul(0.5)).mul(0.38)).div(0.38).add(100.0));
    const tileToneRand = hash31Tsl(vec3(tileCol, tileRow, aSlabId.mul(1.73)));
    const isTerraceTop = smoothstep(float(0.72), float(0.92), gn.y).mul(aKindFlags.y).mul(isTravertine);

    // 5. Analytical Architectural Masonry Relief (V-grooves, P-cones, Stair Flutes, Paving Joints)
    const archRelief = evalArchitecturalMasonryRelief(
      lp,
      gn,
      aSlabLocal,
      aSlabSize,
      aKindFlags,
      isConcrete,
      isTravertine,
      pxWorld
    );
    const archGrad = archRelief.xyz;
    const archCavityAO = archRelief.w;

    // 6. Un-Occluded Mineral Substrate Albedos (strictly from MINERAL_PALETTE)
    // Class 0: Board-formed pale architectural concrete with cedar shuttering & fine mineral grit
    const concreteBase = mix(boneCalciteRgb.mul(0.78), warmKaolinRgb.mul(0.69), macroFbm);
    const concreteNeutral = mix(concreteBase, vec3(0.53, 0.52, 0.49), float(0.24));
    const plankShift = mix(float(0.955), float(1.042), plankToneRand);
    const concreteGrainMix = mix(fineGrain, cedarGrain, isVertical.mul(0.45));
    const concreteColor = concreteNeutral
      .mul(mix(float(1.0), plankShift, isVertical))
      .mul(mix(float(0.95), float(1.04), concreteGrainMix))
      .mul(mix(float(0.95), float(1.05), microGrit))
      .mul(mix(float(1.0), float(1.045), aggregateFleck.mul(0.5)))
      .mul(float(1.0).sub(bugHole.mul(0.34)));

    // Class 1: Honed Roman travertine (cantilevered stairs, terrace dais, landing, front curb)
    const travWarm = mix(warmKaolinRgb.mul(0.71), boneCalciteRgb.mul(0.78), travBedding);
    const travVein = mix(
      travWarm,
      yellowOchreRgb.mul(0.58),
      smoothstep(float(0.58), float(0.84), travFineBand).mul(0.18)
    );
    const travAshlar = travVein.mul(mix(float(1.0), mix(float(0.955), float(1.045), tileToneRand), isTerraceTop));
    const travColor = mix(travAshlar, rawUmberRgb.mul(0.58), travVug.mul(0.36))
      .mul(mix(float(0.96), float(1.04), fineGrain))
      .mul(mix(float(0.97), float(1.03), microGrit));

    // Class 2: Dark honed Raw-Umber structural basalt/stone (stair spine & sub-plinth)
    const basaltSparkle = smoothstep(float(0.74), float(0.91), microGrit);
    const darkStoneColor = mix(rawUmberRgb.mul(0.68), rawUmberRgb.mul(1.06), macroFbm)
      .mul(mix(float(0.92), float(1.08), fineGrain))
      .mul(mix(float(0.95), float(1.05), microGrit))
      .add(boneCalciteRgb.mul(basaltSparkle.mul(0.024)));

    const substrateAlbedo = mix(
      mix(concreteColor, travColor, isTravertine),
      darkStoneColor,
      isDarkStone
    );

    // 7. Evaluate Felice Varini Projective Stencil with Sub-Millimeter Close-Up Capillary Tooth
    // Close-up zoom weight: 0 at sweet-spot orbit (5.35 -> razor-sharp Euclidean geometry),
    // smoothly rises to 1 at close macro zoom (<= 2.2) to reveal mineral pigment tooth on concrete/travertine!
    const macroWeight = float(1.0).sub(smoothstep(float(0.0014), float(0.0036), pxWorld));
    const stoneToothNoise = microGrit.sub(0.5).mul(0.68).add(fineGrain.sub(0.5).mul(0.32));
    // Convert ~0.9mm world-scale mineral tooth into stencil UV units safely clamped
    const uvPerWorld = clamp(uvFw.div(max(float(0.0005), pxWorld)), float(0.25), float(2.2));
    const edgeToothJitter = stoneToothNoise
      .mul(float(0.0012))
      .mul(uvPerWorld)
      .mul(macroWeight)
      .add(travVug.mul(0.0014).mul(macroWeight));

    const stencilMasks = evalStencilTsl(projUv, aaWidth, edgeToothJitter);

    // Radial optical projector scan-flash reveal on live re-projection
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

    // 8. True 2-Flux Kubelka-Munk Mineral Pigment Glaze over Un-Occluded Stone Albedo
    // Subtle micro-wear on crisp 45-deg stone chamfers & stair noses + pooling in stone pores/vugs
    const bevelCatch = smoothstep(
      float(1.16),
      float(1.36),
      abs(gn.x).add(abs(gn.y)).add(abs(gn.z))
    );
    const chamferThin = float(1.0).sub(bevelCatch.mul(mix(float(0.14), float(0.30), microGrit)));
    const porePool = mix(float(0.84), float(1.22), microGrit).add(travVug.mul(0.32)).add(bugHole.mul(0.28));
    const priThick = priPaint.mul(2.15).mul(porePool).mul(chamferThin);
    const secThick = secPaint.mul(2.05).mul(porePool).mul(chamferThin);

    const afterPri = kubelkaMunkGlaze(uPriK, uPriS, priThick, substrateAlbedo);
    const paintedAlbedo = kubelkaMunkGlaze(uSecK, uSecS, secThick, afterPri);

    // 9. Apply TRUE Physical Incident Ambient Occlusion AFTER Pigment Glaze!
    const structAO = clamp(
      float(1.0).sub(tightAO.mul(0.46)).sub(broadAO.mul(0.25)),
      0.30,
      1.0
    );
    const microCavityAO = float(1.0).sub(travVug.mul(0.26)).sub(bugHole.mul(0.28));
    const totalAO = clamp(structAO.mul(archCavityAO).mul(microCavityAO), 0.22, 1.0);

    // Subtle pointer hover warmth on bare stone + optical scan-flash + lock-in resonance
    const ptrDist = length(wp.sub(uPointerWorld));
    const ptrWarmth = exp(ptrDist.mul(ptrDist).mul(-3.2))
      .mul(uPointerActive)
      .mul(float(1.0).sub(totalPaint))
      .mul(0.042);

    const flashTint = mix(boneCalciteRgb, realgarAmberRgb, float(0.32)).mul(flashBand.mul(0.55));
    const lockHighlight = totalPaint.mul(uLockGlow).mul(0.045);

    const finalRgb = paintedAlbedo
      .mul(totalAO)
      .mul(float(1.0).add(bevelCatch.mul(0.075)))
      .add(boneCalciteRgb.mul(ptrWarmth))
      .add(flashTint)
      .mul(float(1.0).add(lockHighlight));

    // 10. Calibrated Roughness Contrast: Honed Travertine vs Cast Concrete vs Matte Mineral Paint
    const stairWear = aKindFlags.x
      .mul(smoothstep(float(0.72), float(0.92), gn.y))
      .mul(clamp(float(1.0).sub(abs(aSlabLocal.x.sub(0.52)).mul(1.8)), 0.0, 1.0));
    const baseRough = mix(
      mix(float(0.74), float(0.52).sub(stairWear.mul(0.07)), isTravertine),
      float(0.64),
      isDarkStone
    )
      .add(microGrit.sub(0.5).mul(0.14))
      .add(travVug.mul(0.36))
      .add(bugHole.mul(0.24));
    const finalRough = clamp(mix(baseRough, float(0.90), totalPaint), 0.40, 0.96);

    // 11. Physical Composite Normal: Architectural Masonry Relief + Crisp Dual-Scale Mineral Micro-Grain
    const concreteGrainScale = mix(
      vec3(54.0, 54.0, 54.0),
      select(isXFace, vec3(32.0, 74.0, 16.0), vec3(16.0, 74.0, 32.0)),
      isVertical
    );
    const grainScale = mix(
      mix(concreteGrainScale, vec3(28.0, 88.0, 28.0), isTravertine),
      vec3(68.0, 68.0, 68.0),
      isDarkStone
    );
    const rawMicroGrad = fbmGradient3D(lp, grainScale, float(0.075));
    // Crisp millimeter cement/stone grit micro-normal (fades gently when zoomed far out to prevent shimmer)
    const gritNormalWeight = float(1.0).sub(smoothstep(float(0.0022), float(0.0055), pxWorld));
    const rawGritGrad = fbmGradient3D(lp, vec3(175.0, 175.0, 175.0), float(0.085));

    const microStrength = mix(
      mix(float(0.115), float(0.16).add(travVug.mul(0.28)), isTravertine),
      float(0.10),
      isDarkStone
    ).mul(mix(float(1.0), float(0.80), totalPaint));
    const gritStrength = mix(float(0.135), float(0.085), isTravertine)
      .mul(gritNormalWeight)
      .mul(mix(float(1.0), float(0.76), totalPaint));

    // Combine analytical masonry relief (100% preserved under paint!) + dual-scale tactile stone grain
    const totalGrad = archGrad
      .add(rawMicroGrad.mul(microStrength))
      .add(rawGritGrad.mul(gritStrength));
    const nLoc = normalize(normalLocal);
    const tanGrad = totalGrad.sub(nLoc.mul(dot(totalGrad, nLoc)));
    const finalNormalView = transformNormalToView(normalize(nLoc.sub(tanGrad)));

    return {
      colorNode: finalRgb,
      roughnessNode: finalRough,
      normalNode: finalNormalView,
    };
  };

  const surfaceNodes = evalCourtyardGraph();
  courtyardMaterial.colorNode = surfaceNodes.colorNode;
  courtyardMaterial.roughnessNode = surfaceNodes.roughnessNode;
  courtyardMaterial.normalNode = surfaceNodes.normalNode;

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
    const hoveredBrass = mix(
      satinBrass,
      realgarAmberRgb.mul(0.95).add(boneCalciteRgb.mul(0.32)),
      uRingHover.mul(0.35)
    );
    brassBezelMaterial.colorNode = hoveredBrass.mul(mix(float(0.92), float(1.06), radialGrain));
    brassBezelMaterial.roughnessNode = clamp(
      float(0.28).add(radialGrain.sub(0.5).mul(0.10)).sub(uRingHover.mul(0.07)),
      0.16,
      0.44
    );
    brassBezelMaterial.opacityNode = uRingAlpha;
    brassBezelMaterial.normalNode = perturbBrassNormal(
      normalLocal,
      positionLocal,
      vec3(40.0, 40.0, 14.0),
      float(0.055)
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
