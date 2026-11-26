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
  exp2,
  float,
  floor,
  fract,
  fwidth,
  length,
  log2,
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
  texture,
  texture3DLevel,
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
import {
  DETAIL_TEX_SIZE,
  createArchitecturalPetrographyTextures,
} from './stonePetrography';

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
 * Irrational 3D lithic rotation helpers (matches Day 13 zero-grid-artifact petrography).
 */
const rotLithicA = Fn(([v_in]: [TslNode]) => {
  const v = vec3(v_in);
  return vec3(
    v.x.mul(0.809017).sub(v.z.mul(0.587785)),
    v.y.mul(0.866025).add(v.x.mul(0.293893)).add(v.z.mul(0.404508)),
    v.z.mul(0.699527).add(v.x.mul(0.509037)).sub(v.y.mul(0.5))
  );
});

const rotLithicB = Fn(([v_in]: [TslNode]) => {
  const v = vec3(v_in);
  return vec3(
    v.y.mul(0.766044).sub(v.x.mul(0.642788)),
    v.z.mul(0.819152).add(v.y.mul(0.368854)).add(v.x.mul(0.439558)),
    v.x.mul(0.628574).add(v.y.mul(0.526925)).sub(v.z.mul(0.573576))
  );
});

const rotLithicBTranspose = Fn(([g_in]: [TslNode]) => {
  const g = vec3(g_in);
  return vec3(
    g.x.mul(-0.642788).add(g.y.mul(0.439558)).add(g.z.mul(0.628574)),
    g.x.mul(0.766044).add(g.y.mul(0.368854)).add(g.z.mul(0.526925)),
    g.y.mul(0.819152).sub(g.z.mul(0.573576))
  );
});

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal gradient (used for satin brass ring).
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
  // Deterministic procedural petrography & architectural paint textures (zero network fetches)
  const petro = createArchitecturalPetrographyTextures();

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
  const uRingAlpha = uniform(0.0);
  const uRingHover = uniform(0.0);

  // Mineral substrate anchor colors (strictly from MINERAL_PALETTE, zero neons)
  const boneCalciteRgb = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const warmKaolinRgb = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const rawUmberRgb = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const yellowOchreRgb = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const realgarAmberRgb = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const azuriteRgb = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);

  /**
   * Single-pass 30-slab analytical ray-box occlusion (from vantagePos -> wp)
   * with exact per-slab exclusion (`aSlabId`) so contact seams tighten to 2 mm
   * without bevel self-shadowing, plus 2-radius analytical architectural crevice AO.
   * Returns vec3(visibility01, tightCreviceAO, broadCornerAO).
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

      // Tight 2 mm clearance against all OTHER slabs (self-slab excluded cleanly by aSlabId)
      const maxT = dist.sub(float(OCCLUSION_OTHER_BIAS));
      const chordEdgeHi = clamp(pxWorld.mul(1.65), float(0.0012), float(0.0046));

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
   * with precomputed uniform-control-flow `aaWidth` and physically scaled capillary
   * masonry tooth perturbation (`edgeJitter`).
   * Returns vec2(primaryMask, secondaryMask).
   */
  const evalStencilTsl = Fn(([uv_in, aa_in, jitter_in]: [TslNode, TslNode, TslNode]) => {
    const uv = vec2(uv_in);
    const aa = max(float(0.00055), float(aa_in));
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
  // 1. Unified Brutalist Concrete / Roman Travertine / Basalt + Physical Varini Paint
  // =========================================================================
  const courtyardMaterial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.74,
    envMapIntensity: 0.86,
  });

  const aMatClass = attribute('aMatClass', 'float');
  const aSlabId = attribute('aSlabId', 'float');
  const aSlabLocal = attribute('aSlabLocal', 'vec3');
  const aSlabSize = attribute('aSlabSize', 'vec3');
  const aKindFlags = attribute('aKindFlags', 'vec2');

  /**
   * Pure analytical architectural masonry relief evaluator:
   * returns vec4(gradX, gradY, gradZ, cavityAO) plus separate auxiliary signals via
   * a second compact helper for seam/mortar masks so paint physically reacts to joints & cones!
   *  - Board-formed horizontal shuttering V-grooves + subtle raised mortar laitance fin
   *  - Staggered vertical shuttering plank butt-joints (margin-gated)
   *  - Upgraded 3-part Tadao Ando conical tie-rod P-cones (outer chamfer + deep throat + recessed mortar plug)
   *  - 3-flute honed travertine stair nosing grooves + underside drip groove + spine socket reveal
   *  - Running-bond honed Roman travertine ashlar paving mortar joints
   */
  const evalArchitecturalMasonryRelief = Fn(
    ([
      lp_in,
      gn_in,
      slabLocal_in,
      slabSize_in,
      kindFlags_in,
      slabId_in,
      isConcrete_in,
      isTravertine_in,
      pxWorld_in,
    ]: [TslNode, TslNode, TslNode, TslNode, TslNode, TslNode, TslNode, TslNode, TslNode]) => {
      const lp = vec3(lp_in);
      const gn = normalize(vec3(gn_in));
      const sLoc = vec3(slabLocal_in);
      const sSize = vec3(slabSize_in);
      const isStairTread = vec2(kindFlags_in).x;
      const isTerrace = vec2(kindFlags_in).y;
      const slabId = float(slabId_in);
      const isConcrete = float(isConcrete_in);
      const isTravertine = float(isTravertine_in);
      const pxW = float(pxWorld_in);

      const isVertical = float(1.0).sub(smoothstep(float(0.55), float(0.85), abs(gn.y)));
      const isTopFace = smoothstep(float(0.72), float(0.92), gn.y);
      const isBottomFace = smoothstep(float(0.72), float(0.92), gn.y.negate());

      const grad = vec3(0.0, 0.0, 0.0).toVar();
      const cavityAO = float(1.0).toVar();

      // ---------------------------------------------------------------------
      // A. Board-Formed Concrete Horizontal Shuttering Plank V-Grooves & Step Lip
      // ---------------------------------------------------------------------
      const plankH = float(0.16);
      const plankPhase = fract(lp.y.div(plankH).add(100.0));
      const seamSigned = select(
        plankPhase.greaterThan(0.5),
        plankPhase.sub(1.0),
        plankPhase
      ).mul(plankH);
      const seamAbs = abs(seamSigned);
      const seamHalfW = max(float(0.0018), pxW.mul(0.92));
      const inSeam = float(1.0).sub(smoothstep(seamHalfW.mul(0.22), seamHalfW.mul(1.25), seamAbs));
      // Crisp V-groove slope + subtle squeezed cement fin right at shoulder
      const seamSlopeY = clamp(seamSigned.div(max(float(0.0009), seamHalfW)), -1.0, 1.0)
        .mul(inSeam)
        .mul(isVertical)
        .mul(isConcrete);
      const finShoulder = smoothstep(seamHalfW.mul(1.0), seamHalfW.mul(2.0), seamAbs)
        .mul(float(1.0).sub(smoothstep(seamHalfW.mul(2.0), seamHalfW.mul(3.4), seamAbs)))
        .mul(select(seamSigned.greaterThan(0.0), float(-1.0), float(1.0)))
        .mul(isVertical)
        .mul(isConcrete);
      grad.addAssign(vec3(0.0, seamSlopeY.mul(0.86).add(finShoulder.mul(0.16)), 0.0));
      cavityAO.mulAssign(float(1.0).sub(inSeam.mul(isVertical).mul(isConcrete).mul(0.38)));

      // Staggered vertical cedar plank butt-joints (subtle crisp joint every ~0.78 m)
      const isXFace = abs(gn.x).greaterThan(abs(gn.z));
      const wallSizeU = select(isXFace, sSize.z, sSize.x);
      const wallLocalU = select(isXFace, sLoc.z, sLoc.x);
      const wallCoordU = select(isXFace, lp.z, lp.x);
      const wallDirU = select(isXFace, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0));

      const marginU = min(wallLocalU, float(1.0).sub(wallLocalU)).mul(wallSizeU);
      const marginY = min(sLoc.y, float(1.0).sub(sLoc.y)).mul(sSize.y);
      const plankRowId = floor(lp.y.div(plankH).add(100.0));
      const rowButtShift = hash31Tsl(vec3(plankRowId, slabId.mul(2.71), float(4.19))).mul(0.76);
      const buttPitch = float(0.76);
      const buttPhase = fract(wallCoordU.add(rowButtShift).div(buttPitch).add(100.0));
      const buttSigned = select(buttPhase.greaterThan(0.5), buttPhase.sub(1.0), buttPhase).mul(buttPitch);
      const buttHalfW = max(float(0.0014), pxW.mul(0.85));
      const allowButt = isVertical
        .mul(isConcrete)
        .mul(step(float(0.32), wallSizeU))
        .mul(smoothstep(float(0.055), float(0.08), marginU));
      const inButt = float(1.0)
        .sub(smoothstep(buttHalfW.mul(0.25), buttHalfW.mul(1.25), abs(buttSigned)))
        .mul(allowButt);
      const buttSlope = clamp(buttSigned.div(max(float(0.0008), buttHalfW)), -1.0, 1.0).mul(inButt);
      grad.addAssign(wallDirU.mul(buttSlope.mul(0.55)));
      cavityAO.mulAssign(float(1.0).sub(inButt.mul(0.28)));

      // ---------------------------------------------------------------------
      // B. Upgraded 3-Part Tadao Ando Recessed Conical Tie-Rod P-Cones
      //    (Crisp outer chamfer + deep conical recess + inset grouted mortar plug)
      // ---------------------------------------------------------------------
      const allowCone = isVertical
        .mul(isConcrete)
        .mul(step(float(0.30), wallSizeU))
        .mul(step(float(0.46), sSize.y))
        .mul(smoothstep(float(0.068), float(0.088), marginU))
        .mul(smoothstep(float(0.068), float(0.088), marginY));

      const uPitch = float(0.34);
      const yPitch = float(0.32);
      const uCentered = wallLocalU.sub(0.5).mul(wallSizeU);
      const du = fract(uCentered.div(uPitch).add(100.5)).sub(0.5).mul(uPitch);
      const dy = fract(lp.y.sub(0.08).div(yPitch).add(100.5)).sub(0.5).mul(yPitch);
      const rCone = length(vec2(du, dy));
      const coneRad = float(0.0165);
      const plugRad = float(0.0082);
      const coneAa = max(float(0.0011), pxW.mul(0.85));

      const coneInside = float(1.0).sub(smoothstep(coneRad.sub(coneAa), coneRad.add(coneAa), rCone));
      const coneBevelRing = smoothstep(plugRad.sub(coneAa), plugRad.add(coneAa), rCone).mul(coneInside);
      // Hairline recessed ring around central grouted mortar plug
      const plugGroove = float(1.0)
        .sub(smoothstep(float(0.0008), float(0.0022), abs(rCone.sub(plugRad))))
        .mul(coneInside);
      const safeR = max(float(0.0012), rCone);
      const coneRadialDir = wallDirU.mul(du.div(safeR)).add(vec3(0.0, dy.div(safeR), 0.0));
      grad.addAssign(coneRadialDir.mul(coneBevelRing.mul(1.22).add(plugGroove.mul(0.45))).mul(allowCone));

      // Directional inner-cone darkening (upper lip shadowed more deeply than lower catch lip!)
      const upperConeShadow = clamp(dy.div(max(float(0.004), coneRad)).mul(0.5).add(0.5), 0.0, 1.0);
      const coneDepthShadow = coneBevelRing
        .mul(mix(float(0.24), float(0.62), upperConeShadow))
        .add(plugGroove.mul(0.42))
        .add(float(1.0).sub(smoothstep(plugRad.sub(coneAa), plugRad.add(coneAa), rCone)).mul(0.34));
      cavityAO.mulAssign(float(1.0).sub(clamp(coneDepthShadow, 0.0, 0.72).mul(allowCone)));

      // ---------------------------------------------------------------------
      // C. Cantilevered Travertine Stair Treads: 3-Flute Honed Nosing & Root Reveal
      // ---------------------------------------------------------------------
      const distFromNose = float(1.0).sub(sLoc.z).mul(sSize.z);
      const distFromRoot = sLoc.x.mul(sSize.x);
      const distFromFreeEnd = float(1.0).sub(sLoc.x).mul(sSize.x);
      const insideStairSpan = smoothstep(float(0.026), float(0.036), distFromRoot).mul(
        smoothstep(float(0.022), float(0.032), distFromFreeEnd)
      );

      const f1 = distFromNose.sub(0.022);
      const f2 = distFromNose.sub(0.034);
      const f3 = distFromNose.sub(0.046);
      const fluteSigned = select(
        abs(f1).lessThan(abs(f2)),
        select(abs(f1).lessThan(abs(f3)), f1, f3),
        select(abs(f2).lessThan(abs(f3)), f2, f3)
      );
      const fluteHalfW = max(float(0.0022), pxW.mul(0.95));
      const inFlute = float(1.0)
        .sub(smoothstep(fluteHalfW.mul(0.28), fluteHalfW.mul(1.22), abs(fluteSigned)))
        .mul(isTopFace)
        .mul(isStairTread)
        .mul(insideStairSpan);
      const fluteSlopeZ = clamp(fluteSigned.div(max(float(0.0010), fluteHalfW)), -1.0, 1.0)
        .negate()
        .mul(inFlute);
      grad.addAssign(vec3(0.0, 0.0, fluteSlopeZ.mul(0.76)));
      cavityAO.mulAssign(float(1.0).sub(inFlute.mul(0.36)));

      const dripSigned = distFromNose.sub(0.017);
      const inDrip = float(1.0)
        .sub(smoothstep(fluteHalfW.mul(0.3), fluteHalfW.mul(1.22), abs(dripSigned)))
        .mul(isBottomFace)
        .mul(isStairTread)
        .mul(insideStairSpan);
      const dripSlopeZ = clamp(dripSigned.div(max(float(0.0010), fluteHalfW)), -1.0, 1.0)
        .negate()
        .mul(inDrip);
      grad.addAssign(vec3(0.0, 0.0, dripSlopeZ.mul(0.55)));
      cavityAO.mulAssign(float(1.0).sub(inDrip.mul(0.32)));

      const rootGapMask = float(1.0)
        .sub(smoothstep(float(0.006), float(0.013), distFromRoot))
        .mul(isStairTread);
      const rootEdgeSlope = smoothstep(float(0.005), float(0.010), distFromRoot)
        .mul(float(1.0).sub(smoothstep(float(0.010), float(0.015), distFromRoot)))
        .mul(isStairTread);
      grad.addAssign(vec3(rootEdgeSlope.mul(0.65), 0.0, 0.0));
      cavityAO.mulAssign(float(1.0).sub(rootGapMask.mul(0.48)));

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
      const jointHalfW = max(float(0.0024), pxW.mul(0.95));
      const inJointX = float(1.0).sub(smoothstep(jointHalfW.mul(0.32), jointHalfW.mul(1.25), abs(dJointX)));
      const inJointZ = float(1.0).sub(smoothstep(jointHalfW.mul(0.32), jointHalfW.mul(1.25), abs(dJointZ)));
      const pavingGate = isTopFace.mul(isTravertine).mul(isTerrace);

      grad.addAssign(
        vec3(
          clamp(dJointX.div(max(float(0.0010), jointHalfW)), -1.0, 1.0).mul(inJointX).mul(pavingGate).mul(0.62),
          0.0,
          clamp(dJointZ.div(max(float(0.0010), jointHalfW)), -1.0, 1.0).mul(inJointZ).mul(pavingGate).mul(0.62)
        )
      );
      cavityAO.mulAssign(float(1.0).sub(max(inJointX, inJointZ).mul(pavingGate).mul(0.34)));

      // ---------------------------------------------------------------------
      // E. Dressed Dark Volcanic Basalt Horizontal Ashlar Courses (Stair Spine & Sub-Plinth)
      // ---------------------------------------------------------------------
      const isDarkStoneFlag = clamp(float(1.0).sub(isConcrete).sub(isTravertine), 0.0, 1.0);
      const basaltCourseH = float(0.24);
      const bPhase = fract(lp.y.div(basaltCourseH).add(100.0));
      const bSigned = select(bPhase.greaterThan(0.5), bPhase.sub(1.0), bPhase).mul(basaltCourseH);
      const bHalfW = max(float(0.0020), pxW.mul(0.95));
      const inBasaltJoint = float(1.0)
        .sub(smoothstep(bHalfW.mul(0.25), bHalfW.mul(1.25), abs(bSigned)))
        .mul(isVertical)
        .mul(isDarkStoneFlag);
      const bSlopeY = clamp(bSigned.div(max(float(0.0010), bHalfW)), -1.0, 1.0).mul(inBasaltJoint);
      grad.addAssign(vec3(0.0, bSlopeY.mul(0.72), 0.0));
      cavityAO.mulAssign(float(1.0).sub(inBasaltJoint.mul(0.38)));

      return vec4(grad, clamp(cavityAO, 0.28, 1.0));
    }
  );

  // =========================================================================
  // Shared Fragment Evaluation (single TSL graph shared across color, roughness, normal)
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
    const aaWidth = clamp(uvFw.mul(0.72), float(0.00055), float(0.013));
    const pxWorld = clamp(
      fwidth(wp.x).add(fwidth(wp.y)).add(fwidth(wp.z)).mul(0.52),
      float(0.00025),
      float(0.015)
    );

    // 2. Material Class weights: 0 = Board-Formed Concrete, 1 = Honed Travertine, 2 = Dark Basalt
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

    // 4. Face-aligned metric UVs & exact 3D tangent frames (zero seams + explicit mipLod!)
    const isTopOrBottom = abs(gn.y).greaterThan(0.62);
    const isXFace = abs(gn.x).greaterThan(abs(gn.z));
    const rawFaceUv = select(
      isTopOrBottom,
      vec2(lp.x, lp.z),
      select(isXFace, vec2(lp.z, lp.y), vec2(lp.x, lp.y))
    );
    const tanU = select(
      isTopOrBottom,
      vec3(1.0, 0.0, 0.0),
      select(isXFace, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0))
    );
    const tanV = select(isTopOrBottom, vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));

    // Per-plank horizontal woodgrain seed so neighboring concrete shuttering courses never repeat
    const plankBandId = floor(lp.y.div(0.16).add(100.0));
    const plankToneRand = hash31Tsl(vec3(plankBandId, aSlabId.mul(3.17), float(1.37)));
    const plankUvShift = hash31Tsl(vec3(plankBandId.mul(1.73), aSlabId.mul(5.11), float(2.83)));
    const plankFlipV = hash31Tsl(vec3(plankBandId.mul(2.39), aSlabId.mul(1.93), float(6.17)));
    const plankOffsetU = select(isTopOrBottom, float(0.0), plankUvShift.mul(0.79));
    const plankOffsetV = select(isTopOrBottom, float(0.0), plankFlipV.mul(0.31));

    // 4a. Seamless 3D Volumetric Petrography (64^3 Voronoi clasts, 3D vesicles, sand)
    const uvwMacro = rotLithicA(lp.mul(2.15));
    const uvwMicro = rotLithicB(lp.mul(7.6)).add(vec3(0.37, 0.61, 0.19));
    const volMacro = texture3DLevel(petro.stoneVolumeTex, fract(uvwMacro.add(10.0)), float(0.0));
    const volMicro = texture3DLevel(petro.stoneVolumeTex, fract(uvwMicro.add(10.0)), float(0.0));

    // 4b. High-Resolution 512^2 Concrete Detail Texture (Dual-Scale explicit mipLod!)
    const concFreq0 = 2.15; // ~46 cm/tile -> 0.90 mm/texel (cedar woodgrain + realistic sparse bugholes)
    const concFreq1 = 6.9; // ~14.5 cm/tile -> 0.28 mm/texel (ultra-fine Portland sand & quartz aggregate!)
    const concUv0 = vec2(rawFaceUv.x.add(plankOffsetU), rawFaceUv.y.add(plankOffsetV)).mul(concFreq0);
    // Irrational-angle rotated fine sand octave so micro-sand never aligns with meso tile
    const concUv1 = vec2(
      rawFaceUv.x.mul(0.809).sub(rawFaceUv.y.mul(0.588)),
      rawFaceUv.x.mul(0.588).add(rawFaceUv.y.mul(0.809))
    )
      .mul(concFreq1)
      .add(vec2(0.31, 0.67));

    const mipConc0 = clamp(
      log2(max(float(1.0), pxWorld.mul(DETAIL_TEX_SIZE * concFreq0))).sub(0.38),
      float(0.0),
      float(6.5)
    );
    const mipConc1 = clamp(
      log2(max(float(1.0), pxWorld.mul(DETAIL_TEX_SIZE * concFreq1))).sub(0.30),
      float(0.0),
      float(6.5)
    );

    const conc0 = texture(petro.concreteDetailTex, fract(concUv0.add(10.0))).level(mipConc0);
    const conc1 = texture(petro.concreteDetailTex, fract(concUv1.add(10.0))).level(mipConc1);

    // Finite-difference concrete height taps (strictly uniform control flow, scaled to active mip!)
    const duConc0 = clamp(exp2(mipConc0).div(DETAIL_TEX_SIZE), float(1.0 / DETAIL_TEX_SIZE), float(6.0 / DETAIL_TEX_SIZE));
    const duConc1 = clamp(exp2(mipConc1).div(DETAIL_TEX_SIZE), float(1.0 / DETAIL_TEX_SIZE), float(6.0 / DETAIL_TEX_SIZE));
    const conc0Hx = texture(petro.concreteDetailTex, fract(concUv0.add(vec2(duConc0, 0.0)).add(10.0))).level(mipConc0).a;
    const conc0Hy = texture(petro.concreteDetailTex, fract(concUv0.add(vec2(0.0, duConc0)).add(10.0))).level(mipConc0).a;
    const conc1Hx = texture(petro.concreteDetailTex, fract(concUv1.add(vec2(duConc1, 0.0)).add(10.0))).level(mipConc1).a;
    const conc1Hy = texture(petro.concreteDetailTex, fract(concUv1.add(vec2(0.0, duConc1)).add(10.0))).level(mipConc1).a;

    // 4c. High-Resolution 512^2 Honed Roman Travertine Texture (flowing bedding + steep open vugs)
    const travFreq0 = 2.05;
    const travFreq1 = 5.1;
    const travBaseU = select(isTopOrBottom, lp.x.add(lp.z.mul(0.28)), select(isXFace, lp.z, lp.x).add(aSlabId.mul(0.17)));
    const travBaseV = select(isTopOrBottom, lp.z.mul(0.88).add(lp.x.mul(0.14)), lp.y.add(aSlabId.mul(0.09)));
    const travUv0 = vec2(travBaseU, travBaseV).mul(travFreq0);
    const travUv1 = vec2(travBaseU.add(0.41), travBaseV.add(0.23)).mul(travFreq1);

    const mipTrav0 = clamp(
      log2(max(float(1.0), pxWorld.mul(DETAIL_TEX_SIZE * travFreq0))).sub(0.38),
      float(0.0),
      float(6.5)
    );
    const mipTrav1 = clamp(
      log2(max(float(1.0), pxWorld.mul(DETAIL_TEX_SIZE * travFreq1))).sub(0.30),
      float(0.0),
      float(6.5)
    );
    const trav0 = texture(petro.travertineDetailTex, fract(travUv0.add(10.0))).level(mipTrav0);
    const trav1 = texture(petro.travertineDetailTex, fract(travUv1.add(10.0))).level(mipTrav1);

    const duTrav0 = clamp(exp2(mipTrav0).div(DETAIL_TEX_SIZE), float(1.0 / DETAIL_TEX_SIZE), float(6.0 / DETAIL_TEX_SIZE));
    const trav0Hx = texture(petro.travertineDetailTex, fract(travUv0.add(vec2(duTrav0, 0.0)).add(10.0))).level(mipTrav0).a;
    const trav0Hy = texture(petro.travertineDetailTex, fract(travUv0.add(vec2(0.0, duTrav0)).add(10.0))).level(mipTrav0).a;

    // 4d. High-Resolution 512^2 Architectural Paint Stipple & Edge-Tooth Texture
    const paintFreq0 = 3.2;
    const paintFreq1 = 9.4;
    const paintUv0 = rawFaceUv.mul(paintFreq0);
    const paintUv1 = vec2(
      rawFaceUv.x.mul(0.866).add(rawFaceUv.y.mul(0.5)),
      rawFaceUv.y.mul(0.866).sub(rawFaceUv.x.mul(0.5))
    ).mul(paintFreq1);
    const mipPaint0 = clamp(
      log2(max(float(1.0), pxWorld.mul(DETAIL_TEX_SIZE * paintFreq0))).sub(0.35),
      float(0.0),
      float(6.5)
    );
    const mipPaint1 = clamp(
      log2(max(float(1.0), pxWorld.mul(DETAIL_TEX_SIZE * paintFreq1))).sub(0.25),
      float(0.0),
      float(6.5)
    );
    const paintTex0 = texture(petro.paintStippleTex, fract(paintUv0.add(10.0))).level(mipPaint0);
    const paintTex1 = texture(petro.paintStippleTex, fract(paintUv1.add(10.0))).level(mipPaint1);
    const duPaint = clamp(exp2(mipPaint1).div(DETAIL_TEX_SIZE), float(1.0 / DETAIL_TEX_SIZE), float(6.0 / DETAIL_TEX_SIZE));
    const paintHx = texture(petro.paintStippleTex, fract(paintUv1.add(vec2(duPaint, 0.0)).add(10.0))).level(mipPaint1).b;
    const paintHy = texture(petro.paintStippleTex, fract(paintUv1.add(vec2(0.0, duPaint)).add(10.0))).level(mipPaint1).b;

    // Zero-ALU-hash curing laitance clouding & subtle vertical moisture runoff beneath top ledges
    const macroLaitance = volMacro.b.mul(0.65).add(volMacro.r.mul(0.35));
    const topLedgeDist = float(1.0).sub(aSlabLocal.y).mul(aSlabSize.y);
    const rainStreak = smoothstep(float(0.52), float(0.84), volMicro.b.mul(0.55).add(conc0.r.mul(0.45)))
      .mul(exp(topLedgeDist.mul(-5.8)))
      .mul(isVertical)
      .mul(isConcrete);

    // 5. Analytical Architectural Masonry Relief (V-grooves, butt-joints, P-cones, stair flutes, paving joints, basalt courses)
    const archRelief = evalArchitecturalMasonryRelief(
      lp,
      gn,
      aSlabLocal,
      aSlabSize,
      aKindFlags,
      aSlabId,
      isConcrete,
      isTravertine,
      pxWorld
    );
    const archGrad = archRelief.xyz;
    const archCavityAO = archRelief.w;
    const archGrooveMask = clamp(float(1.0).sub(archCavityAO).mul(2.4), 0.0, 1.0);

    // Concrete micro-features: smooth Portland paste + ultra-fine sand + sparse authentic bugholes (NO 2nd-octave pepper!)
    const concSand = conc0.r.mul(0.48).add(conc1.r.mul(0.38)).add(volMicro.b.mul(0.14));
    const concBughole = clamp(
      conc0.g.mul(0.92).add(smoothstep(float(0.72), float(0.95), volMicro.g).mul(0.28)),
      0.0,
      1.0
    ).mul(isConcrete);
    const cedarFiber = conc0.b;
    const quartzGlint = smoothstep(float(0.78), float(0.94), conc1.r.mul(0.72).add(volMicro.r.mul(0.28)));
    const darkSlagSpeck = smoothstep(float(0.20), float(0.06), conc1.r.mul(0.65).add(volMicro.r.mul(0.35)));

    // Roman Travertine micro-features (sharp elongated horizontal vugs + sedimentary banding + calcite sparkle)
    const travStrata = trav0.r.mul(0.72).add(trav1.r.mul(0.28));
    const travVug = clamp(
      trav0.g.mul(0.92).add(trav1.g.mul(0.36)),
      0.0,
      1.0
    ).mul(isTravertine);
    const travCalcite = trav0.b.mul(0.55).add(trav1.b.mul(0.45));
    const calciteSparkle = smoothstep(float(0.79), float(0.95), trav1.b.mul(0.68).add(volMicro.r.mul(0.32)))
      .mul(float(1.0).sub(travVug));

    // Honed travertine courtyard ashlar paving per-tile tonal variation
    const tileRow = floor(lp.z.div(0.32).add(100.0));
    const tileCol = floor(lp.x.add(fract(tileRow.mul(0.5)).mul(0.38)).div(0.38).add(100.0));
    const tileToneRand = hash31Tsl(vec3(tileCol, tileRow, aSlabId.mul(1.73)));
    const isTerraceTop = smoothstep(float(0.72), float(0.92), gn.y).mul(aKindFlags.y).mul(isTravertine);

    // 45-degree chamfer arris catch (subtle pale formwork edge wear + micro-spall breakup)
    const rawBevel = smoothstep(
      float(1.16),
      float(1.36),
      abs(gn.x).add(abs(gn.y)).add(abs(gn.z))
    );
    const arrisChipNoise = volMicro.r.mul(0.55).add(conc1.r.mul(0.45));
    const bevelCatch = rawBevel.mul(mix(float(0.68), float(1.22), arrisChipNoise));

    // 6. Un-Occluded Mineral Substrate Albedos (strictly from MINERAL_PALETTE)
    // Class 0: Refined Cast Board-Formed Architectural Concrete (Tadao Ando smooth paste + cedar grain + fine sand + sparse bugholes)
    const concWarm = boneCalciteRgb.mul(0.77);
    const concCool = mix(warmKaolinRgb.mul(0.67), azuriteRgb.mul(0.30).add(vec3(0.46, 0.46, 0.44)), float(0.32));
    const concBase = mix(concCool, concWarm, clamp(macroLaitance.mul(0.76).add(volMacro.r.mul(0.24)), 0.0, 1.0));
    const plankShift = mix(float(0.94), float(1.052), plankToneRand);
    const cedarContrast = mix(float(0.935), float(1.055), cedarFiber);
    const concreteColor = concBase
      .mul(mix(float(1.0), plankShift, isVertical))
      .mul(mix(float(1.0), cedarContrast, isVertical.mul(0.72)))
      .mul(mix(float(0.93), float(1.07), concSand))
      .mul(float(1.0).sub(darkSlagSpeck.mul(0.19)))
      .add(boneCalciteRgb.mul(quartzGlint.mul(0.045)))
      .mul(float(1.0).sub(rainStreak.mul(0.11)))
      .mul(float(1.0).sub(concBughole.mul(0.38)));

    // Class 1: Honed & Open-Pore Roman Travertine (warm layered calcite + ochre veins + dark cavernous vugs)
    const travCream = mix(warmKaolinRgb.mul(0.68), boneCalciteRgb.mul(0.83), travStrata);
    const travOchreVein = mix(
      travCream,
      yellowOchreRgb.mul(0.60),
      smoothstep(float(0.35), float(0.72), float(1.0).sub(travStrata)).mul(0.26)
    );
    const travAshlar = travOchreVein.mul(
      mix(float(1.0), mix(float(0.945), float(1.055), tileToneRand), isTerraceTop)
    );
    const travVugFloor = mix(rawUmberRgb.mul(0.46), yellowOchreRgb.mul(0.30), travCalcite);
    const travColor = mix(travAshlar, travVugFloor, clamp(travVug.mul(0.82), 0.0, 0.85))
      .mul(mix(float(0.94), float(1.06), travCalcite))
      .add(boneCalciteRgb.mul(calciteSparkle.mul(0.042)));

    // Class 2: Dressed Dark Volcanic Basalt (cool charcoal-umber matrix + vesicles + crystalline mineral flecks)
    const basaltVesicle = clamp(volMicro.g.mul(0.82).add(volMacro.g.mul(0.46)), 0.0, 1.0);
    const basaltCrystal = smoothstep(float(0.70), float(0.90), conc1.r.mul(0.58).add(volMicro.r.mul(0.42)));
    const basaltClast = volMacro.r.mul(0.45).add(volMicro.r.mul(0.55));
    const basaltBase = mix(
      rawUmberRgb.mul(0.34).add(azuriteRgb.mul(0.10)),
      rawUmberRgb.mul(0.66).add(azuriteRgb.mul(0.08)),
      basaltClast
    );
    const darkStoneColor = basaltBase
      .mul(mix(float(0.86), float(1.14), volMicro.b))
      .mul(float(1.0).sub(basaltVesicle.mul(0.58)))
      .add(boneCalciteRgb.mul(basaltCrystal.mul(0.068)));

    const substrateAlbedo = mix(
      mix(concreteColor, travColor, isTravertine),
      darkStoneColor,
      isDarkStone
    );

    // 7. Physical Projective Stencil with Masonry Capillary Edge Tooth & Pore Micro-Skipping
    // Distant overview stays razor-crisp so the locked-in illusion looks impossibly seamless;
    // close-up & micro-oblique inspection reveal real pigment capillary tooth along concrete & travertine grain!
    const closeInspection = float(1.0).sub(smoothstep(float(0.0012), float(0.0038), pxWorld));
    const capillaryNoise = paintTex1.g
      .sub(0.5)
      .mul(0.48)
      .add(conc1.r.sub(0.5).mul(0.32).mul(isConcrete))
      .add(cedarFiber.sub(0.5).mul(0.28).mul(isConcrete).mul(isVertical))
      .add(trav1.b.sub(0.5).mul(0.24).mul(isTravertine));

    const uvPerWorld = clamp(uvFw.div(max(float(0.0004), pxWorld)), float(0.25), float(2.4));
    // Sub-millimeter capillary bleed (0.4 mm baseline rising smoothly to ~2.0 mm on close macro zoom)
    const toothWorldScale = mix(float(0.00045), float(0.0021), closeInspection);
    const edgeToothJitter = capillaryNoise
      .mul(toothWorldScale)
      .mul(uvPerWorld)
      .add(travVug.mul(0.0022).mul(uvPerWorld).mul(closeInspection))
      .add(concBughole.mul(0.0018).mul(uvPerWorld).mul(closeInspection));

    const stencilMasks = evalStencilTsl(projUv, aaWidth, edgeToothJitter);

    // Clean, quiet simultaneous stencil settle (zero depth sweep, zero wave, zero particles)
    const revealWave = smoothstep(float(0.0), float(1.0), uFlashProgress);

    const rawPriPaint = stencilMasks.x.mul(visibility).mul(revealWave);
    const rawSecPaint = stencilMasks.y.mul(visibility).mul(revealWave);
    const rawTotalPaint = clamp(rawPriPaint.add(rawSecPaint), 0.0, 1.0);

    // Physical paint bridging & skipping across rough masonry relief:
    // - Deep concrete bugholes & deep travertine vugs punch raw stone cavities right through the paint!
    // - Sharp chamfered corners show tiny worn dry-brush mineral flecks
    // - Sparse micro-skips over extreme rough sand peaks break up flat vector homogeneity at macro zoom
    const bugholeBreak = smoothstep(float(0.30), float(0.76), concBughole).mul(0.72);
    const vugBreak = smoothstep(float(0.26), float(0.72), travVug).mul(0.78);
    const chamferChipSkip = rawBevel
      .mul(smoothstep(float(0.56), float(0.86), arrisChipNoise))
      .mul(mix(float(0.22), float(0.58), closeInspection));
    const microDrySkip = paintTex1.a
      .mul(smoothstep(float(0.68), float(0.92), conc1.r))
      .mul(isConcrete)
      .mul(0.30)
      .mul(closeInspection);
    const grooveHairlineBreak = smoothstep(float(0.62), float(0.95), archGrooveMask)
      .mul(0.26)
      .mul(closeInspection);

    const paintRetention = clamp(
      float(1.0)
        .sub(max(bugholeBreak, vugBreak))
        .sub(chamferChipSkip)
        .sub(microDrySkip)
        .sub(grooveHairlineBreak),
      0.16,
      1.0
    );

    const priPaint = rawPriPaint.mul(paintRetention);
    const secPaint = rawSecPaint.mul(paintRetention);
    const totalPaint = clamp(priPaint.add(secPaint), 0.0, 1.0);

    // Masking-tape edge build-up meniscus (subtle thicker pigment lip along stencil boundary)
    const edgeMeniscus = rawTotalPaint.mul(float(1.0).sub(rawTotalPaint)).mul(4.0);

    // 8. True 2-Flux Kubelka-Munk Mineral Pigment Glaze over Real Masonry Albedo
    // Moderate physical coat thickness (1.10..1.95) lets underlying concrete plank variation,
    // cedar grain absorption, and travertine strata live authentically inside painted bands!
    const rollerCoat = mix(float(0.80), float(1.18), paintTex0.r.mul(0.55).add(paintTex1.r.mul(0.45)));
    const stoneAbsorb = mix(
      mix(float(0.88), float(1.16), concSand),
      mix(float(0.96), float(1.12), travStrata),
      isTravertine
    );
    const edgeLipBuild = float(1.0).add(edgeMeniscus.mul(0.22));
    const priThick = priPaint.mul(1.52).mul(rollerCoat).mul(stoneAbsorb).mul(edgeLipBuild);
    const secThick = secPaint.mul(1.48).mul(rollerCoat).mul(stoneAbsorb).mul(edgeLipBuild);

    const afterPri = kubelkaMunkGlaze(uPriK, uPriS, priThick, substrateAlbedo);
    const paintedAlbedo = kubelkaMunkGlaze(uSecK, uSecS, secThick, afterPri);

    // Subtle physical pigment-absorption darkening inside masonry pores/grooves & plank absorption shift
    const substrateToneMod = mix(
      float(1.0),
      mix(float(0.93), float(1.05), concSand.mul(0.5).add(cedarFiber.mul(0.5).mul(isVertical))),
      totalPaint.mul(isConcrete).mul(0.55)
    );

    // 9. Apply TRUE Physical Incident Ambient Occlusion AFTER Pigment Glaze!
    const structAO = clamp(
      float(1.0).sub(tightAO.mul(0.48)).sub(broadAO.mul(0.26)),
      0.28,
      1.0
    );
    const microCavityAO = clamp(
      float(1.0)
        .sub(travVug.mul(0.48))
        .sub(concBughole.mul(0.44))
        .sub(basaltVesicle.mul(isDarkStone).mul(0.42)),
      0.32,
      1.0
    );
    const totalAO = clamp(structAO.mul(archCavityAO).mul(microCavityAO), 0.18, 1.0);

    // Subtle pointer hover warmth on bare stone + optical scan-flash + lock-in resonance
    const ptrDist = length(wp.sub(uPointerWorld));
    const ptrWarmth = exp(ptrDist.mul(ptrDist).mul(-3.2))
      .mul(uPointerActive)
      .mul(float(1.0).sub(totalPaint))
      .mul(0.036);

    const lockHighlight = totalPaint.mul(uLockGlow).mul(0.032);

    const finalRgb = paintedAlbedo
      .mul(substrateToneMod)
      .mul(totalAO)
      .mul(float(1.0).add(bevelCatch.mul(0.085)))
      .add(boneCalciteRgb.mul(ptrWarmth))
      .mul(float(1.0).add(lockHighlight));

    // 10. Authentic Roughness Contrast: Chalky Concrete vs Satin Honed Travertine vs Eggshell Mineral Paint!
    // At subtle micro-oblique raking angles, this distinct specular transition reveals the physical paint film
    // bonded right onto the rough concrete and honed stone!
    const stairWear = aKindFlags.x
      .mul(smoothstep(float(0.72), float(0.92), gn.y))
      .mul(clamp(float(1.0).sub(abs(aSlabLocal.x.sub(0.52)).mul(1.8)), 0.0, 1.0));

    const concreteRough = float(0.88)
      .add(concSand.sub(0.5).mul(0.12))
      .sub(quartzGlint.mul(0.24))
      .sub(macroLaitance.mul(0.05))
      .add(concBughole.mul(0.12));

    const travRough = float(0.56)
      .sub(stairWear.mul(0.09))
      .add(travCalcite.sub(0.5).mul(0.10))
      .sub(calciteSparkle.mul(0.12))
      .add(travVug.mul(0.40))
      .add(archGrooveMask.mul(0.36));

    const basaltRough = float(0.62)
      .add(volMicro.b.sub(0.5).mul(0.14))
      .sub(basaltCrystal.mul(0.24))
      .add(basaltVesicle.mul(0.32));

    const bareStoneRough = mix(
      mix(concreteRough, travRough, isTravertine),
      basaltRough,
      isDarkStone
    );

    // Dried mineral/casein architectural paint:
    // - Velvety eggshell pigment binder over chalky cast concrete (~0.56..0.66) for crisp raking micro-oblique sheen contrast!
    // - Slightly more matte over satin-honed travertine (~0.66..0.75)
    // - Still turns dead-rough inside deep masonry pits & grooves!
    const paintFilmRough = mix(float(0.57), float(0.70), isTravertine)
      .add(paintTex0.r.sub(0.5).mul(0.10))
      .add(paintTex1.r.sub(0.5).mul(0.08))
      .add(edgeMeniscus.mul(-0.05))
      .add(max(concBughole, travVug).mul(0.32))
      .add(archGrooveMask.mul(0.22));

    const finalRough = clamp(mix(bareStoneRough, paintFilmRough, totalPaint), 0.36, 0.97);

    // 11. Multi-Scale Physical Composite Normal:
    //     Analytical Masonry Relief + Dual-Scale 512^2 Concrete/Travertine Height Gradients
    //     + Seamless 3D Volumetric Lithic Normal + Fine Paint-Roller Stipple Normal!
    const concGrad0 = tanU
      .mul(conc0Hx.sub(conc0.a))
      .add(tanV.mul(conc0Hy.sub(conc0.a)))
      .div(max(float(0.002), duConc0.mul(2.1)));
    // Rotate fine sand gradient back from irrational angle (c=0.809, s=0.588) into (tanU, tanV)
    const d1u = conc1Hx.sub(conc1.a).div(max(float(0.002), duConc1.mul(2.4)));
    const d1v = conc1Hy.sub(conc1.a).div(max(float(0.002), duConc1.mul(2.4)));
    const concGrad1 = tanU
      .mul(d1u.mul(0.809).add(d1v.mul(0.588)))
      .add(tanV.mul(d1u.mul(-0.588).add(d1v.mul(0.809))));

    const travTanU = select(
      isTopOrBottom,
      vec3(1.0, 0.0, 0.28),
      select(isXFace, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0))
    );
    const travTanV = select(isTopOrBottom, vec3(0.14, 0.0, 0.88), vec3(0.0, 1.0, 0.0));
    const travGrad0 = travTanU
      .mul(trav0Hx.sub(trav0.a))
      .add(travTanV.mul(trav0Hy.sub(trav0.a)))
      .div(max(float(0.002), duTrav0.mul(1.85)));

    // 3D volumetric finite-difference normal for seamless basalt vesicles & corner bevel grain
    const du3 = float(0.018);
    const vHx = texture3DLevel(petro.stoneVolumeTex, fract(uvwMicro.add(vec3(du3, 0.0, 0.0)).add(10.0)), float(0.0)).a;
    const vHy = texture3DLevel(petro.stoneVolumeTex, fract(uvwMicro.add(vec3(0.0, du3, 0.0)).add(10.0)), float(0.0)).a;
    const vHz = texture3DLevel(petro.stoneVolumeTex, fract(uvwMicro.add(vec3(0.0, 0.0, du3)).add(10.0)), float(0.0)).a;
    const volGradLocal = rotLithicBTranspose(
      vec3(vHx.sub(volMicro.a), vHy.sub(volMicro.a), vHz.sub(volMicro.a)).div(du3.mul(2.6))
    );

    // Fine paint-roller stipple micro-normal (only active on painted surfaces & fades cleanly at far distance)
    const pDu = paintHx.sub(paintTex1.b).div(max(float(0.002), duPaint.mul(3.2)));
    const pDv = paintHy.sub(paintTex1.b).div(max(float(0.002), duPaint.mul(3.2)));
    const paintGrad = tanU
      .mul(pDu.mul(0.866).sub(pDv.mul(0.5)))
      .add(tanV.mul(pDu.mul(0.5).add(pDv.mul(0.866))));

    const fineNormalFade = float(1.0).sub(smoothstep(float(0.0024), float(0.0062), pxWorld));

    // Preserve 88% of concrete/travertine relief right THROUGH the paint film so stone reads unmistakably underneath!
    const paintConform = mix(float(1.0), float(0.88), totalPaint);

    const concreteNormGrad = concGrad0
      .mul(0.032)
      .add(concGrad1.mul(0.022).mul(fineNormalFade))
      .add(volGradLocal.mul(0.085));

    const travNormGrad = travGrad0
      .mul(0.056)
      .add(volGradLocal.mul(0.070));

    const basaltNormGrad = volGradLocal
      .mul(0.32)
      .add(concGrad1.mul(0.024).mul(fineNormalFade));

    const stoneGrad = mix(
      mix(concreteNormGrad, travNormGrad, isTravertine),
      basaltNormGrad,
      isDarkStone
    ).mul(paintConform);

    const paintStippleGrad = paintGrad.mul(0.015).mul(totalPaint).mul(fineNormalFade);

    const totalGrad = archGrad.add(stoneGrad).add(paintStippleGrad);
    const nLoc = normalize(normalLocal);
    const tanGrad = totalGrad.sub(nLoc.mul(dot(totalGrad, nLoc)));
    // Softly bound extreme tangent gradient magnitudes so grazing faces never invert
    const clampedTan = tanGrad.div(max(float(1.0), length(tanGrad).div(1.28)));
    const finalNormalView = transformNormalToView(normalize(nLoc.sub(clampedTan)));

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
      petro.dispose();
      courtyardMaterial.dispose();
      brassBezelMaterial.dispose();
      brassTicksMaterial.dispose();
    },
  };
}
