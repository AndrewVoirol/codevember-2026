import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  cross,
  dot,
  exp,
  float,
  floor,
  fract,
  length,
  max,
  mix,
  normalGeometry,
  normalLocal,
  normalWorld,
  normalize,
  positionGeometry,
  positionLocal,
  positionWorld,
  pow,
  select,
  sin,
  smoothstep,
  sqrt,
  texture,
  texture3DLevel,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { CHALKY_SLATE_LINEAR } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { type SpecimenStage } from '../../core/stage/SpecimenStage';
import { WET_ATLAS_HEIGHT, WET_ATLAS_WIDTH } from './types';
import { createPetrographicVolumeTexture } from './labyrinthPetrography';

export { WET_ATLAS_HEIGHT, WET_ATLAS_WIDTH };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface LabyrinthOpticsBundle {
  stoneMaterial: THREE.MeshStandardNodeMaterial;
  sanctumMaterial: THREE.MeshStandardNodeMaterial;
  startSpringMaterial: THREE.MeshStandardNodeMaterial;
  waystoneMaterial: THREE.MeshStandardNodeMaterial;
  beadMaterial: THREE.MeshPhysicalNodeMaterial;
  wetTrailTexture: THREE.DataTexture;
  updateBeadUniforms: (
    velocityLocal: readonly [number, number, number] | THREE.Vector3,
    normalLocalVec: readonly [number, number, number] | THREE.Vector3,
    impactImpulse: number,
    solveProgress01?: number,
    dtSec?: number,
    rollQuat?: readonly [number, number, number, number] | THREE.Quaternion
  ) => void;
  getBeadSloshLocal: (out: [number, number, number]) => [number, number, number];
  setProgressUniforms: (checkpointsUnlocked: number, dowsingHint01: number) => void;
  setVictoryUniforms: (
    solveProgress01: number,
    victoryIgnition01: number,
    riverFlow01?: number
  ) => void;
  dispose: () => void;
}

/**
 * Primary orthonormal off-axis 3D basis rotation eliminating any axis-aligned
 * trilinear lattice artifacts on X/Y/Z corridor faces or 45-degree bevels.
 */
const rotLithicA = Fn(([pos_in]: [TslNode]) => {
  const p = vec3(pos_in);
  const u = vec3(0.6324555, 0.5477226, -0.5477226);
  const v = vec3(-0.4647580, 0.8340136, 0.2973573);
  const w = vec3(0.6196773, 0.0664932, 0.7820349);
  return vec3(dot(p, u), dot(p, v), dot(p, w));
});

/**
 * Secondary skew-orthogonal 3D rotation basis (1/7 Pythagorean SO(3) matrix)
 * decorrelating fine mineral clasts & pin-pores from macro coordinates and 45-deg bevels.
 */
const rotLithicB = Fn(([pos_in]: [TslNode]) => {
  const p = vec3(pos_in);
  const u = vec3(0.4285714, -0.8571429, 0.2857143);
  const v = vec3(0.8571429, 0.2857143, -0.4285714);
  const w = vec3(0.2857143, 0.4285714, 0.8571429);
  return vec3(dot(p, u), dot(p, v), dot(p, w));
});

const rotLithicATranspose = Fn(([grad_in]: [TslNode]) => {
  const g = vec3(grad_in);
  const x = dot(g, vec3(0.6324555, -0.4647580, 0.6196773));
  const y = dot(g, vec3(0.5477226, 0.8340136, 0.0664932));
  const z = dot(g, vec3(-0.5477226, 0.2973573, 0.7820349));
  return vec3(x, y, z);
});

const rotLithicBTranspose = Fn(([grad_in]: [TslNode]) => {
  const g = vec3(grad_in);
  const x = dot(g, vec3(0.4285714, 0.8571429, 0.2857143));
  const y = dot(g, vec3(-0.8571429, 0.2857143, 0.4285714));
  const z = dot(g, vec3(0.2857143, -0.4285714, 0.8571429));
  return vec3(x, y, z);
});

/**
 * Metal-safe 3D lattice hash shifted strictly onto positive inputs before fract().
 */
const hash31SafeTsl = Fn(([p_in]: [TslNode]) => {
  const p3 = fract(vec3(p_in).mul(vec3(0.1031, 0.103, 0.0973)).add(100.0));
  const d = dot(p3, p3.yzx.add(33.33));
  const pShift = p3.add(d);
  return fract(pShift.x.add(pShift.y).mul(pShift.z));
});

/**
 * Analytical C1 3D value noise returning vec4(val, dVal/dx, dVal/dy, dVal/dz)
 * from a single 8-corner hash evaluation (infinite resolution, zero texel blur).
 */
const valueNoise3DGradTsl = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const i = floor(p);
  const f = p.sub(i);
  const u = f.mul(f).mul(float(3.0).sub(f.mul(2.0)));
  const du = f.mul(float(1.0).sub(f)).mul(6.0);

  const n000 = hash31SafeTsl(i);
  const n100 = hash31SafeTsl(i.add(vec3(1.0, 0.0, 0.0)));
  const n010 = hash31SafeTsl(i.add(vec3(0.0, 1.0, 0.0)));
  const n110 = hash31SafeTsl(i.add(vec3(1.0, 1.0, 0.0)));
  const n001 = hash31SafeTsl(i.add(vec3(0.0, 0.0, 1.0)));
  const n101 = hash31SafeTsl(i.add(vec3(1.0, 0.0, 1.0)));
  const n011 = hash31SafeTsl(i.add(vec3(0.0, 1.0, 1.0)));
  const n111 = hash31SafeTsl(i.add(vec3(1.0, 1.0, 1.0)));

  const dx00 = n100.sub(n000);
  const dx10 = n110.sub(n010);
  const dx01 = n101.sub(n001);
  const dx11 = n111.sub(n011);

  const nx00 = n000.add(u.x.mul(dx00));
  const nx10 = n010.add(u.x.mul(dx10));
  const nx01 = n001.add(u.x.mul(dx01));
  const nx11 = n011.add(u.x.mul(dx11));

  const nxy0 = mix(nx00, nx10, u.y);
  const nxy1 = mix(nx01, nx11, u.y);
  const val = mix(nxy0, nxy1, u.z);

  const dvdx = du.x.mul(mix(mix(dx00, dx10, u.y), mix(dx01, dx11, u.y), u.z));
  const dvdy = du.y.mul(mix(nx10.sub(nx00), nx11.sub(nx01), u.z));
  const dvdz = du.z.mul(nxy1.sub(nxy0));

  return vec4(val, dvdx, dvdy, dvdz);
});

/**
 * Builds all four physical Kubelka-Munk / liquid-metal materials and pre-allocated
 * GPU uniform & wet-trail texture bindings for Day 13 — Labyrinth.
 */
export function createLabyrinthOptics(
  stage: SpecimenStage,
  wetAtlasData: Uint8Array,
  beadRadius = 0.046,
  goalLocalPos: readonly [number, number, number] = [0.43, 0.735, 0]
): LabyrinthOpticsBundle {
  // -------------------------------------------------------------------------
  // Mineral Pigment Packs (Strictly Natural Pigments from palette.ts)
  // -------------------------------------------------------------------------
  const rawUmber = createPigmentUniforms('rawUmber');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const burntSienna = createPigmentUniforms('burntSienna');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const azuritePrussian = createPigmentUniforms('azuritePrussian');
  const verdigrisMalachite = createPigmentUniforms('verdigrisMalachite');
  const celadonGlauconite = createPigmentUniforms('celadonGlauconite');

  const petroVolumeTex = createPetrographicVolumeTexture();

  /**
   * Tangent-projected 3-tap forward-difference 3D volume relief normal perturbation
   * strictly evaluated in uniform control flow with positive fract(+10) guard.
   */
  const perturbPetroNormalLocal = Fn(
    ([baseNorm_in, uvwMicro_in, hCenter_in, strength_in]: [
      TslNode,
      TslNode,
      TslNode,
      TslNode,
    ]) => {
      const uvw = vec3(uvwMicro_in);
      const du = float(0.016);
      const hx = texture3DLevel(petroVolumeTex, fract(uvw.add(vec3(du, 0.0, 0.0)).add(10.0)), float(0.0)).a;
      const hy = texture3DLevel(petroVolumeTex, fract(uvw.add(vec3(0.0, du, 0.0)).add(10.0)), float(0.0)).a;
      const hz = texture3DLevel(petroVolumeTex, fract(uvw.add(vec3(0.0, 0.0, du)).add(10.0)), float(0.0)).a;
      const gradUvw = vec3(hx.sub(hCenter_in), hy.sub(hCenter_in), hz.sub(hCenter_in)).div(
        du.mul(3.0)
      );
      const gradLocal = rotLithicBTranspose(gradUvw);
      const bn = normalize(vec3(baseNorm_in));
      const gradTan = gradLocal.sub(bn.mul(dot(gradLocal, bn)));
      const clampedTan = gradTan.div(max(float(1.0), length(gradTan).div(1.25)));
      return normalize(bn.sub(clampedTan.mul(float(strength_in))));
    }
  );

  /**
   * Tri-band stone normal perturbation fusing 3D petrography volume gradients
   * with analytical C1 crystalline rock facets projected into the tangent plane.
   */
  const perturbStoneNormalLocal = Fn(
    ([
      baseNorm_in,
      uvwMicro_in,
      hMicro_in,
      uvwGrit_in,
      hGrit_in,
      gradCrysA_in,
      gradCrysB_in,
      strength_in,
    ]: [
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
    ]) => {
      const uvwM = vec3(uvwMicro_in);
      const duM = float(0.016);
      const mx = texture3DLevel(petroVolumeTex, fract(uvwM.add(vec3(duM, 0.0, 0.0)).add(10.0)), float(0.0)).a;
      const my = texture3DLevel(petroVolumeTex, fract(uvwM.add(vec3(0.0, duM, 0.0)).add(10.0)), float(0.0)).a;
      const mz = texture3DLevel(petroVolumeTex, fract(uvwM.add(vec3(0.0, 0.0, duM)).add(10.0)), float(0.0)).a;
      const gradMicro = rotLithicBTranspose(
        vec3(mx.sub(hMicro_in), my.sub(hMicro_in), mz.sub(hMicro_in)).div(duM.mul(2.4))
      );

      const uvwG = vec3(uvwGrit_in);
      const duG = float(0.016);
      const gx = texture3DLevel(petroVolumeTex, fract(uvwG.add(vec3(duG, 0.0, 0.0)).add(10.0)), float(0.0)).a;
      const gy = texture3DLevel(petroVolumeTex, fract(uvwG.add(vec3(0.0, duG, 0.0)).add(10.0)), float(0.0)).a;
      const gz = texture3DLevel(petroVolumeTex, fract(uvwG.add(vec3(0.0, 0.0, duG)).add(10.0)), float(0.0)).a;
      const gradGrit = rotLithicATranspose(
        vec3(gx.sub(hGrit_in), gy.sub(hGrit_in), gz.sub(hGrit_in)).div(duG.mul(2.6))
      );

      const gradCrys = rotLithicBTranspose(gradCrysA_in)
        .mul(0.85)
        .add(rotLithicATranspose(gradCrysB_in).mul(0.58));

      const gradLocal = gradMicro.mul(0.52).add(gradGrit.mul(0.42)).add(gradCrys);
      const bn = normalize(vec3(baseNorm_in));
      const gradTan = gradLocal.sub(bn.mul(dot(gradLocal, bn)));
      const clampedTan = gradTan.div(max(float(1.0), length(gradTan).div(1.45)));
      return normalize(bn.sub(clampedTan.mul(float(strength_in))));
    }
  );

  // -------------------------------------------------------------------------
  // Live 1536x512 Wet Yellow Ochre Trail DataTexture (2 tier rows x 6 faces)
  // -------------------------------------------------------------------------
  const wetTrailTexture = new THREE.DataTexture(
    wetAtlasData,
    WET_ATLAS_WIDTH,
    WET_ATLAS_HEIGHT,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  wetTrailTexture.colorSpace = THREE.NoColorSpace;
  wetTrailTexture.flipY = false;
  wetTrailTexture.minFilter = THREE.LinearFilter;
  wetTrailTexture.magFilter = THREE.LinearFilter;
  wetTrailTexture.wrapS = THREE.ClampToEdgeWrapping;
  wetTrailTexture.wrapT = THREE.ClampToEdgeWrapping;
  wetTrailTexture.generateMipmaps = false;
  wetTrailTexture.needsUpdate = true;

  const uSolveProgress = uniform(0.0);
  const uVictoryIgnition = uniform(0.0);
  const uCheckpointsUnlocked = uniform(0.0);
  const uDowsingHint = uniform(0.0);
  const uRiverFlow = uniform(0.0);

  // -------------------------------------------------------------------------
  // 1. Carved Stone Monolith Material (Rotated Lithic Limestone + Vesicular Basalt
  //    + Crystalline Verdigris Patina + Floor-Gated Wet Yellow Ochre Trail Glaze)
  // -------------------------------------------------------------------------
  const stoneMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.80,
    metalness: 0.02,
  });

  {
    const aMatZone = attribute('aMaterialZone', 'vec4'); // [limestone, basalt, creviceAo, patinaMask]
    const limestoneW = clamp(aMatZone.x, 0.0, 1.0);
    const basaltW = clamp(aMatZone.y, 0.0, 1.0);
    const creviceAo = clamp(aMatZone.z, 0.0, 1.0);
    const patinaInput = clamp(aMatZone.w, 0.0, 1.0);

    const aDetail = attribute('aStoneDetail', 'vec2'); // [floorMask, convexEdgeWear]
    const floorMask = clamp(aDetail.x, 0.0, 1.0);
    const convexEdgeWear = clamp(aDetail.y, 0.0, 1.0);

    // Strictly uniform control flow: explicit-LOD texture fetches unconditionally at shader entry
    const wetSample = texture(wetTrailTexture, uv()).level(float(0.0));
    const floorGate = smoothstep(float(0.08), float(0.55), floorMask);
    const wetness = clamp(wetSample.r, 0.0, 1.0).mul(floorGate);
    const freshSheen = clamp(wetSample.g, 0.0, 1.0).mul(floorGate);
    const ochreThickness = clamp(wetSample.b, 0.0, 1.0).mul(floorGate);
    const verdigrisMix = clamp(wetSample.a, 0.0, 1.0).mul(floorGate);

    const p = positionLocal;
    const uvwMacro = rotLithicA(p.mul(2.85));
    const uvwMicro = rotLithicB(p.mul(6.8)).add(vec3(0.37, 0.61, 0.19));
    const uvwGrit = rotLithicA(p.mul(19.5)).add(vec3(0.71, 0.13, 0.49));
    const uvwFlow = rotLithicA(p.mul(2.6)).add(
      vec3(stage.uTime.mul(0.28), stage.uTime.mul(-0.19), stage.uTime.mul(0.23))
    );
    const macroPetro = texture3DLevel(petroVolumeTex, fract(uvwMacro.add(10.0)), float(0.0));
    const microPetro = texture3DLevel(petroVolumeTex, fract(uvwMicro.add(10.0)), float(0.0));
    const gritPetro = texture3DLevel(petroVolumeTex, fract(uvwGrit.add(10.0)), float(0.0));
    const flowPetro = texture3DLevel(petroVolumeTex, fract(uvwFlow.add(10.0)), float(0.0));
    const broadTone = valueNoise3DTsl(rotLithicA(p.mul(3.2)));

    // Analytical infinite-resolution 3D crystalline mineral grain & micro-facets
    const pCrysA = rotLithicB(p.mul(34.0)).add(vec3(17.3, 29.1, 41.7));
    const pCrysB = rotLithicA(p.mul(78.0)).add(vec3(53.9, 11.7, 73.1));
    const crysA = valueNoise3DGradTsl(pCrysA);
    const crysB = valueNoise3DGradTsl(pCrysB);
    const crysTooth = crysA.x.mul(0.58).add(crysB.x.mul(0.42));

    // Keep raised walls/plateaus rugged while honing corridor floors for crisp route contrast
    const outerRockWeight = float(1.0).sub(floorGate.mul(0.58));
    const biotiteCluster = float(1.0).sub(macroPetro.r.mul(0.24));
    const biotiteFleck = smoothstep(
      float(0.64),
      float(0.83),
      crysA.x.mul(0.50).add(crysB.x.mul(0.50)).mul(biotiteCluster)
    );
    const quartzFleck = smoothstep(
      float(0.68),
      float(0.86),
      float(1.0).sub(crysA.x).mul(0.46).add(crysB.x.mul(0.54))
    );

    // -----------------------------------------------------------------------
    // Dressed Crystalline Stone (Mineral Clasts + Pin-Pores + Biotite/Quartz Tooth)
    // -----------------------------------------------------------------------
    const fineGrain = microPetro.b.mul(0.42).add(gritPetro.b.mul(0.28)).add(crysTooth.mul(0.30));
    const clastSignal = clamp(
      macroPetro.r.mul(0.34)
        .add(microPetro.r.mul(0.28))
        .add(fineGrain.mul(0.38))
        .add(broadTone.sub(0.5).mul(0.16)),
      0.06,
      0.96
    );
    const pinPore = clamp(microPetro.g.mul(0.72).add(macroPetro.g.mul(0.28)), 0.0, 1.0);
    const hairlineSeam = smoothstep(float(0.24), float(0.08), microPetro.r.mul(0.65).add(macroPetro.r.mul(0.35)));
    const calciteClast = smoothstep(float(0.74), float(0.91), microPetro.r.mul(0.64).add(macroPetro.r.mul(0.36)))
      .mul(float(0.52).add(fineGrain.mul(0.48)));

    const umberShadowBed = rawUmber.rInf.mul(0.60);
    const limestoneMixT = clamp(
      float(0.34)
        .add(clastSignal.mul(0.50))
        .sub(pinPore.mul(0.22))
        .sub(hairlineSeam.mul(0.18))
        .sub(biotiteFleck.mul(0.28).mul(outerRockWeight)),
      0.10,
      0.95
    );
    const warmMineralK = mix(
      mix(rawUmber.K, burntSienna.K, float(0.22).add(macroPetro.r.mul(0.18))),
      mix(warmKaolin.K, yellowOchre.K, float(0.24)),
      limestoneMixT
    );
    const warmMineralS = mix(
      mix(rawUmber.S, burntSienna.S, float(0.22).add(macroPetro.r.mul(0.18))),
      mix(warmKaolin.S, yellowOchre.S, float(0.24)),
      limestoneMixT
    );
    const limestoneDepth = clamp(
      float(1.02).add(clastSignal.sub(0.5).mul(0.62)).sub(pinPore.mul(0.48)),
      0.38,
      1.68
    );

    // -----------------------------------------------------------------------
    // Porous Charcoal Scoria / Basalt (island pockets, tier=1 vaults, core cleavage)
    // -----------------------------------------------------------------------
    const scoriaVesicle = clamp(macroPetro.g.mul(0.75).add(microPetro.g.mul(0.55)), 0.0, 1.0);
    const slateBacking = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    ).mul(0.68);
    const basaltDepth = float(0.34)
      .add(fineGrain.mul(0.24))
      .mul(float(1.0).sub(scoriaVesicle.mul(0.58)));

    // Unified Kubelka-Munk base pass for stone + basalt (call 1 of 3)
    const stoneK = mix(
      rawUmber.K.mul(1.75).add(azuritePrussian.K.mul(0.18)),
      warmMineralK,
      limestoneW
    );
    const stoneS = mix(rawUmber.S.mul(0.42), warmMineralS, limestoneW);
    const stoneDepth = mix(basaltDepth, limestoneDepth, limestoneW);
    const stoneBacking = mix(slateBacking, umberShadowBed, limestoneW);
    const stoneKmBase = kubelkaMunkGlaze(stoneK, stoneS, stoneDepth, stoneBacking);

    const arrisWearMask = convexEdgeWear.mul(float(0.60).add(fineGrain.mul(0.40)));
    const darkMicaBed = mix(rawUmber.rInf.mul(0.38), slateBacking.mul(1.25), float(0.28));
    const stoneWithPores = mix(
      stoneKmBase,
      darkMicaBed,
      clamp(
        hairlineSeam.mul(0.30)
          .add(pinPore.mul(0.34))
          .add(biotiteFleck.mul(0.34).mul(outerRockWeight))
          .add(float(0.5).sub(gritPetro.b).mul(0.14)),
        0.0,
        0.54
      ).mul(limestoneW)
    );
    const basaltSeptaRim = smoothstep(float(0.56), float(0.84), fineGrain)
      .mul(float(1.0).sub(scoriaVesicle));
    const paleCalcite = mix(warmKaolin.rInf, boneCalcite.rInf, float(0.72));
    const limestoneHighlight = clamp(
      arrisWearMask.mul(0.30).add(calciteClast.mul(0.24)).add(quartzFleck.mul(0.26).mul(outerRockWeight)),
      0.0,
      0.54
    ).mul(limestoneW);
    const stoneBase = mix(stoneWithPores, paleCalcite, limestoneHighlight)
      .add(warmKaolin.rInf.mul(basaltSeptaRim.mul(0.06).mul(basaltW)));

    const cavityDepth = clamp(
      pinPore.mul(0.72)
        .add(hairlineSeam.mul(0.42))
        .add(smoothstep(float(0.44), float(0.14), crysTooth).mul(0.34)),
      0.0,
      1.0
    );
    const cavityOcc = float(1.0).sub(cavityDepth.mul(0.42).mul(outerRockWeight));
    const aoFactor = mix(float(0.46), float(1.04), creviceAo);

    // Compute wet mask first so live liquid trail suppresses older copper patina & floor tint underneath
    const wetMask = smoothstep(
      float(0.02),
      float(0.20),
      max(max(wetness, ochreThickness), verdigrisMix)
    );
    const honedFloorContrast = mix(float(1.0), float(0.84), floorGate.mul(float(1.0).sub(wetMask)));
    const stoneWithAo = stoneBase.mul(aoFactor).mul(cavityOcc).mul(honedFloorContrast);

    // Soft dusty Verdigris + Celadon + Kaolin mineral crust tucked strictly in recessed seams
    const patinaTex = macroPetro.b.mul(0.55).add(microPetro.b.mul(0.45));
    const patinaSignal = patinaInput
      .mul(float(0.42).add(patinaTex.mul(0.86)))
      .add(fineGrain.sub(0.5).mul(0.20));
    const patinaCoverage = smoothstep(float(0.36), float(0.82), patinaSignal);
    const patinaFinal = patinaCoverage.mul(float(1.0).sub(wetMask.mul(0.92)));
    const patinaK = verdigrisMalachite.K.mul(0.55)
      .add(celadonGlauconite.K.mul(0.30))
      .add(warmKaolin.K.mul(0.15));
    const patinaS = verdigrisMalachite.S.mul(0.55)
      .add(celadonGlauconite.S.mul(0.30))
      .add(warmKaolin.S.mul(0.15));
    // Patina glaze (call 2 of 3)
    const stoneWithPatina = kubelkaMunkGlaze(
      patinaK,
      patinaS,
      patinaFinal.mul(0.72),
      stoneWithAo
    );

    // Floor-gated glossy wet trail glaze (warm Yellow Ochre -> subtle Celadon whisper -> full Verdigris Malachite)
    const greenFraction = clamp(
      verdigrisMix.div(max(float(0.02), max(ochreThickness, verdigrisMix))),
      0.0,
      1.0
    );
    const flowCrest = smoothstep(
      float(0.30),
      float(0.76),
      flowPetro.r.mul(0.64).add(flowPetro.b.mul(0.36))
    );
    const malachiteCeladonT = mix(
      float(0.34),
      float(0.20).add(flowCrest.mul(0.58)),
      uRiverFlow
    );
    const verdigrisRiverK = mix(verdigrisMalachite.K, celadonGlauconite.K, malachiteCeladonT);
    const verdigrisRiverS = mix(verdigrisMalachite.S, celadonGlauconite.S, malachiteCeladonT);
    const targetGreenK = mix(
      celadonGlauconite.K,
      verdigrisRiverK,
      smoothstep(float(0.25), float(0.75), greenFraction)
    );
    const targetGreenS = mix(
      celadonGlauconite.S,
      verdigrisRiverS,
      smoothstep(float(0.25), float(0.75), greenFraction)
    );

    const wetDarkenedBacking = stoneWithPatina.mul(mix(float(1.0), float(0.82), wetMask));
    const ochreK = mix(yellowOchre.K, realgarAmber.K, float(0.28).add(freshSheen.mul(0.14)));
    const ochreS = mix(yellowOchre.S, realgarAmber.S, float(0.28).add(freshSheen.mul(0.14)));
    const trailK = mix(ochreK, targetGreenK, greenFraction);
    const trailS = mix(ochreS, targetGreenS, greenFraction);
    const trailOpticalDepth = ochreThickness
      .mul(1.45)
      .add(wetness.mul(0.50))
      .add(verdigrisMix.mul(0.35));
    const finalStoneRgb = kubelkaMunkGlaze(
      trailK,
      trailS,
      trailOpticalDepth,
      wetDarkenedBacking
    );

    stoneMaterial.colorNode = clamp(finalStoneRgb, 0.0, 1.0);

    // Matte porous stone matrix (0.86-0.97) with satin mineral facets (0.54) -> wet trail (0.11)
    const matrixRough = clamp(
      mix(float(0.86), float(0.92), basaltW)
        .add(pinPore.mul(0.06))
        .add(biotiteFleck.mul(0.04))
        .sub(fineGrain.mul(0.05))
        .sub(arrisWearMask.mul(0.05)),
      0.74,
      0.97
    );
    const dryRoughness = clamp(
      mix(matrixRough, float(0.54), quartzFleck.mul(outerRockWeight).mul(0.68)),
      0.52,
      0.97
    );
    const wetGlossFactor = clamp(
      wetness
        .mul(1.20)
        .add(freshSheen.mul(0.32))
        .add(verdigrisMix.mul(uRiverFlow).mul(0.85))
        .add(verdigrisMix.mul(0.28)),
      0.0,
      1.0
    );
    stoneMaterial.roughnessNode = clamp(
      mix(dryRoughness, float(0.11), wetGlossFactor),
      0.10,
      0.97
    );
    stoneMaterial.metalnessNode = mix(float(0.02), float(0.08), wetGlossFactor);

    // Tri-band tangent-projected rock relief: crisp walls/plateaus, honed corridor beds, smooth wet meniscus
    const bumpAmp = mix(float(0.42), float(0.58), basaltW)
      .add(patinaFinal.mul(0.06))
      .mul(mix(float(1.0), float(0.34), floorGate))
      .mul(float(1.0).sub(wetGlossFactor.mul(0.90)));
    const perturbedLocalNorm = perturbStoneNormalLocal(
      normalLocal,
      uvwMicro,
      microPetro.a,
      uvwGrit,
      gritPetro.a,
      crysA.yzw,
      crysB.yzw,
      bumpAmp
    );
    stoneMaterial.normalNode = transformNormalToView(perturbedLocalNorm);

    // Soft lustrous studio sheen along wet meniscus + flowing malachite/celadon river on solve
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const keySoftboxDir = normalize(vec3(0.6, 4.4, 4.2));
    const halfVec = normalize(keySoftboxDir.add(viewDirW));
    const meniscusSpec = pow(max(float(0.0), dot(nW, halfVec)), float(24.0));
    const meniscusFresnel = pow(abs(float(1.0).sub(clamp(dot(nW, viewDirW), 0.0, 1.0))), float(3.2));
    const meniscusPigment = mix(
      mix(yellowOchre.rInf, warmKaolin.rInf, float(0.28)),
      mix(verdigrisMalachite.rInf, celadonGlauconite.rInf, float(0.45)),
      greenFraction
    );
    const meniscusGlow = meniscusPigment
      .mul(meniscusSpec.mul(0.22).add(meniscusFresnel.mul(0.10)))
      .mul(wetGlossFactor)
      .mul(float(0.55).add(freshSheen.mul(0.45)).add(verdigrisMix.mul(0.35)));
    const riverMask = smoothstep(float(0.42), float(0.85), verdigrisMix).mul(greenFraction).mul(uRiverFlow);
    const riverGlow = mix(
      verdigrisMalachite.rInf,
      mix(celadonGlauconite.rInf, boneCalcite.rInf, float(0.22)),
      flowCrest
    )
      .mul(float(0.26).add(flowCrest.mul(0.28)).add(uVictoryIgnition.mul(0.42)))
      .mul(riverMask);
    const ignitionGlow = mix(realgarAmber.rInf, yellowOchre.rInf, float(0.42))
      .mul(uVictoryIgnition)
      .mul(0.60)
      .mul(wetMask)
      .mul(float(1.0).sub(greenFraction));
    stoneMaterial.emissiveNode = meniscusGlow.add(riverGlow).add(ignitionGlow);
  }

  // -------------------------------------------------------------------------
  // 2. Goal Sanctum Material (Bone Calcite Collar + 3-Sector Malachite Seal + Cinnabar Cup)
  // -------------------------------------------------------------------------
  const sanctumMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.68,
    metalness: 0.02,
  });

  {
    const aSanctum = attribute('aSanctumZone', 'vec2'); // [calciteWeight, cinnabarWeight]
    const calciteW = clamp(aSanctum.x, 0.0, 1.0);
    const cinnabarW = clamp(aSanctum.y, 0.0, 1.0);
    const normSum = max(float(1e-4), calciteW.add(cinnabarW));
    const cupBlend = smoothstep(float(0.28), float(0.72), clamp(cinnabarW.div(normSum), 0.0, 1.0));

    const p = positionLocal;
    const uvwSanctum = rotLithicB(p.mul(6.1)).add(vec3(0.21, 0.43, 0.77));
    const sanctumPetro = texture3DLevel(petroVolumeTex, fract(uvwSanctum.add(10.0)), float(0.0));
    const grain = sanctumPetro.b;
    const microPit = smoothstep(float(0.45), float(0.85), sanctumPetro.g);

    const calciteCollarBase = kubelkaMunkGlaze(
      boneCalcite.K,
      boneCalcite.S,
      float(0.88).add(grain.mul(0.32)).sub(microPit.mul(0.24)),
      warmKaolin.rInf.mul(0.76)
    );

    // 3-Sector Malachite Seal Recessed Channel on the Bone Calcite Collar (smooth angular dot masks)
    const relXZ = vec2(p.x.sub(float(goalLocalPos[0])), p.z.sub(float(goalLocalPos[2])));
    const rCollar = length(relXZ);
    const dirXZ = normalize(relXZ.add(vec2(1e-5, 0.0)));
    const collarBand = smoothstep(float(0.057), float(0.062), rCollar)
      .mul(float(1.0).sub(smoothstep(float(0.074), float(0.079), rCollar)))
      .mul(float(1.0).sub(cupBlend));
    const sec0 = smoothstep(float(0.535), float(0.585), dot(dirXZ, vec2(0.0, -1.0)));
    const sec1 = smoothstep(float(0.535), float(0.585), dot(dirXZ, vec2(0.8660254, 0.5)));
    const sec2 = smoothstep(float(0.535), float(0.585), dot(dirXZ, vec2(-0.8660254, 0.5)));
    const lit0 = clamp(uCheckpointsUnlocked, 0.0, 1.0);
    const lit1 = clamp(uCheckpointsUnlocked.sub(1.0), 0.0, 1.0);
    const lit2 = clamp(uCheckpointsUnlocked.sub(2.0), 0.0, 1.0);
    const anySec = clamp(sec0.add(sec1).add(sec2), 0.0, 1.0).mul(collarBand);
    const litSec = clamp(sec0.mul(lit0).add(sec1.mul(lit1)).add(sec2.mul(lit2)), 0.0, 1.0).mul(
      collarBand
    );

    const unlitSlotRgb = mix(calciteCollarBase, rawUmber.rInf.mul(0.46), anySec.mul(0.58));
    const sealGlazeRgb = kubelkaMunkGlaze(
      mix(verdigrisMalachite.K, celadonGlauconite.K, float(0.32)),
      mix(verdigrisMalachite.S, celadonGlauconite.S, float(0.32)),
      float(1.35).add(grain.mul(0.28)),
      rawUmber.rInf.mul(0.32)
    );
    const calciteCollar = mix(unlitSlotRgb, sealGlazeRgb, litSec);

    const mortarBacking = mix(burntSienna.rInf, rawUmber.rInf, float(0.32)).mul(0.52);
    const cinnabarBowl = kubelkaMunkGlaze(
      mix(cinnabarVermilion.K, realgarAmber.K, grain.mul(0.16)),
      mix(cinnabarVermilion.S, realgarAmber.S, grain.mul(0.16)),
      float(1.15).add(grain.mul(0.45)),
      mortarBacking
    );

    sanctumMaterial.colorNode = clamp(mix(calciteCollar, cinnabarBowl, cupBlend), 0.0, 1.0);
    sanctumMaterial.roughnessNode = clamp(
      mix(mix(float(0.76), float(0.54), cupBlend), float(0.18), litSec).add(grain.mul(0.06)),
      0.16,
      0.88
    );
    sanctumMaterial.normalNode = transformNormalToView(
      perturbPetroNormalLocal(normalLocal, uvwSanctum, sanctumPetro.a, float(0.065))
    );
    const cupEmissive = mix(cinnabarVermilion.rInf, realgarAmber.rInf, float(0.48))
      .mul(cupBlend)
      .mul(uSolveProgress.mul(0.14).add(uVictoryIgnition.mul(0.45)));
    const sealEmissive = mix(verdigrisMalachite.rInf, celadonGlauconite.rInf, float(0.38))
      .mul(litSec)
      .mul(float(0.24).add(uVictoryIgnition.mul(0.45)));
    sanctumMaterial.emissiveNode = cupEmissive.add(sealEmissive);
  }

  // -------------------------------------------------------------------------
  // 3. Start Spring Socket Material (Basaltic Housing + Realgar/Ochre Rim)
  // -------------------------------------------------------------------------
  const startSpringMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.56,
    metalness: 0.08,
  });

  {
    const aStart = attribute('aStartZone', 'vec2'); // [basaltWeight, amberOchreWeight]
    const basaltW = clamp(aStart.x, 0.0, 1.0);
    const amberW = clamp(aStart.y, 0.0, 1.0);
    const normSum = max(float(1e-4), basaltW.add(amberW));
    const rimBlend = clamp(amberW.div(normSum), 0.0, 1.0);

    const p = positionLocal;
    const uvwStart = rotLithicB(p.mul(6.4)).add(vec3(0.59, 0.17, 0.83));
    const startPetro = texture3DLevel(petroVolumeTex, fract(uvwStart.add(10.0)), float(0.0));
    const grain = startPetro.b;
    const slateBacking = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    ).mul(0.9);
    const basaltSocket = kubelkaMunkGlaze(
      rawUmber.K.mul(1.35),
      rawUmber.S.mul(0.52),
      float(0.28).add(grain.mul(0.25)),
      slateBacking
    );

    const springOchre = kubelkaMunkGlaze(
      mix(yellowOchre.K, realgarAmber.K, float(0.44)),
      mix(yellowOchre.S, realgarAmber.S, float(0.44)),
      float(1.10).add(grain.mul(0.35)),
      rawUmber.rInf.mul(0.45)
    );
    const springWithCalcite = mix(
      springOchre,
      boneCalcite.rInf.mul(0.9),
      smoothstep(float(0.76), float(0.94), startPetro.r).mul(0.22)
    );

    startSpringMaterial.colorNode = clamp(mix(basaltSocket, springWithCalcite, rimBlend), 0.0, 1.0);
    startSpringMaterial.roughnessNode = clamp(
      mix(float(0.84), float(0.24), rimBlend).add(grain.mul(0.06)),
      0.18,
      0.92
    );
    startSpringMaterial.normalNode = transformNormalToView(
      perturbPetroNormalLocal(normalLocal, uvwStart, startPetro.a, mix(float(0.115), float(0.045), rimBlend))
    );
  }

  // -------------------------------------------------------------------------
  // 3b. One-Way Checkpoint Waystone Inlay Material (Bronze Rim + Malachite Seal Basin)
  // -------------------------------------------------------------------------
  const waystoneMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.42,
    metalness: 0.12,
  });

  {
    const aWaystone = attribute('aWaystoneZone', 'vec3'); // [rimW, basinW, cpIdx]
    const rimW = clamp(aWaystone.x, 0.0, 1.0);
    const basinW = clamp(aWaystone.y, 0.0, 1.0);
    const cpIdx = aWaystone.z;
    const normSum = max(float(1e-4), rimW.add(basinW));
    const basinMask = smoothstep(float(0.25), float(0.75), clamp(basinW.div(normSum), 0.0, 1.0));
    const rimMask = float(1.0).sub(basinMask);
    const litW = clamp(uCheckpointsUnlocked.sub(cpIdx), 0.0, 1.0);

    const p = positionLocal;
    const uvwWay = rotLithicB(p.mul(7.2)).add(vec3(0.41, 0.67, 0.29));
    const wayPetro = texture3DLevel(petroVolumeTex, fract(uvwWay.add(10.0)), float(0.0));
    const grain = wayPetro.b;

    const rimBronze = mix(
      mix(rawUmber.rInf.mul(0.74), yellowOchre.rInf, float(0.62)),
      verdigrisMalachite.rInf.mul(0.78),
      float(0.12).add(litW.mul(0.20))
    );
    const unlitBasin = kubelkaMunkGlaze(
      mix(rawUmber.K, verdigrisMalachite.K, float(0.42)),
      mix(rawUmber.S, verdigrisMalachite.S, float(0.42)),
      float(0.82).add(grain.mul(0.22)),
      rawUmber.rInf.mul(0.32)
    );
    const litBasin = kubelkaMunkGlaze(
      mix(verdigrisMalachite.K, celadonGlauconite.K, float(0.36).add(grain.mul(0.18))),
      mix(verdigrisMalachite.S, celadonGlauconite.S, float(0.36).add(grain.mul(0.18))),
      float(1.40).add(grain.mul(0.28)),
      verdigrisMalachite.rInf.mul(0.45)
    );
    const basinColor = mix(unlitBasin, litBasin, litW);

    waystoneMaterial.colorNode = clamp(mix(rimBronze, basinColor, basinMask), 0.0, 1.0);
    waystoneMaterial.metalnessNode = mix(float(0.84), float(0.06), basinMask);
    waystoneMaterial.roughnessNode = clamp(
      mix(float(0.26).add(grain.mul(0.08)), mix(float(0.56), float(0.16), litW), basinMask),
      0.14,
      0.78
    );
    waystoneMaterial.normalNode = transformNormalToView(
      perturbPetroNormalLocal(normalLocal, uvwWay, wayPetro.a, mix(float(0.04), float(0.06), rimMask))
    );
    waystoneMaterial.emissiveNode = mix(
      verdigrisMalachite.rInf,
      celadonGlauconite.rInf,
      float(0.42)
    )
      .mul(basinMask)
      .mul(litW.mul(0.24).add(uRiverFlow.mul(0.34)));
  }

  // -------------------------------------------------------------------------
  // 4. Liquid Quicksilver Bead Material (Weber Stretch, Sessile Contact
  //    Flattening, Damped Rayleigh-Lamb l=2 Oscillation & Studio Softbox Dome)
  // -------------------------------------------------------------------------
  const uBeadVelocityLocal = uniform(new THREE.Vector3(0, 0, 0));
  const uBeadNormalLocal = uniform(new THREE.Vector3(0, 1, 0));
  const uBeadImpact = uniform(0.0);
  const uBeadRollQuat = uniform(new THREE.Vector4(0, 0, 0, 1));
  const uBeadSloshLocal = uniform(new THREE.Vector3(0, 0, 0));

  const beadRadiusFloat = float(beadRadius);

  const deformBeadVertex = Fn(([pt_in]: [TslNode]) => {
    const p0 = vec3(pt_in);
    const nFloor = normalize(uBeadNormalLocal.add(vec3(0.0, 1e-4, 0.0)));

    // 1. Sessile capillary floor-contact flattening along uBeadNormalLocal
    const h0 = dot(p0, nFloor);
    const pRad = p0.sub(nFloor.mul(h0));
    const hNorm = clamp(h0.div(beadRadiusFloat), -1.0, 1.0);
    const botMask = clamp(hNorm.negate(), 0.0, 1.0);
    // Shift center -0.13*R toward floor while pushing bottom pole +0.13*R so contact remains exact at -R
    const hFlat = h0
      .sub(beadRadiusFloat.mul(0.13))
      .add(beadRadiusFloat.mul(0.13).mul(pow(botMask, float(2.1))));
    const bellyFlare = exp(pow(hNorm.add(0.34), float(2.0)).negate().mul(3.4)).mul(0.085);
    const pSessile = pRad.mul(float(1.0).add(bellyFlare)).add(nFloor.mul(hFlat));

    // 2. Volume-preserving Weber velocity stretch along uBeadVelocityLocal
    const velMag = length(uBeadVelocityLocal);
    const velDir = uBeadVelocityLocal.div(max(velMag, float(1e-4)));
    const stretch = clamp(velMag.mul(0.11), 0.0, 0.24);
    const parallelLen = dot(pSessile, velDir);
    const pParallel = velDir.mul(parallelLen);
    const pPerp = pSessile.sub(pParallel);
    const perpScale = float(1.0).div(sqrt(float(1.0).add(stretch)));
    const pStretched = pParallel.mul(float(1.0).add(stretch)).add(pPerp.mul(perpScale));

    // 2b. Tangent-plane inertial capillary dome rock (floor contact pinned at hNorm = -1)
    const domeRockMask = smoothstep(float(-0.85), float(0.65), hNorm);
    const sloshTan = uBeadSloshLocal.sub(nFloor.mul(dot(uBeadSloshLocal, nFloor)));
    const pRocked = pStretched.add(sloshTan.mul(beadRadiusFloat.mul(domeRockMask)));

    // 3. Damped Rayleigh-Lamb l=2 axisymmetric oscillation P2(cos theta) from uBeadImpact
    const pUnit = normalize(p0.add(vec3(0.0, 1e-5, 0.0)));
    const cosTheta = dot(pUnit, nFloor);
    const p2Mode = float(0.5).mul(float(3.0).mul(cosTheta).mul(cosTheta).sub(1.0));
    const lambOsc = uBeadImpact.mul(sin(stage.uTime.mul(32.0))).mul(0.15);
    return pRocked.add(pUnit.mul(beadRadiusFloat.mul(lambOsc).mul(p2Mode)));
  });

  const computeBeadDeformedNormal = Fn(([pt_in, norm_in]: [TslNode, TslNode]) => {
    const p0 = vec3(pt_in);
    const n0 = normalize(vec3(norm_in));
    const upFallback = select(
      abs(n0.y).lessThan(0.92),
      vec3(0.0, 1.0, 0.0),
      vec3(1.0, 0.0, 0.0)
    );
    const t1 = normalize(cross(upFallback, n0));
    const t2 = normalize(cross(n0, t1));
    const eps = beadRadiusFloat.mul(0.035);

    const d0 = deformBeadVertex(p0);
    const d1 = deformBeadVertex(p0.add(t1.mul(eps))).sub(d0);
    const d2 = deformBeadVertex(p0.add(t2.mul(eps))).sub(d0);
    return normalize(cross(d1, d2));
  });

  const beadMaterial = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.97,
    roughness: 0.055,
    clearcoat: 0.35,
    clearcoatRoughness: 0.04,
  });

  {
    beadMaterial.positionNode = deformBeadVertex(positionGeometry);
    const deformedNormalVarying = computeBeadDeformedNormal(
      positionGeometry,
      normalGeometry
    ).toVarying('vBeadDeformedNormalLocal');
    beadMaterial.normalNode = transformNormalToView(normalize(deformedNormalVarying));

    // Rotate sphere coordinates into bead-intrinsic frame via conjugate of uBeadRollQuat so no-slip spin reads cleanly
    const pBeadUnit = normalize(positionGeometry);
    const qConj = uBeadRollQuat.xyz.negate();
    const tCross = cross(qConj, pBeadUnit).mul(2.0);
    const pRolled = normalize(
      pBeadUnit.add(tCross.mul(uBeadRollQuat.w)).add(cross(qConj, tCross))
    );
    const uvwBeadRoll = pRolled.mul(0.55).add(vec3(0.37, 0.71, 0.19));
    const beadMicro = texture3DLevel(petroVolumeTex, fract(uvwBeadRoll.add(10.0)), float(0.0));
    const rollGrain = beadMicro.r.mul(0.58).add(beadMicro.g.mul(0.42));
    const rollMicroBand = smoothstep(float(0.28), float(0.76), rollGrain);
    const rollLuster = mix(float(0.94), float(1.06), rollMicroBand);

    // Silver-quicksilver metallic albedo anchored strictly in boneCalcite neutrality
    beadMaterial.colorNode = boneCalcite.rInf.mul(1.04).mul(mix(float(0.97), float(1.03), rollMicroBand));
    beadMaterial.roughnessNode = mix(float(0.042), float(0.115), rollMicroBand);

    // Procedural Studio Softbox Tent & Ochre/Cinnabar Trench Reflection Sheen
    const V = normalize(cameraPosition.sub(positionWorld));
    const N = normalize(normalWorld);
    const R = normalize(V.negate().sub(N.mul(dot(V.negate(), N).mul(2.0))));
    const ndotv = clamp(dot(N, V), 0.0, 1.0);
    const mercuryFresnel = float(0.86).add(
      float(0.14).mul(pow(abs(float(1.0).sub(ndotv)), float(5.0)))
    );

    // Deep workshop horizon + warm limestone/ochre channel bounce (+ cinnabar sanctum glow at solve)
    const darkHorizon = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    ).mul(0.28);
    const lowerTrenchBase = mix(
      mix(rawUmber.rInf.mul(0.26), yellowOchre.rInf.mul(0.48), smoothstep(float(-0.78), float(-0.08), R.y)),
      cinnabarVermilion.rInf.mul(0.58),
      uSolveProgress.mul(0.75)
    );
    const lowerTrenchWithDowsing = mix(
      lowerTrenchBase,
      celadonGlauconite.rInf.mul(0.48),
      uDowsingHint.mul(0.16)
    );
    const lowerTrenchTint = mix(
      lowerTrenchWithDowsing,
      verdigrisMalachite.rInf.mul(0.54),
      uRiverFlow.mul(0.38)
    );
    const lowerDome = mix(
      lowerTrenchTint,
      darkHorizon,
      smoothstep(float(-0.18), float(0.16), R.y)
    );

    // Primary rectangular studio overhead softbox aligned toward key light (0.6, 4.4, 4.2)
    const keyDir = normalize(vec3(0.6, 4.4, 4.2));
    const keyRight = normalize(cross(vec3(0.0, 1.0, 0.0), keyDir));
    const keyUp = cross(keyDir, keyRight);
    const rx = dot(R, keyRight);
    const ry = dot(R, keyUp);
    const rz = dot(R, keyDir);
    const softboxMain = smoothstep(float(0.18), float(0.46), rz)
      .mul(float(1.0).sub(smoothstep(float(0.24), float(0.44), abs(rx))))
      .mul(float(1.0).sub(smoothstep(float(0.18), float(0.35), abs(ry))));

    // Secondary zenith overhead strip & cool azurite-slate fill window
    const zenithStrip = smoothstep(float(0.68), float(0.90), R.y).mul(0.58);
    const leftFillDir = normalize(vec3(-3.4, 0.4, 2.0));
    const leftFillStrip = smoothstep(float(0.72), float(0.95), dot(R, leftFillDir)).mul(0.34);

    // Grazing back-top calcite rim strip separating sphere silhouette cleanly
    const rimDir = normalize(vec3(-1.5, 3.2, -3.6));
    const backRimStrip = smoothstep(float(0.66), float(0.92), dot(R, rimDir)).mul(0.76);

    const softboxColor = mix(boneCalcite.rInf, warmKaolin.rInf, float(0.35)).mul(1.08);
    const coolSlateColor = mix(
      vec3(CHALKY_SLATE_LINEAR[0], CHALKY_SLATE_LINEAR[1], CHALKY_SLATE_LINEAR[2]).mul(4.8),
      azuritePrussian.rInf.mul(2.4),
      float(0.32)
    );

    const studioEnv = lowerDome
      .add(softboxColor.mul(softboxMain.mul(1.18).add(zenithStrip).add(backRimStrip)))
      .add(coolSlateColor.mul(leftFillStrip));

    beadMaterial.emissiveNode = studioEnv.mul(mercuryFresnel).mul(0.90).mul(rollLuster);
  }

  // -------------------------------------------------------------------------
  // Zero-Allocation Bead Uniform Updater with Damped Rayleigh-Lamb & Dome Slosh
  // -------------------------------------------------------------------------
  let ringAmplitude = 0;
  let prevRawImpact = 0;
  let sloshX = 0;
  let sloshY = 0;
  let sloshZ = 0;
  let sloshVx = 0;
  let sloshVy = 0;
  let sloshVz = 0;
  let prevVelX = 0;
  let prevVelY = 0;
  let prevVelZ = 0;
  let hasPrevVel = false;

  const updateBeadUniforms: LabyrinthOpticsBundle['updateBeadUniforms'] = (
    velocityLocal,
    normalLocalVec,
    impactImpulse,
    solveProgress01 = 0,
    dtSec = 1 / 60,
    rollQuat
  ) => {
    let vx: number;
    let vy: number;
    let vz: number;
    if (Array.isArray(velocityLocal)) {
      vx = velocityLocal[0];
      vy = velocityLocal[1];
      vz = velocityLocal[2];
    } else {
      const v = velocityLocal as THREE.Vector3;
      vx = v.x;
      vy = v.y;
      vz = v.z;
    }
    uBeadVelocityLocal.value.set(vx, vy, vz);

    let nx: number;
    let ny: number;
    let nz: number;
    if (Array.isArray(normalLocalVec)) {
      nx = normalLocalVec[0];
      ny = normalLocalVec[1];
      nz = normalLocalVec[2];
    } else {
      const n = normalLocalVec as THREE.Vector3;
      nx = n.x;
      ny = n.y;
      nz = n.z;
    }
    uBeadNormalLocal.value.set(nx, ny, nz);
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;

    if (dtSec <= 0) {
      ringAmplitude = 0;
      prevRawImpact = 0;
      sloshX = 0;
      sloshY = 0;
      sloshZ = 0;
      sloshVx = 0;
      sloshVy = 0;
      sloshVz = 0;
      prevVelX = vx;
      prevVelY = vy;
      prevVelZ = vz;
      hasPrevVel = false;
      uBeadImpact.value = 0;
      uBeadSloshLocal.value.set(0, 0, 0);
      uSolveProgress.value = Math.max(0, Math.min(1, solveProgress01));
    } else {
      const dt = Math.max(0.0005, Math.min(0.05, dtSec));

      const dvx = vx - prevVelX;
      const dvy = vy - prevVelY;
      const dvz = vz - prevVelZ;
      const dvMag = Math.hypot(dvx, dvy, dvz);

      let axTan = 0;
      let ayTan = 0;
      let azTan = 0;
      if (!hasPrevVel || dvMag > 3.5) {
        hasPrevVel = true;
      } else {
        const ax = -dvx / dt;
        const ay = -dvy / dt;
        const az = -dvz / dt;
        const aDotN = ax * nx + ay * ny + az * nz;
        axTan = ax - aDotN * nx;
        ayTan = ay - aDotN * ny;
        azTan = az - aDotN * nz;
        const aTanMag = Math.hypot(axTan, ayTan, azTan);
        if (aTanMag > 28.0) {
          const s = 28.0 / aTanMag;
          axTan *= s;
          ayTan *= s;
          azTan *= s;
        }
      }
      prevVelX = vx;
      prevVelY = vy;
      prevVelZ = vz;

      // 2nd-order underdamped capillary-gravity slosh (omega0 = 26 rad/s, 2*zeta*omega0 = 14.5 / s)
      const sloshSteps = Math.max(1, Math.ceil(dt / (1 / 120)));
      const subDt = dt / sloshSteps;
      const omega0Sq = 676.0;
      const damping = 14.5;
      const driveGain = 5.5;
      for (let s = 0; s < sloshSteps; s++) {
        sloshVx += (driveGain * axTan - omega0Sq * sloshX - damping * sloshVx) * subDt;
        sloshVy += (driveGain * ayTan - omega0Sq * sloshY - damping * sloshVy) * subDt;
        sloshVz += (driveGain * azTan - omega0Sq * sloshZ - damping * sloshVz) * subDt;
        sloshX += sloshVx * subDt;
        sloshY += sloshVy * subDt;
        sloshZ += sloshVz * subDt;
      }

      const sDotN = sloshX * nx + sloshY * ny + sloshZ * nz;
      sloshX -= sDotN * nx;
      sloshY -= sDotN * ny;
      sloshZ -= sDotN * nz;
      const svDotN = sloshVx * nx + sloshVy * ny + sloshVz * nz;
      sloshVx -= svDotN * nx;
      sloshVy -= svDotN * ny;
      sloshVz -= svDotN * nz;

      const sloshMag = Math.hypot(sloshX, sloshY, sloshZ);
      if (sloshMag > 0.22) {
        const s = 0.22 / sloshMag;
        sloshX *= s;
        sloshY *= s;
        sloshZ *= s;
      }
      const sloshVMag = Math.hypot(sloshVx, sloshVy, sloshVz);
      const vMag = Math.hypot(vx, vy, vz);
      if (sloshMag < 1e-4 && sloshVMag < 1e-3 && vMag < 1e-3) {
        sloshX = 0;
        sloshY = 0;
        sloshZ = 0;
        sloshVx = 0;
        sloshVy = 0;
        sloshVz = 0;
      }
      uBeadSloshLocal.value.set(sloshX, sloshY, sloshZ);

      const clampedImpact = Math.max(0, Math.min(1.5, impactImpulse));
      if (clampedImpact > prevRawImpact + 0.02) {
        ringAmplitude = Math.max(ringAmplitude, clampedImpact);
      } else {
        ringAmplitude = Math.max(ringAmplitude * Math.exp(-dt * 5.2), clampedImpact);
      }
      if (ringAmplitude < 1e-4) ringAmplitude = 0;
      prevRawImpact = clampedImpact;

      uBeadImpact.value = ringAmplitude;
      uSolveProgress.value = Math.max(0, Math.min(1, solveProgress01));
    }

    if (rollQuat !== undefined) {
      if (Array.isArray(rollQuat)) {
        uBeadRollQuat.value.set(rollQuat[0], rollQuat[1], rollQuat[2], rollQuat[3]);
      } else {
        const rq = rollQuat as THREE.Quaternion;
        uBeadRollQuat.value.set(rq.x, rq.y, rq.z, rq.w);
      }
    } else {
      uBeadRollQuat.value.set(0, 0, 0, 1);
    }
  };

  const getBeadSloshLocal: LabyrinthOpticsBundle['getBeadSloshLocal'] = (out) => {
    out[0] = uBeadSloshLocal.value.x;
    out[1] = uBeadSloshLocal.value.y;
    out[2] = uBeadSloshLocal.value.z;
    return out;
  };

  const setProgressUniforms: LabyrinthOpticsBundle['setProgressUniforms'] = (
    checkpointsUnlocked,
    dowsingHint01
  ) => {
    uCheckpointsUnlocked.value = Math.max(0, Math.min(3, checkpointsUnlocked));
    uDowsingHint.value = Math.max(0, Math.min(1, dowsingHint01));
  };

  const setVictoryUniforms: LabyrinthOpticsBundle['setVictoryUniforms'] = (
    solveProgress01,
    victoryIgnition01,
    riverFlow01 = 0
  ) => {
    uSolveProgress.value = Math.max(0, Math.min(1, solveProgress01));
    uVictoryIgnition.value = Math.max(0, Math.min(1, victoryIgnition01));
    uRiverFlow.value = Math.max(0, Math.min(1, riverFlow01));
  };

  return {
    stoneMaterial,
    sanctumMaterial,
    startSpringMaterial,
    waystoneMaterial,
    beadMaterial,
    wetTrailTexture,
    updateBeadUniforms,
    getBeadSloshLocal,
    setProgressUniforms,
    setVictoryUniforms,
    dispose: () => {
      petroVolumeTex.dispose();
      wetTrailTexture.dispose();
      stoneMaterial.dispose();
      sanctumMaterial.dispose();
      startSpringMaterial.dispose();
      waystoneMaterial.dispose();
      beadMaterial.dispose();
    },
  };
}
