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
  vec3,
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
  beadMaterial: THREE.MeshPhysicalNodeMaterial;
  wetTrailTexture: THREE.DataTexture;
  updateBeadUniforms: (
    velocityLocal: readonly [number, number, number] | THREE.Vector3,
    normalLocalVec: readonly [number, number, number] | THREE.Vector3,
    impactImpulse: number,
    solveProgress01?: number,
    dtSec?: number
  ) => void;
  setVictoryUniforms: (solveProgress01: number, victoryIgnition01: number) => void;
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
 * Builds all four physical Kubelka-Munk / liquid-metal materials and pre-allocated
 * GPU uniform & wet-trail texture bindings for Day 13 — Labyrinth.
 */
export function createLabyrinthOptics(
  stage: SpecimenStage,
  wetAtlasData: Uint8Array,
  beadRadius = 0.046
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
   * 3-tap forward-difference 3D volume relief normal perturbation strictly
   * evaluated in uniform control flow against the pre-baked 64^3 Data3DTexture.
   */
  const perturbPetroNormalLocal = Fn(
    ([baseNorm_in, uvwMicro_in, hCenter_in, strength_in]: [
      TslNode,
      TslNode,
      TslNode,
      TslNode,
    ]) => {
      const uvw = vec3(uvwMicro_in);
      const du = float(0.014);
      const hx = texture3DLevel(petroVolumeTex, fract(uvw.add(vec3(du, 0.0, 0.0))), float(0.0)).a;
      const hy = texture3DLevel(petroVolumeTex, fract(uvw.add(vec3(0.0, du, 0.0))), float(0.0)).a;
      const hz = texture3DLevel(petroVolumeTex, fract(uvw.add(vec3(0.0, 0.0, du))), float(0.0)).a;
      const gradUvw = vec3(hx.sub(hCenter_in), hy.sub(hCenter_in), hz.sub(hCenter_in)).div(
        du.mul(10.0)
      );
      const gradLocal = rotLithicBTranspose(gradUvw);
      return normalize(vec3(baseNorm_in).sub(gradLocal.mul(float(strength_in))));
    }
  );

  /**
   * 2-band 3D volume relief normal perturbation (micro clast/pores + sub-mm sandstone grit)
   * evaluated strictly in uniform control flow.
   */
  const perturbDualBandPetroNormalLocal = Fn(
    ([baseNorm_in, uvwMicro_in, hMicro_in, uvwGrit_in, hGrit_in, strength_in]: [
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
    ]) => {
      const uvwM = vec3(uvwMicro_in);
      const duM = float(0.014);
      const mx = texture3DLevel(petroVolumeTex, fract(uvwM.add(vec3(duM, 0.0, 0.0))), float(0.0)).a;
      const my = texture3DLevel(petroVolumeTex, fract(uvwM.add(vec3(0.0, duM, 0.0))), float(0.0)).a;
      const mz = texture3DLevel(petroVolumeTex, fract(uvwM.add(vec3(0.0, 0.0, duM))), float(0.0)).a;
      const gradMicro = rotLithicBTranspose(
        vec3(mx.sub(hMicro_in), my.sub(hMicro_in), mz.sub(hMicro_in)).div(duM.mul(10.0))
      );

      const uvwG = vec3(uvwGrit_in);
      const duG = float(0.016);
      const gx = texture3DLevel(petroVolumeTex, fract(uvwG.add(vec3(duG, 0.0, 0.0))), float(0.0)).a;
      const gy = texture3DLevel(petroVolumeTex, fract(uvwG.add(vec3(0.0, duG, 0.0))), float(0.0)).a;
      const gz = texture3DLevel(petroVolumeTex, fract(uvwG.add(vec3(0.0, 0.0, duG))), float(0.0)).a;
      const gradGrit = rotLithicATranspose(
        vec3(gx.sub(hGrit_in), gy.sub(hGrit_in), gz.sub(hGrit_in)).div(duG.mul(10.0))
      );

      const gradLocal = gradMicro.mul(0.65).add(gradGrit.mul(0.55));
      return normalize(vec3(baseNorm_in).sub(gradLocal.mul(float(strength_in))));
    }
  );

  // -------------------------------------------------------------------------
  // Live 1536x256 Wet Yellow Ochre Trail DataTexture (linear scalar RGBA channels)
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

    // Strictly uniform control flow: explicit-LOD texture fetch unconditionally at shader entry
    const wetSample = texture(wetTrailTexture, uv()).level(float(0.0));
    const floorGate = smoothstep(float(0.08), float(0.55), floorMask);
    const wetness = clamp(wetSample.r, 0.0, 1.0).mul(floorGate);
    const freshSheen = clamp(wetSample.g, 0.0, 1.0).mul(floorGate);
    const ochreThickness = clamp(wetSample.b, 0.0, 1.0).mul(floorGate);

    const p = positionLocal;
    const uvwMacro = rotLithicA(p.mul(1.55));
    const uvwMicro = rotLithicB(p.mul(5.25)).add(vec3(0.37, 0.61, 0.19));
    const uvwGrit = rotLithicA(p.mul(17.5)).add(vec3(0.71, 0.13, 0.49));
    const macroPetro = texture3DLevel(petroVolumeTex, fract(uvwMacro), float(0.0));
    const microPetro = texture3DLevel(petroVolumeTex, fract(uvwMicro), float(0.0));
    const gritPetro = texture3DLevel(petroVolumeTex, fract(uvwGrit), float(0.0));
    const broadTone = valueNoise3DTsl(rotLithicA(p.mul(1.85)));

    // -----------------------------------------------------------------------
    // Dressed Fossiliferous Limestone (3D Clasts + Pin-Pores + Hairline Seams)
    // -----------------------------------------------------------------------
    const fineGrain = microPetro.b.mul(0.62).add(gritPetro.b.mul(0.38));
    const clastSignal = clamp(
      macroPetro.r.mul(0.46)
        .add(microPetro.r.mul(0.24))
        .add(fineGrain.mul(0.30))
        .add(broadTone.sub(0.5).mul(0.20)),
      0.06,
      0.96
    );
    const pinPore = clamp(microPetro.g.mul(0.68).add(macroPetro.g.mul(0.32)), 0.0, 1.0);
    const hairlineSeam = smoothstep(float(0.24), float(0.09), macroPetro.r);
    const calciteClast = smoothstep(float(0.76), float(0.92), microPetro.r.mul(0.60).add(macroPetro.r.mul(0.40)))
      .mul(float(0.58).add(fineGrain.mul(0.42)));

    const umberShadowBed = rawUmber.rInf.mul(0.62);
    const limestoneMixT = clamp(
      float(0.30)
        .add(clastSignal.mul(0.54))
        .sub(pinPore.mul(0.18))
        .sub(hairlineSeam.mul(0.16)),
      0.14,
      0.95
    );
    const warmMineralK = mix(
      mix(rawUmber.K, burntSienna.K, float(0.26).add(macroPetro.r.mul(0.20))),
      mix(warmKaolin.K, yellowOchre.K, float(0.28)),
      limestoneMixT
    );
    const warmMineralS = mix(
      mix(rawUmber.S, burntSienna.S, float(0.26).add(macroPetro.r.mul(0.20))),
      mix(warmKaolin.S, yellowOchre.S, float(0.28)),
      limestoneMixT
    );
    const limestoneDepth = clamp(
      float(1.05).add(clastSignal.sub(0.5).mul(0.55)).sub(pinPore.mul(0.42)),
      0.42,
      1.65
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

    // Unified Kubelka-Munk base pass for limestone + basalt (call 1 of 3)
    const stoneK = mix(
      rawUmber.K.mul(1.75).add(azuritePrussian.K.mul(0.18)),
      warmMineralK,
      limestoneW
    );
    const stoneS = mix(rawUmber.S.mul(0.42), warmMineralS, limestoneW);
    const stoneDepth = mix(basaltDepth, limestoneDepth, limestoneW);
    const stoneBacking = mix(slateBacking, umberShadowBed, limestoneW);
    const stoneKmBase = kubelkaMunkGlaze(stoneK, stoneS, stoneDepth, stoneBacking);

    const arrisWearMask = convexEdgeWear.mul(float(0.65).add(fineGrain.mul(0.35)));
    const stoneWithPores = mix(
      stoneKmBase,
      rawUmber.rInf.mul(0.46),
      clamp(
        hairlineSeam.mul(0.28)
          .add(pinPore.mul(0.32))
          .add(float(0.5).sub(gritPetro.b).mul(0.16)),
        0.0,
        0.46
      ).mul(limestoneW)
    );
    const basaltSeptaRim = smoothstep(float(0.56), float(0.84), fineGrain)
      .mul(float(1.0).sub(scoriaVesicle));
    const paleCalcite = mix(warmKaolin.rInf, boneCalcite.rInf, float(0.64));
    const limestoneHighlight = clamp(
      arrisWearMask.mul(0.30).add(calciteClast.mul(0.28)),
      0.0,
      0.48
    ).mul(limestoneW);
    const stoneBase = mix(stoneWithPores, paleCalcite, limestoneHighlight)
      .add(warmKaolin.rInf.mul(basaltSeptaRim.mul(0.06).mul(basaltW)));

    const aoFactor = mix(float(0.48), float(1.03), creviceAo);
    const stoneWithAo = stoneBase.mul(aoFactor);

    // Compute wet mask first so live liquid trail suppresses older copper patina underneath
    const wetMask = smoothstep(float(0.02), float(0.20), max(wetness, ochreThickness));

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

    // Floor-gated glossy wet Yellow Ochre trail glaze
    const wetDarkenedBacking = stoneWithPatina.mul(mix(float(1.0), float(0.82), wetMask));
    const trailK = mix(yellowOchre.K, realgarAmber.K, float(0.28).add(freshSheen.mul(0.14)));
    const trailS = mix(yellowOchre.S, realgarAmber.S, float(0.28).add(freshSheen.mul(0.14)));
    const trailOpticalDepth = ochreThickness.mul(1.45).add(wetness.mul(0.50));
    const finalStoneRgb = kubelkaMunkGlaze(
      trailK,
      trailS,
      trailOpticalDepth,
      wetDarkenedBacking
    );

    stoneMaterial.colorNode = clamp(finalStoneRgb, 0.0, 1.0);

    // Matte porous stone (0.68-0.95) -> glossy wet Yellow Ochre meniscus (0.11)
    const dryRoughness = clamp(
      mix(float(0.82), float(0.90), basaltW)
        .add(pinPore.mul(0.06))
        .sub(fineGrain.mul(0.06))
        .sub(arrisWearMask.mul(0.05)),
      0.68,
      0.95
    );
    const wetGlossFactor = clamp(wetness.mul(1.20).add(freshSheen.mul(0.32)), 0.0, 1.0);
    stoneMaterial.roughnessNode = clamp(
      mix(dryRoughness, float(0.11), wetGlossFactor),
      0.10,
      0.96
    );
    stoneMaterial.metalnessNode = mix(float(0.02), float(0.08), wetGlossFactor);

    // 2-band 3D petrographic micro-relief damped smoothly inside wet ochre pools
    const bumpAmp = mix(float(0.135), float(0.220), basaltW)
      .add(patinaFinal.mul(0.035))
      .mul(float(1.0).sub(wetGlossFactor.mul(0.88)));
    const perturbedLocalNorm = perturbDualBandPetroNormalLocal(
      normalLocal,
      uvwMicro,
      microPetro.a,
      uvwGrit,
      gritPetro.a,
      bumpAmp
    );
    stoneMaterial.normalNode = transformNormalToView(perturbedLocalNorm);

    // Soft lustrous studio sheen along wet ochre meniscus + warm capillary ignition wave on solve
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const keySoftboxDir = normalize(vec3(0.6, 4.4, 4.2));
    const halfVec = normalize(keySoftboxDir.add(viewDirW));
    const meniscusSpec = pow(max(float(0.0), dot(nW, halfVec)), float(24.0));
    const meniscusFresnel = pow(abs(float(1.0).sub(clamp(dot(nW, viewDirW), 0.0, 1.0))), float(3.2));
    const meniscusGlow = mix(yellowOchre.rInf, warmKaolin.rInf, float(0.28))
      .mul(meniscusSpec.mul(0.22).add(meniscusFresnel.mul(0.10)))
      .mul(wetGlossFactor)
      .mul(float(0.55).add(freshSheen.mul(0.45)));
    const ignitionGlow = mix(realgarAmber.rInf, yellowOchre.rInf, float(0.42))
      .mul(uVictoryIgnition)
      .mul(0.85)
      .mul(wetMask);
    stoneMaterial.emissiveNode = meniscusGlow.add(ignitionGlow);
  }

  // -------------------------------------------------------------------------
  // 2. Goal Sanctum Material (Bone Calcite Collar + Cinnabar Vermilion Cup)
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
    const sanctumPetro = texture3DLevel(petroVolumeTex, fract(uvwSanctum), float(0.0));
    const grain = sanctumPetro.b;
    const microPit = smoothstep(float(0.45), float(0.85), sanctumPetro.g);

    const calciteCollar = kubelkaMunkGlaze(
      boneCalcite.K,
      boneCalcite.S,
      float(0.88).add(grain.mul(0.32)).sub(microPit.mul(0.24)),
      warmKaolin.rInf.mul(0.76)
    );

    const mortarBacking = mix(burntSienna.rInf, rawUmber.rInf, float(0.32)).mul(0.52);
    const cinnabarBowl = kubelkaMunkGlaze(
      mix(cinnabarVermilion.K, realgarAmber.K, grain.mul(0.16)),
      mix(cinnabarVermilion.S, realgarAmber.S, grain.mul(0.16)),
      float(1.15).add(grain.mul(0.45)),
      mortarBacking
    );

    sanctumMaterial.colorNode = clamp(mix(calciteCollar, cinnabarBowl, cupBlend), 0.0, 1.0);
    sanctumMaterial.roughnessNode = clamp(
      mix(float(0.76), float(0.54), cupBlend).add(grain.mul(0.08)),
      0.42,
      0.88
    );
    sanctumMaterial.normalNode = transformNormalToView(
      perturbPetroNormalLocal(normalLocal, uvwSanctum, sanctumPetro.a, float(0.065))
    );
    sanctumMaterial.emissiveNode = mix(cinnabarVermilion.rInf, realgarAmber.rInf, float(0.48))
      .mul(cupBlend)
      .mul(uSolveProgress.mul(0.14).add(uVictoryIgnition.mul(0.45)));
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
    const startPetro = texture3DLevel(petroVolumeTex, fract(uvwStart), float(0.0));
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
  // 4. Liquid Quicksilver Bead Material (Weber Stretch, Sessile Contact
  //    Flattening, Damped Rayleigh-Lamb l=2 Oscillation & Studio Softbox Dome)
  // -------------------------------------------------------------------------
  const uBeadVelocityLocal = uniform(new THREE.Vector3(0, 0, 0));
  const uBeadNormalLocal = uniform(new THREE.Vector3(0, 1, 0));
  const uBeadImpact = uniform(0.0);

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

    // 3. Damped Rayleigh-Lamb l=2 axisymmetric oscillation P2(cos theta) from uBeadImpact
    const pUnit = normalize(p0.add(vec3(0.0, 1e-5, 0.0)));
    const cosTheta = dot(pUnit, nFloor);
    const p2Mode = float(0.5).mul(float(3.0).mul(cosTheta).mul(cosTheta).sub(1.0));
    const lambOsc = uBeadImpact.mul(sin(stage.uTime.mul(32.0))).mul(0.15);
    return pStretched.add(pUnit.mul(beadRadiusFloat.mul(lambOsc).mul(p2Mode)));
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

    // Silver-quicksilver metallic albedo anchored strictly in boneCalcite neutrality
    beadMaterial.colorNode = boneCalcite.rInf.mul(1.04);

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
    const lowerTrenchTint = mix(
      mix(rawUmber.rInf.mul(0.26), yellowOchre.rInf.mul(0.48), smoothstep(float(-0.78), float(-0.08), R.y)),
      cinnabarVermilion.rInf.mul(0.58),
      uSolveProgress.mul(0.75)
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

    beadMaterial.emissiveNode = studioEnv.mul(mercuryFresnel).mul(0.90);
  }

  // -------------------------------------------------------------------------
  // Zero-Allocation Bead Uniform Updater with Damped Rayleigh-Lamb Envelope
  // -------------------------------------------------------------------------
  let ringAmplitude = 0;
  let prevRawImpact = 0;

  const updateBeadUniforms: LabyrinthOpticsBundle['updateBeadUniforms'] = (
    velocityLocal,
    normalLocalVec,
    impactImpulse,
    solveProgress01 = 0,
    dtSec = 1 / 60
  ) => {
    const dt = Math.max(0.0005, Math.min(0.05, dtSec));

    if (Array.isArray(velocityLocal)) {
      uBeadVelocityLocal.value.set(velocityLocal[0], velocityLocal[1], velocityLocal[2]);
    } else {
      const v = velocityLocal as THREE.Vector3;
      uBeadVelocityLocal.value.set(v.x, v.y, v.z);
    }

    if (Array.isArray(normalLocalVec)) {
      uBeadNormalLocal.value.set(normalLocalVec[0], normalLocalVec[1], normalLocalVec[2]);
    } else {
      const n = normalLocalVec as THREE.Vector3;
      uBeadNormalLocal.value.set(n.x, n.y, n.z);
    }

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
  };

  const setVictoryUniforms: LabyrinthOpticsBundle['setVictoryUniforms'] = (
    solveProgress01,
    victoryIgnition01
  ) => {
    uSolveProgress.value = Math.max(0, Math.min(1, solveProgress01));
    uVictoryIgnition.value = Math.max(0, Math.min(1, victoryIgnition01));
  };

  return {
    stoneMaterial,
    sanctumMaterial,
    startSpringMaterial,
    beadMaterial,
    wetTrailTexture,
    updateBeadUniforms,
    setVictoryUniforms,
    dispose: () => {
      petroVolumeTex.dispose();
      wetTrailTexture.dispose();
      stoneMaterial.dispose();
      sanctumMaterial.dispose();
      startSpringMaterial.dispose();
      beadMaterial.dispose();
    },
  };
}
