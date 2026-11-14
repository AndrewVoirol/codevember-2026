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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export const WET_ATLAS_WIDTH = 384;
export const WET_ATLAS_HEIGHT = 64;

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
  dispose: () => void;
}

/**
 * Lightweight 2-octave irrational-angle rotated 3D FBM using `valueNoise3DTsl`
 * for crisp mineral pores, fossil flecks, and derivative-free normal gradients.
 */
const stoneDetailFbm = Fn(([pos_in, scale_in]: [TslNode, TslNode]) => {
  const p0 = vec3(pos_in).mul(scale_in);
  const v0 = valueNoise3DTsl(p0);

  const c = float(0.80901699);
  const s = float(0.58778525);
  const p1 = vec3(
    p0.x.mul(c).sub(p0.z.mul(s)),
    p0.y.mul(c).add(p0.x.mul(s)),
    p0.z.mul(c).add(p0.y.mul(s))
  ).mul(2.11);
  const v1 = valueNoise3DTsl(p1);

  return v0.mul(0.64).add(v1.mul(0.36));
});

/**
 * Single-octave derivative-free 3D value-noise micro-relief local-space normal perturbation.
 * Normalizes finite-difference gradient by frequency so `strength_in` stays scale-invariant.
 */
const perturbStoneNormalLocal = Fn(
  ([baseNorm_in, pos3D_in, freq_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const freq = float(freq_in);
    const p = vec3(pos3D_in).mul(freq);
    const eps = float(0.085);
    const h0 = valueNoise3DTsl(p);
    const hx = valueNoise3DTsl(p.add(vec3(eps, 0.0, 0.0)));
    const hy = valueNoise3DTsl(p.add(vec3(0.0, eps, 0.0)));
    const hz = valueNoise3DTsl(p.add(vec3(0.0, 0.0, eps)));
    const grad = vec3(hx.sub(h0), hy.sub(h0), hz.sub(h0)).div(eps);
    return normalize(vec3(baseNorm_in).sub(grad.mul(float(strength_in))));
  }
);

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

  // -------------------------------------------------------------------------
  // Live 384x64 Wet Yellow Ochre Trail DataTexture (linear scalar RGBA channels)
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

  // -------------------------------------------------------------------------
  // 1. Carved Stone Monolith Material (Fossiliferous Limestone + Vesicular Basalt
  //    + Verdigris Copper-Carbonate Patina + Wet Yellow Ochre Trail Glaze)
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

    // Strictly uniform control flow: explicit-LOD texture fetch unconditionally at shader entry
    const wetSample = texture(wetTrailTexture, uv()).level(float(0.0));
    const wetness = clamp(wetSample.r, 0.0, 1.0);
    const freshSheen = clamp(wetSample.g, 0.0, 1.0);
    const ochreThickness = clamp(wetSample.b, 0.0, 1.0);

    const p = positionLocal;
    const macroWarp = stoneDetailFbm(p.mul(2.1), float(1.75));

    // Fossiliferous limestone: tilted warm travertine bedding + micro-fissures + subtle warm kaolin bioclasts
    const strataCoord = vec3(
      p.x.mul(1.85),
      p.y.mul(8.6).add(p.x.mul(1.15)).add(macroWarp.mul(1.75)),
      p.z.mul(1.85)
    );
    const strata = stoneDetailFbm(strataCoord, float(1.85));
    const stoneGrain = stoneDetailFbm(p.mul(18.0), float(2.05));

    const fleckNoise = valueNoise3DTsl(
      p.mul(vec3(56.0, 92.0, 56.0)).add(vec3(3.7, 8.1, 2.4))
    );
    const fossilFlecks = smoothstep(
      float(0.86),
      float(0.96),
      fleckNoise.mul(float(0.80).add(stoneGrain.mul(0.32)))
    );

    const veinBase = valueNoise3DTsl(
      p.mul(vec3(7.4, 5.2, 7.4)).add(macroWarp.mul(1.25)).add(vec3(6.3, 1.9, 4.5))
    );
    const microFissure = float(1.0).sub(
      smoothstep(float(0.002), float(0.009), abs(veinBase.sub(0.505)))
    );

    const pitNoise = valueNoise3DTsl(p.mul(52.0).add(vec3(2.1, 5.3, 7.7)));
    const limestonePits = smoothstep(float(0.28), float(0.11), pitNoise);

    const umberShadowBed = rawUmber.rInf.mul(0.62);
    const limestoneMixT = clamp(
      float(0.55).add(strata.mul(0.34)).add(stoneGrain.mul(0.14)).sub(limestonePits.mul(0.16)),
      0.55,
      0.95
    );
    const warmLimestoneK = mix(
      rawUmber.K,
      mix(warmKaolin.K, yellowOchre.K, float(0.26)),
      limestoneMixT
    );
    const warmLimestoneS = mix(
      rawUmber.S,
      mix(warmKaolin.S, yellowOchre.S, float(0.26)),
      limestoneMixT
    );
    const limestoneDepth = clamp(
      float(1.08).add(strata.mul(0.65)).sub(limestonePits.mul(0.38)),
      0.28,
      1.75
    );
    const limestoneGlaze = kubelkaMunkGlaze(
      warmLimestoneK,
      warmLimestoneS,
      limestoneDepth,
      umberShadowBed
    );
    const limestoneFissured = limestoneGlaze.mul(float(1.0).sub(microFissure.mul(0.12)));
    const fossilGlazeDepth = clamp(fossilFlecks.mul(0.09), 0.0, 0.09);
    const limestoneFinal = kubelkaMunkGlaze(
      warmKaolin.K,
      warmKaolin.S,
      fossilGlazeDepth,
      limestoneFissured
    );

    // Dark vesicular basalt: rawUmber heavily absorbing over deep Chalky Slate backing
    const vesicleMacro = valueNoise3DTsl(p.mul(34.0).add(vec3(4.9, 2.3, 7.6)));
    const vesicleMicro = valueNoise3DTsl(p.mul(85.0));
    const vesiclePit = max(
      smoothstep(float(0.39), float(0.18), vesicleMacro),
      smoothstep(float(0.32), float(0.13), vesicleMicro).mul(0.68)
    );
    const slateBacking = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    ).mul(0.55);
    const basaltBody = kubelkaMunkGlaze(
      rawUmber.K.mul(2.1),
      rawUmber.S.mul(0.35),
      float(0.26).add(vesicleMacro.mul(0.22)).mul(float(1.0).sub(vesiclePit.mul(0.88))),
      slateBacking
    );
    const basaltSeptaRim = smoothstep(float(0.62), float(0.88), vesicleMacro).mul(
      float(1.0).sub(vesiclePit)
    );
    const basaltFinal = basaltBody.add(warmKaolin.rInf.mul(basaltSeptaRim.mul(0.085)));

    // Blend limestone & basalt zones + apply carved crevice & underpass soffit AO
    const stoneBase = mix(basaltFinal, limestoneFinal, limestoneW);
    const aoFactor = mix(float(0.24), float(1.04), creviceAo);
    const stoneWithAo = stoneBase.mul(aoFactor);

    // Compute wet mask first so live liquid trail suppresses older copper patina underneath
    const wetMask = smoothstep(float(0.02), float(0.20), max(wetness, ochreThickness));

    // Fine mineral-granule Verdigris Malachite & Celadon Glauconite patina in deep seams & soffits
    const patinaMacro = valueNoise3DTsl(p.mul(36.0).add(vec3(1.5, 6.2, 3.8)));
    const patinaGranule = valueNoise3DTsl(p.mul(78.0).add(vec3(7.2, 2.9, 5.1)));
    const patinaTex = mix(patinaMacro, patinaGranule, float(0.45));
    const patinaSignal = patinaInput
      .mul(float(0.48).add(patinaTex.mul(0.84)))
      .add(stoneGrain.sub(0.5).mul(0.22));
    const patinaCoverage = smoothstep(float(0.40), float(0.86), patinaSignal);
    const patinaFinal = patinaCoverage.mul(float(1.0).sub(wetMask.mul(0.90)));
    const patinaK = mix(verdigrisMalachite.K, celadonGlauconite.K, patinaTex.mul(0.32));
    const patinaS = mix(verdigrisMalachite.S, celadonGlauconite.S, patinaTex.mul(0.32));
    const stoneWithPatina = kubelkaMunkGlaze(
      patinaK,
      patinaS,
      patinaFinal.mul(1.32),
      stoneWithAo
    );

    // Vivid glossy wet Yellow Ochre trail glaze
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

    // Matte porous stone (0.78-0.92) -> glossy wet Yellow Ochre meniscus (0.11)
    const dryRoughness = clamp(
      mix(float(0.78), float(0.90), basaltW)
        .add(patinaFinal.mul(0.08))
        .sub(stoneGrain.mul(0.06)),
      0.68,
      0.96
    );
    const wetGlossFactor = clamp(wetness.mul(1.20).add(freshSheen.mul(0.32)), 0.0, 1.0);
    stoneMaterial.roughnessNode = clamp(
      mix(dryRoughness, float(0.11), wetGlossFactor),
      0.10,
      0.96
    );
    stoneMaterial.metalnessNode = mix(float(0.02), float(0.08), wetGlossFactor);

    // Fine 3D mineral tooth bump (deepened vesicle pores on basalt) damped inside wet pools
    const bumpAmp = mix(float(0.095), float(0.20), basaltW)
      .add(patinaFinal.mul(0.032))
      .mul(float(1.0).sub(wetGlossFactor.mul(0.88)));
    const perturbedLocalNorm = perturbStoneNormalLocal(
      normalLocal,
      p,
      float(64.0),
      bumpAmp
    );
    stoneMaterial.normalNode = transformNormalToView(perturbedLocalNorm);

    // Soft lustrous studio sheen along wet ochre meniscus pools without white facet blowout
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const keySoftboxDir = normalize(vec3(0.6, 4.4, 4.2));
    const halfVec = normalize(keySoftboxDir.add(viewDirW));
    const meniscusSpec = pow(max(float(0.0), dot(nW, halfVec)), float(24.0));
    const meniscusFresnel = pow(abs(float(1.0).sub(clamp(dot(nW, viewDirW), 0.0, 1.0))), float(3.2));
    stoneMaterial.emissiveNode = mix(yellowOchre.rInf, warmKaolin.rInf, float(0.28))
      .mul(meniscusSpec.mul(0.22).add(meniscusFresnel.mul(0.10)))
      .mul(wetGlossFactor)
      .mul(float(0.55).add(freshSheen.mul(0.45)));
  }

  // -------------------------------------------------------------------------
  // 2. Goal Sanctum Material (Bone Calcite Collar + Cinnabar Vermilion Cup)
  // -------------------------------------------------------------------------
  const uSolveProgress = uniform(0.0);

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
    const grain = stoneDetailFbm(p.mul(22.0), float(2.2));
    const microPit = smoothstep(float(0.28), float(0.14), valueNoise3DTsl(p.mul(58.0)));

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
      perturbStoneNormalLocal(normalLocal, p, float(46.0), float(0.045))
    );
    sanctumMaterial.emissiveNode = mix(cinnabarVermilion.rInf, realgarAmber.rInf, float(0.48))
      .mul(cupBlend)
      .mul(uSolveProgress.mul(0.14));
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
    const grain = stoneDetailFbm(p.mul(26.0), float(2.2));
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
      smoothstep(float(0.78), float(0.95), grain).mul(0.22)
    );

    startSpringMaterial.colorNode = clamp(mix(basaltSocket, springWithCalcite, rimBlend), 0.0, 1.0);
    startSpringMaterial.roughnessNode = clamp(
      mix(float(0.84), float(0.24), rimBlend).add(grain.mul(0.06)),
      0.18,
      0.92
    );
    startSpringMaterial.normalNode = transformNormalToView(
      perturbStoneNormalLocal(normalLocal, p, float(52.0), mix(float(0.10), float(0.035), rimBlend))
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

  return {
    stoneMaterial,
    sanctumMaterial,
    startSpringMaterial,
    beadMaterial,
    wetTrailTexture,
    updateBeadUniforms,
    dispose: () => {
      wetTrailTexture.dispose();
      stoneMaterial.dispose();
      sanctumMaterial.dispose();
      startSpringMaterial.dispose();
      beadMaterial.dispose();
    },
  };
}
