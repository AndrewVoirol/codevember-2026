import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  floor,
  fract,
  length,
  max,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { PERISTOME_NOZZLE_Y, SAC_CENTER_Y } from './bellowsPhysics';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface EarthstarMaterialRig {
  sacMaterial: THREE.MeshStandardNodeMaterial;
  petalMaterial: THREE.MeshStandardNodeMaterial;
  rootMaterial: THREE.MeshStandardNodeMaterial;
  uCompression: { value: number };
  uIndentStrength: { value: number };
  uIndentDir: { value: THREE.Vector3 };
  uPeristomeFlare: { value: number };
  uPetalRecoil: { value: number };
  uHoverWarmth: { value: number };
  uTime: { value: number };
}

/**
 * Deterministic 2D hash in TSL (strictly positive-safe via sin-fract).
 */
const hash21Tsl = Fn(([p_in]: [TslNode]) => {
  const p = vec2(p_in);
  return fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453123));
});

/**
 * Per-pixel 3x3 Voronoi Tectonic Desiccation Crack Field in TSL.
 * Evaluates razor-sharp polygonal fissures (`f2 - f1`) + cell tint ID on the
 * dried fungal crust at full Retina pixel resolution with zero vertex blur!
 */
const evalVoronoiCracksTsl = Fn(([uv_in, seed_in]: [TslNode, TslNode]) => {
  const uvBase = vec2(uv_in);
  const seed = float(seed_in);

  // Domain-warp slightly using 3D mineral FBM so fissure lines fracture jaggedly like dried clay
  const wU = mineralToothFbm(vec3(uvBase.x.mul(1.35).add(seed), uvBase.y.mul(1.35), 2.3), float(1.0)).sub(0.5);
  const wV = mineralToothFbm(vec3(uvBase.x.mul(1.35), uvBase.y.mul(1.35).add(seed), 7.1), float(1.0)).sub(0.5);
  const fineJagged = mineralToothFbm(vec3(uvBase.x.mul(6.8).add(seed), uvBase.y.mul(6.8), 11.4), float(1.0)).sub(0.5);

  const wp = uvBase.add(vec2(wU.mul(0.36).add(fineJagged.mul(0.08)), wV.mul(0.36).add(fineJagged.mul(0.08))));
  const iCell = floor(wp);

  const f1 = float(10.0).toVar();
  const f2 = float(10.0).toVar();
  const bestId = float(0.5).toVar();

  // Unrolled 3x3 Voronoi neighborhood (straight-line WGSL, zero dynamic loop issues)
  for (let dv = -1; dv <= 1; dv++) {
    for (let du = -1; du <= 1; du++) {
      const cell = iCell.add(vec2(du, dv));
      const rx = hash21Tsl(vec2(cell.x.add(seed.mul(13.7)), cell.y.mul(1.13).add(4.1)));
      const ry = hash21Tsl(vec2(cell.x.mul(1.29).add(9.2), cell.y.add(seed.mul(7.3))));
      const pt = cell.add(vec2(0.14, 0.14)).add(vec2(rx, ry).mul(0.72));
      const d = length(pt.sub(wp));

      If(d.lessThan(f1), () => {
        f2.assign(f1);
        f1.assign(d);
        bestId.assign(hash21Tsl(cell.mul(vec2(3.7, 5.3))));
      }).ElseIf(d.lessThan(f2), () => {
        f2.assign(d);
      });
    }
  }

  return vec2(max(float(0.0), f2.sub(f1)), bestId);
});

/**
 * Analytical 3D tangent-projected bump normal perturbation using finite differences
 * of a scalar 3D height field — zero screen-space `dpdx`/`dpdy` artifacts!
 */
const evalPaperyHeightTsl = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  // Vertically stretched organic tension veins & smooth taut parchment creases (not pebble dimples)
  const pStretched = vec3(p.x.mul(1.35), p.y.mul(0.62), p.z.mul(1.35));
  const f1 = mineralToothFbm(pStretched.mul(2.4), float(1.0));
  const f2 = mineralToothFbm(pStretched.mul(5.8).add(vec3(3.1, 7.4, 2.2)), float(1.0));
  const ridge1 = float(1.0).sub(abs(f1.mul(2.0).sub(1.0)));
  const ridge2 = float(1.0).sub(abs(f2.mul(2.0).sub(1.0)));
  return ridge1.mul(ridge1).mul(0.76).add(ridge2.mul(ridge2).mul(0.24));
});

/**
 * Builds the 3 physical mineral-pigment TSL NodeMaterials for the suspended
 * Geastrum triplex Earthstar puffball specimen with Retina-sharp per-pixel
 * papery creases, Voronoi tectonic desiccation cracks, and Kubelka-Munk glazes.
 */
export function createEarthstarMaterials(): EarthstarMaterialRig {
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const rawUmber = createPigmentUniforms('rawUmber');
  const burntSienna = createPigmentUniforms('burntSienna');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const sporeSulfur = createPigmentUniforms('sporeSulfur');
  const mossUmbrous = createPigmentUniforms('mossUmbrous');

  const uCompression = uniform(0);
  const uIndentStrength = uniform(0);
  const uIndentDir = uniform(new THREE.Vector3(0, 0.35, 0.936));
  const uPeristomeFlare = uniform(0);
  const uPetalRecoil = uniform(0);
  const uHoverWarmth = uniform(0);
  const uTime = uniform(0);

  // -------------------------------------------------------------------------
  // 1. Papery Endoperidium Spore Sac Material
  // -------------------------------------------------------------------------
  const sacMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.86,
    metalness: 0.0,
    side: THREE.FrontSide,
  });

  const sacMeta = attribute('aSacMeta', 'vec4');
  const peristomeMask = sacMeta.x;
  const sulfurDustMask = sacMeta.y;
  const creaseAO = sacMeta.z;
  const throatMask = sacMeta.w;

  const pLoc = positionLocal;
  const nLoc = normalLocal;

  // Direction from sac center for elastic bellows deformation
  const sacCenter = vec3(0.0, SAC_CENTER_Y, 0.0);
  const relPos = pLoc.sub(sacCenter);
  const relDir = normalize(relPos.add(vec3(0.0, 1e-4, 0.0)));
  const alignPoke = dot(relDir, normalize(uIndentDir));

  // Localized concave thumb-dimple at the pressed side + surrounding papery bucking wave
  const dimpleEnv = exp(float(1.0).sub(alignPoke).mul(-5.8));
  const buckleRing = exp(pow(alignPoke.sub(0.62), float(2.0)).mul(-18.0));
  const bodyFlexibility = float(1.0).sub(peristomeMask.mul(0.85)).mul(float(1.0).sub(throatMask));
  const dimplePush = dimpleEnv.mul(uIndentStrength).mul(0.15).mul(bodyFlexibility);
  const buckleWave = sin(alignPoke.mul(22.0).sub(uIndentStrength.mul(4.0)))
    .mul(0.009)
    .mul(buckleRing)
    .mul(uIndentStrength)
    .mul(bodyFlexibility);

  const heightNorm = clamp(pLoc.y.sub(SAC_CENTER_Y).div(0.58), -1.0, 1.2);
  const radialXZ = vec3(pLoc.x, 0.0, pLoc.z);
  const radialDirXZ = normalize(radialXZ.add(vec3(1e-5, 0.0, 0.0)));
  const squashY = heightNorm.add(0.45).mul(uCompression).mul(-0.095);
  const volumeBulgeXZ = float(1.0).sub(clamp(abs(heightNorm), 0.0, 1.0)).mul(uCompression).mul(0.055).mul(float(1.0).sub(peristomeMask));

  const rimExpand = peristomeMask
    .mul(smoothstep(0.09, PERISTOME_NOZZLE_Y, pLoc.y))
    .mul(uPeristomeFlare)
    .mul(0.030);

  const breath = sin(uTime.mul(1.65)).mul(0.0028).mul(bodyFlexibility);

  const displacedSac = pLoc
    .sub(nLoc.mul(dimplePush))
    .add(nLoc.mul(buckleWave.add(breath)))
    .add(vec3(0.0, squashY, 0.0))
    .add(radialDirXZ.mul(volumeBulgeXZ.add(rimExpand)));

  sacMaterial.positionNode = displacedSac;

  // Per-pixel papery parchment bump normals (subtle taut tension creases -> view space)
  const epsSac = float(0.012);
  const hC = evalPaperyHeightTsl(pLoc);
  const hX = evalPaperyHeightTsl(pLoc.add(vec3(epsSac, 0.0, 0.0)));
  const hY = evalPaperyHeightTsl(pLoc.add(vec3(0.0, epsSac, 0.0)));
  const hZ = evalPaperyHeightTsl(pLoc.add(vec3(0.0, 0.0, epsSac)));
  const gradSac = vec3(hX.sub(hC), hY.sub(hC), hZ.sub(hC)).div(epsSac);
  const tangentGradSac = gradSac.sub(nLoc.mul(dot(gradSac, nLoc)));
  const sacBumpAmp = float(0.015).mul(float(1.0).sub(throatMask.mul(0.85)));
  sacMaterial.normalNode = transformNormalToView(normalize(nLoc.sub(tangentGradSac.mul(sacBumpAmp))));

  // Papery mineral fiber micro-tooth & multi-pigment Kubelka-Munk glaze
  const paperTooth = mineralToothFbm(pLoc.mul(10.0), float(1.0));
  const fineGrain = mineralToothFbm(pLoc.mul(48.0), float(1.0));

  const kaolinThickness = clamp(
    float(0.50)
      .add(paperTooth.mul(0.28))
      .add(hC.mul(0.18))
      .sub(creaseAO.mul(0.56))
      .sub(uIndentStrength.mul(dimpleEnv).mul(0.26)),
    0.04,
    1.05
  );
  const paperyBase = kubelkaMunkGlaze(
    warmKaolin.K,
    warmKaolin.S,
    kaolinThickness,
    rawUmber.rInf
  );

  // Warm parchment highlight along raised crease ridges + fluted peristome ribs
  const ridgeLift = smoothstep(0.42, 0.86, hC).mul(float(1.0).sub(creaseAO));
  const sacSkin = mix(paperyBase, boneCalcite.rInf, ridgeLift.mul(0.26).add(peristomeMask.mul(0.14)));

  // Crisp granular sulfur pollen dust clinging densely to the pleated peristome collar & rim
  const pollenSpecks = smoothstep(0.26, 0.68, fineGrain);
  const activeDust = clamp(
    sulfurDustMask.mul(float(0.68).add(pollenSpecks.mul(0.75))).add(uPeristomeFlare.mul(peristomeMask).mul(0.22)),
    0.0,
    1.0
  );
  const pollenColor = kubelkaMunkGlaze(
    sporeSulfur.K,
    sporeSulfur.S,
    activeDust.mul(1.45),
    yellowOchre.rInf
  );
  const dustedSkin = mix(sacSkin, pollenColor, smoothstep(0.14, 0.72, activeDust).mul(0.95));

  // Warm backlit Beer-Lambert papery rim translucency
  const viewDir = normalize(cameraPosition.sub(positionWorld));
  const nDotV = clamp(dot(normalize(normalWorld), viewDir), 0.0, 1.0);
  const grazingRim = pow(float(1.0).sub(nDotV), float(2.8));
  const rimTrans = beerLambertTransmittance(yellowOchre.extinction, float(0.46).add(nDotV.mul(0.65)));
  const paperyBackscatter = rimTrans.mul(grazingRim).mul(float(0.18).add(peristomeMask.mul(0.16)).add(uHoverWarmth.mul(0.07)));

  const throatDark = rawUmber.rInf.mul(0.12);
  const sacFinal = mix(dustedSkin, throatDark, throatMask.mul(0.94))
    .mul(float(1.0).sub(creaseAO.mul(0.50)))
    .add(paperyBackscatter);

  sacMaterial.colorNode = sacFinal;
  sacMaterial.roughnessNode = clamp(
    float(0.84).add(fineGrain.mul(0.12)).add(activeDust.mul(0.09)).sub(ridgeLift.mul(0.07)),
    0.70,
    0.98
  );

  // -------------------------------------------------------------------------
  // 2. Recurved Cracked-Crust Exoperidium Star Petals Material
  //    Razor-sharp per-pixel Voronoi tectonic fissures + beveled plate normals
  //    + arch-to-tip Bone Calcite -> Raw Umber weathering transition!
  // -------------------------------------------------------------------------
  const petalMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.88,
    metalness: 0.0,
    side: THREE.FrontSide,
  });

  const petalMeta = attribute('aPetalMeta', 'vec4');
  const crustMask = petalMeta.x;
  const fleshyEdgeMask = petalMeta.y;
  const flexWeight = petalMeta.w;

  const petalCoord = attribute('aPetalCoord', 'vec4');
  const pU = petalCoord.x;
  const pSide = petalCoord.y;
  const pSeed = petalCoord.z;

  const ppLoc = positionLocal;
  const pnLoc = normalLocal;
  const pRadXZ = vec3(ppLoc.x, 0.0, ppLoc.z);
  const pRadDir = normalize(pRadXZ.add(vec3(1e-5, 0.0, 0.0)));

  const petalFlexOffset = vec3(0.0, uPetalRecoil.mul(-0.088).mul(flexWeight), 0.0)
    .add(pRadDir.mul(uPetalRecoil.mul(0.042).mul(flexWeight)));
  petalMaterial.positionNode = ppLoc.add(petalFlexOffset);

  // Per-pixel isometric Voronoi desiccation cracks (1:1 world aspect: length ~1.3 / 4.8 vs width ~0.48 / (2*0.88))
  const crackUv = vec2(pU.mul(4.8), pSide.mul(0.88));
  const vorC = evalVoronoiCracksTsl(crackUv, pSeed);
  const edgeDist = vorC.x;
  const cellId = vorC.y;

  // Razor-sharp fissure groove (dark core < 0.045, beveled Burnt Sienna shoulder 0.045..0.11)
  const crackCore = float(1.0).sub(smoothstep(0.014, 0.056, edgeDist)).mul(crustMask);
  const crackBevel = float(1.0).sub(smoothstep(0.046, 0.115, edgeDist)).mul(crustMask);

  // Finite-difference normal bump for crisp chiseled plate bevels + rough bark tooth
  const eCr = float(0.014);
  const vorDu = evalVoronoiCracksTsl(crackUv.add(vec2(eCr, 0.0)), pSeed).x;
  const vorDv = evalVoronoiCracksTsl(crackUv.add(vec2(0.0, eCr)), pSeed).x;
  const hPlateC = smoothstep(0.015, 0.085, edgeDist);
  const hPlateDu = smoothstep(0.015, 0.085, vorDu);
  const hPlateDv = smoothstep(0.015, 0.085, vorDv);
  const dU = hPlateDu.sub(hPlateC).div(eCr);
  const dV = hPlateDv.sub(hPlateC).div(eCr);

  const barkBump = mineralToothFbm(ppLoc.mul(22.0), float(1.0)).sub(0.5);
  const barkBumpX = mineralToothFbm(ppLoc.add(vec3(0.01, 0.0, 0.0)).mul(22.0), float(1.0)).sub(0.5);
  const barkBumpY = mineralToothFbm(ppLoc.add(vec3(0.0, 0.01, 0.0)).mul(22.0), float(1.0)).sub(0.5);
  const barkGrad = vec3(barkBumpX.sub(barkBump), barkBumpY.sub(barkBump), 0.0).mul(14.0);

  // Approximate radial & lateral tangent basis for crisp plate crack-wall lighting
  const latTangent = normalize(vec3(pRadDir.z.negate(), 0.0, pRadDir.x));
  const radTangent = normalize(vec3(pRadDir.x, -0.45, pRadDir.z));
  const crackNormalTilt = radTangent.mul(dU).add(latTangent.mul(dV)).mul(crustMask).mul(0.085);
  const perturbedPetalN = normalize(pnLoc.sub(crackNormalTilt).sub(barkGrad.mul(0.035)));
  petalMaterial.normalNode = transformNormalToView(perturbedPetalN);

  const petalGrain = mineralToothFbm(ppLoc.mul(10.5), float(1.0));
  const petalMicro = mineralToothFbm(ppLoc.mul(32.0), float(1.0));

  // Outer / underside velvety dark Raw Umber bark
  const umberBase = kubelkaMunkGlaze(
    rawUmber.K,
    rawUmber.S,
    float(0.95).add(petalGrain.mul(0.45)),
    vec3(0.014, 0.011, 0.009)
  );

  // Exposed middle fleshy Burnt Sienna rim & cracked fissure interior walls
  const siennaFlesh = kubelkaMunkGlaze(
    burntSienna.K,
    burntSienna.S,
    float(0.72).add(petalMicro.mul(0.35)),
    rawUmber.rInf
  );

  // Dried Bone Calcite / Warm Kaolin fissured top plates with subtle per-polygon tone shift
  const plateThickness = clamp(
    float(0.82).add(cellId.mul(0.28)).add(petalGrain.mul(0.25)).sub(crackBevel.mul(0.78)),
    0.06,
    1.35
  );
  const calcCrust = kubelkaMunkGlaze(
    boneCalcite.K,
    boneCalcite.S,
    plateThickness,
    warmKaolin.rInf
  );

  // Natural arch-to-tip weathering matching Concept A:
  // Aged Warm Kaolin/Bone Calcite cracked plates over the upper curved arch (u = 0.08..0.48),
  // weathering deeply into earthy Raw Umber bark down the lower hanging legs (u = 0.48..1.0)!
  const agedPlates = mix(calcCrust, warmKaolin.rInf.mul(0.88), 0.35);
  const tipWeathering = smoothstep(0.32, 0.68, pU.add(petalGrain.sub(0.5).mul(0.18)));
  const weatheredCrust = mix(
    agedPlates,
    mix(rawUmber.rInf.mul(0.72), burntSienna.rInf.mul(0.45), 0.22),
    tipWeathering.mul(0.92)
  );

  // Crisp Burnt Sienna beveled fissure shoulders framing dark Raw Umber crack interiors
  const plateWithBevel = mix(weatheredCrust, siennaFlesh, crackBevel.mul(0.72));
  const plateWithFissure = mix(plateWithBevel, umberBase.mul(0.28), crackCore.mul(0.95));

  // Combine underside -> fleshy beveled side rim -> cracked upper crust
  const bodyPlusFlesh = mix(umberBase, siennaFlesh, fleshyEdgeMask.mul(0.88));
  const layeredPetal = mix(bodyPlusFlesh, plateWithFissure, crustMask);

  // Deep shadow along underside curl & inside crack trenches
  const undersideDepth = mix(float(0.62), float(1.0), smoothstep(-0.96, -0.38, ppLoc.y));
  petalMaterial.colorNode = layeredPetal.mul(undersideDepth);
  petalMaterial.roughnessNode = clamp(
    float(0.84).add(petalMicro.mul(0.12)).add(crackCore.mul(0.12)).sub(crustMask.mul(0.06)),
    0.72,
    0.98
  );

  // -------------------------------------------------------------------------
  // 3. Gnarled Dark Raw Umber Mycelial Rootlets & Shadowy Moss Umbrous Tuft
  // -------------------------------------------------------------------------
  const rootMaterial = new THREE.MeshStandardNodeMaterial({
    roughness: 0.94,
    metalness: 0.0,
    side: THREE.FrontSide,
  });

  const rootMeta = attribute('aRootMeta', 'vec4');
  const lenFrac = rootMeta.x;
  const mossFlag = rootMeta.y;
  const rootSeed = rootMeta.z;
  const swayWeight = rootMeta.w;

  const rpLoc = positionLocal;
  const swayX = sin(uTime.mul(1.35).add(rootSeed.mul(14.0)).add(rpLoc.y.mul(4.0)))
    .mul(0.009)
    .mul(swayWeight);
  const swayZ = sin(uTime.mul(1.08).add(rootSeed.mul(19.0)))
    .mul(0.008)
    .mul(swayWeight);
  rootMaterial.positionNode = rpLoc.add(vec3(swayX, uPetalRecoil.mul(-0.022), swayZ));

  const rootNoise = mineralToothFbm(rpLoc.mul(24.0), float(1.0));
  // Rich earthy Raw Umber tendrils warming slightly toward aged Ochre-Kaolin at hairline tips
  const rootWood = mix(
    rawUmber.rInf.mul(0.46),
    mix(rawUmber.rInf, warmKaolin.rInf, 0.32),
    lenFrac.mul(0.55).add(rootNoise.mul(0.22))
  );
  // Shadowy dark forest-floor Moss Umbrous blended deep into Raw Umber peat (zero bright cartoon green)
  const mossColor = kubelkaMunkGlaze(
    mossUmbrous.K,
    mossUmbrous.S,
    float(0.52).add(rootNoise.mul(0.35)),
    rawUmber.rInf.mul(0.22)
  ).mul(0.46);

  rootMaterial.colorNode = mix(rootWood, mossColor, mossFlag);
  rootMaterial.roughnessNode = clamp(float(0.90).add(rootNoise.mul(0.08)), 0.84, 0.99);

  return {
    sacMaterial,
    petalMaterial,
    rootMaterial,
    uCompression,
    uIndentStrength,
    uIndentDir,
    uPeristomeFlare,
    uPetalRecoil,
    uHoverWarmth,
    uTime,
  };
}
