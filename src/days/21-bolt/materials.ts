import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  atan,
  clamp,
  cos,
  dot,
  float,
  fract,
  length,
  max,
  mix,
  normalLocal,
  normalize,
  positionLocal,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { LEVER_COUNT } from './lockMechanism';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface BoltMaterials {
  charredOak: THREE.MeshStandardNodeMaterial;
  gunmetalCase: THREE.MeshStandardNodeMaterial;
  bluedSteelBolt: THREE.MeshStandardNodeMaterial;
  forendBrass: THREE.MeshStandardNodeMaterial;
  machinedGearBrass: THREE.MeshStandardNodeMaterial;
  machinedBronzeRack: THREE.MeshStandardNodeMaterial;
  phosphorBronze: THREE.MeshStandardNodeMaterial;
  keyBrass: THREE.MeshStandardNodeMaterial;
  hardenedSteel: THREE.MeshStandardNodeMaterial;
  leverMaterials: THREE.MeshStandardNodeMaterial[];
  uKnobHover: { value: number };
  /** Alias for uKnobHover */
  uKeyHover: { value: number };
  uBoltHover: { value: number };
  uLeverHovers: Array<{ value: number }>;
  uLeverBinds: Array<{ value: number }>;
  dispose: () => void;
}

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 */
const perturbNormalAnisoObject = Fn(
  ([baseNorm_in, pos_in, scaleVec_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in).mul(vec3(scaleVec_in));
    const eps = float(0.08);

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
    const anisoDirWeight = vec3(scaleVec_in).div(maxScale);
    const grad = k1
      .mul(f1)
      .add(k2.mul(f2))
      .add(k3.mul(f3))
      .add(k4.mul(f4))
      .div(eps.mul(4.0))
      .mul(anisoDirWeight);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

const perturbNormalAniso = Fn(
  ([baseNorm_in, pos_in, scaleVec_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    return transformNormalToView(
      perturbNormalAnisoObject(baseNorm_in, pos_in, scaleVec_in, strength_in)
    );
  }
);

const computeBevelEdgeMask = Fn(([nObj_in]: [TslNode]) => {
  const n = normalize(vec3(nObj_in));
  const maxAxis = max(abs(n.x), max(abs(n.y), abs(n.z)));
  return smoothstep(float(0.975), float(0.76), maxAxis);
});

export function createBoltMaterials(): BoltMaterials {
  const uKnobHover = uniform(0);
  const uBoltHover = uniform(0);
  const uLeverHovers: Array<{ value: number }> = [];
  const uLeverBinds: Array<{ value: number }> = [];

  const pUmber = createPigmentUniforms('rawUmber');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pSienna = createPigmentUniforms('burntSienna');
  const pPrussian = createPigmentUniforms('azuritePrussian');

  const rawUmberRgb = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const burntSiennaRgb = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const yellowOchreRgb = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const realgarAmberRgb = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const boneCalciteRgb = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const warmKaolinRgb = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const azuriteRgb = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const lapisRgb = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);
  const hematiteRgb = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear);
  const verdigrisRgb = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Natural Oiled Charred-Oak Architectural Door-Stile Specimen Block
  // ---------------------------------------------------------------------------
  const charredOak = new THREE.MeshStandardNodeMaterial({
    metalness: 0.03,
    roughness: 0.48,
    envMapIntensity: 0.92,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const endGrainMask = smoothstep(float(0.54), float(0.86), abs(nObj.y));

    const pithOffset = vec2(positionLocal.x.add(1.48), positionLocal.z.add(1.12));
    const gentleBow = positionLocal.y.mul(positionLocal.y).mul(0.115);
    const slowWander = mineralToothFbm(positionLocal.mul(vec3(2.2, 0.48, 2.2)), float(1.45));
    const midWander = mineralToothFbm(positionLocal.mul(vec3(6.8, 1.4, 6.8)), float(1.25));
    const growthRateVar = mineralToothFbm(
      positionLocal.mul(vec3(4.1, 0.35, 4.1)).add(vec3(3.2, 1.1, 5.7)),
      float(1.3)
    );

    const rPith = length(pithOffset)
      .add(gentleBow)
      .add(slowWander.sub(0.5).mul(0.24))
      .add(midWander.sub(0.5).mul(0.055));
    const warpedRingCoord = rPith.mul(22.5).add(growthRateVar.sub(0.5).mul(3.4));
    const warpedEndCoord = rPith.mul(26.0).add(growthRateVar.sub(0.5).mul(3.6));

    const phaseLong = fract(warpedRingCoord.add(20.0));
    const latewoodLong = smoothstep(float(0.70), float(0.93), phaseLong).mul(
      float(1.0).sub(smoothstep(float(0.95), float(1.0), phaseLong))
    );
    const earlySoftGradLong = smoothstep(float(0.08), float(0.72), phaseLong);
    const earlyPorousLong = smoothstep(float(0.02), float(0.14), phaseLong).mul(
      float(1.0).sub(smoothstep(float(0.22), float(0.42), phaseLong))
    );

    const phaseEnd = fract(warpedEndCoord.add(20.0));
    const latewoodEnd = smoothstep(float(0.68), float(0.92), phaseEnd).mul(
      float(1.0).sub(smoothstep(float(0.95), float(1.0), phaseEnd))
    );
    const earlySoftGradEnd = smoothstep(float(0.08), float(0.70), phaseEnd);
    const earlyPorousEnd = smoothstep(float(0.02), float(0.16), phaseEnd).mul(
      float(1.0).sub(smoothstep(float(0.24), float(0.44), phaseEnd))
    );

    const latewood = mix(latewoodLong, latewoodEnd, endGrainMask);
    const ringGrad = mix(earlySoftGradLong, earlySoftGradEnd, endGrainMask);
    const earlyPorousBand = mix(earlyPorousLong, earlyPorousEnd, endGrainMask);

    const broadFiber = mineralToothFbm(positionLocal.mul(vec3(36.0, 2.6, 36.0)), float(1.3));
    const fineFibers = mineralToothFbm(positionLocal.mul(vec3(115.0, 6.2, 115.0)), float(1.25));
    const microVesselsLong = mineralToothFbm(positionLocal.mul(vec3(165.0, 11.5, 165.0)), float(1.2));
    const endPoresRaw = mineralToothFbm(positionLocal.mul(vec3(135.0, 135.0, 135.0)), float(1.3));
    const activePores = mix(microVesselsLong, endPoresRaw, endGrainMask);

    const vesselGroove = clamp(
      float(1.0)
        .sub(smoothstep(float(0.18), float(0.46), activePores))
        .mul(float(0.42).add(earlyPorousBand.mul(1.15)))
        .add(float(1.0).sub(smoothstep(float(0.12), float(0.36), fineFibers)).mul(0.28)),
      0.0,
      1.0
    );

    const rayAngle = atan(pithOffset.y, pithOffset.x);
    const rayNoise = mineralToothFbm(
      vec3(rayAngle.mul(42.0), positionLocal.y.mul(16.0), rPith.mul(6.5)),
      float(1.2)
    );
    const rayFleckLong = smoothstep(float(0.62), float(0.88), rayNoise).mul(
      float(1.0).sub(latewood.mul(0.65))
    );
    const rayLinesEnd = smoothstep(
      float(0.48),
      float(0.94),
      sin(rayAngle.mul(104.0).add(slowWander.mul(2.6))).mul(0.5).add(0.5)
    );
    const rayAccent = mix(rayFleckLong, rayLinesEnd.mul(0.74), endGrainMask);

    const heartwoodBase = mix(
      rawUmberRgb.mul(0.64),
      burntSiennaRgb.mul(0.50),
      float(0.28).add(broadFiber.mul(0.28))
    );
    const heartwoodRich = mix(
      heartwoodBase,
      mix(yellowOchreRgb.mul(0.44), burntSiennaRgb.mul(0.42), 0.45),
      ringGrad.mul(0.38).add(fineFibers.mul(0.18))
    );
    const raySheen = mix(yellowOchreRgb.mul(0.64), warmKaolinRgb.mul(0.38), 0.24);
    const earlyWithRays = mix(heartwoodRich, raySheen, rayAccent.mul(0.42));

    const darkLate = mix(rawUmberRgb.mul(0.085), azuriteRgb.mul(0.045), 0.22);
    const woodFigure = mix(earlyWithRays, darkLate, latewood.mul(0.86));

    const glazeThick = float(0.28).add(latewood.mul(0.26)).add(vesselGroove.mul(0.24));
    const kmGlazed = kubelkaMunkGlaze(pUmber.K, pUmber.S, glazeThick, woodFigure).mul(0.76);
    const oiledTimber = mix(woodFigure.mul(0.68), kmGlazed, 0.48);

    const inPocketX = smoothstep(float(-0.87), float(-0.81), positionLocal.x);
    const inPocketY = float(1.0).sub(smoothstep(float(0.60), float(0.67), abs(positionLocal.y.sub(0.01))));
    const inRebateWall = smoothstep(float(0.89), float(0.95), positionLocal.x);
    const chiselCutMask = clamp(
      inPocketX
        .mul(inPocketY)
        .mul(smoothstep(float(0.24), float(0.10), positionLocal.z))
        .add(inRebateWall.mul(0.52)),
      0.0,
      1.0
    );
    const chiseledTone = mix(burntSiennaRgb.mul(0.56), yellowOchreRgb.mul(0.46), 0.46).mul(
      mix(float(1.0), float(0.28), latewood)
    );
    const withMortiseWalls = mix(oiledTimber, chiseledTone, chiselCutMask.mul(0.44));

    const chamferLift = mix(withMortiseWalls, withMortiseWalls.mul(1.24), bevelMask.mul(0.52));
    const endGrainTone = chamferLift.mul(mix(float(1.0), float(0.88), endGrainMask));

    const deepPocketAO = mix(
      float(1.0),
      float(0.76),
      inPocketX.mul(inPocketY).mul(smoothstep(float(0.06), float(-0.22), positionLocal.z))
    );
    charredOak.colorNode = endGrainTone
      .mul(mix(float(1.05), float(0.32), vesselGroove))
      .mul(deepPocketAO);

    const normLongObj = perturbNormalAnisoObject(nObj, positionLocal, vec3(110.0, 6.2, 110.0), float(0.082));
    const normEndObj = perturbNormalAnisoObject(nObj, positionLocal, vec3(130.0, 130.0, 130.0), float(0.098));
    const ringSlope = sin(mix(warpedRingCoord, warpedEndCoord, endGrainMask).mul(6.283185)).mul(0.022);
    const pithDir = normalize(vec3(pithOffset.x, float(0.0), pithOffset.y));
    const ringTan = pithDir.sub(nObj.mul(dot(pithDir, nObj)));
    const blendedObjNorm = normalize(mix(normLongObj, normEndObj, endGrainMask).add(ringTan.mul(ringSlope)));
    charredOak.normalNode = transformNormalToView(blendedObjNorm);

    charredOak.roughnessNode = clamp(
      float(0.44)
        .add(latewood.mul(0.11))
        .add(vesselGroove.mul(0.34))
        .sub(rayAccent.mul(0.22))
        .sub(bevelMask.mul(0.08))
        .add(endGrainMask.mul(0.10))
        .sub(fineFibers.mul(0.05)),
      0.24,
      0.88
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Oil-Blackened Machined Gunmetal Steel Lock Case Interior & Cutaway Rim
  // ---------------------------------------------------------------------------
  const gunmetalCase = new THREE.MeshStandardNodeMaterial({
    metalness: 0.85,
    roughness: 0.34,
    envMapIntensity: 1.18,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);

    const fineMill = mineralToothFbm(positionLocal.mul(vec3(12.0, 210.0, 72.0)), float(1.15));
    const millArcR = length(vec2(positionLocal.x.add(0.15), positionLocal.y.sub(0.08)));
    const flyCutterArc = sin(millArcR.mul(165.0).add(fineMill.mul(1.8))).mul(0.5).add(0.5);
    const oilMottle = mineralToothFbm(positionLocal.add(vec3(3.1, 1.7, 5.4)), float(12.5));

    const steelDeep = mix(rawUmberRgb.mul(0.28), azuriteRgb.mul(0.34), 0.56);
    const steelMid = mix(rawUmberRgb.mul(0.44), azuriteRgb.mul(0.52), 0.54);
    const steelHighlight = mix(steelMid, boneCalciteRgb.mul(0.28), 0.42);
    const oilCrevice = mix(rawUmberRgb.mul(0.18), burntSiennaRgb.mul(0.14), 0.45);

    const millBlend = smoothstep(float(0.15), float(0.85), fineMill.mul(0.72).add(flyCutterArc.mul(0.28)));
    const baseSteel = mix(steelDeep, steelMid, millBlend.mul(0.65).add(0.22));
    const withOil = mix(baseSteel, oilCrevice, smoothstep(float(0.62), float(0.92), oilMottle).mul(0.26));
    const withBevelWear = mix(withOil, steelHighlight, bevelMask.mul(0.58));

    const backAO = smoothstep(float(-0.22), float(0.06), positionLocal.z);
    gunmetalCase.colorNode = withBevelWear.mul(mix(float(0.78), float(1.04), backAO));

    const millObjNorm = perturbNormalAnisoObject(nObj, positionLocal, vec3(14.0, 220.0, 76.0), float(0.017));
    const arcDir = normalize(vec3(positionLocal.x.add(0.15), positionLocal.y.sub(0.08), float(0.001)));
    const arcTan = arcDir.sub(nObj.mul(dot(arcDir, nObj)));
    const arcSlope = cos(millArcR.mul(165.0)).mul(0.009).mul(smoothstep(float(0.85), float(0.98), abs(nObj.z)));
    gunmetalCase.normalNode = transformNormalToView(normalize(millObjNorm.add(arcTan.mul(arcSlope))));

    gunmetalCase.roughnessNode = clamp(
      float(0.32)
        .add(fineMill.sub(0.5).mul(0.14))
        .add(oilMottle.sub(0.5).mul(0.09))
        .sub(bevelMask.mul(0.10)),
      0.20,
      0.52
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Heavy Thermally Blued-Steel Deadbolt, Detent Cams & Accents
  // ---------------------------------------------------------------------------
  const bluedSteelBolt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.38,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const noseFaceMask = smoothstep(float(0.55), float(0.85), abs(nObj.x));

    const drawSide = mineralToothFbm(positionLocal.mul(vec3(8.5, 285.0, 285.0)), float(1.15));
    const drawMicro = mineralToothFbm(positionLocal.mul(vec3(22.0, 560.0, 560.0)), float(1.1));
    const groundNose = mineralToothFbm(positionLocal.mul(vec3(54.0, 220.0, 220.0)), float(1.15));
    const activeGrain = mix(drawSide.mul(0.62).add(drawMicro.mul(0.38)), groundNose, noseFaceMask);
    const temperWave = mineralToothFbm(positionLocal.add(vec3(4.2, 8.1, 2.3)), float(7.8));

    const blueDeep = mix(azuriteRgb.mul(0.52), lapisRgb.mul(0.46), 0.42);
    const blueMid = mix(azuriteRgb.mul(0.94), boneCalciteRgb.mul(0.24), 0.26);
    const temperVioletStraw = mix(lapisRgb.mul(0.68), burntSiennaRgb.mul(0.32), 0.32);
    const brightSteel = mix(boneCalciteRgb.mul(0.66), warmKaolinRgb.mul(0.74), 0.42);

    const kmBlue = kubelkaMunkGlaze(
      pPrussian.K,
      pPrussian.S,
      float(0.44).add(temperWave.mul(0.24)),
      blueMid
    );
    const grainFactor = smoothstep(float(0.14), float(0.86), activeGrain).mul(0.48).add(0.28);
    const boltBase = mix(blueDeep, mix(blueMid, kmBlue, 0.36), grainFactor);
    const withTemper = mix(boltBase, temperVioletStraw, smoothstep(float(0.54), float(0.88), temperWave).mul(0.30));

    const railWearY = smoothstep(float(0.115), float(0.162), abs(positionLocal.y.sub(0.16))).mul(
      float(1.0).sub(noseFaceMask)
    );
    const rubThrough = clamp(
      bevelMask.mul(0.44).add(railWearY.mul(0.38).mul(smoothstep(float(0.22), float(0.82), drawSide))),
      0.0,
      0.62
    );
    const burnished = mix(withTemper, brightSteel, rubThrough);

    bluedSteelBolt.colorNode = burnished.add(brightSteel.mul(uBoltHover.mul(0.16)));

    const normSide = perturbNormalAnisoObject(nObj, positionLocal, vec3(9.0, 240.0, 240.0), float(0.017));
    const normNose = perturbNormalAnisoObject(nObj, positionLocal, vec3(64.0, 210.0, 210.0), float(0.014));
    bluedSteelBolt.normalNode = transformNormalToView(normalize(mix(normSide, normNose, noseFaceMask)));

    bluedSteelBolt.roughnessNode = clamp(
      float(0.21)
        .add(activeGrain.sub(0.5).mul(0.11))
        .sub(rubThrough.mul(0.08))
        .add(temperWave.sub(0.5).mul(0.04)),
      0.12,
      0.34
    );
  }

  // ---------------------------------------------------------------------------
  // 4. Precision Lathe-Turned Horological Gear Brass
  // ---------------------------------------------------------------------------
  const machinedGearBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.22,
    envMapIntensity: 1.38,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const faceZMask = smoothstep(float(0.75), float(0.96), abs(nObj.z));
    const rXY = length(positionLocal.xy);
    const angXY = atan(positionLocal.y, positionLocal.x);

    const latheFine = sin(rXY.mul(420.0)).mul(0.5).add(0.5);
    const latheMed = sin(rXY.mul(175.0).add(mineralToothFbm(positionLocal, float(24.0)).mul(1.8))).mul(0.5).add(0.5);
    const latheWave = latheFine.mul(0.58).add(latheMed.mul(0.42));
    const patinaMottle = mineralToothFbm(positionLocal.add(vec3(5.3, 2.7, 1.9)), float(26.0));
    const microFile = mineralToothFbm(positionLocal.mul(vec3(110.0, 110.0, 22.0)), float(1.15));

    const rootRecessMask = smoothstep(float(0.125), float(0.042), rXY).mul(
      smoothstep(float(0.32), float(0.84), patinaMottle)
    );
    const tipBurnishMask = clamp(
      bevelMask.mul(0.78).add(smoothstep(float(0.095), float(0.145), rXY).mul(0.42)),
      0.0,
      1.0
    );

    const agedUmberBrass = mix(rawUmberRgb.mul(0.52), burntSiennaRgb.mul(0.44), 0.35);
    const patinaVerdigris = mix(agedUmberBrass, verdigrisRgb.mul(0.48), 0.38);
    const satinBrass = mix(yellowOchreRgb.mul(0.98), boneCalciteRgb.mul(0.78), 0.26);
    const brightBurnished = mix(yellowOchreRgb.mul(1.24), boneCalciteRgb.mul(0.96), 0.44);
    const kmBrass = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.48), satinBrass);

    const baseGear = mix(
      mix(agedUmberBrass, satinBrass, 0.56),
      mix(satinBrass, kmBrass, 0.22),
      latheWave.mul(0.55).add(microFile.mul(0.25)).add(0.20)
    );
    const withOilPatina = mix(baseGear, agedUmberBrass, rootRecessMask.mul(0.48));
    const creviceVerdigris = smoothstep(float(0.74), float(0.94), patinaMottle).mul(rootRecessMask);
    const withAge = mix(withOilPatina, patinaVerdigris, creviceVerdigris.mul(0.32));
    machinedGearBrass.colorNode = mix(withAge, brightBurnished, tipBurnishMask.mul(0.46));

    const baseMicroNorm = perturbNormalAnisoObject(nObj, positionLocal, vec3(92.0, 92.0, 26.0), float(0.014));
    const safeR = max(rXY, float(0.002));
    const radialDir = vec3(positionLocal.x.div(safeR), positionLocal.y.div(safeR), float(0.0));
    const radialTan = radialDir.sub(nObj.mul(dot(radialDir, nObj)));
    const latheSlope = cos(rXY.mul(420.0))
      .mul(0.022)
      .add(cos(rXY.mul(175.0)).mul(0.011))
      .add(sin(angXY.mul(2.0)).mul(0.004))
      .mul(faceZMask);
    machinedGearBrass.normalNode = transformNormalToView(
      normalize(baseMicroNorm.add(radialTan.mul(latheSlope)))
    );

    machinedGearBrass.roughnessNode = clamp(
      float(0.21)
        .add(latheWave.sub(0.5).mul(0.11))
        .add(rootRecessMask.mul(0.18))
        .add(creviceVerdigris.mul(0.20))
        .sub(tipBurnishMask.mul(0.09)),
      0.12,
      0.48
    );
  }

  // ---------------------------------------------------------------------------
  // 5. Precision-Milled Aged Dark Horological Bronze (Rack, C1 & Rear Pinions)
  // ---------------------------------------------------------------------------
  const machinedBronzeRack = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.24,
    envMapIntensity: 1.34,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const millGrain = mineralToothFbm(positionLocal.mul(vec3(11.0, 250.0, 250.0)), float(1.15));
    const oilPatina = mineralToothFbm(positionLocal.add(vec3(1.7, 6.4, 3.2)), float(24.0));

    const bronzeRoot = mix(rawUmberRgb.mul(0.42), burntSiennaRgb.mul(0.36), 0.36);
    const bronzeMid = mix(rawUmberRgb.mul(0.76), yellowOchreRgb.mul(0.72), 0.48);
    const bronzeTip = mix(yellowOchreRgb.mul(1.10), boneCalciteRgb.mul(0.84), 0.34);
    const kmBronze = kubelkaMunkGlaze(pUmber.K, pUmber.S, float(0.38), bronzeMid);

    const milled = mix(
      bronzeRoot,
      mix(bronzeMid, kmBronze, 0.26),
      smoothstep(float(0.16), float(0.84), millGrain).mul(0.54).add(0.28)
    );
    const patinated = mix(milled, bronzeRoot, smoothstep(float(0.58), float(0.90), oilPatina).mul(0.38));
    machinedBronzeRack.colorNode = mix(patinated, bronzeTip, bevelMask.mul(0.56));

    machinedBronzeRack.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(13.0, 260.0, 260.0),
      float(0.015)
    );
    machinedBronzeRack.roughnessNode = clamp(
      float(0.23)
        .add(millGrain.sub(0.5).mul(0.10))
        .add(oilPatina.sub(0.5).mul(0.10))
        .sub(bevelMask.mul(0.09)),
      0.13,
      0.44
    );
  }

  // ---------------------------------------------------------------------------
  // 6. Mortised Forend Satin Brass Faceplate, Glazing Trim, Rosette & Screws
  // ---------------------------------------------------------------------------
  const forendBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.23,
    envMapIntensity: 1.32,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const brushY = mineralToothFbm(positionLocal.mul(vec3(240.0, 10.5, 240.0)), float(1.15));
    const fineSatin = mineralToothFbm(positionLocal.add(vec3(6.3, 2.1, 4.7)), float(42.0));

    const brassAged = mix(rawUmberRgb.mul(0.56), yellowOchreRgb.mul(0.68), 0.42);
    const brassMid = mix(yellowOchreRgb.mul(1.06), boneCalciteRgb.mul(0.82), 0.28);
    const brassBright = mix(yellowOchreRgb.mul(1.22), boneCalciteRgb.mul(0.94), 0.42);
    const kmBrass = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.52), brassMid);

    const grainMix = smoothstep(float(0.16), float(0.84), brushY.mul(0.68).add(fineSatin.mul(0.32)))
      .mul(0.54)
      .add(0.28);
    const base = mix(brassAged, mix(brassMid, kmBrass, 0.20), grainMix);
    const withAge = mix(base, brassAged, smoothstep(float(0.66), float(0.92), fineSatin).mul(0.24));
    forendBrass.colorNode = mix(withAge, brassBright, bevelMask.mul(0.42));

    forendBrass.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(260.0, 11.0, 260.0),
      float(0.014)
    );
    forendBrass.roughnessNode = clamp(
      float(0.22)
        .add(brushY.sub(0.5).mul(0.11))
        .add(fineSatin.sub(0.5).mul(0.06))
        .sub(bevelMask.mul(0.07)),
      0.14,
      0.38
    );
  }

  // ---------------------------------------------------------------------------
  // 7. Tempered Phosphor-Bronze Curved Leaf Springs & Click Springs
  // ---------------------------------------------------------------------------
  const phosphorBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.89,
    roughness: 0.21,
    envMapIntensity: 1.36,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const rollGrain = mineralToothFbm(positionLocal.mul(vec3(12.0, 195.0, 195.0)), float(1.15));
    const heatBand = mineralToothFbm(positionLocal.add(vec3(3.7, 1.2, 8.4)), float(16.0));

    const temperDark = mix(burntSiennaRgb.mul(0.62), hematiteRgb.mul(0.50), 0.34);
    const temperBright = mix(realgarAmberRgb.mul(1.02), yellowOchreRgb.mul(1.04), 0.44);
    const kmBronze = kubelkaMunkGlaze(pSienna.K, pSienna.S, float(0.42), temperBright);

    const base = mix(
      temperDark,
      mix(temperBright, kmBronze, 0.24),
      smoothstep(float(0.16), float(0.84), rollGrain).mul(0.54).add(0.28)
    );
    const withHeat = mix(base, temperDark, smoothstep(float(0.60), float(0.90), heatBand).mul(0.24));
    phosphorBronze.colorNode = mix(withHeat, temperBright.mul(1.12), bevelMask.mul(0.36));

    phosphorBronze.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(14.0, 215.0, 215.0),
      float(0.013)
    );
    phosphorBronze.roughnessNode = clamp(
      float(0.19)
        .add(rollGrain.sub(0.5).mul(0.09))
        .add(heatBand.sub(0.5).mul(0.06))
        .sub(bevelMask.mul(0.05)),
      0.13,
      0.34
    );
  }

  // ---------------------------------------------------------------------------
  // 8. Architectural Hand-Rubbed Antique Brass Deadbolt Thumb-Turn Knob & Cam Lobes
  // ---------------------------------------------------------------------------
  const keyBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.20,
    envMapIntensity: 1.40,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const handRub = mineralToothFbm(positionLocal, float(34.0));
    const fineBuff = mineralToothFbm(positionLocal.add(vec3(2.9, 7.1, 1.3)), float(115.0));

    const antiqueDeep = mix(rawUmberRgb.mul(0.52), burntSiennaRgb.mul(0.46), 0.28);
    const antiqueMid = mix(yellowOchreRgb.mul(0.98), rawUmberRgb.mul(0.66), 0.14);
    const burnishedGold = mix(yellowOchreRgb.mul(1.26), boneCalciteRgb.mul(0.95), 0.44);

    const rubMix = smoothstep(float(0.16), float(0.84), handRub.mul(0.55).add(fineBuff.mul(0.45)))
      .mul(0.52)
      .add(0.28);
    const patinated = mix(antiqueDeep, antiqueMid, rubMix);
    const withEdgeRub = mix(patinated, burnishedGold, clamp(bevelMask.mul(0.64), 0.0, 0.72));

    keyBrass.colorNode = withEdgeRub.add(burnishedGold.mul(uKnobHover.mul(0.18)));
    keyBrass.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(96.0, 96.0, 96.0),
      float(0.009)
    );
    keyBrass.roughnessNode = clamp(
      float(0.20)
        .add(handRub.sub(0.5).mul(0.10))
        .sub(bevelMask.mul(0.08))
        .sub(uKnobHover.mul(0.04)),
      0.11,
      0.34
    );
  }

  // ---------------------------------------------------------------------------
  // 9. Lathe-Turned Hardened Tool Steel (Arbors, Pawls, Bolt Stump, Pivot Post, Shafts)
  // ---------------------------------------------------------------------------
  const hardenedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.94,
    roughness: 0.19,
    envMapIntensity: 1.42,
  });
  {
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const rXY = length(positionLocal.xy);
    const faceZMask = smoothstep(float(0.70), float(0.95), abs(nObj.z));

    const latheRadial = sin(rXY.mul(520.0)).mul(0.5).add(0.5);
    const latheAxial = sin(positionLocal.z.mul(720.0)).mul(0.5).add(0.5);
    const latheGroove = mix(latheAxial.mul(0.32).add(0.34), latheRadial, faceZMask);
    const fineTurn = mineralToothFbm(positionLocal.mul(vec3(68.0, 68.0, 290.0)), float(1.15));
    const oilFilm = mineralToothFbm(positionLocal.add(vec3(7.1, 3.4, 2.8)), float(19.0));

    const steelOil = mix(rawUmberRgb.mul(0.32), azuriteRgb.mul(0.40), 0.42);
    const steelCool = mix(boneCalciteRgb.mul(0.48), azuriteRgb.mul(0.50), 0.24);
    const steelBright = mix(boneCalciteRgb.mul(0.84), warmKaolinRgb.mul(0.92), 0.44);

    const turned = mix(
      steelCool,
      steelBright,
      smoothstep(float(0.16), float(0.84), fineTurn.mul(0.76).add(latheGroove.mul(0.24))).mul(0.52).add(0.30)
    );
    const withOil = mix(turned, steelOil, smoothstep(float(0.66), float(0.94), oilFilm).mul(0.24));
    const withChamfer = mix(withOil, steelBright.mul(1.06), bevelMask.mul(0.46));

    hardenedSteel.colorNode = withChamfer.add(steelBright.mul(uBoltHover.mul(0.10)));

    const microObj = perturbNormalAnisoObject(nObj, positionLocal, vec3(72.0, 72.0, 280.0), float(0.011));
    const safeR = max(rXY, float(0.002));
    const radialDir = vec3(positionLocal.x.div(safeR), positionLocal.y.div(safeR), float(0.0));
    const radialTan = radialDir.sub(nObj.mul(dot(radialDir, nObj)));
    const radialBump = cos(rXY.mul(520.0)).mul(0.014).mul(faceZMask);
    hardenedSteel.normalNode = transformNormalToView(normalize(microObj.add(radialTan.mul(radialBump))));

    hardenedSteel.roughnessNode = clamp(
      float(0.18)
        .add(latheGroove.sub(0.5).mul(0.05))
        .add(oilFilm.sub(0.5).mul(0.07))
        .sub(bevelMask.mul(0.06)),
      0.11,
      0.32
    );
  }

  // ---------------------------------------------------------------------------
  // 10. Five Distinctly Patinated Hand-Filed Satin-Brass Chubb Lever Tumbler Plates
  // ---------------------------------------------------------------------------
  const leverMaterials: THREE.MeshStandardNodeMaterial[] = [];
  const leverToneScale = [0.76, 0.96, 0.83, 1.04, 0.91];
  const leverWarmShift = [0.30, 0.08, 0.22, 0.04, 0.14];

  for (let i = 0; i < LEVER_COUNT; i++) {
    const uHover = uniform(0);
    const uBind = uniform(0);
    uLeverHovers.push(uHover);
    uLeverBinds.push(uBind);

    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.91,
      roughness: 0.22,
      envMapIntensity: 1.34,
    });

    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const seedOff = vec3(float(i * 3.71 + 1.1), float(i * 2.13 + 0.7), float(i * 4.37));

    const brushX = mineralToothFbm(positionLocal.add(seedOff).mul(vec3(10.0, 280.0, 128.0)), float(1.15));
    const microFile = mineralToothFbm(positionLocal.add(seedOff.mul(1.3)).mul(vec3(26.0, 560.0, 210.0)), float(1.1));
    const ageMottle = mineralToothFbm(positionLocal.add(seedOff.mul(1.7)), float(22.0));

    const tone = float(leverToneScale[i]);
    const warmMix = float(leverWarmShift[i]);

    const brassOilPatina = mix(rawUmberRgb.mul(0.48), burntSiennaRgb.mul(0.44), warmMix).mul(tone);
    const brassVerdigris = mix(brassOilPatina, verdigrisRgb.mul(0.44), 0.34);
    const brassMid = mix(
      yellowOchreRgb.mul(1.06),
      boneCalciteRgb.mul(0.84),
      float(0.30).sub(warmMix.mul(0.40))
    ).mul(tone);
    const brassBurnish = mix(yellowOchreRgb.mul(1.24), boneCalciteRgb.mul(0.95), 0.44).mul(
      mix(float(1.0), tone, 0.45)
    );

    const kmLever = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.48), brassMid);
    const grainMix = smoothstep(float(0.16), float(0.84), brushX.mul(0.64).add(microFile.mul(0.36)))
      .mul(0.50)
      .add(0.28);
    const brushed = mix(brassOilPatina, mix(brassMid, kmLever, 0.18), grainMix);

    const gatePocketMask = smoothstep(float(0.40), float(0.52), positionLocal.x)
      .mul(float(1.0).sub(smoothstep(float(0.86), float(0.98), positionLocal.x)))
      .mul(float(1.0).sub(smoothstep(float(0.14), float(0.22), abs(positionLocal.y.sub(0.04)))));
    const oilSpot = smoothstep(float(0.56), float(0.88), ageMottle).mul(float(0.35).add(gatePocketMask.mul(0.65)));
    const aged = mix(brushed, brassOilPatina, oilSpot.mul(0.42));
    const withVerdigris = mix(
      aged,
      brassVerdigris,
      smoothstep(float(0.76), float(0.95), ageMottle).mul(gatePocketMask).mul(0.28)
    );

    const bellyWear = smoothstep(float(-0.08), float(-0.15), positionLocal.y);
    const crestWear = smoothstep(float(0.22), float(0.262), positionLocal.y);
    const wearMask = clamp(bevelMask.mul(0.54).add(bellyWear.mul(0.42)).add(crestWear.mul(0.38)), 0.0, 0.72);
    const finished = mix(withVerdigris, brassBurnish, wearMask);

    const bindTint = mix(realgarAmberRgb, yellowOchreRgb, 0.35).mul(0.32);
    mat.colorNode = finished
      .add(brassMid.mul(uHover.mul(0.22)))
      .add(bindTint.mul(uBind));

    mat.normalNode = perturbNormalAniso(
      nObj,
      positionLocal.add(seedOff),
      vec3(11.0, 240.0, 112.0),
      float(0.015)
    );
    mat.roughnessNode = clamp(
      float(0.21)
        .add(brushX.sub(0.5).mul(0.10))
        .add(oilSpot.mul(0.16))
        .sub(wearMask.mul(0.09))
        .sub(uHover.mul(0.04)),
      0.12,
      0.44
    );

    leverMaterials.push(mat);
  }

  return {
    charredOak,
    gunmetalCase,
    bluedSteelBolt,
    forendBrass,
    machinedGearBrass,
    machinedBronzeRack,
    phosphorBronze,
    keyBrass,
    hardenedSteel,
    leverMaterials,
    uKnobHover,
    uKeyHover: uKnobHover,
    uBoltHover,
    uLeverHovers,
    uLeverBinds,
    dispose: () => {
      charredOak.dispose();
      gunmetalCase.dispose();
      bluedSteelBolt.dispose();
      forendBrass.dispose();
      machinedGearBrass.dispose();
      machinedBronzeRack.dispose();
      phosphorBronze.dispose();
      keyBrass.dispose();
      hardenedSteel.dispose();
      for (const m of leverMaterials) m.dispose();
    },
  };
}
