import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  atan,
  clamp,
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
  phosphorBronze: THREE.MeshStandardNodeMaterial;
  keyBrass: THREE.MeshStandardNodeMaterial;
  hardenedSteel: THREE.MeshStandardNodeMaterial;
  leverMaterials: THREE.MeshStandardNodeMaterial[];
  uKeyHover: { value: number };
  uBoltHover: { value: number };
  uLeverHovers: Array<{ value: number }>;
  uLeverBinds: Array<{ value: number }>;
  dispose: () => void;
}

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Normalized independent of `scaleVec` so ultra-fine machined brush streaks stay
 * crisp and razor-flat (zero low-frequency crumpled-foil blotches on flat metal plates).
 */
const perturbNormalAniso = Fn(
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
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

export function createBoltMaterials(): BoltMaterials {
  const uKeyHover = uniform(0);
  const uBoltHover = uniform(0);
  const uLeverHovers: Array<{ value: number }> = [];
  const uLeverBinds: Array<{ value: number }> = [];

  const pUmber = createPigmentUniforms('rawUmber');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pSienna = createPigmentUniforms('burntSienna');
  const pPrussian = createPigmentUniforms('azuritePrussian');

  // Strictly canonical physical pigments from src/core/palette.ts (zero hardcoded RGBs)
  const rawUmberRgb = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const burntSiennaRgb = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const yellowOchreRgb = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const realgarAmberRgb = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const boneCalciteRgb = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const warmKaolinRgb = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const azuriteRgb = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const lapisRgb = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);
  const hematiteRgb = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Oiled Charred-Oak Architectural Door-Stile Specimen Block
  //    True cylindrical quarter-sawn oak cathedral growth rings + medullary rays
  //    on longitudinal faces (|N.x|, |N.z|) AND tight concentric cross-cut end-grain
  //    with open pin-pores on horizontal cuts (|N.y|)!
  // ---------------------------------------------------------------------------
  const charredOak = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.52,
    envMapIntensity: 0.88,
  });
  {
    const nObj = normalize(normalLocal);
    // Horizontal cross-cut faces (top/bottom stile ends & horizontal mortise shelves) show end-grain
    const endGrainMask = smoothstep(float(0.52), float(0.86), abs(nObj.y));

    // Offset tree pith in XZ creates authentic tight quarter-sawn & gentle cathedral arch rings along Y
    const pithOffset = vec2(positionLocal.x.add(2.25), positionLocal.z.add(1.65));
    const gentleBow = positionLocal.y.mul(positionLocal.y).mul(0.085);
    const fiberWander = mineralToothFbm(positionLocal.mul(vec3(2.4, 0.42, 2.4)), float(1.4));
    const fineWander = mineralToothFbm(positionLocal.mul(vec3(7.5, 1.3, 7.5)), float(1.2));

    // Radial ring coordinate around pith (tight realistic ring spacing, subtle organic wander -> zero camo blobs!)
    const rPith = length(pithOffset)
      .add(gentleBow)
      .add(fiberWander.sub(0.5).mul(0.16))
      .add(fineWander.sub(0.5).mul(0.035));

    // Asymmetric oak growth rings: thin dark scorched latewood lines separating warm oiled earlywood bands
    const ringCoordLong = fract(rPith.mul(14.5).add(10.0));
    const dLong = abs(ringCoordLong.sub(0.5)).mul(2.0);
    const latewoodLong = smoothstep(float(0.52), float(0.95), dLong.mul(dLong));

    const ringCoordEnd = fract(rPith.mul(18.5).add(10.0));
    const dEnd = abs(ringCoordEnd.sub(0.5)).mul(2.0);
    const latewoodEnd = smoothstep(float(0.46), float(0.94), dEnd.mul(dEnd));
    const latewood = mix(latewoodLong, latewoodEnd, endGrainMask);

    // Fine elongated vertical oak vessel pores (longitudinal) vs dense circular end-grain pin pores
    const longPores = mineralToothFbm(positionLocal.mul(vec3(78.0, 5.2, 78.0)), float(1.3));
    const endPores = mineralToothFbm(positionLocal.mul(vec3(96.0, 96.0, 96.0)), float(1.2));
    const microTooth = mineralToothFbm(positionLocal.mul(vec3(120.0, 18.0, 120.0)), float(1.1));
    const activePores = mix(longPores, endPores, endGrainMask);

    // Subtle quarter-sawn medullary rays radiating from tree pith
    const rayAngle = atan(pithOffset.y, pithOffset.x);
    const rayFleckLong = smoothstep(
      float(0.68),
      float(0.92),
      mineralToothFbm(vec3(rayAngle.mul(42.0), positionLocal.y.mul(14.0), rPith.mul(6.0)), float(1.0))
    );
    const rayLinesEnd = sin(rayAngle.mul(110.0).add(fiberWander.mul(2.0))).mul(0.5).add(0.5);
    const rayAccent = mix(rayFleckLong.mul(0.16), rayLinesEnd.mul(0.22), endGrainMask);

    // Rich linseed-oiled charred raw-umber palette (earlywood warm umber-sienna vs scorched latewood)
    const earlyWood = mix(rawUmberRgb.mul(0.30), burntSiennaRgb.mul(0.22), float(0.32).add(rayAccent));
    const lateWood = rawUmberRgb.mul(0.055);
    const woodBase = mix(earlyWood, lateWood, latewood.mul(0.84));

    // Kubelka-Munk oiled raw-umber glaze for deep museum-varnished timber saturation
    const glazeThick = float(0.34).add(activePores.mul(0.40));
    const oiledTimber = kubelkaMunkGlaze(pUmber.K, pUmber.S, glazeThick, woodBase).mul(0.33);

    // Subtle warm heartwood lift on front chamfered cutaway face & darker end-grain oil absorption
    const frontCutWarmth = smoothstep(float(0.12), float(0.25), positionLocal.z).mul(float(1.0).sub(endGrainMask));
    const warmCut = mix(rawUmberRgb.mul(0.38), burntSiennaRgb.mul(0.28), float(1.0).sub(latewood));
    const withFrontLift = mix(oiledTimber, warmCut.mul(0.48), frontCutWarmth.mul(0.28));
    const withEndDarken = withFrontLift.mul(mix(float(1.0), float(0.74), endGrainMask));

    // Crisp dark open-pore valleys + gentle interior mortise pocket shadow
    const poreMask = smoothstep(float(0.18), float(0.54), activePores);
    const inPocketX = smoothstep(float(-0.88), float(-0.82), positionLocal.x);
    const inPocketY = float(1.0).sub(smoothstep(float(0.60), float(0.66), abs(positionLocal.y.sub(0.01))));
    const pocketAO = mix(float(1.0), float(0.76), inPocketX.mul(inPocketY).mul(smoothstep(float(0.20), float(-0.18), positionLocal.z)));

    charredOak.colorNode = withEndDarken.mul(mix(float(0.60), float(1.05), poreMask)).mul(pocketAO);

    // Blend longitudinal fiber-grain normal relief with cross-cut end-grain pore relief (unconditional uniform flow)
    const normLong = perturbNormalAniso(nObj, positionLocal, vec3(76.0, 6.5, 76.0), float(0.022));
    const normEnd = perturbNormalAniso(nObj, positionLocal, vec3(92.0, 92.0, 92.0), float(0.032));
    charredOak.normalNode = normalize(mix(normLong, normEnd, endGrainMask));

    charredOak.roughnessNode = clamp(
      float(0.42)
        .add(latewood.mul(0.12))
        .add(float(1.0).sub(poreMask).mul(0.18))
        .sub(rayAccent.mul(0.18))
        .sub(microTooth.mul(0.04))
        .add(endGrainMask.mul(0.15))
        .sub(frontCutWarmth.mul(0.05)),
      0.34,
      0.84
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Oil-Blackened Surface-Ground Gunmetal Steel Lock Case Interior
  //    Fine high-frequency machining grain (zero blurry zebra stripes!)
  // ---------------------------------------------------------------------------
  const gunmetalCase = new THREE.MeshStandardNodeMaterial({
    metalness: 0.82,
    roughness: 0.38,
    envMapIntensity: 1.06,
  });
  {
    // Ultra-fine machined surface-grinder finish along X + subtle oil-blackened temper variation
    const fineMill = mineralToothFbm(positionLocal.mul(vec3(14.0, 210.0, 68.0)), float(1.1));
    const oilMottle = mineralToothFbm(positionLocal.add(vec3(3.1, 1.7, 5.4)), float(14.0));

    // Strictly derived from azuritePrussian, rawUmber & boneCalcite (zero hardcoded RGBs)
    const steelDeep = mix(rawUmberRgb.mul(0.22), azuriteRgb.mul(0.26), 0.58);
    const steelMid = mix(rawUmberRgb.mul(0.34), azuriteRgb.mul(0.40), 0.56);
    const steelHighlight = mix(steelMid, boneCalciteRgb.mul(0.14), 0.28);

    // Subtle tonal modulation (fine micro-grain, never loud zebra banding)
    const baseSteel = mix(steelDeep, steelMid, smoothstep(float(0.18), float(0.84), fineMill).mul(0.55).add(0.22));
    const patinated = mix(baseSteel, steelHighlight, smoothstep(float(0.55), float(0.90), oilMottle).mul(0.32));

    // Gentle recessed case depth AO toward backplate so stacked golden levers pop vividly forward
    const backAO = smoothstep(float(-0.22), float(0.06), positionLocal.z);
    gunmetalCase.colorNode = patinated.mul(mix(float(0.68), float(1.0), backAO));
    gunmetalCase.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(16.0, 220.0, 72.0),
      float(0.0055)
    );
    gunmetalCase.roughnessNode = clamp(
      float(0.34)
        .add(fineMill.sub(0.5).mul(0.10))
        .add(oilMottle.sub(0.5).mul(0.07)),
      0.26,
      0.50
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Heavy Thermally Blued-Steel Deadbolt (Prussian-Blued Temper Steel)
  //    Fine draw-filed X grain on side faces + crisp cross-ground nose face (zero blotches!)
  // ---------------------------------------------------------------------------
  const bluedSteelBolt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.24,
    envMapIntensity: 1.32,
  });
  {
    const nObj = normalize(normalLocal);
    const noseFaceMask = smoothstep(float(0.55), float(0.85), abs(nObj.x));

    // High-frequency draw-filed satin grain on sides vs fine cross-ground satin on +X nose face
    const drawSide = mineralToothFbm(positionLocal.mul(vec3(9.0, 220.0, 220.0)), float(1.1));
    const groundNose = mineralToothFbm(positionLocal.mul(vec3(48.0, 180.0, 180.0)), float(1.1));
    const activeGrain = mix(drawSide, groundNose, noseFaceMask);
    const temperWave = mineralToothFbm(positionLocal.add(vec3(4.2, 8.1, 2.3)), float(9.5));

    // Rich Prussian-blued temper steel anchored strictly in azuritePrussian, lapisUltramarine & boneCalcite
    const blueDeep = mix(azuriteRgb.mul(0.58), lapisRgb.mul(0.52), 0.35);
    const blueMid = mix(azuriteRgb.mul(0.92), boneCalciteRgb.mul(0.24), 0.28);
    const blueSheen = mix(azuriteRgb.mul(1.12), warmKaolinRgb.mul(0.36), 0.38);

    const kmBlue = kubelkaMunkGlaze(
      pPrussian.K,
      pPrussian.S,
      float(0.46).add(temperWave.mul(0.22)),
      blueSheen
    );
    // Low-contrast micro-brushed albedo + subtle thermal temper iridescence (zero thick horizontal stripes!)
    const grainFactor = smoothstep(float(0.15), float(0.85), activeGrain).mul(0.38).add(0.32);
    const boltBase = mix(blueDeep, mix(blueMid, kmBlue, 0.36), grainFactor);
    const withTemper = mix(boltBase, blueSheen, smoothstep(float(0.58), float(0.88), temperWave).mul(0.24));

    // Subtle burnished sliding-wear sheen along top/bottom faces passing through the brass forend slot
    const wearBand = float(1.0)
      .sub(smoothstep(float(0.09), float(0.17), abs(positionLocal.y.sub(0.16))))
      .mul(float(1.0).sub(noseFaceMask));
    const burnished = mix(withTemper, mix(withTemper, boneCalciteRgb.mul(0.46), 0.20), wearBand.mul(0.28));

    bluedSteelBolt.colorNode = burnished.add(blueSheen.mul(uBoltHover.mul(0.20)));

    const normSide = perturbNormalAniso(nObj, positionLocal, vec3(10.0, 240.0, 240.0), float(0.0046));
    const normNose = perturbNormalAniso(nObj, positionLocal, vec3(64.0, 200.0, 200.0), float(0.0038));
    bluedSteelBolt.normalNode = normalize(mix(normSide, normNose, noseFaceMask));

    bluedSteelBolt.roughnessNode = clamp(
      float(0.21)
        .add(activeGrain.sub(0.5).mul(0.08))
        .sub(wearBand.mul(0.04))
        .add(temperWave.sub(0.5).mul(0.03)),
      0.15,
      0.32
    );
  }

  // ---------------------------------------------------------------------------
  // 4. Mortised Forend Satin Brass Faceplate, Escutcheon, Washers & Slotted Screws
  //    Fine directional vertical satin brushing along Y (zero coarse blurry stripes!)
  // ---------------------------------------------------------------------------
  const forendBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.24,
    envMapIntensity: 1.26,
  });
  {
    const brushY = mineralToothFbm(positionLocal.mul(vec3(240.0, 11.0, 240.0)), float(1.1));
    const fineSatin = mineralToothFbm(positionLocal.add(vec3(6.3, 2.1, 4.7)), float(64.0));

    const brassDark = mix(yellowOchreRgb.mul(0.76), rawUmberRgb.mul(0.68), 0.22);
    const brassBright = mix(yellowOchreRgb.mul(1.14), boneCalciteRgb.mul(0.86), 0.34);
    const kmBrass = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.56), brassBright);

    // Subtle satin micro-grain contrast driven primarily by normal + roughness anisotropy
    const grainMix = smoothstep(float(0.18), float(0.84), brushY.mul(0.65).add(fineSatin.mul(0.35))).mul(0.45).add(0.30);
    forendBrass.colorNode = mix(brassDark, mix(brassBright, kmBrass, 0.22), grainMix);
    forendBrass.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(260.0, 12.0, 260.0),
      float(0.0040)
    );
    forendBrass.roughnessNode = clamp(
      float(0.22)
        .add(brushY.sub(0.5).mul(0.08))
        .add(fineSatin.sub(0.5).mul(0.03)),
      0.17,
      0.33
    );
  }

  // ---------------------------------------------------------------------------
  // 5. Tempered Phosphor-Bronze Curved Leaf Springs (Warm Rolled Coppery-Amber Sheen)
  // ---------------------------------------------------------------------------
  const phosphorBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.21,
    envMapIntensity: 1.32,
  });
  {
    const rollGrain = mineralToothFbm(positionLocal.mul(vec3(14.0, 180.0, 180.0)), float(1.1));
    const temperDark = mix(burntSiennaRgb.mul(0.74), hematiteRgb.mul(0.56), 0.28);
    const temperBright = mix(realgarAmberRgb.mul(0.96), yellowOchreRgb.mul(0.98), 0.42);
    const kmBronze = kubelkaMunkGlaze(pSienna.K, pSienna.S, float(0.42), temperBright);

    phosphorBronze.colorNode = mix(
      temperDark,
      mix(temperBright, kmBronze, 0.26),
      smoothstep(float(0.18), float(0.84), rollGrain).mul(0.48).add(0.28)
    );
    phosphorBronze.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(16.0, 210.0, 210.0),
      float(0.0036)
    );
    phosphorBronze.roughnessNode = clamp(float(0.19).add(rollGrain.sub(0.5).mul(0.07)), 0.15, 0.29);
  }

  // ---------------------------------------------------------------------------
  // 6. Ornate Victorian Key Hand-Buffed Satin Brass Bow & Stepped Bitting Teeth
  // ---------------------------------------------------------------------------
  const keyBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.20,
    envMapIntensity: 1.34,
  });
  {
    const handRub = mineralToothFbm(positionLocal, float(46.0));
    const fineBuff = mineralToothFbm(positionLocal.add(vec3(2.9, 7.1, 1.3)), float(120.0));
    const antiqueDeep = mix(yellowOchreRgb.mul(0.72), rawUmberRgb.mul(0.60), 0.24);
    const antiqueGold = mix(yellowOchreRgb.mul(1.22), boneCalciteRgb.mul(0.90), 0.36);

    const base = mix(
      antiqueDeep,
      antiqueGold,
      smoothstep(float(0.18), float(0.82), handRub.mul(0.55).add(fineBuff.mul(0.45))).mul(0.48).add(0.28)
    );
    keyBrass.colorNode = base.add(antiqueGold.mul(uKeyHover.mul(0.18)));
    keyBrass.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(96.0, 96.0, 96.0),
      float(0.0024)
    );
    keyBrass.roughnessNode = clamp(
      float(0.18).add(handRub.sub(0.5).mul(0.06)).sub(uKeyHover.mul(0.04)),
      0.13,
      0.28
    );
  }

  // ---------------------------------------------------------------------------
  // 7. Bright Lathe-Turned Hardened Tool Steel (Bolt Stump, Pivot Post, Pins, Key Pipe)
  //    Strictly palette-derived (boneCalcite + warmKaolin + azuritePrussian)
  // ---------------------------------------------------------------------------
  const hardenedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.93,
    roughness: 0.20,
    envMapIntensity: 1.36,
  });
  {
    const latheTurn = mineralToothFbm(positionLocal.mul(vec3(54.0, 54.0, 220.0)), float(1.1));
    const steelCool = mix(boneCalciteRgb.mul(0.46), azuriteRgb.mul(0.48), 0.26);
    const steelBright = mix(boneCalciteRgb.mul(0.76), warmKaolinRgb.mul(0.84), 0.40);

    hardenedSteel.colorNode = mix(
      steelCool,
      steelBright,
      smoothstep(float(0.20), float(0.82), latheTurn).mul(0.46).add(0.32)
    ).add(steelBright.mul(uBoltHover.mul(0.10)));
    hardenedSteel.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(64.0, 64.0, 240.0),
      float(0.0032)
    );
    hardenedSteel.roughnessNode = clamp(float(0.18).add(latheTurn.sub(0.5).mul(0.07)), 0.13, 0.27);
  }

  // ---------------------------------------------------------------------------
  // 8. Five Distinctly Patinated Fine Hand-Filed Satin-Brass Lever Tumbler Plates (lever[0..4])
  //    Subtle alloy/age shifts make all 5 stacked layers & anti-pick serrations read razor-clear!
  // ---------------------------------------------------------------------------
  const leverMaterials: THREE.MeshStandardNodeMaterial[] = [];
  // Individual alloy/patina shifts across the 5 stacked plates
  const leverToneScale = [0.78, 0.96, 0.84, 1.03, 0.92];
  const leverWarmShift = [0.26, 0.07, 0.19, 0.04, 0.12];

  for (let i = 0; i < LEVER_COUNT; i++) {
    const uHover = uniform(0);
    const uBind = uniform(0);
    uLeverHovers.push(uHover);
    uLeverBinds.push(uBind);

    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.91,
      roughness: 0.22,
      envMapIntensity: 1.30,
    });

    const seedOff = vec3(float(i * 3.71 + 1.1), float(i * 2.13 + 0.7), float(i * 4.37));
    // Ultra-fine horizontal hand-filed satin brushing along X (zero blotchy streaks!)
    const brushX = mineralToothFbm(positionLocal.add(seedOff).mul(vec3(11.0, 220.0, 96.0)), float(1.1));
    const microFile = mineralToothFbm(positionLocal.add(seedOff.mul(1.7)), float(88.0));

    const tone = float(leverToneScale[i]);
    const warmMix = float(leverWarmShift[i]);

    const brassShadow = mix(
      yellowOchreRgb.mul(0.66),
      burntSiennaRgb.mul(0.60),
      warmMix
    ).mul(tone);
    const brassMid = mix(
      yellowOchreRgb.mul(1.08),
      boneCalciteRgb.mul(0.84),
      float(0.32).sub(warmMix.mul(0.45))
    ).mul(tone);

    const kmLever = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.50), brassMid);
    const grainMix = smoothstep(float(0.18), float(0.84), brushX.mul(0.66).add(microFile.mul(0.34)))
      .mul(0.44)
      .add(0.30);
    const brushed = mix(brassShadow, mix(brassMid, kmLever, 0.18), grainMix);

    // Subtle hover sheen lift + warm Realgar-Amber contact gleam when lever physically binds bolt stump
    const bindTint = mix(realgarAmberRgb, yellowOchreRgb, 0.35).mul(0.32);
    mat.colorNode = brushed
      .add(brassMid.mul(uHover.mul(0.22)))
      .add(bindTint.mul(uBind));

    mat.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal.add(seedOff),
      vec3(12.0, 240.0, 108.0),
      float(0.0038)
    );
    mat.roughnessNode = clamp(
      float(0.20)
        .add(brushX.sub(0.5).mul(0.08))
        .add(microFile.sub(0.5).mul(0.03))
        .sub(uHover.mul(0.04)),
      0.14,
      0.31
    );

    leverMaterials.push(mat);
  }

  return {
    charredOak,
    gunmetalCase,
    bluedSteelBolt,
    forendBrass,
    phosphorBronze,
    keyBrass,
    hardenedSteel,
    leverMaterials,
    uKeyHover,
    uBoltHover,
    uLeverHovers,
    uLeverBinds,
    dispose: () => {
      charredOak.dispose();
      gunmetalCase.dispose();
      bluedSteelBolt.dispose();
      forendBrass.dispose();
      phosphorBronze.dispose();
      keyBrass.dispose();
      hardenedSteel.dispose();
      for (const m of leverMaterials) m.dispose();
    },
  };
}
