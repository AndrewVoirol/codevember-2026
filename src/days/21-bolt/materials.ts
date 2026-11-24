import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  clamp,
  dot,
  float,
  max,
  mix,
  normalLocal,
  normalize,
  positionLocal,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
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
 * Normalized independent of `scaleVec` so high-frequency machined brush streaks stay
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

    // Direction-weighted normalized gradient (bounded O(1) independent of scale magnitude!)
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

  const rawUmberRgb = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const burntSiennaRgb = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const yellowOchreRgb = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const realgarAmberRgb = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const boneCalciteRgb = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const azuriteRgb = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const hematiteRgb = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Oiled Charred-Oak Architectural Door-Stile Specimen Block
  //    Deep scorched raw-umber grain valleys + oiled wax satin edge catch
  // ---------------------------------------------------------------------------
  const charredOak = new THREE.MeshStandardNodeMaterial({
    metalness: 0.05,
    roughness: 0.50,
    envMapIntensity: 0.92,
  });
  {
    // Grain runs vertically along Y (door stile orientation), rings wrap across XZ
    const warp = mineralToothFbm(positionLocal.mul(vec3(3.4, 0.85, 3.4)), float(2.2));
    const radialDist = positionLocal.x
      .mul(6.8)
      .add(positionLocal.z.mul(5.2))
      .add(warp.mul(3.6));
    const ringWave = sin(radialDist.mul(5.2)).mul(0.5).add(0.5);
    const fineRings = sin(radialDist.mul(14.5).add(warp.mul(4.0))).mul(0.5).add(0.5);
    const pores = mineralToothFbm(positionLocal.mul(vec3(26.0, 4.2, 26.0)), float(1.6));
    const fineTooth = mineralToothFbm(positionLocal.mul(vec3(48.0, 12.0, 48.0)), float(1.4));

    // Higher heirloom contrast: dark scorched latewood valleys vs rich linseed-oiled raw-umber ridges
    const earlyWood = mix(rawUmberRgb.mul(0.28), burntSiennaRgb.mul(0.21), fineRings.mul(0.42));
    const lateWood = rawUmberRgb.mul(0.048);
    const grainBand = smoothstep(float(0.18), float(0.84), ringWave.mul(0.74).add(fineRings.mul(0.26)));
    const darkCharBacking = mix(lateWood, earlyWood, grainBand);

    // Kubelka-Munk oiled raw-umber glaze for authentic hand-rubbed charred timber depth
    const glazeThick = float(0.38).add(pores.mul(0.48));
    const oiledTimber = kubelkaMunkGlaze(pUmber.K, pUmber.S, glazeThick, darkCharBacking).mul(0.32);

    // Rich warm oiled oak heartwood highlight along the front chamfered cutaway shoulders
    const pocketProximity = smoothstep(float(0.08), float(0.24), positionLocal.z);
    const warmCut = mix(rawUmberRgb.mul(0.38), burntSiennaRgb.mul(0.30), ringWave);
    const withCutWarmth = mix(oiledTimber, warmCut.mul(0.52), pocketProximity.mul(0.34));

    // Deepen scorched wood pores for crisp tactile contrast against satin brass & blued steel
    const poreDarken = smoothstep(float(0.16), float(0.48), pores);
    charredOak.colorNode = withCutWarmth.mul(mix(float(0.54), float(1.06), poreDarken));
    charredOak.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(26.0, 4.5, 26.0),
      float(0.028)
    );
    charredOak.roughnessNode = clamp(
      float(0.38)
        .add(grainBand.mul(0.14))
        .sub(fineTooth.mul(0.07))
        .add(float(1.0).sub(poreDarken).mul(0.22))
        .sub(pocketProximity.mul(0.06)),
      0.30,
      0.76
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Dark Gunmetal / Oil-Sheen Blued-Steel Lock Case Interior
  // ---------------------------------------------------------------------------
  const gunmetalCase = new THREE.MeshStandardNodeMaterial({
    metalness: 0.80,
    roughness: 0.40,
    envMapIntensity: 1.05,
  });
  {
    // Subtle machined milling lines along X + micro oil-patina mottling
    const millLines = mineralToothFbm(positionLocal.mul(vec3(6.0, 48.0, 18.0)), float(1.2));
    const oilMottle = mineralToothFbm(positionLocal.add(vec3(3.1, 1.7, 5.4)), float(9.0));

    const steelDeep = mix(vec3(0.028, 0.032, 0.040), azuriteRgb.mul(0.18), 0.35);
    const steelMid = mix(vec3(0.056, 0.062, 0.074), azuriteRgb.mul(0.26), 0.30);
    const steelHighlight = mix(vec3(0.095, 0.104, 0.118), boneCalciteRgb.mul(0.14), 0.25);

    const baseSteel = mix(steelDeep, steelMid, smoothstep(float(0.25), float(0.78), millLines));
    const patinated = mix(baseSteel, steelHighlight, smoothstep(float(0.62), float(0.90), oilMottle).mul(0.45));

    // Gentle crevice AO toward backplate and perimeter walls so bright brass levers pop forward
    const backAO = smoothstep(float(-0.22), float(0.04), positionLocal.z);
    gunmetalCase.colorNode = patinated.mul(mix(float(0.62), float(1.0), backAO));
    gunmetalCase.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(8.0, 72.0, 20.0),
      float(0.0045)
    );
    gunmetalCase.roughnessNode = clamp(
      float(0.34).add(millLines.mul(0.12)).add(oilMottle.sub(0.5).mul(0.08)),
      0.26,
      0.54
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Heavy Thermally Blued-Steel Deadbolt (Draw-Brushed Along X Throw Axis)
  // ---------------------------------------------------------------------------
  const bluedSteelBolt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.26,
    envMapIntensity: 1.32,
  });
  {
    // Crisp longitudinal draw-brushed metal marks along X (the throw axis)
    const drawBrush = mineralToothFbm(positionLocal.mul(vec3(3.5, 62.0, 62.0)), float(1.2));
    const microTooth = mineralToothFbm(positionLocal.add(vec3(4.2, 8.1, 2.3)), float(38.0));

    // Rich Prussian-blued temper steel anchored in azuritePrussian + boneCalcite steel sheen
    const blueDeep = mix(vec3(0.045, 0.068, 0.108), azuriteRgb.mul(0.56), 0.65);
    const blueBright = mix(vec3(0.145, 0.188, 0.255), boneCalciteRgb.mul(0.28), 0.30);
    const kmBlue = kubelkaMunkGlaze(
      pPrussian.K,
      pPrussian.S,
      float(0.42).add(drawBrush.mul(0.26)),
      blueBright
    );
    const boltBase = mix(blueDeep, mix(blueBright, kmBlue, 0.35), smoothstep(float(0.22), float(0.80), drawBrush));

    // Burnished sliding-wear sheen along the horizontal faces passing through the brass forend
    const wearBand = float(1.0).sub(smoothstep(float(0.08), float(0.18), abs(positionLocal.y.sub(0.16))));
    const burnished = mix(boltBase, mix(boltBase, boneCalciteRgb.mul(0.52), 0.24), wearBand.mul(0.32));

    bluedSteelBolt.colorNode = burnished.add(blueBright.mul(uBoltHover.mul(0.22)));
    bluedSteelBolt.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(5.0, 96.0, 96.0),
      float(0.0038)
    );
    bluedSteelBolt.roughnessNode = clamp(
      float(0.22).add(drawBrush.mul(0.10)).sub(wearBand.mul(0.05)).add(microTooth.sub(0.5).mul(0.03)),
      0.16,
      0.36
    );
  }

  // ---------------------------------------------------------------------------
  // 4. Mortised Forend Satin Brass Faceplate, Escutcheon, Washers & Screws
  // ---------------------------------------------------------------------------
  const forendBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.26,
    envMapIntensity: 1.26,
  });
  {
    // Fine vertical brush strokes along Y
    const brushY = mineralToothFbm(positionLocal.mul(vec3(54.0, 5.5, 54.0)), float(1.2));
    const fineSatin = mineralToothFbm(positionLocal.add(vec3(6.3, 2.1, 4.7)), float(42.0));

    const brassDark = mix(yellowOchreRgb.mul(0.68), rawUmberRgb.mul(0.62), 0.25);
    const brassBright = mix(yellowOchreRgb.mul(1.14), boneCalciteRgb.mul(0.88), 0.34);
    const kmBrass = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.58), brassBright);
    const base = mix(brassDark, mix(brassBright, kmBrass, 0.22), smoothstep(float(0.20), float(0.80), brushY));

    forendBrass.colorNode = base;
    forendBrass.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(88.0, 7.0, 88.0),
      float(0.0032)
    );
    forendBrass.roughnessNode = clamp(
      float(0.22).add(brushY.mul(0.09)).add(fineSatin.sub(0.5).mul(0.03)),
      0.18,
      0.36
    );
  }

  // ---------------------------------------------------------------------------
  // 5. Tempered Phosphor-Bronze Curved Leaf Springs (Warm Coppery-Amber Contrast)
  // ---------------------------------------------------------------------------
  const phosphorBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.87,
    roughness: 0.24,
    envMapIntensity: 1.30,
  });
  {
    const rollGrain = mineralToothFbm(positionLocal.mul(vec3(6.0, 44.0, 44.0)), float(1.2));
    const temperDark = mix(burntSiennaRgb.mul(0.68), hematiteRgb.mul(0.52), 0.32);
    const temperBright = mix(realgarAmberRgb.mul(0.95), yellowOchreRgb.mul(0.98), 0.42);
    const kmBronze = kubelkaMunkGlaze(pSienna.K, pSienna.S, float(0.45), temperBright);

    phosphorBronze.colorNode = mix(
      temperDark,
      mix(temperBright, kmBronze, 0.28),
      smoothstep(float(0.22), float(0.82), rollGrain)
    );
    phosphorBronze.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(8.0, 78.0, 78.0),
      float(0.0032)
    );
    phosphorBronze.roughnessNode = clamp(float(0.20).add(rollGrain.mul(0.10)), 0.16, 0.34);
  }

  // ---------------------------------------------------------------------------
  // 6. Ornate Victorian Key Satin Brass Bow
  // ---------------------------------------------------------------------------
  const keyBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.21,
    envMapIntensity: 1.34,
  });
  {
    const handRub = mineralToothFbm(positionLocal, float(26.0));
    const fineBuff = mineralToothFbm(positionLocal.add(vec3(2.9, 7.1, 1.3)), float(64.0));
    const antiqueDeep = mix(yellowOchreRgb.mul(0.64), rawUmberRgb.mul(0.55), 0.30);
    const antiqueGold = mix(yellowOchreRgb.mul(1.22), boneCalciteRgb.mul(0.92), 0.38);

    const base = mix(
      antiqueDeep,
      antiqueGold,
      smoothstep(float(0.22), float(0.78), handRub.mul(0.65).add(fineBuff.mul(0.35)))
    );
    keyBrass.colorNode = base.add(antiqueGold.mul(uKeyHover.mul(0.18)));
    keyBrass.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(54.0, 54.0, 54.0),
      float(0.0024)
    );
    keyBrass.roughnessNode = clamp(
      float(0.19).add(handRub.mul(0.08)).sub(uKeyHover.mul(0.04)),
      0.14,
      0.32
    );
  }

  // ---------------------------------------------------------------------------
  // 7. Bright Turned Hardened Tool Steel (Bolt Stump, Pivot Post, Guide Pins, Key Shaft & Bitting)
  // ---------------------------------------------------------------------------
  const hardenedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.93,
    roughness: 0.21,
    envMapIntensity: 1.36,
  });
  {
    const latheTurn = mineralToothFbm(positionLocal.mul(vec3(24.0, 24.0, 68.0)), float(1.2));
    const steelCool = mix(boneCalciteRgb.mul(0.46), azuriteRgb.mul(0.42), 0.26);
    const steelBright = mix(boneCalciteRgb.mul(0.78), vec3(0.68, 0.72, 0.78), 0.45);

    hardenedSteel.colorNode = mix(
      steelCool,
      steelBright,
      smoothstep(float(0.24), float(0.78), latheTurn)
    ).add(steelBright.mul(uBoltHover.mul(0.10)));
    hardenedSteel.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal,
      vec3(36.0, 36.0, 96.0),
      float(0.0028)
    );
    hardenedSteel.roughnessNode = clamp(float(0.18).add(latheTurn.mul(0.09)), 0.14, 0.30);
  }

  // ---------------------------------------------------------------------------
  // 8. Five Distinctly Patinated Satin-Brass Lever Tumbler Plates (lever[0..4])
  //    Subtle warm-to-pale-gold tonal progression makes every stacked layer pop!
  // ---------------------------------------------------------------------------
  const leverMaterials: THREE.MeshStandardNodeMaterial[] = [];
  // Individual alloy/patina shifts across the 5 stacked plates (backmost slightly deeper aged, frontmost crisp satin gold)
  const leverToneScale = [0.76, 0.94, 0.84, 1.02, 0.91];
  const leverWarmShift = [0.24, 0.08, 0.18, 0.04, 0.12];

  for (let i = 0; i < LEVER_COUNT; i++) {
    const uHover = uniform(0);
    const uBind = uniform(0);
    uLeverHovers.push(uHover);
    uLeverBinds.push(uBind);

    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.91,
      roughness: 0.23,
      envMapIntensity: 1.30,
    });

    const seedOff = vec3(float(i * 3.71 + 1.1), float(i * 2.13 + 0.7), float(i * 4.37));
    // Crisp horizontal hand-filed brass brushing along X
    const brushX = mineralToothFbm(positionLocal.add(seedOff).mul(vec3(5.5, 58.0, 28.0)), float(1.2));
    const microFile = mineralToothFbm(positionLocal.add(seedOff.mul(1.7)), float(44.0));

    const tone = float(leverToneScale[i]);
    const warmMix = float(leverWarmShift[i]);

    const brassShadow = mix(
      yellowOchreRgb.mul(0.58),
      burntSiennaRgb.mul(0.55),
      warmMix
    ).mul(tone);
    const brassMid = mix(
      yellowOchreRgb.mul(1.05),
      boneCalciteRgb.mul(0.84),
      float(0.32).sub(warmMix.mul(0.5))
    ).mul(tone);

    const kmLever = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.52), brassMid);
    const brushed = mix(
      brassShadow,
      mix(brassMid, kmLever, 0.18),
      smoothstep(float(0.20), float(0.80), brushX.mul(0.72).add(microFile.mul(0.28)))
    );

    // Subtle hover sheen lift + warm Realgar-Amber contact gleam when lever physically binds bolt stump
    const bindTint = mix(realgarAmberRgb, yellowOchreRgb, 0.35).mul(0.32);
    mat.colorNode = brushed
      .add(brassMid.mul(uHover.mul(0.22)))
      .add(bindTint.mul(uBind));

    mat.normalNode = perturbNormalAniso(
      normalLocal,
      positionLocal.add(seedOff),
      vec3(7.0, 92.0, 42.0),
      float(0.0030)
    );
    mat.roughnessNode = clamp(
      float(0.20)
        .add(brushX.mul(0.09))
        .add(microFile.sub(0.5).mul(0.03))
        .sub(uHover.mul(0.04)),
      0.15,
      0.34
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
