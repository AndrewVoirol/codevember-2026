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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface HiveMaterials {
  limestone: THREE.MeshStandardNodeMaterial;
  darkBronze: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  needleNorthCinnabar: THREE.MeshStandardNodeMaterial;
  needleSouthPrussian: THREE.MeshStandardNodeMaterial;
  lodestone: THREE.MeshStandardNodeMaterial;
  polarityBand: THREE.MeshStandardNodeMaterial;
  darkGrip: THREE.MeshStandardNodeMaterial;
  /** 0 = Cinnabar North-pull (+1), 1 = Prussian South-pull (-1) */
  uPolarityBlend: { value: number };
  /** 0..1 stylus proximity/press engagement */
  uStylusEngaged: { value: number };
  /** Stylus tip position in specimen-local coordinates for subtle proximity rim sheen */
  uStylusLocalPos: { value: THREE.Vector3 };
  dispose: () => void;
}

function createFeatheredSoftboxTexture(r: number, g: number, b: number, size = 128): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = (y / (size - 1)) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = (x / (size - 1)) * 2 - 1;
      const d = Math.hypot(nx, ny);
      const falloff = Math.max(0, Math.min(1, 1 - d * d));
      const soft = falloff * falloff;
      const idx = (y * size + x) * 4;
      data[idx] = Math.min(255, Math.round(r * soft * 255));
      data[idx + 1] = Math.min(255, Math.round(g * soft * 255));
      data[idx + 2] = Math.min(255, Math.round(b * soft * 255));
      data[idx + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Generates a broadly diffused photographer's studio softbox PMREM environment cubemap
 * calibrated for satin brass honeycomb walls, oxidized bronze bezel, and blued-steel needles.
 */
export function createHiveStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Warm-neutral museum studio ambient room (prevents vertical brass honeycomb walls from collapsing dark)
  envScene.background = new THREE.Color(0x282b2f);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: [number, number, number],
    r: number,
    g: number,
    b: number,
    gain: number
  ) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const tex = createFeatheredSoftboxTexture(r, g, b);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      color: new THREE.Color().setRGB(gain, gain, gain),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.lookAt(0, 0, 0);
    envScene.add(mesh);
    disposables.push(geo, tex, mat);
  };

  // Overhead-left broad warm calcite scrim softbox
  addSoftbox(12.5, 10.0, [-4.8, 5.8, 4.2], 1.0, 0.96, 0.90, 1.95);
  // Right-side tall warm studio scrim (illuminates vertical brass hex walls & stylus barrel)
  addSoftbox(10.5, 10.5, [6.2, 2.8, 3.2], 0.99, 0.95, 0.88, 1.75);
  // Front-horizon soft bounce strip (keeps front interior hex brass walls warm gold, not dark shadow)
  addSoftbox(14.0, 5.5, [0.0, 1.4, 6.8], 0.98, 0.94, 0.87, 1.28);
  // Left-horizon soft bounce strip
  addSoftbox(11.0, 6.0, [-6.8, 1.8, -0.4], 0.96, 0.93, 0.88, 1.18);
  // Overhead zenith studio bank (direct specular wrap on tilted compass needle roof facets)
  addSoftbox(10.5, 10.5, [0.0, 7.6, 0.5], 0.97, 0.95, 0.91, 1.35);
  // Rear-top cool-warm grazing rim scrim (carves bezel & limestone chamfers cleanly from #212529)
  addSoftbox(12.5, 5.5, [-1.0, 4.8, -5.8], 0.92, 0.95, 1.0, 1.55);
  // Rear-right cool accent scrim
  addSoftbox(8.5, 6.5, [4.8, 2.2, -5.4], 0.90, 0.94, 1.0, 1.20);
  // Subtle warm underside studio floor bounce (illuminates rear brass medallion cleanly)
  addSoftbox(12.0, 12.0, [0, -6.2, 0], 0.95, 0.92, 0.88, 0.55);

  const renderTarget = pmrem.fromScene(envScene, 0.085);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Zero derivatives, zero UV seams, uniform control flow.
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

/**
 * Identifies chamfer/bevel faces (neither strictly horizontal nor strictly vertical)
 * for authentic worn-brass edge rubbing and honed limestone highlight catch.
 */
const computeChamferMask = Fn(([nObj_in]: [TslNode]) => {
  const n = normalize(vec3(nObj_in));
  const upness = abs(n.y);
  // Peaks around 45° chamfers (abs(n.y) ~ 0.45..0.85)
  const lowCut = smoothstep(float(0.18), float(0.45), upness);
  const highCut = float(1.0).sub(smoothstep(float(0.84), float(0.96), upness));
  return lowCut.mul(highCut);
});

export function createHiveMaterials(): HiveMaterials {
  const uPolarityBlend = uniform(0);
  const uStylusEngaged = uniform(0);
  const uStylusLocalPos = uniform(new THREE.Vector3(0.86, 0.24, -0.78));

  const pBone = createPigmentUniforms('boneCalcite');
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pCinnabar = createPigmentUniforms('cinnabarVermilion');
  const pPrussian = createPigmentUniforms('azuritePrussian');

  const boneRgb = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const ochreRgb = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const umberRgb = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const cinnabarRgb = vec3(...MINERAL_PALETTE.cinnabarVermilion.rInfLinear);
  const prussianRgb = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const lapisRgb = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Honed Bone-Calcite Sedimentary Limestone Chassis & Recessed Chamber Deck
  // ---------------------------------------------------------------------------
  const limestone = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.76,
    envMapIntensity: 0.72,
  });
  {
    const nObj = normalize(normalLocal);
    const chamfer = computeChamferMask(nObj);

    // Calm sedimentary travertine/limestone grain (tight warm-neutral value range, zero yellow blotches)
    const strata = mineralToothFbm(positionLocal.mul(vec3(2.4, 9.5, 2.4)), float(1.15));
    const finePores = mineralToothFbm(positionLocal.mul(vec3(18.0, 18.0, 18.0)), float(1.35));
    const microTooth = mineralToothFbm(positionLocal.mul(vec3(62.0, 62.0, 62.0)), float(1.2));

    const kaolinBase = kubelkaMunkGlaze(
      pKaolin.K,
      pKaolin.S,
      float(0.85).add(strata.mul(0.35)),
      boneRgb.mul(0.88)
    );
    // Whisper-subtle raw-umber mineral limestone grain + trace warm travertine tone
    const mineralWash = kubelkaMunkGlaze(
      pUmber.K,
      pUmber.S,
      smoothstep(float(0.42), float(0.88), strata).mul(0.055),
      kaolinBase
    );
    const warmStone = kubelkaMunkGlaze(
      pOchre.K,
      pOchre.S,
      smoothstep(float(0.55), float(0.92), finePores).mul(0.022),
      mineralWash
    );

    // Crisp honed micro-tooth + subtle recessed-well depth + cool-pale edge polish on 45° chamfers
    const poreDim = float(0.90).add(finePores.mul(0.09)).add(microTooth.sub(0.5).mul(0.045));
    const deckSoftShade = mix(float(0.91), float(0.98), smoothstep(float(-0.02), float(0.055), positionLocal.y));
    const col = mix(warmStone.mul(poreDim).mul(deckSoftShade), boneRgb.mul(0.94), chamfer.mul(0.22));

    limestone.colorNode = clamp(col, 0.0, 1.0);
    limestone.roughnessNode = clamp(
      float(0.78).sub(chamfer.mul(0.14)).add(finePores.sub(0.5).mul(0.12)),
      0.56,
      0.92
    );
    limestone.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(28.0, 36.0, 28.0),
      float(0.075)
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Oxidized Darkened-Bronze Bolted Bezel Frame & Underside Medallion Plate
  // ---------------------------------------------------------------------------
  const darkBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.82,
    roughness: 0.36,
    envMapIntensity: 1.18,
  });
  {
    const nObj = normalize(normalLocal);
    const chamfer = computeChamferMask(nObj);

    const patina = mineralToothFbm(positionLocal.mul(vec3(9.5, 9.5, 9.5)), float(1.3));
    const microBrush = mineralToothFbm(positionLocal.mul(vec3(54.0, 22.0, 54.0)), float(1.25));

    // Deep dignified antique oil-rubbed bronze (dark cool-umber metal, not chocolate brown)
    const deepBronze = umberRgb.mul(0.25).add(prussianRgb.mul(0.045)).add(vec3(0.020, 0.019, 0.017));
    const rubbedBrass = ochreRgb.mul(0.58).add(boneRgb.mul(0.14));

    const edgeWear = clamp(
      chamfer.mul(0.62).add(smoothstep(float(0.65), float(0.93), patina).mul(0.18)),
      0.0,
      0.72
    );
    const bronzeCol = mix(deepBronze, rubbedBrass, edgeWear.mul(0.42))
      .mul(float(0.90).add(microBrush.mul(0.18)));

    darkBronze.colorNode = clamp(bronzeCol, 0.0, 1.0);
    darkBronze.roughnessNode = clamp(
      float(0.40).sub(chamfer.mul(0.14)).add(patina.sub(0.5).mul(0.12)),
      0.20,
      0.56
    );
    darkBronze.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(46.0, 24.0, 46.0),
      float(0.055)
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Machined Satin-Brass Honeycomb Matrix, Pivot Pins, Bolts & Stylus Barrel
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.80,
    roughness: 0.28,
    envMapIntensity: 1.38,
  });
  {
    const nObj = normalize(normalLocal);
    const chamfer = computeChamferMask(nObj);
    const topLip = smoothstep(float(0.75), float(0.98), nObj.y);

    const brush = mineralToothFbm(positionLocal.mul(vec3(62.0, 30.0, 62.0)), float(1.2));
    const toneVar = mineralToothFbm(positionLocal.mul(vec3(5.5, 5.5, 5.5)), float(1.15));

    // Luminous antique satin brass anchored in yellowOchre + boneCalcite
    const baseBrass = mix(ochreRgb.mul(1.16), boneRgb.mul(0.98), float(0.36));
    const shadedBrass = mix(baseBrass.mul(0.88), baseBrass.mul(1.08), toneVar);
    // Warm gilded catchlight along top hexagonal crests & 45° bevel chamfers
    const rimBright = mix(
      shadedBrass,
      mix(ochreRgb.mul(1.10), boneRgb.mul(1.02), float(0.54)),
      chamfer.mul(0.36).add(topLip.mul(0.22))
    );

    satinBrass.colorNode = clamp(rimBright.mul(float(0.94).add(brush.mul(0.12))), 0.0, 1.0);
    satinBrass.roughnessNode = clamp(
      float(0.31).sub(chamfer.mul(0.09)).sub(topLip.mul(0.05)).add(brush.sub(0.5).mul(0.08)),
      0.16,
      0.42
    );
    satinBrass.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(68.0, 32.0, 68.0),
      float(0.036)
    );
  }

  // ---------------------------------------------------------------------------
  // 4. Glossy Vitreous Cinnabar-Vermilion Enamel (North Compass Needle Halves)
  // ---------------------------------------------------------------------------
  const needleNorthCinnabar = new THREE.MeshStandardNodeMaterial({
    metalness: 0.12,
    roughness: 0.25,
    envMapIntensity: 1.18,
  });
  {
    const nObj = normalize(normalLocal);
    const fineGrain = mineralToothFbm(positionLocal.mul(vec3(72.0, 72.0, 72.0)), float(1.2));
    const ridgeGlow = smoothstep(float(0.004), float(0.014), positionLocal.y);

    // Physical Kubelka-Munk glaze of Cinnabar Vermilion over pale calcite undercoat
    const kmCinnabar = kubelkaMunkGlaze(
      pCinnabar.K,
      pCinnabar.S,
      float(1.45).add(fineGrain.mul(0.45)),
      boneRgb
    );
    const col = mix(kmCinnabar.mul(0.90), cinnabarRgb.mul(1.12), ridgeGlow.mul(0.28));

    needleNorthCinnabar.colorNode = clamp(col, 0.0, 1.0);
    needleNorthCinnabar.roughnessNode = clamp(
      float(0.24).add(fineGrain.sub(0.5).mul(0.06)),
      0.16,
      0.36
    );
    needleNorthCinnabar.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(80.0, 80.0, 80.0),
      float(0.022)
    );
  }

  // ---------------------------------------------------------------------------
  // 5. Heat-Blued Prussian Steel (South Compass Needle Halves)
  // ---------------------------------------------------------------------------
  const needleSouthPrussian = new THREE.MeshStandardNodeMaterial({
    metalness: 0.78,
    roughness: 0.28,
    envMapIntensity: 1.35,
  });
  {
    const nObj = normalize(normalLocal);
    // Longitudinal fine brushed steel grain along needle axis (X)
    const steelBrush = mineralToothFbm(positionLocal.mul(vec3(18.0, 95.0, 95.0)), float(1.25));
    const ridgeLift = smoothstep(float(0.004), float(0.014), positionLocal.y);

    const kmPrussian = kubelkaMunkGlaze(
      pPrussian.K,
      pPrussian.S,
      float(1.55).add(steelBrush.mul(0.35)),
      lapisRgb.mul(0.85)
    );
    const bluedSteel = mix(kmPrussian.mul(1.18), prussianRgb.mul(1.38).add(boneRgb.mul(0.08)), ridgeLift.mul(0.30));

    needleSouthPrussian.colorNode = clamp(bluedSteel.mul(float(0.92).add(steelBrush.mul(0.16))), 0.0, 1.0);
    needleSouthPrussian.roughnessNode = clamp(
      float(0.28).sub(ridgeLift.mul(0.06)).add(steelBrush.sub(0.5).mul(0.08)),
      0.16,
      0.40
    );
    needleSouthPrussian.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(22.0, 110.0, 110.0),
      float(0.036)
    );
  }

  // ---------------------------------------------------------------------------
  // 6. Cleaved Dark Magnetite Lodestone Tip (Stylus Business End)
  // ---------------------------------------------------------------------------
  const lodestone = new THREE.MeshStandardNodeMaterial({
    metalness: 0.58,
    roughness: 0.44,
    envMapIntensity: 1.15,
  });
  {
    const nObj = normalize(normalLocal);
    const crystal = mineralToothFbm(positionLocal.mul(vec3(55.0, 55.0, 55.0)), float(1.3));
    const baseDark = umberRgb.mul(0.20).add(prussianRgb.mul(0.14)).add(vec3(0.028, 0.030, 0.035));
    const tipMask = float(1.0).sub(smoothstep(float(0.0), float(0.07), positionLocal.y));
    const col = baseDark.mul(float(0.88).add(crystal.mul(0.24))).add(boneRgb.mul(tipMask.mul(uStylusEngaged).mul(0.06)));

    lodestone.colorNode = clamp(col, 0.0, 1.0);
    lodestone.roughnessNode = clamp(float(0.46).add(crystal.sub(0.5).mul(0.16)), 0.28, 0.64);
    lodestone.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(65.0, 65.0, 65.0),
      float(0.095)
    );
  }

  // ---------------------------------------------------------------------------
  // 7. Stylus Polarity Indicator Collar (Cinnabar <-> Prussian Smooth Blend)
  // ---------------------------------------------------------------------------
  const polarityBand = new THREE.MeshStandardNodeMaterial({
    metalness: 0.24,
    roughness: 0.26,
    envMapIntensity: 1.22,
  });
  {
    const activeCol = mix(cinnabarRgb.mul(1.08), prussianRgb.mul(1.35), clamp(uPolarityBlend, 0.0, 1.0));
    polarityBand.colorNode = clamp(activeCol, 0.0, 1.0);
  }

  // ---------------------------------------------------------------------------
  // 8. Ebonized Charred-Oak Upper Stylus Grip Sleeve
  // ---------------------------------------------------------------------------
  const darkGrip = new THREE.MeshStandardNodeMaterial({
    metalness: 0.08,
    roughness: 0.48,
    envMapIntensity: 0.90,
  });
  {
    const nObj = normalize(normalLocal);
    const woodGrain = mineralToothFbm(positionLocal.mul(vec3(42.0, 9.0, 42.0)), float(1.25));
    const kmUmber = kubelkaMunkGlaze(pUmber.K, pUmber.S, float(1.25).add(woodGrain.mul(0.4)), umberRgb.mul(0.22));
    darkGrip.colorNode = clamp(kmUmber.mul(0.44).mul(float(0.88).add(woodGrain.mul(0.24))), 0.0, 1.0);
    darkGrip.roughnessNode = clamp(float(0.50).add(woodGrain.sub(0.5).mul(0.14)), 0.34, 0.66);
    darkGrip.normalNode = perturbNormalAniso(
      nObj,
      positionLocal,
      vec3(52.0, 12.0, 52.0),
      float(0.075)
    );
  }

  // Keep unused pigment pack references clean (boneCalcite used in KM backing)
  void pBone;

  return {
    limestone,
    darkBronze,
    satinBrass,
    needleNorthCinnabar,
    needleSouthPrussian,
    lodestone,
    polarityBand,
    darkGrip,
    uPolarityBlend,
    uStylusEngaged,
    uStylusLocalPos,
    dispose: () => {
      limestone.dispose();
      darkBronze.dispose();
      satinBrass.dispose();
      needleNorthCinnabar.dispose();
      needleSouthPrussian.dispose();
      lodestone.dispose();
      polarityBand.dispose();
      darkGrip.dispose();
    },
  };
}
