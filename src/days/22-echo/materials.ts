import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  atan,
  attribute,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  fract,
  length,
  max,
  min,
  mix,
  normalLocal,
  normalize,
  positionLocal,
  pow,
  reflect,
  sin,
  smoothstep,
  sqrt,
  transformNormalToView,
  uniform,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { mineralToothFbm } from '../../core/tslMaterials';
import {
  APERTURE_HALF,
  CAVITY_HALF,
  ECHO_BOUNCE_COUNT,
  SODA_LIME_CHORD_PER_BOUNCE,
  SODA_LIME_EXTINCTION_RGB,
} from './optics';
import { CORE_ANCHOR_Y } from './physics';
import {
  BEZEL_BACK_Z,
  BEZEL_FRONT_Z,
  type EchoMaterialsContract,
} from './geometry';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface EchoMaterials extends EchoMaterialsContract {
  agedBronze: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  bezelBrass: THREE.MeshStandardNodeMaterial;
  vernierBrass: THREE.MeshStandardNodeMaterial;
  knurledBrass: THREE.MeshStandardNodeMaterial;
  bluedSteel: THREE.MeshStandardNodeMaterial;
  infinityMirror: THREE.MeshBasicNodeMaterial;
  uCamLocal: { value: THREE.Vector3 };
  uMirrorNormals: Array<{ value: THREE.Vector3 }>;
  uCorePos: { value: THREE.Vector3 };
  uCoreAxis: { value: THREE.Vector3 };
  uCoreSpin: { value: number };
  uPluckFlash: { value: number };
  uTime: { value: number };
  uThumbHover: { value: number };
  uGlassHover: { value: number };
  dispose: () => void;
}

/**
 * Feathered radial softbox luminance card so PMREM reflections roll smoothly across
 * machined bronze chamfers, turned brass bezels, and 4-sided diamond knurling.
 */
function createFeatheredSoftboxTexture(r: number, g: number, b: number): THREE.DataTexture {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = ((y + 0.5) / size) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = ((x + 0.5) / size) * 2 - 1;
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
 * Generates an enveloping studio cyclorama + softbox PMREM environment cubemap calibrated
 * so high-metalness dark bronze, turned satin brass, and blued steel read with crisp continuous
 * bevel speculars and zero dark-void collapse on shadowed rails or underside orbits.
 */
export function createEchoStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x1f2429);

  const disposables: Array<{ dispose: () => void }> = [];

  // Enveloping studio cyclorama gradient dome so metals never reflect dead black gaps
  const domeGeo = new THREE.SphereGeometry(18, 24, 16);
  const posAttr = domeGeo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(posAttr.count * 3);
  const cFloor = new THREE.Color(0x282624);
  const cHorizon = new THREE.Color(0x3a4046);
  const cZenith = new THREE.Color(0x4c5258);
  const tmpCol = new THREE.Color();
  for (let i = 0; i < posAttr.count; i++) {
    const ny = posAttr.getY(i) / 18;
    if (ny < 0) {
      tmpCol.lerpColors(cHorizon, cFloor, Math.min(1, -ny));
    } else {
      tmpCol.lerpColors(cHorizon, cZenith, Math.min(1, ny));
    }
    colors[i * 3] = tmpCol.r;
    colors[i * 3 + 1] = tmpCol.g;
    colors[i * 3 + 2] = tmpCol.b;
  }
  domeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const domeMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.BackSide,
  });
  envScene.add(new THREE.Mesh(domeGeo, domeMat));
  disposables.push(domeGeo, domeMat);

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

  // Overhead-left warm calcite key softbox
  addSoftbox(12.0, 9.5, [-5.2, 5.8, 4.2], 1.0, 0.96, 0.9, 1.95);
  // Front-right tall studio strip scrim (carves crisp bronze vertical bevels & diamond knurling)
  addSoftbox(8.0, 12.5, [6.2, 2.4, 3.6], 0.99, 0.95, 0.89, 1.72);
  // Front-left low warm fill scrim (lifts front dark-bronze rails off #212529)
  addSoftbox(10.0, 6.5, [-3.8, -1.8, 6.4], 0.95, 0.91, 0.85, 0.92);
  // Rear-top cool calcite rim strip
  addSoftbox(13.0, 5.8, [-1.2, 5.2, -5.8], 0.9, 0.95, 1.0, 1.6);
  // Zenith diffused gallery canopy
  addSoftbox(10.0, 10.0, [0.0, 7.6, 0.4], 0.96, 0.94, 0.9, 0.92);
  // Rear-right celadon-slate fill card
  addSoftbox(9.0, 7.0, [5.0, 1.2, -5.6], 0.88, 0.95, 0.98, 1.18);
  // Warm gallery underside table bounce (illuminates underside lock caps cleanly)
  addSoftbox(13.0, 13.0, [0.0, -6.5, 0.8], 0.96, 0.92, 0.86, 0.56);

  const renderTarget = pmrem.fromScene(envScene, 0.068);
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
 * Scale-normalized 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Supports anisotropic scaling vector `scaleVec_in` for fine linear brushed/milled metal grain.
 */
const perturbNormal3DAniso = Fn(
  ([baseNorm_in, pos_in, scaleVec_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in).mul(vec3(scaleVec_in));
    const eps = float(0.16);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), float(1.0));
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), float(1.0));
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), float(1.0));
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), float(1.0));

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

/**
 * Analytical studio softbox dome radiance along a reflected exterior ray direction,
 * matching the PMREM softbox scrims for crisp exterior two-way mirror glass Fresnel reflections.
 */
const evalStudioSoftboxDome = Fn(([dir_in]: [TslNode]) => {
  const d = normalize(vec3(dir_in));

  // Gallery Chalky Slate room gradient matching the studio cyclorama
  const skyT = clamp(d.y.mul(0.5).add(0.5), 0.0, 1.0);
  const roomBase = mix(vec3(0.018, 0.021, 0.024), vec3(0.044, 0.052, 0.059), skyT);

  // Overhead-left warm calcite key softbox
  const keyDir = normalize(vec3(-0.55, 0.64, 0.53));
  const keyDot = max(float(0.0), dot(d, keyDir));
  const keyBox = smoothstep(float(0.76), float(0.96), keyDot).mul(0.68);

  // Right tall studio strip scrim
  const rightDir = normalize(vec3(0.76, 0.3, 0.57));
  const rightDot = max(float(0.0), dot(d, rightDir));
  const rightBox = smoothstep(float(0.78), float(0.96), rightDot).mul(0.52);

  // Rear-top cool calcite rim strip
  const rimDir = normalize(vec3(-0.14, 0.66, -0.74));
  const rimDot = max(float(0.0), dot(d, rimDir));
  const rimBox = smoothstep(float(0.8), float(0.97), rimDot).mul(0.45);

  // Broad soft studio overhead sheen
  const broadGlow = pow(max(float(0.0), d.y.mul(0.68).add(0.32)), 3.0).mul(0.09);

  return roomBase
    .add(vec3(0.98, 0.94, 0.87).mul(keyBox.add(rightBox)))
    .add(vec3(0.84, 0.92, 0.98).mul(rimBox))
    .add(vec3(0.88, 0.86, 0.82).mul(broadGlow));
});

/**
 * Creates all materials for Day 22 - Echo (strictly anchored in `MINERAL_PALETTE`):
 * - Brushed oil-rubbed dark bronze skeleton frame with warm burnished 45° chamfer catchlights
 * - Brushed antique satin brass gussets, rivets & lock caps
 * - Recessed stepped satin-brass glass retaining bezels (with rabbet depth occlusion)
 * - Calibrated micrometer vernier sleeve brass with engraved radial graduation ticks
 * - 4-sided diamond-knurled satin-brass micrometer drums with oxidized valleys
 * - Tempered Prussian-blued steel helical lead-screw shafts
 * - 40-bounce exact tilted-plane TSL two-way infinity mirror glass shader with Beer-Lambert
 *   soda-lime celadon/verdigris absorption, patinated dark-bronze interior lattice,
 *   ceiling suspension rosette, and swinging Bone-Calcite prism enclosing a crisp 3D helical
 *   Realgar-Amber Edison filament!
 */
export function createEchoMaterials(): EchoMaterials {
  const uCamLocal = uniform(new THREE.Vector3(2.0, 1.3, 3.1));
  const uMirrorNormals = [
    uniform(new THREE.Vector3(-1, 0, 0)),
    uniform(new THREE.Vector3(1, 0, 0)),
    uniform(new THREE.Vector3(0, -1, 0)),
    uniform(new THREE.Vector3(0, 1, 0)),
    uniform(new THREE.Vector3(0, 0, -1)),
    uniform(new THREE.Vector3(0, 0, 1)),
  ];
  const uCorePos = uniform(new THREE.Vector3(0, -0.02, 0));
  const uCoreAxis = uniform(new THREE.Vector3(0, 1, 0));
  const uCoreSpin = uniform(0.35);
  const uPluckFlash = uniform(0);
  const uTime = uniform(0);
  const uThumbHover = uniform(0);
  const uGlassHover = uniform(0);

  // Shared canonical mineral pigment vectors (zero hardcoded neon RGBs)
  const pRawUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const pYellowOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const pBurntSienna = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const pBoneCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const pRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const pSpore = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);
  const pVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
  const pCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const pAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const pLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Brushed Oil-Rubbed Dark-Bronze Precision Optical Skeleton Frame
  //    Fine CNC-lapped metallic grain (zero wood-fiber streaks, zero dark collapse)
  // ---------------------------------------------------------------------------
  const agedBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.27,
    envMapIntensity: 1.36,
  });
  {
    // Ultra-fine satin-lapped directional metal brushing (no coarse woody stripes)
    const brushA = mineralToothFbm(positionLocal.mul(vec3(58.0, 260.0, 260.0)), float(1.0));
    const brushB = mineralToothFbm(positionLocal.mul(vec3(260.0, 58.0, 260.0)), float(1.0));
    const brushC = mineralToothFbm(positionLocal.mul(vec3(260.0, 260.0, 58.0)), float(1.0));
    const absN = abs(normalize(normalLocal));
    const fineGrain = mineralToothFbm(positionLocal, float(140.0));
    const brushed = brushA
      .mul(absN.y.add(absN.z).mul(0.5))
      .add(brushB.mul(absN.x.add(absN.z).mul(0.5)))
      .add(brushC.mul(absN.x.add(absN.y).mul(0.5)))
      .mul(0.62)
      .add(fineGrain.mul(0.38));

    // Detect 45-deg machined rail chamfers (`maxAxis < 0.85` -> crisp burnished bronze bevel!)
    const maxAxisN = max(absN.x, max(absN.y, absN.z));
    const chamferMask = smoothstep(float(0.92), float(0.74), maxAxisN);

    // Rich dark museum oil-rubbed bronze (tight metallic value range -> cold heavy metal, never wood)
    const bronzeDeep = mix(pRawUmber, pBoneCalcite, float(0.035)).mul(0.76);
    const bronzeSatin = mix(pRawUmber, pYellowOchre, float(0.14)).add(pBoneCalcite.mul(0.032));
    const bronzeChamfer = mix(pRawUmber, pYellowOchre, float(0.34)).add(pBoneCalcite.mul(0.068));
    const verdigrisMicro = pVerdigris.mul(0.18);

    const baseBronze = mix(
      bronzeDeep,
      bronzeSatin,
      smoothstep(float(0.34), float(0.68), brushed).mul(0.65)
    );
    const edgeBurnished = mix(baseBronze, bronzeChamfer, chamferMask.mul(0.62));
    agedBronze.colorNode = mix(
      edgeBurnished,
      verdigrisMicro,
      smoothstep(float(0.86), float(0.97), fineGrain).mul(0.028)
    );
    agedBronze.normalNode = perturbNormal3DAniso(
      normalLocal,
      positionLocal,
      vec3(110.0, 110.0, 110.0),
      float(0.0085)
    );
    agedBronze.roughnessNode = clamp(
      float(0.28)
        .add(brushed.sub(0.5).mul(0.055))
        .sub(chamferMask.mul(0.065)),
      0.19,
      0.34
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Turned Antique Satin Brass (Gussets, Rivets, Lock Caps)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.23,
    envMapIntensity: 1.24,
  });
  {
    const fineSatin = mineralToothFbm(positionLocal.mul(vec3(195.0, 95.0, 195.0)), float(1.0));
    const microGrain = mineralToothFbm(positionLocal.add(vec3(4.1, 1.8, 6.2)), float(185.0));
    const brassAged = mix(pRawUmber, pYellowOchre, float(0.74)).mul(1.02);
    const brassBright = mix(pYellowOchre, pBoneCalcite, float(0.22)).mul(1.03);
    const absN = abs(normalize(normalLocal));
    const chamferEdge = smoothstep(float(0.92), float(0.72), max(absN.x, max(absN.y, absN.z)));

    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.32), float(0.72), fineSatin.mul(0.58).add(microGrain.mul(0.42)))
        .mul(0.68)
        .add(chamferEdge.mul(0.26))
    );
    satinBrass.normalNode = perturbNormal3DAniso(
      normalLocal,
      positionLocal,
      vec3(140.0, 140.0, 140.0),
      float(0.0038)
    );
    satinBrass.roughnessNode = clamp(
      float(0.23)
        .add(fineSatin.sub(0.5).mul(0.04))
        .sub(chamferEdge.mul(0.045)),
      0.17,
      0.29
    );
  }

  // ---------------------------------------------------------------------------
  // 2b. Stepped Recessed Brass Retaining Bezel (Depth-Occluded Rabbet — Never Blows Out!)
  // ---------------------------------------------------------------------------
  const bezelBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.32,
    envMapIntensity: 1.04,
  });
  {
    const fineBezel = mineralToothFbm(positionLocal, float(145.0));
    // Local Z spans [BEZEL_BACK_Z (0.594) .. BEZEL_FRONT_Z (0.630)] -> subtle recess shadow toward glass seat
    const recessT = smoothstep(float(BEZEL_BACK_Z), float(BEZEL_FRONT_Z), positionLocal.z);
    const rabbetShadow = mix(pRawUmber, pYellowOchre, float(0.38)).mul(0.72);
    const bezelRim = mix(pRawUmber, pYellowOchre, float(0.74)).mul(0.84);
    const bevelCatch = smoothstep(float(0.88), float(0.65), abs(normalize(normalLocal).z));

    const baseCol = mix(rabbetShadow, bezelRim, recessT.mul(0.72).add(0.28));
    bezelBrass.colorNode = baseCol
      .mul(float(0.95).add(fineBezel.mul(0.1)))
      .add(pYellowOchre.mul(bevelCatch.mul(0.025)));
    bezelBrass.normalNode = perturbNormal3DAniso(
      normalLocal,
      positionLocal,
      vec3(145.0, 145.0, 145.0),
      float(0.0038)
    );
    bezelBrass.roughnessNode = clamp(
      mix(float(0.42), float(0.3), recessT).add(fineBezel.sub(0.5).mul(0.045)),
      0.26,
      0.46
    );
  }

  // ---------------------------------------------------------------------------
  // 2c. Calibrated Micrometer Vernier Sleeve Brass (Engraved Radial Index Ticks!)
  // ---------------------------------------------------------------------------
  const vernierBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.9,
    roughness: 0.24,
    envMapIntensity: 1.24,
  });
  {
    const p = positionLocal;
    const fineTurn = mineralToothFbm(vec3(p.x.mul(30.0), p.y.mul(220.0), p.z.mul(30.0)), float(1.0));
    const safeX = abs(p.x).lessThan(1e-5).select(float(1e-5), p.x);
    const phi = atan(p.z, safeX);
    // 36 engraved radial micrometer graduations around the vernier barrel (`y in [0.028, 0.048]`)
    const tickU = fract(phi.mul(36.0 / (Math.PI * 2)).add(36.0));
    const tickDist = abs(tickU.sub(0.5));
    const majorU = fract(phi.mul(6.0 / (Math.PI * 2)).add(6.0));
    const isMajor = abs(majorU.sub(0.5)).lessThan(0.09);
    const tickTopY = isMajor.select(float(0.0455), float(0.0415));
    const inVernierBand = smoothstep(float(0.027), float(0.029), p.y).mul(
      float(1.0).sub(smoothstep(tickTopY, tickTopY.add(0.002), p.y))
    );
    const radBarrel = smoothstep(float(0.024), float(0.028), length(vec3(p.x, float(0.0), p.z)));
    const tickMask = float(1.0)
      .sub(smoothstep(float(0.055), float(0.11), tickDist))
      .mul(inVernierBand)
      .mul(radBarrel);
    // Fine circumferential witness groove at y = 0.0475
    const datumRing = float(1.0)
      .sub(smoothstep(float(0.0007), float(0.0014), abs(p.y.sub(0.0475))))
      .mul(radBarrel);
    const engraved = clamp(max(tickMask, datumRing), 0.0, 1.0);

    const brassBase = mix(
      mix(pRawUmber, pYellowOchre, float(0.7)).mul(1.05),
      mix(pYellowOchre, pBoneCalcite, float(0.24)).mul(1.05),
      fineTurn
    );
    const inkDark = pRawUmber.mul(0.24);
    vernierBrass.colorNode = mix(brassBase, inkDark, engraved.mul(0.86));
    vernierBrass.roughnessNode = clamp(
      mix(float(0.23), float(0.62), engraved).add(fineTurn.sub(0.5).mul(0.04)),
      0.18,
      0.66
    );
    vernierBrass.metalnessNode = mix(float(0.91), float(0.35), engraved);
  }

  // ---------------------------------------------------------------------------
  // 3. 4-Sided Diamond-Knurled Satin Brass Micrometer Drums (Oxidized Valley AO + Hover)
  // ---------------------------------------------------------------------------
  const knurledBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.22,
    envMapIntensity: 1.32,
  });
  {
    const aKnurlValley = attribute('aKnurlValley', 'float');
    const fineGrain = mineralToothFbm(positionLocal, float(84.0));
    const valleyDark = mix(pRawUmber, pBurntSienna, float(0.18)).mul(0.52);
    const crestBright = mix(pYellowOchre, pBoneCalcite, float(0.28)).mul(1.16);
    const grooveFactor = clamp(aKnurlValley, 0.0, 1.0);
    const baseKnurl = mix(
      valleyDark,
      crestBright,
      smoothstep(float(0.06), float(0.94), grooveFactor)
    );
    const hoverWarmth = pYellowOchre.mul(uThumbHover.mul(0.16));

    knurledBrass.colorNode = baseKnurl.add(hoverWarmth);
    knurledBrass.roughnessNode = clamp(
      mix(float(0.5), float(0.18), grooveFactor).add(fineGrain.sub(0.5).mul(0.035)),
      0.15,
      0.54
    );
    knurledBrass.metalnessNode = mix(float(0.7), float(0.94), grooveFactor);
  }

  // ---------------------------------------------------------------------------
  // 4. Tempered Prussian-Blued Steel Helical Lead-Screw Threads
  // ---------------------------------------------------------------------------
  const bluedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.25,
    envMapIntensity: 1.24,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(72.0));
    const rCyl = length(vec3(positionLocal.x, float(0.0), positionLocal.z));
    // Subtle burnished gleam along the raised helical V-thread crests (`r > 0.017`)
    const crestPolished = smoothstep(float(0.016), float(0.0192), rCyl);
    const steelRoot = mix(pAzurite, pLapis, float(0.35)).mul(0.46);
    const steelCrest = mix(pAzurite, pBoneCalcite, float(0.34)).mul(0.76);
    bluedSteel.colorNode = mix(
      steelRoot,
      steelCrest,
      clamp(crestPolished.mul(0.72).add(grain.mul(0.28)), 0.0, 1.0)
    );
    bluedSteel.roughnessNode = clamp(
      mix(float(0.34), float(0.18), crestPolished).add(grain.sub(0.5).mul(0.05)),
      0.16,
      0.4
    );
  }

  // ---------------------------------------------------------------------------
  // 5. True 40-Bounce Exact Tilted-Plane Analytic TSL Infinity Mirror Fragment Raytracer
  //    Zero texture fetches and zero fwidth/dpdx/dpdy inside dynamic branches!
  // ---------------------------------------------------------------------------
  const infinityMirror = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
  });

  const evalInfinityMirror = Fn(() => {
    const pEntry = vec3(positionLocal);
    const nOuter = normalize(vec3(normalLocal));
    const rd0 = normalize(pEntry.sub(uCamLocal));
    const viewDir = rd0.negate();
    const ndotv = clamp(dot(nOuter, viewDir), 0.0, 1.0);

    // Soda-lime ferrous-iron Beer-Lambert extinction per bounce (hoisted outside 40-bounce loop)
    const sigmaSodaLime = vec3(
      SODA_LIME_EXTINCTION_RGB[0],
      SODA_LIME_EXTINCTION_RGB[1],
      SODA_LIME_EXTINCTION_RGB[2]
    );
    const beerPerBounce = exp(sigmaSodaLime.negate().mul(float(SODA_LIME_CHORD_PER_BOUNCE))).toVar();

    // Orthonormal local basis (cU, cV, cW) of the swinging faceted crystal prism (ONCE per pixel)
    const cW = normalize(uCoreAxis).toVar();
    const cU0 = normalize(cross(vec3(0.0, 0.0, 1.0), cW));
    const cV0 = cross(cW, cU0);
    const cosSpin = cos(uCoreSpin);
    const sinSpin = sin(uCoreSpin);
    const cU = cU0.mul(cosSpin).add(cV0.mul(sinSpin)).toVar();
    const cV = cV0.mul(cosSpin).sub(cU0.mul(sinSpin)).toVar();

    // Transform stationary studio key & rim directions into the prism's rotating basis ONCE per pixel
    // (facet speculars catch stationary gallery softboxes naturally as the crystal spins & swings!)
    const lStudioKeyW = normalize(vec3(-0.5, 0.72, 0.48));
    const lStudioRimW = normalize(vec3(0.46, 0.56, -0.69));
    const lKeyL = vec3(dot(lStudioKeyW, cU), dot(lStudioKeyW, cV), dot(lStudioKeyW, cW)).toVar();
    const lRimL = vec3(dot(lStudioRimW, cU), dot(lStudioRimW, cV), dot(lStudioRimW, cW)).toVar();

    // Subtle organic incandescent shimmer + pluck warmth surge
    const filamentPulse = float(1.0)
      .add(sin(uTime.mul(4.2)).mul(0.03))
      .add(uPluckFlash.mul(0.58))
      .toVar();

    // Taut brass suspension wire runs strictly along the physical pendulum axis up to CORE_ANCHOR_Y,
    // where a turned brass ceiling eyelet stem connects solidly to the +Y ceiling mirror rosette
    const wireTop = vec3(0.0, CORE_ANCHOR_Y, 0.0);
    const wireBot = uCorePos.add(cW.mul(0.146)).toVar();
    const wireVec = wireTop.sub(wireBot).toVar();
    const wireLenSq = max(float(1e-5), dot(wireVec, wireVec)).toVar();
    const stemTop = vec3(0.0, CAVITY_HALF, 0.0);
    const stemVec = stemTop.sub(wireTop);
    const stemLenSq = float((CAVITY_HALF - CORE_ANCHOR_Y) * (CAVITY_HALF - CORE_ANCHOR_Y));

    // Faceted Bone-Calcite bipyramid crystal slab normals (4 opposing pairs -> 8 razor facets)
    const invSlabNorm = float(1.0 / Math.sqrt(1.0 + 1.0 + 0.78 * 0.78));
    const sN0 = vec3(1.0, 1.0, 0.78).mul(invSlabNorm);
    const sN1 = vec3(1.0, -1.0, 0.78).mul(invSlabNorm);
    const sN2 = vec3(-1.0, 1.0, 0.78).mul(invSlabNorm);
    const sN3 = vec3(-1.0, -1.0, 0.78).mul(invSlabNorm);
    const prismHalfD = float(0.082);

    // Exterior two-way mirror Fresnel reflection of studio softboxes + faint optical micro-sheen
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const glassMicroSheen = mineralToothFbm(pEntry, float(46.0)).sub(0.5).mul(0.012);
    const fresnelExt = clamp(
      float(0.056).add(float(0.5).mul(pow(oneMinusV, 4.2))).add(glassMicroSheen.mul(oneMinusV)),
      0.04,
      0.66
    );
    const rExt = reflect(rd0, nOuter);
    const extStudioRefl = evalStudioSoftboxDome(rExt);

    // Thick soda-lime float glass green-celadon beveled edge catch along the brass retaining bezel
    const aEntry = abs(pEntry);
    const maxE = max(aEntry.x, max(aEntry.y, aEntry.z));
    const minE = min(aEntry.x, min(aEntry.y, aEntry.z));
    const midE = aEntry.x.add(aEntry.y).add(aEntry.z).sub(maxE).sub(minE);
    const apertureEdge = float(APERTURE_HALF).sub(midE);
    const glassRimBevel = float(1.0).sub(smoothstep(float(0.0), float(0.022), apertureEdge));
    const glassRimTint = mix(pVerdigris, pCeladon, float(0.52)).mul(
      glassRimBevel.mul(float(0.22).add(fresnelExt.mul(0.46)).add(uGlassHover.mul(0.09)))
    );

    // Raytracer state across 40 optical bounces
    const ro = pEntry.toVar();
    const rd = rd0.toVar();
    const accum = vec3(0.0, 0.0, 0.0).toVar();
    const throughput = vec3(1.0, 1.0, 1.0)
      .mul(float(1.0).sub(fresnelExt.mul(0.48)))
      .toVar();
    const totalDist = float(0.0).toVar();

    Loop(ECHO_BOUNCE_COUNT, ({ i }) => {
      const bounceF = float(i);

      // -----------------------------------------------------------------------
      // A. Exact Analytic Ray vs. 6 Slightly Tilted Interior Mirror Planes
      // -----------------------------------------------------------------------
      const tWall = float(1e5).toVar();
      const hitN = vec3(0.0, 1.0, 0.0).toVar();

      for (let k = 0; k < 6; k++) {
        const planeN = uMirrorNormals[k];
        const denom = dot(rd, planeN);
        If(denom.lessThan(-1e-5), () => {
          const tCand = dot(ro, planeN).add(float(CAVITY_HALF)).negate().div(denom);
          If(tCand.greaterThan(1e-4).and(tCand.lessThan(tWall)), () => {
            tWall.assign(tCand);
            hitN.assign(planeN);
          });
        });
      }

      If(tWall.greaterThan(1e4), () => {
        Break();
      });

      // -----------------------------------------------------------------------
      // B1. Restrained Warm Realgar-Amber Optical Halo Around Swinging Filament
      // -----------------------------------------------------------------------
      const toCore = uCorePos.sub(ro);
      const tClosest = clamp(dot(toCore, rd), float(0.0), tWall);
      const closestVec = ro.add(rd.mul(tClosest)).sub(uCorePos);
      const dCoreSq = dot(closestVec, closestVec);
      const haloTight = exp(dCoreSq.mul(-58.0)).mul(0.22);
      const haloSoft = exp(dCoreSq.mul(-10.5)).mul(0.044);
      const haloColor = mix(pRealgar, pYellowOchre, float(0.28)).mul(
        haloTight.add(haloSoft).mul(filamentPulse)
      );
      accum.addAssign(haloColor.mul(throughput));

      // -----------------------------------------------------------------------
      // B2. Analytic Taut Satin-Brass Suspension Wire + Hanging Ceiling Eyelet
      // -----------------------------------------------------------------------
      const oaW = ro.sub(wireBot);
      const dOa = dot(rd, oaW);
      const wOa = dot(wireVec, oaW);
      const dW = dot(rd, wireVec);
      const denomW = max(float(1e-5), wireLenSq.sub(dW.mul(dW)));
      const sWire = clamp(wOa.sub(dOa.mul(dW)).div(denomW), 0.0, 1.0);
      const tWire = max(float(0.0), dOa.negate().add(dW.mul(sWire)));

      If(tWire.greaterThan(1e-4).and(tWire.lessThan(tWall)), () => {
        const pRayW = ro.add(rd.mul(tWire));
        const pSegW = wireBot.add(wireVec.mul(sWire));
        const dWireSq = dot(pRayW.sub(pSegW), pRayW.sub(pSegW));
        // Subtle brass eyelet flare right at the top ceiling anchor (`sWire > 0.96`)
        const eyeletFlare = smoothstep(float(0.955), float(0.995), sWire).mul(0.011);
        const wireRad = float(0.0056).add(eyeletFlare).add(totalDist.mul(0.00038));
        If(dWireSq.lessThan(wireRad.mul(wireRad)), () => {
          const wLit = float(0.42).add(float(0.58).mul(float(1.0).sub(sWire)));
          const wireCol = mix(pRawUmber.mul(0.72), pYellowOchre.mul(1.22), wLit);
          const wireEdge = float(1.0).sub(
            smoothstep(wireRad.mul(0.55), wireRad, sqrt(max(float(0.0), dWireSq)))
          );
          accum.addAssign(wireCol.mul(wireEdge).mul(throughput));
          throughput.mulAssign(float(1.0).sub(wireEdge.mul(0.9)));
        });
      });

      // Turned brass ceiling eyelet stem connecting CORE_ANCHOR_Y solidly to the +Y mirror rosette
      const oaS = ro.sub(wireTop);
      const dOaS = dot(rd, oaS);
      const wOaS = dot(stemVec, oaS);
      const dS = dot(rd, stemVec);
      const denomS = max(float(1e-5), stemLenSq.sub(dS.mul(dS)));
      const sStem = clamp(wOaS.sub(dOaS.mul(dS)).div(denomS), 0.0, 1.0);
      const tStem = max(float(0.0), dOaS.negate().add(dS.mul(sStem)));
      If(tStem.greaterThan(1e-4).and(tStem.lessThan(tWall)), () => {
        const pRayS = ro.add(rd.mul(tStem));
        const pSegS = wireTop.add(stemVec.mul(sStem));
        const dStemSq = dot(pRayS.sub(pSegS), pRayS.sub(pSegS));
        const stemRad = mix(float(0.014), float(0.022), sStem);
        If(dStemSq.lessThan(stemRad.mul(stemRad)), () => {
          const stemCol = mix(pRawUmber.mul(0.7), pYellowOchre.mul(1.22), float(0.75));
          const stemEdge = float(1.0).sub(
            smoothstep(stemRad.mul(0.65), stemRad, sqrt(max(float(0.0), dStemSq)))
          );
          accum.addAssign(stemCol.mul(stemEdge).mul(throughput));
          throughput.mulAssign(float(1.0).sub(stemEdge.mul(0.95)));
        });
      });

      // -----------------------------------------------------------------------
      // B3. Suspended Faceted Bone-Calcite Crystal Prism + Crisp 3D Helical Edison Filament
      // -----------------------------------------------------------------------
      const relCore = ro.sub(uCorePos);
      const bSphere = dot(relCore, rd);
      const cSphere = dot(relCore, relCore).sub(float(0.195 * 0.195));
      const discSphere = bSphere.mul(bSphere).sub(cSphere);

      If(discSphere.greaterThan(0.0), () => {
        const sqrtDisc = sqrt(discSphere);
        const tSphFar = bSphere.negate().add(sqrtDisc);
        const tSphNear = bSphere.negate().sub(sqrtDisc);

        If(tSphFar.greaterThan(1e-4).and(tSphNear.lessThan(tWall)), () => {
          const roL = vec3(dot(relCore, cU), dot(relCore, cV), dot(relCore, cW)).toVar();
          const rdL = vec3(dot(rd, cU), dot(rd, cV), dot(rd, cW)).toVar();

          // Exact Kay-Kajiya 4-slab intersection tracking BOTH entry and exit crystal normals
          const tNear = float(-1e5).toVar();
          const tFar = float(1e5).toVar();
          const nPrismL = vec3(0.0, 0.0, 1.0).toVar();
          const nExitL = vec3(0.0, 0.0, -1.0).toVar();

          const slabNormals = [sN0, sN1, sN2, sN3];
          for (let m = 0; m < 4; m++) {
            const sn = slabNormals[m];
            const oDot = dot(roL, sn);
            const dDot = dot(rdL, sn);
            const absD = abs(dDot);
            If(absD.greaterThan(1e-5), () => {
              const invD = float(1.0).div(dDot);
              const t0 = prismHalfD.negate().sub(oDot).mul(invD);
              const t1 = prismHalfD.sub(oDot).mul(invD);
              const tMinS = min(t0, t1);
              const tMaxS = max(t0, t1);
              If(tMinS.greaterThan(tNear), () => {
                tNear.assign(tMinS);
                const flipSign = dDot.greaterThan(0.0).select(float(-1.0), float(1.0));
                nPrismL.assign(sn.mul(flipSign));
              });
              If(tMaxS.lessThan(tFar), () => {
                tFar.assign(tMaxS);
                const exitSign = dDot.greaterThan(0.0).select(float(1.0), float(-1.0));
                nExitL.assign(sn.mul(exitSign));
              });
            }).Else(() => {
              If(abs(oDot).greaterThan(prismHalfD), () => {
                tNear.assign(float(1e5));
              });
            });
          }

          If(tFar.greaterThan(tNear).and(tFar.greaterThan(1e-4)).and(tNear.lessThan(tWall)), () => {
            const tEnter = max(float(0.0), tNear);
            const tExit = min(tWall, tFar);
            const chord = max(float(0.0), tExit.sub(tEnter));
            const pEnterL = roL.add(rdL.mul(tEnter));

            // Turned satin-brass prism collet caps at top & bottom tips (`|z| > 0.120`)
            const absZEnter = abs(pEnterL.z);
            const isBrassCap = absZEnter.greaterThan(0.12);

            const facetNdotV = clamp(abs(dot(nPrismL, rdL)), 0.0, 1.0);
            const facetFresnel = float(0.065).add(
              float(0.84).mul(pow(max(float(0.0), float(1.0).sub(facetNdotV)), 3.6))
            );

            // Studio-locked directional key + cool rim speculars across all 8 rotating facets
            const hKey = normalize(lKeyL.sub(rdL));
            const hRim = normalize(lRimL.sub(rdL));
            const facetSpecKey = pow(max(float(0.0), dot(nPrismL, hKey)), 52.0);
            const facetSpecRim = pow(max(float(0.0), dot(nPrismL, hRim)), 32.0);
            const facetDiff = clamp(dot(nPrismL, lKeyL).mul(0.5).add(0.5), 0.18, 1.0);

            // Internal rear-facet reflection gleam (authentic double-facet calcite depth!)
            const hExit = normalize(lKeyL.add(rdL));
            const rearGlint = pow(max(float(0.0), dot(nExitL.negate(), hExit)), 28.0).mul(0.45);

            // Exact crystal facet ridge proximity via 2nd-largest |dot(pEnterL, sN_m)|
            const sd0 = abs(dot(pEnterL, sN0));
            const sd1 = abs(dot(pEnterL, sN1));
            const sd2 = abs(dot(pEnterL, sN2));
            const sd3 = abs(dot(pEnterL, sN3));
            const hi01 = max(sd0, sd1);
            const lo01 = min(sd0, sd1);
            const hi23 = max(sd2, sd3);
            const lo23 = min(sd2, sd3);
            const secondSlab = max(min(hi01, hi23), max(lo01, lo23));
            const ridgeGap = max(float(0.0), prismHalfD.sub(secondSlab));
            const ridgeCatch = float(1.0).sub(smoothstep(float(0.0), float(0.0065), ridgeGap));

            If(isBrassCap, () => {
              // Dark machined collet seam groove right where crystal meets brass cap (`|z| in [0.120, 0.127]`)
              const seamGroove = float(1.0).sub(
                smoothstep(float(0.12), float(0.127), absZEnter)
              );
              const capBase = mix(pRawUmber.mul(0.65), pYellowOchre.mul(1.22), facetDiff);
              const capCol = mix(capBase, pRawUmber.mul(0.28), seamGroove.mul(0.75))
                .add(pBoneCalcite.mul(facetSpecKey.mul(0.9).add(facetSpecRim.mul(0.42))));
              accum.addAssign(capCol.mul(throughput));
              throughput.assign(vec3(0.0, 0.0, 0.0));
              Break();
            }).Else(() => {
              // Crisp 3D See-Through Helical Realgar-Amber Edison Filament (`R_coil = 0.017`)
              const rCoil = float(0.017);
              const aC = rdL.x.mul(rdL.x).add(rdL.y.mul(rdL.y));
              const bC = roL.x.mul(rdL.x).add(roL.y.mul(rdL.y));
              const cC = roL.x.mul(roL.x).add(roL.y.mul(roL.y)).sub(rCoil.mul(rCoil));
              const discC = bC.mul(bC).sub(aC.mul(cC));

              const filCore = float(0.0).toVar();
              const filHalo = float(0.0).toVar();

              // Front & rear crossing points on the open cylindrical helix shell
              If(aC.greaterThan(1e-4).and(discC.greaterThan(0.0)), () => {
                const sqC = sqrt(discC);
                const invA = float(1.0).div(aC);
                const tC1 = bC.negate().sub(sqC).mul(invA);
                const tC2 = bC.negate().add(sqC).mul(invA);

                // Front helix wall crossing
                If(tC1.greaterThan(tEnter).and(tC1.lessThan(tExit)), () => {
                  const p1 = roL.add(rdL.mul(tC1));
                  const zMask1 = float(1.0).sub(smoothstep(float(0.088), float(0.104), abs(p1.z)));
                  const sx1 = abs(p1.x).lessThan(1e-5).select(float(1e-5), p1.x);
                  const phi1 = atan(p1.y, sx1);
                  const ph1 = fract(p1.z.mul(40.0).sub(phi1.mul(1.0 / (Math.PI * 2))).add(24.0));
                  const dz1 = abs(ph1.sub(0.5)).div(40.0);
                  const c1 = float(1.0).sub(smoothstep(float(0.0022), float(0.0056), dz1)).mul(zMask1);
                  const g1 = exp(dz1.mul(dz1).mul(-38000.0)).mul(zMask1);
                  filCore.assign(max(filCore, c1));
                  filHalo.addAssign(g1.mul(0.82));
                });

                // Rear helix wall crossing (visible through the open center of the spiral!)
                If(tC2.greaterThan(tEnter).and(tC2.lessThan(tExit)), () => {
                  const p2 = roL.add(rdL.mul(tC2));
                  const zMask2 = float(1.0).sub(smoothstep(float(0.088), float(0.104), abs(p2.z)));
                  const sx2 = abs(p2.x).lessThan(1e-5).select(float(1e-5), p2.x);
                  const phi2 = atan(p2.y, sx2);
                  const ph2 = fract(p2.z.mul(40.0).sub(phi2.mul(1.0 / (Math.PI * 2))).add(24.0));
                  const dz2 = abs(ph2.sub(0.5)).div(40.0);
                  const c2 = float(1.0).sub(smoothstep(float(0.0022), float(0.0058), dz2)).mul(zMask2);
                  const g2 = exp(dz2.mul(dz2).mul(-34000.0)).mul(zMask2);
                  filCore.assign(max(filCore, c2.mul(0.84)));
                  filHalo.addAssign(g2.mul(0.62));
                });
              });

              // Closest point to central Z-axis -> delicate brass lead-in electrode stem + grazing end-on coil view
              const aSafe = max(float(1e-5), aC);
              const tAxisMid = tEnter.add(tExit).mul(0.5);
              const tAxis = clamp(
                mix(tAxisMid, bC.negate().div(aSafe), smoothstep(float(1e-5), float(1e-3), aC)),
                tEnter,
                tExit
              );
              const pAxis = roL.add(rdL.mul(tAxis));
              const rAxis = length(vec3(pAxis.x, pAxis.y, float(0.0)));
              const zInTube = float(1.0).sub(smoothstep(float(0.095), float(0.122), abs(pAxis.z)));

              // End-on circular glowing filament fallback when looking steeply down/up the prism Z-axis
              const axialView = float(1.0).sub(smoothstep(float(0.002), float(0.08), aC));
              const ringDist = abs(rAxis.sub(rCoil));
              const endOnCoil = exp(ringDist.mul(ringDist).mul(-26000.0)).mul(axialView).mul(zInTube);
              filCore.assign(max(filCore, endOnCoil.mul(0.95)));
              filHalo.addAssign(endOnCoil.mul(0.55));

              // Hairline vertical brass/tungsten lead-in electrode anchoring helix into top & bottom collet sockets
              const zInCollet = float(1.0).sub(smoothstep(float(0.114), float(0.124), abs(pAxis.z)));
              const stemMask = float(1.0)
                .sub(smoothstep(float(0.0026), float(0.0048), rAxis))
                .mul(zInCollet)
                .mul(float(1.0).sub(filCore));
              const leadWarm = float(1.0).sub(smoothstep(float(0.088), float(0.12), abs(pAxis.z)));
              const stemCol = mix(pYellowOchre.mul(1.15), pRealgar.mul(2.4), leadWarm);

              // Soft localized incandescent aura tightly hugging the coil envelope (zero bloated blob!)
              const zClamped = clamp(pAxis.z, -0.092, 0.092);
              const dEnv = length(vec3(pAxis.x, pAxis.y, pAxis.z.sub(zClamped)));
              const innerAura = exp(dEnv.mul(dEnv).mul(-1950.0)).mul(0.58);

              // Incandescent Realgar-Amber -> Spore-Sulfur -> White-Gold Bone-Calcite filament radiance
              const hotFilCol = mix(
                pRealgar.mul(2.5),
                mix(pSpore.mul(3.6), pBoneCalcite.mul(4.8), filCore),
                clamp(filCore.mul(0.75).add(filHalo.mul(0.25)), 0.0, 1.0)
              ).mul(filamentPulse);
              const filEmit = hotFilCol
                .mul(filCore.mul(1.05).add(filHalo.mul(0.48)))
                .add(pRealgar.mul(innerAura.mul(1.45).mul(filamentPulse)))
                .add(stemCol.mul(stemMask.mul(0.85)));

              // Clear faceted Bone-Calcite (Iceland Spar) crystal shading:
              // crisp facet speculars, razor 12-edge crystal catchlights, subtle warm internal caustics
              const crystalBodyTint = mix(pRealgar.mul(0.24), pBoneCalcite.mul(0.28), facetDiff)
                .mul(float(0.32).add(chord.mul(1.35)));
              const crystalGlints = pBoneCalcite
                .mul(
                  facetFresnel
                    .mul(0.52)
                    .add(facetSpecKey.mul(1.45))
                    .add(facetSpecRim.mul(0.68))
                    .add(rearGlint)
                    .add(ridgeCatch.mul(0.88))
                )
                .add(pCeladon.mul(ridgeCatch.mul(0.16)));

              const crystalAlpha = clamp(
                filCore
                  .mul(0.92)
                  .add(stemMask.mul(0.72))
                  .add(facetFresnel.mul(0.34))
                  .add(ridgeCatch.mul(0.36))
                  .add(chord.mul(0.72)),
                0.13,
                0.96
              );

              accum.addAssign(crystalBodyTint.add(crystalGlints).add(filEmit).mul(throughput));
              throughput.mulAssign(float(1.0).sub(crystalAlpha));

              const remMax = max(throughput.x, max(throughput.y, throughput.z));
              If(remMax.lessThan(0.014), () => {
                Break();
              });
            });
          });
        });
      });

      // -----------------------------------------------------------------------
      // C. Interior Mirror Wall Hit: Exact Tilted-Plane Edges, Dark-Bronze Cage,
      //    Ceiling Rosette & Refined Filament-Lit Brass Chamfer Catch-Lines
      // -----------------------------------------------------------------------
      const pWall = ro.add(rd.mul(tWall));
      totalDist.addAssign(tWall);

      // Exact branchless 3-smallest signed distances `d_k = dot(pWall, n_k) + CAVITY_HALF`
      // across all 6 tilted mirror planes -> zero rail bloat or strip tearing at ±3.2°!
      const d1 = float(1e5).toVar();
      const d2 = float(1e5).toVar();
      const d3 = float(1e5).toVar();

      for (let k = 0; k < 6; k++) {
        const dk = max(float(0.0), dot(pWall, uMirrorNormals[k]).add(float(CAVITY_HALF)));
        const next3 = min(d3, max(d2, dk));
        const next2 = min(d2, max(d1, dk));
        const next1 = min(d1, dk);
        d3.assign(next3);
        d2.assign(next2);
        d1.assign(next1);
      }

      const edgeDist = d2;
      const cornerDist = d3;
      const aaW = float(0.0022).add(totalDist.mul(0.0012));

      // Sleek patinated dark-bronze interior rail (edgeDist < 0.030) + true 45° beveled brass corner gusset
      const railMask = float(1.0).sub(smoothstep(float(0.03).sub(aaW), float(0.03).add(aaW), edgeDist));
      const gussetDiag = edgeDist.add(cornerDist);
      const gussetMask = float(1.0)
        .sub(smoothstep(float(0.128).sub(aaW), float(0.128).add(aaW), gussetDiag))
        .mul(float(1.0).sub(smoothstep(float(0.074).sub(aaW), float(0.074).add(aaW), edgeDist)));
      const gussetBevelRim = float(1.0)
        .sub(smoothstep(float(0.001), float(0.007).add(aaW), abs(gussetDiag.sub(0.124))))
        .mul(gussetMask);
      // Subtle recessed dome-rivet detail on each interior brass corner gusset
      const rivetDist = length(vec3(edgeDist.sub(0.044), cornerDist.sub(0.056), float(0.0)));
      const rivetMask = float(1.0)
        .sub(smoothstep(float(0.0065), float(0.0095).add(aaW), rivetDist))
        .mul(gussetMask);

      // Turned satin-brass ceiling suspension rosette flush on the +Y ceiling mirror (`hitN.y < -0.5`)
      const isCeilingWall = smoothstep(float(-0.5), float(-0.85), hitN.y);
      const rCeil = length(vec3(pWall.x, float(0.0), pWall.z));
      const rosetteMask = float(1.0)
        .sub(smoothstep(float(0.04).sub(aaW), float(0.04).add(aaW), rCeil))
        .mul(isCeilingWall);
      const rosetteInnerRing = float(1.0).sub(
        smoothstep(float(0.022).sub(aaW), float(0.022).add(aaW), rCeil)
      );

      const solidMetalMask = max(max(railMask, gussetMask), rosetteMask);

      // Refined warm satin-brass beveled inner catch-edge seated flush against the dark-bronze rail (`0.030`)
      const stripCenter = float(0.0324);
      const stripHalfW = float(0.0036).add(aaW.mul(0.75));
      const stripDist = abs(edgeDist.sub(stripCenter));
      const stripMask = float(1.0).sub(smoothstep(stripHalfW.mul(0.32), stripHalfW, stripDist));
      const stripOpacity = stripMask.mul(float(1.0).sub(solidMetalMask));

      const frameMask = clamp(solidMetalMask.add(stripOpacity), 0.0, 1.0);

      If(frameMask.greaterThan(0.004), () => {
        // Point-light shading from swinging Realgar-Amber filament onto the interior bronze/brass frame
        const toLight = uCorePos.sub(pWall);
        const lDistSq = max(float(0.14), dot(toLight, toLight));
        const lDir = normalize(toLight);
        const atten = float(1.0).div(float(1.0).add(lDistSq.mul(1.25)));
        const nDotL = clamp(dot(hitN, lDir), 0.0, 1.0);
        const halfV = normalize(lDir.sub(rd));
        const metalSpec = pow(max(float(0.0), dot(hitN, halfV)), 38.0);

        // Exact material parity with exterior oil-rubbed dark bronze rails vs. beveled satin-brass gussets
        const brassWeight = max(gussetMask, rosetteMask.mul(float(0.95).sub(rosetteInnerRing.mul(0.35))));
        const darkBronzeRail = mix(pRawUmber.mul(0.24), pBoneCalcite.mul(0.065), float(0.28));
        const satinBrassBracket = mix(pRawUmber.mul(0.46), pYellowOchre.mul(0.92), float(0.72));
        const railMetal = mix(darkBronzeRail, satinBrassBracket, brassWeight);

        // Subtle machined bevel highlight along interior gusset hypotenuse & dome rivet catchlight
        const gussetDetail = gussetBevelRim.mul(0.42).add(rivetMask.mul(0.28));
        const chamferBounce = exp(stripDist.mul(-65.0)).mul(0.075).mul(atten);
        const litFrame = railMetal
          .mul(float(0.24).add(nDotL.mul(1.12).mul(atten)).add(gussetDetail.mul(atten)))
          .add(
            mix(pYellowOchre, pBoneCalcite, float(0.32)).mul(
              metalSpec.mul(1.18).add(gussetDetail.mul(0.35)).mul(atten)
            )
          )
          .add(pRealgar.mul(float(0.032).mul(atten).add(chamferBounce)));

        // Filament-lit flush satin-brass optical bevel rim (keeps 40-bounce emerald corridor razor-crisp)
        const distSoften = float(1.0).div(float(1.0).add(totalDist.mul(0.022)));
        const cornerGlint = float(1.0).add(
          float(1.0).sub(smoothstep(float(0.03), float(0.18), cornerDist)).mul(0.32)
        );
        const stripCol = mix(pYellowOchre, pRealgar, float(0.28))
          .mul(float(0.76).add(atten.mul(1.22)).add(metalSpec.mul(0.72)))
          .mul(filamentPulse.mul(distSoften).mul(cornerGlint));

        const combinedFrameCol = mix(stripCol, litFrame, solidMetalMask);
        accum.addAssign(combinedFrameCol.mul(frameMask).mul(throughput));
        throughput.mulAssign(float(1.0).sub(frameMask));

        If(frameMask.greaterThan(0.94), () => {
          Break();
        });
      });

      // Authentic soda-lime emerald glass veil & subtle core bounce scatter across deep echoes
      const coreToWall = uCorePos.sub(pWall);
      const wallCoreAtten = float(1.0).div(float(1.0).add(dot(coreToWall, coreToWall).mul(2.5)));
      const glassDustVeil = mix(pVerdigris, pCeladon, float(0.4))
        .mul(0.0115)
        .add(pRealgar.mul(0.007).mul(wallCoreAtten))
        .mul(float(1.0).sub(smoothstep(float(18.0), float(39.0), bounceF)).mul(0.72).add(0.28));
      accum.addAssign(glassDustVeil.mul(throughput));

      // Apply Authentic Soda-Lime Mirror Optics:
      // Fresnel dielectric reflectivity * Beer-Lambert ferrous-iron verdigris/celadon absorption
      const cosInc = clamp(abs(dot(rd, hitN)), 0.0, 1.0);
      const rMirror = mix(
        float(0.935),
        float(0.97),
        pow(max(float(0.0), float(1.0).sub(cosInc)), 3.0)
      );
      throughput.mulAssign(beerPerBounce.mul(rMirror));

      const maxEnergy = max(throughput.x, max(throughput.y, throughput.z));
      If(maxEnergy.lessThan(0.008), () => {
        Break();
      });

      // Reflect ray cleanly off the tilted mirror plane and step slightly inward
      rd.assign(normalize(reflect(rd, hitN)));
      ro.assign(pWall.add(hitN.mul(0.0005)));
    });

    // Residual depth floor -> shadowed Verdigris-Malachite / Lapis-Slate infinity abyss
    const abyssFloor = mix(pVerdigris.mul(0.062), pLapis.mul(0.042), float(0.32));
    accum.addAssign(abyssFloor.mul(throughput));

    // Composite exterior studio softbox Fresnel reflection + thick beveled soda-lime glass rim
    const rawRadiance = accum.add(extStudioRefl.mul(fresnelExt)).add(glassRimTint);

    // Smooth hue-preserving filmic highlight shoulder (incandescent white-gold filament core)
    const lum = dot(rawRadiance, vec3(0.2126, 0.7152, 0.0722));
    const compressed = rawRadiance.div(vec3(1.0).add(rawRadiance.mul(0.45)));
    const whiteGoldLift = smoothstep(float(1.75), float(5.8), lum).mul(0.3);
    return mix(compressed, pBoneCalcite.mul(0.97), whiteGoldLift);
  });

  infinityMirror.colorNode = evalInfinityMirror();

  return {
    agedBronze,
    satinBrass,
    bezelBrass,
    vernierBrass,
    knurledBrass,
    bluedSteel,
    infinityMirror,
    uCamLocal,
    uMirrorNormals,
    uCorePos,
    uCoreAxis,
    uCoreSpin,
    uPluckFlash,
    uTime,
    uThumbHover,
    uGlassHover,
    dispose: () => {
      agedBronze.dispose();
      satinBrass.dispose();
      bezelBrass.dispose();
      vernierBrass.dispose();
      knurledBrass.dispose();
      bluedSteel.dispose();
      infinityMirror.dispose();
    },
  };
}
