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
  normalView,
  normalWorld,
  normalize,
  positionLocal,
  positionViewDirection,
  positionWorld,
  pow,
  reflect,
  sin,
  smoothstep,
  sqrt,
  transformNormalToView,
  uniform,
  uv,
  vec3,
  cameraPosition,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { mineralToothFbm } from '../../core/tslMaterials';
import {
  APERTURE_HALF,
  BULB_BOT_Y,
  BULB_MAX_R,
  BULB_TOP_Y,
  CAVITY_HALF,
  CEILING_ROSETTE_BOT_Y,
  CEILING_ROSETTE_RADIUS,
  CEILING_ROSETTE_TOP_Y,
  CORD_GRIP_BOT_Y,
  CORD_GRIP_TOP_Y,
  CORD_RADIUS,
  CORD_SOCKET_ATTACH_Y,
  E27_BASE_BOT_Y,
  E27_BASE_TOP_Y,
  ECHO_BOUNCE_COUNT,
  FILAMENT_BOT_R,
  FILAMENT_BOT_Y,
  FILAMENT_LOOPS,
  FILAMENT_TOP_R,
  FILAMENT_TOP_Y,
  FINIAL_TIP_Y,
  FINIAL_TOP_Y,
  GLAND_BOT_Y,
  GLASS_PANE_HALF,
  GROMMET_BOT_Y,
  LAMP_BOUND_RADIUS,
  PRISM_HALF_D,
  PRISM_SLAB_NORM_INV,
  PRISM_SLAB_Y_RATIO,
  PRISM_TRUNC_Y,
  SOCKET_CAP_BOT_Y,
  SOCKET_INSULATOR_BOT_Y,
  SOCKET_KNURL_BOT_Y,
  SOCKET_SKIRT_BOT_Y,
  SOCKET_TOP_Y,
  SOCKET_VENT_BOT_Y,
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
  ceramicInsulator: THREE.MeshStandardNodeMaterial;
  darkCeramic: THREE.MeshStandardNodeMaterial;
  braidedCord: THREE.MeshStandardNodeMaterial;
  ventBaffleGlow: THREE.MeshStandardNodeMaterial;
  edisonFilament: THREE.MeshBasicNodeMaterial;
  filamentAura: THREE.MeshBasicNodeMaterial;
  amberBulbGlass: THREE.MeshBasicNodeMaterial;
  crystalPrismGlass: THREE.MeshBasicNodeMaterial;
  infinityMirror: THREE.MeshBasicNodeMaterial;
  infinityMirrorFront: THREE.MeshBasicNodeMaterial;
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
 * machined bronze chamfers, turned brass bezels, Edison socket knurling, and 4-sided thumbscrews.
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
 * bevel speculars and zero dark-void collapse on shadowed rails, interior Edison hardware, or underside orbits.
 */
export function createEchoStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x1f2429);

  const disposables: Array<{ dispose: () => void }> = [];

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
  addSoftbox(10.0, 6.5, [-3.8, -1.8, 6.4], 0.95, 0.91, 0.85, 0.94);
  // Rear-top cool calcite rim strip
  addSoftbox(13.0, 5.8, [-1.2, 5.2, -5.8], 0.9, 0.95, 1.0, 1.6);
  // Zenith diffused gallery canopy
  addSoftbox(10.0, 10.0, [0.0, 7.6, 0.4], 0.96, 0.94, 0.9, 0.92);
  // Rear-right celadon-slate fill card
  addSoftbox(9.0, 7.0, [5.0, 1.2, -5.6], 0.88, 0.95, 0.98, 1.18);
  // Warm gallery underside table bounce (illuminates underside lock caps & interior ceiling rosette)
  addSoftbox(13.0, 13.0, [0.0, -6.5, 0.8], 0.96, 0.92, 0.86, 0.62);

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
 * matching the PMREM softbox scrims for crisp exterior two-way mirror glass & bulb reflections.
 */
const evalStudioSoftboxDome = Fn(([dir_in]: [TslNode]) => {
  const d = normalize(vec3(dir_in));

  const skyT = clamp(d.y.mul(0.5).add(0.5), 0.0, 1.0);
  const roomBase = mix(vec3(0.018, 0.021, 0.024), vec3(0.044, 0.052, 0.059), skyT);

  const keyDir = normalize(vec3(-0.55, 0.64, 0.53));
  const keyDot = max(float(0.0), dot(d, keyDir));
  const keyBox = smoothstep(float(0.76), float(0.96), keyDot).mul(0.68);

  const rightDir = normalize(vec3(0.76, 0.3, 0.57));
  const rightDot = max(float(0.0), dot(d, rightDir));
  const rightBox = smoothstep(float(0.78), float(0.96), rightDot).mul(0.52);

  const rimDir = normalize(vec3(-0.14, 0.66, -0.74));
  const rimDot = max(float(0.0), dot(d, rimDir));
  const rimBox = smoothstep(float(0.8), float(0.97), rimDot).mul(0.45);

  const broadGlow = pow(max(float(0.0), d.y.mul(0.68).add(0.32)), 3.0).mul(0.09);

  return roomBase
    .add(vec3(0.98, 0.94, 0.87).mul(keyBox.add(rightBox)))
    .add(vec3(0.84, 0.92, 0.98).mul(rimBox))
    .add(vec3(0.88, 0.86, 0.82).mul(broadGlow));
});

/**
 * Creates all materials for Day 22 - Echo (strictly anchored in `MINERAL_PALETTE`):
 * - Oil-rubbed dark bronze frame with burnished 45° chamfer highlights & recessed corner patina
 * - Turned antique satin brass, depth-occluded recessed bezels, crisp engraved vernier collars,
 *   4-sided diamond-knurled brass drums, and tempered Prussian-blued steel helical lead screws
 * - Complete physical Edison lamp suite: glazed Bone-Calcite ceramic grommet, dark stoneware socket
 *   insulator, GPU cubic-Hermite woven braided textile cord, backlit vent baffle, thermally-graded
 *   6-loop squirrel-cage tungsten filament (`aHeat`), hand-blown ST64 amber glass envelope, and
 *   crystal-clear beveled optical calcite prism panes
 * - 2-Layer Semi-Silvered Two-Way Infinity Mirror (`BackSide` 40-bounce raytracer with full Bounce-1..40
 *   Edison lamp parity + `FrontSide` transparent Fresnel & soda-lime celadon glass edge-bevel shell)
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
  const uCorePos = uniform(new THREE.Vector3(0, -0.055, 0));
  const uCoreAxis = uniform(new THREE.Vector3(0, 1, 0));
  const uCoreSpin = uniform(0.35);
  const uPluckFlash = uniform(0);
  const uTime = uniform(0);
  const uThumbHover = uniform(0);
  const uGlassHover = uniform(0);

  // Shared canonical mineral pigment vectors (strictly zero digital neon RGBs)
  const pRawUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const pYellowOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const pBurntSienna = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const pHematite = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear);
  const pCinnabar = vec3(...MINERAL_PALETTE.cinnabarVermilion.rInfLinear);
  const pBoneCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const pWarmKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const pRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const pSpore = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);
  const pVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
  const pCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const pAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const pLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Brushed Oil-Rubbed Dark-Bronze Precision Frame (Edge Burnishing + Inner Corner Patina)
  // ---------------------------------------------------------------------------
  const agedBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.85,
    roughness: 0.33,
    envMapIntensity: 1.12,
  });
  {
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

    // Detect 45-deg machined rail chamfers -> warm hand-rubbed bronze edge burnishing
    const maxAxisN = max(absN.x, max(absN.y, absN.z));
    const chamferMask = smoothstep(float(0.93), float(0.73), maxAxisN);

    // Subtle dark oil-rubbed patina pooling inside inner corner junctions (`worldPos` near corners)
    const apW = abs(positionWorld);
    const innerCornerPatina = smoothstep(float(0.52), float(0.68), min(apW.x, max(apW.y, apW.z)))
      .mul(float(1.0).sub(chamferMask));

    const bronzeDeep = mix(pRawUmber, pBoneCalcite, float(0.03)).mul(0.72);
    const bronzePatina = mix(pRawUmber.mul(0.52), pVerdigris.mul(0.22), float(0.16));
    const bronzeSatin = mix(pRawUmber, pYellowOchre, float(0.15)).add(pBoneCalcite.mul(0.032));
    const bronzeChamfer = mix(pRawUmber, pYellowOchre, float(0.42)).add(pBoneCalcite.mul(0.075));

    const baseBronze = mix(
      bronzeDeep,
      bronzeSatin,
      smoothstep(float(0.32), float(0.68), brushed).mul(0.65)
    );
    const patinated = mix(baseBronze, bronzePatina, innerCornerPatina.mul(0.48));
    const edgeBurnished = mix(patinated, bronzeChamfer, chamferMask.mul(0.68));
    agedBronze.colorNode = mix(
      edgeBurnished,
      pVerdigris.mul(0.2),
      smoothstep(float(0.84), float(0.96), fineGrain).mul(0.035).add(innerCornerPatina.mul(0.025))
    );
    agedBronze.normalNode = perturbNormal3DAniso(
      normalLocal,
      positionLocal,
      vec3(110.0, 110.0, 110.0),
      float(0.0085)
    );
    agedBronze.roughnessNode = clamp(
      float(0.34)
        .add(brushed.sub(0.5).mul(0.055))
        .add(innerCornerPatina.mul(0.07))
        .sub(chamferMask.mul(0.115)),
      0.21,
      0.44
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Turned Antique Satin Brass (Gussets, Screws, Edison Socket, Rosette, Finial)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.22,
    envMapIntensity: 1.28,
  });
  {
    const fineSatin = mineralToothFbm(positionLocal.mul(vec3(195.0, 95.0, 195.0)), float(1.0));
    const microGrain = mineralToothFbm(positionLocal.add(vec3(4.1, 1.8, 6.2)), float(185.0));
    const brassAged = mix(pRawUmber, pYellowOchre, float(0.74)).mul(1.02);
    const brassBright = mix(pYellowOchre, pBoneCalcite, float(0.24)).mul(1.04);
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
      float(0.0036)
    );
    satinBrass.roughnessNode = clamp(
      float(0.22)
        .add(fineSatin.sub(0.5).mul(0.04))
        .sub(chamferEdge.mul(0.045)),
      0.16,
      0.29
    );
  }

  // ---------------------------------------------------------------------------
  // 2b. Stepped Recessed Brass Retaining Bezel (Depth-Occluded Rabbet)
  // ---------------------------------------------------------------------------
  const bezelBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.32,
    envMapIntensity: 1.04,
  });
  {
    const fineBezel = mineralToothFbm(positionLocal, float(145.0));
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
  // 2c. Calibrated Micrometer Vernier Sleeve Brass (Crisp Engraved Calibration Ticks!)
  // ---------------------------------------------------------------------------
  const vernierBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.9,
    roughness: 0.23,
    envMapIntensity: 1.26,
  });
  {
    const p = positionLocal;
    const fineTurn = mineralToothFbm(vec3(p.x.mul(32.0), p.y.mul(240.0), p.z.mul(32.0)), float(1.0));
    const safeX = abs(p.x).lessThan(1e-5).select(float(1e-5), p.x);
    const phi = atan(p.z, safeX);

    // 36 crisp engraved minor ticks, 12 medium half-marks, and 6 tall major index graduations
    const tickU = fract(phi.mul(36.0 / (Math.PI * 2)).add(36.0));
    const tickDist = abs(tickU.sub(0.5));
    const medU = fract(phi.mul(12.0 / (Math.PI * 2)).add(12.0));
    const isMed = abs(medU.sub(0.5)).lessThan(0.09);
    const majorU = fract(phi.mul(6.0 / (Math.PI * 2)).add(6.0));
    const isMajor = abs(majorU.sub(0.5)).lessThan(0.09);
    const tickTopY = isMajor.select(float(0.0465), isMed.select(float(0.0432), float(0.040)));
    const tickWidth = isMajor.select(float(0.088), float(0.062));

    const inVernierBand = smoothstep(float(0.0275), float(0.0292), p.y).mul(
      float(1.0).sub(smoothstep(tickTopY, tickTopY.add(0.0014), p.y))
    );
    const radBarrel = smoothstep(float(0.0245), float(0.0282), length(vec3(p.x, float(0.0), p.z)));
    const tickMask = float(1.0)
      .sub(smoothstep(tickWidth.mul(0.52), tickWidth, tickDist))
      .mul(inVernierBand)
      .mul(radBarrel);

    // Dual engraved circumferential datum witness grooves flanking the calibration scale
    const datumTop = float(1.0)
      .sub(smoothstep(float(0.0006), float(0.0012), abs(p.y.sub(0.0478))))
      .mul(radBarrel);
    const datumBot = float(1.0)
      .sub(smoothstep(float(0.0006), float(0.0012), abs(p.y.sub(0.028))))
      .mul(radBarrel);
    const engraved = clamp(max(tickMask, max(datumTop, datumBot)), 0.0, 1.0);

    const brassBase = mix(
      mix(pRawUmber, pYellowOchre, float(0.72)).mul(1.06),
      mix(pYellowOchre, pBoneCalcite, float(0.26)).mul(1.06),
      fineTurn
    );
    const inkDark = pRawUmber.mul(0.16);
    vernierBrass.colorNode = mix(brassBase, inkDark, engraved.mul(0.92));
    vernierBrass.roughnessNode = clamp(
      mix(float(0.22), float(0.68), engraved).add(fineTurn.sub(0.5).mul(0.04)),
      0.17,
      0.72
    );
    vernierBrass.metalnessNode = mix(float(0.91), float(0.28), engraved);
  }

  // ---------------------------------------------------------------------------
  // 3. 4-Sided Diamond-Knurled Satin Brass (Micrometer Drums, Lock-Rings, Cord-Grip, Socket Collar)
  // ---------------------------------------------------------------------------
  const knurledBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.22,
    envMapIntensity: 1.32,
  });
  {
    const aKnurlValley = attribute('aKnurlValley', 'float');
    const fineGrain = mineralToothFbm(positionLocal, float(84.0));
    const valleyDark = mix(pRawUmber, pBurntSienna, float(0.18)).mul(0.5);
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
      mix(float(0.52), float(0.18), grooveFactor).add(fineGrain.sub(0.5).mul(0.035)),
      0.15,
      0.56
    );
    knurledBrass.metalnessNode = mix(float(0.68), float(0.94), grooveFactor);
  }

  // ---------------------------------------------------------------------------
  // 4. Tempered Prussian-Blued Steel (Helical Lead Screws & Molybdenum Filament Anchor Hooks)
  // ---------------------------------------------------------------------------
  const bluedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.25,
    envMapIntensity: 1.24,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(72.0));
    const rCyl = length(vec3(positionLocal.x, float(0.0), positionLocal.z));
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
  // 5. Glazed Bone-Calcite Porcelain Ceiling Insulator Grommet & Dark Ceramic Socket Ring
  // ---------------------------------------------------------------------------
  const ceramicInsulator = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.18,
    envMapIntensity: 1.15,
  });
  {
    const pore = mineralToothFbm(positionLocal, float(130.0));
    ceramicInsulator.colorNode = mix(pWarmKaolin, pBoneCalcite, smoothstep(float(0.25), float(0.8), pore))
      .mul(0.96);
    ceramicInsulator.roughnessNode = clamp(float(0.17).add(pore.mul(0.07)), 0.14, 0.26);
  }

  const darkCeramic = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.36,
    envMapIntensity: 0.9,
  });
  {
    const glaze = mineralToothFbm(positionLocal, float(115.0));
    darkCeramic.colorNode = mix(pRawUmber.mul(0.24), pAzurite.mul(0.14), float(0.28))
      .mul(float(0.88).add(glaze.mul(0.22)));
    darkCeramic.roughnessNode = clamp(float(0.34).add(glaze.mul(0.1)), 0.28, 0.48);
  }

  // ---------------------------------------------------------------------------
  // 6. Warm Backlit Dark-Bronze Vent Baffle Core Inside Edison Socket Cooling Slots
  // ---------------------------------------------------------------------------
  const ventBaffleGlow = new THREE.MeshStandardNodeMaterial({
    metalness: 0.55,
    roughness: 0.52,
    envMapIntensity: 0.65,
  });
  {
    const pulse = float(1.0).add(sin(uTime.mul(4.2)).mul(0.03)).add(uPluckFlash.mul(0.45));
    // Warm incandescent light escaping upward from the Edison bulb neck into the lower vent openings
    const ventHeightT = float(1.0).sub(
      smoothstep(float(SOCKET_VENT_BOT_Y), float(SOCKET_CAP_BOT_Y), positionLocal.y)
    );
    ventBaffleGlow.colorNode = pRawUmber.mul(0.22);
    ventBaffleGlow.emissiveNode = mix(pBurntSienna, pRealgar, ventHeightT)
      .mul(float(0.18).add(ventHeightT.mul(0.55)))
      .mul(pulse);
  }

  // ---------------------------------------------------------------------------
  // 7. Articulated Woven Dark-Umber & Ochre Textile Pendant Cord (GPU Cubic Hermite Swing!)
  // ---------------------------------------------------------------------------
  const braidedCord = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.86,
    envMapIntensity: 0.72,
  });
  {
    const s = attribute('aCordS', 'float');
    const radOff = attribute('aCordRadial', 'vec3');
    const uvCoord = uv();

    // Cubic Hermite centerline from ceiling ceramic grommet (exiting strictly vertical -Y)
    // down into the swinging Edison socket cord-grip nut (entering strictly coaxial along -uCoreAxis)
    const p0 = vec3(0.0, float(CORE_ANCHOR_Y), 0.0);
    const p1 = uCorePos.add(uCoreAxis.mul(float(CORD_SOCKET_ATTACH_Y)));
    const chordLen = max(float(0.06), length(p1.sub(p0)));
    const t0 = vec3(0.0, float(-1.0), 0.0).mul(chordLen.mul(0.42));
    const t1 = uCoreAxis.negate().mul(chordLen.mul(0.58));

    const s2 = s.mul(s);
    const s3 = s2.mul(s);
    const h00 = s3.mul(2.0).sub(s2.mul(3.0)).add(1.0);
    const h10 = s3.sub(s2.mul(2.0)).add(s);
    const h01 = s3.mul(-2.0).add(s2.mul(3.0));
    const h11 = s3.sub(s2);
    const centerPos = p0.mul(h00).add(t0.mul(h10)).add(p1.mul(h01)).add(t1.mul(h11));

    const dh00 = s2.mul(6.0).sub(s.mul(6.0));
    const dh10 = s2.mul(3.0).sub(s.mul(4.0)).add(1.0);
    const dh01 = s2.mul(-6.0).add(s.mul(6.0));
    const dh11 = s2.mul(3.0).sub(s.mul(2.0));
    const tanDown = normalize(p0.mul(dh00).add(t0.mul(dh10)).add(p1.mul(dh01)).add(t1.mul(dh11)));

    // Exact twist-free orthonormal cross-section basis (eX, eZ) matching (1,0,0) & (0,0,1) at rest
    const upAx = tanDown.negate();
    const denom = max(float(0.15), float(1.0).add(upAx.y));
    const eX = vec3(
      float(1.0).sub(upAx.x.mul(upAx.x).div(denom)),
      upAx.x.negate(),
      upAx.x.mul(upAx.z).negate().div(denom)
    );
    const eZ = vec3(
      upAx.x.mul(upAx.z).negate().div(denom),
      upAx.z.negate(),
      float(1.0).sub(upAx.z.mul(upAx.z).div(denom))
    );

    braidedCord.positionNode = centerPos.add(eX.mul(radOff.x)).add(eZ.mul(radOff.z));

    // Crisp herringbone diagonal woven cloth warp & weft texture (dark raw umber + antique ochre flecks)
    const uStrand = uvCoord.x.mul(8.0);
    const vStrand = uvCoord.y.mul(54.0);
    const diagA = sin(uStrand.add(vStrand).mul(Math.PI * 2)).mul(0.5).add(0.5);
    const diagB = sin(uStrand.sub(vStrand).mul(Math.PI * 2)).mul(0.5).add(0.5);
    const weaveCrest = max(diagA, diagB);
    const weaveGroove = diagA.mul(diagB);
    const lintNoise = mineralToothFbm(vec3(uvCoord.x.mul(24.0), uvCoord.y.mul(160.0), float(1.3)), float(1.0));

    const clothDark = pRawUmber.mul(0.24);
    const clothMid = mix(pRawUmber.mul(0.56), pBurntSienna.mul(0.45), float(0.32));
    const clothOchreThread = mix(pRawUmber.mul(0.45), pYellowOchre.mul(0.72), float(0.58));
    // Alternating subtle golden-ochre tracer strand woven into the dark-umber textile braid
    const tracerMask = smoothstep(float(0.68), float(0.95), diagA).mul(
      smoothstep(float(0.35), float(0.85), sin(uStrand.mul(Math.PI)).mul(0.5).add(0.5))
    );
    const wovenCol = mix(clothDark, clothMid, smoothstep(float(0.18), float(0.86), weaveCrest))
      .mul(float(0.84).add(weaveGroove.mul(0.26)))
      .mul(float(0.9).add(lintNoise.mul(0.2)));
    braidedCord.colorNode = mix(wovenCol, clothOchreThread, tracerMask.mul(0.58));
    braidedCord.roughnessNode = clamp(
      float(0.88).sub(weaveCrest.mul(0.12)).add(lintNoise.sub(0.5).mul(0.08)),
      0.74,
      0.95
    );
  }

  // ---------------------------------------------------------------------------
  // 8. Incandescent 6-Loop Squirrel-Cage Tungsten Filament (`aHeat` Thermal Gradient!)
  // ---------------------------------------------------------------------------
  const edisonFilament = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
  });
  {
    const aHeat = attribute('aHeat', 'float');
    const pulse = float(1.0)
      .add(sin(uTime.mul(4.2)).mul(0.032))
      .add(uPluckFlash.mul(0.52));
    const ndotv = clamp(abs(dot(normalize(normalView), normalize(positionViewDirection))), 0.0, 1.0);
    const coreHot = pow(ndotv, 0.65);

    // Authentic tungsten filament thermal ramp:
    //   aHeat ~ 0.16..0.42 (near molybdenum hooks) -> cooler Cinnabar-Vermilion / Hematite-orange
    //   aHeat ~ 0.42..0.82 (strand limbs) -> saturated glowing Realgar-Amber / Spore-Sulfur
    //   aHeat ~ 0.82..1.00 (hot centerline core) -> white-hot Bone-Calcite gold!
    const h = clamp(aHeat, 0.0, 1.0);
    const hookCol = mix(pHematite.mul(1.15), pCinnabar.mul(2.1), smoothstep(float(0.12), float(0.42), h));
    const midCol = mix(hookCol, pRealgar.mul(2.9), smoothstep(float(0.28), float(0.68), h));
    const goldCol = mix(midCol, pSpore.mul(3.9), smoothstep(float(0.55), float(0.9), h));
    const whiteCore = mix(goldCol, pBoneCalcite.mul(5.2), smoothstep(float(0.66), float(0.98), h.mul(coreHot)));
    const rawRad = whiteCore.mul(pulse);

    // Hue-preserving filmic highlight shoulder so filament glows fiercely without hard clipping
    const lum = dot(rawRad, vec3(0.2126, 0.7152, 0.0722));
    const compressed = rawRad.div(vec3(1.0).add(rawRad.mul(0.38)));
    const hotLift = smoothstep(float(1.6), float(5.2), lum).mul(0.36);
    edisonFilament.colorNode = mix(compressed, pBoneCalcite.mul(0.99), hotLift);
  }

  // ---------------------------------------------------------------------------
  // 9. Delicate Soft Inner Filament Glow Aura (Strictly Hugging the Squirrel-Cage!)
  // ---------------------------------------------------------------------------
  const filamentAura = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.FrontSide,
  });
  {
    const pulse = float(1.0).add(sin(uTime.mul(4.2)).mul(0.03)).add(uPluckFlash.mul(0.58));
    const ndotv = clamp(dot(normalize(normalView), normalize(positionViewDirection)), 0.0, 1.0);
    const softCenter = pow(ndotv, 2.5);
    const auraCol = mix(pRealgar, pSpore, pow(ndotv, 4.0).mul(0.45)).mul(softCenter.mul(0.24).mul(pulse));
    filamentAura.colorNode = auraCol;
    filamentAura.opacityNode = clamp(softCenter.mul(0.55), 0.0, 0.65);
  }

  // ---------------------------------------------------------------------------
  // 10. Hand-Blown Vintage ST64 Amber-Tinted Clear Glass Envelope & Borosilicate Pinch Stem
  // ---------------------------------------------------------------------------
  const amberBulbGlass = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  {
    const nW = normalize(normalWorld);
    const vW = normalize(cameraPosition.sub(positionWorld));
    const ndotv = clamp(dot(nW, vW), 0.0, 1.0);
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const fresnel = clamp(float(0.035).add(float(0.68).mul(pow(oneMinusV, 2.7))), 0.03, 0.74);

    const reflDir = reflect(vW.negate(), nW);
    const studioRefl = evalStudioSoftboxDome(reflDir);

    // Crisp studio key & rim glass specular streaks following the ST64 teardrop waist
    const lKey = normalize(vec3(-0.5, 0.72, 0.48));
    const lRim = normalize(vec3(0.46, 0.56, -0.69));
    const specKey = pow(max(float(0.0), dot(nW, normalize(lKey.add(vW)))), 88.0);
    const specRim = pow(max(float(0.0), dot(nW, normalize(lRim.add(vW)))), 48.0);
    const pulse = float(1.0).add(uPluckFlash.mul(0.42));

    // Warm antique amber-tinted glass rim (crystal clear at center so filament is razor-sharp!)
    const amberRim = mix(pRealgar, pYellowOchre, float(0.48)).mul(
      float(0.06).add(pow(oneMinusV, 2.3).mul(0.75)).mul(pulse)
    );
    const glassGlint = pBoneCalcite.mul(specKey.mul(1.35).add(specRim.mul(0.55)));
    const rawGlass = amberRim.add(studioRefl.mul(fresnel.mul(0.65))).add(glassGlint);
    const alpha = clamp(
      float(0.045)
        .add(pow(oneMinusV, 2.4).mul(0.42))
        .add(specKey.mul(0.55))
        .add(specRim.mul(0.24)),
      0.045,
      0.72
    );

    amberBulbGlass.colorNode = rawGlass.div(max(float(0.08), alpha)).min(vec3(1.6, 1.45, 1.25));
    amberBulbGlass.opacityNode = alpha;
  }

  // ---------------------------------------------------------------------------
  // 11. Crystal-Clear Optical Beveled Calcite/Borosilicate Prism Panes (ZERO Milky Haze!)
  // ---------------------------------------------------------------------------
  const crystalPrismGlass = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  {
    const pL = vec3(positionLocal);
    const nW = normalize(normalWorld);
    const vW = normalize(cameraPosition.sub(positionWorld));
    const ndotv = clamp(dot(nW, vW), 0.0, 1.0);
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const fresnel = clamp(float(0.032).add(float(0.56).mul(pow(oneMinusV, 3.2))), 0.03, 0.62);

    // Exact 4-slab bevel proximity along all 12 faceted crystal ridges (`y` is prism axis in local space)
    const sN0 = vec3(1.0, float(PRISM_SLAB_Y_RATIO), 1.0).mul(float(PRISM_SLAB_NORM_INV));
    const sN1 = vec3(1.0, float(PRISM_SLAB_Y_RATIO), -1.0).mul(float(PRISM_SLAB_NORM_INV));
    const sN2 = vec3(-1.0, float(PRISM_SLAB_Y_RATIO), 1.0).mul(float(PRISM_SLAB_NORM_INV));
    const sN3 = vec3(-1.0, float(PRISM_SLAB_Y_RATIO), -1.0).mul(float(PRISM_SLAB_NORM_INV));
    const sd0 = abs(dot(pL, sN0));
    const sd1 = abs(dot(pL, sN1));
    const sd2 = abs(dot(pL, sN2));
    const sd3 = abs(dot(pL, sN3));
    const hi01 = max(sd0, sd1);
    const lo01 = min(sd0, sd1);
    const hi23 = max(sd2, sd3);
    const lo23 = min(sd2, sd3);
    const secondSlab = max(min(hi01, hi23), max(lo01, lo23));
    const ridgeGap = max(float(0.0), float(PRISM_HALF_D).sub(secondSlab));
    // Crisp beveled glass edge dispersion highlight hugging just inside the brass edge-came ribs
    const bevelEdge = float(1.0).sub(smoothstep(float(0.002), float(0.011), ridgeGap));

    const reflDir = reflect(vW.negate(), nW);
    const studioRefl = evalStudioSoftboxDome(reflDir);
    const lKey = normalize(vec3(-0.5, 0.72, 0.48));
    const lRim = normalize(vec3(0.46, 0.56, -0.69));
    const specKey = pow(max(float(0.0), dot(nW, normalize(lKey.add(vW)))), 78.0);
    const specRim = pow(max(float(0.0), dot(nW, normalize(lRim.add(vW)))), 42.0);

    // Subtle warm filament edge refraction + delicate celadon optical bevel catch
    const bevelCol = mix(pBoneCalcite, pCeladon, float(0.32)).mul(bevelEdge.mul(0.42));
    const facetGlint = pBoneCalcite
      .mul(specKey.mul(1.25).add(specRim.mul(0.55)))
      .add(studioRefl.mul(fresnel.mul(0.55)))
      .add(pRealgar.mul(fresnel.mul(0.1)));
    const rawCrystal = facetGlint.add(bevelCol);

    // Keep pane centers ~92% transparent so the vintage Edison bulb & squirrel-cage filament shine cleanly!
    const alpha = clamp(
      float(0.045)
        .add(fresnel.mul(0.28))
        .add(bevelEdge.mul(0.26))
        .add(specKey.mul(0.58))
        .add(specRim.mul(0.28)),
      0.045,
      0.68
    );

    crystalPrismGlass.colorNode = rawCrystal.div(max(float(0.08), alpha)).min(vec3(1.5, 1.45, 1.38));
    crystalPrismGlass.opacityNode = alpha;
  }

  // ---------------------------------------------------------------------------
  // 12. FrontSide Transparent Exterior Two-Way Mirror Glass (Fresnel + Celadon Bevel Rim)
  // ---------------------------------------------------------------------------
  const infinityMirrorFront = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: false,
  });
  {
    const pEntry = vec3(positionLocal);
    const nOuter = normalize(vec3(normalLocal));
    const rd0 = normalize(pEntry.sub(uCamLocal));
    const viewDir = rd0.negate();
    const ndotv = clamp(dot(nOuter, viewDir), 0.0, 1.0);
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const glassMicroSheen = mineralToothFbm(pEntry, float(46.0)).sub(0.5).mul(0.011);
    const fresnelExt = clamp(
      float(0.046).add(float(0.46).mul(pow(oneMinusV, 4.2))).add(glassMicroSheen.mul(oneMinusV)),
      0.036,
      0.6
    );
    const extStudioRefl = evalStudioSoftboxDome(reflect(rd0, nOuter));

    // Delicate soda-lime celadon glass edge-tint depth along interior mirror bevels
    const aEntry = abs(pEntry);
    const maxE = max(aEntry.x, max(aEntry.y, aEntry.z));
    const minE = min(aEntry.x, min(aEntry.y, aEntry.z));
    const midE = aEntry.x.add(aEntry.y).add(aEntry.z).sub(maxE).sub(minE);
    const apertureEdge = float(APERTURE_HALF).sub(midE);
    const glassRimBevel = float(1.0).sub(smoothstep(float(0.0), float(0.026), apertureEdge));
    const innerBevelLine = float(1.0).sub(smoothstep(float(0.001), float(0.007), abs(apertureEdge.sub(0.009))));
    const glassRimTint = mix(pVerdigris, pCeladon, float(0.54))
      .mul(glassRimBevel.mul(float(0.24).add(fresnelExt.mul(0.44)).add(uGlassHover.mul(0.1))))
      .add(pCeladon.mul(innerBevelLine.mul(0.16)));

    const totalAdd = extStudioRefl.mul(fresnelExt).add(glassRimTint);
    const alpha = clamp(
      fresnelExt.mul(0.44).add(glassRimBevel.mul(0.28)).add(innerBevelLine.mul(0.08)),
      0.02,
      0.66
    );

    // Subtle hue-preserving highlight shoulder matching the interior raytracer
    const unpremul = totalAdd.div(max(float(0.025), alpha));
    const comp = unpremul.div(vec3(1.0).add(unpremul.mul(0.35)));
    infinityMirrorFront.colorNode = comp;
    infinityMirrorFront.opacityNode = alpha;
  }

  // ---------------------------------------------------------------------------
  // 13. BackSide True 40-Bounce Exact Tilted-Plane Analytic TSL Infinity Mirror Raytracer
  //     Renders Bounces 1..40 behind the real 3D Edison lamp with 1:1 connected lamp parity:
  //     ceiling rosette + knurled cord-grip + ceramic grommet -> braided cloth cord ->
  //     ventilated brass Edison socket -> framed crystal prism + brass edge-came ribs +
  //     drop finial -> ST64 amber bulb + 6-loop squirrel-cage glowing tungsten filament!
  // ---------------------------------------------------------------------------
  const infinityMirror = new THREE.MeshBasicNodeMaterial({
    side: THREE.BackSide,
    depthWrite: true,
  });

  const evalInfinityMirror = Fn(() => {
    const pBack = vec3(positionLocal);
    const rd0 = normalize(pBack.sub(uCamLocal));

    // Analytic entry point on the front face of [-GLASS_PANE_HALF, +GLASS_PANE_HALF]^3
    // so optical path distance `totalDist` starts seamlessly from the front glass pane!
    const safeRdX = abs(rd0.x).lessThan(1e-5).select(float(1e-5), rd0.x);
    const safeRdY = abs(rd0.y).lessThan(1e-5).select(float(1e-5), rd0.y);
    const safeRdZ = abs(rd0.z).lessThan(1e-5).select(float(1e-5), rd0.z);
    const tX1 = float(-GLASS_PANE_HALF).sub(uCamLocal.x).div(safeRdX);
    const tX2 = float(GLASS_PANE_HALF).sub(uCamLocal.x).div(safeRdX);
    const tY1 = float(-GLASS_PANE_HALF).sub(uCamLocal.y).div(safeRdY);
    const tY2 = float(GLASS_PANE_HALF).sub(uCamLocal.y).div(safeRdY);
    const tZ1 = float(-GLASS_PANE_HALF).sub(uCamLocal.z).div(safeRdZ);
    const tZ2 = float(GLASS_PANE_HALF).sub(uCamLocal.z).div(safeRdZ);
    const tBoxIn = max(min(tX1, tX2), max(min(tY1, tY2), min(tZ1, tZ2)));
    const tCamToBack = length(pBack.sub(uCamLocal));
    const initDist = max(float(0.0), tCamToBack.sub(max(float(0.0), tBoxIn)));

    // Soda-lime ferrous-iron Beer-Lambert extinction per bounce (hoisted outside loop)
    const sigmaSodaLime = vec3(
      SODA_LIME_EXTINCTION_RGB[0],
      SODA_LIME_EXTINCTION_RGB[1],
      SODA_LIME_EXTINCTION_RGB[2]
    );
    const beerPerBounce = exp(sigmaSodaLime.negate().mul(float(SODA_LIME_CHORD_PER_BOUNCE))).toVar();

    // Orthonormal local basis (cU, cV, cW) of the swinging Edison lamp (cW = coreAxis = local +Y of 3D lamp!)
    const cW = normalize(uCoreAxis).toVar();
    const cU0 = normalize(cross(vec3(0.0, 0.0, 1.0), cW));
    const cV0 = cross(cW, cU0);
    const cosSpin = cos(uCoreSpin);
    const sinSpin = sin(uCoreSpin);
    const cU = cU0.mul(cosSpin).add(cV0.mul(sinSpin)).toVar();
    const cV = cV0.mul(cosSpin).sub(cU0.mul(sinSpin)).toVar();

    // Studio key & rim light directions projected into lamp local basis ONCE per pixel
    const lStudioKeyW = normalize(vec3(-0.5, 0.72, 0.48));
    const lStudioRimW = normalize(vec3(0.46, 0.56, -0.69));
    const lKeyL = vec3(dot(lStudioKeyW, cU), dot(lStudioKeyW, cV), dot(lStudioKeyW, cW)).toVar();
    const lRimL = vec3(dot(lStudioRimW, cU), dot(lStudioRimW, cV), dot(lStudioRimW, cW)).toVar();

    // Organic incandescent shimmer + pluck warmth surge
    const filamentPulse = float(1.0)
      .add(sin(uTime.mul(4.2)).mul(0.032))
      .add(uPluckFlash.mul(0.56))
      .toVar();

    // Braided cloth pendant cord endpoints matching Bounce-0 3D cord EXACTLY
    const cordTop = vec3(0.0, float(CORE_ANCHOR_Y), 0.0);
    const cordBot = uCorePos.add(cW.mul(float(CORD_SOCKET_ATTACH_Y))).toVar();
    const cordVec = cordTop.sub(cordBot).toVar();
    const cordLenSq = max(float(1e-5), dot(cordVec, cordVec)).toVar();
    // Subtle pendulum bend midpoint vector matching the 3D Hermite swing curvature
    const cordChordLen = sqrt(cordLenSq);
    const cordMidBend = vec3(0.0, float(-1.0), 0.0)
      .sub(cW.negate())
      .mul(cordChordLen.mul(0.085))
      .toVar();

    // Faceted optical crystal prism slab normals (local z = lamp axis cW, matching PRISM_SLAB_Y_RATIO exactly!)
    const sN0 = vec3(1.0, 1.0, float(PRISM_SLAB_Y_RATIO)).mul(float(PRISM_SLAB_NORM_INV));
    const sN1 = vec3(1.0, -1.0, float(PRISM_SLAB_Y_RATIO)).mul(float(PRISM_SLAB_NORM_INV));
    const sN2 = vec3(-1.0, 1.0, float(PRISM_SLAB_Y_RATIO)).mul(float(PRISM_SLAB_NORM_INV));
    const sN3 = vec3(-1.0, -1.0, float(PRISM_SLAB_Y_RATIO)).mul(float(PRISM_SLAB_NORM_INV));
    const prismHalfD = float(PRISM_HALF_D);

    // Raytracer state across Bounce 0 wall hit + 40 interior mirror reflections
    const ro = pBack.toVar();
    const rd = rd0.toVar();
    const accum = vec3(0.0, 0.0, 0.0).toVar();
    const throughput = vec3(1.0, 1.0, 1.0).toVar();
    const totalDist = initDist.toVar();

    // Gentle Bounce-0 ambient incandescent glow scatter in the cavity behind the real 3D lamp
    {
      const pEntryEst = pBack.sub(rd0.mul(initDist));
      const toCore0 = uCorePos.sub(pEntryEst);
      const tCl0 = clamp(dot(toCore0, rd0), float(0.0), initDist);
      const clVec0 = pEntryEst.add(rd0.mul(tCl0)).sub(uCorePos);
      const d0Sq = dot(clVec0, clVec0);
      const softBgHalo = exp(d0Sq.mul(-11.5)).mul(0.042).mul(filamentPulse);
      accum.addAssign(mix(pRealgar, pYellowOchre, float(0.28)).mul(softBgHalo));
    }

    Loop(ECHO_BOUNCE_COUNT + 1, ({ i }) => {
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
      // B. Reflected Edison Prism Lamp Assembly (Bounces 1..40; Bounce 0 is Real 3D Mesh!)
      // -----------------------------------------------------------------------
      If(bounceF.greaterThan(0.5), () => {
        // B1. Restrained Warm Realgar-Amber Optical Halo Around Reflected Filament
        const toCore = uCorePos.sub(ro);
        const tClosest = clamp(dot(toCore, rd), float(0.0), tWall);
        const closestVec = ro.add(rd.mul(tClosest)).sub(uCorePos);
        const dCoreSq = dot(closestVec, closestVec);
        const haloTight = exp(dCoreSq.mul(-56.0)).mul(0.2);
        const haloSoft = exp(dCoreSq.mul(-10.5)).mul(0.042);
        const haloColor = mix(pRealgar, pYellowOchre, float(0.28)).mul(
          haloTight.add(haloSoft).mul(filamentPulse)
        );
        accum.addAssign(haloColor.mul(throughput));

        // B2. Connected Top Ceiling Rosette Canopy + Knurled Cord-Grip + Ceramic Grommet
        //     Bridges solidly from +Y mirror plane (`CAVITY_HALF = 0.668`) down to `GROMMET_BOT_Y = 0.514`
        const aRos = rd.x.mul(rd.x).add(rd.z.mul(rd.z));
        const bRos = ro.x.mul(rd.x).add(ro.z.mul(rd.z));
        const rRosBound = float(CEILING_ROSETTE_RADIUS);
        const cRos = ro.x.mul(ro.x).add(ro.z.mul(ro.z)).sub(rRosBound.mul(rRosBound));
        const discRos = bRos.mul(bRos).sub(aRos.mul(cRos));

        If(aRos.greaterThan(1e-5).and(discRos.greaterThan(0.0)), () => {
          const sqR = sqrt(discRos);
          const invAR = float(1.0).div(aRos);
          const tR0 = max(float(1e-4), bRos.negate().sub(sqR).mul(invAR));
          const tR1 = min(tWall, bRos.negate().add(sqR).mul(invAR));
          If(tR1.greaterThan(tR0), () => {
            const tRosMid = clamp(bRos.negate().mul(invAR), tR0, tR1);
            const pRos = ro.add(rd.mul(tRosMid));
            If(pRos.y.greaterThan(float(GROMMET_BOT_Y)).and(pRos.y.lessThan(float(CAVITY_HALF + 0.002))), () => {
              const rAxisRos = length(vec3(pRos.x, float(0.0), pRos.z));
              // Stepped physical profile matching Bounce-0 ceiling rosette + collet + gland + ceramic grommet
              const isUpperStem = pRos.y.greaterThan(float(CEILING_ROSETTE_TOP_Y));
              const isUpperCanopy = pRos.y.greaterThan(float(CEILING_ROSETTE_BOT_Y));
              const isCollet = pRos.y.greaterThan(float(CORD_GRIP_BOT_Y)).and(pRos.y.lessThan(float(CORD_GRIP_TOP_Y)));
              const isGrommet = pRos.y.lessThan(float(GLAND_BOT_Y));
              const canopyProfileR = isUpperStem.select(
                float(0.032),
                mix(
                  float(0.054),
                  float(CEILING_ROSETTE_RADIUS),
                  smoothstep(float(CEILING_ROSETTE_BOT_Y), float(CEILING_ROSETTE_TOP_Y), pRos.y)
                )
              );
              const targetR = isUpperCanopy.select(
                canopyProfileR,
                isCollet.select(float(0.0315), isGrommet.select(float(0.0195), float(0.025)))
              );
              If(rAxisRos.lessThan(targetR), () => {
                // Compute true front cylindrical surface normal for rich PBR-matched brass IBL reflections
                const halfChordR = sqrt(max(float(0.0), targetR.mul(targetR).sub(rAxisRos.mul(rAxisRos))));
                const tSurfRos = max(tR0, tRosMid.sub(halfChordR.div(sqrt(max(float(1e-5), aRos)))));
                const pSurfRos = ro.add(rd.mul(tSurfRos));
                const nRosW = normalize(vec3(pSurfRos.x, float(-0.18), pSurfRos.z));
                const sxR = abs(pSurfRos.x).lessThan(1e-5).select(float(1e-5), pSurfRos.x);
                const phiR = atan(pSurfRos.z, sxR);
                const reflRos = evalStudioSoftboxDome(reflect(rd, nRosW));
                const diffRos = max(float(0.0), dot(nRosW, lStudioKeyW));
                const specRos = pow(max(float(0.0), dot(nRosW, normalize(lStudioKeyW.sub(rd)))), 48.0);
                const knurlRos = sin(phiR.mul(28.0).add(pSurfRos.y.mul(420.0)))
                  .mul(sin(phiR.mul(28.0).sub(pSurfRos.y.mul(420.0))))
                  .mul(0.5)
                  .add(0.5)
                  .mul(isCollet.select(float(1.0), float(0.0)));
                const brassTint = mix(pRawUmber.mul(0.32), pYellowOchre.mul(1.06), float(0.76));
                const bronzeStemTint = mix(pRawUmber.mul(0.22), pYellowOchre.mul(0.45), float(0.28));
                const metalTint = isUpperStem.select(bronzeStemTint, brassTint);
                const rosMetal = metalTint
                  .mul(float(0.18).add(diffRos.mul(0.55)).add(reflRos.mul(1.65)))
                  .mul(float(1.0).sub(knurlRos.mul(0.36)))
                  .add(pBoneCalcite.mul(specRos.mul(0.95)))
                  .add(pRealgar.mul(0.1).mul(filamentPulse));
                const rosCeramic = mix(pWarmKaolin.mul(0.46), pBoneCalcite.mul(0.98), diffRos.mul(0.7).add(0.3))
                  .add(pBoneCalcite.mul(specRos.mul(0.42)));
                const rosCol = isGrommet.select(rosCeramic, rosMetal);
                const rosEdge = float(1.0).sub(smoothstep(targetR.mul(0.86), targetR, rAxisRos));
                accum.addAssign(rosCol.mul(rosEdge).mul(throughput));
                throughput.mulAssign(float(1.0).sub(rosEdge.mul(0.97)));
              });
            });
          });
        });

        // B3. Articulated Woven Dark-Umber & Ochre Braided Textile Pendant Cord
        const oaW = ro.sub(cordBot);
        const dOa = dot(rd, oaW);
        const wOa = dot(cordVec, oaW);
        const dW = dot(rd, cordVec);
        const denomW = max(float(1e-5), cordLenSq.sub(dW.mul(dW)));
        const sCord = clamp(wOa.sub(dOa.mul(dW)).div(denomW), 0.0, 1.0);
        const tCord = max(float(0.0), dOa.negate().add(dW.mul(sCord)));

        If(tCord.greaterThan(1e-4).and(tCord.lessThan(tWall)), () => {
          const pRayC = ro.add(rd.mul(tCord));
          // Include subtle curved pendulum swing bow matching the 3D Hermite cord
          const bendW = sCord.mul(float(1.0).sub(sCord)).mul(4.0);
          const pSegC = cordBot.add(cordVec.mul(sCord)).add(cordMidBend.mul(bendW));
          const diffC = pRayC.sub(pSegC);
          const dCordSq = dot(diffC, diffC);
          const cordRad = float(CORD_RADIUS).add(totalDist.mul(0.00042));
          If(dCordSq.lessThan(cordRad.mul(cordRad)), () => {
            const rNorm = sqrt(max(float(0.0), dCordSq)).div(cordRad);
            const cylLit = sqrt(max(float(0.05), float(1.0).sub(rNorm.mul(rNorm))));
            // Diagonal herringbone braided cloth pattern matching Bounce-0 woven textile cord
            const braidTex = sin(sCord.mul(88.0).add(rNorm.mul(7.0))).mul(0.5).add(0.5);
            const clothDark = pRawUmber.mul(0.28);
            const clothOchre = mix(pRawUmber.mul(0.55), pYellowOchre.mul(0.86), float(0.45));
            const cordWarmth = float(1.0).sub(sCord).mul(0.32).add(0.68);
            const cordCol = mix(clothDark, clothOchre, braidTex.mul(0.62))
              .mul(cylLit.mul(0.75).add(0.32))
              .mul(cordWarmth);
            const cordEdge = float(1.0).sub(smoothstep(float(0.62), float(1.0), rNorm));
            accum.addAssign(cordCol.mul(cordEdge).mul(throughput));
            throughput.mulAssign(float(1.0).sub(cordEdge.mul(0.95)));
          });
        });

        // B4. Full Swinging Vintage Edison Lamp Head (Expanded Bounding Sphere `r = LAMP_BOUND_RADIUS`)
        //     Socket + Cord-Grip Nut + Vents + Knurl + Gallery + Crystal Prism Cage + Finial + ST64 Bulb + Squirrel-Cage Filament!
        const relCore = ro.sub(uCorePos);
        const bSphere = dot(relCore, rd);
        const cSphere = dot(relCore, relCore).sub(float(LAMP_BOUND_RADIUS * LAMP_BOUND_RADIUS));
        const discSphere = bSphere.mul(bSphere).sub(cSphere);

        If(discSphere.greaterThan(0.0), () => {
          const sqrtDisc = sqrt(discSphere);
          const tSphFar = bSphere.negate().add(sqrtDisc);
          const tSphNear = bSphere.negate().sub(sqrtDisc);

          If(tSphFar.greaterThan(1e-4).and(tSphNear.lessThan(tWall)), () => {
            const roL = vec3(dot(relCore, cU), dot(relCore, cV), dot(relCore, cW)).toVar();
            const rdL = vec3(dot(rd, cU), dot(rd, cV), dot(rd, cW)).toVar();
            const tSpanMin = max(float(1e-4), tSphNear);
            const tSpanMax = min(tWall, tSphFar);

            // Closest approach to local lamp Z-axis (used for Edison socket, E27 base, finial, & filament)
            const aCyl = rdL.x.mul(rdL.x).add(rdL.y.mul(rdL.y));
            const bCyl = roL.x.mul(rdL.x).add(roL.y.mul(rdL.y));
            const aCylSafe = max(float(1e-5), aCyl);

            // ---------------------------------------------------------------
            // B4a. Machined Vintage Brass Edison Socket & Cord-Grip Nut (`zL in [0.146, 0.316]`)
            // ---------------------------------------------------------------
            const rSockBound = float(0.054);
            const cSock = roL.x.mul(roL.x).add(roL.y.mul(roL.y)).sub(rSockBound.mul(rSockBound));
            const discSock = bCyl.mul(bCyl).sub(aCyl.mul(cSock));
            const hitOpaqueHardware = float(0.0).toVar();

            If(aCyl.greaterThan(1e-5).and(discSock.greaterThan(0.0)), () => {
              const sqS = sqrt(discSock);
              const invAS = float(1.0).div(aCylSafe);
              const tS0 = max(tSpanMin, bCyl.negate().sub(sqS).mul(invAS));
              const tS1 = min(tSpanMax, bCyl.negate().add(sqS).mul(invAS));
              If(tS1.greaterThan(tS0), () => {
                // Evaluate exact stepped profile along ray intersection with socket/finial cylinder
                const tSCheck = clamp(bCyl.negate().mul(invAS), tS0, tS1);
                const pS = roL.add(rdL.mul(tSCheck));
                const rS = length(vec3(pS.x, pS.y, float(0.0)));

                // Upper Edison Brass Socket (`zL in [0.146, SOCKET_TOP_Y]`)
                If(pS.z.greaterThan(float(0.146)).and(pS.z.lessThan(float(SOCKET_TOP_Y))), () => {
                  const isNut = pS.z.greaterThan(float(0.274));
                  const isCap = pS.z.greaterThan(float(SOCKET_CAP_BOT_Y)).and(pS.z.lessThan(float(0.274)));
                  const isVent = pS.z.greaterThan(float(SOCKET_VENT_BOT_Y)).and(pS.z.lessThan(float(SOCKET_CAP_BOT_Y)));
                  const isKnurl = pS.z.greaterThan(float(SOCKET_KNURL_BOT_Y)).and(pS.z.lessThan(float(SOCKET_VENT_BOT_Y)));
                  const isGallery = pS.z.lessThan(float(SOCKET_SKIRT_BOT_Y));
                  const isInsulator = pS.z
                    .greaterThan(float(SOCKET_INSULATOR_BOT_Y))
                    .and(pS.z.lessThan(float(SOCKET_SKIRT_BOT_Y)));
                  const capT = smoothstep(float(SOCKET_CAP_BOT_Y), float(0.274), pS.z);
                  const profR = isNut.select(
                    float(0.024),
                    isCap.select(
                      mix(float(0.049), float(0.027), capT),
                      isKnurl.select(
                        float(0.052),
                        isInsulator.select(float(0.042), isGallery.select(float(0.044), float(0.049)))
                      )
                    )
                  );
                  If(rS.lessThan(profR), () => {
                    // Exact front cylindrical hit + studio metallic IBL reflection matching Bounce-0 PBR brass!
                    const halfChord = sqrt(max(float(0.0), profR.mul(profR).sub(rS.mul(rS))));
                    const tSurf = max(tS0, tSCheck.sub(halfChord.div(sqrt(aCylSafe))));
                    const pSurf = roL.add(rdL.mul(tSurf));
                    const nSurf = normalize(vec3(pSurf.x, pSurf.y, float(0.0)));
                    const nSurfW = normalize(cU.mul(nSurf.x).add(cV.mul(nSurf.y)));
                    const sxS = abs(pSurf.x).lessThan(1e-5).select(float(1e-5), pSurf.x);
                    const phiS = atan(pSurf.y, sxS);

                    const studioMetalRefl = evalStudioSoftboxDome(reflect(rd, nSurfW));
                    const diffS = max(float(0.0), dot(nSurf, lKeyL));
                    const rimS = max(float(0.0), dot(nSurf, lRimL));
                    const hKeyS = normalize(lKeyL.sub(rdL));
                    const specS = pow(max(float(0.0), dot(nSurf, hKeyS)), 54.0);
                    // Crisp machined horizontal lathe step grooves separating socket bands
                    const seamG = min(
                      abs(pSurf.z.sub(float(0.274))),
                      min(
                        abs(pSurf.z.sub(float(SOCKET_CAP_BOT_Y))),
                        min(
                          abs(pSurf.z.sub(float(SOCKET_VENT_BOT_Y))),
                          min(abs(pSurf.z.sub(float(SOCKET_KNURL_BOT_Y))), abs(pSurf.z.sub(float(SOCKET_SKIRT_BOT_Y))))
                        )
                      )
                    );
                    const seamDark = float(1.0).sub(smoothstep(float(0.0008), float(0.0032), seamG));

                    // Warm upward incandescent bounce onto lower socket skirt
                    const filBounce = float(1.0).sub(smoothstep(float(0.146), float(0.235), pSurf.z)).mul(0.32);

                    // 12 radial dark/amber-lit cooling ventilation slots on the upper bell!
                    const ventSlot = smoothstep(float(-0.18), float(0.22), cos(phiS.mul(12.0))).mul(
                      isVent.select(float(1.0), float(0.0))
                    );
                    // Fine diamond knurling relief on the socket grip collar!
                    const knurlRelief = sin(phiS.mul(40.0).add(pSurf.z.mul(480.0)))
                      .mul(sin(phiS.mul(40.0).sub(pSurf.z.mul(480.0))))
                      .mul(0.5)
                      .add(0.5)
                      .mul(isKnurl.select(float(1.0), float(0.0)));

                    const brassAlbedo = mix(pRawUmber.mul(0.36), pYellowOchre.mul(1.08), float(0.76));
                    const brassBody = brassAlbedo
                      .mul(float(0.16).add(diffS.mul(0.48)).add(rimS.mul(0.18)).add(studioMetalRefl.mul(1.85)))
                      .mul(float(1.0).sub(knurlRelief.mul(0.42)))
                      .mul(float(1.0).sub(seamDark.mul(0.62)))
                      .add(pBoneCalcite.mul(specS.mul(1.05)))
                      .add(pRealgar.mul(filBounce.mul(filamentPulse)));
                    const ventRecess = mix(pRawUmber.mul(0.12), pRealgar.mul(0.95).mul(filamentPulse), float(0.48));
                    const sockCol = mix(brassBody, ventRecess, ventSlot.mul(0.9));
                    const ceramicDarkCol = pRawUmber.mul(0.16).add(pRealgar.mul(0.08).mul(filamentPulse));
                    const galleryBronze = mix(pRawUmber.mul(0.22), pYellowOchre.mul(0.5), float(0.26))
                      .mul(float(0.22).add(diffS.mul(0.52)).add(studioMetalRefl.mul(1.3)))
                      .add(pRealgar.mul(0.16).mul(filamentPulse));
                    const withIns = isInsulator.select(ceramicDarkCol, sockCol);
                    const finalSockCol = isGallery.and(isInsulator.not()).select(galleryBronze, withIns);

                    accum.addAssign(finalSockCol.mul(throughput));
                    throughput.assign(vec3(0.0, 0.0, 0.0));
                    hitOpaqueHardware.assign(float(1.0));
                  });
                });

                // Bottom Machined Satin-Brass Drop Finial (`zL in [FINIAL_TIP_Y, FINIAL_TOP_Y]`)
                If(
                  hitOpaqueHardware
                    .lessThan(0.5)
                    .and(pS.z.greaterThan(float(FINIAL_TIP_Y)))
                    .and(pS.z.lessThan(float(FINIAL_TOP_Y))),
                  () => {
                    const fT = clamp(
                      pS.z.sub(float(FINIAL_TIP_Y)).div(float(FINIAL_TOP_Y - FINIAL_TIP_Y)),
                      0.0,
                      1.0
                    );
                    const finR = fT
                      .lessThan(0.58)
                      .select(
                        sin(fT.div(0.58).mul(Math.PI * 0.88)).mul(0.0195).add(0.002),
                        mix(float(0.011), float(0.034), smoothstep(float(0.58), float(1.0), fT))
                      );
                    If(rS.lessThan(finR), () => {
                      const halfChordF = sqrt(max(float(0.0), finR.mul(finR).sub(rS.mul(rS))));
                      const tSurfF = max(tS0, tSCheck.sub(halfChordF.div(sqrt(aCylSafe))));
                      const pSurfF = roL.add(rdL.mul(tSurfF));
                      const nFinL = normalize(vec3(pSurfF.x, pSurfF.y, float(0.0)));
                      const nFinW = normalize(cU.mul(nFinL.x).add(cV.mul(nFinL.y)));
                      const reflFin = evalStudioSoftboxDome(reflect(rd, nFinW));
                      const diffFin = max(float(0.0), dot(nFinL, lKeyL));
                      const specFin = pow(max(float(0.0), dot(nFinL, normalize(lKeyL.sub(rdL)))), 48.0);
                      const brassFin = mix(pRawUmber.mul(0.36), pYellowOchre.mul(1.08), float(0.76));
                      const finCol = brassFin
                        .mul(float(0.16).add(diffFin.mul(0.48)).add(reflFin.mul(1.85)))
                        .add(pBoneCalcite.mul(specFin.mul(0.95)))
                        .add(pRealgar.mul(fT.mul(0.24).mul(filamentPulse)));
                      accum.addAssign(finCol.mul(throughput));
                      throughput.assign(vec3(0.0, 0.0, 0.0));
                      hitOpaqueHardware.assign(float(1.0));
                    });
                  }
                );
              });
            });

            If(hitOpaqueHardware.greaterThan(0.5), () => {
              Break();
            });

            // ---------------------------------------------------------------
            // B4b. Truncated 8-Facet Optical Crystal Prism + Brass Edge-Came Ribs +
            //      E27 Base + ST64 Amber Bulb + 6-Loop Squirrel-Cage Tungsten Filament!
            // ---------------------------------------------------------------
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

            // Clamp top & bottom by exact truncation planes (`zL in [-PRISM_TRUNC_Y, +PRISM_TRUNC_Y]`)
            const absDZ = abs(rdL.z);
            If(absDZ.greaterThan(1e-5), () => {
              const invDZ = float(1.0).div(rdL.z);
              const tz0 = float(-PRISM_TRUNC_Y).sub(roL.z).mul(invDZ);
              const tz1 = float(PRISM_TRUNC_Y).sub(roL.z).mul(invDZ);
              const tMinZ = min(tz0, tz1);
              const tMaxZ = max(tz0, tz1);
              If(tMinZ.greaterThan(tNear), () => {
                tNear.assign(tMinZ);
                nPrismL.assign(vec3(0.0, 0.0, rdL.z.greaterThan(0.0).select(float(-1.0), float(1.0))));
              });
              If(tMaxZ.lessThan(tFar), () => {
                tFar.assign(tMaxZ);
                nExitL.assign(vec3(0.0, 0.0, rdL.z.greaterThan(0.0).select(float(1.0), float(-1.0))));
              });
            }).Else(() => {
              If(abs(roL.z).greaterThan(float(PRISM_TRUNC_Y)), () => {
                tNear.assign(float(1e5));
              });
            });

            If(tFar.greaterThan(tNear).and(tFar.greaterThan(1e-4)).and(tNear.lessThan(tWall)), () => {
              const tEnter = max(float(0.0), tNear);
              const tExit = min(tWall, tFar);
              const pEnterL = roL.add(rdL.mul(tEnter));

              const facetNdotV = clamp(abs(dot(nPrismL, rdL)), 0.0, 1.0);
              const facetFresnel = float(0.04).add(
                float(0.62).mul(pow(max(float(0.0), float(1.0).sub(facetNdotV)), 3.4))
              );

              const hKey = normalize(lKeyL.sub(rdL));
              const hRim = normalize(lRimL.sub(rdL));
              const facetSpecKey = pow(max(float(0.0), dot(nPrismL, hKey)), 64.0);
              const facetSpecRim = pow(max(float(0.0), dot(nPrismL, hRim)), 36.0);
              const facetDiff = clamp(dot(nPrismL, lKeyL).mul(0.5).add(0.5), 0.18, 1.0);

              const hExit = normalize(lKeyL.add(rdL));
              const rearGlint = pow(max(float(0.0), dot(nExitL.negate(), hExit)), 32.0).mul(0.35);

              // Exact brass edge-came prism rib detection along all 12 facet ridges + spider claws!
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
              const eqGap = abs(pEnterL.z);
              const ribDist = min(ridgeGap, eqGap.mul(0.85));
              const brassCameMask = float(1.0).sub(smoothstep(float(0.0042), float(0.0072), ribDist));
              // Spider gallery claw grip accent on the 4 upper ridges (`z > 0.095`)
              const clawGrip = float(1.0)
                .sub(smoothstep(float(0.007), float(0.011), ridgeGap))
                .mul(smoothstep(float(0.092), float(0.115), pEnterL.z));
              const frameMetalMask = max(brassCameMask, clawGrip);

              // Crystal bevel edge catch just inside the brass came ribs
              const bevelCatch = float(1.0)
                .sub(smoothstep(float(0.006), float(0.014), ribDist))
                .mul(float(1.0).sub(frameMetalMask));

              // Threaded Brass E27 Screw Base inside top of prism (`zL in [E27_BASE_BOT_Y, PRISM_TRUNC_Y]`)
              const tAxisMid = tEnter.add(tExit).mul(0.5);
              const tAxis = clamp(
                mix(tAxisMid, bCyl.negate().div(aCylSafe), smoothstep(float(1e-5), float(1e-3), aCyl)),
                tEnter,
                tExit
              );
              const pAxis = roL.add(rdL.mul(tAxis));
              const rAxis = length(vec3(pAxis.x, pAxis.y, float(0.0)));
              const isE27Hit = pAxis.z
                .greaterThan(float(E27_BASE_BOT_Y))
                .and(pAxis.z.lessThan(float(E27_BASE_TOP_Y)))
                .and(rAxis.lessThan(float(0.032)));

              // Dramatic 6-Loop Squirrel-Cage Hairpin Glowing Tungsten Filament!
              // Intersect tapered squirrel-cage envelope (`r_cage = 0.031` avg, `z in [FILAMENT_BOT_Y, FILAMENT_TOP_Y]`)
              const rFilAvg = float((FILAMENT_TOP_R + FILAMENT_BOT_R) * 0.5);
              const cFil = roL.x.mul(roL.x).add(roL.y.mul(roL.y)).sub(rFilAvg.mul(rFilAvg));
              const discFil = bCyl.mul(bCyl).sub(aCyl.mul(cFil));

              const filCore = float(0.0).toVar();
              const filHalo = float(0.0).toVar();
              const filHeat = float(0.85).toVar();

              If(aCyl.greaterThan(1e-4).and(discFil.greaterThan(0.0)), () => {
                const sqF = sqrt(discFil);
                const invAF = float(1.0).div(aCylSafe);
                const tF1 = bCyl.negate().sub(sqF).mul(invAF);
                const tF2 = bCyl.negate().add(sqF).mul(invAF);

                // Helper math for 6 V-hairpin squirrel-cage loops (`12` luminous angled vertical strands!)
                const evalCageCrossing = (tHit: TslNode, weight: number) => {
                  If(tHit.greaterThan(tEnter).and(tHit.lessThan(tExit)), () => {
                    const pH = roL.add(rdL.mul(tHit));
                    const zNorm = clamp(
                      pH.z.sub(float(FILAMENT_BOT_Y)).div(float(FILAMENT_TOP_Y - FILAMENT_BOT_Y)),
                      0.0,
                      1.0
                    );
                    const inZ = smoothstep(float(FILAMENT_BOT_Y - 0.005), float(FILAMENT_BOT_Y + 0.004), pH.z).mul(
                      float(1.0).sub(smoothstep(float(FILAMENT_TOP_Y - 0.004), float(FILAMENT_TOP_Y + 0.005), pH.z))
                    );
                    const sxH = abs(pH.x).lessThan(1e-5).select(float(1e-5), pH.x);
                    const phiH = atan(pH.y, sxH);
                    // Triangular V-zig-zag wave matching the exact 6-loop squirrel-cage hairpin geometry!
                    const loopPhase = fract(phiH.mul(float(FILAMENT_LOOPS / (Math.PI * 2))).add(12.0));
                    const triWave = abs(loopPhase.mul(2.0).sub(1.0)); // 0 at top hooks, 1 at bottom hooks
                    // Distance in normalized phase-z space to the closest of the 12 hairpin strands
                    const strandDist = abs(triWave.sub(float(1.0).sub(zNorm)));
                    const cStrand = float(1.0)
                      .sub(smoothstep(float(0.045), float(0.115), strandDist))
                      .mul(inZ);
                    const gStrand = exp(strandDist.mul(strandDist).mul(-120.0)).mul(inZ);
                    // Thermal conduction cooling right at the upper & lower molybdenum anchor hooks
                    const endHeat = float(0.18).add(
                      pow(max(float(0.0), sin(zNorm.mul(Math.PI))), 0.45).mul(0.82)
                    );
                    If(cStrand.mul(weight).greaterThan(filCore), () => {
                      filCore.assign(cStrand.mul(weight));
                      filHeat.assign(endHeat);
                    });
                    filHalo.addAssign(gStrand.mul(weight * 0.72));
                  });
                };

                evalCageCrossing(tF1, 1.0);
                evalCageCrossing(tF2, 0.86);
              });

              // End-on glowing 6-loop lantern crown view when looking steeply down/up through top/bottom glass
              const axialView = float(1.0).sub(smoothstep(float(0.004), float(0.09), aCyl));
              const zInBulb = float(1.0).sub(smoothstep(float(0.068), float(0.088), abs(pAxis.z)));
              const ringDist = abs(rAxis.sub(rFilAvg));
              const endOnCage = exp(ringDist.mul(ringDist).mul(-14000.0)).mul(axialView).mul(zInBulb);
              filCore.assign(max(filCore, endOnCage.mul(0.92)));
              filHalo.addAssign(endOnCage.mul(0.48));

              // Delicate borosilicate glass pinch stem & central arbor inside the squirrel-cage
              const stemMask = float(1.0)
                .sub(smoothstep(float(0.0032), float(0.006), rAxis))
                .mul(
                  smoothstep(float(FILAMENT_BOT_Y - 0.005), float(FILAMENT_BOT_Y + 0.006), pAxis.z).mul(
                    float(1.0).sub(smoothstep(float(BULB_TOP_Y - 0.005), float(BULB_TOP_Y + 0.004), pAxis.z))
                  )
                )
                .mul(float(1.0).sub(filCore));
              const stemCol = mix(pBoneCalcite.mul(0.45), pRealgar.mul(0.95), float(0.55));

              // Subtle ST64 amber glass envelope grazing Fresnel rim inside the crystal prism
              const bulbProfileR = mix(
                float(BULB_MAX_R),
                float(0.033),
                smoothstep(float(-0.052), float(BULB_TOP_Y), pAxis.z)
              );
              const inBulbY = smoothstep(float(BULB_BOT_Y), float(BULB_BOT_Y + 0.025), pAxis.z).mul(
                float(1.0).sub(smoothstep(float(BULB_TOP_Y - 0.015), float(BULB_TOP_Y), pAxis.z))
              );
              const bulbRim = smoothstep(bulbProfileR.mul(0.72), bulbProfileR, rAxis)
                .mul(float(1.0).sub(smoothstep(bulbProfileR, bulbProfileR.add(0.008), rAxis)))
                .mul(inBulbY);
              const bulbRimCol = mix(pRealgar, pYellowOchre, float(0.45)).mul(bulbRim.mul(0.52).mul(filamentPulse));

              // Tight localized warm incandescent aura hugging the squirrel-cage envelope
              const zClamped = clamp(pAxis.z, float(FILAMENT_BOT_Y), float(FILAMENT_TOP_Y));
              const dEnv = length(vec3(pAxis.x, pAxis.y, pAxis.z.sub(zClamped)));
              const innerAura = exp(dEnv.mul(dEnv).mul(-1250.0)).mul(0.48);

              // Exact thermal filament color ramp matching Bounce-0 3D filament (`Cinnabar -> Realgar -> Spore -> White-Gold`)
              const hookC = mix(pHematite.mul(1.2), pCinnabar.mul(2.1), smoothstep(float(0.15), float(0.45), filHeat));
              const midC = mix(hookC, pRealgar.mul(2.85), smoothstep(float(0.3), float(0.7), filHeat));
              const hotFilCol = mix(
                midC,
                mix(pSpore.mul(3.8), pBoneCalcite.mul(5.0), filCore),
                smoothstep(float(0.55), float(0.92), filHeat.mul(filCore.mul(0.8).add(0.2)))
              ).mul(filamentPulse);

              const filEmit = hotFilCol
                .mul(filCore.mul(1.1).add(filHalo.mul(0.42)))
                .add(pRealgar.mul(innerAura.mul(1.2).mul(filamentPulse)))
                .add(stemCol.mul(stemMask.mul(0.55)))
                .add(bulbRimCol);

              // E27 helical brass screw collar shading when ray hits top neck inside prism
              const e27Thread = sin(pAxis.z.mul(480.0)).mul(0.5).add(0.5);
              const e27Col = mix(pRawUmber.mul(0.55), pYellowOchre.mul(1.22), e27Thread.mul(0.55).add(0.45))
                .add(pRealgar.mul(0.22).mul(filamentPulse));

              // Brass edge-came ribs & spider claws shading (matched metallic studio IBL contrast!)
              const nPrismW = normalize(cU.mul(nPrismL.x).add(cV.mul(nPrismL.y)).add(cW.mul(nPrismL.z)));
              const cameRefl = evalStudioSoftboxDome(reflect(rd, nPrismW));
              const cameBrass = mix(pRawUmber.mul(0.36), pYellowOchre.mul(1.06), float(0.75));
              const cameCol = cameBrass
                .mul(float(0.2).add(facetDiff.mul(0.45)).add(cameRefl.mul(1.55)))
                .add(pBoneCalcite.mul(facetSpecKey.mul(0.85).add(facetSpecRim.mul(0.38))))
                .add(pRealgar.mul(0.18).mul(filamentPulse));

              // Crystal-clear optical prism glass glints (ZERO milky opacity -> clear view of bulb inside!)
              const crystalGlints = pBoneCalcite
                .mul(
                  facetFresnel
                    .mul(0.36)
                    .add(facetSpecKey.mul(1.28))
                    .add(facetSpecRim.mul(0.58))
                    .add(rearGlint)
                    .add(bevelCatch.mul(0.46))
                )
                .add(pCeladon.mul(bevelCatch.mul(0.18)));

              const innerCol = isE27Hit.select(e27Col, filEmit);
              const innerAlpha = isE27Hit.select(
                float(0.95),
                clamp(
                  filCore
                    .mul(0.94)
                    .add(stemMask.mul(0.42))
                    .add(bulbRim.mul(0.32))
                    .add(facetFresnel.mul(0.24))
                    .add(bevelCatch.mul(0.2)),
                  0.06,
                  0.95
                )
              );

              const combinedLampCol = mix(innerCol.add(crystalGlints), cameCol, frameMetalMask);
              const combinedAlpha = max(innerAlpha, frameMetalMask.mul(0.96));

              accum.addAssign(combinedLampCol.mul(throughput));
              throughput.mulAssign(float(1.0).sub(combinedAlpha));

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

      const railMask = float(1.0).sub(smoothstep(float(0.03).sub(aaW), float(0.03).add(aaW), edgeDist));
      const gussetDiag = edgeDist.add(cornerDist);
      const gussetMask = float(1.0)
        .sub(smoothstep(float(0.128).sub(aaW), float(0.128).add(aaW), gussetDiag))
        .mul(float(1.0).sub(smoothstep(float(0.074).sub(aaW), float(0.074).add(aaW), edgeDist)));
      const gussetBevelRim = float(1.0)
        .sub(smoothstep(float(0.001), float(0.007).add(aaW), abs(gussetDiag.sub(0.124))))
        .mul(gussetMask);
      const rivetDist = length(vec3(edgeDist.sub(0.044), cornerDist.sub(0.056), float(0.0)));
      const rivetMask = float(1.0)
        .sub(smoothstep(float(0.0065), float(0.0095).add(aaW), rivetDist))
        .mul(gussetMask);

      // Flush stepped ceiling rosette base plate on the +Y ceiling mirror (`r <= CEILING_ROSETTE_RADIUS`)
      const isCeilingWall = smoothstep(float(-0.5), float(-0.85), hitN.y);
      const rCeil = length(vec3(pWall.x, float(0.0), pWall.z));
      const rosetteMask = float(1.0)
        .sub(
          smoothstep(
            float(CEILING_ROSETTE_RADIUS).sub(aaW),
            float(CEILING_ROSETTE_RADIUS).add(aaW),
            rCeil
          )
        )
        .mul(isCeilingWall);
      const rosetteInnerRing = float(1.0).sub(
        smoothstep(float(0.054).sub(aaW), float(0.054).add(aaW), rCeil)
      );

      const solidMetalMask = max(max(railMask, gussetMask), rosetteMask);

      // Refined warm satin-brass beveled inner catch-edge + delicate celadon glass edge-depth trace
      const stripCenter = float(0.0324);
      const stripHalfW = float(0.0036).add(aaW.mul(0.75));
      const stripDist = abs(edgeDist.sub(stripCenter));
      const stripMask = float(1.0).sub(smoothstep(stripHalfW.mul(0.32), stripHalfW, stripDist));
      const stripOpacity = stripMask.mul(float(1.0).sub(solidMetalMask));

      const frameMask = clamp(solidMetalMask.add(stripOpacity), 0.0, 1.0);

      If(frameMask.greaterThan(0.004), () => {
        const toLight = uCorePos.sub(pWall);
        const lDistSq = max(float(0.14), dot(toLight, toLight));
        const lDir = normalize(toLight);
        const atten = float(1.0).div(float(1.0).add(lDistSq.mul(1.25)));
        const nDotL = clamp(dot(hitN, lDir), 0.0, 1.0);
        const halfV = normalize(lDir.sub(rd));
        const metalSpec = pow(max(float(0.0), dot(hitN, halfV)), 38.0);

        const brassWeight = max(gussetMask, rosetteMask.mul(float(0.95).sub(rosetteInnerRing.mul(0.28))));
        const darkBronzeRail = mix(pRawUmber.mul(0.24), pBoneCalcite.mul(0.065), float(0.28));
        const satinBrassBracket = mix(pRawUmber.mul(0.46), pYellowOchre.mul(0.94), float(0.74));
        const railMetal = mix(darkBronzeRail, satinBrassBracket, brassWeight);

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

      // Delicate celadon soda-lime mirror bevel edge tint along the interior perimeter
      const celadonEdgeBand = float(1.0)
        .sub(smoothstep(float(0.032), float(0.062), edgeDist))
        .mul(float(1.0).sub(frameMask));
      const coreToWall = uCorePos.sub(pWall);
      const wallCoreAtten = float(1.0).div(float(1.0).add(dot(coreToWall, coreToWall).mul(2.5)));
      const glassDustVeil = mix(pVerdigris, pCeladon, float(0.45))
        .mul(float(0.0115).add(celadonEdgeBand.mul(0.024)))
        .add(pRealgar.mul(0.007).mul(wallCoreAtten))
        .mul(float(1.0).sub(smoothstep(float(18.0), float(39.0), bounceF)).mul(0.72).add(0.28));
      accum.addAssign(glassDustVeil.mul(throughput));

      // Soda-Lime Mirror Optics: Fresnel dielectric reflectivity * Beer-Lambert ferrous-iron absorption
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

    // Smooth hue-preserving filmic highlight shoulder (incandescent white-gold filament echoes)
    const lum = dot(accum, vec3(0.2126, 0.7152, 0.0722));
    const compressed = accum.div(vec3(1.0).add(accum.mul(0.45)));
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
    ceramicInsulator,
    darkCeramic,
    braidedCord,
    ventBaffleGlow,
    edisonFilament,
    filamentAura,
    amberBulbGlass,
    crystalPrismGlass,
    infinityMirror,
    infinityMirrorFront,
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
      ceramicInsulator.dispose();
      darkCeramic.dispose();
      braidedCord.dispose();
      ventBaffleGlow.dispose();
      edisonFilament.dispose();
      filamentAura.dispose();
      amberBulbGlass.dispose();
      crystalPrismGlass.dispose();
      infinityMirror.dispose();
      infinityMirrorFront.dispose();
    },
  };
}
