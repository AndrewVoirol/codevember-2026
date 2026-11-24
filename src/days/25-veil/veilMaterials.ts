import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  atan,
  attribute,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
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
  select,
  sin,
  smoothstep,
  sqrt,
  texture,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { type SlitLampMaterialsContract } from './slitLampGeometry';
import {
  BEAM_HALF_THICK_BOT,
  BEAM_HALF_THICK_TOP,
  CONE_RX_BOT,
  CONE_RX_TOP,
  CONE_RZ_BOT,
  CONE_RZ_TOP,
  CURVED_S_AMP_BOT,
  CURVED_S_AMP_TOP,
  CURVED_S_HALF_W_BOT,
  CURVED_S_HALF_W_TOP,
  PLANAR_HALF_W_BOT,
  PLANAR_HALF_W_TOP,
  VEIL_HEIGHT,
  VEIL_Y_TOP,
} from './veilSurface';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface VeilStudioEnvironment {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface VeilMaterials extends SlitLampMaterialsContract {
  forgedIron: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  vernierBrass: THREE.MeshStandardNodeMaterial;
  knurledBrass: THREE.MeshStandardNodeMaterial;
  bluedSteel: THREE.MeshStandardNodeMaterial;
  honedSlate: THREE.MeshStandardNodeMaterial;
  ceramicInsulator: THREE.MeshStandardNodeMaterial;
  opticalGlass: THREE.MeshStandardNodeMaterial;
  opticalBaffle: THREE.MeshStandardNodeMaterial;
  linearFilament: THREE.MeshBasicNodeMaterial;
  coneFilament: THREE.MeshBasicNodeMaterial;
  veilMaterial: THREE.MeshBasicNodeMaterial;
  grazingMaterial: THREE.MeshBasicNodeMaterial;
  uTime: { value: number };
  uModeWeights: { value: THREE.Vector3 };
  uLampHover: { value: number };
  uTransitionPulse: { value: number };
  dispose: () => void;
}

/**
 * Feathered radial softbox luminance card so PMREM studio reflections glide smoothly
 * across oil-blackened iron chamfers, brushed brass jaws, engraved vernier scales, and diamond knurling.
 */
function createFeatheredSoftboxTexture(r: number, g: number, b: number, size = 96): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = ((y + 0.5) / size) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = ((x + 0.5) / size) * 2 - 1;
      const d = Math.hypot(nx, ny);
      const falloff =
        Math.exp(-d * d * 1.85) * Math.max(0, 1 - Math.pow(Math.min(1, d), 4));
      const idx = (y * size + x) * 4;
      data[idx] = Math.min(255, Math.round(r * falloff * 255));
      data[idx + 1] = Math.min(255, Math.round(g * falloff * 255));
      data[idx + 2] = Math.min(255, Math.round(b * falloff * 255));
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
 * Generates an enveloping studio cyclorama dome + 7 feathered softbox scrim PMREM environment
 * cubemap calibrated so oil-blackened gunmetal iron, brushed & engraved brass, honed dark slate,
 * glazed steatite porcelain, blued steel, and optical quartz glass exhibit unbroken studio
 * reflections from 360-degree orbits and millimeter-range macro zoom-ins (zero dead-black voids).
 */
export function createVeilStudioEnvironment(
  renderer: THREE.WebGPURenderer
): VeilStudioEnvironment {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x1e2329);

  const disposables: Array<{ dispose: () => void }> = [];

  // Continuous graded gallery cyclorama enclosure (prevents dark-void collapse between scrims)
  const domeGeo = new THREE.SphereGeometry(18, 28, 18);
  const posAttr = domeGeo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(posAttr.count * 3);
  const cFloor = new THREE.Color(0x282522);
  const cHorizon = new THREE.Color(0x383e45);
  const cZenith = new THREE.Color(0x4d545c);
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

  // Overhead-left warm calcite key softbox (carves iron cooling fins, slate bevels & brass chamfers)
  addSoftbox(12.0, 9.0, [-5.0, 5.8, 4.4], 1.0, 0.96, 0.9, 1.9);
  // Right tall studio strip scrim (lights right knurled focus drum, helical lead screw & vernier ticks)
  addSoftbox(8.5, 12.0, [6.4, 2.6, 3.4], 0.99, 0.95, 0.89, 1.68);
  // Front-left low warm fill scrim (lifts front gunmetal plenum & split brass guide rails cleanly)
  addSoftbox(10.5, 6.2, [-4.2, -1.2, 6.4], 0.95, 0.91, 0.85, 0.92);
  // Rear-top cool slate grazing rim scrim (separates dark cooling fins & bail rings off #212529)
  addSoftbox(13.5, 5.6, [-1.0, 5.4, -5.8], 0.88, 0.94, 1.0, 1.62);
  // Rear-right celadon-slate wrap scrim (keeps rear orbits rich & sculpted)
  addSoftbox(9.5, 7.2, [5.2, 1.4, -5.6], 0.88, 0.95, 0.98, 1.16);
  // Zenith diffused gallery canopy (gleams across the top-down brass vernier collar & turret cap)
  addSoftbox(10.5, 10.5, [0.0, 7.6, 0.4], 0.97, 0.95, 0.91, 1.08);
  // Warm Realgar-Amber / Bone-Calcite lower bounce strip simulating upward glow from the light veil
  addSoftbox(12.0, 5.0, [0.0, -6.0, 0.6], 1.0, 0.79, 0.54, 1.28);

  const renderTarget = pmrem.fromScene(envScene, 0.062);
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
 * Scale-normalized 4-tap tetrahedral anisotropic 3D FBM micro-normal perturbation returned
 * in OBJECT space so concentric lathe grooves or bevel details can be composed cleanly
 * before transforming into view space.
 */
const perturbNormalAnisoObject = Fn(
  ([baseNorm_in, pos_in, scaleVec_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in).mul(vec3(scaleVec_in));
    const eps = float(0.14);

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
    const anisoWeight = vec3(scaleVec_in).div(maxScale);
    const grad = k1
      .mul(f1)
      .add(k2.mul(f2))
      .add(k3.mul(f3))
      .add(k4.mul(f4))
      .mul(1.785)
      .mul(anisoWeight);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

/**
 * Detects 45-deg chamfered bevel arrises & fillets in object space:
 * returns ~0 on flat cardinal faces and ~1 along machined bevel edges.
 */
const computeBevelEdgeMask = Fn(([nObj_in]: [TslNode]) => {
  const n = normalize(vec3(nObj_in));
  const maxAxis = max(abs(n.x), max(abs(n.y), abs(n.z)));
  return smoothstep(float(0.972), float(0.75), maxAxis);
});

/**
 * Shared GPU TSL parametric veil surface evaluator matching `evalVeilSurface` in `veilSurface.ts`.
 */
function buildVeilParametricNodes(uNode: TslNode, vNode: TslNode, uModeWeights: TslNode) {
  const s = uNode.sub(0.5);
  const theta = s.mul(Math.PI * 2);
  const sinT = sin(theta);
  const cosT = cos(theta);
  const y = float(VEIL_Y_TOP).sub(vNode.mul(VEIL_HEIGHT));

  const hwP = mix(float(PLANAR_HALF_W_TOP), float(PLANAR_HALF_W_BOT), vNode);
  const dHwP = float(PLANAR_HALF_W_BOT - PLANAR_HALF_W_TOP);
  const xP = s.mul(2.0).mul(hwP);
  const zP = float(0.0);

  const hwS = mix(float(CURVED_S_HALF_W_TOP), float(CURVED_S_HALF_W_BOT), vNode);
  const dHwS = float(CURVED_S_HALF_W_BOT - CURVED_S_HALF_W_TOP);
  const ampS = mix(float(CURVED_S_AMP_TOP), float(CURVED_S_AMP_BOT), vNode);
  const dAmpS = float(CURVED_S_AMP_BOT - CURVED_S_AMP_TOP);
  const xS = s.mul(2.0).mul(hwS);
  const zS = ampS.mul(sinT);

  const rxC = mix(float(CONE_RX_TOP), float(CONE_RX_BOT), vNode);
  const rzC = mix(float(CONE_RZ_TOP), float(CONE_RZ_BOT), vNode);
  const dRxC = float(CONE_RX_BOT - CONE_RX_TOP);
  const dRzC = float(CONE_RZ_BOT - CONE_RZ_TOP);
  const xC = rxC.mul(sinT);
  const zC = rzC.mul(cosT);

  const w0 = uModeWeights.x;
  const w1 = uModeWeights.y;
  const w2 = uModeWeights.z;

  const pos = vec3(
    w0.mul(xP).add(w1.mul(xS)).add(w2.mul(xC)),
    y,
    w0.mul(zP).add(w1.mul(zS)).add(w2.mul(zC))
  );

  const twoPi = float(Math.PI * 2);
  const dxdu = w0.mul(hwP.mul(2.0)).add(w1.mul(hwS.mul(2.0))).add(w2.mul(rxC.mul(twoPi).mul(cosT)));
  const dzdu = w1.mul(ampS.mul(twoPi).mul(cosT)).add(w2.mul(rzC.negate().mul(twoPi).mul(sinT)));

  const dxdv = w0.mul(s.mul(2.0).mul(dHwP)).add(w1.mul(s.mul(2.0).mul(dHwS))).add(w2.mul(dRxC.mul(sinT)));
  const dydv = float(-VEIL_HEIGHT);
  const dzdv = w1.mul(dAmpS.mul(sinT)).add(w2.mul(dRzC.mul(cosT)));

  const tanU = vec3(dxdu, float(0.0), dzdu);
  const tanV = vec3(dxdv, dydv, dzdv);
  const surfNormal = normalize(cross(tanV, tanU).add(vec3(0.0, 0.0, 1e-5)));

  return { pos, surfNormal };
}

/**
 * Builds all TSL materials for Day 25 - Veil strictly from `MINERAL_PALETTE`:
 *   1. Oil-blackened cast & milled gunmetal iron (micro-peened cast walls, draw-filed rails, burnished chamfers)
 *   2. Machined turned & satin-brushed antique brass (Kubelka-Munk ochre glaze, crevice patina, edge burnish)
 *   3. Engraved calibrated optical vernier scale brass (72-tick black-inked dial + lathe turning grooves)
 *   4. True 4-sided raised-pyramid diamond-knurled brass (darkened valleys, gleaming crests, turned cap)
 *   5. Honed dark veined slate / basalt thermal cheek inlays (calcite hairline veins, sedimentary tooth, rabbet AO)
 *   6. Glazed Bone-Calcite steatite ceramic thermal insulators (ribbed porcelain sheen & waist crease shading)
 *   7. Tempered Prussian-blued steel (Kubelka-Munk azurite temper, draw-filed brushing, bright edge rub-through)
 *   8. Optical borosilicate condenser lens glass (Fresnel rim reflection + warm celadon-amber transmission)
 *   9. Matte-black optical anti-reflection light-trap throat baffle
 *  10. Continuous incandescent slit & annular cone filaments + 5-stratum volumetric Mie light veil
 */
export function createVeilMaterials(fluidTex: THREE.DataTexture): VeilMaterials {
  const uTime = uniform(0);
  const uModeWeights = uniform(new THREE.Vector3(1, 0, 0));
  const uLampHover = uniform(0);
  const uTransitionPulse = uniform(0);

  // Physical mineral pigment anchors (strictly zero digital RGB neons)
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pPrussian = createPigmentUniforms('azuritePrussian');

  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const cKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const cRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const cOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const cSienna = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const cLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);
  const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
  const cUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const cAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Oil-Blackened Cast & Milled Gunmetal Iron (Plenum, Central Chest, Fins, Turret)
  //    Replaces coarse Perlin clouds with authentic micro-peened sand-cast texture,
  //    directional milled steel brushing, crisp hand-rubbed 45-deg burnished chamfers,
  //    and subtle upward warm Realgar-Amber veil bounce strictly on lower edges!
  // ---------------------------------------------------------------------------
  const forgedIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.83,
    roughness: 0.35,
    envMapIntensity: 1.32,
  });
  {
    const nObj = normalize(normalLocal);
    const nW = normalize(normalWorld);
    const pW = positionWorld;
    const bevelMask = computeBevelEdgeMask(nObj);
    const topFaceMask = smoothstep(float(0.68), float(0.95), abs(nObj.y));

    // Fine sand-cast micro-peening + directional longitudinal milling tool marks (zero big marble blobs!)
    const castPeen = mineralToothFbm(pW.mul(float(145.0)), float(1.15));
    const microPore = mineralToothFbm(pW.mul(float(320.0)).add(vec3(3.7, 1.9, 5.3)), float(1.1));
    const millBrush = mineralToothFbm(pW.mul(vec3(14.0, 220.0, 180.0)), float(1.15));
    const oilSheen = mineralToothFbm(pW.mul(float(11.0)).add(vec3(1.4, 6.2, 2.8)), float(1.1));

    // Subtle concentric lathe turning sheen strictly on cylindrical turret caps (`rXZ` around 0)
    const rTurret = length(vec2(positionLocal.x, positionLocal.z));
    const latheRing = sin(rTurret.mul(410.0)).mul(0.5).add(0.5);

    // Tight calibrated oil-blackened gunmetal reflectance range (charcoal steel, never mottled granite!)
    const ironDeep = mix(cUmber.mul(0.18), cAzurite.mul(0.24), float(0.58));
    const ironMid = mix(cUmber.mul(0.28), cAzurite.mul(0.36), float(0.54)).add(cCalcite.mul(0.025));
    const ironMilled = mix(ironMid, cKaolin.mul(0.22), float(0.32));
    const ironBurnished = mix(cKaolin.mul(0.28), cCalcite.mul(0.30), float(0.52));
    const oilCrevice = mix(cUmber.mul(0.11), cSienna.mul(0.09), float(0.35));

    const surfaceGrain = mix(
      castPeen.mul(0.58).add(microPore.mul(0.42)),
      millBrush.mul(0.65).add(latheRing.mul(0.35)),
      topFaceMask.mul(0.65)
    );

    const baseIron = mix(
      ironDeep,
      mix(ironMid, ironMilled, topFaceMask.mul(0.45)),
      smoothstep(float(0.18), float(0.84), surfaceGrain).mul(0.58).add(0.22)
    );
    const withOil = mix(baseIron, oilCrevice, smoothstep(float(0.64), float(0.92), oilSheen).mul(0.24));
    const withBevelWear = mix(
      withOil,
      ironBurnished,
      bevelMask.mul(float(0.46).add(microPore.mul(0.14)))
    );

    // Subtle warm Realgar-Amber bounce strictly on downward-facing underside rails near slit exit
    const undersideBounceMask = smoothstep(float(-0.12), float(-0.85), nW.y)
      .mul(smoothstep(float(1.36), float(1.17), pW.y));
    const veilBounce = mix(cRealgar, cOchre, float(0.38))
      .mul(undersideBounceMask)
      .mul(float(0.055).add(uTransitionPulse.mul(0.04)));

    const hoverLift = cOchre.mul(uLampHover.mul(0.038));
    forgedIron.colorNode = withBevelWear.add(veilBounce).add(hoverLift);

    // Tactile micro-peened cast walls transitioning smoothly to finer milled/burnished bevels
    const castNorm = perturbNormalAnisoObject(nObj, pW, vec3(165.0, 165.0, 165.0), float(0.022));
    const millNorm = perturbNormalAnisoObject(nObj, pW, vec3(16.0, 240.0, 185.0), float(0.015));
    const blendedN = normalize(mix(castNorm, millNorm, clamp(topFaceMask.mul(0.75).add(bevelMask.mul(0.5)), 0.0, 1.0)));
    forgedIron.normalNode = transformNormalToView(blendedN);

    forgedIron.roughnessNode = clamp(
      float(0.43)
        .add(castPeen.sub(0.5).mul(0.10))
        .add(oilSheen.sub(0.5).mul(0.07))
        .sub(topFaceMask.mul(0.08))
        .sub(bevelMask.mul(0.19)),
      0.18,
      0.58
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Turned & Satin-Brushed Antique Brass (End-Caps, Rails, Jaws, Iris Bezel, Shackles)
  //    Replaces oatmeal FBM blotches with crisp directional satin brushing, Kubelka-Munk
  //    yellow-ochre brass depth, oil-patinated undercuts, sheltered verdigris trace, and bright chamfers!
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.22,
    envMapIntensity: 1.34,
  });
  {
    const nObj = normalize(normalLocal);
    const nW = normalize(normalWorld);
    const pW = positionWorld;
    const bevelMask = computeBevelEdgeMask(nObj);

    // Directional longitudinal brushed satin tool lines along X + concentric radial lathe micro-rings
    const brushLong = mineralToothFbm(pW.mul(vec3(16.0, 260.0, 240.0)), float(1.12));
    const fineSatin = mineralToothFbm(pW.mul(float(185.0)).add(vec3(4.3, 1.7, 5.9)), float(1.1));
    const agePatina = mineralToothFbm(pW.mul(float(14.5)).add(vec3(2.1, 7.4, 3.8)), float(1.15));

    const rLoc = length(vec2(positionLocal.x, positionLocal.z));
    const latheRing = sin(rLoc.mul(480.0)).mul(0.5).add(0.5);
    const endCapFace = smoothstep(float(0.75), float(0.96), abs(nObj.y));

    const brassUmber = mix(cUmber.mul(0.68), cSienna.mul(0.52), float(0.28));
    const brassMid = mix(cOchre.mul(1.06), cCalcite.mul(0.84), float(0.26));
    const brassBright = mix(cOchre.mul(1.24), cCalcite.mul(0.96), float(0.42));
    const kmBrass = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.48), brassMid);

    const grainWave = mix(
      brushLong.mul(0.68).add(fineSatin.mul(0.32)),
      latheRing.mul(0.55).add(fineSatin.mul(0.45)),
      endCapFace.mul(0.55)
    );
    const baseBrass = mix(
      mix(brassUmber, brassMid, float(0.58)),
      mix(brassMid, kmBrass, float(0.22)),
      smoothstep(float(0.16), float(0.84), grainWave).mul(0.56).add(0.26)
    );

    // Subtle oil darkening in sheltered undersides + delicate trace of sheltered verdigris age
    const patinaZone = smoothstep(float(0.62), float(0.92), agePatina).mul(float(1.0).sub(bevelMask));
    const withPatina = mix(baseBrass, brassUmber, patinaZone.mul(0.26));
    const verdigrisHint = smoothstep(float(0.82), float(0.96), agePatina)
      .mul(float(1.0).sub(bevelMask))
      .mul(smoothstep(float(0.2), float(-0.5), nW.y));
    const aged = mix(withPatina, mix(brassUmber, cVerdigris.mul(0.45), float(0.42)), verdigrisHint.mul(0.18));

    // Crisp bright-burnished chamfered edges + warm underside filament glow catch on slit jaws
    const burnished = mix(aged, brassBright, bevelMask.mul(0.48));
    const jawGlowMask = smoothstep(float(1.21), float(1.16), pW.y).mul(
      smoothstep(float(0.18), float(-0.75), nW.y)
    );
    const jawGlow = mix(cRealgar, cCalcite, float(0.32)).mul(jawGlowMask).mul(0.11);
    const hoverWarmth = cRealgar.mul(uLampHover.mul(0.10));

    satinBrass.colorNode = burnished.add(jawGlow).add(hoverWarmth);

    const brushedN = perturbNormalAnisoObject(nObj, pW, vec3(18.0, 270.0, 250.0), float(0.011));
    satinBrass.normalNode = transformNormalToView(brushedN);

    satinBrass.roughnessNode = clamp(
      float(0.21)
        .add(grainWave.sub(0.5).mul(0.09))
        .add(patinaZone.mul(0.11))
        .sub(bevelMask.mul(0.07)),
      0.13,
      0.38
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Calibrated Engraved Optical Vernier Brass
  //    Dedicated shader for TurretBrassVernierCollar (72 ticks), FrontSelectorBezel (36 ticks),
  //    and SideVernierSleeves (24 ticks): crisp black-inked optical graduations on both rim and
  //    flat annular deck + watchmaker concentric lathe turning rings (zero oatmeal clouds!).
  // ---------------------------------------------------------------------------
  const vernierBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.21,
    envMapIntensity: 1.36,
  });
  {
    const p = positionLocal;
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);
    const rXZ = length(vec2(p.x, p.z));
    const safeX = abs(p.x).lessThan(1e-5).select(float(1e-5), p.x);
    const phi = atan(p.z, safeX);

    // Adapt tick density cleanly to piece radius (72 on large turret collar, 36 on front/side dials)
    const isLargeTurret = smoothstep(float(0.11), float(0.14), rXZ);
    const isMediumBezel = smoothstep(float(0.052), float(0.068), rXZ).mul(float(1.0).sub(isLargeTurret));
    const minorFreq = select(
      isLargeTurret.greaterThan(0.5),
      float(72.0),
      select(isMediumBezel.greaterThan(0.5), float(36.0), float(24.0))
    );
    const majorFreq = select(
      isLargeTurret.greaterThan(0.5),
      float(12.0),
      select(isMediumBezel.greaterThan(0.5), float(6.0), float(6.0))
    );
    const medFreq = majorFreq.mul(2.0);

    const minorU = fract(phi.mul(minorFreq.div(Math.PI * 2)).add(72.0));
    const minorDist = abs(minorU.sub(0.5));
    const medU = fract(phi.mul(medFreq.div(Math.PI * 2)).add(24.0));
    const isMed = abs(medU.sub(0.5)).lessThan(0.11);
    const majorU = fract(phi.mul(majorFreq.div(Math.PI * 2)).add(12.0));
    const isMajor = abs(majorU.sub(0.5)).lessThan(0.12);

    // Cylindrical barrel graduations (visible from front/side/3-4 orbits)
    const isBarrelWall = float(1.0).sub(smoothstep(float(0.45), float(0.75), abs(nObj.y)));
    const tickHalfLen = isMajor.select(float(0.0115), isMed.select(float(0.0085), float(0.0058)));
    const tickHalfW = isMajor.select(float(0.125), isMed.select(float(0.095), float(0.072)));
    const inBarrelBand = smoothstep(float(-0.013), float(-0.0105), p.y).mul(
      float(1.0).sub(smoothstep(tickHalfLen.sub(0.002), tickHalfLen, p.y))
    );
    const barrelTicks = float(1.0)
      .sub(smoothstep(tickHalfW.mul(0.55), tickHalfW, minorDist))
      .mul(inBarrelBand)
      .mul(isBarrelWall);

    // Dual engraved circumferential datum witness lines flanking the cylindrical scale
    const datumTop = float(1.0)
      .sub(smoothstep(float(0.00055), float(0.00125), abs(p.y.sub(0.012))))
      .mul(isBarrelWall);
    const datumBot = float(1.0)
      .sub(smoothstep(float(0.00055), float(0.00125), abs(p.y.add(0.012))))
      .mul(isBarrelWall);

    // Flat top / front annular deck graduations (razor sharp in top-down turret & front bezel macros!)
    const isTopDeck = smoothstep(float(0.76), float(0.96), nObj.y);
    // Turret collar top deck lives in r in [0.118, 0.165]; front bezel deck in r in [0.050, 0.073]
    const deckOuterR = select(isLargeTurret.greaterThan(0.5), float(0.163), float(0.072));
    const deckInnerR = select(isLargeTurret.greaterThan(0.5), float(0.122), float(0.052));
    const tickRadialDepth = isMajor.select(
      deckOuterR.sub(deckInnerR).mul(0.72),
      isMed.select(deckOuterR.sub(deckInnerR).mul(0.50), deckOuterR.sub(deckInnerR).mul(0.32))
    );
    const tickStartR = deckOuterR.sub(tickRadialDepth);
    const inDeckRadial = smoothstep(tickStartR.sub(0.0012), tickStartR.add(0.0012), rXZ).mul(
      float(1.0).sub(smoothstep(deckOuterR.sub(0.0012), deckOuterR.add(0.0012), rXZ))
    );
    const deckTicks = float(1.0)
      .sub(smoothstep(tickHalfW.mul(0.55), tickHalfW, minorDist))
      .mul(inDeckRadial)
      .mul(isTopDeck);

    // Concentric engraved datum circle on the top deck
    const deckDatumR = deckOuterR.sub(0.0015);
    const deckCircle = float(1.0)
      .sub(smoothstep(float(0.00065), float(0.00145), abs(rXZ.sub(deckDatumR))))
      .mul(isTopDeck);
    const deckInnerCircle = float(1.0)
      .sub(smoothstep(float(0.00065), float(0.00145), abs(rXZ.sub(deckInnerR.add(0.003)))))
      .mul(isTopDeck);

    const engraved = clamp(
      max(max(barrelTicks, max(datumTop, datumBot)), max(deckTicks, max(deckCircle, deckInnerCircle))),
      0.0,
      1.0
    );

    // Watchmaker concentric lathe turning grooves (both radial on deck & axial on barrel)
    const latheRadial = sin(rXZ.mul(620.0)).mul(0.5).add(0.5);
    const latheAxial = sin(p.y.mul(520.0)).mul(0.5).add(0.5);
    const latheWave = mix(latheAxial, latheRadial, isTopDeck);
    const fineTurn = mineralToothFbm(vec3(p.x.mul(45.0), p.y.mul(280.0), p.z.mul(45.0)), float(1.1));

    const brassDark = mix(cUmber.mul(0.76), cOchre.mul(0.84), float(0.62));
    const brassMid = mix(cOchre.mul(1.08), cCalcite.mul(0.84), float(0.28));
    const brassHi = mix(cOchre.mul(1.24), cCalcite.mul(0.95), float(0.42));
    const kmDial = kubelkaMunkGlaze(pOchre.K, pOchre.S, float(0.44), brassMid);

    const dialMetal = mix(
      brassDark,
      mix(brassMid, kmDial, float(0.20)),
      latheWave.mul(0.42).add(fineTurn.mul(0.32)).add(0.26)
    );
    const dialBeveled = mix(dialMetal, brassHi, bevelMask.mul(0.44));
    const engravingInk = mix(cUmber.mul(0.11), cAzurite.mul(0.08), float(0.35));

    vernierBrass.colorNode = mix(dialBeveled, engravingInk, engraved.mul(0.92)).add(
      cRealgar.mul(uLampHover.mul(0.10))
    );

    // Physical lathe turning micro-normal relief on the annular dial face + barrel
    const safeR = max(rXZ, float(0.002));
    const radialDir = vec3(p.x.div(safeR), float(0.0), p.z.div(safeR));
    const radialTan = radialDir.sub(nObj.mul(dot(radialDir, nObj)));
    const latheSlope = cos(rXZ.mul(620.0)).mul(0.018).mul(isTopDeck);
    const baseTurnN = perturbNormalAnisoObject(nObj, p, vec3(90.0, 260.0, 90.0), float(0.008));
    vernierBrass.normalNode = transformNormalToView(normalize(baseTurnN.add(radialTan.mul(latheSlope))));

    vernierBrass.roughnessNode = clamp(
      mix(float(0.20), float(0.74), engraved)
        .add(latheWave.sub(0.5).mul(0.06))
        .sub(bevelMask.mul(0.05)),
      0.14,
      0.78
    );
    vernierBrass.metalnessNode = mix(float(0.92), float(0.22), engraved);
  }

  // ---------------------------------------------------------------------------
  // 4. 4-Sided Raised-Pyramid Diamond-Knurled Satin Brass Focus & Mode Drums
  //    Deep raw-umber oxidized knurl valleys + bright burnished diamond pyramid crowns +
  //    concentric lathe turning rings across the recessed concave end dimple!
  // ---------------------------------------------------------------------------
  const knurledBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.21,
    envMapIntensity: 1.36,
  });
  {
    const aKnurlValley = attribute('aKnurlValley', 'float');
    const nObj = normalize(normalLocal);
    const p = positionLocal;
    const fineGrain = mineralToothFbm(p.mul(float(120.0)), float(1.1));
    const rCap = length(vec2(p.x, p.z));
    const capLathe = sin(rCap.mul(580.0)).mul(0.5).add(0.5);
    const isCapFace = smoothstep(float(0.72), float(0.95), abs(nObj.y));

    const grooveFactor = clamp(aKnurlValley, 0.0, 1.0);
    const valleyDark = mix(cUmber, cSienna, float(0.22)).mul(0.42);
    const crestBright = mix(cOchre.mul(1.22), cCalcite.mul(0.92), float(0.36));
    const capSatin = mix(
      mix(cUmber.mul(0.84), cOchre.mul(0.96), float(0.70)),
      crestBright,
      capLathe.mul(0.45).add(fineGrain.mul(0.25)).add(0.25)
    );

    const knurlCol = mix(
      valleyDark,
      crestBright,
      smoothstep(float(0.05), float(0.94), grooveFactor)
    );
    const hoverLift = cOchre.mul(uLampHover.mul(0.16));

    knurledBrass.colorNode = mix(knurlCol, capSatin, isCapFace.mul(0.82)).add(hoverLift);

    // Subtle concentric lathe turning grooves strictly on the smooth end-cap dimple
    const safeR = max(rCap, float(0.002));
    const radialDir = vec3(p.x.div(safeR), float(0.0), p.z.div(safeR));
    const radialTan = radialDir.sub(nObj.mul(dot(radialDir, nObj)));
    const capSlope = cos(rCap.mul(580.0)).mul(0.015).mul(isCapFace);
    knurledBrass.normalNode = transformNormalToView(normalize(nObj.add(radialTan.mul(capSlope))));

    knurledBrass.roughnessNode = clamp(
      mix(float(0.54), float(0.17), grooveFactor)
        .add(fineGrain.sub(0.5).mul(0.04))
        .add(capLathe.sub(0.5).mul(0.04).mul(isCapFace)),
      0.14,
      0.58
    );
    knurledBrass.metalnessNode = mix(float(0.66), float(0.94), grooveFactor);
  }

  // ---------------------------------------------------------------------------
  // 5. Honed Dark Veined Slate / Basalt Thermal Cheek Inlays
  //    Authentic matte honed sedimentary mineral stone with subtle slate foliation,
  //    micro-crystalline quartz tooth, delicate domain-warped Bone-Calcite hairline veins,
  //    and darkened perimeter rabbet contact-AO where seated inside the brass frame!
  // ---------------------------------------------------------------------------
  const honedSlate = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.64,
    envMapIntensity: 0.92,
  });
  {
    const pLoc = positionLocal;
    const pW = positionWorld;
    const nObj = normalize(normalLocal);
    const bevelMask = computeBevelEdgeMask(nObj);

    // Organic sedimentary bedding warp + fine metamorphic slate cleavage lamellae
    const warpA = mineralToothFbm(pW.mul(float(5.2)), float(1.3)).sub(0.5);
    const warpB = mineralToothFbm(pW.mul(float(11.5)).add(vec3(3.1, 7.4, 2.2)), float(1.2)).sub(0.5);
    const slateStrata = mineralToothFbm(
      vec3(
        pW.x.mul(7.5).add(warpA.mul(1.1)),
        pW.y.mul(44.0).add(warpA.mul(2.4)).add(warpB.mul(0.8)),
        pW.z.mul(14.0)
      ),
      float(1.4)
    );
    const mineralTooth = mineralToothFbm(pW.mul(float(118.0)), float(1.5));
    const microPits = mineralToothFbm(pW.mul(float(235.0)).add(vec3(6.2, 2.7, 4.9)), float(1.4));

    // Delicate razor-thin Bone-Calcite / Warm-Kaolin quartz mineral hairline veins (authentic Portoro/Belgian slate)
    const veinCoord1 = pW.x.mul(7.2).add(pW.y.mul(18.5)).add(warpA.mul(2.8)).add(warpB.mul(0.95));
    const veinField1 = valueNoise3DTsl(vec3(veinCoord1, pW.y.mul(7.4), pW.z.mul(5.5)));
    const veinRidge1 = float(1.0).sub(abs(veinField1.mul(2.0).sub(1.0)));
    const calciteVein1 = smoothstep(float(0.954), float(0.993), veinRidge1);

    const veinCoord2 = pW.x.mul(-11.4).add(pW.y.mul(22.0)).add(warpB.mul(2.4));
    const veinField2 = valueNoise3DTsl(vec3(veinCoord2.add(4.7), pW.x.mul(6.8), pW.z.mul(6.8).add(2.3)));
    const veinRidge2 = float(1.0).sub(abs(veinField2.mul(2.0).sub(1.0)));
    const calciteVein2 = smoothstep(float(0.962), float(0.995), veinRidge2).mul(0.55);

    // Subtle secondary faint mineral ghosting adjacent to primary hairline veins
    const veinGhost = smoothstep(float(0.905), float(0.965), veinRidge1).mul(0.18);
    const mineralVein = clamp(max(max(calciteVein1, calciteVein2), veinGhost), 0.0, 1.0);

    // Fine crystalline mica/quartz mineral flecks catching crisp raking studio light
    const mineralSpeck = smoothstep(float(0.78), float(0.92), microPits);
    const slatePit = float(1.0).sub(smoothstep(float(0.14), float(0.32), mineralTooth));

    // Authentic deep honed slate / basalt mineral palette (lifted gently off iron so it reads distinctly as stone!)
    const slateDeep = mix(cUmber.mul(0.24), cAzurite.mul(0.28), float(0.42));
    const slateMid = mix(cUmber.mul(0.34), cCeladon.mul(0.26), float(0.32)).add(cKaolin.mul(0.032));
    const slateLight = mix(slateMid, cKaolin.mul(0.32), float(0.34));
    const kmSlate = kubelkaMunkGlaze(pUmber.K, pUmber.S, float(0.18).add(slateStrata.mul(0.16)), slateMid);

    const baseStone = mix(
      slateDeep,
      mix(slateMid, kmSlate, float(0.38)),
      smoothstep(float(0.22), float(0.82), slateStrata.mul(0.62).add(mineralTooth.mul(0.38)))
    );
    const withPits = mix(baseStone, slateDeep.mul(0.65), slatePit.mul(0.34));
    const withVeins = mix(withPits, mix(cCalcite.mul(0.58), cKaolin.mul(0.52), float(0.45)), mineralVein.mul(0.62));
    const withSpecks = mix(withVeins, cCalcite.mul(0.46), mineralSpeck.mul(0.16));
    const withChamfer = mix(withSpecks, slateLight, bevelMask.mul(0.36));

    // Perimeter rabbet contact-AO where the honed slate seats inside its brass cheek frame (`hx=0.264, hy=0.044`)
    const edgeAoX = smoothstep(float(0.232), float(0.264), abs(pLoc.x));
    const edgeAoY = smoothstep(float(0.028), float(0.044), abs(pLoc.y));
    const rabbetAo = clamp(max(edgeAoX, edgeAoY), 0.0, 1.0).mul(float(1.0).sub(bevelMask.mul(0.6)));

    honedSlate.colorNode = withChamfer.mul(float(1.0).sub(rabbetAo.mul(0.26)));

    // Dual-scale honed stone normal relief: sedimentary cleavage planes + crystalline micro-tooth!
    const normCoarse = perturbNormalAnisoObject(nObj, pW, vec3(18.0, 62.0, 24.0), float(0.042));
    const normFine = perturbNormalAnisoObject(normCoarse, pW, vec3(145.0, 145.0, 145.0), float(0.036));
    honedSlate.normalNode = transformNormalToView(normFine);

    honedSlate.roughnessNode = clamp(
      float(0.62)
        .add(mineralTooth.sub(0.5).mul(0.16))
        .add(slatePit.mul(0.20))
        .sub(mineralVein.mul(0.18))
        .sub(mineralSpeck.mul(0.24))
        .sub(bevelMask.mul(0.08)),
      0.34,
      0.88
    );
  }

  // ---------------------------------------------------------------------------
  // 6. Glazed Bone-Calcite Steatite Porcelain Thermal Insulators
  //    Creamy kiln-glazed laboratory ceramic with subtle waist groove shadow occlusion
  //    and soft glossy porcelain speculars contrasting against iron and slate!
  // ---------------------------------------------------------------------------
  const ceramicInsulator = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.19,
    envMapIntensity: 1.18,
  });
  {
    const pLoc = positionLocal;
    const pW = positionWorld;
    const nObj = normalize(normalLocal);
    const glazeMicro = mineralToothFbm(pW.mul(float(135.0)), float(1.2));
    const kilnWarmth = mineralToothFbm(pW.mul(float(28.0)).add(vec3(2.4, 5.1, 1.3)), float(1.15));

    // Subtle dark tea-stain crevice shading inside the ribbed porcelain waist groove (`abs(pLoc.y) < 0.0045`)
    const waistRecess = float(1.0).sub(smoothstep(float(0.0015), float(0.0068), abs(pLoc.y)));
    const porcelainLight = mix(cKaolin, cCalcite, float(0.52)).mul(0.92);
    const porcelainWarm = mix(cKaolin, cOchre, float(0.12)).mul(0.84);
    const porcelainCrevice = mix(cUmber.mul(0.45), cKaolin.mul(0.52), float(0.42));

    const bodyCol = mix(
      porcelainWarm,
      porcelainLight,
      smoothstep(float(0.24), float(0.80), glazeMicro.mul(0.6).add(kilnWarmth.mul(0.4)))
    );
    ceramicInsulator.colorNode = mix(bodyCol, porcelainCrevice, waistRecess.mul(0.28));

    const glazeN = perturbNormalAnisoObject(nObj, pW, vec3(95.0, 95.0, 95.0), float(0.006));
    ceramicInsulator.normalNode = transformNormalToView(glazeN);

    ceramicInsulator.roughnessNode = clamp(
      float(0.18).add(glazeMicro.sub(0.5).mul(0.06)).add(waistRecess.mul(0.12)),
      0.13,
      0.34
    );
  }

  // ---------------------------------------------------------------------------
  // 7. Tempered Prussian-Blued Steel (Helical Lead Screws, Slotted Bolts, Tie-Rods, Spider Vanes)
  //    Kubelka-Munk Prussian-blue temper sheen + bright steel rub-through on chamfers & thread crests!
  // ---------------------------------------------------------------------------
  const bluedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.24,
    envMapIntensity: 1.32,
  });
  {
    const nObj = normalize(normalLocal);
    const pLoc = positionLocal;
    const pW = positionWorld;
    const bevelMask = computeBevelEdgeMask(nObj);

    const drawGrain = mineralToothFbm(pW.mul(vec3(36.0, 230.0, 230.0)), float(1.12));
    const temperWave = mineralToothFbm(pW.mul(float(18.0)).add(vec3(4.2, 8.1, 2.3)), float(1.1));
    const rCyl = length(vec2(pLoc.x, pLoc.z));
    const threadCrest = smoothstep(float(0.0158), float(0.0190), rCyl);

    const blueDeep = mix(cAzurite.mul(0.42), cLapis.mul(0.38), float(0.42));
    const blueMid = mix(cAzurite.mul(0.86), cCalcite.mul(0.22), float(0.26));
    const temperStraw = mix(cLapis.mul(0.58), cSienna.mul(0.28), float(0.30));
    const brightSteel = mix(cCalcite.mul(0.62), cKaolin.mul(0.68), float(0.44));

    const kmBlue = kubelkaMunkGlaze(
      pPrussian.K,
      pPrussian.S,
      float(0.42).add(temperWave.mul(0.22)),
      blueMid
    );
    const baseBlue = mix(
      blueDeep,
      mix(blueMid, kmBlue, float(0.36)),
      smoothstep(float(0.16), float(0.84), drawGrain).mul(0.54).add(0.28)
    );
    const withTemper = mix(baseBlue, temperStraw, smoothstep(float(0.56), float(0.88), temperWave).mul(0.26));
    const wearMask = clamp(bevelMask.mul(0.42).add(threadCrest.mul(0.34)), 0.0, 0.58);

    bluedSteel.colorNode = mix(withTemper, brightSteel, wearMask);
    bluedSteel.normalNode = transformNormalToView(
      perturbNormalAnisoObject(nObj, pW, vec3(42.0, 240.0, 240.0), float(0.012))
    );
    bluedSteel.roughnessNode = clamp(
      float(0.24)
        .add(drawGrain.sub(0.5).mul(0.09))
        .sub(wearMask.mul(0.08)),
      0.14,
      0.38
    );
  }

  // ---------------------------------------------------------------------------
  // 8. Optical Borosilicate Cylindrical Condenser Lens (Underside Slit Throat)
  //    Crisp dielectric Fresnel rim reflection + warm Realgar-Amber & soda-lime Celadon clarity!
  // ---------------------------------------------------------------------------
  const opticalGlass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.08,
    envMapIntensity: 1.48,
    transparent: true,
    depthWrite: false,
  });
  {
    const nW = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const ndotv = clamp(abs(dot(nW, viewDir)), 0.0, 1.0);
    const fresnel = pow(max(float(0.0001), float(1.0).sub(ndotv)), float(3.2));
    const openVis = clamp(float(1.0).sub(uModeWeights.z), 0.0, 1.0);
    const endFade = smoothstep(float(0.86), float(0.68), abs(positionWorld.x));

    const sodaEdge = mix(cCeladon.mul(0.85), cCalcite, float(0.48));
    const warmCore = mix(cRealgar.mul(1.25), cCalcite.mul(1.35), float(0.56)).mul(
      float(0.72).add(endFade.mul(0.28))
    );
    opticalGlass.colorNode = mix(warmCore, sodaEdge.mul(1.2), fresnel.mul(0.62));
    opticalGlass.emissiveNode = mix(cRealgar, cCalcite, float(0.46))
      .mul(float(0.42).add(uTransitionPulse.mul(0.25)))
      .mul(openVis)
      .mul(float(0.45).add(endFade.mul(0.55)));
    opticalGlass.opacityNode = openVis.mul(clamp(float(0.32).add(fresnel.mul(0.56)), 0.0, 0.88));
  }

  // ---------------------------------------------------------------------------
  // 9. Matte-Black Optical Anti-Reflection Light-Trap Cavity Lining
  // ---------------------------------------------------------------------------
  const opticalBaffle = new THREE.MeshStandardNodeMaterial({
    metalness: 0.12,
    roughness: 0.86,
    envMapIntensity: 0.35,
  });
  {
    const tooth = mineralToothFbm(positionWorld.mul(float(180.0)), float(1.2));
    const baffleBase = mix(cUmber.mul(0.06), cAzurite.mul(0.07), float(0.5)).mul(
      float(0.85).add(tooth.mul(0.30))
    );
    opticalBaffle.colorNode = baffleBase;
    opticalBaffle.roughnessNode = clamp(float(0.84).add(tooth.sub(0.5).mul(0.10)), 0.76, 0.94);
  }

  // ---------------------------------------------------------------------------
  // 10. Recessed Glowing Incandescent Optical Slit Filaments (Continuous Linear & Annular)
  // ---------------------------------------------------------------------------
  const linearFilament = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  });
  {
    const openVis = clamp(float(1.0).sub(uModeWeights.z), 0.0, 1.0);
    const pulse = float(1.0)
      .add(sin(uTime.mul(3.4)).mul(0.03))
      .add(uTransitionPulse.mul(0.28));
    // World-space end taper -> seamless bright continuous core across world x = 0!
    const endFade = smoothstep(float(0.86), float(0.64), abs(positionWorld.x));
    const filColor = mix(cRealgar.mul(1.85), cCalcite.mul(2.25), endFade.mul(0.55).add(0.28)).mul(
      pulse
    );
    linearFilament.colorNode = filColor;
    linearFilament.opacityNode = openVis.mul(endFade.mul(0.82).add(0.18));
  }

  const coneFilament = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  });
  {
    const coneVis = clamp(uModeWeights.z, 0.0, 1.0);
    const pulse = float(1.0)
      .add(sin(uTime.mul(3.4).add(0.7)).mul(0.03))
      .add(uTransitionPulse.mul(0.32));
    coneFilament.colorNode = mix(cRealgar.mul(1.8), cCalcite.mul(2.15), float(0.58)).mul(pulse);
    coneFilament.opacityNode = coneVis;
  }

  // ---------------------------------------------------------------------------
  // 11. Volumetric 5-Stratum Light-and-Haze Veil Material (TSL)
  // ---------------------------------------------------------------------------
  const veilMaterial = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });

  {
    const vUvIn = uv();
    const aStratum = attribute('aStratum', 'float');
    const { pos: surfPos, surfNormal } = buildVeilParametricNodes(vUvIn.x, vUvIn.y, uModeWeights);

    const vertFluid = texture(fluidTex, vUvIn).level(float(0.0));
    const halfThick = mix(float(BEAM_HALF_THICK_TOP), float(BEAM_HALF_THICK_BOT), vUvIn.y);
    const aeroFlutter = clamp(vertFluid.a.mul(0.0011), -0.026, 0.026).mul(
      sin(vUvIn.y.mul(Math.PI))
    );
    const displacedPos = surfPos.add(surfNormal.mul(aStratum.mul(halfThick).add(aeroFlutter)));

    veilMaterial.positionNode = displacedPos;

    const vCurUv = vUvIn.toVarying('vVeilUv');
    const vStratum = aStratum.toVarying('vVeilStratum');
    const vWorldPos = displacedPos.toVarying('vVeilWorldPos');
    const vSurfNormal = surfNormal.toVarying('vVeilSurfNormal');

    const evalVeilFragment = Fn(() => {
      const u = vCurUv.x;
      const v = vCurUv.y;

      const fNow = texture(fluidTex, vec2(u, v)).level(float(0.0));
      const rho = clamp(fNow.r, 0.0, 1.0);
      const velU = fNow.g;
      const velV = fNow.b;
      const vort = fNow.a;

      const rhoUp1 = clamp(texture(fluidTex, vec2(u, max(float(0.0), v.sub(0.11)))).level(float(0.0)).r, 0.0, 1.0);
      const rhoUp2 = clamp(texture(fluidTex, vec2(u, max(float(0.0), v.sub(0.25)))).level(float(0.0)).r, 0.0, 1.0);
      const rhoUp3 = clamp(texture(fluidTex, vec2(u, max(float(0.0), v.sub(0.42)))).level(float(0.0)).r, 0.0, 1.0);
      const upstreamMean = rhoUp1.mul(0.42).add(rhoUp2.mul(0.34)).add(rhoUp3.mul(0.24));
      const tauUp = upstreamMean.mul(v).mul(1.22);
      const rayTrans = exp(tauUp.negate());

      const advPos = vec3(
        vWorldPos.x.mul(1.85).sub(velU.mul(0.14)),
        vWorldPos.y.mul(1.55).sub(uTime.mul(0.075)).sub(velV.mul(0.14)),
        vWorldPos.z.mul(1.85).add(vStratum.mul(0.42)).add(uTime.mul(0.025))
      );
      const mistFbm = mineralToothFbm(advPos, float(1.0));

      // Seam-free periodic shaft noise in Cone mode (u = 0 | 1 wraps identically around 2*PI!)
      const uTheta = u.mul(Math.PI * 2);
      const shaftCoordOpen = vec3(
        u.mul(11.5).sub(velU.mul(0.18)),
        v.mul(2.1).sub(uTime.mul(0.05)),
        vStratum.mul(0.65)
      );
      const shaftCoordCone = vec3(
        cos(uTheta).mul(1.95).sub(velU.mul(0.12)),
        v.mul(2.1).sub(uTime.mul(0.05)),
        sin(uTheta).mul(1.95).add(vStratum.mul(0.45))
      );
      const shaftFbm = mineralToothFbm(mix(shaftCoordOpen, shaftCoordCone, uModeWeights.z), float(1.0));

      const tearGate = smoothstep(float(0.045), float(0.28), rho);
      const mistWeight = rho
        .mul(tearGate)
        .mul(float(0.72).add(mistFbm.mul(0.52)))
        .mul(float(0.82).add(shaftFbm.mul(0.36)));

      const beamCross = exp(vStratum.mul(vStratum).mul(-1.95));
      const topEntry = smoothstep(float(0.0), float(0.015), v);
      const bottomFeather = smoothstep(float(1.0), float(0.68), v);
      const colSpread = float(1.0).div(float(1.0).add(v.mul(0.92)));
      const vertEnv = topEntry.mul(bottomFeather).mul(colSpread);

      const openEdgeL = smoothstep(float(0.0), float(0.055), u);
      const openEdgeR = smoothstep(float(1.0), float(0.945), u);
      const openLateral = openEdgeL.mul(openEdgeR);
      const lateralEnv = mix(openLateral, float(1.0), uModeWeights.z);

      const edgeLipL = exp(u.sub(0.055).mul(u.sub(0.055)).mul(-1150.0));
      const edgeLipR = exp(u.sub(0.945).mul(u.sub(0.945)).mul(-1150.0));
      const slitEdgeCatch = edgeLipL.add(edgeLipR).mul(float(1.0).sub(uModeWeights.z)).mul(0.22);

      const viewDir = normalize(cameraPosition.sub(vWorldPos));
      const nUnit = normalize(vSurfNormal);
      const ndotv = abs(dot(nUnit, viewDir));
      const pathLen = clamp(float(0.30).div(sqrt(ndotv.mul(ndotv).add(0.065))), 0.3, 1.24);

      const beamDir = vec3(0.0, -1.0, 0.0);
      const cosScatter = clamp(dot(viewDir.negate(), beamDir), -1.0, 1.0);
      const hgDen = max(float(0.25), float(1.22).sub(cosScatter.mul(0.62)));
      const miePhase = float(0.74).add(float(0.34).div(hgDen));

      const absVort = abs(vort);
      const shearCaustic = smoothstep(float(3.2), float(22.0), absVort).mul(
        smoothstep(float(0.08), float(0.45), rho)
      );

      const lampNeckGlow = exp(v.mul(-6.2)).mul(float(1.0).add(uTransitionPulse.mul(0.35)));
      const warmCore = mix(cKaolin, cCalcite, clamp(rayTrans.mul(0.75).add(0.2), 0.0, 1.0));
      const amberRim = mix(cRealgar, cOchre, float(0.42));
      const coolTail = mix(cLapis.mul(1.35), cCeladon.mul(0.85), float(0.34));

      const coolFactor = clamp(
        float(1.0).sub(rayTrans).mul(0.48).add(smoothstep(float(0.32), float(0.95), v).mul(0.42)),
        0.0,
        0.72
      );
      let scatterCol = mix(warmCore, coolTail, coolFactor);
      scatterCol = mix(scatterCol, amberRim.mul(1.35), clamp(lampNeckGlow.mul(0.68), 0.0, 0.72));
      scatterCol = scatterCol.add(
        mix(cRealgar, cCalcite, float(0.45)).mul(shearCaustic.mul(0.42))
      );

      const airBeamFloor = float(0.11)
        .add(lampNeckGlow.mul(0.48))
        .add(slitEdgeCatch)
        .mul(tearGate.mul(0.84).add(0.16));

      const totalScatter = mistWeight
        .mul(float(0.82).add(rayTrans.mul(0.85)))
        .add(airBeamFloor)
        .mul(beamCross)
        .mul(vertEnv)
        .mul(lateralEnv)
        .mul(pathLen)
        .mul(miePhase);

      const rawRgb = scatterCol.mul(totalScatter).mul(0.56);
      const compressedRgb = rawRgb.div(vec3(1.0).add(rawRgb.mul(0.62)));
      return compressedRgb;
    });

    veilMaterial.colorNode = evalVeilFragment();
    veilMaterial.opacityNode = float(1.0);
  }

  // ---------------------------------------------------------------------------
  // 12. Grazing Cross-Section Blade Ribbon Material (Zero vanishing card at 90 deg side profile)
  // ---------------------------------------------------------------------------
  const grazingMaterial = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });

  {
    const gUvIn = uv();
    const gStratum = attribute('aStratum', 'float');
    const { pos: surfPos, surfNormal } = buildVeilParametricNodes(gUvIn.x, gUvIn.y, uModeWeights);
    const halfThick = mix(float(BEAM_HALF_THICK_TOP), float(BEAM_HALF_THICK_BOT), gUvIn.y);
    const gPos = surfPos.add(surfNormal.mul(gStratum.mul(halfThick)));

    grazingMaterial.positionNode = gPos;

    const vgUv = gUvIn.toVarying('vGrazeUv');
    const vgStratum = gStratum.toVarying('vGrazeStratum');
    const vgWorld = gPos.toVarying('vGrazeWorld');
    const vgSurfN = surfNormal.toVarying('vGrazeSurfN');

    const evalGrazing = Fn(() => {
      const fSample = texture(fluidTex, vgUv).level(float(0.0));
      const rho = clamp(fSample.r, 0.0, 1.0);
      const viewDir = normalize(cameraPosition.sub(vgWorld));
      const ndotv = abs(dot(normalize(vgSurfN), viewDir));
      const grazeGate = smoothstep(float(0.36), float(0.05), ndotv).mul(
        float(1.0).sub(uModeWeights.z)
      );
      const beamCross = exp(vgStratum.mul(vgStratum).mul(-2.4));
      const vertEnv = smoothstep(float(0.0), float(0.02), vgUv.y)
        .mul(smoothstep(float(1.0), float(0.72), vgUv.y))
        .div(float(1.0).add(vgUv.y.mul(0.85)));
      const lampNeck = exp(vgUv.y.mul(-5.8));
      const col = mix(cCalcite, mix(cRealgar, cOchre, float(0.45)), lampNeck.mul(0.65));
      const strength = grazeGate.mul(beamCross).mul(vertEnv).mul(float(0.24).add(rho.mul(0.62)));
      return col.mul(strength);
    });

    grazingMaterial.colorNode = evalGrazing();
    grazingMaterial.opacityNode = float(1.0);
  }

  return {
    forgedIron,
    satinBrass,
    vernierBrass,
    knurledBrass,
    bluedSteel,
    honedSlate,
    ceramicInsulator,
    opticalGlass,
    opticalBaffle,
    linearFilament,
    coneFilament,
    veilMaterial,
    grazingMaterial,
    uTime,
    uModeWeights,
    uLampHover,
    uTransitionPulse,
    dispose: () => {
      forgedIron.dispose();
      satinBrass.dispose();
      vernierBrass.dispose();
      knurledBrass.dispose();
      bluedSteel.dispose();
      honedSlate.dispose();
      ceramicInsulator.dispose();
      opticalGlass.dispose();
      opticalBaffle.dispose();
      linearFilament.dispose();
      coneFilament.dispose();
      veilMaterial.dispose();
      grazingMaterial.dispose();
    },
  };
}
