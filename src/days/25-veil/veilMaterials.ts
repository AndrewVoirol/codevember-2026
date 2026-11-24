import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  max,
  mix,
  normalLocal,
  normalize,
  positionLocal,
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
import { mineralToothFbm } from '../../core/tslMaterials';
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
  knurledBrass: THREE.MeshStandardNodeMaterial;
  bluedSteel: THREE.MeshStandardNodeMaterial;
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
 * across dark forged iron cooling fins, turned satin brass jaws, and diamond knurling.
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
 * Generates a calibrated studio softbox PMREM environment cubemap so dark forged iron and
 * satin brass read crisp and tactile against #212529 Chalky Slate, including a warm
 * Realgar-Amber / Bone-Calcite lower bounce strip simulating upward glow from the light veil.
 */
export function createVeilStudioEnvironment(
  renderer: THREE.WebGPURenderer
): VeilStudioEnvironment {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0d1115);

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

  // Overhead-left warm calcite key softbox (carves iron cooling fins & brass bevels)
  addSoftbox(11.5, 8.5, [-4.8, 5.6, 4.2], 1.0, 0.96, 0.9, 1.95);
  // Right tall studio scrim (catches knurled brass focus drums & end-cap bolts)
  addSoftbox(8.5, 11.0, [6.2, 2.8, 3.2], 0.99, 0.95, 0.89, 1.75);
  // Rear-top cool slate grazing rim scrim (separates dark forged iron cleanly off #212529)
  addSoftbox(13.0, 5.2, [-0.8, 5.2, -5.8], 0.88, 0.94, 1.0, 1.65);
  // Zenith diffused canopy
  addSoftbox(9.5, 9.5, [0.0, 7.2, 0.4], 0.96, 0.94, 0.9, 1.05);
  // Warm upward bounce card from the illuminated Bone-Calcite & Realgar-Amber light curtain below!
  addSoftbox(11.0, 4.2, [0.0, -5.8, 0.6], 1.0, 0.78, 0.52, 1.25);

  const renderTarget = pmrem.fromScene(envScene, 0.08);
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
 * Produces crisp sub-degree brushed-metal grain without lumpy macro distortion.
 */
const perturbNormal3DFbm = Fn(
  ([baseNorm_in, pos_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const sc = float(scale_in);
    const eps = float(0.16).div(max(float(1.0), sc));

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), sc);
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), sc);
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), sc);
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), sc);

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

/**
 * Shared GPU TSL parametric veil surface evaluator matching `evalVeilSurface` in `veilSurface.ts`.
 * Returns `vec3` surface point; normal is computed alongside via analytical `tanV x tanU`.
 */
function buildVeilParametricNodes(uNode: TslNode, vNode: TslNode, uModeWeights: TslNode) {
  const s = uNode.sub(0.5);
  const theta = s.mul(Math.PI * 2);
  const sinT = sin(theta);
  const cosT = cos(theta);
  const y = float(VEIL_Y_TOP).sub(vNode.mul(VEIL_HEIGHT));

  // Mode 0: Planar collimated curtain
  const hwP = mix(float(PLANAR_HALF_W_TOP), float(PLANAR_HALF_W_BOT), vNode);
  const dHwP = float(PLANAR_HALF_W_BOT - PLANAR_HALF_W_TOP);
  const xP = s.mul(2.0).mul(hwP);
  const zP = float(0.0);

  // Mode 1: Graceful Curved S-Veil
  const hwS = mix(float(CURVED_S_HALF_W_TOP), float(CURVED_S_HALF_W_BOT), vNode);
  const dHwS = float(CURVED_S_HALF_W_BOT - CURVED_S_HALF_W_TOP);
  const ampS = mix(float(CURVED_S_AMP_TOP), float(CURVED_S_AMP_BOT), vNode);
  const dAmpS = float(CURVED_S_AMP_BOT - CURVED_S_AMP_TOP);
  const xS = s.mul(2.0).mul(hwS);
  const zS = ampS.mul(sinT);

  // Mode 2: Hollow Elliptical Light Cone
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

  // Exact partial derivatives dP/du and dP/dv
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
 *   - Forged dark-iron collimator housing & cooling fins
 *   - Machined satin brass jaws, bezels, rails & end-caps
 *   - Diamond-knurled satin brass focus drums
 *   - Blued steel bolts & suspension hardware
 *   - Incandescent Realgar-Amber / Bone-Calcite internal slit filaments
 *   - Multi-stratum volumetric Mie & Beer-Lambert light-and-haze curtain
 *   - Grazing cross-section optical blade ribbon (for exact 90° side-profile depth)
 */
export function createVeilMaterials(fluidTex: THREE.DataTexture): VeilMaterials {
  const uTime = uniform(0);
  const uModeWeights = uniform(new THREE.Vector3(1, 0, 0));
  const uLampHover = uniform(0);
  const uTransitionPulse = uniform(0);

  // Mineral pigment linear RGB anchors (strictly zero digital neons)
  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const cKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const cRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const cOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const cLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);
  const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const cUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const cAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Dark Forged-Iron Optical Housing & Cooling Fins
  // ---------------------------------------------------------------------------
  const forgedIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.84,
    roughness: 0.34,
    envMapIntensity: 1.28,
  });
  {
    const fineGrain = mineralToothFbm(positionLocal, float(72.0));
    const patina = mineralToothFbm(positionLocal.add(vec3(2.9, 4.1, 1.7)), float(18.0));
    const ironDeep = mix(vec3(0.058, 0.064, 0.072), cUmber.mul(0.28), float(0.25));
    const ironMachined = mix(vec3(0.135, 0.145, 0.158), cKaolin.mul(0.22), float(0.2));
    const hoverLift = cOchre.mul(uLampHover.mul(0.045));
    forgedIron.colorNode = mix(
      ironDeep,
      ironMachined,
      smoothstep(float(0.3), float(0.78), fineGrain.mul(0.68).add(patina.mul(0.32)))
    ).add(hoverLift);
    forgedIron.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(86.0), float(0.009));
    forgedIron.roughnessNode = clamp(
      float(0.32).add(fineGrain.sub(0.5).mul(0.08)).add(patina.sub(0.5).mul(0.05)),
      0.24,
      0.44
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Satin Brass (End-Caps, Aperture Jaws, Iris Bezel, Guide Rails)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.32,
  });
  {
    const fineSatin = mineralToothFbm(positionLocal, float(92.0));
    const micro = mineralToothFbm(positionLocal.add(vec3(4.3, 1.7, 5.9)), float(145.0));
    const brassAged = mix(cUmber.mul(1.2), cOchre.mul(0.95), float(0.72));
    const brassBright = mix(cOchre.mul(1.18), cCalcite.mul(0.88), float(0.36));
    const hoverWarmth = cRealgar.mul(uLampHover.mul(0.12));
    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.25), float(0.78), fineSatin.mul(0.6).add(micro.mul(0.4)))
    ).add(hoverWarmth);
    satinBrass.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(112.0), float(0.004));
    satinBrass.roughnessNode = clamp(
      float(0.21).add(fineSatin.sub(0.5).mul(0.05)).add(micro.sub(0.5).mul(0.03)),
      0.16,
      0.29
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Diamond-Knurled Satin Brass Focus & Mode Selector Drums
  // ---------------------------------------------------------------------------
  const knurledBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.23,
    envMapIntensity: 1.34,
  });
  {
    const aKnurlValley = attribute('aKnurlValley', 'float');
    const fineGrain = mineralToothFbm(positionLocal, float(64.0));
    const grooveFactor = clamp(aKnurlValley, 0.0, 1.0);
    const valleyDark = cUmber.mul(0.52);
    const crestBright = mix(cOchre.mul(1.25), cCalcite.mul(0.94), float(0.38));
    const hoverLift = cOchre.mul(uLampHover.mul(0.18));
    knurledBrass.colorNode = mix(
      valleyDark,
      crestBright,
      smoothstep(float(0.08), float(0.92), grooveFactor)
    ).add(hoverLift);
    knurledBrass.roughnessNode = clamp(
      mix(float(0.46), float(0.18), grooveFactor).add(fineGrain.sub(0.5).mul(0.04)),
      0.15,
      0.52
    );
    knurledBrass.metalnessNode = mix(float(0.74), float(0.94), grooveFactor);
  }

  // ---------------------------------------------------------------------------
  // 4. Tempered Blued-Steel Fasteners & Shafts
  // ---------------------------------------------------------------------------
  const bluedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.27,
    envMapIntensity: 1.18,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(56.0));
    const steelDark = vec3(0.052, 0.068, 0.092);
    const steelHi = mix(vec3(0.16, 0.2, 0.25), cAzurite.mul(0.88), float(0.35));
    bluedSteel.colorNode = mix(steelDark, steelHi, smoothstep(float(0.24), float(0.8), grain));
    bluedSteel.roughnessNode = clamp(float(0.24).add(grain.mul(0.11)), 0.18, 0.39);
  }

  // ---------------------------------------------------------------------------
  // 5. Recessed Glowing Incandescent Optical Slit Filaments (Linear & Annular)
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
    const endFade = smoothstep(float(0.44), float(0.28), abs(positionLocal.x));
    const filColor = mix(cRealgar.mul(1.85), cCalcite.mul(2.2), endFade.mul(0.55).add(0.25)).mul(
      pulse
    );
    linearFilament.colorNode = filColor;
    linearFilament.opacityNode = openVis.mul(endFade.mul(0.85).add(0.15));
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
  // 6. Volumetric 5-Stratum Light-and-Haze Veil Material (TSL)
  //    Coupled directly to the live 128x128 RGBA16F Navier-Stokes fluid texture,
  //    with exact upstream Beer-Lambert ray-occlusion god-rays, Henyey-Greenstein
  //    Mie warmth, grazing optical path-length lift, and sheared caustic curl rims!
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

    // Subtle 3D aerodynamic flutter along surface normal driven by live vorticity
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

      // Unconditional explicit-LOD texture fetches (100% uniform control flow)
      const fNow = texture(fluidTex, vec2(u, v)).level(float(0.0));
      const rho = clamp(fNow.r, 0.0, 1.0);
      const velU = fNow.g;
      const velV = fNow.b;
      const vort = fNow.a;

      // Authentic 3-tap upstream Beer-Lambert optical ray occlusion along constant ray `u`
      // Thick mist curls up top cast natural soft light-shafts & shadow streaks downward!
      const rhoUp1 = clamp(texture(fluidTex, vec2(u, max(float(0.0), v.sub(0.11)))).level(float(0.0)).r, 0.0, 1.0);
      const rhoUp2 = clamp(texture(fluidTex, vec2(u, max(float(0.0), v.sub(0.25)))).level(float(0.0)).r, 0.0, 1.0);
      const rhoUp3 = clamp(texture(fluidTex, vec2(u, max(float(0.0), v.sub(0.42)))).level(float(0.0)).r, 0.0, 1.0);
      const upstreamMean = rhoUp1.mul(0.42).add(rhoUp2.mul(0.34)).add(rhoUp3.mul(0.24));
      const tauUp = upstreamMean.mul(v).mul(1.22);
      const rayTrans = exp(tauUp.negate());

      // Smooth velocity-advected 3D FBM micro-mist variation across strata (zero Voronoi or C0 noise)
      const advPos = vec3(
        vWorldPos.x.mul(1.85).sub(velU.mul(0.14)),
        vWorldPos.y.mul(1.55).sub(uTime.mul(0.075)).sub(velV.mul(0.14)),
        vWorldPos.z.mul(1.85).add(vStratum.mul(0.42)).add(uTime.mul(0.025))
      );
      const mistFbm = mineralToothFbm(advPos, float(1.0));
      const shaftFbm = mineralToothFbm(
        vec3(u.mul(11.5).sub(velU.mul(0.18)), v.mul(2.1).sub(uTime.mul(0.05)), vStratum.mul(0.65)),
        float(1.0)
      );

      // Gate cleanly on fluid density so pointer-sliced tears open into crisp dark #212529 voids
      const tearGate = smoothstep(float(0.045), float(0.28), rho);
      const mistWeight = rho
        .mul(tearGate)
        .mul(float(0.72).add(mistFbm.mul(0.52)))
        .mul(float(0.82).add(shaftFbm.mul(0.36)));

      // Beam cross-section Gaussian profile across strata (bright central optical axis, soft penumbra)
      const beamCross = exp(vStratum.mul(vStratum).mul(-1.95));

      // Physical collimated slit-lamp vertical & lateral aperture envelopes
      const topEntry = smoothstep(float(0.0), float(0.015), v);
      const bottomFeather = smoothstep(float(1.0), float(0.68), v);
      const colSpread = float(1.0).div(float(1.0).add(v.mul(0.92)));
      const vertEnv = topEntry.mul(bottomFeather).mul(colSpread);

      // Horizontal penumbra fades open curtain edges smoothly; blends to 1.0 in seamless 360° Cone mode
      const openEdgeL = smoothstep(float(0.0), float(0.055), u);
      const openEdgeR = smoothstep(float(1.0), float(0.945), u);
      const openLateral = openEdgeL.mul(openEdgeR);
      const lateralEnv = mix(openLateral, float(1.0), uModeWeights.z);

      // Subtle collimated slit-edge crisp optical highlight along outer boundaries in open modes
      const edgeLipL = exp(u.sub(0.055).mul(u.sub(0.055)).mul(-1150.0));
      const edgeLipR = exp(u.sub(0.945).mul(u.sub(0.945)).mul(-1150.0));
      const slitEdgeCatch = edgeLipL.add(edgeLipR).mul(float(1.0).sub(uModeWeights.z)).mul(0.22);

      // Physical thin-sheet optical path length: delicate face-on, luminous sculptural blade at grazing tangents
      const viewDir = normalize(cameraPosition.sub(vWorldPos));
      const nUnit = normalize(vSurfNormal);
      const ndotv = abs(dot(nUnit, viewDir));
      const pathLen = clamp(float(0.30).div(sqrt(ndotv.mul(ndotv).add(0.065))), 0.3, 1.24);

      // Forward-peaked Mie phase factor relative to overhead slit-lamp beam axis (-Y)
      const beamDir = vec3(0.0, -1.0, 0.0);
      const cosScatter = clamp(dot(viewDir.negate(), beamDir), -1.0, 1.0);
      const hgDen = max(float(0.25), float(1.22).sub(cosScatter.mul(0.62)));
      const miePhase = float(0.74).add(float(0.34).div(hgDen));

      // Sheared Kelvin-Helmholtz vortex caustic lip highlight
      const absVort = abs(vort);
      const shearCaustic = smoothstep(float(3.2), float(22.0), absVort).mul(
        smoothstep(float(0.08), float(0.45), rho)
      );

      // Pure Mineral Pigment Spectral Shading (Zero Digital Neons!)
      const lampNeckGlow = exp(v.mul(-6.2)).mul(float(1.0).add(uTransitionPulse.mul(0.35)));
      const warmCore = mix(cKaolin, cCalcite, clamp(rayTrans.mul(0.75).add(0.2), 0.0, 1.0));
      const amberRim = mix(cRealgar, cOchre, float(0.42));
      const coolTail = mix(cLapis.mul(1.35), cCeladon.mul(0.85), float(0.34));

      // Blend warm Bone-Calcite core + Realgar-Amber slit rim + cool Lapis shadow falloff at lower/outer edges
      const coolFactor = clamp(
        float(1.0).sub(rayTrans).mul(0.48).add(smoothstep(float(0.32), float(0.95), v).mul(0.42)),
        0.0,
        0.72
      );
      let scatterCol = mix(warmCore, coolTail, coolFactor);
      scatterCol = mix(scatterCol, amberRim.mul(1.35), clamp(lampNeckGlow.mul(0.68), 0.0, 0.72));
      // Warm sheared amber-calcite caustic rim along energetic Kelvin-Helmholtz curl edges
      scatterCol = scatterCol.add(
        mix(cRealgar, cCalcite, float(0.45)).mul(shearCaustic.mul(0.42))
      );

      // Subtle continuous luminous beam presence directly beneath the slit even in thin air
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

      // Calibrated linear radiance across 5 additive strata with smooth highlight shoulder
      const rawRgb = scatterCol.mul(totalScatter).mul(0.56);
      const compressedRgb = rawRgb.div(vec3(1.0).add(rawRgb.mul(0.62)));
      return compressedRgb;
    });

    veilMaterial.colorNode = evalVeilFragment();
    veilMaterial.opacityNode = float(1.0);
  }

  // ---------------------------------------------------------------------------
  // 7. Grazing Cross-Section Blade Ribbon Material (Zero vanishing card at 90° side profile!)
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
      // Invisible face-on (`ndotv > 0.34`), smoothly illuminates ONLY at near-90° grazing profile!
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
    knurledBrass,
    bluedSteel,
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
      knurledBrass.dispose();
      bluedSteel.dispose();
      linearFilament.dispose();
      coneFilament.dispose();
      veilMaterial.dispose();
      grazingMaterial.dispose();
    },
  };
}
