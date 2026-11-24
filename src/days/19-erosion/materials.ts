import * as THREE from 'three/webgpu';
import {
  Fn,
  PI,
  TWO_PI,
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
  floor,
  fract,
  length,
  max,
  mix,
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
  step,
  texture,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
} from 'three/tsl';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { AMMONITE_THETA_MAX } from './ammonite';
import {
  STRATA_BASE_FREQ,
  STRATA_WARP_AMP,
  evalBaseBoulderRadiusTsl,
  octDirToUvTsl,
} from './octahedral';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface ErosionMaterials {
  limestone: THREE.MeshStandardNodeMaterial;
  ammonite: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  darkIron: THREE.MeshStandardNodeMaterial;
  carbideTip: THREE.MeshStandardNodeMaterial;
  hoseRubber: THREE.MeshStandardNodeMaterial;
  abrasiveJet: THREE.MeshBasicNodeMaterial;
  gritChips: THREE.MeshStandardNodeMaterial;
  dustPlume: THREE.MeshBasicNodeMaterial;
  uStrataAxis: { value: THREE.Vector3 };
  uStrataSeed: { value: number };
  uBlastSpotWorld: { value: THREE.Vector4 }; // xyz = impact pos, w = active intensity [0..1]
  uJetAlpha: { value: number };
  uTime: { value: number };
  dispose: () => void;
}

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Scale-adaptive eps keeps finite differences cleanly below Nyquist across every FBM frequency.
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

    // Dimensionless noise-lattice gradient: 1 / (4 * 0.16) = 1.5625 (eliminates 1/eps frequency blow-up)
    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

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
 * Generates a broadly diffused photographer's studio softbox PMREM environment cubemap
 * so pyritized bronze-gold ammonite ribs, lathe-turned satin brass nozzle, and oil-blackened
 * forged iron gleam richly against the #212529 Chalky Slate backdrop without dark-void collapse.
 */
export function createStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0c1015);

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

  // Broad warm calcite overhead-left studio key softbox
  addSoftbox(11.5, 9.2, [-4.8, 5.4, 4.2], 1.0, 0.96, 0.89, 1.98);
  // Tall warm-neutral right studio scrim (catches pyrite rib crests & satin brass nozzle barrel)
  addSoftbox(8.8, 10.5, [6.0, 2.8, 3.4], 0.99, 0.95, 0.88, 1.85);
  // Front-upper gentle fill diffuser (prevents metallic ribs & diamond knurling from collapsing dark)
  addSoftbox(8.4, 6.4, [0.4, 2.2, 6.8], 0.98, 0.95, 0.91, 1.24);
  // Cool slate rear-top grazing rim scrim (carves clean metallic separation from #212529)
  addSoftbox(12.2, 5.4, [-1.2, 5.2, -5.8], 0.88, 0.94, 1.0, 1.56);
  // Rear-right secondary soft fill (for back-face 360-deg orbit blasting)
  addSoftbox(9.5, 8.0, [4.8, 1.8, -5.6], 0.94, 0.94, 0.96, 1.36);
  // Subtle warm specimen-bench floor bounce
  addSoftbox(12.0, 12.0, [0, -6.2, 0.8], 0.96, 0.92, 0.86, 0.46);

  const renderTarget = pmrem.fromScene(envScene, 0.08);
  for (const d of disposables) d.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

/**
 * Constructs all TSL WebGPU materials for Day 19 - Erosion using strictly canonical
 * physical mineral pigments (`boneCalcite`, `warmKaolin`, `yellowOchre`, `rawUmber`,
 * `burntSienna`, `hematiteRust`, `azuritePrussian`, `verdigrisMalachite`, `realgarAmber`).
 */
export function createErosionMaterials(erosionTex: THREE.DataTexture): ErosionMaterials {
  const uStrataAxis = uniform(new THREE.Vector3(0.24, 0.94, 0.24).normalize());
  const uStrataSeed = uniform(1.0);
  const uBlastSpotWorld = uniform(new THREE.Vector4(0, 0, 1, 0));
  const uJetAlpha = uniform(0.0);
  const uTime = uniform(0.0);

  const pCalcite = createPigmentUniforms('boneCalcite');
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pSienna = createPigmentUniforms('burntSienna');
  const pHematite = createPigmentUniforms('hematiteRust');
  const pAzurite = createPigmentUniforms('azuritePrussian');
  const pVerdigris = createPigmentUniforms('verdigrisMalachite');
  const pAmber = createPigmentUniforms('realgarAmber');

  // -------------------------------------------------------------------------
  // 1. Carvable Stratified Jurassic Limestone Concretion (True 3D Position Petrography)
  // -------------------------------------------------------------------------
  const limestone = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.88,
    envMapIntensity: 0.56,
    alphaTest: 0.5,
  });
  {
    const unitDirVert = normalize(positionGeometry);
    const vUnitDir = unitDirVert.toVarying('vNoduleUnitDir');

    // Vertex displacement: exact C-infinity analytical boulder hull * multiplicative scaleFactor
    // Eliminates 100% of octahedral x=0 / y=0 creases at EVERY carve depth from intact (1.0) down to zero (0.0)
    const octUvVert = octDirToUvTsl(unitDirVert);
    const hSampleVert = texture(erosionTex, octUvVert).level(float(0.0));
    const rVert = evalBaseBoulderRadiusTsl(unitDirVert).mul(max(float(0.0), hSampleVert.r));
    const displacedVert = unitDirVert.mul(rVert);
    // Pass TRUE interpolated 3D displaced position to fragment shader so steep carved crater walls
    // slice cleanly across horizontal 3D strata beds with ZERO radial waterfall streak smearing!
    const vDisplacedPos = displacedVert.toVarying('vNoduleLocalPos');
    limestone.positionNode = displacedVert;

    // Fragment evaluation (strictly uniform control flow for every texture tap)
    const d = normalize(vec3(vUnitDir));
    const pObj = vec3(vDisplacedPos);
    const rFragObj = length(pObj);
    const octUvFrag = octDirToUvTsl(d);
    const hSample = texture(erosionTex, octUvFrag).level(float(0.0));
    const rCenter = evalBaseBoulderRadiusTsl(d).mul(max(float(0.0), hSample.r));
    const cutDepth = max(float(0.0), hSample.g);
    const freshChalk = clamp(hSample.b, 0.0, 1.0);
    const fossilClearance = hSample.a;

    // Exact basis-invariant spherical heightfield normal with slope stabilization
    // so carved crater cliffs stay crisp and chiseled without noisy radial spikes
    const upHint = select(abs(d.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const T = normalize(cross(upHint, d));
    const B = normalize(cross(d, T));
    const eps = float(0.0125);

    const dTp = normalize(d.add(T.mul(eps)));
    const dTm = normalize(d.sub(T.mul(eps)));
    const dBp = normalize(d.add(B.mul(eps)));
    const dBm = normalize(d.sub(B.mul(eps)));

    const rTp = evalBaseBoulderRadiusTsl(dTp).mul(
      max(float(0.0), texture(erosionTex, octDirToUvTsl(dTp)).level(float(0.0)).r)
    );
    const rTm = evalBaseBoulderRadiusTsl(dTm).mul(
      max(float(0.0), texture(erosionTex, octDirToUvTsl(dTm)).level(float(0.0)).r)
    );
    const rBp = evalBaseBoulderRadiusTsl(dBp).mul(
      max(float(0.0), texture(erosionTex, octDirToUvTsl(dBp)).level(float(0.0)).r)
    );
    const rBm = evalBaseBoulderRadiusTsl(dBm).mul(
      max(float(0.0), texture(erosionTex, octDirToUvTsl(dBm)).level(float(0.0)).r)
    );

    const drdt = clamp(rTp.sub(rTm).div(eps.mul(2.0)), -1.85, 1.85);
    const drdb = clamp(rBp.sub(rBm).div(eps.mul(2.0)), -1.85, 1.85);
    const nMacro = normalize(d.mul(max(rCenter, float(0.02))).sub(T.mul(drdt)).sub(B.mul(drdb)));

    // 3D Object-Space Jurassic Calcareous Limestone Petrography evaluated on TRUE 3D `pObj`
    const seedShift = vec3(
      uStrataSeed.mul(1.73),
      uStrataSeed.mul(0.91),
      uStrataSeed.mul(-1.37)
    );
    const pSample = pObj.add(seedShift);

    const macroWarp = mineralToothFbm(pSample, float(2.15));
    const chiseledFacet = mineralToothFbm(pSample.add(vec3(4.1, 2.3, 6.7)), float(14.5));
    const poreField = mineralToothFbm(pSample.add(vec3(11.3, 3.7, 7.9)), float(36.0));
    const fineOolite = mineralToothFbm(pSample.add(vec3(9.2, 5.8, 1.4)), float(68.0));
    const microTooth = mineralToothFbm(pSample.add(vec3(2.8, 12.1, 5.6)), float(150.0));

    // Unified aperiodic geological bedding (matches erosionField.ts STRATA_BASE_FREQ = 10.5 1:1!)
    // Broad pale calcareous benches alternating irregularly with warm ochre marl & umber shale partings
    const strataCoord = dot(pObj, uStrataAxis)
      .mul(STRATA_BASE_FREQ)
      .add(macroWarp.mul(STRATA_WARP_AMP))
      .add(chiseledFacet.mul(0.28));

    const bedRaw = sin(strataCoord)
      .mul(0.52)
      .add(sin(strataCoord.mul(0.43).sub(0.85)).mul(0.34))
      .add(sin(strataCoord.mul(2.17).add(1.35)).mul(0.18));
    const bedNorm = clamp(bedRaw.mul(0.5).add(0.5), 0.0, 1.0);

    // Rich aperiodic geological band masks (unmistakably stratified sedimentary limestone, zero zebra periodicity)
    const calciteBench = smoothstep(float(0.38), float(0.78), bedNorm);
    const marlBand = smoothstep(float(0.20), float(0.62), float(1.0).sub(bedNorm)).mul(
      float(0.60).add(macroWarp.mul(0.40))
    );
    const hairlineParting = pow(clamp(float(1.0).sub(bedNorm), 0.0, 1.0), 4.6);
    const ironStain = pow(
      clamp(sin(strataCoord.mul(1.37).add(0.45)).mul(0.5).add(0.5), 0.0, 1.0),
      4.4
    ).mul(smoothstep(float(0.26), float(0.74), macroWarp));

    // Distinct calcareous sedimentary dissolution pores / vugs & pale bioclastic oolite grains
    const vugPit = smoothstep(float(0.71), float(0.89), poreField);
    const ooliteFlecks = smoothstep(float(0.64), float(0.88), microTooth);

    // 2-Flux Kubelka-Munk calcareous limestone, yellow-ochre marl & ironstone laminae
    const marlBacking = mix(pUmber.rInf.mul(0.44), pSienna.rInf.mul(0.56), ironStain.mul(0.75));
    const ochreMarlGlaze = kubelkaMunkGlaze(
      pOchre.K,
      pOchre.S,
      float(0.38).add(marlBand.mul(0.68)).add(ironStain.mul(0.48)),
      marlBacking
    );
    const kaolinBase = kubelkaMunkGlaze(
      pKaolin.K,
      pKaolin.S,
      float(0.58).add(calciteBench.mul(0.95)).add(chiseledFacet.mul(0.26)),
      ochreMarlGlaze
    );
    const calciteCore = kubelkaMunkGlaze(
      pCalcite.K,
      pCalcite.S,
      float(0.34).add(calciteBench.mul(0.92)).add(fineOolite.mul(0.24)),
      kaolinBase
    );

    // Authentic Lyme Regis weathered calcareous limestone exterior vs. luminous freshly air-abraded interior
    const stoneWithSeams = mix(
      mix(calciteCore, pOchre.rInf.mul(0.82), marlBand.mul(0.34).add(ironStain.mul(0.32))),
      mix(pUmber.rInf.mul(0.34), pSienna.rInf.mul(0.48), ironStain.mul(0.5)),
      hairlineParting.mul(0.62)
    );
    const withPoresAndClasts = mix(
      stoneWithSeams.mul(float(0.86).add(fineOolite.mul(0.22)).add(ooliteFlecks.mul(0.08))),
      pUmber.rInf.mul(0.30),
      vugPit.mul(0.46)
    );

    // Warm weathered mineral patina on outer crust -> bright freshly cleaved calcareous strata inside craters
    const freshCutMask = smoothstep(float(0.005), float(0.036), cutDepth);
    const weatheredRind = mix(
      withPoresAndClasts.mul(0.76),
      mix(pUmber.rInf.mul(0.44), pOchre.rInf.mul(0.58), macroWarp),
      smoothstep(float(0.56), float(0.88), chiseledFacet).mul(0.20).add(vugPit.mul(0.18))
    );
    const freshInterior = mix(
      withPoresAndClasts.mul(1.06),
      pCalcite.rInf.mul(0.98),
      float(0.14).add(ooliteFlecks.mul(0.10))
    );
    const rockRgb = mix(weatheredRind, freshInterior, freshCutMask);

    // Textured crushed dolomite powder dusted along freshly abraded crater rims (never solid white smears!)
    const chalkLift = clamp(
      freshChalk.mul(float(0.38).add(fineOolite.mul(0.36)).add(microTooth.mul(0.26))),
      0.0,
      0.68
    );
    const withChalk = mix(rockRgb, pCalcite.rInf.mul(1.02), chalkLift.mul(0.36));

    // Smooth crevice contact AO where limestone matrix hugs the emerging pyritized ammonite
    const collarSeam = exp(fossilClearance.mul(fossilClearance).mul(-8.0));
    const deepCreviceAO = smoothstep(float(-0.42), float(0.32), fossilClearance).mul(0.26);
    const withCollar = mix(
      withChalk.mul(float(1.0).sub(deepCreviceAO)),
      pCalcite.rInf.mul(0.96),
      collarSeam.mul(0.08)
    );

    // Subtle live abrasive spot highlight directly under active blast impact
    const dBlast = length(positionWorld.sub(uBlastSpotWorld.xyz));
    const blastSpot = exp(dBlast.mul(dBlast).mul(-175.0)).mul(uBlastSpotWorld.w);
    const finalColor = mix(withCollar, pCalcite.rInf.mul(1.06), blastSpot.mul(0.34));

    // Micro-normals: crisp chiseled conchoidal rock facets, gritty oolite & porous vug pits,
    // and sharp unified geological bedding-bench ledges along uStrataAxis
    const nCoarse = perturbNormal3DFbm(
      nMacro,
      pSample,
      float(15.5),
      mix(float(0.34), float(0.46), freshCutMask)
    );
    const nPores = perturbNormal3DFbm(
      nCoarse,
      pSample.add(vec3(11.3, 3.7, 7.9)),
      float(38.0),
      float(0.18).add(vugPit.mul(0.26))
    );
    const nFine = perturbNormal3DFbm(
      nPores,
      pSample.add(vec3(5.4, 8.2, 3.1)),
      float(88.0),
      float(0.22)
    );
    // Exact analytical derivative of aperiodic bedRaw along uStrataAxis
    const bedDeriv = cos(strataCoord)
      .mul(0.52)
      .add(cos(strataCoord.mul(0.43).sub(0.85)).mul(0.146))
      .add(cos(strataCoord.mul(2.17).add(1.35)).mul(0.39));
    const bedSlope = bedDeriv.mul(0.24).mul(float(0.65).add(freshCutMask.mul(0.45)));
    const strataTan = uStrataAxis.sub(nFine.mul(dot(uStrataAxis, nFine)));
    const nStratified = normalize(nFine.sub(strataTan.mul(bedSlope)));

    limestone.normalNode = transformNormalToView(nStratified);
    limestone.colorNode = finalColor;
    // Clean opaque alpha-cutout retirement once limestone is blasted away to zero thickness
    // (discards strictly inside the ammonite core / collapsed origin in both color & shadow passes)
    limestone.opacityNode = step(float(0.018), rFragObj);
    limestone.roughnessNode = clamp(
      float(0.86)
        .add(fineOolite.sub(0.5).mul(0.14))
        .add(chalkLift.mul(0.08))
        .add(vugPit.mul(0.10))
        .add(hairlineParting.mul(0.06))
        .sub(ironStain.mul(0.08))
        .sub(ooliteFlecks.mul(0.07)),
      0.66,
      0.97
    );
  }

  // -------------------------------------------------------------------------
  // 2. Embedded Pyritized Ammonite Relic (Gleaming FeS2 + Etched Dendritic Septal Sutures)
  // -------------------------------------------------------------------------
  const ammonite = new THREE.MeshStandardNodeMaterial({
    metalness: 0.87,
    roughness: 0.28,
    envMapIntensity: 1.44,
  });
  {
    const aMeta = attribute('aAmmoniteMeta', 'vec4'); // [ribCrest, whorlNorm, sutureZone, venterMask]
    const ribCrest = clamp(aMeta.x, 0.0, 1.0);
    const whorlNorm = clamp(aMeta.y, 0.0, 1.0);
    const venterMask = clamp(aMeta.w, 0.0, 1.0);

    // Sample live erosion heightfield to dust newly-exposed inter-rib valleys with receding calcite chalk
    const dirAm = normalize(positionLocal);
    const hAm = texture(erosionTex, octDirToUvTsl(dirAm)).level(float(0.0));
    const clearance = hAm.a; // <0 buried, 0..1 clearing from freshly exposed to brushed clean
    const residualChalkMask = smoothstep(float(0.76), float(0.04), clearance)
      .mul(float(1.0).sub(ribCrest.mul(0.86)))
      .mul(float(1.0).sub(venterMask.mul(0.58)));

    // Continuous spiral coordinates for high-resolution fragment-space phylloceratic septal sutures & growth striae
    const thetaU = uv().x.mul(AMMONITE_THETA_MAX);
    const psiV = uv().y.mul(TWO_PI);
    const absSinPsi = abs(sin(psiV));
    const sutureFlankGate = smoothstep(float(0.20), float(0.72), absSinPsi)
      .mul(smoothstep(float(0.16), float(0.48), whorlNorm))
      .mul(float(1.0).sub(venterMask.mul(0.85)));

    // Intricate multi-harmonic foliaceous phylloceratic septal suture lines (~11 septa per turn)
    // Calibrated so dendritic lobes & saddles read sharply at 1:1 Retina macro without sub-pixel aliasing!
    const lobe1 = sin(psiV.mul(5.0)).mul(1.45);
    const lobe2 = sin(psiV.mul(13.0).add(thetaU.mul(2.0))).mul(0.58);
    const lobe3 = sin(psiV.mul(23.0).sub(thetaU.mul(4.0))).mul(0.22);
    const sutureArg = thetaU.mul(11.0).add(lobe1).add(lobe2).add(lobe3);
    const sutureSin = sin(sutureArg);
    const sutureCos = cos(sutureArg);
    const sutureCrease = float(1.0)
      .sub(smoothstep(float(0.0), float(0.32), abs(sutureSin)))
      .mul(sutureFlankGate);
    const sutureShoulder = float(1.0)
      .sub(smoothstep(float(0.0), float(0.56), abs(sutureSin)))
      .mul(sutureFlankGate);

    // Fine paleontological shell growth striae along whorl flanks
    const growthStriae = sin(thetaU.mul(185.0).add(cos(psiV).mul(2.4)))
      .mul(0.5)
      .add(0.5)
      .mul(sutureFlankGate);

    // Crystalline cubic pyrite micro-grain & metallic glint
    const crystalMed = mineralToothFbm(positionLocal, float(28.0));
    const crystalFine = mineralToothFbm(positionLocal.add(vec3(3.7, 7.1, 2.9)), float(96.0));
    const pyriteGlint = smoothstep(float(0.66), float(0.92), crystalFine);

    // Authentic Jurassic pyritized ammonite FeS2 metallic reflectance (strictly natural pigments):
    // Deep oxidized bronze-umber in inter-rib valleys -> warm marcasite-gold flanks -> burnished pale pyrite crests
    const darkBronzeValley = mix(
      pUmber.rInf.mul(0.30),
      pHematite.rInf.mul(0.28),
      crystalMed.mul(0.45)
    );
    const warmPyriteMid = mix(
      pOchre.rInf.mul(0.78),
      pAmber.rInf.mul(0.48),
      float(0.22).add(whorlNorm.mul(0.14))
    );
    const brightPyriteCrest = mix(
      pOchre.rInf.mul(1.06),
      pCalcite.rInf.mul(0.90),
      float(0.36).add(pyriteGlint.mul(0.26))
    );

    const ribShaded = mix(
      darkBronzeValley,
      mix(warmPyriteMid, brightPyriteCrest, smoothstep(float(0.28), float(0.90), ribCrest)),
      smoothstep(float(0.05), float(0.70), ribCrest.add(venterMask.mul(0.36)))
    ).mul(float(0.95).add(growthStriae.mul(0.07)));

    // Restrained natural bornite / marcasite / verdigris thin-oxide mineral iridescence along flanks & sutures
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const ndotv = clamp(dot(normalize(normalWorld), viewDir), 0.0, 1.0);
    const fresnelGrazing = pow(float(1.0).sub(ndotv), 2.5);
    const tarnishTone = mix(
      pAzurite.rInf.mul(0.72),
      pVerdigris.rInf.mul(0.76),
      smoothstep(float(0.30), float(0.75), crystalMed)
    );
    const grazingTarnish = fresnelGrazing
      .mul(smoothstep(float(0.32), float(0.76), crystalMed).add(sutureShoulder.mul(0.45)))
      .mul(0.17);

    // Dark oxidized bronze-umber etched inside dendritic septal suture lines, subtly rimmed with pale pyrite
    const withSutureRim = mix(ribShaded, brightPyriteCrest.mul(1.02), sutureShoulder.mul(0.24));
    const withSuture = mix(withSutureRim, darkBronzeValley.mul(0.38), sutureCrease.mul(0.84));
    const withTarnish = mix(withSuture, tarnishTone, clamp(grazingTarnish.mul(1.25), 0.0, 0.28));

    // Oolitic pale Bone-Calcite matrix dust caught in unblasted inter-rib furrows until blasted clean
    const chalkNoise = smoothstep(
      float(0.24),
      float(0.76),
      crystalMed.mul(0.55).add(crystalFine.mul(0.45))
    );
    const dustFill = clamp(
      residualChalkMask.mul(float(0.44).add(chalkNoise.mul(0.56))),
      0.0,
      0.84
    );
    const finalPyrite = mix(withTarnish, pCalcite.rInf.mul(0.88), dustFill);

    // Etch dendritic septal sutures & subtle spiral growth striae directly into surface normals!
    const nBase = normalize(normalLocal);
    const radialXY = vec3(positionLocal.x, positionLocal.y, float(0.0));
    const rLenSafe = max(float(0.01), length(radialXY));
    const spiralTanRaw = vec3(
      positionLocal.y.negate().div(rLenSafe),
      positionLocal.x.div(rLenSafe),
      float(0.0)
    );
    const spiralTan = normalize(spiralTanRaw.sub(nBase.mul(dot(spiralTanRaw, nBase))).add(vec3(1e-5, 0.0, 0.0)));
    const sutureBevel = sutureCos.mul(sutureCrease).mul(0.40);
    const striaeBevel = cos(thetaU.mul(185.0).add(cos(psiV).mul(2.4)))
      .mul(sutureFlankGate)
      .mul(0.032);
    const nEtched = normalize(nBase.sub(spiralTan.mul(sutureBevel.add(striaeBevel))));

    const nPyrite = perturbNormal3DFbm(
      nEtched,
      positionLocal,
      float(78.0),
      mix(float(0.13), float(0.048), ribCrest)
    );

    ammonite.normalNode = transformNormalToView(nPyrite);
    ammonite.colorNode = finalPyrite;

    const metalRough = clamp(
      float(0.42)
        .sub(ribCrest.mul(0.18))
        .sub(venterMask.mul(0.08))
        .sub(pyriteGlint.mul(0.08))
        .add(sutureCrease.mul(0.24))
        .add(crystalMed.sub(0.5).mul(0.07)),
      0.15,
      0.62
    );
    ammonite.roughnessNode = mix(metalRough, float(0.90), dustFill);
    ammonite.metalnessNode = mix(
      clamp(
        float(0.80)
          .add(ribCrest.mul(0.13))
          .add(pyriteGlint.mul(0.04))
          .sub(sutureCrease.mul(0.18)),
        0.58,
        0.96
      ),
      float(0.04),
      dustFill
    );
  }

  // -------------------------------------------------------------------------
  // 3. Machined Satin-Brass Collet & True Cylindrical Diamond-Knurled Grip
  // -------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.42,
  });
  {
    const zLoc = positionLocal.z;
    const radXY = sqrt(positionLocal.x.mul(positionLocal.x).add(positionLocal.y.mul(positionLocal.y)));
    const safeX = select(abs(positionLocal.x).lessThan(1e-5), float(1e-5), positionLocal.x);
    const azim = atan(positionLocal.y, safeX);

    // Anisotropic circumferential lathe-turning micro-grain (ZERO muddy cow-spot FBM blotches!)
    const latheFine = sin(zLoc.mul(760.0)).mul(0.5).add(0.5);
    const latheBand = mineralToothFbm(vec3(zLoc.mul(52.0), float(1.7), float(3.1)), float(1.8));
    const microPatina = mineralToothFbm(positionLocal, float(190.0));

    // Machined patina deepening inside recessed collet neck, chamfer grooves & rear crimped ferrule flutes
    const neckGroove = smoothstep(float(0.175), float(0.188), zLoc).mul(
      float(1.0).sub(smoothstep(float(0.212), float(0.226), zLoc))
    );
    const foreChamfer = exp(zLoc.sub(0.252).mul(zLoc.sub(0.252)).mul(-18000.0));
    const aftChamfer = exp(zLoc.sub(0.524).mul(zLoc.sub(0.524)).mul(-18000.0));
    const crimpGroove1 = exp(zLoc.sub(1.036).mul(zLoc.sub(1.036)).mul(-22000.0));
    const crimpGroove2 = exp(zLoc.sub(1.078).mul(zLoc.sub(1.078)).mul(-22000.0));
    const grooveShadow = clamp(
      neckGroove
        .mul(0.45)
        .add(foreChamfer.mul(0.32))
        .add(aftChamfer.mul(0.32))
        .add(crimpGroove1.mul(0.52))
        .add(crimpGroove2.mul(0.52)),
      0.0,
      0.62
    );

    // True Cylindrical Helical 45-deg Diamond Knurling on ergonomic grip barrel (z in [0.264, 0.514])
    // Exact integer tooth count N = 32 guarantees 100% seamless 360-deg wrap around atan2(y, x)!
    const knurlZone = smoothstep(float(0.262), float(0.272), zLoc).mul(
      float(1.0).sub(smoothstep(float(0.506), float(0.516), zLoc))
    );
    const p1 = zLoc.mul(340.0).add(azim.mul(32.0));
    const p2 = zLoc.mul(340.0).sub(azim.mul(32.0));
    const c1 = cos(p1).mul(0.5).add(0.5);
    const c2 = cos(p2).mul(0.5).add(0.5);
    const s1 = sin(p1).mul(-0.5);
    const s2 = sin(p2).mul(-0.5);
    // Crisp 4-sided raised diamond pyramid peaks & deep patinated valleys
    const diamondPeak = pow(clamp(c1.mul(c2), 0.0, 1.0), 0.78);
    const dkDz = s1.mul(c2).add(c1.mul(s2));
    const dkDTheta = s1.mul(c2).sub(c1.mul(s2));

    // Exact cylindrical tangent basis (T_z = +Z, T_theta = (-y, x, 0)/r) for analytical knurl facet normals
    const invR = float(1.0).div(max(float(0.005), radXY));
    const tTheta = vec3(positionLocal.y.negate().mul(invR), positionLocal.x.mul(invR), float(0.0));
    const tAxis = vec3(0.0, 0.0, 1.0);
    const nRaw = normalize(normalLocal);
    const latheNormalDelta = cos(zLoc.mul(760.0)).mul(0.018).mul(float(1.0).sub(knurlZone));
    const nKnurled = normalize(
      nRaw
        .sub(tAxis.mul(dkDz.mul(0.46).mul(knurlZone).add(latheNormalDelta)))
        .sub(tTheta.mul(dkDTheta.mul(0.46).mul(knurlZone)))
    );

    // Rich machined satin-brass palette derived strictly from canonical pigments (Yellow Ochre + Bone Calcite + Raw Umber)
    const brassMid = mix(pOchre.rInf.mul(0.96), pCalcite.rInf.mul(0.84), float(0.28));
    const brassBright = mix(pOchre.rInf.mul(1.04), pCalcite.rInf.mul(0.96), float(0.42));
    const brassValley = mix(pUmber.rInf.mul(0.52), pOchre.rInf.mul(0.58), float(0.38));

    const turnedBrass = mix(
      brassMid,
      brassBright,
      clamp(latheBand.mul(0.55).add(latheFine.mul(0.25)).add(microPatina.mul(0.20)), 0.0, 1.0)
    );
    const withGrooves = mix(turnedBrass, brassValley, grooveShadow);
    const knurledColor = mix(brassValley.mul(0.86), brassBright.mul(1.04), diamondPeak);

    satinBrass.colorNode = mix(withGrooves, knurledColor, knurlZone);
    satinBrass.normalNode = transformNormalToView(nKnurled);
    satinBrass.roughnessNode = clamp(
      mix(
        float(0.19).add(latheBand.mul(0.05)).add(grooveShadow.mul(0.18)),
        mix(float(0.42), float(0.15), diamondPeak),
        knurlZone
      ),
      0.14,
      0.46
    );
  }

  // -------------------------------------------------------------------------
  // 4. Oil-Blackened Forged-Iron Sleeve, Sintered Carbide Tip & Braided Hose
  //    (100% derived from canonical mineral pigments in src/core/palette.ts!)
  // -------------------------------------------------------------------------
  const darkIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.84,
    roughness: 0.34,
    envMapIntensity: 1.32,
  });
  {
    const zLoc = positionLocal.z;
    const radXY = sqrt(positionLocal.x.mul(positionLocal.x).add(positionLocal.y.mul(positionLocal.y)));
    const latheMicro = sin(zLoc.mul(620.0)).mul(0.5).add(0.5);
    const grain = mineralToothFbm(positionLocal, float(95.0));
    // Recessed valley darkening inside the 4-rib accordion strain-relief boot (z > 1.12)
    const bootMask = smoothstep(float(1.116), float(1.126), zLoc);
    const bootValley = smoothstep(float(0.0218), float(0.0188), radXY).mul(bootMask);
    // Oil-blackened Prussian-blued forged steel derived strictly from Raw Umber + Azurite Prussian + Bone Calcite
    const ironDeep = mix(pUmber.rInf, pAzurite.rInf, float(0.55)).mul(0.32);
    const ironSheen = mix(
      mix(pUmber.rInf, pAzurite.rInf, float(0.48)).mul(0.58),
      pCalcite.rInf.mul(0.24),
      float(0.32)
    );
    const ironBase = mix(ironDeep, ironSheen, clamp(grain.mul(0.82).add(latheMicro.mul(0.14)), 0.0, 1.0));
    darkIron.colorNode = mix(ironBase, ironDeep.mul(0.42), bootValley.mul(0.68));
    darkIron.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(125.0), float(0.018))
    );
    darkIron.roughnessNode = clamp(
      float(0.28).add(grain.mul(0.10)).add(bootMask.mul(0.14)).add(bootValley.mul(0.24)),
      0.22,
      0.72
    );
    darkIron.metalnessNode = mix(float(0.84), float(0.36), bootMask);
  }

  const carbideTip = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.22,
    envMapIntensity: 1.40,
  });
  {
    const zLoc = positionLocal.z;
    const radXY = sqrt(positionLocal.x.mul(positionLocal.x).add(positionLocal.y.mul(positionLocal.y)));
    const latheSheen = mineralToothFbm(vec3(zLoc.mul(90.0), float(0.8), float(2.1)), float(2.2));
    // Sintered tungsten-carbide gunmetal derived strictly from Bone Calcite + Azurite + Raw Umber
    const carbideDark = mix(pUmber.rInf, pAzurite.rInf, float(0.45)).mul(0.62);
    const carbideLight = mix(pCalcite.rInf, pAzurite.rInf, float(0.22)).mul(0.34);
    const tipMetal = mix(carbideDark, carbideLight, latheSheen);
    // Dark recessed micro-bore orifice at the carbide tip exit (z < 0.0025, r < 0.0044)
    const boreMask = float(1.0)
      .sub(smoothstep(float(0.0026), float(0.0048), radXY))
      .mul(float(1.0).sub(smoothstep(float(0.0008), float(0.0032), zLoc)));
    const boreDark = pUmber.rInf.mul(0.04);

    carbideTip.colorNode = mix(tipMetal, boreDark, boreMask);
    carbideTip.roughnessNode = mix(clamp(float(0.18).add(latheSheen.mul(0.09)), 0.14, 0.30), float(0.88), boreMask);
    carbideTip.metalnessNode = mix(float(0.90), float(0.10), boreMask);
  }

  // Genuine 2-over-2 Interlocking Woven Braided Stainless-Steel & Bronze-Hemp Overbraid Mesh Sheath
  const hoseRubber = new THREE.MeshStandardNodeMaterial({
    metalness: 0.78,
    roughness: 0.34,
    envMapIntensity: 1.48,
  });
  {
    const vHoseTan = attribute('aHoseTangent', 'vec3').toVarying('vHoseTan');
    const vHoseCirc = attribute('aHoseCircum', 'vec3').toVarying('vHoseCirc');

    // True physical arc-length s = uv().x and seamless seam-duplicated thetaNorm = uv().y ∈ [0, 1]
    const sArc = uv().x;
    const thetaNorm = uv().y;

    // 8 helical carriers in each handedness (divisible by 4 -> exact seamless 360-deg 2-over-2 wrap!)
    // +64.0 integer multiple of 4 guarantees strictly positive fract() inputs on Apple Silicon Metal
    const uA = sArc.mul(76.0).add(thetaNorm.mul(8.0)).add(64.0);
    const uB = sArc.mul(76.0).sub(thetaNorm.mul(8.0)).add(64.0);

    const fA = fract(uA).sub(0.5);
    const fB = fract(uB).sub(0.5);

    // Crisp carrier ribbon masks leaving genuine dark diamond mesh pores between woven bundles
    const wA = smoothstep(float(0.455), float(0.355), abs(fA));
    const wB = smoothstep(float(0.455), float(0.355), abs(fB));

    // Rounded cylindrical ribbon cross-section arch
    const nfA = clamp(fA.div(0.44), -1.0, 1.0);
    const nfB = clamp(fB.div(0.44), -1.0, 1.0);
    const crownA = sqrt(max(float(0.0), float(1.0).sub(nfA.mul(nfA))));
    const crownB = sqrt(max(float(0.0), float(1.0).sub(nfB.mul(nfB))));

    // 4 tightly-packed parallel metallic wire filaments per woven carrier bundle
    const wirePhaseA = fA.mul(4.0).mul(TWO_PI);
    const wirePhaseB = fB.mul(4.0).mul(TWO_PI);
    const wireA = cos(wirePhaseA).mul(0.5).add(0.5);
    const wireB = cos(wirePhaseB).mul(0.5).add(0.5);

    // Exact 2-over-2 interlocking twill/basket height wave (exactly out-of-phase at every crossing!)
    const halfPi = PI.mul(0.5);
    const phaseA = uB.add(floor(uA)).mul(halfPi);
    const phaseB = uA.add(floor(uB)).mul(halfPi);
    const waveA = cos(phaseA);
    const waveB = cos(phaseB).negate();

    const zA = wA.mul(float(0.52).add(waveA.mul(0.32)).add(crownA.mul(0.14)).add(wireA.mul(0.06)));
    const zB = wB.mul(float(0.52).add(waveB.mul(0.32)).add(crownB.mul(0.14)).add(wireB.mul(0.06)));

    const aOverB = smoothstep(float(-0.032), float(0.032), zA.sub(zB));
    const strandCov = max(wA, wB);
    const activeCrown = mix(crownB, crownA, aOverB);
    const activeWave = mix(waveB, waveA, aOverB);
    const activeWire = mix(wireB, wireA, aOverB);

    // Deep ambient occlusion in weave recesses & right where under-strands duck beneath crossing crowns
    const bothMask = wA.mul(wB);
    const underLight = smoothstep(float(-0.78), float(0.58), activeWave);
    const overlapLipAO = float(1.0).sub(
      bothMask
        .mul(float(1.0).sub(smoothstep(float(0.0), float(0.22), abs(zA.sub(zB)))))
        .mul(0.56)
    );
    const weaveAO = clamp(
      float(0.28).add(underLight.mul(0.52)).add(activeCrown.mul(0.20)),
      0.16,
      1.0
    ).mul(overlapLipAO);

    // Analytical tangent-space micro-normals along exact Frenet hose basis (zero dpdx/dpdy aliasing!)
    const n0 = normalize(normalLocal);
    const rawTan = vec3(vHoseTan);
    const rawCirc = vec3(vHoseCirc);
    const tS = normalize(rawTan.sub(n0.mul(dot(rawTan, n0))).add(vec3(1e-5, 0.0, 0.0)));
    const tC = normalize(rawCirc.sub(n0.mul(dot(rawCirc, n0))).add(vec3(0.0, 1e-5, 0.0)));
    const dirA = normalize(tS.add(tC));
    const dirB = normalize(tS.sub(tC));

    const dHA_dA = fA.mul(-2.1).sub(sin(wirePhaseA).mul(0.62));
    const dHA_dB = sin(phaseA).mul(-0.58);
    const dHB_dB = fB.mul(-2.1).sub(sin(wirePhaseB).mul(0.62));
    const dHB_dA = sin(phaseB).mul(0.58);

    const slopeA = mix(dHB_dA, dHA_dA, aOverB).mul(strandCov);
    const slopeB = mix(dHB_dB, dHA_dB, aOverB).mul(strandCov);
    const nBraid = normalize(
      n0.sub(dirA.mul(slopeA.mul(0.38))).sub(dirB.mul(slopeB.mul(0.38)))
    );

    // Subtle handling wear & bend-zone oil patina near the rear strain-relief collar
    const wearNoise = mineralToothFbm(positionLocal, float(26.0));
    const ferruleWear = float(0.76).add(smoothstep(float(0.0), float(0.16), sArc).mul(0.24));
    const patinaTone = float(0.86).add(wearNoise.mul(0.22)).mul(ferruleWear);

    // Canonical natural mineral pigment metallic overbraid:
    // Woven satin stainless-steel carriers interleaved with subtle warm bronze-hemp tonal anisotropy
    // over a deep oil-blackened inner pressure-tube core peeking through diamond interstices
    const steelCarrier = mix(pCalcite.rInf, pAzurite.rInf, float(0.16)).mul(0.58);
    const bronzeCarrier = mix(
      mix(pKaolin.rInf, pOchre.rInf, float(0.28)).mul(0.46),
      pUmber.rInf.mul(0.40),
      float(0.22)
    );
    const carrierBase = mix(bronzeCarrier, steelCarrier, aOverB);
    const wireHighlight = mix(
      carrierBase.mul(0.74),
      carrierBase.mul(1.22),
      clamp(activeWire.mul(0.72).add(activeCrown.mul(0.28)), 0.0, 1.0)
    );
    const innerCore = mix(pUmber.rInf, pAzurite.rInf, float(0.35)).mul(0.055);

    hoseRubber.colorNode = mix(innerCore, wireHighlight.mul(weaveAO).mul(patinaTone), strandCov);
    hoseRubber.normalNode = transformNormalToView(nBraid);
    hoseRubber.metalnessNode = mix(
      float(0.06),
      clamp(float(0.74).add(activeWire.mul(0.14)).add(activeCrown.mul(0.08)), 0.62, 0.94),
      strandCov
    );
    hoseRubber.roughnessNode = mix(
      float(0.90),
      clamp(
        float(0.35)
          .sub(activeWire.mul(0.13))
          .sub(activeCrown.mul(0.06))
          .add(float(1.0).sub(underLight).mul(0.16)),
        0.16,
        0.58
      ),
      strandCov
    );
  }

  // -------------------------------------------------------------------------
  // 5. Focused High-Velocity Dolomite Abrasive Grit Stream Cone
  // -------------------------------------------------------------------------
  const abrasiveJet = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  {
    // uv.y = 1 at carbide orifice (z = 0), uv.y = 0 at rock impact point (z = -jetLength)
    const vCoord = uv().y;
    const uCoord = uv().x;

    // Rapid downward streaming dolomite-grit filaments flowing from orifice (1) -> impact (0)
    const stream1 = mineralToothFbm(
      vec3(uCoord.mul(10.0), vCoord.mul(3.6).add(uTime.mul(26.0)), float(1.7)),
      float(2.2)
    );
    const stream2 = mineralToothFbm(
      vec3(uCoord.mul(18.0).add(3.1), vCoord.mul(6.8).add(uTime.mul(38.0)), float(4.3)),
      float(2.4)
    );
    const filaments = smoothstep(
      float(0.24),
      float(0.80),
      stream1.mul(0.56).add(stream2.mul(0.44))
    );

    // Crisp core right out of carbide orifice, gentle turbulent flare right onto rock impact point
    const endFade = smoothstep(float(0.0), float(0.10), vCoord).mul(
      smoothstep(float(1.0), float(0.90), vCoord)
    );
    const coreDensity = mix(float(0.55), float(0.94), vCoord);

    abrasiveJet.colorNode = mix(
      pKaolin.rInf.mul(1.10),
      pCalcite.rInf.mul(1.22),
      filaments.mul(0.62).add(vCoord.mul(0.38))
    );
    abrasiveJet.opacityNode = clamp(
      endFade.mul(coreDensity).mul(float(0.48).add(filaments.mul(0.92))).mul(uJetAlpha),
      0.0,
      0.84
    );
  }

  // -------------------------------------------------------------------------
  // 6. Rebounding Grit Sparks & Limestone Spall Chips (Per-Instance Mineral Attribute)
  // -------------------------------------------------------------------------
  const gritChips = new THREE.MeshStandardNodeMaterial({
    metalness: 0.1,
    roughness: 0.72,
    envMapIntensity: 1.32,
  });
  {
    const aChipMeta = attribute('aChipMeta', 'vec4'); // [r, g, b, metallicSpark]
    gritChips.colorNode = aChipMeta.rgb;
    gritChips.metalnessNode = clamp(aChipMeta.w, 0.04, 0.94);
    gritChips.roughnessNode = mix(float(0.84), float(0.18), clamp(aChipMeta.w, 0.0, 1.0));
  }

  // -------------------------------------------------------------------------
  // 7. Billowing Calcite Dust Plume Billboards (Soft FBM Mineral Powder Cloud)
  // -------------------------------------------------------------------------
  const dustPlume = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  {
    const aDustMeta = attribute('aDustMeta', 'vec4'); // [alphaEnv, seed, warmth, ageNorm]
    const alphaEnv = aDustMeta.x;
    const puffSeed = aDustMeta.y;
    const warmth = aDustMeta.z;
    const ageNorm = aDustMeta.w;

    const centered = uv().sub(0.5).mul(2.0);
    const rad = length(centered);

    // Seeded rotation + slow expansion drift so adjacent dust puffs never look stamped
    const ang = puffSeed.mul(TWO_PI).add(ageNorm.mul(0.65));
    const ca = cos(ang);
    const sa = sin(ang);
    const rotUv = vec2(
      centered.x.mul(ca).sub(centered.y.mul(sa)),
      centered.x.mul(sa).add(centered.y.mul(ca))
    );

    const billow = mineralToothFbm(
      vec3(rotUv.mul(1.85), puffSeed.mul(13.7).add(ageNorm.mul(0.8))),
      float(2.4)
    );
    const finePowder = mineralToothFbm(
      vec3(rotUv.mul(4.2), puffSeed.mul(27.3)),
      float(2.6)
    );

    const cloudRadius = rad.add(billow.sub(0.5).mul(0.52)).add(finePowder.sub(0.5).mul(0.18));
    const softEdge = exp(rad.mul(rad).mul(-3.0)).mul(
      smoothstep(float(0.98), float(0.22), cloudRadius)
    );

    // Warm overhead softbox shading across each dust puff (top brighter Bone Calcite, lower soft Kaolin)
    const topLight = clamp(centered.y.mul(0.32).add(0.68), 0.42, 1.0);
    const baseDust = mix(
      pKaolin.rInf.mul(0.88),
      pCalcite.rInf.mul(1.04),
      clamp(billow.mul(0.55).add(topLight.mul(0.45)), 0.0, 1.0)
    );
    const warmDust = mix(baseDust, pOchre.rInf.mul(0.92), warmth.mul(0.18));

    dustPlume.colorNode = warmDust;
    dustPlume.opacityNode = clamp(softEdge.mul(alphaEnv), 0.0, 0.46);
  }

  return {
    limestone,
    ammonite,
    satinBrass,
    darkIron,
    carbideTip,
    hoseRubber,
    abrasiveJet,
    gritChips,
    dustPlume,
    uStrataAxis,
    uStrataSeed,
    uBlastSpotWorld,
    uJetAlpha,
    uTime,
    dispose: () => {
      limestone.dispose();
      ammonite.dispose();
      satinBrass.dispose();
      darkIron.dispose();
      carbideTip.dispose();
      hoseRubber.dispose();
      abrasiveJet.dispose();
      gritChips.dispose();
      dustPlume.dispose();
    },
  };
}
