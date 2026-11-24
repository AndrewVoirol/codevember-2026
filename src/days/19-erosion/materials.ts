import * as THREE from 'three/webgpu';
import {
  Fn,
  TWO_PI,
  abs,
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
  positionGeometry,
  positionLocal,
  positionWorld,
  pow,
  select,
  sin,
  smoothstep,
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
import { evalBaseBoulderRadiusTsl, octDirToUvTsl } from './octahedral';

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
 * so pyritized bronze-gold ammonite ribs, satin brass nozzle, and dark forged iron gleam
 * richly against the #212529 Chalky Slate backdrop without dark-void collapse.
 */
export function createStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0b0e13);

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
  addSoftbox(11.5, 9.2, [-4.8, 5.4, 4.2], 1.0, 0.96, 0.89, 1.92);
  // Tall warm-neutral right studio scrim (catches pyrite rib crests & brass nozzle barrel)
  addSoftbox(8.8, 10.5, [6.0, 2.8, 3.4], 0.99, 0.95, 0.88, 1.78);
  // Front-upper gentle fill diffuser (prevents metallic ribs facing camera from collapsing black)
  addSoftbox(8.0, 6.2, [0.4, 2.2, 6.8], 0.98, 0.95, 0.91, 1.15);
  // Cool slate rear-top grazing rim scrim (carves clean metallic separation from #212529)
  addSoftbox(12.2, 5.4, [-1.2, 5.2, -5.8], 0.88, 0.94, 1.0, 1.52);
  // Rear-right secondary soft fill (for back-face 360-deg orbit blasting)
  addSoftbox(9.5, 8.0, [4.8, 1.8, -5.6], 0.94, 0.94, 0.96, 1.32);
  // Subtle warm specimen-bench floor bounce
  addSoftbox(12.0, 12.0, [0, -6.2, 0.8], 0.96, 0.92, 0.86, 0.42);

  const renderTarget = pmrem.fromScene(envScene, 0.084);
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
 * `burntSienna`, `hematiteRust`, `azuritePrussian`, `realgarAmber`).
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
  const pAmber = createPigmentUniforms('realgarAmber');

  // -------------------------------------------------------------------------
  // 1. Carvable Stratified Limestone Matrix Nodule (GPU Heightfield + Exact Sphere Normals)
  // -------------------------------------------------------------------------
  const limestone = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.86,
    envMapIntensity: 0.62,
  });
  {
    const unitDirVert = normalize(positionGeometry);
    const vUnitDir = unitDirVert.toVarying('vNoduleUnitDir');

    // Vertex displacement: exact C-infinity analytical boulder hull + isotropic octahedral deltaR
    // Eliminates 100% of bilinear V-trough fold creases along y = 0 / x = 0!
    const octUvVert = octDirToUvTsl(unitDirVert);
    const hSampleVert = texture(erosionTex, octUvVert).level(float(0.0));
    const rVert = evalBaseBoulderRadiusTsl(unitDirVert).add(hSampleVert.r);
    limestone.positionNode = unitDirVert.mul(rVert);

    // Fragment evaluation (strictly uniform control flow for every texture tap)
    const d = normalize(vec3(vUnitDir));
    const octUvFrag = octDirToUvTsl(d);
    const hSample = texture(erosionTex, octUvFrag).level(float(0.0));
    const rCenter = evalBaseBoulderRadiusTsl(d).add(hSample.r);
    const cutDepth = max(float(0.0), hSample.g);
    const freshChalk = clamp(hSample.b, 0.0, 1.0);
    const fossilClearance = hSample.a;

    // Exact basis-invariant spherical heightfield normal: N = normalize(r*d - dr/dt * T - dr/db * B)
    const upHint = select(abs(d.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const T = normalize(cross(upHint, d));
    const B = normalize(cross(d, T));
    const eps = float(0.0115);

    const dTp = normalize(d.add(T.mul(eps)));
    const dTm = normalize(d.sub(T.mul(eps)));
    const dBp = normalize(d.add(B.mul(eps)));
    const dBm = normalize(d.sub(B.mul(eps)));

    const rTp = evalBaseBoulderRadiusTsl(dTp).add(texture(erosionTex, octDirToUvTsl(dTp)).level(float(0.0)).r);
    const rTm = evalBaseBoulderRadiusTsl(dTm).add(texture(erosionTex, octDirToUvTsl(dTm)).level(float(0.0)).r);
    const rBp = evalBaseBoulderRadiusTsl(dBp).add(texture(erosionTex, octDirToUvTsl(dBp)).level(float(0.0)).r);
    const rBm = evalBaseBoulderRadiusTsl(dBm).add(texture(erosionTex, octDirToUvTsl(dBm)).level(float(0.0)).r);

    const drdt = rTp.sub(rTm).div(eps.mul(2.0));
    const drdb = rBp.sub(rBm).div(eps.mul(2.0));
    const nMacro = normalize(d.mul(rCenter).sub(T.mul(drdt)).sub(B.mul(drdb)));

    // 3D Object-Space Dipping Sedimentary Limestone Strata (crisp chiseled beds on BOTH exterior & fresh cuts!)
    const pObj = d.mul(rCenter);
    const seedShift = vec3(uStrataSeed.mul(1.73), uStrataSeed.mul(0.91), uStrataSeed.mul(-1.37));
    const pSample = pObj.add(seedShift);

    const macroWarp = mineralToothFbm(pSample, float(2.4));
    const chiseledFacet = mineralToothFbm(pSample.add(vec3(4.1, 2.3, 6.7)), float(15.5));
    const fineOolite = mineralToothFbm(pSample.add(vec3(9.2, 5.8, 1.4)), float(48.0));

    // Distinct layered Jurassic limestone/marl sedimentary bedding planes
    const strataCoord = dot(pObj, uStrataAxis)
      .mul(21.0)
      .add(macroWarp.mul(1.85))
      .add(chiseledFacet.mul(0.32));
    const bedPrimary = sin(strataCoord).mul(0.5).add(0.5);
    const bedFine = sin(strataCoord.mul(3.15).add(1.15)).mul(0.5).add(0.5);

    // Crisp geological band masks (distinct limestone beds, ochre marl seams, thin dark shale partings)
    const calciteBed = smoothstep(float(0.34), float(0.68), bedPrimary);
    const ochreBed = smoothstep(float(0.24), float(0.58), bedFine).mul(
      float(1.0).sub(smoothstep(float(0.62), float(0.88), bedPrimary))
    );
    const ironSeam = pow(clamp(sin(strataCoord.mul(1.48).add(2.2)).mul(0.5).add(0.5), 0.0, 1.0), 6.0);
    const shaleParting = pow(clamp(float(1.0).sub(bedPrimary), 0.0, 1.0), 6.5);

    // 2-Flux Kubelka-Munk stratified calcareous marl + ochre/sienna ironstone bands
    const rawSubstrate = mix(
      pUmber.rInf.mul(0.58),
      pSienna.rInf.mul(0.64),
      ironSeam.mul(0.82)
    );
    const ochreGlaze = kubelkaMunkGlaze(
      pOchre.K,
      pOchre.S,
      float(0.38).add(ochreBed.mul(0.75)).add(chiseledFacet.mul(0.22)),
      rawSubstrate
    );
    const calciteLayer = kubelkaMunkGlaze(
      pCalcite.K,
      pCalcite.S,
      float(0.52).add(calciteBed.mul(1.15)).add(fineOolite.mul(0.24)),
      ochreGlaze
    );

    // Rich stratified rock petrography (crisp Bone Calcite, warm Yellow Ochre, Burnt Sienna & Raw Umber)
    const layeredStone = mix(
      mix(calciteLayer, pOchre.rInf.mul(0.86), ochreBed.mul(0.48)),
      pUmber.rInf.mul(0.38),
      shaleParting.mul(0.72)
    );
    const withIronVeins = mix(
      layeredStone,
      pSienna.rInf.mul(0.72),
      ironSeam.mul(0.54)
    ).mul(float(0.86).add(fineOolite.mul(0.24)));

    // Subtle exterior mineral aging vs. luminous freshly-abraded interior strata
    // Exterior preserves 100% of the crisp sedimentary bedding planes (zero blurry grey-brown camo!)
    const freshCutMask = smoothstep(float(0.004), float(0.032), cutDepth);
    const agedExterior = mix(
      withIronVeins.mul(0.78),
      pUmber.rInf.mul(0.42),
      shaleParting.mul(0.34).add(smoothstep(float(0.70), float(0.92), chiseledFacet).mul(0.14))
    );
    const freshInterior = withIronVeins.mul(1.06);
    const rockRgb = mix(agedExterior, freshInterior, freshCutMask);

    // Powdery white crushed Bone-Calcite blast dust blooming inside fresh craters & around fracture rims
    const chalkLift = clamp(freshChalk.mul(float(0.58).add(fineOolite.mul(0.42))), 0.0, 0.88);
    const withChalk = mix(rockRgb, pCalcite.rInf.mul(1.04), chalkLift.mul(0.56));

    // Contact AO + bright crushed calcite collar right where limestone meets the emerging pyritized ammonite
    const collarSeam = exp(fossilClearance.mul(fossilClearance).mul(-9.5));
    const deepCreviceAO = smoothstep(float(-0.45), float(0.25), fossilClearance).mul(0.26);
    const withCollar = mix(
      withChalk.mul(float(1.0).sub(deepCreviceAO)),
      pCalcite.rInf.mul(0.98),
      collarSeam.mul(0.24)
    );

    // Subtle live abrasive spot dust flash right under active blast stream
    const dBlast = length(positionWorld.sub(uBlastSpotWorld.xyz));
    const blastSpot = exp(dBlast.mul(dBlast).mul(-160.0)).mul(uBlastSpotWorld.w);
    const finalColor = mix(withCollar, pCalcite.rInf.mul(1.08), blastSpot.mul(0.35));

    // Micro-normal: combine exact carved heightfield normal + crisp chiseled fracture planes + stepped bedding ledges
    const nCoarse = perturbNormal3DFbm(nMacro, pSample, float(18.0), float(0.38));
    const nFine = perturbNormal3DFbm(nCoarse, pSample.add(vec3(5.4, 8.2, 3.1)), float(64.0), float(0.20));
    // Crisp sedimentary bedding ledge bevel along uStrataAxis (catches overhead studio softbox sharply!)
    const bedSlope = cos(strataCoord)
      .mul(0.22)
      .add(cos(strataCoord.mul(3.15).add(1.15)).mul(0.08))
      .mul(float(0.68).add(freshCutMask.mul(0.38)));
    const strataTan = uStrataAxis.sub(nFine.mul(dot(uStrataAxis, nFine)));
    const nStratified = normalize(nFine.sub(strataTan.mul(bedSlope)));

    limestone.normalNode = transformNormalToView(nStratified);
    limestone.colorNode = finalColor;
    limestone.roughnessNode = clamp(
      float(0.84)
        .add(fineOolite.sub(0.5).mul(0.14))
        .add(chalkLift.mul(0.10))
        .add(shaleParting.mul(0.08))
        .sub(ironSeam.mul(0.12)),
      0.62,
      0.96
    );
  }

  // -------------------------------------------------------------------------
  // 2. Embedded Pyritized Ammonite Relic (Gleaming Dark-Bronze & Golden Pyrite FeS2)
  // -------------------------------------------------------------------------
  const ammonite = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.30,
    envMapIntensity: 1.38,
  });
  {
    const aMeta = attribute('aAmmoniteMeta', 'vec4'); // [ribCrest, whorlNorm, sutureZone, venterMask]
    const ribCrest = clamp(aMeta.x, 0.0, 1.0);
    const whorlNorm = clamp(aMeta.y, 0.0, 1.0);
    const sutureZone = clamp(aMeta.z, 0.0, 1.0);
    const venterMask = clamp(aMeta.w, 0.0, 1.0);

    // Sample live erosion heightfield to dust newly-exposed inter-rib crevices with receding limestone chalk!
    const dirAm = normalize(positionLocal);
    const hAm = texture(erosionTex, octDirToUvTsl(dirAm)).level(float(0.0));
    const clearance = hAm.a; // <0 buried, 0..1 clearing from freshly exposed to brushed clean
    const residualChalkMask = smoothstep(float(0.82), float(0.04), clearance)
      .mul(float(1.0).sub(ribCrest.mul(0.82)))
      .mul(float(1.0).sub(venterMask.mul(0.55)));

    // Crystalline cubic pyrite micro-grain & subtle mineral tarnish
    const crystalMed = mineralToothFbm(positionLocal, float(26.0));
    const crystalFine = mineralToothFbm(positionLocal.add(vec3(3.7, 7.1, 2.9)), float(88.0));
    const pyriteGlint = smoothstep(float(0.68), float(0.92), crystalFine);

    // Authentic pyritized Jurassic ammonite metallic reflectance:
    // Deep oxidized bronze-umber in valleys -> rich golden-ochre / realgar-brass FeS2 on burnished rib crests
    const darkBronzeValley = mix(pUmber.rInf.mul(0.32), pHematite.rInf.mul(0.28), crystalMed.mul(0.5));
    const warmPyriteMid = mix(pOchre.rInf.mul(0.76), pAmber.rInf.mul(0.52), float(0.28).add(whorlNorm.mul(0.16)));
    const brightPyriteCrest = mix(pOchre.rInf.mul(1.05), pCalcite.rInf.mul(0.88), float(0.32).add(pyriteGlint.mul(0.28)));

    const ribShaded = mix(
      darkBronzeValley,
      mix(warmPyriteMid, brightPyriteCrest, smoothstep(float(0.32), float(0.92), ribCrest)),
      smoothstep(float(0.06), float(0.72), ribCrest.add(venterMask.mul(0.35)))
    );

    // Subtle phylloceratic septal suture etching & restrained natural bornite/azurite mineral tarnish at grazing angles
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const ndotv = clamp(dot(normalize(normalWorld), viewDir), 0.0, 1.0);
    const grazingTarnish = pow(float(1.0).sub(ndotv), 2.8)
      .mul(smoothstep(float(0.38), float(0.78), crystalMed))
      .mul(0.18);

    const withSuture = mix(ribShaded, darkBronzeValley.mul(0.65), sutureZone.mul(0.45));
    const withBornite = mix(withSuture, pAzurite.rInf.mul(0.72), grazingTarnish);

    // Powdery Bone-Calcite matrix dust caught inside inter-rib furrows until fully blasted clear
    const chalkNoise = smoothstep(float(0.24), float(0.76), crystalMed.mul(0.6).add(crystalFine.mul(0.4)));
    const dustFill = clamp(residualChalkMask.mul(float(0.48).add(chalkNoise.mul(0.52))), 0.0, 0.86);
    const finalPyrite = mix(withBornite, pCalcite.rInf.mul(0.88), dustFill);

    // Micro-crystalline pyrite normal perturbation (sleek burnished rib crests, crystalline micro-glint in valleys)
    const nPyrite = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(72.0),
      mix(float(0.14), float(0.055), ribCrest)
    );
    ammonite.normalNode = transformNormalToView(nPyrite);
    ammonite.colorNode = finalPyrite;

    const metalRough = clamp(
      float(0.44)
        .sub(ribCrest.mul(0.18))
        .sub(venterMask.mul(0.08))
        .sub(pyriteGlint.mul(0.08))
        .add(crystalMed.sub(0.5).mul(0.08)),
      0.16,
      0.56
    );
    ammonite.roughnessNode = mix(metalRough, float(0.90), dustFill);
    ammonite.metalnessNode = mix(
      clamp(float(0.78).add(ribCrest.mul(0.14)).add(pyriteGlint.mul(0.05)), 0.72, 0.96),
      float(0.04),
      dustFill
    );
  }

  // -------------------------------------------------------------------------
  // 3. Articulated Machined Satin-Brass Grip & Collet (Air-Abrasion Pencil)
  // -------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.91,
    roughness: 0.24,
    envMapIntensity: 1.32,
  });
  {
    const brush = mineralToothFbm(positionLocal, float(38.0));
    const knurlZone = smoothstep(float(0.25), float(0.275), positionLocal.z).mul(
      float(1.0).sub(smoothstep(float(0.515), float(0.54), positionLocal.z))
    );
    // Diamond micro-knurl along the ergonomic brass finger grip
    const ang = positionLocal.x.mul(110.0).add(positionLocal.y.mul(110.0));
    const knurl = sin(positionLocal.z.mul(220.0).add(ang))
      .mul(sin(positionLocal.z.mul(220.0).sub(ang)))
      .mul(0.5)
      .add(0.5);

    const brassDark = mix(pUmber.rInf, pOchre.rInf, 0.52).mul(0.95);
    const brassBright = mix(pOchre.rInf, pCalcite.rInf, 0.36).mul(1.06);
    const baseBrass = mix(brassDark, brassBright, smoothstep(float(0.22), float(0.78), brush));
    satinBrass.colorNode = mix(baseBrass, baseBrass.mul(float(0.78).add(knurl.mul(0.32))), knurlZone);
    satinBrass.normalNode = transformNormalToView(
      perturbNormal3DFbm(
        normalLocal,
        positionLocal,
        float(54.0),
        mix(float(0.016), float(0.095), knurlZone)
      )
    );
    satinBrass.roughnessNode = clamp(
      float(0.21).add(brush.mul(0.08)).add(knurlZone.mul(0.14).mul(float(1.0).sub(knurl))),
      0.16,
      0.44
    );
  }

  // -------------------------------------------------------------------------
  // 4. Oil-Blackened Forged-Iron Pneumatic Sleeve & Sintered Carbide Tip
  // -------------------------------------------------------------------------
  const darkIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.80,
    roughness: 0.38,
    envMapIntensity: 1.24,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(46.0));
    darkIron.colorNode = mix(vec3(0.032, 0.035, 0.042), vec3(0.068, 0.072, 0.082), grain);
    darkIron.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(62.0), float(0.028))
    );
    darkIron.roughnessNode = clamp(float(0.34).add(grain.mul(0.14)), 0.24, 0.52);
  }

  const carbideTip = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.24,
    envMapIntensity: 1.35,
  });
  {
    const sheen = mineralToothFbm(positionLocal, float(70.0));
    carbideTip.colorNode = mix(vec3(0.085, 0.090, 0.100), vec3(0.165, 0.172, 0.185), sheen);
    carbideTip.roughnessNode = clamp(float(0.20).add(sheen.mul(0.10)), 0.16, 0.34);
  }

  const hoseRubber = new THREE.MeshStandardNodeMaterial({
    metalness: 0.08,
    roughness: 0.72,
    envMapIntensity: 0.65,
  });
  {
    const braid = sin(uv().x.mul(130.0).add(uv().y.mul(TWO_PI).mul(6.0)))
      .mul(sin(uv().x.mul(130.0).sub(uv().y.mul(TWO_PI).mul(6.0))))
      .mul(0.5)
      .add(0.5);
    hoseRubber.colorNode = mix(pUmber.rInf.mul(0.16), pUmber.rInf.mul(0.30), braid);
    hoseRubber.roughnessNode = clamp(float(0.64).add(braid.mul(0.16)), 0.55, 0.85);
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
      vec3(uCoord.mul(9.0), vCoord.mul(3.2).add(uTime.mul(22.0)), float(1.7)),
      float(2.2)
    );
    const stream2 = mineralToothFbm(
      vec3(uCoord.mul(16.0).add(3.1), vCoord.mul(6.0).add(uTime.mul(34.0)), float(4.3)),
      float(2.4)
    );
    const filaments = smoothstep(float(0.28), float(0.82), stream1.mul(0.58).add(stream2.mul(0.42)));

    // Soft longitudinal fade right at nozzle lip & rock flare
    const endFade = smoothstep(float(0.0), float(0.18), vCoord).mul(
      smoothstep(float(1.0), float(0.84), vCoord)
    );
    // Concentrate core density tighter near the nozzle orifice, fanning gently toward the rock
    const coreDensity = mix(float(0.32), float(0.62), vCoord);

    abrasiveJet.colorNode = mix(
      pKaolin.rInf.mul(1.06),
      pCalcite.rInf.mul(1.16),
      filaments.mul(0.7).add(vCoord.mul(0.3))
    );
    abrasiveJet.opacityNode = clamp(
      endFade.mul(coreDensity).mul(float(0.35).add(filaments.mul(0.85))).mul(uJetAlpha),
      0.0,
      0.58
    );
  }

  // -------------------------------------------------------------------------
  // 6. Rebounding Grit Sparks & Limestone Spall Chips (Per-Instance Mineral Attribute)
  // -------------------------------------------------------------------------
  const gritChips = new THREE.MeshStandardNodeMaterial({
    metalness: 0.1,
    roughness: 0.72,
    envMapIntensity: 1.25,
  });
  {
    const aChipMeta = attribute('aChipMeta', 'vec4'); // [r, g, b, metallicSpark]
    gritChips.colorNode = aChipMeta.rgb;
    gritChips.metalnessNode = clamp(aChipMeta.w, 0.04, 0.92);
    gritChips.roughnessNode = mix(float(0.82), float(0.22), clamp(aChipMeta.w, 0.0, 1.0));
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
    const softEdge = exp(rad.mul(rad).mul(-3.2)).mul(smoothstep(float(0.98), float(0.24), cloudRadius));

    // Warm overhead softbox shading across each dust puff (top brighter Bone Calcite, lower soft Kaolin/Umber)
    const topLight = clamp(centered.y.mul(0.32).add(0.68), 0.38, 1.0);
    const baseDust = mix(
      pKaolin.rInf.mul(0.88),
      pCalcite.rInf.mul(1.04),
      clamp(billow.mul(0.55).add(topLight.mul(0.45)), 0.0, 1.0)
    );
    const warmDust = mix(baseDust, pOchre.rInf.mul(0.95), warmth.mul(0.22));

    dustPlume.colorNode = warmDust;
    dustPlume.opacityNode = clamp(softEdge.mul(alphaEnv), 0.0, 0.42);
  }

  // Keep fract imported if needed or avoid unused import warning
  void fract;

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
