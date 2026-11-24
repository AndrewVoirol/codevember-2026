import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  faceDirection,
  float,
  frontFacing,
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
  transformNormalToView,
  uniform,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { type CoreMaterialSet } from './coreAssemblies';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface CocoonStudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface CocoonMaterialBundle {
  coreMaterials: CoreMaterialSet;
  filmFrontMat: THREE.MeshPhysicalNodeMaterial;
  filmBackMat: THREE.MeshPhysicalNodeMaterial;
  uniforms: {
    uTime: { value: number };
    uAmberGlow: { value: number };
    uHeatOrigin: { value: THREE.Vector3 };
    uHeatActive: { value: number };
  };
  dispose: () => void;
}

// ============================================================================
// Procedural Feathered Studio Softbox PMREM Environment
// ============================================================================

function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number,
  size = 64
): THREE.DataTexture {
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
 * Generates a studio softbox PMREM environment cubemap calibrated for Day 20:
 * crisp feathered softbox strips illuminate satin brass calipers, honed basalt facets,
 * faceted Bone-Calcite prisms, and glossy vacuum-formed shrink-film highlights cleanly
 * against the Chalky Slate (#212529) room ground.
 */
export function createCocoonStudioEnvironment(
  renderer: THREE.WebGPURenderer
): CocoonStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0b0e12);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: readonly [number, number, number],
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

  // Overhead-right warm calcite key softbox
  addSoftbox(11.0, 9.0, [4.8, 5.6, 4.2], 1.0, 0.96, 0.89, 1.85);
  // Opposite tall cool-neutral scrim (carves razor-crisp metallic bevels & film Fresnel rims)
  addSoftbox(8.8, 10.5, [-5.8, 2.4, 3.0], 0.90, 0.95, 1.0, 1.55);
  // Rear-top grazing calcite rim scrim
  addSoftbox(12.0, 5.2, [-1.2, 5.2, -5.8], 0.96, 0.95, 0.92, 1.50);
  // Zenith soft fill card
  addSoftbox(9.5, 9.5, [0.0, 7.2, 0.5], 0.97, 0.95, 0.91, 1.05);
  // Rear-right cool mineral fill
  addSoftbox(8.2, 6.8, [4.8, 1.4, -5.4], 0.88, 0.94, 1.0, 1.15);
  // Subtle warm underside table bounce
  addSoftbox(11.5, 11.5, [0.0, -6.4, 0.4], 0.98, 0.88, 0.74, 0.36);

  const renderTarget = pmrem.fromScene(envScene, 0.084);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

// ============================================================================
// Analytical 4-Tap Tetrahedral 3D FBM Micro-Normal Perturbation
// ============================================================================

/**
 * Scale-invariant 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Divides by normalized noise-space step (0.64) rather than world-space `4*eps` so
 * high-frequency machined metals and crystals never amplify into crumpled foil.
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

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).div(float(0.64));
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

// ============================================================================
// Material Factory: Inner Sculptural Core + Translucent Shrink-Film Skin
// ============================================================================

export function createCocoonMaterials(): CocoonMaterialBundle {
  const uTime = uniform(0);
  const uAmberGlow = uniform(1.0);
  const uHeatOrigin = uniform(new THREE.Vector3(0, 0, 1.6));
  const uHeatActive = uniform(0);

  const pKaolin = createPigmentUniforms('warmKaolin');
  const pCalcite = createPigmentUniforms('boneCalcite');
  const pRealgar = createPigmentUniforms('realgarAmber');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pVerdigris = createPigmentUniforms('verdigrisMalachite');
  const pAzurite = createPigmentUniforms('azuritePrussian');

  // -------------------------------------------------------------------------
  // 1. Satin Machined Brass (Astrolabe Rings, Caliper Spokes, Collars)
  // -------------------------------------------------------------------------
  const brass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.26,
    envMapIntensity: 1.28,
  });
  {
    const brush = mineralToothFbm(positionLocal, float(28.0));
    const fineSatin = mineralToothFbm(positionLocal.add(vec3(3.1, 1.7, 5.9)), float(68.0));
    const brassWarm = mix(pOchre.rInf, pCalcite.rInf, float(0.24));
    const brassPatina = mix(pOchre.rInf, pUmber.rInf, float(0.54));
    const patinaMix = smoothstep(float(0.24), float(0.78), brush.mul(0.62).add(fineSatin.mul(0.38)));

    brass.colorNode = mix(brassPatina, brassWarm, patinaMix);
    brass.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(40.0), float(0.015))
    );
    brass.roughnessNode = clamp(
      float(0.21).add(float(1.0).sub(patinaMix).mul(0.14)).add(fineSatin.sub(0.5).mul(0.04)),
      0.16,
      0.42
    );
  }

  // -------------------------------------------------------------------------
  // 2. Honed Vesicular Mineral Basalt (Central Polyhedral Hub & Gyro Ring)
  // -------------------------------------------------------------------------
  const basalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.16,
    roughness: 0.56,
    envMapIntensity: 0.96,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(18.0));
    const fine = mineralToothFbm(positionLocal.add(vec3(4.3, 8.1, 2.7)), float(64.0));
    const spark = smoothstep(float(0.74), float(0.92), fine);

    const basaltDeep = mix(pAzurite.rInf, pUmber.rInf, float(0.58)).mul(0.16);
    const basaltMid = mix(pAzurite.rInf, pUmber.rInf, float(0.42)).mul(0.34);
    const basaltGlint = pCalcite.rInf.mul(0.22);

    const baseRgb = mix(basaltDeep, basaltMid, smoothstep(float(0.25), float(0.80), grain));
    basalt.colorNode = mix(baseRgb, basaltGlint, spark.mul(0.24));
    basalt.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(48.0), float(0.068))
    );
    basalt.roughnessNode = clamp(
      float(0.58).sub(grain.mul(0.12)).sub(spark.mul(0.24)),
      0.24,
      0.72
    );
  }

  // -------------------------------------------------------------------------
  // 3. Faceted Bone-Calcite Crystal Prisms (Kubelka-Munk + Subsurface Edge)
  // -------------------------------------------------------------------------
  const calcite = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.26,
    envMapIntensity: 1.15,
  });
  {
    const strata = mineralToothFbm(positionLocal.mul(vec3(2.8, 3.6, 2.8)), float(3.4));
    const micro = mineralToothFbm(positionLocal.add(vec3(6.2, 2.4, 9.1)), float(38.0));
    const band = sin(positionLocal.y.mul(14.0).add(strata.mul(3.8))).mul(0.5).add(0.5);

    const thickness = float(0.88).add(band.mul(0.58)).add(micro.mul(0.22));
    const backing = mix(pKaolin.rInf.mul(0.88), pOchre.rInf.mul(0.76), band.mul(0.38));
    const kmCalcite = kubelkaMunkGlaze(pCalcite.K, pCalcite.S, thickness, backing);

    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const rim = pow(
      clamp(float(1.0).sub(max(float(0.0), dot(normalize(normalWorld), viewDir))), 0.0, 1.0),
      2.8
    );
    const warmEdge = mix(pRealgar.rInf, pOchre.rInf, float(0.55)).mul(rim.mul(0.18));

    calcite.colorNode = kmCalcite.add(warmEdge);
    calcite.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(34.0), float(0.022))
    );
    calcite.roughnessNode = clamp(float(0.20).add(micro.mul(0.16)), 0.16, 0.42);
  }

  // -------------------------------------------------------------------------
  // 4. Verdigris Malachite Copper-Patina Bosses & Accent Spires
  // -------------------------------------------------------------------------
  const verdigris = new THREE.MeshStandardNodeMaterial({
    metalness: 0.14,
    roughness: 0.54,
    envMapIntensity: 0.94,
  });
  {
    const mottle = mineralToothFbm(positionLocal, float(14.0));
    const fine = mineralToothFbm(positionLocal.add(vec3(1.5, 7.2, 4.9)), float(42.0));
    const thickness = float(0.65).add(mottle.mul(0.85));
    const kmVerd = kubelkaMunkGlaze(pVerdigris.K, pVerdigris.S, thickness, pUmber.rInf.mul(0.42));

    verdigris.colorNode = mix(
      kmVerd,
      pUmber.rInf.mul(0.38),
      smoothstep(float(0.64), float(0.92), fine).mul(0.32)
    );
    verdigris.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(38.0), float(0.058))
    );
    verdigris.roughnessNode = clamp(float(0.48).add(fine.mul(0.22)), 0.36, 0.78);
  }

  // -------------------------------------------------------------------------
  // 5. Luminous Realgar-Amber Core Cavity Heart
  // -------------------------------------------------------------------------
  const amber = new THREE.MeshStandardNodeMaterial({
    metalness: 0.06,
    roughness: 0.22,
    envMapIntensity: 1.1,
  });
  {
    const swirl = mineralToothFbm(positionLocal.add(vec3(0.0, uTime.mul(0.15), 0.0)), float(6.5));
    const breath = sin(uTime.mul(1.6)).mul(0.12).add(0.92);
    const amberRgb = mix(pRealgar.rInf, pOchre.rInf, swirl.mul(0.45));
    amber.colorNode = amberRgb;
    amber.emissiveNode = mix(pRealgar.rInf, pOchre.rInf, float(0.32).add(swirl.mul(0.28)))
      .mul(breath)
      .mul(uAmberGlow)
      .mul(1.45);
    amber.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(26.0), float(0.024))
    );
  }

  // -------------------------------------------------------------------------
  // 6. Hero Translucent Heat-Shrink / Vacuum-Form Film Shader (Front & Back Pass)
  //    Shared TSL node graph with faceDirection-flipped micro-normals,
  //    Kubelka-Munk frosted scatter -> Beer-Lambert contact clarity,
  //    Realgar-Amber infrared thermal rim glow, and radial tension sheen.
  // -------------------------------------------------------------------------
  const buildShrinkFilmMaterial = (side: THREE.Side): THREE.MeshPhysicalNodeMaterial => {
    const mat = new THREE.MeshPhysicalNodeMaterial({
      side,
      transparent: true,
      depthWrite: false,
      metalness: 0.02,
      roughness: 0.52,
      ior: 1.49,
      clearcoat: 1.0,
      clearcoatRoughness: 0.38,
      envMapIntensity: 1.35,
    });

    const aRestDir = attribute('aRestDir', 'vec3');
    const aState = attribute('aState', 'vec4'); // x=shrink, y=tension, z=heat, w=contact
    const aTearEdge = attribute('aTearEdge', 'float');

    const shrinkVal = clamp(aState.x, 0.0, 1.0);
    const tensionVal = clamp(aState.y, 0.0, 1.0);
    const heatVal = clamp(aState.z, 0.0, 1.0);
    const contactVal = clamp(aState.w, 0.0, 1.0);

    // Combined optical tautness & contact-clarity transition factor
    const tautness = clamp(
      tensionVal.mul(0.68).add(contactVal.mul(0.52)).add(shrinkVal.mul(0.25)),
      0.0,
      1.0
    );
    const contactClarity = clamp(
      contactVal.mul(0.82).add(tensionVal.mul(0.32)).mul(smoothstep(float(0.04), float(0.55), shrinkVal.add(tensionVal.mul(0.4)))),
      0.0,
      1.0
    );

    // Material-anchored 3D organic polymer-parchment crinkle grain (zero texture swimming during shrink/peel!)
    const toothMacro = mineralToothFbm(aRestDir, float(4.2));
    const toothFine = mineralToothFbm(aRestDir.add(vec3(5.3, 2.1, 7.8)), float(16.5));
    const microTooth = mineralToothFbm(aRestDir.add(vec3(11.4, 4.7, 3.2)), float(38.0));

    // Delicate tensile micro-pleat relief on high-tension bridges spanning concave voids between spokes
    const bridgeMask = tensionVal.mul(float(1.0).sub(contactVal.mul(0.72)));
    const pleatGrain = mineralToothFbm(aRestDir.mul(vec3(10.5, 22.0, 10.5)), float(1.4));

    // Perturb local normal in material coordinates; smooth out loose crinkles as skin pulls vacuum-tight!
    const crinkleBumpStrength = mix(float(0.145), float(0.024), tautness).add(bridgeMask.mul(0.055));
    const perturbedLocalN = perturbNormal3DFbm(
      normalLocal,
      aRestDir.add(vec3(0.0, pleatGrain.mul(bridgeMask).mul(0.08), 0.0)),
      mix(float(14.0), float(28.0), tautness),
      crinkleBumpStrength
    );
    // CRITICAL for DoubleSide / BackSide parity: multiply view-space normal by faceDirection!
    mat.normalNode = transformNormalToView(perturbedLocalN).mul(faceDirection);

    // Physical Schlick dielectric Fresnel using front/back-aware view normal
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = float(0.045).add(float(0.955).mul(pow(float(1.0).sub(nDotV), 3.6)));

    // 2-flux Kubelka-Munk frosted milky kaolin-parchment film reflectance
    // Unheated loose film has high optical scattering thickness; stretched/contact film thins dramatically
    const kmThickness = mix(float(1.12), float(0.14), contactClarity)
      .add(toothMacro.sub(0.5).mul(0.28).mul(float(1.0).sub(contactClarity.mul(0.65))))
      .add(bridgeMask.mul(pleatGrain.sub(0.5).mul(0.18)));

    const warmBacking = mix(
      vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.44),
      vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear).mul(0.62),
      float(0.32).add(toothMacro.mul(0.25))
    );
    const kmFrosted = kubelkaMunkGlaze(pKaolin.K, pKaolin.S, max(float(0.05), kmThickness), warmBacking);

    // Beer-Lambert contact-clear tint when stretched tight over underlying brass & crystal surfaces
    const opticalChord = mix(float(0.62), float(0.16), contactClarity).div(max(float(0.28), nDotV));
    const blClearTint = beerLambertTransmittance(pKaolin.extinction, opticalChord.mul(0.45));
    const clearGlossColor = mix(pCalcite.rInf.mul(0.95), blClearTint.mul(pCalcite.rInf), float(0.55));

    // Subtle warm parchment tint on interior BackSide lining of peeled-back flaps
    const exteriorRgb = mix(kmFrosted, clearGlossColor, contactClarity.mul(0.72));
    const interiorRgb = mix(kmFrosted.mul(0.78), mix(pOchre.rInf, pKaolin.rInf, float(0.62)), float(0.28));
    const stressSheen = pCalcite.rInf.mul(bridgeMask.mul(0.16).mul(float(0.5).add(pleatGrain.mul(0.5))));
    const baseFilmRgb = select(frontFacing, exteriorRgb, interiorRgb).add(stressSheen);

    mat.colorNode = baseFilmRgb;

    // Frosted matte polymer (roughness ~0.52) clarifies into sleek glossy taut shrink-film (~0.10)
    const frostedRough = float(0.52).add(toothFine.sub(0.5).mul(0.12)).add(microTooth.sub(0.5).mul(0.06));
    const tautRough = float(0.10).add(microTooth.mul(0.06)).add(bridgeMask.mul(0.04));
    mat.roughnessNode = clamp(mix(frostedRough, tautRough, tautness), 0.08, 0.72);
    mat.clearcoatRoughnessNode = clamp(mix(float(0.44), float(0.07), tautness), 0.05, 0.56);

    // Optical clarity alpha:
    // - Unheated loose bag: milky frosted translucent veil (~0.78 normal -> ~0.92 rim) obscuring inner silhouette
    // - Contact-clarified taut skin: optically clear contact window (~0.16 normal) with sharp Fresnel rim (~0.80)
    // - Tensioned concave bridges stay visible (~0.36..0.48) so stretched radial creases gleam cleanly
    const looseNormalAlpha = float(0.78).add(toothMacro.sub(0.5).mul(0.10));
    const clearNormalAlpha = float(0.16).add(bridgeMask.mul(0.24));
    const normalAlpha = mix(looseNormalAlpha, clearNormalAlpha, contactClarity);
    const rimAlpha = mix(float(0.92), float(0.80), contactClarity);
    const frontAlpha = clamp(
      mix(normalAlpha, rimAlpha, fresnel).add(aTearEdge.mul(0.18)),
      0.12,
      0.95
    );
    const backAlpha = clamp(frontAlpha.mul(0.72), 0.08, 0.80);
    mat.opacityNode = select(frontFacing, frontAlpha, backAlpha);

    // 1. Subsurface amber glow bleeding gently from the hidden internal core heart
    const distFromCore = length(positionLocal);
    const coreFalloff = float(1.0).div(float(0.36).add(distFromCore.mul(distFromCore).mul(1.25)));
    const subsurfaceChord = mix(float(1.18), float(0.42), contactClarity);
    const amberTrans = beerLambertTransmittance(pRealgar.extinction, subsurfaceChord);
    const coreSubsurface = mix(pRealgar.rInf, pOchre.rInf, float(0.34))
      .mul(amberTrans)
      .mul(coreFalloff)
      .mul(uAmberGlow)
      .mul(select(frontFacing, float(0.46), float(0.62)));

    // 2. Active Realgar-Amber infrared rim-glow along the warm thermal shrink front
    const beamDist = length(positionWorld.sub(uHeatOrigin));
    const beamRimRing = exp(beamDist.mul(beamDist).negate().mul(5.2)).mul(uHeatActive);
    const thermalFront = clamp(heatVal.mul(0.88).add(beamRimRing.mul(0.35)), 0.0, 1.0);
    const irColor = mix(pRealgar.rInf, pOchre.rInf, thermalFront.mul(0.42));
    const thermalGlow = irColor
      .mul(thermalFront)
      .mul(float(0.34).add(fresnel.mul(0.78)).add(toothFine.mul(0.16)))
      .mul(0.88);

    // 3. Subtle crisp translucent edge glint along ruptured seam lips
    const tearLipGlow = mix(pCalcite.rInf, pRealgar.rInf, float(0.25))
      .mul(aTearEdge)
      .mul(float(0.28).add(fresnel.mul(0.42)))
      .mul(0.45);

    mat.emissiveNode = coreSubsurface.add(thermalGlow).add(tearLipGlow);

    return mat;
  };

  const filmBackMat = buildShrinkFilmMaterial(THREE.BackSide);
  const filmFrontMat = buildShrinkFilmMaterial(THREE.FrontSide);

  return {
    coreMaterials: {
      brass,
      basalt,
      calcite,
      verdigris,
      amber,
    },
    filmFrontMat,
    filmBackMat,
    uniforms: {
      uTime,
      uAmberGlow,
      uHeatOrigin,
      uHeatActive,
    },
    dispose: () => {
      brass.dispose();
      basalt.dispose();
      calcite.dispose();
      verdigris.dispose();
      amber.dispose();
      filmFrontMat.dispose();
      filmBackMat.dispose();
    },
  };
}
