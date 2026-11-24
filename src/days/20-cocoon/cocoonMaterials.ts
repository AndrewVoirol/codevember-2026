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
 * - Lifted mid-slate ambient floor + soft wraparound horizon bounce cards prevent grazing
 *   Fresnel / clearcoat reflections from ever crushing to black slits on the cocoon silhouette.
 * - High-contrast feathered strip softboxes carve crisp metallic speculars onto satin brass
 *   astrolabe bevels, honed basalt chamfers, faceted Bone-Calcite prisms, and taut shrink-film.
 */
export function createCocoonStudioEnvironment(
  renderer: THREE.WebGPURenderer
): CocoonStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Lifted slate studio ambient ensures grazing clearcoat reflections never hit a pitch-black void
  envScene.background = new THREE.Color(0x1b2027);

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

  // Primary warm calcite key softbox (upper-right studio strip)
  addSoftbox(10.5, 8.5, [4.8, 5.4, 4.4], 1.0, 0.96, 0.89, 2.35);
  // Crisp narrow architectural rim strip (left) for sharp horological brass & taut film catchlights
  addSoftbox(5.4, 11.2, [-6.0, 2.5, 3.2], 0.92, 0.96, 1.0, 1.95);
  // Rear-top grazing calcite rim scrim
  addSoftbox(12.2, 4.8, [-1.0, 5.2, -5.8], 0.98, 0.95, 0.90, 1.85);
  // Zenith soft overhead canopy
  addSoftbox(9.5, 9.5, [0.0, 7.2, 0.4], 0.97, 0.95, 0.90, 1.22);
  // Rear-right cool mineral fill strip
  addSoftbox(8.0, 6.8, [5.0, 1.6, -5.2], 0.89, 0.94, 1.0, 1.38);
  // Soft wraparound studio horizon cards (eliminate dark equatorial reflection dead-zones)
  addSoftbox(14.0, 5.2, [7.2, -0.4, 0.2], 0.86, 0.84, 0.80, 0.42);
  addSoftbox(14.0, 5.2, [-7.2, -0.4, -0.2], 0.82, 0.86, 0.90, 0.40);
  addSoftbox(14.0, 5.2, [0.0, -0.5, 7.2], 0.88, 0.86, 0.82, 0.36);
  // Warm underside studio table bounce
  addSoftbox(12.0, 12.0, [0.0, -6.4, 0.4], 0.98, 0.90, 0.78, 0.48);

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

/**
 * Pole-free multi-directional 3D fibrous cellulose/parchment weave field in material coordinates.
 * Combines 3 non-orthogonal anisotropic 3D noise strata so close-up 1:1 Retina macro views
 * reveal crisp paper-parchment fibers rather than isotropic clay clouds.
 */
const parchmentFiberField = Fn(([dir_in]: [TslNode]) => {
  const d = vec3(dir_in);
  // Three non-axis-aligned rotated anisotropic strata -> ultra-fine museum mulberry/vellum fibers
  const dA = vec3(
    d.x.mul(0.84).add(d.y.mul(0.54)),
    d.y.mul(0.84).sub(d.x.mul(0.54)),
    d.z
  );
  const fA = mineralToothFbm(dA.mul(vec3(68.0, 16.0, 62.0)), float(1.0));

  const dB = vec3(
    d.x.mul(0.64).sub(d.z.mul(0.77)),
    d.y.mul(0.92).add(d.z.mul(0.39)),
    d.x.mul(0.77).add(d.z.mul(0.64))
  );
  const fB = mineralToothFbm(dB.mul(vec3(17.0, 72.0, 58.0)).add(vec3(4.3, 1.9, 7.1)), float(1.0));

  const dC = vec3(
    d.y.mul(0.58).add(d.z.mul(0.81)),
    d.x.mul(0.88).sub(d.z.mul(0.47)),
    d.z.mul(0.58).sub(d.y.mul(0.81))
  );
  const fC = mineralToothFbm(dC.mul(vec3(62.0, 66.0, 16.0)).add(vec3(9.1, 3.7, 2.4)), float(1.0));

  const sA = smoothstep(float(0.50), float(0.77), fA);
  const sB = smoothstep(float(0.50), float(0.77), fB);
  const sC = smoothstep(float(0.52), float(0.79), fC);
  const peakStrand = max(sA, max(sB, sC));
  const softWeave = fA.mul(0.36).add(fB.mul(0.34)).add(fC.mul(0.30));
  return mix(softWeave, peakStrand, float(0.46));
});

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
  const pSienna = createPigmentUniforms('burntSienna');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pVerdigris = createPigmentUniforms('verdigrisMalachite');
  const pAzurite = createPigmentUniforms('azuritePrussian');

  // -------------------------------------------------------------------------
  // 1. Satin Machined Horological Brass (Astrolabe Rings, Gimbal Pins, Spokes)
  // -------------------------------------------------------------------------
  const brass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.22,
    envMapIntensity: 1.48,
  });
  {
    const turnGrain = mineralToothFbm(positionLocal.mul(vec3(26.0, 62.0, 26.0)), float(1.0));
    const fineSatin = mineralToothFbm(positionLocal.add(vec3(3.1, 1.7, 5.9)), float(74.0));
    const brassBright = mix(pOchre.rInf, pCalcite.rInf, float(0.30));
    const brassPatina = mix(pOchre.rInf, pUmber.rInf, float(0.44));
    const patinaMix = smoothstep(float(0.22), float(0.80), turnGrain.mul(0.60).add(fineSatin.mul(0.40)));

    brass.colorNode = mix(brassPatina, brassBright, patinaMix);
    brass.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(46.0), float(0.018))
    );
    brass.roughnessNode = clamp(
      float(0.17).add(float(1.0).sub(patinaMix).mul(0.13)).add(fineSatin.sub(0.5).mul(0.05)),
      0.14,
      0.36
    );
  }

  // -------------------------------------------------------------------------
  // 2. Honed Vesicular Mineral Basalt (Split Caliper Hub & Mid Gyro Ring)
  // -------------------------------------------------------------------------
  const basalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.14,
    roughness: 0.48,
    envMapIntensity: 1.18,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(20.0));
    const fine = mineralToothFbm(positionLocal.add(vec3(4.3, 8.1, 2.7)), float(68.0));
    const spark = smoothstep(float(0.72), float(0.91), fine);

    // Calibrated charcoal-mineral slate so honed facets read crisply inside the cocoon (never black void!)
    const basaltDeep = mix(pAzurite.rInf, pUmber.rInf, float(0.54)).mul(0.28);
    const basaltMid = mix(pAzurite.rInf, pUmber.rInf, float(0.44)).mul(0.52);
    const basaltGlint = pCalcite.rInf.mul(0.32);

    const baseRgb = mix(basaltDeep, basaltMid, smoothstep(float(0.22), float(0.80), grain));
    basalt.colorNode = mix(baseRgb, basaltGlint, spark.mul(0.28));
    basalt.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(52.0), float(0.075))
    );
    basalt.roughnessNode = clamp(
      float(0.52).sub(grain.mul(0.14)).sub(spark.mul(0.22)),
      0.26,
      0.66
    );
  }

  // -------------------------------------------------------------------------
  // 3. Faceted Bone-Calcite Crystal Prisms (Kubelka-Munk + Crystalline Catchlight)
  // -------------------------------------------------------------------------
  const calcite = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.19,
    envMapIntensity: 1.32,
  });
  {
    const strata = mineralToothFbm(positionLocal.mul(vec3(3.0, 4.0, 3.0)), float(3.8));
    const micro = mineralToothFbm(positionLocal.add(vec3(6.2, 2.4, 9.1)), float(42.0));
    const band = sin(positionLocal.y.mul(16.0).add(strata.mul(4.0))).mul(0.5).add(0.5);

    const thickness = float(0.92).add(band.mul(0.56)).add(micro.mul(0.22));
    const backing = mix(pKaolin.rInf.mul(0.90), pOchre.rInf.mul(0.78), band.mul(0.34));
    const kmCalcite = kubelkaMunkGlaze(pCalcite.K, pCalcite.S, thickness, backing);

    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const rim = pow(
      clamp(float(1.0).sub(max(float(0.0), dot(normalize(normalWorld), viewDir))), 0.0, 1.0),
      2.6
    );
    const warmEdge = mix(pOchre.rInf, pCalcite.rInf, float(0.58)).mul(rim.mul(0.22));

    calcite.colorNode = kmCalcite.add(warmEdge);
    calcite.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(38.0), float(0.020))
    );
    calcite.roughnessNode = clamp(float(0.14).add(micro.mul(0.14)), 0.12, 0.34);
  }

  // -------------------------------------------------------------------------
  // 4. Verdigris Malachite Copper-Patina Bosses & Faceted Spires
  // -------------------------------------------------------------------------
  const verdigris = new THREE.MeshStandardNodeMaterial({
    metalness: 0.12,
    roughness: 0.44,
    envMapIntensity: 1.12,
  });
  {
    const mottle = mineralToothFbm(positionLocal, float(16.0));
    const fine = mineralToothFbm(positionLocal.add(vec3(1.5, 7.2, 4.9)), float(52.0));
    const thickness = float(0.72).add(mottle.mul(0.85));
    const kmVerd = kubelkaMunkGlaze(pVerdigris.K, pVerdigris.S, thickness, pUmber.rInf.mul(0.46));

    verdigris.colorNode = mix(
      kmVerd,
      pUmber.rInf.mul(0.42),
      smoothstep(float(0.66), float(0.92), fine).mul(0.26)
    );
    verdigris.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(44.0), float(0.056))
    );
    verdigris.roughnessNode = clamp(float(0.38).add(fine.mul(0.22)), 0.28, 0.66);
  }

  // -------------------------------------------------------------------------
  // 5. Luminous Faceted Realgar-Amber Fossil-Resin Core Heart
  //    Deep burnt-sienna internal core transitioning to incandescent ochre-amber
  //    on beveled crystal shoulders (replaces flat orange polygon shading!)
  // -------------------------------------------------------------------------
  const amber = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.04,
    roughness: 0.16,
    clearcoat: 0.65,
    clearcoatRoughness: 0.12,
    envMapIntensity: 1.35,
  });
  {
    const swirl = mineralToothFbm(positionLocal.add(vec3(0.0, uTime.mul(0.12), 0.0)), float(8.5));
    const fineInc = mineralToothFbm(positionLocal.add(vec3(3.4, 7.2, 1.8)), float(28.0));
    const breath = sin(uTime.mul(1.6)).mul(0.10).add(0.94);

    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const nDotV = clamp(abs(dot(normalize(normalWorld), viewDir)), 0.0, 1.0);
    const facetEdge = pow(float(1.0).sub(nDotV), 1.65);

    const deepResin = mix(pSienna.rInf.mul(0.65), pRealgar.rInf.mul(0.85), swirl);
    const brightAmber = mix(pRealgar.rInf, pOchre.rInf, float(0.52).add(fineInc.mul(0.25)));
    amber.colorNode = mix(deepResin, brightAmber, facetEdge.mul(0.75).add(0.18));

    const emissiveCore = mix(
      pSienna.rInf.mul(0.45),
      mix(pRealgar.rInf, pOchre.rInf, float(0.42).add(swirl.mul(0.26))),
      float(0.35).add(facetEdge.mul(0.65))
    );
    amber.emissiveNode = emissiveCore.mul(breath).mul(uAmberGlow).mul(1.18);
    amber.normalNode = transformNormalToView(
      perturbNormal3DFbm(normalLocal, positionLocal, float(32.0), float(0.026))
    );
  }

  // -------------------------------------------------------------------------
  // 6. Hero Translucent Polymer-Parchment Heat-Shrink Film (Front & Back Pass)
  //    - Matte fibrous warm-kaolin vellum at rest -> glossy amber-clear taut skin on contact
  //    - Grazing-attenuated normal perturbation (ZERO black silhouette slivers!)
  //    - True pole-free 3D crossed cellulose fiber weave + dry crinkle micro-facets
  //    - Focused golden-amber core glow (ZERO salmon-pink wash!)
  //    - Distinct DoubleSide inner parchment lining on peeled-back flaps
  // -------------------------------------------------------------------------
  const buildShrinkFilmMaterial = (side: THREE.Side): THREE.MeshPhysicalNodeMaterial => {
    const mat = new THREE.MeshPhysicalNodeMaterial({
      side,
      transparent: true,
      depthWrite: false,
      metalness: 0.01,
      roughness: 0.62,
      ior: 1.48,
      clearcoat: 1.0,
      clearcoatRoughness: 0.45,
      envMapIntensity: 1.25,
    });

    const aRestDir = attribute('aRestDir', 'vec3');
    const aState = attribute('aState', 'vec4'); // x=shrink, y=tension, z=heat, w=contact
    const aTearEdge = attribute('aTearEdge', 'float');

    const shrinkVal = clamp(aState.x, 0.0, 1.0);
    const tensionVal = clamp(aState.y, 0.0, 1.0);
    const heatVal = clamp(aState.z, 0.0, 1.0);
    const contactVal = clamp(aState.w, 0.0, 1.0);

    // Combined optical tautness & localized contact-clarity transition
    const tautness = clamp(
      tensionVal.mul(0.62).add(contactVal.mul(0.56)).add(shrinkVal.mul(0.24)),
      0.0,
      1.0
    );
    const contactClarity = clamp(
      contactVal
        .mul(0.88)
        .add(tensionVal.mul(0.28))
        .mul(smoothstep(float(0.05), float(0.48), shrinkVal.add(tensionVal.mul(0.35)))),
      0.0,
      1.0
    );

    // Physical view angle (used both for Schlick Fresnel AND grazing normal-horizon guard)
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = float(0.04).add(float(0.96).mul(pow(float(1.0).sub(nDotV), 3.4)));

    // Material-anchored pole-free 3D parchment fiber weave + crisp crinkle facets (zero texture swim!)
    const fibers = parchmentFiberField(aRestDir);
    const crinkleMeso = mineralToothFbm(aRestDir.add(vec3(2.7, 6.1, 4.3)), float(9.6));
    const microTooth = mineralToothFbm(aRestDir.add(vec3(11.4, 4.7, 3.2)), float(42.0));

    // Surface-tangent tensile micro-pleat relief on stretched concave bridges between core ridges
    const bridgeMask = tensionVal.mul(float(1.0).sub(contactVal.mul(0.66)));
    const pleatGrain = mineralToothFbm(
      aRestDir.add(vec3(crinkleMeso.mul(0.18), fibers.mul(0.14), float(0.0))),
      float(19.0)
    );

    // Grazing attenuation guarantees perturbed normals NEVER tilt past the silhouette horizon
    // (eliminating the black razor-slit rim artifacts 100%!) while keeping crisp interior paper-crinkle facets
    const grazingAtten = smoothstep(float(0.12), float(0.46), nDotV);
    const crinkleStrength = mix(float(0.145), float(0.018), tautness)
      .add(bridgeMask.mul(0.056))
      .mul(grazingAtten);
    const fiberBumpStrength = mix(float(0.068), float(0.018), contactClarity).mul(grazingAtten);

    const nCrinkled = perturbNormal3DFbm(
      normalLocal,
      aRestDir.add(vec3(0.0, pleatGrain.mul(bridgeMask).mul(0.06), 0.0)),
      mix(float(11.5), float(22.0), tautness),
      crinkleStrength
    );
    const nFibrous = perturbNormal3DFbm(
      nCrinkled,
      aRestDir.mul(vec3(2.2, 1.0, 2.2)).add(vec3(5.2, 2.9, 8.1)),
      float(38.0),
      fiberBumpStrength
    );

    const viewNormBase = normalize(transformNormalToView(normalLocal).mul(faceDirection));
    const viewNormPerturbed = normalize(transformNormalToView(nFibrous).mul(faceDirection));
    mat.normalNode = normalize(mix(viewNormBase, viewNormPerturbed, grazingAtten));

    // 2-Flux Kubelka-Munk Warm-Kaolin & Bone-Calcite Archival Parchment Reflectance
    // Unheated loose film: milky-ivory fibrous parchment; Vacuum-contact film: thinned optical window
    const kmThickness = mix(float(1.18), float(0.11), contactClarity)
      .add(fibers.sub(0.5).mul(0.52).mul(float(1.0).sub(contactClarity.mul(0.58))))
      .add(crinkleMeso.sub(0.5).mul(0.32).mul(float(1.0).sub(tautness.mul(0.75))))
      .add(bridgeMask.mul(pleatGrain.sub(0.5).mul(0.18)));

    // Warm archival parchment backing (Raw Umber + Yellow Ochre — zero salmon-pink bias!)
    const warmParchmentBacking = mix(
      pUmber.rInf.mul(0.46),
      pOchre.rInf.mul(0.60),
      float(0.38).add(fibers.mul(0.28))
    );
    const kmFrosted = kubelkaMunkGlaze(
      pKaolin.K,
      pKaolin.S,
      max(float(0.04), kmThickness),
      warmParchmentBacking
    );

    // Tactile ivory cellulose strand & dry crinkle facet contrast at 1:1 Retina magnification
    const fiberLift = pCalcite.rInf.mul(
      fibers.sub(0.46).mul(0.19).mul(float(1.0).sub(contactClarity.mul(0.62)))
    );

    // Beer-Lambert golden honey-ochre contact-clarity glaze right where film hugs metal/crystal ridges
    const opticalChord = mix(float(0.52), float(0.13), contactClarity).div(max(float(0.30), nDotV));
    const blAmberTint = beerLambertTransmittance(pOchre.extinction, opticalChord.mul(0.28));
    const clearGlossColor = mix(pCalcite.rInf.mul(0.96), blAmberTint.mul(pCalcite.rInf), float(0.50));

    // Distinct exterior vs. interior BackSide parchment lining on peeled-back flaps
    const exteriorRgb = mix(kmFrosted.add(fiberLift), clearGlossColor, contactClarity.mul(0.78));
    const interiorRgb = mix(
      kmFrosted.mul(0.84),
      mix(pOchre.rInf.mul(0.82), pKaolin.rInf, float(0.68)),
      float(0.32)
    );
    // Crisp silvery-calcite stress sheen along stretched bridges spanning hollows
    const stressSheen = pCalcite.rInf.mul(
      bridgeMask.mul(0.18).mul(float(0.48).add(pleatGrain.mul(0.52)))
    );
    // Subtle eggshell-calcite frayed fiber highlight right along ruptured Mode-I seam lips
    const tearLipColorLift = pCalcite.rInf.mul(aTearEdge.mul(0.22));
    const baseFilmRgb = select(frontFacing, exteriorRgb, interiorRgb)
      .add(stressSheen)
      .add(tearLipColorLift);

    mat.colorNode = baseFilmRgb;

    // Key optical transformation:
    // - Loose unheated parchment: dry matte-satin fibrous paper (roughness ~0.64, clearcoat ~0.06)
    // - Taut vacuum-formed film: sleek glossy clear polymer skin (roughness ~0.08, clearcoat ~0.88)
    // - Peeled interior lining stays slightly more matte/fibrous than outer stretched film
    const frostedRough = float(0.64)
      .add(fibers.sub(0.5).mul(0.16))
      .add(microTooth.sub(0.5).mul(0.08));
    const tautRough = float(0.08).add(microTooth.mul(0.05)).add(bridgeMask.mul(0.05));
    const frontRough = clamp(mix(frostedRough, tautRough, tautness), 0.07, 0.80);
    const backRough = clamp(frontRough.add(0.14), 0.18, 0.86);
    mat.roughnessNode = select(frontFacing, frontRough, backRough);

    const dynamicClearcoat = mix(float(0.06), float(0.88), tautness);
    mat.clearcoatNode = select(frontFacing, dynamicClearcoat, dynamicClearcoat.mul(0.35));
    mat.clearcoatRoughnessNode = clamp(mix(float(0.58), float(0.06), tautness), 0.05, 0.66);

    // Optical contact-clarity opacity transition:
    // - Loose lifted veil: frosted translucent parchment (~0.70 normal -> ~0.88 rim)
    // - Contact-clarified skin: crystal-clear window (~0.10 normal) with taut Fresnel edge (~0.74)
    // - Stretched concave bridges remain clearly legible (~0.30..0.44)
    const looseNormalAlpha = float(0.70)
      .add(fibers.sub(0.5).mul(0.11))
      .add(crinkleMeso.sub(0.5).mul(0.06));
    const clearNormalAlpha = float(0.10).add(bridgeMask.mul(0.22));
    const normalAlpha = mix(looseNormalAlpha, clearNormalAlpha, contactClarity);
    const rimAlpha = mix(float(0.88), float(0.74), contactClarity);
    const frontAlpha = clamp(
      mix(normalAlpha, rimAlpha, fresnel).add(aTearEdge.mul(0.16)),
      0.09,
      0.93
    );
    const backAlpha = clamp(frontAlpha.mul(0.76), 0.08, 0.82);
    mat.opacityNode = select(frontFacing, frontAlpha, backAlpha);

    // 1. Focused warm golden-amber subsurface glow from the recessed central heart
    //    (Tightly localized to core radius & golden-ochre balanced -> ZERO pink flesh cast!)
    const distFromCore = length(positionLocal);
    const coreFalloff = exp(distFromCore.mul(distFromCore).negate().mul(1.95));
    const subsurfaceChord = mix(float(0.75), float(0.24), contactClarity);
    const amberTrans = beerLambertTransmittance(pOchre.extinction, subsurfaceChord.mul(0.45));
    const coreSubsurface = mix(pOchre.rInf, pRealgar.rInf, float(0.36))
      .mul(amberTrans)
      .mul(coreFalloff)
      .mul(uAmberGlow)
      .mul(select(frontFacing, float(0.26), float(0.38)));

    // 2. Subtle warm eggshell-kaolin rim translucency (soft studio backlight wrap, zero dark rims)
    const softRimLift = pKaolin.rInf
      .mul(pow(fresnel, 1.5))
      .mul(mix(float(0.10), float(0.06), contactClarity))
      .mul(select(frontFacing, float(1.0), float(0.65)));

    // 3. Cohesive Realgar-Amber & Ochre infrared thermal cone halo along active shrink front
    const beamDist = length(positionWorld.sub(uHeatOrigin));
    const beamRimRing = exp(beamDist.mul(beamDist).negate().mul(4.8)).mul(uHeatActive);
    const thermalFront = clamp(heatVal.mul(0.86).add(beamRimRing.mul(0.34)), 0.0, 1.0);
    const irColor = mix(pRealgar.rInf, pOchre.rInf, float(0.32).add(thermalFront.mul(0.34)));
    const thermalGlow = irColor
      .mul(thermalFront)
      .mul(float(0.30).add(fresnel.mul(0.72)).add(fibers.mul(0.14)))
      .mul(0.84);

    // 4. Delicate backlit eggshell edge catch along ruptured parchment seam lips
    const tearLipGlow = mix(pCalcite.rInf, pOchre.rInf, float(0.22))
      .mul(aTearEdge)
      .mul(float(0.20).add(fresnel.mul(0.36)))
      .mul(0.34);

    mat.emissiveNode = coreSubsurface.add(softRimLift).add(thermalGlow).add(tearLipGlow);

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
