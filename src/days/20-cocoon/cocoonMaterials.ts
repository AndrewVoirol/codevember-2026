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
  fract,
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
  uv,
  vec3,
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  valueNoise3DTsl,
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
  tetherMat: THREE.MeshStandardNodeMaterial;
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
 * - High-contrast feathered strip softboxes carve crisp metallic speculars onto turned brass
 *   astrolabe bevels, vesicular basalt edges, faceted calcite prisms, vitreous amber, and taut film.
 */
export function createCocoonStudioEnvironment(
  renderer: THREE.WebGPURenderer
): CocoonStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
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
  // Soft wraparound studio horizon cards
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
// Fast Irrational-Angle 2-Octave FBM & Domain-Warped Organic Relief Helpers
// ============================================================================

const fbm2Fast = Fn(([pos_in]: [TslNode]) => {
  const p0 = vec3(pos_in);
  const v0 = valueNoise3DTsl(p0);
  const c = float(0.80901699);
  const s = float(0.58778525);
  const p1 = vec3(
    p0.x.mul(c).sub(p0.z.mul(s)),
    p0.y.mul(c).add(p0.x.mul(s)),
    p0.z.mul(c).add(p0.y.mul(s))
  ).mul(2.07);
  const v1 = valueNoise3DTsl(p1);
  return v0.mul(0.66).add(v1.mul(0.34));
});

/**
 * Scale-invariant 4-tap tetrahedral 3D normal perturbation using fast 2-octave FBM.
 */
const perturbNormalFast = Fn(
  ([baseNorm_in, pos_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const sc = float(scale_in);
    const eps = float(0.14).div(max(float(1.0), sc));

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = fbm2Fast(p.add(k1.mul(eps)).mul(sc));
    const f2 = fbm2Fast(p.add(k2.mul(eps)).mul(sc));
    const f3 = fbm2Fast(p.add(k3.mul(eps)).mul(sc));
    const f4 = fbm2Fast(p.add(k4.mul(eps)).mul(sc));

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).div(float(0.56));
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

/**
 * Evaluates a single scalar height value for organic finite-length intersecting mulberry-bark
 * & silkworm-cocoon fibers at domain-warped material coordinate `dw`.
 * Strongly anisotropic curved strands gated by independent longitudinal length envelopes
 * prevent closed "O-ring" contour loops and yield natural crossing paper/silk filaments!
 */
const evalMeanderingFiberRelief = Fn(([dw_in]: [TslNode]) => {
  const dw = vec3(dw_in);

  // Stratum 1: Long gently curved diagonal-horizontal silkworm wrap fibers (~7:1 stretch + length gate)
  const f1 = vec3(
    dw.x.mul(7.5).add(dw.z.mul(4.5)),
    dw.y.mul(46.0).sub(dw.x.mul(14.0)),
    dw.z.mul(36.0).sub(dw.x.mul(18.0))
  );
  const n1 = valueNoise3DTsl(f1);
  const env1 = smoothstep(float(0.24), float(0.76), valueNoise3DTsl(dw.mul(9.5).add(vec3(1.4, 5.2, 3.8))));
  const strand1 = pow(clamp(float(1.0).sub(abs(n1.sub(0.5)).mul(7.4)), 0.0, 1.0), 2.2).mul(env1);

  // Stratum 2: Crossing oblique mulberry-bast fibers (non-orthogonal rotated flow + independent length gate)
  const f2 = vec3(
    dw.x.mul(44.0).add(dw.y.mul(16.0)).add(4.3),
    dw.y.mul(7.0).add(dw.z.mul(5.0)).add(1.9),
    dw.z.mul(38.0).add(dw.y.mul(12.0)).add(6.7)
  );
  const n2 = valueNoise3DTsl(f2);
  const env2 = smoothstep(float(0.26), float(0.78), valueNoise3DTsl(dw.mul(10.5).add(vec3(8.1, 2.3, 6.4))));
  const strand2 = pow(clamp(float(1.0).sub(abs(n2.sub(0.5)).mul(7.6)), 0.0, 1.0), 2.3).mul(env2);

  // Stratum 3: Fine shorter wispy cellulose micro-filaments bridging the matrix
  const f3 = vec3(
    dw.y.mul(62.0).sub(dw.z.mul(22.0)).add(11.2),
    dw.z.mul(11.0).add(dw.x.mul(9.0)).add(3.4),
    dw.x.mul(58.0).add(dw.y.mul(18.0)).add(7.9)
  );
  const n3 = valueNoise3DTsl(f3);
  const env3 = smoothstep(float(0.30), float(0.80), valueNoise3DTsl(dw.mul(16.0).add(vec3(3.7, 9.5, 1.2))));
  const microA = pow(clamp(float(1.0).sub(abs(n3.sub(0.5)).mul(8.2)), 0.0, 1.0), 2.4).mul(env3);

  // Stratum 4: Secondary cross micro-threads
  const f4 = vec3(
    dw.z.mul(64.0).add(dw.x.mul(19.0)).add(5.7),
    dw.x.mul(56.0).sub(dw.y.mul(21.0)).add(9.3),
    dw.y.mul(12.0).add(dw.z.mul(8.0)).add(2.1)
  );
  const n4 = valueNoise3DTsl(f4);
  const env4 = smoothstep(float(0.32), float(0.82), valueNoise3DTsl(dw.mul(17.5).add(vec3(6.9, 4.1, 8.8))));
  const microB = pow(clamp(float(1.0).sub(abs(n4.sub(0.5)).mul(8.4)), 0.0, 1.0), 2.5).mul(env4);

  const macroPeak = max(strand1, strand2.mul(0.94));
  const microPeak = max(microA, microB.mul(0.92));
  return clamp(macroPeak.mul(0.72).add(microPeak.mul(0.42)), 0.0, 1.0);
});

/**
 * 4-tap tetrahedral micro-normal perturbation directly aligned to the curved domain-warped
 * silk/mulberry fibers + broad dry-parchment organic crinkle heightfield.
 */
const perturbNormalOrganicVellum = Fn(
  ([baseNorm_in, dw_in, crinkleAmt_in, fiberAmt_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const dw = vec3(dw_in);
    const eps = float(0.0085);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const evalH = (pt: TslNode) => {
      const cr = valueNoise3DTsl(pt.mul(10.5).add(vec3(2.3, 6.1, 4.7))).mul(float(crinkleAmt_in));
      const fb = evalMeanderingFiberRelief(pt).mul(float(fiberAmt_in));
      return cr.add(fb);
    };

    const h1 = evalH(dw.add(k1.mul(eps)));
    const h2 = evalH(dw.add(k2.mul(eps)));
    const h3 = evalH(dw.add(k3.mul(eps)));
    const h4 = evalH(dw.add(k4.mul(eps)));

    const grad = k1.mul(h1).add(k2.mul(h2)).add(k3.mul(h3)).add(k4.mul(h4)).div(float(0.32));
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad));
  }
);

/**
 * 4-tap tetrahedral normal perturbation for etched vesicular volcanic basalt gas-bubble pits
 * and honed mineral fracture grain.
 */
const perturbNormalVesicularBasalt = Fn(
  ([baseNorm_in, pos_in, strength_in]: [TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const eps = float(0.0045);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const evalBasaltH = (pt: TslNode) => {
      const macroPit = smoothstep(float(0.69), float(0.82), valueNoise3DTsl(pt.mul(36.0)));
      const microPit = smoothstep(float(0.71), float(0.84), valueNoise3DTsl(pt.mul(84.0).add(vec3(4.1, 7.3, 2.9))));
      const grain = valueNoise3DTsl(pt.mul(56.0).add(vec3(1.9, 5.2, 8.6))).mul(0.28);
      return grain.sub(macroPit.mul(0.82)).sub(microPit.mul(0.55));
    };

    const h1 = evalBasaltH(p.add(k1.mul(eps)));
    const h2 = evalBasaltH(p.add(k2.mul(eps)));
    const h3 = evalBasaltH(p.add(k3.mul(eps)));
    const h4 = evalBasaltH(p.add(k4.mul(eps)));

    const grad = k1.mul(h1).add(k2.mul(h2)).add(k3.mul(h3)).add(k4.mul(h4)).div(float(0.26));
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

// ============================================================================
// Material Factory: Inner Sculptural Core + Organic Fibrous Vellum + Silk Tethers
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
  const pCeladon = createPigmentUniforms('celadonGlauconite');
  const pAzurite = createPigmentUniforms('azuritePrussian');
  const pSpore = createPigmentUniforms('sporeSulfur');

  // -------------------------------------------------------------------------
  // 1. Aged Machined Horological Brass
  //    - Clean fine circumferential lathe-brushed sheen (zero wavy chevron artifacts!)
  //    - Dark oil-oxidized raw-umber patina nestled into recessed grooves & bores
  //    - Warm burnished yellow-ochre/calcite specular lift along proud chamfer bevels
  // -------------------------------------------------------------------------
  const brass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.24,
    envMapIntensity: 1.54,
  });
  {
    const coreDet = attribute('aCoreDetail', 'vec3'); // [edgeBevel, creviceAO, axisFrac]
    const edgeBevel = clamp(coreDet.x, 0.0, 1.0);
    const creviceAO = clamp(coreDet.y, 0.0, 1.0);
    const vUv = uv();

    // Fine straight circumferential lathe-brushed satin streaks along turning axis (vUv.x)
    const brushedStreak = valueNoise3DTsl(vec3(vUv.x.mul(7.0), vUv.y.mul(240.0), float(2.7)));
    const fineHairline = sin(vUv.y.mul(380.0)).mul(0.5).add(0.5);
    const microWear = valueNoise3DTsl(positionLocal.mul(68.0).add(vec3(4.1, 1.7, 9.2)));
    const turningRelief = brushedStreak.mul(0.62).add(fineHairline.mul(0.20)).add(microWear.mul(0.18));

    // Fine oxidized patina tarnish concentrated in recessed fillets, bores & collar steps
    const tarnishNoise = fbm2Fast(positionLocal.mul(24.0).add(vec3(3.1, 7.4, 5.2)));
    const recessMask = clamp(
      float(1.0).sub(creviceAO).mul(1.16).add(tarnishNoise.sub(0.5).mul(0.26)),
      0.0,
      1.0
    );
    const burnishMask = clamp(edgeBevel.mul(0.88).add(turningRelief.mul(0.16)), 0.0, 1.0);

    const brassDarkPatina = mix(pUmber.rInf, pSienna.rInf, float(0.26)).mul(0.42);
    const brassSatinMid = mix(pOchre.rInf, pUmber.rInf, float(0.20));
    const brassBurnished = mix(pOchre.rInf, pCalcite.rInf, float(0.38));

    const baseBrass = mix(brassSatinMid, brassBurnished, smoothstep(float(0.35), float(0.95), burnishMask));
    const grooveDark = baseBrass.mul(float(0.93).add(turningRelief.mul(0.12)));
    brass.colorNode = mix(grooveDark, brassDarkPatina, smoothstep(float(0.36), float(0.86), recessMask).mul(0.76));

    // Crisp restrained micro-brushed normal perturbation (no distorted sine ridges)
    brass.normalNode = transformNormalToView(
      perturbNormalFast(normalLocal, positionLocal.mul(vec3(1.0, 1.4, 1.0)), float(78.0), float(0.022))
    );

    brass.metalnessNode = clamp(float(0.93).sub(recessMask.mul(0.24)), 0.65, 0.94);
    brass.roughnessNode = clamp(
      float(0.22)
        .sub(edgeBevel.mul(0.08))
        .add(recessMask.mul(0.28))
        .add(turningRelief.sub(0.5).mul(0.07)),
      0.13,
      0.54
    );
  }

  // -------------------------------------------------------------------------
  // 2. Honed Vesicular Volcanic Basalt (Split Caliper Hub & Mid Gyro Ring)
  //    - True matte volcanic mineral stone (`metalness: 0.02`, NEVER grey plastic!)
  //    - Fine etched concave vesicular gas-bubble micro-pits + plagioclase sparkles
  //    - Subtle honed-chamfer mineral edge wear along polygonal facets
  // -------------------------------------------------------------------------
  const basalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.74,
    envMapIntensity: 0.95,
  });
  {
    const coreDet = attribute('aCoreDetail', 'vec3');
    const edgeBevel = clamp(coreDet.x, 0.0, 1.0);
    const creviceAO = clamp(coreDet.y, 0.0, 1.0);

    const rockMatrix = fbm2Fast(positionLocal.mul(22.0).add(vec3(2.1, 4.7, 1.3)));
    const macroPit = smoothstep(float(0.69), float(0.82), valueNoise3DTsl(positionLocal.mul(36.0)));
    const microPit = smoothstep(
      float(0.71),
      float(0.84),
      valueNoise3DTsl(positionLocal.mul(84.0).add(vec3(4.1, 7.3, 2.9)))
    );
    const pitMask = clamp(max(macroPit, microPit.mul(0.85)), 0.0, 1.0);

    // Fine crystalline plagioclase & pyroxene mineral cleavage glints in the volcanic matrix
    const mineralSpark = smoothstep(
      float(0.82),
      float(0.93),
      valueNoise3DTsl(positionLocal.mul(148.0).add(vec3(9.4, 2.2, 6.8)))
    );

    // Warm stony volcanic charcoal-slate (Raw Umber + subtle Azurite mineral tint — zero blue plastic!)
    const stoneDark = mix(pUmber.rInf, pAzurite.rInf, float(0.24)).mul(0.25);
    const stoneMid = mix(pUmber.rInf, pAzurite.rInf, float(0.20)).mul(0.45);
    const pitCavity = stoneDark.mul(0.26);
    const chamferDust = mix(pKaolin.rInf, pUmber.rInf, float(0.55)).mul(0.38);
    const micaGlint = pCalcite.rInf.mul(0.48);

    const bodyStone = mix(stoneDark, stoneMid, smoothstep(float(0.20), float(0.82), rockMatrix))
      .mul(float(0.76).add(creviceAO.mul(0.28)));
    const withChalkEdge = mix(bodyStone, chamferDust, smoothstep(float(0.64), float(0.98), edgeBevel).mul(0.24));
    const withPits = mix(withChalkEdge, pitCavity, pitMask.mul(0.86));
    basalt.colorNode = mix(withPits, micaGlint, mineralSpark.mul(float(1.0).sub(pitMask)).mul(0.45));

    basalt.normalNode = transformNormalToView(
      perturbNormalVesicularBasalt(normalLocal, positionLocal, float(0.155))
    );
    basalt.roughnessNode = clamp(
      float(0.68)
        .add(pitMask.mul(0.24))
        .sub(rockMatrix.mul(0.08))
        .sub(mineralSpark.mul(0.44)),
      0.22,
      0.92
    );
  }

  // -------------------------------------------------------------------------
  // 3. Natural Faceted Bone-Calcite Crystal Prisms
  //    - Transverse c-axis crystal growth striations (`uv.y`) carved into normals & value
  //    - Crisp oblique internal rhombohedral cleavage planes & mineral fracture veils
  //    - Warm ochre-umber mineral base zoning -> vitreous translucent Iceland-spar apex
  // -------------------------------------------------------------------------
  const calcite = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.0,
    roughness: 0.16,
    clearcoat: 0.84,
    clearcoatRoughness: 0.07,
    envMapIntensity: 1.42,
  });
  {
    const coreDet = attribute('aCoreDetail', 'vec3');
    const edgeBevel = clamp(coreDet.x, 0.0, 1.0);
    const axisFrac = clamp(coreDet.z, 0.0, 1.0);
    const vUv = uv();

    const warp = fbm2Fast(positionLocal.mul(8.5).add(vec3(1.8, 3.2, 5.1)));
    const micro = valueNoise3DTsl(positionLocal.mul(52.0).add(vec3(6.2, 2.4, 9.1)));

    // Stepped transverse crystal growth striations perpendicular to prism c-axis
    const striaA = sin(vUv.y.mul(92.0).add(warp.mul(2.4))).mul(0.5).add(0.5);
    const striaB = sin(vUv.y.mul(215.0).add(warp.mul(3.6))).mul(0.5).add(0.5);
    const striation = striaA.mul(0.62).add(striaB.mul(0.38));

    // Distinct internal rhombohedral cleavage fracture planes (both shadowed trough & pearlescent lip!)
    const cleaveCoordA = dot(positionLocal, vec3(3.6, 4.4, 2.2)).add(warp.mul(0.48));
    const cleaveCoordB = dot(positionLocal, vec3(-2.8, 3.9, 3.2)).add(warp.mul(0.40));
    const cleaveDarkA = smoothstep(float(0.11), float(0.0), abs(fract(cleaveCoordA.add(4.0)).sub(0.5)));
    const cleaveDarkB = smoothstep(float(0.09), float(0.0), abs(fract(cleaveCoordB.add(4.0)).sub(0.5)));
    const cleavageTrough = clamp(max(cleaveDarkA, cleaveDarkB.mul(0.75)), 0.0, 1.0);
    const cleavageGlint = smoothstep(float(0.045), float(0.0), abs(fract(cleaveCoordA.add(4.025)).sub(0.5)));

    // Warm mineral root staining near brass collar transitioning to crisp vitreous spar apex
    const baseRootMask = smoothstep(float(0.52), float(0.06), axisFrac);
    const thickness = float(0.64)
      .add(baseRootMask.mul(1.05))
      .add(striation.mul(0.28))
      .add(micro.mul(0.18));
    const rootBacking = mix(pOchre.rInf.mul(0.52), pUmber.rInf.mul(0.44), baseRootMask.mul(0.55));
    const tipBacking = mix(pKaolin.rInf.mul(0.72), pOchre.rInf.mul(0.58), float(0.28).add(warp.mul(0.24)));
    const backing = mix(tipBacking, rootBacking, baseRootMask);
    const kmCalcite = kubelkaMunkGlaze(pCalcite.K, pCalcite.S, thickness, backing).mul(0.84);

    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const nDotV = clamp(abs(dot(normalize(normalWorld), viewDir)), 0.0, 1.0);
    const vitreousRim = pow(float(1.0).sub(nDotV), 2.65);
    const tipClarity = smoothstep(float(0.15), float(0.95), axisFrac);

    // Crisp cleavage vein darkening + transverse striation tonal relief so prisms NEVER blow out white!
    const veinColor = mix(pUmber.rInf.mul(0.56), pOchre.rInf.mul(0.68), float(0.5));
    const striaTone = kmCalcite.mul(mix(float(0.84), float(1.04), striation));
    const withCleavage = mix(striaTone, veinColor, cleavageTrough.mul(0.36))
      .add(pCalcite.rInf.mul(cleavageGlint.mul(0.22)));
    calcite.colorNode = withCleavage;

    // Restrained grazing backlit vitreous edge translucency strictly on thin beveled crystal tips
    const internalTrans = mix(pCalcite.rInf, pOchre.rInf, float(0.34))
      .mul(vitreousRim.mul(0.18).add(edgeBevel.mul(0.025)))
      .mul(float(0.22).add(tipClarity.mul(0.78)))
      .mul(float(0.85).add(uAmberGlow.mul(0.15)));
    calcite.emissiveNode = internalTrans;

    // Stepped striation + cleavage micro-normals
    const striaPos = vec3(
      positionLocal.x,
      positionLocal.y.mul(3.6).add(striaA.mul(0.032)),
      positionLocal.z
    );
    calcite.normalNode = transformNormalToView(
      perturbNormalFast(normalLocal, striaPos, float(48.0), float(0.068))
    );
    calcite.roughnessNode = clamp(
      float(0.09)
        .add(baseRootMask.mul(0.22))
        .add(striation.mul(0.12))
        .add(cleavageTrough.mul(0.14)),
      0.08,
      0.42
    );
  }

  // -------------------------------------------------------------------------
  // 4. Banded Verdigris Malachite & Powdery Copper-Carbonate Patina
  //    - Concentric botryoidal green-to-celadon malachite mineral strata
  //    - Powdery chalky matte verdigris bloom nestled inside recesses & pits
  //    - Darker oxidized bronze-umber patina peeking along worn mineral ridges
  // -------------------------------------------------------------------------
  const verdigris = new THREE.MeshStandardNodeMaterial({
    metalness: 0.06,
    roughness: 0.50,
    envMapIntensity: 1.08,
  });
  {
    const coreDet = attribute('aCoreDetail', 'vec3');
    const edgeBevel = clamp(coreDet.x, 0.0, 1.0);
    const creviceAO = clamp(coreDet.y, 0.0, 1.0);

    const warpA = fbm2Fast(positionLocal.mul(9.5).add(vec3(2.4, 5.1, 1.7)));
    const warpB = fbm2Fast(positionLocal.mul(22.0).add(vec3(7.3, 1.9, 6.2)));
    const finePore = valueNoise3DTsl(positionLocal.mul(62.0).add(vec3(1.5, 7.2, 4.9)));

    // Botryoidal concentric malachite banding (deep malachite <-> silky celadon veins)
    const bandPhase = length(positionLocal.mul(vec3(1.0, 0.72, 1.0)))
      .mul(24.0)
      .add(warpA.mul(5.8))
      .add(warpB.mul(1.8));
    const bandA = sin(bandPhase).mul(0.5).add(0.5);
    const bandB = sin(bandPhase.mul(2.15).add(1.3)).mul(0.5).add(0.5);
    const malachiteBand = bandA.mul(0.65).add(bandB.mul(0.35));

    // Powdery matte verdigris bloom pools inside collar crevices and surface micro-cavities
    const bloomMask = clamp(
      float(1.0).sub(creviceAO).mul(1.15).add(smoothstep(float(0.62), float(0.86), finePore).mul(0.45)),
      0.0,
      1.0
    );
    // Subtle dark oxidized bronze-umber ridge peeking on sharp exposed mineral shoulders
    const bronzeRidge = smoothstep(float(0.68), float(0.98), edgeBevel).mul(
      smoothstep(float(0.46), float(0.82), warpB)
    );

    const deepMalachite = mix(pVerdigris.rInf, pAzurite.rInf, float(0.24)).mul(0.56);
    const silkyCeladon = mix(pVerdigris.rInf, pCeladon.rInf, float(0.68));
    const powderyBloom = mix(pCeladon.rInf, pKaolin.rInf, float(0.26));
    const darkBronze = mix(pUmber.rInf, pSienna.rInf, float(0.32)).mul(0.44);

    const bandedBody = mix(deepMalachite, silkyCeladon, smoothstep(float(0.18), float(0.86), malachiteBand));
    const withBronze = mix(bandedBody, darkBronze, bronzeRidge.mul(0.58));
    verdigris.colorNode = mix(withBronze, powderyBloom, smoothstep(float(0.32), float(0.84), bloomMask).mul(0.58));

    verdigris.normalNode = transformNormalToView(
      perturbNormalFast(normalLocal, positionLocal.add(vec3(0.0, malachiteBand.mul(0.03), 0.0)), float(48.0), float(0.105))
    );
    verdigris.metalnessNode = clamp(bronzeRidge.mul(0.38).mul(float(1.0).sub(bloomMask)), 0.0, 0.36);
    verdigris.roughnessNode = clamp(
      float(0.30)
        .add(float(1.0).sub(malachiteBand).mul(0.18))
        .add(bloomMask.mul(0.44))
        .sub(bronzeRidge.mul(0.10)),
      0.24,
      0.88
    );
  }

  // -------------------------------------------------------------------------
  // 5. Layered Fossil Realgar-Amber Resin Core Heart
  //    - View-parallax internal resin depth: deep burnt-sienna core -> fiery amber -> honey edges
  //    - Suspended dark organic mineral inclusions & internal conchoidal micro-fracture planes
  //    - Mirror-polished vitreous clearcoat facets catching crisp studio highlights
  // -------------------------------------------------------------------------
  const amber = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.02,
    roughness: 0.10,
    clearcoat: 1.0,
    clearcoatRoughness: 0.05,
    envMapIntensity: 1.48,
  });
  {
    const coreDet = attribute('aCoreDetail', 'vec3');
    const edgeBevel = clamp(coreDet.x, 0.0, 1.0);

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nDotV = clamp(abs(dot(normalize(normalWorld), viewDirW)), 0.0, 1.0);
    const facetEdge = pow(float(1.0).sub(nDotV), 1.85);
    const breath = sin(uTime.mul(1.6)).mul(0.09).add(0.95);

    // View-parallax refracted interior coordinates for true 3D fossil-resin depth illusion
    const depthPos1 = positionLocal.sub(viewDirW.mul(0.10));
    const depthPos2 = positionLocal.sub(viewDirW.mul(0.22));

    const swirlMid = fbm2Fast(depthPos1.mul(9.5).add(vec3(0.0, uTime.mul(0.08), 0.0)));
    const swirlDeep = fbm2Fast(depthPos2.mul(14.5).add(vec3(3.7, 6.2, 2.1)));

    // Tiny suspended dark botanical/mineral inclusions & subtle internal conchoidal fracture veil
    const darkInclusions = smoothstep(
      float(0.73),
      float(0.86),
      valueNoise3DTsl(depthPos1.mul(32.0).add(vec3(5.2, 1.9, 8.4)))
    );
    const microFracture = pow(
      clamp(float(1.0).sub(abs(valueNoise3DTsl(depthPos2.mul(16.0)).sub(0.5)).mul(8.5)), 0.0, 1.0),
      2.2
    );

    const wineCore = mix(pSienna.rInf.mul(0.36), pUmber.rInf.mul(0.28), float(0.35));
    const fieryAmber = mix(pSienna.rInf.mul(0.82), pRealgar.rInf, swirlMid.mul(0.75).add(0.25));
    const honeyEdge = mix(pRealgar.rInf, pSpore.rInf, float(0.45).add(swirlDeep.mul(0.35)));

    const resinBase = mix(
      mix(wineCore, fieryAmber, smoothstep(float(0.15), float(0.85), swirlMid)),
      honeyEdge,
      clamp(facetEdge.mul(0.72).add(edgeBevel.mul(0.22)), 0.0, 1.0)
    );
    // Darken suspended fossil specks & brighten internal amber cleavage sparkles
    const withInclusions = resinBase
      .mul(float(1.0).sub(darkInclusions.mul(0.62)))
      .add(pSpore.rInf.mul(microFracture.mul(0.24)));
    amber.colorNode = withInclusions;

    // Deep layered internal glow: darker saturated core center -> warm luminous amber facets
    const emissiveBody = mix(
      pSienna.rInf.mul(0.34),
      mix(pRealgar.rInf, pOchre.rInf, float(0.38).add(swirlDeep.mul(0.32))),
      clamp(float(0.26).add(facetEdge.mul(0.64)).add(microFracture.mul(0.25)), 0.0, 1.0)
    ).mul(float(1.0).sub(darkInclusions.mul(0.48)));

    amber.emissiveNode = emissiveBody.mul(breath).mul(uAmberGlow).mul(1.24);
    amber.normalNode = transformNormalToView(
      perturbNormalFast(normalLocal, positionLocal, float(28.0), float(0.022))
    );
    amber.roughnessNode = clamp(float(0.08).add(swirlMid.mul(0.06)), 0.07, 0.16);
  }

  // -------------------------------------------------------------------------
  // 6. Organic Fibrous Mulberry/Silkworm Vellum Heat-Shrink Skin (Front & Back Pass)
  //    - Domain-warped meandering silk & mulberry-bark fibers (ZERO straight grid!)
  //    - Mottled natural wall-thickness translucency (thin luminous honey-kaolin windows
  //      vs. denser warm-ochre fibrous veins & raw-silk underside)
  //    - Fiber-aligned micro-normals (every curved fiber catches real studio light!)
  //    - Fragment-level ragged fibrous deckle tear-edge alpha feathering (ZERO straight
  //      polygonal triangle cuts along split seams!)
  // -------------------------------------------------------------------------
  const buildShrinkFilmMaterial = (side: THREE.Side): THREE.MeshPhysicalNodeMaterial => {
    const mat = new THREE.MeshPhysicalNodeMaterial({
      side,
      transparent: true,
      depthWrite: false,
      metalness: 0.0,
      roughness: 0.72,
      ior: 1.48,
      clearcoat: 1.0,
      clearcoatRoughness: 0.45,
      envMapIntensity: 1.24,
    });

    const aRestDir = attribute('aRestDir', 'vec3');
    const aState = attribute('aState', 'vec4'); // x=shrink, y=tension, z=heat, w=contact
    const aTearEdge = attribute('aTearEdge', 'float'); // 1.0 on torn lip verts, 0.0 interior
    const aOrganic = attribute('aOrganic', 'vec4'); // x=wallThickness, y=lipOpenness, z=fiberPhase, w=edgeTaper

    const shrinkVal = clamp(aState.x, 0.0, 1.0);
    const tensionVal = clamp(aState.y, 0.0, 1.0);
    const heatVal = clamp(aState.z, 0.0, 1.0);
    const contactVal = clamp(aState.w, 0.0, 1.0);
    const wallThick = clamp(aOrganic.x, 0.2, 1.0);
    const lipOpen = clamp(aOrganic.y, 0.0, 1.0);
    const edgeTaper = clamp(aOrganic.w, 0.0, 1.0);

    // Combined optical tautness & localized waxy-glossy heat-shrink contact clarity
    const tautness = clamp(
      tensionVal.mul(0.60).add(contactVal.mul(0.58)).add(shrinkVal.mul(0.24)),
      0.0,
      1.0
    );
    const contactClarity = clamp(
      contactVal
        .mul(0.90)
        .add(tensionVal.mul(0.26))
        .mul(smoothstep(float(0.05), float(0.46), shrinkVal.add(tensionVal.mul(0.35)))),
      0.0,
      1.0
    );

    // Physical view angle (Schlick Fresnel + grazing silhouette normal guard)
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = float(0.04).add(float(0.96).mul(pow(float(1.0).sub(nDotV), 3.2)));

    // Smooth 3D domain-warp field anchored in material coordinates (zero texture swim, zero grid!)
    const warpX = valueNoise3DTsl(aRestDir.mul(3.8).add(vec3(1.7, 4.3, 2.9)));
    const warpY = valueNoise3DTsl(aRestDir.mul(3.8).add(vec3(7.1, 1.2, 5.4)));
    const warpZ = valueNoise3DTsl(aRestDir.mul(3.8).add(vec3(4.5, 8.6, 1.8)));
    const dw = aRestDir.add(vec3(warpX, warpY, warpZ).sub(0.5).mul(0.18));

    // Domain-warped curved meandering silk/mulberry fibers + mottled handmade paper cloud
    const curvedFibers = evalMeanderingFiberRelief(dw);
    const sheetMottle = fbm2Fast(dw.mul(6.4).add(vec3(3.1, 7.4, 2.8)));
    const barkFlecks = smoothstep(
      float(0.74),
      float(0.90),
      fbm2Fast(dw.mul(18.5).add(vec3(9.3, 2.7, 6.1)))
    );
    const paperTooth = valueNoise3DTsl(aRestDir.mul(96.0).add(vec3(4.2, 8.9, 1.5)));

    // Stretched concave bridge mask between core ridges
    const bridgeMask = tensionVal.mul(float(1.0).sub(contactVal.mul(0.68)));

    // Organic ragged fibrous deckle tear-edge profile (carved smoothly BEFORE straight polygon boundary!)
    const edgeActive = smoothstep(float(0.04), float(0.34), lipOpen.mul(edgeTaper));
    const tearScallop = fbm2Fast(dw.mul(13.5).add(vec3(1.7, 8.3, 4.9)));
    // Cutoff stays strictly <= 0.92 so geometric triangle boundary (aTearEdge -> 1.0) NEVER renders straight!
    const tearCutoff = clamp(
      float(0.35).add(tearScallop.mul(0.32)).add(curvedFibers.mul(0.24)),
      0.24,
      0.91
    );
    const rawDeckle = smoothstep(tearCutoff.add(0.045), tearCutoff.sub(0.068), aTearEdge);
    const deckleMask = mix(float(1.0), rawDeckle, edgeActive);
    // Dry fibrous paper-pulp fray halo right along the ragged torn frontier
    const frayBand = smoothstep(float(0.14), float(0.76), aTearEdge).mul(rawDeckle).mul(edgeActive);

    // Unified fiber-aligned normal perturbation (grazing-attenuated so silhouettes stay crisp & clean)
    const grazingAtten = smoothstep(float(0.14), float(0.48), nDotV);
    const crinkleAmt = mix(float(0.092), float(0.010), tautness)
      .add(bridgeMask.mul(0.032))
      .mul(grazingAtten);
    const fiberAmt = mix(float(0.072), float(0.012), contactClarity)
      .add(frayBand.mul(0.045))
      .mul(grazingAtten);

    const nOrganic = perturbNormalOrganicVellum(normalLocal, dw, crinkleAmt, fiberAmt);
    const viewNormBase = normalize(transformNormalToView(normalLocal).mul(faceDirection));
    const viewNormOrganic = normalize(transformNormalToView(nOrganic).mul(faceDirection));
    mat.normalNode = normalize(mix(viewNormBase, viewNormOrganic, grazingAtten));

    // Mottled natural parchment thickness: thin luminous honey-kaolin windows vs denser fibrous veins
    const localDensity = clamp(
      wallThick.mul(0.56).add(sheetMottle.mul(0.40)).add(curvedFibers.mul(0.32)),
      0.14,
      1.12
    );
    const kmThickness = mix(float(0.48).add(localDensity.mul(0.88)), float(0.07), contactClarity)
      .add(bridgeMask.mul(0.10));

    // Natural archival parchment backing (Warm Raw Umber + Yellow Ochre — rich handmade mulberry tone)
    const warmParchmentBacking = mix(
      pUmber.rInf.mul(0.44),
      pOchre.rInf.mul(0.64),
      float(0.38).add(sheetMottle.mul(0.30)).add(curvedFibers.mul(0.16))
    );
    const kmVellum = kubelkaMunkGlaze(
      pKaolin.K,
      pKaolin.S,
      max(float(0.04), kmThickness),
      warmParchmentBacking
    );

    // Crisp creamy-ivory raised silk/mulberry fibers + subtle natural warm-umber bark flecks
    // Wet out cleanly under heat-shrink contact so taut skin reveals the gleaming inner core!
    const fiberHighlight = pCalcite.rInf.mul(
      curvedFibers.sub(0.28).mul(0.22).mul(float(1.0).sub(contactClarity.mul(0.86)))
    );
    const withBark = mix(
      kmVellum.add(fiberHighlight),
      mix(pOchre.rInf.mul(0.68), pUmber.rInf.mul(0.52), float(0.45)),
      barkFlecks.mul(0.20).mul(float(1.0).sub(contactClarity.mul(0.85)))
    );

    // Beer-Lambert golden honey-ochre waxy-glossy clarity where heat-shrunk skin hugs the core
    const opticalChord = mix(float(0.52), float(0.11), contactClarity).div(max(float(0.30), nDotV));
    const blHoneyTint = beerLambertTransmittance(pOchre.extinction, opticalChord.mul(0.24));
    const tautWaxColor = mix(pCalcite.rInf.mul(0.96), blHoneyTint.mul(pCalcite.rInf), float(0.52));

    // Distinct matte fibrous inner cocoon lining on peeled-back underside (!frontFacing)
    const exteriorRgb = mix(withBark, tautWaxColor, contactClarity.mul(0.88));
    const interiorRgb = mix(
      withBark.mul(0.80),
      mix(pUmber.rInf.mul(0.58), pOchre.rInf.mul(0.76), float(0.56).add(curvedFibers.mul(0.24))),
      float(0.38)
    );

    // Silvery-calcite tensile sheen across stretched bridges + fuzzy eggshell paper pulp on frayed lips
    const stressSheen = pCalcite.rInf.mul(
      bridgeMask.mul(0.15).mul(float(0.45).add(curvedFibers.mul(0.55)))
    );
    const frayedEdgeLift = pCalcite.rInf.mul(
      frayBand.mul(float(0.16).add(curvedFibers.mul(0.24)))
    );

    mat.colorNode = select(frontFacing, exteriorRgb, interiorRgb)
      .add(stressSheen)
      .add(frayedEdgeLift);

    // Optical surface roughness & clearcoat:
    // - Resting loose vellum: tactile matte-velvety handmade mulberry paper (roughness ~0.74, clearcoat ~0.04)
    // - Taut vacuum-formed contact skin: waxy-glossy clear shrunken membrane (roughness ~0.07, clearcoat ~0.96)
    // - Torn deckled lip & peeled underside: strictly dry matte fibrous paper (clearcoat -> 0)
    const frostedRough = float(0.74)
      .add(curvedFibers.sub(0.35).mul(0.14))
      .add(paperTooth.sub(0.5).mul(0.07));
    const tautRough = float(0.07)
      .add(curvedFibers.mul(0.04))
      .add(bridgeMask.mul(0.05));
    const frontRough = clamp(
      mix(frostedRough, tautRough, tautness).add(frayBand.mul(0.36)),
      0.06,
      0.92
    );
    const backRough = clamp(frontRough.add(0.16).add(curvedFibers.mul(0.06)), 0.34, 0.94);
    mat.roughnessNode = select(frontFacing, frontRough, backRough);

    const dynamicClearcoat = mix(float(0.04), float(0.96), tautness).mul(float(1.0).sub(frayBand.mul(0.94)));
    mat.clearcoatNode = select(frontFacing, dynamicClearcoat, float(0.0));
    mat.clearcoatRoughnessNode = clamp(mix(float(0.62), float(0.05), tautness), 0.04, 0.68);

    // Mottled translucent parchment opacity -> crystal-clear taut contact window, cleanly carved by deckleMask!
    // Thin luminous resting windows (~0.48) let the amber core & brass silhouette softly glow through!
    const thinWindowAlpha = mix(float(0.46), float(0.72), localDensity);
    const looseNormalAlpha = clamp(
      thinWindowAlpha.add(curvedFibers.mul(0.14)).add(paperTooth.sub(0.5).mul(0.04)),
      0.42,
      0.82
    );
    const clearNormalAlpha = float(0.08)
      .add(curvedFibers.mul(0.04))
      .add(bridgeMask.mul(0.18));
    const normalAlpha = mix(looseNormalAlpha, clearNormalAlpha, contactClarity);
    const rimAlpha = mix(float(0.87), float(0.68), contactClarity);
    const frontAlphaBase = clamp(
      mix(normalAlpha, rimAlpha, fresnel).add(frayBand.mul(0.14)),
      0.08,
      0.92
    );
    const backAlphaBase = clamp(frontAlphaBase.mul(0.76), 0.07, 0.82);
    mat.opacityNode = select(frontFacing, frontAlphaBase, backAlphaBase).mul(deckleMask);

    // 1. Warm honey-kaolin subsurface glow shining stronger through thin organic vellum windows!
    const distFromCore = length(positionLocal);
    const coreFalloff = exp(distFromCore.mul(distFromCore).negate().mul(1.85));
    const thinWindowGlow = float(1.28).sub(localDensity.mul(0.56));
    const subsurfaceChord = mix(float(0.72), float(0.22), contactClarity);
    const amberTrans = beerLambertTransmittance(pOchre.extinction, subsurfaceChord.mul(0.42));
    const coreSubsurface = mix(pOchre.rInf, pRealgar.rInf, float(0.34))
      .mul(amberTrans)
      .mul(coreFalloff)
      .mul(thinWindowGlow)
      .mul(uAmberGlow)
      .mul(select(frontFacing, float(0.28), float(0.40)));

    // 2. Soft backlit archival kaolin rim translucency
    const softRimLift = pKaolin.rInf
      .mul(pow(fresnel, 1.55))
      .mul(mix(float(0.10), float(0.06), contactClarity))
      .mul(select(frontFacing, float(1.0), float(0.62)));

    // 3. Cohesive Realgar-Amber & Ochre infrared thermal cone halo along active heat-shrink front
    const beamDist = length(positionWorld.sub(uHeatOrigin));
    const beamRimRing = exp(beamDist.mul(beamDist).negate().mul(4.8)).mul(uHeatActive);
    const thermalFront = clamp(heatVal.mul(0.86).add(beamRimRing.mul(0.34)), 0.0, 1.0);
    const irColor = mix(pRealgar.rInf, pOchre.rInf, float(0.32).add(thermalFront.mul(0.34)));
    const thermalGlow = irColor
      .mul(thermalFront)
      .mul(float(0.28).add(fresnel.mul(0.70)).add(curvedFibers.mul(0.18)))
      .mul(0.82);

    // 4. Delicate backlit fibrous eggshell edge glow along ragged frayed deckle lips
    const deckleLipGlow = mix(pCalcite.rInf, pOchre.rInf, float(0.24))
      .mul(frayBand)
      .mul(float(0.22).add(curvedFibers.mul(0.34)).add(fresnel.mul(0.28)))
      .mul(0.40);

    mat.emissiveNode = coreSubsurface
      .add(softRimLift)
      .add(thermalGlow)
      .add(deckleLipGlow)
      .mul(deckleMask);

    return mat;
  };

  const filmBackMat = buildShrinkFilmMaterial(THREE.BackSide);
  const filmFrontMat = buildShrinkFilmMaterial(THREE.FrontSide);

  // -------------------------------------------------------------------------
  // 7. Delicate Stretched Fibrous Silk Tethers Spanning Freshly Opened Splits
  // -------------------------------------------------------------------------
  const tetherMat = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    metalness: 0.0,
    roughness: 0.56,
    envMapIntensity: 1.1,
  });
  {
    const aTetherStrain = attribute('aTetherStrain', 'vec2'); // x=strain, y=visAlpha
    const strain = clamp(aTetherStrain.x, 0.0, 1.0);
    const visAlpha = clamp(aTetherStrain.y, 0.0, 1.0);
    const vUv = uv();

    // Soft gossamer fiber cross-section & seamless end-fade into deckled parchment lips
    const crossProf = sin(clamp(vUv.y, 0.0, 1.0).mul(3.1415926));
    const endFade = smoothstep(float(0.0), float(0.16), vUv.x).mul(smoothstep(float(1.0), float(0.84), vUv.x));
    const microTwist = sin(vUv.x.mul(28.0).add(vUv.y.mul(4.5))).mul(0.14).add(0.86);

    const silkColor = mix(pKaolin.rInf, pCalcite.rInf, float(0.55).add(strain.mul(0.35)));
    tetherMat.colorNode = silkColor.mul(microTwist);
    tetherMat.emissiveNode = mix(pCalcite.rInf, pOchre.rInf, float(0.20))
      .mul(float(0.26).add(strain.mul(0.24)))
      .mul(crossProf)
      .mul(endFade);
    tetherMat.opacityNode = clamp(crossProf.mul(endFade).mul(0.84).mul(visAlpha), 0.0, 0.86);
  }

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
    tetherMat,
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
      tetherMat.dispose();
    },
  };
}
