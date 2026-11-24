import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  length,
  max,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionGeometry,
  positionLocal,
  positionViewDirection,
  positionWorld,
  select,
  smoothstep,
  texture,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { MINERAL_PALETTE, type PigmentKey } from '../../core/palette';
import {
  beerLambertTransmittance,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import {
  BEAKER_INNER_FLOOR_Y,
  MAX_ACTIVE_CUBES,
  WATER_RADIUS,
  type BeakerDissolveSim,
} from './dissolveSim';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface StudioGlassEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

/**
 * Generates a crisp vertical studio strip-window texture for borosilicate glassware &
 * silver micro-bubble PMREM reflections against the Chalky Slate (#212529) darkroom void.
 */
function createStudioStripTexture(
  r: number,
  g: number,
  b: number,
  sharpEdges: boolean
): THREE.DataTexture {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = ((y + 0.5) / size) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = ((x + 0.5) / size) * 2 - 1;
      let val: number;
      if (sharpEdges) {
        // Crisp photographer's vertical glass-lab softbox window with soft 14% scrim border
        const wx = Math.max(0, Math.min(1, (1.0 - Math.abs(nx)) / 0.18));
        const wy = Math.max(0, Math.min(1, (1.0 - Math.abs(ny)) / 0.16));
        const sx = wx * wx * (3 - 2 * wx);
        const sy = wy * wy * (3 - 2 * wy);
        // Subtle inner diffuser roll-off
        val = sx * sy * (0.82 + 0.18 * (1.0 - 0.4 * nx * nx));
      } else {
        const d = Math.hypot(nx, ny);
        const f = Math.max(0, 1.0 - d * d);
        val = f * f;
      }
      const idx = (y * size + x) * 4;
      data[idx] = Math.min(255, Math.round(r * val * 255));
      data[idx + 1] = Math.min(255, Math.round(g * val * 255));
      data[idx + 2] = Math.min(255, Math.round(b * val * 255));
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
 * Builds a glass-lab photographer's PMREM cubemap with tall vertical curved strip windows,
 * an overhead calcite softbox, and deep negative fill so borosilicate walls, flared spout,
 * stirring rod, and rising silver micro-bubbles sparkle crisply against #212529.
 */
export function createBeakerStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioGlassEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0c1015);

  const disposables: Array<{ dispose: () => void }> = [];

  const addPanel = (
    w: number,
    h: number,
    pos: [number, number, number],
    r: number,
    g: number,
    b: number,
    gain: number,
    sharpEdges = true
  ) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const tex = createStudioStripTexture(r, g, b, sharpEdges);
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

  // 1. Tall vertical front-left warm Eggshell strip window (creates classic curved glass lab-beaker highlight!)
  addPanel(2.3, 11.5, [-5.8, 1.4, 4.4], 1.0, 0.97, 0.92, 2.15, true);

  // 2. Narrow vertical front-right crisp rim strip (defines the opposite cylindrical silhouette & spout curve)
  addPanel(1.5, 10.8, [6.0, 1.6, 3.6], 0.96, 0.98, 1.0, 1.85, true);

  // 3. Broad overhead studio softbox (illuminates the rippling meniscus & dissolving crystal top facets)
  addPanel(8.5, 6.2, [0.4, 7.6, 1.8], 1.0, 0.96, 0.90, 1.65, false);

  // 4. Cool rear-top rim strip (carves crisp grazing highlight along the back lip & rising bubble streams)
  addPanel(9.5, 3.6, [-0.6, 4.8, -6.2], 0.90, 0.95, 1.0, 1.55, true);

  // 5. Subtle cool underside laboratory bench bounce (illuminates the thick solid glass base puck)
  addPanel(10.0, 10.0, [0.0, -6.8, 0.6], 0.86, 0.92, 0.95, 0.48, false);

  // Tighter 0.022 pre-blur keeps borosilicate studio window reflections razor crisp
  const renderTarget = pmrem.fromScene(envScene, 0.022);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Per-Cube Dissolving Crystalline Material Handle (GPU Curvature-Rounding + Pits)
// ---------------------------------------------------------------------------
export interface CrystalCubeMaterialHandle {
  material: THREE.MeshPhysicalNodeMaterial;
  /** x = halfExtent, y = coreHalf, z = filletRadius, w = pitAmp */
  uShapeParams: { value: THREE.Vector4 };
  /** x = scaleXZ, y = scaleY, z = seed, w = progress */
  uAspectSeed: { value: THREE.Vector4 };
  uPigmentRInf: { value: THREE.Vector3 };
  uPigmentK: { value: THREE.Vector3 };
  uPigmentS: { value: THREE.Vector3 };
  uPigmentExt: { value: THREE.Vector3 };
  setPigment: (key: PigmentKey) => void;
  dispose: () => void;
}

/**
 * Evaluates the exact volume-conserving rounded-box + dissolution-pitting local position
 * in TSL, mirroring `evalCrystalSurfacePoint()` in `dissolveSim.ts`.
 */
const evalCrystalMorphLocal = Fn(
  ([u_in, shapeParams_in, aspectSeed_in]: [TslNode, TslNode, TslNode]) => {
    const u = vec3(u_in);
    const halfExtent = vec4(shapeParams_in).x;
    const coreHalf = vec4(shapeParams_in).y;
    const filletRadius = vec4(shapeParams_in).z;
    const pitAmp = vec4(shapeParams_in).w;

    const scaleXZ = vec4(aspectSeed_in).x;
    const scaleY = vec4(aspectSeed_in).y;
    const seed = vec4(aspectSeed_in).z;

    const p = u.mul(halfExtent);
    const q = clamp(p, vec3(coreHalf.negate()), vec3(coreHalf));
    const d = p.sub(q);
    const dLen = length(d);

    const absU = abs(u);
    const isXDom = absU.x.greaterThanEqual(absU.y).and(absU.x.greaterThanEqual(absU.z));
    const isYDom = absU.y.greaterThan(absU.x).and(absU.y.greaterThanEqual(absU.z));
    const sgnX = select(u.x.greaterThanEqual(0.0), float(1.0), float(-1.0));
    const sgnY = select(u.y.greaterThanEqual(0.0), float(1.0), float(-1.0));
    const sgnZ = select(u.z.greaterThanEqual(0.0), float(1.0), float(-1.0));
    const faceNorm = select(
      isXDom,
      vec3(sgnX, 0.0, 0.0),
      select(isYDom, vec3(0.0, sgnY, 0.0), vec3(0.0, 0.0, sgnZ))
    );

    const nBox = select(dLen.greaterThan(1e-5), d.div(max(dLen, float(1e-5))), faceNorm);
    const rPos = select(dLen.greaterThan(1e-5), q.add(nBox.mul(filletRadius)), p);

    // Curvature exposure [0 at face center -> 1 at 90-deg corner] weights deeper edge pitting
    const curvExp = clamp(length(u).sub(1.0).div(0.7320508), 0.0, 1.0);
    const fbmPos = u.mul(3.1).add(vec3(seed.mul(4.7), seed.mul(2.3), seed.mul(6.1)));
    const fbm = mineralToothFbm(fbmPos, float(1.0));
    const pitDepth = pitAmp
      .mul(smoothstep(0.32, 0.78, fbm))
      .mul(float(0.55).add(curvExp.mul(0.45)));

    const eroded = rPos.sub(nBox.mul(pitDepth));
    return vec3(eroded.x.mul(scaleXZ), eroded.y.mul(scaleY), eroded.z.mul(scaleXZ));
  }
);

function createDissolvingCrystalMaterial(initialPigment: PigmentKey): CrystalCubeMaterialHandle {
  const pal = MINERAL_PALETTE[initialPigment];
  const calcite = MINERAL_PALETTE.boneCalcite;

  const uShapeParams = uniform(new THREE.Vector4(0.178, 0.168, 0.010, 0.0));
  const uAspectSeed = uniform(new THREE.Vector4(1.0, 1.0, 2.4, 0.0));
  const uPigmentRInf = uniform(new THREE.Vector3(...pal.rInfLinear));
  const uPigmentK = uniform(new THREE.Vector3(...pal.absorptionK));
  const uPigmentS = uniform(new THREE.Vector3(...pal.scatteringS));
  const uPigmentExt = uniform(new THREE.Vector3(...pal.extinctionRgb));

  const mat = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.02,
    ior: 1.54, // Coarse halite / mineral salt crystal refractive index
    clearcoat: 0.42,
    clearcoatRoughness: 0.24,
  });

  // Pass raw undeformed unit-cube coordinates [-1..1]^3 to fragment stage (zero cycle with positionNode)
  const vUnitCube = positionGeometry.toVarying();

  // 1. Exact analytical curvature-weighted rounded-box + etching-pit vertex morph
  mat.positionNode = evalCrystalMorphLocal(positionGeometry, uShapeParams, uAspectSeed);

  // 2. Crisp derivative-free analytical rounded-box + fine crystalline etching normal in view space
  const evalCrystalNormalView = Fn(() => {
    const u = vUnitCube;
    const halfExtent = uShapeParams.x;
    const coreHalf = uShapeParams.y;
    const pitAmp = uShapeParams.w;
    const scaleXZ = uAspectSeed.x;
    const scaleY = uAspectSeed.y;
    const seed = uAspectSeed.z;
    const progress = uAspectSeed.w;

    const p = u.mul(halfExtent);
    const q = clamp(p, vec3(coreHalf.negate()), vec3(coreHalf));
    const d = p.sub(q);
    const dLen = length(d);

    const absU = abs(u);
    const isXDom = absU.x.greaterThanEqual(absU.y).and(absU.x.greaterThanEqual(absU.z));
    const isYDom = absU.y.greaterThan(absU.x).and(absU.y.greaterThanEqual(absU.z));
    const sgnX = select(u.x.greaterThanEqual(0.0), float(1.0), float(-1.0));
    const sgnY = select(u.y.greaterThanEqual(0.0), float(1.0), float(-1.0));
    const sgnZ = select(u.z.greaterThanEqual(0.0), float(1.0), float(-1.0));
    const faceNorm = select(
      isXDom,
      vec3(sgnX, 0.0, 0.0),
      select(isYDom, vec3(0.0, sgnY, 0.0), vec3(0.0, 0.0, sgnZ))
    );

    const baseN = normalize(
      vec3(
        select(dLen.greaterThan(1e-5), d.div(max(dLen, float(1e-5))), faceNorm)
      ).div(vec3(scaleXZ, scaleY, scaleXZ))
    );

    // 4-tap tetrahedral gradient for deep dissolution pits + micro-crystalline cleavage grain
    const eps = float(0.045);
    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);
    const seedShift = vec3(seed.mul(4.7), seed.mul(2.3), seed.mul(6.1));

    const f1 = mineralToothFbm(u.add(k1.mul(eps)).mul(3.1).add(seedShift), float(1.0));
    const f2 = mineralToothFbm(u.add(k2.mul(eps)).mul(3.1).add(seedShift), float(1.0));
    const f3 = mineralToothFbm(u.add(k3.mul(eps)).mul(3.1).add(seedShift), float(1.0));
    const f4 = mineralToothFbm(u.add(k4.mul(eps)).mul(3.1).add(seedShift), float(1.0));

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).div(eps.mul(4.0));
    const tanGrad = grad.sub(baseN.mul(dot(grad, baseN)));

    // Fine crystalline salt sparkle facets
    const epsFine = float(0.018);
    const g1 = mineralToothFbm(u.add(k1.mul(epsFine)).mul(11.5).add(seedShift), float(1.0));
    const g2 = mineralToothFbm(u.add(k2.mul(epsFine)).mul(11.5).add(seedShift), float(1.0));
    const g3 = mineralToothFbm(u.add(k3.mul(epsFine)).mul(11.5).add(seedShift), float(1.0));
    const g4 = mineralToothFbm(u.add(k4.mul(epsFine)).mul(11.5).add(seedShift), float(1.0));
    const fineGrad = k1.mul(g1).add(k2.mul(g2)).add(k3.mul(g3)).add(k4.mul(g4)).div(epsFine.mul(4.0));
    const tanFine = fineGrad.sub(baseN.mul(dot(fineGrad, baseN)));

    const pitSlopeWeight = clamp(pitAmp.mul(16.5).add(0.14), 0.14, 0.62);
    const fineWeight = float(0.032).add(progress.mul(0.045));
    return transformNormalToView(
      normalize(baseN.add(tanGrad.mul(pitSlopeWeight)).add(tanFine.mul(fineWeight)))
    );
  });

  const crystalNormalView = evalCrystalNormalView();
  mat.normalNode = crystalNormalView;

  // 3. Coarse-Grained Mineral Salt Crystal Color, Pit AO & Translucent Frosted Rim
  const seedShiftFrag = vec3(uAspectSeed.z.mul(4.7), uAspectSeed.z.mul(2.3), uAspectSeed.z.mul(6.1));
  const pitField = mineralToothFbm(vUnitCube.mul(3.1).add(seedShiftFrag), float(1.0));
  const grainField = mineralToothFbm(vUnitCube.mul(11.5).add(seedShiftFrag), float(1.0));
  const curvExpFrag = clamp(length(vUnitCube).sub(1.0).div(0.7320508), 0.0, 1.0);
  const progressFrag = uAspectSeed.w;

  const pitCavity = smoothstep(0.32, 0.78, pitField).mul(smoothstep(0.02, 0.25, progressFrag));
  const colCalcite = vec3(...calcite.rInfLinear);

  // Kubelka-Munk mineral pigment glaze layered over pale translucent calcite salt matrix
  const saltBacking = mix(colCalcite.mul(0.86), uPigmentRInf.mul(1.15), float(0.35));
  const pigmentFilmThickness = mix(float(1.45), float(0.68), curvExpFrag.mul(0.55).add(progressFrag.mul(0.35)))
    .mul(float(0.82).add(grainField.mul(0.38)));

  const kmCrystal = kubelkaMunkGlaze(uPigmentK, uPigmentS, pigmentFilmThickness, saltBacking);

  // Deep etched dissolution cavities concentrate darker Beer-Lambert mineral hue; ridges frost with boneCalcite salt effervescence
  const cavityShade = mix(float(1.0), float(0.64), pitCavity.mul(0.72));
  const frostEdge = smoothstep(0.45, 0.94, curvExpFrag)
    .mul(float(0.28).add(grainField.mul(0.42)))
    .mul(float(1.0).sub(progressFrag.mul(0.35)));

  const rimTerm = float(1.0).sub(
    clamp(abs(dot(normalize(crystalNormalView), normalize(positionViewDirection))), 0.0, 1.0)
  );
  const rimPow = rimTerm.mul(rimTerm);

  // Subtle internal mineral crystal light transmission (edges & thin late-stage slivers glow softly under studio key/rim!)
  const thinChord = clamp(uShapeParams.x.mul(3.2).mul(float(1.0).sub(rimTerm.mul(0.55))), 0.08, 0.95);
  const internalTrans = beerLambertTransmittance(uPigmentExt, thinChord);
  const subSurfaceLift = uPigmentRInf
    .mul(internalTrans)
    .mul(float(0.18).add(rimPow.mul(0.38)).add(progressFrag.mul(0.22)));

  mat.colorNode = mix(kmCrystal.mul(cavityShade), colCalcite, clamp(frostEdge, 0.0, 0.52));
  mat.emissiveNode = subSurfaceLift;
  // Rough etched porous mineral crust inside pits vs glossier wet crystalline facets
  mat.roughnessNode = clamp(
    float(0.26).add(pitCavity.mul(0.48)).add(grainField.mul(0.16)).add(progressFrag.mul(0.12)),
    0.18,
    0.86
  );

  const setPigment = (key: PigmentKey) => {
    const p = MINERAL_PALETTE[key];
    uPigmentRInf.value.set(p.rInfLinear[0], p.rInfLinear[1], p.rInfLinear[2]);
    uPigmentK.value.set(p.absorptionK[0], p.absorptionK[1], p.absorptionK[2]);
    uPigmentS.value.set(p.scatteringS[0], p.scatteringS[1], p.scatteringS[2]);
    uPigmentExt.value.set(p.extinctionRgb[0], p.extinctionRgb[1], p.extinctionRgb[2]);
  };

  return {
    material: mat,
    uShapeParams,
    uAspectSeed,
    uPigmentRInf,
    uPigmentK,
    uPigmentS,
    uPigmentExt,
    setPigment,
    dispose: () => mat.dispose(),
  };
}

// ---------------------------------------------------------------------------
// Complete Day 30 Material Suite
// ---------------------------------------------------------------------------
export interface DissolveMaterialSuite {
  beakerFrontMat: THREE.MeshPhysicalNodeMaterial;
  beakerBackMat: THREE.MeshBasicNodeMaterial;
  waterMat: THREE.MeshPhysicalNodeMaterial;
  bubbleMat: THREE.MeshPhysicalNodeMaterial;
  rodMat: THREE.MeshPhysicalNodeMaterial;
  markingsMat: THREE.MeshBasicNodeMaterial;
  causticDiscMat: THREE.MeshBasicNodeMaterial;
  cubeHandles: CrystalCubeMaterialHandle[];
  uBathExtinction: { value: THREE.Vector3 };
  uBathTint: { value: THREE.Vector3 };
  uBathConcentration: { value: number };
  uSwirlOmega: { value: number };
  uRodOpacity: { value: number };
  uTime: { value: number };
  syncFromSim: (sim: BeakerDissolveSim) => void;
  dispose: () => void;
}

export function createDissolveMaterials(
  sim: BeakerDissolveSim,
  markingsTexture: THREE.DataTexture
): DissolveMaterialSuite {
  const calciteCol = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const celadonCol = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);

  const uBathExtinction = uniform(new THREE.Vector3(0.095, 0.062, 0.052));
  const uBathTint = uniform(new THREE.Vector3(0.68, 0.78, 0.80));
  const uBathConcentration = uniform(0.0);
  const uSwirlOmega = uniform(0.0);
  const uRodOpacity = uniform(0.0);
  const uTime = uniform(0.0);

  const uKeyDir = uniform(new THREE.Vector3(2.8, 4.2, 3.2).normalize());
  const uRimDir = uniform(new THREE.Vector3(-1.5, 3.2, -3.6).normalize());
  const uLeftStripDir = uniform(new THREE.Vector3(-0.78, 0.22, 0.58).normalize());
  const uRightStripDir = uniform(new THREE.Vector3(0.84, 0.24, 0.48).normalize());

  // -------------------------------------------------------------------------
  // 1. Thick-Walled Borosilicate Beaker Front-Face & Back-Face Glass Materials
  // -------------------------------------------------------------------------
  const aGlassMeta = attribute('aGlassMeta', 'vec4'); // x=isExt, y=rimBead, z=basePuck, w=thicknessNorm

  // Front-face borosilicate glass (razor-sharp PMREM + analytical curved studio strip windows)
  const beakerFrontMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.04,
    roughness: 0.045,
    ior: 1.474, // True Borosilicate 3.3 (Pyrex) refractive index!
    reflectivity: 0.92,
    clearcoat: 1.0,
    clearcoatRoughness: 0.035,
    envMapIntensity: 1.75,
  });

  const gView = normalize(cameraPosition.sub(positionWorld));
  const gNorm = normalize(normalWorld);
  const gNdotV = clamp(abs(dot(gNorm, gView)), 0.0, 1.0);
  const oneMinusNdotV = float(1.0).sub(gNdotV);
  const gFresnel = float(0.038).add(
    float(0.962).mul(oneMinusNdotV.mul(oneMinusNdotV).mul(oneMinusNdotV).mul(oneMinusNdotV))
  );

  // Reflected ray for crisp curved vertical studio window reflection ribbons
  const gRefl = normalize(gView.negate().sub(gNorm.mul(dot(gView.negate(), gNorm).mul(2.0))));
  const stripLeft = smoothstep(0.86, 0.985, dot(gRefl, uLeftStripDir));
  const stripRight = smoothstep(0.89, 0.99, dot(gRefl, uRightStripDir)).mul(0.82);
  const keyGlint = smoothstep(0.90, 0.992, dot(gRefl, uKeyDir)).mul(0.92);
  const rimGlint = smoothstep(0.84, 0.98, dot(gRefl, uRimDir)).mul(0.62);

  // Subtle borosilicate green-calcite optical edge depth in thick base puck & rolled pouring lip
  const boroEdgeTint = mix(celadonCol, calciteCol, float(0.74));
  const basePuckGlow = aGlassMeta.z
    .mul(float(0.18).add(gFresnel.mul(0.45)))
    .mul(mix(boroEdgeTint, uBathTint, clamp(uBathConcentration.mul(0.32), 0.0, 0.55)));
  const rimBeadCatch = aGlassMeta.y.mul(float(0.36).add(keyGlint.mul(0.65)).add(rimGlint.mul(0.45)));

  const studioWindowSum = stripLeft.mul(0.55).add(stripRight.mul(0.42)).add(keyGlint.mul(0.48)).add(rimGlint.mul(0.30));

  beakerFrontMat.colorNode = mix(boroEdgeTint, calciteCol, clamp(studioWindowSum.add(rimBeadCatch), 0.0, 1.0));
  beakerFrontMat.emissiveNode = calciteCol
    .mul(stripLeft.mul(0.34).add(stripRight.mul(0.24)).add(rimBeadCatch.mul(0.26)))
    .add(basePuckGlow.mul(0.42));

  // Clear center wall so cubes/plumes/bubbles read crisply; strong refractive silhouettes at limb, spout & thick base puck
  beakerFrontMat.opacityNode = clamp(
    float(0.032)
      .add(gFresnel.mul(0.48))
      .add(aGlassMeta.y.mul(0.34))
      .add(aGlassMeta.z.mul(0.22).mul(float(0.45).add(gFresnel.mul(0.55))))
      .add(studioWindowSum.mul(0.26)),
    0.03,
    0.84
  );

  // Subtle back-face inner glass shell rendered BEHIND plumes/bubbles for genuine hollow depth
  const beakerBackMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
  });
  const backNdotV = clamp(abs(dot(normalize(normalWorld), normalize(cameraPosition.sub(positionWorld)))), 0.0, 1.0);
  const backRim = float(1.0).sub(backNdotV);
  const backFresnel = backRim.mul(backRim).mul(backRim);
  beakerBackMat.colorNode = mix(boroEdgeTint.mul(0.75), calciteCol, backFresnel.mul(0.6));
  beakerBackMat.opacityNode = clamp(
    float(0.025).add(backFresnel.mul(0.22)).add(aGlassMeta.z.mul(0.14)).add(aGlassMeta.y.mul(0.16)),
    0.02,
    0.42
  );

  // -------------------------------------------------------------------------
  // 2. Watertight Deformable Water Volume (Capillary Ripples + Beer-Lambert Solute)
  // -------------------------------------------------------------------------
  const aWaterMeta = attribute('aWaterMeta', 'vec4'); // x=isTop, y=radialFrac, z=heightFrac, w=capillaryRim

  const waterMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.0,
    roughness: 0.055,
    ior: 1.333,
    reflectivity: 0.85,
    clearcoat: 0.85,
    clearcoatRoughness: 0.06,
    envMapIntensity: 1.45,
  });

  // Unconditional uniform-control-flow sample of live 48x48 gravity-capillary wave PDE texture
  const waveUv = clamp(
    vec2(
      positionLocal.x.div(WATER_RADIUS * 2.0).add(0.5),
      positionLocal.z.div(WATER_RADIUS * 2.0).add(0.5)
    ),
    0.02,
    0.98
  );
  const waveSamp = texture(sim.waveTexture, waveUv).level(float(0.0));

  // Perturb top free-surface normal via live wave gradients (dH/dx, dH/dz) so ripples steer pure Fresnel & studio softbox!
  const waterWaveSlopeX = waveSamp.g.mul(aWaterMeta.x).mul(0.95);
  const waterWaveSlopeZ = waveSamp.b.mul(aWaterMeta.x).mul(0.95);
  const waterLocalPerturbed = normalize(
    normalLocal.add(vec3(waterWaveSlopeX.negate(), float(0.0), waterWaveSlopeZ.negate()))
  );
  waterMat.normalNode = transformNormalToView(waterLocalPerturbed);

  const wView = normalize(cameraPosition.sub(positionWorld));
  const wNorm = normalize(
    normalWorld.add(vec3(waterWaveSlopeX.negate(), float(0.0), waterWaveSlopeZ.negate()))
  );
  const wNdotV = clamp(abs(dot(wNorm, wView)), 0.0, 1.0);
  const wOneMinus = float(1.0).sub(wNdotV);
  const wFresnel = float(0.02).add(
    float(0.98).mul(wOneMinus.mul(wOneMinus).mul(wOneMinus).mul(wOneMinus))
  );

  // Beer-Lambert optical chord transmittance across the cylindrical water column
  const opticalChord = mix(float(1.45), float(0.65), wNdotV).mul(
    float(0.75).add(float(1.0).sub(aWaterMeta.z).mul(0.35))
  );
  const bathTrans = beerLambertTransmittance(uBathExtinction, opticalChord);
  const soluteStrength = float(1.0).sub(exp(uBathConcentration.negate().mul(1.05)));

  // Top meniscus Fresnel & capillary wetting ring right where water climbs the inner borosilicate wall
  const wRefl = normalize(wView.negate().sub(wNorm.mul(dot(wView.negate(), wNorm).mul(2.0))));
  const topSoftbox = smoothstep(0.74, 0.975, dot(wRefl, uKeyDir)).mul(aWaterMeta.x);
  const topRimBand = smoothstep(0.78, 0.98, dot(wRefl, uRimDir)).mul(aWaterMeta.x).mul(0.65);

  // Subtle internal directional slope-refraction cue on high-incidence meniscus ripples (anti-matte dielectric guard)
  const rippleSlopeCue = clamp(waveSamp.g.mul(0.45).sub(waveSamp.b.mul(0.35)), -0.25, 0.25)
    .mul(aWaterMeta.x);

  const meniscusRing = aWaterMeta.w.mul(float(0.38).add(wFresnel.mul(0.45)));
  const clearWaterHue = mix(celadonCol, calciteCol, float(0.64));
  const tintedBathColor = mix(clearWaterHue, uBathTint, clamp(soluteStrength.mul(1.15), 0.0, 0.92))
    .mul(mix(vec3(1.0), bathTrans.mul(0.65).add(0.45), soluteStrength.mul(0.65)));

  // Gentle depth darkening at bottom where heavy solute pools
  const floorPoolRichness = float(1.0).sub(aWaterMeta.z).mul(soluteStrength).mul(0.24);
  const waterSurfaceCol = mix(
    tintedBathColor,
    calciteCol,
    clamp(wFresnel.mul(0.42).add(meniscusRing.mul(0.55)).add(topSoftbox.mul(0.55)), 0.0, 0.90)
  ).add(calciteCol.mul(rippleSlopeCue.mul(0.14)));

  waterMat.colorNode = waterSurfaceCol;
  waterMat.emissiveNode = calciteCol
    .mul(meniscusRing.mul(0.24).add(topSoftbox.mul(0.28)).add(topRimBand.mul(0.18)))
    .add(uBathTint.mul(soluteStrength.mul(0.065).add(floorPoolRichness.mul(0.08))));

  // Keep liquid optically limpid when clear and progressively richer as mineral pigment cubes dissolve!
  waterMat.opacityNode = clamp(
    float(0.075)
      .add(soluteStrength.mul(0.24))
      .add(floorPoolRichness.mul(0.14))
      .add(wFresnel.mul(0.28))
      .add(meniscusRing.mul(0.36))
      .add(aWaterMeta.x.mul(0.06).add(topSoftbox.mul(0.24))),
    0.07,
    0.66
  );

  // -------------------------------------------------------------------------
  // 3. Rising Effervescent Silver Micro-Bubbles (Glossy Dielectric Fresnel)
  // -------------------------------------------------------------------------
  const bubbleMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.12,
    roughness: 0.03,
    ior: 1.333,
    clearcoat: 1.0,
    clearcoatRoughness: 0.02,
    envMapIntensity: 2.1,
  });

  const bView = normalize(cameraPosition.sub(positionWorld));
  const bNorm = normalize(normalWorld);
  const bNdotV = clamp(abs(dot(bNorm, bView)), 0.0, 1.0);
  const bRim = float(1.0).sub(bNdotV);
  // Air bubble inside water exhibits intense silver Total Internal Reflection around the rim!
  const bSilverRim = bRim.mul(bRim).mul(float(0.35).add(bRim.mul(0.65)));
  const bRefl = normalize(bView.negate().sub(bNorm.mul(dot(bView.negate(), bNorm).mul(2.0))));
  const bKeySpec = smoothstep(0.84, 0.985, dot(bRefl, uKeyDir));
  const bLeftSpec = smoothstep(0.82, 0.98, dot(bRefl, uLeftStripDir)).mul(0.85);

  const bubbleTint = mix(calciteCol, uBathTint, clamp(uBathConcentration.mul(0.18), 0.0, 0.28));
  bubbleMat.colorNode = mix(bubbleTint, calciteCol, clamp(bSilverRim.add(bKeySpec), 0.0, 1.0));
  bubbleMat.emissiveNode = calciteCol.mul(
    bSilverRim.mul(0.58).add(bKeySpec.mul(0.85)).add(bLeftSpec.mul(0.45))
  );
  bubbleMat.opacityNode = clamp(
    float(0.14).add(bSilverRim.mul(0.76)).add(bKeySpec.mul(0.65)).add(bLeftSpec.mul(0.35)),
    0.14,
    0.96
  );

  // -------------------------------------------------------------------------
  // 4. Slender Tactile Borosilicate Glass Stirring Rod Material
  // -------------------------------------------------------------------------
  const rodMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.05,
    roughness: 0.04,
    ior: 1.474,
    clearcoat: 1.0,
    clearcoatRoughness: 0.03,
    envMapIntensity: 1.85,
  });

  const rView = normalize(cameraPosition.sub(positionWorld));
  const rNorm = normalize(normalWorld);
  const rNdotV = clamp(abs(dot(rNorm, rView)), 0.0, 1.0);
  const rEdge = float(1.0).sub(rNdotV);
  const rFresnel = float(0.12).add(float(0.88).mul(rEdge.mul(rEdge).mul(rEdge)));
  const rRefl = normalize(rView.negate().sub(rNorm.mul(dot(rView.negate(), rNorm).mul(2.0))));
  const rHighlight = smoothstep(0.78, 0.98, dot(rRefl, uKeyDir))
    .add(smoothstep(0.80, 0.98, dot(rRefl, uLeftStripDir)).mul(0.75));

  rodMat.colorNode = mix(boroEdgeTint, calciteCol, clamp(rFresnel.add(rHighlight), 0.0, 1.0));
  rodMat.emissiveNode = calciteCol.mul(rFresnel.mul(0.24).add(rHighlight.mul(0.45)));
  rodMat.opacityNode = clamp(
    float(0.18).add(rFresnel.mul(0.58)).add(rHighlight.mul(0.34)),
    0.0,
    0.92
  ).mul(uRodOpacity);

  // -------------------------------------------------------------------------
  // 5. Graduated Etched Eggshell Volumetric Markings Material
  // -------------------------------------------------------------------------
  const markingsMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const markSamp = texture(markingsTexture, uv());
  markingsMat.colorNode = calciteCol;
  markingsMat.opacityNode = markSamp.a.mul(0.78);

  // -------------------------------------------------------------------------
  // 6. Thick Borosilicate Glass Base Caustic & Refracted Solute Pool Disc
  // -------------------------------------------------------------------------
  const causticDiscMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });

  const cUv = clamp(
    vec2(
      positionLocal.x.div(WATER_RADIUS * 2.0).add(0.5),
      positionLocal.z.div(WATER_RADIUS * 2.0).add(0.5)
    ),
    0.02,
    0.98
  );
  const cWave = texture(sim.waveTexture, cUv).level(float(0.0));
  const cRad = length(positionLocal.xz).div(WATER_RADIUS);
  const cSlopeMag = length(vec2(cWave.g, cWave.b));
  const cCausticNoise = mineralToothFbm(
    vec3(
      positionLocal.x.mul(3.8).add(cWave.g.mul(2.4)),
      uTime.mul(0.32),
      positionLocal.z.mul(3.8).add(cWave.b.mul(2.4))
    ),
    float(1.6)
  );
  // Bright refracted meniscus caustic ring near cylindrical glass wall + rippling floor caustics
  const wallCausticRing = smoothstep(0.66, 0.93, cRad).mul(smoothstep(1.0, 0.93, cRad));
  const causticWeb = smoothstep(0.52, 0.84, cCausticNoise)
    .mul(float(0.35).add(cSlopeMag.mul(2.6)))
    .add(wallCausticRing.mul(0.55));
  const floorEdgeFade = float(1.0).sub(smoothstep(0.92, 1.0, cRad));

  const floorTint = mix(calciteCol, uBathTint, clamp(uBathConcentration.mul(0.65), 0.0, 0.85));
  causticDiscMat.colorNode = mix(floorTint, calciteCol, clamp(causticWeb.mul(0.62), 0.0, 0.85));
  causticDiscMat.opacityNode = clamp(
    causticWeb.mul(0.26).add(uBathConcentration.mul(0.08).mul(float(1.0).sub(cRad.mul(0.5)))),
    0.0,
    0.42
  ).mul(floorEdgeFade);

  // -------------------------------------------------------------------------
  // 7. Four Independent Dissolving Crystalline Mineral Cube Materials
  // -------------------------------------------------------------------------
  const cubeHandles: CrystalCubeMaterialHandle[] = [];
  for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
    cubeHandles.push(createDissolvingCrystalMaterial('celadonGlauconite'));
  }

  const syncFromSim = (s: BeakerDissolveSim) => {
    uBathExtinction.value.copy(s.bathExtinctionRgb);
    uBathTint.value.copy(s.bathTintRgb);
    uBathConcentration.value = s.bathSoluteMass;
    uSwirlOmega.value = s.swirlAngularVel;
    uRodOpacity.value = Math.min(1.0, s.rod.dip * 1.45);
    uTime.value = s.simTime;
  };

  const dispose = () => {
    beakerFrontMat.dispose();
    beakerBackMat.dispose();
    waterMat.dispose();
    bubbleMat.dispose();
    rodMat.dispose();
    markingsMat.dispose();
    causticDiscMat.dispose();
    for (let i = 0; i < cubeHandles.length; i++) {
      cubeHandles[i].dispose();
    }
  };

  return {
    beakerFrontMat,
    beakerBackMat,
    waterMat,
    bubbleMat,
    rodMat,
    markingsMat,
    causticDiscMat,
    cubeHandles,
    uBathExtinction,
    uBathTint,
    uBathConcentration,
    uSwirlOmega,
    uRodOpacity,
    uTime,
    syncFromSim,
    dispose,
  };
}

// Reference BEAKER_INNER_FLOOR_Y cleanly so unused import lints never fire
export const _materialsFloorRef = BEAKER_INNER_FLOOR_Y;
