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
  length,
  max,
  mix,
  normalLocal,
  normalView,
  normalWorld,
  normalize,
  positionGeometry,
  positionLocal,
  positionViewDirection,
  positionWorld,
  screenUV,
  select,
  smoothstep,
  texture,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
  viewportSharedTexture,
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
 * Generates a high-contrast dark-field glass-photography strip softbox texture
 * with crisp scrim borders and subtle center-gradient falloff so curved borosilicate
 * glass walls carry razor-clean vertical reflection ribbons instead of hazy grey fog.
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
        // Razor-crisp dark-field glassware photographer's softbox strip (tight 8% scrim edge)
        const wx = Math.max(0, Math.min(1, (1.0 - Math.abs(nx)) / 0.10));
        const wy = Math.max(0, Math.min(1, (1.0 - Math.abs(ny)) / 0.12));
        const sx = wx * wx * (3 - 2 * wx);
        const sy = wy * wy * (3 - 2 * wy);
        // Bright crisp inner spine tapering slightly toward one side like a studio feathered strip
        const feather = 0.72 + 0.28 * Math.max(0, 1.0 - 0.55 * (nx + 0.25) * (nx + 0.25));
        val = sx * sy * feather;
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
 * Builds a high-contrast dark-field glass-lab studio PMREM cubemap with deep charcoal-black
 * negative fill and razor-sharp tall vertical strip windows so borosilicate glassware,
 * flared pouring spout, stirring rod, and silver micro-bubbles read as true crystal-clear glass.
 */
export function createBeakerStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioGlassEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Deep darkroom studio negative fill maximizes glass-to-backdrop contrast
  envScene.background = new THREE.Color(0x030507);

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

  // 1. Tall narrow front-left warm Eggshell strip blade (iconic curved vertical glassware reflection)
  addPanel(1.35, 13.8, [-6.1, 1.0, 4.5], 1.0, 0.97, 0.92, 2.35, true);

  // 2. Smooth feathered front-right cool-calcite accent strip (angled cleanly off spout pinch azimuth)
  addPanel(1.15, 13.2, [6.6, 0.8, 2.7], 0.94, 0.98, 1.0, 1.85, true);

  // 3. Focused overhead laboratory strip softbox (illuminates meniscus ripples & crystal top facets)
  addPanel(5.6, 2.4, [0.3, 8.2, 1.8], 1.0, 0.96, 0.91, 1.75, true);

  // 4. Crisp rear-top rim strip (catches the rolled back pouring rim & rising silver bubble columns)
  addPanel(8.2, 1.8, [-0.5, 4.4, -6.5], 0.91, 0.96, 1.0, 1.95, true);

  // 5. Tight subtle cool rim kicker from rear-left so 360-deg orbits maintain crisp glass definition
  addPanel(0.95, 11.5, [-5.6, 0.4, -5.2], 0.88, 0.95, 1.0, 1.45, true);

  // Ultra-low 0.008 PMREM blur keeps borosilicate studio window reflections razor sharp
  const renderTarget = pmrem.fromScene(envScene, 0.008);
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
// Per-Cube Dual-Layer Dissolving Crystalline Material Handle:
//   1) Outer Porous Eroding & Fragment-Clipped Mineral Crust (`DoubleSide` + `maskNode`)
//   2) Nested Wet Saturated Receding Inner Core (late-stage honeycomb perforation)
// ---------------------------------------------------------------------------
export interface CrystalCubeMaterialHandle {
  /** Outer porous eroding mineral crust that visibly notches & clips pixels away */
  crustMaterial: THREE.MeshPhysicalNodeMaterial;
  /** Recessed darker wet inner core revealed inside the clipped cavities */
  coreMaterial: THREE.MeshPhysicalNodeMaterial;
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
 * in TSL, mirroring `evalCrystalSurfacePoint()` in `dissolveSim.ts`, with optional inner-core
 * radial scale & extra pit scoop multiplier.
 */
const evalCrystalMorphLocal = Fn(
  ([u_in, shapeParams_in, aspectSeed_in, layerScale_in, pitMult_in]: [
    TslNode,
    TslNode,
    TslNode,
    TslNode,
    TslNode,
  ]) => {
    const u = vec3(u_in);
    const halfExtent = vec4(shapeParams_in).x.mul(layerScale_in);
    const coreHalf = vec4(shapeParams_in).y.mul(layerScale_in);
    const filletRadius = vec4(shapeParams_in).z.mul(layerScale_in);
    const pitAmp = vec4(shapeParams_in).w.mul(layerScale_in).mul(pitMult_in);

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
      .mul(smoothstep(0.28, 0.76, fbm))
      .mul(float(0.55).add(curvExp.mul(0.45)));

    const eroded = rPos.sub(nBox.mul(pitDepth));
    return vec3(eroded.x.mul(scaleXZ), eroded.y.mul(scaleY), eroded.z.mul(scaleXZ));
  }
);

/**
 * Shared analytical + multi-octave cleavage/pit view-space normal for both crystal layers.
 */
const evalCrystalNormalView = Fn(
  ([u_in, shapeParams_in, aspectSeed_in, slopeBoost_in]: [
    TslNode,
    TslNode,
    TslNode,
    TslNode,
  ]) => {
    const u = vec3(u_in);
    const halfExtent = vec4(shapeParams_in).x;
    const coreHalf = vec4(shapeParams_in).y;
    const pitAmp = vec4(shapeParams_in).w;
    const scaleXZ = vec4(aspectSeed_in).x;
    const scaleY = vec4(aspectSeed_in).y;
    const seed = vec4(aspectSeed_in).z;
    const progress = vec4(aspectSeed_in).w;

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

    // 4-tap tetrahedral gradient for deep karst dissolution scoops + coarse cleavage
    const eps = float(0.042);
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

    // Fine crystalline salt sparkle & porous grain micro-facets
    const epsFine = float(0.016);
    const g1 = mineralToothFbm(u.add(k1.mul(epsFine)).mul(13.5).add(seedShift), float(1.0));
    const g2 = mineralToothFbm(u.add(k2.mul(epsFine)).mul(13.5).add(seedShift), float(1.0));
    const g3 = mineralToothFbm(u.add(k3.mul(epsFine)).mul(13.5).add(seedShift), float(1.0));
    const g4 = mineralToothFbm(u.add(k4.mul(epsFine)).mul(13.5).add(seedShift), float(1.0));
    const fineGrad = k1.mul(g1).add(k2.mul(g2)).add(k3.mul(g3)).add(k4.mul(g4)).div(epsFine.mul(4.0));
    const tanFine = fineGrad.sub(baseN.mul(dot(fineGrad, baseN)));

    const pitSlopeWeight = clamp(pitAmp.mul(14.5).add(0.18), 0.18, 0.72).mul(slopeBoost_in);
    const fineWeight = float(0.042).add(progress.mul(0.055));
    return transformNormalToView(
      normalize(baseN.add(tanGrad.mul(pitSlopeWeight)).add(tanFine.mul(fineWeight)))
    ).mul(faceDirection);
  }
);

function createDissolvingCrystalPair(initialPigment: PigmentKey): CrystalCubeMaterialHandle {
  const pal = MINERAL_PALETTE[initialPigment];
  const calcite = MINERAL_PALETTE.boneCalcite;

  const uShapeParams = uniform(new THREE.Vector4(0.178, 0.168, 0.010, 0.0));
  const uAspectSeed = uniform(new THREE.Vector4(1.0, 1.0, 2.4, 0.0));
  const uPigmentRInf = uniform(new THREE.Vector3(...pal.rInfLinear));
  const uPigmentK = uniform(new THREE.Vector3(...pal.absorptionK));
  const uPigmentS = uniform(new THREE.Vector3(...pal.scatteringS));
  const uPigmentExt = uniform(new THREE.Vector3(...pal.extinctionRgb));

  const colCalcite = vec3(...calcite.rInfLinear);

  // =========================================================================
  // LAYER 1: Outer Porous Eroding & Fragment-Clipped Mineral Crust
  // =========================================================================
  const crustMaterial = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    metalness: 0.02,
    ior: 1.54,
    clearcoat: 0.34,
    clearcoatRoughness: 0.32,
  });

  const vUnitCrust = positionGeometry.toVarying();
  crustMaterial.positionNode = evalCrystalMorphLocal(
    positionGeometry,
    uShapeParams,
    uAspectSeed,
    float(1.0),
    float(1.0)
  );

  const crustNormalView = evalCrystalNormalView(vUnitCrust, uShapeParams, uAspectSeed, float(1.05));
  crustMaterial.normalNode = crustNormalView;

  // Multi-scale 3D crystalline cleavage + karst pore erosion field driving active pixel-clipping
  const seedC = uAspectSeed.z;
  const progC = uAspectSeed.w;
  const shiftC1 = vec3(seedC.mul(4.7), seedC.mul(2.3), seedC.mul(6.1));
  const shiftC2 = vec3(seedC.mul(8.3).add(3.1), seedC.mul(5.9).sub(1.7), seedC.mul(2.7).add(4.4));

  const coarsePoreC = mineralToothFbm(vUnitCrust.mul(2.75).add(shiftC1), float(1.0));
  const medGrainC = mineralToothFbm(vUnitCrust.mul(7.2).add(shiftC2), float(1.0));
  const fineCrystalC = mineralToothFbm(vUnitCrust.mul(16.5).add(shiftC1), float(1.0));

  // Weighted structural integrity [0..1]: coarse pockets + sharp crystalline chip grains
  const crustIntegrity = coarsePoreC.mul(0.48).add(medGrainC.mul(0.35)).add(fineCrystalC.mul(0.17));
  const curvExpC = clamp(length(vUnitCrust).sub(1.0).div(0.7320508), 0.0, 1.0);

  // Active dissolution front advances aggressively from sharp corners & edges inward!
  // At progress=0 (spawn): cut <= -0.14 (100% solid brick).
  // By progress=0.06..0.22 (first 1..3.5s): corners & porous pockets visibly clip/bite open!
  const cornerLeadC = curvExpC.mul(0.34);
  const erodeFrontC = smoothstep(0.012, 0.84, progC).mul(1.06).sub(0.14);
  const shellMarginC = crustIntegrity.sub(erodeFrontC.add(cornerLeadC));

  // True hardware fragment pixel-clipping carving jagged notches & porous holes in the crust!
  crustMaterial.maskNode = shellMarginC.greaterThan(0.0);

  // Wet saturated darkening halo + sparkling effervescent salt frost right along the active clip edge
  const activeDissolveGate = smoothstep(0.012, 0.10, progC);
  const wetBandC = float(1.0).sub(smoothstep(0.0, 0.16, shellMarginC)).mul(activeDissolveGate);
  const fizzRimC = float(1.0)
    .sub(smoothstep(0.0, 0.052, shellMarginC))
    .mul(smoothstep(0.40, 0.82, fineCrystalC))
    .mul(activeDissolveGate);

  const pitCavityC = smoothstep(0.28, 0.76, coarsePoreC).mul(smoothstep(0.015, 0.20, progC));
  const saltBackingC = mix(colCalcite.mul(0.88), uPigmentRInf.mul(1.12), float(0.38));
  const filmThickC = mix(float(1.55), float(0.75), curvExpC.mul(0.48).add(progC.mul(0.30)))
    .mul(float(0.84).add(fineCrystalC.mul(0.36)));

  const kmCrust = kubelkaMunkGlaze(uPigmentK, uPigmentS, filmThickC, saltBackingC);

  // Water-soaked darker mineral collar along freshly eroding notches + interior hole wall darkening
  const wetDarkenC = mix(float(1.0), float(0.46), wetBandC);
  const cavityShadeC = mix(float(1.0), float(0.62), pitCavityC.mul(0.75));
  const backCutShadeC = select(faceDirection.greaterThan(0.0), float(1.0), float(0.64));

  const dryFrostEdgeC = smoothstep(0.50, 0.94, curvExpC)
    .mul(float(0.24).add(fineCrystalC.mul(0.40)))
    .mul(float(1.0).sub(wetBandC.mul(0.75)));

  const crustRimTerm = float(1.0).sub(
    clamp(abs(dot(normalize(crustNormalView), normalize(positionViewDirection))), 0.0, 1.0)
  );
  const crustRimPow = crustRimTerm.mul(crustRimTerm);

  const thinChordC = clamp(
    uShapeParams.x.mul(2.8).mul(float(1.0).sub(crustRimTerm.mul(0.55))),
    0.06,
    0.92
  );
  const transC = beerLambertTransmittance(uPigmentExt, thinChordC);

  const crustBaseCol = kmCrust.mul(cavityShadeC).mul(wetDarkenC).mul(backCutShadeC);
  crustMaterial.colorNode = mix(
    crustBaseCol,
    colCalcite,
    clamp(dryFrostEdgeC.add(fizzRimC.mul(0.65)), 0.0, 0.62)
  );
  crustMaterial.emissiveNode = uPigmentRInf
    .mul(transC)
    .mul(float(0.14).add(crustRimPow.mul(0.32)).add(wetBandC.mul(0.18)))
    .add(colCalcite.mul(fizzRimC.mul(0.24)));
  crustMaterial.roughnessNode = clamp(
    float(0.42)
      .add(pitCavityC.mul(0.34))
      .add(fineCrystalC.mul(0.16))
      .sub(wetBandC.mul(0.24)),
    0.16,
    0.88
  );

  // =========================================================================
  // LAYER 2: Nested Darker Wet Receding Inner Mineral Core
  // =========================================================================
  const coreMaterial = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    metalness: 0.03,
    ior: 1.54,
    clearcoat: 0.78,
    clearcoatRoughness: 0.16,
  });

  const vUnitCore = positionGeometry.toVarying();
  const coreLayerScale = mix(float(0.92), float(0.78), smoothstep(0.04, 0.65, uAspectSeed.w));
  coreMaterial.positionNode = evalCrystalMorphLocal(
    positionGeometry,
    uShapeParams,
    uAspectSeed,
    coreLayerScale,
    float(1.38)
  );

  const coreNormalView = evalCrystalNormalView(vUnitCore, uShapeParams, uAspectSeed, float(1.22));
  coreMaterial.normalNode = coreNormalView;

  const seedK = uAspectSeed.z;
  const progK = uAspectSeed.w;
  const shiftK1 = vec3(seedK.mul(4.7), seedK.mul(2.3), seedK.mul(6.1));
  const shiftK2 = vec3(seedK.mul(6.1).add(5.3), seedK.mul(3.7).sub(2.2), seedK.mul(7.9).add(1.1));

  const corePore = mineralToothFbm(vUnitCore.mul(3.1).add(shiftK1), float(1.0));
  const coreGrain = mineralToothFbm(vUnitCore.mul(8.6).add(shiftK2), float(1.0));
  const coreIntegrity = corePore.mul(0.55).add(coreGrain.mul(0.45));
  const curvExpK = clamp(length(vUnitCore).sub(1.0).div(0.7320508), 0.0, 1.0);

  // Inner core stays solid beneath the eroding outer crust until mid-late stage (progress > 0.56),
  // then perforates into a glowing lacy mineral skeleton right before dissolving completely!
  const coreCut = smoothstep(0.54, 0.995, progK).mul(1.16).sub(0.22).add(curvExpK.mul(0.22));
  const coreMargin = coreIntegrity.sub(coreCut);
  coreMaterial.maskNode = coreMargin.greaterThan(0.0);

  const coreWetEdge = float(1.0)
    .sub(smoothstep(0.0, 0.14, coreMargin))
    .mul(smoothstep(0.54, 0.68, progK));

  // Saturated darker wet mineral interior contrasts crisply inside every carved outer-crust notch!
  const wetBackingK = uPigmentRInf.mul(0.68);
  const kmCore = kubelkaMunkGlaze(
    uPigmentK,
    uPigmentS,
    float(2.15).mul(float(0.85).add(coreGrain.mul(0.30))),
    wetBackingK
  );

  const coreRimTerm = float(1.0).sub(
    clamp(abs(dot(normalize(coreNormalView), normalize(positionViewDirection))), 0.0, 1.0)
  );
  const coreRimPow = coreRimTerm.mul(coreRimTerm);
  const coreChord = clamp(
    uShapeParams.x.mul(2.2).mul(float(1.0).sub(coreRimTerm.mul(0.60))),
    0.05,
    0.85
  );
  const coreTrans = beerLambertTransmittance(uPigmentExt, coreChord);

  coreMaterial.colorNode = mix(
    kmCore.mul(float(0.66).add(corePore.mul(0.20))),
    colCalcite,
    coreWetEdge.mul(0.28)
  );
  coreMaterial.emissiveNode = uPigmentRInf
    .mul(coreTrans)
    .mul(float(0.16).add(coreRimPow.mul(0.44)).add(progK.mul(0.26)).add(coreWetEdge.mul(0.22)));
  coreMaterial.roughnessNode = clamp(
    float(0.20).add(corePore.mul(0.22)).add(coreGrain.mul(0.10)),
    0.14,
    0.58
  );

  const setPigment = (key: PigmentKey) => {
    const p = MINERAL_PALETTE[key];
    uPigmentRInf.value.set(p.rInfLinear[0], p.rInfLinear[1], p.rInfLinear[2]);
    uPigmentK.value.set(p.absorptionK[0], p.absorptionK[1], p.absorptionK[2]);
    uPigmentS.value.set(p.scatteringS[0], p.scatteringS[1], p.scatteringS[2]);
    uPigmentExt.value.set(p.extinctionRgb[0], p.extinctionRgb[1], p.extinctionRgb[2]);
  };

  return {
    crustMaterial,
    coreMaterial,
    uShapeParams,
    uAspectSeed,
    uPigmentRInf,
    uPigmentK,
    uPigmentS,
    uPigmentExt,
    setPigment,
    dispose: () => {
      crustMaterial.dispose();
      coreMaterial.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Complete Day 30 Material Suite
// ---------------------------------------------------------------------------
export interface DissolveMaterialSuite {
  beakerBackOuterMat: THREE.MeshBasicNodeMaterial;
  beakerBackInnerMat: THREE.MeshBasicNodeMaterial;
  beakerNearInnerMat: THREE.MeshBasicNodeMaterial;
  beakerFrontMat: THREE.MeshPhysicalNodeMaterial;
  waterMat: THREE.MeshPhysicalNodeMaterial;
  bubbleMat: THREE.MeshPhysicalNodeMaterial;
  crumbMat: THREE.MeshPhysicalNodeMaterial;
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

  const uKeyDir = uniform(new THREE.Vector3(2.8, 4.4, 3.3).normalize());
  const uRimDir = uniform(new THREE.Vector3(-1.5, 3.2, -3.6).normalize());
  const uLeftStripDir = uniform(new THREE.Vector3(-0.78, 0.16, 0.60).normalize());
  const uRightStripDir = uniform(new THREE.Vector3(0.84, 0.18, 0.51).normalize());

  // -------------------------------------------------------------------------
  // 1. Crystal-Clear Dark-Field Optical Borosilicate Glassware (Split Far/Near Passes)
  //    Zero milky alpha washout; real viewportSharedTexture cylindrical Snell refraction,
  //    delicate edge dispersion, double-wall thickness contours & razor-sharp studio strips.
  // -------------------------------------------------------------------------
  const aGlassMeta = attribute('aGlassMeta', 'vec4'); // x=isExt, y=rimBead, z=basePuck, w=thicknessNorm
  const boroEdgeTint = mix(celadonCol, calciteCol, float(0.62));

  // 1A. Far Exterior Glass Wall (`BackSide`, `isExt > 0.45`, rendered BEHIND liquid at renderOrder=2)
  const beakerBackOuterMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
  });
  beakerBackOuterMat.maskNode = aGlassMeta.x.greaterThan(0.45);
  {
    const bView = normalize(cameraPosition.sub(positionWorld));
    const bNdotV = clamp(abs(dot(normalize(normalWorld), bView)), 0.0, 1.0);
    const bEdge = float(1.0).sub(bNdotV);
    const bEdge2 = bEdge.mul(bEdge);
    const bFresnel = bEdge2.mul(bEdge2);
    // Crisp far outer silhouette & rolled back lip ONLY — zero milky center haze
    const backLip = aGlassMeta.y.mul(float(0.18).add(bEdge2.mul(0.32)));
    const backHeel = aGlassMeta.z.mul(bEdge2.mul(bEdge)).mul(0.28);
    beakerBackOuterMat.colorNode = mix(boroEdgeTint.mul(0.82), calciteCol, bFresnel.mul(0.75).add(backLip));
    beakerBackOuterMat.opacityNode = clamp(
      bFresnel.mul(0.34).add(backLip).add(backHeel),
      0.0,
      0.52
    );
  }

  // 1B. Far Inner Glass Wall & Inner Floor Edge (`FrontSide`, `isExt <= 0.45`, rendered BEHIND liquid at renderOrder=2)
  const beakerBackInnerMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  beakerBackInnerMat.maskNode = aGlassMeta.x.lessThanEqual(0.45);
  {
    const biView = normalize(cameraPosition.sub(positionWorld));
    const biNdotV = clamp(abs(dot(normalize(normalWorld), biView)), 0.0, 1.0);
    const biEdge = float(1.0).sub(biNdotV);
    const biEdge2 = biEdge.mul(biEdge);
    const biRim = biEdge2.mul(biEdge2).mul(biEdge);
    // Crisp inner cylindrical wall contour & inner floor corner-fillet optical catch (ZERO flat floor haze!)
    const floorRadNorm = clamp(length(positionLocal.xz).div(WATER_RADIUS), 0.0, 1.0);
    const innerFloorFillet = aGlassMeta.z
      .mul(smoothstep(0.76, 0.99, floorRadNorm))
      .mul(float(0.14).add(biEdge.mul(0.22)));
    const innerLip = aGlassMeta.y.mul(float(0.12).add(biEdge2.mul(0.24)));
    beakerBackInnerMat.colorNode = mix(boroEdgeTint.mul(0.85), calciteCol, biRim.mul(0.7).add(innerLip));
    beakerBackInnerMat.opacityNode = clamp(
      biRim.mul(0.30).add(innerFloorFillet).add(innerLip),
      0.0,
      0.46
    );
  }

  // 1C. Near Inner Glass Wall Interface (`BackSide`, `isExt <= 0.45`, rendered IN FRONT of water at renderOrder=8)
  //     Carves the crisp double-wall thickness line where borosilicate glass meets the water column!
  const beakerNearInnerMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
  });
  beakerNearInnerMat.maskNode = aGlassMeta.x.lessThanEqual(0.45);
  {
    const niView = normalize(cameraPosition.sub(positionWorld));
    const niNdotV = clamp(abs(dot(normalize(normalWorld), niView)), 0.0, 1.0);
    const niEdge = float(1.0).sub(niNdotV);
    const niEdge2 = niEdge.mul(niEdge);
    const niTangent = niEdge2.mul(niEdge2).mul(niEdge);
    const floorRadNorm = clamp(length(positionLocal.xz).div(WATER_RADIUS), 0.0, 1.0);
    const innerFloorRim = aGlassMeta.z.mul(smoothstep(0.82, 0.99, floorRadNorm)).mul(0.20);
    beakerNearInnerMat.colorNode = mix(boroEdgeTint, calciteCol, niTangent.mul(0.78));
    beakerNearInnerMat.opacityNode = clamp(
      niTangent.mul(0.34).add(innerFloorRim).add(aGlassMeta.y.mul(niEdge2).mul(0.22)),
      0.0,
      0.48
    );
  }

  // 1D. Hero Front-Face Refractive Borosilicate Glass Shell (`FrontSide`, `isExt > 0.45`, renderOrder=9)
  //     Uses `viewportSharedTexture` + `backdropNode` for TRUE cylindrical Snell refraction,
  //     subtle borosilicate chromatic dispersion, crystal-clear 0% haze center, & sharp studio strips!
  const beakerFrontMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.0,
    roughness: 0.018,
    ior: 1.474, // True Borosilicate 3.3 (Pyrex) refractive index
    reflectivity: 1.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.015,
    envMapIntensity: 2.25,
  });
  beakerFrontMat.maskNode = aGlassMeta.x.greaterThan(0.45);

  const gView = normalize(cameraPosition.sub(positionWorld));
  const gNorm = normalize(normalWorld);
  const gNdotV = clamp(abs(dot(gNorm, gView)), 0.0, 1.0);
  const gEdge = float(1.0).sub(gNdotV);
  const gEdge2 = gEdge.mul(gEdge);
  // Exact Schlick Fresnel for n = 1.474 (F0 = 0.0367)
  const gFresnel = float(0.0367).add(float(0.9633).mul(gEdge2.mul(gEdge2).mul(gEdge)));

  // View-space surface normal drives physically consistent cylindrical Snell refraction + edge dispersion
  const nViewG = normalize(normalView);
  const refractDir = vec2(nViewG.x.negate(), nViewG.y);
  const baseHeelMask = aGlassMeta.z.mul(smoothstep(0.22, 0.95, gEdge));
  const refractAmp = float(0.0045)
    .add(gEdge2.mul(0.032))
    .add(baseHeelMask.mul(0.024))
    .add(aGlassMeta.y.mul(0.016));
  // Ultra-subtle borosilicate chromatic dispersion (Abbe V_d ~ 65.4) — zero rainbow smear on spout rim!
  const dispAmp = float(0.00016)
    .add(gEdge2.mul(0.00062))
    .add(baseHeelMask.mul(0.00055));

  const uvR = clamp(screenUV.add(refractDir.mul(refractAmp.add(dispAmp))), vec2(0.003), vec2(0.997));
  const uvG = clamp(screenUV.add(refractDir.mul(refractAmp)), vec2(0.003), vec2(0.997));
  const uvB = clamp(screenUV.add(refractDir.mul(refractAmp.sub(dispAmp))), vec2(0.003), vec2(0.997));

  const refractedBg = vec3(
    viewportSharedTexture(uvR).r,
    viewportSharedTexture(uvG).g,
    viewportSharedTexture(uvB).b
  );

  // Borosilicate optical chord absorption: lossless crystal-clear dead center, subtle mineral-celadon depth in thick walls & heel
  const glassOpticalChord = float(0.025)
    .add(gEdge2.mul(gEdge).mul(0.62))
    .add(aGlassMeta.z.mul(0.22).mul(float(0.35).add(gEdge2.mul(0.65))))
    .add(aGlassMeta.y.mul(0.28));
  const boroAtten = exp(vec3(0.22, 0.085, 0.125).negate().mul(glassOpticalChord));

  // Classic dark-field glassware internal-refraction contrast line just inside the outer glass silhouette
  const innerDarkBand = smoothstep(0.58, 0.80, gEdge).mul(float(1.0).sub(smoothstep(0.84, 0.97, gEdge)));
  const glassTransmission = refractedBg
    .mul(boroAtten)
    .mul(float(1.0).sub(gFresnel.mul(0.72)))
    .mul(float(1.0).sub(innerDarkBand.mul(0.34)));

  // Restrained optical edge catches (letting physical PMREM GGX reflections drive the main curved strips cleanly)
  const gRefl = normalize(gView.negate().sub(gNorm.mul(dot(gView.negate(), gNorm).mul(2.0))));
  const stripLeft = smoothstep(0.955, 0.997, dot(gRefl, uLeftStripDir));
  const stripRight = smoothstep(0.96, 0.998, dot(gRefl, uRightStripDir)).mul(0.65);
  const keySpark = smoothstep(0.965, 0.998, dot(gRefl, uKeyDir)).mul(0.72);
  const rimWire = smoothstep(0.93, 0.995, dot(gRefl, uRimDir)).mul(0.50);

  const sharpEdgeSilhouette = smoothstep(0.78, 0.988, gEdge).mul(gEdge2);
  const lipHighlight = aGlassMeta.y.mul(float(0.14).add(keySpark.mul(0.55)).add(rimWire.mul(0.45)));
  const heelCausticRing = aGlassMeta.z
    .mul(smoothstep(0.62, 0.97, gEdge))
    .mul(float(0.06).add(gEdge2.mul(0.26)))
    .mul(mix(boroEdgeTint, uBathTint, clamp(uBathConcentration.mul(0.42), 0.0, 0.62)));

  // Zero diffuse chalkiness: 100% refracted backdrop + physical GGX PMREM reflections + crisp edge optics!
  beakerFrontMat.colorNode = vec3(0.0);
  beakerFrontMat.backdropNode = glassTransmission;
  beakerFrontMat.backdropAlphaNode = float(1.0);
  beakerFrontMat.opacityNode = float(1.0);
  beakerFrontMat.emissiveNode = calciteCol
    .mul(
      stripLeft
        .mul(0.10)
        .add(stripRight.mul(0.07))
        .add(lipHighlight.mul(0.34))
        .add(sharpEdgeSilhouette.mul(0.20))
    )
    .add(heelCausticRing.mul(0.32));

  // -------------------------------------------------------------------------
  // 2. Limpid Watertight Deformable Water Volume (Crystal Clear -> Rich Beer-Lambert Solute)
  // -------------------------------------------------------------------------
  const aWaterMeta = attribute('aWaterMeta', 'vec4'); // x=isTop, y=radialFrac, z=heightFrac, w=capillaryRim

  const waterMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.0,
    roughness: 0.032,
    ior: 1.333,
    reflectivity: 0.92,
    clearcoat: 1.0,
    clearcoatRoughness: 0.035,
    envMapIntensity: 1.65,
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

  // Perturb top free-surface normal via live wave gradients (dH/dx, dH/dz) so ripples steer Fresnel & softbox!
  const waterWaveSlopeX = waveSamp.g.mul(aWaterMeta.x).mul(1.08);
  const waterWaveSlopeZ = waveSamp.b.mul(aWaterMeta.x).mul(1.08);
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
  const soluteStrength = float(1.0).sub(exp(uBathConcentration.negate().mul(1.25)));

  const wRefl = normalize(wView.negate().sub(wNorm.mul(dot(wView.negate(), wNorm).mul(2.0))));
  const topSoftbox = smoothstep(0.80, 0.985, dot(wRefl, uKeyDir)).mul(aWaterMeta.x);
  const topRimBand = smoothstep(0.84, 0.985, dot(wRefl, uRimDir)).mul(aWaterMeta.x).mul(0.65);

  const rippleSlopeCue = clamp(waveSamp.g.mul(0.50).sub(waveSamp.b.mul(0.38)), -0.28, 0.28)
    .mul(aWaterMeta.x);

  // Authentic double-band concave optical meniscus right where water climbs the inner borosilicate glass wall:
  // thin dark refractive trough immediately below a razor-bright silvery capillary rim crest!
  const meniscusCrest = smoothstep(0.62, 0.98, aWaterMeta.w);
  const meniscusTrough = smoothstep(0.18, 0.56, aWaterMeta.w).mul(float(1.0).sub(meniscusCrest));

  const clearWaterHue = mix(celadonCol.mul(0.55), vec3(0.08, 0.11, 0.13), float(0.75));
  const tintedBathColor = mix(clearWaterHue, uBathTint, clamp(soluteStrength.mul(1.25), 0.0, 0.96))
    .mul(mix(vec3(1.0), bathTrans.mul(0.62).add(0.46), soluteStrength.mul(0.70)));

  const floorPoolRichness = float(1.0).sub(aWaterMeta.z).mul(soluteStrength).mul(0.30);
  const waterSurfaceCol = mix(
    tintedBathColor.mul(float(1.0).sub(meniscusTrough.mul(0.58))),
    calciteCol,
    clamp(wFresnel.mul(0.35).add(meniscusCrest.mul(0.82)).add(topSoftbox.mul(0.65)), 0.0, 0.94)
  ).add(calciteCol.mul(rippleSlopeCue.mul(0.16)));

  waterMat.colorNode = waterSurfaceCol;
  waterMat.emissiveNode = calciteCol
    .mul(meniscusCrest.mul(0.34).add(topSoftbox.mul(0.28)).add(topRimBand.mul(0.18)))
    .add(uBathTint.mul(soluteStrength.mul(0.075).add(floorPoolRichness.mul(0.09))));

  // Crystal-limpid water when pure (0.012 side baseline — ZERO grey fog!), deepening cleanly as pigment dissolves
  waterMat.opacityNode = clamp(
    float(0.012)
      .add(soluteStrength.mul(0.26))
      .add(floorPoolRichness.mul(0.16))
      .add(wFresnel.mul(0.18).mul(aWaterMeta.x.add(0.28)))
      .add(meniscusCrest.mul(0.46))
      .add(meniscusTrough.mul(0.28))
      .add(aWaterMeta.x.mul(0.032).add(topSoftbox.mul(0.26))),
    0.012,
    0.62
  );

  // -------------------------------------------------------------------------
  // 3. Rising Effervescent Silver Micro-Bubbles (High-Contrast TIR Dielectric Rim)
  // -------------------------------------------------------------------------
  const bubbleMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.12,
    roughness: 0.025,
    ior: 1.333,
    clearcoat: 1.0,
    clearcoatRoughness: 0.018,
    envMapIntensity: 2.2,
  });

  const bView = normalize(cameraPosition.sub(positionWorld));
  const bNorm = normalize(normalWorld);
  const bNdotV = clamp(abs(dot(bNorm, bView)), 0.0, 1.0);
  const bRim = float(1.0).sub(bNdotV);
  const bSilverRim = bRim.mul(bRim).mul(float(0.35).add(bRim.mul(0.65)));
  const bRefl = normalize(bView.negate().sub(bNorm.mul(dot(bView.negate(), bNorm).mul(2.0))));
  const bKeySpec = smoothstep(0.86, 0.988, dot(bRefl, uKeyDir));
  const bLeftSpec = smoothstep(0.86, 0.985, dot(bRefl, uLeftStripDir)).mul(0.85);

  const bubbleTint = mix(calciteCol, uBathTint, clamp(uBathConcentration.mul(0.20), 0.0, 0.30));
  bubbleMat.colorNode = mix(bubbleTint, calciteCol, clamp(bSilverRim.add(bKeySpec), 0.0, 1.0));
  bubbleMat.emissiveNode = calciteCol.mul(
    bSilverRim.mul(0.60).add(bKeySpec.mul(0.90)).add(bLeftSpec.mul(0.48))
  );
  bubbleMat.opacityNode = clamp(
    float(0.10).add(bSilverRim.mul(0.80)).add(bKeySpec.mul(0.68)).add(bLeftSpec.mul(0.35)),
    0.10,
    0.96
  );

  // -------------------------------------------------------------------------
  // 4. Solid Faceted Detaching Mineral Crumbs / Granules (Per-Instance Pigment Attribute)
  // -------------------------------------------------------------------------
  const aCrumbColor = attribute('aCrumbColor', 'vec3');
  const crumbMat = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.02,
    roughness: 0.34,
    ior: 1.54,
    clearcoat: 0.45,
    clearcoatRoughness: 0.24,
  });

  const crView = normalize(cameraPosition.sub(positionWorld));
  const crNorm = normalize(normalWorld);
  const crNdotV = clamp(abs(dot(crNorm, crView)), 0.0, 1.0);
  const crRim = float(1.0).sub(crNdotV);
  const crFacetFrost = crRim.mul(crRim).mul(0.32);

  // Wet mineral chip slightly deeper in core with crisp frosted grazing edges matching the eroding brick
  crumbMat.colorNode = mix(aCrumbColor.mul(0.88), calciteCol, crFacetFrost);
  crumbMat.emissiveNode = aCrumbColor.mul(float(0.14).add(crRim.mul(0.28)));

  // -------------------------------------------------------------------------
  // 5. Slender Tactile Borosilicate Glass Stirring Rod Material
  // -------------------------------------------------------------------------
  const rodMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    metalness: 0.04,
    roughness: 0.022,
    ior: 1.474,
    clearcoat: 1.0,
    clearcoatRoughness: 0.02,
    envMapIntensity: 2.1,
  });

  const rView = normalize(cameraPosition.sub(positionWorld));
  const rNorm = normalize(normalWorld);
  const rNdotV = clamp(abs(dot(rNorm, rView)), 0.0, 1.0);
  const rEdge = float(1.0).sub(rNdotV);
  const rFresnel = float(0.06).add(float(0.94).mul(rEdge.mul(rEdge).mul(rEdge)));
  const rRefl = normalize(rView.negate().sub(rNorm.mul(dot(rView.negate(), rNorm).mul(2.0))));
  const rHighlight = smoothstep(0.86, 0.99, dot(rRefl, uKeyDir))
    .add(smoothstep(0.88, 0.99, dot(rRefl, uLeftStripDir)).mul(0.82));

  rodMat.colorNode = mix(boroEdgeTint, calciteCol, clamp(rFresnel.add(rHighlight), 0.0, 1.0));
  rodMat.emissiveNode = calciteCol.mul(rFresnel.mul(0.22).add(rHighlight.mul(0.55)));
  rodMat.opacityNode = clamp(
    float(0.08).add(rFresnel.mul(0.68)).add(rHighlight.mul(0.38)),
    0.0,
    0.92
  ).mul(uRodOpacity);

  // -------------------------------------------------------------------------
  // 6. High-Resolution Fired-Enamel Volumetric Markings Material
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
  markingsMat.opacityNode = markSamp.a.mul(0.84);

  // -------------------------------------------------------------------------
  // 7. Focused Refracted Floor Caustic Filaments (ZERO Milky Disc Washout)
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
      positionLocal.x.mul(4.4).add(cWave.g.mul(2.8)),
      uTime.mul(0.32),
      positionLocal.z.mul(4.4).add(cWave.b.mul(2.8))
    ),
    float(1.8)
  );
  // Crisp thin wall caustic ring + high-threshold bright ripple filaments ONLY (zero flat floor fog!)
  const wallCausticRing = smoothstep(0.80, 0.955, cRad).mul(smoothstep(1.0, 0.955, cRad));
  const causticFilaments = smoothstep(0.64, 0.88, cCausticNoise)
    .mul(float(0.22).add(cSlopeMag.mul(2.8)));
  const causticEnergy = causticFilaments.add(wallCausticRing.mul(0.62));
  const floorEdgeFade = float(1.0).sub(smoothstep(0.95, 1.0, cRad));

  const floorTint = mix(calciteCol, uBathTint, clamp(uBathConcentration.mul(0.68), 0.0, 0.88));
  causticDiscMat.colorNode = mix(floorTint, calciteCol, clamp(causticEnergy.mul(0.70), 0.0, 0.90));
  causticDiscMat.opacityNode = clamp(causticEnergy.mul(0.24), 0.0, 0.30).mul(floorEdgeFade);

  // -------------------------------------------------------------------------
  // 8. Four Dual-Layer Dissolving Crystalline Mineral Cube Material Pairs
  // -------------------------------------------------------------------------
  const cubeHandles: CrystalCubeMaterialHandle[] = [];
  for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
    cubeHandles.push(createDissolvingCrystalPair('celadonGlauconite'));
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
    beakerBackOuterMat.dispose();
    beakerBackInnerMat.dispose();
    beakerNearInnerMat.dispose();
    beakerFrontMat.dispose();
    waterMat.dispose();
    bubbleMat.dispose();
    crumbMat.dispose();
    rodMat.dispose();
    markingsMat.dispose();
    causticDiscMat.dispose();
    for (let i = 0; i < cubeHandles.length; i++) {
      cubeHandles[i].dispose();
    }
  };

  return {
    beakerBackOuterMat,
    beakerBackInnerMat,
    beakerNearInnerMat,
    beakerFrontMat,
    waterMat,
    bubbleMat,
    crumbMat,
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
