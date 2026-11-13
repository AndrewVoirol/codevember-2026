import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  dot,
  float,
  max,
  min,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  select,
  sin,
  smoothstep,
  sqrt,
  texture,
  uniform,
  uv,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import { CHALKY_SLATE_HEX, EGGSHELL_INK_HEX } from '../../core/palette';
import {
  beerLambertTransmittance,
  chalkySlateBackdropNode,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  DeformableWaterPrism,
  DelaminatingRustFlakeSystem,
  FlocculeMoteSystem,
  SPIKE_MAX_X,
  SPIKE_MIN_X,
  SpikeScrapeWearField,
  TANK_BASE_YAW,
  TANK_INNER_X,
  TANK_INNER_Y_BOT,
  TANK_INNER_Y_TOP,
  TANK_INNER_Z,
  TANK_OUTER_X,
  TANK_OUTER_Y,
  TANK_OUTER_Z,
  TankHydroSim,
  createBeveledOpticalTankGlassGeometry,
  createForgedShipwreckSpikeGeometry,
  createOsmoticRusticlesGeometry,
  createOsmoticSiltVeilsGeometry,
  evalSpikeSpine,
} from './shipwreckApparatus';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Day 11 — Rust:
 * "Submerged Shipwreck Spike & Osmotic Rusticles"
 *
 * Physical & Optical Architecture:
 * 1. Multi-Pane Snell Virtual-Camera Refraction + Total Internal Reflection (TIR) Water Cube:
 *    - Renders the submerged forged wrought-iron hull spike, coalesced Halomonas titanicae
 *      coral-drapery "rusticles", delaminating 3D rust chips, and gossamer ferric-hydroxide
 *      silt veils into per-face 4x-MSAA HalfFloat offscreen targets (`rtPaneZ`, `rtPaneX`,
 *      `rtPaneY`) aligned with each visible glass pane's Snell-refracted optical axis.
 *    - Eliminates oblique 2D screen-projection taffy stretching across 360° orbit while
 *      preserving 3-band Cauchy dispersion (`n = 1.331, 1.336, 1.342`) across the 90°
 *      vertical corner seam and Total Internal Reflection (`48.6°`) on grazing inner walls.
 * 2. Authentic Hand-Forged Square Wrought-Iron Spike & 5-Blow Faceted Rose-Head:
 *    - Neutral dark gunmetal magnetite (`Fe3O4`) with longitudinal silicate slag grain,
 *      stepped spall craters (`hematiteRust` / `burntSienna`), and lower-belly / rim
 *      `realgarAmber` + `yellowOchre` concretion where the hanging rusticle curtains nucleate.
 * 3. Tactile Brittle Rust Delamination & Underwater Stokes Leaf-Flutter:
 *    - Brushing, dragging, sloshing, or tapping the optical tank peels bursts of ragged
 *      curved 3D iron-oxide chips (`640` instances with dark dorsal scale & freshly cleaved
 *      ochre undersides) plus blooming ferric silt puffs that flutter and settle onto the
 *      bottom glass plate.
 */

const SPECIMEN_CENTER_LOCAL = new THREE.Vector3(-0.04, -0.04, -0.045);
const WATER_IOR_REF = 1.336;
const PANE_RT_SCALE = 0.88;

// Pre-allocated module-scope scratch objects (strictly zero per-frame GC allocations)
const _slateClearColor = new THREE.Color(CHALKY_SLATE_HEX);
const _invWorldMat = new THREE.Matrix4();
const _raycaster = new THREE.Raycaster();
const _ndcVec2 = new THREE.Vector2();
const _tankBoxLocal = new THREE.Box3(
  new THREE.Vector3(-TANK_OUTER_X, -TANK_OUTER_Y, -TANK_OUTER_Z),
  new THREE.Vector3(TANK_OUTER_X, TANK_OUTER_Y, TANK_OUTER_Z)
);
const _localRay = new THREE.Ray();
const _localHit = new THREE.Vector3();
const _specimenHit = new THREE.Vector3();
const _nailHitPt = new THREE.Vector3();
const _nailHitNorm = new THREE.Vector3(0, 1, 0);
const _prevPointerLocal = new THREE.Vector3();
const _curPointerLocal = new THREE.Vector3();
const _flakeImpulse = new THREE.Vector3();
const _prevCamDir = new THREE.Vector3();
const _curCamDir = new THREE.Vector3();

const _camLocal = new THREE.Vector3();
const _paneAnchorLocal = new THREE.Vector3();
const _paneNormalLocal = new THREE.Vector3();
const _incDirLocal = new THREE.Vector3();
const _refrDirLocal = new THREE.Vector3();
const _virtPosLocal = new THREE.Vector3();
const _virtPosWorld = new THREE.Vector3();
const _targetWorld = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _altUp = new THREE.Vector3(0, 0, -1);
const _waterRayOrigin = new THREE.Vector3();
const _waterRayDir = new THREE.Vector3();
const _hitNormalLocal = new THREE.Vector3();

/**
 * Exact 3D Snell refraction vector on CPU (`eta = n1 / n2`), writing into `out`.
 */
function refractDirCpu(
  inc: THREE.Vector3,
  normal: THREE.Vector3,
  eta: number,
  out: THREE.Vector3
): THREE.Vector3 {
  const cosI = Math.max(0.0, Math.min(1.0, -inc.dot(normal)));
  const sin2T = eta * eta * (1.0 - cosI * cosI);
  const cosT = Math.sqrt(Math.max(1e-5, 1.0 - sin2T));
  return out
    .copy(inc)
    .multiplyScalar(eta)
    .addScaledVector(normal, eta * cosI - cosT)
    .normalize();
}

export async function bootDay11Rust(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(11);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.02, 4.26],
    fov: 33,
    enableShadows: true,
  });

  stage.controls.target.set(0.0, -0.02, 0.0);
  stage.controls.minDistance = 2.35;
  stage.controls.maxDistance = 7.0;
  stage.controls.minPolarAngle = 0.48;
  stage.controls.maxPolarAngle = 2.38;
  stage.controls.rotateSpeed = 0.58;
  stage.controls.update();
  _prevCamDir.copy(stage.camera.position).normalize();

  // -------------------------------------------------------------------------
  // Mineral Pigment Uniform Packs (Strictly Zero Digital RGB Neons)
  // -------------------------------------------------------------------------
  const hematite = createPigmentUniforms('hematiteRust');
  const sienna = createPigmentUniforms('burntSienna');
  const ochre = createPigmentUniforms('yellowOchre');
  const realgar = createPigmentUniforms('realgarAmber');
  const umber = createPigmentUniforms('rawUmber');
  const celadon = createPigmentUniforms('celadonGlauconite');
  const calcite = createPigmentUniforms('boneCalcite');

  // -------------------------------------------------------------------------
  // Coupled Hydrodynamic Simulation & Per-Pane Snell Virtual Camera Uniforms
  // -------------------------------------------------------------------------
  const hydroSim = new TankHydroSim();

  const uTankLocalToWorld = uniform(new THREE.Matrix4());
  const uTankWorldToLocal = uniform(new THREE.Matrix4());
  const uSloshTilt = uniform(new THREE.Vector2(0, 0));
  const uSwayMode1 = uniform(new THREE.Vector2(0, 0));
  const uSwayMode2 = uniform(new THREE.Vector2(0, 0));
  const uStirLocalPos = uniform(new THREE.Vector3(0, 0, 0));
  const uStirMeta = uniform(new THREE.Vector3(0.06, 0, 0));

  const uViewProjZ = uniform(new THREE.Matrix4());
  const uViewProjX = uniform(new THREE.Matrix4());
  const uViewProjY = uniform(new THREE.Matrix4());

  const uCamDirZ = uniform(new THREE.Vector3(0, 0, -1));
  const uCamDirX = uniform(new THREE.Vector3(-1, 0, 0));
  const uCamDirY = uniform(new THREE.Vector3(0, -1, 0));

  const uKeyLightDir = uniform(new THREE.Vector3(2.6, 4.2, 3.1).normalize());
  const uRimLightDir = uniform(new THREE.Vector3(-1.6, 3.1, -3.5).normalize());

  // -------------------------------------------------------------------------
  // Offscreen 4x-MSAA HalfFloat Per-Pane RenderTargets & Virtual Cameras
  // -------------------------------------------------------------------------
  const getTargetSize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * PANE_RT_SCALE;
    return {
      w: Math.max(2, Math.floor(window.innerWidth * dpr)),
      h: Math.max(2, Math.floor(window.innerHeight * dpr)),
    };
  };

  const makePaneRT = (w: number, h: number) =>
    new THREE.RenderTarget(w, h, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      stencilBuffer: false,
      samples: 4,
    });

  const initSize = getTargetSize();
  const rtPaneZ = makePaneRT(initSize.w, initSize.h);
  const rtPaneX = makePaneRT(initSize.w, initSize.h);
  const rtPaneY = makePaneRT(initSize.w, initSize.h);

  const camPaneZ = new THREE.PerspectiveCamera(33, stage.camera.aspect, 0.1, 30);
  const camPaneX = new THREE.PerspectiveCamera(33, stage.camera.aspect, 0.1, 30);
  const camPaneY = new THREE.PerspectiveCamera(33, stage.camera.aspect, 0.1, 30);

  const interiorScene = new THREE.Scene();
  interiorScene.background = null;

  // Crisp 3-point museum vitrine illumination inside the optical water chamber
  const intHemi = new THREE.HemisphereLight(EGGSHELL_INK_HEX, CHALKY_SLATE_HEX, 0.56);
  interiorScene.add(intHemi);

  const intKey = new THREE.DirectionalLight(0xf8f1e4, 3.15);
  intKey.position.set(2.4, 4.3, 3.0);
  intKey.castShadow = true;
  intKey.shadow.autoUpdate = false;
  intKey.shadow.needsUpdate = true;
  intKey.shadow.mapSize.set(1024, 1024);
  intKey.shadow.camera.near = 1.2;
  intKey.shadow.camera.far = 9.5;
  const shExtent = 1.25;
  intKey.shadow.camera.left = -shExtent;
  intKey.shadow.camera.right = shExtent;
  intKey.shadow.camera.top = shExtent;
  intKey.shadow.camera.bottom = -shExtent;
  intKey.shadow.bias = -0.0004;
  interiorScene.add(intKey);

  const intFill = new THREE.DirectionalLight(0x98a8b2, 0.78);
  intFill.position.set(-3.2, -0.3, 2.3);
  interiorScene.add(intFill);

  const intFloorBounce = new THREE.DirectionalLight(0xeae3d2, 0.65);
  intFloorBounce.position.set(0.2, -3.4, 1.5);
  interiorScene.add(intFloorBounce);

  const intRim = new THREE.DirectionalLight(0xf3efe6, 1.65);
  intRim.position.set(-1.6, 3.0, -3.4);
  interiorScene.add(intRim);

  const interiorTankGroup = new THREE.Group();
  interiorTankGroup.rotation.y = TANK_BASE_YAW;
  interiorScene.add(interiorTankGroup);

  const exteriorTankGroup = new THREE.Group();
  exteriorTankGroup.rotation.y = TANK_BASE_YAW;
  stage.specimenGroup.add(exteriorTankGroup);

  // Derivative-free 3D FBM micro-relief normal perturbation helper (uniform control flow safe)
  const perturbNormal3D = Fn(
    ([baseNorm_in, pos3D_in, freq_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
      const p = vec3(pos3D_in);
      const freq = float(freq_in);
      const eps = float(0.0052);
      const h0 = mineralToothFbm(p, freq);
      const hx = mineralToothFbm(p.add(vec3(eps, 0.0, 0.0)), freq);
      const hy = mineralToothFbm(p.add(vec3(0.0, eps, 0.0)), freq);
      const hz = mineralToothFbm(p.add(vec3(0.0, 0.0, eps)), freq);
      const grad = vec3(hx.sub(h0), hy.sub(h0), hz.sub(h0)).div(eps);
      return normalize(vec3(baseNorm_in).sub(grad.mul(float(strength_in))));
    }
  );

  // -------------------------------------------------------------------------
  // 1. Hand-Forged Square Wrought-Iron Shipwreck Spike & 5-Blow Rose-Head
  // -------------------------------------------------------------------------
  const { geometry: spikeGeom, flakeSites } = createForgedShipwreckSpikeGeometry();
  const rusticlesGeom = createOsmoticRusticlesGeometry(flakeSites);
  const scrapeField = new SpikeScrapeWearField();
  const uHoverNailLocal = uniform(new THREE.Vector3(0, 0, 0));
  const uHoverNailActive = uniform(0.0);

  const spikeMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.14,
  });

  // aCorr: x = spallRustMask, y = cavityAO, z = slagFiber, w = bellyOchreMask
  const aCorr = attribute('aCorrosionData', 'vec4');
  const spikePos = positionLocal;

  // Unconditional uniform-control-flow sample of live 128x64 `(uSpatial, vTheta)` rust-scrape wear map
  const scrapeSample = texture(scrapeField.dataTexture, uv()).level(float(0.0));
  const rawWear = scrapeSample.r;
  const freshScrapeGlint = scrapeSample.g;

  const slagGrain = mineralToothFbm(spikePos.mul(vec3(3.4, 38.0, 38.0)), float(2.2));
  const crustDetail = mineralToothFbm(spikePos.mul(vec3(18.0, 24.0, 24.0)), float(3.6));

  const crustPlates = mineralToothFbm(spikePos.mul(vec3(9.5, 13.5, 13.5)), float(2.6));

  // Brittle spall-plate scrape mask: pops off raised crust plates in irregular jagged islands while leaving oxide in deep pits!
  const pitRetention = smoothstep(float(0.28), float(0.66), aCorr.y);
  const chippedWearField = rawWear
    .mul(float(0.22).add(crustPlates.mul(0.82)).add(crustDetail.mul(0.54)).add(slagGrain.sub(0.5).mul(0.28)))
    .mul(float(0.45).add(pitRetention.mul(0.65)));
  const effectiveWear = smoothstep(float(0.28), float(0.62), chippedWearField);
  const intactCrustKeep = float(1.0).sub(effectiveWear);

  // Tier 1: Intact forged wrought-iron / magnetite mill-scale (`Fe3O4`) + freshly scraped iron sheen
  const fiberLift = smoothstep(float(0.28), float(0.84), aCorr.z.mul(0.62).add(slagGrain.mul(0.38)));
  const gunmetalMagnetite = mix(
    mix(umber.rInf.mul(0.30), calcite.rInf.mul(0.045), float(0.32)),
    mix(umber.rInf.mul(0.46), calcite.rInf.mul(0.085), float(0.36)),
    fiberLift
  );
  const scrapedIronSteel = mix(
    gunmetalMagnetite.mul(0.82),
    mix(calcite.rInf.mul(0.24), umber.rInf.mul(0.32), float(0.32)),
    freshScrapeGlint.mul(0.72).add(fiberLift.mul(0.25))
  );
  const bareIronBase = mix(gunmetalMagnetite, scrapedIronSteel, effectiveWear);

  // Tier 2: Earthy roasted umber-sienna corrosion crusts (peeled off cleanly where scraped!)
  const spallEdge = smoothstep(float(0.24), float(0.74), aCorr.x.add(crustDetail.sub(0.5).mul(0.26))).mul(
    intactCrustKeep
  );
  const darkOxideBed = kubelkaMunkGlaze(
    mix(umber.K, sienna.K, float(0.54)),
    mix(umber.S, sienna.S, float(0.54)),
    spallEdge.mul(1.18),
    bareIronBase
  );

  // Tier 3: Localized golden-ochre concretion strictly at rusticle holdfasts (zero continuous yellow belly stripe!)
  // + crisp dusty ochre snapped fracture rim outlining scraped bare-iron scars!
  const fractureRim = smoothstep(float(0.05), float(0.32), rawWear)
    .mul(intactCrustKeep)
    .mul(smoothstep(float(0.16), float(0.62), aCorr.x));
  const neckWindowKeep = float(1.0).sub(
    smoothstep(float(0.17), float(0.23), spikePos.x).mul(smoothstep(float(0.37), float(0.32), spikePos.x))
  );
  const bolsterKeep = smoothstep(float(0.46), float(0.39), spikePos.x);
  const amberSeamMask = smoothstep(float(0.64), float(0.98), aCorr.w.add(crustDetail.sub(0.48).mul(0.44)))
    .mul(neckWindowKeep)
    .mul(bolsterKeep)
    .mul(intactCrustKeep)
    .add(fractureRim.mul(0.95));

  // Clear warm fracture-edge highlight right under the cursor on unscraped rust crust!
  const toHoverNail = spikePos.sub(uHoverNailLocal);
  const hoverDistSq = dot(toHoverNail, toHoverNail);
  const hoverRing = smoothstep(float(0.015), float(0.0012), hoverDistSq)
    .mul(uHoverNailActive)
    .mul(intactCrustKeep)
    .mul(smoothstep(float(0.24), float(0.68), crustDetail));

  const forgedIronFinal = kubelkaMunkGlaze(
    mix(ochre.K, realgar.K, float(0.38)),
    mix(ochre.S, realgar.S, float(0.38)),
    clamp(amberSeamMask.mul(1.25).add(hoverRing.mul(0.85)), 0.0, 1.65),
    darkOxideBed
  );

  const spikeAO = mix(float(0.24), float(1.06), mix(aCorr.y, float(0.94), effectiveWear.mul(0.60)));
  spikeMat.colorNode = forgedIronFinal.mul(spikeAO);
  spikeMat.metalnessNode = mix(float(0.10), float(0.74), effectiveWear);
  spikeMat.normalNode = perturbNormal3D(
    normalLocal,
    spikePos.mul(2.1),
    float(20.0),
    mix(float(0.020), float(0.007), effectiveWear)
  );
  spikeMat.roughnessNode = clamp(
    mix(
      mix(float(0.60), float(0.92), spallEdge).sub(fiberLift.mul(0.04)),
      float(0.34),
      effectiveWear.mul(float(0.58).add(freshScrapeGlint.mul(0.42)))
    ),
    0.32,
    0.95
  );

  const spikeMesh = new THREE.Mesh(spikeGeom, spikeMat);
  spikeMesh.castShadow = true;
  spikeMesh.receiveShadow = true;
  spikeMesh.renderOrder = 1;
  interiorTankGroup.add(spikeMesh);

  // -------------------------------------------------------------------------
  // 2. Reference-Matched Craggy Porous Coral-Sponge "Rusticles"
  // -------------------------------------------------------------------------
  const rusticleMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.01,
  });

  const aKin = attribute('aRusticleKinematics', 'vec4'); // x: vDrop, y: colLen, z: seed, w: poreAO
  const vDrop = aKin.x;
  const colLen = aKin.y;
  const colSeed = aKin.z;

  const bend1Weight = vDrop.pow(1.75).mul(colLen.mul(0.92).add(0.20));
  const bend2Weight = sin(vDrop.mul(3.14159)).mul(vDrop).mul(colLen);

  const kelpyPhaseX = sin(stage.uTime.mul(1.28).sub(vDrop.mul(2.6)).add(colSeed))
    .mul(0.0075)
    .mul(bend1Weight);
  const kelpyPhaseZ = cos(stage.uTime.mul(1.04).sub(vDrop.mul(2.3)).add(colSeed.mul(1.3)))
    .mul(0.0068)
    .mul(bend1Weight);

  const toStir = positionLocal.sub(uStirLocalPos);
  const stirDistSq = dot(toStir, toStir).add(0.14);
  const localEddyWeight = uStirMeta.x.sub(0.06).div(stirDistSq).mul(bend1Weight).mul(0.065);

  const swayOffsetX = uSwayMode1.x
    .mul(bend1Weight)
    .add(uSwayMode2.x.mul(bend2Weight))
    .add(kelpyPhaseX)
    .add(uStirMeta.y.mul(localEddyWeight));
  const swayOffsetZ = uSwayMode1.y
    .mul(bend1Weight)
    .add(uSwayMode2.y.mul(bend2Weight))
    .add(kelpyPhaseZ)
    .add(uStirMeta.z.mul(localEddyWeight));

  const swayMagSq = swayOffsetX.mul(swayOffsetX).add(swayOffsetZ.mul(swayOffsetZ));
  const arcLiftY = swayMagSq.div(max(float(0.12), colLen.mul(2.0))).mul(0.45);

  rusticleMat.positionNode = positionLocal.add(vec3(swayOffsetX, arcLiftY, swayOffsetZ));

  const dBendDy = vDrop.pow(0.75).mul(1.6);
  const swayedNorm = normalize(
    normalLocal.add(vec3(uSwayMode1.x.mul(dBendDy).mul(0.32), float(0.0), uSwayMode1.y.mul(dBendDy).mul(0.32)))
  );
  // Crisp sub-millimeter porous coral/sponge normal relief
  rusticleMat.normalNode = perturbNormal3D(swayedNorm, positionLocal.mul(3.2), float(34.0), float(0.036));

  const rLocal = positionLocal;
  const spongeMacro = mineralToothFbm(rLocal.mul(vec3(18.0, 22.0, 18.0)), float(2.8));
  const spongeFine = mineralToothFbm(rLocal.mul(62.0), float(4.8));
  const microPits = smoothstep(float(0.50), float(0.18), spongeFine);
  const poreLight = clamp(aKin.w.sub(microPits.mul(0.48)), 0.06, 1.0);

  // Depth-layer cue separating bright golden front curtains from deep burnt-sienna rear blades
  const zDepthShade = smoothstep(float(-0.068), float(-0.020), rLocal.z);

  const poreShadow = mix(umber.rInf.mul(0.16), mix(sienna.rInf, hematite.rInf, float(0.45)).mul(0.38), float(0.52));
  const siennaMid = kubelkaMunkGlaze(
    mix(sienna.K, realgar.K, float(0.48)),
    mix(sienna.S, realgar.S, float(0.48)),
    clamp(poreLight.mul(1.22).add(spongeMacro.mul(0.28)), 0.08, 1.60),
    poreShadow
  );
  const goldenCoral = kubelkaMunkGlaze(
    mix(ochre.K, realgar.K, float(0.32)),
    mix(ochre.S, realgar.S, float(0.32)),
    smoothstep(float(0.32), float(0.86), poreLight.add(spongeMacro.mul(0.20))).mul(1.42),
    siennaMid
  );

  const rustViewDir = normalize(cameraPosition.sub(positionWorld));
  const rustNormW = normalize(normalWorld);
  const rustRim = float(1.0).sub(clamp(abs(dot(rustNormW, rustViewDir)), 0.0, 1.0)).pow(2.4);
  const warmFlocGlow = mix(ochre.rInf, realgar.rInf, float(0.42))
    .mul(rustRim)
    .mul(poreLight)
    .mul(0.20);

  rusticleMat.colorNode = goldenCoral
    .mul(mix(float(0.16), float(1.14), poreLight))
    .mul(mix(float(0.68), float(1.04), zDepthShade));
  rusticleMat.emissiveNode = warmFlocGlow;
  rusticleMat.roughnessNode = clamp(float(0.95).sub(spongeFine.mul(0.08)), 0.84, 0.98);

  const rusticlesMesh = new THREE.Mesh(rusticlesGeom, rusticleMat);
  rusticlesMesh.castShadow = true;
  // Analytical `softBellyAO` + `poreAO` avoids harsh directional shadow cutoff band across upper rusticles!
  rusticlesMesh.receiveShadow = false;
  rusticlesMesh.renderOrder = 2;
  interiorTankGroup.add(rusticlesMesh);

  // -------------------------------------------------------------------------
  // 3. Interactive 3D Flaky Rust-Bark Scales, Jagged Shards & Crumble Grit
  // -------------------------------------------------------------------------
  const flakeMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    side: THREE.DoubleSide,
  });

  // aFlakeVert: x = faceSide (1.0 outer crust, 0.0 inner cleaved), y = fractureEdgeMask [0..1]
  // aFlakeInst: x = seed, y = curlArch, z = toneMix, w = hoverHighlight
  const aFlakeVert = attribute('aFlakeVert', 'vec2');
  const aFlakeInst = attribute('aFlakeInstance', 'vec4');

  const fSeed = aFlakeInst.x;
  const fTone = aFlakeInst.z;
  const fHover = aFlakeInst.w;

  const flakeGrain = mineralToothFbm(
    positionWorld.mul(44.0).add(vec3(fSeed, 0.0, fSeed.mul(0.61))),
    float(3.4)
  );
  const darkIronPatch = smoothstep(float(0.56), float(0.28), flakeGrain);

  // Multi-toned crusty iron-oxide bark (matches the nail's dark magnetite + burnt-sienna + ochre crust!)
  const darkScaleSubstrate = mix(
    umber.rInf.mul(0.14),
    mix(hematite.rInf, umber.rInf, float(0.62)).mul(0.28),
    fTone
  );
  const dorsalBarkCol = kubelkaMunkGlaze(
    mix(umber.K, sienna.K, fTone.mul(0.65).add(0.22)),
    mix(umber.S, sienna.S, fTone.mul(0.65).add(0.22)),
    clamp(float(1.0).sub(darkIronPatch).mul(1.18), 0.04, 1.15),
    darkScaleSubstrate
  );
  const ventralRustyCol = kubelkaMunkGlaze(
    mix(sienna.K, ochre.K, clamp(flakeGrain.mul(0.75), 0.10, 0.68)),
    mix(sienna.S, ochre.S, clamp(flakeGrain.mul(0.75), 0.10, 0.68)),
    clamp(flakeGrain.mul(1.12).add(0.18), 0.16, 1.25),
    umber.rInf.mul(0.18)
  );
  const baseFlakeCol = mix(
    ventralRustyCol,
    dorsalBarkCol,
    smoothstep(float(0.22), float(0.78), aFlakeVert.x)
  );

  // Dusty golden-ochre fractured perimeter rim (brightens when hovering pries a nail scab open!)
  const fractureEdgeTint = mix(sienna.rInf, mix(ochre.rInf, realgar.rInf, float(0.38)), float(0.76));
  const rimBlend = clamp(
    smoothstep(float(0.42), float(0.98), aFlakeVert.y)
      .mul(flakeGrain.add(0.22))
      .mul(float(0.52).add(fHover.mul(0.48))),
    0.0,
    0.90
  );

  flakeMat.colorNode = mix(baseFlakeCol, fractureEdgeTint, rimBlend);
  flakeMat.normalNode = perturbNormal3D(
    normalLocal,
    positionWorld.mul(38.0).add(vec3(fSeed, fSeed, 0.0)),
    float(28.0),
    float(0.055)
  );
  flakeMat.emissiveNode = mix(ochre.rInf, realgar.rInf, float(0.45))
    .mul(fHover)
    .mul(smoothstep(float(0.35), float(1.0), aFlakeVert.y))
    .mul(0.22);
  flakeMat.roughnessNode = clamp(mix(float(0.94), float(0.85), aFlakeVert.x), 0.82, 0.96);

  const flakeSystem = new DelaminatingRustFlakeSystem(flakeMat, flakeSites);
  flakeSystem.mesh.renderOrder = 2;
  interiorTankGroup.add(flakeSystem.mesh);

  // -------------------------------------------------------------------------
  // 4. Gossamer Osmotic Ferric-Hydroxide Silt Ribbons & Colloidal Puff Motes
  // -------------------------------------------------------------------------
  const siltVeilsGeom = createOsmoticSiltVeilsGeometry();
  const siltVeilMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  const aVeil = attribute('aVeilMeta', 'vec4');
  const uAlong = aVeil.x;
  const vAcross = aVeil.y;
  const vSeed = aVeil.z;

  const veilWaveX = sin(uAlong.mul(4.4).sub(stage.uTime.mul(0.42)).add(vSeed))
    .mul(0.034)
    .mul(uAlong)
    .add(uSwayMode1.x.mul(uAlong).mul(0.62));
  const veilWaveY = cos(uAlong.mul(3.4).sub(stage.uTime.mul(0.35)).add(vSeed.mul(1.7)))
    .mul(0.020)
    .mul(uAlong);
  const veilWaveZ = cos(uAlong.mul(4.8).sub(stage.uTime.mul(0.38)).add(vSeed.mul(0.8)))
    .mul(0.036)
    .mul(uAlong)
    .add(uSwayMode1.y.mul(uAlong).mul(0.62));

  siltVeilMat.positionNode = positionLocal.add(vec3(veilWaveX, veilWaveY, veilWaveZ));

  const veilView = normalize(cameraPosition.sub(positionWorld));
  const veilNorm = normalize(normalWorld);
  const veilNdotV = abs(dot(veilNorm, veilView));
  const veilGrazingFade = smoothstep(float(0.05), float(0.38), veilNdotV);
  // Gossamer folded-silk limb accent (creates delicate golden smoke-fold filaments like the reference!)
  const silkFoldBoost = float(1.0).add(
    float(1.0).sub(veilNdotV).pow(1.6).mul(smoothstep(float(0.03), float(0.18), veilNdotV)).mul(0.85)
  );

  const axialFade = smoothstep(float(0.0), float(0.22), uAlong).mul(
    smoothstep(float(1.0), float(0.32), uAlong)
  );
  const crossBell = sin(vAcross.mul(3.14159)).pow(1.85);

  const wispNoise = mineralToothFbm(
    vec3(uAlong.mul(2.4).sub(stage.uTime.mul(0.07)), vAcross.mul(3.2), vSeed),
    float(2.0)
  );
  const wispSoft = smoothstep(float(0.22), float(0.76), wispNoise);

  const siltTint = mix(ochre.rInf, realgar.rInf, float(0.26)).mul(1.02);
  const stirBoost = clamp(float(1.0).add(uStirMeta.x.mul(0.38)), 1.0, 1.5);
  const veilAlpha = clamp(
    aVeil.w.mul(axialFade).mul(crossBell).mul(veilGrazingFade).mul(silkFoldBoost).mul(wispSoft).mul(stirBoost),
    0.0,
    0.22
  );

  siltVeilMat.colorNode = siltTint;
  siltVeilMat.opacityNode = veilAlpha;

  const siltVeilsMesh = new THREE.Mesh(siltVeilsGeom, siltVeilMat);
  siltVeilsMesh.renderOrder = 4;
  interiorTankGroup.add(siltVeilsMesh);

  const moteMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  });
  moteMat.colorNode = mix(sienna.rInf, ochre.rInf, float(0.55)).mul(0.82);
  moteMat.opacityNode = float(0.36);

  const moteSystem = new FlocculeMoteSystem(moteMat);
  moteSystem.mesh.renderOrder = 3;
  interiorTankGroup.add(moteSystem.mesh);

  // -------------------------------------------------------------------------
  // 5. Exterior Pass: Unified Deformable Multi-Pane Snell + TIR Water Prism
  // -------------------------------------------------------------------------
  const safeDiv = Fn(([num_in, den_in]: [TslNode, TslNode]) => {
    const den = float(den_in);
    const sgn = select(den.greaterThanEqual(0.0), float(1.0), float(-1.0));
    return float(num_in).div(sgn.mul(max(float(1e-4), abs(den))));
  });

  const projectLocalToPaneUv = Fn(([pLocal_in, viewProjMat_in]: [TslNode, TslNode]) => {
    const pWorld = uTankLocalToWorld.mul(vec4(vec3(pLocal_in), 1.0));
    const clip = viewProjMat_in.mul(pWorld);
    const wSafe = max(float(0.05), clip.w);
    const ndc = clip.xy.div(wSafe);
    return clamp(vec2(ndc.x.mul(0.5).add(0.5), float(0.5).sub(ndc.y.mul(0.5))), 0.002, 0.998);
  });

  // Sample from all 3 pane RenderTargets unconditionally in uniform control flow, then select active pane
  const samplePaneRT = Fn(
    ([pLocal_in, isXPane_in, isYPane_in]: [TslNode, TslNode, TslNode]) => {
      const uvZ = projectLocalToPaneUv(pLocal_in, uViewProjZ);
      const uvX = projectLocalToPaneUv(pLocal_in, uViewProjX);
      const uvY = projectLocalToPaneUv(pLocal_in, uViewProjY);

      const sZ = texture(rtPaneZ.texture, uvZ).level(float(0.0));
      const sX = texture(rtPaneX.texture, uvX).level(float(0.0));
      const sY = texture(rtPaneY.texture, uvY).level(float(0.0));

      return select(isXPane_in, sX, select(isYPane_in, sY, sZ));
    }
  );

  /**
   * 360°-Continuous Multi-Pane Snell + TIR Water-Cube Ray Tracer.
   */
  const sampleRefractedBand = Fn(
    ([
      pEntry_in,
      nEntry_in,
      incLocal_in,
      ior_in,
      isXPane_in,
      isYPane_in,
      paneCamDir_in,
    ]: [TslNode, TslNode, TslNode, TslNode, TslNode, TslNode, TslNode]) => {
      const p0 = vec3(pEntry_in);
      const nEntry = normalize(vec3(nEntry_in));
      const inc = normalize(vec3(incLocal_in));
      const ior = float(ior_in);
      const eta = float(1.0).div(ior);

      const cosI = clamp(dot(nEntry, inc).negate(), 0.0, 1.0);
      const sin2T = eta.mul(eta).mul(float(1.0).sub(cosI.mul(cosI)));
      const cosT = sqrt(max(float(1e-4), float(1.0).sub(sin2T)));
      const dWater = normalize(inc.mul(eta).add(nEntry.mul(eta.mul(cosI).sub(cosT))));

      // Parallel soda-lime float-glass lateral shift on vertical/bottom glass walls ONLY (zero shift on open water top!),
      // clamped strictly inside the inner water chamber so perimeter rays never clip prematurely!
      const etaGlass = float(1.0 / 1.52);
      const cosGlass = sqrt(
        max(float(1e-4), float(1.0).sub(etaGlass.mul(etaGlass).mul(float(1.0).sub(cosI.mul(cosI)))))
      );
      const dGlass = normalize(inc.mul(etaGlass).add(nEntry.mul(etaGlass.mul(cosI).sub(cosGlass))));
      const glassShift = select(isYPane_in, vec3(0.0, 0.0, 0.0), dGlass.sub(nEntry.negate()).mul(0.012));
      const pRawStart = p0.add(glassShift);
      const startTopCeil = float(TANK_INNER_Y_TOP)
        .add(uSloshTilt.x.mul(0.45).mul(pRawStart.x))
        .add(uSloshTilt.y.mul(0.45).mul(pRawStart.z))
        .add(0.010);
      const pStart = vec3(
        clamp(pRawStart.x, float(-TANK_INNER_X + 0.0005), float(TANK_INNER_X - 0.0005)),
        clamp(pRawStart.y, float(TANK_INNER_Y_BOT + 0.0005), startTopCeil),
        clamp(pRawStart.z, float(-TANK_INNER_Z + 0.0005), float(TANK_INNER_Z - 0.0005))
      );

      // Distances to exit walls of the inner water chamber (matching deformed slosh plane `0.45 * uSloshTilt`!)
      const targetX = select(dWater.x.greaterThanEqual(0.0), float(TANK_INNER_X), float(-TANK_INNER_X));
      const sloshedTopAtStart = float(TANK_INNER_Y_TOP)
        .add(uSloshTilt.x.mul(0.45).mul(pStart.x))
        .add(uSloshTilt.y.mul(0.45).mul(pStart.z));
      const targetY = select(
        dWater.y.greaterThanEqual(0.0),
        max(pStart.y.add(0.001), sloshedTopAtStart),
        float(TANK_INNER_Y_BOT)
      );
      const targetZ = select(dWater.z.greaterThanEqual(0.0), float(TANK_INNER_Z), float(-TANK_INNER_Z));

      const tExitX = max(float(0.0), safeDiv(targetX.sub(pStart.x), dWater.x));
      const tExitY = max(float(0.0), safeDiv(targetY.sub(pStart.y), dWater.y));
      const tExitZ = max(float(0.0), safeDiv(targetZ.sub(pStart.z), dWater.z));
      const tBoxExit = min(tExitX, min(tExitY, tExitZ));

      // Intersect refracted ray with the virtual camera's exact focal plane through SPECIMEN_CENTER_LOCAL
      const specCenter = vec3(
        SPECIMEN_CENTER_LOCAL.x,
        SPECIMEN_CENTER_LOCAL.y,
        SPECIMEN_CENTER_LOCAL.z
      );
      const paneCamDir = normalize(vec3(paneCamDir_in));
      const denomFocus = max(float(0.25), dot(dWater, paneCamDir));
      const tFocus = clamp(dot(specCenter.sub(pStart), paneCamDir).div(denomFocus), float(0.04), tBoxExit);

      const pDirect = pStart.add(dWater.mul(tFocus));
      const sampDirect = samplePaneRT(pDirect, isXPane_in, isYPane_in);

      // Inner-Wall & Underside-Meniscus Total Internal Reflection (TIR)
      const hitXFirst = tExitX.lessThan(tExitY).and(tExitX.lessThan(tExitZ));
      const hitYFirst = tExitY.lessThanEqual(tExitX).and(tExitY.lessThan(tExitZ));
      const tSideHit = select(hitXFirst, tExitX, select(hitYFirst, tExitY, tExitZ));
      const pWallHit = pStart.add(dWater.mul(tSideHit));

      const menUv = clamp(
        vec2(pWallHit.x.div(TANK_INNER_X * 2.0).add(0.5), pWallHit.z.div(TANK_INNER_Z * 2.0).add(0.5)),
        0.02,
        0.98
      );
      const menWave = texture(hydroSim.dataTexture, menUv).level(float(0.0));
      const isTopHit = hitYFirst.and(dWater.y.greaterThan(0.0));

      const wallNorm = normalize(
        select(
          hitXFirst,
          vec3(select(dWater.x.greaterThan(0.0), float(-1.0), float(1.0)), float(0.0), float(0.0)),
          select(
            isTopHit,
            vec3(uSloshTilt.x.mul(0.45).add(menWave.g), float(-1.0), uSloshTilt.y.mul(0.45).add(menWave.b)),
            select(
              hitYFirst,
              vec3(float(0.0), float(1.0), float(0.0)),
              vec3(float(0.0), float(0.0), select(dWater.z.greaterThan(0.0), float(-1.0), float(1.0)))
            )
          )
        )
      );

      const cosWall = clamp(abs(dot(dWater, wallNorm)), 0.0, 1.0);
      const sin2Air = ior.mul(ior).mul(float(1.0).sub(cosWall.mul(cosWall)));
      const tirFresnel = smoothstep(float(0.66), float(0.98), sin2Air);
      const hitBeforeSpec = smoothstep(tFocus, tFocus.sub(0.08), tSideHit);

      const dRefl = dWater.sub(wallNorm.mul(dot(dWater, wallNorm).mul(2.0)));
      const reflAlign = smoothstep(float(0.20), float(0.60), dot(dRefl, paneCamDir));
      const tirMask = tirFresnel.mul(hitBeforeSpec).mul(reflAlign).mul(float(0.65));

      const remDist = clamp(dot(specCenter.sub(pWallHit), dRefl), float(0.06), float(1.30));
      const pRefl = pWallHit.add(dRefl.mul(remDist));
      const sampRefl = samplePaneRT(pRefl, isXPane_in, isYPane_in);

      // Refracted back-wall / bottom-plate inner glass miter seams (vertical corners & floor ONLY —
      // never draw a false chromatic seam at `TANK_INNER_Y_TOP`, and give underside TIR `isTopHit` a subtle continuous sheen!)
      const pExit = pStart.add(dWater.mul(tBoxExit));
      const dxIn = abs(float(TANK_INNER_X).sub(abs(pExit.x)));
      const dyBotIn = abs(pExit.y.sub(TANK_INNER_Y_BOT));
      const sloshedTopAtExit = float(TANK_INNER_Y_TOP)
        .add(uSloshTilt.x.mul(0.45).mul(pExit.x))
        .add(uSloshTilt.y.mul(0.45).mul(pExit.z));
      const dyTopIn = abs(sloshedTopAtExit.sub(pExit.y));
      const dzIn = abs(float(TANK_INNER_Z).sub(abs(pExit.z)));

      // Vertical back corner posts (`dxIn` & `dzIn`) + bottom floor edges (`dyBotIn` & (`dxIn` or `dzIn`))
      const vertCornerDist = max(dxIn, dzIn);
      const floorEdgeDist = max(dyBotIn, min(dxIn, dzIn));
      const backWaterlineDist = max(dyTopIn, min(dxIn, dzIn));
      const edgeDistIn = min(vertCornerDist, min(floorEdgeDist, backWaterlineDist));
      const innerSeamGlow = smoothstep(float(0.018), float(0.002), edgeDistIn).mul(0.068);

      // Subtle continuous underside-meniscus TIR ripple sheen when looking upward through front/side glass
      const undersideRipple = float(0.75).add(menWave.g.sub(menWave.b).mul(8.0));
      const undersideSheen = select(
        isTopHit,
        tirFresnel.mul(0.011).mul(clamp(undersideRipple, 0.45, 1.45)),
        float(0.0)
      );

      const onFloorMask = smoothstep(float(0.045), float(0.004), dyBotIn);
      const floorSheen = onFloorMask
        .mul(smoothstep(float(0.82), float(0.12), abs(pExit.x)))
        .mul(0.048);

      const slateBg = chalkySlateBackdropNode();
      const glassCueTint = mix(celadon.rInf, calcite.rInf, float(0.55));
      const backEnvironment = slateBg.add(
        glassCueTint.mul(innerSeamGlow.add(floorSheen).add(undersideSheen))
      );

      const combinedFgRgb = mix(sampDirect.rgb, sampRefl.rgb, tirMask.mul(sampRefl.a));
      const combinedFgAlpha = clamp(max(sampDirect.a, sampRefl.a.mul(tirMask)), 0.0, 1.0);

      const rawComposite = combinedFgRgb.add(backEnvironment.mul(float(1.0).sub(combinedFgAlpha)));
      const effChord = mix(tBoxExit, tFocus, combinedFgAlpha.mul(0.65));

      return vec4(rawComposite, effChord);
    }
  );

  const waterPrismMat = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
  });

  // aWaterMeta: .x = vertical fraction [0 floor .. 1 deformed top surface], .y = inner-wall proximity [0..1]
  const aWaterMeta = attribute('aWaterMeta', 'vec2');
  const camLocal4 = uTankWorldToLocal.mul(vec4(cameraPosition, 1.0));
  const pLocalEntry = positionLocal;
  const incLocal = normalize(pLocalEntry.sub(camLocal4.xyz));

  const isXPane = abs(normalLocal.x).greaterThan(0.5);
  const isYPane = abs(normalLocal.y).greaterThan(0.5);
  const isTopEntry = normalLocal.y.greaterThan(0.5);
  const activePaneCamDir = select(isXPane, uCamDirX, select(isYPane, uCamDirY, uCamDirZ));

  const entryMenUv = clamp(
    vec2(pLocalEntry.x.div(TANK_INNER_X * 2.0).add(0.5), pLocalEntry.z.div(TANK_INNER_Z * 2.0).add(0.5)),
    0.02,
    0.98
  );
  const entryWave = texture(hydroSim.dataTexture, entryMenUv).level(float(0.0));
  const brineShimmer = mineralToothFbm(
    pLocalEntry.mul(2.2).add(vec3(0.0, stage.uTime.mul(-0.07), 0.0)),
    float(1.8)
  )
    .sub(0.5)
    .mul(0.0040);

  const nEntryPerturbed = normalize(
    select(
      isTopEntry,
      vec3(
        uSloshTilt.x.mul(0.45).add(entryWave.g).negate(),
        float(1.0),
        uSloshTilt.y.mul(0.45).add(entryWave.b).negate()
      ),
      normalLocal.add(vec3(brineShimmer, brineShimmer.mul(0.4), brineShimmer.negate()))
    )
  );

  const bandR = sampleRefractedBand(
    pLocalEntry,
    nEntryPerturbed,
    incLocal,
    float(1.331),
    isXPane,
    isYPane,
    activePaneCamDir
  );
  const bandG = sampleRefractedBand(
    pLocalEntry,
    nEntryPerturbed,
    incLocal,
    float(1.336),
    isXPane,
    isYPane,
    activePaneCamDir
  );
  const bandB = sampleRefractedBand(
    pLocalEntry,
    nEntryPerturbed,
    incLocal,
    float(1.342),
    isXPane,
    isYPane,
    activePaneCamDir
  );

  const refractedRgb = vec3(bandR.r, bandG.g, bandB.b);

  const marineExtinction = mix(celadon.extinction, ochre.extinction, float(0.12)).mul(0.058);
  const waterTrans = beerLambertTransmittance(marineExtinction, bandG.w);
  const baseWaterRgb = refractedRgb.mul(waterTrans);

  // Unified water-surface Fresnel & capillary wetting meniscus directly on `waterPrismMat`
  // (zero detached floating meniscus plane; 100% shared vertices across top & side water faces!)
  const topCosI = clamp(incLocal.negate().dot(nEntryPerturbed), 0.0, 1.0);
  const topFresnel = float(0.006).add(float(0.042).mul(float(1.0).sub(topCosI).pow(4.5)));
  const topReflLocal = incLocal.sub(nEntryPerturbed.mul(dot(incLocal, nEntryPerturbed).mul(2.0)));
  const topReflWorld = normalize(uTankLocalToWorld.mul(vec4(topReflLocal, 0.0)).xyz);
  const topSoftbox = smoothstep(float(0.72), float(0.97), dot(topReflWorld, uKeyLightDir));

  // Crisp ~2mm capillary wetting rim right where the water meets the 4 inner glass walls (both from above & side!)
  const topWallRim = aWaterMeta.y.pow(13.0).mul(0.095);
  const sideMeniscusLine = smoothstep(float(0.990), float(0.9995), aWaterMeta.x).mul(0.085);
  const meniscusTint = mix(celadon.rInf, calcite.rInf, float(0.68));

  const topSurfaceSheen = select(
    isTopEntry,
    meniscusTint.mul(topFresnel.add(topSoftbox.mul(0.075)).add(topWallRim)),
    meniscusTint.mul(sideMeniscusLine)
  );

  waterPrismMat.colorNode = baseWaterRgb.add(topSurfaceSheen);

  const deformableWaterPrism = new DeformableWaterPrism(waterPrismMat);
  deformableWaterPrism.updateFromSim(hydroSim);
  deformableWaterPrism.mesh.renderOrder = 6;
  exteriorTankGroup.add(deformableWaterPrism.mesh);

  // -------------------------------------------------------------------------
  // 6. Double-Lined Beveled Soda-Lime Float-Glass Vitrine Frame
  // -------------------------------------------------------------------------
  const { edgeSlabGeom, outerWindowGeom } = createBeveledOpticalTankGlassGeometry();

  const glassEdgeMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });

  const aGlass = attribute('aGlassMeta', 'vec4');
  const gView = normalize(cameraPosition.sub(positionWorld));
  const gNorm = normalize(normalWorld);
  const gNdotV = clamp(dot(gNorm, gView), 0.0, 1.0);
  const gFresnel = float(0.06).add(float(0.94).mul(float(1.0).sub(gNdotV).pow(3.6)));

  const slabChord = aGlass.x.mul(mix(float(0.85), float(1.55), float(1.0).sub(gNdotV)));
  const sodaLimeTint = beerLambertTransmittance(celadon.extinction.mul(1.42), slabChord);
  const deepCeladonSlab = celadon.rInf.mul(sodaLimeTint).mul(0.72);

  const gRefl = gView.negate().sub(gNorm.mul(dot(gView.negate(), gNorm).mul(2.0)));
  const keyStrip = smoothstep(float(0.72), float(0.97), dot(normalize(gRefl), uKeyLightDir));
  const rimStrip = smoothstep(float(0.68), float(0.95), dot(normalize(gRefl), uRimLightDir)).mul(0.62);
  const arrisHighlight = aGlass.y.pow(2.2).mul(0.62).add(keyStrip.mul(0.95)).add(rimStrip);

  const vGlassCol = varying(
    mix(deepCeladonSlab, calcite.rInf, clamp(arrisHighlight.mul(0.68).add(gFresnel.mul(0.32)), 0.0, 1.0)).add(
      calcite.rInf.mul(arrisHighlight.mul(0.34))
    )
  );
  const vGlassAlpha = varying(
    clamp(float(0.14).add(aGlass.y.pow(2.0).mul(0.46)).add(gFresnel.mul(0.28)).add(keyStrip.mul(0.28)), 0.14, 0.86)
  );

  glassEdgeMat.colorNode = vGlassCol;
  glassEdgeMat.opacityNode = vGlassAlpha;

  const glassEdgeMesh = new THREE.Mesh(edgeSlabGeom, glassEdgeMat);
  glassEdgeMesh.renderOrder = 10;
  exteriorTankGroup.add(glassEdgeMesh);

  const outerWindowMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });

  const winView = normalize(cameraPosition.sub(positionWorld));
  const winNorm = normalize(normalWorld);
  const winNdotV = clamp(dot(winNorm, winView), 0.0, 1.0);
  const winFresnel = float(1.0).sub(winNdotV).pow(5.2);
  const winRefl = winView.negate().sub(winNorm.mul(dot(winView.negate(), winNorm).mul(2.0)));

  const studioWindowBand = smoothstep(float(0.80), float(0.97), dot(normalize(winRefl), uKeyLightDir)).mul(
    smoothstep(float(-0.45), float(0.65), positionLocal.y)
  );
  const rimWindowBand = smoothstep(float(0.82), float(0.98), dot(normalize(winRefl), uRimLightDir)).mul(0.42);

  outerWindowMat.colorNode = mix(celadon.rInf, calcite.rInf, float(0.82));
  outerWindowMat.opacityNode = clamp(
    winFresnel.mul(0.22).add(studioWindowBand.mul(0.065)).add(rimWindowBand.mul(0.045)),
    0.002,
    0.34
  );

  const outerWindowMesh = new THREE.Mesh(outerWindowGeom, outerWindowMat);
  outerWindowMesh.renderOrder = 11;
  exteriorTankGroup.add(outerWindowMesh);

  // -------------------------------------------------------------------------
  // Tactile Pointer Interaction: Snell-Refracted Specimen Raycast & Rust Flaking
  // -------------------------------------------------------------------------
  let hasPrevHover = false;
  let hoverNailTarget = 0.0;
  let downX = 0;
  let downY = 0;
  let movedDistPx = 0;
  let isPointerDown = false;
  let allPanesPrimed = false;

  /**
   * Traces the pointer ray into the tank, refracts it across the hit water-cube pane
   * via Snell's law (`eta = 1 / 1.336`), and resolves both pinpoint forged-nail surface hits
   * (`hitNail` -> `_nailHitPt`, `_nailHitNorm`) and hanging rusticle curtain proximity.
   */
  const raycastTankAndSpecimen = (
    clientX: number,
    clientY: number,
    outWallHit: THREE.Vector3,
    outSpecimenPt: THREE.Vector3
  ): { hitTank: boolean; nearSpecimen: boolean; hitNail: boolean } => {
    const rect = stage.shell.canvas.getBoundingClientRect();
    _ndcVec2.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2.0 - 1.0;
    _ndcVec2.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2.0 + 1.0;
    _raycaster.setFromCamera(_ndcVec2, stage.camera);

    _invWorldMat.copy(exteriorTankGroup.matrixWorld).invert();
    _localRay.copy(_raycaster.ray).applyMatrix4(_invWorldMat);
    if (_localRay.intersectBox(_tankBoxLocal, outWallHit) === null) {
      return { hitTank: false, nearSpecimen: false, hitNail: false };
    }

    // Determine outward normal of hit tank face
    const rx = Math.abs(outWallHit.x) / TANK_OUTER_X;
    const ry = Math.abs(outWallHit.y) / TANK_OUTER_Y;
    const rz = Math.abs(outWallHit.z) / TANK_OUTER_Z;
    if (rx >= ry && rx >= rz) {
      _hitNormalLocal.set(Math.sign(outWallHit.x) || 1, 0, 0);
    } else if (ry >= rx && ry >= rz) {
      _hitNormalLocal.set(0, Math.sign(outWallHit.y) || 1, 0);
    } else {
      _hitNormalLocal.set(0, 0, Math.sign(outWallHit.z) || 1);
    }

    refractDirCpu(_localRay.direction, _hitNormalLocal, 1.0 / WATER_IOR_REF, _waterRayDir);
    _waterRayOrigin.copy(outWallHit);

    // Sample 28 fine steps along the underwater Snell ray to detect exact forged-nail hits + rusticle curtain proximity
    let bestDistSq = 1e9;
    let bestT = 0.85;
    let hitNail = false;

    for (let s = 1; s <= 36; s++) {
      const t = 0.18 + s * 0.048;
      const px = _waterRayOrigin.x + _waterRayDir.x * t;
      const py = _waterRayOrigin.y + _waterRayDir.y * t;
      const pz = _waterRayOrigin.z + _waterRayDir.z * t;

      // Pinpoint first entry into the submerged wrought-iron nail (`halfY + 0.036`, `halfZ + 0.036`)
      if (!hitNail && px >= SPIKE_MIN_X - 0.018 && px <= SPIKE_MAX_X + 0.020) {
        const spNail = evalSpikeSpine(Math.max(SPIKE_MIN_X, Math.min(SPIKE_MAX_X, px)));
        const dyN = py - spNail.y;
        const dzN = pz - spNail.z;
        const tolY = spNail.halfY + 0.036;
        const tolZ = spNail.halfZ + 0.036;
        if (Math.abs(dyN) <= tolY && Math.abs(dzN) <= tolZ) {
          hitNail = true;
          const surfScaleY = Math.abs(dyN) > 1e-4 ? dyN / Math.max(0.02, spNail.halfY) : 0.0;
          const surfScaleZ = Math.abs(dzN) > 1e-4 ? dzN / Math.max(0.02, spNail.halfZ) : 1.0;
          _nailHitNorm.set(px > 0.63 ? 0.45 : 0.0, surfScaleY, surfScaleZ).normalize();
          _nailHitPt.set(
            Math.max(SPIKE_MIN_X, Math.min(SPIKE_MAX_X, px)),
            spNail.y + Math.max(-spNail.halfY, Math.min(spNail.halfY, dyN)),
            spNail.z + Math.max(-spNail.halfZ, Math.min(spNail.halfZ, dzN))
          );
        }
      }

      const clampedX = Math.max(SPIKE_MIN_X, Math.min(SPIKE_MAX_X, px));
      const sp = evalSpikeSpine(clampedX);
      const curtainDepth = clampedX > -0.52 && clampedX < 0.46 ? 0.68 : 0.08;
      const minY = sp.y - sp.halfY - curtainDepth;
      const maxY = sp.y + sp.halfY;
      const clampedY = Math.max(minY, Math.min(maxY, py));

      const dx = px - clampedX;
      const dy = py - clampedY;
      const dz = pz - sp.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestDistSq) {
        bestDistSq = d2;
        bestT = t;
      }
    }

    outSpecimenPt.copy(_waterRayOrigin).addScaledVector(_waterRayDir, bestT);
    if (hitNail) {
      outSpecimenPt.copy(_nailHitPt);
    }
    return { hitTank: true, nearSpecimen: hitNail || bestDistSq < 0.14, hitNail };
  };

  const onPointerDown = (ev: PointerEvent) => {
    isPointerDown = true;
    downX = ev.clientX;
    downY = ev.clientY;
    movedDistPx = 0;
    const res = raycastTankAndSpecimen(ev.clientX, ev.clientY, _curPointerLocal, _specimenHit);
    if (res.hitTank) {
      _prevPointerLocal.copy(_curPointerLocal);
      hasPrevHover = true;
      if (res.hitNail) {
        scrapeField.scrapeAtLocalPoint(_nailHitPt, 0.068, 0.48);
        _flakeImpulse.set(0.0, -0.05, 0.03);
        flakeSystem.spawnScrapeCrumble(_nailHitPt, _nailHitNorm, 0.95, _flakeImpulse);
        moteSystem.emitSiltPuff(_nailHitPt, 6, _flakeImpulse);
      }
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (isPointerDown) {
      movedDistPx += Math.hypot(ev.clientX - downX, ev.clientY - downY);
      downX = ev.clientX;
      downY = ev.clientY;
    }

    const res = raycastTankAndSpecimen(ev.clientX, ev.clientY, _curPointerLocal, _specimenHit);
    stage.shell.canvas.style.cursor = res.hitNail ? 'crosshair' : '';
    hoverNailTarget = res.hitNail ? 1.0 : 0.0;
    if (res.hitNail) {
      uHoverNailLocal.value.copy(_nailHitPt);
      flakeSystem.setHoverNailPoint(_nailHitPt);
    } else {
      flakeSystem.setHoverNailPoint(null);
    }

    if (res.hitTank) {
      if (hasPrevHover) {
        const dx = _curPointerLocal.x - _prevPointerLocal.x;
        const dy = _curPointerLocal.y - _prevPointerLocal.y;
        const dz = _curPointerLocal.z - _prevPointerLocal.z;
        const moveMag = Math.hypot(dx, dy, dz);

        if (moveMag > 3.5e-4) {
          const gain = isPointerDown ? 2.2 : 1.25;
          hydroSim.injectFluidImpulse(dx * gain, dz * gain, _specimenHit);
          if (moveMag > 1.5e-3) {
            hydroSim.injectWaveImpulse(
              _curPointerLocal.x / TANK_INNER_X,
              _curPointerLocal.z / TANK_INNER_Z,
              (isPointerDown ? 0.008 : 0.004) * Math.min(1.8, Math.hypot(dx, dz) * 18.0),
              0.24
            );
          }

          // Direct mouseover rust scraping right on the nail: erodes crust to dark iron & peels flaky bark scabs!
          if (res.hitNail) {
            const removed = scrapeField.scrapeAtLocalPoint(
              _nailHitPt,
              isPointerDown ? 0.090 : 0.062,
              isPointerDown ? 0.64 : 0.42
            );
            _flakeImpulse.set(dx * 2.2, Math.min(-0.02, dy * 1.8 - 0.03), dz * 2.2);
            const crumbleIntensity =
              (isPointerDown ? 1.15 : 0.72) + Math.min(0.65, removed * 0.40);
            flakeSystem.spawnScrapeCrumble(_nailHitPt, _nailHitNorm, crumbleIntensity, _flakeImpulse);
            moteSystem.emitSiltPuff(_nailHitPt, isPointerDown ? 8 : 4, _flakeImpulse);
          } else if ((res.nearSpecimen || isPointerDown) && moveMag > 2.2e-3) {
            _flakeImpulse.set(dx * 2.4, Math.min(-0.02, dy * 1.8 - 0.03), dz * 2.4);
            flakeSystem.dislodgeNear(_specimenHit, isPointerDown ? 0.28 : 0.18, isPointerDown ? 8 : 4, _flakeImpulse);
            moteSystem.emitSiltPuff(_specimenHit, isPointerDown ? 6 : 3, _flakeImpulse);
          }
        }
      }
      _prevPointerLocal.copy(_curPointerLocal);
      hasPrevHover = true;
    } else {
      hasPrevHover = false;
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (isPointerDown && movedDistPx < 8.0) {
      const res = raycastTankAndSpecimen(ev.clientX, ev.clientY, _localHit, _specimenHit);
      if (res.hitTank) {
        const tapSignX = _localHit.x >= 0 ? -0.32 : 0.32;
        const tapSignZ = _localHit.z >= 0 ? -0.24 : 0.24;
        hydroSim.injectFluidImpulse(tapSignX, tapSignZ, _specimenHit);
        hydroSim.injectWaveImpulse(
          Math.max(-0.85, Math.min(0.85, _localHit.x / TANK_INNER_X)),
          Math.max(-0.85, Math.min(0.85, _localHit.z / TANK_INNER_Z)),
          -0.016,
          0.22
        );
        if (res.hitNail) {
          scrapeField.scrapeAtLocalPoint(_nailHitPt, 0.088, 0.65);
          _flakeImpulse.set(tapSignX * 0.25, -0.05, tapSignZ * 0.25);
          flakeSystem.spawnScrapeCrumble(_nailHitPt, _nailHitNorm, 1.35, _flakeImpulse);
          moteSystem.emitSiltPuff(_nailHitPt, 16, _flakeImpulse);
        } else {
          _flakeImpulse.set(tapSignX * 0.28, -0.04, tapSignZ * 0.28);
          flakeSystem.dislodgeNear(_specimenHit, 0.55, 20, _flakeImpulse);
          moteSystem.emitSiltPuff(_specimenHit, 18, _flakeImpulse);
        }
      } else {
        hydroSim.injectFluidImpulse(0.24, -0.18);
        hydroSim.injectWaveImpulse(0.15, -0.12, -0.014, 0.24);
        _flakeImpulse.set(0.08, -0.03, -0.06);
        flakeSystem.dislodgeNear(SPECIMEN_CENTER_LOCAL, 0.60, 14, _flakeImpulse);
      }
    }
    isPointerDown = false;
  };

  const onResize = () => {
    const { w, h } = getTargetSize();
    rtPaneZ.setSize(w, h);
    rtPaneX.setSize(w, h);
    rtPaneY.setSize(w, h);
    allPanesPrimed = false;
  };

  stage.shell.canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });
  window.addEventListener('resize', onResize);

  stage.shell.registerDispose(() => {
    stage.shell.canvas.style.cursor = '';
    stage.shell.canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('resize', onResize);
    rtPaneZ.dispose();
    rtPaneX.dispose();
    rtPaneY.dispose();
    hydroSim.dataTexture.dispose();
    scrapeField.dataTexture.dispose();
    spikeGeom.dispose();
    rusticlesGeom.dispose();
    flakeSystem.geometry.dispose();
    siltVeilsGeom.dispose();
    moteSystem.geometry.dispose();
    deformableWaterPrism.geometry.dispose();
    edgeSlabGeom.dispose();
    outerWindowGeom.dispose();
  });

  /**
   * Configures and renders a single virtual Snell camera aligned with one glass pane's
   * refracted underwater ray bundle.
   */
  const renderSnellPanePass = (
    renderer: THREE.WebGPURenderer,
    mainCam: THREE.PerspectiveCamera,
    virtCam: THREE.PerspectiveCamera,
    paneRT: THREE.RenderTarget,
    paneAnchor: THREE.Vector3,
    paneNormal: THREE.Vector3,
    camLocalDist: number,
    outViewProj: THREE.Matrix4,
    outCamDirLocal: THREE.Vector3
  ) => {
    _incDirLocal.copy(paneAnchor).sub(_camLocal).normalize();
    refractDirCpu(_incDirLocal, paneNormal, 1.0 / WATER_IOR_REF, _refrDirLocal);
    outCamDirLocal.copy(_refrDirLocal);

    const virtDist = camLocalDist * 1.04;
    _virtPosLocal.copy(SPECIMEN_CENTER_LOCAL).addScaledVector(_refrDirLocal, -virtDist);

    _virtPosWorld.copy(_virtPosLocal).applyMatrix4(interiorTankGroup.matrixWorld);
    _targetWorld.copy(SPECIMEN_CENTER_LOCAL).applyMatrix4(interiorTankGroup.matrixWorld);

    virtCam.fov = mainCam.fov;
    virtCam.aspect = mainCam.aspect;
    virtCam.near = mainCam.near;
    virtCam.far = mainCam.far;
    virtCam.up.copy(Math.abs(_refrDirLocal.y) > 0.94 ? _altUp : _worldUp);
    virtCam.position.copy(_virtPosWorld);
    virtCam.lookAt(_targetWorld);
    virtCam.updateProjectionMatrix();
    virtCam.updateMatrixWorld(true);

    outViewProj.multiplyMatrices(virtCam.projectionMatrix, virtCam.matrixWorldInverse);

    renderer.setRenderTarget(paneRT);
    renderer.setClearColor(0x000000, 0);
    renderer.render(interiorScene, virtCam);
  };

  // -------------------------------------------------------------------------
  // Main Animation Loop: Couple Orbit Slosh -> Flake Physics -> Snell Passes
  // -------------------------------------------------------------------------
  stage.onTick(({ dtSec, elapsedSec, camera, renderer }) => {
    _curCamDir.copy(camera.position).normalize();
    const camDeltaX = _curCamDir.x - _prevCamDir.x;
    const camDeltaZ = _curCamDir.z - _prevCamDir.z;
    if (Math.abs(camDeltaX) + Math.abs(camDeltaZ) > 1e-5) {
      hydroSim.injectFluidImpulse(-camDeltaX * 1.55, -camDeltaZ * 1.55);
    }
    _prevCamDir.copy(_curCamDir);

    uHoverNailActive.value +=
      (hoverNailTarget - uHoverNailActive.value) * Math.min(1.0, dtSec * 12.0);

    hydroSim.step(dtSec, elapsedSec);
    scrapeField.step(dtSec);
    flakeSystem.step(dtSec, elapsedSec, hydroSim);
    moteSystem.step(dtSec, elapsedSec, hydroSim);
    deformableWaterPrism.updateFromSim(hydroSim);

    const tankTiltZ = -hydroSim.sloshX * 0.20;
    const tankTiltX = hydroSim.sloshZ * 0.20;
    exteriorTankGroup.rotation.set(tankTiltX, TANK_BASE_YAW, tankTiltZ);
    exteriorTankGroup.updateMatrixWorld(true);

    interiorTankGroup.rotation.copy(exteriorTankGroup.rotation);
    interiorTankGroup.updateMatrixWorld(true);

    uTankLocalToWorld.value.copy(exteriorTankGroup.matrixWorld);
    uTankWorldToLocal.value.copy(exteriorTankGroup.matrixWorld).invert();
    uSloshTilt.value.set(hydroSim.sloshX, hydroSim.sloshZ);
    uSwayMode1.value.set(hydroSim.sway1X, hydroSim.sway1Z);
    uSwayMode2.value.set(hydroSim.sway2X, hydroSim.sway2Z);
    uStirLocalPos.value.copy(hydroSim.stirCenter);
    uStirMeta.value.set(hydroSim.stirEnergy, hydroSim.stirVelocity.x, hydroSim.stirVelocity.z);

    // Compute camera position in tank-local space & render active Snell pane RTs
    _camLocal.copy(camera.position).applyMatrix4(uTankWorldToLocal.value);
    const camLocalDist = Math.max(1.8, _camLocal.distanceTo(SPECIMEN_CENTER_LOCAL));

    // Update shared interior shadow map once per frame across all virtual pane passes
    intKey.shadow.needsUpdate = true;

    const signZ = _camLocal.z >= 0 ? 1.0 : -1.0;
    if (!allPanesPrimed || Math.abs(_camLocal.z) > TANK_INNER_Z * 0.98) {
      _paneAnchorLocal.set(
        Math.max(-0.45, Math.min(0.45, _camLocal.x * 0.18)),
        0.0,
        signZ * TANK_INNER_Z
      );
      _paneNormalLocal.set(0, 0, signZ);
      renderSnellPanePass(
        renderer,
        camera,
        camPaneZ,
        rtPaneZ,
        _paneAnchorLocal,
        _paneNormalLocal,
        camLocalDist,
        uViewProjZ.value,
        uCamDirZ.value
      );
    }

    const signX = _camLocal.x >= 0 ? 1.0 : -1.0;
    if (!allPanesPrimed || Math.abs(_camLocal.x) > TANK_INNER_X * 0.98) {
      _paneAnchorLocal.set(
        signX * TANK_INNER_X,
        0.0,
        Math.max(-0.45, Math.min(0.45, _camLocal.z * 0.18))
      );
      _paneNormalLocal.set(signX, 0, 0);
      renderSnellPanePass(
        renderer,
        camera,
        camPaneX,
        rtPaneX,
        _paneAnchorLocal,
        _paneNormalLocal,
        camLocalDist,
        uViewProjX.value,
        uCamDirX.value
      );
    }

    const signY = _camLocal.y >= 0 ? 1.0 : -1.0;
    const yBound = signY > 0 ? TANK_INNER_Y_TOP : Math.abs(TANK_INNER_Y_BOT);
    if (!allPanesPrimed || Math.abs(_camLocal.y) > yBound * 0.96) {
      _paneAnchorLocal.set(0.0, signY > 0 ? TANK_INNER_Y_TOP : TANK_INNER_Y_BOT, 0.0);
      _paneNormalLocal.set(0, signY, 0);
      renderSnellPanePass(
        renderer,
        camera,
        camPaneY,
        rtPaneY,
        _paneAnchorLocal,
        _paneNormalLocal,
        camLocalDist,
        uViewProjY.value,
        uCamDirY.value
      );
    }

    allPanesPrimed = true;

    renderer.setRenderTarget(null);
    renderer.setClearColor(_slateClearColor, 1);
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay11Rust();
}
