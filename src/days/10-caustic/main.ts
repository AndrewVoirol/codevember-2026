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
  instanceIndex,
  length,
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
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  BOWL_CENTER_Y,
  BOWL_LIP_Y,
  BOWL_RADIUS_XZ,
  BOWL_RADIUS_Y,
  CAUSTIC_MAP_EXTENT,
  DynamicLiquidMesh,
  GOBLET_POS_X,
  GOBLET_POS_Z,
  LIQUID_FILL_Y,
  LiquidMeniscusSim,
  SLAB_THICKNESS,
  SLAB_TOP_Y,
  createBeveledCalciteSlabGeometry,
  createVenetianCrystalBowlAndFoot,
} from './gobletApparatus';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Day 10 — Caustic:
 * "Blown Venetian Crystal Goblet & Catastrophe Nephroid / Meniscus Caustics"
 *
 * Physical & Optical Architecture:
 * 1. Forward 3-Manifold Wavefront Ray-Tracing (`1024x1024` HalfFloat HDR `causticRT`):
 *    - Traces 9 continuous optical wavefront sheets (`3` smooth manifolds × `3` mineral
 *      Cauchy dispersion bands) from the steerable brass condenser lamp through the
 *      Venetian goblet onto the Bone Calcite stone slab:
 *      * Manifold 0: Free-surface liquid meniscus refraction (`meniscusTex` live 2D
 *        gravity-capillary wave PDE + coupled pendulum slosh) -> curved lower bowl exit.
 *      * Manifold 1: Curved liquid-filled bowl diacaustic condenser -> sharp A3 Cardioid
 *        focal spearhead tinted by Beer-Lambert chord transmittance.
 *      * Manifold 2: Concave back-wall crystal reflection catacaustic -> Cayley's
 *        2-cusped Venetian Nephroid arch framing the base inside the refractive shadow.
 *    - Evaluates exact slab-plane Jacobian area compression `A_0 / sqrt(A_slab^2 + eps^2)`
 *      per vertex with finite-source Airy regularization (100% camera-angle invariant).
 * 2. Energy-Conserving Dielectric Fresnel Shadow + Subsurface Calcite Scattering:
 *    - The goblet deflects direct condenser light out of its geometric shadow footprint
 *      on the Bone Calcite slab while concentrating that redirected flux into incandescent
 *      Realgar-Amber, Spore-Sulfur, and Celadon-Glauconite caustic cusps.
 * 3. Tactile Pointer Coupling:
 *    - Dragging the brass condenser gnomon or calcite slab smoothly steers incident beam
 *      azimuth & elevation (unfolding/folding the nephroid & cardioid cusps across the stone).
 *    - Sweeping across or clicking the goblet sloshes the botanical-mineral elixir and
 *      excites live capillary wave ripples that dance across the projected caustic web.
 */

// Pre-allocated module-scope scratch objects for zero-GC interaction & lamp updates
const _raycaster = new THREE.Raycaster();
const _slabPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -SLAB_TOP_Y);
const _meniscusPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -LIQUID_FILL_Y);
const _hitVec = new THREE.Vector3();
const _gobletBowlCenter = new THREE.Vector3(GOBLET_POS_X, BOWL_CENTER_Y, GOBLET_POS_Z);
const _lampWorldPos = new THREE.Vector3();
const _beamForward = new THREE.Vector3();
const _beamRight = new THREE.Vector3();
const _beamUp = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);

export async function bootDay10Caustic(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(10);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.22, 2.38, 3.82],
    fov: 34,
    enableShadows: true,
  });

  stage.controls.target.set(0.08, -0.18, 0.04);
  stage.controls.minPolarAngle = 0.22;
  stage.controls.maxPolarAngle = Math.PI * 0.48;
  stage.controls.minDistance = 2.1;
  stage.controls.maxDistance = 7.2;
  stage.controls.update();

  // -------------------------------------------------------------------------
  // Mineral Pigment Uniform Packs (Strictly Zero Digital RGB Neons)
  // -------------------------------------------------------------------------
  const celadon = createPigmentUniforms('celadonGlauconite');
  const realgar = createPigmentUniforms('realgarAmber');
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const ochre = createPigmentUniforms('yellowOchre');
  const umber = createPigmentUniforms('rawUmber');
  const spore = createPigmentUniforms('sporeSulfur');
  const azurite = createPigmentUniforms('azuritePrussian');

  // -------------------------------------------------------------------------
  // Live Meniscus Wave PDE & Shared Optical Uniforms
  // -------------------------------------------------------------------------
  const meniscusSim = new LiquidMeniscusSim();

  const uLampPos = uniform(new THREE.Vector3(-2.65, 2.35, -1.55));
  const uBeamForward = uniform(new THREE.Vector3(0.7, -0.62, 0.35));
  const uBeamRight = uniform(new THREE.Vector3(0.45, 0.0, -0.89));
  const uBeamUp = uniform(new THREE.Vector3(0.55, 0.78, 0.27));
  const uSloshTilt = uniform(new THREE.Vector2(meniscusSim.sloshX, meniscusSim.sloshZ));
  const uBowlCenter = uniform(_gobletBowlCenter.clone());

  // Key directional light synced gently to condenser angle without overexposing the calcite slab
  const condenserSpot = new THREE.DirectionalLight(0xfff4e2, 0.28);
  stage.scene.add(condenserSpot);
  stage.scene.add(condenserSpot.target);
  condenserSpot.target.position.set(GOBLET_POS_X, SLAB_TOP_Y + 0.35, GOBLET_POS_Z);

  // -------------------------------------------------------------------------
  // Offscreen 1024x1024 HalfFloat Forward Caustic Wavefront Accumulator
  // -------------------------------------------------------------------------
  const causticRT = new THREE.RenderTarget(1024, 1024, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });

  const causticScene = new THREE.Scene();
  causticScene.background = new THREE.Color(0x000000);
  const causticCam = new THREE.OrthographicCamera(
    -CAUSTIC_MAP_EXTENT,
    CAUSTIC_MAP_EXTENT,
    CAUSTIC_MAP_EXTENT,
    -CAUSTIC_MAP_EXTENT,
    -5.0,
    5.0
  );
  causticCam.position.set(0, 0, 2.0);
  causticCam.lookAt(0, 0, 0);

  // Guarded 3D Snell refraction helper: returns vec4(refractedDir.xyz, nonTirWeight)
  const safeRefract = Fn(([incDir, surfNorm, etaVal]: [TslNode, TslNode, TslNode]) => {
    const I = normalize(vec3(incDir));
    const N = normalize(vec3(surfNorm));
    const eta = float(etaVal);
    const cosI = clamp(dot(N, I).negate(), 0.0, 1.0);
    const sin2T = eta.mul(eta).mul(float(1.0).sub(cosI.mul(cosI)));
    const k = max(float(0.0004), float(1.0).sub(sin2T));
    const valid = smoothstep(float(0.992), float(0.84), sin2T);
    const refrDir = normalize(I.mul(eta).add(N.mul(eta.mul(cosI).sub(sqrt(k)))));
    return vec4(refrDir, valid);
  });

  /**
   * Traces a single ray on optical manifold `sheetId` (0, 1, or 2) at normalized
   * parameter `(uNorm, vNorm)` in [-1..1]^2 with liquid/glass refractive indices
   * `(nLiq, nGlass)`. Returns `vec4(slabHitX, slabHitZ, chordLen, validWeight)`.
   */
  const traceManifoldRay = Fn(
    ([uNorm_in, vNorm_in, sheetId_in, nLiq_in, nGlass_in]: [
      TslNode,
      TslNode,
      TslNode,
      TslNode,
      TslNode,
    ]) => {
      const uN = float(uNorm_in);
      const vN = float(vNorm_in);
      const sheetId = float(sheetId_in);
      const nLiq = float(nLiq_in);
      const nGlass = float(nGlass_in);

      const rxz = float(BOWL_RADIUS_XZ - 0.014);
      const ry = float(BOWL_RADIUS_Y - 0.014);
      const slabY = float(SLAB_TOP_Y);

      // Horizontal forward unit direction from lamp toward goblet center in XZ
      const toGobletXZ = normalize(
        vec2(uBowlCenter.x.sub(uLampPos.x), uBowlCenter.z.sub(uLampPos.z))
      );
      const rightXZ = vec2(toGobletXZ.y.negate(), toGobletXZ.x);

      // ---------------------------------------------------------------------
      // Manifold 0: Free-Surface Rippling Meniscus -> Oblique Goblet Projection
      // ---------------------------------------------------------------------
      const discX = uN.mul(sqrt(max(float(0.0), float(1.0).sub(vN.mul(vN).mul(0.5)))));
      const discZ = vN.mul(sqrt(max(float(0.0), float(1.0).sub(uN.mul(uN).mul(0.5)))));
      const rMenFrac = sqrt(min(float(1.0), discX.mul(discX).add(discZ.mul(discZ))));

      const rFill = float(0.402);
      const localMenX = discX.mul(rFill);
      const localMenZ = discZ.mul(rFill);

      const menUv = clamp(vec2(discX, discZ).mul(0.47).add(0.5), 0.02, 0.98);
      const waveSample = texture(meniscusSim.dataTexture, menUv).level(float(0.0));
      const waveH = waveSample.r;
      const dWaveDx = waveSample.g.mul(2.35);
      const dWaveDz = waveSample.b.mul(2.35);

      const rimClimb = rMenFrac.mul(rMenFrac).mul(rMenFrac).mul(rMenFrac).mul(0.018);
      const rimSlope = rMenFrac.mul(rMenFrac).mul(rMenFrac).mul(0.085);
      const menY = float(LIQUID_FILL_Y)
        .add(uSloshTilt.x.mul(localMenX))
        .add(uSloshTilt.y.mul(localMenZ))
        .add(rimClimb)
        .add(waveH);

      const pMen = vec3(
        uBowlCenter.x.add(localMenX),
        menY,
        uBowlCenter.z.add(localMenZ)
      );
      const incMen = normalize(pMen.sub(uLampPos));
      const nMen = normalize(
        vec3(
          uSloshTilt.x.add(dWaveDx).add(discX.mul(rimSlope)).negate(),
          float(1.0),
          uSloshTilt.y.add(dWaveDz).add(discZ.mul(rimSlope)).negate()
        )
      );

      // Meniscus Snell refraction + thin-shell tulip side-wall exit (horizontal focusing + oblique reach)
      const refr0In = safeRefract(incMen, nMen, float(1.0).div(nLiq));
      const dLiq0 = refr0In.xyz;

      const o0 = pMen.sub(uBowlCenter).div(vec3(rxz, ry, rxz));
      const d0 = dLiq0.div(vec3(rxz, ry, rxz));
      const a0 = max(float(1e-5), dot(d0, d0));
      const b0 = dot(o0, d0);
      const c0 = dot(o0, o0).sub(1.0);
      const disc0 = max(float(0.0), b0.mul(b0).sub(a0.mul(c0)));
      const tExit0 = max(float(0.02), b0.negate().add(sqrt(disc0)).div(a0));
      const pExit0 = pMen.add(dLiq0.mul(tExit0));

      // Horizontal wall curvature focuses laterally while preserving oblique forward reach into the shadow
      const exitRelXZ0 = vec2(pExit0.x.sub(uBowlCenter.x), pExit0.z.sub(uBowlCenter.z));
      const nExitHoriz0 = normalize(vec3(exitRelXZ0.x, float(-0.22), exitRelXZ0.y));
      const refr0Out = safeRefract(dLiq0, nExitHoriz0.negate(), float(1.0).add(nLiq.sub(1.0).mul(0.58)));
      const dAir0 = normalize(mix(incMen, refr0Out.xyz, float(0.68)));
      const tSlab0 = slabY.sub(pExit0.y).div(min(float(-0.08), dAir0.y));
      const hit0 = pExit0.add(dAir0.mul(max(float(0.0), tSlab0)));
      const w0 = refr0In.w
        .mul(refr0Out.w)
        .mul(smoothstep(float(0.99), float(0.86), rMenFrac))
        .mul(smoothstep(float(-0.08), float(-0.20), dAir0.y));

      // ---------------------------------------------------------------------
      // Manifold 1: Astigmatic Tulip-Bowl Diacaustic ("Sweeping Cardioid Spear")
      // ---------------------------------------------------------------------
      const az1 = uN.mul(1.26);
      const cosA1 = cos(az1);
      const sinA1 = sin(az1);
      const frontDirXZ = toGobletXZ.negate().mul(cosA1).add(rightXZ.mul(sinA1));
      const yFrac1 = mix(float(-0.82), float(0.06), vN.mul(0.5).add(0.5));
      const rHoriz1 = rxz.mul(sqrt(max(float(0.06), float(1.0).sub(yFrac1.mul(yFrac1).mul(0.75)))));
      const pFront1 = vec3(
        uBowlCenter.x.add(frontDirXZ.x.mul(rHoriz1)),
        uBowlCenter.y.add(yFrac1.mul(ry)),
        uBowlCenter.z.add(frontDirXZ.y.mul(rHoriz1))
      );

      // Cylindrical-tulip horizontal refraction produces a long, razor-sharp A3 Cardioid/Astroid spear
      // spanning the full length of the goblet's shadow oval!
      const nCylFront1 = normalize(
        vec3(
          frontDirXZ.x.add(uSloshTilt.x.mul(0.24)),
          yFrac1.mul(0.22).add(waveH.mul(2.2)),
          frontDirXZ.y.add(uSloshTilt.y.mul(0.24))
        )
      );
      const inc1 = normalize(pFront1.sub(uLampPos));
      const refr1In = safeRefract(inc1, nCylFront1, float(1.0).div(nLiq));
      const dLiq1 = refr1In.xyz;

      // Chord across circular horizontal section of radius rHoriz1
      const incXZ1 = vec2(dLiq1.x, dLiq1.z);
      const relXZ1 = vec2(pFront1.x.sub(uBowlCenter.x), pFront1.z.sub(uBowlCenter.z));
      const aCyl1 = max(float(1e-4), dot(incXZ1, incXZ1));
      const bCyl1 = dot(relXZ1, incXZ1);
      const tChord1 = max(float(0.05), bCyl1.negate().mul(2.0).div(aCyl1));
      const pExit1 = pFront1.add(dLiq1.mul(tChord1));
      const exitRel1 = vec2(pExit1.x.sub(uBowlCenter.x), pExit1.z.sub(uBowlCenter.z));
      const nCylBack1 = normalize(vec3(exitRel1.x, yFrac1.mul(-0.14), exitRel1.y));
      const refr1Out = safeRefract(dLiq1, nCylBack1.negate(), nLiq);
      const dAir1 = refr1Out.xyz;
      const tSlab1 = slabY.sub(pExit1.y).div(min(float(-0.08), dAir1.y));
      const hit1 = pExit1.add(dAir1.mul(max(float(0.0), tSlab1)));
      const edgeFade1 = smoothstep(float(1.0), float(0.82), abs(uN)).mul(
        smoothstep(float(1.0), float(0.84), abs(vN))
      );
      const w1 = refr1In.w
        .mul(refr1Out.w)
        .mul(edgeFade1)
        .mul(smoothstep(float(-0.08), float(-0.20), dAir1.y));

      // ---------------------------------------------------------------------
      // Manifold 2: Concave Back-Wall Reflection Catacaustic ("Venetian Nephroid")
      // ---------------------------------------------------------------------
      // Cayley's catacaustic off the concave inner crystal bowl wall: reflects
      // forward into the dark shadow footprint as a crisp 2-cusped Nephroid heart!
      const az2 = uN.mul(1.46);
      const cosA2 = cos(az2);
      const sinA2 = sin(az2);
      const backDirXZ = toGobletXZ.mul(cosA2).add(rightXZ.mul(sinA2));
      const yFrac2 = mix(float(-0.42), float(0.72), vN.mul(0.5).add(0.5));
      const rHoriz2 = float(BOWL_RADIUS_XZ).mul(
        sqrt(max(float(0.08), float(1.0).sub(yFrac2.mul(yFrac2).mul(0.62))))
      );
      const pBack2 = vec3(
        uBowlCenter.x.add(backDirXZ.x.mul(rHoriz2)),
        uBowlCenter.y.add(yFrac2.mul(BOWL_RADIUS_Y)),
        uBowlCenter.z.add(backDirXZ.y.mul(rHoriz2))
      );
      const dispTilt = nGlass.sub(1.522).mul(0.95);
      // Slight upward inward-normal tilt (+0.16) throws the Nephroid forward into the shadow oval!
      const nConcaveIn = normalize(
        vec3(
          backDirXZ.x.negate(),
          float(0.16).sub(yFrac2.mul(0.12)).add(dispTilt),
          backDirXZ.y.negate()
        )
      );
      const inc2 = normalize(pBack2.sub(uLampPos));
      // Reflect horizontal components across concave circle while keeping oblique downward ray travel
      const reflRaw2 = inc2.sub(nConcaveIn.mul(dot(inc2, nConcaveIn).mul(2.0)));
      // Second pass forward fold: negate horizontal bounce so concave mirror focuses forward along +toGobletXZ!
      const reflHorizFwd = reflRaw2.xz.negate();
      const reflDir2 = normalize(vec3(reflHorizFwd.x, inc2.y.mul(0.92).sub(dispTilt.mul(0.4)), reflHorizFwd.y));
      const tSlab2 = slabY.sub(pBack2.y).div(min(float(-0.08), reflDir2.y));
      const hit2 = pBack2.add(reflDir2.mul(max(float(0.0), tSlab2)));
      const edgeFade2 = smoothstep(float(1.0), float(0.84), abs(uN)).mul(
        smoothstep(float(1.0), float(0.84), abs(vN))
      );
      const facing2 = smoothstep(float(0.02), float(0.24), dot(inc2, nConcaveIn.negate()));
      const w2 = edgeFade2.mul(facing2).mul(smoothstep(float(-0.08), float(-0.20), reflDir2.y));

      // Select active manifold cleanly without dynamic derivative issues
      const isSheet0 = sheetId.lessThan(0.5);
      const isSheet1 = sheetId.greaterThanEqual(0.5).and(sheetId.lessThan(1.5));

      const outXZ = select(
        isSheet0,
        vec2(hit0.x, hit0.z),
        select(isSheet1, vec2(hit1.x, hit1.z), vec2(hit2.x, hit2.z))
      );
      const outChord = select(isSheet0, tExit0, select(isSheet1, tChord1, float(0.18)));
      const outWeight = select(isSheet0, w0, select(isSheet1, w1, w2));

      const clampedXZ = clamp(
        outXZ,
        vec2(-CAUSTIC_MAP_EXTENT * 1.35),
        vec2(CAUSTIC_MAP_EXTENT * 1.35)
      );
      return vec4(clampedXZ.x, clampedXZ.y, outChord, outWeight);
    }
  );

  // Build 112x112 continuous wavefront plane instanced 9 times (3 manifolds x 3 mineral spectral bands)
  const WAVEFRONT_SEGS = 112;
  const wavefrontGeom = new THREE.PlaneGeometry(2.0, 2.0, WAVEFRONT_SEGS, WAVEFRONT_SEGS);
  const causticWavefrontMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  // Compute vertex position on causticRT + camera-invariant slab-space Jacobian flux!
  const wfUv = uv();
  const uParam = wfUv.x.mul(2.0).sub(1.0);
  const vParam = wfUv.y.mul(2.0).sub(1.0);

  const instF = float(instanceIndex);
  const bandIdx = instF.mod(3.0); // 0: Long (Realgar), 1: Mid (Spore/Ochre), 2: Short (Celadon/Azurite)
  const manifoldIdx = instF.div(3.0).floor(); // 0: Meniscus, 1: Bowl Lens, 2: Nephroid Catacaustic

  const isBand0 = bandIdx.lessThan(0.5);
  const isBand1 = bandIdx.greaterThanEqual(0.5).and(bandIdx.lessThan(1.5));

  const nLiqBand = select(isBand0, float(1.354), select(isBand1, float(1.368), float(1.384)));
  const nGlassBand = select(isBand0, float(1.506), select(isBand1, float(1.522), float(1.541)));

  const duStep = float(2.0 / WAVEFRONT_SEGS);
  const hitCenter = traceManifoldRay(uParam, vParam, manifoldIdx, nLiqBand, nGlassBand);
  const hitRight = traceManifoldRay(
    uParam.add(duStep),
    vParam,
    manifoldIdx,
    nLiqBand,
    nGlassBand
  );
  const hitUp = traceManifoldRay(uParam, vParam.add(duStep), manifoldIdx, nLiqBand, nGlassBand);

  // Exact 2D Jacobian area element on the Bone Calcite slab: |dP/du x dP/dv|
  const dPdu = hitRight.xy.sub(hitCenter.xy);
  const dPdv = hitUp.xy.sub(hitCenter.xy);
  const areaSlab = abs(dPdu.x.mul(dPdv.y).sub(dPdu.y.mul(dPdv.x)));
  const areaSource = duStep.mul(duStep).mul(0.24);

  // Suppress stretched silhouette/grazing triangles where neighbor jump exceeds physical threshold
  const maxEdgeLenSq = max(dot(dPdu, dPdu), dot(dPdv, dPdv));
  const stretchGuard = smoothstep(float(0.095), float(0.022), maxEdgeLenSq);

  // Finite-aperture condenser Airy/coma regularization prevents infinite pixel spikes while preserving razor cusps
  const epsArea = duStep.mul(duStep).mul(0.024);
  const fluxRatio = clamp(
    areaSource.div(sqrt(areaSlab.mul(areaSlab).add(epsArea.mul(epsArea)))),
    0.0,
    16.0
  );

  // Beer-Lambert liquid chord transmittance through Celadon Glauconite + Realgar Amber elixir
  const elixirExtinction = mix(celadon.extinction, realgar.extinction, float(0.44)).mul(0.78);
  const beerTint = beerLambertTransmittance(elixirExtinction, hitCenter.z);

  // Mineral-grounded 3-band spectral carriers (summing to warm incandescent calcite studio beam)
  const colBand0 = mix(realgar.rInf, calcite.rInf, float(0.32)).mul(vec3(1.22, 0.56, 0.16));
  const colBand1 = mix(spore.rInf, ochre.rInf, float(0.35)).mul(vec3(0.58, 0.90, 0.36));
  const colBand2 = mix(celadon.rInf, azurite.rInf, float(0.24)).mul(vec3(0.20, 0.58, 1.12));
  const spectralCarrier = select(isBand0, colBand0, select(isBand1, colBand1, colBand2));

  const sheetGain = select(
    manifoldIdx.lessThan(0.5),
    float(0.165), // Manifold 0: Rippling meniscus caustic web
    select(
      manifoldIdx.lessThan(1.5),
      float(0.185), // Manifold 1: Cardioid focal spear
      float(0.155) // Manifold 2: Venetian Nephroid arch
    )
  );

  const vCausticRadiance = varying(
    spectralCarrier
      .mul(beerTint)
      .mul(fluxRatio)
      .mul(sheetGain)
      .mul(hitCenter.w)
      .mul(hitRight.w)
      .mul(hitUp.w)
      .mul(stretchGuard)
  );

  causticWavefrontMat.positionNode = vec3(hitCenter.x, hitCenter.y.negate(), float(0.0));
  causticWavefrontMat.colorNode = vCausticRadiance;
  causticWavefrontMat.opacityNode = float(1.0);

  const causticInstancedMesh = new THREE.InstancedMesh(wavefrontGeom, causticWavefrontMat, 9);
  causticInstancedMesh.frustumCulled = false;
  const _idMat = new THREE.Matrix4();
  for (let i = 0; i < 9; i++) {
    causticInstancedMesh.setMatrixAt(i, _idMat);
  }
  causticScene.add(causticInstancedMesh);

  // -------------------------------------------------------------------------
  // 1. Beveled Octagonal Bone Calcite Receiver Slab + Turned Brass Feet
  // -------------------------------------------------------------------------
  const slabGeom = createBeveledCalciteSlabGeometry();
  const slabMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.68,
    metalness: 0.02,
  });

  const wPos = positionWorld;
  const wNorm = normalize(normalWorld);
  const topFaceMask = smoothstep(float(0.75), float(0.96), wNorm.y);

  // Honed Warm Kaolin & Bone Calcite travertine stone body (calibrated so HDR caustics pop with high contrast)
  const stoneGrain = mineralToothFbm(wPos.mul(vec3(1.8, 4.5, 1.8)), float(2.4));
  const fineTooth = mineralToothFbm(wPos.mul(5.5), float(4.2));
  const kmThickness = clamp(float(0.58).add(stoneGrain.mul(0.55)).add(fineTooth.mul(0.18)), 0.22, 1.5);
  const rawStone = kubelkaMunkGlaze(kaolin.K, kaolin.S, kmThickness, umber.rInf);
  const baseStone = mix(rawStone, umber.rInf, float(0.16));
  const veinedStone = mix(baseStone, umber.rInf.mul(0.95), smoothstep(0.62, 0.86, stoneGrain).mul(0.18));

  // Analytical ray-goblet dielectric Fresnel shadow footprint on the top face!
  // Deflects direct studio light out of the goblet's silhouette so caustics blaze inside it.
  const rayToLamp = normalize(uLampPos.sub(wPos));
  const occlBowlO = wPos.sub(uBowlCenter).div(vec3(BOWL_RADIUS_XZ, BOWL_RADIUS_Y, BOWL_RADIUS_XZ));
  const occlBowlD = rayToLamp.div(vec3(BOWL_RADIUS_XZ, BOWL_RADIUS_Y, BOWL_RADIUS_XZ));
  const qa = max(float(1e-5), dot(occlBowlD, occlBowlD));
  const qb = dot(occlBowlO, occlBowlD);
  const qc = dot(occlBowlO, occlBowlO).sub(1.0);
  const qDisc = qb.mul(qb).sub(qa.mul(qc));
  const tBowlHit = qb.negate().sub(sqrt(max(float(0.0), qDisc))).div(qa);
  const pBowlHit = wPos.add(rayToLamp.mul(max(float(0.0), tBowlHit)));
  const bowlShadow = smoothstep(float(-0.04), float(0.14), qDisc)
    .mul(smoothstep(float(0.0), float(0.08), tBowlHit))
    .mul(smoothstep(float(BOWL_LIP_Y + 0.03), float(BOWL_LIP_Y - 0.04), pBowlHit.y));

  // Stem + flared foot shadow contribution near goblet base
  const dFootXZ = length(vec2(wPos.x.sub(GOBLET_POS_X), wPos.z.sub(GOBLET_POS_Z)));
  const footRingShadow = smoothstep(float(0.39), float(0.26), dFootXZ).mul(0.52);
  const totalRefractiveShadow = clamp(max(bowlShadow.mul(0.84), footRingShadow), 0.0, 0.86).mul(
    topFaceMask
  );

  // Sample 1024x1024 HDR caustic map + 4-tap subsurface calcite lateral scatter
  const slabUv = clamp(
    vec2(wPos.x, wPos.z).div(CAUSTIC_MAP_EXTENT * 2.0).add(0.5),
    0.002,
    0.998
  );
  const texelScatter = float(0.0032);
  const cSharp = texture(causticRT.texture, slabUv).rgb;
  const cSub1 = texture(causticRT.texture, slabUv.add(vec2(texelScatter, 0.0))).rgb;
  const cSub2 = texture(causticRT.texture, slabUv.sub(vec2(texelScatter, 0.0))).rgb;
  const cSub3 = texture(causticRT.texture, slabUv.add(vec2(0.0, texelScatter))).rgb;
  const cSub4 = texture(causticRT.texture, slabUv.sub(vec2(0.0, texelScatter))).rgb;
  const cSubsurface = cSub1
    .add(cSub2)
    .add(cSub3)
    .add(cSub4)
    .mul(0.25)
    .mul(mix(realgar.rInf, calcite.rInf, float(0.50)));

  // Soft edge mask near beveled octagonal slab boundary
  const octEdgeMask = smoothstep(float(1.66), float(1.48), abs(wPos.x))
    .mul(smoothstep(float(1.36), float(1.20), abs(wPos.z)))
    .mul(topFaceMask);

  // Cool slate-umber shadow tint where goblet steals direct light
  const coolShadowTint = mix(umber.rInf, azurite.rInf, float(0.34)).mul(0.34);
  const shadowedStone = mix(veinedStone, veinedStone.mul(coolShadowTint.add(0.15)), totalRefractiveShadow);

  // Compress ultra-bright cusp peaks gently via shoulder curve while keeping glowing warm calcite hearts
  const rawCaustic = cSharp.mul(1.65).add(cSubsurface.mul(0.68)).mul(octEdgeMask);
  const compressedCaustic = rawCaustic.div(vec3(1.0).add(rawCaustic.mul(0.22)));

  slabMat.colorNode = shadowedStone;
  slabMat.emissiveNode = compressedCaustic.mul(calcite.rInf.add(0.28));

  const slabMesh = new THREE.Mesh(slabGeom, slabMat);
  slabMesh.receiveShadow = true;
  slabMesh.castShadow = true;
  stage.specimenGroup.add(slabMesh);

  // 4 Turned Goethite-Brass Leveling Feet under the suspended slab
  const brassMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.28,
    metalness: 0.86,
  });
  brassMat.colorNode = mix(ochre.rInf, calcite.rInf, float(0.22));

  const footPostGeom = new THREE.CylinderGeometry(0.055, 0.072, 0.09, 24);
  const footCoords: Array<[number, number]> = [
    [-1.12, -0.92],
    [1.12, -0.92],
    [1.12, 0.92],
    [-1.12, 0.92],
  ];
  for (const [fx, fz] of footCoords) {
    const fMesh = new THREE.Mesh(footPostGeom, brassMat);
    fMesh.position.set(fx, SLAB_TOP_Y - SLAB_THICKNESS - 0.045, fz);
    stage.specimenGroup.add(fMesh);
  }

  // -------------------------------------------------------------------------
  // 2. Hand-Blown Venetian Crystal Goblet + Swirling Botanical-Mineral Elixir
  // -------------------------------------------------------------------------
  const gobletGroup = new THREE.Group();
  gobletGroup.position.set(GOBLET_POS_X, SLAB_TOP_Y, GOBLET_POS_Z);
  stage.specimenGroup.add(gobletGroup);

  const { bowlGeom, footGeom, stemGeom, lipRingGeom } = createVenetianCrystalBowlAndFoot();

  // Dynamic Swirling Liquid Volume Material (Clear Beer-Lambert Celadon-Emerald & Golden-Amber Elixir)
  const liquidMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    roughness: 0.04,
    metalness: 0.0,
    side: THREE.FrontSide,
  });

  const aLiqOpt = attribute('aLiquidOptics', 'vec3');
  const liqViewDir = normalize(cameraPosition.sub(positionWorld));
  const liqNormW = normalize(normalWorld);
  const liqNdotV = clamp(abs(dot(liqNormW, liqViewDir)), 0.0, 1.0);
  const liqFresnel = float(0.024).add(float(0.976).mul(float(1.0).sub(liqNdotV).pow(5.0)));

  // Pure Beer-Lambert transmission of warm calcite plinth bounce light through Celadon + Spore/Amber elixir
  const liqExtinction = mix(celadon.extinction, ochre.extinction, float(0.34)).mul(1.15);
  const liqTrans = beerLambertTransmittance(liqExtinction, aLiqOpt.x.add(0.18));
  const warmBounce = mix(calcite.rInf, spore.rInf, float(0.28));
  const clearElixir = warmBounce.mul(liqTrans);

  // Curved studio softbox window reflection on the rippling free meniscus (zero Phong dots!)
  const liqReflDir = liqViewDir.negate().sub(liqNormW.mul(dot(liqViewDir.negate(), liqNormW).mul(2.0)));
  const softboxAlign = smoothstep(
    float(0.58),
    float(0.92),
    dot(normalize(liqReflDir), normalize(uLampPos.sub(positionWorld)))
  );
  const rimGlow = mix(spore.rInf, realgar.rInf, float(0.38)).mul(aLiqOpt.y.mul(0.28));

  liquidMat.colorNode = clearElixir.mul(0.55);
  liquidMat.emissiveNode = clearElixir
    .mul(0.46)
    .add(rimGlow)
    .add(calcite.rInf.mul(softboxAlign.mul(0.52)))
    .add(calcite.rInf.mul(liqFresnel.mul(0.24)));
  liquidMat.opacityNode = clamp(
    float(0.38).add(aLiqOpt.x.mul(0.22)).add(liqFresnel.mul(0.28)),
    0.36,
    0.82
  );

  const dynamicLiquid = new DynamicLiquidMesh(liquidMat);
  dynamicLiquid.updateFromSim(meniscusSim);
  gobletGroup.add(dynamicLiquid.mesh);

  // Thin-Walled Blown Celadon Soda-Lime Crystal Bowl & Foot Material
  const crystalMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    roughness: 0.04,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });

  const glassViewDir = normalize(cameraPosition.sub(positionWorld));
  const glassNormW = normalize(normalWorld);
  const glassNdotV = clamp(abs(dot(glassNormW, glassViewDir)), 0.0, 1.0);
  const glassFresnel = float(0.042).add(float(0.958).mul(float(1.0).sub(glassNdotV).pow(4.2)));

  // Grazing Beer-Lambert celadon-calcite glass rim tint + curved studio strip speculars
  const wallChord = float(0.08).div(max(float(0.14), glassNdotV));
  const glassTint = beerLambertTransmittance(celadon.extinction.mul(0.48), wallChord);
  const glassRefl = glassViewDir.negate().sub(glassNormW.mul(dot(glassViewDir.negate(), glassNormW).mul(2.0)));
  const stripSpec1 = smoothstep(float(0.76), float(0.96), dot(normalize(glassRefl), normalize(uLampPos.sub(positionWorld))));
  const stripSpec2 = smoothstep(float(0.65), float(0.92), glassRefl.y).mul(0.32);

  crystalMat.colorNode = mix(glassTint.mul(calcite.rInf), calcite.rInf, glassFresnel);
  crystalMat.emissiveNode = calcite.rInf.mul(stripSpec1.mul(0.75).add(stripSpec2).add(glassFresnel.mul(0.22)));
  crystalMat.opacityNode = clamp(float(0.11).add(glassFresnel.mul(0.72)).add(stripSpec1.mul(0.35)), 0.11, 0.88);

  const bowlMesh = new THREE.Mesh(bowlGeom, crystalMat);
  bowlMesh.renderOrder = 3;
  gobletGroup.add(bowlMesh);

  const footMesh = new THREE.Mesh(footGeom, crystalMat);
  footMesh.renderOrder = 2;
  gobletGroup.add(footMesh);

  // Murano Realgar-Amber Knopped Baluster Stem + Fire-Polished Lip Thread
  const amberGlassMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: true,
    roughness: 0.08,
    metalness: 0.0,
  });
  const stemViewDir = normalize(cameraPosition.sub(positionWorld));
  const stemNormW = normalize(normalWorld);
  const stemNdotV = clamp(abs(dot(stemNormW, stemViewDir)), 0.0, 1.0);
  const stemFresnel = float(0.045).add(float(0.955).mul(float(1.0).sub(stemNdotV).pow(4.0)));
  const stemChord = float(0.42).add(float(1.0).sub(stemNdotV).mul(0.55));
  const amberTrans = beerLambertTransmittance(realgar.extinction.mul(1.15), stemChord);
  // Subtle blown gadroon rib modulation on the central knop
  const stemRib = sin(positionLocal.y.mul(36.0).add(normalLocal.x.mul(3.0))).mul(0.06);
  amberGlassMat.colorNode = realgar.rInf.mul(amberTrans.add(0.18)).add(stemRib);
  amberGlassMat.emissiveNode = mix(realgar.rInf, calcite.rInf, float(0.45)).mul(stemFresnel.mul(0.42));
  amberGlassMat.opacityNode = clamp(float(0.76).add(stemFresnel.mul(0.22)), 0.76, 0.98);

  const stemMesh = new THREE.Mesh(stemGeom, amberGlassMat);
  stemMesh.renderOrder = 2;
  gobletGroup.add(stemMesh);

  const lipRingMesh = new THREE.Mesh(lipRingGeom, amberGlassMat);
  lipRingMesh.renderOrder = 4;
  gobletGroup.add(lipRingMesh);

  // -------------------------------------------------------------------------
  // 3. Steerable Burnished Brass Condenser Collimator Lamp Housing
  // -------------------------------------------------------------------------
  const lampGroup = new THREE.Group();
  stage.specimenGroup.add(lampGroup);

  const housingGeom = new THREE.CylinderGeometry(0.095, 0.125, 0.22, 32, 1, true);
  housingGeom.rotateX(Math.PI * 0.5);
  const housingMesh = new THREE.Mesh(housingGeom, brassMat);
  lampGroup.add(housingMesh);

  const bezelGeom = new THREE.TorusGeometry(0.125, 0.014, 12, 32);
  const bezelMesh = new THREE.Mesh(bezelGeom, brassMat);
  bezelMesh.position.z = 0.11;
  lampGroup.add(bezelMesh);

  const condenserLensMat = new THREE.MeshBasicNodeMaterial();
  condenserLensMat.colorNode = mix(calcite.rInf, realgar.rInf, float(0.22)).mul(1.55);
  const lensDiscGeom = new THREE.CircleGeometry(0.118, 32);
  const lensDiscMesh = new THREE.Mesh(lensDiscGeom, condenserLensMat);
  lensDiscMesh.position.z = 0.105;
  lampGroup.add(lensDiscMesh);

  const backCapGeom = new THREE.SphereGeometry(0.095, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.5);
  backCapGeom.rotateX(-Math.PI * 0.5);
  const backCapMesh = new THREE.Mesh(backCapGeom, brassMat);
  backCapMesh.position.z = -0.11;
  lampGroup.add(backCapMesh);

  // -------------------------------------------------------------------------
  // Monotonic Steerable Light State & Tactile Pointer Interaction
  // -------------------------------------------------------------------------
  let lightAzimuth = Math.PI * 1.16; // ~209 deg: shines diagonally from back-left across slab
  let lightElevation = 0.69; // ~39.5 deg: unfolds crisp nephroid + long focal spear
  let targetAzimuth = lightAzimuth;
  let targetElevation = lightElevation;
  const LAMP_DISTANCE = 2.65;

  const syncLampBasis = (az: number, el: number) => {
    const cosEl = Math.cos(el);
    const sinEl = Math.sin(el);
    _lampWorldPos.set(
      GOBLET_POS_X + cosEl * Math.cos(az) * LAMP_DISTANCE,
      BOWL_CENTER_Y + sinEl * LAMP_DISTANCE,
      GOBLET_POS_Z + cosEl * Math.sin(az) * LAMP_DISTANCE
    );

    uLampPos.value.copy(_lampWorldPos);
    _beamForward.copy(_gobletBowlCenter).sub(_lampWorldPos).normalize();
    _beamRight.crossVectors(_beamForward, _worldUp).normalize();
    _beamUp.crossVectors(_beamRight, _beamForward).normalize();

    uBeamForward.value.copy(_beamForward);
    uBeamRight.value.copy(_beamRight);
    uBeamUp.value.copy(_beamUp);

    lampGroup.position.copy(_lampWorldPos);
    lampGroup.lookAt(_gobletBowlCenter);
    condenserSpot.position.copy(_lampWorldPos);
  };

  syncLampBasis(lightAzimuth, lightElevation);

  // Continuous, zero-quadrant-jump pointer gesture controller
  type DragMode = 'none' | 'light' | 'goblet';
  let dragMode: DragMode = 'none';
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartAz = lightAzimuth;
  let dragStartEl = lightElevation;
  let prevClientX = 0;
  let prevClientY = 0;
  let pointerMovedPx = 0;

  const interactiveGobletTargets: THREE.Object3D[] = [bowlMesh, dynamicLiquid.mesh, stemMesh, footMesh];
  const interactiveLightTargets: THREE.Object3D[] = [slabMesh, housingMesh, lensDiscMesh];

  const updateRaycasterFromEvent = (ev: PointerEvent) => {
    const rect = stage.shell.canvas.getBoundingClientRect();
    const nx = ((ev.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    const ny = -((ev.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
    _raycaster.setFromCamera(new THREE.Vector2(nx, ny), stage.camera);
  };

  const onPointerDown = (ev: PointerEvent) => {
    updateRaycasterFromEvent(ev);
    dragStartX = ev.clientX;
    dragStartY = ev.clientY;
    prevClientX = ev.clientX;
    prevClientY = ev.clientY;
    pointerMovedPx = 0;

    if (_raycaster.intersectObjects(interactiveGobletTargets, false).length > 0) {
      dragMode = 'goblet';
      stage.controls.enabled = false;
      if (_raycaster.ray.intersectPlane(_meniscusPlane, _hitVec)) {
        const nx = (_hitVec.x - GOBLET_POS_X) / BOWL_RADIUS_XZ;
        const nz = (_hitVec.z - GOBLET_POS_Z) / BOWL_RADIUS_XZ;
        meniscusSim.injectDropImpulse(
          Math.max(-0.85, Math.min(0.85, nx)),
          Math.max(-0.85, Math.min(0.85, nz)),
          -0.018,
          0.22
        );
      }
      return;
    }

    if (_raycaster.intersectObjects(interactiveLightTargets, false).length > 0) {
      dragMode = 'light';
      stage.controls.enabled = false;
      dragStartAz = targetAzimuth;
      dragStartEl = targetElevation;
      return;
    }

    dragMode = 'none';
    stage.controls.enabled = true;
  };

  const onPointerMove = (ev: PointerEvent) => {
    const dxStep = ev.clientX - prevClientX;
    const dyStep = ev.clientY - prevClientY;
    prevClientX = ev.clientX;
    prevClientY = ev.clientY;
    pointerMovedPx += Math.hypot(dxStep, dyStep);

    updateRaycasterFromEvent(ev);

    // Passive hover breeze across the open goblet stirs gentle capillary ripples
    if (dragMode === 'none') {
      if (_raycaster.ray.intersectPlane(_meniscusPlane, _hitVec)) {
        const lx = (_hitVec.x - GOBLET_POS_X) / BOWL_RADIUS_XZ;
        const lz = (_hitVec.z - GOBLET_POS_Z) / BOWL_RADIUS_XZ;
        if (lx * lx + lz * lz < 0.92 && Math.hypot(dxStep, dyStep) > 1.5) {
          meniscusSim.injectDropImpulse(lx, lz, 0.0055, 0.19);
          meniscusSim.addSloshImpulse(dxStep * 0.012, dyStep * 0.012);
        }
      }
      return;
    }

    if (dragMode === 'goblet') {
      meniscusSim.addSloshImpulse(dxStep * 0.042, dyStep * 0.042);
      if (_raycaster.ray.intersectPlane(_meniscusPlane, _hitVec)) {
        const lx = Math.max(-0.86, Math.min(0.86, (_hitVec.x - GOBLET_POS_X) / BOWL_RADIUS_XZ));
        const lz = Math.max(-0.86, Math.min(0.86, (_hitVec.z - GOBLET_POS_Z) / BOWL_RADIUS_XZ));
        meniscusSim.injectDropImpulse(lx, lz, 0.009, 0.18);
      }
      // Subtle sympathetic light sweep so dragging right through center produces rich caustic shift
      targetAzimuth -= dxStep * 0.0022;
      return;
    }

    if (dragMode === 'light') {
      const totalDx = ev.clientX - dragStartX;
      const totalDy = ev.clientY - dragStartY;
      targetAzimuth = dragStartAz + totalDx * 0.0072;
      targetElevation = Math.max(0.42, Math.min(1.16, dragStartEl - totalDy * 0.0048));
      // Coupled gentle table-vibration ripple so meniscus shimmers while orbiting the beam
      meniscusSim.addSloshImpulse(dxStep * 0.008, dyStep * 0.008);
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (dragMode !== 'none' && pointerMovedPx < 6) {
      updateRaycasterFromEvent(ev);
      // Crisp glass-rim tap impulse: excites slosh + off-center capillary splash
      meniscusSim.addSloshImpulse(0.42, -0.31);
      if (_raycaster.ray.intersectPlane(_slabPlane, _hitVec)) {
        const lx = Math.max(-0.75, Math.min(0.75, (_hitVec.x - GOBLET_POS_X) / BOWL_RADIUS_XZ));
        const lz = Math.max(-0.75, Math.min(0.75, (_hitVec.z - GOBLET_POS_Z) / BOWL_RADIUS_XZ));
        meniscusSim.injectDropImpulse(lx * 0.65, lz * 0.65, -0.024, 0.21);
      } else {
        meniscusSim.injectDropImpulse(0.24, -0.18, -0.024, 0.21);
      }
    }
    dragMode = 'none';
    stage.controls.enabled = true;
  };

  stage.shell.canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });

  stage.shell.registerDispose(() => {
    stage.shell.canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    causticRT.dispose();
    meniscusSim.dataTexture.dispose();
    wavefrontGeom.dispose();
    slabGeom.dispose();
    bowlGeom.dispose();
    footGeom.dispose();
    stemGeom.dispose();
    lipRingGeom.dispose();
    dynamicLiquid.geometry.dispose();
  });

  // -------------------------------------------------------------------------
  // Main Frame Loop: Step Meniscus PDE -> Render Forward Caustic Map -> Scene
  // -------------------------------------------------------------------------
  stage.onTick(({ dtSec, elapsedSec, renderer }) => {
    // Smooth critically-damped light azimuth/elevation tracking + subtle breathing drift when idle
    const idleDrift = dragMode === 'none' ? Math.sin(elapsedSec * 0.42) * 0.0018 : 0.0;
    const smoothRate = 1.0 - Math.exp(-dtSec * 14.0);
    lightAzimuth += (targetAzimuth - lightAzimuth) * smoothRate + idleDrift;
    lightElevation += (targetElevation - lightElevation) * smoothRate;
    syncLampBasis(lightAzimuth, lightElevation);

    // Step 2D meniscus wave PDE + slosh pendulum and sync 3D liquid volume mesh
    meniscusSim.step(dtSec, elapsedSec);
    uSloshTilt.value.set(meniscusSim.sloshX, meniscusSim.sloshZ);
    dynamicLiquid.updateFromSim(meniscusSim);

    // Render 9-sheet forward ray-traced caustic wavefront into 1024x1024 HalfFloat RT
    renderer.setRenderTarget(causticRT);
    renderer.render(causticScene, causticCam);
    renderer.setRenderTarget(null);
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay10Caustic();
}
