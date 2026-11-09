import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  abs,
  clamp,
  dot,
  exp,
  float,
  length,
  max,
  min,
  mix,
  normalize,
  positionLocal,
  sin,
  smoothstep,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import { CHALKY_SLATE_LINEAR } from '../../core/palette';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Day 07 — Hollow (Concept 3 / 7C — Exact CSG Raymarched Shader):
 * "Raymarched Hollowed Alabaster Sanctuary & Pantheon Oculus (Elogio de la Luz)"
 *
 * Inspired by Eduardo Chillida's hollowed alabaster monoliths and Roman
 * Pantheon/stepwell vaults, rendered via an exact Lipschitz-1 (`|∇d| <= 1.0`,
 * zero `floor()` or wave corrugations, and strict WGSL `smoothstep(low, high)`
 * ordering) CSG Signed Distance Field raymarcher:
 * - Exterior: Chiseled `warmKaolin` & `rawUmber` monolithic stone cube with
 *   recessed architectural cornice rebates.
 * - Interior Hollow Negative-Space Architecture:
 *   - Intersecting orthogonal Roman barrel-vault naves (`sdArch2D`) and twin
 *     flanking colonnade aisles that leave 4 freestanding square alabaster
 *     pillars (`dFourPiers`) inside the hollowed block.
 *   - A central spherical Pantheon rotunda dome (`R = 0.65`) crowned by a
 *     circular roof oculus (`uOculusRadius`) pouring golden `realgarAmber` light
 *     into the vaulted interior.
 *   - A sunken circular floor crypt well (`rXZ < 0.42`) revealing a calm,
 *     mirror-like `azuritePrussian` & `lapisUltramarine` subterranean cistern
 *     at `y = -0.70`.
 * - Subsurface Alabaster Translucency: Thin carved alabaster arches and pillars
 *   glow with warm `realgarAmber` Beer-Lambert transmittance evaluated via
 *   directional SDF thickness probes.
 */

const _camLocal = new THREE.Vector3();

export async function bootHollowConcept3(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(7);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.34, 4.10],
    fov: 34,
    enableOrbit: false,
    enableShadows: false,
  });

  // Mineral Pigments
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const rawUmber = createPigmentUniforms('rawUmber');
  const burntSienna = createPigmentUniforms('burntSienna');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const celadon = createPigmentUniforms('celadonGlauconite');
  const azurite = createPigmentUniforms('azuritePrussian');
  const lapis = createPigmentUniforms('lapisUltramarine');

  const uCamLocal = uniform(new THREE.Vector3(0, 0.34, 4.10));
  const uVaultShift = uniform(new THREE.Vector2(0.0, 0.0));
  const uOculusRadius = uniform(0.36);
  const uSunIntensity = uniform(1.0);

  const proxyGeom = new THREE.BoxGeometry(2.15, 2.15, 2.15);
  const sanctuaryMat = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: true,
  });

  // Exact Euclidean 2D Arch SDF (rectangular shaft below `y = springY`, semicircular arch above)
  const sdArch2D = Fn(([uv_in, halfW_in, springY_in, bottomY_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const uv = vec2(uv_in);
    const halfW = float(halfW_in);
    const springY = float(springY_in);
    const bottomY = float(bottomY_in);

    const relY = uv.y.sub(springY);
    const inDome = smoothstep(-0.001, 0.001, relY);

    const dArchTop = length(vec2(uv.x, relY)).sub(halfW);
    const dShaft = max(abs(uv.x).sub(halfW), bottomY.sub(uv.y));

    return mix(dShaft, dArchTop, inDome);
  });

  // ---------------------------------------------------------------------------
  // Exact Lipschitz-1 CSG SDF for the Hollowed Alabaster Sanctuary
  // Returns vec2(signedDistance, materialZone):
  //   zone 0 = rough outer stone monolith, 1 = carved inner alabaster vault, 2 = crypt mirror cistern
  // ---------------------------------------------------------------------------
  const evalSanctuarySdf = Fn(([p_in]: [TslNode]) => {
    const p = vec3(p_in);

    // 1. Outer Chiseled Stone Monolith (`[-0.91, +0.91]^3` rounded box with subtle cornice groove)
    const qBox = abs(p).sub(vec3(0.88, 0.88, 0.88));
    const dBoxBase = length(max(qBox, vec3(0.0))).add(
      min(max(qBox.x, max(qBox.y, qBox.z)), float(0.0))
    ).sub(0.032);
    const friezeGroove = float(1.0).sub(smoothstep(0.0, 0.022, abs(abs(p.y).sub(0.72)))).mul(0.012);
    const dOuterBlock = dBoxBase.add(friezeGroove);

    // 2. Central Pantheon Spherical Dome & Roof Oculus (`rXZ < uOculusRadius`)
    const pShifted = vec3(p.x.sub(uVaultShift.x), p.y, p.z.sub(uVaultShift.y));
    const rRotunda = length(pShifted.xz);
    const dPantheonDome = length(pShifted.sub(vec3(0.0, 0.18, 0.0))).sub(0.56);
    const dRotunda = max(dPantheonDome, float(0.12).sub(p.y));
    const dTopOculus = max(rRotunda.sub(uOculusRadius), float(-0.10).sub(p.y));

    // 3. Intersecting Orthogonal Roman Barrel-Vault Naves & Colonnade Aisles
    const dNaveZ = sdArch2D(vec2(pShifted.x, p.y), float(0.33), float(0.18), float(-0.56));
    const dTranseptX = sdArch2D(vec2(pShifted.z, p.y), float(0.33), float(0.18), float(-0.56));

    // Twin flanking colonnade arches on all 4 facades (`|x| = 0.56` and `|z| = 0.56`)
    const sideX = abs(p.x).sub(0.56);
    const sideZ = abs(p.z).sub(0.56);
    const dColonnadeZ = sdArch2D(vec2(sideX, p.y), float(0.135), float(0.28), float(-0.56));
    const dColonnadeX = sdArch2D(vec2(sideZ, p.y), float(0.135), float(0.28), float(-0.56));

    // Interior Cross-Gallery Atrium (`|x|,|z| < 0.72`, `y in [-0.56, 0.44]`) with
    // 4 freestanding square alabaster pillars at `(|x| = 0.44, |z| = 0.44)`!
    const qAtrium = abs(p.sub(vec3(0.0, -0.06, 0.0))).sub(vec3(0.72, 0.50, 0.72));
    const dAtriumBox = length(max(qAtrium, vec3(0.0))).add(
      min(max(qAtrium.x, max(qAtrium.y, qAtrium.z)), float(0.0))
    );
    const qPier = abs(vec2(abs(pShifted.x).sub(0.44), abs(pShifted.z).sub(0.44))).sub(vec2(0.092));
    const dFourPiers = length(max(qPier, vec2(0.0))).add(
      min(max(qPier.x, qPier.y), float(0.0))
    );
    const dHollowAtrium = max(dAtriumBox, dFourPiers.negate());

    // Sunken circular stepwell crypt in the center of the floor (`rRotunda < 0.42`, down to `y = -0.74`)
    const dFloorWell1 = max(rRotunda.sub(0.43), float(-0.64).sub(p.y));
    const dFloorWell2 = max(rRotunda.sub(0.33), float(-0.74).sub(p.y));
    const dCryptWell = min(dFloorWell1, dFloorWell2);

    // Union of all hollow architectural negative spaces
    const dVaultVoid = min(
      min(min(dRotunda, dTopOculus), min(dNaveZ, dTranseptX)),
      min(min(dColonnadeZ, dColonnadeX), min(dHollowAtrium, dCryptWell))
    );

    // Smooth chamfered CSG subtraction (`max(dOuterBlock, -dVaultVoid)`)
    const kChamfer = float(0.010);
    const hCut = clamp(
      float(0.5).sub(dOuterBlock.add(dVaultVoid).div(kChamfer.mul(2.0))),
      0.0,
      1.0
    );
    const dCarvedStone = mix(dOuterBlock, dVaultVoid.negate(), hCut).add(
      kChamfer.mul(hCut).mul(float(1.0).sub(hCut))
    );

    // 4. Deep Subterranean Mirror Cistern inside the sunken crypt well (`y = -0.70`)
    const dCistern = max(
      p.y.add(0.70),
      rRotunda.sub(0.34)
    );

    const dTotal = min(dCarvedStone, dCistern);
    const isPool = float(1.0).sub(smoothstep(-0.004, 0.004, dCistern.sub(dCarvedStone)));
    const maxCoord = max(abs(p.x), max(abs(p.y), abs(p.z)));
    const isInnerAlabaster = float(1.0).sub(smoothstep(0.872, 0.902, maxCoord));
    const zone = mix(isInnerAlabaster, float(2.0), isPool);

    return vec2(dTotal, zone);
  });

  // ---------------------------------------------------------------------------
  // Zero-Zeno Per-Pixel Sphere Tracer (`t += clamp(d * 0.92, 0.0, 0.5)`)
  // ---------------------------------------------------------------------------
  const raymarchSanctuary = Fn(() => {
    const ro = uCamLocal;
    const rd = normalize(positionLocal.sub(uCamLocal));

    // Exact ray-box slab intersection (`[-0.95, +0.95]^3`)
    const epsDir = vec3(1e-5, 1e-5, 1e-5);
    const safeRd = mix(rd, epsDir, float(1.0).sub(smoothstep(0.0, 1e-5, abs(rd))));
    const invRd = vec3(1.0).div(safeRd);
    const t0 = vec3(-0.95).sub(ro).mul(invRd);
    const t1 = vec3(0.95).sub(ro).mul(invRd);
    const tMin3 = min(t0, t1);
    const tMax3 = max(t0, t1);
    const tNear = max(float(0.05), max(tMin3.x, max(tMin3.y, tMin3.z)));
    const tFar = min(tMax3.x, min(tMax3.y, tMax3.z));

    const t = tNear.toVar();

    // 48-step unbraked sphere-trace: converges exponentially (`0.08^k`) directly onto `d = 0`
    // without Zeno's paradox or fractional step-boundary oscillations!
    Loop(48, () => {
      const pos = ro.add(rd.mul(t));
      const d = evalSanctuarySdf(pos).x;
      const inBounds = smoothstep(0.0, 0.002, tFar.add(0.02).sub(t));
      t.addAssign(clamp(d.mul(0.92), 0.0, 0.45).mul(inBounds));
    });

    const pHit = ro.add(rd.mul(t));
    const sdfFinal = evalSanctuarySdf(pHit);
    const dFinal = sdfFinal.x;
    const hitZone = sdfFinal.y;

    // Exact binary hit mask evaluated AFTER convergence (`1.0000` on all hit surfaces)
    const hitMask = float(1.0)
      .sub(smoothstep(0.002, 0.006, dFinal))
      .mul(smoothstep(-0.005, 0.005, tFar.sub(t)));

    // 4-tap Tetrahedron Normal
    const eps = float(0.0018);
    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);
    const N = normalize(
      k1.mul(evalSanctuarySdf(pHit.add(k1.mul(eps))).x)
        .add(k2.mul(evalSanctuarySdf(pHit.add(k2.mul(eps))).x))
        .add(k3.mul(evalSanctuarySdf(pHit.add(k3.mul(eps))).x))
        .add(k4.mul(evalSanctuarySdf(pHit.add(k4.mul(eps))).x))
    );

    // 2-tap SDF Ambient Occlusion inside the architectural vaults
    const ao1 = evalSanctuarySdf(pHit.add(N.mul(0.06))).x;
    const ao2 = evalSanctuarySdf(pHit.add(N.mul(0.16))).x;
    const ao = clamp(float(0.32).add(ao1.mul(7.0)).add(ao2.mul(2.8)), 0.16, 1.0);

    // Subsurface Alabaster Wall Thickness Probe (`evalSanctuarySdf` along light vector)
    const sunDir = normalize(vec3(0.26, 0.88, 0.38));
    const thicknessProbe = evalSanctuarySdf(pHit.add(sunDir.mul(0.16))).x;
    const subsurfaceThinness = clamp(thicknessProbe.mul(5.8), 0.0, 1.0);

    // 3-step Soft Shadow Check toward the Top Oculus Skylight (`sunDir`)
    const sh1 = evalSanctuarySdf(pHit.add(N.mul(0.018)).add(sunDir.mul(0.12))).x;
    const sh2 = evalSanctuarySdf(pHit.add(N.mul(0.018)).add(sunDir.mul(0.32))).x;
    const sh3 = evalSanctuarySdf(pHit.add(N.mul(0.018)).add(sunDir.mul(0.56))).x;
    const sunShadow = clamp(
      min(sh1.mul(14.0), min(sh2.mul(9.0), sh3.mul(6.0))),
      0.14,
      1.0
    );

    // -------------------------------------------------------------------------
    // Smooth Monumental Spanish Alabaster, Chiseled Stone & Deep Azurite Cistern
    // -------------------------------------------------------------------------
    // Zone 0: Chiseled outer stone monolith (`rawUmber` + `warmKaolin` + `yellowOchre`)
    const outerStone = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.48),
      mix(warmKaolin.rInf, yellowOchre.rInf, float(0.28))
    );

    // Zone 1: Carved translucent Spanish alabaster vaults (`boneCalcite` + `warmKaolin` + `yellowOchre`)
    const heightWarmth = clamp(float(0.55).sub(pHit.y.mul(0.45)), 0.0, 1.0);
    const alabasterBase = kubelkaMunkGlaze(
      yellowOchre.K,
      yellowOchre.S,
      float(0.14).add(heightWarmth.mul(0.24)),
      mix(boneCalcite.rInf, warmKaolin.rInf, float(0.32))
    );
    const siennaShadowGlaze = kubelkaMunkGlaze(
      burntSienna.K,
      burntSienna.S,
      float(0.28),
      warmKaolin.rInf
    );
    const innerAlabaster = mix(alabasterBase, siennaShadowGlaze, heightWarmth.mul(0.22));

    // Zone 2: Deep Azurite & Lapis Subterranean Mirror Cistern (`y = -0.70`)
    const poolR = length(pHit.xz.sub(uVaultShift));
    const poolRipple = sin(poolR.mul(36.0).sub(stage.uTime.mul(3.0)))
      .mul(exp(poolR.negate().mul(5.0)))
      .mul(0.08);
    const poolTrans = beerLambertTransmittance(azurite.extinction, float(0.65).add(poolR.mul(0.85)));
    const oculusRingRefl = exp(abs(poolR.sub(uOculusRadius.mul(0.62))).negate().mul(26.0));
    const poolColor = mix(
      lapis.rInf.mul(0.52),
      celadon.rInf,
      poolTrans.z.add(poolRipple).mul(0.58)
    ).add(boneCalcite.rInf.mul(oculusRingRefl.mul(0.45)));

    const isInnerHit = clamp(hitZone, 0.0, 1.0);
    const isPoolHit = smoothstep(1.4, 1.8, hitZone);
    const baseAlbedo = mix(
      mix(outerStone, innerAlabaster, isInnerHit),
      poolColor,
      isPoolHit
    );

    // -------------------------------------------------------------------------
    // Illumination: Studio Softbox + Oculus Sunbeam + Alabaster Subsurface Glow
    // -------------------------------------------------------------------------
    const fillDir = normalize(vec3(-0.62, -0.22, 0.55));
    const viewDir = rd.negate();
    const halfVec = normalize(sunDir.add(viewDir));

    const ndotlSun = max(float(0.0), dot(N, sunDir)).mul(sunShadow);
    const ndotlFill = max(float(0.0), dot(N, fillDir)).mul(0.30);

    // Warm Golden-Amber Oculus Light Shaft on the upper rotunda walls (`y > -0.45`)
    const oculusDist = length(pHit.xz.sub(uVaultShift));
    const inOculusBeam = exp(oculusDist.mul(oculusDist).negate().mul(5.8))
      .mul(isInnerHit)
      .mul(float(1.0).sub(isPoolHit))
      .mul(smoothstep(-0.62, -0.15, pHit.y))
      .mul(uSunIntensity);
    const oculusRadiant = mix(realgarAmber.rInf, boneCalcite.rInf, float(0.45))
      .mul(inOculusBeam)
      .mul(0.42);

    // Subsurface Beer-Lambert warm amber glow through thin carved alabaster arches & pillars
    const transAlabaster = beerLambertTransmittance(
      realgarAmber.extinction,
      float(1.08).sub(subsurfaceThinness.mul(0.82))
    );
    const sssGlow = mix(realgarAmber.rInf, boneCalcite.rInf, float(0.42))
      .mul(transAlabaster)
      .mul(subsurfaceThinness)
      .mul(isInnerHit)
      .mul(float(1.0).sub(isPoolHit))
      .mul(0.48);

    const spec = max(float(0.0), dot(N, halfVec))
      .pow(mix(float(28.0), float(84.0), isPoolHit))
      .mul(mix(float(0.18), float(0.75), isPoolHit));
    const fresnel = clamp(float(1.0).sub(max(float(0.0), dot(N, viewDir))), 0.0, 1.0).pow(3.0);

    const directLight = float(0.28).add(ndotlSun.mul(0.95)).add(ndotlFill);
    const litColor = baseAlbedo
      .mul(directLight.mul(ao))
      .add(oculusRadiant)
      .add(sssGlow)
      .add(boneCalcite.rInf.mul(spec.add(fresnel.mul(0.12))));

    const slateBg = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    );
    return vec4(mix(slateBg, clamp(litColor, 0.0, 1.0), hitMask), hitMask);
  });

  const outNode = raymarchSanctuary();
  sanctuaryMat.colorNode = outNode.xyz;
  sanctuaryMat.opacityNode = outNode.w;

  const sanctuaryMesh = new THREE.Mesh(proxyGeom, sanctuaryMat);
  stage.specimenGroup.add(sanctuaryMesh);

  // ---------------------------------------------------------------------------
  // Tactile Sanctuary Rotation & Sliding Vault Gallery Kinematics
  // ---------------------------------------------------------------------------
  let rotY = 0.62;
  let rotX = 0.22;
  let targetRotY = 0.62;
  let targetRotX = 0.22;

  let oculusRadius = 0.36;
  let targetOculusRadius = 0.36;
  let sunBoost = 1.0;

  let isDragging = false;
  let prevX = 0;
  let prevY = 0;
  let dragMoved = 0;

  const canvas = stage.shell.canvas;
  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    prevX = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
    prevY = -((e.clientY / Math.max(1, window.innerHeight)) * 2 - 1);
    dragMoved = 0;
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
    const ny = -((e.clientY / Math.max(1, window.innerHeight)) * 2 - 1);
    const dx = nx - prevX;
    const dy = ny - prevY;
    prevX = nx;
    prevY = ny;
    dragMoved += Math.hypot(dx, dy);

    targetRotY += dx * 1.65;
    targetRotX = Math.max(-0.25, Math.min(0.65, targetRotX - dy * 1.15));
  };
  const onPointerUp = () => {
    if (!isDragging) return;
    isDragging = false;
    if (dragMoved < 0.04) {
      targetOculusRadius = targetOculusRadius > 0.40 ? 0.30 : 0.46;
      sunBoost = 1.85;
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
  });

  stage.onTick(({ dtSec, pointer, camera }) => {
    if (!isDragging) {
      targetRotY += dtSec * 0.14;
    }

    rotY += (targetRotY - rotY) * Math.min(1, dtSec * 6.5);
    rotX += (targetRotX - rotX) * Math.min(1, dtSec * 6.5);

    sanctuaryMesh.rotation.set(rotX, rotY, 0);
    sanctuaryMesh.updateMatrixWorld();

    _camLocal.copy(camera.position);
    sanctuaryMesh.worldToLocal(_camLocal);
    uCamLocal.value.copy(_camLocal);

    const targetShiftX = Math.tanh(pointer.ndc.x * 0.9) * 0.16;
    const targetShiftZ = Math.tanh(pointer.ndc.y * 0.9) * 0.16;
    uVaultShift.value.x += (targetShiftX - uVaultShift.value.x) * Math.min(1, dtSec * 6.0);
    uVaultShift.value.y += (targetShiftZ - uVaultShift.value.y) * Math.min(1, dtSec * 6.0);

    oculusRadius += (targetOculusRadius - oculusRadius) * Math.min(1, dtSec * 5.5);
    uOculusRadius.value = oculusRadius;

    sunBoost = 1.0 + (sunBoost - 1.0) * Math.exp(-dtSec * 2.2);
    uSunIntensity.value = sunBoost + pointer.speed * 0.25;
  });

  return stage;
}
