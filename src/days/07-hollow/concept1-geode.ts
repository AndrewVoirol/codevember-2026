import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  abs,
  atan,
  clamp,
  cos,
  dot,
  exp,
  float,
  floor,
  length,
  log,
  max,
  min,
  mix,
  normalize,
  positionLocal,
  sin,
  smoothstep,
  sqrt,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import { CHALKY_SLATE_LINEAR } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Day 07 — Hollow (Concept 1 / 7A — Pure Shader):
 * "Raymarched Sagittal-Cleaved Nautilus Phragmocone (Equiangular Buoyancy Chambers & Nacre Septa)"
 *
 * A single-pass logarithmic-spiral 3D Signed Distance Field raymarcher in TSL
 * of a diamond-saw sagittal-sectioned *Nautilus pompilius* shell:
 * - Exact Branch-Cut-Free Polar Spiral Coordinates:
 *   `wRaw = ln(rXY / R0) / (2π * β) - θ / (2π)`
 *   Because `θ / (2π)` jumps by `-1.0` across `θ = ±π` while `wRaw` jumps by
 *   `+1.0`, `uWhorl = wRaw - floor(wRaw)` and `φ = θ + 2π * floor(wRaw)` are
 *   100% continuous across `θ = ±π` with zero branch-cut seam.
 * - Direct CSG Interior Solid Formulation:
 *   `dInteriorSolid = min(0.0 - dTunnel, smin(dSepta, dSiph, kFillet))`
 *   carves 28 deep 3D hollow buoyancy chambers (`camerae`) separated by concave
 *   3D mother-of-pearl septa walls and pierced by a continuous central calcified
 *   siphuncle tube, opening into the wide outer living chamber hood.
 */

const _camLocal = new THREE.Vector3();
const _ptrLocal = new THREE.Vector3();
const _yAxis = new THREE.Vector3(0, 1, 0);
const _targetLantern = new THREE.Vector3(0.0, 0.0, 0.05);
const _currentLantern = new THREE.Vector3(0.0, 0.0, 0.05);

export async function bootHollowConcept1(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(7);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.06, 3.45],
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
  const cinnabar = createPigmentUniforms('cinnabarVermilion');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const celadon = createPigmentUniforms('celadonGlauconite');
  const lapis = createPigmentUniforms('lapisUltramarine');
  const azurite = createPigmentUniforms('azuritePrussian');

  const uCamLocal = uniform(new THREE.Vector3(0, 0.06, 3.45));
  const uLanternLocal = uniform(new THREE.Vector3(0.0, 0.0, 0.05));
  const uSliceZ = uniform(0.0);
  const uLanternBoost = uniform(1.0);

  const proxyGeom = new THREE.SphereGeometry(1.24, 64, 64);
  const nautilusMat = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: true,
  });

  // Smooth polynomial minimum (`smin`) for organic septal-neck fillets
  const smin = Fn(([a_in, b_in, k_in]: [TslNode, TslNode, TslNode]) => {
    const a = float(a_in);
    const b = float(b_in);
    const k = float(k_in);
    const h = clamp(float(0.5).add(float(0.5).mul(b.sub(a)).div(k)), 0.0, 1.0);
    return mix(b, a, h).sub(k.mul(h).mul(float(1.0).sub(h)));
  });

  // ---------------------------------------------------------------------------
  // Single-Pass Branch-Cut-Free Sagittal-Cleaved Nautilus Phragmocone SDF
  // Returns vec4(dTotal, materialZone, spiralPhi, septaAndSiphDist)
  // ---------------------------------------------------------------------------
  const evalNautilusSdf = Fn(([p_in]: [TslNode]) => {
    const p = vec3(p_in).sub(vec3(0.06, -0.05, 0.0));
    const rXY = sqrt(p.x.mul(p.x).add(p.y.mul(p.y)).add(1e-6));
    const theta = atan(p.y, p.x);

    // Exact logarithmic spiral polar coordinates (`R0 = 0.045`, `β = 0.176`, `2πβ = 1.10584`):
    // `wRaw` jumps by `+1.0` when `theta` wraps from `+π` to `-π`, so `uWhorl = fract(wRaw)`
    // and `phi = theta + 2π * floor(wRaw)` have ZERO branch-cut discontinuity at `θ = ±π`!
    const wRaw = log(max(float(1e-4), rXY.div(0.045))).div(1.1058407).sub(theta.div(6.2831853));
    const kTurn = floor(wRaw);
    const uWhorl = wRaw.sub(kTurn); // in [0, 1) across each spiral whorl
    const phi = theta.add(kTurn.mul(6.2831853));

    // Centered radial coordinate across the whorl (`uC in [-0.5, +0.5]`) and axial coordinate `uZ`
    const scaleZ = max(float(0.018), rXY.mul(0.42));
    const uC = uWhorl.sub(0.5);
    const uZ = p.z.div(scaleZ);

    // 1. Outer Shell Envelope (`phi in [0.4, 15.2]` radians = ~2.35 full revolutions):
    // For points where the outermost turn has terminated past the aperture (`phi > 15.2`),
    // seamlessly fall back to the rounded dorsal wall of the preceding inner whorl (`uC + 1.0`)!
    const phiMin = float(0.40);
    const phiMax = float(15.20);
    const dCurrentTurnTube = sqrt(uC.mul(uC).add(uZ.mul(uZ)).add(1e-6)).sub(0.52).mul(scaleZ);
    // Signed angular bounds MUST be negative inside `[phiMin, phiMax]` so `max(dTube, dPhiBounds)` stays `< 0` inside!
    const dPhiBounds = max(phi.sub(phiMax), phiMin.sub(phi)).mul(rXY).mul(0.42);
    const dCurrentSolid = max(dCurrentTurnTube, dPhiBounds);

    // Preceding inner whorl dorsal wall (`uC + 1.0`) exposed beneath the aperture hood (`phi > phiMax`)
    const uCInner = uC.add(1.0);
    const dInnerTurnDorsal = max(
      sqrt(uCInner.mul(uCInner).add(uZ.mul(uZ)).add(1e-6)).sub(0.51).mul(scaleZ),
      phi.sub(phiMax.add(6.2831853)).mul(rXY).mul(0.42)
    );
    const dOuterShell = min(dCurrentSolid, dInnerTurnDorsal);

    // 2. Hollow Spiral Tunnel inside each whorl (`dTunnel < 0` inside the chamber cavity):
    // Preserves a solid spiral wall (`|uC| in [0.40, 0.50]`) between adjacent whorls and a
    // solid concave back wall (`sqrt(uC^2 + uZ^2) in [0.42, 0.52]`), while opening cleanly
    // out through the aperture mouth (`phiMax + 0.12`)!
    const dTunnelCross = max(
      sqrt(uC.mul(uC).add(uZ.mul(uZ)).add(1e-6)).sub(0.42),
      abs(uC).sub(0.40)
    ).mul(scaleZ);
    const dTunnelEnd = max(
      phi.sub(phiMax.add(0.12)),
      phiMin.add(0.15).sub(phi)
    ).mul(rXY).mul(0.42);
    const dTunnel = max(dTunnelCross, dTunnelEnd);

    // 3. 28 Concave 3D Mother-of-Pearl Septa Walls & Central Calcified Siphuncle Tube:
    // Septa curve backward toward the apex in the middle of the chamber (`uC = 0, uZ = 0.1`)
    // and sweep forward to suture against the spiral walls (`|uC| = 0.40`).
    const sutureCurve = uC.mul(uC).mul(1.85).add(uZ.sub(0.12).mul(uZ.sub(0.12)).mul(0.85));
    const phiSeptum = phi.add(sutureCurve);

    // Final septum sits at `phiFinal = 24.0 * π / 5.5 = 13.708`, leaving the outer
    // `[13.71, 15.20]` hood (~85 degrees) as the wide-open living chamber!
    const phiFinal = float(13.708);
    const phiClampedSep = min(phiSeptum, phiFinal);
    const dSeptaWave = abs(sin(phiClampedSep.mul(5.5))).div(5.5).mul(rXY).mul(0.52);
    const dPastFinalSep = max(float(0.0), phiSeptum.sub(phiFinal)).mul(rXY).mul(0.52);
    const septumHalfThick = rXY.mul(0.018).add(0.0045);
    const dSepta = dSeptaWave.add(dPastFinalSep).sub(septumHalfThick);

    // Continuous central siphuncle tube piercing every septum up to `phiFinal`
    const dSiphRad = sqrt(
      uC.add(0.04).mul(uC.add(0.04)).add(uZ.add(0.08).mul(uZ.add(0.08))).add(1e-6)
    ).sub(0.072).mul(scaleZ);
    const dSiph = max(dSiphRad, phiSeptum.sub(phiFinal.add(0.06)).mul(rXY).mul(0.52));

    const dSeptaAndSiph = smin(dSepta, dSiph, scaleZ.mul(0.08).add(0.003));

    // Direct CSG Interior Solid (`< 0` inside spiral walls, back walls, septa, or siphuncle;
    // `> 0` inside the 28 hollow buoyancy chambers!):
    const dInteriorSolid = min(float(0.0).sub(dTunnel), dSeptaAndSiph);

    // Intersect with outer shell envelope and diamond-saw sagittal cleavage plane `p.z - uSliceZ`
    const dSlicePlane = p.z.sub(uSliceZ);
    const dTotal = max(dOuterShell, max(dInteriorSolid, dSlicePlane));

    // Zone classification (strict `low < high` in all `smoothstep` calls!):
    // - Zone 1 (Cleaved Nacre Rim): right on the sagittal cut `|p.z - uSliceZ| < 0.0045`
    // - Zone 2 (Hollow 3D Chambers, Concave Septa & Siphuncle): recessed behind `uSliceZ` inside `dOuterShell < -0.005`
    // - Zone 0 (Exterior Flame-Striped Dorsal Shell): outer surface `dOuterShell > -0.005` behind the cut
    const isInnerChamber = float(1.0).sub(smoothstep(-0.010, -0.003, dOuterShell));
    const isCleavageRim = float(1.0).sub(smoothstep(0.0008, 0.0045, abs(dSlicePlane)));
    const zone = mix(mix(float(0.0), float(2.0), isInnerChamber), float(1.0), isCleavageRim);

    return vec4(dTotal, zone, phi, dSeptaAndSiph);
  });

  // ---------------------------------------------------------------------------
  // Zero-Zeno Slab-Bounded Per-Pixel Sphere Tracer
  // ---------------------------------------------------------------------------
  const raymarchNautilus = Fn(() => {
    const ro = uCamLocal;
    const rd = normalize(positionLocal.sub(uCamLocal));

    // 1. Ray-sphere bounding intersection (`R_bound = 1.14`)
    const b = dot(ro, rd);
    const c = dot(ro, ro).sub(1.14 * 1.14);
    const disc = max(float(0.0), b.mul(b).sub(c));
    const sqrtDisc = sqrt(disc);
    const tSphereNear = max(float(0.05), b.negate().sub(sqrtDisc));
    const tSphereFar = b.negate().add(sqrtDisc);

    // 2. Exact Z-Slab bounding (`z in [-0.42, uSliceZ + 0.004]`)
    const safeRdZ = mix(
      rd.z,
      float(1e-5),
      float(1.0).sub(smoothstep(0.0, 1e-5, abs(rd.z)))
    );
    const invRdZ = float(1.0).div(safeRdZ);
    const tz0 = float(-0.42).sub(ro.z).mul(invRdZ);
    const tz1 = uSliceZ.add(0.004).sub(ro.z).mul(invRdZ);
    const tSlabNear = min(tz0, tz1);
    const tSlabFar = max(tz0, tz1);

    const tNear = max(tSphereNear, tSlabNear);
    const tFar = min(tSphereFar, tSlabFar);

    const t = tNear.toVar();

    // 40-step unbraked sphere-trace across the narrow Z-slab
    Loop(40, () => {
      const pos = ro.add(rd.mul(t));
      const d = evalNautilusSdf(pos).x;
      const inBounds = smoothstep(0.0, 0.002, tFar.add(0.015).sub(t));
      t.addAssign(clamp(d.mul(0.88), 0.0, 0.30).mul(inBounds));
    });

    const pHit = ro.add(rd.mul(t));
    const sdfInfo = evalNautilusSdf(pHit);
    const dFinal = sdfInfo.x;
    const hitZone = sdfInfo.y;
    const spiralPhi = sdfInfo.z;
    const septaDist = sdfInfo.w;

    const hitMask = float(1.0)
      .sub(smoothstep(0.002, 0.006, dFinal))
      .mul(smoothstep(-0.004, 0.004, tFar.sub(t)));

    // 4-tap Tetrahedron SDF Normal
    const eps = float(0.0014);
    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);
    const N = normalize(
      k1.mul(evalNautilusSdf(pHit.add(k1.mul(eps))).x)
        .add(k2.mul(evalNautilusSdf(pHit.add(k2.mul(eps))).x))
        .add(k3.mul(evalNautilusSdf(pHit.add(k3.mul(eps))).x))
        .add(k4.mul(evalNautilusSdf(pHit.add(k4.mul(eps))).x))
    );

    const isOuter = float(1.0).sub(smoothstep(0.1, 0.5, hitZone));
    const isRim = smoothstep(0.3, 0.85, hitZone).mul(
      float(1.0).sub(smoothstep(1.15, 1.7, hitZone))
    );
    const isChamber = smoothstep(1.3, 1.85, hitZone);

    const pCentered = pHit.sub(vec3(0.06, -0.05, 0.0));
    const rXY = length(pCentered.xy);
    const localAz = max(float(0.022), rXY.mul(0.36));
    const pocketDepth = clamp(uSliceZ.sub(pHit.z).div(localAz), 0.0, 1.0);

    // Distinguish 3D curved septa & siphuncle (`|septaDist| < 0.008`) from the deep chamber back wall!
    const onSeptum3D = float(1.0).sub(smoothstep(0.001, 0.010, abs(septaDist))).mul(isChamber);

    // 1-tap SDF Ambient Occlusion + Strong Deep Chamber Cavity Shadowing
    const ao1 = evalNautilusSdf(pHit.add(N.mul(0.038))).x;
    const macroAo = clamp(float(0.38).add(ao1.mul(12.0)), 0.20, 1.0);
    const chamberOcclusion = mix(
      float(1.0).sub(pocketDepth.mul(0.76)),
      float(0.84).sub(pocketDepth.mul(0.28)),
      onSeptum3D
    );
    const ao = macroAo.mul(mix(float(1.0), chamberOcclusion, isChamber));

    // -------------------------------------------------------------------------
    // Material 0: Exterior Nautilus Shell (`boneCalcite` + `burntSienna`/`cinnabar` flame stripes)
    // -------------------------------------------------------------------------
    const flamePhase = spiralPhi.mul(16.0).sub(rXY.mul(14.0));
    const flameStripe = smoothstep(0.10, 0.60, sin(flamePhase)).mul(
      float(1.0).sub(smoothstep(13.2, 14.8, spiralPhi))
    );
    const shellPorcelain = mix(boneCalcite.rInf, warmKaolin.rInf, float(0.28));
    const flamePigment = kubelkaMunkGlaze(
      burntSienna.K,
      burntSienna.S,
      float(0.58).add(flameStripe.mul(0.30)),
      cinnabar.rInf
    );
    const outerShellAlbedo = mix(shellPorcelain, flamePigment, flameStripe.mul(0.78));

    // -------------------------------------------------------------------------
    // Material 1 & 2: Cleaved Nacre Rim, 3D Curved Septa & Deep Hollow Chamber Cavities
    // -------------------------------------------------------------------------
    const viewDir = rd.negate();
    const ndotv = max(float(0.0), dot(N, viewDir));
    const nacrePhase = float(1.0).sub(ndotv).mul(9.0).add(spiralPhi.mul(1.8)).add(pocketDepth.mul(6.0));
    const nacreRosy = mix(cinnabar.rInf, realgarAmber.rInf, float(0.62));
    const nacreIridescence = mix(
      mix(celadon.rInf, nacreRosy, sin(nacrePhase).mul(0.5).add(0.5)),
      lapis.rInf,
      cos(nacrePhase.mul(0.75)).mul(0.25).add(0.25)
    );

    // Tier 1: Diamond-saw sagittal cleavage rim (`isRim`) gleams in bright `boneCalcite` ivory-pearl
    const cleavageRimAlbedo = mix(boneCalcite.rInf, warmKaolin.rInf, float(0.14));

    // Tier 2: 3D Curved Concave Septa & Siphuncle (`onSeptum3D`) glow with warm golden-amber & celadon nacre
    const septum3DAlbedo = mix(
      kubelkaMunkGlaze(yellowOchre.K, yellowOchre.S, float(0.36), boneCalcite.rInf),
      nacreIridescence,
      float(0.48)
    );

    // Tier 3: Deep Recessed Chamber Back Wall (`pocketDepth -> 1`) is dark `rawUmber` & `azuritePrussian`
    const deepCavityAlbedo = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.68).add(pocketDepth.mul(0.35)),
      mix(azurite.rInf, celadon.rInf, float(0.38))
    );

    const interiorAlbedo = mix(
      mix(septum3DAlbedo, deepCavityAlbedo, smoothstep(0.12, 0.78, pocketDepth)),
      septum3DAlbedo,
      onSeptum3D.mul(0.75)
    );

    const baseAlbedo = mix(
      mix(outerShellAlbedo, interiorAlbedo, isChamber),
      cleavageRimAlbedo,
      isRim
    );

    // -------------------------------------------------------------------------
    // Studio Softbox + Subtle Pointer Lantern
    // -------------------------------------------------------------------------
    const keyDir = normalize(vec3(0.48, 0.66, 0.58));
    const fillDir = normalize(vec3(-0.62, -0.25, 0.52));
    const halfKey = normalize(keyDir.add(viewDir));

    const ndotlKey = max(float(0.0), dot(N, keyDir));
    const ndotlFill = max(float(0.0), dot(N, fillDir)).mul(0.26);

    const toLantern = uLanternLocal.sub(pHit);
    const lanternDistSq = dot(toLantern, toLantern);
    const lanternDir = normalize(toLantern);
    const lanternAtten = exp(lanternDistSq.negate().mul(4.5)).mul(uLanternBoost);
    const lanternNdotL = max(float(0.0), dot(N, lanternDir));

    const lanternGlow = mix(realgarAmber.rInf, cinnabar.rInf, float(0.38))
      .mul(lanternNdotL)
      .mul(lanternAtten)
      .mul(onSeptum3D.mul(0.32).add(isChamber.mul(0.14)));

    const specPower = mix(float(28.0), float(72.0), max(isRim, onSeptum3D));
    const specStrength = mix(float(0.12), float(0.38), max(isRim, onSeptum3D));
    const spec = max(float(0.0), dot(N, halfKey)).pow(specPower).mul(specStrength);
    const fresnel = clamp(float(1.0).sub(ndotv), 0.0, 1.0).pow(3.0);

    const directLight = float(0.22).add(ndotlKey.mul(0.78)).add(ndotlFill).add(isRim.mul(0.14));
    const litColor = baseAlbedo
      .mul(directLight.mul(ao))
      .add(lanternGlow)
      .add(
        mix(boneCalcite.rInf, nacreIridescence, float(0.35)).mul(
          spec.add(fresnel.mul(0.12).mul(float(1.0).sub(isOuter.mul(0.5))))
        )
      );

    const slateBg = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    );
    return vec4(mix(slateBg, clamp(litColor, 0.0, 1.0), hitMask), hitMask);
  });

  const outNode = raymarchNautilus();
  nautilusMat.colorNode = outNode.xyz;
  nautilusMat.opacityNode = outNode.w;

  const nautilusMesh = new THREE.Mesh(proxyGeom, nautilusMat);
  stage.specimenGroup.add(nautilusMesh);

  // ---------------------------------------------------------------------------
  // Tactile Rotation, Sagittal Cleavage Depth & Spiral Chamber Lantern Kinematics
  // ---------------------------------------------------------------------------
  let rotY = -0.18;
  let rotX = 0.08;
  let targetRotY = -0.18;
  let targetRotX = 0.08;

  let sliceZ = 0.0;
  let targetSliceZ = 0.0;
  let lanternBoost = 1.0;

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

    targetRotY = Math.max(-0.95, Math.min(0.95, targetRotY + dx * 1.45));
    targetRotX = Math.max(-0.55, Math.min(0.55, targetRotX - dy * 1.15));
  };
  const onPointerUp = () => {
    if (!isDragging) return;
    isDragging = false;
    if (dragMoved < 0.04) {
      targetSliceZ = targetSliceZ > -0.04 ? -0.07 : 0.0;
      lanternBoost = 1.8;
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

  stage.onTick(({ elapsedSec, dtSec, pointer, camera }) => {
    if (!isDragging) {
      targetRotY = Math.sin(elapsedSec * 0.42) * 0.26 + pointer.ndc.x * 0.22;
      targetRotX = 0.08 - pointer.ndc.y * 0.14;
    }

    rotY += (targetRotY - rotY) * Math.min(1, dtSec * 6.5);
    rotX += (targetRotX - rotX) * Math.min(1, dtSec * 6.5);

    nautilusMesh.rotation.set(rotX, rotY, 0);
    nautilusMesh.updateMatrixWorld();

    _camLocal.copy(camera.position);
    nautilusMesh.worldToLocal(_camLocal);
    uCamLocal.value.copy(_camLocal);

    _ptrLocal.set(
      Math.tanh(pointer.worldPos.x * 0.85) * 0.55,
      Math.tanh(pointer.worldPos.y * 0.85) * 0.55,
      -0.04
    );
    _ptrLocal.applyAxisAngle(_yAxis, -rotY);
    _targetLantern.copy(_ptrLocal);
    _currentLantern.lerp(_targetLantern, Math.min(1, dtSec * 8.0));
    uLanternLocal.value.copy(_currentLantern);

    sliceZ += (targetSliceZ - sliceZ) * Math.min(1, dtSec * 5.5);
    uSliceZ.value = sliceZ;

    lanternBoost = 1.0 + (lanternBoost - 1.0) * Math.exp(-dtSec * 2.4);
    uLanternBoost.value = lanternBoost + pointer.speed * 0.25;
  });

  return stage;
}
