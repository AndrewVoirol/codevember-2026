import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  abs,
  clamp,
  cos,
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
  sqrt,
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
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Day 07 — Hollow:
 * "Analytical 7-Shell Canton Ivory Puzzle Ball (Guigongqiu)"
 *
 * A sub-pixel-exact analytical multi-shell ray-sphere & conical-counterbore CSG
 * shader in TSL rendering 6 concentric, physically separated hollow spherical shells
 * (`R = 1.120, 0.965, 0.810, 0.655, 0.500, 0.345`, wall thickness `0.044..0.032`)
 * plus an innermost carved Cinnabar & Realgar filigree lantern core (`R = 0.215`):
 *
 * 1. Pure Circular Rose-Window & Pierced Eyelet Lattice (Zero Boxy Mitred Corners):
 *    - 6 primary circular rose-windows on the cubic axes (`±X, ±Y, ±Z`)
 *    - 8 secondary circular windows on the octahedral axes (`(±1, ±1, ±1)/√3`)
 *    - 12 tertiary circular windows on the dodecahedral edge axes (`(±1, ±1, 0)/√2`)
 *    - 48 pierced circular Canton-lace eyelets arranged in 8-fold rings around the
 *      primary rose-windows using true Euclidean tangent-plane circles.
 *    - Lathe-turned concentric molding rings strictly follow the primary/secondary/tertiary
 *      circular apertures (`moldingDist`), eliminating rectangular `max()` picture frames.
 * 2. Radial Counterbore Taper + Half-Round Torus Rim Fillet:
 *    - Outer shell faces flare slightly wider (`+0.015` counterbore taper) than inner faces,
 *      revealing 3D carved ivory wall thickness and continuous half-round rim highlights
 *      even on windows facing straight toward the camera.
 * 3. Full Tangent-Limb Grazing Slab Resolution:
 *    - Rays grazing the outer horizon (`rInSq < hSq <= rOutSq`) trace all the way to the
 *      far outer-sphere exit (`-b + sqrt(rOutSq - hSq)`) and chord midpoint (`-b`),
 *      preventing paper-thin or clipped silhouette rims.
 * 4. Studio-Locked World Lighting + Analytical Outer-Shell Key-Light Cast Shadows:
 *    - Studio softbox key and fill directions transform into mesh-local space each frame
 *      and trace an exact analytical shadow ray through the enclosing outer shell (`k - 1`).
 */

const _camLocal = new THREE.Vector3();
const _invQuat = new THREE.Quaternion();
const _keyLocal = new THREE.Vector3();
const _fillLocal = new THREE.Vector3();

const WORLD_KEY_DIR = new THREE.Vector3(0.52, 0.68, 0.56).normalize();
const WORLD_FILL_DIR = new THREE.Vector3(-0.62, -0.28, 0.48).normalize();

export async function bootHollowConcept2(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(7);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.03, 4.35],
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

  const uCamLocal = uniform(new THREE.Vector3(0, 0.03, 4.35));
  const uKeyDirLocal = uniform(WORLD_KEY_DIR.clone());
  const uFillDirLocal = uniform(WORLD_FILL_DIR.clone());
  const uShellRotA = uniform(new THREE.Vector3(0.18, -0.28, 0.36));
  const uShellRotB = uniform(new THREE.Vector3(-0.42, 0.54, -0.64));
  const uHoverParallax = uniform(new THREE.Vector2(0.0, 0.0));
  const uFreeFactor = uniform(1.0);
  const uCoreGlow = uniform(1.0);

  const proxyGeom = new THREE.SphereGeometry(1.22, 64, 64);
  const puzzleMat = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: true,
  });

  // 2D rotation helper in TSL
  const rot2D = Fn(([v_in, ang_in]: [TslNode, TslNode]) => {
    const v = vec2(v_in);
    const c = cos(ang_in);
    const s = sin(ang_in);
    return vec2(v.x.mul(c).sub(v.y.mul(s)), v.x.mul(s).add(v.y.mul(c)));
  });

  // ---------------------------------------------------------------------------
  // Rotate local point `p` into the independent 3D frame of Shell `shellIdx` [0..6]
  // ---------------------------------------------------------------------------
  const rotateToShellFrame = Fn(([p_in, shellIdx_in]: [TslNode, TslNode]) => {
    const p = vec3(p_in);
    const idx = float(shellIdx_in);

    // Alternating harmonic direction sign (+1, -1, +1, -1...) so adjacent shells counter-precess
    const altSign = cos(idx.mul(3.14159265));
    const isInner = smoothstep(2.4, 2.6, idx);

    // Curated rest offset (`idx * 0.21`) keeps all 6 concentric window rims visible in a stepped
    // amphitheater cascade at rest while still separating each shell's lattice
    const hoverYaw = uHoverParallax.x.mul(idx.mul(0.11).add(0.06)).mul(altSign);
    const hoverPitch = uHoverParallax.y.mul(idx.mul(0.09).add(0.05)).mul(altSign);

    const angY = mix(
      mix(uShellRotA.x, uShellRotA.y, clamp(idx, 0.0, 1.0)),
      mix(uShellRotB.x, uShellRotB.y, clamp(idx.sub(3.0), 0.0, 1.0)),
      isInner
    )
      .add(float(0.38).add(idx.mul(0.22).mul(altSign)).mul(uFreeFactor))
      .add(hoverYaw);

    const angX = mix(uShellRotA.z, uShellRotB.z, isInner)
      .mul(altSign)
      .add(float(0.56).add(idx.mul(0.18).mul(altSign)).mul(uFreeFactor))
      .add(hoverPitch);

    const xz = rot2D(vec2(p.x, p.z), angY);
    const p1 = vec3(xz.x, p.y, xz.y);
    const yz = rot2D(vec2(p1.y, p1.z), angX);
    return vec3(p1.x, yz.x, yz.y);
  });

  // ---------------------------------------------------------------------------
  // Exact Spherical Chord Distance to the Circular Rose-Windows & Pierced Eyelets
  // Returns `vec2(sCut, sMolding)`:
  //   - `sCut`: signed distance to all circular openings (negative = inside hole, positive = solid ivory)
  //   - `sMolding`: signed distance to the 26 primary/secondary/tertiary circular apertures
  //                 (drives smooth concentric lathe-turned molding rings with zero boxy corners)
  // ---------------------------------------------------------------------------
  const evalShellApertureField = Fn(
    ([p_in, shellIdx_in, flare_in]: [TslNode, TslNode, TslNode]) => {
      const idx = float(shellIdx_in);
      const flare = float(flare_in);
      const q = rotateToShellFrame(p_in, idx);
      const dir = normalize(q);
      const aDir = abs(dir);

      // Sort components of |dir|: aMax >= aMid >= aMin
      const aMax = max(max(aDir.x, aDir.y), aDir.z);
      const aMin = min(min(aDir.x, aDir.y), aDir.z);
      const aMid = aDir.x.add(aDir.y).add(aDir.z).sub(aMax).sub(aMin);

      // 1. 6 Primary Circular Rose-Windows on cubic axes `(±1,0,0), (0,±1,0), (0,0,±1)`
      const chordCubic = sqrt(max(float(1e-6), float(2.0).mul(float(1.0).sub(aMax))));
      // Balanced primary window radius (`0.372` outer -> `0.312` inner) + radial counterbore `flare`
      const radCubic = float(0.372).sub(idx.mul(0.012)).add(flare);
      const dCubic = chordCubic.sub(radCubic);

      // 2. 48 Pierced Circular Canton-Lace Eyelets (8 true circular drill holes around each cubic window)
      // Project into the local tangent plane of the closest cubic axis using sorted octant coords `(uOct, vOct)`
      const perpLen = max(float(1e-5), sqrt(aMid.mul(aMid).add(aMin.mul(aMin))));
      const scaleChord = chordCubic.div(perpLen);
      const uOct = aMid.mul(scaleChord);
      const vOct = aMin.mul(scaleChord);
      // Center of the eyelet at octant bisector angle pi/8 (`cos(pi/8)=0.92388, sin(pi/8)=0.38268`)
      const eyeRingRad = radCubic.add(0.094).sub(flare.mul(0.5));
      const duEye = uOct.sub(eyeRingRad.mul(0.9238795));
      const dvEye = vOct.sub(eyeRingRad.mul(0.3826834));
      const radEye = float(0.026).sub(idx.mul(0.0014)).add(flare.mul(0.35));
      const dEyelets = sqrt(duEye.mul(duEye).add(dvEye.mul(dvEye)).add(1e-7)).sub(radEye);

      // 3. 8 Secondary Circular Windows on octahedral axes `(±1,±1,±1)/sqrt(3)`
      const cosOct = aDir.x.add(aDir.y).add(aDir.z).mul(0.57735027);
      const chordOct = sqrt(max(float(1e-6), float(2.0).mul(float(1.0).sub(cosOct))));
      const radOct = float(0.252).sub(idx.mul(0.010)).add(flare.mul(0.85));
      const dOct = chordOct.sub(radOct);

      // 4. 12 Tertiary Circular Filigree Windows on dodecahedral edge axes `(±1,±1,0)/sqrt(2)`
      const cosEdge = aMax.add(aMid).mul(0.70710678);
      const chordEdge = sqrt(max(float(1e-6), float(2.0).mul(float(1.0).sub(cosEdge))));
      const radEdge = float(0.134).sub(idx.mul(0.005)).add(flare.mul(0.70));
      const dEdge = chordEdge.sub(radEdge);

      const sMolding = min(min(dCubic, dOct), dEdge);
      const sCut = min(sMolding, dEyelets);
      return vec2(sCut, sMolding);
    }
  );

  // ---------------------------------------------------------------------------
  // Analytical Multi-Shell Ray-Sphere & Counterbored Aperture Wall Shader
  // ---------------------------------------------------------------------------
  const tracePuzzleBall = Fn(() => {
    const ro = uCamLocal;
    const rd = normalize(positionLocal.sub(uCamLocal));

    const b = dot(ro, rd);
    const c0 = dot(ro, ro);
    // Squared perpendicular distance from ray to sphere center (0,0,0)
    const hSq = max(float(0.0), c0.sub(b.mul(b)));

    const hitMask = float(0.0).toVar();
    const hitT = float(100.0).toVar();
    const hitShellIdx = float(0.0).toVar();
    const hitApertureDist = float(0.1).toVar();
    const hitMoldingDist = float(0.1).toVar();
    const hitIsWallBevel = float(0.0).toVar();
    const hitIsBackHemisphere = float(0.0).toVar();
    const rimShadowAtten = float(1.0).toVar();

    // -------------------------------------------------------------------------
    // Pass 1: 6 Front Hemisphere Concentric Shells (outermost k=0 to innermost k=5)
    // -------------------------------------------------------------------------
    Loop(6, ({ i }) => {
      const k = float(i);
      const rOut = float(1.12).sub(k.mul(0.155));
      const wallThick = float(0.044).sub(k.mul(0.002));
      const rIn = rOut.sub(wallThick);

      const rOutSq = rOut.mul(rOut);
      const rInSq = rIn.mul(rIn);

      const validOuter = smoothstep(0.0, 0.0015, rOutSq.sub(hSq)).mul(float(1.0).sub(hitMask));
      const halfChordOut = sqrt(max(float(0.0), rOutSq.sub(hSq)));
      const halfChordIn = sqrt(max(float(0.0), rInSq.sub(hSq)));

      // Front entry into outer sphere
      const tOut = b.negate().sub(halfChordOut);
      // Does the ray pierce the inner cavity (`hSq < rInSq`) or graze the solid tangent limb (`hSq >= rInSq`)?
      const isGrazingLimb = smoothstep(rInSq.sub(0.0015), rInSq.add(0.0015), hSq);
      const tInPierce = b.negate().sub(halfChordIn);
      const tLimbFar = b.negate().add(halfChordOut);
      // Slab exit: inner sphere entry when piercing, or far outer-sphere exit when grazing the horizon limb
      const tSlabExit = mix(tInPierce, tLimbFar, isGrazingLimb);
      const tSlabMid = mix(tOut, tSlabExit, float(0.5));

      const pOut = ro.add(rd.mul(tOut));
      const pMid = ro.add(rd.mul(tSlabMid));
      const pExit = ro.add(rd.mul(tSlabExit));

      // Counterbore flare (`+0.015` at outer rim, `+0.006` at mid-wall, `0.0` at inner rim)
      const fOut = evalShellApertureField(pOut, k, float(0.015));
      const fMid = evalShellApertureField(pMid, k, float(0.006));
      const fExit = evalShellApertureField(pExit, k, mix(float(0.0), float(0.015), isGrazingLimb));

      const sOut = fOut.x;
      const sInnerWall = max(fMid.x, fExit.x);

      // Case A: Ray hits the solid outer spherical face (`sOut >= 0`)
      const hitOuterFace = smoothstep(-0.0015, 0.0015, sOut).mul(validOuter);

      // Case B: Ray enters through an open circular aperture (`sOut < 0`)
      //         and hits the counterbored cylindrical ivory wall (`sInnerWall >= 0`)
      const fracWall = clamp(sOut.negate().div(max(float(1e-4), sInnerWall.sub(sOut))), 0.0, 1.0);
      const tWall = mix(tOut, tSlabExit, fracWall.mul(0.85));
      const hitHoleWall = float(1.0)
        .sub(smoothstep(-0.0015, 0.0015, sOut))
        .mul(smoothstep(-0.0015, 0.0015, sInnerWall))
        .mul(validOuter);

      const shellHit = max(hitOuterFace, hitHoleWall);
      const tResolved = mix(tOut, tWall, hitHoleWall);
      const sResolved = mix(sOut, float(0.002), hitHoleWall);
      const mResolved = mix(fOut.y, fMid.y, hitHoleWall);

      hitT.assign(mix(hitT, tResolved, shellHit));
      hitShellIdx.assign(mix(hitShellIdx, k, shellHit));
      hitApertureDist.assign(mix(hitApertureDist, sResolved, shellHit));
      hitMoldingDist.assign(mix(hitMoldingDist, mResolved, shellHit));
      hitIsWallBevel.assign(mix(hitIsWallBevel, hitHoleWall, shellHit));
      hitMask.assign(max(hitMask, shellHit));

      // Soft ambient aperture-occlusion as viewing ray passes close to outer window rims
      const rimProx = smoothstep(-0.065, 0.0, max(sOut, sInnerWall))
        .mul(validOuter)
        .mul(float(1.0).sub(shellHit));
      rimShadowAtten.mulAssign(float(1.0).sub(rimProx.mul(0.18)));
    });

    // -------------------------------------------------------------------------
    // Pass 2: Central Carved Cinnabar & Realgar Filigree Lantern Core (`R_core = 0.215`)
    // -------------------------------------------------------------------------
    const rCore = float(0.215);
    const rCoreSq = rCore.mul(rCore);
    const validCore = smoothstep(0.0, 0.001, rCoreSq.sub(hSq)).mul(float(1.0).sub(hitMask));
    const tCore = b.negate().sub(sqrt(max(float(0.0), rCoreSq.sub(hSq))));
    const pCore = ro.add(rd.mul(tCore));
    const fCore = evalShellApertureField(pCore, float(6.0), float(0.0));
    const sCore = fCore.x.add(0.085);
    const hitCore = smoothstep(-0.002, 0.002, sCore).mul(validCore);

    hitT.assign(mix(hitT, tCore, hitCore));
    hitShellIdx.assign(mix(hitShellIdx, float(6.0), hitCore));
    hitApertureDist.assign(mix(hitApertureDist, sCore, hitCore));
    hitMoldingDist.assign(mix(hitMoldingDist, fCore.y.add(0.085), hitCore));
    hitIsWallBevel.assign(mix(hitIsWallBevel, float(0.0), hitCore));
    hitMask.assign(max(hitMask, hitCore));

    // -------------------------------------------------------------------------
    // Pass 3: 6 Back Hemisphere Concentric Shells (innermost k=5 to outermost k=0)
    // Seen through the open circular rose-windows of the front hemisphere
    // -------------------------------------------------------------------------
    Loop(6, ({ i }) => {
      const k = float(5).sub(float(i));
      const rOut = float(1.12).sub(k.mul(0.155));
      const wallThick = float(0.044).sub(k.mul(0.002));
      const rIn = rOut.sub(wallThick);

      const rOutSq = rOut.mul(rOut);
      const rInSq = rIn.mul(rIn);

      const validBack = smoothstep(0.0, 0.0015, rInSq.sub(hSq)).mul(float(1.0).sub(hitMask));
      const tInBack = b.negate().add(sqrt(max(float(0.0), rInSq.sub(hSq))));
      const tOutBack = b.negate().add(sqrt(max(float(0.0), rOutSq.sub(hSq))));

      const pInBack = ro.add(rd.mul(tInBack));
      const pOutBack = ro.add(rd.mul(tOutBack));

      const fInBack = evalShellApertureField(pInBack, k, float(0.0));
      const fOutBack = evalShellApertureField(pOutBack, k, float(0.015));
      const sInBack = fInBack.x;
      const sOutBack = fOutBack.x;

      const hitInnerConcave = smoothstep(-0.0015, 0.0015, sInBack).mul(validBack);
      const fracBackWall = clamp(
        sInBack.negate().div(max(float(1e-4), sOutBack.sub(sInBack))),
        0.0,
        1.0
      );
      const tBackWall = mix(tInBack, tOutBack, fracBackWall);
      const hitBackHoleWall = float(1.0)
        .sub(smoothstep(-0.0015, 0.0015, sInBack))
        .mul(smoothstep(-0.0015, 0.0015, sOutBack))
        .mul(validBack);

      const backHit = max(hitInnerConcave, hitBackHoleWall);
      const tBackResolved = mix(tInBack, tBackWall, hitBackHoleWall);
      const sBackResolved = mix(sInBack, float(0.002), hitBackHoleWall);

      hitT.assign(mix(hitT, tBackResolved, backHit));
      hitShellIdx.assign(mix(hitShellIdx, k, backHit));
      hitApertureDist.assign(mix(hitApertureDist, sBackResolved, backHit));
      hitMoldingDist.assign(mix(hitMoldingDist, fInBack.y, backHit));
      hitIsWallBevel.assign(mix(hitIsWallBevel, hitBackHoleWall, backHit));
      hitIsBackHemisphere.assign(mix(hitIsBackHemisphere, float(1.0), backHit));
      hitMask.assign(max(hitMask, backHit));
    });

    // -------------------------------------------------------------------------
    // Exact Analytical + Half-Round Rim Fillet + Lathe-Molding Normal at `pHit`
    // -------------------------------------------------------------------------
    const pHit = ro.add(rd.mul(hitT));
    const rHit = max(float(1e-4), length(pHit));
    const nRadial = pHit.div(rHit).mul(mix(float(1.0), float(-1.0), hitIsBackHemisphere));

    // Tangent-plane gradient of the aperture field `evalShellApertureField`
    const eps = float(0.0035);
    const gx = evalShellApertureField(pHit.add(vec3(eps, 0.0, 0.0)), hitShellIdx, float(0.008)).x.sub(
      evalShellApertureField(pHit.sub(vec3(eps, 0.0, 0.0)), hitShellIdx, float(0.008)).x
    );
    const gy = evalShellApertureField(pHit.add(vec3(0.0, eps, 0.0)), hitShellIdx, float(0.008)).x.sub(
      evalShellApertureField(pHit.sub(vec3(0.0, eps, 0.0)), hitShellIdx, float(0.008)).x
    );
    const gz = evalShellApertureField(pHit.add(vec3(0.0, 0.0, eps)), hitShellIdx, float(0.008)).x.sub(
      evalShellApertureField(pHit.sub(vec3(0.0, 0.0, eps)), hitShellIdx, float(0.008)).x
    );
    const gradS = vec3(gx, gy, gz);
    const tangentGrad = normalize(
      gradS.sub(nRadial.mul(dot(gradS, nRadial))).add(vec3(1e-5, 0.0, 0.0))
    );

    // 1. Concentric lathe-turned decorative molding rings strictly around the circular windows (`hitMoldingDist`)
    const inMoldingBand = float(1.0)
      .sub(smoothstep(0.006, 0.072, hitMoldingDist))
      .mul(smoothstep(0.003, 0.014, hitApertureDist))
      .mul(float(1.0).sub(hitIsWallBevel));
    const moldingWave = sin(hitMoldingDist.mul(220.0)).mul(inMoldingBand);

    // 2. Crisp single circular bead ring around each pierced eyelet (`hitApertureDist < 0.016`)
    const qLocal = rotateToShellFrame(pHit, hitShellIdx);
    const eyeletBead = sin(hitApertureDist.mul(260.0))
      .mul(float(1.0).sub(smoothstep(0.003, 0.018, hitApertureDist)))
      .mul(smoothstep(0.035, 0.065, hitMoldingDist))
      .mul(float(1.0).sub(hitIsWallBevel));

    // 3. Smooth half-round torus lip fillet at every circular aperture rim (`hitApertureDist in [0, 0.018]`)
    const rimFillet = exp(max(float(0.0), hitApertureDist).negate().mul(145.0)).mul(
      float(1.0).sub(hitIsWallBevel)
    );

    // Combine surface normal: satin-smooth turned ivory + concentric circular lathe rings + half-round rim fillet
    const nSurface = nRadial.add(
      tangentGrad.mul(moldingWave.mul(0.22).add(eyeletBead.mul(0.16)).sub(rimFillet.mul(0.48)))
    );
    const nWall = tangentGrad.negate().add(nRadial.mul(0.32));
    const nBeveled = normalize(mix(nSurface, nWall, hitIsWallBevel.mul(0.85)));

    // -------------------------------------------------------------------------
    // Analytical Inter-Shell Key-Light Shadow (`outerShadow`)
    // Casts real directional shadows from the enclosing outer shell (`k - 1`) onto inner shells
    // -------------------------------------------------------------------------
    const keyDir = normalize(uKeyDirLocal);
    const fillDir = normalize(uFillDirLocal);

    const hasEnclosingShell = smoothstep(0.2, 0.8, hitShellIdx);
    const kEnclosing = clamp(hitShellIdx.sub(1.0), 0.0, 5.0);
    const rEnclosing = float(1.12).sub(kEnclosing.mul(0.155)).sub(0.018);
    const bShadow = dot(pHit, keyDir);
    const cShadow = dot(pHit, pHit);
    const discShadow = max(float(0.0), rEnclosing.mul(rEnclosing).sub(cShadow.sub(bShadow.mul(bShadow))));
    const tShadow = bShadow.negate().add(sqrt(discShadow));
    const pShadow = pHit.add(keyDir.mul(tShadow));
    const sShadow = evalShellApertureField(pShadow, kEnclosing, float(0.008)).x;
    // If `sShadow > 0`, the enclosing outer shell blocks the key light!
    const keyBlockedByOuter = smoothstep(-0.032, 0.028, sShadow).mul(hasEnclosingShell);
    const keyShadowFactor = float(1.0).sub(keyBlockedByOuter.mul(0.62));

    // -------------------------------------------------------------------------
    // Kubelka-Munk Mineral Ivory & Distinct 7-Shell Pigment Progression
    // -------------------------------------------------------------------------
    const fineGrain = valueNoise3DTsl(qLocal.mul(22.0));
    const schregerLines = sin(
      qLocal.x.mul(qLocal.y).add(qLocal.z.mul(0.7)).mul(36.0).add(fineGrain.mul(1.2))
    ).mul(0.025);
    const shell01 = clamp(hitShellIdx.div(5.5), 0.0, 1.0);

    // Shell 0: Bone Calcite & Warm Kaolin carved Canton elephant/mammoth mineral ivory
    const s0Ivory = kubelkaMunkGlaze(
      warmKaolin.K,
      warmKaolin.S,
      float(0.28).add(fineGrain.mul(0.09)).add(schregerLines).sub(moldingWave.mul(0.07)),
      boneCalcite.rInf
    );
    // Shell 1: Golden Yellow Ochre & Warm Kaolin
    const s1Ochre = kubelkaMunkGlaze(
      yellowOchre.K,
      yellowOchre.S,
      float(0.52).add(fineGrain.mul(0.14)),
      warmKaolin.rInf
    );
    // Shell 2: Carved Celadon Nephrite Jade (`celadonGlauconite` over `boneCalcite`)
    const s2Celadon = kubelkaMunkGlaze(
      celadon.K,
      celadon.S,
      float(0.58).add(fineGrain.mul(0.14)),
      boneCalcite.rInf
    );
    // Shell 3: Warm Realgar Amber & Yellow Ochre
    const s3Amber = kubelkaMunkGlaze(
      realgarAmber.K,
      realgarAmber.S,
      float(0.56).add(fineGrain.mul(0.15)),
      yellowOchre.rInf
    );
    // Shell 4: Carved Cinnabar Vermilion lacquer
    const s4Cinnabar = kubelkaMunkGlaze(
      cinnabar.K,
      cinnabar.S,
      float(0.64).add(fineGrain.mul(0.16)),
      burntSienna.rInf
    );
    // Shell 5: Deep Lapis Ultramarine
    const s5Lapis = kubelkaMunkGlaze(
      lapis.K,
      lapis.S,
      float(0.64).add(fineGrain.mul(0.16)),
      boneCalcite.rInf
    );

    const p01 = mix(s0Ivory, s1Ochre, smoothstep(0.0, 1.0, hitShellIdx));
    const p12 = mix(p01, s2Celadon, smoothstep(1.0, 2.0, hitShellIdx));
    const p23 = mix(p12, s3Amber, smoothstep(2.0, 3.0, hitShellIdx));
    const p34 = mix(p23, s4Cinnabar, smoothstep(3.0, 4.0, hitShellIdx));
    const p45 = mix(p34, s5Lapis, smoothstep(4.0, 5.0, hitShellIdx));
    const shellAlbedo = mix(p45, s4Cinnabar, smoothstep(5.2, 6.0, hitShellIdx));

    // Subtle Raw Umber antique carving patina inside groove valleys & counterbored walls
    const grooveValley = clamp(moldingWave.negate().add(eyeletBead.negate().mul(0.6)), 0.0, 1.0).max(
      hitIsWallBevel.mul(0.42)
    );
    const antiquePatina = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.52),
      warmKaolin.rInf
    );
    const baseAlbedo = mix(shellAlbedo, antiquePatina, grooveValley.mul(0.44));

    // -------------------------------------------------------------------------
    // Studio Softbox Illumination + Radial Enclosure AO + Warm Core Lantern Bounce
    // -------------------------------------------------------------------------
    const viewDir = rd.negate();
    const halfVec = normalize(keyDir.add(viewDir));

    const ndotlKey = max(float(0.0), dot(nBeveled, keyDir)).mul(keyShadowFactor);
    const ndotlFill = max(float(0.0), dot(nBeveled, fillDir)).mul(0.36);
    // Subsurface warm wrap illumination through thin carved ivory bridges
    const sssWrap = max(float(0.0), dot(nBeveled, keyDir).add(0.45).div(1.45))
      .mul(0.18)
      .mul(float(1.0).sub(shell01.mul(0.5)));

    // Radial enclosure shadowing across concentric shells
    const depthEnclosure = clamp(float(1.0).sub(shell01.mul(0.32)), 0.54, 1.0)
      .mul(mix(float(1.0), float(0.64), hitIsBackHemisphere))
      .mul(rimShadowAtten);

    // Warm subsurface lantern glow radiating outward from the inner Cinnabar/Realgar core
    const coreLightDir = pHit.negate().div(rHit);
    const facesCore = max(float(0.25), dot(nBeveled, coreLightDir));
    const coreAtten = exp(rHit.negate().mul(1.75)).mul(uCoreGlow);
    const transSubsurface = beerLambertTransmittance(realgarAmber.extinction, float(0.36));
    const coreBounce = mix(cinnabar.rInf, realgarAmber.rInf, float(0.55))
      .mul(transSubsurface)
      .mul(facesCore)
      .mul(coreAtten)
      .mul(1.05);

    // Polished turned-ivory specular highlight + half-round rim fillet catchlight
    const spec = max(float(0.0), dot(nBeveled, halfVec))
      .pow(48.0)
      .mul(float(0.32).add(rimFillet.mul(0.22)))
      .mul(keyShadowFactor)
      .mul(depthEnclosure);
    const fresnel = clamp(float(1.0).sub(max(float(0.0), dot(nBeveled, viewDir))), 0.0, 1.0).pow(3.2);
    const isCoreHit = smoothstep(5.4, 5.9, hitShellIdx);

    const directLight = float(0.26).add(ndotlKey.mul(0.92)).add(ndotlFill).add(sssWrap);
    const litColor = baseAlbedo
      .mul(directLight.mul(depthEnclosure))
      .add(coreBounce)
      .add(mix(cinnabar.rInf, realgarAmber.rInf, float(0.45)).mul(isCoreHit.mul(0.75).mul(uCoreGlow)))
      .add(boneCalcite.rInf.mul(spec.add(fresnel.mul(0.15))));

    const slateBg = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    );
    return vec4(mix(slateBg, clamp(litColor, 0.0, 1.0), hitMask), hitMask);
  });

  const outNode = tracePuzzleBall();
  puzzleMat.colorNode = outNode.xyz;
  puzzleMat.opacityNode = outNode.w;

  const puzzleBallMesh = new THREE.Mesh(proxyGeom, puzzleMat);
  stage.specimenGroup.add(puzzleBallMesh);

  // ---------------------------------------------------------------------------
  // Tactile Differential Shell Rotation, Hover Gyro-Parallax & Sightline Syzygy
  // ---------------------------------------------------------------------------
  let rotY = 0.14;
  let rotX = 0.10;
  let targetRotY = 0.14;
  let targetRotX = 0.10;

  let phaseSpin = 0.28;
  let syzygyAligned = false;
  let alignmentBlend = 0.0;
  let hoverX = 0.0;
  let hoverY = 0.0;

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
    const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
    const ny = -((e.clientY / Math.max(1, window.innerHeight)) * 2 - 1);
    if (!isDragging) return;
    const dx = nx - prevX;
    const dy = ny - prevY;
    prevX = nx;
    prevY = ny;
    dragMoved += Math.hypot(dx, dy);

    targetRotY += dx * 1.85;
    targetRotX = Math.max(-0.68, Math.min(0.68, targetRotX - dy * 1.30));
    phaseSpin += (dx - dy * 0.5) * 2.4;
  };
  const onPointerUp = () => {
    if (!isDragging) return;
    isDragging = false;
    if (dragMoved < 0.04) {
      syzygyAligned = !syzygyAligned;
      if (syzygyAligned) {
        // Snap outer ball target orientation to the nearest cubic rose-window sightline
        // with a slight 6° perspective offset so all 6 concentric stepped rims frame the core
        const quarterTurn = Math.PI * 0.5;
        targetRotY = Math.round(targetRotY / quarterTurn) * quarterTurn + 0.11;
        targetRotX = 0.08;
      }
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
      if (!syzygyAligned) {
        targetRotY += dtSec * 0.095;
        phaseSpin += dtSec * 0.28;
      } else {
        phaseSpin += dtSec * 0.03;
      }
    }

    rotY += (targetRotY - rotY) * Math.min(1, dtSec * 6.5);
    rotX += (targetRotX - rotX) * Math.min(1, dtSec * 6.5);

    // Smooth continuous gyroscopic hover parallax from pointer NDC
    const targetHoverX = pointer.ndc.x * 0.55;
    const targetHoverY = pointer.ndc.y * 0.55;
    hoverX += (targetHoverX - hoverX) * Math.min(1, dtSec * 7.5);
    hoverY += (targetHoverY - hoverY) * Math.min(1, dtSec * 7.5);
    uHoverParallax.value.set(hoverX, hoverY);

    puzzleBallMesh.rotation.set(rotX, rotY, 0);
    puzzleBallMesh.updateMatrixWorld();

    // Transform camera and fixed studio softbox lights into mesh-local space
    _camLocal.copy(camera.position);
    puzzleBallMesh.worldToLocal(_camLocal);
    uCamLocal.value.copy(_camLocal);

    _invQuat.copy(puzzleBallMesh.quaternion).invert();
    _keyLocal.copy(WORLD_KEY_DIR).applyQuaternion(_invQuat);
    _fillLocal.copy(WORLD_FILL_DIR).applyQuaternion(_invQuat);
    uKeyDirLocal.value.copy(_keyLocal);
    uFillDirLocal.value.copy(_fillLocal);

    const targetAlign = syzygyAligned ? 1.0 : 0.0;
    alignmentBlend += (targetAlign - alignmentBlend) * Math.min(1, dtSec * 4.8);
    const freeFactor = 1.0 - alignmentBlend * 0.96;
    uFreeFactor.value = freeFactor;

    uShellRotA.value.set(
      Math.sin(phaseSpin * 0.65) * 0.48 * freeFactor,
      -Math.sin(phaseSpin * 0.92) * 0.62 * freeFactor,
      Math.cos(phaseSpin * 0.52) * 0.38 * freeFactor
    );
    uShellRotB.value.set(
      -Math.sin(phaseSpin * 1.15) * 0.72 * freeFactor,
      Math.cos(phaseSpin * 1.38) * 0.84 * freeFactor,
      -Math.sin(phaseSpin * 0.78) * 0.52 * freeFactor
    );

    uCoreGlow.value = 1.08 + alignmentBlend * 0.62 + pointer.speed * 0.22;
  });

  return stage;
}
