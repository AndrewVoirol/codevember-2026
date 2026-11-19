import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  dot,
  exp,
  float,
  length,
  max,
  mix,
  normalWorld,
  normalize,
  positionLocal,
  positionViewDirection,
  positionWorld,
  pow,
  sin,
  smoothstep,
  uniform,
  uv,
  vec2,
  vec3,
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { FLAME_FLOOR_3D } from './mirageRig';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface MirageShaderUniforms {
  uTime: TslNode;
  uReveal: TslNode;
  uAwake: TslNode;
  uLightLift: TslNode;
  uSparkPulse: TslNode;
  uFlicker: TslNode;
  uLightPos: TslNode;
  uEmberPos: TslNode;
  uFrontFade: TslNode;
  uHoverIntensity: TslNode;
}

export function createMirageUniforms(): MirageShaderUniforms {
  return {
    uTime: uniform(0),
    uReveal: uniform(0),
    uAwake: uniform(0),
    uLightLift: uniform(0),
    uSparkPulse: uniform(0),
    uFlicker: uniform(1),
    uLightPos: uniform(new THREE.Vector3(...FLAME_FLOOR_3D)),
    uEmberPos: uniform(new THREE.Vector3(...FLAME_FLOOR_3D)),
    uFrontFade: uniform(1.0),
    uHoverIntensity: uniform(0.0),
  };
}

/**
 * Evaluates distance-adaptive anisotropic finite-length graphite pencil strokes in uniform control flow.
 * Eliminates zoomed periodic zebra bands and checkerboard dots across the 0% -> 100% dolly.
 */
const graphiteHatchField = Fn(
  ([coord2D, camDist, lum, densityScale, baseAngle]: [
    TslNode,
    TslNode,
    TslNode,
    TslNode,
    TslNode,
  ]) => {
    const p = coord2D;
    const octBlend = clamp(float(1.65).sub(camDist).mul(0.85), 0.0, 1.0);
    const tooth = mineralToothFbm(vec3(p.mul(58.0), 0.37), float(1.0));
    const skip = smoothstep(0.18, 0.68, tooth);

    const evalStrokeLayer = (angle: TslNode, freqLo: number, freqHi: number, seed: number) => {
      const ca = cos(angle);
      const sa = sin(angle);
      const uPerp = p.x.mul(ca).sub(p.y.mul(sa));
      const uPar = p.x.mul(sa).add(p.y.mul(ca));
      const nLo = valueNoise3DTsl(
        vec3(uPerp.mul(freqLo), uPar.mul(freqLo * 0.055), float(seed))
      );
      const nHi = valueNoise3DTsl(
        vec3(uPerp.mul(freqHi), uPar.mul(freqHi * 0.055), float(seed + 7.13))
      );
      const nCombined = mix(nLo, nHi, octBlend);
      return smoothstep(0.46, 0.78, nCombined);
    };

    const sLine1 = evalStrokeLayer(baseAngle, 220.0, 780.0, 1.3);
    const sLine2 = evalStrokeLayer(baseAngle.add(0.94), 195.0, 690.0, 4.7);
    const sLine3 = evalStrokeLayer(baseAngle.sub(0.62), 170.0, 610.0, 9.1);

    const w1 = clamp(float(0.84).sub(lum).mul(2.1), 0.0, 1.0);
    const w2 = clamp(float(0.58).sub(lum).mul(2.3), 0.0, 1.0);
    const w3 = clamp(float(0.34).sub(lum).mul(2.8), 0.0, 1.0);

    const strokeDark = clamp(
      sLine1
        .mul(w1)
        .mul(0.42)
        .add(sLine2.mul(w2).mul(0.48))
        .add(sLine3.mul(w3).mul(0.60))
        .mul(skip)
        .mul(densityScale),
      0.0,
      0.92
    );
    return strokeDark;
  }
);

/**
 * Builds the TSL NodeMaterial for the 7 carved torn-paper arches, curved back sanctum sheet,
 * riffled side pages, and cloth cover spine.
 */
export function createArchSheetMaterial(
  u: MirageShaderUniforms,
  isFrontSpread: boolean
): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: isFrontSpread,
    depthWrite: true,
  });

  const bone = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const umber = createPigmentUniforms('rawUmber');
  const lapis = createPigmentUniforms('lapisUltramarine');
  const azurite = createPigmentUniforms('azuritePrussian');
  const amber = createPigmentUniforms('realgarAmber');
  const ochre = createPigmentUniforms('yellowOchre');

  const aDeckle = attribute('aDeckle', 'float');
  const aLayerIdx = attribute('aLayerIdx', 'float');
  const aHatchDensity = attribute('aHatchDensity', 'float');
  const aHatchAngle = attribute('aHatchAngle', 'float');
  const aRadial = attribute('aRadial', 'float');

  mat.colorNode = Fn(() => {
    const wp = positionWorld;
    const nw = normalWorld;

    // Fine cold-press cotton-rag paper tooth
    const paperTooth = mineralToothFbm(wp.mul(22.0), float(1.0));
    const coarseTooth = mineralToothFbm(wp.mul(6.5), float(1.0));
    const ragMod = float(0.95).add(paperTooth.mul(0.085));

    // Select base mineral tone smoothly per layer index (0..9)
    const is0 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(0.0)));
    const is1 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(1.0)));
    const is2 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(2.0)));
    const is3 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(3.0)));
    const is4 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(4.0)));
    const is5 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(5.0)));
    const is6 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(6.0)));
    const is7 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(7.0)));
    const is8 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(8.0)));
    const is9 = smoothstep(0.5, 0.0, abs(aLayerIdx.sub(9.0)));

    // Mineral glazes via Kubelka-Munk
    const boneBase = bone.rInf.mul(0.88);
    const kaolinBase = kaolin.rInf.mul(0.84);
    const umberWash = kubelkaMunkGlaze(
      umber.K,
      umber.S,
      float(0.28).add(coarseTooth.mul(0.22)),
      kaolinBase
    );
    const lapisGranulation = float(0.45).add(coarseTooth.mul(0.55));
    const lapisWash = kubelkaMunkGlaze(
      mix(azurite.K, lapis.K, 0.65),
      mix(azurite.S, lapis.S, 0.65),
      lapisGranulation.mul(0.75),
      kaolinBase
    );
    const charcoalWash = vec3(0.095, 0.098, 0.105).mul(float(0.82).add(coarseTooth.mul(0.36)));
    const darkCover = umber.rInf.mul(0.34);

    let baseCol = boneBase.mul(is0.add(is6).add(is7).add(is8.mul(0.86)));
    const cornerShade = smoothstep(0.05, 0.55, aRadial);
    const layer1Base = mix(kaolinBase, vec3(0.042, 0.044, 0.048), cornerShade.mul(0.84));
    baseCol = baseCol.add(layer1Base.mul(is1));
    baseCol = baseCol.add(kaolinBase.mul(is5));
    baseCol = baseCol.add(umberWash.mul(is2));
    baseCol = baseCol.add(lapisWash.mul(is3));
    baseCol = baseCol.add(charcoalWash.mul(is4));
    baseCol = baseCol.add(darkCover.mul(is9));

    const camDist = max(float(0.12), length(cameraPosition.sub(wp)));

    // Directional 6B graphite hatching on Arches 1, 2, 4 (and subtle page shading)
    const hatchAmt = graphiteHatchField(
      wp.xy,
      camDist,
      float(0.35).sub(aHatchDensity.mul(0.22)),
      aHatchDensity.mul(1.18),
      aHatchAngle
    );
    const graphiteInk = vec3(0.022, 0.023, 0.026);
    baseCol = mix(baseCol, graphiteInk, hatchAmt);

    // Subtle tunnel depth AO per arch step, lifted dynamically as lantern light rises
    const depthStep = clamp(aLayerIdx, 0.0, 6.0);
    const tunnelShadow = mix(
      float(1.0).sub(depthStep.mul(0.068)),
      float(1.0).sub(depthStep.mul(0.028)),
      u.uLightLift
    );
    // Slight inward darkening along beveled lip
    const bevelAO = mix(float(0.92), float(1.0), smoothstep(0.0, 0.2, aRadial));
    baseCol = baseCol.mul(tunnelShadow).mul(bevelAO).mul(ragMod);

    // Front page center gutter crease (Layer 0)
    const gutter = exp(wp.x.mul(wp.x).div(0.0014).negate()).mul(is0);
    baseCol = mix(baseCol, vec3(0.28, 0.25, 0.21), gutter.mul(0.55));

    // Continuous 3D ray-cast crease shadow sweeping forward-left across bottom sills (Layers 6..1 -> Layer 0)
    const dzFromEmber = max(float(0.0), wp.z.sub(u.uEmberPos.z));
    const shCenterX = u.uEmberPos.x.sub(dzFromEmber.mul(0.34));
    const shHalfW = float(0.055).add(dzFromEmber.mul(0.24));
    const yReach = float(-0.22).sub(dzFromEmber.mul(0.52));
    const shCenterY = mix(float(-0.22), yReach, 0.55);
    const shHalfH = max(float(0.032), abs(float(-0.22).sub(yReach)).mul(0.55));
    const shNormDist = length(
      vec2(wp.x.sub(shCenterX).div(shHalfW), wp.y.sub(shCenterY).div(shHalfH))
    );
    const raggedEdge = mineralToothFbm(wp.mul(26.0), float(1.0)).sub(0.5).mul(0.32);
    const isShadowLayer = smoothstep(6.6, 6.4, aLayerIdx).mul(smoothstep(-0.20, -0.225, wp.y));
    const shadowMask = smoothstep(1.05, 0.36, shNormDist.add(raggedEdge))
      .mul(isShadowLayer)
      .mul(clamp(float(1.0).sub(u.uLightLift.mul(1.12)), 0.0, 1.0));
    const shadowHatch = graphiteHatchField(wp.xy, camDist, float(0.16), float(1.12), float(0.25));
    baseCol = mix(baseCol, vec3(0.022, 0.023, 0.026), shadowMask.mul(float(0.76).add(shadowHatch.mul(0.22))));

    // Back Sanctum Wall (Layer 7): soft contra-jour figure shadow at 0% + rising warm amber lantern halo!
    const figShadowDx = wp.x.sub(0.06);
    const figShadowDy = wp.y.add(0.08);
    const figShadowR = length(vec3(figShadowDx.mul(1.3), figShadowDy.mul(0.95), float(0.0)));
    const wallShadow = smoothstep(0.28, 0.06, figShadowR)
      .mul(is7)
      .mul(clamp(float(1.0).sub(u.uLightLift.mul(0.95)), 0.0, 1.0));
    baseCol = mix(baseCol, vec3(0.32, 0.30, 0.28), wallShadow.mul(0.42));

    // Warm golden-amber bounce light on inner arches (4..7) from rising lantern + crease ember
    const dLight = length(wp.sub(u.uLightPos));
    const dEmber = length(wp.sub(u.uEmberPos));
    const lanternBounce = exp(dLight.mul(dLight).mul(5.2).negate())
      .mul(float(0.15).add(u.uLightLift.mul(0.92)).add(u.uSparkPulse.mul(0.25)))
      .mul(u.uFlicker);
    const emberBounce = exp(dEmber.mul(dEmber).mul(32.0).negate())
      .mul(float(0.72).add(u.uSparkPulse.mul(0.45)))
      .mul(u.uFlicker);
    const warmBounceCol = mix(amber.rInf, ochre.rInf, 0.52);

    const innerAccept = clamp(is4.add(is5).add(is6).add(is7).add(is3.mul(0.35)), 0.0, 1.0);
    baseCol = baseCol.add(warmBounceCol.mul(lanternBounce.mul(0.44).add(emberBounce.mul(0.48))).mul(innerAccept));

    // Soft studio softbox vignette across back sanctum wall at 100%
    const wallHaloDist = length(wp.xy.sub(vec2(-0.08, -0.02)));
    const wallHalo = exp(wallHaloDist.mul(wallHaloDist).mul(3.2).negate())
      .mul(is7)
      .mul(float(0.08).add(u.uLightLift.mul(0.28)));
    baseCol = baseCol.add(mix(amber.rInf, vec3(0.96, 0.85, 0.62), 0.45).mul(wallHalo));

    // Crisp uncolored white cotton-rag torn deckle rim (thin fibrous bevel + warm sill ember scorch)
    const bevelLight = clamp(dot(nw, vec3(-0.22, 0.55, 0.80)).mul(0.32).add(0.78), 0.58, 1.08);
    const fiberCol = vec3(0.89, 0.86, 0.79).mul(bevelLight).mul(float(0.92).add(paperTooth.mul(0.14)));
    const deckleWarm = warmBounceCol.mul(lanternBounce.mul(0.32).add(emberBounce.mul(0.65)));
    const deckleMask = smoothstep(0.18, 0.92, aDeckle);
    baseCol = mix(baseCol, fiberCol.add(deckleWarm), deckleMask);

    // Subtle riffled page gaps along stacked side strips (Layer 8)
    const riffleLines = smoothstep(0.3, 0.85, sin(wp.z.mul(180.0)).mul(0.5).add(0.5)).mul(is8);
    baseCol = mix(baseCol, vec3(0.32, 0.29, 0.24), riffleLines.mul(0.45));

    // Gentle overhead studio softbox gradient so paper reads physically suspended in #212529
    const studioKey = clamp(dot(nw, vec3(0.22, 0.42, 0.88)).mul(0.14).add(0.92), 0.74, 1.06);
    const outCol = clamp(baseCol.mul(studioKey), 0.0, 0.98);
    return outCol;
  })();

  if (isFrontSpread) {
    mat.opacityNode = u.uFrontFade;
  }

  return mat;
}

/**
 * Builds the TSL 3-tier Graphite Tonal Art Map material for the curled young woman
 * (contra-jour charcoal silhouette at 0% -> warm amber-lit 2H/2B graphite modeling at 100%).
 */
export function createFigureMaterial(
  u: MirageShaderUniforms,
  mode: 'body' | 'hair' | 'strokes'
): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: mode === 'strokes',
    depthWrite: mode !== 'strokes',
  });

  const bone = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const amber = createPigmentUniforms('realgarAmber');
  const ochre = createPigmentUniforms('yellowOchre');
  const sulfur = createPigmentUniforms('sporeSulfur');
  const aFeature = attribute('aFeature', 'float');

  mat.colorNode = Fn(() => {
    const wp = positionWorld;
    const lp = positionLocal;
    const nw = normalWorld;
    const vd = positionViewDirection;

    const toLight = u.uLightPos.sub(wp);
    const lightDist = max(float(0.05), length(toLight));
    const lightDir = toLight.div(lightDist);
    const ndotl = max(float(0.0), dot(nw, lightDir));
    const nearHandMask = exp(lightDist.mul(10.5).negate()).mul(u.uLightLift);
    const wrapNdotl = mix(ndotl, clamp(dot(nw, lightDir).mul(0.45).add(0.55), 0.0, 1.0), nearHandMask.mul(0.82));

    // At 0% rest (uLightLift=0), uLightPos sits behind her (contra-jour), keeping front normals dark!
    // At 100% reveal, exposure stays balanced in [0.03..0.76] so 2H/2B hatching models cheeks & palms!
    const distAtten = float(1.0).div(float(0.24).add(lightDist.mul(lightDist).mul(6.8)));
    const frontPower = float(0.12).add(u.uLightLift.mul(0.68)).add(u.uSparkPulse.mul(0.14));
    const keyIrr = wrapNdotl.mul(distAtten).mul(frontPower).mul(u.uFlicker);

    // Tiny floor crease ember grazes lower shins/knees without lighting upper face at 0%
    const toEmber = u.uEmberPos.sub(wp);
    const emberDist = max(float(0.04), length(toEmber));
    const emberDir = toEmber.div(emberDist);
    const emberNdotl = max(float(0.0), dot(nw, emberDir));
    const heightMask = exp(max(float(0.0), wp.y.sub(u.uEmberPos.y)).mul(13.0).negate());
    const emberIrr = emberNdotl
      .mul(exp(emberDist.mul(emberDist).mul(26.0).negate()))
      .mul(heightMask)
      .mul(float(0.65).add(u.uSparkPulse.mul(0.45)))
      .mul(u.uFlicker);

    // Subtle grazing back-rim against the bright sanctum wall at 0% so spine contour stays crisp
    const rim = pow(clamp(float(1.0).sub(abs(dot(nw, vd))), 0.0, 1.0), 2.5);
    const backRim = rim.mul(float(0.10).add(u.uLightLift.mul(0.12)));

    const totalLum = clamp(float(0.028).add(keyIrr.mul(0.76)).add(emberIrr.mul(0.52)).add(backRim.mul(0.32)), 0.02, 0.78);

    // Base paper & graphite color per body part
    const isCloth = smoothstep(0.5, 1.0, aFeature);
    const skinTone = mix(kaolin.rInf, bone.rInf, 0.52).mul(0.86);
    const clothTone = kaolin.rInf.mul(0.76);
    const bodyBase = mix(skinTone, clothTone, isCloth);

    // Hair base: authentic monochrome 6B/2B graphite with subtle warm rim only on backRim
    const strandTooth = mineralToothFbm(vec3(lp.x.mul(36.0), lp.y.mul(12.0), lp.z.mul(24.0)), float(1.0));
    const hairDark = vec3(0.022, 0.022, 0.024);
    const hairLit = mix(hairDark, vec3(0.11, 0.105, 0.098), u.uLightLift.mul(0.55));
    const hairBase = mix(hairDark, hairLit, clamp(totalLum.mul(0.65).add(strandTooth.mul(0.18)), 0.0, 1.0))
      .add(amber.rInf.mul(backRim).mul(0.18));

    // Distance-adaptive anisotropic pencil hatching on body & sweater
    const camDist = max(float(0.12), length(cameraPosition.sub(wp)));
    const hatchCoord = lp.xy.add(vec2(lp.z.mul(0.58), lp.z.mul(0.32)));
    const hatchAmt = graphiteHatchField(
      hatchCoord,
      camDist,
      totalLum,
      float(mode === 'hair' ? 1.15 : 1.05),
      float(mode === 'hair' ? 0.85 : -0.52)
    );

    // Fresnel graphite pencil contour along anatomical silhouettes (fingers, nose, jawline, sleeves)
    const fresnelInk = smoothstep(0.36, 0.14, abs(dot(nw, vd))).mul(0.82);

    const warmLanternColor = mix(amber.rInf, mix(ochre.rInf, sulfur.rInf, 0.35), clamp(keyIrr, 0.0, 1.0));
    const litSkinAndCloth = bodyBase
      .mul(float(0.05).add(totalLum.mul(0.88)))
      .add(warmLanternColor.mul(keyIrr.mul(0.28).add(emberIrr.mul(0.36))));

    const charcoalBlack = vec3(0.020, 0.020, 0.022);
    const bodyFinal = mix(litSkinAndCloth, charcoalBlack, clamp(hatchAmt.add(fresnelInk.mul(0.65)), 0.0, 0.94));
    const hairFinal = mix(hairBase, charcoalBlack, clamp(hatchAmt.mul(0.7).add(fresnelInk.mul(0.45)), 0.0, 0.92));
    const strokeFinal = vec3(0.026, 0.025, 0.024);

    if (mode === 'hair') return clamp(hairFinal, 0.0, 0.96);
    if (mode === 'strokes') return strokeFinal;
    return clamp(bodyFinal, 0.0, 0.98);
  })();

  if (mode === 'strokes') {
    mat.opacityNode = smoothstep(0.12, 0.72, u.uLightLift).mul(0.78);
  }

  return mat;
}

/**
 * Builds the TSL material for sculpted sclera, amber irises, pupils, catchlights, lashes, and brows.
 * Catchlights and bright sclera stay strictly dark in 0% contra-jour silhouette and illuminate under palm light!
 */
export function createFaceFeaturesMaterial(u: MirageShaderUniforms): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  const bone = createPigmentUniforms('boneCalcite');
  const amber = createPigmentUniforms('realgarAmber');
  const sulfur = createPigmentUniforms('sporeSulfur');
  const aFeature = attribute('aFeature', 'float');

  mat.colorNode = Fn(() => {
    const wp = positionWorld;
    const nw = normalWorld;
    const toLight = u.uLightPos.sub(wp);
    const lightDist = max(float(0.05), length(toLight));
    const ndotl = max(float(0.0), dot(nw, toLight.div(lightDist)));

    // Strictly dark in 0% contra-jour rest; illuminates cleanly as lantern light swings onto face!
    const faceLit = smoothstep(0.22, 0.86, u.uLightLift).mul(clamp(ndotl.mul(1.25).add(0.25), 0.0, 1.0));
    const darkSilhouette = vec3(0.020, 0.020, 0.022);

    const isSclera = smoothstep(0.5, 0.0, abs(aFeature.sub(3.0)));
    const isIris = smoothstep(0.5, 0.0, abs(aFeature.sub(4.0)));
    const isPupil = smoothstep(0.5, 0.0, abs(aFeature.sub(5.0)));
    const isCatch = smoothstep(0.5, 0.0, abs(aFeature.sub(6.0)));
    const isInk = smoothstep(0.5, 0.0, abs(aFeature.sub(7.0)));

    const scleraCol = mix(darkSilhouette, bone.rInf.mul(0.92), faceLit);
    const irisGlow = mix(amber.rInf, sulfur.rInf, 0.35).mul(float(0.85).add(u.uFlicker.mul(0.2)));
    const irisCol = mix(darkSilhouette, irisGlow, faceLit);
    const pupilCol = vec3(0.012, 0.012, 0.014);
    const catchCol = mix(darkSilhouette, vec3(0.98, 0.92, 0.72), smoothstep(0.38, 0.85, u.uLightLift));
    const inkCol = vec3(0.018, 0.018, 0.020);

    const outCol = scleraCol
      .mul(isSclera)
      .add(irisCol.mul(isIris))
      .add(pupilCol.mul(isPupil))
      .add(catchCol.mul(isCatch))
      .add(inkCol.mul(isInk));
    return clamp(outCol, 0.0, 0.99);
  })();

  return mat;
}

export interface MothShaderControl {
  material: THREE.MeshBasicNodeMaterial;
  uAwake: TslNode;
  uHoverGlow: TslNode;
}

/**
 * Dual-state Beer-Lambert translucent origami moth material:
 *   - mAwake = 0: contra-jour 6B charcoal blending seamlessly into curled silhouette (with hover parallax shimmer)
 *   - mAwake = 1: glowing Realgar Amber / Yellow Ochre / Spore Sulfur translucent folded paper
 */
export function createMothWingMaterial(
  u: MirageShaderUniforms,
  pigmentBias: number
): MothShaderControl {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
  });

  const amber = createPigmentUniforms('realgarAmber');
  const ochre = createPigmentUniforms('yellowOchre');
  const sulfur = createPigmentUniforms('sporeSulfur');
  const umber = createPigmentUniforms('rawUmber');

  const uMothAwake = uniform(0.0);
  const uHoverGlow = uniform(0.0);
  const uBias = uniform(pigmentBias);

  mat.colorNode = Fn(() => {
    const uvCoord = uv();
    const nw = normalWorld;
    const vd = positionViewDirection;
    const wp = positionWorld;

    const tooth = mineralToothFbm(wp.mul(45.0), float(1.0));

    // Contra-jour 6B charcoal tone matching the sleeping silhouette at rest
    const charcoalDormant = vec3(0.022, 0.021, 0.024).add(tooth.mul(0.008));

    // Subtle parallax hover rim glint hinting that the silhouette isn't solid
    const rim = pow(clamp(float(1.0).sub(abs(dot(nw, vd))), 0.0, 1.0), 1.8);
    const hoverHint = amber.rInf.mul(rim).mul(uHoverGlow).mul(0.55);

    // Beer-Lambert optical thickness across paper wing (thinner & hotter near spine uv.x = 0)
    const chord = float(0.18).add(uvCoord.x.mul(0.68)).add(tooth.mul(0.12));
    const ext = mix(amber.extinction, ochre.extinction, uBias);
    const transmittance = beerLambertTransmittance(ext, chord);

    const corePigment = mix(amber.rInf, mix(ochre.rInf, sulfur.rInf, uBias), 0.55);
    const litBase = corePigment.mul(float(0.65).add(transmittance.mul(0.55)));

    // Bright golden-ivory spine fold line along uv.x -> 0
    const spineFold = exp(uvCoord.x.mul(uvCoord.x).mul(42.0).negate());
    const spineCol = mix(litBase, vec3(0.98, 0.92, 0.74), spineFold.mul(0.75));

    // Folded wing facet shading
    const facetShade = clamp(abs(dot(nw, normalize(vec3(0.3, 0.8, 0.5)))).mul(0.35).add(0.72), 0.55, 1.08);
    const glowingWing = mix(umber.rInf.mul(0.45), spineCol, facetShade).mul(u.uFlicker);

    return clamp(mix(charcoalDormant.add(hoverHint), glowingWing, uMothAwake), 0.0, 0.99);
  })();

  return { material: mat, uAwake: uMothAwake, uHoverGlow };
}

/**
 * Unfurling 3D graphite spiral curls rising out of cupped hands as moths wake.
 */
export function createSwirlMaterial(u: MirageShaderUniforms): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
  });

  mat.colorNode = Fn(() => vec3(0.14, 0.12, 0.10))();

  mat.opacityNode = Fn(() => {
    const vCoord = uv().y;
    const progress = smoothstep(0.22, 0.88, u.uAwake);
    const revealMask = smoothstep(progress.add(0.05), progress.sub(0.15), vCoord);
    const taperFade = sin(clamp(vCoord, 0.0, 1.0).mul(Math.PI));
    return revealMask.mul(taperFade).mul(smoothstep(0.18, 0.65, u.uAwake)).mul(0.62);
  })();

  return mat;
}

/**
 * Rising amber/sulfur ember flecks swirling with the awakened origami moths.
 */
export function createFleckMaterial(u: MirageShaderUniforms): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const amber = createPigmentUniforms('realgarAmber');
  const sulfur = createPigmentUniforms('sporeSulfur');

  mat.colorNode = Fn(() => {
    const t = clamp(uv().y, 0.0, 1.0);
    return mix(amber.rInf, sulfur.rInf, t).mul(float(0.95).add(u.uFlicker.mul(0.35)));
  })();

  mat.opacityNode = Fn(() => smoothstep(0.12, 0.72, u.uAwake).mul(0.88))();
  return mat;
}

/**
 * Soft radial glowing billboard material for both the tiny bottom crease ember
 * and the warm golden lantern bowl nestled inside the cupped hands.
 */
export function createLanternGlowMaterial(
  u: MirageShaderUniforms,
  kind: 'creaseEmber' | 'palmBowl'
): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const amber = createPigmentUniforms('realgarAmber');
  const sulfur = createPigmentUniforms('sporeSulfur');

  mat.colorNode = Fn(() => {
    const c = uv().sub(0.5).mul(2.0);
    const r = length(c);
    const core = exp(r.mul(r).mul(kind === 'creaseEmber' ? 14.0 : 6.2).negate());
    const halo = exp(r.mul(r).mul(2.6).negate());
    const hotCenter = vec3(1.0, 0.94, 0.76);
    const col = mix(amber.rInf, sulfur.rInf, 0.45).mul(halo).add(hotCenter.mul(core));
    return col.mul(u.uFlicker);
  })();

  mat.opacityNode = Fn(() => {
    const c = uv().sub(0.5).mul(2.0);
    const r = length(c);
    const falloff = smoothstep(1.0, 0.05, r);
    if (kind === 'creaseEmber') {
      // Stays subtly glowing at bottom crease, gently dimming as moths rise
      const amp = mix(float(0.92), float(0.45), u.uLightLift).add(u.uSparkPulse.mul(0.35));
      return falloff.mul(falloff).mul(amp);
    }
    return falloff.mul(falloff).mul(u.uLightLift).mul(float(0.88).add(u.uSparkPulse.mul(0.25)));
  })();

  return mat;
}

/**
 * Teardrop physical candle/ember flame core at the bottom page spine crease.
 */
export function createTeardropFlameMaterial(u: MirageShaderUniforms): THREE.MeshBasicNodeMaterial {
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
  });

  const amber = createPigmentUniforms('realgarAmber');

  mat.colorNode = Fn(() => {
    const p = uv();
    const hot = smoothstep(0.15, 0.65, p.y);
    return mix(amber.rInf, vec3(1.0, 0.95, 0.78), hot).mul(u.uFlicker);
  })();

  mat.opacityNode = Fn(() => {
    const c = uv().sub(0.5).mul(2.0);
    // Teardrop SDF inside [0..1] quad
    const tx = c.x.mul(1.9);
    const ty = c.y.mul(1.1);
    const taper = mix(float(1.0), float(0.18), smoothstep(-0.4, 0.85, ty));
    const d = length(vec3(tx.div(max(float(0.15), taper)), ty.add(0.15), 0.0));
    const mask = smoothstep(0.75, 0.25, d);
    return mask.mul(mix(float(0.96), float(0.58), u.uLightLift));
  })();

  return mat;
}
