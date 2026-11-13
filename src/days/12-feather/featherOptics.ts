import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  fract,
  length,
  max,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  screenUV,
  select,
  sin,
  smoothstep,
  sqrt,
  step,
  vec2,
  vec3,
} from 'three/tsl';
import { CHALKY_SLATE_LINEAR } from '../../core/palette';
import {
  createPigmentUniforms,
  hash31Tsl,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface SharedFeatherOpticsUniforms {
  uTime: TslNode;
  uPointerMeta: TslNode;
  uGonioBeamDir: TslNode;
  uGonioBeamScreen: TslNode;
  uBeamIntensity: TslNode;
}

export interface BladeOpticsUniforms {
  /** x: cantilever flexZ, y: torsional twist rad, z: ripple phase, w: ripple amp */
  uBladeFlex: TslNode;
  /** Pointer position transformed into blade-local space */
  uPointerLocal: TslNode;
  /** Pointer velocity transformed into blade-local space */
  uPointerVelLocal: TslNode;
  /** Length of this feather shaft */
  uFeatherLen: TslNode;
  /** Axial shingle slide offset inside the clasp (used to clip quill cleanly inside clasp hub) */
  uSlideY: TslNode;
  /** Unit shaft axis in world space (for clean physical goniometer angle alignment) */
  uBladeAxisWorld: TslNode;
}

export function createFeatherPigmentSet() {
  return {
    boneCalcite: createPigmentUniforms('boneCalcite'),
    warmKaolin: createPigmentUniforms('warmKaolin'),
    rawUmber: createPigmentUniforms('rawUmber'),
    yellowOchre: createPigmentUniforms('yellowOchre'),
    realgarAmber: createPigmentUniforms('realgarAmber'),
    burntSienna: createPigmentUniforms('burntSienna'),
    verdigrisMalachite: createPigmentUniforms('verdigrisMalachite'),
    celadonGlauconite: createPigmentUniforms('celadonGlauconite'),
    azuritePrussian: createPigmentUniforms('azuritePrussian'),
    lapisUltramarine: createPigmentUniforms('lapisUltramarine'),
    mossUmbrous: createPigmentUniforms('mossUmbrous'),
    sporeSulfur: createPigmentUniforms('sporeSulfur'),
    cinnabarVermilion: createPigmentUniforms('cinnabarVermilion'),
  };
}

export type FeatherPigmentSet = ReturnType<typeof createFeatherPigmentSet>;

/**
 * Custom Chalky Slate (#212529) backdrop node with a subtle directional studio beam wash
 * synchronized to the steered goniometric light direction (`uGonioBeamScreen`), keeping
 * overall room ground tightly anchored to #212529.
 */
export function createGoniometricBackdropNode(uBeamScreen: TslNode) {
  return Fn(() => {
    const uv = screenUV;
    const centered = uv.sub(0.5);
    const vignetteDistSq = dot(centered, centered);

    const baseSlate = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    );

    const radialFalloff = clamp(float(1.032).sub(vignetteDistSq.mul(0.11)), 0.86, 1.04);

    // Subtle soft studio beam shaft aligned with the goniometer light direction
    const beamDir = normalize(vec2(uBeamScreen.x, uBeamScreen.y).add(vec2(1e-4, 1e-4)));
    const along = dot(centered, beamDir);
    const perp = abs(centered.x.mul(beamDir.y).sub(centered.y.mul(beamDir.x)));
    const shaftMask = smoothstep(-0.18, 0.62, along).mul(exp(perp.mul(perp).negate().mul(7.5)));
    const beamTint = vec3(0.010, 0.0078, 0.0052).mul(shaftMask);

    const grain = hash31Tsl(vec3(uv.mul(1024.0), 0.37)).sub(0.5).mul(0.0013);
    return max(vec3(0.006), baseSlate.mul(radialFalloff).add(beamTint).add(grain));
  })();
}

/**
 * Builds the natural fibrous keratin rachis (quill) material:
 * - Seam-free longitudinal keratin fiber striations & lateral barb-socket notches breaking up smooth plastic highlights.
 * - Anatomical median dorsal furrow shadow + subtle amber pith subsurface rim scatter.
 * - Collapses lower calamus excess cleanly inside the clasp pivot (`pos0.y + uSlideY < -0.010`).
 */
export function createRachisMaterial(
  pigments: FeatherPigmentSet,
  bladeUniforms: BladeOpticsUniforms,
  shared: SharedFeatherOpticsUniforms
): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.48,
    metalness: 0.04,
  });

  const pos0 = positionLocal;
  const n0 = normalLocal;
  const aSpineUv = attribute('aSpineUv', 'vec2');
  const sSpine = aSpineUv.x;
  const radAng = aSpineUv.y;

  const hFrac = clamp(pos0.y.div(bladeUniforms.uFeatherLen), 0.0, 1.0);
  const flexBend = bladeUniforms.uBladeFlex.x.mul(hFrac).mul(hFrac);
  const rippleBend = bladeUniforms.uBladeFlex.w
    .mul(sin(hFrac.mul(7.5).sub(bladeUniforms.uBladeFlex.z)))
    .mul(hFrac);
  const ambientSway = sin(shared.uTime.mul(1.1).add(hFrac.mul(2.4)))
    .mul(0.008)
    .mul(hFrac)
    .mul(hFrac);

  const dz = flexBend.add(rippleBend).add(ambientSway);

  // Tuck lower slide excess cleanly inside the clasp pin at y_pivot = -0.010
  const minLocalY = float(-0.010).sub(bladeUniforms.uSlideY);
  const insideVisible = smoothstep(minLocalY.sub(0.012), minLocalY.add(0.012), pos0.y);
  const clippedY = max(pos0.y, minLocalY);

  mat.positionNode = vec3(
    pos0.x.mul(insideVisible),
    clippedY,
    pos0.z.mul(insideVisible).add(dz)
  );

  // Pure longitudinal fibrous keratin grain along stem axis (zero transverse bamboo rings)
  const angCos = cos(radAng);
  const angSin = sin(radAng);
  const longGrain = mineralToothFbm(
    vec3(angCos.mul(6.8), angSin.mul(6.8), sSpine.mul(26.0)),
    float(3.0)
  );
  const fineFlutes = sin(radAng.mul(34.0).add(longGrain.mul(4.6))).mul(0.5).add(0.5);
  const keratinFiber = float(0.62).mul(longGrain).add(float(0.38).mul(fineFlutes));

  const sulcusHollow = exp(angCos.mul(angCos).negate().mul(7.5)).mul(smoothstep(0.05, 0.45, sSpine));

  // Fine tangential normal breakup along lengthwise keratin fibers (breaks plastic specular into hairline horn sheen)
  const ribTangentBump = cos(radAng.mul(34.0).add(longGrain.mul(4.6))).mul(0.11);
  mat.normalNode = normalize(
    n0.add(vec3(angSin.negate().mul(ribTangentBump), dz.negate().mul(0.9), angCos.mul(ribTangentBump)))
  );

  // Calamus parchment-horn -> dorsal straw-cream pith line -> roasted umber fibrous keratin walls
  const shaftDarken = smoothstep(0.04, 0.20, sSpine);
  const calamusParchment = mix(
    kubelkaMunkGlaze(
      pigments.boneCalcite.K,
      pigments.boneCalcite.S,
      float(0.58).add(longGrain.mul(0.24)),
      pigments.warmKaolin.rInf
    ).mul(0.84),
    pigments.yellowOchre.rInf.mul(0.66),
    float(0.24).add(longGrain.mul(0.20))
  );

  const dorsalHoneyHorn = mix(pigments.yellowOchre.rInf.mul(0.56), pigments.warmKaolin.rInf.mul(0.62), float(0.42));
  const roastedKeratin = mix(pigments.rawUmber.rInf, pigments.burntSienna.rInf, float(0.20)).mul(0.46);
  const dorsalFactor = clamp(angSin.mul(0.5).add(0.5), 0.0, 1.0);
  const shaftBody = mix(roastedKeratin, dorsalHoneyHorn, dorsalFactor.mul(0.56).add(keratinFiber.mul(0.30)));
  const groovedShaft = shaftBody.mul(float(1.0).sub(sulcusHollow.mul(0.45)));

  mat.colorNode = mix(calamusParchment, groovedShaft, shaftDarken);
  // Satin-matte organic roughness breakup along fiber grain & sulcus groove (no plastic gloss)
  mat.roughnessNode = clamp(
    float(0.40).add(shaftDarken.mul(0.16)).add(keratinFiber.mul(0.26)).add(sulcusHollow.mul(0.16)),
    0.34,
    0.86
  );

  // Delicate warm medullary keratin subsurface scatter along thin backlit pith edges
  const V = normalize(cameraPosition.sub(positionWorld));
  const L = normalize(shared.uGonioBeamDir);
  const nW = normalize(normalWorld);
  const rimScatter = pow(abs(float(1.0).sub(clamp(dot(nW, V), 0.0, 1.0))), float(2.2));
  const keyFeed = max(dot(nW, L), float(0.12));
  const hornGlow = mix(pigments.yellowOchre.rInf, pigments.realgarAmber.rInf, float(0.38))
    .mul(rimScatter)
    .mul(keyFeed)
    .mul(float(0.055).mul(float(1.0).sub(sulcusHollow.mul(0.7))))
    .mul(shared.uBeamIntensity);
  mat.emissiveNode = clamp(hornGlow, 0.0, 0.075);

  return mat;
}

/**
 * Builds the radial-brushed antique dark bronze leaf guard material (oil-rubbed patina recesses + worn bevel edges).
 */
export function createBronzeGuardMaterial(pigments: FeatherPigmentSet): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.36,
    metalness: 0.90,
  });

  const pos0 = positionLocal;
  const n0 = normalLocal;
  const radDir = normalize(vec2(pos0.x, pos0.y).add(vec2(1e-4, 1e-4)));
  const radDist = length(vec2(pos0.x, pos0.y));
  const brushed = mineralToothFbm(vec3(radDir.mul(16.0), radDist.mul(14.0)), float(3.2));
  const turnLines = sin(radDist.mul(210.0).add(brushed.mul(2.8))).mul(0.5).add(0.5);

  // Chamfered bevel shoulders show lighter burnished antique brass wear against deep oxidized faces
  const edgeWear = clamp(float(1.0).sub(abs(n0.z)), 0.0, 1.0);
  const darkPatina = mix(pigments.rawUmber.rInf, pigments.burntSienna.rInf, float(0.16)).mul(0.28);
  const burnishedBrass = mix(pigments.yellowOchre.rInf, pigments.realgarAmber.rInf, float(0.32)).mul(0.84);
  const patinaBlend = clamp(edgeWear.mul(0.68).add(brushed.mul(0.28)).add(turnLines.mul(0.10)), 0.0, 1.0);

  mat.colorNode = mix(darkPatina, burnishedBrass, patinaBlend);
  mat.roughnessNode = clamp(
    float(0.52).sub(edgeWear.mul(0.22)).sub(brushed.mul(0.14)).add(turnLines.mul(0.08)),
    0.18,
    0.64
  );
  return mat;
}

/**
 * Builds the lathe-turned aged Bone Calcite ferrule collar & central rivet pin material
 * with longitudinal osteon fibers and tea-stained turning grooves (zero smooth plastic look).
 */
export function createBoneFerruleMaterial(pigments: FeatherPigmentSet): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.44,
    metalness: 0.03,
  });

  const pos0 = positionLocal;
  const osteon = mineralToothFbm(vec3(pos0.x.mul(54.0), pos0.y.mul(8.0), pos0.z.mul(54.0)), float(2.8));
  const latheGrooves = smoothstep(float(0.72), float(0.99), abs(sin(pos0.y.mul(145.0))));
  const stainMask = clamp(osteon.mul(0.65).add(latheGrooves.mul(0.35)), 0.0, 1.0);

  const ivoryBase = kubelkaMunkGlaze(
    pigments.boneCalcite.K,
    pigments.boneCalcite.S,
    float(0.54).add(osteon.mul(0.30)),
    pigments.warmKaolin.rInf
  ).mul(0.76);
  const agedPatina = mix(pigments.yellowOchre.rInf, pigments.rawUmber.rInf, float(0.56)).mul(0.68);

  mat.colorNode = mix(ivoryBase, agedPatina, float(0.18).add(stainMask.mul(0.36)));
  mat.roughnessNode = clamp(float(0.44).add(osteon.mul(0.24)).add(latheGrooves.mul(0.16)), 0.38, 0.84);
  return mat;
}

/**
 * Builds the 3D Peacock Feather Barb & Ocellus Photonic-Crystal TSL Material for one blade.
 */
export function createFeatherBarbMaterial(
  bladeIndex: number,
  pigments: FeatherPigmentSet,
  bladeUniforms: BladeOpticsUniforms,
  shared: SharedFeatherOpticsUniforms
): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    alphaToCoverage: true,
    transparent: false,
    roughness: 0.34,
    metalness: 0.24,
  });

  const pos0 = positionLocal;
  const n0 = normalLocal;
  const aBarbData = attribute('aBarbData', 'vec4'); // [sSpine, uAlongBarb, sideSign, seed + 10*isOcellus]
  const aBarbGeom = attribute('aBarbGeom', 'vec2'); // [vAcross (-1..1), barbLength]
  const aOcellusCoord = attribute('aOcellusCoord', 'vec2');

  const sSpine = aBarbData.x;
  const uBarb = aBarbData.y;
  const sideSign = aBarbData.z;
  const barbSeed = fract(aBarbData.w);
  const isOcellus = step(5.0, aBarbData.w);
  const vAcross = aBarbGeom.x;
  const barbLen = aBarbGeom.y;

  // -------------------------------------------------------------------------
  // 1. Vertex Physics: Shaft Flex, Twist, Ripple & Pointer Breeze Wake (Shot 05)
  // -------------------------------------------------------------------------
  const hFrac = clamp(pos0.y.div(bladeUniforms.uFeatherLen), 0.0, 1.0);
  const bendMask = hFrac.mul(hFrac);

  const flexBend = bladeUniforms.uBladeFlex.x.mul(bendMask);
  const rippleBend = bladeUniforms.uBladeFlex.w
    .mul(sin(hFrac.mul(7.5).sub(bladeUniforms.uBladeFlex.z)))
    .mul(hFrac);

  const rotTwist = bladeUniforms.uBladeFlex.y.mul(hFrac);
  const dzTwist = pos0.x.mul(sin(rotTwist));

  const bladePhase = float(bladeIndex * 0.93);
  const ambientFlutter = sin(
    shared.uTime.mul(1.25).add(pos0.y.mul(2.3)).add(bladePhase).add(barbSeed.mul(2.2))
  )
    .mul(0.013)
    .mul(uBarb)
    .mul(float(1.15).sub(isOcellus.mul(0.45)));

  const deltaPtr = pos0.sub(bladeUniforms.uPointerLocal);
  const d2 = dot(deltaPtr.xy, deltaPtr.xy).add(deltaPtr.z.mul(deltaPtr.z).mul(0.32));
  const wake = exp(d2.negate().mul(4.5));

  const ptrSpeed = length(bladeUniforms.uPointerVelLocal);
  const wakeStrength = clamp(ptrSpeed.mul(0.35).add(shared.uPointerMeta.z.mul(0.55)).add(0.18), 0.15, 1.75);

  const tipMobility = uBarb.mul(float(0.32).add(uBarb.mul(0.68))).mul(float(1.28).sub(isOcellus.mul(0.52)));
  const radialDir = normalize(vec2(deltaPtr.x, deltaPtr.y).add(vec2(1e-4, 1e-4)));

  const partOffsetXY = radialDir
    .mul(wake.mul(tipMobility).mul(float(0.045).add(wakeStrength.mul(0.045))))
    .add(bladeUniforms.uPointerVelLocal.xy.mul(wake.mul(tipMobility).mul(0.022)));

  const wakeZ = wake.mul(tipMobility).mul(float(0.065).add(wakeStrength.mul(0.095)));
  const totalDisp = vec3(
    partOffsetXY.x,
    partOffsetXY.y,
    flexBend.add(rippleBend).add(dzTwist).add(ambientFlutter).add(wakeZ)
  );

  mat.positionNode = pos0.add(totalDisp);

  // Sleek smooth central ramus spine + ultra-fine bilateral herringbone barbules confined strictly to wings
  const barbDist = uBarb.mul(barbLen);
  const ciliaSkew = barbDist.mul(320.0).sub(abs(vAcross).mul(4.2)).add(barbSeed.mul(31.0));
  const ciliaWave = sin(ciliaSkew).mul(0.5).add(0.5);
  const ciliaNoise = mineralToothFbm(
    vec3(barbDist.mul(22.0), vAcross.mul(4.0), barbSeed.mul(7.0)),
    float(2.0)
  );
  const absV = abs(vAcross);
  const vSign = select(vAcross.lessThan(float(0.0)), float(-1.0), float(1.0));
  const ramusCore = float(1.0).sub(smoothstep(float(0.24), float(0.44), absV));
  const wingOnly = float(1.0).sub(ramusCore);
  const wingFade = float(1.0).sub(smoothstep(float(0.28), float(0.98), absV));
  const microBarbules = pow(ciliaWave, float(1.55)).mul(wingFade).mul(float(0.68).add(ciliaNoise.mul(0.40)));

  // Keep central ramus smooth (zero lump beading!); restrict diagonal barbule relief strictly to fine side cilia
  const ciliaRidgeTilt = cos(ciliaSkew).mul(vSign).mul(wingFade).mul(wingOnly).mul(0.16);
  const normalTiltX = bladeUniforms.uBladeFlex.y
    .negate()
    .mul(0.65)
    .add(radialDir.x.mul(wake).mul(tipMobility).mul(0.85))
    .add(ciliaRidgeTilt);
  const normalTiltY = flexBend
    .add(rippleBend)
    .mul(-0.95)
    .add(radialDir.y.mul(wake).mul(tipMobility).mul(0.65));
  mat.normalNode = normalize(n0.add(vec3(normalTiltX, normalTiltY, 0.0)));

  // -------------------------------------------------------------------------
  // 2. Anatomical Ocellus Ring Parameterization & Herringbone Cilia Opacity
  // -------------------------------------------------------------------------
  const ox = aOcellusCoord.x;
  const oy = aOcellusCoord.y;

  // Inner notched cordate (heart/kidney) nucleus where rachis enters from below
  const rxCore = float(0.066);
  const ryCore = float(0.074);
  const px = abs(ox).div(rxCore);
  const py = oy.add(0.012).div(ryCore);
  const cleftIndent = float(0.38)
    .mul(exp(px.mul(px).negate().mul(4.8)))
    .mul(smoothstep(float(0.18), float(-0.56), py));
  const rHeart = length(vec2(px, py.add(cleftIndent)));

  // Concentric oval halos forming the turquoise-emerald corona & crisp bronze shield
  const rOval = length(vec2(ox.div(0.084), oy.sub(0.008).div(0.092)));
  const zoneBlend = smoothstep(float(0.75), float(1.42), rOval);
  const rZone = mix(rHeart, rOval, zoneBlend);

  const microVar = mineralToothFbm(vec3(ox.mul(16.0), oy.mul(16.0), barbSeed.mul(3.1)), float(2.2));
  const rAnatomical = rZone.add(microVar.sub(0.5).mul(0.07));

  // Smooth sleek barb shaft tone vs gossamer bilateral cilia edge highlight
  const ciliaWingShade = float(0.82).add(ciliaWave.mul(0.18));
  const strandShade = mix(ciliaWingShade, float(0.96).add(ciliaNoise.mul(0.06)), ramusCore);

  // Solid interlocked eye core (1.0 coverage = zero MSAA stipple pinholes!) -> delicate cilia outward & down train
  const innerEyeLock = float(1.0).sub(smoothstep(float(1.12), float(1.52), rAnatomical));
  const eyeWeb = isOcellus.mul(innerEyeLock);
  const tipFeather = float(1.0).sub(smoothstep(float(0.86), float(1.0), uBarb));
  mat.opacityNode = clamp(max(ramusCore, max(microBarbules, eyeWeb)).mul(tipFeather), 0.0, 1.0);

  // Root attachment blend so barb roots emerge smoothly from the dark shaft without chevron stripes
  const rootBlend = smoothstep(float(0.0), float(0.16), uBarb);

  // -------------------------------------------------------------------------
  // 3. Physical Goniometric Photonic-Crystal Interference (Shots 01, 03, 05)
  // -------------------------------------------------------------------------
  const V = normalize(cameraPosition.sub(positionWorld));
  const L = normalize(shared.uGonioBeamDir);
  const H = normalize(L.add(V));

  const nW = normalize(normalWorld);
  const nEff = select(dot(nW, V).lessThan(float(0.0)), nW.negate(), nW);

  const ndotL = clamp(dot(nEff, L), 0.0, 1.0);
  const ndotH = clamp(dot(nEff, H), 0.0, 1.0);
  const ndotV = clamp(dot(nEff, V), 0.0, 1.0);
  const fresnel = pow(abs(float(1.0).sub(ndotV)), float(2.8));

  const bladeAxis: TslNode = normalize(vec3(bladeUniforms.uBladeAxisWorld));
  const lateralAlign = bladeAxis.x.mul(L.x).add(bladeAxis.y.mul(L.y).sub(L.y.mul(0.82)).mul(0.45));
  // Real view-elevation angle dispersion across vertical furled stacks (V.y goes more negative higher up)
  const verticalStackFactor = float(1.0).sub(clamp(abs(bladeAxis.x).mul(2.8), 0.0, 1.0));
  const viewElevationShift = V.y.add(0.09).mul(3.4).mul(verticalStackFactor);
  const localNormalAlign = dot(nEff, L).sub(0.49);
  const gonioShift = clamp(
    lateralAlign
      .mul(1.18)
      .add(viewElevationShift)
      .add(localNormalAlign.mul(0.45))
      .add(fresnel.sub(0.12).mul(0.28))
      .add(ciliaWave.sub(0.5).mul(wingOnly).mul(0.09)),
    -1.0,
    1.0
  );
  const gonio01 = gonioShift.mul(0.5).add(0.5); // 0 = cool off-beam lapis/azurite, 0.5 = malachite, 1 = copper-amber

  // Anisotropic Kajiya-Kay cylindrical barb ribbon + faceted photonic barbulet shimmer sparkles
  const sideOrtho: TslNode = normalize(cross(bladeAxis, vec3(0.0, 0.0, 1.0)).add(vec3(1e-4, 0.0, 0.0)));
  const barbTang: TslNode = normalize(
    bladeAxis.mul(0.72).add(sideOrtho.mul(sideSign).mul(0.68)).add(nEff.mul(0.16))
  );
  const tDotH = dot(barbTang, H);
  const sinTH = sqrt(max(float(0.001), float(1.0).sub(tDotH.mul(tDotH))));
  // Narrow high-contrast specular silk ribbon instead of broad milky veil
  const anisoSheen = pow(sinTH, float(18.0)).mul(float(0.30).add(ndotL.mul(0.70)));
  const grazingSpec = pow(abs(ndotH), float(28.0)).mul(float(0.25).add(ndotL.mul(0.75)));

  // Sharp multi-frequency photonic-crystal glint beats traveling across flexing barbules
  const shimmerBeatA = sin(
    barbSeed.mul(47.0).add(sSpine.mul(38.0)).add(gonioShift.mul(11.0)).add(shared.uTime.mul(2.1))
  );
  const shimmerBeatB = sin(
    uBarb.mul(22.0).add(barbSeed.mul(19.0)).sub(gonioShift.mul(8.5)).add(shared.uTime.mul(1.6))
  );
  const glintMask = pow(max(float(0.0), shimmerBeatA.mul(shimmerBeatB).mul(0.5).add(0.5)), float(5.5));
  const shimmerGlint = glintMask
    .mul(float(0.35).add(anisoSheen.mul(0.95)).add(fresnel.mul(0.55)))
    .mul(float(0.35).add(ndotL.mul(0.65)));

  // -------------------------------------------------------------------------
  // 4. Deep-Chroma Mineral Ocellus & Photonic-Crystal Structural Shimmer
  // -------------------------------------------------------------------------
  // Zone 0 (r < 0.78): Deep velvet Lapis-Indigo heart (protected from pale warm glare!)
  const heartBase = mix(
    pigments.lapisUltramarine.rInf.mul(0.68),
    pigments.azuritePrussian.rInf.mul(1.12),
    smoothstep(float(0.36), float(0.84), rAnatomical)
  );
  const heartColor = mix(heartBase, pigments.lapisUltramarine.rInf.mul(1.18), gonio01.mul(0.22));

  // Pure chromatic anchors (avoid chalky celadon dilution inside base bands)
  const peacockSapphire = mix(pigments.lapisUltramarine.rInf.mul(1.45), pigments.azuritePrussian.rInf.mul(1.78), float(0.66));
  const peacockTurquoise = mix(pigments.azuritePrussian.rInf.mul(1.38), pigments.verdigrisMalachite.rInf.mul(1.68), float(0.68));
  const peacockEmerald = mix(
    pigments.verdigrisMalachite.rInf.mul(vec3(0.75, 1.85, 1.38)),
    pigments.mossUmbrous.rInf.mul(vec3(0.80, 1.55, 1.05)),
    float(0.18)
  );
  const copperAmber = mix(pigments.realgarAmber.rInf.mul(1.22), pigments.cinnabarVermilion.rInf.mul(1.08), float(0.26));
  const burnishedGold = mix(pigments.sporeSulfur.rInf.mul(1.06), pigments.realgarAmber.rInf.mul(1.12), float(0.55));

  // Zone 1 (0.86..1.30): Vivid luminous Peacock-Turquoise -> Malachite-Emerald jewel corona
  const coronaCool = mix(peacockTurquoise, pigments.celadonGlauconite.rInf.mul(0.95), float(0.24));
  const coronaWarm = mix(peacockEmerald, peacockTurquoise, float(0.34));
  const coronaColor = mix(coronaCool, coronaWarm, smoothstep(float(0.18), float(0.88), gonio01));

  // Zone 2 (1.30..2.02): Rich Sapphire -> Emerald -> Copper-Bronze Ocellus Shield
  const shieldCool = peacockSapphire;
  const shieldMid = mix(peacockEmerald, burnishedGold, float(0.22));
  const shieldWarm = copperAmber;
  const shieldColor = mix(
    mix(shieldCool, shieldMid, smoothstep(float(0.10), float(0.46), gonio01)),
    shieldWarm,
    smoothstep(float(0.56), float(0.92), gonio01)
  );

  // Zone 3 (2.02..2.65): Outer Sapphire-Emerald-Rose Copper Rim
  const borderCool = mix(peacockSapphire, peacockTurquoise, float(0.45));
  const borderWarm = mix(peacockEmerald, copperAmber, float(0.66));
  const borderColor = mix(borderCool, borderWarm, smoothstep(float(0.18), float(0.86), gonio01));

  // Lower Filigree Herls & Outer Halo Barbs: Deep Cobalt-Sapphire -> Jade Emerald -> Copper-Gold
  const herlCool = mix(pigments.lapisUltramarine.rInf.mul(1.25), pigments.azuritePrussian.rInf.mul(1.68), float(0.70));
  const herlMid = mix(peacockEmerald, pigments.mossUmbrous.rInf.mul(1.15), float(0.18));
  const herlWarm = mix(pigments.yellowOchre.rInf.mul(1.08), copperAmber, float(0.58));
  const herlColor = mix(
    mix(herlCool, herlMid, smoothstep(float(0.15), float(0.49), gonio01)),
    herlWarm,
    smoothstep(float(0.56), float(0.92), gonio01)
  );

  const wCorona = smoothstep(float(0.76), float(0.90), rAnatomical);
  const wShield = smoothstep(float(1.18), float(1.34), rAnatomical);
  const wBorder = smoothstep(float(1.88), float(2.08), rAnatomical);
  const wOuterHerl = smoothstep(float(2.40), float(2.85), rAnatomical);

  let ocellusRgb = mix(heartColor, coronaColor, wCorona);
  ocellusRgb = mix(ocellusRgb, shieldColor, wShield);
  ocellusRgb = mix(ocellusRgb, borderColor, wBorder);
  ocellusRgb = mix(ocellusRgb, herlColor, wOuterHerl);

  const tipTint = mix(herlColor, burnishedGold, uBarb.mul(0.24).mul(gonio01));
  const baseMineralRgb = mix(tipTint, ocellusRgb, isOcellus);

  // Darken barb root junction softly into rawUmber so shaft edges read clean
  const rootDarkened = mix(pigments.rawUmber.rInf.mul(0.44), baseMineralRgb, rootBlend);

  const filmDepth = float(0.38).add(isOcellus.mul(0.16)).sub(uBarb.mul(0.10));
  const kmBody = kubelkaMunkGlaze(
    mix(pigments.azuritePrussian.K, pigments.verdigrisMalachite.K, gonio01),
    mix(pigments.azuritePrussian.S, pigments.verdigrisMalachite.S, gonio01),
    filmDepth,
    rootDarkened
  );

  const finalBase = mix(rootDarkened, kmBody, float(0.12)).mul(strandShade);
  mat.colorNode = clamp(finalBase, 0.0, 1.0);

  // Protect deep indigo eye heart (r < 0.80) from gold/amber washout so it pops like velvet ink!
  const outsideHeart = smoothstep(float(0.72), float(1.05), rAnatomical).max(float(1.0).sub(isOcellus));
  const sparklePalette = mix(
    mix(peacockSapphire, peacockTurquoise, float(0.55)),
    mix(peacockEmerald, burnishedGold, smoothstep(float(0.45), float(0.92), gonio01)),
    gonio01.mul(outsideHeart)
  );
  const iridescenceLift = sparklePalette
    .mul(
      anisoSheen.mul(0.42)
        .add(grazingSpec.mul(0.45).mul(float(0.20).add(outsideHeart.mul(0.80))))
        .add(fresnel.mul(0.12))
        .add(shimmerGlint.mul(0.75))
    )
    .mul(shared.uBeamIntensity)
    .mul(float(0.42).add(isOcellus.mul(0.48)))
    .mul(float(0.08).add(outsideHeart.mul(0.92)))
    .mul(rootBlend);

  // Subtle vibrant corona halo + zero gray glaze inside shadows
  const eyeJewelGlow = mix(heartColor.mul(0.35), coronaColor, wCorona)
    .mul(float(1.0).sub(wShield))
    .mul(isOcellus)
    .mul(0.22);

  const wakeFlash = sparklePalette.mul(wake.mul(wakeStrength).mul(0.22));
  mat.emissiveNode = clamp(iridescenceLift.add(eyeJewelGlow).add(wakeFlash), 0.0, 0.48);

  mat.roughnessNode = clamp(
    float(0.26)
      .sub(isOcellus.mul(0.06))
      .sub(gonio01.mul(0.05))
      .add(uBarb.mul(0.08))
      .add(ciliaWave.sub(0.5).mul(wingOnly).mul(0.06)),
    0.12,
    0.62
  );
  mat.metalnessNode = clamp(float(0.30).add(isOcellus.mul(0.18)).add(gonio01.mul(0.10)), 0.22, 0.56);

  return mat;
}
