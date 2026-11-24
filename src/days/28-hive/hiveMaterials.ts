import * as THREE from 'three/webgpu';
import {
  TWO_PI,
  abs,
  attribute,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  length,
  mix,
  normalLocal,
  normalView,
  normalize,
  positionLocal,
  positionViewDirection,
  pow,
  select,
  sin,
  smoothstep,
  transformNormalToView,
  uv,
  vec3,
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';

export interface HiveStudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface HiveMaterialsBundle {
  readonly branchMat: THREE.MeshStandardNodeMaterial;
  readonly innerCurtainMat: THREE.MeshStandardNodeMaterial;
  readonly thoraxMat: THREE.MeshStandardNodeMaterial;
  readonly abdomenMat: THREE.MeshStandardNodeMaterial;
  readonly wingMat: THREE.MeshStandardNodeMaterial;
  dispose: () => void;
}

/**
 * Generates a smooth Gaussian-feathered softbox luminance texture so PMREM reflections
 * glide smoothly across curved insect chitin, gossamer wings, and dark-walnut bark.
 */
function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number,
  size = 96
): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  for (let py = 0; py < size; py++) {
    const ny = (py / (size - 1)) * 2 - 1;
    for (let px = 0; px < size; px++) {
      const nx = (px / (size - 1)) * 2 - 1;
      const dist = Math.hypot(nx, ny);
      const falloff =
        Math.exp(-dist * dist * 2.1) * Math.max(0, 1 - Math.pow(Math.min(1, dist), 4));
      const idx = (py * size + px) * 4;
      data[idx] = Math.min(255, Math.round(r * falloff * 255));
      data[idx + 1] = Math.min(255, Math.round(g * falloff * 255));
      data[idx + 2] = Math.min(255, Math.round(b * falloff * 255));
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
 * Builds a broadly diffused natural-history studio softbox PMREM cubemap calibrated
 * so flipped golden-ochre tergites, translucent gossamer wings, and gnarled charred-oak
 * bark catch crisp studio highlights against Chalky Slate (#212529).
 */
export function createHiveStudioEnvironment(
  renderer: THREE.WebGPURenderer
): HiveStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0f1318);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: [number, number, number],
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

  // 1. Broad warm overhead-front-left calcite softbox (catches upturned abdomens & wing flicks)
  addSoftbox(11.5, 9.0, [-3.6, 6.2, 4.4], 1.0, 0.96, 0.88, 1.85);
  // 2. Tall right warm-neutral studio scrim (sculpts the curved teardrop flank & bark relief)
  addSoftbox(9.2, 10.2, [5.8, 2.5, 3.6], 0.98, 0.95, 0.90, 1.32);
  // 3. Rear-top cool-calcite grazing rim strip (carves crisp silhouette rim against #212529)
  addSoftbox(12.2, 5.4, [-1.2, 5.2, -5.8], 0.92, 0.96, 1.0, 1.62);
  // 4. Rear-right fill scrim so 360-deg back-face orbit stays equally dimensional
  addSoftbox(9.0, 7.5, [4.2, 1.8, -5.4], 0.96, 0.94, 0.90, 1.12);
  // 5. Subtle warm bounce floor lifting the curved pendant underside
  addSoftbox(10.5, 10.5, [0.0, -6.2, 0.6], 0.96, 0.89, 0.78, 0.58);

  const renderTarget = pmrem.fromScene(envScene, 0.082);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

/**
 * Builds all TSL PBR NodeMaterials for Day 28 - Hive strictly using physical mineral
 * and botanical pigments (`rawUmber`, `burntSienna`, `yellowOchre`, `realgarAmber`,
 * `sporeSulfur`, `boneCalcite`, `warmKaolin`, `mossUmbrous`).
 */
export function createHiveMaterials(): HiveMaterialsBundle {
  const umber = createPigmentUniforms('rawUmber');
  const sienna = createPigmentUniforms('burntSienna');
  const ochre = createPigmentUniforms('yellowOchre');
  const amber = createPigmentUniforms('realgarAmber');
  const sulfur = createPigmentUniforms('sporeSulfur');
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const moss = createPigmentUniforms('mossUmbrous');

  // ---------------------------------------------------------------------------
  // 1. Gnarled Dark-Walnut / Charred-Oak Suspended Timber Branch
  // ---------------------------------------------------------------------------
  const branchMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.78,
    metalness: 0.02,
  });

  const bPos = positionLocal;
  const bUv = uv();
  const aBarkData = attribute('aBarkData', 'vec4'); // [furrow01, burl01, endCut01, moss01]
  const furrow01 = aBarkData.x;
  const burl01 = aBarkData.y;
  const endCut01 = aBarkData.z;
  const mossMask = aBarkData.w;

  // Longitudinal charred-oak bark grain + fine cross-cracks in 3D object space (zero UV seams)
  const barkCoarse = mineralToothFbm(
    vec3(bPos.x.mul(3.4), bPos.y.mul(12.0), bPos.z.mul(12.0)),
    float(2.1)
  );
  const barkFine = mineralToothFbm(
    vec3(bPos.x.mul(11.0), bPos.y.mul(28.0), bPos.z.mul(28.0)),
    float(2.4)
  );
  const crackPattern = smoothstep(float(0.28), float(0.76), barkCoarse.mul(0.58).add(furrow01.mul(0.42)));

  // Deep charred-walnut crevices -> weathered warm umber ridges -> subtle silvered bone highlight
  const creviceBark = umber.rInf.mul(0.16);
  const ridgeBark = mix(
    umber.rInf.mul(0.56),
    mix(sienna.rInf.mul(0.44), kaolin.rInf.mul(0.38), float(0.22)),
    clamp(burl01.mul(0.5).add(barkFine.mul(0.45)), 0.0, 1.0)
  );
  const weatheredBark = mix(creviceBark, ridgeBark, crackPattern);

  // Subtle dry bryophyte moss flecks nestled into upper bark hollows
  const mossNoise = mineralToothFbm(bPos.mul(16.0), float(2.2));
  const mossAmt = clamp(mossMask.mul(smoothstep(float(0.36), float(0.74), mossNoise)).mul(0.85), 0.0, 0.72);
  const mossTint = mix(moss.rInf.mul(0.62), ochre.rInf.mul(0.48), barkFine.mul(0.35));
  const barkWithMoss = mix(weatheredBark, mossTint, mossAmt);

  // Oblique broken heartwood end-grain with concentric growth rings
  const ringDist = length(bUv.sub(0.5));
  const growthRings = sin(ringDist.mul(62.0).add(barkCoarse.mul(5.0))).mul(0.5).add(0.5);
  const heartwood = mix(
    umber.rInf.mul(0.65),
    kubelkaMunkGlaze(
      ochre.K,
      ochre.S,
      float(0.24).add(growthRings.mul(0.28)),
      kaolin.rInf.mul(0.88)
    ),
    float(0.68)
  );

  branchMat.colorNode = mix(barkWithMoss, heartwood, smoothstep(float(0.4), float(0.85), endCut01));
  branchMat.roughnessNode = clamp(
    float(0.88)
      .sub(crackPattern.mul(0.18))
      .sub(barkFine.mul(0.12))
      .add(mossAmt.mul(0.10))
      .sub(endCut01.mul(0.14)),
    0.52,
    0.95
  );

  // Crisp bark micro-normal perturbation (view space via transformNormalToView)
  const bN0 = normalize(normalLocal);
  const bRef = select(abs(bN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const bTanA = normalize(cross(bRef, bN0));
  const bTanB = normalize(cross(bN0, bTanA));
  const barkSlopeA = barkFine.sub(0.5).mul(0.36).add(furrow01.sub(0.5).mul(0.22));
  const barkSlopeB = barkCoarse.sub(0.5).mul(0.24);
  branchMat.normalNode = transformNormalToView(
    normalize(bN0.add(bTanA.mul(barkSlopeA)).add(bTanB.mul(barkSlopeB)))
  );

  // ---------------------------------------------------------------------------
  // 2. Dark Organic Inner Brood-Curtain Mantle (recessed warm shadow behind bees)
  // ---------------------------------------------------------------------------
  const innerCurtainMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.86,
    metalness: 0.01,
  });

  const cPos = positionLocal;
  const combMicro = mineralToothFbm(cPos.mul(18.0), float(2.3));
  const deepNest = mix(umber.rInf.mul(0.18), sienna.rInf.mul(0.24), combMicro.mul(0.55));
  const waxHint = mix(deepNest, ochre.rInf.mul(0.28), smoothstep(float(0.58), float(0.88), combMicro).mul(0.32));
  innerCurtainMat.colorNode = waxHint;
  innerCurtainMat.roughnessNode = clamp(float(0.82).add(combMicro.mul(0.14)), 0.72, 0.96);

  // ---------------------------------------------------------------------------
  // 3. Anchored *Apis dorsata* Head + Velvety Pubescent Thorax (InstancedMesh)
  // ---------------------------------------------------------------------------
  const thoraxMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.72,
    metalness: 0.04,
  });

  const aThoraxData = attribute('aThoraxData', 'vec4'); // [alongY01, dorsal01, isHead01, eyeFlank01]
  const aBeeStateT = attribute('aBeeState', 'vec4'); // [flip01, flashDrive, wingSpread, seed]
  const tDorsal = aThoraxData.y;
  const tIsHead = aThoraxData.z;
  const tEyeFlank = aThoraxData.w;
  const tFlash = aBeeStateT.y;
  const tSeed = aBeeStateT.w;

  const tNView = normalize(normalView);
  const tVDir = normalize(positionViewDirection);
  const tFresnel = pow(clamp(float(1.0).sub(abs(dot(tNView, tVDir))), 1e-4, 1.0), float(2.2));

  // Velvety dark raw-umber thoracic pubescence with golden-ochre hair rim + dark glossy chitin head
  const darkChitin = umber.rInf.mul(0.14);
  const velvetThorax = mix(
    umber.rInf.mul(0.34),
    mix(ochre.rInf.mul(0.52), sienna.rInf.mul(0.45), float(0.42)),
    clamp(tDorsal.mul(0.58).add(tFresnel.mul(0.42)).add(tSeed.sub(0.5).mul(0.14)), 0.0, 1.0)
  );
  const thoraxBase = mix(velvetThorax, darkChitin, smoothstep(float(0.32), float(0.78), tIsHead));
  // Subtle warm setae lift as the defense wave ripples through the thorax
  const setaeAlert = mix(ochre.rInf, amber.rInf, float(0.35)).mul(tFlash.mul(0.24).mul(float(1.0).sub(tIsHead.mul(0.6))));
  thoraxMat.colorNode = clamp(thoraxBase.add(setaeAlert), 0.0, 1.0);
  thoraxMat.roughnessNode = clamp(
    mix(float(0.76).sub(tFlash.mul(0.12)), float(0.28).sub(tEyeFlank.mul(0.08)), tIsHead),
    0.18,
    0.88
  );

  // ---------------------------------------------------------------------------
  // 4. Articulated Hinged *Apis dorsata* Segmented Abdomen (InstancedMesh)
  // ---------------------------------------------------------------------------
  const abdomenMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.54,
    metalness: 0.06,
  });

  // aAbdomenUv: [alongFromPetiole01, dorsal01, tergiteRidge01, sterniteRim01]
  const aAbdUv = attribute('aAbdomenUv', 'vec4');
  // aBeeState: [flip01, flashDrive, wingSpread, seed]
  const aBeeState = attribute('aBeeState', 'vec4');

  const alongT = aAbdUv.x; // 0 = petiole base (golden T1), 1 = apical tip (dark T6)
  const dorsal01 = aAbdUv.y; // 1 = dorsal back, 0 = ventral sternites
  const ridge01 = aAbdUv.z;
  const sterniteRim = aAbdUv.w;

  const flip01 = clamp(aBeeState.x, 0.0, 1.0);
  const flashDrive = clamp(aBeeState.y, 0.0, 1.0);
  const beeSeed = aBeeState.w;

  const aNView = normalize(normalView);
  const aVDir = normalize(positionViewDirection);
  const aFresnel = pow(clamp(float(1.0).sub(abs(dot(aNView, aVDir))), 1e-4, 1.0), float(2.4));

  // Crisp anatomical 5-tergite banding of giant honeybee Apis dorsata:
  // - Basal tergites T1-T3 (alongT 0.05..0.58): vivid honey-gold Yellow Ochre & Realgar-Amber
  // - Sharp dark Raw-Umber inter-tergite suture bands between segments
  // - Apical tergites T4-T6 (alongT 0.60..1.0): deep velvety Raw Umber & Burnt Sienna
  const segPhase = alongT.mul(5.2).add(beeSeed.mul(0.16));
  const segWave = sin(segPhase.mul(TWO_PI)).mul(0.5).add(0.5);
  const sutureStripe = smoothstep(float(0.12), float(0.46), segWave);
  const basalZone = float(1.0).sub(smoothstep(float(0.44), float(0.68), alongT));
  const paleHairFringe = smoothstep(float(0.76), float(0.96), segWave).mul(float(1.0).sub(smoothstep(float(0.75), float(0.96), alongT)));

  // 2-flux Kubelka-Munk mineral pigment glazing (Yellow Ochre + Realgar-Amber over Bone Calcite)
  const honeyOchreGlaze = kubelkaMunkGlaze(
    mix(ochre.K, amber.K, float(0.32).add(beeSeed.mul(0.22))),
    mix(ochre.S, amber.S, float(0.32).add(beeSeed.mul(0.22))),
    float(0.48).add(ridge01.mul(0.24)),
    calcite.rInf
  );

  // Bright ventral/inter-tergal flash tone (Spore Sulfur + Yellow Ochre + Realgar-Amber) revealed on flip!
  const ventralGold = mix(
    mix(ochre.rInf.mul(1.18), sulfur.rInf.mul(1.15), float(0.48)),
    amber.rInf.mul(1.12),
    float(0.22).add(dorsal01.mul(0.24))
  );

  // Dark velvety umber tergite sutures & apical stinger tip
  const darkUmberBand = mix(umber.rInf.mul(0.20), sienna.rInf.mul(0.22), ridge01.mul(0.28));

  // Combine dorsal striped pattern
  const bandStrength = clamp(basalZone.mul(0.92).add(paleHairFringe.mul(0.42)), 0.0, 1.0);
  const dorsalStriped = mix(darkUmberBand, honeyOchreGlaze, clamp(bandStrength.mul(sutureStripe).add(basalZone.mul(0.24)), 0.0, 1.0));

  // Blend ventral pale-gold sternites & pleural rim (dramatically exposed as abdomen tilts 65-75 deg!)
  const ventralWeight = clamp(
    float(1.0).sub(dorsal01).mul(0.72).add(sterniteRim.mul(flip01).mul(0.48)),
    0.0,
    1.0
  );
  const bodyColor = mix(dorsalStriped, ventralGold, ventralWeight);

  // At rest: bees overlap tightly like shingles -> muted, quiet deep Raw Umber & dark banded Ochre.
  // On wave flip: shingle occlusion lifts away & bright golden tergites snap into studio light!
  const restShingleOcclusion = mix(float(0.44), float(0.84), smoothstep(float(0.18), float(0.82), alongT));
  const waveLift = mix(restShingleOcclusion, float(1.18), smoothstep(float(0.05), float(0.85), flip01));
  const perBeeVar = float(0.90).add(beeSeed.mul(0.20));

  const restToned = mix(umber.rInf.mul(0.32), bodyColor, float(0.66).add(flip01.mul(0.34)));
  const finalAbdColor = clamp(restToned.mul(waveLift).mul(perBeeVar), 0.0, 1.0);

  abdomenMat.colorNode = finalAbdColor;

  // Chitin shifts from matte resting pubescence (0.62) to burnished specular golden carapace (0.26) during flip
  abdomenMat.roughnessNode = clamp(
    float(0.64)
      .sub(flip01.mul(0.32))
      .sub(ridge01.mul(0.08))
      .sub(sutureStripe.mul(0.06)),
    0.22,
    0.76
  );
  abdomenMat.metalnessNode = clamp(float(0.04).add(flip01.mul(0.14)).add(paleHairFringe.mul(0.04)), 0.03, 0.22);

  // Subtle physical mineral shimmer scatter along flipped golden tergites & sunlit rims (zero neon!)
  const shimmerPalette = mix(
    mix(ochre.rInf, amber.rInf, float(0.36)),
    mix(sulfur.rInf, calcite.rInf, float(0.28)),
    clamp(aFresnel.mul(0.65).add(basalZone.mul(0.35)), 0.0, 1.0)
  );
  const flipScatter = shimmerPalette
    .mul(flashDrive)
    .mul(float(0.14).add(aFresnel.mul(0.24)).add(basalZone.mul(0.12)))
    .mul(float(0.75).add(sutureStripe.mul(0.25)));
  abdomenMat.emissiveNode = clamp(flipScatter, 0.0, 0.34);

  // Subtle tergite micro-step normal perturbation (view space via transformNormalToView)
  const aN0 = normalize(normalLocal);
  const aRef = select(abs(aN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const aTanA = normalize(cross(aRef, aN0));
  const aTanB = normalize(cross(aN0, aTanA));
  const tergiteSlope = cos(segPhase.mul(TWO_PI)).mul(0.14);
  abdomenMat.normalNode = transformNormalToView(
    normalize(aN0.add(aTanB.mul(tergiteSlope)).add(aTanA.mul(ridge01.sub(0.5).mul(0.06))))
  );

  // ---------------------------------------------------------------------------
  // 5. Paired Translucent Gossamer Forewings (InstancedMesh, DoubleSide)
  // ---------------------------------------------------------------------------
  const wingMat = new THREE.MeshStandardNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    roughness: 0.14,
    metalness: 0.08,
  });

  // aWingUv: [uSpan (0=root..1=tip), vChord (-1=leading costal..+1=trailing), sideSign, camber01]
  const aWingUv = attribute('aWingUv', 'vec4');
  const aBeeStateW = attribute('aBeeState', 'vec4'); // [flip01, flashDrive, wingSpread, seed]

  const uSpan = aWingUv.x;
  const vChord = aWingUv.y;
  const wFlip = clamp(aBeeStateW.x, 0.0, 1.0);
  const wFlash = clamp(aBeeStateW.y, 0.0, 1.0);
  const wSpread = clamp(aBeeStateW.z, 0.0, 1.0);
  const wSeed = aBeeStateW.w;

  const wNView = normalize(normalView);
  const wVDir = normalize(positionViewDirection);
  const wNdotV = abs(dot(wNView, wVDir));
  const wFresnel = pow(clamp(float(1.0).sub(wNdotV), 1e-4, 1.0), float(2.1));

  // Anatomical hymenopteran wing venation:
  // - Thickened dark amber-umber leading costal margin (vChord near -1)
  // - Pterostigma dark spot mid-leading edge (u ~ 0.52, v ~ -0.78)
  // - Fine longitudinal & cross radial veins across translucent membrane
  const costalMargin = float(1.0).sub(smoothstep(float(-0.98), float(-0.62), vChord));
  const stigmaU = uSpan.sub(0.52);
  const stigmaV = vChord.add(0.76);
  const pterostigma = exp(stigmaU.mul(stigmaU).mul(95.0).add(stigmaV.mul(stigmaV).mul(42.0)).negate());
  const veinLong = pow(clamp(abs(sin(vChord.mul(7.5).add(uSpan.mul(4.2)))), 1e-4, 1.0), float(10.0));
  const veinCross = pow(clamp(abs(sin(uSpan.mul(14.0).sub(vChord.mul(3.0)))), 1e-4, 1.0), float(14.0));
  const veinMask = clamp(costalMargin.mul(0.85).add(pterostigma.mul(0.75)).add(veinLong.mul(0.28)).add(veinCross.mul(0.20)), 0.0, 1.0);

  // Beer-Lambert smoked-amber chitin membrane transmittance
  const wingThickness = float(0.18).add(veinMask.mul(0.45)).add(float(1.0).sub(uSpan).mul(0.12));
  const smokedTransmit = beerLambertTransmittance(umber.extinction, wingThickness);
  const membraneTint = mix(
    mix(kaolin.rInf, ochre.rInf, float(0.38)),
    mix(umber.rInf.mul(0.52), sienna.rInf.mul(0.58), float(0.35)),
    clamp(veinMask.mul(0.78).add(float(1.0).sub(smokedTransmit.y).mul(0.22)), 0.0, 1.0)
  );

  // Subtle sunlit specular sheen on translucent wing film when flicking open in a defense wave
  const wingSheenTint = mix(
    calcite.rInf,
    mix(ochre.rInf, amber.rInf, float(0.45)),
    clamp(float(0.35).add(wSeed.mul(0.25)).add(uSpan.mul(0.25)), 0.0, 1.0)
  );
  const wingGlint = clamp(
    wFresnel.mul(0.55).add(wSpread.mul(0.55)).add(wFlash.mul(0.45)),
    0.0,
    1.25
  );

  wingMat.colorNode = mix(membraneTint, wingSheenTint, clamp(wSpread.mul(0.45).add(wFresnel.mul(0.35)), 0.0, 0.85));
  wingMat.emissiveNode = clamp(
    wingSheenTint.mul(wingGlint).mul(float(0.06).add(wSpread.mul(0.26)).add(wFlip.mul(0.14))),
    0.0,
    0.42
  );
  wingMat.roughnessNode = clamp(float(0.18).add(veinMask.mul(0.26)).sub(wSpread.mul(0.06)), 0.08, 0.48);

  // Softly feathered wingtip & trailing edge so wings read as delicate gossamer rather than hard polygons
  const tipFade = float(1.0).sub(smoothstep(float(0.86), float(1.0), uSpan));
  const trailFade = float(1.0).sub(smoothstep(float(0.78), float(1.0), vChord));
  const baseAlpha = float(0.20)
    .add(veinMask.mul(0.34))
    .add(wFresnel.mul(0.28))
    .add(wSpread.mul(0.18));
  wingMat.opacityNode = clamp(baseAlpha.mul(tipFade).mul(trailFade), 0.0, 0.86);

  return {
    branchMat,
    innerCurtainMat,
    thoraxMat,
    abdomenMat,
    wingMat,
    dispose: () => {
      branchMat.dispose();
      innerCurtainMat.dispose();
      thoraxMat.dispose();
      abdomenMat.dispose();
      wingMat.dispose();
    },
  };
}
