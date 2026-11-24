import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  clamp,
  cos,
  cross,
  dot,
  float,
  fract,
  max,
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
  sqrt,
  transformNormalToView,
  uniform,
  vec3,
} from 'three/tsl';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import type { CeladonEggSpec } from './reciprocalDome';

/**
 * Generates a smooth Gaussian-feathered studio softbox texture for PMREM reflections.
 */
function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number,
  size = 128
): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  for (let py = 0; py < size; py++) {
    const ny = (py / (size - 1)) * 2 - 1;
    for (let px = 0; px < size; px++) {
      const nx = (px / (size - 1)) * 2 - 1;
      const dist = Math.hypot(nx, ny);
      const falloff =
        Math.exp(-dist * dist * 2.15) * Math.max(0, 1 - Math.pow(Math.min(1, dist), 4));
      const idx = (py * size + px) * 4;
      data[idx] = Math.min(255, Math.round(r * falloff * 255));
      data[idx + 1] = Math.min(255, Math.round(g * falloff * 255));
      data[idx + 2] = Math.min(255, Math.round(b * falloff * 255));
      data[idx + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return tex;
}

export interface NestStudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

/**
 * Builds a studio softbox PMREM cubemap calibrated for hand-rubbed oiled hardwoods,
 * satin dark architectural brass (zero black-void collapse, crisp bevel highlights),
 * matte honed travertine limestone, and satin Celadon river-pebble eggs on #212529.
 */
export function createNestStudioEnvironment(
  renderer: THREE.WebGPURenderer
): NestStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Lifted warm-slate studio enclosure so metallic dark brass never collapses to dead black
  envScene.background = new THREE.Color(0x161a20);

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

  // Warm overhead-front-right museum softbox key
  addSoftbox(11.0, 8.5, [4.6, 6.4, 4.6], 1.0, 0.96, 0.90, 1.42);
  // Cool mineral-slate left-flank studio scrim for crisp dark-brass & walnut bevel separation
  addSoftbox(9.5, 9.0, [-5.8, 2.8, 3.2], 0.85, 0.91, 1.0, 0.96);
  // Grazing back-top calcite rim scrim separating the woven dome silhouette from #212529
  addSoftbox(12.0, 5.4, [-1.2, 5.2, -5.8], 0.98, 0.95, 0.89, 1.32);
  // Subtle horizon wrap strip so vertical brass outer walls always catch warm architectural sheen
  addSoftbox(12.0, 3.2, [5.6, 0.8, -3.5], 0.96, 0.90, 0.82, 0.72);
  // Gentle warm studio uplift underneath so orbiting below the hollow plinth stays crisp
  addSoftbox(10.0, 10.0, [0.4, -6.2, 0.8], 0.95, 0.88, 0.78, 0.54);

  const renderTarget = pmrem.fromScene(envScene, 0.06);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

function createNestUniforms() {
  return {
    uHoveredRodId: uniform(-1.0),
    uGrabbedRodId: uniform(-1.0),
    uHoverLift: uniform(0.0),
  };
}

export type NestUniforms = ReturnType<typeof createNestUniforms>;

export interface NestMaterialsBundle {
  readonly uHoveredRodId: NestUniforms['uHoveredRodId'];
  readonly uGrabbedRodId: NestUniforms['uGrabbedRodId'];
  readonly uHoverLift: NestUniforms['uHoverLift'];
  readonly timberMat: THREE.MeshStandardNodeMaterial;
  readonly limestoneMat: THREE.MeshStandardNodeMaterial;
  readonly brassMat: THREE.MeshStandardNodeMaterial;
  readonly eggMats: THREE.MeshStandardNodeMaterial[];
  dispose: () => void;
}

/**
 * Creates all TSL PBR NodeMaterials for Day 24 - Nest using strictly physical
 * mineral pigments from `src/core/palette.ts` via 2-flux Kubelka-Munk glazing.
 */
export function createNestMaterials(eggs: readonly CeladonEggSpec[]): NestMaterialsBundle {
  const umber = createPigmentUniforms('rawUmber');
  const sienna = createPigmentUniforms('burntSienna');
  const ochre = createPigmentUniforms('yellowOchre');
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const celadon = createPigmentUniforms('celadonGlauconite');
  const moss = createPigmentUniforms('mossUmbrous');

  const { uHoveredRodId, uGrabbedRodId, uHoverLift } = createNestUniforms();

  // ---------------------------------------------------------------------------
  // 1. Turned Timber Dowels — 3D Solid-Wood Longitudinal Grain + Concentric End-Grain
  // ---------------------------------------------------------------------------
  const timberMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.62,
    metalness: 0.02,
  });

  // Per-instance attribute: x=woodSpecies [0..1], y=grainSeed [0..1], z=rodId, w=tactileFlash [0..1]
  const rodMeta = attribute('aRodMeta', 'vec4');
  const woodSpecies = rodMeta.x;
  const grainSeed = rodMeta.y;
  const rodId = rodMeta.z;
  const tactileFlash = rodMeta.w;
  const notchMask = attribute('aNotchMask', 'float');

  // Strictly dowel-local 3D coordinates (local Y = longitudinal shaft axis, XZ = circular cross-section)
  // Guarantees 0 UV seams AND mathematically continuous shaft cathedral grain <-> concentric end-grain rings!
  const pLoc = positionLocal;
  const wN0 = normalize(normalLocal);
  const absNy = abs(wN0.y);
  // 1.0 on flat cut circular end-grain faces, smooth transition over chamfer bevel, 0.0 on shaft barrel
  const endFaceMask = smoothstep(float(0.58), float(0.92), absNy);
  const bevelMask = smoothstep(float(0.18), float(0.75), absNy).mul(float(1.0).sub(endFaceMask));

  const seedAngle = grainSeed.mul(41.3);
  const seedShift = grainSeed.mul(23.7);

  // Virtual tree-log pith close to each dowel centerline (~1.8..4.6 mm) so every cut end
  // displays unmistakable concentric bullseye/eccentric growth rings while shafts show organic cathedral arches
  const pithRadius = float(0.0018).add(grainSeed.mul(0.0028));
  const pithX = cos(seedAngle).mul(pithRadius);
  const pithZ = sin(seedAngle).mul(pithRadius);

  // Gentle organic wood-fiber wander flowing slowly along longitudinal Y
  // (tighter warp on flat cut end-grain so concentric annual rings stay crisp & readable!)
  const warpCoordA = vec3(
    pLoc.x.mul(22.0).add(seedShift),
    pLoc.y.mul(2.1).add(grainSeed.mul(9.7)),
    pLoc.z.mul(22.0).sub(seedShift.mul(0.6))
  );
  const warpCoordB = vec3(
    pLoc.x.mul(20.0).sub(seedShift.mul(0.7)),
    pLoc.y.mul(2.6).add(grainSeed.mul(13.1)),
    pLoc.z.mul(20.0).add(seedShift)
  );
  const warpA = mineralToothFbm(warpCoordA, float(1.1)).sub(0.5);
  const warpB = mineralToothFbm(warpCoordB, float(1.1)).sub(0.5);
  const warpAmp = mix(float(0.0044), float(0.0017), endFaceMask);

  const rx = pLoc.x.sub(pithX).add(warpA.mul(warpAmp));
  const rz = pLoc.z.sub(pithZ).add(warpB.mul(warpAmp));
  const treeR = sqrt(max(float(1e-7), rx.mul(rx).add(rz.mul(rz))));

  // Natural non-uniform seasonal ring spacing (organic earlywood/latewood bunching — never uniform zebra stripes!)
  const growthMod = mineralToothFbm(
    vec3(rx.mul(55.0).add(seedShift), pLoc.y.mul(1.8), rz.mul(55.0)),
    float(1.2)
  );
  const ringFreq = mix(float(300.0), float(355.0), endFaceMask);
  const ringCoord = treeR.mul(ringFreq).add(growthMod.mul(2.6)).add(warpA.mul(1.1)).add(8.0);
  const ringFrac = fract(ringCoord);
  const latewoodBand = smoothstep(float(0.60), float(0.86), ringFrac).mul(
    float(1.0).sub(smoothstep(float(0.88), float(0.99), ringFrac))
  );
  const ringGrad = smoothstep(float(0.05), float(0.75), ringFrac);
  const ringContrast = clamp(latewoodBand.mul(0.72).add(ringGrad.mul(0.28)), 0.0, 1.0);

  // Tight anisotropic longitudinal fibers & fine open hardwood vascular pores along the dowel shaft
  const fineFiber = mineralToothFbm(
    vec3(
      pLoc.x.mul(185.0).add(seedShift),
      pLoc.y.mul(11.5).add(grainSeed.mul(17.3)),
      pLoc.z.mul(185.0).sub(seedShift)
    ),
    float(1.6)
  );
  const microVessels = mineralToothFbm(
    vec3(
      pLoc.x.mul(360.0).sub(seedShift),
      pLoc.y.mul(24.0).add(grainSeed.mul(7.1)),
      pLoc.z.mul(360.0).add(seedShift)
    ),
    float(1.45)
  );
  const shaftPores = smoothstep(float(0.65), float(0.85), microVessels);

  // Cross-cut open end-grain vessel stipple on flat dowel tips
  const endVesselNoise = mineralToothFbm(
    vec3(
      pLoc.x.mul(410.0).add(seedShift),
      pLoc.y.mul(18.0),
      pLoc.z.mul(410.0).sub(seedShift)
    ),
    float(1.6)
  );
  const endPores = smoothstep(float(0.60), float(0.82), endVesselNoise);

  // Richer fibrous shaft blend + crisp concentric rings on cut end-grain tips
  const grainPattern = mix(
    clamp(ringContrast.mul(0.46).add(fineFiber.mul(0.54)), 0.0, 1.0),
    clamp(latewoodBand.mul(0.84).add(ringGrad.mul(0.16)), 0.0, 1.0),
    endFaceMask
  );

  // Four natural mineral-pigment hardwood species (strictly from src/core/palette.ts):
  //   0.04 -> Deep Oiled Black Walnut (dark espresso-umber heartwood + warm sienna undertone)
  //   0.34 -> Warm Fumed Smoked Ash (rich warm taupe-chestnut ash, never cold grey PVC!)
  //   0.56..0.76 -> Aromatic Red-Honey Cedar to Warm Golden-Ochre Cedar
  //   0.96 -> Pale Creamy Satin Nordic Birch (light Bone-Calcite + Warm-Kaolin framing the eggs)
  const walnutEarly = mix(umber.rInf.mul(0.58), sienna.rInf.mul(0.38), float(0.26));
  const walnutLate = umber.rInf.mul(0.21);
  const walnutRgb = mix(walnutEarly, walnutLate, grainPattern.mul(0.76)).mul(
    float(1.0).sub(shaftPores.mul(0.24))
  );

  const ashBase = mix(umber.rInf.mul(0.86), sienna.rInf.mul(0.52), float(0.32));
  const ashEarly = mix(ashBase, ochre.rInf.mul(0.54), float(0.22));
  const ashLate = mix(umber.rInf.mul(0.36), sienna.rInf.mul(0.24), float(0.24));
  const smokedAshRgb = mix(ashEarly, ashLate, grainPattern.mul(0.72)).mul(
    float(1.0).sub(shaftPores.mul(0.26))
  );

  const cedarEarly = kubelkaMunkGlaze(
    ochre.K,
    ochre.S,
    clamp(float(0.30).add(fineFiber.mul(0.20)), 0.12, 0.68),
    mix(sienna.rInf.mul(0.68), kaolin.rInf.mul(0.56), float(0.34))
  );
  const cedarLate = mix(sienna.rInf.mul(0.48), umber.rInf.mul(0.46), float(0.44));
  const cedarRgb = mix(cedarEarly, cedarLate, grainPattern.mul(0.66)).mul(
    float(1.0).sub(shaftPores.mul(0.18))
  );

  const birchEarly = kubelkaMunkGlaze(
    ochre.K,
    ochre.S,
    clamp(float(0.045).add(fineFiber.mul(0.055)), 0.02, 0.14),
    mix(calcite.rInf.mul(0.86), kaolin.rInf.mul(0.80), float(0.50))
  );
  const birchLate = mix(kaolin.rInf.mul(0.56), umber.rInf.mul(0.74), float(0.42));
  const birchRgb = mix(birchEarly, birchLate, grainPattern.mul(0.54)).mul(
    float(1.0).sub(shaftPores.mul(0.11))
  );

  // Smooth per-tier species interpolation + natural per-dowel tone variance
  const tAsh = smoothstep(float(0.06), float(0.35), woodSpecies);
  const tCedar = smoothstep(float(0.36), float(0.72), woodSpecies);
  const tBirch = smoothstep(float(0.78), float(0.95), woodSpecies);

  const speciesRgb0 = mix(walnutRgb, smokedAshRgb, tAsh);
  const speciesRgb1 = mix(speciesRgb0, cedarRgb, tCedar);
  const speciesRgb = mix(speciesRgb1, birchRgb, tBirch);

  // Cross-cut end-grain naturally absorbs more finishing oil (richer dark tone + crisp concentric rings)
  // Burnished joinery saddle bands darken subtly where crossing rods seat under load
  const endOilDarken = endFaceMask.mul(float(0.26).add(endPores.mul(0.16))).add(bevelMask.mul(0.10));
  const notchDarken = notchMask.mul(0.26);
  const perRodTone = float(0.91).add(grainSeed.mul(0.18));
  const baseWoodRgb = speciesRgb
    .mul(perRodTone)
    .mul(clamp(float(1.0).sub(endOilDarken).sub(notchDarken), 0.42, 1.0));

  // Warm subtle hover/grab grazing rim highlight — STRICTLY ZERO tint on unhovered dowels!
  const hasHoverTarget = smoothstep(float(-0.5), float(-0.05), uHoveredRodId);
  const hasGrabTarget = smoothstep(float(-0.5), float(-0.05), uGrabbedRodId);
  const isHovered = float(1.0)
    .sub(smoothstep(float(0.10), float(0.45), abs(rodId.sub(uHoveredRodId))))
    .mul(hasHoverTarget);
  const isGrabbed = float(1.0)
    .sub(smoothstep(float(0.10), float(0.45), abs(rodId.sub(uGrabbedRodId))))
    .mul(hasGrabTarget);
  const hoverAmt = isHovered.mul(uHoverLift);

  const nView = normalize(normalView);
  const vDir = normalize(positionViewDirection);
  const fresnelRim = pow(clamp(float(1.0).sub(abs(dot(nView, vDir))), 1e-4, 1.0), float(2.8));

  // Keep wood grain 100% rich and legible during hover (warm golden-calcite rim, never chalk-white!)
  const warmRimTint = mix(ochre.rInf, calcite.rInf, float(0.42));
  const highlightMix = clamp(
    hoverAmt.mul(0.065).add(isGrabbed.mul(0.11)).add(tactileFlash.mul(0.08)),
    0.0,
    0.18
  );

  timberMat.colorNode = mix(baseWoodRgb, warmRimTint, highlightMix);
  timberMat.emissiveNode = warmRimTint.mul(
    hoverAmt.mul(float(0.012).add(fresnelRim.mul(0.16)))
      .add(isGrabbed.mul(float(0.028).add(fresnelRim.mul(0.24))))
      .add(tactileFlash.mul(float(0.03).add(fresnelRim.mul(0.12))))
  );

  // Hand-rubbed satin-matte oiled hardwood roughness (never glossy plastic!)
  const activePores = mix(shaftPores, endPores, endFaceMask);
  timberMat.roughnessNode = clamp(
    float(0.54)
      .add(fineFiber.mul(0.12))
      .add(activePores.mul(0.18))
      .add(endFaceMask.mul(0.16))
      .sub(latewoodBand.mul(0.05))
      .sub(notchMask.mul(0.08))
      .sub(isGrabbed.mul(0.04)),
    0.46,
    0.88
  );

  // Singularity-free tangent-space micro-normal relief:
  //   - Shaft: crisp circumferential slope across longitudinal wood fibers & sunken vessels
  //   - End-grain cut face: radial concentric growth-ring ridge & open-vessel stipple relief
  const wRef = select(absNy.lessThan(0.85), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const wTanA = normalize(cross(wRef, wN0));
  const wTanB = normalize(cross(wN0, wTanA));
  const radialEndDir = normalize(vec3(rx, float(0.0), rz).add(vec3(1e-5, 0.0, 0.0)));

  const shaftSlopeA = fineFiber.sub(0.5).mul(0.095).add(latewoodBand.sub(0.5).mul(0.045)).sub(shaftPores.mul(0.08));
  const shaftSlopeB = microVessels.sub(0.5).mul(0.032);
  const shaftNormal = normalize(wN0.add(wTanA.mul(shaftSlopeA)).add(wTanB.mul(shaftSlopeB)));

  const endRingSlope = cos(ringFrac.mul(6.283185)).mul(0.068).add(endPores.sub(0.5).mul(0.065));
  const endNormal = normalize(wN0.add(radialEndDir.mul(endRingSlope)));

  timberMat.normalNode = transformNormalToView(
    normalize(mix(shaftNormal, endNormal, endFaceMask))
  );

  // ---------------------------------------------------------------------------
  // 2. Honed Travertine / Jura Limestone Plinth Ring (Warm Mineral Relief, Zero Blowout)
  // ---------------------------------------------------------------------------
  const limestoneMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.78,
    metalness: 0.01,
  });

  const sPos = positionLocal;
  const sRad = sqrt(max(float(1e-6), sPos.x.mul(sPos.x).add(sPos.z.mul(sPos.z))));

  // Sedimentary bedding layers warped smoothly across the carved limestone ring
  const bedWarp = mineralToothFbm(sPos.mul(3.4), float(1.4)).sub(0.5);
  const strata = mineralToothFbm(
    vec3(sPos.x.mul(2.9).add(bedWarp.mul(0.8)), sPos.y.mul(18.0).add(bedWarp.mul(1.6)), sPos.z.mul(2.9)),
    float(1.65)
  );
  const ooliteGrain = mineralToothFbm(sPos.mul(26.0), float(2.1));
  const travertinePits = mineralToothFbm(
    vec3(sPos.x.mul(44.0), sPos.y.mul(68.0), sPos.z.mul(44.0)),
    float(2.2)
  );

  const veinBand = smoothstep(float(0.44), float(0.74), strata);
  const pitMask = smoothstep(float(0.68), float(0.86), travertinePits);
  const fossilFleck = smoothstep(float(0.74), float(0.88), ooliteGrain);

  // Calibrated pale honed Jura / travertine limestone (crisp stone contrast against dark walnut & brass, zero white blowout!)
  const stoneBacking = mix(kaolin.rInf.mul(0.78), calcite.rInf.mul(0.74), float(0.52));
  const stoneGlaze = kubelkaMunkGlaze(
    umber.K,
    umber.S,
    clamp(float(0.035).add(strata.mul(0.065)), 0.02, 0.14),
    stoneBacking
  );
  const veinedStone = mix(stoneGlaze, umber.rInf.mul(0.78), veinBand.mul(0.24));
  const pittedStone = mix(veinedStone, umber.rInf.mul(0.46), pitMask.mul(0.38));
  const detailedStone = mix(pittedStone, calcite.rInf.mul(0.84), fossilFleck.mul(0.18));

  // Subtle darkened wood-contact seating patina near the outer Tier-0 dowel track & underside AO
  const seatTrackMask = smoothstep(float(1.19), float(1.255), sRad).mul(
    smoothstep(float(-0.008), float(0.0), sPos.y)
  );
  const undersideShade = smoothstep(float(-0.098), float(-0.012), sPos.y);
  limestoneMat.colorNode = detailedStone
    .mul(float(0.76).add(undersideShade.mul(0.24)))
    .mul(float(1.0).sub(seatTrackMask.mul(0.09)));

  limestoneMat.roughnessNode = clamp(
    float(0.72)
      .add(ooliteGrain.mul(0.14))
      .add(pitMask.mul(0.16))
      .sub(veinBand.mul(0.05)),
    0.66,
    0.94
  );

  const lN0 = normalize(normalLocal);
  const lRef = select(abs(lN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const lTan = normalize(cross(lRef, lN0));
  const lBit = normalize(cross(lN0, lTan));
  const lBumpA = ooliteGrain.sub(0.5).mul(0.075).sub(pitMask.mul(0.11));
  const lBumpB = strata.sub(0.5).mul(0.055).add(fossilFleck.mul(0.04));
  limestoneMat.normalNode = transformNormalToView(
    normalize(lN0.add(lTan.mul(lBumpA)).add(lBit.mul(lBumpB)))
  );

  // ---------------------------------------------------------------------------
  // 3. Hand-Rubbed Satin Dark Architectural Brass Outer Lip & Inner Oculus Bezel
  // ---------------------------------------------------------------------------
  const brassMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.36,
    metalness: 0.88,
  });

  const bPos = positionLocal;
  const bN0 = normalize(normalLocal);
  const bRad = sqrt(max(float(1e-6), bPos.x.mul(bPos.x).add(bPos.z.mul(bPos.z))));

  // Fine concentric lathe-turning micro-grooves + subtle architectural bronze patina
  const lathePhase = bRad.mul(420.0).add(bPos.y.mul(360.0));
  const latheGroove = sin(lathePhase).mul(0.5).add(0.5);
  const brassPatina = mineralToothFbm(bPos.mul(14.0), float(1.65));
  const patinaMask = smoothstep(float(0.20), float(0.82), brassPatina);

  // Subtle chamfer edge burnishing where fingers & light polish dark bronze toward antique brass
  const bevelHighlight = float(1.0).sub(abs(bN0.y)).mul(abs(bN0.y)).mul(4.0);

  // Deep dark oil-rubbed architectural brass (strictly from rawUmber + yellowOchre + burntSienna)
  const darkPatinaBronze = mix(umber.rInf.mul(0.42), sienna.rInf.mul(0.28), float(0.20));
  const warmAntiqueBrass = mix(ochre.rInf.mul(0.54), umber.rInf.mul(0.56), float(0.50));
  const brassMix = clamp(
    float(0.34).add(patinaMask.mul(0.30)).add(latheGroove.mul(0.08)).add(bevelHighlight.mul(0.22)),
    0.26,
    0.92
  );
  brassMat.colorNode = mix(darkPatinaBronze, warmAntiqueBrass, brassMix);

  brassMat.roughnessNode = clamp(
    float(0.38)
      .add(float(1.0).sub(patinaMask).mul(0.11))
      .add(latheGroove.mul(0.045))
      .sub(bevelHighlight.mul(0.05)),
    0.33,
    0.52
  );

  // True singularity-free meridian-tangent projection for lathe-turned micro-grooves:
  // works identically on flat horizontal top bevels, 45 deg chamfers, AND vertical walls!
  const circTan = normalize(vec3(bPos.z.negate(), float(0.0), bPos.x));
  const meridTan = normalize(cross(bN0, circTan));
  const bLatheSlope = cos(lathePhase).mul(0.026).add(brassPatina.sub(0.5).mul(0.018));
  brassMat.normalNode = transformNormalToView(
    normalize(bN0.add(meridTan.mul(bLatheSlope)))
  );

  // ---------------------------------------------------------------------------
  // 4. Three Distinct Speckled Celadon & Bone-Calcite River-Pebble "Eggs"
  // ---------------------------------------------------------------------------
  const eggMats: THREE.MeshStandardNodeMaterial[] = eggs.map((egg) => {
    const mat = new THREE.MeshStandardNodeMaterial({
      roughness: 0.42,
      metalness: 0.01,
    });

    const ePos = positionLocal;
    const eSeed = float(egg.seed * 19.3);

    const cloud = mineralToothFbm(ePos.mul(12.5).add(eSeed), float(1.85));
    const veinNoise = mineralToothFbm(
      vec3(ePos.x.mul(18.0).add(eSeed), ePos.y.mul(24.0), ePos.z.mul(18.0).sub(eSeed)),
      float(2.0)
    );
    const speckFieldA = mineralToothFbm(ePos.mul(42.0).add(eSeed.mul(1.7)), float(2.3));
    const speckFieldB = mineralToothFbm(ePos.mul(84.0).sub(eSeed.mul(2.3)), float(2.1));

    // Delicate wild-egg & river-stone mineral speckles + faint mineral hairline veining
    const coarseSpecks = smoothstep(float(0.68), float(0.80), speckFieldA);
    const fineSpecks = smoothstep(float(0.69), float(0.83), speckFieldB).mul(0.72);
    const speckleMask = clamp(coarseSpecks.add(fineSpecks), 0.0, 1.0);
    const hairVein = float(1.0).sub(smoothstep(float(0.0), float(0.045), abs(veinNoise.sub(0.52))));

    // Wide organic progression across the 3 eggs:
    //   Egg 0 (mix=0.08): deep misty Celadon Glauconite glaze
    //   Egg 1 (mix=0.48): pale translucent sea-glass Celadon over Bone Calcite
    //   Egg 2 (mix=0.94): warm speckled Bone-Calcite & Kaolin river stone
    const celadonStrength = Math.pow(Math.max(0, 1.0 - egg.celadonMix), 1.65);
    const glazeThickness = clamp(
      float(celadonStrength).mul(float(0.55).add(cloud.mul(0.65))),
      0.012,
      1.20
    );
    const shellBacking = mix(
      mix(calcite.rInf.mul(0.80), kaolin.rInf.mul(0.76), float(0.55)),
      ochre.rInf.mul(0.76),
      cloud.mul(0.12)
    );
    const shellBaseRgb = kubelkaMunkGlaze(
      celadon.K,
      celadon.S,
      glazeThickness,
      shellBacking
    );

    // Subtle organic mineral clouding + dark umber/moss/sienna natural wild-egg flecks
    const veinColor = mix(kaolin.rInf.mul(0.84), umber.rInf.mul(0.62), float(0.35));
    const veinedEgg = mix(shellBaseRgb, veinColor, hairVein.mul(0.09).mul(cloud));
    const fleckColor = mix(
      mix(umber.rInf.mul(0.34), moss.rInf.mul(0.46), cloud),
      sienna.rInf.mul(0.40),
      float(egg.celadonMix * 0.45)
    );
    mat.colorNode = mix(veinedEgg, fleckColor, speckleMask.mul(0.74));

    // River-tumbled velvety matte-satin stone/eggshell sheen (zero glossy marble hotspot!)
    mat.roughnessNode = clamp(
      float(0.44).add(cloud.mul(0.11)).add(speckleMask.mul(0.20)),
      0.40,
      0.72
    );

    const eN0 = normalize(normalLocal);
    const eRef = select(abs(eN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const eTan = normalize(cross(eRef, eN0));
    const eBit = normalize(cross(eN0, eTan));
    const eMicroA = speckFieldA.sub(0.5).mul(0.028).add(hairVein.mul(0.016));
    const eMicroB = cloud.sub(0.5).mul(0.020);
    mat.normalNode = transformNormalToView(
      normalize(eN0.add(eTan.mul(eMicroA)).add(eBit.mul(eMicroB)))
    );

    return mat;
  });

  return {
    uHoveredRodId,
    uGrabbedRodId,
    uHoverLift,
    timberMat,
    limestoneMat,
    brassMat,
    eggMats,
    dispose: () => {
      timberMat.dispose();
      limestoneMat.dispose();
      brassMat.dispose();
      for (const m of eggMats) m.dispose();
    },
  };
}
