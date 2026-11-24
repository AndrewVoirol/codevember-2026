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
  length,
  max,
  mix,
  normalGeometry,
  normalLocal,
  normalView,
  normalWorld,
  normalize,
  positionGeometry,
  positionLocal,
  positionViewDirection,
  positionWorld,
  pow,
  select,
  sin,
  smoothstep,
  sqrt,
  transformNormalToView,
  uniform,
  vec3,
  vec4,
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
        Math.exp(-dist * dist * 1.95) * Math.max(0, 1 - Math.pow(Math.min(1, dist), 4));
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
 * Builds a museum-grade studio softbox PMREM cubemap calibrated for hand-rubbed oiled
 * hardwoods, brushed dark architectural brass (continuous horizon wrap — zero harsh
 * black bands or blown hotspots), honed Jura limestone, and translucent satin Celadon
 * river-pebble eggs on #212529.
 */
export function createNestStudioEnvironment(
  renderer: THREE.WebGPURenderer
): NestStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Lifted warm-slate gallery enclosure so brushed dark brass retains subtle bronze body everywhere
  envScene.background = new THREE.Color(0x1b2027);

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

  // Warm overhead-front-right museum softbox key (raking across timber lattice & celadon sheen)
  addSoftbox(11.5, 8.8, [4.6, 6.2, 4.0], 1.0, 0.96, 0.89, 1.44);
  // Cool mineral-slate left-flank studio scrim for crisp walnut/brass/egg edge separation
  addSoftbox(10.0, 8.5, [-5.8, 3.0, 2.8], 0.84, 0.91, 1.0, 0.86);
  // Grazing back-top calcite rim scrim separating the woven crown silhouette from #212529
  addSoftbox(13.0, 5.6, [-1.4, 5.4, -5.8], 0.98, 0.95, 0.89, 1.38);
  // Continuous 4-quadrant soft gallery horizon wrap strips so brushed dark brass collars
  // gleam with silky unbroken satin gradations around 360 degrees
  addSoftbox(13.5, 3.6, [5.8, 1.0, -3.2], 0.96, 0.90, 0.82, 0.64);
  addSoftbox(13.5, 3.6, [-4.8, 1.1, -4.6], 0.92, 0.91, 0.94, 0.54);
  addSoftbox(13.5, 3.4, [3.6, 0.9, 5.8], 0.97, 0.93, 0.86, 0.56);
  // Warm limestone-shelf lower bounce scrim bathing the underside of the dome weave
  addSoftbox(11.5, 11.5, [0.3, -5.8, 0.6], 0.96, 0.89, 0.78, 0.72);

  const renderTarget = pmrem.fromScene(envScene, 0.055);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

function createNestUniforms(eggs: readonly CeladonEggSpec[]) {
  const e0 = eggs[0]?.pos ?? [0, 0.75, 0];
  const e1 = eggs[1]?.pos ?? [0, 0.75, 0];
  const e2 = eggs[2]?.pos ?? [0, 0.75, 0];
  return {
    uHoveredRodId: uniform(-1.0),
    uGrabbedRodId: uniform(-1.0),
    uHoverLift: uniform(0.0),
    uEggPos0: uniform(new THREE.Vector3(e0[0], e0[1], e0[2])),
    uEggPos1: uniform(new THREE.Vector3(e1[0], e1[1], e1[2])),
    uEggPos2: uniform(new THREE.Vector3(e2[0], e2[1], e2[2])),
    uEggCradleAo: uniform(new THREE.Vector3(1.0, 1.0, 1.0)),
  };
}

export type NestUniforms = ReturnType<typeof createNestUniforms>;

export interface NestMaterialsBundle {
  readonly uHoveredRodId: NestUniforms['uHoveredRodId'];
  readonly uGrabbedRodId: NestUniforms['uGrabbedRodId'];
  readonly uHoverLift: NestUniforms['uHoverLift'];
  readonly uEggPos0: NestUniforms['uEggPos0'];
  readonly uEggPos1: NestUniforms['uEggPos1'];
  readonly uEggPos2: NestUniforms['uEggPos2'];
  readonly uEggCradleAo: NestUniforms['uEggCradleAo'];
  readonly timberMat: THREE.MeshStandardNodeMaterial;
  readonly limestoneMat: THREE.MeshStandardNodeMaterial;
  readonly brassMat: THREE.MeshStandardNodeMaterial;
  readonly eggMats: THREE.MeshStandardNodeMaterial[];
  readonly dustMat: THREE.MeshStandardNodeMaterial;
  dispose: () => void;
}

/**
 * Creates all TSL PBR NodeMaterials for Day 24 - Nest using strictly physical
 * mineral pigments from `src/core/palette.ts` via 2-flux Kubelka-Munk glazing.
 */
export function createNestMaterials(eggs: readonly CeladonEggSpec[]): NestMaterialsBundle {
  const umber = createPigmentUniforms('rawUmber');
  const sienna = createPigmentUniforms('burntSienna');
  const hematite = createPigmentUniforms('hematiteRust');
  const ochre = createPigmentUniforms('yellowOchre');
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const celadon = createPigmentUniforms('celadonGlauconite');
  const moss = createPigmentUniforms('mossUmbrous');

  const {
    uHoveredRodId,
    uGrabbedRodId,
    uHoverLift,
    uEggPos0,
    uEggPos1,
    uEggPos2,
    uEggCradleAo,
  } = createNestUniforms(eggs);

  // ---------------------------------------------------------------------------
  // 1. Turned Timber Dowels — 3D Solid-Wood Grain, Lathe-Turned Micro-Rings,
  //    Crossing Contact Burnish, Warm Chamfer Oil-Sheen & End-Grain Checks
  // ---------------------------------------------------------------------------
  const timberMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.58,
    metalness: 0.02,
  });

  // Per-instance attribute: x=woodSpecies [0..1], y=grainSeed [0..1], z=rodId, w=tactileFlash [0..1]
  const rodMeta = attribute('aRodMeta', 'vec4');
  const woodSpecies = rodMeta.x;
  const grainSeed = rodMeta.y;
  const rodId = rodMeta.z;
  const tactileFlash = rodMeta.w;
  const notchMask = attribute('aNotchMask', 'float');
  // Per-instance live quaternion matching `instanceMatrix` rotation for exact normal perturbation transform
  const rodQuatRaw = attribute('aRodQuat', 'vec4');
  const rodQuatLen = length(rodQuatRaw);
  const rodQuat = select(
    rodQuatLen.greaterThan(0.01),
    rodQuatRaw.div(max(float(1e-4), rodQuatLen)),
    vec4(0.0, 0.0, 0.0, 1.0)
  );

  // Strictly PRE-INSTANCE dowel-local 3D coordinates (`positionGeometry` / `normalGeometry`!):
  // On InstancedMesh in Three.js r187, `positionLocal` is already instance-transformed into world space,
  // whereas `positionGeometry` stays strictly in dowel-local space (Y = shaft axis, XZ = circular cap).
  // This guarantees concentric end-grain bullseyes, zero texture swimming during tumbles, and crisp chamfers!
  const pLoc = positionGeometry;
  const wN0 = normalize(normalGeometry);
  const absNy = abs(wN0.y);
  // 1.0 on flat cut circular end-grain faces, smooth band strictly on 45 deg chamfer bevel, 0.0 on shaft barrel
  const endFaceMask = smoothstep(float(0.84), float(0.97), absNy);
  const bevelMask = smoothstep(float(0.18), float(0.68), absNy).mul(float(1.0).sub(endFaceMask));
  const shaftMask = float(1.0).sub(endFaceMask);

  const seedAngle = grainSeed.mul(41.3);
  const seedShift = grainSeed.mul(23.7);

  // Virtual tree-log pith slightly off-center (~3.2..8.6 mm inside 18..26 mm dowel radius) so EVERY flat
  // cut end displays unmistakable concentric growth rings while shafts slice organic cathedral grain arches
  const pithRadius = float(0.0032).add(grainSeed.mul(0.0054));
  const pithX = cos(seedAngle).mul(pithRadius);
  const pithZ = sin(seedAngle).mul(pithRadius);

  // Gentle organic wood-fiber wander flowing slowly along longitudinal Y (sculpting natural cathedral grain arches)
  const warpCoordA = vec3(
    pLoc.x.mul(24.0).add(seedShift),
    pLoc.y.mul(3.6).add(grainSeed.mul(9.7)),
    pLoc.z.mul(24.0).sub(seedShift.mul(0.6))
  );
  const warpCoordB = vec3(
    pLoc.x.mul(22.0).sub(seedShift.mul(0.7)),
    pLoc.y.mul(4.1).add(grainSeed.mul(13.1)),
    pLoc.z.mul(22.0).add(seedShift)
  );
  const warpA = mineralToothFbm(warpCoordA, float(1.15)).sub(0.5);
  const warpB = mineralToothFbm(warpCoordB, float(1.15)).sub(0.5);
  const warpAmp = mix(float(0.0062), float(0.0014), endFaceMask);

  const rx = pLoc.x.sub(pithX).add(warpA.mul(warpAmp));
  const rz = pLoc.z.sub(pithZ).add(warpB.mul(warpAmp));
  const treeR = sqrt(max(float(1e-7), rx.mul(rx).add(rz.mul(rz))));

  // Natural non-uniform seasonal ring spacing (earlywood/latewood bunching — zero uniform zebra stripes!)
  const growthMod = mineralToothFbm(
    vec3(rx.mul(52.0).add(seedShift), pLoc.y.mul(1.8), rz.mul(52.0)),
    float(1.2)
  );
  const ringFreq = mix(float(285.0), float(330.0), endFaceMask);
  const ringCoord = treeR.mul(ringFreq).add(growthMod.mul(2.3)).add(warpA.mul(0.95)).add(8.0);
  const ringFrac = fract(ringCoord);
  const latewoodBand = smoothstep(float(0.58), float(0.84), ringFrac).mul(
    float(1.0).sub(smoothstep(float(0.88), float(0.99), ringFrac))
  );
  const ringGrad = smoothstep(float(0.04), float(0.74), ringFrac);
  const ringContrast = clamp(latewoodBand.mul(0.78).add(ringGrad.mul(0.22)), 0.0, 1.0);

  // Tight anisotropic longitudinal fibers, open hardwood vascular pores, and quartersawn medullary rays
  const fineFiber = mineralToothFbm(
    vec3(
      pLoc.x.mul(255.0).add(seedShift),
      pLoc.y.mul(12.5).add(grainSeed.mul(17.3)),
      pLoc.z.mul(255.0).sub(seedShift)
    ),
    float(1.75)
  );
  const microVessels = mineralToothFbm(
    vec3(
      pLoc.x.mul(440.0).sub(seedShift),
      pLoc.y.mul(26.0).add(grainSeed.mul(7.1)),
      pLoc.z.mul(440.0).add(seedShift)
    ),
    float(1.6)
  );
  const shaftPores = smoothstep(float(0.61), float(0.81), microVessels);

  const rayNoise = mineralToothFbm(
    vec3(
      pLoc.x.mul(120.0).add(seedShift.mul(1.3)),
      pLoc.y.mul(44.0).sub(grainSeed.mul(11.3)),
      pLoc.z.mul(120.0).sub(seedShift)
    ),
    float(1.45)
  );
  const medullaryRays = smoothstep(float(0.72), float(0.87), rayNoise).mul(shaftMask);

  // Subtle hand-turned lathe micro-tool rings strictly circumferential around dowel shaft Y
  const latheFeedDrift = mineralToothFbm(
    vec3(grainSeed.mul(13.1), pLoc.y.mul(22.0).add(seedShift), grainSeed.mul(7.7)),
    float(1.0)
  );
  const latheToolPhase = pLoc.y.mul(470.0).add(latheFeedDrift.mul(5.8)).add(grainSeed.mul(31.7));
  const latheToolWave = cos(latheToolPhase);
  const latheToolVal = sin(latheToolPhase).mul(0.5).add(0.5);

  // Cross-cut open end-grain vessel stipple + pith-to-bark heartwood gradient
  const endVesselNoise = mineralToothFbm(
    vec3(
      pLoc.x.mul(440.0).add(seedShift),
      pLoc.y.mul(18.0),
      pLoc.z.mul(440.0).sub(seedShift)
    ),
    float(1.65)
  );
  const endPores = smoothstep(float(0.58), float(0.80), endVesselNoise);
  const pithDarken = smoothstep(float(0.013), float(0.001), treeR).mul(endFaceMask).mul(0.22);

  // Faint organic radial drying checks radiating from pith on cut flat end-grain (zero star harmonics!)
  const checkAng1 = grainSeed.mul(25.13).add(0.75);
  const checkAng2 = checkAng1.add(2.38).add(fract(grainSeed.mul(5.7)).mul(1.55));
  const c1x = cos(checkAng1);
  const c1z = sin(checkAng1);
  const c2x = cos(checkAng2);
  const c2z = sin(checkAng2);

  const checkWarp = warpB.mul(0.00075);
  const rxC = rx.add(checkWarp);
  const rzC = rz.sub(checkWarp);

  const perp1 = abs(rxC.mul(c1z).sub(rzC.mul(c1x)));
  const along1 = rxC.mul(c1x).add(rzC.mul(c1z));
  const checkLen1 = float(0.0095).add(fract(grainSeed.mul(11.3)).mul(0.0095));
  const checkWidth1 = mix(float(0.00068), float(0.00018), clamp(along1.div(checkLen1), 0.0, 1.0));
  const checkMask1 = smoothstep(checkWidth1, float(0.00011), perp1)
    .mul(smoothstep(float(0.0003), float(0.0018), along1))
    .mul(float(1.0).sub(smoothstep(checkLen1.mul(0.55), checkLen1, along1)));

  const perp2 = abs(rxC.mul(c2z).sub(rzC.mul(c2x)));
  const along2 = rxC.mul(c2x).add(rzC.mul(c2z));
  const checkLen2 = float(0.0062).add(fract(grainSeed.mul(17.9)).mul(0.0068));
  const check2Gate = smoothstep(float(0.28), float(0.56), fract(grainSeed.mul(7.31)));
  const checkMask2 = smoothstep(float(0.00048), float(0.00010), perp2)
    .mul(smoothstep(float(0.0004), float(0.0016), along2))
    .mul(float(1.0).sub(smoothstep(checkLen2.mul(0.50), checkLen2, along2)))
    .mul(check2Gate);

  const endCheckMask = clamp(checkMask1.add(checkMask2.mul(0.75)), 0.0, 1.0).mul(endFaceMask);

  // Richer fibrous shaft blend + crisp concentric rings on cut end-grain tips
  const grainPattern = mix(
    clamp(ringContrast.mul(0.48).add(fineFiber.mul(0.52)), 0.0, 1.0),
    clamp(latewoodBand.mul(0.88).add(ringGrad.mul(0.12)), 0.0, 1.0),
    endFaceMask
  );

  // Four natural mineral-pigment hardwood species (strictly from src/core/palette.ts):
  //   0.04 -> Deep Oiled Black Walnut (rich espresso-umber heartwood + warm sienna chatoyance)
  //   0.34 -> Warm Fumed Smoked Ash (rich warm taupe-chestnut ash, never cold grey PVC)
  //   0.56..0.76 -> Aromatic Red-Honey Cedar to Warm Golden-Ochre Cedar
  //   0.96 -> Pale Creamy Satin Nordic Birch (clean Bone-Calcite + Warm-Kaolin framing the celadon eggs)
  const walnutEarly = mix(umber.rInf.mul(0.68), sienna.rInf.mul(0.46), float(0.30));
  const walnutLate = mix(umber.rInf.mul(0.24), hematite.rInf.mul(0.22), float(0.18));
  const walnutRgb = mix(walnutEarly, walnutLate, grainPattern.mul(0.76)).mul(
    float(1.0).sub(shaftPores.mul(0.25))
  );

  const ashBase = mix(umber.rInf.mul(0.92), sienna.rInf.mul(0.56), float(0.34));
  const ashEarly = mix(ashBase, ochre.rInf.mul(0.56), float(0.22));
  const ashLate = mix(umber.rInf.mul(0.36), sienna.rInf.mul(0.25), float(0.26));
  const smokedAshRgb = mix(ashEarly, ashLate, grainPattern.mul(0.74)).mul(
    float(1.0).sub(shaftPores.mul(0.27))
  );

  const cedarEarly = kubelkaMunkGlaze(
    ochre.K,
    ochre.S,
    clamp(float(0.28).add(fineFiber.mul(0.18)), 0.12, 0.62),
    mix(sienna.rInf.mul(0.72), kaolin.rInf.mul(0.60), float(0.36))
  );
  const cedarLate = mix(sienna.rInf.mul(0.50), umber.rInf.mul(0.46), float(0.42));
  const cedarRgb = mix(cedarEarly, cedarLate, grainPattern.mul(0.68)).mul(
    float(1.0).sub(shaftPores.mul(0.19))
  );

  // Clean creamy Nordic birch (Bone-Calcite + Warm-Kaolin + delicate caramel-ochre latewood lines)
  const birchEarly = kubelkaMunkGlaze(
    sienna.K,
    sienna.S,
    clamp(float(0.014).add(fineFiber.mul(0.020)), 0.008, 0.048),
    mix(calcite.rInf.mul(0.88), kaolin.rInf.mul(0.84), float(0.46))
  );
  const birchLate = mix(kaolin.rInf.mul(0.62), mix(ochre.rInf.mul(0.62), umber.rInf.mul(0.74), float(0.55)), float(0.46));
  const birchRgb = mix(birchEarly, birchLate, grainPattern.mul(0.52)).mul(
    float(1.0).sub(shaftPores.mul(0.12))
  );

  // Smooth per-tier species interpolation + organic per-dowel heartwood/sapwood tone & warmth variation
  const tAsh = smoothstep(float(0.06), float(0.35), woodSpecies);
  const tCedar = smoothstep(float(0.36), float(0.72), woodSpecies);
  const tBirch = smoothstep(float(0.78), float(0.95), woodSpecies);

  const speciesRgb0 = mix(walnutRgb, smokedAshRgb, tAsh);
  const speciesRgb1 = mix(speciesRgb0, cedarRgb, tCedar);
  const speciesRgbRaw = mix(speciesRgb1, birchRgb, tBirch);

  const rodToneHash = fract(grainSeed.mul(13.7).add(rodId.mul(0.173)));
  const rodWarmHash = fract(grainSeed.mul(29.3).add(rodId.mul(0.317)));
  const perRodTone = float(0.89).add(rodToneHash.mul(0.22));
  const warmHeartTint = mix(umber.rInf.mul(0.92), sienna.rInf.mul(0.98), rodWarmHash);
  const speciesRgb = mix(speciesRgbRaw, speciesRgbRaw.mul(warmHeartTint).mul(1.85), float(0.11));

  // Medullary ray chatoyance + warm oil-sheen edge highlight along 45 deg chamfered tips
  const rayLiftRgb = mix(speciesRgb, mix(ochre.rInf, kaolin.rInf, float(0.52)), medullaryRays.mul(0.10));
  const chamferOilTint = mix(ochre.rInf.mul(0.82), kaolin.rInf.mul(0.86), tBirch.mul(0.65).add(0.25));
  const chamferLiftedRgb = mix(rayLiftRgb, chamferOilTint, bevelMask.mul(0.14));

  // Cross-cut end-grain oil absorption + burnished contact-wear patina bands at ALL crossing stations
  // + darkened hairline radial drying checks
  const endOilDarken = endFaceMask.mul(float(0.24).add(endPores.mul(0.16))).add(pithDarken);
  const burnishPatinaRgb = mix(umber.rInf.mul(0.25), sienna.rInf.mul(0.32), float(0.45));
  const burnishedRgb = mix(chamferLiftedRgb, burnishPatinaRgb, notchMask.mul(0.42));
  const baseWoodRgb = burnishedRgb
    .mul(perRodTone)
    .mul(clamp(float(1.0).sub(endOilDarken).sub(endCheckMask.mul(0.68)), 0.22, 1.0))
    .mul(float(0.985).add(latheToolVal.mul(0.032).mul(shaftMask)));

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
  const fresnelRim = pow(clamp(float(1.0).sub(abs(dot(nView, vDir))), 1e-4, 1.0), float(2.7));

  // Keep wood grain 100% rich and legible during hover (warm golden-calcite rim, never chalk-white!)
  const warmRimTint = mix(ochre.rInf, calcite.rInf, float(0.45));
  const highlightMix = clamp(
    hoverAmt.mul(0.068).add(isGrabbed.mul(0.115)).add(tactileFlash.mul(0.085)),
    0.0,
    0.18
  );

  timberMat.colorNode = mix(baseWoodRgb, warmRimTint, highlightMix);
  timberMat.emissiveNode = warmRimTint.mul(
    hoverAmt
      .mul(float(0.012).add(fresnelRim.mul(0.17)))
      .add(isGrabbed.mul(float(0.028).add(fresnelRim.mul(0.25))))
      .add(tactileFlash.mul(float(0.032).add(fresnelRim.mul(0.13))))
  );

  // Hand-rubbed satin-matte oiled hardwood roughness:
  //   - Burnished crossing bands polish smoother (-0.16)
  //   - 45 deg chamfered tips gleam with rubbed oil sheen (-0.13)
  //   - Open pores & radial end-grain checks stay deeply matte (+0.18..+0.26)
  const activePores = mix(shaftPores, endPores, endFaceMask);
  timberMat.roughnessNode = clamp(
    float(0.55)
      .add(fineFiber.mul(0.11))
      .add(activePores.mul(0.18))
      .add(endFaceMask.mul(0.16))
      .add(endCheckMask.mul(0.26))
      .add(latheToolVal.sub(0.5).mul(0.032).mul(shaftMask))
      .sub(latewoodBand.mul(0.055))
      .sub(medullaryRays.mul(0.07))
      .sub(notchMask.mul(0.16))
      .sub(bevelMask.mul(0.13))
      .sub(isGrabbed.mul(0.04)),
    0.34,
    0.92
  );

  // Singularity-free tangent-space micro-normal relief:
  //   - Shaft: longitudinal wood fiber grooves + open vessels + subtle circumferential lathe-tool rings
  //     (compressed smoothly inside burnished crossing bands!)
  //   - End-grain cut face: concentric growth-ring ridges + vessel stipple + indented radial drying checks
  const wRef = select(absNy.lessThan(0.85), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const wTanA = normalize(cross(wRef, wN0));
  const wTanB = normalize(cross(wN0, wTanA));
  const radialEndDir = normalize(vec3(rx, float(0.0), rz).add(vec3(1e-5, 0.0, 0.0)));
  const checkTan1 = vec3(c1z.negate(), float(0.0), c1x);

  const burnishFlatten = mix(float(1.0), float(0.45), notchMask.mul(0.75));
  const shaftSlopeA = fineFiber
    .sub(0.5)
    .mul(0.105)
    .add(latewoodBand.sub(0.5).mul(0.052))
    .sub(shaftPores.mul(0.095))
    .mul(burnishFlatten);
  const shaftSlopeB = microVessels
    .sub(0.5)
    .mul(0.032)
    .add(latheToolWave.mul(0.022).mul(shaftMask).mul(burnishFlatten));
  const shaftNormal = normalize(wN0.add(wTanA.mul(shaftSlopeA)).add(wTanB.mul(shaftSlopeB)));

  const endRingSlope = cos(ringFrac.mul(6.283185)).mul(0.075).add(endPores.sub(0.5).mul(0.072));
  const endCheckSlope = endCheckMask.mul(0.18);
  const endNormal = normalize(
    wN0.add(radialEndDir.mul(endRingSlope)).add(checkTan1.mul(endCheckSlope))
  );

  // Rotate dowel-local perturbed normal by live per-instance quaternion `rodQuat` into mesh-local (= world)
  // space before `transformNormalToView`, matching `InstancedMesh` `normalLocal` with 0.0 error!
  const localPerturbedN = normalize(mix(shaftNormal, endNormal, endFaceMask));
  const qxyz = rodQuat.xyz;
  const tQ = cross(qxyz, localPerturbedN).mul(2.0);
  const worldPerturbedN = normalize(
    localPerturbedN.add(tQ.mul(rodQuat.w)).add(cross(qxyz, tQ))
  );
  timberMat.normalNode = transformNormalToView(worldPerturbedN);

  // ---------------------------------------------------------------------------
  // 2. Honed Jura / Travertine Limestone Plinth Ring
  //    Fine matte sedimentary banding, fossiliferous mineral specks,
  //    crystalline calcite spar flecks, dowel-seat contact-AO & rabbet reveals
  // ---------------------------------------------------------------------------
  const limestoneMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.78,
    metalness: 0.01,
  });

  const sPos = positionLocal;
  const sRad = sqrt(max(float(1e-6), sPos.x.mul(sPos.x).add(sPos.z.mul(sPos.z))));

  // Fine matte-honed Jura/travertine sedimentary bedding planes
  const bedWarp = mineralToothFbm(sPos.mul(3.2), float(1.4)).sub(0.5);
  const strata = mineralToothFbm(
    vec3(sPos.x.mul(3.1).add(bedWarp.mul(0.85)), sPos.y.mul(22.0).add(bedWarp.mul(1.9)), sPos.z.mul(3.1)),
    float(1.65)
  );
  const fineBand = mineralToothFbm(
    vec3(sPos.x.mul(7.5).sub(bedWarp.mul(0.6)), sPos.y.mul(48.0), sPos.z.mul(7.5).add(bedWarp.mul(0.6))),
    float(1.5)
  );
  const ooliteGrain = mineralToothFbm(sPos.mul(32.0), float(2.1));
  const travertinePits = mineralToothFbm(
    vec3(sPos.x.mul(48.0), sPos.y.mul(72.0), sPos.z.mul(48.0)),
    float(2.2)
  );
  const fossilField = mineralToothFbm(sPos.mul(92.0).add(vec3(4.3, 1.7, 9.1)), float(2.15));

  const veinBand = smoothstep(float(0.42), float(0.75), strata);
  const fineSediment = smoothstep(float(0.45), float(0.72), fineBand);
  const pitMask = smoothstep(float(0.70), float(0.87), travertinePits);
  const calciteSpar = smoothstep(float(0.73), float(0.88), ooliteGrain);
  const darkFossilSpeck = smoothstep(float(0.72), float(0.84), fossilField);

  // Calibrated pale honed Jura limestone (warm travertine stone contrast against walnut & dark brass)
  const stoneBacking = mix(kaolin.rInf.mul(0.79), calcite.rInf.mul(0.75), float(0.52));
  const stoneGlaze = kubelkaMunkGlaze(
    umber.K,
    umber.S,
    clamp(float(0.032).add(strata.mul(0.058)).add(fineSediment.mul(0.025)), 0.02, 0.14),
    stoneBacking
  );
  const veinedStone = mix(stoneGlaze, umber.rInf.mul(0.76), veinBand.mul(0.22));
  const pittedStone = mix(veinedStone, umber.rInf.mul(0.44), pitMask.mul(0.36));
  const fossilTint = mix(umber.rInf.mul(0.38), hematite.rInf.mul(0.42), fineSediment);
  const fossilStone = mix(pittedStone, fossilTint, darkFossilSpeck.mul(0.34));
  const detailedStone = mix(fossilStone, calcite.rInf.mul(0.88), calciteSpar.mul(0.20));

  // Darkened wood-contact AO & rubbed patina ring where the 16 Tier-0 dowels rest + crisp rabbet AO
  const topShelfGate = smoothstep(float(-0.009), float(-0.0005), sPos.y);
  const seatTrackMask = smoothstep(float(1.158), float(1.235), sRad)
    .mul(float(1.0).sub(smoothstep(float(1.255), float(1.264), sRad)))
    .mul(topShelfGate);
  const innerRabbetAo = float(1.0)
    .sub(smoothstep(float(1.135), float(1.148), sRad))
    .mul(topShelfGate);
  const outerRabbetAo = smoothstep(float(1.248), float(1.261), sRad).mul(topShelfGate);
  const rabbetAo = clamp(innerRabbetAo.add(outerRabbetAo), 0.0, 1.0);
  const undersideShade = smoothstep(float(-0.098), float(-0.012), sPos.y);

  limestoneMat.colorNode = detailedStone
    .mul(float(0.75).add(undersideShade.mul(0.25)))
    .mul(clamp(float(1.0).sub(seatTrackMask.mul(0.15)).sub(rabbetAo.mul(0.18)), 0.56, 1.0));

  limestoneMat.roughnessNode = clamp(
    float(0.73)
      .add(ooliteGrain.mul(0.13))
      .add(pitMask.mul(0.17))
      .add(darkFossilSpeck.mul(0.08))
      .sub(veinBand.mul(0.05))
      .sub(calciteSpar.mul(0.14))
      .sub(seatTrackMask.mul(0.07)),
    0.58,
    0.95
  );

  const lN0 = normalize(normalLocal);
  const lRef = select(abs(lN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const lTan = normalize(cross(lRef, lN0));
  const lBit = normalize(cross(lN0, lTan));
  const lBumpA = ooliteGrain.sub(0.5).mul(0.078).sub(pitMask.mul(0.125)).sub(darkFossilSpeck.mul(0.04));
  const lBumpB = strata.sub(0.5).mul(0.052).add(fineBand.sub(0.5).mul(0.032)).add(calciteSpar.mul(0.045));
  limestoneMat.normalNode = transformNormalToView(
    normalize(lN0.add(lTan.mul(lBumpA)).add(lBit.mul(lBumpB)))
  );

  // ---------------------------------------------------------------------------
  // 3. Hand-Rubbed Brushed Dark-Brass Outer Retaining Collar & Inner Oculus Bezel
  //    Subtle rubbed-gold specular edge on chamfered lips vs. aged umber-bronze rabbet
  // ---------------------------------------------------------------------------
  const brassMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.36,
    metalness: 0.89,
  });

  const bPos = positionLocal;
  const bN0 = normalize(normalLocal);
  const bRad = sqrt(max(float(1e-6), bPos.x.mul(bPos.x).add(bPos.z.mul(bPos.z))));

  // Multi-harmonic fine circumferential lathe-brushed micro-grooves + oxidized bronze patina
  const lathePhaseA = bRad.mul(460.0).add(bPos.y.mul(390.0));
  const lathePhaseB = bRad.mul(1040.0).sub(bPos.y.mul(820.0));
  const latheGroove = sin(lathePhaseA).mul(0.35).add(sin(lathePhaseB).mul(0.15)).add(0.5);
  const brassPatina = mineralToothFbm(bPos.mul(15.0), float(1.7));
  const fineBrushNoise = mineralToothFbm(
    vec3(bPos.x.mul(6.0), bPos.y.mul(95.0), bPos.z.mul(6.0)),
    float(1.4)
  );
  const patinaMask = smoothstep(float(0.20), float(0.82), brassPatina);

  // Separate rubbed-gold chamfered upper lips from aged umber-bronze recessed rabbet & inner wall
  const bevelHighlight = float(1.0).sub(abs(bN0.y)).mul(abs(bN0.y)).mul(4.0);
  const outerTopLip = smoothstep(float(0.022), float(0.045), bPos.y);
  const innerBezelTop = smoothstep(float(-0.005), float(0.0), bPos.y).mul(
    float(1.0).sub(smoothstep(float(1.132), float(1.16), bRad))
  );
  const rubbedLipEdge = clamp(bevelHighlight.mul(clamp(outerTopLip.add(innerBezelTop), 0.0, 1.0)), 0.0, 1.0);
  const rabbetRecessMask = clamp(
    smoothstep(float(-0.012), float(-0.044), bPos.y).add(
      smoothstep(float(1.274), float(1.261), bRad)
        .mul(smoothstep(float(0.036), float(0.002), bPos.y))
        .mul(smoothstep(float(1.20), float(1.25), bRad))
    ),
    0.0,
    1.0
  );

  // Deep oil-rubbed architectural bronze-brass strictly from rawUmber + burntSienna + yellowOchre + boneCalcite
  const agedUmberBronze = mix(umber.rInf.mul(0.34), sienna.rInf.mul(0.24), float(0.22));
  const warmAntiqueBrass = mix(ochre.rInf.mul(0.56), umber.rInf.mul(0.54), float(0.48));
  const rubbedGoldEdge = mix(ochre.rInf.mul(0.76), calcite.rInf.mul(0.68), float(0.24));

  const bodyBrassMix = clamp(
    float(0.32)
      .add(patinaMask.mul(0.28))
      .add(latheGroove.mul(0.08))
      .add(fineBrushNoise.mul(0.07))
      .sub(rabbetRecessMask.mul(0.24)),
    0.14,
    0.85
  );
  const bodyBrassRgb = mix(agedUmberBronze, warmAntiqueBrass, bodyBrassMix);
  brassMat.colorNode = mix(bodyBrassRgb, rubbedGoldEdge, rubbedLipEdge.mul(0.58));

  brassMat.roughnessNode = clamp(
    float(0.41)
      .add(float(1.0).sub(patinaMask).mul(0.11))
      .add(rabbetRecessMask.mul(0.12))
      .add(latheGroove.mul(0.04))
      .sub(rubbedLipEdge.mul(0.18)),
    0.23,
    0.58
  );

  // Singularity-free meridian-tangent projection for brushed circumferential lathe micro-grooves
  const circTan = normalize(vec3(bPos.z.negate(), float(0.0), bPos.x));
  const meridTan = normalize(cross(bN0, circTan));
  const bLatheSlope = cos(lathePhaseA)
    .mul(0.024)
    .add(cos(lathePhaseB).mul(0.011))
    .add(brassPatina.sub(0.5).mul(0.018));
  brassMat.normalNode = transformNormalToView(
    normalize(bN0.add(meridTan.mul(bLatheSlope)))
  );

  // ---------------------------------------------------------------------------
  // 4. Three Ovoid River-Pebble Eggs — Translucent Satin Sea-Glass / Celadon Sheen,
  //    Layered Mineral Veining, Speckled Umber/Hematite Inclusions,
  //    Micro-Polished Water-Worn Pits & Live Mutual Contact-AO
  // ---------------------------------------------------------------------------
  const eggPosUniforms = [uEggPos0, uEggPos1, uEggPos2] as const;
  const keyDirNorm = vec3(0.49, 0.72, 0.49);

  const eggMats: THREE.MeshStandardNodeMaterial[] = eggs.map((egg, eggIdx) => {
    // MeshPhysicalNodeMaterial extends MeshStandardNodeMaterial: subtle water-worn mineral luster!
    const mat = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.40,
      metalness: 0.0,
      clearcoat: 0.22,
      clearcoatRoughness: 0.42,
      ior: 1.52,
    });

    const ePos = positionLocal;
    const eSeed = float(egg.seed * 19.3);

    // Layered mineral clouding + sinuous quartz/calcite sedimentary banding & hairline veins
    const cloud = mineralToothFbm(ePos.mul(12.0).add(eSeed), float(1.8));
    const veinWarp = mineralToothFbm(ePos.mul(9.5).sub(eSeed.mul(0.7)), float(1.5)).sub(0.5);
    const veinCoord = vec3(
      ePos.x.mul(16.0).add(veinWarp.mul(2.4)).add(eSeed),
      ePos.y.mul(24.0).add(veinWarp.mul(1.8)),
      ePos.z.mul(16.0).sub(veinWarp.mul(2.1)).sub(eSeed)
    );
    const veinField = mineralToothFbm(veinCoord, float(1.95));
    const bandField = mineralToothFbm(
      vec3(ePos.x.mul(10.0).add(veinWarp.mul(1.5)), ePos.y.mul(20.0), ePos.z.mul(10.0)).add(eSeed),
      float(1.6)
    );

    const softMineralBand = smoothstep(float(0.38), float(0.68), bandField);
    const hairVeinA = float(1.0).sub(smoothstep(float(0.0), float(0.036), abs(veinField.sub(0.50))));
    const hairVeinB = float(1.0).sub(smoothstep(float(0.0), float(0.028), abs(veinField.sub(0.67))));
    const hairVein = clamp(hairVeinA.add(hairVeinB.mul(0.65)), 0.0, 1.0);

    // Organic 3-scale dark umber / hematite / moss mineral inclusions (slightly denser near blunt -X end)
    const bluntBias = smoothstep(float(0.11), float(-0.11), ePos.x).mul(0.07);
    const speckFieldA = mineralToothFbm(ePos.mul(36.0).add(eSeed.mul(1.7)), float(2.25));
    const speckFieldB = mineralToothFbm(ePos.mul(76.0).sub(eSeed.mul(2.3)), float(2.15));
    const speckFieldC = mineralToothFbm(ePos.mul(145.0).add(eSeed.mul(3.9)), float(2.0));

    const coarseSpecks = smoothstep(float(0.72).sub(bluntBias), float(0.82).sub(bluntBias), speckFieldA);
    const medSpecks = smoothstep(float(0.71).sub(bluntBias), float(0.83).sub(bluntBias), speckFieldB).mul(0.76);
    const pinSpecks = smoothstep(float(0.71), float(0.84), speckFieldC).mul(0.62);
    const speckleMask = clamp(coarseSpecks.add(medSpecks).add(pinSpecks), 0.0, 1.0);

    // Delicate micro-polished water-worn mineral pits
    const pitField = mineralToothFbm(ePos.mul(58.0).add(eSeed.mul(3.1)), float(2.2));
    const waterPitMask = smoothstep(float(0.75), float(0.88), pitField);

    // Three harmonious natural river-pebble finishes:
    //   Egg 0 (mix=0.08): deep translucent sea-glass Celadon Glauconite
    //   Egg 1 (mix=0.48): misty pale Celadon over translucent warm Calcite
    //   Egg 2 (mix=0.94): warm river-tumbled Bone-Calcite with hematite/umber mineral flecks
    const celadonStrength = Math.pow(Math.max(0, 1.0 - egg.celadonMix), 1.55);
    const glazeThickness = clamp(
      float(celadonStrength).mul(float(0.50).add(cloud.mul(0.58)).add(softMineralBand.mul(0.28))),
      0.012,
      1.28
    );
    const shellBacking = mix(
      mix(calcite.rInf.mul(0.83), kaolin.rInf.mul(0.78), float(0.50)),
      ochre.rInf.mul(0.76),
      cloud.mul(0.11)
    );
    const shellBaseRgb = kubelkaMunkGlaze(
      celadon.K,
      celadon.S,
      glazeThickness,
      shellBacking
    );

    // Subtle layered translucent calcite veins + delicate mineral depth contrast
    const veinColor = mix(
      mix(calcite.rInf.mul(0.92), kaolin.rInf.mul(0.86), float(0.45)),
      umber.rInf.mul(0.58),
      float(egg.celadonMix * 0.58)
    );
    const veinedEgg = mix(shellBaseRgb, veinColor, hairVein.mul(0.28).add(softMineralBand.mul(0.08)));

    // Natural hematite iron-oxide + raw-umber + moss mineral inclusions
    const fleckColor = mix(
      mix(umber.rInf.mul(0.30), moss.rInf.mul(0.42), cloud),
      hematite.rInf.mul(0.36),
      clamp(speckFieldC.mul(0.75).add(egg.celadonMix * 0.35), 0.0, 1.0)
    );
    const speckledEgg = mix(veinedEgg, fleckColor, speckleMask.mul(0.68));
    const pittedEgg = speckledEgg.mul(float(1.0).sub(waterPitMask.mul(0.11)));

    // Live warm mutual contact-AO where the 3 pebbles nestle closely together + seated birch cradle shadow
    const otherA = eggPosUniforms[(eggIdx + 1) % 3];
    const otherB = eggPosUniforms[(eggIdx + 2) % 3];
    const distOtherA = length(positionWorld.sub(otherA));
    const distOtherB = length(positionWorld.sub(otherB));
    const contactA = float(1.0).sub(smoothstep(float(0.105), float(0.185), distOtherA));
    const contactB = float(1.0).sub(smoothstep(float(0.105), float(0.185), distOtherB));
    const mutualContactAo = clamp(contactA.add(contactB), 0.0, 1.0);

    const cradleGate = eggIdx === 0 ? uEggCradleAo.x : eggIdx === 1 ? uEggCradleAo.y : uEggCradleAo.z;
    const eNWorld = normalize(normalWorld);
    const cradleSeatAo = smoothstep(float(0.15), float(-0.72), eNWorld.y).mul(cradleGate);
    const warmContactTint = mix(
      pittedEgg,
      pittedEgg.mul(mix(umber.rInf.mul(0.72), sienna.rInf.mul(0.82), float(0.4))).mul(2.1),
      mutualContactAo.mul(0.24).add(cradleSeatAo.mul(0.18))
    );
    mat.colorNode = warmContactTint.mul(
      float(1.0).sub(mutualContactAo.mul(0.20)).sub(cradleSeatAo.mul(0.14))
    );

    // Subtle physical grazing sea-glass / celadon jade subsurface translucency lift
    // (gated by key-light wrap & occluded inside mutual contact pockets — zero dark-room glow!)
    const eNView = normalize(normalView);
    const eVDir = normalize(positionViewDirection);
    const eFresnel = pow(clamp(float(1.0).sub(abs(dot(eNView, eVDir))), 1e-4, 1.0), float(2.4));
    const keyWrap = clamp(dot(eNWorld, keyDirNorm).mul(0.55).add(0.45), 0.08, 1.0);
    const translucencyRgb = mix(celadon.rInf, calcite.rInf, float(egg.celadonMix * 0.65));
    const sssStrength = float(0.030 + celadonStrength * 0.028);
    mat.emissiveNode = translucencyRgb.mul(
      eFresnel
        .mul(keyWrap)
        .mul(sssStrength)
        .mul(float(1.0).sub(mutualContactAo.mul(0.75)))
        .mul(float(1.0).sub(cradleSeatAo.mul(0.6)))
    );

    // Water-worn velvety satin stone polish: smooth translucent surface with matte micro-pits & inclusions
    mat.roughnessNode = clamp(
      float(0.36)
        .add(cloud.mul(0.10))
        .add(speckleMask.mul(0.22))
        .add(waterPitMask.mul(0.24))
        .sub(hairVein.mul(0.04)),
      0.31,
      0.78
    );

    const eN0 = normalize(normalLocal);
    const eRef = select(abs(eN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const eTan = normalize(cross(eRef, eN0));
    const eBit = normalize(cross(eN0, eTan));
    const eMicroA = speckFieldA.sub(0.5).mul(0.026).add(hairVein.mul(0.020)).sub(waterPitMask.mul(0.075));
    const eMicroB = cloud.sub(0.5).mul(0.018).add(pitField.sub(0.5).mul(0.032));
    mat.normalNode = transformNormalToView(
      normalize(eN0.add(eTan.mul(eMicroA)).add(eBit.mul(eMicroB)))
    );

    return mat;
  });

  // ---------------------------------------------------------------------------
  // 5. Fine Dry Wood-Dust & Limestone-Chalk Motes (Instanced, Zero-Alloc Pool)
  // ---------------------------------------------------------------------------
  const dustMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.90,
    metalness: 0.0,
    transparent: true,
    depthWrite: false,
  });

  // Per-instance attribute `aDustMeta`: x=materialTone [0=walnut, 0.5=cedar/birch, 1=limestone chalk], y=opacity
  const dustMeta = attribute('aDustMeta', 'vec4');
  const dustTone = dustMeta.x;
  const dustAlpha = dustMeta.y;
  const woodDustWarm = mix(umber.rInf.mul(0.78), ochre.rInf.mul(0.84), clamp(dustTone.mul(2.0), 0.0, 1.0));
  const chalkDustRgb = mix(kaolin.rInf.mul(0.92), calcite.rInf.mul(0.94), float(0.5));
  dustMat.colorNode = mix(woodDustWarm, chalkDustRgb, smoothstep(float(0.55), float(0.95), dustTone));
  dustMat.opacityNode = clamp(dustAlpha, 0.0, 0.85);

  return {
    uHoveredRodId,
    uGrabbedRodId,
    uHoverLift,
    uEggPos0,
    uEggPos1,
    uEggPos2,
    uEggCradleAo,
    timberMat,
    limestoneMat,
    brassMat,
    eggMats,
    dustMat,
    dispose: () => {
      timberMat.dispose();
      limestoneMat.dispose();
      brassMat.dispose();
      for (const m of eggMats) m.dispose();
      dustMat.dispose();
    },
  };
}
