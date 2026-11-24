import * as THREE from 'three/webgpu';
import {
  TWO_PI,
  abs,
  attribute,
  clamp,
  cos,
  cross,
  dot,
  float,
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
  transformNormalToView,
  uniform,
  uv,
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
        Math.exp(-dist * dist * 2.3) * Math.max(0, 1 - Math.pow(Math.min(1, dist), 4));
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
 * Builds a studio softbox PMREM cubemap calibrated for oiled hardwoods, satin dark-brass,
 * honed limestone, and speckled celadon stones suspended against Chalky Slate (#212529).
 */
export function createNestStudioEnvironment(
  renderer: THREE.WebGPURenderer
): NestStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x101419);

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

  // Warm overhead-front-right calcite key softbox
  addSoftbox(10.5, 8.5, [4.2, 6.2, 4.4], 1.0, 0.96, 0.89, 1.72);
  // Cool mineral-slate opposite-flank scrim for brass & walnut chamfer separation
  addSoftbox(8.8, 9.2, [-5.8, 2.6, 3.0], 0.86, 0.92, 1.0, 1.08);
  // Grazing back-top calcite rim scrim separating the woven timber silhouette from #212529
  addSoftbox(11.5, 5.2, [-1.4, 5.4, -5.6], 0.98, 0.96, 0.91, 1.48);
  // Gentle warm studio bounce underneath so inspecting the hollow ring plinth from below stays legible
  addSoftbox(9.5, 9.5, [0.4, -6.2, 0.8], 0.96, 0.88, 0.76, 0.56);

  const renderTarget = pmrem.fromScene(envScene, 0.08);
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
  // 1. Turned Timber Dowels (Instanced Longitudinal Wood Grain + Joinery Notches)
  // ---------------------------------------------------------------------------
  const timberMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.46,
    metalness: 0.03,
  });

  // Per-instance attribute: x=woodSpecies [0..1], y=grainSeed [0..1], z=rodId, w=tactileFlash [0..1]
  const rodMeta = attribute('aRodMeta', 'vec4');
  const woodSpecies = rodMeta.x;
  const grainSeed = rodMeta.y;
  const rodId = rodMeta.z;
  const tactileFlash = rodMeta.w;
  const notchMask = attribute('aNotchMask', 'float');

  // Seamless circumferential + strictly longitudinal UV coordinate along each dowel shaft
  // (NEVER world-cylindrical length(positionLocal.xz), zero UV seam across u=0|1!)
  const dUv = uv();
  const phi = dUv.x.mul(TWO_PI);
  const cosPhi = cos(phi);
  const sinPhi = sin(phi);
  const seedShift = grainSeed.mul(23.7);

  // Anisotropic 3D fiber domain: tight radially/circumferentially, elongated longitudinally along dUv.y
  const fiberDomain = vec3(
    cosPhi.mul(1.85).add(seedShift),
    dUv.y.mul(2.45).add(grainSeed.mul(7.3)),
    sinPhi.mul(1.85).sub(seedShift.mul(0.6))
  );
  const grainWarp = mineralToothFbm(fiberDomain, float(1.75));
  const finePores = mineralToothFbm(
    vec3(
      cosPhi.mul(5.8).add(seedShift),
      dUv.y.mul(9.2).add(grainSeed.mul(11.1)),
      sinPhi.mul(5.8)
    ),
    float(2.1)
  );

  // Cathedral heartwood growth rings flowing strictly longitudinally down the dowel
  const ringPhase = cosPhi
    .mul(5.2)
    .add(sinPhi.mul(3.4))
    .add(grainWarp.mul(7.4))
    .add(dUv.y.mul(4.5));
  const ringWave = sin(ringPhase).mul(0.5).add(0.5);
  const sharpFiber = pow(clamp(ringWave, 1e-4, 1.0), float(1.45));

  // Four natural mineral wood palettes (Kubelka-Munk glazed, zero digital neons):
  //   0.00 -> Deep Oiled Raw-Umber Walnut (Tier 0 base timbers)
  //   0.35 -> Smoked-Ash & Burnt-Sienna (Tier 1 counter-weave)
  //   0.70 -> Warm Yellow-Ochre Cedar (Tiers 2-3 woven basket ribs)
  //   1.00 -> Pale Bone-Calcite & Ochre Birch (Tier 4 inner nest cradle twigs)
  const walnutDark = umber.rInf.mul(0.24);
  const walnutLight = mix(umber.rInf.mul(0.62), sienna.rInf.mul(0.46), float(0.36));
  const walnutRgb = mix(walnutDark, walnutLight, sharpFiber.mul(0.68).add(finePores.mul(0.16)));

  const ashBase = kubelkaMunkGlaze(
    sienna.K,
    sienna.S,
    clamp(float(0.34).add(sharpFiber.mul(0.42)), 0.12, 0.92),
    umber.rInf.mul(0.46)
  );
  const smokedAshRgb = mix(umber.rInf.mul(0.34), ashBase, float(0.66).add(finePores.mul(0.18)));

  const cedarGlaze = kubelkaMunkGlaze(
    ochre.K,
    ochre.S,
    clamp(float(0.42).add(sharpFiber.mul(0.48)), 0.14, 1.05),
    sienna.rInf.mul(0.58)
  );
  const cedarRgb = mix(sienna.rInf.mul(0.54), cedarGlaze, float(0.72).add(finePores.mul(0.16)));

  const birchGlaze = kubelkaMunkGlaze(
    ochre.K,
    ochre.S,
    clamp(float(0.18).add(sharpFiber.mul(0.24)), 0.06, 0.54),
    calcite.rInf.mul(0.92)
  );
  const birchRgb = mix(umber.rInf.mul(0.65), birchGlaze, float(0.78).add(finePores.mul(0.12)));

  // Smooth per-tier species interpolation + subtle per-rod organic hue/value jitter
  const tAsh = smoothstep(float(0.05), float(0.36), woodSpecies);
  const tCedar = smoothstep(float(0.36), float(0.72), woodSpecies);
  const tBirch = smoothstep(float(0.72), float(0.96), woodSpecies);

  const speciesRgb0 = mix(walnutRgb, smokedAshRgb, tAsh);
  const speciesRgb1 = mix(speciesRgb0, cedarRgb, tCedar);
  const speciesRgb = mix(speciesRgb1, birchRgb, tBirch);

  // Subtle natural darkening near chamfered end-grain tips & inside carved joinery notches
  const endDist = max(float(0.0), float(1.0).sub(abs(dUv.y.sub(0.5)).mul(2.0)));
  const endGrainMask = float(1.0).sub(smoothstep(float(0.0), float(0.055), endDist));
  const notchDarken = clamp(notchMask.mul(0.42).add(endGrainMask.mul(0.26)), 0.0, 0.56);
  const perRodTone = float(0.92).add(grainSeed.mul(0.16));
  const baseWoodRgb = speciesRgb.mul(perRodTone).mul(float(1.0).sub(notchDarken));

  // Hover & Grab selection masks (uniform control flow — zero dynamic branches)
  const isHovered = float(1.0).sub(smoothstep(float(0.15), float(0.55), abs(rodId.sub(uHoveredRodId))));
  const isGrabbed = float(1.0).sub(smoothstep(float(0.15), float(0.55), abs(rodId.sub(uGrabbedRodId))));
  const hoverAmt = isHovered.mul(uHoverLift);

  const nView = normalize(normalView);
  const vDir = normalize(positionViewDirection);
  const fresnelRim = pow(clamp(float(1.0).sub(abs(dot(nView, vDir))), 1e-4, 1.0), float(2.4));

  const highlightMix = clamp(hoverAmt.mul(0.18).add(isGrabbed.mul(0.28)).add(tactileFlash.mul(0.22)), 0.0, 0.45);
  const highlightTint = mix(ochre.rInf, calcite.rInf, float(0.35).add(isGrabbed.mul(0.3)));

  timberMat.colorNode = mix(baseWoodRgb, highlightTint, highlightMix);
  timberMat.emissiveNode = highlightTint.mul(
    hoverAmt.mul(float(0.05).add(fresnelRim.mul(0.24)))
      .add(isGrabbed.mul(float(0.10).add(fresnelRim.mul(0.34))))
      .add(tactileFlash.mul(0.18))
  );
  timberMat.roughnessNode = clamp(
    float(0.36)
      .add(finePores.mul(0.22))
      .add(notchMask.mul(0.14))
      .sub(sharpFiber.mul(0.08))
      .sub(isGrabbed.mul(0.06)),
    0.26,
    0.72
  );

  // Subtle longitudinal wood-pore normal perturbation (singularity-free tangent frame)
  const wN0 = normalize(normalLocal);
  const wRef = select(abs(wN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const wTanA = normalize(cross(wRef, wN0));
  const wTanB = normalize(cross(wN0, wTanA));
  const poreSlopeA = finePores.sub(0.5).mul(0.062);
  const poreSlopeB = sharpFiber.sub(0.5).mul(0.034);
  timberMat.normalNode = transformNormalToView(
    normalize(wN0.add(wTanA.mul(poreSlopeA)).add(wTanB.mul(poreSlopeB)))
  );

  // ---------------------------------------------------------------------------
  // 2. Honed Limestone Circular Plinth Ring (Bone Calcite + Warm Kaolin Travertine)
  // ---------------------------------------------------------------------------
  const limestoneMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.74,
    metalness: 0.02,
  });

  const sPos = positionLocal;
  const strata = mineralToothFbm(vec3(sPos.x.mul(2.8), sPos.y.mul(14.0), sPos.z.mul(2.8)), float(1.6));
  const finePits = mineralToothFbm(sPos.mul(16.0), float(2.2));
  const veinBand = smoothstep(float(0.46), float(0.76), strata);

  const stoneGlaze = kubelkaMunkGlaze(
    kaolin.K,
    kaolin.S,
    clamp(float(0.45).add(strata.mul(0.35)), 0.18, 0.95),
    calcite.rInf
  );
  const stoneVeinRgb = mix(stoneGlaze, umber.rInf.mul(0.85), veinBand.mul(0.24));
  // Subtle underside ambient occlusion darkening on the bottom face of the floating ring
  const undersideShade = smoothstep(float(-0.095), float(-0.015), sPos.y);
  limestoneMat.colorNode = stoneVeinRgb.mul(float(0.76).add(undersideShade.mul(0.24)));
  limestoneMat.roughnessNode = clamp(float(0.66).add(finePits.mul(0.22)).sub(veinBand.mul(0.06)), 0.56, 0.90);

  const lN0 = normalize(normalLocal);
  const lRef = select(abs(lN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const lTan = normalize(cross(lRef, lN0));
  const lBit = normalize(cross(lN0, lTan));
  const lBumpA = finePits.sub(0.5).mul(0.045);
  const lBumpB = strata.sub(0.5).mul(0.032);
  limestoneMat.normalNode = transformNormalToView(
    normalize(lN0.add(lTan.mul(lBumpA)).add(lBit.mul(lBumpB)))
  );

  // ---------------------------------------------------------------------------
  // 3. Refined Lathe-Turned Satin Dark-Brass Recessed Lip & Inner Bezel (PMREM Sheen)
  // ---------------------------------------------------------------------------
  const brassMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.30,
    metalness: 0.86,
  });

  const bPos = positionLocal;
  const bRad = max(float(1e-6), bPos.x.mul(bPos.x).add(bPos.z.mul(bPos.z))).sqrt();
  const latheGroove = sin(bRad.mul(260.0).add(bPos.y.mul(180.0))).mul(0.5).add(0.5);
  // Subtle fine-scale oxidation variation (tight value range — zero blotchy tortoiseshell patches)
  const brassPatina = mineralToothFbm(bPos.mul(11.5), float(1.6));
  const patinaMask = smoothstep(float(0.22), float(0.82), brassPatina);

  const antiqueBrass = mix(ochre.rInf.mul(0.82), umber.rInf.mul(0.76), float(0.46));
  const darkBronze = antiqueBrass.mul(0.66);
  const brassToneMix = clamp(float(0.62).add(patinaMask.mul(0.26)).add(latheGroove.mul(0.10)), 0.58, 1.0);
  brassMat.colorNode = mix(darkBronze, antiqueBrass, brassToneMix);
  brassMat.roughnessNode = clamp(
    float(0.26).add(float(1.0).sub(patinaMask).mul(0.09)).add(latheGroove.mul(0.04)),
    0.24,
    0.40
  );

  const bN0 = normalize(normalLocal);
  const bLatheBump = cos(bRad.mul(260.0).add(bPos.y.mul(180.0))).mul(0.018);
  brassMat.normalNode = transformNormalToView(normalize(bN0.add(vec3(0.0, bLatheBump, 0.0))));

  // ---------------------------------------------------------------------------
  // 4. Speckled Celadon & Bone-Calcite River-Pebble "Eggs" (1 Material per Egg)
  // ---------------------------------------------------------------------------
  const eggMats: THREE.MeshStandardNodeMaterial[] = eggs.map((egg) => {
    const mat = new THREE.MeshStandardNodeMaterial({
      roughness: 0.34,
      metalness: 0.02,
    });

    const ePos = positionLocal;
    const eSeed = float(egg.seed * 19.3);
    const cloud = mineralToothFbm(ePos.mul(11.5).add(eSeed), float(1.9));
    const speckFieldA = mineralToothFbm(ePos.mul(36.0).add(eSeed.mul(1.7)), float(2.4));
    const speckFieldB = mineralToothFbm(ePos.mul(68.0).sub(eSeed.mul(2.3)), float(2.1));

    // Delicate wild-egg & river-stone mineral speckles (sharp organic flecks, never blotchy polka-dots)
    const coarseSpecks = smoothstep(float(0.67), float(0.79), speckFieldA);
    const fineSpecks = smoothstep(float(0.70), float(0.84), speckFieldB).mul(0.75);
    const speckleMask = clamp(coarseSpecks.add(fineSpecks), 0.0, 1.0);

    // Rich 2-flux Kubelka-Munk Celadon Glauconite glaze over Bone Calcite shell
    const glazeThickness = clamp(
      float(1.0 - egg.celadonMix * 0.72).mul(float(0.45).add(cloud.mul(0.55))),
      0.08,
      1.15
    );
    const shellBaseRgb = kubelkaMunkGlaze(
      celadon.K,
      celadon.S,
      glazeThickness,
      calcite.rInf
    );
    // Subtle moss-umber mineral flecks & warm calcite mottling
    const fleckColor = mix(umber.rInf.mul(0.42), moss.rInf.mul(0.55), cloud);
    const mottledRgb = mix(shellBaseRgb, kaolin.rInf, cloud.mul(0.16));
    mat.colorNode = mix(mottledRgb, fleckColor, speckleMask.mul(0.72));

    // Smooth river-worn satin sheen (slightly more matte over mineral speckles)
    mat.roughnessNode = clamp(
      float(0.28).add(cloud.mul(0.12)).add(speckleMask.mul(0.18)),
      0.24,
      0.60
    );

    const eN0 = normalize(normalLocal);
    const eRef = select(abs(eN0.y).lessThan(0.9), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const eTan = normalize(cross(eRef, eN0));
    const eBit = normalize(cross(eN0, eTan));
    const eMicroA = speckFieldA.sub(0.5).mul(0.022);
    const eMicroB = cloud.sub(0.5).mul(0.016);
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
