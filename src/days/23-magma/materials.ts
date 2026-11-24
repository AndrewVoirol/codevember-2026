import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  max,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionGeometry,
  positionLocal,
  positionWorld,
  pow,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import {
  CHAMBER_ROOF_Y,
  DIKE_SEAM_X,
  OVERBURDEN_BOTTOM_Y,
  SILL_BED_Y,
} from './tectonicSolver';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface MagmaMaterials {
  forgedPressIron: THREE.MeshStandardNodeMaterial;
  machinedThreadSteel: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  quartzGlass: THREE.MeshStandardNodeMaterial;
  stratifiedRockCore: THREE.MeshStandardNodeMaterial;
  xenolithBasalt: THREE.MeshStandardNodeMaterial;
  chamberMagma: THREE.MeshStandardNodeMaterial;
  dikeMagma: THREE.MeshStandardNodeMaterial;
  sillMagma: THREE.MeshStandardNodeMaterial;
  pahoehoeLobe: THREE.MeshStandardNodeMaterial;
  emberSpark: THREE.MeshBasicNodeMaterial;
  // Shared live uniforms (mutated in-place inside onTick, zero per-frame allocations)
  uTime: { value: number };
  uChamberState: { value: THREE.Vector4 }; // x=pressure, y=compression, z=maxHeat, w=surge
  uDikeHeights: { value: THREE.Vector4 }; // x=h0, y=h1, z=h2, w=dikeTemp
  uDikeThermal: { value: THREE.Vector4 }; // x=coreTemp, y=skinTemp, z=reCrack, w=quenched
  uSillState: { value: THREE.Vector4 }; // x=hw0, y=hw1, z=sillTemp, w=reCrack
  uHoverState: { value: THREE.Vector4 }; // x=pistonHover, y=leftClampHover, z=rightClampHover, w=coreHover
  dispose: () => void;
}

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Uses a scale-invariant dimensionless lattice gradient (no 1/eps frequency blow-up)
 * so `strength_in` directly governs physical surface slope across both sleek metals and rough rock.
 */
const perturbNormal3DFbm = Fn(
  ([baseNorm_in, pos_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const sc = float(scale_in);
    const eps = float(0.16).div(max(float(1.0), sc));

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), sc);
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), sc);
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), sc);
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), sc);

    // Dimensionless lattice gradient (1 / (0.16 * 4.0) = 1.5625) — eliminates sc-proportional foil crinkling
    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

/**
 * Physical magma blackbody mineral color ramp in TSL (matching evalMagmaBlackbody in tectonicSolver.ts).
 * Transitions smoothly from glossy jet-black obsidian (t=0) -> dark hematite rust ->
 * cinnabar vermilion -> bright realgar amber -> incandescent pale-gold core (t=1).
 * Strictly mineral-anchored (zero digital RGB neons) and uniform control flow.
 */
const evalBlackbodyAlbedoTsl = Fn(([temp_in]: [TslNode]) => {
  const t = clamp(float(temp_in), 0.0, 1.0);

  const cObsidian = vec3(0.011, 0.012, 0.015);
  const cHematite = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(vec3(0.55, 0.28, 0.22));
  const cCinnabar = vec3(...MINERAL_PALETTE.cinnabarVermilion.rInfLinear).mul(vec3(1.12, 0.72, 0.25));
  const cRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear).mul(vec3(1.28, 1.08, 0.40));
  const cPaleGold = mix(
    vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear),
    vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear),
    float(0.52)
  ).mul(vec3(1.32, 1.24, 1.02));

  const s1 = smoothstep(float(0.0), float(0.22), t);
  const s2 = smoothstep(float(0.22), float(0.48), t);
  const s3 = smoothstep(float(0.48), float(0.76), t);
  const s4 = smoothstep(float(0.76), float(1.0), t);

  const m1 = mix(cObsidian, cHematite, s1);
  const m2 = mix(m1, cCinnabar, s2);
  const m3 = mix(m2, cRealgar, s3);
  return mix(m3, cPaleGold, s4);
});

/**
 * Physical obsidian-to-blister-to-melt roughness curve:
 *  - Quenched obsidian (t -> 0): glossy volcanic glass (~0.14 roughness)
 *  - Cooling blistering skin (t ~ 0.26): rough scorched clinker (~0.52 roughness)
 *  - Hot fluid melt (t -> 1): smooth viscous sheen (~0.24 roughness)
 */
const evalMagmaRoughnessTsl = Fn(([temp_in]: [TslNode]) => {
  const t = clamp(float(temp_in), 0.0, 1.0);
  const d = t.sub(0.26).div(0.14);
  const blisterBump = exp(d.mul(d).negate()).mul(0.38);
  const meltLift = smoothstep(float(0.45), float(0.95), t).mul(0.10);
  return clamp(float(0.14).add(blisterBump).add(meltLift), 0.13, 0.58);
});

/**
 * Physical Stefan-Boltzmann-inspired thermal emission radiance (strictly 0 on obsidian).
 * Safe positive-base pow() guard prevents WGSL NaN propagation.
 */
const evalBlackbodyEmissiveTsl = Fn(([temp_in, boost_in]: [TslNode, TslNode]) => {
  const t = clamp(float(temp_in), 0.0, 1.0);
  const ramp = smoothstep(float(0.06), float(1.0), t);
  const gain = pow(max(float(0.0001), ramp), float(2.35)).mul(float(boost_in));
  return evalBlackbodyAlbedoTsl(t).mul(gain);
});

export function createMagmaMaterials(): MagmaMaterials {
  const uTime = uniform(0);
  const uChamberState = uniform(new THREE.Vector4(0.56, -0.32, 0.72, 0.0));
  const uDikeHeights = uniform(new THREE.Vector4(0.55, 0.18, 0.18, 0.72));
  const uDikeThermal = uniform(new THREE.Vector4(0.84, 0.62, 0.15, 0.10));
  const uSillState = uniform(new THREE.Vector4(0.24, 0.08, 0.56, 0.08));
  const uHoverState = uniform(new THREE.Vector4(0, 0, 0, 0));

  const pOchre = createPigmentUniforms('yellowOchre');
  const pSienna = createPigmentUniforms('burntSienna');
  const pUmber = createPigmentUniforms('rawUmber');
  const pCalcite = createPigmentUniforms('boneCalcite');

  // -------------------------------------------------------------------------
  // 1. Oil-Blackened Forged Cast-Iron Geology Press Frame
  // -------------------------------------------------------------------------
  const forgedPressIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.84,
    roughness: 0.42,
    envMapIntensity: 1.22,
  });
  {
    const macroScale = mineralToothFbm(positionLocal, float(8.0));
    const microGrain = mineralToothFbm(positionLocal.add(vec3(3.1, 7.4, 2.2)), float(38.0));

    // Tight oil-blackened forged iron palette (silky machined/cast steel, zero salt-and-pepper speckle)
    const ironDeep = vec3(0.022, 0.024, 0.029);
    const ironMid = vec3(0.034, 0.037, 0.044);
    const ironSheen = vec3(0.046, 0.050, 0.058);

    const baseIron = mix(
      mix(ironDeep, ironMid, smoothstep(float(0.26), float(0.76), macroScale)),
      ironSheen,
      smoothstep(float(0.48), float(0.88), microGrain).mul(0.20)
    );

    // Warm subtle under-bounce from glowing lower magma chamber on nearby frame ledges
    const chamberProximity = exp(
      positionWorld.y.add(0.42).mul(positionWorld.y.add(0.42)).mul(-4.5)
    ).mul(exp(positionWorld.x.mul(positionWorld.x).mul(-0.9)));
    const magmaBounce = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear)
      .mul(chamberProximity)
      .mul(float(0.038).add(uChamberState.x.mul(0.055)));

    forgedPressIron.colorNode = baseIron.add(magmaBounce);
    // Subtle cast-iron micro-tooth (<= 0.25 deg normal deflection -> silky softbox reflection roll)
    forgedPressIron.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(30.0),
      float(0.016)
    );
    forgedPressIron.roughnessNode = clamp(
      float(0.39).add(macroScale.mul(0.08)).sub(microGrain.mul(0.04)),
      0.30,
      0.52
    );
  }

  // -------------------------------------------------------------------------
  // 2. Machined Gunmetal Steel (Tie-Rods, ACME Screw Threads, Hydraulic Piston Shaft)
  // -------------------------------------------------------------------------
  const machinedThreadSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.26,
    envMapIntensity: 1.28,
  });
  {
    const latheGrain = mineralToothFbm(positionLocal.mul(vec3(2.2, 18.0, 2.2)), float(8.5));
    const fineGrain = mineralToothFbm(positionLocal.add(vec3(1.4, 5.8, 3.3)), float(48.0));

    const steelDark = vec3(0.115, 0.124, 0.138);
    const steelBright = vec3(0.185, 0.196, 0.212);
    const hoverBoost = max(uHoverState.x, max(uHoverState.y, uHoverState.z)).mul(0.045);

    machinedThreadSteel.colorNode = mix(
      steelDark,
      steelBright,
      smoothstep(float(0.28), float(0.76), latheGrain.mul(0.65).add(fineGrain.mul(0.35)))
    ).add(vec3(hoverBoost));
    // Ultra-fine lathe-turned steel sheen (<= 0.12 deg slope -> crisp axial specular highlights)
    machinedThreadSteel.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(42.0),
      float(0.008)
    );
    machinedThreadSteel.roughnessNode = clamp(
      float(0.22).add(latheGrain.mul(0.07)).add(fineGrain.sub(0.5).mul(0.03)),
      0.17,
      0.33
    );
  }

  // -------------------------------------------------------------------------
  // 3. Turned Satin Brass (Hydraulic Collars, Handwheel Grips, Quartz Window Bezels)
  // -------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.23,
    envMapIntensity: 1.28,
  });
  {
    const brush = mineralToothFbm(positionLocal, float(26.0));
    const fine = mineralToothFbm(positionLocal.add(vec3(4.1, 2.3, 6.7)), float(64.0));

    // Anchored in Yellow Ochre mineral tone shifted to physical turned satin brass reflectance
    const brassAged = vec3(0.54, 0.41, 0.19);
    const brassBright = vec3(0.70, 0.55, 0.26);
    const hoverLift = max(uHoverState.x, max(uHoverState.y, uHoverState.z)).mul(0.085);

    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.24), float(0.76), brush.mul(0.60).add(fine.mul(0.40)))
    ).add(brassBright.mul(hoverLift));
    // Sleek micro-brushed satin brass (<= 0.1 deg slope -> eliminates crumpled gold-foil distortion)
    satinBrass.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(38.0),
      float(0.005)
    );
    satinBrass.roughnessNode = clamp(
      float(0.21).add(brush.mul(0.06)).add(fine.sub(0.5).mul(0.02)),
      0.17,
      0.31
    );
  }

  // -------------------------------------------------------------------------
  // 4. Beveled Laboratory Quartz Cutaway Window Glass (Non-Blooming Dielectric Parallax)
  // -------------------------------------------------------------------------
  const quartzGlass = new THREE.MeshStandardNodeMaterial({
    transparent: true,
    depthWrite: false,
    metalness: 0.04,
    roughness: 0.08,
    envMapIntensity: 1.45,
  });
  {
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDir)), 0.0, 1.0);
    const fresnel = pow(max(float(0.0001), float(1.0).sub(nDotV)), float(3.4));

    // Subtle parallax diagonal overhead studio reflection band across quartz face
    const parallaxCoord = positionLocal.x.mul(1.8).add(positionLocal.y.mul(2.4)).add(viewDir.x.mul(0.65));
    const glareBand = exp(pow(max(float(0.0001), abs(sin(parallaxCoord.mul(1.35)).sub(0.55))), float(2.0)).mul(-18.0));

    // Faint beveled border specular catch near pane perimeter (x in [-0.67, 0.67], y in [-0.16, 0.16])
    const edgeX = smoothstep(float(0.56), float(0.67), abs(positionLocal.x));
    const edgeY = smoothstep(float(0.11), float(0.16), abs(positionLocal.y));
    const edgeCatch = clamp(max(edgeX, edgeY), 0.0, 1.0);

    const coolGlass = vec3(0.58, 0.66, 0.72);
    const warmRim = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear).mul(0.62);

    quartzGlass.colorNode = mix(coolGlass, warmRim, edgeCatch.mul(0.45).add(uChamberState.x.mul(0.20)))
      .add(vec3(0.24, 0.25, 0.27).mul(glareBand.mul(0.45)));
    quartzGlass.opacityNode = clamp(
      float(0.10).add(fresnel.mul(0.32)).add(glareBand.mul(0.11)).add(edgeCatch.mul(0.22)),
      0.09,
      0.52
    );
  }

  // -------------------------------------------------------------------------
  // 5. 3D Object-Space Stratified Basalt & Banded Ochre Tuff Overburden Rock
  //    (Zero 2D UV stretching across vertical fracture walls + live magma seam glow)
  // -------------------------------------------------------------------------
  const stratifiedRockCore = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.84,
    envMapIntensity: 0.85,
  });
  {
    const aRestPos = attribute('aRestPos', 'vec3');
    const aStratum = attribute('aStratum', 'float');
    const aFractureMask = attribute('aFractureMask', 'float');
    const aSlabIdx = attribute('aSlabIdx', 'float');

    // Pure 3D rest-core petrography so bedding lines align across closed joints and visibly offset on heave!
    const warp = mineralToothFbm(aRestPos.mul(vec3(2.4, 3.6, 2.4)), float(2.8));
    const coarseRock = mineralToothFbm(aRestPos.add(vec3(aSlabIdx.mul(0.17), 1.3, 4.1)), float(6.5));
    const fineGrain = mineralToothFbm(aRestPos.add(vec3(9.1, 2.7, 5.4)), float(28.0));
    const vesiclePit = mineralToothFbm(aRestPos.add(vec3(3.7, 8.2, 1.9)), float(56.0));
    const pitMask = smoothstep(float(0.68), float(0.88), vesiclePit);

    // Horizon 0 & 2: Dark charcoal-umber vesicular basalt (Lower bed & Upper columnar cap)
    const basaltDeep = vec3(0.032, 0.030, 0.031);
    const basaltMid = vec3(0.068, 0.062, 0.058);
    const basaltUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.55);
    const basaltHematite = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(0.42);

    const basaltBody = mix(
      mix(basaltDeep, basaltMid, smoothstep(float(0.22), float(0.82), coarseRock)),
      basaltUmber,
      smoothstep(float(0.45), float(0.86), fineGrain).mul(0.32)
    );
    // Upper basalt cap gets subtle warm iron-oxide weathering on exposed top crown facets
    const crownWeather = smoothstep(float(0.38), float(0.85), normalLocal.y).mul(
      smoothstep(float(1.4), float(1.8), aStratum)
    );
    const basaltFinal = mix(basaltBody, basaltHematite, crownWeather.mul(0.28)).mul(
      float(1.0).sub(pitMask.mul(0.36))
    );

    // Horizon 1: Distinct Banded Yellow-Ochre Tuff & Burnt-Sienna Ironstone Sedimentary Strata
    const strataY = aRestPos.y.mul(22.0).add(warp.mul(2.8)).add(aRestPos.x.mul(1.6));
    const majorBand = sin(strataY).mul(0.5).add(0.5);
    const fineBand = sin(strataY.mul(2.75).add(1.4)).mul(0.5).add(0.5);
    const ashSeam = smoothstep(float(0.88), float(0.97), sin(strataY.mul(1.45).add(0.6)).mul(0.5).add(0.5));

    const tuffThickness = float(0.65).add(majorBand.mul(0.75)).add(fineGrain.mul(0.35));
    const umberBacking = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.62);
    const ochreGlaze = kubelkaMunkGlaze(pOchre.K, pOchre.S, tuffThickness, umberBacking);
    const siennaGlaze = kubelkaMunkGlaze(
      pSienna.K,
      pSienna.S,
      float(0.55).add(fineBand.mul(0.85)),
      umberBacking
    );
    const calciteAsh = kubelkaMunkGlaze(
      pCalcite.K,
      pCalcite.S,
      float(0.52).add(fineGrain.mul(0.35)),
      ochreGlaze
    );

    const tuffLaminated = mix(
      mix(ochreGlaze.mul(0.84), siennaGlaze.mul(0.82), smoothstep(float(0.46), float(0.82), fineBand)),
      calciteAsh.mul(0.86),
      ashSeam.mul(0.42)
    ).mul(float(0.84).add(coarseRock.mul(0.28)));

    // Smooth stratum selection: middle tier (aStratum == 1) is Banded Ochre Tuff; tiers 0 & 2 are Basalt
    const isTuffTier = float(1.0).sub(clamp(abs(aStratum.sub(1.0)).mul(2.0), 0.0, 1.0));
    // Thin sedimentary contact transition band right along bedding planes
    const rockAlbedo = mix(basaltFinal, tuffLaminated, isTuffTier);

    // Subtle darkened bedding-plane & fracture-wall patina
    const fractureDarken = float(1.0).sub(aFractureMask.mul(0.24));

    // Real-time incandescent magma bounce & thermal wall heating along active dike/sill fractures
    const yAboveRoof = max(float(0.0), aRestPos.y.sub(float(OVERBURDEN_BOTTOM_Y)));
    const d0Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[0])));
    const d1Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[1])));
    const d2Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[2])));

    const nearD0 = smoothstep(float(0.14), float(0.0), d0Dist).mul(
      smoothstep(uDikeHeights.x.add(0.06), uDikeHeights.x.sub(0.14), yAboveRoof)
    );
    const nearD1 = smoothstep(float(0.12), float(0.0), d1Dist).mul(
      smoothstep(uDikeHeights.y.add(0.06), uDikeHeights.y.sub(0.14), yAboveRoof)
    );
    const nearD2 = smoothstep(float(0.12), float(0.0), d2Dist).mul(
      smoothstep(uDikeHeights.z.add(0.06), uDikeHeights.z.sub(0.14), yAboveRoof)
    );
    const dikeProx = clamp(max(nearD0, max(nearD1, nearD2)), 0.0, 1.0).mul(uDikeHeights.w);

    const nearS0 = smoothstep(float(0.095), float(0.0), abs(aRestPos.y.sub(float(SILL_BED_Y[0])))).mul(
      smoothstep(uSillState.x.add(0.08), max(float(0.01), uSillState.x.sub(0.10)), abs(aRestPos.x))
    );
    const nearS1 = smoothstep(float(0.095), float(0.0), abs(aRestPos.y.sub(float(SILL_BED_Y[1])))).mul(
      smoothstep(uSillState.y.add(0.08), max(float(0.01), uSillState.y.sub(0.10)), abs(aRestPos.x))
    );
    const sillProx = clamp(max(nearS0, nearS1), 0.0, 1.0).mul(uSillState.z);

    // Base chamber roof thermal conduction along bottom of Stratum 0
    const roofProx = smoothstep(float(CHAMBER_ROOF_Y + 0.13), float(CHAMBER_ROOF_Y), aRestPos.y).mul(
      float(0.38).add(uChamberState.x.mul(0.62))
    );

    const wallHeat = clamp(
      max(max(dikeProx, sillProx), roofProx.mul(0.75)).mul(float(0.32).add(aFractureMask.mul(0.95))),
      0.0,
      1.0
    );

    stratifiedRockCore.colorNode = rockAlbedo.mul(fractureDarken);
    // Warm volcanic thermal bounce & scorched cinnabar/amber edge glow inside opening fractures
    stratifiedRockCore.emissiveNode = evalBlackbodyEmissiveTsl(
      clamp(wallHeat.mul(0.78), 0.0, 0.82),
      float(0.72)
    ).mul(float(0.45).add(fineGrain.mul(0.55)));

    // Crisp 3D chiseled mineral micro-normal (rougher on vesicular basalt, layered tooth on tuff)
    stratifiedRockCore.normalNode = perturbNormal3DFbm(
      normalLocal,
      aRestPos,
      mix(float(22.0), float(30.0), isTuffTier),
      mix(float(0.52), float(0.36), isTuffTier)
    );
    stratifiedRockCore.roughnessNode = clamp(
      mix(float(0.84), float(0.74), isTuffTier)
        .add(pitMask.mul(0.10))
        .sub(coarseRock.mul(0.08)),
      0.62,
      0.95
    );
  }

  // -------------------------------------------------------------------------
  // 5b. Suspended Dark Vesicular Basalt Xenoliths Inside Lower Magma Chamber
  // -------------------------------------------------------------------------
  const xenolithBasalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.06,
    roughness: 0.78,
    envMapIntensity: 0.75,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(18.0));
    const rim = float(1.0).sub(smoothstep(float(-0.08), float(0.10), positionLocal.y));
    xenolithBasalt.colorNode = mix(
      vec3(0.022, 0.020, 0.022),
      vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.38),
      grain
    );
    xenolithBasalt.emissiveNode = evalBlackbodyEmissiveTsl(
      clamp(rim.mul(0.62).add(uChamberState.x.mul(0.24)), 0.0, 0.85),
      float(0.85)
    );
    xenolithBasalt.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(24.0), float(0.45));
  }

  // -------------------------------------------------------------------------
  // 6. Churning Incandescent Lower Magma Chamber Melt (Visible Through Quartz Windows)
  // -------------------------------------------------------------------------
  const chamberMagma = new THREE.MeshStandardNodeMaterial({
    metalness: 0.06,
    roughness: 0.28,
    envMapIntensity: 1.05,
  });
  {
    // Convective upward thermal plumes driven by uTime & chamberPressure
    const flowSpeed = float(0.22).add(uChamberState.x.mul(0.48));
    const convPos = positionLocal.add(vec3(float(0.0), uTime.mul(flowSpeed).negate(), float(0.0)));
    const plumeCoarse = mineralToothFbm(convPos, float(3.4));
    const plumeFine = mineralToothFbm(
      convPos.mul(vec3(1.6, 1.1, 1.6)).add(vec3(plumeCoarse.mul(0.8), 2.3, 1.7)),
      float(7.2)
    );

    // Hotter incandescent convection cells at chamber center & bottom piston contact, cooler dark slag crusts near roof
    const pressureLift = float(0.48).add(uChamberState.x.mul(0.40));
    const cellHeat = clamp(
      pressureLift.mul(float(0.56).add(plumeCoarse.mul(0.42)).add(plumeFine.mul(0.22))),
      0.14,
      0.94
    );

    chamberMagma.colorNode = evalBlackbodyAlbedoTsl(cellHeat);
    chamberMagma.emissiveNode = evalBlackbodyEmissiveTsl(cellHeat, float(1.68));
    chamberMagma.roughnessNode = evalMagmaRoughnessTsl(cellHeat);
    chamberMagma.normalNode = perturbNormal3DFbm(normalLocal, convPos, float(8.5), float(0.28));
  }

  // -------------------------------------------------------------------------
  // 7. Dynamic Magma Intrusion Material Factory (Vertical Dikes & Horizontal Sills)
  //    Models hot centerlines, chilled obsidian margins, and live re-cracking seams!
  // -------------------------------------------------------------------------
  function createIntrusionMaterial(isSill: boolean): THREE.MeshStandardNodeMaterial {
    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.08,
      roughness: 0.25,
      envMapIntensity: 1.45,
    });

    const coreTempNode = isSill ? uSillState.z : uDikeThermal.x;
    const skinTempNode = isSill ? uSillState.z.mul(0.82) : uDikeThermal.y;
    const reCrackNode = isSill ? uSillState.w : uDikeThermal.z;

    // Advect molten magma upward along dikes or radially outward along sills
    const flowOffset = isSill
      ? vec3(abs(positionWorld.x).sub(uTime.mul(0.28)), positionWorld.y, positionWorld.z)
      : vec3(positionWorld.x, positionWorld.y.sub(uTime.mul(0.36)), positionWorld.z);

    const fbmMacro = mineralToothFbm(flowOffset, float(5.5));
    const fbmMicro = mineralToothFbm(flowOffset.add(vec3(fbmMacro.mul(0.65), 3.2, 5.1)), float(14.0));

    // Ridge crack pattern: sharp incandescent fissures splitting dark obsidian crust
    const ridgeField = float(1.0).sub(abs(fbmMicro.mul(2.0).sub(1.0)));
    const crackLine = smoothstep(float(0.68).sub(reCrackNode.mul(0.24)), float(0.92), ridgeField);

    // Chilled margins along exposed front/back face edges preserve rich cinnabar/amber marbling without white clip
    const marginCool = smoothstep(float(0.36), float(0.54), abs(positionWorld.z)).mul(0.28);
    const crustThermal = clamp(
      skinTempNode.mul(0.88).sub(marginCool).add(fbmMacro.sub(0.5).mul(0.34)),
      0.0,
      0.86
    );
    const fissureTemp = clamp(max(coreTempNode.mul(0.92), skinTempNode.add(0.22)), 0.0, 0.94);
    const crackWeight = clamp(
      crackLine.mul(float(0.42).add(reCrackNode.mul(0.82)).add(uChamberState.w.mul(0.42))),
      0.0,
      1.0
    );

    const finalTemp = mix(crustThermal, fissureTemp, crackWeight);

    mat.colorNode = evalBlackbodyAlbedoTsl(finalTemp);
    // Restrained HDR emissive (1.58) preserves rich realgar-amber & cinnabar convection detail at peak pressure
    mat.emissiveNode = evalBlackbodyEmissiveTsl(finalTemp, float(1.58));
    mat.roughnessNode = evalMagmaRoughnessTsl(finalTemp);
    mat.normalNode = perturbNormal3DFbm(normalLocal, flowOffset, float(14.5), float(0.34));

    return mat;
  }

  const dikeMagma = createIntrusionMaterial(false);
  const sillMagma = createIntrusionMaterial(true);

  // -------------------------------------------------------------------------
  // 8. Instanced Viscoplastic Pahoehoe Magma Lobes (Live Obsidian Quenching & Re-Cracking)
  //    Reads per-instance aLobeThermal = vec4(coreTemp, crustTemp, crackIntensity, seed)
  //    Zero multiplicative positionNode scaling (strict Three.js r187 InstancedMesh contract!)
  // -------------------------------------------------------------------------
  const pahoehoeLobe = new THREE.MeshStandardNodeMaterial({
    metalness: 0.09,
    roughness: 0.16,
    envMapIntensity: 1.55,
  });
  {
    const aLobeThermal = attribute('aLobeThermal', 'vec4');
    const coreTemp = clamp(aLobeThermal.x, 0.0, 1.0);
    const crustTemp = clamp(aLobeThermal.y, 0.0, 1.0);
    const crackIntensity = clamp(aLobeThermal.z, 0.0, 1.0);
    const seed = aLobeThermal.w;

    // Pure un-instanced geometry coordinates + per-lobe seed so every pillow has crisp ropy wrinkles
    const pObj = positionGeometry.add(vec3(seed.mul(1.7), seed.mul(2.3), seed.mul(0.9)));
    const flowCrawl = uTime.mul(float(0.08).add(crustTemp.mul(0.24)));
    const pRopy = pObj.mul(vec3(1.8, 3.2, 1.8)).add(vec3(0.0, flowCrawl.negate(), 0.0));

    // Dual-scale irrational-angle FBM for ropy pahoehoe skin folds & obsidian shrinkage cracks
    const foldCoarse = mineralToothFbm(pRopy, float(2.8));
    const foldFine = mineralToothFbm(pRopy.add(vec3(foldCoarse.mul(0.75), 1.9, 4.3)), float(6.4));

    // Concentrated fracture ridge cracks that split open across dark glossy obsidian when re-pressurized
    const ridgeVal = float(1.0).sub(abs(foldFine.mul(2.0).sub(1.0)));
    const crackThresh = clamp(float(0.84).sub(crackIntensity.mul(0.32)).sub(crustTemp.mul(0.18)), 0.36, 0.88);
    const crackMask = smoothstep(crackThresh, float(0.94), ridgeVal);

    // Underside/contact seam stays hotter longer while top exposed dome quenches fastest to black obsidian
    const topExposure = smoothstep(float(-0.25), float(0.65), normalLocal.y);
    const localSkinTemp = clamp(
      crustTemp.mul(float(1.08).sub(topExposure.mul(0.36))).add(foldCoarse.sub(0.5).mul(0.20)),
      0.0,
      0.90
    );

    // Re-cracked fissures expose molten interior coreTemp (bright cinnabar -> realgar amber)
    const exposedCrackHeat = clamp(
      max(coreTemp, localSkinTemp.add(0.25)).mul(float(0.40).add(crackIntensity.mul(0.72))),
      0.0,
      0.94
    );
    const finalTemp = mix(localSkinTemp, exposedCrackHeat, crackMask);

    // Glossy jet-black obsidian glass sheen (roughness ~ 0.14) on cooled unbroken skin,
    // contrasting sharply with blazing matte-viscous cinnabar cracks!
    const baseRough = evalMagmaRoughnessTsl(finalTemp);
    const obsidianGlassBoost = smoothstep(float(0.24), float(0.05), finalTemp).mul(float(1.0).sub(crackMask));
    const finalRough = clamp(mix(baseRough, float(0.135), obsidianGlassBoost), 0.13, 0.56);

    pahoehoeLobe.colorNode = evalBlackbodyAlbedoTsl(finalTemp);
    pahoehoeLobe.emissiveNode = evalBlackbodyEmissiveTsl(finalTemp, float(1.75));
    pahoehoeLobe.roughnessNode = finalRough;
    pahoehoeLobe.normalNode = perturbNormal3DFbm(
      normalLocal,
      pRopy,
      float(5.2),
      mix(float(0.42), float(0.18), obsidianGlassBoost)
    );
  }

  // -------------------------------------------------------------------------
  // 9. Restrained Eruptive Incandescent Ash / Spatter Ember Motes
  // -------------------------------------------------------------------------
  const emberSpark = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  });
  {
    const aEmberHeat = attribute('aEmberHeat', 'float');
    const c = evalBlackbodyAlbedoTsl(aEmberHeat).add(evalBlackbodyEmissiveTsl(aEmberHeat, float(1.45)));
    emberSpark.colorNode = c;
    emberSpark.opacityNode = smoothstep(float(0.02), float(0.25), aEmberHeat);
  }

  // Keep unused pigment handle clean in strict TS
  void pUmber;

  return {
    forgedPressIron,
    machinedThreadSteel,
    satinBrass,
    quartzGlass,
    stratifiedRockCore,
    xenolithBasalt,
    chamberMagma,
    dikeMagma,
    sillMagma,
    pahoehoeLobe,
    emberSpark,
    uTime,
    uChamberState,
    uDikeHeights,
    uDikeThermal,
    uSillState,
    uHoverState,
    dispose: () => {
      forgedPressIron.dispose();
      machinedThreadSteel.dispose();
      satinBrass.dispose();
      quartzGlass.dispose();
      stratifiedRockCore.dispose();
      xenolithBasalt.dispose();
      chamberMagma.dispose();
      dikeMagma.dispose();
      sillMagma.dispose();
      pahoehoeLobe.dispose();
      emberSpark.dispose();
    },
  };
}
