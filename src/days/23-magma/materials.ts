import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  atan,
  attribute,
  cameraPosition,
  clamp,
  cos,
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
  sqrt,
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
  gaugeDial: THREE.MeshStandardNodeMaterial;
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

    // Dimensionless lattice gradient (1 / (0.16 * 4.0) = 1.5625)
    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

/**
 * Dual-scale rock micro-normal perturbation combining medium mineral cleavage facets
 * with crisp high-frequency porous vesicular pits / sedimentary grit (strictly uniform flow).
 */
const perturbRockNormalDual = Fn(
  ([
    baseNorm_in,
    pos_in,
    scaleCoarse_in,
    strCoarse_in,
    scaleFine_in,
    strFine_in,
  ]: [TslNode, TslNode, TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const sc1 = float(scaleCoarse_in);
    const e1 = float(0.16).div(max(float(1.0), sc1));
    const c1 = mineralToothFbm(p.add(k1.mul(e1)), sc1);
    const c2 = mineralToothFbm(p.add(k2.mul(e1)), sc1);
    const c3 = mineralToothFbm(p.add(k3.mul(e1)), sc1);
    const c4 = mineralToothFbm(p.add(k4.mul(e1)), sc1);
    const g1 = k1.mul(c1).add(k2.mul(c2)).add(k3.mul(c3)).add(k4.mul(c4)).mul(1.5625);

    const sc2 = float(scaleFine_in);
    const e2 = float(0.16).div(max(float(1.0), sc2));
    const pFine = p.add(vec3(4.1, 7.3, 2.9));
    const f1 = mineralToothFbm(pFine.add(k1.mul(e2)), sc2);
    const f2 = mineralToothFbm(pFine.add(k2.mul(e2)), sc2);
    const f3 = mineralToothFbm(pFine.add(k3.mul(e2)), sc2);
    const f4 = mineralToothFbm(pFine.add(k4.mul(e2)), sc2);
    // Sharpen pit rims for vesicular basalt & clastic grit pores
    const s1 = smoothstep(float(0.54), float(0.86), f1);
    const s2 = smoothstep(float(0.54), float(0.86), f2);
    const s3 = smoothstep(float(0.54), float(0.86), f3);
    const s4 = smoothstep(float(0.54), float(0.86), f4);
    const g2 = k1.mul(s1).add(k2.mul(s2)).add(k3.mul(s3)).add(k4.mul(s4)).mul(1.5625);

    const totalGrad = g1.mul(float(strCoarse_in)).add(g2.mul(float(strFine_in)));
    const tanGrad = totalGrad.sub(n.mul(dot(totalGrad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad)));
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
 *  - Quenched obsidian (t -> 0): glossy volcanic glass (~0.135 roughness)
 *  - Cooling blistering skin (t ~ 0.26): rough scorched clinker (~0.52 roughness)
 *  - Hot fluid melt (t -> 1): smooth viscous sheen (~0.24 roughness)
 */
const evalMagmaRoughnessTsl = Fn(([temp_in]: [TslNode]) => {
  const t = clamp(float(temp_in), 0.0, 1.0);
  const d = t.sub(0.26).div(0.14);
  const blisterBump = exp(d.mul(d).negate()).mul(0.38);
  const meltLift = smoothstep(float(0.45), float(0.95), t).mul(0.10);
  return clamp(float(0.135).add(blisterBump).add(meltLift), 0.125, 0.58);
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
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pCalcite = createPigmentUniforms('boneCalcite');

  // -------------------------------------------------------------------------
  // 1. Oil-Blackened Forged & Milled Gunmetal Steel Geology Press Frame
  //    Calibrated metallic value range so bevels, crosshead yokes & milled base
  //    platen read heavy and steel-like under studio PMREM IBL (never flat black).
  // -------------------------------------------------------------------------
  const forgedPressIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.78,
    roughness: 0.38,
    envMapIntensity: 1.38,
  });
  {
    const macroScale = mineralToothFbm(positionLocal, float(7.5));
    // Fine directional milled-steel tool marks along X + cast peening grain
    const milledBrush = mineralToothFbm(positionLocal.mul(vec3(3.5, 24.0, 22.0)), float(6.5));
    const microPeen = mineralToothFbm(positionLocal.add(vec3(3.1, 7.4, 2.2)), float(42.0));

    // Authentic oil-blackened gunmetal cast-iron / tool-steel reflectance
    const ironDeep = vec3(0.048, 0.052, 0.060);
    const ironMid = vec3(0.074, 0.079, 0.090);
    const ironMachined = vec3(0.110, 0.118, 0.130);

    const baseIron = mix(
      mix(ironDeep, ironMid, smoothstep(float(0.24), float(0.76), macroScale)),
      ironMachined,
      smoothstep(float(0.42), float(0.86), milledBrush.mul(0.65).add(microPeen.mul(0.35))).mul(0.38)
    );

    // Subtle upward-facing machined-table polish (gives top edges of beams & base bed metallic definition)
    const topPolish = smoothstep(float(0.65), float(0.95), abs(normalLocal.y)).mul(0.010);

    // Restrained warm hearth bounce strictly on chamber casing walls (not washing out the whole base bed)
    const dyChamber = positionWorld.y.add(0.42);
    const chamberProximity = exp(dyChamber.mul(dyChamber).mul(-12.0))
      .mul(exp(positionWorld.x.mul(positionWorld.x).mul(-1.4)));
    const magmaBounce = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear)
      .mul(chamberProximity)
      .mul(float(0.022).add(uChamberState.x.mul(0.036)));

    forgedPressIron.colorNode = baseIron.add(vec3(topPolish)).add(magmaBounce);
    // Crisp cast/milled iron micro-normal texture so studio softbox speculars roll authentically
    forgedPressIron.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal.mul(vec3(1.0, 1.8, 1.4)),
      float(34.0),
      float(0.028)
    );
    forgedPressIron.roughnessNode = clamp(
      float(0.36).add(macroScale.mul(0.09)).sub(milledBrush.mul(0.06)).sub(topPolish.mul(1.8)),
      0.26,
      0.49
    );
  }

  // -------------------------------------------------------------------------
  // 2. Machined Gunmetal Steel (Tie-Rods, Helical ACME Screw Threads, Piston Rod)
  // -------------------------------------------------------------------------
  const machinedThreadSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.24,
    envMapIntensity: 1.40,
  });
  {
    const latheGrain = mineralToothFbm(positionLocal.mul(vec3(2.4, 22.0, 2.4)), float(9.0));
    const fineGrain = mineralToothFbm(positionLocal.add(vec3(1.4, 5.8, 3.3)), float(52.0));

    const steelDark = vec3(0.155, 0.166, 0.184);
    const steelBright = vec3(0.255, 0.268, 0.288);
    const hoverBoost = max(uHoverState.x, max(uHoverState.y, uHoverState.z)).mul(0.05);

    machinedThreadSteel.colorNode = mix(
      steelDark,
      steelBright,
      smoothstep(float(0.26), float(0.76), latheGrain.mul(0.65).add(fineGrain.mul(0.35)))
    ).add(vec3(hoverBoost));
    machinedThreadSteel.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal.mul(vec3(1.0, 2.5, 1.0)),
      float(46.0),
      float(0.011)
    );
    machinedThreadSteel.roughnessNode = clamp(
      float(0.20).add(latheGrain.mul(0.07)).add(fineGrain.sub(0.5).mul(0.03)),
      0.16,
      0.31
    );
  }

  // -------------------------------------------------------------------------
  // 3. Turned Satin Brass (Hydraulic Collars, Bourdon Gauge Bezel, Handwheel Grips)
  // -------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.89,
    roughness: 0.22,
    envMapIntensity: 1.38,
  });
  {
    const brush = mineralToothFbm(positionLocal, float(28.0));
    const fine = mineralToothFbm(positionLocal.add(vec3(4.1, 2.3, 6.7)), float(66.0));

    // Anchored in Yellow Ochre mineral tone shifted to physical turned satin brass reflectance
    const brassAged = vec3(0.56, 0.43, 0.20);
    const brassBright = vec3(0.74, 0.59, 0.29);
    const hoverLift = max(uHoverState.x, max(uHoverState.y, uHoverState.z)).mul(0.085);

    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.24), float(0.76), brush.mul(0.60).add(fine.mul(0.40)))
    ).add(brassBright.mul(hoverLift));
    satinBrass.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(42.0),
      float(0.007)
    );
    satinBrass.roughnessNode = clamp(
      float(0.19).add(brush.mul(0.06)).add(fine.sub(0.5).mul(0.02)),
      0.15,
      0.30
    );
  }

  // -------------------------------------------------------------------------
  // 3b. Diegetic Bourdon Pressure Gauge Enamel Dial Face (Mineral Calcite + Cinnabar Redline)
  // -------------------------------------------------------------------------
  const gaugeDial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.46,
    envMapIntensity: 0.85,
  });
  {
    // Local cylinder cap lies in XZ plane (y = +h/2) or rotated so radial coords are (positionLocal.x, positionLocal.z)
    const r = sqrt(
      positionLocal.x.mul(positionLocal.x).add(positionLocal.z.mul(positionLocal.z))
    );
    const rNorm = clamp(r.div(0.072), 0.0, 1.0);
    // Angle sweeping 240 deg from bottom-left (-150 deg) clockwise across top to bottom-right (+150 deg)
    const ang = atan(positionLocal.z, positionLocal.x);
    const tooth = mineralToothFbm(positionLocal, float(85.0));

    const enamelPaper = mix(
      vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear),
      vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear),
      float(0.55).add(tooth.sub(0.5).mul(0.18))
    );
    const inkDark = vec3(0.032, 0.030, 0.028);
    const cinnabarRed = vec3(...MINERAL_PALETTE.cinnabarVermilion.rInfLinear);

    // Outer calibrated track ring & radial graduation ticks
    const tickBand = smoothstep(float(0.66), float(0.70), rNorm).mul(
      smoothstep(float(0.92), float(0.88), rNorm)
    );
    const majorBand = smoothstep(float(0.58), float(0.62), rNorm).mul(
      smoothstep(float(0.92), float(0.88), rNorm)
    );
    const ringLine = smoothstep(float(0.88), float(0.91), rNorm).mul(
      smoothstep(float(0.95), float(0.92), rNorm)
    );
    // Exclude bottom dead-zone sector (where pressure scale wraps)
    const inScaleArc = smoothstep(float(0.58), float(0.48), positionLocal.z.div(max(float(0.001), r)));

    const minorTick = smoothstep(float(0.84), float(0.96), cos(ang.mul(24.0))).mul(tickBand).mul(inScaleArc);
    const majorTick = smoothstep(float(0.94), float(0.99), cos(ang.mul(6.0))).mul(majorBand).mul(inScaleArc);
    const tickMask = clamp(max(max(minorTick, majorTick), ringLine.mul(0.75)), 0.0, 1.0);

    // Overpressure mineral-cinnabar danger arc on right-hand quadrant (+X, slight +Z)
    const redSector = smoothstep(float(0.18), float(0.32), positionLocal.x.div(max(float(0.001), r)))
      .mul(smoothstep(float(-0.45), float(-0.20), positionLocal.z.div(max(float(0.001), r))))
      .mul(inScaleArc)
      .mul(smoothstep(float(0.74), float(0.78), rNorm))
      .mul(smoothstep(float(0.89), float(0.85), rNorm));

    // Recessed bezel shadow around perimeter
    const rimShade = float(1.0).sub(smoothstep(float(0.82), float(1.0), rNorm).mul(0.38));

    const withRed = mix(enamelPaper, cinnabarRed, redSector.mul(0.88));
    gaugeDial.colorNode = mix(withRed, inkDark, tickMask.mul(0.85)).mul(rimShade);
  }

  // -------------------------------------------------------------------------
  // 4. Beveled Laboratory Quartz Cutaway Window Glass (Non-Blooming Dielectric Parallax)
  // -------------------------------------------------------------------------
  const quartzGlass = new THREE.MeshStandardNodeMaterial({
    transparent: true,
    depthWrite: false,
    metalness: 0.04,
    roughness: 0.07,
    envMapIntensity: 1.50,
  });
  {
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDir)), 0.0, 1.0);
    const fresnel = pow(max(float(0.0001), float(1.0).sub(nDotV)), float(3.4));

    // Subtle parallax diagonal overhead studio reflection band across quartz face
    const parallaxCoord = positionLocal.x.mul(1.8).add(positionLocal.y.mul(2.4)).add(viewDir.x.mul(0.65));
    const glareBand = exp(pow(max(float(0.0001), abs(sin(parallaxCoord.mul(1.35)).sub(0.55))), float(2.0)).mul(-18.0));

    // Crisp beveled border specular catch near pane perimeter (x in [-0.67, 0.67], y in [-0.145, 0.145])
    const edgeX = smoothstep(float(0.58), float(0.67), abs(positionLocal.x));
    const edgeY = smoothstep(float(0.105), float(0.145), abs(positionLocal.y));
    const edgeCatch = clamp(max(edgeX, edgeY), 0.0, 1.0);

    const coolGlass = vec3(0.60, 0.68, 0.75);
    const warmRim = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear).mul(0.58);

    quartzGlass.colorNode = mix(coolGlass, warmRim, edgeCatch.mul(0.42).add(uChamberState.x.mul(0.16)))
      .add(vec3(0.26, 0.27, 0.29).mul(glareBand.mul(0.42)));
    quartzGlass.opacityNode = clamp(
      float(0.09).add(fresnel.mul(0.30)).add(glareBand.mul(0.11)).add(edgeCatch.mul(0.24)),
      0.08,
      0.50
    );
  }

  // -------------------------------------------------------------------------
  // 5. 3D Object-Space Stratified Vesicular Basalt & Pyroclastic Ochre Tuff Rock
  //    - Zero 2D UV stretching across vertical slab faces & fracture cleavage walls
  //    - True dual-scale vesicular basalt pits + angular cleavage micro-normals
  //    - Authentic fine-laminated volcanic yellow-ochre tuff & kaolin ash (zero bacon stripes!)
  //    - Contact-metamorphism vitrified black char halo along magma-filled seams
  // -------------------------------------------------------------------------
  const stratifiedRockCore = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.85,
    envMapIntensity: 0.90,
  });
  {
    const aRestPos = attribute('aRestPos', 'vec3');
    const aStratum = attribute('aStratum', 'float');
    const aFractureMask = attribute('aFractureMask', 'float');
    const aSlabIdx = attribute('aSlabIdx', 'float');

    // Pure 3D object-space rest-core petrography so bedding & grain align across closed joints
    const coarseRock = mineralToothFbm(aRestPos.add(vec3(aSlabIdx.mul(0.11), 1.3, 4.1)), float(7.2));
    const medGrain = mineralToothFbm(aRestPos.add(vec3(9.1, 2.7, 5.4)), float(24.0));
    const fineGrain = mineralToothFbm(aRestPos.add(vec3(2.3, 6.9, 8.1)), float(58.0));

    // Dual-scale vesicular basalt gas-bubble pits (medium vesicles + fine micro-pores)
    const vesMed = mineralToothFbm(aRestPos.add(vec3(3.7, 8.2, 1.9)), float(36.0));
    const vesFine = mineralToothFbm(aRestPos.add(vec3(11.4, 2.1, 6.8)), float(76.0));
    const pitMed = smoothstep(float(0.65), float(0.84), vesMed);
    const pitFine = smoothstep(float(0.66), float(0.85), vesFine);
    const pitMask = clamp(pitMed.mul(0.72).add(pitFine.mul(0.45)), 0.0, 1.0);

    // Subtle tiny plagioclase feldspar mineral micro-flecks inside dark basalt matrix
    const mineralFleck = smoothstep(float(0.80), float(0.92), fineGrain).mul(0.18);

    // Horizon 0 & 2: Dark Charcoal-Slate Vesicular Columnar Basalt
    const basaltDeep = vec3(0.030, 0.030, 0.033);
    const basaltMid = vec3(0.064, 0.061, 0.060);
    const basaltUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.48);
    const basaltHematite = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(0.36);
    const basaltPlag = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear).mul(0.35);

    const basaltBody = mix(
      mix(basaltDeep, basaltMid, smoothstep(float(0.22), float(0.80), coarseRock)),
      basaltUmber,
      smoothstep(float(0.44), float(0.84), medGrain).mul(0.26)
    ).add(basaltPlag.mul(mineralFleck));

    // Subtle rust-ochre iron weathering only on exposed upper crown top facets
    const crownWeather = smoothstep(float(0.45), float(0.88), normalLocal.y).mul(
      smoothstep(float(1.4), float(1.8), aStratum)
    );
    const basaltFinal = mix(basaltBody, basaltHematite, crownWeather.mul(0.22)).mul(
      float(1.0).sub(pitMask.mul(0.56))
    );

    // Horizon 1: Authentic Pyroclastic Yellow-Ochre Volcanic Tuff & Ignimbrite Strata
    // Built from multi-frequency domain-warped 3D FBM beds + fine ash laminae + dark lithic clasts
    const bedWarpCoarse = mineralToothFbm(aRestPos.mul(vec3(1.6, 2.8, 1.6)), float(2.2));
    const bedWarpFine = mineralToothFbm(aRestPos.mul(vec3(4.5, 6.2, 4.5)), float(3.4));
    const yBed = aRestPos.y
      .add(bedWarpCoarse.sub(0.5).mul(0.048))
      .add(bedWarpFine.sub(0.5).mul(0.014))
      .add(aRestPos.x.mul(0.028));

    // Non-periodic laminated bedding planes (irregular bed thicknesses — zero cartoon sine stripes!)
    const bedBroad = mineralToothFbm(vec3(aRestPos.x.mul(0.45), yBed.mul(14.5), aRestPos.z.mul(0.45)), float(2.3));
    const bedFine = mineralToothFbm(vec3(aRestPos.x.mul(0.65), yBed.mul(44.0).add(3.7), aRestPos.z.mul(0.65)), float(2.1));
    const bedMicro = mineralToothFbm(vec3(aRestPos.x.mul(0.80), yBed.mul(92.0).add(7.1), aRestPos.z.mul(0.80)), float(1.9));

    // Razor-thin dark ironstone/manganese parting lines & pale chalky kaolin tuff ash horizons
    const ironParting = smoothstep(float(0.71), float(0.85), bedFine).mul(
      smoothstep(float(0.35), float(0.78), bedBroad)
    );
    const fineSiltStreak = smoothstep(float(0.66), float(0.84), bedMicro);
    const kaolinHorizon = smoothstep(float(0.62), float(0.82), float(1.0).sub(bedBroad));

    // Embedded dark volcanic pumice/basalt lithic lapilli specks scattered in 3D inside the tuff
    const lapilliNoise = mineralToothFbm(aRestPos.add(vec3(5.3, 1.7, 8.9)), float(46.0));
    const lapilliMask = smoothstep(float(0.72), float(0.87), lapilliNoise);
    const tuffPoreMask = smoothstep(float(0.70), float(0.88), vesFine);

    // Physical 2-flux Kubelka-Munk mineral glazes calibrated to earthy geological tuff values
    const umberBacking = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.52);
    const ochreThickness = float(0.44).add(bedBroad.mul(0.55)).add(fineGrain.mul(0.22));
    const ochreTuff = kubelkaMunkGlaze(pOchre.K, pOchre.S, ochreThickness, umberBacking);
    const kaolinAsh = kubelkaMunkGlaze(
      pKaolin.K,
      pKaolin.S,
      float(0.36).add(kaolinHorizon.mul(0.45)),
      ochreTuff.mul(0.82)
    );
    const siennaSeam = kubelkaMunkGlaze(
      pSienna.K,
      pSienna.S,
      float(0.42).add(ironParting.mul(0.65)),
      umberBacking.mul(0.72)
    );
    const umberSeam = kubelkaMunkGlaze(
      pUmber.K,
      pUmber.S,
      float(0.55).add(fineSiltStreak.mul(0.50)),
      basaltDeep
    );

    // Layer dusty yellow-ochre tuff -> pale kaolin ash -> thin sienna/umber ironstone partings -> dark lapilli
    const tuffBase = mix(ochreTuff.mul(0.78), kaolinAsh.mul(0.76), kaolinHorizon.mul(0.58));
    const tuffBanded = mix(tuffBase, siennaSeam.mul(0.72), ironParting.mul(0.56));
    const tuffMicroLaminated = mix(tuffBanded, umberSeam.mul(0.68), fineSiltStreak.mul(0.32));
    const tuffLaminated = mix(tuffMicroLaminated, basaltDeep.mul(1.35), lapilliMask.mul(0.55))
      .mul(float(0.86).add(coarseRock.mul(0.24)))
      .mul(float(1.0).sub(tuffPoreMask.mul(0.26)));

    // Smooth stratum selection: middle tier (aStratum == 1) is Banded Volcanic Tuff; tiers 0 & 2 are Basalt
    const isTuffTier = float(1.0).sub(clamp(abs(aStratum.sub(1.0)).mul(2.0), 0.0, 1.0));
    const rockAlbedo = mix(basaltFinal, tuffLaminated, isTuffTier);

    // Real-time magma thermal wall proximity + contact-metamorphism scorched char aureole
    const yAboveRoof = max(float(0.0), aRestPos.y.sub(float(OVERBURDEN_BOTTOM_Y)));
    const d0Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[0])));
    const d1Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[1])));
    const d2Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[2])));

    // Dike proximity (tracks active molten dikes AND retains scorched char margins after quenching)
    const dikeCharGain = clamp(max(uDikeHeights.w, uDikeThermal.w.mul(0.85)), 0.0, 1.0);
    const reach0 = smoothstep(uDikeHeights.x.add(0.05), uDikeHeights.x.sub(0.14), yAboveRoof);
    const reach1 = smoothstep(uDikeHeights.y.add(0.05), uDikeHeights.y.sub(0.14), yAboveRoof);
    const reach2 = smoothstep(uDikeHeights.z.add(0.05), uDikeHeights.z.sub(0.14), yAboveRoof);

    const nearD0 = smoothstep(float(0.11), float(0.0), d0Dist).mul(reach0);
    const nearD1 = smoothstep(float(0.095), float(0.0), d1Dist).mul(reach1);
    const nearD2 = smoothstep(float(0.095), float(0.0), d2Dist).mul(reach2);
    const dikeGeoProx = clamp(max(nearD0, max(nearD1, nearD2)), 0.0, 1.0);
    const dikeProx = dikeGeoProx.mul(uDikeHeights.w);

    const nearS0 = smoothstep(float(0.078), float(0.0), abs(aRestPos.y.sub(float(SILL_BED_Y[0])))).mul(
      smoothstep(uSillState.x.add(0.06), max(float(0.01), uSillState.x.sub(0.10)), abs(aRestPos.x))
    );
    const nearS1 = smoothstep(float(0.078), float(0.0), abs(aRestPos.y.sub(float(SILL_BED_Y[1])))).mul(
      smoothstep(uSillState.y.add(0.06), max(float(0.01), uSillState.y.sub(0.10)), abs(aRestPos.x))
    );
    const sillGeoProx = clamp(max(nearS0, nearS1), 0.0, 1.0);
    const sillProx = sillGeoProx.mul(uSillState.z);

    // Base chamber roof thermal conduction along bottom of Stratum 0
    const roofProx = smoothstep(float(CHAMBER_ROOF_Y + 0.11), float(CHAMBER_ROOF_Y), aRestPos.y).mul(
      float(0.32).add(uChamberState.x.mul(0.68))
    );

    // Scorched charcoal-vitrified thermal margin right along intruded fracture seams (visible on front & walls!)
    const charBand = clamp(
      max(dikeGeoProx.mul(dikeCharGain), sillGeoProx.mul(max(uSillState.z, uDikeThermal.w.mul(0.5))))
        .mul(float(0.75).add(medGrain.mul(0.50))),
      0.0,
      1.0
    );
    const innerChar = smoothstep(float(0.32), float(0.82), charBand);
    const bakedRustRim = smoothstep(float(0.12), float(0.45), charBand).mul(float(1.0).sub(innerChar));

    // Darken fracture-wall patina & bake contact-metamorphism char margin onto tuff and basalt
    const fractureDarken = float(1.0).sub(aFractureMask.mul(0.28));
    const rustBakedAlbedo = mix(
      rockAlbedo,
      vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(0.46),
      bakedRustRim.mul(0.35).mul(isTuffTier.mul(0.65).add(0.35))
    );
    const scorchedAlbedo = mix(
      rustBakedAlbedo,
      vec3(0.016, 0.014, 0.014),
      innerChar.mul(0.64)
    );

    // Keep radiant wall heat focused inside cleaved fracture walls & immediate lip so exterior faces don't wash out
    const wallHeat = clamp(
      max(max(dikeProx, sillProx), roofProx.mul(0.70)).mul(float(0.18).add(aFractureMask.mul(1.05))),
      0.0,
      1.0
    );

    stratifiedRockCore.colorNode = scorchedAlbedo.mul(fractureDarken);
    stratifiedRockCore.emissiveNode = evalBlackbodyEmissiveTsl(
      clamp(wallHeat.mul(0.76), 0.0, 0.80),
      float(0.68)
    ).mul(float(0.48).add(fineGrain.mul(0.52)));

    // Crisp dual-scale 3D rock micro-normal: chiseled cleavage + vesicular basalt pits / sandy tuff grain
    const rockNormalPos = mix(
      aRestPos,
      vec3(aRestPos.x.mul(0.85), yBed.mul(2.1), aRestPos.z.mul(0.85)),
      isTuffTier
    );
    stratifiedRockCore.normalNode = perturbRockNormalDual(
      normalLocal,
      rockNormalPos,
      mix(float(16.0), float(22.0), isTuffTier),
      mix(float(0.48), float(0.32), isTuffTier),
      mix(float(52.0), float(46.0), isTuffTier),
      mix(float(0.44), float(0.30), isTuffTier)
    );

    // Slight glassy vitrification dip on contact-scorched seam margins, rougher inside basalt gas pits
    stratifiedRockCore.roughnessNode = clamp(
      mix(float(0.86), float(0.78), isTuffTier)
        .add(pitMask.mul(0.10).mul(float(1.0).sub(isTuffTier)))
        .sub(coarseRock.mul(0.07))
        .sub(innerChar.mul(0.16)),
      0.56,
      0.96
    );
  }

  // -------------------------------------------------------------------------
  // 5b. Suspended Dark Vesicular Basalt Xenoliths Inside Lower Magma Chamber
  //     Dark angular clasts with glowing incandescent melt meniscus around their flanks!
  // -------------------------------------------------------------------------
  const xenolithBasalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.05,
    roughness: 0.72,
    envMapIntensity: 0.85,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(22.0));
    const ves = smoothstep(float(0.64), float(0.85), mineralToothFbm(positionLocal.add(vec3(2.1, 5.4, 3.3)), float(48.0)));
    // Glowing rim along grazing silhouette + immersed edges where magma wraps around the basalt clast
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const nDotV = clamp(abs(dot(normalize(normalWorld), viewDir)), 0.0, 1.0);
    const grazeHeat = pow(max(float(0.0001), float(1.0).sub(nDotV)), float(1.8));
    const rimHeat = clamp(
      grazeHeat.mul(0.68).add(float(0.18)).add(uChamberState.x.mul(0.22)),
      0.0,
      0.86
    );
    xenolithBasalt.colorNode = mix(
      vec3(0.018, 0.016, 0.017),
      vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.34),
      grain
    ).mul(float(1.0).sub(ves.mul(0.45)));
    xenolithBasalt.emissiveNode = evalBlackbodyEmissiveTsl(rimHeat, float(0.95));
    xenolithBasalt.normalNode = perturbRockNormalDual(
      normalLocal,
      positionLocal,
      float(18.0),
      float(0.46),
      float(48.0),
      float(0.36)
    );
  }

  // -------------------------------------------------------------------------
  // 6. Churning Incandescent Lower Magma Chamber Melt (Visible Through Quartz Windows)
  //    Rich convective dark-slag cellular lanes framing blazing realgar-gold plumes!
  // -------------------------------------------------------------------------
  const chamberMagma = new THREE.MeshStandardNodeMaterial({
    metalness: 0.06,
    roughness: 0.26,
    envMapIntensity: 1.10,
  });
  {
    const flowSpeed = float(0.20).add(uChamberState.x.mul(0.46));
    const convPos = positionLocal.add(vec3(float(0.0), uTime.mul(flowSpeed).negate(), float(0.0)));
    const plumeCoarse = mineralToothFbm(convPos.mul(vec3(1.15, 0.9, 1.15)), float(3.6));
    const plumeWarp = convPos.mul(vec3(1.5, 1.15, 1.5)).add(vec3(plumeCoarse.mul(0.95), 2.3, 1.7));
    const plumeFine = mineralToothFbm(plumeWarp, float(7.8));

    // Convective filament ridges (high-contrast bright hot upwellings vs cooler dark viscous crust lanes)
    const filament = float(1.0).sub(abs(plumeFine.mul(2.0).sub(1.0)));
    const hotFilament = smoothstep(float(0.42), float(0.88), filament);
    const roofChill = smoothstep(float(-0.30), float(-0.22), positionWorld.y).mul(0.22);

    const pressureLift = float(0.42).add(uChamberState.x.mul(0.46));
    const cellHeat = clamp(
      pressureLift
        .mul(float(0.46).add(plumeCoarse.mul(0.44)).add(hotFilament.mul(0.38)))
        .sub(roofChill),
      0.16,
      0.95
    );

    chamberMagma.colorNode = evalBlackbodyAlbedoTsl(cellHeat);
    chamberMagma.emissiveNode = evalBlackbodyEmissiveTsl(cellHeat, float(1.65));
    chamberMagma.roughnessNode = evalMagmaRoughnessTsl(cellHeat);
    chamberMagma.normalNode = perturbNormal3DFbm(normalLocal, plumeWarp, float(8.5), float(0.32));
  }

  // -------------------------------------------------------------------------
  // 7. Dynamic Magma Intrusion Material Factory (Vertical Dikes & Horizontal Sills)
  //    Models hot centerlines, chilled obsidian margins, and live re-cracking seams!
  // -------------------------------------------------------------------------
  function createIntrusionMaterial(isSill: boolean): THREE.MeshStandardNodeMaterial {
    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.08,
      roughness: 0.24,
      envMapIntensity: 1.50,
    });

    const coreTempNode = isSill ? uSillState.z : uDikeThermal.x;
    const skinTempNode = isSill ? uSillState.z.mul(0.84) : uDikeThermal.y;
    const reCrackNode = isSill ? uSillState.w : uDikeThermal.z;

    // Advect molten magma upward along dikes or radially outward along sills
    const flowOffset = isSill
      ? vec3(abs(positionWorld.x).sub(uTime.mul(0.28)), positionWorld.y, positionWorld.z)
      : vec3(positionWorld.x, positionWorld.y.sub(uTime.mul(0.36)), positionWorld.z);

    const fbmMacro = mineralToothFbm(flowOffset, float(5.5));
    const fbmMicro = mineralToothFbm(flowOffset.add(vec3(fbmMacro.mul(0.65), 3.2, 5.1)), float(14.5));

    // Crisp incandescent fracture lines splitting darker chilled obsidian crust
    const ridgeField = float(1.0).sub(abs(fbmMicro.mul(2.0).sub(1.0)));
    const crackLine = smoothstep(float(0.66).sub(reCrackNode.mul(0.24)), float(0.92), ridgeField);

    // Chilled margins along exposed front/back face edges preserve rich cinnabar/amber marbling without white clip
    const marginCool = smoothstep(float(0.38), float(0.54), abs(positionWorld.z)).mul(0.32);
    const crustThermal = clamp(
      skinTempNode.mul(0.86).sub(marginCool).add(fbmMacro.sub(0.5).mul(0.32)),
      0.0,
      0.86
    );
    const fissureTemp = clamp(max(coreTempNode.mul(0.94), skinTempNode.add(0.22)), 0.0, 0.94);
    const crackWeight = clamp(
      crackLine.mul(float(0.45).add(reCrackNode.mul(0.85)).add(uChamberState.w.mul(0.42))),
      0.0,
      1.0
    );

    const finalTemp = mix(crustThermal, fissureTemp, crackWeight);

    mat.colorNode = evalBlackbodyAlbedoTsl(finalTemp);
    mat.emissiveNode = evalBlackbodyEmissiveTsl(finalTemp, float(1.56));
    mat.roughnessNode = evalMagmaRoughnessTsl(finalTemp);
    mat.normalNode = perturbNormal3DFbm(normalLocal, flowOffset, float(14.0), float(0.32));

    return mat;
  }

  const dikeMagma = createIntrusionMaterial(false);
  const sillMagma = createIntrusionMaterial(true);

  // -------------------------------------------------------------------------
  // 8. Instanced Viscoplastic Pahoehoe Magma Lobes (Glossy Ropy Obsidian + Glowing Cracks)
  //    Reads per-instance aLobeThermal = vec4(coreTemp, crustTemp, crackIntensity, seed)
  //    Zero multiplicative positionNode scaling (strict Three.js r187 InstancedMesh contract!)
  // -------------------------------------------------------------------------
  const pahoehoeLobe = new THREE.MeshStandardNodeMaterial({
    metalness: 0.07,
    roughness: 0.14,
    envMapIntensity: 1.72,
  });
  {
    const aLobeThermal = attribute('aLobeThermal', 'vec4');
    const coreTemp = clamp(aLobeThermal.x, 0.0, 1.0);
    const crustTemp = clamp(aLobeThermal.y, 0.0, 1.0);
    const crackIntensity = clamp(aLobeThermal.z, 0.0, 1.0);
    const seed = aLobeThermal.w;

    // Pure un-instanced geometry coordinates + per-lobe seed
    const pSeed = positionGeometry.add(vec3(seed.mul(1.7), seed.mul(2.3), seed.mul(0.9)));
    const flowCrawl = uTime.mul(float(0.06).add(crustTemp.mul(0.22)));

    // Organic arcuate pahoehoe ropy flow wrinkles curving across the viscous pillow tongue
    const warp = mineralToothFbm(pSeed.mul(vec3(1.6, 2.2, 1.6)), float(2.4));
    const flowArc = positionGeometry.z
      .mul(6.8)
      .add(positionGeometry.x.mul(positionGeometry.x).mul(3.2))
      .add(warp.mul(2.6))
      .sub(flowCrawl.mul(3.0));
    const ropyRidge = sin(flowArc).mul(0.5).add(0.5);
    const ropyTrough = float(1.0).sub(ropyRidge);

    // Fine fracture network concentrated inside ropy troughs when cooling crust stretches open
    const pCrack = pSeed.mul(vec3(2.2, 2.8, 2.2)).add(vec3(warp.mul(0.55), flowCrawl.negate(), 1.9));
    const foldFine = mineralToothFbm(pCrack, float(4.8));
    const ridgeVal = float(1.0).sub(abs(foldFine.mul(2.0).sub(1.0)));

    // Razor-crisp tension fissures (zero noisy static speckle when quenched!)
    const crackSignal = ridgeVal.mul(0.76).add(ropyTrough.mul(0.24));
    const crackThresh = clamp(float(0.84).sub(crackIntensity.mul(0.30)).sub(crustTemp.mul(0.16)), 0.42, 0.89);
    const crackMask = smoothstep(crackThresh, float(0.94), crackSignal).mul(
      smoothstep(float(0.03), float(0.18), max(crackIntensity, crustTemp))
    );

    // Exposed convex dome top quenches fastest into smooth jet-black volcanic glass; underside stays hotter
    const topExposure = smoothstep(float(-0.20), float(0.68), normalLocal.y);
    const localSkinTemp = clamp(
      crustTemp.mul(float(1.04).sub(topExposure.mul(0.42))).add(ropyTrough.mul(crustTemp).mul(0.18)),
      0.0,
      0.88
    );

    // Cracks expose incandescent cinnabar -> realgar molten core underneath the glossy black skin
    const exposedCrackHeat = clamp(
      max(coreTemp, localSkinTemp.add(0.28)).mul(float(0.46).add(crackIntensity.mul(0.68))),
      0.0,
      0.95
    );
    const finalTemp = mix(localSkinTemp, exposedCrackHeat, crackMask);

    // Mirror-glossy volcanic obsidian glass (roughness ~ 0.13) on unbroken cooled skin
    const baseRough = evalMagmaRoughnessTsl(finalTemp);
    const obsidianGlassBoost = smoothstep(float(0.24), float(0.05), finalTemp).mul(float(1.0).sub(crackMask));
    const finalRough = clamp(mix(baseRough, float(0.13), obsidianGlassBoost), 0.125, 0.54);

    pahoehoeLobe.colorNode = evalBlackbodyAlbedoTsl(finalTemp);
    pahoehoeLobe.emissiveNode = evalBlackbodyEmissiveTsl(finalTemp, float(1.78));
    pahoehoeLobe.roughnessNode = finalRough;

    // Smooth ropy viscous fold normal (silky glass ripples, ZERO crumpled walnut micro-noise!)
    const ropyNormalCoord = vec3(
      positionGeometry.x.mul(1.8),
      positionGeometry.y.mul(2.4),
      flowArc.mul(0.28)
    ).add(vec3(seed, seed.mul(0.7), 0.0));
    pahoehoeLobe.normalNode = perturbNormal3DFbm(
      normalLocal,
      ropyNormalCoord,
      float(3.4),
      mix(float(0.26), float(0.14), obsidianGlassBoost)
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

  // Suppress unused handle warning in strict TS
  void pCalcite;

  return {
    forgedPressIron,
    machinedThreadSteel,
    satinBrass,
    gaugeDial,
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
      gaugeDial.dispose();
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
