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
  valueNoise3DTsl,
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
  rockChip: THREE.MeshStandardNodeMaterial;
  emberSpark: THREE.MeshBasicNodeMaterial;
  // Shared live uniforms (mutated in-place inside onTick, zero per-frame allocations)
  uTime: { value: number };
  uChamberState: { value: THREE.Vector4 }; // x=pressure, y=compression, z=maxHeat, w=surge
  uDikeHeights: { value: THREE.Vector4 }; // x=h0, y=h1, z=h2, w=dikeTemp
  uDikeThermal: { value: THREE.Vector4 }; // x=coreTemp, y=skinTemp, z=reCrack, w=quenched
  uSillState: { value: THREE.Vector4 }; // x=hw0, y=hw1, z=sillTemp, w=reCrack
  uFractureState: { value: THREE.Vector4 }; // x=hairlineOpen, y=hairlineHeat, z=dikeApNorm, w=sillThickNorm
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
 * Lean, strictly uniform-control-flow 3D anisotropic brittle hairline fracture field.
 * Uses elongated stress-aligned conjugate fracture planes (zero circular O-loops or wide
 * airbrushed flame ribbons!) with crisp high-frequency brittle mineral cleavage kinks so
 * radiating fissures read as authentic razor-sharp rock cracks even at 1:1 Retina macro.
 */
const evalBranchingCrackFieldTsl = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  // Crisp angular brittle zig-zag deflection (high-freq, small amplitude -> jagged stone cleavage!)
  const kX = valueNoise3DTsl(p.mul(15.5).add(vec3(1.2, 4.7, 2.3))).sub(0.5);
  const kY = valueNoise3DTsl(p.mul(15.5).add(vec3(6.8, 1.9, 5.4))).sub(0.5);
  const kZ = valueNoise3DTsl(p.mul(15.5).add(vec3(3.1, 8.2, 9.1))).sub(0.5);
  const kink1 = vec3(kX, kY, kZ).mul(0.056);
  const kink2 = valueNoise3DTsl(p.mul(36.0).add(kink1.mul(10.0))).sub(0.5).mul(0.018);
  const pw = p.add(kink1).add(vec3(kink2));

  // Anisotropic conjugate fracture Family A (+34° upward-branching shear/tension fractures)
  // Clean spacious brittle rock spacing (13.8 cross-freq) so cracks read as bold fractured rock slabs, not busy webs!
  const uAlongA = pw.x.mul(0.829).add(pw.y.mul(0.559)).mul(3.4);
  const uAcrossA = pw.x.mul(-0.559).add(pw.y.mul(0.829)).mul(13.8);
  const cA = valueNoise3DTsl(vec3(uAlongA, uAcrossA, pw.z.mul(4.4)));
  const rA = float(1.0).sub(abs(cA.mul(2.0).sub(1.0)));

  // Anisotropic conjugate fracture Family B (-34° downward-branching conjugate fractures)
  const uAlongB = pw.x.mul(0.829).sub(pw.y.mul(0.559)).mul(3.5).add(7.3);
  const uAcrossB = pw.x.mul(0.559).add(pw.y.mul(0.829)).mul(13.2).add(3.1);
  const cB = valueNoise3DTsl(vec3(uAlongB, uAcrossB, pw.z.mul(4.2).add(5.2)));
  const rB = float(1.0).sub(abs(cB.mul(2.0).sub(1.0)));

  // Sub-horizontal bedding-splinter fractures peeling off sills & strata contacts
  const cC = valueNoise3DTsl(vec3(pw.x.mul(3.8).add(11.4), pw.y.mul(17.5).add(2.9), pw.z.mul(4.2).add(8.1)));
  const rC = float(1.0).sub(abs(cC.mul(2.0).sub(1.0)));

  const primary = max(max(rA, rB), rC.mul(0.94));

  // Delicate secondary micro-splinter hairlines budding tightly off primary fracture roots
  const cTwig = valueNoise3DTsl(vec3(pw.x.mul(24.0).add(4.3), pw.y.mul(13.0).add(11.7), pw.z.mul(15.0).add(7.1)));
  const rTwig = float(1.0).sub(abs(cTwig.mul(2.0).sub(1.0)));
  const splinter = rTwig.mul(smoothstep(float(0.84), float(0.97), primary));

  return clamp(max(primary, splinter.mul(0.97)), 0.0, 1.0);
});

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
  const uFractureState = uniform(new THREE.Vector4(0.45, 0.55, 0.40, 0.10));
  const uHoverState = uniform(new THREE.Vector4(0, 0, 0, 0));

  const pOchre = createPigmentUniforms('yellowOchre');
  const pSienna = createPigmentUniforms('burntSienna');
  const pUmber = createPigmentUniforms('rawUmber');
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pCalcite = createPigmentUniforms('boneCalcite');

  // -------------------------------------------------------------------------
  // 1. Oil-Blackened Forged & Milled Gunmetal Steel Geology Press Frame
  //    Calibrated metallic value range + directional brushed-oil sheen + edge-burnished
  //    45° chamfer wear so steel beams, platens & chamber frame gleam crisply under IBL.
  // -------------------------------------------------------------------------
  const forgedPressIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.80,
    roughness: 0.36,
    envMapIntensity: 1.44,
  });
  {
    const macroScale = mineralToothFbm(positionLocal, float(7.5));
    // Fine directional brushed-oil machined steel marks along X + forged peening grain
    const milledBrush = mineralToothFbm(positionLocal.mul(vec3(2.8, 30.0, 26.0)), float(7.2));
    const crossTool = mineralToothFbm(positionLocal.mul(vec3(26.0, 4.2, 22.0)).add(vec3(4.1, 1.3, 6.2)), float(6.0));
    const microPeen = mineralToothFbm(positionLocal.add(vec3(3.1, 7.4, 2.2)), float(44.0));

    // Edge-burnished highlight peaking cleanly on 45° beveled block chamfers
    const nAbs = abs(normalize(normalLocal));
    const chamferEdge = clamp(
      nAbs.x.mul(nAbs.y).add(nAbs.y.mul(nAbs.z)).add(nAbs.x.mul(nAbs.z)).mul(2.15),
      0.0,
      1.0
    );
    const burnishMask = smoothstep(float(0.32), float(0.84), chamferEdge).mul(
      float(0.68).add(microPeen.mul(0.32))
    );

    // Authentic oil-blackened gunmetal cast-iron / rubbed tool-steel reflectance
    const ironDeep = vec3(0.044, 0.048, 0.056);
    const ironMid = vec3(0.072, 0.078, 0.089);
    const ironMachined = vec3(0.118, 0.126, 0.140);
    const ironBurnished = vec3(0.195, 0.205, 0.220);

    const brushMix = milledBrush.mul(0.58).add(crossTool.mul(0.18)).add(microPeen.mul(0.24));
    const baseIron = mix(
      mix(ironDeep, ironMid, smoothstep(float(0.24), float(0.76), macroScale)),
      ironMachined,
      smoothstep(float(0.40), float(0.84), brushMix).mul(0.42)
    );
    const edgeRubbed = mix(baseIron, ironBurnished, burnishMask.mul(0.48));

    // Subtle upward-facing machined-table oil polish (gives top faces of beams & base bed metallic sheen)
    const topPolish = smoothstep(float(0.65), float(0.95), nAbs.y).mul(0.012);

    // Restrained warm hearth bounce strictly on chamber casing walls (not washing out the whole base bed)
    const dyChamber = positionWorld.y.add(0.42);
    const chamberProximity = exp(dyChamber.mul(dyChamber).mul(-12.0))
      .mul(exp(positionWorld.x.mul(positionWorld.x).mul(-1.4)));
    const magmaBounce = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear)
      .mul(chamberProximity)
      .mul(float(0.024).add(uChamberState.x.mul(0.038)));

    forgedPressIron.colorNode = edgeRubbed.add(vec3(topPolish)).add(magmaBounce);
    // Crisp brushed-oil / milled iron micro-normal texture so studio softbox speculars roll cleanly
    forgedPressIron.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal.mul(vec3(0.85, 2.4, 1.9)),
      float(36.0),
      mix(float(0.030), float(0.014), burnishMask)
    );
    forgedPressIron.roughnessNode = clamp(
      float(0.36)
        .add(macroScale.mul(0.10))
        .sub(milledBrush.mul(0.065))
        .sub(topPolish.mul(2.0))
        .sub(burnishMask.mul(0.12)),
      0.20,
      0.50
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
  // 4. Beveled Laboratory Quartz Cutaway Window Glass
  //    Sharp dual-strip studio reflections, beveled rim catch, & subtle upper thermal soot edging
  // -------------------------------------------------------------------------
  const quartzGlass = new THREE.MeshStandardNodeMaterial({
    transparent: true,
    depthWrite: false,
    metalness: 0.05,
    roughness: 0.055,
    envMapIntensity: 1.65,
  });
  {
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDir)), 0.0, 1.0);
    const fresnel = pow(max(float(0.0001), float(1.0).sub(nDotV)), float(3.2));

    // Dual-strip parallax studio softbox reflections (broad overhead + crisp narrow key catchlight)
    const parallaxCoord = positionLocal.x.mul(1.85).add(positionLocal.y.mul(2.55)).add(viewDir.x.mul(0.72));
    const glareBroad = exp(pow(max(float(0.0001), abs(sin(parallaxCoord.mul(1.30)).sub(0.52))), float(2.0)).mul(-20.0));
    const glareSharp = exp(
      pow(max(float(0.0001), abs(sin(parallaxCoord.mul(1.30).add(0.42)).sub(0.78))), float(2.0)).mul(-72.0)
    );
    const glareTotal = clamp(glareBroad.mul(0.65).add(glareSharp.mul(0.85)), 0.0, 1.0);

    // Crisp beveled border specular catch near pane perimeter (x in [-0.68, 0.68], y in [-0.145, 0.145])
    const edgeX = smoothstep(float(0.59), float(0.685), abs(positionLocal.x));
    const edgeY = smoothstep(float(0.106), float(0.144), abs(positionLocal.y));
    const edgeCatch = clamp(max(edgeX, edgeY), 0.0, 1.0);

    // Subtle wispy volcanic thermal soot / condensed vapor film along upper bevel & pane corners
    const sootWisp = mineralToothFbm(positionLocal.mul(vec3(5.5, 11.0, 4.0)).add(vec3(2.1, 0.4, 1.7)), float(2.2));
    const upperSootBand = smoothstep(float(0.052), float(0.142), positionLocal.y);
    const cornerSoot = smoothstep(float(0.46), float(0.68), abs(positionLocal.x)).mul(
      smoothstep(float(0.04), float(0.14), abs(positionLocal.y))
    );
    const sootMask = clamp(
      max(upperSootBand.mul(0.82), cornerSoot.mul(0.75)).mul(float(0.55).add(sootWisp.mul(0.65))),
      0.0,
      0.85
    );

    const coolGlass = vec3(0.62, 0.70, 0.78);
    const warmRim = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear).mul(0.62);
    const sootColor = vec3(0.018, 0.014, 0.013);

    const glassTint = mix(coolGlass, warmRim, edgeCatch.mul(0.40).add(uChamberState.x.mul(0.15)))
      .add(vec3(0.32, 0.34, 0.36).mul(glareTotal.mul(0.55)));
    quartzGlass.colorNode = mix(glassTint, sootColor, sootMask.mul(0.76));
    quartzGlass.opacityNode = clamp(
      float(0.08)
        .add(fresnel.mul(0.30))
        .add(glareTotal.mul(0.14))
        .add(edgeCatch.mul(0.24))
        .add(sootMask.mul(0.22)),
      0.08,
      0.56
    );
  }

  // -------------------------------------------------------------------------
  // 5. 3D Object-Space Stratified Vesicular Basalt & Pyroclastic Ochre Tuff Rock
  //    - Branching secondary tectonic fracture networks & radiating hairline pressure cracks
  //      that dynamically widen & illuminate cherry -> cinnabar -> realgar from inside out!
  //    - Analytical normal-mapped beveled fracture lips + blackened metamorphism scorch halos
  //    - Angular basalt columnar jointing facets, plagioclase flecks & deep multi-scale vesicles
  //    - Sharp dark volcanic lithic clasts & micro-bedding relief in yellow-ochre tuff
  //    - Vertical fault-slickenside slip striations along parted fracture walls (aSlipMask)
  // -------------------------------------------------------------------------
  const stratifiedRockCore = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.86,
    envMapIntensity: 0.92,
  });
  {
    const aRestPos = attribute('aRestPos', 'vec3');
    const aStratum = attribute('aStratum', 'float');
    const aFractureMask = attribute('aFractureMask', 'float');
    const aSlipMask = attribute('aSlipMask', 'float');
    const aSlabIdx = attribute('aSlabIdx', 'float');

    // Pure 3D object-space rest-core petrography so bedding & grain align across closed joints
    const coarseRock = mineralToothFbm(aRestPos.add(vec3(aSlabIdx.mul(0.11), 1.3, 4.1)), float(7.5));
    const medGrain = mineralToothFbm(aRestPos.add(vec3(9.1, 2.7, 5.4)), float(26.0));
    const fineGrain = mineralToothFbm(aRestPos.add(vec3(2.3, 6.9, 8.1)), float(64.0));

    // Angular basalt columnar-jointing prismatic facet shading (subtle stepped mineral cleavage planes!)
    const colRaw = valueNoise3DTsl(vec3(aRestPos.x.mul(14.0), aRestPos.y.mul(4.2), aRestPos.z.mul(14.0)));
    const colFacet = smoothstep(float(0.28), float(0.72), colRaw);
    const colEdgeLine = smoothstep(float(0.84), float(0.97), float(1.0).sub(abs(colRaw.mul(2.0).sub(1.0))));

    // Deep multi-scale vesicular basalt gas-bubble pits (crisp cavities in albedo + steep rim normals)
    const vesMed = mineralToothFbm(aRestPos.add(vec3(3.7, 8.2, 1.9)), float(34.0));
    const vesFine = mineralToothFbm(aRestPos.add(vec3(11.4, 2.1, 6.8)), float(82.0));
    const pitMed = smoothstep(float(0.62), float(0.80), vesMed);
    const pitFine = smoothstep(float(0.64), float(0.82), vesFine);
    const pitMask = clamp(pitMed.mul(0.78).add(pitFine.mul(0.48)), 0.0, 1.0);

    // Distinct angular plagioclase feldspar laths & subtle olivine/pyroxene crystalline flecks inside basalt
    const mineralFleck = smoothstep(float(0.76), float(0.89), fineGrain).mul(0.24);
    const darkClastBasalt = smoothstep(float(0.72), float(0.84), medGrain).mul(0.32);

    // Horizon 0 & 2: Dark Charcoal-Slate Vesicular Columnar Basalt
    const basaltDeep = vec3(0.026, 0.027, 0.031);
    const basaltMid = vec3(0.068, 0.065, 0.065);
    const basaltFacetHi = vec3(0.095, 0.091, 0.090);
    const basaltUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.46);
    const basaltHematite = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(0.38);
    const basaltPlag = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear).mul(0.36);

    const basaltFaceted = mix(
      mix(basaltDeep, basaltMid, smoothstep(float(0.22), float(0.78), coarseRock)),
      basaltFacetHi,
      colFacet.mul(0.32)
    );
    const basaltBody = mix(
      basaltFaceted,
      basaltUmber,
      smoothstep(float(0.44), float(0.84), medGrain).mul(0.24)
    )
      .mul(float(1.0).sub(darkClastBasalt.mul(0.45)))
      .mul(float(1.0).sub(colEdgeLine.mul(0.28)))
      .add(basaltPlag.mul(mineralFleck));

    // Subtle rust-ochre iron weathering on exposed upper crown facets
    const crownWeather = smoothstep(float(0.45), float(0.88), normalLocal.y).mul(
      smoothstep(float(1.4), float(1.8), aStratum)
    );
    const basaltFinal = mix(basaltBody, basaltHematite, crownWeather.mul(0.24)).mul(
      float(1.0).sub(pitMask.mul(0.64))
    );

    // Horizon 1: Authentic Pyroclastic Yellow-Ochre Volcanic Tuff & Ignimbrite Strata
    // Non-periodic domain-warped 3D FBM beds + fine ash laminae + crisp angular lithic clasts
    const bedWarpCoarse = mineralToothFbm(aRestPos.mul(vec3(1.6, 2.8, 1.6)), float(2.2));
    const bedWarpFine = mineralToothFbm(aRestPos.mul(vec3(4.5, 6.2, 4.5)), float(3.4));
    const yBed = aRestPos.y
      .add(bedWarpCoarse.sub(0.5).mul(0.048))
      .add(bedWarpFine.sub(0.5).mul(0.014))
      .add(aRestPos.x.mul(0.028));

    const bedBroad = mineralToothFbm(vec3(aRestPos.x.mul(0.45), yBed.mul(14.5), aRestPos.z.mul(0.45)), float(2.3));
    const bedFine = mineralToothFbm(vec3(aRestPos.x.mul(0.65), yBed.mul(46.0).add(3.7), aRestPos.z.mul(0.65)), float(2.1));
    const bedMicro = mineralToothFbm(vec3(aRestPos.x.mul(0.80), yBed.mul(96.0).add(7.1), aRestPos.z.mul(0.80)), float(1.9));

    const ironParting = smoothstep(float(0.69), float(0.84), bedFine).mul(
      smoothstep(float(0.32), float(0.76), bedBroad)
    );
    const fineSiltStreak = smoothstep(float(0.64), float(0.82), bedMicro);
    const kaolinHorizon = smoothstep(float(0.60), float(0.82), float(1.0).sub(bedBroad));

    // Crisp angular volcanic basalt lapilli & pale calcite pumice clasts embedded inside the tuff
    const lapilliCoarse = valueNoise3DTsl(aRestPos.mul(30.0).add(vec3(5.3, 1.7, 8.9)));
    const lapilliFine = mineralToothFbm(aRestPos.add(vec3(5.3, 1.7, 8.9)), float(52.0));
    const lapilliMask = clamp(
      smoothstep(float(0.75), float(0.84), lapilliCoarse).mul(0.85).add(
        smoothstep(float(0.72), float(0.84), lapilliFine).mul(0.55)
      ),
      0.0,
      1.0
    );
    const pumiceClast = smoothstep(float(0.78), float(0.89), valueNoise3DTsl(aRestPos.mul(38.0).add(vec3(9.2, 3.1, 1.4))));
    const tuffPoreMask = smoothstep(float(0.68), float(0.85), vesFine);

    // Physical 2-flux Kubelka-Munk mineral glazes calibrated to earthy geological tuff values
    const umberBacking = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.50);
    const ochreThickness = float(0.46).add(bedBroad.mul(0.55)).add(fineGrain.mul(0.22));
    const ochreTuff = kubelkaMunkGlaze(pOchre.K, pOchre.S, ochreThickness, umberBacking);
    const kaolinAsh = kubelkaMunkGlaze(
      pKaolin.K,
      pKaolin.S,
      float(0.38).add(kaolinHorizon.mul(0.48)),
      ochreTuff.mul(0.82)
    );
    const siennaSeam = kubelkaMunkGlaze(
      pSienna.K,
      pSienna.S,
      float(0.44).add(ironParting.mul(0.68)),
      umberBacking.mul(0.70)
    );
    const umberSeam = kubelkaMunkGlaze(
      pUmber.K,
      pUmber.S,
      float(0.58).add(fineSiltStreak.mul(0.52)),
      basaltDeep
    );

    const tuffBase = mix(ochreTuff.mul(0.78), kaolinAsh.mul(0.78), kaolinHorizon.mul(0.60));
    const tuffBanded = mix(tuffBase, siennaSeam.mul(0.70), ironParting.mul(0.58));
    const tuffMicroLaminated = mix(tuffBanded, umberSeam.mul(0.66), fineSiltStreak.mul(0.36));
    const tuffWithPumice = mix(
      tuffMicroLaminated,
      vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear).mul(0.64),
      pumiceClast.mul(0.42)
    );
    const tuffLaminated = mix(tuffWithPumice, basaltDeep.mul(1.15), lapilliMask.mul(0.74))
      .mul(float(0.86).add(coarseRock.mul(0.26)))
      .mul(float(1.0).sub(tuffPoreMask.mul(0.32)));

    // Smooth stratum selection: middle tier (aStratum == 1) is Banded Volcanic Tuff; tiers 0 & 2 are Basalt
    const isTuffTier = float(1.0).sub(clamp(abs(aStratum.sub(1.0)).mul(2.0), 0.0, 1.0));
    const rockAlbedo = mix(basaltFinal, tuffLaminated, isTuffTier);

    // Vertical fault-slickenside mineral striations strictly on parted vertical fault walls (aSlipMask)
    const slickWave = sin(aRestPos.z.mul(185.0).add(medGrain.mul(4.5))).mul(0.5).add(0.5);
    const slickGroove = valueNoise3DTsl(vec3(aRestPos.x.mul(8.0), aRestPos.y.mul(4.5), aRestPos.z.mul(64.0)));
    const slickBand = slickWave.mul(0.55).add(slickGroove.mul(0.45));
    const slickSheen = smoothstep(float(0.55), float(0.90), slickBand).mul(aSlipMask);
    const slickFlour = smoothstep(float(0.45), float(0.12), slickBand).mul(aSlipMask);
    const rockWithSlick = mix(
      rockAlbedo.add(vec3(0.022, 0.020, 0.019).mul(slickSheen)),
      vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear).mul(0.28),
      slickFlour.mul(0.22)
    );

    // -----------------------------------------------------------------------
    // Real-Time Tectonic Fracture Network, Radiating Hairline Cracks & Scorch
    // -----------------------------------------------------------------------
    const yAboveRoof = max(float(0.0), aRestPos.y.sub(float(OVERBURDEN_BOTTOM_Y)));
    const d0Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[0])));
    const d1Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[1])));
    const d2Dist = abs(aRestPos.x.sub(float(DIKE_SEAM_X[2])));

    // Dike proximity + precursor fracture process zone ahead of rising dike tip!
    const dikeCharGain = clamp(max(uDikeHeights.w, uDikeThermal.w.mul(0.88)), 0.0, 1.0);
    const reach0 = smoothstep(uDikeHeights.x.add(0.06), uDikeHeights.x.sub(0.14), yAboveRoof);
    const reach1 = smoothstep(uDikeHeights.y.add(0.06), uDikeHeights.y.sub(0.14), yAboveRoof);
    const reach2 = smoothstep(uDikeHeights.z.add(0.06), uDikeHeights.z.sub(0.14), yAboveRoof);

    // Precursor process zone extends ~0.22m ahead of active dike tip for branching hairline fissures
    const preReach0 = smoothstep(uDikeHeights.x.add(0.26), uDikeHeights.x.sub(0.04), yAboveRoof).mul(
      smoothstep(float(0.04), float(0.20), uDikeHeights.x)
    );
    const preReach1 = smoothstep(uDikeHeights.y.add(0.22), uDikeHeights.y.sub(0.04), yAboveRoof).mul(
      smoothstep(float(0.04), float(0.20), uDikeHeights.y)
    );
    const preReach2 = smoothstep(uDikeHeights.z.add(0.22), uDikeHeights.z.sub(0.04), yAboveRoof).mul(
      smoothstep(float(0.04), float(0.20), uDikeHeights.z)
    );

    const nearD0 = smoothstep(float(0.115), float(0.0), d0Dist).mul(reach0);
    const nearD1 = smoothstep(float(0.095), float(0.0), d1Dist).mul(reach1);
    const nearD2 = smoothstep(float(0.095), float(0.0), d2Dist).mul(reach2);
    const dikeGeoProx = clamp(max(nearD0, max(nearD1, nearD2)), 0.0, 1.0);
    const dikeProx = dikeGeoProx.mul(uDikeHeights.w);

    const nearS0 = smoothstep(float(0.082), float(0.0), abs(aRestPos.y.sub(float(SILL_BED_Y[0])))).mul(
      smoothstep(uSillState.x.add(0.08), max(float(0.01), uSillState.x.sub(0.10)), abs(aRestPos.x))
    );
    const nearS1 = smoothstep(float(0.082), float(0.0), abs(aRestPos.y.sub(float(SILL_BED_Y[1])))).mul(
      smoothstep(uSillState.y.add(0.08), max(float(0.01), uSillState.y.sub(0.10)), abs(aRestPos.x))
    );
    const sillGeoProx = clamp(max(nearS0, nearS1), 0.0, 1.0);
    const sillProx = sillGeoProx.mul(uSillState.z);

    // Radiating secondary fracture envelope extending up to 0.24m off dikes & 0.19m off sills + dome crown!
    const dikeBranch0 = smoothstep(float(0.25), float(0.012), d0Dist).mul(preReach0);
    const dikeBranch1 = smoothstep(float(0.19), float(0.012), d1Dist).mul(preReach1);
    const dikeBranch2 = smoothstep(float(0.19), float(0.012), d2Dist).mul(preReach2);
    const dikeBranchEnv = clamp(max(dikeBranch0, max(dikeBranch1, dikeBranch2)), 0.0, 1.0);

    const sillBranch0 = smoothstep(float(0.18), float(0.01), abs(aRestPos.y.sub(float(SILL_BED_Y[0])))).mul(
      smoothstep(uSillState.x.add(0.15), max(float(0.02), uSillState.x.sub(0.06)), abs(aRestPos.x))
    ).mul(smoothstep(float(0.08), float(0.24), uSillState.x));
    const sillBranch1 = smoothstep(float(0.18), float(0.01), abs(aRestPos.y.sub(float(SILL_BED_Y[1])))).mul(
      smoothstep(uSillState.y.add(0.15), max(float(0.02), uSillState.y.sub(0.06)), abs(aRestPos.x))
    ).mul(smoothstep(float(0.08), float(0.24), uSillState.y));
    const sillBranchEnv = clamp(max(sillBranch0, sillBranch1), 0.0, 1.0);

    // Upper basalt cap anticlinal dome / crest tension fracture envelope
    const onCrownSurface = smoothstep(float(0.35), float(0.80), normalLocal.y).mul(
      smoothstep(float(1.4), float(1.8), aStratum)
    );
    const crownCenterWeight = smoothstep(float(0.72), float(0.05), abs(aRestPos.x));
    const crownBranchEnv = onCrownSurface
      .mul(crownCenterWeight)
      .mul(clamp(uFractureState.x.mul(0.80).add(uFractureState.w.mul(0.65)), 0.0, 1.0));

    const branchEnv = clamp(max(max(dikeBranchEnv, sillBranchEnv), crownBranchEnv), 0.0, 1.0);

    // Evaluate razor-fine branching hairline fracture field + 3-tap finite-difference beveled lips!
    const crackOpen = uFractureState.x;
    const crackHeatDrive = uFractureState.y;
    const crackVal0 = evalBranchingCrackFieldTsl(aRestPos);
    const cEps = float(0.0028);
    const crackValX = evalBranchingCrackFieldTsl(aRestPos.add(vec3(cEps, 0.0, 0.0)));
    const crackValY = evalBranchingCrackFieldTsl(aRestPos.add(vec3(0.0, cEps, 0.0)));
    const crackValZ = evalBranchingCrackFieldTsl(aRestPos.add(vec3(0.0, 0.0, cEps)));

    // Razor-crisp physical hairline widths (~3mm at pressurized root -> ~0.6mm dark hairline at tip!)
    const grooveThresh = clamp(
      float(0.982).sub(branchEnv.mul(float(0.025).add(crackOpen.mul(0.032)))),
      0.922,
      0.986
    );
    const envGate = smoothstep(float(0.08), float(0.28), branchEnv);
    const crackGroove = smoothstep(grooveThresh, float(0.994), crackVal0).mul(envGate);
    // Razor-fine glowing magma capillary strictly inside the root of the beveled rock groove!
    const coreThresh = clamp(grooveThresh.add(float(0.022).sub(crackOpen.mul(0.005))), 0.946, 0.995);
    const crackCore = smoothstep(coreThresh, float(0.998), crackVal0).mul(envGate);

    // Inside-out thermal gradient: cinnabar-realgar strictly at dike/sill root -> cherry-red mid -> unlit dark hairline tips!
    const rootProx = pow(max(float(0.0001), branchEnv), float(1.65));
    const hairlineLocalTemp = clamp(
      crackCore.mul(rootProx).mul(float(0.32).add(crackHeatDrive.mul(0.76)).add(uChamberState.w.mul(0.24))),
      0.0,
      0.86
    );

    // Tight blackened contact-metamorphism scorch rim hugging each hairline fracture lip (never muddy smear!)
    const hairlineScorch = smoothstep(grooveThresh.sub(0.032), grooveThresh.add(0.006), crackVal0)
      .mul(branchEnv)
      .mul(clamp(max(crackHeatDrive, uDikeThermal.w.mul(0.85)).add(0.22), 0.0, 1.0));

    // Base chamber roof thermal conduction along bottom of Stratum 0
    const roofProx = smoothstep(float(CHAMBER_ROOF_Y + 0.11), float(CHAMBER_ROOF_Y), aRestPos.y).mul(
      float(0.32).add(uChamberState.x.mul(0.68))
    );

    const charBand = clamp(
      max(
        max(dikeGeoProx.mul(dikeCharGain), sillGeoProx.mul(max(uSillState.z, uDikeThermal.w.mul(0.55)))),
        hairlineScorch.mul(0.92)
      ).mul(float(0.72).add(medGrain.mul(0.42))),
      0.0,
      1.0
    );
    const innerChar = smoothstep(float(0.34), float(0.80), charBand);
    const bakedRustRim = smoothstep(float(0.12), float(0.44), charBand).mul(float(1.0).sub(innerChar));

    // Darken fracture walls, bake tight metamorphism char halos, and drop crisp recessed AO into hairline cracks
    const fractureDarken = float(1.0).sub(aFractureMask.mul(0.26));
    const rustBakedAlbedo = mix(
      rockWithSlick,
      vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(0.44),
      bakedRustRim.mul(0.34).mul(isTuffTier.mul(0.65).add(0.35))
    );
    const scorchedAlbedo = mix(
      rustBakedAlbedo,
      vec3(0.013, 0.011, 0.011),
      innerChar.mul(0.68)
    );
    // Crisp dark shadow trough inside hairline cracks + subtle freshly cleaved stone rim highlight!
    const crackLipRing = smoothstep(grooveThresh.sub(0.016), grooveThresh.add(0.004), crackVal0)
      .mul(float(1.0).sub(crackGroove))
      .mul(envGate);
    const crackedAlbedo = scorchedAlbedo
      .mul(float(1.0).sub(crackGroove.mul(0.82)))
      .add(mix(vec3(0.055, 0.052, 0.050), vec3(0.18, 0.13, 0.07), isTuffTier).mul(crackLipRing.mul(0.48)));

    // Blend restrained incandescent magma albedo strictly inside glowing hairline capillary cores
    const finalRockAlbedo = mix(
      crackedAlbedo.mul(fractureDarken),
      evalBlackbodyAlbedoTsl(hairlineLocalTemp),
      crackCore.mul(smoothstep(float(0.14), float(0.48), hairlineLocalTemp))
    );

    // Radiant thermal glow: cleaved fracture walls + restrained cherry->cinnabar->realgar hairline capillaries!
    const wallHeat = clamp(
      max(max(dikeProx, sillProx), roofProx.mul(0.70)).mul(float(0.16).add(aFractureMask.mul(1.05))),
      0.0,
      1.0
    );
    const wallEmissive = evalBlackbodyEmissiveTsl(
      clamp(wallHeat.mul(0.74), 0.0, 0.78),
      float(0.64)
    ).mul(float(0.48).add(fineGrain.mul(0.52)));
    const hairlineEmissive = evalBlackbodyEmissiveTsl(hairlineLocalTemp, float(1.28)).mul(crackCore);

    stratifiedRockCore.colorNode = finalRockAlbedo;
    stratifiedRockCore.emissiveNode = wallEmissive.add(hairlineEmissive);

    // Combined 3D Micro-Normal: Dual-Scale Rock/Vesicles + Tuff Bedding + Fault Slickensides + Beveled Crack Lips!
    const rockNormalPos = mix(
      aRestPos,
      vec3(aRestPos.x.mul(0.85), yBed.mul(2.3), aRestPos.z.mul(0.85)),
      isTuffTier
    );
    const nBase = normalize(normalLocal);
    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const sc1 = mix(float(16.0), float(22.0), isTuffTier);
    const e1 = float(0.16).div(sc1);
    const c1 = mineralToothFbm(rockNormalPos.add(k1.mul(e1)), sc1);
    const c2 = mineralToothFbm(rockNormalPos.add(k2.mul(e1)), sc1);
    const c3 = mineralToothFbm(rockNormalPos.add(k3.mul(e1)), sc1);
    const c4 = mineralToothFbm(rockNormalPos.add(k4.mul(e1)), sc1);
    const g1 = k1.mul(c1).add(k2.mul(c2)).add(k3.mul(c3)).add(k4.mul(c4)).mul(1.5625);

    const sc2 = mix(float(54.0), float(48.0), isTuffTier);
    const e2 = float(0.16).div(sc2);
    const pFine = rockNormalPos.add(vec3(4.1, 7.3, 2.9));
    const f1 = mineralToothFbm(pFine.add(k1.mul(e2)), sc2);
    const f2 = mineralToothFbm(pFine.add(k2.mul(e2)), sc2);
    const f3 = mineralToothFbm(pFine.add(k3.mul(e2)), sc2);
    const f4 = mineralToothFbm(pFine.add(k4.mul(e2)), sc2);
    // Sharper basalt gas-bubble pits & clastic tuff grit
    const s1 = smoothstep(float(0.50), float(0.82), f1);
    const s2 = smoothstep(float(0.50), float(0.82), f2);
    const s3 = smoothstep(float(0.50), float(0.82), f3);
    const s4 = smoothstep(float(0.50), float(0.82), f4);
    const g2 = k1.mul(s1).add(k2.mul(s2)).add(k3.mul(s3)).add(k4.mul(s4)).mul(1.5625);

    // Analytical concave bevel gradient along razor-crisp branching hairline pressure cracks!
    const gCrackRaw = vec3(
      crackValX.sub(crackVal0),
      crackValY.sub(crackVal0),
      crackValZ.sub(crackVal0)
    ).div(cEps);
    const crackBevelMask = smoothstep(grooveThresh.sub(0.022), grooveThresh.add(0.025), crackVal0).mul(envGate);
    const gCrack = gCrackRaw.mul(crackBevelMask).mul(float(0.10).add(crackOpen.mul(0.08)));

    // Vertical fault-slickenside slip groove gradient strictly along Z on parted vertical walls
    const slickSlopeZ = cos(aRestPos.z.mul(185.0).add(medGrain.mul(4.5))).mul(0.32).mul(aSlipMask);
    // Sedimentary tuff horizontal bedding ledge micro-normal along Y
    const bedLedgeY = bedFine.sub(0.5).mul(0.34).mul(isTuffTier);

    const rockGrad = g1
      .mul(mix(float(0.52), float(0.36), isTuffTier))
      .add(g2.mul(mix(float(0.52), float(0.34), isTuffTier)))
      .add(vec3(float(0.0), bedLedgeY, slickSlopeZ))
      .sub(gCrack);
    const tanRockGrad = rockGrad.sub(nBase.mul(dot(rockGrad, nBase)));
    stratifiedRockCore.normalNode = transformNormalToView(normalize(nBase.sub(tanRockGrad)));

    // Vitrified sheen on scorched margins & sheared slickensides; matte rough inside basalt vesicles
    stratifiedRockCore.roughnessNode = clamp(
      mix(float(0.87), float(0.79), isTuffTier)
        .add(pitMask.mul(0.10).mul(float(1.0).sub(isTuffTier)))
        .sub(coarseRock.mul(0.07))
        .sub(innerChar.mul(0.18))
        .sub(slickSheen.mul(0.16))
        .sub(mineralFleck.mul(0.38).mul(float(1.0).sub(isTuffTier))),
      0.48,
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
  //    High-contrast convective turbulence: blazing pale-gold/realgar plumes twisting
  //    through dark viscous basaltic crust lanes behind beveled laboratory quartz!
  // -------------------------------------------------------------------------
  const chamberMagma = new THREE.MeshStandardNodeMaterial({
    metalness: 0.06,
    roughness: 0.25,
    envMapIntensity: 1.18,
  });
  {
    const flowSpeed = float(0.24).add(uChamberState.x.mul(0.52));
    const swirlX = sin(positionLocal.y.mul(5.2).sub(uTime.mul(0.9))).mul(0.14);
    const convPos = positionLocal.add(vec3(swirlX, uTime.mul(flowSpeed).negate(), float(0.0)));
    const plumeCoarse = mineralToothFbm(convPos.mul(vec3(1.25, 0.95, 1.25)), float(3.8));
    const plumeWarp = convPos
      .mul(vec3(1.65, 1.25, 1.65))
      .add(vec3(plumeCoarse.mul(1.15), uTime.mul(0.18).add(2.3), 1.7));
    const plumeFine = mineralToothFbm(plumeWarp, float(8.4));

    // Sharp convective upwelling filaments framing cooler dark crimson-slag eddies
    const filament = float(1.0).sub(abs(plumeFine.mul(2.0).sub(1.0)));
    const hotFilament = smoothstep(float(0.36), float(0.86), filament);
    const brightCore = smoothstep(float(0.68), float(0.94), filament.mul(0.65).add(plumeCoarse.mul(0.45)));
    const roofChill = smoothstep(float(-0.30), float(-0.22), positionWorld.y).mul(0.24);

    const pressureLift = float(0.42).add(uChamberState.x.mul(0.50)).add(uChamberState.w.mul(0.14));
    const cellHeat = clamp(
      pressureLift
        .mul(float(0.40).add(plumeCoarse.mul(0.42)).add(hotFilament.mul(0.42)).add(brightCore.mul(0.26)))
        .sub(roofChill),
      0.14,
      0.98
    );

    chamberMagma.colorNode = evalBlackbodyAlbedoTsl(cellHeat);
    chamberMagma.emissiveNode = evalBlackbodyEmissiveTsl(cellHeat, float(1.82));
    chamberMagma.roughnessNode = evalMagmaRoughnessTsl(cellHeat);
    chamberMagma.normalNode = perturbNormal3DFbm(normalLocal, plumeWarp, float(9.2), float(0.36));
  }

  // -------------------------------------------------------------------------
  // 7. Dynamic Magma Intrusion Material Factory (Vertical Dikes & Horizontal Sills)
  //    True blackbody cooling-crust plate dynamics: dark glossy volcanic obsidian
  //    skin-islands tearing apart across incandescent pale-gold/realgar/cinnabar
  //    convective channels during pressure surges, then knitting into silky obsidian
  //    with hairline quench-cracks when quenched!
  // -------------------------------------------------------------------------
  function createIntrusionMaterial(isSill: boolean): THREE.MeshStandardNodeMaterial {
    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.08,
      roughness: 0.20,
      envMapIntensity: 1.65,
    });

    const coreTempNode = isSill ? uSillState.z : uDikeThermal.x;
    const skinTempNode = isSill ? uSillState.z.mul(0.82) : uDikeThermal.y;
    const reCrackNode = isSill ? uSillState.w : uDikeThermal.z;
    const quenchedNode = uDikeThermal.w;

    // Advect molten magma upward along dikes or laterally outward along sills (stalls smoothly when quenched)
    const flowCrawl = uTime.mul(float(0.08).add(skinTempNode.mul(0.34)));
    const flowOffset = isSill
      ? vec3(abs(positionWorld.x).sub(flowCrawl), positionWorld.y.mul(1.2), positionWorld.z)
      : vec3(positionWorld.x.mul(1.2), positionWorld.y.sub(flowCrawl), positionWorld.z);

    // Coherent crust-raft domains + jagged pull-apart rifts (strictly uniform control flow)
    const fbmMacro = mineralToothFbm(flowOffset, float(4.8));
    const warpPos = flowOffset.add(vec3(fbmMacro.mul(0.55), fbmMacro.mul(0.45).add(3.2), 5.1));
    const fbmMicro = mineralToothFbm(warpPos, float(10.5));

    // Clean primary tearing magma veins framing glossy black obsidian crust islands (zero speckled confetti!)
    const ridgeMain = float(1.0).sub(abs(fbmMicro.mul(2.0).sub(1.0)));

    // Exposed outer meniscus cools rapidly into coherent glossy obsidian rafts torn apart by surging melt!
    const marginCool = smoothstep(float(0.30), float(0.54), abs(positionWorld.z)).mul(0.48);
    const tearDrive = clamp(
      skinTempNode.mul(0.52).add(reCrackNode.mul(0.48)).add(uChamberState.w.mul(0.42)).sub(quenchedNode.mul(0.42)),
      0.0,
      1.0
    );
    // Disciplined threshold: 55-75% glossy dark obsidian crust plates split by 25-45% blazing liquid channels!
    const riftThresh = clamp(float(0.86).sub(tearDrive.mul(0.26)), 0.56, 0.90);
    const liquidRiftMask = smoothstep(riftThresh, riftThresh.add(0.11), ridgeMain);

    // Ultra-fine smoldering hairline quench cracks retained cleanly between knitted obsidian plates
    const quenchHairline = smoothstep(float(0.91), float(0.982), ridgeMain).mul(
      smoothstep(float(0.06), float(0.28), max(coreTempNode, skinTempNode))
    );

    // Dark obsidian crust plate temperature vs incandescent convective liquid channel temperature
    const crustIslandTemp = clamp(
      skinTempNode.mul(0.30).sub(marginCool).add(fbmMacro.sub(0.5).mul(0.14)),
      0.0,
      0.28
    );
    const channelTemp = clamp(
      max(coreTempNode.mul(0.88), skinTempNode.add(0.22)).add(
        smoothstep(float(0.78), float(0.97), ridgeMain).mul(0.14)
      ),
      0.0,
      0.92
    );
    const riftStrength = clamp(
      max(
        liquidRiftMask.mul(float(0.58).add(tearDrive.mul(0.42))),
        quenchHairline.mul(0.68)
      ),
      0.0,
      1.0
    );
    const finalTemp = mix(crustIslandTemp, channelTemp, riftStrength);

    // Glossy jet-black obsidian skin rafts (roughness ~0.13) contrasting against viscous molten channels
    const baseRough = evalMagmaRoughnessTsl(finalTemp);
    const obsidianRaftBoost = smoothstep(float(0.24), float(0.04), finalTemp).mul(float(1.0).sub(riftStrength));
    const finalRough = clamp(mix(baseRough, float(0.125), obsidianRaftBoost), 0.12, 0.54);

    mat.colorNode = evalBlackbodyAlbedoTsl(finalTemp);
    mat.emissiveNode = evalBlackbodyEmissiveTsl(finalTemp, float(1.42));
    mat.roughnessNode = finalRough;

    // Beveled obsidian crust-plate edge normals + ropy magma shear folds
    const nBase = normalize(normalLocal);
    const rEps = float(0.008);
    const rX = float(1.0).sub(abs(mineralToothFbm(warpPos.add(vec3(rEps, 0.0, 0.0)), float(10.5)).mul(2.0).sub(1.0)));
    const rY = float(1.0).sub(abs(mineralToothFbm(warpPos.add(vec3(0.0, rEps, 0.0)), float(10.5)).mul(2.0).sub(1.0)));
    const rZ = float(1.0).sub(abs(mineralToothFbm(warpPos.add(vec3(0.0, 0.0, rEps)), float(10.5)).mul(2.0).sub(1.0)));
    const raftEdgeBand = smoothstep(riftThresh.sub(0.05), riftThresh.add(0.05), ridgeMain).mul(
      float(1.0).sub(liquidRiftMask.mul(0.75))
    );
    const crustEdgeGrad = vec3(rX.sub(ridgeMain), rY.sub(ridgeMain), rZ.sub(ridgeMain))
      .div(rEps)
      .mul(raftEdgeBand)
      .mul(0.08);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);
    const scF = float(12.5);
    const eF = float(0.16).div(scF);
    const gFlow = k1
      .mul(mineralToothFbm(flowOffset.add(k1.mul(eF)), scF))
      .add(k2.mul(mineralToothFbm(flowOffset.add(k2.mul(eF)), scF)))
      .add(k3.mul(mineralToothFbm(flowOffset.add(k3.mul(eF)), scF)))
      .add(k4.mul(mineralToothFbm(flowOffset.add(k4.mul(eF)), scF)))
      .mul(1.5625)
      .mul(mix(float(0.32), float(0.16), obsidianRaftBoost));

    const totalGrad = gFlow.sub(crustEdgeGrad);
    const tanGrad = totalGrad.sub(nBase.mul(dot(totalGrad, nBase)));
    mat.normalNode = transformNormalToView(normalize(nBase.sub(tanGrad)));

    return mat;
  }

  const dikeMagma = createIntrusionMaterial(false);
  const sillMagma = createIntrusionMaterial(true);

  // -------------------------------------------------------------------------
  // 8. Instanced Viscoplastic Braided Pahoehoe Magma Lobes
  //    Silky mirror-glossy jet-black obsidian cord ridges pulling apart over blazing
  //    realgar-gold/cinnabar liquid seams, knitting into ropy glass with hairline quench-cracks!
  //    Zero multiplicative positionNode scaling (strict Three.js r187 InstancedMesh contract!)
  // -------------------------------------------------------------------------
  const pahoehoeLobe = new THREE.MeshStandardNodeMaterial({
    metalness: 0.08,
    roughness: 0.135,
    envMapIntensity: 1.82,
  });
  {
    const aLobeThermal = attribute('aLobeThermal', 'vec4');
    const coreTemp = clamp(aLobeThermal.x, 0.0, 1.0);
    const crustTemp = clamp(aLobeThermal.y, 0.0, 1.0);
    const crackIntensity = clamp(aLobeThermal.z, 0.0, 1.0);
    const seed = aLobeThermal.w;

    // Pure un-instanced geometry coordinates + per-lobe seed
    const pSeed = positionGeometry.add(vec3(seed.mul(1.7), seed.mul(2.3), seed.mul(0.9)));
    const flowCrawl = uTime.mul(float(0.05).add(crustTemp.mul(0.24)));

    // Braided arcuate transverse pahoehoe cord folds curving smoothly across the bulbous advancing tongue
    const warp = mineralToothFbm(pSeed.mul(vec3(1.4, 1.8, 1.4)), float(1.8));
    const flowArc = positionGeometry.z
      .mul(5.8)
      .add(positionGeometry.x.mul(positionGeometry.x).mul(2.6))
      .add(warp.mul(2.0))
      .sub(flowCrawl.mul(2.2));
    const ropyRidge = sin(flowArc).mul(0.5).add(0.5);
    const ropyTrough = float(1.0).sub(ropyRidge);

    // Broad coherent glossy obsidian crust plates separated by clean glowing magma seams (ZERO noisy static speckle!)
    const pCrack = pSeed.mul(vec3(1.35, 1.55, 1.35)).add(vec3(warp.mul(0.45), flowCrawl.negate().mul(0.5), 1.9));
    const foldBroad = mineralToothFbm(pCrack, float(2.1));
    const ridgeVal = float(1.0).sub(abs(foldBroad.mul(2.0).sub(1.0)));

    // Clean bold tearing veins nestled along ropy troughs + ultra-fine quench-cracks when cooled
    const crackSignal = ridgeVal.mul(0.68).add(ropyTrough.mul(0.32));
    const crackThresh = clamp(float(0.88).sub(crackIntensity.mul(0.22)).sub(crustTemp.mul(0.14)), 0.58, 0.91);
    const activeTearMask = smoothstep(crackThresh, float(0.955), crackSignal).mul(
      smoothstep(float(0.04), float(0.20), max(crackIntensity, crustTemp))
    );
    const quenchCrackMask = smoothstep(float(0.92), float(0.985), crackSignal).mul(
      smoothstep(float(0.08), float(0.28), coreTemp)
    );
    const crackMask = clamp(max(activeTearMask, quenchCrackMask.mul(0.62)), 0.0, 1.0);

    // Glossy jet-black obsidian cord ridges (65-85% of lobe!) vs hotter blistering creases
    const topExposure = smoothstep(float(-0.25), float(0.60), normalLocal.y);
    const cordCrownChill = ropyRidge.mul(0.58);
    const localSkinTemp = clamp(
      crustTemp.mul(float(0.62).sub(topExposure.mul(0.38)).sub(cordCrownChill)).add(ropyTrough.mul(crustTemp).mul(0.22)),
      0.0,
      0.46
    );

    // Pulling dark obsidian plates apart exposes glowing cinnabar -> realgar-amber liquid beneath (never clipped white confetti!)
    const exposedCrackHeat = clamp(
      max(coreTemp.mul(0.86), localSkinTemp.add(0.28)).mul(float(0.52).add(crackIntensity.mul(0.48))),
      0.0,
      0.90
    );
    const finalTemp = mix(localSkinTemp, exposedCrackHeat, crackMask);

    // Mirror-glossy jet-black volcanic obsidian glass (roughness ~0.125) on cooled corded skin islands
    const baseRough = evalMagmaRoughnessTsl(finalTemp);
    const obsidianGlassBoost = smoothstep(float(0.24), float(0.04), finalTemp).mul(float(1.0).sub(crackMask));
    const finalRough = clamp(mix(baseRough, float(0.125), obsidianGlassBoost), 0.12, 0.52);

    pahoehoeLobe.colorNode = evalBlackbodyAlbedoTsl(finalTemp);
    pahoehoeLobe.emissiveNode = evalBlackbodyEmissiveTsl(finalTemp, float(1.38));
    pahoehoeLobe.roughnessNode = finalRough;

    // Silky smooth transverse ropy cord normal gradient (genuine glossy folded lava pillows!)
    const nBase = normalize(normalLocal);
    const cordSlope = cos(flowArc).mul(mix(float(0.34), float(0.26), obsidianGlassBoost));
    const cordGrad = vec3(
      positionGeometry.x.mul(5.2).mul(cordSlope).mul(0.10),
      float(0.0),
      cordSlope.mul(0.42)
    );
    const ropyNormalCoord = vec3(
      positionGeometry.x.mul(1.4),
      positionGeometry.y.mul(1.8),
      flowArc.mul(0.24)
    ).add(vec3(seed, seed.mul(0.7), 0.0));
    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);
    const scR = float(2.6);
    const eR = float(0.16).div(scR);
    const gSub = k1
      .mul(mineralToothFbm(ropyNormalCoord.add(k1.mul(eR)), scR))
      .add(k2.mul(mineralToothFbm(ropyNormalCoord.add(k2.mul(eR)), scR)))
      .add(k3.mul(mineralToothFbm(ropyNormalCoord.add(k3.mul(eR)), scR)))
      .add(k4.mul(mineralToothFbm(ropyNormalCoord.add(k4.mul(eR)), scR)))
      .mul(1.5625)
      .mul(mix(float(0.18), float(0.10), obsidianGlassBoost));

    const totalLobeGrad = gSub.add(cordGrad);
    const tanLobeGrad = totalLobeGrad.sub(nBase.mul(dot(totalLobeGrad, nBase)));
    pahoehoeLobe.normalNode = transformNormalToView(normalize(nBase.sub(tanLobeGrad)));
  }

  // -------------------------------------------------------------------------
  // 8b. Dislodged Angular Basalt & Yellow-Ochre Tuff Spall Chips (1 Draw Call)
  // -------------------------------------------------------------------------
  const rockChip = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.86,
    envMapIntensity: 0.85,
  });
  {
    const aChipTone = attribute('aChipTone', 'float');
    const grain = mineralToothFbm(positionGeometry, float(6.5));
    const basaltChipCol = mix(
      vec3(0.032, 0.032, 0.035),
      vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.45),
      grain
    );
    const tuffChipCol = mix(
      vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear).mul(0.68),
      vec3(...MINERAL_PALETTE.burntSienna.rInfLinear).mul(0.55),
      grain
    );
    rockChip.colorNode = mix(basaltChipCol, tuffChipCol, clamp(aChipTone, 0.0, 1.0));
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
    const c = evalBlackbodyAlbedoTsl(aEmberHeat).add(evalBlackbodyEmissiveTsl(aEmberHeat, float(1.60)));
    emberSpark.colorNode = c;
    emberSpark.opacityNode = smoothstep(float(0.02), float(0.24), aEmberHeat);
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
    rockChip,
    emberSpark,
    uTime,
    uChamberState,
    uDikeHeights,
    uDikeThermal,
    uSillState,
    uFractureState,
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
      rockChip.dispose();
      emberSpark.dispose();
    },
  };
}
