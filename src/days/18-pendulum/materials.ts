import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  acos,
  attribute,
  cameraPosition,
  clamp,
  distance,
  dot,
  exp,
  float,
  length,
  max,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  sin,
  smoothstep,
  sqrt,
  texture,
  transformNormalToView,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import {
  HAMMERED_SLOPE_DECODE_MAX,
  LAEA_UV_SCALE,
  type HammeredIronSurfaceMaps,
} from './domeSurface';
import { DOME_MAX_PHI, DOME_RADIUS, MAX_STONES } from './physics';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export const MAX_IMPACT_RIPPLES = 4;

export interface PendulumMaterials {
  domeIron: THREE.MeshStandardNodeMaterial;
  domeUnderside: THREE.MeshStandardNodeMaterial;
  boneCalciteBob: THREE.MeshStandardNodeMaterial;
  boneCalciteRim: THREE.MeshStandardNodeMaterial;
  agedBronze: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  plumbWire: THREE.MeshStandardNodeMaterial;
  cinnabarStone: THREE.MeshStandardNodeMaterial;
  verdigrisStone: THREE.MeshStandardNodeMaterial;
  trailRibbon: THREE.MeshBasicNodeMaterial;
  uBobWorldPos: { value: THREE.Vector3 };
  uBobTipPos: { value: THREE.Vector3 };
  uBobHover: { value: number };
  stoneAOData: THREE.Vector4[];
  impactOriginsAge: THREE.Vector4[]; // xyz = dome unit normal, w = ageSec (-1 inactive)
  uImpactAmps: { value: THREE.Vector4 };
  dispose: () => void;
}

/**
 * Analytical 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Zero screen-space dpdx/dpdy foil spikes and strictly uniform control flow.
 */
const perturbNormal3DFbm = Fn(
  ([baseNorm_in, pos_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const sc = float(scale_in);
    const eps = float(0.012);

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), sc);
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), sc);
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), sc);
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), sc);

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).div(eps.mul(4.0));
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

export function createPendulumMaterials(surfaceMaps: HammeredIronSurfaceMaps): PendulumMaterials {
  const uBobWorldPos = uniform(new THREE.Vector3(0, 0.35, 0));
  const uBobTipPos = uniform(new THREE.Vector3(0, 0.18, 0));
  const uBobHover = uniform(0);

  // 18-stone real-time contact AO uniform array: xyz = dome unit normal, w = angularRadius * seatFactor
  const stoneAOData: THREE.Vector4[] = [];
  for (let i = 0; i < MAX_STONES; i++) {
    stoneAOData.push(new THREE.Vector4(0, 1, 0, 0));
  }
  const uStoneAO = uniformArray<'vec4'>(stoneAOData, 'vec4');

  // 4-slot acoustic flexural impact ripple ring buffer: xyz = impact unit normal, w = ageSec (< 0 inactive)
  const impactOriginsAge: THREE.Vector4[] = [];
  for (let i = 0; i < MAX_IMPACT_RIPPLES; i++) {
    impactOriginsAge.push(new THREE.Vector4(0, 1, 0, -1));
  }
  const uImpactRipples = uniformArray<'vec4'>(impactOriginsAge, 'vec4');
  const uImpactAmps = uniform(new THREE.Vector4(0, 0, 0, 0));

  const pCalcite = createPigmentUniforms('boneCalcite');
  const pCinnabar = createPigmentUniforms('cinnabarVermilion');
  const pVerdigris = createPigmentUniforms('verdigrisMalachite');

  // Proper TSL Fn() stack for 4-slot acoustic flexural impact waves (returns vec4(dNx, dNy, dNz, envSum))
  const evalAcousticRipples = Fn(([nBase_in]: [TslNode]) => {
    const nBase = normalize(vec3(nBase_in));
    const normalDelta = vec3(0.0, 0.0, 0.0).toVar();
    const envSum = float(0.0).toVar();

    for (let k = 0; k < MAX_IMPACT_RIPPLES; k++) {
      const imp = vec4(uImpactRipples.element(k));
      const amp =
        k === 0
          ? uImpactAmps.x
          : k === 1
            ? uImpactAmps.y
            : k === 2
              ? uImpactAmps.z
              : uImpactAmps.w;
      const age = imp.w;
      const activeGate = smoothstep(float(0.002), float(0.03), age).mul(
        float(1.0).sub(smoothstep(float(0.50), float(0.92), age))
      );
      const origin = normalize(imp.xyz);
      const cosAng = clamp(dot(nBase, origin), -0.9999, 0.9999);
      const ang = acos(cosAng);
      const frontR = age.mul(2.85);
      const distFromFront = ang.sub(frontR);
      const env = exp(distFromFront.mul(distFromFront).mul(-42.0))
        .mul(exp(age.mul(-3.8)))
        .mul(amp)
        .mul(activeGate);
      const waveSlope = sin(distFromFront.mul(34.0)).mul(env).mul(0.14);

      const rawOut = nBase.sub(origin.mul(cosAng));
      const outDir = rawOut.div(max(float(0.015), length(rawOut)));
      normalDelta.addAssign(outDir.mul(waveSlope));
      envSum.addAssign(env);
    }

    return vec4(normalDelta, envSum);
  });

  // Proper TSL Fn() stack for real-time 18-stone analytical velvety contact AO on the curved iron dome
  const evalStoneContactAO = Fn(([nBase_in]: [TslNode]) => {
    const nBase = normalize(vec3(nBase_in));
    const occ = float(0.0).toVar();
    Loop(MAX_STONES, ({ i }) => {
      const s = vec4(uStoneAO.element(i));
      const seatRad = s.w;
      const chordD = length(nBase.sub(s.xyz));
      const coreAO = float(1.0).sub(smoothstep(seatRad.mul(0.50), seatRad.mul(1.42), chordD));
      const softAO = float(1.0).sub(smoothstep(seatRad.mul(0.95), seatRad.mul(2.15), chordD));
      const valid = smoothstep(float(0.001), float(0.012), seatRad);
      occ.assign(max(occ, coreAO.mul(0.82).add(softAO.mul(0.24)).mul(valid)));
    });
    return clamp(occ, 0.0, 0.90);
  });

  // -------------------------------------------------------------------------
  // 1. Oil-Blackened Hand-Planished Hammered Iron Dome
  // -------------------------------------------------------------------------
  const domeIron = new THREE.MeshStandardNodeMaterial({
    metalness: 0.78,
    roughness: 0.48,
    envMapIntensity: 1.05,
  });
  {
    const nBase = normalize(positionLocal);

    // Exact Lambert Azimuthal Equal-Area UV projection (C-infinity smooth at apex, zero seam)
    const laeaK = sqrt(float(2.0).div(max(float(0.05), float(1.0).add(nBase.y)))).mul(
      float(LAEA_UV_SCALE)
    );
    const laeaUV = vec2(float(0.5).add(nBase.x.mul(laeaK)), float(0.5).add(nBase.z.mul(laeaK)));

    // Strictly uniform control-flow texture fetches
    const staticSample = texture(surfaceMaps.staticMap, laeaUV);
    const dynSample = texture(surfaceMaps.dynamicMap, laeaUV);

    const gT = staticSample.r.mul(2.0).sub(1.0).mul(float(HAMMERED_SLOPE_DECODE_MAX));
    const gB = staticSample.g.mul(2.0).sub(1.0).mul(float(HAMMERED_SLOPE_DECODE_MAX));
    const ridgeMask = clamp(staticSample.b, 0.0, 1.0);
    const cavityMask = clamp(staticSample.a, 0.0, 1.0);

    const burnishScuff = clamp(dynSample.r, 0.0, 1.0);
    const bobWakeSheen = clamp(dynSample.g, 0.0, 1.0);
    const cinnabarDust = clamp(dynSample.b, 0.0, 1.0);
    const verdigrisDust = clamp(dynSample.a, 0.0, 1.0);

    // Pole-safe orthonormal tangent basis (T, B) identical to domeSurface.ts CPU baker
    const invOnePlusNy = float(1.0).div(max(float(0.05), float(1.0).add(nBase.y)));
    const T = vec3(
      float(1.0).sub(nBase.x.mul(nBase.x).mul(invOnePlusNy)),
      nBase.x.negate(),
      nBase.x.mul(nBase.z).mul(invOnePlusNy).negate()
    );
    const B = vec3(
      nBase.x.mul(nBase.z).mul(invOnePlusNy).negate(),
      nBase.z.negate(),
      float(1.0).sub(nBase.z.mul(nBase.z).mul(invOnePlusNy))
    );

    const microScale = mineralToothFbm(positionLocal, float(46.0));
    const fineGrain = mineralToothFbm(positionLocal.add(vec3(2.7, 5.1, 1.9)), float(96.0));

    const ripplePack = evalAcousticRipples(nBase);
    const rippleNormalDelta = ripplePack.xyz;
    const rippleHighlight = ripplePack.w;

    // Gentle, authentic hand-planished hammer facets (reads softly under studio scrims, zero disco glare)
    const peenStrength = float(0.118);
    const rawPerturbed = normalize(
      nBase
        .sub(T.mul(gT.mul(peenStrength)))
        .sub(B.mul(gB.mul(peenStrength)))
        .sub(rippleNormalDelta)
        .add(T.mul(fineGrain.sub(0.5).mul(0.022)))
        .add(B.mul(microScale.sub(0.5).mul(0.022)))
    );
    domeIron.normalNode = transformNormalToView(rawPerturbed);

    // Real-time analytical contact AO directly beneath all 18 seated magnetic stones
    const stoneOcc = evalStoneContactAO(nBase);

    // Subtle soft drop-shadow + warm Bone Calcite bounce directly beneath grazing bob
    const dBobTip = distance(positionWorld, uBobTipPos);
    const dBobCore = distance(positionWorld, uBobWorldPos);
    const bobCoreShadow = float(1.0).sub(smoothstep(float(0.09), float(0.44), dBobTip));
    const bobCalciteBounce = float(1.0).sub(smoothstep(float(0.14), float(0.52), dBobCore));

    // Crevice contact darkening where the iron skirt seats into the Bone Calcite collar
    const phi = acos(clamp(nBase.y, -1.0, 1.0));
    const rimSeamAO = smoothstep(float(DOME_MAX_PHI - 0.038), float(DOME_MAX_PHI + 0.005), phi);

    // Tight oil-blackened forged iron palette: concave normal-map bowls carry 90%+ of the highlight
    // breakup without painting bright Voronoi ridge outlines at 1:1 macro zoom!
    const ironCavity = vec3(0.016, 0.018, 0.021);
    const ironBody = vec3(0.023, 0.026, 0.030);
    const ironRidge = vec3(0.029, 0.032, 0.037);
    const ironBurnished = vec3(0.062, 0.066, 0.074);

    const forgedBase = mix(ironCavity, ironBody, smoothstep(float(0.18), float(0.84), cavityMask));
    const withRidges = mix(forgedBase, ironRidge, ridgeMask.mul(0.14).add(microScale.mul(0.10)));

    // Subtle burnished steel skitter tracks along stone sliding paths
    const scuffMask = clamp(burnishScuff.mul(float(0.55).add(ridgeMask.mul(0.45))), 0.0, 1.0);
    const withScuffs = mix(withRidges, ironBurnished, scuffMask.mul(0.46));

    // Ultra-restrained earthy mineral chalk dust trapped subtly in dimple troughs (never blotchy paint!)
    const pocketTrap = float(1.0).sub(cavityMask.mul(0.70));
    const cinnabarRgb = vec3(...MINERAL_PALETTE.cinnabarVermilion.rInfLinear).mul(0.65);
    const verdigrisRgb = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear).mul(0.68);
    const withCinnabarDust = mix(
      withScuffs,
      cinnabarRgb,
      clamp(cinnabarDust.mul(pocketTrap).mul(0.11), 0.0, 0.10)
    );
    const withMineralDust = mix(
      withCinnabarDust,
      verdigrisRgb,
      clamp(verdigrisDust.mul(pocketTrap).mul(0.11), 0.0, 0.10)
    );

    const calciteTint = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
    const withBobReact = withMineralDust
      .add(calciteTint.mul(bobCalciteBounce.mul(0.026).mul(float(0.65).add(ridgeMask.mul(0.35)))))
      .add(vec3(0.018, 0.020, 0.024).mul(bobWakeSheen.mul(0.32)))
      .add(vec3(0.032, 0.034, 0.038).mul(clamp(rippleHighlight, 0.0, 1.0).mul(0.42)));

    const totalAO = float(1.0)
      .sub(stoneOcc)
      .mul(float(1.0).sub(bobCoreShadow.mul(0.24)))
      .mul(float(1.0).sub(rimSeamAO.mul(0.44)));

    domeIron.colorNode = withBobReact.mul(max(float(0.08), totalAO));

    // Oil-blackened satin forged-iron roughness (broad velvety softbox roll; zero Voronoi web glare)
    const dustMatteLift = clamp(cinnabarDust.add(verdigrisDust), 0.0, 1.0).mul(0.10);
    domeIron.roughnessNode = clamp(
      float(0.52)
        .sub(cavityMask.mul(0.06))
        .sub(ridgeMask.mul(0.035))
        .sub(scuffMask.mul(0.13))
        .sub(bobWakeSheen.mul(0.04))
        .add(fineGrain.sub(0.5).mul(0.05))
        .add(dustMatteLift),
      0.28,
      0.66
    );
    domeIron.metalnessNode = clamp(
      float(0.78)
        .add(ridgeMask.mul(0.04))
        .add(scuffMask.mul(0.08))
        .sub(dustMatteLift.mul(1.2)),
      0.54,
      0.90
    );
  }

  // 1b. Clean dark forged underside plate
  const domeUnderside = new THREE.MeshStandardNodeMaterial({
    metalness: 0.72,
    roughness: 0.56,
    envMapIntensity: 0.82,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(26.0));
    domeUnderside.colorNode = mix(
      vec3(0.014, 0.016, 0.020),
      vec3(0.030, 0.033, 0.038),
      smoothstep(float(0.25), float(0.80), grain)
    );
    domeUnderside.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(32.0), float(0.030));
    domeUnderside.roughnessNode = clamp(float(0.50).add(grain.mul(0.14)), 0.40, 0.70);
  }

  // -------------------------------------------------------------------------
  // 2. Honed & Veined Bone Calcite (Faceted Teardrop Pendulum Bob)
  // -------------------------------------------------------------------------
  const boneCalciteBob = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.36,
    envMapIntensity: 0.90,
  });
  {
    const warp = mineralToothFbm(positionLocal.mul(vec3(3.2, 2.4, 3.2)), float(2.6));
    const strataCoord = positionLocal.y
      .mul(16.0)
      .add(positionLocal.x.mul(5.2))
      .add(warp.mul(3.4));
    const bandWave = sin(strataCoord).mul(0.5).add(0.5);
    const fineBand = sin(strataCoord.mul(2.85).add(1.3)).mul(0.5).add(0.5);
    const fineGrain = mineralToothFbm(positionLocal.add(vec3(5.2, 1.3, 8.7)), float(30.0));

    const thickness = float(0.95).add(bandWave.mul(0.65)).add(fineGrain.mul(0.25));
    const warmBacking = mix(
      vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear).mul(0.86),
      vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear).mul(0.74),
      bandWave.mul(0.46)
    );
    const glazed = kubelkaMunkGlaze(pCalcite.K, pCalcite.S, thickness, warmBacking);

    const amberVein = mix(
      vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear).mul(0.70),
      vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.86),
      fineBand
    );
    const veinMask = smoothstep(float(0.54), float(0.90), bandWave.mul(0.72).add(fineBand.mul(0.28)));
    const veined = mix(glazed, amberVein, veinMask.mul(0.34));

    const colletSeam = smoothstep(float(0.088), float(0.118), positionLocal.y)
      .add(smoothstep(float(-0.082), float(-0.112), positionLocal.y));

    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const rimWrap = pow(
      clamp(float(1.0).sub(max(float(0.0), dot(normalize(normalWorld), viewDir))), 0.0, 1.0),
      2.6
    );
    const sssWarmth = vec3(0.14, 0.10, 0.055).mul(rimWrap.mul(0.36));

    boneCalciteBob.colorNode = veined
      .mul(float(1.0).sub(clamp(colletSeam, 0.0, 1.0).mul(0.28)))
      .add(sssWarmth);
    boneCalciteBob.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(36.0), float(0.016));
    boneCalciteBob.roughnessNode = clamp(
      float(0.32).add(fineGrain.mul(0.18)).add(veinMask.mul(0.06)),
      0.24,
      0.56
    );
  }

  // -------------------------------------------------------------------------
  // 2b. Slim Honed Architectural Bone Calcite Equatorial Rim
  // -------------------------------------------------------------------------
  const boneCalciteRim = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.56,
    envMapIntensity: 0.76,
  });
  {
    const tooth = mineralToothFbm(positionLocal, float(9.5));
    const fineGrain = mineralToothFbm(positionLocal.add(vec3(4.1, 2.7, 6.3)), float(32.0));
    const thickness = float(0.90).add(tooth.mul(0.65));
    const backing = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear).mul(0.70);
    const glazed = kubelkaMunkGlaze(pCalcite.K, pCalcite.S, thickness, backing).mul(0.86);

    const honed = mix(
      glazed,
      vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.74),
      smoothstep(float(0.68), float(0.92), fineGrain).mul(0.16)
    );

    const rimLocalY = DOME_RADIUS * Math.cos(DOME_MAX_PHI);
    const ironSeatAO = smoothstep(float(rimLocalY - 0.002), float(rimLocalY + 0.016), positionLocal.y);
    boneCalciteRim.colorNode = honed.mul(float(1.0).sub(ironSeatAO.mul(0.34)));
    boneCalciteRim.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(38.0), float(0.022));
    boneCalciteRim.roughnessNode = clamp(float(0.46).add(fineGrain.mul(0.22)), 0.36, 0.72);
  }

  // -------------------------------------------------------------------------
  // 3. Dark Oxidized Forged Bronze (Overhead Gimbal Horseshoe, Swivel Clevis & Axle)
  // -------------------------------------------------------------------------
  const agedBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.30,
    envMapIntensity: 1.20,
  });
  {
    // Isotropic fine patinated bronze grain (zero horizontal rings along vertical arch legs!)
    const grain = mineralToothFbm(positionLocal, float(22.0));
    const patina = mineralToothFbm(positionLocal.add(vec3(1.9, 6.2, 3.1)), float(9.5));
    const bronzeDark = vec3(0.055, 0.038, 0.027);
    const bronzeWarm = vec3(0.20, 0.142, 0.085);
    agedBronze.colorNode = mix(
      bronzeDark,
      bronzeWarm,
      smoothstep(float(0.32), float(0.80), grain.mul(0.60).add(patina.mul(0.40))).mul(0.58)
    );
    agedBronze.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(34.0),
      float(0.012)
    );
    agedBronze.roughnessNode = clamp(float(0.26).add(grain.mul(0.16)), 0.22, 0.46);
  }

  // -------------------------------------------------------------------------
  // 4. Horological Turned Satin Brass (Pulley Sheave, Counterweight, Bob Collets, Finials)
  // -------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.24,
    envMapIntensity: 1.24,
  });
  {
    // Fine isotropic satin micro-grain (eliminates horizontal stacked-ring banding at 1:1 macro zoom!)
    const brush = mineralToothFbm(positionLocal, float(32.0));
    const fineSatin = mineralToothFbm(positionLocal.add(vec3(3.7, 1.4, 5.2)), float(74.0));
    const brassBright = vec3(0.72, 0.57, 0.28);
    const brassAged = vec3(0.52, 0.40, 0.185);
    const hoverLift = uBobHover.mul(0.10);

    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.24), float(0.76), brush.mul(0.65).add(fineSatin.mul(0.35)))
    ).add(brassBright.mul(hoverLift));
    satinBrass.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(42.0),
      float(0.006)
    );
    satinBrass.roughnessNode = clamp(
      float(0.22).add(brush.mul(0.08)).add(fineSatin.sub(0.5).mul(0.03)),
      0.18,
      0.35
    );
  }

  // -------------------------------------------------------------------------
  // 4b. Drawn Taut Brass Plumb Wire (Sleek, Uniform Drawn Filament — Zero Chain/Spring Banding!)
  // -------------------------------------------------------------------------
  const plumbWire = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.32,
  });
  {
    // Subtle longitudinal drawn-wire polish (compressed in Y, fine in XZ -> sleek taut metallic strand)
    const wireSheen = mineralToothFbm(positionLocal.mul(vec3(3.5, 0.45, 3.5)), float(26.0));
    const wireBright = vec3(0.74, 0.60, 0.33);
    const wireMid = vec3(0.58, 0.46, 0.24);
    plumbWire.colorNode = mix(wireMid, wireBright, smoothstep(float(0.25), float(0.75), wireSheen)).add(
      wireBright.mul(uBobHover.mul(0.08))
    );
    plumbWire.roughnessNode = clamp(float(0.20).add(wireSheen.mul(0.07)), 0.17, 0.30);
  }

  // -------------------------------------------------------------------------
  // 5. Raw Cleaved Magnetite Lodestone Crystals with High-Frequency Crystalline Grain
  //    & Organic Jagged Mineral Crown Border
  // -------------------------------------------------------------------------
  function createCleavedStoneMaterial(isCinnabar: boolean): THREE.MeshStandardNodeMaterial {
    const mat = new THREE.MeshStandardNodeMaterial({
      metalness: 0.45,
      roughness: 0.65,
      envMapIntensity: 1.05,
    });

    const aCapMask = attribute('aCapMask', 'float');
    const aBottomAO = attribute('aBottomAO', 'float');

    // Crisp high-frequency crystalline ore fracture (replaces cloudy low-freq grey patches!)
    const oreMedium = mineralToothFbm(positionLocal, float(46.0));
    const oreFine = mineralToothFbm(positionLocal.add(vec3(7.3, 2.9, 4.1)), float(108.0));
    const oreSpark = mineralToothFbm(positionLocal.add(vec3(2.1, 11.7, 6.4)), float(185.0));
    const crystalGlint = smoothstep(float(0.74), float(0.93), oreSpark);

    // Organic jagged mineral encrustation border right across the beveled crown rim
    const borderNoise = oreMedium.sub(0.5).mul(0.46).add(oreFine.sub(0.5).mul(0.26));
    const capBlend = smoothstep(float(0.36), float(0.64), aCapMask.add(borderNoise));

    // Tight dark charcoal-obsidian magnetite ore with razor micro-crystal glints
    const lodeDeep = vec3(0.012, 0.014, 0.018);
    const lodeMid = vec3(0.025, 0.028, 0.033);
    const lodeSpark = vec3(0.115, 0.120, 0.132);
    const bodyRgb = mix(
      mix(lodeDeep, lodeMid, smoothstep(float(0.26), float(0.76), oreMedium.mul(0.55).add(oreFine.mul(0.45)))),
      lodeSpark,
      crystalGlint.mul(0.58)
    );

    let mineralRgb: TslNode;
    if (isCinnabar) {
      const backing = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear).mul(0.72);
      const thickness = float(0.56).add(oreMedium.mul(0.80)).add(oreFine.mul(0.32));
      const km = kubelkaMunkGlaze(pCinnabar.K, pCinnabar.S, thickness, backing);
      mineralRgb = mix(
        km.mul(0.92),
        vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.42),
        smoothstep(float(0.62), float(0.90), oreFine).mul(0.34)
      );
    } else {
      const backing = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.48);
      const thickness = float(0.60).add(oreMedium.mul(0.88)).add(oreFine.mul(0.32));
      const km = kubelkaMunkGlaze(pVerdigris.K, pVerdigris.S, thickness, backing);
      const mottled = mix(
        km.mul(0.95),
        vec3(...MINERAL_PALETTE.rawUmber.rInfLinear).mul(0.34),
        smoothstep(float(0.60), float(0.88), oreMedium).mul(0.36)
      );
      mineralRgb = mix(
        mottled,
        vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear).mul(0.94),
        smoothstep(float(0.68), float(0.92), oreFine).mul(0.28)
      );
    }

    // Weathered dark mineral-to-ore transition edge
    const edgeWeather = capBlend.mul(float(1.0).sub(capBlend)).mul(4.0);
    const combinedRgb = mix(bodyRgb, mineralRgb, capBlend).mul(
      float(1.0).sub(edgeWeather.mul(0.28))
    );

    const seatAO = mix(float(0.40), float(1.0), clamp(aBottomAO, 0.0, 1.0));
    mat.colorNode = combinedRgb.mul(seatAO);

    // Finer high-frequency crystalline cleavage micro-normal
    mat.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(64.0),
      mix(float(0.048), float(0.030), capBlend)
    );

    const oreRough = clamp(
      float(0.70).sub(oreFine.mul(0.16)).sub(crystalGlint.mul(0.36)),
      0.24,
      0.84
    );
    const mineralRough = clamp(float(0.68).add(oreFine.mul(0.20)), 0.56, 0.90);
    mat.roughnessNode = mix(oreRough, mineralRough, capBlend);

    const oreMetal = clamp(float(0.48).add(crystalGlint.mul(0.30)), 0.36, 0.78);
    mat.metalnessNode = mix(oreMetal, float(0.03), capBlend);

    return mat;
  }

  const cinnabarStone = createCleavedStoneMaterial(true);
  const verdigrisStone = createCleavedStoneMaterial(false);

  // -------------------------------------------------------------------------
  // 6. Ephemeral Ultra-Delicate Alpha-Faded Orbital Motion Whisper
  // -------------------------------------------------------------------------
  const trailRibbon = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexColors: true,
  });
  {
    const aAlpha = attribute('aAlpha', 'float');
    trailRibbon.opacityNode = clamp(aAlpha, 0.0, 0.14);
  }

  return {
    domeIron,
    domeUnderside,
    boneCalciteBob,
    boneCalciteRim,
    agedBronze,
    satinBrass,
    plumbWire,
    cinnabarStone,
    verdigrisStone,
    trailRibbon,
    uBobWorldPos,
    uBobTipPos,
    uBobHover,
    stoneAOData,
    impactOriginsAge,
    uImpactAmps,
    dispose: () => {
      domeIron.dispose();
      domeUnderside.dispose();
      boneCalciteBob.dispose();
      boneCalciteRim.dispose();
      agedBronze.dispose();
      satinBrass.dispose();
      plumbWire.dispose();
      cinnabarStone.dispose();
      verdigrisStone.dispose();
      trailRibbon.dispose();
    },
  };
}
