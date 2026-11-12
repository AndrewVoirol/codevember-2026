import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  dot,
  float,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  sin,
  smoothstep,
  uniform,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

/**
 * Day 09 — Membrane:
 * "Polished Mineral Marbles on a Stretched Gravity-Well Vellum / Latex Membrane"
 *
 * Physical & Optical Architecture:
 * 1. Suspended Brass & Steam-Bent Walnut Tension Hoop Apparatus:
 *    - A horizontal `R = 1.48` drum tension hoop clamped by a burnished Goethite
 *      brass counter-hoop, a shallow steam-bent walnut collar (open underneath
 *      so deep sagging pockets remain visible in silhouette), and 12 machined
 *      brass tension claws + turnbuckle rods spaced at 30° around the perimeter.
 * 2. Coupled 2-Way Gravity-Well Funnel + 2D Transverse Drumhead Wave Physics:
 *    - Parameterized on a singularity-free `104 × 104` concentric Elliptical Disc
 *      mesh (`0` central pole pinch).
 *    - Every marble pressing into the membrane contributes a wide logarithmic-Bessel
 *      tension depression `G_k(x, z)` pinned strictly to `0` at the brass rim,
 *      creating a true long-range radial slope field (`-∇h`) across the sheet so
 *      dropped stones roll, curve into orbits, clack together with 3D sphere-sphere
 *      collisions & rolling spin, and coalesce into a deep central sagging pocket.
 *    - Simultaneously, a 2D damped elastic drumhead wave PDE (`_waveDisp`, `_waveVel`)
 *      propagates concentric impact ripples and rim reflections on every strike,
 *      bounce, and trampoline snap.
 * 3. Translucent Natural Latex / Calfskin Vellum Drumskin Optics (`membraneMat`):
 *    - Warm Bone Calcite, Warm Kaolin, and Raw Amber semi-translucent skin with
 *      subtle micro-fiber tooth.
 *    - Exhibits coupled **Beer-Lambert optical thinning** (warm amber subsurface
 *      transmission glowing through stretched funnel walls from underneath),
 *      **elastomer stress-whitening** along high-strain slopes, and **contact AO
 *      darkening** directly beneath each heavy mineral sphere.
 * 4. Six Procedural Polished Mineral Stone Varieties:
 *    - Lapis Lazuli w/ Pyrite flecks, Banded Cinnabar-Calcite Fortification Agate,
 *      Translucent Realgar Honey-Amber, Celadon Nephrite Jade, Specular Hematite
 *      Ironstone, and Chatoyant Yellow-Ochre Tiger's-Eye—evaluated in `positionLocal`
 *      so 3D veins physically roll as each sphere's quaternion updates.
 */

// ---------------------------------------------------------------------------
// Apparatus & Singularity-Free Disc Grid Constants (Zero Per-Frame GC)
// ---------------------------------------------------------------------------
const GRID_STEPS = 104;
const GRID_STRIDE = GRID_STEPS + 1;
const TOTAL_VERTS = GRID_STRIDE * GRID_STRIDE;

const MEMBRANE_RADIUS = 1.48;
const HOOP_RADIUS = 1.508;
const WALNUT_COLLAR_RADIUS = 1.542;
const LUG_COUNT = 12;

const MAX_MARBLES = 18;
const GRAVITY = 8.4; // Crisp studio tabletop gravity (world units / s^2)
const PHYSICS_SUBSTEPS = 6;

// Pre-allocated Concentric Elliptical Grid-to-Disc Lookup Tables
const _discX = new Float32Array(TOTAL_VERTS); // unit disc x in [-1, 1]
const _discZ = new Float32Array(TOTAL_VERTS); // unit disc z in [-1, 1]
const _discRho = new Float32Array(TOTAL_VERTS); // radial fraction [0..1]
const _worldX = new Float32Array(TOTAL_VERTS); // local membrane X in [-R, R]
const _worldZ = new Float32Array(TOTAL_VERTS); // local membrane Z in [-R, R]
const _rimEnv = new Float32Array(TOTAL_VERTS); // (1 - rho^2), strictly 0 on brass hoop

for (let iv = 0; iv <= GRID_STEPS; iv++) {
  const v = (iv / GRID_STEPS) * 2.0 - 1.0;
  for (let iu = 0; iu <= GRID_STEPS; iu++) {
    const u = (iu / GRID_STEPS) * 2.0 - 1.0;
    const idx = iv * GRID_STRIDE + iu;

    const dx = u * Math.sqrt(Math.max(0.0, 1.0 - 0.5 * v * v));
    const dz = v * Math.sqrt(Math.max(0.0, 1.0 - 0.5 * u * u));
    const rho2 = Math.min(1.0, dx * dx + dz * dz);
    const rho = Math.sqrt(rho2);

    _discX[idx] = dx;
    _discZ[idx] = dz;
    _discRho[idx] = rho;
    _worldX[idx] = dx * MEMBRANE_RADIUS;
    _worldZ[idx] = dz * MEMBRANE_RADIUS;

    // Boundary envelope strictly 0 on outer hoop boundary (|u|=1 or |v|=1)
    const sqEnv = (1.0 - u * u) * (1.0 - v * v);
    _rimEnv[idx] = sqEnv > 0 ? Math.max(0.0, 1.0 - rho2) : 0.0;
  }
}

// Pre-allocated Membrane Vertex Buffers
const _memPositions = new Float32Array(TOTAL_VERTS * 3);
const _memNormals = new Float32Array(TOTAL_VERTS * 3);
// aMembraneMeta:
//   x: radial rho [0..1]
//   y: local areal/slope stretch strain [0..1.5]
//   z: marble contact ambient occlusion [0..1]
//   w: normalized downward funnel depth [0..1]
const _memMeta = new Float32Array(TOTAL_VERTS * 4);

// Pre-allocated 2D Transverse Drumhead Wave PDE Buffers
const _waveDisp = new Float32Array(TOTAL_VERTS);
const _waveVel = new Float32Array(TOTAL_VERTS);
const _waveNext = new Float32Array(TOTAL_VERTS);
const _totalHeight = new Float32Array(TOTAL_VERTS);

// ---------------------------------------------------------------------------
// Marble Pool Data Structures (Pre-Allocated, Zero Runtime Allocations)
// ---------------------------------------------------------------------------
const MINERAL_TYPES = 6; // 0: Lapis, 1: Cinnabar Agate, 2: Realgar Amber, 3: Celadon Jade, 4: Hematite, 5: Ochre Tiger-Eye

interface MarbleRecord {
  active: boolean;
  mineralType: number;
  radius: number;
  mass: number;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  /** Smoothed vertical normal load currently borne by the membrane (for funnel depth) */
  smoothedLoad: number;
  /** True while being dragged directly on the sheet by the pointer */
  isGrabbed: boolean;
  spawnAgeSec: number;
  mesh: THREE.Mesh;
}

// Pre-allocated scratch vectors & matrices for physics, normals, and raycasting
const _dU = new THREE.Vector3();
const _dV = new THREE.Vector3();
const _normVec = new THREE.Vector3();
const _rotAxis = new THREE.Vector3();
const _stepQuat = new THREE.Quaternion();
const _raycaster = new THREE.Raycaster();
const _rimPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const _hitPoint = new THREE.Vector3();
const _localHit = new THREE.Vector3();
const _invApparatusMat = new THREE.Matrix4();
const _sphereHitTmp = new THREE.Vector3();
const _marbleWorldCenter = new THREE.Vector3();

//Scratch arrays for active load-bearing marbles during funnel evaluation
const _activeLoadX = new Float32Array(MAX_MARBLES);
const _activeLoadZ = new Float32Array(MAX_MARBLES);
const _activeLoadR0Sq = new Float32Array(MAX_MARBLES);
const _activeLoadStrength = new Float32Array(MAX_MARBLES);
const _activeContactR2Inv = new Float32Array(MAX_MARBLES);
const _activeInContact = new Uint8Array(MAX_MARBLES);

/**
 * Evaluates the quasi-static logarithmic-Bessel tension funnel height `h <= 0`
 * and analytical horizontal gradient `(dh/dx, dh/dz)` at `(px, pz)` generated
 * by all marbles except optionally `excludeIndex` (or including self-compliance).
 */
const _funnelEvalOut = new Float32Array(3); // [h, dhDx, dhDz]

function evaluateTensionFunnelAt(
  px: number,
  pz: number,
  marbles: readonly MarbleRecord[],
  globalStiffScale: number
): void {
  const r2 = (px * px + pz * pz) / (MEMBRANE_RADIUS * MEMBRANE_RADIUS);
  if (r2 >= 1.0) {
    _funnelEvalOut[0] = 0.0;
    _funnelEvalOut[1] = 0.0;
    _funnelEvalOut[2] = 0.0;
    return;
  }

  const env = 1.0 - r2;
  const dEnvDx = (-2.0 * px) / (MEMBRANE_RADIUS * MEMBRANE_RADIUS);
  const dEnvDz = (-2.0 * pz) / (MEMBRANE_RADIUS * MEMBRANE_RADIUS);

  const rRef = MEMBRANE_RADIUS * 1.12;
  const logRRef = Math.log(rRef);

  let sumPhi = 0.0;
  let sumDPhiDx = 0.0;
  let sumDPhiDz = 0.0;

  for (let k = 0; k < MAX_MARBLES; k++) {
    const m = marbles[k];
    if (!m.active || m.smoothedLoad <= 1e-4) continue;

    const s = m.smoothedLoad * globalStiffScale;
    const dx = px - m.x;
    const dz = pz - m.z;
    const r0 = m.radius * 1.28;
    const distSq = dx * dx + dz * dz + r0 * r0;
    const dist = Math.sqrt(distSq);

    // Negative logarithmic Green's kernel: phi_k(d) = -s * max(0, ln(rRef / dist))
    const logTerm = Math.max(0.0, logRRef - Math.log(dist));
    if (logTerm > 0.0) {
      sumPhi -= s * logTerm;
      // d(-s * (ln rRef - 0.5 ln distSq)) / dx = +s * dx / distSq
      const invDistSq = s / distSq;
      sumDPhiDx += dx * invDistSq;
      sumDPhiDz += dz * invDistSq;
    }
  }

  // Product rule: h(x, z) = env(x, z) * sumPhi(x, z)
  _funnelEvalOut[0] = env * sumPhi;
  _funnelEvalOut[1] = dEnvDx * sumPhi + env * sumDPhiDx;
  _funnelEvalOut[2] = dEnvDz * sumPhi + env * sumDPhiDz;
}

/**
 * Samples the dynamic 2D transverse wave displacement `_waveDisp` bilinearly at `(px, pz)`
 * for smooth marble-to-ripple coupling.
 */
function sampleWaveDispAt(px: number, pz: number): number {
  // Inverse approximate disc-to-square mapping near interior for wave lookup
  const nx = Math.max(-0.99, Math.min(0.99, px / MEMBRANE_RADIUS));
  const nz = Math.max(-0.99, Math.min(0.99, pz / MEMBRANE_RADIUS));
  // Inverse of Fernández-Guasti mapping:
  const x2 = nx * nx;
  const z2 = nz * nz;
  const termA = 2.0 + x2 - z2;
  const termB = 2.0 - x2 + z2;
  const disc = Math.max(0.0, 4.0 * (1.0 - x2 - z2) + (x2 + z2) * (x2 + z2));
  const root = Math.sqrt(disc);
  const u = Math.sign(nx) * Math.sqrt(Math.max(0.0, 0.5 * (termA - root)));
  const v = Math.sign(nz) * Math.sqrt(Math.max(0.0, 0.5 * (termB - root)));

  const fu = Math.max(0, Math.min(GRID_STEPS - 1.001, (u * 0.5 + 0.5) * GRID_STEPS));
  const fv = Math.max(0, Math.min(GRID_STEPS - 1.001, (v * 0.5 + 0.5) * GRID_STEPS));
  const iu = Math.floor(fu);
  const iv = Math.floor(fv);
  const tx = fu - iu;
  const tz = fv - iv;

  const idx00 = iv * GRID_STRIDE + iu;
  const idx10 = idx00 + 1;
  const idx01 = idx00 + GRID_STRIDE;
  const idx11 = idx01 + 1;

  const w0 = _waveDisp[idx00] * (1 - tx) + _waveDisp[idx10] * tx;
  const w1 = _waveDisp[idx01] * (1 - tx) + _waveDisp[idx11] * tx;
  return w0 * (1 - tz) + w1 * tz;
}

/**
 * Injects a vertical momentum impulse into the `104 × 104` drumhead wave PDE around `(px, pz)`.
 */
function injectDrumheadWaveImpulse(
  px: number,
  pz: number,
  impulseStrength: number,
  spreadRadius: number
): void {
  const invSigma2 = 1.0 / Math.max(0.004, spreadRadius * spreadRadius);
  for (let iv = 1; iv < GRID_STEPS; iv++) {
    const row = iv * GRID_STRIDE;
    for (let iu = 1; iu < GRID_STEPS; iu++) {
      const idx = row + iu;
      const dx = _worldX[idx] - px;
      const dz = _worldZ[idx] - pz;
      const d2 = dx * dx + dz * dz;
      if (d2 > spreadRadius * spreadRadius * 6.0) continue;
      _waveVel[idx] += impulseStrength * Math.exp(-d2 * invSigma2) * _rimEnv[idx];
    }
  }
}

export async function bootDay09Membrane(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(9);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 2.42, 3.76],
    fov: 34,
    enableOrbit: false,
    enableShadows: true,
  });

  stage.camera.lookAt(0, -0.12, 0);
  stage.controls.target.set(0, -0.12, 0);

  // Subtle warm amber/calcite under-membrane uplight so stretched funnels
  // transmit a gentle warm subsurface glow from below without blowing out highlights
  const underGlowLight = new THREE.PointLight(0xe8a864, 1.15, 5.0, 1.5);
  underGlowLight.position.set(0.0, -1.05, 0.15);
  stage.scene.add(underGlowLight);

  // -------------------------------------------------------------------------
  // Physical Mineral & Botanical Pigments (`src/core/palette.ts`)
  // -------------------------------------------------------------------------
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const rawUmber = createPigmentUniforms('rawUmber');
  const burntSienna = createPigmentUniforms('burntSienna');
  const hematiteRust = createPigmentUniforms('hematiteRust');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const verdigrisMalachite = createPigmentUniforms('verdigrisMalachite');
  const celadonGlauconite = createPigmentUniforms('celadonGlauconite');
  const azuritePrussian = createPigmentUniforms('azuritePrussian');
  const lapisUltramarine = createPigmentUniforms('lapisUltramarine');
  const sporeSulfur = createPigmentUniforms('sporeSulfur');

  const uSnapFlash = uniform(0.0);

  // -------------------------------------------------------------------------
  // 1. Hardware Materials: Burnished Brass Hoop, Tension Lugs & Walnut Collar
  // -------------------------------------------------------------------------
  const brassMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.23,
    metalness: 0.85,
    clearcoat: 0.6,
    clearcoatRoughness: 0.15,
  });
  {
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const fresnel = clamp(float(1.0).sub(abs(dot(nWorld, viewDir))), 0.0, 1.0);
    const grain = mineralToothFbm(positionLocal.mul(7.0), float(2.4));

    const agedBrass = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.15).add(grain.mul(0.12)),
      yellowOchre.rInf
    );
    brassMat.colorNode = mix(agedBrass, boneCalcite.rInf, fresnel.pow(2.2).mul(0.45).add(0.14));
  }

  const walnutWoodMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.42,
    metalness: 0.06,
    clearcoat: 0.38,
    clearcoatRoughness: 0.28,
  });
  {
    const pLoc = positionLocal;
    const ringCoord = vec3(pLoc.x.mul(5.5), pLoc.y.mul(18.0), pLoc.z.mul(5.5));
    const grain = mineralToothFbm(ringCoord, float(2.6));
    const streaks = sin(pLoc.y.mul(42.0).add(grain.mul(7.5))).mul(0.5).add(0.5);
    const baseWalnut = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.68).add(streaks.mul(0.35)),
      burntSienna.rInf
    );
    walnutWoodMat.colorNode = baseWalnut;
  }

  // -------------------------------------------------------------------------
  // 2. Translucent Natural Latex / Vellum Drumskin Material (`membraneMat`)
  // -------------------------------------------------------------------------
  const membraneMat = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: true,
    roughness: 0.34,
    metalness: 0.0,
    clearcoat: 0.32,
    clearcoatRoughness: 0.26,
  });
  {
    const aMeta = attribute('aMembraneMeta', 'vec4'); // x: rho, y: stretchStrain, z: contactAO, w: funnelDepth
    const pLoc = positionLocal;
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const ndotv = abs(dot(nWorld, viewDir));
    const fresnel = clamp(float(1.0).sub(ndotv), 0.0, 1.0);

    const rho = aMeta.x;
    const strain = clamp(aMeta.y, 0.0, 1.4);
    const contactAO = clamp(aMeta.z, 0.0, 1.0);
    const funnelDepth = clamp(aMeta.w, 0.0, 1.0);

    // Fine natural calfskin vellum / raw latex mottling & micro-fiber tooth
    const hideCloud = mineralToothFbm(
      vec3(pLoc.x.mul(1.65), pLoc.z.mul(1.65), float(0.35)),
      float(2.1)
    );
    const fiberFbm = mineralToothFbm(
      vec3(pLoc.x.mul(4.2), pLoc.z.mul(4.2), float(0.7)),
      float(2.8)
    );
    const finePore = mineralToothFbm(
      vec3(pLoc.x.mul(13.0), pLoc.z.mul(13.0), float(2.1)),
      float(2.2)
    );

    // Subtle concentric drumhead tension-hoop seating patina right against the brass counter-hoop
    const rimSeating = smoothstep(0.88, 1.0, rho);

    // Unstretched vellum/latex base: authentic warm unbleached Bone Calcite + Warm Kaolin parchment
    // (avoiding heavy mustard yellow while keeping organic natural-vellum warmth & subtle clouding)
    const rawLatexBase = mix(
      boneCalcite.rInf,
      warmKaolin.rInf,
      float(0.34).add(hideCloud.mul(0.28)).add(fiberFbm.mul(0.14))
    );
    const warmVellum = kubelkaMunkGlaze(
      yellowOchre.K,
      yellowOchre.S,
      float(0.026).add(hideCloud.mul(0.022)).add(rimSeating.mul(0.085)),
      rawLatexBase
    );

    // Physical Elastomer Thinning & Warm Amber Subsurface Shift under Marble Tension Funnels:
    // Where heavy stones stretch the skin (`strain > 0`), optical thickness thins out,
    // shifting toward warm translucent amber-ivory along the taut funnel walls without white blowout,
    // plus delicate elastic strain-contour sheen!
    const stressStrainMask = smoothstep(0.05, 0.55, strain);
    const amberWarmTint = mix(
      warmVellum,
      mix(yellowOchre.rInf, realgarAmber.rInf, float(0.32)),
      float(0.34)
    );
    const tautCrest = mix(amberWarmTint, boneCalcite.rInf, float(0.38));

    const stretchedSkinColor = mix(
      warmVellum,
      tautCrest,
      stressStrainMask.mul(0.72)
    );

    // Rim seating patina near the brass counter-hoop + crisp contact AO shadow directly under resting stones
    const rimTinted = mix(
      stretchedSkinColor,
      kubelkaMunkGlaze(rawUmber.K, rawUmber.S, float(0.20), warmVellum),
      rimSeating.mul(0.52)
    );
    const aoShadowTint = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.42),
      rawUmber.rInf
    );
    const finalAlbedo = mix(rimTinted, aoShadowTint.mul(0.52), contactAO.mul(0.68));

    membraneMat.colorNode = clamp(finalAlbedo, 0.03, 0.94);

    // Stretched latex becomes tighter and slightly glossier on funnel slopes (`0.38 -> 0.18`)
    membraneMat.roughnessNode = clamp(
      float(0.38).sub(stressStrainMask.mul(0.18)).add(finePore.mul(0.05)),
      0.16,
      0.48
    );

    // Controlled Subsurface Backlight Translucency Glow (warm amber-vellum glow, zero white clipping):
    const transStrength = float(0.03)
      .add(funnelDepth.mul(0.11))
      .add(stressStrainMask.mul(0.09))
      .add(fresnel.pow(2.2).mul(0.08))
      .add(uSnapFlash.mul(0.14))
      .mul(float(1.0).sub(contactAO.mul(0.85)));

    const subsurfaceColor = mix(
      yellowOchre.rInf,
      realgarAmber.rInf,
      float(0.38)
    );
    membraneMat.emissiveNode = subsurfaceColor.mul(transStrength);

    // Semi-translucent vellum opacity: `~0.76` in stretched funnel walls so stones nestled inside
    // softly silhouette through the front lip, rising to `~0.94` on the flat unstretched perimeter
    membraneMat.opacityNode = clamp(
      float(0.93)
        .sub(funnelDepth.mul(0.14))
        .sub(stressStrainMask.mul(0.08))
        .add(rimSeating.mul(0.05)),
      0.74,
      0.97
    );
  }

  // -------------------------------------------------------------------------
  // 3. Six Procedural Polished Mineral Sphere Materials (Object-Space 3D Veins)
  // -------------------------------------------------------------------------
  function buildMineralMarbleMaterials(): THREE.Material[] {
    // 0: Lapis Ultramarine with Golden Pyrite Flecks & Calcite Veins
    const matLapis = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.09,
      metalness: 0.08,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
    });
    {
      const p = positionLocal.mul(2.8);
      const fbm1 = mineralToothFbm(p, float(2.2));
      const fbm2 = mineralToothFbm(p.mul(3.1).add(vec3(4.1, 1.7, 8.3)), float(2.8));
      // Deep Lapis Ultramarine & Azurite matrix crossed by thin Bone Calcite veins & golden Pyrite flecks
      const isCalciteVein = float(1.0).sub(smoothstep(0.0, 0.055, abs(fbm1.sub(0.51))));
      const pyriteMask = smoothstep(0.76, 0.85, fbm2);

      const deepLapis = mix(lapisUltramarine.rInf, azuritePrussian.rInf, fbm1.mul(0.65));
      const withCalcite = mix(deepLapis, boneCalcite.rInf.mul(0.86), isCalciteVein.mul(0.82));
      const pyriteGold = mix(yellowOchre.rInf, sporeSulfur.rInf, float(0.55)).mul(1.25);
      matLapis.colorNode = mix(withCalcite, pyriteGold, pyriteMask);
      matLapis.metalnessNode = mix(float(0.04), float(0.88), pyriteMask);
      matLapis.roughnessNode = mix(float(0.09), float(0.18), pyriteMask);
    }

    // 1: Banded Cinnabar-Vermilion & Bone Calcite Fortification Agate
    const matAgate = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.07,
      metalness: 0.02,
      clearcoat: 1.0,
      clearcoatRoughness: 0.04,
    });
    {
      const p = positionLocal.mul(2.2);
      const warp = mineralToothFbm(p, float(1.8));
      const bandCoord = p.x.mul(1.1).add(p.y.mul(1.4)).add(warp.mul(1.25));
      const b1 = sin(bandCoord.mul(5.4)).mul(0.5).add(0.5);
      const b2 = cos(bandCoord.mul(10.8)).mul(0.5).add(0.5);

      const cinnabarBand = mix(cinnabarVermilion.rInf, burntSienna.rInf, b2.mul(0.52));
      const creamBand = mix(boneCalcite.rInf, realgarAmber.rInf, float(0.24));
      matAgate.colorNode = mix(cinnabarBand, creamBand, smoothstep(0.36, 0.74, b1));
    }

    // 2: Translucent Realgar Honey-Amber Resin w/ Internal Depth Glow
    const matAmber = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.05,
      metalness: 0.0,
      clearcoat: 1.0,
      clearcoatRoughness: 0.03,
    });
    {
      const p = positionLocal.mul(2.6);
      const nWorld = normalize(normalWorld);
      const viewDir = normalize(cameraPosition.sub(positionWorld));
      const ndotv = abs(dot(nWorld, viewDir));
      const swirl = mineralToothFbm(p, float(2.5));

      const coreChord = float(1.45).sub(ndotv.mul(0.85)).add(swirl.mul(0.4));
      const amberTrans = beerLambertTransmittance(realgarAmber.extinction.mul(0.55), coreChord);
      const richAmber = mix(realgarAmber.rInf, sporeSulfur.rInf, ndotv.mul(0.45));
      matAmber.colorNode = mix(richAmber, amberTrans, float(0.5));
      matAmber.emissiveNode = realgarAmber.rInf.mul(ndotv.pow(1.6).mul(0.22));
    }

    // 3: Celadon Nephrite & Verdigris Malachite Banded Jade
    const matJade = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.11,
      metalness: 0.02,
      clearcoat: 0.95,
      clearcoatRoughness: 0.07,
    });
    {
      const p = positionLocal.mul(3.0);
      const nWorld = normalize(normalWorld);
      const viewDir = normalize(cameraPosition.sub(positionWorld));
      const ndotv = abs(dot(nWorld, viewDir));
      const cloud = mineralToothFbm(p, float(2.3));
      const mossVein = smoothstep(0.62, 0.78, mineralToothFbm(p.mul(2.2), float(2.6)));

      const jadeBody = mix(celadonGlauconite.rInf, verdigrisMalachite.rInf, cloud.mul(0.72));
      const veinedJade = mix(jadeBody, rawUmber.rInf.mul(0.65), mossVein.mul(0.45));
      matJade.colorNode = mix(veinedJade, boneCalcite.rInf, float(1.0).sub(ndotv).pow(2.5).mul(0.28));
      matJade.emissiveNode = celadonGlauconite.rInf.mul(0.06);
    }

    // 4: Polished Specular Hematite & Rust-Banded Ironstone
    const matHematite = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.14,
      metalness: 0.82,
      clearcoat: 0.85,
      clearcoatRoughness: 0.08,
    });
    {
      const p = positionLocal.mul(3.4);
      const strata = mineralToothFbm(p, float(2.5));
      const rustStreak = smoothstep(0.58, 0.76, sin(p.y.mul(8.0).add(strata.mul(4.0))).mul(0.5).add(0.5));
      const darkIron = mix(
        vec3(0.16, 0.17, 0.19),
        boneCalcite.rInf.mul(0.62),
        strata.mul(0.45)
      );
      matHematite.colorNode = mix(darkIron, hematiteRust.rInf, rustStreak.mul(0.55));
      matHematite.metalnessNode = mix(float(0.88), float(0.28), rustStreak.mul(0.6));
    }

    // 5: Chatoyant Yellow-Ochre & Raw Umber Tiger's-Eye Quartz
    const matTigerEye = new THREE.MeshPhysicalNodeMaterial({
      roughness: 0.08,
      metalness: 0.18,
      clearcoat: 1.0,
      clearcoatRoughness: 0.04,
    });
    {
      const p = positionLocal;
      const nLoc = normalize(normalLocal);
      const viewDir = normalize(cameraPosition.sub(positionWorld));
      const grain = mineralToothFbm(p.mul(4.0), float(2.3));
      const bands = sin(p.x.mul(16.0).add(grain.mul(4.5))).mul(0.5).add(0.5);
      // Silk-fiber chatoyancy band that glides across the sphere as it rolls
      const chatoyant = abs(dot(nLoc, normalize(vec3(0.8, 0.5, 0.3)))).add(
        abs(dot(normalize(normalWorld), viewDir)).mul(0.35)
      );
      const silkFlash = smoothstep(0.55, 0.98, chatoyant);

      const umberBase = mix(rawUmber.rInf.mul(0.55), yellowOchre.rInf, bands.mul(0.78));
      const goldenSilk = mix(yellowOchre.rInf, sporeSulfur.rInf, float(0.55));
      matTigerEye.colorNode = mix(umberBase, goldenSilk, silkFlash.mul(0.68));
    }

    return [matLapis, matAgate, matAmber, matJade, matHematite, matTigerEye];
  }

  const mineralMaterials = buildMineralMarbleMaterials();

  // -------------------------------------------------------------------------
  // Assemble Suspended Brass & Walnut Tension Drum Apparatus
  // -------------------------------------------------------------------------
  const apparatusGroup = new THREE.Group();
  apparatusGroup.name = 'GravityWellDrumApparatus';
  stage.specimenGroup.add(apparatusGroup);

  // --- A. Rolled Brass Counter-Hoop Clamping the Membrane Perimeter ---
  const brassHoopGeom = new THREE.TorusGeometry(HOOP_RADIUS, 0.036, 24, 128);
  brassHoopGeom.rotateX(Math.PI * 0.5);
  const brassHoopMesh = new THREE.Mesh(brassHoopGeom, brassMat);
  brassHoopMesh.position.y = 0.008;
  brassHoopMesh.castShadow = true;
  brassHoopMesh.receiveShadow = true;
  apparatusGroup.add(brassHoopMesh);

  // Lower Brass Under-Ring (open spider base ring framing the sagging pouch silhouette)
  const lowerBrassRingGeom = new THREE.TorusGeometry(HOOP_RADIUS * 0.99, 0.022, 18, 128);
  lowerBrassRingGeom.rotateX(Math.PI * 0.5);
  const lowerBrassRingMesh = new THREE.Mesh(lowerBrassRingGeom, brassMat);
  lowerBrassRingMesh.position.y = -0.155;
  apparatusGroup.add(lowerBrassRingMesh);

  // --- B. Shallow Open Steam-Bent Walnut Shell Collar ---
  // Open cylinder (`openEnded: true`) so the deep stretched vellum funnel hangs visibly below!
  const walnutShellGeom = new THREE.CylinderGeometry(
    WALNUT_COLLAR_RADIUS,
    WALNUT_COLLAR_RADIUS,
    0.145,
    96,
    1,
    true
  );
  const walnutShellMesh = new THREE.Mesh(walnutShellGeom, walnutWoodMat);
  walnutShellMesh.position.y = -0.072;
  walnutShellMesh.material.side = THREE.DoubleSide;
  apparatusGroup.add(walnutShellMesh);

  // Top & bottom walnut thickness bevel rings so the open wooden cylinder has solid wall thickness
  const walnutLipGeom = new THREE.TorusGeometry(WALNUT_COLLAR_RADIUS - 0.014, 0.016, 14, 96);
  walnutLipGeom.rotateX(Math.PI * 0.5);
  const walnutBottomLip = new THREE.Mesh(walnutLipGeom, walnutWoodMat);
  walnutBottomLip.position.y = -0.144;
  apparatusGroup.add(walnutBottomLip);

  // --- C. 12 Machined Brass Tension Claws & Turnbuckle Rods ---
  const clawHookGeom = new THREE.CylinderGeometry(0.022, 0.022, 0.065, 16);
  const tensionRodGeom = new THREE.CylinderGeometry(0.011, 0.011, 0.22, 14);
  const lugBossGeom = new THREE.CylinderGeometry(0.024, 0.028, 0.075, 16);
  const teeBarGeom = new THREE.CylinderGeometry(0.008, 0.008, 0.072, 12);
  teeBarGeom.rotateZ(Math.PI * 0.5);

  for (let i = 0; i < LUG_COUNT; i++) {
    const theta = (i / LUG_COUNT) * Math.PI * 2 + Math.PI / 12;
    const lugGroup = new THREE.Group();
    lugGroup.rotation.y = -theta;

    const rLug = WALNUT_COLLAR_RADIUS + 0.024;
    const claw = new THREE.Mesh(clawHookGeom, brassMat);
    claw.position.set(rLug - 0.018, 0.014, 0);
    claw.rotation.z = 0.35;
    lugGroup.add(claw);

    const rod = new THREE.Mesh(tensionRodGeom, brassMat);
    rod.position.set(rLug, -0.085, 0);
    lugGroup.add(rod);

    const boss = new THREE.Mesh(lugBossGeom, brassMat);
    boss.position.set(rLug - 0.01, -0.085, 0);
    lugGroup.add(boss);

    const tee = new THREE.Mesh(teeBarGeom, brassMat);
    tee.position.set(rLug, -0.195, 0);
    lugGroup.add(tee);

    apparatusGroup.add(lugGroup);
  }

  // --- D. Singularity-Free Stretched Vellum/Latex Membrane Mesh (`104 × 104`) ---
  const membraneGeom = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(_memPositions, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  const normAttr = new THREE.BufferAttribute(_memNormals, 3);
  normAttr.setUsage(THREE.DynamicDrawUsage);
  const metaAttr = new THREE.BufferAttribute(_memMeta, 4);
  metaAttr.setUsage(THREE.DynamicDrawUsage);

  membraneGeom.setAttribute('position', posAttr);
  membraneGeom.setAttribute('normal', normAttr);
  membraneGeom.setAttribute('aMembraneMeta', metaAttr);

  const memIndices: number[] = [];
  for (let iv = 0; iv < GRID_STEPS; iv++) {
    for (let iu = 0; iu < GRID_STEPS; iu++) {
      const i00 = iv * GRID_STRIDE + iu;
      const i10 = iv * GRID_STRIDE + (iu + 1);
      const i11 = (iv + 1) * GRID_STRIDE + (iu + 1);
      const i01 = (iv + 1) * GRID_STRIDE + iu;
      // Wind counter-clockwise looking down (+Y normal)
      memIndices.push(i00, i01, i11, i00, i11, i10);
    }
  }
  membraneGeom.setIndex(memIndices);

  const membraneMesh = new THREE.Mesh(membraneGeom, membraneMat);
  membraneMesh.receiveShadow = true;
  membraneMesh.frustumCulled = false;
  apparatusGroup.add(membraneMesh);

  // --- E. Pre-Allocated Pool of 18 Polished Mineral Spheres + 1 Preview Charge Sphere ---
  const sharedSphereGeom = new THREE.SphereGeometry(1.0, 40, 28);
  const marbles: MarbleRecord[] = [];

  for (let i = 0; i < MAX_MARBLES; i++) {
    const mType = i % MINERAL_TYPES;
    const mesh = new THREE.Mesh(sharedSphereGeom, mineralMaterials[mType]);
    mesh.visible = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    apparatusGroup.add(mesh);

    marbles.push({
      active: false,
      mineralType: mType,
      radius: 0.11,
      mass: 1.0,
      x: 0,
      y: 1.0,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      smoothedLoad: 0,
      isGrabbed: false,
      spawnAgeSec: 0,
      mesh,
    });
  }

  // Suspended preview sphere shown while holding pointer down before release
  const previewMesh = new THREE.Mesh(sharedSphereGeom, mineralMaterials[0]);
  previewMesh.visible = false;
  previewMesh.castShadow = true;
  apparatusGroup.add(previewMesh);

  let nextMineralCursor = 0;

  function spawnMarble(
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    radius: number,
    mineralType?: number,
    settledOnSheet = false
  ): MarbleRecord {
    // Find an inactive slot, or recycle the oldest non-grabbed marble
    let slot: MarbleRecord | null = null;
    for (let i = 0; i < MAX_MARBLES; i++) {
      if (!marbles[i].active) {
        slot = marbles[i];
        break;
      }
    }
    if (!slot) {
      let oldestAge = -1;
      for (let i = 0; i < MAX_MARBLES; i++) {
        if (!marbles[i].isGrabbed && marbles[i].spawnAgeSec > oldestAge) {
          oldestAge = marbles[i].spawnAgeSec;
          slot = marbles[i];
        }
      }
      slot = slot ?? marbles[0];
    }

    const mType =
      mineralType !== undefined ? mineralType % MINERAL_TYPES : nextMineralCursor++ % MINERAL_TYPES;

    // Clamp spawn inside safe hoop radius
    const maxR = MEMBRANE_RADIUS - radius - 0.05;
    const rad = Math.hypot(x, z);
    const clampScale = rad > maxR && rad > 1e-5 ? maxR / rad : 1.0;

    slot.active = true;
    slot.mineralType = mType;
    slot.radius = radius;
    // Mineral density scale (Hematite & Lapis slightly denser than Amber)
    const densityMult = mType === 4 ? 1.35 : mType === 0 ? 1.18 : mType === 2 ? 0.85 : 1.05;
    slot.mass = Math.pow(radius / 0.11, 3.0) * densityMult;

    slot.x = x * clampScale;
    slot.z = z * clampScale;
    slot.y = y;
    slot.vx = vx;
    slot.vy = vy;
    slot.vz = vz;
    slot.smoothedLoad = settledOnSheet ? slot.mass * 0.105 : 0.0;
    slot.isGrabbed = false;
    slot.spawnAgeSec = 0.0;

    slot.mesh.material = mineralMaterials[mType];
    slot.mesh.scale.setScalar(radius);
    slot.mesh.position.set(slot.x, slot.y, slot.z);
    // Random initial mineral vein orientation
    slot.mesh.quaternion.setFromEuler(
      new THREE.Euler(mType * 1.1 + x * 2.3, mType * 2.1 + z * 1.7, x - z)
    );
    slot.mesh.visible = true;

    return slot;
  }

  // -------------------------------------------------------------------------
  // Initial Boot Choreography:
  // 1 heavy Lapis-Pyrite stone already resting in its tension funnel +
  // 2 stones (Cinnabar Agate & Realgar Amber) dropping in on gentle curved
  // trajectories so the user immediately sees bounce, flex, roll, and coalescence!
  // -------------------------------------------------------------------------
  {
    const anchor = spawnMarble(-0.06, -0.05, 0.04, 0, 0, 0, 0.145, 0, true);
    // Place anchor right on its own initial funnel depth
    evaluateTensionFunnelAt(anchor.x, anchor.z, marbles, 1.0);
    anchor.y = _funnelEvalOut[0] + anchor.radius;
    anchor.mesh.position.set(anchor.x, anchor.y, anchor.z);

    // Second stone mid-drop from upper-right with gentle orbital tangent
    spawnMarble(0.68, 0.95, -0.32, -0.42, -0.35, 0.58, 0.118, 1, false);
    // Third stone slightly higher on opposite flank
    spawnMarble(-0.74, 1.58, -0.24, 0.36, 0.0, -0.52, 0.105, 2, false);
    nextMineralCursor = 3;
  }

  // -------------------------------------------------------------------------
  // Coupled Substepped Physics & Membrane Geometry Updater (Zero GC)
  // -------------------------------------------------------------------------
  function stepCoupledMembraneAndMarbles(dtFrameSec: number): void {
    const dtClamped = Math.min(0.033, Math.max(0.002, dtFrameSec));
    const dtSub = dtClamped / PHYSICS_SUBSTEPS;

    for (let step = 0; step < PHYSICS_SUBSTEPS; step++) {
      // Nonlinear elastomer stiffening as total load accumulates so deep clusters
      // sag dramatically (`~ -0.45`) without tearing to infinity
      let totalLoadSum = 0.0;
      for (let i = 0; i < MAX_MARBLES; i++) {
        if (marbles[i].active) totalLoadSum += marbles[i].smoothedLoad;
      }
      const globalStiffScale = 1.0 / (1.0 + totalLoadSum * 0.62);

      // 1. Advance each active marble under gravity, membrane normal contact, and funnel slope
      for (let i = 0; i < MAX_MARBLES; i++) {
        const m = marbles[i];
        if (!m.active) continue;
        m.spawnAgeSec += dtSub;

        if (m.isGrabbed) {
          // Grabbed marble tracks the pointer smoothly on the membrane surface,
          // pressing down with extra tactile hand weight so its tension funnel follows!
          const targetLoad = m.mass * 0.145;
          m.smoothedLoad += (targetLoad - m.smoothedLoad) * Math.min(1.0, dtSub * 22.0);
          evaluateTensionFunnelAt(m.x, m.z, marbles, globalStiffScale);
          const targetY = _funnelEvalOut[0] + sampleWaveDispAt(m.x, m.z) + m.radius * 0.96;
          m.y += (targetY - m.y) * Math.min(1.0, dtSub * 28.0);
          continue;
        }

        // Evaluate local tension funnel height & slope `(dh/dx, dh/dz)`
        evaluateTensionFunnelAt(m.x, m.z, marbles, globalStiffScale);
        const hFunnel = _funnelEvalOut[0];
        const dhDx = _funnelEvalOut[1];
        const dhDz = _funnelEvalOut[2];
        const hWave = sampleWaveDispAt(m.x, m.z);
        const hSurface = hFunnel + hWave;

        const bottomY = m.y - m.radius;
        const penetration = hSurface - bottomY;

        if (penetration > 0.0) {
          // Marble is pressing into the stretched membrane!
          // Elastic Hertzian/drumhead normal spring + impact damping (underdamped so drops bounce!)
          const kSpring = 245.0 * (0.65 + 0.35 * m.mass);
          const cDamp = m.vy < -0.35 ? 3.2 * m.mass : 9.5 * m.mass;
          const normalForce = Math.max(0.0, kSpring * penetration - cDamp * m.vy);

          // First-strike impact wave injection when falling onto the sheet
          if (m.vy < -0.55 && penetration < 0.045) {
            const waveKick = Math.max(-0.95, m.vy * 0.16 * Math.min(1.8, m.mass));
            injectDrumheadWaveImpulse(m.x, m.z, waveKick, m.radius * 1.45);
          }

          // Vertical acceleration: upward membrane normal reaction minus gravity
          const ay = normalForce / m.mass - GRAVITY;
          m.vy += ay * dtSub;

          // Prevent tunneling through extreme thin sheet on high drops
          if (penetration > m.radius * 1.35) {
            m.y = hSurface + m.radius - m.radius * 1.35;
            if (m.vy < 0) m.vy *= -0.52;
          }

          // Couple vertical contact pressure into the marble's quasi-static tension funnel load
          // Static equilibrium load = m.mass * 0.105; spikes higher during peak bounce compression!
          const instLoad = Math.min(
            m.mass * 0.24,
            (normalForce / (m.mass * GRAVITY)) * (m.mass * 0.105)
          );
          const loadRate = instLoad > m.smoothedLoad ? 26.0 : 14.0;
          m.smoothedLoad += (instLoad - m.smoothedLoad) * Math.min(1.0, dtSub * loadRate);

          // Tangential gravity-well acceleration down the membrane slope `(-g_eff * dh/dx, -g_eff * dh/dz)`
          const slopeAccelScale = (5.0 / 7.0) * GRAVITY * 2.05;
          m.vx += -slopeAccelScale * dhDx * dtSub;
          m.vz += -slopeAccelScale * dhDz * dtSub;

          // Gentle rolling viscoelastic membrane drag (low enough to allow 1–2 orbits before coalescing)
          const rollDrag = Math.exp(-dtSub * 0.58);
          m.vx *= rollDrag;
          m.vz *= rollDrag;

          // Subtle rolling wake into the drumhead wave field when skimming fast
          const horizSpeed = Math.hypot(m.vx, m.vz);
          if (horizSpeed > 0.45 && step === 0) {
            injectDrumheadWaveImpulse(
              m.x,
              m.z,
              -0.018 * horizSpeed * m.mass,
              m.radius * 1.25
            );
          }
        } else {
          // Marble is airborne above the membrane (initial drop or mid-bounce!)
          m.vy -= GRAVITY * dtSub;
          // Funnel relaxes rapidly while airborne so the membrane springs back up underneath it
          m.smoothedLoad *= Math.exp(-dtSub * 18.0);
          if (m.smoothedLoad < 1e-4) m.smoothedLoad = 0.0;
        }

        // Integrate position
        m.x += m.vx * dtSub;
        m.y += m.vy * dtSub;
        m.z += m.vz * dtSub;

        // Inner Brass Counter-Hoop Rim Boundary Collision (`r + radius <= MEMBRANE_RADIUS - 0.025`)
        const maxR = MEMBRANE_RADIUS - m.radius - 0.025;
        const rad = Math.hypot(m.x, m.z);
        if (rad > maxR && rad > 1e-6) {
          const nx = m.x / rad;
          const nz = m.z / rad;
          m.x = nx * maxR;
          m.z = nz * maxR;
          const vOut = m.vx * nx + m.vz * nz;
          if (vOut > 0) {
            m.vx -= 1.58 * vOut * nx;
            m.vz -= 1.58 * vOut * nz;
          }
        }

        // Update rolling mineral rotation quaternion (`omega = (up x v_horiz) / radius`)
        const hSpd = Math.hypot(m.vx, m.vz);
        if (hSpd > 1e-4) {
          _rotAxis.set(m.vz / hSpd, 0, -m.vx / hSpd);
          const dAngle = (hSpd * dtSub) / m.radius;
          _stepQuat.setFromAxisAngle(_rotAxis, dAngle);
          m.mesh.quaternion.premultiply(_stepQuat);
        }
      }

      // 2. 3D Sphere-Sphere Elastic Collisions & Coalescence Settling
      for (let i = 0; i < MAX_MARBLES; i++) {
        const a = marbles[i];
        if (!a.active) continue;
        for (let j = i + 1; j < MAX_MARBLES; j++) {
          const b = marbles[j];
          if (!b.active) continue;

          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dz = b.z - a.z;
          const minDist = a.radius + b.radius;
          const distSq = dx * dx + dy * dy + dz * dz;

          if (distSq < minDist * minDist && distSq > 1e-9) {
            const dist = Math.sqrt(distSq);
            const nx = dx / dist;
            const ny = dy / dist;
            const nz = dz / dist;
            const overlap = minDist - dist;

            const invMassA = a.isGrabbed ? 0.0 : 1.0 / a.mass;
            const invMassB = b.isGrabbed ? 0.0 : 1.0 / b.mass;
            const invMassSum = invMassA + invMassB;
            if (invMassSum <= 0) continue;

            // Positional separation
            const corr = overlap / invMassSum;
            a.x -= nx * corr * invMassA;
            a.y -= ny * corr * invMassA;
            a.z -= nz * corr * invMassA;
            b.x += nx * corr * invMassB;
            b.y += ny * corr * invMassB;
            b.z += nz * corr * invMassB;

            // Relative normal velocity & stone-on-stone clack restitution
            const rvx = b.vx - a.vx;
            const rvy = b.vy - a.vy;
            const rvz = b.vz - a.vz;
            const velAlongNormal = rvx * nx + rvy * ny + rvz * nz;

            if (velAlongNormal < 0) {
              // Crisp mineral clack on fast hits, damped settling at tiny speeds so clusters rest quietly
              const restitution = velAlongNormal < -0.25 ? 0.56 : 0.12;
              const jImp = (-(1.0 + restitution) * velAlongNormal) / invMassSum;
              a.vx -= jImp * invMassA * nx;
              a.vy -= jImp * invMassA * ny;
              a.vz -= jImp * invMassA * nz;
              b.vx += jImp * invMassB * nx;
              b.vy += jImp * invMassB * ny;
              b.vz += jImp * invMassB * nz;

              // Tangential stone-on-stone friction
              const tx = rvx - velAlongNormal * nx;
              const ty = rvy - velAlongNormal * ny;
              const tz = rvz - velAlongNormal * nz;
              const tLen = Math.hypot(tx, ty, tz);
              if (tLen > 1e-5) {
                const jt = Math.min(jImp * 0.18, (0.25 * tLen) / invMassSum);
                a.vx += (jt * invMassA * tx) / tLen;
                a.vz += (jt * invMassA * tz) / tLen;
                b.vx -= (jt * invMassB * tx) / tLen;
                b.vz -= (jt * invMassB * tz) / tLen;
              }

              // Strong stone clacks transmit a tiny acoustic ripple into the membrane
              if (velAlongNormal < -0.65 && step === 0) {
                injectDrumheadWaveImpulse(
                  0.5 * (a.x + b.x),
                  0.5 * (a.z + b.z),
                  0.035 * velAlongNormal,
                  0.14
                );
              }
            }
          }
        }
      }
    }

    // 3. Step the 2D Transverse Drumhead Wave PDE (`104 × 104` grid, 2 substeps per frame)
    const dtWave = dtClamped * 0.5;
    const waveC2 = 42.0;
    const waveSpring = 6.5;
    const waveDamp = 2.85;

    for (let wStep = 0; wStep < 2; wStep++) {
      for (let iv = 1; iv < GRID_STEPS; iv++) {
        const row = iv * GRID_STRIDE;
        for (let iu = 1; iu < GRID_STEPS; iu++) {
          const idx = row + iu;
          const env = _rimEnv[idx];
          if (env <= 0.0) {
            _waveNext[idx] = 0.0;
            _waveVel[idx] = 0.0;
            continue;
          }

          const uC = _waveDisp[idx];
          const lap =
            (_waveDisp[idx - 1] +
              _waveDisp[idx + 1] +
              _waveDisp[idx - GRID_STRIDE] +
              _waveDisp[idx + GRID_STRIDE] -
              4.0 * uC) *
            0.25;

          const acc = waveC2 * lap * 12.0 - waveSpring * uC - waveDamp * _waveVel[idx];
          const vNext = (_waveVel[idx] + acc * dtWave) * 0.994;
          _waveVel[idx] = vNext;
          _waveNext[idx] = Math.max(-0.22, Math.min(0.22, (uC + vNext * dtWave) * Math.min(1.0, env * 4.0)));
        }
      }
      _waveDisp.set(_waveNext);
    }

    // 4. Sync Marble Mesh Transforms & Pack Active Load Table for Fast Vertex Evaluation
    let totalLoadSum = 0.0;
    let activeCount = 0;
    for (let i = 0; i < MAX_MARBLES; i++) {
      const m = marbles[i];
      if (!m.active) continue;
      totalLoadSum += m.smoothedLoad;

      if (m.smoothedLoad > 1e-4) {
        _activeLoadX[activeCount] = m.x;
        _activeLoadZ[activeCount] = m.z;
        const r0 = m.radius * 1.28;
        _activeLoadR0Sq[activeCount] = r0 * r0;
        _activeLoadStrength[activeCount] = m.smoothedLoad;
        _activeContactR2Inv[activeCount] = 1.0 / (m.radius * m.radius * 1.65);
        _activeInContact[activeCount] = 1;
        activeCount++;
      }
    }
    const globalStiffScale = 1.0 / (1.0 + totalLoadSum * 0.62);
    for (let a = 0; a < activeCount; a++) {
      _activeLoadStrength[a] *= globalStiffScale;
    }

    // 5. Recompute All `104 × 104` Membrane Vertex Positions & Optical Strain Metadata
    const logRRef = Math.log(MEMBRANE_RADIUS * 1.12);

    for (let idx = 0; idx < TOTAL_VERTS; idx++) {
      const env = _rimEnv[idx];
      const px = _worldX[idx];
      const pz = _worldZ[idx];

      let sumPhi = 0.0;
      let contactAO = 0.0;

      if (env > 0.0 && activeCount > 0) {
        for (let a = 0; a < activeCount; a++) {
          const dx = px - _activeLoadX[a];
          const dz = pz - _activeLoadZ[a];
          const d2 = dx * dx + dz * dz;
          const dist = Math.sqrt(d2 + _activeLoadR0Sq[a]);
          const logTerm = logRRef - Math.log(dist);
          if (logTerm > 0.0) {
            sumPhi -= _activeLoadStrength[a] * logTerm;
          }
          if (_activeInContact[a]) {
            const aoContrib = Math.exp(-d2 * _activeContactR2Inv[a]);
            if (aoContrib > contactAO) contactAO = aoContrib;
          }
        }
      }

      const hY = env * sumPhi + _waveDisp[idx];
      _totalHeight[idx] = hY;

      const idx3 = idx * 3;
      _memPositions[idx3 + 0] = px;
      _memPositions[idx3 + 1] = hY;
      _memPositions[idx3 + 2] = pz;

      const idx4 = idx * 4;
      _memMeta[idx4 + 0] = _discRho[idx];
      // `_memMeta[idx4 + 1]` (slope stretch strain) is filled in step 6 alongside normals!
      _memMeta[idx4 + 2] = contactAO;
      _memMeta[idx4 + 3] = Math.min(1.0, Math.max(0.0, -hY * 2.4));
    }

    // Ensure resting/grabbed marbles sit flush on top of the updated membrane surface without gap/clip
    for (let i = 0; i < MAX_MARBLES; i++) {
      const m = marbles[i];
      if (!m.active) continue;
      evaluateTensionFunnelAt(m.x, m.z, marbles, globalStiffScale);
      const hSurf = _funnelEvalOut[0] + sampleWaveDispAt(m.x, m.z);
      const minCenterY = hSurf + m.radius * 0.94;
      if (m.y < minCenterY && m.vy >= -0.18) {
        m.y = minCenterY;
      }
      m.mesh.position.set(m.x, m.y, m.z);
    }

    // 6. Recompute Smooth Surface Normals (`+Y` up) & Local Elastomer Slope Strain
    for (let iv = 0; iv <= GRID_STEPS; iv++) {
      const ivPrev = Math.max(0, iv - 1);
      const ivNext = Math.min(GRID_STEPS, iv + 1);

      for (let iu = 0; iu <= GRID_STEPS; iu++) {
        const iuPrev = Math.max(0, iu - 1);
        const iuNext = Math.min(GRID_STEPS, iu + 1);

        const idx = iv * GRID_STRIDE + iu;
        const idx3 = idx * 3;

        const idxU0 = (iv * GRID_STRIDE + iuPrev) * 3;
        const idxU1 = (iv * GRID_STRIDE + iuNext) * 3;
        const idxV0 = (ivPrev * GRID_STRIDE + iu) * 3;
        const idxV1 = (ivNext * GRID_STRIDE + iu) * 3;

        _dU.set(
          _memPositions[idxU1] - _memPositions[idxU0],
          _memPositions[idxU1 + 1] - _memPositions[idxU0 + 1],
          _memPositions[idxU1 + 2] - _memPositions[idxU0 + 2]
        );
        _dV.set(
          _memPositions[idxV1] - _memPositions[idxV0],
          _memPositions[idxV1 + 1] - _memPositions[idxV0 + 1],
          _memPositions[idxV1 + 2] - _memPositions[idxV0 + 2]
        );

        // Cross product `(dV x dU)` points upward (`+Y`) since U=+X, V=+Z
        _normVec.crossVectors(_dV, _dU).normalize();
        _memNormals[idx3 + 0] = _normVec.x;
        _memNormals[idx3 + 1] = _normVec.y;
        _memNormals[idx3 + 2] = _normVec.z;

        // Local elastomer stretch strain `sqrt(1 / ny^2) - 1 = sec(theta) - 1`
        const nyClamped = Math.max(0.32, Math.abs(_normVec.y));
        const stretchStrain = Math.min(1.4, (1.0 / nyClamped - 1.0) * 1.45);
        _memMeta[idx * 4 + 1] = stretchStrain;
      }
    }

    posAttr.needsUpdate = true;
    normAttr.needsUpdate = true;
    metaAttr.needsUpdate = true;
  }

  // Run initial geometry build before first render
  stepCoupledMembraneAndMarbles(0.016);

  // -------------------------------------------------------------------------
  // Tactile Interaction Controller:
  //   - Click / Hold-and-Release inside hoop -> Drop polished mineral marble
  //     (holding grows stone size slightly; flicking launches orbital velocity!)
  //   - Click & Drag an existing marble -> Pull its gravity funnel across the sheet
  //   - Double-click (or Spacebar) -> Trampoline upward snap that pops & scatters stones
  //   - Drag outside the hoop -> Smooth camera inspection tilt/yaw
  // -------------------------------------------------------------------------
  type InteractionMode = 'none' | 'drop-charge' | 'grab-marble' | 'orbit-rig';
  let mode: InteractionMode = 'none';
  let grabbedMarble: MarbleRecord | null = null;

  let chargeHoldSec = 0.0;
  let chargeMineralType = nextMineralCursor % MINERAL_TYPES;
  let chargeLocalX = 0.0;
  let chargeLocalZ = 0.0;
  let prevChargeLocalX = 0.0;
  let prevChargeLocalZ = 0.0;
  let flickVelX = 0.0;
  let flickVelZ = 0.0;

  let targetYaw = 0.18;
  let targetPitch = 0.0;
  let curYaw = 0.18;
  let curPitch = 0.0;
  let orbitStartX = 0.0;
  let orbitStartY = 0.0;
  let dragStartYaw = 0.18;
  let dragStartPitch = 0.0;

  let snapFlash = 0.0;

  function projectNdcToMembranePlane(ndc: THREE.Vector2, outLocal: THREE.Vector3): boolean {
    _raycaster.setFromCamera(ndc, stage.camera);
    apparatusGroup.updateMatrixWorld();
    _invApparatusMat.copy(apparatusGroup.matrixWorld).invert();

    // Intersect ray in apparatus local space against y = 0 rim plane
    _rimPlane.set(new THREE.Vector3(0, 1, 0), 0);
    _rimPlane.applyMatrix4(apparatusGroup.matrixWorld);
    const hit = _raycaster.ray.intersectPlane(_rimPlane, _hitPoint);
    if (!hit) return false;
    outLocal.copy(_hitPoint).applyMatrix4(_invApparatusMat);
    return true;
  }

  function pickMarbleUnderCursor(ndc: THREE.Vector2): MarbleRecord | null {
    _raycaster.setFromCamera(ndc, stage.camera);
    apparatusGroup.updateMatrixWorld();

    let bestMarble: MarbleRecord | null = null;
    let bestDist = Infinity;

    for (let i = 0; i < MAX_MARBLES; i++) {
      const m = marbles[i];
      if (!m.active) continue;
      _marbleWorldCenter.set(m.x, m.y, m.z).applyMatrix4(apparatusGroup.matrixWorld);
      const pickRadius = m.radius * 1.35;
      const distToRaySq = _raycaster.ray.distanceSqToPoint(_marbleWorldCenter);
      if (distToRaySq <= pickRadius * pickRadius) {
        const camDist = _raycaster.ray.origin.distanceTo(_marbleWorldCenter);
        if (camDist < bestDist) {
          bestDist = camDist;
          bestMarble = m;
        }
      }
    }
    return bestMarble;
  }

  function triggerTrampolineSnap(): void {
    snapFlash = 1.0;
    injectDrumheadWaveImpulse(0, 0, 1.35, 0.72);
    for (let i = 0; i < MAX_MARBLES; i++) {
      const m = marbles[i];
      if (!m.active) continue;
      // Pop resting stones upward into the air with slight radial scatter!
      const rad = Math.hypot(m.x, m.z) + 1e-4;
      m.vy = Math.max(m.vy, 2.35 + (0.16 / Math.max(0.08, m.radius)) * 0.45);
      m.vx += (m.x / rad) * 0.55 + Math.sin(i * 3.7) * 0.28;
      m.vz += (m.z / rad) * 0.55 + Math.cos(i * 2.9) * 0.28;
      m.smoothedLoad *= 0.2;
    }
  }

  const canvasEl = stage.shell.canvas;

  const onDblClick = () => {
    triggerTrampolineSnap();
  };
  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      triggerTrampolineSnap();
    }
  };

  canvasEl.addEventListener('dblclick', onDblClick);
  window.addEventListener('keydown', onKeyDown);
  stage.shell.registerDispose(() => {
    canvasEl.removeEventListener('dblclick', onDblClick);
    window.removeEventListener('keydown', onKeyDown);
  });

  let wasDown = false;

  stage.onTick(({ dtSec, pointer }) => {
    const dt = Math.min(0.05, Math.max(0.001, dtSec));

    // Handle pointerdown transition
    if (pointer.isDown && !wasDown) {
      const hitMarble = pickMarbleUnderCursor(pointer.ndc);
      if (hitMarble) {
        mode = 'grab-marble';
        grabbedMarble = hitMarble;
        grabbedMarble.isGrabbed = true;
        grabbedMarble.vx = 0;
        grabbedMarble.vy = 0;
        grabbedMarble.vz = 0;
      } else if (projectNdcToMembranePlane(pointer.ndc, _localHit)) {
        const hitRad = Math.hypot(_localHit.x, _localHit.z);
        if (hitRad < MEMBRANE_RADIUS * 0.96) {
          mode = 'drop-charge';
          chargeHoldSec = 0.0;
          chargeMineralType = nextMineralCursor % MINERAL_TYPES;
          const maxR = MEMBRANE_RADIUS - 0.18;
          const clampS = hitRad > maxR && hitRad > 1e-5 ? maxR / hitRad : 1.0;
          chargeLocalX = _localHit.x * clampS;
          chargeLocalZ = _localHit.z * clampS;
          prevChargeLocalX = chargeLocalX;
          prevChargeLocalZ = chargeLocalZ;
          flickVelX = 0.0;
          flickVelZ = 0.0;

          previewMesh.material = mineralMaterials[chargeMineralType];
          previewMesh.visible = true;
        } else {
          mode = 'orbit-rig';
          orbitStartX = pointer.ndc.x;
          orbitStartY = pointer.ndc.y;
          dragStartYaw = targetYaw;
          dragStartPitch = targetPitch;
        }
      } else {
        mode = 'orbit-rig';
        orbitStartX = pointer.ndc.x;
        orbitStartY = pointer.ndc.y;
        dragStartYaw = targetYaw;
        dragStartPitch = targetPitch;
      }
    }

    // Handle active drag/hold
    if (pointer.isDown) {
      if (mode === 'grab-marble' && grabbedMarble) {
        if (projectNdcToMembranePlane(pointer.ndc, _localHit)) {
          const maxR = MEMBRANE_RADIUS - grabbedMarble.radius - 0.04;
          const r = Math.hypot(_localHit.x, _localHit.z);
          const s = r > maxR && r > 1e-5 ? maxR / r : 1.0;
          const tx = _localHit.x * s;
          const tz = _localHit.z * s;
          grabbedMarble.vx = (tx - grabbedMarble.x) / dt;
          grabbedMarble.vz = (tz - grabbedMarble.z) / dt;
          grabbedMarble.x = tx;
          grabbedMarble.z = tz;
        }
      } else if (mode === 'drop-charge') {
        chargeHoldSec = Math.min(1.2, chargeHoldSec + dt);
        if (projectNdcToMembranePlane(pointer.ndc, _localHit)) {
          const previewRadius = 0.095 + Math.min(0.075, chargeHoldSec * 0.075);
          const maxR = MEMBRANE_RADIUS - previewRadius - 0.05;
          const r = Math.hypot(_localHit.x, _localHit.z);
          const s = r > maxR && r > 1e-5 ? maxR / r : 1.0;
          chargeLocalX = _localHit.x * s;
          chargeLocalZ = _localHit.z * s;

          const instVx = (chargeLocalX - prevChargeLocalX) / dt;
          const instVz = (chargeLocalZ - prevChargeLocalZ) / dt;
          const ema = 1.0 - Math.exp(-dt * 16.0);
          flickVelX += (instVx - flickVelX) * ema;
          flickVelZ += (instVz - flickVelZ) * ema;

          prevChargeLocalX = chargeLocalX;
          prevChargeLocalZ = chargeLocalZ;

          const hoverHeight = 0.92 - Math.min(0.18, chargeHoldSec * 0.14);
          previewMesh.scale.setScalar(previewRadius);
          previewMesh.position.set(chargeLocalX, hoverHeight, chargeLocalZ);
          previewMesh.rotation.y += dt * 1.4;
        }
      } else if (mode === 'orbit-rig') {
        const dx = pointer.ndc.x - orbitStartX;
        const dy = pointer.ndc.y - orbitStartY;
        targetYaw = dragStartYaw + dx * 1.35;
        targetPitch = Math.max(-0.28, Math.min(0.28, dragStartPitch - dy * 0.65));
      }
    }

    // Handle pointerup release!
    if (!pointer.isDown && wasDown) {
      if (mode === 'grab-marble' && grabbedMarble) {
        grabbedMarble.isGrabbed = false;
        // Clamp release fling speed so it orbits smoothly inside the hoop
        const spd = Math.hypot(grabbedMarble.vx, grabbedMarble.vz);
        if (spd > 3.2) {
          grabbedMarble.vx *= 3.2 / spd;
          grabbedMarble.vz *= 3.2 / spd;
        }
        grabbedMarble = null;
      } else if (mode === 'drop-charge') {
        previewMesh.visible = false;
        // Quick tap gets varied natural size; holding charges a heavier boulder!
        const tapVariation = 0.098 + ((nextMineralCursor * 37) % 10) * 0.0042;
        const dropRadius =
          chargeHoldSec < 0.16
            ? tapVariation
            : Math.min(0.172, 0.102 + chargeHoldSec * 0.072);
        const dropHeight = 0.92 - Math.min(0.18, chargeHoldSec * 0.14);

        const maxFlick = 2.6;
        const fSpd = Math.hypot(flickVelX, flickVelZ);
        const fScale = fSpd > maxFlick ? maxFlick / fSpd : 1.0;

        spawnMarble(
          chargeLocalX,
          dropHeight,
          chargeLocalZ,
          flickVelX * fScale * 0.55,
          -0.25,
          flickVelZ * fScale * 0.55,
          dropRadius,
          chargeMineralType,
          false
        );
        nextMineralCursor = chargeMineralType + 1;
      }
      mode = 'none';
    }

    // Subtle pointer hover breeze ripple when skimming across the empty membrane without clicking
    if (!pointer.isDown && pointer.isActive && pointer.speed > 0.65) {
      if (projectNdcToMembranePlane(pointer.ndc, _localHit)) {
        const r = Math.hypot(_localHit.x, _localHit.z);
        if (r < MEMBRANE_RADIUS * 0.92) {
          injectDrumheadWaveImpulse(
            _localHit.x,
            _localHit.z,
            -0.022 * Math.min(2.5, pointer.speed),
            0.16
          );
        }
      }
    }

    wasDown = pointer.isDown;

    // Gentle idle turntable drift so rim speculars & subsurface glow stay alive
    if (mode !== 'orbit-rig') {
      targetYaw += dt * 0.035;
    }
    const rotLerp = Math.min(1.0, dt * 7.5);
    curYaw += (targetYaw - curYaw) * rotLerp;
    curPitch += (targetPitch - curPitch) * rotLerp;
    apparatusGroup.rotation.set(curPitch, curYaw, 0);

    snapFlash = Math.max(0.0, snapFlash * Math.exp(-dt * 4.5));
    uSnapFlash.value = snapFlash;

    // Suppress unused scratch warning
    void _sphereHitTmp;

    stepCoupledMembraneAndMarbles(dt);
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay09Membrane();
}
