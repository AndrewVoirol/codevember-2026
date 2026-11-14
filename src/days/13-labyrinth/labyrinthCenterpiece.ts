import * as THREE from 'three/webgpu';
import {
  abs,
  clamp,
  dot,
  float,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  smoothstep,
  transformNormalToView,
  uniform,
  vec3,
  cameraPosition,
} from 'three/tsl';
import { CHALKY_SLATE_LINEAR, MINERAL_PALETTE } from '../../core/palette';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { type SpecimenStage } from '../../core/stage/SpecimenStage';

export interface LabyrinthCenterpiece {
  group: THREE.Group;
  clickTargets: THREE.Object3D[];
  getBeadOrbitTarget: (
    totalSolvedSec: number,
    outPos: THREE.Vector3,
    outVel: THREE.Vector3
  ) => void;
  update: (dtSec: number, totalSolvedSec: number, open01: number) => void;
  dispose: () => void;
}

const PIN_SPREAD_OFFSET = 0.56 / Math.sqrt(3);
const PIN_BASE_ANCHOR = 0.36;
const MAIN_BEAD_ORBIT_R = 0.228;
const MAIN_BEAD_ORBIT_OMEGA = 1.65;
const MAIN_BEAD_TILT_X = 0.48;
const MAIN_BEAD_TILT_Z = -0.34;

const PIN_SIGNS: ReadonlyArray<readonly [number, number]> = [
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

interface SatelliteOrbitSpec {
  radius: number;
  omega: number;
  phase: number;
  tiltX: number;
  tiltZ: number;
}

const SATELLITE_SPECS: readonly SatelliteOrbitSpec[] = [
  { radius: 0.155, omega: 2.35, phase: 0.0, tiltX: 0.55, tiltZ: 0.25 },
  { radius: 0.185, omega: -1.95, phase: 1.05, tiltX: -0.72, tiltZ: 0.62 },
  { radius: 0.215, omega: 1.60, phase: 2.18, tiltX: 0.35, tiltZ: -0.84 },
  { radius: 0.250, omega: -1.35, phase: 3.30, tiltX: -0.45, tiltZ: -0.52 },
  { radius: 0.280, omega: 1.18, phase: 4.42, tiltX: 0.88, tiltZ: 0.38 },
  { radius: 0.305, omega: -1.02, phase: 5.50, tiltX: -0.28, tiltZ: 0.92 },
];

const ORBIT_EULER_SCRATCH = new THREE.Euler();
const ORBIT_QUAT_SCRATCH = new THREE.Quaternion();
const VEC_TMP_SCRATCH = new THREE.Vector3();

/**
 * Creates a chamfered armillary astrolabe ring profile via LatheGeometry
 * with beveled outer edges so rim and studio lights catch flat brass chamfers.
 */
function buildAstrolabeRingGeometry(radius: number, radialThickness: number, halfHeight: number): THREE.LatheGeometry {
  const rIn = radius - radialThickness * 0.5;
  const rOut = radius + radialThickness * 0.5;
  const bev = Math.min(radialThickness, halfHeight * 2) * 0.24;
  const profile = [
    new THREE.Vector2(rIn + bev, -halfHeight),
    new THREE.Vector2(rOut - bev, -halfHeight),
    new THREE.Vector2(rOut, -halfHeight + bev),
    new THREE.Vector2(rOut, halfHeight - bev),
    new THREE.Vector2(rOut - bev, halfHeight),
    new THREE.Vector2(rIn + bev, halfHeight),
    new THREE.Vector2(rIn, halfHeight - bev),
    new THREE.Vector2(rIn, -halfHeight + bev),
    new THREE.Vector2(rIn + bev, -halfHeight),
  ];
  return new THREE.LatheGeometry(profile, 64);
}

/**
 * Builds a single stepped bronze guide pin (`y in [-0.34, +0.34]`, shaft `r = 0.011`,
 * integral collars `r = 0.0175` at `y = +/-0.24`) so each pin is a single draw call.
 */
function buildSteppedPinGeometry(): THREE.LatheGeometry {
  const profile = [
    new THREE.Vector2(0, -0.34),
    new THREE.Vector2(0.011, -0.34),
    new THREE.Vector2(0.011, -0.252),
    new THREE.Vector2(0.0175, -0.252),
    new THREE.Vector2(0.0175, -0.228),
    new THREE.Vector2(0.011, -0.228),
    new THREE.Vector2(0.011, 0.228),
    new THREE.Vector2(0.0175, 0.228),
    new THREE.Vector2(0.0175, 0.252),
    new THREE.Vector2(0.011, 0.252),
    new THREE.Vector2(0.011, 0.34),
    new THREE.Vector2(0, 0.34),
  ];
  return new THREE.LatheGeometry(profile, 12);
}

/**
 * Merges the 8 filigree cradle prongs into one indexed BufferGeometry draw call.
 */
function buildCradleGeometry(): THREE.BufferGeometry {
  const prong = new THREE.CylinderGeometry(0.0045, 0.006, 0.155, 8, 1);
  const srcPos = prong.getAttribute('position') as THREE.BufferAttribute;
  const srcNorm = prong.getAttribute('normal') as THREE.BufferAttribute;
  const srcIdx = prong.getIndex()!;
  const vPerProng = srcPos.count;
  const iPerProng = srcIdx.count;
  const prongCount = 8;

  const positions = new Float32Array(vPerProng * prongCount * 3);
  const normals = new Float32Array(vPerProng * prongCount * 3);
  const indices = new Uint16Array(iPerProng * prongCount);

  const mat = new THREE.Matrix4();
  const normMat = new THREE.Matrix3();
  const euler = new THREE.Euler();
  const quat = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scale = new THREE.Vector3(1, 1, 1);
  const vTmp = new THREE.Vector3();
  const nTmp = new THREE.Vector3();

  for (let pi = 0; pi < prongCount; pi++) {
    const theta = (pi * Math.PI * 2) / prongCount;
    const rCradle = 0.118;
    pos.set(Math.cos(theta) * rCradle, 0, Math.sin(theta) * rCradle);
    euler.set(0, -theta, 0.34 * (pi % 2 === 0 ? 1 : -1), 'YXZ');
    quat.setFromEuler(euler);
    mat.compose(pos, quat, scale);
    normMat.getNormalMatrix(mat);

    const vBase = pi * vPerProng;
    for (let vi = 0; vi < vPerProng; vi++) {
      vTmp.fromBufferAttribute(srcPos, vi).applyMatrix4(mat);
      nTmp.fromBufferAttribute(srcNorm, vi).applyMatrix3(normMat).normalize();
      const dst = (vBase + vi) * 3;
      positions[dst] = vTmp.x;
      positions[dst + 1] = vTmp.y;
      positions[dst + 2] = vTmp.z;
      normals[dst] = nTmp.x;
      normals[dst + 1] = nTmp.y;
      normals[dst + 2] = nTmp.z;
    }

    const iBase = pi * iPerProng;
    for (let ii = 0; ii < iPerProng; ii++) {
      indices[iBase + ii] = vBase + srcIdx.getX(ii);
    }
  }

  prong.dispose();

  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  merged.setIndex(new THREE.BufferAttribute(indices, 1));
  return merged;
}

/**
 * Builds the post-solve Lithic Astrolabe & Lapis-Realgar Reliquary centerpiece
 * housed inside the 8 parting stone octants. Strictly uses natural palette pigments
 * and performs zero per-frame allocations inside `update` or `getBeadOrbitTarget`.
 */
export function createLabyrinthCenterpiece(
  stage: SpecimenStage,
  _beadMaterial: THREE.MeshPhysicalNodeMaterial
): LabyrinthCenterpiece {
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const rawUmber = createPigmentUniforms('rawUmber');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const verdigrisMalachite = createPigmentUniforms('verdigrisMalachite');
  const azuritePrussian = createPigmentUniforms('azuritePrussian');
  const lapisUltramarine = createPigmentUniforms('lapisUltramarine');
  const sporeSulfur = createPigmentUniforms('sporeSulfur');

  const uOpenLevel = uniform(0.0);
  const uCorePulse = uniform(0.8);

  // -------------------------------------------------------------------------
  // 1. Gilded & Patinated Astrolabe Bronze Material
  // -------------------------------------------------------------------------
  const bronzeMaterial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.90,
    roughness: 0.20,
  });
  {
    const p = positionLocal;
    const brushGrain = valueNoise3DTsl(p.mul(vec3(92.0, 24.0, 92.0)));
    const seamOxide = smoothstep(float(0.68), float(0.92), valueNoise3DTsl(p.mul(38.0).add(vec3(3.1, 7.4, 2.2))));
    const brassBase = mix(
      mix(rawUmber.rInf, yellowOchre.rInf, float(0.74)),
      warmKaolin.rInf,
      brushGrain.mul(0.26)
    );
    const bronzeWithOxide = mix(brassBase, verdigrisMalachite.rInf.mul(0.72), seamOxide.mul(0.22));
    bronzeMaterial.colorNode = clamp(bronzeWithOxide, 0.0, 1.0);
    bronzeMaterial.roughnessNode = clamp(
      float(0.17).add(brushGrain.mul(0.10)).add(seamOxide.mul(0.18)),
      0.14,
      0.46
    );

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const rim = pow(abs(float(1.0).sub(clamp(dot(nW, viewDirW), 0.0, 1.0))), float(2.8));
    const coreGlow = mix(realgarAmber.rInf, yellowOchre.rInf, float(0.35)).mul(0.22).mul(uOpenLevel);
    bronzeMaterial.emissiveNode = coreGlow.add(
      mix(yellowOchre.rInf, warmKaolin.rInf, float(0.45)).mul(rim.mul(0.16)).mul(uOpenLevel)
    );
  }

  // -------------------------------------------------------------------------
  // 2. Patinated Cleavage Guide Pins Material
  // -------------------------------------------------------------------------
  const pinMaterial = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.24,
  });
  {
    const p = positionLocal;
    const fineBrush = valueNoise3DTsl(p.mul(68.0));
    const pinPatina = smoothstep(float(0.64), float(0.90), valueNoise3DTsl(p.mul(22.0).add(vec3(1.4, 5.2, 6.8))));
    const baseBronze = mix(rawUmber.rInf.mul(0.78), yellowOchre.rInf, float(0.68));
    pinMaterial.colorNode = clamp(mix(baseBronze, verdigrisMalachite.rInf.mul(0.65), pinPatina.mul(0.35)), 0.0, 1.0);
    pinMaterial.roughnessNode = clamp(float(0.20).add(fineBrush.mul(0.09)).add(pinPatina.mul(0.16)), 0.16, 0.48);
  }

  // -------------------------------------------------------------------------
  // 3. Cardinal Cabochon Boss Materials (Cinnabar Vermilion & Verdigris)
  // -------------------------------------------------------------------------
  const cinnabarCabochonMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.22,
    metalness: 0.04,
    clearcoat: 0.85,
    clearcoatRoughness: 0.08,
  });
  cinnabarCabochonMat.colorNode = kubelkaMunkGlaze(
    cinnabarVermilion.K,
    cinnabarVermilion.S,
    float(1.35),
    rawUmber.rInf.mul(0.35)
  );
  cinnabarCabochonMat.emissiveNode = mix(cinnabarVermilion.rInf, realgarAmber.rInf, float(0.4))
    .mul(0.18)
    .mul(uOpenLevel);

  const verdigrisCabochonMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.24,
    metalness: 0.04,
    clearcoat: 0.80,
    clearcoatRoughness: 0.10,
  });
  verdigrisCabochonMat.colorNode = kubelkaMunkGlaze(
    verdigrisMalachite.K,
    verdigrisMalachite.S,
    float(1.25),
    rawUmber.rInf.mul(0.30)
  );

  // -------------------------------------------------------------------------
  // 4. Faceted Lapis Ultramarine & Realgar Amber Reliquary Crystal Material
  // -------------------------------------------------------------------------
  const crystalMaterial = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.08,
    metalness: 0.05,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
  });
  {
    const p = positionLocal;
    const nL = normalize(normalLocal);
    const facetShift = valueNoise3DTsl(p.mul(18.0).add(nL.mul(2.1)));
    const lapisWeight = float(0.78).sub(facetShift.mul(0.24));
    const lapisK = mix(azuritePrussian.K, lapisUltramarine.K, lapisWeight);
    const lapisS = mix(azuritePrussian.S, lapisUltramarine.S, lapisWeight);
    const crystalAlbedo = kubelkaMunkGlaze(
      lapisK,
      lapisS,
      float(1.45).add(facetShift.mul(0.45)),
      rawUmber.rInf.mul(0.25)
    );
    const pyriteMask = smoothstep(
      float(0.84),
      float(0.95),
      valueNoise3DTsl(p.mul(54.0).add(vec3(2.3, 8.1, 4.7)))
    );
    crystalMaterial.colorNode = clamp(
      mix(crystalAlbedo, yellowOchre.rInf.mul(1.15), pyriteMask.mul(0.55)),
      0.0,
      1.0
    );

    const V = normalize(cameraPosition.sub(positionWorld));
    const N = normalize(normalWorld);
    const ndotv = clamp(dot(N, V), 0.0, 1.0);
    const fresnelRim = pow(abs(float(1.0).sub(ndotv)), float(2.2));
    const coreHeat = smoothstep(float(0.18), float(0.92), ndotv);
    const deepLapisGlow = lapisUltramarine.rInf.mul(0.28);
    const amberRim = mix(realgarAmber.rInf, sporeSulfur.rInf, float(0.38)).mul(fresnelRim.mul(1.05));
    const innerHeart = mix(realgarAmber.rInf, sporeSulfur.rInf, float(0.52)).mul(
      pow(coreHeat, float(4.0)).mul(0.32)
    );
    crystalMaterial.emissiveNode = deepLapisGlow
      .add(amberRim)
      .add(innerHeart)
      .mul(uCorePulse)
      .mul(uOpenLevel);
    crystalMaterial.normalNode = transformNormalToView(nL);
  }

  // -------------------------------------------------------------------------
  // 5. Orbital Quicksilver Micro-Satellite Material (Pristine Undeformed Mercury)
  // -------------------------------------------------------------------------
  const satelliteMaterial = new THREE.MeshPhysicalNodeMaterial({
    metalness: 0.97,
    roughness: 0.05,
    clearcoat: 0.35,
    clearcoatRoughness: 0.04,
  });
  {
    satelliteMaterial.colorNode = boneCalcite.rInf.mul(1.04);
    const V = normalize(cameraPosition.sub(positionWorld));
    const N = normalize(normalWorld);
    const R = normalize(V.negate().sub(N.mul(dot(V.negate(), N).mul(2.0))));
    const ndotv = clamp(dot(N, V), 0.0, 1.0);
    const fresnel = float(0.86).add(float(0.14).mul(pow(abs(float(1.0).sub(ndotv)), float(5.0))));

    const darkHorizon = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    ).mul(0.28);
    const warmCoreReflect = mix(realgarAmber.rInf, yellowOchre.rInf, float(0.35)).mul(0.62);
    const lowerDome = mix(warmCoreReflect, darkHorizon, smoothstep(float(-0.18), float(0.18), R.y));

    const keyDir = normalize(vec3(0.6, 4.4, 4.2));
    const keyHighlight = smoothstep(float(0.68), float(0.95), dot(R, keyDir)).mul(1.25);
    const zenith = smoothstep(float(0.66), float(0.92), R.y).mul(0.58);
    const rimDir = normalize(vec3(-1.5, 3.2, -3.6));
    const rimHighlight = smoothstep(float(0.66), float(0.92), dot(R, rimDir)).mul(0.75);
    const softbox = mix(boneCalcite.rInf, warmKaolin.rInf, float(0.35)).mul(1.08);
    const env = lowerDome.add(softbox.mul(keyHighlight.add(zenith).add(rimHighlight)));
    satelliteMaterial.emissiveNode = env.mul(fresnel).mul(0.90);
  }

  // -------------------------------------------------------------------------
  // Assemble Centerpiece Hierarchy
  // -------------------------------------------------------------------------
  const group = new THREE.Group();
  group.name = 'LithicAstrolabeCenterpiece';

  const pinGroup = new THREE.Group();
  pinGroup.name = 'CleavageGuidePins';
  group.add(pinGroup);

  const pinGeom = buildSteppedPinGeometry();

  interface PinEntry {
    mesh: THREE.Mesh;
    axis: 0 | 1 | 2;
    s1: number;
    s2: number;
  }
  const pins: PinEntry[] = [];
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis = (axis + 1) as 0 | 1 | 2) {
    for (let i = 0; i < PIN_SIGNS.length; i++) {
      const [s1, s2] = PIN_SIGNS[i];
      const pinMesh = new THREE.Mesh(pinGeom, pinMaterial);
      pinMesh.castShadow = false;
      pinMesh.receiveShadow = true;

      if (axis === 0) {
        pinMesh.rotation.z = Math.PI * 0.5;
      } else if (axis === 2) {
        pinMesh.rotation.x = Math.PI * 0.5;
      }
      pinGroup.add(pinMesh);
      pins.push({ mesh: pinMesh, axis, s1, s2 });
    }
  }

  // 3 Counter-rotating Gilded Astrolabe Gimbal Rings
  const gimbalsGroup = new THREE.Group();
  gimbalsGroup.name = 'AstrolabeGimbals';
  group.add(gimbalsGroup);

  const ringRadii = [0.345, 0.268, 0.192] as const;
  const ringGeometries: THREE.BufferGeometry[] = [];
  const gimbalRings: THREE.Group[] = [];
  const cabochonGeom = new THREE.SphereGeometry(0.0165, 16, 12);

  for (let ri = 0; ri < 3; ri++) {
    const R = ringRadii[ri];
    const rGeom = buildAstrolabeRingGeometry(R, 0.022 - ri * 0.002, 0.013 - ri * 0.0012);
    ringGeometries.push(rGeom);

    const ringGroup = new THREE.Group();
    const ringMesh = new THREE.Mesh(rGeom, bronzeMaterial);
    ringMesh.castShadow = true;
    ringMesh.receiveShadow = true;
    ringGroup.add(ringMesh);

    // 4 inlaid cardinal bosses per ring alternating Cinnabar and Verdigris
    for (let ci = 0; ci < 4; ci++) {
      const ang = (ci * Math.PI) / 2;
      const boss = new THREE.Mesh(
        cabochonGeom,
        ci % 2 === 0 ? cinnabarCabochonMat : verdigrisCabochonMat
      );
      boss.position.set(Math.cos(ang) * R, 0, Math.sin(ang) * R);
      boss.scale.set(1.0, 0.65, 1.0);
      boss.castShadow = false;
      ringGroup.add(boss);
    }

    gimbalsGroup.add(ringGroup);
    gimbalRings.push(ringGroup);
  }

  // Faceted Lapis Ultramarine & Realgar Amber Reliquary Core + Bronze Filigree Cradle
  const coreGroup = new THREE.Group();
  coreGroup.name = 'ReliquaryCrystalCore';
  gimbalsGroup.add(coreGroup);

  const rawCrystal = new THREE.IcosahedronGeometry(0.112, 1);
  const crystalGeom = rawCrystal.toNonIndexed();
  rawCrystal.dispose();
  crystalGeom.computeVertexNormals();

  const crystalMesh = new THREE.Mesh(crystalGeom, crystalMaterial);
  crystalMesh.name = 'LapisRealgarReliquaryCrystal';
  crystalMesh.castShadow = true;
  crystalMesh.receiveShadow = true;
  coreGroup.add(crystalMesh);

  // 8-prong bronze cradle merged into a single draw call around the crystal equator
  const cradleGeom = buildCradleGeometry();
  const cradleMesh = new THREE.Mesh(cradleGeom, bronzeMaterial);
  cradleMesh.name = 'CrystalFiligreeCradle';
  cradleMesh.castShadow = false;
  cradleMesh.receiveShadow = true;
  coreGroup.add(cradleMesh);

  // Warm Realgar Amber + Yellow Ochre internal PointLight (shadow-free for crisp 60-120fps)
  const rAmber = MINERAL_PALETTE.realgarAmber.rInfLinear;
  const yOchre = MINERAL_PALETTE.yellowOchre.rInfLinear;
  const lightColor = new THREE.Color().setRGB(
    rAmber[0] * 0.72 + yOchre[0] * 0.28,
    rAmber[1] * 0.72 + yOchre[1] * 0.28,
    rAmber[2] * 0.72 + yOchre[2] * 0.28,
    THREE.LinearSRGBColorSpace
  );
  const coreLight = new THREE.PointLight(lightColor, 0, 3.1, 1.55);
  coreLight.castShadow = false;
  coreGroup.add(coreLight);

  // 6 Orbital Quicksilver Micro-Satellites
  const satGroup = new THREE.Group();
  satGroup.name = 'QuicksilverMicroSatellites';
  gimbalsGroup.add(satGroup);

  const satGeom = new THREE.IcosahedronGeometry(0.0175, 2);
  const satMeshes: THREE.Mesh[] = [];
  for (let si = 0; si < SATELLITE_SPECS.length; si++) {
    const sat = new THREE.Mesh(satGeom, satelliteMaterial);
    sat.castShadow = false;
    sat.receiveShadow = false;
    satGroup.add(sat);
    satMeshes.push(sat);
  }

  const clickTargets: THREE.Object3D[] = [crystalMesh, ...gimbalRings];

  // Ensure initial hidden/closed transforms are clean
  const getBeadOrbitTarget: LabyrinthCenterpiece['getBeadOrbitTarget'] = (
    totalSolvedSec,
    outPos,
    outVel
  ) => {
    const theta = totalSolvedSec * MAIN_BEAD_ORBIT_OMEGA;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    ORBIT_EULER_SCRATCH.set(MAIN_BEAD_TILT_X, totalSolvedSec * 0.28, MAIN_BEAD_TILT_Z, 'XYZ');
    ORBIT_QUAT_SCRATCH.setFromEuler(ORBIT_EULER_SCRATCH);

    outPos.set(MAIN_BEAD_ORBIT_R * cosT, 0, MAIN_BEAD_ORBIT_R * sinT).applyQuaternion(ORBIT_QUAT_SCRATCH);
    outVel
      .set(
        -MAIN_BEAD_ORBIT_R * MAIN_BEAD_ORBIT_OMEGA * sinT,
        0,
        MAIN_BEAD_ORBIT_R * MAIN_BEAD_ORBIT_OMEGA * cosT
      )
      .applyQuaternion(ORBIT_QUAT_SCRATCH);
  };

  let crystalSpin = 0;

  const update: LabyrinthCenterpiece['update'] = (dtSec, totalSolvedSec, open01) => {
    const clampedOpen = Math.max(0, Math.min(1.25, open01));
    const appear = Math.max(0, Math.min(1, clampedOpen / 0.72));
    const coreAppear = Math.max(0, Math.min(1, (clampedOpen - 0.08) / 0.65));

    uOpenLevel.value = appear;
    uCorePulse.value = 0.74 + 0.26 * Math.sin(totalSolvedSec * 2.6);
    coreLight.intensity = 4.6 * Math.min(1, appear);

    // Extend cleavage bronze guide pins squarely across X, Y, Z parting walls
    const pinScaleY = 0.25 + 0.75 * Math.min(1.15, clampedOpen);
    const wallShift = PIN_BASE_ANCHOR + PIN_SPREAD_OFFSET * clampedOpen;
    for (let i = 0; i < pins.length; i++) {
      const p = pins[i];
      p.mesh.scale.set(1, pinScaleY, 1);
      if (p.axis === 0) {
        p.mesh.position.set(0, p.s1 * wallShift, p.s2 * wallShift);
      } else if (p.axis === 1) {
        p.mesh.position.set(p.s1 * wallShift, 0, p.s2 * wallShift);
      } else {
        p.mesh.position.set(p.s1 * wallShift, p.s2 * wallShift, 0);
      }
    }

    // Smoothly emerge gimbal rings & core from interior recess without poking closed walls
    const gimScale = 0.16 + 0.84 * coreAppear;
    gimbalsGroup.scale.setScalar(gimScale);

    const t = totalSolvedSec;
    gimbalRings[0].rotation.set(
      0.42 + 0.16 * Math.sin(t * 0.34),
      t * 0.38,
      0.18 * Math.cos(t * 0.27)
    );
    gimbalRings[1].rotation.set(
      -t * 0.62,
      0.24 * Math.sin(t * 0.45),
      0.58 + 0.20 * Math.cos(t * 0.41)
    );
    gimbalRings[2].rotation.set(
      0.34 + 0.15 * Math.sin(t * 0.58),
      -t * 0.52,
      t * 0.94
    );

    crystalSpin += dtSec * 0.78;
    crystalMesh.rotation.set(crystalSpin * 0.45, crystalSpin, crystalSpin * 0.28);
    cradleMesh.rotation.set(0, -crystalSpin * 0.55, 0.22 * Math.sin(t * 0.7));

    // Orbit 6 quicksilver micro-satellites on inclined planes expanding radially with open01
    for (let si = 0; si < satMeshes.length; si++) {
      const spec = SATELLITE_SPECS[si];
      const ang = spec.phase + t * spec.omega;
      const rCur = spec.radius * (0.25 + 0.75 * appear);
      ORBIT_EULER_SCRATCH.set(spec.tiltX, t * 0.18 * (si % 2 === 0 ? 1 : -1), spec.tiltZ, 'XYZ');
      ORBIT_QUAT_SCRATCH.setFromEuler(ORBIT_EULER_SCRATCH);
      VEC_TMP_SCRATCH.set(rCur * Math.cos(ang), 0, rCur * Math.sin(ang)).applyQuaternion(ORBIT_QUAT_SCRATCH);
      satMeshes[si].position.copy(VEC_TMP_SCRATCH);
    }
  };

  // Initialize deterministic state
  update(0, 0, 0);
  void stage;

  return {
    group,
    clickTargets,
    getBeadOrbitTarget,
    update,
    dispose: () => {
      pinGeom.dispose();
      cabochonGeom.dispose();
      crystalGeom.dispose();
      cradleGeom.dispose();
      satGeom.dispose();
      for (let i = 0; i < ringGeometries.length; i++) {
        ringGeometries[i].dispose();
      }
      bronzeMaterial.dispose();
      pinMaterial.dispose();
      cinnabarCabochonMat.dispose();
      verdigrisCabochonMat.dispose();
      crystalMaterial.dispose();
      satelliteMaterial.dispose();
    },
  };
}
