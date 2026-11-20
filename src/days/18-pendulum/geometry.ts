import * as THREE from 'three/webgpu';
import {
  BOB_TIP_OFFSET,
  BOB_TOP_COLLET_OFFSET,
  COUNTERWEIGHT_LENGTH,
  DOME_CENTER,
  DOME_MAX_PHI,
  DOME_RADIUS,
  PIVOT_POS,
  PULLEY_AXLE_Y,
  PULLEY_INNER_OFFSET_X,
  PULLEY_INNER_RADIUS,
  PULLEY_OUTER_OFFSET_X,
  PULLEY_OUTER_RADIUS,
  STONE_CROWN_HEIGHT,
  STONE_RADIUS,
  TRAIL_LENGTH,
  type PulleyKinematicsFrame,
} from './physics';
import { type PendulumMaterials } from './materials';
import { MINERAL_PALETTE } from '../../core/palette';

/**
 * Subtle organic macro hand-forging undulation for the dome silhouette (<= 1.4 mm).
 * Crisp planishing hammer facets live in the 768x768 zero-seam LAEA normal map.
 */
function subtleForgeMacroDisp(nx: number, ny: number, nz: number): number {
  const a = Math.sin(nx * 5.7 + ny * 3.1 - nz * 4.3) * Math.cos(nz * 6.1 + ny * 4.7);
  const b = Math.sin(nx * 9.3 - nz * 8.1 + ny * 5.4);
  return (a * 0.65 + b * 0.35) * 0.0014;
}

/**
 * Builds the suspended Hand-Planished Hammered-Iron Dome with a slim, flush
 * architectural Bone Calcite equatorial collar and finished forged underside.
 */
export function buildDomeAssembly(materials: PendulumMaterials): {
  group: THREE.Group;
  domeHitMesh: THREE.Mesh;
  rimMesh: THREE.Mesh;
  dispose: () => void;
} {
  const group = new THREE.Group();
  group.name = 'HammeredIronDomeGroup';
  group.position.copy(DOME_CENTER);

  const geometries: THREE.BufferGeometry[] = [];

  const domeGeo = new THREE.SphereGeometry(
    DOME_RADIUS,
    144,
    80,
    0,
    Math.PI * 2,
    0,
    DOME_MAX_PHI
  );
  const posAttr = domeGeo.attributes.position as THREE.BufferAttribute;
  const rimLocalY = DOME_RADIUS * Math.cos(DOME_MAX_PHI);
  for (let i = 0; i < posAttr.count; i++) {
    const vx = posAttr.getX(i);
    const vy = posAttr.getY(i);
    const vz = posAttr.getZ(i);
    const r = Math.hypot(vx, vy, vz);
    if (r > 1e-5) {
      const rimMask = Math.max(0, Math.min(1, (vy - rimLocalY) / 0.14));
      const smoothMask = rimMask * rimMask * (3 - 2 * rimMask);
      const disp = subtleForgeMacroDisp(vx / r, vy / r, vz / r) * smoothMask;
      const scale = (r + disp) / r;
      posAttr.setXYZ(i, vx * scale, vy * scale, vz * scale);
    }
  }
  domeGeo.computeVertexNormals();
  geometries.push(domeGeo);

  const domeMesh = new THREE.Mesh(domeGeo, materials.domeIron);
  domeMesh.name = 'HammeredIronDome';
  domeMesh.castShadow = true;
  domeMesh.receiveShadow = true;
  group.add(domeMesh);

  // Slim, refined beveled Bone Calcite equatorial base collar (flush with iron skirt)
  const rimRadius = DOME_RADIUS * Math.sin(DOME_MAX_PHI);
  const rimProfile: THREE.Vector2[] = [
    new THREE.Vector2(rimRadius - 0.068, rimLocalY - 0.036),
    new THREE.Vector2(rimRadius - 0.002, rimLocalY - 0.036),
    new THREE.Vector2(rimRadius + 0.009, rimLocalY - 0.027),
    new THREE.Vector2(rimRadius + 0.011, rimLocalY - 0.003),
    new THREE.Vector2(rimRadius + 0.005, rimLocalY + 0.008),
    new THREE.Vector2(rimRadius - 0.012, rimLocalY + 0.004),
    new THREE.Vector2(rimRadius - 0.068, rimLocalY - 0.014),
  ];
  const rimGeo = new THREE.LatheGeometry(rimProfile, 112);
  rimGeo.computeVertexNormals();
  geometries.push(rimGeo);

  const rimMesh = new THREE.Mesh(rimGeo, materials.boneCalciteRim);
  rimMesh.name = 'BoneCalciteRim';
  rimMesh.castShadow = true;
  rimMesh.receiveShadow = true;
  group.add(rimMesh);

  // Recessed forged-iron underside plate (inner-to-outer lathe order makes FrontSide normals face -Y outward!)
  const underProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, rimLocalY - 0.016),
    new THREE.Vector2(rimRadius * 0.45, rimLocalY - 0.020),
    new THREE.Vector2(rimRadius - 0.064, rimLocalY - 0.035),
  ];
  const underGeo = new THREE.LatheGeometry(underProfile, 72);
  underGeo.computeVertexNormals();
  geometries.push(underGeo);

  const underMesh = new THREE.Mesh(underGeo, materials.domeUnderside);
  underMesh.name = 'DomeUndersidePlate';
  underMesh.receiveShadow = true;
  group.add(underMesh);

  // Turned satin-brass central underside maker's boss (bottom-to-top outward winding)
  const bossUnderProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, rimLocalY - 0.036),
    new THREE.Vector2(0.034, rimLocalY - 0.034),
    new THREE.Vector2(0.068, rimLocalY - 0.027),
    new THREE.Vector2(0.086, rimLocalY - 0.021),
    new THREE.Vector2(0.086, rimLocalY - 0.015),
    new THREE.Vector2(0.001, rimLocalY - 0.015),
  ];
  const bossUnderGeo = new THREE.LatheGeometry(bossUnderProfile, 36);
  bossUnderGeo.computeVertexNormals();
  geometries.push(bossUnderGeo);

  const bossUnderMesh = new THREE.Mesh(bossUnderGeo, materials.satinBrass);
  bossUnderMesh.name = 'DomeUndersideMakerBoss';
  bossUnderMesh.receiveShadow = true;
  group.add(bossUnderMesh);

  return {
    group,
    domeHitMesh: domeMesh,
    rimMesh,
    dispose: () => {
      for (const g of geometries) g.dispose();
    },
  };
}

/**
 * Builds the overhead Horological Gimbal & Differential Pulley Assembly:
 * - Forged dark oxidized-bronze horseshoe arch crowned by a turned brass suspension bow/ring
 * - 360° swivel bronze clevis tracking the pendulum bob's horizontal meridian
 * - Spoked machined brass V-groove outer sheave + coaxial inner counterweight take-up drum
 * - Rising/falling turned satin-brass cylindrical counterweight plumb
 */
export function buildGimbalPulleyAssembly(materials: PendulumMaterials): {
  group: THREE.Group;
  archFrameGroup: THREE.Group;
  swivelGroup: THREE.Group;
  wheelGroup: THREE.Group;
  counterweightGroup: THREE.Group;
  dispose: () => void;
} {
  const group = new THREE.Group();
  group.name = 'OverheadGimbalPulleyGroup';
  group.position.copy(PIVOT_POS);

  const archFrameGroup = new THREE.Group();
  archFrameGroup.name = 'GimbalArchFrame';
  group.add(archFrameGroup);

  const geometries: THREE.BufferGeometry[] = [];

  // 1. Sleek open-clearance dark-bronze horseshoe arch (±78° span: guarantees >= 14 cm wire clearance
  //    at every 360° azimuth and drag angle while framing the central differential pulley cleanly)
  const arcRadius = 0.255;
  const halfSpan = 1.36; // ~78 deg each side -> tips sit cleanly above all taut wire trajectories
  const arcPoints: THREE.Vector3[] = [];
  const arcSteps = 48;

  for (let i = 0; i <= arcSteps; i++) {
    const angle = -halfSpan + (i / arcSteps) * (halfSpan * 2);
    arcPoints.push(
      new THREE.Vector3(Math.sin(angle) * arcRadius, Math.cos(angle) * arcRadius - arcRadius, 0)
    );
  }

  const arcCurve = new THREE.CatmullRomCurve3(arcPoints, false, 'centripetal', 0.25);
  const archGeo = new THREE.TubeGeometry(arcCurve, 64, 0.0145, 18, false);
  geometries.push(archGeo);
  const archMesh = new THREE.Mesh(archGeo, materials.agedBronze);
  archMesh.castShadow = true;
  archFrameGroup.add(archMesh);

  // Delicate turned brass tip collars & spherical finials aligned coaxially with arch leg tips
  // Ordered strictly bottom-to-top for verified outward FrontSide winding
  const collarProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, -0.009),
    new THREE.Vector2(0.012, -0.008),
    new THREE.Vector2(0.015, 0.0),
    new THREE.Vector2(0.020, 0.004),
    new THREE.Vector2(0.020, 0.014),
    new THREE.Vector2(0.014, 0.020),
    new THREE.Vector2(0.001, 0.020),
  ];
  const collarGeo = new THREE.LatheGeometry(collarProfile, 22);
  collarGeo.computeVertexNormals();
  const finialGeo = new THREE.SphereGeometry(0.020, 20, 16);
  geometries.push(collarGeo, finialGeo);

  const upVec = new THREE.Vector3(0, 1, 0);
  for (const side of [-1, 1]) {
    const endPt = side < 0 ? arcPoints[0] : arcPoints[arcPoints.length - 1];
    const prevPt = side < 0 ? arcPoints[1] : arcPoints[arcPoints.length - 2];
    const tipDir = new THREE.Vector3().subVectors(endPt, prevPt).normalize();

    const collar = new THREE.Mesh(collarGeo, materials.satinBrass);
    collar.position.copy(endPt);
    // Collar local -Y points outward along tipDir (so local +Y points back along -tipDir)
    collar.quaternion.setFromUnitVectors(upVec, tipDir.clone().negate());
    collar.castShadow = true;
    archFrameGroup.add(collar);

    const finial = new THREE.Mesh(finialGeo, materials.satinBrass);
    finial.position.copy(endPt).addScaledVector(tipDir, 0.021);
    finial.castShadow = true;
    archFrameGroup.add(finial);
  }

  // 2. Top Suspension Starting Point: bronze saddle boss + stepped brass crown + hanging bow ring
  const bossProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.020, -0.024),
    new THREE.Vector2(0.027, -0.016),
    new THREE.Vector2(0.028, 0.018),
    new THREE.Vector2(0.022, 0.026),
    new THREE.Vector2(0.014, 0.036),
    new THREE.Vector2(0.001, 0.038),
  ];
  const bossGeo = new THREE.LatheGeometry(bossProfile, 28);
  bossGeo.computeVertexNormals();
  geometries.push(bossGeo);
  const boss = new THREE.Mesh(bossGeo, materials.agedBronze);
  boss.castShadow = true;
  archFrameGroup.add(boss);

  const crownPostProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.017, 0.024),
    new THREE.Vector2(0.019, 0.030),
    new THREE.Vector2(0.012, 0.042),
    new THREE.Vector2(0.014, 0.050),
    new THREE.Vector2(0.008, 0.058),
    new THREE.Vector2(0.001, 0.060),
  ];
  const crownPostGeo = new THREE.LatheGeometry(crownPostProfile, 24);
  crownPostGeo.computeVertexNormals();
  geometries.push(crownPostGeo);
  const crownPost = new THREE.Mesh(crownPostGeo, materials.satinBrass);
  crownPost.castShadow = true;
  archFrameGroup.add(crownPost);

  // Turned horological hanging bow ring crowning the instrument (clear visual starting point)
  const bowRingGeo = new THREE.TorusGeometry(0.032, 0.0068, 14, 32);
  geometries.push(bowRingGeo);
  const bowRing = new THREE.Mesh(bowRingGeo, materials.satinBrass);
  bowRing.position.set(0, 0.084, 0);
  bowRing.castShadow = true;
  archFrameGroup.add(bowRing);

  // 3. Rotating 360° Swivel Pulley Carriage mounted directly below the arch crown
  const pulleyLocalY = PULLEY_AXLE_Y - PIVOT_POS.y; // -0.08
  const swivelGroup = new THREE.Group();
  swivelGroup.name = 'PulleySwivelCarriage';
  swivelGroup.position.set(0, pulleyLocalY, 0);
  group.add(swivelGroup);

  // Brass swivel spindle connecting clevis cleanly up into the dark-bronze arch boss
  const spindleGeo = new THREE.CylinderGeometry(0.010, 0.012, 0.046, 18);
  const swivelCollarGeo = new THREE.CylinderGeometry(0.019, 0.017, 0.013, 20);
  geometries.push(spindleGeo, swivelCollarGeo);

  const spindle = new THREE.Mesh(spindleGeo, materials.satinBrass);
  spindle.position.set(0, 0.058, 0);
  swivelGroup.add(spindle);

  const swivelCollar = new THREE.Mesh(swivelCollarGeo, materials.satinBrass);
  swivelCollar.position.set(0, 0.066, 0);
  swivelGroup.add(swivelCollar);

  // Machined dark-bronze sculpted clevis yoke bridging across the differential pulley axle
  const yokeShape = new THREE.Shape();
  yokeShape.moveTo(-0.048, 0.040);
  yokeShape.lineTo(0.048, 0.040);
  yokeShape.lineTo(0.048, 0.051);
  yokeShape.lineTo(0.022, 0.055);
  yokeShape.lineTo(-0.022, 0.055);
  yokeShape.lineTo(-0.048, 0.051);
  yokeShape.closePath();
  const bridgeGeo = new THREE.ExtrudeGeometry(yokeShape, {
    depth: 0.020,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.002,
    bevelSegments: 2,
  });
  bridgeGeo.translate(0, 0, -0.010);

  const cheekShape = new THREE.Shape();
  cheekShape.moveTo(-0.011, -0.007);
  cheekShape.lineTo(0.011, -0.007);
  cheekShape.lineTo(0.013, 0.044);
  cheekShape.lineTo(-0.013, 0.044);
  cheekShape.closePath();
  const cheekGeo = new THREE.ExtrudeGeometry(cheekShape, {
    depth: 0.006,
    bevelEnabled: true,
    bevelThickness: 0.0015,
    bevelSize: 0.0015,
    bevelSegments: 2,
  });
  cheekGeo.rotateY(Math.PI * 0.5);
  cheekGeo.translate(-0.003, 0, 0);

  const bearingBossGeo = new THREE.CylinderGeometry(0.0135, 0.0135, 0.011, 20);
  bearingBossGeo.rotateZ(Math.PI * 0.5);
  const axleCapGeo = new THREE.CylinderGeometry(0.0095, 0.0095, 0.007, 18);
  axleCapGeo.rotateZ(Math.PI * 0.5);
  geometries.push(bridgeGeo, cheekGeo, bearingBossGeo, axleCapGeo);

  const yokeBridge = new THREE.Mesh(bridgeGeo, materials.agedBronze);
  yokeBridge.castShadow = true;
  swivelGroup.add(yokeBridge);

  for (const side of [-1, 1]) {
    const cheek = new THREE.Mesh(cheekGeo, materials.agedBronze);
    cheek.position.set(side * 0.043, 0, 0);
    cheek.castShadow = true;
    swivelGroup.add(cheek);

    const bossRing = new THREE.Mesh(bearingBossGeo, materials.agedBronze);
    bossRing.position.set(side * 0.043, 0, 0);
    swivelGroup.add(bossRing);

    const cap = new THREE.Mesh(axleCapGeo, materials.satinBrass);
    cap.position.set(side * 0.050, 0, 0);
    swivelGroup.add(cap);
  }

  // 4. Coaxial Rotating Differential Pulley Wheel (rotates around local X by +wheelAngle)
  const wheelGroup = new THREE.Group();
  wheelGroup.name = 'DifferentialPulleyWheel';
  swivelGroup.add(wheelGroup);

  // Shared central dark-bronze axle shaft
  const axleGeo = new THREE.CylinderGeometry(0.009, 0.009, 0.092, 20);
  axleGeo.rotateZ(Math.PI * 0.5);
  geometries.push(axleGeo);
  const axleMesh = new THREE.Mesh(axleGeo, materials.agedBronze);
  wheelGroup.add(axleMesh);

  // 4a. Outer Machined V-Groove Satin-Brass Sheave (radius = PULLEY_OUTER_RADIUS at x = PULLEY_OUTER_OFFSET_X)
  const outerRimInner = PULLEY_OUTER_RADIUS - 0.015;
  const outerRimProfile: THREE.Vector2[] = [
    new THREE.Vector2(outerRimInner, -0.0095),
    new THREE.Vector2(PULLEY_OUTER_RADIUS + 0.0085, -0.0095),
    new THREE.Vector2(PULLEY_OUTER_RADIUS + 0.0085, -0.0055),
    new THREE.Vector2(PULLEY_OUTER_RADIUS - 0.002, -0.0012),
    new THREE.Vector2(PULLEY_OUTER_RADIUS - 0.002, 0.0012),
    new THREE.Vector2(PULLEY_OUTER_RADIUS + 0.0085, 0.0055),
    new THREE.Vector2(PULLEY_OUTER_RADIUS + 0.0085, 0.0095),
    new THREE.Vector2(outerRimInner, 0.0095),
    new THREE.Vector2(outerRimInner, -0.0095),
  ];
  const outerRimGeo = new THREE.LatheGeometry(outerRimProfile, 44);
  outerRimGeo.rotateZ(Math.PI * 0.5);
  outerRimGeo.computeVertexNormals();
  geometries.push(outerRimGeo);
  const outerRimMesh = new THREE.Mesh(outerRimGeo, materials.satinBrass);
  outerRimMesh.position.set(PULLEY_OUTER_OFFSET_X, 0, 0);
  outerRimMesh.castShadow = true;
  wheelGroup.add(outerRimMesh);

  // Central hub collar + 5 slim radial horological wheel spokes so wheel spin reads cleanly
  const hubCollarGeo = new THREE.CylinderGeometry(0.018, 0.018, 0.017, 22);
  hubCollarGeo.rotateZ(Math.PI * 0.5);
  const spokeLen = outerRimInner - 0.013;
  const spokeGeo = new THREE.BoxGeometry(0.0055, spokeLen, 0.0052);
  spokeGeo.translate(0, 0.014 + spokeLen * 0.5, 0);
  geometries.push(hubCollarGeo, spokeGeo);

  const hubCollar = new THREE.Mesh(hubCollarGeo, materials.agedBronze);
  hubCollar.position.set(PULLEY_OUTER_OFFSET_X, 0, 0);
  wheelGroup.add(hubCollar);

  for (let s = 0; s < 5; s++) {
    const spoke = new THREE.Mesh(spokeGeo, materials.satinBrass);
    spoke.position.set(PULLEY_OUTER_OFFSET_X, 0, 0);
    spoke.rotation.x = (s * Math.PI * 2) / 5;
    spoke.castShadow = true;
    wheelGroup.add(spoke);
  }

  // 4b. Coaxial Inner Counterweight Take-Up Drum (radius = PULLEY_INNER_RADIUS at x = PULLEY_INNER_OFFSET_X)
  const innerDrumProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.009, -0.012),
    new THREE.Vector2(PULLEY_INNER_RADIUS + 0.0075, -0.012),
    new THREE.Vector2(PULLEY_INNER_RADIUS + 0.0075, -0.0085),
    new THREE.Vector2(PULLEY_INNER_RADIUS - 0.001, -0.0065),
    new THREE.Vector2(PULLEY_INNER_RADIUS - 0.001, 0.0065),
    new THREE.Vector2(PULLEY_INNER_RADIUS + 0.0075, 0.0085),
    new THREE.Vector2(PULLEY_INNER_RADIUS + 0.0075, 0.012),
    new THREE.Vector2(0.009, 0.012),
  ];
  const innerDrumGeo = new THREE.LatheGeometry(innerDrumProfile, 30);
  innerDrumGeo.rotateZ(Math.PI * 0.5);
  innerDrumGeo.computeVertexNormals();
  geometries.push(innerDrumGeo);
  const innerDrumMesh = new THREE.Mesh(innerDrumGeo, materials.satinBrass);
  innerDrumMesh.position.set(PULLEY_INNER_OFFSET_X, 0, 0);
  innerDrumMesh.castShadow = true;
  wheelGroup.add(innerDrumMesh);

  // 5 delicate wound brass wire coils seated snugly inside the inner counterweight take-up drum
  const coilGeo = new THREE.TorusGeometry(PULLEY_INNER_RADIUS + 0.0014, 0.0022, 8, 28);
  coilGeo.rotateY(Math.PI * 0.5);
  geometries.push(coilGeo);
  for (let c = -2; c <= 2; c++) {
    const coil = new THREE.Mesh(coilGeo, materials.plumbWire);
    coil.position.set(PULLEY_INNER_OFFSET_X + c * 0.0026, 0, 0);
    wheelGroup.add(coil);
  }

  // 5. Rising & Falling Turned Satin-Brass Horological Counterweight Plumb
  // (Rooted in world space outside gimbal group; positioned each frame at cwEyeletWorld)
  const counterweightGroup = new THREE.Group();
  counterweightGroup.name = 'PulleyCounterweight';

  // Top suspension eyelet ring at local (0, 0, 0)
  const cwEyeGeo = new THREE.TorusGeometry(0.008, 0.0026, 10, 22);
  geometries.push(cwEyeGeo);
  const cwEye = new THREE.Mesh(cwEyeGeo, materials.satinBrass);
  cwEye.position.set(0, -0.005, 0);
  counterweightGroup.add(cwEye);

  // Turned brass cylindrical horological plumb weight (strictly bottom-to-top outward winding)
  const cwProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, -COUNTERWEIGHT_LENGTH),
    new THREE.Vector2(0.011, -COUNTERWEIGHT_LENGTH + 0.016),
    new THREE.Vector2(0.025, -COUNTERWEIGHT_LENGTH + 0.032),
    new THREE.Vector2(0.027, -COUNTERWEIGHT_LENGTH + 0.038),
    new THREE.Vector2(0.027, -0.032),
    new THREE.Vector2(0.022, -0.024),
    new THREE.Vector2(0.012, -0.016),
    new THREE.Vector2(0.0065, -0.010),
    new THREE.Vector2(0.001, -0.009),
  ];
  const cwGeo = new THREE.LatheGeometry(cwProfile, 30);
  cwGeo.computeVertexNormals();
  geometries.push(cwGeo);
  const cwBody = new THREE.Mesh(cwGeo, materials.satinBrass);
  cwBody.castShadow = true;
  cwBody.receiveShadow = true;
  counterweightGroup.add(cwBody);

  // Dark oxidized-bronze weighted waist band around the counterweight barrel
  const cwBandGeo = new THREE.CylinderGeometry(0.0278, 0.0278, 0.018, 30);
  geometries.push(cwBandGeo);
  const cwBand = new THREE.Mesh(cwBandGeo, materials.agedBronze);
  cwBand.position.set(0, -COUNTERWEIGHT_LENGTH * 0.50, 0);
  cwBand.castShadow = true;
  counterweightGroup.add(cwBand);

  return {
    group,
    archFrameGroup,
    swivelGroup,
    wheelGroup,
    counterweightGroup,
    dispose: () => {
      for (const g of geometries) g.dispose();
    },
  };
}

/**
 * Builds the elongated 8-facet Bone-Calcite teardrop pendulum bob with seamless
 * turned satin-brass upper wire ferrule and sharp conical horological plumb tip.
 */
export function buildPendulumBob(materials: PendulumMaterials): {
  group: THREE.Group;
  hitMesh: THREE.Mesh;
  dispose: () => void;
} {
  const group = new THREE.Group();
  group.name = 'PendulumBob';

  const geometries: THREE.BufferGeometry[] = [];

  // Smooth-curved 8-panel honed Bone-Calcite teardrop (smooth spline normals longitudinally,
  // crisp vertical ridge bevels between the 8 facets — zero ring creases or diagonal seams!)
  const bodySpline = new THREE.SplineCurve([
    new THREE.Vector2(0.056, -0.095),
    new THREE.Vector2(0.082, -0.056),
    new THREE.Vector2(0.087, -0.012),
    new THREE.Vector2(0.073, 0.050),
    new THREE.Vector2(0.048, 0.102),
  ]);
  const vSteps = 16;
  const facets = 8;
  const stationPts = bodySpline.getPoints(vSteps);
  const bPos: number[] = [];
  const bNorm: number[] = [];

  for (let f = 0; f < facets; f++) {
    const th0 = (f / facets) * Math.PI * 2;
    const th1 = ((f + 1) / facets) * Math.PI * 2;
    const thMid = 0.5 * (th0 + th1);
    const nxMid = Math.sin(thMid);
    const nzMid = Math.cos(thMid);
    const sin0 = Math.sin(th0);
    const cos0 = Math.cos(th0);
    const sin1 = Math.sin(th1);
    const cos1 = Math.cos(th1);

    const facetNorms: THREE.Vector3[] = [];
    for (let i = 0; i <= vSteps; i++) {
      const pPrev = stationPts[Math.max(0, i - 1)];
      const pNext = stationPts[Math.min(vSteps, i + 1)];
      const dr = pNext.x - pPrev.x;
      const dy = pNext.y - pPrev.y;
      const fn = new THREE.Vector3(dy * nxMid, -dr, dy * nzMid).normalize();
      facetNorms.push(fn);
    }

    for (let i = 0; i < vSteps; i++) {
      const r0 = stationPts[i].x;
      const y0 = stationPts[i].y;
      const r1 = stationPts[i + 1].x;
      const y1 = stationPts[i + 1].y;
      const n0 = facetNorms[i];
      const n1 = facetNorms[i + 1];

      const l0x = r0 * sin0, l0z = r0 * cos0;
      const r0x = r0 * sin1, r0z = r0 * cos1;
      const r1x = r1 * sin1, r1z = r1 * cos1;
      const l1x = r1 * sin0, l1z = r1 * cos0;

      // Tri 1: L0 -> R0 -> L1 (outward CCW)
      bPos.push(l0x, y0, l0z, r0x, y0, r0z, l1x, y1, l1z);
      bNorm.push(n0.x, n0.y, n0.z, n0.x, n0.y, n0.z, n1.x, n1.y, n1.z);

      // Tri 2: R0 -> R1 -> L1 (outward CCW)
      bPos.push(r0x, y0, r0z, r1x, y1, r1z, l1x, y1, l1z);
      bNorm.push(n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n1.x, n1.y, n1.z);
    }

    // Close bottom and top 8-gon caps inside brass ferrule/tip for watertight signedVol > 0
    const rBot = stationPts[0].x;
    const yBot = stationPts[0].y;
    bPos.push(0, yBot, 0, rBot * sin1, yBot, rBot * cos1, rBot * sin0, yBot, rBot * cos0);
    bNorm.push(0, -1, 0, 0, -1, 0, 0, -1, 0);

    const rTop = stationPts[vSteps].x;
    const yTop = stationPts[vSteps].y;
    bPos.push(0, yTop, 0, rTop * sin0, yTop, rTop * cos0, rTop * sin1, yTop, rTop * cos1);
    bNorm.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
  }

  const bodyGeo = new THREE.BufferGeometry();
  bodyGeo.setAttribute('position', new THREE.Float32BufferAttribute(bPos, 3));
  bodyGeo.setAttribute('normal', new THREE.Float32BufferAttribute(bNorm, 3));
  geometries.push(bodyGeo);

  const bodyMesh = new THREE.Mesh(bodyGeo, materials.boneCalciteBob);
  bodyMesh.castShadow = true;
  bodyMesh.receiveShadow = true;
  group.add(bodyMesh);

  // Refined turned satin-brass upper wire ferrule tapering cleanly down to wire diameter (0.0062)
  const capProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, 0.096),
    new THREE.Vector2(0.047, 0.096),
    new THREE.Vector2(0.052, 0.104),
    new THREE.Vector2(0.049, 0.112),
    new THREE.Vector2(0.034, 0.138),
    new THREE.Vector2(0.019, 0.155),
    new THREE.Vector2(0.0115, 0.165),
    new THREE.Vector2(0.0072, BOB_TOP_COLLET_OFFSET),
    new THREE.Vector2(0.0062, BOB_TOP_COLLET_OFFSET + 0.007),
    new THREE.Vector2(0.001, BOB_TOP_COLLET_OFFSET + 0.008),
  ];
  const capGeo = new THREE.LatheGeometry(capProfile, 32);
  capGeo.computeVertexNormals();
  geometries.push(capGeo);
  const capMesh = new THREE.Mesh(capGeo, materials.satinBrass);
  capMesh.castShadow = true;
  capMesh.receiveShadow = true;
  group.add(capMesh);

  // Crisp conical satin-brass horological plumb tip terminating at -BOB_TIP_OFFSET
  const tipProfile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, -BOB_TIP_OFFSET),
    new THREE.Vector2(0.013, -BOB_TIP_OFFSET + 0.016),
    new THREE.Vector2(0.044, -0.114),
    new THREE.Vector2(0.059, -0.095),
    new THREE.Vector2(0.057, -0.090),
    new THREE.Vector2(0.001, -0.090),
  ];
  const tipGeo = new THREE.LatheGeometry(tipProfile, 32);
  tipGeo.computeVertexNormals();
  geometries.push(tipGeo);
  const tipMesh = new THREE.Mesh(tipGeo, materials.satinBrass);
  tipMesh.castShadow = true;
  tipMesh.receiveShadow = true;
  group.add(tipMesh);

  const hitGeo = new THREE.SphereGeometry(0.17, 14, 10);
  geometries.push(hitGeo);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const hitMesh = new THREE.Mesh(hitGeo, hitMat);
  hitMesh.name = 'PendulumBobHit';
  group.add(hitMesh);

  return {
    group,
    hitMesh,
    dispose: () => {
      for (const g of geometries) g.dispose();
      hitMat.dispose();
    },
  };
}

interface CleavagePlane {
  normal: THREE.Vector3;
  dist: number;
  capMask: number;
}

function detRand(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

/**
 * Exact 3D convex polyhedron builder from half-space intersection `{ n_k · x <= d_k }`.
 * Computes smooth crown-rim proximity on adjacent shoulder vertices so high-frequency mineral FBM
 * breaks the mineral encrustation border organically instead of leaving a razor-straight polygon edge.
 */
function buildCleavedCrystalFromPlanes(
  planes: CleavagePlane[],
  crownNormal: THREE.Vector3,
  crownDist: number
): THREE.BufferGeometry {
  const positions: number[] = [];
  const normals: number[] = [];
  const capMasks: number[] = [];
  const bottomAOs: number[] = [];

  const upRef = new THREE.Vector3(0, 1, 0);
  const altRef = new THREE.Vector3(1, 0, 0);
  const uAxis = new THREE.Vector3();
  const vAxis = new THREE.Vector3();
  const edgeA = new THREE.Vector3();
  const edgeB = new THREE.Vector3();
  const crossN = new THREE.Vector3();

  const big = 1.0;
  const eps = 1e-6;

  for (let i = 0; i < planes.length; i++) {
    const pi = planes[i];
    const n = pi.normal;
    const ref = Math.abs(n.y) < 0.9 ? upRef : altRef;
    uAxis.crossVectors(ref, n).normalize();
    vAxis.crossVectors(n, uAxis).normalize();

    const origin = n.clone().multiplyScalar(pi.dist);
    let poly: THREE.Vector3[] = [
      origin.clone().addScaledVector(uAxis, -big).addScaledVector(vAxis, -big),
      origin.clone().addScaledVector(uAxis, big).addScaledVector(vAxis, -big),
      origin.clone().addScaledVector(uAxis, big).addScaledVector(vAxis, big),
      origin.clone().addScaledVector(uAxis, -big).addScaledVector(vAxis, big),
    ];

    for (let j = 0; j < planes.length; j++) {
      if (i === j || poly.length < 3) continue;
      const pj = planes[j];
      const nextPoly: THREE.Vector3[] = [];

      for (let k = 0; k < poly.length; k++) {
        const cur = poly[k];
        const nxt = poly[(k + 1) % poly.length];
        const dCur = pj.normal.dot(cur) - pj.dist;
        const dNxt = pj.normal.dot(nxt) - pj.dist;
        const inCur = dCur <= eps;
        const inNxt = dNxt <= eps;

        if (inCur && inNxt) {
          nextPoly.push(nxt);
        } else if (inCur && !inNxt) {
          const t = dCur / (dCur - dNxt);
          nextPoly.push(cur.clone().lerp(nxt, Math.max(0, Math.min(1, t))));
        } else if (!inCur && inNxt) {
          const t = dCur / (dCur - dNxt);
          nextPoly.push(cur.clone().lerp(nxt, Math.max(0, Math.min(1, t))));
          nextPoly.push(nxt);
        }
      }

      poly = nextPoly;
    }

    if (poly.length < 3) continue;

    edgeA.subVectors(poly[1], poly[0]);
    edgeB.subVectors(poly[2], poly[0]);
    crossN.crossVectors(edgeA, edgeB);
    if (crossN.dot(n) < 0) {
      poly.reverse();
    }

    const v0 = poly[0];
    for (let k = 1; k < poly.length - 1; k++) {
      const v1 = poly[k];
      const v2 = poly[k + 1];
      const tri = [v0, v1, v2];
      for (let m = 0; m < 3; m++) {
        const v = tri[m];
        positions.push(v.x, v.y, v.z);
        normals.push(n.x, n.y, n.z);

        // Smooth crown-rim spill on adjacent bevel vertices -> jagged organic mineral crust edge!
        let mask = pi.capMask;
        if (mask < 0.99 && n.y > -0.3) {
          const distBelowCrown = Math.max(0, crownDist - crownNormal.dot(v));
          const rimSpill = Math.max(0, 1.0 - distBelowCrown / 0.014) * 0.56;
          if (rimSpill > mask) mask = rimSpill;
        }
        capMasks.push(mask);

        const aoT = Math.max(0, Math.min(1, (v.y + 0.004) / 0.036));
        bottomAOs.push(aoT * aoT * (3 - 2 * aoT));
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setAttribute('aCapMask', new THREE.Float32BufferAttribute(capMasks, 1));
  geo.setAttribute('aBottomAO', new THREE.Float32BufferAttribute(bottomAOs, 1));
  return geo;
}

/**
 * Generates 4 distinct raw cleaved magnetite lodestone crystal polyhedra
 * seated strictly flush on the curved iron dome with broad mineral crown faces.
 */
export function createStoneGeometryVariants(): {
  variants: THREE.BufferGeometry[];
  dispose: () => void;
} {
  const variants: THREE.BufferGeometry[] = [];

  for (let v = 0; v < 4; v++) {
    const seedBase = (v + 1) * 37.3;
    const planes: CleavagePlane[] = [];

    planes.push({
      normal: new THREE.Vector3(0, -1, 0),
      dist: 0.006,
      capMask: 0.0,
    });

    const tiltAngle = 0.14 + detRand(seedBase + 1) * 0.16;
    const tiltAz = detRand(seedBase + 2) * Math.PI * 2;
    const crownNormal = new THREE.Vector3(
      Math.sin(tiltAngle) * Math.cos(tiltAz),
      Math.cos(tiltAngle),
      Math.sin(tiltAngle) * Math.sin(tiltAz)
    ).normalize();
    const crownDist = STONE_CROWN_HEIGHT * (0.76 + detRand(seedBase + 3) * 0.10);
    planes.push({
      normal: crownNormal,
      dist: crownDist,
      capMask: 1.0,
    });

    const sideCount = v % 2 === 0 ? 5 : 6;
    const azOffset = detRand(seedBase + 5) * Math.PI * 2;
    for (let s = 0; s < sideCount; s++) {
      const az =
        azOffset + (s / sideCount) * Math.PI * 2 + (detRand(seedBase + 10 + s) - 0.5) * 0.30;
      const elev = -0.10 + (detRand(seedBase + 20 + s) - 0.5) * 0.18;
      const n = new THREE.Vector3(
        Math.cos(elev) * Math.cos(az),
        Math.sin(elev),
        Math.cos(elev) * Math.sin(az)
      ).normalize();
      const d = STONE_RADIUS * (0.94 + detRand(seedBase + 30 + s) * 0.16);
      planes.push({ normal: n, dist: d, capMask: 0.0 });
    }

    const shoulderCount = sideCount;
    const shOffset = azOffset + Math.PI / sideCount;
    for (let s = 0; s < shoulderCount; s++) {
      const az =
        shOffset + (s / shoulderCount) * Math.PI * 2 + (detRand(seedBase + 40 + s) - 0.5) * 0.28;
      const elev = 0.44 + detRand(seedBase + 50 + s) * 0.18;
      const n = new THREE.Vector3(
        Math.cos(elev) * Math.cos(az),
        Math.sin(elev),
        Math.cos(elev) * Math.sin(az)
      ).normalize();
      const d = STONE_RADIUS * (0.96 + detRand(seedBase + 60 + s) * 0.14);
      const aligned = n.dot(crownNormal) > 0.72 ? 0.36 : 0.0;
      planes.push({ normal: n, dist: d, capMask: aligned });
    }

    variants.push(buildCleavedCrystalFromPlanes(planes, crownNormal, crownDist));
  }

  return {
    variants,
    dispose: () => {
      for (const g of variants) g.dispose();
    },
  };
}

/**
 * Helper: zero-allocation swept circular tube mesh with verified outward FrontSide winding.
 */
class DynamicWireTube {
  readonly mesh: THREE.Mesh;
  private readonly geometry: THREE.BufferGeometry;
  private readonly segs: number;
  private readonly radSegs = 10;
  private readonly positions: Float32Array;
  private readonly normals: Float32Array;
  private readonly tan = new THREE.Vector3();
  private readonly prevTan = new THREE.Vector3();
  private readonly norm = new THREE.Vector3();
  private readonly binorm = new THREE.Vector3();
  private readonly rotQ = new THREE.Quaternion();
  private readonly tmp = new THREE.Vector3();

  constructor(segs: number, material: THREE.Material, name: string) {
    this.segs = segs;
    const vertCount = (segs + 1) * (this.radSegs + 1);
    this.positions = new Float32Array(vertCount * 3);
    this.normals = new Float32Array(vertCount * 3);

    const indices: number[] = [];
    for (let i = 0; i < segs; i++) {
      for (let j = 0; j < this.radSegs; j++) {
        const a = i * (this.radSegs + 1) + j;
        const b = (i + 1) * (this.radSegs + 1) + j;
        const c = (i + 1) * (this.radSegs + 1) + (j + 1);
        const d = i * (this.radSegs + 1) + (j + 1);
        // Verified outward FrontSide winding: (d - a) x (b - a) = +radialOut > 0
        indices.push(a, d, b, b, d, c);
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3));
    this.geometry.setIndex(indices);

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.name = name;
    this.mesh.castShadow = true;
    this.mesh.frustumCulled = false;
  }

  updateFromPoints(pts: THREE.Vector3[], radius: number): void {
    this.prevTan.subVectors(pts[1], pts[0]);
    if (this.prevTan.lengthSq() > 1e-10) {
      this.prevTan.normalize();
    } else {
      this.prevTan.set(0, -1, 0);
    }

    if (Math.abs(this.prevTan.x) < 0.85) {
      this.norm.set(1, 0, 0);
    } else {
      this.norm.set(0, 0, 1);
    }
    this.binorm.crossVectors(this.prevTan, this.norm).normalize();
    this.norm.crossVectors(this.binorm, this.prevTan).normalize();

    let vIdx = 0;
    for (let i = 0; i <= this.segs; i++) {
      if (i === 0) {
        this.tan.copy(this.prevTan);
      } else if (i === this.segs) {
        this.tan.subVectors(pts[i], pts[i - 1]);
      } else {
        this.tan.subVectors(pts[i + 1], pts[i - 1]);
      }
      if (this.tan.lengthSq() > 1e-10) {
        this.tan.normalize();
      } else {
        this.tan.copy(this.prevTan);
      }

      if (i > 0) {
        this.rotQ.setFromUnitVectors(this.prevTan, this.tan);
        this.norm.applyQuaternion(this.rotQ).normalize();
        this.binorm.crossVectors(this.tan, this.norm).normalize();
        this.prevTan.copy(this.tan);
      }

      const center = pts[i];
      for (let j = 0; j <= this.radSegs; j++) {
        const angle = (j / this.radSegs) * Math.PI * 2;
        const ca = Math.cos(angle);
        const sa = Math.sin(angle);
        this.tmp
          .set(0, 0, 0)
          .addScaledVector(this.norm, ca)
          .addScaledVector(this.binorm, sa);

        const o = vIdx * 3;
        this.normals[o] = this.tmp.x;
        this.normals[o + 1] = this.tmp.y;
        this.normals[o + 2] = this.tmp.z;

        this.positions[o] = center.x + this.tmp.x * radius;
        this.positions[o + 1] = center.y + this.tmp.y * radius;
        this.positions[o + 2] = center.z + this.tmp.z * radius;
        vIdx++;
      }
    }

    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.normal as THREE.BufferAttribute).needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

/**
 * Zero-allocation 3-segment Razor-Taut Brass Plumb Wire system:
 * 1. Main taut straight wire from exact outer pulley tangent departure -> bob brass collet
 * 2. Groove wrap arc seated cleanly inside the outer brass sheave V-groove
 * 3. Vertical taut counterweight line from coaxial inner drum -> rising/falling brass counterweight
 */
export class TautPlumbWire {
  readonly group: THREE.Group;
  private readonly mainTube: DynamicWireTube;
  private readonly wrapTube: DynamicWireTube;
  private readonly cwTube: DynamicWireTube;

  private readonly mainSegs = 20;
  private readonly wrapSegs = 16;
  private readonly cwSegs = 6;

  private readonly mainPts: THREE.Vector3[] = [];
  private readonly wrapPts: THREE.Vector3[] = [];
  private readonly cwPts: THREE.Vector3[] = [];
  private readonly perpDir = new THREE.Vector3();

  constructor(material: THREE.Material) {
    this.group = new THREE.Group();
    this.group.name = 'TautBrassPlumbWireGroup';

    this.mainTube = new DynamicWireTube(this.mainSegs, material, 'MainTautPlumbWire');
    this.wrapTube = new DynamicWireTube(this.wrapSegs, material, 'PulleyGrooveWrapWire');
    this.cwTube = new DynamicWireTube(this.cwSegs, material, 'CounterweightDropWire');

    this.group.add(this.mainTube.mesh, this.wrapTube.mesh, this.cwTube.mesh);

    for (let i = 0; i <= this.mainSegs; i++) this.mainPts.push(new THREE.Vector3());
    for (let i = 0; i <= this.wrapSegs; i++) this.wrapPts.push(new THREE.Vector3());
    for (let i = 0; i <= this.cwSegs; i++) this.cwPts.push(new THREE.Vector3());
  }

  update(frame: PulleyKinematicsFrame): void {
    const sinAz = Math.sin(frame.azimuth);
    const cosAz = Math.cos(frame.azimuth);
    const rightX = cosAz;
    const rightZ = -sinAz;
    const fwdX = sinAz;
    const fwdZ = cosAz;

    // 1. Main taut straight plumb wire (exact straight line under tension + tiny kick micro-shimmer)
    this.perpDir.set(rightX, 0, rightZ);
    const pluckOffset = Math.sin(frame.pluckPhase) * frame.pluckAmp * 0.0022;

    for (let i = 0; i <= this.mainSegs; i++) {
      const t = i / this.mainSegs;
      const shimmer = Math.sin(t * Math.PI) * pluckOffset;
      this.mainPts[i]
        .lerpVectors(frame.bobTangentWorld, frame.bobColletWorld, t)
        .addScaledVector(this.perpDir, shimmer);
    }
    this.mainTube.updateFromPoints(this.mainPts, 0.0046);

    // 2. Pulley V-groove wrap arc wrapping cleanly over top of outer brass sheave to departAngleLocal
    const outerAxleX = rightX * PULLEY_OUTER_OFFSET_X;
    const outerAxleZ = rightZ * PULLEY_OUTER_OFFSET_X;
    const startAng = Math.PI * 0.92; // stays cleanly nested inside upper V-groove with zero bottom nub
    let endAng = frame.departAngleLocal;
    if (endAng > startAng) endAng -= Math.PI * 2;

    for (let i = 0; i <= this.wrapSegs; i++) {
      const t = i / this.wrapSegs;
      const ang = startAng + (endAng - startAng) * t;
      const cz = Math.cos(ang) * PULLEY_OUTER_RADIUS;
      const cy = Math.sin(ang) * PULLEY_OUTER_RADIUS;
      this.wrapPts[i].set(outerAxleX + fwdX * cz, PULLEY_AXLE_Y + cy, outerAxleZ + fwdZ * cz);
    }
    this.wrapTube.updateFromPoints(this.wrapPts, 0.0045);

    // 3. Taut straight vertical counterweight take-up line from inner drum down to counterweight eyelet
    for (let i = 0; i <= this.cwSegs; i++) {
      const t = i / this.cwSegs;
      this.cwPts[i].lerpVectors(frame.cwTangentWorld, frame.cwEyeletWorld, t);
    }
    this.cwTube.updateFromPoints(this.cwPts, 0.0036);
  }

  dispose(): void {
    this.mainTube.dispose();
    this.wrapTube.dispose();
    this.cwTube.dispose();
  }
}

/**
 * Zero-allocation ephemeral alpha-faded orbital phase filament trailing softly behind the bob tip.
 */
export class OrbitalWakeRibbon {
  readonly mesh: THREE.Mesh;
  private readonly geometry: THREE.BufferGeometry;
  private readonly positions: Float32Array;
  private readonly colors: Float32Array;
  private readonly alphas: Float32Array;
  private readonly pCur = new THREE.Vector3();
  private readonly pPrev = new THREE.Vector3();
  private readonly tan = new THREE.Vector3();
  private readonly toCam = new THREE.Vector3();
  private readonly side = new THREE.Vector3();
  private readonly prevSide = new THREE.Vector3(0, 1, 0);
  private readonly ochreLin = MINERAL_PALETTE.yellowOchre.rInfLinear;
  private readonly calciteLin = MINERAL_PALETTE.boneCalcite.rInfLinear;
  private readonly cinnabarLin = MINERAL_PALETTE.cinnabarVermilion.rInfLinear;
  private readonly verdigrisLin = MINERAL_PALETTE.verdigrisMalachite.rInfLinear;

  constructor(material: THREE.Material) {
    this.positions = new Float32Array(TRAIL_LENGTH * 2 * 3);
    this.colors = new Float32Array(TRAIL_LENGTH * 2 * 3);
    this.alphas = new Float32Array(TRAIL_LENGTH * 2);

    const indices: number[] = [];
    for (let i = 0; i < TRAIL_LENGTH - 1; i++) {
      const a = i * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      indices.push(a, c, b, b, c, d);
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    this.geometry.setAttribute('aAlpha', new THREE.BufferAttribute(this.alphas, 1));
    this.geometry.setIndex(indices);

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.name = 'OrbitalWakeRibbon';
    this.mesh.frustumCulled = false;
  }

  update(
    trailPositions: Float32Array,
    trailSpeeds: Float32Array,
    trailCharge: Float32Array,
    head: number,
    count: number,
    cameraPos: THREE.Vector3
  ): void {
    const active = Math.min(count, TRAIL_LENGTH);
    if (active < 4) {
      this.mesh.visible = false;
      return;
    }
    this.mesh.visible = true;

    for (let i = 0; i < TRAIL_LENGTH; i++) {
      const ageIdx = Math.max(0, active - 1 - (TRAIL_LENGTH - 1 - i));
      const ringIdx = (head - 1 - (active - 1 - ageIdx) + TRAIL_LENGTH * 2) % TRAIL_LENGTH;

      this.pCur.set(
        trailPositions[ringIdx * 3],
        trailPositions[ringIdx * 3 + 1],
        trailPositions[ringIdx * 3 + 2]
      );

      if (i > 0) {
        this.tan.subVectors(this.pCur, this.pPrev);
        if (this.tan.lengthSq() > 1e-7) {
          this.tan.normalize();
        } else {
          this.tan.set(1, 0, 0);
        }
      } else {
        this.tan.set(1, 0, 0);
      }
      this.pPrev.copy(this.pCur);

      this.toCam.subVectors(cameraPos, this.pCur).normalize();
      this.side.crossVectors(this.tan, this.toCam);
      if (this.side.lengthSq() > 1e-5) {
        this.side.normalize();
        if (i > 0 && this.side.dot(this.prevSide) < 0) {
          this.side.negate();
        }
        this.prevSide.copy(this.side);
      } else {
        this.side.copy(this.prevSide);
      }

      const frac = i / (TRAIL_LENGTH - 1);
      const speedNorm = Math.min(1, Math.max(0, (trailSpeeds[ringIdx] - 0.35) / 2.4));
      const charge = trailCharge[ringIdx];

      const headFade = Math.min(1, Math.max(0, (1.0 - frac) / 0.08));
      const tailFade = Math.pow(frac, 2.2);
      const alpha = tailFade * headFade * (0.35 + 0.65 * speedNorm) * 0.28;
      const halfWidth = 0.0005 + Math.pow(frac, 1.8) * 0.0033 * (0.40 + 0.60 * speedNorm);

      let r = this.ochreLin[0] * (1 - frac) + this.calciteLin[0] * frac;
      let g = this.ochreLin[1] * (1 - frac) + this.calciteLin[1] * frac;
      let b = this.ochreLin[2] * (1 - frac) + this.calciteLin[2] * frac;
      if (charge > 0.05) {
        const w = Math.min(0.78, charge * 0.78);
        r = r * (1 - w) + this.cinnabarLin[0] * w;
        g = g * (1 - w) + this.cinnabarLin[1] * w;
        b = b * (1 - w) + this.cinnabarLin[2] * w;
      } else if (charge < -0.05) {
        const w = Math.min(0.78, -charge * 0.78);
        r = r * (1 - w) + this.verdigrisLin[0] * w;
        g = g * (1 - w) + this.verdigrisLin[1] * w;
        b = b * (1 - w) + this.verdigrisLin[2] * w;
      }

      const base = i * 6;
      this.positions[base] = this.pCur.x + this.side.x * halfWidth;
      this.positions[base + 1] = this.pCur.y + this.side.y * halfWidth;
      this.positions[base + 2] = this.pCur.z + this.side.z * halfWidth;

      this.positions[base + 3] = this.pCur.x - this.side.x * halfWidth;
      this.positions[base + 4] = this.pCur.y - this.side.y * halfWidth;
      this.positions[base + 5] = this.pCur.z - this.side.z * halfWidth;

      this.colors[base] = r;
      this.colors[base + 1] = g;
      this.colors[base + 2] = b;
      this.colors[base + 3] = r;
      this.colors[base + 4] = g;
      this.colors[base + 5] = b;

      this.alphas[i * 2] = alpha;
      this.alphas[i * 2 + 1] = alpha;
    }

    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.aAlpha as THREE.BufferAttribute).needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
  }
}
