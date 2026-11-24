import * as THREE from 'three/webgpu';
import {
  CONE_RX_TOP,
  CONE_RZ_TOP,
  PLANAR_HALF_W_TOP,
  VEIL_Y_TOP,
  type ModeWeights,
} from './veilSurface';

export interface SlitLampMaterialsContract {
  forgedIron: THREE.Material;
  satinBrass: THREE.Material;
  knurledBrass: THREE.Material;
  bluedSteel: THREE.Material;
  linearFilament: THREE.Material;
  coneFilament: THREE.Material;
}

export interface SlitLampAssembly {
  group: THREE.Group;
  hitMeshes: THREE.Mesh[];
  focusDrums: {
    left: THREE.Object3D;
    right: THREE.Object3D;
    front: THREE.Object3D;
  };
  jaws: {
    frontLeft: THREE.Object3D;
    frontRight: THREE.Object3D;
    rearLeft: THREE.Object3D;
    rearRight: THREE.Object3D;
  };
  coneTurretGroup: THREE.Group;
  linearFilamentMesh: THREE.Mesh;
  coneFilamentMesh: THREE.Mesh;
  updateArticulation: (weights: ModeWeights, drumAngleRad: number) => void;
  dispose: () => void;
}

interface MeshBuilder {
  positions: number[];
  normals: number[];
  uvs: number[];
  knurlValleys: number[];
}

function createMeshBuilder(): MeshBuilder {
  return { positions: [], normals: [], uvs: [], knurlValleys: [] };
}

function pushTri(
  b: MeshBuilder,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  n0: THREE.Vector3,
  n1: THREE.Vector3 = n0,
  n2: THREE.Vector3 = n0,
  kv0 = 1,
  kv1 = 1,
  kv2 = 1
): void {
  b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  b.normals.push(n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z);
  b.uvs.push(0, 0, 1, 0, 0, 1);
  b.knurlValleys.push(kv0, kv1, kv2);
}

/**
 * Emits a convex polygon sorted strictly counter-clockwise around `outwardNormal`,
 * guaranteeing 100% outward-facing triangle winding (`signedVol > 0`).
 */
function pushConvexFace(
  b: MeshBuilder,
  verts: THREE.Vector3[],
  outwardNormal: THREE.Vector3
): void {
  if (verts.length < 3) return;
  const n = outwardNormal.clone().normalize();
  const center = new THREE.Vector3();
  for (let i = 0; i < verts.length; i++) center.add(verts[i]);
  center.divideScalar(verts.length);

  const up = Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const tangent = new THREE.Vector3().crossVectors(up, n).normalize();
  const bitangent = new THREE.Vector3().crossVectors(n, tangent).normalize();

  const sorted = verts
    .map((v) => {
      const rel = v.clone().sub(center);
      return { v, ang: Math.atan2(rel.dot(bitangent), rel.dot(tangent)) };
    })
    .sort((a, c) => a.ang - c.ang);

  for (let i = 1; i < sorted.length - 1; i++) {
    pushTri(b, sorted[0].v, sorted[i].v, sorted[i + 1].v, n);
  }
}

function finalizeBuilder(b: MeshBuilder, includeKnurlAttr = false): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(b.positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(b.normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uvs, 2));
  if (includeKnurlAttr) {
    geo.setAttribute('aKnurlValley', new THREE.Float32BufferAttribute(b.knurlValleys, 1));
  }
  return geo;
}

/**
 * Watertight 26-face chamfered rectangular block (6 faces + 12 bevels + 8 corners)
 * with strictly outward-wound triangles and crisp machined specular bevel edges.
 */
export function createChamferedBlockGeometry(
  hx: number,
  hy: number,
  hz: number,
  bevel: number
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const bx = Math.min(bevel, hx * 0.45);
  const by = Math.min(bevel, hy * 0.45);
  const bz = Math.min(bevel, hz * 0.45);
  const ix = hx - bx;
  const iy = hy - by;
  const iz = hz - bz;

  const vx = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * hx, sy * iy, sz * iz);
  const vy = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * ix, sy * hy, sz * iz);
  const vz = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * ix, sy * iy, sz * hz);

  for (const s of [-1, 1]) {
    pushConvexFace(b, [vx(s, -1, -1), vx(s, 1, -1), vx(s, 1, 1), vx(s, -1, 1)], new THREE.Vector3(s, 0, 0));
    pushConvexFace(b, [vy(-1, s, -1), vy(1, s, -1), vy(1, s, 1), vy(-1, s, 1)], new THREE.Vector3(0, s, 0));
    pushConvexFace(b, [vz(-1, -1, s), vz(1, -1, s), vz(1, 1, s), vz(-1, 1, s)], new THREE.Vector3(0, 0, s));
  }

  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      pushConvexFace(b, [vx(s1, s2, -1), vx(s1, s2, 1), vy(s1, s2, 1), vy(s1, s2, -1)], new THREE.Vector3(s1, s2, 0));
      pushConvexFace(b, [vx(s1, -1, s2), vx(s1, 1, s2), vz(s1, 1, s2), vz(s1, -1, s2)], new THREE.Vector3(s1, 0, s2));
      pushConvexFace(b, [vy(-1, s1, s2), vy(1, s1, s2), vz(1, s1, s2), vz(-1, s1, s2)], new THREE.Vector3(0, s1, s2));
    }
  }

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        pushConvexFace(b, [vx(sx, sy, sz), vy(sx, sy, sz), vz(sx, sy, sz)], new THREE.Vector3(sx, sy, sz));
      }
    }
  }

  return finalizeBuilder(b);
}

/**
 * Watertight beveled cylinder centered at origin along the Y axis (`y in [-halfLen, +halfLen]`)
 * with strictly outward CCW winding (`signedVol > 0`).
 */
export function createBeveledCylinderGeometry(
  radius: number,
  halfLen: number,
  bevel: number,
  radialSegs = 28,
  rxScale = 1,
  rzScale = 1
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const bev = Math.min(bevel, radius * 0.35, halfLen * 0.42);
  const rInner = radius - bev;
  const yInner = halfLen - bev;

  // Profile rings ordered strictly bottom-to-top for outward normals
  const rings: Array<{ r: number; y: number }> = [
    { r: rInner, y: -halfLen },
    { r: radius, y: -yInner },
    { r: radius, y: yInner },
    { r: rInner, y: halfLen },
  ];

  const pt = (r: number, y: number, ang: number) =>
    new THREE.Vector3(r * rxScale * Math.sin(ang), y, r * rzScale * Math.cos(ang));

  const ringOutNorm = (ang: number, ny: number) =>
    new THREE.Vector3(Math.sin(ang) / rxScale, ny, Math.cos(ang) / rzScale).normalize();

  for (let ri = 0; ri < rings.length - 1; ri++) {
    const r0 = rings[ri];
    const r1 = rings[ri + 1];
    const dr = r1.r - r0.r;
    const dy = r1.y - r0.y;
    const nyComp = -dr / Math.max(1e-5, Math.hypot(dr, dy));
    const nrComp = dy / Math.max(1e-5, Math.hypot(dr, dy));

    for (let s = 0; s < radialSegs; s++) {
      const a0 = (s / radialSegs) * Math.PI * 2;
      const a1 = ((s + 1) / radialSegs) * Math.PI * 2;
      const p00 = pt(r0.r, r0.y, a0);
      const p10 = pt(r0.r, r0.y, a1);
      const p11 = pt(r1.r, r1.y, a1);
      const p01 = pt(r1.r, r1.y, a0);

      const n0 = ringOutNorm(a0, nyComp / Math.max(0.25, nrComp));
      const n1 = ringOutNorm(a1, nyComp / Math.max(0.25, nrComp));
      pushTri(b, p00, p10, p11, n0, n1, n1);
      pushTri(b, p00, p11, p01, n0, n1, n0);
    }
  }

  // Top cap (+Y) and bottom cap (-Y)
  const topCenter = new THREE.Vector3(0, halfLen, 0);
  const botCenter = new THREE.Vector3(0, -halfLen, 0);
  const upN = new THREE.Vector3(0, 1, 0);
  const downN = new THREE.Vector3(0, -1, 0);

  for (let s = 0; s < radialSegs; s++) {
    const a0 = (s / radialSegs) * Math.PI * 2;
    const a1 = ((s + 1) / radialSegs) * Math.PI * 2;
    pushTri(b, topCenter, pt(rInner, halfLen, a0), pt(rInner, halfLen, a1), upN);
    pushTri(b, botCenter, pt(rInner, -halfLen, a1), pt(rInner, -halfLen, a0), downN);
  }

  return finalizeBuilder(b);
}

/**
 * Watertight 3D diamond-knurled satin-brass focus drum along the Y axis, carrying
 * vertex attribute `aKnurlValley` in `[0, 1]` for crisp machined groove occlusion.
 */
export function createDiamondKnurledDrumGeometry(
  radius = 0.058,
  halfLen = 0.048,
  teethAround = 26,
  ringsAlong = 6
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const bevel = Math.min(0.008, halfLen * 0.25);
  const yKnurlMin = -halfLen + bevel;
  const yKnurlMax = halfLen - bevel;
  const knurlDepth = radius * 0.095;
  const rCap = radius - bevel * 0.85;

  const evalDiamond = (sIdx: number, rIdx: number): { p: THREE.Vector3; kv: number } => {
    const u = sIdx / (teethAround * 2);
    const v = rIdx / ringsAlong;
    const ang = u * Math.PI * 2;
    const y = yKnurlMin + v * (yKnurlMax - yKnurlMin);
    const phaseA = Math.abs((((sIdx + rIdx) % 2) + 2) % 2);
    const kv = phaseA === 0 ? 1.0 : 0.0;
    const r = radius - (1 - kv) * knurlDepth;
    return { p: new THREE.Vector3(r * Math.sin(ang), y, r * Math.cos(ang)), kv };
  };

  const nCols = teethAround * 2;
  const triN = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();

  // Faceted diamond knurled barrel
  for (let r = 0; r < ringsAlong; r++) {
    for (let s = 0; s < nCols; s++) {
      const sNext = (s + 1) % nCols;
      const v00 = evalDiamond(s, r);
      const v10 = evalDiamond(sNext, r);
      const v11 = evalDiamond(sNext, r + 1);
      const v01 = evalDiamond(s, r + 1);

      e1.subVectors(v10.p, v00.p);
      e2.subVectors(v11.p, v00.p);
      triN.crossVectors(e1, e2).normalize();
      pushTri(b, v00.p, v10.p, v11.p, triN, triN, triN, v00.kv, v10.kv, v11.kv);

      e1.subVectors(v11.p, v00.p);
      e2.subVectors(v01.p, v00.p);
      triN.crossVectors(e1, e2).normalize();
      pushTri(b, v00.p, v11.p, v01.p, triN, triN, triN, v00.kv, v11.kv, v01.kv);
    }
  }

  // Smooth beveled collars connecting knurl ends to flat end-caps
  const topCenter = new THREE.Vector3(0, halfLen, 0);
  const botCenter = new THREE.Vector3(0, -halfLen, 0);
  const upN = new THREE.Vector3(0, 1, 0);
  const downN = new THREE.Vector3(0, -1, 0);

  for (let s = 0; s < nCols; s++) {
    const sNext = (s + 1) % nCols;
    const a0 = (s / nCols) * Math.PI * 2;
    const a1 = ((s + 1) / nCols) * Math.PI * 2;

    const topK0 = evalDiamond(s, ringsAlong);
    const topK1 = evalDiamond(sNext, ringsAlong);
    const topC0 = new THREE.Vector3(rCap * Math.sin(a0), halfLen, rCap * Math.cos(a0));
    const topC1 = new THREE.Vector3(rCap * Math.sin(a1), halfLen, rCap * Math.cos(a1));

    e1.subVectors(topK1.p, topK0.p);
    e2.subVectors(topC1, topK0.p);
    triN.crossVectors(e1, e2).normalize();
    pushTri(b, topK0.p, topK1.p, topC1, triN, triN, triN, topK0.kv, topK1.kv, 0.92);
    pushTri(b, topK0.p, topC1, topC0, triN, triN, triN, topK0.kv, 0.92, 0.92);
    pushTri(b, topCenter, topC0, topC1, upN, upN, upN, 0.92, 0.92, 0.92);

    const botK0 = evalDiamond(s, 0);
    const botK1 = evalDiamond(sNext, 0);
    const botC0 = new THREE.Vector3(rCap * Math.sin(a0), -halfLen, rCap * Math.cos(a0));
    const botC1 = new THREE.Vector3(rCap * Math.sin(a1), -halfLen, rCap * Math.cos(a1));

    e1.subVectors(botC1, botC0);
    e2.subVectors(botK1.p, botC0);
    triN.crossVectors(e1, e2).normalize();
    pushTri(b, botC0, botC1, botK1.p, triN, triN, triN, 0.92, 0.92, botK1.kv);
    pushTri(b, botC0, botK1.p, botK0.p, triN, triN, triN, 0.92, botK1.kv, botK0.kv);
    pushTri(b, botCenter, botC1, botC0, downN, downN, downN, 0.92, 0.92, 0.92);
  }

  return finalizeBuilder(b, true);
}

/**
 * Watertight closed elliptical torus ring in the XZ plane (`signedVol > 0`, 100% outward winding).
 */
export function createEllipticalTorusGeometry(
  rx: number,
  rz: number,
  tubeR: number,
  uSegs = 40,
  vSegs = 14
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const evalPt = (u: number, v: number): { p: THREE.Vector3; n: THREE.Vector3 } => {
    const sinU = Math.sin(u);
    const cosU = Math.cos(u);
    const cx = rx * sinU;
    const cz = rz * cosU;
    // Outward horizontal normal on the ellipse
    const outX = sinU / rx;
    const outZ = cosU / rz;
    const outLen = Math.hypot(outX, outZ) || 1;
    const nx = outX / outLen;
    const nz = outZ / outLen;

    const cosV = Math.cos(v);
    const sinV = Math.sin(v);
    const n = new THREE.Vector3(nx * cosV, sinV, nz * cosV).normalize();
    const p = new THREE.Vector3(cx + tubeR * nx * cosV, tubeR * sinV, cz + tubeR * nz * cosV);
    return { p, n };
  };

  for (let iu = 0; iu < uSegs; iu++) {
    const u0 = (iu / uSegs) * Math.PI * 2;
    const u1 = ((iu + 1) / uSegs) * Math.PI * 2;
    for (let iv = 0; iv < vSegs; iv++) {
      const v0 = (iv / vSegs) * Math.PI * 2;
      const v1 = ((iv + 1) / vSegs) * Math.PI * 2;
      const q00 = evalPt(u0, v0);
      const q10 = evalPt(u1, v0);
      const q11 = evalPt(u1, v1);
      const q01 = evalPt(u0, v1);

      pushTri(b, q00.p, q10.p, q11.p, q00.n, q10.n, q11.n);
      pushTri(b, q00.p, q11.p, q01.p, q00.n, q11.n, q01.n);
    }
  }

  return finalizeBuilder(b);
}

/**
 * Builds the machined overhead brass & dark-iron studio optical slit-lamp collimator assembly.
 * Every solid component is closed and strictly outward-wound (`signedVol > 0`).
 */
export function buildSlitLampAssembly(mats: SlitLampMaterialsContract): SlitLampAssembly {
  const group = new THREE.Group();
  group.name = 'OverheadOpticalSlitLamp';

  const ownedGeos: THREE.BufferGeometry[] = [];
  const hitMeshes: THREE.Mesh[] = [];

  const trackGeo = <T extends THREE.BufferGeometry>(g: T): T => {
    ownedGeos.push(g);
    return g;
  };

  const addPart = (
    name: string,
    geo: THREE.BufferGeometry,
    mat: THREE.Material,
    parent: THREE.Object3D = group,
    clickable = true
  ): THREE.Mesh => {
    const m = new THREE.Mesh(geo, mat);
    m.name = name;
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    if (clickable) hitMeshes.push(m);
    return m;
  };

  const barrelY = VEIL_Y_TOP + 0.165; // 1.325

  // 1. Main Forged-Iron Chamfered Optical Plenum Barrel
  const barrelGeo = trackGeo(createChamferedBlockGeometry(0.94, 0.105, 0.152, 0.022));
  const barrelMesh = addPart('OpticalPlenumBarrel', barrelGeo, mats.forgedIron);
  barrelMesh.position.set(0, barrelY, 0);

  // Lower forged-iron optical baffle rails framing the underside brass aperture channel
  const lowerRailGeo = trackGeo(createChamferedBlockGeometry(0.92, 0.032, 0.032, 0.008));
  for (const s of [-1, 1]) {
    const rail = addPart(`LowerBaffleRail_${s > 0 ? 'Front' : 'Rear'}`, lowerRailGeo, mats.forgedIron);
    rail.position.set(0, VEIL_Y_TOP + 0.044, s * 0.118);
  }

  // 2. 17 Milled Dark-Iron Transverse Cooling Fins along the upper spine
  const wingFinGeo = trackGeo(createChamferedBlockGeometry(0.015, 0.068, 0.165, 0.005));
  const finCount = 17;
  for (let i = 0; i < finCount; i++) {
    const fx = -0.82 + (i / (finCount - 1)) * 1.64;
    // Leave clean clearance around the central rotary condenser turret dome
    if (Math.abs(fx) < 0.23) continue;
    const fin = addPart(`CoolingFin_${i}`, wingFinGeo, mats.forgedIron);
    fin.position.set(fx, barrelY + 0.118, 0);
  }

  // 3. Central Coaxial Optical Condenser Turret Dome & Brass Vernier Collar
  const turretBaseGeo = trackGeo(createBeveledCylinderGeometry(0.225, 0.068, 0.014, 32));
  const turretBase = addPart('CentralCondenserTurret', turretBaseGeo, mats.forgedIron);
  turretBase.position.set(0, barrelY + 0.125, 0);

  const turretRingGeo = trackGeo(createBeveledCylinderGeometry(0.182, 0.022, 0.006, 32));
  const turretRing = addPart('TurretBrassVernierCollar', turretRingGeo, mats.satinBrass);
  turretRing.position.set(0, barrelY + 0.202, 0);

  const turretCapGeo = trackGeo(createBeveledCylinderGeometry(0.115, 0.024, 0.009, 28));
  const turretCap = addPart('TurretCrownCap', turretCapGeo, mats.forgedIron);
  turretCap.position.set(0, barrelY + 0.236, 0);

  const finialGeo = trackGeo(createBeveledCylinderGeometry(0.032, 0.022, 0.007, 20));
  const finial = addPart('TurretBrassFinial', finialGeo, mats.satinBrass);
  finial.position.set(0, barrelY + 0.272, 0);

  // Twin precision brass top suspension yokes & short blued-steel tension pins
  const yokeGeo = trackGeo(createChamferedBlockGeometry(0.036, 0.032, 0.042, 0.007));
  const pinGeo = trackGeo(createBeveledCylinderGeometry(0.011, 0.044, 0.003, 14));
  for (const s of [-1, 1]) {
    const yoke = addPart(`SuspensionYoke_${s > 0 ? 'R' : 'L'}`, yokeGeo, mats.satinBrass);
    yoke.position.set(s * 0.68, barrelY + 0.196, 0);
    const pin = addPart(`SuspensionPin_${s > 0 ? 'R' : 'L'}`, pinGeo, mats.bluedSteel);
    pin.position.set(s * 0.68, barrelY + 0.246, 0);
  }

  // 4. Stepped Satin-Brass End-Caps, Hex Cap Bolts & Longitudinal Micrometer Guide Rods
  const endCapGeo = trackGeo(createChamferedBlockGeometry(0.034, 0.122, 0.168, 0.011));
  const boltGeo = trackGeo(createBeveledCylinderGeometry(0.015, 0.012, 0.004, 6));
  for (const s of [-1, 1]) {
    const cap = addPart(`BrassEndCap_${s > 0 ? 'R' : 'L'}`, endCapGeo, mats.satinBrass);
    cap.position.set(s * 0.964, barrelY, 0);

    // 4 machined brass hex socket bolts on each end-cap
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const bolt = addPart(`EndCapBolt_${s}_${sy}_${sz}`, boltGeo, mats.bluedSteel);
        bolt.rotation.z = Math.PI * 0.5;
        bolt.position.set(s * 1.002, barrelY + sy * 0.078, sz * 0.118);
      }
    }
  }

  // Longitudinal satin-brass collimator rails along front and rear flanks
  const guideRodGeo = trackGeo(createBeveledCylinderGeometry(0.015, 0.975, 0.004, 18));
  const bracketGeo = trackGeo(createChamferedBlockGeometry(0.034, 0.032, 0.036, 0.006));
  for (const sz of [-1, 1]) {
    const rod = addPart(`CollimatorGuideRod_${sz > 0 ? 'Front' : 'Rear'}`, guideRodGeo, mats.satinBrass);
    rod.rotation.z = Math.PI * 0.5;
    rod.position.set(0, barrelY - 0.045, sz * 0.172);

    for (const bx of [-0.54, 0.54]) {
      const br = addPart(`GuideBracket_${sz}_${bx > 0 ? 'R' : 'L'}`, bracketGeo, mats.satinBrass);
      br.position.set(bx, barrelY - 0.045, sz * 0.162);
    }
  }

  // Front graduated brass micrometer bezel housing
  const frontBossGeo = trackGeo(createBeveledCylinderGeometry(0.072, 0.026, 0.008, 24));
  const frontBoss = addPart('FrontSelectorBezel', frontBossGeo, mats.satinBrass);
  frontBoss.rotation.x = Math.PI * 0.5;
  frontBoss.position.set(0, barrelY, 0.162);

  // 5. Three Machined Diamond-Knurled Satin-Brass Focus & Mode Thumbscrews
  const sideShaftGeo = trackGeo(createBeveledCylinderGeometry(0.022, 0.042, 0.004, 18));
  const sideCollarGeo = trackGeo(createBeveledCylinderGeometry(0.046, 0.016, 0.005, 22));
  const sideDrumGeo = trackGeo(createDiamondKnurledDrumGeometry(0.058, 0.048, 24, 6));
  const frontDrumGeo = trackGeo(createDiamondKnurledDrumGeometry(0.052, 0.034, 22, 5));

  const buildSideThumbscrew = (side: -1 | 1): THREE.Group => {
    const root = new THREE.Group();
    root.name = `SideFocusThumbscrew_${side > 0 ? 'R' : 'L'}`;
    root.position.set(side * 1.0, barrelY, 0);
    group.add(root);

    const collar = addPart(`SideCollar_${side > 0 ? 'R' : 'L'}`, sideCollarGeo, mats.satinBrass, root);
    collar.rotation.z = Math.PI * 0.5;
    collar.position.set(side * 0.014, 0, 0);

    const shaft = addPart(`SideShaft_${side > 0 ? 'R' : 'L'}`, sideShaftGeo, mats.bluedSteel, root);
    shaft.rotation.z = Math.PI * 0.5;
    shaft.position.set(side * 0.052, 0, 0);

    const rotor = new THREE.Group();
    rotor.position.set(side * 0.095, 0, 0);
    root.add(rotor);

    const drum = addPart(`KnurledDrum_${side > 0 ? 'R' : 'L'}`, sideDrumGeo, mats.knurledBrass, rotor);
    drum.rotation.z = Math.PI * 0.5;
    return rotor;
  };

  const leftDrum = buildSideThumbscrew(-1);
  const rightDrum = buildSideThumbscrew(1);

  const frontDrumRotor = new THREE.Group();
  frontDrumRotor.name = 'FrontApertureSelectorDrum';
  frontDrumRotor.position.set(0, barrelY, 0.214);
  group.add(frontDrumRotor);
  const frontDrumMesh = addPart('KnurledDrum_Front', frontDrumGeo, mats.knurledBrass, frontDrumRotor);
  frontDrumMesh.rotation.x = Math.PI * 0.5;

  // 6. Articulated Underside Brass Optical Slit-Jaw Blades (Physically Animate Across Modes!)
  const jawGeo = trackGeo(createChamferedBlockGeometry(0.438, 0.015, 0.044, 0.005));
  const jawY = VEIL_Y_TOP + 0.016;

  const frontLeftJaw = addPart('BrassSlitJaw_FL', jawGeo, mats.satinBrass);
  const frontRightJaw = addPart('BrassSlitJaw_FR', jawGeo, mats.satinBrass);
  const rearLeftJaw = addPart('BrassSlitJaw_RL', jawGeo, mats.satinBrass);
  const rearRightJaw = addPart('BrassSlitJaw_RR', jawGeo, mats.satinBrass);

  frontLeftJaw.position.set(-0.44, jawY, 0.058);
  frontRightJaw.position.set(0.44, jawY, 0.058);
  rearLeftJaw.position.set(-0.44, jawY, -0.058);
  rearRightJaw.position.set(0.44, jawY, -0.058);

  // 7. Central Deployable Brass Annular Iris Collar & Dark-Iron Conical Baffle Plug (Cone Mode)
  const coneTurretGroup = new THREE.Group();
  coneTurretGroup.name = 'CentralAnnularIrisTurret';
  coneTurretGroup.position.set(0, VEIL_Y_TOP + 0.042, 0);
  group.add(coneTurretGroup);

  const outerIrisRingGeo = trackGeo(
    createEllipticalTorusGeometry(CONE_RX_TOP + 0.032, CONE_RZ_TOP + 0.032, 0.019, 44, 14)
  );
  addPart('AnnularIrisOuterBezel', outerIrisRingGeo, mats.satinBrass, coneTurretGroup);

  const innerBaffleGeo = trackGeo(
    createBeveledCylinderGeometry(CONE_RX_TOP - 0.036, 0.019, 0.006, 36, 1.0, CONE_RZ_TOP / CONE_RX_TOP)
  );
  addPart('AnnularIrisCenterPlug', innerBaffleGeo, mats.forgedIron, coneTurretGroup);

  // 8. Recessed Glowing Optical Condenser Slit Filaments (Articulated Left/Right Linear + Annular Cone Ring)
  const linearFilGeo = trackGeo(
    createChamferedBlockGeometry(PLANAR_HALF_W_TOP * 0.495, 0.011, 0.013, 0.003)
  );
  const linearFilamentMesh = addPart('LinearSlitFilamentCore_L', linearFilGeo, mats.linearFilament, group, false);
  linearFilamentMesh.castShadow = false;
  linearFilamentMesh.receiveShadow = false;
  linearFilamentMesh.position.set(-0.43, VEIL_Y_TOP + 0.02, 0);

  const linearFilamentRight = addPart('LinearSlitFilamentCore_R', linearFilGeo, mats.linearFilament, group, false);
  linearFilamentRight.castShadow = false;
  linearFilamentRight.receiveShadow = false;
  linearFilamentRight.position.set(0.43, VEIL_Y_TOP + 0.02, 0);

  const coneFilGeo = trackGeo(createEllipticalTorusGeometry(CONE_RX_TOP, CONE_RZ_TOP, 0.011, 44, 12));
  const coneFilamentMesh = addPart('AnnularConeFilamentRing', coneFilGeo, mats.coneFilament, coneTurretGroup, false);
  coneFilamentMesh.castShadow = false;
  coneFilamentMesh.receiveShadow = false;
  coneFilamentMesh.position.set(0, -0.004, 0);

  // Continuous mechanical articulation driven directly by `ApertureModeController` weights
  const updateArticulation = (weights: ModeWeights, drumAngleRad: number): void => {
    const wS = weights.curvedS;
    const wC = weights.cone;

    // Spin all three diamond-knurled brass micrometer focus drums
    leftDrum.rotation.x = drumAngleRad;
    rightDrum.rotation.x = -drumAngleRad;
    frontDrumRotor.rotation.z = drumAngleRad * 1.15;

    // Articulated underside brass slit jaws:
    // - Planar (wS=0, wC=0): parallel razor slit at z = ±0.058, x = ±0.44
    // - Curved-S (wS=1): anti-symmetric S-cam skew (±0.032 Z offset + ±0.085 rad yaw)
    // - Cone (wC=1): jaws slide laterally outward to clear central annular iris & close linear flanks
    const jawGapZ = 0.058 * (1 - wC) + 0.044 * wC;
    const jawX = 0.44 + wC * 0.14;
    const sShiftZ = wS * 0.03;
    const sYaw = wS * 0.082;

    frontLeftJaw.position.set(-jawX, jawY, jawGapZ - sShiftZ);
    frontLeftJaw.rotation.y = sYaw;

    rearLeftJaw.position.set(-jawX, jawY, -jawGapZ - sShiftZ);
    rearLeftJaw.rotation.y = sYaw;

    frontRightJaw.position.set(jawX, jawY, jawGapZ + sShiftZ);
    frontRightJaw.rotation.y = -sYaw;

    rearRightJaw.position.set(jawX, jawY, -jawGapZ + sShiftZ);
    rearRightJaw.rotation.y = -sYaw;

    // Keep recessed incandescent filament segments flush inside the articulating jaw gaps
    linearFilamentMesh.position.set(-0.43, VEIL_Y_TOP + 0.02, -sShiftZ);
    linearFilamentMesh.rotation.y = sYaw;
    linearFilamentRight.position.set(0.43, VEIL_Y_TOP + 0.02, sShiftZ);
    linearFilamentRight.rotation.y = -sYaw;

    // Deploy central brass annular iris collar smoothly downward in Cone mode
    coneTurretGroup.position.y = VEIL_Y_TOP + 0.056 * (1 - wC) + 0.011 * wC;
    const irisScale = 0.72 + 0.28 * wC;
    coneTurretGroup.scale.set(irisScale, 1, irisScale);
    coneTurretGroup.rotation.y = (1 - wC) * 0.26;
  };

  updateArticulation({ planar: 1, curvedS: 0, cone: 0 }, 0);

  return {
    group,
    hitMeshes,
    focusDrums: {
      left: leftDrum,
      right: rightDrum,
      front: frontDrumRotor,
    },
    jaws: {
      frontLeft: frontLeftJaw,
      frontRight: frontRightJaw,
      rearLeft: rearLeftJaw,
      rearRight: rearRightJaw,
    },
    coneTurretGroup,
    linearFilamentMesh,
    coneFilamentMesh,
    updateArticulation,
    dispose: () => {
      for (let i = 0; i < ownedGeos.length; i++) ownedGeos[i].dispose();
    },
  };
}

/**
 * Builds the 1-draw-call 5-stratum volumetric light-curtain geometry (`84 x 64 x 5`).
 * Vertex positions live in canonical parametric `(u, v)` + `aStratum in [-1, +1]` and are
 * deformed live on GPU in TSL so all 3 aperture modes morph smoothly at 120 Hz with zero CPU rebuilds.
 */
export function buildVolumetricVeilGeometry(
  uSegs = 84,
  vSegs = 64,
  strataCount = 5
): THREE.BufferGeometry {
  const vertsPerLayer = (uSegs + 1) * (vSegs + 1);
  const totalVerts = vertsPerLayer * strataCount;
  const trisPerLayer = uSegs * vSegs * 2;
  const totalIndices = trisPerLayer * strataCount * 3;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const strata = new Float32Array(totalVerts);
  const indices = new Uint32Array(totalIndices);

  let vOffset = 0;
  let iOffset = 0;

  for (let s = 0; s < strataCount; s++) {
    const stratumNorm = strataCount > 1 ? (s / (strataCount - 1)) * 2 - 1 : 0;
    const baseVertex = vOffset;

    for (let iv = 0; iv <= vSegs; iv++) {
      const v = iv / vSegs;
      for (let iu = 0; iu <= uSegs; iu++) {
        const u = iu / uSegs;
        const p3 = vOffset * 3;
        const p2 = vOffset * 2;
        // Canonical bounding-box proxy coords (exact positions computed in GPU positionNode)
        positions[p3] = (u - 0.5) * 2.6;
        positions[p3 + 1] = VEIL_Y_TOP - v * 2.64;
        positions[p3 + 2] = stratumNorm * 0.05;
        normals[p3] = 0;
        normals[p3 + 1] = 0;
        normals[p3 + 2] = 1;
        uvs[p2] = u;
        uvs[p2 + 1] = v;
        strata[vOffset] = stratumNorm;
        vOffset++;
      }
    }

    const rowStride = uSegs + 1;
    for (let iv = 0; iv < vSegs; iv++) {
      for (let iu = 0; iu < uSegs; iu++) {
        const a = baseVertex + iv * rowStride + iu;
        const b = a + 1;
        const c = a + rowStride;
        const d = c + 1;
        indices[iOffset++] = a;
        indices[iOffset++] = c;
        indices[iOffset++] = b;
        indices[iOffset++] = b;
        indices[iOffset++] = c;
        indices[iOffset++] = d;
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('aStratum', new THREE.BufferAttribute(strata, 1));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  // Generous static bounding sphere enclosing all 3 morphing aperture shapes (no frustum pop)
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, -0.15, 0), 2.8);
  return geo;
}

/**
 * Builds twin grazing cross-section edge ribbons at `u = 0.04` and `u = 0.96` spanning the
 * physical slit beam thickness (`aStratum in [-1, +1]`), giving the flat/curved light sheet
 * a razor-clean luminous wedge cross-section even when orbited to an exact 90° side profile.
 */
export function buildGrazingCrossSectionGeometry(vSegs = 48): THREE.BufferGeometry {
  const edgeUs = [0.045, 0.955];
  const vertsPerEdge = (vSegs + 1) * 2;
  const totalVerts = vertsPerEdge * edgeUs.length;
  const totalIndices = vSegs * 2 * 3 * edgeUs.length;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const strata = new Float32Array(totalVerts);
  const indices = new Uint32Array(totalIndices);

  let vOffset = 0;
  let iOffset = 0;

  for (let e = 0; e < edgeUs.length; e++) {
    const uEdge = edgeUs[e];
    const baseVertex = vOffset;
    const nxSign = e === 0 ? -1 : 1;

    for (let iv = 0; iv <= vSegs; iv++) {
      const v = iv / vSegs;
      for (let side = 0; side < 2; side++) {
        const sNorm = side === 0 ? -1 : 1;
        const p3 = vOffset * 3;
        const p2 = vOffset * 2;
        positions[p3] = (uEdge - 0.5) * 2.4;
        positions[p3 + 1] = VEIL_Y_TOP - v * 2.64;
        positions[p3 + 2] = sNorm * 0.04;
        normals[p3] = nxSign;
        normals[p3 + 1] = 0;
        normals[p3 + 2] = 0;
        uvs[p2] = uEdge;
        uvs[p2 + 1] = v;
        strata[vOffset] = sNorm;
        vOffset++;
      }
    }

    for (let iv = 0; iv < vSegs; iv++) {
      const a = baseVertex + iv * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      indices[iOffset++] = a;
      indices[iOffset++] = c;
      indices[iOffset++] = b;
      indices[iOffset++] = b;
      indices[iOffset++] = c;
      indices[iOffset++] = d;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('aStratum', new THREE.BufferAttribute(strata, 1));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, -0.15, 0), 2.8);
  return geo;
}
