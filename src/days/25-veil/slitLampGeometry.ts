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
  vernierBrass?: THREE.Material;
  knurledBrass: THREE.Material;
  bluedSteel: THREE.Material;
  honedSlate?: THREE.Material;
  ceramicInsulator?: THREE.Material;
  opticalGlass?: THREE.Material;
  opticalBaffle?: THREE.Material;
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
    pushConvexFace(
      b,
      [vx(s, -1, -1), vx(s, 1, -1), vx(s, 1, 1), vx(s, -1, 1)],
      new THREE.Vector3(s, 0, 0)
    );
    pushConvexFace(
      b,
      [vy(-1, s, -1), vy(1, s, -1), vy(1, s, 1), vy(-1, s, 1)],
      new THREE.Vector3(0, s, 0)
    );
    pushConvexFace(
      b,
      [vz(-1, -1, s), vz(1, -1, s), vz(1, 1, s), vz(-1, 1, s)],
      new THREE.Vector3(0, 0, s)
    );
  }

  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      pushConvexFace(
        b,
        [vx(s1, s2, -1), vx(s1, s2, 1), vy(s1, s2, 1), vy(s1, s2, -1)],
        new THREE.Vector3(s1, s2, 0)
      );
      pushConvexFace(
        b,
        [vx(s1, -1, s2), vx(s1, 1, s2), vz(s1, 1, s2), vz(s1, -1, s2)],
        new THREE.Vector3(s1, 0, s2)
      );
      pushConvexFace(
        b,
        [vy(-1, s1, s2), vy(1, s1, s2), vz(1, s1, s2), vz(-1, s1, s2)],
        new THREE.Vector3(0, s1, s2)
      );
    }
  }

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        pushConvexFace(
          b,
          [vx(sx, sy, sz), vy(sx, sy, sz), vz(sx, sy, sz)],
          new THREE.Vector3(sx, sy, sz)
        );
      }
    }
  }

  return finalizeBuilder(b);
}

/**
 * Watertight closed profile lathe centered on the Y axis, built from rings ordered strictly
 * bottom-to-top (`r > 0`) with crisp per-band analytical normals and capped ends (`signedVol > 0`).
 */
export function createTurnedProfileGeometry(
  rings: ReadonlyArray<{ r: number; y: number }>,
  radialSegs = 32,
  rxScale = 1,
  rzScale = 1
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const pt = (r: number, y: number, ang: number) =>
    new THREE.Vector3(r * rxScale * Math.sin(ang), y, r * rzScale * Math.cos(ang));

  const ringOutNorm = (ang: number, nr: number, ny: number) =>
    new THREE.Vector3((Math.sin(ang) * nr) / rxScale, ny, (Math.cos(ang) * nr) / rzScale).normalize();

  for (let ri = 0; ri < rings.length - 1; ri++) {
    const r0 = rings[ri];
    const r1 = rings[ri + 1];
    const dr = r1.r - r0.r;
    const dy = r1.y - r0.y;
    const segLen = Math.hypot(dr, dy);
    if (segLen < 1e-6) continue;
    const nyComp = -dr / segLen;
    const nrComp = dy / segLen;

    for (let s = 0; s < radialSegs; s++) {
      const a0 = (s / radialSegs) * Math.PI * 2;
      const a1 = ((s + 1) / radialSegs) * Math.PI * 2;
      const p00 = pt(r0.r, r0.y, a0);
      const p10 = pt(r0.r, r0.y, a1);
      const p11 = pt(r1.r, r1.y, a1);
      const p01 = pt(r1.r, r1.y, a0);

      const n0 = ringOutNorm(a0, nrComp, nyComp);
      const n1 = ringOutNorm(a1, nrComp, nyComp);
      pushTri(b, p00, p10, p11, n0, n1, n1);
      pushTri(b, p00, p11, p01, n0, n1, n0);
    }
  }

  const first = rings[0];
  const last = rings[rings.length - 1];
  const botCenter = new THREE.Vector3(0, first.y, 0);
  const topCenter = new THREE.Vector3(0, last.y, 0);
  const upN = new THREE.Vector3(0, 1, 0);
  const downN = new THREE.Vector3(0, -1, 0);

  for (let s = 0; s < radialSegs; s++) {
    const a0 = (s / radialSegs) * Math.PI * 2;
    const a1 = ((s + 1) / radialSegs) * Math.PI * 2;
    pushTri(b, topCenter, pt(last.r, last.y, a0), pt(last.r, last.y, a1), upN);
    pushTri(b, botCenter, pt(first.r, first.y, a1), pt(first.r, first.y, a0), downN);
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
  const bev = Math.min(bevel, radius * 0.35, halfLen * 0.42);
  const rInner = radius - bev;
  const yInner = halfLen - bev;
  return createTurnedProfileGeometry(
    [
      { r: rInner, y: -halfLen },
      { r: radius, y: -yInner },
      { r: radius, y: yInner },
      { r: rInner, y: halfLen },
    ],
    radialSegs,
    rxScale,
    rzScale
  );
}

/**
 * Watertight ribbed steatite ceramic thermal insulator sleeve along Y (`y in [-halfLen, +halfLen]`),
 * featuring two crisp turned porcelain sheds flanking a deep shadow waist groove.
 */
export function createRibbedCeramicInsulatorGeometry(
  outerR: number,
  halfLen: number,
  radialSegs = 28
): THREE.BufferGeometry {
  const waistR = outerR * 0.78;
  const lipR = outerR * 0.88;
  return createTurnedProfileGeometry(
    [
      { r: lipR, y: -halfLen },
      { r: outerR, y: -halfLen * 0.76 },
      { r: outerR, y: -halfLen * 0.32 },
      { r: waistR, y: -halfLen * 0.14 },
      { r: waistR, y: halfLen * 0.14 },
      { r: outerR, y: halfLen * 0.32 },
      { r: outerR, y: halfLen * 0.76 },
      { r: lipR, y: halfLen },
    ],
    radialSegs
  );
}

/**
 * Watertight outward-wound 3D cross-hatched 4-sided raised-pyramid diamond-knurled satin-brass
 * focus drum centered along the Y axis (`y in [-halfLen, +halfLen]`), with smooth chamfered
 * brass collars, recessed lathe-turned concave end dimples, and `aKnurlValley` vertex AO.
 *
 * Uses 4 flat triangles per diamond cell -> perfectly symmetric +-45 deg diamond lattice
 * with zero diagonal barber-pole bias even under 4x macro inspection!
 */
export function createDiamondKnurledDrumGeometry(
  radius = 0.058,
  halfLen = 0.036,
  teethAround = 34,
  ringsAlong = 7
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const height = halfLen * 2;
  const safeChamfer = Math.min(0.0065, height * 0.18);
  const collarPad = Math.min(0.0022, height * 0.06);
  const knurlStart = -halfLen + safeChamfer + collarPad;
  const knurlEnd = halfLen - safeChamfer - collarPad;
  const knurlAmp = radius * 0.048;
  const valleyR = radius - knurlAmp * 0.74;
  const peakR = radius + knurlAmp;
  const dPhi = (Math.PI * 2) / teethAround;

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  const emitQuad = (
    p00: THREE.Vector3,
    p10: THREE.Vector3,
    p11: THREE.Vector3,
    p01: THREE.Vector3,
    kv00: number,
    kv10: number,
    kv11: number,
    kv01: number
  ) => {
    ab.subVectors(p10, p00);
    ac.subVectors(p11, p00);
    faceN.crossVectors(ab, ac).normalize();
    pushTri(b, p00, p10, p11, faceN, faceN, faceN, kv00, kv10, kv11);

    ab.subVectors(p11, p00);
    ac.subVectors(p01, p00);
    faceN.crossVectors(ab, ac).normalize();
    pushTri(b, p00, p11, p01, faceN, faceN, faceN, kv00, kv11, kv01);
  };

  // Angle parameterized so (p00 -> p10 -> p11) is strictly CCW from outside (+radial)
  const makeRing = (y: number, r: number): THREE.Vector3[] => {
    const pts: THREE.Vector3[] = [];
    for (let j = 0; j < teethAround; j++) {
      const phi = j * dPhi;
      pts.push(new THREE.Vector3(Math.cos(phi) * r, y, -Math.sin(phi) * r));
    }
    return pts;
  };

  const ringBot0 = makeRing(-halfLen, radius * 0.84);
  const ringBot1 = makeRing(-halfLen + safeChamfer, radius);
  const ringKnurlBot = makeRing(knurlStart, valleyR);
  const ringKnurlTop = makeRing(knurlEnd, valleyR);
  const ringTop1 = makeRing(halfLen - safeChamfer, radius);
  const ringTop0 = makeRing(halfLen, radius * 0.84);
  const ringDimpleRim = makeRing(halfLen, radius * 0.62);

  for (let j = 0; j < teethAround; j++) {
    const jNext = (j + 1) % teethAround;
    emitQuad(ringBot0[j], ringBot0[jNext], ringBot1[jNext], ringBot1[j], 0.86, 0.86, 0.96, 0.96);
    emitQuad(
      ringBot1[j],
      ringBot1[jNext],
      ringKnurlBot[jNext],
      ringKnurlBot[j],
      0.96,
      0.96,
      0.34,
      0.34
    );
    emitQuad(
      ringKnurlTop[j],
      ringKnurlTop[jNext],
      ringTop1[jNext],
      ringTop1[j],
      0.34,
      0.34,
      0.96,
      0.96
    );
    emitQuad(ringTop1[j], ringTop1[jNext], ringTop0[jNext], ringTop0[j], 0.96, 0.96, 0.92, 0.92);
    emitQuad(
      ringTop0[j],
      ringTop0[jNext],
      ringDimpleRim[jNext],
      ringDimpleRim[j],
      0.92,
      0.92,
      0.88,
      0.88
    );
  }

  // True 4-sided raised diamond pyramids (`ringsAlong` x `teethAround`)
  const bandRows: THREE.Vector3[][] = [ringKnurlBot];
  for (let r = 1; r < ringsAlong; r++) {
    const y = knurlStart + ((knurlEnd - knurlStart) * r) / ringsAlong;
    bandRows.push(makeRing(y, valleyR));
  }
  bandRows.push(ringKnurlTop);

  for (let r = 0; r < ringsAlong; r++) {
    const row0 = bandRows[r];
    const row1 = bandRows[r + 1];
    const yMid = knurlStart + ((knurlEnd - knurlStart) * (r + 0.5)) / ringsAlong;
    const edgeScale = r === 0 || r === ringsAlong - 1 ? 0.82 : 1.0;
    const apexR = valleyR + (peakR - valleyR) * edgeScale;

    for (let j = 0; j < teethAround; j++) {
      const jNext = (j + 1) % teethAround;
      const p00 = row0[j];
      const p10 = row0[jNext];
      const p11 = row1[jNext];
      const p01 = row1[j];
      const phiMid = (j + 0.5) * dPhi;
      const pMid = new THREE.Vector3(Math.cos(phiMid) * apexR, yMid, -Math.sin(phiMid) * apexR);

      const kvVal = 0.06;
      const kvTop = 1.0;

      // Bottom facet
      ab.subVectors(p10, p00);
      ac.subVectors(pMid, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTri(b, p00, p10, pMid, faceN, faceN, faceN, kvVal, kvVal, kvTop);

      // Right facet
      ab.subVectors(p11, p10);
      ac.subVectors(pMid, p10);
      faceN.crossVectors(ab, ac).normalize();
      pushTri(b, p10, p11, pMid, faceN, faceN, faceN, kvVal, kvVal, kvTop);

      // Top facet
      ab.subVectors(p01, p11);
      ac.subVectors(pMid, p11);
      faceN.crossVectors(ab, ac).normalize();
      pushTri(b, p11, p01, pMid, faceN, faceN, faceN, kvVal, kvVal, kvTop);

      // Left facet
      ab.subVectors(p00, p01);
      ac.subVectors(pMid, p01);
      faceN.crossVectors(ab, ac).normalize();
      pushTri(b, p01, p00, pMid, faceN, faceN, faceN, kvVal, kvVal, kvTop);
    }
  }

  // Bottom cap (-Y outward)
  const botCenter = new THREE.Vector3(0, -halfLen, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  for (let j = 0; j < teethAround; j++) {
    const jNext = (j + 1) % teethAround;
    pushTri(b, botCenter, ringBot0[jNext], ringBot0[j], downN, downN, downN, 0.84, 0.84, 0.84);
  }

  // Recessed machined concave top dimple cap (+Y outward)
  const topCenter = new THREE.Vector3(0, halfLen - 0.004, 0);
  for (let j = 0; j < teethAround; j++) {
    const jNext = (j + 1) % teethAround;
    ab.subVectors(ringDimpleRim[j], topCenter);
    ac.subVectors(ringDimpleRim[jNext], topCenter);
    faceN.crossVectors(ab, ac).normalize();
    pushTri(b, topCenter, ringDimpleRim[j], ringDimpleRim[jNext], faceN, faceN, faceN, 0.72, 0.9, 0.9);
  }

  return finalizeBuilder(b, true);
}

/**
 * Watertight single-start helical blued-steel micrometer lead screw centered along Y
 * (`y in [-halfLen, +halfLen]`) with crisp ISO V-threads that visibly advance when rotated.
 */
export function createHelicalLeadScrewGeometry(
  halfLen = 0.044,
  minorR = 0.0145,
  majorR = 0.0192,
  turns = 8,
  radialSegs = 26,
  axialRings = 52
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const yMin = -halfLen;
  const yMax = halfLen;
  const threadStart = yMin + 0.005;
  const threadEnd = yMax - 0.005;
  const pitch = (threadEnd - threadStart) / turns;
  const dPhi = (Math.PI * 2) / radialSegs;

  const rings: THREE.Vector3[][] = [];
  for (let r = 0; r <= axialRings; r++) {
    const t = r / axialRings;
    const y = yMin + (yMax - yMin) * t;
    const endFadeIn = Math.min(1, Math.max(0, (y - threadStart) / (pitch * 0.6)));
    const endFadeOut = Math.min(1, Math.max(0, (threadEnd - y) / (pitch * 0.6)));
    const threadEnv = endFadeIn * endFadeOut;

    const pts: THREE.Vector3[] = [];
    for (let j = 0; j < radialSegs; j++) {
      const phi = j * dPhi;
      const phaseRaw = (y - threadStart) / pitch - phi / (Math.PI * 2);
      const phase = ((phaseRaw % 1) + 1) % 1;
      const vTooth = 1.0 - Math.abs(2.0 * phase - 1.0);
      const shaped = Math.min(1, Math.max(0, (vTooth - 0.1) / 0.8));
      const rad = minorR + (majorR - minorR) * shaped * threadEnv;
      pts.push(new THREE.Vector3(Math.cos(phi) * rad, y, -Math.sin(phi) * rad));
    }
    rings.push(pts);
  }

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  for (let r = 0; r < axialRings; r++) {
    const row0 = rings[r];
    const row1 = rings[r + 1];
    for (let j = 0; j < radialSegs; j++) {
      const jNext = (j + 1) % radialSegs;
      const p00 = row0[j];
      const p10 = row0[jNext];
      const p11 = row1[jNext];
      const p01 = row1[j];

      ab.subVectors(p10, p00);
      ac.subVectors(p11, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTri(b, p00, p10, p11, faceN);

      ab.subVectors(p11, p00);
      ac.subVectors(p01, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTri(b, p00, p11, p01, faceN);
    }
  }

  const botCenter = new THREE.Vector3(0, yMin, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  const topCenter = new THREE.Vector3(0, yMax, 0);
  const upN = new THREE.Vector3(0, 1, 0);
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    pushTri(b, botCenter, rings[0][jNext], rings[0][j], downN);
    pushTri(b, topCenter, rings[axialRings][j], rings[axialRings][jNext], upN);
  }

  return finalizeBuilder(b);
}

/**
 * Watertight machined cheese-head slotted instrument screw centered along Y
 * (`y in [-0.004, headH]`), with a true recessed screwdriver slot milled cleanly across the crown.
 */
export function createSlottedCapScrewGeometry(
  radius = 0.015,
  headH = 0.011
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const segs = 20;
  const yBot = -0.004;
  const yShoulder = headH * 0.62;
  const ySlotFloor = headH * 0.54;
  const rTop = radius * 0.84;
  const slotHalfW = radius * 0.21;
  const dPhi = (Math.PI * 2) / segs;

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  const botRing: THREE.Vector3[] = [];
  const shRing: THREE.Vector3[] = [];
  const topRing: THREE.Vector3[] = [];

  for (let j = 0; j < segs; j++) {
    const phi = j * dPhi;
    const c = Math.cos(phi);
    const s = -Math.sin(phi);
    botRing.push(new THREE.Vector3(c * radius, yBot, s * radius));
    shRing.push(new THREE.Vector3(c * radius, yShoulder, s * radius));
    topRing.push(new THREE.Vector3(c * rTop, headH, s * rTop));
  }

  // Outer cylinder + top chamfer bevel
  for (let j = 0; j < segs; j++) {
    const jNext = (j + 1) % segs;
    const b0 = botRing[j];
    const b1 = botRing[jNext];
    const s0 = shRing[j];
    const s1 = shRing[jNext];
    const t0 = topRing[j];
    const t1 = topRing[jNext];

    ab.subVectors(b1, b0);
    ac.subVectors(s1, b0);
    faceN.crossVectors(ab, ac).normalize();
    pushTri(b, b0, b1, s1, faceN);
    pushTri(b, b0, s1, s0, faceN);

    ab.subVectors(s1, s0);
    ac.subVectors(t1, s0);
    faceN.crossVectors(ab, ac).normalize();
    pushTri(b, s0, s1, t1, faceN);
    pushTri(b, s0, t1, t0, faceN);
  }

  // Bottom cap (-Y outward)
  const botCenter = new THREE.Vector3(0, yBot, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  for (let j = 0; j < segs; j++) {
    const jNext = (j + 1) % segs;
    pushTri(b, botCenter, botRing[jNext], botRing[j], downN);
  }

  // Slotted crown: split top disc cleanly into +Z half-moon, -Z half-moon, and recessed slot trench
  const upN = new THREE.Vector3(0, 1, 0);
  const slotX = Math.sqrt(Math.max(1e-6, rTop * rTop - slotHalfW * slotHalfW));

  const buildLobe = (zSign: 1 | -1) => {
    const arcPts: THREE.Vector3[] = [];
    const steps = 10;
    const aStart = zSign > 0 ? Math.asin(slotHalfW / rTop) : Math.PI + Math.asin(slotHalfW / rTop);
    const aEnd = zSign > 0 ? Math.PI - Math.asin(slotHalfW / rTop) : Math.PI * 2 - Math.asin(slotHalfW / rTop);
    for (let i = 0; i <= steps; i++) {
      const a = aStart + ((aEnd - aStart) * i) / steps;
      arcPts.push(new THREE.Vector3(Math.cos(a) * rTop, headH, Math.sin(a) * rTop));
    }
    pushConvexFace(b, arcPts, upN);
  };
  buildLobe(1);
  buildLobe(-1);

  // Recessed screwdriver slot floor & vertical side walls (closed at both chord ends)
  const pFL = new THREE.Vector3(-slotX, ySlotFloor, slotHalfW);
  const pFR = new THREE.Vector3(slotX, ySlotFloor, slotHalfW);
  const pBL = new THREE.Vector3(-slotX, ySlotFloor, -slotHalfW);
  const pBR = new THREE.Vector3(slotX, ySlotFloor, -slotHalfW);

  const tFL = new THREE.Vector3(-slotX, headH, slotHalfW);
  const tFR = new THREE.Vector3(slotX, headH, slotHalfW);
  const tBL = new THREE.Vector3(-slotX, headH, -slotHalfW);
  const tBR = new THREE.Vector3(slotX, headH, -slotHalfW);

  pushConvexFace(b, [pFL, pFR, pBR, pBL], upN);
  pushConvexFace(b, [pFL, pFR, tFR, tFL], new THREE.Vector3(0, 0, -1));
  pushConvexFace(b, [pBL, pBR, tBR, tBL], new THREE.Vector3(0, 0, 1));
  // Subtle end-cap wedges sealing the slot mouth flush with the top rim
  pushConvexFace(b, [pFL, pBL, tBL, tFL], new THREE.Vector3(-1, 0, 0));
  pushConvexFace(b, [pFR, pBR, tBR, tFR], new THREE.Vector3(1, 0, 0));

  return finalizeBuilder(b);
}

/**
 * Watertight closed elliptical torus ring in the XZ plane (`signedVol > 0`, 100% outward winding).
 */
export function createEllipticalTorusGeometry(
  rx: number,
  rz: number,
  tubeR: number,
  uSegs = 44,
  vSegs = 16
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const evalPt = (u: number, v: number): { p: THREE.Vector3; n: THREE.Vector3 } => {
    const sinU = Math.sin(u);
    const cosU = Math.cos(u);
    const cx = rx * sinU;
    const cz = rz * cosU;
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
 * Builds the museum-grade machined overhead optical slit-lamp collimator assembly:
 *   - Oil-blackened cast & milled iron optical plenum + integrated central condenser chest
 *   - Recessed honed dark veined-slate thermal cheek inlays framed in brass with slotted rivets
 *   - 12 milled transverse cooling fins bound by twin blued-steel tie-rods & brass acorn nuts
 *   - Anchored rigging bail shackles insulated on ribbed Bone-Calcite steatite porcelain pads
 *   - Coaxial optical condenser turret with ceramic thermal ring & 72-tick engraved brass vernier collar
 *   - Split micrometer guide rails with clamped mid-span brackets & front mode-detent selector bezel
 *   - Helical blued-steel lead screws & true 4-sided raised-pyramid diamond-knurled focus drums
 *   - Underside optical light-trap cavity, cylindrical borosilicate condenser lens, zero-overhang
 *     articulated brass slit jaws, and deployable Cassegrain spider-vane annular iris turret
 *
 * Every solid component is closed and strictly outward-wound (`signedVol > 0`).
 */
export function buildSlitLampAssembly(mats: SlitLampMaterialsContract): SlitLampAssembly {
  const group = new THREE.Group();
  group.name = 'OverheadOpticalSlitLamp';

  const vernierMat = mats.vernierBrass ?? mats.satinBrass;
  const slateMat = mats.honedSlate ?? mats.forgedIron;
  const ceramicMat = mats.ceramicInsulator ?? mats.satinBrass;
  const glassMat = mats.opticalGlass ?? mats.linearFilament;
  const baffleMat = mats.opticalBaffle ?? mats.forgedIron;

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
    clickable = true,
    shadows = true
  ): THREE.Mesh => {
    const m = new THREE.Mesh(geo, mat);
    m.name = name;
    m.castShadow = shadows;
    m.receiveShadow = shadows;
    parent.add(m);
    if (clickable) hitMeshes.push(m);
    return m;
  };

  const barrelY = VEIL_Y_TOP + 0.165; // 1.325

  // ---------------------------------------------------------------------------
  // 1. Main Oil-Blackened Iron Optical Plenum Barrel + Integrated Central Condenser Chest
  // ---------------------------------------------------------------------------
  const barrelGeo = trackGeo(createChamferedBlockGeometry(0.935, 0.105, 0.152, 0.02));
  const barrelMesh = addPart('OpticalPlenumBarrel', barrelGeo, mats.forgedIron);
  barrelMesh.position.set(0, barrelY, 0);

  // Integrated wider central condenser chest seats the circular turret flush with zero overhang
  // AND houses the retracted annular iris cleanly inside its Z footprint!
  const chestGeo = trackGeo(createChamferedBlockGeometry(0.258, 0.112, 0.214, 0.022));
  const chestMesh = addPart('CentralCondenserChest', chestGeo, mats.forgedIron);
  chestMesh.position.set(0, barrelY, 0);

  // Machined satin-brass shoulder trim collars flanking the central condenser chest
  const chestCollarGeo = trackGeo(createChamferedBlockGeometry(0.014, 0.106, 0.194, 0.005));
  for (const sx of [-1, 1]) {
    const trim = addPart(`CentralChestBrassTrim_${sx > 0 ? 'R' : 'L'}`, chestCollarGeo, mats.satinBrass);
    trim.position.set(sx * 0.258, barrelY, 0);
  }

  // Split lower forged-iron optical baffle rails flanking the central iris bay (Z: +-0.114..+-0.152)
  const lowerRailGeo = trackGeo(createChamferedBlockGeometry(0.335, 0.031, 0.02, 0.0055));
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const rail = addPart(
        `LowerBaffleRail_${sz > 0 ? 'Front' : 'Rear'}_${sx > 0 ? 'R' : 'L'}`,
        lowerRailGeo,
        mats.forgedIron
      );
      rail.position.set(sx * 0.59, VEIL_Y_TOP + 0.044, sz * 0.132);
    }
  }

  // Stepped matte optical light-trap cavity ceiling ribs recessed above the slit throat
  const throatCeilingGeo = trackGeo(createChamferedBlockGeometry(0.91, 0.014, 0.108, 0.003));
  const throatCeiling = addPart('OpticalThroatLightTrapCeiling', throatCeilingGeo, baffleMat, group, false);
  throatCeiling.position.set(0, VEIL_Y_TOP + 0.056, 0);

  const trapRibGeo = trackGeo(createChamferedBlockGeometry(0.90, 0.008, 0.007, 0.002));
  for (const rz of [-0.072, -0.044, 0.044, 0.072]) {
    const rib = addPart(`OpticalTrapRib_${rz > 0 ? 'F' : 'R'}_${Math.round(Math.abs(rz) * 100)}`, trapRibGeo, baffleMat, group, false);
    rib.position.set(0, VEIL_Y_TOP + 0.044, rz);
  }

  // ---------------------------------------------------------------------------
  // 2. Recessed Honed Dark Veined-Slate Thermal Cheek Inlays + Framed Brass Bezels & Rivets
  // ---------------------------------------------------------------------------
  const cheekFrameGeo = trackGeo(createChamferedBlockGeometry(0.292, 0.062, 0.009, 0.003));
  const cheekSlateGeo = trackGeo(createChamferedBlockGeometry(0.262, 0.043, 0.011, 0.004));
  const miniScrewGeo = trackGeo(createSlottedCapScrewGeometry(0.0072, 0.006));

  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const tag = `${sz > 0 ? 'F' : 'R'}_${sx > 0 ? 'R' : 'L'}`;
      const cx = sx * 0.596;
      const cy = barrelY + 0.022; // 1.347 -> clean clearance above guide rails (y = 1.246)

      const frame = addPart(`SlateCheekBrassFrame_${tag}`, cheekFrameGeo, mats.satinBrass);
      frame.position.set(cx, cy, sz * 0.153);

      const plaque = addPart(`HonedSlateThermalInlay_${tag}`, cheekSlateGeo, slateMat);
      plaque.position.set(cx, cy, sz * 0.158);

      // 4 crisp slotted blued-steel corner retaining screws seated strictly inside the flat brass border
      for (const rx of [-1, 1]) {
        for (const ry of [-1, 1]) {
          const sc = addPart(
            `CheekScrew_${tag}_${rx > 0 ? 'R' : 'L'}_${ry > 0 ? 'T' : 'B'}`,
            miniScrewGeo,
            mats.bluedSteel
          );
          sc.rotation.x = sz > 0 ? Math.PI * 0.5 : -Math.PI * 0.5;
          sc.rotation.z = (rx * 0.45 + ry * 0.65 + sx * 0.3) * 0.9;
          sc.position.set(cx + rx * 0.277, cy + ry * 0.0505, sz * 0.161);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 3. 12 Milled Transverse Cooling Fins (2 + Rigging Bay + 4 per side) + Blued-Steel Tie-Rods
  // ---------------------------------------------------------------------------
  const finRootGeo = trackGeo(createChamferedBlockGeometry(0.022, 0.012, 0.156, 0.004));
  const wingFinGeo = trackGeo(createChamferedBlockGeometry(0.013, 0.056, 0.158, 0.0042));
  const finXs = [
    -0.85, -0.77,
    // Dedicated rigging shackle bay at x = -0.665 (zero fin collision!)
    -0.555, -0.475, -0.395, -0.315,
    0.315, 0.395, 0.475, 0.555,
    // Dedicated rigging shackle bay at x = +0.665
    0.77, 0.85,
  ];
  for (let i = 0; i < finXs.length; i++) {
    const fx = finXs[i];
    const root = addPart(`CoolingFinRoot_${i}`, finRootGeo, mats.forgedIron);
    root.position.set(fx, barrelY + 0.105, 0);
    const fin = addPart(`CoolingFin_${i}`, wingFinGeo, mats.forgedIron);
    fin.position.set(fx, barrelY + 0.156, 0);
  }

  // Twin longitudinal blued-steel cooling-fin spine tie-rods + brass acorn end-nuts per wing
  const finTieRodGeo = trackGeo(createBeveledCylinderGeometry(0.0062, 0.295, 0.002, 14));
  const acornNutGeo = trackGeo(createBeveledCylinderGeometry(0.0105, 0.007, 0.003, 6));
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const tie = addPart(`FinTieRod_${sx > 0 ? 'R' : 'L'}_${sz > 0 ? 'F' : 'B'}`, finTieRodGeo, mats.bluedSteel);
      tie.rotation.z = Math.PI * 0.5;
      tie.position.set(sx * 0.58, barrelY + 0.168, sz * 0.118);

      for (const endX of [sx * 0.292, sx * 0.872]) {
        const nut = addPart(
          `FinTieNut_${sx > 0 ? 'R' : 'L'}_${sz > 0 ? 'F' : 'B'}_${Math.abs(endX) > 0.5 ? 'Out' : 'In'}`,
          acornNutGeo,
          mats.satinBrass
        );
        nut.rotation.z = Math.PI * 0.5;
        nut.position.set(endX, barrelY + 0.168, sz * 0.118);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 4. Twin Anchored Suspension Rigging Shackles on Ribbed Ceramic Thermal Insulators
  // ---------------------------------------------------------------------------
  const rigBossGeo = trackGeo(createChamferedBlockGeometry(0.044, 0.016, 0.056, 0.005));
  const rigCeramicGeo = trackGeo(createRibbedCeramicInsulatorGeometry(0.034, 0.016, 26));
  const rigBaseGeo = trackGeo(createChamferedBlockGeometry(0.032, 0.011, 0.034, 0.004));
  const rigCheekGeo = trackGeo(createChamferedBlockGeometry(0.0075, 0.025, 0.022, 0.0028));
  const rigPinGeo = trackGeo(createBeveledCylinderGeometry(0.0075, 0.034, 0.0022, 16));
  const rigBailGeo = trackGeo(createEllipticalTorusGeometry(0.030, 0.028, 0.0068, 32, 12));

  for (const s of [-1, 1]) {
    const rx = s * 0.665;
    const tag = s > 0 ? 'R' : 'L';

    const boss = addPart(`SuspensionIronBoss_${tag}`, rigBossGeo, mats.forgedIron);
    boss.position.set(rx, barrelY + 0.112, 0);

    const insul = addPart(`SuspensionCeramicInsulator_${tag}`, rigCeramicGeo, ceramicMat);
    insul.position.set(rx, barrelY + 0.142, 0);

    const saddle = addPart(`SuspensionYoke_${tag}`, rigBaseGeo, mats.satinBrass);
    saddle.position.set(rx, barrelY + 0.168, 0);

    for (const cs of [-1, 1]) {
      const cheek = addPart(`SuspensionCheek_${tag}_${cs > 0 ? 'R' : 'L'}`, rigCheekGeo, mats.satinBrass);
      cheek.position.set(rx + cs * 0.019, barrelY + 0.194, 0);
    }

    const crossPin = addPart(`SuspensionPin_${tag}`, rigPinGeo, mats.bluedSteel);
    crossPin.rotation.z = Math.PI * 0.5;
    crossPin.position.set(rx, barrelY + 0.206, 0);

    const bail = addPart(`SuspensionBailRing_${tag}`, rigBailGeo, mats.satinBrass);
    bail.rotation.x = Math.PI * 0.5;
    bail.position.set(rx, barrelY + 0.232, 0);
  }

  // ---------------------------------------------------------------------------
  // 5. Central Coaxial Optical Condenser Turret, Ceramic Thermal Ring & Engraved Vernier Collar
  // ---------------------------------------------------------------------------
  const turretBaseGeo = trackGeo(
    createTurnedProfileGeometry(
      [
        { r: 0.204, y: -0.052 },
        { r: 0.214, y: -0.042 },
        { r: 0.214, y: 0.036 },
        { r: 0.200, y: 0.050 },
        { r: 0.192, y: 0.052 },
      ],
      44
    )
  );
  const turretBase = addPart('CentralCondenserTurret', turretBaseGeo, mats.forgedIron);
  turretBase.position.set(0, barrelY + 0.148, 0);

  // 8 circumferential blued-steel flange cap screws around the condenser turret deck
  const turretScrewGeo = trackGeo(createSlottedCapScrewGeometry(0.010, 0.007));
  for (let i = 0; i < 8; i++) {
    const ang = (i / 8) * Math.PI * 2 + Math.PI / 8;
    const sc = addPart(`TurretFlangeScrew_${i}`, turretScrewGeo, mats.bluedSteel);
    sc.rotation.y = ang + 0.4;
    sc.position.set(Math.cos(ang) * 0.193, barrelY + 0.200, Math.sin(ang) * 0.193);
  }

  // Glazed Bone-Calcite steatite porcelain thermal isolation ring beneath the brass vernier collar
  const turretCeramicGeo = trackGeo(createRibbedCeramicInsulatorGeometry(0.184, 0.011, 44));
  const turretCeramic = addPart('TurretCeramicThermalRing', turretCeramicGeo, ceramicMat);
  turretCeramic.position.set(0, barrelY + 0.210, 0);

  // Dedicated 72-tick engraved brass optical azimuth / aperture vernier collar!
  const turretRingGeo = trackGeo(
    createTurnedProfileGeometry(
      [
        { r: 0.168, y: -0.022 },
        { r: 0.176, y: -0.015 },
        { r: 0.176, y: 0.014 },
        { r: 0.164, y: 0.022 },
      ],
      48
    )
  );
  const turretRing = addPart('TurretBrassVernierCollar', turretRingGeo, vernierMat);
  turretRing.position.set(0, barrelY + 0.240, 0);

  // Small satin-brass vernier datum index pointer on the front lip of the crown cap
  const indexLugGeo = trackGeo(createChamferedBlockGeometry(0.010, 0.009, 0.020, 0.0025));
  const indexLug = addPart('TurretVernierIndexPointer', indexLugGeo, mats.satinBrass);
  indexLug.position.set(0, barrelY + 0.264, 0.120);

  const turretCapGeo = trackGeo(
    createTurnedProfileGeometry(
      [
        { r: 0.118, y: -0.022 },
        { r: 0.122, y: -0.014 },
        { r: 0.114, y: 0.012 },
        { r: 0.098, y: 0.022 },
      ],
      36
    )
  );
  const turretCap = addPart('TurretCrownCap', turretCapGeo, mats.forgedIron);
  turretCap.position.set(0, barrelY + 0.278, 0);

  // Turned brass turret crown finial with stepped shoulder + slotted cap screw
  const finialGeo = trackGeo(
    createTurnedProfileGeometry(
      [
        { r: 0.044, y: -0.020 },
        { r: 0.048, y: -0.012 },
        { r: 0.036, y: -0.002 },
        { r: 0.038, y: 0.012 },
        { r: 0.030, y: 0.020 },
      ],
      28
    )
  );
  const finial = addPart('TurretBrassFinial', finialGeo, mats.satinBrass);
  finial.position.set(0, barrelY + 0.314, 0);

  const finialScrew = addPart('TurretFinialScrew', turretScrewGeo, mats.bluedSteel);
  finialScrew.position.set(0, barrelY + 0.335, 0);

  // ---------------------------------------------------------------------------
  // 6. Stepped Satin-Brass End-Caps, Machined Washers & Slotted Cap Bolts
  // ---------------------------------------------------------------------------
  const endCapGeo = trackGeo(createChamferedBlockGeometry(0.032, 0.120, 0.166, 0.010));
  const washerGeo = trackGeo(createBeveledCylinderGeometry(0.019, 0.003, 0.0014, 20));
  const endBoltGeo = trackGeo(createSlottedCapScrewGeometry(0.0145, 0.011));

  for (const s of [-1, 1]) {
    const cap = addPart(`BrassEndCap_${s > 0 ? 'R' : 'L'}`, endCapGeo, mats.satinBrass);
    cap.position.set(s * 0.962, barrelY, 0);

    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const w = addPart(`EndCapWasher_${s}_${sy}_${sz}`, washerGeo, mats.satinBrass);
        w.rotation.z = s > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
        w.position.set(s * 0.994, barrelY + sy * 0.080, sz * 0.118);

        const bolt = addPart(`EndCapBolt_${s}_${sy}_${sz}`, endBoltGeo, mats.bluedSteel);
        bolt.rotation.z = s > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
        bolt.rotation.x = (sy * 0.7 + sz * 0.5) * 0.85;
        bolt.position.set(s * 0.998, barrelY + sy * 0.080, sz * 0.118);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 7. Split Longitudinal Satin-Brass Collimator Guide Rails & Clamped Brackets (Zero Intersections!)
  // ---------------------------------------------------------------------------
  const splitRailGeo = trackGeo(createBeveledCylinderGeometry(0.0125, 0.352, 0.0035, 20));
  const railLugGeo = trackGeo(createBeveledCylinderGeometry(0.022, 0.014, 0.004, 20));
  const bracketGeo = trackGeo(createChamferedBlockGeometry(0.032, 0.028, 0.032, 0.0055));
  const clampCollarGeo = trackGeo(createBeveledCylinderGeometry(0.0175, 0.007, 0.0025, 20));

  const railY = barrelY - 0.079; // 1.246 (clears both slate cheek plaques & front selector bezel!)
  for (const sz of [-1, 1]) {
    const faceTag = sz > 0 ? 'Front' : 'Rear';
    for (const sx of [-1, 1]) {
      const sideTag = sx > 0 ? 'R' : 'L';
      // Split rail spans cleanly between CentralCondenserChest (x = +-0.258) and BrassEndCap (x = +-0.945)
      const rod = addPart(`CollimatorGuideRod_${faceTag}_${sideTag}`, splitRailGeo, mats.satinBrass);
      rod.rotation.z = Math.PI * 0.5;
      rod.position.set(sx * 0.602, railY, sz * 0.172);

      // Stepped brass rail termination lug anchored into CentralCondenserChest flank
      const innerLug = addPart(`RailChestLug_${faceTag}_${sideTag}`, railLugGeo, mats.satinBrass);
      innerLug.rotation.z = Math.PI * 0.5;
      innerLug.position.set(sx * 0.264, railY, sz * 0.172);

      // Mid-span split-clamp mounting bracket + twin rail retention collars + blued-steel clamping screw
      const br = addPart(`GuideBracket_${faceTag}_${sideTag}`, bracketGeo, mats.satinBrass);
      br.position.set(sx * 0.596, railY, sz * 0.164);

      for (const cs of [-1, 1]) {
        const col = addPart(
          `GuideBracketCollar_${faceTag}_${sideTag}_${cs > 0 ? 'R' : 'L'}`,
          clampCollarGeo,
          mats.satinBrass
        );
        col.rotation.z = Math.PI * 0.5;
        col.position.set(sx * 0.596 + cs * 0.038, railY, sz * 0.172);
      }

      const clampScrew = addPart(`GuideBracketScrew_${faceTag}_${sideTag}`, miniScrewGeo, mats.bluedSteel);
      clampScrew.rotation.x = sz > 0 ? Math.PI * 0.5 : -Math.PI * 0.5;
      clampScrew.position.set(sx * 0.596, railY, sz * 0.197);
    }
  }

  // ---------------------------------------------------------------------------
  // 8. Front Engraved Mode-Selector Vernier Bezel, Detent Pips & True Diamond-Knurled Drum
  // ---------------------------------------------------------------------------
  const frontBossGeo = trackGeo(
    createTurnedProfileGeometry(
      [
        { r: 0.078, y: -0.016 },
        { r: 0.082, y: -0.010 },
        { r: 0.082, y: 0.008 },
        { r: 0.072, y: 0.016 },
      ],
      36
    )
  );
  const frontBoss = addPart('FrontSelectorBezel', frontBossGeo, vernierMat);
  frontBoss.rotation.x = Math.PI * 0.5;
  frontBoss.position.set(0, barrelY, 0.222);

  // 3 blued-steel optical mode detent pins (120 deg intervals on bezel rim: Planar / S-Veil / Cone)
  const detentPinGeo = trackGeo(createBeveledCylinderGeometry(0.006, 0.007, 0.002, 12));
  for (let i = 0; i < 3; i++) {
    const a = (i * Math.PI * 2) / 3 + Math.PI * 0.5;
    const dp = addPart(`ModeDetentPin_${i}`, detentPinGeo, mats.bluedSteel);
    dp.rotation.x = Math.PI * 0.5;
    dp.position.set(Math.cos(a) * 0.066, barrelY + Math.sin(a) * 0.066, 0.237);
  }

  const frontDrumGeo = trackGeo(createDiamondKnurledDrumGeometry(0.050, 0.025, 32, 6));
  const frontCapScrewGeo = trackGeo(createSlottedCapScrewGeometry(0.015, 0.008));

  const frontDrumRotor = new THREE.Group();
  frontDrumRotor.name = 'FrontApertureSelectorDrum';
  frontDrumRotor.position.set(0, barrelY, 0.262);
  group.add(frontDrumRotor);

  const frontDrumMesh = addPart('KnurledDrum_Front', frontDrumGeo, mats.knurledBrass, frontDrumRotor);
  frontDrumMesh.rotation.x = Math.PI * 0.5;

  const frontDrumScrew = addPart('FrontDrumLockScrew', frontCapScrewGeo, mats.bluedSteel, frontDrumRotor);
  frontDrumScrew.rotation.x = Math.PI * 0.5;
  frontDrumScrew.position.set(0, 0, 0.022);

  // Rotating index pointer needle on front drum collar pointing to active mode detent
  const needleGeo = trackGeo(createChamferedBlockGeometry(0.0045, 0.016, 0.005, 0.0014));
  const needle = addPart('FrontSelectorNeedle', needleGeo, mats.bluedSteel, frontDrumRotor);
  needle.position.set(0, 0.044, -0.022);

  // ---------------------------------------------------------------------------
  // 9. Left & Right Micrometer Focus Assemblies (Ceramic Sleeve + Vernier + Helical Screw + 4-Sided Knurl)
  // ---------------------------------------------------------------------------
  const sideFlangeGeo = trackGeo(createBeveledCylinderGeometry(0.052, 0.012, 0.004, 28));
  const sideCeramicGeo = trackGeo(createRibbedCeramicInsulatorGeometry(0.040, 0.009, 26));
  const sideVernierGeo = trackGeo(createBeveledCylinderGeometry(0.036, 0.018, 0.004, 32));
  const sideLeadScrewGeo = trackGeo(createHelicalLeadScrewGeometry(0.044, 0.0145, 0.0195, 8, 26, 48));
  const sideDrumGeo = trackGeo(createDiamondKnurledDrumGeometry(0.058, 0.038, 36, 8));

  const buildSideThumbscrew = (side: -1 | 1): THREE.Group => {
    const tag = side > 0 ? 'R' : 'L';
    const root = new THREE.Group();
    root.name = `SideFocusThumbscrew_${tag}`;
    root.position.set(side * 0.992, barrelY, 0);
    group.add(root);

    const flange = addPart(`SideCollar_${tag}`, sideFlangeGeo, mats.satinBrass, root);
    flange.rotation.z = Math.PI * 0.5;
    flange.position.set(side * 0.010, 0, 0);

    const cer = addPart(`SideCeramicRing_${tag}`, sideCeramicGeo, ceramicMat, root);
    cer.rotation.z = Math.PI * 0.5;
    cer.position.set(side * 0.029, 0, 0);

    const vern = addPart(`SideVernierSleeve_${tag}`, sideVernierGeo, vernierMat, root);
    vern.rotation.z = Math.PI * 0.5;
    vern.position.set(side * 0.052, 0, 0);

    const rotor = new THREE.Group();
    rotor.position.set(side * 0.078, 0, 0);
    root.add(rotor);

    const shaft = addPart(`SideShaft_${tag}`, sideLeadScrewGeo, mats.bluedSteel, rotor);
    shaft.rotation.z = Math.PI * 0.5;
    shaft.position.set(-side * 0.006, 0, 0);

    const drum = addPart(`KnurledDrum_${tag}`, sideDrumGeo, mats.knurledBrass, rotor);
    drum.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
    drum.position.set(side * 0.054, 0, 0);

    const endScrew = addPart(`SideDrumLockScrew_${tag}`, frontCapScrewGeo, mats.bluedSteel, rotor);
    endScrew.rotation.z = side > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
    endScrew.position.set(side * 0.090, 0, 0);

    return rotor;
  };

  const leftDrum = buildSideThumbscrew(-1);
  const rightDrum = buildSideThumbscrew(1);

  // ---------------------------------------------------------------------------
  // 10. Articulated Underside Brass Optical Slit-Jaw Blades + Fixed Outer Dovetail Guides
  //     Strictly zero overhang past BrassEndCap (`|x| <= 0.930`) and zero rail/iris overlap!
  // ---------------------------------------------------------------------------
  const outerTrackGeo = trackGeo(createChamferedBlockGeometry(0.118, 0.014, 0.026, 0.004));
  const jawGeo = trackGeo(createChamferedBlockGeometry(0.335, 0.014, 0.028, 0.0045));
  const jawY = VEIL_Y_TOP + 0.015; // 1.175

  // Fixed machined brass outer slit cheek runners anchoring the sliding shutter leaves
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const tr = addPart(
        `SlitOuterRunner_${sz > 0 ? 'F' : 'R'}_${sx > 0 ? 'R' : 'L'}`,
        outerTrackGeo,
        mats.satinBrass
      );
      tr.position.set(sx * 0.808, jawY + 0.012, sz * 0.078);
    }
  }

  const frontLeftJaw = addPart('BrassSlitJaw_FL', jawGeo, mats.satinBrass);
  const frontRightJaw = addPart('BrassSlitJaw_FR', jawGeo, mats.satinBrass);
  const rearLeftJaw = addPart('BrassSlitJaw_RL', jawGeo, mats.satinBrass);
  const rearRightJaw = addPart('BrassSlitJaw_RR', jawGeo, mats.satinBrass);

  // ---------------------------------------------------------------------------
  // 11. Central Deployable Brass Annular Iris Turret with 4 Blued-Steel Cassegrain Spider Vanes
  //     Retracts 100% cleanly inside CentralCondenserChest in Planar/S-Veil modes!
  // ---------------------------------------------------------------------------
  const coneTurretGroup = new THREE.Group();
  coneTurretGroup.name = 'CentralAnnularIrisTurret';
  coneTurretGroup.position.set(0, VEIL_Y_TOP + 0.088, 0);
  group.add(coneTurretGroup);

  const outerIrisRingGeo = trackGeo(
    createEllipticalTorusGeometry(CONE_RX_TOP + 0.022, CONE_RZ_TOP + 0.022, 0.015, 48, 16)
  );
  addPart('AnnularIrisOuterBezel', outerIrisRingGeo, mats.satinBrass, coneTurretGroup);

  const innerBaffleGeo = trackGeo(
    createTurnedProfileGeometry(
      [
        { r: CONE_RX_TOP - 0.044, y: -0.016 },
        { r: CONE_RX_TOP - 0.036, y: -0.010 },
        { r: CONE_RX_TOP - 0.036, y: 0.012 },
        { r: CONE_RX_TOP - 0.046, y: 0.016 },
      ],
      40,
      1.0,
      CONE_RZ_TOP / CONE_RX_TOP
    )
  );
  addPart('AnnularIrisCenterPlug', innerBaffleGeo, mats.forgedIron, coneTurretGroup);

  // Machined brass central retaining medallion + slotted screw on bottom of center occulting plug
  const plugBossGeo = trackGeo(createBeveledCylinderGeometry(0.046, 0.008, 0.003, 28));
  const plugBoss = addPart('AnnularIrisCenterBrassHub', plugBossGeo, mats.satinBrass, coneTurretGroup);
  plugBoss.position.set(0, -0.013, 0);

  const plugScrew = addPart('AnnularIrisCenterScrew', miniScrewGeo, mats.bluedSteel, coneTurretGroup);
  plugScrew.rotation.x = Math.PI;
  plugScrew.position.set(0, -0.020, 0);

  // 4 razor-thin blued-steel Cassegrain spider vanes anchoring the central occulting baffle to outer bezel
  const vaneXGeo = trackGeo(createChamferedBlockGeometry(0.036, 0.006, 0.0035, 0.001));
  const vaneZGeo = trackGeo(createChamferedBlockGeometry(0.0035, 0.006, 0.032, 0.001));
  for (const s of [-1, 1]) {
    const vx = addPart(`AnnularSpiderVane_X_${s > 0 ? 'R' : 'L'}`, vaneXGeo, mats.bluedSteel, coneTurretGroup);
    vx.position.set(s * (CONE_RX_TOP - 0.006), -0.002, 0);

    const vz = addPart(`AnnularSpiderVane_Z_${s > 0 ? 'F' : 'B'}`, vaneZGeo, mats.bluedSteel, coneTurretGroup);
    vz.position.set(0, -0.002, s * (CONE_RZ_TOP - 0.004));
  }

  // ---------------------------------------------------------------------------
  // 12. Optical Borosilicate Cylindrical Condenser Lens + Continuous Glowing Slit Filaments
  // ---------------------------------------------------------------------------
  const linearFilGeo = trackGeo(
    createChamferedBlockGeometry(PLANAR_HALF_W_TOP * 0.498, 0.009, 0.011, 0.0025)
  );
  const linearFilamentMesh = addPart(
    'LinearSlitFilamentCore_L',
    linearFilGeo,
    mats.linearFilament,
    group,
    false,
    false
  );
  const linearFilamentRight = addPart(
    'LinearSlitFilamentCore_R',
    linearFilGeo,
    mats.linearFilament,
    group,
    false,
    false
  );

  // Curved optical borosilicate/quartz cylindrical condenser lens rod cradling the linear slit throat
  const condenserLensGeo = trackGeo(createBeveledCylinderGeometry(0.022, 0.415, 0.004, 24));
  const condenserLensLeft = addPart(
    'OpticalCondenserLens_L',
    condenserLensGeo,
    glassMat,
    group,
    false,
    false
  );
  condenserLensLeft.rotation.z = Math.PI * 0.5;
  const condenserLensRight = addPart(
    'OpticalCondenserLens_R',
    condenserLensGeo,
    glassMat,
    group,
    false,
    false
  );
  condenserLensRight.rotation.z = Math.PI * 0.5;

  const coneFilGeo = trackGeo(createEllipticalTorusGeometry(CONE_RX_TOP, CONE_RZ_TOP, 0.010, 48, 12));
  const coneFilamentMesh = addPart(
    'AnnularConeFilamentRing',
    coneFilGeo,
    mats.coneFilament,
    coneTurretGroup,
    false,
    false
  );
  coneFilamentMesh.position.set(0, -0.003, 0);

  // Continuous mechanical articulation driven directly by `ApertureModeController` weights
  const updateArticulation = (weights: ModeWeights, drumAngleRad: number): void => {
    const wS = weights.curvedS;
    const wC = weights.cone;

    // Spin all three diamond-knurled brass micrometer focus drums + helical lead screws;
    // ratio (2/3)/0.85 aligns the front pointer needle dead-center on each 120-deg mode detent pin!
    leftDrum.rotation.x = drumAngleRad;
    rightDrum.rotation.x = -drumAngleRad;
    frontDrumRotor.rotation.z = -drumAngleRad * ((2 / 3) / 0.85);

    // Articulated underside brass slit jaws (strictly inside barrel X <= 0.930 & rail Z <= 0.112!):
    // - Planar (wS=0, wC=0): razor-tight parallel slit at z = +-0.052, x = +-0.340 (inner gap 0.005)
    // - Curved-S (wS=1): anti-symmetric S-cam skew (+-0.018 Z offset + +-0.044 rad yaw, zero rail clip!)
    // - Cone (wC=1): jaws slide laterally outward to x = +-0.595 (inner = 0.260 > 0.247 iris, outer = 0.930 < 0.935 barrel!)
    const jawGapZ = 0.052 * (1 - wC) + 0.042 * wC;
    const jawX = 0.340 + wC * 0.255;
    const sShiftZ = wS * 0.018;
    const sYaw = wS * 0.044;

    frontLeftJaw.position.set(-jawX, jawY, jawGapZ - sShiftZ);
    frontLeftJaw.rotation.y = sYaw;

    rearLeftJaw.position.set(-jawX, jawY, -jawGapZ - sShiftZ);
    rearLeftJaw.rotation.y = sYaw;

    frontRightJaw.position.set(jawX, jawY, jawGapZ + sShiftZ);
    frontRightJaw.rotation.y = -sYaw;

    rearRightJaw.position.set(jawX, jawY, -jawGapZ + sShiftZ);
    rearRightJaw.rotation.y = -sYaw;

    // Keep continuous incandescent filament & quartz condenser lens flush inside articulating jaw throat
    linearFilamentMesh.position.set(-0.427, VEIL_Y_TOP + 0.032, -sShiftZ);
    linearFilamentMesh.rotation.y = sYaw;
    linearFilamentRight.position.set(0.427, VEIL_Y_TOP + 0.032, sShiftZ);
    linearFilamentRight.rotation.y = -sYaw;

    condenserLensLeft.position.set(-0.425, VEIL_Y_TOP + 0.023, -sShiftZ);
    condenserLensLeft.rotation.y = sYaw;
    condenserLensRight.position.set(0.425, VEIL_Y_TOP + 0.023, sShiftZ);
    condenserLensRight.rotation.y = -sYaw;
    const linearLensVis = 1 - wC > 0.04;
    condenserLensLeft.visible = linearLensVis;
    condenserLensRight.visible = linearLensVis;

    // Deploy central brass annular iris collar smoothly downward in Cone mode;
    // when retracted (wC = 0), scale 0.78 & y = VEIL_Y_TOP + 0.088 hides 100% inside CentralCondenserChest!
    coneTurretGroup.position.y = VEIL_Y_TOP + 0.088 * (1 - wC) + 0.012 * wC;
    const irisScale = 0.78 + 0.22 * wC;
    coneTurretGroup.scale.set(irisScale, 1, irisScale);
    coneTurretGroup.rotation.y = (1 - wC) * 0.22;
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
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, -0.15, 0), 2.8);
  return geo;
}

/**
 * Builds twin grazing cross-section edge ribbons at `u = 0.045` and `u = 0.955` spanning the
 * physical slit beam thickness (`aStratum in [-1, +1]`), giving the flat/curved light sheet
 * a razor-clean luminous wedge cross-section even when orbited to an exact 90 deg side profile.
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
