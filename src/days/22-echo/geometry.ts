import * as THREE from 'three/webgpu';
import { APERTURE_HALF, CUBE_OUTER_HALF, GLASS_PANE_HALF } from './optics';

/** Exact hardware clearance & seating constants exported for regression unit testing. */
export const RAIL_OFFSET = 0.668;
export const CORNER_BLOCK_HALF = CUBE_OUTER_HALF - RAIL_OFFSET; // 0.057 -> outer strictly flush at 0.725
export const RAIL_HALF_W = CORNER_BLOCK_HALF - 0.0003; // 0.0567 -> razor-tight machined seam, zero coplanar z-fighting
export const RAIL_HALF_LEN = 0.648; // mortised 37 mm into corner blocks, well inside outer chamfer (0.714)
export const GUSSET_BASE_PLANE = CUBE_OUTER_HALF; // 0.725 -> seats cleanly proud of flush corner blocks
export const GUSSET_FRONT_PROTRUSION = 0.0085;
export const BEZEL_FRONT_Z = 0.630;
export const BEZEL_BACK_Z = GLASS_PANE_HALF - 0.004; // 0.594 -> 4 mm watertight seat behind glass
export const BEZEL_OUTER_HALF = 0.618; // embeds 7 mm solidly inside bronze rail inner wall (0.611)
export const THUMBSCREW_MOUNT_OFFSET = CUBE_OUTER_HALF - 0.004; // 0.721
export const VERNIER_TOP_Y = 0.056;
export const LEAD_SCREW_MIN_Y = 0.024;
export const LEAD_SCREW_MAX_Y = 0.104;
export const MAX_LEAD_SCREW_TRAVEL = 0.008;

export interface EchoMaterialsContract {
  agedBronze: THREE.Material;
  satinBrass: THREE.Material;
  knurledBrass: THREE.Material;
  bluedSteel: THREE.Material;
  infinityMirror: THREE.Material;
  /** Optional specialized calibrated brass for micrometer vernier sleeves (defaults to satinBrass). */
  vernierBrass?: THREE.Material;
  /** Optional specialized stepped inner retaining bezel brass (defaults to satinBrass). */
  bezelBrass?: THREE.Material;
}

export interface ThumbscrewHandle {
  axis: 'x' | 'y';
  sign: 1 | -1;
  group: THREE.Group;
  rotorGroup: THREE.Group;
  hitMesh: THREE.Mesh;
}

export interface EchoCubeAssembly {
  group: THREE.Group;
  glassMesh: THREE.Mesh;
  frameHitMeshes: THREE.Mesh[];
  thumbscrewHitMeshes: THREE.Mesh[];
  thumbscrews: ThumbscrewHandle[];
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

function pushTriangle(
  b: MeshBuilder,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  n: THREE.Vector3,
  kv0 = 1,
  kv1 = 1,
  kv2 = 1
): void {
  b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  b.normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z);
  b.uvs.push(0, 0, 1, 0, 0, 1);
  b.knurlValleys.push(kv0, kv1, kv2);
}

/**
 * Emits a convex polygon sorted strictly counter-clockwise around `outwardNormal`,
 * guaranteeing 100% outward-facing winding (`inwardTri === 0`) and crisp flat machined normals.
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
    .sort((a, bItem) => a.ang - bItem.ang);

  for (let i = 1; i < sorted.length - 1; i++) {
    pushTriangle(b, sorted[0].v, sorted[i].v, sorted[i + 1].v, n);
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
 * Builds a watertight 26-face chamfered rectangular block (6 faces + 12 edge bevels + 8 corner bevels)
 * with exact outward-wound triangles and crisp machined bevel highlights.
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
 * Builds a watertight uniformly-beveled triangular satin-brass corner gusset plate in XY
 * spanning `[0..armLen]` along +X and +Y, with back face embedded `-0.0025` into the bronze face
 * and uniform 45° beveled plateau rising cleanly proud to `+GUSSET_FRONT_PROTRUSION`.
 */
export function createCornerGussetGeometry(
  armLen: number,
  frontSkirtZ = 0.0055,
  tipChamfer = 0.02
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const c = tipChamfer;
  const backZ = -0.0025;
  const plateauZ = GUSSET_FRONT_PROTRUSION;
  const inset = 0.0042;
  const diagInset = inset * Math.SQRT2;

  const pts2d: Array<[number, number]> = [
    [0, 0],
    [armLen, 0],
    [armLen, c],
    [c, armLen],
    [0, armLen],
  ];

  // Uniform perpendicular inward offset of every edge -> razor-even machined 45° perimeter chamfer
  const inset2d: Array<[number, number]> = [
    [inset, inset],
    [armLen - inset, inset],
    [armLen - inset, Math.max(inset + 0.002, c + inset - diagInset)],
    [Math.max(inset + 0.002, c + inset - diagInset), armLen - inset],
    [inset, armLen - inset],
  ];

  const backVerts = pts2d.map(([x, y]) => new THREE.Vector3(x, y, backZ));
  const frontVerts = pts2d.map(([x, y]) => new THREE.Vector3(x, y, frontSkirtZ));
  const bevelFront = inset2d.map(([x, y]) => new THREE.Vector3(x, y, plateauZ));

  pushConvexFace(b, bevelFront, new THREE.Vector3(0, 0, 1));
  pushConvexFace(b, backVerts, new THREE.Vector3(0, 0, -1));

  const nPts = pts2d.length;
  for (let i = 0; i < nPts; i++) {
    const j = (i + 1) % nPts;
    const edgeX = pts2d[j][0] - pts2d[i][0];
    const edgeY = pts2d[j][1] - pts2d[i][1];
    const out2d = new THREE.Vector3(edgeY, -edgeX, 0).normalize();

    pushConvexFace(b, [backVerts[i], backVerts[j], frontVerts[j], frontVerts[i]], out2d);

    const bevelNorm = out2d.clone().add(new THREE.Vector3(0, 0, 0.85)).normalize();
    pushConvexFace(b, [frontVerts[i], frontVerts[j], bevelFront[j], bevelFront[i]], bevelNorm);
  }

  return finalizeBuilder(b);
}

/**
 * Builds a watertight stepped mitered picture-frame brass retaining bezel in XY
 * framing clear window aperture `[-innerHalf, +innerHalf]` with a crisp two-tier machined step:
 * front satin-brass flange (`zFront`), 45° inner lip chamfer, and vertical rabbet wall down to `zBack`.
 */
export function createMiteredBezelFrameGeometry(
  innerHalf: number,
  outerHalf: number,
  zFront: number,
  zBack: number,
  innerStep = 0.014
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const midInner = innerHalf + innerStep;
  const zChamferBot = zFront - innerStep * 1.15;

  const bars: Array<{
    fOutA: [number, number];
    fOutB: [number, number];
    fInA: [number, number];
    fInB: [number, number];
    mInA: [number, number];
    mInB: [number, number];
    bOutA: [number, number];
    bOutB: [number, number];
    bInA: [number, number];
    bInB: [number, number];
    outDir: [number, number];
    inDir: [number, number];
  }> = [
    // +Y top bar
    {
      fOutA: [-outerHalf, outerHalf],
      fOutB: [outerHalf, outerHalf],
      fInA: [-midInner, midInner],
      fInB: [midInner, midInner],
      mInA: [-innerHalf, innerHalf],
      mInB: [innerHalf, innerHalf],
      bOutA: [-outerHalf, outerHalf],
      bOutB: [outerHalf, outerHalf],
      bInA: [-innerHalf, innerHalf],
      bInB: [innerHalf, innerHalf],
      outDir: [0, 1],
      inDir: [0, -1],
    },
    // -Y bottom bar
    {
      fOutA: [outerHalf, -outerHalf],
      fOutB: [-outerHalf, -outerHalf],
      fInA: [midInner, -midInner],
      fInB: [-midInner, -midInner],
      mInA: [innerHalf, -innerHalf],
      mInB: [-innerHalf, -innerHalf],
      bOutA: [outerHalf, -outerHalf],
      bOutB: [-outerHalf, -outerHalf],
      bInA: [innerHalf, -innerHalf],
      bInB: [-innerHalf, -innerHalf],
      outDir: [0, -1],
      inDir: [0, 1],
    },
    // +X right bar
    {
      fOutA: [outerHalf, outerHalf],
      fOutB: [outerHalf, -outerHalf],
      fInA: [midInner, midInner],
      fInB: [midInner, -midInner],
      mInA: [innerHalf, innerHalf],
      mInB: [innerHalf, -innerHalf],
      bOutA: [outerHalf, outerHalf],
      bOutB: [outerHalf, -outerHalf],
      bInA: [innerHalf, innerHalf],
      bInB: [innerHalf, -innerHalf],
      outDir: [1, 0],
      inDir: [-1, 0],
    },
    // -X left bar
    {
      fOutA: [-outerHalf, -outerHalf],
      fOutB: [-outerHalf, outerHalf],
      fInA: [-midInner, -midInner],
      fInB: [-midInner, midInner],
      mInA: [-innerHalf, -innerHalf],
      mInB: [-innerHalf, innerHalf],
      bOutA: [-outerHalf, -outerHalf],
      bOutB: [-outerHalf, outerHalf],
      bInA: [-innerHalf, -innerHalf],
      bInB: [-innerHalf, innerHalf],
      outDir: [-1, 0],
      inDir: [1, 0],
    },
  ];

  for (const bar of bars) {
    const fOA = new THREE.Vector3(bar.fOutA[0], bar.fOutA[1], zFront);
    const fOB = new THREE.Vector3(bar.fOutB[0], bar.fOutB[1], zFront);
    const fIA = new THREE.Vector3(bar.fInA[0], bar.fInA[1], zFront);
    const fIB = new THREE.Vector3(bar.fInB[0], bar.fInB[1], zFront);

    const mIA = new THREE.Vector3(bar.mInA[0], bar.mInA[1], zChamferBot);
    const mIB = new THREE.Vector3(bar.mInB[0], bar.mInB[1], zChamferBot);

    const bOA = new THREE.Vector3(bar.bOutA[0], bar.bOutA[1], zBack);
    const bOB = new THREE.Vector3(bar.bOutB[0], bar.bOutB[1], zBack);
    const bIA = new THREE.Vector3(bar.bInA[0], bar.bInA[1], zBack);
    const bIB = new THREE.Vector3(bar.bInB[0], bar.bInB[1], zBack);

    // Front flat satin-brass retaining face (+Z)
    pushConvexFace(b, [fOA, fOB, fIB, fIA], new THREE.Vector3(0, 0, 1));
    // Back face (-Z)
    pushConvexFace(b, [bOA, bOB, bIB, bIA], new THREE.Vector3(0, 0, -1));
    // Outer perimeter wall (embedded inside bronze rails)
    pushConvexFace(b, [bOA, bOB, fOB, fOA], new THREE.Vector3(bar.outDir[0], bar.outDir[1], 0));
    // Crisp 45-deg optical lip chamfer
    const slopeNorm = new THREE.Vector3(
      bar.inDir[0] * (zFront - zChamferBot),
      bar.inDir[1] * (zFront - zChamferBot),
      innerStep
    ).normalize();
    pushConvexFace(b, [fIA, fIB, mIB, mIA], slopeNorm);
    // Short vertical inner rabbet wall seating directly against the glass perimeter
    pushConvexFace(b, [mIA, mIB, bIB, bIA], new THREE.Vector3(bar.inDir[0], bar.inDir[1], 0));
  }

  return finalizeBuilder(b);
}

/**
 * Builds a watertight, outward-wound 3D cross-hatched diamond-knurled satin-brass micrometer drum
 * along +Y (`y in [0, height]`) using genuine 4-sided raised diamond pyramids (`4` flat triangles per
 * diamond cell -> perfectly symmetric ±45° diamond lattice with zero one-way diagonal rib bias),
 * chamfered brass collars, recessed machined top dimple, and `aKnurlValley` vertex AO attribute.
 */
export function createKnurledDrumGeometry(
  baseRadius = 0.051,
  height = 0.064,
  teeth = 32,
  axialBands = 9
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const chamfer = 0.007;
  const collarPad = 0.0025;
  const knurlStart = chamfer + collarPad;
  const knurlEnd = height - chamfer - collarPad;
  const knurlAmp = 0.0023;
  const valleyR = baseRadius - knurlAmp * 0.72;
  const peakR = baseRadius + knurlAmp;
  const dPhi = (Math.PI * 2) / teeth;

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
    pushTriangle(b, p00, p10, p11, faceN, kv00, kv10, kv11);

    ab.subVectors(p11, p00);
    ac.subVectors(p01, p00);
    faceN.crossVectors(ab, ac).normalize();
    pushTriangle(b, p00, p11, p01, faceN, kv00, kv11, kv01);
  };

  const makeRing = (y: number, radius: number): THREE.Vector3[] => {
    const pts: THREE.Vector3[] = [];
    for (let j = 0; j < teeth; j++) {
      const phi = j * dPhi;
      pts.push(new THREE.Vector3(Math.cos(phi) * radius, y, -Math.sin(phi) * radius));
    }
    return pts;
  };

  // Smooth chamfered bottom collar & smooth top collar rings
  const ringBot0 = makeRing(0, baseRadius * 0.85);
  const ringBot1 = makeRing(chamfer, baseRadius);
  const ringKnurlBot = makeRing(knurlStart, valleyR);
  const ringKnurlTop = makeRing(knurlEnd, valleyR);
  const ringTop1 = makeRing(height - chamfer, baseRadius);
  const ringTop0 = makeRing(height, baseRadius * 0.84);
  const ringDimpleRim = makeRing(height, baseRadius * 0.64);

  for (let j = 0; j < teeth; j++) {
    const jNext = (j + 1) % teeth;
    emitQuad(ringBot0[j], ringBot0[jNext], ringBot1[jNext], ringBot1[j], 0.85, 0.85, 0.96, 0.96);
    emitQuad(
      ringBot1[j],
      ringBot1[jNext],
      ringKnurlBot[jNext],
      ringKnurlBot[j],
      0.96,
      0.96,
      0.35,
      0.35
    );
    emitQuad(
      ringKnurlTop[j],
      ringKnurlTop[jNext],
      ringTop1[jNext],
      ringTop1[j],
      0.35,
      0.35,
      0.96,
      0.96
    );
    emitQuad(ringTop1[j], ringTop1[jNext], ringTop0[jNext], ringTop0[j], 0.96, 0.96, 0.92, 0.92);
    // Flat machined top rim annulus
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

  // True 4-sided diamond pyramid bands (`axialBands` × `teeth` watertight pyramids)
  const bandRows: THREE.Vector3[][] = [ringKnurlBot];
  for (let r = 1; r < axialBands; r++) {
    const y = knurlStart + ((knurlEnd - knurlStart) * r) / axialBands;
    bandRows.push(makeRing(y, valleyR));
  }
  bandRows.push(ringKnurlTop);

  for (let r = 0; r < axialBands; r++) {
    const row0 = bandRows[r];
    const row1 = bandRows[r + 1];
    const yMid = knurlStart + ((knurlEnd - knurlStart) * (r + 0.5)) / axialBands;
    const edgeScale = r === 0 || r === axialBands - 1 ? 0.82 : 1.0;
    const apexR = valleyR + (peakR - valleyR) * edgeScale;

    for (let j = 0; j < teeth; j++) {
      const jNext = (j + 1) % teeth;
      const p00 = row0[j];
      const p10 = row0[jNext];
      const p11 = row1[jNext];
      const p01 = row1[j];
      const phiMid = (j + 0.5) * dPhi;
      const pMid = new THREE.Vector3(Math.cos(phiMid) * apexR, yMid, -Math.sin(phiMid) * apexR);

      // 4 outward-wound CCW flat facets meeting at the raised diamond crest
      const kvVal = 0.08;
      const kvTop = 1.0;

      // Bottom facet
      ab.subVectors(p10, p00);
      ac.subVectors(pMid, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p00, p10, pMid, faceN, kvVal, kvVal, kvTop);

      // Right facet
      ab.subVectors(p11, p10);
      ac.subVectors(pMid, p10);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p10, p11, pMid, faceN, kvVal, kvVal, kvTop);

      // Top facet
      ab.subVectors(p01, p11);
      ac.subVectors(pMid, p11);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p11, p01, pMid, faceN, kvVal, kvVal, kvTop);

      // Left facet
      ab.subVectors(p00, p01);
      ac.subVectors(pMid, p01);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p01, p00, pMid, faceN, kvVal, kvVal, kvTop);
    }
  }

  // Bottom cap (-Y outward)
  const bottomCenter = new THREE.Vector3(0, 0, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  for (let j = 0; j < teeth; j++) {
    const jNext = (j + 1) % teeth;
    pushTriangle(b, bottomCenter, ringBot0[jNext], ringBot0[j], downN, 0.82, 0.82, 0.82);
  }

  // Recessed concave machined dimple cap (+Y outward)
  const topCenter = new THREE.Vector3(0, height - 0.0042, 0);
  for (let j = 0; j < teeth; j++) {
    const jNext = (j + 1) % teeth;
    ab.subVectors(ringDimpleRim[j], topCenter);
    ac.subVectors(ringDimpleRim[jNext], topCenter);
    faceN.crossVectors(ab, ac).normalize();
    pushTriangle(b, topCenter, ringDimpleRim[j], ringDimpleRim[jNext], faceN, 0.68, 0.9, 0.9);
  }

  return finalizeBuilder(b, true);
}

/**
 * Builds a genuine watertight single-start helical blued-steel micrometer lead screw along +Y
 * spanning `[LEAD_SCREW_MIN_Y, LEAD_SCREW_MAX_Y]` with 11 crisp V-thread turns that visibly
 * advance through the brass vernier collar when the micrometer thumbscrew rotates.
 */
export function createHelicalLeadScrewGeometry(
  yMin = LEAD_SCREW_MIN_Y,
  yMax = LEAD_SCREW_MAX_Y,
  minorR = 0.015,
  majorR = 0.0195,
  turns = 11,
  radialSegs = 28,
  axialRings = 60
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const threadStart = yMin + 0.005;
  const threadEnd = yMax - 0.005;
  const pitch = (threadEnd - threadStart) / turns;
  const dPhi = (Math.PI * 2) / radialSegs;

  const rings: THREE.Vector3[][] = [];
  for (let r = 0; r <= axialRings; r++) {
    const t = r / axialRings;
    const y = yMin + (yMax - yMin) * t;
    // Smoothly taper thread depth into cylindrical root shanks at both ends
    const endFadeIn = Math.min(1, Math.max(0, (y - threadStart) / (pitch * 0.65)));
    const endFadeOut = Math.min(1, Math.max(0, (threadEnd - y) / (pitch * 0.65)));
    const threadEnv = endFadeIn * endFadeOut;

    const pts: THREE.Vector3[] = [];
    for (let j = 0; j < radialSegs; j++) {
      const phi = j * dPhi;
      const phaseRaw = (y - threadStart) / pitch - phi / (Math.PI * 2);
      const phase = ((phaseRaw % 1) + 1) % 1;
      const vTooth = 1.0 - Math.abs(2.0 * phase - 1.0);
      // Slight flat truncation on crest & root like real ISO/UNF precision micrometer threads
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
      pushTriangle(b, p00, p10, p11, faceN);

      ab.subVectors(p11, p00);
      ac.subVectors(p01, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p00, p11, p01, faceN);
    }
  }

  // Bottom cap (-Y outward)
  const botCenter = new THREE.Vector3(0, yMin, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    pushTriangle(b, botCenter, rings[0][jNext], rings[0][j], downN);
  }

  // Top cap (+Y outward)
  const topCenter = new THREE.Vector3(0, yMax, 0);
  const upN = new THREE.Vector3(0, 1, 0);
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    pushTriangle(b, topCenter, rings[axialRings][j], rings[axialRings][jNext], upN);
  }

  return finalizeBuilder(b);
}

/**
 * Builds a watertight outward-wound Victorian slotted brass fulcrum lock-cap along +Y
 * (`y in [-0.008, 0.024]`) with an actual machined screwdriver slot cut cleanly across its head.
 */
export function createSlottedLockCapGeometry(radius = 0.036): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const segs = 24;
  const yBot = -0.008;
  const yCollar = 0.012;
  const yTop = 0.023;
  const ySlotFloor = 0.0145;
  const rTop = radius * 0.84;
  const slotHalfW = 0.0046;

  const dPhi = (Math.PI * 2) / segs;
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  // Lower cylindrical base collar (yBot -> yCollar)
  const botRing: THREE.Vector3[] = [];
  const colRing: THREE.Vector3[] = [];
  for (let j = 0; j < segs; j++) {
    const phi = j * dPhi;
    botRing.push(new THREE.Vector3(Math.cos(phi) * radius, yBot, -Math.sin(phi) * radius));
    colRing.push(new THREE.Vector3(Math.cos(phi) * radius, yCollar, -Math.sin(phi) * radius));
  }

  const botCenter = new THREE.Vector3(0, yBot, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  const upN = new THREE.Vector3(0, 1, 0);

  for (let j = 0; j < segs; j++) {
    const jNext = (j + 1) % segs;
    // Bottom cap
    pushTriangle(b, botCenter, botRing[jNext], botRing[j], downN);
    // Side cylinder wall
    ab.subVectors(botRing[jNext], botRing[j]);
    ac.subVectors(colRing[jNext], botRing[j]);
    faceN.crossVectors(ab, ac).normalize();
    pushTriangle(b, botRing[j], botRing[jNext], colRing[jNext], faceN);
    pushTriangle(b, botRing[j], colRing[jNext], colRing[j], faceN);
    // Top cap of base collar (at yCollar)
    pushTriangle(b, new THREE.Vector3(0, yCollar, 0), colRing[j], colRing[jNext], upN);
  }

  // Two opposing chamfered D-lobes flanking the machined screwdriver slot (`z >= +slotHalfW` and `z <= -slotHalfW`)
  const arcSegs = 12;
  const theta0 = Math.asin(Math.min(0.95, slotHalfW / rTop));
  const theta1 = Math.PI - theta0;

  const buildLobe = (zSign: 1 | -1) => {
    const basePts: THREE.Vector3[] = [];
    const topPts: THREE.Vector3[] = [];
    for (let i = 0; i <= arcSegs; i++) {
      const th = theta0 + ((theta1 - theta0) * i) / arcSegs;
      const x = Math.cos(th) * rTop;
      const z = zSign * Math.sin(th) * rTop;
      basePts.push(new THREE.Vector3(x, yCollar - 0.001, z));
      topPts.push(
        new THREE.Vector3(
          x * 0.91,
          yTop,
          zSign * Math.max(slotHalfW, Math.abs(z) * 0.91)
        )
      );
    }

    // Build flat top & bottom lobe faces via pushConvexFace (strictly outward!)
    pushConvexFace(b, topPts, upN);
    pushConvexFace(b, basePts, downN);

    // Curved chamfered perimeter wall
    for (let i = 0; i < arcSegs; i++) {
      const b0 = basePts[i];
      const b1 = basePts[i + 1];
      const t1 = topPts[i + 1];
      const t0 = topPts[i];
      const midX = (b0.x + b1.x) * 0.5;
      const midZ = (b0.z + b1.z) * 0.5;
      const outN = new THREE.Vector3(midX, (rTop - rTop * 0.91) * 2.2, midZ).normalize();
      pushConvexFace(b, [b0, b1, t1, t0], outN);
    }

    // Inner vertical screwdriver slot wall facing `z = 0` (`normal = (0, 0, -zSign)`)
    pushConvexFace(
      b,
      [basePts[0], basePts[arcSegs], topPts[arcSegs], topPts[0]],
      new THREE.Vector3(0, 0, -zSign)
    );
  };

  buildLobe(1);
  buildLobe(-1);

  // Solid recessed slot floor bar (`y in [yCollar - 0.001, ySlotFloor]`) so slot bottom has crisp depth
  const slotHalfLen = Math.sqrt(Math.max(1e-6, rTop * rTop - slotHalfW * slotHalfW)) * 0.98;
  const f0 = new THREE.Vector3(-slotHalfLen, ySlotFloor, -slotHalfW);
  const f1 = new THREE.Vector3(slotHalfLen, ySlotFloor, -slotHalfW);
  const f2 = new THREE.Vector3(slotHalfLen, ySlotFloor, slotHalfW);
  const f3 = new THREE.Vector3(-slotHalfLen, ySlotFloor, slotHalfW);
  const g0 = new THREE.Vector3(-slotHalfLen, yCollar - 0.001, -slotHalfW);
  const g1 = new THREE.Vector3(slotHalfLen, yCollar - 0.001, -slotHalfW);
  const g2 = new THREE.Vector3(slotHalfLen, yCollar - 0.001, slotHalfW);
  const g3 = new THREE.Vector3(-slotHalfLen, yCollar - 0.001, slotHalfW);
  pushConvexFace(b, [f0, f1, f2, f3], upN);
  pushConvexFace(b, [g0, g1, g2, g3], downN);
  pushConvexFace(b, [g0, g3, f3, f0], new THREE.Vector3(-1, 0, 0));
  pushConvexFace(b, [g1, g2, f2, f1], new THREE.Vector3(1, 0, 0));
  pushConvexFace(b, [g0, g1, f1, f0], new THREE.Vector3(0, 0, -1));
  pushConvexFace(b, [g2, g3, f3, f2], new THREE.Vector3(0, 0, 1));

  return finalizeBuilder(b);
}

/**
 * Helper to create a watertight outward-wound LatheGeometry along +Y from bottom-to-top profile.
 */
function createOutwardLathe(profile: Array<[number, number]>, segments = 32): THREE.BufferGeometry {
  const pts = profile.map(([r, y]) => new THREE.Vector2(r, y));
  const geo = new THREE.LatheGeometry(pts, segments);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds the complete museum-grade Day 22 - Echo Brass-Framed Two-Way Infinity Mirror Cube:
 * - 12 chamfered dark-bronze outer structural rails (~1.45 units wide) mortised solidly into
 *   8 flush-outer chamfered dark-bronze corner blocks (`max === CUBE_OUTER_HALF = 0.725`)
 * - 24 cleanly proud satin-brass beveled corner gusset plates with 72 machined brass dome bolts
 * - 6 stepped satin-brass recessed mirror retaining bezels seating watertight over the glass panes
 * - 6 precision 4-sided diamond-knurled brass micrometer thumbscrews with engraved vernier sleeves
 *   and genuine helical blued-steel lead screws deeply seated across full ±3.2° axial travel
 * - 2 opposing slotted brass fulcrum lock-caps on the -Y underside rails
 * - 6 hermetically recessed semi-silvered infinity mirror glass panes
 */
export function buildEchoCubeAssembly(materials: EchoMaterialsContract): EchoCubeAssembly {
  const group = new THREE.Group();
  group.name = 'InfinityMirrorCubeAssembly';

  const geometries: THREE.BufferGeometry[] = [];
  const frameHitMeshes: THREE.Mesh[] = [];
  const thumbscrewHitMeshes: THREE.Mesh[] = [];
  const thumbscrews: ThumbscrewHandle[] = [];

  const vernierMat = materials.vernierBrass ?? materials.satinBrass;
  const bezelMat = materials.bezelBrass ?? materials.satinBrass;

  // 1. 12 Chamfered Dark-Bronze Frame Rails (recessed 0.8 mm inside flush corner lugs -> zero coplanar z-fighting)
  const railGeoX = createChamferedBlockGeometry(RAIL_HALF_LEN, RAIL_HALF_W, RAIL_HALF_W, 0.011);
  const railGeoY = createChamferedBlockGeometry(RAIL_HALF_W, RAIL_HALF_LEN, RAIL_HALF_W, 0.011);
  const railGeoZ = createChamferedBlockGeometry(RAIL_HALF_W, RAIL_HALF_W, RAIL_HALF_LEN, 0.011);
  geometries.push(railGeoX, railGeoY, railGeoZ);

  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      const mx = new THREE.Mesh(railGeoX, materials.agedBronze);
      mx.name = `BronzeRailX_${s1}_${s2}`;
      mx.position.set(0, s1 * RAIL_OFFSET, s2 * RAIL_OFFSET);
      mx.castShadow = true;
      mx.receiveShadow = true;
      group.add(mx);
      frameHitMeshes.push(mx);

      const my = new THREE.Mesh(railGeoY, materials.agedBronze);
      my.name = `BronzeRailY_${s1}_${s2}`;
      my.position.set(s1 * RAIL_OFFSET, 0, s2 * RAIL_OFFSET);
      my.castShadow = true;
      my.receiveShadow = true;
      group.add(my);
      frameHitMeshes.push(my);

      const mz = new THREE.Mesh(railGeoZ, materials.agedBronze);
      mz.name = `BronzeRailZ_${s1}_${s2}`;
      mz.position.set(s1 * RAIL_OFFSET, s2 * RAIL_OFFSET, 0);
      mz.castShadow = true;
      mz.receiveShadow = true;
      group.add(mz);
      frameHitMeshes.push(mz);
    }
  }

  // 2. 8 Flush Dark-Bronze Corner Blocks (`RAIL_OFFSET + CORNER_BLOCK_HALF === CUBE_OUTER_HALF = 0.725`)
  //    + 24 Cleanly Proud Satin-Brass Gusset Plates & 3 Dome Bolts each
  const cornerGeo = createChamferedBlockGeometry(
    CORNER_BLOCK_HALF,
    CORNER_BLOCK_HALF,
    CORNER_BLOCK_HALF,
    0.011
  );
  const gussetGeo = createCornerGussetGeometry(0.144, 0.0055, 0.02);
  const boltGeo = createOutwardLathe(
    [
      [0.0005, 0.0],
      [0.0115, 0.0],
      [0.0115, 0.0045],
      [0.009, 0.0085],
      [0.0042, 0.0102],
      [0.0005, 0.0106],
    ],
    14
  );
  geometries.push(cornerGeo, gussetGeo, boltGeo);

  // Corner vertex tucked 1 mm inside the 0.714 outer chamfer shoulder -> 10 mm clean bronze chamfer
  // separates all 3 gussets at each corner with zero gusset-to-gusset clipping!
  const cPos = CUBE_OUTER_HALF - 0.012; // 0.713

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const cornerMesh = new THREE.Mesh(cornerGeo, materials.agedBronze);
        cornerMesh.name = `BronzeCorner_${sx}_${sy}_${sz}`;
        cornerMesh.position.set(sx * RAIL_OFFSET, sy * RAIL_OFFSET, sz * RAIL_OFFSET);
        cornerMesh.castShadow = true;
        cornerMesh.receiveShadow = true;
        group.add(cornerMesh);
        frameHitMeshes.push(cornerMesh);

        const mountGusset = (
          name: string,
          pos: [number, number, number],
          rotX: number,
          rotY: number,
          rotZ: number
        ) => {
          const gGroup = new THREE.Group();
          gGroup.position.set(pos[0], pos[1], pos[2]);
          gGroup.rotation.set(rotX, rotY, rotZ);

          const plate = new THREE.Mesh(gussetGeo, materials.satinBrass);
          plate.name = name;
          plate.castShadow = true;
          plate.receiveShadow = true;
          gGroup.add(plate);

          // 3 machined brass dome rivets seated cleanly inside the beveled plateau
          const boltCoords: Array<[string, number, number]> = [
            ['Corner', 0.032, 0.032],
            ['ArmA', 0.102, 0.026],
            ['ArmB', 0.026, 0.102],
          ];
          for (const [suffix, bx, by] of boltCoords) {
            const bolt = new THREE.Mesh(boltGeo, materials.satinBrass);
            bolt.name = `${name}_Bolt${suffix}`;
            bolt.position.set(bx, by, 0.0055);
            bolt.rotation.x = Math.PI / 2;
            gGroup.add(bolt);
          }

          group.add(gGroup);
        };

        if (sz === 1) {
          const rotZ =
            sx === -1 && sy === -1
              ? 0
              : sx === 1 && sy === -1
                ? Math.PI / 2
                : sx === 1 && sy === 1
                  ? Math.PI
                  : -Math.PI / 2;
          mountGusset(
            `GussetZpos_${sx}_${sy}`,
            [sx * cPos, sy * cPos, GUSSET_BASE_PLANE],
            0,
            0,
            rotZ
          );
        } else {
          const rotZ =
            sx === 1 && sy === -1
              ? 0
              : sx === -1 && sy === -1
                ? Math.PI / 2
                : sx === -1 && sy === 1
                  ? Math.PI
                  : -Math.PI / 2;
          mountGusset(
            `GussetZneg_${sx}_${sy}`,
            [sx * cPos, sy * cPos, -GUSSET_BASE_PLANE],
            0,
            Math.PI,
            rotZ
          );
        }

        if (sx === 1) {
          const rotZ =
            sz === 1 && sy === -1
              ? 0
              : sz === -1 && sy === -1
                ? Math.PI / 2
                : sz === -1 && sy === 1
                  ? Math.PI
                  : -Math.PI / 2;
          mountGusset(
            `GussetXpos_${sy}_${sz}`,
            [GUSSET_BASE_PLANE, sy * cPos, sz * cPos],
            0,
            Math.PI / 2,
            rotZ
          );
        } else {
          const rotZ =
            sz === -1 && sy === -1
              ? 0
              : sz === 1 && sy === -1
                ? Math.PI / 2
                : sz === 1 && sy === 1
                  ? Math.PI
                  : -Math.PI / 2;
          mountGusset(
            `GussetXneg_${sy}_${sz}`,
            [-GUSSET_BASE_PLANE, sy * cPos, sz * cPos],
            0,
            -Math.PI / 2,
            rotZ
          );
        }

        if (sy === 1) {
          const rotZ =
            sx === -1 && sz === 1
              ? 0
              : sx === 1 && sz === 1
                ? Math.PI / 2
                : sx === 1 && sz === -1
                  ? Math.PI
                  : -Math.PI / 2;
          mountGusset(
            `GussetYpos_${sx}_${sz}`,
            [sx * cPos, GUSSET_BASE_PLANE, sz * cPos],
            -Math.PI / 2,
            0,
            rotZ
          );
        } else {
          const rotZ =
            sx === -1 && sz === -1
              ? 0
              : sx === 1 && sz === -1
                ? Math.PI / 2
                : sx === 1 && sz === 1
                  ? Math.PI
                  : -Math.PI / 2;
          mountGusset(
            `GussetYneg_${sx}_${sz}`,
            [sx * cPos, -GUSSET_BASE_PLANE, sz * cPos],
            Math.PI / 2,
            0,
            rotZ
          );
        }
      }
    }
  }

  // 3. 6 Refined Stepped Satin-Brass Inner Glass Retaining Bezels
  //    Recessed from zFront = 0.630 down to zBack = 0.594 (4 mm overlap onto GLASS_PANE_HALF = 0.598)
  const bezelGeo = createMiteredBezelFrameGeometry(
    APERTURE_HALF,
    BEZEL_OUTER_HALF,
    BEZEL_FRONT_Z,
    BEZEL_BACK_Z,
    0.014
  );
  geometries.push(bezelGeo);

  const bezelOrientations: Array<{ name: string; rx: number; ry: number }> = [
    { name: 'BrassBezelPosZ', rx: 0, ry: 0 },
    { name: 'BrassBezelNegZ', rx: 0, ry: Math.PI },
    { name: 'BrassBezelPosX', rx: 0, ry: Math.PI / 2 },
    { name: 'BrassBezelNegX', rx: 0, ry: -Math.PI / 2 },
    { name: 'BrassBezelPosY', rx: -Math.PI / 2, ry: 0 },
    { name: 'BrassBezelNegY', rx: Math.PI / 2, ry: 0 },
  ];

  for (const bo of bezelOrientations) {
    const bezelMesh = new THREE.Mesh(bezelGeo, bezelMat);
    bezelMesh.name = bo.name;
    bezelMesh.rotation.set(bo.rx, bo.ry, 0);
    bezelMesh.castShadow = true;
    bezelMesh.receiveShadow = true;
    group.add(bezelMesh);
    frameHitMeshes.push(bezelMesh);
  }

  // 4. Precision 4-Sided Diamond-Knurled Brass Micrometer Thumbscrews
  //    Boss radius 0.040 fits strictly inside the 0.045 flat land of the chamfered bronze rail!
  const bossGeo = createOutwardLathe(
    [
      [0.0005, -0.014],
      [0.04, -0.014],
      [0.04, 0.012],
      [0.034, 0.02],
      [0.03, 0.023],
      [0.0005, 0.023],
    ],
    28
  );
  const vernierGeo = createOutwardLathe(
    [
      [0.0005, 0.021],
      [0.0295, 0.021],
      [0.0295, 0.049],
      [0.0245, VERNIER_TOP_Y],
      [0.0005, VERNIER_TOP_Y],
    ],
    32
  );
  // Genuine single-start helical blued-steel lead screw deeply seated down to y = 0.024!
  const threadGeo = createHelicalLeadScrewGeometry(
    LEAD_SCREW_MIN_Y,
    LEAD_SCREW_MAX_Y,
    0.015,
    0.0195,
    11,
    28,
    60
  );

  const shoulderGeo = createOutwardLathe(
    [
      [0.0005, 0.096],
      [0.03, 0.096],
      [0.039, 0.104],
      [0.045, 0.108],
      [0.0005, 0.108],
    ],
    28
  );
  const knurledDrumGeo = createKnurledDrumGeometry(0.051, 0.064, 32, 9);
  const hitCylGeo = new THREE.CylinderGeometry(0.086, 0.086, 0.21, 14);
  hitCylGeo.translate(0, 0.105, 0);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });

  // Genuine machined slotted brass fulcrum lock-caps for the -Y underside rails
  const lockCapGeo = createSlottedLockCapGeometry(0.036);

  geometries.push(
    bossGeo,
    vernierGeo,
    threadGeo,
    shoulderGeo,
    knurledDrumGeo,
    hitCylGeo,
    lockCapGeo
  );

  const mountSpec: Array<{
    name: string;
    axis: 'x' | 'y';
    sign: 1 | -1;
    pos: [number, number, number];
    quat: THREE.Quaternion;
  }> = [
    // +X pair -> controls tiltY (horizontal sweep)
    {
      name: 'ThumbscrewPosX_TopFront',
      axis: 'y',
      sign: 1,
      pos: [THUMBSCREW_MOUNT_OFFSET, RAIL_OFFSET, 0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2),
    },
    {
      name: 'ThumbscrewPosX_BottomBack',
      axis: 'y',
      sign: -1,
      pos: [THUMBSCREW_MOUNT_OFFSET, -RAIL_OFFSET, -0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2),
    },
    // -X pair -> opposing synchronized tiltY adjusters visible on rear orbit
    {
      name: 'ThumbscrewNegX_TopBack',
      axis: 'y',
      sign: -1,
      pos: [-THUMBSCREW_MOUNT_OFFSET, RAIL_OFFSET, -0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2),
    },
    {
      name: 'ThumbscrewNegX_BottomFront',
      axis: 'y',
      sign: 1,
      pos: [-THUMBSCREW_MOUNT_OFFSET, -RAIL_OFFSET, 0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2),
    },
    // +Y pair -> controls tiltX (vertical sweep)
    {
      name: 'ThumbscrewPosY_FrontLeft',
      axis: 'x',
      sign: 1,
      pos: [-0.41, THUMBSCREW_MOUNT_OFFSET, RAIL_OFFSET],
      quat: new THREE.Quaternion(),
    },
    {
      name: 'ThumbscrewPosY_BackRight',
      axis: 'x',
      sign: -1,
      pos: [0.41, THUMBSCREW_MOUNT_OFFSET, -RAIL_OFFSET],
      quat: new THREE.Quaternion(),
    },
  ];

  for (const spec of mountSpec) {
    const tGroup = new THREE.Group();
    tGroup.name = spec.name;
    tGroup.position.set(spec.pos[0], spec.pos[1], spec.pos[2]);
    tGroup.quaternion.copy(spec.quat);

    const boss = new THREE.Mesh(bossGeo, materials.agedBronze);
    boss.name = `${spec.name}_Boss`;
    boss.castShadow = true;
    boss.receiveShadow = true;
    tGroup.add(boss);

    const vernier = new THREE.Mesh(vernierGeo, vernierMat);
    vernier.name = `${spec.name}_Vernier`;
    vernier.castShadow = true;
    vernier.receiveShadow = true;
    tGroup.add(vernier);

    const rotorGroup = new THREE.Group();
    rotorGroup.name = `${spec.name}_Rotor`;

    const thread = new THREE.Mesh(threadGeo, materials.bluedSteel);
    thread.name = `${spec.name}_LeadScrew`;
    thread.castShadow = true;
    rotorGroup.add(thread);

    const shoulder = new THREE.Mesh(shoulderGeo, vernierMat);
    shoulder.name = `${spec.name}_Shoulder`;
    shoulder.castShadow = true;
    rotorGroup.add(shoulder);

    const drum = new THREE.Mesh(knurledDrumGeo, materials.knurledBrass);
    drum.name = `${spec.name}_KnurledDrum`;
    drum.position.y = 0.106;
    drum.castShadow = true;
    drum.receiveShadow = true;
    rotorGroup.add(drum);

    tGroup.add(rotorGroup);

    const hitMesh = new THREE.Mesh(hitCylGeo, hitMat);
    hitMesh.name = `${spec.name}_Hit`;
    hitMesh.userData.thumbAxis = spec.axis;
    tGroup.add(hitMesh);

    group.add(tGroup);
    thumbscrewHitMeshes.push(hitMesh);
    thumbscrews.push({
      axis: spec.axis,
      sign: spec.sign,
      group: tGroup,
      rotorGroup,
      hitMesh,
    });
  }

  // 2 opposing slotted brass fulcrum lock-caps on the -Y underside rails
  const lockPositions: Array<[number, number, number]> = [
    [0.41, -THUMBSCREW_MOUNT_OFFSET, RAIL_OFFSET],
    [-0.41, -THUMBSCREW_MOUNT_OFFSET, -RAIL_OFFSET],
  ];
  lockPositions.forEach((lp, idx) => {
    const cap = new THREE.Mesh(lockCapGeo, materials.satinBrass);
    cap.name = `FulcrumLockCap_${idx}`;
    cap.position.set(lp[0], lp[1], lp[2]);
    cap.rotation.x = Math.PI;
    cap.rotation.y = idx === 0 ? 0.42 : -0.35;
    cap.castShadow = true;
    cap.receiveShadow = true;
    group.add(cap);
  });

  // 5. 6 Semi-Silvered Two-Way Infinity Mirror Glass Panes (Hermetically Seated Behind Brass Bezels)
  const glassGeo = new THREE.BoxGeometry(
    GLASS_PANE_HALF * 2,
    GLASS_PANE_HALF * 2,
    GLASS_PANE_HALF * 2
  );
  geometries.push(glassGeo);

  const glassMesh = new THREE.Mesh(glassGeo, materials.infinityMirror);
  glassMesh.name = 'InfinityMirrorGlassPanes';
  glassMesh.castShadow = false;
  glassMesh.receiveShadow = false;
  group.add(glassMesh);

  return {
    group,
    glassMesh,
    frameHitMeshes,
    thumbscrewHitMeshes,
    thumbscrews,
    dispose: () => {
      for (const g of geometries) g.dispose();
      hitMat.dispose();
    },
  };
}
