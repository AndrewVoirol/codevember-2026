import * as THREE from 'three/webgpu';
import {
  APERTURE_HALF,
  BULB_BOT_Y,
  BULB_MAX_R,
  BULB_TOP_Y,
  CEILING_ROSETTE_BOT_Y,
  CEILING_ROSETTE_RADIUS,
  CEILING_ROSETTE_TOP_Y,
  CORD_GRIP_BOT_Y,
  CORD_GRIP_TOP_Y,
  CORD_RADIUS,
  CORD_SOCKET_ATTACH_Y,
  CUBE_OUTER_HALF,
  E27_BASE_BOT_Y,
  E27_BASE_TOP_Y,
  FILAMENT_BOT_R,
  FILAMENT_BOT_Y,
  FILAMENT_LOOPS,
  FILAMENT_TOP_R,
  FILAMENT_TOP_Y,
  FINIAL_TIP_Y,
  FINIAL_TOP_Y,
  GLAND_BOT_Y,
  GLAND_TOP_Y,
  GLASS_PANE_HALF,
  GROMMET_BOT_Y,
  GROMMET_TOP_Y,
  PRISM_APEX_Y,
  PRISM_EQUATOR_R,
  PRISM_TRUNC_Y,
  SOCKET_CAP_BOT_Y,
  SOCKET_INSULATOR_BOT_Y,
  SOCKET_KNURL_BOT_Y,
  SOCKET_SKIRT_BOT_Y,
  SOCKET_TOP_Y,
  SOCKET_VENT_BOT_Y,
} from './optics';
import { CORE_ANCHOR_Y, CORE_REST_Y } from './physics';

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
  /** Optional FrontSide transparent exterior Fresnel & celadon glass shell (defaults to infinityMirror). */
  infinityMirrorFront?: THREE.Material;
  /** Optional glazed Bone-Calcite ceramic ceiling grommet material. */
  ceramicInsulator?: THREE.Material;
  /** Optional dark porcelain inner Edison socket insulator ring material. */
  darkCeramic?: THREE.Material;
  /** Optional woven dark-umber & ochre textile braided pendant cord material. */
  braidedCord?: THREE.Material;
  /** Optional incandescent squirrel-cage tungsten filament material (reads `aHeat`). */
  edisonFilament?: THREE.Material;
  /** Optional delicate inner amber volumetric glow halo hugging the squirrel-cage. */
  filamentAura?: THREE.Material;
  /** Optional hand-blown ST64 amber glass envelope & borosilicate stem material. */
  amberBulbGlass?: THREE.Material;
  /** Optional crystal-clear beveled optical calcite prism glass material. */
  crystalPrismGlass?: THREE.Material;
  /** Optional warm backlit dark-bronze interior baffle behind the Edison socket vent slots. */
  ventBaffleGlow?: THREE.Material;
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
  glassBackMesh: THREE.Mesh;
  ceilingAnchorGroup: THREE.Group;
  braidedCordMesh: THREE.Mesh;
  lampHeadGroup: THREE.Group;
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
  axialBands = 9,
  knurlAmp = 0.0023,
  chamfer = 0.007
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const collarPad = Math.min(0.0025, height * 0.08);
  const safeChamfer = Math.min(chamfer, height * 0.22);
  const knurlStart = safeChamfer + collarPad;
  const knurlEnd = height - safeChamfer - collarPad;
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
  const ringBot1 = makeRing(safeChamfer, baseRadius);
  const ringKnurlBot = makeRing(knurlStart, valleyR);
  const ringKnurlTop = makeRing(knurlEnd, valleyR);
  const ringTop1 = makeRing(height - safeChamfer, baseRadius);
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
 * Appends a watertight, capped, strictly outward-wound swept tube along `points` into `b`
 * using a twist-free parallel-transport (Bishop) frame `(T, N, B)`.
 * Optionally emits per-vertex scalar attribute values into `extraAttr`.
 */
function appendCappedTube(
  b: MeshBuilder,
  points: readonly THREE.Vector3[],
  radiusFn: (s: number, phi: number) => number,
  radialSegs = 12,
  extraAttr?: { values: number[]; evalFn: (s: number, phi: number) => number }
): void {
  const nPts = points.length;
  if (nPts < 2) return;

  const tangents: THREE.Vector3[] = [];
  for (let i = 0; i < nPts; i++) {
    const t = new THREE.Vector3();
    if (i === 0) {
      t.subVectors(points[1], points[0]);
    } else if (i === nPts - 1) {
      t.subVectors(points[nPts - 1], points[nPts - 2]);
    } else {
      t.subVectors(points[i + 1], points[i - 1]);
    }
    if (t.lengthSq() < 1e-12) t.set(0, 1, 0);
    t.normalize();
    tangents.push(t);
  }

  // Initial right-handed frame (N, B, T) with N × B = T
  const refUp = Math.abs(tangents[0].y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const normals: THREE.Vector3[] = [];
  const binormals: THREE.Vector3[] = [];
  const n0 = new THREE.Vector3().crossVectors(refUp, tangents[0]).normalize();
  const b0 = new THREE.Vector3().crossVectors(tangents[0], n0).normalize();
  normals.push(n0);
  binormals.push(b0);

  const rotAxis = new THREE.Vector3();
  const rotQuat = new THREE.Quaternion();
  for (let i = 1; i < nPts; i++) {
    const prevT = tangents[i - 1];
    const curT = tangents[i];
    const prevN = normals[i - 1].clone();
    rotAxis.crossVectors(prevT, curT);
    const sinLen = rotAxis.length();
    if (sinLen > 1e-6) {
      rotAxis.divideScalar(sinLen);
      const ang = Math.atan2(sinLen, Math.max(-1, Math.min(1, prevT.dot(curT))));
      rotQuat.setFromAxisAngle(rotAxis, ang);
      prevN.applyQuaternion(rotQuat);
    }
    // Re-orthonormalize strictly
    prevN.addScaledVector(curT, -prevN.dot(curT)).normalize();
    const curB = new THREE.Vector3().crossVectors(curT, prevN).normalize();
    normals.push(prevN);
    binormals.push(curB);
  }

  const dPhi = (Math.PI * 2) / radialSegs;
  const rings: THREE.Vector3[][] = [];
  const ringNorms: THREE.Vector3[][] = [];
  const ringExtra: number[][] = [];

  for (let i = 0; i < nPts; i++) {
    const s = i / (nPts - 1);
    const c = points[i];
    const nVec = normals[i];
    const bVec = binormals[i];
    const ring: THREE.Vector3[] = [];
    const rNorms: THREE.Vector3[] = [];
    const rEx: number[] = [];
    for (let j = 0; j < radialSegs; j++) {
      const phi = j * dPhi;
      const cs = Math.cos(phi);
      const sn = Math.sin(phi);
      const radDir = new THREE.Vector3(
        cs * nVec.x + sn * bVec.x,
        cs * nVec.y + sn * bVec.y,
        cs * nVec.z + sn * bVec.z
      ).normalize();
      const r = radiusFn(s, phi);
      ring.push(c.clone().addScaledVector(radDir, r));
      rNorms.push(radDir);
      if (extraAttr) rEx.push(extraAttr.evalFn(s, phi));
    }
    rings.push(ring);
    ringNorms.push(rNorms);
    ringExtra.push(rEx);
  }

  const pushSmoothTri = (
    p0: THREE.Vector3,
    p1: THREE.Vector3,
    p2: THREE.Vector3,
    vn0: THREE.Vector3,
    vn1: THREE.Vector3,
    vn2: THREE.Vector3,
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    u2: number,
    v2: number,
    e0 = 1,
    e1 = 1,
    e2 = 1
  ) => {
    b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    b.normals.push(vn0.x, vn0.y, vn0.z, vn1.x, vn1.y, vn1.z, vn2.x, vn2.y, vn2.z);
    b.uvs.push(u0, v0, u1, v1, u2, v2);
    b.knurlValleys.push(1, 1, 1);
    if (extraAttr) extraAttr.values.push(e0, e1, e2);
  };

  for (let i = 0; i < nPts - 1; i++) {
    const s0 = i / (nPts - 1);
    const s1 = (i + 1) / (nPts - 1);
    for (let j = 0; j < radialSegs; j++) {
      const jNext = (j + 1) % radialSegs;
      const u0 = j / radialSegs;
      const u1 = (j + 1) / radialSegs;
      const e00 = extraAttr ? ringExtra[i][j] : 1;
      const e01 = extraAttr ? ringExtra[i][jNext] : 1;
      const e11 = extraAttr ? ringExtra[i + 1][jNext] : 1;
      const e10 = extraAttr ? ringExtra[i + 1][j] : 1;

      // Outward CCW winding: (+phi) × (+s) = +radialOut
      pushSmoothTri(
        rings[i][j],
        rings[i][jNext],
        rings[i + 1][jNext],
        ringNorms[i][j],
        ringNorms[i][jNext],
        ringNorms[i + 1][jNext],
        u0,
        s0,
        u1,
        s0,
        u1,
        s1,
        e00,
        e01,
        e11
      );
      pushSmoothTri(
        rings[i][j],
        rings[i + 1][jNext],
        rings[i + 1][j],
        ringNorms[i][j],
        ringNorms[i + 1][jNext],
        ringNorms[i + 1][j],
        u0,
        s0,
        u1,
        s1,
        u0,
        s1,
        e00,
        e11,
        e10
      );
    }
  }

  // Start cap (outward normal = -tangents[0])
  const startCapN = tangents[0].clone().negate();
  const startCenter = points[0];
  const startEx = extraAttr ? extraAttr.evalFn(0, 0) : 1;
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    pushSmoothTri(
      startCenter,
      rings[0][jNext],
      rings[0][j],
      startCapN,
      startCapN,
      startCapN,
      0.5,
      0,
      1,
      0,
      0,
      0,
      startEx,
      extraAttr ? ringExtra[0][jNext] : 1,
      extraAttr ? ringExtra[0][j] : 1
    );
  }

  // End cap (outward normal = +tangents[nPts - 1])
  const endCapN = tangents[nPts - 1].clone();
  const endCenter = points[nPts - 1];
  const endEx = extraAttr ? extraAttr.evalFn(1, 0) : 1;
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    pushSmoothTri(
      endCenter,
      rings[nPts - 1][j],
      rings[nPts - 1][jNext],
      endCapN,
      endCapN,
      endCapN,
      0.5,
      1,
      0,
      1,
      1,
      1,
      endEx,
      extraAttr ? ringExtra[nPts - 1][j] : 1,
      extraAttr ? ringExtra[nPts - 1][jNext] : 1
    );
  }
}

/**
 * Builds the articulated braided dark-umber & ochre textile pendant cord in its exact rest pose
 * (`y in [CORE_REST_Y + CORD_SOCKET_ATTACH_Y, CORE_ANCHOR_Y]`) with physical twin-ply helical
 * surface relief and vertex attributes (`aCordS`, `aCordRadial`) for zero-GC GPU cubic Hermite swing!
 */
export function createBraidedCordGeometry(
  radialSegs = 20,
  axialSegs = 34
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const yTop = CORE_ANCHOR_Y;
  const yBot = CORE_REST_Y + CORD_SOCKET_ATTACH_Y;
  const dPhi = (Math.PI * 2) / radialSegs;
  const twists = 16;

  const cordS: number[] = [];
  const cordRadial: number[] = [];

  const rings: Array<Array<{ pos: THREE.Vector3; radVec: THREE.Vector3; s: number; phi: number }>> = [];
  for (let i = 0; i <= axialSegs; i++) {
    const s = i / axialSegs; // 0 = top inside ceiling grommet, 1 = bottom inside socket cord-grip nut
    const y = yTop + (yBot - yTop) * s;
    // Taper braided relief smoothly near ends so cord seats tightly inside both compression collets
    const endSmooth = Math.min(1, s * 12) * Math.min(1, (1 - s) * 12);
    const ring: Array<{ pos: THREE.Vector3; radVec: THREE.Vector3; s: number; phi: number }> = [];
    for (let j = 0; j < radialSegs; j++) {
      const phi = j * dPhi;
      const braidWave =
        0.085 * Math.cos(2 * phi + Math.PI * 2 * twists * s) +
        0.045 * Math.cos(4 * phi - Math.PI * 2 * twists * s);
      const r = CORD_RADIUS * (1 + braidWave * endSmooth);
      const rx = Math.cos(phi) * r;
      const rz = -Math.sin(phi) * r;
      ring.push({
        pos: new THREE.Vector3(rx, y, rz),
        radVec: new THREE.Vector3(rx, 0, rz),
        s,
        phi,
      });
    }
    rings.push(ring);
  }

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  const emitTri = (
    p0: THREE.Vector3,
    p1: THREE.Vector3,
    p2: THREE.Vector3,
    n: THREE.Vector3,
    s0: number,
    s1: number,
    s2: number,
    r0: THREE.Vector3,
    r1: THREE.Vector3,
    r2: THREE.Vector3,
    u0: number,
    u1: number,
    u2: number
  ) => {
    b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    b.normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z);
    b.uvs.push(u0, s0, u1, s1, u2, s2);
    cordS.push(s0, s1, s2);
    cordRadial.push(r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z);
  };

  // Rings descend from yTop (i = 0) down to yBot (i = axialSegs), with phi increasing CCW looking from +Y
  // So outward CCW quad winding is (row[i+1][j], row[i+1][jNext], row[i][jNext], row[i][j])
  for (let i = 0; i < axialSegs; i++) {
    const rowTop = rings[i];
    const rowBot = rings[i + 1];
    for (let j = 0; j < radialSegs; j++) {
      const jNext = (j + 1) % radialSegs;
      const p00 = rowBot[j];
      const p10 = rowBot[jNext];
      const p11 = rowTop[jNext];
      const p01 = rowTop[j];
      const u0 = j / radialSegs;
      const u1 = (j + 1) / radialSegs;

      ab.subVectors(p10.pos, p00.pos);
      ac.subVectors(p11.pos, p00.pos);
      faceN.crossVectors(ab, ac).normalize();
      emitTri(p00.pos, p10.pos, p11.pos, faceN, p00.s, p10.s, p11.s, p00.radVec, p10.radVec, p11.radVec, u0, u1, u1);

      ab.subVectors(p11.pos, p00.pos);
      ac.subVectors(p01.pos, p00.pos);
      faceN.crossVectors(ab, ac).normalize();
      emitTri(p00.pos, p11.pos, p01.pos, faceN, p00.s, p11.s, p01.s, p00.radVec, p11.radVec, p01.radVec, u0, u1, u0);
    }
  }

  // Top cap (+Y outward at s = 0)
  const zeroRad = new THREE.Vector3(0, 0, 0);
  const topCenter = new THREE.Vector3(0, yTop, 0);
  const upN = new THREE.Vector3(0, 1, 0);
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    emitTri(
      topCenter,
      rings[0][j].pos,
      rings[0][jNext].pos,
      upN,
      0,
      0,
      0,
      zeroRad,
      rings[0][j].radVec,
      rings[0][jNext].radVec,
      0.5,
      j / radialSegs,
      (j + 1) / radialSegs
    );
  }

  // Bottom cap (-Y outward at s = 1)
  const botCenter = new THREE.Vector3(0, yBot, 0);
  const downN = new THREE.Vector3(0, -1, 0);
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    emitTri(
      botCenter,
      rings[axialSegs][jNext].pos,
      rings[axialSegs][j].pos,
      downN,
      1,
      1,
      1,
      zeroRad,
      rings[axialSegs][jNext].radVec,
      rings[axialSegs][j].radVec,
      0.5,
      (j + 1) / radialSegs,
      j / radialSegs
    );
  }

  const geo = finalizeBuilder(b);
  geo.setAttribute('aCordS', new THREE.Float32BufferAttribute(cordS, 1));
  geo.setAttribute('aCordRadial', new THREE.Float32BufferAttribute(cordRadial, 3));
  return geo;
}

/**
 * Builds the watertight 8-facet truncated optical crystal prism (`y in [-PRISM_TRUNC_Y, +PRISM_TRUNC_Y]`)
 * with exact outward-wound facets & flat end-caps seated strictly inside the top brass gallery crown
 * and bottom brass finial cup.
 */
export function createTruncatedPrismGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const rEq = PRISM_EQUATOR_R;
  const yTr = PRISM_TRUNC_Y;
  const rTr = rEq * (1 - yTr / PRISM_APEX_Y);

  const topPts = [
    new THREE.Vector3(rTr, yTr, 0),
    new THREE.Vector3(0, yTr, -rTr),
    new THREE.Vector3(-rTr, yTr, 0),
    new THREE.Vector3(0, yTr, rTr),
  ];
  const eqPts = [
    new THREE.Vector3(rEq, 0, 0),
    new THREE.Vector3(0, 0, -rEq),
    new THREE.Vector3(-rEq, 0, 0),
    new THREE.Vector3(0, 0, rEq),
  ];
  const botPts = [
    new THREE.Vector3(rTr, -yTr, 0),
    new THREE.Vector3(0, -yTr, -rTr),
    new THREE.Vector3(-rTr, -yTr, 0),
    new THREE.Vector3(0, -yTr, rTr),
  ];

  // Top & bottom flat truncation caps tucked inside brass gallery & finial cups
  pushConvexFace(b, topPts, new THREE.Vector3(0, 1, 0));
  pushConvexFace(b, botPts, new THREE.Vector3(0, -1, 0));

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const n = new THREE.Vector3();

  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;

    // Upper trapezoidal optical facet
    ab.subVectors(eqPts[j], eqPts[i]);
    ac.subVectors(topPts[i], eqPts[i]);
    n.crossVectors(ab, ac).normalize();
    pushConvexFace(b, [eqPts[i], eqPts[j], topPts[j], topPts[i]], n);

    // Lower trapezoidal optical facet
    ab.subVectors(botPts[j], botPts[i]);
    ac.subVectors(eqPts[i], botPts[i]);
    n.crossVectors(ab, ac).normalize();
    pushConvexFace(b, [botPts[i], botPts[j], eqPts[j], eqPts[i]], n);
  }

  return finalizeBuilder(b);
}

/**
 * Builds the 12 machined satin-brass edge-came prism cage ribs + 4 curved cast-brass spider
 * gallery claws gripping the upper prism facets cleanly into the Edison socket gallery crown.
 */
export function createPrismCageAndClawsGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const rEq = PRISM_EQUATOR_R;
  const yTr = PRISM_TRUNC_Y;
  const rTr = rEq * (1 - yTr / PRISM_APEX_Y);

  const topPts = [
    new THREE.Vector3(rTr, yTr, 0),
    new THREE.Vector3(0, yTr, -rTr),
    new THREE.Vector3(-rTr, yTr, 0),
    new THREE.Vector3(0, yTr, rTr),
  ];
  const eqPts = [
    new THREE.Vector3(rEq, 0, 0),
    new THREE.Vector3(0, 0, -rEq),
    new THREE.Vector3(-rEq, 0, 0),
    new THREE.Vector3(0, 0, rEq),
  ];
  const botPts = [
    new THREE.Vector3(rTr, -yTr, 0),
    new THREE.Vector3(0, -yTr, -rTr),
    new THREE.Vector3(-rTr, -yTr, 0),
    new THREE.Vector3(0, -yTr, rTr),
  ];

  const ribR = 0.0044;
  const addRib = (a: THREE.Vector3, c: THREE.Vector3, r = ribR) => {
    const dir = c.clone().sub(a).normalize();
    // Extend 3 mm past endpoints to embed cleanly inside corner lugs, gallery crown, & finial cup
    const pStart = a.clone().addScaledVector(dir, -0.003);
    const pEnd = c.clone().addScaledVector(dir, 0.003);
    const pMid = a.clone().lerp(c, 0.5);
    appendCappedTube(b, [pStart, pMid, pEnd], () => r, 8);
  };

  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    addRib(topPts[i], eqPts[i]);
    addRib(eqPts[i], eqPts[j]);
    addRib(eqPts[i], botPts[i]);

    // Cast curved brass spider gallery claw sweeping from the socket gallery skirt (`y = 0.178, r = 0.043`)
    // cleanly onto the EXACT mathematical upper prism ridge (`y = 0.098, r = rEq * (1 - 0.098 / PRISM_APEX_Y)`)!
    const radDir = eqPts[i].clone().normalize();
    const clawTipY = 0.098;
    const clawTipR = rEq * (1 - clawTipY / PRISM_APEX_Y); // exact ridge center radius (0.0653)
    const clawMidY = 0.128;
    const clawMidR = rEq * (1 - clawMidY / PRISM_APEX_Y) + 0.0055;
    const clawCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(radDir.x * 0.042, 0.178, radDir.z * 0.042),
      new THREE.Vector3(radDir.x * 0.052, 0.168, radDir.z * 0.052),
      new THREE.Vector3(radDir.x * 0.046, 0.148, radDir.z * 0.046),
      new THREE.Vector3(radDir.x * clawMidR, clawMidY, radDir.z * clawMidR),
      new THREE.Vector3(radDir.x * clawTipR, clawTipY, radDir.z * clawTipR),
    ]);
    const clawPts = clawCurve.getPoints(14);
    appendCappedTube(
      b,
      clawPts,
      (s) => 0.0054 - 0.0012 * s + 0.001 * Math.sin(s * Math.PI),
      8
    );

    // Machined brass gripping ferrule seated dead-flush around the upper prism edge-came rib at `y = 0.098`
    const tip = clawPts[clawPts.length - 1];
    const padDir = eqPts[i].clone().sub(topPts[i]).normalize();
    appendCappedTube(
      b,
      [tip.clone().addScaledVector(padDir, -0.008), tip.clone().addScaledVector(padDir, 0.008)],
      () => 0.0066,
      8
    );
  }

  return finalizeBuilder(b);
}

/**
 * Builds the 12 real 3D chamfered brass heat-ventilation louver struts around the Edison socket upper bell.
 */
export function createSocketVentStrutsGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const count = 12;
  const rInner = 0.0445;
  const rOuter = 0.0502;
  const y0 = SOCKET_VENT_BOT_Y;
  const y1 = SOCKET_CAP_BOT_Y + 0.003;
  const halfAng = (Math.PI * 2) / (count * 4.4); // Leaves crisp ~55% open radial cooling slots!

  for (let i = 0; i < count; i++) {
    const phi = (i * Math.PI * 2) / count;
    const pA = phi - halfAng;
    const pB = phi + halfAng;
    const cA = Math.cos(pA);
    const sA = -Math.sin(pA);
    const cB = Math.cos(pB);
    const sB = -Math.sin(pB);
    const cM = Math.cos(phi);
    const sM = -Math.sin(phi);

    const inA0 = new THREE.Vector3(cA * rInner, y0, sA * rInner);
    const inB0 = new THREE.Vector3(cB * rInner, y0, sB * rInner);
    const outA0 = new THREE.Vector3(cA * rOuter, y0, sA * rOuter);
    const outB0 = new THREE.Vector3(cB * rOuter, y0, sB * rOuter);

    const inA1 = new THREE.Vector3(cA * rInner, y1, sA * rInner);
    const inB1 = new THREE.Vector3(cB * rInner, y1, sB * rInner);
    const outA1 = new THREE.Vector3(cA * rOuter, y1, sA * rOuter);
    const outB1 = new THREE.Vector3(cB * rOuter, y1, sB * rOuter);

    pushConvexFace(b, [outA0, outB0, outB1, outA1], new THREE.Vector3(cM, 0, sM));
    pushConvexFace(b, [inA0, inB0, inB1, inA1], new THREE.Vector3(-cM, 0, -sM));
    pushConvexFace(b, [inA0, outA0, outA1, inA1], new THREE.Vector3(sA, 0, -cA));
    pushConvexFace(b, [inB0, outB0, outB1, inB1], new THREE.Vector3(-sB, 0, cB));
    pushConvexFace(b, [inA1, inB1, outB1, outA1], new THREE.Vector3(0, 1, 0));
    pushConvexFace(b, [inA0, inB0, outB0, outA0], new THREE.Vector3(0, -1, 0));
  }

  return finalizeBuilder(b);
}

/**
 * Builds the 12 delicate radiating molybdenum anchor hooks (6 upper + 6 lower) + 2 curved nickel
 * lead-in power stems emerging from the glass exhaust pinch press inside the ST64 Edison bulb.
 */
export function createMolybdenumHooksGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const dPhi = (Math.PI * 2) / FILAMENT_LOOPS;

  for (let k = 0; k < FILAMENT_LOOPS; k++) {
    const phiTop = k * dPhi;
    const phiBot = (k + 0.5) * dPhi;

    // Upper anchor hook radiating from top glass button (`y = FILAMENT_TOP_Y`) with subtle hooked curl
    const tx = Math.cos(phiTop);
    const tz = -Math.sin(phiTop);
    appendCappedTube(
      b,
      [
        new THREE.Vector3(tx * 0.003, FILAMENT_TOP_Y + 0.002, tz * 0.003),
        new THREE.Vector3(tx * (FILAMENT_TOP_R * 0.58), FILAMENT_TOP_Y + 0.004, tz * (FILAMENT_TOP_R * 0.58)),
        new THREE.Vector3(tx * (FILAMENT_TOP_R + 0.0015), FILAMENT_TOP_Y + 0.0025, tz * (FILAMENT_TOP_R + 0.0015)),
        new THREE.Vector3(tx * (FILAMENT_TOP_R + 0.0035), FILAMENT_TOP_Y - 0.002, tz * (FILAMENT_TOP_R + 0.0035)),
      ],
      () => 0.00135,
      6
    );

    // Lower anchor hook radiating from bottom glass button (`y = FILAMENT_BOT_Y`) with upward catch curl
    const bx = Math.cos(phiBot);
    const bz = -Math.sin(phiBot);
    appendCappedTube(
      b,
      [
        new THREE.Vector3(bx * 0.003, FILAMENT_BOT_Y - 0.001, bz * 0.003),
        new THREE.Vector3(bx * (FILAMENT_BOT_R * 0.58), FILAMENT_BOT_Y - 0.003, bz * (FILAMENT_BOT_R * 0.58)),
        new THREE.Vector3(bx * (FILAMENT_BOT_R + 0.0015), FILAMENT_BOT_Y - 0.002, bz * (FILAMENT_BOT_R + 0.0015)),
        new THREE.Vector3(bx * (FILAMENT_BOT_R + 0.0035), FILAMENT_BOT_Y + 0.0025, bz * (FILAMENT_BOT_R + 0.0035)),
      ],
      () => 0.00135,
      6
    );
  }

  // Two slightly thicker nickel-copper lead-in stems descending from the glass pinch seal into the filament circuit
  for (const side of [-1, 1]) {
    const phi = side > 0 ? 0.14 : Math.PI - 0.14;
    const cx = Math.cos(phi);
    const cz = -Math.sin(phi);
    appendCappedTube(
      b,
      [
        new THREE.Vector3(cx * 0.007, 0.095, cz * 0.007),
        new THREE.Vector3(cx * 0.013, 0.078, cz * 0.013),
        new THREE.Vector3(cx * (FILAMENT_TOP_R * 0.92), FILAMENT_TOP_Y + 0.003, cz * (FILAMENT_TOP_R * 0.92)),
      ],
      () => 0.00185,
      6
    );
  }

  return finalizeBuilder(b);
}

/**
 * Builds the signature antique 6-loop (12-strand) Squirrel-Cage Hairpin Glowing Tungsten Filament
 * strung cleanly between the 6 upper and 6 lower molybdenum anchor hooks, carrying per-vertex
 * thermal attribute `aHeat in [0.14, 1.0]` (white-hot mid-spans -> cooler Cinnabar-orange anchor ends!).
 */
export function createSquirrelCageFilamentGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const heatValues: number[] = [];
  const dPhi = (Math.PI * 2) / FILAMENT_LOOPS;

  const evalPoint = (loopIdx: number, localU: number): { pos: THREE.Vector3; heat: number } => {
    // Each loop consists of:
    //   [0.0 .. 0.44]: ascending strand from lower hook (k - 0.5) to upper hook (k)
    //   [0.44 .. 0.56]: tight rounded hairpin arch over upper molybdenum hook (k)
    //   [0.56 .. 0.94]: descending strand from upper hook (k) to lower hook (k + 0.5)
    //   [0.94 .. 1.00]: tight rounded hairpin trough around lower molybdenum hook (k + 0.5)
    const phiTop = loopIdx * dPhi;
    const phiBotPrev = (loopIdx - 0.5) * dPhi;
    const phiBotNext = (loopIdx + 0.5) * dPhi;

    const topPt = new THREE.Vector3(
      Math.cos(phiTop) * FILAMENT_TOP_R,
      FILAMENT_TOP_Y,
      -Math.sin(phiTop) * FILAMENT_TOP_R
    );
    const botPrev = new THREE.Vector3(
      Math.cos(phiBotPrev) * FILAMENT_BOT_R,
      FILAMENT_BOT_Y,
      -Math.sin(phiBotPrev) * FILAMENT_BOT_R
    );
    const botNext = new THREE.Vector3(
      Math.cos(phiBotNext) * FILAMENT_BOT_R,
      FILAMENT_BOT_Y,
      -Math.sin(phiBotNext) * FILAMENT_BOT_R
    );

    const p = new THREE.Vector3();
    let heat = 1.0;

    if (localU < 0.5) {
      // Bottom-prev -> Top
      const t = localU / 0.5; // 0..1
      // Subtle graceful vintage hairpin S-curve + slight outward bow in the middle
      const smoothT = t;
      const phi = phiBotPrev + (phiTop - phiBotPrev) * smoothT;
      const bow = Math.sin(t * Math.PI) * 0.0032;
      const r = FILAMENT_BOT_R + (FILAMENT_TOP_R - FILAMENT_BOT_R) * t + bow;
      // Smooth rounded arch right at top/bottom turns via cosine ease near endpoints
      const y = FILAMENT_BOT_Y + (FILAMENT_TOP_Y - FILAMENT_BOT_Y) * (0.5 - 0.5 * Math.cos(t * Math.PI));
      p.set(Math.cos(phi) * r, y, -Math.sin(phi) * r);
      // Thermal conduction cooling right at both molybdenum hooks (`t -> 0` or `t -> 1`)
      const endProx = Math.sin(t * Math.PI);
      heat = 0.16 + 0.84 * Math.pow(Math.max(0, endProx), 0.42);
    } else {
      // Top -> Bottom-next
      const t = (localU - 0.5) / 0.5; // 0..1
      const phi = phiTop + (phiBotNext - phiTop) * t;
      const bow = Math.sin(t * Math.PI) * 0.0032;
      const r = FILAMENT_TOP_R + (FILAMENT_BOT_R - FILAMENT_TOP_R) * t + bow;
      const y = FILAMENT_TOP_Y + (FILAMENT_BOT_Y - FILAMENT_TOP_Y) * (0.5 - 0.5 * Math.cos(t * Math.PI));
      p.set(Math.cos(phi) * r, y, -Math.sin(phi) * r);
      const endProx = Math.sin(t * Math.PI);
      heat = 0.16 + 0.84 * Math.pow(Math.max(0, endProx), 0.42);
    }

    // Subtle organic handmade tungsten filament micro-character (zero mechanical stiffness)
    const charAngle = loopIdx * 1.37 + localU * Math.PI * 2;
    p.x += Math.sin(charAngle * 3.0) * 0.00055;
    p.z += Math.cos(charAngle * 3.0) * 0.00055;

    // Anchor hooks contact sanity (blend toward exact hook coords at turn extrema)
    void topPt;
    void botPrev;
    void botNext;
    return { pos: p, heat };
  };

  // Emit each of the 6 complete hairpin V-loops as a sleek smooth closed micro-tube
  const segsPerLoop = 44;
  for (let k = 0; k < FILAMENT_LOOPS; k++) {
    const pts: THREE.Vector3[] = [];
    const heats: number[] = [];
    for (let i = 0; i <= segsPerLoop; i++) {
      const u = i / segsPerLoop;
      const sample = evalPoint(k, u);
      pts.push(sample.pos);
      heats.push(sample.heat);
    }
    appendCappedTube(
      b,
      pts,
      (s) => {
        const idx = Math.min(segsPerLoop, Math.round(s * segsPerLoop));
        // Subtle glowing expansion visually at hottest mid-spans (`0.00225`), tauter at anchor bends (`0.00185`)
        return 0.00185 + 0.00045 * heats[idx];
      },
      8,
      {
        values: heatValues,
        evalFn: (s) => {
          const idxF = s * segsPerLoop;
          const i0 = Math.floor(idxF);
          const i1 = Math.min(segsPerLoop, i0 + 1);
          const frac = idxF - i0;
          return heats[i0] * (1 - frac) + heats[i1] * frac;
        },
      }
    );
  }

  const geo = finalizeBuilder(b);
  geo.setAttribute('aHeat', new THREE.Float32BufferAttribute(heatValues, 1));
  return geo;
}

/**
 * Builds the authentic hand-blown ST64 Vintage Teardrop Edison Bulb Glass Envelope (`y in [BULB_BOT_Y, BULB_TOP_Y]`)
 * with its signature bottom blown glass exhaust tip, full pear belly, and tapered waist into the E27 screw collar.
 */
export function createST64BulbGeometry(segments = 36): THREE.BufferGeometry {
  return createOutwardLathe(
    [
      [0.0005, BULB_BOT_Y], // Iconic hand-blown exhaust tip nipple
      [0.0038, BULB_BOT_Y + 0.004],
      [0.0075, BULB_BOT_Y + 0.011],
      [0.018, BULB_BOT_Y + 0.017],
      [0.036, BULB_BOT_Y + 0.028],
      [0.050, BULB_BOT_Y + 0.044],
      [BULB_MAX_R, -0.052], // Full pear belly (r = 0.056)
      [0.055, -0.032],
      [0.050, -0.008],
      [0.044, 0.018],
      [0.0385, 0.046],
      [0.034, 0.074],
      [0.0315, 0.098],
      [0.0308, BULB_TOP_Y],
      [0.0005, BULB_TOP_Y],
    ],
    segments
  );
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
  const frontGlassMat = materials.infinityMirrorFront ?? materials.infinityMirror;
  const ceramicMat = materials.ceramicInsulator ?? materials.satinBrass;
  const darkCeramicMat = materials.darkCeramic ?? materials.agedBronze;
  const cordMat = materials.braidedCord ?? materials.agedBronze;
  const filamentMat = materials.edisonFilament ?? materials.satinBrass;
  const auraMat = materials.filamentAura ?? materials.satinBrass;
  const bulbGlassMat = materials.amberBulbGlass ?? materials.satinBrass;
  const prismGlassMat = materials.crystalPrismGlass ?? materials.satinBrass;
  const ventBaffleMat = materials.ventBaffleGlow ?? materials.agedBronze;

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
  // Chamfered brass washer rim + stepped domed horological instrument screw head
  const boltGeo = createOutwardLathe(
    [
      [0.0005, 0.0],
      [0.0136, 0.0],
      [0.0136, 0.0034],
      [0.0114, 0.0046],
      [0.0098, 0.0048],
      [0.0098, 0.0076],
      [0.0076, 0.0098],
      [0.0036, 0.0112],
      [0.0005, 0.0116],
    ],
    16
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
  const lockRingGeo = createKnurledDrumGeometry(0.0322, 0.011, 28, 2, 0.0012, 0.0022);
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
    lockRingGeo,
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

    const lockRing = new THREE.Mesh(lockRingGeo, materials.knurledBrass);
    lockRing.name = `${spec.name}_LockRing`;
    lockRing.position.y = 0.0155;
    lockRing.castShadow = true;
    lockRing.receiveShadow = true;
    tGroup.add(lockRing);

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

  // 5. Complete Watertight Suspended Edison Prism Lamp & Top Ceiling Anchor Assembly
  //    (Real 3D PBR + TSL horological hardware for Bounce 0, paired 1:1 with Bounces 1..40)
  const ceilingAnchorGroup = new THREE.Group();
  ceilingAnchorGroup.name = 'CeilingSuspensionAnchor';

  // 5a. Exterior Overhead Bronze Bridge Yoke Bar + Satin-Brass Rail Saddles + Bulkhead Crown
  //     Mortised directly into the -X and +X top bronze rails (`±0.6113`) so the pendant mount
  //     is structurally anchored across the top frame with zero floating glass-only fittings!
  const bridgeBarGeo = createChamferedBlockGeometry(0.622, 0.010, 0.0135, 0.0032);
  const bridgeSaddleGeo = createChamferedBlockGeometry(0.018, 0.016, 0.028, 0.003);
  const extBulkheadWasherGeo = createOutwardLathe(
    [
      [0.0005, GLASS_PANE_HALF + 0.0005],
      [0.068, GLASS_PANE_HALF + 0.0005],
      [0.068, GLASS_PANE_HALF + 0.008],
      [0.056, GLASS_PANE_HALF + 0.016],
      [0.044, GLASS_PANE_HALF + 0.018],
      [0.0005, GLASS_PANE_HALF + 0.018],
    ],
    32
  );
  const extHexNutGeo = createOutwardLathe(
    [
      [0.0005, 0.613],
      [0.044, 0.613],
      [0.044, 0.636],
      [0.036, 0.642],
      [0.0005, 0.642],
    ],
    6
  );
  const extCrownCapGeo = createOutwardLathe(
    [
      [0.0005, 0.640],
      [0.030, 0.640],
      [0.028, 0.648],
      [0.018, 0.656],
      [0.006, 0.661],
      [0.0005, 0.662],
    ],
    24
  );

  // 5b. Interior Stepped Satin-Brass Ceiling Rosette Canopy (flush at underside of +Y glass: 0.5975)
  const rosetteCanopyGeo = createOutwardLathe(
    [
      [0.0005, CEILING_ROSETTE_BOT_Y],
      [0.054, CEILING_ROSETTE_BOT_Y],
      [0.064, CEILING_ROSETTE_BOT_Y + 0.005],
      [0.082, CEILING_ROSETTE_BOT_Y + 0.011],
      [CEILING_ROSETTE_RADIUS, CEILING_ROSETTE_TOP_Y - 0.006],
      [CEILING_ROSETTE_RADIUS, CEILING_ROSETTE_TOP_Y],
      [0.0005, CEILING_ROSETTE_TOP_Y],
    ],
    36
  );
  const ceilingColletGeo = createKnurledDrumGeometry(
    0.031,
    CORD_GRIP_TOP_Y - CORD_GRIP_BOT_Y,
    28,
    6,
    0.0012,
    0.0042
  );
  const ceilingGlandGeo = createOutwardLathe(
    [
      [0.0005, GLAND_BOT_Y],
      [0.0195, GLAND_BOT_Y],
      [0.0245, (GLAND_BOT_Y + GLAND_TOP_Y) * 0.5],
      [0.0285, GLAND_TOP_Y],
      [0.0005, GLAND_TOP_Y],
    ],
    24
  );
  const ceramicGrommetGeo = createOutwardLathe(
    [
      [0.0005, GROMMET_BOT_Y],
      [0.0145, GROMMET_BOT_Y],
      [0.0195, GROMMET_BOT_Y + 0.004],
      [0.021, (GROMMET_BOT_Y + GROMMET_TOP_Y) * 0.5],
      [0.0195, GROMMET_TOP_Y],
      [0.0005, GROMMET_TOP_Y],
    ],
    24
  );

  geometries.push(
    bridgeBarGeo,
    bridgeSaddleGeo,
    extBulkheadWasherGeo,
    extHexNutGeo,
    extCrownCapGeo,
    rosetteCanopyGeo,
    ceilingColletGeo,
    ceilingGlandGeo,
    ceramicGrommetGeo
  );

  const bridgeBar = new THREE.Mesh(bridgeBarGeo, materials.agedBronze);
  bridgeBar.name = 'TopBridgeYokeBar';
  bridgeBar.position.set(0, 0.624, 0);
  bridgeBar.castShadow = true;
  bridgeBar.receiveShadow = true;
  ceilingAnchorGroup.add(bridgeBar);
  frameHitMeshes.push(bridgeBar);

  for (const sx of [-1, 1]) {
    const saddle = new THREE.Mesh(bridgeSaddleGeo, materials.satinBrass);
    saddle.name = `TopBridgeSaddle_${sx}`;
    saddle.position.set(sx * 0.606, 0.624, 0);
    saddle.castShadow = true;
    saddle.receiveShadow = true;
    ceilingAnchorGroup.add(saddle);
    frameHitMeshes.push(saddle);

    for (const sz of [-1, 1]) {
      const sBolt = new THREE.Mesh(boltGeo, materials.satinBrass);
      sBolt.name = `TopBridgeSaddle_${sx}_Bolt_${sz}`;
      sBolt.position.set(sx * 0.606, 0.638, sz * 0.014);
      sBolt.scale.setScalar(0.68);
      ceilingAnchorGroup.add(sBolt);
    }
  }

  const extWasher = new THREE.Mesh(extBulkheadWasherGeo, materials.satinBrass);
  extWasher.name = 'TopBulkheadWasher';
  extWasher.castShadow = true;
  extWasher.receiveShadow = true;
  ceilingAnchorGroup.add(extWasher);
  frameHitMeshes.push(extWasher);

  const extNut = new THREE.Mesh(extHexNutGeo, materials.satinBrass);
  extNut.name = 'TopBulkheadHexNut';
  extNut.castShadow = true;
  extNut.receiveShadow = true;
  ceilingAnchorGroup.add(extNut);
  frameHitMeshes.push(extNut);

  const extCrown = new THREE.Mesh(extCrownCapGeo, materials.satinBrass);
  extCrown.name = 'TopBulkheadCrownCap';
  extCrown.castShadow = true;
  ceilingAnchorGroup.add(extCrown);

  const rosetteMesh = new THREE.Mesh(rosetteCanopyGeo, materials.satinBrass);
  rosetteMesh.name = 'InteriorCeilingRosette';
  rosetteMesh.renderOrder = 1;
  ceilingAnchorGroup.add(rosetteMesh);

  // 4 radial brass screws visibly bolting the ceiling rosette to the top mirror
  for (let i = 0; i < 4; i++) {
    const ang = (i * Math.PI) / 2 + Math.PI / 4;
    const rBolt = new THREE.Mesh(boltGeo, materials.satinBrass);
    rBolt.name = `CeilingRosetteBolt_${i}`;
    rBolt.position.set(Math.cos(ang) * 0.072, CEILING_ROSETTE_BOT_Y + 0.008, Math.sin(ang) * 0.072);
    rBolt.rotation.x = Math.PI;
    rBolt.scale.setScalar(0.76);
    rBolt.renderOrder = 1;
    ceilingAnchorGroup.add(rBolt);
  }

  const ceilingCollet = new THREE.Mesh(ceilingColletGeo, materials.knurledBrass);
  ceilingCollet.name = 'CeilingCordGripCollet';
  ceilingCollet.position.y = CORD_GRIP_BOT_Y;
  ceilingCollet.renderOrder = 1;
  ceilingAnchorGroup.add(ceilingCollet);

  const ceilingGland = new THREE.Mesh(ceilingGlandGeo, materials.satinBrass);
  ceilingGland.name = 'CeilingStrainReliefGland';
  ceilingGland.renderOrder = 1;
  ceilingAnchorGroup.add(ceilingGland);

  const grommetMesh = new THREE.Mesh(ceramicGrommetGeo, ceramicMat);
  grommetMesh.name = 'CeilingCeramicGrommet';
  grommetMesh.renderOrder = 1;
  ceilingAnchorGroup.add(grommetMesh);

  group.add(ceilingAnchorGroup);

  // 5c. Articulated Woven Braided Cloth Pendant Cord (emerges cleanly from ceramic grommet into Edison socket)
  const braidedCordGeo = createBraidedCordGeometry(20, 34);
  geometries.push(braidedCordGeo);
  const braidedCordMesh = new THREE.Mesh(braidedCordGeo, cordMat);
  braidedCordMesh.name = 'BraidedPendantCord';
  braidedCordMesh.renderOrder = 1;
  braidedCordMesh.frustumCulled = false;
  group.add(braidedCordMesh);

  // 5d. Swinging Vintage Brass Edison Socket + Crystal Prism Cage + ST64 Bulb + Squirrel-Cage Filament
  const lampHeadGroup = new THREE.Group();
  lampHeadGroup.name = 'SuspendedEdisonPrismLamp';
  lampHeadGroup.position.set(0, CORE_REST_Y, 0);

  // Top brass cord-grip compression nut & hanging bail enclosing CORD_SOCKET_ATTACH_Y (+0.304)
  const socketGripNutGeo = createOutwardLathe(
    [
      [0.0005, 0.272],
      [0.028, 0.272],
      [0.026, 0.286],
      [0.0235, 0.290],
      [0.0235, 0.306],
      [0.019, 0.312],
      [0.0155, SOCKET_TOP_Y],
      [0.0005, SOCKET_TOP_Y],
    ],
    24
  );
  const socketCapGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_CAP_BOT_Y],
      [0.0505, SOCKET_CAP_BOT_Y],
      [0.049, SOCKET_CAP_BOT_Y + 0.006],
      [0.043, SOCKET_CAP_BOT_Y + 0.014],
      [0.032, 0.272],
      [0.027, 0.276],
      [0.0005, 0.276],
    ],
    32
  );
  const ventBaffleGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_VENT_BOT_Y],
      [0.0435, SOCKET_VENT_BOT_Y],
      [0.0435, SOCKET_CAP_BOT_Y + 0.003],
      [0.0005, SOCKET_CAP_BOT_Y + 0.003],
    ],
    28
  );
  const ventStrutsGeo = createSocketVentStrutsGeometry();
  const socketKnurlGeo = createKnurledDrumGeometry(
    0.0515,
    SOCKET_VENT_BOT_Y - SOCKET_KNURL_BOT_Y + 0.002,
    40,
    5,
    0.0011,
    0.0032
  );
  const socketSkirtGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_SKIRT_BOT_Y],
      [0.0475, SOCKET_SKIRT_BOT_Y],
      [0.0525, SOCKET_SKIRT_BOT_Y + 0.005],
      [0.0505, SOCKET_SKIRT_BOT_Y + 0.012],
      [0.0518, SOCKET_KNURL_BOT_Y + 0.002],
      [0.0005, SOCKET_KNURL_BOT_Y + 0.002],
    ],
    32
  );
  const darkInsulatorGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_INSULATOR_BOT_Y],
      [0.0405, SOCKET_INSULATOR_BOT_Y],
      [0.042, SOCKET_SKIRT_BOT_Y + 0.004],
      [0.0005, SOCKET_SKIRT_BOT_Y + 0.004],
    ],
    28
  );
  // Vintage radial brass rotary paddle turn-key switch on the side of the Edison socket
  const switchStemGeo = createOutwardLathe(
    [
      [0.0005, 0.0],
      [0.009, 0.0],
      [0.009, 0.008],
      [0.0056, 0.012],
      [0.0056, 0.028],
      [0.0005, 0.028],
    ],
    16
  );
  const switchPaddleGeo = createChamferedBlockGeometry(0.012, 0.016, 0.0044, 0.0018);

  // Threaded brass E27 Edison screw shell + glass-seal neck ring (visible inside the top of the prism!)
  const e27ScrewGeo = createHelicalLeadScrewGeometry(
    E27_BASE_BOT_Y + 0.004,
    E27_BASE_TOP_Y,
    0.0285,
    0.0324,
    5,
    28,
    36
  );
  const e27FlareGeo = createOutwardLathe(
    [
      [0.0005, E27_BASE_BOT_Y],
      [0.0315, E27_BASE_BOT_Y],
      [0.0325, E27_BASE_BOT_Y + 0.006],
      [0.0295, E27_BASE_BOT_Y + 0.01],
      [0.0005, E27_BASE_BOT_Y + 0.01],
    ],
    28
  );

  // Stepped brass prism gallery crown seating the top of the crystal prism + spider claws + edge-came ribs
  const galleryCrownGeo = createOutwardLathe(
    [
      [0.0005, 0.146],
      [0.035, 0.146],
      [0.044, 0.156],
      [0.0485, 0.172],
      [0.047, 0.178],
      [0.0005, 0.178],
    ],
    28
  );
  const prismCageGeo = createPrismCageAndClawsGeometry();
  const eqBossGeo = createChamferedBlockGeometry(0.0092, 0.0092, 0.0092, 0.003);

  // Sculpted machined brass bottom drop finial cradling the lower crystal prism apex
  const bottomFinialGeo = createOutwardLathe(
    [
      [0.0005, FINIAL_TIP_Y],
      [0.0038, FINIAL_TIP_Y + 0.006],
      [0.0115, FINIAL_TIP_Y + 0.016],
      [0.019, -0.212],
      [0.0195, -0.200],
      [0.0135, -0.188],
      [0.0105, -0.180],
      [0.0155, -0.172],
      [0.024, -0.164],
      [0.034, -0.154],
      [0.0335, FINIAL_TOP_Y],
      [0.0005, FINIAL_TOP_Y],
    ],
    28
  );

  // Glass exhaust pinch stem, central arbor & blown anchor buttons + 12 molybdenum hooks + 6-loop filament
  const pinchStemGeo = createOutwardLathe(
    [
      [0.0005, FILAMENT_BOT_Y - 0.005],
      [0.0085, FILAMENT_BOT_Y - 0.003],
      [0.0092, FILAMENT_BOT_Y + 0.002],
      [0.0044, FILAMENT_BOT_Y + 0.008],
      [0.0042, FILAMENT_TOP_Y - 0.006],
      [0.0088, FILAMENT_TOP_Y],
      [0.0092, FILAMENT_TOP_Y + 0.005],
      [0.0065, 0.072],
      [0.014, 0.088],
      [0.0165, 0.105],
      [0.019, BULB_TOP_Y],
      [0.0005, BULB_TOP_Y],
    ],
    20
  );
  const molyHooksGeo = createMolybdenumHooksGeometry();
  const filamentGeo = createSquirrelCageFilamentGeometry();
  const filamentAuraGeo = createOutwardLathe(
    [
      [0.0005, FILAMENT_BOT_Y - 0.018],
      [0.026, FILAMENT_BOT_Y - 0.010],
      [0.046, -0.042],
      [0.047, 0.005],
      [0.038, 0.045],
      [0.022, FILAMENT_TOP_Y + 0.014],
      [0.0005, FILAMENT_TOP_Y + 0.018],
    ],
    24
  );
  const st64BulbGeo = createST64BulbGeometry(36);
  const crystalPrismGeo = createTruncatedPrismGeometry();

  geometries.push(
    socketGripNutGeo,
    socketCapGeo,
    ventBaffleGeo,
    ventStrutsGeo,
    socketKnurlGeo,
    socketSkirtGeo,
    darkInsulatorGeo,
    switchStemGeo,
    switchPaddleGeo,
    e27ScrewGeo,
    e27FlareGeo,
    galleryCrownGeo,
    prismCageGeo,
    eqBossGeo,
    bottomFinialGeo,
    pinchStemGeo,
    molyHooksGeo,
    filamentGeo,
    filamentAuraGeo,
    st64BulbGeo,
    crystalPrismGeo
  );

  const addLampPart = (
    name: string,
    geo: THREE.BufferGeometry,
    mat: THREE.Material,
    renderOrder = 1
  ): THREE.Mesh => {
    const m = new THREE.Mesh(geo, mat);
    m.name = name;
    m.renderOrder = renderOrder;
    lampHeadGroup.add(m);
    return m;
  };

  addLampPart('EdisonSocketCordGripNut', socketGripNutGeo, materials.satinBrass, 1);
  addLampPart('EdisonSocketCap', socketCapGeo, materials.satinBrass, 1);
  addLampPart('EdisonSocketVentBaffle', ventBaffleGeo, ventBaffleMat, 1);
  addLampPart('EdisonSocketVentStruts', ventStrutsGeo, materials.satinBrass, 1);

  const socketKnurlMesh = addLampPart('EdisonSocketKnurledCollar', socketKnurlGeo, materials.knurledBrass, 1);
  socketKnurlMesh.position.y = SOCKET_KNURL_BOT_Y;

  const switchStemMesh = addLampPart('EdisonSocketSwitchStem', switchStemGeo, materials.satinBrass, 1);
  switchStemMesh.position.set(0.048, 0.212, 0);
  switchStemMesh.rotation.z = -Math.PI / 2;

  const switchPaddleMesh = addLampPart('EdisonSocketSwitchPaddle', switchPaddleGeo, materials.agedBronze, 1);
  switchPaddleMesh.position.set(0.082, 0.212, 0);

  addLampPart('EdisonSocketSkirt', socketSkirtGeo, materials.satinBrass, 1);
  addLampPart('EdisonSocketCeramicRing', darkInsulatorGeo, darkCeramicMat, 1);
  addLampPart('EdisonE27ScrewBase', e27ScrewGeo, materials.satinBrass, 1);
  addLampPart('EdisonE27NeckRing', e27FlareGeo, materials.satinBrass, 1);
  addLampPart('PrismGalleryCrown', galleryCrownGeo, materials.agedBronze, 1);
  addLampPart('PrismCageAndSpiderClaws', prismCageGeo, materials.satinBrass, 1);

  for (let i = 0; i < 4; i++) {
    const ang = (i * Math.PI) / 2;
    const boss = addLampPart(`PrismEquatorBoss_${i}`, eqBossGeo, materials.satinBrass, 1);
    boss.position.set(Math.cos(ang) * PRISM_EQUATOR_R, 0, -Math.sin(ang) * PRISM_EQUATOR_R);
    boss.rotation.y = Math.PI / 4;
  }

  addLampPart('PrismBottomDropFinial', bottomFinialGeo, materials.satinBrass, 1);
  addLampPart('BulbMolybdenumHooks', molyHooksGeo, materials.bluedSteel, 1);
  addLampPart('EdisonSquirrelCageFilament', filamentGeo, filamentMat, 1);

  // Transparent interior glass & crystal layers (rendered after opaque filament/cage, before exterior glass)
  addLampPart('BulbGlassPinchStem', pinchStemGeo, bulbGlassMat, 2);
  addLampPart('EdisonFilamentInnerAura', filamentAuraGeo, auraMat, 2);
  addLampPart('VintageST64GlassBulb', st64BulbGeo, bulbGlassMat, 2);
  addLampPart('OpticalCrystalPrismPanes', crystalPrismGeo, prismGlassMat, 2);

  group.add(lampHeadGroup);

  // 6. 2-Layer Semi-Silvered Two-Way Infinity Mirror Glass Panes (Hermetically Seated Behind Brass Bezels)
  //    - BackSide (`renderOrder = 0`): 40-bounce analytic tilted-plane raytracer behind the real 3D Edison lamp
  //    - FrontSide (`renderOrder = 3`): transparent exterior studio softbox Fresnel + soda-lime celadon edge bevel
  const glassGeo = new THREE.BoxGeometry(
    GLASS_PANE_HALF * 2,
    GLASS_PANE_HALF * 2,
    GLASS_PANE_HALF * 2
  );
  geometries.push(glassGeo);

  const glassBackMesh = new THREE.Mesh(glassGeo, materials.infinityMirror);
  glassBackMesh.name = 'InfinityMirrorCavityBack';
  glassBackMesh.castShadow = false;
  glassBackMesh.receiveShadow = false;
  glassBackMesh.renderOrder = 0;
  group.add(glassBackMesh);

  const glassMesh = new THREE.Mesh(glassGeo, frontGlassMat);
  glassMesh.name = 'InfinityMirrorGlassPanes';
  glassMesh.castShadow = false;
  glassMesh.receiveShadow = false;
  glassMesh.renderOrder = 3;
  group.add(glassMesh);

  return {
    group,
    glassMesh,
    glassBackMesh,
    ceilingAnchorGroup,
    braidedCordMesh,
    lampHeadGroup,
    frameHitMeshes,
    thumbscrewHitMeshes,
    thumbscrews,
    dispose: () => {
      for (const g of geometries) g.dispose();
      hitMat.dispose();
    },
  };
}
