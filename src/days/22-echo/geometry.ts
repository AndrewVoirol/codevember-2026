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
  /** Optional spun antique anisotropic copper-bronze for turned Edison fixture & rosette (defaults to satinBrass). */
  spunCopper?: THREE.Material;
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
  tangents: number[];
  uvs: number[];
  knurlValleys: number[];
  crevices: number[];
}

function createMeshBuilder(): MeshBuilder {
  return { positions: [], normals: [], tangents: [], uvs: [], knurlValleys: [], crevices: [] };
}

/**
 * Computes a deterministic unit tangent vector perpendicular to `n` aligned as closely as possible
 * to `prefTangent` (so Three.js `TBNViewMatrix` & `MeshPhysicalNodeMaterial` GGX anisotropy
 * stretches highlights strictly along `T` and places brushed micro-grooves along `B = N × T`).
 */
function orthoTangent(n: THREE.Vector3, prefTangent?: THREE.Vector3): THREE.Vector3 {
  const t = new THREE.Vector3();
  if (prefTangent && prefTangent.lengthSq() > 1e-8) {
    t.copy(prefTangent).addScaledVector(n, -prefTangent.dot(n));
  }
  if (t.lengthSq() < 1e-6) {
    const ref = Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    t.crossVectors(ref, n);
  }
  return t.normalize();
}

function pushTriangle(
  b: MeshBuilder,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  n: THREE.Vector3,
  kv0 = 1,
  kv1 = 1,
  kv2 = 1,
  prefT?: THREE.Vector3,
  c0 = 1,
  c1 = 1,
  c2 = 1
): void {
  const tVec = orthoTangent(n, prefT);
  const bVec = new THREE.Vector3().crossVectors(n, tVec).normalize();
  b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  b.normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z);
  b.tangents.push(
    tVec.x, tVec.y, tVec.z, 1,
    tVec.x, tVec.y, tVec.z, 1,
    tVec.x, tVec.y, tVec.z, 1
  );
  // Metric planar UVs aligned cleanly to (T, B) so brush-groove textures never distort per triangle
  b.uvs.push(
    p0.dot(tVec), p0.dot(bVec),
    p1.dot(tVec), p1.dot(bVec),
    p2.dot(tVec), p2.dot(bVec)
  );
  b.knurlValleys.push(kv0, kv1, kv2);
  b.crevices.push(c0, c1, c2);
}

/**
 * Emits a convex polygon sorted strictly counter-clockwise around `outwardNormal`,
 * guaranteeing 100% outward-facing winding (`inwardTri === 0`), crisp flat machined normals,
 * and deterministic anisotropic brush tangents.
 */
function pushConvexFace(
  b: MeshBuilder,
  verts: THREE.Vector3[],
  outwardNormal: THREE.Vector3,
  prefTangent?: THREE.Vector3,
  creviceVal = 1.0
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
    pushTriangle(b, sorted[0].v, sorted[i].v, sorted[i + 1].v, n, 1, 1, 1, prefTangent, creviceVal, creviceVal, creviceVal);
  }
}

function finalizeBuilder(b: MeshBuilder, includeKnurlAttr = false): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(b.positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(b.normals, 3));
  geo.setAttribute('tangent', new THREE.Float32BufferAttribute(b.tangents, 4));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uvs, 2));
  geo.setAttribute('aCrevice', new THREE.Float32BufferAttribute(b.crevices, 1));
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

  // Align brush bitangent (B = N × T) along the block's longest axis so rail grain runs longitudinally!
  const longAxis =
    hx >= hy && hx >= hz
      ? new THREE.Vector3(1, 0, 0)
      : hy >= hx && hy >= hz
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(0, 0, 1);
  const prefTFor = (n: THREE.Vector3): THREE.Vector3 => {
    const t = new THREE.Vector3().crossVectors(longAxis, n);
    return t.lengthSq() > 1e-6 ? t.normalize() : orthoTangent(n);
  };

  const vx = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * hx, sy * iy, sz * iz);
  const vy = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * ix, sy * hy, sz * iz);
  const vz = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * ix, sy * iy, sz * hz);

  for (const s of [-1, 1]) {
    const nx = new THREE.Vector3(s, 0, 0);
    pushConvexFace(b, [vx(s, -1, -1), vx(s, 1, -1), vx(s, 1, 1), vx(s, -1, 1)], nx, prefTFor(nx), 1.0);
    const ny = new THREE.Vector3(0, s, 0);
    pushConvexFace(b, [vy(-1, s, -1), vy(1, s, -1), vy(1, s, 1), vy(-1, s, 1)], ny, prefTFor(ny), 1.0);
    const nz = new THREE.Vector3(0, 0, s);
    pushConvexFace(b, [vz(-1, -1, s), vz(1, -1, s), vz(1, 1, s), vz(-1, 1, s)], nz, prefTFor(nz), 1.0);
  }

  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      const nxy = new THREE.Vector3(s1, s2, 0).normalize();
      pushConvexFace(b, [vx(s1, s2, -1), vx(s1, s2, 1), vy(s1, s2, 1), vy(s1, s2, -1)], nxy, prefTFor(nxy), 1.0);
      const nxz = new THREE.Vector3(s1, 0, s2).normalize();
      pushConvexFace(b, [vx(s1, -1, s2), vx(s1, 1, s2), vz(s1, 1, s2), vz(s1, -1, s2)], nxz, prefTFor(nxz), 1.0);
      const nyz = new THREE.Vector3(0, s1, s2).normalize();
      pushConvexFace(b, [vy(-1, s1, s2), vy(1, s1, s2), vz(1, s1, s2), vz(-1, s1, s2)], nyz, prefTFor(nyz), 1.0);
    }
  }

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const nc = new THREE.Vector3(sx, sy, sz).normalize();
        pushConvexFace(b, [vx(sx, sy, sz), vy(sx, sy, sz), vz(sx, sy, sz)], nc, prefTFor(nc), 1.0);
      }
    }
  }

  return finalizeBuilder(b);
}

/**
 * Builds a watertight uniformly-beveled triangular satin-brass corner gusset plate in XY
 * spanning `[0..armLen]` along +X and +Y, with back face embedded `-0.0025` into the bronze face,
 * diagonal brushed anisotropic grain, darkened oxidized skirt edges, and crisp 45° perimeter bevels.
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

  // Diagonal brush grain across the triangular corner plate (B along hypotenuse [-1, 1, 0], T along radial [1, 1, 0])
  const gussetRadialT = new THREE.Vector3(1, 1, 0).normalize();
  pushConvexFace(b, bevelFront, new THREE.Vector3(0, 0, 1), gussetRadialT, 0.96);
  pushConvexFace(b, backVerts, new THREE.Vector3(0, 0, -1), gussetRadialT, 0.2);

  const nPts = pts2d.length;
  for (let i = 0; i < nPts; i++) {
    const j = (i + 1) % nPts;
    const edgeX = pts2d[j][0] - pts2d[i][0];
    const edgeY = pts2d[j][1] - pts2d[i][1];
    const out2d = new THREE.Vector3(edgeY, -edgeX, 0).normalize();

    pushConvexFace(b, [backVerts[i], backVerts[j], frontVerts[j], frontVerts[i]], out2d, new THREE.Vector3(0, 0, 1), 0.34);

    const bevelNorm = out2d.clone().add(new THREE.Vector3(0, 0, 0.85)).normalize();
    pushConvexFace(b, [frontVerts[i], frontVerts[j], bevelFront[j], bevelFront[i]], bevelNorm, out2d, 0.88);
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

    // Align brushed grain bitangent B along the length of each mitered bezel bar (`T` = across bar width!)
    const acrossT = new THREE.Vector3(bar.outDir[0], bar.outDir[1], 0);
    // Front flat satin-brass retaining face (+Z)
    pushConvexFace(b, [fOA, fOB, fIB, fIA], new THREE.Vector3(0, 0, 1), acrossT, 0.96);
    // Back face (-Z)
    pushConvexFace(b, [bOA, bOB, bIB, bIA], new THREE.Vector3(0, 0, -1), acrossT, 0.25);
    // Outer perimeter wall (embedded inside bronze rails)
    pushConvexFace(b, [bOA, bOB, fOB, fOA], new THREE.Vector3(bar.outDir[0], bar.outDir[1], 0), new THREE.Vector3(0, 0, 1), 0.3);
    // Crisp 45-deg optical lip chamfer
    const slopeNorm = new THREE.Vector3(
      bar.inDir[0] * (zFront - zChamferBot),
      bar.inDir[1] * (zFront - zChamferBot),
      innerStep
    ).normalize();
    pushConvexFace(b, [fIA, fIB, mIB, mIA], slopeNorm, acrossT, 0.92);
    // Short vertical inner rabbet wall seating directly against the glass perimeter (dark oxidized depth)
    pushConvexFace(b, [mIA, mIB, bIB, bIA], new THREE.Vector3(bar.inDir[0], bar.inDir[1], 0), new THREE.Vector3(0, 0, 1), 0.32);
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
 * Precision crease-aware outward-wound lathe builder along +Y from a bottom-to-top profile
 * `[radius, y, optionalCrevice?]`.
 *
 * Guarantees:
 * - Strictly outward triangle winding (`signedVol > 0`, `flippedNormTris === 0` across every cap & step!)
 * - Automatic crisp hard-edge splitting across machined steps (`angle > creaseAngleDeg`) while keeping
 *   domes, cones, and ST64 glass curves silky smooth
 * - Exact analytic meridional `tangent` (`vec4`) at every vertex so `B = N × T` runs purely
 *   circumferentially around the lathe axis -> authentic `AnisotropyBarnLamp` radial/bow-tie GGX & IBL sheen!
 * - Arc-length V UVs + per-vertex `aCrevice` (`0` = recessed step oxidation groove, `1` = exposed spun copper)
 */
function createOutwardLathe(
  profile: Array<[number, number, number?]>,
  segments = 32,
  creaseAngleDeg = 38
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const nProf = profile.length;
  if (nProf < 2) return finalizeBuilder(b);

  const cosCrease = Math.cos((creaseAngleDeg * Math.PI) / 180);
  const dPhi = (Math.PI * 2) / segments;

  // Compute cumulative arc length along the profile for uniform lathe-groove V coordinates
  const arcV: number[] = [0];
  let totalArc = 0;
  for (let i = 1; i < nProf; i++) {
    const dr = profile[i][0] - profile[i - 1][0];
    const dy = profile[i][1] - profile[i - 1][1];
    totalArc += Math.hypot(dr, dy);
    arcV.push(totalArc);
  }
  const invArc = totalArc > 1e-6 ? 1 / totalArc : 1;

  // Compute 2D profile segment outward normals (nr, ny) and meridional tangents (tr, ty)
  // For bottom-to-top exterior profile (dy >= 0): outward normal is (+dy, -dr) / len.
  const segCount = nProf - 1;
  const segN: Array<[number, number]> = [];
  const segT: Array<[number, number]> = [];
  const segValid: boolean[] = [];
  for (let i = 0; i < segCount; i++) {
    const dr = profile[i + 1][0] - profile[i][0];
    const dy = profile[i + 1][1] - profile[i][1];
    const len = Math.hypot(dr, dy);
    if (len < 1e-7) {
      segN.push([1, 0]);
      segT.push([0, 1]);
      segValid.push(false);
    } else {
      segN.push([dy / len, -dr / len]);
      segT.push([dr / len, dy / len]);
      segValid.push(true);
    }
  }

  // Auto-detect concave step recesses for authentic BarnLamp oxidized crevice pooling when not explicitly supplied
  const profCrevice: number[] = [];
  for (let i = 0; i < nProf; i++) {
    if (profile[i][2] !== undefined) {
      profCrevice.push(profile[i][2]!);
    } else if (i === 0 || i === nProf - 1) {
      profCrevice.push(0.85);
    } else {
      const nPrev = segN[i - 1];
      const nNext = segN[i];
      // Concave step corner: cross2D(nPrev, nNext) < 0 (normal bends inward -> recessed step groove!)
      const cross2 = nPrev[0] * nNext[1] - nPrev[1] * nNext[0];
      const dot2 = nPrev[0] * nNext[0] + nPrev[1] * nNext[1];
      if (cross2 < -0.18 && dot2 < 0.85) {
        profCrevice.push(0.18);
      } else if (cross2 > 0.25 && dot2 < 0.8) {
        // Convex machined chamfer lip -> slightly burnished & clean
        profCrevice.push(1.0);
      } else {
        profCrevice.push(0.92);
      }
    }
  }

  const pushLatheVert = (
    r: number,
    y: number,
    phi: number,
    nr: number,
    ny: number,
    tr: number,
    ty: number,
    u: number,
    v: number,
    crev: number
  ) => {
    const cs = Math.cos(phi);
    const sn = Math.sin(phi);
    // Right-handed cylindrical frame matching rest of geometry.ts: x = r cos(phi), z = -r sin(phi)
    b.positions.push(r * cs, y, -r * sn);
    const nLen = Math.hypot(nr, ny) || 1;
    const nx = (nr / nLen) * cs;
    const nyOut = ny / nLen;
    const nz = -(nr / nLen) * sn;
    b.normals.push(nx, nyOut, nz);

    // Exact meridional unit tangent T perpendicular to N so B = N × T = (sin(phi), 0, cos(phi)) (circumferential!)
    const tLen = Math.hypot(tr, ty) || 1;
    const tx = (tr / tLen) * cs;
    const tyOut = ty / tLen;
    const tz = -(tr / tLen) * sn;
    b.tangents.push(tx, tyOut, tz, 1);

    b.uvs.push(u, v);
    b.knurlValleys.push(1);
    b.crevices.push(crev);
  };

  for (let i = 0; i < segCount; i++) {
    if (!segValid[i]) continue;
    const [r0, y0] = profile[i];
    const [r1, y1] = profile[i + 1];
    const v0 = arcV[i] * invArc;
    const v1 = arcV[i + 1] * invArc;
    const c0 = profCrevice[i];
    const c1 = profCrevice[i + 1];

    // Smooth normal at start of segment i (blend with segment i-1 only if within crease angle)
    let n0r = segN[i][0];
    let n0y = segN[i][1];
    let t0r = segT[i][0];
    let t0y = segT[i][1];
    if (i > 0 && segValid[i - 1]) {
      const dotPrev = segN[i - 1][0] * segN[i][0] + segN[i - 1][1] * segN[i][1];
      if (dotPrev >= cosCrease) {
        n0r = segN[i - 1][0] + segN[i][0];
        n0y = segN[i - 1][1] + segN[i][1];
        t0r = -n0y;
        t0y = n0r;
      }
    }

    // Smooth normal at end of segment i (blend with segment i+1 only if within crease angle)
    let n1r = segN[i][0];
    let n1y = segN[i][1];
    let t1r = segT[i][0];
    let t1y = segT[i][1];
    if (i < segCount - 1 && segValid[i + 1]) {
      const dotNext = segN[i][0] * segN[i + 1][0] + segN[i][1] * segN[i + 1][1];
      if (dotNext >= cosCrease) {
        n1r = segN[i][0] + segN[i + 1][0];
        n1y = segN[i][1] + segN[i + 1][1];
        t1r = -n1y;
        t1y = n1r;
      }
    }

    for (let j = 0; j < segments; j++) {
      const phi0 = j * dPhi;
      const phi1 = (j + 1) * dPhi;
      const u0 = j / segments;
      const u1 = (j + 1) / segments;

      // Outward CCW winding: (phi0, i) -> (phi1, i) -> (phi1, i+1) and (phi0, i) -> (phi1, i+1) -> (phi0, i+1)
      pushLatheVert(r0, y0, phi0, n0r, n0y, t0r, t0y, u0, v0, c0);
      pushLatheVert(r0, y0, phi1, n0r, n0y, t0r, t0y, u1, v0, c0);
      pushLatheVert(r1, y1, phi1, n1r, n1y, t1r, t1y, u1, v1, c1);

      pushLatheVert(r0, y0, phi0, n0r, n0y, t0r, t0y, u0, v0, c0);
      pushLatheVert(r1, y1, phi1, n1r, n1y, t1r, t1y, u1, v1, c1);
      pushLatheVert(r1, y1, phi0, n1r, n1y, t1r, t1y, u0, v1, c1);
    }
  }

  return finalizeBuilder(b);
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
    // Metric UVs along the tube: u = circumferential, v = arc-length; use T = circumferential, B = axial
    // so drawn metal rods (`PrismCageAndSpiderClaws`) brush cleanly along their length instead of ringing!
    b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    b.normals.push(vn0.x, vn0.y, vn0.z, vn1.x, vn1.y, vn1.z, vn2.x, vn2.y, vn2.z);
    const t0 = orthoTangent(vn0);
    const t1 = orthoTangent(vn1);
    const t2 = orthoTangent(vn2);
    b.tangents.push(t0.x, t0.y, t0.z, 1, t1.x, t1.y, t1.z, 1, t2.x, t2.y, t2.z, 1);
    b.uvs.push(v0 * 0.14, u0 * 0.04, v1 * 0.14, u1 * 0.04, v2 * 0.14, u2 * 0.04);
    b.knurlValleys.push(1, 1, 1);
    b.crevices.push(0.95, 0.95, 0.95);
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
 * Builds the authentic early turn-of-the-century (c. 1900) tight twin-conductor twisted
 * cloth-braided Edison pendant cord (`y in [CORE_REST_Y + CORD_SOCKET_ATTACH_Y, CORE_ANCHOR_Y]`):
 * - Two tightly interwoven helical cloth-covered conductors (`10.5` tight full turns,
 *   `140` axial steps = >6.6x above Nyquist -> zero aliasing or flat-facet lumps!)
 * - Strictly outward-wound smooth analytical vertex normals & strand-aligned weave tangents
 *   stored in `aCordRestNormal` & `aCordRestTangent` so the GPU cubic Hermite swing transforms
 *   BOTH positions AND surface normals seamlessly as the wire leans!
 */
export function createBraidedCordGeometry(
  radialSegs = 14,
  axialSegs = 140
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const yTop = CORE_ANCHOR_Y;
  const yBot = CORE_REST_Y + CORD_SOCKET_ATTACH_Y;
  const cordLen = Math.abs(yTop - yBot);
  const dAlpha = (Math.PI * 2) / radialSegs;
  const turns = 12.5;
  const rStrandBase = CORD_RADIUS * 0.565;
  const rHelixBase = CORD_RADIUS - rStrandBase; // outer radius === CORD_RADIUS, solid core overlap -> zero gaps!

  const cordS: number[] = [];
  const cordRadial: number[] = [];
  const cordRestNormal: number[] = [];
  const cordRestTangent: number[] = [];

  interface StrandVert {
    pos: THREE.Vector3;
    radOff: THREE.Vector3;
    norm: THREE.Vector3;
    tan: THREE.Vector3;
    s: number;
    u: number;
  }

  const emitTri = (v0: StrandVert, v1: StrandVert, v2: StrandVert) => {
    b.positions.push(v0.pos.x, v0.pos.y, v0.pos.z, v1.pos.x, v1.pos.y, v1.pos.z, v2.pos.x, v2.pos.y, v2.pos.z);
    b.normals.push(v0.norm.x, v0.norm.y, v0.norm.z, v1.norm.x, v1.norm.y, v1.norm.z, v2.norm.x, v2.norm.y, v2.norm.z);
    b.tangents.push(v0.tan.x, v0.tan.y, v0.tan.z, 1, v1.tan.x, v1.tan.y, v1.tan.z, 1, v2.tan.x, v2.tan.y, v2.tan.z, 1);
    b.uvs.push(v0.u, v0.s, v1.u, v1.s, v2.u, v2.s);
    b.knurlValleys.push(1, 1, 1);
    b.crevices.push(1, 1, 1);
    cordS.push(v0.s, v1.s, v2.s);
    cordRadial.push(
      v0.radOff.x, v0.radOff.y, v0.radOff.z,
      v1.radOff.x, v1.radOff.y, v1.radOff.z,
      v2.radOff.x, v2.radOff.y, v2.radOff.z
    );
    cordRestNormal.push(
      v0.norm.x, v0.norm.y, v0.norm.z,
      v1.norm.x, v1.norm.y, v1.norm.z,
      v2.norm.x, v2.norm.y, v2.norm.z
    );
    cordRestTangent.push(
      v0.tan.x, v0.tan.y, v0.tan.z,
      v1.tan.x, v1.tan.y, v1.tan.z,
      v2.tan.x, v2.tan.y, v2.tan.z
    );
  };

  for (let strand = 0; strand < 2; strand++) {
    const phase0 = strand * Math.PI;
    const rings: StrandVert[][] = [];
    const centers: THREE.Vector3[] = [];
    const upVecs: THREE.Vector3[] = [];

    for (let i = 0; i <= axialSegs; i++) {
      const s = i / axialSegs; // 0 = top inside ceiling grommet, 1 = bottom inside socket cord-grip nut
      const y = yTop + (yBot - yTop) * s;
      // Smoothly cinch both conductors cleanly into the round porcelain grommet & brass compression collet
      const endEaseTop = Math.min(1, s / 0.045);
      const endEaseBot = Math.min(1, (1 - s) / 0.045);
      const endSmooth = endEaseTop * endEaseTop * (3 - 2 * endEaseTop) * (endEaseBot * endEaseBot * (3 - 2 * endEaseBot));
      const rH = rHelixBase * (0.28 + 0.72 * endSmooth);
      const theta = phase0 + Math.PI * 2 * turns * s;
      const dThetaDs = Math.PI * 2 * turns;

      const csT = Math.cos(theta);
      const snT = Math.sin(theta);
      const cx = rH * csT;
      const cz = -rH * snT;
      const cPos = new THREE.Vector3(cx, y, cz);
      centers.push(cPos);

      // Exact upward unit tangent U_c = -normalize(dC/ds), straightening smoothly to vertical at both collet seats
      const dCdx = -rH * snT * dThetaDs * endSmooth;
      const dCdy = -cordLen;
      const dCdz = -rH * csT * dThetaDs * endSmooth;
      const uVec = new THREE.Vector3(-dCdx, -dCdy, -dCdz).normalize();
      upVecs.push(uVec);

      // Twist-locked cross-section basis (N0, B0, U_c) with N0 × B0 = U_c
      const radAxis = new THREE.Vector3(csT, 0, -snT);
      const n0 = radAxis.clone().addScaledVector(uVec, -radAxis.dot(uVec)).normalize();
      const b0 = new THREE.Vector3().crossVectors(uVec, n0).normalize();

      const ring: StrandVert[] = [];
      for (let j = 0; j < radialSegs; j++) {
        const alpha = j * dAlpha;
        const ca = Math.cos(alpha);
        const sa = Math.sin(alpha);
        // Subtle woven micro-ovality only where conductors don't pinch
        const dOut = new THREE.Vector3(
          ca * n0.x + sa * b0.x,
          ca * n0.y + sa * b0.y,
          ca * n0.z + sa * b0.z
        ).normalize();
        const pos = cPos.clone().addScaledVector(dOut, rStrandBase);
        // Store exact 3D offset from centerline `(0, y, 0)` so helical tilt rotates identically with `(eX, upAx, eZ)`!
        const radOff = new THREE.Vector3(pos.x, pos.y - y, pos.z);
        ring.push({
          pos,
          radOff,
          norm: dOut,
          tan: uVec.clone(),
          s,
          u: j / radialSegs,
        });
      }
      rings.push(ring);
    }

    // Side tube strips: outward CCW quad = (bot[j], bot[jNext], top[jNext], top[j]) since (+alpha) × (+U_c) = +dOut!
    for (let i = 0; i < axialSegs; i++) {
      const rowTop = rings[i];
      const rowBot = rings[i + 1];
      for (let j = 0; j < radialSegs; j++) {
        const jNext = (j + 1) % radialSegs;
        const p00 = rowBot[j];
        const p10 = jNext === 0 ? { ...rowBot[0], u: 1.0 } : rowBot[jNext];
        const p11 = jNext === 0 ? { ...rowTop[0], u: 1.0 } : rowTop[jNext];
        const p01 = rowTop[j];
        emitTri(p00, p10, p11);
        emitTri(p00, p11, p01);
      }
    }

    // Top cap (outward normal = +upVecs[0])
    const topN = upVecs[0];
    const topC = centers[0];
    const topCenterVert: StrandVert = {
      pos: topC,
      radOff: new THREE.Vector3(topC.x, topC.y - yTop, topC.z),
      norm: topN,
      tan: orthoTangent(topN),
      s: 0,
      u: 0.5,
    };
    for (let j = 0; j < radialSegs; j++) {
      const jNext = (j + 1) % radialSegs;
      const vA: StrandVert = { ...rings[0][j], norm: topN };
      const vB: StrandVert = { ...rings[0][jNext], norm: topN };
      emitTri(topCenterVert, vA, vB);
    }

    // Bottom cap (outward normal = -upVecs[axialSegs])
    const botN = upVecs[axialSegs].clone().negate();
    const botC = centers[axialSegs];
    const botCenterVert: StrandVert = {
      pos: botC,
      radOff: new THREE.Vector3(botC.x, botC.y - yBot, botC.z),
      norm: botN,
      tan: orthoTangent(botN),
      s: 1,
      u: 0.5,
    };
    for (let j = 0; j < radialSegs; j++) {
      const jNext = (j + 1) % radialSegs;
      const vA: StrandVert = { ...rings[axialSegs][jNext], norm: botN };
      const vB: StrandVert = { ...rings[axialSegs][j], norm: botN };
      emitTri(botCenterVert, vA, vB);
    }
  }

  const geo = finalizeBuilder(b);
  geo.setAttribute('aCordS', new THREE.Float32BufferAttribute(cordS, 1));
  geo.setAttribute('aCordRadial', new THREE.Float32BufferAttribute(cordRadial, 3));
  geo.setAttribute('aCordRestNormal', new THREE.Float32BufferAttribute(cordRestNormal, 3));
  geo.setAttribute('aCordRestTangent', new THREE.Float32BufferAttribute(cordRestTangent, 3));
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

    // Cast curved spun-copper spider gallery claw sweeping from the flared gallery collar rim (`y = 0.168, r = 0.052`)
    // cleanly onto the EXACT mathematical upper prism ridge (`y = 0.098, r = rEq * (1 - 0.098 / PRISM_APEX_Y)`)!
    const radDir = eqPts[i].clone().normalize();
    const clawTipY = 0.098;
    const clawTipR = rEq * (1 - clawTipY / PRISM_APEX_Y); // exact ridge center radius (0.0653)
    const clawMidY = 0.126;
    const clawMidR = rEq * (1 - clawMidY / PRISM_APEX_Y) + 0.006;
    const clawCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(radDir.x * 0.051, 0.171, radDir.z * 0.051),
      new THREE.Vector3(radDir.x * 0.058, 0.162, radDir.z * 0.058),
      new THREE.Vector3(radDir.x * 0.053, 0.145, radDir.z * 0.053),
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
 * Builds the 12 sleek chamfered spun-copper heat-ventilation louver struts tucked cleanly around
 * the lower Edison socket shoulder (`[SOCKET_VENT_BOT_Y, SOCKET_VENT_BOT_Y + 0.014]`) above the flared gallery.
 */
export function createSocketVentStrutsGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const count = 12;
  const rInner = 0.0448;
  const rOuter = 0.0504;
  const y0 = SOCKET_VENT_BOT_Y - 0.002;
  const y1 = SOCKET_VENT_BOT_Y + 0.014;
  const halfAng = (Math.PI * 2) / (count * 3.9);

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

    pushConvexFace(b, [outA0, outB0, outB1, outA1], new THREE.Vector3(cM, 0, sM), new THREE.Vector3(0, 1, 0), 0.94);
    pushConvexFace(b, [inA0, inB0, inB1, inA1], new THREE.Vector3(-cM, 0, -sM), new THREE.Vector3(0, 1, 0), 0.25);
    pushConvexFace(b, [inA0, outA0, outA1, inA1], new THREE.Vector3(sA, 0, -cA), new THREE.Vector3(0, 1, 0), 0.35);
    pushConvexFace(b, [inB0, outB0, outB1, inB1], new THREE.Vector3(-sB, 0, cB), new THREE.Vector3(0, 1, 0), 0.35);
    pushConvexFace(b, [inA1, inB1, outB1, outA1], new THREE.Vector3(0, 1, 0), new THREE.Vector3(cM, 0, sM), 0.8);
    pushConvexFace(b, [inA0, inB0, outB0, outA0], new THREE.Vector3(0, -1, 0), new THREE.Vector3(cM, 0, sM), 0.8);
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

  // Emit each of the 6 graceful antique hairpin V-loops with wide rounded arches over the molybdenum hooks
  // (`minBendRadius >= 0.0055 > 2.5 * tubeRadius` -> strictly zero self-folding or flipped triangles!)
  const segsPerLoop = 36;
  for (let k = 0; k < FILAMENT_LOOPS; k++) {
    const phiTop = k * dPhi;
    const phiBotPrev = (k - 0.5) * dPhi;
    const phiBotNext = (k + 0.5) * dPhi;
    const archDPhi = dPhi * 0.14;

    const ptAt = (phi: number, r: number, y: number) =>
      new THREE.Vector3(Math.cos(phi) * r, y, -Math.sin(phi) * r);

    const midY = (FILAMENT_TOP_Y + FILAMENT_BOT_Y) * 0.5;
    const midR = (FILAMENT_TOP_R + FILAMENT_BOT_R) * 0.5 + 0.0028;

    const controlPts = [
      ptAt(phiBotPrev + archDPhi * 0.55, FILAMENT_BOT_R, FILAMENT_BOT_Y + 0.002),
      ptAt(phiBotPrev + dPhi * 0.16, FILAMENT_BOT_R * 0.72 + FILAMENT_TOP_R * 0.28 + 0.002, FILAMENT_BOT_Y + 0.032),
      ptAt((phiBotPrev + phiTop) * 0.5, midR, midY),
      ptAt(phiTop - dPhi * 0.16, FILAMENT_TOP_R * 0.76 + FILAMENT_BOT_R * 0.24 + 0.002, FILAMENT_TOP_Y - 0.026),
      ptAt(phiTop - archDPhi * 0.65, FILAMENT_TOP_R + 0.001, FILAMENT_TOP_Y - 0.003),
      ptAt(phiTop, FILAMENT_TOP_R + 0.0015, FILAMENT_TOP_Y + 0.0015),
      ptAt(phiTop + archDPhi * 0.65, FILAMENT_TOP_R + 0.001, FILAMENT_TOP_Y - 0.003),
      ptAt(phiTop + dPhi * 0.16, FILAMENT_TOP_R * 0.76 + FILAMENT_BOT_R * 0.24 + 0.002, FILAMENT_TOP_Y - 0.026),
      ptAt((phiTop + phiBotNext) * 0.5, midR, midY),
      ptAt(phiBotNext - dPhi * 0.16, FILAMENT_BOT_R * 0.72 + FILAMENT_TOP_R * 0.28 + 0.002, FILAMENT_BOT_Y + 0.032),
      ptAt(phiBotNext - archDPhi * 0.55, FILAMENT_BOT_R, FILAMENT_BOT_Y + 0.002),
    ];

    const loopCurve = new THREE.CatmullRomCurve3(controlPts, false, 'centripetal', 0.45);
    const pts = loopCurve.getPoints(segsPerLoop);
    const heats: number[] = [];
    for (let i = 0; i <= segsPerLoop; i++) {
      const u = i / segsPerLoop;
      // Thermal conduction cooling right at bottom (`u ~ 0, 1`) and top (`u ~ 0.5`) molybdenum anchor hooks
      const limbT = u < 0.5 ? u * 2 : (u - 0.5) * 2;
      const endProx = Math.sin(limbT * Math.PI);
      heats.push(0.16 + 0.84 * Math.pow(Math.max(0, endProx), 0.42));
    }

    appendCappedTube(
      b,
      pts,
      (s) => {
        const idx = Math.min(segsPerLoop, Math.round(s * segsPerLoop));
        return 0.00175 + 0.00042 * heats[idx];
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

  const spunCopperMat = materials.spunCopper ?? materials.satinBrass;
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

          // 3 machined spun-copper dome rivets seated cleanly inside the beveled plateau
          const boltCoords: Array<[string, number, number]> = [
            ['Corner', 0.032, 0.032],
            ['ArmA', 0.102, 0.026],
            ['ArmB', 0.026, 0.102],
          ];
          for (const [suffix, bx, by] of boltCoords) {
            const bolt = new THREE.Mesh(boltGeo, spunCopperMat);
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

  // 2 opposing slotted spun-copper fulcrum lock-caps on the -Y underside rails
  const lockPositions: Array<[number, number, number]> = [
    [0.41, -THUMBSCREW_MOUNT_OFFSET, RAIL_OFFSET],
    [-0.41, -THUMBSCREW_MOUNT_OFFSET, -RAIL_OFFSET],
  ];
  lockPositions.forEach((lp, idx) => {
    const cap = new THREE.Mesh(lockCapGeo, spunCopperMat);
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

  // 5a. Exterior Overhead Bronze Bridge Yoke Bar + Brushed Copper-Brass Rail Saddles + Spun Bulkhead Crown
  //     Mortised directly into the -X and +X top bronze rails (`±0.6113`) so the pendant mount
  //     is structurally anchored across the top frame with zero floating glass-only fittings!
  const bridgeBarGeo = createChamferedBlockGeometry(0.622, 0.010, 0.0135, 0.0032);
  const bridgeSaddleGeo = createChamferedBlockGeometry(0.018, 0.016, 0.028, 0.003);
  // Stepped spun-copper exterior bulkhead flange with concentric oxidized lathe groove
  const extBulkheadWasherGeo = createOutwardLathe(
    [
      [0.0005, GLASS_PANE_HALF + 0.0005, 0.9],
      [0.074, GLASS_PANE_HALF + 0.0005, 0.85],
      [0.074, GLASS_PANE_HALF + 0.006, 0.96],
      [0.068, GLASS_PANE_HALF + 0.010, 1.0],
      [0.061, GLASS_PANE_HALF + 0.010, 0.92],
      [0.059, GLASS_PANE_HALF + 0.0085, 0.14], // Turned concentric oxidized groove
      [0.056, GLASS_PANE_HALF + 0.0115, 0.95],
      [0.048, GLASS_PANE_HALF + 0.017, 1.0],
      [0.042, GLASS_PANE_HALF + 0.018, 0.3],
      [0.0005, GLASS_PANE_HALF + 0.018, 0.85],
    ],
    40
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
    28
  );

  // 5b. Interior Stepped Spun-Copper Ceiling Rosette Canopy (modeled on BarnLamp concentric wall canopy!)
  //     Sits flush at underside of +Y glass (`0.5975`) with two crisp concentric oxidized lathe grooves
  const rosetteCanopyGeo = createOutwardLathe(
    [
      [0.0005, CEILING_ROSETTE_BOT_Y, 0.85],
      [0.042, CEILING_ROSETTE_BOT_Y, 0.95],
      [0.049, CEILING_ROSETTE_BOT_Y + 0.004, 1.0],
      [0.055, CEILING_ROSETTE_BOT_Y + 0.0085, 0.92],
      [0.058, CEILING_ROSETTE_BOT_Y + 0.0105, 0.14], // Inner turned oxidized recess ring
      [0.062, CEILING_ROSETTE_BOT_Y + 0.0085, 0.96],
      [0.075, CEILING_ROSETTE_BOT_Y + 0.012, 1.0],
      [0.079, CEILING_ROSETTE_BOT_Y + 0.0145, 0.14], // Outer turned oxidized recess ring
      [0.083, CEILING_ROSETTE_BOT_Y + 0.013, 0.96],
      [CEILING_ROSETTE_RADIUS - 0.005, CEILING_ROSETTE_TOP_Y - 0.011, 1.0],
      [CEILING_ROSETTE_RADIUS, CEILING_ROSETTE_TOP_Y - 0.006, 0.95],
      [CEILING_ROSETTE_RADIUS, CEILING_ROSETTE_TOP_Y, 0.88],
      [0.0005, CEILING_ROSETTE_TOP_Y, 0.85],
    ],
    44
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
      [0.0255, (GLAND_BOT_Y + GLAND_TOP_Y) * 0.5],
      [0.0295, GLAND_TOP_Y],
      [0.0005, GLAND_TOP_Y],
    ],
    28
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
    28
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
      const sBolt = new THREE.Mesh(boltGeo, spunCopperMat);
      sBolt.name = `TopBridgeSaddle_${sx}_Bolt_${sz}`;
      sBolt.position.set(sx * 0.606, 0.638, sz * 0.014);
      sBolt.scale.setScalar(0.68);
      ceilingAnchorGroup.add(sBolt);
    }
  }

  const extWasher = new THREE.Mesh(extBulkheadWasherGeo, spunCopperMat);
  extWasher.name = 'TopBulkheadWasher';
  extWasher.castShadow = true;
  extWasher.receiveShadow = true;
  ceilingAnchorGroup.add(extWasher);
  frameHitMeshes.push(extWasher);

  const extNut = new THREE.Mesh(extHexNutGeo, spunCopperMat);
  extNut.name = 'TopBulkheadHexNut';
  extNut.castShadow = true;
  extNut.receiveShadow = true;
  ceilingAnchorGroup.add(extNut);
  frameHitMeshes.push(extNut);

  const extCrown = new THREE.Mesh(extCrownCapGeo, spunCopperMat);
  extCrown.name = 'TopBulkheadCrownCap';
  extCrown.castShadow = true;
  ceilingAnchorGroup.add(extCrown);

  const rosetteMesh = new THREE.Mesh(rosetteCanopyGeo, spunCopperMat);
  rosetteMesh.name = 'InteriorCeilingRosette';
  rosetteMesh.renderOrder = 1;
  ceilingAnchorGroup.add(rosetteMesh);

  // 4 radial spun-copper dome screws visibly bolting the ceiling rosette to the top mirror
  for (let i = 0; i < 4; i++) {
    const ang = (i * Math.PI) / 2 + Math.PI / 4;
    const rBolt = new THREE.Mesh(boltGeo, spunCopperMat);
    rBolt.name = `CeilingRosetteBolt_${i}`;
    rBolt.position.set(Math.cos(ang) * 0.072, CEILING_ROSETTE_BOT_Y + 0.011, Math.sin(ang) * 0.072);
    rBolt.rotation.x = Math.PI;
    rBolt.scale.setScalar(0.74);
    rBolt.renderOrder = 1;
    ceilingAnchorGroup.add(rBolt);
  }

  const ceilingCollet = new THREE.Mesh(ceilingColletGeo, materials.knurledBrass);
  ceilingCollet.name = 'CeilingCordGripCollet';
  ceilingCollet.position.y = CORD_GRIP_BOT_Y;
  ceilingCollet.renderOrder = 1;
  ceilingAnchorGroup.add(ceilingCollet);

  const ceilingGland = new THREE.Mesh(ceilingGlandGeo, spunCopperMat);
  ceilingGland.name = 'CeilingStrainReliefGland';
  ceilingGland.renderOrder = 1;
  ceilingAnchorGroup.add(ceilingGland);

  const grommetMesh = new THREE.Mesh(ceramicGrommetGeo, ceramicMat);
  grommetMesh.name = 'CeilingCeramicGrommet';
  grommetMesh.renderOrder = 1;
  ceilingAnchorGroup.add(grommetMesh);

  group.add(ceilingAnchorGroup);

  // 5c. Early Turn-of-the-Century Tight Twin-Conductor Twisted Cloth-Braided Pendant Cord
  const braidedCordGeo = createBraidedCordGeometry(14, 140);
  geometries.push(braidedCordGeo);
  const braidedCordMesh = new THREE.Mesh(braidedCordGeo, cordMat);
  braidedCordMesh.name = 'BraidedPendantCord';
  braidedCordMesh.renderOrder = 1;
  braidedCordMesh.frustumCulled = false;
  group.add(braidedCordMesh);

  // 5d. Swinging Spun-Copper BarnLamp-Inspired Vintage Edison Socket + Crystal Prism Cage + ST64 Bulb
  const lampHeadGroup = new THREE.Group();
  lampHeadGroup.name = 'SuspendedEdisonPrismLamp';
  lampHeadGroup.position.set(0, CORE_REST_Y, 0);

  // 1) Top spun-copper cord-grip compression nut & stepped bushing enclosing CORD_SOCKET_ATTACH_Y (+0.304)
  const socketGripNutGeo = createOutwardLathe(
    [
      [0.0005, 0.270, 0.9],
      [0.0285, 0.270, 0.25],
      [0.0285, 0.276, 0.95],
      [0.0255, 0.283, 1.0],
      [0.0235, 0.285, 0.2], // Crisp oxidized lathe step
      [0.0242, 0.288, 0.95],
      [0.0242, 0.305, 1.0],
      [0.0195, 0.311, 0.96],
      [0.016, SOCKET_TOP_Y, 0.92],
      [0.0005, SOCKET_TOP_Y, 0.85],
    ],
    32
  );

  // 2) Smooth spun-copper bell dome cap (`[SOCKET_CAP_BOT_Y = 0.246, 0.274]`) with turned oxidation groove
  const socketCapGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_CAP_BOT_Y, 0.9],
      [0.0505, SOCKET_CAP_BOT_Y, 0.22],
      [0.0512, SOCKET_CAP_BOT_Y + 0.0035, 0.98],
      [0.0488, SOCKET_CAP_BOT_Y + 0.0085, 1.0],
      [0.0442, SOCKET_CAP_BOT_Y + 0.0135, 0.96],
      [0.0418, SOCKET_CAP_BOT_Y + 0.0155, 0.16], // Turned BarnLamp dome crevice ring
      [0.0402, SOCKET_CAP_BOT_Y + 0.0175, 0.96],
      [0.0338, 0.268, 1.0],
      [0.028, 0.273, 0.92],
      [0.025, 0.275, 0.22],
      [0.0005, 0.275, 0.85],
    ],
    40
  );

  // 3) Crisp 4-sided diamond-knurled grip collar right beneath the dome cap (`[SOCKET_KNURL_BOT_Y = 0.220, 0.248]`)
  const socketKnurlGeo = createKnurledDrumGeometry(
    0.0518,
    SOCKET_CAP_BOT_Y - SOCKET_KNURL_BOT_Y + 0.002,
    42,
    5,
    0.0011,
    0.003
  );

  // 4) Smooth spun antique-copper cylindrical lamp-holder barrel (`[SOCKET_VENT_BOT_Y + 0.011, SOCKET_KNURL_BOT_Y + 0.002]`)
  //    Provides an uninterrupted spun-copper zone for vertical stretched anisotropic reflections + clean switch boss!
  const socketBarrelGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_VENT_BOT_Y + 0.011, 0.9],
      [0.0488, SOCKET_VENT_BOT_Y + 0.011, 0.22],
      [0.0498, SOCKET_VENT_BOT_Y + 0.014, 0.96],
      [0.0492, SOCKET_VENT_BOT_Y + 0.022, 1.0],
      [0.0492, SOCKET_KNURL_BOT_Y - 0.004, 1.0],
      [0.0504, SOCKET_KNURL_BOT_Y - 0.0015, 0.95],
      [0.0482, SOCKET_KNURL_BOT_Y + 0.002, 0.18], // Oxidized seam tucked cleanly under knurl collar
      [0.0005, SOCKET_KNURL_BOT_Y + 0.002, 0.85],
    ],
    40
  );

  // 5) Subtle lower cooling louver collar (`[SOCKET_VENT_BOT_Y - 0.002, SOCKET_VENT_BOT_Y + 0.014]`)
  const ventBaffleGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_VENT_BOT_Y - 0.002],
      [0.0438, SOCKET_VENT_BOT_Y - 0.002],
      [0.0438, SOCKET_VENT_BOT_Y + 0.014],
      [0.0005, SOCKET_VENT_BOT_Y + 0.014],
    ],
    28
  );
  const ventStrutsGeo = createSocketVentStrutsGeometry();

  // 6) Stepped flared BarnLamp-inspired spun-copper gallery collar (`[SOCKET_SKIRT_BOT_Y = 0.162, SOCKET_VENT_BOT_Y + 0.002]`)
  const socketSkirtGeo = createOutwardLathe(
    [
      [0.0005, SOCKET_SKIRT_BOT_Y, 0.85],
      [0.0465, SOCKET_SKIRT_BOT_Y, 0.92],
      [0.0585, SOCKET_SKIRT_BOT_Y + 0.0035, 1.0],
      [0.0605, SOCKET_SKIRT_BOT_Y + 0.007, 1.0],
      [0.0598, SOCKET_SKIRT_BOT_Y + 0.0135, 0.96],
      [0.0545, SOCKET_SKIRT_BOT_Y + 0.0175, 0.98],
      [0.0512, SOCKET_VENT_BOT_Y - 0.002, 0.18], // Dark oxidized step right below louvers
      [0.0495, SOCKET_VENT_BOT_Y + 0.002, 0.9],
      [0.0005, SOCKET_VENT_BOT_Y + 0.002, 0.85],
    ],
    40
  );

  // 4 miniature radial spun-copper knurled gallery thumbscrews echoing the BarnLamp shade-holder screws!
  const miniScrewGeo = createOutwardLathe(
    [
      [0.0005, 0.0, 0.85],
      [0.0038, 0.0, 0.3],
      [0.0038, 0.006, 0.85],
      [0.0068, 0.007, 0.96],
      [0.0072, 0.0135, 1.0],
      [0.0056, 0.0155, 0.95],
      [0.0005, 0.0155, 0.88],
    ],
    16
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

  // Vintage radial rotary paddle turn-key switch seated cleanly on a turned spun-copper barrel boss (`y = 0.206`)
  const switchBossGeo = createOutwardLathe(
    [
      [0.0005, -0.006, 0.85],
      [0.0125, -0.006, 0.2],
      [0.0125, 0.0035, 0.95],
      [0.0102, 0.0065, 1.0],
      [0.0072, 0.0075, 0.25],
      [0.0005, 0.0075, 0.85],
    ],
    20
  );
  const switchStemGeo = createOutwardLathe(
    [
      [0.0005, 0.004],
      [0.0058, 0.004],
      [0.0052, 0.025],
      [0.0005, 0.025],
    ],
    16
  );
  const switchPaddleGeo = createChamferedBlockGeometry(0.0125, 0.0165, 0.0044, 0.0018);

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

  // Stepped dark-bronze prism gallery crown seating the top of the crystal prism + spun-copper spider claws & ribs
  const galleryCrownGeo = createOutwardLathe(
    [
      [0.0005, 0.146],
      [0.035, 0.146],
      [0.044, 0.155],
      [0.0485, 0.167],
      [0.047, 0.172],
      [0.0005, 0.172],
    ],
    32
  );
  const prismCageGeo = createPrismCageAndClawsGeometry();
  const eqBossGeo = createChamferedBlockGeometry(0.0092, 0.0092, 0.0092, 0.003);

  // Sculpted spun antique-copper bottom drop finial with oxidized turning rings cradling lower crystal prism apex
  const bottomFinialGeo = createOutwardLathe(
    [
      [0.0005, FINIAL_TIP_Y, 0.9],
      [0.0038, FINIAL_TIP_Y + 0.006, 0.96],
      [0.012, FINIAL_TIP_Y + 0.017, 1.0],
      [0.0195, -0.212, 1.0],
      [0.0198, -0.201, 0.96],
      [0.014, -0.190, 0.85],
      [0.0102, -0.183, 0.18], // Oxidized turned finial neck groove
      [0.0108, -0.178, 0.25],
      [0.0165, -0.172, 0.98],
      [0.0255, -0.163, 1.0],
      [0.0345, -0.154, 0.96],
      [0.0338, FINIAL_TOP_Y, 0.3],
      [0.0005, FINIAL_TOP_Y, 0.85],
    ],
    36
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
    socketKnurlGeo,
    socketBarrelGeo,
    ventBaffleGeo,
    ventStrutsGeo,
    socketSkirtGeo,
    miniScrewGeo,
    darkInsulatorGeo,
    switchBossGeo,
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

  addLampPart('EdisonSocketCordGripNut', socketGripNutGeo, spunCopperMat, 1);
  addLampPart('EdisonSocketCap', socketCapGeo, spunCopperMat, 1);

  const socketKnurlMesh = addLampPart('EdisonSocketKnurledCollar', socketKnurlGeo, materials.knurledBrass, 1);
  socketKnurlMesh.position.y = SOCKET_KNURL_BOT_Y;

  addLampPart('EdisonSocketBarrel', socketBarrelGeo, spunCopperMat, 1);
  addLampPart('EdisonSocketVentBaffle', ventBaffleGeo, ventBaffleMat, 1);
  addLampPart('EdisonSocketVentStruts', ventStrutsGeo, spunCopperMat, 1);

  const switchBossMesh = addLampPart('EdisonSocketSwitchBoss', switchBossGeo, spunCopperMat, 1);
  switchBossMesh.position.set(0.048, 0.207, 0);
  switchBossMesh.rotation.z = -Math.PI / 2;

  const switchStemMesh = addLampPart('EdisonSocketSwitchStem', switchStemGeo, spunCopperMat, 1);
  switchStemMesh.position.set(0.048, 0.207, 0);
  switchStemMesh.rotation.z = -Math.PI / 2;

  const switchPaddleMesh = addLampPart('EdisonSocketSwitchPaddle', switchPaddleGeo, materials.agedBronze, 1);
  switchPaddleMesh.position.set(0.079, 0.207, 0);

  addLampPart('EdisonSocketSkirt', socketSkirtGeo, spunCopperMat, 1);

  // 4 radial mini spun-copper thumbscrews seated cleanly around the flared gallery collar (`45°` offsets)
  for (let i = 0; i < 4; i++) {
    const ang = (i * Math.PI) / 2 + Math.PI / 4;
    const ms = addLampPart(`EdisonGalleryThumbscrew_${i}`, miniScrewGeo, spunCopperMat, 1);
    const rMount = 0.056;
    ms.position.set(Math.cos(ang) * rMount, SOCKET_SKIRT_BOT_Y + 0.010, -Math.sin(ang) * rMount);
    ms.rotation.order = 'YXZ';
    ms.rotation.y = ang;
    ms.rotation.z = -Math.PI / 2;
  }

  addLampPart('EdisonSocketCeramicRing', darkInsulatorGeo, darkCeramicMat, 1);
  addLampPart('EdisonE27ScrewBase', e27ScrewGeo, materials.satinBrass, 1);
  addLampPart('EdisonE27NeckRing', e27FlareGeo, materials.satinBrass, 1);
  addLampPart('PrismGalleryCrown', galleryCrownGeo, materials.agedBronze, 1);
  addLampPart('PrismCageAndSpiderClaws', prismCageGeo, materials.satinBrass, 1);

  for (let i = 0; i < 4; i++) {
    const ang = (i * Math.PI) / 2;
    const boss = addLampPart(`PrismEquatorBoss_${i}`, eqBossGeo, spunCopperMat, 1);
    boss.position.set(Math.cos(ang) * PRISM_EQUATOR_R, 0, -Math.sin(ang) * PRISM_EQUATOR_R);
    boss.rotation.y = Math.PI / 4;
  }

  addLampPart('PrismBottomDropFinial', bottomFinialGeo, spunCopperMat, 1);
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
