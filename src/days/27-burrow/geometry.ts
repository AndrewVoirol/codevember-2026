import * as THREE from 'three/webgpu';
import {
  ANT_COUNT,
  BACKWALL_Z,
  CAVITY_HALF_D,
  CAVITY_HALF_H,
  CAVITY_HALF_W,
  CAVITY_H,
  CAVITY_W,
  FRAME_OUTER_D,
  FRAME_OUTER_H,
  FRAME_OUTER_W,
  GLASS_CENTER_Z,
  GLASS_PANE_H,
  GLASS_PANE_THICK,
  GLASS_PANE_W,
  GRID_H,
  GRID_W,
  MAX_3D_GRAINS,
  MAX_DUST_MOTES,
  SAND_PIGMENTS,
} from './constants';
import { type DiggerAgent, type DustMote3D, type Grain3D } from './sandSim';

export interface BurrowMaterialsContract {
  oiledWalnut: THREE.Material;
  satinBrass: THREE.Material;
  agedBronze: THREE.Material;
  beveledGlass: THREE.Material;
  sandForeground: THREE.Material;
  sandBackwall: THREE.Material;
  sandCrestTop: THREE.Material;
  beetleChitin: THREE.Material;
  beetleAccent: THREE.Material;
  antGaster?: THREE.Material;
  antEye?: THREE.Material;
  mineralGrain3D?: THREE.Material;
  dustMote3D?: THREE.Material;
}

/** Read-only handle per worker ant for backward contract inspection & diagnostics */
export interface DiggerVisualRig {
  id: number;
  root: THREE.Object3D;
}

export interface TerrariumAssembly {
  group: THREE.Group;
  frontSandMesh: THREE.Mesh;
  backSandMesh: THREE.Mesh;
  frontBackwallMesh: THREE.Mesh;
  backBackwallMesh: THREE.Mesh;
  sandCrestMesh: THREE.Mesh;
  sandCrestGeo: THREE.BufferGeometry;
  frontGlassMesh: THREE.Mesh;
  backGlassMesh: THREE.Mesh;
  cavityHitMesh: THREE.Mesh;
  frameHitMeshes: THREE.Mesh[];
  diggers: DiggerVisualRig[];
  /** Instanced anatomical worker ant colony meshes (8 draw calls total for all 12 ants!) */
  antThoraxMesh: THREE.InstancedMesh;
  antGasterMesh: THREE.InstancedMesh;
  antHeadMesh: THREE.InstancedMesh;
  antEyesMesh: THREE.InstancedMesh;
  antMandibleMesh: THREE.InstancedMesh;
  antAntennaMesh: THREE.InstancedMesh;
  antLegMesh: THREE.InstancedMesh;
  antCarriedGrainMesh: THREE.InstancedMesh;
  /** Live 3D tumbling mineral sand granules & avalanche dust puffs inside glass cavity */
  tumblingGrainsMesh: THREE.InstancedMesh;
  dustMotesMesh: THREE.InstancedMesh;
  /** Updates the 3D top sand crest ribbon vertices & colors in-place (zero allocations). */
  updateSandCrest: (surfaceY: Float32Array, surfaceRgb: Float32Array) => void;
  /** Syncs all 12 articulated 3D worker ants with wall-hugging orthonormal frames (zero allocations). */
  updateWorkerAnts: (ants: readonly DiggerAgent[], elapsedSec: number) => void;
  /** Syncs live 3D tumbling sand granules & subtle avalanche dust puffs (zero allocations). */
  updateParticles3D: (
    grains: readonly Grain3D[],
    dust: readonly DustMote3D[],
    pigmentRgbLinear: Float32Array
  ) => void;
  dispose: () => void;
}

interface MeshBuilder {
  positions: number[];
  normals: number[];
  uvs: number[];
}

// Module-scope zero-GC scratch math objects for instanced ant & grain transforms
const _mAntRoot = new THREE.Matrix4();
const _mLocal = new THREE.Matrix4();
const _mPart = new THREE.Matrix4();
const _mHeadWorld = new THREE.Matrix4();
const _mBasis = new THREE.Matrix4();
const _qPart = new THREE.Quaternion();
const _qRoll = new THREE.Quaternion();
const _vPos = new THREE.Vector3();
const _vScale = new THREE.Vector3();
const _vFwd = new THREE.Vector3();
const _vUp = new THREE.Vector3();
const _vLat = new THREE.Vector3();
const _eRot = new THREE.Euler();
const _vZeroScale = new THREE.Vector3(0, 0, 0);
const _vOneScale = new THREE.Vector3(1, 1, 1);

function createMeshBuilder(): MeshBuilder {
  return { positions: [], normals: [], uvs: [] };
}

function pushTriangle(
  b: MeshBuilder,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  n: THREE.Vector3,
  u0 = 0,
  v0 = 0,
  u1 = 1,
  v1 = 0,
  u2 = 0,
  v2 = 1
): void {
  b.positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  b.normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z);
  b.uvs.push(u0, v0, u1, v1, u2, v2);
}

/**
 * Emits a convex polygon sorted strictly counter-clockwise around `outwardNormal`,
 * guaranteeing 100% outward-facing winding (`inwardTri === 0`) and crisp machined flats.
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
      const u = rel.dot(tangent);
      const vv = rel.dot(bitangent);
      return { v, u, vv, ang: Math.atan2(vv, u) };
    })
    .sort((a, bItem) => a.ang - bItem.ang);

  for (let i = 1; i < sorted.length - 1; i++) {
    pushTriangle(
      b,
      sorted[0].v,
      sorted[i].v,
      sorted[i + 1].v,
      n,
      sorted[0].u,
      sorted[0].vv,
      sorted[i].u,
      sorted[i].vv,
      sorted[i + 1].u,
      sorted[i + 1].vv
    );
  }
}

function finalizeBuilder(b: MeshBuilder): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(b.positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(b.normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uvs, 2));
  return geo;
}

/**
 * Watertight 26-face chamfered rectangular block (6 flat faces + 12 beveled 45-deg edges + 8 corners)
 * with strictly outward-wound triangles.
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
 * Builds a watertight outward-wound LatheGeometry along +Y from a bottom-to-top profile.
 */
export function createOutwardLathe(
  profile: Array<[number, number]>,
  segments = 24
): THREE.BufferGeometry {
  const pts = profile.map(([r, y]) => new THREE.Vector2(r, y));
  const geo = new THREE.LatheGeometry(pts, segments);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a watertight, strictly outward-wound bent & tapered tubular insect limb
 * (elbowed geniculate antenna, curved pincer mandible, or 2-segment femur+tibia leg!)
 * using parallel-transport cross-section rings and explicit outward radial winding verification.
 */
export function createBentTaperedLimbGeometry(
  nodes: Array<[number, number, number, number]>,
  radialSegs = 8
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const nNodes = nodes.length;
  const centers = nodes.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const radii = nodes.map(([, , , r]) => r);

  // Compute smooth node tangents
  const tangents: THREE.Vector3[] = [];
  for (let i = 0; i < nNodes; i++) {
    const t = new THREE.Vector3();
    if (i === 0) {
      t.subVectors(centers[1], centers[0]);
    } else if (i === nNodes - 1) {
      t.subVectors(centers[nNodes - 1], centers[nNodes - 2]);
    } else {
      const tPrev = new THREE.Vector3().subVectors(centers[i], centers[i - 1]).normalize();
      const tNext = new THREE.Vector3().subVectors(centers[i + 1], centers[i]).normalize();
      t.addVectors(tPrev, tNext);
    }
    if (t.lengthSq() < 1e-10) t.set(1, 0, 0);
    else t.normalize();
    tangents.push(t);
  }

  // Parallel-transport orthonormal frames (U, V) around each ring
  const framesU: THREE.Vector3[] = [];
  const framesV: THREE.Vector3[] = [];
  const initRef = Math.abs(tangents[0].y) < 0.85 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
  let prevU = new THREE.Vector3().crossVectors(initRef, tangents[0]).normalize();
  let prevV = new THREE.Vector3().crossVectors(tangents[0], prevU).normalize();
  framesU.push(prevU.clone());
  framesV.push(prevV.clone());

  for (let i = 1; i < nNodes; i++) {
    // Project previous U perpendicular to new tangent
    const curU = prevU.clone().addScaledVector(tangents[i], -prevU.dot(tangents[i]));
    if (curU.lengthSq() < 1e-8) {
      const ref = Math.abs(tangents[i].y) < 0.85 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
      curU.crossVectors(ref, tangents[i]);
    }
    curU.normalize();
    const curV = new THREE.Vector3().crossVectors(tangents[i], curU).normalize();
    framesU.push(curU);
    framesV.push(curV);
    prevU = curU;
    prevV = curV;
  }

  // Build ring vertices
  const rings: THREE.Vector3[][] = [];
  for (let i = 0; i < nNodes; i++) {
    const ring: THREE.Vector3[] = [];
    const c = centers[i];
    const r = radii[i];
    const u = framesU[i];
    const v = framesV[i];
    for (let s = 0; s < radialSegs; s++) {
      const ang = (s / radialSegs) * Math.PI * 2;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      ring.push(
        new THREE.Vector3(
          c.x + (u.x * ca + v.x * sa) * r,
          c.y + (u.y * ca + v.y * sa) * r,
          c.z + (u.z * ca + v.z * sa) * r
        )
      );
    }
    rings.push(ring);
  }

  const pushVerifiedTri = (
    p0: THREE.Vector3,
    p1: THREE.Vector3,
    p2: THREE.Vector3,
    outDir: THREE.Vector3
  ) => {
    const ab = new THREE.Vector3().subVectors(p1, p0);
    const ac = new THREE.Vector3().subVectors(p2, p0);
    const cross = new THREE.Vector3().crossVectors(ab, ac);
    if (cross.lengthSq() < 1e-16) return;
    if (cross.dot(outDir) >= 0) {
      cross.normalize();
      pushTriangle(b, p0, p1, p2, cross);
    } else {
      cross.negate().normalize();
      pushTriangle(b, p0, p2, p1, cross);
    }
  };

  // Emit tubular shaft quads with guaranteed outward radial winding
  for (let i = 0; i < nNodes - 1; i++) {
    const r0 = rings[i];
    const r1 = rings[i + 1];
    const midCenter = new THREE.Vector3().addVectors(centers[i], centers[i + 1]).multiplyScalar(0.5);
    for (let s = 0; s < radialSegs; s++) {
      const sn = (s + 1) % radialSegs;
      const p00 = r0[s];
      const p01 = r0[sn];
      const p10 = r1[s];
      const p11 = r1[sn];

      const out1 = new THREE.Vector3()
        .add(p00)
        .add(p01)
        .add(p11)
        .multiplyScalar(1 / 3)
        .sub(midCenter);
      const out2 = new THREE.Vector3()
        .add(p00)
        .add(p11)
        .add(p10)
        .multiplyScalar(1 / 3)
        .sub(midCenter);
      pushVerifiedTri(p00, p01, p11, out1);
      pushVerifiedTri(p00, p11, p10, out2);
    }
  }

  // Watertight start & end caps facing outward along -tangent[0] and +tangent[n-1]
  const capStartN = tangents[0].clone().negate();
  const capEndN = tangents[nNodes - 1].clone();
  for (let s = 0; s < radialSegs; s++) {
    const sn = (s + 1) % radialSegs;
    pushVerifiedTri(centers[0], rings[0][sn], rings[0][s], capStartN);
    pushVerifiedTri(centers[nNodes - 1], rings[nNodes - 1][s], rings[nNodes - 1][sn], capEndN);
  }

  const geo = finalizeBuilder(b);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Merges an array of non-indexed BufferGeometries (position/normal/uv) into one clean closed solid.
 */
function mergeSimpleGeometries(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const b = createMeshBuilder();
  for (const g of geos) {
    const nonIdx = g.index ? g.toNonIndexed() : g;
    const pos = nonIdx.attributes.position as THREE.BufferAttribute;
    const norm = nonIdx.attributes.normal as THREE.BufferAttribute;
    const uvAttr = nonIdx.attributes.uv as THREE.BufferAttribute | undefined;
    for (let i = 0; i < pos.count; i++) {
      b.positions.push(pos.getX(i), pos.getY(i), pos.getZ(i));
      b.normals.push(norm.getX(i), norm.getY(i), norm.getZ(i));
      if (uvAttr) b.uvs.push(uvAttr.getX(i), uvAttr.getY(i));
      else b.uvs.push(0, 0);
    }
    if (nonIdx !== g) nonIdx.dispose();
    g.dispose();
  }
  return finalizeBuilder(b);
}

/**
 * Builds a crisp, irregular faceted quartz/mineral sand granule (watertight outward-wound polyhedron)
 * for both ant-carried grains and 3D tumbling avalanche grains!
 */
export function createFacetedMineralGrainGeometry(radius = 0.0042): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(radius, 0);
  const nonIdx = geo.index ? geo.toNonIndexed() : geo;
  if (nonIdx !== geo) geo.dispose();
  const pos = nonIdx.attributes.position as THREE.BufferAttribute;
  // Slightly irregular crystal proportions (elongated angular silica grain, strictly outward-wound!)
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const jitter = 0.86 + 0.24 * Math.abs(Math.sin(x * 380 + y * 510 - z * 290));
    pos.setXYZ(i, x * 1.18 * jitter, y * 0.92 * jitter, z * 0.84 * jitter);
  }
  nonIdx.computeVertexNormals();
  return nonIdx;
}

/**
 * Builds a closed outward-wound soft puff polyhedron for subtle 3D avalanche mineral dust motes.
 */
export function createSoftDustPuffGeometry(radius = 0.018): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(radius, 1);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a rectangular mitered picture-frame satin-brass glass retaining bezel in the XY plane
 * framing a clear viewing window `[-innerHalfW, +innerHalfW] x [-innerHalfH, +innerHalfH]`
 * with a crisp sloped inner optical chamfer down onto the beveled glass pane.
 */
export function createRectangularBezelGeometry(
  innerHalfW: number,
  innerHalfH: number,
  outerHalfW: number,
  outerHalfH: number,
  zFront: number,
  zBack: number,
  innerStep = 0.012
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const midW = innerHalfW + innerStep;
  const midH = innerHalfH + innerStep;

  const bars: Array<{
    fOA: [number, number];
    fOB: [number, number];
    fIA: [number, number];
    fIB: [number, number];
    bOA: [number, number];
    bOB: [number, number];
    bIA: [number, number];
    bIB: [number, number];
    outDir: [number, number];
    inDir: [number, number];
  }> = [
    {
      fOA: [-outerHalfW, outerHalfH],
      fOB: [outerHalfW, outerHalfH],
      fIA: [-midW, midH],
      fIB: [midW, midH],
      bOA: [-outerHalfW, outerHalfH],
      bOB: [outerHalfW, outerHalfH],
      bIA: [-innerHalfW, innerHalfH],
      bIB: [innerHalfW, innerHalfH],
      outDir: [0, 1],
      inDir: [0, -1],
    },
    {
      fOA: [outerHalfW, -outerHalfH],
      fOB: [-outerHalfW, -outerHalfH],
      fIA: [midW, -midH],
      fIB: [-midW, -midH],
      bOA: [outerHalfW, -outerHalfH],
      bOB: [-outerHalfW, -outerHalfH],
      bIA: [innerHalfW, -innerHalfH],
      bIB: [-innerHalfW, -innerHalfH],
      outDir: [0, -1],
      inDir: [0, 1],
    },
    {
      fOA: [outerHalfW, outerHalfH],
      fOB: [outerHalfW, -outerHalfH],
      fIA: [midW, midH],
      fIB: [midW, -midH],
      bOA: [outerHalfW, outerHalfH],
      bOB: [outerHalfW, -outerHalfH],
      bIA: [innerHalfW, innerHalfH],
      bIB: [innerHalfW, -innerHalfH],
      outDir: [1, 0],
      inDir: [-1, 0],
    },
    {
      fOA: [-outerHalfW, -outerHalfH],
      fOB: [-outerHalfW, outerHalfH],
      fIA: [-midW, -midH],
      fIB: [-midW, midH],
      bOA: [-outerHalfW, -outerHalfH],
      bOB: [-outerHalfW, outerHalfH],
      bIA: [-innerHalfW, -innerHalfH],
      bIB: [-innerHalfW, innerHalfH],
      outDir: [-1, 0],
      inDir: [1, 0],
    },
  ];

  for (const bar of bars) {
    const fOA = new THREE.Vector3(bar.fOA[0], bar.fOA[1], zFront);
    const fOB = new THREE.Vector3(bar.fOB[0], bar.fOB[1], zFront);
    const fIA = new THREE.Vector3(bar.fIA[0], bar.fIA[1], zFront);
    const fIB = new THREE.Vector3(bar.fIB[0], bar.fIB[1], zFront);

    const bOA = new THREE.Vector3(bar.bOA[0], bar.bOA[1], zBack);
    const bOB = new THREE.Vector3(bar.bOB[0], bar.bOB[1], zBack);
    const bIA = new THREE.Vector3(bar.bIA[0], bar.bIA[1], zBack);
    const bIB = new THREE.Vector3(bar.bIB[0], bar.bIB[1], zBack);

    pushConvexFace(b, [fOA, fOB, fIB, fIA], new THREE.Vector3(0, 0, 1));
    pushConvexFace(b, [bOA, bOB, bIB, bIA], new THREE.Vector3(0, 0, -1));
    pushConvexFace(b, [bOA, bOB, fOB, fOA], new THREE.Vector3(bar.outDir[0], bar.outDir[1], 0));
    const slopeNorm = new THREE.Vector3(
      bar.inDir[0] * (zFront - zBack),
      bar.inDir[1] * (zFront - zBack),
      innerStep
    ).normalize();
    pushConvexFace(b, [fIA, fIB, bIB, bIA], slopeNorm);
  }

  return finalizeBuilder(b);
}

/**
 * Builds a watertight beveled L-bracket corner strap arm.
 */
export function createCornerBracketArmGeometry(
  armLen: number,
  armWidth: number,
  halfThick: number
): THREE.BufferGeometry {
  const geo = createChamferedBlockGeometry(armLen * 0.5, armWidth * 0.5, halfThick, 0.0028);
  geo.translate(armLen * 0.5, armWidth * 0.5, 0);
  return geo;
}

/**
 * Builds an oriented sand-slice plane quad directly in terrarium-local coordinates
 * so BOTH front (`facingSign = +1`) and rear (`facingSign = -1`) quads map `uv.x` identically
 * to terrarium world `X` with exact outward normals `(0, 0, ±1)` and zero UV mirror inversion!
 */
export function createAlignedCavityQuad(
  halfW: number,
  halfH: number,
  zPos: number,
  facingSign: 1 | -1
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const bl = new THREE.Vector3(-halfW, -halfH, zPos);
  const br = new THREE.Vector3(halfW, -halfH, zPos);
  const tr = new THREE.Vector3(halfW, halfH, zPos);
  const tl = new THREE.Vector3(-halfW, halfH, zPos);
  const n = new THREE.Vector3(0, 0, facingSign);

  if (facingSign === 1) {
    pushTriangle(b, bl, br, tr, n, 0, 0, 1, 0, 1, 1);
    pushTriangle(b, bl, tr, tl, n, 0, 0, 1, 1, 0, 1);
  } else {
    pushTriangle(b, br, bl, tl, n, 1, 0, 0, 0, 0, 1);
    pushTriangle(b, br, tl, tr, n, 1, 0, 0, 1, 1, 1);
  }

  return finalizeBuilder(b);
}

/**
 * Builds the live 3D Top Sand Crest Ribbon Geometry spanning `X in [-CAVITY_HALF_W, +CAVITY_HALF_W]`
 * and `Z in [-CAVITY_HALF_D, +CAVITY_HALF_D]` (`cols x depthSegs`) with outward `+Y` winding.
 */
export function createSandCrestRibbonGeometry(
  cols = GRID_W,
  depthSegs = 4
): THREE.BufferGeometry {
  const rows = depthSegs + 1;
  const vertCount = cols * rows;
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const colors = new Float32Array(vertCount * 3);

  for (let r = 0; r < rows; r++) {
    const v = r / depthSegs;
    const z = -CAVITY_HALF_D + v * (CAVITY_HALF_D * 2);
    for (let c = 0; c < cols; c++) {
      const u = c / (cols - 1);
      const x = -CAVITY_HALF_W + u * CAVITY_W;
      const vi = r * cols + c;
      positions[vi * 3] = x;
      positions[vi * 3 + 1] = 0.34;
      positions[vi * 3 + 2] = z;
      normals[vi * 3] = 0;
      normals[vi * 3 + 1] = 1;
      normals[vi * 3 + 2] = 0;
      uvs[vi * 2] = u;
      uvs[vi * 2 + 1] = v;
      colors[vi * 3] = 0.7;
      colors[vi * 3 + 1] = 0.65;
      colors[vi * 3 + 2] = 0.56;
    }
  }

  const indices: number[] = [];
  for (let r = 0; r < depthSegs; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const i00 = r * cols + c;
      const i10 = (r + 1) * cols + c;
      const i11 = (r + 1) * cols + (c + 1);
      const i01 = r * cols + (c + 1);
      indices.push(i00, i10, i11);
      indices.push(i00, i11, i01);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setIndex(indices);
  return geo;
}

/**
 * Builds anatomically faithful 3D sculpted worker ant geometries (local +X = forward,
 * +Y = dorsal up, +Z = right lateral; 100% closed watertight outward-wound meshes!):
 * - Slender humped mesosoma thorax + pinched petiole waist nodes
 * - Segmented teardrop gaster with 3 delicate tergite rings
 * - Cordate ant head + twin glossy compound eye domes
 * - Curved inward-hooking pincer mandibles
 * - Elbowed geniculate antennae (scape + sharply bent funiculus)
 * - Bent 2-segment femur + tibia/tarsus legs
 */
function buildWorkerAntGeometries(): {
  thorax: THREE.BufferGeometry;
  gaster: THREE.BufferGeometry;
  head: THREE.BufferGeometry;
  eyes: THREE.BufferGeometry;
  mandible: THREE.BufferGeometry;
  antenna: THREE.BufferGeometry;
  leg: THREE.BufferGeometry;
  carriedGrain: THREE.BufferGeometry;
} {
  // 1. Slender humped Mesosoma + Pinched Petiole & Post-Petiole Waist (rotated so lathe +Y -> +X!)
  const thorax = createOutwardLathe(
    [
      [0.0004, -0.0145],
      [0.0017, -0.0138],
      [0.0028, -0.0124], // Post-petiole waist node
      [0.0015, -0.0108], // Pinched waist!
      [0.0026, -0.0092], // Petiole node
      [0.0014, -0.0074], // Propodeum pinch
      [0.0038, -0.0046], // Propodeum slope
      [0.0036, -0.0005], // Mesonotum saddle
      [0.0052, 0.0042],  // Pronotum shoulder hump
      [0.0046, 0.0078],
      [0.0023, 0.0104],  // Narrow neck joint
      [0.0004, 0.0110],
    ],
    14
  );
  thorax.rotateZ(-Math.PI * 0.5);
  thorax.scale(1.0, 0.94, 0.84);
  thorax.translate(0, 0.0038, 0);
  thorax.computeVertexNormals();

  // 2. Segmented Teardrop Gaster (pivoted at petiole socket x=0, extending to -X with 3 tergite rings!)
  const gaster = createOutwardLathe(
    [
      [0.0004, -0.0205], // Pointed rear tip
      [0.0034, -0.0182],
      [0.0058, -0.0148],
      [0.0054, -0.0138], // Tergite groove 3
      [0.0073, -0.0105],
      [0.0069, -0.0095], // Tergite groove 2
      [0.0077, -0.0062], // Broadest gaster belly
      [0.0070, -0.0048], // Tergite groove 1
      [0.0056, -0.0022],
      [0.0024, -0.0005],
      [0.0004, 0.0002],
    ],
    16
  );
  gaster.rotateZ(-Math.PI * 0.5);
  gaster.scale(1.0, 0.92, 0.88);
  gaster.computeVertexNormals();

  // 3. Cordate Ant Head Capsule (pivoted at cervix x=0, extending to +X)
  const head = createOutwardLathe(
    [
      [0.0004, -0.0012],
      [0.0036, 0.0002],
      [0.0053, 0.0026], // Broad rear temple cheeks
      [0.0051, 0.0058],
      [0.0038, 0.0086], // Tapered clypeus
      [0.0019, 0.0102],
      [0.0004, 0.0108],
    ],
    14
  );
  head.rotateZ(-Math.PI * 0.5);
  head.scale(1.0, 0.84, 1.04);
  head.computeVertexNormals();

  // 4. Twin Glossy Compound Eye Beads (merged cleanly in head-local coordinates!)
  const eyeL = new THREE.IcosahedronGeometry(0.0017, 1);
  eyeL.scale(1.15, 0.95, 0.72);
  eyeL.translate(0.0048, 0.0018, 0.0044);
  const eyeR = new THREE.IcosahedronGeometry(0.0017, 1);
  eyeR.scale(1.15, 0.95, 0.72);
  eyeR.translate(0.0048, 0.0018, -0.0044);
  const eyes = mergeSimpleGeometries([eyeL, eyeR]);

  // 5. Curved Inward-Hooking Pincer Mandible (+Z side canonical; mirrored via rotation for -Z!)
  const mandible = createBentTaperedLimbGeometry(
    [
      [0.0, 0.0, 0.0, 0.0013],
      [0.0028, -0.0003, 0.0008, 0.0012],
      [0.0056, -0.0008, 0.0002, 0.0009],
      [0.0074, -0.0011, -0.0012, 0.0004], // Hooked chewing tip!
    ],
    7
  );

  // 6. Elbowed Geniculate Ant Antenna (Scape -> Sharp Elbow -> Sweeping Funiculus!)
  const antenna = createBentTaperedLimbGeometry(
    [
      [0.0, 0.0, 0.0, 0.00075],
      [0.0034, 0.0026, 0.0022, 0.00068],
      [0.0062, 0.0046, 0.0038, 0.00072], // Signature geniculate elbow joint!
      [0.0108, 0.0034, 0.0048, 0.00058],
      [0.0154, 0.0010, 0.0044, 0.00046],
      [0.0186, -0.0012, 0.0036, 0.00032], // Sensory tip
    ],
    6
  );

  // 7. Articulated 2-Segment Ant Leg (Coxa -> High Arched Femur Knee -> Angled Tibia/Tarsus Foot!)
  // Canonical leg extends outward along +Z and arches up then down onto floor plane (y ≈ -0.009)
  const leg = createBentTaperedLimbGeometry(
    [
      [0.0, 0.0, 0.0, 0.0011],           // Coxa socket
      [0.0008, 0.0034, 0.0052, 0.00095], // Elevated femur knee!
      [0.0016, -0.0022, 0.0086, 0.00075], // Tibia mid-shaft
      [0.0026, -0.0082, 0.0102, 0.00052], // Tarsus ankle
      [0.0042, -0.0094, 0.0110, 0.00036], // Planted claw foot!
    ],
    6
  );

  // 8. Faceted Carried Mineral Sand Granule held prominently between mandibles
  const carriedGrain = createFacetedMineralGrainGeometry(0.0046);

  return { thorax, gaster, head, eyes, mandible, antenna, leg, carriedGrain };
}

/**
 * Constructs the complete museum-grade Day 27 - Burrow Glass-Walled Stratified Sand-Slice Terrarium:
 * - Oiled dark raw-umber walnut structural stiles, flanged museum base plinth & open-slot top cap
 * - Satin-brass wrap-around corner brackets, 24 slotted dome-head brass screws, glass retaining bezels,
 *   open top hopper lips, side breather grommets, brass leveling feet & tactile depth gauge ticks
 * - Front & back 45-deg beveled clear observation glass panes
 * - Front & back 3D parallax sand-bed cross-section planes + physically separated recessed tunnel back-walls
 * - Live 3D Top Sand Crest Ribbon Mesh bridging across the glass slice thickness
 * - 12 anatomically sculpted 3D worker ants (`InstancedMesh`, 8 draw calls total!)
 * - Live 3D tumbling mineral sand granules (`420` instances) & soft avalanche dust puffs (`72` instances)
 */
export function buildTerrariumAssembly(materials: BurrowMaterialsContract): TerrariumAssembly {
  const group = new THREE.Group();
  group.name = 'SandSliceTerrariumAssembly';

  const geometries: THREE.BufferGeometry[] = [];
  const frameHitMeshes: THREE.Mesh[] = [];
  const extraMats: THREE.Material[] = [];

  const halfOuterW = FRAME_OUTER_W * 0.5; // 0.88
  const halfOuterH = FRAME_OUTER_H * 0.5; // 0.68
  const halfOuterD = FRAME_OUTER_D * 0.5; // 0.12

  // ---------------------------------------------------------------------------
  // 1. Oiled Dark Raw-Umber Walnut Structural Frame
  // ---------------------------------------------------------------------------
  const stileHalfW = (halfOuterW - CAVITY_HALF_W) * 0.5; // 0.05
  const stileCenterX = CAVITY_HALF_W + stileHalfW; // 0.83
  const stileHalfH = 0.635;
  const stileHalfD = halfOuterD - 0.002; // 0.118

  const stileGeo = createChamferedBlockGeometry(stileHalfW, stileHalfH, stileHalfD, 0.011);
  geometries.push(stileGeo);

  for (const sx of [-1, 1]) {
    const stile = new THREE.Mesh(stileGeo, materials.oiledWalnut);
    stile.name = sx < 0 ? 'WalnutStileLeft' : 'WalnutStileRight';
    stile.position.set(sx * stileCenterX, 0.008, 0);
    stile.castShadow = true;
    stile.receiveShadow = true;
    group.add(stile);
    frameHitMeshes.push(stile);
  }

  // Heavy walnut base rail + stepped museum plinth footing
  const baseRailGeo = createChamferedBlockGeometry(halfOuterW + 0.004, 0.043, halfOuterD + 0.002, 0.01);
  const basePlinthGeo = createChamferedBlockGeometry(halfOuterW + 0.036, 0.019, halfOuterD + 0.028, 0.008);
  geometries.push(baseRailGeo, basePlinthGeo);

  const baseRail = new THREE.Mesh(baseRailGeo, materials.oiledWalnut);
  baseRail.name = 'WalnutBaseRail';
  baseRail.position.set(0, -CAVITY_HALF_H - 0.041, 0);
  baseRail.castShadow = true;
  baseRail.receiveShadow = true;
  group.add(baseRail);
  frameHitMeshes.push(baseRail);

  const basePlinth = new THREE.Mesh(basePlinthGeo, materials.oiledWalnut);
  basePlinth.name = 'WalnutBasePlinth';
  basePlinth.position.set(0, -halfOuterH + 0.008, 0);
  basePlinth.castShadow = true;
  basePlinth.receiveShadow = true;
  group.add(basePlinth);
  frameHitMeshes.push(basePlinth);

  // Open-slot top walnut observation cap (front & back rails flanking open top pouring slot!)
  const topSlotHalfZ = 0.044;
  const topRailHalfD = (stileHalfD - topSlotHalfZ) * 0.5; // 0.037
  const topRailCenterZ = topSlotHalfZ + topRailHalfD; // 0.081
  const topRailGeo = createChamferedBlockGeometry(halfOuterW + 0.002, 0.039, topRailHalfD, 0.008);
  const topEndCapGeo = createChamferedBlockGeometry(0.058, 0.039, stileHalfD, 0.008);
  geometries.push(topRailGeo, topEndCapGeo);

  for (const sz of [-1, 1]) {
    const topRail = new THREE.Mesh(topRailGeo, materials.oiledWalnut);
    topRail.name = sz > 0 ? 'WalnutTopRailFront' : 'WalnutTopRailBack';
    topRail.position.set(0, CAVITY_HALF_H + 0.039, sz * topRailCenterZ);
    topRail.castShadow = true;
    topRail.receiveShadow = true;
    group.add(topRail);
    frameHitMeshes.push(topRail);
  }
  for (const sx of [-1, 1]) {
    const cap = new THREE.Mesh(topEndCapGeo, materials.oiledWalnut);
    cap.name = sx < 0 ? 'WalnutTopEndLeft' : 'WalnutTopEndRight';
    cap.position.set(sx * (halfOuterW - 0.056), CAVITY_HALF_H + 0.039, 0);
    cap.castShadow = true;
    cap.receiveShadow = true;
    group.add(cap);
    frameHitMeshes.push(cap);
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Satin-Brass Glass Retaining Bezels, Hopper Lips, Corner Wraps & Slotted Screws
  // ---------------------------------------------------------------------------
  const bezelGeo = createRectangularBezelGeometry(
    CAVITY_HALF_W - 0.004,
    CAVITY_HALF_H - 0.004,
    CAVITY_HALF_W + 0.034,
    CAVITY_HALF_H + 0.032,
    stileHalfD - 0.003,
    GLASS_CENTER_Z + GLASS_PANE_THICK * 0.5 - 0.002,
    0.011
  );
  geometries.push(bezelGeo);

  const frontBezel = new THREE.Mesh(bezelGeo, materials.satinBrass);
  frontBezel.name = 'BrassGlassBezelFront';
  frontBezel.castShadow = true;
  frontBezel.receiveShadow = true;
  group.add(frontBezel);
  frameHitMeshes.push(frontBezel);

  const backBezel = new THREE.Mesh(bezelGeo, materials.satinBrass);
  backBezel.name = 'BrassGlassBezelBack';
  backBezel.rotation.y = Math.PI;
  backBezel.castShadow = true;
  backBezel.receiveShadow = true;
  group.add(backBezel);
  frameHitMeshes.push(backBezel);

  const hopperLipGeo = createChamferedBlockGeometry(CAVITY_HALF_W - 0.004, 0.011, 0.008, 0.003);
  geometries.push(hopperLipGeo);
  for (const sz of [-1, 1]) {
    const lip = new THREE.Mesh(hopperLipGeo, materials.satinBrass);
    lip.name = sz > 0 ? 'BrassHopperLipFront' : 'BrassHopperLipBack';
    lip.position.set(0, CAVITY_HALF_H + 0.072, sz * (topSlotHalfZ + 0.004));
    lip.castShadow = true;
    lip.receiveShadow = true;
    group.add(lip);
  }

  const tickMajorGeo = createChamferedBlockGeometry(0.011, 0.0026, 0.0022, 0.0007);
  const tickMinorGeo = createChamferedBlockGeometry(0.0065, 0.002, 0.002, 0.0006);
  geometries.push(tickMajorGeo, tickMinorGeo);
  for (let k = 0; k <= 8; k++) {
    const ty = -CAVITY_HALF_H + 0.1 + (k / 8) * (CAVITY_H - 0.2);
    const isMajor = k % 2 === 0;
    const tMesh = new THREE.Mesh(isMajor ? tickMajorGeo : tickMinorGeo, materials.agedBronze);
    tMesh.name = `DepthGaugeTick_${k}`;
    tMesh.position.set(CAVITY_HALF_W + (isMajor ? 0.019 : 0.016), ty, stileHalfD - 0.0015);
    group.add(tMesh);
  }

  const hArmGeo = createCornerBracketArmGeometry(0.155, 0.044, 0.005);
  const vArmGeo = createCornerBracketArmGeometry(0.044, 0.145, 0.005);
  const sideWrapGeo = createChamferedBlockGeometry(0.0055, 0.0725, stileHalfD + 0.004, 0.0028);
  geometries.push(hArmGeo, vArmGeo, sideWrapGeo);

  const screwDomeGeo = createOutwardLathe(
    [
      [0.0005, 0.0],
      [0.0145, 0.0],
      [0.0145, 0.0038],
      [0.012, 0.0072],
      [0.0065, 0.0092],
      [0.0005, 0.0098],
    ],
    18
  );
  const screwSlotGeo = createChamferedBlockGeometry(0.0115, 0.0018, 0.0026, 0.0005);
  geometries.push(screwDomeGeo, screwSlotGeo);

  const addSlottedScrew = (
    name: string,
    pos: [number, number, number],
    rx: number,
    ry: number,
    rz: number,
    slotAngle: number
  ) => {
    const sGrp = new THREE.Group();
    sGrp.name = name;
    sGrp.position.set(pos[0], pos[1], pos[2]);
    sGrp.rotation.set(rx, ry, rz);

    const dome = new THREE.Mesh(screwDomeGeo, materials.satinBrass);
    dome.castShadow = true;
    dome.receiveShadow = true;
    sGrp.add(dome);

    const slot = new THREE.Mesh(screwSlotGeo, materials.agedBronze);
    slot.position.set(0, 0.0084, 0);
    slot.rotation.y = slotAngle;
    sGrp.add(slot);

    group.add(sGrp);
  };

  const cornerYTop = CAVITY_HALF_H + 0.078;
  const cornerYBot = -0.651;
  const bracketZ = stileHalfD + 0.004;
  const sideWrapX = halfOuterW + 0.003;

  let screwIdx = 0;
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const cy = sy > 0 ? cornerYTop : cornerYBot;
      const cx = sx * (halfOuterW + 0.002);

      for (const sz of [-1, 1]) {
        const bGrp = new THREE.Group();
        bGrp.name = `BrassCornerBracket_${sx}_${sy}_${sz}`;
        bGrp.position.set(cx, cy, sz * bracketZ);

        const hArm = new THREE.Mesh(hArmGeo, materials.satinBrass);
        hArm.position.set(sx > 0 ? -0.155 : 0, sy > 0 ? -0.044 : 0, 0);
        hArm.castShadow = true;
        hArm.receiveShadow = true;
        bGrp.add(hArm);

        const vArm = new THREE.Mesh(vArmGeo, materials.satinBrass);
        vArm.position.set(sx > 0 ? -0.044 : 0, sy > 0 ? -0.145 : 0, 0);
        vArm.castShadow = true;
        vArm.receiveShadow = true;
        bGrp.add(vArm);

        group.add(bGrp);

        const faceZ = sz * (bracketZ + 0.0045);
        const rxScrew = sz > 0 ? Math.PI * 0.5 : -Math.PI * 0.5;
        addSlottedScrew(
          `CornerScrewH_${screwIdx++}`,
          [cx - sx * 0.112, cy - sy * 0.022, faceZ],
          rxScrew,
          0,
          0,
          0.42 + screwIdx * 0.73
        );
        addSlottedScrew(
          `CornerScrewV_${screwIdx++}`,
          [cx - sx * 0.022, cy - sy * 0.106, faceZ],
          rxScrew,
          0,
          0,
          1.18 + screwIdx * 0.57
        );
      }

      const sideWrap = new THREE.Mesh(sideWrapGeo, materials.satinBrass);
      sideWrap.name = `BrassSideWrap_${sx}_${sy}`;
      sideWrap.position.set(sx * sideWrapX, cy - sy * 0.0725, 0);
      sideWrap.castShadow = true;
      sideWrap.receiveShadow = true;
      group.add(sideWrap);

      const rzSide = sx > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;
      addSlottedScrew(
        `SideScrewA_${screwIdx++}`,
        [sx * (sideWrapX + 0.005), cy - sy * 0.04, 0.056],
        0,
        0,
        rzSide,
        0.85 + screwIdx * 0.41
      );
      addSlottedScrew(
        `SideScrewB_${screwIdx++}`,
        [sx * (sideWrapX + 0.005), cy - sy * 0.105, -0.056],
        0,
        0,
        rzSide,
        2.1 + screwIdx * 0.63
      );
    }
  }

  const grommetRingGeo = createOutwardLathe(
    [
      [0.0005, 0.0],
      [0.036, 0.0],
      [0.036, 0.006],
      [0.03, 0.011],
      [0.021, 0.011],
      [0.019, 0.004],
      [0.0005, 0.004],
    ],
    24
  );
  const grommetCoreGeo = createOutwardLathe(
    [
      [0.0005, 0.002],
      [0.02, 0.002],
      [0.02, 0.007],
      [0.0005, 0.007],
    ],
    20
  );
  geometries.push(grommetRingGeo, grommetCoreGeo);

  for (const sx of [-1, 1]) {
    const gGrp = new THREE.Group();
    gGrp.name = sx < 0 ? 'SideBreatherLeft' : 'SideBreatherRight';
    gGrp.position.set(sx * (halfOuterW + 0.001), 0.34, 0);
    gGrp.rotation.z = sx > 0 ? -Math.PI * 0.5 : Math.PI * 0.5;

    const ring = new THREE.Mesh(grommetRingGeo, materials.satinBrass);
    ring.castShadow = true;
    ring.receiveShadow = true;
    gGrp.add(ring);

    const core = new THREE.Mesh(grommetCoreGeo, materials.agedBronze);
    gGrp.add(core);
    group.add(gGrp);
  }

  const footGeo = createOutwardLathe(
    [
      [0.0005, -0.024],
      [0.026, -0.024],
      [0.031, -0.017],
      [0.028, -0.008],
      [0.035, 0.0],
      [0.0005, 0.0],
    ],
    22
  );
  geometries.push(footGeo);

  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const foot = new THREE.Mesh(footGeo, materials.satinBrass);
      foot.name = `BrassLevelFoot_${sx}_${sz}`;
      foot.position.set(sx * 0.78, -halfOuterH - 0.01, sz * 0.084);
      foot.castShadow = true;
      foot.receiveShadow = true;
      group.add(foot);
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Stratified Sand Cross-Section Layers + 3D Top Crest Ribbon
  // ---------------------------------------------------------------------------
  const frontSandGeo = createAlignedCavityQuad(CAVITY_HALF_W, CAVITY_HALF_H, CAVITY_HALF_D, 1);
  const backSandGeo = createAlignedCavityQuad(CAVITY_HALF_W, CAVITY_HALF_H, -CAVITY_HALF_D, -1);
  const frontBackwallGeo = createAlignedCavityQuad(CAVITY_HALF_W, CAVITY_HALF_H, -BACKWALL_Z, 1);
  const backBackwallGeo = createAlignedCavityQuad(CAVITY_HALF_W, CAVITY_HALF_H, BACKWALL_Z, -1);
  const sandCrestGeo = createSandCrestRibbonGeometry(GRID_W, 4);
  geometries.push(frontSandGeo, backSandGeo, frontBackwallGeo, backBackwallGeo, sandCrestGeo);

  const frontSandMesh = new THREE.Mesh(frontSandGeo, materials.sandForeground);
  frontSandMesh.name = 'SandCrossSectionFront';
  frontSandMesh.receiveShadow = true;
  group.add(frontSandMesh);

  const backSandMesh = new THREE.Mesh(backSandGeo, materials.sandForeground);
  backSandMesh.name = 'SandCrossSectionBack';
  backSandMesh.receiveShadow = true;
  group.add(backSandMesh);

  const frontBackwallMesh = new THREE.Mesh(frontBackwallGeo, materials.sandBackwall);
  frontBackwallMesh.name = 'TunnelBackwallFront';
  group.add(frontBackwallMesh);

  const backBackwallMesh = new THREE.Mesh(backBackwallGeo, materials.sandBackwall);
  backBackwallMesh.name = 'TunnelBackwallBack';
  group.add(backBackwallMesh);

  const sandCrestMesh = new THREE.Mesh(sandCrestGeo, materials.sandCrestTop);
  sandCrestMesh.name = 'SandCrestTopRibbon';
  sandCrestMesh.receiveShadow = true;
  group.add(sandCrestMesh);

  // ---------------------------------------------------------------------------
  // 4. 12 Anatomically Sculpted 3D Worker Ants (InstancedMesh: 8 Draw Calls Total!)
  // ---------------------------------------------------------------------------
  const antGeos = buildWorkerAntGeometries();
  geometries.push(
    antGeos.thorax,
    antGeos.gaster,
    antGeos.head,
    antGeos.eyes,
    antGeos.mandible,
    antGeos.antenna,
    antGeos.leg,
    antGeos.carriedGrain
  );

  // Per-ant carried grain color attribute (linear RGB + glint)
  const carriedColorArr = new Float32Array(ANT_COUNT * 4);
  const carriedColorAttr = new THREE.InstancedBufferAttribute(carriedColorArr, 4);
  antGeos.carriedGrain.setAttribute('aGrainColor', carriedColorAttr);

  const gasterMat = materials.antGaster ?? materials.beetleChitin;
  const eyeMat = materials.antEye ?? materials.agedBronze;
  const grainMat = materials.mineralGrain3D ?? materials.beetleAccent;
  const dustMat = materials.dustMote3D ?? materials.beveledGlass;

  const antThoraxMesh = new THREE.InstancedMesh(antGeos.thorax, materials.beetleChitin, ANT_COUNT);
  antThoraxMesh.name = 'WorkerAntThoraxColony';
  antThoraxMesh.frustumCulled = false;
  group.add(antThoraxMesh);

  const antGasterMesh = new THREE.InstancedMesh(antGeos.gaster, gasterMat, ANT_COUNT);
  antGasterMesh.name = 'WorkerAntGasterColony';
  antGasterMesh.frustumCulled = false;
  group.add(antGasterMesh);

  const antHeadMesh = new THREE.InstancedMesh(antGeos.head, materials.beetleChitin, ANT_COUNT);
  antHeadMesh.name = 'WorkerAntHeadColony';
  antHeadMesh.frustumCulled = false;
  group.add(antHeadMesh);

  const antEyesMesh = new THREE.InstancedMesh(antGeos.eyes, eyeMat, ANT_COUNT);
  antEyesMesh.name = 'WorkerAntEyesColony';
  antEyesMesh.frustumCulled = false;
  group.add(antEyesMesh);

  const antMandibleMesh = new THREE.InstancedMesh(
    antGeos.mandible,
    materials.beetleAccent,
    ANT_COUNT * 2
  );
  antMandibleMesh.name = 'WorkerAntMandiblesColony';
  antMandibleMesh.frustumCulled = false;
  group.add(antMandibleMesh);

  const antAntennaMesh = new THREE.InstancedMesh(
    antGeos.antenna,
    materials.beetleAccent,
    ANT_COUNT * 2
  );
  antAntennaMesh.name = 'WorkerAntAntennaeColony';
  antAntennaMesh.frustumCulled = false;
  group.add(antAntennaMesh);

  const antLegMesh = new THREE.InstancedMesh(antGeos.leg, materials.beetleAccent, ANT_COUNT * 6);
  antLegMesh.name = 'WorkerAntLegsColony';
  antLegMesh.frustumCulled = false;
  group.add(antLegMesh);

  const antCarriedGrainMesh = new THREE.InstancedMesh(antGeos.carriedGrain, grainMat, ANT_COUNT);
  antCarriedGrainMesh.name = 'WorkerAntCarriedGrains';
  antCarriedGrainMesh.frustumCulled = false;
  group.add(antCarriedGrainMesh);

  // Lightweight logical handles for contract parity (`assembly.diggers.length === DIGGER_COUNT`)
  const diggers: DiggerVisualRig[] = [];
  for (let i = 0; i < ANT_COUNT; i++) {
    const marker = new THREE.Object3D();
    marker.name = `WorkerAntHandle_${i}`;
    diggers.push({ id: i, root: marker });
  }

  // ---------------------------------------------------------------------------
  // 5. Live 3D Tumbling Mineral Sand Granules & Subtle Avalanche Dust Motes
  // ---------------------------------------------------------------------------
  const tumblingGrainGeo = createFacetedMineralGrainGeometry(0.0036);
  const tumblingColorArr = new Float32Array(MAX_3D_GRAINS * 4);
  const tumblingColorAttr = new THREE.InstancedBufferAttribute(tumblingColorArr, 4);
  tumblingGrainGeo.setAttribute('aGrainColor', tumblingColorAttr);
  geometries.push(tumblingGrainGeo);

  const tumblingGrainsMesh = new THREE.InstancedMesh(tumblingGrainGeo, grainMat, MAX_3D_GRAINS);
  tumblingGrainsMesh.name = 'TumblingSandGranules3D';
  tumblingGrainsMesh.count = 0;
  tumblingGrainsMesh.frustumCulled = false;
  group.add(tumblingGrainsMesh);

  const dustGeo = createSoftDustPuffGeometry(0.022);
  const dustColorArr = new Float32Array(MAX_DUST_MOTES * 4);
  const dustColorAttr = new THREE.InstancedBufferAttribute(dustColorArr, 4);
  dustGeo.setAttribute('aDustColor', dustColorAttr);
  geometries.push(dustGeo);

  const dustMotesMesh = new THREE.InstancedMesh(dustGeo, dustMat, MAX_DUST_MOTES);
  dustMotesMesh.name = 'AvalancheMineralDust3D';
  dustMotesMesh.count = 0;
  dustMotesMesh.frustumCulled = false;
  dustMotesMesh.renderOrder = 4;
  group.add(dustMotesMesh);

  // ---------------------------------------------------------------------------
  // 6. Front & Back 45-Degree Beveled Observation Glass Panes
  // ---------------------------------------------------------------------------
  const glassGeo = createChamferedBlockGeometry(
    GLASS_PANE_W * 0.5,
    GLASS_PANE_H * 0.5,
    GLASS_PANE_THICK * 0.5,
    0.0044
  );
  geometries.push(glassGeo);

  const frontGlassMesh = new THREE.Mesh(glassGeo, materials.beveledGlass);
  frontGlassMesh.name = 'BeveledGlassFront';
  frontGlassMesh.position.set(0, 0, GLASS_CENTER_Z);
  frontGlassMesh.renderOrder = 6;
  group.add(frontGlassMesh);

  const backGlassMesh = new THREE.Mesh(glassGeo, materials.beveledGlass);
  backGlassMesh.name = 'BeveledGlassBack';
  backGlassMesh.position.set(0, 0, -GLASS_CENTER_Z);
  backGlassMesh.renderOrder = 5;
  group.add(backGlassMesh);

  const cavityHitGeo = new THREE.PlaneGeometry(CAVITY_W, CAVITY_H);
  const cavityHitMat = new THREE.MeshBasicMaterial({
    visible: false,
    side: THREE.DoubleSide,
  });
  geometries.push(cavityHitGeo);
  extraMats.push(cavityHitMat);
  const cavityHitMesh = new THREE.Mesh(cavityHitGeo, cavityHitMat);
  cavityHitMesh.name = 'CavityRaycastPlane';
  group.add(cavityHitMesh);

  // ---------------------------------------------------------------------------
  // Live Zero-GC Updaters
  // ---------------------------------------------------------------------------
  const depthSegs = 4;
  const rows = depthSegs + 1;
  const crestPosAttr = sandCrestGeo.attributes.position as THREE.BufferAttribute;
  const crestNormAttr = sandCrestGeo.attributes.normal as THREE.BufferAttribute;
  const crestColAttr = sandCrestGeo.attributes.color as THREE.BufferAttribute;
  const posArr = crestPosAttr.array as Float32Array;
  const normArr = crestNormAttr.array as Float32Array;
  const colArr = crestColAttr.array as Float32Array;

  const updateSandCrest = (surfaceY: Float32Array, surfaceRgb: Float32Array): void => {
    const dxWorld = CAVITY_W / (GRID_W - 1);
    const dyPerCell = CAVITY_H / GRID_H;

    for (let c = 0; c < GRID_W; c++) {
      const yCell = surfaceY[c];
      const yBase = -CAVITY_HALF_H + (yCell + 0.5) * dyPerCell;
      const yL = surfaceY[c > 0 ? c - 1 : 0];
      const yR = surfaceY[c < GRID_W - 1 ? c + 1 : GRID_W - 1];
      const slopeX = ((yR - yL) * dyPerCell) / (2 * dxWorld);

      const cr = surfaceRgb[c * 3];
      const cg = surfaceRgb[c * 3 + 1];
      const cb = surfaceRgb[c * 3 + 2];

      for (let r = 0; r < rows; r++) {
        const v = r / depthSegs;
        const zNorm = v * 2 - 1;
        const crownLift = (1 - zNorm * zNorm) * 0.0055;
        const slopeZ = (-2 * zNorm * 0.0055) / CAVITY_HALF_D;

        const vi = r * GRID_W + c;
        const o3 = vi * 3;
        posArr[o3 + 1] = Math.min(CAVITY_HALF_H - 0.004, yBase + crownLift);

        const nx = -slopeX;
        const ny = 1.0;
        const nz = -slopeZ;
        const invL = 1 / Math.hypot(nx, ny, nz);
        normArr[o3] = nx * invL;
        normArr[o3 + 1] = ny * invL;
        normArr[o3 + 2] = nz * invL;

        const edgeShade = 0.86 + 0.14 * (1 - zNorm * zNorm);
        colArr[o3] = cr * edgeShade;
        colArr[o3 + 1] = cg * edgeShade;
        colArr[o3 + 2] = cb * edgeShade;
      }
    }

    crestPosAttr.needsUpdate = true;
    crestNormAttr.needsUpdate = true;
    crestColAttr.needsUpdate = true;
  };

  // Coxa socket positions for Front (0), Mid (1), Hind (2) leg pairs in ant-local space
  const legAttachX = [0.0056, -0.0004, -0.0058];
  const legAttachY = [0.0014, 0.0012, 0.0014];
  const legAttachZ = [0.0032, 0.0034, 0.0030];
  const legBaseYaw = [0.58, 0.02, -0.62];

  const updateWorkerAnts = (ants: readonly DiggerAgent[], elapsedSec: number): void => {
    const count = Math.min(ANT_COUNT, ants.length);
    for (let i = 0; i < count; i++) {
      const a = ants[i];
      const wx = -CAVITY_HALF_W + ((a.x + 0.5) / GRID_W) * CAVITY_W;
      const wy = -CAVITY_HALF_H + ((a.y + 0.5) / GRID_H) * CAVITY_H;
      const wz = a.zLane * 0.032;

      // Construct strictly right-handed (det = +1) wall-hugging 3D orthonormal frame:
      // +X = forward heading tangent, +Y = dorsal back pointing away from sand floor/wall,
      // +Z = right lateral across glass thickness!
      const cosH = Math.cos(a.heading);
      const sinH = Math.sin(a.heading);
      _vFwd.set(cosH, sinH, 0);

      // Project dorsal surface normal perpendicular to forward vector
      const dotUp = a.upX * cosH + a.upY * sinH;
      let ux = a.upX - cosH * dotUp;
      let uy = a.upY - sinH * dotUp;
      const uLen = Math.hypot(ux, uy);
      if (uLen > 0.08) {
        ux /= uLen;
        uy /= uLen;
      } else {
        // Fallback perpendicular pointing upward in Y
        ux = -sinH;
        uy = cosH;
        if (uy < 0) {
          ux = -ux;
          uy = -uy;
        }
      }
      _vUp.set(ux, uy, 0);
      _vLat.crossVectors(_vFwd, _vUp);
      if (_vLat.lengthSq() < 1e-6) {
        _vLat.set(0, 0, 1);
      } else {
        _vLat.normalize();
      }
      _vUp.crossVectors(_vLat, _vFwd).normalize();

      // Gentle three-quarter body roll toward the glass pane (~18 deg) so ALL 6 jointed legs
      // and BOTH elbowed antennae read crisply in 3D relief from front & rear views!
      const rollSign = _vLat.z >= 0 ? -1 : 1;
      const tumbleRoll = a.state === 'falling' ? Math.sin(elapsedSec * 22 + i * 1.9) * 0.65 : 0;
      _qRoll.setFromAxisAngle(_vFwd, rollSign * 0.3 + tumbleRoll);
      _vUp.applyQuaternion(_qRoll);
      _vLat.applyQuaternion(_qRoll);

      // Subtle living gait bob & excavation heave
      const isExc = a.state === 'excavate' || a.state === 'rescue_dig' || a.state === 'buried';
      const bob = isExc
        ? Math.sin(a.mandiblePhase * 2.1) * 0.0014
        : Math.abs(Math.sin(a.walkPhase)) * 0.0011;
      const s = a.bodyScale * 1.06;

      _mBasis.makeBasis(_vFwd, _vUp, _vLat);
      _mAntRoot.copy(_mBasis);
      _mAntRoot.scale(_vScale.set(s, s, s));
      _mAntRoot.setPosition(wx + _vUp.x * bob, wy + _vUp.y * bob, wz);

      diggers[i].root.position.set(wx, wy, wz);

      // 1. Mesosoma Thorax + Pinched Petiole
      antThoraxMesh.setMatrixAt(i, _mAntRoot);

      // 2. Articulated Segmented Gaster (pivots gently at petiole socket (-0.0135, 0.0034, 0))
      const gasterPitch =
        -0.24 +
        (isExc ? -0.14 + Math.sin(a.mandiblePhase * 1.9) * 0.1 : Math.sin(a.walkPhase * 0.5) * 0.06) +
        (a.state === 'greet' ? 0.12 : 0);
      _eRot.set(0, Math.sin(a.walkPhase) * 0.06, gasterPitch);
      _qPart.setFromEuler(_eRot);
      _mLocal.compose(_vPos.set(-0.0138, 0.0036, 0), _qPart, _vOneScale);
      _mPart.multiplyMatrices(_mAntRoot, _mLocal);
      antGasterMesh.setMatrixAt(i, _mPart);

      // 3. Cordate Head Capsule + Compound Eyes (pivots at cervix (+0.0108, 0.0044, 0))
      const headPitch =
        a.carryingPigment >= 0
          ? 0.08 // Hold head proudly up while hauling grain!
          : isExc
            ? -0.34 + Math.sin(a.mandiblePhase * 2.4) * 0.14
            : a.state === 'greet'
              ? 0.14 + Math.sin(a.antennaPhase * 1.6) * 0.08
              : a.state === 'groom'
                ? -0.42
                : -0.18 + Math.sin(a.antennaPhase * 0.45) * 0.06;
      const headYaw =
        a.state === 'greet'
          ? Math.sin(a.antennaPhase * 1.1) * 0.22
          : Math.sin(a.antennaPhase * 0.35 + i) * 0.12;
      _eRot.set(0, headYaw, headPitch);
      _qPart.setFromEuler(_eRot);
      _mLocal.compose(_vPos.set(0.0106, 0.0044, 0), _qPart, _vOneScale);
      _mHeadWorld.multiplyMatrices(_mAntRoot, _mLocal);
      antHeadMesh.setMatrixAt(i, _mHeadWorld);
      antEyesMesh.setMatrixAt(i, _mHeadWorld);

      // 4. Twin Articulated Curved Pincer Mandibles (attached to front of head)
      const jawOpen =
        a.carryingPigment >= 0
          ? 0.14
          : isExc
            ? 0.28 + Math.abs(Math.sin(a.mandiblePhase * 2.5)) * 0.34
            : a.state === 'greet'
              ? 0.24
              : 0.1;

      for (let side = 0; side < 2; side++) {
        const sgn = side === 0 ? 1 : -1;
        // Left (+Z) uses native mandible; Right (-Z) flips cleanly via 180-deg X roll (proper det=+1!)
        _eRot.set(side === 0 ? 0 : Math.PI, -sgn * (0.16 + jawOpen), -0.12);
        _qPart.setFromEuler(_eRot);
        _mLocal.compose(_vPos.set(0.0096, -0.0006, sgn * 0.0024), _qPart, _vOneScale);
        _mPart.multiplyMatrices(_mHeadWorld, _mLocal);
        antMandibleMesh.setMatrixAt(i * 2 + side, _mPart);
      }

      // 5. Twin Elbowed Geniculate Antennae (sweeping, probing & rapid greeting taps!)
      const antSweepAmp = a.state === 'greet' ? 0.42 : a.state === 'groom' ? 0.32 : 0.2;
      for (let side = 0; side < 2; side++) {
        const sgn = side === 0 ? 1 : -1;
        const phase = a.antennaPhase + side * 1.45;
        const pitchSweep = Math.sin(phase) * antSweepAmp;
        const yawSweep = Math.cos(phase * 0.78) * (antSweepAmp * 0.85);
        _eRot.set(
          side === 0 ? 0.1 : Math.PI - 0.1,
          sgn * (0.14 + yawSweep),
          0.12 + pitchSweep
        );
        _qPart.setFromEuler(_eRot);
        _mLocal.compose(_vPos.set(0.0074, 0.0022, sgn * 0.0022), _qPart, _vOneScale);
        _mPart.multiplyMatrices(_mHeadWorld, _mLocal);
        antAntennaMesh.setMatrixAt(i * 2 + side, _mPart);
      }

      // 6. Six Jointed 2-Segment Legs (Authentic Alternating Tripod Gait: L0+R1+L2 vs R0+L1+R2!)
      for (let k = 0; k < 3; k++) {
        for (let side = 0; side < 2; side++) {
          const sgn = side === 0 ? 1 : -1;
          const tripodSign = (k + side) % 2 === 0 ? 1 : -1;
          const phase = a.walkPhase + (tripodSign > 0 ? 0 : Math.PI);
          const strideAmp = a.state === 'falling' ? 0.56 : a.state === 'greet' ? 0.05 : 0.34;
          const swingYaw = Math.sin(phase) * strideAmp;
          const footLift = Math.max(0, Math.cos(phase)) * (a.state === 'falling' ? 0.36 : 0.18);

          // Proper det=+1 orientation: Right (-Z) legs rotate 180 deg around Y to face -Z!
          if (side === 0) {
            _eRot.set(-footLift, legBaseYaw[k] + swingYaw, 0);
          } else {
            _eRot.set(-footLift, Math.PI - (legBaseYaw[k] + swingYaw), 0);
          }
          _qPart.setFromEuler(_eRot);
          const legScaleY = k === 2 ? 1.12 : k === 0 ? 0.94 : 1.0;
          _mLocal.compose(
            _vPos.set(legAttachX[k], legAttachY[k], sgn * legAttachZ[k]),
            _qPart,
            _vScale.set(1.0, legScaleY, 1.0)
          );
          _mPart.multiplyMatrices(_mAntRoot, _mLocal);
          antLegMesh.setMatrixAt(i * 6 + k * 2 + side, _mPart);
        }
      }

      // 7. Carried Colored Mineral Sand Granule clamped cleanly between the mandibles!
      const cOff = i * 4;
      if (a.carryingPigment >= 0 && a.carryingPigment < SAND_PIGMENTS.length) {
        const rgb = SAND_PIGMENTS[a.carryingPigment].rgbLinear;
        const shadeMul = 0.85 + (a.carryingShade / 255) * 0.34;
        carriedColorArr[cOff] = Math.min(1, rgb[0] * shadeMul);
        carriedColorArr[cOff + 1] = Math.min(1, rgb[1] * shadeMul);
        carriedColorArr[cOff + 2] = Math.min(1, rgb[2] * shadeMul);
        carriedColorArr[cOff + 3] = 1.0;

        _eRot.set(i * 1.1, i * 0.7, 0.3);
        _qPart.setFromEuler(_eRot);
        _mLocal.compose(_vPos.set(0.0156, -0.0011, 0), _qPart, _vOneScale);
        _mPart.multiplyMatrices(_mHeadWorld, _mLocal);
        antCarriedGrainMesh.setMatrixAt(i, _mPart);
      } else {
        _mLocal.compose(_vPos.set(0, 0, 0), _qPart.identity(), _vZeroScale);
        antCarriedGrainMesh.setMatrixAt(i, _mLocal);
      }
    }

    antThoraxMesh.instanceMatrix.needsUpdate = true;
    antGasterMesh.instanceMatrix.needsUpdate = true;
    antHeadMesh.instanceMatrix.needsUpdate = true;
    antEyesMesh.instanceMatrix.needsUpdate = true;
    antMandibleMesh.instanceMatrix.needsUpdate = true;
    antAntennaMesh.instanceMatrix.needsUpdate = true;
    antLegMesh.instanceMatrix.needsUpdate = true;
    antCarriedGrainMesh.instanceMatrix.needsUpdate = true;
    carriedColorAttr.needsUpdate = true;
  };

  const updateParticles3D = (
    grains: readonly Grain3D[],
    dust: readonly DustMote3D[],
    pigmentRgbLinear: Float32Array
  ): void => {
    let gCount = 0;
    for (let i = 0; i < grains.length && gCount < MAX_3D_GRAINS; i++) {
      const g = grains[i];
      if (!g.active) continue;

      const wx = -CAVITY_HALF_W + ((g.x + 0.5) / GRID_W) * CAVITY_W;
      const wy = -CAVITY_HALF_H + ((g.y + 0.5) / GRID_H) * CAVITY_H;
      const wz = g.z * (CAVITY_HALF_D - 0.012);

      _eRot.set(g.rotX, g.rotY, g.rotZ);
      _qPart.setFromEuler(_eRot);
      _mPart.compose(_vPos.set(wx, wy, wz), _qPart, _vScale.set(g.scale, g.scale, g.scale));
      tumblingGrainsMesh.setMatrixAt(gCount, _mPart);

      const pOff = Math.max(0, Math.min(SAND_PIGMENTS.length - 1, g.pigmentId)) * 3;
      const shadeMul = 0.86 + (g.shade / 255) * 0.34;
      const cOff = gCount * 4;
      tumblingColorArr[cOff] = Math.min(1, pigmentRgbLinear[pOff] * shadeMul);
      tumblingColorArr[cOff + 1] = Math.min(1, pigmentRgbLinear[pOff + 1] * shadeMul);
      tumblingColorArr[cOff + 2] = Math.min(1, pigmentRgbLinear[pOff + 2] * shadeMul);
      tumblingColorArr[cOff + 3] = 0.65 + (i % 5) * 0.08;
      gCount++;
    }
    tumblingGrainsMesh.count = gCount;
    if (gCount > 0) {
      tumblingGrainsMesh.instanceMatrix.needsUpdate = true;
      tumblingColorAttr.needsUpdate = true;
    }

    let dCount = 0;
    for (let i = 0; i < dust.length && dCount < MAX_DUST_MOTES; i++) {
      const m = dust[i];
      if (!m.active) continue;
      const tNorm = m.age / Math.max(1, m.life);
      const env = Math.sin(Math.min(1, tNorm) * Math.PI) * m.alpha;
      if (env <= 0.005) continue;

      const wx = -CAVITY_HALF_W + ((m.x + 0.5) / GRID_W) * CAVITY_W;
      const wy = -CAVITY_HALF_H + ((m.y + 0.5) / GRID_H) * CAVITY_H;
      const wz = m.z * (CAVITY_HALF_D - 0.016);

      _qPart.identity();
      _mPart.compose(
        _vPos.set(wx, wy, wz),
        _qPart,
        _vScale.set(m.size, m.size * 0.78, m.size * 0.55)
      );
      dustMotesMesh.setMatrixAt(dCount, _mPart);

      const pOff = Math.max(0, Math.min(SAND_PIGMENTS.length - 1, m.pigmentId)) * 3;
      const cOff = dCount * 4;
      // Lift toward warm kaolin mineral haze so dust reads airy rather than muddy
      dustColorArr[cOff] = pigmentRgbLinear[pOff] * 0.55 + 0.36;
      dustColorArr[cOff + 1] = pigmentRgbLinear[pOff + 1] * 0.55 + 0.33;
      dustColorArr[cOff + 2] = pigmentRgbLinear[pOff + 2] * 0.55 + 0.28;
      dustColorArr[cOff + 3] = env;
      dCount++;
    }
    dustMotesMesh.count = dCount;
    if (dCount > 0) {
      dustMotesMesh.instanceMatrix.needsUpdate = true;
      dustColorAttr.needsUpdate = true;
    }
  };

  return {
    group,
    frontSandMesh,
    backSandMesh,
    frontBackwallMesh,
    backBackwallMesh,
    sandCrestMesh,
    sandCrestGeo,
    frontGlassMesh,
    backGlassMesh,
    cavityHitMesh,
    frameHitMeshes,
    diggers,
    antThoraxMesh,
    antGasterMesh,
    antHeadMesh,
    antEyesMesh,
    antMandibleMesh,
    antAntennaMesh,
    antLegMesh,
    antCarriedGrainMesh,
    tumblingGrainsMesh,
    dustMotesMesh,
    updateSandCrest,
    updateWorkerAnts,
    updateParticles3D,
    dispose: () => {
      for (const g of geometries) g.dispose();
      for (const m of extraMats) m.dispose();
    },
  };
}
