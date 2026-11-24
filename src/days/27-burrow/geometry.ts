import * as THREE from 'three/webgpu';
import {
  BACKWALL_Z,
  CAVITY_HALF_D,
  CAVITY_HALF_H,
  CAVITY_HALF_W,
  CAVITY_H,
  CAVITY_W,
  DIGGER_COUNT,
  FRAME_OUTER_D,
  FRAME_OUTER_H,
  FRAME_OUTER_W,
  GLASS_CENTER_Z,
  GLASS_PANE_H,
  GLASS_PANE_THICK,
  GLASS_PANE_W,
  GRID_H,
  GRID_W,
} from './constants';

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
}

export interface DiggerVisualRig {
  root: THREE.Group;
  bodyGroup: THREE.Group;
  leftMandible: THREE.Mesh;
  rightMandible: THREE.Mesh;
  legsLeft: THREE.Mesh[];
  legsRight: THREE.Mesh[];
  carriedGrain: THREE.Mesh;
  carriedGrainMat: THREE.MeshStandardNodeMaterial;
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
  /** Updates the 3D top sand crest ribbon vertices & colors in-place (zero allocations). */
  updateSandCrest: (surfaceY: Float32Array, surfaceRgb: Float32Array) => void;
  dispose: () => void;
}

interface MeshBuilder {
  positions: number[];
  normals: number[];
  uvs: number[];
}

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
    // Top (+Y)
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
    // Bottom (-Y)
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
    // Right (+X)
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
    // Left (-X)
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
 * Builds a watertight beveled L-bracket corner strap (arms extending along +X and +Y from `(0,0)`)
 * as two flush-joined chamfered brass arms so corner reinforcement reads crisp under studio IBL.
 */
export function createCornerBracketArmGeometry(
  armLen: number,
  armWidth: number,
  halfThick: number
): THREE.BufferGeometry {
  // Single rectangular chamfered plate centered in [0..armLen] x [0..armWidth]
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
    // CCW when viewed from +Z
    pushTriangle(b, bl, br, tr, n, 0, 0, 1, 0, 1, 1);
    pushTriangle(b, bl, tr, tl, n, 0, 0, 1, 1, 0, 1);
  } else {
    // CCW when viewed from -Z (x still runs left=-halfW -> uv.x=0, right=+halfW -> uv.x=1!)
    pushTriangle(b, br, bl, tl, n, 1, 0, 0, 0, 0, 1);
    pushTriangle(b, br, tl, tr, n, 1, 0, 0, 1, 1, 1);
  }

  return finalizeBuilder(b);
}

/**
 * Builds the live 3D Top Sand Crest Ribbon Geometry spanning `X in [-CAVITY_HALF_W, +CAVITY_HALF_W]`
 * and `Z in [-CAVITY_HALF_D, +CAVITY_HALF_D]` (`cols x depthSegs`) with outward `+Y` winding
 * and vertex colors driven live from the top sand stratum & deposited beetle tailings!
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
    // z runs from -CAVITY_HALF_D (back) to +CAVITY_HALF_D (front)
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

  // CCW winding when viewed from +Y (top looking down):
  // (r, c) at z0, (r+1, c) at z1 > z0, (r+1, c+1) at (x1, z1), (r, c+1) at (x1, z0)
  // Cross((r+1,c)-(r,c), (r,c+1)-(r,c)) = (0,0,+dz) x (+dx,0,0) = (0, +dz*dx, 0) -> strictly +Y outward!
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
 * Builds the 3D articulated subterranean burrowing beetle rig (watertight outward-wound parts):
 * - Rounded glossy dark bronze-umber elytra abdomen (lathe along X)
 * - Sculpted pronotum thorax & head capsule
 * - 2 articulated curved mandibles + 2 antennae + 6 jointed scuttling legs
 * - 1 faceted carried mineral sand grain held cleanly between the mandibles!
 */
function buildDiggerBeetleVisual(
  id: number,
  sharedGeos: {
    elytra: THREE.BufferGeometry;
    thorax: THREE.BufferGeometry;
    head: THREE.BufferGeometry;
    mandible: THREE.BufferGeometry;
    leg: THREE.BufferGeometry;
    antenna: THREE.BufferGeometry;
    grain: THREE.BufferGeometry;
  },
  materials: BurrowMaterialsContract
): DiggerVisualRig {
  const root = new THREE.Group();
  root.name = `BurrowerBeetle_${id}`;

  const bodyGroup = new THREE.Group();
  root.add(bodyGroup);

  // Elytra abdomen (oriented along -X behind thorax; beetle faces +X in local space)
  const elytra = new THREE.Mesh(sharedGeos.elytra, materials.beetleChitin);
  elytra.name = `BeetleElytra_${id}`;
  elytra.rotation.z = Math.PI * 0.5;
  elytra.scale.set(1.0, 1.0, 0.78);
  elytra.position.set(-0.008, 0, 0);
  bodyGroup.add(elytra);

  // Pronotum thorax
  const thorax = new THREE.Mesh(sharedGeos.thorax, materials.beetleChitin);
  thorax.name = `BeetleThorax_${id}`;
  thorax.rotation.z = Math.PI * 0.5;
  thorax.scale.set(1.0, 1.0, 0.82);
  thorax.position.set(0.011, 0.001, 0);
  bodyGroup.add(thorax);

  // Head capsule
  const head = new THREE.Mesh(sharedGeos.head, materials.beetleChitin);
  head.name = `BeetleHead_${id}`;
  head.position.set(0.021, 0.0005, 0);
  bodyGroup.add(head);

  // Twin curved brass-bronze excavating mandibles
  const leftMandible = new THREE.Mesh(sharedGeos.mandible, materials.beetleAccent);
  leftMandible.name = `BeetleMandibleL_${id}`;
  leftMandible.position.set(0.025, 0.0038, 0.0025);
  leftMandible.rotation.z = -0.35;
  bodyGroup.add(leftMandible);

  const rightMandible = new THREE.Mesh(sharedGeos.mandible, materials.beetleAccent);
  rightMandible.name = `BeetleMandibleR_${id}`;
  rightMandible.position.set(0.025, -0.0038, -0.0025);
  rightMandible.rotation.z = 0.35;
  bodyGroup.add(rightMandible);

  // Swept antennae
  for (const s of [-1, 1]) {
    const ant = new THREE.Mesh(sharedGeos.antenna, materials.beetleAccent);
    ant.name = `BeetleAntenna_${id}_${s}`;
    ant.position.set(0.023, s * 0.004, s * 0.003);
    ant.rotation.z = s * 0.65;
    ant.rotation.y = s * 0.25;
    bodyGroup.add(ant);
  }

  // 6 Articulated scuttling legs (3 per side)
  const legsLeft: THREE.Mesh[] = [];
  const legsRight: THREE.Mesh[] = [];
  const legXs = [0.012, 0.002, -0.009];
  for (let k = 0; k < 3; k++) {
    const lLeg = new THREE.Mesh(sharedGeos.leg, materials.beetleAccent);
    lLeg.name = `BeetleLegL_${id}_${k}`;
    lLeg.position.set(legXs[k], 0.007, 0.0035);
    lLeg.rotation.z = 0.25 - k * 0.28;
    bodyGroup.add(lLeg);
    legsLeft.push(lLeg);

    const rLeg = new THREE.Mesh(sharedGeos.leg, materials.beetleAccent);
    rLeg.name = `BeetleLegR_${id}_${k}`;
    rLeg.position.set(legXs[k], -0.007, -0.0035);
    rLeg.rotation.z = Math.PI - (0.25 - k * 0.28);
    bodyGroup.add(rLeg);
    legsRight.push(rLeg);
  }

  // Carried colored mineral sand grain held right between the mandibles!
  const carriedGrainMat = new THREE.MeshStandardNodeMaterial({
    color: new THREE.Color(0x9e472a),
    roughness: 0.78,
    metalness: 0.04,
  });
  const carriedGrain = new THREE.Mesh(sharedGeos.grain, carriedGrainMat);
  carriedGrain.name = `BeetleCarriedGrain_${id}`;
  carriedGrain.position.set(0.031, 0, 0);
  carriedGrain.visible = false;
  bodyGroup.add(carriedGrain);

  return {
    root,
    bodyGroup,
    leftMandible,
    rightMandible,
    legsLeft,
    legsRight,
    carriedGrain,
    carriedGrainMat,
  };
}

/**
 * Constructs the complete museum-grade Day 27 - Burrow Glass-Walled Stratified Sand-Slice Terrarium:
 * - Oiled dark raw-umber walnut structural stiles, flanged museum base plinth & open-slot top cap
 * - Satin-brass wrap-around corner brackets, 24 slotted dome-head brass screws, glass retaining bezels,
 *   open top hopper lips, side breather grommets, brass leveling feet & tactile depth gauge ticks
 * - Front & back 45-deg beveled clear observation glass panes
 * - Front & back 3D parallax sand-bed cross-section planes + physically separated recessed tunnel back-walls
 * - Live 3D Top Sand Crest Ribbon Mesh bridging across the glass slice thickness
 * - 4 articulated 3D subterranean burrowing beetles crawling inside the carved cavities!
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
  // Front & back mitered satin-brass glass retaining bezels
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

  // Satin-brass top pouring hopper lip rails lining the open top vent
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

  // Tactile machined brass depth-scale tick marks along front-right & rear-left retaining bezels
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

  // Heavy satin-brass wrap-around corner brackets (all 4 corners x front/back/side wraps)
  const hArmGeo = createCornerBracketArmGeometry(0.155, 0.044, 0.005);
  const vArmGeo = createCornerBracketArmGeometry(0.044, 0.145, 0.005);
  const sideWrapGeo = createChamferedBlockGeometry(0.0055, 0.0725, stileHalfD + 0.004, 0.0028);
  geometries.push(hArmGeo, vArmGeo, sideWrapGeo);

  // Machined slotted dome-head brass screw (turned dome + recessed dark bronze screwdriver slot!)
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

  // Seat bottom corner brackets cleanly above WalnutBasePlinth top shelf (y = -0.653 -> cornerYBot = -0.651)
  const cornerYTop = CAVITY_HALF_H + 0.078; // +0.658
  const cornerYBot = -0.651; // +2 mm clean step clearance above WalnutBasePlinth top face (-0.653)
  const bracketZ = stileHalfD + 0.004; // 0.122
  const sideWrapX = halfOuterW + 0.003; // 0.883

  let screwIdx = 0;
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const cy = sy > 0 ? cornerYTop : cornerYBot;
      const cx = sx * (halfOuterW + 0.002);

      // Front & Back L-brackets
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

        // Two slotted brass screws seated flush on each L-bracket face
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

      // Outer side wrap strap joining front & back corner brackets flush along exact vArm span (0.145)
      const sideWrap = new THREE.Mesh(sideWrapGeo, materials.satinBrass);
      sideWrap.name = `BrassSideWrap_${sx}_${sy}`;
      sideWrap.position.set(sx * sideWrapX, cy - sy * 0.0725, 0);
      sideWrap.castShadow = true;
      sideWrap.receiveShadow = true;
      group.add(sideWrap);

      // Two slotted brass screws on the outer side strap
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

  // Side walnut ventilation breather grommets (classic naturalist specimen terrarium detail!)
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

  // 4 Turned Satin-Brass Leveling Feet seated flush under the walnut plinth
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
  // 4. 4 Articulated 3D Subterranean Burrowing Beetles Inside Cavity (z ~ 0)
  // ---------------------------------------------------------------------------
  const beetleGeos = {
    elytra: createOutwardLathe(
      [
        [0.0004, -0.015],
        [0.0068, -0.012],
        [0.0112, -0.005],
        [0.0118, 0.004],
        [0.0095, 0.011],
        [0.004, 0.0135],
        [0.0004, 0.014],
      ],
      14
    ),
    thorax: createOutwardLathe(
      [
        [0.0004, -0.0065],
        [0.0082, -0.005],
        [0.0094, 0.0],
        [0.0074, 0.0055],
        [0.0004, 0.0068],
      ],
      12
    ),
    head: createChamferedBlockGeometry(0.0052, 0.0056, 0.0044, 0.0022),
    mandible: createChamferedBlockGeometry(0.0046, 0.0014, 0.0012, 0.0004),
    leg: createChamferedBlockGeometry(0.0014, 0.0062, 0.0014, 0.0004),
    antenna: createChamferedBlockGeometry(0.0065, 0.0009, 0.0009, 0.0003),
    grain: createChamferedBlockGeometry(0.0046, 0.0042, 0.004, 0.0015),
  };
  geometries.push(
    beetleGeos.elytra,
    beetleGeos.thorax,
    beetleGeos.head,
    beetleGeos.mandible,
    beetleGeos.leg,
    beetleGeos.antenna,
    beetleGeos.grain
  );

  const diggers: DiggerVisualRig[] = [];
  for (let i = 0; i < DIGGER_COUNT; i++) {
    const rig = buildDiggerBeetleVisual(i, beetleGeos, materials);
    group.add(rig.root);
    diggers.push(rig);
    extraMats.push(rig.carriedGrainMat);
  }

  // ---------------------------------------------------------------------------
  // 5. Front & Back 45-Degree Beveled Observation Glass Panes
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

  // Invisible double-sided central cavity raycast target for effortless front/rear excavation & pouring
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

  // Live in-place updater for the 3D top sand crest ribbon (strictly zero allocations)
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
        const zNorm = v * 2 - 1; // -1..+1
        // Gentle convex dune crown across the 0.136 cavity thickness
        const crownLift = (1 - zNorm * zNorm) * 0.0055;
        const slopeZ = -2 * zNorm * 0.0055 / CAVITY_HALF_D;

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

        // Slight glass-edge contact darkening near front/back panes
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
    updateSandCrest,
    dispose: () => {
      for (const g of geometries) g.dispose();
      for (const m of extraMats) m.dispose();
    },
  };
}
