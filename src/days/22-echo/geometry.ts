import * as THREE from 'three/webgpu';
import { APERTURE_HALF, CUBE_OUTER_HALF, GLASS_PANE_HALF } from './optics';

export interface EchoMaterialsContract {
  agedBronze: THREE.Material;
  satinBrass: THREE.Material;
  knurledBrass: THREE.Material;
  bluedSteel: THREE.Material;
  infinityMirror: THREE.Material;
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

  // 24 chamfered vertices grouped by octant sign (sx, sy, sz):
  // For each octant, 3 vertices lie on the 3 outer faces:
  //   vx = (sx*hx, sy*iy, sz*iz)
  //   vy = (sx*ix, sy*hy, sz*iz)
  //   vz = (sx*ix, sy*iy, sz*hz)
  const vx = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * hx, sy * iy, sz * iz);
  const vy = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * ix, sy * hy, sz * iz);
  const vz = (sx: number, sy: number, sz: number) => new THREE.Vector3(sx * ix, sy * iy, sz * hz);

  // 6 flat axis-aligned faces
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

  // 12 beveled 45-degree edge strips
  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      // Edges parallel to Z (XY bevel)
      pushConvexFace(
        b,
        [vx(s1, s2, -1), vx(s1, s2, 1), vy(s1, s2, 1), vy(s1, s2, -1)],
        new THREE.Vector3(s1, s2, 0)
      );
      // Edges parallel to Y (XZ bevel)
      pushConvexFace(
        b,
        [vx(s1, -1, s2), vx(s1, 1, s2), vz(s1, 1, s2), vz(s1, -1, s2)],
        new THREE.Vector3(s1, 0, s2)
      );
      // Edges parallel to X (YZ bevel)
      pushConvexFace(
        b,
        [vy(-1, s1, s2), vy(1, s1, s2), vz(1, s1, s2), vz(-1, s1, s2)],
        new THREE.Vector3(0, s1, s2)
      );
    }
  }

  // 8 corner triangular facets
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
 * Builds a watertight beveled triangular satin-brass corner gusset plate in the XY plane
 * spanning `[0..armLen]` along +X and +Y, thickness `hz * 2` centered at z = 0.
 */
export function createCornerGussetGeometry(
  armLen: number,
  halfThick: number,
  tipChamfer = 0.018
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const c = tipChamfer;
  // 5-vertex chamfered right-triangle outline in XY (flush corner at 0,0)
  const pts2d: Array<[number, number]> = [
    [0, 0],
    [armLen, 0],
    [armLen, c],
    [c, armLen],
    [0, armLen],
  ];

  const frontVerts = pts2d.map(([x, y]) => new THREE.Vector3(x, y, halfThick));
  const backVerts = pts2d.map(([x, y]) => new THREE.Vector3(x, y, -halfThick));
  const bevelFront = pts2d.map(([x, y]) => {
    const cx = armLen * 0.28;
    const cy = armLen * 0.28;
    const dx = x - cx;
    const dy = y - cy;
    const scale = 0.94;
    return new THREE.Vector3(cx + dx * scale, cy + dy * scale, halfThick + 0.004);
  });

  // Top beveled plateau (+Z) and back flush seating face (-Z)
  pushConvexFace(b, bevelFront, new THREE.Vector3(0, 0, 1));
  pushConvexFace(b, backVerts, new THREE.Vector3(0, 0, -1));

  // Perimeter vertical skirts + 45-deg front chamfer strips
  const nPts = pts2d.length;
  for (let i = 0; i < nPts; i++) {
    const j = (i + 1) % nPts;
    const edgeX = pts2d[j][0] - pts2d[i][0];
    const edgeY = pts2d[j][1] - pts2d[i][1];
    // Outward 2D normal for CCW polygon (edgeY, -edgeX)
    const out2d = new THREE.Vector3(edgeY, -edgeX, 0).normalize();

    pushConvexFace(
      b,
      [backVerts[i], backVerts[j], frontVerts[j], frontVerts[i]],
      out2d
    );

    const bevelNorm = out2d.clone().add(new THREE.Vector3(0, 0, 0.75)).normalize();
    pushConvexFace(
      b,
      [frontVerts[i], frontVerts[j], bevelFront[j], bevelFront[i]],
      bevelNorm
    );
  }

  return finalizeBuilder(b);
}

/**
 * Builds a single watertight stepped mitered picture-frame bezel in the XY plane
 * framing a clear window aperture `[-innerHalf, +innerHalf]` with sloped inner optical chamfer
 * stepping cleanly from `zFront` down to `zBack` (where the glass pane seats flush).
 */
export function createMiteredBezelFrameGeometry(
  innerHalf: number,
  outerHalf: number,
  zFront: number,
  zBack: number,
  innerStep = 0.018
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const midInner = innerHalf + innerStep;

  // 4 mitered trapezoidal bars (top, right, bottom, left) -> each bar is a closed convex 6-face prism
  const bars: Array<{
    fOutA: [number, number];
    fOutB: [number, number];
    fInA: [number, number];
    fInB: [number, number];
    bOutA: [number, number];
    bOutB: [number, number];
    bInA: [number, number];
    bInB: [number, number];
    outDir: [number, number];
    inDir: [number, number];
    miterA: [number, number];
    miterB: [number, number];
  }> = [
    // +Y top bar
    {
      fOutA: [-outerHalf, outerHalf],
      fOutB: [outerHalf, outerHalf],
      fInA: [-midInner, midInner],
      fInB: [midInner, midInner],
      bOutA: [-outerHalf, outerHalf],
      bOutB: [outerHalf, outerHalf],
      bInA: [-innerHalf, innerHalf],
      bInB: [innerHalf, innerHalf],
      outDir: [0, 1],
      inDir: [0, -1],
      miterA: [-1, -1],
      miterB: [1, -1],
    },
    // -Y bottom bar
    {
      fOutA: [outerHalf, -outerHalf],
      fOutB: [-outerHalf, -outerHalf],
      fInA: [midInner, -midInner],
      fInB: [-midInner, -midInner],
      bOutA: [outerHalf, -outerHalf],
      bOutB: [-outerHalf, -outerHalf],
      bInA: [innerHalf, -innerHalf],
      bInB: [-innerHalf, -innerHalf],
      outDir: [0, -1],
      inDir: [0, 1],
      miterA: [1, 1],
      miterB: [-1, 1],
    },
    // +X right bar
    {
      fOutA: [outerHalf, outerHalf],
      fOutB: [outerHalf, -outerHalf],
      fInA: [midInner, midInner],
      fInB: [midInner, -midInner],
      bOutA: [outerHalf, outerHalf],
      bOutB: [outerHalf, -outerHalf],
      bInA: [innerHalf, innerHalf],
      bInB: [innerHalf, -innerHalf],
      outDir: [1, 0],
      inDir: [-1, 0],
      miterA: [-1, 1],
      miterB: [-1, -1],
    },
    // -X left bar
    {
      fOutA: [-outerHalf, -outerHalf],
      fOutB: [-outerHalf, outerHalf],
      fInA: [-midInner, -midInner],
      fInB: [-midInner, midInner],
      bOutA: [-outerHalf, -outerHalf],
      bOutB: [-outerHalf, outerHalf],
      bInA: [-innerHalf, -innerHalf],
      bInB: [-innerHalf, innerHalf],
      outDir: [-1, 0],
      inDir: [1, 0],
      miterA: [1, -1],
      miterB: [1, 1],
    },
  ];

  for (const bar of bars) {
    const fOA = new THREE.Vector3(bar.fOutA[0], bar.fOutA[1], zFront);
    const fOB = new THREE.Vector3(bar.fOutB[0], bar.fOutB[1], zFront);
    const fIA = new THREE.Vector3(bar.fInA[0], bar.fInA[1], zFront);
    const fIB = new THREE.Vector3(bar.fInB[0], bar.fInB[1], zFront);

    const bOA = new THREE.Vector3(bar.bOutA[0], bar.bOutA[1], zBack);
    const bOB = new THREE.Vector3(bar.bOutB[0], bar.bOutB[1], zBack);
    const bIA = new THREE.Vector3(bar.bInA[0], bar.bInA[1], zBack);
    const bIB = new THREE.Vector3(bar.bInB[0], bar.bInB[1], zBack);

    // Front face (+Z)
    pushConvexFace(b, [fOA, fOB, fIB, fIA], new THREE.Vector3(0, 0, 1));
    // Back face (-Z)
    pushConvexFace(b, [bOA, bOB, bIB, bIA], new THREE.Vector3(0, 0, -1));
    // Outer perimeter wall
    pushConvexFace(b, [bOA, bOB, fOB, fOA], new THREE.Vector3(bar.outDir[0], bar.outDir[1], 0));
    // Inner sloped optical bevel leading directly down onto the glass pane
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
 * Builds a watertight 3D diamond-knurled satin-brass micrometer thumbscrew drum along +Y (`y in [0, height]`)
 * with genuine raised diamond pyramids, beveled top/bottom rims, recessed dimpled face cap,
 * and `aKnurlValley` vertex attribute (`0 = valley groove`, `1 = raised brass crest`).
 */
export function createKnurledDrumGeometry(
  baseRadius = 0.054,
  height = 0.068,
  teeth = 28,
  axialRings = 10
): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const radialSegs = teeth * 2; // 56 segments -> crisp diamond tooth peaks & valleys
  const yRings: Array<{ y: number; rScale: number; knurlAmp: number }> = [];

  const chamfer = 0.008;
  yRings.push({ y: 0, rScale: 0.84, knurlAmp: 0 });
  yRings.push({ y: chamfer, rScale: 1.0, knurlAmp: 0 });

  const knurlStart = chamfer + 0.003;
  const knurlEnd = height - chamfer - 0.003;
  for (let r = 0; r <= axialRings; r++) {
    const t = r / axialRings;
    const edgeFade = Math.min(1, Math.min(r, axialRings - r) / 1.0);
    yRings.push({
      y: knurlStart + (knurlEnd - knurlStart) * t,
      rScale: 1.0,
      knurlAmp: 0.0042 * (0.25 + 0.75 * edgeFade),
    });
  }

  yRings.push({ y: height - chamfer, rScale: 1.0, knurlAmp: 0 });
  yRings.push({ y: height, rScale: 0.84, knurlAmp: 0 });

  // Precompute grid points and knurl valley factor
  const ringPts: THREE.Vector3[][] = [];
  const ringValleys: number[][] = [];

  for (let ri = 0; ri < yRings.length; ri++) {
    const ring = yRings[ri];
    const pts: THREE.Vector3[] = [];
    const vals: number[] = [];
    const axialPhase = ri % 2;

    for (let j = 0; j < radialSegs; j++) {
      const phi = (j / radialSegs) * Math.PI * 2;
      let deltaR = 0;
      let kv = 1.0;
      if (ring.knurlAmp > 0) {
        // Checkerboard peak/valley produces crisp 3D diamond knurling
        const isPeak = (j + axialPhase) % 2 === 0;
        deltaR = isPeak ? ring.knurlAmp : -ring.knurlAmp * 0.85;
        kv = isPeak ? 1.0 : 0.12;
      }
      const rad = baseRadius * ring.rScale + deltaR;
      pts.push(new THREE.Vector3(Math.cos(phi) * rad, ring.y, -Math.sin(phi) * rad));
      vals.push(kv);
    }
    ringPts.push(pts);
    ringValleys.push(vals);
  }

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  // Side quads (flat-faceted triangles -> ultra-sharp studio specular sparkle on knurl diamonds!)
  for (let ri = 0; ri < yRings.length - 1; ri++) {
    const row0 = ringPts[ri];
    const row1 = ringPts[ri + 1];
    const v0 = ringValleys[ri];
    const v1 = ringValleys[ri + 1];

    for (let j = 0; j < radialSegs; j++) {
      const jNext = (j + 1) % radialSegs;
      const p00 = row0[j];
      const p10 = row0[jNext];
      const p01 = row1[j];
      const p11 = row1[jNext];

      // Triangle 1: (p00, p10, p11) -> outward CCW
      ab.subVectors(p10, p00);
      ac.subVectors(p11, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p00, p10, p11, faceN, v0[j], v0[jNext], v1[jNext]);

      // Triangle 2: (p00, p11, p01) -> outward CCW
      ab.subVectors(p11, p00);
      ac.subVectors(p01, p00);
      faceN.crossVectors(ab, ac).normalize();
      pushTriangle(b, p00, p11, p01, faceN, v0[j], v1[jNext], v1[j]);
    }
  }

  // Bottom cap (-Y outward)
  const bottomCenter = new THREE.Vector3(0, 0, 0);
  const bottomRow = ringPts[0];
  const downN = new THREE.Vector3(0, -1, 0);
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    pushTriangle(b, bottomCenter, bottomRow[jNext], bottomRow[j], downN, 0.85, 0.85, 0.85);
  }

  // Top cap with slight machined concentric dimple (+Y outward)
  const topCenter = new THREE.Vector3(0, height - 0.0045, 0);
  const topRow = ringPts[yRings.length - 1];
  for (let j = 0; j < radialSegs; j++) {
    const jNext = (j + 1) % radialSegs;
    ab.subVectors(topRow[j], topCenter);
    ac.subVectors(topRow[jNext], topCenter);
    faceN.crossVectors(ab, ac).normalize();
    pushTriangle(b, topCenter, topRow[j], topRow[jNext], faceN, 0.72, 1.0, 1.0);
  }

  return finalizeBuilder(b, true);
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
 * - 12 chamfered dark-bronze outer structural rails (~1.45 units wide)
 * - 8 heavy chamfered dark-bronze corner blocks + 24 bolted satin-brass corner gusset plates
 * - 6 stepped satin-brass recessed mirror retaining bezels
 * - 6 precision knurled brass micrometer thumbscrews (4 primary tilt adjusters + 2 counter-coupled)
 *   plus 2 opposing slotted brass fulcrum lock caps
 * - 6 hermetically recessed semi-silvered infinity mirror glass panes
 */
export function buildEchoCubeAssembly(materials: EchoMaterialsContract): EchoCubeAssembly {
  const group = new THREE.Group();
  group.name = 'InfinityMirrorCubeAssembly';

  const geometries: THREE.BufferGeometry[] = [];
  const frameHitMeshes: THREE.Mesh[] = [];
  const thumbscrewHitMeshes: THREE.Mesh[] = [];
  const thumbscrews: ThumbscrewHandle[] = [];

  const railOffset = 0.668; // rail centerline -> outer face = 0.668 + 0.057 = 0.725 (CUBE_OUTER_HALF)
  const railHalfW = CUBE_OUTER_HALF - railOffset; // 0.057
  const railHalfLen = 0.642; // overlaps solidly into corner blocks (0.602..0.734)

  // 1. 12 Chamfered Dark-Bronze Frame Rails
  const railGeoX = createChamferedBlockGeometry(railHalfLen, railHalfW, railHalfW, 0.015);
  const railGeoY = createChamferedBlockGeometry(railHalfW, railHalfLen, railHalfW, 0.015);
  const railGeoZ = createChamferedBlockGeometry(railHalfW, railHalfW, railHalfLen, 0.015);
  geometries.push(railGeoX, railGeoY, railGeoZ);

  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      const mx = new THREE.Mesh(railGeoX, materials.agedBronze);
      mx.name = `BronzeRailX_${s1}_${s2}`;
      mx.position.set(0, s1 * railOffset, s2 * railOffset);
      mx.castShadow = true;
      mx.receiveShadow = true;
      group.add(mx);
      frameHitMeshes.push(mx);

      const my = new THREE.Mesh(railGeoY, materials.agedBronze);
      my.name = `BronzeRailY_${s1}_${s2}`;
      my.position.set(s1 * railOffset, 0, s2 * railOffset);
      my.castShadow = true;
      my.receiveShadow = true;
      group.add(my);
      frameHitMeshes.push(my);

      const mz = new THREE.Mesh(railGeoZ, materials.agedBronze);
      mz.name = `BronzeRailZ_${s1}_${s2}`;
      mz.position.set(s1 * railOffset, s2 * railOffset, 0);
      mz.castShadow = true;
      mz.receiveShadow = true;
      group.add(mz);
      frameHitMeshes.push(mz);
    }
  }

  // 2. 8 Heavy Machined Chamfered Dark-Bronze Corner Blocks + 24 Satin-Brass Gusset Plates & Dome Bolts
  const cornerHalf = 0.066;
  const cornerGeo = createChamferedBlockGeometry(cornerHalf, cornerHalf, cornerHalf, 0.017);
  const gussetGeo = createCornerGussetGeometry(0.148, 0.006, 0.02);
  const boltGeo = createOutwardLathe(
    [
      [0.0005, 0.0],
      [0.014, 0.0],
      [0.014, 0.006],
      [0.011, 0.011],
      [0.005, 0.013],
      [0.0005, 0.0135],
    ],
    14
  );
  geometries.push(cornerGeo, gussetGeo, boltGeo);

  const gussetPlane = CUBE_OUTER_HALF - 0.001; // flush onto outer rail faces

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const cornerMesh = new THREE.Mesh(cornerGeo, materials.agedBronze);
        cornerMesh.name = `BronzeCorner_${sx}_${sy}_${sz}`;
        cornerMesh.position.set(sx * railOffset, sy * railOffset, sz * railOffset);
        cornerMesh.castShadow = true;
        cornerMesh.receiveShadow = true;
        group.add(cornerMesh);
        frameHitMeshes.push(cornerMesh);

        // Helper to mount a triangular brass gusset flush on face normal + 2 tiny dome bolts
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

          // Two machined hex-dome brass rivets seated cleanly on the gusset arms
          const b1 = new THREE.Mesh(boltGeo, materials.satinBrass);
          b1.name = `${name}_BoltA`;
          b1.position.set(0.098, 0.028, 0.006);
          b1.rotation.x = Math.PI / 2;
          gGroup.add(b1);

          const b2 = new THREE.Mesh(boltGeo, materials.satinBrass);
          b2.name = `${name}_BoltB`;
          b2.position.set(0.028, 0.098, 0.006);
          b2.rotation.x = Math.PI / 2;
          gGroup.add(b2);

          group.add(gGroup);
        };

        // Z-face gussets (front & back corners)
        const cPos = CUBE_OUTER_HALF - 0.003;
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
            [sx * cPos, sy * cPos, gussetPlane],
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
            [sx * cPos, sy * cPos, -gussetPlane],
            0,
            Math.PI,
            rotZ
          );
        }

        // X-face gussets (left & right corners)
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
            [gussetPlane, sy * cPos, sz * cPos],
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
            [-gussetPlane, sy * cPos, sz * cPos],
            0,
            -Math.PI / 2,
            rotZ
          );
        }

        // Y-face gussets (top & bottom corners)
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
            [sx * cPos, gussetPlane, sz * cPos],
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
            [sx * cPos, -gussetPlane, sz * cPos],
            Math.PI / 2,
            0,
            rotZ
          );
        }
      }
    }
  }

  // 3. 6 Stepped Satin-Brass Inner Glass Retaining Bezels
  // Recessed from zFront = 0.678 down to zBack = 0.594 (4 mm overlap onto GLASS_PANE_HALF = 0.598!)
  const bezelGeo = createMiteredBezelFrameGeometry(
    APERTURE_HALF,
    0.635,
    0.678,
    GLASS_PANE_HALF - 0.004,
    0.022
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
    const bezelMesh = new THREE.Mesh(bezelGeo, materials.satinBrass);
    bezelMesh.name = bo.name;
    bezelMesh.rotation.set(bo.rx, bo.ry, 0);
    bezelMesh.castShadow = true;
    bezelMesh.receiveShadow = true;
    group.add(bezelMesh);
    frameHitMeshes.push(bezelMesh);
  }

  // 4. Precision Knurled Brass Micrometer Thumbscrews (Seated Flush Into Bronze Frame Rails!)
  const bossGeo = createOutwardLathe(
    [
      [0.0005, -0.012],
      [0.044, -0.012],
      [0.044, 0.012],
      [0.036, 0.022],
      [0.032, 0.024],
      [0.0005, 0.024],
    ],
    28
  );
  const vernierGeo = createOutwardLathe(
    [
      [0.0005, 0.022],
      [0.031, 0.022],
      [0.031, 0.048],
      [0.027, 0.056],
      [0.0005, 0.056],
    ],
    28
  );
  // Threaded blued-steel micrometer lead screw shaft (bottom-to-top with crisp V-threads)
  const threadProfile: Array<[number, number]> = [[0.0005, 0.052]];
  for (let t = 0; t < 5; t++) {
    const y0 = 0.052 + t * 0.009;
    threadProfile.push([0.0195, y0]);
    threadProfile.push([0.016, y0 + 0.0045]);
  }
  threadProfile.push([0.0195, 0.098], [0.0005, 0.098]);
  const threadGeo = createOutwardLathe(threadProfile, 24);

  const shoulderGeo = createOutwardLathe(
    [
      [0.0005, 0.094],
      [0.034, 0.094],
      [0.042, 0.104],
      [0.046, 0.108],
      [0.0005, 0.108],
    ],
    28
  );
  const knurledDrumGeo = createKnurledDrumGeometry(0.054, 0.068, 28, 10);
  const hitCylGeo = new THREE.CylinderGeometry(0.088, 0.088, 0.21, 14);
  hitCylGeo.translate(0, 0.105, 0);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });

  // Opposing low-profile slotted brass fulcrum lock-caps for mechanical symmetry
  const lockCapGeo = createOutwardLathe(
    [
      [0.0005, -0.008],
      [0.038, -0.008],
      [0.038, 0.014],
      [0.031, 0.024],
      [0.024, 0.026],
      [0.0005, 0.026],
    ],
    24
  );

  geometries.push(bossGeo, vernierGeo, threadGeo, shoulderGeo, knurledDrumGeo, hitCylGeo, lockCapGeo);

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
      pos: [CUBE_OUTER_HALF - 0.004, railOffset, 0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2),
    },
    {
      name: 'ThumbscrewPosX_BottomBack',
      axis: 'y',
      sign: -1,
      pos: [CUBE_OUTER_HALF - 0.004, -railOffset, -0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2),
    },
    // -X pair -> opposing synchronized tiltY adjusters visible on rear orbit
    {
      name: 'ThumbscrewNegX_TopBack',
      axis: 'y',
      sign: -1,
      pos: [-(CUBE_OUTER_HALF - 0.004), railOffset, -0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2),
    },
    {
      name: 'ThumbscrewNegX_BottomFront',
      axis: 'y',
      sign: 1,
      pos: [-(CUBE_OUTER_HALF - 0.004), -railOffset, 0.41],
      quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2),
    },
    // +Y pair -> controls tiltX (vertical sweep)
    {
      name: 'ThumbscrewPosY_FrontLeft',
      axis: 'x',
      sign: 1,
      pos: [-0.41, CUBE_OUTER_HALF - 0.004, railOffset],
      quat: new THREE.Quaternion(),
    },
    {
      name: 'ThumbscrewPosY_BackRight',
      axis: 'x',
      sign: -1,
      pos: [0.41, CUBE_OUTER_HALF - 0.004, -railOffset],
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

    const vernier = new THREE.Mesh(vernierGeo, materials.satinBrass);
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

    const shoulder = new THREE.Mesh(shoulderGeo, materials.satinBrass);
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
    [0.41, -(CUBE_OUTER_HALF - 0.004), railOffset],
    [-0.41, -(CUBE_OUTER_HALF - 0.004), -railOffset],
  ];
  lockPositions.forEach((lp, idx) => {
    const cap = new THREE.Mesh(lockCapGeo, materials.satinBrass);
    cap.name = `FulcrumLockCap_${idx}`;
    cap.position.set(lp[0], lp[1], lp[2]);
    cap.rotation.x = Math.PI;
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
