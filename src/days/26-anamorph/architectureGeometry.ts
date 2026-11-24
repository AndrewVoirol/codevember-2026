import * as THREE from 'three/webgpu';
import {
  ARCHITECTURAL_SLABS,
  type ArchitecturalSlab,
} from './anamorphMath';

type Pt3 = readonly [number, number, number];

function pushOutwardTri(
  positions: number[],
  normals: number[],
  uvs: number[],
  matClasses: number[],
  a: Pt3,
  b: Pt3,
  c: Pt3,
  center: Pt3,
  matClass: number
): void {
  let p1 = b;
  let p2 = c;

  let abx = p1[0] - a[0];
  let aby = p1[1] - a[1];
  let abz = p1[2] - a[2];
  let acx = p2[0] - a[0];
  let acy = p2[1] - a[1];
  let acz = p2[2] - a[2];

  let nx = aby * acz - abz * acy;
  let ny = abz * acx - abx * acz;
  let nz = abx * acy - aby * acx;

  const cx = (a[0] + p1[0] + p2[0]) / 3 - center[0];
  const cy = (a[1] + p1[1] + p2[1]) / 3 - center[1];
  const cz = (a[2] + p1[2] + p2[2]) / 3 - center[2];

  if (nx * cx + ny * cy + nz * cz < 0) {
    const tmp = p1;
    p1 = p2;
    p2 = tmp;
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }

  const nLen = Math.hypot(nx, ny, nz);
  if (nLen < 1e-10) return;
  nx /= nLen;
  ny /= nLen;
  nz /= nLen;

  // Metric triplanar UV projection aligned to dominant face axis
  const absX = Math.abs(nx);
  const absY = Math.abs(ny);
  const absZ = Math.abs(nz);
  const projectUv = (p: Pt3): [number, number] => {
    if (absY >= absX && absY >= absZ) return [p[0], p[2]];
    if (absX >= absY && absX >= absZ) return [p[2], p[1]];
    return [p[0], p[1]];
  };

  const verts: readonly Pt3[] = [a, p1, p2];
  for (let i = 0; i < 3; i++) {
    const v = verts[i];
    positions.push(v[0], v[1], v[2]);
    normals.push(nx, ny, nz);
    const [u, w] = projectUv(v);
    uvs.push(u, w);
    matClasses.push(matClass);
  }
}

function pushOutwardQuad(
  positions: number[],
  normals: number[],
  uvs: number[],
  matClasses: number[],
  a: Pt3,
  b: Pt3,
  c: Pt3,
  d: Pt3,
  center: Pt3,
  matClass: number
): void {
  pushOutwardTri(positions, normals, uvs, matClasses, a, b, c, center, matClass);
  pushOutwardTri(positions, normals, uvs, matClasses, a, c, d, center, matClass);
}

/**
 * Builds a watertight, strictly outward-wound 26-face chamfered architectural slab
 * (6 main planar faces + 12 45-degree chamfer edge strips + 8 corner bevel facets).
 * All vertices lie cleanly on or inside [slab.min, slab.max].
 */
export function buildBeveledSlabGeometry(slab: ArchitecturalSlab): THREE.BufferGeometry {
  const [minX, minY, minZ] = slab.min;
  const [maxX, maxY, maxZ] = slab.max;
  const w = maxX - minX;
  const h = maxY - minY;
  const d = maxZ - minZ;
  const b = Math.max(0.001, Math.min(slab.bevel, Math.min(w, h, d) * 0.22));
  const center: Pt3 = [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5];

  // For each corner (sx, sy, sz) in {-1, +1}^3, generate its 3 chamfer vertices on ±X, ±Y, ±Z faces
  const vx = (sx: number, sy: number, sz: number): Pt3 => [
    sx < 0 ? minX : maxX,
    sy < 0 ? minY + b : maxY - b,
    sz < 0 ? minZ + b : maxZ - b,
  ];
  const vy = (sx: number, sy: number, sz: number): Pt3 => [
    sx < 0 ? minX + b : maxX - b,
    sy < 0 ? minY : maxY,
    sz < 0 ? minZ + b : maxZ - b,
  ];
  const vz = (sx: number, sy: number, sz: number): Pt3 => [
    sx < 0 ? minX + b : maxX - b,
    sy < 0 ? minY + b : maxY - b,
    sz < 0 ? minZ : maxZ,
  ];

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const matClasses: number[] = [];
  const mc = slab.materialClass;

  // 1. Six main planar faces
  // +X and -X
  for (const sx of [-1, 1]) {
    pushOutwardQuad(
      positions,
      normals,
      uvs,
      matClasses,
      vx(sx, -1, -1),
      vx(sx, 1, -1),
      vx(sx, 1, 1),
      vx(sx, -1, 1),
      center,
      mc
    );
  }
  // +Y and -Y
  for (const sy of [-1, 1]) {
    pushOutwardQuad(
      positions,
      normals,
      uvs,
      matClasses,
      vy(-1, sy, -1),
      vy(1, sy, -1),
      vy(1, sy, 1),
      vy(-1, sy, 1),
      center,
      mc
    );
  }
  // +Z and -Z
  for (const sz of [-1, 1]) {
    pushOutwardQuad(
      positions,
      normals,
      uvs,
      matClasses,
      vz(-1, -1, sz),
      vz(1, -1, sz),
      vz(1, 1, sz),
      vz(-1, 1, sz),
      center,
      mc
    );
  }

  // 2. Twelve 45-degree chamfer edge strips
  // 4 edges parallel to X (between ±Y and ±Z)
  for (const sy of [-1, 1]) {
    for (const sz of [-1, 1]) {
      pushOutwardQuad(
        positions,
        normals,
        uvs,
        matClasses,
        vy(-1, sy, sz),
        vy(1, sy, sz),
        vz(1, sy, sz),
        vz(-1, sy, sz),
        center,
        mc
      );
    }
  }
  // 4 edges parallel to Y (between ±X and ±Z)
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      pushOutwardQuad(
        positions,
        normals,
        uvs,
        matClasses,
        vx(sx, -1, sz),
        vx(sx, 1, sz),
        vz(sx, 1, sz),
        vz(sx, -1, sz),
        center,
        mc
      );
    }
  }
  // 4 edges parallel to Z (between ±X and ±Y)
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      pushOutwardQuad(
        positions,
        normals,
        uvs,
        matClasses,
        vx(sx, sy, -1),
        vx(sx, sy, 1),
        vy(sx, sy, 1),
        vy(sx, sy, -1),
        center,
        mc
      );
    }
  }

  // 3. Eight 3-axis corner bevel triangles
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        pushOutwardTri(
          positions,
          normals,
          uvs,
          matClasses,
          vx(sx, sy, sz),
          vy(sx, sy, sz),
          vz(sx, sy, sz),
          center,
          mc
        );
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('aMatClass', new THREE.Float32BufferAttribute(matClasses, 1));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return geo;
}

/**
 * Merges all 30 beveled architectural slabs into a single watertight BufferGeometry
 * (1 draw call for the entire brutalist courtyard maquette).
 */
export function buildArchitecturalCourtyardGeometry(
  slabs: readonly ArchitecturalSlab[] = ARCHITECTURAL_SLABS
): THREE.BufferGeometry {
  const slabGeos = slabs.map((s) => buildBeveledSlabGeometry(s));
  let totalVerts = 0;
  for (const g of slabGeos) {
    totalVerts += g.attributes.position.count;
  }

  const pos = new Float32Array(totalVerts * 3);
  const norm = new Float32Array(totalVerts * 3);
  const uv = new Float32Array(totalVerts * 2);
  const matClass = new Float32Array(totalVerts);

  let offset = 0;
  for (const g of slabGeos) {
    const count = g.attributes.position.count;
    pos.set((g.attributes.position as THREE.BufferAttribute).array as Float32Array, offset * 3);
    norm.set((g.attributes.normal as THREE.BufferAttribute).array as Float32Array, offset * 3);
    uv.set((g.attributes.uv as THREE.BufferAttribute).array as Float32Array, offset * 2);
    matClass.set((g.attributes.aMatClass as THREE.BufferAttribute).array as Float32Array, offset);
    offset += count;
    g.dispose();
  }

  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(norm, 3));
  merged.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  merged.setAttribute('aMatClass', new THREE.BufferAttribute(matClass, 1));
  merged.computeBoundingBox();
  merged.computeBoundingSphere();
  return merged;
}

export interface SurveyorSightRingParts {
  readonly group: THREE.Group;
  readonly bezelGeometry: THREE.BufferGeometry;
  readonly ticksGeometry: THREE.BufferGeometry;
  readonly dispose: () => void;
}

/**
 * Builds the machined satin-brass optical surveyor's sight-ring (closed outward-wound
 * stepped lathe bezel + 4 cardinal vernier reticle index teeth + outer sighting lugs).
 * Oriented so its optical aperture looks straight along local +Z.
 */
export function buildSurveyorSightRingGeometry(): {
  bezelGeometry: THREE.BufferGeometry;
  ticksGeometry: THREE.BufferGeometry;
} {
  // Closed cross-section profile revolved around Y, then rotated +90 deg around X -> aperture faces +Z.
  // Ordered strictly counter-clockwise around the cross-section centroid so outward normals point away from metal solid!
  // Wait: Three.js LatheGeometry points traced bottom-to-top along outer wall produce outward normals (+r),
  // and top-to-bottom along inner bore produce inward-toward-axis (-r) normals facing the hollow aperture!
  const rInner = 0.088;
  const rOuter = 0.114;
  const rFlange = 0.122;
  const halfDepth = 0.015;
  const lip = 0.004;

  const profile: THREE.Vector2[] = [
    // Inner bore top -> bottom (faces toward hollow center -r)
    new THREE.Vector2(rInner, halfDepth - lip),
    new THREE.Vector2(rInner, -halfDepth + lip),
    // Bottom-inner chamfer -> bottom rim
    new THREE.Vector2(rInner + lip, -halfDepth),
    new THREE.Vector2(rOuter - lip, -halfDepth),
    // Outer wall bottom -> stepped knurled flange -> top (faces outward +r)
    new THREE.Vector2(rOuter, -halfDepth + lip),
    new THREE.Vector2(rOuter, -0.005),
    new THREE.Vector2(rFlange, -0.003),
    new THREE.Vector2(rFlange, 0.003),
    new THREE.Vector2(rOuter, 0.005),
    new THREE.Vector2(rOuter, halfDepth - lip),
    // Top rim -> top-inner chamfer closing loop cleanly
    new THREE.Vector2(rOuter - lip, halfDepth),
    new THREE.Vector2(rInner + lip, halfDepth),
    new THREE.Vector2(rInner, halfDepth - lip),
  ];

  const bezelGeometry = new THREE.LatheGeometry(profile, 48);
  bezelGeometry.rotateX(Math.PI * 0.5);
  bezelGeometry.computeVertexNormals();
  bezelGeometry.computeBoundingBox();
  bezelGeometry.computeBoundingSphere();

  // Four precision cardinal reticle teeth (pointing slightly inward) + 4 fine outer index lugs
  const tickSlabs: ArchitecturalSlab[] = [
    // Inner cardinal hairline ticks (leaving generous open center aperture r < 0.056)
    {
      id: 'tick-right',
      kind: 'pier',
      materialClass: 0,
      min: [0.056, -0.0032, -0.0035],
      max: [0.092, 0.0032, 0.0035],
      bevel: 0.0008,
    },
    {
      id: 'tick-left',
      kind: 'pier',
      materialClass: 0,
      min: [-0.092, -0.0032, -0.0035],
      max: [-0.056, 0.0032, 0.0035],
      bevel: 0.0008,
    },
    {
      id: 'tick-top',
      kind: 'pier',
      materialClass: 0,
      min: [-0.0032, 0.056, -0.0035],
      max: [0.0032, 0.092, 0.0035],
      bevel: 0.0008,
    },
    {
      id: 'tick-bottom',
      kind: 'pier',
      materialClass: 0,
      min: [-0.0032, -0.092, -0.0035],
      max: [0.0032, -0.056, 0.0035],
      bevel: 0.0008,
    },
    // Subtle outer brass alignment fins at 12, 3, 6, 9 o'clock
    {
      id: 'lug-top',
      kind: 'pier',
      materialClass: 0,
      min: [-0.0055, 0.112, -0.0055],
      max: [0.0055, 0.140, 0.0055],
      bevel: 0.0012,
    },
    {
      id: 'lug-bottom',
      kind: 'pier',
      materialClass: 0,
      min: [-0.0055, -0.140, -0.0055],
      max: [0.0055, -0.112, 0.0055],
      bevel: 0.0012,
    },
    {
      id: 'lug-right',
      kind: 'pier',
      materialClass: 0,
      min: [0.112, -0.0055, -0.0055],
      max: [0.140, 0.0055, 0.0055],
      bevel: 0.0012,
    },
    {
      id: 'lug-left',
      kind: 'pier',
      materialClass: 0,
      min: [-0.140, -0.0055, -0.0055],
      max: [-0.112, 0.0055, 0.0055],
      bevel: 0.0012,
    },
  ];

  const ticksGeometry = buildArchitecturalCourtyardGeometry(tickSlabs);

  return { bezelGeometry, ticksGeometry };
}

export function buildSurveyorSightRing(
  bezelMaterial: THREE.Material,
  ticksMaterial: THREE.Material
): SurveyorSightRingParts {
  const { bezelGeometry, ticksGeometry } = buildSurveyorSightRingGeometry();
  const group = new THREE.Group();
  group.name = 'BrassSurveyorSightRing';

  const bezelMesh = new THREE.Mesh(bezelGeometry, bezelMaterial);
  bezelMesh.name = 'SightRingBezel';
  bezelMesh.castShadow = false;
  bezelMesh.receiveShadow = false;
  group.add(bezelMesh);

  const ticksMesh = new THREE.Mesh(ticksGeometry, ticksMaterial);
  ticksMesh.name = 'SightRingReticleTicks';
  ticksMesh.castShadow = false;
  ticksMesh.receiveShadow = false;
  group.add(ticksMesh);

  return {
    group,
    bezelGeometry,
    ticksGeometry,
    dispose: () => {
      bezelGeometry.dispose();
      ticksGeometry.dispose();
    },
  };
}
