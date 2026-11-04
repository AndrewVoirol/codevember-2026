import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  floor,
  fract,
  max,
  mix,
  normalWorld,
  normalize,
  positionWorld,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
  uv,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  hash31Tsl,
  kubelkaMunkGlaze,
  mineralToothFbm,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

const BLOCK_HALF_W = 0.86;
const BLOCK_HALF_H = 0.86;
const BLOCK_HALF_Z = 0.44;
// Front & Back sculpted stone slices separated by a deep, recessed molten lava-cheese channel
const SLICE_INNER_Z = 0.078;
const CORE_CUSHION_HALF_Z = 0.095;
const INV_SQRT2 = Math.SQRT1_2;

// 45° diagonal cut plane: x - y = 0
// Seam runs from bottom-left (-W, -H) to top-right (+W, +H) along tangent T = (1/sqrt(2), 1/sqrt(2), 0)
// Bottom-Right wedge moves along +N = (+1/sqrt(2), -1/sqrt(2), 0); Top-Left moves along -N
const PULL_NX = INV_SQRT2;
const PULL_NY = -INV_SQRT2;
const DIAG_TANGENT_X = INV_SQRT2;
const DIAG_TANGENT_Y = INV_SQRT2;

const STRAND_RADIAL_SEGMENTS = 14;
const STRAND_LENGTH_SEGMENTS = 32;

const BEAD_RADIAL_SEGMENTS = 16;
const BEAD_LENGTH_SEGMENTS = 44;

type StrandArchetype = 'mother-lobe' | 'twisted-rope' | 'gossamer-thread' | 'gravity-drip';

interface MoltenStrand {
  index: number;
  archetype: StrandArchetype;
  mesh: THREE.Mesh;
  posAttr: THREE.BufferAttribute;
  normAttr: THREE.BufferAttribute;
  metaAttr: THREE.BufferAttribute;
  /** Position along the 45° diagonal seam in [-0.78, +0.78] */
  seamS: number;
  /** Corner envelope in [0.15, 1.0] so geometry tapers cleanly near the triangular tips */
  cornerTaper: number;
  /** Depth across the molten core in [-0.26, +0.26] */
  coreZ: number;
  /** Normalized depth layer [0 = back/deep interior, 1 = front surface] */
  depthLayer01: number;
  /** Asymmetric necking center along t in [0.24, 0.76] so web openings stagger organically */
  neckCenterT: number;
  /** Asymmetric pinch sharpness (low = broad sheet, high = sharp localized neck) */
  neckSharpness: number;
  /** Helical twist amplitude for braided/twisted ropes */
  twistAmp: number;
  /** Helical twist frequency & direction */
  twistTurns: number;
  /** Slow-moving viscous lava globule phase & amplitude along the strand */
  globuleAmp: number;
  globuleSpeed: number;
  /** Which wedge this strand retracts into if snapped (-1 = Top-Left, +1 = Bottom-Right) */
  retractSide: -1 | 1;
  /** Local anchor offset on Top-Left wedge (-N) in wedge local coordinates */
  anchorA: THREE.Vector3;
  /** Local anchor offset on Bottom-Right wedge (+N) in wedge local coordinates */
  anchorB: THREE.Vector3;
  /** Base radius of this viscoelastic cheese/amber element */
  baseRadius: number;
  /** Cross-sectional ribbon flattening along the diagonal seam tangent */
  ribbonWidthScale: number;
  /** Critical separation distance where this strand necks down and snaps */
  criticalGap: number;
  /** 0 = intact viscoelastic bridge, 0..1 = snapped & retracting smoothly into its anchor slice */
  snapProgress: number;
  isSnapped: boolean;
  /** Viscoelastic 2D pointer pull/stirring displacement & velocity (fondue drag response) */
  stirDispX: number;
  stirDispY: number;
  stirVelX: number;
  stirVelY: number;
  /** Marbled pigment ratio (0 = Creamy Yellow Ochre/Amber, 1 = Incandescent Cinnabar/Realgar lava) */
  pigmentMix: number;
  phaseOffset: number;
}

interface OozeBeadTube {
  mesh: THREE.Mesh;
  posAttr: THREE.BufferAttribute;
  normAttr: THREE.BufferAttribute;
  metaAttr: THREE.BufferAttribute;
  /** +1 for front lip (+Z), -1 for back lip (-Z), 0 for central cut-channel reservoir */
  sideZ: number;
  /** -1 = adhered to Top-Left Wedge A, +1 = adhered to Bottom-Right Wedge B (NEVER 0 in empty air!) */
  wedgeSide: -1 | 1;
}

function clampScalar(v: number, minVal: number, maxVal: number): number {
  return Math.max(minVal, Math.min(maxVal, v));
}

function smoothstepScalar(edge0: number, edge1: number, x: number): number {
  const t = clampScalar((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

function hash31Cpu(x: number, y: number, z: number): number {
  let px = (x * 0.1031) % 1;
  let py = (y * 0.103) % 1;
  let pz = (z * 0.0973) % 1;
  if (px < 0) px += 1;
  if (py < 0) py += 1;
  if (pz < 0) pz += 1;
  const d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33);
  px += d;
  py += d;
  pz += d;
  const v = ((px + py) * pz) % 1;
  return v < 0 ? v + 1 : v;
}

function valueNoise3DCpu(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = hash31Cpu(ix, iy, iz);
  const n100 = hash31Cpu(ix + 1, iy, iz);
  const n010 = hash31Cpu(ix, iy + 1, iz);
  const n110 = hash31Cpu(ix + 1, iy + 1, iz);
  const n001 = hash31Cpu(ix, iy, iz + 1);
  const n101 = hash31Cpu(ix + 1, iy, iz + 1);
  const n011 = hash31Cpu(ix, iy + 1, iz + 1);
  const n111 = hash31Cpu(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;
  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

function fbm3DCpu(x: number, y: number, z: number): number {
  const v0 = valueNoise3DCpu(x, y, z);
  const c = 0.80901699;
  const s = 0.58778525;
  const x1 = (x * c - z * s) * 2.03;
  const y1 = (y * c + x * s) * 2.03;
  const z1 = (z * c + y * s) * 2.03;
  const v1 = valueNoise3DCpu(x1, y1, z1);
  const x2 = (y1 * c - x1 * s) * 2.07;
  const y2 = (z1 * c + y1 * s) * 2.07;
  const z2 = (x1 * c + z1 * s) * 2.07;
  const v2 = valueNoise3DCpu(x2, y2, z2);
  return v0 * 0.57 + v1 * 0.29 + v2 * 0.14;
}

/**
 * Clips a convex polygon loop against a plane n · p <= d
 */
function clipPolygonByPlane(
  vertices: THREE.Vector3[],
  normal: THREE.Vector3,
  offset: number
): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  const n = vertices.length;
  if (n === 0) return out;

  for (let i = 0; i < n; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % n];
    const da = normal.dot(a) - offset;
    const db = normal.dot(b) - offset;

    if (da <= 1e-6) {
      out.push(a.clone());
      if (db > 1e-6) {
        const t = da / (da - db);
        out.push(a.clone().lerp(b, t));
      }
    } else if (db <= 1e-6) {
      const t = da / (da - db);
      out.push(a.clone().lerp(b, t));
    }
  }
  return out;
}

function clipPolyhedronByPlane(
  faces: THREE.Vector3[][],
  normal: THREE.Vector3,
  offset: number
): THREE.Vector3[][] {
  const nextFaces: THREE.Vector3[][] = [];
  const capPoints: THREE.Vector3[] = [];

  for (let f = 0; f < faces.length; f++) {
    const clipped = clipPolygonByPlane(faces[f], normal, offset);
    if (clipped.length >= 3) {
      nextFaces.push(clipped);
      for (let i = 0; i < clipped.length; i++) {
        const pt = clipped[i];
        if (Math.abs(normal.dot(pt) - offset) < 1e-4) {
          let duplicate = false;
          for (let j = 0; j < capPoints.length; j++) {
            if (capPoints[j].distanceToSquared(pt) < 1e-6) {
              duplicate = true;
              break;
            }
          }
          if (!duplicate) capPoints.push(pt.clone());
        }
      }
    }
  }

  if (capPoints.length >= 3) {
    const center = new THREE.Vector3();
    for (let i = 0; i < capPoints.length; i++) center.add(capPoints[i]);
    center.divideScalar(capPoints.length);

    const ref = Math.abs(normal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(normal, ref).normalize();
    const v = new THREE.Vector3().crossVectors(normal, u).normalize();

    capPoints.sort((a, b) => {
      const angA = Math.atan2(
        (a.x - center.x) * v.x + (a.y - center.y) * v.y + (a.z - center.z) * v.z,
        (a.x - center.x) * u.x + (a.y - center.y) * u.y + (a.z - center.z) * u.z
      );
      const angB = Math.atan2(
        (b.x - center.x) * v.x + (b.y - center.y) * v.y + (b.z - center.z) * v.z,
        (b.x - center.x) * u.x + (b.y - center.y) * u.y + (b.z - center.z) * u.z
      );
      return angA - angB;
    });

    nextFaces.push(capPoints);
  }

  return nextFaces;
}

interface RawTri {
  p0: THREE.Vector3;
  p1: THREE.Vector3;
  p2: THREE.Vector3;
  norm: THREE.Vector3;
  isCutFace: number;
  isPerimeterRim: number;
  isInnerFace: number;
}

/**
 * Uniformly subdivides a triangle mesh into a dense, welded topological lattice (`4^passes` triangles per base tri)
 * so 3D rock-slab sculpting and chiseled edge displacement are 100% watertight with zero T-junction cracks!
 */
function uniformSubdivideTriangles(tris: RawTri[], passes: number): RawTri[] {
  let current = tris;
  for (let p = 0; p < passes; p++) {
    const next: RawTri[] = [];
    for (let i = 0; i < current.length; i++) {
      const t = current[i];
      const m01 = t.p0.clone().add(t.p1).multiplyScalar(0.5);
      const m12 = t.p1.clone().add(t.p2).multiplyScalar(0.5);
      const m20 = t.p2.clone().add(t.p0).multiplyScalar(0.5);
      next.push(
        {
          p0: t.p0.clone(),
          p1: m01,
          p2: m20,
          norm: t.norm.clone(),
          isCutFace: t.isCutFace,
          isPerimeterRim: t.isPerimeterRim,
          isInnerFace: t.isInnerFace,
        },
        {
          p0: m01.clone(),
          p1: t.p1.clone(),
          p2: m12,
          norm: t.norm.clone(),
          isCutFace: t.isCutFace,
          isPerimeterRim: t.isPerimeterRim,
          isInnerFace: t.isInnerFace,
        },
        {
          p0: m20.clone(),
          p1: m12.clone(),
          p2: t.p2.clone(),
          norm: t.norm.clone(),
          isCutFace: t.isCutFace,
          isPerimeterRim: t.isPerimeterRim,
          isInnerFace: t.isInnerFace,
        },
        {
          p0: m01.clone(),
          p1: m12.clone(),
          p2: m20.clone(),
          norm: t.norm.clone(),
          isCutFace: t.isCutFace,
          isPerimeterRim: t.isPerimeterRim,
          isInnerFace: t.isInnerFace,
        }
      );
    }
    current = next;
  }
  return current;
}

/**
 * Evaluates a smooth 3D geological rock-slab displacement field:
 * - Stepped sedimentary travertine/flagstone ledges along horizontal bedding planes
 * - Chiseled stone crust undulations around the outer perimeter
 * - Concave vesicular stone pockets (travertine vugs / stone-bread crumb cavities)
 * Strictly continuous in 3D `(x, y, z)` so welded vertices never crack or notch!
 */
function evalSculptedStoneFieldCpu(bx: number, by: number, bz: number): THREE.Vector3 {
  const warp = fbm3DCpu(bx * 1.8 + 2.1, by * 1.8 - 1.3, bz * 2.2 + 0.7) - 0.5;
  // Gently undulating horizontal flagstone/travertine bedding layers
  const bedCoord = by * 5.4 + bx * 0.9 + bz * 1.8 + warp * 1.35;
  const bedFloor = Math.floor(bedCoord);
  const bedFract = bedCoord - bedFloor;
  const steppedLedge = (bedFloor + smoothstepScalar(0.32, 0.68, bedFract)) / 5.4 - by;

  // Medium chiseled stone face undulation
  const rockPlateX = fbm3DCpu(bx * 3.1 + 0.4, by * 3.1 + 1.9, bz * 3.6) - 0.5;
  const rockPlateY = fbm3DCpu(bx * 3.1 - 2.7, by * 3.1 + 0.8, bz * 3.6 + 1.4) - 0.5;
  const rockPlateZ = fbm3DCpu(bx * 3.4 + 1.1, by * 3.4 - 1.6, bz * 3.4 - 0.9) - 0.5;

  // Keep the 45° diagonal cut seam tight so closed halves meet cleanly, while sculpting the stone slab & crust
  const diagDist = Math.abs(bx - by) * INV_SQRT2;
  const cutKeepFlat = smoothstepScalar(0.015, 0.14, diagDist);
  const radialDist = Math.max(Math.abs(bx), Math.abs(by)) / BLOCK_HALF_W;
  const crustSculptBoost = 0.65 + 0.75 * smoothstepScalar(0.55, 0.95, radialDist);

  const scale = cutKeepFlat * crustSculptBoost;
  return new THREE.Vector3(
    (rockPlateX * 0.022 + steppedLedge * 0.014) * scale,
    (rockPlateY * 0.022 + steppedLedge * 0.01) * scale,
    (rockPlateZ * 0.016 + steppedLedge * 0.012) * cutKeepFlat
  );
}

/**
 * Builds one of the two sculpted Travertine / Flagstone Bread Slices for a given triangular half
 * (`side = -1` for Top-Left wedge, `+1` for Bottom-Right wedge;
 *  `sliceZSign = +1` for Front Slice, `-1` for Back Slice).
 */
function buildSandstoneSliceGeometry(side: -1 | 1, sliceZSign: -1 | 1): THREE.BufferGeometry {
  const sx = BLOCK_HALF_W * 1.15;
  const sy = BLOCK_HALF_H * 1.15;
  const sz = BLOCK_HALF_Z * 1.15;

  const v = [
    new THREE.Vector3(-sx, -sy, -sz),
    new THREE.Vector3(sx, -sy, -sz),
    new THREE.Vector3(sx, sy, -sz),
    new THREE.Vector3(-sx, sy, -sz),
    new THREE.Vector3(-sx, -sy, sz),
    new THREE.Vector3(sx, -sy, sz),
    new THREE.Vector3(sx, sy, sz),
    new THREE.Vector3(-sx, sy, sz),
  ];

  let faces: THREE.Vector3[][] = [
    [v[0], v[3], v[2], v[1]],
    [v[4], v[5], v[6], v[7]],
    [v[0], v[1], v[5], v[4]],
    [v[2], v[3], v[7], v[6]],
    [v[0], v[4], v[7], v[3]],
    [v[1], v[2], v[6], v[5]],
  ];

  // Primary XY bounding planes
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(1, 0, 0), BLOCK_HALF_W);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(-1, 0, 0), BLOCK_HALF_W);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 1, 0), BLOCK_HALF_H);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, -1, 0), BLOCK_HALF_H);

  // Slice Z bounds:
  // Front slice (+1): z in [SLICE_INNER_Z, BLOCK_HALF_Z]
  // Back slice  (-1): z in [-BLOCK_HALF_Z, -SLICE_INNER_Z]
  if (sliceZSign === 1) {
    faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 0, 1), BLOCK_HALF_Z);
    faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 0, -1), -SLICE_INNER_Z);
  } else {
    faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 0, -1), BLOCK_HALF_Z);
    faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 0, 1), -SLICE_INNER_Z);
  }

  // Multi-stage chiseled stone edge bevels (2-tier beveling so corners and edges are rounded/faceted rock, not a sharp box!)
  const bevelXY1 = (BLOCK_HALF_W + BLOCK_HALF_H) * INV_SQRT2 - 0.068;
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(INV_SQRT2, INV_SQRT2, 0), bevelXY1);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(-INV_SQRT2, -INV_SQRT2, 0), bevelXY1);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(-INV_SQRT2, INV_SQRT2, 0), bevelXY1);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(INV_SQRT2, -INV_SQRT2, 0), bevelXY1);

  const outerBevelZ = (BLOCK_HALF_W + BLOCK_HALF_Z) * INV_SQRT2 - 0.054;
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(INV_SQRT2, 0, sliceZSign * INV_SQRT2),
    outerBevelZ
  );
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(-INV_SQRT2, 0, sliceZSign * INV_SQRT2),
    outerBevelZ
  );
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(0, INV_SQRT2, sliceZSign * INV_SQRT2),
    outerBevelZ
  );
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(0, -INV_SQRT2, sliceZSign * INV_SQRT2),
    outerBevelZ
  );

  // Generous inner Z bevels so the recessed molten cleft between the Front & Back stone slices is clearly visible!
  const innerBevelDist = (BLOCK_HALF_W - SLICE_INNER_Z) * INV_SQRT2 - 0.038;
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(INV_SQRT2, 0, -sliceZSign * INV_SQRT2),
    innerBevelDist
  );
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(-INV_SQRT2, 0, -sliceZSign * INV_SQRT2),
    innerBevelDist
  );
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(0, INV_SQRT2, -sliceZSign * INV_SQRT2),
    innerBevelDist
  );
  faces = clipPolyhedronByPlane(
    faces,
    new THREE.Vector3(0, -INV_SQRT2, -sliceZSign * INV_SQRT2),
    innerBevelDist
  );

  // 45° diagonal slice plane
  const cutNormal =
    side === -1
      ? new THREE.Vector3(PULL_NX, PULL_NY, 0)
      : new THREE.Vector3(-PULL_NX, -PULL_NY, 0);
  faces = clipPolyhedronByPlane(faces, cutNormal, -0.002);

  // Chamfer along the 45° diagonal cut rim so the glowing oozing bolster sits proud in a stone V-channel
  const lipNormalZ = new THREE.Vector3(
    cutNormal.x * 0.78,
    cutNormal.y * 0.78,
    sliceZSign * 0.625
  ).normalize();
  faces = clipPolyhedronByPlane(faces, lipNormalZ, BLOCK_HALF_Z * 0.625 - 0.022);

  const sliceMidZ = sliceZSign * ((BLOCK_HALF_Z + SLICE_INNER_Z) * 0.5);
  const wedgeInteriorCheck = new THREE.Vector3(-cutNormal.x * 0.35, -cutNormal.y * 0.35, sliceMidZ);

  const baseTris: RawTri[] = [];
  const faceCenter = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNorm = new THREE.Vector3();

  for (let f = 0; f < faces.length; f++) {
    const poly = faces[f];
    if (poly.length < 3) continue;

    faceCenter.set(0, 0, 0);
    for (let i = 0; i < poly.length; i++) faceCenter.add(poly[i]);
    faceCenter.divideScalar(poly.length);

    ab.subVectors(poly[1], poly[0]);
    ac.subVectors(poly[2], poly[0]);
    faceNorm.crossVectors(ab, ac).normalize();

    if (faceNorm.dot(faceCenter.clone().sub(wedgeInteriorCheck)) < 0) {
      faceNorm.negate();
      poly.reverse();
    }

    const cutAlign = faceNorm.dot(cutNormal);
    const isCutFace = cutAlign > 0.62 ? 1.0 : 0.0;
    const isInnerFace = faceNorm.z * sliceZSign < -0.45 ? 1.0 : 0.0;
    const isPerimeterRim = Math.abs(faceNorm.z) < 0.65 && isCutFace < 0.5 ? 1.0 : 0.0;

    for (let i = 0; i < poly.length; i++) {
      baseTris.push({
        p0: faceCenter.clone(),
        p1: poly[i].clone(),
        p2: poly[(i + 1) % poly.length].clone(),
        norm: faceNorm.clone(),
        isCutFace,
        isPerimeterRim,
        isInnerFace,
      });
    }
  }

  // Uniform 3-pass subdivision (64x triangles per base triangle) for watertight 3D sculpted stone relief
  const fineTris = uniformSubdivideTriangles(baseTris, 3);

  const quantScale = 4000;
  const keyOf = (p: THREE.Vector3) =>
    `${Math.round(p.x * quantScale)},${Math.round(p.y * quantScale)},${Math.round(p.z * quantScale)}`;

  // Displace all vertices deterministically by their 3D block position so shared edges remain 100% welded!
  const dispCache = new Map<string, THREE.Vector3>();
  const getDisplacedPos = (orig: THREE.Vector3): THREE.Vector3 => {
    const k = keyOf(orig);
    let dp = dispCache.get(k);
    if (!dp) {
      const offset = evalSculptedStoneFieldCpu(orig.x, orig.y, orig.z);
      dp = orig.clone().add(offset);
      dispCache.set(k, dp);
    }
    return dp;
  };

  const weldedNormMap = new Map<string, THREE.Vector3>();
  const triNormals: THREE.Vector3[] = [];
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();

  for (let i = 0; i < fineTris.length; i++) {
    const t = fineTris[i];
    const d0 = getDisplacedPos(t.p0);
    const d1 = getDisplacedPos(t.p1);
    const d2 = getDisplacedPos(t.p2);

    e1.subVectors(d1, d0);
    e2.subVectors(d2, d0);
    const tn = new THREE.Vector3().crossVectors(e1, e2);
    if (tn.lengthSq() > 1e-12) {
      tn.normalize();
    } else {
      tn.copy(t.norm);
    }
    triNormals.push(tn);

    for (const pt of [t.p0, t.p1, t.p2]) {
      const k = keyOf(pt);
      let wn = weldedNormMap.get(k);
      if (!wn) {
        wn = new THREE.Vector3();
        weldedNormMap.set(k, wn);
      }
      wn.add(tn);
    }
  }

  weldedNormMap.forEach((n) => n.normalize());

  const positions: number[] = [];
  const normals: number[] = [];
  const blockCoords: number[] = [];
  const faceMeta: number[] = [];

  const finalN = new THREE.Vector3();

  for (let i = 0; i < fineTris.length; i++) {
    const t = fineTris[i];
    const tn = triNormals[i];
    const pts = [t.p0, t.p1, t.p2];

    for (let k = 0; k < 3; k++) {
      const orig = pts[k];
      const dp = getDisplacedPos(orig);
      const wn = weldedNormMap.get(keyOf(orig)) ?? tn;

      // 72% welded smooth normal + 28% facet normal gives organic rounded stone bevels with crisp chiseled ledges
      finalN.copy(wn).multiplyScalar(0.72).addScaledVector(tn, 0.28).normalize();

      positions.push(dp.x, dp.y, dp.z);
      blockCoords.push(orig.x, orig.y, orig.z);
      normals.push(finalN.x, finalN.y, finalN.z);
      faceMeta.push(t.isCutFace, t.isPerimeterRim, side, t.isInnerFace);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geom.setAttribute('blockPos', new THREE.Float32BufferAttribute(blockCoords, 3));
  geom.setAttribute('faceMeta', new THREE.Float32BufferAttribute(faceMeta, 4));
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Builds the physical Molten Cheese / Lava Cushion wedged inside each triangular sandwich half
 * (`side = -1` for Top-Left, `+1` for Bottom-Right).
 * Recessed slightly inside the outer perimeter (`BLOCK_HALF_W - 0.042`) so the two stone bread slices
 * have a distinct, shadowy cleft between them around the edges, with molten lava bulging in the middle!
 */
function buildMoltenSandwichCushionGeometry(side: -1 | 1): THREE.BufferGeometry {
  const sx = BLOCK_HALF_W * 1.1;
  const sy = BLOCK_HALF_H * 1.1;
  const sz = CORE_CUSHION_HALF_Z * 1.15;

  const v = [
    new THREE.Vector3(-sx, -sy, -sz),
    new THREE.Vector3(sx, -sy, -sz),
    new THREE.Vector3(sx, sy, -sz),
    new THREE.Vector3(-sx, sy, -sz),
    new THREE.Vector3(-sx, -sy, sz),
    new THREE.Vector3(sx, -sy, sz),
    new THREE.Vector3(sx, sy, sz),
    new THREE.Vector3(-sx, sy, sz),
  ];

  let faces: THREE.Vector3[][] = [
    [v[0], v[3], v[2], v[1]],
    [v[4], v[5], v[6], v[7]],
    [v[0], v[1], v[5], v[4]],
    [v[2], v[3], v[7], v[6]],
    [v[0], v[4], v[7], v[3]],
    [v[1], v[2], v[6], v[5]],
  ];

  // Recessed 0.038 units inside the stone slice perimeter so the two stone slices read as separate slabs on the sides!
  const cushionHalfW = BLOCK_HALF_W - 0.038;
  const cushionHalfH = BLOCK_HALF_H - 0.038;
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(1, 0, 0), cushionHalfW);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(-1, 0, 0), cushionHalfW);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 1, 0), cushionHalfH);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, -1, 0), cushionHalfH);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 0, 1), CORE_CUSHION_HALF_Z);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(0, 0, -1), CORE_CUSHION_HALF_Z);

  const bevelXY = (cushionHalfW + cushionHalfH) * INV_SQRT2 - 0.048;
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(INV_SQRT2, INV_SQRT2, 0), bevelXY);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(-INV_SQRT2, -INV_SQRT2, 0), bevelXY);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(-INV_SQRT2, INV_SQRT2, 0), bevelXY);
  faces = clipPolyhedronByPlane(faces, new THREE.Vector3(INV_SQRT2, -INV_SQRT2, 0), bevelXY);

  const cutNormal =
    side === -1
      ? new THREE.Vector3(PULL_NX, PULL_NY, 0)
      : new THREE.Vector3(-PULL_NX, -PULL_NY, 0);
  faces = clipPolyhedronByPlane(faces, cutNormal, 0.002);

  const interiorCheck = new THREE.Vector3(-cutNormal.x * 0.35, -cutNormal.y * 0.35, 0);
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const strandMeta: number[] = [];

  const faceCenter = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNorm = new THREE.Vector3();

  for (let f = 0; f < faces.length; f++) {
    const poly = faces[f];
    if (poly.length < 3) continue;

    faceCenter.set(0, 0, 0);
    for (let i = 0; i < poly.length; i++) faceCenter.add(poly[i]);
    faceCenter.divideScalar(poly.length);

    ab.subVectors(poly[1], poly[0]);
    ac.subVectors(poly[2], poly[0]);
    faceNorm.crossVectors(ab, ac).normalize();

    if (faceNorm.dot(faceCenter.clone().sub(interiorCheck)) < 0) {
      faceNorm.negate();
      poly.reverse();
    }

    for (let i = 0; i < poly.length; i++) {
      const pts = [faceCenter, poly[i], poly[(i + 1) % poly.length]];
      for (let k = 0; k < 3; k++) {
        const p = pts[k];
        positions.push(p.x, p.y, p.z);
        normals.push(faceNorm.x, faceNorm.y, faceNorm.z);
        const sCoord = (p.x * DIAG_TANGENT_X + p.y * DIAG_TANGENT_Y) * 0.5 + 0.5;
        uvs.push(sCoord, (p.z / CORE_CUSHION_HALF_Z) * 0.5 + 0.5);
        const swirl = clampScalar(
          0.35 + 0.45 * Math.sin(p.x * 4.5 - p.y * 3.8 + p.z * 6.0),
          0.1,
          0.9
        );
        strandMeta.push(0.48, 0.15, swirl, 0.44);
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geom.setAttribute('strandMeta', new THREE.Float32BufferAttribute(strandMeta, 4));
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Allocates a procedural tubular BufferGeometry for a viscoelastic molten strand or slice-adhered bolster.
 */
function createDynamicTubeGeometry(
  lengthSegments: number,
  radialSegments: number
): {
  geometry: THREE.BufferGeometry;
  posAttr: THREE.BufferAttribute;
  normAttr: THREE.BufferAttribute;
  metaAttr: THREE.BufferAttribute;
} {
  const ringCount = lengthSegments + 1;
  const vertPerRing = radialSegments + 1;
  const vertexCount = ringCount * vertPerRing;

  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  // .x = radialThickness (mm chord for Beer-Lambert), .y = stretchStress [0..1], .z = pigmentMix, .w = neckGlow
  const strandMeta = new Float32Array(vertexCount * 4);
  const indices: number[] = [];

  for (let i = 0; i < ringCount; i++) {
    const v = i / lengthSegments;
    for (let j = 0; j < vertPerRing; j++) {
      const u = j / radialSegments;
      const idx = i * vertPerRing + j;
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  for (let i = 0; i < lengthSegments; i++) {
    for (let j = 0; j < radialSegments; j++) {
      const a = i * vertPerRing + j;
      const b = (i + 1) * vertPerRing + j;
      const c = (i + 1) * vertPerRing + (j + 1);
      const d = i * vertPerRing + (j + 1);
      indices.push(a, b, d, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  const normAttr = new THREE.BufferAttribute(normals, 3);
  const metaAttr = new THREE.BufferAttribute(strandMeta, 4);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  normAttr.setUsage(THREE.DynamicDrawUsage);
  metaAttr.setUsage(THREE.DynamicDrawUsage);

  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('normal', normAttr);
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('strandMeta', metaAttr);
  geometry.setIndex(indices);

  return { geometry, posAttr, normAttr, metaAttr };
}

const _pA = new THREE.Vector3();
const _pB = new THREE.Vector3();
const _pt = new THREE.Vector3();
const _tang = new THREE.Vector3();
const _upRef = new THREE.Vector3(0, 0, 1);
const _binorm = new THREE.Vector3();
const _normVec = new THREE.Vector3();
const _seamTang = new THREE.Vector3(DIAG_TANGENT_X, DIAG_TANGENT_Y, 0);

const _spineBufX = new Float32Array(64);
const _spineBufY = new Float32Array(64);
const _spineBufZ = new Float32Array(64);
const _spineBufR = new Float32Array(64);
const _spineBufW = new Float32Array(64);
const _spineBufGlow = new Float32Array(64);

/**
 * Updates the sculptural molten bolsters & lip ooze tubes.
 * EVERY bolster is parented to either Wedge A (`wedgeSide = -1`) or Wedge B (`wedgeSide = +1`)
 * so ZERO geometry ever floats in empty air at `(0, 0)` when the slices pull apart!
 * Features strong organic variability (slumped lava-cheese gobs, thick pooled lobes, and thin stretches).
 */
function updateOozeBeadTube(
  bead: OozeBeadTube,
  openDist: number,
  squeezeAmount: number,
  elapsedSec: number
): void {
  const ringCount = BEAD_LENGTH_SEGMENTS + 1;
  const vertPerRing = BEAD_RADIAL_SEGMENTS + 1;

  const isOuterLip = Math.abs(bead.sideZ) > 0.5;
  const posArr = bead.posAttr.array as Float32Array;
  const normArr = bead.normAttr.array as Float32Array;
  const metaArr = bead.metaAttr.array as Float32Array;

  for (let i = 0; i < ringCount; i++) {
    const u = i / BEAD_LENGTH_SEGMENTS;
    // Span along the 45° diagonal cut inside the beveled stone corners (-1.05 to +1.05)
    const s = -1.05 + u * 2.1;

    const endEnvelope = Math.pow(Math.sin(u * Math.PI), 0.65);

    // Highly irregular, non-uniform lava-cheese gobs along the seam (3 asymmetric pooled reservoirs + drips)
    const gob1 = Math.exp(-Math.pow((u - 0.17) / 0.11, 2.0)) * (0.52 + squeezeAmount * 0.65);
    const gob2 = Math.exp(-Math.pow((u - 0.46) / 0.14, 2.0)) * (0.44 + squeezeAmount * 0.48);
    const gob3 = Math.exp(-Math.pow((u - 0.74) / 0.1, 2.0)) * (0.35 + squeezeAmount * 0.42);
    const irregularLavaLobes =
      0.62 +
      0.55 * (gob1 + gob2 * 0.85 + gob3 * 0.7) +
      0.24 * Math.sin(u * Math.PI * 5.0 + bead.wedgeSide * 1.4 + elapsedSec * 0.85) +
      0.16 * Math.cos(u * Math.PI * 11.0 - bead.sideZ * 1.9 - elapsedSec * 0.6);

    const baseR = isOuterLip
      ? (0.017 + squeezeAmount * 0.025) * endEnvelope * clampScalar(1.0 - openDist * 0.55, 0.35, 1.0)
      : (0.032 + squeezeAmount * 0.022 + Math.min(0.024, openDist * 0.055)) * endEnvelope;

    const radius = baseR * irregularLavaLobes;

    // Asymmetric gravity slump & dripping lobes that hang down in -Y from the cut lip
    const gravSlump =
      (0.01 + squeezeAmount * 0.028 + openDist * 0.018) *
      (gob1 * 1.15 + gob2 * 0.75 + 0.3 * endEnvelope);
    const cornerDroopY = -(gob1 * 0.038 + gob2 * 0.022) * (0.7 + squeezeAmount * 0.65);

    // In local wedge coordinates: `bead.wedgeSide = -1` is Top-Left wedge; `+1` is Bottom-Right wedge
    // When closed (`openDist == 0`), the outer lip bolsters sit right on the 45° cut line (`inwardEmbed ~ 0`)
    // and fuse together; when pulled open, they stay adhered to their slice's cut lip!
    const inwardEmbed = isOuterLip
      ? bead.wedgeSide * (0.004 + Math.min(0.018, openDist * 0.04))
      : bead.wedgeSide * 0.012;

    const cx = DIAG_TANGENT_X * s + PULL_NX * (inwardEmbed + gravSlump * 0.35);
    const cy = DIAG_TANGENT_Y * s + PULL_NY * inwardEmbed - gravSlump * 0.65 + cornerDroopY;

    const cz = isOuterLip
      ? bead.sideZ *
        (BLOCK_HALF_Z -
          0.022 +
          (0.01 + squeezeAmount * 0.024) * endEnvelope * irregularLavaLobes)
      : Math.sin(u * Math.PI * 4.5 + bead.wedgeSide) * 0.022;

    _spineBufX[i] = cx;
    _spineBufY[i] = cy;
    _spineBufZ[i] = cz;
    _spineBufR[i] = Math.max(0.0008, radius);
    _spineBufGlow[i] = clampScalar(0.36 + squeezeAmount * 0.38 + gob1 * 0.32 + gob2 * 0.22, 0.1, 0.95);
  }

  _tang.set(DIAG_TANGENT_X, DIAG_TANGENT_Y, 0);
  _binorm.set(PULL_NX, PULL_NY, 0);
  _normVec.set(0, 0, 1);

  const zScale = isOuterLip ? 0.85 : 2.35;
  const xyScale = isOuterLip ? 1.12 : 1.22;

  for (let i = 0; i < ringCount; i++) {
    const r = _spineBufR[i];
    const glow = _spineBufGlow[i];
    const cx = _spineBufX[i];
    const cy = _spineBufY[i];
    const cz = _spineBufZ[i];
    const u = i / BEAD_LENGTH_SEGMENTS;
    const pigmentSwirl = 0.25 + 0.55 * Math.sin(u * Math.PI * 4.2 + bead.sideZ + bead.wedgeSide);

    for (let j = 0; j < vertPerRing; j++) {
      const ang = (j / BEAD_RADIAL_SEGMENTS) * Math.PI * 2.0;
      const cosA = Math.cos(ang);
      const sinA = Math.sin(ang);

      const nx = _binorm.x * cosA;
      const ny = _binorm.y * cosA;
      const nz = sinA;

      const vIdx = (i * vertPerRing + j) * 3;
      posArr[vIdx] = cx + nx * r * xyScale;
      posArr[vIdx + 1] = cy + ny * r * xyScale;
      posArr[vIdx + 2] = cz + nz * r * zScale;

      normArr[vIdx] = nx;
      normArr[vIdx + 1] = ny;
      normArr[vIdx + 2] = nz;

      const mIdx = (i * vertPerRing + j) * 4;
      metaArr[mIdx] = r * 16.0;
      metaArr[mIdx + 1] = squeezeAmount * 0.45;
      metaArr[mIdx + 2] = pigmentSwirl;
      metaArr[mIdx + 3] = glow;
    }
  }

  bead.posAttr.needsUpdate = true;
  bead.normAttr.needsUpdate = true;
  bead.metaAttr.needsUpdate = true;
  bead.mesh.geometry.computeBoundingSphere();
}

/**
 * Updates the vertex positions & analytical Frenet normals of a viscoelastic lava-cheese bridge.
 * - Supports 4 distinct archetypes (`mother-lobe`, `twisted-rope`, `gossamer-thread`, `gravity-drip`)
 * - Deforms smoothly when stirred/touched by the pointer (`stirDispX`, `stirDispY`)
 * - If snapped (`snapProgress > 0`), retracts as a single continuous pendant lobe into its assigned anchor
 *   wedge (`retractSide`), NEVER splitting one tube mesh across the open gap (eliminating all bridge lines!).
 */
function updateMoltenStrandGeometry(
  strand: MoltenStrand,
  wedgeGroupA: THREE.Group,
  wedgeGroupB: THREE.Group,
  gapDist: number,
  squeezeAmount: number,
  elapsedSec: number
): void {
  // Exact rigid-body transform (rotation + translation) of each wedge so strands NEVER detach when slices tilt
  _pA.copy(strand.anchorA).applyEuler(wedgeGroupA.rotation).add(wedgeGroupA.position);
  _pB.copy(strand.anchorB).applyEuler(wedgeGroupB.rotation).add(wedgeGroupB.position);

  const stretchRatio = clampScalar(gapDist / Math.max(0.1, strand.criticalGap), 0, 1.3);
  const snapP = strand.snapProgress;

  // Archetype-specific gravitational sag:
  // `gravity-drip` and `mother-lobe` sag much heavier in -Y than taut `gossamer-thread` filaments!
  const archGravityMult =
    strand.archetype === 'gravity-drip'
      ? 1.95
      : strand.archetype === 'mother-lobe'
        ? 1.35
        : strand.archetype === 'twisted-rope'
          ? 1.05
          : 0.52;

  const bottomBias = clampScalar((0.6 - strand.seamS) * 0.55, 0.32, 1.35);
  const gravitySag =
    (0.016 + Math.max(0, gapDist) * 0.28 * bottomBias * archGravityMult) *
    (0.75 + 0.25 * Math.sin(strand.index * 2.1 + elapsedSec * 1.25)) *
    strand.cornerTaper;

  const ringCount = STRAND_LENGTH_SEGMENTS + 1;
  const vertPerRing = STRAND_RADIAL_SEGMENTS + 1;

  const posArr = strand.posAttr.array as Float32Array;
  const normArr = strand.normAttr.array as Float32Array;
  const metaArr = strand.metaAttr.array as Float32Array;

  // Travelling viscous lava-cheese globule / bead position along the strand
  const globuleCenterT =
    0.5 + 0.34 * Math.sin(elapsedSec * strand.globuleSpeed + strand.phaseOffset);

  for (let i = 0; i < ringCount; i++) {
    const t = i / STRAND_LENGTH_SEGMENTS;

    if (snapP <= 0.0001) {
      // Intact viscoelastic bridge between Slice Wedge A and Slice Wedge B
      _pt.lerpVectors(_pA, _pB, t);

      const arch = 4.0 * t * (1.0 - t);
      const asymDist = (t - strand.neckCenterT) / 0.48;
      const asymArch = Math.max(0.0, 1.0 - asymDist * asymDist) * arch;

      // Viscous catenary sag downward in -Y and along lower-right pull normal
      _pt.y -= gravitySag * arch;
      _pt.x += PULL_NX * gravitySag * 0.2 * arch;

      // Helical twist for braided `twisted-rope` strands + organic cluster bowing
      const twistAngle = t * Math.PI * 2.0 * strand.twistTurns + strand.phaseOffset + elapsedSec * 0.9;
      const twistOffsetTang = Math.sin(twistAngle) * strand.twistAmp * Math.min(1.0, gapDist * 2.6) * arch;
      const twistOffsetZ = Math.cos(twistAngle) * strand.twistAmp * 0.75 * Math.min(1.0, gapDist * 2.6) * arch;

      // Interactive pointer stirring ("fondue drag") + cluster attraction
      const clusterBow =
        Math.sin(strand.seamS * 3.8 + strand.phaseOffset) *
        0.028 *
        Math.min(1.0, gapDist * 2.5) *
        asymArch;

      _pt.x += strand.stirDispX * arch + DIAG_TANGENT_X * (twistOffsetTang + clusterBow);
      _pt.y += strand.stirDispY * arch + DIAG_TANGENT_Y * (twistOffsetTang + clusterBow);

      const zSign = strand.coreZ >= 0 ? 1 : -1;
      const isOuterCurtain = Math.abs(strand.coreZ) > 0.14 ? 1.0 : 0.45;
      const squeezeOutZ =
        squeezeAmount * 0.042 * clampScalar(gapDist * 3.5, 0, 1) * arch * zSign * isOuterCurtain;
      _pt.z += squeezeOutZ + twistOffsetZ + Math.sin(elapsedSec * 1.5 + strand.phaseOffset) * 0.004 * arch;

      // Multi-scale radius profile with strong archetype variability:
      // 1) Wide parabolic meniscus feet at t -> 0 and t -> 1
      const distFromEnds = Math.abs(t - 0.5) * 2.0;
      const footFlare = Math.pow(distFromEnds, 1.65) * 1.42 * strand.cornerTaper;

      // 2) Asymmetric viscoelastic necking centered at `strand.neckCenterT`
      const neckPinchEnv = Math.exp(
        -Math.pow((t - strand.neckCenterT) * strand.neckSharpness, 2.0)
      );
      const minNeckFloor = strand.archetype === 'gossamer-thread' ? 0.09 : 0.18;
      const neckThinning = Math.max(minNeckFloor, 1.0 - stretchRatio * 0.82 * neckPinchEnv);

      // 3) Slow-moving molten lava-cheese globules / Rayleigh-Plateau beads along the strand
      const globuleBump =
        strand.globuleAmp *
        Math.exp(-Math.pow((t - globuleCenterT) / 0.14, 2.0)) *
        arch *
        (0.5 + 0.5 * stretchRatio);
      const secondaryBead =
        strand.globuleAmp *
        0.55 *
        Math.exp(-Math.pow((t - (1.0 - globuleCenterT)) / 0.11, 2.0)) *
        arch;

      const squeezeFatten = 1.0 + squeezeAmount * 0.48 * arch;
      const rawRadius =
        strand.baseRadius *
        (neckThinning * squeezeFatten * (1.0 + globuleBump + secondaryBead) +
          footFlare * (0.85 + stretchRatio * 0.45));

      const maxAllowedR =
        (strand.archetype === 'mother-lobe' ? 0.076 : 0.06) * strand.cornerTaper;
      const radius = clampScalar(rawRadius, 0.0024, maxAllowedR);

      const ribbonW = clampScalar(
        strand.ribbonWidthScale * (1.0 + Math.pow(distFromEnds, 1.5) * 0.36 * strand.cornerTaper),
        0.95,
        2.15
      );

      _spineBufX[i] = _pt.x;
      _spineBufY[i] = _pt.y;
      _spineBufZ[i] = _pt.z;
      _spineBufR[i] = radius;
      _spineBufW[i] = ribbonW;
      _spineBufGlow[i] = clampScalar(
        (stretchRatio * neckPinchEnv * 0.68 + globuleBump * 0.45 + squeezeAmount * 0.35 + 0.16) *
          (0.52 + 0.48 * strand.depthLayer01),
        0.06,
        0.95
      );
    } else {
      // Snapped strand: retracts as ONE continuous pendant droplet lobe into its assigned `retractSide` wedge!
      // Because all rings [0..STRAND_LENGTH_SEGMENTS] stay on the same side of the gap, ZERO cross-gap lines can ever form!
      const recoilEase = 1.0 - Math.exp(-snapP * 4.8) * Math.cos(snapP * 7.5) * 0.2;
      const stubSpan = Math.max(0.06, (1.0 - clampScalar(recoilEase, 0, 0.86)) * 0.44);

      // uFromAnchor = 0 at the slice anchor, 1 at the recoiling pendant droplet tip
      const uFromAnchor = strand.retractSide === -1 ? t : 1.0 - t;
      const reachT = strand.retractSide === -1 ? uFromAnchor * stubSpan : 1.0 - uFromAnchor * stubSpan;
      _pt.lerpVectors(_pA, _pB, reachT);

      // Droop into the open gap (along the inward cut normal toward the center of the gap + gravity -Y)
      const intoGapSign = strand.retractSide === -1 ? 1.0 : -1.0;
      const gapCurvature = uFromAnchor * uFromAnchor * (0.035 + snapP * 0.055) * strand.cornerTaper;
      _pt.x += PULL_NX * intoGapSign * gapCurvature * 0.45;
      _pt.y += PULL_NY * intoGapSign * gapCurvature * 0.45 - gapCurvature * 0.65;

      // Pendant tear-drop bulb at uFromAnchor -> 0.82, pinching closed smoothly at uFromAnchor === 1.0
      const anchorFlare = Math.pow(1.0 - uFromAnchor, 1.8) * 1.2 * strand.cornerTaper;
      const dropletBulb =
        Math.sin(uFromAnchor * Math.PI * 0.86) * (0.85 + uFromAnchor * 0.75 * strand.globuleAmp);
      const tipClose = smoothstepScalar(1.0, 0.88, uFromAnchor);

      const radius = clampScalar(
        strand.baseRadius * (0.65 * dropletBulb + anchorFlare) * tipClose * (1.0 - snapP * 0.25),
        0.0,
        0.058 * strand.cornerTaper
      );

      _spineBufX[i] = _pt.x;
      _spineBufY[i] = _pt.y;
      _spineBufZ[i] = _pt.z;
      _spineBufR[i] = radius;
      _spineBufW[i] = clampScalar(strand.ribbonWidthScale * (1.0 - uFromAnchor * 0.35), 1.0, 1.65);
      _spineBufGlow[i] = clampScalar(
        ((1.0 - snapP * 0.4) * uFromAnchor * 0.65 + 0.18) * (0.52 + 0.48 * strand.depthLayer01),
        0.05,
        0.9
      );
    }
  }

  for (let i = 0; i < ringCount; i++) {
    const iPrev = Math.max(0, i - 1);
    const iNext = Math.min(ringCount - 1, i + 1);

    _tang.set(
      _spineBufX[iNext] - _spineBufX[iPrev],
      _spineBufY[iNext] - _spineBufY[iPrev],
      _spineBufZ[iNext] - _spineBufZ[iPrev]
    );
    if (_tang.lengthSq() < 1e-8) {
      _tang.set(PULL_NX, PULL_NY, 0);
    } else {
      _tang.normalize();
    }

    _binorm.crossVectors(_tang, _upRef);
    if (_binorm.lengthSq() < 1e-6) {
      _binorm.copy(_seamTang);
    } else {
      _binorm.normalize();
    }
    _normVec.crossVectors(_binorm, _tang).normalize();

    const effR = _spineBufR[i];
    const wScale = _spineBufW[i];
    const glow = _spineBufGlow[i];
    const cx = _spineBufX[i];
    const cy = _spineBufY[i];
    const cz = _spineBufZ[i];

    for (let j = 0; j < vertPerRing; j++) {
      const ang = (j / STRAND_RADIAL_SEGMENTS) * Math.PI * 2.0;
      const cosA = Math.cos(ang);
      const sinA = Math.sin(ang);

      const rx = _binorm.x * cosA * wScale + _normVec.x * sinA * 0.92;
      const ry = _binorm.y * cosA * wScale + _normVec.y * sinA * 0.92;
      const rz = _binorm.z * cosA * wScale + _normVec.z * sinA * 1.04;

      const nx = _binorm.x * (cosA / wScale) + _normVec.x * sinA;
      const ny = _binorm.y * (cosA / wScale) + _normVec.y * sinA;
      const nz = _binorm.z * (cosA / wScale) + _normVec.z * sinA;
      const nLen = Math.hypot(nx, ny, nz) || 1.0;

      const vIdx = (i * vertPerRing + j) * 3;
      posArr[vIdx] = cx + rx * effR;
      posArr[vIdx + 1] = cy + ry * effR;
      posArr[vIdx + 2] = cz + rz * effR;

      normArr[vIdx] = nx / nLen;
      normArr[vIdx + 1] = ny / nLen;
      normArr[vIdx + 2] = nz / nLen;

      const mIdx = (i * vertPerRing + j) * 4;
      metaArr[mIdx] = effR * 18.0;
      metaArr[mIdx + 1] = stretchRatio;
      metaArr[mIdx + 2] = strand.pigmentMix;
      metaArr[mIdx + 3] = glow;
    }
  }

  strand.posAttr.needsUpdate = true;
  strand.normAttr.needsUpdate = true;
  strand.metaAttr.needsUpdate = true;
  strand.mesh.geometry.computeBoundingSphere();
}

export async function bootDay03Melt(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(3);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.05, 4.65],
    fov: 35,
    enableOrbit: false,
  });

  // Physical mineral pigments from src/core/palette.ts
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const burntSienna = createPigmentUniforms('burntSienna');
  const rawUmber = createPigmentUniforms('rawUmber');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const sporeSulfur = createPigmentUniforms('sporeSulfur');

  // Shared TSL Uniforms for shader interaction & thermal glow
  const uGapOpen = uniform(0.0); // [0..1.1] diagonal pull separation
  const uSqueeze = uniform(0.26); // [0..1] compression ooze state
  const uCoreHeat = uniform(0.74); // [0..1.4] thermal incandescence along the molten seam

  const specimenRoot = new THREE.Group();
  specimenRoot.name = 'DiagonalPressedAmberSandwichSpecimen';
  // Presentation angle that clearly reveals the Front Stone Slice, the recessed molten cleft, and the Back Stone Slice!
  specimenRoot.rotation.set(0.18, -0.34, 0.03);
  stage.specimenGroup.add(specimenRoot);

  const seamCoreLight = new THREE.PointLight(0xf28729, 0.4, 1.85, 1.8);
  seamCoreLight.position.set(0.0, 0.0, 0.0);
  specimenRoot.add(seamCoreLight);

  const seamBackLight = new THREE.PointLight(0xe35b28, 0.3, 1.65, 1.8);
  seamBackLight.position.set(0.0, 0.0, 0.04);
  specimenRoot.add(seamBackLight);

  // ---------------------------------------------------------------------------
  // 1. Sculpted Roman Travertine & Chiseled Flagstone Slab Material (`wedgeMat`, TSL)
  // ---------------------------------------------------------------------------
  const wedgeMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.68,
    metalness: 0.02,
  });

  const attrBlockPos = attribute('blockPos', 'vec3');
  const attrFaceMeta = attribute('faceMeta', 'vec4');
  const pWorld = positionWorld;
  const nWorld = normalize(normalWorld);
  const viewDir = normalize(cameraPosition.sub(pWorld));

  const isCutFace = attrFaceMeta.x;
  const isInnerFace = attrFaceMeta.w;

  // Geological Travertine & Flagstone Architecture (Zero salt-and-pepper static!):
  // A) Domain-warped horizontal/slightly-tilted sedimentary flagstone benches (`fract` + `smoothstep` terrace steps)
  const macroWarp = mineralToothFbm(
    attrBlockPos.mul(vec3(1.85, 1.85, 2.4)).add(vec3(2.1, -1.3, 0.7)),
    float(1.9)
  );
  const mediumWarp = mineralToothFbm(
    attrBlockPos.mul(vec3(4.6, 4.6, 5.2)).add(vec3(-1.4, 2.7, -0.8)),
    float(2.1)
  );

  // Nearly horizontal geological stone bedding (~6° dip, contrasting strongly with the 45° diagonal cut)
  const bedCoord = attrBlockPos.y
    .mul(5.8)
    .add(attrBlockPos.x.mul(0.75))
    .add(attrBlockPos.z.mul(1.9))
    .add(macroWarp.sub(0.5).mul(1.55))
    .add(mediumWarp.sub(0.5).mul(0.45));

  const bedCellId = floor(bedCoord);
  const bedFract = fract(bedCoord.add(10.0));
  // Crisp sedimentary bedding seam lines where adjacent stone layers meet (`bedFract` near 0 or 1)
  const beddingSeamCrevice = float(1.0)
    .sub(smoothstep(0.0, 0.09, bedFract))
    .add(smoothstep(0.91, 1.0, bedFract));

  // Per-stratum mineral identity hash so each horizontal stone bench has its own distinct travertine/ochre tone!
  const stratumHash = hash31Tsl(vec3(bedCellId, bedCellId.mul(1.37), float(3.1)));
  const fineSubStrata = sin(bedCoord.mul(12.566)).mul(0.5).add(0.5);

  // B) Discrete Vesicular Travertine Cavities / Vugs (Elongated along the stone bedding plane, like real cut travertine!)
  const vugCoord = vec3(
    attrBlockPos.x.mul(7.5).add(macroWarp.mul(2.2)),
    attrBlockPos.y.mul(18.5).add(mediumWarp.mul(1.8)),
    attrBlockPos.z.mul(9.5)
  );
  const vugField1 = valueNoise3DTsl(vugCoord);
  const vugField2 = valueNoise3DTsl(vugCoord.mul(2.15).add(vec3(5.3, -3.1, 1.9)));
  // Discrete concave stone pockets only where both noise fields dip low (creates authentic scattered travertine pores!)
  const travertineVugMask = smoothstep(0.27, 0.13, vugField1.mul(0.68).add(vugField2.mul(0.32)));

  // C) Chiseled Rock Cleavage Facets, Fine Mineral Grain Tooth & Crystalline Calcite Sparkle
  const cleavagePlanes = mineralToothFbm(
    attrBlockPos.mul(vec3(3.2, 3.2, 4.0)).add(vec3(0.8, 1.9, -1.2)),
    float(2.0)
  );
  const fineMineralTooth = mineralToothFbm(
    attrBlockPos.mul(vec3(22.0, 22.0, 20.0)).add(vec3(3.1, -4.2, 1.7)),
    float(2.2)
  );
  const crystalFacetNoise = valueNoise3DTsl(attrBlockPos.mul(vec3(38.0, 38.0, 36.0)));

  // D) True 3D Stone Heightfield Gradient for View-Space Bump Normals (Ledges + Vug Crater Walls + Mineral Tooth)
  const hStoneCenter = cleavagePlanes
    .mul(0.42)
    .add(fineMineralTooth.mul(0.28))
    .add(mediumWarp.mul(0.3))
    .sub(beddingSeamCrevice.mul(0.32))
    .sub(travertineVugMask.mul(0.48));

  const eps = float(0.015);
  const posDx = attrBlockPos.add(vec3(eps, 0.0, 0.0));
  const posDy = attrBlockPos.add(vec3(0.0, eps, 0.0));

  const vugDx = smoothstep(
    0.27,
    0.13,
    valueNoise3DTsl(
      vec3(
        posDx.x.mul(7.5).add(macroWarp.mul(2.2)),
        posDx.y.mul(18.5).add(mediumWarp.mul(1.8)),
        posDx.z.mul(9.5)
      )
    )
      .mul(0.68)
      .add(vugField2.mul(0.32))
  );
  const vugDy = smoothstep(
    0.27,
    0.13,
    valueNoise3DTsl(
      vec3(
        posDy.x.mul(7.5).add(macroWarp.mul(2.2)),
        posDy.y.mul(18.5).add(mediumWarp.mul(1.8)),
        posDy.z.mul(9.5)
      )
    )
      .mul(0.68)
      .add(vugField2.mul(0.32))
  );

  const cleavDx = mineralToothFbm(
    posDx.mul(vec3(3.2, 3.2, 4.0)).add(vec3(0.8, 1.9, -1.2)),
    float(2.0)
  )
    .mul(0.42)
    .add(
      mineralToothFbm(
        posDx.mul(vec3(22.0, 22.0, 20.0)).add(vec3(3.1, -4.2, 1.7)),
        float(2.2)
      ).mul(0.28)
    );
  const cleavDy = mineralToothFbm(
    posDy.mul(vec3(3.2, 3.2, 4.0)).add(vec3(0.8, 1.9, -1.2)),
    float(2.0)
  )
    .mul(0.42)
    .add(
      mineralToothFbm(
        posDy.mul(vec3(22.0, 22.0, 20.0)).add(vec3(3.1, -4.2, 1.7)),
        float(2.2)
      ).mul(0.28)
    );

  const hStoneX = cleavDx
    .add(mediumWarp.mul(0.3))
    .sub(beddingSeamCrevice.mul(0.32))
    .sub(vugDx.mul(0.48));
  const hStoneY = cleavDy
    .add(mediumWarp.mul(0.3))
    .sub(beddingSeamCrevice.mul(0.32))
    .sub(vugDy.mul(0.48));

  const ledgeStepSlopeY = sin(bedCoord.mul(6.28318)).mul(0.09);
  const dHx = clamp(hStoneX.sub(hStoneCenter).mul(1.65), -0.24, 0.24);
  const dHy = clamp(hStoneY.sub(hStoneCenter).mul(1.65).add(ledgeStepSlopeY), -0.24, 0.24);
  const rawGrad = vec3(dHx, dHy, float(0.0));
  const gradPerp = rawGrad.sub(nWorld.mul(dot(rawGrad, nWorld)));

  // Distance to the 45° diagonal cut plane & inner sandwich molten core layer
  const diagDist = abs(attrBlockPos.x.sub(attrBlockPos.y)).mul(INV_SQRT2);
  const seamLipMask = exp(diagDist.negate().mul(16.5));
  const seamThermalHalo = exp(diagDist.negate().mul(11.5));
  const innerZProximity = float(1.0).sub(
    smoothstep(SLICE_INNER_Z - 0.01, SLICE_INNER_Z + 0.055, abs(attrBlockPos.z))
  );

  // Molten lava-cheese glaze coating the cut face, inner sandwich face, and adjacent stone lip
  const coreVisibility = clamp(
    isCutFace
      .mul(0.82)
      .add(isInnerFace.mul(0.88))
      .add(innerZProximity.mul(0.72))
      .add(seamLipMask.mul(0.66)),
    0.0,
    1.0
  );

  const bumpWeight = mix(float(0.75), float(0.08), coreVisibility);
  const perturbedWorldNormal = normalize(nWorld.sub(gradPerp.mul(bumpWeight)));
  wedgeMat.normalNode = transformNormalToView(perturbedWorldNormal);

  // Continuous 3D chiseled outer crust rind (`radialEdgeDist` + `zOuterRim`)
  const radialEdgeDist = max(abs(attrBlockPos.x), abs(attrBlockPos.y)).div(BLOCK_HALF_W);
  const crustRindMask = clamp(
    smoothstep(0.64, 0.96, radialEdgeDist.add(macroWarp.sub(0.5).mul(0.12))).add(
      seamThermalHalo.mul(0.22)
    ),
    0.0,
    1.0
  );

  // Kubelka-Munk Travertine & Toasted Sienna Flagstone Palette:
  // 1) Creamy Bone Calcite + Warm Kaolin travertine benches
  const ivoryTravertine = kubelkaMunkGlaze(
    boneCalcite.K,
    boneCalcite.S,
    clamp(float(0.68).add(cleavagePlanes.mul(0.26)), 0.28, 1.15),
    warmKaolin.rInf
  );

  // 2) Warm Golden Yellow-Ochre & Terracotta Burnt-Sienna stone strata
  const ochreFlagstone = kubelkaMunkGlaze(
    yellowOchre.K,
    yellowOchre.S,
    clamp(float(0.36).add(mediumWarp.mul(0.32)), 0.14, 0.88),
    burntSienna.rInf
  );

  const siennaVeinLayer = kubelkaMunkGlaze(
    burntSienna.K,
    burntSienna.S,
    clamp(float(0.42).add(macroWarp.mul(0.28)), 0.16, 0.86),
    rawUmber.rInf
  );

  // Combine distinct horizontal stone benches by `stratumHash` + fine mineral banding
  const benchBlend = clamp(
    stratumHash.mul(0.62).add(fineSubStrata.mul(0.24)).add(mediumWarp.sub(0.5).mul(0.2)),
    0.05,
    0.92
  );
  const bandedTravertine = mix(
    mix(ivoryTravertine, ochreFlagstone, benchBlend),
    siennaVeinLayer,
    smoothstep(0.68, 0.94, stratumHash).mul(0.48)
  );

  // 3) Dark kiln-fired Raw Umber & Burnt Sienna outer stone crust rind
  const darkStoneCrust = mix(
    burntSienna.rInf,
    rawUmber.rInf,
    clamp(crustRindMask.mul(0.64).add(cleavagePlanes.mul(0.22)), 0.1, 0.86)
  );

  // 4) Shade recessed sedimentary bedding seams & travertine vug cavities with deep Raw Umber + fine mineral grain
  const cavityDepthMask = clamp(
    travertineVugMask.mul(0.76).add(beddingSeamCrevice.mul(0.46)),
    0.0,
    1.0
  );
  const crystalHighlight = smoothstep(0.65, 0.92, crystalFacetNoise).mul(0.08);
  const grainMod = float(1.0).add(fineMineralTooth.sub(0.5).mul(0.14));

  const baseStoneRgb = mix(
    mix(bandedTravertine, darkStoneCrust, crustRindMask.mul(0.76))
      .mul(grainMod)
      .add(boneCalcite.rInf.mul(crystalHighlight)),
    rawUmber.rInf.mul(0.54),
    cavityDepthMask
  );

  // Incandescent lava-amber heat bleeding into the stone crevices & vugs near the 45° cut!
  const creviceLavaBleed = cavityDepthMask.mul(seamThermalHalo).mul(0.85);
  const heatedStoneRgb = mix(
    baseStoneRgb,
    mix(burntSienna.rInf, realgarAmber.rInf, float(0.65)),
    clamp(seamThermalHalo.mul(0.42).add(innerZProximity.mul(0.45)).add(creviceLavaBleed), 0.0, 0.82)
  );

  // Rich molten lava-cheese glaze on the inner sandwich & cut interface
  const coreSwirl = mineralToothFbm(
    attrBlockPos.mul(vec3(5.2, 5.2, 7.0)).add(vec3(0.0, 0.0, stage.uTime.mul(0.14))),
    float(2.2)
  );
  const moltenCoreGlaze = kubelkaMunkGlaze(
    realgarAmber.K,
    realgarAmber.S,
    clamp(float(0.26).add(coreSwirl.mul(0.48)), 0.08, 0.92),
    cinnabarVermilion.rInf
  );
  const creamyCheddarHighlight = mix(
    mix(yellowOchre.rInf, sporeSulfur.rInf, float(0.42)),
    boneCalcite.rInf,
    float(0.18)
  );
  const moltenCoreRgb = mix(
    moltenCoreGlaze,
    creamyCheddarHighlight,
    smoothstep(0.42, 0.84, coreSwirl)
  );

  const toPointer = pWorld.sub(stage.uPointerWorld);
  const pointerGlow = exp(dot(toPointer, toPointer).negate().mul(2.1));

  wedgeMat.colorNode = mix(heatedStoneRgb, moltenCoreRgb, coreVisibility);

  // Honed travertine stone roughness (`0.72` on flat stone benches, `0.94` inside matte vug cavities, `0.07` on wet molten core)
  const stoneRoughness = clamp(
    float(0.72)
      .add(cavityDepthMask.mul(0.22))
      .sub(seamThermalHalo.mul(0.14)),
    0.58,
    0.94
  );
  wedgeMat.roughnessNode = mix(stoneRoughness, float(0.07), coreVisibility);

  // Subsurface thermal glow along the cut edge AND inside the stone crevices near the seam
  const fresnelRim = smoothstep(
    0.15,
    0.92,
    float(1.0).sub(max(0.0, dot(perturbedWorldNormal, viewDir)))
  );
  const seamEmissionStrength = coreVisibility
    .mul(
      float(0.24)
        .add(uCoreHeat.mul(0.26))
        .add(uSqueeze.mul(0.26))
        .add(pointerGlow.mul(0.24))
    )
    .add(seamThermalHalo.mul(float(0.14).add(creviceLavaBleed.mul(0.45))).mul(uCoreHeat));

  wedgeMat.emissiveNode = mix(cinnabarVermilion.rInf, creamyCheddarHighlight, coreSwirl.mul(0.72)).mul(
    seamEmissionStrength.add(fresnelRim.mul(coreVisibility).mul(0.18))
  );

  // ---------------------------------------------------------------------------
  // 2. Gooey Viscoelastic Lava-Cheese / Realgar-Amber Material (`strandMat`, TSL)
  // ---------------------------------------------------------------------------
  const strandMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.07,
    metalness: 0.01,
  });

  const attrStrandMeta = attribute('strandMeta', 'vec4');
  const strandUv = uv();

  const opticalChord = attrStrandMeta.x;
  const stretchStress = attrStrandMeta.y;
  const pigmentMixAttr = attrStrandMeta.z;
  const neckGlowAttr = attrStrandMeta.w;

  // Longitudinal viscoelastic stretch fibers + molten lava-blister crusting (zero polar star harmonics)
  const fiberCoord = vec3(
    strandUv.x.mul(6.0),
    strandUv.y.mul(12.0).sub(stage.uTime.mul(0.26)),
    pigmentMixAttr.mul(8.5)
  );
  const fiberFbm = mineralToothFbm(fiberCoord, float(2.05));

  // High-aspect-ratio (8:1) longitudinal stretched-curd striations (`strandUv.x * 28.0`, `strandUv.y * 3.5`)
  const stretchStriationCoord = vec3(
    strandUv.x.mul(28.0).add(fiberFbm.sub(0.5).mul(2.4)),
    strandUv.y.mul(3.5).sub(stage.uTime.mul(0.18)),
    pigmentMixAttr.mul(14.0)
  );
  const stretchFibers = mineralToothFbm(stretchStriationCoord, float(2.15));
  const stretchFiberCrest = smoothstep(0.46, 0.82, stretchFibers).mul(
    clamp(stretchStress.mul(0.75).add(0.25), 0.25, 1.0)
  );

  // Discrete micro-blister air pockets & toasted `burntSienna` blister rims on wide molten sheets
  const blisterCoord = vec3(
    strandUv.x.mul(16.5).add(fiberFbm.sub(0.5).mul(1.6)),
    strandUv.y.mul(9.5).sub(stage.uTime.mul(0.12)),
    pigmentMixAttr.mul(11.3).add(2.7)
  );
  const blisterField = valueNoise3DTsl(blisterCoord)
    .mul(0.65)
    .add(valueNoise3DTsl(blisterCoord.mul(2.1).add(vec3(4.1, -2.3, 1.7))).mul(0.35));
  // Concave thin-skinned bubble pocket + browned toasted Maillard rim
  const blisterPocketMask = smoothstep(0.25, 0.12, blisterField);
  const toastedBlisterRim = smoothstep(0.35, 0.21, blisterField).mul(
    float(1.0).sub(blisterPocketMask.mul(0.85))
  );

  // Subtle view-space normal relief for longitudinal curd fibers & micro-blister dimples
  const fiberSlopeCross = sin(strandUv.x.mul(175.0).add(stretchFibers.mul(6.283))).mul(0.085);
  const blisterSlope = blisterPocketMask.sub(toastedBlisterRim.mul(0.65)).mul(0.11);
  const strandBumpOffset = vec3(
    fiberSlopeCross.add(blisterSlope),
    blisterSlope.mul(0.6),
    float(0.0)
  );
  const strandBumpPerp = strandBumpOffset.sub(nWorld.mul(dot(strandBumpOffset, nWorld)));
  const strandNormalWorld = normalize(nWorld.add(strandBumpPerp));
  strandMat.normalNode = transformNormalToView(strandNormalWorld);

  // Beer-Lambert optical chord transmittance: thick lava-cheese lobes have deep Cinnabar/Realgar cores;
  // stretched thin webs & droplets glow warm golden Ochre/Sulfur
  const extinctionVec = mix(realgarAmber.extinction, cinnabarVermilion.extinction, pigmentMixAttr);
  const transRgb = beerLambertTransmittance(
    extinctionVec,
    opticalChord.mul(float(1.05).sub(neckGlowAttr.mul(0.52)).sub(blisterPocketMask.mul(0.35)))
  );

  const thickLavaCheeseColor = kubelkaMunkGlaze(
    realgarAmber.K,
    realgarAmber.S,
    clamp(float(0.32).add(fiberFbm.mul(0.34)).add(stretchFibers.mul(0.18)).sub(stretchStress.mul(0.14)), 0.09, 0.88),
    mix(cinnabarVermilion.rInf, burntSienna.rInf, float(0.3))
  );

  const goldenStretchColor = mix(
    mix(realgarAmber.rInf, yellowOchre.rInf, float(0.62)),
    mix(sporeSulfur.rInf, boneCalcite.rInf, float(0.28)),
    clamp(stretchStress.mul(0.42).add(stretchFiberCrest.mul(0.24)).add(0.12), 0.12, 0.72)
  );

  const baseStrandRgb = mix(
    thickLavaCheeseColor,
    goldenStretchColor.mul(transRgb.add(0.36)),
    clamp(neckGlowAttr.mul(0.72).add(fiberFbm.mul(0.16)).add(stretchFiberCrest.mul(0.22)), 0.0, 0.94)
  );

  // Layer toasted Burnt Sienna blister rims & creamy Bone Calcite bubble-skin highlights
  const toastedRimColor = mix(burntSienna.rInf, rawUmber.rInf, float(0.28));
  const bubbleSkinColor = mix(goldenStretchColor, boneCalcite.rInf, float(0.38));
  const strandSurfaceRgb = mix(
    mix(baseStrandRgb, toastedRimColor, toastedBlisterRim.mul(0.46)),
    bubbleSkinColor,
    blisterPocketMask.mul(0.38).add(stretchFiberCrest.mul(0.18))
  );

  const strandViewDot = max(0.0, dot(strandNormalWorld, viewDir));
  const strandFresnel = smoothstep(0.08, 0.94, float(1.0).sub(strandViewDot));

  strandMat.colorNode = mix(strandSurfaceRgb, boneCalcite.rInf, strandFresnel.mul(0.28));
  strandMat.roughnessNode = clamp(
    float(0.055)
      .add(fiberFbm.mul(0.045))
      .add(stretchFiberCrest.mul(0.035))
      .add(toastedBlisterRim.mul(0.12))
      .sub(blisterPocketMask.mul(0.025)),
    0.035,
    0.22
  );

  const strandEmissiveRgb = mix(
    cinnabarVermilion.rInf,
    goldenStretchColor,
    clamp(neckGlowAttr.mul(0.85).add(0.15), 0.0, 1.0)
  );
  strandMat.emissiveNode = strandEmissiveRgb.mul(
    float(0.18)
      .add(neckGlowAttr.mul(0.36))
      .add(uCoreHeat.mul(0.2))
      .add(strandFresnel.mul(0.28))
      .add(pointerGlow.mul(0.34))
  );

  // ---------------------------------------------------------------------------
  // 3. Build the Two Sandwich Halves: Each Has Front Stone Slice + Back Stone Slice + Molten Cushion
  // ---------------------------------------------------------------------------
  const wedgeTopLeftGroup = new THREE.Group();
  wedgeTopLeftGroup.name = 'SandwichWedgeTopLeft';
  const wedgeBottomRightGroup = new THREE.Group();
  wedgeBottomRightGroup.name = 'SandwichWedgeBottomRight';

  const sliceTopLeftFront = new THREE.Mesh(buildSandstoneSliceGeometry(-1, 1), wedgeMat);
  const sliceTopLeftBack = new THREE.Mesh(buildSandstoneSliceGeometry(-1, -1), wedgeMat);
  const cushionTopLeft = new THREE.Mesh(buildMoltenSandwichCushionGeometry(-1), strandMat);
  sliceTopLeftFront.castShadow = true;
  sliceTopLeftFront.receiveShadow = true;
  sliceTopLeftBack.castShadow = true;
  sliceTopLeftBack.receiveShadow = true;
  wedgeTopLeftGroup.add(sliceTopLeftFront, sliceTopLeftBack, cushionTopLeft);

  const sliceBottomRightFront = new THREE.Mesh(buildSandstoneSliceGeometry(1, 1), wedgeMat);
  const sliceBottomRightBack = new THREE.Mesh(buildSandstoneSliceGeometry(1, -1), wedgeMat);
  const cushionBottomRight = new THREE.Mesh(buildMoltenSandwichCushionGeometry(1), strandMat);
  sliceBottomRightFront.castShadow = true;
  sliceBottomRightFront.receiveShadow = true;
  sliceBottomRightBack.castShadow = true;
  sliceBottomRightBack.receiveShadow = true;
  wedgeBottomRightGroup.add(sliceBottomRightFront, sliceBottomRightBack, cushionBottomRight);

  specimenRoot.add(wedgeTopLeftGroup);
  specimenRoot.add(wedgeBottomRightGroup);

  // ---------------------------------------------------------------------------
  // 4. Slice-Locked Molten Cut Bolsters & Lip Ooze Tubes (ZERO Floating Origin Tubes!)
  // ---------------------------------------------------------------------------
  const oozeBeads: OozeBeadTube[] = [];

  for (const wedgeSide of [-1, 1] as const) {
    for (const sideZ of [-1, 0, 1]) {
      const { geometry, posAttr, normAttr, metaAttr } = createDynamicTubeGeometry(
        BEAD_LENGTH_SEGMENTS,
        BEAD_RADIAL_SEGMENTS
      );
      const mesh = new THREE.Mesh(geometry, strandMat);
      mesh.frustumCulled = false;
      if (wedgeSide === -1) {
        wedgeTopLeftGroup.add(mesh);
      } else {
        wedgeBottomRightGroup.add(mesh);
      }
      oozeBeads.push({
        mesh,
        posAttr,
        normAttr,
        metaAttr,
        sideZ,
        wedgeSide,
      });
    }
  }

  // ---------------------------------------------------------------------------
  // 5. Create Multi-Archetype Lava-Cheese Strings & Sheets (High Morphological Variability!)
  // ---------------------------------------------------------------------------
  const strandsGroup = new THREE.Group();
  strandsGroup.name = 'ViscoelasticMoltenStrands';
  specimenRoot.add(strandsGroup);

  // Hand-crafted + irregular clustered distribution across 4 distinct viscoelastic archetypes:
  // 1) 5 Massive `mother-lobe` lava sheets (clustered around lower-left, center, and upper-mid)
  // 2) 12 `twisted-rope` braided viscoelastic cords that spiral around each other
  // 3) 18 `gossamer-thread` fine high-tension capillary strings with pearling lava droplets
  // 4) 7 `gravity-drip` heavy sagging stalactite lobes
  const strands: MoltenStrand[] = [];

  // Asymmetric clustered anchor centers along seamS spanning the full diagonal [-0.76, +0.76]
  const clusterCenters = [-0.64, -0.38, -0.09, 0.18, 0.44, 0.66];
  const motherLobeRadii = [0.03, 0.048, 0.043, 0.038, 0.034, 0.026];
  const totalStrands = 46;

  for (let i = 0; i < totalStrands; i++) {
    const hash1 = Math.sin(i * 19.31 + 7.13) * 43758.5453;
    const f1 = hash1 - Math.floor(hash1);
    const hash2 = Math.sin(i * 41.77 + 13.91) * 19341.17;
    const f2 = hash2 - Math.floor(hash2);
    const hash3 = Math.sin(i * 73.19 + 5.47) * 29183.61;
    const f3 = hash3 - Math.floor(hash3);
    const hash4 = Math.sin(i * 97.43 + 2.89) * 15731.43;
    const f4 = hash4 - Math.floor(hash4);

    let archetype: StrandArchetype;
    let seamS: number;
    let baseRadius: number;
    let ribbonWidthScale: number;
    let neckSharpness: number;
    let twistAmp = 0.0;
    let twistTurns = 1.0;
    let globuleAmp = 0.18;
    let globuleSpeed = 1.2;
    let criticalGap: number;
    let customSkew = 0.0;

    if (i < 6) {
      // Archetype 1: 6 Asymmetric Gooey Mother Lobes / Lava Sheets across the diagonal
      archetype = 'mother-lobe';
      seamS = clusterCenters[i] + (f1 - 0.5) * 0.06;
      baseRadius = motherLobeRadii[i] * (0.92 + f2 * 0.22);
      ribbonWidthScale = 1.7 + f3 * 0.45;
      neckSharpness = 1.65 + f4 * 0.85;
      globuleAmp = 0.34 + f1 * 0.26;
      globuleSpeed = 0.85 + f2 * 0.5;
      criticalGap = 0.98 + f3 * 0.28;
      customSkew = (f2 - 0.5) * 0.06;
    } else if (i < 20) {
      // Archetype 2: 14 Twisted / Diagonal Webbing Ropes that bridge BETWEEN adjacent clusters!
      archetype = 'twisted-rope';
      const idxInRole = i - 6;
      const uSpan = (idxInRole + 0.3 + (f1 - 0.5) * 0.45) / 14.0;
      seamS = -0.7 + uSpan * 1.4;
      baseRadius = 0.016 + f2 * 0.017;
      ribbonWidthScale = 1.25 + f3 * 0.38;
      neckSharpness = 2.3 + f4 * 1.4;
      twistAmp = 0.018 + f1 * 0.024;
      twistTurns = (i % 2 === 0 ? 1 : -1) * (1.1 + f2 * 1.2);
      globuleAmp = 0.42 + f3 * 0.35;
      globuleSpeed = 1.2 + f4 * 0.8;
      criticalGap = 0.86 + f2 * 0.34;
      // Strong diagonal web obliquity so ropes cross between mother lobes in X and Y junctions
      customSkew = (idxInRole % 2 === 0 ? 1 : -1) * (0.075 + f3 * 0.085);
    } else if (i < 38) {
      // Archetype 3: 18 Gossamer Hair-Thin Capillary Filaments with Pearling Lava Droplets across [-0.75, +0.75]
      archetype = 'gossamer-thread';
      const idxInRole = i - 20;
      const uSpan = (idxInRole + 0.25 + (f1 - 0.5) * 0.55) / 18.0;
      seamS = -0.75 + uSpan * 1.5;
      baseRadius = 0.007 + f2 * 0.009;
      ribbonWidthScale = 1.0 + f3 * 0.18;
      neckSharpness = 3.4 + f4 * 2.1;
      twistAmp = 0.009 * f1;
      twistTurns = f2 > 0.5 ? 1.0 : -1.0;
      globuleAmp = 0.78 + f3 * 0.68;
      globuleSpeed = 1.45 + f4 * 1.1;
      criticalGap = 0.78 + f1 * 0.38;
      customSkew = (f4 > 0.5 ? 1 : -1) * (0.09 + f2 * 0.11);
    } else {
      // Archetype 4: 8 Heavy Gravity-Drip Stalactite Lobes draping across [-0.66, +0.58]
      archetype = 'gravity-drip';
      const idxInRole = i - 38;
      seamS = -0.66 + ((idxInRole + 0.35 + (f1 - 0.5) * 0.4) / 8.0) * 1.24;
      baseRadius = 0.024 + f2 * 0.018;
      ribbonWidthScale = 1.4 + f3 * 0.42;
      neckSharpness = 2.1 + f4 * 1.1;
      globuleAmp = 0.55 + f1 * 0.4;
      globuleSpeed = 0.95 + f2 * 0.6;
      criticalGap = 0.88 + f3 * 0.32;
      customSkew = (f3 - 0.5) * 0.1;
    }

    seamS = clampScalar(seamS, -0.76, 0.76);
    const cornerTaper = clampScalar(1.0 - Math.pow(Math.abs(seamS) / 0.84, 2.1), 0.18, 1.0);
    baseRadius *= 0.5 + 0.5 * cornerTaper;

    // Multi-tier Z depth across the molten core channel (|z| <= 0.25)
    const tierZ = ((i % 7) - 3) * 0.076 + (f2 - 0.5) * 0.035;
    const depthLayer01 = clampScalar((tierZ + 0.26) / 0.52, 0.0, 1.0);

    const baseX = DIAG_TANGENT_X * seamS;
    const baseY = DIAG_TANGENT_Y * seamS;

    const skewS = customSkew * cornerTaper;
    const anchorDepthInset = 0.042 * cornerTaper;

    const anchorA = new THREE.Vector3(
      baseX - PULL_NX * anchorDepthInset + DIAG_TANGENT_X * skewS,
      baseY - PULL_NY * anchorDepthInset + DIAG_TANGENT_Y * skewS,
      tierZ
    );
    const anchorB = new THREE.Vector3(
      baseX + PULL_NX * anchorDepthInset - DIAG_TANGENT_X * skewS,
      baseY + PULL_NY * anchorDepthInset - DIAG_TANGENT_Y * skewS,
      tierZ + (f4 - 0.5) * 0.045
    );

    const neckCenterT = 0.24 + f4 * 0.52;
    const retractSide: -1 | 1 = i % 2 === 0 ? -1 : 1;

    const { geometry, posAttr, normAttr, metaAttr } = createDynamicTubeGeometry(
      STRAND_LENGTH_SEGMENTS,
      STRAND_RADIAL_SEGMENTS
    );
    const mesh = new THREE.Mesh(geometry, strandMat);
    mesh.frustumCulled = false;
    strandsGroup.add(mesh);

    strands.push({
      index: i,
      archetype,
      mesh,
      posAttr,
      normAttr,
      metaAttr,
      seamS,
      cornerTaper,
      coreZ: tierZ,
      depthLayer01,
      neckCenterT,
      neckSharpness,
      twistAmp,
      twistTurns,
      globuleAmp,
      globuleSpeed,
      retractSide,
      anchorA,
      anchorB,
      baseRadius,
      ribbonWidthScale,
      criticalGap,
      snapProgress: 0,
      isSnapped: false,
      stirDispX: 0,
      stirDispY: 0,
      stirVelX: 0,
      stirVelY: 0,
      pigmentMix: clampScalar(
        0.18 + 0.62 * f1 + (1.0 - depthLayer01) * 0.22,
        0.06,
        0.95
      ),
      phaseOffset: i * 1.17 + f2 * 3.14,
    });
  }

  // ---------------------------------------------------------------------------
  // 6. Tactile Pointer State Machine: Drag-to-Squeeze & Pull + Fondue Stir + Click-to-Snap
  // ---------------------------------------------------------------------------
  let gapState = 0.0;
  let targetGapState = 0.0;
  let gapVelocity = 0.0;

  let shearState = 0.0;
  let targetShearState = 0.0;

  let wasPointerDown = false;
  let isDraggingGesture = false;
  let dragMovedDistance = 0;
  let downProjNormal = 0;
  let downProjTangent = 0;
  let downStartGap = 0;
  let downStartShear = 0;
  let dragAnchorSign = 1;

  const _wedgeOffsetA = new THREE.Vector3();
  const _wedgeOffsetB = new THREE.Vector3();
  const _worldAnchorA = new THREE.Vector3();
  const _worldAnchorB = new THREE.Vector3();
  const _strandMid = new THREE.Vector3();

  // ---------------------------------------------------------------------------
  // 7. Per-Frame Viscoelastic Simulation & Ooze / Strand Coupling
  // ---------------------------------------------------------------------------
  stage.onTick(({ elapsedSec, dtSec, pointer }) => {
    const px = pointer.worldPos.x;
    const py = pointer.worldPos.y;
    const projN = px * PULL_NX + py * PULL_NY;
    const projT = px * DIAG_TANGENT_X + py * DIAG_TANGENT_Y;

    if (pointer.isDown && !wasPointerDown) {
      wasPointerDown = true;
      isDraggingGesture = false;
      dragMovedDistance = 0;
      downProjNormal = projN;
      downProjTangent = projT;
      downStartGap = targetGapState;
      downStartShear = targetShearState;

      if (downStartGap < 0.08) {
        dragAnchorSign = projN < -0.58 ? -1 : 1;
      } else {
        dragAnchorSign = projN >= 0 ? 1 : -1;
      }
    } else if (pointer.isDown && wasPointerDown) {
      const rawDeltaN = projN - downProjNormal;
      const deltaT = (projT - downProjTangent) * dragAnchorSign;
      dragMovedDistance = Math.hypot(rawDeltaN, projT - downProjTangent);

      if (dragMovedDistance > 0.035) {
        isDraggingGesture = true;
      }

      if (isDraggingGesture) {
        let signedDeltaN = rawDeltaN * dragAnchorSign;
        if (downStartGap < 0.08 && Math.abs(downProjNormal) <= 0.58) {
          signedDeltaN = rawDeltaN >= -0.02 ? rawDeltaN * 1.15 : rawDeltaN;
        }
        targetGapState = clampScalar(downStartGap + signedDeltaN * 1.15, -0.15, 1.08);
        gapState = targetGapState;
        gapVelocity = 0.0;
        targetShearState = clampScalar(downStartShear + deltaT * 0.24, -0.22, 0.22);
      }
    } else if (!pointer.isDown && wasPointerDown) {
      wasPointerDown = false;

      if (!isDraggingGesture) {
        if (gapState > 0.14) {
          // Crisp click on a taut strand snaps that specific strand (and plucks its neighbors)
          let closestIdx = -1;
          let closestDist = 0.24;

          for (let i = 0; i < strands.length; i++) {
            const st = strands[i];
            if (st.isSnapped) continue;
            _worldAnchorA
              .copy(st.anchorA)
              .applyEuler(wedgeTopLeftGroup.rotation)
              .add(wedgeTopLeftGroup.position);
            _worldAnchorB
              .copy(st.anchorB)
              .applyEuler(wedgeBottomRightGroup.rotation)
              .add(wedgeBottomRightGroup.position);
            _strandMid.addVectors(_worldAnchorA, _worldAnchorB).multiplyScalar(0.5);
            const d = Math.hypot(px - _strandMid.x, py - _strandMid.y);
            if (d < closestDist) {
              closestDist = d;
              closestIdx = i;
            }
          }

          if (closestIdx !== -1) {
            const st = strands[closestIdx];
            st.isSnapped = true;
            for (let i = 0; i < strands.length; i++) {
              if (i === closestIdx || strands[i].isSnapped) continue;
              const dSeam = Math.abs(strands[i].seamS - st.seamS);
              if (dSeam < 0.25) {
                const kick = (i % 2 === 0 ? 1 : -1) * (0.25 - dSeam) * 2.2;
                strands[i].stirVelX += DIAG_TANGENT_X * kick;
                strands[i].stirVelY += DIAG_TANGENT_Y * kick;
              }
            }
            uCoreHeat.value = Math.min(1.38, uCoreHeat.value + 0.25);
          } else if (py < 0.72) {
            targetGapState = -0.1;
            targetShearState = 0.0;
          }
        } else {
          gapVelocity = -0.6;
          targetGapState = 0.58;
        }
      } else {
        if (targetGapState < 0.02) {
          targetGapState = 0.0;
          targetShearState = 0.0;
        }
      }
      isDraggingGesture = false;
    }

    const stiffness = pointer.isDown && isDraggingGesture ? 75.0 : 14.0;
    const damping = pointer.isDown && isDraggingGesture ? 0.65 : 0.86;

    const forceGap = (targetGapState - gapState) * stiffness;
    gapVelocity = (gapVelocity + forceGap * dtSec) * Math.pow(damping, dtSec * 60);
    gapState = clampScalar(gapState + gapVelocity * dtSec, -0.15, 1.12);

    shearState += (targetShearState - shearState) * Math.min(1, dtSec * 12.0);

    const idleBreath =
      !pointer.isDown && Math.abs(gapState) < 0.05
        ? Math.sin(elapsedSec * 2.1) * 0.008
        : 0.0;

    const openDist = Math.max(0.0, gapState + idleBreath);
    const squeezeCompress = clampScalar(
      (-gapState / 0.15) * 0.82 + Math.max(0, 0.3 - openDist * 1.15),
      0.0,
      1.0
    );

    uGapOpen.value = openDist;
    uSqueeze.value = squeezeCompress;

    const targetHeat = clampScalar(
      0.66 +
        squeezeCompress * 0.4 +
        Math.min(0.32, openDist * 0.4) +
        Math.min(0.32, pointer.dwellTimeSec * 0.16) +
        Math.min(0.22, pointer.speed * 0.08),
      0.58,
      1.32
    );
    uCoreHeat.value += (targetHeat - uCoreHeat.value) * Math.min(1, dtSec * 5.0);

    // Position & articulate the two 45°-cut triangular sandwich halves
    const physicalHalfGap = Math.max(-0.003, gapState * 0.5 + idleBreath);
    const halfShear = shearState * 0.5;

    _wedgeOffsetA.set(
      -PULL_NX * physicalHalfGap - DIAG_TANGENT_X * halfShear,
      -PULL_NY * physicalHalfGap - DIAG_TANGENT_Y * halfShear,
      0
    );
    _wedgeOffsetB.set(
      PULL_NX * physicalHalfGap + DIAG_TANGENT_X * halfShear,
      PULL_NY * physicalHalfGap + DIAG_TANGENT_Y * halfShear,
      0
    );

    wedgeTopLeftGroup.position.copy(_wedgeOffsetA);
    wedgeBottomRightGroup.position.copy(_wedgeOffsetB);

    const pullTilt = openDist * 0.085;
    wedgeTopLeftGroup.rotation.set(-pullTilt * 0.32, -pullTilt * 0.24, pullTilt * 0.52);
    wedgeBottomRightGroup.rotation.set(pullTilt * 0.32, pullTilt * 0.24, -pullTilt * 0.52);

    // Update all slice-adhered molten bolsters
    for (let b = 0; b < oozeBeads.length; b++) {
      updateOozeBeadTube(oozeBeads[b], openDist, squeezeCompress, elapsedSec);
    }

    // Update all 42 viscoelastic lava-cheese elements with interactive pointer fondue-stirring!
    const ptrWorld = pointer.worldPos;
    const ptrVel = pointer.velocity;
    const ptrSpeed = pointer.speed;

    for (let i = 0; i < strands.length; i++) {
      const st = strands[i];

      if (openDist < 0.075 && st.isSnapped) {
        st.isSnapped = false;
      }

      if (openDist > st.criticalGap && !st.isSnapped) {
        st.isSnapped = true;
      }

      if (st.isSnapped) {
        st.snapProgress = Math.min(1.0, st.snapProgress + dtSec * 2.6);
      } else {
        st.snapProgress = Math.max(0.0, st.snapProgress - dtSec * 4.5);
      }

      // Interactive Fondue / Lava-Cheese Pointer Coupling:
      // Touching or moving the pointer across the stretched cheese pulls and bows the nearby viscoelastic strings
      // smoothly along with the hand (NEVER auto-cutting them into uniform teeth on hover!)
      if (openDist > 0.05 && !st.isSnapped) {
        _worldAnchorA
          .copy(st.anchorA)
          .applyEuler(wedgeTopLeftGroup.rotation)
          .add(wedgeTopLeftGroup.position);
        _worldAnchorB
          .copy(st.anchorB)
          .applyEuler(wedgeBottomRightGroup.rotation)
          .add(wedgeBottomRightGroup.position);
        const midX = (_worldAnchorA.x + _worldAnchorB.x) * 0.5 + st.stirDispX * 0.5;
        const midY = (_worldAnchorA.y + _worldAnchorB.y) * 0.5 + st.stirDispY * 0.5;
        const distToPtr = Math.hypot(ptrWorld.x - midX, ptrWorld.y - midY);

        const influenceRadius = st.archetype === 'mother-lobe' ? 0.34 : 0.26;
        if (distToPtr < influenceRadius) {
          const falloff = 1.0 - distToPtr / influenceRadius;
          // Viscoelastic drag follows pointer velocity + soft capillary attraction toward warm pointer tip
          const compliance =
            st.archetype === 'gossamer-thread'
              ? 1.35
              : st.archetype === 'mother-lobe'
                ? 0.65
                : 1.0;
          if (ptrSpeed > 0.03) {
            st.stirVelX += ptrVel.x * falloff * 0.28 * compliance;
            st.stirVelY += ptrVel.y * falloff * 0.28 * compliance;
          }
          // Gentle capillary adhesion toward pointer when hovering/dwelling on a cheese lobe
          st.stirVelX += (ptrWorld.x - midX) * falloff * 6.5 * dtSec * compliance;
          st.stirVelY += (ptrWorld.y - midY) * falloff * 6.5 * dtSec * compliance;
        }
      }

      const springK = st.archetype === 'mother-lobe' ? 22.0 : 32.0;
      const damp = Math.pow(0.82, dtSec * 60);
      st.stirVelX = (st.stirVelX - st.stirDispX * springK * dtSec) * damp;
      st.stirVelY = (st.stirVelY - st.stirDispY * springK * dtSec) * damp;
      st.stirDispX = clampScalar(st.stirDispX + st.stirVelX * dtSec, -0.18, 0.18);
      st.stirDispY = clampScalar(st.stirDispY + st.stirVelY * dtSec, -0.18, 0.18);

      updateMoltenStrandGeometry(
        st,
        wedgeTopLeftGroup,
        wedgeBottomRightGroup,
        openDist,
        squeezeCompress,
        elapsedSec
      );
    }

    seamCoreLight.intensity = 0.25 + openDist * 2.2 + squeezeCompress * 0.35;
    seamBackLight.intensity = 0.2 + openDist * 1.4;

    const floatY = Math.sin(elapsedSec * 1.15) * 0.018;
    specimenRoot.position.y = floatY;
    specimenRoot.rotation.y = -0.34 + pointer.worldPos.x * 0.05;
    specimenRoot.rotation.x = 0.18 - pointer.worldPos.y * 0.04;
  });

  return stage;
}

if (typeof document !== 'undefined') {
  void bootDay03Melt();
}
