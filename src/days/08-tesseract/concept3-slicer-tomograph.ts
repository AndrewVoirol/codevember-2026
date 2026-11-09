import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  floor,
  length,
  mix,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  sin,
  smoothstep,
  uniform,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

/**
 * Day 08 — Tesseract (Concept 8C — Pure 4D Hyperplane Crystallographic Slicer):
 * "4D Hyperplane Crystallographic Tomograph (Live w = w₀ Polyhedral Slicer)"
 *
 * Key Architectural & Optical Invariants:
 * 1. Exact Projective Strut-to-Crystal Registration (`0.000` Discrepancy):
 *    - Projects the 16 4D Tesseract vertices into R³ using central perspective relative to
 *      the active slicing hyperplane `w = w0`:
 *        `g(w) = SCALE_3D * (W_PROJ_DIST / (W_PROJ_DIST - (w - w0) * W_PERSP_K))`
 *    - By projective colinearity, `(1 - t)/g(w_a) + t/g(w_b) === 1` at `w(t) = w0`, so every
 *      3D crystal corner lies with mathematical exactness on the 3D tubular axis of its parent
 *      4D Tesseract strut. Sculpted Cinnabar/Realgar piercing caliper nodes ride directly along
 *      the 32 3D tubular struts at the live crystal vertices.
 * 2. True Euclidean Constant-Width Fortification Agate & Straight-Skeleton Sutures:
 *    - Every hyperplane section of a 3D cubic cell is a convex polygon with `3 <= m <= 6` edges.
 *    - Precomputes exact inward perpendicular Euclidean distances `(d0..d5)` to all `m` edges
 *      at each polygon vertex (`aEdgeDist012`, `aEdgeDist345`). Because `d_k(x)` is affine linear
 *      in R³, hardware barycentric interpolation yields exact physical millimeter edge distances
 *      at every pixel—eliminating radial centroid distortion and center noise patches, and
 *      forming authentic straight-skeleton medial sutures (`dSecond - dMin == 0`).
 * 3. Per-Cell Mineral Identity Across the 4 Dual Hyper-Axis Cell Pairs (`±X, ±Y, ±Z, ±W`):
 *    - `±X` Cells (0, 1): Lapis Ultramarine & Azurite Prussian Fortification Agate
 *    - `±Y` Cells (2, 3): Cinnabar Vermilion & Realgar Amber Carnelian Sardonyx
 *    - `±Z` Cells (4, 5): Verdigris Malachite & Celadon Glauconite Banded Jasper
 *    - `±W` Cells (6, 7): Yellow Ochre, Raw Umber & Burnt Sienna Tiger-Agate
 */

// ---------------------------------------------------------------------------
// 4D Tesseract Cell & Edge Topology Constants (Zero Per-Frame GC)
// ---------------------------------------------------------------------------
const NUM_VERTICES = 16;
const NUM_EDGES = 32;
const NUM_CELLS = 8;
const NUM_LAMINA_SLICES = 6;
const LAMINA_OFFSETS: readonly number[] = [
  -0.58, -0.36, -0.18,
  0.18, 0.36, 0.58,
];

const SCALE_3D = 0.68;
const W_PROJ_DIST = 2.55;
const W_PERSP_K = 0.38;

const _baseVerts4D = new Float32Array(NUM_VERTICES * 4);
for (let i = 0; i < NUM_VERTICES; i++) {
  _baseVerts4D[i * 4 + 0] = i & 1 ? 1 : -1;
  _baseVerts4D[i * 4 + 1] = i & 2 ? 1 : -1;
  _baseVerts4D[i * 4 + 2] = i & 4 ? 1 : -1;
  _baseVerts4D[i * 4 + 3] = i & 8 ? 1 : -1;
}

const _edgePairs = new Uint8Array(NUM_EDGES * 2);
{
  let eIdx = 0;
  for (let i = 0; i < NUM_VERTICES; i++) {
    for (let bit = 0; bit < 4; bit++) {
      const j = i ^ (1 << bit);
      if (i < j) {
        _edgePairs[eIdx * 2 + 0] = i;
        _edgePairs[eIdx * 2 + 1] = j;
        eIdx++;
      }
    }
  }
}

// Each of the 8 cubic cells `c in [0..7]` fixes axis `d = c >> 1` to sign `s = (c & 1) ? +1 : -1`
// and contains 12 of the 32 edges.
const _cellEdges = new Uint8Array(NUM_CELLS * 12);
const _baseCellNormals4D = new Float32Array(NUM_CELLS * 4);
{
  for (let c = 0; c < NUM_CELLS; c++) {
    const axis = c >> 1;
    const bitVal = c & 1;
    _baseCellNormals4D[c * 4 + axis] = bitVal ? 1 : -1;

    let count = 0;
    for (let e = 0; e < NUM_EDGES; e++) {
      const v0 = _edgePairs[e * 2 + 0];
      const v1 = _edgePairs[e * 2 + 1];
      if (((v0 >> axis) & 1) === bitVal && ((v1 >> axis) & 1) === bitVal) {
        _cellEdges[c * 12 + count] = e;
        count++;
      }
    }
  }
}

// Pre-allocated 4D rotation & vertex buffers
const _rotMat4D = new Float32Array(16);
const _tmpMat4D = new Float32Array(16);
const _rotVerts4D = new Float32Array(NUM_VERTICES * 4);
const _rotCellNormals4D = new Float32Array(NUM_CELLS * 4);

// Pre-allocated Edge Intersection Tables for a single hyperplane slice `w = wSlice`
const _edgeHitValid = new Uint8Array(NUM_EDGES);
const _edgeHitPoints3D = new Float32Array(NUM_EDGES * 3);
const _primaryEdgeHitValid = new Uint8Array(NUM_EDGES);
const _primaryEdgeHitPoints3D = new Float32Array(NUM_EDGES * 3);

// Scratch polygon buffer for clipping a single cell (up to 6 unique vertices for a cube slice)
const _polyPts = new Float32Array(8 * 3);
const _polyAngles = new Float32Array(8);
const _polyOrder = new Uint8Array(8);
const _orderedPolyPts = new Float32Array(8 * 3);
const _edgeInwardNormals = new Float32Array(8 * 3);
const _vertEdgeDists = new Float32Array(6);

// Pre-allocated Dynamic Mesh Buffers for the Active 3D Crystal Polyhedron
const MAX_CRYSTAL_VERTS = 240;
const _crystalPositions = new Float32Array(MAX_CRYSTAL_VERTS * 3);
const _crystalNormals = new Float32Array(MAX_CRYSTAL_VERTS * 3);
const _crystalMeta = new Float32Array(MAX_CRYSTAL_VERTS * 3); // x: cellId [0..7], y: inradius, z: edgeCount
const _crystalEdgeDist012 = new Float32Array(MAX_CRYSTAL_VERTS * 3); // exact Euclidean dists to edges 0, 1, 2
const _crystalEdgeDist345 = new Float32Array(MAX_CRYSTAL_VERTS * 3); // exact Euclidean dists to edges 3, 4, 5

// Pre-allocated Tubular/Box Bevel Struts around every polygonal face edge of the active slice
const MAX_BEVEL_EDGES = 48;
const VERTS_PER_BEVEL = 24;
const INDICES_PER_BEVEL = 36;
const _bevelPositions = new Float32Array(MAX_BEVEL_EDGES * VERTS_PER_BEVEL * 3);
const _bevelNormals = new Float32Array(MAX_BEVEL_EDGES * VERTS_PER_BEVEL * 3);

// Pre-allocated 3D Tubular Struts for the 32 4D Tesseract Armature Edges
const _ghostStrutPositions = new Float32Array(NUM_EDGES * VERTS_PER_BEVEL * 3);
const _ghostStrutNormals = new Float32Array(NUM_EDGES * VERTS_PER_BEVEL * 3);
const _ghostStrutMeta = new Float32Array(NUM_EDGES * VERTS_PER_BEVEL * 3); // x: rw - w0, y: isPierced, z: axisId

// Pre-allocated Contour Lines for the Neighboring Tomographic Slices
const MAX_CONTOUR_SEGMENTS = NUM_LAMINA_SLICES * 48 + 48;
const _contourLinePositions = new Float32Array(MAX_CONTOUR_SEGMENTS * 2 * 3);

// Scratch vectors for polygon sorting & bevel strut construction
const _polyCenter = new THREE.Vector3();
const _faceCentroid = new THREE.Vector3();
const _faceNormal = new THREE.Vector3();
const _tangent1 = new THREE.Vector3();
const _tangent2 = new THREE.Vector3();
const _diffVec = new THREE.Vector3();
const _edgeDir = new THREE.Vector3();
const _edgeSide = new THREE.Vector3();
const _edgeUp = new THREE.Vector3();
const _inwardN = new THREE.Vector3();

// ---------------------------------------------------------------------------
// 4D Rotation Helpers
// ---------------------------------------------------------------------------
function setIdentity4D(m: Float32Array): void {
  m.fill(0);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
}

function applyGivens4D(m: Float32Array, a: number, b: number, angle: number): void {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  _tmpMat4D.set(m);
  for (let r = 0; r < 4; r++) {
    const va = _tmpMat4D[r * 4 + a];
    const vb = _tmpMat4D[r * 4 + b];
    m[r * 4 + a] = c * va - s * vb;
    m[r * 4 + b] = s * va + c * vb;
  }
}

function projectScaleAtW(rw: number, w0: number): number {
  const denom = Math.max(0.95, W_PROJ_DIST - (rw - w0) * W_PERSP_K);
  return SCALE_3D * (W_PROJ_DIST / denom);
}

// ---------------------------------------------------------------------------
// Canonical 4D Crystallographic Symmetry Presets
// ---------------------------------------------------------------------------
interface SymmetryPreset {
  name: string;
  xw: number;
  yw: number;
  zw: number;
  xy: number;
}

const SYMMETRY_PRESETS: readonly SymmetryPreset[] = [
  // 0: Vertex-First [1, 1, 1, 1] Long Diagonal (Tetrahedron <-> Truncated Tetrahedron <-> Octahedron)
  {
    name: 'vertex-first-1111',
    xw: Math.PI * 0.25,
    yw: Math.atan(1 / Math.SQRT2),
    zw: Math.atan(1 / Math.sqrt(3)),
    xy: 0.22,
  },
  // 1: Edge-First [1, 1, 1, 0] Diagonal (Rhombic Dodecahedral & Hexagonal Cleavage Polyhedra)
  {
    name: 'edge-first-1110',
    xw: Math.PI * 0.25,
    yw: Math.atan(1 / Math.SQRT2),
    zw: 0.16,
    xy: -0.26,
  },
  // 2: Face-First [1, 1, 0, 0] Diagonal (Octagonal & Hexagonal Tomographic Barrel)
  {
    name: 'face-first-1100',
    xw: Math.PI * 0.25,
    yw: 0.42,
    zw: -0.34,
    xy: 0.34,
  },
  // 3: Oblique Truncated-Hexahedral Axis (8-Cell Beveled Polyhedron)
  {
    name: 'oblique-8cell',
    xw: 0.56,
    yw: 0.52,
    zw: 0.44,
    xy: 0.18,
  },
];

// ---------------------------------------------------------------------------
// Main Concept 8C Boot Function
// ---------------------------------------------------------------------------
export async function bootTesseractConcept3(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(8);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.22, 4.95],
    fov: 34,
    enableOrbit: false,
    enableShadows: true,
  });

  stage.camera.lookAt(0, 0.0, 0);
  stage.controls.target.set(0, 0.0, 0);

  // -------------------------------------------------------------------------
  // Mineral & Botanical Pigments (`src/core/palette.ts`)
  // -------------------------------------------------------------------------
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const rawUmber = createPigmentUniforms('rawUmber');
  const burntSienna = createPigmentUniforms('burntSienna');
  const lapisUltramarine = createPigmentUniforms('lapisUltramarine');
  const azuritePrussian = createPigmentUniforms('azuritePrussian');
  const celadonGlauconite = createPigmentUniforms('celadonGlauconite');
  const verdigrisMalachite = createPigmentUniforms('verdigrisMalachite');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const realgarAmber = createPigmentUniforms('realgarAmber');

  const uSliceW0 = uniform(0.0);
  const uModeTransitionPulse = uniform(0.0);

  // -------------------------------------------------------------------------
  // TSL Node Materials
  // -------------------------------------------------------------------------

  // 1. Primary 3D Crystal Polyhedron Material:
  //    True Euclidean Constant-Width Fortification-Agate Banding + Straight-Skeleton Sutures
  //    + 4 Distinct Per-Cell Mineral Families (`±X, ±Y, ±Z, ±W`)
  const crystalSliceMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.22,
    metalness: 0.04,
    ior: 1.56,
    clearcoat: 0.28,
    clearcoatRoughness: 0.22,
    side: THREE.DoubleSide,
  });
  {
    const aCrystalMeta = attribute('aCrystalMeta', 'vec3'); // x: cellId [0..7], y: inradius, z: edgeCount
    const aEdgeDist012 = attribute('aEdgeDist012', 'vec3'); // physical dists to polygon edges 0, 1, 2
    const aEdgeDist345 = attribute('aEdgeDist345', 'vec3'); // physical dists to polygon edges 3, 4, 5

    const pLoc = positionLocal;
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const ndotv = abs(dot(nWorld, viewDir));
    const fresnel = clamp(float(1.0).sub(ndotv), 0.0, 1.0);

    const ptrDist = length(pWorld.sub(stage.uPointerWorld));
    const ptrGlow = exp(ptrDist.mul(ptrDist).negate().mul(1.5));

    // Branchless 6-edge tournament for 1st and 2nd minimum Euclidean edge distance:
    //   dMin    = exact physical distance from the facet perimeter (constant width on all edges!)
    //   dSecond = second-closest edge distance (dSecond - dMin == 0 along the straight skeleton!)
    let dMin: any = aEdgeDist012.x;
    let dSecond: any = float(10.0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pushEdgeDist = (d: any) => {
      const nextMin = dMin.min(d);
      const nextSecond = dSecond.min(dMin.max(d));
      dMin = nextMin;
      dSecond = nextSecond;
    };
    pushEdgeDist(aEdgeDist012.y);
    pushEdgeDist(aEdgeDist012.z);
    pushEdgeDist(aEdgeDist345.x);
    pushEdgeDist(aEdgeDist345.y);
    pushEdgeDist(aEdgeDist345.z);

    const cellId = aCrystalMeta.x;
    const inradius = aCrystalMeta.y.max(0.08);
    const axisFamily = floor(cellId.mul(0.5)); // 0: ±X, 1: ±Y, 2: ±Z, 3: ±W
    const cellParity = cellId.sub(axisFamily.mul(2.0)); // 0: -axis, 1: +axis

    // Branchless 1-hot weights for the 4 dual cubic-cell families:
    const wAxis0 = float(1.0).sub(smoothstep(0.0, 0.5, abs(axisFamily.sub(0.0))));
    const wAxis1 = float(1.0).sub(smoothstep(0.0, 0.5, abs(axisFamily.sub(1.0))));
    const wAxis2 = float(1.0).sub(smoothstep(0.0, 0.5, abs(axisFamily.sub(2.0))));
    const wAxis3 = float(1.0).sub(smoothstep(0.0, 0.5, abs(axisFamily.sub(3.0))));

    // Subtle micro-crystalline mineral tooth (keeps fortification lines razor-parallel to edges)
    const microTooth = mineralToothFbm(pLoc.mul(6.0), float(2.2)).sub(0.5).mul(0.0024);

    // Straight-skeleton mitre & medial-axis crystal suture where growth fronts collide (`dSecond == dMin`):
    const sutureDelta = clamp(dSecond.sub(dMin), 0.0, 2.0);
    const straightSkeletonSeam = exp(sutureDelta.negate().mul(96.0)).mul(
      smoothstep(0.010, 0.040, dMin)
    );
    const medialSpineGlow = exp(sutureDelta.negate().mul(32.0)).mul(
      smoothstep(0.04, 0.14, dMin)
    );

    // Helper evaluating constant-width Euclidean Liesegang fortification banding at depth `dPhys`:
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const evalEuclideanAgate = Fn(([dPhys]: [any]) => {
      const d = clamp(dPhys.add(microTooth), 0.0, 1.2);
      // Normalized depth [0 at bevel perimeter -> 1 at facet incenter] for stratigraphic zone transitions,
      // while keeping wave frequency strictly in physical world millimeters `d`!
      const rNorm = clamp(d.div(inradius), 0.0, 1.0);
      const dScaled = d.pow(0.93);

      // Primary Liesegang rhythmic precipitation wave (~12-16 constant-width bands across a facet)
      const phase1 = dScaled
        .mul(132.0)
        .add(cellParity.mul(2.1))
        .sub(uSliceW0.mul(2.4));
      const wave1 = sin(phase1);
      const wave1Norm = wave1.mul(0.5).add(0.5);

      // Secondary fine chalcedony micro-lamellae (~34 silk bands)
      const wave2 = sin(dScaled.mul(295.0).add(cellId.mul(1.3)));
      const microBand = wave2.mul(0.5).add(0.5);

      // Razor-sharp white Bone Calcite & Warm Kaolin fortification needles at Liesegang crests
      const needle1 = exp(wave1.mul(wave1).negate().mul(34.0));
      const needle2 = exp(
        sin(phase1.mul(0.5).add(0.9)).pow(2.0).negate().mul(42.0)
      );
      const calciteNeedles = needle1.max(needle2.mul(0.68));

      // Proportional stratigraphic zone transitions across the facet (`0 -> 1` from edge to core):
      const mMid = smoothstep(0.32, 0.52, rNorm);
      const mInner = smoothstep(0.66, 0.86, rNorm);

      // --- Family 0 (±X Cells): Lapis Ultramarine & Azurite Lazurite Agate ---
      const f0Outer = mix(
        kubelkaMunkGlaze(
          lapisUltramarine.K,
          lapisUltramarine.S,
          float(0.62).add(wave1Norm.mul(0.48)),
          boneCalcite.rInf
        ),
        kubelkaMunkGlaze(
          azuritePrussian.K,
          azuritePrussian.S,
          float(0.68).add(microBand.mul(0.38)),
          warmKaolin.rInf
        ),
        wave1Norm
      );
      const f0Mid = mix(
        kubelkaMunkGlaze(
          celadonGlauconite.K,
          celadonGlauconite.S,
          float(0.52).add(wave1Norm.mul(0.45)),
          boneCalcite.rInf
        ),
        kubelkaMunkGlaze(
          lapisUltramarine.K,
          lapisUltramarine.S,
          float(0.58).add(microBand.mul(0.38)),
          warmKaolin.rInf
        ),
        smoothstep(0.2, 0.8, wave1Norm)
      );
      const f0Inner = mix(
        kubelkaMunkGlaze(
          azuritePrussian.K,
          azuritePrussian.S,
          float(0.54).add(wave1Norm.mul(0.42)),
          boneCalcite.rInf
        ),
        kubelkaMunkGlaze(
          yellowOchre.K,
          yellowOchre.S,
          float(0.42).add(microBand.mul(0.30)),
          warmKaolin.rInf
        ),
        float(0.38)
      );
      const fam0 = mix(mix(f0Outer, f0Mid, mMid), f0Inner, mInner);

      // --- Family 1 (±Y Cells): Cinnabar Carnelian, Burnt Sienna & Realgar Sardonyx ---
      const f1Outer = mix(
        kubelkaMunkGlaze(
          cinnabarVermilion.K,
          cinnabarVermilion.S,
          float(0.58).add(wave1Norm.mul(0.46)),
          realgarAmber.rInf
        ),
        kubelkaMunkGlaze(
          burntSienna.K,
          burntSienna.S,
          float(0.62).add(microBand.mul(0.38)),
          warmKaolin.rInf
        ),
        wave1Norm
      );
      const f1Mid = mix(
        kubelkaMunkGlaze(
          realgarAmber.K,
          realgarAmber.S,
          float(0.48).add(wave1Norm.mul(0.48)),
          boneCalcite.rInf
        ),
        kubelkaMunkGlaze(
          cinnabarVermilion.K,
          cinnabarVermilion.S,
          float(0.46).add(microBand.mul(0.36)),
          warmKaolin.rInf
        ),
        smoothstep(0.2, 0.8, wave1Norm)
      );
      const f1Inner = mix(
        kubelkaMunkGlaze(
          burntSienna.K,
          burntSienna.S,
          float(0.52).add(wave1Norm.mul(0.42)),
          realgarAmber.rInf
        ),
        f1Mid,
        float(0.42)
      );
      const fam1 = mix(mix(f1Outer, f1Mid, mMid), f1Inner, mInner);

      // --- Family 2 (±Z Cells): Verdigris Malachite & Celadon Glauconite Jasper ---
      const f2Outer = mix(
        kubelkaMunkGlaze(
          verdigrisMalachite.K,
          verdigrisMalachite.S,
          float(0.60).add(wave1Norm.mul(0.46)),
          boneCalcite.rInf
        ),
        kubelkaMunkGlaze(
          celadonGlauconite.K,
          celadonGlauconite.S,
          float(0.64).add(microBand.mul(0.38)),
          warmKaolin.rInf
        ),
        wave1Norm
      );
      const f2Mid = mix(
        kubelkaMunkGlaze(
          celadonGlauconite.K,
          celadonGlauconite.S,
          float(0.52).add(wave1Norm.mul(0.44)),
          warmKaolin.rInf
        ),
        kubelkaMunkGlaze(
          verdigrisMalachite.K,
          verdigrisMalachite.S,
          float(0.48).add(microBand.mul(0.36)),
          boneCalcite.rInf
        ),
        smoothstep(0.2, 0.8, wave1Norm)
      );
      const f2Inner = mix(
        kubelkaMunkGlaze(
          azuritePrussian.K,
          azuritePrussian.S,
          float(0.48).add(wave1Norm.mul(0.40)),
          celadonGlauconite.rInf
        ),
        kubelkaMunkGlaze(
          yellowOchre.K,
          yellowOchre.S,
          float(0.42).add(microBand.mul(0.32)),
          boneCalcite.rInf
        ),
        float(0.42)
      );
      const fam2 = mix(mix(f2Outer, f2Mid, mMid), f2Inner, mInner);

      // --- Family 3 (±W Cells): Golden Ochre, Raw Umber & Tiger-Agate ---
      const f3Outer = mix(
        kubelkaMunkGlaze(
          rawUmber.K,
          rawUmber.S,
          float(0.62).add(wave1Norm.mul(0.44)),
          yellowOchre.rInf
        ),
        kubelkaMunkGlaze(
          yellowOchre.K,
          yellowOchre.S,
          float(0.58).add(microBand.mul(0.38)),
          warmKaolin.rInf
        ),
        wave1Norm
      );
      const f3Mid = mix(
        kubelkaMunkGlaze(
          yellowOchre.K,
          yellowOchre.S,
          float(0.52).add(wave1Norm.mul(0.48)),
          boneCalcite.rInf
        ),
        kubelkaMunkGlaze(
          burntSienna.K,
          burntSienna.S,
          float(0.46).add(microBand.mul(0.34)),
          warmKaolin.rInf
        ),
        smoothstep(0.2, 0.8, wave1Norm)
      );
      const f3Inner = mix(
        kubelkaMunkGlaze(
          rawUmber.K,
          rawUmber.S,
          float(0.48).add(wave1Norm.mul(0.40)),
          realgarAmber.rInf
        ),
        f3Mid,
        float(0.48)
      );
      const fam3 = mix(mix(f3Outer, f3Mid, mMid), f3Inner, mInner);

      // Select the active cell's mineral family
      const cellAgate = fam0
        .mul(wAxis0)
        .add(fam1.mul(wAxis1))
        .add(fam2.mul(wAxis2))
        .add(fam3.mul(wAxis3));

      // Inlay razor-sharp Bone Calcite & Warm Kaolin Liesegang fortification needles
      const needleColor = mix(boneCalcite.rInf, warmKaolin.rInf, microBand.mul(0.32));
      const withNeedles = mix(cellAgate, needleColor, calciteNeedles.mul(0.72));

      // Inlay the straight-skeleton mitre & medial-ridge crystal growth suture
      const sutureColor = mix(boneCalcite.rInf, yellowOchre.rInf, medialSpineGlow.mul(0.48));
      return mix(withNeedles, sutureColor, straightSkeletonSeam.mul(0.65));
    });

    // Evaluate surface + deep subsurface parallax layer (shifted inward along grazing angle)
    const surfaceAgate = evalEuclideanAgate(dMin);
    const parallaxShift = float(1.0).sub(ndotv).mul(0.010).add(0.004);
    const deepAgate = evalEuclideanAgate(dMin.add(parallaxShift));
    const layeredAgate = mix(surfaceAgate, deepAgate, float(0.28));

    // Crisp golden-ochre & calcite bevel collar right along `dMin -> 0` (polygon edge)
    const edgeCollar = float(1.0).sub(smoothstep(float(0.003), float(0.020), dMin));
    const rimGlaze = mix(boneCalcite.rInf, yellowOchre.rInf, float(0.42));
    const finalSurface = mix(layeredAgate, rimGlaze, edgeCollar.mul(0.70));

    // Subsurface Beer-Lambert glow along the straight-skeleton core & under pointer proximity
    const coreDepthNorm = clamp(dMin.div(inradius), 0.0, 1.0);
    const trans = beerLambertTransmittance(
      realgarAmber.extinction,
      float(1.25).sub(coreDepthNorm.mul(0.65))
    );
    const subsurfaceWarmth = realgarAmber.rInf
      .mul(trans)
      .mul(
        medialSpineGlow.mul(0.14)
          .add(ptrGlow.mul(0.18))
          .add(uModeTransitionPulse.mul(0.22))
      );

    crystalSliceMat.colorNode = clamp(finalSurface.add(subsurfaceWarmth), 0.02, 0.96);
    crystalSliceMat.emissiveNode = subsurfaceWarmth.add(
      layeredAgate.mul(0.12).add(rimGlaze.mul(fresnel.pow(2.4).mul(0.10)))
    );
    crystalSliceMat.roughnessNode = clamp(
      float(0.20).sub(straightSkeletonSeam.mul(0.06)),
      0.14,
      0.28
    );
  }

  // 2. Beveled Calcite & Ochre Crystal Facet Edge Frame Material
  const bevelFrameMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.22,
    metalness: 0.36,
  });
  {
    const fbm = mineralToothFbm(positionLocal, float(6.0));
    const calciteOchre = mix(
      kubelkaMunkGlaze(boneCalcite.K, boneCalcite.S, float(0.72), warmKaolin.rInf),
      kubelkaMunkGlaze(yellowOchre.K, yellowOchre.S, float(0.58), rawUmber.rInf),
      float(0.45).add(fbm.mul(0.2))
    );
    bevelFrameMat.colorNode = calciteOchre;
  }

  // 3. 32 4D Tesseract Tubular Armature Strut Material (`ghostStrutMat`)
  //    Colors Ana (`w > w0`) in warm Calcite/Ochre, Kata (`w < w0`) in Azurite/Lapis,
  //    illuminates the exact `w == w0` piercing collar in Cinnabar & Realgar Amber,
  //    and fades far polar segments so foreground struts never visually block the crystal faces!
  const ghostStrutMat = new THREE.MeshStandardNodeMaterial({
    transparent: true,
    depthWrite: false,
    roughness: 0.26,
    metalness: 0.22,
  });
  {
    const aStrutMeta = attribute('aStrutMeta', 'vec3'); // x: rw - w0, y: isPierced, z: axisNorm
    const deltaW = aStrutMeta.x;
    const isPierced = aStrutMeta.y;

    const anaBlend = smoothstep(float(-0.25), float(0.25), deltaW);
    const kataGlaze = kubelkaMunkGlaze(
      azuritePrussian.K,
      azuritePrussian.S,
      float(0.52),
      boneCalcite.rInf
    );
    const anaGlaze = mix(
      boneCalcite.rInf,
      kubelkaMunkGlaze(yellowOchre.K, yellowOchre.S, float(0.42), warmKaolin.rInf),
      float(0.45)
    );
    const baseStrutColor = mix(kataGlaze, anaGlaze, anaBlend);

    // Hot Cinnabar/Realgar collar right where the 4D strut pierces `w = w0` (`deltaW == 0`)
    const pierceCollar = exp(deltaW.mul(deltaW).negate().mul(22.0)).mul(isPierced);
    const nearSliceMask = exp(deltaW.mul(deltaW).negate().mul(2.4));
    const collarColor = mix(cinnabarVermilion.rInf, realgarAmber.rInf, float(0.45));

    ghostStrutMat.colorNode = mix(baseStrutColor, collarColor, pierceCollar.mul(0.88));
    ghostStrutMat.emissiveNode = collarColor
      .mul(pierceCollar.mul(0.48))
      .add(baseStrutColor.mul(0.10));
    ghostStrutMat.opacityNode = clamp(
      float(0.22).add(nearSliceMask.mul(0.45)).add(pierceCollar.mul(0.33)),
      0.20,
      0.95
    );
  }

  // 4. Outer 4D Vertex Nodes & Active `w = w0` Piercing Caliper Jewels
  const ghostVertexMat = new THREE.MeshStandardNodeMaterial({
    transparent: true,
    opacity: 0.55,
    roughness: 0.28,
    metalness: 0.18,
  });
  ghostVertexMat.colorNode = kubelkaMunkGlaze(
    azuritePrussian.K,
    azuritePrussian.S,
    float(0.46),
    boneCalcite.rInf
  );

  const piercingCaliperMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.16,
    metalness: 0.24,
  });
  piercingCaliperMat.colorNode = kubelkaMunkGlaze(
    cinnabarVermilion.K,
    cinnabarVermilion.S,
    float(0.66),
    realgarAmber.rInf
  );
  piercingCaliperMat.emissiveNode = mix(
    cinnabarVermilion.rInf,
    realgarAmber.rInf,
    float(0.55)
  ).mul(0.38);

  // -------------------------------------------------------------------------
  // Assemble Pure 4D Crystal Slicer & Projective Armature Hierarchy
  // -------------------------------------------------------------------------
  const tomographRoot = new THREE.Group();
  tomographRoot.name = 'HyperplaneCrystallographicTomograph';
  tomographRoot.rotation.set(0.16, -0.28, 0.0);
  stage.specimenGroup.add(tomographRoot);

  // --- A. 32 3D Tubular 4D Tesseract Armature Struts + 16 Vertex Nodes + 32 Piercing Calipers ---
  const ghostStrutGeom = new THREE.BufferGeometry();
  const ghostStrutPosAttr = new THREE.BufferAttribute(_ghostStrutPositions, 3);
  ghostStrutPosAttr.setUsage(THREE.DynamicDrawUsage);
  const ghostStrutNormAttr = new THREE.BufferAttribute(_ghostStrutNormals, 3);
  ghostStrutNormAttr.setUsage(THREE.DynamicDrawUsage);
  const ghostStrutMetaAttr = new THREE.BufferAttribute(_ghostStrutMeta, 3);
  ghostStrutMetaAttr.setUsage(THREE.DynamicDrawUsage);

  ghostStrutGeom.setAttribute('position', ghostStrutPosAttr);
  ghostStrutGeom.setAttribute('normal', ghostStrutNormAttr);
  ghostStrutGeom.setAttribute('aStrutMeta', ghostStrutMetaAttr);

  const ghostStrutIndices: number[] = [];
  for (let e = 0; e < NUM_EDGES; e++) {
    const b = e * VERTS_PER_BEVEL;
    for (let f = 0; f < 6; f++) {
      const fb = b + f * 4;
      ghostStrutIndices.push(fb, fb + 1, fb + 2, fb, fb + 2, fb + 3);
    }
  }
  ghostStrutGeom.setIndex(ghostStrutIndices);

  const ghostStrutMesh = new THREE.Mesh(ghostStrutGeom, ghostStrutMat);
  ghostStrutMesh.renderOrder = 3;
  ghostStrutMesh.frustumCulled = false;
  tomographRoot.add(ghostStrutMesh);

  const ghostVertexMeshes: THREE.Mesh[] = [];
  const ghostSphereGeom = new THREE.SphereGeometry(0.022, 12, 10);
  for (let v = 0; v < NUM_VERTICES; v++) {
    const vMesh = new THREE.Mesh(ghostSphereGeom, ghostVertexMat);
    tomographRoot.add(vMesh);
    ghostVertexMeshes.push(vMesh);
  }

  // 32 Sculpted Octahedral Caliper Nodes that ride on the 32 4D struts at `w = w0`
  const piercingNodeMeshes: THREE.Mesh[] = [];
  const caliperOctGeom = new THREE.OctahedronGeometry(0.036, 1);
  for (let e = 0; e < NUM_EDGES; e++) {
    const cMesh = new THREE.Mesh(caliperOctGeom, piercingCaliperMat);
    cMesh.visible = false;
    tomographRoot.add(cMesh);
    piercingNodeMeshes.push(cMesh);
  }

  // --- B. Active 3D Crystal Slice Polyhedron Mesh (`w = w0`) ---
  const crystalGeom = new THREE.BufferGeometry();
  const crystalPosAttr = new THREE.BufferAttribute(_crystalPositions, 3);
  crystalPosAttr.setUsage(THREE.DynamicDrawUsage);
  const crystalNormAttr = new THREE.BufferAttribute(_crystalNormals, 3);
  crystalNormAttr.setUsage(THREE.DynamicDrawUsage);
  const crystalMetaAttr = new THREE.BufferAttribute(_crystalMeta, 3);
  crystalMetaAttr.setUsage(THREE.DynamicDrawUsage);
  const crystalEdgeDist012Attr = new THREE.BufferAttribute(_crystalEdgeDist012, 3);
  crystalEdgeDist012Attr.setUsage(THREE.DynamicDrawUsage);
  const crystalEdgeDist345Attr = new THREE.BufferAttribute(_crystalEdgeDist345, 3);
  crystalEdgeDist345Attr.setUsage(THREE.DynamicDrawUsage);

  crystalGeom.setAttribute('position', crystalPosAttr);
  crystalGeom.setAttribute('normal', crystalNormAttr);
  crystalGeom.setAttribute('aCrystalMeta', crystalMetaAttr);
  crystalGeom.setAttribute('aEdgeDist012', crystalEdgeDist012Attr);
  crystalGeom.setAttribute('aEdgeDist345', crystalEdgeDist345Attr);

  const crystalMesh = new THREE.Mesh(crystalGeom, crystalSliceMat);
  crystalMesh.castShadow = true;
  crystalMesh.receiveShadow = true;
  crystalMesh.frustumCulled = false;
  tomographRoot.add(crystalMesh);

  // --- C. Beveled Calcite/Ochre Edge Frame Struts on the Active Polyhedron ---
  const bevelGeom = new THREE.BufferGeometry();
  const bevelPosAttr = new THREE.BufferAttribute(_bevelPositions, 3);
  bevelPosAttr.setUsage(THREE.DynamicDrawUsage);
  const bevelNormAttr = new THREE.BufferAttribute(_bevelNormals, 3);
  bevelNormAttr.setUsage(THREE.DynamicDrawUsage);
  bevelGeom.setAttribute('position', bevelPosAttr);
  bevelGeom.setAttribute('normal', bevelNormAttr);

  const bevelIndices: number[] = [];
  for (let e = 0; e < MAX_BEVEL_EDGES; e++) {
    const b = e * VERTS_PER_BEVEL;
    for (let f = 0; f < 6; f++) {
      const fb = b + f * 4;
      bevelIndices.push(fb, fb + 1, fb + 2, fb, fb + 2, fb + 3);
    }
  }
  bevelGeom.setIndex(bevelIndices);

  const bevelFrameMesh = new THREE.Mesh(bevelGeom, bevelFrameMat);
  bevelFrameMesh.castShadow = true;
  bevelFrameMesh.frustumCulled = false;
  tomographRoot.add(bevelFrameMesh);

  // --- D. Projective Neighboring Tomographic Contour Wireframes (`w0 + Δw_k`) ---
  //     (Vertices sit with 0.000 projective accuracy on the 32 4D struts!)
  const contourLineGeom = new THREE.BufferGeometry();
  const contourLineAttr = new THREE.BufferAttribute(_contourLinePositions, 3);
  contourLineAttr.setUsage(THREE.DynamicDrawUsage);
  contourLineGeom.setAttribute('position', contourLineAttr);
  const contourLineMat = new THREE.LineBasicNodeMaterial({
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
  });
  contourLineMat.colorNode = mix(boneCalcite.rInf, celadonGlauconite.rInf, float(0.42));
  const contourLinesMesh = new THREE.LineSegments(contourLineGeom, contourLineMat);
  contourLinesMesh.frustumCulled = false;
  tomographRoot.add(contourLinesMesh);

  // -------------------------------------------------------------------------
  // Zero-Allocation 3D Box Strut Emitter into Target Position/Normal Arrays
  // -------------------------------------------------------------------------
  function writeBoxStrutInto(
    posArr: Float32Array,
    normArr: Float32Array,
    strutIdx: number,
    maxStruts: number,
    p0x: number,
    p0y: number,
    p0z: number,
    p1x: number,
    p1y: number,
    p1z: number,
    radius: number,
    metaArr?: Float32Array,
    m0x = 0,
    m1x = 0,
    my = 0,
    mz = 0
  ): void {
    if (strutIdx >= maxStruts) return;
    _edgeDir.set(p1x - p0x, p1y - p0y, p1z - p0z);
    const len = _edgeDir.length();
    if (len < 1e-5) return;
    _edgeDir.divideScalar(len);

    if (Math.abs(_edgeDir.y) < 0.9) {
      _edgeUp.set(0, 1, 0);
    } else {
      _edgeUp.set(1, 0, 0);
    }
    _edgeSide.crossVectors(_edgeDir, _edgeUp).normalize().multiplyScalar(radius);
    _edgeUp.crossVectors(_edgeSide, _edgeDir).normalize().multiplyScalar(radius);

    const sx = _edgeSide.x;
    const sy = _edgeSide.y;
    const sz = _edgeSide.z;
    const ux = _edgeUp.x;
    const uy = _edgeUp.y;
    const uz = _edgeUp.z;

    const c00x = p0x - sx - ux,
      c00y = p0y - sy - uy,
      c00z = p0z - sz - uz;
    const c10x = p0x + sx - ux,
      c10y = p0y + sy - uy,
      c10z = p0z + sz - uz;
    const c11x = p0x + sx + ux,
      c11y = p0y + sy + uy,
      c11z = p0z + sz + uz;
    const c01x = p0x - sx + ux,
      c01y = p0y - sy + uy,
      c01z = p0z - sz + uz;

    const d00x = p1x - sx - ux,
      d00y = p1y - sy - uy,
      d00z = p1z - sz - uz;
    const d10x = p1x + sx - ux,
      d10y = p1y + sy - uy,
      d10z = p1z + sz - uz;
    const d11x = p1x + sx + ux,
      d11y = p1y + sy + uy,
      d11z = p1z + sz + uz;
    const d01x = p1x - sx + ux,
      d01y = p1y - sy + uy,
      d01z = p1z - sz + uz;

    const base = strutIdx * VERTS_PER_BEVEL * 3;
    const nSideX = sx / radius,
      nSideY = sy / radius,
      nSideZ = sz / radius;
    const nUpX = ux / radius,
      nUpY = uy / radius,
      nUpZ = uz / radius;

    const writeQuad = (
      qOffset: number,
      ax: number,
      ay: number,
      az: number,
      amx: number,
      bx: number,
      by: number,
      bz: number,
      bmx: number,
      cx: number,
      cy: number,
      cz: number,
      cmx: number,
      dx: number,
      dy: number,
      dz: number,
      dmx: number,
      nx: number,
      ny: number,
      nz: number
    ) => {
      const idx = base + qOffset * 12;
      posArr[idx + 0] = ax;
      posArr[idx + 1] = ay;
      posArr[idx + 2] = az;
      posArr[idx + 3] = bx;
      posArr[idx + 4] = by;
      posArr[idx + 5] = bz;
      posArr[idx + 6] = cx;
      posArr[idx + 7] = cy;
      posArr[idx + 8] = cz;
      posArr[idx + 9] = dx;
      posArr[idx + 10] = dy;
      posArr[idx + 11] = dz;

      for (let k = 0; k < 4; k++) {
        normArr[idx + k * 3 + 0] = nx;
        normArr[idx + k * 3 + 1] = ny;
        normArr[idx + k * 3 + 2] = nz;
      }

      if (metaArr) {
        metaArr[idx + 0] = amx;
        metaArr[idx + 1] = my;
        metaArr[idx + 2] = mz;
        metaArr[idx + 3] = bmx;
        metaArr[idx + 4] = my;
        metaArr[idx + 5] = mz;
        metaArr[idx + 6] = cmx;
        metaArr[idx + 7] = my;
        metaArr[idx + 8] = mz;
        metaArr[idx + 9] = dmx;
        metaArr[idx + 10] = my;
        metaArr[idx + 11] = mz;
      }
    };

    writeQuad(
      0,
      c10x, c10y, c10z, m0x,
      d10x, d10y, d10z, m1x,
      d11x, d11y, d11z, m1x,
      c11x, c11y, c11z, m0x,
      nSideX, nSideY, nSideZ
    );
    writeQuad(
      1,
      d00x, d00y, d00z, m1x,
      c00x, c00y, c00z, m0x,
      c01x, c01y, c01z, m0x,
      d01x, d01y, d01z, m1x,
      -nSideX, -nSideY, -nSideZ
    );
    writeQuad(
      2,
      c01x, c01y, c01z, m0x,
      c11x, c11y, c11z, m0x,
      d11x, d11y, d11z, m1x,
      d01x, d01y, d01z, m1x,
      nUpX, nUpY, nUpZ
    );
    writeQuad(
      3,
      c10x, c10y, c10z, m0x,
      c00x, c00y, c00z, m0x,
      d00x, d00y, d00z, m1x,
      d10x, d10y, d10z, m1x,
      -nUpX, -nUpY, -nUpZ
    );
    writeQuad(
      4,
      d00x, d00y, d00z, m1x,
      d01x, d01y, d01z, m1x,
      d11x, d11y, d11z, m1x,
      d10x, d10y, d10z, m1x,
      _edgeDir.x, _edgeDir.y, _edgeDir.z
    );
    writeQuad(
      5,
      c00x, c00y, c00z, m0x,
      c10x, c10y, c10z, m0x,
      c11x, c11y, c11z, m0x,
      c01x, c01y, c01z, m0x,
      -_edgeDir.x, -_edgeDir.y, -_edgeDir.z
    );
  }

  // -------------------------------------------------------------------------
  // Core 4D Hyperplane Slicer (`w = wSlice`) for Both Active Core & Contour Rings
  // -------------------------------------------------------------------------
  let activeCrystalVertexCount = 0;
  let activeBevelEdgeCount = 0;
  let activeContourSegmentCount = 0;
  let activePiercedEdgeCount = 0;

  function evalEdgeDistsAtPoint(px: number, py: number, pz: number, m: number): void {
    for (let k = 0; k < 6; k++) {
      if (k < m) {
        const ox = _orderedPolyPts[k * 3 + 0];
        const oy = _orderedPolyPts[k * 3 + 1];
        const oz = _orderedPolyPts[k * 3 + 2];
        const nx = _edgeInwardNormals[k * 3 + 0];
        const ny = _edgeInwardNormals[k * 3 + 1];
        const nz = _edgeInwardNormals[k * 3 + 2];
        const d = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz;
        _vertEdgeDists[k] = d > 0 ? d : 0;
      } else {
        _vertEdgeDists[k] = 10.0;
      }
    }
  }

  function writeCrystalVertex(
    vSlot: number,
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
    cellId: number,
    inradius: number,
    edgeCount: number
  ): void {
    const base = vSlot * 3;
    _crystalPositions[base + 0] = px;
    _crystalPositions[base + 1] = py;
    _crystalPositions[base + 2] = pz;
    _crystalNormals[base + 0] = nx;
    _crystalNormals[base + 1] = ny;
    _crystalNormals[base + 2] = nz;
    _crystalMeta[base + 0] = cellId;
    _crystalMeta[base + 1] = inradius;
    _crystalMeta[base + 2] = edgeCount;

    evalEdgeDistsAtPoint(px, py, pz, edgeCount);
    _crystalEdgeDist012[base + 0] = _vertEdgeDists[0];
    _crystalEdgeDist012[base + 1] = _vertEdgeDists[1];
    _crystalEdgeDist012[base + 2] = _vertEdgeDists[2];
    _crystalEdgeDist345[base + 0] = _vertEdgeDists[3];
    _crystalEdgeDist345[base + 1] = _vertEdgeDists[4];
    _crystalEdgeDist345[base + 2] = _vertEdgeDists[5];
  }

  function clipTesseractAtHyperplane(
    wSlice: number,
    scale3D: number,
    isPrimaryCore: boolean
  ): void {
    let hitCount = 0;
    _polyCenter.set(0, 0, 0);

    for (let e = 0; e < NUM_EDGES; e++) {
      const i0 = _edgePairs[e * 2 + 0] * 4;
      const i1 = _edgePairs[e * 2 + 1] * 4;
      const w0 = _rotVerts4D[i0 + 3];
      const w1 = _rotVerts4D[i1 + 3];
      const d0 = w0 - wSlice;
      const d1 = w1 - wSlice;

      if ((d0 <= 0 && d1 >= 0) || (d0 >= 0 && d1 <= 0)) {
        const denom = w1 - w0;
        const t = Math.abs(denom) < 1e-6 ? 0.5 : (wSlice - w0) / denom;
        const hx = (_rotVerts4D[i0 + 0] + (_rotVerts4D[i1 + 0] - _rotVerts4D[i0 + 0]) * t) * scale3D;
        const hy = (_rotVerts4D[i0 + 1] + (_rotVerts4D[i1 + 1] - _rotVerts4D[i0 + 1]) * t) * scale3D;
        const hz = (_rotVerts4D[i0 + 2] + (_rotVerts4D[i1 + 2] - _rotVerts4D[i0 + 2]) * t) * scale3D;

        _edgeHitValid[e] = 1;
        _edgeHitPoints3D[e * 3 + 0] = hx;
        _edgeHitPoints3D[e * 3 + 1] = hy;
        _edgeHitPoints3D[e * 3 + 2] = hz;
        _polyCenter.x += hx;
        _polyCenter.y += hy;
        _polyCenter.z += hz;
        hitCount++;
      } else {
        _edgeHitValid[e] = 0;
      }

      if (isPrimaryCore) {
        _primaryEdgeHitValid[e] = _edgeHitValid[e];
        _primaryEdgeHitPoints3D[e * 3 + 0] = _edgeHitPoints3D[e * 3 + 0];
        _primaryEdgeHitPoints3D[e * 3 + 1] = _edgeHitPoints3D[e * 3 + 1];
        _primaryEdgeHitPoints3D[e * 3 + 2] = _edgeHitPoints3D[e * 3 + 2];
      }
    }

    if (isPrimaryCore) {
      activePiercedEdgeCount = hitCount;
    }

    if (hitCount < 4) return;
    _polyCenter.divideScalar(hitCount);

    for (let c = 0; c < NUM_CELLS; c++) {
      let m = 0;
      for (let k = 0; k < 12; k++) {
        const e = _cellEdges[c * 12 + k];
        if (!_edgeHitValid[e]) continue;

        const px = _edgeHitPoints3D[e * 3 + 0];
        const py = _edgeHitPoints3D[e * 3 + 1];
        const pz = _edgeHitPoints3D[e * 3 + 2];

        let isDup = false;
        for (let prev = 0; prev < m; prev++) {
          const dx = _polyPts[prev * 3 + 0] - px;
          const dy = _polyPts[prev * 3 + 1] - py;
          const dz = _polyPts[prev * 3 + 2] - pz;
          if (dx * dx + dy * dy + dz * dz < 1e-5) {
            isDup = true;
            break;
          }
        }
        if (!isDup && m < 8) {
          _polyPts[m * 3 + 0] = px;
          _polyPts[m * 3 + 1] = py;
          _polyPts[m * 3 + 2] = pz;
          m++;
        }
      }

      if (m < 3) continue;

      _faceCentroid.set(0, 0, 0);
      for (let j = 0; j < m; j++) {
        _faceCentroid.x += _polyPts[j * 3 + 0];
        _faceCentroid.y += _polyPts[j * 3 + 1];
        _faceCentroid.z += _polyPts[j * 3 + 2];
      }
      _faceCentroid.divideScalar(m);

      _faceNormal.set(
        _rotCellNormals4D[c * 4 + 0],
        _rotCellNormals4D[c * 4 + 1],
        _rotCellNormals4D[c * 4 + 2]
      );
      if (_faceNormal.lengthSq() < 1e-5) {
        _tangent1.set(
          _polyPts[0] - _faceCentroid.x,
          _polyPts[1] - _faceCentroid.y,
          _polyPts[2] - _faceCentroid.z
        );
        _tangent2.set(
          _polyPts[3] - _faceCentroid.x,
          _polyPts[4] - _faceCentroid.y,
          _polyPts[5] - _faceCentroid.z
        );
        _faceNormal.crossVectors(_tangent1, _tangent2);
      }
      _faceNormal.normalize();

      _diffVec.subVectors(_faceCentroid, _polyCenter);
      if (_faceNormal.dot(_diffVec) < 0) {
        _faceNormal.negate();
      }

      _tangent1
        .set(
          _polyPts[0] - _faceCentroid.x,
          _polyPts[1] - _faceCentroid.y,
          _polyPts[2] - _faceCentroid.z
        )
        .normalize();
      _tangent2.crossVectors(_faceNormal, _tangent1).normalize();

      for (let j = 0; j < m; j++) {
        const dx = _polyPts[j * 3 + 0] - _faceCentroid.x;
        const dy = _polyPts[j * 3 + 1] - _faceCentroid.y;
        const dz = _polyPts[j * 3 + 2] - _faceCentroid.z;
        const u = dx * _tangent1.x + dy * _tangent1.y + dz * _tangent1.z;
        const v = dx * _tangent2.x + dy * _tangent2.y + dz * _tangent2.z;
        _polyAngles[j] = Math.atan2(v, u);
        _polyOrder[j] = j;
      }

      for (let i = 1; i < m; i++) {
        const keyIdx = _polyOrder[i];
        const keyAng = _polyAngles[keyIdx];
        let j = i - 1;
        while (j >= 0 && _polyAngles[_polyOrder[j]] > keyAng) {
          _polyOrder[j + 1] = _polyOrder[j];
          j--;
        }
        _polyOrder[j + 1] = keyIdx;
      }

      // Copy ordered polygon points and compute inward unit 2D edge normals for exact Euclidean distances
      const edgeCount = Math.min(m, 6);
      for (let j = 0; j < edgeCount; j++) {
        const idxA = _polyOrder[j] * 3;
        _orderedPolyPts[j * 3 + 0] = _polyPts[idxA + 0];
        _orderedPolyPts[j * 3 + 1] = _polyPts[idxA + 1];
        _orderedPolyPts[j * 3 + 2] = _polyPts[idxA + 2];
      }

      let inradius = 10.0;
      if (isPrimaryCore) {
        for (let j = 0; j < edgeCount; j++) {
          const jNext = (j + 1) % edgeCount;
          const ax = _orderedPolyPts[j * 3 + 0];
          const ay = _orderedPolyPts[j * 3 + 1];
          const az = _orderedPolyPts[j * 3 + 2];
          const bx = _orderedPolyPts[jNext * 3 + 0];
          const by = _orderedPolyPts[jNext * 3 + 1];
          const bz = _orderedPolyPts[jNext * 3 + 2];

          _edgeDir.set(bx - ax, by - ay, bz - az);
          const eLen = _edgeDir.length();
          if (eLen > 1e-6) {
            _edgeDir.divideScalar(eLen);
          }
          _inwardN.crossVectors(_faceNormal, _edgeDir).normalize();
          const toCenterDot =
            (_faceCentroid.x - ax) * _inwardN.x +
            (_faceCentroid.y - ay) * _inwardN.y +
            (_faceCentroid.z - az) * _inwardN.z;
          if (toCenterDot < 0) {
            _inwardN.negate();
          }
          _edgeInwardNormals[j * 3 + 0] = _inwardN.x;
          _edgeInwardNormals[j * 3 + 1] = _inwardN.y;
          _edgeInwardNormals[j * 3 + 2] = _inwardN.z;

          const cDist = Math.abs(toCenterDot);
          if (cDist < inradius) {
            inradius = cDist;
          }
        }
        inradius = Math.max(0.05, inradius);
      }

      for (let j = 0; j < edgeCount; j++) {
        const jNext = (j + 1) % edgeCount;
        const ax = _orderedPolyPts[j * 3 + 0];
        const ay = _orderedPolyPts[j * 3 + 1];
        const az = _orderedPolyPts[j * 3 + 2];
        const bx = _orderedPolyPts[jNext * 3 + 0];
        const by = _orderedPolyPts[jNext * 3 + 1];
        const bz = _orderedPolyPts[jNext * 3 + 2];

        if (isPrimaryCore) {
          if (activeCrystalVertexCount + 3 <= MAX_CRYSTAL_VERTS) {
            writeCrystalVertex(
              activeCrystalVertexCount + 0,
              _faceCentroid.x,
              _faceCentroid.y,
              _faceCentroid.z,
              _faceNormal.x,
              _faceNormal.y,
              _faceNormal.z,
              c,
              inradius,
              edgeCount
            );
            writeCrystalVertex(
              activeCrystalVertexCount + 1,
              ax,
              ay,
              az,
              _faceNormal.x,
              _faceNormal.y,
              _faceNormal.z,
              c,
              inradius,
              edgeCount
            );
            writeCrystalVertex(
              activeCrystalVertexCount + 2,
              bx,
              by,
              bz,
              _faceNormal.x,
              _faceNormal.y,
              _faceNormal.z,
              c,
              inradius,
              edgeCount
            );
            activeCrystalVertexCount += 3;
          }

          if (activeBevelEdgeCount < MAX_BEVEL_EDGES) {
            writeBoxStrutInto(
              _bevelPositions,
              _bevelNormals,
              activeBevelEdgeCount,
              MAX_BEVEL_EDGES,
              ax,
              ay,
              az,
              bx,
              by,
              bz,
              0.013
            );
            activeBevelEdgeCount++;
          }
        } else {
          if (activeContourSegmentCount < MAX_CONTOUR_SEGMENTS) {
            const lBase = activeContourSegmentCount * 6;
            _contourLinePositions[lBase + 0] = ax;
            _contourLinePositions[lBase + 1] = ay;
            _contourLinePositions[lBase + 2] = az;
            _contourLinePositions[lBase + 3] = bx;
            _contourLinePositions[lBase + 4] = by;
            _contourLinePositions[lBase + 5] = bz;
            activeContourSegmentCount++;
          }
        }
      }
    }
  }

  function updateTomographGeometry(
    w0: number,
    angleXW: number,
    angleYW: number,
    angleZW: number,
    angleXY: number,
    angleXZ: number
  ): void {
    // Compose 4D crystallographic orientation first, then 3D subspace rotation (which preserves W!)
    setIdentity4D(_rotMat4D);
    applyGivens4D(_rotMat4D, 0, 3, angleXW);
    applyGivens4D(_rotMat4D, 1, 3, angleYW);
    applyGivens4D(_rotMat4D, 2, 3, angleZW);
    applyGivens4D(_rotMat4D, 0, 1, angleXY);
    applyGivens4D(_rotMat4D, 0, 2, angleXZ);

    for (let i = 0; i < NUM_VERTICES; i++) {
      const bx = _baseVerts4D[i * 4 + 0];
      const by = _baseVerts4D[i * 4 + 1];
      const bz = _baseVerts4D[i * 4 + 2];
      const bw = _baseVerts4D[i * 4 + 3];

      const rx = _rotMat4D[0] * bx + _rotMat4D[1] * by + _rotMat4D[2] * bz + _rotMat4D[3] * bw;
      const ry = _rotMat4D[4] * bx + _rotMat4D[5] * by + _rotMat4D[6] * bz + _rotMat4D[7] * bw;
      const rz = _rotMat4D[8] * bx + _rotMat4D[9] * by + _rotMat4D[10] * bz + _rotMat4D[11] * bw;
      const rw = _rotMat4D[12] * bx + _rotMat4D[13] * by + _rotMat4D[14] * bz + _rotMat4D[15] * bw;

      _rotVerts4D[i * 4 + 0] = rx;
      _rotVerts4D[i * 4 + 1] = ry;
      _rotVerts4D[i * 4 + 2] = rz;
      _rotVerts4D[i * 4 + 3] = rw;

      // Exact projective scale relative to `w = w0`:
      // At `rw == w0`, `gScale === SCALE_3D`, guaranteeing 0.000 discrepancy with crystal corners!
      const gScale = projectScaleAtW(rw, w0);
      const gx = rx * gScale;
      const gy = ry * gScale;
      const gz = rz * gScale;
      const vMesh = ghostVertexMeshes[i];
      vMesh.position.set(gx, gy, gz);
      const wProx = Math.exp(-Math.pow((rw - w0) * 2.2, 2));
      vMesh.scale.setScalar(0.80 + wProx * 0.50);
    }

    for (let c = 0; c < NUM_CELLS; c++) {
      const nx = _baseCellNormals4D[c * 4 + 0];
      const ny = _baseCellNormals4D[c * 4 + 1];
      const nz = _baseCellNormals4D[c * 4 + 2];
      const nw = _baseCellNormals4D[c * 4 + 3];

      _rotCellNormals4D[c * 4 + 0] =
        _rotMat4D[0] * nx + _rotMat4D[1] * ny + _rotMat4D[2] * nz + _rotMat4D[3] * nw;
      _rotCellNormals4D[c * 4 + 1] =
        _rotMat4D[4] * nx + _rotMat4D[5] * ny + _rotMat4D[6] * nz + _rotMat4D[7] * nw;
      _rotCellNormals4D[c * 4 + 2] =
        _rotMat4D[8] * nx + _rotMat4D[9] * ny + _rotMat4D[10] * nz + _rotMat4D[11] * nw;
      _rotCellNormals4D[c * 4 + 3] =
        _rotMat4D[12] * nx + _rotMat4D[13] * ny + _rotMat4D[14] * nz + _rotMat4D[15] * nw;
    }

    activeCrystalVertexCount = 0;
    activeBevelEdgeCount = 0;
    activeContourSegmentCount = 0;

    // 1. Slice active 3D crystal polyhedron at `w = w0` (`scale = SCALE_3D`)
    clipTesseractAtHyperplane(w0, SCALE_3D, true);

    // 2. Build the 32 3D Tubular 4D Tesseract Armature Struts & position the 32 Piercing Calipers
    for (let e = 0; e < NUM_EDGES; e++) {
      const i0 = _edgePairs[e * 2 + 0];
      const i1 = _edgePairs[e * 2 + 1];
      const p0 = ghostVertexMeshes[i0].position;
      const p1 = ghostVertexMeshes[i1].position;
      const dw0 = _rotVerts4D[i0 * 4 + 3] - w0;
      const dw1 = _rotVerts4D[i1 * 4 + 3] - w0;
      const isPierced = _primaryEdgeHitValid[e] ? 1.0 : 0.0;

      writeBoxStrutInto(
        _ghostStrutPositions,
        _ghostStrutNormals,
        e,
        NUM_EDGES,
        p0.x,
        p0.y,
        p0.z,
        p1.x,
        p1.y,
        p1.z,
        isPierced > 0.5 ? 0.0078 : 0.0052,
        _ghostStrutMeta,
        dw0,
        dw1,
        isPierced,
        (e & 3) / 3.0
      );

      const caliperNode = piercingNodeMeshes[e];
      if (_primaryEdgeHitValid[e]) {
        caliperNode.visible = true;
        caliperNode.position.set(
          _primaryEdgeHitPoints3D[e * 3 + 0],
          _primaryEdgeHitPoints3D[e * 3 + 1],
          _primaryEdgeHitPoints3D[e * 3 + 2]
        );
      } else {
        caliperNode.visible = false;
      }
    }
    ghostStrutPosAttr.needsUpdate = true;
    ghostStrutNormAttr.needsUpdate = true;
    ghostStrutMetaAttr.needsUpdate = true;

    // 3. Slice the 6 neighboring projective tomographic contour rings (`w_k = w0 + Δw_k`)
    //    Because `scaleLamina = projectScaleAtW(wk, w0)`, every contour corner sits on the 4D struts!
    for (let k = 0; k < NUM_LAMINA_SLICES; k++) {
      const dw = LAMINA_OFFSETS[k];
      const wk = w0 + dw;
      if (Math.abs(wk) < 1.85) {
        const scaleLamina = projectScaleAtW(wk, w0);
        clipTesseractAtHyperplane(wk, scaleLamina, false);
      }
    }

    crystalGeom.setDrawRange(0, activeCrystalVertexCount);
    crystalPosAttr.needsUpdate = true;
    crystalNormAttr.needsUpdate = true;
    crystalMetaAttr.needsUpdate = true;
    crystalEdgeDist012Attr.needsUpdate = true;
    crystalEdgeDist345Attr.needsUpdate = true;

    bevelGeom.setDrawRange(0, activeBevelEdgeCount * INDICES_PER_BEVEL);
    bevelPosAttr.needsUpdate = true;
    bevelNormAttr.needsUpdate = true;

    contourLineGeom.setDrawRange(0, activeContourSegmentCount * 2);
    contourLineAttr.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Tactile Hyperplane Sliding & 4D Symmetry Cycling Kinematics
  // -------------------------------------------------------------------------
  let presetIndex = 0;
  let baseXW = SYMMETRY_PRESETS[0].xw;
  let baseYW = SYMMETRY_PRESETS[0].yw;
  let baseZW = SYMMETRY_PRESETS[0].zw;

  let angleXW = baseXW;
  let angleYW = baseYW;
  let angleZW = baseZW;
  let angleXY = SYMMETRY_PRESETS[0].xy;

  // Bounded 4D nutation offsets around the active crystallographic symmetry pole
  let tiltXW = 0.14;
  let tiltYW = -0.12;

  let sliceW0 = 0.16;
  let targetSliceW0 = 0.16;
  let autoScanActive = true;
  let autoScanPhase = 0.22;
  let modePulse = 0.0;

  let rot3DY = -0.34;
  let rot3DX = 0.22;

  let wasPointerDown = false;
  let prevNdcX = 0;
  let prevNdcY = 0;
  let downTimeSec = 0;
  let dragAccum = 0;

  updateTomographGeometry(sliceW0, angleXW, angleYW, angleZW, angleXY, -0.22);

  const baseGetState = window.__SPECIMEN_STAGE__?.getSceneState;
  if (window.__SPECIMEN_STAGE__ && baseGetState) {
    window.__SPECIMEN_STAGE__.getSceneState = () => ({
      ...baseGetState(),
      concept: '8C-slicer-tomograph',
      symmetryPreset: SYMMETRY_PRESETS[presetIndex].name,
      sliceW0: Number(sliceW0.toFixed(3)),
      activeCrystalVerts: activeCrystalVertexCount,
      activePiercedEdges: activePiercedEdgeCount,
      activeContourSegments: activeContourSegmentCount,
    });
  }

  stage.onTick(({ elapsedSec, dtSec, pointer }) => {
    if (pointer.isDown) {
      if (!wasPointerDown) {
        wasPointerDown = true;
        prevNdcX = pointer.ndc.x;
        prevNdcY = pointer.ndc.y;
        downTimeSec = elapsedSec;
        dragAccum = 0;
        autoScanActive = false;
      }

      const dx = pointer.ndc.x - prevNdcX;
      const dy = pointer.ndc.y - prevNdcY;
      prevNdcX = pointer.ndc.x;
      prevNdcY = pointer.ndc.y;
      dragAccum += Math.hypot(dx, dy);

      // Vertical drag sweeps the 4D hyperplane slice depth `w = w0`
      targetSliceW0 = Math.max(-1.22, Math.min(1.22, targetSliceW0 + dy * 1.65));
      sliceW0 = targetSliceW0;

      // Horizontal drag orbits the 3D specimen AND applies bounded 4D nutation around the symmetry pole
      tiltXW = Math.max(-0.42, Math.min(0.42, tiltXW + dx * 0.65));
      tiltYW = Math.max(-0.35, Math.min(0.35, tiltYW + dy * 0.32));
      rot3DY += dx * 1.35;
      rot3DX = Math.max(-0.42, Math.min(0.48, rot3DX - dy * 0.35));
    } else {
      if (wasPointerDown) {
        wasPointerDown = false;
        if (elapsedSec - downTimeSec < 0.26 && dragAccum < 0.06) {
          presetIndex = (presetIndex + 1) % SYMMETRY_PRESETS.length;
          const p = SYMMETRY_PRESETS[presetIndex];
          baseXW = p.xw;
          baseYW = p.yw;
          baseZW = p.zw;
          tiltXW = 0.12;
          tiltYW = -0.10;
          modePulse = 1.0;
          autoScanActive = true;
        }
      }

      if (autoScanActive) {
        autoScanPhase += dtSec * 0.56;
        targetSliceW0 = Math.sin(autoScanPhase) * 0.72;
      }

      // Gently relax 4D nutation toward a slight off-axis crystallographic angle (so polar 4D vertices frame the crystal)
      tiltXW += (0.14 - tiltXW) * (1 - Math.exp(-dtSec * 0.85));
      tiltYW += (-0.12 - tiltYW) * (1 - Math.exp(-dtSec * 0.85));
      rot3DY += dtSec * 0.14 + pointer.velocity.x * 0.0004;
    }

    const damp = 1 - Math.exp(-dtSec * 7.5);
    sliceW0 += (targetSliceW0 - sliceW0) * damp;

    const targetXW = baseXW + tiltXW + Math.sin(elapsedSec * 0.45) * 0.06;
    const targetYW = baseYW + tiltYW + Math.cos(elapsedSec * 0.38) * 0.05;
    const targetZW = baseZW;

    angleXW += (targetXW - angleXW) * damp;
    angleYW += (targetYW - angleYW) * damp;
    angleZW += (targetZW - angleZW) * damp;
    angleXY += dtSec * 0.08;

    modePulse *= Math.exp(-dtSec * 2.6);
    uSliceW0.value = sliceW0;
    uModeTransitionPulse.value = modePulse;

    tomographRoot.rotation.x += (rot3DX - tomographRoot.rotation.x) * damp;
    tomographRoot.rotation.y += (rot3DY - tomographRoot.rotation.y) * damp;

    updateTomographGeometry(sliceW0, angleXW, angleYW, angleZW, angleXY, -0.22);
  });

  return stage;
}
