import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  length,
  mix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  smoothstep,
  uniform,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import { CHALKY_SLATE_LINEAR } from '../../core/palette';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
  valueNoise3DTsl,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

/**
 * Day 08 — Tesseract (Concept 8B):
 * "The Dalí Hypercube Reliquary: 8-Cube Architectural Net Folding through R⁴"
 *
 * Mathematical & Architectural Structure:
 * - Stages the iconic 8-cube 3D unfolding net of a 4D Tesseract (Corpus Hypercubus):
 *   Central anchor cell `C0 (0,0,0)`, six face-hinged cells `C1..C6 (±X, ±Y, ±Z)`, and
 *   the lower extension tail cell `C7 (-2Y)`.
 * - Parameterizes the 4D hinge fold `foldProgress in [0, 1]` (`theta = foldProgress * PI / 2`)
 *   with gravitational/spring cascade overlap (`CASCADE_OVERLAP = 1.18`) between the primary
 *   cross arms and the double-hinged `-2Y` tail cell.
 * - In R⁴ `(x, y, z, w)`, each cell hinges around its shared 2D face from `w = -h` into `+w`,
 *   rotates under interactive 4D `(X, W)` and `(Y, W)` Givens transformations, and projects
 *   into R³ via 4D perspective (`wCam / (wCam - w)`).
 * - At `foldProgress = 0`, the 8 cells form the monumental 3D Dalí Latin Cross; at
 *   `foldProgress = 1`, all shared faces weld seamlessly in R⁴ to form the nested 8-cell
 *   Schlegel hypercube (`C0` inner cube, `C7` outer cube, `C1..C6` radial frustum cells).
 * - Each of the 8 cells is crafted as an architectural mineral reliquary cage:
 *   chamfered `boneCalcite` & `warmKaolin` structural beams, recessed `azuritePrussian`
 *   stepped filigree frames, `yellowOchre` & `rawUmber` corner gussets + 28 hinge knuckles,
 *   translucent `lapisUltramarine` & `celadonGlauconite` glass panes, and 8 suspended
 *   `cinnabarVermilion` & `realgarAmber` octahedral cores held by 64 taut tie-cables.
 */

const NUM_CELLS = 8;
const CUBE_HALF_SIZE = 0.285; // Half-width `h` of each cubic cell in 4D

// ---------------------------------------------------------------------------
// Template Box Builder in Normalized Cell Coordinates `(u, v, s) in [-1, 1]^3`
// ---------------------------------------------------------------------------
interface CellTemplateGeometryData {
  localCoords: Float32Array; // 3 floats per vertex: u, v, s in [-1, 1]
  localNormals: Float32Array; // 3 floats per vertex
  cellMeta: Float32Array; // 3 floats per vertex: x = cellId/8, y = partType (0=beam, 1=stepFrame, 2=gusset/hinge), z = edgeCoord
  indices: number[];
  vertexCount: number;
}

function appendLocalBox(
  uMin: number,
  uMax: number,
  vMin: number,
  vMax: number,
  sMin: number,
  sMax: number,
  cellId: number,
  partType: number,
  coordsOut: number[],
  normsOut: number[],
  metaOut: number[],
  indicesOut: number[]
): void {
  const baseVertex = coordsOut.length / 3;
  const cellNorm = cellId / NUM_CELLS;

  // 6 faces of the box: +U, -U, +V, -V, +S, -S
  const faces: Array<{
    n: [number, number, number];
    verts: Array<[number, number, number]>;
  }> = [
    {
      n: [1, 0, 0],
      verts: [
        [uMax, vMin, sMax],
        [uMax, vMin, sMin],
        [uMax, vMax, sMin],
        [uMax, vMax, sMax],
      ],
    },
    {
      n: [-1, 0, 0],
      verts: [
        [uMin, vMin, sMin],
        [uMin, vMin, sMax],
        [uMin, vMax, sMax],
        [uMin, vMax, sMin],
      ],
    },
    {
      n: [0, 1, 0],
      verts: [
        [uMin, vMax, sMax],
        [uMax, vMax, sMax],
        [uMax, vMax, sMin],
        [uMin, vMax, sMin],
      ],
    },
    {
      n: [0, -1, 0],
      verts: [
        [uMin, vMin, sMin],
        [uMax, vMin, sMin],
        [uMax, vMin, sMax],
        [uMin, vMin, sMax],
      ],
    },
    {
      n: [0, 0, 1],
      verts: [
        [uMin, vMin, sMax],
        [uMax, vMin, sMax],
        [uMax, vMax, sMax],
        [uMin, vMax, sMax],
      ],
    },
    {
      n: [0, 0, -1],
      verts: [
        [uMax, vMin, sMin],
        [uMin, vMin, sMin],
        [uMin, vMax, sMin],
        [uMax, vMax, sMin],
      ],
    },
  ];

  for (let f = 0; f < 6; f++) {
    const face = faces[f];
    const fBase = baseVertex + f * 4;
    for (let k = 0; k < 4; k++) {
      const [u, v, s] = face.verts[k];
      coordsOut.push(u, v, s);
      normsOut.push(face.n[0], face.n[1], face.n[2]);
      metaOut.push(cellNorm, partType, k / 3);
    }
    indicesOut.push(fBase, fBase + 1, fBase + 2, fBase, fBase + 2, fBase + 3);
  }
}

/**
 * Builds the combined template for all 8 Architectural Reliquary Cages:
 * - 12 outer Bone-Calcite structural beams per cell (`partType = 0`)
 * - 12 recessed Azurite/Kaolin stepped inner frame bars per cell (`partType = 1`)
 * - 8 Oxidized-Brass corner gusset blocks + shared seam hinge knuckles per cell (`partType = 2`)
 */
function buildReliquaryCageTemplate(): CellTemplateGeometryData {
  const coords: number[] = [];
  const norms: number[] = [];
  const meta: number[] = [];
  const indices: number[] = [];

  const bThick = 0.088; // Outer beam thickness in [-1, 1] space
  const stepInset = 0.84;
  const stepThick = 0.052;
  const gussetSize = 0.175;

  for (let c = 0; c < NUM_CELLS; c++) {
    // 1. 12 Outer Chamfered Structural Beams along the edges of [-1, 1]^3
    for (const sA of [-1, 1]) {
      for (const sB of [-1, 1]) {
        // 4 U-aligned beams
        const v0 = sA > 0 ? 1 - bThick : -1;
        const v1 = sA > 0 ? 1 : -1 + bThick;
        const s0 = sB > 0 ? 1 - bThick : -1;
        const s1 = sB > 0 ? 1 : -1 + bThick;
        appendLocalBox(
          -1 + bThick,
          1 - bThick,
          v0,
          v1,
          s0,
          s1,
          c,
          0.0,
          coords,
          norms,
          meta,
          indices
        );
        // 4 V-aligned beams
        appendLocalBox(
          v0,
          v1,
          -1 + bThick,
          1 - bThick,
          s0,
          s1,
          c,
          0.0,
          coords,
          norms,
          meta,
          indices
        );
        // 4 S-aligned beams
        appendLocalBox(
          v0,
          v1,
          s0,
          s1,
          -1 + bThick,
          1 - bThick,
          c,
          0.0,
          coords,
          norms,
          meta,
          indices
        );
      }
    }

    // 2. 12 Recessed Stepped Interior Filigree Frame Bars (`partType = 1.0`)
    for (const sA of [-1, 1]) {
      for (const sB of [-1, 1]) {
        const v0 = sA > 0 ? stepInset - stepThick : -stepInset;
        const v1 = sA > 0 ? stepInset : -stepInset + stepThick;
        const s0 = sB > 0 ? stepInset - stepThick : -stepInset;
        const s1 = sB > 0 ? stepInset : -stepInset + stepThick;
        appendLocalBox(
          -stepInset,
          stepInset,
          v0,
          v1,
          s0,
          s1,
          c,
          1.0,
          coords,
          norms,
          meta,
          indices
        );
        appendLocalBox(
          v0,
          v1,
          -stepInset,
          stepInset,
          s0,
          s1,
          c,
          1.0,
          coords,
          norms,
          meta,
          indices
        );
        appendLocalBox(
          v0,
          v1,
          s0,
          s1,
          -stepInset,
          stepInset,
          c,
          1.0,
          coords,
          norms,
          meta,
          indices
        );
      }
    }

    // 3. 8 Oxidized-Brass Corner Gusset Blocks (`partType = 2.0`)
    for (const su of [-1, 1]) {
      for (const sv of [-1, 1]) {
        for (const ss of [-1, 1]) {
          const u0 = su > 0 ? 1.015 - gussetSize : -1.015;
          const u1 = su > 0 ? 1.015 : -1.015 + gussetSize;
          const v0 = sv > 0 ? 1.015 - gussetSize : -1.015;
          const v1 = sv > 0 ? 1.015 : -1.015 + gussetSize;
          const s0 = ss > 0 ? 1.015 - gussetSize : -1.015;
          const s1 = ss > 0 ? 1.015 : -1.015 + gussetSize;
          appendLocalBox(u0, u1, v0, v1, s0, s1, c, 2.0, coords, norms, meta, indices);
        }
      }
    }

    // 4. Mechanical Hinge Knuckle Collars along the folding seams (`partType = 2.0`)
    if (c === 0) {
      // Hinge knuckles on Cell 0's 12 edges
      for (const kPos of [-0.42, 0.42]) {
        appendLocalBox(
          0.92,
          1.06,
          -1.03,
          -0.88,
          kPos - 0.09,
          kPos + 0.09,
          c,
          2.0,
          coords,
          norms,
          meta,
          indices
        );
        appendLocalBox(
          -1.06,
          -0.92,
          -1.03,
          -0.88,
          kPos - 0.09,
          kPos + 0.09,
          c,
          2.0,
          coords,
          norms,
          meta,
          indices
        );
        appendLocalBox(
          kPos - 0.09,
          kPos + 0.09,
          0.92,
          1.06,
          0.88,
          1.03,
          c,
          2.0,
          coords,
          norms,
          meta,
          indices
        );
        appendLocalBox(
          kPos - 0.09,
          kPos + 0.09,
          -1.06,
          -0.92,
          0.88,
          1.03,
          c,
          2.0,
          coords,
          norms,
          meta,
          indices
        );
      }
    }
  }

  return {
    localCoords: new Float32Array(coords),
    localNormals: new Float32Array(norms),
    cellMeta: new Float32Array(meta),
    indices,
    vertexCount: coords.length / 3,
  };
}

/**
 * Builds 48 recessed translucent reliquary window panes (6 faces * 8 cells).
 */
function buildReliquaryWindowTemplate(): CellTemplateGeometryData {
  const coords: number[] = [];
  const norms: number[] = [];
  const meta: number[] = [];
  const indices: number[] = [];

  const paneSpan = 0.78;
  const paneInset = 0.90;

  for (let c = 0; c < NUM_CELLS; c++) {
    const cellNorm = c / NUM_CELLS;
    const faces: Array<{
      n: [number, number, number];
      verts: Array<[number, number, number]>;
    }> = [
      {
        n: [1, 0, 0],
        verts: [
          [paneInset, -paneSpan, paneSpan],
          [paneInset, -paneSpan, -paneSpan],
          [paneInset, paneSpan, -paneSpan],
          [paneInset, paneSpan, paneSpan],
        ],
      },
      {
        n: [-1, 0, 0],
        verts: [
          [-paneInset, -paneSpan, -paneSpan],
          [-paneInset, -paneSpan, paneSpan],
          [-paneInset, paneSpan, paneSpan],
          [-paneInset, paneSpan, -paneSpan],
        ],
      },
      {
        n: [0, 1, 0],
        verts: [
          [-paneSpan, paneInset, paneSpan],
          [paneSpan, paneInset, paneSpan],
          [paneSpan, paneInset, -paneSpan],
          [-paneSpan, paneInset, -paneSpan],
        ],
      },
      {
        n: [0, -1, 0],
        verts: [
          [-paneSpan, -paneInset, -paneSpan],
          [paneSpan, -paneInset, -paneSpan],
          [paneSpan, -paneInset, paneSpan],
          [-paneSpan, -paneInset, paneSpan],
        ],
      },
      {
        n: [0, 0, 1],
        verts: [
          [-paneSpan, -paneSpan, paneInset],
          [paneSpan, -paneSpan, paneInset],
          [paneSpan, paneSpan, paneInset],
          [-paneSpan, paneSpan, paneInset],
        ],
      },
      {
        n: [0, 0, -1],
        verts: [
          [paneSpan, -paneSpan, -paneInset],
          [-paneSpan, -paneSpan, -paneInset],
          [-paneSpan, paneSpan, -paneInset],
          [paneSpan, paneSpan, -paneInset],
        ],
      },
    ];

    for (let f = 0; f < 6; f++) {
      const baseV = coords.length / 3;
      const face = faces[f];
      const uvs: Array<[number, number]> = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ];
      for (let k = 0; k < 4; k++) {
        coords.push(face.verts[k][0], face.verts[k][1], face.verts[k][2]);
        norms.push(face.n[0], face.n[1], face.n[2]);
        const edgeDist = Math.max(Math.abs(uvs[k][0]), Math.abs(uvs[k][1]));
        meta.push(cellNorm, edgeDist, f / 6);
      }
      indices.push(baseV, baseV + 1, baseV + 2, baseV, baseV + 2, baseV + 3);
    }
  }

  return {
    localCoords: new Float32Array(coords),
    localNormals: new Float32Array(norms),
    cellMeta: new Float32Array(meta),
    indices,
    vertexCount: coords.length / 3,
  };
}

// Pre-allocated 4D Corner & Projection Tables (Zero Per-Frame GC)
// Each cell has 8 corners indexed by `cornerIdx = (su>0?1:0) | (sv>0?2:0) | (ss>0?4:0)`
const _cellCorners3D = new Float32Array(NUM_CELLS * 8 * 3);
const _cellCornersW = new Float32Array(NUM_CELLS * 8);
const _cellCenters3D = new Float32Array(NUM_CELLS * 3);
const _cellConformal = new Float32Array(NUM_CELLS);

// 64 internal cell corner-to-core tie cables + 24 inter-cell 4D stitching cables
const TOTAL_TIE_SEGMENTS = NUM_CELLS * 8 + 24;
const _tieLinePositions = new Float32Array(TOTAL_TIE_SEGMENTS * 2 * 3);

// Pairs of `(cellA, cornerA, cellB, cornerB)` that weld together when folded into R^4
const _stitchPairs: ReadonlyArray<[number, number, number, number]> = [
  // +X (Cell 1, u=+1 outer face) welds to Cell 7 (u=+1 face)
  [1, 1, 7, 1],
  [1, 3, 7, 3],
  [1, 5, 7, 5],
  [1, 7, 7, 7],
  // -X (Cell 2, u=-1 outer face) welds to Cell 7 (u=-1 face)
  [2, 0, 7, 0],
  [2, 2, 7, 2],
  [2, 4, 7, 4],
  [2, 6, 7, 6],
  // +Y (Cell 3, v=+1 outer face) welds to Cell 7 (v=+1 face)
  [3, 2, 7, 2],
  [3, 3, 7, 3],
  [3, 6, 7, 6],
  [3, 7, 7, 7],
  // +Z (Cell 5, s=+1 outer face) welds to Cell 7 (s=+1 face)
  [5, 4, 7, 4],
  [5, 5, 7, 5],
  [5, 6, 7, 6],
  [5, 7, 7, 7],
  // -Z (Cell 6, s=-1 outer face) welds to Cell 7 (s=-1 face)
  [6, 0, 7, 0],
  [6, 1, 7, 1],
  [6, 2, 7, 2],
  [6, 3, 7, 3],
  // Lateral cross seams (+X to +Y, -X to +Y, +X to -Y, -X to -Y)
  [1, 3, 3, 3],
  [2, 2, 3, 2],
  [1, 1, 4, 1],
  [2, 0, 4, 0],
];

// ---------------------------------------------------------------------------
// Exact 4D Hinge & Perspective Projection Evaluator for `(cellId, u, v, s)`
// ---------------------------------------------------------------------------
const _pt4Out = new Float32Array(5); // x3, y3, z3, wRot, conf
function evaluateFoldedCellPoint(
  cellId: number,
  u: number,
  v: number,
  s: number,
  theta1: number,
  theta2: number,
  foldProgress: number,
  angleXW: number,
  angleYW: number,
  angleZW: number,
  wCam: number
): void {
  const h = CUBE_HALF_SIZE;
  const c1 = Math.cos(theta1);
  const s1 = Math.sin(theta1);
  const c12 = Math.cos(theta1 + theta2);
  const s12 = Math.sin(theta1 + theta2);

  let x = 0;
  let y = 0;
  let z = 0;
  let w = -h;

  switch (cellId) {
    case 0: // Central Anchor Cube (w = -h)
      x = u * h;
      y = v * h;
      z = s * h;
      w = -h;
      break;
    case 1: // +X Arm (hinges at x = +h, u = -1)
      x = h + (u + 1) * h * c1;
      y = v * h;
      z = s * h;
      w = -h + (u + 1) * h * s1;
      break;
    case 2: // -X Arm (hinges at x = -h, u = +1)
      x = -h - (1 - u) * h * c1;
      y = v * h;
      z = s * h;
      w = -h + (1 - u) * h * s1;
      break;
    case 3: // +Y Top Arm (hinges at y = +h, v = -1)
      x = u * h;
      y = h + (v + 1) * h * c1;
      z = s * h;
      w = -h + (v + 1) * h * s1;
      break;
    case 4: // -Y Lower-Middle Arm (hinges at y = -h, v = +1)
      x = u * h;
      y = -h - (1 - v) * h * c1;
      z = s * h;
      w = -h + (1 - v) * h * s1;
      break;
    case 5: // +Z Front Arm (hinges at z = +h, s = -1)
      x = u * h;
      y = v * h;
      z = h + (s + 1) * h * c1;
      w = -h + (s + 1) * h * s1;
      break;
    case 6: // -Z Back Arm (hinges at z = -h, s = +1)
      x = u * h;
      y = v * h;
      z = -h - (1 - s) * h * c1;
      w = -h + (1 - s) * h * s1;
      break;
    case 7: // -2Y Bottom Tail Cube (chained hinge on Cell 4's outer face v = +1)
      x = u * h;
      // Add smooth midpoint volume preservation `v * 0.38 * h * s12` so Cell 7 never flattens at 90 deg
      y = -h - 2 * h * c1 - (1 - v) * h * c12 + v * 0.38 * h * s12;
      z = s * h;
      w = -h + 2 * h * s1 + (1 - v) * h * s12;
      break;
  }

  // Smoothly shift vertical center from +0.26 (framing the tall 4-cube Dalí cross)
  // to 0.0 (centering the folded 4D hypercube at the origin)
  y += (1.0 - s1) * 0.28;

  // Apply 4D Givens Rotations in (X,W), (Y,W), and (Z,W)
  const cxw = Math.cos(angleXW);
  const sxw = Math.sin(angleXW);
  const x1 = x * cxw - w * sxw;
  const w1 = x * sxw + w * cxw;

  const cyw = Math.cos(angleYW);
  const syw = Math.sin(angleYW);
  const y2 = y * cyw - w1 * syw;
  const w2 = y * syw + w1 * cyw;

  const czw = Math.cos(angleZW);
  const szw = Math.sin(angleZW);
  const z3 = z * czw - w2 * szw;
  const w3 = z * szw + w2 * czw;

  // 4D Perspective Projection along W (`wCam / (wCam - w3)`)
  // Scale slightly larger when folded so the nested Schlegel hypercube fills the studio frame
  const masterScale = 1.0 + foldProgress * 0.42;
  const denom = Math.max(0.32, wCam - w3);
  const conf = (wCam / denom) * masterScale;

  _pt4Out[0] = x1 * conf;
  _pt4Out[1] = y2 * conf;
  _pt4Out[2] = z3 * conf;
  _pt4Out[3] = w3 / h; // Normalized 4D depth in [-1.5, 1.5]
  _pt4Out[4] = conf;
}

// ---------------------------------------------------------------------------
// Main Concept 8B Boot Function
// ---------------------------------------------------------------------------
export async function bootTesseractConcept2(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(8);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.16, 4.75],
    fov: 35,
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
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const verdigrisMalachite = createPigmentUniforms('verdigrisMalachite');

  const uFoldProgress = uniform(0.35);

  // -------------------------------------------------------------------------
  // TSL Node Materials
  // -------------------------------------------------------------------------

  // 1. Multi-Part Architectural Reliquary Cage Material
  // (`aCageMeta.y`: 0 = Bone Calcite Beam, 1 = Azurite/Kaolin Step Frame, 2 = Brass Gusset/Hinge)
  const cageMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.30,
    metalness: 0.25,
  });
  {
    const aCageMeta = attribute('aCageMeta', 'vec3'); // x: cellId/8, y: partType, z: w4D
    const pLoc = positionLocal;
    const pWorld = positionWorld;
    const nLoc = normalLocal;

    const fbm = mineralToothFbm(pLoc, float(5.2));
    const ptrDist = length(pWorld.sub(stage.uPointerWorld));
    const ptrGlow = exp(ptrDist.mul(ptrDist).negate().mul(1.5));

    // Part 0: Bone Calcite & Warm Kaolin Structural Beams
    const calciteBeam = kubelkaMunkGlaze(
      boneCalcite.K,
      boneCalcite.S,
      float(0.62).add(fbm.mul(0.24)),
      warmKaolin.rInf
    );

    // Part 1: Recessed Azurite Prussian & Kaolin Interior Step-Frames
    const azuriteFrame = kubelkaMunkGlaze(
      azuritePrussian.K,
      azuritePrussian.S,
      float(0.54).add(fbm.mul(0.22)),
      boneCalcite.rInf
    );

    // Part 2: Oxidized Horological Brass Corner Gussets & Hinge Knuckles
    const brassBase = kubelkaMunkGlaze(
      yellowOchre.K,
      yellowOchre.S,
      float(0.66).add(fbm.mul(0.18)),
      rawUmber.rInf
    );
    const verdigrisTouch = kubelkaMunkGlaze(
      verdigrisMalachite.K,
      verdigrisMalachite.S,
      float(0.45),
      brassBase
    );
    const brassGusset = mix(
      brassBase,
      verdigrisTouch,
      smoothstep(float(0.66), float(0.90), fbm).mul(0.28)
    );

    const isStepFrame = smoothstep(float(0.4), float(0.6), aCageMeta.y).mul(
      float(1.0).sub(smoothstep(float(1.4), float(1.6), aCageMeta.y))
    );
    const isGusset = smoothstep(float(1.4), float(1.6), aCageMeta.y);

    const baseColor = mix(mix(calciteBeam, azuriteFrame, isStepFrame), brassGusset, isGusset);

    // Subtle 4D depth patina + pointer amber warmth
    const wDepthShade = clamp(aCageMeta.z.mul(0.12), -0.12, 0.14);
    const edgeCrevice = clamp(float(1.0).sub(abs(nLoc.y)).mul(0.12), 0.0, 0.15);
    const warmHighlight = realgarAmber.rInf.mul(ptrGlow.mul(0.16).add(uFoldProgress.mul(0.06)));

    cageMat.colorNode = clamp(
      baseColor.add(wDepthShade).sub(edgeCrevice).add(warmHighlight),
      0.02,
      0.98
    );
    cageMat.roughnessNode = mix(
      mix(float(0.34), float(0.38), isStepFrame),
      float(0.22),
      isGusset
    );
    cageMat.metalnessNode = mix(
      mix(float(0.06), float(0.18), isStepFrame),
      float(0.84),
      isGusset
    );
  }

  // 2. Translucent Lapis & Celadon Beveled Reliquary Glass Panes
  const windowPaneMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    roughness: 0.14,
    metalness: 0.05,
    ior: 1.46,
  });
  {
    const aPaneMeta = attribute('aPaneMeta', 'vec3'); // x: cellId/8, y: edgeDist, z: w4D
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const fresnel = clamp(float(1.0).sub(abs(dot(nWorld, viewDir))), 0.0, 1.0);

    const fbm = mineralToothFbm(pWorld.mul(1.8), float(2.5));
    const wBlend = clamp(aPaneMeta.z.mul(0.4).add(0.5), 0.0, 1.0);

    const lapisPane = kubelkaMunkGlaze(
      lapisUltramarine.K,
      lapisUltramarine.S,
      float(0.46).add(fbm.mul(0.25)),
      boneCalcite.rInf
    );
    const celadonPane = kubelkaMunkGlaze(
      celadonGlauconite.K,
      celadonGlauconite.S,
      float(0.48).add(fbm.mul(0.22)),
      boneCalcite.rInf
    );
    const paneTint = mix(lapisPane, celadonPane, wBlend);

    const chord = float(0.18).add(fresnel.mul(0.45)).add(uFoldProgress.mul(0.15));
    const trans = beerLambertTransmittance(lapisUltramarine.extinction, chord);
    const slateBacking = vec3(
      CHALKY_SLATE_LINEAR[0],
      CHALKY_SLATE_LINEAR[1],
      CHALKY_SLATE_LINEAR[2]
    );

    windowPaneMat.colorNode = mix(slateBacking, paneTint, float(1.0).sub(trans.z.mul(0.35)));
    windowPaneMat.opacityNode = clamp(
      float(0.12).add(fresnel.mul(fresnel).mul(0.24)).add(aPaneMeta.y.mul(0.08)),
      0.10,
      0.42
    );
  }

  // 3. Suspended Cinnabar & Realgar Octahedral Reliquary Core Material
  const coreJewelMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.18,
    metalness: 0.14,
  });
  {
    const pLoc = positionLocal;
    const fbm = mineralToothFbm(pLoc, float(6.5));
    const facetNoise = valueNoise3DTsl(pLoc.mul(16.0));
    const cinnabarGlaze = kubelkaMunkGlaze(
      cinnabarVermilion.K,
      cinnabarVermilion.S,
      float(0.58).add(fbm.mul(0.28)),
      realgarAmber.rInf
    );
    const amberGlaze = kubelkaMunkGlaze(
      realgarAmber.K,
      realgarAmber.S,
      float(0.52).add(facetNoise.mul(0.25)),
      boneCalcite.rInf
    );
    const coreColor = mix(cinnabarGlaze, amberGlaze, uFoldProgress.mul(0.55));
    coreJewelMat.colorNode = coreColor;
    coreJewelMat.emissiveNode = mix(realgarAmber.rInf, cinnabarVermilion.rInf, float(0.5)).mul(
      float(0.26).add(uFoldProgress.mul(0.28))
    );
  }

  // 4. Brass Gimbal Ring Material for the 8 Reliquary Cores
  const coreRingMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.24,
    metalness: 0.85,
  });
  coreRingMat.colorNode = kubelkaMunkGlaze(
    yellowOchre.K,
    yellowOchre.S,
    float(0.68),
    burntSienna.rInf
  );

  // 5. 4D Vertex Tie-Cables & Stitching Filaments Material
  const tieLineMat = new THREE.LineBasicNodeMaterial();
  tieLineMat.colorNode = mix(yellowOchre.rInf, realgarAmber.rInf, uFoldProgress);

  // -------------------------------------------------------------------------
  // Build & Mount Scene Geometry
  // -------------------------------------------------------------------------
  const netRoot = new THREE.Group();
  netRoot.name = 'DaliHypercubeReliquaryNet';
  stage.specimenGroup.add(netRoot);

  // 1. Deformable 8-Cell Architectural Cage Mesh
  const cageTemplate = buildReliquaryCageTemplate();
  const cagePositions = new Float32Array(cageTemplate.vertexCount * 3);
  const cageNormals = new Float32Array(cageTemplate.vertexCount * 3);
  const cageMetaBuf = new Float32Array(cageTemplate.cellMeta);

  const cageGeom = new THREE.BufferGeometry();
  const cagePosAttr = new THREE.BufferAttribute(cagePositions, 3);
  cagePosAttr.setUsage(THREE.DynamicDrawUsage);
  const cageNormAttr = new THREE.BufferAttribute(cageNormals, 3);
  cageNormAttr.setUsage(THREE.DynamicDrawUsage);
  const cageMetaAttr = new THREE.BufferAttribute(cageMetaBuf, 3);
  cageMetaAttr.setUsage(THREE.DynamicDrawUsage);

  cageGeom.setAttribute('position', cagePosAttr);
  cageGeom.setAttribute('normal', cageNormAttr);
  cageGeom.setAttribute('aCageMeta', cageMetaAttr);
  cageGeom.setIndex(cageTemplate.indices);

  const cageMesh = new THREE.Mesh(cageGeom, cageMat);
  cageMesh.castShadow = true;
  cageMesh.receiveShadow = true;
  cageMesh.frustumCulled = false;
  netRoot.add(cageMesh);

  // 2. Deformable 48 Reliquary Window Panes Mesh
  const paneTemplate = buildReliquaryWindowTemplate();
  const panePositions = new Float32Array(paneTemplate.vertexCount * 3);
  const paneNormals = new Float32Array(paneTemplate.vertexCount * 3);
  const paneMetaBuf = new Float32Array(paneTemplate.cellMeta);

  const paneGeom = new THREE.BufferGeometry();
  const panePosAttr = new THREE.BufferAttribute(panePositions, 3);
  panePosAttr.setUsage(THREE.DynamicDrawUsage);
  const paneNormAttr = new THREE.BufferAttribute(paneNormals, 3);
  paneNormAttr.setUsage(THREE.DynamicDrawUsage);
  const paneMetaAttr = new THREE.BufferAttribute(paneMetaBuf, 3);
  paneMetaAttr.setUsage(THREE.DynamicDrawUsage);

  paneGeom.setAttribute('position', panePosAttr);
  paneGeom.setAttribute('normal', paneNormAttr);
  paneGeom.setAttribute('aPaneMeta', paneMetaAttr);
  paneGeom.setIndex(paneTemplate.indices);

  const paneMesh = new THREE.Mesh(paneGeom, windowPaneMat);
  paneMesh.renderOrder = 2;
  paneMesh.frustumCulled = false;
  netRoot.add(paneMesh);

  // 3. 8 Suspended Octahedral Reliquary Cores + Gimbal Rings
  const cellCoreGroups: THREE.Group[] = [];
  const coreOctaGeom = new THREE.OctahedronGeometry(0.088, 0);
  const coreRingGeom = new THREE.TorusGeometry(0.118, 0.012, 10, 28);

  for (let c = 0; c < NUM_CELLS; c++) {
    const coreGroup = new THREE.Group();
    const crystalMesh = new THREE.Mesh(coreOctaGeom, coreJewelMat);
    crystalMesh.castShadow = true;
    const gimbalMesh = new THREE.Mesh(coreRingGeom, coreRingMat);
    coreGroup.add(crystalMesh, gimbalMesh);
    netRoot.add(coreGroup);
    cellCoreGroups.push(coreGroup);
  }

  // 4. Taut 4D Vertex Tie-Cables & Cross-Cell Stitching Filaments
  const tieGeom = new THREE.BufferGeometry();
  const tiePosAttr = new THREE.BufferAttribute(_tieLinePositions, 3);
  tiePosAttr.setUsage(THREE.DynamicDrawUsage);
  tieGeom.setAttribute('position', tiePosAttr);
  const tieLinesMesh = new THREE.LineSegments(tieGeom, tieLineMat);
  tieLinesMesh.frustumCulled = false;
  netRoot.add(tieLinesMesh);

  // -------------------------------------------------------------------------
  // Trilinear Cell Deformation Helper (Zero Per-Frame Allocations)
  // -------------------------------------------------------------------------
  const _jU = new THREE.Vector3();
  const _jV = new THREE.Vector3();
  const _jS = new THREE.Vector3();
  const _nOut = new THREE.Vector3();

  function deformCellTemplateInPlace(
    template: CellTemplateGeometryData,
    posOut: Float32Array,
    normOut: Float32Array,
    metaOut: Float32Array
  ): void {
    const nVerts = template.vertexCount;
    const lCoords = template.localCoords;
    const lNorms = template.localNormals;
    const lMeta = template.cellMeta;

    for (let i = 0; i < nVerts; i++) {
      const i3 = i * 3;
      const cellId = Math.min(NUM_CELLS - 1, Math.round(lMeta[i3] * NUM_CELLS));
      const u = lCoords[i3 + 0];
      const v = lCoords[i3 + 1];
      const s = lCoords[i3 + 2];

      // Normalized [0, 1] trilinear weights across the 8 projected corners of `cellId`
      const tu = (u + 1) * 0.5;
      const tv = (v + 1) * 0.5;
      const ts = (s + 1) * 0.5;
      const mu = 1 - tu;
      const mv = 1 - tv;
      const ms = 1 - ts;

      const w000 = mu * mv * ms;
      const w100 = tu * mv * ms;
      const w010 = mu * tv * ms;
      const w110 = tu * tv * ms;
      const w001 = mu * mv * ts;
      const w101 = tu * mv * ts;
      const w011 = mu * tv * ts;
      const w111 = tu * tv * ts;

      const cBase = cellId * 24;
      const wBase = cellId * 8;

      const px =
        _cellCorners3D[cBase + 0] * w000 +
        _cellCorners3D[cBase + 3] * w100 +
        _cellCorners3D[cBase + 6] * w010 +
        _cellCorners3D[cBase + 9] * w110 +
        _cellCorners3D[cBase + 12] * w001 +
        _cellCorners3D[cBase + 15] * w101 +
        _cellCorners3D[cBase + 18] * w011 +
        _cellCorners3D[cBase + 21] * w111;

      const py =
        _cellCorners3D[cBase + 1] * w000 +
        _cellCorners3D[cBase + 4] * w100 +
        _cellCorners3D[cBase + 7] * w010 +
        _cellCorners3D[cBase + 10] * w110 +
        _cellCorners3D[cBase + 13] * w001 +
        _cellCorners3D[cBase + 16] * w101 +
        _cellCorners3D[cBase + 19] * w011 +
        _cellCorners3D[cBase + 22] * w111;

      const pz =
        _cellCorners3D[cBase + 2] * w000 +
        _cellCorners3D[cBase + 5] * w100 +
        _cellCorners3D[cBase + 8] * w010 +
        _cellCorners3D[cBase + 11] * w110 +
        _cellCorners3D[cBase + 14] * w001 +
        _cellCorners3D[cBase + 17] * w101 +
        _cellCorners3D[cBase + 20] * w011 +
        _cellCorners3D[cBase + 23] * w111;

      const pw =
        _cellCornersW[wBase + 0] * w000 +
        _cellCornersW[wBase + 1] * w100 +
        _cellCornersW[wBase + 2] * w010 +
        _cellCornersW[wBase + 3] * w110 +
        _cellCornersW[wBase + 4] * w001 +
        _cellCornersW[wBase + 5] * w101 +
        _cellCornersW[wBase + 6] * w011 +
        _cellCornersW[wBase + 7] * w111;

      posOut[i3 + 0] = px;
      posOut[i3 + 1] = py;
      posOut[i3 + 2] = pz;
      metaOut[i3 + 2] = pw;

      // Approximate local Jacobian basis from cell corners to transform normal
      _jU.set(
        _cellCorners3D[cBase + 3] - _cellCorners3D[cBase + 0],
        _cellCorners3D[cBase + 4] - _cellCorners3D[cBase + 1],
        _cellCorners3D[cBase + 5] - _cellCorners3D[cBase + 2]
      );
      _jV.set(
        _cellCorners3D[cBase + 6] - _cellCorners3D[cBase + 0],
        _cellCorners3D[cBase + 7] - _cellCorners3D[cBase + 1],
        _cellCorners3D[cBase + 8] - _cellCorners3D[cBase + 2]
      );
      _jS.set(
        _cellCorners3D[cBase + 12] - _cellCorners3D[cBase + 0],
        _cellCorners3D[cBase + 13] - _cellCorners3D[cBase + 1],
        _cellCorners3D[cBase + 14] - _cellCorners3D[cBase + 2]
      );

      const nx = lNorms[i3 + 0];
      const ny = lNorms[i3 + 1];
      const nz = lNorms[i3 + 2];

      _nOut
        .set(
          _jU.x * nx + _jV.x * ny + _jS.x * nz,
          _jU.y * nx + _jV.y * ny + _jS.y * nz,
          _jU.z * nx + _jV.z * ny + _jS.z * nz
        )
        .normalize();

      normOut[i3 + 0] = _nOut.x;
      normOut[i3 + 1] = _nOut.y;
      normOut[i3 + 2] = _nOut.z;
    }
  }

  function updateDaliNetGeometry(
    foldProgress: number,
    angleXW: number,
    angleYW: number,
    angleZW: number,
    elapsedSec: number
  ): void {
    // Cascade overlap between primary 6 cross arms (`theta1`) and lower `-2Y` tail (`theta2`)
    const p1 = Math.max(0, Math.min(1, foldProgress * 1.08));
    const p2 = Math.max(0, Math.min(1, (foldProgress - 0.06) / 0.94));
    const theta1 = p1 * (Math.PI * 0.5);
    const theta2 = p2 * (Math.PI * 0.5);
    const wCam = 1.48;

    // 1. Evaluate 8 corners + center of each of the 8 cubic cells in R^4 -> R^3
    for (let c = 0; c < NUM_CELLS; c++) {
      const cBase = c * 24;
      const wBase = c * 8;
      for (let corner = 0; corner < 8; corner++) {
        const su = corner & 1 ? 1 : -1;
        const sv = corner & 2 ? 1 : -1;
        const ss = corner & 4 ? 1 : -1;

        evaluateFoldedCellPoint(
          c,
          su,
          sv,
          ss,
          theta1,
          theta2,
          foldProgress,
          angleXW,
          angleYW,
          angleZW,
          wCam
        );
        _cellCorners3D[cBase + corner * 3 + 0] = _pt4Out[0];
        _cellCorners3D[cBase + corner * 3 + 1] = _pt4Out[1];
        _cellCorners3D[cBase + corner * 3 + 2] = _pt4Out[2];
        _cellCornersW[wBase + corner] = _pt4Out[3];
      }

      // Cell center `(0, 0, 0)` for the suspended octahedral reliquary core
      evaluateFoldedCellPoint(
        c,
        0,
        0,
        0,
        theta1,
        theta2,
        foldProgress,
        angleXW,
        angleYW,
        angleZW,
        wCam
      );
      _cellCenters3D[c * 3 + 0] = _pt4Out[0];
      _cellCenters3D[c * 3 + 1] = _pt4Out[1];
      _cellCenters3D[c * 3 + 2] = _pt4Out[2];
      _cellConformal[c] = _pt4Out[4];

      const coreGroup = cellCoreGroups[c];
      coreGroup.position.set(_pt4Out[0], _pt4Out[1], _pt4Out[2]);
      const coreScale = Math.max(0.45, Math.min(1.65, _pt4Out[4] * 0.85));
      coreGroup.scale.setScalar(coreScale);
      coreGroup.rotation.set(
        elapsedSec * 0.7 + c * 0.4,
        elapsedSec * 1.1 - c * 0.3,
        foldProgress * Math.PI * 0.5
      );
    }

    // 2. Deform the 8 Architectural Cages & 48 Window Panes in-place
    deformCellTemplateInPlace(cageTemplate, cagePositions, cageNormals, cageMetaBuf);
    cagePosAttr.needsUpdate = true;
    cageNormAttr.needsUpdate = true;
    cageMetaAttr.needsUpdate = true;

    deformCellTemplateInPlace(paneTemplate, panePositions, paneNormals, paneMetaBuf);
    panePosAttr.needsUpdate = true;
    paneNormAttr.needsUpdate = true;
    paneMetaAttr.needsUpdate = true;

    // 3. Update the 64 Internal Corner-to-Core Tie Cables + 24 Cross-Cell 4D Stitching Filaments
    let segIdx = 0;
    for (let c = 0; c < NUM_CELLS; c++) {
      const cx = _cellCenters3D[c * 3 + 0];
      const cy = _cellCenters3D[c * 3 + 1];
      const cz = _cellCenters3D[c * 3 + 2];
      const cBase = c * 24;
      for (let corner = 0; corner < 8; corner++) {
        const pIdx = segIdx * 6;
        _tieLinePositions[pIdx + 0] = cx;
        _tieLinePositions[pIdx + 1] = cy;
        _tieLinePositions[pIdx + 2] = cz;
        _tieLinePositions[pIdx + 3] = _cellCorners3D[cBase + corner * 3 + 0];
        _tieLinePositions[pIdx + 4] = _cellCorners3D[cBase + corner * 3 + 1];
        _tieLinePositions[pIdx + 5] = _cellCorners3D[cBase + corner * 3 + 2];
        segIdx++;
      }
    }

    for (let s = 0; s < _stitchPairs.length; s++) {
      const [cA, corA, cB, corB] = _stitchPairs[s];
      const aBase = cA * 24 + corA * 3;
      const bBase = cB * 24 + corB * 3;
      const pIdx = segIdx * 6;
      _tieLinePositions[pIdx + 0] = _cellCorners3D[aBase + 0];
      _tieLinePositions[pIdx + 1] = _cellCorners3D[aBase + 1];
      _tieLinePositions[pIdx + 2] = _cellCorners3D[aBase + 2];
      _tieLinePositions[pIdx + 3] = _cellCorners3D[bBase + 0];
      _tieLinePositions[pIdx + 4] = _cellCorners3D[bBase + 1];
      _tieLinePositions[pIdx + 5] = _cellCorners3D[bBase + 2];
      segIdx++;
    }
    tiePosAttr.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Tactile 4D Folding & Gravitational Cascade Kinematics
  // (Smooth monotonic drag deltas; honors drag direction on release)
  // -------------------------------------------------------------------------
  let foldProgress = 0.22; // Start in the open, recognizable 8-cube Dalí cross beginning to hinge into R^4
  let targetFold = 0.22;
  let foldVelocity = 0.0;
  let autoCycleActive = true;
  let autoCyclePhase = -0.82;

  let angleXW = 0.22;
  let angleYW = 0.16;
  let rot3DY = -0.34;
  let rot3DX = 0.18;

  let wasPointerDown = false;
  let prevNdcX = 0;
  let prevNdcY = 0;
  let downTimeSec = 0;
  let dragAccum = 0;
  let lastDragFoldDelta = 0;

  netRoot.rotation.set(rot3DX, rot3DY, 0);
  updateDaliNetGeometry(foldProgress, angleXW, angleYW, 0.08, 0);

  const baseGetState = window.__SPECIMEN_STAGE__?.getSceneState;
  if (window.__SPECIMEN_STAGE__ && baseGetState) {
    window.__SPECIMEN_STAGE__.getSceneState = () => ({
      ...baseGetState(),
      concept: '8B-unfolding-net',
      foldProgress: Number(foldProgress.toFixed(3)),
      angleXW: Number(angleXW.toFixed(3)),
      angleYW: Number(angleYW.toFixed(3)),
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
        lastDragFoldDelta = 0;
        autoCycleActive = false;
      }

      const dx = pointer.ndc.x - prevNdcX;
      const dy = pointer.ndc.y - prevNdcY;
      prevNdcX = pointer.ndc.x;
      prevNdcY = pointer.ndc.y;
      dragAccum += Math.hypot(dx, dy);

      // Drag Y (and diagonal X) folds/unfolds the 8-cube net through R^4;
      // Drag X rotates both the 3D viewing angle and 4D (X,W) perspective
      const foldDelta = dy * 1.15 + dx * 0.45;
      if (Math.abs(foldDelta) > 1e-4) {
        lastDragFoldDelta = foldDelta;
      }
      foldProgress = Math.max(0.0, Math.min(1.0, foldProgress + foldDelta));
      targetFold = foldProgress;

      rot3DY += dx * 1.65;
      rot3DX = Math.max(-0.52, Math.min(0.58, rot3DX - dy * 0.55));
      angleXW += dx * 1.35;
      angleYW += dy * 0.85;
    } else {
      if (wasPointerDown) {
        wasPointerDown = false;
        if (elapsedSec - downTimeSec < 0.26 && dragAccum < 0.06) {
          // Click toggles between open Dalí Cross (0.0) and locked 4D Hypercube (1.0)
          targetFold = foldProgress < 0.52 ? 1.0 : 0.0;
          foldVelocity = targetFold > foldProgress ? 0.65 : -0.65;
        } else if (Math.abs(lastDragFoldDelta) > 0.004) {
          // Honor user's drag direction on release if near a transition
          if (lastDragFoldDelta > 0 && foldProgress > 0.62) targetFold = 1.0;
          else if (lastDragFoldDelta < 0 && foldProgress < 0.38) targetFold = 0.0;
        }
      }

      if (autoCycleActive) {
        // Gentle gravitational breathing between the Dalí cross and the folded 4D hypercube
        autoCyclePhase += dtSec * 0.48;
        const rawWave = Math.sin(autoCyclePhase) * 0.5 + 0.5;
        // Power curve acceleration (`gamma = 1.48`, zero robotic smoothstep pause)
        const num = Math.pow(rawWave, 1.48);
        const den = num + Math.pow(1 - rawWave, 1.48);
        targetFold = 0.06 + (num / Math.max(1e-5, den)) * 0.90;
      }

      // Gravitational spring-damper integration with subtle contact rebound at endpoints
      const springErr = targetFold - foldProgress;
      foldVelocity += springErr * 14.0 * dtSec;
      foldVelocity *= Math.exp(-dtSec * 4.2);
      foldProgress += foldVelocity * dtSec;

      if (foldProgress < 0.0) {
        foldProgress = 0.0;
        foldVelocity = -foldVelocity * 0.28;
      } else if (foldProgress > 1.0) {
        foldProgress = 1.0;
        foldVelocity = -foldVelocity * 0.28;
      }

      // Gentle 4D rotation when folded so the nested inner/outer cells revolve in R^4
      const foldedWeight = 0.25 + foldProgress * 0.75;
      angleXW += dtSec * 0.28 * foldedWeight + pointer.velocity.x * 0.0004;
      angleYW = 0.18 + Math.sin(elapsedSec * 0.42) * 0.22 * foldedWeight;
      rot3DY += dtSec * 0.14 + pointer.velocity.x * 0.0005;
    }

    uFoldProgress.value = foldProgress;

    const damp = 1 - Math.exp(-dtSec * 8.0);
    netRoot.rotation.x += (rot3DX - netRoot.rotation.x) * damp;
    netRoot.rotation.y += (rot3DY - netRoot.rotation.y) * damp;

    const angleZW = Math.sin(elapsedSec * 0.35) * 0.14 * foldProgress;
    updateDaliNetGeometry(foldProgress, angleXW, angleYW, angleZW, elapsedSec);
  });

  return stage;
}
