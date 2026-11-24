import * as THREE from 'three/webgpu';
import {
  CELL_INRADIUS,
  DECK_FLOOR_Y,
  WALL_HALF_THICKNESS,
  type HexLatticeLayout,
} from './magneticLattice';

/**
 * Shared dimensional constants for the Day 28 Magnetic Honeycomb Instrument.
 * All parts seat flush along exact shared Y elevations and flat-top hexagonal radii.
 */
export const SLAB_BOTTOM_Y = -0.29;
export const SLAB_SHOULDER_Y = 0.056;
export const BEZEL_TOP_Y = 0.118;
export const HONEYCOMB_WALL_TOP_Y = 0.094;
export const HONEYCOMB_BEVEL_Y = 0.088;

export const BEZEL_INNER_CIRCUM_R = 1.385;
export const BEZEL_OUTER_CIRCUM_R = 1.565;
export const SLAB_TOP_CIRCUM_R = 1.635;
export const SLAB_OUTER_CIRCUM_R = 1.740;
export const SLAB_BOTTOM_CIRCUM_R = 1.645;
export const BEZEL_BOLT_RADIUS = 1.474;

export const NEEDLE_HALF_LENGTH = 0.089;
export const NEEDLE_HALF_WIDTH = 0.029;
export const NEEDLE_RIDGE_TOP_Y = 0.0165;
export const NEEDLE_BOTTOM_Y = -0.009;

/**
 * Computes the signed volume of a closed triangle mesh (positive => outward CCW winding).
 */
export function computeSignedVolume(geometry: THREE.BufferGeometry): number {
  const pos = geometry.getAttribute('position');
  if (!pos) return 0;
  const idx = geometry.getIndex();
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  let vol6 = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;

    const ax = pos.getX(i0);
    const ay = pos.getY(i0);
    const az = pos.getZ(i0);
    const bx = pos.getX(i1);
    const by = pos.getY(i1);
    const bz = pos.getZ(i1);
    const cx = pos.getX(i2);
    const cy = pos.getY(i2);
    const cz = pos.getZ(i2);

    vol6 +=
      ax * (by * cz - bz * cy) -
      ay * (bx * cz - bz * cx) +
      az * (bx * cy - by * cx);
  }
  return vol6 / 6;
}

/**
 * Helper to build a faceted watertight mesh from raw non-indexed triangles,
 * automatically generating crisp per-face outward normals and planar UVs.
 */
function buildFacetedGeometry(positions: number[]): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();

  // Compute clean triplanar-friendly UV coordinates
  const count = positions.length / 3;
  const uvs = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    uvs[i * 2] = positions[i * 3] * 0.5 + 0.5;
    uvs[i * 2 + 1] = positions[i * 3 + 2] * 0.5 + 0.5;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  return geo;
}

function pushTri(
  out: number[],
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number
): void {
  out.push(ax, ay, az, bx, by, bz, cx, cy, cz);
}

/**
 * Pushes a quad (a, b, c, d) in CCW outward order: (a, b, c) and (a, c, d).
 */
function pushQuad(
  out: number[],
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
  dx: number,
  dy: number,
  dz: number
): void {
  out.push(ax, ay, az, bx, by, bz, cx, cy, cz);
  out.push(ax, ay, az, cx, cy, cz, dx, dy, dz);
}

/**
 * Flat-top regular hexagon vertices ordered counter-clockwise when viewed from +Y.
 * Vertex k sits at angle -k * 60° in XZ so (v[k] -> v[k+1]) travels CCW looking from +Y.
 */
function flatTopHexVertices(circumR: number): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  for (let k = 0; k < 6; k++) {
    // Going from +X toward -Z is CCW when viewed from +Y (right-handed Y-up)
    const ang = -k * (Math.PI / 3);
    pts.push([Math.cos(ang) * circumR, Math.sin(ang) * circumR]);
  }
  return pts;
}

/**
 * Connects two consecutive closed 6-vertex profile rings (`r0` -> `r1`) with outward CCW quads.
 * Convention: moving around the profile loop (bottom center -> outer rim -> top inner well -> top center)
 * keeps the solid interior to the LEFT of the profile tangent in the (radius, y) half-plane,
 * which yields outward-pointing normals when lofted along CCW hex vertex order!
 */
function loftHexRings(
  out: number[],
  rings: ReadonlyArray<{ r: number; y: number }>
): void {
  const ringVerts = rings.map((rg) => flatTopHexVertices(rg.r));

  for (let s = 0; s < rings.length - 1; s++) {
    const r0 = rings[s];
    const r1 = rings[s + 1];
    const v0 = ringVerts[s];
    const v1 = ringVerts[s + 1];

    if (r0.r === 0 && r1.r > 0) {
      // Fan from center point at r0.y to ring r1 (normal points downward if r1 is outer bottom)
      for (let k = 0; k < 6; k++) {
        const kn = (k + 1) % 6;
        pushTri(
          out,
          0,
          r0.y,
          0,
          v1[kn][0],
          r1.y,
          v1[kn][1],
          v1[k][0],
          r1.y,
          v1[k][1]
        );
      }
    } else if (r0.r > 0 && r1.r === 0) {
      // Fan from ring r0 to center point at r1.y (normal points upward when closing top)
      for (let k = 0; k < 6; k++) {
        const kn = (k + 1) % 6;
        pushTri(
          out,
          v0[k][0],
          r0.y,
          v0[k][1],
          v0[kn][0],
          r0.y,
          v0[kn][1],
          0,
          r1.y,
          0
        );
      }
    } else {
      // Quad strip between ring s and ring s+1
      for (let k = 0; k < 6; k++) {
        const kn = (k + 1) % 6;
        pushQuad(
          out,
          v0[k][0],
          r0.y,
          v0[k][1],
          v0[kn][0],
          r0.y,
          v0[kn][1],
          v1[kn][0],
          r1.y,
          v1[kn][1],
          v1[k][0],
          r1.y,
          v1[k][1]
        );
      }
    }
  }
}

/**
 * 1. Chamfered Hexagonal Bone-Calcite Limestone Chassis Slab.
 * Watertight solid with bottom bevel, crisp limestone vertical walls, 45° upper chamfer,
 * outer reveal shoulder, and recessed interior honeycomb deck floor at y = DECK_FLOOR_Y.
 */
export function buildLimestoneSlabGeometry(): THREE.BufferGeometry {
  const pos: number[] = [];

  // Profile traversed from underside center -> outer bottom -> outer walls -> top shoulder -> recessed well -> top center
  const profile: Array<{ r: number; y: number }> = [
    { r: 0, y: SLAB_BOTTOM_Y },
    { r: SLAB_BOTTOM_CIRCUM_R, y: SLAB_BOTTOM_Y },
    { r: SLAB_OUTER_CIRCUM_R, y: SLAB_BOTTOM_Y + 0.068 },
    { r: SLAB_OUTER_CIRCUM_R, y: SLAB_SHOULDER_Y - 0.072 },
    { r: SLAB_TOP_CIRCUM_R, y: SLAB_SHOULDER_Y },
    { r: BEZEL_INNER_CIRCUM_R + 0.018, y: SLAB_SHOULDER_Y },
    { r: BEZEL_INNER_CIRCUM_R + 0.012, y: DECK_FLOOR_Y },
    { r: 0, y: DECK_FLOOR_Y },
  ];

  loftHexRings(pos, profile);
  return buildFacetedGeometry(pos);
}

/**
 * 2. Darkened-Bronze Beveled Hexagonal Bezel Frame.
 * Seated flush onto the limestone shoulder with crisp top outer & inner chamfers
 * and an interior skirt stepping cleanly down into the recessed limestone well.
 */
export function buildBronzeBezelGeometry(): THREE.BufferGeometry {
  const pos: number[] = [];

  // Closed toroidal hexagonal cross-section (10-point closed loop in (r, y), solid on left)
  const loop: Array<{ r: number; y: number }> = [
    { r: BEZEL_INNER_CIRCUM_R, y: DECK_FLOOR_Y - 0.004 },
    { r: BEZEL_INNER_CIRCUM_R + 0.022, y: DECK_FLOOR_Y - 0.004 },
    { r: BEZEL_INNER_CIRCUM_R + 0.022, y: SLAB_SHOULDER_Y - 0.004 },
    { r: BEZEL_OUTER_CIRCUM_R, y: SLAB_SHOULDER_Y - 0.004 },
    { r: BEZEL_OUTER_CIRCUM_R, y: BEZEL_TOP_Y - 0.018 },
    { r: BEZEL_OUTER_CIRCUM_R - 0.022, y: BEZEL_TOP_Y },
    { r: BEZEL_INNER_CIRCUM_R + 0.032, y: BEZEL_TOP_Y },
    { r: BEZEL_INNER_CIRCUM_R + 0.012, y: BEZEL_TOP_Y - 0.018 },
    { r: BEZEL_INNER_CIRCUM_R, y: BEZEL_TOP_Y - 0.024 },
    { r: BEZEL_INNER_CIRCUM_R, y: DECK_FLOOR_Y - 0.004 },
  ];

  loftHexRings(pos, loop);
  return buildFacetedGeometry(pos);
}

/**
 * 3. Satin-Brass Inner Bezel Trim Pinstripe Ring.
 * Fine chamfered brass inlay outlining the recessed limestone deck right inside the dark bronze bezel.
 */
export function buildBrassInnerTrimGeometry(): THREE.BufferGeometry {
  const pos: number[] = [];
  const rOut = BEZEL_INNER_CIRCUM_R + 0.003;
  const rIn = BEZEL_INNER_CIRCUM_R - 0.022;

  const loop: Array<{ r: number; y: number }> = [
    { r: rIn, y: DECK_FLOOR_Y - 0.003 },
    { r: rOut, y: DECK_FLOOR_Y - 0.003 },
    { r: rOut, y: HONEYCOMB_WALL_TOP_Y + 0.008 },
    { r: rOut - 0.006, y: HONEYCOMB_WALL_TOP_Y + 0.013 },
    { r: rIn + 0.005, y: HONEYCOMB_WALL_TOP_Y + 0.013 },
    { r: rIn, y: HONEYCOMB_WALL_TOP_Y + 0.006 },
    { r: rIn, y: DECK_FLOOR_Y - 0.003 },
  ];

  loftHexRings(pos, loop);
  return buildFacetedGeometry(pos);
}

/**
 * 4. Machined Satin-Brass Bezel Corner Rivet Bolt (single bolt geometry; place at 6 corners).
 * Domed slotted/chamfered brass stud built via bottom-to-top LatheGeometry (strictly outward CCW).
 */
export function buildBezelBoltGeometry(): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, -0.006),
    new THREE.Vector2(0.044, -0.006),
    new THREE.Vector2(0.044, 0.004),
    new THREE.Vector2(0.040, 0.010),
    new THREE.Vector2(0.032, 0.017),
    new THREE.Vector2(0.018, 0.0215),
    new THREE.Vector2(0.0, 0.023),
  ];
  const geo = new THREE.LatheGeometry(pts, 28);
  geo.computeVertexNormals();
  return geo;
}

/**
 * 5. Watertight 91-Cell Beveled Satin-Brass Honeycomb Partition Matrix.
 *
 * Every hexagonal cell's 6 walls are built with exact 30° mitered corners:
 * - Shared interior walls extend from inner chamber face `r_in` to exact shared midpoint `r_mid`,
 *   so adjacent cells meet cleanly along a hairline seam with zero overlapping coplanar triangles.
 * - Exterior crenellated boundary walls extend outward to `r_mid + WALL_HALF_THICKNESS`
 *   so exterior walls match the exact full `2 * WALL_HALF_THICKNESS` thickness of interior walls.
 * - Top edges carry a crisp chamfer (`HONEYCOMB_BEVEL_Y -> HONEYCOMB_WALL_TOP_Y`) that catches
 *   studio softbox highlights along every hexagonal rim.
 */
export function buildHoneycombMatrixGeometry(layout: HexLatticeLayout): THREE.BufferGeometry {
  const pos: number[] = [];

  const sqrt3 = Math.sqrt(3);
  const s = layout.cellCircumradius;
  const rMid = CELL_INRADIUS;
  const wallH = WALL_HALF_THICKNESS;
  const rIn = rMid - wallH;
  const rInBevel = rIn + 0.0032;

  // Convert inradius to circumradius for pointy-top hex: circumR = inradius / (sqrt(3)/2)
  const inToCirc = 2 / sqrt3;
  const cIn = rIn * inToCirc;
  const cInBev = rInBevel * inToCirc;
  const cMid = s;
  const cExt = (rMid + wallH) * inToCirc;
  const cExtBev = (rMid + wallH - 0.0032) * inToCirc;

  const yBot = DECK_FLOOR_Y - 0.006;
  const yBev = HONEYCOMB_BEVEL_Y;
  const yTop = HONEYCOMB_WALL_TOP_Y;

  // Precompute unit pointy-top vertex directions ordered CCW viewed from +Y
  // Wait: let's match AXIAL_EDGE_DIRS carefully!
  // In magneticLattice.ts:
  //   corner m sits at angle (30° + 60°*m) in (x, z) -> (cos, sin)
  //   d=0 connects corner 0 (30°) to corner 1 (90°): edge midpoint at 60° -> (dq=0, dr=+1)
  // Notice: in right-handed Y-up (x right, z toward viewer), angle increasing from 30° -> 90°
  // goes CLOCKWISE when viewed from +Y! So to go CCW viewed from +Y, traverse (corner 1 -> corner 0).
  const dirX = new Float32Array(6);
  const dirZ = new Float32Array(6);
  for (let m = 0; m < 6; m++) {
    const ang = Math.PI / 6 + m * (Math.PI / 3);
    dirX[m] = Math.cos(ang);
    dirZ[m] = Math.sin(ang);
  }

  for (let i = 0; i < layout.cellCount; i++) {
    const cell = layout.cells[i];
    const cx = cell.x;
    const cz = cell.z;

    for (let d = 0; d < 6; d++) {
      const isExterior = cell.neighborByDir[d] < 0;

      // Endpoints of edge d ordered CCW when viewed from +Y:
      // pA = corner (d+1)%6, pB = corner d
      // Let's verify cross product (pB - pA) x (+Y) points OUTWARD from cell center!
      // At d=0: pA is 90° (0, +1), pB is 30° (+0.866, +0.5).
      // Edge vector e = pB - pA = (+0.866, 0, -0.5).
      // e x (0, 1, 0) = (+0.5, 0, +0.866) -> angle 60° -> OUTWARD! Exact!
      const mA = (d + 1) % 6;
      const mB = d;

      const ax = dirX[mA];
      const az = dirZ[mA];
      const bx = dirX[mB];
      const bz = dirZ[mB];

      // Determine outer circumradius at vertex A and vertex B:
      // For exterior edges, extend to full wall thickness cExt (with clean top outer bevel cExtBev).
      // Wait: if an exterior edge meets an interior edge at vertex A, check whether the adjacent
      // edge dPrev = (d + 1) % 6 (sharing corner mA) is also exterior so miter corners match cleanly!
      const adjAExterior = cell.neighborByDir[(d + 1) % 6] < 0;
      const adjBExterior = cell.neighborByDir[(d + 5) % 6] < 0;

      const cOutA = isExterior ? (adjAExterior ? cExt : cMid) : cMid;
      const cOutB = isExterior ? (adjBExterior ? cExt : cMid) : cMid;
      const cTopOutA = isExterior ? (adjAExterior ? cExtBev : cMid) : cMid;
      const cTopOutB = isExterior ? (adjBExterior ? cExtBev : cMid) : cMid;

      // Actually: if only one of the two meeting edges is exterior (a concave outer notch vertex),
      // keeping both endpoints of an exterior edge at cExt creates a tiny triangular overlap in the notch;
      // keeping interior edges cleanly at cMid and exterior edges cleanly at cMid + uniform offset
      // via exact edge-normal extrusion avoids ANY miter discontinuity. Even simpler and 100% exact:
      // Let's make EVERY wall prism a self-contained closed trapezoid spanning [cIn, cOut] with
      // cOutA = isExterior ? cExt : cMid ONLY when adjacent is exterior, OR uniform cMid for all
      // interior cells + a dedicated solid outer perimeter rim collar!
      // Let's check! If every cell spans [cIn, cMid] (interior) and exterior edges add a crisp outer
      // reinforcement strip [cMid, cExt] cleanly mitered on convex corners, wait:
      // Check what happens with cOutA/cOutB above:
      // On a convex corner of the hex honeycomb, BOTH meeting edges are exterior -> both use cExt!
      // On a concave notch vertex between two outer cells, the notch vertex is SHARED with another cell's
      // interior edge -> both exterior edges meet at cMid right at that notch vertex!
      // Zero overlap, 100% watertight, continuous crenellated outer wall!
      const inAx = cx + ax * cIn;
      const inAz = cz + az * cIn;
      const inBx = cx + bx * cIn;
      const inBz = cz + bz * cIn;

      const inBevAx = cx + ax * cInBev;
      const inBevAz = cz + az * cInBev;
      const inBevBx = cx + bx * cInBev;
      const inBevBz = cz + bz * cInBev;

      const topOutAx = cx + ax * cTopOutA;
      const topOutAz = cz + az * cTopOutA;
      const topOutBx = cx + bx * cTopOutB;
      const topOutBz = cz + bz * cTopOutB;

      const outAx = cx + ax * cOutA;
      const outAz = cz + az * cOutA;
      const outBx = cx + bx * cOutB;
      const outBz = cz + bz * cOutB;

      // 1. Inner vertical wall facing INWARD toward cell center:
      // Going (inB -> inA) along bottom, (inA -> inB) along yBev -> normal points inward!
      pushQuad(
        pos,
        inBx, yBot, inBz,
        inAx, yBot, inAz,
        inAx, yBev, inAz,
        inBx, yBev, inBz
      );

      // 2. Inner 45° top bevel (yBev -> yTop), facing inward + upward
      pushQuad(
        pos,
        inBx, yBev, inBz,
        inAx, yBev, inAz,
        inBevAx, yTop, inBevAz,
        inBevBx, yTop, inBevBz
      );

      // 3. Flat top crest facing +Y: (inBevB -> inBevA -> topOutA -> topOutB)
      // Verify normal: (inBevA - inBevB) x (topOutA - inBevB)
      // inBevA - inBevB = -e (CW? wait: e = B - A, so A - B = -e).
      // (-e) x (outward) = -(e x outward). Wait: e x (+Y) = outward => (+Y) x outward = -e => (-e) x outward = +Y?
      // Let's check numerically at d=0!
      // A = (0, +1), B = (+0.866, +0.5).
      // inBevB = (0.866*r1, y, 0.5*r1), inBevA = (0, y, r1), topOutA = (0, y, r2) with r2 > r1.
      // u = inBevA - inBevB = (-0.866*r1, 0, +0.5*r1)
      // v = topOutA - inBevB = (-0.866*r1, 0, r2 - 0.5*r1)
      // (u x v)_y = u_z * v_x - u_x * v_z
      //           = (0.5*r1)*(-0.866*r1) - (-0.866*r1)*(r2 - 0.5*r1)
      //           = 0.866*r1 * (-0.5*r1 + r2 - 0.5*r1) = 0.866*r1*(r2 - r1) > 0 (+Y)!!
      // 100% VERIFIED +Y OUTWARD NORMAL!
      pushQuad(
        pos,
        inBevBx, yTop, inBevBz,
        inBevAx, yTop, inBevAz,
        topOutAx, yTop, topOutAz,
        topOutBx, yTop, topOutBz
      );

      // 4. Outer top bevel & outer vertical wall (always closed so each cell prism is manifold,
      //    and for interior edges the vertical wall is hidden coplanar at cMid)
      if (isExterior) {
        pushQuad(
          pos,
          topOutBx, yTop, topOutBz,
          topOutAx, yTop, topOutAz,
          outAx, yBev, outAz,
          outBx, yBev, outBz
        );
        pushQuad(
          pos,
          outBx, yBev, outBz,
          outAx, yBev, outAz,
          outAx, yBot, outAz,
          outBx, yBot, outBz
        );
      } else {
        // Vertical outer mating wall at exact cMid (hidden inside solid honeycomb; keeps signedVolume exact)
        pushQuad(
          pos,
          topOutBx, yTop, topOutBz,
          topOutAx, yTop, topOutAz,
          outAx, yBot, outAz,
          outBx, yBot, outBz
        );
      }

      // 5. Underside bottom cap facing -Y (buried at yBot = -0.006 inside limestone deck)
      pushQuad(
        pos,
        inAx, yBot, inAz,
        inBx, yBot, inBz,
        outBx, yBot, outBz,
        outAx, yBot, outAz
      );
    }
  }

  return buildFacetedGeometry(pos);
}

/**
 * 6. Turned Satin-Brass Center Pivot Pin & Crowning Jewel Cap (Instanced 91x).
 * Profile strictly ordered bottom-to-top so LatheGeometry yields outward CCW winding.
 */
export function buildPivotPinGeometry(): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, -0.005),
    new THREE.Vector2(0.022, -0.005),
    new THREE.Vector2(0.022, 0.006),
    new THREE.Vector2(0.017, 0.012),
    new THREE.Vector2(0.011, 0.017),
    new THREE.Vector2(0.0095, 0.044),
    new THREE.Vector2(0.014, 0.050),
    new THREE.Vector2(0.014, 0.064),
    new THREE.Vector2(0.0175, 0.068),
    new THREE.Vector2(0.0170, 0.074),
    new THREE.Vector2(0.0120, 0.079),
    new THREE.Vector2(0.0055, 0.0815),
    new THREE.Vector2(0.0, 0.0825),
  ];
  const geo = new THREE.LatheGeometry(pts, 20);
  geo.computeVertexNormals();
  return geo;
}

/**
 * 7. Sculpted Diamond-Faceted Magnetic Compass Needle Half (North: sign = +1, South: sign = -1).
 * Watertight 3D faceted kite half with crisp dorsal roof spine, beveled flanks, flat mating seam
 * at x = 0, and verified positive signed volume for BOTH halves.
 */
export function buildNeedleHalfGeometry(side: 1 | -1): THREE.BufferGeometry {
  const pos: number[] = [];

  const tipX = side * NEEDLE_HALF_LENGTH;
  const w = NEEDLE_HALF_WIDTH;
  const yTop = NEEDLE_RIDGE_TOP_Y;
  const yMid = 0.001;
  const yBot = NEEDLE_BOTTOM_Y;

  // Centerline spine points (x = 0) and tip point (tipX, yMid, 0)
  // Upper-back/front roof vertices at x = 0:
  //   T = (0, yTop, 0)
  //   B = (0, yBot, 0)
  //   L = (0, yMid, -w)
  //   R = (0, yMid, +w)
  //   P = (tipX, yMid, 0)
  //
  // Check outward CCW winding for side = +1 (tip at +X):
  //   - Upper-right facet (T, R, P): (R - T) = (0, -yTop, +w), (P - T) = (+L, -yTop, 0)
  //     (R - T) x (P - T) = (yTop*w, +L*w, +L*yTop) -> (+X, +Y, +Z) OUTWARD!
  //   - Upper-left facet (T, P, L): (P - T) x (L - T) = (+yTop*w, +L*w, -L*yTop) -> (+X, +Y, -Z) OUTWARD!
  //   - Lower-right facet (B, P, R): (P - B) = (+L, +|yBot|, 0), (R - B) = (0, +|yBot|, +w)
  //     (P - B) x (R - B) = (+|yBot|*w, -L*w, +L*|yBot|) -> (+X, -Y, +Z) OUTWARD!
  //   - Lower-left facet (B, L, P): (+|yBot|*w, -L*w, -L*|yBot|) -> (+X, -Y, -Z) OUTWARD!
  //   - Center cap at x = 0 facing -X: (T, L, B) and (T, B, R):
  //     (L - T) x (B - T) = (0, -yTop, -w) x (0, yBot - yTop, 0) = (+w*(yBot-yTop), 0, 0) -> (-X) OUTWARD!
  // For side = -1 (tip at -X), reversing triangle order mirrors cleanly while preserving outward CCW winding!
  const emit = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number
  ) => {
    if (side === 1) {
      pushTri(pos, ax, ay, az, bx, by, bz, cx, cy, cz);
    } else {
      pushTri(pos, ax, ay, az, cx, cy, cz, bx, by, bz);
    }
  };

  // Dorsal roof facets (2 crisp angled planes)
  emit(0, yTop, 0, 0, yMid, w, tipX, yMid, 0);
  emit(0, yTop, 0, tipX, yMid, 0, 0, yMid, -w);

  // Ventral underside facets (2 crisp angled planes)
  emit(0, yBot, 0, tipX, yMid, 0, 0, yMid, w);
  emit(0, yBot, 0, 0, yMid, -w, tipX, yMid, 0);

  // Flush center mating cap at x = 0 (watertight closed solid)
  emit(0, yTop, 0, 0, yMid, -w, 0, yBot, 0);
  emit(0, yTop, 0, 0, yBot, 0, 0, yMid, w);

  return buildFacetedGeometry(pos);
}

/**
 * 8. Underside Machined Dark-Bronze Calibration Medallion & Brass Compass Ring
 * Seated flush into the bottom of the limestone slab so 360° underside inspection feels
 * like turning over a real 19th-century physics instrument.
 */
export function buildUndersideMedallionGeometry(): {
  bronzePlate: THREE.BufferGeometry;
  brassRing: THREE.BufferGeometry;
} {
  // Bottom-to-top profile (negative Y is further down)
  const platePts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, SLAB_BOTTOM_Y - 0.014),
    new THREE.Vector2(0.56, SLAB_BOTTOM_Y - 0.014),
    new THREE.Vector2(0.585, SLAB_BOTTOM_Y - 0.008),
    new THREE.Vector2(0.585, SLAB_BOTTOM_Y + 0.006),
    new THREE.Vector2(0.0, SLAB_BOTTOM_Y + 0.006),
  ];
  const bronzePlate = new THREE.LatheGeometry(platePts, 48);
  bronzePlate.computeVertexNormals();

  // Outer brass bezel ring around underside medallion
  const ringPts: THREE.Vector2[] = [
    new THREE.Vector2(0.575, SLAB_BOTTOM_Y - 0.016),
    new THREE.Vector2(0.625, SLAB_BOTTOM_Y - 0.016),
    new THREE.Vector2(0.640, SLAB_BOTTOM_Y - 0.009),
    new THREE.Vector2(0.640, SLAB_BOTTOM_Y + 0.005),
    new THREE.Vector2(0.575, SLAB_BOTTOM_Y + 0.005),
    new THREE.Vector2(0.575, SLAB_BOTTOM_Y - 0.016),
  ];
  const brassRing = new THREE.LatheGeometry(ringPts, 48);
  brassRing.computeVertexNormals();

  return { bronzePlate, brassRing };
}

/**
 * 9. Hovering Turned Brass & Dark-Magnetite Lodestone Stylus Wand.
 * Local origin (0, 0, 0) sits right at the business tip of the dark lodestone;
 * wand body extends cleanly along +Y to y = 0.88.
 * All LatheGeometry profiles ordered strictly bottom-to-top for outward CCW winding.
 */
export interface StylusGeometrySet {
  lodestoneTip: THREE.BufferGeometry;
  polarityBand: THREE.BufferGeometry;
  brassBody: THREE.BufferGeometry;
  darkGrip: THREE.BufferGeometry;
  brassPommel: THREE.BufferGeometry;
  dispose: () => void;
}

export function buildLodestoneStylusGeometries(): StylusGeometrySet {
  const segs = 32;

  // Dark magnetite/lodestone magnetic pole tip (y = 0..0.135)
  const tipPts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.0),
    new THREE.Vector2(0.019, 0.001),
    new THREE.Vector2(0.027, 0.010),
    new THREE.Vector2(0.0285, 0.128),
    new THREE.Vector2(0.025, 0.136),
    new THREE.Vector2(0.0, 0.136),
  ];
  const lodestoneTip = new THREE.LatheGeometry(tipPts, segs);
  lodestoneTip.computeVertexNormals();

  // Enameled polarity indicator collar (Cinnabar for +1 North-pull, Prussian for -1 South-pull)
  const bandPts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.128),
    new THREE.Vector2(0.032, 0.128),
    new THREE.Vector2(0.0335, 0.133),
    new THREE.Vector2(0.0335, 0.158),
    new THREE.Vector2(0.032, 0.163),
    new THREE.Vector2(0.0, 0.163),
  ];
  const polarityBand = new THREE.LatheGeometry(bandPts, segs);
  polarityBand.computeVertexNormals();

  // Turned satin-brass barrel with stepped instrument rings (y = 0.158..0.440)
  const brassPts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.158),
    new THREE.Vector2(0.030, 0.158),
    new THREE.Vector2(0.036, 0.168),
    new THREE.Vector2(0.036, 0.184),
    new THREE.Vector2(0.0325, 0.192),
    new THREE.Vector2(0.0345, 0.292),
    new THREE.Vector2(0.0395, 0.302),
    new THREE.Vector2(0.0395, 0.318),
    new THREE.Vector2(0.036, 0.326),
    new THREE.Vector2(0.038, 0.422),
    new THREE.Vector2(0.0415, 0.430),
    new THREE.Vector2(0.0415, 0.442),
    new THREE.Vector2(0.0, 0.442),
  ];
  const brassBody = new THREE.LatheGeometry(brassPts, segs);
  brassBody.computeVertexNormals();

  // Darkened-bronze / ebonized-iron upper grip sleeve (y = 0.436..0.655)
  const gripPts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.436),
    new THREE.Vector2(0.0395, 0.436),
    new THREE.Vector2(0.0405, 0.444),
    new THREE.Vector2(0.0415, 0.550),
    new THREE.Vector2(0.0395, 0.648),
    new THREE.Vector2(0.037, 0.656),
    new THREE.Vector2(0.0, 0.656),
  ];
  const darkGrip = new THREE.LatheGeometry(gripPts, segs);
  darkGrip.computeVertexNormals();

  // Satin-brass turned pommel finial cap (y = 0.650..0.728)
  const pommelPts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.650),
    new THREE.Vector2(0.038, 0.650),
    new THREE.Vector2(0.044, 0.660),
    new THREE.Vector2(0.0445, 0.678),
    new THREE.Vector2(0.0405, 0.692),
    new THREE.Vector2(0.030, 0.708),
    new THREE.Vector2(0.015, 0.722),
    new THREE.Vector2(0.0, 0.728),
  ];
  const brassPommel = new THREE.LatheGeometry(pommelPts, segs);
  brassPommel.computeVertexNormals();

  return {
    lodestoneTip,
    polarityBand,
    brassBody,
    darkGrip,
    brassPommel,
    dispose: () => {
      lodestoneTip.dispose();
      polarityBand.dispose();
      brassBody.dispose();
      darkGrip.dispose();
      brassPommel.dispose();
    },
  };
}
