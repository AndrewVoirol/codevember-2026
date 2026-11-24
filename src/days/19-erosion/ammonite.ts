import * as THREE from 'three/webgpu';
import {
  EROSION_GRID_INNER,
  EROSION_GRID_SIZE,
  mirrorOctahedralScalarGutter,
  octEncode,
  texelDirAndSolidAngle,
} from './octahedral';

/**
 * Logarithmic planispiral Jurassic ammonite geometry constants.
 * Designed bilaterally symmetric across Z = 0 so both front (+Z) and back (-Z)
 * limestone faces expose dramatic, deeply ribbed pyritized whorls.
 */
export const AMMONITE_TURNS = 3.22;
export const AMMONITE_THETA_MAX = AMMONITE_TURNS * Math.PI * 2;
export const AMMONITE_R0 = 0.0375;
export const AMMONITE_GROWTH_B = 0.132;
export const AMMONITE_RIBS_PER_TURN = 30;
/** Slight orientation twist so the wide ribbed body chamber faces upper-right & lower-front */
export const AMMONITE_PHASE_OFFSET = -0.35;

export interface AmmoniteBuildResult {
  geometry: THREE.BufferGeometry;
  maxOuterRadius: number;
  maxHalfThicknessZ: number;
}

function smoothstepCpu(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / Math.max(1e-6, e1 - e0)));
  return t * t * (3 - 2 * t);
}

/**
 * Evaluates a single surface point and metadata on the coiled pyritized ammonite shell.
 */
function evalAmmoniteSurface(
  theta: number,
  psi: number,
  outPos: [number, number, number],
  outMeta: [number, number, number, number]
): void {
  const whorlNorm = theta / AMMONITE_THETA_MAX;
  const rc = AMMONITE_R0 * Math.exp(AMMONITE_GROWTH_B * theta);

  // Semi-involute whorl cross-section: strictly a < 0.62 * rc everywhere so dorsal wall
  // never folds past the origin (guarantees inwardTri === 0 across every spiral ring)
  const a = 0.465 * rc;
  // Plump involute Jurassic ammonite profile sitting snugly inside the independent boulder hull:
  // rises cleanly from sculpted umbilical boss (0.225) out to ribbed outer body chamber (0.315)
  const c = 0.225 + 0.095 * Math.pow(whorlNorm, 1.25);

  // Smooth rounded embryonic protoconch closure at theta = 0
  const startCapTaper = smoothstepCpu(0.0, 0.35, theta);
  // Subtle aperture lip flare at terminal mouth (theta -> thetaMax)
  const apertureFlare = 1.0 + 0.038 * smoothstepCpu(AMMONITE_THETA_MAX - 0.32, AMMONITE_THETA_MAX, theta);
  const aEff = a * (0.65 + 0.35 * startCapTaper) * apertureFlare;
  const cEff = c * (0.86 + 0.14 * startCapTaper) * apertureFlare;

  const cosPsi = Math.cos(psi);
  const sinPsi = Math.sin(psi);
  const absSinPsi = Math.abs(sinPsi);

  // Wrapped angle from outer venter keel (psi = 0 -> 0, dorsal seam psi = PI -> PI)
  let psiWrap = psi % (Math.PI * 2);
  if (psiWrap > Math.PI) psiWrap -= Math.PI * 2;
  if (psiWrap < -Math.PI) psiWrap += Math.PI * 2;
  const absPsi = Math.abs(psiWrap);

  // Ventral keel + twin subtle sulci grooves (classic Jurassic Arietites / Hildoceras silhouette)
  const keelGauss = Math.exp(-Math.pow(absPsi / 0.21, 2));
  const sulcusGauss = Math.exp(-Math.pow((absPsi - 0.35) / 0.14, 2));
  const venterMask = smoothstepCpu(0.42, 0.04, absPsi);

  // Bold sigmoidal prorsiradiate primary ribs (bilaterally symmetric across Z=0 via cosPsi & absSinPsi)
  const ribSweep = 0.92 * cosPsi - 0.36 * (2.0 * absSinPsi * cosPsi);
  const ribPhase = AMMONITE_RIBS_PER_TURN * theta + ribSweep;
  const rawSin = 0.5 + 0.5 * Math.sin(ribPhase);
  const primaryRib = Math.pow(rawSin, 2.35);

  // Subtle bifurcating secondary ribs near the ventrolateral shoulder
  const shoulderMask = smoothstepCpu(-0.15, 0.65, cosPsi) * (1.0 - 0.85 * keelGauss);
  const secondaryRib =
    Math.pow(0.5 + 0.5 * Math.sin(ribPhase * 2.0 + 0.95), 2.8) * 0.24 * shoulderMask;

  // Ventrolateral tubercles (raised rhythmic nodes along outer flank shoulder)
  const tubercleZone = Math.exp(-Math.pow((absPsi - 0.68) / 0.28, 2));
  const tubercle = Math.pow(rawSin, 3.6) * tubercleZone * 0.32;

  // Fade ribs cleanly over dorsal overlap seam (cosPsi -> -1) and slightly across ventral keel groove
  const flankGate =
    smoothstepCpu(-0.88, -0.25, cosPsi) *
    (1.0 - 0.62 * keelGauss) *
    smoothstepCpu(0.35, 1.35, theta);

  const ribHeight = (primaryRib + secondaryRib + tubercle) * flankGate;
  const ribCrestNorm = Math.max(0, Math.min(1, ribHeight * 0.95 + keelGauss * 0.65));

  // Periodic growth-stage constrictions (paleontological varices ~4 per whorl)
  const varix = Math.pow(0.5 + 0.5 * Math.sin(theta * 4.0 + 0.6), 14.0) * 0.032 * flankGate;

  // Radial & axial displacements around whorl centerline
  const ribAmpR = aEff * 0.135;
  const ribAmpZ = cEff * 0.148;

  const baseR = aEff * (cosPsi + 0.095 * keelGauss - 0.045 * sulcusGauss);
  const baseZ = cEff * sinPsi * (0.94 + 0.08 * cosPsi);

  const localR = baseR + (ribHeight - 0.28 * flankGate) * ribAmpR * (0.35 + 0.65 * Math.max(0, cosPsi)) - varix * aEff;
  const localZ = baseZ + Math.sign(sinPsi || 1) * (ribHeight - 0.28 * flankGate) * ribAmpZ * absSinPsi - Math.sign(sinPsi || 1) * varix * cEff * absSinPsi;

  // Place along logarithmic spiral in XY plane
  const ang = theta + AMMONITE_PHASE_OFFSET;
  const cosA = Math.cos(ang);
  const sinA = Math.sin(ang);

  const radialDist = rc + localR;
  outPos[0] = radialDist * cosA;
  outPos[1] = radialDist * sinA;
  outPos[2] = localZ;

  // Subtle phylloceratic septal suture zone indicator (most visible along mid-flank)
  const sutureWave =
    0.5 +
    0.5 *
      Math.sin(
        theta * 14.0 +
          Math.sin(absPsi * 9.0) * 1.35 +
          Math.sin(absPsi * 23.0 + theta * 5.0) * 0.55
      );
  const sutureZone =
    smoothstepCpu(0.84, 0.98, sutureWave) *
    smoothstepCpu(0.25, 0.95, absSinPsi) *
    smoothstepCpu(0.18, 0.55, whorlNorm);

  outMeta[0] = ribCrestNorm;
  outMeta[1] = whorlNorm;
  outMeta[2] = sutureZone;
  outMeta[3] = venterMask;
}

/**
 * Generates a watertight, outward-wound 3D Pyritized Ammonite fossil geometry
 * with sharp radial costae ribs, ventral keel/sulci, ventrolateral tubercles,
 * closed protoconch cap, and closed terminal aperture wall.
 */
export function buildAmmoniteGeometry(
  spiralSteps = 280,
  profileSteps = 36
): AmmoniteBuildResult {
  const ringCount = spiralSteps + 1;
  const ringVerts = profileSteps; // closed seam via modulo indexing -> watertight welded ring
  const tubeVertCount = ringCount * ringVerts;
  const startCapCenterIdx = tubeVertCount;
  const endCapCenterIdx = tubeVertCount + 1;

  // Closed embryonic protoconch umbilical boss at (0,0,0) covering rXY <= 0.058, |z| <= 0.214
  // Uses true single-vertex north/south pole caps so ZERO degenerate zero-area pole triangles exist!
  const bossSegU = 18;
  const bossSegV = 12; // interior latitude rings: iv = 1 .. bossSegV - 1
  const bossInnerRings = bossSegV - 1;
  const bossNorthPoleIdx = tubeVertCount + 2;
  const bossSouthPoleIdx = tubeVertCount + 3;
  const bossRingBaseVert = tubeVertCount + 4;
  const bossVertCount = 2 + bossInnerRings * bossSegU;
  const totalVerts = bossNorthPoleIdx + bossVertCount;

  const positions = new Float32Array(totalVerts * 3);
  const metas = new Float32Array(totalVerts * 4);
  const uvs = new Float32Array(totalVerts * 2);

  const tmpPos: [number, number, number] = [0, 0, 0];
  const tmpMeta: [number, number, number, number] = [0, 0, 0, 0];

  let maxOuterRadius = 0;
  let maxHalfThicknessZ = 0;

  for (let i = 0; i <= spiralSteps; i++) {
    const u = i / spiralSteps;
    const theta = u * AMMONITE_THETA_MAX;
    for (let j = 0; j < profileSteps; j++) {
      const v = j / profileSteps;
      const psi = v * Math.PI * 2;
      evalAmmoniteSurface(theta, psi, tmpPos, tmpMeta);

      const idx = i * ringVerts + j;
      positions[idx * 3] = tmpPos[0];
      positions[idx * 3 + 1] = tmpPos[1];
      positions[idx * 3 + 2] = tmpPos[2];

      metas[idx * 4] = tmpMeta[0];
      metas[idx * 4 + 1] = tmpMeta[1];
      metas[idx * 4 + 2] = tmpMeta[2];
      metas[idx * 4 + 3] = tmpMeta[3];

      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;

      const rXY = Math.hypot(tmpPos[0], tmpPos[1]);
      if (rXY > maxOuterRadius) maxOuterRadius = rXY;
      const az = Math.abs(tmpPos[2]);
      if (az > maxHalfThicknessZ) maxHalfThicknessZ = az;
    }
  }

  // Start cap center (protoconch at theta = 0, pushed slightly outward along -E_t for convex closure)
  {
    const rc0 = AMMONITE_R0;
    const ang0 = AMMONITE_PHASE_OFFSET;
    const tx0 = -Math.sin(ang0);
    const ty0 = Math.cos(ang0);
    positions[startCapCenterIdx * 3] = rc0 * Math.cos(ang0) - tx0 * 0.008;
    positions[startCapCenterIdx * 3 + 1] = rc0 * Math.sin(ang0) - ty0 * 0.008;
    positions[startCapCenterIdx * 3 + 2] = 0;
    metas[startCapCenterIdx * 4] = 0.25;
    metas[startCapCenterIdx * 4 + 1] = 0.0;
    metas[startCapCenterIdx * 4 + 2] = 0.0;
    metas[startCapCenterIdx * 4 + 3] = 0.0;
    uvs[startCapCenterIdx * 2] = 0;
    uvs[startCapCenterIdx * 2 + 1] = 0.5;
  }

  // End cap center (solid mineralized terminal chamber aperture at theta = thetaMax)
  {
    const rcEnd = AMMONITE_R0 * Math.exp(AMMONITE_GROWTH_B * AMMONITE_THETA_MAX);
    const angEnd = AMMONITE_THETA_MAX + AMMONITE_PHASE_OFFSET;
    const txEnd = -Math.sin(angEnd);
    const tyEnd = Math.cos(angEnd);
    // Slightly domed outward along +E_t for positive volume & clean chiseled mineral infill
    positions[endCapCenterIdx * 3] = rcEnd * Math.cos(angEnd) + txEnd * 0.025;
    positions[endCapCenterIdx * 3 + 1] = rcEnd * Math.sin(angEnd) + tyEnd * 0.025;
    positions[endCapCenterIdx * 3 + 2] = 0;
    metas[endCapCenterIdx * 4] = 0.35;
    metas[endCapCenterIdx * 4 + 1] = 1.0;
    metas[endCapCenterIdx * 4 + 2] = 0.0;
    metas[endCapCenterIdx * 4 + 3] = 0.0;
    uvs[endCapCenterIdx * 2] = 1;
    uvs[endCapCenterIdx * 2 + 1] = 0.5;
  }

  // Embryonic protoconch umbilical pearl vertices (pole-capped outward-wound prolate ellipsoid along Z)
  const bossRxy = 0.056;
  const bossRz = 0.214;

  // North pole (+Z)
  positions[bossNorthPoleIdx * 3] = 0;
  positions[bossNorthPoleIdx * 3 + 1] = 0;
  positions[bossNorthPoleIdx * 3 + 2] = bossRz;
  metas[bossNorthPoleIdx * 4] = 0.56;
  metas[bossNorthPoleIdx * 4 + 1] = 0.0;
  metas[bossNorthPoleIdx * 4 + 2] = 0.0;
  metas[bossNorthPoleIdx * 4 + 3] = 0.0;
  uvs[bossNorthPoleIdx * 2] = 0.0;
  uvs[bossNorthPoleIdx * 2 + 1] = 0.25;

  // South pole (-Z)
  positions[bossSouthPoleIdx * 3] = 0;
  positions[bossSouthPoleIdx * 3 + 1] = 0;
  positions[bossSouthPoleIdx * 3 + 2] = -bossRz;
  metas[bossSouthPoleIdx * 4] = 0.56;
  metas[bossSouthPoleIdx * 4 + 1] = 0.0;
  metas[bossSouthPoleIdx * 4 + 2] = 0.0;
  metas[bossSouthPoleIdx * 4 + 3] = 0.0;
  uvs[bossSouthPoleIdx * 2] = 0.0;
  uvs[bossSouthPoleIdx * 2 + 1] = 0.75;

  for (let iv = 1; iv <= bossInnerRings; iv++) {
    const phi = (iv / bossSegV) * Math.PI; // strictly inside (0, PI)
    const sinP = Math.sin(phi);
    const cosP = Math.cos(phi);
    for (let iu = 0; iu < bossSegU; iu++) {
      const az = (iu / bossSegU) * Math.PI * 2;
      const vIdx = bossRingBaseVert + (iv - 1) * bossSegU + iu;
      positions[vIdx * 3] = Math.cos(az) * sinP * bossRxy;
      positions[vIdx * 3 + 1] = Math.sin(az) * sinP * bossRxy;
      positions[vIdx * 3 + 2] = cosP * bossRz;
      metas[vIdx * 4] = 0.48 + 0.08 * Math.abs(cosP);
      metas[vIdx * 4 + 1] = 0.0;
      metas[vIdx * 4 + 2] = 0.0;
      metas[vIdx * 4 + 3] = 0.0;
      uvs[vIdx * 2] = iu / bossSegU;
      uvs[vIdx * 2 + 1] = iv / bossSegV;
    }
  }

  const bossTriCount = bossSegU * 2 + (bossInnerRings - 1) * bossSegU * 2;
  const triCount = spiralSteps * profileSteps * 2 + profileSteps * 2 + bossTriCount;
  const indices = new Uint32Array(triCount * 3);
  let ptr = 0;

  // Tube surface quads: (d/dTheta) x (d/dPsi) points OUTWARD -> (a, b, c) and (a, c, d)
  for (let i = 0; i < spiralSteps; i++) {
    const row0 = i * ringVerts;
    const row1 = (i + 1) * ringVerts;
    for (let j = 0; j < profileSteps; j++) {
      const jNext = (j + 1) % profileSteps;
      const a = row0 + j;
      const b = row1 + j;
      const c = row1 + jNext;
      const d = row0 + jNext;

      indices[ptr++] = a;
      indices[ptr++] = b;
      indices[ptr++] = c;

      indices[ptr++] = a;
      indices[ptr++] = c;
      indices[ptr++] = d;
    }
  }

  // Start cap (faces -E_t outward): (center, ring(0, j), ring(0, jNext))
  for (let j = 0; j < profileSteps; j++) {
    const jNext = (j + 1) % profileSteps;
    indices[ptr++] = startCapCenterIdx;
    indices[ptr++] = j;
    indices[ptr++] = jNext;
  }

  // End cap (faces +E_t outward): (center, ring(last, jNext), ring(last, j))
  const lastRow = spiralSteps * ringVerts;
  for (let j = 0; j < profileSteps; j++) {
    const jNext = (j + 1) % profileSteps;
    indices[ptr++] = endCapCenterIdx;
    indices[ptr++] = lastRow + jNext;
    indices[ptr++] = lastRow + j;
  }

  // Embryonic protoconch boss north-pole fan (+Z outward):
  // (ring(0, iu) - north) x (ring(0, iuNext) - north) has +Z component!
  for (let iu = 0; iu < bossSegU; iu++) {
    const iuNext = (iu + 1) % bossSegU;
    indices[ptr++] = bossNorthPoleIdx;
    indices[ptr++] = bossRingBaseVert + iu;
    indices[ptr++] = bossRingBaseVert + iuNext;
  }

  // Embryonic protoconch boss interior quads: (d/dPhi) x (d/dAz) points OUTWARD
  for (let r = 0; r < bossInnerRings - 1; r++) {
    const row0 = bossRingBaseVert + r * bossSegU;
    const row1 = bossRingBaseVert + (r + 1) * bossSegU;
    for (let iu = 0; iu < bossSegU; iu++) {
      const iuNext = (iu + 1) % bossSegU;
      const a = row0 + iu;
      const b = row1 + iu;
      const c = row1 + iuNext;
      const d = row0 + iuNext;

      indices[ptr++] = a;
      indices[ptr++] = b;
      indices[ptr++] = c;

      indices[ptr++] = a;
      indices[ptr++] = c;
      indices[ptr++] = d;
    }
  }

  // Embryonic protoconch boss south-pole fan (-Z outward):
  // (ring(last, iuNext) - south) x (ring(last, iu) - south) has -Z component!
  const bossLastRow = bossRingBaseVert + (bossInnerRings - 1) * bossSegU;
  for (let iu = 0; iu < bossSegU; iu++) {
    const iuNext = (iu + 1) % bossSegU;
    indices[ptr++] = bossSouthPoleIdx;
    indices[ptr++] = bossLastRow + iuNext;
    indices[ptr++] = bossLastRow + iu;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aAmmoniteMeta', new THREE.BufferAttribute(metas, 4));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();

  return {
    geometry,
    maxOuterRadius,
    maxHalfThicknessZ,
  };
}

/**
 * Exact origin-ray rasterizer baking the 3D ammonite triangle mesh onto the
 * octahedral erosion grid (`size x size`).
 *
 * For every interior texel direction `dir`, computes the exact outermost radial
 * intersection distance `t > 0` where ray `p(t) = t * dir` intersects the ammonite
 * mesh (`0` outside the ammonite footprint). This binds the differential-hardness
 * limestone erosion floor directly to the true 3D fossil ribs with sub-mm precision.
 */
export function bakeAmmoniteRadialEnvelope(
  geometry: THREE.BufferGeometry,
  size = EROSION_GRID_SIZE
): Float32Array {
  const inner = size - 2;
  const fossilRadius = new Float32Array(size * size);

  // Precompute unit directions for interior texels so ray-triangle tests are allocation-free
  const dirX = new Float32Array(size * size);
  const dirY = new Float32Array(size * size);
  const dirZ = new Float32Array(size * size);
  const tmpDir: [number, number, number] = [0, 0, 0];

  for (let iy = 1; iy <= inner; iy++) {
    for (let ix = 1; ix <= inner; ix++) {
      texelDirAndSolidAngle(ix, iy, tmpDir, size);
      const idx = iy * size + ix;
      dirX[idx] = tmpDir[0];
      dirY[idx] = tmpDir[1];
      dirZ[idx] = tmpDir[2];
    }
  }

  const pos = geometry.attributes.position as THREE.BufferAttribute;
  const index = geometry.index;
  const triCount = index ? index.count / 3 : pos.count / 3;

  const o0: [number, number] = [0, 0];
  const o1: [number, number] = [0, 0];
  const o2: [number, number] = [0, 0];

  // Helper: origin-ray vs triangle (v0, v1, v2) intersection using Cramer's rule
  // Solves u*v0 + v*v1 + w*v2 = t*dir  <=>  det(v0, v1, v2) / (u' + v' + w') = t
  const testTexel = (
    idx: number,
    v0x: number,
    v0y: number,
    v0z: number,
    e1x: number,
    e1y: number,
    e1z: number,
    e2x: number,
    e2y: number,
    e2z: number
  ) => {
    const dx = dirX[idx];
    const dy = dirY[idx];
    const dz = dirZ[idx];

    // Moller-Trumbore with ray origin = (0, 0, 0), so T = -v0
    const px = dy * e2z - dz * e2y;
    const py = dz * e2x - dx * e2z;
    const pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) < 1e-10) return;

    const invDet = 1.0 / det;
    const tx = -v0x;
    const ty = -v0y;
    const tz = -v0z;
    const u = (tx * px + ty * py + tz * pz) * invDet;
    if (u < -1e-4 || u > 1.0 + 1e-4) return;

    const qx = ty * e1z - tz * e1y;
    const qy = tz * e1x - tx * e1z;
    const qz = tx * e1y - ty * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * invDet;
    if (v < -1e-4 || u + v > 1.0 + 1e-4) return;

    const t = (e2x * qx + e2y * qy + e2z * qz) * invDet;
    if (t > fossilRadius[idx]) {
      fossilRadius[idx] = t;
    }
  };

  for (let t = 0; t < triCount; t++) {
    const i0 = index ? index.getX(t * 3) : t * 3;
    const i1 = index ? index.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = index ? index.getX(t * 3 + 2) : t * 3 + 2;

    const v0x = pos.getX(i0);
    const v0y = pos.getY(i0);
    const v0z = pos.getZ(i0);
    const v1x = pos.getX(i1);
    const v1y = pos.getY(i1);
    const v1z = pos.getZ(i1);
    const v2x = pos.getX(i2);
    const v2y = pos.getY(i2);
    const v2z = pos.getZ(i2);

    const e1x = v1x - v0x;
    const e1y = v1y - v0y;
    const e1z = v1z - v0z;
    const e2x = v2x - v0x;
    const e2y = v2y - v0y;
    const e2z = v2z - v0z;

    const minX = Math.min(v0x, v1x, v2x);
    const maxX = Math.max(v0x, v1x, v2x);
    const minY = Math.min(v0y, v1y, v2y);
    const maxY = Math.max(v0y, v1y, v2y);
    const minZ = Math.min(v0z, v1z, v2z);
    const maxZ = Math.max(v0z, v1z, v2z);

    octEncode(v0x, v0y, v0z, o0);
    octEncode(v1x, v1y, v1z, o1);
    octEncode(v2x, v2y, v2z, o2);

    let minOx = Math.min(o0[0], o1[0], o2[0]);
    let maxOx = Math.max(o0[0], o1[0], o2[0]);
    let minOy = Math.min(o0[1], o1[1], o2[1]);
    let maxOy = Math.max(o0[1], o1[1], o2[1]);

    // When a triangle straddles the z = 0 equatorial crease, include its exact z = 0 crossings
    // and any enclosed cardinal octahedral corner (+/-1, 0) or (0, +/-1) in the local bbox
    if (minZ <= 0 && maxZ >= 0) {
      const addZeroZ = (
        ax: number,
        ay: number,
        az: number,
        bx: number,
        by: number,
        bz: number
      ) => {
        if ((az < 0 && bz > 0) || (az > 0 && bz < 0)) {
          const s = -az / (bz - az);
          octEncode(ax + (bx - ax) * s, ay + (by - ay) * s, 0, o0);
          if (o0[0] < minOx) minOx = o0[0];
          if (o0[0] > maxOx) maxOx = o0[0];
          if (o0[1] < minOy) minOy = o0[1];
          if (o0[1] > maxOy) maxOy = o0[1];
        }
      };
      addZeroZ(v0x, v0y, v0z, v1x, v1y, v1z);
      addZeroZ(v1x, v1y, v1z, v2x, v2y, v2z);
      addZeroZ(v2x, v2y, v2z, v0x, v0y, v0z);
      if (minY <= 0 && maxY >= 0) {
        const sx = v0x + v1x + v2x >= 0 ? 1.0 : -1.0;
        if (sx < minOx) minOx = sx;
        if (sx > maxOx) maxOx = sx;
        if (0 < minOy) minOy = 0;
        if (0 > maxOy) maxOy = 0;
      }
      if (minX <= 0 && maxX >= 0) {
        const sy = v0y + v1y + v2y >= 0 ? 1.0 : -1.0;
        if (0 < minOx) minOx = 0;
        if (0 > maxOx) maxOx = 0;
        if (sy < minOy) minOy = sy;
        if (sy > maxOy) maxOy = sy;
      }
    }

    // Only triangles truly wrapping across opposite octahedral back-hemisphere edges need spherical-cone scan
    const wrapsSeam =
      maxOx - minOx > 0.65 ||
      maxOy - minOy > 0.65 ||
      (maxZ < 0 && ((minX <= 0 && maxX >= 0) || (minY <= 0 && maxY >= 0)) && (Math.abs(v0z) > 0.08 || Math.abs(v1z) > 0.08));

    if (!wrapsSeam) {
      const pad = 2.0 / inner;
      const ixMin = Math.max(1, Math.floor(((minOx - pad) * 0.5 + 0.5) * inner + 0.5));
      const ixMax = Math.min(inner, Math.ceil(((maxOx + pad) * 0.5 + 0.5) * inner + 0.5));
      const iyMin = Math.max(1, Math.floor(((minOy - pad) * 0.5 + 0.5) * inner + 0.5));
      const iyMax = Math.min(inner, Math.ceil(((maxOy + pad) * 0.5 + 0.5) * inner + 0.5));

      for (let iy = iyMin; iy <= iyMax; iy++) {
        const row = iy * size;
        for (let ix = ixMin; ix <= ixMax; ix++) {
          testTexel(row + ix, v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z);
        }
      }
    } else {
      // Seam-crossing triangle near -Z pole: test against tight spherical bounding cone
      const l0 = Math.hypot(v0x, v0y, v0z) || 1;
      const l1 = Math.hypot(v1x, v1y, v1z) || 1;
      const l2 = Math.hypot(v2x, v2y, v2z) || 1;
      let cx = v0x / l0 + v1x / l1 + v2x / l2;
      let cy = v0y / l0 + v1y / l1 + v2y / l2;
      let cz = v0z / l0 + v1z / l1 + v2z / l2;
      const cl = Math.hypot(cx, cy, cz) || 1;
      cx /= cl;
      cy /= cl;
      cz /= cl;
      const minDot =
        Math.min(
          (v0x / l0) * cx + (v0y / l0) * cy + (v0z / l0) * cz,
          (v1x / l1) * cx + (v1y / l1) * cy + (v1z / l1) * cz,
          (v2x / l2) * cx + (v2y / l2) * cy + (v2z / l2) * cz
        ) - 0.04;

      for (let iy = 1; iy <= inner; iy++) {
        const row = iy * size;
        for (let ix = 1; ix <= inner; ix++) {
          const idx = row + ix;
          const d = dirX[idx] * cx + dirY[idx] * cy + dirZ[idx] * cz;
          if (d >= minDot) {
            testTexel(idx, v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z);
          }
        }
      }
    }
  }

  // Fill any microscopic sub-texel umbilical center pinholes with local 4-neighbor bridge if surrounded
  const filled = new Float32Array(fossilRadius);
  for (let iy = 2; iy <= inner - 1; iy++) {
    for (let ix = 2; ix <= inner - 1; ix++) {
      const idx = iy * size + ix;
      if (fossilRadius[idx] === 0) {
        const a = fossilRadius[idx - 1];
        const b = fossilRadius[idx + 1];
        const c = fossilRadius[idx - size];
        const d = fossilRadius[idx + size];
        if (a > 0 && b > 0 && c > 0 && d > 0) {
          filled[idx] = 0.25 * (a + b + c + d);
        }
      }
    }
  }
  fossilRadius.set(filled);

  mirrorOctahedralScalarGutter(fossilRadius, size);
  return fossilRadius;
}

export const _ammoniteGridMeta = { EROSION_GRID_INNER };
