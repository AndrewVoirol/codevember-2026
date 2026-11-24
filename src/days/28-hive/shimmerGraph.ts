/**
 * Day 28 — Hive: Apis dorsata Shimmering Defense-Wave Graph & Excitable-Media Solver
 *
 * Pure mathematical & biomechanical simulation module (zero DOM/WebGPU dependencies):
 * - Shared parametric gnarled-branch centerline & 3D hanging teardrop bivouac manifold
 * - Arc-length stratified, half-pitch shingled tiling (~2,640 articulated honeybee units)
 * - Symmetrized 3D surface k-NN neighbor graph (6-8 local neighbors wrapping front,
 *   flanks, back, and rounded pendant rim without cross-wall short-circuiting)
 * - Continuous-time coupled excitable-media defense-wave dynamics
 *   (Resting -> Firing [Abdomen Flip!] -> Refractory -> Resting) with finite biological
 *   wave speed (~0.55-0.80 m/s), clean head-on wave annihilation, spiral wave support,
 *   damped biomechanical abdomen-flip actuators, and coupled cluster pendulum recoil.
 */

export const CURTAIN_PIVOT_Y = 0.84;
export const WORLD_TO_METERS = 0.42; // 2.12-unit hanging bivouac ≈ 0.89 m wild Apis dorsata comb

export const UNIT_STATE_RESTING = 0;
export const UNIT_STATE_FIRING = 1;
export const UNIT_STATE_REFRACTORY = 2;

export interface BranchSample {
  x: number;
  y: number;
  z: number;
  r: number;
}

export interface SurfaceFrame {
  px: number;
  py: number;
  pz: number;
  nx: number;
  ny: number;
  nz: number;
  ux: number;
  uy: number;
  uz: number;
  tx: number;
  ty: number;
  tz: number;
}

export interface HiveClusterGraph {
  unitCount: number;
  /** Anchor positions in curtain-local coordinates (pivot at world Y = CURTAIN_PIVOT_Y) */
  positions: Float32Array;
  /** Outward unit surface normals */
  normals: Float32Array;
  /** Upward meridian unit tangent (head/thorax orientation toward branch) */
  upDirs: Float32Array;
  /** Right circumferential unit tangent (T = U x N, abdomen hinge axis) */
  tangents: Float32Array;
  /** Per-unit biological scale variation [0.90..1.10] */
  scales: Float32Array;
  /** Per-unit resting shingle pitch outward [0.06..0.13 rad] */
  restTilts: Float32Array;
  /** Per-unit deterministic seed [0..1] for pigmentation & micro-phase */
  seeds: Float32Array;
  /** Per-unit biological excitability / flip gain [0.92..1.08] */
  gains: Float32Array;
  /** CSR neighbor graph offsets (length = unitCount + 1) */
  neighborOffsets: Int32Array;
  /** Flat CSR neighbor unit indices */
  neighborIds: Int32Array;
  /** Normalized coupling weights (sum = 1 per unit) */
  neighborWeights: Float32Array;
  /** Euclidean 3D distance to each neighbor */
  neighborDists: Float32Array;
  /** Average spacing between adjacent units (world units) */
  meanSpacing: number;
  /** Minimum neighbor count across all units */
  minDegree: number;
  /** Maximum neighbor count across all units */
  maxDegree: number;
}

export interface HiveTelemetry {
  unitCount: number;
  activeFiringCount: number;
  refractoryCount: number;
  peakWaveSpeed: number;
  totalWavesTriggered: number;
  clusterSwayRad: number;
}

/** Deterministic fast PRNG (Mulberry32) */
export function createSeededRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Evaluates the gnarled dark-walnut/charred-oak branch centerline and radius
 * in world space at coordinate x in [-1.92, +1.92].
 * Shared identically by hiveGeometry.ts and the hanging curtain top collar.
 */
export function evalBranchCenterline(x: number, out?: BranchSample): BranchSample {
  const u = Math.max(0, Math.min(1, (x + 1.92) / 3.84));
  const y =
    0.86 -
    0.048 * x * x +
    0.046 * Math.sin(1.22 * x + 0.35) +
    0.018 * Math.cos(2.6 * x - 0.7);
  const z = 0.052 * Math.sin(1.05 * x - 0.2) + 0.016 * Math.cos(2.3 * x);

  // Taper naturally from gnarled left butt to slender right limb with knot swells
  const knotLeft = 0.014 * Math.exp(-Math.pow((x + 1.26) / 0.18, 2));
  const knotRight = 0.010 * Math.exp(-Math.pow((x - 1.32) / 0.16, 2));
  const knotMid = 0.007 * Math.exp(-Math.pow((x + 0.18) / 0.22, 2));
  const gnarl = 0.0055 * Math.sin(3.7 * x + 0.45) + 0.003 * Math.cos(6.1 * x - 1.1);
  const r = 0.146 * (1 - u) + 0.088 * u + knotLeft + knotRight + knotMid + gnarl;

  if (out) {
    out.x = x;
    out.y = y;
    out.z = z;
    out.r = r;
    return out;
  }
  return { x, y, z, r };
}

const _branchScratch: BranchSample = { x: 0, y: 0, z: 0, r: 0.12 };

/**
 * Evaluates a 3D point on the smooth closed hanging Apis dorsata teardrop curtain
 * in curtain-local coordinates (where local y = 0 sits at world Y = CURTAIN_PIVOT_Y).
 *
 * - theta in [0, PI/2]: 0 = top anchor collar clasping the gnarled branch,
 *                       PI/2 = smoothly rounded C2 bottom teardrop crest.
 * - phi in [0, 2*PI):   0 = +X right flank, PI/2 = +Z front face,
 *                       PI = -X left flank, 3*PI/2 = -Z back face.
 * - radialOffset: optional normal inset/outset (used for inner brood core & raycast hull).
 */
export function evalCurtainPoint(
  theta: number,
  phi: number,
  out: [number, number, number],
  insetScale = 1.0
): void {
  const th = Math.max(0, Math.min(Math.PI * 0.5, theta));
  const cosT = Math.cos(th);
  const sinT = Math.sin(th);
  const cosP = Math.cos(phi);
  const sinP = Math.sin(phi);

  // Full semicircular shoulders tapering gracefully into a plump pendant teardrop
  const rx = 1.05 * cosT * (1.0 + 0.115 * Math.sin(2.0 * th)) * insetScale;
  const xApprox = rx * cosP;
  evalBranchCenterline(xApprox, _branchScratch);

  // Hug branch circumference at collar (theta=0), swell into thick living curtain below
  const collarZ = (_branchScratch.r + 0.015) * (0.85 + 0.15 * insetScale);
  const rz = (collarZ + 0.42 * Math.sin(th * 1.52) * insetScale) * cosT;

  const branchFollow = Math.pow(cosT, 1.38);

  // Slight organic biological lean on lower pendant lobe
  const asymX = -0.048 * sinT * sinT * insetScale;
  let x = xApprox + asymX;

  // Top collar clasps gently up the front & back lower flanks of the bark
  const barkClasp = 0.056 * Math.abs(sinP) * Math.pow(cosT, 5.0);
  const yCollar = _branchScratch.y - CURTAIN_PIVOT_Y - _branchScratch.r * 0.18 + barkClasp;
  const dropDepth = 1.50 * (0.985 + 0.015 * insetScale);
  let y =
    yCollar * branchFollow +
    -0.055 * (1.0 - branchFollow) -
    dropDepth * Math.pow(sinT, 1.04);

  let z = _branchScratch.z * branchFollow + rz * sinP;

  // Low-frequency 3D Cartesian organic living-curtain undulation (zero polar harmonics!)
  const env = sinT * cosT;
  if (env > 1e-4) {
    const u1 = Math.sin(2.15 * x + 1.73 * y + 1.37 * z + 0.4);
    const u2 = Math.cos(1.47 * x - 2.58 * y + 2.19 * z - 0.8);
    const disp = (0.022 * u1 + 0.014 * u2) * env * insetScale;
    const horizLen = Math.hypot(x, z) + 1e-5;
    x += (x / horizLen) * disp * 0.55;
    z += (z / horizLen) * disp * 1.15;
    y += 0.012 * u2 * env * insetScale;
  }

  out[0] = x;
  out[1] = y;
  out[2] = z;
}

const _p0: [number, number, number] = [0, 0, 0];
const _pT1: [number, number, number] = [0, 0, 0];
const _pT2: [number, number, number] = [0, 0, 0];
const _pP1: [number, number, number] = [0, 0, 0];
const _pP2: [number, number, number] = [0, 0, 0];

/**
 * Evaluates exact position and right-handed orthonormal bee frame (T, U, N)
 * on the hanging teardrop surface:
 * - N: outward surface normal
 * - U: upward along meridian toward branch (bee head direction)
 * - T = U x N: circumferential right tangent (abdomen flip hinge axis)
 */
export function evalCurtainFrame(
  theta: number,
  phi: number,
  out: SurfaceFrame,
  insetScale = 1.0
): SurfaceFrame {
  const th = Math.max(0.002, Math.min(Math.PI * 0.5 - 0.002, theta));
  const epsT = 0.002;
  const epsP = 0.002;

  evalCurtainPoint(th, phi, _p0, insetScale);
  evalCurtainPoint(Math.min(Math.PI * 0.5 - 1e-4, th + epsT), phi, _pT1, insetScale);
  evalCurtainPoint(Math.max(1e-4, th - epsT), phi, _pT2, insetScale);
  evalCurtainPoint(th, phi + epsP, _pP1, insetScale);
  evalCurtainPoint(th, phi - epsP, _pP2, insetScale);

  // dP/dtheta points DOWNWARD toward the bottom crest
  const dTx = _pT1[0] - _pT2[0];
  const dTy = _pT1[1] - _pT2[1];
  const dTz = _pT1[2] - _pT2[2];

  // dP/dphi points counter-clockwise when viewed from +Y (front face +Z -> -X)
  const dPx = _pP1[0] - _pP2[0];
  const dPy = _pP1[1] - _pP2[1];
  const dPz = _pP1[2] - _pP2[2];

  // Outward normal: dP/dphi x dP/dtheta
  let nx = dPy * dTz - dPz * dTy;
  let ny = dPz * dTx - dPx * dTz;
  let nz = dPx * dTy - dPy * dTx;
  let nLen = Math.hypot(nx, ny, nz);
  if (nLen < 1e-8) {
    nx = 0;
    ny = -1;
    nz = 0;
    nLen = 1;
  }
  nx /= nLen;
  ny /= nLen;
  nz /= nLen;

  // Enforce strict outward orientation relative to curtain interior centroid (0, -0.52, 0)
  const relX = _p0[0];
  const relY = _p0[1] - -0.52;
  const relZ = _p0[2];
  if (nx * relX + ny * relY + nz * relZ < 0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }

  // Upward tangent U along -dP/dtheta, Gram-Schmidt orthogonalized against N
  let ux = -dTx;
  let uy = -dTy;
  let uz = -dTz;
  const uDotN = ux * nx + uy * ny + uz * nz;
  ux -= uDotN * nx;
  uy -= uDotN * ny;
  uz -= uDotN * nz;
  let uLen = Math.hypot(ux, uy, uz);
  if (uLen < 1e-6) {
    // Fallback near apex
    ux = 0;
    uy = 1 - ny * ny;
    uz = -ny * nz;
    uLen = Math.hypot(ux, uy, uz) || 1;
  }
  ux /= uLen;
  uy /= uLen;
  uz /= uLen;

  // Right-handed circumferential hinge tangent T = U x N  (check: T x U = N)
  let tx = uy * nz - uz * ny;
  let ty = uz * nx - ux * nz;
  let tz = ux * ny - uy * nx;
  const tLen = Math.hypot(tx, ty, tz) || 1;
  tx /= tLen;
  ty /= tLen;
  tz /= tLen;

  out.px = _p0[0];
  out.py = _p0[1];
  out.pz = _p0[2];
  out.nx = nx;
  out.ny = ny;
  out.nz = nz;
  out.ux = ux;
  out.uy = uy;
  out.uz = uz;
  out.tx = tx;
  out.ty = ty;
  out.tz = tz;
  return out;
}

/**
 * Generates ~2,640 evenly spaced, staggered shingled honeybee units wrapping
 * the full 3D hanging teardrop curtain and builds the symmetrized 3D k-NN graph.
 */
export function buildHiveClusterGraph(seed = 202628): HiveClusterGraph {
  const rng = createSeededRng(seed);

  // 1. Sample meridian arc-length profile across front & flank averages so row pitch is uniform
  const M_SAMPLES = 240;
  const thetaMin = 0.022;
  const thetaMax = Math.PI * 0.5 - 0.055;
  const meridianThetas = new Float64Array(M_SAMPLES + 1);
  const meridianArc = new Float64Array(M_SAMPLES + 1);
  const pPrevF: [number, number, number] = [0, 0, 0];
  const pCurF: [number, number, number] = [0, 0, 0];
  const pPrevS: [number, number, number] = [0, 0, 0];
  const pCurS: [number, number, number] = [0, 0, 0];

  meridianThetas[0] = thetaMin;
  evalCurtainPoint(thetaMin, Math.PI * 0.5, pPrevF);
  evalCurtainPoint(thetaMin, 0.0, pPrevS);

  for (let i = 1; i <= M_SAMPLES; i++) {
    const th = thetaMin + ((thetaMax - thetaMin) * i) / M_SAMPLES;
    meridianThetas[i] = th;
    evalCurtainPoint(th, Math.PI * 0.5, pCurF);
    evalCurtainPoint(th, 0.0, pCurS);
    const dsFront = Math.hypot(
      pCurF[0] - pPrevF[0],
      pCurF[1] - pPrevF[1],
      pCurF[2] - pPrevF[2]
    );
    const dsSide = Math.hypot(
      pCurS[0] - pPrevS[0],
      pCurS[1] - pPrevS[1],
      pCurS[2] - pPrevS[2]
    );
    meridianArc[i] = meridianArc[i - 1] + 0.58 * dsFront + 0.42 * dsSide;
    pPrevF[0] = pCurF[0];
    pPrevF[1] = pCurF[1];
    pPrevF[2] = pCurF[2];
    pPrevS[0] = pCurS[0];
    pPrevS[1] = pCurS[1];
    pPrevS[2] = pCurS[2];
  }

  const totalMeridianArc = meridianArc[M_SAMPLES];
  const rowPitch = 0.054;
  const colPitch = 0.049;
  const rowCount = Math.max(20, Math.round(totalMeridianArc / rowPitch));

  // Temporary dynamic lists before packing into typed arrays
  const posList: number[] = [];
  const normList: number[] = [];
  const upList: number[] = [];
  const tangList: number[] = [];
  const scaleList: number[] = [];
  const tiltList: number[] = [];
  const seedList: number[] = [];
  const gainList: number[] = [];

  // Spatial grid to guarantee strictly zero overlapping degenerate positions
  const minAllowedDist = 0.026;
  const minAllowedDistSq = minAllowedDist * minAllowedDist;
  const checkCellSize = 0.06;
  const checkGrid = new Map<string, number[]>();

  const cellKey = (cx: number, cy: number, cz: number) => `${cx},${cy},${cz}`;

  const tryPlaceUnit = (frame: SurfaceFrame): boolean => {
    const cx = Math.floor(frame.px / checkCellSize);
    const cy = Math.floor(frame.py / checkCellSize);
    const cz = Math.floor(frame.pz / checkCellSize);

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = checkGrid.get(cellKey(cx + dx, cy + dy, cz + dz));
          if (!bucket) continue;
          for (let b = 0; b < bucket.length; b++) {
            const idx3 = bucket[b] * 3;
            const ddx = frame.px - posList[idx3];
            const ddy = frame.py - posList[idx3 + 1];
            const ddz = frame.pz - posList[idx3 + 2];
            if (ddx * ddx + ddy * ddy + ddz * ddz < minAllowedDistSq) {
              return false;
            }
          }
        }
      }
    }

    const newIdx = posList.length / 3;
    posList.push(frame.px, frame.py, frame.pz);
    normList.push(frame.nx, frame.ny, frame.nz);
    upList.push(frame.ux, frame.uy, frame.uz);
    tangList.push(frame.tx, frame.ty, frame.tz);
    scaleList.push(0.96 + rng() * 0.17);
    tiltList.push(0.065 + rng() * 0.065);
    seedList.push(rng());
    gainList.push(0.92 + rng() * 0.16);

    const k = cellKey(cx, cy, cz);
    let bucket = checkGrid.get(k);
    if (!bucket) {
      bucket = [];
      checkGrid.set(k, bucket);
    }
    bucket.push(newIdx);
    return true;
  };

  // Helper to invert meridian arc length -> theta
  const sampleMeridianTheta = (targetArc: number): number => {
    const clamped = Math.max(0, Math.min(totalMeridianArc, targetArc));
    let lo = 0;
    let hi = M_SAMPLES;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (meridianArc[mid] < clamped) lo = mid + 1;
      else hi = mid;
    }
    const idx = Math.max(1, lo);
    const a0 = meridianArc[idx - 1];
    const a1 = meridianArc[idx];
    const frac = a1 > a0 ? (clamped - a0) / (a1 - a0) : 0;
    return meridianThetas[idx - 1] + (meridianThetas[idx] - meridianThetas[idx - 1]) * frac;
  };

  const RING_SAMPLES = 256;
  const ringArc = new Float64Array(RING_SAMPLES + 1);
  const rpPrev: [number, number, number] = [0, 0, 0];
  const rpCur: [number, number, number] = [0, 0, 0];
  const frameScratch: SurfaceFrame = {
    px: 0,
    py: 0,
    pz: 0,
    nx: 0,
    ny: 0,
    nz: 1,
    ux: 0,
    uy: 1,
    uz: 0,
    tx: 1,
    ty: 0,
    tz: 0,
  };

  // 2. Populate arc-length stratified shingled rows from branch collar down to lower teardrop
  for (let r = 0; r <= rowCount; r++) {
    const targetArc = (totalMeridianArc * r) / rowCount;
    const baseTheta = sampleMeridianTheta(targetArc);

    ringArc[0] = 0;
    evalCurtainPoint(baseTheta, 0, rpPrev);
    for (let s = 1; s <= RING_SAMPLES; s++) {
      const phi = (s / RING_SAMPLES) * Math.PI * 2;
      evalCurtainPoint(baseTheta, phi, rpCur);
      ringArc[s] =
        ringArc[s - 1] +
        Math.hypot(rpCur[0] - rpPrev[0], rpCur[1] - rpPrev[1], rpCur[2] - rpPrev[2]);
      rpPrev[0] = rpCur[0];
      rpPrev[1] = rpCur[1];
      rpPrev[2] = rpCur[2];
    }

    const ringPerim = ringArc[RING_SAMPLES];
    const nRow = Math.max(5, Math.round(ringPerim / colPitch));
    const stepArc = ringPerim / nRow;
    const rowOffset = (r % 2) * 0.5 * stepArc;

    for (let j = 0; j < nRow; j++) {
      const jitterArc = (rng() - 0.5) * 0.14 * stepArc;
      let sTarget = (j * stepArc + rowOffset + jitterArc) % ringPerim;
      if (sTarget < 0) sTarget += ringPerim;

      // Binary search ring arc-length -> exact phi
      let lo = 0;
      let hi = RING_SAMPLES;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (ringArc[mid] < sTarget) lo = mid + 1;
        else hi = mid;
      }
      const idx = Math.max(1, lo);
      const a0 = ringArc[idx - 1];
      const a1 = ringArc[idx];
      const frac = a1 > a0 ? (sTarget - a0) / (a1 - a0) : 0;
      const phi = ((idx - 1 + frac) / RING_SAMPLES) * Math.PI * 2;
      const thetaJittered = Math.max(
        thetaMin,
        Math.min(thetaMax + 0.015, baseTheta + (rng() - 0.5) * 0.008)
      );

      evalCurtainFrame(thetaJittered, phi, frameScratch);
      tryPlaceUnit(frameScratch);
    }
  }

  // Place bottom-crest terminal unit so the lower teardrop rim closes seamlessly
  evalCurtainFrame(Math.PI * 0.5 - 0.014, Math.PI * 0.5, frameScratch);
  tryPlaceUnit(frameScratch);

  const unitCount = posList.length / 3;
  const positions = new Float32Array(posList);
  const normals = new Float32Array(normList);
  const upDirs = new Float32Array(upList);
  const tangents = new Float32Array(tangList);
  const scales = new Float32Array(scaleList);
  const restTilts = new Float32Array(tiltList);
  const seeds = new Float32Array(seedList);
  const gains = new Float32Array(gainList);

  // 3. Build true 3D surface k-NN neighbor graph (k = 7, filtered by normal alignment)
  const K_TARGET = 7;
  const MAX_DEGREE = 10;
  const gridCell = 0.14;
  const spatialHash = new Map<string, number[]>();

  for (let i = 0; i < unitCount; i++) {
    const cx = Math.floor(positions[i * 3] / gridCell);
    const cy = Math.floor(positions[i * 3 + 1] / gridCell);
    const cz = Math.floor(positions[i * 3 + 2] / gridCell);
    const k = cellKey(cx, cy, cz);
    let list = spatialHash.get(k);
    if (!list) {
      list = [];
      spatialHash.set(k, list);
    }
    list.push(i);
  }

  const adjSets: Array<Map<number, number>> = Array.from({ length: unitCount }, () => new Map());

  for (let i = 0; i < unitCount; i++) {
    const ix = positions[i * 3];
    const iy = positions[i * 3 + 1];
    const iz = positions[i * 3 + 2];
    const inx = normals[i * 3];
    const iny = normals[i * 3 + 1];
    const inz = normals[i * 3 + 2];

    const cx = Math.floor(ix / gridCell);
    const cy = Math.floor(iy / gridCell);
    const cz = Math.floor(iz / gridCell);

    const strictCandidates: Array<{ id: number; dist: number }> = [];
    const relaxedCandidates: Array<{ id: number; dist: number }> = [];

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = spatialHash.get(cellKey(cx + dx, cy + dy, cz + dz));
          if (!bucket) continue;
          for (let b = 0; b < bucket.length; b++) {
            const j = bucket[b];
            if (j === i) continue;
            const ddx = positions[j * 3] - ix;
            const ddy = positions[j * 3 + 1] - iy;
            const ddz = positions[j * 3 + 2] - iz;
            const distSq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (distSq > 0.16 * 0.16) continue;

            const nDot =
              inx * normals[j * 3] +
              iny * normals[j * 3 + 1] +
              inz * normals[j * 3 + 2];
            const dist = Math.sqrt(distSq);

            if (nDot > 0.18) {
              strictCandidates.push({ id: j, dist });
            } else if (nDot > -0.25) {
              relaxedCandidates.push({ id: j, dist });
            }
          }
        }
      }
    }

    strictCandidates.sort((a, b) => a.dist - b.dist);
    const chosen = strictCandidates.slice(0, K_TARGET);
    if (chosen.length < 5) {
      relaxedCandidates.sort((a, b) => a.dist - b.dist);
      for (let rIdx = 0; rIdx < relaxedCandidates.length && chosen.length < K_TARGET; rIdx++) {
        chosen.push(relaxedCandidates[rIdx]);
      }
    }

    for (let k = 0; k < chosen.length; k++) {
      const c = chosen[k];
      adjSets[i].set(c.id, c.dist);
      // Symmetrize so wave propagation is reciprocal across curved boundaries
      if (!adjSets[c.id].has(i)) {
        adjSets[c.id].set(i, c.dist);
      }
    }
  }

  // Pack symmetrized adjacency into contiguous CSR TypedArrays
  const neighborOffsets = new Int32Array(unitCount + 1);
  let totalEdges = 0;
  let minDegree = Infinity;
  let maxDegree = 0;

  const trimmedLists: Array<Array<[number, number]>> = new Array(unitCount);
  for (let i = 0; i < unitCount; i++) {
    const entries = Array.from(adjSets[i].entries());
    if (entries.length > MAX_DEGREE) {
      entries.sort((a, b) => a[1] - b[1]);
      entries.length = MAX_DEGREE;
    }
    trimmedLists[i] = entries;
    neighborOffsets[i] = totalEdges;
    totalEdges += entries.length;
    if (entries.length < minDegree) minDegree = entries.length;
    if (entries.length > maxDegree) maxDegree = entries.length;
  }
  neighborOffsets[unitCount] = totalEdges;

  const neighborIds = new Int32Array(totalEdges);
  const neighborWeights = new Float32Array(totalEdges);
  const neighborDists = new Float32Array(totalEdges);

  let distSum = 0;
  for (let i = 0; i < unitCount; i++) {
    const list = trimmedLists[i];
    const start = neighborOffsets[i];
    let wSum = 0;
    for (let k = 0; k < list.length; k++) {
      const dist = list[k][1];
      const w = 1.0 / (dist + 0.018);
      wSum += w;
      distSum += dist;
      neighborIds[start + k] = list[k][0];
      neighborDists[start + k] = dist;
      neighborWeights[start + k] = w;
    }
    const invW = wSum > 0 ? 1.0 / wSum : 1.0;
    for (let k = 0; k < list.length; k++) {
      neighborWeights[start + k] *= invW;
    }
  }

  const meanSpacing = totalEdges > 0 ? distSum / totalEdges : colPitch;

  return {
    unitCount,
    positions,
    normals,
    upDirs,
    tangents,
    scales,
    restTilts,
    seeds,
    gains,
    neighborOffsets,
    neighborIds,
    neighborWeights,
    neighborDists,
    meanSpacing,
    minDegree: minDegree === Infinity ? 0 : minDegree,
    maxDegree,
  };
}

/**
 * Coupled Excitable-Media Defense-Wave Solver (*Apis dorsata* Shimmering)
 *
 * Strictly zero per-frame allocations inside `step(dtSec)`.
 */
export class HiveShimmerSimulation {
  readonly graph: HiveClusterGraph;

  // Per-unit excitable-media & biomechanical actuator state (zero-GC typed arrays)
  readonly state: Uint8Array;
  readonly excitation: Float32Array;
  readonly nextExcitation: Float32Array;
  readonly phaseTimer: Float32Array;
  readonly flipAngle: Float32Array;
  readonly flipVel: Float32Array;
  readonly wingSpread: Float32Array;
  readonly flashIntensity: Float32Array;
  readonly lastIgniteTime: Float32Array;
  /** Bitmask/counter of unique units ever reached since last reset or seed */
  readonly visitedFlag: Uint8Array;

  // Excitable-media physical constants
  static readonly FIXED_DT = 1 / 120;
  static readonly FIRING_DURATION = 0.22; // Crisp ~180-220 ms abdomen-flip flash
  static readonly TRANSMIT_WINDOW = 0.165; // Excitatory transmission window into resting neighbors
  static readonly REFRACTORY_DURATION = 0.56; // Impassable cooldown -> clean wave annihilation!
  static readonly COUPLING_RATE = 96.0; // Produces ~0.58-0.78 m/s biological wave front
  static readonly LEAK_RATE = 6.2; // Sub-threshold noise decays rapidly back to rest
  static readonly PEAK_FLIP_RAD = 1.15; // ~66 deg upward/outward abdomen defense snap (55-75 deg range)

  // Coupled physical pendulum recoil of the hanging bivouac beneath the branch
  swayX = 0;
  swayZ = 0;
  swayVelX = 0;
  swayVelZ = 0;

  // Telemetry & simulation clock
  simTime = 0;
  activeFiringCount = 0;
  refractoryCount = 0;
  peakWaveSpeed = 0;
  totalWavesTriggered = 0;

  private stepAccum = 0;
  private lastWaveCountTime = -10;
  private frontSpeedEma = 0;

  constructor(graph?: HiveClusterGraph) {
    this.graph = graph ?? buildHiveClusterGraph();
    const n = this.graph.unitCount;

    this.state = new Uint8Array(n);
    this.excitation = new Float32Array(n);
    this.nextExcitation = new Float32Array(n);
    this.phaseTimer = new Float32Array(n);
    this.flipAngle = new Float32Array(n);
    this.flipVel = new Float32Array(n);
    this.wingSpread = new Float32Array(n);
    this.flashIntensity = new Float32Array(n);
    this.lastIgniteTime = new Float32Array(n);
    this.lastIgniteTime.fill(-10);
    this.visitedFlag = new Uint8Array(n);
  }

  /**
   * Immediately ignites a single unit index (if resting, or forced) and records wave trigger.
   */
  igniteUnit(idx: number, impulseBoost = 1.0, force = false): boolean {
    if (idx < 0 || idx >= this.graph.unitCount) return false;
    if (!force && this.state[idx] !== UNIT_STATE_RESTING) return false;

    this.state[idx] = UNIT_STATE_FIRING;
    this.excitation[idx] = 1.0;
    this.phaseTimer[idx] = 0;
    this.lastIgniteTime[idx] = this.simTime;
    this.visitedFlag[idx] = 1;
    // Crisp initial snap impulse on the abdomen hinge
    this.flipVel[idx] = Math.max(this.flipVel[idx], (5.6 + 2.2 * impulseBoost) * this.graph.gains[idx]);
    this.wingSpread[idx] = Math.min(1.0, Math.max(this.wingSpread[idx], 0.72 * impulseBoost));
    return true;
  }

  /**
   * Ignites a compact seed patch around a curtain-local 3D point (hover predator sweep or click).
   * Returns number of newly ignited units.
   */
  exciteAtPoint(
    lx: number,
    ly: number,
    lz: number,
    radius = 0.145,
    impulseBoost = 1.0,
    recoilStrength = 0.08
  ): number {
    const pos = this.graph.positions;
    const norm = this.graph.normals;
    const n = this.graph.unitCount;
    const rSq = radius * radius;
    let ignited = 0;
    let avgNx = 0;
    let avgNz = 0;

    for (let i = 0; i < n; i++) {
      const dx = pos[i * 3] - lx;
      const dy = pos[i * 3 + 1] - ly;
      const dz = pos[i * 3 + 2] - lz;
      const dSq = dx * dx + dy * dy + dz * dz;
      if (dSq <= rSq) {
        if (this.igniteUnit(i, impulseBoost, false)) {
          ignited++;
          avgNx += norm[i * 3];
          avgNz += norm[i * 3 + 2];
        }
      }
    }

    if (ignited > 0) {
      if (this.simTime - this.lastWaveCountTime > 0.14) {
        this.totalWavesTriggered++;
        this.lastWaveCountTime = this.simTime;
      }
      // Subtle physical recoil opposite to the triggered surface normal
      const inv = 1.0 / ignited;
      this.swayVelZ += -avgNx * inv * recoilStrength * 0.55;
      this.swayVelX += avgNz * inv * recoilStrength * 0.65;
    }

    return ignited;
  }

  /**
   * High-amplitude click shock-shimmer: ignites a strong radial burst & imparts weighty recoil.
   */
  triggerShockWave(lx: number, ly: number, lz: number): number {
    const count = this.exciteAtPoint(lx, ly, lz, 0.24, 1.38, 0.34);
    if (count === 0) {
      // Even if local patch was mid-refractory, find nearest resting unit and ignite
      const nearest = this.findNearestUnit(lx, ly, lz, true);
      if (nearest !== -1) {
        const p = this.graph.positions;
        return this.exciteAtPoint(
          p[nearest * 3],
          p[nearest * 3 + 1],
          p[nearest * 3 + 2],
          0.22,
          1.35,
          0.28
        );
      }
    }
    return count;
  }

  /**
   * Ignites twin opposing waves on the left and right flanks so they race across both faces
   * of the hanging bivouac and annihilate cleanly on collision down the centerline!
   */
  triggerTwinCollision(): number {
    this.quietAll();
    const left = this.findNearestUnit(-0.84, -0.34, 0.24, false);
    const right = this.findNearestUnit(0.84, -0.34, 0.24, false);
    let total = 0;
    const p = this.graph.positions;
    if (left !== -1) {
      total += this.exciteAtPoint(p[left * 3], p[left * 3 + 1], p[left * 3 + 2], 0.17, 1.25, 0.16);
    }
    if (right !== -1) {
      this.lastWaveCountTime = -10; // Count both opposing poles distinctly or cleanly
      total += this.exciteAtPoint(
        p[right * 3],
        p[right * 3 + 1],
        p[right * 3 + 2],
        0.17,
        1.25,
        0.16
      );
    }
    return total;
  }

  /**
   * Triggers a classic excitable-media broken-wavefront spiral seed on the front cheek:
   * a firing strip backed immediately by a refractory strip curls into a self-sustaining spiral.
   */
  triggerSpiralSeed(): number {
    this.quietAll();
    const pos = this.graph.positions;
    const norm = this.graph.normals;
    const n = this.graph.unitCount;
    let ignited = 0;

    for (let i = 0; i < n; i++) {
      // Front face only
      if (norm[i * 3 + 2] < 0.35) continue;
      const x = pos[i * 3];
      const y = pos[i * 3 + 1];
      // Vertical half-line from center downward: left strip refractory, right strip firing
      if (y > -1.05 && y < -0.38) {
        if (x >= -0.02 && x <= 0.13) {
          if (this.igniteUnit(i, 1.15, true)) ignited++;
        } else if (x >= -0.22 && x < -0.02) {
          this.state[i] = UNIT_STATE_REFRACTORY;
          this.phaseTimer[i] = 0.04;
          this.excitation[i] = 0;
        }
      }
    }

    if (ignited > 0) {
      this.totalWavesTriggered++;
      this.lastWaveCountTime = this.simTime;
      this.swayVelX += 0.14;
    }
    return ignited;
  }

  /**
   * Smoothly hushes all active & refractory units back to calm resting posture.
   */
  quietAll(): void {
    const n = this.graph.unitCount;
    for (let i = 0; i < n; i++) {
      this.state[i] = UNIT_STATE_RESTING;
      this.excitation[i] = 0;
      this.nextExcitation[i] = 0;
      this.phaseTimer[i] = 0;
    }
    this.activeFiringCount = 0;
    this.refractoryCount = 0;
  }

  /**
   * Resets visited tracking (for testing wave coverage from a fresh seed).
   */
  clearVisited(): void {
    this.visitedFlag.fill(0);
  }

  /**
   * Returns fraction of units visited since last `clearVisited()`.
   */
  getVisitedFraction(): number {
    const n = this.graph.unitCount;
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (this.visitedFlag[i] !== 0) count++;
    }
    return n > 0 ? count / n : 0;
  }

  /**
   * Finds nearest unit index to curtain-local coordinates (optionally requiring RESTING state).
   */
  findNearestUnit(lx: number, ly: number, lz: number, restingOnly = false): number {
    const pos = this.graph.positions;
    const n = this.graph.unitCount;
    let bestIdx = -1;
    let bestDistSq = Infinity;

    for (let i = 0; i < n; i++) {
      if (restingOnly && this.state[i] !== UNIT_STATE_RESTING) continue;
      const dx = pos[i * 3] - lx;
      const dy = pos[i * 3 + 1] - ly;
      const dz = pos[i * 3 + 2] - lz;
      const dSq = dx * dx + dy * dy + dz * dz;
      if (dSq < bestDistSq) {
        bestDistSq = dSq;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /**
   * Advances excitable-media wave physics and biomechanical abdomen actuators
   * by `dtSec` using fixed-dt substeps (120 Hz) with zero heap allocations.
   */
  step(dtSec: number): void {
    const clampedDt = Math.max(0, Math.min(0.1, dtSec));
    this.stepAccum += clampedDt;
    const fixedDt = HiveShimmerSimulation.FIXED_DT;
    let substeps = 0;

    while (this.stepAccum >= fixedDt && substeps < 12) {
      this.stepSub(fixedDt);
      this.stepAccum -= fixedDt;
      substeps++;
    }
  }

  private stepSub(dt: number): void {
    this.simTime += dt;

    const g = this.graph;
    const n = g.unitCount;
    const offsets = g.neighborOffsets;
    const nIds = g.neighborIds;
    const nWeights = g.neighborWeights;
    const nDists = g.neighborDists;
    const norms = g.normals;
    const gains = g.gains;

    const state = this.state;
    const exc = this.excitation;
    const nextExc = this.nextExcitation;
    const timer = this.phaseTimer;
    const flip = this.flipAngle;
    const flipV = this.flipVel;
    const wings = this.wingSpread;
    const flash = this.flashIntensity;
    const lastIgn = this.lastIgniteTime;
    const visited = this.visitedFlag;

    const fireDur = HiveShimmerSimulation.FIRING_DURATION;
    const txWin = HiveShimmerSimulation.TRANSMIT_WINDOW;
    const refDur = HiveShimmerSimulation.REFRACTORY_DURATION;
    const coupling = HiveShimmerSimulation.COUPLING_RATE;
    const leak = HiveShimmerSimulation.LEAK_RATE;
    const peakFlip = HiveShimmerSimulation.PEAK_FLIP_RAD;

    // Pass 1: Synchronous Jacobi neighbor coupling into resting membranes
    for (let i = 0; i < n; i++) {
      if (state[i] !== UNIT_STATE_RESTING) {
        nextExc[i] = state[i] === UNIT_STATE_FIRING ? 1.0 : 0.0;
        continue;
      }

      let drive = 0;
      const start = offsets[i];
      const end = offsets[i + 1];
      for (let e = start; e < end; e++) {
        const j = nIds[e];
        if (state[j] === UNIT_STATE_FIRING && timer[j] <= txWin) {
          // Smooth bell transmit profile peaking early in abdomen snap
          const txFrac = timer[j] / txWin;
          const pulse = 1.0 - 0.35 * txFrac;
          drive += nWeights[e] * pulse;
        }
      }

      const v = exc[i] + (coupling * drive - leak * exc[i]) * dt;
      nextExc[i] = v > 0 ? v : 0;
    }

    // Pass 2: Phase transitions, wave-speed sampling, biomechanical abdomen hinge & recoil
    let firingCount = 0;
    let refractoryCount = 0;
    let recoilTorqueX = 0;
    let recoilTorqueZ = 0;

    for (let i = 0; i < n; i++) {
      let st = state[i];

      if (st === UNIT_STATE_RESTING) {
        const v = nextExc[i];
        if (v >= 1.0) {
          // Ignite! Measure true perpendicular wave-front speed from the upstream driver
          // (earliest-ignited active neighbor inside transmit window, avoiding transverse
          // wavefront-sibling phase-velocity artifacts).
          let driverDt = 0;
          let driverDist = 0;
          const start = offsets[i];
          const end = offsets[i + 1];
          for (let e = start; e < end; e++) {
            const j = nIds[e];
            if (state[j] === UNIT_STATE_FIRING && lastIgn[j] >= 0) {
              const hopDt = this.simTime - lastIgn[j];
              if (hopDt >= 0.015 && hopDt <= txWin && hopDt > driverDt) {
                driverDt = hopDt;
                driverDist = nDists[e];
              }
            }
          }
          if (driverDist > 0 && driverDt > 0) {
            const speedMps = (driverDist * WORLD_TO_METERS) / driverDt;
            this.frontSpeedEma =
              this.frontSpeedEma <= 0
                ? speedMps
                : this.frontSpeedEma + (speedMps - this.frontSpeedEma) * 0.12;
            if (this.frontSpeedEma > this.peakWaveSpeed) {
              this.peakWaveSpeed = this.frontSpeedEma;
            }
          }

          st = UNIT_STATE_FIRING;
          state[i] = UNIT_STATE_FIRING;
          exc[i] = 1.0;
          timer[i] = 0;
          lastIgn[i] = this.simTime;
          visited[i] = 1;
          // Crisp biomechanical muscle snap-up
          flipV[i] += 6.0 * gains[i];
          wings[i] = Math.min(1.0, wings[i] + 0.78);
        } else {
          exc[i] = v;
        }
      } else if (st === UNIT_STATE_FIRING) {
        timer[i] += dt;
        if (timer[i] >= fireDur) {
          st = UNIT_STATE_REFRACTORY;
          state[i] = UNIT_STATE_REFRACTORY;
          timer[i] = 0;
          exc[i] = 0;
        }
      } else {
        // UNIT_STATE_REFRACTORY
        timer[i] += dt;
        exc[i] = 0;
        if (timer[i] >= refDur) {
          st = UNIT_STATE_RESTING;
          state[i] = UNIT_STATE_RESTING;
          timer[i] = 0;
        }
      }

      if (st === UNIT_STATE_FIRING) {
        firingCount++;
      } else if (st === UNIT_STATE_REFRACTORY) {
        refractoryCount++;
      }

      // Asymmetric biomechanical abdomen hinge spring:
      // Fast snap-up (~170 ms) to ~65-75 deg when firing, graceful damped spring return when refractory/resting
      const targetAngle = st === UNIT_STATE_FIRING ? peakFlip * gains[i] : 0.0;
      const kSpring = st === UNIT_STATE_FIRING ? 275.0 : 84.0;
      const cDamp = st === UNIT_STATE_FIRING ? 21.5 : 12.2;

      const accel = (targetAngle - flip[i]) * kSpring - flipV[i] * cDamp;
      flipV[i] += accel * dt;
      flip[i] += flipV[i] * dt;
      if (flip[i] < -0.04) {
        flip[i] = -0.04;
        if (flipV[i] < 0) flipV[i] = -flipV[i] * 0.25;
      } else if (flip[i] > 1.42) {
        flip[i] = 1.42;
        if (flipV[i] > 0) flipV[i] = 0;
      }

      // Accumulate net physical reaction torque on the hanging bivouac pendulum from rapid flips
      if (accel > 2.0 || flipV[i] > 1.0) {
        const impulse = Math.max(0, flipV[i]) * 0.000018;
        recoilTorqueZ += -norms[i * 3] * impulse;
        recoilTorqueX += norms[i * 3 + 2] * impulse;
      }

      // Gossamer wing flick envelope: sharp peak during snap-up, crisp settle back into folded chevron
      const wingTarget = st === UNIT_STATE_FIRING ? 0.92 : 0.0;
      const wingRate = st === UNIT_STATE_FIRING ? 24.0 : 8.5;
      wings[i] += (wingTarget - wings[i]) * Math.min(1.0, dt * wingRate);
      if (wings[i] < 1e-4) wings[i] = 0;

      // Optical shimmer drive smoothly tracks live flip posture + firing energy
      const flipNorm = Math.max(0, Math.min(1.0, flip[i] / (peakFlip * 0.92)));
      const flashTarget = Math.min(
        1.0,
        flipNorm * 0.88 + (st === UNIT_STATE_FIRING ? 0.22 : 0.0)
      );
      flash[i] += (flashTarget - flash[i]) * Math.min(1.0, dt * 22.0);
    }

    this.activeFiringCount = firingCount;
    this.refractoryCount = refractoryCount;

    // 2-axis damped physical pendulum sway for the heavy hanging curtain
    const kPend = 7.4;
    const cPend = 1.85;
    this.swayVelX += (recoilTorqueX - this.swayX * kPend - this.swayVelX * cPend) * dt;
    this.swayVelZ += (recoilTorqueZ - this.swayZ * kPend - this.swayVelZ * cPend) * dt;
    this.swayX = Math.max(-0.14, Math.min(0.14, this.swayX + this.swayVelX * dt));
    this.swayZ = Math.max(-0.14, Math.min(0.14, this.swayZ + this.swayVelZ * dt));
  }

  getTelemetry(): HiveTelemetry {
    return {
      unitCount: this.graph.unitCount,
      activeFiringCount: this.activeFiringCount,
      refractoryCount: this.refractoryCount,
      peakWaveSpeed: Number(this.peakWaveSpeed.toFixed(3)),
      totalWavesTriggered: this.totalWavesTriggered,
      clusterSwayRad: Number(Math.hypot(this.swayX, this.swayZ).toFixed(4)),
    };
  }
}
