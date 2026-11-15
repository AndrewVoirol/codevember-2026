import { describe, expect, it } from 'vitest';
import { RubberBandBallSim } from '../bandPhysics';
import { BandSpec, CORE_RADIUS, PigmentKey, SEGMENTS_PER_BAND, TOTAL_BANDS } from '../types';

const ALLOWED_PIGMENTS = new Set<PigmentKey>([
  'rawUmber',
  'yellowOchre',
  'boneCalcite',
  'cinnabarVermilion',
  'burntSienna',
  'celadonGlauconite',
  'azuritePrussian',
]);

// Exact 8-vertex chamfered strap profile from rubberOptics.ts
const CORNER_R = [1.0, 1.0, 0.52, -0.52, -1.0, -1.0, -0.52, 0.52] as const;
const CORNER_B = [-0.78, 0.78, 1.0, 1.0, 0.78, -0.78, -1.0, -1.0] as const;

interface BandFrame {
  pts: Float32Array;
  strains: Float32Array;
  minR: number;
  maxR: number;
  verts: Float32Array; // (SEGMENTS_PER_BAND + 1) * 8 * 3
  segCenter: Float32Array; // SEGMENTS_PER_BAND * 3
  segT: Float32Array;
  segB: Float32Array;
  segR: Float32Array;
  segHalfL: Float32Array;
  segHalfW: Float32Array;
  segHalfH: Float32Array;
}

/**
 * Reconstructs exact 8-vertex chamfered 3D ribbon geometry matching rubberOptics.ts
 * and segment OBB frames so tests verify true 3D ribbon-edge and chord-midpoint clearance.
 */
function buildExactRibbonGeometry(sim: RubberBandBallSim, bandIdx: number): BandFrame {
  const spec = sim.getBandSpecs()[bandIdx];
  const pts = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
  const strains = new Float32Array(SEGMENTS_PER_BAND + 1);
  const segs = sim.sampleBandCenterline(bandIdx, pts, strains);

  const verts = new Float32Array((segs + 1) * 8 * 3);
  const halfWBase = spec.width * 0.5;
  const halfTBase = spec.thickness * 0.5;

  for (let r = 0; r <= segs; r++) {
    if (r === segs) {
      verts.set(verts.subarray(0, 8 * 3), r * 8 * 3);
      continue;
    }
    const px = pts[r * 3];
    const py = pts[r * 3 + 1];
    const pz = pts[r * 3 + 2];

    const prevIdx = (r > 0 ? r - 1 : segs - 1) * 3;
    const nextIdx = (r < segs ? r + 1 : 1) * 3;
    let tx = pts[nextIdx] - pts[prevIdx];
    let ty = pts[nextIdx + 1] - pts[prevIdx + 1];
    let tz = pts[nextIdx + 2] - pts[prevIdx + 2];
    const tLen = Math.hypot(tx, ty, tz) || 1;
    tx /= tLen;
    ty /= tLen;
    tz /= tLen;

    let rx = px;
    let ry = py;
    let rz = pz;
    const rLen = Math.hypot(rx, ry, rz) || 1;
    rx /= rLen;
    ry /= rLen;
    rz /= rLen;

    let bx = ry * tz - rz * ty;
    let by = rz * tx - rx * tz;
    let bz = rx * ty - ry * tx;
    let bLen = Math.hypot(bx, by, bz);
    if (bLen < 1e-4) {
      bx = spec.normal[0];
      by = spec.normal[1];
      bz = spec.normal[2];
      bLen = Math.hypot(bx, by, bz) || 1;
    }
    bx /= bLen;
    by /= bLen;
    bz /= bLen;

    rx = ty * bz - tz * by;
    ry = tz * bx - tx * bz;
    rz = tx * by - ty * bx;
    const rOrtho = Math.hypot(rx, ry, rz) || 1;
    rx /= rOrtho;
    ry /= rOrtho;
    rz /= rOrtho;

    const strain = strains[r];
    const extraStrain = strain > spec.restStrain ? strain - spec.restStrain : 0;
    const hw = halfWBase * Math.max(0.32, 1 / Math.sqrt(1 + extraStrain * 2.1));
    const ht = halfTBase * Math.max(0.45, 1 / Math.sqrt(1 + extraStrain * 1.2));

    for (let k = 0; k < 8; k++) {
      const v3 = (r * 8 + k) * 3;
      const cr = CORNER_R[k] * ht;
      const cb = CORNER_B[k] * hw;
      verts[v3] = px + rx * cr + bx * cb;
      verts[v3 + 1] = py + ry * cr + by * cb;
      verts[v3 + 2] = pz + rz * cr + bz * cb;
    }
  }

  const segCenter = new Float32Array(segs * 3);
  const segT = new Float32Array(segs * 3);
  const segB = new Float32Array(segs * 3);
  const segR = new Float32Array(segs * 3);
  const segHalfL = new Float32Array(segs);
  const segHalfW = new Float32Array(segs);
  const segHalfH = new Float32Array(segs);

  for (let s = 0; s < segs; s++) {
    const a3 = s * 3;
    const b3 = (s + 1) * 3;
    const mx = 0.5 * (pts[a3] + pts[b3]);
    const my = 0.5 * (pts[a3 + 1] + pts[b3 + 1]);
    const mz = 0.5 * (pts[a3 + 2] + pts[b3 + 2]);
    segCenter[a3] = mx;
    segCenter[a3 + 1] = my;
    segCenter[a3 + 2] = mz;

    let tx = pts[b3] - pts[a3];
    let ty = pts[b3 + 1] - pts[a3 + 1];
    let tz = pts[b3 + 2] - pts[a3 + 2];
    const segLen = Math.hypot(tx, ty, tz) || 1;
    tx /= segLen;
    ty /= segLen;
    tz /= segLen;
    segT[a3] = tx;
    segT[a3 + 1] = ty;
    segT[a3 + 2] = tz;

    let bx = my * tz - mz * ty;
    let by = mz * tx - mx * tz;
    let bz = mx * ty - my * tx;
    const bLen = Math.hypot(bx, by, bz) || 1;
    bx /= bLen;
    by /= bLen;
    bz /= bLen;
    segB[a3] = bx;
    segB[a3 + 1] = by;
    segB[a3 + 2] = bz;

    let rx = ty * bz - tz * by;
    let ry = tz * bx - tx * bz;
    let rz = tx * by - ty * bx;
    const rLen = Math.hypot(rx, ry, rz) || 1;
    rx /= rLen;
    ry /= rLen;
    rz /= rLen;
    segR[a3] = rx;
    segR[a3 + 1] = ry;
    segR[a3 + 2] = rz;

    const avgStrain = 0.5 * (strains[s] + strains[s + 1]);
    const extraStrain = avgStrain > spec.restStrain ? avgStrain - spec.restStrain : 0;
    segHalfL[s] = segLen * 0.5;
    segHalfW[s] = halfWBase * Math.max(0.32, 1 / Math.sqrt(1 + extraStrain * 2.1));
    segHalfH[s] = halfTBase * Math.max(0.45, 1 / Math.sqrt(1 + extraStrain * 1.2));
  }

  let minR = Infinity;
  let maxR = 0;
  for (let r = 0; r < segs; r++) {
    const rad = Math.hypot(pts[r * 3], pts[r * 3 + 1], pts[r * 3 + 2]);
    if (rad < minR) minR = rad;
    if (rad > maxR) maxR = rad;
  }

  return { pts, strains, minR, maxR, verts, segCenter, segT, segB, segR, segHalfL, segHalfW, segHalfH };
}

/**
 * Counts 3D penetration events across all (i, j) pairs with j < i, testing:
 *   - All 8 chamfered vertices of ring k
 *   - Segment chord midpoints (0.5*(ring k + ring k+1)) across full ribbon width
 * Checks if any inner/edge point of outer band i penetrates inside the 3D OBB volume of inner band j.
 */
function measureRibbonPenetrations(
  specs: readonly BandSpec[],
  frames: readonly BandFrame[]
): { count: number; maxDepth: number; worstPair: [number, number] | null } {
  let count = 0;
  let maxDepth = 0;
  let worstPair: [number, number] | null = null;

  for (let i = 1; i < specs.length; i++) {
    const fi = frames[i];
    const vertsI = fi.verts;
    const specI = specs[i];

    for (let j = 0; j < i; j++) {
      const fj = frames[j];
      const specJ = specs[j];
      if (fi.minR - specI.thickness > fj.maxR + specJ.thickness + 0.02) continue;

      const nj = specJ.normal;
      const halfWJPad = specJ.width * 0.5 + 0.015;

      for (let r = 0; r < SEGMENTS_PER_BAND; r++) {
        // Fast plane-slab reject for intact band j unless j itself is dynamically pulled/bowed
        const cx = fi.pts[r * 3];
        const cy = fi.pts[r * 3 + 1];
        const cz = fi.pts[r * 3 + 2];
        const slabDist = Math.abs(cx * nj[0] + cy * nj[1] + cz * nj[2]);
        if (slabDist > halfWJPad + specI.width * 0.6 + 0.04) continue;

        const rNext = r + 1;
        // Test all 8 vertices of ring r AND chord midpoints between ring r and r+1
        for (let mode = 0; mode < 2; mode++) {
          for (let v = 0; v < 8; v++) {
            const idx0 = (r * 8 + v) * 3;
            let px = vertsI[idx0];
            let py = vertsI[idx0 + 1];
            let pz = vertsI[idx0 + 2];
            if (mode === 1) {
              const idx1 = (rNext * 8 + v) * 3;
              px = 0.5 * (px + vertsI[idx1]);
              py = 0.5 * (py + vertsI[idx1 + 1]);
              pz = 0.5 * (pz + vertsI[idx1 + 2]);
            }

            const pRad = Math.hypot(px, py, pz);
            for (let s = 0; s < SEGMENTS_PER_BAND; s++) {
              const s3 = s * 3;
              const mx = fj.segCenter[s3];
              const my = fj.segCenter[s3 + 1];
              const mz = fj.segCenter[s3 + 2];
              const dx = px - mx;
              const dy = py - my;
              const dz = pz - mz;
              if (dx * dx + dy * dy + dz * dz > 0.012) continue;

              const locT = dx * fj.segT[s3] + dy * fj.segT[s3 + 1] + dz * fj.segT[s3 + 2];
              if (Math.abs(locT) >= fj.segHalfL[s]) continue;

              const locB = dx * fj.segB[s3] + dy * fj.segB[s3 + 1] + dz * fj.segB[s3 + 2];
              if (Math.abs(locB) >= fj.segHalfW[s]) continue;

              const locR = dx * fj.segR[s3] + dy * fj.segR[s3 + 1] + dz * fj.segR[s3 + 2];
              const mRad = Math.hypot(mx, my, mz);
              if (Math.abs(locR) < fj.segHalfH[s] && pRad <= mRad + fj.segHalfH[s]) {
                const depth = fj.segHalfH[s] - locR;
                if (depth > 1e-4) {
                  count++;
                  if (depth > maxDepth) {
                    maxDepth = depth;
                    worstPair = [i, j];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return { count, maxDepth, worstPair };
}

describe('RubberBandBallSim', () => {
  it('generates 200 tightly packed great-circle bands with orthonormal frames and mineral pigments', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const diag = sim.getDiagnostics();

    expect(specs.length).toBe(TOTAL_BANDS);
    expect(runtimes.length).toBe(TOTAL_BANDS);
    expect(diag.totalBands).toBe(TOTAL_BANDS);
    expect(diag.intactCount).toBe(TOTAL_BANDS);
    expect(diag.ballRadius).toBeGreaterThan(0.68);
    expect(diag.ballRadius).toBeLessThan(1.05);

    for (let i = 0; i < specs.length; i++) {
      const s = specs[i];
      expect(ALLOWED_PIGMENTS.has(s.pigment)).toBe(true);
      expect(s.baseRadius).toBeGreaterThan(CORE_RADIUS);
      expect(s.baseRadius).toBeLessThan(0.98);
      expect(s.thickness).toBeGreaterThanOrEqual(0.013);
      expect(s.thickness).toBeLessThanOrEqual(0.021);
      expect(s.restStrain).toBeLessThan(s.breakStrain);

      const nLen = Math.hypot(...s.normal);
      const uLen = Math.hypot(...s.tangentU);
      const vLen = Math.hypot(...s.tangentV);
      expect(nLen).toBeCloseTo(1, 5);
      expect(uLen).toBeCloseTo(1, 5);
      expect(vLen).toBeCloseTo(1, 5);

      const dotNU = s.normal[0] * s.tangentU[0] + s.normal[1] * s.tangentU[1] + s.normal[2] * s.tangentU[2];
      const dotUV = s.tangentU[0] * s.tangentV[0] + s.tangentU[1] * s.tangentV[1] + s.tangentU[2] * s.tangentV[2];
      expect(Math.abs(dotNU)).toBeLessThan(1e-5);
      expect(Math.abs(dotUV)).toBeLessThan(1e-5);
    }
  });

  it('raycasts onto the true topmost wrapped band and misses off-ball', () => {
    const sim = new RubberBandBallSim();
    const hit = sim.raycastBand([0, 0, 4], [0, 0, -1]);
    expect(hit).not.toBeNull();
    if (hit) {
      expect(hit.bandId).toBeGreaterThanOrEqual(0);
      expect(hit.bandId).toBeLessThan(TOTAL_BANDS);
      expect(hit.uHit).toBeGreaterThanOrEqual(0);
      expect(hit.uHit).toBeLessThan(1);
      const hitR = Math.hypot(...hit.hitPoint);
      expect(hitR).toBeGreaterThan(0.45);
    }

    // Verify grid of rays across the front hemisphere always picks the topmost visible band
    const testCoords = [
      [0, 0],
      [0.25, 0.15],
      [-0.28, 0.22],
      [0.18, -0.32],
      [-0.35, -0.18],
    ] as const;
    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);

    for (const [rx, ry] of testCoords) {
      const h = sim.raycastBand([rx, ry, 4], [0, 0, -1]);
      expect(h).not.toBeNull();
      if (!h) continue;
      // Verify hitPoint sits on bandId's actual wrapped 3D ribbon
      sim.sampleBandCenterline(h.bandId, posBuf, strainBuf);
      const spec = sim.getBandSpecs()[h.bandId];
      let minDist = Infinity;
      for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
        const dx = h.hitPoint[0] - posBuf[k * 3];
        const dy = h.hitPoint[1] - posBuf[k * 3 + 1];
        const dz = h.hitPoint[2] - posBuf[k * 3 + 2];
        const d = Math.hypot(dx, dy, dz);
        if (d < minDist) minDist = d;
      }
      expect(minDist).toBeLessThan(spec.width * 0.65 + 0.02);
    }

    const miss = sim.raycastBand([0, 3.5, 4], [0, 0, -1]);
    expect(miss).toBeNull();
  });

  it('pulls and snaps ergonomically within 0.35-0.50 world units on front-face drag', () => {
    const sim = new RubberBandBallSim();
    const bandId = TOTAL_BANDS - 1;
    const spec = sim.getBandSpecs()[bandId];
    const rt = sim.getBandRuntimes()[bandId];

    sim.startPull(bandId, 0.1, [0, 0, 1]);
    const snappedEarly = sim.updatePull([0.12, 0.04, 0]);
    expect(snappedEarly).toBe(false);
    expect(rt.pullAmount).toBeGreaterThan(0.1);
    expect(rt.currentStrain).toBeGreaterThan(spec.restStrain);

    // Drag further within viewport (~0.40 total) until it breaks
    const snappedFar = sim.updatePull([0.38, 0.14, 0]);
    expect(snappedFar).toBe(true);
    expect(rt.status).toBe('snapping');
    expect(sim.getDiagnostics().snapCount).toBe(1);
  });

  it('supports pull -> release damped harmonic twang without snapping', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const bandId = TOTAL_BANDS - 1;
    const spec = specs[bandId];

    sim.startPull(bandId, 0, [...spec.tangentU]);
    const snapped = sim.updatePull([
      spec.tangentU[0] * 0.14,
      spec.tangentU[1] * 0.14,
      spec.tangentU[2] * 0.14,
    ]);
    expect(snapped).toBe(false);

    const rt = sim.getBandRuntimes()[bandId];
    expect(rt.status).toBe('pulled');
    expect(rt.currentStrain).toBeGreaterThan(spec.restStrain);

    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);
    const segs = sim.sampleBandCenterline(bandId, posBuf, strainBuf);
    expect(segs).toBe(SEGMENTS_PER_BAND);
    const pulledR = Math.hypot(posBuf[0], posBuf[1], posBuf[2]);
    expect(pulledR).toBeGreaterThan(spec.baseRadius + 0.08);

    sim.releasePull();
    expect(rt.status).toBe('intact');
    expect(rt.twangAmp).toBeGreaterThan(0.08);

    for (let i = 0; i < 65; i++) {
      sim.step(0.02);
    }
    expect(rt.twangAmp).toBeLessThan(0.005);
    expect(rt.currentStrain).toBeCloseTo(spec.restStrain, 2);
  });

  it('snaps band when pulled past breakStrain and spawns crisp fast talc micro-specks <= 0.32s', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const bandId = 135;
    const spec = specs[bandId];

    sim.startPull(bandId, 0.25, [...spec.tangentV]);
    const snapped = sim.updatePull([
      spec.tangentV[0] * 0.65,
      spec.tangentV[1] * 0.65,
      spec.tangentV[2] * 0.65,
    ]);
    expect(snapped).toBe(true);

    const rt = sim.getBandRuntimes()[bandId];
    expect(rt.status).toBe('snapping');
    expect(sim.getDiagnostics().snapCount).toBe(1);
    expect(sim.getDiagnostics().activePullBandId).toBeNull();

    const activePuffs = sim.getDustPuffs().filter((p) => p.active);
    const specks = activePuffs.filter((p) => p.isSpeck);
    expect(specks.length).toBeGreaterThanOrEqual(30);
    for (const p of activePuffs) {
      expect(p.maxAgeSec).toBeLessThanOrEqual(0.32);
    }

    // All talc particles clear completely within 0.34s (no lingering smoke)
    for (let i = 0; i < 18; i++) {
      sim.step(0.02);
    }
    expect(sim.getDustPuffs().every((p) => !p.active)).toBe(true);
  });

  it('nickBand initiates readable 150-250ms whip-back, natural C-loop curl (no radial stick), and settles to gone', () => {
    const sim = new RubberBandBallSim();
    const bandId = 138;
    const rt = sim.getBandRuntimes()[bandId];

    sim.nickBand(bandId, 0.4);
    expect(rt.status).toBe('snapping');
    expect(rt.leftArc).toBe(0.5);
    expect(rt.rightArc).toBe(0.5);
    expect(rt.leftVel).toBeGreaterThanOrEqual(1.9);
    expect(rt.leftVel).toBeLessThanOrEqual(3.0);

    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);

    // Step 0.08s -> ends are visibly retracting around the ball (NOT collapsed in 65ms!)
    for (let i = 0; i < 4; i++) {
      sim.step(0.02);
    }
    expect(rt.leftArc).toBeGreaterThan(0.18);
    expect(rt.leftArc).toBeLessThan(0.46);
    expect(rt.rightArc).toBeGreaterThan(0.18);
    expect(rt.rightArc).toBeLessThan(0.46);
    expect(rt.curlAmount).toBeGreaterThan(0.2);

    // Step to 0.18s -> curled tips stay compact spiral C-loops (not +0.62 rigid stick)
    for (let i = 0; i < 5; i++) {
      sim.step(0.02);
    }
    const segs = sim.sampleBandCenterline(bandId, posBuf, strainBuf);
    const fx = rt.fallOffset[0];
    const fy = rt.fallOffset[1];
    const fz = rt.fallOffset[2];
    let minLocalR = Infinity;
    let maxLocalR = 0;
    for (let k = 0; k <= segs; k++) {
      const lr = Math.hypot(posBuf[k * 3] - fx, posBuf[k * 3 + 1] - fy, posBuf[k * 3 + 2] - fz);
      if (lr < minLocalR) minLocalR = lr;
      if (lr > maxLocalR) maxLocalR = lr;
    }
    expect(maxLocalR - minLocalR).toBeLessThan(0.22);

    // Step 2.0s -> band fades out and becomes 'gone'
    for (let i = 0; i < 100; i++) {
      sim.step(0.02);
    }
    expect(rt.status).toBe('gone');
    expect(rt.opacity).toBe(0);
    expect(sim.sampleBandCenterline(bandId, posBuf, strainBuf)).toBe(0);
  });

  it('exhibits compressive breathing expansion and sympathetic cascade when outer bands pop', () => {
    const sim = new RubberBandBallSim();
    const innerBandId = 10;
    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);

    sim.sampleBandCenterline(innerBandId, posBuf, strainBuf);
    const initialInnerRadius = Math.hypot(posBuf[0], posBuf[1], posBuf[2]);

    for (let id = TOTAL_BANDS - 1; id >= TOTAL_BANDS - 60; id--) {
      sim.nickBand(id, 0.1);
    }

    for (let i = 0; i < 30; i++) {
      sim.step(0.02);
    }

    sim.sampleBandCenterline(innerBandId, posBuf, strainBuf);
    const relaxedInnerRadius = Math.hypot(posBuf[0], posBuf[1], posBuf[2]);
    expect(relaxedInnerRadius).toBeGreaterThan(initialInnerRadius + 0.005);

    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const targetId = TOTAL_BANDS - 10;
    const target = specs[targetId];
    let neighborId = -1;
    for (let i = TOTAL_BANDS - 35; i < TOTAL_BANDS; i++) {
      if (i === targetId || runtimes[i].status !== 'intact') continue;
      const align = Math.abs(
        specs[i].normal[0] * target.normal[0] +
          specs[i].normal[1] * target.normal[1] +
          specs[i].normal[2] * target.normal[2]
      );
      if (align > 0.6) {
        neighborId = i;
        break;
      }
    }
    if (neighborId !== -1) {
      (runtimes[neighborId] as { currentStrain: number }).currentStrain =
        specs[neighborId].breakStrain * 0.95;
      sim.nickBand(targetId, 0.2);
      for (let i = 0; i < 15; i++) {
        sim.step(0.02);
      }
      expect(
        runtimes[neighborId].status === 'snapping' || runtimes[neighborId].status === 'gone'
      ).toBe(true);
    }
  });

  it('reset() re-wraps a fresh 200-band ball and bumps bandVersion', () => {
    const sim = new RubberBandBallSim();
    const vBefore = sim.getBandVersion(0);
    sim.nickBand(TOTAL_BANDS - 1, 0.5);
    sim.nickBand(TOTAL_BANDS - 2, 0.2);
    sim.step(0.05);
    expect(sim.getDiagnostics().snapCount).toBeGreaterThanOrEqual(2);
    expect(sim.getLastSnapInfo()).not.toBeNull();

    sim.reset();
    const diag = sim.getDiagnostics();
    expect(diag.intactCount).toBe(TOTAL_BANDS);
    expect(diag.snappingCount).toBe(0);
    expect(diag.goneCount).toBe(0);
    expect(diag.snapCount).toBe(0);
    expect(sim.getLastSnapInfo()).toBeNull();
    expect(sim.getBandVersion(0)).toBeGreaterThan(vBefore);
    expect(sim.getDustPuffs().every((p) => !p.active)).toBe(true);
  });

  it('guarantees ZERO resting 3D ribbon-edge and chord-midpoint intersections across all 200 bands', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const frames: BandFrame[] = [];

    for (let i = 0; i < specs.length; i++) {
      const f = buildExactRibbonGeometry(sim, i);
      frames.push(f);
      const halfHI = specs[i].thickness * 0.5;
      for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
        const r = Math.hypot(f.pts[k * 3], f.pts[k * 3 + 1], f.pts[k * 3 + 2]);
        expect(r - halfHI).toBeGreaterThanOrEqual(CORE_RADIUS + 0.007);
      }
    }

    const result = measureRibbonPenetrations(specs, frames);
    expect(result.count).toBe(0);
    expect(result.maxDepth).toBe(0);
  });

  it('enforces zero pull-time intersections and bows overlying cross-bands upward cleanly', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();

    // Find a visible raycast-grabbed band that has overlying bands j > i crossing its flanks
    const hit = sim.raycastBand([0.1, 0.1, 4], [0, 0, -1]);
    expect(hit).not.toBeNull();
    if (!hit) return;

    // Also explicitly test pulling an upper-mid band (say id ~ 185) that definitely has overlying bands j > 185
    const testBandId = Math.min(hit.bandId, TOTAL_BANDS - 15);
    const testSpec = specs[testBandId];
    const restFrames: BandFrame[] = [];
    for (let i = 0; i < specs.length; i++) {
      restFrames.push(buildExactRibbonGeometry(sim, i));
    }

    const outDir = [...testSpec.tangentU] as [number, number, number];
    sim.startPull(testBandId, 0.0, outDir);
    const snapped = sim.updatePull([outDir[0] * 0.28, outDir[1] * 0.28, outDir[2] * 0.28]);
    expect(snapped).toBe(false);

    const pullFrames: BandFrame[] = [];
    let bowedOverlyingCount = 0;
    for (let i = 0; i < specs.length; i++) {
      const f = buildExactRibbonGeometry(sim, i);
      pullFrames.push(f);
      if (i > testBandId) {
        for (let k = 0; k < SEGMENTS_PER_BAND; k++) {
          const rRest = Math.hypot(
            restFrames[i].pts[k * 3],
            restFrames[i].pts[k * 3 + 1],
            restFrames[i].pts[k * 3 + 2]
          );
          const rPull = Math.hypot(f.pts[k * 3], f.pts[k * 3 + 1], f.pts[k * 3 + 2]);
          if (rPull > rRest + 0.01) {
            bowedOverlyingCount++;
            break;
          }
        }
      }
    }

    expect(bowedOverlyingCount).toBeGreaterThan(0);

    const pullPen = measureRibbonPenetrations(specs, pullFrames);
    expect(pullPen.count).toBe(0);
    expect(pullPen.maxDepth).toBe(0);

    // Release -> verify twang time also has zero penetrations and bandVersion settles when idle
    sim.releasePull();
    sim.step(0.016);
    const twangFrames: BandFrame[] = [];
    for (let i = 0; i < specs.length; i++) {
      twangFrames.push(buildExactRibbonGeometry(sim, i));
    }
    const twangPen = measureRibbonPenetrations(specs, twangFrames);
    expect(twangPen.count).toBe(0);

    for (let i = 0; i < 80; i++) {
      sim.step(0.02);
    }
    const vIdle1 = sim.getBandVersion(testBandId);
    sim.step(0.02);
    const vIdle2 = sim.getBandVersion(testBandId);
    expect(vIdle2).toBe(vIdle1);
  });

  it('delivers energetic snap kinematics, punchy recoil, snapInfo, and non-stick C-loop whip', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const bandId = TOTAL_BANDS - 1;
    const spec = specs[bandId];

    sim.startPull(bandId, 0.0, [...spec.tangentU]);
    const snapped = sim.updatePull([
      spec.tangentU[0] * 0.55,
      spec.tangentU[1] * 0.55,
      spec.tangentU[2] * 0.55,
    ]);
    expect(snapped).toBe(true);

    const rt = runtimes[bandId];
    expect(rt.leftVel).toBeGreaterThanOrEqual(2.0);
    expect(rt.leftVel).toBeLessThanOrEqual(3.2);
    expect(rt.rightVel).toBeGreaterThanOrEqual(2.0);
    expect(rt.rightVel).toBeLessThanOrEqual(3.2);

    const snapInfo = sim.getLastSnapInfo();
    expect(snapInfo).not.toBeNull();
    expect(snapInfo?.bandId).toBe(bandId);
    expect(snapInfo?.intensity).toBeGreaterThan(0.6);

    const activePuffs = sim.getDustPuffs().filter((p) => p.active);
    const speckPuffs = activePuffs.filter((p) => p.isSpeck);
    expect(speckPuffs.length).toBeGreaterThanOrEqual(30);
    for (const p of activePuffs) {
      expect(p.maxAgeSec).toBeLessThanOrEqual(0.32);
    }

    let maxNeighborTwang = 0;
    for (let i = TOTAL_BANDS - 45; i < bandId; i++) {
      if (runtimes[i].twangAmp > maxNeighborTwang) {
        maxNeighborTwang = runtimes[i].twangAmp;
      }
    }
    expect(maxNeighborTwang).toBeGreaterThan(0.04);

    let maxRecoil = 0;
    for (let i = 0; i < 8; i++) {
      sim.step(0.02);
      const rec = sim.getBallRecoil();
      const mag = Math.hypot(rec[0], rec[1], rec[2]);
      if (mag > maxRecoil) maxRecoil = mag;
    }
    expect(maxRecoil).toBeGreaterThan(0.05);
    expect(rt.curlAmount).toBeGreaterThan(0.6);
  });

  it('enforces zero penetrations during oblique tangential pulls over cross-bands', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();

    // Pick an exposed band covered by at least ~18 overlying bands so oblique lift stresses crossings
    const hit = sim.raycastBand([0.06, -0.04, 4], [0, 0, -1]);
    expect(hit).not.toBeNull();
    const testBandId = hit ? Math.min(hit.bandId, TOTAL_BANDS - 18) : TOTAL_BANDS - 18;
    const spec = specs[testBandId];
    const rt = runtimes[testBandId];

    const uHit = 0.15;
    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);
    sim.sampleBandCenterline(testBandId, posBuf, strainBuf);

    const k0 = Math.round(uHit * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
    const kPrev = (k0 - 1 + SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
    const kNext = (k0 + 1) % SEGMENTS_PER_BAND;

    const px = posBuf[k0 * 3];
    const py = posBuf[k0 * 3 + 1];
    const pz = posBuf[k0 * 3 + 2];
    const pLen = Math.hypot(px, py, pz) || 1;
    const nx = px / pLen;
    const ny = py / pLen;
    const nz = pz / pLen;

    let tx = posBuf[kNext * 3] - posBuf[kPrev * 3];
    let ty = posBuf[kNext * 3 + 1] - posBuf[kPrev * 3 + 1];
    let tz = posBuf[kNext * 3 + 2] - posBuf[kPrev * 3 + 2];
    const tLen = Math.hypot(tx, ty, tz) || 1;
    tx /= tLen;
    ty /= tLen;
    tz /= tLen;

    const bx = spec.normal[0];
    const by = spec.normal[1];
    const bz = spec.normal[2];

    sim.startPull(testBandId, uHit, [nx, ny, nz]);

    // Total incremental drag ~0.268 (< 0.35 break threshold), heavily tilted along tangent + binormal
    const stepCount = 5;
    const stepDelta: [number, number, number] = [
      (nx * 0.14 + tx * 0.22 + bx * 0.06) / stepCount,
      (ny * 0.14 + ty * 0.22 + by * 0.06) / stepCount,
      (nz * 0.14 + tz * 0.22 + bz * 0.06) / stepCount,
    ];

    for (let s = 0; s < stepCount; s++) {
      const snapped = sim.updatePull(stepDelta);
      expect(snapped).toBe(false);
      sim.step(0.016);
    }

    expect(rt.status).toBe('pulled');
    expect(rt.pullAmount).toBeGreaterThan(0.2);
    const dirDotNormal = rt.pullDir[0] * nx + rt.pullDir[1] * ny + rt.pullDir[2] * nz;
    expect(dirDotNormal).toBeLessThan(0.9);

    const pullFrames: BandFrame[] = [];
    for (let i = 0; i < specs.length; i++) {
      pullFrames.push(buildExactRibbonGeometry(sim, i));
    }

    const result = measureRibbonPenetrations(specs, pullFrames);
    expect(result.count).toBe(0);
    expect(result.maxDepth).toBe(0);
  });

  it('raycasts only the unoccluded frontmost band across multiple rays and never hits gone bands', () => {
    const rayIntersectsBandFrame = (
      origin: readonly [number, number, number],
      unitDir: readonly [number, number, number],
      frame: BandFrame
    ): number => {
      let minT = Infinity;
      const [ox, oy, oz] = origin;
      const [dx, dy, dz] = unitDir;

      for (let s = 0; s < SEGMENTS_PER_BAND; s++) {
        const s3 = s * 3;
        const omx = ox - frame.segCenter[s3];
        const omy = oy - frame.segCenter[s3 + 1];
        const omz = oz - frame.segCenter[s3 + 2];

        let tMin = 0;
        let tMax = minT;
        let missed = false;

        const axes = [
          [frame.segT[s3], frame.segT[s3 + 1], frame.segT[s3 + 2], frame.segHalfL[s] + 0.0005],
          [frame.segB[s3], frame.segB[s3 + 1], frame.segB[s3 + 2], frame.segHalfW[s]],
          [frame.segR[s3], frame.segR[s3 + 1], frame.segR[s3 + 2], frame.segHalfH[s]],
        ] as const;

        for (let a = 0; a < 3; a++) {
          const [ax, ay, az, halfExt] = axes[a];
          const oProj = omx * ax + omy * ay + omz * az;
          const dProj = dx * ax + dy * ay + dz * az;
          if (Math.abs(dProj) < 1e-8) {
            if (Math.abs(oProj) > halfExt) {
              missed = true;
              break;
            }
          } else {
            let t1 = (-halfExt - oProj) / dProj;
            let t2 = (halfExt - oProj) / dProj;
            if (t1 > t2) {
              const tmp = t1;
              t1 = t2;
              t2 = tmp;
            }
            if (t1 > tMin) tMin = t1;
            if (t2 < tMax) tMax = t2;
            if (tMin > tMax) {
              missed = true;
              break;
            }
          }
        }

        if (!missed && tMin > 0 && tMin < minT) {
          minT = tMin;
        }
      }

      return minT;
    };

    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();

    const buildIntactFrames = (): (BandFrame | null)[] => {
      const frames: (BandFrame | null)[] = [];
      for (let i = 0; i < specs.length; i++) {
        frames.push(runtimes[i].status === 'intact' ? buildExactRibbonGeometry(sim, i) : null);
      }
      return frames;
    };

    const verifyFrontmostHit = (
      origin: readonly [number, number, number],
      dir: readonly [number, number, number],
      frames: readonly (BandFrame | null)[]
    ) => {
      const dLen = Math.hypot(dir[0], dir[1], dir[2]);
      const unitDir: [number, number, number] = [dir[0] / dLen, dir[1] / dLen, dir[2] / dLen];
      const hit = sim.raycastBand([origin[0], origin[1], origin[2]], [dir[0], dir[1], dir[2]]);
      expect(hit).not.toBeNull();
      if (!hit) return null;

      expect(runtimes[hit.bandId].status).toBe('intact');

      const hitT =
        (hit.hitPoint[0] - origin[0]) * unitDir[0] +
        (hit.hitPoint[1] - origin[1]) * unitDir[1] +
        (hit.hitPoint[2] - origin[2]) * unitDir[2];

      let frontBandId = -1;
      let frontT = Infinity;
      for (let j = 0; j < frames.length; j++) {
        const f = frames[j];
        if (!f) continue;
        const tj = rayIntersectsBandFrame(origin, unitDir, f);
        if (tj < frontT) {
          frontT = tj;
          frontBandId = j;
        }
      }

      // Whenever an exact OBB ray hit exists, raycastBand must return that exact unoccluded frontmost band
      if (frontBandId !== -1) {
        expect(hit.bandId).toBe(frontBandId);
        expect(Math.abs(hitT - frontT)).toBeLessThan(1e-4);
      }

      return { hit, hitT };
    };

    const frames0 = buildIntactFrames();
    const probeRays: ReadonlyArray<readonly [[number, number, number], [number, number, number]]> = [
      [[0.0, 0.0, 4.0], [0, 0, -1]],
      [[0.16, 0.12, 4.0], [0, 0, -1]],
      [[-0.22, 0.14, 4.0], [0, 0, -1]],
      [[0.14, -0.24, 4.0], [0, 0, -1]],
      [[-0.18, -0.19, 4.0], [0, 0, -1]],
      [[2.5, 1.8, 2.5], [-2.5, -1.8, -2.5]],
      [[-2.2, 2.0, -2.6], [2.2, -2.0, 2.6]],
    ];

    const distinctBandIds = new Set<number>();
    for (const [origin, dir] of probeRays) {
      const res = verifyFrontmostHit(origin, dir, frames0);
      if (res) distinctBandIds.add(res.hit.bandId);
    }
    expect(distinctBandIds.size).toBeGreaterThanOrEqual(4);

    // Peel off two successive bands along the center ray and verify occlusion clears strictly to deeper intact bands
    const centerOrigin: [number, number, number] = [0.04, 0.03, 4.0];
    const centerDir: [number, number, number] = [0, 0, -1];
    const removedBands: number[] = [];

    for (let layer = 0; layer < 2; layer++) {
      const preFrames = buildIntactFrames();
      const firstRes = verifyFrontmostHit(centerOrigin, centerDir, preFrames);
      expect(firstRes).not.toBeNull();
      if (!firstRes) break;

      const targetId = firstRes.hit.bandId;
      expect(removedBands).not.toContain(targetId);
      removedBands.push(targetId);

      sim.nickBand(targetId, firstRes.hit.uHit);
      for (let s = 0; s < 40; s++) {
        sim.step(0.02);
      }
      expect(runtimes[targetId].status).toBe('gone');

      const postFrames = buildIntactFrames();
      const nextRes = verifyFrontmostHit(centerOrigin, centerDir, postFrames);
      expect(nextRes).not.toBeNull();
      if (nextRes) {
        expect(nextRes.hit.bandId).not.toBe(targetId);
        expect(removedBands).not.toContain(nextRes.hit.bandId);
      }
    }
  });
});
