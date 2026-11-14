import { describe, expect, it } from 'vitest';
import { RubberBandBallSim } from '../bandPhysics';
import { PigmentKey, SEGMENTS_PER_BAND } from '../types';

const ALLOWED_PIGMENTS = new Set<PigmentKey>([
  'rawUmber',
  'yellowOchre',
  'boneCalcite',
  'cinnabarVermilion',
  'burntSienna',
  'celadonGlauconite',
  'azuritePrussian',
]);

describe('RubberBandBallSim', () => {
  it('generates ~180 layered great-circle bands with orthonormal frames and mineral pigments', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const diag = sim.getDiagnostics();

    expect(specs.length).toBe(180);
    expect(runtimes.length).toBe(180);
    expect(diag.totalBands).toBe(180);
    expect(diag.intactCount).toBe(180);
    expect(diag.ballRadius).toBeGreaterThan(0.9);
    expect(diag.ballRadius).toBeLessThan(1.0);

    for (let i = 0; i < specs.length; i++) {
      const s = specs[i];
      expect(ALLOWED_PIGMENTS.has(s.pigment)).toBe(true);
      expect(s.baseRadius).toBeGreaterThan(0.28);
      expect(s.baseRadius).toBeLessThan(0.98);
      expect(s.restStrain).toBeLessThan(s.breakStrain);

      // Orthonormal frame check
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

  it('raycasts onto an intact outer band and misses when aimed off-ball', () => {
    const sim = new RubberBandBallSim();
    const hit = sim.raycastBand([0, 0, 4], [0, 0, -1]);
    expect(hit).not.toBeNull();
    if (hit) {
      expect(hit.bandId).toBeGreaterThanOrEqual(0);
      expect(hit.bandId).toBeLessThan(180);
      expect(hit.uHit).toBeGreaterThanOrEqual(0);
      expect(hit.uHit).toBeLessThan(1);
      const hitR = Math.hypot(...hit.hitPoint);
      expect(hitR).toBeGreaterThan(0.5);
    }

    const miss = sim.raycastBand([0, 3.5, 4], [0, 0, -1]);
    expect(miss).toBeNull();
  });

  it('pulls and snaps on tangential/XY drag from a front-face [0, 0, 1] hit point', () => {
    const sim = new RubberBandBallSim();
    const bandId = 179;
    const spec = sim.getBandSpecs()[bandId];
    const rt = sim.getBandRuntimes()[bandId];

    sim.startPull(bandId, 0.1, [0, 0, 1]);
    const snappedEarly = sim.updatePull([0.12, 0.04, 0]);
    expect(snappedEarly).toBe(false);
    expect(rt.pullAmount).toBeGreaterThan(0.1);
    expect(rt.currentStrain).toBeGreaterThan(spec.restStrain);

    // Drag further in XY until it snaps
    const snappedFar = sim.updatePull([0.9, 0.3, 0]);
    expect(snappedFar).toBe(true);
    expect(rt.status).toBe('snapping');
    expect(sim.getDiagnostics().snapCount).toBe(1);
  });

  it('supports pull -> release damped harmonic twang without snapping', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const bandId = 179;
    const spec = specs[bandId];

    // Outward direction at u = 0 is tangentU
    sim.startPull(bandId, 0, [...spec.tangentU]);
    const snapped = sim.updatePull([
      spec.tangentU[0] * 0.12,
      spec.tangentU[1] * 0.12,
      spec.tangentU[2] * 0.12,
    ]);
    expect(snapped).toBe(false);

    const rt = sim.getBandRuntimes()[bandId];
    expect(rt.status).toBe('pulled');
    expect(rt.currentStrain).toBeGreaterThan(spec.restStrain);

    // Centerline at u=0 should be pulled outward beyond baseRadius
    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);
    const segs = sim.sampleBandCenterline(bandId, posBuf, strainBuf);
    expect(segs).toBe(SEGMENTS_PER_BAND);
    const pulledR = Math.hypot(posBuf[0], posBuf[1], posBuf[2]);
    expect(pulledR).toBeGreaterThan(spec.baseRadius + 0.08);

    // Release -> twang
    sim.releasePull();
    expect(rt.status).toBe('intact');
    expect(rt.twangAmp).toBeGreaterThan(0.08);

    // Step forward 1.2s -> twang damps out
    for (let i = 0; i < 60; i++) {
      sim.step(0.02);
    }
    expect(rt.twangAmp).toBeLessThan(0.005);
    expect(rt.currentStrain).toBeCloseTo(spec.restStrain, 2);
  });

  it('snaps band when pulled past breakStrain and spawns dust puffs', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const bandId = 115;
    const spec = specs[bandId];

    sim.startPull(bandId, 0.25, [...spec.tangentV]);
    const snapped = sim.updatePull([
      spec.tangentV[0] * 0.9,
      spec.tangentV[1] * 0.9,
      spec.tangentV[2] * 0.9,
    ]);
    expect(snapped).toBe(true);

    const rt = sim.getBandRuntimes()[bandId];
    expect(rt.status).toBe('snapping');
    expect(sim.getDiagnostics().snapCount).toBe(1);
    expect(sim.getDiagnostics().activePullBandId).toBeNull();

    const activePuffs = sim.getDustPuffs().filter((p) => p.active);
    expect(activePuffs.length).toBeGreaterThan(0);
  });

  it('nickBand initiates whip-back, C-curl, gravity tumble, and transitions to gone', () => {
    const sim = new RubberBandBallSim();
    const bandId = 118;
    const rt = sim.getBandRuntimes()[bandId];

    sim.nickBand(bandId, 0.4);
    expect(rt.status).toBe('snapping');
    expect(rt.leftArc).toBe(0.5);
    expect(rt.rightArc).toBe(0.5);

    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);

    // Step 0.1s -> ends should be retreating and curling
    for (let i = 0; i < 5; i++) {
      sim.step(0.02);
    }
    expect(rt.leftArc).toBeLessThan(0.5);
    expect(rt.rightArc).toBeLessThan(0.5);
    expect(rt.curlAmount).toBeGreaterThan(0);

    // Step 2.0s -> band should fade out and become 'gone'
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

    // Nick top 50 outer bands
    for (let id = 179; id >= 130; id--) {
      sim.nickBand(id, 0.1);
    }

    // Step enough for breathing relaxation and any sympathetic cascades
    for (let i = 0; i < 30; i++) {
      sim.step(0.02);
    }

    sim.sampleBandCenterline(innerBandId, posBuf, strainBuf);
    const relaxedInnerRadius = Math.hypot(posBuf[0], posBuf[1], posBuf[2]);
    expect(relaxedInnerRadius).toBeGreaterThan(initialInnerRadius + 0.005);

    // Sympathetic cascade test: boost a neighbor of band 170 near breakStrain and nick 170
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const target = specs[170];
    let neighborId = -1;
    for (let i = 150; i < 180; i++) {
      if (i === 170 || runtimes[i].status !== 'intact') continue;
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
      // Over-strain neighbor right below break threshold
      (runtimes[neighborId] as { currentStrain: number }).currentStrain =
        specs[neighborId].breakStrain * 0.95;
      sim.nickBand(170, 0.2);
      for (let i = 0; i < 15; i++) {
        sim.step(0.02);
      }
      expect(
        runtimes[neighborId].status === 'snapping' || runtimes[neighborId].status === 'gone'
      ).toBe(true);
    }
  });

  it('reset() re-wraps a fresh 180-band ball', () => {
    const sim = new RubberBandBallSim();
    sim.nickBand(179, 0.5);
    sim.nickBand(178, 0.2);
    sim.step(0.05);
    expect(sim.getDiagnostics().snapCount).toBeGreaterThanOrEqual(2);

    sim.reset();
    const diag = sim.getDiagnostics();
    expect(diag.intactCount).toBe(180);
    expect(diag.snappingCount).toBe(0);
    expect(diag.goneCount).toBe(0);
    expect(diag.snapCount).toBe(0);
    expect(sim.getDustPuffs().every((p) => !p.active)).toBe(true);
  });

  it('enforces zero core or crossing-layer intersection across all 180 intact bands', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);

    // Cache sampled positions for all bands
    const allPos: Float32Array[] = [];
    for (let i = 0; i < specs.length; i++) {
      const buf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
      sim.sampleBandCenterline(i, buf, strainBuf);
      allPos.push(buf);

      const halfHI = specs[i].thickness * 0.5;
      for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
        const r = Math.hypot(buf[k * 3], buf[k * 3 + 1], buf[k * 3 + 2]);
        // Must clear starter core + inner knots (~0.286)
        expect(r - halfHI).toBeGreaterThanOrEqual(0.29);
      }
    }

    // Verify no outer band i intersects an underlying band j < i where their widths cross
    for (let i = 1; i < specs.length; i++) {
      const specI = specs[i];
      const halfHI = specI.thickness * 0.5;
      const bufI = allPos[i];

      for (let j = 0; j < i; j++) {
        const specJ = specs[j];
        const halfHJ = specJ.thickness * 0.5;
        const halfWJ = specJ.width * 0.5;
        const bufJ = allPos[j];

        for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
          const px = bufI[k * 3];
          const py = bufI[k * 3 + 1];
          const pz = bufI[k * 3 + 2];
          const rI = Math.hypot(px, py, pz);
          const ux = px / rI;
          const uy = py / rI;
          const uz = pz / rI;

          const latDist = Math.abs(
            ux * specJ.normal[0] + uy * specJ.normal[1] + uz * specJ.normal[2]
          ) * rI;
          if (latDist > halfWJ) continue;

          // Find closest vertex on band j
          const pu = ux * specJ.tangentU[0] + uy * specJ.tangentU[1] + uz * specJ.tangentU[2];
          const pv = ux * specJ.tangentV[0] + uy * specJ.tangentV[1] + uz * specJ.tangentV[2];
          let ang = Math.atan2(pv, pu);
          if (ang < 0) ang += Math.PI * 2;
          const kJ = Math.round((ang / (Math.PI * 2)) * SEGMENTS_PER_BAND) % SEGMENTS_PER_BAND;
          const rJ = Math.hypot(bufJ[kJ * 3], bufJ[kJ * 3 + 1], bufJ[kJ * 3 + 2]);

          // Outer band inner surface must clear underlying band outer surface
          expect(rI - halfHI).toBeGreaterThanOrEqual(rJ + halfHJ - 0.002);
        }
      }
    }

    // Silence unused variable warning
    expect(posBuf.length).toBeGreaterThan(0);
  });

  it('prevents pulled and snapping bands from clipping into the ball or core', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const bandId = 175;
    const spec = specs[bandId];
    const posBuf = new Float32Array((SEGMENTS_PER_BAND + 1) * 3);
    const strainBuf = new Float32Array(SEGMENTS_PER_BAND + 1);

    // Sample resting profile
    sim.sampleBandCenterline(bandId, posBuf, strainBuf);
    const restRadii = new Float32Array(SEGMENTS_PER_BAND + 1);
    for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
      restRadii[k] = Math.hypot(posBuf[k * 3], posBuf[k * 3 + 1], posBuf[k * 3 + 2]);
    }

    // Pull obliquely
    sim.startPull(bandId, 0.2, [...spec.tangentU]);
    sim.updatePull([
      spec.tangentU[0] * 0.25 + spec.normal[0] * 0.15,
      spec.tangentU[1] * 0.25 + spec.normal[1] * 0.15,
      spec.tangentU[2] * 0.25 + spec.normal[2] * 0.15,
    ]);
    sim.sampleBandCenterline(bandId, posBuf, strainBuf);
    for (let k = 0; k <= SEGMENTS_PER_BAND; k++) {
      const r = Math.hypot(posBuf[k * 3], posBuf[k * 3 + 1], posBuf[k * 3 + 2]);
      expect(r).toBeGreaterThanOrEqual(restRadii[k] - 1e-4);
    }

    // Snap and verify no vertex penetrates the ball core or band radius during whip-back
    sim.nickBand(bandId, 0.2);
    for (let step = 0; step < 15; step++) {
      sim.step(0.02);
      const segs = sim.sampleBandCenterline(bandId, posBuf, strainBuf);
      for (let k = 0; k <= segs; k++) {
        const r = Math.hypot(posBuf[k * 3], posBuf[k * 3 + 1], posBuf[k * 3 + 2]);
        expect(r).toBeGreaterThanOrEqual(spec.baseRadius - 1e-3);
      }
    }
  });

  it('delivers energetic snap kinematics: fast whip-back, rapid curl, ball jolt, neighbor twang, and talc burst', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const bandId = 179;
    const spec = specs[bandId];

    // Pull to snap
    sim.startPull(bandId, 0.0, [...spec.tangentU]);
    const snapped = sim.updatePull([
      spec.tangentU[0] * 0.95,
      spec.tangentU[1] * 0.95,
      spec.tangentU[2] * 0.95,
    ]);
    expect(snapped).toBe(true);

    const rt = runtimes[bandId];
    expect(rt.leftVel).toBeGreaterThan(3.0);
    expect(rt.rightVel).toBeGreaterThan(3.0);

    // Dense talcum puff burst (>= 30 active puffs)
    const activePuffs = sim.getDustPuffs().filter((p) => p.active);
    expect(activePuffs.length).toBeGreaterThanOrEqual(30);

    // Sympathetic neighbor twang
    let maxNeighborTwang = 0;
    for (let i = 140; i < 179; i++) {
      if (runtimes[i].twangAmp > maxNeighborTwang) {
        maxNeighborTwang = runtimes[i].twangAmp;
      }
    }
    expect(maxNeighborTwang).toBeGreaterThan(0.04);

    // Track ball recoil peak and rapid curl over first 0.12s
    let maxRecoil = 0;
    for (let i = 0; i < 6; i++) {
      sim.step(0.02);
      const rec = sim.getBallRecoil();
      const mag = Math.hypot(rec[0], rec[1], rec[2]);
      if (mag > maxRecoil) maxRecoil = mag;
    }
    expect(maxRecoil).toBeGreaterThan(0.025);
    expect(rt.curlAmount).toBeGreaterThan(0.5);
  });
});
