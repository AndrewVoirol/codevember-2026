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
  it('generates ~120 layered great-circle bands with orthonormal frames and mineral pigments', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const diag = sim.getDiagnostics();

    expect(specs.length).toBe(120);
    expect(runtimes.length).toBe(120);
    expect(diag.totalBands).toBe(120);
    expect(diag.intactCount).toBe(120);
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
      expect(hit.bandId).toBeLessThan(120);
      expect(hit.uHit).toBeGreaterThanOrEqual(0);
      expect(hit.uHit).toBeLessThan(1);
      const hitR = Math.hypot(...hit.hitPoint);
      expect(hitR).toBeGreaterThan(0.5);
    }

    const miss = sim.raycastBand([0, 3.5, 4], [0, 0, -1]);
    expect(miss).toBeNull();
  });

  it('supports pull -> release damped harmonic twang without snapping', () => {
    const sim = new RubberBandBallSim();
    const specs = sim.getBandSpecs();
    const bandId = 119;
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

    // Nick top 40 outer bands
    for (let id = 119; id >= 80; id--) {
      sim.nickBand(id, 0.1);
    }

    // Step enough for breathing relaxation and any sympathetic cascades
    for (let i = 0; i < 30; i++) {
      sim.step(0.02);
    }

    sim.sampleBandCenterline(innerBandId, posBuf, strainBuf);
    const relaxedInnerRadius = Math.hypot(posBuf[0], posBuf[1], posBuf[2]);
    expect(relaxedInnerRadius).toBeGreaterThan(initialInnerRadius + 0.005);

    // Sympathetic cascade test: boost a neighbor of band 110 near breakStrain and nick 110
    const specs = sim.getBandSpecs();
    const runtimes = sim.getBandRuntimes();
    const target = specs[110];
    let neighborId = -1;
    for (let i = 95; i < 120; i++) {
      if (i === 110 || runtimes[i].status !== 'intact') continue;
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
      sim.nickBand(110, 0.2);
      for (let i = 0; i < 15; i++) {
        sim.step(0.02);
      }
      expect(
        runtimes[neighborId].status === 'snapping' || runtimes[neighborId].status === 'gone'
      ).toBe(true);
    }
  });

  it('reset() re-wraps a fresh 120-band ball', () => {
    const sim = new RubberBandBallSim();
    sim.nickBand(119, 0.5);
    sim.nickBand(118, 0.2);
    sim.step(0.05);
    expect(sim.getDiagnostics().snapCount).toBeGreaterThanOrEqual(2);

    sim.reset();
    const diag = sim.getDiagnostics();
    expect(diag.intactCount).toBe(120);
    expect(diag.snappingCount).toBe(0);
    expect(diag.goneCount).toBe(0);
    expect(diag.snapCount).toBe(0);
    expect(sim.getDustPuffs().every((p) => !p.active)).toBe(true);
  });
});
