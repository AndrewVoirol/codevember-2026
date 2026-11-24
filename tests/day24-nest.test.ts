import { beforeAll, describe, expect, it } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  auditLayoutClearances,
  generateReciprocalDomeLayout,
  type ReciprocalDomeLayout,
} from '../src/days/24-nest/reciprocalDome';
import {
  buildDowelGeometry,
  buildPebbleEggGeometry,
  buildPlinthBrassInnerBezelGeometry,
  buildPlinthBrassOuterLipGeometry,
  buildPlinthLimestoneGeometry,
  computeSignedVolume,
} from '../src/days/24-nest/nestGeometry';
import { createNestMaterials } from '../src/days/24-nest/nestMaterials';
import { ReciprocalNestSim } from '../src/days/24-nest/nestSimulation';

let sharedLayout: ReciprocalDomeLayout;

beforeAll(async () => {
  await RAPIER.init();
  sharedLayout = generateReciprocalDomeLayout();
});

describe('Day 24 - Nest: Reciprocal-Frame Timber Dome (Geometry, Equilibrium & Collapse)', () => {
  it('generates 48..64 non-overlapping reciprocal timber dowels + 3 eggs with outward triangle winding', () => {
    expect(sharedLayout.rods.length).toBeGreaterThanOrEqual(48);
    expect(sharedLayout.rods.length).toBeLessThanOrEqual(64);
    expect(sharedLayout.rods.length).toBe(56);
    expect(sharedLayout.eggs.length).toBe(3);

    // Zero initial interpenetration across ALL rod-rod, egg-rod, and egg-egg pairs
    const audit = auditLayoutClearances(sharedLayout.rods, sharedLayout.eggs);
    expect(audit.minRodClearance).toBeGreaterThanOrEqual(0);
    expect(audit.minEggRodClearance).toBeGreaterThanOrEqual(0);
    expect(audit.minEggEggClearance).toBeGreaterThanOrEqual(0);
    expect(sharedLayout.minRodClearance).toBeGreaterThanOrEqual(0);

    // Every closed solid geometry must have strictly positive signed volume (outward CCW winding)
    const plinthStone = buildPlinthLimestoneGeometry(sharedLayout.plinth);
    const plinthLip = buildPlinthBrassOuterLipGeometry(sharedLayout.plinth);
    const plinthInner = buildPlinthBrassInnerBezelGeometry(sharedLayout.plinth);
    expect(computeSignedVolume(plinthStone)).toBeGreaterThan(0);
    expect(computeSignedVolume(plinthLip)).toBeGreaterThan(0);
    expect(computeSignedVolume(plinthInner)).toBeGreaterThan(0);
    plinthStone.dispose();
    plinthLip.dispose();
    plinthInner.dispose();

    // Every tier dowel lathe must have positive signed volume, notch mask, and valid longitudinal UVs
    let cursor = 0;
    for (const tierMeta of sharedLayout.tiers) {
      const sampleRod = sharedLayout.rods[cursor];
      const dowelGeom = buildDowelGeometry(
        tierMeta.halfHeight,
        tierMeta.radius,
        sampleRod.notchParams,
        24
      );
      expect(computeSignedVolume(dowelGeom)).toBeGreaterThan(0);
      const uvAttr = dowelGeom.getAttribute('uv');
      const notchAttr = dowelGeom.getAttribute('aNotchMask');
      expect(uvAttr).toBeDefined();
      expect(notchAttr).toBeDefined();
      let maxNotch = 0;
      for (let i = 0; i < notchAttr.count; i++) {
        const v = uvAttr.getY(i);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        const n = notchAttr.getX(i);
        if (n > maxNotch) maxNotch = n;
      }
      expect(maxNotch).toBeGreaterThan(0.5);
      dowelGeom.dispose();
      cursor += tierMeta.count;
    }

    // All 3 Celadon river-pebble egg geometries must be welded indexed meshes (silky smooth normals) with outward winding
    for (const egg of sharedLayout.eggs) {
      const eggGeom = buildPebbleEggGeometry(egg.radius, egg.visualScale, egg.seed, 4);
      expect(computeSignedVolume(eggGeom)).toBeGreaterThan(0);
      const eggIdx = eggGeom.getIndex();
      expect(eggIdx).not.toBeNull();
      expect(eggGeom.getAttribute('position').count).toBeLessThan(eggIdx!.count / 3);
      eggGeom.dispose();
    }

    // Verify all TSL NodeMaterials construct cleanly with mineral pigments & dispose cleanly
    const mats = createNestMaterials(sharedLayout.eggs);
    expect(mats.timberMat.colorNode).toBeDefined();
    expect(mats.timberMat.normalNode).toBeDefined();
    expect(mats.limestoneMat.colorNode).toBeDefined();
    expect(mats.brassMat.colorNode).toBeDefined();
    expect(mats.eggMats.length).toBe(3);
    mats.dispose();
  });

  it('Boot Equilibrium Gate: self-supports in real Rapier 3D gravity for 3.0 s (180 frames) and 10.0 s (600 frames) with 0 fallen rods', () => {
    const sim = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      for (let f = 0; f < 180; f++) {
        sim.stepFixedFrame();
      }
      const state3s = sim.getState();
      const maxDrop3s = sim.getMaxRodDrop();
      const eggY3s = sim.getEggPositionsY();

      expect(state3s.intactRodCount).toBe(56);
      expect(state3s.fallenRodCount).toBe(0);
      expect(state3s.collapsed).toBe(false);
      expect(maxDrop3s).toBeLessThan(0.08);
      expect(state3s.simTime).toBeCloseTo(3.0, 2);
      for (const y of eggY3s) {
        expect(y).toBeGreaterThan(0.55);
      }

      // Continue to 10 full seconds (600 total frames) proving zero creep or delayed collapse
      for (let f = 0; f < 420; f++) {
        sim.stepFixedFrame();
      }
      const state10s = sim.getState();
      const maxDrop10s = sim.getMaxRodDrop();
      const eggY10s = sim.getEggPositionsY();

      expect(state10s.intactRodCount).toBe(56);
      expect(state10s.fallenRodCount).toBe(0);
      expect(state10s.collapsed).toBe(false);
      expect(maxDrop10s).toBeLessThan(0.08);
      expect(Math.abs(maxDrop10s - maxDrop3s)).toBeLessThan(0.005);
      for (const y of eggY10s) {
        expect(y).toBeGreaterThan(0.55);
      }
    } finally {
      sim.dispose();
    }
  });

  it('Ablated Solver Gate: remains genuinely self-supporting even with minimal damping (0.02) and single-substep 6-iter solver', () => {
    const ablated = new ReciprocalNestSim(RAPIER, sharedLayout, {
      substepsPer60Hz: 1,
      solverIterations: 6,
      internalPgsIterations: 1,
      allowedLinearError: 0.001,
      linearDamping: 0.02,
      angularDamping: 0.02,
    });
    try {
      for (let f = 0; f < 180; f++) {
        ablated.stepFixedFrame();
      }
      const state = ablated.getState();
      expect(state.fallenRodCount).toBe(0);
      expect(state.collapsed).toBe(false);
      expect(ablated.getMaxRodDrop()).toBeLessThan(0.08);
      for (const y of ablated.getEggPositionsY()) {
        expect(y).toBeGreaterThan(0.55);
      }
    } finally {
      ablated.dispose();
    }
  });

  it('Adversarial Keystone-Pull Collapse Gate: pulling 3 or 4 lower-ring rods causes genuine dynamic structural collapse and drops all Celadon eggs', () => {
    // 1. Remove 3 adjacent Tier-0 load-bearing rods after settling briefly
    const sim3 = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      for (let f = 0; f < 30; f++) sim3.stepFixedFrame();
      sim3.removeRods([0, 1, 2]);
      for (let f = 0; f < 180; f++) sim3.stepFixedFrame();

      const state3 = sim3.getState();
      const eggDrops3 = sim3.getEggDrops();
      expect(state3.fallenRodCount).toBeGreaterThanOrEqual(8);
      expect(state3.collapsed).toBe(true);
      expect(sim3.getMaxRodDrop()).toBeGreaterThan(1.0);
      for (const drop of eggDrops3) {
        expect(drop).toBeGreaterThan(1.0);
      }
    } finally {
      sim3.dispose();
    }

    // 2. Remove 4 load-bearing Tier-0 rods directly at boot
    const sim4 = new ReciprocalNestSim(RAPIER, sharedLayout, {
      removedRodIds: [0, 1, 2, 3],
    });
    try {
      for (let f = 0; f < 180; f++) sim4.stepFixedFrame();

      const state4 = sim4.getState();
      const eggDrops4 = sim4.getEggDrops();
      expect(state4.fallenRodCount).toBeGreaterThanOrEqual(8);
      expect(state4.collapsed).toBe(true);
      expect(sim4.getMaxRodDrop()).toBeGreaterThan(5.0);
      for (const drop of eggDrops4) {
        expect(drop).toBeGreaterThan(5.0);
      }
    } finally {
      sim4.dispose();
    }
  });

  it('Interactive 3D Grab-Spring Extraction, Quick Tap Poke, and Reset restore equilibrium cleanly', () => {
    const sim = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      // Settle briefly
      for (let f = 0; f < 20; f++) sim.stepFixedFrame();

      // Quick tap poke on a dowel imparts real impulse without toppling intact dome
      const rod0 = sim.rods[0];
      expect(
        sim.pokeRod(0, rod0.spec.center, [0.5, 0.0, 0.86], 0.008)
      ).toBe(true);
      sim.stepFixedFrame();
      const velAfterPoke = rod0.body.linvel();
      expect(Math.hypot(velAfterPoke.x, velAfterPoke.y, velAfterPoke.z)).toBeGreaterThan(0.01);

      // Grab rod 0 via 3D camera-plane spring tether and pull it outward cleanly
      const startPos = rod0.spec.center;
      const pullTarget: [number, number, number] = [
        startPos[0] * 1.9,
        startPos[1] + 0.35,
        startPos[2] * 1.9,
      ];
      expect(sim.startGrab(0, [0, 0, 0], pullTarget)).toBe(true);
      expect(sim.getState().activeGrabId).toBe(0);

      for (let f = 0; f < 60; f++) {
        sim.stepFixedFrame();
      }
      expect(sim.rods[0].drift).toBeGreaterThan(0.35);

      sim.endGrab();
      expect(sim.getState().activeGrabId).toBeNull();

      // Reset cleanly re-weaves the intact 56-rod dome and restores rock-solid equilibrium
      sim.reset();
      const resetState = sim.getState();
      expect(resetState.intactRodCount).toBe(56);
      expect(resetState.fallenRodCount).toBe(0);
      expect(resetState.collapsed).toBe(false);
      expect(resetState.maxRodDrift).toBe(0);
      expect(resetState.activeGrabId).toBeNull();

      for (let f = 0; f < 120; f++) sim.stepFixedFrame();
      expect(sim.getState().fallenRodCount).toBe(0);
      expect(sim.getMaxRodDrop()).toBeLessThan(0.08);
    } finally {
      sim.dispose();
    }
  });
});
