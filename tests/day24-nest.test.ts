import { beforeAll, describe, expect, it } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  DOWEL_CHAMFER_FRAC,
  auditLayoutClearances,
  buildOvoidPebbleHullPoints,
  computeAllRodContactStations,
  generateReciprocalDomeLayout,
  pointToRoundCylinderSignedDist,
  rotateVecByQuat,
  segmentSegmentClosest,
  type ReciprocalDomeLayout,
} from '../src/days/24-nest/reciprocalDome';
import {
  buildDowelGeometry,
  buildDustMoteGeometry,
  buildPebbleEggGeometry,
  buildPlinthBrassInnerBezelGeometry,
  buildPlinthBrassOuterLipGeometry,
  buildPlinthLimestoneGeometry,
  computeSignedVolume,
} from '../src/days/24-nest/nestGeometry';
import { createNestMaterials } from '../src/days/24-nest/nestMaterials';
import {
  MAX_DUST_MOTES,
  NestDustPool,
  NestReweaveController,
  REWEAVE_DURATION_SEC,
  ReciprocalNestSim,
} from '../src/days/24-nest/nestSimulation';

let sharedLayout: ReciprocalDomeLayout;

beforeAll(async () => {
  await RAPIER.init();
  sharedLayout = generateReciprocalDomeLayout();
});

function getSettledRodWorld(rec: ReciprocalNestSim['rods'][number]) {
  const tr = rec.body.translation();
  const q = rec.body.rotation();
  const hh = rec.spec.halfHeight;
  const vx = 2 * (q.x * q.y - q.w * q.z);
  const vy = 1 - 2 * (q.x * q.x + q.z * q.z);
  const vz = 2 * (q.y * q.z + q.w * q.x);
  return {
    center: [tr.x, tr.y, tr.z] as readonly [number, number, number],
    axis: [vx, vy, vz] as readonly [number, number, number],
    a: [tr.x - vx * hh, tr.y - vy * hh, tr.z - vz * hh] as readonly [number, number, number],
    b: [tr.x + vx * hh, tr.y + vy * hh, tr.z + vz * hh] as readonly [number, number, number],
    hh,
    r: rec.spec.radius,
    tier: rec.spec.tier,
  };
}

describe('Day 24 - Nest: Reciprocal-Frame Timber Dome (Geometry, Equilibrium & Collapse)', () => {
  it('generates 48..64 non-overlapping reciprocal timber dowels + 3 ovoid eggs with outward triangle winding', () => {
    expect(sharedLayout.rods.length).toBeGreaterThanOrEqual(48);
    expect(sharedLayout.rods.length).toBeLessThanOrEqual(64);
    expect(sharedLayout.rods.length).toBe(56);
    expect(sharedLayout.eggs.length).toBe(3);

    // Zero initial interpenetration across ALL rod-rod, ovoid egg-rod, and egg-egg pairs
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

    // Every tier dowel lathe must have positive signed volume, flat chamfered end-grain envelope,
    // exact full-radius shaft parity, un-aliased notch mask, and valid longitudinal UVs
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
      const posAttr = dowelGeom.getAttribute('position');
      const uvAttr = dowelGeom.getAttribute('uv');
      const notchAttr = dowelGeom.getAttribute('aNotchMask');
      expect(uvAttr).toBeDefined();
      expect(notchAttr).toBeDefined();

      const expectedHalfLen = tierMeta.halfHeight + tierMeta.radius * DOWEL_CHAMFER_FRAC;
      let maxY = -Infinity;
      let maxR = 0;
      let maxNotch = 0;
      for (let i = 0; i < posAttr.count; i++) {
        const y = posAttr.getY(i);
        const r = Math.hypot(posAttr.getX(i), posAttr.getZ(i));
        if (y > maxY) maxY = y;
        if (r > maxR) maxR = r;
        const v = uvAttr.getY(i);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        const n = notchAttr.getX(i);
        if (n > maxNotch) maxNotch = n;
      }
      // Chamfered flat end-grain matches roundCylinder half-length within 0.2 mm
      expect(Math.abs(maxY - expectedHalfLen)).toBeLessThan(0.0002);
      // Shaft radius matches Rapier collider outer radius within 0.2 mm (zero air-gap taper)
      expect(Math.abs(maxR - tierMeta.radius)).toBeLessThan(0.0002);
      expect(maxNotch).toBeGreaterThan(0.9);
      dowelGeom.dispose();
      cursor += tierMeta.count;
    }

    // All 3 ovoid Celadon & Bone-Calcite river-pebble egg geometries must be welded indexed meshes
    // with outward winding and genuine ovoid (non-spherical) aspect ratios + distinct mineral mix
    const mixes = sharedLayout.eggs.map((e) => e.celadonMix);
    expect(Math.min(...mixes)).toBeLessThanOrEqual(0.15);
    expect(Math.max(...mixes)).toBeGreaterThanOrEqual(0.85);
    for (const egg of sharedLayout.eggs) {
      expect(egg.visualScale[0] / egg.visualScale[1]).toBeGreaterThan(1.2);
      const eggGeom = buildPebbleEggGeometry(egg.radius, egg.visualScale, egg.seed, 4);
      expect(computeSignedVolume(eggGeom)).toBeGreaterThan(0);
      const eggIdx = eggGeom.getIndex();
      expect(eggIdx).not.toBeNull();
      expect(eggGeom.getAttribute('position').count).toBeLessThan(eggIdx!.count / 3);
      eggGeom.dispose();
    }

    // Verify all TSL NodeMaterials construct cleanly with strict zero hover tint at init
    const mats = createNestMaterials(sharedLayout.eggs);
    expect(mats.timberMat.colorNode).toBeDefined();
    expect(mats.timberMat.normalNode).toBeDefined();
    expect(mats.limestoneMat.colorNode).toBeDefined();
    expect(mats.brassMat.colorNode).toBeDefined();
    expect(mats.eggMats.length).toBe(3);
    expect(mats.uHoveredRodId.value).toBe(-1);
    expect(mats.uGrabbedRodId.value).toBe(-1);
    expect(mats.uHoverLift.value).toBe(0);
    mats.dispose();
  });

  it('Plinth Hollow-Bore & Non-Overlapping Seam Parity: inner brass bezel sits flush <= 0 and zero limestone/brass overlap', () => {
    const p = sharedLayout.plinth;
    const stone = buildPlinthLimestoneGeometry(p);
    const outerLip = buildPlinthBrassOuterLipGeometry(p);
    const innerBezel = buildPlinthBrassInnerBezelGeometry(p);

    const radialExtents = (geom: ReturnType<typeof buildPlinthLimestoneGeometry>, yMinFilter = -Infinity) => {
      const pos = geom.getAttribute('position');
      let minR = Infinity;
      let maxR = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (let i = 0; i < pos.count; i++) {
        const y = pos.getY(i);
        if (y < yMinFilter) continue;
        const r = Math.hypot(pos.getX(i), pos.getZ(i));
        if (r < minR) minR = r;
        if (r > maxR) maxR = r;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      return { minR, maxR, minY, maxY };
    };

    try {
      const innerExt = radialExtents(innerBezel);
      const outerExt = radialExtents(outerLip);
      const stoneUpperExt = radialExtents(stone, -0.043);
      const stoneAllExt = radialExtents(stone);

      // Hollow center bore matches Rapier innerRadius (1.12 m) with zero solid floor blocking the hole
      expect(innerExt.minR).toBeGreaterThanOrEqual(p.innerRadius - 0.001);
      expect(stoneAllExt.minR).toBeGreaterThan(innerExt.maxR);

      // Inner brass bezel sits strictly flush at y <= 0.0 so Tier-0 dowels never clip a proud lip
      expect(innerExt.maxY).toBeLessThanOrEqual(1e-5);
      expect(stoneAllExt.maxY).toBeLessThanOrEqual(1e-5);

      // Zero interpenetration between upper limestone seating shelf and outer brass retaining collar
      expect(stoneUpperExt.maxR).toBeLessThan(outerExt.minR);
      // Outer brass inner retaining wall matches Rapier seatOuterRadius to < 0.2 mm
      expect(Math.abs(outerExt.minR - p.seatOuterRadius)).toBeLessThan(0.0002);
    } finally {
      stone.dispose();
      outerLip.dispose();
      innerBezel.dispose();
    }
  });

  it('Boot Equilibrium, Sub-Millimeter Contact Parity & Zero-Jitter Stability Gate (2s, 3s, 10s)', () => {
    const sim = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      // Settle for 2.0 s (120 frames)
      for (let f = 0; f < 120; f++) {
        sim.stepFixedFrame();
      }

      // 1. Measure exact settled dowel-to-dowel barrel penetration & nearest-neighbor gap
      const rods = sim.rods.map(getSettledRodWorld);
      let maxBarrelPen = 0;
      let maxNearestGap = -Infinity;
      for (let i = 0; i < rods.length; i++) {
        let nearest = Infinity;
        for (let j = 0; j < rods.length; j++) {
          if (i === j) continue;
          const res = segmentSegmentClosest(rods[i].a, rods[i].b, rods[j].a, rods[j].b);
          const d = res.dist - (rods[i].r + rods[j].r);
          if (d < nearest) nearest = d;
          if (j > i && res.s > 0.03 && res.s < 0.97 && res.t > 0.03 && res.t < 0.97) {
            if (-d > maxBarrelPen) maxBarrelPen = -d;
          }
        }
        if (nearest > maxNearestGap) maxNearestGap = nearest;
      }
      // Sub-millimeter tangent contact: zero severe interpenetration, zero floating rods
      expect(maxBarrelPen).toBeLessThan(0.0012);
      expect(maxNearestGap).toBeLessThan(0.0006);

      // 2. Base Tier-0 chamfered dowels seat flush on the honed limestone shelf (y = 0)
      for (let i = 0; i < 16; i++) {
        const r = rods[i];
        const b = r.r * DOWEL_CHAMFER_FRAC;
        const sinPhi = Math.abs(r.axis[1]);
        const cosPhi = Math.sqrt(Math.max(0, 1 - sinPhi * sinPhi));
        const lowY = r.center[1] - r.hh * sinPhi - (r.r - b) * cosPhi - b;
        expect(lowY).toBeGreaterThan(-0.0015);
        expect(lowY).toBeLessThan(0.0015);
      }

      // 3. All 3 ovoid Celadon & Bone-Calcite eggs rest smoothly on the birch oculus (zero tunneling/hovering)
      for (const egg of sim.eggs) {
        const tr = egg.body.translation();
        const rq = egg.body.rotation();
        const hull = buildOvoidPebbleHullPoints(egg.spec.radius, egg.spec.visualScale, egg.spec.seed);
        let eggMinDist = Infinity;
        for (let v = 0; v < hull.length; v += 3) {
          const rv = rotateVecByQuat([hull[v], hull[v + 1], hull[v + 2]], rq);
          const wp: [number, number, number] = [tr.x + rv[0], tr.y + rv[1], tr.z + rv[2]];
          for (let rIdx = 0; rIdx < rods.length; rIdx++) {
            const rod = rods[rIdx];
            const d = pointToRoundCylinderSignedDist(wp, rod.center, rod.axis, rod.hh, rod.r);
            if (d < eggMinDist) eggMinDist = d;
          }
        }
        expect(eggMinDist).toBeGreaterThan(-0.0012);
        expect(eggMinDist).toBeLessThan(0.002);
      }

      // 4. Zero micro-jitter between t = 2.0 s and t = 3.0 s (60 frames)
      const prevPos = sim.rods.map((r) => {
        const t = r.body.translation();
        return [t.x, t.y, t.z] as [number, number, number];
      });
      let maxFrameDelta = 0;
      for (let f = 0; f < 60; f++) {
        sim.stepFixedFrame();
        for (let i = 0; i < sim.rods.length; i++) {
          const t = sim.rods[i].body.translation();
          const d = Math.hypot(t.x - prevPos[i][0], t.y - prevPos[i][1], t.z - prevPos[i][2]);
          if (d > maxFrameDelta) maxFrameDelta = d;
          prevPos[i] = [t.x, t.y, t.z];
        }
      }
      expect(maxFrameDelta).toBeLessThan(0.0001);

      const state3s = sim.getState();
      const maxDrop3s = sim.getMaxRodDrop();
      const eggY3s = sim.getEggPositionsY();

      expect(state3s.intactRodCount).toBe(56);
      expect(state3s.fallenRodCount).toBe(0);
      expect(state3s.collapsed).toBe(false);
      // Tight hardwood contact keeps total settling drop < 25 mm (vs 67 mm soft-rubber sag)
      expect(maxDrop3s).toBeLessThan(0.025);
      expect(state3s.simTime).toBeCloseTo(3.0, 2);
      for (const y of eggY3s) {
        expect(y).toBeGreaterThan(0.65);
      }

      // 5. Continue to 10 full seconds (600 total frames) proving zero creep or delayed collapse
      for (let f = 0; f < 420; f++) {
        sim.stepFixedFrame();
      }
      const state10s = sim.getState();
      const maxDrop10s = sim.getMaxRodDrop();
      const eggY10s = sim.getEggPositionsY();

      expect(state10s.intactRodCount).toBe(56);
      expect(state10s.fallenRodCount).toBe(0);
      expect(state10s.collapsed).toBe(false);
      expect(maxDrop10s).toBeLessThan(0.025);
      expect(Math.abs(maxDrop10s - maxDrop3s)).toBeLessThan(0.002);
      for (const y of eggY10s) {
        expect(y).toBeGreaterThan(0.65);
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

  it('Adversarial Keystone-Pull Collapse Gate: pulling 3 or 4 lower-ring rods causes genuine collapse and eggs fall cleanly through the hollow plinth bore', () => {
    // 1. Remove 3 adjacent Tier-0 load-bearing rods after settling briefly
    const sim3 = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      for (let f = 0; f < 30; f++) sim3.stepFixedFrame();
      sim3.removeRods([0, 1, 2]);

      const crossedRingInsideBore = [false, false, false];
      for (let f = 0; f < 180; f++) {
        sim3.stepFixedFrame();
        for (let e = 0; e < sim3.eggs.length; e++) {
          const p = sim3.eggs[e].body.translation();
          if (p.y < -0.15 && Math.hypot(p.x, p.z) < sharedLayout.plinth.innerRadius) {
            crossedRingInsideBore[e] = true;
          }
        }
      }

      const state3 = sim3.getState();
      const eggDrops3 = sim3.getEggDrops();
      expect(state3.fallenRodCount).toBeGreaterThanOrEqual(8);
      expect(state3.collapsed).toBe(true);
      expect(sim3.getMaxRodDrop()).toBeGreaterThan(1.0);
      for (let e = 0; e < eggDrops3.length; e++) {
        expect(eggDrops3[e]).toBeGreaterThan(1.0);
        // Every ovoid pebble must pass genuinely through the open hollow plinth ring (zero invisible floor)
        expect(crossedRingInsideBore[e]).toBe(true);
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

  it('Interactive Tap-Test Distinction, 3D Grab-Spring Extraction, and Reset restore equilibrium cleanly', () => {
    // 1. Tap-testing a tightly compressed Tier-0 dowel vs lightly loaded Tier-4 dowel gives distinct physical feedback
    const measureTapDisp = (rodId: number) => {
      const s = new ReciprocalNestSim(RAPIER, sharedLayout);
      try {
        for (let f = 0; f < 90; f++) s.stepFixedFrame();
        const before = { ...s.rods[rodId].body.translation() };
        expect(s.pokeRod(rodId, [before.x, before.y, before.z], [0.6, 0.2, 0.8], 0.0028)).toBe(true);
        for (let f = 0; f < 30; f++) s.stepFixedFrame();
        const after = s.rods[rodId].body.translation();
        expect(s.getState().collapsed).toBe(false);
        return Math.hypot(after.x - before.x, after.y - before.y, after.z - before.z);
      } finally {
        s.dispose();
      }
    };
    const dispLoadBearing = measureTapDisp(0);
    const dispUpperLoose = measureTapDisp(50);
    expect(dispUpperLoose).toBeGreaterThan(dispLoadBearing * 2.5);
    expect(dispUpperLoose).toBeLessThan(0.08); // Lively tactile shift, never launched into orbit

    // 2. Grab-spring extraction + Space/double-click reset cleanly restores 56-rod equilibrium
    const sim = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      for (let f = 0; f < 20; f++) sim.stepFixedFrame();

      const rod0 = sim.rods[0];
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
      expect(sim.getMaxRodDrop()).toBeLessThan(0.025);
    } finally {
      sim.dispose();
    }
  });

  it('Choreographed Spiral Re-Weave converges monotonically to exact rest equilibrium with zero stray velocity', () => {
    const sim = new ReciprocalNestSim(RAPIER, sharedLayout);
    const reweave = new NestReweaveController(sharedLayout);
    try {
      // Collapse the dome genuinely first so rods and eggs tumble through the hollow bore
      for (let f = 0; f < 25; f++) sim.stepFixedFrame();
      sim.removeRods([0, 1, 2]);
      for (let f = 0; f < 130; f++) sim.stepFixedFrame();
      expect(sim.getState().collapsed).toBe(true);

      reweave.beginFromSim(sim);
      expect(reweave.isActive()).toBe(true);
      expect(reweave.getMaxErrorToRest()).toBeGreaterThan(0.2);

      // Step to mid-transition (t ≈ 0.35 s): Tier-0 should already be much closer to rest than eggs
      const dt = 1 / 60;
      let completed = false;
      for (let f = 0; f < 21; f++) {
        completed = reweave.step(dt);
      }
      expect(completed).toBe(false);
      expect(reweave.isActive()).toBe(true);
      expect(reweave.getProgress()).toBeGreaterThan(0.35);
      expect(reweave.getProgress()).toBeLessThan(0.5);

      // Verify bottom-to-top stagger: Tier-0 mean error < egg mean error mid-transition
      let tier0Err = 0;
      for (let i = 0; i < 16; i++) {
        const s = sharedLayout.rods[i].center;
        tier0Err += Math.hypot(
          reweave.rodCurrentPos[i * 3] - s[0],
          reweave.rodCurrentPos[i * 3 + 1] - s[1],
          reweave.rodCurrentPos[i * 3 + 2] - s[2]
        );
      }
      tier0Err /= 16;
      let eggErr = 0;
      for (let e = 0; e < 3; e++) {
        const s = sharedLayout.eggs[e].pos;
        eggErr += Math.hypot(
          reweave.eggCurrentPos[e * 3] - s[0],
          reweave.eggCurrentPos[e * 3 + 1] - s[1],
          reweave.eggCurrentPos[e * 3 + 2] - s[2]
        );
      }
      eggErr /= 3;
      expect(tier0Err).toBeLessThan(eggErr);

      // Verify monotonic convergence during the final settling window (t = 0.65 s .. 0.85 s)
      for (let f = 21; f < 39; f++) {
        completed = reweave.step(dt);
      }
      let prevErr = reweave.getMaxErrorToRest();
      let totalFrames = 39;
      while (!completed && totalFrames < 120) {
        completed = reweave.step(dt);
        totalFrames++;
        const curErr = reweave.getMaxErrorToRest();
        expect(curErr).toBeLessThanOrEqual(prevErr + 1e-6);
        prevErr = curErr;
      }

      expect(completed).toBe(true);
      expect(reweave.isActive()).toBe(false);
      expect(totalFrames * dt).toBeGreaterThanOrEqual(REWEAVE_DURATION_SEC - dt);
      expect(reweave.getMaxErrorToRest()).toBeLessThan(1e-6);

      // Commit Rapier handoff and assert exact zero stray velocity across all 56 rods & 3 eggs
      sim.reset();
      for (const rec of sim.rods) {
        const lv = rec.body.linvel();
        const av = rec.body.angvel();
        expect(Math.hypot(lv.x, lv.y, lv.z)).toBe(0);
        expect(Math.hypot(av.x, av.y, av.z)).toBe(0);
      }
      for (const egg of sim.eggs) {
        const lv = egg.body.linvel();
        const av = egg.body.angvel();
        expect(Math.hypot(lv.x, lv.y, lv.z)).toBe(0);
        expect(Math.hypot(av.x, av.y, av.z)).toBe(0);
      }

      // Post-handoff equilibrium holds cleanly for 120 physics frames
      for (let f = 0; f < 120; f++) sim.stepFixedFrame();
      expect(sim.getState().intactRodCount).toBe(56);
      expect(sim.getState().fallenRodCount).toBe(0);
      expect(sim.getState().collapsed).toBe(false);
      expect(sim.getMaxRodDrop()).toBeLessThan(0.025);
    } finally {
      sim.dispose();
    }
  });

  it('Multi-tier contact burnish stations, load-aware pin factor, and zero-allocation dust mote pool', () => {
    // 1. Every tier discovers real physical crossing stations and builds outward-wound geometry
    let cursor = 0;
    for (const tierMeta of sharedLayout.tiers) {
      const sampleRod = sharedLayout.rods[cursor];
      const stations = computeAllRodContactStations(sampleRod, sharedLayout.rods);
      expect(stations.length).toBeGreaterThanOrEqual(2);
      const geom = buildDowelGeometry(
        tierMeta.halfHeight,
        tierMeta.radius,
        sampleRod.notchParams,
        28,
        stations
      );
      expect(computeSignedVolume(geom)).toBeGreaterThan(0);
      geom.dispose();
      cursor += tierMeta.count;
    }

    const dustGeom = buildDustMoteGeometry();
    expect(computeSignedVolume(dustGeom)).toBeGreaterThan(0);
    dustGeom.dispose();

    // 2. Load-bearing Tier-0 rod reports strictly higher pin factor than loose Tier-4 crown rod
    const sim = new ReciprocalNestSim(RAPIER, sharedLayout);
    try {
      for (let f = 0; f < 30; f++) sim.stepFixedFrame();
      expect(sim.estimateRodPinFactor(0)).toBeGreaterThan(0.75);
      expect(sim.estimateRodPinFactor(50)).toBeLessThan(0.25);
    } finally {
      sim.dispose();
    }

    // 3. Dust pool reuses pre-allocated typed arrays (zero reallocation), steps cleanly, and decays to 0
    const pool = new NestDustPool(MAX_DUST_MOTES);
    const posBuf = pool.pos.buffer;
    const velBuf = pool.vel.buffer;
    const opBuf = pool.opacity.buffer;

    expect(pool.getActiveCount()).toBe(0);
    pool.emitBurst(0.4, 0.3, 0.2, 12, 0.22, 0.4);
    expect(pool.getActiveCount()).toBe(12);

    // Step across 150 frames (2.5 s) -> all motes finite, fade cleanly, buffers identical
    let peakOpacity = 0;
    for (let f = 0; f < 150; f++) {
      pool.step(1 / 60);
      for (let i = 0; i < pool.capacity; i++) {
        expect(Number.isFinite(pool.pos[i * 3])).toBe(true);
        expect(Number.isFinite(pool.opacity[i])).toBe(true);
        if (pool.opacity[i] > peakOpacity) peakOpacity = pool.opacity[i];
      }
    }
    expect(peakOpacity).toBeGreaterThan(0.3);
    expect(pool.getActiveCount()).toBe(0);
    expect(pool.pos.buffer).toBe(posBuf);
    expect(pool.vel.buffer).toBe(velBuf);
    expect(pool.opacity.buffer).toBe(opBuf);
  });
});
