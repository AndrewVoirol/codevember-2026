import { describe, expect, it } from 'vitest';
import {
  EarthstarBellowsPhysics,
  MAX_VORTEX_RINGS,
  PERISTOME_NOZZLE_Y,
} from '../bellowsPhysics';
import {
  buildExoperidiumPetalsGeometry,
  buildMycelialRootletsGeometry,
  buildSporeSacGeometry,
} from '../earthstarGeometry';

describe('Day 16 — Spore: Earthstar Bellows & Toroidal Vortex Physics', () => {
  it('seeds a mature hero toroidal vortex ring hovering above the peristome on boot', () => {
    const physics = new EarthstarBellowsPhysics();
    const ring = physics.seedInitialHeroRing();

    expect(ring.active).toBe(true);
    expect(ring.center[1]).toBeGreaterThanOrEqual(PERISTOME_NOZZLE_Y + 0.38);
    expect(ring.majorRadius).toBeGreaterThan(0.32);
    expect(ring.circulation).toBeGreaterThan(1.5);
    expect(physics.getActiveRingCount()).toBe(1);
  });

  it('indents elastically on sac squeeze, flares the peristome, and tilts the ring away from off-center pokes', () => {
    const physics = new EarthstarBellowsPhysics();
    // Poke strongly on the +X flank
    physics.beginSqueeze(0.45, -0.32, 0.20);
    const res = physics.step(0.016, true);

    expect(res.puffEvents.length).toBe(1);
    const ev = res.puffEvents[0];
    expect(ev.particleCount).toBeGreaterThan(4000);
    // Squeezing the +X flank should tilt the vented ring slightly toward -X
    expect(ev.axis[0]).toBeLessThan(-0.08);
    expect(ev.axis[1]).toBeGreaterThan(0.92);

    // Advance a few frames while holding squeeze
    for (let i = 0; i < 20; i++) {
      physics.step(0.016, true);
    }
    expect(physics.compression).toBeGreaterThan(0.35);
    expect(physics.indentStrength).toBeGreaterThan(0.35);
    expect(physics.peristomeFlare).toBeGreaterThan(0.2);
    expect(physics.petalRecoil).toBeGreaterThan(0.05);

    // Releasing after >0.24s launches a fuller rebound ring
    physics.endSqueeze();
    const releaseRes = physics.step(0.016, false);
    expect(releaseRes.puffEvents.length).toBe(1);
    expect(physics.getActiveRingCount()).toBe(2);

    // Spring-mass bellows recovers cleanly back near 0 at rest
    for (let i = 0; i < 120; i++) {
      physics.step(0.016, false);
    }
    expect(Math.abs(physics.compression)).toBeLessThan(0.03);
    expect(Math.abs(physics.indentStrength)).toBeLessThan(0.03);
  });

  it('expands toroidal ring major radius via entrainment and decays circulation smoothly without NaN', () => {
    const physics = new EarthstarBellowsPhysics();
    const ring = physics.triggerPuff(0.95, 0, 0);
    const startR = ring.majorRadius;
    const startY = ring.center[1];
    const startCirc = ring.circulation;

    for (let i = 0; i < 180; i++) {
      physics.step(1 / 60, false);
    }

    expect(ring.active).toBe(true);
    expect(ring.center[1]).toBeGreaterThan(startY + 0.8);
    expect(ring.majorRadius).toBeGreaterThan(startR + 0.2);
    expect(ring.circulation).toBeLessThan(startCirc);
    expect(Number.isFinite(ring.center[1])).toBe(true);
  });

  it('caps simultaneous rings cleanly in circular slots across rapid taps', () => {
    const physics = new EarthstarBellowsPhysics();
    for (let k = 0; k < 12; k++) {
      physics.triggerPuff(0.8, (k % 3 - 1) * 0.25, 0.1);
      physics.step(0.05, false);
    }
    expect(physics.getActiveRingCount()).toBeLessThanOrEqual(MAX_VORTEX_RINGS);
    expect(physics.totalPuffs).toBe(12);
  });
});

describe('Day 16 — Spore: Procedural Geastrum Triplex Geometry Integrity', () => {
  const assertCleanGeometry = (geom: ReturnType<typeof buildSporeSacGeometry>, requiredAttr: string) => {
    const pos = geom.getAttribute('position');
    const norm = geom.getAttribute('normal');
    const meta = geom.getAttribute(requiredAttr);
    expect(pos.count).toBeGreaterThan(500);
    expect(norm.count).toBe(pos.count);
    expect(meta.count).toBe(pos.count);

    for (let i = 0; i < pos.count; i++) {
      expect(Number.isFinite(pos.getX(i))).toBe(true);
      expect(Number.isFinite(pos.getY(i))).toBe(true);
      expect(Number.isFinite(pos.getZ(i))).toBe(true);
      const nLen = Math.hypot(norm.getX(i), norm.getY(i), norm.getZ(i));
      expect(nLen).toBeGreaterThan(0.9);
    }
    geom.dispose();
  };

  it('builds finite outward-wound spore sac, recurved cracked petals, and hanging rootlets', () => {
    assertCleanGeometry(buildSporeSacGeometry(), 'aSacMeta');
    assertCleanGeometry(buildExoperidiumPetalsGeometry(), 'aPetalMeta');
    assertCleanGeometry(buildMycelialRootletsGeometry(), 'aRootMeta');
  });
});
