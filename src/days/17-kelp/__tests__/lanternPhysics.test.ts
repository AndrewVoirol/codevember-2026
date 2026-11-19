import { describe, expect, it } from 'vitest';
import {
  applyLanternImpulse,
  createLanternPhysicsState,
  cycleFilamentMode,
  pluckParchment,
  setParchmentPoke,
  stepLanternPhysics,
} from '../lanternPhysics';
import {
  buildBrassHardwareGeometry,
  buildBulbGlassGeometry,
  buildFilamentGeometry,
  buildKelpFrillsGeometry,
  buildKelpPanelsGeometry,
  buildLacingGeometry,
  buildWalnutRibsGeometry,
  sampleLanternProfile,
} from '../lanternGeometry';
import { DEFAULT_LANTERN_CONFIG } from '../types';

describe('Day 17 - Kelp Lantern Physics', () => {
  it('swings and settles smoothly under tangential impulse without NaN or blowup', () => {
    const state = createLanternPhysicsState();
    applyLanternImpulse(state, 1.8, -0.4, 2.2);

    let peakAbsZ = 0;
    for (let i = 0; i < 60; i++) {
      stepLanternPhysics(state, 1 / 60);
      peakAbsZ = Math.max(peakAbsZ, Math.abs(state.swingZ));
    }
    expect(peakAbsZ).toBeGreaterThan(0.05);
    expect(peakAbsZ).toBeLessThanOrEqual(0.58);

    // Run 8 seconds to verify damping settles back near equilibrium
    for (let i = 0; i < 480; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    expect(Number.isFinite(state.swingX)).toBe(true);
    expect(Number.isFinite(state.swingZ)).toBe(true);
    expect(Math.abs(state.swingZ)).toBeLessThan(0.05);
    expect(Math.abs(state.yaw)).toBeLessThan(0.08);
  });

  it('depresses parchment skin on poke and decays elastic drumhead ripple after pluck', () => {
    const state = createLanternPhysicsState();
    setParchmentPoke(state, 0.2, 0.1, 0.95, 0.12);

    for (let i = 0; i < 20; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    expect(state.pokeDepth).toBeGreaterThan(0.08);

    // Release and pluck
    state.pokeTargetDepth = 0;
    pluckParchment(state, 0.2, 0.1, 0.95, 1.0);
    expect(state.rippleAmp).toBeGreaterThan(0.5);

    for (let i = 0; i < 180; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    expect(state.rippleAmp).toBeLessThan(0.02);
    expect(Math.abs(state.pokeDepth)).toBeLessThan(0.01);
  });

  it('cycles filament warmth modes cleanly (warm -> bright -> ember -> warm)', () => {
    const state = createLanternPhysicsState();
    expect(state.filamentMode).toBe('warm');

    expect(cycleFilamentMode(state)).toBe('bright');
    for (let i = 0; i < 60; i++) stepLanternPhysics(state, 1 / 60);
    expect(state.filamentGlow).toBeGreaterThan(1.25);

    expect(cycleFilamentMode(state)).toBe('ember');
    for (let i = 0; i < 60; i++) stepLanternPhysics(state, 1 / 60);
    expect(state.filamentGlow).toBeLessThan(0.6);

    expect(cycleFilamentMode(state)).toBe('warm');
  });
});

describe('Day 17 - Kelp Lantern Geometry Integrity', () => {
  it('evaluates monotonic vertical profile with wider equatorial belly than collar and hem', () => {
    const top = sampleLanternProfile(0.0, DEFAULT_LANTERN_CONFIG);
    const belly = sampleLanternProfile(0.48, DEFAULT_LANTERN_CONFIG);
    const bottom = sampleLanternProfile(1.0, DEFAULT_LANTERN_CONFIG);

    expect(top.y).toBeGreaterThan(belly.y);
    expect(belly.y).toBeGreaterThan(bottom.y);
    expect(belly.r).toBeGreaterThan(top.r * 2.0);
    expect(belly.r).toBeGreaterThan(bottom.r * 1.35);
  });

  it('constructs finite, non-empty geometries for all lantern components', () => {
    const geometries = [
      buildWalnutRibsGeometry(),
      buildKelpPanelsGeometry(),
      buildKelpFrillsGeometry(),
      buildLacingGeometry(),
      buildBrassHardwareGeometry(),
      buildBulbGlassGeometry(),
      buildFilamentGeometry(),
    ];

    for (const g of geometries) {
      const pos = g.attributes.position;
      expect(pos.count).toBeGreaterThan(20);
      for (let i = 0; i < pos.array.length; i++) {
        expect(Number.isFinite(pos.array[i])).toBe(true);
      }
      g.dispose();
    }
  });
});
