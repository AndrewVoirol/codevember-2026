import * as THREE from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import {
  applyLanternImpulse,
  applyWorldLanternImpulse,
  createLanternPhysicsState,
  cycleFilamentMode,
  pluckParchment,
  setLanternGrabTorque,
  setParchmentPoke,
  stepLanternPhysics,
} from '../lanternPhysics';
import {
  buildBrassHardwareGeometry,
  buildBulbGlassGeometry,
  buildBulbHardwareGeometry,
  buildCeramicSocketGeometry,
  buildFilamentAuraGeometry,
  buildFilamentGeometry,
  buildKelpFrillsGeometry,
  buildKelpPanelsGeometry,
  buildLacingGeometry,
  buildWalnutRibsGeometry,
  createDynamicHempCord,
  sampleLanternProfile,
} from '../lanternGeometry';
import { DEFAULT_LANTERN_CONFIG } from '../types';

function computeSignedVolume(geom: THREE.BufferGeometry): number {
  const pos = geom.attributes.position;
  const idx = geom.index;
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const cross = new THREE.Vector3();
  let volume = 0;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  for (let t = 0; t < triCount; t++) {
    const a = idx ? idx.getX(t * 3) : t * 3;
    const b = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const c = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    v0.fromBufferAttribute(pos, a);
    v1.fromBufferAttribute(pos, b);
    v2.fromBufferAttribute(pos, c);
    cross.crossVectors(v1, v2);
    volume += v0.dot(cross) / 6;
  }
  return volume;
}

function countRadialWinding(geom: THREE.BufferGeometry): { outward: number; inward: number } {
  const pos = geom.attributes.position;
  const idx = geom.index;
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const tn = new THREE.Vector3();
  const centroid = new THREE.Vector3();
  let outward = 0;
  let inward = 0;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  for (let t = 0; t < triCount; t++) {
    const a = idx ? idx.getX(t * 3) : t * 3;
    const b = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const c = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    v0.fromBufferAttribute(pos, a);
    v1.fromBufferAttribute(pos, b);
    v2.fromBufferAttribute(pos, c);
    e1.subVectors(v1, v0);
    e2.subVectors(v2, v0);
    tn.crossVectors(e1, e2);
    centroid.addVectors(v0, v1).add(v2).multiplyScalar(1 / 3);
    const d = centroid.x * tn.x + centroid.z * tn.z;
    if (d > 1e-8) outward++;
    else if (d < -1e-8) inward++;
  }
  return { outward, inward };
}

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
    expect(peakAbsZ).toBeLessThanOrEqual(0.54);

    // Run 8 seconds to verify damping settles back near equilibrium
    for (let i = 0; i < 480; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    expect(Number.isFinite(state.swingX)).toBe(true);
    expect(Number.isFinite(state.swingZ)).toBe(true);
    expect(Math.abs(state.swingZ)).toBeLessThan(0.05);
    expect(Math.abs(state.yaw)).toBeLessThan(0.08);
  });

  it('holds pulled equilibrium under continuous spring-tether grab torque and releases cleanly', () => {
    const state = createLanternPhysicsState();
    setLanternGrabTorque(state, 0, 1.4, 0.8, true);

    for (let i = 0; i < 90; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    // Lantern stays visibly displaced toward +X (+swingZ) while held
    expect(state.swingZ).toBeGreaterThan(0.25);
    expect(state.yaw).toBeGreaterThan(0.15);

    // Release -> smoothly swings back across center
    setLanternGrabTorque(state, 0, 0, 0, false);
    let crossedNegative = false;
    for (let i = 0; i < 180; i++) {
      stepLanternPhysics(state, 1 / 60);
      if (state.swingZ < -0.02) crossedNegative = true;
    }
    expect(crossedNegative).toBe(true);
  });

  it('rotates world acceleration into the yaw-spun lantern frame for internal bulb lag', () => {
    const state = createLanternPhysicsState();
    // Spin lantern 180 degrees around vertical cord axis, then kick +X world force
    state.yaw = Math.PI;
    applyWorldLanternImpulse(state, 1.5, 0, 0);
    for (let i = 0; i < 10; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    expect(Number.isFinite(state.bulbSwingX)).toBe(true);
    expect(Number.isFinite(state.bulbSwingZ)).toBe(true);
  });

  it('depresses parchment skin on poke (tracking bay index) and decays elastic drumhead ripple after pluck', () => {
    const state = createLanternPhysicsState();
    setParchmentPoke(state, 0.2, 0.1, 0.95, 0.12, 0);
    expect(state.pokeBay).toBe(0);

    for (let i = 0; i < 20; i++) {
      stepLanternPhysics(state, 1 / 60);
    }
    expect(state.pokeDepth).toBeGreaterThan(0.08);

    // Release and pluck
    state.pokeTargetDepth = 0;
    pluckParchment(state, 0.2, 0.1, 0.95, 1.0, 0);
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

describe('Day 17 - Kelp Lantern Geometry & Winding Integrity', () => {
  it('evaluates monotonic vertical profile with wider equatorial belly than collar and hem', () => {
    const top = sampleLanternProfile(0.0, DEFAULT_LANTERN_CONFIG);
    const belly = sampleLanternProfile(0.48, DEFAULT_LANTERN_CONFIG);
    const bottom = sampleLanternProfile(1.0, DEFAULT_LANTERN_CONFIG);

    expect(top.y).toBeGreaterThan(belly.y);
    expect(belly.y).toBeGreaterThan(bottom.y);
    expect(belly.r).toBeGreaterThan(top.r * 2.0);
    expect(belly.r).toBeGreaterThan(bottom.r * 1.35);
  });

  it('constructs strictly outward-wound, non-inverted solids and shells for all components', () => {
    const walnut = buildWalnutRibsGeometry();
    const panels = buildKelpPanelsGeometry();
    const frills = buildKelpFrillsGeometry();
    const lacing = buildLacingGeometry();
    const brass = buildBrassHardwareGeometry();
    const ceramic = buildCeramicSocketGeometry();
    const bulbGlass = buildBulbGlassGeometry();
    const bulbHw = buildBulbHardwareGeometry();
    const filament = buildFilamentGeometry();
    const aura = buildFilamentAuraGeometry();
    const cord = createDynamicHempCord();
    cord.update(DEFAULT_LANTERN_CONFIG.pivotY, new THREE.Vector3(0, 1.5, 0), 0.35);

    // Closed / solid structural meshes MUST have strictly positive signed volume (zero inside-out shells!)
    expect(computeSignedVolume(walnut)).toBeGreaterThan(0.01);
    expect(computeSignedVolume(brass)).toBeGreaterThan(0.01);
    expect(computeSignedVolume(ceramic)).toBeGreaterThan(0.0005);
    expect(computeSignedVolume(bulbHw)).toBeGreaterThan(0.001);
    expect(computeSignedVolume(cord.geometry)).toBeGreaterThan(0.0005);

    // Open radial shells MUST face 100% outward
    const panelWind = countRadialWinding(panels);
    expect(panelWind.outward).toBeGreaterThan(10000);
    expect(panelWind.inward).toBe(0);

    const cordWind = countRadialWinding(cord.geometry);
    expect(cordWind.outward).toBeGreaterThan(200);
    expect(cordWind.inward).toBe(0);

    const glassWind = countRadialWinding(bulbGlass);
    expect(glassWind.outward).toBeGreaterThan(200);
    expect(glassWind.inward).toBe(0);

    // Lacing carries aBayMeta so GPU membrane flex stays locked to the kelp panels
    expect(lacing.attributes.aBayMeta).toBeDefined();
    expect(lacing.attributes.aBayMeta.count).toBe(lacing.attributes.position.count);

    // Panel UVs stay strictly inside their assigned seamless atlas strip bounds (zero wrap-seam bleed)
    const uvs = panels.attributes.uv;
    const bounds = panels.attributes.aStripBounds;
    expect(bounds).toBeDefined();
    for (let i = 0; i < uvs.count; i++) {
      const u = uvs.getX(i);
      const uMin = bounds.getX(i);
      const uMax = bounds.getY(i);
      expect(u).toBeGreaterThanOrEqual(uMin - 1e-5);
      expect(u).toBeLessThanOrEqual(uMax + 1e-5);
    }

    const allGeoms = [
      walnut,
      panels,
      frills,
      lacing,
      brass,
      ceramic,
      bulbGlass,
      bulbHw,
      filament,
      aura,
      cord.geometry,
    ];
    for (const g of allGeoms) {
      const pos = g.attributes.position;
      expect(pos.count).toBeGreaterThan(20);
      for (let i = 0; i < pos.array.length; i++) {
        expect(Number.isFinite(pos.array[i])).toBe(true);
      }
      g.dispose();
    }
  });
});
