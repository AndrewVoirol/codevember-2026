import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { KozoCocoonSimulation } from '../src/standby/20-cocoon-blossom/cocoonSimulation';
import {
  buildCalyxBaseCapGeometry,
  buildEmberPearlGeometry,
  buildOrigamiWingGeometry,
  buildPollenBillboardQuadGeometry,
  buildStamenCrownGeometry,
  PETAL_BASE_Y,
} from '../src/standby/20-cocoon-blossom/kozoGeometry';
import { DEFAULT_KOZO_CONFIG } from '../src/standby/20-cocoon-blossom/types';

function measureSignedVolume(geo: THREE.BufferGeometry): {
  signedVol: number;
  triCount: number;
} {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const idx = geo.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const bc = new THREE.Vector3();
  let signedVol = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);
    bc.crossVectors(b, c);
    signedVol += a.dot(bc) / 6;
  }

  return { signedVol, triCount };
}

function measureOutwardRadialDot(geo: THREE.BufferGeometry): {
  outwardCount: number;
  inwardCount: number;
} {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const idx = geo.index!;
  const triCount = idx.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const n = new THREE.Vector3();
  const cen = new THREE.Vector3();

  let outwardCount = 0;
  let inwardCount = 0;
  for (let t = 0; t < triCount; t++) {
    a.fromBufferAttribute(pos, idx.getX(t * 3));
    b.fromBufferAttribute(pos, idx.getX(t * 3 + 1));
    c.fromBufferAttribute(pos, idx.getX(t * 3 + 2));
    e1.subVectors(b, a);
    e2.subVectors(c, a);
    n.crossVectors(e1, e2);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    cen.y = 0;
    if (n.dot(cen) >= 0) outwardCount++;
    else inwardCount++;
  }
  return { outwardCount, inwardCount };
}

describe('Day 20 - Cocoon: Torn Rice-Paper 3-Whorl Blossom, Curl Pollen & Origami Moths', () => {
  it('builds outward-wound petals with varied opacityBias, full-perimeter deckle masks, and finite auxiliary meshes', () => {
    const sim = new KozoCocoonSimulation(DEFAULT_KOZO_CONFIG);
    sim.setTargetBloom(0.0);
    for (let f = 0; f < 140; f++) sim.step(1 / 60);

    expect(sim.petals.length).toBe(18);

    let minOpacityBias = Infinity;
    let maxOpacityBias = -Infinity;

    for (const pet of sim.petals) {
      const { outwardCount, inwardCount } = measureOutwardRadialDot(pet.geometry);
      expect(outwardCount, `petal ${pet.spec.id} outward tris`).toBeGreaterThan(inwardCount * 4);

      if (pet.spec.opacityBias < minOpacityBias) minOpacityBias = pet.spec.opacityBias;
      if (pet.spec.opacityBias > maxOpacityBias) maxOpacityBias = pet.spec.opacityBias;

      const pos = pet.geometry.attributes.position as THREE.BufferAttribute;
      const edgeMask = pet.geometry.attributes.aEdgeMask as THREE.BufferAttribute;
      const petalMeta = pet.geometry.attributes.aPetalMeta as THREE.BufferAttribute;
      const cols = pet.uSegs + 1;

      for (let i = 0; i < pos.count; i++) {
        expect(Number.isFinite(pos.getX(i))).toBe(true);
        expect(Number.isFinite(pos.getY(i))).toBe(true);
        expect(Number.isFinite(pos.getZ(i))).toBe(true);
        const em = edgeMask.getX(i);
        expect(Number.isFinite(em) && em >= 0 && em <= 1).toBe(true);
        expect(Number.isFinite(petalMeta.getZ(i))).toBe(true);
      }

      // Base row stays anchored solid inside calyx (aEdgeMask === 0)
      for (let iu = 0; iu <= pet.uSegs; iu++) {
        const rBase = Math.hypot(pos.getX(iu), pos.getZ(iu));
        expect(rBase).toBeLessThanOrEqual(0.14);
        expect(edgeMask.getX(iu)).toBe(0);
      }

      // Full-perimeter deckle mask covers both mid-lateral edges and top tip across ALL 3 whorls
      const midRow = Math.floor(pet.vSegs * 0.5) * cols;
      expect(edgeMask.getX(midRow)).toBeGreaterThan(0.9);
      expect(edgeMask.getX(midRow + pet.uSegs)).toBeGreaterThan(0.9);
      const tipCenter = pet.vSegs * cols + Math.floor(pet.uSegs * 0.5);
      expect(edgeMask.getX(tipCenter)).toBeGreaterThan(0.9);
    }

    // Confirm genuine per-petal opacity variation across sheer vs denser rice paper
    expect(maxOpacityBias - minOpacityBias).toBeGreaterThanOrEqual(0.40);

    const calyx = buildCalyxBaseCapGeometry();
    const stamen = buildStamenCrownGeometry(18);
    const ember = buildEmberPearlGeometry();
    const pollenQuad = buildPollenBillboardQuadGeometry();
    const lw = buildOrigamiWingGeometry(-1);
    const rw = buildOrigamiWingGeometry(1);

    expect(measureSignedVolume(calyx).signedVol).toBeGreaterThan(0);
    expect(measureSignedVolume(ember).signedVol).toBeGreaterThan(0);
    expect(stamen.attributes.position.count).toBeGreaterThan(500);
    expect(pollenQuad.attributes.position.count).toBe(4);
    expect(lw.attributes.position.count).toBe(15);
    expect(rw.attributes.position.count).toBe(15);

    calyx.dispose();
    stamen.dispose();
    ember.dispose();
    pollenQuad.dispose();
    lw.dispose();
    rw.dispose();
    sim.dispose();
  });

  it('maintains a poised upward-cupping initial bloom without downward droop and cascades smoothly across 0..1', () => {
    const simInitial = new KozoCocoonSimulation(DEFAULT_KOZO_CONFIG);
    for (let f = 0; f < 60; f++) simInitial.step(1 / 60);

    // Zero floppy droop below calyx floor at initialBloom (0.74)
    let minInitialY = Infinity;
    for (const pet of simInitial.petals) {
      const pos = pet.geometry.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i++) {
        const y = pos.getY(i);
        if (y < minInitialY) minInitialY = y;
      }
    }
    expect(minInitialY).toBeGreaterThan(PETAL_BASE_Y - 0.15);
    simInitial.dispose();

    const sim = new KozoCocoonSimulation({ ...DEFAULT_KOZO_CONFIG, initialBloom: 0.0 });
    for (let f = 0; f < 40; f++) sim.step(1 / 60);

    let maxClosedRad = 0;
    for (const pet of sim.petals) {
      const pos = pet.geometry.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i++) {
        const r = Math.hypot(pos.getX(i), pos.getZ(i));
        if (r > maxClosedRad) maxClosedRad = r;
      }
    }
    expect(maxClosedRad).toBeLessThan(1.08);

    sim.setTargetBloom(1.0);
    for (let f = 0; f < 30; f++) sim.step(1 / 60);
    expect(sim.petalLocalBloom[0]).toBeGreaterThan(sim.petalLocalBloom[12] + 0.08);

    for (let f = 0; f < 160; f++) sim.step(1 / 60);
    expect(sim.bloom).toBeGreaterThan(0.98);

    let maxOpenRad = 0;
    for (const pet of sim.petals) {
      const pos = pet.geometry.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i++) {
        const r = Math.hypot(pos.getX(i), pos.getZ(i));
        if (r > maxOpenRad) maxOpenRad = r;
      }
    }
    expect(maxOpenRad).toBeGreaterThan(maxClosedRad * 1.45);

    sim.pullPetal(0, -0.4, 0.2);
    sim.step(1 / 60);
    expect(sim.targetBloom).toBeLessThan(0.8);
    expect(Math.abs(sim.petalPitchVel[0])).toBeGreaterThan(0.1);

    sim.dispose();
  });

  it('animates 12 curl-gliding origami moths and 280 luminous pollen motes with air-wake coupling under CPU budget', () => {
    const sim = new KozoCocoonSimulation(DEFAULT_KOZO_CONFIG);
    for (let f = 0; f < 60; f++) sim.step(1 / 60);

    const tOpen = sim.getTelemetry();
    expect(tOpen.activeMothCount).toBe(12);
    expect(tOpen.activePollenCount).toBeGreaterThan(120);

    sim.setAirWake(0.25, 0.35, 0.25, 3.5, 1.2, -1.5, true);
    sim.pulseBloomBurst();

    const t0 = performance.now();
    for (let f = 0; f < 120; f++) {
      sim.step(1 / 60);
    }
    const avgMs = (performance.now() - t0) / 120;
    expect(avgMs).toBeLessThan(4.5);

    for (let i = 0; i < sim.config.pollenCount * 3; i++) {
      expect(Number.isFinite(sim.pollenPos[i])).toBe(true);
    }

    sim.setTargetBloom(0.0);
    for (let f = 0; f < 220; f++) sim.step(1 / 60);
    expect(sim.getTelemetry().activeMothCount).toBe(0);
    expect(sim.getTelemetry().activePollenCount).toBe(0);

    sim.dispose();
  });
});
