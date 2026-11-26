import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  buildEmberCoreGeometry,
  buildEmberQuadGeometry,
  buildOrigamiButterflyBodyGeometry,
  buildOrigamiButterflyWingGeometry,
  buildOvoidBambooFrameGeometry,
  buildOvoidPaperDomeGeometry,
  buildSilkGossamerVeilGeometry,
  buildSilkWrapBandsGeometry,
  buildSuspensionCordGeometry,
  buildUnspoolingSilkLoopsGeometry,
  LANTERN_TOP_Y,
} from '../src/days/20-cocoon/lanternGeometry';
import { SilkLanternSimulation } from '../src/days/20-cocoon/lanternSimulation';

function assertAllFinite(geo: THREE.BufferGeometry): void {
  const pos = geo.getAttribute('position');
  expect(pos).toBeDefined();
  expect(pos.count).toBeGreaterThan(0);
  for (let i = 0; i < pos.array.length; i++) {
    expect(Number.isFinite(pos.array[i])).toBe(true);
  }
  const nrm = geo.getAttribute('normal');
  if (nrm) {
    for (let i = 0; i < nrm.array.length; i++) {
      expect(Number.isFinite(nrm.array[i])).toBe(true);
    }
  }
  const uv = geo.getAttribute('uv');
  if (uv) {
    for (let i = 0; i < uv.array.length; i++) {
      expect(Number.isFinite(uv.array[i])).toBe(true);
    }
  }
}

function computeSignedVolume(geo: THREE.BufferGeometry): number {
  const pos = geo.getAttribute('position');
  const idx = geo.getIndex();
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const cross = new THREE.Vector3();
  let vol6 = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);
    cross.crossVectors(b, c);
    vol6 += a.dot(cross);
  }
  return vol6 / 6;
}

function countInwardRadialTriangles(geo: THREE.BufferGeometry): number {
  const pos = geo.getAttribute('position');
  const idx = geo.getIndex();
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceN = new THREE.Vector3();
  const centroid = new THREE.Vector3();
  const radial = new THREE.Vector3();
  let inward = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);
    ab.subVectors(b, a);
    ac.subVectors(c, a);
    faceN.crossVectors(ab, ac);
    if (faceN.lengthSq() < 1e-12) continue;

    centroid.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    radial.set(centroid.x, centroid.y >= LANTERN_TOP_Y - 0.02 ? 0.5 : 0, centroid.z);
    if (radial.lengthSq() < 1e-8) continue;
    if (faceN.dot(radial) < 0) inward++;
  }
  return inward;
}

describe('Day 20 Alt 1 (Concept A) — Silk-Spun Lantern Geometry Invariants', () => {
  it('builds strictly finite, outward-wound closed solids and ovoid paper dome', () => {
    const closedSolids: Array<[string, THREE.BufferGeometry]> = [
      ['ovoidBambooFrame', buildOvoidBambooFrameGeometry(12)],
      ['suspensionCord', buildSuspensionCordGeometry()],
      ['silkWrapBands', buildSilkWrapBandsGeometry(340)],
      ['unspoolingSilkLoops', buildUnspoolingSilkLoopsGeometry(38)],
      ['origamiButterflyBody', buildOrigamiButterflyBodyGeometry()],
      ['emberCore', buildEmberCoreGeometry()],
    ];

    for (const [name, geo] of closedSolids) {
      assertAllFinite(geo);
      expect(computeSignedVolume(geo), `${name} must have positive outward signed volume`).toBeGreaterThan(0);
      geo.dispose();
    }

    const bamboo = buildOvoidBambooFrameGeometry(12);
    expect(bamboo.getAttribute('aWoodFace')).toBeDefined();
    expect(bamboo.getAttribute('aNodeMask')).toBeDefined();
    bamboo.dispose();

    const dome = buildOvoidPaperDomeGeometry(12, 56);
    assertAllFinite(dome);
    expect(dome.getAttribute('aRibProximity')).toBeDefined();
    expect(dome.getAttribute('aWaistMask')).toBeDefined();
    expect(dome.getAttribute('aPanelId')).toBeDefined();
    expect(dome.getAttribute('aCrinkleRidge')).toBeDefined();
    expect(countInwardRadialTriangles(dome)).toBe(0);
    dome.dispose();

    const veil = buildSilkGossamerVeilGeometry(12, 28);
    assertAllFinite(veil);
    expect(countInwardRadialTriangles(veil)).toBe(0);
    veil.dispose();

    const leftWing = buildOrigamiButterflyWingGeometry(-1);
    const rightWing = buildOrigamiButterflyWingGeometry(1);
    const quad = buildEmberQuadGeometry();
    assertAllFinite(leftWing);
    assertAllFinite(rightWing);
    assertAllFinite(quad);
    expect(leftWing.getAttribute('aPaperPlies')).toBeDefined();
    expect(leftWing.getAttribute('aCreaseProx')).toBeDefined();
    expect(rightWing.getAttribute('aPaperPlies')).toBeDefined();
    expect(rightWing.getAttribute('aCreaseProx')).toBeDefined();
    leftWing.dispose();
    rightWing.dispose();
    quad.dispose();
  });
});

describe('Day 20 Alt 1 (Concept A) — SilkLanternSimulation Lifecycle & Physics', () => {
  it('progresses cleanly through bound -> unwinding -> emerged -> buoyant with monotonic emergence', () => {
    const sim = new SilkLanternSimulation();
    sim.resetToBound();

    expect(sim.state.stage).toBe('bound');
    expect(sim.state.unravel).toBeLessThan(0.1);
    expect(sim.state.activeButterflies).toBe(0);
    expect(sim.state.butterfliesEmerged).toBe(0);

    const seenStages = new Set<string>([sim.state.stage]);
    let prevEmerged = sim.state.butterfliesEmerged;

    for (let step = 0; step < 720; step++) {
      sim.step(1 / 60);
      seenStages.add(sim.state.stage);
      expect(sim.state.butterfliesEmerged).toBeGreaterThanOrEqual(prevEmerged);
      prevEmerged = sim.state.butterfliesEmerged;
    }

    expect(seenStages.has('bound')).toBe(true);
    expect(seenStages.has('unwinding')).toBe(true);
    expect(seenStages.has('emerged')).toBe(true);
    expect(seenStages.has('buoyant')).toBe(true);
    expect(sim.state.stage).toBe('buoyant');
    expect(sim.state.unravel).toBeGreaterThan(0.8);
    expect(sim.state.ignition).toBeGreaterThan(0.8);
    expect(sim.state.butterfliesEmerged).toBeGreaterThan(0);
    expect(sim.state.activeButterflies).toBeGreaterThan(0);

    sim.resetToBound();
    expect(sim.state.stage).toBe('bound');
    expect(sim.state.activeButterflies).toBe(0);
    expect(sim.state.butterfliesEmerged).toBe(0);
    sim.dispose();
  });

  it('responds to pointer swirl and stokeBurst with zero NaN across 600 steps', () => {
    const sim = new SilkLanternSimulation();
    const pulseBefore = sim.state.emberPulse;
    const released = sim.stokeBurst(0.12, -0.35, 0.08);
    expect(released).toBeGreaterThan(0);
    expect(sim.state.emberPulse).toBeGreaterThan(pulseBefore);

    for (let i = 0; i < 600; i++) {
      const t = i / 60;
      sim.applyPointer(
        Math.sin(t * 2.0) * 0.65,
        Math.cos(t * 1.3) * 0.5,
        0.18,
        Math.cos(t * 2.0) * 1.3,
        -Math.sin(t * 1.3) * 0.85,
        0.05,
        i % 90 < 15,
        1 / 60,
      );
      sim.step(1 / 60);
    }

    const s = sim.state;
    const scalars = [
      s.unravel,
      s.breathExpansion,
      s.ignition,
      s.emberPulse,
      s.lanternSwingX,
      s.lanternSwingZ,
      s.lanternYaw,
      s.lanternLiftY,
      s.time,
    ];
    for (const v of scalars) {
      expect(Number.isFinite(v)).toBe(true);
    }

    for (const bf of sim.butterflies) {
      expect(Number.isFinite(bf.pos[0])).toBe(true);
      expect(Number.isFinite(bf.pos[1])).toBe(true);
      expect(Number.isFinite(bf.pos[2])).toBe(true);
      expect(Number.isFinite(bf.yaw)).toBe(true);
      expect(Number.isFinite(bf.pitch)).toBe(true);
      expect(Number.isFinite(bf.roll)).toBe(true);
    }

    for (const loop of sim.silkLoops) {
      expect(Number.isFinite(loop.peelAmount)).toBe(true);
      expect(Number.isFinite(loop.swayX)).toBe(true);
      expect(Number.isFinite(loop.swayY)).toBe(true);
      expect(Number.isFinite(loop.swayZ)).toBe(true);
    }

    for (const mote of sim.embers) {
      expect(Number.isFinite(mote.pos[0])).toBe(true);
      expect(Number.isFinite(mote.pos[1])).toBe(true);
      expect(Number.isFinite(mote.pos[2])).toBe(true);
    }

    sim.dispose();
  });
});
