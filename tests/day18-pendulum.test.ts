import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  BOB_TOP_COLLET_OFFSET,
  COUNTERWEIGHT_LENGTH,
  DOME_CENTER,
  DOME_RADIUS,
  MAX_STONES,
  PULLEY_INNER_RADIUS,
  PULLEY_OUTER_RADIUS,
  STONE_RADIUS,
  STONE_SEAT_OFFSET,
  DomePendulumSimulation,
  type PulleyKinematicsFrame,
} from '../src/days/18-pendulum/physics';
import { createStoneGeometryVariants } from '../src/days/18-pendulum/geometry';

describe('Day 18 - Hammered-Iron Dome, Differential Pulley & Grazing Orbital Pendulum physics', () => {
  it('tossed stones fly, bounce, emit surface impact/scuff events, cling cleanly, and have zero downhill creep', () => {
    const sim = new DomePendulumSimulation();
    const initialCount = sim.stones.length;
    expect(initialCount).toBe(9);

    const target = new THREE.Vector3(0.35, DOME_CENTER.y + 0.86, 0.35);
    const stone = sim.tossStoneToward(target);
    expect(stone.phase).toBe('flying');
    expect(sim.stones.length).toBe(initialCount + 1);

    let totalImpacts = 0;
    let totalScuffs = 0;

    for (let i = 0; i < 192; i++) {
      sim.step(1 / 60);
      totalImpacts += sim.impactEvents.length;
      totalScuffs += sim.scuffEvents.length;
      sim.clearSurfaceEvents();
    }

    expect(stone.phase).toBe('clung');
    expect(totalImpacts).toBeGreaterThan(0);
    expect(totalScuffs).toBeGreaterThan(0);

    const distFromCenter = stone.pos.distanceTo(DOME_CENTER);
    expect(distFromCenter).toBeCloseTo(DOME_RADIUS + STONE_SEAT_OFFSET, 2);

    // Park pendulum far from stone and verify Coulomb static friction holds stone stationary
    sim.pendulum.isGrabbed = true;
    sim.pendulum.grabTarget.set(-0.9, DOME_CENTER.y + 0.9, -0.9);
    const settledPos = stone.pos.clone();
    for (let i = 0; i < 120; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
    }
    expect(stone.phase).toBe('clung');
    expect(stone.pos.distanceTo(settledPos)).toBeLessThan(0.001);
  });

  it('enforces same-polarity repulsion and opposite-polarity non-overlapping attraction with rigid separation', () => {
    const sim = new DomePendulumSimulation();
    sim.stones.length = 0;

    sim.pendulum.isGrabbed = true;
    sim.pendulum.grabTarget.set(0, DOME_CENTER.y + 0.6, -1.2);

    const s1 = sim.tossStoneToward(
      new THREE.Vector3(-0.08, DOME_CENTER.y + DOME_RADIUS, 0),
      new THREE.Vector3(-0.08, 0.45, 0),
      1
    );
    const s2 = sim.tossStoneToward(
      new THREE.Vector3(0.08, DOME_CENTER.y + DOME_RADIUS, 0),
      new THREE.Vector3(0.08, 0.45, 0),
      1
    );

    for (let i = 0; i < 200; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
    }
    const repelledDist = s1.pos.distanceTo(s2.pos);
    expect(repelledDist).toBeGreaterThan(0.2);

    sim.stones.length = 0;
    const a1 = sim.tossStoneToward(
      new THREE.Vector3(-0.14, DOME_CENTER.y + DOME_RADIUS, 0),
      new THREE.Vector3(-0.14, 0.45, 0),
      1
    );
    const a2 = sim.tossStoneToward(
      new THREE.Vector3(0.14, DOME_CENTER.y + DOME_RADIUS, 0),
      new THREE.Vector3(0.14, 0.45, 0),
      -1
    );

    for (let i = 0; i < 200; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
    }
    const attractedDist = a1.pos.distanceTo(a2.pos);
    expect(attractedDist).toBeLessThan(repelledDist);
    expect(attractedDist).toBeGreaterThan(STONE_RADIUS * (a1.scale + a2.scale) * 0.94);
  });

  it('keeps bob & tip outside dome, couples pulley wheel & counterweight at exact differential ratio, and aligns wire coaxially with bob collet', () => {
    const simA = new DomePendulumSimulation();
    const simB = new DomePendulumSimulation();

    const firstId = simB.stones[0].id;
    expect(simB.flipStone(firstId)).toBe(true);

    const frame: PulleyKinematicsFrame = {
      azimuth: 0,
      wheelAngle: 0,
      bobTangentWorld: new THREE.Vector3(),
      bobColletWorld: new THREE.Vector3(),
      cwTangentWorld: new THREE.Vector3(),
      cwEyeletWorld: new THREE.Vector3(),
      departAngleLocal: 0,
      wireLength: 1.5,
      pluckPhase: 0,
      pluckAmp: 0,
    };
    const wireDir = new THREE.Vector3();
    const bobUp = new THREE.Vector3();

    let minBobClearance = Infinity;
    let minTipClearance = Infinity;
    let minCwBobGap = Infinity;
    let minWireLen = Infinity;
    let maxWireLen = -Infinity;
    let minWheelAng = Infinity;
    let maxWheelAng = -Infinity;
    let minCwY = Infinity;
    let maxCwY = -Infinity;

    for (let i = 0; i < 600; i++) {
      simA.step(1 / 60);
      simB.step(1 / 60);
      simA.clearSurfaceEvents();
      simB.clearSurfaceEvents();

      const bobClearance = simA.pendulum.pos.distanceTo(DOME_CENTER) - DOME_RADIUS;
      const tipClearance = simA.pendulum.tipPos.distanceTo(DOME_CENTER) - DOME_RADIUS;
      if (bobClearance < minBobClearance) minBobClearance = bobClearance;
      if (tipClearance < minTipClearance) minTipClearance = tipClearance;

      simA.computePulleyKinematics(frame);

      if (simA.pendulum.wireLength < minWireLen) minWireLen = simA.pendulum.wireLength;
      if (simA.pendulum.wireLength > maxWireLen) maxWireLen = simA.pendulum.wireLength;
      if (simA.pendulum.pulleyWheelAngle < minWheelAng) minWheelAng = simA.pendulum.pulleyWheelAngle;
      if (simA.pendulum.pulleyWheelAngle > maxWheelAng) maxWheelAng = simA.pendulum.pulleyWheelAngle;
      if (simA.pendulum.counterweightY < minCwY) minCwY = simA.pendulum.counterweightY;
      if (simA.pendulum.counterweightY > maxCwY) maxCwY = simA.pendulum.counterweightY;

      // Counterweight bottom must always clear the top of the bob collet safely
      const cwBottomY = simA.pendulum.counterweightY - COUNTERWEIGHT_LENGTH;
      const bobTopY = simA.pendulum.pos.y + BOB_TOP_COLLET_OFFSET;
      const gap = cwBottomY - bobTopY;
      if (gap < minCwBobGap) minCwBobGap = gap;

      // Taut straight plumb wire must enter the bob's upper brass collet coaxially (zero kink!)
      wireDir.subVectors(frame.bobTangentWorld, frame.bobColletWorld).normalize();
      bobUp.set(0, 1, 0).applyQuaternion(simA.pendulum.quat).normalize();
      expect(wireDir.dot(bobUp)).toBeGreaterThan(0.985);

      expect(Number.isFinite(simA.pendulum.pos.x)).toBe(true);
      expect(Number.isFinite(simA.pendulum.pos.y)).toBe(true);
      expect(Number.isFinite(simA.pendulum.pos.z)).toBe(true);
    }

    expect(minBobClearance).toBeGreaterThan(0.2);
    expect(minTipClearance).toBeGreaterThan(0.03);
    expect(minCwBobGap).toBeGreaterThan(0.18);

    // Verify exact differential pulley transmission ratio:
    // delta(wheelAngle) == delta(wireLength) / R_outer, delta(cwY) == delta(wireLength) * (R_inner / R_outer)
    const dWire = maxWireLen - minWireLen;
    expect(dWire).toBeGreaterThan(0.25);
    expect(maxWheelAng - minWheelAng).toBeCloseTo(dWire / PULLEY_OUTER_RADIUS, 1);
    expect(maxCwY - minCwY).toBeCloseTo(dWire * (PULLEY_INNER_RADIUS / PULLEY_OUTER_RADIUS), 2);

    // Flipping one stone must measurably diverge the chaotic pendulum trajectory
    expect(simA.pendulum.pos.distanceTo(simB.pendulum.pos)).toBeGreaterThan(0.08);
  });

  it('produces smooth non-oscillating magnetic stone tilt (zero 240Hz sign flip) and supports 180-deg flip tumble', () => {
    const sim = new DomePendulumSimulation();
    const stone = sim.stones.find((s) => s.polarity === 1)!;
    sim.pendulum.isGrabbed = true;
    sim.pendulum.grabTarget.copy(stone.pos).add(new THREE.Vector3(0.22, 0.25, 0.1));

    let prevDot = 0;
    let signFlips = 0;
    const prevTilt = new THREE.Vector3();

    for (let i = 0; i < 60; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
      if (i > 10 && prevTilt.lengthSq() > 1e-6 && stone.tugTilt.lengthSq() > 1e-6) {
        const d = prevTilt.dot(stone.tugTilt);
        if (prevDot > 0 && d < 0) signFlips++;
        prevDot = d;
      }
      prevTilt.copy(stone.tugTilt);
    }

    expect(stone.tugTilt.length()).toBeGreaterThan(0.05);
    expect(signFlips).toBe(0);

    const startQuat = stone.quat.clone();
    expect(sim.flipStone(stone.id)).toBe(true);
    let maxQuatAngle = 0;
    let reClampImpacts = 0;

    for (let i = 0; i < 45; i++) {
      sim.step(1 / 60);
      const angle = startQuat.angleTo(stone.quat);
      if (angle > maxQuatAngle) maxQuatAngle = angle;
      reClampImpacts += sim.impactEvents.length;
      sim.clearSurfaceEvents();
    }

    expect(maxQuatAngle).toBeGreaterThan(1.5);
    expect(reClampImpacts).toBeGreaterThan(0);
  });

  it('caps active stones at MAX_STONES, casts fresh handfuls, and builds 4 watertight cleaved crystal variants', () => {
    const sim = new DomePendulumSimulation();
    for (let i = 0; i < 22; i++) {
      sim.tossStoneToward(new THREE.Vector3(0.2, DOME_CENTER.y + DOME_RADIUS, 0.2));
    }
    expect(sim.stones.length).toBeLessThanOrEqual(MAX_STONES);

    sim.castHandful(9);
    expect(sim.stones.length).toBe(9);
    expect(sim.stones.every((s) => s.phase === 'flying')).toBe(true);

    const { variants, dispose } = createStoneGeometryVariants();
    expect(variants.length).toBe(4);
    for (const geo of variants) {
      const pos = geo.attributes.position as THREE.BufferAttribute;
      const cap = geo.attributes.aCapMask as THREE.BufferAttribute;
      expect(pos.count).toBeGreaterThanOrEqual(36);

      let minY = Infinity;
      let maxCap = 0;
      for (let i = 0; i < pos.count; i++) {
        const y = pos.getY(i);
        if (y < minY) minY = y;
        const c = cap.getX(i);
        if (c > maxCap) maxCap = c;
      }
      expect(minY).toBeGreaterThan(-0.007);
      expect(minY).toBeLessThan(-0.005);
      expect(maxCap).toBeCloseTo(1.0, 4);
    }
    dispose();
  });
});
