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
  getStoneVisualPolarity,
  type PulleyKinematicsFrame,
} from '../src/days/18-pendulum/physics';
import {
  buildDomeAssembly,
  buildGimbalPulleyAssembly,
  buildPendulumBob,
  createStoneGeometryVariants,
} from '../src/days/18-pendulum/geometry';
import { createHammeredIronSurfaceMaps } from '../src/days/18-pendulum/domeSurface';

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

  it('maintains 60s chaotic non-trapping swings across the dome with real radial height breathing', () => {
    const sim = new DomePendulumSimulation();
    const visitedStones = new Set<number>();
    let zeroCrossingsX = 0;
    let prevX = sim.pendulum.pos.x;
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    let minRadOff = Infinity;
    let maxRadOff = -Infinity;

    for (let f = 0; f < 3600; f++) {
      sim.step(1 / 60);
      const p = sim.pendulum;
      if ((prevX < 0 && p.pos.x >= 0) || (prevX > 0 && p.pos.x <= 0)) {
        zeroCrossingsX++;
      }
      prevX = p.pos.x;

      if (p.pos.x < minX) minX = p.pos.x;
      if (p.pos.x > maxX) maxX = p.pos.x;
      if (p.pos.z < minZ) minZ = p.pos.z;
      if (p.pos.z > maxZ) maxZ = p.pos.z;
      if (p.radialOffset < minRadOff) minRadOff = p.radialOffset;
      if (p.radialOffset > maxRadOff) maxRadOff = p.radialOffset;

      for (const s of sim.stones) {
        if (p.pos.distanceTo(s.pos) < 0.48) {
          visitedStones.add(s.id);
        }
      }
      sim.clearSurfaceEvents();
    }

    expect(zeroCrossingsX).toBeGreaterThanOrEqual(10);
    expect(maxX - minX).toBeGreaterThan(0.75);
    expect(maxZ - minZ).toBeGreaterThan(0.75);
    expect(visitedStones.size).toBeGreaterThanOrEqual(4);
    expect(maxRadOff - minRadOff).toBeGreaterThan(0.04);
  });

  it('enforces outward triangle winding (outwardDot > 0 and signedVol > 0) across all procedural meshes', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const mats: any = {
      domeIron: dummyMat,
      domeUnderside: dummyMat,
      boneCalciteBob: dummyMat,
      boneCalciteRim: dummyMat,
      agedBronze: dummyMat,
      satinBrass: dummyMat,
      plumbWire: dummyMat,
      cinnabarStone: dummyMat,
      verdigrisStone: dummyMat,
      trailRibbon: dummyMat,
    };

    const assertOutward = (label: string, geo: THREE.BufferGeometry, checkPositiveVol = true) => {
      const pos = geo.attributes.position as THREE.BufferAttribute;
      const idx = geo.index;
      const triCount = idx ? idx.count / 3 : pos.count / 3;
      let signedVol = 0;
      let outwardDot = 0;
      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      const c = new THREE.Vector3();
      const ab = new THREE.Vector3();
      const ac = new THREE.Vector3();
      const n = new THREE.Vector3();
      const cen = new THREE.Vector3();
      geo.computeBoundingBox();
      const boxCenter = new THREE.Vector3();
      geo.boundingBox!.getCenter(boxCenter);

      for (let t = 0; t < triCount; t++) {
        const i0 = idx ? idx.getX(t * 3) : t * 3;
        const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
        const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
        a.fromBufferAttribute(pos, i0);
        b.fromBufferAttribute(pos, i1);
        c.fromBufferAttribute(pos, i2);
        signedVol += a.dot(b.clone().cross(c)) / 6;
        ab.subVectors(b, a);
        ac.subVectors(c, a);
        n.crossVectors(ab, ac);
        cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(boxCenter);
        outwardDot += n.dot(cen);
      }

      expect(outwardDot, `${label} outwardDot`).toBeGreaterThan(0);
      if (checkPositiveVol) {
        expect(signedVol, `${label} signedVol`).toBeGreaterThan(0);
      }
    };

    const dome = buildDomeAssembly(mats);
    dome.group.traverse((obj: any) => {
      if (!obj.isMesh) return;
      if (obj.name === 'DomeUndersidePlate') {
        const pos = obj.geometry.attributes.position as THREE.BufferAttribute;
        const idx = obj.geometry.index;
        const triCount = idx ? idx.count / 3 : pos.count / 3;
        const a = new THREE.Vector3();
        const b = new THREE.Vector3();
        const c = new THREE.Vector3();
        const ab = new THREE.Vector3();
        const ac = new THREE.Vector3();
        const n = new THREE.Vector3();
        let sumY = 0;
        for (let t = 0; t < triCount; t++) {
          const i0 = idx ? idx.getX(t * 3) : t * 3;
          const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
          const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
          a.fromBufferAttribute(pos, i0);
          b.fromBufferAttribute(pos, i1);
          c.fromBufferAttribute(pos, i2);
          ab.subVectors(b, a);
          ac.subVectors(c, a);
          n.crossVectors(ab, ac).normalize();
          sumY += n.y;
        }
        expect(sumY / triCount, 'DomeUndersidePlate avgNormalY').toBeLessThan(-0.95);
      } else {
        assertOutward(`dome/${obj.name || obj.geometry.type}`, obj.geometry);
      }
    });
    dome.dispose();

    const gimbal = buildGimbalPulleyAssembly(mats);
    gimbal.group.traverse((obj: any) => {
      if (obj.isMesh) assertOutward(`gimbal/${obj.name || obj.geometry.type}`, obj.geometry);
    });
    gimbal.counterweightGroup.traverse((obj: any) => {
      if (obj.isMesh) assertOutward(`cw/${obj.name || obj.geometry.type}`, obj.geometry);
    });
    gimbal.dispose();

    const bob = buildPendulumBob(mats);
    bob.group.traverse((obj: any) => {
      if (obj.isMesh && obj.name !== 'PendulumBobHit') {
        assertOutward(`bob/${obj.name || obj.geometry.type}`, obj.geometry);
      }
    });
    bob.dispose();

    const stones = createStoneGeometryVariants();
    stones.variants.forEach((g, idx) => assertOutward(`stone/v${idx}`, g));
    stones.dispose();
    dummyMat.dispose();
  });

  it('supports direct stone dragging, wakes opposite-polarity neighbors, and settles with zero buzz', () => {
    const sim = new DomePendulumSimulation();
    sim.stones.length = 0;
    sim.pendulum.isGrabbed = true;
    sim.pendulum.grabTarget.set(0, DOME_CENTER.y + 0.6, -1.2);

    const mover = sim.tossStoneToward(
      new THREE.Vector3(-0.35, DOME_CENTER.y + DOME_RADIUS, 0),
      new THREE.Vector3(-0.35, 0.45, 0),
      1
    );
    const neighbor = sim.tossStoneToward(
      new THREE.Vector3(0.20, DOME_CENTER.y + DOME_RADIUS, 0),
      new THREE.Vector3(0.20, 0.45, 0),
      -1
    );

    for (let i = 0; i < 150; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
    }

    const startNeighborPos = neighbor.pos.clone();
    let dragScuffs = 0;
    for (let i = 0; i < 40; i++) {
      const targetX = -0.35 + (i / 39) * 0.40;
      expect(sim.setDraggedStone(mover.id, new THREE.Vector3(targetX, DOME_CENTER.y + DOME_RADIUS, 0))).toBe(true);
      sim.step(1 / 60);
      dragScuffs += sim.scuffEvents.length;
      sim.clearSurfaceEvents();
    }
    sim.releaseDraggedStone();
    expect(dragScuffs).toBeGreaterThan(0);

    for (let i = 0; i < 180; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
    }

    expect(mover.phase).toBe('clung');
    expect(neighbor.phase).toBe('clung');
    expect(neighbor.pos.distanceTo(startNeighborPos)).toBeGreaterThan(0.02);

    // Zero buzz after settling
    const prevMover = mover.pos.clone();
    let maxJitter = 0;
    for (let i = 0; i < 60; i++) {
      sim.step(1 / 60);
      maxJitter = Math.max(maxJitter, mover.pos.distanceTo(prevMover));
      prevMover.copy(mover.pos);
      sim.clearSurfaceEvents();
    }
    expect(maxJitter).toBe(0);
  });

  it('swaps stone visual polarity at mid-flip apex and decays bobWakeSheen cleanly to zero', () => {
    const sim = new DomePendulumSimulation();
    const st = sim.stones[0];
    const origPol = st.polarity;
    expect(getStoneVisualPolarity(st)).toBe(origPol);

    expect(sim.flipStone(st.id)).toBe(true);
    expect(st.polarity).toBe(-origPol);
    // Immediately after flip starts (before upside-down apex u = 0.5), visual polarity still shows origPol
    expect(getStoneVisualPolarity(st)).toBe(origPol);

    for (let i = 0; i < 12; i++) {
      sim.step(1 / 60);
      sim.clearSurfaceEvents();
    }
    // Past u = 0.5 (~0.17s), visual polarity swaps cleanly to new polarity
    expect(getStoneVisualPolarity(st)).toBe(-origPol);

    // Verify bobWakeSheen decays cleanly to 0 when scuffs are quiet
    const maps = createHammeredIronSurfaceMaps();
    maps.stampBobWake(new THREE.Vector3(0, 1, 0), 1.0, 2.0, 0.1);
    maps.decayAndUpload(1 / 60);
    expect(maps.getActiveScuffEnergy()).toBeGreaterThan(0);

    for (let i = 0; i < 240; i++) {
      maps.decayAndUpload(0.1);
    }
    expect(maps.getActiveScuffEnergy()).toBe(0);
    maps.dispose();
  });
});
