import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  BOLT_TRAVEL,
  CASE_INNER_MAX_Y,
  CASE_INNER_MIN_Y,
  GATE_ALIGN_TOL_RAD,
  KEY_BIT_RADII,
  KEY_TALON_RADIUS,
  KEY_Y,
  LEVER_ALIGN_DEG,
  LEVER_ALIGN_RAD,
  LEVER_COUNT,
  LEVER_GATE_SPECS,
  PIVOT_X,
  PIVOT_Y,
  POCKET_PLAY_01,
  STUMP_X_LOCKED,
  STUMP_X_UNLOCKED,
  STUMP_Y,
  LockMechanism,
  computeKeyCamLiftRad,
  computeTalonBitTip,
} from '../src/days/21-bolt/lockMechanism';
import {
  buildDeadboltAssembly,
  buildForendAssembly,
  buildKeyAssembly,
  buildLeverPackAssembly,
  buildLockCaseAssembly,
  buildOakSpecimenBlock,
  buildSpringPackAssembly,
} from '../src/days/21-bolt/geometry';

function assertOutwardWinding(label: string, geo: THREE.BufferGeometry): void {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const idx = geo.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  expect(triCount, `${label} triCount`).toBeGreaterThan(0);

  geo.computeBoundingBox();
  const boxCenter = new THREE.Vector3();
  geo.boundingBox!.getCenter(boxCenter);

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const n = new THREE.Vector3();
  const cen = new THREE.Vector3();
  const bCrossC = new THREE.Vector3();

  let signedVol = 0;
  let outwardDot = 0;

  for (let i = 0; i < pos.count; i++) {
    expect(Number.isFinite(pos.getX(i)), `${label} finite X`).toBe(true);
    expect(Number.isFinite(pos.getY(i)), `${label} finite Y`).toBe(true);
    expect(Number.isFinite(pos.getZ(i)), `${label} finite Z`).toBe(true);
  }

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;

    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);

    bCrossC.crossVectors(b, c);
    signedVol += a.dot(bCrossC) / 6;

    ab.subVectors(b, a);
    ac.subVectors(c, a);
    n.crossVectors(ab, ac);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(boxCenter);
    outwardDot += n.dot(cen);
  }

  expect(signedVol, `${label} signedVolume`).toBeGreaterThan(0);
  expect(outwardDot, `${label} outwardDot`).toBeGreaterThan(0);
}

describe('Day 21 - Cutaway 5-Lever Chubb Mortise Deadbolt', () => {
  it('verifies exact 5-lever bitting-to-gate alignment geometry and case clearances', () => {
    // All 5 levers must have distinct alignment heights and distinct stepped key bitting radii
    expect(new Set(LEVER_ALIGN_DEG).size).toBe(LEVER_COUNT);
    expect(new Set(KEY_BIT_RADII).size).toBe(LEVER_COUNT);

    // Key drive bit at 6 o'clock (phi=0) must cleanly clear bottom case wall
    const tipDown = { x: 0, y: 0 };
    computeTalonBitTip(0, tipDown);
    expect(tipDown.y).toBeCloseTo(KEY_Y - KEY_TALON_RADIUS, 5);
    expect(tipDown.y).toBeGreaterThan(CASE_INNER_MIN_Y + 0.02);

    for (let i = 0; i < LEVER_COUNT; i++) {
      // At 6 o'clock (rest), key bit imparts zero lift
      expect(computeKeyCamLiftRad(i, 0)).toBe(0);

      // At 12 o'clock (phi = PI), stepped bitting tooth i lifts lever i to exact alignment angle
      const liftAtTop = computeKeyCamLiftRad(i, Math.PI);
      expect(liftAtTop).toBeCloseTo(LEVER_ALIGN_RAD[i], 6);

      // Verify rotating lever i by +alignRad brings BOTH gate endpoints dead-level to world Y = STUMP_Y
      const spec = LEVER_GATE_SPECS[i];
      const cosA = Math.cos(spec.alignRad);
      const sinA = Math.sin(spec.alignRad);

      const worldUnlockedX =
        PIVOT_X + spec.gateUnlockedLocalX * cosA - spec.gateUnlockedLocalY * sinA;
      const worldUnlockedY =
        PIVOT_Y + spec.gateUnlockedLocalX * sinA + spec.gateUnlockedLocalY * cosA;

      const worldLockedX =
        PIVOT_X + spec.gateLockedLocalX * cosA - spec.gateLockedLocalY * sinA;
      const worldLockedY =
        PIVOT_Y + spec.gateLockedLocalX * sinA + spec.gateLockedLocalY * cosA;

      expect(worldUnlockedX).toBeCloseTo(STUMP_X_UNLOCKED, 5);
      expect(worldUnlockedY).toBeCloseTo(STUMP_Y, 5);
      expect(worldLockedX).toBeCloseTo(STUMP_X_LOCKED, 5);
      expect(worldLockedY).toBeCloseTo(STUMP_Y, 5);
      expect(worldLockedX - worldUnlockedX).toBeCloseTo(BOLT_TRAVEL, 5);

      // Verify lifted lever top corner stays cleanly inside the steel lock case ceiling
      const maxTopWorldY = PIVOT_Y + 0.98 * sinA + 0.262 * cosA;
      expect(maxTopWorldY).toBeLessThan(CASE_INNER_MAX_Y - 0.05);
    }
  });

  it('blocks bolt stump collision whenever ANY 1 of the 5 levers is under-lifted or over-lifted', () => {
    for (let badIdx = 0; badIdx < LEVER_COUNT; badIdx++) {
      for (const offsetSign of [-1, 1]) {
        // Test blocking from UNLOCKED -> attempting to push bolt outward
        const lockU = new LockMechanism(false);
        lockU.beginBoltDrag();
        lockU.setBoltDragTarget(1.0);

        let blockedUnlocked = false;
        for (let step = 0; step < 30; step++) {
          for (let i = 0; i < LEVER_COUNT; i++) {
            const target =
              i === badIdx
                ? Math.max(0, LEVER_ALIGN_RAD[i] + offsetSign * (GATE_ALIGN_TOL_RAD * 2.4))
                : LEVER_ALIGN_RAD[i];
            lockU.setLeverAngleImmediate(i, target);
          }
          lockU.step(1 / 60);
          if (lockU.boltBlocked) blockedUnlocked = true;
        }

        expect(blockedUnlocked, `unlocked block lever ${badIdx} sign ${offsetSign}`).toBe(true);
        expect(lockU.gatesAligned).toBe(false);
        expect(lockU.boltThrow01).toBeLessThanOrEqual(POCKET_PLAY_01 + 1e-4);
        expect(lockU.bindingLeverIndex).toBe(badIdx);

        // Test blocking from LOCKED -> attempting to pull bolt inward
        const lockL = new LockMechanism(true);
        lockL.beginBoltDrag();
        lockL.setBoltDragTarget(0.0);

        let blockedLocked = false;
        for (let step = 0; step < 30; step++) {
          for (let i = 0; i < LEVER_COUNT; i++) {
            const target =
              i === badIdx
                ? Math.max(0, LEVER_ALIGN_RAD[i] + offsetSign * (GATE_ALIGN_TOL_RAD * 2.4))
                : LEVER_ALIGN_RAD[i];
            lockL.setLeverAngleImmediate(i, target);
          }
          lockL.step(1 / 60);
          if (lockL.boltBlocked) blockedLocked = true;
        }

        expect(blockedLocked, `locked block lever ${badIdx} sign ${offsetSign}`).toBe(true);
        expect(lockL.gatesAligned).toBe(false);
        expect(lockL.boltThrow01).toBeGreaterThanOrEqual(1.0 - POCKET_PLAY_01 - 1e-4);
        expect(lockL.bindingLeverIndex).toBe(badIdx);
      }
    }

    // Conversely, when ALL 5 levers are aligned (picked manually with key at rest), bolt slides freely 0 -> 1
    const pickedLock = new LockMechanism(false);
    pickedLock.beginBoltDrag();
    pickedLock.setBoltDragTarget(1.0);
    for (let s = 0; s < 50; s++) {
      for (let i = 0; i < LEVER_COUNT; i++) {
        pickedLock.setLeverAngleImmediate(i, LEVER_ALIGN_RAD[i]);
      }
      pickedLock.step(1 / 60);
    }
    expect(pickedLock.boltBlocked).toBe(false);
    expect(pickedLock.boltThrow01).toBeGreaterThan(0.95);

    // And when the key is parked at 12 o'clock (gates aligned by key), dragging the bolt slides freely
    // AND physically back-drives the key along its talon arc!
    const keyParkedLock = new LockMechanism(false);
    keyParkedLock.keyAngleRad = Math.PI;
    keyParkedLock.step(1 / 60);
    expect(keyParkedLock.gatesAligned).toBe(true);

    keyParkedLock.beginBoltDrag();
    keyParkedLock.setBoltDragTarget(1.0);
    for (let s = 0; s < 55; s++) {
      keyParkedLock.step(1 / 60);
    }
    expect(keyParkedLock.boltBlocked).toBe(false);
    expect(keyParkedLock.boltThrow01).toBeGreaterThan(0.95);
    expect(keyParkedLock.keyAngleRad).toBeGreaterThan(Math.PI + 0.35);
  });

  it('completes a full unlock -> lock -> unlock 360-degree key cycle with staggered lever lift and detent snap', () => {
    const lock = new LockMechanism(false); // start unlocked
    expect(lock.isLocked).toBe(false);
    expect(lock.boltThrow01).toBe(0);

    // Phase 1: Full 360-deg CW key turn -> LOCKS deadbolt (0 -> 1)
    lock.triggerToggleTurn();
    expect(lock.isAutoTurning).toBe(true);

    let sawGatesAlignedDuringLock = false;
    let maxDistinctLeverAngles = 0;

    for (let f = 0; f < 180; f++) {
      lock.step(1 / 60);
      if (lock.gatesAligned) {
        sawGatesAlignedDuringLock = true;
        const rounded = new Set(Array.from(lock.leverAnglesRad, (r) => Math.round(r * 100)));
        maxDistinctLeverAngles = Math.max(maxDistinctLeverAngles, rounded.size);
      }
    }

    expect(sawGatesAlignedDuringLock).toBe(true);
    expect(maxDistinctLeverAngles).toBe(LEVER_COUNT);
    expect(lock.isAutoTurning).toBe(false);
    expect(lock.keyAngleRad).toBeCloseTo(Math.PI * 2, 2);
    expect(lock.boltThrow01).toBeCloseTo(1.0, 2);
    expect(lock.isLocked).toBe(true);
    // All 5 levers must have dropped cleanly back to 0 deg into locked pocket!
    for (let i = 0; i < LEVER_COUNT; i++) {
      expect(lock.leverAnglesRad[i]).toBeLessThan(0.005);
    }
    expect(lock.gatesAligned).toBe(false);

    // Phase 2: Full 360-deg CCW key turn -> UNLOCKS deadbolt (1 -> 0)
    lock.triggerToggleTurn();
    let sawGatesAlignedDuringUnlock = false;
    for (let f = 0; f < 180; f++) {
      lock.step(1 / 60);
      if (lock.gatesAligned) sawGatesAlignedDuringUnlock = true;
    }

    expect(sawGatesAlignedDuringUnlock).toBe(true);
    expect(lock.isAutoTurning).toBe(false);
    expect(lock.keyAngleRad).toBeCloseTo(0, 2);
    expect(lock.boltThrow01).toBeCloseTo(0.0, 2);
    expect(lock.isLocked).toBe(false);
    for (let i = 0; i < LEVER_COUNT; i++) {
      expect(lock.leverAnglesRad[i]).toBeLessThan(0.005);
    }

    // Phase 3: Manual circular drag 360-deg CW locks again cleanly
    lock.beginKeyDrag();
    const steps = 120;
    for (let s = 0; s < steps; s++) {
      lock.applyKeyDragDelta((Math.PI * 2) / steps);
      lock.step(1 / 60);
    }
    lock.endKeyDrag();
    for (let s = 0; s < 40; s++) lock.step(1 / 60);
    expect(lock.isLocked).toBe(true);
    expect(lock.boltThrow01).toBeGreaterThan(0.95);
  });

  it('exerts monotonic phosphor-bronze leaf-spring restoring torque and snaps released levers cleanly to rest', () => {
    const lock = new LockMechanism(true);

    for (let i = 0; i < LEVER_COUNT; i++) {
      const restTorque = lock.getLeafSpringRestoringTorque(i);
      expect(restTorque).toBeGreaterThan(0);

      // Manually drag lever i upward against its phosphor-bronze leaf spring
      const liftRad = (16 * Math.PI) / 180;
      lock.beginLeverDrag(i, liftRad);
      for (let s = 0; s < 35; s++) {
        lock.step(1 / 60);
      }
      expect(lock.leverAnglesRad[i]).toBeGreaterThan(liftRad * 0.85);
      expect(lock.leverSpringFlex01[i]).toBeGreaterThan(0.6);

      const liftedTorque = lock.getLeafSpringRestoringTorque(i);
      expect(liftedTorque).toBeGreaterThan(restTorque);

      // Release lever -> leaf spring snaps it cleanly back to exact rest (0 rad, zero negative overshoot)
      lock.endLeverDrag();
      for (let s = 0; s < 45; s++) {
        lock.step(1 / 60);
      }
      expect(lock.leverAnglesRad[i]).toBe(0);
      expect(lock.leverAngVels[i]).toBe(0);
      expect(lock.leverSpringFlex01[i]).toBe(0);
    }
  });

  it('traps levers physically inside the gate tunnel while the bolt stump is midway through', () => {
    const lock = new LockMechanism(false);
    lock.keyAngleRad = Math.PI;
    lock.step(1 / 60);
    expect(lock.gatesAligned).toBe(true);

    // Move bolt halfway through the 5 aligned lever gates
    lock.boltThrow01 = 0.5;
    // Rotate key away to 0 so cam no longer supports the levers; bolt stump must trap them inside gate clearance!
    lock.keyAngleRad = 0;
    for (let s = 0; s < 30; s++) {
      lock.step(1 / 60);
    }
    for (let i = 0; i < LEVER_COUNT; i++) {
      expect(Math.abs(lock.leverAnglesRad[i] - LEVER_ALIGN_RAD[i])).toBeLessThanOrEqual(
        GATE_ALIGN_TOL_RAD
      );
    }
    expect(lock.gatesAligned).toBe(true);
  });

  it('enforces outward triangle winding (signedVolume > 0, outwardDot > 0) on all custom procedural meshes', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mats: any = {
      charredOak: dummyMat,
      gunmetalCase: dummyMat,
      bluedSteelBolt: dummyMat,
      forendBrass: dummyMat,
      phosphorBronze: dummyMat,
      keyBrass: dummyMat,
      hardenedSteel: dummyMat,
      leverMaterials: Array.from({ length: LEVER_COUNT }, () => dummyMat),
    };

    const oak = buildOakSpecimenBlock(mats);
    oak.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) assertOutwardWinding(`oak/${obj.name}`, obj.geometry);
    });
    oak.dispose();

    const forend = buildForendAssembly(mats);
    forend.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) assertOutwardWinding(`forend/${obj.name}`, obj.geometry);
    });
    forend.dispose();

    const lockCase = buildLockCaseAssembly(mats);
    lockCase.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) assertOutwardWinding(`case/${obj.name}`, obj.geometry);
    });
    lockCase.dispose();

    const deadbolt = buildDeadboltAssembly(mats);
    deadbolt.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) assertOutwardWinding(`bolt/${obj.name}`, obj.geometry);
    });
    deadbolt.dispose();

    const levers = buildLeverPackAssembly(mats);
    expect(levers.leverMeshes.length).toBe(LEVER_COUNT);
    for (const mesh of levers.leverMeshes) {
      assertOutwardWinding(`lever/${mesh.name}`, mesh.geometry);
    }
    levers.dispose();

    const springs = buildSpringPackAssembly(mats);
    expect(springs.springs.length).toBe(LEVER_COUNT);
    // Check winding at rest (0 deg)
    springs.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) assertOutwardWinding(`springRest/${obj.name}`, obj.geometry);
    });
    // Check winding under maximum compression (17 deg)
    const flexed = new Float64Array(LEVER_COUNT).fill((17 * Math.PI) / 180);
    springs.update(flexed);
    for (const sp of springs.springs) {
      assertOutwardWinding(`springFlex/${sp.mesh.name}`, sp.geometry);
    }
    springs.dispose();

    const key = buildKeyAssembly(mats);
    key.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh && obj.name !== 'KeyBowHitProxy') {
        assertOutwardWinding(`key/${obj.name}`, obj.geometry);
      }
    });
    key.dispose();

    dummyMat.dispose();
  });
});
