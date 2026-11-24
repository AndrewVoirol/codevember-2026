import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  BOLT_HEAD_Z_HALF,
  BOLT_TAIL_Z_BACK,
  BOLT_TAIL_Z_FRONT,
  BOLT_TRAVEL,
  CASE_INNER_MAX_X,
  CASE_INNER_MAX_Y,
  CASE_INNER_MIN_Y,
  FOREND_X,
  GATE_ALIGN_TOL_RAD,
  GATE_HALF_HEIGHT,
  GATE_POCKET_BOT_R_Y,
  GATE_POCKET_LEFT_INNER_X,
  GATE_POCKET_LEFT_OUTER_X,
  GATE_POCKET_RIGHT_INNER_X,
  GATE_POCKET_RIGHT_OUTER_X,
  GATE_POCKET_TOP_Y,
  GEAR_B1_CURTAIN,
  GEAR_B2_IDLER,
  GEAR_B3F_STEP,
  GEAR_B3R_PINION,
  GEAR_B4A_KEYHOME,
  GEAR_U1A_PINION,
  GEAR_U1B_SPUR,
  GEAR_U2A_DRIVEN,
  KEY_BIT_HALF_WIDTH,
  KEY_BIT_RADII,
  KEY_TALON_RADIUS,
  KEY_X,
  KEY_Y,
  LEVER_ALIGN_DEG,
  LEVER_ALIGN_RAD,
  LEVER_COUNT,
  LEVER_GATE_SPECS,
  LEVER_MAX_RAD,
  LEVER_THICKNESS,
  LOWER_KEYHOME_CAM_SPEC,
  PIVOT_X,
  PIVOT_Y,
  POCKET_PLAY_01,
  REST_PIN_RADIUS,
  REST_PIN_X,
  REST_PIN_Y,
  STUMP_HEIGHT,
  STUMP_WIDTH,
  STUMP_X_LOCKED,
  STUMP_X_UNLOCKED,
  STUMP_Y,
  STUMP_Z_BACK,
  STUMP_Z_FRONT,
  TALON_ENGAGE_END_RAD,
  TALON_ENGAGE_START_RAD,
  UPPER_DETENT_CAM_SPEC,
  UPPER_RACK_SPEC,
  LockMechanism,
  buildInvoluteGearPolygon2D,
  buildInvoluteRackPolygon2D,
  computeKeyCamLiftRad,
  computeLeverPhysicalMaxRad,
  computeLowerTrainAnglesRad,
  computeTalonBitTip,
  computeUpperDetentWheelAngleRad,
  computeUpperPinionAngleRad,
  evalDetentCamRadius,
  getLeverZ,
  sampleLeverBellyLocalY,
  solveDetentPawlContact,
  type Point2D,
} from '../src/days/21-bolt/lockMechanism';
import { buildGearTrainAssembly } from '../src/days/21-bolt/gearTrain';
import {
  BOLT_HEAD_HALF_Y,
  FOREND_HALF_Z,
  FOREND_MAX_Y,
  FOREND_MIN_Y,
  FOREND_OUTER_FACE_X,
  FOREND_SLOT_HALF_Y,
  FOREND_SLOT_HALF_Z,
  OAK_OUTER_MAX_X,
  OAK_REBATE_MAX_Y,
  OAK_REBATE_MIN_Y,
  OAK_REBATE_X,
  OAK_REBATE_Z_BACK,
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
      expect(liftAtTop).toBeCloseTo(LEVER_ALIGN_RAD[i], 4);

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
      const maxTopWorldY = PIVOT_Y + 0.992 * sinA + 0.264 * cosA;
      expect(maxTopWorldY).toBeLessThan(CASE_INNER_MAX_Y - 0.04);
    }
  });

  it('enforces exact 360-degree key-tooth to lever-belly contact (zero clipping, zero air gap)', () => {
    for (let i = 0; i < LEVER_COUNT; i++) {
      const R = KEY_BIT_RADII[i];
      const halfAng = Math.asin(KEY_BIT_HALF_WIDTH / R);

      for (let deg = 0; deg < 360; deg += 2) {
        const phi = (deg * Math.PI) / 180;
        const theta = computeKeyCamLiftRad(i, phi);

        // During the entire talon engagement stroke, every lever holds exact gate alignment
        if (phi >= TALON_ENGAGE_START_RAD && phi <= TALON_ENGAGE_END_RAD) {
          expect(Math.abs(theta - LEVER_ALIGN_RAD[i]), `lever ${i} dwell at ${deg} deg`).toBeLessThan(0.001);
        }

        const c = Math.cos(theta);
        const s = Math.sin(theta);
        let minGap = Infinity;

        for (const f of [-1, -0.5, 0, 0.5, 1]) {
          const samplePhi = phi + f * halfAng;
          const wx = KEY_X - R * Math.sin(samplePhi);
          const wy = KEY_Y - R * Math.cos(samplePhi);
          const dx = wx - PIVOT_X;
          const dy = wy - PIVOT_Y;
          const lx = dx * c + dy * s;
          const ly = -dx * s + dy * c;
          const bellyY = sampleLeverBellyLocalY(i, lx);
          if (bellyY !== null) {
            // Zero clipping into solid brass lever belly (< 0.0004 chord tolerance)
            expect(ly - bellyY, `lever ${i} clip at ${deg} deg`).toBeLessThan(0.0005);
            minGap = Math.min(minGap, bellyY - ly);
          }
        }

        // Whenever the key is lifting lever i, physical tooth contact must be flush (zero floating air gap!)
        if (theta > 0.004 && Number.isFinite(minGap)) {
          expect(minGap, `lever ${i} air gap at ${deg} deg`).toBeLessThan(0.0008);
        }
      }
    }
  });

  it('verifies lever-stack Z-clearances, stump-to-pocket geometry, leaf-spring contact, and flush forend rebate', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mats: any = {
      charredOak: dummyMat,
      gunmetalCase: dummyMat,
      bluedSteelBolt: dummyMat,
      forendBrass: dummyMat,
      machinedGearBrass: dummyMat,
      machinedBronzeRack: dummyMat,
      phosphorBronze: dummyMat,
      keyBrass: dummyMat,
      hardenedSteel: dummyMat,
      leverMaterials: Array.from({ length: LEVER_COUNT }, () => dummyMat),
    };

    const lockCase = buildLockCaseAssembly(mats);
    const deadbolt = buildDeadboltAssembly(mats);
    const levers = buildLeverPackAssembly(mats);
    const springs = buildSpringPackAssembly(mats);
    const forend = buildForendAssembly(mats);
    const oak = buildOakSpecimenBlock(mats);

    // 1. Lever-stack Z-clearances: zero overlap between bolt tail, rear bushing, 5 levers, 4 washers, front collar
    const rearBushing = lockCase.group.getObjectByName('PivotRearThrustBushing') as THREE.Mesh;
    rearBushing.geometry.computeBoundingBox();
    expect(rearBushing.geometry.boundingBox!.max.z).toBeLessThan(
      getLeverZ(0) - LEVER_THICKNESS * 0.5
    );

    const boltTail = deadbolt.group.getObjectByName('DeadboltTailAndTalon') as THREE.Mesh;
    boltTail.geometry.computeBoundingBox();
    expect(boltTail.geometry.boundingBox!.min.z).toBeCloseTo(BOLT_TAIL_Z_BACK, 3);
    expect(boltTail.geometry.boundingBox!.max.z).toBeCloseTo(BOLT_TAIL_Z_FRONT, 3);
    expect(boltTail.geometry.boundingBox!.max.z).toBeLessThan(
      getLeverZ(0) - LEVER_THICKNESS * 0.5 - 0.01
    );

    for (let i = 0; i < LEVER_COUNT; i++) {
      levers.leverMeshes[i].geometry.computeBoundingBox();
      const lz = getLeverZ(i);
      const lMinZ = lz + levers.leverMeshes[i].geometry.boundingBox!.min.z;
      const lMaxZ = lz + levers.leverMeshes[i].geometry.boundingBox!.max.z;
      expect(lMaxZ - lMinZ).toBeCloseTo(LEVER_THICKNESS, 3);

      if (i < LEVER_COUNT - 1) {
        const washer = lockCase.group.getObjectByName(`LeverSpacerWasher_${i}`) as THREE.Mesh;
        washer.geometry.computeBoundingBox();
        const wMinZ = washer.geometry.boundingBox!.min.z;
        const wMaxZ = washer.geometry.boundingBox!.max.z;
        const nextMinZ = getLeverZ(i + 1) - LEVER_THICKNESS * 0.5;
        expect(wMinZ, `washer ${i} clears lever ${i}`).toBeGreaterThan(lMaxZ);
        expect(wMaxZ, `washer ${i} clears lever ${i + 1}`).toBeLessThan(nextMinZ);
      }
    }

    // Bolt stump spans all 5 levers cleanly and matches exact STUMP_WIDTH x STUMP_HEIGHT
    const stump = deadbolt.group.getObjectByName('BoltStump') as THREE.Mesh;
    stump.geometry.computeBoundingBox();
    const sBox = stump.geometry.boundingBox!;
    expect(sBox.max.x - sBox.min.x).toBeCloseTo(STUMP_WIDTH, 3);
    expect(sBox.max.y - sBox.min.y).toBeCloseTo(STUMP_HEIGHT, 3);
    expect(sBox.min.z).toBeCloseTo(STUMP_Z_BACK, 3);
    expect(sBox.max.z).toBeCloseTo(STUMP_Z_FRONT, 3);
    expect(sBox.max.z).toBeGreaterThan(getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.01);

    // 2. Stump-to-pocket clearance under pocket play & full picking angle [0 .. physicalMax]: ZERO clipping!
    const playWorldX = POCKET_PLAY_01 * BOLT_TRAVEL;
    const hw = STUMP_WIDTH * 0.5;
    const hh = STUMP_HEIGHT * 0.5;
    // Rest unlocked play clears left locking cheek cleanly
    expect(STUMP_X_UNLOCKED - PIVOT_X + playWorldX + hw).toBeLessThan(GATE_POCKET_LEFT_INNER_X);
    expect(STUMP_X_UNLOCKED - PIVOT_X - hw).toBeGreaterThan(GATE_POCKET_LEFT_OUTER_X);
    // Rest locked play clears right locking cheek cleanly
    expect(STUMP_X_LOCKED - PIVOT_X - playWorldX - hw).toBeGreaterThan(GATE_POCKET_RIGHT_INNER_X);
    expect(STUMP_X_LOCKED - PIVOT_X + hw).toBeLessThan(GATE_POCKET_RIGHT_OUTER_X);
    expect(STUMP_Y - PIVOT_Y + hh).toBeLessThan(GATE_POCKET_TOP_Y);
    // Perpendicular gate clearance exceeds stump half-height cleanly
    expect(GATE_HALF_HEIGHT).toBeGreaterThan(hh + 0.005);

    for (let i = 0; i < LEVER_COUNT; i++) {
      const maxLift = computeLeverPhysicalMaxRad(i, 1.0);
      const c = Math.cos(maxLift);
      const s = Math.sin(maxLift);
      const dxL = STUMP_X_LOCKED - PIVOT_X;
      const dy = STUMP_Y - PIVOT_Y;
      const lowestCornerY = -dxL * s + dy * c - (hw * s + hh * c);
      const rightmostCornerX = dxL * c + dy * s + (hw * c + hh * s);
      expect(lowestCornerY, `lever ${i} pocket floor at max pick`).toBeGreaterThanOrEqual(
        GATE_POCKET_BOT_R_Y - 1e-4
      );
      expect(rightmostCornerX, `lever ${i} pocket outer wall at max pick`).toBeLessThan(
        GATE_POCKET_RIGHT_OUTER_X
      );
    }

    // 3. Phosphor-bronze leaf springs: clean post clearance, flush tangential contact, ceiling clearance, constant length
    const restDiag = springs.springs[0].inspectAtAngle(0);
    for (const deg of [0, 6.0, 11.2, 16.5, (LEVER_MAX_RAD * 180) / Math.PI]) {
      const d = springs.springs[0].inspectAtAngle((deg * Math.PI) / 180);
      expect(d.rootDistFromPost, `spring root outside steel post at ${deg} deg`).toBeGreaterThan(0.048);
      expect(d.tipContactGap, `spring flush tip contact at ${deg} deg`).toBeLessThan(0.001);
      expect(d.minClearanceAboveLever, `spring never clips lever at ${deg} deg`).toBeGreaterThanOrEqual(-1e-4);
      expect(d.maxArcY, `spring clears case ceiling at ${deg} deg`).toBeLessThan(CASE_INNER_MAX_Y - 0.025);
      expect(Math.abs(d.arcLength - restDiag.arcLength), `spring constant arc length at ${deg} deg`).toBeLessThan(0.015);
    }

    // 4. Forend faceplate flush inside oak rebate (zero overlap!) & deadbolt head cleanly clears forend slot
    const plate = forend.group.getObjectByName('ForendFaceplate') as THREE.Mesh;
    plate.geometry.computeBoundingBox();
    const pBox = plate.geometry.boundingBox!;
    expect(pBox.min.x).toBeGreaterThan(OAK_REBATE_X);
    expect(pBox.max.x).toBeCloseTo(OAK_OUTER_MAX_X, 3);
    expect(pBox.max.x).toBeCloseTo(FOREND_OUTER_FACE_X, 3);
    expect(pBox.min.x).toBeCloseTo(FOREND_X, 3);
    expect(pBox.min.y).toBeGreaterThan(OAK_REBATE_MIN_Y);
    expect(pBox.max.y).toBeLessThan(OAK_REBATE_MAX_Y);
    expect(pBox.min.y).toBeCloseTo(FOREND_MIN_Y, 3);
    expect(pBox.max.y).toBeCloseTo(FOREND_MAX_Y, 3);
    expect(pBox.min.z).toBeGreaterThan(OAK_REBATE_Z_BACK);
    expect(pBox.max.z).toBeCloseTo(FOREND_HALF_Z, 3);

    const head = deadbolt.group.getObjectByName('DeadboltHead') as THREE.Mesh;
    head.geometry.computeBoundingBox();
    const hBox = head.geometry.boundingBox!;
    expect(hBox.max.y - STUMP_Y).toBeCloseTo(BOLT_HEAD_HALF_Y, 3);
    expect(hBox.max.y - STUMP_Y).toBeLessThan(FOREND_SLOT_HALF_Y - 0.006);
    expect(hBox.max.z).toBeCloseTo(BOLT_HEAD_Z_HALF, 3);
    expect(hBox.max.z).toBeLessThan(FOREND_SLOT_HALF_Z - 0.006);

    oak.dispose();
    forend.dispose();
    springs.dispose();
    levers.dispose();
    deadbolt.dispose();
    lockCase.dispose();
    dummyMat.dispose();
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

  it('completes a full unlock -> lock -> unlock 360-degree key cycle and retreats smoothly from wrong-way stops', () => {
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

    // Phase 4: Pushing wrong way (CW while already locked) hits stop cleanly, then reversing CCW retreats immediately!
    const lockedStart = lock.keyAngleRad;
    lock.beginKeyDrag();
    for (let s = 0; s < 40; s++) {
      lock.applyKeyDragDelta(0.08);
      lock.step(1 / 60);
    }
    const stoppedAngle = lock.keyAngleRad;
    expect(stoppedAngle - lockedStart).toBeLessThan(Math.PI); // blocked before crossing talon!
    for (let s = 0; s < 28; s++) {
      lock.applyKeyDragDelta(-0.08);
      lock.step(1 / 60);
    }
    expect(lock.keyAngleRad).toBeLessThan(stoppedAngle - 1.0); // zero sticky lock on retreat!
    lock.endKeyDrag();
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
      machinedGearBrass: dummyMat,
      machinedBronzeRack: dummyMat,
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
    springs.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) assertOutwardWinding(`springRest/${obj.name}`, obj.geometry);
    });
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

    const gearTrain = buildGearTrainAssembly(mats);
    const lock = new LockMechanism(true);
    gearTrain.update(
      lock.upperPinionAngleRad,
      lock.upperDetentAngleRad,
      lock.upperPawlAngleRad,
      lock.curtainB1AngleRad,
      lock.curtainB2AngleRad,
      lock.curtainB3AngleRad,
      lock.keyHomeB4AngleRad,
      lock.lowerPawlAngleRad
    );
    gearTrain.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) {
        assertOutwardWinding(`gearTrainRest/${obj.name}`, obj.geometry);
      }
    });
    // Also verify outward winding on flexed detent springs mid-throw & mid-key-turn
    const midLock = new LockMechanism(false);
    midLock.keyAngleRad = Math.PI;
    midLock.boltThrow01 = 0.5;
    midLock.syncGearTrainStates();
    gearTrain.update(
      midLock.upperPinionAngleRad,
      midLock.upperDetentAngleRad,
      midLock.upperPawlAngleRad,
      midLock.curtainB1AngleRad,
      midLock.curtainB2AngleRad,
      midLock.curtainB3AngleRad,
      midLock.keyHomeB4AngleRad,
      midLock.lowerPawlAngleRad
    );
    assertOutwardWinding('gearTrainFlex/UpperDetentLeafSpring', gearTrain.upperDetentSpring.geometry);
    assertOutwardWinding('gearTrainFlex/LowerKeyHomeLeafSpring', gearTrain.lowerDetentSpring.geometry);
    gearTrain.dispose();

    dummyMat.dispose();
  });

  it('verifies exact pitch-circle tangency, gear ratios, and ZERO involute tooth overlap across 360° & full throw', () => {
    // Helper: exact 2D segment intersection check (strictly detects any tooth polygon penetration!)
    const segsIntersect = (a: Point2D, b: Point2D, c: Point2D, d: Point2D): boolean => {
      const orient = (p: Point2D, q: Point2D, r: Point2D) =>
        (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
      const o1 = orient(a, b, c);
      const o2 = orient(a, b, d);
      const o3 = orient(c, d, a);
      const o4 = orient(c, d, b);
      return o1 * o2 < -1e-11 && o3 * o4 < -1e-11;
    };

    const transformPoly = (pts: readonly Point2D[], cx: number, cy: number, ang: number): Point2D[] => {
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      return pts.map((p) => ({ x: cx + p.x * c - p.y * s, y: cy + p.x * s + p.y * c }));
    };

    const assertNoPolygonEdgeCrossing = (label: string, polyA: readonly Point2D[], polyB: readonly Point2D[]) => {
      for (let i = 0; i < polyA.length; i++) {
        const a0 = polyA[i];
        const a1 = polyA[(i + 1) % polyA.length];
        const aMinX = Math.min(a0.x, a1.x);
        const aMaxX = Math.max(a0.x, a1.x);
        const aMinY = Math.min(a0.y, a1.y);
        const aMaxY = Math.max(a0.y, a1.y);
        for (let j = 0; j < polyB.length; j++) {
          const b0 = polyB[j];
          const b1 = polyB[(j + 1) % polyB.length];
          if (
            Math.max(b0.x, b1.x) < aMinX ||
            Math.min(b0.x, b1.x) > aMaxX ||
            Math.max(b0.y, b1.y) < aMinY ||
            Math.min(b0.y, b1.y) > aMaxY
          ) {
            continue;
          }
          expect(segsIntersect(a0, a1, b0, b1), `${label} edge ${i} vs ${j}`).toBe(false);
        }
      }
    };

    // 1. Exact pitch-circle tangency for every gear-to-gear pair & rack-to-pinion pair
    const gearPairs = [
      { name: 'U1B<->U2A', g1: GEAR_U1B_SPUR, g2: GEAR_U2A_DRIVEN },
      { name: 'B1<->B2', g1: GEAR_B1_CURTAIN, g2: GEAR_B2_IDLER },
      { name: 'B2<->B3F', g1: GEAR_B2_IDLER, g2: GEAR_B3F_STEP },
      { name: 'B3R<->B4A', g1: GEAR_B3R_PINION, g2: GEAR_B4A_KEYHOME },
    ];
    for (const pair of gearPairs) {
      const dist = Math.hypot(pair.g2.cx - pair.g1.cx, pair.g2.cy - pair.g1.cy);
      expect(Math.abs(dist - (pair.g1.pitchRadius + pair.g2.pitchRadius)), pair.name).toBeLessThan(1e-6);
    }
    expect(Math.abs(GEAR_U1A_PINION.cy - (UPPER_RACK_SPEC.pitchLineY + GEAR_U1A_PINION.pitchRadius))).toBeLessThan(1e-6);

    // 2. Exact angular velocity ratios across full strokes (including exact 1:1 compound Key-Home ratio!)
    const u1_0 = computeUpperPinionAngleRad(0);
    const u1_1 = computeUpperPinionAngleRad(1);
    const u2_0 = computeUpperDetentWheelAngleRad(0);
    const u2_1 = computeUpperDetentWheelAngleRad(1);
    expect((u2_1 - u2_0) / (u1_1 - u1_0)).toBeCloseTo(-GEAR_U1B_SPUR.teeth / GEAR_U2A_DRIVEN.teeth, 6);

    const low0 = computeLowerTrainAnglesRad(0);
    const low2Pi = computeLowerTrainAnglesRad(Math.PI * 2);
    expect((low2Pi.b2 - low0.b2) / (low2Pi.b1 - low0.b1)).toBeCloseTo(-GEAR_B1_CURTAIN.teeth / GEAR_B2_IDLER.teeth, 6);
    expect((low2Pi.b3 - low0.b3) / (low2Pi.b2 - low0.b2)).toBeCloseTo(-GEAR_B2_IDLER.teeth / GEAR_B3F_STEP.teeth, 6);
    expect((low2Pi.b4 - low0.b4) / (low2Pi.b3 - low0.b3)).toBeCloseTo(-GEAR_B3R_PINION.teeth / GEAR_B4A_KEYHOME.teeth, 6);
    // Exact 1:1 compound gear ratio: 1 full CW key revolution rotates B4 by exact +2*PI CCW!
    expect(low2Pi.b4 - low0.b4).toBeCloseTo(Math.PI * 2, 6);

    // 3. Zero involute tooth overlap across full bolt throw [0, 1] for Rack<->U1A and U1B<->U2A
    const rackLocal = buildInvoluteRackPolygon2D(UPPER_RACK_SPEC);
    const u1aLocal = buildInvoluteGearPolygon2D(GEAR_U1A_PINION);
    const u1bLocal = buildInvoluteGearPolygon2D(GEAR_U1B_SPUR);
    const u2aLocal = buildInvoluteGearPolygon2D(GEAR_U2A_DRIVEN);

    for (let step = 0; step <= 40; step++) {
      const throw01 = step / 40;
      const boltX = throw01 * BOLT_TRAVEL;
      const u1Ang = computeUpperPinionAngleRad(throw01);
      const u2Ang = computeUpperDetentWheelAngleRad(throw01);

      const rackWorld = rackLocal.map((p) => ({ x: p.x + boltX, y: p.y }));
      const u1aWorld = transformPoly(u1aLocal, GEAR_U1A_PINION.cx, GEAR_U1A_PINION.cy, u1Ang);
      const u1bWorld = transformPoly(u1bLocal, GEAR_U1B_SPUR.cx, GEAR_U1B_SPUR.cy, u1Ang);
      const u2aWorld = transformPoly(u2aLocal, GEAR_U2A_DRIVEN.cx, GEAR_U2A_DRIVEN.cy, u2Ang);

      assertNoPolygonEdgeCrossing(`Rack<->U1A throw=${throw01}`, rackWorld, u1aWorld);
      assertNoPolygonEdgeCrossing(`U1B<->U2A throw=${throw01}`, u1bWorld, u2aWorld);
    }

    // 4. Zero involute tooth overlap across full 360-degree key revolution for B1<->B2, B2<->B3F, B3R<->B4A
    const b1Local = buildInvoluteGearPolygon2D(GEAR_B1_CURTAIN);
    const b2Local = buildInvoluteGearPolygon2D(GEAR_B2_IDLER);
    const b3fLocal = buildInvoluteGearPolygon2D(GEAR_B3F_STEP);
    const b3rLocal = buildInvoluteGearPolygon2D(GEAR_B3R_PINION);
    const b4aLocal = buildInvoluteGearPolygon2D(GEAR_B4A_KEYHOME);

    for (let deg = 0; deg < 360; deg += 10) {
      const keyRad = (deg * Math.PI) / 180;
      const angs = computeLowerTrainAnglesRad(keyRad);

      const b1W = transformPoly(b1Local, GEAR_B1_CURTAIN.cx, GEAR_B1_CURTAIN.cy, angs.b1);
      const b2W = transformPoly(b2Local, GEAR_B2_IDLER.cx, GEAR_B2_IDLER.cy, angs.b2);
      const b3fW = transformPoly(b3fLocal, GEAR_B3F_STEP.cx, GEAR_B3F_STEP.cy, angs.b3);
      const b3rW = transformPoly(b3rLocal, GEAR_B3R_PINION.cx, GEAR_B3R_PINION.cy, angs.b3);
      const b4aW = transformPoly(b4aLocal, GEAR_B4A_KEYHOME.cx, GEAR_B4A_KEYHOME.cy, angs.b4);

      assertNoPolygonEdgeCrossing(`B1<->B2 key=${deg}deg`, b1W, b2W);
      assertNoPolygonEdgeCrossing(`B2<->B3F key=${deg}deg`, b2W, b3fW);
      assertNoPolygonEdgeCrossing(`B3R<->B4A key=${deg}deg`, b3rW, b4aW);
    }
  });

  it('verifies detent pawl engagement, V-notch seating, and spatial clearances inside lock cavity', () => {
    const sol = {
      pawlWorldAngleRad: 0,
      beakWorldX: 0,
      beakWorldY: 0,
      contactWorldAngleRad: 0,
      camRadiusAtContact: 0,
      lift01: 0,
    };

    // 1. Upper Deadbolt Detent Pawl seats DEAD-FLUSH in V-notches at unlocked(0) & locked(1), climbs smoothly mid-throw!
    solveDetentPawlContact(UPPER_DETENT_CAM_SPEC, computeUpperDetentWheelAngleRad(0), sol);
    expect(sol.lift01).toBeLessThan(0.005);
    solveDetentPawlContact(UPPER_DETENT_CAM_SPEC, computeUpperDetentWheelAngleRad(1), sol);
    expect(sol.lift01).toBeLessThan(0.005);
    solveDetentPawlContact(UPPER_DETENT_CAM_SPEC, computeUpperDetentWheelAngleRad(0.5), sol);
    expect(sol.lift01).toBeGreaterThan(0.85);

    for (let s = 0; s <= 40; s++) {
      const th = s / 40;
      const camAng = computeUpperDetentWheelAngleRad(th);
      solveDetentPawlContact(UPPER_DETENT_CAM_SPEC, camAng, sol);
      const beakDist = Math.hypot(
        sol.beakWorldX - UPPER_DETENT_CAM_SPEC.cx,
        sol.beakWorldY - UPPER_DETENT_CAM_SPEC.cy
      );
      const exactCamR = evalDetentCamRadius(UPPER_DETENT_CAM_SPEC, sol.contactWorldAngleRad - camAng);
      expect(Math.abs(beakDist - (exactCamR + UPPER_DETENT_CAM_SPEC.beakRadius))).toBeLessThan(1e-4);
    }

    // 2. Lower Key-Home Detent Pawl seats DEAD-FLUSH at keyAngle = 0 and 2*PI, climbs cleanly when turned!
    solveDetentPawlContact(LOWER_KEYHOME_CAM_SPEC, computeLowerTrainAnglesRad(0).b4, sol);
    expect(sol.lift01).toBeLessThan(0.005);
    solveDetentPawlContact(LOWER_KEYHOME_CAM_SPEC, computeLowerTrainAnglesRad(Math.PI * 2).b4, sol);
    expect(sol.lift01).toBeLessThan(0.005);
    solveDetentPawlContact(LOWER_KEYHOME_CAM_SPEC, computeLowerTrainAnglesRad(Math.PI).b4, sol);
    expect(sol.lift01).toBeGreaterThan(0.95);

    // 3. Hard spatial clearances (zero forend collision, zero case wall collision, zero RestPin/key-bit/lever Z collision!)
    expect(UPPER_RACK_SPEC.barMaxLocalX + BOLT_TRAVEL).toBeLessThan(CASE_INNER_MAX_X - 0.015);
    expect(UPPER_RACK_SPEC.zFront).toBeLessThan(getLeverZ(0) - LEVER_THICKNESS * 0.5 - 0.015);
    expect(GEAR_U1B_SPUR.zFront).toBeLessThan(getLeverZ(0) - LEVER_THICKNESS * 0.5 - 0.01);

    const boltHeadTopY = STUMP_Y + BOLT_HEAD_HALF_Y;
    const boltHeadBotY = STUMP_Y - BOLT_HEAD_HALF_Y;
    expect(GEAR_U1B_SPUR.cy - GEAR_U1B_SPUR.tipRadius).toBeGreaterThan(boltHeadTopY + 0.004);
    expect(GEAR_U2A_DRIVEN.cy - GEAR_U2A_DRIVEN.tipRadius).toBeGreaterThan(boltHeadTopY + 0.01);
    expect(GEAR_U2A_DRIVEN.cy + GEAR_U2A_DRIVEN.tipRadius).toBeLessThan(CASE_INNER_MAX_Y - 0.015);

    expect(GEAR_B1_CURTAIN.zBack).toBeGreaterThan(getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.006);
    expect(GEAR_B4A_KEYHOME.cy + GEAR_B4A_KEYHOME.tipRadius).toBeLessThan(boltHeadBotY - 0.10);
    expect(GEAR_B4A_KEYHOME.cy - GEAR_B4A_KEYHOME.tipRadius).toBeGreaterThan(CASE_INNER_MIN_Y + 0.05);
    expect(GEAR_B4A_KEYHOME.cx + GEAR_B4A_KEYHOME.tipRadius).toBeLessThan(CASE_INNER_MAX_X - 0.10);

    // RestPin clearance from B2, B3F, B3R, B4A
    for (const g of [GEAR_B2_IDLER, GEAR_B3F_STEP, GEAR_B3R_PINION, GEAR_B4A_KEYHOME]) {
      const dRest = Math.hypot(g.cx - REST_PIN_X, g.cy - REST_PIN_Y);
      expect(dRest, `${g.name} clears RestPin`).toBeGreaterThan(g.tipRadius + REST_PIN_RADIUS + 0.01);
    }

    // Full-depth rear pinion B3R clears rotating key bitting teeth (max radius ~ 0.296)
    const maxBitR = Math.max(...KEY_BIT_RADII);
    const dKeyB3 = Math.hypot(GEAR_B3R_PINION.cx - KEY_X, GEAR_B3R_PINION.cy - KEY_Y);
    expect(dKeyB3).toBeGreaterThan(GEAR_B3R_PINION.tipRadius + maxBitR + 0.008);

    // 4. Verify misaligned lever freezes Train A (upper rack-and-pinion + detent wheel stay locked!)
    const jammed = new LockMechanism(true);
    const u1Start = jammed.upperPinionAngleRad;
    const u2Start = jammed.upperDetentAngleRad;
    jammed.beginBoltDrag();
    jammed.setBoltDragTarget(0.0);
    for (let s = 0; s < 30; s++) jammed.step(1 / 60);
    expect(jammed.boltBlocked).toBe(true);
    expect(Math.abs(jammed.upperPinionAngleRad - u1Start)).toBeLessThan(0.10);
    expect(Math.abs(jammed.upperDetentAngleRad - u2Start)).toBeLessThan(0.10);
  });
});
