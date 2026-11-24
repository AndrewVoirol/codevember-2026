import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  BOLT_HEAD_Z_HALF,
  BOLT_TAIL_Z_BACK,
  BOLT_TAIL_Z_FRONT,
  BOLT_TRAVEL,
  CAM_PHASE_OFFSET_RAD,
  CASE_INNER_MAX_X,
  CASE_INNER_MAX_Y,
  CASE_INNER_MIN_Y,
  CUTAWAY_FRONT_Z,
  CUTAWAY_WIN_MAX_X,
  CUTAWAY_WIN_MAX_Y,
  CUTAWAY_WIN_MIN_X,
  CUTAWAY_WIN_MIN_Y,
  FOREND_X,
  GATE_ALIGN_TOL_RAD,
  GATE_HALF_HEIGHT,
  GATE_POCKET_BOT_R_Y,
  GATE_POCKET_LEFT_INNER_X,
  GATE_POCKET_LEFT_OUTER_X,
  GATE_POCKET_RIGHT_INNER_X,
  GATE_POCKET_RIGHT_OUTER_X,
  GATE_POCKET_TOP_Y,
  GEAR_B1_KNOB_DRIVE,
  GEAR_B2A_PINION,
  GEAR_B2B_STEP,
  GEAR_B3F_TRANSFER,
  GEAR_B3R_PINION,
  GEAR_B4A_DETENT,
  GEAR_C1_CAM_DRIVEN,
  GEAR_U1A_PINION,
  GEAR_U1B_SPUR,
  GEAR_U2A_DRIVEN,
  KEY_BIT_HALF_WIDTH,
  KEY_BIT_RADII,
  KEY_TALON_RADIUS,
  KEY_X,
  KEY_Y,
  KNOB_MAX_RAD,
  KNOB_MIN_RAD,
  KNOB_TO_CAM_RATIO,
  LEVER_ALIGN_DEG,
  LEVER_ALIGN_RAD,
  LEVER_COUNT,
  LEVER_GATE_SPECS,
  LEVER_MAX_RAD,
  LEVER_THICKNESS,
  LOWER_DETENT_CAM_SPEC,
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
  knobToCamAngleRad,
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
  buildCamBarrelAssembly,
  buildDeadboltAssembly,
  buildForendAssembly,
  buildLeverPackAssembly,
  buildLockCaseAssembly,
  buildOakSpecimenBlock,
  buildSpringPackAssembly,
  buildThumbTurnAssembly,
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

function makeDummyMats() {
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
  return { mats, dummyMat };
}

function trueWorldBounds(mesh: THREE.Mesh): { min: THREE.Vector3; max: THREE.Vector3 } {
  mesh.updateWorldMatrix(true, false);
  const pos = mesh.geometry.attributes.position as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  const min = new THREE.Vector3(Infinity, Infinity, Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
    min.min(v);
    max.max(v);
  }
  return { min, max };
}

describe('Day 21 - Cutaway Geared 5-Lever Thumb-Turn Mortise Deadbolt', () => {
  it('verifies exact 5-lever bitting-to-gate alignment geometry and quarter-turn cam endpoints', () => {
    expect(new Set(LEVER_ALIGN_DEG).size).toBe(LEVER_COUNT);
    expect(new Set(KEY_BIT_RADII).size).toBe(LEVER_COUNT);

    // At 0 deg Vertical (Unlocked, cam=45 deg) and 90 deg Horizontal (Locked, cam=315 deg),
    // all 5 levers rest cleanly at 0 deg and the talon bit clears the bottom case wall!
    for (const knobRad of [KNOB_MIN_RAD, KNOB_MAX_RAD]) {
      const camRad = knobToCamAngleRad(knobRad);
      const tip = { x: 0, y: 0 };
      computeTalonBitTip(camRad, tip);
      expect(tip.y).toBeGreaterThan(CASE_INNER_MIN_Y + 0.05);
      for (let i = 0; i < LEVER_COUNT; i++) {
        expect(computeKeyCamLiftRad(i, camRad)).toBe(0);
      }
    }

    // At 45 deg Mid-Throw (cam=180 deg), all 5 stepped cam lobes lift every lever to exact alignment!
    const midCam = knobToCamAngleRad((KNOB_MIN_RAD + KNOB_MAX_RAD) * 0.5);
    expect(midCam).toBeCloseTo(Math.PI, 6);

    for (let i = 0; i < LEVER_COUNT; i++) {
      const liftAtTop = computeKeyCamLiftRad(i, midCam);
      expect(liftAtTop).toBeCloseTo(LEVER_ALIGN_RAD[i], 4);

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

      const maxTopWorldY = PIVOT_Y + 0.992 * sinA + 0.264 * cosA;
      expect(maxTopWorldY).toBeLessThan(CUTAWAY_WIN_MAX_Y - 0.03);
    }
  });

  it('enforces exact 360-degree cam-lobe to lever-belly contact (zero clipping, zero air gap)', () => {
    for (let i = 0; i < LEVER_COUNT; i++) {
      const R = KEY_BIT_RADII[i];
      const halfAng = Math.asin(KEY_BIT_HALF_WIDTH / R);

      for (let deg = 0; deg < 360; deg += 2) {
        const phi = (deg * Math.PI) / 180;
        const theta = computeKeyCamLiftRad(i, phi);

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
            expect(ly - bellyY, `lever ${i} clip at ${deg} deg`).toBeLessThan(0.0005);
            minGap = Math.min(minGap, bellyY - ly);
          }
        }

        if (theta > 0.004 && Number.isFinite(minGap)) {
          expect(minGap, `lever ${i} air gap at ${deg} deg`).toBeLessThan(0.0008);
        }
      }
    }
  });

  it('verifies lever-stack Z-clearances, staggered leaf springs, detent click springs, and flush forend rebate', () => {
    const { mats, dummyMat } = makeDummyMats();

    const lockCase = buildLockCaseAssembly(mats);
    const deadbolt = buildDeadboltAssembly(mats);
    const levers = buildLeverPackAssembly(mats);
    const springs = buildSpringPackAssembly(mats);
    const forend = buildForendAssembly(mats);
    const oak = buildOakSpecimenBlock(mats);
    const gearTrain = buildGearTrainAssembly(mats);

    // 1. Lever-stack Z-clearances
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

    const stump = deadbolt.group.getObjectByName('BoltStump') as THREE.Mesh;
    stump.geometry.computeBoundingBox();
    const sBox = stump.geometry.boundingBox!;
    expect(sBox.max.x - sBox.min.x).toBeCloseTo(STUMP_WIDTH, 3);
    expect(sBox.max.y - sBox.min.y).toBeCloseTo(STUMP_HEIGHT, 3);
    expect(sBox.min.z).toBeCloseTo(STUMP_Z_BACK, 3);
    expect(sBox.max.z).toBeCloseTo(STUMP_Z_FRONT, 3);
    expect(sBox.max.z).toBeGreaterThan(getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.01);

    // 2. Stump-to-pocket clearance under pocket play & full picking angle
    const playWorldX = POCKET_PLAY_01 * BOLT_TRAVEL;
    const hw = STUMP_WIDTH * 0.5;
    const hh = STUMP_HEIGHT * 0.5;
    expect(STUMP_X_UNLOCKED - PIVOT_X + playWorldX + hw).toBeLessThan(GATE_POCKET_LEFT_INNER_X);
    expect(STUMP_X_UNLOCKED - PIVOT_X - hw).toBeGreaterThan(GATE_POCKET_LEFT_OUTER_X);
    expect(STUMP_X_LOCKED - PIVOT_X - playWorldX - hw).toBeGreaterThan(GATE_POCKET_RIGHT_INNER_X);
    expect(STUMP_X_LOCKED - PIVOT_X + hw).toBeLessThan(GATE_POCKET_RIGHT_OUTER_X);
    expect(STUMP_Y - PIVOT_Y + hh).toBeLessThan(GATE_POCKET_TOP_Y);
    expect(GATE_HALF_HEIGHT).toBeGreaterThan(hh + 0.005);

    for (let i = 0; i < LEVER_COUNT; i++) {
      const maxLift = computeLeverPhysicalMaxRad(i, 1.0);
      const c = Math.cos(maxLift);
      const s = Math.sin(maxLift);
      const dxL = STUMP_X_LOCKED - PIVOT_X;
      const dy = STUMP_Y - PIVOT_Y;
      const lowestCornerY = -dxL * s + dy * c - (hw * s + hh * c);
      const rightmostCornerX = dxL * c + dy * s + (hw * c + hh * s);
      expect(lowestCornerY, `lever ${i} pocket floor`).toBeGreaterThanOrEqual(GATE_POCKET_BOT_R_Y - 1e-4);
      expect(rightmostCornerX, `lever ${i} pocket wall`).toBeLessThan(GATE_POCKET_RIGHT_OUTER_X);
    }

    // 3. All 5 staggered phosphor-bronze lever leaf springs: flush tangent contact, constant length, strictly inside glass window!
    for (let i = 0; i < LEVER_COUNT; i++) {
      const restDiag = springs.springs[i].inspectAtAngle(0);
      for (const deg of [0, 6.0, 11.2, 16.5, (LEVER_MAX_RAD * 180) / Math.PI]) {
        const d = springs.springs[i].inspectAtAngle((deg * Math.PI) / 180);
        expect(d.rootDistFromPost, `spring ${i} root outside post`).toBeGreaterThan(0.048);
        expect(d.tipContactGap, `spring ${i} flush tip at ${deg}`).toBeLessThan(0.001);
        expect(d.minClearanceAboveLever, `spring ${i} never clips lever at ${deg}`).toBeGreaterThanOrEqual(-1e-4);
        expect(d.maxArcY, `spring ${i} inside cutaway window top at ${deg}`).toBeLessThan(CUTAWAY_WIN_MAX_Y - 0.025);
        expect(Math.abs(d.arcLength - restDiag.arcLength), `spring ${i} constant arc length`).toBeLessThan(0.02);
      }
    }

    // 4. Both upper & lower detent click springs: exact flush tangent contact on pawl spine & zero cam clipping!
    const sol = {
      pawlWorldAngleRad: 0,
      beakWorldX: 0,
      beakWorldY: 0,
      contactWorldAngleRad: 0,
      camRadiusAtContact: 0,
      lift01: 0,
    };
    for (const boltT of [0, 0.5, 1.0]) {
      solveDetentPawlContact(UPPER_DETENT_CAM_SPEC, computeUpperDetentWheelAngleRad(boltT), sol);
      const diag = gearTrain.upperDetentSpring.inspectAtPawlAngle(sol.pawlWorldAngleRad);
      expect(diag.tipContactGap, `upper detent spring flush at bolt=${boltT}`).toBeLessThan(0.001);
      expect(diag.minDistFromCam, `upper detent spring clears cam`).toBeGreaterThan(UPPER_DETENT_CAM_SPEC.outerRadius + 0.015);
      expect(diag.maxY, `upper detent spring inside cutaway window`).toBeLessThan(CUTAWAY_WIN_MAX_Y - 0.015);
    }
    for (const knobT of [0, 0.5, 1.0]) {
      const kRad = KNOB_MIN_RAD + knobT * (KNOB_MAX_RAD - KNOB_MIN_RAD);
      solveDetentPawlContact(LOWER_DETENT_CAM_SPEC, computeLowerTrainAnglesRad(kRad).b4, sol);
      const diag = gearTrain.lowerDetentSpring.inspectAtPawlAngle(sol.pawlWorldAngleRad);
      expect(diag.tipContactGap, `lower detent spring flush at knob=${knobT}`).toBeLessThan(0.001);
      expect(diag.minDistFromCam, `lower detent spring clears cam`).toBeGreaterThan(LOWER_DETENT_CAM_SPEC.outerRadius + 0.015);
      expect(diag.minY, `lower detent spring inside cutaway window`).toBeGreaterThan(CUTAWAY_WIN_MIN_Y + 0.05);
    }

    // 5. Forend faceplate flush inside oak rebate & deadbolt clears forend slot
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

    gearTrain.dispose();
    oak.dispose();
    forend.dispose();
    springs.dispose();
    levers.dispose();
    deadbolt.dispose();
    lockCase.dispose();
    dummyMat.dispose();
  });

  it('enforces cutaway front-plane Z-clearance and 100% cutaway-window visibility across Locked, Mid-Throw & Unlocked', () => {
    const { mats, dummyMat } = makeDummyMats();

    const root = new THREE.Group();
    const lockCase = buildLockCaseAssembly(mats);
    const deadbolt = buildDeadboltAssembly(mats);
    const levers = buildLeverPackAssembly(mats);
    const springs = buildSpringPackAssembly(mats);
    const camBarrel = buildCamBarrelAssembly(mats);
    const gearTrain = buildGearTrainAssembly(mats);
    const thumbTurn = buildThumbTurnAssembly(mats);

    root.add(lockCase.group, deadbolt.group, levers.group, springs.group, camBarrel.group, gearTrain.group, thumbTurn.group);

    const lock = new LockMechanism(true);
    for (const [label, knobRad, bThrow] of [
      ['UNLOCKED', KNOB_MIN_RAD, 0.0],
      ['MID_THROW', (KNOB_MIN_RAD + KNOB_MAX_RAD) * 0.5, 0.5],
      ['LOCKED', KNOB_MAX_RAD, 1.0],
    ] as const) {
      lock.knobAngleRad = knobRad;
      lock.boltThrow01 = bThrow;
      for (let i = 0; i < LEVER_COUNT; i++) {
        lock.leverAnglesRad[i] = computeKeyCamLiftRad(i, lock.camAngleRad);
      }
      lock.syncGearTrainStates();

      deadbolt.group.position.x = lock.boltThrow01 * BOLT_TRAVEL;
      camBarrel.group.rotation.z = -lock.camAngleRad;
      thumbTurn.group.rotation.z = -lock.knobAngleRad;
      for (let i = 0; i < LEVER_COUNT; i++) levers.leverPivots[i].rotation.z = lock.leverAnglesRad[i];
      springs.update(lock.leverAnglesRad);
      gearTrain.update(
        lock.upperPinionAngleRad,
        lock.upperDetentAngleRad,
        lock.upperPawlAngleRad,
        lock.knobDriveB1AngleRad,
        lock.stepUpB2AngleRad,
        lock.camDrivenC1AngleRad,
        lock.transferB3AngleRad,
        lock.lowerDetentB4AngleRad,
        lock.lowerPawlAngleRad
      );
      root.updateMatrixWorld(true);

      // Every internal gear, lever, spring, pawl & bridge stays strictly inside the open cutaway window!
      const checkInternalGroup = (grp: THREE.Group, maxZLimit: number) => {
        grp.traverse((obj: THREE.Object3D) => {
          if (!(obj instanceof THREE.Mesh)) return;
          const b = trueWorldBounds(obj);
          expect(b.max.z, `${label} ${obj.name} inside cutaway depth`).toBeLessThanOrEqual(maxZLimit);
          expect(b.min.x, `${label} ${obj.name} inside left rim`).toBeGreaterThan(CUTAWAY_WIN_MIN_X);
          expect(b.max.x, `${label} ${obj.name} inside right rim`).toBeLessThan(CUTAWAY_WIN_MAX_X);
          expect(b.min.y, `${label} ${obj.name} inside bottom rim`).toBeGreaterThan(CUTAWAY_WIN_MIN_Y);
          expect(b.max.y, `${label} ${obj.name} inside top rim`).toBeLessThan(CUTAWAY_WIN_MAX_Y);
        });
      };

      checkInternalGroup(levers.group, CUTAWAY_FRONT_Z);
      checkInternalGroup(springs.group, CUTAWAY_FRONT_Z);
      checkInternalGroup(gearTrain.group, CUTAWAY_FRONT_Z + 0.008);
      checkInternalGroup(camBarrel.group, CUTAWAY_FRONT_Z);

      for (const postName of [
        'LeverPivotPost',
        'SpringAbutmentPost',
        'SpringReactionPin',
        'LeverRestStopPin',
        'BoltGuidePin',
        'CamCenterGuidePin',
      ]) {
        const post = lockCase.group.getObjectByName(postName) as THREE.Mesh;
        const pb = trueWorldBounds(post);
        expect(pb.max.z, `${label} ${postName} inside cutaway depth`).toBeLessThanOrEqual(CUTAWAY_FRONT_Z);
      }
    }

    thumbTurn.dispose();
    gearTrain.dispose();
    camBarrel.dispose();
    springs.dispose();
    levers.dispose();
    deadbolt.dispose();
    lockCase.dispose();
    dummyMat.dispose();
  });

  it('blocks bolt stump collision whenever ANY 1 of the 5 levers is misaligned, and back-drives 3:1 knob when aligned', () => {
    for (let badIdx = 0; badIdx < LEVER_COUNT; badIdx++) {
      for (const offsetSign of [-1, 1]) {
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

    // When all 5 levers are picked manually, bolt slides freely 0 -> 1
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

    // When thumb-turn knob is parked mid-throw (45 deg -> cam=180 deg, gates aligned), dragging the bolt
    // slides freely AND physically back-drives the 3:1 gear train & front thumb-turn knob!
    const knobParked = new LockMechanism(false);
    knobParked.knobAngleRad = (KNOB_MIN_RAD + KNOB_MAX_RAD) * 0.5;
    knobParked.step(1 / 60);
    expect(knobParked.gatesAligned).toBe(true);

    const startKnob = knobParked.knobAngleRad;
    knobParked.beginBoltDrag();
    knobParked.setBoltDragTarget(1.0);
    for (let s = 0; s < 55; s++) {
      knobParked.step(1 / 60);
    }
    expect(knobParked.boltBlocked).toBe(false);
    expect(knobParked.boltThrow01).toBeGreaterThan(0.95);
    expect(knobParked.knobAngleRad).toBeGreaterThan(startKnob + 0.12);
  });

  it('completes a full quarter-turn unlock -> lock -> unlock cycle (0 deg Vertical <-> 90 deg Horizontal) with hard stops', () => {
    const lock = new LockMechanism(false); // Start unlocked (0 deg Vertical)
    expect(lock.isLocked).toBe(false);
    expect(lock.knobAngleRad).toBeCloseTo(KNOB_MIN_RAD, 5);
    expect(lock.camAngleRad).toBeCloseTo(CAM_PHASE_OFFSET_RAD, 5);
    expect(lock.boltThrow01).toBe(0);

    // Phase 1: Motorized 90 deg CW turn -> LOCKS deadbolt (0 -> 1, Horizontal)
    lock.triggerToggleTurn();
    expect(lock.isAutoTurning).toBe(true);

    let sawGatesAlignedDuringLock = false;
    let maxDistinctLeverAngles = 0;

    for (let f = 0; f < 150; f++) {
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
    expect(lock.knobAngleRad).toBeCloseTo(KNOB_MAX_RAD, 3);
    expect(lock.camAngleRad).toBeCloseTo(CAM_PHASE_OFFSET_RAD + KNOB_TO_CAM_RATIO * KNOB_MAX_RAD, 3);
    expect(lock.boltThrow01).toBeCloseTo(1.0, 2);
    expect(lock.isLocked).toBe(true);
    for (let i = 0; i < LEVER_COUNT; i++) {
      expect(lock.leverAnglesRad[i]).toBeLessThan(0.005);
    }
    expect(lock.gatesAligned).toBe(false);

    // Phase 2: Motorized 90 deg CCW turn -> UNLOCKS deadbolt (1 -> 0, Vertical)
    lock.triggerToggleTurn();
    let sawGatesAlignedDuringUnlock = false;
    for (let f = 0; f < 150; f++) {
      lock.step(1 / 60);
      if (lock.gatesAligned) sawGatesAlignedDuringUnlock = true;
    }

    expect(sawGatesAlignedDuringUnlock).toBe(true);
    expect(lock.isAutoTurning).toBe(false);
    expect(lock.knobAngleRad).toBeCloseTo(KNOB_MIN_RAD, 3);
    expect(lock.boltThrow01).toBeCloseTo(0.0, 2);
    expect(lock.isLocked).toBe(false);
    for (let i = 0; i < LEVER_COUNT; i++) {
      expect(lock.leverAnglesRad[i]).toBeLessThan(0.005);
    }

    // Phase 3: Manual circular drag 90 deg CW locks again cleanly & respects hard stop at PI/2
    lock.beginKnobDrag();
    const steps = 90;
    for (let s = 0; s < steps; s++) {
      lock.applyKnobDragDelta((Math.PI * 0.65) / steps); // intentionally push past 90 deg stop
      lock.step(1 / 60);
    }
    expect(lock.knobAngleRad).toBeLessThanOrEqual(KNOB_MAX_RAD + 1e-6);
    lock.endKnobDrag();
    for (let s = 0; s < 35; s++) lock.step(1 / 60);
    expect(lock.isLocked).toBe(true);
    expect(lock.boltThrow01).toBeGreaterThan(0.95);
    expect(lock.knobAngleRad).toBeCloseTo(KNOB_MAX_RAD, 3);
  });

  it('exerts monotonic phosphor-bronze leaf-spring restoring torque and traps levers inside gate tunnel', () => {
    const lock = new LockMechanism(true);

    for (let i = 0; i < LEVER_COUNT; i++) {
      const restTorque = lock.getLeafSpringRestoringTorque(i);
      expect(restTorque).toBeGreaterThan(0);

      const liftRad = (16 * Math.PI) / 180;
      lock.beginLeverDrag(i, liftRad);
      for (let s = 0; s < 35; s++) lock.step(1 / 60);
      expect(lock.leverAnglesRad[i]).toBeGreaterThan(liftRad * 0.85);
      expect(lock.getLeafSpringRestoringTorque(i)).toBeGreaterThan(restTorque);

      lock.endLeverDrag();
      for (let s = 0; s < 45; s++) lock.step(1 / 60);
      expect(lock.leverAnglesRad[i]).toBe(0);
      expect(lock.leverAngVels[i]).toBe(0);
    }

    // Gate tunnel lever trapping when bolt stump is midway through
    const trapLock = new LockMechanism(false);
    trapLock.knobAngleRad = (KNOB_MIN_RAD + KNOB_MAX_RAD) * 0.5;
    trapLock.step(1 / 60);
    expect(trapLock.gatesAligned).toBe(true);
    trapLock.boltThrow01 = 0.5;
    trapLock.knobAngleRad = KNOB_MIN_RAD;
    for (let s = 0; s < 30; s++) trapLock.step(1 / 60);
    for (let i = 0; i < LEVER_COUNT; i++) {
      expect(Math.abs(trapLock.leverAnglesRad[i] - LEVER_ALIGN_RAD[i])).toBeLessThanOrEqual(
        GATE_ALIGN_TOL_RAD
      );
    }
    expect(trapLock.gatesAligned).toBe(true);
  });

  it('enforces outward triangle winding (signedVolume > 0, outwardDot > 0) on ALL procedural meshes', () => {
    const { mats, dummyMat } = makeDummyMats();

    const checkGroup = (prefix: string, grp: THREE.Group) => {
      grp.traverse((obj: THREE.Object3D) => {
        if (obj instanceof THREE.Mesh && obj.name !== 'KnobHitProxy') {
          assertOutwardWinding(`${prefix}/${obj.name}`, obj.geometry);
        }
      });
    };

    const oak = buildOakSpecimenBlock(mats);
    checkGroup('oak', oak.group);
    oak.dispose();

    const forend = buildForendAssembly(mats);
    checkGroup('forend', forend.group);
    forend.dispose();

    const lockCase = buildLockCaseAssembly(mats);
    checkGroup('case', lockCase.group);
    lockCase.dispose();

    const deadbolt = buildDeadboltAssembly(mats);
    checkGroup('bolt', deadbolt.group);
    deadbolt.dispose();

    const levers = buildLeverPackAssembly(mats);
    checkGroup('levers', levers.group);
    levers.dispose();

    const springs = buildSpringPackAssembly(mats);
    checkGroup('springsRest', springs.group);
    springs.update(new Float64Array(LEVER_COUNT).fill((17 * Math.PI) / 180));
    for (const sp of springs.springs) assertOutwardWinding(`springsFlex/${sp.mesh.name}`, sp.geometry);
    springs.dispose();

    const camBarrel = buildCamBarrelAssembly(mats);
    checkGroup('camBarrel', camBarrel.group);
    camBarrel.dispose();

    const thumbTurn = buildThumbTurnAssembly(mats);
    checkGroup('thumbTurn', thumbTurn.group);
    thumbTurn.dispose();

    const gearTrain = buildGearTrainAssembly(mats);
    const lock = new LockMechanism(true);
    gearTrain.update(
      lock.upperPinionAngleRad,
      lock.upperDetentAngleRad,
      lock.upperPawlAngleRad,
      lock.knobDriveB1AngleRad,
      lock.stepUpB2AngleRad,
      lock.camDrivenC1AngleRad,
      lock.transferB3AngleRad,
      lock.lowerDetentB4AngleRad,
      lock.lowerPawlAngleRad
    );
    checkGroup('gearTrainRest', gearTrain.group);

    const midLock = new LockMechanism(false);
    midLock.knobAngleRad = (KNOB_MIN_RAD + KNOB_MAX_RAD) * 0.5;
    midLock.boltThrow01 = 0.5;
    midLock.syncGearTrainStates();
    gearTrain.update(
      midLock.upperPinionAngleRad,
      midLock.upperDetentAngleRad,
      midLock.upperPawlAngleRad,
      midLock.knobDriveB1AngleRad,
      midLock.stepUpB2AngleRad,
      midLock.camDrivenC1AngleRad,
      midLock.transferB3AngleRad,
      midLock.lowerDetentB4AngleRad,
      midLock.lowerPawlAngleRad
    );
    assertOutwardWinding('gearTrainFlex/UpperDetentLeafSpring', gearTrain.upperDetentSpring.geometry);
    assertOutwardWinding('gearTrainFlex/LowerDetentLeafSpring', gearTrain.lowerDetentSpring.geometry);
    gearTrain.dispose();

    dummyMat.dispose();
  });

  it('verifies exact 3.000:1 coaxial gear ratio, pitch-circle tangency, and ZERO involute tooth overlap across ALL 6 meshes', () => {
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

    // 1. Exact pitch-circle tangency for every single gear pair & rack-to-pinion pair
    const gearPairs = [
      { name: 'U1B<->U2A', g1: GEAR_U1B_SPUR, g2: GEAR_U2A_DRIVEN },
      { name: 'B1<->B2A', g1: GEAR_B1_KNOB_DRIVE, g2: GEAR_B2A_PINION },
      { name: 'B2B<->C1', g1: GEAR_B2B_STEP, g2: GEAR_C1_CAM_DRIVEN },
      { name: 'B2B<->B3F', g1: GEAR_B2B_STEP, g2: GEAR_B3F_TRANSFER },
      { name: 'B3R<->B4A', g1: GEAR_B3R_PINION, g2: GEAR_B4A_DETENT },
    ];
    for (const pair of gearPairs) {
      const dist = Math.hypot(pair.g2.cx - pair.g1.cx, pair.g2.cy - pair.g1.cy);
      expect(Math.abs(dist - (pair.g1.pitchRadius + pair.g2.pitchRadius)), pair.name).toBeLessThan(1e-6);
    }
    expect(Math.abs(GEAR_U1A_PINION.cy - (UPPER_RACK_SPEC.pitchLineY + GEAR_U1A_PINION.pitchRadius))).toBeLessThan(1e-6);

    // 2. Exact angular velocity ratios including EXACT +3.000000:1 same-direction coaxial knob -> cam ratio!
    const u1_0 = computeUpperPinionAngleRad(0);
    const u1_1 = computeUpperPinionAngleRad(1);
    const u2_0 = computeUpperDetentWheelAngleRad(0);
    const u2_1 = computeUpperDetentWheelAngleRad(1);
    expect((u2_1 - u2_0) / (u1_1 - u1_0)).toBeCloseTo(-GEAR_U1B_SPUR.teeth / GEAR_U2A_DRIVEN.teeth, 6);

    const low0 = computeLowerTrainAnglesRad(KNOB_MIN_RAD);
    const low90 = computeLowerTrainAnglesRad(KNOB_MAX_RAD);
    expect((low90.b2 - low0.b2) / (low90.b1 - low0.b1)).toBeCloseTo(
      -GEAR_B1_KNOB_DRIVE.teeth / GEAR_B2A_PINION.teeth,
      6
    );
    expect((low90.c1 - low0.c1) / (low90.b1 - low0.b1)).toBeCloseTo(KNOB_TO_CAM_RATIO, 6); // EXACT +3.000000:1!
    expect(low90.c1 - low0.c1).toBeCloseTo(-(Math.PI * 1.5), 6); // 90 deg CW knob -> 270 deg CW cam barrel!
    expect((low90.b3 - low0.b3) / (low90.b2 - low0.b2)).toBeCloseTo(
      -GEAR_B2B_STEP.teeth / GEAR_B3F_TRANSFER.teeth,
      6
    );
    expect((low90.b4 - low0.b4) / (low90.b3 - low0.b3)).toBeCloseTo(
      -GEAR_B3R_PINION.teeth / GEAR_B4A_DETENT.teeth,
      6
    );

    // 3. Zero involute tooth overlap across full bolt throw [0, 1] for Rack<->U1A & U1B<->U2A
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

    // 4. Zero involute tooth overlap across full 0..90 deg throw (and extended 360 deg!) for B1<->B2A, B2B<->C1, B2B<->B3F, B3R<->B4A
    const b1Local = buildInvoluteGearPolygon2D(GEAR_B1_KNOB_DRIVE);
    const b2aLocal = buildInvoluteGearPolygon2D(GEAR_B2A_PINION);
    const b2bLocal = buildInvoluteGearPolygon2D(GEAR_B2B_STEP);
    const c1Local = buildInvoluteGearPolygon2D(GEAR_C1_CAM_DRIVEN);
    const b3fLocal = buildInvoluteGearPolygon2D(GEAR_B3F_TRANSFER);
    const b3rLocal = buildInvoluteGearPolygon2D(GEAR_B3R_PINION);
    const b4aLocal = buildInvoluteGearPolygon2D(GEAR_B4A_DETENT);

    for (let deg = 0; deg <= 90; deg += 5) {
      const knobRad = (deg * Math.PI) / 180;
      const angs = computeLowerTrainAnglesRad(knobRad);

      const b1W = transformPoly(b1Local, GEAR_B1_KNOB_DRIVE.cx, GEAR_B1_KNOB_DRIVE.cy, angs.b1);
      const b2aW = transformPoly(b2aLocal, GEAR_B2A_PINION.cx, GEAR_B2A_PINION.cy, angs.b2);
      const b2bW = transformPoly(b2bLocal, GEAR_B2B_STEP.cx, GEAR_B2B_STEP.cy, angs.b2);
      const c1W = transformPoly(c1Local, GEAR_C1_CAM_DRIVEN.cx, GEAR_C1_CAM_DRIVEN.cy, angs.c1);
      const b3fW = transformPoly(b3fLocal, GEAR_B3F_TRANSFER.cx, GEAR_B3F_TRANSFER.cy, angs.b3);
      const b3rW = transformPoly(b3rLocal, GEAR_B3R_PINION.cx, GEAR_B3R_PINION.cy, angs.b3);
      const b4aW = transformPoly(b4aLocal, GEAR_B4A_DETENT.cx, GEAR_B4A_DETENT.cy, angs.b4);

      assertNoPolygonEdgeCrossing(`B1<->B2A knob=${deg}deg`, b1W, b2aW);
      assertNoPolygonEdgeCrossing(`B2B<->C1 knob=${deg}deg`, b2bW, c1W);
      assertNoPolygonEdgeCrossing(`B2B<->B3F knob=${deg}deg`, b2bW, b3fW);
      assertNoPolygonEdgeCrossing(`B3R<->B4A knob=${deg}deg`, b3rW, b4aW);
    }
  });

  it('verifies dual-notch detent pawl seating at Unlocked (0) & Locked (1) and spatial cavity clearances', () => {
    const sol = {
      pawlWorldAngleRad: 0,
      beakWorldX: 0,
      beakWorldY: 0,
      contactWorldAngleRad: 0,
      camRadiusAtContact: 0,
      lift01: 0,
    };

    // 1. Upper Deadbolt Detent Pawl seats DEAD-FLUSH in V-notches at unlocked(0) & locked(1), climbs mid-throw!
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

    // 2. Lower Dual-Notch Detent Pawl seats DEAD-FLUSH at BOTH Vertical Unlocked (0) and Horizontal Locked (PI/2)!
    solveDetentPawlContact(LOWER_DETENT_CAM_SPEC, computeLowerTrainAnglesRad(KNOB_MIN_RAD).b4, sol);
    expect(sol.lift01).toBeLessThan(0.005);
    solveDetentPawlContact(LOWER_DETENT_CAM_SPEC, computeLowerTrainAnglesRad(KNOB_MAX_RAD).b4, sol);
    expect(sol.lift01).toBeLessThan(0.005);
    solveDetentPawlContact(
      LOWER_DETENT_CAM_SPEC,
      computeLowerTrainAnglesRad((KNOB_MIN_RAD + KNOB_MAX_RAD) * 0.5).b4,
      sol
    );
    expect(sol.lift01).toBeGreaterThan(0.85);

    // 3. Hard spatial clearances (zero forend collision, zero bolt-head collision, zero RestPin/cam collision!)
    expect(UPPER_RACK_SPEC.barMaxLocalX + BOLT_TRAVEL).toBeLessThan(CASE_INNER_MAX_X - 0.015);
    expect(UPPER_RACK_SPEC.zFront).toBeLessThan(getLeverZ(0) - LEVER_THICKNESS * 0.5 - 0.015);
    expect(GEAR_U1B_SPUR.zFront).toBeLessThan(getLeverZ(0) - LEVER_THICKNESS * 0.5 - 0.008);

    const boltHeadTopY = STUMP_Y + BOLT_HEAD_HALF_Y;
    const boltHeadBotY = STUMP_Y - BOLT_HEAD_HALF_Y;
    expect(GEAR_U1B_SPUR.cy - GEAR_U1B_SPUR.tipRadius).toBeGreaterThan(boltHeadTopY + 0.005);
    expect(GEAR_U2A_DRIVEN.cy - GEAR_U2A_DRIVEN.tipRadius).toBeGreaterThan(boltHeadTopY + 0.005);
    expect(GEAR_U2A_DRIVEN.cy + GEAR_U2A_DRIVEN.tipRadius).toBeLessThan(CUTAWAY_WIN_MAX_Y - 0.012);

    expect(GEAR_C1_CAM_DRIVEN.zBack).toBeGreaterThan(getLeverZ(LEVER_COUNT - 1) + LEVER_THICKNESS * 0.5 + 0.008);
    expect(GEAR_B1_KNOB_DRIVE.zBack).toBeGreaterThan(GEAR_C1_CAM_DRIVEN.zFront + 0.004);
    expect(GEAR_B4A_DETENT.cy + GEAR_B4A_DETENT.tipRadius).toBeLessThan(boltHeadBotY - 0.12);
    expect(GEAR_B4A_DETENT.cy - GEAR_B4A_DETENT.tipRadius).toBeGreaterThan(CUTAWAY_WIN_MIN_Y + 0.05);

    for (const g of [GEAR_B2B_STEP, GEAR_B3F_TRANSFER, GEAR_B3R_PINION, GEAR_B4A_DETENT]) {
      const dRest = Math.hypot(g.cx - REST_PIN_X, g.cy - REST_PIN_Y);
      expect(dRest, `${g.name} clears RestPin`).toBeGreaterThan(g.tipRadius + REST_PIN_RADIUS + 0.01);
    }

    const maxBitR = Math.max(...KEY_BIT_RADII);
    const dKeyB3 = Math.hypot(GEAR_B3R_PINION.cx - KEY_X, GEAR_B3R_PINION.cy - KEY_Y);
    expect(dKeyB3).toBeGreaterThan(GEAR_B3R_PINION.tipRadius + maxBitR + 0.02);

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
