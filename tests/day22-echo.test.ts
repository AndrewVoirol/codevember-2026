import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  APERTURE_HALF,
  CAVITY_HALF,
  CUBE_OUTER_HALF,
  ECHO_BOUNCE_COUNT,
  GLASS_PANE_HALF,
  MAX_MIRROR_TILT_DEG,
  MAX_MIRROR_TILT_RAD,
  MIRROR_PRESETS,
  computeMirrorPlaneNormals,
  computeTiltedPlaneEdgeDistances,
  evalBounceTransmittanceRgb,
  projectScreenDragToMirrorTilt,
  traceCavityRayCpu,
} from '../src/days/22-echo/optics';
import {
  CORE_ANCHOR_Y,
  CORE_MAX_ANGLE_RAD,
  CORE_REST_Y,
  CORE_WIRE_LENGTH,
  THUMBSCREW_GEAR_RATIO,
  EchoSimulation,
} from '../src/days/22-echo/physics';
import {
  BEZEL_BACK_Z,
  BEZEL_FRONT_Z,
  BEZEL_OUTER_HALF,
  CORNER_BLOCK_HALF,
  GUSSET_BASE_PLANE,
  GUSSET_FRONT_PROTRUSION,
  LEAD_SCREW_MAX_Y,
  LEAD_SCREW_MIN_Y,
  MAX_LEAD_SCREW_TRAVEL,
  RAIL_HALF_W,
  RAIL_OFFSET,
  THUMBSCREW_MOUNT_OFFSET,
  VERNIER_TOP_Y,
  buildEchoCubeAssembly,
  createHelicalLeadScrewGeometry,
  createKnurledDrumGeometry,
  createSlottedLockCapGeometry,
} from '../src/days/22-echo/geometry';
import { createEchoMaterials } from '../src/days/22-echo/materials';

describe('Day 22 - Echo (Brass-Framed Two-Way Infinity Mirror Cube)', () => {
  it('traces exact 40-bounce parallel reflections and curves smoothly when mirror planes are tilted', () => {
    const origin = new THREE.Vector3(0.05, -0.04, GLASS_PANE_HALF);
    const dir = new THREE.Vector3(0.0, 0.0, -1.0);

    // 1. Parallel mirrors (0 deg): axial ray reflects strictly along ±Z for all 40 bounces
    const parallelSteps = traceCavityRayCpu(origin, dir, 0, 0, ECHO_BOUNCE_COUNT);
    expect(parallelSteps.length).toBe(ECHO_BOUNCE_COUNT);
    for (let i = 0; i < parallelSteps.length; i++) {
      const s = parallelSteps[i];
      expect(Math.abs(s.reflectedDir.x)).toBeLessThan(1e-6);
      expect(Math.abs(s.reflectedDir.y)).toBeLessThan(1e-6);
      expect(Math.abs(Math.abs(s.reflectedDir.z) - 1)).toBeLessThan(1e-6);
      expect(s.segmentLength).toBeGreaterThan(0.5);
    }

    // 2. Slightly tilted mirrors (Sweeping Arc 1.55 deg, -1.10 deg):
    // Opposing wedge mirrors rotate the reflected direction monotonically per bounce -> curved corridor
    const tx = (1.55 * Math.PI) / 180;
    const ty = (-1.1 * Math.PI) / 180;
    const curvedSteps = traceCavityRayCpu(origin, dir, tx, ty, ECHO_BOUNCE_COUNT);
    expect(curvedSteps.length).toBe(ECHO_BOUNCE_COUNT);

    // Measure lateral ray deflection (perpendicular to Z) across early bounces 0..5 before hitting side walls
    const earlyDeflections = curvedSteps
      .slice(0, 5)
      .map((s) => Math.hypot(s.reflectedDir.x, s.reflectedDir.y));
    for (let i = 1; i < earlyDeflections.length; i++) {
      expect(earlyDeflections[i]).toBeGreaterThan(earlyDeflections[i - 1]);
    }

    // Cumulative 3D hit points must diverge substantially from the parallel straight corridor
    const dist10 = curvedSteps[10].hitPoint.distanceTo(parallelSteps[10].hitPoint);
    const dist25 = curvedSteps[25].hitPoint.distanceTo(parallelSteps[25].hitPoint);
    expect(dist10).toBeGreaterThan(0.25);
    expect(dist25).toBeGreaterThan(0.25);

    // 3. Convex interior guarantee: every visible glass aperture corner sits strictly inside all 6 tilted planes
    const normals = Array.from({ length: 6 }, () => new THREE.Vector3());
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        computeMirrorPlaneNormals(sx * MAX_MIRROR_TILT_RAD, sy * MAX_MIRROR_TILT_RAD, normals);
        // Test all 8 extreme corners of the visible glass panes (±APERTURE_HALF on two axes, ±GLASS_PANE_HALF on face axis)
        for (const faceAxis of [0, 1, 2]) {
          for (const faceSign of [-1, 1]) {
            for (const uSign of [-1, 1]) {
              for (const vSign of [-1, 1]) {
                const coords = [0, 0, 0];
                coords[faceAxis] = faceSign * GLASS_PANE_HALF;
                coords[(faceAxis + 1) % 3] = uSign * APERTURE_HALF;
                coords[(faceAxis + 2) % 3] = vSign * APERTURE_HALF;
                const pt = new THREE.Vector3(coords[0], coords[1], coords[2]);
                for (let k = 0; k < 6; k++) {
                  const signedInterior = pt.dot(normals[k]) + CAVITY_HALF;
                  expect(signedInterior).toBeGreaterThan(0.004);
                }
              }
            }
          }
        }
      }
    }
  });

  it('enforces strictly monotonic Beer-Lambert soda-lime celadon/verdigris spectral attenuation across bounces 0..40', () => {
    const t0 = evalBounceTransmittanceRgb(0);
    expect(t0).toEqual([1, 1, 1]);

    let prev = t0;
    let prevGreenToRed = 1;
    let prevGreenToBlue = 1;

    for (let k = 1; k <= ECHO_BOUNCE_COUNT; k++) {
      const cur = evalBounceTransmittanceRgb(k);

      // Every spectral channel strictly decays monotonically per bounce and stays positive
      expect(cur[0]).toBeGreaterThan(0);
      expect(cur[1]).toBeGreaterThan(0);
      expect(cur[2]).toBeGreaterThan(0);
      expect(cur[0]).toBeLessThan(prev[0]);
      expect(cur[1]).toBeLessThan(prev[1]);
      expect(cur[2]).toBeLessThan(prev[2]);

      // Ferrous-iron soda-lime glass absorbs red > blue > green, so green dominance increases monotonically
      const gOverR = cur[1] / cur[0];
      const gOverB = cur[1] / cur[2];
      expect(gOverR).toBeGreaterThan(prevGreenToRed);
      expect(gOverB).toBeGreaterThan(prevGreenToBlue);

      prev = cur;
      prevGreenToRed = gOverR;
      prevGreenToBlue = gOverB;
    }

    // By bounce 40, green channel must strongly dominate red (> 5x ratio) while remaining visible (> 0.02)
    expect(prev[1] / prev[0]).toBeGreaterThan(5.0);
    expect(prev[1]).toBeGreaterThan(0.02);
    expect(prev[1]).toBeLessThan(0.15);
  });

  it('simulates damped 3D spherical core pendulum oscillation with exact wire length & monotonic energy decay', () => {
    const sim = new EchoSimulation(0);
    sim.pluckCore(0.92, -0.68);
    expect(sim.pluckFlash).toBe(1.0);

    const anchor = new THREE.Vector3(0, CORE_ANCHOR_Y, 0);
    const eStart = sim.computeCoreMechanicalEnergy();
    expect(eStart).toBeGreaterThan(0.35);

    let prevEnergy = eStart;
    let maxWireErr = 0;
    let maxAngle = 0;
    let zeroCrossingsX = 0;
    let prevX = sim.corePos.x;
    let eMid = 0;

    for (let f = 0; f < 600; f++) {
      sim.step(1 / 60);
      const wireLen = sim.corePos.distanceTo(anchor);
      maxWireErr = Math.max(maxWireErr, Math.abs(wireLen - CORE_WIRE_LENGTH));

      const downDot = (CORE_ANCHOR_Y - sim.corePos.y) / wireLen;
      const angle = Math.acos(Math.max(-1, Math.min(1, downDot)));
      if (angle > maxAngle) maxAngle = angle;

      if ((prevX < 0 && sim.corePos.x >= 0) || (prevX > 0 && sim.corePos.x <= 0)) {
        zeroCrossingsX++;
      }
      prevX = sim.corePos.x;

      const curEnergy = sim.computeCoreMechanicalEnergy();
      expect(Number.isFinite(curEnergy)).toBe(true);
      // Allow tiny symplectic discretization epsilon (~2e-3) while asserting genuine energy loss
      expect(curEnergy).toBeLessThanOrEqual(prevEnergy + 2.5e-3);
      prevEnergy = curEnergy;

      if (f === 300) eMid = curEnergy;
    }

    const eEnd = sim.computeCoreMechanicalEnergy();
    expect(maxWireErr).toBeLessThan(1e-4);
    expect(maxAngle).toBeLessThanOrEqual(CORE_MAX_ANGLE_RAD + 1e-4);
    expect(zeroCrossingsX).toBeGreaterThanOrEqual(6);
    expect(eMid).toBeLessThan(eStart * 0.45);
    expect(eEnd).toBeLessThan(eMid * 0.45);
    expect(sim.pluckFlash).toBeLessThan(0.01);

    // Eventually relaxes close to CORE_REST_Y
    for (let f = 0; f < 900; f++) sim.step(1 / 60);
    expect(Math.abs(sim.corePos.y - CORE_REST_Y)).toBeLessThan(0.005);
  });

  it('clamps micrometer mirror tilts, couples thumbscrew angles, and snaps/cycles canonical presets cleanly', () => {
    const sim = new EchoSimulation(0);
    // Over-drag well past max tilt -> clamped strictly to MAX_MIRROR_TILT_DEG
    sim.adjustMirrorTilt(0.5, -0.5);
    expect(sim.isCustomTilt).toBe(true);

    for (let f = 0; f < 180; f++) sim.step(1 / 60);
    const [txDeg, tyDeg] = sim.getTiltDegrees();
    expect(txDeg).toBeCloseTo(MAX_MIRROR_TILT_DEG, 1);
    expect(tyDeg).toBeCloseTo(-MAX_MIRROR_TILT_DEG, 1);
    expect(sim.screwAngleX).toBeCloseTo(sim.tiltXRad * THUMBSCREW_GEAR_RATIO, 5);
    expect(sim.screwAngleY).toBeCloseTo(sim.tiltYRad * THUMBSCREW_GEAR_RATIO, 5);

    // Double-clicking from custom dragged tilt snaps straight back to Parallel Hall (preset 1, 0° / 0°)
    const snappedIdx = sim.cyclePreset();
    expect(snappedIdx).toBe(1);
    expect(MIRROR_PRESETS[snappedIdx].tiltXDeg).toBe(0);
    expect(MIRROR_PRESETS[snappedIdx].tiltYDeg).toBe(0);

    for (let f = 0; f < 180; f++) sim.step(1 / 60);
    const [parX, parY] = sim.getTiltDegrees();
    expect(Math.abs(parX)).toBeLessThan(0.02);
    expect(Math.abs(parY)).toBeLessThan(0.02);

    // Subsequent double-clicks cycle through remaining canonical geometries
    expect(sim.cyclePreset()).toBe(2);
    expect(sim.cyclePreset()).toBe(3);
    expect(sim.cyclePreset()).toBe(0);
  });

  it('enforces outward triangle winding (signedVol > 0 and outwardDot > 0) across all machined brass & bronze geometry', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const assembly = buildEchoCubeAssembly({
      agedBronze: dummyMat,
      satinBrass: dummyMat,
      knurledBrass: dummyMat,
      bluedSteel: dummyMat,
      infinityMirror: dummyMat,
    });

    const assertOutward = (label: string, geo: THREE.BufferGeometry) => {
      const pos = geo.attributes.position as THREE.BufferAttribute;
      const idx = geo.index;
      const triCount = idx ? idx.count / 3 : pos.count / 3;
      expect(triCount, `${label} triCount`).toBeGreaterThan(0);

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

      expect(signedVol, `${label} signedVol`).toBeGreaterThan(0);
      expect(outwardDot, `${label} outwardDot`).toBeGreaterThan(0);
    };

    const checkedGeos = new Set<THREE.BufferGeometry>();
    let visibleMeshCount = 0;
    assembly.group.traverse((obj: any) => {
      if (!obj.isMesh || !obj.material.visible) return;
      visibleMeshCount++;
      if (!checkedGeos.has(obj.geometry)) {
        checkedGeos.add(obj.geometry);
        assertOutward(obj.name || obj.geometry.type, obj.geometry);
      }
    });

    expect(visibleMeshCount).toBeGreaterThanOrEqual(80);
    expect(assembly.thumbscrews.length).toBe(6);

    // Check 3D diamond-knurled drum attribute bounds
    const knurlGeo = createKnurledDrumGeometry();
    assertOutward('KnurledDrum', knurlGeo);
    const kvAttr = knurlGeo.attributes.aKnurlValley as THREE.BufferAttribute;
    expect(kvAttr).toBeDefined();
    let minKv = Infinity;
    let maxKv = -Infinity;
    for (let i = 0; i < kvAttr.count; i++) {
      const v = kvAttr.getX(i);
      if (v < minKv) minKv = v;
      if (v > maxKv) maxKv = v;
    }
    expect(minKv).toBeGreaterThanOrEqual(0);
    expect(maxKv).toBeLessThanOrEqual(1);
    expect(maxKv - minKv).toBeGreaterThan(0.5);

    knurlGeo.dispose();
    assembly.dispose();
    dummyMat.dispose();
  });

  it('instantiates all TSL materials and 40-bounce raytracer uniforms cleanly without runtime errors', () => {
    const mats = createEchoMaterials();
    expect(mats.uMirrorNormals.length).toBe(6);
    expect(mats.infinityMirror.colorNode).toBeTruthy();
    expect(mats.agedBronze.colorNode).toBeTruthy();
    expect(mats.satinBrass.colorNode).toBeTruthy();
    expect(mats.bezelBrass.colorNode).toBeTruthy();
    expect(mats.vernierBrass.colorNode).toBeTruthy();
    expect(mats.knurledBrass.colorNode).toBeTruthy();
    expect(mats.bluedSteel.colorNode).toBeTruthy();
    mats.dispose();
  });

  it('enforces flush corner-block / proud gusset seating, bezel rabbet overlap, and non-detaching helical lead screws', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const assembly = buildEchoCubeAssembly({
      agedBronze: dummyMat,
      satinBrass: dummyMat,
      knurledBrass: dummyMat,
      bluedSteel: dummyMat,
      infinityMirror: dummyMat,
    });
    assembly.group.updateMatrixWorld(true);

    // 1. Corner blocks sit strictly flush with CUBE_OUTER_HALF (zero protrusion into gussets)
    expect(RAIL_OFFSET + CORNER_BLOCK_HALF).toBeCloseTo(CUBE_OUTER_HALF, 6);
    // Rails stay subtly recessed (0.3 mm) inside corner blocks -> zero coplanar z-fighting
    expect(RAIL_OFFSET + RAIL_HALF_W).toBeLessThan(CUBE_OUTER_HALF - 0.0002);

    const tmpBox = new THREE.Box3();
    let cornerCount = 0;
    let gussetCount = 0;
    assembly.group.traverse((obj: any) => {
      if (!obj.isMesh || !obj.material.visible) return;
      if (obj.name.startsWith('BronzeCorner_')) {
        cornerCount++;
        tmpBox.setFromObject(obj);
        expect(Math.max(Math.abs(tmpBox.min.x), Math.abs(tmpBox.max.x))).toBeLessThanOrEqual(
          CUBE_OUTER_HALF + 1e-5
        );
        expect(Math.max(Math.abs(tmpBox.min.y), Math.abs(tmpBox.max.y))).toBeLessThanOrEqual(
          CUBE_OUTER_HALF + 1e-5
        );
        expect(Math.max(Math.abs(tmpBox.min.z), Math.abs(tmpBox.max.z))).toBeLessThanOrEqual(
          CUBE_OUTER_HALF + 1e-5
        );
      } else if (obj.name.startsWith('Gusset') && !obj.name.includes('_Bolt')) {
        gussetCount++;
        tmpBox.setFromObject(obj);
        const maxExtent = Math.max(
          Math.abs(tmpBox.min.x),
          Math.abs(tmpBox.max.x),
          Math.abs(tmpBox.min.y),
          Math.abs(tmpBox.max.y),
          Math.abs(tmpBox.min.z),
          Math.abs(tmpBox.max.z)
        );
        // Gusset front plateau stands cleanly proud of the 0.725 bronze face with zero coplanar overlap
        expect(maxExtent).toBeCloseTo(GUSSET_BASE_PLANE + GUSSET_FRONT_PROTRUSION, 4);
      }
    });
    expect(cornerCount).toBe(8);
    expect(gussetCount).toBe(24);

    // 2. Stepped brass bezels overlap watertight over the glass pane and embed inside bronze rail walls
    expect(BEZEL_BACK_Z).toBeLessThan(GLASS_PANE_HALF - 0.002);
    expect(BEZEL_FRONT_Z).toBeGreaterThan(GLASS_PANE_HALF + 0.02);
    expect(BEZEL_OUTER_HALF).toBeGreaterThan(RAIL_OFFSET - RAIL_HALF_W + 0.003);
    // Orthogonal bezel inner bevel never intersects adjacent bezel back plane
    expect(APERTURE_HALF + 0.014).toBeLessThanOrEqual(BEZEL_BACK_Z + 1e-6);

    // 3. Helical lead screw remains deeply engaged inside vernier collar across full ±MAX_LEAD_SCREW_TRAVEL
    //    and never pokes back through the outer bronze rail face into the optical cavity
    expect(LEAD_SCREW_MIN_Y + MAX_LEAD_SCREW_TRAVEL).toBeLessThan(VERNIER_TOP_Y - 0.015);
    expect(LEAD_SCREW_MIN_Y - MAX_LEAD_SCREW_TRAVEL).toBeGreaterThan(0.01);
    expect(THUMBSCREW_MOUNT_OFFSET + (LEAD_SCREW_MIN_Y - MAX_LEAD_SCREW_TRAVEL)).toBeGreaterThan(
      CUBE_OUTER_HALF + 0.005
    );
    expect(LEAD_SCREW_MAX_Y - MAX_LEAD_SCREW_TRAVEL).toBeGreaterThan(0.095);

    // Verify genuine helical asymmetry (radius varies along azimuth phi at a constant axial slice)
    const screwGeo = createHelicalLeadScrewGeometry();
    const sPos = screwGeo.attributes.position as THREE.BufferAttribute;
    let rMinMid = Infinity;
    let rMaxMid = -Infinity;
    const midY = (LEAD_SCREW_MIN_Y + LEAD_SCREW_MAX_Y) * 0.5;
    for (let i = 0; i < sPos.count; i++) {
      const y = sPos.getY(i);
      if (Math.abs(y - midY) < 0.002) {
        const r = Math.hypot(sPos.getX(i), sPos.getZ(i));
        if (r < rMinMid) rMinMid = r;
        if (r > rMaxMid) rMaxMid = r;
      }
    }
    expect(rMaxMid - rMinMid).toBeGreaterThan(0.0025);

    const lockCapGeo = createSlottedLockCapGeometry();
    expect(lockCapGeo.attributes.position.count).toBeGreaterThan(100);

    screwGeo.dispose();
    lockCapGeo.dispose();
    assembly.dispose();
    dummyMat.dispose();
  });

  it('maintains exact tilted-plane interior edge distances and bounce-0-to-bounce-1 optical parity at ±3.2°', () => {
    const normals = Array.from({ length: 6 }, () => new THREE.Vector3());
    computeMirrorPlaneNormals(MAX_MIRROR_TILT_RAD, -MAX_MIRROR_TILT_RAD, normals);

    // Solve exact intersection line point of two adjacent tilted mirror planes (+X wall [0] and +Z wall [4]) at y = 0:
    //   n0.x * x + n0.z * z = -CAVITY_HALF
    //   n4.x * x + n4.z * z = -CAVITY_HALF
    const det = normals[0].x * normals[4].z - normals[0].z * normals[4].x;
    expect(Math.abs(det)).toBeGreaterThan(0.5);
    const xEdge = (-CAVITY_HALF * normals[4].z + CAVITY_HALF * normals[0].z) / det;
    const zEdge = (-normals[0].x * CAVITY_HALF + normals[4].x * CAVITY_HALF) / det;
    const trueEdgePt = new THREE.Vector3(xEdge, 0, zEdge);

    // Axis-aligned heuristic (`CAVITY_HALF - midW`) suffers > 35 mm seam drift here, whereas exact
    // tilted-plane distance (`computeTiltedPlaneEdgeDistances`) locks strictly to 0.0!
    const exact = computeTiltedPlaneEdgeDistances(trueEdgePt, normals);
    expect(Math.abs(exact.wallDist)).toBeLessThan(1e-5);
    expect(Math.abs(exact.edgeDist)).toBeLessThan(1e-5);
    expect(exact.cornerDist).toBeGreaterThan(0.5);

    // Bounce-0 -> Bounce-1 continuity check at grazing incidence across extreme tilt (zero NaN rays)
    const grazingOrigins = [
      new THREE.Vector3(0.54, 0.52, GLASS_PANE_HALF),
      new THREE.Vector3(-0.54, -0.52, GLASS_PANE_HALF),
      new THREE.Vector3(GLASS_PANE_HALF, 0.48, -0.45),
    ];
    const grazingDirs = [
      new THREE.Vector3(-0.68, -0.52, -0.51).normalize(),
      new THREE.Vector3(0.71, 0.44, -0.55).normalize(),
      new THREE.Vector3(-0.35, -0.62, 0.7).normalize(),
    ];

    for (let idx = 0; idx < grazingOrigins.length; idx++) {
      const steps = traceCavityRayCpu(
        grazingOrigins[idx],
        grazingDirs[idx],
        MAX_MIRROR_TILT_RAD,
        -MAX_MIRROR_TILT_RAD,
        ECHO_BOUNCE_COUNT
      );
      expect(steps.length).toBe(ECHO_BOUNCE_COUNT);
      for (const s of steps) {
        expect(Number.isFinite(s.hitPoint.x)).toBe(true);
        expect(Number.isFinite(s.hitPoint.y)).toBe(true);
        expect(Number.isFinite(s.hitPoint.z)).toBe(true);
        expect(Math.abs(s.reflectedDir.length() - 1)).toBeLessThan(1e-5);
        // Every wall hit satisfies exact tilted-plane boundary `wallDist ≈ 0`
        const d = computeTiltedPlaneEdgeDistances(s.hitPoint, normals);
        expect(Math.abs(d.wallDist)).toBeLessThan(1e-4);
      }
      // Bounce 0 -> Bounce 1 reflection continuity: outgoing bounce-0 direction feeds bounce-1 seamlessly
      const b0 = steps[0];
      const b1 = steps[1];
      const expectedStepVec = b1.hitPoint.clone().sub(b0.hitPoint).normalize();
      expect(expectedStepVec.dot(b0.reflectedDir)).toBeGreaterThan(0.999);
    }
  });

  it('aligns camera-relative mirror-tilt drags continuously across front, rear, side, and top/bottom orbits', () => {
    const sens = 0.00026;
    const views: Array<{
      name: string;
      forward: THREE.Vector3;
      right: THREE.Vector3;
      up: THREE.Vector3;
      expectRightSignY: 1 | -1;
      expectUpSignX: 1 | -1;
    }> = [
      {
        name: 'front +Z',
        forward: new THREE.Vector3(0, 0, -1),
        right: new THREE.Vector3(1, 0, 0),
        up: new THREE.Vector3(0, 1, 0),
        expectRightSignY: 1,
        expectUpSignX: 1,
      },
      {
        name: 'rear -Z',
        forward: new THREE.Vector3(0, 0, 1),
        right: new THREE.Vector3(-1, 0, 0),
        up: new THREE.Vector3(0, 1, 0),
        expectRightSignY: -1,
        expectUpSignX: 1,
      },
      {
        name: 'side +X',
        forward: new THREE.Vector3(-1, 0, 0),
        right: new THREE.Vector3(0, 0, -1),
        up: new THREE.Vector3(0, 1, 0),
        expectRightSignY: 1,
        expectUpSignX: -1,
      },
      {
        name: 'side -X',
        forward: new THREE.Vector3(1, 0, 0),
        right: new THREE.Vector3(0, 0, 1),
        up: new THREE.Vector3(0, 1, 0),
        expectRightSignY: -1,
        expectUpSignX: -1,
      },
      {
        name: 'top +Y',
        forward: new THREE.Vector3(0, -1, 0),
        right: new THREE.Vector3(1, 0, 0),
        up: new THREE.Vector3(0, 0, -1),
        expectRightSignY: -1,
        expectUpSignX: 1,
      },
      {
        name: 'bottom -Y',
        forward: new THREE.Vector3(0, 1, 0),
        right: new THREE.Vector3(1, 0, 0),
        up: new THREE.Vector3(0, 0, 1),
        expectRightSignY: -1,
        expectUpSignX: -1,
      },
    ];

    for (const v of views) {
      // Rightward drag (dx = +100, dy = 0)
      const [dtxR, dtyR] = projectScreenDragToMirrorTilt(100, 0, v.right, v.up, v.forward, sens);
      expect(dtxR).toBeCloseTo(0, 8);
      expect(Math.sign(dtyR)).toBe(v.expectRightSignY);
      expect(Math.abs(dtyR)).toBeGreaterThan(0.015);

      // Upward drag (dx = 0, dy = -100)
      const [dtxU, dtyU] = projectScreenDragToMirrorTilt(0, -100, v.right, v.up, v.forward, sens);
      expect(dtyU).toBeCloseTo(0, 8);
      expect(Math.sign(dtxU)).toBe(v.expectUpSignX);
      expect(Math.abs(dtxU)).toBeGreaterThan(0.015);
    }

    // Continuous non-zero response across all 360° orbit azimuths (zero singularities)
    for (let deg = 0; deg < 360; deg += 15) {
      const th = (deg * Math.PI) / 180;
      const forward = new THREE.Vector3(-Math.sin(th), -0.25, -Math.cos(th)).normalize();
      const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
      const up = new THREE.Vector3().crossVectors(right, forward).normalize();
      const [dtx, dty] = projectScreenDragToMirrorTilt(60, -60, right, up, forward, sens);
      expect(Number.isFinite(dtx)).toBe(true);
      expect(Number.isFinite(dty)).toBe(true);
      expect(Math.hypot(dtx, dty)).toBeGreaterThan(0.004);
    }
  });
});
