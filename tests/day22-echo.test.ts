import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  APERTURE_HALF,
  CAVITY_HALF,
  ECHO_BOUNCE_COUNT,
  GLASS_PANE_HALF,
  MAX_MIRROR_TILT_DEG,
  MAX_MIRROR_TILT_RAD,
  MIRROR_PRESETS,
  computeMirrorPlaneNormals,
  evalBounceTransmittanceRgb,
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
  buildEchoCubeAssembly,
  createKnurledDrumGeometry,
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
    expect(mats.knurledBrass.colorNode).toBeTruthy();
    expect(mats.bluedSteel.colorNode).toBeTruthy();
    mats.dispose();
  });
});
