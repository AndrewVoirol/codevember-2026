import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  APERTURE_PLANE_OFFSET,
  ARCH_HALF_WIDTH,
  ARCH_SILL_Y,
  ARCH_SPRING_Y,
  CUBE_BOUNDING_RADIUS,
  CUBE_HALF,
  DEFAULT_CAMERA_DISTANCE,
  INITIAL_ORBIT_AZIMUTH_DEG,
  INITIAL_ORBIT_ELEVATION_DEG,
  MAX_CAMERA_DISTANCE,
  MIN_CAMERA_DISTANCE,
  PORTAL_FACES,
  PORTAL_FACE_ORDER,
  PORTAL_WORLD_SCALE,
  ZOOM_CAMERA_DISTANCE,
  apertureUvToCubeLocal,
  clampCameraDistance,
  classifyPortalView,
  computeFaceZoomGoal,
  computeNextQuarterTurnAzimuth,
  evalPortalWorldSdfCpu,
  isPointInsideArchAperture,
  raymarchPortalWorldCpu,
  sphericalOrbitToPosition,
  transformCameraRayToPortalLocal,
  type PortalFaceId,
} from '../src/days/29-portal/portalWorlds';
import {
  buildPortalCubeAssembly,
  createChamferedBasaltBlockGeometry,
} from '../src/days/29-portal/geometry';
import { createPortalMaterials } from '../src/days/29-portal/materials';

describe('Day 29 - Portal (Four-World Non-Euclidean Architectural Cube)', () => {
  it('defines orthonormal right-handed cardinal bases and exact 3D camera-parallax ray transforms', () => {
    for (const faceId of PORTAL_FACE_ORDER) {
      const f = PORTAL_FACES[faceId];
      const n = new THREE.Vector3(...f.normal);
      const r = new THREE.Vector3(...f.right);
      const u = new THREE.Vector3(...f.up);
      const inward = new THREE.Vector3(...f.inward);

      expect(n.length()).toBeCloseTo(1, 6);
      expect(r.length()).toBeCloseTo(1, 6);
      expect(u.length()).toBeCloseTo(1, 6);
      expect(inward.dot(n)).toBeCloseTo(-1, 6);
      expect(r.dot(u)).toBeCloseTo(0, 6);
      expect(r.dot(n)).toBeCloseTo(0, 6);
      expect(u.dot(n)).toBeCloseTo(0, 6);

      // Right-handed frame: right x up == outward normal
      const rxu = r.clone().cross(u);
      expect(rxu.distanceTo(n)).toBeLessThan(1e-6);

      // Camera parallax invariance:
      // Place a virtual 3D landmark deep inside the portal world at (0, 0, depthM = 6.0 m).
      // Shifting the exterior camera laterally along +right shifts the aperture intersection
      // along +u and the transformed portal-local ray passes straight through (0, 0, 6.0 m)!
      const depthCube = 6.0 / PORTAL_WORLD_SCALE;
      const camOffsetNormal = APERTURE_PLANE_OFFSET + 2.4;
      const lateralShift = 0.72;
      const camPos = n
        .clone()
        .multiplyScalar(camOffsetNormal)
        .addScaledVector(r, lateralShift);

      // Exact cube-local point of deep landmark behind the aperture plane
      const landmarkCube = n.clone().multiplyScalar(APERTURE_PLANE_OFFSET - depthCube);
      const tPlane =
        (APERTURE_PLANE_OFFSET - camPos.dot(n)) / (landmarkCube.dot(n) - camPos.dot(n));
      const fragOnPlane = camPos.clone().lerp(landmarkCube, tPlane);
      const uHit = fragOnPlane.dot(r);
      expect(uHit).toBeGreaterThan(0.1); // Lateral camera shift moves image across aperture

      const localRay = transformCameraRayToPortalLocal(
        faceId,
        [fragOnPlane.x, fragOnPlane.y, fragOnPlane.z],
        [camPos.x, camPos.y, camPos.z]
      );

      // Entry point lies on z = 0 threshold, ray points strictly inward (rd.z > 0)
      expect(Math.abs(localRay.ro[2])).toBeLessThan(1e-5);
      expect(localRay.rd[2]).toBeGreaterThan(0.5);

      // Advancing ray to z = 6.0 m converges exactly onto (0, 0, 6.0)!
      const s = 6.0 / localRay.rd[2];
      const hitX = localRay.ro[0] + localRay.rd[0] * s;
      const hitY = localRay.ro[1] + localRay.rd[1] * s;
      expect(Math.abs(hitX)).toBeLessThan(1e-4);
      expect(Math.abs(hitY)).toBeLessThan(1e-4);
    }
  });

  it('evaluates finite, non-degenerate SDF intersections across all 4 cardinal worlds with clear portal thresholds', () => {
    // 1. Every point inside the arched portal aperture at z = 0 must be strictly open air (d > 0.01)
    //    so zero near-plane solid clipping occurs on any of the 4 worlds.
    const uSamples = [-0.38, -0.2, 0, 0.2, 0.38];
    const vSamples = [-0.46, -0.2, 0.08, 0.3, 0.45];

    for (const faceId of PORTAL_FACE_ORDER) {
      let apertureHits = 0;
      let testedRays = 0;

      const camGoal = computeFaceZoomGoal(faceId).position;

      for (const u of uSamples) {
        for (const v of vSamples) {
          if (!isPointInsideArchAperture(u, v, -0.02)) continue;
          const frag = apertureUvToCubeLocal(faceId, u, v);
          const ray = transformCameraRayToPortalLocal(faceId, frag, camGoal);

          const sdfAtThreshold = evalPortalWorldSdfCpu(
            faceId,
            ray.ro[0],
            ray.ro[1],
            ray.ro[2],
            0.5
          );
          expect(Number.isFinite(sdfAtThreshold), `${faceId} threshold SDF finite`).toBe(true);
          expect(
            sdfAtThreshold,
            `${faceId} threshold clear at u=${u}, v=${v}`
          ).toBeGreaterThan(0.01);

          const res = raymarchPortalWorldCpu(faceId, ray.ro, ray.rd, 0.5);
          expect(Number.isFinite(res.distance)).toBe(true);
          expect(res.distance).toBeGreaterThan(0.05);
          testedRays++;
          if (res.hit) apertureHits++;
        }
      }

      expect(testedRays).toBeGreaterThanOrEqual(15);
      // Every world must produce rich architectural intersections across its aperture
      expect(apertureHits, `${faceId} hit count`).toBeGreaterThanOrEqual(Math.floor(testedRays * 0.6));
    }

    // 2. Verify world-specific monumental non-Euclidean spatial signatures:
    // North (+Z) Colonnade: deep center aisle stretches > 12 m into sunlit distance
    // (floor ray intersects deep checkered travertine past 15 m; upper arch ray hits triumphal arch past 21 m)
    const northDeepFloor = raymarchPortalWorldCpu('north', [0, 0.1, 0], [0, -0.11, 0.9939]);
    expect(northDeepFloor.hit).toBe(true);
    expect(northDeepFloor.distance).toBeGreaterThan(15.0);
    expect(northDeepFloor.hitPos[1]).toBeCloseTo(-1.76, 1);
    const northTriumphalArch = raymarchPortalWorldCpu('north', [0.0, 0.95, 0], [0, 0.032, 0.9995]);
    expect(northTriumphalArch.hit).toBe(true);
    expect(northTriumphalArch.distance).toBeGreaterThan(21.5);
    // Flanking ray strikes fluted classical columns cleanly
    const northCol = raymarchPortalWorldCpu('north', [1.2, 0.0, 0], [0.22, 0.0, 0.975]);
    expect(northCol.hit).toBe(true);
    expect(northCol.distance).toBeLessThan(8.0);

    // East (+X) Plunging Lapis Spiral Stairwell:
    // Downward gaze hits lower cantilevered spiral treads well below cube floor (-1.73 m),
    // and steeper overhead gaze plunges deep down the open central oculus (< -6 m)!
    const eastPlunge = raymarchPortalWorldCpu(
      'east',
      [0, 0.35, 0],
      [0.05, -0.78, 0.6237]
    );
    expect(eastPlunge.hit).toBe(true);
    expect(eastPlunge.hitPos[1]).toBeLessThan(-3.5);

    const eastDeepAbyss = raymarchPortalWorldCpu(
      'east',
      [0, 0.9, 0],
      [0, -0.84, 0.5426]
    );
    expect(eastDeepAbyss.hit).toBe(true);
    expect(eastDeepAbyss.hitPos[1]).toBeLessThan(-5.5);

    // South (-Z) Amber Lantern Over Dark Still Water:
    // Lower ray meets calm dark cistern water surface (~y = -1.76) and center ray hits suspended lantern
    const southWater = raymarchPortalWorldCpu('south', [0, -0.75, 0], [0, -0.32, 0.947]);
    expect(southWater.hit).toBe(true);
    expect(southWater.hitPos[1]).toBeCloseTo(-1.76, 1);
    const southLantern = raymarchPortalWorldCpu('south', [0, 0.22, 0], [0, 0, 1], 0);
    expect(southLantern.hit).toBe(true);
    expect(southLantern.distance).toBeGreaterThan(2.9);
    expect(southLantern.distance).toBeLessThan(3.6);

    // West (-X) Warm Ochre Desert Slot-Canyon Arch:
    // Upper-center ray hits the monumental natural rock arch bridge around z ~ 5.5..7.2 m
    const westArch = raymarchPortalWorldCpu('west', [0.06, 0.65, 0], [0, 0.11, 0.994]);
    expect(westArch.hit).toBe(true);
    expect(westArch.hitPos[2]).toBeGreaterThan(5.0);
    expect(westArch.hitPos[2]).toBeLessThan(7.6);

    // 3. Dense West Canyon raymarch convergence guard: across a 21x21 zoomed aperture grid,
    //    zero rays may stall near solid geometry (d < 0.12) by exhausting the step budget early.
    const westGoal = computeFaceZoomGoal('west').position;
    let westExhaustedNearSurface = 0;
    let westSampled = 0;
    for (let iu = -10; iu <= 10; iu++) {
      for (let iv = -10; iv <= 10; iv++) {
        const u = (iu / 10) * (ARCH_HALF_WIDTH - 0.02);
        const v = ARCH_SILL_Y + 0.02 + ((iv + 10) / 20) * (ARCH_SPRING_Y + ARCH_HALF_WIDTH - ARCH_SILL_Y - 0.04);
        if (!isPointInsideArchAperture(u, v, -0.015)) continue;
        westSampled++;
        const frag = apertureUvToCubeLocal('west', u, v);
        const ray = transformCameraRayToPortalLocal('west', frag, westGoal);
        const res = raymarchPortalWorldCpu('west', ray.ro, ray.rd, 0, 96, 28);
        if (!res.hit && res.distance < 27.5 && res.minSdf < 0.08) {
          westExhaustedNearSurface++;
        }
      }
    }
    expect(westSampled).toBeGreaterThan(250);
    expect(westExhaustedNearSurface, 'West canyon step-exhaustion silhouette holes').toBe(0);
  });

  it('reports exactly 2 simultaneously visible distinct worlds from every 45° corner and 1 from each cardinal face', () => {
    const cornerCases: Array<{ az: number; expected: [PortalFaceId, PortalFaceId] }> = [
      { az: 45, expected: ['north', 'east'] },
      { az: 135, expected: ['east', 'south'] },
      { az: 225, expected: ['south', 'west'] },
      { az: 315, expected: ['west', 'north'] },
    ];

    for (const c of cornerCases) {
      const pos = sphericalOrbitToPosition(c.az, 16, DEFAULT_CAMERA_DISTANCE);
      const view = classifyPortalView([pos.x, pos.y, pos.z]);
      expect(view.visibleFaces.length, `corner ${c.az}° visible count`).toBe(2);
      expect(new Set(view.visibleFaces).size).toBe(2);
      expect(view.visibleFaces).toContain(c.expected[0]);
      expect(view.visibleFaces).toContain(c.expected[1]);
      expect(c.expected).toContain(view.dominantFace);
    }

    // Boot view (~42° NE corner) immediately showcases 2 impossible worlds side-by-side
    const bootPos = sphericalOrbitToPosition(
      INITIAL_ORBIT_AZIMUTH_DEG,
      INITIAL_ORBIT_ELEVATION_DEG,
      DEFAULT_CAMERA_DISTANCE
    );
    const bootView = classifyPortalView([bootPos.x, bootPos.y, bootPos.z]);
    expect(bootView.visibleFaces.length).toBe(2);
    expect(bootView.visibleFaces).toContain('north');
    expect(bootView.visibleFaces).toContain('east');

    // Pure cardinal viewpoints report exactly 1 visible face matching dominantFace
    for (const faceId of PORTAL_FACE_ORDER) {
      const spec = PORTAL_FACES[faceId];
      const pos = sphericalOrbitToPosition(spec.azimuthDeg, 10, DEFAULT_CAMERA_DISTANCE);
      const view = classifyPortalView([pos.x, pos.y, pos.z]);
      expect(view.dominantFace).toBe(faceId);
      expect(view.visibleFaces).toEqual([faceId]);
    }

    // Quarter-turn stepping preserves 45° corner track AND cardinal track cleanly
    expect(computeNextQuarterTurnAzimuth(42)).toBe(135);
    expect(computeNextQuarterTurnAzimuth(135)).toBe(225);
    expect(computeNextQuarterTurnAzimuth(225)).toBe(315);
    expect(computeNextQuarterTurnAzimuth(315)).toBe(45);
    expect(computeNextQuarterTurnAzimuth(0)).toBe(90);
    expect(computeNextQuarterTurnAzimuth(270)).toBe(0);
  });

  it('enforces safe face-zoom dolly bounds that never clip behind any portal aperture or cube face', () => {
    expect(MIN_CAMERA_DISTANCE).toBeGreaterThan(CUBE_BOUNDING_RADIUS + 0.25);
    expect(MIN_CAMERA_DISTANCE).toBeGreaterThan(APERTURE_PLANE_OFFSET + 0.9);
    expect(ZOOM_CAMERA_DISTANCE).toBeGreaterThanOrEqual(MIN_CAMERA_DISTANCE);
    expect(ZOOM_CAMERA_DISTANCE).toBeLessThan(DEFAULT_CAMERA_DISTANCE);
    expect(DEFAULT_CAMERA_DISTANCE).toBeLessThan(MAX_CAMERA_DISTANCE);

    expect(clampCameraDistance(0.2)).toBe(MIN_CAMERA_DISTANCE);
    expect(clampCameraDistance(-5.0)).toBe(MIN_CAMERA_DISTANCE);
    expect(clampCameraDistance(25.0)).toBe(MAX_CAMERA_DISTANCE);
    expect(clampCameraDistance(ZOOM_CAMERA_DISTANCE)).toBe(ZOOM_CAMERA_DISTANCE);

    for (const faceId of PORTAL_FACE_ORDER) {
      const f = PORTAL_FACES[faceId];
      const goal = computeFaceZoomGoal(faceId);

      expect(goal.distance).toBe(ZOOM_CAMERA_DISTANCE);
      // Camera normal projection must stay well in front of outer basalt wall AND recessed aperture
      const normalDist =
        goal.position[0] * f.normal[0] +
        goal.position[1] * f.normal[1] +
        goal.position[2] * f.normal[2];
      expect(normalDist, `${faceId} clears outer basalt face`).toBeGreaterThan(CUBE_HALF + 0.95);
      expect(normalDist, `${faceId} clears aperture plane`).toBeGreaterThan(
        APERTURE_PLANE_OFFSET + 1.05
      );

      const view = classifyPortalView(goal.position);
      expect(view.dominantFace).toBe(faceId);
      expect(view.visibleFaces).toContain(faceId);
    }
  });

  it('enforces strictly outward triangle winding across all basalt/brass solids and outward portal aperture planes', () => {
    const dummyMat = new THREE.MeshBasicMaterial();
    const assembly = buildPortalCubeAssembly({
      matteBasalt: dummyMat,
      basaltAccent: dummyMat,
      satinBrass: dummyMat,
      darkBrassFastener: dummyMat,
      portalMaterials: {
        north: dummyMat,
        east: dummyMat,
        south: dummyMat,
        west: dummyMat,
      },
    });

    const assertClosedSolidOutward = (label: string, geo: THREE.BufferGeometry) => {
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
        expect(Number.isFinite(a.x) && Number.isFinite(a.y) && Number.isFinite(a.z)).toBe(true);
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

    // Standalone chamfered block check
    const blockGeo = createChamferedBasaltBlockGeometry(0.4, 0.5, 0.4, 0.02);
    assertClosedSolidOutward('StandaloneChamferedBasaltBlock', blockGeo);
    blockGeo.dispose();

    // Every closed solid mesh in the architectural cube assembly must have signedVol > 0 & outwardDot > 0
    expect(assembly.solidMeshes.length).toBeGreaterThanOrEqual(6);
    for (const mesh of assembly.solidMeshes) {
      assertClosedSolidOutward(mesh.name, mesh.geometry);
    }

    // All 4 recessed arched portal aperture planes must wind strictly outward along face.normal
    // and seat flush at APERTURE_PLANE_OFFSET without intersecting adjacent portal planes
    for (const faceId of PORTAL_FACE_ORDER) {
      const mesh = assembly.portalMeshes[faceId];
      expect(mesh).toBeDefined();
      expect(mesh.userData.portalFace).toBe(faceId);
      const f = PORTAL_FACES[faceId];
      const expectedN = new THREE.Vector3(...f.normal);

      const pos = mesh.geometry.attributes.position as THREE.BufferAttribute;
      const triCount = pos.count / 3;
      expect(triCount).toBeGreaterThanOrEqual(32);

      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      const c = new THREE.Vector3();
      const ab = new THREE.Vector3();
      const ac = new THREE.Vector3();
      const triN = new THREE.Vector3();

      for (let t = 0; t < triCount; t++) {
        a.fromBufferAttribute(pos, t * 3);
        b.fromBufferAttribute(pos, t * 3 + 1);
        c.fromBufferAttribute(pos, t * 3 + 2);

        // Every vertex sits on the recessed aperture plane and stays clear of adjacent face planes
        for (const v of [a, b, c]) {
          expect(v.dot(expectedN)).toBeCloseTo(APERTURE_PLANE_OFFSET, 5);
          const uCoord = Math.abs(v.x * f.right[0] + v.y * f.right[1] + v.z * f.right[2]);
          expect(uCoord).toBeLessThan(APERTURE_PLANE_OFFSET - 0.15);
          expect(uCoord).toBeLessThanOrEqual(ARCH_HALF_WIDTH + 0.01);
          expect(v.y).toBeGreaterThanOrEqual(ARCH_SILL_Y - 0.01);
          expect(v.y).toBeLessThanOrEqual(ARCH_SPRING_Y + ARCH_HALF_WIDTH + 0.01);
        }

        ab.subVectors(b, a);
        ac.subVectors(c, a);
        triN.crossVectors(ab, ac).normalize();
        expect(triN.dot(expectedN), `${faceId} tri ${t} outward winding`).toBeGreaterThan(0.999);
      }
    }

    assembly.dispose();
    dummyMat.dispose();
  });

  it('instantiates all exterior PBR materials and 4 dedicated TSL portal raymarch shaders cleanly', () => {
    const mats = createPortalMaterials();
    expect(mats.matteBasalt.colorNode).toBeTruthy();
    expect(mats.basaltAccent.colorNode).toBeTruthy();
    expect(mats.satinBrass.colorNode).toBeTruthy();
    expect(mats.darkBrassFastener.colorNode).toBeTruthy();

    for (const faceId of PORTAL_FACE_ORDER) {
      expect(mats.portalMaterials[faceId].colorNode).toBeTruthy();
      expect(mats.uHoverWeights[faceId].value).toBe(0);
      expect(mats.uPointerLocal[faceId].value).toBeInstanceOf(THREE.Vector3);
    }

    mats.dispose();
  });

  it('keeps all 8 brass cube corners inside viewport across default orbits and frames all 4 zoomed arched thresholds cleanly', () => {
    const cam = new THREE.PerspectiveCamera(35, 16 / 9, 0.08, 100);
    const pt = new THREE.Vector3();

    // 1. Default suspended framing: all 8 outer brass corner brackets (±0.774) clear top & bottom borders
    const outerCorner = CUBE_HALF + 0.014;
    const testAzimuths = [INITIAL_ORBIT_AZIMUTH_DEG, 45, 135, 225, 315, 0, 90];
    const testElevations = [-45, 0, INITIAL_ORBIT_ELEVATION_DEG, 45, 65];

    for (const az of testAzimuths) {
      for (const el of testElevations) {
        sphericalOrbitToPosition(az, el, DEFAULT_CAMERA_DISTANCE, cam.position);
        cam.lookAt(0, 0, 0);
        cam.updateMatrixWorld();

        for (const sx of [-1, 1]) {
          for (const sy of [-1, 1]) {
            for (const sz of [-1, 1]) {
              pt.set(sx * outerCorner, sy * outerCorner, sz * outerCorner).project(cam);
              expect(
                Math.abs(pt.y),
                `default corner (${sx},${sy},${sz}) at az=${az} el=${el}`
              ).toBeLessThanOrEqual(0.92);
              expect(Math.abs(pt.x)).toBeLessThanOrEqual(0.92);
            }
          }
        }
      }
    }

    // 2. Zoomed cardinal views: full arched portal crown, sill, and springlines stay cleanly inside frame
    for (const faceId of PORTAL_FACE_ORDER) {
      const goal = computeFaceZoomGoal(faceId);
      cam.position.set(goal.position[0], goal.position[1], goal.position[2]);
      cam.lookAt(0, 0, 0);
      cam.updateMatrixWorld();

      const keyPoints: Array<[string, number, number]> = [
        ['crown', 0, ARCH_SPRING_Y + ARCH_HALF_WIDTH],
        ['sill', 0, ARCH_SILL_Y],
        ['leftSpring', -ARCH_HALF_WIDTH, ARCH_SPRING_Y],
        ['rightSpring', ARCH_HALF_WIDTH, ARCH_SPRING_Y],
      ];

      for (const [label, u, v] of keyPoints) {
        const cPos = apertureUvToCubeLocal(faceId, u, v);
        pt.set(cPos[0], cPos[1], cPos[2]).project(cam);
        expect(pt.y, `${faceId} zoomed ${label} top`).toBeLessThanOrEqual(0.92);
        expect(pt.y, `${faceId} zoomed ${label} bottom`).toBeGreaterThanOrEqual(-0.95);
        expect(Math.abs(pt.x), `${faceId} zoomed ${label} horiz`).toBeLessThanOrEqual(0.85);
      }
    }
  });
});
