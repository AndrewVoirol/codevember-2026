import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  ARCHITECTURAL_SLABS,
  AnamorphController,
  DEFAULT_VANTAGE_POS,
  DETENT_CAPTURE_DEG,
  INITIAL_CAMERA_POS,
  OCCLUSION_OTHER_BIAS,
  OCCLUSION_SELF_BIAS,
  PENROSE_PRIMARY_SEGS,
  PENROSE_SECONDARY_SEGS,
  SLAB_COUNT,
  STAGE_FOV_DEG,
  STENCIL_CONFIGS,
  STENCIL_COUNT,
  VANTAGE_TARGET,
  computeAlignmentMetrics,
  computeSightRingPose,
  computeVantageBasis,
  evaluateProjectedPaintCpu,
  evaluateStencilCpu,
  isPointVisibleFromVantage,
  projectWorldToStencilUv,
  sphericalToCartesian,
  type ArchitecturalSlab,
  type Vec3Tuple,
} from '../src/days/26-anamorph/anamorphMath';
import {
  buildArchitecturalCourtyardGeometry,
  buildBeveledSlabGeometry,
  buildSurveyorSightRingGeometry,
} from '../src/days/26-anamorph/architectureGeometry';
import { createAnamorphMaterials } from '../src/days/26-anamorph/materials';
import { MINERAL_PALETTE } from '../src/core/palette';

function computeSignedVolumeAndOutwardDot(geo: THREE.BufferGeometry): {
  triCount: number;
  signedVol: number;
  outwardDot: number;
} {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const idx = geo.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;

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
    expect(Number.isFinite(pos.getX(i))).toBe(true);
    expect(Number.isFinite(pos.getY(i))).toBe(true);
    expect(Number.isFinite(pos.getZ(i))).toBe(true);
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

  return { triCount, signedVol, outwardDot };
}

/**
 * Analytical first-hit raycast against all 30 architectural AABBs returning slab index, hit pos & face normal.
 */
function traceArchitecturalRay(
  origin: Vec3Tuple,
  dir: Vec3Tuple
): { slab: ArchitecturalSlab; t: number; pos: Vec3Tuple; normal: Vec3Tuple } | null {
  let bestT = Infinity;
  let bestSlab: ArchitecturalSlab | null = null;
  let bestNormal: Vec3Tuple = [0, 1, 0];

  for (let i = 0; i < ARCHITECTURAL_SLABS.length; i++) {
    const s = ARCHITECTURAL_SLABS[i];
    let tEnter = -Infinity;
    let tExit = Infinity;
    let enterAxis = 0;
    let enterSign = 1;

    for (let axis = 0; axis < 3; axis++) {
      const d = dir[axis];
      const o = origin[axis];
      if (Math.abs(d) < 1e-8) {
        if (o < s.min[axis] || o > s.max[axis]) {
          tEnter = Infinity;
          break;
        }
      } else {
        let t1 = (s.min[axis] - o) / d;
        let t2 = (s.max[axis] - o) / d;
        let sign = -1;
        if (t1 > t2) {
          const tmp = t1;
          t1 = t2;
          t2 = tmp;
          sign = 1;
        }
        if (t1 > tEnter) {
          tEnter = t1;
          enterAxis = axis;
          enterSign = sign;
        }
        if (t2 < tExit) tExit = t2;
      }
    }

    if (tExit > Math.max(tEnter, 0.01) + 0.006 && tEnter > 0.01 && tEnter < bestT) {
      bestT = tEnter;
      bestSlab = s;
      bestNormal = [
        enterAxis === 0 ? enterSign : 0,
        enterAxis === 1 ? enterSign : 0,
        enterAxis === 2 ? enterSign : 0,
      ];
    }
  }

  if (!bestSlab) return null;
  return {
    slab: bestSlab,
    t: bestT,
    pos: [
      origin[0] + dir[0] * bestT,
      origin[1] + dir[1] * bestT,
      origin[2] + dir[2] * bestT,
    ],
    normal: bestNormal,
  };
}

describe('Day 26 - Anamorph (Felice Varini Brutalist Courtyard Maquette)', () => {
  it('verifies outward triangle winding, exact bounds, watertight thickness, and zero overlaps on all architectural slabs and brass sight-ring', () => {
    expect(SLAB_COUNT).toBe(30);

    // 1. Check every single architectural slab individually
    for (const slab of ARCHITECTURAL_SLABS) {
      const w = slab.max[0] - slab.min[0];
      const h = slab.max[1] - slab.min[1];
      const d = slab.max[2] - slab.min[2];
      // Every slab must be significantly thicker than OCCLUSION_SELF_BIAS for watertight occlusion
      expect(Math.min(w, h, d), `${slab.id} min thickness`).toBeGreaterThan(
        OCCLUSION_SELF_BIAS * 4
      );
      expect(slab.bevel, `${slab.id} bevel`).toBeLessThan(OCCLUSION_SELF_BIAS * 0.5);

      const geo = buildBeveledSlabGeometry(slab);
      const { triCount, signedVol, outwardDot } = computeSignedVolumeAndOutwardDot(geo);
      expect(triCount, `${slab.id} triCount`).toBe(44);
      expect(signedVol, `${slab.id} signedVolume`).toBeGreaterThan(0);
      expect(outwardDot, `${slab.id} outwardDot`).toBeGreaterThan(0);

      // Exact bounding box parity with analytical occlusion AABB
      expect(geo.boundingBox!.min.x).toBeCloseTo(slab.min[0], 5);
      expect(geo.boundingBox!.min.y).toBeCloseTo(slab.min[1], 5);
      expect(geo.boundingBox!.min.z).toBeCloseTo(slab.min[2], 5);
      expect(geo.boundingBox!.max.x).toBeCloseTo(slab.max[0], 5);
      expect(geo.boundingBox!.max.y).toBeCloseTo(slab.max[1], 5);
      expect(geo.boundingBox!.max.z).toBeCloseTo(slab.max[2], 5);
      geo.dispose();
    }

    // 2. Zero unintentional volume overlaps across all 30*29/2 slab pairs
    for (let i = 0; i < ARCHITECTURAL_SLABS.length; i++) {
      const a = ARCHITECTURAL_SLABS[i];
      for (let j = i + 1; j < ARCHITECTURAL_SLABS.length; j++) {
        const b = ARCHITECTURAL_SLABS[j];
        const ox = Math.min(a.max[0], b.max[0]) - Math.max(a.min[0], b.min[0]);
        const oy = Math.min(a.max[1], b.max[1]) - Math.max(a.min[1], b.min[1]);
        const oz = Math.min(a.max[2], b.max[2]) - Math.max(a.min[2], b.min[2]);
        const overlapVol = ox > 0 && oy > 0 && oz > 0 ? ox * oy * oz : 0;
        expect(overlapVol, `overlap ${a.id} vs ${b.id}`).toBeLessThan(1e-6);
      }
    }

    // 3. Merged courtyard geometry integrity
    const merged = buildArchitecturalCourtyardGeometry();
    const mergedCheck = computeSignedVolumeAndOutwardDot(merged);
    expect(mergedCheck.triCount).toBe(SLAB_COUNT * 44);
    expect(mergedCheck.signedVol).toBeGreaterThan(0.5);
    const matAttr = merged.attributes.aMatClass as THREE.BufferAttribute;
    expect(matAttr).toBeDefined();
    expect(matAttr.count).toBe(merged.attributes.position.count);
    const slabIdAttr = merged.attributes.aSlabId as THREE.BufferAttribute;
    const slabLocalAttr = merged.attributes.aSlabLocal as THREE.BufferAttribute;
    const slabSizeAttr = merged.attributes.aSlabSize as THREE.BufferAttribute;
    const kindFlagsAttr = merged.attributes.aKindFlags as THREE.BufferAttribute;
    expect(slabIdAttr.count).toBe(merged.attributes.position.count);
    expect(slabLocalAttr.count).toBe(merged.attributes.position.count);
    expect(slabSizeAttr.count).toBe(merged.attributes.position.count);
    expect(kindFlagsAttr.count).toBe(merged.attributes.position.count);
    // Verify slabLocal stays in [0, 1] and stair treads are flagged cleanly
    let stairFlagCount = 0;
    for (let i = 0; i < slabLocalAttr.count; i++) {
      expect(slabLocalAttr.getX(i)).toBeGreaterThanOrEqual(-1e-5);
      expect(slabLocalAttr.getX(i)).toBeLessThanOrEqual(1 + 1e-5);
      if (kindFlagsAttr.getX(i) > 0.5) stairFlagCount++;
    }
    expect(stairFlagCount).toBe(6 * 44 * 3); // 6 cantilevered travertine treads
    merged.dispose();

    // 4. Machined brass surveyor's sight-ring (closed Lathe bezel + cardinal ticks) outward winding
    const { bezelGeometry, ticksGeometry } = buildSurveyorSightRingGeometry();
    const bezelCheck = computeSignedVolumeAndOutwardDot(bezelGeometry);
    expect(bezelCheck.triCount).toBeGreaterThan(100);
    expect(bezelCheck.signedVol, 'brass sight-ring bezel signedVolume').toBeGreaterThan(0);

    const ticksCheck = computeSignedVolumeAndOutwardDot(ticksGeometry);
    expect(ticksCheck.triCount).toBe(8 * 44);
    expect(ticksCheck.signedVol, 'brass sight-ring ticks signedVolume').toBeGreaterThan(0);
    expect(ticksCheck.outwardDot, 'brass sight-ring ticks outwardDot').toBeGreaterThan(0);
    bezelGeometry.dispose();
    ticksGeometry.dispose();

    // 5. All stencil pigments strictly belong to MINERAL_PALETTE
    for (const cfg of STENCIL_CONFIGS) {
      expect(MINERAL_PALETTE[cfg.primaryPigment]).toBeDefined();
      expect(MINERAL_PALETTE[cfg.secondaryPigment]).toBeDefined();
    }

    // 6. TSL material graph constructs & syncs all 4 stencils and vantage bases cleanly
    const mats = createAnamorphMaterials();
    const basis = computeVantageBasis(DEFAULT_VANTAGE_POS, VANTAGE_TARGET, STAGE_FOV_DEG);
    mats.syncVantageBasis(basis);
    for (let i = 0; i < STENCIL_COUNT; i++) {
      mats.syncStencil(i);
    }
    expect(mats.courtyardMaterial.colorNode).toBeTruthy();
    expect(mats.courtyardMaterial.roughnessNode).toBeTruthy();
    expect(mats.courtyardMaterial.normalNode).toBeTruthy();
    expect(mats.brassBezelMaterial.opacityNode).toBeTruthy();
    mats.dispose();
  });

  it('guarantees exact projective alignment against THREE.PerspectiveCamera across all aspect ratios and depth planes', () => {
    const basis = computeVantageBasis(DEFAULT_VANTAGE_POS, VANTAGE_TARGET, STAGE_FOV_DEG);
    const aspects = [16 / 9, 4 / 3, 1.0, 9 / 16];

    // Collect real visible painted surface hits spanning multiple depth planes from DEFAULT_VANTAGE_POS
    const paintedHits: Array<{ slabId: string; pos: Vec3Tuple; zCam: number; u: number; v: number }> =
      [];
    const hitSlabIds = new Set<string>();

    for (let gy = -24; gy <= 24; gy++) {
      for (let gx = -24; gx <= 24; gx++) {
        const su = (gx / 24) * 0.56;
        const sv = (gy / 24) * 0.56;
        // Ray through isotropic stencil coordinate (su, sv)
        const s = basis.tanHalfFov * basis.stencilScale;
        const rx = basis.forward[0] + basis.right[0] * su * s + basis.up[0] * sv * s;
        const ry = basis.forward[1] + basis.right[1] * su * s + basis.up[1] * sv * s;
        const rz = basis.forward[2] + basis.right[2] * su * s + basis.up[2] * sv * s;
        const rLen = Math.hypot(rx, ry, rz);
        const hit = traceArchitecturalRay(basis.pos, [rx / rLen, ry / rLen, rz / rLen]);
        if (!hit) continue;

        const paint = evaluateProjectedPaintCpu(hit.pos, hit.normal, basis, 0);
        if (paint.total > 0.5) {
          const uv = projectWorldToStencilUv(hit.pos, basis);
          paintedHits.push({
            slabId: hit.slab.id,
            pos: hit.pos,
            zCam: uv.zCam,
            u: uv.u,
            v: uv.v,
          });
          hitSlabIds.add(hit.slab.id);
        }
      }
    }

    // Must span >= 8 distinct architectural slabs & > 1.3 world units of depth along the sightline!
    expect(paintedHits.length).toBeGreaterThan(80);
    expect(hitSlabIds.size).toBeGreaterThanOrEqual(8);
    const zVals = paintedHits.map((h) => h.zCam);
    const depthSpan = Math.max(...zVals) - Math.min(...zVals);
    expect(depthSpan).toBeGreaterThan(1.3);

    // Verify exact pixel-level alignment against a real THREE.PerspectiveCamera at vantagePos
    const tmp = new THREE.Vector3();
    for (const aspect of aspects) {
      const cam = new THREE.PerspectiveCamera(STAGE_FOV_DEG, aspect, 0.1, 100);
      cam.position.set(basis.pos[0], basis.pos[1], basis.pos[2]);
      cam.lookAt(basis.target[0], basis.target[1], basis.target[2]);
      cam.updateMatrixWorld();
      cam.updateProjectionMatrix();

      for (const h of paintedHits) {
        tmp.set(h.pos[0], h.pos[1], h.pos[2]).project(cam);
        const screenIsoU = (tmp.x * aspect) / basis.stencilScale;
        const screenIsoV = tmp.y / basis.stencilScale;
        expect(screenIsoU).toBeCloseTo(h.u, 5);
        expect(screenIsoV).toBeCloseTo(h.v, 5);
      }
    }

    // Verify off-axis oblique INITIAL_CAMERA_POS dramatically fractures depth planes
    const camOblique = new THREE.PerspectiveCamera(STAGE_FOV_DEG, 16 / 9, 0.1, 100);
    camOblique.position.set(INITIAL_CAMERA_POS[0], INITIAL_CAMERA_POS[1], INITIAL_CAMERA_POS[2]);
    camOblique.lookAt(VANTAGE_TARGET[0], VANTAGE_TARGET[1], VANTAGE_TARGET[2]);
    camOblique.updateMatrixWorld();
    camOblique.updateProjectionMatrix();

    // Two collinear points on the same vantage ray separated by 1.4 units in depth
    const pNear: Vec3Tuple = [
      basis.pos[0] + basis.forward[0] * 3.9,
      basis.pos[1] + basis.forward[1] * 3.9,
      basis.pos[2] + basis.forward[2] * 3.9,
    ];
    const pFar: Vec3Tuple = [
      basis.pos[0] + basis.forward[0] * 5.3,
      basis.pos[1] + basis.forward[1] * 5.3,
      basis.pos[2] + basis.forward[2] * 5.3,
    ];
    const uvNear = projectWorldToStencilUv(pNear, basis);
    const uvFar = projectWorldToStencilUv(pFar, basis);
    expect(uvNear.u).toBeCloseTo(uvFar.u, 6);
    expect(uvNear.v).toBeCloseTo(uvFar.v, 6);

    const ndcNear = new THREE.Vector3(...pNear).project(camOblique);
    const ndcFar = new THREE.Vector3(...pFar).project(camOblique);
    const obliqueSeparation = Math.hypot(ndcNear.x - ndcFar.x, ndcNear.y - ndcFar.y);
    expect(obliqueSeparation).toBeGreaterThan(0.15);
  });

  it('rejects hidden occluded surfaces behind foreground pillars/walls and back-facing surfaces with zero paint', () => {
    const basis = computeVantageBasis(DEFAULT_VANTAGE_POS, VANTAGE_TARGET, STAGE_FOV_DEG);

    // 1. Sweep rays from DEFAULT_VANTAGE_POS across the entire stencil grid.
    //    Whenever a ray hits a foreground slab AND also intersects a second slab strictly behind it
    //    on a front-facing face (+X, +Y, or +Z), verify the shadowed rear point is rejected by occlusion
    //    and receives EXACTLY 0 paint across all 4 stencils — including rays right on painted bands!
    let verifiedOccludedCount = 0;
    let verifiedOccludedOnPaintedBand = 0;

    const faces: ReadonlyArray<{ axis: 0 | 1 | 2; normal: Vec3Tuple }> = [
      { axis: 0, normal: [1, 0, 0] },
      { axis: 1, normal: [0, 1, 0] },
      { axis: 2, normal: [0, 0, 1] },
    ];

    const s = basis.tanHalfFov * basis.stencilScale;
    for (let gy = -16; gy <= 16; gy++) {
      for (let gx = -16; gx <= 16; gx++) {
        const su = (gx / 16) * 0.52;
        const sv = (gy / 16) * 0.52;
        const rx = basis.forward[0] + basis.right[0] * su * s + basis.up[0] * sv * s;
        const ry = basis.forward[1] + basis.right[1] * su * s + basis.up[1] * sv * s;
        const rz = basis.forward[2] + basis.right[2] * su * s + basis.up[2] * sv * s;
        const rLen = Math.hypot(rx, ry, rz);
        const dir: Vec3Tuple = [rx / rLen, ry / rLen, rz / rLen];

        const firstHit = traceArchitecturalRay(basis.pos, dir);
        if (!firstHit) continue;

        const stencilAtRay = evaluateStencilCpu(su, sv, 0);

        for (const rear of ARCHITECTURAL_SLABS) {
          if (rear.id === firstHit.slab.id) continue;
          for (const f of faces) {
            // Only check faces that face toward the vantage point (dot(normal, -dir) > 0.1)
            const nDotToVantage = -(
              f.normal[0] * dir[0] +
              f.normal[1] * dir[1] +
              f.normal[2] * dir[2]
            );
            if (nDotToVantage <= 0.1) continue;
            const planeVal = rear.max[f.axis];
            if (Math.abs(dir[f.axis]) < 1e-4) continue;
            const t = (planeVal - basis.pos[f.axis]) / dir[f.axis];
            if (t <= firstHit.t + 0.08) continue;

            const hx = basis.pos[0] + dir[0] * t;
            const hy = basis.pos[1] + dir[1] * t;
            const hz = basis.pos[2] + dir[2] * t;
            const pt: Vec3Tuple = [hx, hy, hz];

            const o1 = (f.axis + 1) % 3;
            const o2 = (f.axis + 2) % 3;
            if (
              pt[o1] > rear.min[o1] + 0.02 &&
              pt[o1] < rear.max[o1] - 0.02 &&
              pt[o2] > rear.min[o2] + 0.02 &&
              pt[o2] < rear.max[o2] - 0.02
            ) {
              expect(
                isPointVisibleFromVantage(pt, f.normal, basis.pos),
                `shadowed point on ${rear.id} behind ${firstHit.slab.id}`
              ).toBe(false);
              for (let sIdx = 0; sIdx < STENCIL_COUNT; sIdx++) {
                expect(evaluateProjectedPaintCpu(pt, f.normal, basis, sIdx).total).toBe(0);
              }
              verifiedOccludedCount++;
              if (stencilAtRay.total > 0.8) {
                verifiedOccludedOnPaintedBand++;
              }
            }
          }
        }
      }
    }

    expect(verifiedOccludedCount).toBeGreaterThan(40);
    expect(verifiedOccludedOnPaintedBand).toBeGreaterThan(10);

    // 2. Every back-facing surface (dot(N, dirToVantage) <= 0) receives EXACTLY 0 paint
    for (const slab of ARCHITECTURAL_SLABS) {
      const cx = (slab.min[0] + slab.max[0]) * 0.5;
      const cy = (slab.min[1] + slab.max[1]) * 0.5;
      const cz = (slab.min[2] + slab.max[2]) * 0.5;
      // Back -Z face and bottom -Y face always face away from front-elevated DEFAULT_VANTAGE_POS
      const backPt: Vec3Tuple = [cx, cy, slab.min[2]];
      const bottomPt: Vec3Tuple = [cx, slab.min[1], cz];
      expect(isPointVisibleFromVantage(backPt, [0, 0, -1], basis.pos)).toBe(false);
      expect(evaluateProjectedPaintCpu(backPt, [0, 0, -1], basis, 0).total).toBe(0);
      expect(isPointVisibleFromVantage(bottomPt, [0, -1, 0], basis.pos)).toBe(false);
      expect(evaluateProjectedPaintCpu(bottomPt, [0, -1, 0], basis, 0).total).toBe(0);
    }

    // 3. Direct unoccluded front face of foreground portal-right-pier IS visible
    const portalRightIdx = ARCHITECTURAL_SLABS.findIndex((s) => s.id === 'portal-right-pier');
    const portalRight = ARCHITECTURAL_SLABS[portalRightIdx]!;
    const visibleFrontPt: Vec3Tuple = [
      (portalRight.min[0] + portalRight.max[0]) * 0.5,
      (portalRight.min[1] + portalRight.max[1]) * 0.5,
      portalRight.max[2],
    ];
    expect(
      isPointVisibleFromVantage(visibleFrontPt, [0, 0, 1], basis.pos, ARCHITECTURAL_SLABS, portalRightIdx)
    ).toBe(true);

    // 4. Per-slab exclusion eliminates the 1.8 cm contact-corner paint bleed:
    //    a point on plinth-deck just 0.008 units behind baffle-mid-wall (inside old 0.018 bias,
    //    outside new 0.002 OCCLUSION_OTHER_BIAS) MUST be blocked cleanly!
    const plinthDeckIdx = ARCHITECTURAL_SLABS.findIndex((s) => s.id === 'plinth-deck');
    const baffle = ARCHITECTURAL_SLABS.find((s) => s.id === 'baffle-mid-wall')!;
    // Ray from DEFAULT_VANTAGE_POS (+X, +Y, +Z) past baffle's -X face onto plinth-deck right behind it
    const cornerShadowPt: Vec3Tuple = [
      baffle.min[0] - 0.006,
      baffle.min[1], // top of plinth-deck (-0.46)
      (baffle.min[2] + baffle.max[2]) * 0.5,
    ];
    expect(OCCLUSION_OTHER_BIAS).toBeLessThan(0.004);
    expect(
      isPointVisibleFromVantage(cornerShadowPt, [0, 1, 0], basis.pos, ARCHITECTURAL_SLABS, plinthDeckIdx)
    ).toBe(false);

    // 5. Truncated L=8 Penrose Tribar segment topology integrity (9 primary + 6 secondary)
    expect(PENROSE_PRIMARY_SEGS.length).toBe(9);
    expect(PENROSE_SECONDARY_SEGS.length).toBe(6);
  });

  it('supports live re-projection from arbitrary new camera angles across all 4 stencils and preserves double-click rollback', () => {
    const ctrl = new AnamorphController();

    // Initial state starts oblique (~20-26 deg off-axis, not locked in, sight-ring visible)
    const bootState = ctrl.getState();
    expect(bootState.alignmentErrorDeg).toBeGreaterThan(18);
    expect(bootState.alignmentErrorDeg).toBeLessThan(28);
    expect(bootState.isLockedIn).toBe(false);
    expect(bootState.activeStencilIndex).toBe(0);
    expect(bootState.projectionCount).toBe(1);
    expect(ctrl.sightRingAlpha).toBeGreaterThan(0.95);

    // Re-project from 4 arbitrary viewpoints (left-front, rear-3/4, high-elevation, close-up)
    const testViewpoints: Vec3Tuple[] = [
      sphericalToCartesian(VANTAGE_TARGET, 4.6, -0.75, 0.28),
      sphericalToCartesian(VANTAGE_TARGET, 5.1, 2.45, 0.36),
      sphericalToCartesian(VANTAGE_TARGET, 4.9, -2.1, 0.62),
      sphericalToCartesian(VANTAGE_TARGET, 3.3, 0.15, 0.22),
    ];

    const tmp = new THREE.Vector3();
    for (let i = 0; i < testViewpoints.length; i++) {
      const vp = testViewpoints[i];
      ctrl.projectFromViewpoint(vp, (i + 1) * 1000);
      const st = ctrl.getState();
      expect(st.activeStencilIndex).toBe((i + 1) % STENCIL_COUNT);
      expect(st.projectionCount).toBe(i + 2);
      expect(st.alignmentErrorDeg).toBeLessThan(0.01);
      expect(st.isLockedIn).toBe(true);
      expect(ctrl.flashProgress).toBe(0);
      expect(ctrl.sightRingAlpha).toBe(0); // Ring cleanly dissolved at exact lock-in!

      // Confirm active stencil paints visible points from this arbitrary angle & matches PerspectiveCamera
      const cam = new THREE.PerspectiveCamera(STAGE_FOV_DEG, 16 / 9, 0.1, 100);
      cam.position.set(vp[0], vp[1], vp[2]);
      cam.lookAt(VANTAGE_TARGET[0], VANTAGE_TARGET[1], VANTAGE_TARGET[2]);
      cam.updateMatrixWorld();
      cam.updateProjectionMatrix();

      let paintedPrimary = 0;
      let paintedSecondary = 0;
      const basis = ctrl.vantageBasis;
      const s = basis.tanHalfFov * basis.stencilScale;

      for (let gy = -18; gy <= 18; gy++) {
        for (let gx = -18; gx <= 18; gx++) {
          const su = (gx / 18) * 0.55;
          const sv = (gy / 18) * 0.55;
          const rx = basis.forward[0] + basis.right[0] * su * s + basis.up[0] * sv * s;
          const ry = basis.forward[1] + basis.right[1] * su * s + basis.up[1] * sv * s;
          const rz = basis.forward[2] + basis.right[2] * su * s + basis.up[2] * sv * s;
          const rLen = Math.hypot(rx, ry, rz);
          const hit = traceArchitecturalRay(basis.pos, [rx / rLen, ry / rLen, rz / rLen]);
          if (!hit) continue;

          const ev = evaluateProjectedPaintCpu(hit.pos, hit.normal, basis, st.activeStencilIndex);
          if (ev.primary > 0.5) paintedPrimary++;
          if (ev.secondary > 0.5) paintedSecondary++;
          if (ev.total > 0.5) {
            const uv = projectWorldToStencilUv(hit.pos, basis);
            tmp.set(hit.pos[0], hit.pos[1], hit.pos[2]).project(cam);
            expect((tmp.x * (16 / 9)) / basis.stencilScale).toBeCloseTo(uv.u, 5);
            expect(tmp.y / basis.stencilScale).toBeCloseTo(uv.v, 5);
          }
        }
      }
      expect(paintedPrimary, `stencil ${st.activeStencilIndex} primary hits`).toBeGreaterThan(15);
      expect(paintedSecondary, `stencil ${st.activeStencilIndex} secondary hits`).toBeGreaterThan(5);

      // Sight-ring pose stays strictly along the ray from target toward vantagePos
      const ringPose = computeSightRingPose(basis);
      const ringDir = new THREE.Vector3(...ringPose.position)
        .sub(new THREE.Vector3(...VANTAGE_TARGET))
        .normalize();
      const vDir = new THREE.Vector3(...vp)
        .sub(new THREE.Vector3(...VANTAGE_TARGET))
        .normalize();
      expect(ringDir.dot(vDir)).toBeCloseTo(1.0, 5);
    }

    // Double-click rollback restores immediately prior projection within window, ignores outside window
    const beforeClick = ctrl.getState();
    const strayPos: Vec3Tuple = sphericalToCartesian(VANTAGE_TARGET, 4.7, 1.2, 0.3);
    ctrl.projectFromViewpoint(strayPos, 9000);
    expect(ctrl.getState().projectionCount).toBe(beforeClick.projectionCount + 1);
    expect(ctrl.rollbackRecentProjectionIfWithin(9200, 360)).toBe(true);
    expect(ctrl.getState().projectionCount).toBe(beforeClick.projectionCount);
    expect(ctrl.getState().activeStencilIndex).toBe(beforeClick.activeStencilIndex);

    // Outside window returns false without mutating
    ctrl.projectFromViewpoint(strayPos, 10000);
    expect(ctrl.rollbackRecentProjectionIfWithin(11000, 360)).toBe(false);
  });

  it('converges smoothly via lock-in glide (including antipodal) and magnetic detent without fighting active drags', () => {
    const ctrl = new AnamorphController();
    const out: Vec3Tuple = [0, 0, 0];

    // 1. Lock-in glide from INITIAL_CAMERA_POS converges smoothly to exact vantagePos
    ctrl.triggerLockInGlide(INITIAL_CAMERA_POS);
    let cur: Vec3Tuple = [...INITIAL_CAMERA_POS];
    for (let f = 0; f < 60; f++) {
      ctrl.step(1 / 60, cur, out);
      cur = [...out];
    }
    expect(ctrl.getState().alignmentErrorDeg).toBeLessThan(0.01);
    expect(ctrl.getState().isLockedIn).toBe(true);
    expect(ctrl.glideActive).toBe(false);

    // 2. Lock-in glide from near-antipodal rear angle arcs cleanly without passing through origin
    const antipodalPos: Vec3Tuple = [
      VANTAGE_TARGET[0] - (DEFAULT_VANTAGE_POS[0] - VANTAGE_TARGET[0]),
      VANTAGE_TARGET[1] - (DEFAULT_VANTAGE_POS[1] - VANTAGE_TARGET[1]),
      VANTAGE_TARGET[2] - (DEFAULT_VANTAGE_POS[2] - VANTAGE_TARGET[2]),
    ];
    ctrl.triggerLockInGlide(antipodalPos);
    cur = [...antipodalPos];
    let minRadius = Infinity;
    for (let f = 0; f < 60; f++) {
      ctrl.step(1 / 60, cur, out);
      cur = [...out];
      const r = Math.hypot(
        cur[0] - VANTAGE_TARGET[0],
        cur[1] - VANTAGE_TARGET[1],
        cur[2] - VANTAGE_TARGET[2]
      );
      minRadius = Math.min(minRadius, r);
    }
    expect(minRadius).toBeGreaterThan(4.0);
    expect(ctrl.getState().alignmentErrorDeg).toBeLessThan(0.01);

    // 3. Magnetic detent never fights while dragging, and cushions smoothly on release inside DETENT_CAPTURE_DEG
    const nearPos = sphericalToCartesian(
      VANTAGE_TARGET,
      4.92,
      0.50 + 0.035, // ~2.0 deg off-axis (< DETENT_CAPTURE_DEG)
      0.30
    );
    const mNear = computeAlignmentMetrics(nearPos, DEFAULT_VANTAGE_POS, VANTAGE_TARGET);
    expect(mNear.angularErrorDeg).toBeLessThan(DETENT_CAPTURE_DEG);
    expect(mNear.angularErrorDeg).toBeGreaterThan(1.5);

    // While dragging, step must NOT modify camera position
    ctrl.notifyDragStart();
    const modifiedWhileDragging = ctrl.step(1 / 60, nearPos, out);
    expect(modifiedWhileDragging).toBe(false);
    expect(out[0]).toBeCloseTo(nearPos[0], 6);

    // On drag release, magnetic detent smoothly cushions to exact alignment
    ctrl.notifyDragEnd();
    cur = [...nearPos];
    for (let f = 0; f < 90; f++) {
      ctrl.step(1 / 60, cur, out);
      cur = [...out];
    }
    expect(ctrl.getState().alignmentErrorDeg).toBeLessThan(0.05);
    expect(ctrl.getState().isLockedIn).toBe(true);

    // 3b. Sight-ring fades out smoothly when zoomed close inside the courtyard (camDist < 2.2)
    const closeZoomPos = sphericalToCartesian(VANTAGE_TARGET, 1.8, -0.45, 0.35);
    ctrl.setCameraPos(closeZoomPos[0], closeZoomPos[1], closeZoomPos[2]);
    expect(ctrl.sightRingAlpha).toBeLessThan(0.01);

    // 4. All 4 stencils produce clean non-overlapping primary & secondary masks in [0, 1]
    for (let sIdx = 0; sIdx < STENCIL_COUNT; sIdx++) {
      let maxPri = 0;
      let maxSec = 0;
      for (let y = -20; y <= 20; y++) {
        for (let x = -20; x <= 20; x++) {
          const ev = evaluateStencilCpu((x / 20) * 0.55, (y / 20) * 0.55, sIdx);
          expect(ev.primary).toBeGreaterThanOrEqual(0);
          expect(ev.primary).toBeLessThanOrEqual(1);
          expect(ev.secondary).toBeGreaterThanOrEqual(0);
          expect(ev.secondary).toBeLessThanOrEqual(1);
          expect(ev.total).toBeGreaterThanOrEqual(0);
          expect(ev.total).toBeLessThanOrEqual(1);
          maxPri = Math.max(maxPri, ev.primary);
          maxSec = Math.max(maxSec, ev.secondary);
        }
      }
      expect(maxPri).toBeGreaterThan(0.95);
      expect(maxSec).toBeGreaterThan(0.95);
    }
  });
});
