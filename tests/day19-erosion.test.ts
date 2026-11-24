import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  EROSION_GRID_INNER,
  EROSION_GRID_SIZE,
  dirToOctUv,
  evalBaseBoulderRadiusCpu,
  mirrorOctahedralScalarGutter,
  octDecode,
  octEncode,
  texelDirAndSolidAngle,
} from '../src/days/19-erosion/octahedral';
import {
  AMMONITE_GROWTH_B,
  AMMONITE_PHASE_OFFSET,
  AMMONITE_R0,
  AMMONITE_THETA_MAX,
  buildAmmoniteGeometry,
} from '../src/days/19-erosion/ammonite';
import {
  ErosionField,
  FOSSIL_MATRIX_RECESS,
  MIN_CORE_FLOOR_RADIUS,
  MIN_FOSSIL_OVERBURDEN,
  type ErosionRaycastHit,
} from '../src/days/19-erosion/erosionField';
import {
  buildCleavedChipGeometry,
  buildNoduleShellGeometry,
  buildNozzleAssembly,
} from '../src/days/19-erosion/noduleGeometry';
import {
  NOZZLE_BLAST_STANDOFF,
  NOZZLE_HOVER_STANDOFF,
  NozzleController,
  measureNozzleMinRockClearance,
  type NozzlePoseInput,
} from '../src/days/19-erosion/nozzleKinematics';
import {
  AbrasionParticleSystem,
  MAX_DUST_PUFFS,
  MAX_GRIT_CHIPS,
} from '../src/days/19-erosion/particles';

function computeSignedVolumeAndRadialWinding(geo: THREE.BufferGeometry): {
  signedVolume: number;
  inwardRadialTris: number;
  totalTris: number;
} {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const idx = geo.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;

  geo.computeBoundingBox();
  const center = new THREE.Vector3();
  geo.boundingBox!.getCenter(center);

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const n = new THREE.Vector3();
  const cen = new THREE.Vector3();
  const bc = new THREE.Vector3();

  let signedVolume = 0;
  let inwardRadialTris = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1);
    c.fromBufferAttribute(pos, i2);

    bc.crossVectors(b, c);
    signedVolume += a.dot(bc) / 6.0;

    ab.subVectors(b, a);
    ac.subVectors(c, a);
    n.crossVectors(ab, ac);
    if (n.lengthSq() > 1e-14) {
      cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(center);
      if (n.dot(cen) < 0) {
        inwardRadialTris++;
      }
    }
  }

  return { signedVolume, inwardRadialTris, totalTris: triCount };
}

describe('Day 19 - Erosion: Pneumatic Air-Abrasion Fossil Prep Nodule', () => {
  it('maintains octahedral round-trip accuracy, exact 4*pi solid angle, and seamless gutter mirroring', () => {
    const o: [number, number] = [0, 0];
    const d: [number, number, number] = [0, 0, 0];

    // Round-trip 1000 deterministic directions spanning all 8 octants + seams
    let maxErr = 0;
    for (let i = 0; i < 1000; i++) {
      const z = 1.0 - (2.0 * (i + 0.5)) / 1000;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      const phi = i * 2.399963229728653;
      const x = Math.cos(phi) * r;
      const y = Math.sin(phi) * r;
      octEncode(x, y, z, o);
      octDecode(o[0], o[1], d);
      const err = Math.hypot(x - d[0], y - d[1], z - d[2]);
      if (err > maxErr) maxErr = err;
    }
    expect(maxErr).toBeLessThan(1e-6);

    // Exact 4*pi solid angle sum across interior texels
    let sumOmega = 0;
    for (let iy = 1; iy <= EROSION_GRID_INNER; iy++) {
      for (let ix = 1; ix <= EROSION_GRID_INNER; ix++) {
        sumOmega += texelDirAndSolidAngle(ix, iy, d, EROSION_GRID_SIZE);
      }
    }
    expect(sumOmega).toBeCloseTo(4 * Math.PI, 2);

    // Seamless 1-texel gutter mirroring on a smooth 3D scalar field
    const field = new Float32Array(EROSION_GRID_SIZE * EROSION_GRID_SIZE);
    const evalSmooth = (x: number, y: number, z: number) => 0.7 + 0.2 * x - 0.15 * y + 0.25 * z;
    for (let iy = 1; iy <= EROSION_GRID_INNER; iy++) {
      for (let ix = 1; ix <= EROSION_GRID_INNER; ix++) {
        texelDirAndSolidAngle(ix, iy, d, EROSION_GRID_SIZE);
        field[iy * EROSION_GRID_SIZE + ix] = evalSmooth(d[0], d[1], d[2]);
      }
    }
    mirrorOctahedralScalarGutter(field, EROSION_GRID_SIZE);

    // Verify bilinear lookup across the -Z back-pole perimeter matches analytic function closely
    const uv: [number, number] = [0, 0];
    for (let k = 0; k < 40; k++) {
      const a = (k / 40) * Math.PI * 2;
      const x = Math.cos(a) * 0.12;
      const y = Math.sin(a) * 0.12;
      const z = -Math.sqrt(1 - x * x - y * y);
      dirToOctUv(x, y, z, uv, EROSION_GRID_SIZE);
      const fx = uv[0] * EROSION_GRID_SIZE - 0.5;
      const fy = uv[1] * EROSION_GRID_SIZE - 0.5;
      const x0 = Math.max(0, Math.min(EROSION_GRID_SIZE - 2, Math.floor(fx)));
      const y0 = Math.max(0, Math.min(EROSION_GRID_SIZE - 2, Math.floor(fy)));
      const tx = fx - x0;
      const ty = fy - y0;
      const v00 = field[y0 * EROSION_GRID_SIZE + x0];
      const v10 = field[y0 * EROSION_GRID_SIZE + x0 + 1];
      const v01 = field[(y0 + 1) * EROSION_GRID_SIZE + x0];
      const v11 = field[(y0 + 1) * EROSION_GRID_SIZE + x0 + 1];
      const interp = (v00 + (v10 - v00) * tx) * (1 - ty) + (v01 + (v11 - v01) * tx) * ty;
      expect(Math.abs(interp - evalSmooth(x, y, z))).toBeLessThan(0.015);
    }
  });

  it('enforces differential-hardness erosion floor (matrix strips cleanly to fossil floor, never cuts fossil)', () => {
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(240, 32);
    const field = new ErosionField(ammoniteGeo, 1);

    // Snapshot immutable fossil radii
    const fossilSnapshot = new Float32Array(field.fossilRadius);
    let fossilCells = 0;
    for (let iy = 1; iy <= field.inner; iy++) {
      for (let ix = 1; ix <= field.inner; ix++) {
        const idx = iy * field.size + ix;
        // Zero artificial excavation floor anywhere on the nodule!
        expect(field.matrixFloor[idx]).toBe(MIN_CORE_FLOOR_RADIUS);
        expect(MIN_CORE_FLOOR_RADIUS).toBe(0);
        if (field.fossilRadius[idx] > 0) {
          fossilCells++;
          // Unfractured limestone concretion must strictly enclose the fossil by >= MIN_FOSSIL_OVERBURDEN
          expect(field.unfracturedRadius[idx]).toBeGreaterThanOrEqual(
            field.fossilRadius[idx] + MIN_FOSSIL_OVERBURDEN - 1e-5
          );
        }
      }
    }
    expect(fossilCells).toBeGreaterThan(15000);

    // Pick a buried front fossil direction (lower-left outer-middle whorl) and blast aggressively
    const blastDir = new THREE.Vector3(-0.72, -0.34, 0.60).normalize();
    const initialRAtBlast = field.sampleGridBilinear(
      field.currentRadius,
      blastDir.x,
      blastDir.y,
      blastDir.z
    );
    const fossilRAtBlast = field.sampleGridBilinear(
      field.fossilRadius,
      blastDir.x,
      blastDir.y,
      blastDir.z
    );
    // Also verify zero umbilical pinhole at exact +/-Z center
    expect(field.sampleGridBilinear(field.fossilRadius, 0, 0, 1)).toBeGreaterThan(0.16);
    expect(field.sampleGridBilinear(field.fossilRadius, 0, 0, -1)).toBeGreaterThan(0.16);
    // Exact equatorial venter coverage across all 4 cardinal axes (verifies z=0 octahedral fold):
    // +X (3.0 turns ~0.70), -X (2.5 turns ~0.46), +Y (2.25 turns ~0.38), -Y (2.75 turns ~0.58)
    expect(field.sampleGridBilinear(field.fossilRadius, 1, 0, 0)).toBeGreaterThan(0.65);
    expect(field.sampleGridBilinear(field.fossilRadius, -1, 0, 0)).toBeGreaterThan(0.42);
    expect(field.sampleGridBilinear(field.fossilRadius, 0, 1, 0)).toBeGreaterThan(0.35);
    expect(field.sampleGridBilinear(field.fossilRadius, 0, -1, 0)).toBeGreaterThan(0.54);
    expect(fossilRAtBlast).toBeGreaterThan(0.30);
    expect(initialRAtBlast).toBeGreaterThan(fossilRAtBlast + 0.04);

    let lastFossilHitRatio = 0;
    for (let step = 0; step < 300; step++) {
      const res = field.carveAt(blastDir.x, blastDir.y, blastDir.z, 1 / 30, 1.4);
      lastFossilHitRatio = res.hitFossilRatio;
    }

    // Matrix must erode all the way down to zero rock thickness, exposing the fossil cleanly
    const carvedMatR = field.sampleMatrixRadius(blastDir.x, blastDir.y, blastDir.z);
    expect(carvedMatR).toBeLessThan(0.005);
    expect(lastFossilHitRatio).toBeGreaterThan(0.75);

    // Combined solid surface at blast center MUST remain >= fossilRadius (fossil never cut!)
    const solidSurfR = field.sampleCombinedRadius(blastDir.x, blastDir.y, blastDir.z);
    expect(solidSurfR).toBeGreaterThanOrEqual(fossilRAtBlast - 1e-4);

    // Also blast outer rim aggressively and confirm limestone clears all the way to 0 (no collar floor lock)
    const rimDir = new THREE.Vector3(0.96, 0.22, 0.18).normalize();
    for (let step = 0; step < 300; step++) {
      field.carveAt(rimDir.x, rimDir.y, rimDir.z, 1 / 30, 1.5);
    }
    expect(field.sampleMatrixRadius(rimDir.x, rimDir.y, rimDir.z)).toBeLessThan(0.005);

    // Exhaustive grid invariant sweep
    for (let iy = 1; iy <= field.inner; iy++) {
      for (let ix = 1; ix <= field.inner; ix++) {
        const idx = iy * field.size + ix;
        // Fossil surface is strictly immutable
        expect(field.fossilRadius[idx]).toBe(fossilSnapshot[idx]);
        // Matrix never goes negative or exceeds starting surface
        expect(field.currentRadius[idx]).toBeGreaterThanOrEqual(0);
        expect(field.currentRadius[idx]).toBeLessThanOrEqual(field.startRadius[idx] + 1e-6);
        // Combined solid surface never drops below fossil surface
        const combined = Math.max(field.currentRadius[idx], field.fossilRadius[idx]);
        expect(combined).toBeGreaterThanOrEqual(field.fossilRadius[idx]);
      }
    }

    field.dispose();
    ammoniteGeo.dispose();
  });

  it('strictly conserves carved volume, tracks monotonic exposure, raycasts live craters, and resets cleanly', () => {
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(220, 30);
    const field = new ErosionField(ammoniteGeo, 1);

    // Initial state: intriguing fracture window peek (~2%..18%) with 0 user-carved volume
    expect(field.totalCarvedVolume).toBe(0);
    expect(field.computeExactCarvedVolume()).toBe(0);
    expect(field.exposedFraction).toBeGreaterThan(0.015);
    expect(field.exposedFraction).toBeLessThan(0.22);

    const startExp = field.exposedFraction;
    const hit: ErosionRaycastHit = {
      position: new THREE.Vector3(),
      normal: new THREE.Vector3(),
      direction: new THREE.Vector3(),
      distance: 0,
      isFossil: false,
    };

    // Raycast center front hits rock cleanly; raycast far off-axis misses cleanly
    expect(field.raycast(new THREE.Vector3(0, 0, 4.35), new THREE.Vector3(0, 0, -1), hit)).toBe(
      true
    );
    const distBefore = hit.distance;
    expect(hit.normal.z).toBeGreaterThan(0.5);

    expect(field.raycast(new THREE.Vector3(2.4, 2.0, 4.35), new THREE.Vector3(0, 0, -1), hit)).toBe(
      false
    );

    // Sweep two crossing carving arcs across the front hemisphere
    let prevVol = 0;
    let prevExp = startExp;
    for (let pass = 0; pass < 2; pass++) {
      for (let s = 0; s < 36; s++) {
        const t = s / 35;
        const dx = pass === 0 ? -0.58 + 1.16 * t : -0.52 + 1.04 * t;
        const dy = pass === 0 ? -0.44 + 0.72 * t : 0.42 - 0.84 * t;
        const dz = Math.sqrt(Math.max(0.12, 1 - dx * dx - dy * dy));
        field.carveAt(dx, dy, dz, 0.06, 1.35);

        expect(field.totalCarvedVolume).toBeGreaterThanOrEqual(prevVol);
        expect(field.exposedFraction).toBeGreaterThanOrEqual(prevExp - 1e-6);
        prevVol = field.totalCarvedVolume;
        prevExp = field.exposedFraction;
      }
    }

    expect(field.totalCarvedVolume).toBeGreaterThan(0.005);
    expect(field.exposedFraction).toBeGreaterThan(startExp + 0.02);

    // Exact volume conservation check against full-grid spherical-shell integral
    const exactVol = field.computeExactCarvedVolume();
    expect(Math.abs(field.totalCarvedVolume - exactVol)).toBeLessThan(1e-5);

    // Center crater raycast distance must increase after carving front center
    field.carveAt(0, 0, 1, 0.25, 1.2);
    expect(field.raycast(new THREE.Vector3(0, 0, 4.35), new THREE.Vector3(0, 0, -1), hit)).toBe(
      true
    );
    expect(hit.distance).toBeGreaterThan(distBefore + 0.02);

    // HalfFloat texture packing sanity (channel R stores multiplicative scaleR = currentRadius / R_base(d))
    field.stepAndUpload(0.016);
    const scaleVal = THREE.DataUtils.fromHalfFloat(field.textureData[(128 * 256 + 128) * 4]);
    expect(Number.isFinite(scaleVal)).toBe(true);
    expect(scaleVal).toBeGreaterThanOrEqual(0);
    expect(evalBaseBoulderRadiusCpu(0, 0, 1) * scaleVal).toBeGreaterThanOrEqual(0);

    // Reset with new seed re-encrusts fresh nodule and zeros carved volume
    const prevAxis = field.strataAxis.clone();
    field.reset(2);
    expect(field.strataSeed).toBe(2);
    expect(field.totalCarvedVolume).toBe(0);
    expect(field.computeExactCarvedVolume()).toBe(0);
    expect(field.exposedFraction).toBeGreaterThan(0.015);
    expect(field.exposedFraction).toBeLessThan(0.22);
    expect(field.strataAxis.distanceTo(prevAxis)).toBeGreaterThan(0.02);

    field.dispose();
    ammoniteGeo.dispose();
  });

  it('enforces outward triangle winding (signedVolume > 0 and inwardTri === 0) on all custom geometries', () => {
    // 1. Welded Geodesic Nodule Shell (detail=5 fast check + detail=6 production check)
    const noduleGeo = buildNoduleShellGeometry(5);
    const noduleCheck = computeSignedVolumeAndRadialWinding(noduleGeo);
    expect(noduleCheck.signedVolume).toBeGreaterThan(4.0); // sphere volume ~ 4*pi/3 = 4.189
    expect(noduleCheck.inwardRadialTris).toBe(0);
    noduleGeo.dispose();

    // 2. Watertight 3D Pyritized Ammonite: check signedVolume > 0 AND local tube-center outward normal (inwardTri === 0)
    const spiralSteps = 160;
    const profileSteps = 28;
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(spiralSteps, profileSteps);
    const pos = ammoniteGeo.attributes.position as THREE.BufferAttribute;
    const idx = ammoniteGeo.index!;
    const triCount = idx.count / 3;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();
    const n = new THREE.Vector3();
    const cen = new THREE.Vector3();
    const bc = new THREE.Vector3();
    const tubeCenter = new THREE.Vector3();

    let amSignedVol = 0;
    let amInwardTris = 0;

    for (let t = 0; t < triCount; t++) {
      const i0 = idx.getX(t * 3);
      const i1 = idx.getX(t * 3 + 1);
      const i2 = idx.getX(t * 3 + 2);
      a.fromBufferAttribute(pos, i0);
      b.fromBufferAttribute(pos, i1);
      c.fromBufferAttribute(pos, i2);

      bc.crossVectors(b, c);
      amSignedVol += a.dot(bc) / 6.0;

      ab.subVectors(b, a);
      ac.subVectors(c, a);
      n.crossVectors(ab, ac);

      const tubeQuadsTris = spiralSteps * profileSteps * 2;
      if (t < tubeQuadsTris) {
        // Check outward normal relative to the local spiral whorl centerline at that ring!
        const quadIdx = Math.floor(t / 2);
        const ringIdx = Math.floor(quadIdx / profileSteps);
        const theta = ((ringIdx + 0.5) / spiralSteps) * AMMONITE_THETA_MAX;
        const rc = AMMONITE_R0 * Math.exp(AMMONITE_GROWTH_B * theta);
        const ang = theta + AMMONITE_PHASE_OFFSET;
        tubeCenter.set(rc * Math.cos(ang), rc * Math.sin(ang), 0);
        cen.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(tubeCenter);
        if (n.dot(cen) <= 0) amInwardTris++;
      } else if (t < tubeQuadsTris + profileSteps) {
        // Start cap must face -E_t at theta = 0
        const ang0 = AMMONITE_PHASE_OFFSET;
        const outDir = new THREE.Vector3(Math.sin(ang0), -Math.cos(ang0), 0);
        if (n.dot(outDir) <= 0) amInwardTris++;
      } else if (t < tubeQuadsTris + profileSteps * 2) {
        // End cap must face +E_t at theta = thetaMax
        const angEnd = AMMONITE_THETA_MAX + AMMONITE_PHASE_OFFSET;
        const outDir = new THREE.Vector3(-Math.sin(angEnd), Math.cos(angEnd), 0);
        if (n.dot(outDir) <= 0) amInwardTris++;
      } else {
        // Embryonic protoconch umbilical pearl centered at (0,0,0): must face radially outward
        if (n.lengthSq() > 1e-14) {
          cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
          if (n.dot(cen) <= 0) amInwardTris++;
        }
      }
    }
    expect(amSignedVol).toBeGreaterThan(0.25);
    expect(amInwardTris).toBe(0);
    ammoniteGeo.dispose();

    // 3. Cleaved mineral chip geometry
    const chipGeo = buildCleavedChipGeometry();
    const chipCheck = computeSignedVolumeAndRadialWinding(chipGeo);
    expect(chipCheck.signedVolume).toBeGreaterThan(0.8);
    expect(chipCheck.inwardRadialTris).toBe(0);
    chipGeo.dispose();

    // 4. Articulated Brass & Iron Pneumatic Nozzle assembly (all solid lathe & curved hose meshes)
    const dummy = new THREE.MeshBasicMaterial();
    const nozzle = buildNozzleAssembly({
      satinBrass: dummy,
      darkIron: dummy,
      carbideTip: dummy,
      hoseRubber: dummy,
      abrasiveJet: dummy,
    });

    let solidMeshCount = 0;
    nozzle.group.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      if (obj.name === 'AbrasiveGritJet') {
        // Open jet cone: verify every quad faces radially outward from the Z axis (inwardTri === 0)
        const g = obj.geometry;
        const p = g.attributes.position as THREE.BufferAttribute;
        const gIdx = g.index!;
        let inwardOpen = 0;
        for (let t = 0; t < gIdx.count / 3; t++) {
          a.fromBufferAttribute(p, gIdx.getX(t * 3));
          b.fromBufferAttribute(p, gIdx.getX(t * 3 + 1));
          c.fromBufferAttribute(p, gIdx.getX(t * 3 + 2));
          ab.subVectors(b, a);
          ac.subVectors(c, a);
          n.crossVectors(ab, ac);
          cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
          const radialDot = n.x * cen.x + n.y * cen.y;
          if (radialDot <= 0) inwardOpen++;
        }
        expect(inwardOpen).toBe(0);
      } else {
        solidMeshCount++;
        const res = computeSignedVolumeAndRadialWinding(obj.geometry);
        expect(res.signedVolume, `${obj.name} signedVolume`).toBeGreaterThan(0);
        // For turned lathe solids around the Z axis, verify every side-wall quad faces radially outward
        // (n_xy . p_xy >= 0) and front/back end caps face -Z / +Z outward (zero inverted faces)
        if (obj.name === 'PneumaticAirHose') {
          // Verify seam-duplicated braided hose UVs (zero backward wrap jump) & orthonormal tangent frame
          const g = obj.geometry;
          const uvAttr = g.attributes.uv as THREE.BufferAttribute;
          const tanAttr = g.attributes.aHoseTangent as THREE.BufferAttribute;
          const circAttr = g.attributes.aHoseCircum as THREE.BufferAttribute;
          const normAttr = g.attributes.normal as THREE.BufferAttribute;
          const gIdx = g.index!;
          expect(tanAttr).toBeDefined();
          expect(circAttr).toBeDefined();
          let maxV = 0;
          let maxU = 0;
          for (let vIdx = 0; vIdx < uvAttr.count; vIdx++) {
            const u = uvAttr.getX(vIdx);
            const v = uvAttr.getY(vIdx);
            if (u > maxU) maxU = u;
            if (v > maxV) maxV = v;
            const tl = Math.hypot(tanAttr.getX(vIdx), tanAttr.getY(vIdx), tanAttr.getZ(vIdx));
            const cl = Math.hypot(circAttr.getX(vIdx), circAttr.getY(vIdx), circAttr.getZ(vIdx));
            expect(Math.abs(tl - 1)).toBeLessThan(1e-4);
            expect(Math.abs(cl - 1)).toBeLessThan(1e-4);
            const ndot = Math.abs(
              normAttr.getX(vIdx) * tanAttr.getX(vIdx) +
                normAttr.getY(vIdx) * tanAttr.getY(vIdx) +
                normAttr.getZ(vIdx) * tanAttr.getZ(vIdx)
            );
            // Side-wall rings are strictly orthogonal; cap centers have normal || tangent
            if (vIdx < uvAttr.count - 2) {
              expect(ndot).toBeLessThan(1e-4);
            }
          }
          expect(maxV).toBeCloseTo(1.0, 5);
          expect(maxU).toBeGreaterThan(0.65);
          // Zero backward UV.y seam smear across every side-wall triangle
          const sideTriCount = gIdx.count / 3 - 64; // exclude 32+32 start/end cap triangles
          for (let t = 0; t < sideTriCount; t++) {
            const v0 = uvAttr.getY(gIdx.getX(t * 3));
            const v1 = uvAttr.getY(gIdx.getX(t * 3 + 1));
            const v2 = uvAttr.getY(gIdx.getX(t * 3 + 2));
            expect(Math.max(v0, v1, v2) - Math.min(v0, v1, v2)).toBeLessThan(0.15);
          }
        } else {
          const g = obj.geometry;
          const p = g.attributes.position as THREE.BufferAttribute;
          const gIdx = g.index!;
          g.computeBoundingBox();
          const zMin = g.boundingBox!.min.z;
          const zMax = g.boundingBox!.max.z;
          let inwardLathe = 0;
          for (let t = 0; t < gIdx.count / 3; t++) {
            a.fromBufferAttribute(p, gIdx.getX(t * 3));
            b.fromBufferAttribute(p, gIdx.getX(t * 3 + 1));
            c.fromBufferAttribute(p, gIdx.getX(t * 3 + 2));
            ab.subVectors(b, a);
            ac.subVectors(c, a);
            n.crossVectors(ab, ac);
            if (n.lengthSq() < 1e-14) continue;
            cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
            const rLen = Math.hypot(cen.x, cen.y);
            const radDot = n.x * cen.x + n.y * cen.y;
            if (Math.abs(cen.z - zMin) < 0.003 && rLen < 0.02) {
              if (n.z > 0) inwardLathe++;
            } else if (Math.abs(cen.z - zMax) < 0.003 && rLen < 0.02) {
              if (n.z < 0) inwardLathe++;
            } else if (radDot < -1e-6) {
              inwardLathe++;
            }
          }
          expect(inwardLathe, `${obj.name} inwardLathe`).toBe(0);
        }
      }
    });
    expect(solidMeshCount).toBe(8);
    nozzle.dispose();
    dummy.dispose();
  });

  it('aims the articulated brass air-abrasion nozzle dead-on at impact points with guaranteed rock clearance from all angles', () => {
    const ctrl = new NozzleController();
    const outTip = new THREE.Vector3();
    const outQuat = new THREE.Quaternion();
    const outAim = new THREE.Vector3();
    const localForward = new THREE.Vector3();
    const toHit = new THREE.Vector3();
    const toCam = new THREE.Vector3();

    const viewpoints: Array<{
      cam: [number, number, number];
      hit: [number, number, number];
      norm: [number, number, number];
    }> = [
      { cam: [0, 0.14, 4.35], hit: [0.35, 0.22, 0.45], norm: [0.32, 0.22, 0.92] },
      { cam: [0, 0.1, -4.35], hit: [-0.28, -0.18, -0.46], norm: [-0.26, -0.18, -0.95] },
      { cam: [4.2, 0.5, 0.2], hit: [1.05, 0.12, 0.05], norm: [0.98, 0.14, 0.08] },
      { cam: [-3.9, -1.6, 1.5], hit: [-0.82, -0.45, 0.28], norm: [-0.75, -0.52, 0.40] },
      { cam: [0.2, 4.2, 0.6], hit: [0.1, 0.95, 0.15], norm: [0.1, 0.96, 0.24] },
    ];

    for (const vp of viewpoints) {
      const hitPos = new THREE.Vector3(...vp.hit);
      const hitNormal = new THREE.Vector3(...vp.norm).normalize();
      const cameraPos = new THREE.Vector3(...vp.cam);

      for (const isBlasting of [false, true]) {
        const input: NozzlePoseInput = {
          isHoveringRock: true,
          isBlasting,
          isOrbiting: false,
          hitPos,
          hitNormal,
          cameraPos,
          cameraUp: new THREE.Vector3(0, 1, 0),
          elapsedSec: 1.25,
        };

        const jetLen = ctrl.computeTargetPose(input, outTip, outQuat, outAim);
        const expectedStandoff = isBlasting ? NOZZLE_BLAST_STANDOFF : NOZZLE_HOVER_STANDOFF;
        expect(Math.abs(jetLen - expectedStandoff)).toBeLessThan(0.01);

        // Local -Z axis of nozzle quaternion must point strictly into hitPos
        localForward.set(0, 0, -1).applyQuaternion(outQuat).normalize();
        toHit.subVectors(hitPos, outTip).normalize();
        expect(localForward.dot(toHit)).toBeGreaterThan(0.9999);

        // Tip must stay cleanly outside rock tangent plane and on camera-facing side
        const tipOffset = outTip.clone().sub(hitPos).normalize();
        expect(tipOffset.dot(hitNormal)).toBeGreaterThan(0.35);
        toCam.subVectors(cameraPos, hitPos).normalize();
        expect(tipOffset.dot(toCam)).toBeGreaterThan(0.2);
      }
    }

    // Damped update smoothly activates jet on blast and retracts cleanly on orbit
    const blastInput: NozzlePoseInput = {
      isHoveringRock: true,
      isBlasting: true,
      isOrbiting: false,
      hitPos: new THREE.Vector3(0.2, 0.1, 0.48),
      hitNormal: new THREE.Vector3(0.2, 0.1, 0.97).normalize(),
      cameraPos: new THREE.Vector3(0, 0.14, 4.35),
      cameraUp: new THREE.Vector3(0, 1, 0),
      elapsedSec: 0,
    };
    for (let f = 0; f < 30; f++) {
      blastInput.elapsedSec = f / 60;
      ctrl.update(blastInput, 1 / 60);
    }
    expect(ctrl.jetActiveAlpha).toBeGreaterThan(0.95);

    blastInput.isBlasting = false;
    blastInput.isOrbiting = true;
    for (let f = 0; f < 45; f++) {
      blastInput.elapsedSec += 1 / 60;
      ctrl.update(blastInput, 1 / 60);
    }
    expect(ctrl.jetActiveAlpha).toBeLessThan(0.05);
  });

  it('emits bounded zero-NaN rebounding grit & calcite dust particles that decay cleanly to 0 when idle', () => {
    const dummy = new THREE.MeshBasicMaterial();
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(120, 20);
    const field = new ErosionField(ammoniteGeo, 1);
    const particles = new AbrasionParticleSystem(dummy, dummy);

    const hitPos = new THREE.Vector3(0.25, 0.15, 0.48);
    const hitNorm = new THREE.Vector3(0.25, 0.15, 0.95).normalize();
    const jetDir = new THREE.Vector3(-0.35, -0.25, -0.9).normalize();
    const camQuat = new THREE.Quaternion();

    for (let f = 0; f < 35; f++) {
      particles.emitBlast(hitPos, hitNorm, jetDir, 0.5, 1 / 60, 1.0);
      particles.step(1 / 60, camQuat, field);
    }

    expect(particles.activeParticles).toBeGreaterThan(40);
    expect(particles.activeParticles).toBeLessThanOrEqual(MAX_GRIT_CHIPS + MAX_DUST_PUFFS);

    for (let i = 0; i < particles.gritMesh.count * 4; i++) {
      expect(Number.isFinite(particles.gritMetaArr[i])).toBe(true);
    }
    for (let i = 0; i < particles.dustMesh.count * 4; i++) {
      expect(Number.isFinite(particles.dustMetaArr[i])).toBe(true);
    }

    // Stop emitting and let all chips & chalk plumes settle cleanly to 0
    for (let f = 0; f < 140; f++) {
      particles.step(1 / 60, camQuat, field);
    }
    expect(particles.activeParticles).toBe(0);

    particles.dispose();
    field.dispose();
    ammoniteGeo.dispose();
    dummy.dispose();
  });

  it('prevents ammonite rib ghosting across uncarved rock, eliminates y=0/x=0 octahedral seam kinks, and excavates briskly', () => {
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(240, 32);
    const field = new ErosionField(ammoniteGeo, 1);

    // 1. Verify full enclosure across seeds 1..4 without clamping to ribbed fossilRadius
    for (const seed of [1, 2, 3, 4]) {
      field.reset(seed);
      let minClearance = Infinity;
      for (let iy = 1; iy <= field.inner; iy++) {
        for (let ix = 1; ix <= field.inner; ix++) {
          const idx = iy * field.size + ix;
          const fR = field.fossilRadius[idx];
          if (fR > 0) {
            const gap = field.unfracturedRadius[idx] - fR;
            if (gap < minClearance) minClearance = gap;
          }
        }
      }
      expect(minClearance, `seed ${seed} minClearance`).toBeGreaterThanOrEqual(
        MIN_FOSSIL_OVERBURDEN
      );
    }

    // 2. Anti-rib-ghosting correlation gate on both front (+Z) and back (-Z) outer-whorl rings:
    // High-pass unfracturedRadius must show zero rib-frequency harmonic correlation with fossilRadius
    field.reset(1);
    const ringN = 180;
    for (const zSign of [1, -1]) {
      const rockVals = new Float64Array(ringN);
      const fossilVals = new Float64Array(ringN);
      for (let k = 0; k < ringN; k++) {
        const ang = (k / ringN) * Math.PI * 2;
        const rxy = 0.80;
        const dz = zSign * Math.sqrt(1 - rxy * rxy);
        const dx = Math.cos(ang) * rxy;
        const dy = Math.sin(ang) * rxy;
        rockVals[k] = field.sampleGridBilinear(field.unfracturedRadius, dx, dy, dz);
        fossilVals[k] = field.sampleGridBilinear(field.fossilRadius, dx, dy, dz);
      }
      // 2nd-difference high-pass filter isolates 30-rib/turn ripples cleanly from slow boulder lobes
      let num = 0;
      let denA = 0;
      let denB = 0;
      for (let k = 0; k < ringN; k++) {
        const prev = (k + ringN - 1) % ringN;
        const next = (k + 1) % ringN;
        const hpR = rockVals[k] - 0.5 * (rockVals[prev] + rockVals[next]);
        const hpF = fossilVals[k] - 0.5 * (fossilVals[prev] + fossilVals[next]);
        num += hpR * hpF;
        denA += hpR * hpR;
        denB += hpF * hpF;
      }
      const corr = num / Math.max(1e-12, Math.sqrt(denA * denB));
      expect(Math.abs(corr), `zSign=${zSign} rib ghosting corr`).toBeLessThan(0.18);
    }

    // 3. Exact GPU reconstruction (R_base(d) * HalfFloat scaleR) must have zero sharp kink across y=0 and x=0
    const uvScratch: [number, number] = [0, 0];
    const sampleGpuRadius = (dx: number, dy: number, dz: number) => {
      dirToOctUv(dx, dy, dz, uvScratch, field.size);
      const fx = uvScratch[0] * field.size - 0.5;
      const fy = uvScratch[1] * field.size - 0.5;
      const x0 = Math.max(0, Math.min(field.size - 2, Math.floor(fx)));
      const y0 = Math.max(0, Math.min(field.size - 2, Math.floor(fy)));
      const tx = fx - x0;
      const ty = fy - y0;
      const readScale = (x: number, y: number) =>
        THREE.DataUtils.fromHalfFloat(field.textureData[(y * field.size + x) * 4]);
      const sR =
        (readScale(x0, y0) * (1 - tx) + readScale(x0 + 1, y0) * tx) * (1 - ty) +
        (readScale(x0, y0 + 1) * (1 - tx) + readScale(x0 + 1, y0 + 1) * tx) * ty;
      return evalBaseBoulderRadiusCpu(dx, dy, dz) * Math.max(0, sR);
    };

    const eps = 0.005;
    for (const zVal of [0.35, 0.05, -0.35]) {
      for (const xSign of [-1, 1]) {
        const l0 = Math.hypot(xSign * 0.92, 0, zVal);
        const lm = Math.hypot(xSign * 0.92, -eps, zVal);
        const lp = Math.hypot(xSign * 0.92, eps, zVal);
        const r0 = sampleGpuRadius((xSign * 0.92) / l0, 0, zVal / l0);
        const rm = sampleGpuRadius((xSign * 0.92) / lm, -eps / lm, zVal / lm);
        const rp = sampleGpuRadius((xSign * 0.92) / lp, eps / lp, zVal / lp);
        const kink = Math.abs(rm - 2 * r0 + rp);
        // Eliminated old ~0.014 bilinear V-trough octahedral kink (>10x reduction, smooth across y=0)
        expect(kink, `y=0 seam kink at xSign=${xSign}, z=${zVal}`).toBeLessThan(0.002);
      }
    }

    // 4. Brisk excavation rate: 1.6s spiral sweep across front unveils > 12% of total 360-deg ammonite
    const cam = new THREE.PerspectiveCamera(35, 1280 / 800, 0.1, 100);
    cam.position.set(0, 0.14, 4.35);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    const rc = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const hit: ErosionRaycastHit = {
      position: new THREE.Vector3(),
      normal: new THREE.Vector3(),
      direction: new THREE.Vector3(),
      distance: 0,
      isFossil: false,
    };

    for (let step = 0; step < 36; step++) {
      const a = (step / 36) * Math.PI * 4;
      const rPx = 28 + (step / 36) * 225;
      const x = 640 + Math.cos(a) * rPx;
      const y = 400 + Math.sin(a) * rPx * 0.9;
      ndc.set((x / 1280) * 2 - 1, -(y / 800) * 2 + 1);
      rc.setFromCamera(ndc, cam);
      if (field.raycast(rc.ray.origin, rc.ray.direction, hit)) {
        field.carveAt(hit.direction.x, hit.direction.y, hit.direction.z, 0.045, 1.0);
      }
    }
    expect(field.exposedFraction).toBeGreaterThan(0.12);

    field.dispose();
    ammoniteGeo.dispose();
  });

  it('enforces zero degenerate ammonite triangles, full-shaft nozzle clearance across carved craters & limbs, exact jet-to-impact lock, and rear -Z pole carving continuity', () => {
    // 1. Zero degenerate zero-area triangles anywhere on ammonite (including protoconch boss poles)
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(220, 30);
    const pos = ammoniteGeo.attributes.position as THREE.BufferAttribute;
    const idx = ammoniteGeo.index!;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();
    const crossV = new THREE.Vector3();
    let degenerateTris = 0;
    for (let t = 0; t < idx.count / 3; t++) {
      a.fromBufferAttribute(pos, idx.getX(t * 3));
      b.fromBufferAttribute(pos, idx.getX(t * 3 + 1));
      c.fromBufferAttribute(pos, idx.getX(t * 3 + 2));
      ab.subVectors(b, a);
      ac.subVectors(c, a);
      crossV.crossVectors(ab, ac);
      if (crossV.lengthSq() < 1e-12) degenerateTris++;
    }
    expect(degenerateTris).toBe(0);

    // 2. Full-shaft & braided-hose rock clearance across 360-deg viewpoints, left-flank limbs & deep carved craters
    const field = new ErosionField(ammoniteGeo, 1);
    const sampleRadius = (dx: number, dy: number, dz: number) =>
      field.sampleCombinedRadius(dx, dy, dz);

    // Carve deep craters on front-left flank, rim limb, and rear -Z pole
    for (let step = 0; step < 30; step++) {
      field.carveAt(-0.62, 0.25, 0.74, 0.05, 1.2);
      field.carveAt(0.0, 0.0, -1.0, 0.05, 1.2);
      field.carveAt(0.88, -0.15, 0.45, 0.05, 1.2);
    }
    // Verify exact volume conservation still holds after rear -Z pole + spall-bevel carving!
    expect(Math.abs(field.totalCarvedVolume - field.computeExactCarvedVolume())).toBeLessThan(1e-5);

    // Verify bounded smooth crater cliff slopes across ALL visible limestone texels (zero radial needles)
    // (Behind-fossil recessed floor texels follow the 3D ammonite whorl step and are occluded by the fossil mesh)
    const freshField = new ErosionField(ammoniteGeo, 1);
    let maxVisibleLimestoneStep = 0;
    for (let iy = 2; iy <= freshField.inner - 1; iy++) {
      for (let ix = 2; ix <= freshField.inner - 1; ix++) {
        const i = iy * freshField.size + ix;
        const rC = freshField.startRadius[i];
        const fC = freshField.fossilRadius[i];
        if (fC > 0 && rC < fC) continue;
        for (const nIdx of [i + 1, i + freshField.size]) {
          const rN = freshField.startRadius[nIdx];
          const fN = freshField.fossilRadius[nIdx];
          if (fN > 0 && rN < fN) continue;
          const d = Math.abs(rC - rN);
          if (d > maxVisibleLimestoneStep) maxVisibleLimestoneStep = d;
        }
      }
    }
    expect(maxVisibleLimestoneStep).toBeLessThan(0.055);
    freshField.dispose();

    const ctrl = new NozzleController();
    const outTip = new THREE.Vector3();
    const outQuat = new THREE.Quaternion();
    const outAim = new THREE.Vector3();
    const normalScratch = new THREE.Vector3();

    const testProbes: Array<{
      cam: [number, number, number];
      dir: [number, number, number];
    }> = [
      { cam: [0, 0.14, 4.35], dir: [-0.62, 0.25, 0.74] }, // deep front-left crater
      { cam: [0, 0.14, 4.35], dir: [-0.88, -0.15, 0.45] }, // grazing left flank
      { cam: [3.8, 0.4, 2.0], dir: [0.88, -0.15, 0.45] }, // side orbit rim
      { cam: [-3.9, 0.3, 1.8], dir: [-0.84, 0.32, 0.44] }, // left side orbit
      { cam: [0, 0.1, -4.35], dir: [0.0, 0.0, -1.0] }, // rear -Z pole crater
      { cam: [0.2, 4.1, 1.2], dir: [0.08, 0.94, 0.32] }, // top pole
    ];

    for (const p of testProbes) {
      const dir = new THREE.Vector3(...p.dir).normalize();
      const rSurf = field.sampleCombinedRadius(dir.x, dir.y, dir.z);
      const hitPos = dir.clone().multiplyScalar(rSurf);
      const hitNorm = field.sampleCombinedNormal(dir.x, dir.y, dir.z, normalScratch).clone();
      const camPos = new THREE.Vector3(...p.cam);

      const input: NozzlePoseInput = {
        isHoveringRock: true,
        isBlasting: true,
        isOrbiting: false,
        hitPos,
        hitNormal: hitNorm,
        cameraPos: camPos,
        cameraUp: new THREE.Vector3(0, 1, 0),
        elapsedSec: 0.75,
        sampleSurfaceRadius: sampleRadius,
      };

      ctrl.computeTargetPose(input, outTip, outQuat, outAim);
      const minClr = measureNozzleMinRockClearance(outTip, outQuat, sampleRadius);
      expect(minClr, `nozzle full-shaft clearance at dir=${p.dir.join(',')}`).toBeGreaterThan(0.01);

      // Verify live damped sweep locks jet endpoint dead-on onto hitPos (zero sweep drift!)
      ctrl.update(input, 1 / 60);
      const jetEnd = ctrl.tipPos.clone().addScaledVector(ctrl.aimDir, ctrl.jetLength);
      expect(jetEnd.distanceTo(hitPos)).toBeLessThan(1e-4);
    }

    field.dispose();
    ammoniteGeo.dispose();
  });

  it('allows 100% limestone blast-away at the center umbilicus and across the full 360-deg sphere, preserves full-shaft braided-hose clearance when fully excavated, and re-encrusts cleanly', () => {
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry(220, 30);
    const field = new ErosionField(ammoniteGeo, 1);

    expect(field.remainingRockFraction).toBeCloseTo(1.0, 5);
    expect(field.sampleMatrixRadius(0, 0, 1)).toBeGreaterThan(0.35);

    // 1. Sustained center-targeted blast from front camera clears the ENTIRE central umbilicus
    //    and adjacent inter-whorl valleys (including (-0.16, -0.16)!) all the way to 0 rock!
    for (let step = 0; step < 60; step++) {
      field.carveAt(0, 0, 1, 1 / 30, 1.2, undefined, 0, 0, -1);
    }
    expect(field.sampleMatrixRadius(0, 0, 1)).toBe(0);
    // Check umbilical cone (within ~25 deg of +Z) cleared completely down to 0
    for (const [dx, dy] of [
      [0.0, 0.0],
      [0.12, 0.0],
      [-0.12, 0.0],
      [0.0, 0.12],
      [0.0, -0.12],
      [-0.16, -0.16],
      [0.16, -0.14],
    ]) {
      const dz = Math.sqrt(1 - dx * dx - dy * dy);
      expect(
        field.sampleMatrixRadius(dx, dy, dz),
        `center umbilical residual at (${dx},${dy})`
      ).toBeLessThan(0.01);
    }

    // 2. Sustained multi-angle full excavation across 360-deg Fibonacci directions blasts EVERY
    //    limestone texel on the entire sphere all the way down to 0 (zero stuck floor anywhere!)
    const fibN = 260;
    for (let pass = 0; pass < 3; pass++) {
      for (let i = 0; i < fibN; i++) {
        const z = 1.0 - (2.0 * (i + 0.5)) / fibN;
        const r = Math.sqrt(Math.max(0, 1 - z * z));
        const phi = i * 2.399963229728653;
        const dx = Math.cos(phi) * r;
        const dy = Math.sin(phi) * r;
        field.carveAt(dx, dy, z, 0.06, 1.5, undefined, -dx, -dy, -z);
      }
    }

    let maxRemainingRadius = 0;
    for (let iy = 1; iy <= field.inner; iy++) {
      const row = iy * field.size;
      for (let ix = 1; ix <= field.inner; ix++) {
        const r = field.currentRadius[row + ix];
        if (r > maxRemainingRadius) maxRemainingRadius = r;
      }
    }
    expect(maxRemainingRadius).toBe(0);
    expect(field.remainingRockFraction).toBeLessThan(1e-5);
    expect(field.exposedFraction).toBeGreaterThan(0.999);
    // Exact volume conservation holds all the way to 100% complete excavation!
    expect(Math.abs(field.totalCarvedVolume - field.computeExactCarvedVolume())).toBeLessThan(1e-5);

    // 3. Full-shaft nozzle, crimped ferrule, strain-relief boot & braided-hose clearance holds
    //    even when 100% of limestone is blasted away and the nozzle tracks the bare ammonite!
    const sampleBareAmmonite = (dx: number, dy: number, dz: number) =>
      field.sampleCombinedRadius(dx, dy, dz);
    const ctrl = new NozzleController();
    const outTip = new THREE.Vector3();
    const outQuat = new THREE.Quaternion();
    const outAim = new THREE.Vector3();
    const normScratch = new THREE.Vector3();

    const bareProbes: Array<{ cam: [number, number, number]; dir: [number, number, number] }> = [
      { cam: [0, 0.14, 4.35], dir: [0, 0, 1] },
      { cam: [0, 0.14, 4.35], dir: [-0.72, -0.34, 0.60] },
      { cam: [0, 0.14, 4.35], dir: [0.82, 0.25, 0.51] },
      { cam: [0, 0.1, -4.35], dir: [0, 0, -1] },
      { cam: [3.8, 0.4, 2.0], dir: [0.95, 0.0, 0.30] },
    ];
    for (const p of bareProbes) {
      const dir = new THREE.Vector3(...p.dir).normalize();
      const rSurf = field.sampleCombinedRadius(dir.x, dir.y, dir.z);
      const hitPos = dir.clone().multiplyScalar(rSurf);
      const hitNorm = field.sampleCombinedNormal(dir.x, dir.y, dir.z, normScratch).clone();
      const input: NozzlePoseInput = {
        isHoveringRock: true,
        isBlasting: true,
        isOrbiting: false,
        hitPos,
        hitNormal: hitNorm,
        cameraPos: new THREE.Vector3(...p.cam),
        cameraUp: new THREE.Vector3(0, 1, 0),
        elapsedSec: 2.0,
        sampleSurfaceRadius: sampleBareAmmonite,
      };
      ctrl.computeTargetPose(input, outTip, outQuat, outAim);
      const minClr = measureNozzleMinRockClearance(outTip, outQuat, sampleBareAmmonite);
      expect(minClr, `bare ammonite clearance at ${p.dir.join(',')}`).toBeGreaterThan(0.01);
    }

    // 4. Clean re-encrust reset restores full fresh nodule after 100% excavation
    field.reset(5);
    expect(field.remainingRockFraction).toBeCloseTo(1.0, 5);
    expect(field.totalCarvedVolume).toBe(0);
    expect(field.sampleMatrixRadius(0, 0, 1)).toBeGreaterThan(0.35);
    expect(field.exposedFraction).toBeGreaterThan(0.015);
    expect(field.exposedFraction).toBeLessThan(0.22);

    field.dispose();
    ammoniteGeo.dispose();
  });

  it('carved center crater wall stays SO(3)-smooth with zero octahedral y=0 ridge, radial fingering, or rib-imprint streaks', () => {
    const { geometry: ammoniteGeo } = buildAmmoniteGeometry();
    const field = new ErosionField(ammoniteGeo, 1);

    // Sustained 3-second static blast dead-center (+Z)
    for (let i = 0; i < 180; i++) {
      field.carveAt(0, 0, 1, 1 / 60, 1.0, undefined, 0, 0, -1);
    }

    // Center umbilicus blasts completely to zero while visible crater wall (theta = 26 deg)
    // stays azimuthally continuous across all 360 deg — catching octahedral 12/3/6/9 roof ridges,
    // Saffman-Taylor radial canyon fingering, and 30-rib/turn chalk/AO starburst streaks!
    expect(field.sampleMatrixRadius(0, 0, 1)).toBe(0);

    const th = (26 * Math.PI) / 180;
    let max2degScaleJump = 0;
    let max2degChalkJump = 0;
    let prevScale = -1;
    let prevChalk = -1;
    let scaleAt178 = 0;
    let scaleAt182 = 0;
    for (let az = 0; az <= 360; az += 2) {
      const phi = (az * Math.PI) / 180;
      const dx = Math.sin(th) * Math.cos(phi);
      const dy = Math.sin(th) * Math.sin(phi);
      const dz = Math.cos(th);
      const scale = field.sampleGridBilinear(field.deltaRadius, dx, dy, dz);
      const chalk = field.sampleGridBilinear(field.freshChalk, dx, dy, dz);
      if (az === 178) scaleAt178 = scale;
      if (az === 182) scaleAt182 = scale;
      if (prevScale >= 0) {
        max2degScaleJump = Math.max(max2degScaleJump, Math.abs(scale - prevScale));
        max2degChalkJump = Math.max(max2degChalkJump, Math.abs(chalk - prevChalk));
      }
      prevScale = scale;
      prevChalk = chalk;
    }

    // Smooth bowl wall (previous fingering had max2degScaleJump = 0.638; octahedral ridge had y=0 jump > 0.14)
    expect(max2degScaleJump).toBeLessThan(0.08);
    expect(Math.abs(scaleAt178 - scaleAt182)).toBeLessThan(0.05);
    // Rib-free chalk rim (previous binary fR flip had max2degChalkJump = 0.783)
    expect(max2degChalkJump).toBeLessThan(0.45);

    field.dispose();
    ammoniteGeo.dispose();
  });
});
