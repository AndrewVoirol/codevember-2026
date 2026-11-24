import * as THREE from 'three/webgpu';
import {
  CURTAIN_PIVOT_Y,
  evalBranchCenterline,
  evalCurtainPoint,
  type BranchSample,
} from './shimmerGraph';

/**
 * Computes exact signed volume of an indexed (or non-indexed) triangle BufferGeometry.
 * Strictly positive (`> 0`) confirms outward-facing CCW triangle winding for closed solids.
 */
export function computeSignedVolume(geometry: THREE.BufferGeometry): number {
  const pos = geometry.getAttribute('position');
  if (!pos) return 0;
  const idx = geometry.getIndex();
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  let vol6 = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3;
    const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;

    const x0 = pos.getX(i0);
    const y0 = pos.getY(i0);
    const z0 = pos.getZ(i0);

    const x1 = pos.getX(i1);
    const y1 = pos.getY(i1);
    const z1 = pos.getZ(i1);

    const x2 = pos.getX(i2);
    const y2 = pos.getY(i2);
    const z2 = pos.getZ(i2);

    vol6 +=
      x0 * (y1 * z2 - z1 * y2) -
      y0 * (x1 * z2 - z1 * x2) +
      z0 * (x1 * y2 - y1 * x2);
  }
  return vol6 / 6;
}

/**
 * Helper: Merges multiple indexed BufferGeometries sharing identical attributes into one geometry.
 */
function mergeIndexedGeometries(geometries: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const attrNames = Object.keys(geometries[0].attributes);
  const itemSizes = new Map<string, number>();
  let totalVerts = 0;
  let totalIndices = 0;

  for (const name of attrNames) {
    itemSizes.set(name, geometries[0].getAttribute(name).itemSize);
  }
  for (const g of geometries) {
    totalVerts += g.getAttribute('position').count;
    const idx = g.getIndex();
    totalIndices += idx ? idx.count : g.getAttribute('position').count;
  }

  const merged = new THREE.BufferGeometry();
  const arrays = new Map<string, Float32Array>();
  for (const name of attrNames) {
    arrays.set(name, new Float32Array(totalVerts * itemSizes.get(name)!));
  }
  const indices = new Uint32Array(totalIndices);

  let vertCursor = 0;
  let idxCursor = 0;
  for (const g of geometries) {
    const vCount = g.getAttribute('position').count;
    for (const name of attrNames) {
      const src = g.getAttribute(name);
      const dst = arrays.get(name)!;
      const size = itemSizes.get(name)!;
      for (let i = 0; i < vCount * size; i++) {
        dst[vertCursor * size + i] = (src.array as Float32Array)[i];
      }
    }
    const idx = g.getIndex();
    if (idx) {
      for (let i = 0; i < idx.count; i++) {
        indices[idxCursor++] = idx.getX(i) + vertCursor;
      }
    } else {
      for (let i = 0; i < vCount; i++) {
        indices[idxCursor++] = i + vertCursor;
      }
    }
    vertCursor += vCount;
    g.dispose();
  }

  for (const name of attrNames) {
    merged.setAttribute(name, new THREE.BufferAttribute(arrays.get(name)!, itemSizes.get(name)!));
  }
  merged.setIndex(new THREE.BufferAttribute(indices, 1));
  merged.computeVertexNormals();
  return merged;
}

interface LimbSpec {
  rings: number;
  radialSegs: number;
  sample: (t: number, out: BranchSample) => void;
  startShear: number;
  endShear: number;
  furrowScale: number;
  seedPhase: number;
}

/**
 * Builds a single closed, watertight, outward-wound gnarled timber limb with
 * twisted bark ridges, burl swells, oblique broken end caps, and moss mask attribute.
 */
function buildTimberLimb(spec: LimbSpec): THREE.BufferGeometry {
  const { rings, radialSegs, sample, startShear, endShear, furrowScale, seedPhase } = spec;
  const cols = radialSegs + 1;
  const sideVertCount = (rings + 1) * cols;
  const capVertCount = 2 * (radialSegs + 2); // center + ring per cap
  const totalVerts = sideVertCount + capVertCount;

  const positions = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  // aBarkData: [furrowDepth01, burlCrest01, endCutMask01, mossMask01]
  const barkData = new Float32Array(totalVerts * 4);
  const indices: number[] = [];

  const centers: BranchSample[] = [];
  for (let i = 0; i <= rings; i++) {
    const s: BranchSample = { x: 0, y: 0, z: 0, r: 0.1 };
    sample(i / rings, s);
    centers.push(s);
  }

  // Parallel-transport frame along the branch spline (zero twist flips)
  const tangents: THREE.Vector3[] = [];
  const b1s: THREE.Vector3[] = [];
  const b2s: THREE.Vector3[] = [];

  for (let i = 0; i <= rings; i++) {
    const prev = centers[Math.max(0, i - 1)];
    const next = centers[Math.min(rings, i + 1)];
    const tang = new THREE.Vector3(next.x - prev.x, next.y - prev.y, next.z - prev.z).normalize();
    tangents.push(tang);

    if (i === 0) {
      const ref = Math.abs(tang.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
      const b1 = new THREE.Vector3().crossVectors(ref, tang).normalize();
      const b2 = new THREE.Vector3().crossVectors(tang, b1).normalize();
      // Verify right-handedness: b1 x b2 = tang
      b1s.push(b1);
      b2s.push(b2);
    } else {
      const prevB1 = b1s[i - 1];
      const b1 = prevB1.clone().addScaledVector(tang, -prevB1.dot(tang)).normalize();
      const b2 = new THREE.Vector3().crossVectors(tang, b1).normalize();
      b1s.push(b1);
      b2s.push(b2);
    }
  }

  const evalRingPoint = (
    i: number,
    j: number,
    outPos: [number, number, number],
    outData: [number, number, number, number]
  ) => {
    const t = i / rings;
    const c = centers[i];
    const ang = (j / radialSegs) * Math.PI * 2;
    const ca = Math.cos(ang);
    const sa = Math.sin(ang);
    const b1 = b1s[i];
    const b2 = b2s[i];
    const tang = tangents[i];

    // Outward radial direction
    const rx = ca * b1.x + sa * b2.x;
    const ry = ca * b1.y + sa * b2.y;
    const rz = ca * b1.z + sa * b2.z;

    // Helically twisted charred-oak / dark-walnut bark furrows + gnarly burl nodules
    const twistAng = ang + t * 2.1 + seedPhase;
    const ridgeA = Math.sin(twistAng * 7.0 + Math.sin(t * 14.0 + seedPhase) * 1.4);
    const ridgeB = Math.cos(twistAng * 13.0 - t * 22.0 + seedPhase * 1.7);
    const ridgeC = Math.sin(twistAng * 3.0 + t * 5.2 - 0.6);
    const furrow = 0.52 * ridgeA + 0.30 * ridgeB + 0.18 * ridgeC; // [-1..1]
    const burl = Math.max(0, ridgeC * 0.6 + Math.sin(t * 9.5 + twistAng * 2.0) * 0.4);

    // Taper bark displacement cleanly right at broken tips
    const endFade = Math.min(1, t * 14) * Math.min(1, (1 - t) * 14);
    const rFinal = c.r * (1.0 + furrow * 0.085 * furrowScale * (0.4 + 0.6 * endFade));

    // Oblique sheared wood fracture angle at the broken ends
    const shearStart = Math.max(0, 1 - t * 8) * startShear * (rx * 0.7 + ry * 0.5);
    const shearEnd = Math.max(0, (t - 0.875) * 8) * endShear * (rx * 0.6 - ry * 0.55);
    const axialShift = (shearStart + shearEnd) * c.r;

    outPos[0] = c.x + rx * rFinal + tang.x * axialShift;
    outPos[1] = c.y + ry * rFinal + tang.y * axialShift;
    outPos[2] = c.z + rz * rFinal + tang.z * axialShift;

    // Upper bark faces accumulate subtle woodland mossUmbrous lichen pockets
    const upFacing = Math.max(0, ry - 0.25) / 0.75;
    const mossPatch =
      upFacing *
      Math.max(0, Math.sin(t * 11.3 + seedPhase * 2.1) * Math.cos(ang * 2.3 - t * 6.5)) *
      (0.4 + 0.6 * Math.max(0, -furrow));

    outData[0] = furrow * 0.5 + 0.5;
    outData[1] = Math.min(1, burl);
    outData[2] = 0; // Bark exterior
    outData[3] = Math.min(1, mossPatch * 1.35);
  };

  const pTmp: [number, number, number] = [0, 0, 0];
  const dTmp: [number, number, number, number] = [0, 0, 0, 0];

  for (let i = 0; i <= rings; i++) {
    const t = i / rings;
    for (let j = 0; j <= radialSegs; j++) {
      const vIdx = i * cols + j;
      evalRingPoint(i, j % radialSegs, pTmp, dTmp);
      positions[vIdx * 3] = pTmp[0];
      positions[vIdx * 3 + 1] = pTmp[1];
      positions[vIdx * 3 + 2] = pTmp[2];
      uvs[vIdx * 2] = j / radialSegs;
      uvs[vIdx * 2 + 1] = t;
      barkData[vIdx * 4] = dTmp[0];
      barkData[vIdx * 4 + 1] = dTmp[1];
      barkData[vIdx * 4 + 2] = dTmp[2];
      barkData[vIdx * 4 + 3] = dTmp[3];
    }
  }

  // Side triangles: (b - a) along +ang, (c - a) along +tang -> (dP/da) x (dP/dt) = +radial outward!
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * cols + j;
      const b = a + 1;
      const c = (i + 1) * cols + j;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  // Watertight broken wood end caps (start i=0 facing -tang, end i=rings facing +tang)
  let capCursor = sideVertCount;
  const addCap = (ringIdx: number, isEnd: boolean) => {
    const c = centers[ringIdx];
    const centerIdx = capCursor++;
    positions[centerIdx * 3] = c.x;
    positions[centerIdx * 3 + 1] = c.y;
    positions[centerIdx * 3 + 2] = c.z;
    uvs[centerIdx * 2] = 0.5;
    uvs[centerIdx * 2 + 1] = 0.5;
    barkData[centerIdx * 4] = 0.5;
    barkData[centerIdx * 4 + 1] = 0.0;
    barkData[centerIdx * 4 + 2] = 1.0; // Heartwood cut end-grain
    barkData[centerIdx * 4 + 3] = 0.0;

    const ringStart = capCursor;
    for (let j = 0; j <= radialSegs; j++) {
      const vIdx = capCursor++;
      evalRingPoint(ringIdx, j % radialSegs, pTmp, dTmp);
      positions[vIdx * 3] = pTmp[0];
      positions[vIdx * 3 + 1] = pTmp[1];
      positions[vIdx * 3 + 2] = pTmp[2];
      const ang = (j / radialSegs) * Math.PI * 2;
      uvs[vIdx * 2] = 0.5 + 0.48 * Math.cos(ang);
      uvs[vIdx * 2 + 1] = 0.5 + 0.48 * Math.sin(ang);
      barkData[vIdx * 4] = dTmp[0];
      barkData[vIdx * 4 + 1] = 0.2;
      barkData[vIdx * 4 + 2] = 0.92; // Outer sapwood edge
      barkData[vIdx * 4 + 3] = 0.0;
    }

    for (let j = 0; j < radialSegs; j++) {
      const v0 = ringStart + j;
      const v1 = ringStart + j + 1;
      if (isEnd) {
        // Outward along +tang: (v0 - center) x (v1 - center) = +tang
        indices.push(centerIdx, v0, v1);
      } else {
        // Outward along -tang: (v1 - center) x (v0 - center) = -tang
        indices.push(centerIdx, v1, v0);
      }
    }
  };

  addCap(0, false);
  addCap(rings, true);

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aBarkData', new THREE.BufferAttribute(barkData, 4));
  geom.setIndex(indices);
  return geom;
}

/**
 * Builds the complete sculptural gnarled Dark-Walnut / Charred-Oak suspended branch
 * with two organic broken side twig forks and a lower knot stub.
 */
export function buildBranchGeometry(): THREE.BufferGeometry {
  // 1. Primary bough spanning x in [-1.92, +1.92] matching evalBranchCenterline()
  const mainBough = buildTimberLimb({
    rings: 88,
    radialSegs: 26,
    sample: (t, out) => {
      const x = -1.92 + t * 3.84;
      evalBranchCenterline(x, out);
    },
    startShear: 0.65,
    endShear: 0.72,
    furrowScale: 1.0,
    seedPhase: 0.4,
  });

  // 2. Left-upper gnarled broken fork emerging from x = -1.26
  const leftRoot = evalBranchCenterline(-1.26);
  const leftFork = buildTimberLimb({
    rings: 20,
    radialSegs: 16,
    sample: (t, out) => {
      const ease = t * (0.88 + 0.12 * t);
      out.x = leftRoot.x - 0.36 * ease - 0.04 * Math.sin(t * 2.5);
      out.y = leftRoot.y + 0.02 + 0.35 * Math.pow(t, 0.92);
      out.z = leftRoot.z - 0.15 * t + 0.03 * Math.sin(t * 3.1);
      out.r = 0.068 * (1 - t * 0.64) + 0.005 * Math.sin(t * 8.0);
    },
    startShear: 0.0,
    endShear: 0.95,
    furrowScale: 0.92,
    seedPhase: 2.3,
  });

  // 3. Right slender bough fork emerging from x = +1.30
  const rightRoot = evalBranchCenterline(1.30);
  const rightFork = buildTimberLimb({
    rings: 22,
    radialSegs: 16,
    sample: (t, out) => {
      out.x = rightRoot.x + 0.48 * t + 0.03 * Math.sin(t * 2.7);
      out.y = rightRoot.y + 0.01 + 0.25 * Math.pow(t, 1.08) + 0.03 * Math.sin(t * 2.2);
      out.z = rightRoot.z + 0.13 * t - 0.04 * Math.sin(t * 3.0);
      out.r = 0.056 * (1 - t * 0.68);
    },
    startShear: 0.0,
    endShear: 0.85,
    furrowScale: 0.85,
    seedPhase: 4.1,
  });

  // 4. Short weathered lower knot stub near x = -1.56 for asymmetrical forest silhouette
  const stubRoot = evalBranchCenterline(-1.56);
  const lowerStub = buildTimberLimb({
    rings: 12,
    radialSegs: 14,
    sample: (t, out) => {
      out.x = stubRoot.x - 0.17 * t;
      out.y = stubRoot.y - 0.02 - 0.13 * t;
      out.z = stubRoot.z - 0.14 * t;
      out.r = 0.046 * (1 - t * 0.56);
    },
    startShear: 0.0,
    endShear: 0.9,
    furrowScale: 0.8,
    seedPhase: 5.8,
  });

  return mergeIndexedGeometries([mainBough, leftFork, rightFork, lowerStub]);
}

/**
 * Builds a closed, watertight, outward-wound 3D bivouac curtain manifold sharing
 * the exact `evalCurtainPoint` surface (used for both the dark inner brood core at
 * `insetScale = 0.945` and the invisible predator-approach raycast hull at `1.07`).
 */
export function buildCurtainShellGeometry(
  insetScale: number,
  meridianSegs = 36,
  ringSegs = 56
): THREE.BufferGeometry {
  const cols = ringSegs + 1;
  // Rows 0..meridianSegs span theta from 0.004 to PI/2 - 0.012, plus top & bottom pole closures
  const sideVerts = (meridianSegs + 1) * cols;
  const topPoleIdx = sideVerts;
  const botPoleIdx = sideVerts + 1;
  const totalVerts = sideVerts + 2;

  const positions = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const indices: number[] = [];

  const p: [number, number, number] = [0, 0, 0];
  const thetaMin = 0.004;
  const thetaMax = Math.PI * 0.5 - 0.012;

  for (let i = 0; i <= meridianSegs; i++) {
    const v = i / meridianSegs;
    const th = thetaMin + (thetaMax - thetaMin) * v;
    for (let j = 0; j <= ringSegs; j++) {
      const u = j / ringSegs;
      const phi = u * Math.PI * 2;
      evalCurtainPoint(th, phi, p, insetScale);
      const idx = i * cols + j;
      positions[idx * 3] = p[0];
      positions[idx * 3 + 1] = p[1];
      positions[idx * 3 + 2] = p[2];
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  // Top closure tucked squarely inside branch centerline (y = branch(0).y - CURTAIN_PIVOT_Y)
  const midBranch = evalBranchCenterline(0);
  positions[topPoleIdx * 3] = 0;
  positions[topPoleIdx * 3 + 1] = midBranch.y - CURTAIN_PIVOT_Y;
  positions[topPoleIdx * 3 + 2] = midBranch.z;
  uvs[topPoleIdx * 2] = 0.5;
  uvs[topPoleIdx * 2 + 1] = 0;

  // Bottom smooth apex closure
  evalCurtainPoint(Math.PI * 0.5, 0, p, insetScale);
  positions[botPoleIdx * 3] = p[0];
  positions[botPoleIdx * 3 + 1] = p[1];
  positions[botPoleIdx * 3 + 2] = p[2];
  uvs[botPoleIdx * 2] = 0.5;
  uvs[botPoleIdx * 2 + 1] = 1;

  // Check winding on curtain surface:
  // As j increases (+phi: front +Z moves toward -X), as i increases (+theta: moves DOWN -Y).
  // So (dP/dphi) x (dP/dtheta) = (-X) x (-Y) = +Z (OUTWARD!).
  // Thus triangle (a, b, c) where b = a+1 (+phi) and c = nextRow (+theta) is OUTWARD CCW!
  for (let i = 0; i < meridianSegs; i++) {
    for (let j = 0; j < ringSegs; j++) {
      const a = i * cols + j;
      const b = a + 1;
      const c = (i + 1) * cols + j;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  // Top cap facing +Y (outward upward into branch): (topPole, row0[j+1], row0[j])
  for (let j = 0; j < ringSegs; j++) {
    indices.push(topPoleIdx, j + 1, j);
  }
  // Bottom cap facing -Y (outward downward): (botPole, lastRow[j], lastRow[j+1])
  const lastRow = meridianSegs * cols;
  for (let j = 0; j < ringSegs; j++) {
    indices.push(botPoleIdx, lastRow + j, lastRow + j + 1);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Helper to build a closed, watertight, outward-wound sculpted insect body segment
 * lofted along the local Y axis (with independent dorsal +Z and ventral -Z profiles).
 *
 * Profile rings MUST be ordered strictly bottom-to-top (`y` strictly increasing) so
 * outward CCW winding is mathematically guaranteed (`computeSignedVolume > 0`).
 */
interface BodyRing {
  y: number;
  rx: number;
  zDorsal: number;
  zVentral: number;
  zCenterOffset?: number;
  tag0: number;
  tag1: number;
  tag2: number;
}

function buildSculptedBodyLoft(
  rings: readonly BodyRing[],
  radialSegs: number,
  attrName: string
): THREE.BufferGeometry {
  const nRings = rings.length;
  const cols = radialSegs + 1;
  const sideVerts = nRings * cols;
  const botCenterIdx = sideVerts;
  const topCenterIdx = sideVerts + 1;
  const totalVerts = sideVerts + 2;

  const positions = new Float32Array(totalVerts * 3);
  const customAttr = new Float32Array(totalVerts * 4);
  const indices: number[] = [];

  for (let i = 0; i < nRings; i++) {
    const r = rings[i];
    const zOff = r.zCenterOffset ?? 0;
    for (let j = 0; j <= radialSegs; j++) {
      // Angle around +Y axis: a=0 -> +X, a=PI/2 -> -Z, a=PI -> -X, a=3PI/2 -> +Z
      // Wait: check right-hand basis with axis = +Y!
      // Set B1 = +X, B2 = -Z -> B1 x B2 = (+X) x (-Z) = +Y!
      // Then (dP/da) x (dP/dy) = +radial outward!
      const ang = (j / radialSegs) * Math.PI * 2;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang); // -Z component! So dorsal (+Z) is where -sa > 0!
      const zSign = -sa; // +1 at dorsal (+Z), -1 at ventral (-Z)

      const x = ca * r.rx;
      const z = zOff + (zSign >= 0 ? zSign * r.zDorsal : zSign * r.zVentral);
      const dorsal01 = Math.max(0, Math.min(1, zSign * 0.5 + 0.5));
      const flank01 = Math.abs(ca);

      const vIdx = i * cols + j;
      positions[vIdx * 3] = x;
      positions[vIdx * 3 + 1] = r.y;
      positions[vIdx * 3 + 2] = z;

      customAttr[vIdx * 4] = r.tag0;
      customAttr[vIdx * 4 + 1] = dorsal01;
      customAttr[vIdx * 4 + 2] = r.tag1;
      customAttr[vIdx * 4 + 3] = r.tag2 * 0.5 + flank01 * 0.5;
    }
  }

  // Bottom cap center (y = rings[0].y)
  const first = rings[0];
  positions[botCenterIdx * 3] = 0;
  positions[botCenterIdx * 3 + 1] = first.y - 0.0012;
  positions[botCenterIdx * 3 + 2] = (first.zCenterOffset ?? 0) + (first.zDorsal - first.zVentral) * 0.25;
  customAttr[botCenterIdx * 4] = first.tag0;
  customAttr[botCenterIdx * 4 + 1] = 0.5;
  customAttr[botCenterIdx * 4 + 2] = first.tag1;
  customAttr[botCenterIdx * 4 + 3] = first.tag2;

  // Top cap center (y = rings[nRings-1].y)
  const last = rings[nRings - 1];
  positions[topCenterIdx * 3] = 0;
  positions[topCenterIdx * 3 + 1] = last.y + 0.0012;
  positions[topCenterIdx * 3 + 2] = (last.zCenterOffset ?? 0) + (last.zDorsal - last.zVentral) * 0.25;
  customAttr[topCenterIdx * 4] = last.tag0;
  customAttr[topCenterIdx * 4 + 1] = 0.5;
  customAttr[topCenterIdx * 4 + 2] = last.tag1;
  customAttr[topCenterIdx * 4 + 3] = last.tag2;

  // Side quads: (a, b, c) and (b, d, c) with B1=+X, B2=-Z, axis=+Y -> strict outward CCW!
  for (let i = 0; i < nRings - 1; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * cols + j;
      const b = a + 1;
      const c = (i + 1) * cols + j;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  // Bottom cap (outward = -Y): (botCenter, j+1, j)
  for (let j = 0; j < radialSegs; j++) {
    indices.push(botCenterIdx, j + 1, j);
  }
  // Top cap (outward = +Y): (topCenter, lastRow+j, lastRow+j+1)
  const lastRow = (nRings - 1) * cols;
  for (let j = 0; j < radialSegs; j++) {
    indices.push(topCenterIdx, lastRow + j, lastRow + j + 1);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute(attrName, new THREE.BufferAttribute(customAttr, 4));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Builds an anatomically sculpted *Apis dorsata* anchored Head + Fuzzy Mesothorax
 * positioned above the petiole hinge `(0, 0, 0)` along `+Y` (toward the branch).
 *
 * Attribute `aThoraxData`: vec4(alongY01, dorsal01, isHead01, eyeFlank01).
 */
export function buildBeeThoraxGeometry(): THREE.BufferGeometry {
  // Ordered strictly bottom-to-top (y increasing) for guaranteed outward CCW winding
  const rings: BodyRing[] = [
    // Petiole waist junction
    { y: 0.000, rx: 0.0055, zDorsal: 0.0065, zVentral: 0.0030, zCenterOffset: 0.002, tag0: 0.0, tag1: 0.0, tag2: 0.0 },
    // Posterior scutellum
    { y: 0.008, rx: 0.0145, zDorsal: 0.0175, zVentral: 0.0055, zCenterOffset: 0.004, tag0: 0.2, tag1: 0.0, tag2: 0.2 },
    // Broad vaulted velvety mesothorax
    { y: 0.019, rx: 0.0185, zDorsal: 0.0225, zVentral: 0.0065, zCenterOffset: 0.005, tag0: 0.45, tag1: 0.0, tag2: 0.3 },
    // Anterior pronotum collar
    { y: 0.030, rx: 0.0150, zDorsal: 0.0185, zVentral: 0.0060, zCenterOffset: 0.003, tag0: 0.68, tag1: 0.15, tag2: 0.2 },
    // Cervical neck pinch
    { y: 0.035, rx: 0.0095, zDorsal: 0.0105, zVentral: 0.0050, zCenterOffset: 0.001, tag0: 0.78, tag1: 0.65, tag2: 0.1 },
    // Sculpted head capsule with bilateral compound-eye width
    { y: 0.042, rx: 0.0148, zDorsal: 0.0125, zVentral: 0.0070, zCenterOffset: -0.001, tag0: 0.90, tag1: 1.0, tag2: 1.0 },
    // Vertex / frons crown gripping the curtain
    { y: 0.049, rx: 0.0075, zDorsal: 0.0065, zVentral: 0.0045, zCenterOffset: -0.003, tag0: 1.0, tag1: 1.0, tag2: 0.5 },
  ];

  return buildSculptedBodyLoft(rings, 8, 'aThoraxData');
}

/**
 * Builds an anatomically sculpted, 5-tergite segmented *Apis dorsata* hinged Abdomen (Gaster)
 * pivoting at the petiole `(0, 0, 0)` and draping downward along `-Y` (`y` down to `-0.070`).
 *
 * Includes real physical telescoping tergite steps so studio softbox light catches
 * genuine 3D chitin segmentation when abdomens snap outward 65-75 deg during a wave!
 *
 * Attribute `aAbdomenUv`: vec4(alongFromPetiole01, dorsal01, tergiteRidge01, sterniteRim01).
 */
export function buildBeeAbdomenGeometry(): THREE.BufferGeometry {
  // Ordered strictly bottom-to-top (apical stinger tip y=-0.070 -> petiole hinge y=0.000)
  // so outward CCW winding is strictly guaranteed (`computeSignedVolume > 0`).
  const rings: BodyRing[] = [
    // T6 tapered dark apical stinger tip (slight natural downward/inward arch)
    { y: -0.070, rx: 0.0026, zDorsal: 0.0032, zVentral: 0.0024, zCenterOffset: 0.001, tag0: 1.00, tag1: 0.15, tag2: 0.0 },
    // T5 dark velvety apical tergite
    { y: -0.059, rx: 0.0095, zDorsal: 0.0110, zVentral: 0.0055, zCenterOffset: 0.004, tag0: 0.84, tag1: 0.72, tag2: 0.2 },
    // T4/T5 suture step
    { y: -0.048, rx: 0.0150, zDorsal: 0.0170, zVentral: 0.0072, zCenterOffset: 0.006, tag0: 0.69, tag1: 0.20, tag2: 0.45 },
    // T4 stepped band ridge
    { y: -0.042, rx: 0.0172, zDorsal: 0.0192, zVentral: 0.0078, zCenterOffset: 0.007, tag0: 0.60, tag1: 0.92, tag2: 0.65 },
    // T3/T4 ochre-umber suture
    { y: -0.032, rx: 0.0192, zDorsal: 0.0212, zVentral: 0.0082, zCenterOffset: 0.007, tag0: 0.46, tag1: 0.25, tag2: 0.85 },
    // T2/T3 widest golden-amber tergite crown (vivid Yellow Ochre & Realgar-Amber flash zone!)
    { y: -0.022, rx: 0.0208, zDorsal: 0.0230, zVentral: 0.0085, zCenterOffset: 0.007, tag0: 0.31, tag1: 1.00, tag2: 1.00 },
    // T1 golden basal shoulder
    { y: -0.010, rx: 0.0172, zDorsal: 0.0190, zVentral: 0.0068, zCenterOffset: 0.005, tag0: 0.14, tag1: 0.78, tag2: 0.85 },
    // Petiole waist hinge at (0, 0, 0)
    { y: 0.000, rx: 0.0060, zDorsal: 0.0068, zVentral: 0.0032, zCenterOffset: 0.002, tag0: 0.00, tag1: 0.10, tag2: 0.35 },
  ];

  return buildSculptedBodyLoft(rings, 8, 'aAbdomenUv');
}

/**
 * Builds a sleek, cambered pair of translucent gossamer insect forewings anchored at
 * local `(±0.005, 0, 0)` (dorsal mesothorax pivot) and sweeping downward over the abdomen.
 *
 * Attribute `aWingUv`: vec4(uSpan [0=root..1=tip], vChord [-1=leading costal..+1=trailing], sideSign [-1|+1], camber01).
 */
export function buildBeeWingsGeometry(): THREE.BufferGeometry {
  // Each wing is a gracefully tapered 5x2 grid (16 tris per wing -> 32 tris per pair)
  const spanSegs = 4;
  const chordSegs = 2;
  const vertsPerWing = (spanSegs + 1) * (chordSegs + 1);
  const totalVerts = vertsPerWing * 2;

  const positions = new Float32Array(totalVerts * 3);
  const wingUv = new Float32Array(totalVerts * 4);
  const indices: number[] = [];

  const wingLen = 0.068;
  const maxWidth = 0.021;

  for (let side = 0; side < 2; side++) {
    const sign = side === 0 ? -1 : 1;
    const baseVert = side * vertsPerWing;

    // Slight natural V-chevron folding angle (~11.5 deg outward from spine)
    const sweepAngle = sign * 0.20;
    const cosS = Math.cos(sweepAngle);
    const sinS = Math.sin(sweepAngle);

    for (let i = 0; i <= spanSegs; i++) {
      const u = i / spanSegs; // 0 = base at dorsal thorax, 1 = rounded wingtip
      // Authentic hymenopteran forewing planform: slender petiole base -> broad sub-apical paddle -> crisp rounded apex
      const widthEnv =
        Math.pow(Math.sin(u * Math.PI * 0.88 + 0.12), 0.72) * (0.28 + 0.72 * Math.sin(u * Math.PI * 0.75 + 0.35));
      const chordW = maxWidth * Math.max(0.12, widthEnv);

      // Leading costal edge sweeps gently outward, trailing margin tucks inward over abdomen
      const spanY = -u * wingLen;
      const centerX = sign * (0.0055 + u * 0.0075);

      for (let j = 0; j <= chordSegs; j++) {
        const vNorm = j / chordSegs; // 0 = outer/leading costal vein, 1 = inner/trailing edge
        const vChord = vNorm * 2 - 1; // -1 = leading costal vein, +1 = inner trailing margin

        // Local chord offset: costal edge sits further outward (`+sign`), trailing edge tucks slightly over midline
        const localChordX = sign * (0.55 - vNorm) * chordW;
        const rotX = localChordX * cosS - spanY * sinS * 0.32;
        const rotY = spanY + Math.abs(localChordX) * 0.08 * (1 - u);

        // Delicate convex aerodynamic wing camber & subtle dihedral lift away from dorsal tergites
        const camberArch = (1 - vChord * vChord) * 0.0042 * Math.sin(u * Math.PI);
        const dorsalClearance = 0.005 + 0.0065 * Math.sin(u * Math.PI * 0.85) + (1 - vNorm) * 0.0028;

        const vIdx = baseVert + i * (chordSegs + 1) + j;
        positions[vIdx * 3] = centerX + rotX;
        positions[vIdx * 3 + 1] = rotY;
        positions[vIdx * 3 + 2] = dorsalClearance + camberArch;

        wingUv[vIdx * 4] = u;
        wingUv[vIdx * 4 + 1] = vChord;
        wingUv[vIdx * 4 + 2] = sign;
        wingUv[vIdx * 4 + 3] = Math.max(0, 1 - vChord * vChord);
      }
    }

    // Winding so dorsal (+Z) surface is FrontSide on BOTH left and right wings:
    // - sign > 0: +j moves -X, +i moves -Y => (-X) x (-Y) = +Z -> (a, b, c)
    // - sign < 0: +j moves +X, +i moves -Y => (-Y) x (+X) = +Z -> (a, c, b)
    for (let i = 0; i < spanSegs; i++) {
      for (let j = 0; j < chordSegs; j++) {
        const a = baseVert + i * (chordSegs + 1) + j;
        const b = a + 1;
        const c = baseVert + (i + 1) * (chordSegs + 1) + j;
        const d = c + 1;
        if (sign > 0) {
          indices.push(a, b, c, b, d, c);
        } else {
          indices.push(a, c, b, b, c, d);
        }
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('aWingUv', new THREE.BufferAttribute(wingUv, 4));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}
