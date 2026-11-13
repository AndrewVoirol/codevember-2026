import * as THREE from 'three/webgpu';

/**
 * Day 12 — Feather ("Goniometric Peacock Fan")
 * Procedural 3D Anatomy & Articulated Clasp Geometry Builder
 *
 * Builds 7 layered, curved 3D Indian peacock (*Pavo cristatus*) train feathers
 * mounted in a stepped 7-leaf dark-bronze & Bone Calcite pivot clasp:
 * - Zero flat textured quads: every feather is a curved tapered 3D rachis plus
 *   368 individual 3D cambered, V-tilted barb ribbons (~41k triangles total).
 * - Preserves exact ocellus-local coordinates (`aOcellusCoord`) on every vertex so
 *   structural photonic bands bend, flex, and separate physically with the barbs.
 */

export const BLADE_COUNT = 7;
export const PIVOT_Y = -0.80;
export const Z_LAYER_STEP = 0.024;

/** Normalized spine position where the ocellus heart sits */
export const OCELLUS_CENTER_S = 0.775;

/** Base length & open/furled angular spread profile for the 7 blades (left-back 0 -> right-front 6) */
export const BLADE_LENGTHS: readonly number[] = [1.74, 1.82, 1.88, 1.91, 1.88, 1.82, 1.74];
export const BLADE_OPEN_ANGLES_RAD: readonly number[] = [
  -1.10, -0.735, -0.368, 0.0, 0.368, 0.735, 1.10,
];
export const BLADE_FURLED_ANGLES_RAD: readonly number[] = [
  -0.052, -0.035, -0.017, 0.0, 0.017, 0.035, 0.052,
];

/**
 * Axial shingle slide offset when furled (`fanOpen = 0`):
 * Frontmost blade (6) sits lowest near the clasp and overlaps in front of 5, 4, ..., 0 (tallest in back),
 * matching the staggered vertical staircase in Shot 02 (`shot_02_furled_stack`).
 */
export const BLADE_FURLED_SLIDES: readonly number[] = [
  0.56, 0.36, 0.17, 0.00, -0.12, -0.22, -0.30,
];

function seededHash(a: number, b: number): number {
  const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

/**
 * Evaluates the 3D centerline of a feather rachis at normalized arc parameter `s in [-0.30, 0.90]`.
 * Negative `s` provides tucked sleeve slack when furled; vertices below the clasp pivot
 * are collapsed cleanly inside the bronze hub in `createRachisMaterial`.
 */
export function evalRachisSpine(
  s: number,
  featherLength: number,
  leanSign: number,
  outPos: THREE.Vector3
): THREE.Vector3 {
  const y = s * featherLength;
  const sClamped = Math.max(0, Math.min(1, s));
  const xCurve = leanSign * 0.022 * Math.sin(sClamped * Math.PI * 1.1);
  const zCamber = 0.044 * Math.sin(sClamped * Math.PI * 0.88) - 0.014 * sClamped * sClamped;
  return outPos.set(xCurve, y, zCamber);
}

/**
 * Builds the curved, tapered 3D bone-calcite-to-horn rachis (quill) BufferGeometry for one blade.
 * Features an anatomical longitudinal dorsal furrow (median sulcus), subtle lateral vane
 * attachment flanges, and continuous seam-free cylindrical UV (`aSpineUv`).
 */
export function createRachisGeometry(bladeIndex: number, featherLength: number): THREE.BufferGeometry {
  const lengthSteps = 56;
  const radialSegs = 14;
  const ringVerts = radialSegs + 1;
  const leanSign = bladeIndex === 3 ? 0.12 : Math.sign(bladeIndex - 3);

  const ringCount = lengthSteps + 1;
  const vertCount = ringCount * ringVerts;
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const spineUv = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  const pCurr = new THREE.Vector3();
  const pNext = new THREE.Vector3();
  const tang = new THREE.Vector3();
  const normA = new THREE.Vector3();
  const normB = new THREE.Vector3();
  const upRef = new THREE.Vector3(0, 0, 1);

  for (let i = 0; i <= lengthSteps; i++) {
    const t = i / lengthSteps;
    // Extend from -0.38 (hidden inside clasp when furled) and terminate cleanly inside ocellus heart (0.785)
    const s = -0.38 + t * 1.165;
    evalRachisSpine(s, featherLength, leanSign, pCurr);
    evalRachisSpine(s + 0.01, featherLength, leanSign, pNext);
    tang.subVectors(pNext, pCurr).normalize();

    normA.crossVectors(upRef, tang).normalize();
    normB.crossVectors(tang, normA).normalize();

    const sPos = Math.max(0, s);
    // Sleek quill taper housing gracefully inside the bronze clasp sheath (fits well inside +-0.017 bronze depth)
    const radius = 0.0138 * Math.pow(Math.max(0.01, 1.0 - 1.24 * Math.min(0.79, sPos)), 0.76);
    const rx = radius * 1.06;
    const rz = radius * 0.80;
    const sulcusMask = Math.min(1, Math.max(0, (sPos - 0.05) / 0.48));

    for (let j = 0; j <= radialSegs; j++) {
      const ang = (j / radialSegs) * Math.PI * 2;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      const vi = i * ringVerts + j;

      // Subtle anatomical central furrow along front/back midline + pinched vane margin
      const sulcus = Math.exp(-ca * ca * 7.5) * 0.16 * sulcusMask;
      const flange = Math.exp(-(1 - Math.abs(ca)) * 9.0) * 0.06 * sulcusMask;
      const profile = 1.0 - sulcus + flange;

      const lx = ca * rx * profile;
      const lz = sa * rz * profile;

      positions[vi * 3 + 0] = pCurr.x + normA.x * lx + normB.x * lz;
      positions[vi * 3 + 1] = pCurr.y + normA.y * lx + normB.y * lz;
      positions[vi * 3 + 2] = pCurr.z + normA.z * lx + normB.z * lz;

      // Slight furrow bevel on surface normal for clear crisp sulcus twin-highlight
      const sulcusSlope = ca * sulcus * 2.2;
      const nx = normA.x * (ca + sulcusSlope) + normB.x * sa;
      const ny = normA.y * (ca + sulcusSlope) + normB.y * sa;
      const nz = normA.z * (ca + sulcusSlope) + normB.z * sa;
      const invLen = 1 / Math.max(1e-5, Math.hypot(nx, ny, nz));
      normals[vi * 3 + 0] = nx * invLen;
      normals[vi * 3 + 1] = ny * invLen;
      normals[vi * 3 + 2] = nz * invLen;

      spineUv[vi * 2 + 0] = s;
      spineUv[vi * 2 + 1] = ang;
    }
  }

  for (let i = 0; i < lengthSteps; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * ringVerts + j;
      const b = i * ringVerts + (j + 1);
      const c = (i + 1) * ringVerts + j;
      const d = (i + 1) * ringVerts + (j + 1);
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('aSpineUv', new THREE.BufferAttribute(spineUv, 2));
  geom.setIndex(indices);
  return geom;
}

interface BarbSpec {
  s: number;
  side: -1 | 1;
  length: number;
  baseAngle: number;
  tipSweep: number;
  widthRoot: number;
  widthTip: number;
  droop: number;
  archZ: number;
  seed: number;
  isOcellus: number;
  segs: number;
}

/**
 * Generates multi-scale 3D barb ribbon mesh for one peacock feather blade:
 * - Zone A (s in [0.10, 0.21]): Soft gossamer basal down emerging above bone ferrule.
 * - Zone B (s in [0.17, 0.61]): 82 pairs of sweeping lower filigree herl strands with
 *   per-strand stagger & herringbone barbule UV envelopes (`aBarbGeom`).
 * - Zone C (s in [0.612, 0.824]): 160 pairs of dense curved ocellus & fringed crown strands.
 */
export function createFeatherBarbsGeometry(
  bladeIndex: number,
  featherLength: number
): THREE.BufferGeometry {
  const specs: BarbSpec[] = [];
  const leanSign = bladeIndex === 3 ? 0.12 : Math.sign(bladeIndex - 3);

  const isOuter = bladeIndex === 0 || bladeIndex === BLADE_COUNT - 1;
  // 0. Soft wispy plumulaceous basal down starting cleanly ABOVE each bone ferrule collar
  const downStartS = isOuter ? 0.188 : 0.145;
  const downSpanS = isOuter ? 0.088 : 0.105;
  const downPairs = 22;
  for (let k = 0; k < downPairs; k++) {
    const frac = k / (downPairs - 1);
    const s = downStartS + frac * downSpanS;
    for (const side of [-1, 1] as const) {
      const r0 = seededHash(bladeIndex * 11 + k + 50, side + 2);
      const r1 = seededHash(bladeIndex * 23 + k + 50, side + 9);
      const r2 = seededHash(bladeIndex * 37 + k + 50, side + 15);

      specs.push({
        s,
        side,
        length: (0.068 + 0.115 * frac) * (0.85 + 0.30 * r0),
        baseAngle: 1.04 - 0.18 * frac + (r1 - 0.5) * 0.14,
        tipSweep: 0.22 + 0.20 * r2,
        widthRoot: 0.0058,
        widthTip: 0.0028,
        droop: -0.022 * (0.4 + 0.6 * r0),
        archZ: (r2 - 0.5) * 0.024 + 0.008,
        seed: r0,
        isOcellus: 0.0,
        segs: 8,
      });
    }
  }

  // 1. Lower graceful filigree herl filaments (82 pairs = 164 strands)
  const lowerStartS = isOuter ? 0.215 : 0.175;
  const lowerEndS = 0.608;
  const lowerPairs = 82;
  for (let k = 0; k < lowerPairs; k++) {
    const frac = k / (lowerPairs - 1);
    const s = lowerStartS + frac * (lowerEndS - lowerStartS);
    for (const side of [-1, 1] as const) {
      const r0 = seededHash(bladeIndex * 19 + k, side + 3);
      const r1 = seededHash(bladeIndex * 31 + k, side + 7);
      const r2 = seededHash(bladeIndex * 43 + k, side + 11);

      const length = (0.19 + 0.345 * Math.sin(frac * Math.PI * 0.72 + 0.22)) * (0.89 + 0.22 * r0);
      const baseAngle = 0.81 - 0.16 * frac + (r1 - 0.5) * 0.075;
      const tipSweep = 0.13 + 0.10 * r2;
      const droop = -0.030 * (0.3 + 0.7 * frac) * (0.82 + 0.36 * r0);

      specs.push({
        s,
        side,
        length,
        baseAngle,
        tipSweep,
        widthRoot: 0.0076,
        widthTip: 0.0038,
        droop,
        archZ: 0.020 * Math.sin(frac * Math.PI) + (r2 - 0.5) * 0.015,
        seed: r0,
        isOcellus: 0.0,
        segs: 10,
      });
    }
  }

  // 2. Broad, rounded peacock ocellus paddle & radiating fringe crown (160 pairs = 320 strands)
  const upperPairs = 160;
  for (let k = 0; k < upperPairs; k++) {
    const frac = k / (upperPairs - 1);
    const s = 0.612 + frac * 0.212; // 0.612 -> 0.824 (preserves all 7 staggered eyes in Shot 02!)
    for (const side of [-1, 1] as const) {
      const r0 = seededHash(bladeIndex * 53 + k + 200, side + 5);
      const r1 = seededHash(bladeIndex * 67 + k + 200, side + 13);
      const r2 = seededHash(bladeIndex * 79 + k + 200, side + 17);

      const waist = Math.sin(Math.pow(frac, 0.76) * Math.PI * 0.84 + 0.26);
      const crownTrim = 1.0 - 0.56 * Math.pow(frac, 1.85);
      // Occasional delicate protruding outer fringe plumelets along halo edge
      const outlierTip = r0 > 0.82 ? 1.06 : 1.0;
      const fringeJitter = (0.91 + 0.18 * r0) * outlierTip;
      const length = (0.16 + 0.218 * waist) * crownTrim * fringeJitter;

      const baseAngle = 0.82 * (1.0 - Math.pow(frac, 1.28)) + 0.08 + (r1 - 0.5) * 0.05;
      const tipSweep = 0.06 * (0.45 - frac) + (r2 - 0.5) * 0.055;

      specs.push({
        s,
        side,
        length,
        baseAngle,
        tipSweep,
        widthRoot: 0.0098,
        widthTip: 0.0040,
        droop: 0.010 * (1.0 - frac),
        archZ: 0.017 * waist + (r2 - 0.5) * 0.007,
        seed: r0,
        isOcellus: 1.0,
        segs: 9,
      });
    }
  }

  let totalVerts = 0;
  const specVertOffsets: number[] = new Array(specs.length);
  for (let b = 0; b < specs.length; b++) {
    specVertOffsets[b] = totalVerts;
    totalVerts += (specs[b].segs + 1) * 2;
  }

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const barbData = new Float32Array(totalVerts * 4);
  const barbGeom = new Float32Array(totalVerts * 2);
  const ocellusCoord = new Float32Array(totalVerts * 2);
  const indices: number[] = [];

  const rootPos = new THREE.Vector3();
  const spineAhead = new THREE.Vector3();
  const spineTang = new THREE.Vector3();

  const ocellusCenterPos = new THREE.Vector3();
  evalRachisSpine(OCELLUS_CENTER_S, featherLength, leanSign, ocellusCenterPos);

  for (let b = 0; b < specs.length; b++) {
    const sp = specs[b];
    evalRachisSpine(sp.s, featherLength, leanSign, rootPos);
    evalRachisSpine(sp.s + 0.01, featherLength, leanSign, spineAhead);
    spineTang.subVectors(spineAhead, rootPos).normalize();

    const dirAngleBase = Math.atan2(spineTang.y, spineTang.x);
    const baseVertexOffset = specVertOffsets[b];
    const stations = sp.segs + 1;

    for (let st = 0; st < stations; st++) {
      const u = st / sp.segs;

      const sweep =
        sp.side === 1
          ? dirAngleBase - sp.baseAngle - sp.tipSweep * u * u
          : dirAngleBase + sp.baseAngle + sp.tipSweep * u * u;

      const curvePhase = sp.side === 1 ? -sp.tipSweep * 0.30 * u : sp.tipSweep * 0.30 * u;
      // Subtle high-order barb waviness on lower/down filaments for micro-organic silk flow
      const waveOffset =
        sp.isOcellus < 0.5
          ? Math.sin(u * 5.2 + sp.seed * 6.28) * 0.010 * u
          : Math.sin(u * 3.4 + sp.seed * 6.28) * 0.003 * u;
      const effAngle = sweep + curvePhase;

      const rootTuckZ = (sp.isOcellus < 0.5 ? -0.016 : -0.008) * (1.0 - u);
      const rDist = (sp.isOcellus < 0.5 ? 0.003 : 0.004) + sp.length * u;
      const tx = Math.cos(effAngle);
      const ty = Math.sin(effAngle);
      const perpX = -ty;
      const perpY = tx;

      const cx = rootPos.x + tx * rDist + perpX * waveOffset;
      const cy = rootPos.y + ty * rDist + perpY * waveOffset + sp.droop * u * u;
      const cz = rootPos.z + sp.archZ * Math.sin(u * Math.PI) - 0.022 * u * u + rootTuckZ;

      const taperExp = sp.isOcellus > 0.5 ? 1.75 : 1.30;
      const tipTaper = 1.0 - Math.pow(u, taperExp);
      const halfW = 0.5 * (sp.widthTip + (sp.widthRoot - sp.widthTip) * tipTaper);
      const crossTiltZ = 0.34;
      const vaneDihedralX = sp.side * 0.14 * (1.0 - u * 0.4);

      for (let edge = 0; edge < 2; edge++) {
        const edgeSign = edge === 0 ? -1 : 1;
        const vi = baseVertexOffset + st * 2 + edge;

        const px = cx + perpX * halfW * edgeSign;
        const py = cy + perpY * halfW * edgeSign;
        const pz = cz - Math.abs(edgeSign) * halfW * 0.28;

        positions[vi * 3 + 0] = px;
        positions[vi * 3 + 1] = py;
        positions[vi * 3 + 2] = pz;

        const nx = perpX * edgeSign * crossTiltZ + vaneDihedralX;
        const ny = perpY * edgeSign * crossTiltZ - 0.06 * u;
        const nz = 1.0 - 0.12 * u;
        const nInv = 1.0 / Math.max(1e-5, Math.hypot(nx, ny, nz));
        normals[vi * 3 + 0] = nx * nInv;
        normals[vi * 3 + 1] = ny * nInv;
        normals[vi * 3 + 2] = nz * nInv;

        barbData[vi * 4 + 0] = sp.s;
        barbData[vi * 4 + 1] = u;
        barbData[vi * 4 + 2] = sp.side;
        barbData[vi * 4 + 3] = sp.seed + sp.isOcellus * 10.0;

        barbGeom[vi * 2 + 0] = edgeSign;
        barbGeom[vi * 2 + 1] = sp.length;

        ocellusCoord[vi * 2 + 0] = px - ocellusCenterPos.x;
        ocellusCoord[vi * 2 + 1] = py - ocellusCenterPos.y;
      }

      if (st < sp.segs) {
        const i0 = baseVertexOffset + st * 2;
        const i1 = i0 + 1;
        const i2 = i0 + 2;
        const i3 = i0 + 3;
        indices.push(i0, i2, i1);
        indices.push(i1, i2, i3);
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('aBarbData', new THREE.BufferAttribute(barbData, 4));
  geom.setAttribute('aBarbGeom', new THREE.BufferAttribute(barbGeom, 2));
  geom.setAttribute('aOcellusCoord', new THREE.BufferAttribute(ocellusCoord, 2));
  geom.setIndex(indices);
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Builds one machined dark-bronze leaf guard tab for the stepped pivot clasp (`Shot 04`),
 * with narrow upper seat tongue that plugs cleanly INSIDE the lathe-turned bone ferrule bore
 * (zero coplanar Z-fighting triangles!).
 */
export function createBronzeGuardLeafGeometry(isOuterArm: boolean): THREE.BufferGeometry {
  const shape = new THREE.Shape();

  if (isOuterArm) {
    shape.moveTo(-0.044, -0.020);
    shape.absarc(0, 0, 0.058, Math.PI * 0.9, Math.PI * 1.95, false);
    shape.quadraticCurveTo(0.044, 0.11, 0.015, 0.222);
    shape.lineTo(0.010, 0.242);
    shape.lineTo(-0.010, 0.242);
    shape.lineTo(-0.015, 0.222);
    shape.quadraticCurveTo(-0.044, 0.11, -0.044, -0.020);
  } else {
    shape.moveTo(-0.042, -0.016);
    shape.absarc(0, -0.025, 0.064, Math.PI * 0.85, Math.PI * 2.15, false);
    shape.quadraticCurveTo(0.040, 0.10, 0.013, 0.168);
    shape.lineTo(0.009, 0.184);
    shape.lineTo(-0.009, 0.184);
    shape.lineTo(-0.013, 0.168);
    shape.quadraticCurveTo(-0.040, 0.10, -0.042, -0.016);
  }

  // Deep cast sheath thickness (z in [-0.017, +0.017]) cleanly encloses the inner calamus quill without Z-poking
  const extrude = new THREE.ExtrudeGeometry(shape, {
    depth: 0.026,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.003,
    bevelSegments: 3,
    curveSegments: 24,
  });
  extrude.translate(0, 0, -0.013);
  extrude.computeVertexNormals();
  return extrude;
}

/**
 * Builds a sleek lathe-turned horn/bone ferrule collar gripping the quill cleanly above the bronze tang.
 */
export function createBoneFerruleGeometry(radiusTop = 0.017, radiusBot = 0.024, height = 0.105): THREE.BufferGeometry {
  const rInnerTop = Math.max(0.011, radiusTop * 0.72);
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(rInnerTop * 0.88, 0.0),
    new THREE.Vector2(radiusBot, 0.0),
    new THREE.Vector2(radiusBot * 1.03, height * 0.06),
    new THREE.Vector2(radiusBot * 0.99, height * 0.16),
    new THREE.Vector2(radiusBot * 0.72 + radiusTop * 0.28, height * 0.48),
    new THREE.Vector2(radiusBot * 0.28 + radiusTop * 0.72, height * 0.78),
    new THREE.Vector2(radiusTop * 1.02, height * 0.88),
    new THREE.Vector2(radiusTop * 0.96, height * 0.96),
    new THREE.Vector2(rInnerTop, height),
  ];
  const geom = new THREE.LatheGeometry(pts, 28);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Builds the central Bone Calcite & stepped bezel pivot rivet pin spanning the 7-leaf stack.
 */
export function createPivotPinGeometry(spanZ: number): THREE.BufferGeometry {
  const halfLen = spanZ * 0.5 + 0.022;
  const pts: THREE.Vector2[] = [];
  const capSegs = 8;
  for (let i = 0; i <= capSegs; i++) {
    const a = (i / capSegs) * (Math.PI * 0.5);
    const r = Math.sin(a) * 0.026;
    const y = -halfLen - 0.010 - Math.cos(a) * 0.008;
    pts.push(new THREE.Vector2(r, y));
  }
  pts.push(new THREE.Vector2(0.032, -halfLen - 0.004));
  pts.push(new THREE.Vector2(0.033, -halfLen));
  pts.push(new THREE.Vector2(0.031, -halfLen));
  pts.push(new THREE.Vector2(0.031, halfLen));
  pts.push(new THREE.Vector2(0.033, halfLen));
  pts.push(new THREE.Vector2(0.032, halfLen + 0.004));
  for (let i = 0; i <= capSegs; i++) {
    const a = (i / capSegs) * (Math.PI * 0.5);
    const r = Math.cos(a) * 0.026;
    const y = halfLen + 0.010 + Math.sin(a) * 0.008;
    pts.push(new THREE.Vector2(r, y));
  }

  const geom = new THREE.LatheGeometry(pts, 28);
  geom.rotateX(Math.PI * 0.5);
  geom.computeVertexNormals();
  return geom;
}
