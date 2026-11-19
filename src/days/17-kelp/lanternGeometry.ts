import * as THREE from 'three/webgpu';
import { DEFAULT_LANTERN_CONFIG, type LanternConfig } from './types';

/**
 * Deterministic 1D/2D smooth hash noise for organic deckled kelp margins.
 * Uses irrational frequency ratios (no polar star harmonics).
 */
function hash2(x: number, y: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return h - Math.floor(h);
}

function smoothNoise2(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);

  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);

  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function organicFbm(x: number, y: number): number {
  let sum = 0;
  let amp = 0.54;
  let freq = 1.0;
  for (let i = 0; i < 3; i++) {
    sum += (smoothNoise2(x * freq, y * freq) - 0.5) * 2.0 * amp;
    freq *= 2.13;
    amp *= 0.48;
  }
  return sum;
}

export interface ProfilePoint {
  r: number;
  y: number;
  drDv: number;
  dyDv: number;
}

/**
 * Canonical bowed pendant-lantern meridian profile for v in [-0.04, 1.06].
 * v = 0 is the top brass collar; v = 1 is the bottom kelp hem.
 */
export function sampleLanternProfile(
  v: number,
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): ProfilePoint {
  const t = Math.max(-0.04, Math.min(1.08, v));
  const halfH = config.halfHeight;
  const maxR = config.maxRadius;

  // Smooth gourd/globe meridian: narrow neck at top, broad belly around v=0.47, tapered open bottom
  const angle = t * Math.PI * 0.86 + 0.16;
  const baseSin = Math.sin(angle);
  const belly = Math.pow(Math.max(0.001, baseSin), 0.78);
  const neckFlare = 0.21 * Math.pow(1 - Math.min(1, Math.max(0, t)), 2.2);
  const bottomOpen = 0.22 * Math.pow(Math.max(0, t), 2.4);
  const r = maxR * (0.04 + 0.96 * belly) + neckFlare * 0.22 + bottomOpen * 0.18;

  // Vertical coordinate with gentle shoulder rounding
  const y = halfH * (1.0 - 2.0 * t) + 0.05 * Math.sin(t * Math.PI);

  // Finite-difference tangent for smooth Frenet frames
  const eps = 0.002;
  const tNext = Math.min(1.08, t + eps);
  const angleN = tNext * Math.PI * 0.86 + 0.16;
  const bellyN = Math.pow(Math.max(0.001, Math.sin(angleN)), 0.78);
  const neckFlareN = 0.21 * Math.pow(1 - Math.min(1, Math.max(0, tNext)), 2.2);
  const bottomOpenN = 0.22 * Math.pow(Math.max(0, tNext), 2.4);
  const rN = maxR * (0.04 + 0.96 * bellyN) + neckFlareN * 0.22 + bottomOpenN * 0.18;
  const yN = halfH * (1.0 - 2.0 * tNext) + 0.05 * Math.sin(tNext * Math.PI);

  return {
    r,
    y,
    drDv: (rN - r) / eps,
    dyDv: (yN - y) / eps,
  };
}

/**
 * Computes the organic bottom hem cutoff vMax for a bay at signedU in [-1, 1].
 * Matches the scalloped / deckled kelp hem that arches slightly higher at the stitched seam
 * and dips lower near the walnut ribs.
 */
function sampleDeckledBottomV(bayIdx: number, signedU: number): number {
  const absU = Math.abs(signedU);
  // Arch upward slightly near the center laced seam (|u| -> 0), dip down near ribs (|u| -> 1)
  const seamArch = 0.915 + 0.085 * Math.pow(absU, 1.35);
  const wobble = organicFbm(signedU * 3.4 + bayIdx * 7.13, 2.7 + bayIdx * 1.91) * 0.042;
  return Math.max(0.84, Math.min(1.015, seamArch + wobble));
}

/**
 * Computes the half-width of the central laced seam gap (in normalized bay coordinates [0..1]).
 */
function sampleSeamGapU(v: number): number {
  // Narrower near top collar, slightly wider through the glowing belly
  const belly = Math.sin(Math.min(1, Math.max(0, v)) * Math.PI);
  return 0.075 + 0.052 * belly;
}

/**
 * Helper to merge an array of indexed BufferGeometries that share identical attributes.
 */
function mergeIndexedGeometries(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry {
  let totalVerts = 0;
  let totalIndices = 0;
  for (const g of geoms) {
    totalVerts += g.attributes.position.count;
    totalIndices += g.index ? g.index.count : g.attributes.position.count;
  }

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const hasBayMeta = Boolean(geoms[0]?.attributes.aBayMeta);
  const bayMeta = hasBayMeta ? new Float32Array(totalVerts * 4) : null;
  const indices = new Uint32Array(totalIndices);

  let vOffset = 0;
  let iOffset = 0;

  for (const g of geoms) {
    const pos = g.attributes.position;
    const norm = g.attributes.normal;
    const uv = g.attributes.uv;
    const meta = g.attributes.aBayMeta;
    const count = pos.count;

    positions.set(pos.array as Float32Array, vOffset * 3);
    if (norm) normals.set(norm.array as Float32Array, vOffset * 3);
    if (uv) uvs.set(uv.array as Float32Array, vOffset * 2);
    if (bayMeta && meta) bayMeta.set(meta.array as Float32Array, vOffset * 4);

    if (g.index) {
      const idxArr = g.index.array;
      for (let i = 0; i < idxArr.length; i++) {
        indices[iOffset + i] = idxArr[i] + vOffset;
      }
      iOffset += idxArr.length;
    } else {
      for (let i = 0; i < count; i++) {
        indices[iOffset + i] = vOffset + i;
      }
      iOffset += count;
    }

    vOffset += count;
    g.dispose();
  }

  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  merged.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  if (bayMeta) {
    merged.setAttribute('aBayMeta', new THREE.BufferAttribute(bayMeta, 4));
  }
  merged.setIndex(new THREE.BufferAttribute(indices, 1));
  return merged;
}

/**
 * Builds the 6 curved, chamfered bent dark-walnut meridian ribs.
 * Ribs sit at 30°, 90°, 150°, 210°, 270°, 330° so Bay 0 faces +Z straight at the camera.
 */
export function buildWalnutRibsGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const bayAngle = (Math.PI * 2) / bayCount;
  const segments = 44;

  // Chamfered rectangular bentwood cross-section (8 vertices around perimeter)
  const halfW = 0.029;
  const halfD = 0.022;
  const ch = 0.006;
  const profile2D: Array<[number, number]> = [
    [-halfW + ch, -halfD],
    [halfW - ch, -halfD],
    [halfW, -halfD + ch],
    [halfW, halfD - ch],
    [halfW - ch, halfD],
    [-halfW + ch, halfD],
    [-halfW, halfD - ch],
    [-halfW, -halfD + ch],
  ];
  const ringCount = profile2D.length;
  const geoms: THREE.BufferGeometry[] = [];

  for (let k = 0; k < bayCount; k++) {
    // Offset by half-bay so Bay 0 is centered on theta = 0 (+Z)
    const theta = bayAngle * (k + 0.5);
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);

    const vertCount = (segments + 1) * ringCount;
    const positions = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices: number[] = [];

    for (let i = 0; i <= segments; i++) {
      const s = i / segments;
      const v = -0.02 + s * 1.065; // Tucked into top brass collar, slightly protruding at bottom
      const p = sampleLanternProfile(v, config);

      // Gentle taper at bottom chisel tips and top collar entry
      const topTaper = s < 0.08 ? 0.78 + (s / 0.08) * 0.22 : 1.0;
      const tipTaper = s > 0.9 ? 1.0 - ((s - 0.9) / 0.1) * 0.42 : 1.0;
      const scaleW = topTaper * tipTaper;
      const scaleD = topTaper * (s > 0.92 ? 1.0 - ((s - 0.92) / 0.08) * 0.35 : 1.0);

      // Local orthonormal frame along the meridian
      // Outward normal in (r, y) plane: perpendicular to (drDv, dyDv)
      const tLen = Math.hypot(p.drDv, p.dyDv) || 1;
      const tr = p.drDv / tLen;
      const ty = p.dyDv / tLen;
      // Since v increases downward (ty < 0), outward normal is (-ty, tr)
      const nr = -ty;
      const ny = tr;

      // Tangent along azimuth (+theta)
      const tx = cosT;
      const tz = -sinT;

      // Center of rib slightly proud of kelp skin
      const cx = (p.r + 0.012) * sinT;
      const cy = p.y;
      const cz = (p.r + 0.012) * cosT;

      for (let j = 0; j < ringCount; j++) {
        const [lx, ly] = profile2D[j];
        const wOff = lx * scaleW;
        const dOff = ly * scaleD;

        const px = cx + tx * wOff + nr * sinT * dOff;
        const py = cy + ny * dOff;
        const pz = cz + tz * wOff + nr * cosT * dOff;

        const idx = i * ringCount + j;
        positions[idx * 3] = px;
        positions[idx * 3 + 1] = py;
        positions[idx * 3 + 2] = pz;
        uvs[idx * 2] = j / ringCount;
        uvs[idx * 2 + 1] = s * 3.0;
      }

      if (i < segments) {
        for (let j = 0; j < ringCount; j++) {
          const jNext = (j + 1) % ringCount;
          const a = i * ringCount + j;
          const b = i * ringCount + jNext;
          const c = (i + 1) * ringCount + jNext;
          const d = (i + 1) * ringCount + j;
          indices.push(a, d, b, b, d, c);
        }
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setIndex(indices);
    g.computeVertexNormals();
    geoms.push(g);
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the 6 stretched cured kelp-leather parchment bays (each split into left & right
 * laced half-sheets meeting across the central whip-stitched seam).
 */
export function buildKelpPanelsGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const bayAngle = (Math.PI * 2) / bayCount;
  const halfBayAngle = bayAngle * 0.5;
  const uSegs = 18;
  const vSegs = 42;
  const geoms: THREE.BufferGeometry[] = [];

  for (let bay = 0; bay < bayCount; bay++) {
    const bayCenterTheta = bay * bayAngle;

    for (const side of [-1, 1] as const) {
      const vertCount = (uSegs + 1) * (vSegs + 1);
      const positions = new Float32Array(vertCount * 3);
      const uvs = new Float32Array(vertCount * 2);
      const bayMeta = new Float32Array(vertCount * 4);
      const indices: number[] = [];

      for (let iv = 0; iv <= vSegs; iv++) {
        const vNorm = iv / vSegs;

        for (let iu = 0; iu <= uSegs; iu++) {
          const uFrac = iu / uSegs; // 0 at central laced seam, 1 at walnut rib
          const gapU = sampleSeamGapU(vNorm);
          // Slight natural waviness along the inner laced edge
          const seamWobble = organicFbm(vNorm * 5.2 + bay * 3.7, side * 2.1) * 0.018;
          const absU = gapU + seamWobble * (1 - uFrac) + uFrac * (1.0 - gapU);
          const signedU = side * Math.min(1.0, Math.max(0.02, absU));

          const vMax = sampleDeckledBottomV(bay, signedU);
          const v = 0.015 + vNorm * (vMax - 0.015);
          const prof = sampleLanternProfile(v, config);

          // Stretched taut catenary saddle between ribs + organic dried kelp undulation
          const ribTension = 1.0 - Math.pow(Math.abs(signedU), 1.8);
          const catenaryInset = -0.042 * ribTension * Math.sin(vNorm * Math.PI);
          const organicRipple =
            organicFbm(signedU * 2.8 + bay * 4.3, vNorm * 3.6 + side * 1.3) *
            0.022 *
            (0.25 + 0.75 * ribTension);

          // Slight lip curl right at the deckled bottom edge
          const bottomCurl = Math.pow(Math.max(0, vNorm - 0.82) / 0.18, 2.0) * 0.028;

          const radius = prof.r - 0.008 + catenaryInset + organicRipple + bottomCurl;
          const theta = bayCenterTheta + signedU * halfBayAngle * 0.965;

          const x = radius * Math.sin(theta);
          const y = prof.y;
          const z = radius * Math.cos(theta);

          // Physical parchment thickness: thin & glowing in mid-panel, thicker at ribs, seam & bottom hem
          const ribMargin = Math.pow(Math.max(0, uFrac - 0.68) / 0.32, 1.6);
          const seamMargin = Math.pow(Math.max(0, 0.16 - uFrac) / 0.16, 1.5);
          const hemMargin = Math.pow(Math.max(0, vNorm - 0.84) / 0.16, 1.6);
          const topMargin = Math.pow(Math.max(0, 0.14 - vNorm) / 0.14, 1.5);
          const thickness =
            0.3 +
            ribMargin * 0.58 +
            seamMargin * 0.24 +
            hemMargin * 0.45 +
            topMargin * 0.35 +
            Math.max(0, organicFbm(signedU * 4.1 + bay * 2.1, vNorm * 4.7)) * 0.16;

          const idx = iv * (uSegs + 1) + iu;
          positions[idx * 3] = x;
          positions[idx * 3 + 1] = y;
          positions[idx * 3 + 2] = z;

          // UVs: u goes 0..1 from seam to rib, v goes 0..1 top to bottom
          uvs[idx * 2] = side > 0 ? 0.5 + uFrac * 0.5 : 0.5 - uFrac * 0.5;
          uvs[idx * 2 + 1] = vNorm;

          bayMeta[idx * 4] = signedU;
          bayMeta[idx * 4 + 1] = vNorm;
          bayMeta[idx * 4 + 2] = bay / bayCount;
          bayMeta[idx * 4 + 3] = thickness;
        }
      }

      for (let iv = 0; iv < vSegs; iv++) {
        for (let iu = 0; iu < uSegs; iu++) {
          const a = iv * (uSegs + 1) + iu;
          const b = a + 1;
          const c = (iv + 1) * (uSegs + 1) + iu + 1;
          const d = (iv + 1) * (uSegs + 1) + iu;
          if (side > 0) {
            indices.push(a, d, b, b, d, c);
          } else {
            indices.push(a, b, d, b, c, d);
          }
        }
      }

      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
      g.setAttribute('aBayMeta', new THREE.BufferAttribute(bayMeta, 4));
      g.setIndex(indices);
      g.computeVertexNormals();
      geoms.push(g);
    }
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the 6 dark, cured, ruffled kelp-leather outer flanges protruding along the
 * exterior of each walnut rib (framing the lantern silhouette like the reference).
 */
export function buildKelpFrillsGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const bayAngle = (Math.PI * 2) / bayCount;
  const vSegs = 96;
  const rSegs = 6;
  const geoms: THREE.BufferGeometry[] = [];

  for (let k = 0; k < bayCount; k++) {
    const theta = bayAngle * (k + 0.5);
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    const tx = cosT;
    const tz = -sinT;

    const vertCount = (vSegs + 1) * (rSegs + 1);
    const positions = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices: number[] = [];

    for (let iv = 0; iv <= vSegs; iv++) {
      const vNorm = iv / vSegs;
      const v = 0.03 + vNorm * 0.985;
      const p = sampleLanternProfile(v, config);

      // Taper frill width at top collar and let it flare organically with crisp dried-kelp deckling
      const envelope = Math.sin(Math.pow(vNorm, 0.82) * Math.PI * 0.94);
      const deckle =
        0.54 +
        organicFbm(vNorm * 11.5 + k * 5.1, k * 1.7 + 0.5) * 0.56 +
        organicFbm(vNorm * 24.0 + k * 2.9, 1.3 + k) * 0.24;
      const maxFrillOut = Math.max(0.015, (0.11 * envelope + 0.038) * Math.max(0.22, deckle));

      // Crinkled dried-seaweed lateral curl
      const lateralWave =
        organicFbm(vNorm * 12.5 + k * 3.3, 4.2 + k) * 0.088 +
        organicFbm(vNorm * 26.0 + k * 1.9, 8.1) * 0.032;

      for (let ir = 0; ir <= rSegs; ir++) {
        const u = ir / rSegs; // 0 attached behind walnut rib, 1 jagged outer edge
        const rOut = p.r + 0.006 + u * maxFrillOut;
        const sideBend = Math.pow(u, 1.35) * lateralWave;
        const yJitter = Math.pow(u, 1.4) * organicFbm(vNorm * 13.5 + k * 2.2, u * 2.1) * 0.036;

        const px = rOut * sinT + tx * sideBend;
        const py = p.y + yJitter;
        const pz = rOut * cosT + tz * sideBend;

        const idx = iv * (rSegs + 1) + ir;
        positions[idx * 3] = px;
        positions[idx * 3 + 1] = py;
        positions[idx * 3 + 2] = pz;
        uvs[idx * 2] = u;
        uvs[idx * 2 + 1] = vNorm;
      }

      if (iv < vSegs) {
        for (let ir = 0; ir < rSegs; ir++) {
          const a = iv * (rSegs + 1) + ir;
          const b = a + 1;
          const c = (iv + 1) * (rSegs + 1) + ir + 1;
          const d = (iv + 1) * (rSegs + 1) + ir;
          indices.push(a, d, b, b, d, c);
        }
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setIndex(indices);
    g.computeVertexNormals();
    geoms.push(g);
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Helper to append a thin 3D cylindrical segment between two points into an array of geometries.
 */
function createStrutSegment(
  pA: THREE.Vector3,
  pB: THREE.Vector3,
  radius: number,
  radialSegments = 5
): THREE.BufferGeometry {
  const dir = new THREE.Vector3().subVectors(pB, pA);
  const len = dir.length();
  const geom = new THREE.CylinderGeometry(radius, radius, Math.max(0.001, len), radialSegments, 1, false);
  const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize()
  );
  const mat = new THREE.Matrix4().compose(mid, quat, new THREE.Vector3(1, 1, 1));
  geom.applyMatrix4(mat);
  return geom;
}

/**
 * Builds the 3D cross-laced whip-stitching & seam edge-cords running down the
 * vertical center of all 6 kelp bays.
 */
export function buildLacingGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const bayAngle = (Math.PI * 2) / bayCount;
  const halfBayAngle = bayAngle * 0.5;
    const stitchPairs = 34;
    const geoms: THREE.BufferGeometry[] = [];

  const leftPts: THREE.Vector3[] = [];
  const rightPts: THREE.Vector3[] = [];

  for (let bay = 0; bay < bayCount; bay++) {
    const bayCenterTheta = bay * bayAngle;
    leftPts.length = 0;
    rightPts.length = 0;

    const vBottom = Math.min(sampleDeckledBottomV(bay, -0.08), sampleDeckledBottomV(bay, 0.08)) - 0.015;

    for (let i = 0; i <= stitchPairs; i++) {
      const f = i / stitchPairs;
      const v = 0.03 + f * (vBottom - 0.03);
      const prof = sampleLanternProfile(v, config);
      const gapU = sampleSeamGapU(v) + 0.018; // Anchor right inside the kelp parchment edge
      const catenaryInset = -0.042 * Math.sin(f * Math.PI);
      const r = prof.r - 0.002 + catenaryInset;

      const thetaL = bayCenterTheta - gapU * halfBayAngle * 0.965;
      const thetaR = bayCenterTheta + gapU * halfBayAngle * 0.965;

      leftPts.push(new THREE.Vector3(r * Math.sin(thetaL), prof.y, r * Math.cos(thetaL)));
      rightPts.push(new THREE.Vector3(r * Math.sin(thetaR), prof.y, r * Math.cos(thetaR)));
    }

    for (let i = 0; i < stitchPairs; i++) {
      const l0 = leftPts[i];
      const l1 = leftPts[i + 1];
      const r0 = rightPts[i];
      const r1 = rightPts[i + 1];

      // Delicate handcrafted crisscross whip-stitching (no rigid ladder rungs)
      geoms.push(createStrutSegment(l0, r1, 0.0035, 5));
      geoms.push(createStrutSegment(r0, l1, 0.0035, 5));

      // Fine edge-reinforcement twine along left & right seam margins
      geoms.push(createStrutSegment(l0, l1, 0.0044, 5));
      geoms.push(createStrutSegment(r0, r1, 0.0044, 5));
    }
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the aged-brass lamp socket, knurled grip collar, mounting flange, and 6 rib rivets.
 */
export function buildBrassHardwareGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const topY = config.halfHeight;
  const geoms: THREE.BufferGeometry[] = [];

  // Lathe profile for the antique-brass lamp holder & stepped socket
  const lathePts: THREE.Vector2[] = [
    new THREE.Vector2(0.032, topY + 0.56), // Cord grip ferrule top
    new THREE.Vector2(0.048, topY + 0.54),
    new THREE.Vector2(0.056, topY + 0.42),
    new THREE.Vector2(0.095, topY + 0.38), // Upper cap dome
    new THREE.Vector2(0.135, topY + 0.31),
    new THREE.Vector2(0.145, topY + 0.25),
    new THREE.Vector2(0.162, topY + 0.23), // Knurled collar lip
    new THREE.Vector2(0.162, topY + 0.14),
    new THREE.Vector2(0.142, topY + 0.12),
    new THREE.Vector2(0.15, topY + 0.04),
    new THREE.Vector2(0.285, topY + 0.025), // Wide flange clamping walnut rib tops
    new THREE.Vector2(0.285, topY - 0.015),
    new THREE.Vector2(0.14, topY - 0.025),
    new THREE.Vector2(0.12, topY - 0.14), // Inner Edison socket sleeve
    new THREE.Vector2(0.095, topY - 0.14),
  ];
  const socketLathe = new THREE.LatheGeometry(lathePts, 40);
  geoms.push(socketLathe);

  // Knurled grip ridges around the socket band
  const knurlCount = 32;
  for (let i = 0; i < knurlCount; i++) {
    const a = (i / knurlCount) * Math.PI * 2;
    const ridge = new THREE.BoxGeometry(0.008, 0.075, 0.012);
    ridge.rotateY(-a);
    ridge.translate(Math.sin(a) * 0.163, topY + 0.185, Math.cos(a) * 0.163);
    geoms.push(ridge);
  }

  // 6 brass rivet pins at the top flange + 6 small brass end-pins near the bottom rib tips
  const bayAngle = (Math.PI * 2) / config.bayCount;
  const bottomTip = sampleLanternProfile(1.01, config);
  for (let k = 0; k < config.bayCount; k++) {
    const theta = bayAngle * (k + 0.5);
    const rivetTop = new THREE.SphereGeometry(0.015, 8, 6);
    rivetTop.translate(Math.sin(theta) * 0.255, topY + 0.028, Math.cos(theta) * 0.255);
    geoms.push(rivetTop);

    const rivetBot = new THREE.SphereGeometry(0.011, 8, 6);
    rivetBot.translate(
      Math.sin(theta) * (bottomTip.r + 0.015),
      bottomTip.y,
      Math.cos(theta) * (bottomTip.r + 0.015)
    );
    geoms.push(rivetBot);
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the internal tubular vintage Edison bulb glass envelope.
 */
export function buildBulbGlassGeometry(): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.001, -0.34),
    new THREE.Vector2(0.055, -0.33),
    new THREE.Vector2(0.098, -0.295),
    new THREE.Vector2(0.118, -0.23),
    new THREE.Vector2(0.122, 0.26),
    new THREE.Vector2(0.105, 0.36),
    new THREE.Vector2(0.088, 0.44),
  ];
  return new THREE.LatheGeometry(pts, 28);
}

/**
 * Builds the internal glowing squirrel-cage Edison tungsten filament core.
 */
export function buildFilamentGeometry(): THREE.BufferGeometry {
  const geoms: THREE.BufferGeometry[] = [];
  const strandCount = 6;
  const rTop = 0.026;
  const rBot = 0.044;

  for (let i = 0; i < strandCount; i++) {
    const a0 = (i / strandCount) * Math.PI * 2;
    const a1 = ((i + 0.5) / strandCount) * Math.PI * 2;
    const pTop = new THREE.Vector3(Math.sin(a0) * rTop, 0.24, Math.cos(a0) * rTop);
    const pMid = new THREE.Vector3(Math.sin(a1) * rBot, -0.02, Math.cos(a1) * rBot);
    const pBot = new THREE.Vector3(Math.sin(a0) * (rBot * 0.72), -0.21, Math.cos(a0) * (rBot * 0.72));
    geoms.push(createStrutSegment(pTop, pMid, 0.0075, 5));
    geoms.push(createStrutSegment(pMid, pBot, 0.0075, 5));
  }

  // Glowing inner core rod for warm volumetric readability through the kelp skin
  const core = new THREE.CylinderGeometry(0.022, 0.028, 0.44, 12);
  core.translate(0, 0.02, 0);
  geoms.push(core);

  return mergeIndexedGeometries(geoms);
}

/**
 * Creates a pre-allocated dynamic braided hemp cord geometry that bends cleanly
 * from the ceiling anchor to the swaying brass socket with zero per-frame allocations.
 */
export interface DynamicHempCord {
  geometry: THREE.BufferGeometry;
  update: (anchorY: number, socketWorldPos: THREE.Vector3) => void;
}

export function createDynamicHempCord(segments = 28, radialSegments = 8): DynamicHempCord {
  const vertCount = (segments + 1) * (radialSegments + 1);
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  for (let i = 0; i <= segments; i++) {
    const v = i / segments;
    for (let j = 0; j <= radialSegments; j++) {
      const u = j / radialSegments;
      const idx = i * (radialSegments + 1) + j;
      uvs[idx * 2] = u * 2.0;
      uvs[idx * 2 + 1] = v * 16.0;
    }
    if (i < segments) {
      for (let j = 0; j < radialSegments; j++) {
        const a = i * (radialSegments + 1) + j;
        const b = a + 1;
        const c = (i + 1) * (radialSegments + 1) + j + 1;
        const d = (i + 1) * (radialSegments + 1) + j;
        indices.push(a, d, b, b, d, c);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  const normAttr = new THREE.BufferAttribute(normals, 3);
  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('normal', normAttr);
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);

  const baseRadius = 0.022;

  function update(anchorY: number, socketWorldPos: THREE.Vector3): void {
    const sx = socketWorldPos.x;
    const sy = socketWorldPos.y;
    const sz = socketWorldPos.z;

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      // Subtle quadratic ease so the cord exits the ceiling anchor vertically and curves into the swinging socket
      const curveT = t * t * (0.25 + 0.75 * t);
      const cx = sx * curveT;
      const cy = anchorY + (sy - anchorY) * t;
      const cz = sz * curveT;

      // Braided 2-ply twist perturbation
      const twistAngle = t * 44.0;

      for (let j = 0; j <= radialSegments; j++) {
        const phi = (j / radialSegments) * Math.PI * 2;
        const braidBump = 1.0 + 0.14 * Math.sin(phi * 2.0 + twistAngle);
        const r = baseRadius * braidBump;

        const nx = Math.cos(phi);
        const nz = Math.sin(phi);

        const idx = i * (radialSegments + 1) + j;
        positions[idx * 3] = cx + nx * r;
        positions[idx * 3 + 1] = cy;
        positions[idx * 3 + 2] = cz + nz * r;

        normals[idx * 3] = nx;
        normals[idx * 3 + 1] = 0;
        normals[idx * 3 + 2] = nz;
      }
    }

    posAttr.needsUpdate = true;
    normAttr.needsUpdate = true;
    geometry.computeBoundingSphere();
  }

  return { geometry, update };
}
