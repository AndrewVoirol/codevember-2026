import * as THREE from 'three/webgpu';
import { DEFAULT_LANTERN_CONFIG, type LanternConfig } from './types';

/**
 * Deterministic 1D/2D smooth hash noise for organic deckled kelp margins.
 * Uses irrational frequency ratios (zero polar star harmonics).
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

function evalRawProfile(t: number, config: LanternConfig): { r: number; y: number } {
  const halfH = config.halfHeight;
  const maxR = config.maxRadius;

  // Full, sculptural gourd/globe lantern meridian matching the locked Concept F silhouette:
  // tucked neatly under the top brass flange (r ~ 0.23 at v=0), broad rounded shoulders,
  // generous equatorial belly around v=0.47, graceful open bottom hem (r ~ 0.43 at v=1).
  const angle = t * Math.PI * 0.855 + 0.155;
  const baseSin = Math.sin(angle);
  const belly = Math.pow(Math.max(0.001, baseSin), 0.72);
  const neckTuck = 0.045 * Math.pow(1 - Math.min(1, Math.max(0, t)), 2.5);
  const bottomFlare = 0.036 * Math.pow(Math.max(0, t), 2.3);
  const r = maxR * (0.045 + 0.955 * belly) + neckTuck + bottomFlare;

  // Vertical coordinate with gentle rounded upper shoulder
  const y = halfH * (1.0 - 2.0 * t) + 0.065 * Math.sin(t * Math.PI);
  return { r, y };
}

/**
 * Canonical bowed pendant-lantern meridian profile for v in [-0.04, 1.06].
 * v = 0 is the top brass clamping collar; v = 1 is the bottom kelp hem.
 */
export function sampleLanternProfile(
  v: number,
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): ProfilePoint {
  const t = Math.max(-0.04, Math.min(1.08, v));
  const cur = evalRawProfile(t, config);

  const eps = 0.002;
  const tPrev = Math.max(-0.04, t - eps);
  const tNext = Math.min(1.08, t + eps);
  const prev = evalRawProfile(tPrev, config);
  const next = evalRawProfile(tNext, config);
  const span = Math.max(1e-5, tNext - tPrev);

  return {
    r: cur.r,
    y: cur.y,
    drDv: (next.r - prev.r) / span,
    dyDv: (next.y - prev.y) / span,
  };
}

/**
 * Builds a cumulative meridian arc-length lookup table for uniform, stretch-free V-mapping.
 */
function buildArcLengthLut(config: LanternConfig, steps = 128): Float32Array {
  const lut = new Float32Array(steps + 1);
  let prev = evalRawProfile(0, config);
  let total = 0;
  for (let i = 1; i <= steps; i++) {
    const v = (i / steps) * 1.04;
    const cur = evalRawProfile(v, config);
    total += Math.hypot(cur.r - prev.r, cur.y - prev.y);
    lut[i] = total;
    prev = cur;
  }
  const invTotal = total > 1e-6 ? 1 / total : 1;
  for (let i = 0; i <= steps; i++) {
    lut[i] *= invTotal;
  }
  return lut;
}

function sampleArcFrac(v: number, lut: Float32Array): number {
  const steps = lut.length - 1;
  const f = Math.max(0, Math.min(1, v / 1.04)) * steps;
  const idx = Math.min(steps - 1, Math.floor(f));
  const frac = f - idx;
  return lut[idx] + (lut[idx + 1] - lut[idx]) * frac;
}

/**
 * Computes the organic bottom hem cutoff vMax for a bay at signedU in [-1, 1].
 * Gently arches upward at the laced seam (|u| -> 0) and sweeps lower toward the walnut ribs (|u| -> 1).
 */
export function sampleDeckledBottomV(bayIdx: number, signedU: number): number {
  const absU = Math.abs(signedU);
  const seamArch = 0.895 + 0.102 * Math.pow(absU, 1.25);
  const coarseDeckle = organicFbm(signedU * 2.9 + bayIdx * 7.13, 2.7 + bayIdx * 1.91) * 0.032;
  const fineDeckle = organicFbm(signedU * 7.8 + bayIdx * 3.41, 6.1 + bayIdx * 2.3) * 0.011;
  return Math.max(0.84, Math.min(1.01, seamArch + coarseDeckle + fineDeckle));
}

/**
 * Computes the half-width of the narrow central laced seam gap (in normalized bay coordinates [0..1]).
 */
export function sampleSeamGapU(v: number): number {
  const belly = Math.sin(Math.min(1, Math.max(0, v)) * Math.PI);
  return 0.044 + 0.028 * belly;
}

/**
 * Shared organic hand-cut wobble along the laced kelp margin.
 * Evaluated identically by both kelp panels and whip-stitch eyelets so zero gaps or clipping occur.
 */
function sampleSeamEdgeWobble(bay: number, side: -1 | 1, vNorm: number): number {
  return organicFbm(vNorm * 5.6 + bay * 3.71, side * 2.13 + 0.4) * 0.011;
}

const STITCH_PAIRS = 28;

/**
 * Computes subtle stitched eyelet tension puckering right along the laced margin.
 */
function sampleStitchPucker(vNorm: number, uFrac: number): number {
  if (uFrac > 0.24) return 0;
  const edgeFalloff = Math.pow(1 - uFrac / 0.24, 2.0);
  // Subtle pinch wave aligned to the 28 eyelet pairs
  const phase = vNorm * STITCH_PAIRS * Math.PI * 2;
  const pinch = 0.5 - 0.5 * Math.cos(phase);
  return pinch * edgeFalloff * 0.0055;
}

export interface KelpPanelSample {
  x: number;
  y: number;
  z: number;
  signedU: number;
  v: number;
  vNorm: number;
  r: number;
  theta: number;
  prof: ProfilePoint;
  pucker: number;
}

/**
 * Canonical evaluator for any point on a kelp parchment half-sheet.
 * Shared identically by `buildKelpPanelsGeometry` and `buildLacingGeometry` so every stitch
 * eyelet anchors with exact 0.000 mm registration onto the kelp margin.
 */
export function sampleKelpPanelPoint(
  bay: number,
  side: -1 | 1,
  uFrac: number,
  vNorm: number,
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): KelpPanelSample {
  const bayAngle = (Math.PI * 2) / config.bayCount;
  const halfBayAngle = bayAngle * 0.5;
  const bayCenterTheta = bay * bayAngle;

  const gapU = sampleSeamGapU(vNorm);
  const seamWobble = sampleSeamEdgeWobble(bay, side, vNorm);
  const pucker = sampleStitchPucker(vNorm, uFrac);

  // Pull the kelp margin subtly inward toward the seam at each stitch eyelet
  const uEdge = Math.max(0.018, gapU + seamWobble - pucker * 0.75);
  // Extend uFrac = 1 all the way to 1.005 so the kelp sheet tucks cleanly into the walnut rib groove
  const absU = uEdge + uFrac * (1.005 - uEdge);
  const signedU = side * Math.min(1.005, absU);

  const vMax = sampleDeckledBottomV(bay, Math.min(1, Math.abs(signedU)) * side);
  // Start at v = -0.004 so the top edge tucks completely up inside the brass clamping collar (zero top gap)
  const v = -0.004 + vNorm * (vMax + 0.004);
  const prof = sampleLanternProfile(v, config);

  // Stretched taut catenary saddle between ribs + organic dried kelp undulation
  const clampedAbsU = Math.min(1.0, Math.abs(signedU));
  const ribTension = 1.0 - Math.pow(clampedAbsU, 1.75);
  const catenaryInset = -0.038 * ribTension * Math.sin(vNorm * Math.PI);
  const organicRipple =
    organicFbm(signedU * 2.7 + bay * 4.31, vNorm * 3.8 + side * 1.37) *
    0.017 *
    (0.2 + 0.8 * ribTension);

  // Smooth, gradual dried-parchment curl toward the free deckled bottom hem (zero sharp interior crease knee)
  const bottomCurl = Math.pow(Math.max(0, vNorm - 0.72) / 0.28, 2.6) * 0.021;
  // Subtle outward lip ridge right at the punched eyelet row
  const eyeletRidge = pucker * 0.5;

  const r = prof.r - 0.005 + catenaryInset + organicRipple + bottomCurl + eyeletRidge;
  const theta = bayCenterTheta + signedU * halfBayAngle;

  return {
    x: r * Math.sin(theta),
    y: prof.y,
    z: r * Math.cos(theta),
    signedU: Math.max(-1, Math.min(1, signedU)),
    v,
    vNorm,
    r,
    theta,
    prof,
    pucker,
  };
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
  const hasStripBounds = Boolean(geoms[0]?.attributes.aStripBounds);
  const bayMeta = hasBayMeta ? new Float32Array(totalVerts * 4) : null;
  const stripBounds = hasStripBounds ? new Float32Array(totalVerts * 2) : null;
  const indices = new Uint32Array(totalIndices);

  let vOffset = 0;
  let iOffset = 0;

  for (const g of geoms) {
    const pos = g.attributes.position;
    const norm = g.attributes.normal;
    const uv = g.attributes.uv;
    const meta = g.attributes.aBayMeta;
    const bounds = g.attributes.aStripBounds;
    const count = pos.count;

    positions.set(pos.array as Float32Array, vOffset * 3);
    if (norm) normals.set(norm.array as Float32Array, vOffset * 3);
    if (uv) uvs.set(uv.array as Float32Array, vOffset * 2);
    if (bayMeta && meta) bayMeta.set(meta.array as Float32Array, vOffset * 4);
    if (stripBounds && bounds) stripBounds.set(bounds.array as Float32Array, vOffset * 2);

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
  if (stripBounds) {
    merged.setAttribute('aStripBounds', new THREE.BufferAttribute(stripBounds, 2));
  }
  merged.setIndex(new THREE.BufferAttribute(indices, 1));
  return merged;
}

/**
 * Builds the 6 solid, watertight, chamfered bent dark-walnut meridian ribs with strictly
 * outward-facing normals and closed sculpted top/bottom end caps.
 * Ribs sit at 30°, 90°, 150°, 210°, 270°, 330° so Bay 0 faces +Z directly at the viewer.
 */
export function buildWalnutRibsGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const bayAngle = (Math.PI * 2) / bayCount;
  const segments = 52;
  const arcLut = buildArcLengthLut(config);

  // Chamfered rectangular bentwood stave cross-section (counterclockwise in (T_theta, R_out))
  const halfW = 0.0255;
  const halfD = 0.027;
  const ch = 0.0068;
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
  const ringStride = ringCount + 1; // Duplicate seam vertex for clean continuous UVs
  const geoms: THREE.BufferGeometry[] = [];

  for (let k = 0; k < bayCount; k++) {
    const theta = bayAngle * (k + 0.5);
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    const tx = cosT;
    const tz = -sinT;

    const sideVertCount = (segments + 1) * ringStride;
    // Two flat-normal end caps (center + ringCount perimeter vertices each)
    const capVertCount = 2 * (ringCount + 1);
    const totalVerts = sideVertCount + capVertCount;

    const positions = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const indices: number[] = [];

    const topRingPos: Array<[number, number, number]> = [];
    const botRingPos: Array<[number, number, number]> = [];
    let topCenter: [number, number, number] = [0, 0, 0];
    let botCenter: [number, number, number] = [0, 0, 0];

    for (let i = 0; i <= segments; i++) {
      const s = i / segments;
      // Tuck smoothly under top brass flange (v = -0.005) and extend cleanly past bottom kelp hem (v = 1.032)
      const v = -0.005 + s * 1.037;
      const p = sampleLanternProfile(v, config);
      const arcV = sampleArcFrac(Math.max(0, v), arcLut);

      // Taper neatly where rib enters the top brass clamp and into a hand-chiseled bottom finial
      const topTaperW = s < 0.06 ? 0.72 + (s / 0.06) * 0.28 : 1.0;
      const topTaperD = s < 0.06 ? 0.62 + (s / 0.06) * 0.38 : 1.0;
      const tipTaperW = s > 0.89 ? 1.0 - ((s - 0.89) / 0.11) * 0.56 : 1.0;
      const tipTaperD = s > 0.91 ? 1.0 - ((s - 0.91) / 0.09) * 0.48 : 1.0;
      const scaleW = topTaperW * tipTaperW;
      const scaleD = topTaperD * tipTaperD;

      // Smoothly pull the upper tip slightly inward so it seats completely inside the brass clamping collar
      const topSeatInset = s < 0.05 ? (1 - s / 0.05) * 0.018 : 0.0;

      const tLen = Math.hypot(p.drDv, p.dyDv) || 1;
      const tr = p.drDv / tLen;
      const ty = p.dyDv / tLen;
      // Outward normal in (r, y) plane perpendicular to downward tangent (tr, ty)
      const nr = -ty;
      const ny = tr;

      // Proud exterior wood stave that also overlaps the kelp sheet edge on the interior
      const cx = (p.r + 0.008 - topSeatInset) * sinT;
      const cy = p.y;
      const cz = (p.r + 0.008 - topSeatInset) * cosT;

      if (i === 0) topCenter = [cx, cy, cz];
      if (i === segments) botCenter = [cx, cy, cz];

      for (let j = 0; j <= ringCount; j++) {
        const [lx, ly] = profile2D[j % ringCount];
        const wOff = lx * scaleW;
        const dOff = ly * scaleD;

        const px = cx + tx * wOff + nr * sinT * dOff;
        const py = cy + ny * dOff;
        const pz = cz + tz * wOff + nr * cosT * dOff;

        const idx = i * ringStride + j;
        positions[idx * 3] = px;
        positions[idx * 3 + 1] = py;
        positions[idx * 3 + 2] = pz;
        // Longitudinal grain UVs: uv.x around stave perimeter, uv.y = meridian arc length
        uvs[idx * 2] = j / ringCount;
        uvs[idx * 2 + 1] = arcV * 2.6;

        if (j < ringCount) {
          if (i === 0) topRingPos.push([px, py, pz]);
          if (i === segments) botRingPos.push([px, py, pz]);
        }
      }

      if (i < segments) {
        for (let j = 0; j < ringCount; j++) {
          const a = i * ringStride + j;
          const b = i * ringStride + j + 1;
          const c = (i + 1) * ringStride + j + 1;
          const d = (i + 1) * ringStride + j;
          // Outward winding: (b - a) x (d - a) = E_ring x T_down = +outward on all 8 faces
          indices.push(a, b, d, b, c, d);
        }
      }
    }

    // Watertight top end cap (outward normal along -T_down)
    const topCapCenterIdx = sideVertCount;
    positions[topCapCenterIdx * 3] = topCenter[0];
    positions[topCapCenterIdx * 3 + 1] = topCenter[1];
    positions[topCapCenterIdx * 3 + 2] = topCenter[2];
    uvs[topCapCenterIdx * 2] = 0.5;
    uvs[topCapCenterIdx * 2 + 1] = 0.0;
    for (let j = 0; j < ringCount; j++) {
      const vi = topCapCenterIdx + 1 + j;
      positions[vi * 3] = topRingPos[j][0];
      positions[vi * 3 + 1] = topRingPos[j][1];
      positions[vi * 3 + 2] = topRingPos[j][2];
      uvs[vi * 2] = 0.5;
      uvs[vi * 2 + 1] = 0.0;
    }
    for (let j = 0; j < ringCount; j++) {
      const v0 = topCapCenterIdx + 1 + j;
      const v1 = topCapCenterIdx + 1 + ((j + 1) % ringCount);
      indices.push(topCapCenterIdx, v1, v0);
    }

    // Watertight bottom chisel-tip end cap (outward normal along +T_down)
    const botCapCenterIdx = sideVertCount + ringCount + 1;
    positions[botCapCenterIdx * 3] = botCenter[0];
    positions[botCapCenterIdx * 3 + 1] = botCenter[1];
    positions[botCapCenterIdx * 3 + 2] = botCenter[2];
    uvs[botCapCenterIdx * 2] = 0.5;
    uvs[botCapCenterIdx * 2 + 1] = 2.6;
    for (let j = 0; j < ringCount; j++) {
      const vi = botCapCenterIdx + 1 + j;
      positions[vi * 3] = botRingPos[j][0];
      positions[vi * 3 + 1] = botRingPos[j][1];
      positions[vi * 3 + 2] = botRingPos[j][2];
      uvs[vi * 2] = 0.5;
      uvs[vi * 2 + 1] = 2.6;
    }
    for (let j = 0; j < ringCount; j++) {
      const v0 = botCapCenterIdx + 1 + j;
      const v1 = botCapCenterIdx + 1 + ((j + 1) % ringCount);
      indices.push(botCapCenterIdx, v0, v1);
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
 * Strip atlas assignment table across all 6 bays × 2 half-sheets (4 unique 256×1024 botanical strips).
 * Guarantees zero adjacent duplicates and zero texture-wrap seams.
 */
const BAY_STRIP_ASSIGNMENTS: ReadonlyArray<
  readonly [{ strip: number; flipV: boolean }, { strip: number; flipV: boolean }]
> = [
  [
    { strip: 0, flipV: false },
    { strip: 1, flipV: false },
  ],
  [
    { strip: 2, flipV: false },
    { strip: 3, flipV: false },
  ],
  [
    { strip: 1, flipV: true },
    { strip: 0, flipV: true },
  ],
  [
    { strip: 3, flipV: true },
    { strip: 2, flipV: true },
  ],
  [
    { strip: 0, flipV: false },
    { strip: 2, flipV: true },
  ],
  [
    { strip: 3, flipV: false },
    { strip: 1, flipV: true },
  ],
];

/**
 * Builds the 6 stretched cured kelp-leather parchment bays (12 laced half-sheets)
 * with aspect-preserving botanical strip UVs, exact rib/collar seating, and stitch-hole puckering.
 */
export function buildKelpPanelsGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const uSegs = 24;
  const vSegs = 56;
  const arcLut = buildArcLengthLut(config);
  const geoms: THREE.BufferGeometry[] = [];

  for (let bay = 0; bay < bayCount; bay++) {
    const stripPair = BAY_STRIP_ASSIGNMENTS[bay % BAY_STRIP_ASSIGNMENTS.length];

    for (const side of [-1, 1] as const) {
      const stripCfg = side < 0 ? stripPair[0] : stripPair[1];
      const uMin = stripCfg.strip * 0.25 + 0.005;
      const uMax = (stripCfg.strip + 1) * 0.25 - 0.005;
      const uSpan = uMax - uMin;

      const vertCount = (uSegs + 1) * (vSegs + 1);
      const positions = new Float32Array(vertCount * 3);
      const uvs = new Float32Array(vertCount * 2);
      const bayMeta = new Float32Array(vertCount * 4);
      const stripBounds = new Float32Array(vertCount * 2);
      const indices: number[] = [];

      for (let iv = 0; iv <= vSegs; iv++) {
        const vNorm = iv / vSegs;

        for (let iu = 0; iu <= uSegs; iu++) {
          const uFrac = iu / uSegs; // 0 at central laced seam, 1 tucked into walnut rib
          const sample = sampleKelpPanelPoint(bay, side, uFrac, vNorm, config);

          // Physical parchment thickness: thin & glowing in mid-panel, thicker at ribs, seam fold & bottom hem
          const ribMargin = Math.pow(Math.max(0, uFrac - 0.66) / 0.34, 1.55);
          const seamFold = Math.pow(Math.max(0, 0.14 - uFrac) / 0.14, 1.45);
          const hemMargin = Math.pow(Math.max(0, vNorm - 0.85) / 0.15, 1.55);
          const topMargin = Math.pow(Math.max(0, 0.13 - vNorm) / 0.13, 1.45);
          const thickness =
            0.26 +
            ribMargin * 0.62 +
            seamFold * 0.28 +
            sample.pucker * 18.0 +
            hemMargin * 0.46 +
            topMargin * 0.38 +
            Math.max(0, organicFbm(sample.signedU * 4.1 + bay * 2.1, vNorm * 4.7)) * 0.14;

          // Physical aspect-preserving UVs:
          // Scale horizontal strip sample span by local circumference ratio so veins near
          // narrow top collar and bottom hem keep natural organic proportions instead of pinching.
          const widthRatio = Math.max(0.28, Math.min(1.0, sample.prof.r / config.maxRadius));
          const uLocal = uFrac * (0.28 + 0.72 * widthRatio);
          const arcV = sampleArcFrac(sample.v, arcLut);
          const vClamped = 0.01 + arcV * 0.98;
          const vMapped = stripCfg.flipV ? 1.0 - vClamped : vClamped;

          const idx = iv * (uSegs + 1) + iu;
          positions[idx * 3] = sample.x;
          positions[idx * 3 + 1] = sample.y;
          positions[idx * 3 + 2] = sample.z;

          uvs[idx * 2] = uMin + uLocal * uSpan;
          uvs[idx * 2 + 1] = vMapped;

          bayMeta[idx * 4] = sample.signedU;
          bayMeta[idx * 4 + 1] = vNorm;
          bayMeta[idx * 4 + 2] = bay;
          bayMeta[idx * 4 + 3] = thickness;

          stripBounds[idx * 2] = uMin;
          stripBounds[idx * 2 + 1] = uMax;
        }
      }

      for (let iv = 0; iv < vSegs; iv++) {
        for (let iu = 0; iu < uSegs; iu++) {
          const a = iv * (uSegs + 1) + iu;
          const b = a + 1;
          const c = (iv + 1) * (uSegs + 1) + iu + 1;
          const d = (iv + 1) * (uSegs + 1) + iu;
          // Outward-facing normals on both left and right half-sheets
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
      g.setAttribute('aStripBounds', new THREE.BufferAttribute(stripBounds, 2));
      g.setIndex(indices);
      g.computeVertexNormals();
      geoms.push(g);
    }
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the 6 sleek, organically ruffled dark cured-kombu kelp outer margins emerging from
 * the clamped seam behind each bent walnut rib.
 * - Single sculpted selvage per rib (never spills sideways across the glowing center bays).
 * - Tapers cleanly to zero at the top brass flange and bottom kelp hem so the chiseled walnut
 *   rib tips remain sleek and unobstructed.
 * - Frames the outer lantern silhouette with authentic dried-seaweed deckling like Concept F.
 */
export function buildKelpFrillsGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const bayAngle = (Math.PI * 2) / bayCount;
  const vSegs = 96;
  const rSegs = 8;
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
      // Tuck under top collar and taper cleanly into bottom hem above the exposed walnut rib finials
      const v = 0.028 + vNorm * 0.935;
      const p = sampleLanternProfile(v, config);

      const tLen = Math.hypot(p.drDv, p.dyDv) || 1;
      const tr = p.drDv / tLen;
      const ty = p.dyDv / tLen;
      const nr = -ty;
      const ny = tr;

      // Smooth fade at top brass flange and clean taper into the scalloped bottom kelp hem
      const topEnv = Math.min(1, Math.pow(vNorm / 0.13, 1.35));
      const botEnv = Math.min(1, Math.pow(Math.max(0, 1 - vNorm) / 0.16, 1.25));
      const bellyEnv = 0.52 + 0.48 * Math.sin(vNorm * Math.PI);
      const envelope = topEnv * botEnv * bellyEnv;

      const seed = k * 6.37 + 2.4;
      const deckle =
        0.58 +
        organicFbm(vNorm * 8.8 + seed, seed * 0.53) * 0.48 +
        organicFbm(vNorm * 19.5 + seed * 1.4, 2.7 + k) * 0.24;
      const maxFrillOut = (0.105 * envelope + 0.008 * topEnv * botEnv) * Math.max(0.24, deckle);

      // Restrained dried-kombu lateral ripple: stays strictly clear of the proud rib face near u = 0
      // and curls organically out along the free deckled tip (u -> 1)
      const lateralRipple =
        organicFbm(vNorm * 11.2 + seed * 0.85, 4.9 + k) * 0.028 +
        organicFbm(vNorm * 24.0 + seed * 1.35, 8.6) * 0.012;

      for (let ir = 0; ir <= rSegs; ir++) {
        const u = ir / rSegs; // 0 seated cleanly behind the walnut rib spine, 1 outer deckled edge
        // Root tucked at p.r + 0.012 (inside the walnut rib body) so outer wood face stays proud & clean
        const radialOff = 0.012 + Math.pow(u, 0.95) * maxFrillOut;
        const tangentOff = Math.pow(u, 1.65) * lateralRipple;
        const yCrinkle =
          Math.pow(u, 1.45) * organicFbm(vNorm * 15.0 + seed, u * 2.3 + k) * 0.022;

        const px = p.r * sinT + nr * sinT * radialOff + tx * tangentOff;
        const py = p.y + ny * radialOff + yCrinkle;
        const pz = p.r * cosT + nr * cosT * radialOff + tz * tangentOff;

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
          indices.push(a, b, d, b, c, d);
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
 * Helper to create a smooth curved or straight cylindrical twine segment between two points
 * with baked `aBayMeta` so lacing deforms in exact GPU lockstep with the kelp parchment.
 */
function createLaceStrandSegment(
  pA: THREE.Vector3,
  pB: THREE.Vector3,
  radius: number,
  bayMetaA: readonly [number, number, number, number],
  bayMetaB: readonly [number, number, number, number],
  radialSegments = 6
): THREE.BufferGeometry {
  const dir = new THREE.Vector3().subVectors(pB, pA);
  const len = Math.max(0.001, dir.length());
  const geom = new THREE.CylinderGeometry(radius, radius, len, radialSegments, 1, false);
  const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize()
  );
  const mat = new THREE.Matrix4().compose(mid, quat, new THREE.Vector3(1, 1, 1));
  geom.applyMatrix4(mat);

  // Interpolate aBayMeta along cylinder height so vertex displacement matches panel endpoints
  const pos = geom.attributes.position;
  const metaArr = new Float32Array(pos.count * 4);
  const tmp = new THREE.Vector3();
  const ab = new THREE.Vector3().subVectors(pB, pA);
  const abLenSq = Math.max(1e-6, ab.lengthSq());

  for (let i = 0; i < pos.count; i++) {
    tmp.fromBufferAttribute(pos, i).sub(pA);
    const t = Math.max(0, Math.min(1, tmp.dot(ab) / abLenSq));
    metaArr[i * 4] = bayMetaA[0] + (bayMetaB[0] - bayMetaA[0]) * t;
    metaArr[i * 4 + 1] = bayMetaA[1] + (bayMetaB[1] - bayMetaA[1]) * t;
    metaArr[i * 4 + 2] = bayMetaA[2];
    metaArr[i * 4 + 3] = bayMetaA[3] + (bayMetaB[3] - bayMetaA[3]) * t;
  }
  geom.setAttribute('aBayMeta', new THREE.BufferAttribute(metaArr, 4));
  return geom;
}

/**
 * Helper to place a tiny stitched eyelet knot/grommet loop seated directly on the kelp margin.
 */
function createEyeletRing(
  center: THREE.Vector3,
  outwardNormal: THREE.Vector3,
  bayMeta: readonly [number, number, number, number]
): THREE.BufferGeometry {
  const torus = new THREE.TorusGeometry(0.0065, 0.0022, 4, 7);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    outwardNormal.clone().normalize()
  );
  const mat = new THREE.Matrix4().compose(center, quat, new THREE.Vector3(1, 1, 1));
  torus.applyMatrix4(mat);

  const count = torus.attributes.position.count;
  const metaArr = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    metaArr[i * 4] = bayMeta[0];
    metaArr[i * 4 + 1] = bayMeta[1];
    metaArr[i * 4 + 2] = bayMeta[2];
    metaArr[i * 4 + 3] = bayMeta[3];
  }
  torus.setAttribute('aBayMeta', new THREE.BufferAttribute(metaArr, 4));
  return torus;
}

/**
 * Builds the handcrafted corset/whip-stitch cross-lacing and punched margin eyelets down all 6 bays.
 * - Zero rigid parallel vertical steel rails.
 * - Anchors directly into the exact organic deckled kelp margin via `sampleKelpPanelPoint`.
 * - Weaves alternating over/under cross strands so crossing cords never Z-fight.
 * - Carries `aBayMeta` so lacing flexes & ripples in 100% GPU lockstep with the kelp parchment.
 */
export function buildLacingGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const bayCount = config.bayCount;
  const geoms: THREE.BufferGeometry[] = [];

  for (let bay = 0; bay < bayCount; bay++) {
    const leftPts: THREE.Vector3[] = [];
    const rightPts: THREE.Vector3[] = [];
    const leftOut: THREE.Vector3[] = [];
    const rightOut: THREE.Vector3[] = [];
    const leftMeta: Array<[number, number, number, number]> = [];
    const rightMeta: Array<[number, number, number, number]> = [];

    // Sample eyelet anchors along the exact shared kelp margin (uFrac = 0.024 inset inside edge)
    for (let i = 0; i <= STITCH_PAIRS; i++) {
      const f = i / STITCH_PAIRS;
      // Slight organic hand-punched spacing jitter in interior stitches
      const jitter =
        i > 0 && i < STITCH_PAIRS
          ? organicFbm(i * 1.73 + bay * 5.1, bay * 2.3 + 0.7) * 0.0045
          : 0;
      const vNorm = Math.max(0.025, Math.min(0.975, 0.032 + f * 0.935 + jitter));

      const sL = sampleKelpPanelPoint(bay, -1, 0.026, vNorm, config);
      const sR = sampleKelpPanelPoint(bay, 1, 0.026, vNorm, config);

      const nL = new THREE.Vector3(Math.sin(sL.theta), 0, Math.cos(sL.theta)).normalize();
      const nR = new THREE.Vector3(Math.sin(sR.theta), 0, Math.cos(sR.theta)).normalize();

      const pL = new THREE.Vector3(sL.x, sL.y, sL.z).addScaledVector(nL, 0.0022);
      const pR = new THREE.Vector3(sR.x, sR.y, sR.z).addScaledVector(nR, 0.0022);

      leftPts.push(pL);
      rightPts.push(pR);
      leftOut.push(nL);
      rightOut.push(nR);
      leftMeta.push([sL.signedU, vNorm, bay, 0]);
      rightMeta.push([sR.signedU, vNorm, bay, 0]);

      // Small punched twine eyelet loop seated flush on each kelp margin
      geoms.push(createEyeletRing(pL, nL, leftMeta[i]));
      geoms.push(createEyeletRing(pR, nR, rightMeta[i]));
    }

    for (let i = 0; i < STITCH_PAIRS; i++) {
      const l0 = leftPts[i];
      const l1 = leftPts[i + 1];
      const r0 = rightPts[i];
      const r1 = rightPts[i + 1];
      const mL0 = leftMeta[i];
      const mL1 = leftMeta[i + 1];
      const mR0 = rightMeta[i];
      const mR1 = rightMeta[i + 1];

      // Radial midpoint weave offset (+0.0032 over / -0.0032 under) so crossing cords never Z-fight
      const midOut = new THREE.Vector3().addVectors(leftOut[i], rightOut[i + 1]).normalize();
      const midAB = new THREE.Vector3().addVectors(l0, r1).multiplyScalar(0.5);
      const midBA = new THREE.Vector3().addVectors(r0, l1).multiplyScalar(0.5);
      const weaveSign = i % 2 === 0 ? 1 : -1;
      midAB.addScaledVector(midOut, 0.0034 * weaveSign);
      midBA.addScaledVector(midOut, -0.0034 * weaveSign);

      const midMeta: [number, number, number, number] = [
        0,
        (mL0[1] + mR1[1]) * 0.5,
        bay,
        0,
      ];

      // Over/under crisscross waxed hemp whip-stitching (2 segments per diagonal for genuine 3D weave)
      geoms.push(createLaceStrandSegment(l0, midAB, 0.0031, mL0, midMeta, 5));
      geoms.push(createLaceStrandSegment(midAB, r1, 0.0031, midMeta, mR1, 5));
      geoms.push(createLaceStrandSegment(r0, midBA, 0.0031, mR0, midMeta, 5));
      geoms.push(createLaceStrandSegment(midBA, l1, 0.0031, midMeta, mL1, 5));

      // Delicate tucked horizontal tension whip-bar every stitch pair (lets maximum warm glow breathe through)
      const hMid = new THREE.Vector3()
        .addVectors(l0, r0)
        .multiplyScalar(0.5)
        .addScaledVector(midOut, -0.0024);
      const hMeta: [number, number, number, number] = [0, mL0[1], bay, 0];
      geoms.push(createLaceStrandSegment(l0, hMid, 0.0018, mL0, hMeta, 5));
      geoms.push(createLaceStrandSegment(hMid, r0, 0.0018, hMeta, mR0, 5));
    }

    // Bottom horizontal closing knot stitch
    const last = STITCH_PAIRS;
    geoms.push(
      createLaceStrandSegment(leftPts[last], rightPts[last], 0.0034, leftMeta[last], rightMeta[last], 5)
    );
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the solid, strictly outward-facing aged-brass lamp canopy, cord-grip ferrule,
 * knurled collar, walnut clamping flange, interior retaining ring, and 12 flush brass fasteners.
 *
 * CRITICAL: Three.js LatheGeometry requires points ordered BOTTOM-TO-TOP on exterior walls
 * (and top-to-bottom on interior cavity return walls) for outward-facing normals!
 */
export function buildBrassHardwareGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const topY = config.halfHeight;
  const geoms: THREE.BufferGeometry[] = [];

  // Closed, solid-walled turned-brass lamp holder profile:
  // Outer wall traced strictly BOTTOM -> TOP (+R outward normals), then inner cavity TOP -> BOTTOM (-R inward normals).
  const lathePts: THREE.Vector2[] = [
    // Inner socket bottom rim -> outer lower clamping lip (bottom-to-top outward)
    new THREE.Vector2(0.105, topY - 0.145),
    new THREE.Vector2(0.128, topY - 0.145),
    new THREE.Vector2(0.132, topY - 0.042),
    new THREE.Vector2(0.278, topY - 0.032), // Lower lip of rib-clamping collar overhanging kelp top edge
    new THREE.Vector2(0.286, topY - 0.014),
    new THREE.Vector2(0.286, topY + 0.022), // Wide brass clamping flange over walnut rib arches
    new THREE.Vector2(0.268, topY + 0.034),
    new THREE.Vector2(0.156, topY + 0.042), // Stepped collar riser
    new THREE.Vector2(0.152, topY + 0.118),
    new THREE.Vector2(0.166, topY + 0.132), // Knurled ring bottom bevel
    new THREE.Vector2(0.166, topY + 0.228), // Knurled ring cylinder seat
    new THREE.Vector2(0.154, topY + 0.242), // Upper stepped bead
    new THREE.Vector2(0.146, topY + 0.275),
    new THREE.Vector2(0.132, topY + 0.322), // Turned shoulder dome
    new THREE.Vector2(0.092, topY + 0.372),
    new THREE.Vector2(0.064, topY + 0.394), // Cord-grip nut collar
    new THREE.Vector2(0.062, topY + 0.435),
    new THREE.Vector2(0.046, topY + 0.515), // Tapered brass strain-relief ferrule
    new THREE.Vector2(0.036, topY + 0.532),
    new THREE.Vector2(0.024, topY + 0.532), // Inner cord bore rim (top -> bottom cavity return)
    new THREE.Vector2(0.024, topY + 0.38),
    new THREE.Vector2(0.105, topY + 0.12),
    new THREE.Vector2(0.105, topY - 0.145), // Close back to start watertight
  ];
  const socketLathe = new THREE.LatheGeometry(lathePts, 48);
  geoms.push(socketLathe);

  // 48 flush machined knurled grip flutes seated directly into the knurled band (r = 0.165)
  const knurlCount = 48;
  for (let i = 0; i < knurlCount; i++) {
    const a = (i / knurlCount) * Math.PI * 2;
    const ridge = new THREE.BoxGeometry(0.0075, 0.084, 0.009);
    ridge.rotateY(a);
    ridge.translate(Math.sin(a) * 0.166, topY + 0.18, Math.cos(a) * 0.166);
    geoms.push(ridge);
  }

  // 6 machined brass cap screws seated flush on top clamping flange + 6 subtle brass finial pins at bottom rib tips
  const bayAngle = (Math.PI * 2) / config.bayCount;
  const bottomTip = sampleLanternProfile(0.995, config);
  for (let k = 0; k < config.bayCount; k++) {
    const theta = bayAngle * (k + 0.5);
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);

    // Top dome-head brass bolt seated flush at y = topY + 0.026
    const boltHead = new THREE.CylinderGeometry(0.015, 0.0165, 0.012, 10);
    boltHead.translate(sinT * 0.244, topY + 0.03, cosT * 0.244);
    geoms.push(boltHead);

    const boltDome = new THREE.SphereGeometry(0.0145, 10, 6, 0, Math.PI * 2, 0, Math.PI * 0.5);
    boltDome.translate(sinT * 0.244, topY + 0.035, cosT * 0.244);
    geoms.push(boltDome);

    // Bottom flush brass keeper pin securing kelp hem to walnut rib tip
    const rivetBot = new THREE.SphereGeometry(0.0105, 8, 6);
    rivetBot.translate(
      sinT * (bottomTip.r + 0.022),
      bottomTip.y,
      cosT * (bottomTip.r + 0.022)
    );
    geoms.push(rivetBot);
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds the interior glazed dark-ceramic E26 socket insulator cup and threaded brass lamp base
 * visible when orbiting underneath or looking through the open bottom aperture.
 */
export function buildCeramicSocketGeometry(
  config: LanternConfig = DEFAULT_LANTERN_CONFIG
): THREE.BufferGeometry {
  const topY = config.halfHeight;
  // Strictly bottom-to-top exterior, top-to-bottom interior cavity
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.088, topY - 0.165),
    new THREE.Vector2(0.106, topY - 0.165),
    new THREE.Vector2(0.108, topY - 0.085),
    new THREE.Vector2(0.095, topY - 0.06),
    new THREE.Vector2(0.082, topY - 0.06),
    new THREE.Vector2(0.082, topY - 0.165),
    new THREE.Vector2(0.088, topY - 0.165),
  ];
  return new THREE.LatheGeometry(pts, 28);
}

/**
 * Builds the vintage tubular T45/T10 Edison bulb glass envelope (strictly bottom-to-top outward)
 * complete with an authentic blown-glass bottom exhaust pip and curved upper neck.
 */
export function buildBulbGlassGeometry(): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.001, -0.365), // Blown glass exhaust pip tip
    new THREE.Vector2(0.008, -0.352),
    new THREE.Vector2(0.014, -0.336),
    new THREE.Vector2(0.052, -0.326),
    new THREE.Vector2(0.094, -0.296),
    new THREE.Vector2(0.116, -0.238),
    new THREE.Vector2(0.122, -0.15),
    new THREE.Vector2(0.122, 0.21),
    new THREE.Vector2(0.112, 0.285),
    new THREE.Vector2(0.094, 0.345),
    new THREE.Vector2(0.082, 0.39),
  ];
  return new THREE.LatheGeometry(pts, 32);
}

/**
 * Builds the non-glowing internal vintage Edison bulb hardware:
 * E26 threaded brass screw cap, black glass pinch collar, internal glass stem press,
 * and delicate top/bottom filament support hooks.
 */
export function buildBulbHardwareGeometry(): THREE.BufferGeometry {
  const geoms: THREE.BufferGeometry[] = [];

  // E26 threaded brass screw cap + slender brass socket drop-stem connecting cleanly up into the ceramic socket
  const basePts: THREE.Vector2[] = [
    new THREE.Vector2(0.082, 0.365),
    new THREE.Vector2(0.086, 0.378),
    new THREE.Vector2(0.083, 0.392),
    new THREE.Vector2(0.087, 0.406),
    new THREE.Vector2(0.083, 0.42),
    new THREE.Vector2(0.087, 0.434),
    new THREE.Vector2(0.083, 0.448),
    new THREE.Vector2(0.095, 0.468), // Brass Edison socket cup rim
    new THREE.Vector2(0.095, 0.56),
    new THREE.Vector2(0.042, 0.585), // Slender brass drop-stem up into top collar
    new THREE.Vector2(0.036, 0.72),
    new THREE.Vector2(0.002, 0.725),
  ];
  geoms.push(new THREE.LatheGeometry(basePts, 24));

  // Internal glass stem pillar descending from the E26 press down to the lower support arbor
  const stemPts: THREE.Vector2[] = [
    new THREE.Vector2(0.002, -0.22),
    new THREE.Vector2(0.011, -0.215),
    new THREE.Vector2(0.009, -0.18),
    new THREE.Vector2(0.007, 0.22),
    new THREE.Vector2(0.022, 0.28),
    new THREE.Vector2(0.036, 0.365),
  ];
  geoms.push(new THREE.LatheGeometry(stemPts, 14));

  // Delicate support arbor spokes radiating from stem to hold the squirrel-cage tungsten wires
  const spokeCount = 8;
  for (let i = 0; i < spokeCount; i++) {
    const a = (i / spokeCount) * Math.PI * 2;
    const pTopA = new THREE.Vector3(0, 0.235, 0);
    const pTopB = new THREE.Vector3(Math.sin(a) * 0.038, 0.225, Math.cos(a) * 0.038);
    geoms.push(createSimpleStrut(pTopA, pTopB, 0.0018, 4));

    const pBotA = new THREE.Vector3(0, -0.205, 0);
    const pBotB = new THREE.Vector3(Math.sin(a) * 0.044, -0.205, Math.cos(a) * 0.044);
    geoms.push(createSimpleStrut(pBotA, pBotB, 0.0018, 4));
  }

  return mergeIndexedGeometries(geoms);
}

function createSimpleStrut(
  pA: THREE.Vector3,
  pB: THREE.Vector3,
  radius: number,
  radialSegments = 5
): THREE.BufferGeometry {
  const dir = new THREE.Vector3().subVectors(pB, pA);
  const len = Math.max(0.001, dir.length());
  const geom = new THREE.CylinderGeometry(radius, radius, len, radialSegments, 1, false);
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
 * Builds the delicate, authentic squirrel-cage tungsten filament wires inside the vintage Edison bulb
 * (no chunky solid center cylinder blocking the view through the glass!).
 */
export function buildFilamentGeometry(): THREE.BufferGeometry {
  const geoms: THREE.BufferGeometry[] = [];
  const strandCount = 8;
  const rTop = 0.038;
  const rMid = 0.049;
  const rBot = 0.044;

  for (let i = 0; i < strandCount; i++) {
    const a0 = (i / strandCount) * Math.PI * 2;
    const aMid = ((i + 0.35) / strandCount) * Math.PI * 2;
    const a1 = ((i + 0.5) / strandCount) * Math.PI * 2;

    const pTop = new THREE.Vector3(Math.sin(a0) * rTop, 0.225, Math.cos(a0) * rTop);
    const pUpMid = new THREE.Vector3(Math.sin(aMid) * rMid, 0.03, Math.cos(aMid) * rMid);
    const pBot = new THREE.Vector3(Math.sin(a1) * rBot, -0.205, Math.cos(a1) * rBot);
    const pNextTop = new THREE.Vector3(
      Math.sin(((i + 1) / strandCount) * Math.PI * 2) * rTop,
      0.225,
      Math.cos(((i + 1) / strandCount) * Math.PI * 2) * rTop
    );

    // Fine glowing tungsten hairpin loops
    geoms.push(createSimpleStrut(pTop, pUpMid, 0.0029, 5));
    geoms.push(createSimpleStrut(pUpMid, pBot, 0.0029, 5));
    geoms.push(createSimpleStrut(pBot, pNextTop, 0.0026, 5));
  }

  return mergeIndexedGeometries(geoms);
}

/**
 * Builds a subtle soft volumetric core capsule inside the Edison bulb to give warm radiant
 * candlelit readability through the narrow laced seam without solid-bar artifacts.
 */
export function buildFilamentAuraGeometry(): THREE.BufferGeometry {
  const geom = new THREE.CapsuleGeometry(0.068, 0.34, 8, 18);
  geom.translate(0, 0.01, 0);
  return geom;
}

/**
 * Creates a pre-allocated dynamic 3-ply braided hemp cord geometry with strictly OUTWARD-facing
 * normals and parallel-transported cross-section frames perpendicular to the taut hanging rope.
 * Zero per-frame allocations!
 */
export interface DynamicHempCord {
  geometry: THREE.BufferGeometry;
  update: (anchorY: number, socketWorldPos: THREE.Vector3, cordTwistRad?: number) => void;
}

export function createDynamicHempCord(segments = 32, radialSegments = 12): DynamicHempCord {
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
      uvs[idx * 2] = u * 3.0;
      uvs[idx * 2 + 1] = v * 20.0;
    }
    if (i < segments) {
      for (let j = 0; j < radialSegments; j++) {
        const a = i * (radialSegments + 1) + j;
        const b = a + 1;
        const c = (i + 1) * (radialSegments + 1) + j + 1;
        const d = (i + 1) * (radialSegments + 1) + j;
        // Outward winding: (b - a) x (d - a) = E_phi x T_down = +radialOut!
        indices.push(a, b, d, b, c, d);
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

  const baseRadius = 0.023;
  // Pre-allocated centerline & orthonormal frame scratch arrays (zero GC inside update)
  const centerX = new Float32Array(segments + 1);
  const centerY = new Float32Array(segments + 1);
  const centerZ = new Float32Array(segments + 1);

  function update(anchorY: number, socketWorldPos: THREE.Vector3, cordTwistRad = 0): void {
    const sx = socketWorldPos.x;
    const sy = socketWorldPos.y;
    const sz = socketWorldPos.z;

    // A weighted pendant lantern pulls the braided hemp cord nearly straight along the tension line,
    // with only a subtle 7% strain-relief ease at the top ceiling ferrule so it never bows like a stiff pipe.
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const strainEase = t < 0.12 ? (t / 0.12) * (t / 0.12) * 0.12 * 0.25 + t * 0.75 : t;
      centerX[i] = sx * strainEase;
      centerY[i] = anchorY + (sy - anchorY) * t;
      centerZ[i] = sz * strainEase;
    }

    // Parallel-transported orthonormal frame (T downward, N, B with N x B = +T)
    let nx = 1;
    let ny = 0;
    let nz = 0;

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const iPrev = Math.max(0, i - 1);
      const iNext = Math.min(segments, i + 1);
      let tx = centerX[iNext] - centerX[iPrev];
      let ty = centerY[iNext] - centerY[iPrev];
      let tz = centerZ[iNext] - centerZ[iPrev];
      const tLen = Math.hypot(tx, ty, tz) || 1;
      tx /= tLen;
      ty /= tLen;
      tz /= tLen;

      // Gram-Schmidt orthogonalize N against T
      const nDotT = nx * tx + ny * ty + nz * tz;
      nx -= nDotT * tx;
      ny -= nDotT * ty;
      nz -= nDotT * tz;
      const nLen = Math.hypot(nx, ny, nz) || 1;
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      // B = T x N so N x B = +T (guaranteeing E_phi x T = +radialOut!)
      const bx = ty * nz - tz * ny;
      const by = tz * nx - tx * nz;
      const bz = tx * ny - ty * nx;

      // 3-ply twisted hemp helical braid coupled dynamically to lantern yaw twist
      const twistAngle = t * 62.0 + cordTwistRad * t;
      const cx = centerX[i];
      const cy = centerY[i];
      const cz = centerZ[i];

      for (let j = 0; j <= radialSegments; j++) {
        const phi = (j / radialSegments) * Math.PI * 2;
        const cosP = Math.cos(phi);
        const sinP = Math.sin(phi);

        // 3-strand rope ridge + analytical outward slope normal
        const braidPhase = phi * 3.0 + twistAngle;
        const braidBump = 1.0 + 0.16 * Math.sin(braidPhase);
        const r = baseRadius * braidBump;
        const drDphi = 0.48 * Math.cos(braidPhase);

        const rx = cosP * nx + sinP * bx;
        const ry = cosP * ny + sinP * by;
        const rz = cosP * nz + sinP * bz;

        const px = cx + rx * r;
        const py = cy + ry * r;
        const pz = cz + rz * r;

        // Tangent along +phi
        const phix = -sinP * nx + cosP * bx;
        const phiy = -sinP * ny + cosP * by;
        const phiz = -sinP * nz + cosP * bz;

        // Perturbed outward surface normal for crisp 3-ply braid lighting
        let snx = rx - phix * (drDphi * 0.45);
        let sny = ry - phiy * (drDphi * 0.45);
        let snz = rz - phiz * (drDphi * 0.45);
        const snLen = Math.hypot(snx, sny, snz) || 1;
        snx /= snLen;
        sny /= snLen;
        snz /= snLen;

        const idx = i * (radialSegments + 1) + j;
        positions[idx * 3] = px;
        positions[idx * 3 + 1] = py;
        positions[idx * 3 + 2] = pz;

        normals[idx * 3] = snx;
        normals[idx * 3 + 1] = sny;
        normals[idx * 3 + 2] = snz;
      }
    }

    posAttr.needsUpdate = true;
    normAttr.needsUpdate = true;
    geometry.computeBoundingSphere();
  }

  return { geometry, update };
}
