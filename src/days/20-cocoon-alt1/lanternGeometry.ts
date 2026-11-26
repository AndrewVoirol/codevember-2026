import * as THREE from 'three/webgpu';

export const LANTERN_HEIGHT = 1.62;
export const LANTERN_BOTTOM_Y = -0.78;
export const LANTERN_TOP_Y = 0.84;
export const LANTERN_MOUTH_RADIUS = 0.26;
export const LANTERN_COLLAR_RADIUS = 0.20;
export const LANTERN_MAX_RADIUS = 0.62;
export const SUSPENSION_CORD_TOP_Y = 2.4;

/**
 * Clean ovoid-cocoon lantern radial profile r(v) for v in [0, 1]:
 * - v = 0.00 (bottom bamboo mouth hoop, y = -0.78): r = 0.26
 * - v = 0.52 (equatorial belly): r ~ 0.62
 * - v = 1.00 (top wooden collar, y = +0.84): r = 0.20
 */
export function evaluateOvoidLanternRadius(vRaw: number): number {
  const v = Math.max(0, Math.min(1, vRaw));
  const linearBase = LANTERN_MOUTH_RADIUS * (1.0 - v) + LANTERN_COLLAR_RADIUS * v;
  // Asymmetric ovoid swell peaking around v ~ 0.52 with full rounded shoulders
  const s = Math.sin(Math.pow(v, 0.94) * Math.PI);
  const bulge = Math.pow(Math.max(0, s), 0.72);
  return linearBase + 0.391 * bulge;
}

export function evaluateOvoidLanternY(vRaw: number): number {
  const v = Math.max(0, Math.min(1, vRaw));
  return LANTERN_BOTTOM_Y + v * LANTERN_HEIGHT;
}

// ---------------------------------------------------------------------------
// Deterministic PRNG & MeshBuilder Helpers (zero NaN, strictly outward-wound)
// ---------------------------------------------------------------------------

function hash01(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

interface MeshBuilder {
  positions: number[];
  uvs: number[];
  indices: number[];
}

function createMeshBuilder(): MeshBuilder {
  return { positions: [], uvs: [], indices: [] };
}

function finalizeBuilder(
  b: MeshBuilder,
  extraAttrs?: Record<string, { data: Float32Array; itemSize: number }>,
): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(b.positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uvs, 2));
  if (extraAttrs) {
    for (const [name, attr] of Object.entries(extraAttrs)) {
      geo.setAttribute(name, new THREE.BufferAttribute(attr.data, attr.itemSize));
    }
  }
  geo.setIndex(b.indices);
  geo.computeVertexNormals();
  return geo;
}

const _p0 = new THREE.Vector3();
const _tan = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _altUp = new THREE.Vector3(1, 0, 0);

/**
 * Appends a closed, strictly outward-wound swept tube along a CatmullRomCurve3
 * using parallel transport frames to prevent twisting artifacts.
 */
function appendSweptTube(
  b: MeshBuilder,
  curve: THREE.CatmullRomCurve3,
  steps: number,
  radialSegs: number,
  radiusFn: (u: number, angle: number) => number,
  capStart = true,
  capEnd = true,
  closedLoop = false,
  onVertex?: (u: number) => void,
): void {
  const baseVert = b.positions.length / 3;
  const stride = radialSegs + 1;

  const prevTan = new THREE.Vector3();
  const curN = new THREE.Vector3();
  const curB = new THREE.Vector3();
  const rotAxis = new THREE.Vector3();
  const rotQuat = new THREE.Quaternion();

  curve.getTangent(0, prevTan).normalize();
  const ref = Math.abs(prevTan.y) < 0.92 ? _up : _altUp;
  curN.crossVectors(prevTan, ref).normalize();
  curB.crossVectors(prevTan, curN).normalize();

  for (let i = 0; i <= steps; i++) {
    const u = i / steps;
    const sampleU = closedLoop ? (i === steps ? 0 : u) : u;
    curve.getPoint(sampleU, _p0);
    curve.getTangent(sampleU, _tan).normalize();

    if (i > 0) {
      const dot = Math.max(-1, Math.min(1, prevTan.dot(_tan)));
      if (dot < 0.99999) {
        rotAxis.crossVectors(prevTan, _tan);
        if (rotAxis.lengthSq() > 1e-10) {
          rotAxis.normalize();
          rotQuat.setFromAxisAngle(rotAxis, Math.acos(dot));
          curN.applyQuaternion(rotQuat).normalize();
          curB.crossVectors(_tan, curN).normalize();
        }
      }
      prevTan.copy(_tan);
    }

    for (let j = 0; j <= radialSegs; j++) {
      const v = j / radialSegs;
      const ang = v * Math.PI * 2;
      const r = Math.max(1e-4, radiusFn(u, ang));
      const cx = Math.cos(ang) * r;
      const cy = Math.sin(ang) * r;
      b.positions.push(
        _p0.x + curN.x * cx + curB.x * cy,
        _p0.y + curN.y * cx + curB.y * cy,
        _p0.z + curN.z * cx + curB.z * cy,
      );
      b.uvs.push(v, u);
      if (onVertex) onVertex(u);
    }
  }

  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = baseVert + i * stride + j;
      const d = a + 1;
      const bIdx = a + stride;
      const c = bIdx + 1;
      b.indices.push(a, d, c, a, c, bIdx);
    }
  }

  if (!closedLoop && capStart) {
    const startCenter = b.positions.length / 3;
    curve.getPoint(0, _p0);
    b.positions.push(_p0.x, _p0.y, _p0.z);
    b.uvs.push(0.5, 0);
    if (onVertex) onVertex(0);
    for (let j = 0; j < radialSegs; j++) {
      b.indices.push(startCenter, baseVert + j + 1, baseVert + j);
    }
  }

  if (!closedLoop && capEnd) {
    const endCenter = b.positions.length / 3;
    curve.getPoint(1, _p0);
    b.positions.push(_p0.x, _p0.y, _p0.z);
    b.uvs.push(0.5, 1);
    if (onVertex) onVertex(1);
    const endRing = baseVert + steps * stride;
    for (let j = 0; j < radialSegs; j++) {
      b.indices.push(endCenter, endRing + j, endRing + j + 1);
    }
  }
}

/**
 * Appends an outward-wound closed circular chamfered split-bamboo hoop centered at (0, yCenter, 0).
 */
function appendSplitBambooHoop(
  b: MeshBuilder,
  yCenter: number,
  radius: number,
  radialThick: number,
  height: number,
  segments = 48,
): void {
  const hr = radialThick * 0.5;
  const hh = height * 0.5;
  const bevel = Math.min(hr, hh) * 0.26;

  const profile: Array<[number, number]> = [
    [hr - bevel, -hh],
    [hr, -hh + bevel],
    [hr, hh - bevel],
    [hr - bevel, hh],
    [-hr + bevel, hh],
    [-hr, hh - bevel],
    [-hr, -hh + bevel],
    [-hr + bevel, -hh],
  ];

  const pCount = profile.length;
  const stride = pCount + 1;
  const baseVert = b.positions.length / 3;

  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const theta = u * Math.PI * 2;
    const sT = Math.sin(theta);
    const cT = Math.cos(theta);

    for (let j = 0; j <= pCount; j++) {
      const [dr, dy] = profile[j % pCount];
      const r = radius + dr;
      b.positions.push(sT * r, yCenter + dy, cT * r);
      b.uvs.push(u * 4.0, j / pCount);
    }
  }

  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < pCount; j++) {
      const a = baseVert + i * stride + j;
      const bIdx = a + stride;
      const c = bIdx + 1;
      const d = a + 1;
      b.indices.push(a, bIdx, c, a, c, d);
    }
  }
}

/**
 * Appends a straight outward-wound split-bamboo stick between pA and pB.
 */
function appendStrut(
  b: MeshBuilder,
  pA: THREE.Vector3,
  pB: THREE.Vector3,
  radius: number,
  radialSegs = 6,
): void {
  const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
  const curve = new THREE.CatmullRomCurve3([pA.clone(), mid, pB.clone()]);
  appendSweptTube(b, curve, 4, radialSegs, () => radius, true, true, false);
}

// ---------------------------------------------------------------------------
// 1. Ovoid Kozo Rice-Paper Dome Geometry (12-Panel Faceted-Ribbed Canopy)
// ---------------------------------------------------------------------------

/**
 * Builds the 12-panel ovoid handmade kozo rice-paper canopy matching Concept A's
 * Hero & Macro renders, with subtle outward panel billow between meridian ribs,
 * open bottom mouth (for looking up into the glowing burner cavity), sealed top crown,
 * and per-vertex `aRibProximity` (0 panel center, 1 rib/rim seams) & `aWaistMask`
 * (1 across the silk-wrapped waist/lower belly) attributes. Strictly outward-wound!
 */
export function buildOvoidPaperDomeGeometry(ribCount = 12, heightSegs = 28): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const subPerPanel = 4;
  const totalRadial = ribCount * subPerPanel;
  const cols = totalRadial + 1;
  const rows = heightSegs + 1;

  const ribProximity = new Float32Array(cols * rows + 1); // +1 for top cap center
  const waistMask = new Float32Array(cols * rows + 1);

  for (let iy = 0; iy <= heightSegs; iy++) {
    const v = iy / heightSegs;
    const y = evaluateOvoidLanternY(v);
    const baseR = evaluateOvoidLanternRadius(v);

    // Rim proximity at mouth (v=0) and top collar (v=1)
    const rimProx = Math.max(
      Math.exp(-Math.pow(v / 0.04, 2)),
      Math.exp(-Math.pow((1.0 - v) / 0.04, 2)),
    );

    // Smooth waist mask: 1.0 across silk-wrapped zone v in [0.10, 0.56], tapering smoothly to 0
    const waistRise = smoothstepCpu(0.03, 0.14, v);
    const waistFall = 1.0 - smoothstepCpu(0.52, 0.72, v);
    const wMask = waistRise * waistFall;

    for (let ix = 0; ix <= totalRadial; ix++) {
      const u = ix / totalRadial;
      const panelIdx = Math.floor(ix / subPerPanel);
      const panelT = (ix % subPerPanel) / subPerPanel;

      const a0 = (panelIdx / ribCount) * Math.PI * 2;
      const a1 = ((panelIdx + 1) / ribCount) * Math.PI * 2;
      const theta = u * Math.PI * 2;

      const chordX = Math.sin(a0) * baseR * (1.0 - panelT) + Math.sin(a1) * baseR * panelT;
      const chordZ = Math.cos(a0) * baseR * (1.0 - panelT) + Math.cos(a1) * baseR * panelT;
      const circX = Math.sin(theta) * baseR;
      const circZ = Math.cos(theta) * baseR;

      // Subtle outward kozo-paper billow between the 12 slender bamboo ribs
      const billow = 0.011 * Math.sin(panelT * Math.PI) * Math.sin(v * Math.PI);
      const x = chordX * 0.42 + circX * 0.58 + Math.sin(theta) * billow;
      const z = chordZ * 0.42 + circZ * 0.58 + Math.cos(theta) * billow;

      const vIdx = iy * cols + ix;
      b.positions.push(x, y, z);
      b.uvs.push(u, v);

      const seamDist = Math.min(panelT, 1.0 - panelT);
      const meridianProx = Math.exp(-Math.pow(seamDist / 0.14, 2));
      ribProximity[vIdx] = Math.min(1.0, Math.max(meridianProx, rimProx));
      waistMask[vIdx] = wMask;
    }
  }

  // Strictly outward winding: (+ix) x (+iy) = (+sin(theta), 0, +cos(theta))
  for (let iy = 0; iy < heightSegs; iy++) {
    for (let ix = 0; ix < totalRadial; ix++) {
      const a = iy * cols + ix;
      const bIdx = a + 1;
      const c = bIdx + cols;
      const d = a + cols;
      b.indices.push(a, bIdx, c, a, c, d);
    }
  }

  // Seal top collar opening cleanly (outward +Y)
  const apexIdx = b.positions.length / 3;
  b.positions.push(0, LANTERN_TOP_Y + 0.008, 0);
  b.uvs.push(0.5, 1.0);
  ribProximity[apexIdx] = 1.0;
  waistMask[apexIdx] = 0.0;
  const topRing = heightSegs * cols;
  for (let ix = 0; ix < totalRadial; ix++) {
    b.indices.push(topRing + ix, topRing + ix + 1, apexIdx);
  }

  return finalizeBuilder(b, {
    aRibProximity: { data: ribProximity, itemSize: 1 },
    aWaistMask: { data: waistMask, itemSize: 1 },
  });
}

// ---------------------------------------------------------------------------
// 2. Ovoid Split-Bamboo Frame & Hanging Cord Geometry
// ---------------------------------------------------------------------------

/**
 * Builds the 12-rib dark-umber split-bamboo lantern skeleton matching Concept A:
 * - 12 slender curved meridian split-bamboo ribs flush along the ovoid exterior
 * - Flat-band circular bent split-bamboo mouth hoop at v = 0 (y = -0.78)
 * - Fine crossed burner support sticks + central burner ring inside the mouth hoop
 * - Top wooden/bamboo collar ring at v = 1 (y = +0.84) with cross-tie hub
 * - Slender twisted dark suspension cord rising vertically from y = +0.84 to y = +2.4
 *   with a neat knotted bulge right above the top collar.
 */
export function buildOvoidBambooFrameGeometry(ribCount = 12): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const ribOffset = 0.007;

  // 2a. Bottom flat-band circular bent split-bamboo mouth hoop (prominent rim in Macro render)
  appendSplitBambooHoop(b, LANTERN_BOTTOM_Y, LANTERN_MOUTH_RADIUS + ribOffset, 0.020, 0.044, 56);

  // 2b. Fine crossed burner support sticks + central ember ring inside the mouth hoop
  const innerMouthR = LANTERN_MOUTH_RADIUS - 0.003;
  const burnerY = LANTERN_BOTTOM_Y + 0.008;
  appendStrut(
    b,
    new THREE.Vector3(-innerMouthR, burnerY, 0),
    new THREE.Vector3(innerMouthR, burnerY, 0),
    0.0038,
    6,
  );
  appendStrut(
    b,
    new THREE.Vector3(0, burnerY, -innerMouthR),
    new THREE.Vector3(0, burnerY, innerMouthR),
    0.0038,
    6,
  );
  appendSplitBambooHoop(b, burnerY, 0.042, 0.007, 0.009, 20);

  // 2c. Top wooden/bamboo collar ring + cross-tie hub
  appendSplitBambooHoop(b, LANTERN_TOP_Y, LANTERN_COLLAR_RADIUS + ribOffset * 0.7, 0.018, 0.026, 40);
  const innerCollarR = LANTERN_COLLAR_RADIUS - 0.004;
  appendStrut(
    b,
    new THREE.Vector3(-innerCollarR, LANTERN_TOP_Y, 0),
    new THREE.Vector3(innerCollarR, LANTERN_TOP_Y, 0),
    0.0045,
    6,
  );
  appendStrut(
    b,
    new THREE.Vector3(0, LANTERN_TOP_Y, -innerCollarR),
    new THREE.Vector3(0, LANTERN_TOP_Y, innerCollarR),
    0.0045,
    6,
  );

  // 2d. 12 slender curved meridian split-bamboo ribs flush on the ovoid exterior
  const ribSamples = 18;
  for (let k = 0; k < ribCount; k++) {
    const theta = (k / ribCount) * Math.PI * 2;
    const sT = Math.sin(theta);
    const cT = Math.cos(theta);
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= ribSamples; i++) {
      const v = i / ribSamples;
      const r = evaluateOvoidLanternRadius(v) + ribOffset;
      const y = evaluateOvoidLanternY(v);
      pts.push(new THREE.Vector3(sT * r, y, cT * r));
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    appendSweptTube(
      b,
      curve,
      24,
      6,
      (_u, ang) => 0.0092 + 0.0024 * Math.cos(ang * 2), // slightly flat-banded split bamboo
      true,
      true,
      false,
    );
  }

  // 2e. Slender twisted suspension cord rising from y = +0.84 up to y = +2.4 with collar knot
  const cordCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, LANTERN_TOP_Y - 0.005, 0),
    new THREE.Vector3(0.002, LANTERN_TOP_Y + 0.04, 0.001),
    new THREE.Vector3(-0.001, (LANTERN_TOP_Y + SUSPENSION_CORD_TOP_Y) * 0.5, -0.001),
    new THREE.Vector3(0, SUSPENSION_CORD_TOP_Y, 0),
  ]);
  appendSweptTube(
    b,
    cordCurve,
    24,
    7,
    (u, ang) => {
      // Subtle knot bulge right above top collar (u ~ 0.015..0.045) + subtle twisted-cord micro-ridge
      const knot = Math.exp(-Math.pow((u - 0.025) / 0.014, 2)) * 0.011;
      const twist = 0.0008 * Math.sin(u * 140.0 + ang * 2.0);
      return 0.0068 + knot + twist;
    },
    true,
    true,
    false,
  );

  return finalizeBuilder(b);
}

// ---------------------------------------------------------------------------
// 3. Crisscross Raw-Silk Wrap Bands Geometry (140 Diagonal Cocoon Filaments)
// ---------------------------------------------------------------------------

/**
 * Builds dense overlapping diagonal crisscross bands of fine tubular raw-silk
 * cocoon filaments wrapped tightly around the lower-mid lantern belly and waist
 * (v in [0.06, 0.62]), hugging just outside the 12 bamboo ribs at crisscrossing
 * pitch angles (+/-18°..+/-34°).
 * Per-vertex attributes:
 * - `aStrandT`: 0..1 along strand
 * - `aStrandSeed`: deterministic 0..1 per strand
 * - `aUnwindOrder`: 0..1 (outer/lower strands carry smaller values so they unspool first!)
 */
export function buildSilkWrapBandsGeometry(strandCount = 140): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const strandTList: number[] = [];
  const strandSeedList: number[] = [];
  const unwindOrderList: number[] = [];

  // 6 overlapping diagonal wrap swaths (alternating + and - tilt angles) matching Hero & Macro
  const swathCount = 6;
  const swathVCenters = [0.22, 0.30, 0.36, 0.42, 0.27, 0.46];
  const swathVAmps = [0.14, 0.17, 0.19, 0.16, 0.15, 0.14];
  const swathPhases = [0.15, 1.45, 2.75, 4.10, 0.85, 3.35];

  const stepsPerStrand = 30;
  const radialSegs = 4;

  for (let k = 0; k < strandCount; k++) {
    const seed = hash01(k * 1.37 + 0.19);
    const seed2 = hash01(k * 2.91 + 7.31);
    const seed3 = hash01(k * 5.13 + 13.7);

    const swathIdx = k % swathCount;
    const tiltSign = swathIdx % 2 === 0 ? 1 : -1;
    const vCenter = swathVCenters[swathIdx] + (seed - 0.5) * 0.085;
    const vAmp = (swathVAmps[swathIdx] + (seed2 - 0.5) * 0.05) * tiltSign;
    const phase = swathPhases[swathIdx] + (seed3 - 0.5) * 0.55;

    // Layer depth: outer strands sit slightly further out (0.011..0.024) and unwind earlier
    const layerNorm = k / Math.max(1, strandCount - 1); // 0 = inner, 1 = outer
    const radialOffset = 0.012 + layerNorm * 0.013 + (seed2 - 0.5) * 0.002;
    const unwindOrder = Math.max(0, Math.min(1, (1.0 - layerNorm) * 0.65 + (vCenter - 0.12) * 0.7));

    // Build a smooth closed oblique wrap loop hugging the ovoid exterior
    const controlCount = 16;
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < controlCount; i++) {
      const t = i / controlCount;
      const theta = t * Math.PI * 2 + phase;
      const rawV = vCenter + vAmp * Math.sin(theta - phase) + 0.014 * Math.sin(theta * 2.0 + seed * 6.28);
      const v = Math.max(0.055, Math.min(0.63, rawV));
      const r = evaluateOvoidLanternRadius(v) + radialOffset;
      const y = evaluateOvoidLanternY(v);
      pts.push(new THREE.Vector3(Math.sin(theta) * r, y, Math.cos(theta) * r));
    }

    const curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal');
    const filamentRadius = 0.0021 + seed * 0.0014;

    appendSweptTube(
      b,
      curve,
      stepsPerStrand,
      radialSegs,
      () => filamentRadius,
      false,
      false,
      true,
      (u) => {
        strandTList.push(u);
        strandSeedList.push(seed);
        unwindOrderList.push(unwindOrder);
      },
    );
  }

  return finalizeBuilder(b, {
    aStrandT: { data: new Float32Array(strandTList), itemSize: 1 },
    aStrandSeed: { data: new Float32Array(strandSeedList), itemSize: 1 },
    aUnwindOrder: { data: new Float32Array(unwindOrderList), itemSize: 1 },
  });
}

// ---------------------------------------------------------------------------
// 4. Unspooling Gossamer Silk Loops Geometry (28 Peeling Outward Arcs)
// ---------------------------------------------------------------------------

/**
 * Builds 28 graceful outward-peeling gossamer raw-silk loop arcs anchored along
 * the left flank, right flank, rear flanks, and lower rim under the bamboo mouth hoop
 * (matching the wispy glowing loops in both Concept A Hero & Macro renders).
 * Per-vertex attributes:
 * - `aLoopFreedom`: 0 at both lantern anchors -> 1 at furthest floating apex
 * - `aLoopSeed`: deterministic 0..1 per loop for staggered GPU billowing & wind sway
 */
export function buildUnspoolingSilkLoopsGeometry(loopCount = 28): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const freedomList: number[] = [];
  const seedList: number[] = [];

  const stepsPerLoop = 24;
  const radialSegs = 5;

  for (let k = 0; k < loopCount; k++) {
    const seed = hash01(k * 3.17 + 1.13);
    const seed2 = hash01(k * 4.73 + 5.91);
    const seed3 = hash01(k * 7.19 + 2.41);

    // Distribute loops matching Hero & Macro:
    // ~40% lush left-flank billowing loops, ~35% right-flank loops, ~25% under-mouth drooping loops
    const isUnderMouth = k >= Math.floor(loopCount * 0.75);
    let aStart: number;
    let aEnd: number;
    let vStart: number;
    let vEnd: number;
    let apexRAdd: number;
    let apexYShift: number;

    if (isUnderMouth) {
      // Loops anchored near the bottom mouth hoop, dipping gracefully below y = -0.78 (Macro render!)
      const localIdx = k - Math.floor(loopCount * 0.75);
      const spanCount = Math.max(1, loopCount - Math.floor(loopCount * 0.75));
      aStart = (localIdx / spanCount) * Math.PI * 2 + (seed - 0.5) * 0.45;
      aEnd = aStart + 0.85 + seed2 * 0.75;
      vStart = 0.04 + seed * 0.14;
      vEnd = 0.03 + seed2 * 0.12;
      apexRAdd = 0.08 + seed3 * 0.16;
      apexYShift = -(0.14 + seed * 0.18);
    } else {
      // Flank loops: emphasize left (-X -> theta ~ -PI/2) and right (+X -> theta ~ +PI/2), plus full 360° coverage
      const sideBias = k % 3 === 0 ? -Math.PI * 0.5 : k % 3 === 1 ? Math.PI * 0.5 : (k / loopCount) * Math.PI * 2;
      aStart = sideBias + (seed - 0.5) * 1.15;
      aEnd = aStart + (seed2 > 0.5 ? 1 : -1) * (0.45 + seed3 * 0.55);
      vStart = 0.10 + seed * 0.40;
      vEnd = Math.max(0.06, Math.min(0.56, vStart + (seed2 - 0.5) * 0.24));
      apexRAdd = 0.18 + seed3 * 0.26;
      apexYShift = -(0.04 + seed2 * 0.14);
    }

    const rStart = evaluateOvoidLanternRadius(vStart) + 0.014;
    const yStart = evaluateOvoidLanternY(vStart);
    const rEnd = evaluateOvoidLanternRadius(vEnd) + 0.014;
    const yEnd = evaluateOvoidLanternY(vEnd);

    const aMid = (aStart + aEnd) * 0.5 + (seed - 0.5) * 0.25;
    const vMid = (vStart + vEnd) * 0.5;
    const rMidBase = evaluateOvoidLanternRadius(vMid) + 0.015;
    const yMidBase = evaluateOvoidLanternY(vMid);

    // 5-point gracefully arching organic gossamer loop
    const p0 = new THREE.Vector3(Math.sin(aStart) * rStart, yStart, Math.cos(aStart) * rStart);
    const p1 = new THREE.Vector3(
      Math.sin(aStart + (aMid - aStart) * 0.45) * (rMidBase + apexRAdd * 0.72),
      yStart + (yMidBase - yStart) * 0.4 + apexYShift * 0.45 + 0.04 * (seed2 - 0.5),
      Math.cos(aStart + (aMid - aStart) * 0.45) * (rMidBase + apexRAdd * 0.72),
    );
    const p2 = new THREE.Vector3(
      Math.sin(aMid) * (rMidBase + apexRAdd),
      yMidBase + apexYShift,
      Math.cos(aMid) * (rMidBase + apexRAdd),
    );
    const p3 = new THREE.Vector3(
      Math.sin(aMid + (aEnd - aMid) * 0.55) * (rMidBase + apexRAdd * 0.68),
      yEnd + (yMidBase - yEnd) * 0.4 + apexYShift * 0.55 - 0.03 * (seed3 - 0.5),
      Math.cos(aMid + (aEnd - aMid) * 0.55) * (rMidBase + apexRAdd * 0.68),
    );
    const p4 = new THREE.Vector3(Math.sin(aEnd) * rEnd, yEnd, Math.cos(aEnd) * rEnd);

    const curve = new THREE.CatmullRomCurve3([p0, p1, p2, p3, p4]);
    const loopRadius = 0.0020 + seed * 0.0013;

    appendSweptTube(
      b,
      curve,
      stepsPerLoop,
      radialSegs,
      (u) => loopRadius * (0.85 + 0.25 * Math.sin(u * Math.PI)),
      true,
      true,
      false,
      (u) => {
        const freedom = Math.pow(Math.max(0, Math.sin(u * Math.PI)), 1.15);
        freedomList.push(freedom);
        seedList.push(seed);
      },
    );
  }

  return finalizeBuilder(b, {
    aLoopFreedom: { data: new Float32Array(freedomList), itemSize: 1 },
    aLoopSeed: { data: new Float32Array(seedList), itemSize: 1 },
  });
}

// ---------------------------------------------------------------------------
// 5. Origami Butterfly Wing Geometry (Crisp-Folded Faceted Forewing + Hindwing)
// ---------------------------------------------------------------------------

/**
 * Builds one refined, crisp-folded origami butterfly wing (`side: -1 | 1`) with
 * distinct swept forewing + notched tailed hindwing facets and non-indexed flat/crisp
 * crease normals matching Concept A's Macro render. Hinged cleanly along x = 0.
 */
export function buildOrigamiButterflyWingGeometry(side: -1 | 1): THREE.BufferGeometry {
  const s = side;

  const verts = new Float32Array([
    0.0, 0.02, 0.34,        // 0: anterior thorax hinge
    0.0, 0.065, 0.02,       // 1: mid-thorax mountain peak
    0.0, -0.02, -0.44,      // 2: posterior abdomen tail hinge
    s * 1.18, 0.24, 0.64,   // 3: forewing outer apex
    s * 0.62, -0.09, 0.30,  // 4: forewing inner valley fold
    s * 0.96, 0.06, 0.02,   // 5: forewing trailing corner
    s * 0.56, -0.10, -0.10, // 6: wing waist notch valley
    s * 0.78, 0.10, -0.34,  // 7: hindwing outer lobe
    s * 0.34, -0.06, -0.78, // 8: hindwing pointed tail tip
  ]);

  const uvs = new Float32Array([
    0.0, 0.80,
    0.0, 0.52,
    0.0, 0.16,
    1.0, 0.96,
    0.54, 0.68,
    0.84, 0.50,
    0.48, 0.40,
    0.68, 0.22,
    0.32, 0.02,
  ]);

  const indices =
    side === 1
      ? [
          0, 3, 4,
          0, 4, 1,
          4, 3, 5,
          1, 4, 5,
          1, 5, 6,
          1, 6, 7,
          1, 7, 2,
          2, 7, 8,
        ]
      : [
          0, 4, 3,
          0, 1, 4,
          4, 5, 3,
          1, 5, 4,
          1, 6, 5,
          1, 7, 6,
          1, 2, 7,
          2, 8, 7,
        ];

  const indexed = new THREE.BufferGeometry();
  indexed.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  indexed.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  indexed.setIndex(indices);
  const geom = indexed.toNonIndexed();
  indexed.dispose();
  geom.computeVertexNormals();
  return geom;
}

// ---------------------------------------------------------------------------
// 6. Ember Billboard Quad & Interior Flame Core Helpers
// ---------------------------------------------------------------------------

/**
 * Unit 1x1 billboard quad for instanced rising warm ember/pollen motes.
 */
export function buildEmberQuadGeometry(): THREE.BufferGeometry {
  return new THREE.PlaneGeometry(1, 1);
}

/**
 * Compact outward-wound sphere for the warm glowing ember/flame core inside the lantern cavity.
 */
export function buildEmberCoreGeometry(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(0.14, 20, 14);
  geo.computeVertexNormals();
  return geo;
}

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
