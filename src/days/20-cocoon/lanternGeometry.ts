import * as THREE from 'three/webgpu';

export const LANTERN_HEIGHT = 1.62;
export const LANTERN_BOTTOM_Y = -0.78;
export const LANTERN_TOP_Y = 0.84;
export const LANTERN_MOUTH_RADIUS = 0.26;
export const LANTERN_COLLAR_RADIUS = 0.205;
export const LANTERN_MAX_RADIUS = 0.62;
export const SUSPENSION_CORD_TOP_Y = 2.45;

/**
 * Clean ovoid-cocoon lantern radial profile r(v) for v in [0, 1]:
 * - v = 0.00 (bottom bamboo mouth hoop, y = -0.78): r = 0.26
 * - v = 0.52 (equatorial belly): r ~ 0.62
 * - v = 1.00 (top wooden collar, y = +0.84): r = 0.205
 */
export function evaluateOvoidLanternRadius(vRaw: number): number {
  const v = Math.max(0, Math.min(1, vRaw));
  const linearBase = LANTERN_MOUTH_RADIUS * (1.0 - v) + LANTERN_COLLAR_RADIUS * v;
  // Asymmetric ovoid swell peaking around v ~ 0.53 with full rounded shoulders
  const s = Math.sin(Math.pow(v, 0.93) * Math.PI);
  const bulge = Math.pow(Math.max(0, s), 0.71);
  return linearBase + 0.391 * bulge;
}

export function evaluateOvoidLanternY(vRaw: number): number {
  const v = Math.max(0, Math.min(1, vRaw));
  return LANTERN_BOTTOM_Y + v * LANTERN_HEIGHT;
}

/**
 * Evaluates exact outward meridian profile normal (nr, ny) and tangent (tr, ty)
 * at parameter v in [0, 1] via central finite difference.
 */
function evaluateOvoidProfileFrame(vRaw: number): {
  r: number;
  y: number;
  tr: number;
  ty: number;
  nr: number;
  ny: number;
} {
  const v = Math.max(0, Math.min(1, vRaw));
  const eps = 1e-4;
  const v0 = Math.max(0, v - eps);
  const v1 = Math.min(1, v + eps);
  const dv = Math.max(1e-6, v1 - v0);
  const dr = (evaluateOvoidLanternRadius(v1) - evaluateOvoidLanternRadius(v0)) / dv;
  const dy = LANTERN_HEIGHT;
  const len = Math.hypot(dr, dy);
  const tr = dr / len;
  const ty = dy / len;
  // Outward normal perpendicular to upward meridian tangent (tr, ty): (+ty, -tr)
  return {
    r: evaluateOvoidLanternRadius(v),
    y: evaluateOvoidLanternY(v),
    tr,
    ty,
    nr: ty,
    ny: -tr,
  };
}

// ---------------------------------------------------------------------------
// Deterministic PRNG, 2D Fold Noise & MeshBuilder Helpers (outward-wound)
// ---------------------------------------------------------------------------

function hash01(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function hash2D(ix: number, iy: number, salt: number): number {
  const h = Math.sin(ix * 127.1 + iy * 311.7 + salt * 74.7) * 43758.5453123;
  return h - Math.floor(h);
}

function valueNoise2D(x: number, y: number, salt: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const tx = x - ix;
  const ty = y - iy;
  const ux = tx * tx * (3 - 2 * tx);
  const uy = ty * ty * (3 - 2 * ty);
  const n00 = hash2D(ix, iy, salt);
  const n10 = hash2D(ix + 1, iy, salt);
  const n01 = hash2D(ix, iy + 1, salt);
  const n11 = hash2D(ix + 1, iy + 1, salt);
  const nx0 = n00 + (n10 - n00) * ux;
  const nx1 = n01 + (n11 - n01) * ux;
  return nx0 + (nx1 - nx0) * uy;
}

/**
 * Evaluates organic crumpled handmade kozo paper fold relief for a panel.
 * Returns { disp, crease } where disp is radial displacement in meters
 * and crease in [-1, 1] indicates valley crease (-1) vs raised ridge (+1).
 */
function evalPanelPaperFold(
  panelIdx: number,
  panelT: number,
  v: number,
): { disp: number; crease: number } {
  const salt = panelIdx * 11.37 + 3.19;
  const ox = hash01(panelIdx * 2.13 + 0.7) * 9.0;
  const oy = hash01(panelIdx * 5.71 + 1.9) * 9.0;

  // Domain-warped coordinates for diagonal paper tension creases
  const wx = panelT * 3.1 + ox + (valueNoise2D(panelT * 2.6 + ox, v * 5.4 + oy, salt + 1.1) - 0.5) * 0.55;
  const wy = v * 7.8 + oy + (valueNoise2D(panelT * 2.8 + ox, v * 5.2 + oy, salt + 4.3) - 0.5) * 0.55;

  // Two rotated ridge octaves creating crisp, natural dry-paper fold planes
  const r1x = wx * 0.819 - wy * 0.574;
  const r1y = wx * 0.574 + wy * 0.819;
  const n1 = valueNoise2D(r1x * 1.15, r1y * 1.15, salt + 7.7);
  const ridge1 = 1.0 - Math.abs(n1 * 2.0 - 1.0); // sharp fold crest
  const fold1 = ridge1 * ridge1 - 0.38;

  const r2x = wx * 0.422 + wy * 0.906;
  const r2y = -wx * 0.906 + wy * 0.422;
  const n2 = valueNoise2D(r2x * 2.35, r2y * 2.35, salt + 15.2);
  const ridge2 = 1.0 - Math.abs(n2 * 2.0 - 1.0);
  const fold2 = ridge2 * ridge2 - 0.42;

  // Fine diagonal corner-pull wrinkles radiating slightly inward from the glued ribs
  const edgeDist = Math.min(panelT, 1.0 - panelT);
  const pullBias = Math.exp(-Math.pow(edgeDist / 0.24, 2));
  const ribWrinkle =
    (valueNoise2D(panelT * 4.5 + ox, v * 16.5 + oy, salt + 23.9) - 0.5) * 0.9 * pullBias;

  const rawFold = fold1 * 0.58 + fold2 * 0.30 + ribWrinkle * 0.26;
  // Strictly pinned to 0 along the two bamboo ribs and top/bottom hoops
  const pinSpan = Math.pow(Math.max(0, Math.sin(panelT * Math.PI)), 0.78);
  const pinVert = smoothstepCpu(0.015, 0.09, v) * (1.0 - smoothstepCpu(0.91, 0.985, v));
  const env = pinSpan * pinVert;

  return {
    disp: rawFold * 0.0085 * env,
    crease: Math.max(-1, Math.min(1, rawFold * 1.85 * env)),
  };
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
  onVertex?: (u: number, v: number) => void,
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
      if (onVertex) onVertex(u, v);
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
    if (onVertex) onVertex(0, 0.5);
    for (let j = 0; j < radialSegs; j++) {
      b.indices.push(startCenter, baseVert + j + 1, baseVert + j);
    }
  }

  if (!closedLoop && capEnd) {
    const endCenter = b.positions.length / 3;
    curve.getPoint(1, _p0);
    b.positions.push(_p0.x, _p0.y, _p0.z);
    b.uvs.push(0.5, 1);
    if (onVertex) onVertex(1, 0.5);
    const endRing = baseVert + steps * stride;
    for (let j = 0; j < radialSegs; j++) {
      b.indices.push(endCenter, endRing + j, endRing + j + 1);
    }
  }
}

/**
 * Appends an outward-wound closed circular chamfered split-bamboo hoop centered at (0, yCenter, 0).
 * UVs place cross-section around U and circumference around V so longitudinal bamboo
 * vascular fibers flow horizontally along the bent bamboo ring (matching Macro target).
 */
function appendSplitBambooHoop(
  b: MeshBuilder,
  yCenter: number,
  radius: number,
  radialThick: number,
  height: number,
  segments = 64,
  nodeAngles: readonly number[] = [],
  woodFaceOut?: number[],
  nodeMaskOut?: number[],
): void {
  const hr = radialThick * 0.5;
  const hh = height * 0.5;
  const bevel = Math.min(hr, hh) * 0.25;

  // Profile ordered CCW in (dr, dy) so outward winding is guaranteed:
  // outer face (0..3) has aWoodFace = 1.0 (dark roasted culm skin),
  // inner pith face (4..7) has aWoodFace = 0.0 (warmer honey inner split bamboo).
  const profile: Array<[number, number, number]> = [
    [hr - bevel, -hh, 0.72],
    [hr, -hh + bevel, 1.0],
    [hr, hh - bevel, 1.0],
    [hr - bevel, hh, 0.72],
    [-hr + bevel, hh, 0.22],
    [-hr, hh - bevel, 0.0],
    [-hr, -hh + bevel, 0.0],
    [-hr + bevel, -hh, 0.22],
  ];

  const pCount = profile.length;
  const stride = pCount + 1;
  const baseVert = b.positions.length / 3;

  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const theta = u * Math.PI * 2;
    const sT = Math.sin(theta);
    const cT = Math.cos(theta);

    // Subtle bamboo culm node ring swell & darkened joint scar
    let nodeVal = 0;
    for (let n = 0; n < nodeAngles.length; n++) {
      const diff = Math.atan2(Math.sin(theta - nodeAngles[n]), Math.cos(theta - nodeAngles[n]));
      nodeVal = Math.max(nodeVal, Math.exp(-Math.pow(diff / 0.036, 2)));
    }
    const nodeSwell = nodeVal * 0.0022;

    for (let j = 0; j <= pCount; j++) {
      const [dr, dy, faceWeight] = profile[j % pCount];
      const r = radius + dr + (dr > 0 ? nodeSwell : 0);
      b.positions.push(sT * r, yCenter + dy, cT * r);
      // U = cross-section perimeter, V = hoop circumference (longitudinal grain!)
      b.uvs.push(j / pCount, u * 4.5);
      if (woodFaceOut) woodFaceOut.push(faceWeight);
      if (nodeMaskOut) nodeMaskOut.push(nodeVal);
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
  woodFaceOut?: number[],
  nodeMaskOut?: number[],
  woodFaceVal = 0.35,
): void {
  const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
  const curve = new THREE.CatmullRomCurve3([pA.clone(), mid, pB.clone()]);
  appendSweptTube(
    b,
    curve,
    4,
    radialSegs,
    () => radius,
    true,
    true,
    false,
    () => {
      if (woodFaceOut) woodFaceOut.push(woodFaceVal);
      if (nodeMaskOut) nodeMaskOut.push(0);
    },
  );
}

// ---------------------------------------------------------------------------
// 1. Ovoid Kozo Rice-Paper Dome Geometry (12-Panel Faceted + Crumpled Folds)
// ---------------------------------------------------------------------------

/**
 * Builds the 12-panel ovoid handmade kozo rice-paper canopy matching Concept A's
 * Hero & Macro renders:
 * - Subtle 12-faceted chord-tensioned panels pinned tightly along each meridian bamboo rib
 * - Sculpted multi-scale organic crumpled-paper fold creases & tension wrinkles (unique per panel)
 * - Slight folded-down glowing hem tucked cleanly inside the bottom bamboo mouth hoop
 * - Per-vertex attributes:
 *   - `aRibProximity`: 0 panel center -> 1 pasted bamboo rib seam & mouth/collar rims
 *   - `aWaistMask`: 1 across silk-wrapped lower-mid belly, smoothly tapering to 0
 *   - `aPanelId`: panel index for per-panel unique kozo fiber & crinkle UV offsets
 *   - `aCrinkleRidge`: signed fold curvature (-1 crease valley .. +1 dry paper crest)
 * Strictly outward-wound (`countInwardRadialTriangles === 0`).
 */
export function buildOvoidPaperDomeGeometry(ribCount = 12, heightSegs = 56): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const subPerPanel = 8;
  const colsPerPanel = subPerPanel + 1;

  // Include 1 extra bottom hem row (row = 0) tucked cleanly inside the bottom mouth hoop.
  // Each of the 12 panels owns unshared meridian-seam vertices so `aPanelId` stays strictly
  // constant across every triangle of each panel (zero UV-shift smearing beside bamboo ribs!).
  const totalRows = heightSegs + 2; // row 0 = hem lip, rows 1..heightSegs+1 = v in [0..1]
  const ribProximity: number[] = [];
  const waistMask: number[] = [];
  const panelIdArr: number[] = [];
  const crinkleRidge: number[] = [];

  for (let panelIdx = 0; panelIdx < ribCount; panelIdx++) {
    const panelBaseVert = b.positions.length / 3;
    const panelIdVal = panelIdx / ribCount;
    const a0 = (panelIdx / ribCount) * Math.PI * 2;
    const a1 = ((panelIdx + 1) / ribCount) * Math.PI * 2;

    for (let row = 0; row < totalRows; row++) {
      const isHemRow = row === 0;
      const v = isHemRow ? 0.0 : (row - 1) / heightSegs;
      const baseY = evaluateOvoidLanternY(v);
      const baseR = evaluateOvoidLanternRadius(v);

      const rimProx = isHemRow
        ? 0.92
        : Math.max(
            Math.exp(-Math.pow(v / 0.032, 2)),
            Math.exp(-Math.pow((1.0 - v) / 0.035, 2)),
          );

      const waistRise = smoothstepCpu(0.03, 0.13, v);
      const waistFall = 1.0 - smoothstepCpu(0.54, 0.72, v);
      const wMask = isHemRow ? 0.12 : waistRise * waistFall;

      for (let localIx = 0; localIx <= subPerPanel; localIx++) {
        const panelT = localIx / subPerPanel;
        const u = (panelIdx + panelT) / ribCount;
        const theta = a0 + (a1 - a0) * panelT;
        const sT = Math.sin(theta);
        const cT = Math.cos(theta);

        const chordX = Math.sin(a0) * baseR * (1.0 - panelT) + Math.sin(a1) * baseR * panelT;
        const chordZ = Math.cos(a0) * baseR * (1.0 - panelT) + Math.cos(a1) * baseR * panelT;
        const circX = sT * baseR;
        const circZ = cT * baseR;

        // Smoothly transition from 100% circular at bottom mouth hoop & top collar into
        // subtly 12-faceted chord-tensioned paper across the ovoid belly
        const chordWeight =
          0.78 * smoothstepCpu(0.0, 0.14, v) * (1.0 - smoothstepCpu(0.86, 1.0, v));
        let x = chordX * chordWeight + circX * (1.0 - chordWeight);
        let z = chordZ * chordWeight + circZ * (1.0 - chordWeight);
        let y = baseY;

        let creaseVal = 0;
        if (isHemRow) {
          const hemWave =
            0.0028 * Math.sin(theta * 5.0 + 1.1) +
            0.0018 * Math.cos(theta * 11.0 - 0.7);
          y = LANTERN_BOTTOM_Y - 0.014 + hemWave;
          const hemR = LANTERN_MOUTH_RADIUS - 0.0045;
          x = sT * hemR;
          z = cT * hemR;
        } else {
          const mouthTuck = (1.0 - smoothstepCpu(0.0, 0.045, v)) * 0.004;
          const gentleBillow = 0.0044 * Math.sin(panelT * Math.PI) * Math.sin(v * Math.PI);
          const fold = evalPanelPaperFold(panelIdx, panelT, v);
          const totalDisp = gentleBillow + fold.disp - mouthTuck;
          x += sT * totalDisp;
          z += cT * totalDisp;
          creaseVal = fold.crease;
        }

        b.positions.push(x, y, z);
        b.uvs.push(u, isHemRow ? -0.012 : v);

        const seamDist = Math.min(panelT, 1.0 - panelT);
        const meridianProx = Math.exp(-Math.pow(seamDist / 0.11, 2));
        ribProximity.push(Math.min(1.0, Math.max(meridianProx, rimProx)));
        waistMask.push(wMask);
        panelIdArr.push(panelIdVal);
        crinkleRidge.push(creaseVal);
      }
    }

    // Strictly outward winding within panel: (+localIx) x (+row) = outward radial normal
    for (let row = 0; row < totalRows - 1; row++) {
      for (let localIx = 0; localIx < subPerPanel; localIx++) {
        const a = panelBaseVert + row * colsPerPanel + localIx;
        const bIdx = a + 1;
        const c = bIdx + colsPerPanel;
        const d = a + colsPerPanel;
        b.indices.push(a, bIdx, c, a, c, d);
      }
    }
  }

  // Recessed top kozo crown disc seated cleanly BELOW the top wooden cross-tie struts
  // (y = LANTERN_TOP_Y - 0.008 -> -0.0045 at center) with planar (x, z) UVs and constant
  // aPanelId so zero cross-tie z-fighting and zero polar UV pinching occur in macro!
  const capSegs = 48;
  const capRadius = LANTERN_COLLAR_RADIUS - 0.002;
  const capCenterY = LANTERN_TOP_Y - 0.0055;
  const capRimY = LANTERN_TOP_Y - 0.0085;
  const capCenterIdx = b.positions.length / 3;
  b.positions.push(0, capCenterY, 0);
  b.uvs.push(0.5, 0.5);
  ribProximity.push(0.18);
  waistMask.push(0.0);
  panelIdArr.push(0.25);
  crinkleRidge.push(0.0);

  const capRingStart = b.positions.length / 3;
  for (let i = 0; i <= capSegs; i++) {
    const theta = (i / capSegs) * Math.PI * 2;
    const px = Math.sin(theta) * capRadius;
    const pz = Math.cos(theta) * capRadius;
    b.positions.push(px, capRimY, pz);
    b.uvs.push(0.5 + px * 1.45, 0.5 + pz * 1.45);
    ribProximity.push(0.86);
    waistMask.push(0.0);
    panelIdArr.push(0.25);
    crinkleRidge.push(0.0);
  }
  for (let i = 0; i < capSegs; i++) {
    // Outward +Y winding: (rim[i] - center) x (rim[i+1] - center) = +Y
    b.indices.push(capCenterIdx, capRingStart + i, capRingStart + i + 1);
  }

  return finalizeBuilder(b, {
    aRibProximity: { data: new Float32Array(ribProximity), itemSize: 1 },
    aWaistMask: { data: new Float32Array(waistMask), itemSize: 1 },
    aPanelId: { data: new Float32Array(panelIdArr), itemSize: 1 },
    aCrinkleRidge: { data: new Float32Array(crinkleRidge), itemSize: 1 },
  });
}

// ---------------------------------------------------------------------------
// 2. Ovoid Split-Bamboo Frame Geometry (Exact Analytical Flat-Band Ribs & Hoops)
// ---------------------------------------------------------------------------

/**
 * Appends one crisp, twist-free, flat-banded chamfered split-bamboo meridian rib
 * built in the exact analytical ovoid normal/binormal frame from bottom mouth to top collar,
 * complete with two subtle organic bamboo culm node knuckles.
 */
function appendMeridianSplitBambooRib(
  b: MeshBuilder,
  theta: number,
  ribIndex: number,
  steps: number,
  woodFaceOut: number[],
  nodeMaskOut: number[],
): void {
  const sT = Math.sin(theta);
  const cT = Math.cos(theta);
  // Azimuth tangent (CCW around +Y): B_az = (+cos(theta), 0, -sin(theta))
  // Wait: verify cross(T_meridian, N_out) orientation!
  // T_meridian = (sT * tr, ty, cT * tr), N_out = (sT * nr, ny, cT * nr)
  // T_meridian x N_out:
  //   x = ty * cT * nr - cT * tr * ny = cT * (ty * nr - tr * ny) = cT * (ty^2 + tr^2) = +cT
  //   y = cT * tr * sT * nr - sT * tr * cT * nr = 0
  //   z = sT * tr * ny - ty * sT * nr = -sT * (ty^2 + tr^2) = -sT
  // So B_az = (+cT, 0, -sT) satisfies (N_out, B_az, T_meridian) right-handed:
  //   N_out x B_az = (ny * (-sT) - 0, cT * cT * nr - (-sT) * sT * nr, 0 - ny * cT) -- wait:
  //   actually T_meridian x N_out = B_az => N_out x B_az = T_meridian!
  // Check outward winding: along loop j increasing CCW in (B_az, N_out) plane?
  // If p(i, j) = P(i) + wb * B_az + wn * N_out:
  //   dP/di = +T_meridian, dP/dj = (dwb * B_az + dwn * N_out).
  //   For triangle (a, bIdx, c) where bIdx = a + 1 (increasing j) and d = a + stride (increasing i):
  //   (dP/dj) x (dP/di) = (dwb * B_az + dwn * N_out) x T_meridian
  //     Since B_az x T_meridian = N_out and N_out x T_meridian = -B_az:
  //     = dwb * N_out - dwn * B_az!
  //   That points OUTWARD when (wb, wn) traverses CLOCKWISE in (wb, wn) — i.e., starting at
  //   (-hw, +ht) -> (+hw, +ht) -> (+hw, -ht) -> (-hw, -ht)!
  // Let's verify on top outer face: wb increases (+dwb > 0, dwn = 0) => (dP/dj) x (dP/di) = +dwb * N_out = +N_out (OUTWARD!!).
  // Verify on right side (+hw): wn decreases (dwb = 0, dwn < 0) => -dwn * B_az = +B_az (OUTWARD!!).
  // Verify on inner face (-ht): wb decreases (dwb < 0, dwn = 0) => dwb * N_out = -N_out (OUTWARD toward center!!).
  // Verify on left side (-hw): wn increases (dwb = 0, dwn > 0) => -dwn * B_az = -B_az (OUTWARD!!).
  // 100% mathematically verified outward!

  const hw = 0.0112; // half-width of flat split-bamboo strip (total width ~2.24 cm)
  const ht = 0.0068; // half-thickness radially (total thickness ~1.36 cm)
  const bev = 0.0022;

  // Clockwise in (wb, wn): [wb, wn, woodFace (1 = dark roasted outer rind, 0 = inner pith)]
  const profile: Array<[number, number, number]> = [
    [-hw + bev, ht, 1.0],
    [hw - bev, ht, 1.0],
    [hw, ht - bev, 0.82],
    [hw, -ht + bev, 0.45],
    [hw - bev, -ht, 0.15],
    [-hw + bev, -ht, 0.15],
    [-hw, -ht + bev, 0.45],
    [-hw, ht - bev, 0.82],
  ];

  const pCount = profile.length;
  const stride = pCount + 1;
  const baseVert = b.positions.length / 3;

  // Two subtle bamboo culm node stations along each meridian rib
  const nodeV1 = 0.25 + hash01(ribIndex * 3.17 + 0.5) * 0.15;
  const nodeV2 = 0.65 + hash01(ribIndex * 7.41 + 2.1) * 0.16;

  // Flush seating: inner face sits right at baseR - 0.0018 (tucked flush into paper seam),
  // outer roasted rind sits proudly at baseR + 0.0118 (zero gap, crisp dark framing).
  const centerRadialOffset = 0.0050;

  for (let i = 0; i <= steps; i++) {
    const v = i / steps;
    const f = evaluateOvoidProfileFrame(v);

    // Softer organic double-ring bamboo culm knuckle (subtle ridge swell, never a sharp hard seam)
    const n1 = Math.exp(-Math.pow((v - nodeV1) / 0.022, 2));
    const n2 = Math.exp(-Math.pow((v - nodeV2) / 0.022, 2));
    const nodeVal = Math.max(n1, n2);
    const nodeSwell = nodeVal * 0.0012;

    // Slight graceful width taper at top collar & bottom mouth socket
    const endTaper = 0.86 + 0.14 * Math.sin(v * Math.PI);

    const cx = sT * (f.r + centerRadialOffset);
    const cy = f.y;
    const cz = cT * (f.r + centerRadialOffset);

    const nx = sT * f.nr;
    const ny = f.ny;
    const nz = cT * f.nr;

    const bx = cT;
    const bz = -sT;

    for (let j = 0; j <= pCount; j++) {
      const [wb0, wn0, faceW] = profile[j % pCount];
      const wb = wb0 * endTaper + (wb0 > 0 ? nodeSwell * 0.5 : -nodeSwell * 0.5);
      const wn = wn0 + (wn0 > 0 ? nodeSwell : 0);

      b.positions.push(
        cx + bx * wb + nx * wn,
        cy + ny * wn,
        cz + bz * wb + nz * wn,
      );
      // U = cross-section, V = along rib length (vascular grain follows the rib!)
      b.uvs.push(j / pCount, v * 2.2);
      woodFaceOut.push(faceW);
      nodeMaskOut.push(nodeVal);
    }
  }

  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < pCount; j++) {
      const a = baseVert + i * stride + j;
      const bIdx = a + 1;
      const d = a + stride;
      const c = d + 1;
      b.indices.push(a, bIdx, c, a, c, d);
    }
  }

  // Cap bottom end (i = 0, facing -T_meridian)
  {
    const f0 = evaluateOvoidProfileFrame(0);
    const capCenter = b.positions.length / 3;
    b.positions.push(
      sT * (f0.r + centerRadialOffset),
      f0.y,
      cT * (f0.r + centerRadialOffset),
    );
    b.uvs.push(0.5, 0);
    woodFaceOut.push(0.2);
    nodeMaskOut.push(0);
    for (let j = 0; j < pCount; j++) {
      b.indices.push(capCenter, baseVert + j + 1, baseVert + j);
    }
  }

  // Cap top end (i = steps, facing +T_meridian)
  {
    const f1 = evaluateOvoidProfileFrame(1);
    const capCenter = b.positions.length / 3;
    b.positions.push(
      sT * (f1.r + centerRadialOffset),
      f1.y,
      cT * (f1.r + centerRadialOffset),
    );
    b.uvs.push(0.5, 2.2);
    woodFaceOut.push(0.2);
    nodeMaskOut.push(0);
    const topRing = baseVert + steps * stride;
    for (let j = 0; j < pCount; j++) {
      b.indices.push(capCenter, topRing + j, topRing + j + 1);
    }
  }
}

/**
 * Builds the 12-rib dark-roasted split-bamboo lantern skeleton matching Concept A:
 * - Flat-band circular bent split-bamboo mouth hoop at v = 0 (y = -0.78) with horizontal
 *   vascular grain and natural bamboo culm node joints (matching Macro target!)
 * - Fine crossed burner support sticks + central ember ring inside the mouth hoop
 * - Top wooden/bamboo collar ring at v = 1 (y = +0.84) with stepped lip & cross-tie hub
 * - 12 twist-free, flat-banded chamfered meridian split-bamboo ribs with subtle culm nodes
 * Per-vertex attributes: `aWoodFace` (outer roasted rind vs warm inner pith), `aNodeMask`.
 */
export function buildOvoidBambooFrameGeometry(ribCount = 12): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const woodFace: number[] = [];
  const nodeMask: number[] = [];

  // 2a. Bottom flat-band circular bent split-bamboo mouth hoop (hero detail in Macro!)
  // Inner wall sits cleanly at r = 0.259 (strictly outside the circular r = 0.2555 paper hem!)
  // Two bamboo culm node scars: left-front (theta = -0.36, matching Macro!) & rear (theta = 2.72)
  appendSplitBambooHoop(
    b,
    LANTERN_BOTTOM_Y,
    LANTERN_MOUTH_RADIUS + 0.0095,
    0.021,
    0.048,
    72,
    [-0.36, 2.72],
    woodFace,
    nodeMask,
  );

  // 2b. Fine crossed burner support sticks + recessed ember saucer cup inside mouth hoop
  const innerMouthR = LANTERN_MOUTH_RADIUS + 0.007; // embeds flush into hoop inner wall (zero gap!)
  const burnerY = LANTERN_BOTTOM_Y + 0.006;
  appendStrut(
    b,
    new THREE.Vector3(-innerMouthR, burnerY, 0),
    new THREE.Vector3(innerMouthR, burnerY, 0),
    0.0038,
    6,
    woodFace,
    nodeMask,
    0.3,
  );
  appendStrut(
    b,
    new THREE.Vector3(0, burnerY, -innerMouthR),
    new THREE.Vector3(0, burnerY, innerMouthR),
    0.0038,
    6,
    woodFace,
    nodeMask,
    0.3,
  );
  appendSplitBambooHoop(b, burnerY, 0.044, 0.009, 0.012, 32, [], woodFace, nodeMask);
  // Small turned ember/incense holder dish seated cleanly inside the central burner ring
  appendSplitBambooHoop(b, burnerY - 0.002, 0.022, 0.024, 0.005, 24, [], woodFace, nodeMask);

  // 2c. Refined sleek top wooden/bamboo collar ring + proud cross-tie hub (zero cap clipping!)
  appendSplitBambooHoop(
    b,
    LANTERN_TOP_Y,
    LANTERN_COLLAR_RADIUS + 0.0055,
    0.016,
    0.024,
    56,
    [0.85, -2.25],
    woodFace,
    nodeMask,
  );
  const innerCollarR = LANTERN_COLLAR_RADIUS + 0.002;
  const topStrutY = LANTERN_TOP_Y + 0.003;
  appendStrut(
    b,
    new THREE.Vector3(-innerCollarR, topStrutY, 0),
    new THREE.Vector3(innerCollarR, topStrutY, 0),
    0.0046,
    8,
    woodFace,
    nodeMask,
    0.65,
  );
  appendStrut(
    b,
    new THREE.Vector3(0, topStrutY, -innerCollarR),
    new THREE.Vector3(0, topStrutY, innerCollarR),
    0.0046,
    8,
    woodFace,
    nodeMask,
    0.65,
  );

  // 2d. 12 crisp, twist-free flat-banded meridian split-bamboo ribs
  for (let k = 0; k < ribCount; k++) {
    const theta = (k / ribCount) * Math.PI * 2;
    appendMeridianSplitBambooRib(b, theta, k, 36, woodFace, nodeMask);
  }

  return finalizeBuilder(b, {
    aWoodFace: { data: new Float32Array(woodFace), itemSize: 1 },
    aNodeMask: { data: new Float32Array(nodeMask), itemSize: 1 },
  });
}

// ---------------------------------------------------------------------------
// 2b. Twisted 3-Ply Suspension Cord & Looped Overhand Collar Knot Geometry
// ---------------------------------------------------------------------------

/**
 * Builds the standalone matte dark raw-umber twisted 3-ply hemp/silk suspension cord
 * and tied 3D collar knot right above the top wooden collar (matching Hero render).
 * Strictly outward-wound closed solid (`computeSignedVolume > 0`).
 */
export function buildSuspensionCordGeometry(): THREE.BufferGeometry {
  const b = createMeshBuilder();

  // Main 3-ply helically fluted vertical suspension cord rising out of frame
  const cordCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, LANTERN_TOP_Y + 0.008, 0),
    new THREE.Vector3(0.0015, LANTERN_TOP_Y + 0.06, 0.001),
    new THREE.Vector3(-0.001, (LANTERN_TOP_Y + SUSPENSION_CORD_TOP_Y) * 0.5, -0.0008),
    new THREE.Vector3(0, SUSPENSION_CORD_TOP_Y, 0),
  ]);
  appendSweptTube(
    b,
    cordCurve,
    80,
    12,
    (u, ang) => {
      // Crisp 3-ply helical cord silhouette along the entire length
      const ply = 0.0012 * Math.cos(ang * 3.0 + u * 210.0);
      return 0.0056 + ply;
    },
    true,
    true,
    false,
  );

  // Tied lashing hitch & overhand collar knot right above the top cross-tie hub
  const knotPts1: THREE.Vector3[] = [];
  const knotSegs = 20;
  for (let i = 0; i < knotSegs; i++) {
    const t = (i / knotSegs) * Math.PI * 2;
    const r = 0.0125 + 0.003 * Math.cos(t * 2.0);
    knotPts1.push(
      new THREE.Vector3(
        Math.cos(t) * r,
        LANTERN_TOP_Y + 0.017 + Math.sin(t) * 0.0075,
        Math.sin(t) * r * 0.88,
      ),
    );
  }
  const knotLoop1 = new THREE.CatmullRomCurve3(knotPts1, true, 'centripetal');
  appendSweptTube(
    b,
    knotLoop1,
    24,
    8,
    (u, ang) => 0.0054 + 0.0009 * Math.cos(ang * 3.0 + u * 36.0),
    false,
    false,
    true,
  );

  // Second diagonal clove-hitch knot turn locking the vertical cord onto the hub
  const knotPts2: THREE.Vector3[] = [];
  for (let i = 0; i < knotSegs; i++) {
    const t = (i / knotSegs) * Math.PI * 2;
    const r = 0.011 + 0.0025 * Math.sin(t * 2.0);
    knotPts2.push(
      new THREE.Vector3(
        Math.cos(t + 1.1) * r * 0.92,
        LANTERN_TOP_Y + 0.032 + Math.cos(t) * 0.0065,
        Math.sin(t + 1.1) * r,
      ),
    );
  }
  const knotLoop2 = new THREE.CatmullRomCurve3(knotPts2, true, 'centripetal');
  appendSweptTube(
    b,
    knotLoop2,
    24,
    8,
    (u, ang) => 0.0050 + 0.0008 * Math.cos(ang * 3.0 + u * 36.0),
    false,
    false,
    true,
  );

  // Short neatly trimmed cord tail tucking beside the knot
  const tailCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.005, LANTERN_TOP_Y + 0.025, 0.004),
    new THREE.Vector3(0.015, LANTERN_TOP_Y + 0.018, 0.010),
    new THREE.Vector3(0.019, LANTERN_TOP_Y + 0.006, 0.012),
  ]);
  appendSweptTube(b, tailCurve, 8, 6, (u) => 0.0042 * (1.0 - u * 0.35), true, true, false);

  return finalizeBuilder(b);
}

// ---------------------------------------------------------------------------
// 3. Crisscross Raw-Silk Wrap Bands Geometry (340 Hair-Fine Taut Skein Filaments)
// ---------------------------------------------------------------------------

/**
 * Evaluates a point on the convex rib-bridging silk envelope at angle `theta`,
 * height parameter `v`, and radial layer offset `layerOffset`.
 * Uses 82% straight chord between adjacent bamboo ribs + 18% gentle circular arc
 * so taut silk skeins span crisply from rib to rib while clearing the faceted paper dome!
 */
function evaluateSilkEnvelopePoint(
  theta: number,
  v: number,
  layerOffset: number,
  ribCount: number,
  out: THREE.Vector3,
): THREE.Vector3 {
  // C-infinity smooth rib-bridging envelope: peaks flush over every bamboo rib
  // (cos = +1) and dips tautly across each panel chord (cos = -1) with zero C0 kinks
  const rBase = evaluateOvoidLanternRadius(v) + 0.0138 + layerOffset;
  const ribWave = 0.5 + 0.5 * Math.cos(theta * ribCount); // 1 at ribs, 0 at panel center
  const r = rBase * (0.972 + 0.028 * ribWave);
  const y = evaluateOvoidLanternY(v);

  out.set(Math.sin(theta) * r, y, Math.cos(theta) * r);
  return out;
}

interface SilkSwathSpec {
  vCenter: number;
  vAmp: number;
  phase: number;
  halfWidth: number;
  layerBase: number;
}

const SILK_WRAP_SWATHS: readonly SilkSwathSpec[] = [
  { vCenter: 0.29, vAmp: 0.235, phase: 0.32, halfWidth: 0.044, layerBase: 0.0005 },
  { vCenter: 0.31, vAmp: -0.245, phase: 1.78, halfWidth: 0.046, layerBase: 0.0028 },
  { vCenter: 0.35, vAmp: 0.220, phase: 3.12, halfWidth: 0.042, layerBase: 0.0052 },
  { vCenter: 0.24, vAmp: -0.185, phase: 4.48, halfWidth: 0.038, layerBase: 0.0076 },
  { vCenter: 0.36, vAmp: -0.215, phase: 0.88, halfWidth: 0.040, layerBase: 0.0102 },
];

function evaluateSwathV(sw: SilkSwathSpec, theta: number, vShift = 0): number {
  const rawV = sw.vCenter + vShift + sw.vAmp * Math.sin(theta - sw.phase);
  return Math.max(0.036, Math.min(0.615, rawV));
}

function evaluateSwathDvDtheta(sw: SilkSwathSpec, theta: number): number {
  return sw.vAmp * Math.cos(theta - sw.phase);
}

/**
 * Builds 340 hair-fine taut raw-silk cocoon filaments organized into 5 coherent
 * crisscrossing diagonal geodesic skeins wrapped tightly around the lower-mid
 * lantern belly and waist (`v in [0.036, 0.615]`), spanning smoothly from rib to rib
 * with high macro-grade longitudinal & radial subdivision (zero polygon kinks).
 *
 * Per-vertex attributes:
 * - `aStrandT`: 0..1 along strand
 * - `aStrandSeed`: deterministic 0..1 per strand (governs champagne..gold..amber filament tone)
 * - `aUnwindOrder`: 0..1 (outer/lower swaths unwind earlier as `uUnravel` rises)
 * - `aSkeinId`: 0..1 skein group index for subtle layered shading separation
 */
export function buildSilkWrapBandsGeometry(strandCount = 340): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const strandTList: number[] = [];
  const strandSeedList: number[] = [];
  const unwindOrderList: number[] = [];
  const skeinIdList: number[] = [];

  const swaths = SILK_WRAP_SWATHS;
  const ribCount = 12;
  const stepsPerStrand = 120; // 10 steps per rib span (3 deg/step) -> zero chord angles even at loupe zoom
  const radialSegs = 4; // smooth cylindrical Phong normals with lean vertex budget
  const tmpPt = new THREE.Vector3();

  for (let k = 0; k < strandCount; k++) {
    const seed = hash01(k * 1.37 + 0.19);
    const seed2 = hash01(k * 2.91 + 7.31);
    const seed3 = hash01(k * 5.13 + 13.7);
    const seed4 = hash01(k * 8.63 + 2.41);

    const skeinIdx = k % swaths.length;
    const sw = swaths[skeinIdx];

    const rankInSkein = Math.floor(k / swaths.length);
    const perSkein = Math.max(1, Math.ceil(strandCount / swaths.length));
    const ribbonU = rankInSkein / Math.max(1, perSkein - 1);
    const centeredU = (ribbonU - 0.5) * 2.0;
    const curvedU = Math.sign(centeredU) * Math.pow(Math.abs(centeredU), 1.18);
    const bundleJitter = (seed - 0.5) * 0.011 + Math.sin(ribbonU * Math.PI * 5.0) * 0.004;
    const vShift = curvedU * sw.halfWidth + bundleJitter;
    const vAmp = sw.vAmp * (0.95 + seed2 * 0.10);
    const phase = sw.phase + (seed3 - 0.5) * 0.12;

    const layerNorm = skeinIdx / Math.max(1, swaths.length - 1);
    const layerOffset = sw.layerBase + seed4 * 0.0020;
    const unwindOrder = Math.max(
      0,
      Math.min(1, (1.0 - layerNorm) * 0.62 + (sw.vCenter - 0.15) * 0.65 + (seed2 - 0.5) * 0.12),
    );

    // 48 rib-aligned sample knots (4 per rib span) so every rib crossing is exact & kink-free
    const knotCount = 48;
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < knotCount; i++) {
      const t = i / knotCount;
      const theta = t * Math.PI * 2;
      const rawV =
        sw.vCenter +
        vShift +
        vAmp * Math.sin(theta - phase) +
        0.003 * Math.sin(theta * 2.0 + seed * 6.28);
      const v = Math.max(0.036, Math.min(0.615, rawV));
      evaluateSilkEnvelopePoint(theta, v, layerOffset, ribCount, tmpPt);
      pts.push(tmpPt.clone());
    }

    const curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal');
    // Delicate gossamer spun-silk gauge (0.00056..0.00110) — reads as fine silk, never plastic wire
    const filamentRadius = 0.00056 + Math.pow(seed, 1.35) * 0.00054;
    const skeinNorm = skeinIdx / Math.max(1, swaths.length - 1);

    appendSweptTube(
      b,
      curve,
      stepsPerStrand,
      radialSegs,
      (u) => filamentRadius * (0.92 + 0.16 * Math.sin(u * Math.PI * 12.0 + seed * 6.28)),
      false,
      false,
      true,
      (u) => {
        strandTList.push(u);
        strandSeedList.push(seed);
        unwindOrderList.push(unwindOrder);
        skeinIdList.push(skeinNorm);
      },
    );
  }

  return finalizeBuilder(b, {
    aStrandT: { data: new Float32Array(strandTList), itemSize: 1 },
    aStrandSeed: { data: new Float32Array(strandSeedList), itemSize: 1 },
    aUnwindOrder: { data: new Float32Array(unwindOrderList), itemSize: 1 },
    aSkeinId: { data: new Float32Array(skeinIdList), itemSize: 1 },
  });
}

// ---------------------------------------------------------------------------
// 3b. Translucent Inner Gossamer Silk-Veil Band Geometry (Micro-Fibril Sheen)
// ---------------------------------------------------------------------------

/**
 * Builds a gossamer translucent inner micro-fibril silk veil seated right inside
 * the 340 taut 3D filaments (`v in [0.05, 0.59]`), supplying the delicate pale-gold
 * translucent silk sheen created by thousands of overlapping sub-pixel fibrils in Macro.
 * Strictly outward-wound (`countInwardRadialTriangles === 0`).
 */
export function buildSilkGossamerVeilGeometry(ribCount = 12, heightSegs = 28): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const subPerPanel = 4;
  const totalRadial = ribCount * subPerPanel;
  const cols = totalRadial + 1;
  const vMin = 0.04;
  const vMax = 0.60;
  const tmpPt = new THREE.Vector3();

  for (let iy = 0; iy <= heightSegs; iy++) {
    const tY = iy / heightSegs;
    const v = vMin + tY * (vMax - vMin);
    for (let ix = 0; ix <= totalRadial; ix++) {
      const u = ix / totalRadial;
      const theta = u * Math.PI * 2;
      evaluateSilkEnvelopePoint(theta, v, -0.0012, ribCount, tmpPt);
      b.positions.push(tmpPt.x, tmpPt.y, tmpPt.z);
      b.uvs.push(u, tY);
    }
  }

  for (let iy = 0; iy < heightSegs; iy++) {
    for (let ix = 0; ix < totalRadial; ix++) {
      const a = iy * cols + ix;
      const bIdx = a + 1;
      const c = bIdx + cols;
      const d = a + cols;
      b.indices.push(a, bIdx, c, a, c, d);
    }
  }

  return finalizeBuilder(b);
}

// ---------------------------------------------------------------------------
// 4. Unspooling Gossamer Silk Loops Geometry (38 Hair-Fine Smooth Catenary Arcs)
// ---------------------------------------------------------------------------

/**
 * Builds 38 ultra-fine, smooth catenary gossamer raw-silk loop arcs peeling tangentially
 * off the left flank (lush airy cluster), right/rear flanks (delicate wisps), and draping
 * gracefully beneath the flat bamboo mouth hoop (matching Concept A Hero & Macro!).
 *
 * Per-vertex attributes:
 * - `aLoopFreedom`: 0 at both lantern anchors -> 1 at furthest floating apex
 * - `aLoopSeed`: deterministic 0..1 per loop for staggered GPU billowing & wind sway
 */
export function buildUnspoolingSilkLoopsGeometry(loopCount = 38): THREE.BufferGeometry {
  const b = createMeshBuilder();
  const freedomList: number[] = [];
  const seedList: number[] = [];

  const stepsPerLoop = 88; // butter-smooth catenary curves even at loupe zoom under the mouth
  const radialSegs = 4;
  const ribCount = 12;
  const p0 = new THREE.Vector3();
  const p4 = new THREE.Vector3();

  const leftCount = Math.round(loopCount * 0.42); // ~16 lush concentric left-flank loops
  const mouthCount = Math.round(loopCount * 0.29); // ~11 graceful under-mouth drooping loops

  for (let k = 0; k < loopCount; k++) {
    const seed = hash01(k * 3.17 + 1.13);
    const seed2 = hash01(k * 4.73 + 5.91);
    const seed3 = hash01(k * 7.19 + 2.41);
    const seed4 = hash01(k * 11.03 + 4.27);

    const swStart = SILK_WRAP_SWATHS[k % SILK_WRAP_SWATHS.length];
    const swEnd = SILK_WRAP_SWATHS[(k + (k % 2 === 0 ? 0 : 2)) % SILK_WRAP_SWATHS.length];
    const vShiftStart = (seed - 0.5) * swStart.halfWidth * 1.15;
    const vShiftEnd = (seed2 - 0.5) * swEnd.halfWidth * 1.15;

    let aStart: number;
    let aEnd: number;
    let vStart: number;
    let vEnd: number;
    let apexRAdd: number;
    let apexYShift: number;

    if (k < leftCount) {
      // Left-flank billowing gossamer halo anchored strictly inside the woven silk swaths
      aStart = -Math.PI * 0.5 + (seed - 0.5) * 0.85;
      const spanSign = k % 2 === 0 ? 1 : -1;
      aEnd = aStart + spanSign * (0.48 + seed2 * 0.56);
      vStart = evaluateSwathV(swStart, aStart, vShiftStart);
      vEnd = evaluateSwathV(swEnd, aEnd, vShiftEnd);
      apexRAdd = 0.15 + Math.pow(seed3, 0.85) * 0.27;
      apexYShift = -(0.08 + seed4 * 0.16);
    } else if (k < leftCount + mouthCount) {
      // Under-mouth drooping catenary loops tucked cleanly inside the bottom bamboo hoop rim
      const localIdx = k - leftCount;
      aStart = -0.85 + (localIdx / Math.max(1, mouthCount)) * Math.PI * 2 + (seed - 0.5) * 0.36;
      aEnd = aStart + 0.92 + seed2 * 0.82;
      vStart = 0.012 + seed * 0.018; // seated flush inside bottom bamboo mouth hoop
      vEnd = 0.012 + seed2 * 0.018;
      apexRAdd = 0.06 + seed3 * 0.15;
      apexYShift = -(0.12 + seed4 * 0.20);
    } else {
      // Right-flank & rear airy unspooling wisps anchored strictly inside the woven silk swaths
      const localIdx = k - leftCount - mouthCount;
      const rem = Math.max(1, loopCount - leftCount - mouthCount);
      const isRight = localIdx < Math.ceil(rem * 0.65);
      const baseAng = isRight
        ? Math.PI * 0.5 + (seed - 0.5) * 0.85
        : Math.PI + (seed - 0.5) * 1.45;
      aStart = baseAng;
      aEnd = aStart + (seed2 > 0.5 ? 1 : -1) * (0.42 + seed3 * 0.52);
      vStart = evaluateSwathV(swStart, aStart, vShiftStart);
      vEnd = evaluateSwathV(swEnd, aEnd, vShiftEnd);
      apexRAdd = 0.13 + seed3 * 0.24;
      apexYShift = -(0.05 + seed4 * 0.14);
    }

    // Tuck both ends flush inside the woven silk band / mouth hoop (zero floating blunt tips!)
    evaluateSilkEnvelopePoint(aStart, vStart, swStart.layerBase - 0.0012, ribCount, p0);
    evaluateSilkEnvelopePoint(aEnd, vEnd, swEnd.layerBase - 0.0012, ribCount, p4);

    // Tangential departure along the woven swath slope so strands peel organically off the band
    const dirSign = aEnd >= aStart ? 1 : -1;
    const daSpan = Math.abs(aEnd - aStart);
    const dvStart = evaluateSwathDvDtheta(swStart, aStart) * dirSign * Math.min(0.22, daSpan * 0.28);
    const dvEnd = -evaluateSwathDvDtheta(swEnd, aEnd) * dirSign * Math.min(0.22, daSpan * 0.28);

    const aMid = (aStart + aEnd) * 0.5 + (seed - 0.5) * 0.16;
    const vMid = (vStart + vEnd) * 0.5;
    const rMidBase = evaluateOvoidLanternRadius(vMid) + 0.018;
    const yMidBase = evaluateOvoidLanternY(vMid);

    const a1 = aStart + (aMid - aStart) * 0.34;
    const a3 = aMid + (aEnd - aMid) * 0.66;
    const v1 = Math.max(0.01, Math.min(0.62, vStart + dvStart * 0.55));
    const v3 = Math.max(0.01, Math.min(0.62, vEnd + dvEnd * 0.55));
    const r1 = evaluateOvoidLanternRadius(v1) + 0.018 + apexRAdd * 0.42;
    const r3 = evaluateOvoidLanternRadius(v3) + 0.018 + apexRAdd * 0.42;

    const p1 = new THREE.Vector3(
      Math.sin(a1) * r1,
      evaluateOvoidLanternY(v1) + apexYShift * 0.22,
      Math.cos(a1) * r1,
    );
    const p2 = new THREE.Vector3(
      Math.sin(aMid) * (rMidBase + apexRAdd),
      yMidBase + apexYShift,
      Math.cos(aMid) * (rMidBase + apexRAdd),
    );
    const p3 = new THREE.Vector3(
      Math.sin(a3) * r3,
      evaluateOvoidLanternY(v3) + apexYShift * 0.28,
      Math.cos(a3) * r3,
    );

    const curve = new THREE.CatmullRomCurve3([p0.clone(), p1, p2, p3, p4.clone()], false, 'centripetal');
    // Hair-fine gossamer unspooling filaments with smoothly tapered anchored roots
    const loopRadius = 0.00050 + Math.pow(seed, 1.25) * 0.00046;

    appendSweptTube(
      b,
      curve,
      stepsPerLoop,
      radialSegs,
      (u) => {
        // Pinch-taper anchored roots cleanly to hair-thin points inside the silk band (zero blunt tips!)
        const taper = Math.pow(Math.max(0, Math.sin(u * Math.PI)), 0.45);
        return Math.max(0.00010, loopRadius * taper);
      },
      false,
      false,
      false,
      (u) => {
        const freedom = Math.pow(Math.max(0, Math.sin(u * Math.PI)), 1.35);
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
// 5. Traditional Origami Butterfly Geometry (Yoshizawa-Style Multi-Ply Folds)
// ---------------------------------------------------------------------------

/**
 * Builds one refined, crisp-folded traditional origami butterfly wing (`side: -1 | 1`)
 * matching the Macro target's unmistakable folded paper architecture:
 * - Tucked central dorsal thorax shoulder pleat (double/triple-ply paper overlap!)
 * - Swept 4-facet forewing with diagonal mountain ridge & inner folded valley pocket
 * - Overlapping 3-facet pointed swallowtail hindwing with crisp central mountain fold
 *   and deep central V-split between left and right tails
 *
 * Non-indexed with razor-sharp crease normals plus per-vertex attributes:
 * - `aPaperPlies` (1.0 single-ply glowing tips .. 2.8 multi-ply dark folded body pleats)
 * - `aCreaseProx` (0 facet interior .. 1 crisp folded mountain/valley paper crease edge)
 */
export function buildOrigamiButterflyWingGeometry(side: -1 | 1): THREE.BufferGeometry {
  const s = side;

  // Vertex table: [x, y, z, u, v, paperPlies, creaseProx]
  const rawVerts: Array<[number, number, number, number, number, number, number]> = [
    // 0: anterior head/thorax pinch apex (triple-ply dark center fold)
    [s * 0.015, 0.045, 0.34, 0.02, 0.84, 2.8, 1.0],
    // 1: dorsal central ridge peak (raised folded pleat)
    [s * 0.020, 0.095, 0.06, 0.04, 0.56, 2.6, 1.0],
    // 2: pinched thorax-abdomen waist valley
    [s * 0.015, -0.015, -0.18, 0.03, 0.36, 2.4, 1.0],
    // 3: shoulder valley crease (where forewing unfolds out from thorax pleat)
    [s * 0.24, -0.045, 0.24, 0.22, 0.74, 2.2, 0.92],
    // 4: forewing outer swept apex tip (crisp single-ply bright glow!)
    [s * 1.24, 0.25, 0.66, 0.98, 0.98, 1.0, 0.18],
    // 5: forewing mid-leading costal ridge
    [s * 0.68, 0.19, 0.50, 0.56, 0.88, 1.25, 0.88],
    // 6: forewing central diagonal mountain-fold apex
    [s * 0.64, 0.02, 0.18, 0.54, 0.60, 1.55, 0.96],
    // 7: forewing outer trailing corner
    [s * 1.00, 0.05, 0.04, 0.84, 0.50, 1.05, 0.22],
    // 8: forewing inner trailing notch (overlapping above hindwing)
    [s * 0.42, -0.035, -0.04, 0.38, 0.44, 1.95, 0.84],
    // 9: hindwing tucked under-shoulder (slightly lower Y for true folded-paper layering!)
    [s * 0.22, -0.065, -0.14, 0.22, 0.36, 2.15, 0.88],
    // 10: hindwing outer shoulder facet corner
    [s * 0.76, 0.02, -0.26, 0.68, 0.28, 1.20, 0.24],
    // 11: hindwing longitudinal mountain crease midpoint
    [s * 0.44, 0.055, -0.44, 0.42, 0.18, 1.35, 0.96],
    // 12: hindwing inner cleft shoulder (forms sharp V-split between twin tails)
    [s * 0.09, -0.035, -0.34, 0.12, 0.24, 1.85, 0.82],
    // 13: hindwing sharp swallowtail point (single-ply glowing translucent paper!)
    [s * 0.40, -0.065, -0.86, 0.36, 0.02, 1.0, 0.20],
  ];

  // 14 crisp origami triangles per wing (8 forewing + 6 hindwing)
  const rightTris: Array<[number, number, number]> = [
    // Forewing folded shoulder & swept costal facets
    [0, 5, 3],
    [0, 3, 1],
    [5, 4, 6],
    [3, 5, 6],
    [4, 7, 6],
    [1, 3, 6],
    [1, 6, 8],
    [6, 7, 8],
    // Hindwing overlapping layered swallowtail facets
    [1, 8, 9],
    [1, 9, 2],
    [8, 10, 11],
    [8, 11, 9],
    [9, 11, 12],
    [2, 9, 12],
    [10, 13, 11],
    [11, 13, 12],
  ];

  const triCount = rightTris.length;
  const positions = new Float32Array(triCount * 3 * 3);
  const uvs = new Float32Array(triCount * 3 * 2);
  const paperPlies = new Float32Array(triCount * 3);
  const creaseProx = new Float32Array(triCount * 3);

  for (let t = 0; t < triCount; t++) {
    const [i0, i1, i2] = rightTris[t];
    const order = side === 1 ? [i0, i1, i2] : [i0, i2, i1];
    for (let k = 0; k < 3; k++) {
      const rv = rawVerts[order[k]];
      const dst = t * 3 + k;
      positions[dst * 3] = rv[0];
      positions[dst * 3 + 1] = rv[1];
      positions[dst * 3 + 2] = rv[2];
      uvs[dst * 2] = rv[3];
      uvs[dst * 2 + 1] = rv[4];
      paperPlies[dst] = rv[5];
      creaseProx[dst] = rv[6];
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aPaperPlies', new THREE.BufferAttribute(paperPlies, 1));
  geom.setAttribute('aCreaseProx', new THREE.BufferAttribute(creaseProx, 1));
  geom.computeVertexNormals();
  return geom;
}

/**
 * Builds the crisp pinched central dorsal body pleat sitting between the left
 * and right origami wings so the butterfly reads as one continuous folded sheet
 * from every angle (especially in Macro close-ups!).
 */
export function buildOrigamiButterflyBodyGeometry(): THREE.BufferGeometry {
  // Compact folded-paper triangular prism body along x in [-0.042, +0.042]
  const rawVerts: Array<[number, number, number, number, number, number, number]> = [
    [0.0, 0.085, 0.36, 0.05, 0.85, 2.9, 1.0],     // 0: anterior head tip
    [0.0, 0.115, 0.05, 0.05, 0.55, 2.8, 1.0],     // 1: dorsal spine peak
    [0.0, 0.025, -0.30, 0.05, 0.28, 2.6, 1.0],    // 2: posterior abdomen tip
    [0.048, 0.015, 0.20, 0.12, 0.72, 2.5, 0.9],   // 3: right anterior cheek
    [0.056, -0.028, -0.04, 0.14, 0.48, 2.7, 0.9], // 4: right lower keel
    [-0.048, 0.015, 0.20, 0.12, 0.72, 2.5, 0.9],  // 5: left anterior cheek
    [-0.056, -0.028, -0.04, 0.14, 0.48, 2.7, 0.9], // 6: left lower keel
    [0.0, -0.068, 0.04, 0.05, 0.52, 3.0, 1.0],    // 7: ventral folded tuck keel
  ];

  // Closed outward-wound folded origami body facets (12 triangles)
  const tris: Array<[number, number, number]> = [
    // Right flank
    [0, 3, 1],
    [1, 3, 4],
    [1, 4, 2],
    [0, 7, 3],
    [3, 7, 4],
    [4, 7, 2],
    // Left flank
    [0, 1, 5],
    [1, 6, 5],
    [1, 2, 6],
    [0, 5, 7],
    [5, 6, 7],
    [6, 2, 7],
  ];

  const positions = new Float32Array(tris.length * 3 * 3);
  const uvs = new Float32Array(tris.length * 3 * 2);
  const paperPlies = new Float32Array(tris.length * 3);
  const creaseProx = new Float32Array(tris.length * 3);

  for (let t = 0; t < tris.length; t++) {
    const tri = tris[t];
    for (let k = 0; k < 3; k++) {
      const rv = rawVerts[tri[k]];
      const dst = t * 3 + k;
      positions[dst * 3] = rv[0];
      positions[dst * 3 + 1] = rv[1];
      positions[dst * 3 + 2] = rv[2];
      uvs[dst * 2] = rv[3];
      uvs[dst * 2 + 1] = rv[4];
      paperPlies[dst] = rv[5];
      creaseProx[dst] = rv[6];
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aPaperPlies', new THREE.BufferAttribute(paperPlies, 1));
  geom.setAttribute('aCreaseProx', new THREE.BufferAttribute(creaseProx, 1));
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
