import * as THREE from 'three/webgpu';

/**
 * Day 11 — Rust:
 * "Submerged Shipwreck Spike & Osmotic Rusticles" — Forged Iron, Rusticle & Flaking Apparatus
 *
 * Zero per-frame heap allocations. All physics state arrays, HalfFloat buffers,
 * and scratch matrices/vectors are pre-allocated at module or class scope.
 */

export const TANK_OUTER_X = 0.90;
export const TANK_OUTER_Y = 0.89;
export const TANK_OUTER_Z = 0.86;
export const GLASS_THICKNESS = 0.028;

export const TANK_INNER_X = TANK_OUTER_X - GLASS_THICKNESS; // 0.872
export const TANK_INNER_Y_BOT = -TANK_OUTER_Y + GLASS_THICKNESS; // -0.862
export const TANK_INNER_Y_TOP = 0.810;
export const TANK_INNER_Z = TANK_OUTER_Z - GLASS_THICKNESS; // 0.832

export const TANK_BASE_YAW = -0.43;

export const SPIKE_MIN_X = -0.74;
export const SPIKE_MAX_X = 0.59;
export const SPIKE_BASE_Y = 0.155;

// ---------------------------------------------------------------------------
// Deterministic 3D Cartesian Noise (Strictly Zero Polar Star Harmonics)
// ---------------------------------------------------------------------------
function fractCpu(v: number): number {
  return v - Math.floor(v);
}

export function hash31Cpu(x: number, y: number, z: number): number {
  let px = fractCpu(x * 0.1031 + 1.0);
  let py = fractCpu(y * 0.103 + 1.0);
  let pz = fractCpu(z * 0.0973 + 1.0);
  const d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33);
  px += d;
  py += d;
  pz += d;
  return fractCpu((px + py) * pz);
}

export function valueNoise3DCpu(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;

  const ux = fx * fx * (3.0 - 2.0 * fx);
  const uy = fy * fy * (3.0 - 2.0 * fy);
  const uz = fz * fz * (3.0 - 2.0 * fz);

  const n000 = hash31Cpu(ix, iy, iz);
  const n100 = hash31Cpu(ix + 1, iy, iz);
  const n010 = hash31Cpu(ix, iy + 1, iz);
  const n110 = hash31Cpu(ix + 1, iy + 1, iz);
  const n001 = hash31Cpu(ix, iy, iz + 1);
  const n101 = hash31Cpu(ix + 1, iy, iz + 1);
  const n011 = hash31Cpu(ix, iy + 1, iz + 1);
  const n111 = hash31Cpu(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

export function fbm3DCpu(x: number, y: number, z: number): number {
  const c = 0.80901699;
  const s = 0.58778525;
  const v0 = valueNoise3DCpu(x, y, z);

  const x1 = (x * c - z * s) * 2.03;
  const y1 = (y * c + x * s) * 2.03;
  const z1 = (z * c + y * s) * 2.03;
  const v1 = valueNoise3DCpu(x1, y1, z1);

  const x2 = (y1 * c - x1 * s) * 2.07;
  const y2 = (z1 * c + y1 * s) * 2.07;
  const z2 = (x1 * c + z1 * s) * 2.07;
  const v2 = valueNoise3DCpu(x2, y2, z2);

  return v0 * 0.56 + v1 * 0.30 + v2 * 0.14;
}

export const BOLSTER_X = 0.522;

/**
 * Evaluates the hand-forged 18th-century square wrought-iron hull spike spine & cross-section
 * along the shank/bolster coordinate `x` (used by rusticle anchors and spatial queries).
 */
export function evalSpikeSpine(x: number): {
  y: number;
  z: number;
  halfY: number;
  halfZ: number;
  u: number;
} {
  const u = Math.max(0.0, Math.min(1.0, (x - SPIKE_MIN_X) / (SPIKE_MAX_X - SPIKE_MIN_X)));
  const y =
    SPIKE_BASE_Y +
    0.018 * Math.sin(u * Math.PI) -
    0.005 * Math.sin(u * 3.1 * Math.PI + 0.4) +
    0.010 * (u - 0.5);
  const z = 0.042 - 0.165 * u + 0.008 * Math.sin(u * 1.8 * Math.PI);

  let halfY: number;
  let halfZ: number;

  if (x < SPIKE_MIN_X + 0.31) {
    const t = Math.max(0.0, (x - SPIKE_MIN_X) / 0.31);
    halfY = 0.006 + 0.042 * Math.pow(t, 0.80);
    halfZ = 0.013 + 0.037 * Math.pow(t, 0.68);
  } else if (x < BOLSTER_X) {
    const t = (x - (SPIKE_MIN_X + 0.31)) / (BOLSTER_X - (SPIKE_MIN_X + 0.31));
    halfY = 0.048 + 0.025 * t;
    halfZ = 0.050 + 0.023 * t;
  } else {
    const t = Math.min(1.0, (x - BOLSTER_X) / (SPIKE_MAX_X - BOLSTER_X));
    halfY = 0.162 * Math.max(0.0, 1.0 - 0.82 * t);
    halfZ = 0.156 * Math.max(0.0, 1.0 - 0.82 * t);
  }

  return { y, z, halfY, halfZ, u };
}

/**
 * Parametric axial profile `u in [0, 1]` with non-linear ring concentration across the
 * sharp ~90-degree blacksmith bolster undercut, ragged forged flange rim, and 5-blow
 * flat-faceted Rose-Head crown (`0.328` diameter vs `0.066` head thickness — zero stubby dome).
 */
function evalForgedSpikeRing(u: number): {
  x: number;
  y: number;
  z: number;
  halfY: number;
  halfZ: number;
  isHead: boolean;
  isCrownFacet: boolean;
  undercutAO: number;
} {
  let x: number;
  let halfY: number;
  let halfZ: number;
  let isHead = false;
  let isCrownFacet = false;
  let undercutAO = 1.0;

  if (u < 0.21) {
    // 4-flat shipwright chisel point
    const t = u / 0.21;
    x = SPIKE_MIN_X + t * 0.31;
    halfY = 0.006 + 0.042 * Math.pow(t, 0.80);
    halfZ = 0.013 + 0.037 * Math.pow(t, 0.68);
  } else if (u < 0.82) {
    // Long 4-flat anvil-forged square wrought-iron shank
    const t = (u - 0.21) / 0.61;
    x = SPIKE_MIN_X + 0.31 + t * (BOLSTER_X - (SPIKE_MIN_X + 0.31));
    halfY = 0.048 + 0.025 * t;
    halfZ = 0.050 + 0.023 * t;
    if (t > 0.94) undercutAO = 0.68;
  } else if (u < 0.875) {
    // Crisp ~88-degree forged header bolster shoulder (strictly monotonic `dx/du > 0` — zero Z-fighting fold!)
    const t = (u - 0.82) / 0.055;
    x = BOLSTER_X + 0.011 * (0.35 * t + 0.65 * t * t);
    const flare = Math.pow(t, 0.56);
    halfY = 0.073 + 0.091 * flare;
    halfZ = 0.073 + 0.085 * flare;
    isHead = t > 0.25;
    undercutAO = 0.36 + 0.42 * t;
  } else if (u < 0.910) {
    // Forged rose-head outer flange rim lip
    const t = (u - 0.875) / 0.035;
    x = BOLSTER_X + 0.011 + 0.016 * t;
    halfY = 0.164 - 0.006 * t;
    halfZ = 0.158 - 0.006 * t;
    isHead = true;
    undercutAO = 0.86;
  } else if (u < 0.978) {
    // 4 flat angled blacksmith hammer-peened trapezoidal bevel facets
    const t = (u - 0.910) / 0.068;
    x = BOLSTER_X + 0.027 + 0.037 * t;
    const linearSlope = 1.0 - 0.69 * t;
    halfY = 0.158 * linearSlope;
    halfZ = 0.152 * linearSlope;
    isHead = true;
    isCrownFacet = true;
  } else {
    // 5th central flat blacksmith set-hammer strike table closing cleanly at the crown
    const t = (u - 0.978) / 0.022;
    x = BOLSTER_X + 0.064 + 0.004 * t;
    const capClose = Math.max(0.0, 1.0 - t);
    halfY = 0.049 * capClose;
    halfZ = 0.047 * capClose;
    isHead = true;
    isCrownFacet = true;
  }

  const uSpatial = Math.max(0.0, Math.min(1.0, (x - SPIKE_MIN_X) / (SPIKE_MAX_X - SPIKE_MIN_X)));
  const y =
    SPIKE_BASE_Y +
    0.018 * Math.sin(uSpatial * Math.PI) -
    0.005 * Math.sin(uSpatial * 3.1 * Math.PI + 0.4) +
    0.010 * (uSpatial - 0.5);
  const z = 0.042 - 0.165 * uSpatial + 0.008 * Math.sin(uSpatial * 1.8 * Math.PI);

  return { x, y, z, halfY, halfZ, isHead, isCrownFacet, undercutAO };
}

function mergeIndexedGeometries(
  geoms: THREE.BufferGeometry[],
  extraAttrName: string,
  extraItemSize: number
): THREE.BufferGeometry {
  let totalVerts = 0;
  let totalIndices = 0;
  for (const g of geoms) {
    totalVerts += g.attributes.position.count;
    totalIndices += g.index ? g.index.count : 0;
  }

  const posOut = new Float32Array(totalVerts * 3);
  const normOut = new Float32Array(totalVerts * 3);
  const extraOut = new Float32Array(totalVerts * extraItemSize);
  const idxOut = totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

  let vOffset = 0;
  let iOffset = 0;
  for (const g of geoms) {
    const pArr = g.attributes.position.array as Float32Array;
    const nArr = g.attributes.normal.array as Float32Array;
    const eArr = g.attributes[extraAttrName].array as Float32Array;
    const vCount = g.attributes.position.count;

    posOut.set(pArr, vOffset * 3);
    normOut.set(nArr, vOffset * 3);
    extraOut.set(eArr, vOffset * extraItemSize);

    if (g.index) {
      const idxArr = g.index.array;
      for (let k = 0; k < idxArr.length; k++) {
        idxOut[iOffset + k] = idxArr[k] + vOffset;
      }
      iOffset += idxArr.length;
    }

    vOffset += vCount;
    g.dispose();
  }

  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(posOut, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normOut, 3));
  merged.setAttribute(extraAttrName, new THREE.BufferAttribute(extraOut, extraItemSize));
  merged.setIndex(new THREE.BufferAttribute(idxOut, 1));
  return merged;
}

export interface CorrosionSurfaceSample {
  pos: [number, number, number];
  norm: [number, number, number];
  isNail?: boolean;
}

// ---------------------------------------------------------------------------
// 1. Authentic Hand-Forged Square Wrought-Iron Spike + 5-Facet Rose-Head
// ---------------------------------------------------------------------------
export function createForgedShipwreckSpikeGeometry(): {
  geometry: THREE.BufferGeometry;
  flakeSites: CorrosionSurfaceSample[];
} {
  const AXIAL_SEGS = 256;
  const CIRC_SEGS = 80;
  const ringStride = CIRC_SEGS + 1;
  const bodyVerts = (AXIAL_SEGS + 1) * ringStride;

  const pos = new Float32Array(bodyVerts * 3);
  const uvs = new Float32Array(bodyVerts * 2);
  const corr = new Float32Array(bodyVerts * 4);
  const flakeSites: CorrosionSurfaceSample[] = [];

  // 4 asymmetric blacksmith rose-head hammer facet directions in YZ
  const roseFacets: ReadonlyArray<[number, number, number]> = [
    [0.92, 0.39, 0.96],
    [-0.35, 0.94, 0.93],
    [-0.95, -0.31, 0.98],
    [0.32, -0.95, 0.94],
  ];

  for (let i = 0; i <= AXIAL_SEGS; i++) {
    const u = i / AXIAL_SEGS;
    const ring = evalForgedSpikeRing(u);
    const rawX = ring.x;

    // Smooth C1 transitions across bolster (`headBlend`) and crown bevel (`crownBlend`) — zero step-jumps or folded quads!
    const isHead = ring.isHead;
    const rawHeadT = Math.max(0.0, Math.min(1.0, (u - 0.822) / 0.048));
    const headBlend = rawHeadT * rawHeadT * (3.0 - 2.0 * rawHeadT);
    const rawCrownT = Math.max(0.0, Math.min(1.0, (u - 0.888) / 0.036));
    const crownBlend = rawCrownT * rawCrownT * (3.0 - 2.0 * rawCrownT);

    const superExp = 9.0 - 3.6 * headBlend + 1.8 * crownBlend;
    const pPow = 2.0 / superExp;
    const rotAng = 0.6283 * crownBlend;
    const cosA = Math.cos(rotAng);
    const sinA = Math.sin(rotAng);

    for (let j = 0; j <= CIRC_SEGS; j++) {
      const theta = (j / CIRC_SEGS) * Math.PI * 2.0;
      const ct = Math.cos(theta);
      const st = Math.sin(theta);

      // Smoothly rotate the 4 crown hammer facets ~36 deg relative to the square shank
      const cRot = ct * cosA - st * sinA;
      const sRot = st * cosA + ct * sinA;

      const sqY = Math.sign(cRot) * Math.pow(Math.abs(cRot), pPow);
      const sqZ = Math.sign(sRot) * Math.pow(Math.abs(sRot), pPow);

      let sy = sqY * cosA + sqZ * sinA;
      let sz = -sqY * sinA + sqZ * cosA;

      // On the Rose-Head flange & crown, blend in 4 asymmetric planar blacksmith hammer blows smoothly
      if (headBlend > 0.001 && i < AXIAL_SEGS) {
        let maxPlaneProj = 0.0;
        for (let f = 0; f < roseFacets.length; f++) {
          const [fy, fz, fScale] = roseFacets[f];
          const proj = (sy * fy + sz * fz) / fScale;
          if (proj > maxPlaneProj) maxPlaneProj = proj;
        }
        if (maxPlaneProj > 0.84) {
          const clipScale = 0.84 + (maxPlaneProj - 0.84) * 0.14;
          const factor = 1.0 + (clipScale / maxPlaneProj - 1.0) * headBlend;
          sy *= factor;
          sz *= factor;
        }
      }

      const poleScale = i === 0 || i === AXIAL_SEGS ? 0.0 : 1.0;
      let basePx = rawX;
      let basePy = ring.y + sy * ring.halfY * poleScale;
      let basePz = ring.z + sz * ring.halfZ * poleScale;

      // Smooth blacksmith off-square hammer cant (`dx/du > 0` preserved everywhere)
      if (headBlend > 0.001 && poleScale > 0) {
        const cantGate = i === AXIAL_SEGS ? 0.0 : headBlend * Math.min(1.0, (1.0 - u) * 32.0);
        basePx += (sy * 0.0055 - sz * 0.0040) * cantGate;
      }

      // Subtle blacksmith anvil fuller marks along the 4 flat shank faces
      const hammerFuller =
        !isHead && u > 0.18
          ? 0.0020 * Math.sin(rawX * 34.0 + (ct > 0 ? 0.0 : 1.7) + (st > 0 ? 0.8 : 2.5))
          : 0.0;

      // Authentic wrought-iron corrosion relief:
      // 1) Longitudinal silicate slag stringers (deep fibrous wood-like iron grain along X)
      const slagGroove = fbm3DCpu(basePx * 3.6 + 2.1, basePy * 44.0 - 1.3, basePz * 44.0 + 3.7);
      // 2) Sharp stepped spall craters & raised delaminating crust blisters
      const spallField = fbm3DCpu(basePx * 10.5 - 3.4, basePy * 14.5 + 2.8, basePz * 14.5 - 1.9);
      const spallDetail = valueNoise3DCpu(basePx * 28.0 + 1.1, basePy * 32.0, basePz * 32.0);
      const spallCombined = spallField * 0.72 + spallDetail * 0.28;

      // Crisp wrought-iron spall relief + organic upward-creeping rusticle concretion tongues
      const intactPlate = Math.max(0.0, Math.min(1.0, (spallCombined - 0.39) / 0.17));
      const deepPit = Math.max(0.0, 0.38 - spallCombined) * 1.75;
      const dorsalBlister =
        !isHead && sy > 0.22 && spallCombined > 0.60 ? (spallCombined - 0.60) * 0.032 : 0.0;

      // Arris / edge corrosion nibbling (corners of the square bar get chewed irregularly by seawater)
      const cornerProximity = Math.min(Math.abs(sy), Math.abs(sz));
      const edgeNibble =
        !ring.isCrownFacet && cornerProximity > 0.66
          ? Math.max(0.0, fbm3DCpu(basePx * 19.0 + 5.2, basePy * 19.0, basePz * 19.0) - 0.47) * 0.015
          : 0.0;

      // Organic FBM tongues of golden marine concretion creeping up from the lower belly onto the flanks
      // (eliminates any straight horizontal belly ruler line and welds rusticles onto the iron shank!)
      const axialRustZone =
        Math.max(0.0, Math.min(1.0, (u - 0.10) / 0.08)) *
        Math.max(0.0, Math.min(1.0, (0.88 - u) / 0.08));
      const creepNoise = fbm3DCpu(basePx * 8.2 + 1.7, basePy * 6.5 - 2.1, basePz * 9.0 + 0.8);
      const creepFine = valueNoise3DCpu(basePx * 21.0, basePy * 19.0 + 3.0, basePz * 21.0);
      const creepField = -sy * 0.72 + (creepNoise - 0.43) * 1.35 + (creepFine - 0.5) * 0.32;
      const bellyCreepMask =
        axialRustZone * Math.max(0.0, Math.min(1.0, (creepField - 0.14) / 0.42));
      const bellyCrust = bellyCreepMask * 0.014;

      const headReliefScale = ring.isCrownFacet ? 0.38 : 1.0;
      const endGuard =
        Math.sin(Math.min(1.0, u * 22.0) * Math.PI * 0.5) *
        Math.sin(Math.min(1.0, (1.0 - u) * 38.0) * Math.PI * 0.5);

      const radialDisp =
        endGuard *
        headReliefScale *
        (hammerFuller +
          0.0068 * intactPlate -
          0.0092 * deepPit +
          dorsalBlister +
          0.0028 * (slagGroove - 0.5) -
          edgeNibble +
          bellyCrust);

      const outLen = Math.hypot(sy, sz) || 1.0;
      const ny = sy / outLen;
      const nz = sz / outLen;

      basePy += ny * radialDisp * poleScale;
      basePz += nz * radialDisp * poleScale;

      const vIdx = i * ringStride + j;
      pos[vIdx * 3] = basePx;
      pos[vIdx * 3 + 1] = basePy;
      pos[vIdx * 3 + 2] = basePz;

      const uSpatialUv = Math.max(0.0, Math.min(1.0, (basePx - SPIKE_MIN_X) / (SPIKE_MAX_X - SPIKE_MIN_X)));
      uvs[vIdx * 2] = uSpatialUv;
      uvs[vIdx * 2 + 1] = j / CIRC_SEGS;

      // Harvest dense attachment sites for 3D detachable rust scabs along corroded shank faces, blisters & head rim
      if (
        i > 10 &&
        i < AXIAL_SEGS - 3 &&
        i % 3 === 0 &&
        j % 4 === 0 &&
        j < CIRC_SEGS &&
        (intactPlate < 0.85 || bellyCreepMask > 0.22)
      ) {
        flakeSites.push({
          pos: [basePx, basePy, basePz],
          norm: [ring.isCrownFacet ? 0.55 : 0.0, ny, nz],
          isNail: true,
        });
      }

      // Attribute layout:
      // .x = spallRustMask (earthy sienna/umber oxide crust vs weathered pewter-charcoal forged iron)
      // .y = cavityAO (darkens deep pits, slag grooves & rose-head bolster undercut)
      // .z = slagFiber [0..1] (longitudinal fibrous wrought-iron grain)
      // .w = bellyOchreMask (organic golden-ochre concretion tongues & dorsal crust rims)
      const dorsalCrust =
        sy > 0.25 && spallCombined > 0.54
          ? Math.min(1.0, (spallCombined - 0.54) * 2.6)
          : 0.0;
      const spallRim = intactPlate > 0.14 && intactPlate < 0.68 ? 0.45 : 0.0;
      const spallRustMask = Math.max(
        0.0,
        Math.min(1.0, (1.0 - intactPlate) * 0.88 + dorsalCrust * 0.78 + bellyCreepMask * 0.72)
      );
      const bellyOchre = Math.max(
        0.0,
        Math.min(1.0, bellyCreepMask * 0.92 + dorsalCrust * 0.58 + spallRim * 0.36)
      );
      const cavityAO = Math.max(
        0.18,
        Math.min(1.0, (0.93 - deepPit * 1.05 - (0.5 - slagGroove) * 0.20) * ring.undercutAO)
      );

      corr[vIdx * 4] = spallRustMask;
      corr[vIdx * 4 + 1] = cavityAO;
      corr[vIdx * 4 + 2] = slagGroove;
      corr[vIdx * 4 + 3] = bellyOchre;
    }
  }

  const indices: number[] = [];
  for (let i = 0; i < AXIAL_SEGS; i++) {
    for (let j = 0; j < CIRC_SEGS; j++) {
      const a = i * ringStride + j;
      const b = (i + 1) * ringStride + j;
      const c = (i + 1) * ringStride + (j + 1);
      const d = i * ringStride + (j + 1);
      // Outward CCW winding: `(d - a) x (b - a) = (0, 0, +1) x (+1, 0, 0) = (0, +1, 0)` outward!
      indices.push(a, d, b, b, d, c);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aCorrosionData', new THREE.BufferAttribute(corr, 4));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return { geometry, flakeSites };
}

/**
 * Live 128x64 `(uSpatial, vTheta)` Corrosion Crust Scrape Map for the Iron Spike.
 * Scraping with the mouse erodes brittle orange/umber rust crust down to cold dark forged
 * wrought-iron metal (`r = wear [0..255]`, `g = freshScrapeGlint [0..255]`), making hover/drag
 * rust scraping immediately obvious right on the nail surface!
 */
export const SCRAPE_MAP_U = 128;
export const SCRAPE_MAP_V = 64;

export class SpikeScrapeWearField {
  readonly data = new Uint8Array(SCRAPE_MAP_U * SCRAPE_MAP_V * 4);
  readonly dataTexture: THREE.DataTexture;
  private dirty = false;
  private decayAccumSec = 0;

  constructor() {
    this.dataTexture = new THREE.DataTexture(
      this.data,
      SCRAPE_MAP_U,
      SCRAPE_MAP_V,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    );
    this.dataTexture.minFilter = THREE.LinearFilter;
    this.dataTexture.magFilter = THREE.LinearFilter;
    this.dataTexture.wrapS = THREE.ClampToEdgeWrapping;
    this.dataTexture.wrapT = THREE.RepeatWrapping;
    this.dataTexture.generateMipmaps = false;
    this.dataTexture.needsUpdate = true;
  }

  /**
   * Stamps an irregular FBM-modulated scrape abrasion onto the nail at `localPt`.
   * Returns `[0..1+]` newly removed rust mass so fresh scrapes trigger rich flaking bursts!
   */
  scrapeAtLocalPoint(localPt: THREE.Vector3, worldRadius: number, strength: number): number {
    const uCenter = (localPt.x - SPIKE_MIN_X) / (SPIKE_MAX_X - SPIKE_MIN_X);
    if (uCenter < -0.05 || uCenter > 1.05) return 0.0;

    const sp = evalSpikeSpine(Math.max(SPIKE_MIN_X, Math.min(SPIKE_MAX_X, localPt.x)));
    const dy = localPt.y - sp.y;
    const dz = localPt.z - sp.z;
    const theta = Math.atan2(dz, dy);
    const vCenter = ((theta / (Math.PI * 2.0)) % 1.0 + 1.0) % 1.0;

    const uRad = Math.max(0.014, (worldRadius * 0.68) / (SPIKE_MAX_X - SPIKE_MIN_X));
    const vRad = Math.min(0.145, Math.max(0.038, worldRadius / (Math.max(0.052, sp.halfY) * 10.2)));

    // Loop 1.85x wider than `(uRad, vRad)` so radial falloff reaches strictly `0.0` well inside the loop —
    // NEVER clipping against integer `[iuMin..iuMax]` or `[-ivSpan..+ivSpan]` box boundaries!
    const uReach = uRad * 1.85;
    const vReach = vRad * 1.85;
    const iuMin = Math.max(0, Math.floor((uCenter - uReach) * SCRAPE_MAP_U));
    const iuMax = Math.min(SCRAPE_MAP_U - 1, Math.ceil((uCenter + uReach) * SCRAPE_MAP_U));
    const ivSpan = Math.max(2, Math.ceil(vReach * SCRAPE_MAP_V));
    const ivCenter = Math.floor(vCenter * SCRAPE_MAP_V);

    let removedRust = 0.0;

    for (let dv = -ivSpan; dv <= ivSpan; dv++) {
      const iv = ((ivCenter + dv) % SCRAPE_MAP_V + SCRAPE_MAP_V) % SCRAPE_MAP_V;
      const vDelta = dv / SCRAPE_MAP_V;
      const nv = vDelta / vRad;

      for (let iu = iuMin; iu <= iuMax; iu++) {
        const uTex = (iu + 0.5) / SCRAPE_MAP_U;
        const nu = (uTex - uCenter) / uRad;
        const r2 = nu * nu + nv * nv * 1.12;
        if (r2 >= 2.85) continue;

        // Smooth radial envelope guaranteed to hit `0.0` before `r2 = 2.85`, modulated by crusty spall plates
        const radialEnv = Math.max(0.0, 1.0 - r2 / 2.65);
        const plateMacro = fbm3DCpu(iu * 0.22 + 2.1, iv * 0.28 - 1.7, 3.9);
        const plateFine = valueNoise3DCpu(iu * 0.68 - 0.9, iv * 0.78 + 1.4, 2.7);
        const spallMask = Math.max(
          0.0,
          Math.min(1.0, radialEnv * (0.48 + 0.95 * plateMacro + 0.38 * (plateFine - 0.45)) - 0.14)
        );
        if (spallMask <= 0.01) continue;

        // Per-texel spall ceiling prevents repeated sweeps from saturating outer margins into a flat rectangle!
        const localCeiling = Math.min(246, Math.round(252 * Math.pow(spallMask, 1.28)));
        const deltaWear = Math.max(1, Math.round(strength * spallMask * 215));
        const base = (iv * SCRAPE_MAP_U + iu) * 4;
        const prevWear = this.data[base];
        const nextWear = Math.max(prevWear, Math.min(localCeiling, prevWear + deltaWear));

        if (nextWear > prevWear) {
          removedRust += (nextWear - prevWear) / 255.0;
          this.data[base] = nextWear;
          this.data[base + 1] = Math.min(255, this.data[base + 1] + Math.round((nextWear - prevWear) * 1.25));
          this.data[base + 3] = 255;
          this.dirty = true;
        }
      }
    }

    return removedRust;
  }

  step(dtSec: number): void {
    this.decayAccumSec += dtSec;
    // Update slow re-oxidation / glint cooling at ~15 Hz (`0` texture uploads when completely untouched!)
    if (this.decayAccumSec >= 0.065) {
      const steps = Math.min(4, Math.floor(this.decayAccumSec / 0.065));
      this.decayAccumSec -= steps * 0.065;

      let anyActive = false;
      const total = SCRAPE_MAP_U * SCRAPE_MAP_V * 4;
      for (let idx = 0; idx < total; idx += 4) {
        const w = this.data[idx];
        const g = this.data[idx + 1];
        if (w > 0 || g > 0) {
          anyActive = true;
          // Fresh metallic glint cools in ~3.5s; scraped iron slowly re-films with thin oxide over ~32s
          this.data[idx + 1] = Math.max(0, g - 5 * steps);
          if (g < 40 && w > 0) {
            this.data[idx] = Math.max(0, w - 1 * steps);
          }
        }
      }
      if (anyActive) {
        this.dirty = true;
      }
    }

    if (this.dirty) {
      this.dataTexture.needsUpdate = true;
      this.dirty = false;
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Reference-Locked Sub-Belly Porous Goethite "Rusticles" (Craggy Fluted Stalactites)
// ---------------------------------------------------------------------------
interface RusticleSpec {
  x: number;
  zOff: number;
  length: number;
  baseRadius: number;
  webWidthX: number;
  bladeAspectZ: number;
  bladeYaw: number;
  flankLipZ: number;
  seed: number;
}

const RUSTICLE_SPECS: readonly RusticleSpec[] = [
  // Zone A: Left Fore-Shank — two distinct gnarled encrusted icicles with open water between them + golden flank blister
  { x: -0.482, zOff: 0.008, length: 0.215, baseRadius: 0.024, webWidthX: 0.032, bladeAspectZ: 0.56, bladeYaw: 0.16, flankLipZ: 0.020, seed: 1.7 },
  { x: -0.368, zOff: 0.012, length: 0.295, baseRadius: 0.027, webWidthX: 0.044, bladeAspectZ: 0.48, bladeYaw: -0.20, flankLipZ: 0.026, seed: 5.3 },

  // Zone B: Left-Center Sculptural Folded Stalactite Columns (distinct broad lobed columns separated by dark clefts, matching reference!)
  { x: -0.268, zOff: 0.006, length: 0.410, baseRadius: 0.033, webWidthX: 0.048, bladeAspectZ: 0.48, bladeYaw: 0.22, flankLipZ: 0.016, seed: 9.1 },
  // Primary deep hero stalactite column (`len = 0.815`) — retains broad folded mineral body down to its jagged wedge tip
  { x: -0.188, zOff: 0.012, length: 0.815, baseRadius: 0.052, webWidthX: 0.062, bladeAspectZ: 0.46, bladeYaw: 0.10, flankLipZ: 0.020, seed: 15.8 },
  // Secondary broad crinkled paddle-lobe stalactite (`len = 0.625`) separated from primary by a clear vertical shadow cleft
  { x: -0.078, zOff: 0.009, length: 0.625, baseRadius: 0.046, webWidthX: 0.056, bladeAspectZ: 0.44, bladeYaw: -0.22, flankLipZ: 0.014, seed: 22.4 },
  // Recessed dark burnt-sienna rear drapery blades for 3D depth separation
  { x: -0.142, zOff: -0.030, length: 0.530, baseRadius: 0.034, webWidthX: 0.048, bladeAspectZ: 0.44, bladeYaw: -0.28, flankLipZ: 0.000, seed: 27.9 },
  { x: -0.032, zOff: -0.028, length: 0.375, baseRadius: 0.028, webWidthX: 0.040, bladeAspectZ: 0.44, bladeYaw: 0.26, flankLipZ: 0.000, seed: 31.4 },

  // Zone C: Mid-Shank Ruffled Curled Golden Coral-Curtain Sheet peeling outward off the `+Z` front flank (signature reference feature!)
  { x: 0.078, zOff: 0.020, length: 0.455, baseRadius: 0.038, webWidthX: 0.084, bladeAspectZ: 0.28, bladeYaw: 0.20, flankLipZ: 0.046, seed: 35.6 },
  { x: 0.146, zOff: 0.016, length: 0.255, baseRadius: 0.030, webWidthX: 0.068, bladeAspectZ: 0.30, bladeYaw: -0.16, flankLipZ: 0.036, seed: 39.8 },

  // Zone D: Pointed Mid-Neck Stalactite + Collar/Head-Side Stalactites (also refracted prominently in the right glass pane!)
  { x: 0.286, zOff: 0.006, length: 0.365, baseRadius: 0.031, webWidthX: 0.052, bladeAspectZ: 0.44, bladeYaw: -0.16, flankLipZ: 0.018, seed: 44.2 },
  { x: 0.426, zOff: -0.008, length: 0.565, baseRadius: 0.034, webWidthX: 0.048, bladeAspectZ: 0.46, bladeYaw: 0.18, flankLipZ: 0.010, seed: 48.7 },
  { x: 0.468, zOff: 0.008, length: 0.360, baseRadius: 0.025, webWidthX: 0.036, bladeAspectZ: 0.48, bladeYaw: -0.22, flankLipZ: 0.012, seed: 53.1 },
];

export const RUSTICLE_TIP_SOURCES: ReadonlyArray<{ x: number; y: number; z: number; len: number; seed: number }> =
  RUSTICLE_SPECS.map((s) => {
    const sp = evalSpikeSpine(s.x);
    const rootY = sp.y - sp.halfY * 0.85;
    return {
      x: s.x,
      y: rootY - s.length,
      z: sp.z + s.zOff,
      len: s.length,
      seed: s.seed,
    };
  });

export function createOsmoticRusticlesGeometry(
  flakeSitesOut?: CorrosionSurfaceSample[]
): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const V_SEGS = 112;
  const U_SEGS = 52;
  const ringStride = U_SEGS + 1;

  for (let sIdx = 0; sIdx < RUSTICLE_SPECS.length; sIdx++) {
    const spec = RUSTICLE_SPECS[sIdx];
    const spine = evalSpikeSpine(spec.x);

    // Anchor high enough inside the lower half of the shank so front-flank concretion brackets hug the iron!
    const hasFlankBracket = spec.flankLipZ >= 0.014;
    const rootY = spine.y - spine.halfY * (hasFlankBracket ? 0.08 : 0.40);
    const bellyFloorY = spine.y - spine.halfY * 0.94;
    const totalLen = spec.length + (rootY - bellyFloorY);

    const cosYaw = Math.cos(spec.bladeYaw);
    const sinYaw = Math.sin(spec.bladeYaw);

    const vertCount = (V_SEGS + 1) * ringStride;
    const pos = new Float32Array(vertCount * 3);
    const kin = new Float32Array(vertCount * 4);

    for (let iv = 0; iv <= V_SEGS; iv++) {
      const v = iv / V_SEGS;
      const pyBase = rootY - v * totalLen;

      // Full collar inside iron (`v <= 0.024`), concave stalactite holdfast fillet under belly
      const emergeGate = iv === 0 ? 0.0 : Math.min(1.0, v / 0.024);
      const emergeSmooth = emergeGate * emergeGate * (3.0 - 2.0 * emergeGate);

      const rawOut = Math.min(1.0, Math.max(0.0, (bellyFloorY - pyBase) / 0.085));
      const outsideRamp = rawOut * rawOut * (3.0 - 2.0 * rawOut);
      const rootZ = spine.z + spec.zOff * outsideRamp;

      // Gnarled meandering stalactite centerline (organic drip path, never stiff plumb cylinders)
      const driftX =
        outsideRamp *
        Math.pow(v, 0.82) *
        0.036 *
        (fbm3DCpu(spec.seed * 1.9, pyBase * 5.4, 1.7) - 0.48) *
        2.0;
      const driftZ =
        outsideRamp *
        Math.pow(v, 0.82) *
        0.022 *
        (fbm3DCpu(3.1, pyBase * 5.4, spec.seed * 2.3) - 0.48) *
        2.0;

      // Arched upper drapery curtain (`bracketFan`) + substantial sculpted stalactite body down to `v = 0.80`!
      const distBelowBelly = Math.max(0.0, bellyFloorY - pyBase);
      const holdfastFillet = Math.exp(-distBelowBelly / 0.056);
      const bracketFan = Math.pow(Math.max(0.0, 1.0 - Math.min(1.0, v / 0.42)), 1.45);

      // Organic dripstone paddle undulations (keeps lower stalactite lobes chunky & sculptural like reference, never hair-thin strings!)
      const lobeMod =
        1.0 +
        0.34 * (fbm3DCpu(spec.seed * 1.6, v * 5.2, 0.7) - 0.46) +
        0.16 * (valueNoise3DCpu(spec.seed * 2.3, v * 14.0, 1.3) - 0.50);

      let taper = Math.pow(Math.max(0.0, 1.0 - v * 0.78), 0.44) * Math.max(0.56, lobeMod);
      if (v > 0.80) {
        // Crisp jagged mineral wedge tip over the bottom 20% (zero needle-thread tail!)
        const tipT = (v - 0.80) / 0.20;
        taper *= Math.pow(Math.max(0.0, 1.0 - tipT), 1.12);
        if (iv === V_SEGS) taper = 0.0;
      }

      const poleZero = iv === 0 || iv === V_SEGS ? 0.0 : emergeSmooth;
      const rxLocal =
        (spec.baseRadius * (1.12 * taper + 0.42 * holdfastFillet) + spec.webWidthX * bracketFan) * poleZero;
      const rzLocal =
        (spec.baseRadius * spec.bladeAspectZ * (1.06 * taper + 0.36 * holdfastFillet) +
          spec.baseRadius * 0.30 * bracketFan) *
        poleZero;

      for (let iu = 0; iu <= U_SEGS; iu++) {
        const phi = (iu / U_SEGS) * Math.PI * 2.0;
        const cp = Math.cos(phi);
        const sp = Math.sin(phi);

        const bx = cp * rxLocal;
        const bz = sp * rzLocal;
        const rotOffX = bx * cosYaw - bz * sinYaw;
        const rotOffZ = bx * sinYaw + bz * cosYaw;

        const rawX = spec.x + driftX + rotOffX;
        const rawY = pyBase;
        const rawZ = rootZ + driftZ + rotOffZ;

        // Two-scale shipwreck goethite rusticle morphology (matching reference render!):
        // 1) Macro folded vertical drapery ribs & hollow channels (ridged 3D FBM — zero polar star harmonics!)
        const foldNoise = fbm3DCpu(rawX * 19.0 + spec.seed, rawY * 5.8 - 0.9, rawZ * 19.0 - spec.seed);
        const draperyFold = 1.0 - Math.abs(foldNoise * 2.0 - 0.96);
        const foldCleft = Math.max(0.0, 0.46 - draperyFold) * 1.95;

        // 2) Meso crusty stepped mineral plates & cavernous sponge pores (breaks folds into crumbly ochre plates!)
        const crustPlate = fbm3DCpu(rawX * 26.0 - 2.3, rawY * 32.0 + spec.seed, rawZ * 26.0 + 3.7);
        const cellNoise = valueNoise3DCpu(rawX * 38.0 + 3.1, rawY * 40.0 - 1.8, rawZ * 38.0 + spec.seed);
        const spongeCavity = Math.max(0.0, 0.22 - Math.abs(cellNoise - 0.48)) * 3.1;

        const poreIn = foldCleft * 0.68 + spongeCavity * 0.65;
        const cragOut =
          Math.pow(Math.max(0.0, draperyFold - 0.30), 1.15) * 1.25 +
          Math.max(0.0, crustPlate - 0.40) * 1.45;

        const preSpine = evalSpikeSpine(rawX);
        const strictBelowBelly = preSpine.y - preSpine.halfY * 0.96;
        const rawLocalOut = Math.min(1.0, Math.max(0.0, (strictBelowBelly - pyBase) / 0.075));
        const localOut = rawLocalOut * rawLocalOut * (3.0 - 2.0 * rawLocalOut);

        // Curled ruffled coral-curtain bracket peeling outward toward `+Z` off the iron shank (`Zone C` & `Zone A` ears!)
        const distFromBellySigned = pyBase - strictBelowBelly;
        const archShape = 1.0 - 0.55 * Math.min(1.0, Math.pow(rotOffX / Math.max(0.025, rxLocal), 2.0));
        const topTuck = Math.min(1.0, Math.max(0.0, (v - 0.035) / 0.085));
        const curtainArch =
          Math.exp(-Math.pow((distFromBellySigned + 0.012) / 0.056, 2.0)) *
          Math.max(0.0, sp) *
          archShape *
          topTuck;
        const curtainWave = 0.50 + 0.95 * draperyFold + 0.48 * (crustPlate - 0.45);
        const flankLipPushZ = spec.flankLipZ * curtainArch * curtainWave;

        const flankExpose = hasFlankBracket ? curtainArch * 0.85 : 0.0;
        const reliefGate =
          iv === 0 || iv === V_SEGS
            ? 0.0
            : Math.max(flankExpose, localOut) * Math.min(1.0, (1.0 - v) * 8.5);

        const dispScale =
          Math.max(0.32, 0.56 + 0.82 * cragOut - 0.54 * poreIn) * reliefGate + (1.0 - reliefGate);

        const px = spec.x + driftX + rotOffX * dispScale;
        const localSpine = evalSpikeSpine(px);
        // Stepped crusty dripstone ledges
        const py =
          rawY +
          ((crustPlate - 0.50) * 0.016 - Math.max(0.0, cragOut - 0.70) * 0.011) * reliefGate;

        const targetCenterZ = localSpine.z + spec.zOff * localOut + flankLipPushZ;
        const rawFinalZ = targetCenterZ + driftZ * localOut + rotOffZ * dispScale;

        const minZBound = localSpine.z - (localSpine.halfZ * (0.68 + 0.28 * localOut) + localOut * 0.11);
        const frontBracketAllowance =
          localSpine.halfZ * (0.72 + 0.36 * Math.min(1.0, curtainArch * 1.8)) + flankLipPushZ;
        const maxZBound =
          localSpine.z +
          Math.max(frontBracketAllowance, localSpine.halfZ * (0.72 + 0.28 * localOut) + localOut * 0.11);
        const pz = Math.max(minZBound, Math.min(maxZBound, rawFinalZ));

        const idx = iv * ringStride + iu;
        pos[idx * 3] = px;
        pos[idx * 3 + 1] = py;
        pos[idx * 3 + 2] = pz;

        if (flakeSitesOut && iv > 12 && iv < V_SEGS - 10 && iv % 22 === 0 && iu % 18 === 0 && iu < U_SEGS) {
          flakeSitesOut.push({
            pos: [px, py, pz],
            norm: [cp, -0.18, sp],
            isNail: false,
          });
        }

        // Deep burnt-sienna drapery clefts & sponge pores (`0.06`) vs bright golden-ochre folded ridges (`1.0`)!
        const softBellyAO = 0.82 + 0.18 * Math.min(1.0, localOut * 1.6 + curtainArch * 1.1);
        const poreAO = Math.max(
          0.06,
          Math.min(1.0, (0.54 + cragOut * 0.62 - poreIn * 1.38) * softBellyAO)
        );
        const vSway = (localOut * Math.max(0.0, v - 0.05)) / 0.95;

        kin[idx * 4] = vSway;
        kin[idx * 4 + 1] = spec.length;
        kin[idx * 4 + 2] = spec.seed;
        kin[idx * 4 + 3] = poreAO;
      }
    }

    const indices: number[] = [];
    for (let iv = 0; iv < V_SEGS; iv++) {
      for (let iu = 0; iu < U_SEGS; iu++) {
        const a = iv * ringStride + iu;
        const b = (iv + 1) * ringStride + iu;
        const c = (iv + 1) * ringStride + (iu + 1);
        const d = iv * ringStride + (iu + 1);
        indices.push(a, d, b, b, d, c);
      }
    }

    const colGeom = new THREE.BufferGeometry();
    colGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    colGeom.setAttribute('aRusticleKinematics', new THREE.BufferAttribute(kin, 4));
    colGeom.setIndex(indices);
    colGeom.computeVertexNormals();
    parts.push(colGeom);
  }

  return mergeIndexedGeometries(parts, 'aRusticleKinematics', 4);
}

// ---------------------------------------------------------------------------
// 3. Gossamer Osmotic Ferric-Hydroxide Silt Ribbons (Crisp S-Curved Silk Wisps)
// ---------------------------------------------------------------------------
interface SiltVeilSpec {
  p0: [number, number, number];
  p1: [number, number, number];
  p2: [number, number, number];
  p3: [number, number, number];
  widthStart: number;
  widthEnd: number;
  twistRad: number;
  seed: number;
  opacity: number;
}

const SILT_VEIL_SPECS: readonly SiltVeilSpec[] = [
  // Upper-left buoyant S-curve gossamer wisp rising off the dorsal mid-shank (matching reference!)
  {
    p0: [0.10, 0.21, -0.05],
    p1: [-0.04, 0.34, 0.05],
    p2: [-0.16, 0.42, -0.06],
    p3: [-0.42, 0.58, 0.04],
    widthStart: 0.09,
    widthEnd: 0.22,
    twistRad: 1.85,
    seed: 1.4,
    opacity: 0.16,
  },
  // Mid-right sweeping silk veil curling down behind the flared bracket & neck
  {
    p0: [0.22, 0.14, -0.07],
    p1: [0.12, -0.08, 0.05],
    p2: [0.30, -0.28, -0.09],
    p3: [0.14, -0.54, 0.03],
    widthStart: 0.10,
    widthEnd: 0.24,
    twistRad: -1.95,
    seed: 5.2,
    opacity: 0.15,
  },
  // Subtle lower-center trailing ferric veil behind the main stalactite curtain
  {
    p0: [-0.04, 0.06, -0.06],
    p1: [0.08, -0.16, -0.12],
    p2: [0.14, -0.38, -0.03],
    p3: [0.02, -0.62, -0.08],
    widthStart: 0.11,
    widthEnd: 0.25,
    twistRad: 1.55,
    seed: 9.7,
    opacity: 0.13,
  },
];

function cubicBezierPt(
  p0: [number, number, number],
  p1: [number, number, number],
  p2: [number, number, number],
  p3: [number, number, number],
  t: number,
  out: [number, number, number]
): void {
  const it = 1.0 - t;
  const b0 = it * it * it;
  const b1 = 3.0 * it * it * t;
  const b2 = 3.0 * it * t * t;
  const b3 = t * t * t;
  out[0] = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0];
  out[1] = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1];
  out[2] = b0 * p0[2] + b1 * p1[2] + b2 * p2[2] + b3 * p3[2];
}

export function createOsmoticSiltVeilsGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const U_STEPS = 64;
  const V_STEPS = 18;
  const vStride = V_STEPS + 1;

  const pCur: [number, number, number] = [0, 0, 0];
  const pNext: [number, number, number] = [0, 0, 0];

  for (const spec of SILT_VEIL_SPECS) {
    const count = (U_STEPS + 1) * vStride;
    const pos = new Float32Array(count * 3);
    const meta = new Float32Array(count * 4);

    for (let iu = 0; iu <= U_STEPS; iu++) {
      const u = iu / U_STEPS;
      cubicBezierPt(spec.p0, spec.p1, spec.p2, spec.p3, u, pCur);
      cubicBezierPt(spec.p0, spec.p1, spec.p2, spec.p3, Math.min(1.0, u + 0.01), pNext);

      let tx = pNext[0] - pCur[0];
      let ty = pNext[1] - pCur[1];
      let tz = pNext[2] - pCur[2];
      const tLen = Math.hypot(tx, ty, tz) || 1.0;
      tx /= tLen;
      ty /= tLen;
      tz /= tLen;

      let refX = 0.0;
      const refY = 0.0;
      let refZ = 1.0;
      if (Math.abs(tz) > 0.9) {
        refX = 1.0;
        refZ = 0.0;
      }
      let bx = ty * refZ - tz * refY;
      let by = tz * refX - tx * refZ;
      let bz = tx * refY - ty * refX;
      const bLen = Math.hypot(bx, by, bz) || 1.0;
      bx /= bLen;
      by /= bLen;
      bz /= bLen;

      const nx = by * tz - bz * ty;
      const ny = bz * tx - bx * tz;
      const nz = bx * ty - by * tx;

      const twist = spec.twistRad * u + 0.4 * Math.sin(u * Math.PI * 2.0 + spec.seed);
      const ct = Math.cos(twist);
      const st = Math.sin(twist);

      const sideX = bx * ct + nx * st;
      const sideY = by * ct + ny * st;
      const sideZ = bz * ct + nz * st;

      const outNormX = -bx * st + nx * ct;
      const outNormY = -by * st + ny * ct;
      const outNormZ = -bz * st + nz * ct;

      const halfW = 0.5 * (spec.widthStart + (spec.widthEnd - spec.widthStart) * Math.pow(u, 0.85));

      for (let iv = 0; iv <= V_STEPS; iv++) {
        const v = iv / V_STEPS;
        const s = (v - 0.5) * 2.0;
        const cup = (1.0 - s * s) * halfW * 0.25;

        const idx = iu * vStride + iv;
        pos[idx * 3] = pCur[0] + sideX * (s * halfW) + outNormX * cup;
        pos[idx * 3 + 1] = pCur[1] + sideY * (s * halfW) + outNormY * cup;
        pos[idx * 3 + 2] = pCur[2] + sideZ * (s * halfW) + outNormZ * cup;

        meta[idx * 4] = u;
        meta[idx * 4 + 1] = v;
        meta[idx * 4 + 2] = spec.seed;
        meta[idx * 4 + 3] = spec.opacity;
      }
    }

    const indices: number[] = [];
    for (let iu = 0; iu < U_STEPS; iu++) {
      for (let iv = 0; iv < V_STEPS; iv++) {
        const a = iu * vStride + iv;
        const b = (iu + 1) * vStride + iv;
        const c = (iu + 1) * vStride + (iv + 1);
        const d = iu * vStride + (iv + 1);
        // Outward CCW winding: `(d - a) x (b - a) = (0, 0, +1) x (+1, 0, 0) = (0, +1, 0)` outward!
      indices.push(a, d, b, b, d, c);
      }
    }

    const veilGeom = new THREE.BufferGeometry();
    veilGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    veilGeom.setAttribute('aVeilMeta', new THREE.BufferAttribute(meta, 4));
    veilGeom.setIndex(indices);
    veilGeom.computeVertexNormals();
    parts.push(veilGeom);
  }

  return mergeIndexedGeometries(parts, 'aVeilMeta', 4);
}

// ---------------------------------------------------------------------------
// 4. Museum 6-Plate Beveled Soda-Lime Float-Glass Tank Frame (Non-Overlapping)
// ---------------------------------------------------------------------------
function makeChamferedBar(
  cx: number,
  cy: number,
  cz: number,
  sx: number,
  sy: number,
  sz: number,
  opticalChord: number
): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(sx, sy, sz, 4, 4, 4);
  const pos = g.attributes.position as THREE.BufferAttribute;
  const meta = new Float32Array(pos.count * 4);
  const bevel = 0.003;

  for (let i = 0; i < pos.count; i++) {
    let lx = pos.getX(i);
    let ly = pos.getY(i);
    let lz = pos.getZ(i);

    const ax = Math.abs(lx) / Math.max(1e-4, sx * 0.5);
    const ay = Math.abs(ly) / Math.max(1e-4, sy * 0.5);
    const az = Math.abs(lz) / Math.max(1e-4, sz * 0.5);

    let shortEdgeHits = 0;
    if (sx < 0.1 && ax > 0.85) shortEdgeHits++;
    if (sy < 0.1 && ay > 0.85) shortEdgeHits++;
    if (sz < 0.1 && az > 0.85) shortEdgeHits++;

    if (shortEdgeHits >= 2) {
      if (sx < 0.1 && ax > 0.92) lx -= Math.sign(lx) * bevel;
      if (sy < 0.1 && ay > 0.92) ly -= Math.sign(ly) * bevel;
      if (sz < 0.1 && az > 0.92) lz -= Math.sign(lz) * bevel;
    }

    pos.setXYZ(i, cx + lx, cy + ly, cz + lz);
    const edgeRim = shortEdgeHits >= 2 ? 1.0 : 0.0;
    meta[i * 4] = opticalChord;
    meta[i * 4 + 1] = edgeRim;
    meta[i * 4 + 2] = cy / TANK_OUTER_Y;
    meta[i * 4 + 3] = 1.0;
  }

  g.setAttribute('aGlassMeta', new THREE.BufferAttribute(meta, 4));
  g.computeVertexNormals();
  return g;
}

export function createBeveledOpticalTankGlassGeometry(): {
  edgeSlabGeom: THREE.BufferGeometry;
  outerWindowGeom: THREE.BufferGeometry;
} {
  const t = GLASS_THICKNESS;
  const ox = TANK_OUTER_X;
  const oy = TANK_OUTER_Y;
  const oz = TANK_OUTER_Z;

  const bars: THREE.BufferGeometry[] = [];
  const vertSpanY = (oy - t) * 2.0;
  const vxCoords: Array<[number, number]> = [
    [-ox + t * 0.5, oz - t * 0.5],
    [ox - t * 0.5, oz - t * 0.5],
    [ox - t * 0.5, -oz + t * 0.5],
    [-ox + t * 0.5, -oz + t * 0.5],
  ];
  for (const [vx, vz] of vxCoords) {
    bars.push(makeChamferedBar(vx, 0, vz, t, vertSpanY, t, 1.65));
  }

  const fullSpanX = ox * 2.0;
  const innerSpanZ = (oz - t) * 2.0;
  for (const signY of [-1, 1]) {
    const py = signY * (oy - t * 0.5);
    bars.push(makeChamferedBar(0, py, oz - t * 0.5, fullSpanX, t, t, 1.45));
    bars.push(makeChamferedBar(0, py, -oz + t * 0.5, fullSpanX, t, t, 1.45));
    bars.push(makeChamferedBar(-ox + t * 0.5, py, 0, t, t, innerSpanZ, 1.45));
    bars.push(makeChamferedBar(ox - t * 0.5, py, 0, t, t, innerSpanZ, 1.45));
  }

  const edgeSlabGeom = mergeIndexedGeometries(bars, 'aGlassMeta', 4);
  const outerWindowGeom = new THREE.BoxGeometry(ox * 2.0 + 0.001, oy * 2.0 + 0.001, oz * 2.0 + 0.001);

  return { edgeSlabGeom, outerWindowGeom };
}

// ---------------------------------------------------------------------------
// 5. Coupled Tank Fluid Slosh, Meniscus Wave PDE & Rusticle Sway Solver
// ---------------------------------------------------------------------------
export const MENISCUS_RES = 64;
const MENISCUS_CELLS = MENISCUS_RES * MENISCUS_RES;

export class TankHydroSim {
  readonly hCurr = new Float32Array(MENISCUS_CELLS);
  readonly hPrev = new Float32Array(MENISCUS_CELLS);
  readonly hNext = new Float32Array(MENISCUS_CELLS);
  readonly texHalf = new Uint16Array(MENISCUS_CELLS * 4);
  readonly dataTexture: THREE.DataTexture;

  sloshX = 0.0;
  sloshZ = 0.0;
  sloshVelX = 0.0;
  sloshVelZ = 0.0;

  sway1X = 0.0;
  sway1Z = 0.0;
  sway1VelX = 0.0;
  sway1VelZ = 0.0;

  sway2X = 0.0;
  sway2Z = 0.0;
  sway2VelX = 0.0;
  sway2VelZ = 0.0;

  stirEnergy = 0.06;
  stirCenter = new THREE.Vector3(0, 0, 0);
  stirVelocity = new THREE.Vector3(0, 0, 0);

  private ambientClock = 0;

  constructor() {
    this.dataTexture = new THREE.DataTexture(
      this.texHalf,
      MENISCUS_RES,
      MENISCUS_RES,
      THREE.RGBAFormat,
      THREE.HalfFloatType
    );
    this.dataTexture.minFilter = THREE.LinearFilter;
    this.dataTexture.magFilter = THREE.LinearFilter;
    this.dataTexture.wrapS = THREE.ClampToEdgeWrapping;
    this.dataTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.dataTexture.generateMipmaps = false;

    this.injectWaveImpulse(0.20, -0.16, 0.0035, 0.28);
    this.packTexture();
  }

  injectWaveImpulse(normX: number, normZ: number, strength: number, radiusNorm = 0.22): void {
    const invSig2 = 1.0 / Math.max(0.005, radiusNorm * radiusNorm);
    for (let iy = 0; iy < MENISCUS_RES; iy++) {
      const gz = ((iy + 0.5) / MENISCUS_RES) * 2.0 - 1.0;
      const dz = gz - normZ;
      for (let ix = 0; ix < MENISCUS_RES; ix++) {
        const gx = ((ix + 0.5) / MENISCUS_RES) * 2.0 - 1.0;
        const dx = gx - normX;
        const d2 = dx * dx + dz * dz;
        if (d2 > 5.5 * radiusNorm * radiusNorm) continue;
        const idx = iy * MENISCUS_RES + ix;
        const g = Math.exp(-d2 * invSig2) * (1.0 - 0.42 * d2 * invSig2);
        this.hCurr[idx] = Math.max(-0.032, Math.min(0.032, this.hCurr[idx] + strength * g));
      }
    }
  }

  injectFluidImpulse(impX: number, impZ: number, localPos?: THREE.Vector3): void {
    this.sloshVelX = Math.max(-1.3, Math.min(1.3, this.sloshVelX + impX));
    this.sloshVelZ = Math.max(-1.3, Math.min(1.3, this.sloshVelZ + impZ));

    this.sway1VelX = Math.max(-1.5, Math.min(1.5, this.sway1VelX + impX * 0.85));
    this.sway1VelZ = Math.max(-1.5, Math.min(1.5, this.sway1VelZ + impZ * 0.85));
    this.sway2VelX = Math.max(-1.8, Math.min(1.8, this.sway2VelX - impX * 0.55));
    this.sway2VelZ = Math.max(-1.8, Math.min(1.8, this.sway2VelZ - impZ * 0.55));

    const mag = Math.hypot(impX, impZ);
    this.stirEnergy = Math.min(1.6, this.stirEnergy + mag * 1.45);
    if (localPos) {
      this.stirCenter.lerp(localPos, 0.55);
    }
    this.stirVelocity.x = this.stirVelocity.x * 0.6 + impX * 2.2;
    this.stirVelocity.z = this.stirVelocity.z * 0.6 + impZ * 2.2;
  }

  step(dtSec: number, elapsedSec: number): void {
    const dt = Math.min(0.033, Math.max(0.002, dtSec));

    this.ambientClock += dt;
    if (this.ambientClock > 1.9) {
      this.ambientClock = 0;
      const phi = elapsedSec * 2.399963;
      this.injectWaveImpulse(Math.cos(phi) * 0.42, Math.sin(phi) * 0.42, 0.002, 0.30);
    }

    const omegaSlosh2 = 16.5;
    const dampSlosh = Math.exp(-dt * 2.2);
    this.sloshVelX = (this.sloshVelX - omegaSlosh2 * this.sloshX * dt) * dampSlosh;
    this.sloshVelZ = (this.sloshVelZ - omegaSlosh2 * this.sloshZ * dt) * dampSlosh;
    this.sloshX = Math.max(-0.075, Math.min(0.075, this.sloshX + this.sloshVelX * dt));
    this.sloshZ = Math.max(-0.075, Math.min(0.075, this.sloshZ + this.sloshVelZ * dt));

    const breathX = Math.sin(elapsedSec * 0.68) * 0.0032;
    const breathZ = Math.cos(elapsedSec * 0.54 + 1.1) * 0.0026;

    const omega1Sq = 11.2;
    const damp1 = Math.exp(-dt * 2.45);
    const fluidDragX = this.sloshVelX * 0.42;
    const fluidDragZ = this.sloshVelZ * 0.42;

    this.sway1VelX = (this.sway1VelX + (fluidDragX - omega1Sq * (this.sway1X - breathX)) * dt) * damp1;
    this.sway1VelZ = (this.sway1VelZ + (fluidDragZ - omega1Sq * (this.sway1Z - breathZ)) * dt) * damp1;
    this.sway1X = Math.max(-0.13, Math.min(0.13, this.sway1X + this.sway1VelX * dt));
    this.sway1Z = Math.max(-0.13, Math.min(0.13, this.sway1Z + this.sway1VelZ * dt));

    const omega2Sq = 31.0;
    const damp2 = Math.exp(-dt * 3.4);
    this.sway2VelX = (this.sway2VelX + (-fluidDragX * 0.34 - omega2Sq * this.sway2X) * dt) * damp2;
    this.sway2VelZ = (this.sway2VelZ + (-fluidDragZ * 0.34 - omega2Sq * this.sway2Z) * dt) * damp2;
    this.sway2X = Math.max(-0.055, Math.min(0.055, this.sway2X + this.sway2VelX * dt));
    this.sway2Z = Math.max(-0.055, Math.min(0.055, this.sway2Z + this.sway2Z * 0.0 + this.sway2VelZ * dt));

    this.stirEnergy = 0.06 + (this.stirEnergy - 0.06) * Math.exp(-dt * 1.65);
    this.stirVelocity.multiplyScalar(Math.exp(-dt * 2.8));

    const c2 = 0.16;
    const waveDamp = 0.986;
    for (let iy = 0; iy < MENISCUS_RES; iy++) {
      const yUp = Math.max(0, iy - 1);
      const yDn = Math.min(MENISCUS_RES - 1, iy + 1);
      for (let ix = 0; ix < MENISCUS_RES; ix++) {
        const xLt = Math.max(0, ix - 1);
        const xRt = Math.min(MENISCUS_RES - 1, ix + 1);
        const idx = iy * MENISCUS_RES + ix;
        const c = this.hCurr[idx];
        const lap =
          this.hCurr[iy * MENISCUS_RES + xLt] +
          this.hCurr[iy * MENISCUS_RES + xRt] +
          this.hCurr[yUp * MENISCUS_RES + ix] +
          this.hCurr[yDn * MENISCUS_RES + ix] -
          4.0 * c;
        this.hNext[idx] = (2.0 * c - this.hPrev[idx] + c2 * lap) * waveDamp;
      }
    }

    this.hPrev.set(this.hCurr);
    this.hCurr.set(this.hNext);
    this.packTexture();
  }

  private packTexture(): void {
    const toHalf = THREE.DataUtils.toHalfFloat;
    const gradScale = MENISCUS_RES * 0.32;
    for (let iy = 0; iy < MENISCUS_RES; iy++) {
      const y0 = Math.max(0, iy - 1);
      const y1 = Math.min(MENISCUS_RES - 1, iy + 1);
      for (let ix = 0; ix < MENISCUS_RES; ix++) {
        const x0 = Math.max(0, ix - 1);
        const x1 = Math.min(MENISCUS_RES - 1, ix + 1);
        const idx = iy * MENISCUS_RES + ix;
        const h = this.hCurr[idx];
        const dhdx = (this.hCurr[iy * MENISCUS_RES + x1] - this.hCurr[iy * MENISCUS_RES + x0]) * gradScale;
        const dhdz = (this.hCurr[y1 * MENISCUS_RES + ix] - this.hCurr[y0 * MENISCUS_RES + ix]) * gradScale;

        const b = idx * 4;
        this.texHalf[b] = toHalf(h);
        this.texHalf[b + 1] = toHalf(dhdx);
        this.texHalf[b + 2] = toHalf(dhdz);
        this.texHalf[b + 3] = toHalf(1.0);
      }
    }
    this.dataTexture.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// 6. Unified Deformable Water Prism Volume (Zero Detached Meniscus Sheet!)
// ---------------------------------------------------------------------------
export class DeformableWaterPrism {
  readonly geometry: THREE.BoxGeometry;
  readonly mesh: THREE.Mesh;
  private readonly baseXYZ: Float32Array;
  private readonly vFrac: Float32Array;
  private readonly wallClimbFactor: Float32Array;

  constructor(material: THREE.Material) {
    const SEG_XZ = 28;
    const SEG_Y = 6;
    const waterHeight = TANK_INNER_Y_TOP - TANK_INNER_Y_BOT;
    const waterCenterY = 0.5 * (TANK_INNER_Y_TOP + TANK_INNER_Y_BOT);

    this.geometry = new THREE.BoxGeometry(
      TANK_INNER_X * 2.0,
      waterHeight,
      TANK_INNER_Z * 2.0,
      SEG_XZ,
      SEG_Y,
      SEG_XZ
    );
    this.geometry.translate(0, waterCenterY, 0);

    const pos = this.geometry.attributes.position as THREE.BufferAttribute;
    this.baseXYZ = new Float32Array(pos.count * 3);
    this.vFrac = new Float32Array(pos.count);
    this.wallClimbFactor = new Float32Array(pos.count);
    const waterMeta = new Float32Array(pos.count * 2);

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      this.baseXYZ[i * 3] = x;
      this.baseXYZ[i * 3 + 1] = y;
      this.baseXYZ[i * 3 + 2] = z;

      const vf = Math.max(0.0, Math.min(1.0, (y - TANK_INNER_Y_BOT) / waterHeight));
      const wx = Math.abs(x) / TANK_INNER_X;
      const wz = Math.abs(z) / TANK_INNER_Z;
      const wallProx = Math.max(wx, wz);

      this.vFrac[i] = vf;
      this.wallClimbFactor[i] = Math.pow(wallProx, 9.0);
      // .x = vertical fraction [0 floor .. 1 water surface], .y = inner-wall proximity [0 center .. 1 glass wall]
      waterMeta[i * 2] = vf;
      waterMeta[i * 2 + 1] = wallProx;
    }

    this.geometry.setAttribute('aWaterMeta', new THREE.BufferAttribute(waterMeta, 2));
    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.frustumCulled = false;
  }

  updateFromSim(sim: TankHydroSim): void {
    const pos = this.geometry.attributes.position as THREE.BufferAttribute;
    const maxLidY = TANK_OUTER_Y - GLASS_THICKNESS - 0.006;

    for (let i = 0; i < pos.count; i++) {
      const vf = this.vFrac[i];
      if (vf <= 1e-5) continue;

      const x = this.baseXYZ[i * 3];
      const z = this.baseXYZ[i * 3 + 2];
      const ux = Math.max(
        0,
        Math.min(MENISCUS_RES - 1, Math.floor(((x / TANK_INNER_X) * 0.5 + 0.5) * MENISCUS_RES))
      );
      const uz = Math.max(
        0,
        Math.min(MENISCUS_RES - 1, Math.floor(((z / TANK_INNER_Z) * 0.5 + 0.5) * MENISCUS_RES))
      );
      const waveH = sim.hCurr[uz * MENISCUS_RES + ux];

      // Subtle 7mm capillary wetting climb right at the glass walls — shared identically by top & side faces!
      const wallClimb = this.wallClimbFactor[i] * 0.007;
      const rawTopY =
        TANK_INNER_Y_TOP + sim.sloshX * 0.45 * x + sim.sloshZ * 0.45 * z + waveH + wallClimb;
      const topY = Math.min(maxLidY, rawTopY);

      pos.setY(i, TANK_INNER_Y_BOT + vf * (topY - TANK_INNER_Y_BOT));
    }
    pos.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// 7. Interactive 3D Delaminating Rust-Bark Scales, Jagged Shards & Crumble Grit
// ---------------------------------------------------------------------------
const FLAKE_COUNT = 780;
const VISIBLE_NAIL_SCAB_COUNT = 88;
const DYNAMIC_POOL_END = 700;

const FLAKE_STATE_PERCHED = 0;
const FLAKE_STATE_FALLING = 1;
const FLAKE_STATE_SETTLED = 2;

const _flakeMat = new THREE.Matrix4();
const _flakeQuat = new THREE.Quaternion();
const _alignQuat = new THREE.Quaternion();
const _spinQuat = new THREE.Quaternion();
const _tiltQuat = new THREE.Quaternion();
const _flakeEuler = new THREE.Euler();
const _flakePos = new THREE.Vector3();
const _flakeScale = new THREE.Vector3();
const _upVec = new THREE.Vector3(0, 1, 0);
const _xAxisVec = new THREE.Vector3(1, 0, 0);
const _normVec = new THREE.Vector3();

/**
 * Builds a paper-thin, irregular cleaved 18-segment iron-oxide bark shard / scale
 * (smooth irregular polygonal fracture edges — strictly ZERO radial star/sawblade teeth!).
 * `aFlakeVert` (`vec2`): `x = faceSide` (`1.0` outer crust, `0.0` inner cleaved face, `0.45` rim),
 *                        `y = fractureEdgeMask` (`0.0` center .. `1.0` snapped perimeter).
 */
function createFlakyRustScaleGeometry(): THREE.BufferGeometry {
  // 11-vertex angular stepped brittle rust-crust plate (straight snapped cleavage edges + stepped spall bite —
  // strictly ZERO rounded oval petals and ZERO radial star/sawblade symmetry!)
  const rim2D: ReadonlyArray<[number, number]> = [
    [1.16, -0.10],
    [0.86, 0.56],
    [0.20, 0.72],
    [-0.18, 0.34], // stepped concave spall fracture notch
    [-0.68, 0.64],
    [-1.14, 0.28],
    [-1.02, -0.34],
    [-0.44, -0.70],
    [0.10, -0.56],
    [0.54, -0.74],
    [0.96, -0.44],
  ];
  const SEGMENTS = rim2D.length;
  const halfThick = 0.015;

  const positions: number[] = [];
  const vertMeta: number[] = [];
  const indices: number[] = [];

  const evalRimPt = (i: number): [number, number, number] => {
    const [px, pz] = rim2D[i];
    // Pronounced dry-bark peel arch + stepped ridge so every falling shard has 3D shell shading
    const curl = -0.22 * px * px - 0.10 * pz * pz + 0.06 * px * pz;
    return [px, curl, pz];
  };

  // Dorsal outer crust (+Y, side = 1.0): center + inner ring (0.52r) + notched outer rim
  const topCenter = 0;
  positions.push(0, halfThick + 0.02, 0);
  vertMeta.push(1.0, 0.0);

  const topMidStart = positions.length / 3;
  for (let i = 0; i < SEGMENTS; i++) {
    const [rx, curl, rz] = evalRimPt(i);
    positions.push(rx * 0.52, halfThick + 0.02 + curl * 0.28, rz * 0.52);
    vertMeta.push(1.0, 0.28);
  }
  const topRimStart = positions.length / 3;
  for (let i = 0; i < SEGMENTS; i++) {
    const [rx, curl, rz] = evalRimPt(i);
    positions.push(rx, halfThick + 0.02 + curl, rz);
    vertMeta.push(1.0, 1.0);
  }
  for (let i = 0; i < SEGMENTS; i++) {
    const next = (i + 1) % SEGMENTS;
    indices.push(topCenter, topMidStart + next, topMidStart + i);
    const m0 = topMidStart + i;
    const m1 = topMidStart + next;
    const r0 = topRimStart + i;
    const r1 = topRimStart + next;
    indices.push(m0, m1, r0, m1, r1, r0);
  }

  // Ventral cleaved underside (-Y, side = 0.0)
  const botCenter = positions.length / 3;
  positions.push(0, -halfThick + 0.02, 0);
  vertMeta.push(0.0, 0.0);

  const botMidStart = positions.length / 3;
  for (let i = 0; i < SEGMENTS; i++) {
    const [rx, curl, rz] = evalRimPt(i);
    positions.push(rx * 0.52, -halfThick + 0.02 + curl * 0.28, rz * 0.52);
    vertMeta.push(0.0, 0.32);
  }
  const botRimStart = positions.length / 3;
  for (let i = 0; i < SEGMENTS; i++) {
    const [rx, curl, rz] = evalRimPt(i);
    positions.push(rx, -halfThick + 0.02 + curl, rz);
    vertMeta.push(0.0, 1.0);
  }
  for (let i = 0; i < SEGMENTS; i++) {
    const next = (i + 1) % SEGMENTS;
    indices.push(botCenter, botMidStart + i, botMidStart + next);
    const m0 = botMidStart + i;
    const m1 = botMidStart + next;
    const r0 = botRimStart + i;
    const r1 = botRimStart + next;
    indices.push(m0, r0, m1, m1, r0, r1);
  }

  // Crisp fractured perimeter wall
  const wallStart = positions.length / 3;
  for (let i = 0; i < SEGMENTS; i++) {
    const [rx, curl, rz] = evalRimPt(i);
    positions.push(rx, halfThick + 0.02 + curl, rz);
    vertMeta.push(0.45, 1.0);
    positions.push(rx, -halfThick + 0.02 + curl, rz);
    vertMeta.push(0.25, 1.0);
  }
  for (let i = 0; i < SEGMENTS; i++) {
    const next = (i + 1) % SEGMENTS;
    const i0 = wallStart + i * 2;
    const i1 = i0 + 1;
    const j0 = wallStart + next * 2;
    const j1 = j0 + 1;
    indices.push(i0, j0, i1, j0, j1, i1);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('aFlakeVert', new THREE.Float32BufferAttribute(vertMeta, 2));
  g.setIndex(indices);
  g.computeVertexNormals();
  return g;
}

export class DelaminatingRustFlakeSystem {
  readonly geometry: THREE.BufferGeometry;
  readonly mesh: THREE.InstancedMesh;
  readonly instanceMetaAttr: THREE.InstancedBufferAttribute;

  private readonly sites: CorrosionSurfaceSample[];
  private readonly nailSiteIndices: number[] = [];
  private readonly state = new Uint8Array(FLAKE_COUNT);
  private readonly siteIdx = new Uint16Array(FLAKE_COUNT);

  private readonly px = new Float32Array(FLAKE_COUNT);
  private readonly py = new Float32Array(FLAKE_COUNT);
  private readonly pz = new Float32Array(FLAKE_COUNT);

  private readonly vx = new Float32Array(FLAKE_COUNT);
  private readonly vy = new Float32Array(FLAKE_COUNT);
  private readonly vz = new Float32Array(FLAKE_COUNT);

  private readonly rx = new Float32Array(FLAKE_COUNT);
  private readonly ry = new Float32Array(FLAKE_COUNT);
  private readonly rz = new Float32Array(FLAKE_COUNT);

  private readonly wx = new Float32Array(FLAKE_COUNT);
  private readonly wy = new Float32Array(FLAKE_COUNT);
  private readonly wz = new Float32Array(FLAKE_COUNT);

  private readonly scaleW = new Float32Array(FLAKE_COUNT);
  private readonly scaleL = new Float32Array(FLAKE_COUNT);
  private readonly sizeTier = new Float32Array(FLAKE_COUNT); // 0 = fine grit, 0.5 = shard, 1.0 = bark scab
  private readonly baseYaw = new Float32Array(FLAKE_COUNT);
  private readonly baseTilt = new Float32Array(FLAKE_COUNT);
  private readonly hoverPeel = new Float32Array(FLAKE_COUNT);
  private readonly phase = new Float32Array(FLAKE_COUNT);
  private readonly ageSec = new Float32Array(FLAKE_COUNT);
  private readonly blisterLift = new Float32Array(FLAKE_COUNT);
  private readonly instanceMetaData = new Float32Array(FLAKE_COUNT * 4);

  private hasHoverNail = false;
  private readonly hoverNailPt = new THREE.Vector3();
  private hoverDwellSec = 0;
  private recycleCursor = VISIBLE_NAIL_SCAB_COUNT;
  private metaDirty = true;

  constructor(material: THREE.Material, sites: CorrosionSurfaceSample[]) {
    this.sites = sites.length > 0 ? sites : [{ pos: [0, 0.14, 0], norm: [0, 1, 0], isNail: true }];
    for (let s = 0; s < this.sites.length; s++) {
      const sp = this.sites[s];
      // Place visible layered rust scabs strictly on the corroded shank (`-0.52 .. +0.48`), keeping tip & rose-head arrises clean!
      if (sp.isNail && sp.pos[0] > -0.52 && sp.pos[0] < 0.48 && (sp.norm[2] > -0.25 || sp.norm[1] > 0.30)) {
        this.nailSiteIndices.push(s);
      }
    }
    if (this.nailSiteIndices.length === 0) {
      this.nailSiteIndices.push(0);
    }

    this.geometry = createFlakyRustScaleGeometry();
    this.instanceMetaAttr = new THREE.InstancedBufferAttribute(this.instanceMetaData, 4);
    this.instanceMetaAttr.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('aFlakeInstance', this.instanceMetaAttr);

    this.mesh = new THREE.InstancedMesh(this.geometry, material, FLAKE_COUNT);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = false;

    for (let i = 0; i < FLAKE_COUNT; i++) {
      const h1 = hash31Cpu(i * 1.71, 3.3, 8.9);
      const h2 = hash31Cpu(i * 4.13, 9.1, 2.7);
      const h3 = hash31Cpu(i * 6.37, 1.9, 5.4);
      this.phase[i] = h1 * Math.PI * 2.0;

      // Instance shader variety: [seed, curlArch, toneMix, hoverHighlight]
      this.instanceMetaData[i * 4] = h1 * 17.3 + h2 * 5.1;
      this.instanceMetaData[i * 4 + 1] = 0.65 + h2 * 0.95;
      this.instanceMetaData[i * 4 + 2] = h3;
      this.instanceMetaData[i * 4 + 3] = 0.0;

      if (i < VISIBLE_NAIL_SCAB_COUNT) {
        // 88 visible flush rust scabs & blisters clinging to the iron nail (pry up & peel on mouseover!)
        this.scaleW[i] = 0.0085 + Math.pow(h1, 1.25) * 0.0095;
        this.scaleL[i] = 0.0098 + Math.pow(h2, 1.15) * 0.0115;
        this.sizeTier[i] = 1.0;
        const nIdx = this.nailSiteIndices[(i * 7) % this.nailSiteIndices.length];
        this.attachToSite(i, nIdx, 1.0);
      } else if (i < 250) {
        // Irregular peeled rust-crust scabs & bark plates in dynamic scrape pool (varied broad chip to strip proportions!)
        const isBroadChip = h3 > 0.42;
        this.scaleW[i] = isBroadChip ? 0.0082 + h1 * 0.0088 : 0.0056 + h1 * 0.0062;
        this.scaleL[i] = isBroadChip ? 0.0092 + h2 * 0.0098 : 0.0095 + h2 * 0.0110;
        this.sizeTier[i] = 1.0;
        this.attachToSite(i, i % this.sites.length, 1.0);
      } else if (i < 485) {
        // Medium irregular chipped oxide flakes
        this.scaleW[i] = 0.0044 + h1 * 0.0052;
        this.scaleL[i] = 0.0052 + h2 * 0.0066;
        this.sizeTier[i] = 0.5;
        this.attachToSite(i, i % this.sites.length, 1.0);
      } else if (i < DYNAMIC_POOL_END) {
        // Fine irregular rust grit crumbs (trailing powdery crumble around larger peeling scabs!)
        this.scaleW[i] = 0.0018 + h1 * 0.0024;
        this.scaleL[i] = 0.0022 + h2 * 0.0030;
        this.sizeTier[i] = 0.0;
        this.attachToSite(i, i % this.sites.length, 1.0);
      } else if (i < 770) {
        // 70 settled irregular rust scales & grit directly beneath the nail & rusticles on the bottom glass
        const isBig = i % 5 === 0;
        this.scaleW[i] = isBig ? 0.0075 + h1 * 0.0075 : 0.0028 + h1 * 0.0038;
        this.scaleL[i] = isBig ? 0.0088 + h2 * 0.0095 : 0.0034 + h2 * 0.0050;
        this.sizeTier[i] = isBig ? 1.0 : 0.3;
        const tip = RUSTICLE_TIP_SOURCES[i % RUSTICLE_TIP_SOURCES.length];
        this.state[i] = FLAKE_STATE_SETTLED;
        this.px[i] = tip.x + (h1 - 0.5) * 0.24;
        this.py[i] = TANK_INNER_Y_BOT + 0.0025 + (i % 5) * 0.0004;
        this.pz[i] = tip.z + (h2 - 0.5) * 0.16;
        this.rx[i] = (h3 > 0.45 ? 0.0 : Math.PI) + (h1 - 0.5) * 0.16;
        this.ry[i] = h2 * Math.PI * 2.0;
        this.rz[i] = (h2 - 0.5) * 0.16;
        this.blisterLift[i] = 0.92;
        this.ageSec[i] = h1 * 5.0;
      } else {
        // Just 10 tiny ambient crumbs slowly settling beneath the longest stalactite tips
        this.scaleW[i] = 0.0030 + h1 * 0.0035;
        this.scaleL[i] = 0.0045 + h2 * 0.0055;
        this.sizeTier[i] = 0.2;
        const tip = RUSTICLE_TIP_SOURCES[(i * 3) % RUSTICLE_TIP_SOURCES.length];
        this.state[i] = FLAKE_STATE_FALLING;
        this.px[i] = tip.x + (h1 - 0.5) * 0.04;
        this.py[i] = TANK_INNER_Y_BOT + 0.06 + h2 * Math.max(0.10, tip.y - TANK_INNER_Y_BOT);
        this.pz[i] = tip.z + (h2 - 0.5) * 0.04;
        this.rx[i] = h1 * 3.14;
        this.ry[i] = h2 * 6.28;
        this.rz[i] = h1 * 2.1;
        this.wx[i] = (h1 - 0.5) * 1.4;
        this.wy[i] = (h2 - 0.5) * 1.0;
        this.wz[i] = (h1 - 0.5) * 1.4;
        this.vy[i] = -0.12;
        this.blisterLift[i] = 0.90;
      }
    }

    this.syncMatrices();
  }

  /**
   * Updates the active nail-surface hover point so nearby 3D rust scabs visibly curl/pry up
   * under the cursor, signaling immediately that the rust on the nail can be scraped off!
   */
  setHoverNailPoint(localPt: THREE.Vector3 | null): void {
    if (localPt) {
      this.hasHoverNail = true;
      this.hoverNailPt.copy(localPt);
    } else {
      this.hasHoverNail = false;
      this.hoverDwellSec = 0;
    }
  }

  private computePerchedEuler(i: number, peelExtraRad: number): void {
    const s = this.sites[this.siteIdx[i]];
    _normVec.set(s.norm[0], s.norm[1], s.norm[2]).normalize();
    _alignQuat.setFromUnitVectors(_upVec, _normVec);
    _spinQuat.setFromAxisAngle(_upVec, this.baseYaw[i]);
    _tiltQuat.setFromAxisAngle(_xAxisVec, this.baseTilt[i] + peelExtraRad);
    _flakeQuat.copy(_alignQuat).multiply(_spinQuat).multiply(_tiltQuat);
    _flakeEuler.setFromQuaternion(_flakeQuat);
    this.rx[i] = _flakeEuler.x;
    this.ry[i] = _flakeEuler.y;
    this.rz[i] = _flakeEuler.z;
  }

  private attachToSite(i: number, sIdx: number, initialGrowth: number): void {
    const safeIdx = sIdx % this.sites.length;
    const s = this.sites[safeIdx];
    const h = hash31Cpu(i * 2.3, safeIdx * 0.7, 5.1);
    this.state[i] = FLAKE_STATE_PERCHED;
    this.siteIdx[i] = safeIdx;

    // Dead-flush layered scab offset (`0.4..0.9 mm`) hugging the iron shank skin
    const liftDist = 0.0004 + h * 0.0005;
    this.px[i] = s.pos[0] + (h - 0.5) * 0.005 + s.norm[0] * liftDist;
    this.py[i] = s.pos[1] + s.norm[1] * liftDist;
    this.pz[i] = s.pos[2] + s.norm[2] * liftDist;

    this.baseYaw[i] = h * Math.PI * 2.0;
    this.baseTilt[i] = i < VISIBLE_NAIL_SCAB_COUNT ? 0.035 + h * 0.045 : (h - 0.5) * 0.12;
    this.hoverPeel[i] = 0.0;
    this.computePerchedEuler(i, 0.0);

    this.vx[i] = 0;
    this.vy[i] = 0;
    this.vz[i] = 0;
    this.wx[i] = 0;
    this.wy[i] = 0;
    this.wz[i] = 0;
    this.ageSec[i] = 0;
    this.blisterLift[i] = initialGrowth;
    if (this.instanceMetaData[i * 4 + 3] !== 0.0) {
      this.instanceMetaData[i * 4 + 3] = 0.0;
      this.metaDirty = true;
    }
  }

  private detachFlake(
    i: number,
    impX: number,
    impY: number,
    impZ: number,
    overridePos?: THREE.Vector3,
    overrideNorm?: THREE.Vector3
  ): void {
    const s = this.sites[this.siteIdx[i]];
    const nx = overrideNorm ? overrideNorm.x : s.norm[0];
    const ny = overrideNorm ? overrideNorm.y : s.norm[1];
    const nz = overrideNorm ? overrideNorm.z : s.norm[2];

    const h1 = hash31Cpu(i * 3.7, this.ageSec[i] + 1.1, 2.9);
    const h2 = hash31Cpu(i * 5.1, 4.4, this.ageSec[i] + 0.7);
    const h3 = hash31Cpu(i * 7.3, 2.2, this.ageSec[i] + 1.9);

    if (overridePos) {
      this.px[i] = overridePos.x + (h1 - 0.5) * 0.022 + nx * 0.004;
      this.py[i] = overridePos.y + (h2 - 0.5) * 0.016 + ny * 0.004;
      this.pz[i] = overridePos.z + (h3 - 0.5) * 0.022 + nz * 0.004;
      _normVec.set(nx, ny, nz).normalize();
      _alignQuat.setFromUnitVectors(_upVec, _normVec);
      _spinQuat.setFromAxisAngle(_upVec, h1 * Math.PI * 2.0);
      _tiltQuat.setFromAxisAngle(_xAxisVec, 0.32 + h2 * 0.35);
      _flakeQuat.copy(_alignQuat).multiply(_spinQuat).multiply(_tiltQuat);
      _flakeEuler.setFromQuaternion(_flakeQuat);
      this.rx[i] = _flakeEuler.x;
      this.ry[i] = _flakeEuler.y;
      this.rz[i] = _flakeEuler.z;
    }

    this.state[i] = FLAKE_STATE_FALLING;
    this.blisterLift[i] = 1.0;
    this.hoverPeel[i] = 0.0;
    this.ageSec[i] = 0;
    if (this.instanceMetaData[i * 4 + 3] !== 0.0) {
      this.instanceMetaData[i * 4 + 3] = 0.0;
      this.metaDirty = true;
    }

    // Tight, heavy crumbling detachment right off the scraped spot (ZERO wide sideways confetti scatter!)
    const downwardImpY = Math.min(-0.025, impY * 0.25);
    const lateralSpread = 0.024 + (1.0 - this.sizeTier[i]) * 0.018;
    this.vx[i] = nx * 0.032 + impX * 0.16 + (h1 - 0.5) * lateralSpread;
    this.vy[i] = downwardImpY - 0.045 - h2 * 0.085 + Math.min(0.0, ny * 0.04);
    this.vz[i] = nz * 0.032 + impZ * 0.16 + (h2 - 0.5) * lateralSpread;

    // Gentle damped bark-hinge tip-over instead of frantic sequin spinning
    const spinGain = 2.4 + (1.0 - this.sizeTier[i]) * 2.2;
    this.wx[i] = (h1 - 0.5) * spinGain;
    this.wy[i] = (h2 - 0.5) * (spinGain * 0.55);
    this.wz[i] = (0.5 - h3) * spinGain;
  }

  /**
   * Spawns a flaky multi-scale crumble (curled bark scabs + jagged splinters + fine rust grit)
   * directly off the exact scraped spot on the nail (`scrapePt`, `scrapeNorm`).
   */
  spawnScrapeCrumble(
    scrapePt: THREE.Vector3,
    scrapeNorm: THREE.Vector3,
    intensity: number,
    impulseVec: THREE.Vector3
  ): void {
    const contactR2 = 0.075 * 0.075;
    let peeledScabs = 0;
    const maxScabs = intensity > 0.95 ? 2 : 1;

    // 1. Pop off visible perched nail scabs right under the cursor first!
    for (let i = 0; i < VISIBLE_NAIL_SCAB_COUNT && peeledScabs < maxScabs; i++) {
      if (this.state[i] !== FLAKE_STATE_PERCHED || this.blisterLift[i] < 0.60) continue;
      const dx = this.px[i] - scrapePt.x;
      const dy = this.py[i] - scrapePt.y;
      const dz = this.pz[i] - scrapePt.z;
      if (dx * dx + dy * dy + dz * dz <= contactR2) {
        this.detachFlake(i, impulseVec.x, impulseVec.y, impulseVec.z);
        peeledScabs++;
      }
    }

    // 2. Emit crisp bark splinters + fine rust grit right from `scrapePt` (legible individual shards, zero overcrowded blob!)
    const targetBurst = Math.max(2, Math.min(8, Math.round(intensity * 5.0)));
    let spawned = 0;
    const poolSpan = DYNAMIC_POOL_END - VISIBLE_NAIL_SCAB_COUNT;

    for (let pass = 0; pass < poolSpan && spawned < targetBurst; pass++) {
      const idx = VISIBLE_NAIL_SCAB_COUNT + ((this.recycleCursor + pass * 17) % poolSpan);
      if (
        this.state[idx] === FLAKE_STATE_PERCHED ||
        (this.state[idx] === FLAKE_STATE_SETTLED && this.ageSec[idx] > 0.6)
      ) {
        // Keep big bark strips to 1-2 per scrape stroke unless gouging hard, pairing them with splinters & grit!
        if (this.sizeTier[idx] > 0.8 && peeledScabs >= maxScabs) continue;
        if (this.sizeTier[idx] > 0.8) peeledScabs++;
        this.detachFlake(idx, impulseVec.x, impulseVec.y, impulseVec.z, scrapePt, scrapeNorm);
        spawned++;
      }
    }

    this.recycleCursor = (this.recycleCursor + 19) % poolSpan;
  }

  /**
   * Dislodges brittle rust flakes near `localPoint` (pointer brush / tap shockwave).
   */
  dislodgeNear(
    localPoint: THREE.Vector3,
    radius: number,
    maxBurstCount: number,
    impulseVec: THREE.Vector3
  ): void {
    const r2 = radius * radius;
    let shed = 0;

    for (let pass = 0; pass < FLAKE_COUNT && shed < maxBurstCount; pass++) {
      const idx = (this.recycleCursor + pass * 11) % FLAKE_COUNT;
      if (this.state[idx] !== FLAKE_STATE_PERCHED || this.blisterLift[idx] < 0.65) continue;

      const dx = this.px[idx] - localPoint.x;
      const dy = this.py[idx] - localPoint.y;
      const dz = this.pz[idx] - localPoint.z;
      if (dx * dx + dy * dy + dz * dz <= r2) {
        this.detachFlake(idx, impulseVec.x, impulseVec.y, impulseVec.z);
        shed++;
      }
    }

    if (shed < Math.min(10, maxBurstCount)) {
      let bestSite = 0;
      let bestDist2 = 1e9;
      for (let s = 0; s < this.sites.length; s++) {
        const sp = this.sites[s].pos;
        const d2 =
          (sp[0] - localPoint.x) * (sp[0] - localPoint.x) +
          (sp[1] - localPoint.y) * (sp[1] - localPoint.y) +
          (sp[2] - localPoint.z) * (sp[2] - localPoint.z);
        if (d2 < bestDist2) {
          bestDist2 = d2;
          bestSite = s;
        }
      }

      for (let k = VISIBLE_NAIL_SCAB_COUNT; k < FLAKE_COUNT && shed < Math.min(16, maxBurstCount); k++) {
        if (this.state[k] === FLAKE_STATE_SETTLED && this.ageSec[k] > 0.8) {
          const sOff = (bestSite + shed * 5) % this.sites.length;
          this.attachToSite(k, sOff, 1.0);
          this.detachFlake(k, impulseVec.x, impulseVec.y, impulseVec.z);
          shed++;
        }
      }
    }

    this.recycleCursor = (this.recycleCursor + 23) % FLAKE_COUNT;
  }

  step(dtSec: number, elapsedSec: number, sim: TankHydroSim): void {
    const dt = Math.min(0.033, Math.max(0.002, dtSec));
    const bx = TANK_INNER_X - 0.035;
    const bz = TANK_INNER_Z - 0.035;
    const floorY = TANK_INNER_Y_BOT + 0.004;

    if (this.hasHoverNail) {
      this.hoverDwellSec += dt;
    }

    const sloshSpeed = Math.hypot(sim.sloshVelX, sim.sloshVelZ);
    if (sloshSpeed > 0.16) {
      const burst = sloshSpeed > 0.45 ? 3 : 1;
      for (let b = 0; b < burst; b++) {
        const idx = Math.floor(fractCpu(elapsedSec * 37.3 + b * 0.29) * FLAKE_COUNT);
        if (this.state[idx] === FLAKE_STATE_PERCHED && this.blisterLift[idx] > 0.75) {
          this.detachFlake(idx, sim.sloshVelX * 0.16, -0.02, sim.sloshVelZ * 0.16);
        }
      }
    }

    const pryRadius = 0.165;
    const pryR2 = pryRadius * pryRadius;

    for (let i = 0; i < FLAKE_COUNT; i++) {
      this.ageSec[i] += dt;
      const st = this.state[i];

      if (st === FLAKE_STATE_PERCHED) {
        if (this.blisterLift[i] < 1.0) {
          this.blisterLift[i] = Math.min(1.0, this.blisterLift[i] + dt * 0.38);
        }
        const s = this.sites[this.siteIdx[i]];

        if (i < VISIBLE_NAIL_SCAB_COUNT) {
          // Visible nail scabs: curl/pry upward when cursor hovers near them on the nail!
          let targetPeel = 0.0;
          if (this.hasHoverNail && this.blisterLift[i] > 0.7) {
            const dx = s.pos[0] - this.hoverNailPt.x;
            const dy = s.pos[1] - this.hoverNailPt.y;
            const dz = s.pos[2] - this.hoverNailPt.z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < pryR2) {
              const prox = 1.0 - Math.sqrt(d2) / pryRadius;
              targetPeel = prox * prox;
              // Stationary hover dwell gently pops off the closest lifted scab to cue interactivity!
              if (prox > 0.68 && this.hoverDwellSec > 0.22) {
                this.hoverDwellSec = 0.04;
                this.detachFlake(i, 0.0, -0.04, 0.02);
                continue;
              }
            }
          }

          const prevPeel = this.hoverPeel[i];
          const nextPeel = prevPeel + (targetPeel - prevPeel) * Math.min(1.0, dt * 14.0);
          if (Math.abs(nextPeel - prevPeel) > 1e-4) {
            this.hoverPeel[i] = nextPeel;
            this.computePerchedEuler(i, nextPeel * 0.52);
            this.instanceMetaData[i * 4 + 3] = nextPeel;
            this.metaDirty = true;
          }

          const liftOut = 0.0005 + this.hoverPeel[i] * 0.0048;
          this.px[i] = s.pos[0] + s.norm[0] * liftOut;
          this.py[i] = s.pos[1] + s.norm[1] * liftOut;
          this.pz[i] = s.pos[2] + s.norm[2] * liftOut;
        } else {
          const belowSpike = Math.max(0.0, (SPIKE_BASE_Y - 0.06 - s.pos[1]) * 0.85);
          const swayWt = Math.pow(Math.min(1.0, belowSpike), 1.65);
          this.px[i] = s.pos[0] + s.norm[0] * 0.003 + sim.sway1X * swayWt;
          this.pz[i] = s.pos[2] + s.norm[2] * 0.003 + sim.sway1Z * swayWt;
        }
      } else if (st === FLAKE_STATE_FALLING) {
        // Heavy iron-oxide scale vs fine grit differential Stokes settling!
        const ph = this.phase[i];
        const tier = this.sizeTier[i];
        const flutterFreq = 2.8 + (1.0 - tier) * 1.6 + (i % 4) * 0.32;
        const glideAmp = 0.022 + tier * 0.032;

        // Tilted-plate side-slip coupled to instantaneous inclination (heavy scabs slide along tilted planes!)
        const tiltSlipX = Math.sin(this.rz[i]) * 0.055 * tier + Math.sin(elapsedSec * flutterFreq + ph) * glideAmp;
        const tiltSlipZ = -Math.sin(this.rx[i]) * 0.055 * tier + Math.cos(elapsedSec * flutterFreq * 0.86 + ph) * glideAmp;

        const dx = this.px[i] - sim.stirCenter.x;
        const dz = this.pz[i] - sim.stirCenter.z;
        const d2 = dx * dx + dz * dz + 0.10;
        const eddy = (sim.stirEnergy - 0.06) / d2;

        // Big scales sink decisively (`-0.30 .. -0.46`); fine rust grit trails slower (`-0.14 .. -0.22`)!
        const plateHorizontal = Math.abs(Math.cos(this.rx[i]) * Math.cos(this.rz[i]));
        const baseSink = 0.15 + tier * 0.20;
        const edgeDive = (0.07 + tier * 0.14) * (1.0 - 0.65 * plateHorizontal);
        const targetFallVy = -(baseSink + edgeDive);

        const drag = Math.exp(-dt * 5.2);
        this.vx[i] =
          this.vx[i] * drag +
          (tiltSlipX + sim.sloshVelX * 0.24 - dz * eddy * 0.045 + sim.stirVelocity.x * 0.04) * (1.0 - drag);
        this.vy[i] = this.vy[i] * drag + targetFallVy * (1.0 - drag);
        this.vz[i] =
          this.vz[i] * drag +
          (tiltSlipZ + sim.sloshVelZ * 0.26 + dx * eddy * 0.050 + sim.stirVelocity.z * 0.05) * (1.0 - drag);

        this.px[i] = Math.max(-bx, Math.min(bx, this.px[i] + this.vx[i] * dt));
        this.py[i] += this.vy[i] * dt;
        this.pz[i] = Math.max(-bz, Math.min(bz, this.pz[i] + this.vz[i] * dt));

        // Viscous water strongly damps continuous tumbling into gentle rocking flutter (zero sequin strobe!)
        const spinDamp = Math.exp(-dt * 2.6);
        this.wx[i] *= spinDamp;
        this.wy[i] *= spinDamp;
        this.wz[i] *= spinDamp;
        const rockAmp = 0.85 + tier * 0.55;
        this.rx[i] += (this.wx[i] + Math.cos(elapsedSec * flutterFreq + ph) * rockAmp) * dt;
        this.ry[i] += this.wy[i] * dt;
        this.rz[i] += (this.wz[i] + Math.sin(elapsedSec * flutterFreq * 0.86 + ph) * rockAmp) * dt;

        if (this.py[i] <= floorY) {
          this.py[i] = floorY + (i % 6) * 0.0005;
          this.state[i] = FLAKE_STATE_SETTLED;
          this.ageSec[i] = 0;
          const faceUp = Math.cos(this.rx[i]) >= 0 ? 0.0 : Math.PI;
          this.rx[i] = faceUp + (hash31Cpu(i, 1.2, 3.4) - 0.5) * 0.20;
          this.rz[i] = (hash31Cpu(i, 5.6, 7.8) - 0.5) * 0.20;
        }
      } else {
        // SETTLED on bottom glass floor: build up a visible bed of flaked rust scales beneath the nail,
        // recycling back after ~16s so scraping never runs out of scabs!
        if (sloshSpeed > 0.35) {
          this.px[i] = Math.max(-bx, Math.min(bx, this.px[i] + sim.sloshVelX * 0.05 * dt));
          this.pz[i] = Math.max(-bz, Math.min(bz, this.pz[i] + sim.sloshVelZ * 0.05 * dt));
        }
        if (this.ageSec[i] > 16.0 && i < DYNAMIC_POOL_END) {
          if (i < VISIBLE_NAIL_SCAB_COUNT) {
            const nIdx = this.nailSiteIndices[(i * 7) % this.nailSiteIndices.length];
            this.attachToSite(i, nIdx, 0.08);
          } else {
            this.attachToSite(i, this.siteIdx[i], 0.25);
          }
        }
      }
    }

    if (this.metaDirty) {
      this.instanceMetaAttr.needsUpdate = true;
      this.metaDirty = false;
    }
    this.syncMatrices();
  }

  private syncMatrices(): void {
    for (let i = 0; i < FLAKE_COUNT; i++) {
      let vis = this.blisterLift[i];
      if (this.state[i] === FLAKE_STATE_PERCHED) {
        // Render the 180 nail-mounted scabs flush on the iron so hover-peeling is visibly obvious;
        // keep dynamic-pool reserve shards hidden (`0.0`) until scraped loose!
        vis = i < VISIBLE_NAIL_SCAB_COUNT ? 0.88 * this.blisterLift[i] : 0.0;
      }
      _flakePos.set(this.px[i], this.py[i], this.pz[i]);
      _flakeEuler.set(this.rx[i], this.ry[i], this.rz[i]);
      _flakeQuat.setFromEuler(_flakeEuler);
      // Align `scaleL` with `rim2D`'s long X splinter axis and `scaleW` with Z width -> crisp 2.6:1..3.8:1 bark splinters!
      _flakeScale.set(this.scaleL[i] * vis, this.scaleL[i] * 0.44 * vis, this.scaleW[i] * vis);
      _flakeMat.compose(_flakePos, _flakeQuat, _flakeScale);
      this.mesh.setMatrixAt(i, _flakeMat);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// 8. Zero-GC Sub-Millimeter Ferric Silt Puff System (Blooms on Flake Detachment)
// ---------------------------------------------------------------------------
const MOTE_COUNT = 180;
const _moteDummyMat = new THREE.Matrix4();
const _moteQuat = new THREE.Quaternion();
const _motePos = new THREE.Vector3();
const _moteScale = new THREE.Vector3();

export class FlocculeMoteSystem {
  readonly geometry: THREE.IcosahedronGeometry;
  readonly mesh: THREE.InstancedMesh;

  private readonly px = new Float32Array(MOTE_COUNT);
  private readonly py = new Float32Array(MOTE_COUNT);
  private readonly pz = new Float32Array(MOTE_COUNT);
  private readonly vx = new Float32Array(MOTE_COUNT);
  private readonly vy = new Float32Array(MOTE_COUNT);
  private readonly vz = new Float32Array(MOTE_COUNT);
  private readonly radii = new Float32Array(MOTE_COUNT);
  private readonly phase = new Float32Array(MOTE_COUNT);
  private puffCursor = 0;

  constructor(material: THREE.Material) {
    this.geometry = new THREE.IcosahedronGeometry(1.0, 0);
    this.mesh = new THREE.InstancedMesh(this.geometry, material, MOTE_COUNT);
    this.mesh.frustumCulled = false;

    for (let i = 0; i < MOTE_COUNT; i++) {
      this.respawnMote(i, true);
    }
    this.syncMatrices();
  }

  emitSiltPuff(origin: THREE.Vector3, count: number, velBias: THREE.Vector3): void {
    for (let k = 0; k < count; k++) {
      const i = (this.puffCursor + k) % MOTE_COUNT;
      const h1 = hash31Cpu(i * 1.9, k * 3.1, origin.x);
      const h2 = hash31Cpu(k * 2.7, i * 0.9, origin.z);
      const h3 = hash31Cpu(origin.y, i * 2.3, k * 1.7);

      this.px[i] = origin.x + (h1 - 0.5) * 0.045;
      this.py[i] = origin.y - 0.008 - h2 * 0.032;
      this.pz[i] = origin.z + (h3 - 0.5) * 0.045;

      this.vx[i] = velBias.x * 0.35 + (h1 - 0.5) * 0.045;
      this.vy[i] = Math.min(-0.015, velBias.y * 0.45 - 0.020 - h2 * 0.045);
      this.vz[i] = velBias.z * 0.35 + (h3 - 0.5) * 0.045;
    }
    this.puffCursor = (this.puffCursor + count) % MOTE_COUNT;
  }

  private respawnMote(i: number, initialSpread: boolean): void {
    const src = RUSTICLE_TIP_SOURCES[i % RUSTICLE_TIP_SOURCES.length];
    const h1 = hash31Cpu(i * 1.37, src.seed, 4.1);
    const h2 = hash31Cpu(i * 2.91, 7.3, src.seed);
    const h3 = hash31Cpu(src.seed, i * 0.83, 9.7);

    if (initialSpread) {
      this.px[i] = src.x + (h1 - 0.5) * 0.14;
      this.py[i] = src.y - h2 * 0.16 + 0.04;
      this.pz[i] = src.z + (h3 - 0.5) * 0.14;
    } else {
      this.px[i] = src.x + (h1 - 0.5) * 0.035;
      this.py[i] = Math.max(TANK_INNER_Y_BOT + 0.06, src.y + (h2 - 0.3) * 0.06);
      this.pz[i] = src.z + (h3 - 0.5) * 0.035;
    }

    this.vx[i] = (h1 - 0.5) * 0.012;
    this.vy[i] = -0.012 - h2 * 0.020;
    this.vz[i] = (h3 - 0.5) * 0.012;
    this.radii[i] = 0.0010 + Math.pow(h1 * h3, 1.4) * 0.0018;
    this.phase[i] = h2 * Math.PI * 2.0;
  }

  step(dtSec: number, elapsedSec: number, sim: TankHydroSim): void {
    const dt = Math.min(0.033, Math.max(0.002, dtSec));
    const bx = TANK_INNER_X - 0.05;
    const bz = TANK_INNER_Z - 0.05;

    for (let i = 0; i < MOTE_COUNT; i++) {
      let x = this.px[i];
      let y = this.py[i];
      let z = this.pz[i];

      const ph = this.phase[i];
      const curlX = Math.sin(y * 4.2 + elapsedSec * 0.52 + ph) * 0.014;
      const curlY = Math.cos(x * 3.8 - elapsedSec * 0.40 + ph) * 0.006 - 0.018;
      const curlZ = Math.cos(y * 4.6 + x * 2.5 + elapsedSec * 0.46) * 0.014;

      const dx = x - sim.stirCenter.x;
      const dz = z - sim.stirCenter.z;
      const d2 = dx * dx + dz * dz + 0.09;
      const eddy = (sim.stirEnergy - 0.06) / d2;

      this.vx[i] = this.vx[i] * 0.94 + (curlX + sim.sloshVelX * 0.25 - dz * eddy * 0.035 + sim.stirVelocity.x * 0.030) * dt * 6.0;
      this.vy[i] = this.vy[i] * 0.95 + (curlY - Math.min(0.030, Math.abs(eddy) * 0.003)) * dt * 6.0;
      this.vz[i] = this.vz[i] * 0.94 + (curlZ + sim.sloshVelZ * 0.25 + dx * eddy * 0.035 + sim.stirVelocity.z * 0.030) * dt * 6.0;

      x += this.vx[i] * dt * 7.5;
      y += this.vy[i] * dt * 7.5;
      z += this.vz[i] * dt * 7.5;

      if (y < TANK_INNER_Y_BOT + 0.04 || y > TANK_INNER_Y_TOP - 0.04 || Math.abs(x) > bx || Math.abs(z) > bz) {
        this.respawnMote(i, false);
      } else {
        this.px[i] = x;
        this.py[i] = y;
        this.pz[i] = z;
      }
    }

    this.syncMatrices();
  }

  private syncMatrices(): void {
    for (let i = 0; i < MOTE_COUNT; i++) {
      const r = this.radii[i];
      _motePos.set(this.px[i], this.py[i], this.pz[i]);
      _moteScale.set(r * 1.2, r * 0.9, r);
      _moteDummyMat.compose(_motePos, _moteQuat, _moteScale);
      this.mesh.setMatrixAt(i, _moteDummyMat);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
