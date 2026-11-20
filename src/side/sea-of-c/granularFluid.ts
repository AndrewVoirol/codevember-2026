import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  floor,
  length,
  max,
  mix,
  normalize,
  normalGeometry,
  positionGeometry,
  pow,
  select,
  sin,
  smoothstep,
  sqrt,
  transformNormalToView,
  uint,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import {
  HERO_SOLITONS,
  WAKE_SLOT_COUNT,
  evaluateWaveTSL,
  type WaveFieldUniforms,
} from './waveField';
import { createBeveledCGeometry, createSprayCGeometry } from './glyphGeometry';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface GranularFluidRig {
  group: THREE.Group;
  glassMesh: THREE.InstancedMesh;
  sprayMesh: THREE.InstancedMesh;
  glassCount: number;
  sprayCount: number;
  dispose: () => void;
}

// Instance counts calibrated for dense full-wall curling crests + crisp spray well under 16.6 ms
const GLASS_SOLITON_COUNTS = [2200, 1200, 600] as const; // 4,000
const GLASS_STRATIFIED_COUNT = 450; // Flanking swells strictly outside |x| <= 3.2 central sun lane
const GLASS_VORTEX_COUNT = 350;
const GLASS_ROGUE_COUNT = 240;
const GLASS_WAKE_COUNT = 160; // Interactive total: 750
const GLASS_TOTAL_COUNT =
  GLASS_SOLITON_COUNTS[0] +
  GLASS_SOLITON_COUNTS[1] +
  GLASS_SOLITON_COUNTS[2] +
  GLASS_STRATIFIED_COUNT +
  GLASS_VORTEX_COUNT +
  GLASS_ROGUE_COUNT +
  GLASS_WAKE_COUNT; // 5,200

const SPRAY_SOLITON_COUNTS = [880, 460, 260] as const; // 1,600
const SPRAY_VORTEX_COUNT = 320;
const SPRAY_ROGUE_COUNT = 220;
const SPRAY_WAKE_COUNT = 260; // Interactive total: 800
const SPRAY_TOTAL_COUNT =
  SPRAY_SOLITON_COUNTS[0] +
  SPRAY_SOLITON_COUNTS[1] +
  SPRAY_SOLITON_COUNTS[2] +
  SPRAY_VORTEX_COUNT +
  SPRAY_ROGUE_COUNT +
  SPRAY_WAKE_COUNT; // 2,400

// Exact rogue wave constants mirroring waveField.ts so interactive 3D crests track 1:1
const ROGUE_DIR_LEN = Math.hypot(-0.34, 0.94);
const ROGUE_DIR_X = -0.34 / ROGUE_DIR_LEN;
const ROGUE_DIR_Z = 0.94 / ROGUE_DIR_LEN;
const ROGUE_TAN_X = -ROGUE_DIR_Z;
const ROGUE_TAN_Z = ROGUE_DIR_X;
const ROGUE_SIGMA_N = 2.45;
const ROGUE_SIGMA_T = 4.8;
const ROGUE_BOW = 0.038;
const ROGUE_SPEED = 4.2;

/**
 * Deterministic PRNG (Mulberry32) so blue-noise instance packing is 100% reproducible.
 */
function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Halton low-discrepancy radical inverse for organic, clump-free spatial packing.
 */
function halton(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

/**
 * Sets all instance matrices on an InstancedMesh to IDENTITY once at init.
 * Per Three.js r187 NodeMaterial.setupPosition, instanceMatrix * positionLocal runs
 * BEFORE material.positionNode; keeping instanceMatrix = I and transforming directly
 * from positionGeometry guarantees zero CPU per-frame matrix uploads and zero offset bugs.
 */
function initIdentityInstances(mesh: THREE.InstancedMesh, count: number): void {
  const identity = new THREE.Matrix4();
  for (let i = 0; i < count; i++) {
    mesh.setMatrixAt(i, identity);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
}

/**
 * Negative-safe fractional part (`x - floor(x)`) preventing Metal fast-math `fract()` bugs.
 */
const safeFract1 = Fn(([x_in]: [TslNode]) => {
  const x = float(x_in);
  return x.sub(floor(x));
});

/**
 * Rodrigues axis-angle 3D rotation of vector `v` around unit axis `axis` by `angle` radians.
 * Strictly orthonormal (preserves vector length for both positions and surface normals).
 */
const rotateAxisAngle = Fn(
  ([v_in, axis_in, angle_in]: [TslNode, TslNode, TslNode]) => {
    const v = vec3(v_in);
    const k = normalize(vec3(axis_in));
    const a = float(angle_in);
    const c = cos(a);
    const s = sin(a);
    return v
      .mul(c)
      .add(cross(k, v).mul(s))
      .add(k.mul(dot(k, v).mul(float(1.0).sub(c))));
  }
);

/**
 * Smooth minimal-rotation SO(3) mapping that carries +Y (0, 1, 0) onto wave normal N
 * with zero branch discontinuities and zero gimbal lock across flat troughs, vertical
 * wave faces, and curling overhangs (N.y > -0.92).
 */
const rotateUpToNormal = Fn(([v_in, n_in]: [TslNode, TslNode]) => {
  const v = vec3(v_in);
  const n = normalize(vec3(n_in));
  const nx = n.x;
  const ny = n.y;
  const nz = n.z;
  const denom = max(float(1.0).add(ny), float(0.08));
  const k = float(1.0).div(denom);

  const colX = vec3(
    float(1.0).sub(nx.mul(nx).mul(k)),
    nx.negate(),
    nx.mul(nz).mul(k).negate()
  );
  const colY = n;
  const colZ = vec3(
    nx.mul(nz).mul(k).negate(),
    nz.negate(),
    float(1.0).sub(nz.mul(nz).mul(k))
  );

  return colX.mul(v.x).add(colY.mul(v.y)).add(colZ.mul(v.z));
});

/**
 * Chroma-preserving luminance compression keeping emissive strictly below `cap` linear
 * so rich emerald-turquoise and molten golden-amber glass NEVER clip or desaturate to milk-white.
 */
const softCapChroma = Fn(([col_in, cap_in]: [TslNode, TslNode]) => {
  const col = max(vec3(col_in), vec3(0.0));
  const cap = max(float(cap_in), float(0.05));
  const peak = max(col.x, max(col.y, col.z));
  const scale = cap.div(cap.add(peak));
  return col.mul(scale);
});

interface SolitonTrackNodes {
  restXZ: TslNode;
  dirXZ: TslNode;
  tanXZ: TslNode;
  riseEnv: TslNode;
  curlLife: TslNode;
  envT: TslNode;
}

/**
 * Evaluates live traveling soliton anchor coordinates & lifecycle on GPU matching
 * `HERO_SOLITONS` in `waveField.ts` identically.
 */
function buildSolitonTrackerTSL(
  uCoord: TslNode,
  vCoord: TslNode,
  solIdx: TslNode,
  time: TslNode
): SolitonTrackNodes {
  const s0 = HERO_SOLITONS[0];
  const s1 = HERO_SOLITONS[1];
  const s2 = HERO_SOLITONS[2];

  const evalSingle = (sol: (typeof HERO_SOLITONS)[number]) => {
    const raw = time.div(sol.period).add(sol.phase0);
    const cycle = raw.sub(floor(raw));
    const riseEnv = smoothstep(float(0.0), float(0.34), cycle).mul(
      float(1.0).sub(smoothstep(float(0.7), float(1.0), cycle))
    );
    const curlLife = smoothstep(float(0.2), float(0.5), cycle).mul(
      float(1.0).sub(smoothstep(float(0.68), float(0.94), cycle))
    );
    const travel = cycle.sub(0.48).mul(sol.travelSpan);
    const cx = travel.mul(sol.dirX).add(sol.baseX);
    const cz = travel.mul(sol.dirZ).add(sol.baseZ);

    const tDist = vCoord.mul(sol.sigmaT);
    const nRaw = uCoord.mul(sol.sigmaN).add(tDist.mul(tDist).mul(sol.bow));
    const wx = cx.add(nRaw.mul(sol.dirX)).add(tDist.mul(sol.tanX));
    const wz = cz.add(nRaw.mul(sol.dirZ)).add(tDist.mul(sol.tanZ));

    return {
      restXZ: vec2(wx, wz),
      dirXZ: vec2(sol.dirX, sol.dirZ),
      tanXZ: vec2(sol.tanX, sol.tanZ),
      riseEnv,
      curlLife,
    };
  };

  const k0 = evalSingle(s0);
  const k1 = evalSingle(s1);
  const k2 = evalSingle(s2);

  const is0 = solIdx.lessThan(float(0.5));
  const is1 = solIdx.lessThan(float(1.5));

  const restXZ = select(is0, k0.restXZ, select(is1, k1.restXZ, k2.restXZ));
  const dirXZ = select(is0, k0.dirXZ, select(is1, k1.dirXZ, k2.dirXZ));
  const tanXZ = select(is0, k0.tanXZ, select(is1, k1.tanXZ, k2.tanXZ));
  const riseEnv = select(is0, k0.riseEnv, select(is1, k1.riseEnv, k2.riseEnv));
  const curlLife = select(is0, k0.curlLife, select(is1, k1.curlLife, k2.curlLife));
  const envT = exp(vCoord.mul(vCoord).negate());

  return { restXZ, dirXZ, tanXZ, riseEnv, curlLife, envT };
}

/**
 * Populates static per-instance attributes (`aAnchor`, `aSeed`, `aLayer`) for the
 * 4,800 beveled 3D sea-glass `C` instances across 3 smart pools:
 *  - Pool 0 (`mode = 0`): Soliton-locked breaking crests & translucent bellies (3,350)
 *  - Pool 1 (`mode = 1`): Crest-only foreground swells leaving central sun-path open (700)
 *  - Pool 2 (`mode = 2`): Dynamic interactive vortex, rogue breaker & wake surges (750)
 */
function buildGlassAttributes(geom: THREE.BufferGeometry): void {
  const anchorData = new Float32Array(GLASS_TOTAL_COUNT * 4);
  const seedData = new Float32Array(GLASS_TOTAL_COUNT * 4);
  const layerData = new Float32Array(GLASS_TOTAL_COUNT * 4);
  const rng = createRng(0x5ea0c317);

  let idx = 0;

  const pushInstance = (
    a0: number,
    a1: number,
    a2: number,
    a3: number,
    s0: number,
    s1: number,
    s2: number,
    s3: number,
    l0: number,
    l1: number,
    l2: number,
    l3: number
  ) => {
    const o = idx * 4;
    anchorData[o] = a0;
    anchorData[o + 1] = a1;
    anchorData[o + 2] = a2;
    anchorData[o + 3] = a3;

    seedData[o] = s0;
    seedData[o + 1] = s1;
    seedData[o + 2] = s2;
    seedData[o + 3] = s3;

    layerData[o] = l0;
    layerData[o + 1] = l1;
    layerData[o + 2] = l2;
    layerData[o + 3] = l3;
    idx++;
  };

  // ---------------------------------------------------------------------------
  // Pool 0: Soliton-Locked Full Front-Wall, Belly, Lip & Back-Shoulder (4,000)
  // Covers uCoord in [-0.65, +1.35] so the entire camera-facing curling wave belly
  // and lower sapphire trough entry are densely wrapped with 3D sea-glass C's!
  // ---------------------------------------------------------------------------
  for (let solIdx = 0; solIdx < HERO_SOLITONS.length; solIdx++) {
    const count = GLASS_SOLITON_COUNTS[solIdx];
    for (let j = 0; j < count; j++) {
      const h1 = halton(j + 17 + solIdx * 997, 2);
      const h2 = halton(j + 17 + solIdx * 997, 3);
      const h3 = halton(j + 17 + solIdx * 997, 5);

      // Partition full soliton cross-section (u = +0.30 is crest apex, u > +0.30 is steep front wall!):
      // 12% back shoulder (u in [-0.65, +0.16]) — royal sapphire wrapping over rear ridge
      // 30% lower front trough entry & sweeping base (u in [+0.68, +1.35]) — shingled sapphire base
      // 38% steep front translucent curling belly (u in [+0.24, +0.76]) — dense jewel emerald/amber
      // 20% crest crown & overhanging curling lip (u in [+0.06, +0.42]) — tumbling aqua/frosted lip
      const band = j / count;
      let uCoord: number;
      let lipArch = 0;
      let layerDepth: number;
      let sizeClass: number;

      if (band < 0.12) {
        // Back shoulder: solid sapphire coverage from elevated panoramic views
        uCoord = -0.65 + h1 * 0.81 + (rng() - 0.5) * 0.05;
        layerDepth = h3 * 0.58;
        lipArch = 0.0;
        sizeClass = 0.48 + rng() * 0.40;
      } else if (band < 0.42) {
        // Lower front trough entry & sweeping front base: bold sapphire C's rising out of ocean
        uCoord = 0.68 + h1 * 0.67 + (rng() - 0.5) * 0.06;
        layerDepth = h3 * 0.68;
        lipArch = 0.0;
        sizeClass = 0.50 + rng() * 0.46;
      } else if (band < 0.80) {
        // Hero front steep curling belly: highest multi-layer density of jewel emerald & amber C's
        uCoord = 0.24 + h1 * 0.52 + (rng() - 0.5) * 0.05;
        layerDepth = h3;
        // Upper edge of belly smoothly feeds into the forward-arching lip
        lipArch = Math.max(0, (0.46 - uCoord) / 0.24) * (0.12 + 0.34 * rng());
        sizeClass = 0.34 + rng() * 0.44;
      } else {
        // Compact crest crown & overhanging curling lip feeding airborne spray
        uCoord = 0.06 + h1 * 0.36 + (rng() - 0.5) * 0.04;
        layerDepth = 0.28 + h3 * 0.72;
        lipArch = 0.34 + rng() * 0.66;
        sizeClass = 0.12 + rng() * 0.34;
      }

      // Along-ridge profile concentrated in breaking center, feathering cleanly down flanks
      const sv = h2 * 2.0 - 1.0;
      const vCoord = sv * (0.64 + 0.58 * Math.abs(sv)) + (rng() - 0.5) * 0.06;
      const aspect = 0.88 + rng() * 0.24;

      pushInstance(
        uCoord,
        vCoord,
        solIdx,
        0.0, // poolMode = 0 (Hero Soliton)
        rng(), // s0: tumble axis azimuth
        rng(), // s1: tumble axis pitch / chroma jitter
        rng(), // s2: phase offset / yaw jitter
        rng(), // s3: thickness & subtle proportion variation
        layerDepth,
        lipArch,
        sizeClass,
        aspect
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Pool 1: Crest-Only Flanking Swells (450)
  // Strictly avoids |x| <= 3.2 and bottom foreground z > 7.2 so central golden sun lane stays open
  // ---------------------------------------------------------------------------
  for (let j = 0; j < GLASS_STRATIFIED_COUNT; j++) {
    const h1 = halton(j + 131, 2);
    const h2 = halton(j + 131, 3);
    const h3 = halton(j + 131, 5);

    // Place on left/right flanking swells (|x| in [3.4, 15.5]) and midground z in [-15.0, 6.8]
    const side = j % 2 === 0 ? 1.0 : -1.0;
    const x = side * (3.4 + Math.pow(h1, 0.88) * 12.1) + (rng() - 0.5) * 0.35;
    const zBiased = 1.0 - Math.pow(h2, 1.25);
    const z = -15.0 + zBiased * 21.8 + (rng() - 0.5) * 0.35;

    // Smooth radial edge fade (zero rectangular box boundary)
    const rx = x / 16.0;
    const rz = (z + 3.5) / 13.0;
    const radialDist = Math.hypot(rx, rz);
    const radialFade = Math.max(0, Math.min(1, 1.0 - Math.max(0, (radialDist - 0.62) / 0.38)));

    const layerDepth = h3 * 0.62;
    const sizeClass = 0.34 + rng() * 0.42;
    const aspect = 0.90 + rng() * 0.20;

    pushInstance(
      x,
      z,
      radialFade,
      1.0, // poolMode = 1 (Crest-only Flanking Swells)
      rng(),
      rng(),
      rng(),
      rng(),
      layerDepth,
      0.0,
      sizeClass,
      aspect
    );
  }

  // ---------------------------------------------------------------------------
  // Pool 2A: Interactive Maelstrom Vortex Ring Wall (350)
  // ---------------------------------------------------------------------------
  for (let j = 0; j < GLASS_VORTEX_COUNT; j++) {
    const h1 = halton(j + 211, 2);
    const h2 = halton(j + 211, 3);
    const h3 = halton(j + 211, 5);

    const r = 0.90 + Math.pow(h1, 0.78) * 3.85;
    const theta = h2 * Math.PI * 2.0 + (rng() - 0.5) * 0.14;
    const layerDepth = h3;
    const lipArch = Math.exp(-Math.pow((r - 2.6) / 0.90, 2)) * (0.20 + 0.75 * rng());
    const sizeClass = 0.24 + rng() * 0.50;
    const aspect = 0.88 + rng() * 0.24;

    pushInstance(
      r,
      theta,
      0.0, // subMode = 0 (Vortex)
      2.0, // poolMode = 2 (Interactive)
      rng(),
      rng(),
      rng(),
      rng(),
      layerDepth,
      lipArch,
      sizeClass,
      aspect
    );
  }

  // ---------------------------------------------------------------------------
  // Pool 2B: Interactive Traveling Rogue Breaker Surge (240)
  // ---------------------------------------------------------------------------
  for (let j = 0; j < GLASS_ROGUE_COUNT; j++) {
    const h1 = halton(j + 307, 2);
    const h2 = halton(j + 307, 3);
    const h3 = halton(j + 307, 5);

    const uCoord = -0.72 + h1 * 1.24 + (rng() - 0.5) * 0.06;
    const sv = h2 * 2.0 - 1.0;
    const vCoord = sv * (0.68 + 0.62 * Math.abs(sv));
    const layerDepth = h3;
    const lipArch = Math.max(0, (uCoord - 0.10) / 0.45) * (0.25 + 0.70 * rng());
    const sizeClass = 0.22 + rng() * 0.50;
    const aspect = 0.88 + rng() * 0.24;

    pushInstance(
      uCoord,
      vCoord,
      1.0, // subMode = 1 (Rogue Wave)
      2.0, // poolMode = 2 (Interactive)
      rng(),
      rng(),
      rng(),
      rng(),
      layerDepth,
      lipArch,
      sizeClass,
      aspect
    );
  }

  // ---------------------------------------------------------------------------
  // Pool 2C: Interactive 8-Slot Expanding Pointer Wake Rings (160)
  // ---------------------------------------------------------------------------
  for (let j = 0; j < GLASS_WAKE_COUNT; j++) {
    const slotIdx = j % WAKE_SLOT_COUNT;
    const localJ = Math.floor(j / WAKE_SLOT_COUNT);
    const theta = halton(localJ + 19 + slotIdx * 43, 2) * Math.PI * 2.0 + (rng() - 0.5) * 0.2;
    const ringJitter = (halton(localJ + 19 + slotIdx * 43, 3) - 0.5) * 1.1;
    const layerDepth = rng() * 0.65;
    const sizeClass = 0.20 + rng() * 0.38;
    const aspect = 0.90 + rng() * 0.20;

    pushInstance(
      theta,
      ringJitter,
      2.0 + slotIdx, // subMode >= 2 encodes (2 + slotIdx)
      2.0, // poolMode = 2 (Interactive)
      rng(),
      rng(),
      rng(),
      rng(),
      layerDepth,
      0.18 * rng(),
      sizeClass,
      aspect
    );
  }

  const aAnchor = new THREE.InstancedBufferAttribute(anchorData, 4);
  const aSeed = new THREE.InstancedBufferAttribute(seedData, 4);
  const aLayer = new THREE.InstancedBufferAttribute(layerData, 4);
  geom.setAttribute('aAnchor', aAnchor);
  geom.setAttribute('aSeed', aSeed);
  geom.setAttribute('aLayer', aLayer);
}

/**
 * Populates static per-instance attributes (`aAnchor`, `aSeed`, `aLayer`) for the
 * 2,400 airborne frosted-white / pale-cyan 3D `c` foam spray droplets:
 *  - Pool 0 (`mode = 0`): Hero Soliton breaking-lip spray plumes (1,600)
 *  - Pool 1 (`mode = 1`): Interactive vortex, rogue wave & pointer wake bursts (800)
 */
function buildSprayAttributes(geom: THREE.BufferGeometry): void {
  const anchorData = new Float32Array(SPRAY_TOTAL_COUNT * 4);
  const seedData = new Float32Array(SPRAY_TOTAL_COUNT * 4);
  const layerData = new Float32Array(SPRAY_TOTAL_COUNT * 4);
  const rng = createRng(0xc0de59a7);

  let idx = 0;

  const pushSpray = (
    a0: number,
    a1: number,
    a2: number,
    a3: number,
    s0: number,
    s1: number,
    s2: number,
    s3: number,
    l0: number,
    l1: number,
    l2: number,
    l3: number
  ) => {
    const o = idx * 4;
    anchorData[o] = a0;
    anchorData[o + 1] = a1;
    anchorData[o + 2] = a2;
    anchorData[o + 3] = a3;

    seedData[o] = s0;
    seedData[o + 1] = s1;
    seedData[o + 2] = s2;
    seedData[o + 3] = s3;

    layerData[o] = l0;
    layerData[o + 1] = l1;
    layerData[o + 2] = l2;
    layerData[o + 3] = l3;
    idx++;
  };

  // Pool 0: Hero Soliton Breaking-Lip Spray (1,600)
  for (let solIdx = 0; solIdx < HERO_SOLITONS.length; solIdx++) {
    const count = SPRAY_SOLITON_COUNTS[solIdx];
    for (let j = 0; j < count; j++) {
      const h1 = halton(j + 29 + solIdx * 733, 2);
      const h2 = halton(j + 29 + solIdx * 733, 3);
      const h3 = halton(j + 29 + solIdx * 733, 5);

      const uCoord = 0.16 + h1 * 0.34 + (rng() - 0.5) * 0.05;
      const sv = h2 * 2.0 - 1.0;
      const vCoord = sv * (0.55 + 0.50 * Math.abs(sv)) + (rng() - 0.5) * 0.06;

      const rate = 0.58 + rng() * 0.48;
      const phaseOffset = h3;
      const sizeRand = rng();
      const spinSeed = rng();

      const velUpBias = 0.52 + rng() * 0.88;
      const velFwdBias = 0.42 + rng() * 0.96;
      const fanSpread = (rng() - 0.5) * 1.65;
      const archStart = 0.55 + rng() * 0.45;

      pushSpray(
        uCoord,
        vCoord,
        solIdx,
        0.0, // poolMode = 0 (Hero Soliton Spray)
        rate,
        phaseOffset,
        sizeRand,
        spinSeed,
        velUpBias,
        velFwdBias,
        fanSpread,
        archStart
      );
    }
  }

  // Pool 1A: Vortex Maelstrom Spray (320)
  for (let j = 0; j < SPRAY_VORTEX_COUNT; j++) {
    const h1 = halton(j + 53, 2);
    const h2 = halton(j + 53, 3);
    const r = 1.45 + h1 * 2.25;
    const theta = h2 * Math.PI * 2.0 + (rng() - 0.5) * 0.18;

    pushSpray(
      r,
      theta,
      0.0, // subMode = 0 (Vortex)
      1.0, // poolMode = 1 (Interactive Spray)
      0.64 + rng() * 0.52,
      rng(),
      rng(),
      rng(),
      0.52 + rng() * 0.88,
      0.45 + rng() * 0.95,
      (rng() - 0.5) * 1.8,
      0.45 + rng() * 0.55
    );
  }

  // Pool 1B: Rogue Wave Breaking-Lip Spray (220)
  for (let j = 0; j < SPRAY_ROGUE_COUNT; j++) {
    const h1 = halton(j + 89, 2);
    const h2 = halton(j + 89, 3);
    const uCoord = 0.15 + h1 * 0.36;
    const sv = h2 * 2.0 - 1.0;
    const vCoord = sv * (0.60 + 0.50 * Math.abs(sv));

    pushSpray(
      uCoord,
      vCoord,
      1.0, // subMode = 1 (Rogue Wave)
      1.0, // poolMode = 1 (Interactive Spray)
      0.62 + rng() * 0.50,
      rng(),
      rng(),
      rng(),
      0.60 + rng() * 0.88,
      0.50 + rng() * 1.0,
      (rng() - 0.5) * 1.75,
      0.55 + rng() * 0.45
    );
  }

  // Pool 1C: 8-Slot Pointer Wake Spray (260)
  for (let j = 0; j < SPRAY_WAKE_COUNT; j++) {
    const slotIdx = j % WAKE_SLOT_COUNT;
    const localJ = Math.floor(j / WAKE_SLOT_COUNT);
    const theta = halton(localJ + 11 + slotIdx * 29, 2) * Math.PI * 2.0 + (rng() - 0.5) * 0.25;
    const ringJitter = (halton(localJ + 11 + slotIdx * 29, 3) - 0.5) * 0.85;

    pushSpray(
      theta,
      ringJitter,
      2.0 + slotIdx, // subMode >= 2 encodes (2 + slotIdx)
      1.0, // poolMode = 1 (Interactive Spray)
      0.74 + rng() * 0.60,
      rng(),
      rng(),
      rng(),
      0.45 + rng() * 0.80,
      0.38 + rng() * 0.85,
      (rng() - 0.5) * 1.7,
      0.25 + rng() * 0.45
    );
  }

  const aAnchor = new THREE.InstancedBufferAttribute(anchorData, 4);
  const aSeed = new THREE.InstancedBufferAttribute(seedData, 4);
  const aLayer = new THREE.InstancedBufferAttribute(layerData, 4);
  geom.setAttribute('aAnchor', aAnchor);
  geom.setAttribute('aSeed', aSeed);
  geom.setAttribute('aLayer', aLayer);
}

/**
 * Builds the 2-layer GPU-driven 3D beveled sea-glass `C` fluid (`glassMesh`, 4,800) and
 * ballistic airborne `c` foam spray (`sprayMesh`, 2,400), mathematically locked to
 * `evaluateWaveTSL` with zero per-frame CPU buffer uploads.
 */
export function createGranularFluid(
  u: WaveFieldUniforms,
  envMap: THREE.Texture
): GranularFluidRig {
  const group = new THREE.Group();
  group.name = 'SeaOfC_GranularFluid';

  // ===========================================================================
  // LAYER 1: 3D BEVELED SEA-GLASS 'C' FLUID (`glassMesh`, 4,800 instances)
  // ===========================================================================
  const glassGeom = createBeveledCGeometry();
  buildGlassAttributes(glassGeom);

  const glassMat = new THREE.MeshPhysicalNodeMaterial({
    envMap,
    envMapIntensity: 0.52,
    metalness: 0.03,
    roughness: 0.20,
    clearcoat: 0.85,
    clearcoatRoughness: 0.16,
    ior: 1.48,
    fog: false,
    depthWrite: true,
    depthTest: true,
  });

  {
    const aAnchor = attribute('aAnchor', 'vec4');
    const aSeed = attribute('aSeed', 'vec4');
    const aLayer = attribute('aLayer', 'vec4');
    const time = float(u.uTime);
    const breakerHeight = clamp(float(u.uBreakerHeight), float(0.25), float(1.85));

    const poolMode = aAnchor.w;
    const isSolitonPool = poolMode.lessThan(float(0.5));
    const isStratifiedPool = poolMode
      .greaterThanEqual(float(0.5))
      .and(poolMode.lessThan(float(1.5)));

    // --- Pool 0: Hero Soliton traveling coordinates ---
    const solTrack = buildSolitonTrackerTSL(aAnchor.x, aAnchor.y, aAnchor.z, time);
    const solU = aAnchor.x;
    // Overhanging lip arch strictly around the crest crown (u in [0.06, 0.46])
    const solLipZone = smoothstep(float(0.04), float(0.22), solU).mul(
      float(1.0).sub(smoothstep(float(0.36), float(0.56), solU))
    );
    // Translucent jewel emerald/amber backlit belly sits tightly inside the thin upper throat
    // (u in [0.18, 0.64], narrowed along the ridge via envT^1.8) so lower trough base, dark
    // back shoulder, and outer flanks stay deep glossy Prussian-sapphire glass like fused_hero_sea_of_c!
    const solInnerWall = smoothstep(float(0.14), float(0.32), solU).mul(
      float(1.0).sub(smoothstep(float(0.50), float(0.76), solU))
    );
    const solBellyZone = solInnerWall.mul(
      pow(clamp(float(solTrack.envT), float(1e-4), float(1.0)), float(1.8))
    );
    // Smooth flush toe taper at lower front entry (u -> 1.35) & back edge (u -> -0.65)
    const solToeTaper = smoothstep(float(1.40), float(1.04), solU).mul(
      smoothstep(float(-0.70), float(-0.36), solU)
    );

    // --- Pool 2A: Interactive Maelstrom Vortex coordinates ---
    const vortex = vec4(u.uVortex);
    const vStrength = clamp(vortex.z, float(0.0), float(2.0));
    const vRadius = aAnchor.x;
    const vTheta = aAnchor.y.add(
      vortex.w.mul(float(1.35).div(vRadius.mul(0.38).add(0.62)))
    );
    const vCos = cos(vTheta);
    const vSin = sin(vTheta);
    const vortexRestXZ = vec2(
      vortex.x.add(vCos.mul(vRadius)),
      vortex.y.add(vSin.mul(vRadius))
    );
    const vortexDirXZ = normalize(
      vec2(
        vSin.negate().mul(0.82).add(vCos.mul(0.57)),
        vCos.mul(0.82).add(vSin.mul(0.57))
      )
    );
    const vortexTanXZ = vec2(vortexDirXZ.y.negate(), vortexDirXZ.x);
    const vortexActive = smoothstep(float(0.02), float(0.30), vStrength);
    const vRingProx = exp(vRadius.sub(2.6).mul(vRadius.sub(2.6)).mul(-0.46));

    // --- Pool 2B: Interactive Traveling Rogue Wave coordinates ---
    const rogue = vec4(u.uRogueWave);
    const rogueAge = time.sub(rogue.z);
    const rogueAmp = clamp(rogue.w, float(0.0), float(2.5));
    const rogueLife = smoothstep(float(0.02), float(0.38), rogueAge)
      .mul(float(1.0).sub(smoothstep(float(2.8), float(5.2), rogueAge)))
      .mul(rogueAmp);
    const rogueCurl = smoothstep(float(0.16), float(0.68), rogueAge).mul(
      float(1.0).sub(smoothstep(float(2.6), float(4.8), rogueAge))
    );
    const rogueAgePos = max(float(0.0), rogueAge);
    const rogueCx = rogue.x.add(rogueAgePos.mul(ROGUE_DIR_X * ROGUE_SPEED));
    const rogueCz = rogue.y.add(rogueAgePos.mul(ROGUE_DIR_Z * ROGUE_SPEED));
    const rogueTDist = aAnchor.y.mul(ROGUE_SIGMA_T);
    const rogueNRaw = aAnchor.x
      .mul(ROGUE_SIGMA_N)
      .add(rogueTDist.mul(rogueTDist).mul(ROGUE_BOW));
    const rogueRestXZ = vec2(
      rogueCx.add(rogueNRaw.mul(ROGUE_DIR_X)).add(rogueTDist.mul(ROGUE_TAN_X)),
      rogueCz.add(rogueNRaw.mul(ROGUE_DIR_Z)).add(rogueTDist.mul(ROGUE_TAN_Z))
    );
    const rogueEnvT = exp(aAnchor.y.mul(aAnchor.y).negate());
    const rogueActive = smoothstep(float(0.02), float(0.32), rogueLife);

    // --- Pool 2C: Interactive 8-Slot Wake Ring coordinates ---
    const wakeSlotIdx = uint(
      clamp(floor(aAnchor.z.sub(2.0).add(0.5)), float(0.0), float(WAKE_SLOT_COUNT - 1))
    );
    const wakeSlot = vec4(u.uWakeSlots.element(wakeSlotIdx));
    const wakeAge = time.sub(wakeSlot.z);
    const wakeStr = clamp(wakeSlot.w, float(0.0), float(2.5));
    const wakeActive = smoothstep(float(0.005), float(0.06), wakeAge)
      .mul(float(1.0).sub(smoothstep(float(2.4), float(4.4), wakeAge)))
      .mul(wakeStr);
    const wakeR = max(float(0.25), wakeAge.mul(3.1).add(aAnchor.y));
    const wCos = cos(aAnchor.x);
    const wSin = sin(aAnchor.x);
    const wakeRestXZ = vec2(
      wakeSlot.x.add(wCos.mul(wakeR)),
      wakeSlot.y.add(wSin.mul(wakeR))
    );
    const wakeDirXZ = vec2(wCos, wSin);
    const wakeTanXZ = vec2(wSin.negate(), wCos);

    const isVortexSub = aAnchor.z.lessThan(float(0.5));
    const isRogueSub = aAnchor.z
      .greaterThanEqual(float(0.5))
      .and(aAnchor.z.lessThan(float(1.5)));

    const interRestXZ = select(
      isVortexSub,
      vortexRestXZ,
      select(isRogueSub, rogueRestXZ, wakeRestXZ)
    );
    const interDirXZ = select(
      isVortexSub,
      vortexDirXZ,
      select(isRogueSub, vec2(ROGUE_DIR_X, ROGUE_DIR_Z), wakeDirXZ)
    );
    const interTanXZ = select(
      isVortexSub,
      vortexTanXZ,
      select(isRogueSub, vec2(ROGUE_TAN_X, ROGUE_TAN_Z), wakeTanXZ)
    );
    const interActive = select(
      isVortexSub,
      vortexActive,
      select(isRogueSub, rogueActive, smoothstep(float(0.04), float(0.32), wakeActive))
    );
    const interCurlAmt = select(
      isVortexSub,
      vortexActive.mul(vRingProx).mul(0.75),
      select(isRogueSub, rogueLife.mul(rogueCurl).mul(rogueEnvT), wakeActive.mul(0.28))
    );

    const restXZ = select(
      isSolitonPool,
      solTrack.restXZ,
      select(isStratifiedPool, aAnchor.xy, interRestXZ)
    );

    // SINGLE unified wave-field evaluation per vertex (exact lock with oceanSurface.ts!)
    const wave = evaluateWaveTSL(restXZ, u);
    const waveDisp = vec3(wave.displacement);
    const waveNorm = normalize(vec3(wave.normal));
    const crestEnergy = float(wave.crestEnergy);
    const curlFactor = float(wave.curlFactor);
    const wakeEnergy = float(wave.wakeEnergy);
    const orbitalPhase = float(wave.orbitalPhase);

    // --- Pool-specific emergence gate ---
    // Keep foreground-center golden sun-glitter lane (|x| <= 3.2) 100% open unless wake-energized
    const solEmerge = smoothstep(float(0.08), float(0.34), float(solTrack.riseEnv))
      .mul(smoothstep(float(0.08), float(0.32), float(solTrack.envT)))
      .mul(mix(float(0.55), float(1.0), solToeTaper));

    const outsideSunLane = smoothstep(float(3.15), float(4.25), abs(restXZ.x));
    const wakeOverride = smoothstep(float(0.25), float(0.55), wakeEnergy);

    const stratCrestOnly = max(
      smoothstep(float(0.42), float(0.76), crestEnergy),
      smoothstep(float(0.34), float(0.66), curlFactor)
    ).mul(outsideSunLane);
    const stratEmerge = clamp(max(stratCrestOnly, wakeOverride), float(0.0), float(1.0)).mul(
      aAnchor.z
    );

    const emergence = select(
      isSolitonPool,
      solEmerge,
      select(isStratifiedPool, stratEmerge, interActive)
    );

    // --- Tightened multi-layer depth (<= 0.26m) + sculptural curling-lip arch (<= 0.68m) ---
    const layerDepth = aLayer.x;
    const lipArch = aLayer.y;
    const sizeClass = aLayer.z;
    const aspect = aLayer.w;

    const activeDirXZ = select(
      isSolitonPool,
      solTrack.dirXZ,
      select(isStratifiedPool, vec2(0.0, 1.0), interDirXZ)
    );
    const activeTanXZ = select(
      isSolitonPool,
      solTrack.tanXZ,
      select(isStratifiedPool, vec2(1.0, 0.0), interTanXZ)
    );

    const solCurlStrength = float(solTrack.curlLife)
      .mul(float(solTrack.riseEnv))
      .mul(float(solTrack.envT))
      .mul(breakerHeight)
      .mul(solLipZone);

    const curlStrength = select(
      isSolitonPool,
      solCurlStrength,
      select(isStratifiedPool, float(0.0), interCurlAmt)
    );

    // Sculptural curling-lip hook arching forward & slightly downward like a real breaking barrel
    const archPhase = lipArch.mul(clamp(curlStrength.mul(1.24), float(0.0), float(1.30)));
    const lipForward = sin(archPhase.mul(1.92))
      .mul(float(0.76).add(aSeed.x.mul(0.22)))
      .mul(curlStrength);
    const lipLift = sin(archPhase.mul(2.35))
      .mul(0.56)
      .sub(archPhase.mul(archPhase).mul(0.38))
      .mul(curlStrength);

    // Dense multi-layer wave body thickness tapering flush to the ocean at the lower trough toe
    const thickFactor = clamp(
      curlFactor.mul(0.68).add(crestEnergy.mul(0.36)).add(wakeEnergy.mul(0.32)),
      float(0.0),
      float(1.0)
    );
    const stackTaper = select(isSolitonPool, solToeTaper, float(1.0));
    const normalStack = mix(
      float(-0.04),
      float(0.02).add(
        layerDepth
          .mul(0.24)
          .mul(float(0.35).add(thickFactor.mul(0.65)))
          .mul(mix(float(0.18), float(1.0), stackTaper))
      ),
      emergence
    );

    const bodyJitter = aSeed.z.sub(0.5).mul(0.18).mul(thickFactor);

    const instanceCenter = vec3(
      restXZ.x
        .add(waveDisp.x)
        .add(waveNorm.x.mul(normalStack))
        .add(activeDirXZ.x.mul(lipForward))
        .add(activeTanXZ.x.mul(bodyJitter)),
      waveDisp.y.add(waveNorm.y.mul(normalStack)).add(lipLift),
      restXZ.y
        .add(waveDisp.z)
        .add(waveNorm.z.mul(normalStack))
        .add(activeDirXZ.y.mul(lipForward))
        .add(activeTanXZ.y.mul(bodyJitter))
    );

    // --- Tightened physical scale graduation: 0.22..0.52m ---
    const lipShrink = float(1.0).sub(lipArch.mul(curlStrength).mul(0.34));
    const midSolitonBoost = select(
      isSolitonPool.and(aAnchor.z.greaterThan(float(0.5))),
      float(1.10),
      float(1.0)
    );
    const baseDiam = mix(float(0.22), float(0.52), sizeClass)
      .mul(lipShrink)
      .mul(midSolitonBoost)
      .mul(mix(float(0.76), float(1.04), emergence));

    // Zero scale when inactive collapses triangles to zero-area (zero fragment cost)
    const activeScale = baseDiam.mul(smoothstep(float(0.02), float(0.22), emergence));
    const sx = activeScale.mul(aspect);
    const sy = activeScale.div(sqrt(max(aspect, float(0.5))));
    const depthRatio = float(0.78).add(aSeed.w.mul(0.26));
    const sz = activeScale.mul(depthRatio);

    // --- Cohesive typographic wave-wall shingling -> lively tumbling strictly at breaking lip ---
    // Keep front wave-wall C's predominantly tangent/outward-facing so every glyph reads crisply
    // as a beveled capital 'C' (matching fused_macro_sea_of_c), tumbling freely only on breaking lips.
    const tumbleFreedom = clamp(
      smoothstep(float(0.28), float(0.78), lipArch)
        .mul(0.82)
        .add(smoothstep(float(0.68), float(0.98), curlFactor).mul(0.22))
        .add(wakeEnergy.mul(0.48)),
      float(0.06),
      float(0.95)
    );
    const energyTumble = clamp(
      curlFactor.mul(0.85).add(crestEnergy.mul(0.22)).add(wakeEnergy.mul(0.55)),
      float(0.0),
      float(1.10)
    ).mul(tumbleFreedom);
    const baseYaw = aSeed.z
      .sub(0.5)
      .mul(mix(float(0.26), float(2.15), energyTumble))
      .add(orbitalPhase.mul(0.14).mul(energyTumble));

    const axAz = aSeed.x.mul(6.2831853);
    const axEl = aSeed.y.sub(0.5).mul(1.65);
    const tumbleAxis = normalize(
      vec3(cos(axAz).mul(cos(axEl)), sin(axEl), sin(axAz).mul(cos(axEl)))
    );
    const tumbleAngle = aSeed.w
      .sub(0.5)
      .mul(mix(float(0.16), float(0.46), energyTumble))
      .add(
        orbitalPhase
          .mul(float(0.42).add(aSeed.x.mul(0.36)))
          .add(aSeed.y.mul(6.2831853))
          .mul(energyTumble.mul(0.42))
      );

    const pScaled = vec3(
      positionGeometry.x.mul(sx),
      positionGeometry.y.mul(sy),
      positionGeometry.z.mul(sz)
    );
    const nUnscaled = normalize(
      vec3(
        normalGeometry.x.div(max(aspect, float(0.5))),
        normalGeometry.y.mul(sqrt(max(aspect, float(0.5)))),
        normalGeometry.z.div(depthRatio)
      )
    );

    const cYaw = cos(baseYaw);
    const sYaw = sin(baseYaw);
    const pYaw = vec3(
      pScaled.x.mul(cYaw).sub(pScaled.y.mul(sYaw)),
      pScaled.x.mul(sYaw).add(pScaled.y.mul(cYaw)),
      pScaled.z
    );
    const nYaw = vec3(
      nUnscaled.x.mul(cYaw).sub(nUnscaled.y.mul(sYaw)),
      nUnscaled.x.mul(sYaw).add(nUnscaled.y.mul(cYaw)),
      nUnscaled.z
    );

    const pTumbled = rotateAxisAngle(pYaw, tumbleAxis, tumbleAngle);
    const nTumbled = rotateAxisAngle(nYaw, tumbleAxis, tumbleAngle);

    // Lay flat onto horizontal ocean plane (+Z face -> +Y up, +Y top -> -Z horizon, +X gap -> +X)
    const pFlat = vec3(pTumbled.x, pTumbled.z, pTumbled.y.negate());
    const nFlat = vec3(nTumbled.x, nTumbled.z, nTumbled.y.negate());

    // Align +Y onto live wave surface normal N_wave via minimal-rotation SO(3)
    const pWave = rotateUpToNormal(pFlat, waveNorm);
    const nWave = rotateUpToNormal(nFlat, waveNorm);

    // Pitch overhanging lip C's gracefully forward-downward around the crest ridge tangent
    const lipPitch = lipArch.mul(curlStrength).mul(-1.12);
    const ridgeAxis = normalize(vec3(activeTanXZ.x, float(0.0), activeTanXZ.y));
    const pCurled = rotateAxisAngle(pWave, ridgeAxis, lipPitch);
    const nCurled = normalize(rotateAxisAngle(nWave, ridgeAxis, lipPitch));

    const finalPos = pCurled.add(instanceCenter);
    glassMat.positionNode = finalPos;

    // --- Compact Vertex-to-Fragment Varyings (wave math evaluated strictly ONCE in VS) ---
    // Backlit belly signal strictly gated to thin inner curling belly (u > -0.05) on solitons
    const solBellyGlow = solBellyZone
      .mul(float(solTrack.riseEnv))
      .mul(float(0.55).add(float(solTrack.curlLife).mul(0.50)));
    const stratBellyGlow = smoothstep(float(0.52), float(0.92), curlFactor).mul(0.58);
    const interBellyGlow = clamp(
      smoothstep(float(0.28), float(0.76), curlFactor)
        .mul(0.82)
        .add(smoothstep(float(0.32), float(0.85), wakeEnergy).mul(0.48)),
      float(0.0),
      float(1.0)
    );
    const bellyGlowRaw = select(
      isSolitonPool,
      solBellyGlow,
      select(isStratifiedPool, stratBellyGlow, interBellyGlow)
    );
    const bellyGlow = clamp(bellyGlowRaw, float(0.0), float(1.0));

    // Narrow lip frost strictly at outermost curling lip tip
    const lipFraction = clamp(
      smoothstep(float(0.32), float(0.88), lipArch)
        .mul(clamp(curlStrength.mul(1.25), float(0.0), float(1.0)))
        .add(smoothstep(float(0.68), float(0.98), curlFactor).mul(0.30))
        .add(smoothstep(float(0.60), float(1.15), wakeEnergy).mul(0.26)),
      float(0.0),
      float(1.0)
    );

    const vRotNormal = nCurled.toVarying('vGlassRotNormal');
    const vWorldPos = finalPos.toVarying('vGlassWorldPos');
    const vWaveInfo = vec4(crestEnergy, curlFactor, wakeEnergy, bellyGlow).toVarying(
      'vGlassWaveInfo'
    );
    const vOpticalMeta = vec4(
      lipFraction,
      layerDepth,
      aSeed.y, // per-glyph subtle tint & roughness diversity
      clamp(waveDisp.y.div(2.8), float(0.0), float(1.0))
    ).toVarying('vGlassOpticalMeta');

    glassMat.normalNode = transformNormalToView(normalize(vRotNormal));

    // --- Fragment Stage: Deep Glossy Prussian Sapphire + Saturated Emerald/Amber Sea-Glass ---
    const waterDeep = u.uWaterDeep.rgb;
    const waterShallow = u.uWaterShallow.rgb;
    const glintColor = u.uGlintColor.rgb;
    const foamColor = u.uFoamColor.rgb;
    const sunDir = normalize(vec3(u.uSunDir));
    const biolumGain = float(u.uBiolumGain);
    const fogDensity = float(u.uFogDensity);

    const fCrest = vWaveInfo.x;
    const fCurl = vWaveInfo.y;
    const fWake = vWaveInfo.z;
    const fBelly = vWaveInfo.w;
    const fLip = vOpticalMeta.x;
    const fDepth = vOpticalMeta.y;
    const fRand = vOpticalMeta.z;

    const N = normalize(vRotNormal);
    const toCam = cameraPosition.sub(vWorldPos);
    const camDist = max(length(toCam), float(0.01));
    const V = toCam.div(camDist);

    const NdotV = max(dot(N, V), float(0.0));
    // Sharp chamfer bevel rim highlight (high exponent keeps flat faces rich dark sapphire)
    const bevelRim = pow(max(float(1.0).sub(NdotV), float(1e-4)), float(4.2));
    const NdotL = max(dot(N, sunDir), float(0.0));

    // Directional sun-through-wave forward scatter (peaks along inner curl facing sunset)
    const backDot = max(dot(V.negate(), sunDir), float(1e-4));
    const backScatter = pow(backDot, float(2.8));
    const warmCoreScatter = pow(backDot, float(4.6));
    const sunWrap = pow(
      clamp(dot(N.negate(), sunDir).mul(0.55).add(0.45), float(1e-4), float(1.0)),
      float(2.0)
    );

    // Distance atmospheric haze matching oceanSurface.ts (0% inside 12m hero zone)
    const hazeDist = max(camDist.sub(12.0), float(0.0));
    const distFog = clamp(
      float(1.0).sub(exp(hazeDist.negate().mul(max(fogDensity, float(0.001))).mul(1.38))),
      float(0.0),
      float(0.80)
    );
    const hazeCol = mix(u.uSkyHorizon.rgb, u.uSkyZenith.rgb, float(0.42));

    // Deep crevice AO across stacked multi-layer C's: inner layers sit in deep sapphire shadow
    // so individual 3D letter silhouettes stay razor crisp in macro close-ups!
    const creviceAO = mix(float(0.22), float(1.04), fDepth);

    // 1. Stratified True Sea-Glass Albedo:
    // Lower/outer/flank wave face -> rich dark royal Prussian-sapphire glass
    // Tight backlit wave throat -> deep jewel emerald-teal glass (zero pastel mint washout!)
    // Outermost breaking lip -> crisp seafoam aqua edge
    const bellyGate = smoothstep(float(0.30), float(0.82), fBelly);
    const royalSapphire = mix(
      mix(vec3(0.010, 0.038, 0.145), vec3(0.003, 0.012, 0.048), biolumGain),
      waterDeep.mul(2.8),
      float(0.24)
    );
    const sapphireBase = royalSapphire
      .mul(float(0.82).add(fRand.mul(0.34)))
      .mul(creviceAO);

    const jewelEmerald = mix(
      vec3(0.002, 0.068, 0.078),
      vec3(0.002, 0.075, 0.16),
      biolumGain
    );
    const deepEmeraldGlass = jewelEmerald
      .mul(float(0.78).add(fRand.mul(0.34)))
      .mul(creviceAO);

    const bodyGlass = mix(sapphireBase, deepEmeraldGlass, bellyGate.mul(0.92));

    const frostedLipCol = mix(deepEmeraldGlass, foamColor.mul(0.34), float(0.42)).mul(
      mix(float(0.48), float(1.0), fDepth)
    );
    const lipMix = smoothstep(float(0.56), float(0.96), fLip).mul(0.68);
    glassMat.colorNode = mix(mix(bodyGlass, frostedLipCol, lipMix), hazeCol, distFog.mul(0.52));

    // 2. Crisp optical sea-glass roughness (0.14..0.22); mildly frosted strictly on outer lip
    glassMat.roughnessNode = clamp(
      mix(float(0.14), float(0.22), fRand)
        .add(lipMix.mul(0.12))
        .sub(bevelRim.mul(0.03)),
      float(0.10),
      float(0.34)
    );

    // 3. True Sea-Glass Internal Volume + Concentrated Backlit Jewel Core
    const daySunGate = float(1.0).sub(biolumGain.mul(0.84));

    // Deep-sapphire internal body depth so shadowed sapphire C's read as luminous dark blue glass
    const sapphireInternalTint = mix(
      vec3(0.007, 0.028, 0.11),
      vec3(0.002, 0.010, 0.042),
      biolumGain
    );
    const sapphireInternal = sapphireInternalTint
      .mul(float(0.20).add(NdotV.mul(0.44)))
      .mul(float(1.0).sub(bellyGate.mul(0.85)))
      .mul(creviceAO);

    // Concentrated saturated jewel emerald-turquoise backlit core inside thin curling throat
    const emeraldCoreTint = mix(waterShallow.mul(0.65), vec3(0.003, 0.38, 0.36), float(0.60));
    const sssTurquoise = emeraldCoreTint
      .mul(pow(max(bellyGate, float(1e-4)), float(1.55)))
      .mul(float(0.12).add(sunWrap.mul(0.32)).add(backScatter.mul(0.30)))
      .mul(mix(float(0.28), float(1.08), fDepth))
      .mul(daySunGate);

    // Molten golden-amber sunset shining straight through sun-facing thin curl (Macro ref!)
    const deepAmberTint = mix(glintColor, vec3(1.0, 0.44, 0.03), float(0.56));
    const warmTransmission = deepAmberTint
      .mul(warmCoreScatter)
      .mul(bellyGate)
      .mul(float(0.32).add(bevelRim.mul(0.55)))
      .mul(daySunGate);

    // Crisp translucent bevel rim catchlights separating stacked individual glass C's
    const goldenBevelCatch = glintColor
      .mul(bevelRim)
      .mul(float(0.05).add(NdotL.mul(0.22)).add(warmCoreScatter.mul(0.18)))
      .mul(daySunGate);

    const coolBevelCatch = mix(waterShallow, foamColor, float(0.28))
      .mul(bevelRim)
      .mul(float(0.03).add(bellyGate.mul(0.20)).add(lipMix.mul(0.22)))
      .mul(daySunGate);

    // Restrained frosted sea-glass rim strictly on outer breaking lip tips
    const lipFrostLift = foamColor
      .mul(lipMix)
      .mul(float(0.08).add(bevelRim.mul(0.22)))
      .mul(daySunGate);

    // Vivid electric-cyan dinoflagellate bioluminescence on crests, lips & wakes in Biolum mode!
    const biolumExcite = clamp(
      fWake
        .mul(1.55)
        .add(smoothstep(float(0.28), float(0.82), fCurl).mul(0.95))
        .add(bellyGate.mul(0.95))
        .add(lipMix.mul(0.85))
        .add(smoothstep(float(0.50), float(0.92), fCrest).mul(0.35)),
      float(0.0),
      float(1.45)
    );
    const biolumEmit = vec3(0.02, 1.05, 1.38)
      .mul(biolumExcite)
      .mul(biolumGain)
      .mul(float(0.44).add(bevelRim.mul(0.82)))
      .mul(mix(float(0.35), float(1.18), fDepth));

    const rawEmissive = sapphireInternal
      .add(sssTurquoise)
      .add(warmTransmission)
      .add(goldenBevelCatch)
      .add(coolBevelCatch)
      .add(lipFrostLift)
      .add(biolumEmit);

    // Chroma-preserving cap (<= 0.58 linear in golden/storm for jewel depth, <= 1.32 in biolum)
    const emissiveCap = mix(float(0.58), float(1.32), biolumGain);
    glassMat.emissiveNode = softCapChroma(rawEmissive, emissiveCap).mul(
      float(1.0).sub(distFog.mul(0.68))
    );
  }

  const glassMesh = new THREE.InstancedMesh(glassGeom, glassMat, GLASS_TOTAL_COUNT);
  glassMesh.name = 'SeaOfC_BeveledGlassFluid';
  initIdentityInstances(glassMesh, GLASS_TOTAL_COUNT);
  group.add(glassMesh);

  // ===========================================================================
  // LAYER 2: AIRBORNE BREAKING-LIP & WAKE 'c' SPRAY (`sprayMesh`, 2,400 instances)
  // ===========================================================================
  const sprayGeom = createSprayCGeometry();
  buildSprayAttributes(sprayGeom);

  const sprayMat = new THREE.MeshPhysicalNodeMaterial({
    envMap,
    envMapIntensity: 0.65,
    metalness: 0.02,
    roughness: 0.22,
    clearcoat: 0.85,
    clearcoatRoughness: 0.12,
    ior: 1.44,
    fog: false,
    depthWrite: true,
    depthTest: true,
  });

  {
    const aAnchor = attribute('aAnchor', 'vec4');
    const aSeed = attribute('aSeed', 'vec4');
    const aLayer = attribute('aLayer', 'vec4');
    const time = float(u.uTime);
    const breakerHeight = clamp(float(u.uBreakerHeight), float(0.3), float(1.85));

    const isSolitonSpray = aAnchor.w.lessThan(float(0.5));

    // Periodic GPU ballistic flight phase in [0, 1) (negative-safe fract)
    const rate = aSeed.x;
    const phase = safeFract1(time.mul(rate).add(aSeed.y));
    const flightDuration = clamp(float(1.0).div(max(rate, float(0.25))), float(0.80), float(1.70));
    const age = phase.mul(flightDuration);

    // --- Pool 0: Hero Soliton Breaking-Lip origin ---
    const solTrack = buildSolitonTrackerTSL(aAnchor.x, aAnchor.y, aAnchor.z, time);
    const solSprayActivity = smoothstep(float(0.20), float(0.56), float(solTrack.curlLife))
      .mul(smoothstep(float(0.14), float(0.42), float(solTrack.riseEnv)))
      .mul(smoothstep(float(0.12), float(0.38), float(solTrack.envT)));

    // --- Pool 1A: Vortex Spray origin ---
    const vortex = vec4(u.uVortex);
    const vStrength = clamp(vortex.z, float(0.0), float(2.0));
    const vTheta = aAnchor.y.add(vortex.w.mul(1.15));
    const vCos = cos(vTheta);
    const vSin = sin(vTheta);
    const vortexOriginXZ = vec2(
      vortex.x.add(vCos.mul(aAnchor.x)),
      vortex.y.add(vSin.mul(aAnchor.x))
    );
    const vortexThrowXZ = normalize(
      vec2(
        vSin.negate().mul(0.75).add(vCos.mul(0.66)),
        vCos.mul(0.75).add(vSin.mul(0.66))
      )
    );
    const vortexSprayAct = smoothstep(float(0.05), float(0.36), vStrength);

    // --- Pool 1B: Rogue Wave Breaking-Lip Spray origin ---
    const rogue = vec4(u.uRogueWave);
    const rogueAge = time.sub(rogue.z);
    const rogueAmp = clamp(rogue.w, float(0.0), float(2.5));
    const rogueLife = smoothstep(float(0.04), float(0.42), rogueAge)
      .mul(float(1.0).sub(smoothstep(float(2.6), float(4.9), rogueAge)))
      .mul(rogueAmp);
    const rogueCurl = smoothstep(float(0.16), float(0.65), rogueAge).mul(
      float(1.0).sub(smoothstep(float(2.5), float(4.6), rogueAge))
    );
    const rogueAgePos = max(float(0.0), rogueAge);
    const rogueCx = rogue.x.add(rogueAgePos.mul(ROGUE_DIR_X * ROGUE_SPEED));
    const rogueCz = rogue.y.add(rogueAgePos.mul(ROGUE_DIR_Z * ROGUE_SPEED));
    const rogueTDist = aAnchor.y.mul(ROGUE_SIGMA_T);
    const rogueNRaw = aAnchor.x
      .mul(ROGUE_SIGMA_N)
      .add(rogueTDist.mul(rogueTDist).mul(ROGUE_BOW));
    const rogueOriginXZ = vec2(
      rogueCx.add(rogueNRaw.mul(ROGUE_DIR_X)).add(rogueTDist.mul(ROGUE_TAN_X)),
      rogueCz.add(rogueNRaw.mul(ROGUE_DIR_Z)).add(rogueTDist.mul(ROGUE_TAN_Z))
    );
    const rogueEnvT = exp(aAnchor.y.mul(aAnchor.y).negate());
    const rogueSprayAct = smoothstep(
      float(0.06),
      float(0.36),
      rogueLife.mul(rogueCurl).mul(rogueEnvT)
    );

    // --- Pool 1C: 8-Slot Wake Spray origin ---
    const wakeSlotIdx = uint(
      clamp(floor(aAnchor.z.sub(2.0).add(0.5)), float(0.0), float(WAKE_SLOT_COUNT - 1))
    );
    const wakeSlot = vec4(u.uWakeSlots.element(wakeSlotIdx));
    const wakeAge = time.sub(wakeSlot.z);
    const wakeStr = clamp(wakeSlot.w, float(0.0), float(2.5));
    const wakeActRaw = smoothstep(float(0.008), float(0.06), wakeAge)
      .mul(float(1.0).sub(smoothstep(float(1.8), float(3.6), wakeAge)))
      .mul(wakeStr);
    const wakeR = max(float(0.18), wakeAge.mul(3.1).add(aAnchor.y));
    const wCos = cos(aAnchor.x);
    const wSin = sin(aAnchor.x);
    const wakeOriginXZ = vec2(
      wakeSlot.x.add(wCos.mul(wakeR)),
      wakeSlot.y.add(wSin.mul(wakeR))
    );
    const wakeSprayAct = smoothstep(float(0.06), float(0.35), wakeActRaw);

    const isVortexSub = aAnchor.z.lessThan(float(0.5));
    const isRogueSub = aAnchor.z
      .greaterThanEqual(float(0.5))
      .and(aAnchor.z.lessThan(float(1.5)));

    const interOriginXZ = select(
      isVortexSub,
      vortexOriginXZ,
      select(isRogueSub, rogueOriginXZ, wakeOriginXZ)
    );
    const interThrowXZ = select(
      isVortexSub,
      vortexThrowXZ,
      select(isRogueSub, vec2(ROGUE_DIR_X, ROGUE_DIR_Z), vec2(wCos, wSin))
    );
    const interTanXZ = select(
      isVortexSub,
      vec2(vortexThrowXZ.y.negate(), vortexThrowXZ.x),
      select(isRogueSub, vec2(ROGUE_TAN_X, ROGUE_TAN_Z), vec2(wSin.negate(), wCos))
    );
    const interSprayAct = select(
      isVortexSub,
      vortexSprayAct,
      select(isRogueSub, rogueSprayAct, wakeSprayAct)
    );

    const originXZ = select(isSolitonSpray, solTrack.restXZ, interOriginXZ);
    const throwDirXZ = select(isSolitonSpray, solTrack.dirXZ, interThrowXZ);
    const spreadTanXZ = select(isSolitonSpray, solTrack.tanXZ, interTanXZ);

    // Evaluate unified wave field ONCE at launch origin
    const wave = evaluateWaveTSL(originXZ, u);
    const waveDisp = vec3(wave.displacement);
    const waveNorm = normalize(vec3(wave.normal));

    // Align launch anchor cleanly onto the tightened curling lip tip (<= 0.68m overhang)
    const lipArchStart = aLayer.w;
    const solCurlGate = float(solTrack.curlLife)
      .mul(float(solTrack.riseEnv))
      .mul(float(solTrack.envT))
      .mul(breakerHeight);
    const lipOffsetScale = select(
      isSolitonSpray,
      solCurlGate,
      select(isRogueSub, rogueLife.mul(rogueCurl).mul(rogueEnvT), float(0.0))
    );
    const launchFwd = sin(lipArchStart.mul(2.05)).mul(0.66).mul(lipOffsetScale);
    const launchLift = sin(lipArchStart.mul(2.45))
      .mul(0.40)
      .sub(lipArchStart.mul(lipArchStart).mul(0.22))
      .mul(lipOffsetScale);

    const launchPos = vec3(
      originXZ.x.add(waveDisp.x).add(waveNorm.x.mul(0.10)).add(throwDirXZ.x.mul(launchFwd)),
      waveDisp.y.add(waveNorm.y.mul(0.10)).add(launchLift),
      originXZ.y.add(waveDisp.z).add(waveNorm.z.mul(0.10)).add(throwDirXZ.y.mul(launchFwd))
    );

    // Ballistic initial launch velocity (upward + forward curl throw + crisp fan spread)
    const energyGain = mix(
      float(0.84),
      float(1.30),
      clamp(breakerHeight.sub(0.5), float(0.0), float(1.0))
    );
    const vy0 = float(2.05).mul(aLayer.x).mul(energyGain);
    const vFwd = float(1.85).mul(aLayer.y).mul(energyGain);
    const vLat = aLayer.z.mul(0.74);

    const v0 = vec3(
      throwDirXZ.x.mul(vFwd).add(spreadTanXZ.x.mul(vLat)),
      vy0,
      throwDirXZ.y.mul(vFwd).add(spreadTanXZ.y.mul(vLat))
    );

    // True ballistic gravity trajectory: p = p0 + v0 * age - 0.5 * g * age^2
    const gravity = float(8.6);
    const sprayCenter = vec3(
      launchPos.x.add(v0.x.mul(age)),
      launchPos.y.add(v0.y.mul(age)).sub(age.mul(age).mul(gravity.mul(0.5))),
      launchPos.z.add(v0.z.mul(age))
    );

    // Strict activity & sub-surface gating: zero ghost dots over calm troughs or underwater!
    const solWaveGate = smoothstep(float(0.24), float(0.58), float(wave.curlFactor)).mul(
      smoothstep(float(0.24), float(0.60), float(wave.crestEnergy))
    );
    const activity = select(isSolitonSpray, solSprayActivity.mul(solWaveGate), interSprayAct);

    const fadeIn = smoothstep(float(0.0), float(0.10), phase);
    const fadeOut = float(1.0).sub(smoothstep(float(0.64), float(0.98), phase));
    const aboveWater = smoothstep(float(-0.08), float(0.18), sprayCenter.y.sub(waveDisp.y));

    const arcTaper = mix(float(1.08), float(0.64), phase);
    const midSprayBoost = select(
      isSolitonSpray.and(aAnchor.z.greaterThan(float(0.5))),
      float(1.12),
      float(1.0)
    );
    const baseSprayDiam = mix(float(0.068), float(0.175), aSeed.z)
      .mul(arcTaper)
      .mul(midSprayBoost);
    const finalSprayScale = baseSprayDiam.mul(activity).mul(fadeIn).mul(fadeOut).mul(aboveWater);

    // Brisk mid-air 3D tumbling spin applied identically to positionGeometry and normalGeometry
    const axAz = aSeed.w.mul(6.2831853);
    const axEl = aLayer.z.mul(1.1);
    const spinAxis = normalize(
      vec3(cos(axAz).mul(cos(axEl)), sin(axEl), sin(axAz).mul(cos(axEl)))
    );
    const spinSpeed = float(3.8).add(aSeed.z.mul(4.8));
    const spinAngle = aSeed.y.mul(6.2831853).add(time.mul(spinSpeed));

    const pLocal = vec3(
      positionGeometry.x.mul(finalSprayScale),
      positionGeometry.y.mul(finalSprayScale),
      positionGeometry.z.mul(finalSprayScale.mul(1.34))
    );
    const nLocal = normalize(
      vec3(normalGeometry.x, normalGeometry.y, normalGeometry.z.div(1.34))
    );

    const pRot = rotateAxisAngle(pLocal, spinAxis, spinAngle);
    const nRot = normalize(rotateAxisAngle(nLocal, spinAxis, spinAngle));

    const finalSprayPos = pRot.add(sprayCenter);
    sprayMat.positionNode = finalSprayPos;

    const vSprayNorm = nRot.toVarying('vSprayRotNormal');
    const vSprayWorldPos = finalSprayPos.toVarying('vSprayWorldPos');
    const vSprayMeta = vec4(phase, aSeed.z, float(wave.wakeEnergy), activity).toVarying(
      'vSprayMeta'
    );

    sprayMat.normalNode = transformNormalToView(normalize(vSprayNorm));

    // --- Fragment Stage: Crisp Sculpted Frosted 'c' Spray (calibrated ~0.35x emissive) ---
    const sunDir = normalize(vec3(u.uSunDir));
    const foamColor = u.uFoamColor.rgb;
    const waterShallow = u.uWaterShallow.rgb;
    const glintColor = u.uGlintColor.rgb;
    const biolumGain = float(u.uBiolumGain);
    const fogDensity = float(u.uFogDensity);

    const N = normalize(vSprayNorm);
    const toCam = cameraPosition.sub(vSprayWorldPos);
    const camDist = max(length(toCam), float(0.01));
    const V = toCam.div(camDist);

    const NdotV = max(dot(N, V), float(0.0));
    const fresnelRim = pow(max(float(1.0).sub(NdotV), float(1e-4)), float(3.0));
    const backLight = pow(max(dot(V.negate(), sunDir), float(1e-4)), float(2.8));
    const NdotL = max(dot(N, sunDir), float(0.0));

    const hazeDist = max(camDist.sub(12.0), float(0.0));
    const distFog = clamp(
      float(1.0).sub(exp(hazeDist.negate().mul(max(fogDensity, float(0.001))).mul(1.38))),
      float(0.0),
      float(0.78)
    );
    const hazeCol = mix(u.uSkyHorizon.rgb, u.uSkyZenith.rgb, float(0.42));

    // Sculpted pale-aqua to seafoam ivory albedo (moderate value so shaded sides show crisp 3D form)
    const sprayBaseCol = mix(
      mix(waterShallow.mul(0.55), foamColor.mul(0.68), float(0.64)),
      foamColor.mul(0.76),
      clamp(vSprayMeta.x.mul(1.15).add(0.18), float(0.0), float(1.0))
    );
    sprayMat.colorNode = mix(sprayBaseCol, hazeCol, distFog.mul(0.50));

    const warmGate = float(1.0).sub(biolumGain.mul(0.84));
    const goldenBackHalo = glintColor
      .mul(backLight)
      .mul(float(0.20).add(fresnelRim.mul(0.32)))
      .mul(warmGate)
      .mul(0.52);

    const crispFoamRim = foamColor
      .mul(float(0.09).add(fresnelRim.mul(0.24)).add(NdotL.mul(0.14)))
      .mul(float(1.0).sub(biolumGain.mul(0.65)));

    const biolumSpark = vec3(0.04, 0.96, 1.25)
      .mul(biolumGain)
      .mul(float(0.42).add(fresnelRim.mul(0.55)).add(vSprayMeta.z.mul(0.32)));

    const sprayCap = mix(float(0.64), float(1.18), biolumGain);
    sprayMat.emissiveNode = softCapChroma(
      goldenBackHalo.add(crispFoamRim).add(biolumSpark),
      sprayCap
    ).mul(float(1.0).sub(distFog.mul(0.64)));
  }

  const sprayMesh = new THREE.InstancedMesh(sprayGeom, sprayMat, SPRAY_TOTAL_COUNT);
  sprayMesh.name = 'SeaOfC_AirborneSprayC';
  initIdentityInstances(sprayMesh, SPRAY_TOTAL_COUNT);
  group.add(sprayMesh);

  return {
    group,
    glassMesh,
    sprayMesh,
    glassCount: GLASS_TOTAL_COUNT,
    sprayCount: SPRAY_TOTAL_COUNT,
    dispose: () => {
      glassGeom.dispose();
      glassMat.dispose();
      sprayGeom.dispose();
      sprayMat.dispose();
    },
  };
}
