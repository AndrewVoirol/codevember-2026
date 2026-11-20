import * as THREE from 'three/webgpu';
import { DOME_MAX_PHI, type StonePolarity } from './physics';

/**
 * Shared Lambert Azimuthal Equal-Area (LAEA) projection constants for the dome cap.
 * Identical constants are consumed by the TSL dome shader so CPU baking, dynamic scuffs,
 * and GPU shading align with zero polar pinch and zero meridian seam.
 */
export const LAEA_PHI_PAD = 0.09;
export const LAEA_K_MAX = Math.sqrt(2 * (1 - Math.cos(DOME_MAX_PHI + LAEA_PHI_PAD)));
export const LAEA_UV_RADIUS = 0.47;
export const LAEA_UV_SCALE = LAEA_UV_RADIUS / LAEA_K_MAX;
export const HAMMERED_SLOPE_DECODE_MAX = 1.15;

export interface HammeredIronSurfaceMaps {
  staticMap: THREE.DataTexture;
  dynamicMap: THREE.DataTexture;
  stampScuff: (
    normal: THREE.Vector3,
    polarity: StonePolarity,
    intensity: number,
    isImpact: boolean
  ) => void;
  stampBobWake: (normal: THREE.Vector3, proximity01: number, speed: number, dtSec: number) => void;
  decayAndUpload: (dtSec: number) => void;
  getActiveScuffEnergy: () => number;
  dispose: () => void;
}

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

function hash3(ix: number, iy: number, iz: number): number {
  const n = Math.sin(ix * 127.1 + iy * 311.7 + iz * 74.7) * 43758.5453123;
  return n - Math.floor(n);
}

interface PeenSample {
  gx: number;
  gy: number;
  gz: number;
  ridge: number;
  cavity: number;
}

/**
 * Single-pass exact analytical 3D Voronoi planished hammer-peen evaluator.
 * Produces shallow, hand-forged concave bowls with softly beveled forged boundaries
 * (no harsh polygonal wireframe grid or sharp golf-ball dimples).
 */
function evalPeenLayer(
  nx: number,
  ny: number,
  nz: number,
  scale: number,
  seedOffset: number,
  out: PeenSample
): void {
  const px = nx * scale + seedOffset;
  const py = ny * scale + seedOffset * 1.37;
  const pz = nz * scale + seedOffset * 0.73;

  const ix = Math.floor(px);
  const iy = Math.floor(py);
  const iz = Math.floor(pz);

  let d1Sq = 1e9;
  let c1x = 0;
  let c1y = 0;
  let c1z = 0;
  let id1 = 0.5;

  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const cz = iz + dz;
        const jx = 0.14 + 0.72 * hash3(cx, cy, cz);
        const jy = 0.14 + 0.72 * hash3(cx + 19, cy + 7, cz + 3);
        const jz = 0.14 + 0.72 * hash3(cx + 5, cy + 23, cz + 11);
        const rx = cx + jx - px;
        const ry = cy + jy - py;
        const rz = cz + jz - pz;
        const dSq = rx * rx + ry * ry + rz * rz;
        if (dSq < d1Sq) {
          d1Sq = dSq;
          c1x = rx;
          c1y = ry;
          c1z = rz;
          id1 = hash3(cx + 41, cy + 13, cz + 29);
        }
      }
    }
  }

  let minEdgeDist = 1e9;
  let ex = 0;
  let ey = 0;
  let ez = 0;

  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const cz = iz + dz;
        const jx = 0.14 + 0.72 * hash3(cx, cy, cz);
        const jy = 0.14 + 0.72 * hash3(cx + 19, cy + 7, cz + 3);
        const jz = 0.14 + 0.72 * hash3(cx + 5, cy + 23, cz + 11);
        const rx = cx + jx - px;
        const ry = cy + jy - py;
        const rz = cz + jz - pz;

        const vx = rx - c1x;
        const vy = ry - c1y;
        const vz = rz - c1z;
        const vLenSq = vx * vx + vy * vy + vz * vz;
        if (vLenSq > 1e-5) {
          const invLen = 1 / Math.sqrt(vLenSq);
          const mx = 0.5 * (c1x + rx);
          const my = 0.5 * (c1y + ry);
          const mz = 0.5 * (c1z + rz);
          const edgeDist = (mx * vx + my * vy + mz * vz) * invLen;
          if (edgeDist < minEdgeDist) {
            minEdgeDist = edgeDist;
            ex = vx * invLen;
            ey = vy * invLen;
            ez = vz * invLen;
          }
        }
      }
    }
  }

  // Organic strike-depth variation across individual hand-hammered planishing blows
  const blowDepth = 0.72 + 0.52 * id1;

  // Softly rolled ridge shoulder so the concave bowl transitions smoothly into adjacent strikes
  // (eliminates crisp bright Voronoi cell webs on bright specular highlights at 1:1 macro zoom)
  const ridgeWidth = 0.28;
  const ridgeFactor = Math.exp(-Math.pow(Math.max(0, minEdgeDist) / ridgeWidth, 2.0));
  const edgeRoll = Math.min(1.0, Math.max(0.0, minEdgeDist / 0.24));
  const smoothRoll = edgeRoll * edgeRoll * (3 - 2 * edgeRoll);

  const bowlScale = 1.20 * blowDepth * (0.34 + 0.66 * smoothRoll);
  const ridgeSlopeWeight = 0.12 * ridgeFactor * (minEdgeDist / Math.max(0.04, ridgeWidth));

  out.gx = -c1x * bowlScale - ex * ridgeSlopeWeight;
  out.gy = -c1y * bowlScale - ey * ridgeSlopeWeight;
  out.gz = -c1z * bowlScale - ez * ridgeSlopeWeight;

  out.ridge = Math.min(1, Math.max(0, ridgeFactor * (0.48 + 0.32 * id1)));
  out.cavity = Math.min(1, Math.max(0, Math.sqrt(Math.min(1, d1Sq)) * 1.05 + ridgeFactor * 0.15));
}

/**
 * Bakes the static 768x768 Lambert Azimuthal Equal-Area hammered-iron surface map
 * and initializes the paired 512x512 dynamic surface-reaction patina map.
 */
export function createHammeredIronSurfaceMaps(): HammeredIronSurfaceMaps {
  const staticSize = 768;
  const staticData = new Uint8Array(staticSize * staticSize * 4);
  const p1: PeenSample = { gx: 0, gy: 0, gz: 0, ridge: 0, cavity: 0 };
  const p2: PeenSample = { gx: 0, gy: 0, gz: 0, ridge: 0, cavity: 0 };

  const invUvScale = 1.0 / LAEA_UV_SCALE;

  for (let py = 0; py < staticSize; py++) {
    const v = (py + 0.5) / staticSize;
    const Z = (v - 0.5) * invUvScale;
    for (let px = 0; px < staticSize; px++) {
      const u = (px + 0.5) / staticSize;
      const X = (u - 0.5) * invUvScale;
      const rSq = X * X + Z * Z;
      const outIdx = (py * staticSize + px) * 4;

      if (rSq > 3.85) {
        staticData[outIdx] = 128;
        staticData[outIdx + 1] = 128;
        staticData[outIdx + 2] = 128;
        staticData[outIdx + 3] = 180;
        continue;
      }

      const ny = 1.0 - 0.5 * rSq;
      const s = Math.sqrt(Math.max(0, 1.0 - 0.25 * rSq));
      const nx = X * s;
      const nz = Z * s;
      const phi = Math.acos(Math.max(-1, Math.min(1, ny)));

      // Refined fine-grain hand-planished hammer facets matching studio reference (~52 facets across dome arc)
      evalPeenLayer(nx, ny, nz, 22.8, 1.7, p1);
      evalPeenLayer(nx, ny, nz, 41.5, 19.3, p2);

      // Smooth transition into a finely spun-lathe circumferential finish near the Bone Calcite base collar
      const rimT = Math.max(0, Math.min(1, (phi - (DOME_MAX_PHI - 0.16)) / 0.15));
      const rimSmooth = rimT * rimT * (3 - 2 * rimT);
      const peenWeight = 1.0 - rimSmooth * 0.85;

      let gx = (p1.gx * 0.78 + p2.gx * 0.24) * peenWeight;
      let gy = (p1.gy * 0.78 + p2.gy * 0.24) * peenWeight;
      let gz = (p1.gz * 0.78 + p2.gz * 0.24) * peenWeight;

      const a = 1.0 / Math.max(0.05, 1.0 + ny);
      const tx = 1.0 - nx * nx * a;
      const ty = -nx;
      const tz = -nx * nz * a;
      const bx = -nx * nz * a;
      const by = -nz;
      const bz = 1.0 - nz * nz * a;

      if (rimSmooth > 0.001) {
        const lathePhase = phi * 340.0;
        const latheSlope =
          (Math.cos(lathePhase) * 0.10 + Math.cos(lathePhase * 2.17 + 1.1) * 0.05) * rimSmooth;
        const horiz = Math.hypot(nx, nz);
        if (horiz > 1e-4) {
          const mx = (nx / horiz) * ny;
          const my = -horiz;
          const mz = (nz / horiz) * ny;
          gx += mx * latheSlope;
          gy += my * latheSlope;
          gz += mz * latheSlope;
        }
      }

      const gT = gx * tx + gy * ty + gz * tz;
      const gB = gx * bx + gy * by + gz * bz;

      const normT = Math.max(-1, Math.min(1, gT / HAMMERED_SLOPE_DECODE_MAX));
      const normB = Math.max(-1, Math.min(1, gB / HAMMERED_SLOPE_DECODE_MAX));

      const ridgeVal = Math.min(
        1,
        (p1.ridge * 0.76 + p2.ridge * 0.24) * peenWeight + rimSmooth * 0.35
      );
      const cavityVal = Math.min(
        1,
        (p1.cavity * 0.8 + p2.cavity * 0.2) * peenWeight + rimSmooth * 0.85
      );

      staticData[outIdx] = Math.round(128 + normT * 127);
      staticData[outIdx + 1] = Math.round(128 + normB * 127);
      staticData[outIdx + 2] = Math.round(ridgeVal * 255);
      staticData[outIdx + 3] = Math.round(cavityVal * 255);
    }
  }

  const staticMap = new THREE.DataTexture(staticData, staticSize, staticSize, THREE.RGBAFormat);
  staticMap.colorSpace = THREE.NoColorSpace;
  staticMap.flipY = false;
  staticMap.wrapS = THREE.ClampToEdgeWrapping;
  staticMap.wrapT = THREE.ClampToEdgeWrapping;
  staticMap.magFilter = THREE.LinearFilter;
  staticMap.minFilter = THREE.LinearMipmapLinearFilter;
  staticMap.generateMipmaps = true;
  staticMap.anisotropy = 4;
  staticMap.needsUpdate = true;

  // High-resolution 512x512 dynamic surface-reaction patina map (Gaussian smooth, zero blocky squares!)
  // R = burnished metallic scuff/skitter polish
  // G = grazing bob inductive/satin wake memory
  // B = subtle cinnabar mineral chalk dust caught in hammered dimple troughs
  // A = subtle verdigris mineral chalk dust caught in hammered dimple troughs
  const dynSize = 512;
  const dynFloat = new Float32Array(dynSize * dynSize * 4);
  const dynData = new Uint8Array(dynSize * dynSize * 4);
  const dynamicMap = new THREE.DataTexture(dynData, dynSize, dynSize, THREE.RGBAFormat);
  dynamicMap.colorSpace = THREE.NoColorSpace;
  dynamicMap.flipY = false;
  dynamicMap.wrapS = THREE.ClampToEdgeWrapping;
  dynamicMap.wrapT = THREE.ClampToEdgeWrapping;
  dynamicMap.magFilter = THREE.LinearFilter;
  dynamicMap.minFilter = THREE.LinearFilter;
  dynamicMap.generateMipmaps = false;
  dynamicMap.needsUpdate = true;

  let dirty = false;
  let activeEnergy = 0;

  function normalToDynTexel(normal: THREE.Vector3): [number, number] {
    const k = Math.sqrt(2.0 / Math.max(0.05, 1.0 + normal.y)) * LAEA_UV_SCALE;
    const u = 0.5 + normal.x * k;
    const v = 0.5 + normal.z * k;
    return [u * dynSize, v * dynSize];
  }

  function stampGaussian(
    cx: number,
    cy: number,
    radiusPx: number,
    addR: number,
    addG: number,
    addB: number,
    addA: number
  ): void {
    const minX = Math.max(0, Math.floor(cx - radiusPx));
    const maxX = Math.min(dynSize - 1, Math.ceil(cx + radiusPx));
    const minY = Math.max(0, Math.floor(cy - radiusPx));
    const maxY = Math.min(dynSize - 1, Math.ceil(cy + radiusPx));
    const invR2 = 1.0 / Math.max(1e-4, radiusPx * radiusPx);

    for (let y = minY; y <= maxY; y++) {
      const dy = y + 0.5 - cy;
      for (let x = minX; x <= maxX; x++) {
        const dx = x + 0.5 - cx;
        const d2 = (dx * dx + dy * dy) * invR2;
        if (d2 >= 1.0) continue;
        // Smooth Gaussian bell tapered to exact 0 at boundary -> zero circular rings or pixel blocks
        const edge = 1.0 - d2;
        const w = Math.exp(-d2 * 3.6) * edge * edge;
        const idx = (y * dynSize + x) * 4;
        if (addR > 0) dynFloat[idx] = Math.min(1.0, dynFloat[idx] + addR * w);
        if (addG > 0) dynFloat[idx + 1] = Math.min(1.0, dynFloat[idx + 1] + addG * w);
        if (addB > 0) dynFloat[idx + 2] = Math.min(1.0, dynFloat[idx + 2] + addB * w);
        if (addA > 0) dynFloat[idx + 3] = Math.min(1.0, dynFloat[idx + 3] + addA * w);
      }
    }
    dirty = true;
  }

  const stampScuff = (
    normal: THREE.Vector3,
    polarity: StonePolarity,
    intensity: number,
    isImpact: boolean
  ): void => {
    const [cx, cy] = normalToDynTexel(normal);
    const radiusPx = isImpact ? 19.0 : 12.5;
    const burnish = isImpact ? intensity * 0.62 : intensity * 0.30;
    const chalk = isImpact ? intensity * 0.34 : intensity * 0.16;
    stampGaussian(
      cx,
      cy,
      radiusPx,
      burnish,
      0,
      polarity === 1 ? chalk : 0,
      polarity === -1 ? chalk : 0
    );
  };

  const stampBobWake = (
    normal: THREE.Vector3,
    proximity01: number,
    speed: number,
    dtSec: number
  ): void => {
    if (proximity01 <= 0.02 || speed < 0.08) return;
    const [cx, cy] = normalToDynTexel(normal);
    const gain = Math.min(0.38, proximity01 * Math.min(1.0, speed * 0.65) * dtSec * 3.4);
    if (gain <= 0.002) return;
    stampGaussian(cx, cy, 24.0, 0, gain, 0, 0);
  };

  const decayAndUpload = (dtSec: number): void => {
    if (!dirty && activeEnergy <= 0.0001) return;

    const scuffDecay = Math.exp(-dtSec / 26.0);
    const wakeDecay = Math.exp(-dtSec / 2.2);
    const chalkDecay = Math.exp(-dtSec / 20.0);

    let sumEnergy = 0;
    const len = dynFloat.length;
    for (let i = 0; i < len; i += 4) {
      const r = dynFloat[i] * scuffDecay;
      const g = dynFloat[i + 1] * wakeDecay;
      const b = dynFloat[i + 2] * chalkDecay;
      const a = dynFloat[i + 3] * chalkDecay;

      dynFloat[i] = r > 0.002 ? r : 0;
      dynFloat[i + 1] = g > 0.002 ? g : 0;
      dynFloat[i + 2] = b > 0.002 ? b : 0;
      dynFloat[i + 3] = a > 0.002 ? a : 0;

      dynData[i] = Math.round(dynFloat[i] * 255);
      dynData[i + 1] = Math.round(dynFloat[i + 1] * 255);
      dynData[i + 2] = Math.round(dynFloat[i + 2] * 255);
      dynData[i + 3] = Math.round(dynFloat[i + 3] * 255);

      sumEnergy += dynFloat[i] + dynFloat[i + 2] + dynFloat[i + 3];
    }

    activeEnergy = sumEnergy;
    dynamicMap.needsUpdate = true;
    dirty = activeEnergy > 0.0001;
  };

  return {
    staticMap,
    dynamicMap,
    stampScuff,
    stampBobWake,
    decayAndUpload,
    getActiveScuffEnergy: () => Number(activeEnergy.toFixed(2)),
    dispose: () => {
      staticMap.dispose();
      dynamicMap.dispose();
    },
  };
}

/**
 * Creates a smooth radial-feathered softbox luminance texture so studio reflection panels
 * roll smoothly across curved hammered iron without hard rectangular card edges.
 */
function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number
): THREE.DataTexture {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = (y + 0.5) / size * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = (x + 0.5) / size * 2 - 1;
      const d = Math.hypot(nx, ny);
      const falloff = Math.max(0, Math.min(1, 1 - d * d));
      const soft = falloff * falloff;
      const idx = (y * size + x) * 4;
      data[idx] = Math.min(255, Math.round(r * soft * 255));
      data[idx + 1] = Math.min(255, Math.round(g * soft * 255));
      data[idx + 2] = Math.min(255, Math.round(b * soft * 255));
      data[idx + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Generates a broadly diffused photographer's studio softbox PMREM environment cubemap
 * calibrated to match the broad silky shoulder sheen & deep front-belly negative fill
 * of hand-forged blackened iron, satin brass, and oxidized bronze.
 */
export function createStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Deep darkroom negative fill -> preserves dramatic dark iron contrast band down center belly
  envScene.background = new THREE.Color(0x090c10);

  const disposables: Array<{ dispose: () => void }> = [];

  const addFeatheredSoftbox = (
    w: number,
    h: number,
    pos: [number, number, number],
    lookAt: [number, number, number],
    r: number,
    g: number,
    b: number,
    gain: number
  ) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const tex = createFeatheredSoftboxTexture(r, g, b);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      color: new THREE.Color().setRGB(gain, gain, gain),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.lookAt(lookAt[0], lookAt[1], lookAt[2]);
    envScene.add(mesh);
    disposables.push(geo, tex, mat);
  };

  // 1. Broad overhead-left warm calcite scrim softbox (wraps broad silky reflection across upper-left iron dome)
  addFeatheredSoftbox(11.5, 9.0, [-5.2, 5.6, 3.8], [0, 0, 0], 1.0, 0.96, 0.90, 1.82);

  // 2. Broad right-side tall studio scrim (illuminates sweeping brass S-rod, calcite bob, and right iron shoulder)
  addFeatheredSoftbox(8.5, 10.5, [6.2, 3.2, 2.8], [0, 0, 0], 0.98, 0.95, 0.89, 1.48);

  // 3. Broad rear-top cool slate grazing rim scrim (carves continuous metallic rim along dome top & bronze arch)
  addFeatheredSoftbox(12.0, 5.2, [-0.8, 5.4, -5.8], [0, 0, 0], 0.90, 0.95, 1.0, 1.45);

  // 4. Overhead zenith soft fill dome card (continuous specular wrap over gimbal horseshoe top)
  addFeatheredSoftbox(9.0, 9.0, [0.0, 7.2, 0.4], [0, 0, 0], 0.96, 0.94, 0.90, 0.92);

  // 5. Rear-right subtle cool fill scrim (keeps rear magnetite crystal facets legible when orbiting)
  addFeatheredSoftbox(8.0, 6.5, [4.6, 1.6, -5.6], [0, 0, 0], 0.88, 0.93, 1.0, 0.72);

  // 6. Subtle warm studio table bounce floor (lifts underside of brass plumb tip & dome base naturally)
  addFeatheredSoftbox(12.0, 12.0, [0, -6.2, 0], [0, 0, 0], 0.95, 0.92, 0.88, 0.28);

  // Generous 0.088 sigma pre-blur produces museum-smooth diffused softbox gradients (zero hard card edges)
  const renderTarget = pmrem.fromScene(envScene, 0.088);

  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}
