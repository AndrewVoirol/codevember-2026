import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  faceDirection,
  float,
  frontFacing,
  length,
  max,
  mix,
  modelWorldMatrix,
  normalLocal,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  select,
  sin,
  smoothstep,
  texture,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';

export interface LanternStudioEnvHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface SilkMaterialUniforms {
  uTime: { value: number };
  uUnravel: { value: number };
  uBreath: { value: number };
  uIgnition: { value: number };
  uEmberPulse: { value: number };
  uPointerWorld: { value: THREE.Vector3 };
  uPointerVel: { value: THREE.Vector3 };
  uFlamePos: { value: THREE.Vector3 };
}

export interface SilkMaterialBundle {
  uniforms: SilkMaterialUniforms;
  ricePaper: THREE.MeshPhysicalNodeMaterial;
  bambooFrame: THREE.MeshStandardNodeMaterial;
  suspensionCord: THREE.MeshStandardNodeMaterial;
  silkWrap: THREE.MeshPhysicalNodeMaterial;
  silkVeil: THREE.MeshStandardNodeMaterial;
  silkLoops: THREE.MeshStandardNodeMaterial;
  origamiButterfly: THREE.MeshStandardNodeMaterial;
  origamiButterflies: THREE.MeshStandardNodeMaterial[];
  emberCore: THREE.MeshBasicNodeMaterial;
  emberMotes: THREE.MeshBasicNodeMaterial;
  dispose: () => void;
}

// ============================================================================
// Procedural Studio Softbox PMREM Environment (7-Softbox IBL)
// Calibrated for dry crinkled handmade kozo paper, satin roasted split bamboo,
// anisotropic raw-silk cocoon skeins, and matte braided hemp cord on #212529.
// ============================================================================

function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number,
  size = 64,
): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = ((y + 0.5) / size) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = ((x + 0.5) / size) * 2 - 1;
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

export function createLanternStudioEnvironment(
  renderer: THREE.WebGPURenderer,
): LanternStudioEnvHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x1f242a);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: readonly [number, number, number],
    r: number,
    g: number,
    b: number,
    gain: number,
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
    mesh.lookAt(0, 0, 0);
    envScene.add(mesh);
    disposables.push(geo, tex, mat);
  };

  // Warm-neutral raking key softbox (upper-right-front) for dry kozo crinkle relief
  addSoftbox(9.5, 8.2, [5.0, 5.2, 4.2], 1.0, 0.96, 0.89, 2.2);
  // Crisp cool-neutral fill strip (left) for raw-silk sheen & bamboo rib separation
  addSoftbox(5.2, 10.5, [-5.8, 2.8, 3.0], 0.92, 0.96, 1.0, 1.75);
  // Warm grazing rim kicker (rear-upper) for unspooling gossamer silk loops & cord knot
  addSoftbox(11.5, 4.8, [-0.6, 4.8, -5.6], 1.0, 0.94, 0.86, 1.85);
  // Overhead crown diffuser
  addSoftbox(9.0, 9.0, [0.0, 7.2, 0.5], 0.98, 0.95, 0.90, 1.25);
  // Subtle lateral fill cards so roasted split-bamboo ribs never crush to flat black
  addSoftbox(13.0, 5.0, [7.0, -0.2, 0.2], 0.88, 0.85, 0.80, 0.44);
  addSoftbox(13.0, 5.0, [-7.0, -0.2, -0.2], 0.84, 0.87, 0.91, 0.42);
  // Warm lower bounce card for underside of flat bamboo mouth hoop & drooping silk arcs
  addSoftbox(11.0, 11.0, [0.0, -6.0, 0.4], 1.0, 0.86, 0.68, 0.60);

  const renderTarget = pmrem.fromScene(envScene, 0.062);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

// ============================================================================
// Deterministic Tileable 2D Noise & Voronoi Crumpled-Paper Field Helpers
// ============================================================================

function tileHash2D(ix: number, iy: number, period: number, salt: number): number {
  const px = ((ix % period) + period) % period;
  const py = ((iy % period) + period) % period;
  const h = Math.sin(px * 127.1 + py * 311.7 + salt * 74.7) * 43758.5453123;
  return h - Math.floor(h);
}

function tileNoise2D(
  u: number,
  v: number,
  periodX: number,
  periodY: number,
  salt: number,
): number {
  const fx = u * periodX;
  const fy = v * periodY;
  const ix = Math.floor(fx);
  const iy = Math.floor(fy);
  const tx = fx - ix;
  const ty = fy - iy;
  const ux = tx * tx * (3 - 2 * tx);
  const uy = ty * ty * (3 - 2 * ty);

  const py0 = ((iy % periodY) + periodY) % periodY;
  const py1 = (((iy + 1) % periodY) + periodY) % periodY;

  const n00 = tileHash2D(ix, iy, periodX, salt + py0 * 0.013);
  const n10 = tileHash2D(ix + 1, iy, periodX, salt + py0 * 0.013);
  const n01 = tileHash2D(ix, iy + 1, periodX, salt + py1 * 0.013);
  const n11 = tileHash2D(ix + 1, iy + 1, periodX, salt + py1 * 0.013);
  const nx0 = n00 + (n10 - n00) * ux;
  const nx1 = n01 + (n11 - n01) * ux;
  return nx0 + (nx1 - nx0) * uy;
}

/**
 * Evaluates a tileable organic handmade kozo paper rumple & soft tension-fold heightfield
 * using multi-scale domain-warped FBM + soft elongated diagonal creases (zero closed
 * Voronoi mud-crack polygons!).
 */
function buildTileablePaperRumpleLayer(
  gridSize: number,
  salt: number,
): Float32Array {
  const out = new Float32Array(gridSize * gridSize);
  for (let y = 0; y < gridSize; y++) {
    const v = y / gridSize;
    for (let x = 0; x < gridSize; x++) {
      const u = x / gridSize;

      // Soft multi-octave organic domain warp for flowing stretched-paper wrinkles
      const wu = u + (tileNoise2D(u, v, 4, 4, salt + 1.7) - 0.5) * 0.14;
      const wv = v + (tileNoise2D(u, v, 4, 4, salt + 9.3) - 0.5) * 0.14;

      // Broad soft handmade sheet undulation (stretched across bamboo ribs)
      const broad =
        tileNoise2D(wu, wv, 5, 5, salt + 17.1) * 0.52 +
        tileNoise2D(wu, wv, 10, 10, salt + 29.4) * 0.30 +
        tileNoise2D(wu, wv, 21, 21, salt + 43.8) * 0.18;

      // Gentle elongated diagonal tension folds (anisotropic in u+v and u-v, fading naturally)
      const diagA = tileNoise2D(wu + wv * 0.5, wv - wu * 0.5, 9, 4, salt + 61.2);
      const diagB = tileNoise2D(wu - wv * 0.5, wv + wu * 0.5, 4, 11, salt + 79.5);
      // Soft folded billows (smooth cosine/quadratic modulation, never sharp dark crack trenches)
      const foldA = 1.0 - Math.abs(diagA * 2.0 - 1.0);
      const foldB = 1.0 - Math.abs(diagB * 2.0 - 1.0);
      const softCrease = (foldA * foldA * 0.55 + foldB * foldB * 0.45);

      // Fine handmade couched paper dimple & deckle micro-tooth
      const fine =
        tileNoise2D(u, v, 32, 32, salt + 97.1) * 0.62 +
        tileNoise2D(u, v, 64, 64, salt + 113.9) * 0.38;

      const hVal = broad * 0.54 + softCrease * 0.30 + fine * 0.16;
      out[y * gridSize + x] = Math.max(0, Math.min(1, hVal));
    }
  }
  return out;
}

function sampleBilinearPeriodic(buf: Float32Array, gridSize: number, u: number, v: number): number {
  const fx = ((u % 1) + 1) % 1 * gridSize;
  const fy = ((v % 1) + 1) % 1 * gridSize;
  const x0 = Math.floor(fx) % gridSize;
  const y0 = Math.floor(fy) % gridSize;
  const x1 = (x0 + 1) % gridSize;
  const y1 = (y0 + 1) % gridSize;
  const tx = fx - Math.floor(fx);
  const ty = fy - Math.floor(fy);

  const a = buf[y0 * gridSize + x0];
  const b = buf[y0 * gridSize + x1];
  const c = buf[y1 * gridSize + x0];
  const d = buf[y1 * gridSize + x1];
  return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty;
}

// ============================================================================
// Baked Seamless 1024x1024 Handmade Unryu / Kozo Rice-Paper 4-Channel Atlas
// R = fine & medium pale mulberry cellulose fiber mat + faint laid screen tooth
// G = organic handmade deckle pulp cloud thickness (0.08..0.96)
// B = authentic sinuous dark/warm Unryu mulberry bark-bast strands & flecks (zero spirals!)
// A = crumpled handmade paper fold & crease heightfield for crisp tangent normals
// ============================================================================

function createKozoPaperTexture(size = 1024): THREE.DataTexture {
  const baseGrid = 512;
  const rumpleGrid = buildTileablePaperRumpleLayer(baseGrid, 11.4);
  const pulpGrid = new Float32Array(baseGrid * baseGrid);

  for (let y = 0; y < baseGrid; y++) {
    const v = y / baseGrid;
    for (let x = 0; x < baseGrid; x++) {
      const u = x / baseGrid;
      const fbm =
        tileNoise2D(u, v, 5, 5, 1.7) * 0.44 +
        tileNoise2D(u, v, 11, 11, 6.3) * 0.28 +
        tileNoise2D(u, v, 23, 23, 13.9) * 0.18 +
        tileNoise2D(u, v, 47, 47, 29.1) * 0.10;
      const rawPulp = Math.max(0, Math.min(1, (fbm - 0.18) / 0.64));
      const smoothPulp = rawPulp * rawPulp * (3.0 - 2.0 * rawPulp);
      pulpGrid[y * baseGrid + x] = 0.08 + smoothPulp * 0.88;
    }
  }

  const pixelCount = size * size;
  const fiberBuf = new Float32Array(pixelCount);
  const pulpBuf = new Float32Array(pixelCount);
  const bastBuf = new Float32Array(pixelCount);
  const heightBuf = new Float32Array(pixelCount);

  let grainSeed = 7719283;
  const randGrain = () => {
    grainSeed = (grainSeed * 1664525 + 1013904223) >>> 0;
    return grainSeed / 4294967296;
  };

  for (let y = 0; y < size; y++) {
    const v = y / size;
    // Delicate handmade bamboo vat screen laid-line tooth (sugeta)
    const laidLine = 0.5 + 0.5 * Math.sin(v * 144.0 * Math.PI * 2.0);
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const idx = y * size + x;
      const pulp = sampleBilinearPeriodic(pulpGrid, baseGrid, u, v);
      const rumple = sampleBilinearPeriodic(rumpleGrid, baseGrid, u, v);
      const dryTooth = (randGrain() - 0.5) * 0.045;
      pulpBuf[idx] = pulp;
      fiberBuf[idx] = laidLine * 0.04;
      heightBuf[idx] = Math.max(0, Math.min(1, rumple * 0.82 + pulp * 0.18 + dryTooth));
    }
  }

  let seed = 20261125;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const deposit = (buf: Float32Array, px: number, py: number, radius: number, amp: number) => {
    const ir = Math.ceil(radius);
    const cx = Math.floor(px);
    const cy = Math.floor(py);
    const fx = px - cx;
    const fy = py - cy;
    const invR = 1 / Math.max(1e-4, radius);
    const rSq = radius * radius;
    for (let dy = -ir; dy <= ir; dy++) {
      const yy = (((cy + dy) % size) + size) % size;
      const ddy = dy - fy;
      const rowOff = yy * size;
      for (let dx = -ir; dx <= ir; dx++) {
        const ddx = dx - fx;
        const dSq = ddx * ddx + ddy * ddy;
        if (dSq < rSq) {
          const w = 1 - Math.sqrt(dSq) * invR;
          const xx = (((cx + dx) % size) + size) % size;
          const idx = rowOff + xx;
          const next = buf[idx] + amp * w * w;
          buf[idx] = next > 1.0 ? 1.0 : next;
        }
      }
    }
  };

  // Continuous stroke drawer (sub-half-radius interpolation -> ZERO beaded/stippled dots in 1:1 Retina!)
  const strokeSegment = (
    buf: Float32Array,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    radius: number,
    amp: number,
  ) => {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const spacing = Math.max(0.45, radius * 0.42);
    const subSteps = Math.max(1, Math.ceil(dist / spacing));
    const subAmp = amp / Math.max(1, subSteps * 0.55);
    for (let k = 0; k < subSteps; k++) {
      const f = k / subSteps;
      deposit(buf, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, radius, subAmp);
    }
  };

  // 1,500 crisp hair-fine luminous mulberry cellulose fibers into R (continuous strokes, zero dots!)
  for (let f = 0; f < 1500; f++) {
    let x = rand() * size;
    let y = rand() * size;
    let ang = rand() * Math.PI * 2;
    let angVel = 0;
    const isLong = f < 340;
    const steps = isLong ? 58 : 26;
    const stepLen = isLong ? 2.8 : 1.8;
    const radius = isLong ? 1.02 : 0.72;
    const amp = isLong ? 0.36 : 0.22;

    for (let s = 0; s < steps; s++) {
      const taper = Math.sin((s / Math.max(1, steps - 1)) * Math.PI);
      const wAmp = amp * (0.28 + 0.72 * taper);
      angVel = angVel * 0.78 + (rand() - 0.5) * 0.19;
      ang += angVel;
      const nx = x + Math.cos(ang) * stepLen;
      const ny = y + Math.sin(ang) * stepLen;
      strokeSegment(fiberBuf, x, y, nx, ny, radius, wAmp);
      if (isLong && (s & 1) === 0) {
        strokeSegment(heightBuf, x, y, nx, ny, radius * 1.12, wAmp * 0.11);
      }
      x = nx;
      y = ny;
    }
  }

  // 145 refined sinuous Unryu mulberry bark-bast filaments + tiny natural bark specks into B
  for (let f = 0; f < 145; f++) {
    let x = rand() * size;
    let y = rand() * size;
    let ang = (rand() - 0.5) * Math.PI * 2;
    let angVel = 0;
    const isFleck = f >= 122;
    const isBoldBast = f < 42;
    const steps = isFleck
      ? 2 + Math.floor(rand() * 3)
      : isBoldBast
        ? 64 + Math.floor(rand() * 54)
        : 30 + Math.floor(rand() * 36);
    const stepLen = isFleck ? 1.15 : 2.55;
    const baseRadius = isFleck
      ? 0.95 + rand() * 0.65
      : isBoldBast
        ? 1.18 + rand() * 0.82
        : 0.78 + rand() * 0.48;
    const amp = isFleck ? 0.76 : isBoldBast ? 0.80 : 0.48;

    for (let s = 0; s < steps; s++) {
      const t = s / Math.max(1, steps - 1);
      const taper = Math.pow(Math.sin(t * Math.PI), 0.55);
      const rNow = baseRadius * (0.52 + 0.54 * taper);
      const aNow = amp * (0.28 + 0.72 * taper);

      angVel = angVel * 0.74 + (rand() - 0.5) * 0.25;
      if (!isFleck && s > 5 && rand() < 0.06) {
        ang += (rand() - 0.5) * 0.52;
        angVel *= 0.3;
      }
      ang += angVel;
      const nx = x + Math.cos(ang) * stepLen;
      const ny = y + Math.sin(ang) * stepLen;

      strokeSegment(bastBuf, x, y, nx, ny, rNow, aNow);
      strokeSegment(heightBuf, x, y, nx, ny, rNow * 1.18, aNow * 0.18);

      // Feathery frayed secondary offshoots off bold Unryu bark strands
      if (isBoldBast && (s === Math.floor(steps * 0.46) || s === Math.floor(steps * 0.72)) && rand() < 0.52) {
        let bx = nx;
        let by = ny;
        let bAng = ang + (rand() > 0.5 ? 1 : -1) * (0.28 + rand() * 0.38);
        for (let bs = 0; bs < 16; bs++) {
          const bt = 1 - bs / 16;
          bAng += (rand() - 0.5) * 0.15;
          const bnx = bx + Math.cos(bAng) * 1.9;
          const bny = by + Math.sin(bAng) * 1.9;
          strokeSegment(bastBuf, bx, by, bnx, bny, Math.max(0.55, baseRadius * 0.46 * bt), amp * 0.56 * bt);
          bx = bnx;
          by = bny;
        }
      }

      x = nx;
      y = ny;
    }
  }

  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    rgba[i * 4] = Math.min(255, Math.round(fiberBuf[i] * 255));
    rgba[i * 4 + 1] = Math.min(255, Math.round(pulpBuf[i] * 255));
    rgba[i * 4 + 2] = Math.min(255, Math.round(bastBuf[i] * 255));
    rgba[i * 4 + 3] = Math.min(255, Math.round(heightBuf[i] * 255));
  }

  const tex = new THREE.DataTexture(rgba, size, size, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

// ============================================================================
// Baked Seamless 512x512 Split-Bamboo Culm, Braided Cord & Raw-Silk Atlas
// R = split-bamboo longitudinal vascular striations + dark sclerenchyma vessel pores (along V!)
// G = hand-rubbed bamboo oil burnish & subtle culm color variation
// B = twisted 3-ply dark hemp/silk suspension cord grooves + fiber fuzz
// A = anisotropic raw-silk micro-fibril longitudinal striations (along V)
// ============================================================================

function createBambooAndCordTexture(size = 512): THREE.DataTexture {
  const pixelCount = size * size;
  const bambooBuf = new Float32Array(pixelCount);
  const burnishBuf = new Float32Array(pixelCount);
  const cordBuf = new Float32Array(pixelCount);
  const silkBuf = new Float32Array(pixelCount);

  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const idx = y * size + x;

      // R: Crisp longitudinal split-bamboo vascular fiber bundles running along V
      const bambooFine =
        tileNoise2D(u, v, 84, 6, 3.1) * 0.48 +
        tileNoise2D(u, v, 168, 12, 11.7) * 0.34 +
        tileNoise2D(u, v, 32, 4, 21.3) * 0.18;
      bambooBuf[idx] = Math.max(0, Math.min(1, (bambooFine - 0.14) / 0.70));

      // G: Organic roasted/oiled bamboo grain depth variation
      burnishBuf[idx] =
        tileNoise2D(u, v, 16, 4, 33.1) * 0.62 + tileNoise2D(u, v, 40, 10, 57.7) * 0.38;

      // B: Twisted 3-ply hemp/silk cord helical strand grooves + fuzzy micro-fiber tooth
      const strandPhase = (u * 9.0 + v * 7.0) * Math.PI * 2.0;
      const cordWave = 0.5 + 0.5 * Math.sin(strandPhase);
      const cordProfile = Math.pow(cordWave, 0.68);
      const cordFuzz = tileNoise2D(u, v, 96, 96, 37.9);
      cordBuf[idx] = Math.max(0, Math.min(1, cordProfile * 0.74 + cordFuzz * 0.26));

      // A: Ultra-fine anisotropic raw-silk filament sheen micro-striations along V
      const silkFine =
        tileNoise2D(u, v, 8, 96, 49.3) * 0.56 + tileNoise2D(u, v, 18, 192, 71.9) * 0.44;
      silkBuf[idx] = Math.max(0, Math.min(1, (silkFine - 0.14) / 0.72));
    }
  }

  // Stamp 220 elongated dark bamboo sclerenchyma vascular pore flecks into R (Macro bamboo realism!)
  let seed = 91827364;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let p = 0; p < 220; p++) {
    const cx = Math.floor(rand() * size);
    const cy = Math.floor(rand() * size);
    const halfLen = 4 + Math.floor(rand() * 14);
    const halfW = rand() < 0.72 ? 1 : 2;
    const darken = 0.34 + rand() * 0.44;
    for (let dy = -halfLen; dy <= halfLen; dy++) {
      const yy = (((cy + dy) % size) + size) % size;
      const ty = 1 - Math.abs(dy) / Math.max(1, halfLen + 0.5);
      for (let dx = -halfW; dx <= halfW; dx++) {
        const xx = (((cx + dx) % size) + size) % size;
        const tx = 1 - Math.abs(dx) / Math.max(1, halfW + 0.5);
        const idx = yy * size + xx;
        bambooBuf[idx] = Math.max(0.02, bambooBuf[idx] * (1.0 - darken * tx * ty));
      }
    }
  }

  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    rgba[i * 4] = Math.min(255, Math.round(bambooBuf[i] * 255));
    rgba[i * 4 + 1] = Math.min(255, Math.round(burnishBuf[i] * 255));
    rgba[i * 4 + 2] = Math.min(255, Math.round(cordBuf[i] * 255));
    rgba[i * 4 + 3] = Math.min(255, Math.round(silkBuf[i] * 255));
  }

  const tex = new THREE.DataTexture(rgba, size, size, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

// ============================================================================
// Silk-Spun Lantern Chrysalis Material Bundle Factory
// Strictly physical mineral & botanical pigments from src/core/palette.ts
// ============================================================================

export function createSilkLanternMaterials(): SilkMaterialBundle {
  const uTime = uniform(0);
  const uUnravel = uniform(0.62);
  const uBreath = uniform(0.68);
  const uIgnition = uniform(0.78);
  const uEmberPulse = uniform(1.0);
  const uPointerWorld = uniform(new THREE.Vector3(0, 0, 0));
  const uPointerVel = uniform(new THREE.Vector3(0, 0, 0));
  const uFlamePos = uniform(new THREE.Vector3(0, -0.32, 0));

  const kozoPaperTex = createKozoPaperTexture(1024);
  const bambooCordTex = createBambooAndCordTexture(512);

  // Natural mineral & botanical pigment packs (zero digital neons)
  const pCalcite = createPigmentUniforms('boneCalcite');
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pSienna = createPigmentUniforms('burntSienna');
  const pHematite = createPigmentUniforms('hematiteRust');
  const pCinnabar = createPigmentUniforms('cinnabarVermilion');
  const pRealgar = createPigmentUniforms('realgarAmber');
  const pSulfur = createPigmentUniforms('sporeSulfur');

  // ---------------------------------------------------------------------------
  // 1. Rice Paper (HERO MATERIAL) — Handmade Unryu / Kozo Lantern Canopy
  //    - True orthonormal tangent-space crumpled-paper fold & bast fiber normals
  //    - Per-panel non-repeating UV offset via aPanelId
  //    - Dry unbleached ivory kozo upper crown sculpted by raking studio light
  //    - Rich incandescent honey-amber candled waist glow with high-contrast dark bast fibers
  // ---------------------------------------------------------------------------
  const ricePaper = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.91,
    clearcoat: 0.0,
    envMapIntensity: 0.54,
  });
  {
    const vUv = uv();
    const aRibProximity = clamp(attribute('aRibProximity', 'float'), 0.0, 1.0);
    const aWaistMask = clamp(attribute('aWaistMask', 'float'), 0.0, 1.0);
    const aPanelId = attribute('aPanelId', 'float');
    const aCrinkleRidge = clamp(attribute('aCrinkleRidge', 'float'), -1.0, 1.0);

    // Subtle organic breathing expansion of paper panels between meridian ribs
    const panelFreedom = float(1.0).sub(aRibProximity.mul(0.84));
    const breathAmp = uBreath.mul(0.026).mul(float(0.30).add(aWaistMask.mul(0.70))).mul(panelFreedom);
    ricePaper.positionNode = positionLocal.add(normalLocal.mul(breathAmp));

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 2.5);

    // Per-panel irrational UV shift so no two adjacent panels ever repeat bast strands or crinkles
    const panelShift = vec2(aPanelId.mul(4.731), aPanelId.mul(2.917));
    const paperUvMain = vec2(vUv.x.mul(4.2), vUv.y.mul(2.1)).add(panelShift);
    const paperUvFine = vec2(
      vUv.x.mul(7.6).add(vUv.y.mul(1.9)),
      vUv.y.mul(5.8).sub(vUv.x.mul(1.7)),
    ).add(panelShift.mul(1.37));

    // Strictly uniform-control-flow multi-tap sampling for razor-crisp fold & fiber height gradients
    const epsTex = float(0.0012);
    const sampleMain = texture(kozoPaperTex, paperUvMain);
    const sampleDu = texture(kozoPaperTex, paperUvMain.add(vec2(epsTex, 0.0)));
    const sampleDv = texture(kozoPaperTex, paperUvMain.add(vec2(0.0, epsTex)));
    const sampleFine = texture(kozoPaperTex, paperUvFine);
    const sampleFineDu = texture(kozoPaperTex, paperUvFine.add(vec2(epsTex.mul(1.15), 0.0)));
    const sampleFineDv = texture(kozoPaperTex, paperUvFine.add(vec2(0.0, epsTex.mul(1.15))));

    const fineFibers = clamp(sampleMain.r.mul(0.68).add(sampleFine.r.mul(0.42)), 0.0, 1.0);
    const pulpCloud = clamp(sampleMain.g.mul(0.76).add(sampleFine.g.mul(0.24)), 0.0, 1.0);
    const darkBast = clamp(sampleMain.b.mul(0.88).add(sampleFine.b.mul(0.22)), 0.0, 1.0);
    const crinkleH = clamp(sampleMain.a.mul(0.72).add(sampleFine.a.mul(0.28)), 0.0, 1.0);

    // True orthonormal ovoid tangent-space basis (N, T_azimuth, B_meridian)
    const nLoc = normalize(normalLocal);
    const tAz = normalize(vec3(nLoc.z, float(0.0), nLoc.x.negate()).add(vec3(0.0, 0.0, 1e-4)));
    const bMer = normalize(cross(nLoc, tAz));

    const poleFade = float(1.0).sub(smoothstep(float(0.96), float(0.998), vUv.y));
    const grazingAtten = smoothstep(float(0.07), float(0.42), nDotV).mul(poleFade);

    // Crisp handmade kozo paper rumple height (A) + raised mulberry bast fibers (B) + fine cellulose mat (R)
    const dCrinkleU = sampleDu.a.sub(sampleMain.a).mul(3.8).add(sampleFineDu.a.sub(sampleFine.a).mul(1.9));
    const dCrinkleV = sampleDv.a.sub(sampleMain.a).mul(3.8).add(sampleFineDv.a.sub(sampleFine.a).mul(1.9));
    const dFiberU = sampleDu.r.sub(sampleMain.r).mul(1.45).add(sampleDu.b.sub(sampleMain.b).mul(2.3));
    const dFiberV = sampleDv.r.sub(sampleMain.r).mul(1.45).add(sampleDv.b.sub(sampleMain.b).mul(2.3));

    const slopeU = dCrinkleU.add(dFiberU).mul(0.42).mul(grazingAtten);
    const slopeV = dCrinkleV.add(dFiberV).mul(0.42).mul(grazingAtten);
    const nPaperLocal = normalize(nLoc.sub(tAz.mul(slopeU)).sub(bMer.mul(slopeV)));

    const viewNormBase = normalize(transformNormalToView(nLoc).mul(faceDirection));
    const viewNormPaper = normalize(transformNormalToView(nPaperLocal).mul(faceDirection));
    ricePaper.normalNode = normalize(mix(viewNormBase, viewNormPaper, grazingAtten));

    // Kubelka-Munk 2-flux unbleached cream kozo paper reflectance
    const kmThickness = float(0.32)
      .add(pulpCloud.mul(0.46))
      .add(fineFibers.mul(0.24))
      .add(aRibProximity.mul(0.26));
    const paperBacking = mix(
      mix(pOchre.rInf.mul(0.84), pKaolin.rInf.mul(0.98), smoothstep(float(0.08), float(0.68), vUv.y)),
      pCalcite.rInf,
      float(0.48),
    );
    const kmPaper = kubelkaMunkGlaze(pKaolin.K, pKaolin.S, max(float(0.05), kmThickness), paperBacking);

    // Cool-warm ivory parchment crown (zero peach washout!) with soft organic rumple shading
    const crownParchment = mix(
      kmPaper,
      mix(pKaolin.rInf, pCalcite.rInf, float(0.62)),
      smoothstep(float(0.40), float(0.95), vUv.y).mul(0.55),
    );

    // Soft organic stretched-paper fold shading (zero heavy dark mud-crack grooves)
    const foldShade = float(0.93)
      .add(crinkleH.sub(0.5).mul(0.14))
      .add(aCrinkleRidge.mul(0.075));
    const foldedPaper = crownParchment.mul(clamp(foldShade, 0.86, 1.10));

    // Warm chestnut-umber Unryu bark-bast strands & luminous pale cellulose wisps
    const warmBastCol = mix(pUmber.rInf, pSienna.rInf, float(0.24)).mul(0.34);
    const withBast = mix(foldedPaper, warmBastCol, darkBast.mul(0.74));
    const withFineFibers = mix(withBast, pCalcite.rInf.mul(1.04), fineFibers.mul(0.22));

    // Pasted glue-lap edge darkening right beside each bamboo rib + subtle silk waist contact AO
    const ribShadowed = mix(withFineFibers, pUmber.rInf.mul(0.36), aRibProximity.mul(0.48));
    const waistAO = mix(
      ribShadowed,
      ribShadowed.mul(0.88),
      aWaistMask.mul(float(1.0).sub(clamp(uUnravel, 0.0, 1.0).mul(0.55))).mul(0.34),
    );

    // Interior hollow wall is warmer honey-ochre so looking up through the mouth glows richly
    const interiorTone = mix(waistAO, mix(pOchre.rInf, pRealgar.rInf, float(0.24)), float(0.38));
    ricePaper.colorNode = select(frontFacing, waistAO, interiorTone);

    // Strictly matte handmade dry paper (zero glossy clearcoat!)
    ricePaper.roughnessNode = clamp(
      float(0.88)
        .add(pulpCloud.sub(0.5).mul(0.07))
        .add(fineFibers.mul(0.05))
        .sub(crinkleH.sub(0.5).mul(0.04)),
      0.82,
      0.97,
    );

    // Physical distance-attenuated internal candled subsurface transmission (rich warm honey-amber!)
    const flameVec = positionWorld.sub(uFlamePos);
    const distFlameSq = dot(flameVec, flameVec);
    const flameProximity = exp(distFlameSq.negate().mul(1.58));

    // Strong rich honey-amber glow in lower-mid waist & mouth, dropping cleanly to ~8% in upper ivory crown
    const verticalGlow = float(1.0).sub(smoothstep(float(0.14), float(0.84), vUv.y).mul(0.92));

    // High-contrast silhouette blocking on sinuous dark mulberry bast strands & bamboo rib laps (Macro!)
    const fiberTrans = clamp(
      float(1.20)
        .sub(pulpCloud.mul(0.44))
        .sub(darkBast.mul(0.94))
        .sub(fineFibers.mul(0.20))
        .sub(aRibProximity.mul(0.54))
        .add(crinkleH.sub(0.5).mul(0.18)),
      0.08,
      1.20,
    );

    const opticalChord = float(0.14)
      .add(pulpCloud.mul(0.32))
      .add(darkBast.mul(0.62))
      .add(aRibProximity.mul(0.34));
    const blWarmth = beerLambertTransmittance(pOchre.extinction, opticalChord.mul(0.42));

    // Saturated honey-amber / realgar-ochre candled glow with tight golden-white hot heart (Hero match!)
    const hotCore = clamp(flameProximity.mul(verticalGlow).mul(1.12), 0.0, 1.0);
    const warmHoney = mix(
      mix(pRealgar.rInf, pOchre.rInf, float(0.52)),
      pOchre.rInf.mul(1.12),
      smoothstep(float(0.15), float(0.75), hotCore),
    );
    const coreGlowColor = mix(warmHoney, pCalcite.rInf.mul(1.05), pow(hotCore, 2.35).mul(0.42));

    const ignitionLevel = clamp(uIgnition, 0.0, 1.0);
    const litScale = float(0.08).add(ignitionLevel.mul(1.68)).mul(uEmberPulse);
    const sideGain = select(frontFacing, float(1.0), float(1.52));

    const subsurfaceGlow = coreGlowColor
      .mul(blWarmth)
      .mul(float(0.18).add(flameProximity.mul(1.12)))
      .mul(verticalGlow)
      .mul(fiberTrans)
      .mul(litScale)
      .mul(sideGain);

    // Subtle cool-ivory studio grazing rim + soft tactile pointer warmth
    const studioRim = pCalcite.rInf.mul(fresnel).mul(0.065);
    const ptrVec = positionWorld.sub(uPointerWorld);
    const ptrWarmth = pOchre.rInf
      .mul(exp(dot(ptrVec, ptrVec).negate().mul(2.6)))
      .mul(float(0.04).add(ignitionLevel.mul(0.09)));

    ricePaper.emissiveNode = subsurfaceGlow.add(studioRim).add(ptrWarmth);
  }

  // ---------------------------------------------------------------------------
  // 2. Bamboo Frame — Dark roasted/oiled split-bamboo meridian ribs & mouth hoop
  //    - Horizontal bent-bamboo vascular grain on hoops & vertical grain on ribs
  //    - Dark roasted outer culm rind vs warm honey-ochre inner split pith (aWoodFace)
  //    - Periodic darkened bamboo culm node scars (aNodeMask)
  //    - Firelit golden interior rim bounce around bottom mouth hoop
  // ---------------------------------------------------------------------------
  const bambooFrame = new THREE.MeshStandardNodeMaterial({
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.44,
    envMapIntensity: 1.08,
  });
  {
    const vUv = uv();
    const aWoodFace = clamp(attribute('aWoodFace', 'float'), 0.0, 1.0);
    const aNodeMask = clamp(attribute('aNodeMask', 'float'), 0.0, 1.0);
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);

    // V runs longitudinally along both meridian ribs and circular bent hoops!
    const bUv = vec2(vUv.x.mul(1.4), vUv.y.mul(2.4));
    const bS = texture(bambooCordTex, bUv);
    const bDu = texture(bambooCordTex, bUv.add(vec2(0.005, 0.0)));
    const bDv = texture(bambooCordTex, bUv.add(vec2(0.0, 0.005)));
    const vascular = bS.r;
    const burnish = bS.g;
    const organic = mineralToothFbm(positionLocal, float(6.2));

    // Dark roasted outer bamboo rind (aWoodFace = 1) vs warmer honey-tan split inner pith (aWoodFace = 0)
    const darkRind = pUmber.rInf.mul(0.28);
    const roastedHighlight = mix(pUmber.rInf.mul(0.68), pSienna.rInf.mul(0.64), vascular.mul(0.65));
    const outerSkin = mix(darkRind, roastedHighlight, float(0.28).add(vascular.mul(0.56)));

    const innerPith = mix(
      pUmber.rInf.mul(0.54),
      mix(pOchre.rInf.mul(0.76), pSienna.rInf.mul(0.68), float(0.42)),
      float(0.35).add(vascular.mul(0.50)),
    );

    const baseWood = mix(innerPith, outerSkin, aWoodFace)
      .mul(float(0.84).add(burnish.mul(0.22)).add(organic.sub(0.5).mul(0.14)));

    // Subtle darkened bamboo culm node ring scar (matching bottom hoop left node in Macro!)
    const withNode = mix(baseWood, pUmber.rInf.mul(0.19), aNodeMask.mul(0.58));

    bambooFrame.colorNode = withNode;
    bambooFrame.roughnessNode = clamp(
      mix(float(0.58), float(0.36), aWoodFace)
        .add(float(1.0).sub(vascular).mul(0.18))
        .add(aNodeMask.mul(0.20))
        .add(organic.sub(0.5).mul(0.06)),
      0.30,
      0.74,
    );

    // True orthonormal local tangent basis for crisp longitudinal vascular grooves on BOTH ribs & hoops
    const nLoc = normalize(normalLocal);
    const refUp = select(abs(nLoc.y).lessThan(0.92), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const tLoc = normalize(cross(refUp, nLoc));
    const bLoc = normalize(cross(nLoc, tLoc));
    const dVdU = bDu.r.sub(vascular).mul(2.4);
    const dVdV = bDv.r.sub(vascular).mul(1.8);
    const nBamboo = normalize(nLoc.sub(tLoc.mul(dVdU.mul(0.24))).sub(bLoc.mul(dVdV.mul(0.18))));
    bambooFrame.normalNode = normalize(transformNormalToView(nBamboo));

    // Warm interior amber glow bouncing onto the mouth hoop's inner/lower rim & rib undersides
    const toFlame = uFlamePos.sub(positionWorld);
    const distFlame = length(toFlame);
    const dirFlame = toFlame.div(max(float(0.001), distFlame));
    const faceFlame = clamp(dot(nW, dirFlame).mul(0.68).add(0.40), 0.0, 1.0);
    const flameFalloff = exp(distFlame.mul(distFlame).negate().mul(1.85));
    const rimBoost = float(0.46).add(pow(float(1.0).sub(nDotV), 1.9).mul(0.68));
    // Inner split pith catches warmer interior light than dark roasted outer rind
    const innerGlowGain = mix(float(1.35), float(0.72), aWoodFace);

    bambooFrame.emissiveNode = mix(pRealgar.rInf, pOchre.rInf, float(0.56))
      .mul(flameFalloff)
      .mul(faceFlame)
      .mul(rimBoost)
      .mul(innerGlowGain)
      .mul(clamp(uIgnition, 0.0, 1.0))
      .mul(uEmberPulse)
      .mul(0.62);
  }

  // ---------------------------------------------------------------------------
  // 3. Suspension Cord — Matte dark raw-umber 3-ply braided hemp cord & knot
  //    (Now assigned to its own dedicated mesh!)
  // ---------------------------------------------------------------------------
  const suspensionCord = new THREE.MeshStandardNodeMaterial({
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.92,
    envMapIntensity: 0.58,
  });
  {
    const vUv = uv();
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const rimFuzz = pow(float(1.0).sub(nDotV), 2.4);

    const cUv = vec2(vUv.x.mul(2.0), vUv.y.mul(16.0));
    const cS = texture(bambooCordTex, cUv);
    const cDu = texture(bambooCordTex, cUv.add(vec2(0.008, 0.0)));
    const cDv = texture(bambooCordTex, cUv.add(vec2(0.0, 0.008)));
    const cord = cS.b;

    const deepCord = pUmber.rInf.mul(0.18);
    const fiberCatch = mix(pUmber.rInf.mul(0.46), pKaolin.rInf.mul(0.36), float(0.28));
    suspensionCord.colorNode = mix(deepCord, fiberCatch, cord).add(pKaolin.rInf.mul(rimFuzz.mul(0.08)));
    suspensionCord.roughnessNode = clamp(float(0.87).add(float(1.0).sub(cord).mul(0.10)), 0.82, 0.97);

    const nLoc = normalize(normalLocal);
    const refUp = select(abs(nLoc.y).lessThan(0.92), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const tLoc = normalize(cross(refUp, nLoc));
    const bLoc = normalize(cross(nLoc, tLoc));
    const dCdU = cDu.b.sub(cord).mul(2.8);
    const dCdV = cDv.b.sub(cord).mul(2.8);
    const nCord = normalize(nLoc.sub(tLoc.mul(dCdU.mul(0.32))).sub(bLoc.mul(dCdV.mul(0.32))));
    suspensionCord.normalNode = normalize(transformNormalToView(nCord));
  }

  // ---------------------------------------------------------------------------
  // 4. Silk Wrap — 340 Hair-Fine Taut Raw-Silk Cocoon Skein Filaments
  //    - Pale raw champagne-silk at outer studio-lit rims, shifting to luminous
  //      honey-gold and deep backlit realgar-amber across the glowing lantern core!
  //    - Anisotropic cylindrical sheen + smooth breath/unwind coupling
  // ---------------------------------------------------------------------------
  const silkWrap = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.40,
    clearcoat: 0.06,
    clearcoatRoughness: 0.48,
    envMapIntensity: 1.05,
  });
  {
    const vUv = uv();
    const aStrandT = clamp(attribute('aStrandT', 'float'), 0.0, 1.0);
    const aStrandSeed = clamp(attribute('aStrandSeed', 'float'), 0.0, 1.0);
    const aUnwindOrder = clamp(attribute('aUnwindOrder', 'float'), 0.0, 1.0);
    const aSkeinId = clamp(attribute('aSkeinId', 'float'), 0.0, 1.0);

    // Outer/lower filaments loosen subtly as uUnravel rises; entire wrap expands with uBreath
    const unravelClamped = clamp(uUnravel, 0.0, 1.0);
    const unwindGate = smoothstep(
      aUnwindOrder.mul(0.62),
      aUnwindOrder.mul(0.62).add(0.38),
      unravelClamped,
    );
    const archEnvelope = sin(aStrandT.mul(3.14159265));
    const radialXZ = normalize(
      vec3(positionLocal.x, float(0.0), positionLocal.z).add(vec3(1e-5, 0.0, 0.0)),
    );
    // Slow meditative breathing flutter (zero high-freq buzz)
    const flutter = sin(aStrandT.mul(6.28318).add(uTime.mul(0.55)).add(aStrandSeed.mul(19.3)))
      .mul(0.0045)
      .mul(unwindGate);
    // Match paper panel breathing expansion so silk stays tautly seated with zero clipping
    const breathPush = uBreath.mul(0.025);
    const peelDist = unwindGate.mul(0.022).mul(archEnvelope).add(breathPush);
    const dropY = unwindGate.mul(0.009).mul(archEnvelope).negate();

    // Gentle peaceful local breeze swell from smoothed pointer velocity
    const wrapWorldOrig = modelWorldMatrix.mul(vec4(positionLocal, float(1.0))).xyz;
    const wrapPtrVec = wrapWorldOrig.sub(uPointerWorld);
    const wrapPtrInf = exp(dot(wrapPtrVec, wrapPtrVec).negate().mul(0.95))
      .mul(archEnvelope)
      .mul(float(0.25).add(unwindGate.mul(0.75)));
    const wrapBreeze = clamp(uPointerVel, vec3(-1.2, -1.2, -1.2), vec3(1.2, 1.2, 1.2))
      .mul(0.008)
      .mul(wrapPtrInf);

    silkWrap.positionNode = positionLocal
      .add(radialXZ.mul(peelDist.add(flutter)))
      .add(vec3(float(0.0), dropY, float(0.0)))
      .add(wrapBreeze);

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 2.1);

    // Fine twisted multi-ply raw-silk micro-fibril grain (eliminates smooth plastic-pipe specular)
    const twistUv = vec2(vUv.x.mul(3.0).add(vUv.y.mul(14.0)), vUv.y.mul(38.0).add(aStrandSeed.mul(7.0)));
    const sTex = texture(bambooCordTex, twistUv);
    const plyWave = sin(vUv.y.mul(95.0).add(vUv.x.mul(12.566)).add(aStrandSeed.mul(17.0)))
      .mul(0.5)
      .add(0.5);
    const silkGrain = mix(sTex.a, plyWave, float(0.38));

    // Backlit proximity to the glowing lantern heart:
    // Rich spun golden-ochre & toasted honey-amber silk bands (zero dusty beige washout!),
    // with soft satin raw-silk luster catching studio light along outer curves
    const flameVec = positionWorld.sub(uFlamePos);
    const flameAtten = exp(dot(flameVec, flameVec).negate().mul(1.55));

    const paleChampagne = mix(pOchre.rInf.mul(0.92), pKaolin.rInf.mul(0.94), float(0.45));
    const rawGoldSilk = mix(pOchre.rInf.mul(0.88), pRealgar.rInf.mul(0.76), aStrandSeed.mul(0.42));
    const deepAmberSilk = mix(pSienna.rInf.mul(0.72), pUmber.rInf.mul(0.66), aSkeinId.mul(0.42));

    const strandTone = mix(
      mix(deepAmberSilk, rawGoldSilk, float(0.28).add(aStrandSeed.mul(0.68))),
      paleChampagne,
      pow(aStrandSeed, 2.2).mul(0.46).mul(float(1.0).sub(flameAtten.mul(0.55))),
    );

    // Soft anisotropic raw-silk sheen (no harsh plastic wire highlight)
    const keyDir = normalize(vec3(0.55, 0.72, 0.58));
    const halfVec = normalize(keyDir.add(viewDirW));
    const cylSheen = pow(clamp(float(1.0).sub(abs(dot(nW, halfVec))), 0.0, 1.0), 3.2);
    const silkyGlint = mix(pOchre.rInf, pCalcite.rInf, float(0.58)).mul(
      cylSheen.mul(0.18).add(fresnel.mul(0.14)),
    );

    silkWrap.colorNode = strandTone.mul(float(0.74).add(silkGrain.mul(0.36))).add(silkyGlint);
    silkWrap.roughnessNode = clamp(float(0.32).add(float(1.0).sub(silkGrain).mul(0.22)), 0.28, 0.56);
    silkWrap.normalNode = normalize(transformNormalToView(normalLocal).mul(faceDirection));

    // Translucent backlit honey-amber filament glow (brightest directly in front of the core)
    const backlitCol = mix(pOchre.rInf, pRealgar.rInf, float(0.48).add(flameAtten.mul(0.22)));
    const backlitGlow = backlitCol
      .mul(float(0.09).add(flameAtten.mul(1.04)))
      .mul(float(0.46).add(fresnel.mul(0.68)))
      .mul(float(0.58).add(aStrandSeed.mul(0.58)))
      .mul(float(0.06).add(clamp(uIgnition, 0.0, 1.0).mul(0.92)))
      .mul(uEmberPulse);

    silkWrap.emissiveNode = backlitGlow;
  }

  // ---------------------------------------------------------------------------
  // 4b. Silk Gossamer Veil — Subtle Translucent Crossed Micro-Fibril Underlay
  //     Adds the luminous sub-pixel spun-silk gossamer density behind the 340
  //     3D filaments in Macro without ever looking like a solid plastic band.
  // ---------------------------------------------------------------------------
  const silkVeil = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    metalness: 0.0,
    roughness: 0.42,
    envMapIntensity: 1.05,
  });
  {
    const vUv = uv();
    const radialXZ = normalize(
      vec3(positionLocal.x, float(0.0), positionLocal.z).add(vec3(1e-5, 0.0, 0.0)),
    );
    silkVeil.positionNode = positionLocal.add(radialXZ.mul(uBreath.mul(0.025)));

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 1.9);

    // Two opposing diagonal silk-skein coordinate systems matching the 6 swaths
    const diagA = vec2(vUv.x.mul(6.0).add(vUv.y.mul(3.4)), vUv.y.mul(9.0).sub(vUv.x.mul(5.2)));
    const diagB = vec2(vUv.x.mul(6.0).sub(vUv.y.mul(3.4)), vUv.y.mul(9.0).add(vUv.x.mul(5.2)));
    const fibrilA = texture(bambooCordTex, diagA).a;
    const fibrilB = texture(bambooCordTex, diagB).a;

    // Fine crossed hair-filament micro-streaks gated cleanly inside the waist band
    const streakA = smoothstep(float(0.36), float(0.84), fibrilA);
    const streakB = smoothstep(float(0.36), float(0.84), fibrilB);
    const crossDensity = clamp(streakA.mul(0.62).add(streakB.mul(0.62)), 0.0, 1.0);

    // Smooth vertical feathering so top/bottom waist borders dissolve organically (zero hard edges!)
    const vFeather = smoothstep(float(0.0), float(0.24), vUv.y).mul(
      float(1.0).sub(smoothstep(float(0.72), float(1.0), vUv.y)),
    );
    const unravelThin = mix(float(1.0), float(0.62), clamp(uUnravel, 0.0, 1.0));

    const flameVec = positionWorld.sub(uFlamePos);
    const flameAtten = exp(dot(flameVec, flameVec).negate().mul(1.4));

    const veilBase = mix(
      mix(pKaolin.rInf, pOchre.rInf, float(0.48)),
      pRealgar.rInf,
      flameAtten.mul(0.34),
    );
    silkVeil.colorNode = veilBase.add(pCalcite.rInf.mul(fresnel.mul(0.24)));
    silkVeil.normalNode = normalize(transformNormalToView(normalLocal).mul(faceDirection));
    silkVeil.emissiveNode = mix(pOchre.rInf, pRealgar.rInf, float(0.38))
      .mul(float(0.24).add(flameAtten.mul(0.84)))
      .mul(clamp(uIgnition, 0.0, 1.0))
      .mul(uEmberPulse)
      .mul(0.76);
    silkVeil.opacityNode = clamp(
      crossDensity.mul(vFeather).mul(unravelThin).mul(float(0.12).add(fresnel.mul(0.16))),
      0.0,
      0.24,
    );
  }

  // ---------------------------------------------------------------------------
  // 5. Silk Loops — 38 Ultra-Fine Sweeping Gossamer Catenary Loops
  //    - Tuck cleanly against the waist when bound (uUnravel -> 0)
  //    - Unfurl into delicate luminous champagne-gold catenaries swaying in wind
  // ---------------------------------------------------------------------------
  const silkLoops = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.30,
    envMapIntensity: 1.22,
  });
  {
    const aLoopFreedom = clamp(attribute('aLoopFreedom', 'float'), 0.0, 1.0);
    const aLoopSeed = clamp(attribute('aLoopSeed', 'float'), 0.0, 1.0);
    const vUv = uv();

    const unravelClamped = clamp(uUnravel, 0.0, 1.0);
    const unfurl = smoothstep(float(0.02), float(0.90), unravelClamped);

    // Smoothly tuck protruding outer arcs against the waist when bound
    const radialXZLen = length(vec2(positionLocal.x, positionLocal.z));
    const radialXZ = normalize(
      vec3(positionLocal.x, float(0.0), positionLocal.z).add(vec3(1e-5, 0.0, 0.0)),
    );
    const excessRadius = max(float(0.0), radialXZLen.sub(0.50));
    const tuckInward = radialXZ
      .mul(excessRadius.mul(float(1.0).sub(unfurl)).mul(aLoopFreedom).mul(0.88))
      .negate();

    // Calm, meditative air-current drift when unfurled (slow frequency, peaceful rhythm)
    const wavePhase = uTime
      .mul(float(0.52).add(aLoopSeed.mul(0.28)))
      .add(aLoopSeed.mul(24.7))
      .add(aLoopFreedom.mul(2.4));
    const tangentXZ = vec3(radialXZ.z.negate(), float(0.0), radialXZ.x);
    const ambientSway = radialXZ
      .mul(sin(wavePhase).mul(0.020).add(unfurl.mul(0.020)))
      .add(tangentXZ.mul(cos(wavePhase.mul(0.78)).mul(0.022)))
      .add(vec3(float(0.0), sin(wavePhase.mul(0.62).add(1.3)).mul(0.015), float(0.0)))
      .mul(aLoopFreedom)
      .mul(float(0.14).add(unfurl.mul(0.86)));

    // Viscous, critically-damped breeze displacement (uPointerVel is smoothed in main.ts)
    const worldPosOrig = modelWorldMatrix.mul(vec4(positionLocal, float(1.0))).xyz;
    const ptrVec = worldPosOrig.sub(uPointerWorld);
    const ptrInfluence = exp(dot(ptrVec, ptrVec).negate().mul(0.55))
      .mul(aLoopFreedom)
      .mul(float(0.22).add(unfurl.mul(0.78)));
    const clampedVel = clamp(uPointerVel, vec3(-1.2, -1.2, -1.2), vec3(1.2, 1.2, 1.2));
    const pointerBillow = clampedVel.mul(0.032).mul(ptrInfluence);

    silkLoops.positionNode = positionLocal
      .add(radialXZ.mul(uBreath.mul(0.025)))
      .add(tuckInward)
      .add(ambientSway)
      .add(pointerBillow);

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 1.95);

    const loopTwistUv = vec2(vUv.x.mul(3.0).add(vUv.y.mul(10.0)), vUv.y.mul(28.0).add(aLoopSeed.mul(5.0)));
    const lTex = texture(bambooCordTex, loopTwistUv);
    const loopPly = sin(vUv.y.mul(72.0).add(vUv.x.mul(12.566)).add(aLoopSeed.mul(13.0)))
      .mul(0.5)
      .add(0.5);
    const loopGrain = mix(lTex.a, loopPly, float(0.35));
    const strandTone = mix(
      pOchre.rInf.mul(0.96),
      mix(pKaolin.rInf, pCalcite.rInf, float(0.45)),
      aLoopSeed.mul(0.54).add(loopGrain.mul(0.20)),
    );
    silkLoops.colorNode = strandTone.mul(float(0.80).add(loopGrain.mul(0.28))).add(pCalcite.rInf.mul(fresnel.mul(0.24)));
    silkLoops.roughnessNode = clamp(float(0.32).add(float(1.0).sub(loopGrain).mul(0.20)), 0.28, 0.54);
    silkLoops.normalNode = normalize(transformNormalToView(normalLocal).mul(faceDirection));

    // Luminous champagne-gold catchlight + warm amber lantern scatter along unfurled wisps
    const flameDistSq = dot(positionWorld.sub(uFlamePos), positionWorld.sub(uFlamePos));
    const flameScatter = exp(flameDistSq.negate().mul(0.95));
    const loopGlow = mix(pOchre.rInf, pRealgar.rInf, float(0.38))
      .mul(float(0.24).add(flameScatter.mul(0.84)).add(fresnel.mul(0.48)))
      .mul(float(0.20).add(clamp(uIgnition, 0.0, 1.0).mul(0.86)))
      .mul(uEmberPulse);

    silkLoops.emissiveNode = loopGlow;
  }

  // ---------------------------------------------------------------------------
  // 6. Origami Butterflies — 4 Natural Mineral-Dyed Handmade Washi Variants
  //    - Multi-ply Beer-Lambert paper overlap darkening (aPaperPlies: 1..3 plies!)
  //    - Visible handmade kozo paper fibers & mottled pulp translucency in Macro
  //    - Crisp folded mountain-crease pale-fiber catchlight (aCreaseProx)
  // ---------------------------------------------------------------------------
  type ButterflyPaletteSpec = {
    innerA: typeof pCinnabar;
    innerB: typeof pRealgar;
    innerMix: number;
    outerA: typeof pRealgar;
    outerB: typeof pOchre;
    outerMix: number;
    glowGain: number;
  };

  const butterflySpecs: ButterflyPaletteSpec[] = [
    // 0: Deep Cinnabar-Terracotta washi (Macro foreground butterfly!)
    {
      innerA: pHematite,
      innerB: pCinnabar,
      innerMix: 0.62,
      outerA: pCinnabar,
      outerB: pRealgar,
      outerMix: 0.56,
      glowGain: 0.96,
    },
    // 1: Burnt Realgar-Amber washi (classic warm ember butterfly)
    {
      innerA: pCinnabar,
      innerB: pRealgar,
      innerMix: 0.48,
      outerA: pRealgar,
      outerB: pOchre,
      outerMix: 0.36,
      glowGain: 1.04,
    },
    // 2: Warm Saffron-Apricot washi
    {
      innerA: pRealgar,
      innerB: pSienna,
      innerMix: 0.25,
      outerA: pRealgar,
      outerB: pOchre,
      outerMix: 0.68,
      glowGain: 1.08,
    },
    // 3: Golden Ochre / Spore-Sulfur washi (delicate yellow butterfly in Hero upper-right!)
    {
      innerA: pRealgar,
      innerB: pOchre,
      innerMix: 0.78,
      outerA: pOchre,
      outerB: pSulfur,
      outerMix: 0.55,
      glowGain: 1.02,
    },
  ];

  const origamiButterflies: THREE.MeshStandardNodeMaterial[] = butterflySpecs.map((spec, idx) => {
    const mat = new THREE.MeshStandardNodeMaterial({
      side: THREE.DoubleSide,
      transparent: false,
      depthWrite: true,
      metalness: 0.0,
      roughness: 0.72,
      envMapIntensity: 0.76,
    });

    const vUv = uv();
    const aPaperPlies = clamp(attribute('aPaperPlies', 'float'), 1.0, 3.2);
    const aCreaseProx = clamp(attribute('aCreaseProx', 'float'), 0.0, 1.0);

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 2.1);

    // Handmade dyed kozo washi paper texture scaled so fine fibers & pulp mottle read crisply in Macro!
    const paperShift = vec2(float(idx * 0.27), float(idx * 0.41));
    const bUv = vUv.mul(2.85).add(paperShift);
    const bPaper = texture(kozoPaperTex, bUv);
    const bPaperDu = texture(kozoPaperTex, bUv.add(vec2(0.004, 0.0)));
    const bPaperDv = texture(kozoPaperTex, bUv.add(vec2(0.0, 0.004)));
    const wingSpan = clamp(vUv.x, 0.0, 1.0);
    const wingChord = clamp(vUv.y, 0.0, 1.0);

    // Physical multi-ply Beer-Lambert fold thickness attenuation (1 ply = bright tips, 2-3 plies = deep folded center!)
    const plyExtra = max(float(0.0), aPaperPlies.sub(1.0));
    const plyTrans = exp(plyExtra.mul(-0.82));

    const innerCol = mix(spec.innerA.rInf, spec.innerB.rInf, float(spec.innerMix));
    const outerCol = mix(spec.outerA.rInf, spec.outerB.rInf, float(spec.outerMix));

    // Slightly darker rich folded body/shoulder pleats, glowing warm dyed washi across single-ply wing facets
    const baseHue = mix(innerCol, outerCol, smoothstep(float(0.06), float(0.90), wingSpan));
    const pulpMottle = float(0.80)
      .add(bPaper.r.mul(0.26))
      .add(bPaper.g.sub(0.5).mul(0.24))
      .sub(bPaper.b.mul(0.32));
    const plyAlbedo = baseHue.mul(pulpMottle).mul(mix(float(0.50), float(1.06), plyTrans));

    // Subtle dry-paper stretched-fiber highlight right along sharp mountain crease edges
    const creaseEdge = smoothstep(float(0.76), float(1.0), aCreaseProx);
    const creaseLift = pCalcite.rInf.mul(creaseEdge.mul(0.095).mul(mix(float(0.35), float(1.0), plyTrans)));

    mat.colorNode = plyAlbedo.add(creaseLift);
    mat.roughnessNode = clamp(
      float(0.76).add(bPaper.g.sub(0.5).mul(0.14)).add(bPaper.r.mul(0.08)),
      0.62,
      0.92,
    );

    // Preserve razor-sharp non-indexed origami crease facets PLUS subtle handmade dry-washi tooth micro-bump!
    const nLoc = normalize(normalLocal);
    const refUp = select(abs(nLoc.y).lessThan(0.92), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const tLoc = normalize(cross(refUp, nLoc));
    const bLoc = normalize(cross(nLoc, tLoc));
    const dPaperU = bPaperDu.a.sub(bPaper.a).mul(1.8).add(bPaperDu.r.sub(bPaper.r).mul(1.1));
    const dPaperV = bPaperDv.a.sub(bPaper.a).mul(1.8).add(bPaperDv.r.sub(bPaper.r).mul(1.1));
    const nWashi = normalize(nLoc.sub(tLoc.mul(dPaperU.mul(0.16))).sub(bLoc.mul(dPaperV.mul(0.16))));
    mat.normalNode = normalize(transformNormalToView(nWashi).mul(faceDirection));

    // Backlit ember translucency: single-ply lower swallowtail points & forewing tips glow warmly,
    // while multi-ply central folds stay rich and sculpted (exact Macro match!)
    const flameDistSq = dot(positionWorld.sub(uFlamePos), positionWorld.sub(uFlamePos));
    const flameProx = exp(flameDistSq.negate().mul(0.45));
    const tipMask = smoothstep(
      float(0.10),
      float(0.95),
      wingSpan.mul(0.60).add(float(1.0).sub(wingChord).mul(0.40)),
    );
    const paperGlowTrans = clamp(
      plyTrans.mul(float(1.12).sub(bPaper.g.mul(0.32)).sub(bPaper.b.mul(0.45))),
      0.14,
      1.12,
    );

    const glowTint = mix(innerCol, mix(outerCol, pOchre.rInf, float(0.35)), tipMask.mul(plyTrans));
    const emberRadiance = glowTint
      .mul(paperGlowTrans)
      .mul(float(0.46).add(flameProx.mul(0.56)).add(tipMask.mul(0.28)))
      .mul(float(0.78).add(uEmberPulse.mul(0.22)))
      .mul(float(spec.glowGain));

    const rimCatch = mix(outerCol, pCalcite.rInf, float(0.32)).mul(fresnel).mul(0.26).mul(plyTrans);
    mat.emissiveNode = emberRadiance.add(rimCatch);

    return mat;
  });

  const origamiButterfly = origamiButterflies[1];

  // ---------------------------------------------------------------------------
  // 7. Ember Core — Soft internal lantern ember/flame glow sphere
  // ---------------------------------------------------------------------------
  const emberCore = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  {
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);

    const flicker = sin(uTime.mul(5.2).add(positionLocal.y.mul(8.5))).mul(0.07).add(0.96);
    const hotNucleus = pow(nDotV, 3.6);
    const warmMid = pow(nDotV, 1.7);
    const softEdge = pow(nDotV, 1.35);

    const coreCol = mix(
      mix(pCinnabar.rInf, pRealgar.rInf, warmMid),
      mix(pOchre.rInf, pCalcite.rInf, float(0.78)),
      hotNucleus,
    );

    const ignitionGate = smoothstep(float(0.02), float(0.36), clamp(uIgnition, 0.0, 1.0));
    emberCore.colorNode = coreCol.mul(1.48).mul(uEmberPulse).mul(flicker);
    emberCore.opacityNode = clamp(softEdge.mul(0.88).mul(ignitionGate), 0.0, 1.0);
  }

  // ---------------------------------------------------------------------------
  // 8. Ember & Silk-Dust Motes — Dual-Population Organic Thermal Particles
  //    - 68% ultra-fine pinpoint golden silk-pollen glints catching lantern light
  //    - 32% softly elongated warm rising embers cooling calcite -> realgar -> cinnabar
  //    - Smooth birth/dissolve envelope, calm shimmer, and near-camera macro fade
  // ---------------------------------------------------------------------------
  const emberMotes = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  {
    const vUv = uv();
    // aMoteMeta: [lifeFrac (0..1), seed (0..1), isEmber (0 = fine dust, 1 = warm ember), stretch]
    const meta = attribute('aMoteMeta', 'vec4');
    const life = clamp(meta.x, 0.0, 1.0);
    const seed = clamp(meta.y, 0.0, 1.0);
    const isEmber = clamp(meta.z, 0.0, 1.0);
    const stretch = clamp(meta.w, 0.0, 1.0);

    // Subtly elongate rising embers vertically into soft organic teardrops (zero circular dots!)
    const centered = vUv.sub(0.5).mul(2.0);
    const yScale = float(1.0).div(float(1.0).add(isEmber.mul(stretch).mul(0.55)));
    const p = vec2(centered.x, centered.y.mul(yScale));
    const r = length(p);
    const rSq = r.mul(r);

    // Fine dust = ultra-crisp pinpoint glint; rising ember = tight warm nucleus + delicate soft halo
    const dustKernel = exp(rSq.negate().mul(38.0)).mul(0.95).add(exp(rSq.negate().mul(11.0)).mul(0.14));
    const emberKernel = exp(rSq.negate().mul(22.0)).mul(0.84).add(exp(rSq.negate().mul(5.8)).mul(0.28));
    const edgeCut = smoothstep(float(1.0), float(0.68), r);
    const shapeAlpha = clamp(mix(dustKernel, emberKernel, isEmber), 0.0, 1.0).mul(edgeCut);

    // Smooth thermal birth fade-in & slow ash-cooling dissolve (zero popping)
    const lifeFade = smoothstep(float(0.0), float(0.16), life).mul(
      float(1.0).sub(smoothstep(float(0.60), float(1.0), life)),
    );

    // Calm meditative shimmer (unique slow rhythm per particle)
    const shimmer = sin(uTime.mul(float(1.1).add(seed.mul(1.3))).add(seed.mul(31.4)))
      .mul(0.24)
      .add(0.76);

    // Smooth near-camera macro fade so Jeweler's Loupe zooms never hit giant blurry bokeh dots
    const camDist = length(cameraPosition.sub(positionWorld));
    const nearFade = smoothstep(float(0.45), float(1.05), camDist);

    // Thermal cooling palette: fresh warm calcite-gold -> rich realgar-amber -> fading cinnabar dust
    const freshGold = mix(pOchre.rInf, pCalcite.rInf, float(0.55));
    const midAmber = mix(pRealgar.rInf, pOchre.rInf, float(0.48));
    const coolCinnabar = mix(pCinnabar.rInf, pRealgar.rInf, float(0.38));
    const emberHue = mix(
      mix(freshGold, midAmber, smoothstep(float(0.05), float(0.55), life)),
      coolCinnabar,
      smoothstep(float(0.48), float(0.95), life),
    );
    const dustHue = mix(pOchre.rInf, pCalcite.rInf, float(0.42).add(seed.mul(0.32)));
    const moteCol = mix(dustHue, emberHue, isEmber);

    const moteGate = smoothstep(float(0.08), float(0.45), clamp(uUnravel, 0.0, 1.0)).mul(
      mix(float(0.28), float(0.84), clamp(uIgnition, 0.0, 1.0)),
    );
    const intensityGain = mix(float(0.82), float(1.14), isEmber);

    emberMotes.colorNode = moteCol.mul(intensityGain).mul(uEmberPulse);
    emberMotes.opacityNode = shapeAlpha.mul(lifeFade).mul(shimmer).mul(nearFade).mul(moteGate);
  }

  return {
    uniforms: {
      uTime,
      uUnravel,
      uBreath,
      uIgnition,
      uEmberPulse,
      uPointerWorld,
      uPointerVel,
      uFlamePos,
    },
    ricePaper,
    bambooFrame,
    suspensionCord,
    silkWrap,
    silkVeil,
    silkLoops,
    origamiButterfly,
    origamiButterflies,
    emberCore,
    emberMotes,
    dispose: () => {
      kozoPaperTex.dispose();
      bambooCordTex.dispose();
      ricePaper.dispose();
      bambooFrame.dispose();
      suspensionCord.dispose();
      silkWrap.dispose();
      silkVeil.dispose();
      silkLoops.dispose();
      for (const m of origamiButterflies) m.dispose();
      emberCore.dispose();
      emberMotes.dispose();
    },
  };
}
