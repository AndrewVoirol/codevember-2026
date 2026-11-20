import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
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
  normalLocal,
  normalView,
  normalize,
  positionLocal,
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
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';

function seededRand(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Fast deterministic integer-hash 2D value noise (zero Math.sin in inner loops -> < 35 ms for 1024x1024).
 */
function hash2Int(ix: number, iy: number): number {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}

function vNoiseFast(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2Int(ix, iy);
  const b = hash2Int(ix + 1, iy);
  const c = hash2Int(ix, iy + 1);
  const d = hash2Int(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function fbmFast(x: number, y: number, octaves = 4): number {
  let sum = 0;
  let amp = 0.5;
  let freq = 1.0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += vNoiseFast(x * freq, y * freq) * amp;
    norm += amp;
    // Irrational golden-angle rotation between octaves avoids horizontal/vertical grid artifacts
    const nx = x * 0.809017 - y * 0.587785;
    const ny = x * 0.587785 + y * 0.809017;
    x = nx;
    y = ny;
    freq *= 2.04;
    amp *= 0.5;
  }
  return sum / Math.max(1e-5, norm);
}

/**
 * Generates a crisp 1024×1024 4-strip seamless botanical kelp-parchment atlas:
 * Each 256×1024 vertical strip represents one unique cured kelp half-sheet:
 *   uLocal = 0 -> central laced seam margin
 *   uLocal = 1 -> clamped walnut rib margin
 * Channels:
 * - R: hierarchical curved botanical vascular venation + fine translucent cellular parenchyma web
 * - G: mottled dried-alginate parchment thickness clouds & drying watermarks
 * - B: powdery Bone-Calcite mannitol salt efflorescence concentrated near weathered flanks
 * - A: physical relief heightfield (raised vein ridges + dried parchment shrinkage creases + micro-tooth)
 */
export function createKelpParchmentTexture(size = 1024): THREE.DataTexture {
  const stripCount = 4;
  const stripW = size / stripCount;
  const data = new Uint8Array(size * size * 4);
  const veinCore = new Float32Array(size * size);
  const veinHalo = new Float32Array(size * size);

  const stampSegmentInStrip = (
    stripIdx: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    r0: number,
    r1: number,
    strength: number
  ) => {
    const xBase = stripIdx * stripW;
    const xMinBound = xBase + 1;
    const xMaxBound = xBase + stripW - 2;
    const pad = Math.ceil(Math.max(r0, r1) * 2.4 + 2);
    const minX = Math.max(xMinBound, Math.floor(Math.min(x0, x1) - pad));
    const maxX = Math.min(xMaxBound, Math.ceil(Math.max(x0, x1) + pad));
    const minY = Math.max(1, Math.floor(Math.min(y0, y1) - pad));
    const maxY = Math.min(size - 2, Math.ceil(Math.max(y0, y1) + pad));

    const dx = x1 - x0;
    const dy = y1 - y0;
    const lenSq = Math.max(1e-5, dx * dx + dy * dy);

    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / lenSq));
        const cx = x0 + dx * t;
        const cy = y0 + dy * t;
        const dist = Math.hypot(px - cx, py - cy);
        const rad = r0 + (r1 - r0) * t;
        const haloRad = rad * 2.2 + 1.4;
        if (dist < haloRad) {
          const idx = py * size + px;
          // Crisp tapered inner vascular canal
          if (dist < rad + 1.1) {
            const w = Math.max(0, Math.min(1, 1 - (dist - rad * 0.35) / (rad * 0.72 + 0.9)));
            const coreVal = w * w * strength;
            if (coreVal > veinCore[idx]) veinCore[idx] = coreVal;
          }
          // Soft organic tissue ridge halo around the vein
          const hw = Math.max(0, 1 - dist / haloRad);
          const hVal = hw * hw * strength * 0.55;
          if (hVal > veinHalo[idx]) veinHalo[idx] = hVal;
        }
      }
    }
  };

  for (let strip = 0; strip < stripCount; strip++) {
    const rand = seededRand(172026 + strip * 9973);
    const xBase = strip * stripW;

    // Smooth outward-flowing dendritic botanical vein drawer (never loops backward into messy hair scribbles)
    const growBotanicalVein = (
      x: number,
      y: number,
      angle: number,
      baseDir: number,
      turnBias: number,
      lengthPx: number,
      radius: number,
      depth: number,
      strength: number
    ) => {
      const steps = depth === 0 ? 24 : depth === 1 ? 14 : 9;
      let cx = x;
      let cy = y;
      let curAngle = angle;
      let curTurn = turnBias;
      const stepLen = lengthPx / steps;

      for (let s = 0; s < steps; s++) {
        const f0 = s / steps;
        const f1 = (s + 1) / steps;
        // Smooth organic curvature drift gently guided along outward botanical flow
        curTurn = curTurn * 0.88 + (rand() - 0.5) * (depth === 0 ? 0.048 : 0.085);
        curAngle += curTurn;
        // Keep veins flowing strictly outward from their origin margin (never looping back)
        const rel = curAngle - baseDir;
        if (rel > 1.05) {
          curAngle = baseDir + 1.05;
          curTurn *= -0.35;
        } else if (rel < -1.05) {
          curAngle = baseDir - 1.05;
          curTurn *= -0.35;
        }

        const nx = cx + Math.cos(curAngle) * stepLen;
        const ny = cy + Math.sin(curAngle) * stepLen;
        const taper0 = Math.pow(1 - f0 * 0.76, 1.18);
        const taper1 = Math.pow(1 - f1 * 0.76, 1.18);
        const rA = Math.max(0.36, radius * taper0);
        const rB = Math.max(0.30, radius * taper1);

        stampSegmentInStrip(
          strip,
          cx,
          cy,
          nx,
          ny,
          rA,
          rB,
          strength * (1 - f0 * 0.34)
        );

        // Natural botanical dichotomous / lateral branching at 28°..48° outward
        if (depth < 2 && s >= 2 && s <= steps - 3 && rand() < (depth === 0 ? 0.36 : 0.28)) {
          const forkSign = rand() < 0.5 ? -1 : 1;
          const forkAngle = curAngle + forkSign * (0.42 + rand() * 0.32);
          growBotanicalVein(
            nx,
            ny,
            forkAngle,
            baseDir,
            forkSign * (0.008 + rand() * 0.016),
            lengthPx * (0.46 + rand() * 0.22),
            rB * 0.72,
            depth + 1,
            strength * 0.78
          );
        }

        cx = nx;
        cy = ny;
        if (cx < xBase + 2 || cx > xBase + stripW - 3 || cy < 4 || cy > size - 5) break;
      }
    };

    // 1. Sculptural primary kelp veins fanning smoothly outward from the glowing central seam (uLocal -> 0)
    const seamPrimaryCount = 10;
    for (let i = 0; i < seamPrimaryCount; i++) {
      const vFrac = 0.08 + (i / (seamPrimaryCount - 1)) * 0.84 + (rand() - 0.5) * 0.025;
      const startY = size * Math.max(0.05, Math.min(0.95, vFrac));
      const startX = xBase + 2 + rand() * 4;
      const fanTilt = (vFrac - 0.5) * 0.72 + (rand() - 0.5) * 0.28;
      const len = stripW * (0.68 + rand() * 0.38);
      // Bolder dendritic roots near the equatorial belly (Concept F focal detail)
      const bellyBoost = 1.0 + 0.35 * Math.sin(vFrac * Math.PI);
      const rad = (1.75 + rand() * 0.85) * bellyBoost;
      const turn = (rand() - 0.5) * 0.024;
      growBotanicalVein(startX, startY, fanTilt, 0, turn, len, rad, 0, 0.96);
    }

    // 2. Delicate secondary veins creeping gently inward from the walnut rib margin (uLocal -> 1)
    const ribVeinCount = 6;
    for (let i = 0; i < ribVeinCount; i++) {
      const vFrac = 0.12 + (i / (ribVeinCount - 1)) * 0.76 + (rand() - 0.5) * 0.035;
      const startY = size * vFrac;
      const startX = xBase + stripW - 3 - rand() * 4;
      const baseAngle = Math.PI + (vFrac - 0.5) * 0.48 + (rand() - 0.5) * 0.34;
      const len = stripW * (0.30 + rand() * 0.22);
      growBotanicalVein(startX, startY, baseAngle, Math.PI, (rand() - 0.5) * 0.025, len, 1.25, 1, 0.62);
    }
  }

  // Compose final 1024×1024 RGBA botanical channels strip-by-strip
  for (let py = 0; py < size; py++) {
    const v = py / size;
    for (let px = 0; px < size; px++) {
      const strip = Math.min(stripCount - 1, Math.floor(px / stripW));
      const uLocal = (px - strip * stripW) / stripW; // 0 at seam, 1 at walnut rib
      const seedX = strip * 17.3 + 2.1;
      const seedY = strip * 11.7 + 4.3;

      // Domain-warped organic coordinates scaled for isotropic 1:4 half-sheet aspect ratio
      const wx = uLocal * 4.2 + seedX;
      const wy = v * 15.5 + seedY;
      const warpX = (fbmFast(wx * 0.85 + 3.1, wy * 0.85 + 1.9, 3) - 0.5) * 0.95;
      const warpY = (fbmFast(wx * 0.85 + 9.4, wy * 0.85 + 6.2, 3) - 0.5) * 0.95;

      // Fine translucent cellular parenchyma web (two scales of domain-warped organic ridges)
      const cellA = fbmFast((wx + warpX) * 3.4, (wy + warpY) * 3.4, 3);
      const cellB = fbmFast((wx - warpY * 0.7) * 7.2 + 5.1, (wy + warpX * 0.7) * 7.2 + 2.8, 2);
      const ridgeA = Math.pow(Math.max(0, 1.0 - Math.abs(cellA - 0.5) * 6.8), 2.1) * 0.36;
      const ridgeB = Math.pow(Math.max(0, 1.0 - Math.abs(cellB - 0.5) * 5.8), 2.2) * 0.22;

      const idx = py * size + px;
      const veinSignal = Math.min(1.0, veinCore[idx] * 0.92 + veinHalo[idx] * 0.26 + ridgeA + ridgeB);

      // Organic dried-alginate parchment mottle & subtle tidal drying rings
      const mottleCoarse = fbmFast(wx * 0.72 + warpX * 0.5, wy * 0.72 + warpY * 0.5, 4);
      const mottleFine = fbmFast(wx * 2.4 + 13.2, wy * 2.4 + 8.7, 3);
      // Soft darker mottling patches & subtle dried rim folds
      const thicknessVal = Math.max(
        0,
        Math.min(1, mottleCoarse * 0.72 + mottleFine * 0.28 + veinHalo[idx] * 0.22)
      );

      // Powdery Bone-Calcite mannitol coastal salt bloom (stronger along outer flank & crevices)
      const flankBias = Math.pow(Math.max(0, Math.min(1, uLocal * 1.08)), 1.55);
      const saltField = fbmFast(wx * 1.45 + 21.4 + warpX * 0.6, wy * 1.45 + 14.2 + warpY * 0.6, 4);
      const saltDetail = fbmFast(wx * 5.2 + 4.8, wy * 5.2 + 19.1, 2);
      const saltCluster = Math.max(0, (saltField - 0.52) * 3.2) * (0.45 + 0.55 * saltDetail);
      const saltVal = Math.min(1.0, saltCluster * (0.2 + 0.85 * flankBias));

      // Physical heightfield for tangent-space micro-normals:
      // Raised vascular vein cords + dried parchment shrinkage crinkles + subtle cellular relief
      const crinkleLarge = fbmFast(wx * 1.85 + warpX * 1.1 + 31.0, wy * 1.85 + warpY * 1.1 + 7.5, 3);
      const crinkleCrease = Math.abs(crinkleLarge - 0.5) * 2.0;
      const microTooth = fbmFast(wx * 9.5 + 11.1, wy * 9.5 + 23.4, 2);
      const heightVal = Math.max(
        0,
        Math.min(
          1,
          0.32 +
            veinCore[idx] * 0.36 +
            veinHalo[idx] * 0.14 +
            (1 - crinkleCrease) * 0.16 +
            (ridgeA + ridgeB) * 0.18 +
            (microTooth - 0.5) * 0.12
        )
      );

      const o = idx * 4;
      data[o] = Math.round(veinSignal * 255);
      data[o + 1] = Math.round(thicknessVal * 255);
      data[o + 2] = Math.round(saltVal * 255);
      data[o + 3] = Math.round(heightVal * 255);
    }
  }

  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  // Strictly ClampToEdgeWrapping -> zero wrap seams across strips!
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Creates a smooth Gaussian-feathered softbox luminance texture for PMREM studio reflections.
 */
function createFeatheredSoftboxTexture(r: number, g: number, b: number, size = 128): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  for (let py = 0; py < size; py++) {
    const ny = (py / (size - 1)) * 2 - 1;
    for (let px = 0; px < size; px++) {
      const nx = (px / (size - 1)) * 2 - 1;
      const dist = Math.hypot(nx, ny);
      const falloff = Math.exp(-dist * dist * 2.35) * Math.max(0, 1 - Math.pow(Math.min(1, dist), 4));
      const idx = (py * size + px) * 4;
      data[idx] = Math.min(255, Math.round(r * falloff * 255));
      data[idx + 1] = Math.min(255, Math.round(g * falloff * 255));
      data[idx + 2] = Math.min(255, Math.round(b * falloff * 255));
      data[idx + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return tex;
}

export interface KelpStudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

/**
 * Generates a studio softbox PMREM cubemap calibrated for Day 17's candlelit Cabinet-of-Curiosities
 * mood: darkroom negative fill preserves dramatic backlit contrast while giving aged patinated brass,
 * oiled dark-walnut staves, and blown Edison glass crisp, realistic metallic & dielectric sheen.
 */
export function createKelpStudioEnvironment(
  renderer: THREE.WebGPURenderer
): KelpStudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0b0e12);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: [number, number, number],
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
    mesh.lookAt(0, 0, 0);
    envScene.add(mesh);
    disposables.push(geo, tex, mat);
  };

  // Warm overhead-right studio calcite softbox (matches SpecimenStage keyLight direction)
  addSoftbox(10.0, 8.5, [4.6, 5.8, 4.4], 1.0, 0.96, 0.89, 1.65);
  // Soft cool-neutral opposite-flank scrim (separates aged brass knurling & dark walnut chamfers)
  addSoftbox(8.5, 9.5, [-5.6, 2.2, 3.2], 0.88, 0.93, 1.0, 1.05);
  // Back-top grazing calcite rim scrim (wraps clean sheen across brass canopy top & kelp silhouette)
  addSoftbox(11.0, 5.0, [-1.4, 5.2, -5.6], 0.96, 0.95, 0.92, 1.35);
  // Subtle warm underside bounce (lifts interior brass flange & bottom walnut finials)
  addSoftbox(10.0, 10.0, [0, -6.5, 0.5], 1.0, 0.82, 0.6, 0.42);

  const renderTarget = pmrem.fromScene(envScene, 0.085);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

function createKelpLanternUniforms() {
  return {
    uPokeLocal: uniform(new THREE.Vector3(0, 0, 1.0)),
    uPokeBay: uniform(-1.0),
    uPokeDepth: uniform(0.0),
    uRippleAmp: uniform(0.0),
    uRipplePhase: uniform(0.0),
    uBulbLocal: uniform(new THREE.Vector3(0, 0.08, 0)),
    uGlowIntensity: uniform(1.0),
  };
}

export type KelpLanternUniforms = ReturnType<typeof createKelpLanternUniforms>;

export interface KelpLanternMaterials {
  uniforms: KelpLanternUniforms;
  parchmentTex: THREE.DataTexture;
  parchmentMat: THREE.MeshStandardNodeMaterial;
  frillMat: THREE.MeshStandardNodeMaterial;
  walnutMat: THREE.MeshStandardNodeMaterial;
  brassMat: THREE.MeshStandardNodeMaterial;
  ceramicSocketMat: THREE.MeshStandardNodeMaterial;
  lacingMat: THREE.MeshStandardNodeMaterial;
  hempCordMat: THREE.MeshStandardNodeMaterial;
  bulbGlassMat: THREE.MeshPhysicalNodeMaterial;
  bulbHardwareMat: THREE.MeshStandardNodeMaterial;
  filamentMat: THREE.MeshBasicNodeMaterial;
  filamentAuraMat: THREE.MeshBasicNodeMaterial;
  dispose: () => void;
}

/**
 * Shared TSL displacement builder used identically by both `parchmentMat` and `lacingMat`
 * so hovering, pressing, or plucking a bay flexes the kelp skin AND its cross-laced stitching
 * in 100% geometric lockstep (zero detachment or midair floating laces!).
 */
function buildSharedMembraneFlex(uniforms: KelpLanternUniforms) {
  const bayMeta = attribute('aBayMeta', 'vec4'); // x=signedU, y=vNorm, z=bayIndex, w=thickness
  const signedU = bayMeta.x;
  const vNorm = bayMeta.y;
  const bayIdx = bayMeta.z;
  const baseThickness = bayMeta.w;

  // Pin displacement cleanly at the rigid walnut ribs (|u| -> 1) and top brass collar (v -> 0)
  const ribPin = smoothstep(float(0.97), float(0.38), abs(signedU));
  const collarPin = smoothstep(float(0.03), float(0.16), vNorm);

  // Isolate poke indentation to the actively hovered/pressed bay (if uPokeBay >= 0)
  // so pressing one bay never bleeds across a solid walnut rib into the neighboring bay.
  const hasSpecificBay = smoothstep(float(-0.5), float(-0.1), uniforms.uPokeBay);
  const sameBayMask = float(1.0).sub(smoothstep(float(0.25), float(0.75), abs(bayIdx.sub(uniforms.uPokeBay))));
  const bayGate = mix(float(1.0), sameBayMask, hasSpecificBay);

  const skinFreedom = ribPin.mul(collarPin).mul(bayGate);

  const relPoke = positionLocal.sub(uniforms.uPokeLocal);
  const dPoke = length(relPoke);
  const pokeGauss = exp(dPoke.mul(dPoke).negate().mul(6.8));
  const rippleEnv = exp(dPoke.negate().mul(2.4));
  const rippleArg = dPoke.mul(20.0).sub(uniforms.uRipplePhase);
  const rippleWave = sin(rippleArg).mul(rippleEnv).mul(uniforms.uRippleAmp).mul(0.024);

  const indentAmt = pokeGauss.mul(uniforms.uPokeDepth).add(rippleWave).mul(skinFreedom);

  // Outward radial direction in XZ + subtle Y curvature for consistent inward membrane displacement
  // (used identically by parchment vertices and lacing cylinder vertices regardless of local tube normal!)
  const radDir = normalize(vec3(positionLocal.x, positionLocal.y.mul(0.32), positionLocal.z).add(vec3(0.0, 0.0, 1e-5)));

  // Analytical spatial derivative d(indentAmt)/d(dPoke) for crisp normal tilt on poke & drumhead ripples
  const dGaussDd = pokeGauss.mul(dPoke.mul(-13.6)).mul(uniforms.uPokeDepth);
  const dRippleDd = cos(rippleArg)
    .mul(20.0)
    .sub(sin(rippleArg).mul(2.4))
    .mul(rippleEnv)
    .mul(uniforms.uRippleAmp)
    .mul(0.024);
  const dIndentDd = dGaussDd.add(dRippleDd).mul(skinFreedom);
  const pokeDir = relPoke.div(max(float(0.02), dPoke));

  return {
    signedU,
    vNorm,
    bayIdx,
    baseThickness,
    skinFreedom,
    pokeGauss,
    indentAmt,
    radDir,
    dIndentDd,
    pokeDir,
  };
}

export function createKelpLanternMaterials(): KelpLanternMaterials {
  const realgar = createPigmentUniforms('realgarAmber');
  const ochre = createPigmentUniforms('yellowOchre');
  const moss = createPigmentUniforms('mossUmbrous');
  const umber = createPigmentUniforms('rawUmber');
  const sienna = createPigmentUniforms('burntSienna');
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');

  const uniforms = createKelpLanternUniforms();
  const parchmentTex = createKelpParchmentTexture(1024);

  // ---------------------------------------------------------------------------
  // 1. Cured Kelp-Leather & Alginate Parchment Panels (Backlit Subsurface + Relief TSL)
  // ---------------------------------------------------------------------------
  const parchmentMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.64,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });

  const flex = buildSharedMembraneFlex(uniforms);
  const stripBounds = attribute('aStripBounds', 'vec2'); // x=uMin, y=uMax

  // Push parchment inward along shared radial normal when hovered, pressed, or plucked
  parchmentMat.positionNode = positionLocal.sub(flex.radDir.mul(flex.indentAmt));

  // Sample seamless strip-clamped botanical atlas (ZERO wrap seams across all 6 bays!)
  const baseUv = uv();
  const kelpSample = texture(parchmentTex, baseUv);
  const veinMask = kelpSample.r;
  const mottleVal = kelpSample.g;
  const saltBloom = kelpSample.b;
  const heightCenter = kelpSample.a;

  // 2 clamped neighbor height taps strictly inside [uMin, uMax] for crisp vein & crinkle micro-normals
  const epsU = float(0.0014);
  const epsV = float(0.0014);
  const uvRight = vec2(clamp(baseUv.x.add(epsU), stripBounds.x, stripBounds.y), baseUv.y);
  const uvDown = vec2(baseUv.x, clamp(baseUv.y.add(epsV), float(0.004), float(0.996)));
  const hRight = texture(parchmentTex, uvRight).a;
  const hDown = texture(parchmentTex, uvDown).a;
  const dHu = hRight.sub(heightCenter).mul(11.5);
  const dHv = hDown.sub(heightCenter).mul(11.5);

  // Build orthonormal local tangent basis on the curved gourd meridian for bump & poke-slope normals
  const nGeom = normalize(normalLocal);
  const upRef = vec3(0.0, 1.0, 0.0);
  const tAzim = normalize(cross(upRef, nGeom).add(vec3(1e-5, 0.0, 0.0)));
  const tMerid = normalize(cross(nGeom, tAzim));
  // Flip horizontal bump derivative on left half-sheet (signedU < 0) where u increases toward -tAzim
  const sideSign = select(flex.signedU.greaterThanEqual(0.0), float(1.0), float(-1.0));

  // Combine raised vein/crinkle relief + fine 3D papery grain + analytical poke/ripple dimple tilt
  const microGrainC = mineralToothFbm(positionLocal.mul(24.0), float(1.0));
  const microGrainX = mineralToothFbm(positionLocal.add(tAzim.mul(0.008)).mul(24.0), float(1.0));
  const microGrainY = mineralToothFbm(positionLocal.add(tMerid.mul(0.008)).mul(24.0), float(1.0));
  const microBumpU = microGrainX.sub(microGrainC).mul(0.32);
  const microBumpV = microGrainY.sub(microGrainC).mul(0.32);

  // Tangent projection of poke/ripple gradient (inward indentation tilts surface normal toward poke center)
  const pokeGradTangent = flex.pokeDir.sub(nGeom.mul(dot(flex.pokeDir, nGeom))).mul(flex.dIndentDd.mul(2.1));

  const perturbedLocalN = normalize(
    nGeom
      .sub(tAzim.mul(dHu.mul(sideSign).mul(0.095).add(microBumpU)))
      .sub(tMerid.mul(dHv.mul(0.095).add(microBumpV)))
      .add(pokeGradTangent)
  );
  // CRITICAL for DoubleSide: multiply by faceDirection so BOTH exterior AND interior faces light outward!
  parchmentMat.normalNode = transformNormalToView(perturbedLocalN).mul(faceDirection);

  // Optical thickness: thicker at ribs, veins, mottle clouds & hem; thins slightly under taut poke stretch
  const stretchThinning = clamp(flex.pokeGauss.mul(uniforms.uPokeDepth).mul(2.1).mul(flex.skinFreedom), 0.0, 0.24);
  const opticalThickness = clamp(
    flex.baseThickness
      .add(mottleVal.mul(0.32))
      .add(veinMask.mul(0.48))
      .sub(stretchThinning),
    0.12,
    1.85
  );

  // Exterior reflected albedo via 2-flux Kubelka-Munk: warm ochre-sienna kelp transitioning to dark cured margins
  const marginMix = smoothstep(float(0.52), float(0.96), abs(flex.signedU))
    .add(smoothstep(float(0.83), float(1.0), flex.vNorm))
    .add(smoothstep(float(0.11), float(0.0), flex.vNorm).mul(0.78))
    .clamp(0.0, 1.0);
  // Subtle folded cured leather darkening right along the hand-cut laced seam edge
  const seamRimDarken = smoothstep(float(0.115), float(0.052), abs(flex.signedU)).mul(0.34);

  const kmParchment = kubelkaMunkGlaze(ochre.K, ochre.S, opticalThickness.mul(0.62), umber.rInf);
  const kmCuredMargin = kubelkaMunkGlaze(
    mix(sienna.K, moss.K, float(0.45)),
    mix(sienna.S, moss.S, float(0.45)),
    opticalThickness.mul(1.22),
    umber.rInf.mul(0.48)
  );
  const extSkinRgb = mix(kmParchment, kmCuredMargin, marginMix.mul(0.8));
  // Subtle powdery Bone Calcite mannitol salt bloom on exterior dried flanks
  const extSaltedRgb = mix(extSkinRgb, calcite.rInf.mul(0.92), saltBloom.mul(0.54));
  const extVeinedRgb = mix(
    extSaltedRgb,
    mix(sienna.rInf, umber.rInf, float(0.72)).mul(0.40),
    veinMask.mul(0.62)
  );
  const extFinalAlbedo = mix(extVeinedRgb, umber.rInf.mul(0.32), seamRimDarken);

  // Interior kelp wall (seen when orbiting underneath or peeking through seam): velvety matte honey-ochre lining
  const intFinalAlbedo = mix(
    mix(ochre.rInf, sienna.rInf, float(0.38)),
    umber.rInf.mul(0.48),
    veinMask.mul(0.36).add(marginMix.mul(0.44)).clamp(0.0, 0.88)
  );

  parchmentMat.colorNode = select(frontFacing, extFinalAlbedo, intFinalAlbedo);
  parchmentMat.roughnessNode = select(
    frontFacing,
    clamp(float(0.56).add(heightCenter.mul(0.18)).add(saltBloom.mul(0.24)).sub(veinMask.mul(0.08)), 0.38, 0.94),
    clamp(float(0.58).add(mottleVal.mul(0.2)).add(heightCenter.mul(0.1)), 0.52, 0.86)
  );

  // Physical Beer-Lambert backlit subsurface transmittance from the internal tubular Edison filament
  const toBulb = positionLocal.sub(uniforms.uBulbLocal);
  const clampedY = clamp(toBulb.y, float(-0.22), float(0.24));
  const closestFilamentPt = vec3(
    uniforms.uBulbLocal.x,
    uniforms.uBulbLocal.y.add(clampedY),
    uniforms.uBulbLocal.z
  );
  const bulbDist = length(positionLocal.sub(closestFilamentPt));

  // Rich candlelit chiaroscuro hotspot brightest right around the central Edison belly, deepening toward ribs & hem
  const verticalCore = exp(toBulb.y.mul(toBulb.y).negate().mul(2.45));
  const seamCore = float(0.42).add(float(0.58).mul(exp(abs(flex.signedU).mul(2.5).negate())));
  const irradiance = float(1.0)
    .div(float(0.18).add(bulbDist.mul(bulbDist).mul(2.35)))
    .mul(float(0.24).add(verticalCore.mul(0.96)))
    .mul(seamCore);

  // Spectral Beer-Lambert amber-honey transmittance (veins absorb into deep burnt-sienna rather than flat black!)
  const transRealgar = beerLambertTransmittance(realgar.extinction, opticalThickness.mul(1.08));
  const transOchre = beerLambertTransmittance(ochre.extinction, opticalThickness.mul(0.80));
  const transSienna = beerLambertTransmittance(sienna.extinction, opticalThickness.mul(1.42));
  const baseTransmission = mix(transRealgar, transOchre, float(0.42));
  const veinedTransmission = mix(baseTransmission, transSienna.mul(0.36), veinMask.mul(0.82));

  const saltOcclusion = float(1.0).sub(saltBloom.mul(0.48));
  const topShoulderOcclusion = smoothstep(float(0.015), float(0.2), flex.vNorm);
  const rimFalloff = float(1.0)
    .sub(marginMix.mul(0.8))
    .mul(float(0.26).add(topShoulderOcclusion.mul(0.74)))
    .mul(float(1.0).sub(seamRimDarken.mul(0.65)));

  // Incandescent honey-kaolin center bloom right behind the laced corset slit
  const coreHotMix = clamp(verticalCore.mul(seamCore).mul(0.78), 0.1, 0.78);
  const filamentColor = mix(realgar.rInf, kaolin.rInf, coreHotMix);
  const extBacklight = filamentColor
    .mul(veinedTransmission)
    .mul(irradiance)
    .mul(saltOcclusion)
    .mul(rimFalloff)
    .mul(uniforms.uGlowIntensity)
    .mul(3.35);

  // Interior face glows softly from direct internal tungsten scatter without washing out the Edison bulb
  const intDirectGlow = mix(realgar.rInf, ochre.rInf, float(0.46))
    .mul(irradiance)
    .mul(float(0.38).add(rimFalloff.mul(0.62)))
    .mul(float(1.0).sub(veinMask.mul(0.34)))
    .mul(uniforms.uGlowIntensity)
    .mul(1.05);

  parchmentMat.emissiveNode = select(frontFacing, extBacklight, intDirectGlow);

  // ---------------------------------------------------------------------------
  // 2. Dark Cured Dried-Kombu Ruffled Outer Flanges (Deep Umber-Olive + Salt Bloom)
  // ---------------------------------------------------------------------------
  const frillMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.82,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });

  const frillEdgeU = uv().x; // 0 clamped behind walnut rib, 1 outer crinkled dried edge
  const frillV = uv().y;
  const frillGrain = mineralToothFbm(positionLocal.mul(7.5), float(2.8));
  const frillMicro = mineralToothFbm(positionLocal.mul(22.0), float(2.1));

  // Crisp crinkled dried-kelp micro-normal perturbation (properly flipped for DoubleSide via faceDirection)
  const frillEps = float(0.011);
  const fHx = mineralToothFbm(positionLocal.add(vec3(frillEps, 0.0, 0.0)).mul(11.0), float(2.4));
  const fHy = mineralToothFbm(positionLocal.add(vec3(0.0, frillEps, 0.0)).mul(11.0), float(2.4));
  const fHz = mineralToothFbm(positionLocal.add(vec3(0.0, 0.0, frillEps)).mul(11.0), float(2.4));
  const fHc = mineralToothFbm(positionLocal.mul(11.0), float(2.4));
  const fGrad = vec3(fHx.sub(fHc), fHy.sub(fHc), fHz.sub(fHc)).div(frillEps);
  const fN0 = normalize(normalLocal);
  const fTanGrad = fGrad.sub(fN0.mul(dot(fGrad, fN0)));
  const fPerturbedN = normalize(fN0.sub(fTanGrad.mul(0.028)));
  frillMat.normalNode = transformNormalToView(fPerturbedN).mul(faceDirection);

  // Authentic cured dried-kombu colorway matching Concept F: deep charred umber-olive dusted with ashy salt frost
  const deepKombu = mix(umber.rInf.mul(0.18), moss.rInf.mul(0.20), float(0.32).add(frillGrain.mul(0.24)));
  const driedBarkCrest = mix(umber.rInf.mul(0.36), moss.rInf.mul(0.30), frillMicro);
  const rootWarmTransition = mix(sienna.rInf.mul(0.42), deepKombu, smoothstep(float(0.0), float(0.32), frillEdgeU));
  const frillBaseRgb = mix(rootWarmTransition, driedBarkCrest, frillGrain.mul(0.42));

  // Powdery ashy Bone-Calcite mannitol frost clustered naturally along crinkled exterior kombu ridges
  const frillSalt = smoothstep(float(0.54), float(0.82), frillGrain)
    .mul(smoothstep(float(0.14), float(0.78), frillEdgeU))
    .mul(0.54);
  frillMat.colorNode = mix(frillBaseRgb, calcite.rInf.mul(0.76), frillSalt);
  frillMat.roughnessNode = clamp(float(0.74).add(frillMicro.mul(0.2)).add(frillSalt.mul(0.14)), 0.58, 0.96);

  // Subtle warm amber-sienna edge translucency right where the clamped root meets the glowing bay
  const frillCoreY = exp(positionLocal.y.mul(positionLocal.y).negate().mul(1.8));
  const frillRootGlow = mix(sienna.rInf, realgar.rInf, float(0.55))
    .mul(pow(max(float(0.0), float(1.0).sub(frillEdgeU)), float(2.6)))
    .mul(float(0.25).add(frillCoreY.mul(0.75)))
    .mul(float(0.8).add(sin(frillV.mul(32.0)).mul(0.12)))
    .mul(0.22)
    .mul(uniforms.uGlowIntensity);
  frillMat.emissiveNode = frillRootGlow;

  // ---------------------------------------------------------------------------
  // 3. Steam-Bent Dark-Walnut Ribs (Longitudinal Grain Along Bent Curve + Micro-Pore Normal)
  // ---------------------------------------------------------------------------
  const walnutMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.42,
    metalness: 0.04,
  });

  // Flow wood fibers strictly longitudinally along the curved stave (`uv.y` = meridian arc length!)
  const wUv = uv();
  const woodCoord = vec3(wUv.x.mul(9.0), wUv.y.mul(2.1), positionLocal.x.add(positionLocal.z).mul(3.5));
  const woodWarp = mineralToothFbm(woodCoord, float(1.8));
  const woodPores = mineralToothFbm(vec3(wUv.x.mul(34.0), wUv.y.mul(5.5), float(1.7)), float(1.6));
  const grainBand = sin(wUv.x.mul(26.0).add(woodWarp.mul(5.8)).add(wUv.y.mul(1.8)))
    .mul(0.5)
    .add(0.5);

  const darkWalnut = umber.rInf.mul(0.16);
  const satinWalnut = mix(umber.rInf.mul(0.42), sienna.rInf.mul(0.32), float(0.42));
  walnutMat.colorNode = mix(darkWalnut, satinWalnut, grainBand.mul(0.62).add(woodPores.mul(0.14)));
  walnutMat.roughnessNode = clamp(float(0.34).add(woodPores.mul(0.22)), 0.28, 0.62);

  // Subtle open-pore longitudinal wood grain normal perturbation
  const wN0 = normalize(normalLocal);
  const wBump = woodPores.sub(0.5).mul(0.045);
  const wUp = normalize(cross(vec3(0.0, 1.0, 0.0), wN0).add(vec3(1e-5, 0.0, 0.0)));
  walnutMat.normalNode = transformNormalToView(normalize(wN0.add(wUp.mul(wBump))));

  // ---------------------------------------------------------------------------
  // 4. Heirloom Aged Patinated Brass Hardware (PMREM Metallic Sheen + Lathe Brushing)
  // ---------------------------------------------------------------------------
  const brassMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.32,
    metalness: 0.86,
  });

  // Radial lathe-turned brushed patina + oxidized bronze crevices matching Concept F
  const brassAngle = positionLocal.x.mul(18.0).add(positionLocal.z.mul(18.0));
  const latheTurn = sin(positionLocal.y.mul(260.0)).mul(0.5).add(0.5);
  const brassPatina = mineralToothFbm(positionLocal.mul(8.5), float(2.8));
  const antiqueBronze = mix(ochre.rInf, umber.rInf, float(0.42));
  const agedOxide = umber.rInf.mul(0.22);
  const patinaMask = smoothstep(float(0.26), float(0.76), brassPatina);

  brassMat.colorNode = mix(agedOxide, antiqueBronze, patinaMask.mul(0.78).add(latheTurn.mul(0.1)));
  brassMat.roughnessNode = clamp(
    float(0.24).add(float(1.0).sub(patinaMask).mul(0.26)).add(latheTurn.mul(0.06)),
    0.2,
    0.58
  );

  // Subtle circumferential lathe micro-grooves in normalNode for crisp anisotropic-like studio sheen
  const bN0 = normalize(normalLocal);
  const latheBump = cos(positionLocal.y.mul(260.0).add(brassAngle.mul(0.05))).mul(0.022);
  brassMat.normalNode = transformNormalToView(normalize(bN0.add(vec3(0.0, latheBump, 0.0))));

  // ---------------------------------------------------------------------------
  // 5. Interior Glazed Dark-Ceramic E26 Socket Insulator
  // ---------------------------------------------------------------------------
  const ceramicSocketMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.36,
    metalness: 0.05,
  });
  ceramicSocketMat.colorNode = umber.rInf.mul(0.18);

  // ---------------------------------------------------------------------------
  // 6. Hand-Laced Waxed Hemp Whip-Stitching (COUPLED GPU FLEX + Warm Backlit Edge!)
  // ---------------------------------------------------------------------------
  const lacingMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.74,
    metalness: 0.06,
  });

  // Exact same membrane deformation as parchmentMat -> lacing flexes & ripples IN LOCKSTEP with kelp!
  const laceFlex = buildSharedMembraneFlex(uniforms);
  lacingMat.positionNode = positionLocal.sub(laceFlex.radDir.mul(laceFlex.indentAmt));

  const laceTwineNoise = mineralToothFbm(positionLocal.mul(45.0), float(2.0));
  const laceBase = mix(umber.rInf.mul(0.28), ochre.rInf.mul(0.44), float(0.32).add(laceTwineNoise.mul(0.24)));
  lacingMat.colorNode = laceBase;

  // Warm tungsten bounce light catching the underside/inner rim of the stitches directly in front of the bulb
  const laceToBulb = positionLocal.sub(uniforms.uBulbLocal);
  const laceCoreY = exp(laceToBulb.y.mul(laceToBulb.y).negate().mul(2.4));
  lacingMat.emissiveNode = realgar.rInf
    .mul(laceCoreY)
    .mul(0.22)
    .mul(uniforms.uGlowIntensity);

  // ---------------------------------------------------------------------------
  // 7. 3-Ply Braided Natural Hemp Pendant Cord (Outward Normals + Fibrous Twist)
  // ---------------------------------------------------------------------------
  const hempCordMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.88,
    metalness: 0.0,
  });

  const cordUv = uv();
  const cordPly = sin(cordUv.y.mul(5.8).add(cordUv.x.mul(6.28318))).mul(0.5).add(0.5);
  const cordFiber = mineralToothFbm(vec3(cordUv.x.mul(18.0), cordUv.y.mul(6.5), float(0.7)), float(2.2));
  const hempDark = umber.rInf.mul(0.46);
  const hempLight = mix(kaolin.rInf, ochre.rInf, float(0.28)).mul(0.78);
  hempCordMat.colorNode = mix(hempDark, hempLight, cordPly.mul(0.58).add(cordFiber.mul(0.28)));
  hempCordMat.roughnessNode = clamp(float(0.82).add(cordFiber.mul(0.14)), 0.76, 0.98);

  // ---------------------------------------------------------------------------
  // 8. Vintage Tubular Edison Bulb Glass, Hardware, Tungsten Filaments & Soft Glow Aura
  // ---------------------------------------------------------------------------
  const bulbGlassMat = new THREE.MeshPhysicalNodeMaterial({
    color: new THREE.Color('#f4d8a4'),
    roughness: 0.09,
    metalness: 0.02,
    transparent: true,
    opacity: 0.28,
    ior: 1.5,
    reflectivity: 0.9,
    depthWrite: false,
  });

  const bulbHardwareMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.34,
    metalness: 0.82,
  });
  bulbHardwareMat.colorNode = mix(umber.rInf.mul(0.32), ochre.rInf.mul(0.72), float(0.62));

  const filamentMat = new THREE.MeshBasicNodeMaterial();
  const filamentHot = mix(
    realgar.rInf,
    calcite.rInf,
    clamp(uniforms.uGlowIntensity.mul(0.62), 0.18, 0.92)
  );
  filamentMat.colorNode = filamentHot.mul(float(1.85).add(uniforms.uGlowIntensity.mul(2.15)));

  // Soft translucent volumetric filament glow aura (Fresnel-feathered radial falloff -> zero hard capsule edge!)
  const filamentAuraMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  const auraCoreY = pow(max(float(0.0), float(1.0).sub(abs(positionLocal.y.mul(3.4)))), float(1.6));
  const auraRadialSoft = pow(max(float(0.0), abs(normalView.z)), float(2.4));
  const auraColor = mix(realgar.rInf, kaolin.rInf, float(0.56));
  filamentAuraMat.colorNode = auraColor.mul(float(0.62).mul(uniforms.uGlowIntensity));
  filamentAuraMat.opacityNode = clamp(
    auraCoreY.mul(auraRadialSoft).mul(0.54).mul(uniforms.uGlowIntensity),
    0.0,
    0.8
  );

  return {
    uniforms,
    parchmentTex,
    parchmentMat,
    frillMat,
    walnutMat,
    brassMat,
    ceramicSocketMat,
    lacingMat,
    hempCordMat,
    bulbGlassMat,
    bulbHardwareMat,
    filamentMat,
    filamentAuraMat,
    dispose: () => {
      parchmentTex.dispose();
      parchmentMat.dispose();
      frillMat.dispose();
      walnutMat.dispose();
      brassMat.dispose();
      ceramicSocketMat.dispose();
      lacingMat.dispose();
      hempCordMat.dispose();
      bulbGlassMat.dispose();
      bulbHardwareMat.dispose();
      filamentMat.dispose();
      filamentAuraMat.dispose();
    },
  };
}

export const _kelpMatDeps = { cos, dot, max, pow };
