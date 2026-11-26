import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
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
  silkLoops: THREE.MeshStandardNodeMaterial;
  origamiButterfly: THREE.MeshStandardNodeMaterial;
  emberCore: THREE.MeshBasicNodeMaterial;
  emberMotes: THREE.MeshBasicNodeMaterial;
  dispose: () => void;
}

// ============================================================================
// Procedural Studio Softbox PMREM Environment (7-Softbox IBL)
// Calibrated for oiled dark-umber split bamboo, raw-silk cocoon strands,
// and handmade kozo paper against the canonical Chalky Slate (#212529) ground.
// ============================================================================

function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number,
  size = 64
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
  renderer: THREE.WebGPURenderer
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

  // Warm-neutral studio key softbox (upper-right-front)
  addSoftbox(10.5, 8.5, [4.8, 5.4, 4.4], 1.0, 0.96, 0.88, 2.15);
  // Cool-neutral fill strip (left) for silk sheen & bamboo highlight separation
  addSoftbox(5.6, 10.5, [-5.8, 2.8, 3.2], 0.93, 0.96, 1.0, 1.7);
  // Warm rim kicker (rear-upper) for unspooling silk loops & bamboo ribs
  addSoftbox(11.5, 5.0, [-0.6, 4.8, -5.6], 1.0, 0.94, 0.85, 1.8);
  // Overhead crown diffuser
  addSoftbox(9.5, 9.5, [0.0, 7.2, 0.5], 0.98, 0.95, 0.90, 1.3);
  // Subtle lateral fill cards so oiled bamboo ribs never crush to void
  addSoftbox(13.0, 5.2, [7.0, -0.2, 0.2], 0.88, 0.85, 0.80, 0.46);
  addSoftbox(13.0, 5.2, [-7.0, -0.2, -0.2], 0.84, 0.87, 0.91, 0.44);
  // Warm lower bounce card for underside of flat bamboo mouth hoop & lower silk loops
  addSoftbox(11.0, 11.0, [0.0, -6.0, 0.4], 1.0, 0.86, 0.68, 0.62);

  const renderTarget = pmrem.fromScene(envScene, 0.064);
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
// Deterministic Tileable 2D Noise Helper
// ============================================================================

function tileHash2D(ix: number, iy: number, period: number, salt: number): number {
  const px = ((ix % period) + period) % period;
  const py = ((iy % period) + period) % period;
  const h = Math.sin(px * 127.1 + py * 311.7 + salt * 74.7) * 43758.5453123;
  return h - Math.floor(h);
}

function tileNoise2D(
  x: number,
  y: number,
  size: number,
  periodX: number,
  periodY: number,
  salt: number
): number {
  const fx = (x / size) * periodX;
  const fy = (y / size) * periodY;
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

// ============================================================================
// Baked Seamless 512x512 Handmade Kozo Rice-Paper Texture
// R = fine & medium mulberry cellulose fibers
// G = organic pulp thickness variation (0.08..0.95) for backlit transmission
// B = distinct dark/warm curly bark-bast strands & inclusions (Macro match)
// ============================================================================

function createKozoPaperTexture(size = 512): THREE.DataTexture {
  const fiberBuf = new Float32Array(size * size);
  const pulpBuf = new Float32Array(size * size);
  const bastBuf = new Float32Array(size * size);

  let seed = 20261120;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;
      const fbm =
        tileNoise2D(x, y, size, 5, 5, 1.7) * 0.45 +
        tileNoise2D(x, y, size, 11, 11, 6.3) * 0.28 +
        tileNoise2D(x, y, size, 23, 23, 13.9) * 0.17 +
        tileNoise2D(x, y, size, 47, 47, 29.1) * 0.10;
      const rawPulp = Math.max(0, Math.min(1, (fbm - 0.19) / 0.62));
      const smoothPulp = rawPulp * rawPulp * (3.0 - 2.0 * rawPulp);
      pulpBuf[idx] = 0.08 + smoothPulp * 0.87;
    }
  }

  const deposit = (buf: Float32Array, px: number, py: number, radius: number, amp: number) => {
    const ir = Math.ceil(radius + 1);
    const cx = Math.floor(px);
    const cy = Math.floor(py);
    for (let dy = -ir; dy <= ir; dy++) {
      const yy = (((cy + dy) % size) + size) % size;
      for (let dx = -ir; dx <= ir; dx++) {
        const xx = (((cx + dx) % size) + size) % size;
        const d = Math.hypot(dx - (px - cx), dy - (py - cy));
        if (d < radius) {
          const w = 1 - d / radius;
          const idx = yy * size + xx;
          buf[idx] = Math.min(1.0, buf[idx] + amp * w * w);
        }
      }
    }
  };

  // 580 fine & medium pale mulberry cellulose fibers into R
  for (let f = 0; f < 580; f++) {
    let x = rand() * size;
    let y = rand() * size;
    let ang = rand() * Math.PI * 2;
    const isLong = f < 160;
    const steps = isLong ? 64 : 26;
    const stepLen = isLong ? 3.0 : 2.0;
    const radius = isLong ? 1.45 : 0.92;
    const amp = isLong ? 0.44 : 0.28;
    const curveSign = (rand() - 0.5) * 0.14;

    for (let s = 0; s < steps; s++) {
      const taper = Math.sin((s / Math.max(1, steps - 1)) * Math.PI);
      deposit(fiberBuf, x, y, radius, amp * (0.25 + 0.75 * taper));
      ang += curveSign + (rand() - 0.5) * 0.12;
      x += Math.cos(ang) * stepLen;
      y += Math.sin(ang) * stepLen;
    }
  }

  // 145 distinct curly dark/warm mulberry bark-bast strands into B (Macro hero detail)
  for (let f = 0; f < 145; f++) {
    let x = rand() * size;
    let y = rand() * size;
    let ang = rand() * Math.PI * 2;
    const isFleck = f >= 115;
    const steps = isFleck ? 9 : 40 + Math.floor(rand() * 34);
    const stepLen = isFleck ? 1.4 : 2.3;
    const radius = isFleck ? 1.7 : 1.05 + rand() * 0.65;
    const amp = isFleck ? 0.72 : 0.58;
    const curl = (rand() - 0.5) * 0.22;

    for (let s = 0; s < steps; s++) {
      const taper = Math.sin((s / Math.max(1, steps - 1)) * Math.PI);
      deposit(bastBuf, x, y, radius, amp * (0.30 + 0.70 * taper));
      ang += curl + (rand() - 0.5) * 0.18;
      x += Math.cos(ang) * stepLen;
      y += Math.sin(ang) * stepLen;
    }
  }

  const rgba = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    rgba[i * 4] = Math.min(255, Math.round(fiberBuf[i] * 255));
    rgba[i * 4 + 1] = Math.min(255, Math.round(pulpBuf[i] * 255));
    rgba[i * 4 + 2] = Math.min(255, Math.round(bastBuf[i] * 255));
    rgba[i * 4 + 3] = 255;
  }

  const tex = new THREE.DataTexture(rgba, size, size, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

// ============================================================================
// Baked Seamless 256x256 Structural Fiber Texture
// R = dark roasted/oiled split-bamboo longitudinal vascular fibers
// G = twisted raw-umber suspension cord strands
// B = fine anisotropic raw-silk filament sheen streaks
// ============================================================================

function createStructuralFiberTexture(size = 256): THREE.DataTexture {
  const rgba = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      const u = x / size;

      // R: Split-bamboo vascular striations (longitudinal along V)
      const bambooFine =
        tileNoise2D(x, y, size, 48, 6, 3.1) * 0.55 +
        tileNoise2D(x, y, size, 96, 12, 11.7) * 0.30 +
        tileNoise2D(x, y, size, 16, 4, 21.3) * 0.15;
      const bambooPores = Math.max(0, Math.min(1, (bambooFine - 0.18) / 0.64));

      // G: Twisted hemp/silk suspension cord diagonal strands
      const strandPhase = (u * 8.0 + v * 6.0) * Math.PI * 2.0;
      const cordWave = 0.5 + 0.5 * Math.sin(strandPhase);
      const cordProfile = Math.pow(cordWave, 0.72);
      const cordFuzz = tileNoise2D(x, y, size, 64, 64, 37.9);
      const cordVal = Math.max(0, Math.min(1, cordProfile * 0.76 + cordFuzz * 0.24));

      // B: Fine raw-silk filament sheen micro-striations along strand
      const silkFine =
        tileNoise2D(x, y, size, 6, 64, 49.3) * 0.60 +
        tileNoise2D(x, y, size, 12, 128, 71.9) * 0.40;
      const silkVal = Math.max(0, Math.min(1, (silkFine - 0.15) / 0.70));

      const idx = (y * size + x) * 4;
      rgba[idx] = Math.min(255, Math.round(bambooPores * 255));
      rgba[idx + 1] = Math.min(255, Math.round(cordVal * 255));
      rgba[idx + 2] = Math.min(255, Math.round(silkVal * 255));
      rgba[idx + 3] = 255;
    }
  }

  const tex = new THREE.DataTexture(rgba, size, size, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

// ============================================================================
// Silk-Spun Lantern Chrysalis Material Bundle Factory
// Strictly physical mineral & botanical pigments from src/core/palette.ts
// ============================================================================

export function createSilkLanternMaterials(): SilkMaterialBundle {
  const uTime = uniform(0);
  const uUnravel = uniform(0.55);
  const uBreath = uniform(0.0);
  const uIgnition = uniform(0.82);
  const uEmberPulse = uniform(1.0);
  const uPointerWorld = uniform(new THREE.Vector3(0, 0, 0));
  const uPointerVel = uniform(new THREE.Vector3(0, 0, 0));
  const uFlamePos = uniform(new THREE.Vector3(0, -0.22, 0));

  const kozoPaperTex = createKozoPaperTexture(512);
  const structFiberTex = createStructuralFiberTexture(256);

  // Natural mineral & botanical pigment packs (zero digital neons)
  const pCalcite = createPigmentUniforms('boneCalcite');
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pSienna = createPigmentUniforms('burntSienna');
  const pCinnabar = createPigmentUniforms('cinnabarVermilion');
  const pRealgar = createPigmentUniforms('realgarAmber');

  // ---------------------------------------------------------------------------
  // 1. Rice Paper (THE HERO MATERIAL) — Handmade fibrous kozo lantern canopy
  //    - Rich visible dark/warm mulberry cellulose fibers suspended in cream kozo
  //    - Subtle darkening along aRibProximity bamboo seams
  //    - Gentle outward panel breath driven by uBreath & aWaistMask
  //    - DoubleSide with faceDirection normal flip
  //    - Warm internal realgar-amber / yellow-ochre candled glow from uFlamePos
  // ---------------------------------------------------------------------------
  const ricePaper = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.90,
    clearcoat: 0.0,
    envMapIntensity: 0.62,
  });
  {
    const vUv = uv();
    const aRibProximity = clamp(attribute('aRibProximity', 'float'), 0.0, 1.0);
    const aWaistMask = clamp(attribute('aWaistMask', 'float'), 0.0, 1.0);

    // Subtle organic breathing expansion of paper panels between meridian ribs
    const panelFreedom = float(1.0).sub(aRibProximity.mul(0.82));
    const breathAmp = uBreath.mul(0.032).mul(float(0.32).add(aWaistMask.mul(0.68))).mul(panelFreedom);
    ricePaper.positionNode = positionLocal.add(normalLocal.mul(breathAmp));

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 2.4);

    // Multi-scale kozo fiber & pulp sampling (strictly uniform control flow)
    const paperUvMain = vec2(vUv.x.mul(3.4), vUv.y.mul(2.4));
    const paperUvFine = vec2(
      vUv.x.mul(6.2).add(vUv.y.mul(1.8)),
      vUv.y.mul(5.6).sub(vUv.x.mul(1.6))
    );
    const sampleMain = texture(kozoPaperTex, paperUvMain);
    const sampleDu = texture(kozoPaperTex, paperUvMain.add(vec2(0.0042, 0.0)));
    const sampleDv = texture(kozoPaperTex, paperUvMain.add(vec2(0.0, 0.0042)));
    const sampleFine = texture(kozoPaperTex, paperUvFine);

    const fineFibers = clamp(sampleMain.r.mul(0.70).add(sampleFine.r.mul(0.38)), 0.0, 1.0);
    const pulpCloud = clamp(sampleMain.g.mul(0.78).add(sampleFine.g.mul(0.22)), 0.0, 1.0);
    const darkBast = clamp(sampleMain.b.mul(0.82).add(sampleFine.b.mul(0.34)), 0.0, 1.0);

    // Dry handmade paper micro-normal perturbation (DoubleSide -> faceDirection)
    const grazingAtten = smoothstep(float(0.08), float(0.46), nDotV);
    const dFdU = sampleDu.r.sub(sampleMain.r).mul(2.3).add(sampleDu.b.sub(sampleMain.b).mul(1.4));
    const dFdV = sampleDv.r.sub(sampleMain.r).mul(2.3).add(sampleDv.b.sub(sampleMain.b).mul(1.4));
    const nPaperLocal = normalize(
      normalLocal.add(vec3(dFdU, dFdV, float(0.0)).mul(0.28).mul(grazingAtten))
    );
    const viewNormBase = normalize(transformNormalToView(normalLocal).mul(faceDirection));
    const viewNormPaper = normalize(transformNormalToView(nPaperLocal).mul(faceDirection));
    ricePaper.normalNode = normalize(mix(viewNormBase, viewNormPaper, grazingAtten));

    // Kubelka-Munk 2-flux cream kozo paper reflectance
    const kmThickness = float(0.34)
      .add(pulpCloud.mul(0.44))
      .add(fineFibers.mul(0.22))
      .add(aRibProximity.mul(0.20));
    const paperBacking = mix(
      mix(pOchre.rInf.mul(0.88), pKaolin.rInf.mul(0.96), smoothstep(float(0.08), float(0.82), vUv.y)),
      pCalcite.rInf,
      float(0.42)
    );
    const kmPaper = kubelkaMunkGlaze(pKaolin.K, pKaolin.S, max(float(0.05), kmThickness), paperBacking);

    // Cream parchment crown with distinct warm-umber curly mulberry bast strands
    const crownParchment = mix(kmPaper, pCalcite.rInf.mul(0.97), smoothstep(float(0.48), float(1.0), vUv.y).mul(0.42));
    const withBast = mix(crownParchment, pUmber.rInf.mul(0.56), darkBast.mul(0.48));
    const withFineFibers = mix(withBast, pCalcite.rInf, fineFibers.mul(0.14));
    // Subtle bamboo rib seam darkening & waist wrap ambient occlusion
    const ribShadowed = mix(withFineFibers, pUmber.rInf.mul(0.44), aRibProximity.mul(0.38));
    const waistComp = mix(ribShadowed, ribShadowed.mul(0.92), aWaistMask.mul(float(1.0).sub(clamp(uUnravel, 0.0, 1.0))).mul(0.32));

    const interiorTone = mix(waistComp, mix(pOchre.rInf, pKaolin.rInf, float(0.45)), float(0.30));
    ricePaper.colorNode = select(frontFacing, waistComp, interiorTone);

    ricePaper.roughnessNode = clamp(
      float(0.86).add(pulpCloud.sub(0.5).mul(0.08)).add(fineFibers.mul(0.05)),
      0.80,
      0.96
    );

    // Physical distance-attenuated internal candle-flame subsurface transmission
    const flameVec = positionWorld.sub(uFlamePos);
    const distFlameSq = dot(flameVec, flameVec);
    const flameProximity = exp(distFlameSq.negate().mul(1.18));

    // Brightest golden-amber across lower-mid belly & open mouth, grading smoothly to warm cream crown
    const verticalGlow = float(1.0).sub(smoothstep(float(0.06), float(0.94), vUv.y).mul(0.72));
    // Crisp dark mulberry bast fibers silhouette sharply against the internal amber glow (Macro match!)
    const fiberTrans = clamp(
      float(1.18)
        .sub(pulpCloud.mul(0.38))
        .sub(darkBast.mul(0.72))
        .sub(fineFibers.mul(0.24))
        .sub(aRibProximity.mul(0.42)),
      0.16,
      1.18
    );

    const opticalChord = float(0.14)
      .add(pulpCloud.mul(0.26))
      .add(darkBast.mul(0.42))
      .add(aRibProximity.mul(0.28));
    const blWarmth = beerLambertTransmittance(pOchre.extinction, opticalChord.mul(0.36));

    const coreGlowColor = mix(
      mix(pRealgar.rInf, pCinnabar.rInf, float(0.14)),
      mix(pOchre.rInf, pCalcite.rInf, float(0.44)),
      clamp(flameProximity.mul(verticalGlow), 0.0, 1.0)
    );

    const ignitionLevel = clamp(uIgnition, 0.0, 1.0);
    const litScale = float(0.07).add(ignitionLevel.mul(1.42)).mul(uEmberPulse);
    const sideGain = select(frontFacing, float(1.0), float(1.38));

    const subsurfaceGlow = coreGlowColor
      .mul(blWarmth)
      .mul(float(0.28).add(flameProximity.mul(0.96)))
      .mul(verticalGlow)
      .mul(fiberTrans)
      .mul(litScale)
      .mul(sideGain);

    // Subtle studio rim scatter & tactile pointer ember warmth
    const studioRim = pCalcite.rInf.mul(fresnel).mul(0.08);
    const ptrVec = positionWorld.sub(uPointerWorld);
    const ptrWarmth = pRealgar.rInf
      .mul(exp(dot(ptrVec, ptrVec).negate().mul(2.4)))
      .mul(float(0.05).add(ignitionLevel.mul(0.11)));

    ricePaper.emissiveNode = subsurfaceGlow.add(studioRim).add(ptrWarmth);
  }

  // ---------------------------------------------------------------------------
  // 2. Bamboo Frame — Dark roasted/oiled split-bamboo ribs & flat mouth hoop
  // ---------------------------------------------------------------------------
  const bambooFrame = new THREE.MeshStandardNodeMaterial({
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.46,
    envMapIntensity: 1.05,
  });
  {
    const vUv = uv();
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);

    const bUv = vec2(vUv.x.mul(2.0), vUv.y.mul(5.2));
    const bS = texture(structFiberTex, bUv);
    const bDu = texture(structFiberTex, bUv.add(vec2(0.009, 0.0)));
    const vascular = bS.r;
    const organic = mineralToothFbm(positionLocal, float(5.5));

    // Rich dark roasted/oiled bamboo matching Concept A Hero & Macro
    const darkWood = pUmber.rInf.mul(0.46);
    const warmGrain = mix(pSienna.rInf.mul(0.62), pOchre.rInf.mul(0.76), vascular.mul(0.62));
    const oiledBamboo = mix(darkWood, warmGrain, float(0.32).add(vascular.mul(0.48)))
      .mul(float(0.86).add(organic.mul(0.24)));

    bambooFrame.colorNode = oiledBamboo;
    bambooFrame.roughnessNode = clamp(
      float(0.42).add(float(1.0).sub(vascular).mul(0.20)).add(organic.sub(0.5).mul(0.08)),
      0.34,
      0.64
    );

    const dVdU = bDu.r.sub(vascular).mul(1.8);
    const nBamboo = normalize(normalLocal.add(vec3(dVdU.mul(0.20), float(0.0), dVdU.mul(0.12))));
    bambooFrame.normalNode = normalize(transformNormalToView(nBamboo));

    // Warm interior amber rim-lighting from uFlamePos (matches glowing underside hoop in Macro)
    const toFlame = uFlamePos.sub(positionWorld);
    const distFlame = length(toFlame);
    const dirFlame = toFlame.div(max(float(0.001), distFlame));
    const faceFlame = clamp(dot(nW, dirFlame).mul(0.65).add(0.42), 0.0, 1.0);
    const flameFalloff = exp(distFlame.mul(distFlame).negate().mul(1.65));
    const rimBoost = float(0.52).add(pow(float(1.0).sub(nDotV), 1.8).mul(0.62));

    bambooFrame.emissiveNode = mix(pRealgar.rInf, pOchre.rInf, float(0.52))
      .mul(flameFalloff)
      .mul(faceFlame)
      .mul(rimBoost)
      .mul(clamp(uIgnition, 0.0, 1.0))
      .mul(uEmberPulse)
      .mul(0.64);
  }

  // ---------------------------------------------------------------------------
  // 3. Suspension Cord — Matte dark raw-umber twisted hemp/silk hanging cord
  // ---------------------------------------------------------------------------
  const suspensionCord = new THREE.MeshStandardNodeMaterial({
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.90,
    envMapIntensity: 0.55,
  });
  {
    const vUv = uv();
    const cUv = vec2(vUv.x.mul(2.5), vUv.y.mul(14.0));
    const cS = texture(structFiberTex, cUv);
    const cDu = texture(structFiberTex, cUv.add(vec2(0.012, 0.0)));
    const cDv = texture(structFiberTex, cUv.add(vec2(0.0, 0.012)));
    const cord = cS.g;

    const deepCord = pUmber.rInf.mul(0.26);
    const fiberCatch = mix(pUmber.rInf.mul(0.56), pKaolin.rInf.mul(0.42), float(0.32));
    suspensionCord.colorNode = mix(deepCord, fiberCatch, cord);
    suspensionCord.roughnessNode = clamp(float(0.86).add(float(1.0).sub(cord).mul(0.10)), 0.80, 0.96);

    const dCdU = cDu.g.sub(cord).mul(2.6);
    const dCdV = cDv.g.sub(cord).mul(2.6);
    const nCord = normalize(normalLocal.add(vec3(dCdU.mul(0.28), dCdV.mul(0.28), float(0.0))));
    suspensionCord.normalNode = normalize(transformNormalToView(nCord));
  }

  // ---------------------------------------------------------------------------
  // 4. Silk Wrap — Golden-ochre & warm-kaolin raw-silk cocoon waist bands
  //    - Anisotropic silky sheen + strong backlit golden-amber core glow
  //    - Subtle GPU vertex loosening driven by uUnravel + aUnwindOrder
  // ---------------------------------------------------------------------------
  const silkWrap = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.34,
    clearcoat: 0.18,
    clearcoatRoughness: 0.42,
    envMapIntensity: 1.15,
  });
  {
    const vUv = uv();
    const aStrandT = clamp(attribute('aStrandT', 'float'), 0.0, 1.0);
    const aStrandSeed = clamp(attribute('aStrandSeed', 'float'), 0.0, 1.0);
    const aUnwindOrder = clamp(attribute('aUnwindOrder', 'float'), 0.0, 1.0);

    // Outer/lower strands loosen and peel slightly away from the waist as uUnravel rises
    const unravelClamped = clamp(uUnravel, 0.0, 1.0);
    const unwindGate = smoothstep(
      aUnwindOrder.mul(0.62),
      aUnwindOrder.mul(0.62).add(0.38),
      unravelClamped
    );
    const archEnvelope = sin(aStrandT.mul(3.14159265));
    const radialXZ = normalize(vec3(positionLocal.x, float(0.0), positionLocal.z).add(vec3(1e-5, 0.0, 0.0)));
    const flutter = sin(aStrandT.mul(6.28318).add(uTime.mul(1.5)).add(aStrandSeed.mul(19.3)))
      .mul(0.012)
      .mul(unwindGate);
    const peelDist = unwindGate.mul(0.042).add(uBreath.mul(0.024)).mul(archEnvelope);
    const dropY = unwindGate.mul(0.016).mul(archEnvelope).negate();
    silkWrap.positionNode = positionLocal
      .add(radialXZ.mul(peelDist.add(flutter)))
      .add(vec3(float(0.0), dropY, float(0.0)));

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 2.1);

    const sUv = vec2(vUv.x.mul(2.0), vUv.y.mul(8.0));
    const sTex = texture(structFiberTex, sUv);
    const silkGrain = sTex.b;

    // Raw-silk filament tone variation across individual strands (warm-kaolin to golden-ochre)
    const strandBase = mix(
      mix(pOchre.rInf.mul(0.95), pKaolin.rInf.mul(0.96), aStrandSeed.mul(0.58)),
      pRealgar.rInf.mul(0.92),
      float(0.18).add(silkGrain.mul(0.16))
    );
    const silkySheen = pCalcite.rInf.mul(pow(nDotV, 3.0).mul(0.16).add(fresnel.mul(0.22)));
    silkWrap.colorNode = strandBase.mul(float(0.86).add(silkGrain.mul(0.24))).add(silkySheen);
    silkWrap.roughnessNode = clamp(float(0.32).add(float(1.0).sub(silkGrain).mul(0.16)), 0.24, 0.50);

    // DoubleSide normal flip
    silkWrap.normalNode = normalize(transformNormalToView(normalLocal).mul(faceDirection));

    // Strong translucent backlit golden-amber glow wherever the lantern core shines behind the strands
    const flameVec = positionWorld.sub(uFlamePos);
    const flameAtten = exp(dot(flameVec, flameVec).negate().mul(1.25));
    const backlitGlow = mix(pOchre.rInf, pRealgar.rInf, float(0.46))
      .mul(float(0.32).add(flameAtten.mul(0.92)))
      .mul(float(0.65).add(fresnel.mul(0.55)))
      .mul(float(0.10).add(clamp(uIgnition, 0.0, 1.0).mul(0.98)))
      .mul(uEmberPulse);

    silkWrap.emissiveNode = backlitGlow;
  }

  // ---------------------------------------------------------------------------
  // 5. Silk Loops — Unspooling gossamer silk loop strands with GPU billow/peel
  //    - Tuck close to lantern waist when uUnravel -> 0 (bound cocoon)
  //    - Unfurl into airy glowing arcs as uUnravel rises, swaying in pointer sweeps
  // ---------------------------------------------------------------------------
  const silkLoops = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.36,
    envMapIntensity: 1.15,
  });
  {
    const aLoopFreedom = clamp(attribute('aLoopFreedom', 'float'), 0.0, 1.0);
    const aLoopSeed = clamp(attribute('aLoopSeed', 'float'), 0.0, 1.0);
    const vUv = uv();

    const unravelClamped = clamp(uUnravel, 0.0, 1.0);
    const unfurl = smoothstep(float(0.02), float(0.92), unravelClamped);

    // Smoothly tuck protruding outer loop arcs against the waist when bound (uUnravel -> 0)
    const radialXZLen = length(vec2(positionLocal.x, positionLocal.z));
    const radialXZ = normalize(vec3(positionLocal.x, float(0.0), positionLocal.z).add(vec3(1e-5, 0.0, 0.0)));
    const excessRadius = max(float(0.0), radialXZLen.sub(0.54));
    const tuckInward = radialXZ.mul(excessRadius.mul(float(1.0).sub(unfurl)).mul(aLoopFreedom).mul(0.86)).negate();

    // Gentle buoyant outward bloom & organic air-current billow when unfurled
    const wavePhase = uTime.mul(float(1.45).add(aLoopSeed.mul(0.75))).add(aLoopSeed.mul(24.7)).add(aLoopFreedom.mul(3.4));
    const tangentXZ = vec3(radialXZ.z.negate(), float(0.0), radialXZ.x);
    const ambientSway = radialXZ
      .mul(sin(wavePhase).mul(0.032).add(unfurl.mul(0.028)))
      .add(tangentXZ.mul(cos(wavePhase.mul(0.82)).mul(0.036)))
      .add(vec3(float(0.0), sin(wavePhase.mul(0.68).add(1.3)).mul(0.026), float(0.0)))
      .mul(aLoopFreedom)
      .mul(float(0.14).add(unfurl.mul(0.86)));

    // Pointer-velocity wind displacement gated by proximity to uPointerWorld
    const worldPosOrig = modelWorldMatrix.mul(vec4(positionLocal, float(1.0))).xyz;
    const ptrVec = worldPosOrig.sub(uPointerWorld);
    const ptrInfluence = exp(dot(ptrVec, ptrVec).negate().mul(1.15)).mul(aLoopFreedom).mul(float(0.25).add(unfurl.mul(0.75)));
    const clampedVel = clamp(uPointerVel, vec3(-2.5, -2.5, -2.5), vec3(2.5, 2.5, 2.5));
    const pointerBillow = clampedVel.mul(0.055).mul(ptrInfluence);

    silkLoops.positionNode = positionLocal.add(tuckInward).add(ambientSway).add(pointerBillow);

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 1.9);

    const lTex = texture(structFiberTex, vec2(vUv.x.mul(2.0), vUv.y.mul(6.0)));
    const strandTone = mix(pOchre.rInf, pKaolin.rInf, aLoopSeed.mul(0.55).add(lTex.b.mul(0.20)));
    silkLoops.colorNode = strandTone.add(pCalcite.rInf.mul(fresnel.mul(0.28)));
    silkLoops.roughnessNode = clamp(float(0.34).add(float(1.0).sub(lTex.b).mul(0.16)), 0.26, 0.52);

    // DoubleSide normal flip
    silkLoops.normalNode = normalize(transformNormalToView(normalLocal).mul(faceDirection));

    // Luminous gossamer catchlight + warm amber lantern scatter along unfurled arcs
    const flameDistSq = dot(positionWorld.sub(uFlamePos), positionWorld.sub(uFlamePos));
    const flameScatter = exp(flameDistSq.negate().mul(0.85));
    const loopGlow = mix(pOchre.rInf, pRealgar.rInf, float(0.42))
      .mul(float(0.28).add(flameScatter.mul(0.78)).add(fresnel.mul(0.42)))
      .mul(float(0.22).add(clamp(uIgnition, 0.0, 1.0).mul(0.86)))
      .mul(uEmberPulse);

    silkLoops.emissiveNode = loopGlow;
  }

  // ---------------------------------------------------------------------------
  // 6. Origami Butterfly — Translucent glowing folded ember paper
  // ---------------------------------------------------------------------------
  const origamiButterfly = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.56,
    envMapIntensity: 0.85,
  });
  {
    const vUv = uv();
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = pow(float(1.0).sub(nDotV), 1.85);

    const bPaper = texture(kozoPaperTex, vUv.mul(2.6));
    const wingSpan = clamp(vUv.x, 0.0, 1.0);
    const wingChord = clamp(vUv.y, 0.0, 1.0);

    // Rich cinnabar-vermilion / realgar-amber / yellow-ochre folded paper matching Macro
    const innerCol = mix(pCinnabar.rInf, pRealgar.rInf, float(0.44));
    const outerCol = mix(pRealgar.rInf, pOchre.rInf, float(0.58));
    const wingBase = mix(innerCol, outerCol, smoothstep(float(0.08), float(0.92), wingSpan))
      .mul(float(0.86).add(bPaper.r.mul(0.18)).sub(bPaper.b.mul(0.12)));

    origamiButterfly.colorNode = wingBase;
    origamiButterfly.roughnessNode = clamp(float(0.54).add(bPaper.g.sub(0.5).mul(0.14)), 0.44, 0.72);

    // Preserve crisp non-indexed origami crease facets with DoubleSide faceDirection flip
    origamiButterfly.normalNode = normalize(transformNormalToView(normalLocal).mul(faceDirection));

    // Luminous ember translucency so butterflies glow warmly against #212529
    const creaseRim = mix(pOchre.rInf, pCalcite.rInf, float(0.32)).mul(fresnel).mul(0.42);
    const tipGlow = smoothstep(float(0.12), float(0.95), wingSpan.mul(0.72).add(wingChord.mul(0.28)));
    const emberRadiance = mix(pCinnabar.rInf, pRealgar.rInf, float(0.52).add(tipGlow.mul(0.38)))
      .mul(float(0.56).add(tipGlow.mul(0.46)))
      .mul(float(0.74).add(uEmberPulse.mul(0.26)));

    origamiButterfly.emissiveNode = emberRadiance.add(creaseRim);
  }

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
      mix(pOchre.rInf, pCalcite.rInf, float(0.76)),
      hotNucleus
    );

    const ignitionGate = smoothstep(float(0.02), float(0.36), clamp(uIgnition, 0.0, 1.0));
    emberCore.colorNode = coreCol.mul(1.55).mul(uEmberPulse).mul(flicker);
    emberCore.opacityNode = clamp(softEdge.mul(0.90).mul(ignitionGate), 0.0, 1.0);
  }

  // ---------------------------------------------------------------------------
  // 8. Ember Motes — Soft radial glowing ember/pollen billboard (zero stars!)
  // ---------------------------------------------------------------------------
  const emberMotes = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  {
    const vUv = uv();
    const r = length(vUv.sub(0.5)).mul(2.0);
    const rSq = r.mul(r);
    const halo = exp(rSq.negate().mul(4.6));
    const nucleus = exp(rSq.negate().mul(18.5));
    const edgeCut = smoothstep(float(1.0), float(0.72), r);
    const radialAlpha = clamp(halo.mul(0.48).add(nucleus.mul(0.80)), 0.0, 1.0).mul(edgeCut);

    const moteCol = mix(
      mix(pCinnabar.rInf, pRealgar.rInf, float(0.62)),
      mix(pOchre.rInf, pCalcite.rInf, float(0.60)),
      nucleus
    );

    const moteGate = smoothstep(float(0.06), float(0.42), clamp(uUnravel, 0.0, 1.0))
      .mul(mix(float(0.35), float(1.0), clamp(uIgnition, 0.0, 1.0)));
    emberMotes.colorNode = moteCol.mul(1.25).mul(uEmberPulse);
    emberMotes.opacityNode = radialAlpha.mul(moteGate);
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
    silkLoops,
    origamiButterfly,
    emberCore,
    emberMotes,
    dispose: () => {
      kozoPaperTex.dispose();
      structFiberTex.dispose();
      ricePaper.dispose();
      bambooFrame.dispose();
      suspensionCord.dispose();
      silkWrap.dispose();
      silkLoops.dispose();
      origamiButterfly.dispose();
      emberCore.dispose();
      emberMotes.dispose();
    },
  };
}
