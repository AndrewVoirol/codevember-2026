import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  dot,
  exp,
  faceDirection,
  float,
  frontFacing,
  length,
  max,
  mix,
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
} from 'three/tsl';
import {
  beerLambertTransmittance,
  createPigmentUniforms,
  kubelkaMunkGlaze,
  valueNoise3DTsl,
} from '../../core/tslMaterials';

export interface KozoStudioEnvHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface KozoMaterialBundle {
  bloomPetalMat: THREE.MeshPhysicalNodeMaterial;
  stamenMat: THREE.MeshStandardNodeMaterial;
  mothMat: THREE.MeshStandardNodeMaterial;
  emberMat: THREE.MeshStandardNodeMaterial;
  pollenMat: THREE.MeshBasicNodeMaterial;
  uniforms: {
    uTime: { value: number };
    uBloom: { value: number };
    uEmberGlow: { value: number };
    uCandleOffset: { value: THREE.Vector2 };
    uClosedShadow: { value: number };
    uHoverLocal: { value: THREE.Vector3 };
    uHoverStrength: { value: number };
  };
  dispose: () => void;
}

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

export function createCocoonStudioEnvironment(
  renderer: THREE.WebGPURenderer
): KozoStudioEnvHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x1e2329);

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

  addSoftbox(10.5, 8.5, [4.6, 5.2, 4.2], 1.0, 0.96, 0.89, 2.0);
  addSoftbox(5.2, 10.5, [-5.8, 2.6, 3.0], 0.94, 0.97, 1.0, 1.75);
  addSoftbox(11.5, 4.8, [-0.8, 5.0, -5.6], 0.98, 0.95, 0.90, 1.65);
  addSoftbox(9.0, 9.0, [0.0, 7.0, 0.4], 0.97, 0.95, 0.90, 1.1);
  addSoftbox(13.5, 5.2, [7.0, -0.4, 0.2], 0.86, 0.84, 0.80, 0.38);
  addSoftbox(13.5, 5.2, [-7.0, -0.4, -0.2], 0.82, 0.86, 0.90, 0.36);
  addSoftbox(11.0, 11.0, [0.0, -6.2, 0.4], 0.98, 0.88, 0.74, 0.52);

  const renderTarget = pmrem.fromScene(envScene, 0.068);
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
// Baked Seamless 512x512 Handmade Rice-Paper / Parchment Texture
// R = 620 crisp mulberry bast & gossamer fibers
// G = high-contrast organic pulp-density clouds (0.06..0.96) for varied sheer opacity
// B = high-frequency anisotropic fiber-pull jaggedness & dry tooth
// ============================================================================

function createRicePaperDataTexture(size = 512): THREE.DataTexture {
  const fiberBuf = new Float32Array(size * size);
  const pulpBuf = new Float32Array(size * size);
  const jaggedBuf = new Float32Array(size * size);

  let seed = 133791;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  // Tileable periodic 2D value-noise FBM for organic handmade pulp clouds (zero grid/checkerboard!)
  const tileHash = (ix: number, iy: number, period: number, salt: number) => {
    const px = ((ix % period) + period) % period;
    const py = ((iy % period) + period) % period;
    const h = Math.sin(px * 127.1 + py * 311.7 + salt * 74.7) * 43758.5453123;
    return h - Math.floor(h);
  };
  const tileNoise = (x: number, y: number, period: number, salt: number) => {
    const fx = (x / size) * period;
    const fy = (y / size) * period;
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const tx = fx - ix;
    const ty = fy - iy;
    const ux = tx * tx * (3 - 2 * tx);
    const uy = ty * ty * (3 - 2 * ty);
    const n00 = tileHash(ix, iy, period, salt);
    const n10 = tileHash(ix + 1, iy, period, salt);
    const n01 = tileHash(ix, iy + 1, period, salt);
    const n11 = tileHash(ix + 1, iy + 1, period, salt);
    const nx0 = n00 + (n10 - n00) * ux;
    const nx1 = n01 + (n11 - n01) * ux;
    return nx0 + (nx1 - nx0) * uy;
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;
      // 4-octave periodic organic pulp clouds (periods 5, 11, 23, 47)
      const fbm =
        tileNoise(x, y, 5, 1.3) * 0.46 +
        tileNoise(x, y, 11, 5.7) * 0.28 +
        tileNoise(x, y, 23, 11.1) * 0.16 +
        tileNoise(x, y, 47, 19.4) * 0.10;
      // Smooth contrast boost around mid-range so sheer windows alternate organically with milkier clouds
      const rawPulp = Math.max(0, Math.min(1, (fbm - 0.18) / 0.64));
      const contrastPulp = rawPulp * rawPulp * (3.0 - 2.0 * rawPulp);
      pulpBuf[idx] = 0.08 + contrastPulp * 0.88;

      const tooth = 0.30 + rand() * 0.42 + tileNoise(x, y, 64, 31.2) * 0.28;
      jaggedBuf[idx] = Math.max(0, Math.min(1, tooth));
    }
  }

  const depositFiber = (px: number, py: number, radius: number, amp: number) => {
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
          fiberBuf[idx] = Math.min(1.0, fiberBuf[idx] + amp * w * w);
        }
      }
    }
  };

  // Stamp 620 sharp, fine varied-length mulberry bast & gossamer fibers
  for (let f = 0; f < 620; f++) {
    let x = rand() * size;
    let y = rand() * size;
    let ang = rand() * Math.PI * 2;
    const isLongBast = f < 160;
    const steps = isLongBast ? 64 : 24;
    const stepLen = isLongBast ? 3.1 : 2.1;
    const radius = isLongBast ? 1.45 : 0.92;
    const amp = isLongBast ? 0.48 : 0.32;
    const curveSign = (rand() - 0.5) * 0.13;

    for (let s = 0; s < steps; s++) {
      const taper = Math.sin((s / (steps - 1)) * Math.PI);
      depositFiber(x, y, radius, amp * (0.25 + 0.75 * taper));
      ang += curveSign + (rand() - 0.5) * 0.10;
      x += Math.cos(ang) * stepLen;
      y += Math.sin(ang) * stepLen;
    }
  }

  const rgba = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    rgba[i * 4] = Math.min(255, Math.round(fiberBuf[i] * 255));
    rgba[i * 4 + 1] = Math.min(255, Math.round(pulpBuf[i] * 255));
    rgba[i * 4 + 2] = Math.min(255, Math.round(jaggedBuf[i] * 255));
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

export function createKozoMaterials(): KozoMaterialBundle {
  const uTime = uniform(0);
  const uBloom = uniform(0.74);
  const uEmberGlow = uniform(1.0);
  const uCandleOffset = uniform(new THREE.Vector2(0, 0));
  const uClosedShadow = uniform(0.0);
  const uHoverLocal = uniform(new THREE.Vector3(0, 0.1, 0.8));
  const uHoverStrength = uniform(0.0);

  const ricePaperTex = createRicePaperDataTexture(512);

  const pCalcite = createPigmentUniforms('boneCalcite');
  const pKaolin = createPigmentUniforms('warmKaolin');
  const pOchre = createPigmentUniforms('yellowOchre');
  const pUmber = createPigmentUniforms('rawUmber');
  const pRealgar = createPigmentUniforms('realgarAmber');
  const pCinnabar = createPigmentUniforms('cinnabarVermilion');
  const pLapis = createPigmentUniforms('lapisUltramarine');
  const pSulfur = createPigmentUniforms('sporeSulfur');

  // ---------------------------------------------------------------------------
  // 1. Razor-Thin Hand-Torn Rice-Paper & Parchment Petal Material
  //    - Spatial pulp-cloud opacity (0.16..0.82) x per-petal opacityBias
  //    - Full-perimeter torn deckle contour + protruding backlit edge whiskers
  //    - Strictly dry matte paper surface (clearcoat 0, roughness 0.86..0.96)
  // ---------------------------------------------------------------------------
  const bloomPetalMat = new THREE.MeshPhysicalNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    metalness: 0.0,
    roughness: 0.90,
    ior: 1.42,
    clearcoat: 0.0,
    envMapIntensity: 0.42,
  });
  {
    const aPetalMeta = attribute('aPetalMeta', 'vec4'); // x=whorlNorm, y=v, z=opacityBias, w=deckleSeed
    const aEdgeMask = attribute('aEdgeMask', 'float');
    const aFiberSeed = attribute('aFiberSeed', 'vec3');
    const vUv = uv();

    const whorlNorm = clamp(aPetalMeta.x, 0.0, 1.0);
    const vCoord = clamp(aPetalMeta.y, 0.0, 1.0);
    const opacityBias = clamp(aPetalMeta.z, 0.55, 1.25);
    const w0Mask = float(1.0).sub(smoothstep(float(0.0), float(0.45), whorlNorm));
    const w2Mask = smoothstep(float(0.55), float(1.0), whorlNorm);

    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const fresnel = float(0.04).add(float(0.96).mul(pow(float(1.0).sub(nDotV), 2.8)));

    const paperUv = vec2(
      aFiberSeed.x.mul(1.25).add(aFiberSeed.z.mul(0.85)),
      aFiberSeed.y.mul(1.35).add(aFiberSeed.x.mul(0.28))
    );
    const paperSample = texture(ricePaperTex, paperUv);
    const paperSampleDu = texture(ricePaperTex, paperUv.add(vec2(0.0040, 0.0)));
    const paperSampleDv = texture(ricePaperTex, paperUv.add(vec2(0.0, 0.0040)));

    const fibers = paperSample.r;
    const pulpCloud = paperSample.g;
    const jaggedTooth = paperSample.b;

    // Delicate botanical longitudinal venation (subtle on sheer Whorl 1 & Whorl 2)
    const sCurve = vUv.x.sub(0.5).mul(2.0);
    const fanCoord = sCurve.mul(float(11.0).sub(vCoord.mul(4.8))).sub(vCoord.mul(1.6));
    const veinPrimary = pow(clamp(float(1.0).sub(abs(sin(fanCoord.mul(3.1))).mul(1.85)), 0.0, 1.0), 3.2);

    // Two-stage hand-torn deckle edge + protruding wispy bast fibers across ALL 3 whorls
    // Gate only Whorl 0 lightly at bloom -> 0 so the closed teardrop cocoon seals completely
    const w0ClosedGate = mix(smoothstep(float(0.02), float(0.20), uBloom), float(1.0), float(1.0).sub(w0Mask));
    const tearScallop = valueNoise3DTsl(aFiberSeed.mul(7.8).add(vec3(1.7, 8.3, 4.9)));
    const tearCutoff = clamp(
      float(0.24).add(tearScallop.mul(0.36)).add(jaggedTooth.mul(0.18)).add(fibers.mul(0.12)),
      0.22,
      0.88
    );

    const sheetMask = smoothstep(tearCutoff.add(0.032), tearCutoff.sub(0.045), aEdgeMask);
    const whiskerMask = smoothstep(tearCutoff.add(0.14), tearCutoff.sub(0.03), aEdgeMask)
      .mul(smoothstep(float(0.32), float(0.78), fibers));
    const rawDeckle = max(sheetMask, whiskerMask.mul(0.85));
    const edgeApply = smoothstep(float(0.05), float(0.32), aEdgeMask).mul(w0ClosedGate);
    const deckleMask = mix(float(1.0), rawDeckle, edgeApply);

    // Bright backlit frayed pulp halo along the torn frontier
    const frayBand = smoothstep(tearCutoff.sub(0.18), tearCutoff.add(0.02), aEdgeMask)
      .mul(rawDeckle)
      .mul(edgeApply);

    // Crisp dry matte paper micro-normal perturbation
    const grazingAtten = smoothstep(float(0.12), float(0.48), nDotV);
    const dFdU = paperSampleDu.r.sub(fibers).mul(2.5).mul(grazingAtten);
    const dFdV = paperSampleDv.r.sub(fibers).mul(2.5).mul(grazingAtten);
    const nPaper = normalize(
      normalLocal.add(vec3(dFdU, dFdV, float(0.0)).mul(mix(float(0.22), float(0.42), w0Mask).add(frayBand.mul(0.22))))
    );
    const viewNormBase = normalize(transformNormalToView(normalLocal).mul(faceDirection));
    const viewNormPaper = normalize(transformNormalToView(nPaper).mul(faceDirection));
    bloomPetalMat.normalNode = normalize(mix(viewNormBase, viewNormPaper, grazingAtten));

    // Archival Bone-Calcite & Warm-Kaolin rice paper with subtle dry-brushed mineral ridge patina on Whorl 0
    const mutedPatina = mix(pUmber.rInf.mul(0.44), pLapis.rInf.mul(0.56), float(0.44));
    const kmThickness = float(0.28)
      .add(pulpCloud.mul(0.44))
      .add(fibers.mul(0.26))
      .mul(mix(float(1.0), float(0.52), whorlNorm));
    const warmBacking = mix(
      mix(pUmber.rInf.mul(0.58), pOchre.rInf.mul(0.84), float(0.50).add(whorlNorm.mul(0.36))),
      pRealgar.rInf.mul(0.86),
      whorlNorm.mul(0.46)
    );
    const kmPaper = kubelkaMunkGlaze(pKaolin.K, pKaolin.S, max(float(0.04), kmThickness), warmBacking);

    const dorsalWashed = mix(
      kmPaper,
      mutedPatina,
      w0Mask.mul(smoothstep(float(0.16), float(0.72), vCoord).mul(0.42)).mul(float(0.42).add(pulpCloud.mul(0.52)))
    );

    const sheerSilkPaper = mix(
      mix(pOchre.rInf.mul(0.86), pKaolin.rInf, smoothstep(float(0.0), float(0.44), vCoord)),
      pCalcite.rInf,
      smoothstep(float(0.40), float(1.0), vCoord).mul(0.80)
    );
    const warmParchment = mix(
      mix(pCinnabar.rInf.mul(0.78), pRealgar.rInf, smoothstep(float(0.0), float(0.44), vCoord)),
      mix(pOchre.rInf, pCalcite.rInf.mul(0.95), smoothstep(float(0.68), float(1.0), vCoord).mul(0.58)),
      smoothstep(float(0.30), float(0.86), vCoord)
    );
    const botanicalBase = mix(sheerSilkPaper, warmParchment, w2Mask);
    const veinedBotanical = mix(botanicalBase, pUmber.rInf.mul(0.64), veinPrimary.mul(0.20));

    const fiberHighlight = pCalcite.rInf.mul(fibers.sub(0.16).mul(0.34));
    const frayRimHighlight = pCalcite.rInf.mul(frayBand.mul(float(0.34).add(fibers.mul(0.36))));

    const exteriorColor = mix(veinedBotanical, dorsalWashed, w0Mask)
      .add(fiberHighlight)
      .add(frayRimHighlight);
    const innerWash = mix(
      mix(pKaolin.rInf.mul(0.94), pOchre.rInf.mul(0.96), float(0.44).add(whorlNorm.mul(0.46))),
      pRealgar.rInf.mul(0.92),
      whorlNorm.mul(0.52)
    );
    const interiorColor = mix(exteriorColor, innerWash, float(0.58))
      .add(fiberHighlight.mul(0.75))
      .add(frayRimHighlight);

    bloomPetalMat.colorNode = select(frontFacing, exteriorColor, interiorColor);

    // Strictly dry matte handmade rice paper (zero plastic gloss!)
    bloomPetalMat.roughnessNode = clamp(
      float(0.86).add(pulpCloud.sub(0.5).mul(0.08)).add(jaggedTooth.sub(0.5).mul(0.08)).add(frayBand.mul(0.08)),
      0.78,
      0.97
    );

    // Dramatic spatial & per-petal varying opacity: sheer rice-paper windows (0.16..0.32)
    // alternating with milkier parchment clouds & bast fibers (0.56..0.82)!
    const spatialDensity = pulpCloud.mul(0.52).add(fibers.mul(0.34)).add(veinPrimary.mul(0.10));
    const sheerBaseAlpha = clamp(
      float(0.16).add(spatialDensity.mul(0.62)).mul(opacityBias),
      0.16,
      0.84
    );
    const closedOpacityLift = w0Mask.mul(uClosedShadow).mul(0.18);
    const whiskerAlpha = whiskerMask.mul(float(0.42).add(fibers.mul(0.36)));
    const frontAlpha = clamp(
      max(
        mix(sheerBaseAlpha.add(closedOpacityLift), float(0.82), fresnel.mul(0.34)),
        whiskerAlpha
      ).add(frayBand.mul(0.14)),
      0.14,
      0.88
    );
    const backAlpha = clamp(frontAlpha.mul(0.90), 0.14, 0.84);
    bloomPetalMat.opacityNode = select(frontFacing, frontAlpha, backAlpha).mul(deckleMask);

    // Closed-state candled silhouette when bloom -> 0
    const pCandle = positionLocal.xy.sub(uCandleOffset.mul(0.22));
    const lobeUpper = exp(
      pCandle.sub(vec2(0.05, 0.20)).mul(vec2(3.6, 2.2)).dot(pCandle.sub(vec2(0.05, 0.20)).mul(vec2(3.6, 2.2))).negate()
    );
    const lobeLeft = exp(
      pCandle.sub(vec2(-0.09, -0.04)).mul(vec2(4.1, 2.6)).dot(pCandle.sub(vec2(-0.09, -0.04)).mul(vec2(4.1, 2.6))).negate()
    );
    const wingSilhouette = clamp(lobeUpper.mul(0.85).add(lobeLeft.mul(0.75)), 0.0, 1.0);
    const candleShade = float(1.0).sub(wingSilhouette.mul(0.66).mul(uClosedShadow));

    // Warm internal subsurface radiance shining brightest through thin sheer rice-paper spots!
    const thinWindow = float(1.18).sub(pulpCloud.mul(0.65));
    const heartVec = positionLocal.sub(vec3(0.0, -0.22, 0.0));
    const distHeartSq = dot(heartVec, heartVec);
    const cavityFalloff = exp(distHeartSq.negate().mul(1.05));
    const chord = mix(float(0.50), float(0.16), whorlNorm).sub(fibers.mul(0.10));
    const blAmber = beerLambertTransmittance(pOchre.extinction, max(float(0.06), chord.mul(0.30)));

    const closedCandleBoost = float(1.0).add(uClosedShadow.mul(0.85));
    const heartGlow = mix(pOchre.rInf, pRealgar.rInf, float(0.44).add(w2Mask.mul(0.30)))
      .mul(blAmber)
      .mul(cavityFalloff)
      .mul(thinWindow)
      .mul(candleShade)
      .mul(closedCandleBoost)
      .mul(uEmberGlow)
      .mul(select(frontFacing, mix(float(0.24), float(0.62), whorlNorm), mix(float(0.48), float(0.78), whorlNorm)));

    const tornRimGlow = mix(pCalcite.rInf, pOchre.rInf, float(0.18))
      .mul(frayBand.add(whiskerMask.mul(0.35)))
      .mul(float(0.28).add(fresnel.mul(0.42)))
      .mul(0.50);
    const softRim = pKaolin.rInf.mul(pow(fresnel, 2.0)).mul(0.14);

    const hoverDistSq = dot(positionLocal.sub(uHoverLocal), positionLocal.sub(uHoverLocal));
    const hoverWarmth = pRealgar.rInf.mul(exp(hoverDistSq.negate().mul(3.2))).mul(uHoverStrength).mul(0.18);

    bloomPetalMat.emissiveNode = heartGlow
      .add(tornRimGlow)
      .add(softRim)
      .add(hoverWarmth)
      .mul(deckleMask);
  }

  // ---------------------------------------------------------------------------
  // 2. Botanical Stamen Crown Material
  // ---------------------------------------------------------------------------
  const stamenMat = new THREE.MeshStandardNodeMaterial({
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.42,
    envMapIntensity: 1.0,
  });
  {
    const aStamenMeta = attribute('aStamenMeta', 'vec2');
    const vSt = clamp(aStamenMeta.x, 0.0, 1.0);
    const isAnther = clamp(aStamenMeta.y, 0.0, 1.0);
    const breath = sin(uTime.mul(1.8)).mul(0.06).add(0.96);

    const stemCol = mix(pRealgar.rInf.mul(0.85), pOchre.rInf, vSt);
    const antherCol = mix(pSulfur.rInf, pRealgar.rInf, float(0.35));

    stamenMat.colorNode = mix(stemCol, antherCol, isAnther);
    stamenMat.emissiveNode = mix(stemCol.mul(0.45), antherCol.mul(0.92), isAnther)
      .mul(uEmberGlow)
      .mul(breath)
      .mul(smoothstep(float(0.08), float(0.45), uBloom));
  }

  // ---------------------------------------------------------------------------
  // 3. Faceted Realgar-Amber Origami Moth Material
  // ---------------------------------------------------------------------------
  const mothMat = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.50,
    envMapIntensity: 1.0,
  });
  {
    const vUv = uv();
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const wingTip = clamp(vUv.x, 0.0, 1.0);
    const paperTooth = texture(ricePaperTex, vUv.mul(2.2));

    const wingBase = mix(
      mix(pRealgar.rInf, pOchre.rInf, float(0.38)),
      pCalcite.rInf.mul(0.96),
      pow(wingTip, 2.0).mul(0.50)
    );
    mothMat.colorNode = wingBase.mul(float(0.90).add(paperTooth.r.mul(0.18)));
    mothMat.emissiveNode = mix(pRealgar.rInf, pOchre.rInf, wingTip.mul(0.55))
      .mul(float(0.46).add(pow(float(1.0).sub(nDotV), 1.6).mul(0.38)))
      .mul(uEmberGlow);
    mothMat.opacityNode = clamp(float(0.95).sub(wingTip.mul(0.08)), 0.80, 0.97);
  }

  // ---------------------------------------------------------------------------
  // 4. Soft Floral Nectar Pearl & 280 Radial-Glow Luminous Pollen/Silk Motes
  // ---------------------------------------------------------------------------
  const emberMat = new THREE.MeshStandardNodeMaterial({
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    metalness: 0.0,
    roughness: 0.36,
    envMapIntensity: 0.9,
  });
  {
    const viewDirW = normalize(cameraPosition.sub(positionWorld));
    const nW = normalize(normalWorld);
    const nDotV = clamp(abs(dot(nW, viewDirW)), 0.0, 1.0);
    const coreWarm = mix(pRealgar.rInf, pOchre.rInf, nDotV.mul(0.55));
    const breath = sin(uTime.mul(1.8)).mul(0.05).add(0.96);

    emberMat.colorNode = mix(pRealgar.rInf, pOchre.rInf, float(0.45));
    emberMat.emissiveNode = coreWarm.mul(float(0.52).add(nDotV.mul(0.36))).mul(uEmberGlow).mul(breath);
  }

  const pollenMat = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  {
    const vUv = uv();
    const r = length(vUv.sub(0.5)).mul(2.0);
    const halo = exp(r.mul(r).negate().mul(4.8));
    const nucleus = exp(r.mul(r).negate().mul(20.0));
    const edgeCut = smoothstep(float(1.0), float(0.72), r);
    const radialAlpha = clamp(halo.mul(0.48).add(nucleus.mul(0.75)), 0.0, 1.0).mul(edgeCut);

    pollenMat.colorNode = mix(pRealgar.rInf, pCalcite.rInf, nucleus.mul(0.65)).mul(1.15);
    pollenMat.opacityNode = radialAlpha.mul(smoothstep(float(0.08), float(0.42), uBloom));
  }

  return {
    bloomPetalMat,
    stamenMat,
    mothMat,
    emberMat,
    pollenMat,
    uniforms: {
      uTime,
      uBloom,
      uEmberGlow,
      uCandleOffset,
      uClosedShadow,
      uHoverLocal,
      uHoverStrength,
    },
    dispose: () => {
      ricePaperTex.dispose();
      bloomPetalMat.dispose();
      stamenMat.dispose();
      mothMat.dispose();
      emberMat.dispose();
      pollenMat.dispose();
    },
  };
}
