import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  clamp,
  cos,
  dot,
  exp,
  float,
  length,
  max,
  mix,
  normalLocal,
  positionLocal,
  pow,
  sin,
  smoothstep,
  texture,
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

function hash2d(x: number, y: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return h - Math.floor(h);
}

function vNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2d(ix, iy);
  const b = hash2d(ix + 1, iy);
  const c = hash2d(ix, iy + 1);
  const d = hash2d(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function fbm2d(x: number, y: number, octaves = 4): number {
  let sum = 0;
  let amp = 0.5;
  let freq = 1.0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += vNoise(x * freq, y * freq) * amp;
    norm += amp;
    freq *= 2.07;
    amp *= 0.5;
  }
  return sum / Math.max(1e-5, norm);
}

/**
 * Generates a high-resolution (512x512) procedural botanical kelp-parchment texture:
 * - R: crisp branching vascular venation + fine cellular parenchyma web
 * - G: mottled parchment thickness variation
 * - B: powdery Bone-Calcite mannitol salt-mineral bloom near outer panel flanks
 * - A: papery dried-kelp crinkle micro-roughness
 */
export function createKelpParchmentTexture(size = 512): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  const veins = new Float32Array(size * size);

  const rand = seededRand(172026);

  // Rasterize an anti-aliased tapered organic vein segment into `veins`
  const stampVein = (
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    r0: number,
    r1: number,
    strength: number
  ) => {
    const pad = Math.ceil(Math.max(r0, r1) + 2);
    const minX = Math.max(0, Math.floor(Math.min(x0, x1) - pad));
    const maxX = Math.min(size - 1, Math.ceil(Math.max(x0, x1) + pad));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1) - pad));
    const maxY = Math.min(size - 1, Math.ceil(Math.max(y0, y1) + pad));

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
        if (dist < rad + 1.25) {
          const w = Math.max(0, Math.min(1, 1 - (dist - rad * 0.45) / (rad * 0.75 + 1.0)));
          const idx = py * size + px;
          const val = w * w * strength;
          if (val > veins[idx]) veins[idx] = val;
        }
      }
    }
  };

  // Recursive botanical vein growth branching outward from the central spine (u = 0.5)
  const growBranch = (
    x: number,
    y: number,
    angle: number,
    lengthPx: number,
    radius: number,
    depth: number,
    strength: number
  ) => {
    const steps = depth === 0 ? 14 : depth === 1 ? 9 : 6;
    let cx = x;
    let cy = y;
    let curAngle = angle;
    const stepLen = lengthPx / steps;

    for (let s = 0; s < steps; s++) {
      const f0 = s / steps;
      const f1 = (s + 1) / steps;
      curAngle += (rand() - 0.5) * (depth === 0 ? 0.34 : 0.48);
      const nx = cx + Math.cos(curAngle) * stepLen;
      const ny = cy + Math.sin(curAngle) * stepLen;
      const rA = radius * (1 - f0 * 0.58);
      const rB = radius * (1 - f1 * 0.58);
      stampVein(cx, cy, nx, ny, rA, rB, strength * (1 - f0 * 0.28));

      // Fork secondary and tertiary venules
      if (depth < 2 && s > 1 && s < steps - 1 && rand() < (depth === 0 ? 0.46 : 0.36)) {
        const forkSign = rand() < 0.5 ? -1 : 1;
        const forkAngle = curAngle + forkSign * (0.42 + rand() * 0.45);
        growBranch(
          nx,
          ny,
          forkAngle,
          lengthPx * (0.46 + rand() * 0.22),
          rB * 0.76,
          depth + 1,
          strength * 0.78
        );
      }

      cx = nx;
      cy = ny;
      if (cx < 2 || cx > size - 3 || cy < 2 || cy > size - 3) break;
    }
  };

  // Seed major primary veins emerging from the central stitched seam toward left & right ribs
  const primaryCount = 14;
  for (let i = 0; i < primaryCount; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const startY = size * (0.12 + (i / primaryCount) * 0.78 + (rand() - 0.5) * 0.04);
    const startX = size * (0.5 + side * 0.025);
    const baseAngle = side > 0 ? (rand() - 0.5) * 0.65 : Math.PI + (rand() - 0.5) * 0.65;
    const len = size * (0.27 + rand() * 0.18);
    const rad = 1.45 + rand() * 0.85;
    growBranch(startX, startY, baseAngle, len, rad, 0, 0.9);
  }

  // Secondary veins creeping inward from the walnut rib margins
  for (let i = 0; i < 8; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const startY = size * (0.12 + (i / 8) * 0.76);
    const startX = side > 0 ? size * 0.97 : size * 0.03;
    const baseAngle = side > 0 ? Math.PI + (rand() - 0.5) * 0.7 : (rand() - 0.5) * 0.7;
    growBranch(startX, startY, baseAngle, size * 0.22, 1.35, 1, 0.68);
  }

  // Compose final RGBA channels
  for (let py = 0; py < size; py++) {
    const v = py / size;
    for (let px = 0; px < size; px++) {
      const u = px / size;
      const idx = py * size + px;

      // Fine cellular reticulation (organic double-frequency ridge web)
      const nA = fbm2d(u * 26.0 + 1.3, v * 34.0 + 2.7, 3);
      const nB = fbm2d(u * 52.0 + 7.1, v * 68.0 + 4.3, 2);
      const fineWeb = Math.pow(Math.max(0, 1.0 - Math.abs(nA - 0.5) * 5.2), 2.0) * 0.34;
      const veinVal = Math.min(1.0, veins[idx] + fineWeb * 0.65);

      // Organic mottled parchment thickness
      const mottle = fbm2d(u * 6.5 + 3.1, v * 8.2 + 5.7, 4);
      const thicknessNoise = Math.max(0, Math.min(1, mottle * 0.78 + nB * 0.22));

      // Powdery Bone Calcite mannitol salt bloom (stronger near outer rib flanks)
      const edgeBias = Math.min(1.0, Math.pow(Math.abs(u - 0.5) * 2.0, 1.45));
      const saltNoise = fbm2d(u * 11.5 + 13.7, v * 14.0 + 9.2, 4);
      const saltPatch = Math.max(0, (saltNoise - 0.51) * 3.1) * (0.28 + 0.72 * edgeBias);
      const saltVal = Math.min(1.0, saltPatch);

      // Micro-crinkle roughness
      const crinkle = fbm2d(u * 32.0 + 19.3, v * 42.0 + 11.1, 3);

      const o = idx * 4;
      data[o] = Math.round(veinVal * 255);
      data[o + 1] = Math.round(thicknessNoise * 255);
      data[o + 2] = Math.round(saltVal * 255);
      data[o + 3] = Math.round(crinkle * 255);
    }
  }

  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

function createKelpLanternUniforms() {
  return {
    uPokeLocal: uniform(new THREE.Vector3(0, 0, 1.0)),
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
  lacingMat: THREE.MeshStandardNodeMaterial;
  hempCordMat: THREE.MeshStandardNodeMaterial;
  bulbGlassMat: THREE.MeshPhysicalNodeMaterial;
  filamentMat: THREE.MeshBasicNodeMaterial;
  dispose: () => void;
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

  const parchmentTex = createKelpParchmentTexture(512);

  // ---------------------------------------------------------------------------
  // 1. Cured Kelp-Leather & Alginate Parchment Panels (Backlit Subsurface TSL)
  // ---------------------------------------------------------------------------
  const parchmentMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.68,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });

  const bayMeta = attribute('aBayMeta', 'vec4'); // x=signedU, y=vNorm, z=bayFrac, w=thickness
  const signedU = bayMeta.x;
  const vNorm = bayMeta.y;
  const bayFrac = bayMeta.z;
  const baseThickness = bayMeta.w;

  // Pin parchment displacement at walnut ribs (|u|->1) and top brass collar (v->0)
  const ribPin = smoothstep(float(0.98), float(0.45), abs(signedU));
  const collarPin = smoothstep(float(0.03), float(0.18), vNorm);
  const skinFreedom = ribPin.mul(collarPin);

  const dPoke = length(positionLocal.sub(uniforms.uPokeLocal));
  const pokeGauss = exp(dPoke.mul(dPoke).negate().mul(6.2));
  const rippleWave = sin(dPoke.mul(19.0).sub(uniforms.uRipplePhase))
    .mul(exp(dPoke.negate().mul(2.5)))
    .mul(uniforms.uRippleAmp)
    .mul(0.022);

  const indentAmt = pokeGauss.mul(uniforms.uPokeDepth).add(rippleWave).mul(skinFreedom);
  // Push inward along -normalLocal when pressed
  parchmentMat.positionNode = positionLocal.sub(normalLocal.mul(indentAmt));

  // Sample procedural botanical venation, mottle, mannitol salt bloom & micro-crinkle
  const texUv = vec2(uv().x.add(bayFrac.mul(0.37)), uv().y);
  const kelpSample = texture(parchmentTex, texUv);
  const veinMask = kelpSample.r;
  const mottleVal = kelpSample.g;
  const saltBloom = kelpSample.b;
  const crinkle = kelpSample.a;

  // Optical thickness: thicker at veins, margins & mottle; slightly thinner where stretched by poke
  const stretchThinning = clamp(pokeGauss.mul(uniforms.uPokeDepth).mul(1.9), 0.0, 0.24);
  const opticalThickness = clamp(
    baseThickness
      .add(mottleVal.mul(0.28))
      .add(veinMask.mul(0.55))
      .sub(stretchThinning),
    0.12,
    1.85
  );

  // Exterior reflected albedo via 2-flux Kubelka-Munk (Moss Umbrous + Raw Umber margins over Yellow Ochre)
  const marginMix = smoothstep(float(0.55), float(0.96), abs(signedU))
    .add(smoothstep(float(0.84), float(1.0), vNorm))
    .clamp(0.0, 1.0);
  const kmParchment = kubelkaMunkGlaze(
    ochre.K,
    ochre.S,
    opticalThickness.mul(0.65),
    umber.rInf
  );
  const kmDarkMargin = kubelkaMunkGlaze(
    moss.K,
    moss.S,
    opticalThickness.mul(1.1),
    umber.rInf
  );
  const baseSkinRgb = mix(kmParchment, kmDarkMargin, marginMix.mul(0.75));

  // Dust outer panel areas with powdery Bone Calcite mannitol salt mottling
  const saltedSkinRgb = mix(baseSkinRgb, calcite.rInf, saltBloom.mul(0.52));
  // Darken crisp cellular veins on exterior skin
  parchmentMat.colorNode = mix(saltedSkinRgb, umber.rInf.mul(0.45), veinMask.mul(0.62));
  parchmentMat.roughnessNode = clamp(
    float(0.62).add(crinkle.mul(0.24)).add(saltBloom.mul(0.18)),
    0.42,
    0.95
  );

  // Physical Beer-Lambert backlit subsurface transmittance from the internal tubular filament
  const toBulb = positionLocal.sub(uniforms.uBulbLocal);
  // Cylindrical filament axis distance (vertical filament spans y in [-0.20, +0.26])
  const clampedY = clamp(toBulb.y, float(-0.2), float(0.24));
  const closestFilamentPt = vec3(
    uniforms.uBulbLocal.x,
    uniforms.uBulbLocal.y.add(clampedY),
    uniforms.uBulbLocal.z
  );
  const bulbDist = length(positionLocal.sub(closestFilamentPt));
  // Rich candlelit core hotspot that falls off dramatically toward top shoulder, ribs & bottom hem
  const verticalCoreFalloff = exp(toBulb.y.mul(toBulb.y).negate().mul(2.1));
  const seamCoreFalloff = float(0.55).add(float(0.45).mul(exp(abs(signedU).mul(2.1).negate())));
  const irradiance = float(1.0)
    .div(float(0.22).add(bulbDist.mul(bulbDist).mul(2.15)))
    .mul(float(0.32).add(verticalCoreFalloff.mul(0.88)))
    .mul(seamCoreFalloff);

  // Beer-Lambert amber-gold spectral transmittance through cured kelp parchment
  const transAmber = beerLambertTransmittance(realgar.extinction, opticalThickness.mul(1.08));
  const transOchre = beerLambertTransmittance(ochre.extinction, opticalThickness.mul(0.82));
  const warmTransmission = mix(transAmber, transOchre, float(0.38));

  // Crisp silhouette occlusion from dark vascular kelp veins & outer mannitol crust
  const veinTrans = float(1.0).sub(veinMask.mul(0.82));
  const saltOcclusion = float(1.0).sub(saltBloom.mul(0.48));
  const topShoulderOcclusion = smoothstep(float(0.01), float(0.22), vNorm);
  const rimFalloff = float(1.0)
    .sub(marginMix.mul(0.72))
    .mul(float(0.35).add(topShoulderOcclusion.mul(0.65)));

  // Warm candlelit core bloom strongest right behind the center laced seam
  const coreHotMix = clamp(verticalCoreFalloff.mul(seamCoreFalloff).mul(0.65), 0.12, 0.65);
  const filamentColor = mix(realgar.rInf, kaolin.rInf, coreHotMix);
  const backlightRgb = filamentColor
    .mul(warmTransmission)
    .mul(irradiance)
    .mul(veinTrans)
    .mul(saltOcclusion)
    .mul(rimFalloff)
    .mul(uniforms.uGlowIntensity)
    .mul(2.75);

  parchmentMat.emissiveNode = backlightRgb;

  // ---------------------------------------------------------------------------
  // 2. Dark Cured Ruffled Kelp Outer Frills (Moss Umbrous + Raw Umber)
  // ---------------------------------------------------------------------------
  const frillMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.84,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });

  const frillTooth = mineralToothFbm(positionLocal.mul(6.5), float(3.2));
  const frillEdgeU = uv().x; // 0 at rib, 1 at outer deckled lip
  const frillKm = kubelkaMunkGlaze(
    moss.K,
    moss.S,
    float(1.15).add(frillTooth.mul(0.7)),
    umber.rInf
  );
  const frillColor = mix(umber.rInf.mul(0.36), frillKm.mul(0.72), float(0.42).add(frillTooth.mul(0.45)));
  // Subtle salt-mineral dusting along the crinkled dried outer lip
  const frillSalt = smoothstep(float(0.62), float(0.86), frillTooth).mul(frillEdgeU).mul(0.36);
  frillMat.colorNode = mix(frillColor, calcite.rInf.mul(0.62), frillSalt);
  frillMat.roughnessNode = clamp(float(0.78).add(frillTooth.mul(0.18)), 0.62, 0.96);

  // Subtle warm rim bleed only right where the frill meets the glowing bay
  const frillWarmBleed = realgar.rInf
    .mul(pow(max(float(0.0), float(1.0).sub(frillEdgeU)), float(2.4)))
    .mul(0.16)
    .mul(uniforms.uGlowIntensity);
  frillMat.emissiveNode = frillWarmBleed;

  // ---------------------------------------------------------------------------
  // 3. Bent Dark-Walnut Ribs (3D Procedural Wood Grain)
  // ---------------------------------------------------------------------------
  const walnutMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.44,
    metalness: 0.04,
  });
  const woodCoord = vec3(
    positionLocal.x.mul(7.0),
    positionLocal.y.mul(1.8),
    positionLocal.z.mul(7.0)
  );
  const grainNoise = mineralToothFbm(woodCoord, float(2.2));
  const grainRing = sin(length(positionLocal.xz).mul(26.0).add(grainNoise.mul(6.5)))
    .mul(0.5)
    .add(0.5);
  const darkWalnut = umber.rInf.mul(0.24);
  const warmWalnut = mix(umber.rInf.mul(0.52), sienna.rInf.mul(0.35), float(0.45));
  walnutMat.colorNode = mix(darkWalnut, warmWalnut, grainRing.mul(0.68));
  walnutMat.roughnessNode = clamp(float(0.38).add(grainNoise.mul(0.18)), 0.3, 0.62);

  // ---------------------------------------------------------------------------
  // 4. Aged Patinated Brass Hardware (Socket, Knurled Collar, Flange, Rivets)
  // ---------------------------------------------------------------------------
  const brassMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.34,
    metalness: 0.84,
  });
  const brassPatina = mineralToothFbm(positionLocal.mul(6.0), float(3.1));
  const antiqueGold = mix(ochre.rInf, umber.rInf, float(0.38));
  const darkPatina = umber.rInf.mul(0.35);
  brassMat.colorNode = mix(darkPatina, antiqueGold, smoothstep(float(0.25), float(0.78), brassPatina));
  brassMat.roughnessNode = clamp(float(0.26).add(float(1.0).sub(brassPatina).mul(0.28)), 0.22, 0.58);

  // ---------------------------------------------------------------------------
  // 5. Cross-Laced Whip-Stitching Cord
  // ---------------------------------------------------------------------------
  const lacingMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.78,
    metalness: 0.12,
  });
  lacingMat.colorNode = mix(umber.rInf.mul(0.42), ochre.rInf.mul(0.52), float(0.35));

  // ---------------------------------------------------------------------------
  // 6. Braided Hemp Pendant Cord
  // ---------------------------------------------------------------------------
  const hempCordMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.88,
    metalness: 0.0,
  });
  const cordTwist = sin(uv().y.mul(8.5).add(uv().x.mul(6.283))).mul(0.5).add(0.5);
  hempCordMat.colorNode = mix(umber.rInf.mul(0.62), kaolin.rInf.mul(0.76), cordTwist.mul(0.65));

  // ---------------------------------------------------------------------------
  // 7. Internal Edison Bulb Glass & Glowing Filament Core
  // ---------------------------------------------------------------------------
  const bulbGlassMat = new THREE.MeshPhysicalNodeMaterial({
    color: new THREE.Color('#f3d9a8'),
    roughness: 0.12,
    metalness: 0.0,
    transmission: 0.82,
    transparent: true,
    opacity: 0.42,
    ior: 1.48,
    depthWrite: false,
  });

  const filamentMat = new THREE.MeshBasicNodeMaterial();
  const filamentHot = mix(realgar.rInf, calcite.rInf, clamp(uniforms.uGlowIntensity.mul(0.58), 0.15, 0.88));
  filamentMat.colorNode = filamentHot.mul(float(1.25).add(uniforms.uGlowIntensity.mul(1.45)));

  return {
    uniforms,
    parchmentTex,
    parchmentMat,
    frillMat,
    walnutMat,
    brassMat,
    lacingMat,
    hempCordMat,
    bulbGlassMat,
    filamentMat,
    dispose: () => {
      parchmentTex.dispose();
      parchmentMat.dispose();
      frillMat.dispose();
      walnutMat.dispose();
      brassMat.dispose();
      lacingMat.dispose();
      hempCordMat.dispose();
      bulbGlassMat.dispose();
      filamentMat.dispose();
    },
  };
}

// Keep cos/dot/max/pow referenced cleanly if needed by strict linters
export const _kelpMatDeps = { cos, dot, max, pow };
