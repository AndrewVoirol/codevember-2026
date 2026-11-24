import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  attribute,
  clamp,
  cos,
  dot,
  exp,
  float,
  length,
  max,
  min,
  mix,
  normalLocal,
  normalize,
  positionLocal,
  pow,
  reflect,
  sin,
  smoothstep,
  sqrt,
  texture,
  transformNormalToView,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { beerLambertTransmittance, mineralToothFbm } from '../../core/tslMaterials';
import {
  CAVITY_D,
  CAVITY_H,
  CAVITY_W,
  GLASS_PANE_H,
  GLASS_PANE_W,
  GRID_H,
  GRID_W,
} from './constants';
import { type BurrowMaterialsContract } from './geometry';

import { type PebbleSpec } from './sandSim';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export const MAX_RENDER_PEBBLES = 8;

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface BurrowMaterials extends BurrowMaterialsContract {
  oiledWalnut: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  agedBronze: THREE.MeshStandardNodeMaterial;
  beveledGlass: THREE.MeshBasicNodeMaterial;
  sandForeground: THREE.MeshStandardNodeMaterial;
  sandBackwall: THREE.MeshStandardNodeMaterial;
  sandCrestTop: THREE.MeshStandardNodeMaterial;
  beetleChitin: THREE.MeshStandardNodeMaterial;
  beetleAccent: THREE.MeshStandardNodeMaterial;
  uCamLocal: { value: THREE.Vector3 };
  uTime: { value: number };
  /** vec4(u, v, mode [0=excavate, 1=pour], active [0..1]) */
  uHoverUv: { value: THREE.Vector4 };
  /** Updates analytical sub-pixel smooth river-pebble uniforms from `sim.pebbles` (zero allocations). */
  syncPebbles: (pebbles: readonly PebbleSpec[]) => void;
  dispose: () => void;
}

/**
 * Feathered radial softbox luminance card for smooth PMREM reflections on
 * machined satin brass, oiled walnut, and beveled soda-lime glass panes.
 */
function createFeatheredSoftboxTexture(r: number, g: number, b: number): THREE.DataTexture {
  const size = 64;
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

/**
 * Generates a calibrated PMREM studio softbox environment cubemap for Day 27 - Burrow
 * so satin-brass brackets, slotted screws, dark walnut chamfers, and beveled glass edges
 * gleam razor-sharp against the #212529 Chalky Slate gallery ground.
 */
export function createBurrowStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0c1015);

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

  // Overhead-left warm calcite key softbox
  addSoftbox(11.5, 8.8, [-4.8, 5.6, 4.4], 1.0, 0.96, 0.9, 1.92);
  // Right tall studio scrim (catches side screws, brass wrap brackets & glass bevels)
  addSoftbox(8.2, 11.2, [6.2, 2.4, 3.2], 0.99, 0.95, 0.89, 1.72);
  // Rear-top cool slate grazing rim scrim (illuminates rear cross-section & top brass slot!)
  addSoftbox(12.5, 5.8, [-0.8, 5.2, -5.6], 0.91, 0.95, 1.0, 1.65);
  // Rear-left warm studio fill card (for crisp 360-deg back-pane viewing)
  addSoftbox(9.5, 7.8, [-4.6, 2.2, -5.2], 0.98, 0.94, 0.88, 1.38);
  // Overhead zenith canopy (lights top sand crest & tailings craters through open top slot!)
  addSoftbox(9.5, 9.5, [0.0, 7.4, 0.2], 0.97, 0.95, 0.9, 1.08);
  // Subtle warm gallery plinth floor bounce
  addSoftbox(12.0, 12.0, [0.0, -6.4, 0.0], 0.95, 0.91, 0.85, 0.36);

  const renderTarget = pmrem.fromScene(envScene, 0.078);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

/**
 * Scale-normalized 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Produces crisp matte mineral grain or fine brushed metal without macro foil lumps.
 */
const perturbNormal3DFbm = Fn(
  ([baseNorm_in, pos_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const sc = float(scale_in);
    const eps = float(0.16).div(max(float(1.0), sc));

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), sc);
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), sc);
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), sc);
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), sc);

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

/**
 * Analytical studio softbox dome reflection for crisp dielectric beveled glass highlights
 * matching the PMREM softbox arrangement against Chalky Slate (#212529).
 */
const evalGlassStudioDome = Fn(([dir_in]: [TslNode]) => {
  const d = normalize(vec3(dir_in));

  const skyT = clamp(d.y.mul(0.5).add(0.5), 0.0, 1.0);
  const roomBase = mix(vec3(0.015, 0.018, 0.022), vec3(0.042, 0.05, 0.058), skyT);

  // Overhead-left warm key softbox sweep
  const keyDir = normalize(vec3(-0.54, 0.62, 0.56));
  const keyBox = smoothstep(float(0.76), float(0.95), max(float(0.0), dot(d, keyDir))).mul(0.82);

  // Right vertical strip scrim
  const rightDir = normalize(vec3(0.76, 0.3, 0.56));
  const rightBox = smoothstep(float(0.8), float(0.97), max(float(0.0), dot(d, rightDir))).mul(0.64);

  // Rear-top rim strip (visible when orbiting behind the terrarium)
  const rimDir = normalize(vec3(-0.16, 0.62, -0.76));
  const rimBox = smoothstep(float(0.78), float(0.96), max(float(0.0), dot(d, rimDir))).mul(0.66);

  // Overhead subtle studio streak
  const zenith = pow(max(float(0.0), d.y.mul(0.65).add(0.35)), 3.2).mul(0.14);

  return roomBase
    .add(vec3(0.98, 0.94, 0.87).mul(keyBox.add(rightBox)))
    .add(vec3(0.86, 0.93, 0.98).mul(rimBox))
    .add(vec3(0.88, 0.86, 0.82).mul(zenith));
});

/**
 * Builds all WebGPU/TSL materials for Day 27 - Burrow using strictly natural mineral pigments:
 * `burntSienna`, `yellowOchre`, `boneCalcite`, `rawUmber`, `warmKaolin`, `hematiteRust`,
 * `mossUmbrous`, `verdigrisMalachite`.
 */
export function createBurrowMaterials(
  sandColorTex: THREE.DataTexture,
  sandFieldTex: THREE.DataTexture
): BurrowMaterials {
  const uCamLocal = uniform(new THREE.Vector3(0.28, 0.16, 2.95));
  const uTime = uniform(0);
  const uHoverUv = uniform(new THREE.Vector4(0.5, 0.5, 0, 0));

  // Analytical sub-pixel river-pebble contours synced directly from `sim.pebbles` (zero pixel stairs!)
  // uPebbleGeom[k] = vec4(cxGrid, cyGrid, rxGrid, ryGrid)
  // uPebbleMeta[k] = vec4(cosAngle, sinAngle, isQuartz [1 or 0], active [1 or 0])
  const uPebbleGeom = Array.from({ length: MAX_RENDER_PEBBLES }, () =>
    uniform(new THREE.Vector4(-1000, -1000, 10, 7))
  );
  const uPebbleMeta = Array.from({ length: MAX_RENDER_PEBBLES }, () =>
    uniform(new THREE.Vector4(1, 0, 1, 0))
  );

  const syncPebbles = (pebbles: readonly PebbleSpec[]): void => {
    for (let k = 0; k < MAX_RENDER_PEBBLES; k++) {
      if (k < pebbles.length) {
        const p = pebbles[k];
        uPebbleGeom[k].value.set(p.cx, p.cy, p.rx, p.ry);
        uPebbleMeta[k].value.set(
          Math.cos(p.angle),
          Math.sin(p.angle),
          p.pigmentId === 8 ? 1 : 0,
          1
        );
      } else {
        uPebbleGeom[k].value.set(-1000, -1000, 10, 7);
        uPebbleMeta[k].value.set(1, 0, 1, 0);
      }
    }
  };

  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const cKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const cOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const cSienna = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const cUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
  const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const extSienna = vec3(...MINERAL_PALETTE.burntSienna.extinctionRgb);

  // ---------------------------------------------------------------------------
  // 1. Oiled Dark Raw-Umber Walnut Frame Wood
  // ---------------------------------------------------------------------------
  const oiledWalnut = new THREE.MeshStandardNodeMaterial({
    metalness: 0.04,
    roughness: 0.46,
    envMapIntensity: 0.82,
  });
  {
    const p = positionLocal;
    // Directional wood cathedral grain (horizontal along rails, vertical along outer stiles)
    const isStile = smoothstep(float(0.75), float(0.8), abs(p.x));
    const grainCoordH = vec3(p.x.mul(1.8), p.y.mul(13.5), p.z.mul(13.5));
    const grainCoordV = vec3(p.x.mul(13.5), p.y.mul(1.8), p.z.mul(13.5));
    const grainCoord = mix(grainCoordH, grainCoordV, isStile);

    const warp = mineralToothFbm(grainCoord.mul(0.35), float(1.8));
    const ringArg = grainCoord.y.mul(0.85).add(grainCoord.z.mul(0.65)).add(warp.mul(5.2));
    const rings = sin(ringArg).mul(0.5).add(0.5);
    const finePore = mineralToothFbm(grainCoord.add(vec3(3.1, 7.4, 2.2)), float(9.5));

    const walnutDeep = cUmber.mul(0.24);
    const walnutHeart = mix(cUmber.mul(0.56), cSienna.mul(0.38), float(0.28));
    const walnutWarm = mix(walnutDeep, walnutHeart, smoothstep(float(0.22), float(0.82), rings));

    oiledWalnut.colorNode = walnutWarm.mul(float(0.86).add(finePore.mul(0.24)));
    oiledWalnut.roughnessNode = clamp(
      float(0.41).add(float(1.0).sub(rings).mul(0.11)).add(finePore.sub(0.5).mul(0.08)),
      0.34,
      0.58
    );
    oiledWalnut.normalNode = perturbNormal3DFbm(normalLocal, grainCoord, float(6.5), float(0.018));
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Turned Satin Brass (Corner Brackets, Bezels, Screws, Feet)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.21,
    envMapIntensity: 1.28,
  });
  {
    const fineSatin = mineralToothFbm(positionLocal, float(84.0));
    const microGrain = mineralToothFbm(positionLocal.add(vec3(4.2, 1.9, 6.1)), float(142.0));
    const brassAged = vec3(0.6, 0.47, 0.22);
    const brassBright = vec3(0.76, 0.61, 0.31);
    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.24), float(0.78), fineSatin.mul(0.62).add(microGrain.mul(0.38)))
    );
    satinBrass.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(108.0), float(0.004));
    satinBrass.roughnessNode = clamp(
      float(0.2).add(fineSatin.sub(0.5).mul(0.05)).add(microGrain.sub(0.5).mul(0.03)),
      0.16,
      0.28
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Aged Dark Bronze (Screw Slots, Depth Gauge Ticks, Side Breather Screens)
  // ---------------------------------------------------------------------------
  const agedBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.84,
    roughness: 0.32,
    envMapIntensity: 1.15,
  });
  {
    const fineBrush = mineralToothFbm(positionLocal, float(72.0));
    const bronzeDark = vec3(0.056, 0.044, 0.032);
    const bronzeLit = vec3(0.115, 0.088, 0.062);
    agedBronze.colorNode = mix(bronzeDark, bronzeLit, smoothstep(float(0.28), float(0.78), fineBrush));
    agedBronze.roughnessNode = clamp(float(0.3).add(fineBrush.sub(0.5).mul(0.09)), 0.24, 0.42);
  }

  // ---------------------------------------------------------------------------
  // 4. Beveled Soda-Lime Observation Glass Panes (Crisp Fresnel + 45-Deg Bevel Catch)
  // ---------------------------------------------------------------------------
  const beveledGlass = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  {
    const pL = vec3(positionLocal);
    const nL = normalize(vec3(normalLocal));
    // Note: glass meshes sit at z = ±GLASS_CENTER_Z while uCamLocal is in terrarium root space,
    // so compute view ray cleanly via local surface position relative to pane center or approximate
    // using positionLocal (pane offset ±0.076 is tiny vs cam dist ~2.95, and keeping reflection crisp!)
    const vDir = normalize(uCamLocal.sub(pL));
    const ndotv = clamp(abs(dot(nL, vDir)), 0.0, 1.0);
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const fresnel = float(0.032).add(float(0.46).mul(pow(oneMinusV, 4.0)));

    const reflDir = reflect(vDir.negate(), nL);
    const studioRefl = evalGlassStudioDome(reflDir);

    // 45-degree chamfered perimeter bevel mask (abs(nL.z) < 0.92 on beveled perimeter facets!)
    const isBevelFace = float(1.0).sub(smoothstep(float(0.84), float(0.96), abs(nL.z)));

    // Subtle secondary perimeter optical edge highlight near outer glass rim
    const edgeDistX = float(GLASS_PANE_W * 0.5).sub(abs(pL.x));
    const edgeDistY = float(GLASS_PANE_H * 0.5).sub(abs(pL.y));
    const rimDist = min(edgeDistX, edgeDistY);
    const rimCatch = float(1.0).sub(smoothstep(float(0.0), float(0.018), rimDist));

    // Authentic soda-lime glass ferrous-iron verdigris/celadon beveled rim tint
    const glassEdgeTint = mix(cVerdigris, cCeladon, float(0.52));

    // Delicate diagonal studio sheen sweep across flat glass face
    const diagSheen = smoothstep(
      float(0.18),
      float(0.86),
      sin(pL.x.mul(2.1).add(pL.y.mul(2.7)).add(0.45)).mul(0.5).add(0.5)
    ).mul(0.055);

    // Subtle zero-chrome pour aim cue when hovering open top airspace (uHoverUv.z > 0.5)
    const glassUv = vec2(
      pL.x.div(float(CAVITY_W)).add(0.5),
      pL.y.div(float(CAVITY_H)).add(0.5)
    );
    const dHover = vec2(
      glassUv.x.sub(uHoverUv.x).mul(float(CAVITY_W / CAVITY_H)),
      glassUv.y.sub(uHoverUv.y)
    );
    const hoverDist = length(dHover);
    const pourActive = uHoverUv.w.mul(smoothstep(float(0.4), float(0.6), uHoverUv.z));
    const pourRing = float(1.0)
      .sub(smoothstep(float(0.012), float(0.022), abs(hoverDist.sub(0.024))))
      .mul(pourActive)
      .mul(0.32);
    const pourDot = float(1.0)
      .sub(smoothstep(float(0.004), float(0.011), hoverDist))
      .mul(pourActive)
      .mul(0.42);

    const glassCol = studioRefl
      .mul(float(0.75).add(isBevelFace.mul(0.85)))
      .add(glassEdgeTint.mul(isBevelFace.mul(0.42).add(rimCatch.mul(0.22))))
      .add(cCalcite.mul(diagSheen.add(isBevelFace.mul(0.26))))
      .add(cOchre.mul(pourRing.add(pourDot)));

    const glassAlpha = clamp(
      fresnel.mul(0.82).add(isBevelFace.mul(0.52)).add(rimCatch.mul(0.16)).add(pourRing).add(pourDot),
      0.032,
      0.78
    );

    beveledGlass.colorNode = glassCol;
    beveledGlass.opacityNode = glassAlpha;
  }

  // ---------------------------------------------------------------------------
  // 5. Stratified Sand Foreground Cross-Section with Smooth Bilinear Silhouettes,
  //    Sub-Pixel Analytical 3D River Pebbles & 10-Step Parallax Tunnel Walls
  //    Strictly uniform-control-flow safe (`texture(...).level(float(0.0))` everywhere!)
  // ---------------------------------------------------------------------------
  const sandForeground = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.86,
    envMapIntensity: 0.48,
    alphaTest: 0.5,
    side: THREE.FrontSide,
  });
  {
    const baseUv = uv();
    const pLocal = vec3(positionLocal);
    const nFace = normalize(vec3(normalLocal)); // (0,0,+1) front, (0,0,-1) back

    const texelX = float(1.0 / GRID_W);
    const texelY = float(1.0 / GRID_H);

    // Primary surface samples in uniform control flow (BOTH textures use LinearFilter!)
    // colSurf.rgb = smooth stratified sand/backwall pigment; colSurf.a = airborne falling grain mask
    // fldSurf.r = 3x3 smoothed cohesive solid coverage; fldSurf.g = recess depth; fldSurf.b = wall depth; fldSurf.a = PACKED stress ONLY
    const colSurf = texture(sandColorTex, baseUv).level(float(0.0));
    const fldSurf = texture(sandFieldTex, baseUv).level(float(0.0));

    // Smooth solid-boundary gradient for soft curved tunnel lips
    const fL = texture(sandFieldTex, clamp(baseUv.add(vec2(texelX.negate(), 0.0)), 0.001, 0.999))
      .level(float(0.0)).r;
    const fR = texture(sandFieldTex, clamp(baseUv.add(vec2(texelX, 0.0)), 0.001, 0.999))
      .level(float(0.0)).r;
    const fD = texture(sandFieldTex, clamp(baseUv.add(vec2(0.0, texelY.negate())), 0.001, 0.999))
      .level(float(0.0)).r;
    const fU = texture(sandFieldTex, clamp(baseUv.add(vec2(0.0, texelY)), 0.001, 0.999))
      .level(float(0.0)).r;

    const gradX = fR.sub(fL);
    const gradY = fU.sub(fD);

    // Exact sub-pixel analytical river-pebble evaluation (eliminates 100% of 300x220 pixel-art ovals!)
    const evalAnalyticalPebbles = Fn(([uv_in]: [TslNode]) => {
      const gx = vec2(uv_in).x.mul(float(GRID_W));
      const gy = vec2(uv_in).y.mul(float(GRID_H));

      const bestR2 = float(4.0).toVar();
      const bestNx = float(0.0).toVar();
      const bestNy = float(0.0).toVar();
      const bestQuartz = float(0.0).toVar();

      for (let k = 0; k < MAX_RENDER_PEBBLES; k++) {
        const pg = uPebbleGeom[k];
        const pm = uPebbleMeta[k];
        If(pm.w.greaterThan(0.5), () => {
          const dx = gx.sub(pg.x);
          const dy = gy.sub(pg.y);
          const cosA = pm.x;
          const sinA = pm.y;
          const lx = dx.mul(cosA).add(dy.mul(sinA));
          const ly = dx.negate().mul(sinA).add(dy.mul(cosA));
          const warp = float(1.0).add(
            sin(lx.mul(0.28).add(float(k * 1.7)))
              .mul(cos(ly.mul(0.34).sub(float(k))))
              .mul(0.06)
          );
          const rx = max(float(1.0), pg.z);
          const ry = max(float(1.0), pg.w);
          const r2 = lx.mul(lx).div(rx.mul(rx)).add(ly.mul(ly).div(ry.mul(ry))).div(warp);

          If(r2.lessThan(bestR2), () => {
            bestR2.assign(r2);
            // Smooth 3D ellipsoid dome slope in world XY plane
            const nxL = lx.div(rx);
            const nyL = ly.div(ry);
            bestNx.assign(nxL.mul(cosA).sub(nyL.mul(sinA)));
            bestNy.assign(nxL.mul(sinA).add(nyL.mul(cosA)));
            bestQuartz.assign(pm.z);
          });
        });
      }

      return vec4(bestR2, bestNx, bestNy, bestQuartz);
    });

    const pebRes = evalAnalyticalPebbles(baseUv);
    const pebR2 = pebRes.x;
    const pebNx = pebRes.y;
    const pebNy = pebRes.z;
    const pebIsQuartz = pebRes.w;
    const pebMask = float(1.0).sub(smoothstep(float(0.9), float(1.02), pebR2));
    const pebSocketShadow = float(1.0).sub(
      smoothstep(float(0.95), float(1.06), pebR2)
        .mul(float(1.0).sub(smoothstep(float(1.06), float(1.38), pebR2)))
        .mul(0.24)
    );

    // Parallax step ray heading from glass face inward into the 0.136-thick cavity
    const viewDir = normalize(uCamLocal.sub(pLocal));
    const inwardRay = viewDir.negate();
    const zDenom = max(float(0.34), abs(inwardRay.z));
    const parallaxSpanUv = vec2(
      inwardRay.x.div(float(CAVITY_W)),
      inwardRay.y.div(float(CAVITY_H))
    ).mul(float(CAVITY_D).div(zDenom));

    // Smooth surface classification (zero nearest-neighbor staircasing!)
    const isCohesiveSurf = fldSurf.r.greaterThan(0.48);
    const isPebbleSurf = pebR2.lessThan(1.01);
    const isFallingSurf = colSurf.a.greaterThan(0.32).and(isCohesiveSurf.not()).and(isPebbleSurf.not());
    const isSolidSurf = isCohesiveSurf.or(isPebbleSurf).or(isFallingSurf);

    // 10-step 3D interior parallax raymarch wrapped inside TSL Fn() (required for If/Loop/Break stack!)
    // Strictly checks COHESIVE packed walls ONLY (sFld.r > 0.54) — NEVER extrudes falling loose grains!
    const evalParallaxHit = Fn(([uv0_in, span_in, solidFlag_in]: [TslNode, TslNode, TslNode]) => {
      const uv0 = vec2(uv0_in);
      const span = vec2(span_in);
      const hitWallOut = float(0.0).toVar();
      const hitDepthOut = float(1.0).toVar();
      const hitUvOut = uv0.toVar();

      If(float(solidFlag_in).lessThan(0.5), () => {
        Loop(10, ({ i }) => {
          const tStep = float(i).add(0.6).div(21.0);
          const sUv = clamp(uv0.add(span.mul(tStep)), 0.002, 0.998);
          const sFld = texture(sandFieldTex, sUv).level(float(0.0));

          If(sFld.r.greaterThan(0.54), () => {
            hitWallOut.assign(float(1.0));
            hitDepthOut.assign(tStep.mul(2.0));
            hitUvOut.assign(sUv);
            Break();
          });
        });
      });

      return vec4(hitWallOut, hitDepthOut, hitUvOut.x, hitUvOut.y);
    });

    const parallaxRes = evalParallaxHit(
      baseUv,
      parallaxSpanUv,
      isSolidSurf.select(float(1.0), float(0.0))
    );
    const hitWall = parallaxRes.x;
    const hitDepth = parallaxRes.y;
    const hitUv = vec2(parallaxRes.z, parallaxRes.w);

    // Sample hit color & 4-tap interior wall gradient ONCE in uniform control flow
    const hitRgb = texture(sandColorTex, hitUv).level(float(0.0)).rgb;
    const hL = texture(sandFieldTex, clamp(hitUv.add(vec2(texelX.negate(), 0.0)), 0.002, 0.998))
      .level(float(0.0)).r;
    const hR = texture(sandFieldTex, clamp(hitUv.add(vec2(texelX, 0.0)), 0.002, 0.998))
      .level(float(0.0)).r;
    const hD = texture(sandFieldTex, clamp(hitUv.add(vec2(0.0, texelY.negate())), 0.002, 0.998))
      .level(float(0.0)).r;
    const hU = texture(sandFieldTex, clamp(hitUv.add(vec2(0.0, texelY)), 0.002, 0.998))
      .level(float(0.0)).r;
    const hitGrad = vec2(hR.sub(hL), hU.sub(hD));

    // Fine Retina-resolution mineral sand grain micro-relief & quartz feldspar glints
    const grainPos = vec3(baseUv.mul(vec2(340.0, 250.0)), pLocal.z.mul(40.0));
    const microGrain = mineralToothFbm(grainPos, float(1.25));
    const fineGrain = mineralToothFbm(grainPos.add(vec3(17.3, 9.1, 4.7)), float(2.85));
    const fallingSparkle = mineralToothFbm(
      grainPos.add(vec3(11.4, 5.2, uTime.mul(2.4))),
      float(2.4)
    );

    // Subtle glass-bezel perimeter AO
    const bezelDist = min(
      min(baseUv.x, float(1.0).sub(baseUv.x)),
      min(baseUv.y, float(1.0).sub(baseUv.y))
    );
    const bezelAo = smoothstep(float(0.0), float(0.024), bezelDist).mul(0.2).add(0.8);

    // Gentle thin-wall depth factor (for soft edge bevel & backlit translucency, ZERO stark black outline!)
    const wallThinness = float(1.0).sub(smoothstep(float(0.06), float(0.62), fldSurf.b));
    const lipShadow = float(1.0).sub(wallThinness.mul(0.08));

    // Arch-failure hairline stress darkening strictly gated on COHESIVE PACKED SAND ONLY (never falling grains!)
    const stressGate = isCohesiveSurf.and(isPebbleSurf.not()).select(float(1.0), float(0.0));
    const stressCrack = smoothstep(float(0.45), float(0.92), fldSurf.a)
      .mul(smoothstep(float(0.32), float(0.72), fineGrain))
      .mul(0.36)
      .mul(stressGate);

    // Crisp bright quartz feldspar glint on sand (extra lively & clean on airborne falling grains!)
    const fallingLift = isFallingSurf.select(float(1.16), float(1.0));
    const grainGlint = pow(clamp(isFallingSurf.select(fallingSparkle, fineGrain), 0.0, 1.0), 4.8)
      .mul(0.24)
      .mul(fallingLift);

    const sandGrainCol = colSurf.rgb
      .mul(float(0.86).add(microGrain.mul(0.28)))
      .mul(bezelAo)
      .mul(lipShadow)
      .mul(pebSocketShadow)
      .mul(float(1.0).sub(stressCrack))
      .mul(fallingLift)
      .add(cCalcite.mul(grainGlint));

    // Sub-pixel smooth analytical river-pebble mineral shading (Quartz vs. Honed Basalt)
    const pebNoiseCoarse = mineralToothFbm(vec3(baseUv.mul(vec2(38.0, 28.0)), 1.7), float(1.4));
    const pebNoiseFine = mineralToothFbm(vec3(baseUv.mul(vec2(145.0, 108.0)), 4.2), float(1.8));
    const pebDomeRim = clamp(sqrt(max(float(0.0), float(1.0).sub(clamp(pebR2, 0.0, 1.0)))), 0.0, 1.0);
    const pebEdgeAo = float(0.68).add(pebDomeRim.mul(0.36));

    // Pale quartz cobble with delicate warm ochre/kaolin veins
    const quartzVein = smoothstep(float(0.68), float(0.84), pebNoiseCoarse);
    const quartzCol = mix(
      mix(cKaolin.mul(0.82), cCalcite.mul(0.94), pebNoiseFine),
      cOchre.mul(0.72),
      quartzVein.mul(0.28)
    ).mul(pebEdgeAo);

    // Dark water-worn basalt cobble with subtle feldspar speckles
    const basaltSpeck = smoothstep(float(0.74), float(0.92), pebNoiseFine);
    const basaltCol = mix(
      cUmber.mul(0.28).add(vec3(0.022, 0.025, 0.03)),
      cKaolin.mul(0.32),
      basaltSpeck.mul(0.35).add(pebNoiseCoarse.mul(0.18))
    ).mul(pebEdgeAo);

    const pebbleCol = mix(basaltCol, quartzCol, pebIsQuartz);
    const solidForegroundCol = mix(sandGrainCol, pebbleCol, pebMask);

    // Softened, sculptural 3D parallax interior tunnel wall shading (ZERO 1-texel black outlines!)
    const depthAo = exp(hitDepth.mul(-1.75)).mul(0.64).add(0.36);
    const roofVsFloor = clamp(hitGrad.y.negate().mul(0.42).add(0.74), 0.48, 1.08);
    const wallGrain = mineralToothFbm(
      vec3(hitUv.mul(vec2(260.0, 190.0)), hitDepth.mul(18.0)),
      float(1.15)
    );
    const parallaxWallCol = hitRgb
      .mul(depthAo)
      .mul(roofVsFloor)
      .mul(float(0.82).add(wallGrain.mul(0.28)));

    const finalAlbedo = isSolidSurf.select(solidForegroundCol, parallaxWallCol);

    // Warm backlit translucent mineral edge scatter on thin sand arches/bridges
    const backlitTrans = beerLambertTransmittance(extSienna, float(0.35).add(fldSurf.b.mul(1.8)));
    const thinEdgeGlow = isCohesiveSurf
      .and(isPebbleSurf.not())
      .select(
        mix(cOchre, cSienna, float(0.45)).mul(backlitTrans).mul(wallThinness.mul(0.2)),
        vec3(0.0, 0.0, 0.0)
      );

    // Subtle zero-chrome excavation cursor reticle on the sand bed when hovering
    const dHover = vec2(
      baseUv.x.sub(uHoverUv.x).mul(float(CAVITY_W / CAVITY_H)),
      baseUv.y.sub(uHoverUv.y)
    );
    const hoverDist = length(dHover);
    const excavateHoverActive = uHoverUv.w.mul(
      float(1.0).sub(smoothstep(float(0.4), float(0.6), uHoverUv.z))
    );
    const brushRingR = float(0.036);
    const brushRing = float(1.0)
      .sub(smoothstep(float(0.0025), float(0.007), abs(hoverDist.sub(brushRingR))))
      .mul(excavateHoverActive)
      .mul(0.26);

    // Smooth normal perturbation: gentle sand-lip bevel + analytical 3D pebble dome + soft parallax wall tilt
    const sandLipX = gradX.negate().mul(0.22);
    const sandLipY = gradY.negate().mul(0.22);
    const pebDomeX = pebNx.mul(0.55);
    const pebDomeY = pebNy.mul(0.55);
    const fgTiltX = mix(sandLipX, pebDomeX, pebMask);
    const fgTiltY = mix(sandLipY, pebDomeY, pebMask);

    const wallTiltX = hitGrad.x.negate().mul(0.36);
    const wallTiltY = hitGrad.y.negate().mul(0.36);

    const macroNormal = normalize(
      nFace.add(
        vec3(
          isSolidSurf.select(fgTiltX, wallTiltX),
          isSolidSurf.select(fgTiltY, wallTiltY),
          0.0
        )
      )
    );
    const bumpStrength = mix(float(0.22), float(0.035), pebMask);

    sandForeground.colorNode = finalAlbedo;
    sandForeground.emissiveNode = thinEdgeGlow.add(cCalcite.mul(brushRing));
    sandForeground.normalNode = perturbNormal3DFbm(
      macroNormal,
      vec3(baseUv.mul(vec2(175.0, 130.0)), pLocal.z.mul(22.0)),
      float(1.55),
      bumpStrength
    );
    sandForeground.roughnessNode = mix(
      clamp(float(0.84).add(microGrain.sub(0.5).mul(0.14)), 0.72, 0.96),
      clamp(float(0.3).add(pebNoiseFine.mul(0.12)), 0.24, 0.44),
      pebMask
    );

    // Cutout: smooth cohesive sand, sub-pixel pebbles, bright falling grains, OR cohesive parallax walls -> 1.0;
    // Open tunnel center & clear top sky -> 0.0 so 3D beetles & recessed backwall show cleanly!
    sandForeground.opacityNode = isSolidSurf.select(float(1.0), hitWall);
  }

  // ---------------------------------------------------------------------------
  // 6. Physically Recessed Shadowed Tunnel Back-Wall (`z = ∓0.054`, behind beetles!)
  //    Shows darkened concave undisturbed strata inside carved burrows; smoothly cuts out open SKY!
  // ---------------------------------------------------------------------------
  const sandBackwall = new THREE.MeshStandardNodeMaterial({
    metalness: 0.01,
    roughness: 0.92,
    envMapIntensity: 0.26,
    alphaTest: 0.5,
    side: THREE.FrontSide,
  });
  {
    const baseUv = uv();
    const pLocal = vec3(positionLocal);
    const nFace = normalize(vec3(normalLocal));

    const colBack = texture(sandColorTex, baseUv).level(float(0.0));
    const fldBack = texture(sandFieldTex, baseUv).level(float(0.0));

    // Smooth non-sky gate: solid strata have fldBack.r > 0.48, subterranean AIR_CAVITY has fldBack.g >= 0.20;
    // open top AIR_SKY has BOTH r = 0 and g = 0 -> smooth bilinear cutout with ZERO jagged voids!
    const nonSkySignal = max(fldBack.r, fldBack.g);
    const isNotSky = nonSkySignal.greaterThan(0.12);

    const texelY = float(1.0 / GRID_H);
    const fUp = texture(sandFieldTex, clamp(baseUv.add(vec2(0.0, texelY.mul(3.0))), 0.002, 0.998))
      .level(float(0.0)).r;
    const fDn = texture(sandFieldTex, clamp(baseUv.add(vec2(0.0, texelY.mul(-3.0))), 0.002, 0.998))
      .level(float(0.0)).r;

    // Deep concave tunnel ambient occlusion: deeper interior = darker, arched ceiling casts deep top shadow
    const recessDepth = fldBack.g;
    const ceilingShadow = float(1.0).sub(fUp.mul(0.38));
    const floorBounce = float(0.86).add(fDn.mul(0.28));
    const cavityDarken = mix(float(0.36), float(0.19), smoothstep(float(0.12), float(0.88), recessDepth))
      .mul(ceilingShadow)
      .mul(floorBounce);

    const backGrain = mineralToothFbm(
      vec3(baseUv.mul(vec2(260.0, 190.0)), pLocal.z.mul(16.0)),
      float(1.15)
    );

    // Subtle warm mineral earth depth tint inside shadowed burrows (never flat black!)
    const deepEarthFloor = cUmber.mul(0.048);
    const shadowedStrata = colBack.rgb
      .mul(cavityDarken)
      .mul(float(0.82).add(backGrain.mul(0.32)))
      .add(deepEarthFloor);

    // Warm faint glow where tunnel back-wall meets thin sunlit walls
    const rimBleed = mix(cOchre, cSienna, float(0.5))
      .mul(float(1.0).sub(smoothstep(float(0.15), float(0.45), recessDepth)))
      .mul(0.052);

    sandBackwall.colorNode = shadowedStrata;
    sandBackwall.emissiveNode = rimBleed;
    sandBackwall.normalNode = perturbNormal3DFbm(
      nFace,
      vec3(baseUv.mul(vec2(145.0, 110.0)), pLocal.z.mul(18.0)),
      float(1.4),
      float(0.24)
    );
    sandBackwall.opacityNode = isNotSky.select(float(1.0), float(0.0));
  }

  // ---------------------------------------------------------------------------
  // 7. Live 3D Top Sand Crest Ribbon (visible through top brass pouring slot!)
  // ---------------------------------------------------------------------------
  const sandCrestTop = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.88,
    envMapIntensity: 0.5,
  });
  {
    const vCol = attribute('color', 'vec3');
    const grain = mineralToothFbm(positionLocal.mul(vec3(180.0, 180.0, 180.0)), float(1.15));
    const sparkle = pow(
      clamp(mineralToothFbm(positionLocal.add(vec3(5.2, 3.1, 8.7)), float(240.0)), 0.0, 1.0),
      5.0
    ).mul(0.18);

    sandCrestTop.colorNode = vCol.mul(float(0.85).add(grain.mul(0.28))).add(cCalcite.mul(sparkle));
    sandCrestTop.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(135.0), float(0.22));
  }

  // ---------------------------------------------------------------------------
  // 8. Articulated Subterranean Burrowing Beetle Materials (Dark Glossy Bronze-Umber Chitin)
  // ---------------------------------------------------------------------------
  const beetleChitin = new THREE.MeshStandardNodeMaterial({
    metalness: 0.42,
    roughness: 0.24,
    envMapIntensity: 1.15,
  });
  {
    const chitinGrain = mineralToothFbm(positionLocal, float(92.0));
    const umberDeep = cUmber.mul(0.22);
    const bronzeSheen = mix(cSienna.mul(0.45), cOchre.mul(0.32), float(0.35));
    beetleChitin.colorNode = mix(umberDeep, bronzeSheen, smoothstep(float(0.25), float(0.82), chitinGrain));
    beetleChitin.roughnessNode = clamp(float(0.22).add(chitinGrain.sub(0.5).mul(0.08)), 0.16, 0.32);
  }

  const beetleAccent = new THREE.MeshStandardNodeMaterial({
    metalness: 0.58,
    roughness: 0.3,
    envMapIntensity: 1.05,
  });
  {
    beetleAccent.colorNode = mix(cUmber.mul(0.42), cOchre.mul(0.52), float(0.35));
  }

  return {
    oiledWalnut,
    satinBrass,
    agedBronze,
    beveledGlass,
    sandForeground,
    sandBackwall,
    sandCrestTop,
    beetleChitin,
    beetleAccent,
    uCamLocal,
    uTime,
    uHoverUv,
    syncPebbles,
    dispose: () => {
      oiledWalnut.dispose();
      satinBrass.dispose();
      agedBronze.dispose();
      beveledGlass.dispose();
      sandForeground.dispose();
      sandBackwall.dispose();
      sandCrestTop.dispose();
      beetleChitin.dispose();
      beetleAccent.dispose();
    },
  };
}
