import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  attribute,
  clamp,
  dot,
  exp,
  float,
  floor,
  length,
  max,
  min,
  mix,
  normalLocal,
  normalize,
  positionGeometry,
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
  antGaster: THREE.MeshStandardNodeMaterial;
  antEye: THREE.MeshStandardNodeMaterial;
  mineralGrain3D: THREE.MeshStandardNodeMaterial;
  dustMote3D: THREE.MeshBasicNodeMaterial;
  uCamLocal: { value: THREE.Vector3 };
  uTime: { value: number };
  /** vec4(u, v, mode [0=excavate, 1=pour], active [0..1]) */
  uHoverUv: { value: THREE.Vector4 };
  /** Updates analytical sub-pixel organic river-pebble uniforms from `sim.pebbles` (zero allocations). */
  syncPebbles: (pebbles: readonly PebbleSpec[]) => void;
  dispose: () => void;
}

/**
 * Feathered radial softbox luminance card for smooth PMREM reflections on
 * machined satin brass, oiled walnut, polished ant chitin, and beveled glass panes.
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
 * so satin-brass brackets, slotted screws, dark walnut chamfers, beveled glass edges,
 * and glossy worker ant chitin gleam crisply against the #212529 Chalky Slate gallery ground.
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

  addSoftbox(11.5, 8.8, [-4.8, 5.6, 4.4], 1.0, 0.96, 0.9, 1.92);
  addSoftbox(8.2, 11.2, [6.2, 2.4, 3.2], 0.99, 0.95, 0.89, 1.72);
  addSoftbox(12.5, 5.8, [-0.8, 5.2, -5.6], 0.91, 0.95, 1.0, 1.65);
  addSoftbox(9.5, 7.8, [-4.6, 2.2, -5.2], 0.98, 0.94, 0.88, 1.38);
  addSoftbox(9.5, 9.5, [0.0, 7.4, 0.2], 0.97, 0.95, 0.9, 1.08);
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

  const keyDir = normalize(vec3(-0.54, 0.62, 0.56));
  const keyBox = smoothstep(float(0.76), float(0.95), max(float(0.0), dot(d, keyDir))).mul(0.82);

  const rightDir = normalize(vec3(0.76, 0.3, 0.56));
  const rightBox = smoothstep(float(0.8), float(0.97), max(float(0.0), dot(d, rightDir))).mul(0.64);

  const rimDir = normalize(vec3(-0.16, 0.62, -0.76));
  const rimBox = smoothstep(float(0.78), float(0.96), max(float(0.0), dot(d, rimDir))).mul(0.66);

  const zenith = pow(max(float(0.0), d.y.mul(0.65).add(0.35)), 3.2).mul(0.14);

  return roomBase
    .add(vec3(0.98, 0.94, 0.87).mul(keyBox.add(rightBox)))
    .add(vec3(0.86, 0.93, 0.98).mul(rimBox))
    .add(vec3(0.88, 0.86, 0.82).mul(zenith));
});

/**
 * Builds all WebGPU/TSL materials for Day 27 - Burrow using strictly natural mineral pigments:
 * `burntSienna`, `yellowOchre`, `boneCalcite`, `rawUmber`, `warmKaolin`, `hematiteRust`,
 * `realgarAmber`, `mossUmbrous`, `verdigrisMalachite`, `celadonGlauconite`.
 */
export function createBurrowMaterials(
  sandColorTex: THREE.DataTexture,
  sandFieldTex: THREE.DataTexture
): BurrowMaterials {
  const uCamLocal = uniform(new THREE.Vector3(0.28, 0.16, 2.95));
  const uTime = uniform(0);
  const uHoverUv = uniform(new THREE.Vector4(0.5, 0.5, 0, 0));

  // Analytical sub-pixel organic river-pebble contours synced 1:1 with `sim.pebbles`!
  // uPebbleGeom[k]  = vec4(cxGrid, cyGrid, rxGrid, ryGrid)
  // uPebbleMeta[k]  = vec4(cosAngle, sinAngle, isQuartz [1 or 0], active [1 or 0])
  // uPebbleShape[k] = vec4(asym1, asym2, seedOffset, 0)
  const uPebbleGeom = Array.from({ length: MAX_RENDER_PEBBLES }, () =>
    uniform(new THREE.Vector4(-1000, -1000, 10, 7))
  );
  const uPebbleMeta = Array.from({ length: MAX_RENDER_PEBBLES }, () =>
    uniform(new THREE.Vector4(1, 0, 1, 0))
  );
  const uPebbleShape = Array.from({ length: MAX_RENDER_PEBBLES }, () =>
    uniform(new THREE.Vector4(0, 0, 0, 0))
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
        uPebbleShape[k].value.set(p.asym1 ?? 0, p.asym2 ?? 0, k * 2.13, 0);
      } else {
        uPebbleGeom[k].value.set(-1000, -1000, 10, 7);
        uPebbleMeta[k].value.set(1, 0, 1, 0);
        uPebbleShape[k].value.set(0, 0, 0, 0);
      }
    }
  };

  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const cKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const cOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const cSienna = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const cUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const cRust = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear);
  const cAmber = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
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
    const vDir = normalize(uCamLocal.sub(pL));
    const ndotv = clamp(abs(dot(nL, vDir)), 0.0, 1.0);
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const fresnel = float(0.03).add(float(0.46).mul(pow(oneMinusV, 4.0)));

    const reflDir = reflect(vDir.negate(), nL);
    const studioRefl = evalGlassStudioDome(reflDir);

    const isBevelFace = float(1.0).sub(smoothstep(float(0.84), float(0.96), abs(nL.z)));

    const edgeDistX = float(GLASS_PANE_W * 0.5).sub(abs(pL.x));
    const edgeDistY = float(GLASS_PANE_H * 0.5).sub(abs(pL.y));
    const rimDist = min(edgeDistX, edgeDistY);
    const rimCatch = float(1.0).sub(smoothstep(float(0.0), float(0.018), rimDist));

    const glassEdgeTint = mix(cVerdigris, cCeladon, float(0.52));

    const diagSheen = smoothstep(
      float(0.18),
      float(0.86),
      sin(pL.x.mul(2.1).add(pL.y.mul(2.7)).add(0.45)).mul(0.5).add(0.5)
    ).mul(0.05);

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
      .mul(float(0.74).add(isBevelFace.mul(0.86)))
      .add(glassEdgeTint.mul(isBevelFace.mul(0.42).add(rimCatch.mul(0.22))))
      .add(cCalcite.mul(diagSheen.add(isBevelFace.mul(0.25))))
      .add(cOchre.mul(pourRing.add(pourDot)));

    const glassAlpha = clamp(
      fresnel.mul(0.78).add(isBevelFace.mul(0.52)).add(rimCatch.mul(0.16)).add(pourRing).add(pourDot),
      0.028,
      0.78
    );

    beveledGlass.colorNode = glassCol;
    beveledGlass.opacityNode = glassAlpha;
  }

  // ---------------------------------------------------------------------------
  // 5. Stratified Sand Foreground Cross-Section with Smooth Bilinear Silhouettes,
  //    Sub-Pixel Organic 3D River Pebbles, Artifact-Free Granules & 10-Step Parallax Walls
  // ---------------------------------------------------------------------------
  const sandForeground = new THREE.MeshStandardNodeMaterial({
    metalness: 0.02,
    roughness: 0.87,
    envMapIntensity: 0.46,
    alphaTest: 0.5,
    side: THREE.FrontSide,
  });
  {
    const baseUv = uv();
    const pLocal = vec3(positionLocal);
    const nFace = normalize(vec3(normalLocal));

    const texelX = float(1.0 / GRID_W);
    const texelY = float(1.0 / GRID_H);

    // Fine Retina-resolution mineral sand grain micro-relief & silica glints (evaluated first for sub-texel grain dither!)
    const grainPos = vec3(baseUv.mul(vec2(380.0, 280.0)), pLocal.z.mul(40.0));
    const microGrain = mineralToothFbm(grainPos, float(1.35));
    const fineGrain = mineralToothFbm(grainPos.add(vec3(17.3, 9.1, 4.7)), float(3.2));
    const silicaMicro = mineralToothFbm(grainPos.add(vec3(31.7, 14.2, 6.9)), float(5.8));

    // Sub-texel mineral grain UV dither dissolves 300x220 pixel stair-steps into interlocking sand grains at 1:1 macro zoom!
    const grainDitherUv = clamp(
      baseUv.add(
        vec2(
          fineGrain.sub(0.5).mul(texelX.mul(1.35)),
          silicaMicro.sub(0.5).mul(texelY.mul(1.35))
        )
      ),
      0.001,
      0.999
    );

    // Nearest-texel center UV for un-blurred falling grain albedo (eliminates grey/white bilinear static!)
    const snapUv = vec2(
      floor(baseUv.x.mul(float(GRID_W))).add(0.5).div(float(GRID_W)),
      floor(baseUv.y.mul(float(GRID_H))).add(0.5).div(float(GRID_H))
    );

    // All primary texture samples unconditionally in uniform control flow
    const colSurf = texture(sandColorTex, grainDitherUv).level(float(0.0));
    const colSnap = texture(sandColorTex, snapUv).level(float(0.0));
    const fldSurf = texture(sandFieldTex, baseUv).level(float(0.0));

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

    // Exact sub-pixel organic asymmetric river-cobble evaluation (1:1 parity with `sandSim.ts`!)
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
        const ps = uPebbleShape[k];
        If(pm.w.greaterThan(0.5), () => {
          const dx = gx.sub(pg.x);
          const dy = gy.sub(pg.y);
          const cosA = pm.x;
          const sinA = pm.y;
          const lx = dx.mul(cosA).add(dy.mul(sinA));
          const ly = dx.negate().mul(sinA).add(dy.mul(cosA));
          const rx = max(float(1.0), pg.z);
          const ry = max(float(1.0), pg.w);
          const ux = lx.div(rx);
          const uy = ly.div(ry);
          const taper = max(float(0.55), float(1.0).add(ps.x.mul(ux)));
          const tUy = uy.div(taper);
          const r2 = ux
            .mul(ux)
            .add(tUy.mul(tUy))
            .add(ps.y.mul(uy).mul(float(1.0).sub(ux.mul(ux).mul(0.5))));

          If(r2.lessThan(bestR2), () => {
            bestR2.assign(r2);
            const nxL = ux;
            const nyL = tUy;
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
    const pebMask = float(1.0).sub(smoothstep(float(0.92), float(1.02), pebR2));
    // Natural directional underside sand-socket contact shadow (only beneath/around stone base, NOT a dark outline!)
    const socketBand = smoothstep(float(0.98), float(1.08), pebR2).mul(
      float(1.0).sub(smoothstep(float(1.08), float(1.36), pebR2))
    );
    const undersideBias = clamp(pebNy.negate().mul(0.55).add(0.45), 0.15, 1.0);
    const pebSocketShadow = float(1.0).sub(socketBand.mul(undersideBias).mul(0.22));

    // Parallax step ray heading from glass face inward into the 0.136-thick cavity
    const viewDir = normalize(uCamLocal.sub(pLocal));
    const inwardRay = viewDir.negate();
    const zDenom = max(float(0.34), abs(inwardRay.z));
    const parallaxSpanUv = vec2(
      inwardRay.x.div(float(CAVITY_W)),
      inwardRay.y.div(float(CAVITY_H))
    ).mul(float(CAVITY_D).div(zDenom));

    // Smooth cohesive solid surface + crisp, organically faceted airborne sand grains (never erased by bilinear alpha!)
    const isCohesiveSurf = fldSurf.r.greaterThan(0.48);
    const isPebbleSurf = pebR2.lessThan(1.01);
    const cellGridPos = baseUv.mul(vec2(float(GRID_W), float(GRID_H)));
    const cellLocal = cellGridPos.sub(floor(cellGridPos)).sub(0.5);
    const grainContourR = length(cellLocal).add(microGrain.sub(0.5).mul(0.24));
    const isFallingSurf = colSnap.a
      .greaterThan(0.5)
      .and(grainContourR.lessThan(0.47))
      .and(isCohesiveSurf.not())
      .and(isPebbleSurf.not());
    const isSolidSurf = isCohesiveSurf.or(isPebbleSurf).or(isFallingSurf);

    // 10-step 3D interior parallax raymarch strictly on COHESIVE packed walls (never extrudes falling grains!)
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

    // Subtle glass-bezel perimeter AO
    const bezelDist = min(
      min(baseUv.x, float(1.0).sub(baseUv.x)),
      min(baseUv.y, float(1.0).sub(baseUv.y))
    );
    const bezelAo = smoothstep(float(0.0), float(0.024), bezelDist).mul(0.18).add(0.82);

    const wallThinness = float(1.0).sub(smoothstep(float(0.06), float(0.62), fldSurf.b));
    const lipShadow = float(1.0).sub(wallThinness.mul(0.07));

    // Arch-failure hairline stress darkening strictly gated on COHESIVE PACKED SAND ONLY
    const stressGate = isCohesiveSurf.and(isPebbleSurf.not()).select(float(1.0), float(0.0));
    const stressCrack = smoothstep(float(0.45), float(0.92), fldSurf.a)
      .mul(smoothstep(float(0.3), float(0.72), fineGrain))
      .mul(0.38)
      .mul(stressGate);

    // Crisp Retina-scale mineral quartz/feldspar micro-glints on sand (clean & restrained!)
    const grainGlint = pow(clamp(silicaMicro.mul(0.65).add(fineGrain.mul(0.35)), 0.0, 1.0), 5.6).mul(
      0.22
    );

    // Use un-blurred `colSnap.rgb` for falling grains so they match true stratum pigment with zero grey halo!
    const rawSandRgb = isFallingSurf.select(colSnap.rgb, colSurf.rgb);
    const sandGrainCol = rawSandRgb
      .mul(float(0.84).add(microGrain.mul(0.24)).add(silicaMicro.sub(0.5).mul(0.14)))
      .mul(bezelAo)
      .mul(lipShadow)
      .mul(pebSocketShadow)
      .mul(float(1.0).sub(stressCrack))
      .add(cCalcite.mul(grainGlint));

    // Natural water-worn river cobble shading (mineral banding + fine speckle + directional softbox wrap)
    const pebNoiseCoarse = mineralToothFbm(vec3(baseUv.mul(vec2(34.0, 26.0)), 1.7), float(1.4));
    const pebNoiseMed = mineralToothFbm(vec3(baseUv.mul(vec2(78.0, 58.0)), 2.9), float(1.6));
    const pebNoiseFine = mineralToothFbm(vec3(baseUv.mul(vec2(170.0, 128.0)), 4.2), float(1.8));
    const pebDomeRim = clamp(sqrt(max(float(0.0), float(1.0).sub(clamp(pebR2, 0.0, 1.0)))), 0.0, 1.0);
    // Warm top-left studio wrap on cobble crown, soft warm contact grounding under belly (no black egg-ring!)
    const pebTopLight = clamp(pebNy.mul(0.45).sub(pebNx.mul(0.32)).add(0.68), 0.42, 1.12);
    const pebReliefAo = float(0.76).add(pebDomeRim.mul(0.24)).mul(pebTopLight);

    // Layered pale quartzite cobble with delicate warm ochre/sienna mineral seams & fine crystal grain
    const quartzBand = smoothstep(
      float(0.46),
      float(0.76),
      sin(baseUv.x.mul(58.0).add(baseUv.y.mul(82.0)).add(pebNoiseCoarse.mul(5.5))).mul(0.5).add(0.5)
    );
    const quartzCol = mix(
      mix(cKaolin.mul(0.76), cCalcite.mul(0.88), pebNoiseFine),
      mix(cOchre.mul(0.66), cUmber.mul(0.52), pebNoiseMed),
      quartzBand.mul(0.26)
    ).mul(pebReliefAo);

    // Water-worn speckled basalt cobble with subtle pale feldspar phenocrysts & matte mineral grain
    const basaltSpeck = smoothstep(float(0.76), float(0.93), pebNoiseFine);
    const basaltMineral = mix(
      cUmber.mul(0.26).add(vec3(0.022, 0.024, 0.028)),
      cUmber.mul(0.42).add(vec3(0.032, 0.034, 0.038)),
      pebNoiseMed
    );
    const basaltCol = mix(basaltMineral, cKaolin.mul(0.36), basaltSpeck.mul(0.32)).mul(pebReliefAo);

    const pebbleCol = mix(basaltCol, quartzCol, pebIsQuartz);
    const solidForegroundCol = mix(sandGrainCol, pebbleCol, pebMask);

    // Sculptural 3D parallax interior tunnel wall shading with granular striations
    const depthAo = exp(hitDepth.mul(-1.68)).mul(0.64).add(0.36);
    const roofVsFloor = clamp(hitGrad.y.negate().mul(0.42).add(0.76), 0.5, 1.1);
    const wallGrain = mineralToothFbm(
      vec3(hitUv.mul(vec2(280.0, 210.0)), hitDepth.mul(18.0)),
      float(1.25)
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

    const sandLipX = gradX.negate().mul(0.24);
    const sandLipY = gradY.negate().mul(0.24);
    const pebDomeX = pebNx.mul(0.48);
    const pebDomeY = pebNy.mul(0.48);
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
    const bumpStrength = mix(float(0.24), float(0.065), pebMask);

    sandForeground.colorNode = finalAlbedo;
    sandForeground.emissiveNode = thinEdgeGlow.add(cCalcite.mul(brushRing));
    sandForeground.normalNode = perturbNormal3DFbm(
      macroNormal,
      vec3(baseUv.mul(vec2(195.0, 145.0)), pLocal.z.mul(22.0)),
      float(1.65),
      bumpStrength
    );
    sandForeground.roughnessNode = mix(
      clamp(float(0.85).add(microGrain.sub(0.5).mul(0.14)), 0.72, 0.96),
      clamp(float(0.46).add(pebNoiseFine.mul(0.14)), 0.38, 0.62),
      pebMask
    );

    sandForeground.opacityNode = isSolidSurf.select(float(1.0), hitWall);
  }

  // ---------------------------------------------------------------------------
  // 6. Physically Recessed Shadowed Tunnel Back-Wall (`z = ∓0.056`, behind ants!)
  //    Provides rich warm dark-earth tunnel contrast so glossy worker ants pop cleanly!
  // ---------------------------------------------------------------------------
  const sandBackwall = new THREE.MeshStandardNodeMaterial({
    metalness: 0.01,
    roughness: 0.92,
    envMapIntensity: 0.24,
    alphaTest: 0.5,
    side: THREE.FrontSide,
  });
  {
    const baseUv = uv();
    const pLocal = vec3(positionLocal);
    const nFace = normalize(vec3(normalLocal));

    const colBack = texture(sandColorTex, baseUv).level(float(0.0));
    const fldBack = texture(sandFieldTex, baseUv).level(float(0.0));

    const nonSkySignal = max(fldBack.r, fldBack.g);
    const isNotSky = nonSkySignal.greaterThan(0.12);

    const texelY = float(1.0 / GRID_H);
    const fUp = texture(sandFieldTex, clamp(baseUv.add(vec2(0.0, texelY.mul(3.0))), 0.002, 0.998))
      .level(float(0.0)).r;
    const fDn = texture(sandFieldTex, clamp(baseUv.add(vec2(0.0, texelY.mul(-3.0))), 0.002, 0.998))
      .level(float(0.0)).r;

    const recessDepth = fldBack.g;
    const ceilingShadow = float(1.0).sub(fUp.mul(0.36));
    const floorBounce = float(0.88).add(fDn.mul(0.26));
    const cavityDarken = mix(float(0.38), float(0.21), smoothstep(float(0.12), float(0.88), recessDepth))
      .mul(ceilingShadow)
      .mul(floorBounce);

    const backGrain = mineralToothFbm(
      vec3(baseUv.mul(vec2(260.0, 190.0)), pLocal.z.mul(16.0)),
      float(1.15)
    );

    const deepEarthFloor = cUmber.mul(0.052);
    const shadowedStrata = colBack.rgb
      .mul(cavityDarken)
      .mul(float(0.82).add(backGrain.mul(0.32)))
      .add(deepEarthFloor);

    const rimBleed = mix(cOchre, cSienna, float(0.5))
      .mul(float(1.0).sub(smoothstep(float(0.15), float(0.45), recessDepth)))
      .mul(0.056);

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
    alphaTest: 0.5,
  });
  {
    const vCol = attribute('color', 'vec3');
    const aCrestAlpha = attribute('aCrestAlpha', 'float');
    const grain = mineralToothFbm(positionLocal.mul(vec3(180.0, 180.0, 180.0)), float(1.15));
    const sparkle = pow(
      clamp(mineralToothFbm(positionLocal.add(vec3(5.2, 3.1, 8.7)), float(240.0)), 0.0, 1.0),
      5.0
    ).mul(0.18);

    sandCrestTop.colorNode = vCol.mul(float(0.85).add(grain.mul(0.28))).add(cCalcite.mul(sparkle));
    sandCrestTop.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(135.0), float(0.22));
    sandCrestTop.opacityNode = aCrestAlpha;
  }

  // ---------------------------------------------------------------------------
  // 8. Anatomically Detailed Worker Ant Chitin Materials
  //    Rich mahogany-amber harvester ant cuticle (strictly natural mineral pigments!)
  //    with subtle warm rim lift so ants read razor-sharp inside shadowed burrows!
  // ---------------------------------------------------------------------------
  const beetleChitin = new THREE.MeshStandardNodeMaterial({
    metalness: 0.24,
    roughness: 0.23,
    envMapIntensity: 1.24,
  });
  {
    // Use `positionGeometry` (pre-instanceMatrix!) so chitin micro-grain stays locked onto each ant!
    const p = positionGeometry;
    const chitinGrain = mineralToothFbm(p, float(120.0));
    const mahoganyDeep = mix(cUmber.mul(0.24), cRust.mul(0.28), float(0.42));
    const mahoganyWarm = mix(cSienna.mul(0.62), cAmber.mul(0.45), float(0.32));
    beetleChitin.colorNode = mix(
      mahoganyDeep,
      mahoganyWarm,
      smoothstep(float(0.24), float(0.82), chitinGrain)
    );
    // Subtle translucent amber-sienna edge lift so worker ants pop clearly in dark deep burrows!
    beetleChitin.emissiveNode = mix(cSienna, cAmber, float(0.35)).mul(0.065);
    beetleChitin.roughnessNode = clamp(float(0.21).add(chitinGrain.sub(0.5).mul(0.08)), 0.15, 0.3);
  }

  // Segmented Gaster Abdomen with authentic glossy chestnut-mahogany worker ant tergite sheen!
  const antGaster = new THREE.MeshStandardNodeMaterial({
    metalness: 0.24,
    roughness: 0.19,
    envMapIntensity: 1.34,
  });
  {
    const p = positionGeometry;
    const tergiteSeam = smoothstep(float(0.68), float(0.95), sin(p.x.mul(740.0)).mul(0.5).add(0.5));
    const finePore = mineralToothFbm(p, float(140.0));
    const cuticleBase = mix(cUmber.mul(0.22), cRust.mul(0.32), float(0.48));
    const cuticleSheen = mix(cSienna.mul(0.48), cAmber.mul(0.32), float(0.24));
    const tergiteCol = mix(cuticleBase, cuticleSheen, tergiteSeam.mul(0.55));
    antGaster.colorNode = tergiteCol.mul(float(0.88).add(finePore.mul(0.22)));
    antGaster.emissiveNode = mix(cSienna, cRust, float(0.5))
      .mul(float(0.055).add(tergiteSeam.mul(0.035)));
    antGaster.roughnessNode = clamp(float(0.17).add(finePore.sub(0.5).mul(0.07)), 0.13, 0.25);
  }

  // Warm chestnut-sienna jointed legs, elbowed antennae & curved pincer mandibles
  const beetleAccent = new THREE.MeshStandardNodeMaterial({
    metalness: 0.28,
    roughness: 0.27,
    envMapIntensity: 1.18,
  });
  {
    beetleAccent.colorNode = mix(cUmber.mul(0.34), cSienna.mul(0.52), float(0.58));
    beetleAccent.emissiveNode = mix(cSienna, cOchre, float(0.32)).mul(0.062);
  }

  // Glossy dark obsidian-umber compound eye beads
  const antEye = new THREE.MeshStandardNodeMaterial({
    metalness: 0.18,
    roughness: 0.1,
    envMapIntensity: 1.45,
  });
  {
    antEye.colorNode = cUmber.mul(0.08);
  }

  // ---------------------------------------------------------------------------
  // 9. 3D Faceted Tumbling & Ant-Carried Mineral Sand Granules (`aGrainColor`)
  // ---------------------------------------------------------------------------
  const mineralGrain3D = new THREE.MeshStandardNodeMaterial({
    metalness: 0.05,
    roughness: 0.52,
    envMapIntensity: 0.94,
  });
  {
    const aCol = attribute('aGrainColor', 'vec4');
    const crystalGrain = mineralToothFbm(positionGeometry, float(220.0));
    const glint = pow(clamp(crystalGrain, 0.0, 1.0), 4.2).mul(0.24).mul(aCol.w);
    mineralGrain3D.colorNode = aCol.rgb.mul(float(0.88).add(crystalGrain.mul(0.24))).add(
      cCalcite.mul(glint)
    );
    // Tiny sunlit edge lift so carried grains inside dark shafts stay clearly readable!
    mineralGrain3D.emissiveNode = aCol.rgb.mul(0.08);
  }

  // ---------------------------------------------------------------------------
  // 10. Soft Translucent 3D Avalanche Mineral Dust Wisps (`aDustColor`)
  // ---------------------------------------------------------------------------
  const dustMote3D = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  {
    const aDust = attribute('aDustColor', 'vec4');
    const nL = normalize(vec3(normalLocal));
    const vDir = normalize(uCamLocal.sub(vec3(positionLocal)));
    const softCore = pow(clamp(abs(dot(nL, vDir)), 0.0, 1.0), 2.3);
    const wispNoise = smoothstep(
      float(0.24),
      float(0.82),
      mineralToothFbm(positionGeometry.mul(95.0), float(1.4))
    );
    dustMote3D.colorNode = aDust.rgb;
    dustMote3D.opacityNode = clamp(aDust.w.mul(softCore).mul(wispNoise), 0.0, 0.18);
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
    antGaster,
    antEye,
    mineralGrain3D,
    dustMote3D,
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
      antGaster.dispose();
      antEye.dispose();
      mineralGrain3D.dispose();
      dustMote3D.dispose();
    },
  };
}
