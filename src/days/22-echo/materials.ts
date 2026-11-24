import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  atan,
  attribute,
  clamp,
  cos,
  cross,
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
  transformNormalToView,
  uniform,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { mineralToothFbm } from '../../core/tslMaterials';
import {
  APERTURE_HALF,
  CAVITY_HALF,
  ECHO_BOUNCE_COUNT,
  SODA_LIME_CHORD_PER_BOUNCE,
  SODA_LIME_EXTINCTION_RGB,
} from './optics';
import { CORE_ANCHOR_Y } from './physics';
import { type EchoMaterialsContract } from './geometry';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface EchoMaterials extends EchoMaterialsContract {
  agedBronze: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  knurledBrass: THREE.MeshStandardNodeMaterial;
  bluedSteel: THREE.MeshStandardNodeMaterial;
  infinityMirror: THREE.MeshBasicNodeMaterial;
  uCamLocal: { value: THREE.Vector3 };
  uMirrorNormals: Array<{ value: THREE.Vector3 }>;
  uCorePos: { value: THREE.Vector3 };
  uCoreAxis: { value: THREE.Vector3 };
  uCoreSpin: { value: number };
  uPluckFlash: { value: number };
  uTime: { value: number };
  uThumbHover: { value: number };
  uGlassHover: { value: number };
  dispose: () => void;
}

/**
 * Feathered radial softbox luminance card so PMREM reflections roll smoothly across
 * machined bronze chamfers, turned brass bezels, and diamond knurling without hard card edges.
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
 * Generates a studio softbox PMREM environment cubemap calibrated for machined dark bronze,
 * satin turned brass, and blued steel against the #212529 Chalky Slate gallery ground.
 */
export function createEchoStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0b0f13);

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
  addSoftbox(11.0, 9.0, [-5.0, 5.8, 4.0], 1.0, 0.96, 0.9, 1.9);
  // Right tall studio scrim (catches knurled brass thumbscrews & vertical bevels)
  addSoftbox(8.5, 11.0, [6.2, 2.8, 3.0], 0.99, 0.95, 0.89, 1.75);
  // Rear-top cool slate grazing rim scrim (carves clean bronze contour off #212529)
  addSoftbox(12.5, 5.4, [-1.0, 5.2, -5.8], 0.9, 0.95, 1.0, 1.55);
  // Zenith diffused canopy
  addSoftbox(9.0, 9.0, [0.0, 7.2, 0.5], 0.96, 0.94, 0.9, 0.95);
  // Rear-right cool-celadon fill card
  addSoftbox(8.0, 6.5, [4.8, 1.4, -5.6], 0.87, 0.94, 0.98, 1.12);
  // Subtle warm gallery table bounce
  addSoftbox(12.0, 12.0, [0.0, -6.2, 0.0], 0.96, 0.92, 0.86, 0.34);

  const renderTarget = pmrem.fromScene(envScene, 0.082);
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
 * Normalizes the finite difference in dimensionless noise-lattice coordinates (not 1/eps),
 * keeping machined metal faces optically flat with crisp sub-degree brushed micro-grain
 * and zero crumpled-foil or lumpy macro ripples.
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

    // Dimensionless lattice gradient (O(1) regardless of spatial frequency sc)
    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

/**
 * Analytical studio softbox dome radiance along a reflected exterior ray direction,
 * matching the PMREM softbox scrims for crisp exterior two-way mirror glass Fresnel reflections.
 */
const evalStudioSoftboxDome = Fn(([dir_in]: [TslNode]) => {
  const d = normalize(vec3(dir_in));

  // Base gallery Chalky Slate room gradient
  const skyT = clamp(d.y.mul(0.5).add(0.5), 0.0, 1.0);
  const roomBase = mix(vec3(0.014, 0.017, 0.02), vec3(0.036, 0.043, 0.049), skyT);

  // Overhead-left warm calcite key softbox
  const keyDir = normalize(vec3(-0.55, 0.64, 0.53));
  const keyDot = max(float(0.0), dot(d, keyDir));
  const keyBox = smoothstep(float(0.72), float(0.95), keyDot).mul(0.72);

  // Right tall studio scrim
  const rightDir = normalize(vec3(0.78, 0.34, 0.52));
  const rightDot = max(float(0.0), dot(d, rightDir));
  const rightBox = smoothstep(float(0.75), float(0.96), rightDot).mul(0.56);

  // Rear-top cool calcite rim strip
  const rimDir = normalize(vec3(-0.14, 0.66, -0.74));
  const rimDot = max(float(0.0), dot(d, rimDir));
  const rimBox = smoothstep(float(0.78), float(0.97), rimDot).mul(0.48);

  // Broad soft studio overhead sheen
  const broadGlow = pow(max(float(0.0), d.y.mul(0.7).add(0.3)), 3.0).mul(0.11);

  return roomBase
    .add(vec3(0.98, 0.94, 0.87).mul(keyBox.add(rightBox)))
    .add(vec3(0.84, 0.92, 0.98).mul(rimBox))
    .add(vec3(0.88, 0.86, 0.82).mul(broadGlow));
});

/**
 * Creates all materials for Day 22 - Echo:
 * - Aged dark-bronze skeleton frame
 * - Machined satin-brass bezels, gussets & bolts
 * - Diamond-knurled satin-brass micrometer drums (with vertex groove AO + hover warmth)
 * - Blued-steel micrometer lead-screw shafts
 * - 40-bounce analytic TSL two-way infinity mirror glass shader with soda-lime Beer-Lambert
 *   verdigris/celadon absorption, interior lit brass frame cages, glowing amber chamfer strips,
 *   and suspended swinging Realgar-Amber Edison filament inside a faceted Bone-Calcite prism!
 */
export function createEchoMaterials(): EchoMaterials {
  const uCamLocal = uniform(new THREE.Vector3(2.0, 1.3, 3.1));
  const uMirrorNormals = [
    uniform(new THREE.Vector3(-1, 0, 0)),
    uniform(new THREE.Vector3(1, 0, 0)),
    uniform(new THREE.Vector3(0, -1, 0)),
    uniform(new THREE.Vector3(0, 1, 0)),
    uniform(new THREE.Vector3(0, 0, -1)),
    uniform(new THREE.Vector3(0, 0, 1)),
  ];
  const uCorePos = uniform(new THREE.Vector3(0, -0.02, 0));
  const uCoreAxis = uniform(new THREE.Vector3(0, 1, 0));
  const uCoreSpin = uniform(0.35);
  const uPluckFlash = uniform(0);
  const uTime = uniform(0);
  const uThumbHover = uniform(0);
  const uGlassHover = uniform(0);

  // ---------------------------------------------------------------------------
  // 1. Aged Dark-Bronze Precision Optical Skeleton Frame
  // ---------------------------------------------------------------------------
  const agedBronze = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.3,
    envMapIntensity: 1.24,
  });
  {
    const fineBrush = mineralToothFbm(positionLocal, float(68.0));
    const subtlePatina = mineralToothFbm(positionLocal.add(vec3(2.3, 5.7, 1.4)), float(18.0));
    // Tight precision-machined dark bronze value range (eliminates blotchy light/dark mottling)
    const bronzeDeep = vec3(0.064, 0.049, 0.036);
    const bronzeSatin = vec3(0.118, 0.091, 0.064);
    const verdigrisTrace = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear).mul(0.16);
    const baseBronze = mix(
      bronzeDeep,
      bronzeSatin,
      smoothstep(float(0.32), float(0.76), fineBrush.mul(0.68).add(subtlePatina.mul(0.32))).mul(0.72)
    );
    agedBronze.colorNode = mix(
      baseBronze,
      verdigrisTrace,
      smoothstep(float(0.8), float(0.96), subtlePatina).mul(0.04)
    );
    agedBronze.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(84.0),
      float(0.008)
    );
    agedBronze.roughnessNode = clamp(
      float(0.28).add(fineBrush.sub(0.5).mul(0.08)).add(subtlePatina.sub(0.5).mul(0.04)),
      0.23,
      0.37
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Machined Turned Satin Brass (Bezels, Corner Gussets, Vernier Collars, Bolts)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.28,
  });
  {
    const fineSatin = mineralToothFbm(positionLocal, float(88.0));
    const microGrain = mineralToothFbm(positionLocal.add(vec3(4.1, 1.8, 6.2)), float(145.0));
    const brassAged = vec3(0.61, 0.48, 0.23);
    const brassBright = vec3(0.75, 0.6, 0.3);
    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.26), float(0.78), fineSatin.mul(0.6).add(microGrain.mul(0.4)))
    );
    satinBrass.normalNode = perturbNormal3DFbm(
      normalLocal,
      positionLocal,
      float(110.0),
      float(0.0035)
    );
    satinBrass.roughnessNode = clamp(
      float(0.21).add(fineSatin.sub(0.5).mul(0.05)).add(microGrain.sub(0.5).mul(0.03)),
      0.17,
      0.29
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Diamond-Knurled Satin Brass Micrometer Drums (Vertex Valley AO + Hover Lift)
  // ---------------------------------------------------------------------------
  const knurledBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.32,
  });
  {
    const aKnurlValley = attribute('aKnurlValley', 'float');
    const fineGrain = mineralToothFbm(positionLocal, float(64.0));
    const valleyDark = vec3(0.18, 0.13, 0.062);
    const crestBright = vec3(0.8, 0.65, 0.33);
    const grooveFactor = clamp(aKnurlValley, 0.0, 1.0);
    const baseKnurl = mix(valleyDark, crestBright, smoothstep(float(0.08), float(0.92), grooveFactor));
    const hoverWarmth = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear).mul(uThumbHover.mul(0.18));

    knurledBrass.colorNode = baseKnurl.add(hoverWarmth);
    knurledBrass.roughnessNode = clamp(
      mix(float(0.46), float(0.19), grooveFactor).add(fineGrain.sub(0.5).mul(0.04)),
      0.15,
      0.52
    );
    knurledBrass.metalnessNode = mix(float(0.74), float(0.94), grooveFactor);
  }

  // ---------------------------------------------------------------------------
  // 4. Tempered Blued-Steel Micrometer Lead-Screw Threads
  // ---------------------------------------------------------------------------
  const bluedSteel = new THREE.MeshStandardNodeMaterial({
    metalness: 0.88,
    roughness: 0.28,
    envMapIntensity: 1.18,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(52.0));
    const azuriteTint = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
    const steelDark = vec3(0.055, 0.072, 0.098);
    const steelHighlight = mix(vec3(0.17, 0.2, 0.25), azuriteTint.mul(0.85), float(0.32));
    bluedSteel.colorNode = mix(steelDark, steelHighlight, smoothstep(float(0.25), float(0.8), grain));
    bluedSteel.roughnessNode = clamp(float(0.24).add(grain.mul(0.12)), 0.18, 0.4);
  }

  // ---------------------------------------------------------------------------
  // 5. True 40-Bounce Analytic TSL Infinity Mirror Fragment Raytracer
  //    Zero texture fetches and zero fwidth/dpdx/dpdy inside dynamic branches!
  // ---------------------------------------------------------------------------
  const infinityMirror = new THREE.MeshBasicNodeMaterial({
    side: THREE.FrontSide,
  });

  const evalInfinityMirror = Fn(() => {
    const pEntry = vec3(positionLocal);
    const nOuter = normalize(vec3(normalLocal));
    const rd0 = normalize(pEntry.sub(uCamLocal));
    const viewDir = rd0.negate();
    const ndotv = clamp(dot(nOuter, viewDir), 0.0, 1.0);

    // Physical mineral pigment anchors (strictly zero digital RGB neons)
    const cRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
    const cOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
    const cSpore = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);
    const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
    const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
    const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
    const cUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
    const cLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);

    // Soda-lime ferrous-iron Beer-Lambert extinction per bounce (hoisted outside 40-bounce loop)
    const sigmaSodaLime = vec3(
      SODA_LIME_EXTINCTION_RGB[0],
      SODA_LIME_EXTINCTION_RGB[1],
      SODA_LIME_EXTINCTION_RGB[2]
    );
    const beerPerBounce = exp(sigmaSodaLime.negate().mul(float(SODA_LIME_CHORD_PER_BOUNCE))).toVar();

    // Precompute orthonormal local basis (cU, cV, cW) of the swinging faceted prism ONCE per pixel
    const cW = normalize(uCoreAxis).toVar();
    const cU0 = normalize(cross(vec3(0.0, 0.0, 1.0), cW));
    const cV0 = cross(cW, cU0);
    const cosSpin = cos(uCoreSpin);
    const sinSpin = sin(uCoreSpin);
    const cU = cU0.mul(cosSpin).add(cV0.mul(sinSpin)).toVar();
    const cV = cV0.mul(cosSpin).sub(cU0.mul(sinSpin)).toVar();

    // Subtle organic incandescent shimmer + pluck warmth surge (hoisted outside loop)
    const filamentPulse = float(1.0)
      .add(sin(uTime.mul(4.2)).mul(0.035))
      .add(uPluckFlash.mul(0.55))
      .toVar();
    const wireTop = vec3(0.0, CORE_ANCHOR_Y, 0.0);
    const wireBot = uCorePos.add(cW.mul(0.152)).toVar();
    const wireVec = wireTop.sub(wireBot).toVar();
    const wireLenSq = max(float(1e-5), dot(wireVec, wireVec)).toVar();

    // Faceted Bone-Calcite bipyramid crystal slab normals (4 opposing pairs -> 8 razor facets)
    const invSlabNorm = float(1.0 / Math.sqrt(1.0 + 1.0 + 0.78 * 0.78));
    const sN0 = vec3(1.0, 1.0, 0.78).mul(invSlabNorm);
    const sN1 = vec3(1.0, -1.0, 0.78).mul(invSlabNorm);
    const sN2 = vec3(-1.0, 1.0, 0.78).mul(invSlabNorm);
    const sN3 = vec3(-1.0, -1.0, 0.78).mul(invSlabNorm);
    const prismHalfD = float(0.082);

    // Exterior two-way mirror Fresnel reflection of studio softboxes
    const oneMinusV = max(float(0.0), float(1.0).sub(ndotv));
    const fresnelExt = float(0.068).add(float(0.52).mul(pow(oneMinusV, 4.2)));
    const rExt = reflect(rd0, nOuter);
    const extStudioRefl = evalStudioSoftboxDome(rExt);

    // Thick soda-lime glass green-celadon beveled edge catch along outer brass bezel aperture
    const aEntry = abs(pEntry);
    const maxE = max(aEntry.x, max(aEntry.y, aEntry.z));
    const minE = min(aEntry.x, min(aEntry.y, aEntry.z));
    const midE = aEntry.x.add(aEntry.y).add(aEntry.z).sub(maxE).sub(minE);
    const apertureEdge = float(APERTURE_HALF).sub(midE);
    const glassRimBevel = float(1.0).sub(smoothstep(float(0.0), float(0.024), apertureEdge));
    const glassRimTint = mix(cVerdigris, cCeladon, float(0.55)).mul(
      glassRimBevel.mul(float(0.18).add(fresnelExt.mul(0.45)).add(uGlassHover.mul(0.08)))
    );

    // Raytracer state across 40 optical bounces
    const ro = pEntry.toVar();
    const rd = rd0.toVar();
    const accum = vec3(0.0, 0.0, 0.0).toVar();
    const throughput = vec3(1.0, 1.0, 1.0)
      .mul(float(1.0).sub(fresnelExt.mul(0.52)))
      .toVar();
    const totalDist = float(0.0).toVar();

    Loop(ECHO_BOUNCE_COUNT, ({ i }) => {
      const bounceF = float(i);

      // -----------------------------------------------------------------------
      // A. Exact Analytic Ray vs. 6 Slightly Tilted Interior Mirror Planes
      // -----------------------------------------------------------------------
      const tWall = float(1e5).toVar();
      const hitN = vec3(0.0, 1.0, 0.0).toVar();

      for (let k = 0; k < 6; k++) {
        const planeN = uMirrorNormals[k];
        const denom = dot(rd, planeN);
        If(denom.lessThan(-1e-5), () => {
          const tCand = dot(ro, planeN).add(float(CAVITY_HALF)).negate().div(denom);
          If(tCand.greaterThan(1e-4).and(tCand.lessThan(tWall)), () => {
            tWall.assign(tCand);
            hitN.assign(planeN);
          });
        });
      }

      If(tWall.greaterThan(1e4), () => {
        Break();
      });

      // -----------------------------------------------------------------------
      // B1. Restrained Warm Realgar-Amber Optical Halo Around Swinging Core
      // -----------------------------------------------------------------------
      const toCore = uCorePos.sub(ro);
      const tClosest = clamp(dot(toCore, rd), float(0.0), tWall);
      const closestVec = ro.add(rd.mul(tClosest)).sub(uCorePos);
      const dCoreSq = dot(closestVec, closestVec);
      const haloTight = exp(dCoreSq.mul(-42.0)).mul(0.24);
      const haloSoft = exp(dCoreSq.mul(-8.5)).mul(0.048);
      const haloColor = mix(cRealgar, cOchre, float(0.32)).mul(
        haloTight.add(haloSoft).mul(filamentPulse)
      );
      accum.addAssign(haloColor.mul(throughput));

      // -----------------------------------------------------------------------
      // B2. Analytic Taut Satin-Brass Suspension Wire (Ceiling Anchor -> Prism)
      // -----------------------------------------------------------------------
      const oaW = ro.sub(wireBot);
      const dOa = dot(rd, oaW);
      const wOa = dot(wireVec, oaW);
      const dW = dot(rd, wireVec);
      const denomW = max(float(1e-5), wireLenSq.sub(dW.mul(dW)));
      const sWire = clamp(wOa.sub(dOa.mul(dW)).div(denomW), 0.0, 1.0);
      const tWire = max(float(0.0), dOa.negate().add(dW.mul(sWire)));

      If(tWire.greaterThan(1e-4).and(tWire.lessThan(tWall)), () => {
        const pRayW = ro.add(rd.mul(tWire));
        const pSegW = wireBot.add(wireVec.mul(sWire));
        const dWireSq = dot(pRayW.sub(pSegW), pRayW.sub(pSegW));
        const wireRad = float(0.0068).add(totalDist.mul(0.00045));
        If(dWireSq.lessThan(wireRad.mul(wireRad)), () => {
          const wLit = float(0.45).add(float(0.55).mul(float(1.0).sub(sWire)));
          const wireCol = mix(cUmber.mul(0.65), cOchre.mul(1.25), wLit);
          const wireEdge = float(1.0).sub(
            smoothstep(wireRad.mul(0.55), wireRad, sqrt(max(float(0.0), dWireSq)))
          );
          accum.addAssign(wireCol.mul(wireEdge).mul(throughput));
          throughput.mulAssign(float(1.0).sub(wireEdge.mul(0.88)));
        });
      });

      // -----------------------------------------------------------------------
      // B3. Suspended Faceted Bone-Calcite Prism + Glowing Realgar-Amber Filament
      // -----------------------------------------------------------------------
      const relCore = ro.sub(uCorePos);
      const bSphere = dot(relCore, rd);
      const cSphere = dot(relCore, relCore).sub(float(0.195 * 0.195));
      const discSphere = bSphere.mul(bSphere).sub(cSphere);

      If(discSphere.greaterThan(0.0), () => {
        const sqrtDisc = sqrt(discSphere);
        const tSphFar = bSphere.negate().add(sqrtDisc);
        const tSphNear = bSphere.negate().sub(sqrtDisc);

        If(tSphFar.greaterThan(1e-4).and(tSphNear.lessThan(tWall)), () => {
          // Transform ray into core's local rotated prism coordinate frame
          const roL = vec3(dot(relCore, cU), dot(relCore, cV), dot(relCore, cW)).toVar();
          const rdL = vec3(dot(rd, cU), dot(rd, cV), dot(rd, cW)).toVar();

          // Exact Kay-Kajiya 4-slab intersection against 8-facet Bone-Calcite crystal prism
          const tNear = float(-1e5).toVar();
          const tFar = float(1e5).toVar();
          const nPrismL = vec3(0.0, 0.0, 1.0).toVar();

          const slabNormals = [sN0, sN1, sN2, sN3];
          for (let m = 0; m < 4; m++) {
            const sn = slabNormals[m];
            const oDot = dot(roL, sn);
            const dDot = dot(rdL, sn);
            const absD = abs(dDot);
            If(absD.greaterThan(1e-5), () => {
              const invD = float(1.0).div(dDot);
              const t0 = prismHalfD.negate().sub(oDot).mul(invD);
              const t1 = prismHalfD.sub(oDot).mul(invD);
              const tMinS = min(t0, t1);
              const tMaxS = max(t0, t1);
              If(tMinS.greaterThan(tNear), () => {
                tNear.assign(tMinS);
                // Select outward slab normal facing incoming ray
                const flipSign = dDot.greaterThan(0.0).select(float(-1.0), float(1.0));
                nPrismL.assign(sn.mul(flipSign));
              });
              tFar.assign(min(tFar, tMaxS));
            }).Else(() => {
              If(abs(oDot).greaterThan(prismHalfD), () => {
                tNear.assign(float(1e5));
              });
            });
          }

          If(tFar.greaterThan(tNear).and(tFar.greaterThan(1e-4)).and(tNear.lessThan(tWall)), () => {
            const tEnter = max(float(0.0), tNear);
            const tExit = min(tWall, tFar);
            const chord = max(float(0.0), tExit.sub(tEnter));
            const pEnterL = roL.add(rdL.mul(tEnter));

            // Check top & bottom turned satin-brass prism collet caps (|z| > 0.132)
            const isBrassCap = abs(pEnterL.z).greaterThan(0.132);
            const facetNdotV = clamp(abs(dot(nPrismL, rdL)), 0.0, 1.0);
            const facetFresnel = float(0.14).add(
              float(0.82).mul(pow(max(float(0.0), float(1.0).sub(facetNdotV)), 3.2))
            );

            // Crisp studio key + rim specular highlight on crystal facets & brass caps
            const lKeyL = normalize(vec3(0.48, 0.52, 0.7));
            const hKey = normalize(lKeyL.sub(rdL));
            const facetSpec = pow(max(float(0.0), dot(nPrismL, hKey)), 36.0);
            const facetDiff = clamp(dot(nPrismL, lKeyL).mul(0.5).add(0.5), 0.22, 1.0);

            If(isBrassCap, () => {
              const capCol = mix(cUmber.mul(0.75), cOchre.mul(1.4), facetDiff).add(
                cCalcite.mul(facetSpec.mul(0.85))
              );
              accum.addAssign(capCol.mul(throughput));
              throughput.assign(vec3(0.0, 0.0, 0.0));
              Break();
            }).Else(() => {
              // Analytic inner glowing Realgar-Amber helical Edison filament capsule along local Z
              const rXY2Raw = rdL.x.mul(rdL.x).add(rdL.y.mul(rdL.y));
              const rXY2 = max(float(1e-5), rXY2Raw);
              const tAxisRaw = roL.x.mul(rdL.x).add(roL.y.mul(rdL.y)).negate().div(rXY2);
              const tAxisMid = tEnter.add(tExit).mul(0.5);
              const tAxis = clamp(
                mix(tAxisMid, tAxisRaw, smoothstep(float(1e-5), float(1e-3), rXY2Raw)),
                tEnter,
                tExit
              );
              const pFilL = roL.add(rdL.mul(tAxis));
              const zClamped = clamp(pFilL.z, -0.098, 0.098);
              const dFilRad = length(vec3(pFilL.x, pFilL.y, pFilL.z.sub(zClamped)));

              // Helical tungsten/realgar coil modulation along local Z (atan2 zero-safe)
              const coilPhi = atan(pFilL.y, pFilL.x.add(1e-5));
              const helixBand = sin(pFilL.z.mul(128.0).sub(coilPhi.mul(2.0))).mul(0.5).add(0.5);
              const filCoreMask = float(1.0).sub(smoothstep(float(0.016), float(0.036), dFilRad));
              const filGlowMask = exp(dFilRad.mul(dFilRad).mul(-1200.0));

              const hotCoreCol = mix(
                cRealgar.mul(2.6),
                mix(cSpore.mul(3.4), cCalcite.mul(4.2), helixBand),
                filCoreMask
              ).mul(filamentPulse);

              // Translucent Bone-Calcite crystal body glow + sharp crystalline facet rim
              const prismScatter = mix(cRealgar.mul(0.75), cCalcite.mul(0.82), facetDiff.mul(0.55))
                .mul(float(0.45).add(chord.mul(2.8)))
                .add(cCalcite.mul(facetFresnel.mul(0.68).add(facetSpec.mul(0.95))));

              const filEmit = hotCoreCol.mul(filCoreMask.mul(0.95).add(filGlowMask.mul(0.65)));
              const crystalAlpha = clamp(
                filCoreMask.mul(0.96).add(facetFresnel.mul(0.48)).add(chord.mul(2.1)),
                0.22,
                0.98
              );

              accum.addAssign(prismScatter.add(filEmit).mul(throughput));
              throughput.mulAssign(float(1.0).sub(crystalAlpha));

              const remMax = max(throughput.x, max(throughput.y, throughput.z));
              If(remMax.lessThan(0.014), () => {
                Break();
              });
            });
          });
        });
      });

      // -----------------------------------------------------------------------
      // C. Interior Mirror Wall Hit: Brass Frame Rails, Amber Chamfer Strips & Reflection
      // -----------------------------------------------------------------------
      const pWall = ro.add(rd.mul(tWall));
      totalDist.addAssign(tWall);

      // Exact distance to the 12 interior cube edges and 8 interior corners
      const aW = abs(pWall);
      const maxW = max(aW.x, max(aW.y, aW.z));
      const minW = min(aW.x, min(aW.y, aW.z));
      const midW = aW.x.add(aW.y).add(aW.z).sub(maxW).sub(minW);

      const edgeDist = float(CAVITY_HALF).sub(midW);
      const cornerDist = float(CAVITY_HALF).sub(minW);
      const aaW = float(0.0024).add(totalDist.mul(0.0014));

      // Solid interior brass/bronze rail (edgeDist < 0.036) + corner gusset brackets
      const railMask = float(1.0).sub(smoothstep(float(0.034).sub(aaW), float(0.034).add(aaW), edgeDist));
      const gussetMask = float(1.0)
        .sub(smoothstep(float(0.088).sub(aaW), float(0.088).add(aaW), cornerDist))
        .mul(float(1.0).sub(smoothstep(float(0.072).sub(aaW), float(0.072).add(aaW), edgeDist)));
      const solidMetalMask = max(railMask, gussetMask);

      // Luminous warm Realgar-Amber / Yellow-Ochre recessed optical chamfer strip along interior edges
      const stripCenter = float(0.044);
      const stripHalfW = float(0.0085).add(aaW.mul(0.8));
      const stripDist = abs(edgeDist.sub(stripCenter));
      const stripMask = float(1.0).sub(smoothstep(stripHalfW.mul(0.32), stripHalfW, stripDist));
      const stripOpacity = stripMask.mul(float(1.0).sub(solidMetalMask));

      const frameMask = clamp(solidMetalMask.add(stripOpacity), 0.0, 1.0);

      If(frameMask.greaterThan(0.004), () => {
        // Point-light shading from suspended amber core onto the interior bronze/brass frame
        const toLight = uCorePos.sub(pWall);
        const lDistSq = max(float(0.12), dot(toLight, toLight));
        const lDir = normalize(toLight);
        const atten = float(1.0).div(float(1.0).add(lDistSq.mul(1.15)));
        const nDotL = clamp(dot(hitN, lDir), 0.0, 1.0);
        const halfV = normalize(lDir.sub(rd));
        const metalSpec = pow(max(float(0.0), dot(hitN, halfV)), 28.0);

        // Distinguish dark-bronze outer channel from warmer satin-brass corner gusset
        const railMetal = mix(cUmber.mul(0.55), cOchre.mul(0.95), gussetMask.mul(0.75).add(0.25));
        const chamferBounce = exp(stripDist.mul(-36.0)).mul(0.28);
        const litFrame = railMetal
          .mul(float(0.18).add(nDotL.mul(1.15).mul(atten)))
          .add(cOchre.mul(metalSpec.mul(0.95).mul(atten)))
          .add(cRealgar.mul(float(0.09).mul(atten).add(chamferBounce)));

        // Glowing recessed amber chamfer edge strip (slightly softened across deep path lengths for crispness)
        const distSoften = float(1.0).div(float(1.0).add(totalDist.mul(0.028)));
        const cornerWarmth = float(1.0).add(
          float(1.0).sub(smoothstep(float(0.04), float(0.22), cornerDist)).mul(0.45)
        );
        const stripCol = mix(cRealgar, cOchre, float(0.42))
          .mul(float(2.35).mul(filamentPulse).mul(distSoften).mul(cornerWarmth));

        const combinedFrameCol = mix(stripCol, litFrame, solidMetalMask);
        accum.addAssign(combinedFrameCol.mul(frameMask).mul(throughput));
        throughput.mulAssign(float(1.0).sub(frameMask));

        If(frameMask.greaterThan(0.94), () => {
          Break();
        });
      });

      // Subtle glassy soda-lime surface scatter on mirror pane (gives tangible emerald cavity depth)
      const coreToWall = uCorePos.sub(pWall);
      const wallCoreAtten = float(1.0).div(float(1.0).add(dot(coreToWall, coreToWall).mul(2.4)));
      const glassDustVeil = mix(cVerdigris, cCeladon, float(0.35))
        .mul(0.0095)
        .add(cRealgar.mul(0.0085).mul(wallCoreAtten))
        .mul(float(1.0).sub(smoothstep(float(16.0), float(38.0), bounceF)).mul(0.75).add(0.25));
      accum.addAssign(glassDustVeil.mul(throughput));

      // Apply Authentic Soda-Lime Mirror Optics:
      // Fresnel dielectric reflectivity * Beer-Lambert ferrous-iron verdigris/celadon absorption
      const cosInc = clamp(abs(dot(rd, hitN)), 0.0, 1.0);
      const rMirror = mix(
        float(0.932),
        float(0.968),
        pow(max(float(0.0), float(1.0).sub(cosInc)), 3.0)
      );
      throughput.mulAssign(beerPerBounce.mul(rMirror));

      const maxEnergy = max(throughput.x, max(throughput.y, throughput.z));
      If(maxEnergy.lessThan(0.008), () => {
        Break();
      });

      // Reflect ray cleanly off the tilted mirror plane and step slightly inward
      rd.assign(normalize(reflect(rd, hitN)));
      ro.assign(pWall.add(hitN.mul(0.0005)));
    });

    // Residual depth floor -> shadowed Verdigris-Malachite / Lapis-Slate infinity abyss
    const abyssFloor = mix(cVerdigris.mul(0.055), cLapis.mul(0.045), float(0.35));
    accum.addAssign(abyssFloor.mul(throughput));

    // Composite exterior studio softbox Fresnel reflection + thick beveled soda-lime glass rim
    const rawRadiance = accum.add(extStudioRefl.mul(fresnelExt)).add(glassRimTint);

    // Smooth hue-preserving filmic highlight shoulder (warm incandescent core without neon clipping)
    const lum = dot(rawRadiance, vec3(0.2126, 0.7152, 0.0722));
    const compressed = rawRadiance.div(vec3(1.0).add(rawRadiance.mul(0.46)));
    const whiteGoldLift = smoothstep(float(1.6), float(5.5), lum).mul(0.28);
    return mix(compressed, cCalcite.mul(0.97), whiteGoldLift);
  });

  infinityMirror.colorNode = evalInfinityMirror();

  return {
    agedBronze,
    satinBrass,
    knurledBrass,
    bluedSteel,
    infinityMirror,
    uCamLocal,
    uMirrorNormals,
    uCorePos,
    uCoreAxis,
    uCoreSpin,
    uPluckFlash,
    uTime,
    uThumbHover,
    uGlassHover,
    dispose: () => {
      agedBronze.dispose();
      satinBrass.dispose();
      knurledBrass.dispose();
      bluedSteel.dispose();
      infinityMirror.dispose();
    },
  };
}
