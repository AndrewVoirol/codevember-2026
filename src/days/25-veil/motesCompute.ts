import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  abs,
  atan,
  clamp,
  cos,
  dot,
  exp,
  float,
  fract,
  hash,
  instanceIndex,
  instancedArray,
  length,
  max,
  mix,
  normalize,
  sin,
  smoothstep,
  sqrt,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { valueNoise3DTsl } from '../../core/tslMaterials';
import {
  BEAM_HALF_THICK_BOT,
  BEAM_HALF_THICK_TOP,
  CONE_RX_BOT,
  CONE_RX_TOP,
  CONE_RZ_BOT,
  CONE_RZ_TOP,
  CURVED_S_AMP_BOT,
  CURVED_S_AMP_TOP,
  CURVED_S_HALF_W_BOT,
  CURVED_S_HALF_W_TOP,
  PLANAR_HALF_W_BOT,
  PLANAR_HALF_W_TOP,
  VEIL_HEIGHT,
  VEIL_Y_BOT,
  VEIL_Y_TOP,
  type ModeWeights,
} from './veilSurface';

// Exact multiple of 64 for optimal Apple Silicon Metal compute workgroup occupancy
export const VEIL_MOTE_COUNT = 8192;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

const TWO_PI = Math.PI * 2;

/**
 * Smooth 3D divergence-free ambient studio thermal convection curl noise.
 */
const sampleStudioCurl3D = Fn(([pos_in, time_in]: [TslNode, TslNode]) => {
  const p = vec3(pos_in)
    .mul(1.45)
    .add(vec3(float(time_in).mul(0.055), float(time_in).mul(-0.075), float(time_in).mul(0.04)));
  const e = float(0.07);

  const py1 = valueNoise3DTsl(p.add(vec3(0.0, e, 0.0)));
  const py0 = valueNoise3DTsl(p.sub(vec3(0.0, e, 0.0)));
  const pz1 = valueNoise3DTsl(p.add(vec3(0.0, 0.0, e)));
  const pz0 = valueNoise3DTsl(p.sub(vec3(0.0, 0.0, e)));

  const q = p.add(vec3(19.3, 27.1, 13.7));
  const qx1 = valueNoise3DTsl(q.add(vec3(e, 0.0, 0.0)));
  const qx0 = valueNoise3DTsl(q.sub(vec3(e, 0.0, 0.0)));
  const qz1 = valueNoise3DTsl(q.add(vec3(0.0, 0.0, e)));
  const qz0 = valueNoise3DTsl(q.sub(vec3(0.0, 0.0, e)));

  const r = p.add(vec3(37.9, 11.3, 43.1));
  const rx1 = valueNoise3DTsl(r.add(vec3(e, 0.0, 0.0)));
  const rx0 = valueNoise3DTsl(r.sub(vec3(e, 0.0, 0.0)));
  const ry1 = valueNoise3DTsl(r.add(vec3(0.0, e, 0.0)));
  const ry0 = valueNoise3DTsl(r.sub(vec3(0.0, e, 0.0)));

  const inv2e = float(1.0).div(e.mul(2.0));
  return vec3(
    ry1.sub(ry0).sub(qz1.sub(qz0)).mul(inv2e),
    pz1.sub(pz0).sub(rx1.sub(rx0)).mul(inv2e),
    qx1.sub(qx0).sub(py1.sub(py0)).mul(inv2e)
  );
});

export interface VeilMotesSystem {
  sprite: THREE.Sprite;
  moteCount: number;
  runInit: (renderer: THREE.WebGPURenderer) => void;
  update: (
    renderer: THREE.WebGPURenderer,
    dtSec: number,
    elapsedSec: number,
    weights: ModeWeights,
    meanDensity: number,
    tearEnergy: number
  ) => number;
  dispose: () => void;
}

/**
 * WebGPU TSL-Compute Sunlit Dust Motes System (`8,192` motes, 1 draw call).
 *
 * Every mote drifts in 3D studio air, is advected directly by the live 2D incompressible
 * Navier-Stokes velocity + vorticity field when traversing the curtain envelope, and
 * ignites into a delicate backlit Bone-Calcite / Realgar-Amber spark ONLY while passing
 * through the razor-thin illuminated beam plane!
 */
export function createVeilMotesSystem(fluidTex: THREE.DataTexture): VeilMotesSystem {
  const N = VEIL_MOTE_COUNT;

  const posPhaseBuf = instancedArray(N, 'vec4'); // xyz: world pos, w: twinkle phase
  const velSeedBuf = instancedArray(N, 'vec4'); // xyz: world vel, w: size seed [0..1]
  const illumBuf = instancedArray(N, 'vec4'); // x: beamLight, y: vortSpark, z: rhoLocal, w: toneSeed

  const uDt = uniform(0.016);
  const uTime = uniform(0);
  const uModeWeights = uniform(new THREE.Vector3(1, 0, 0));

  // ---------------------------------------------------------------------------
  // 1. One-Shot Boot Initialization (`initCompute`)
  // ---------------------------------------------------------------------------
  const initCompute = Fn(() => {
    const idx = instanceIndex;
    const fIdx = float(idx);

    const h0 = hash(idx.add(17));
    const h1 = hash(idx.add(353));
    const h2 = hash(idx.add(1019));
    const h3 = hash(idx.add(2741));
    const h4 = hash(idx.add(5279));
    const h5 = hash(idx.add(8111));

    const pos = vec3(0.0, 0.0, 0.0).toVar();

    // 42% seeded in a close buoyant cloud straddling the front curtain plane so Frame 0 sparkles cleanly;
    // 58% spread throughout the full 3D room volume so Curved-S & Hollow Cone always have rich drifting motes!
    If(fIdx.lessThan(float(N * 0.42)), () => {
      const x = h0.sub(0.5).mul(2.55);
      const y = mix(float(VEIL_Y_BOT + 0.08), float(VEIL_Y_TOP - 0.03), h1);
      // Soft Gaussian-like concentration near z = 0 spanning ±0.26 so motes drift across the sheet
      const zSpread = h2.sub(0.5).add(h3.sub(0.5)).mul(0.26);
      pos.assign(vec3(x, y, zSpread));
    }).Else(() => {
      const x = h0.sub(0.5).mul(3.1);
      const y = mix(float(VEIL_Y_BOT), float(VEIL_Y_TOP + 0.04), h1);
      const z = h2.sub(0.5).mul(2.35);
      pos.assign(vec3(x, y, z));
    });

    const initVel = vec3(
      h3.sub(0.5).mul(0.04),
      h4.sub(0.42).mul(0.035),
      h5.sub(0.5).mul(0.04)
    );

    posPhaseBuf.element(idx).assign(vec4(pos, h5.mul(TWO_PI)));
    velSeedBuf.element(idx).assign(vec4(initVel, h4));
    illumBuf.element(idx).assign(vec4(0.0, 0.0, 0.5, h3));
  })().compute(N);

  // ---------------------------------------------------------------------------
  // 2. Per-Frame Fluid-Coupled 3D Advection & Razor-Thin Beam-Plane Gating
  // ---------------------------------------------------------------------------
  const updateCompute = Fn(() => {
    const idx = instanceIndex;
    const pNode = posPhaseBuf.element(idx);
    const vNode = velSeedBuf.element(idx);
    const iNode = illumBuf.element(idx);

    const pos = pNode.xyz.toVar();
    const phase = pNode.w.toVar();
    const vel = vNode.xyz.toVar();
    const sizeSeed = vNode.w;
    const toneSeed = iNode.w;

    const dt = uDt;
    const w0 = uModeWeights.x;
    const w1 = uModeWeights.y;
    const w2 = uModeWeights.z;

    // Project mote's 3D world position onto the active morphing light-veil surface
    const vRaw = float(VEIL_Y_TOP).sub(pos.y).div(VEIL_HEIGHT);
    const vClamped = clamp(vRaw, 0.0, 1.0);
    const inVert = smoothstep(float(-0.005), float(0.025), vRaw).mul(
      smoothstep(float(1.02), float(0.76), vRaw)
    );

    // Mode 0: Planar projection
    const hwP = mix(float(PLANAR_HALF_W_TOP), float(PLANAR_HALF_W_BOT), vClamped);
    const uPRaw = float(0.5).add(pos.x.div(hwP.mul(2.0)));
    const uP = clamp(uPRaw, 0.0, 1.0);
    const inHorizP = smoothstep(float(0.0), float(0.055), uPRaw).mul(
      smoothstep(float(1.0), float(0.945), uPRaw)
    );
    const dP = pos.z;
    const tanUP = vec3(1.0, 0.0, 0.0);
    const nP = vec3(0.0, 0.0, 1.0);

    // Mode 1: Curved S-Veil projection
    const hwS = mix(float(CURVED_S_HALF_W_TOP), float(CURVED_S_HALF_W_BOT), vClamped);
    const ampS = mix(float(CURVED_S_AMP_TOP), float(CURVED_S_AMP_BOT), vClamped);
    const uSRaw = float(0.5).add(pos.x.div(hwS.mul(2.0)));
    const uS = clamp(uSRaw, 0.0, 1.0);
    const inHorizS = smoothstep(float(0.0), float(0.055), uSRaw).mul(
      smoothstep(float(1.0), float(0.945), uSRaw)
    );
    const thetaS = uS.sub(0.5).mul(TWO_PI);
    const zSurfS = ampS.mul(sin(thetaS));
    const dzdx = ampS.mul(TWO_PI).mul(cos(thetaS)).div(hwS.mul(2.0));
    const invSlope = float(1.0).div(sqrt(float(1.0).add(dzdx.mul(dzdx))));
    const dS = pos.z.sub(zSurfS).mul(invSlope);
    const tanUS = vec3(invSlope, float(0.0), dzdx.mul(invSlope));
    const nS = vec3(dzdx.negate().mul(invSlope), float(0.0), invSlope);

    // Mode 2: Hollow Elliptical Light Cone projection
    const rxC = mix(float(CONE_RX_TOP), float(CONE_RX_BOT), vClamped);
    const rzC = mix(float(CONE_RZ_TOP), float(CONE_RZ_BOT), vClamped);
    const ex = pos.x.div(rxC);
    const ez = pos.z.div(rzC);
    const ellR = sqrt(ex.mul(ex).add(ez.mul(ez)).add(1e-6));
    const thetaC = atan(ex, ez.add(1e-6));
    const uC = fract(float(0.5).add(thetaC.div(TWO_PI)).add(1.0));
    const avgR = rxC.add(rzC).mul(0.5);
    const dC = ellR.sub(1.0).mul(avgR);
    const tanUC = normalize(vec3(rxC.mul(cos(thetaC)), float(0.0), rzC.negate().mul(sin(thetaC))).add(vec3(1e-5, 0.0, 0.0)));
    const nC = normalize(vec3(ex.div(rxC), float(0.0), ez.div(rzC)).add(vec3(0.0, 0.0, 1e-5)));

    const uProj = w0.mul(uP).add(w1.mul(uS)).add(w2.mul(uC));
    const dPlane = w0.mul(dP).add(w1.mul(dS)).add(w2.mul(dC));
    const inHoriz = w0.mul(inHorizP).add(w1.mul(inHorizS)).add(w2.mul(1.0));
    const tanU = normalize(tanUP.mul(w0).add(tanUS.mul(w1)).add(tanUC.mul(w2)).add(vec3(1e-5, 0.0, 0.0)));
    const surfN = normalize(nP.mul(w0).add(nS.mul(w1)).add(nC.mul(w2)).add(vec3(0.0, 0.0, 1e-5)));

    // Unconditional explicit-LOD sample of the live 128x128 RGBA16F Navier-Stokes fluid field!
    const fluidSample = texture(fluidTex, vec2(uProj, vClamped)).level(float(0.0));
    const rhoLocal = clamp(fluidSample.r, 0.0, 1.0);
    const fVelU = fluidSample.g;
    const fVelV = fluidSample.b;
    const fCurl = fluidSample.a;

    // Couple dust motes directly to the live fluid velocity + swirling vortex out-of-plane lift
    const fluidEnv = exp(dPlane.mul(dPlane).mul(-16.0)).mul(inVert).mul(inHoriz);
    const fluidScaleX = float(2.25);
    const fluidScaleY = float(VEIL_HEIGHT);
    const outPlaneSwirl = clamp(fCurl.mul(0.014), -0.35, 0.35).mul(
      sin(phase.mul(2.3).add(pos.y.mul(4.5)).add(uTime.mul(1.8)))
    );
    const fluidVel3D = tanU
      .mul(fVelU.mul(fluidScaleX))
      .add(vec3(0.0, fVelV.negate().mul(fluidScaleY), 0.0))
      .add(surfN.mul(outPlaneSwirl));

    // Gentle 3D ambient studio thermal convection drift (brings dark room motes smoothly across the beam!)
    const ambCurl = sampleStudioCurl3D(pos, uTime).mul(0.048);
    const ambDrift = vec3(
      float(0.014).add(sin(uTime.mul(0.19).add(toneSeed.mul(6.0))).mul(0.012)),
      float(0.018).add(sizeSeed.sub(0.5).mul(0.016)),
      float(0.022).mul(cos(uTime.mul(0.15).add(toneSeed.mul(9.0))))
    );

    const targetVel = ambCurl.add(ambDrift).add(fluidVel3D.mul(fluidEnv));
    const dragRate = float(2.6).add(sizeSeed.mul(1.2));
    const relax = float(1.0).sub(exp(dt.negate().mul(dragRate)));
    vel.assign(mix(vel, targetVel, relax));

    // Clamp max mote speed cleanly
    const spd = length(vel);
    const maxSpd = float(3.4);
    If(spd.greaterThan(maxSpd), () => {
      vel.assign(vel.mul(maxSpd.div(spd)));
    });

    pos.addAssign(vel.mul(dt));
    phase.addAssign(dt.mul(float(1.6).add(abs(fCurl).mul(0.14))));

    // Seamless studio room wraparound bounds
    const maxX = float(1.58);
    const maxZ = float(1.22);
    const minY = float(VEIL_Y_BOT - 0.06);
    const maxY = float(VEIL_Y_TOP + 0.06);

    If(pos.x.greaterThan(maxX), () => {
      pos.x.assign(pos.x.sub(maxX.mul(2.0)));
    });
    If(pos.x.lessThan(maxX.negate()), () => {
      pos.x.assign(pos.x.add(maxX.mul(2.0)));
    });
    If(pos.z.greaterThan(maxZ), () => {
      pos.z.assign(pos.z.sub(maxZ.mul(2.0)));
    });
    If(pos.z.lessThan(maxZ.negate()), () => {
      pos.z.assign(pos.z.add(maxZ.mul(2.0)));
    });
    If(pos.y.greaterThan(maxY), () => {
      pos.y.assign(minY.add(0.03));
    });
    If(pos.y.lessThan(minY), () => {
      pos.y.assign(maxY.sub(0.03));
    });

    // Strict Razor-Thin Beam-Plane Illumination Gate:
    // Motes catch golden/calcite sparks ONLY while piercing the thin illuminated slit plane!
    const beamHalfThick = mix(
      float(BEAM_HALF_THICK_TOP * 1.45),
      float(BEAM_HALF_THICK_BOT * 1.25),
      vClamped
    );
    const normDist = dPlane.div(beamHalfThick);
    const planeGate = exp(normDist.mul(normDist).negate());
    const colFalloff = float(1.0).div(float(1.0).add(vClamped.mul(0.65)));
    const vortSpark = smoothstep(float(2.5), float(18.0), abs(fCurl));

    // Motes twinkle brightly inside sunlit mist & along sheared wake eddies, dimming cleanly inside dark tears
    const tearMod = float(0.28).add(smoothstep(float(0.04), float(0.45), rhoLocal).mul(0.72));
    const beamLight = planeGate
      .mul(inVert)
      .mul(inHoriz)
      .mul(colFalloff)
      .mul(tearMod.add(vortSpark.mul(0.42)));

    posPhaseBuf.element(idx).assign(vec4(pos, phase));
    velSeedBuf.element(idx).assign(vec4(vel, sizeSeed));
    illumBuf.element(idx).assign(vec4(beamLight, vortSpark, rhoLocal, toneSeed));
  })().compute(N);

  // ---------------------------------------------------------------------------
  // 3. Pinpoint Backlit Sunlit Dust Mote Sprite Material (`AdditiveBlending`)
  // ---------------------------------------------------------------------------
  const moteMaterial = new THREE.SpriteNodeMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
  });

  const posPhaseAttr = posPhaseBuf.toAttribute();
  const velSeedAttr = velSeedBuf.toAttribute();
  const illumAttr = illumBuf.toAttribute();

  const pWorld = posPhaseAttr.xyz;
  const pPhase = posPhaseAttr.w;
  const pSizeSeed = velSeedAttr.w;
  const pBeamLight = illumAttr.x;
  const pVortSpark = illumAttr.y;
  const pToneSeed = illumAttr.w;

  const vBeamLight = pBeamLight.toVarying();
  const vVortSpark = pVortSpark.toVarying();
  const vPhase = pPhase.toVarying();
  const vTone = pToneSeed.toVarying();
  const vWorldY = pWorld.y.toVarying();

  // Collapse invisible dark-room motes (`pBeamLight < 0.015`) to 0 scale so 0 fill-rate is wasted!
  const litGate = smoothstep(float(0.015), float(0.14), pBeamLight);
  const glintScale = float(0.85).add(sin(pPhase.mul(2.1)).mul(0.5).add(0.5).mul(0.35));
  const baseSize = mix(float(0.006), float(0.0155), pSizeSeed.mul(pSizeSeed));
  const finalSize = baseSize
    .mul(glintScale)
    .mul(float(1.0).add(pVortSpark.mul(0.42)))
    .mul(litGate);

  moteMaterial.positionNode = pWorld;
  moteMaterial.scaleNode = vec2(finalSize, finalSize);

  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const cRealgar = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const cOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);

  const quadUv = uv().sub(0.5).mul(2.0);
  const r2 = dot(quadUv, quadUv);
  // Crisp Airy-disc dust glint core + subtle soft Mie halo (never giant bokeh polka dots!)
  const coreSpark = exp(r2.mul(-10.5));
  const softHalo = exp(r2.mul(-2.9)).mul(0.28);
  const radialFalloff = smoothstep(float(1.0), float(0.25), sqrt(max(r2, float(1e-6))));
  const shape = coreSpark.add(softHalo).mul(radialFalloff);

  const twinkle = float(0.55)
    .add(sin(vPhase.mul(3.1)).mul(0.25))
    .add(smoothstep(float(0.82), float(0.99), sin(vPhase.mul(5.7)).mul(0.5).add(0.5)).mul(0.75));

  const topWarmth = smoothstep(float(0.45), float(VEIL_Y_TOP), vWorldY);
  const baseMoteCol = mix(
    mix(cCalcite, cCeladon, float(0.18)),
    mix(cOchre, cRealgar, float(0.52)),
    clamp(topWarmth.mul(0.62).add(vTone.mul(0.24)).add(vVortSpark.mul(0.48)), 0.0, 1.0)
  );
  const hotSparkCol = mix(baseMoteCol, cCalcite.mul(1.65), coreSpark.mul(0.68));

  moteMaterial.colorNode = hotSparkCol.mul(shape).mul(vBeamLight).mul(twinkle).mul(1.75);
  moteMaterial.opacityNode = float(1.0);

  const sprite = new THREE.Sprite(moteMaterial);
  sprite.name = 'SunlitDustMotes';
  sprite.count = N;
  sprite.frustumCulled = false;
  sprite.renderOrder = 4;

  let initialized = false;

  const runInit = (renderer: THREE.WebGPURenderer): void => {
    if (initialized) return;
    initialized = true;
    renderer.compute(initCompute);
  };

  const update = (
    renderer: THREE.WebGPURenderer,
    dtSec: number,
    elapsedSec: number,
    weights: ModeWeights,
    meanDensity: number,
    tearEnergy: number
  ): number => {
    if (!initialized) {
      runInit(renderer);
    }

    uDt.value = Math.min(0.05, Math.max(0.001, dtSec));
    uTime.value = elapsedSec;
    uModeWeights.value.set(weights.planar, weights.curvedS, weights.cone);

    renderer.compute(updateCompute);

    // Accurate analytical estimate of lit motes currently piercing the razor-thin active beam sheet
    const sheetFactor =
      weights.planar * 0.24 + weights.curvedS * 0.26 + weights.cone * 0.31;
    const densityBoost = 0.72 + Math.min(1, meanDensity) * 0.48 + Math.min(0.25, tearEnergy * 0.08);
    return Math.round(N * sheetFactor * densityBoost);
  };

  const dispose = (): void => {
    moteMaterial.dispose();
  };

  return {
    sprite,
    moteCount: N,
    runInit,
    update,
    dispose,
  };
}
