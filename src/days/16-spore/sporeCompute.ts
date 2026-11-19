import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  abs,
  atan,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  hash,
  instanceIndex,
  instancedArray,
  length,
  max,
  mix,
  mod,
  normalize,
  sin,
  smoothstep,
  sqrt,
  uint,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { valueNoise3DTsl } from '../../core/tslMaterials';
import {
  MAX_VORTEX_RINGS,
  PERISTOME_NOZZLE_Y,
  PERISTOME_RIM_RADIUS,
  SAC_CENTER_Y,
  type EarthstarBellowsPhysics,
} from './bellowsPhysics';
import type { BellowsStepResult } from './types';

// Exact multiple of 64 workgroup size for optimal Apple Silicon Metal SIMD occupancy
export const SPORE_PARTICLE_COUNT = 49152;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Smooth 3D divergence-free curl-noise velocity field in TSL.
 */
const sampleCurlNoise3D = Fn(([pos_in, time_in]: [TslNode, TslNode]) => {
  const p = vec3(pos_in).mul(2.15).add(vec3(0.0, float(time_in).mul(-0.20), float(time_in).mul(0.08)));
  const e = float(0.055);

  const py1 = valueNoise3DTsl(p.add(vec3(0.0, e, 0.0)));
  const py0 = valueNoise3DTsl(p.sub(vec3(0.0, e, 0.0)));
  const pz1 = valueNoise3DTsl(p.add(vec3(0.0, 0.0, e)));
  const pz0 = valueNoise3DTsl(p.sub(vec3(0.0, 0.0, e)));

  const q = p.add(vec3(17.3, 31.7, 11.9));
  const qx1 = valueNoise3DTsl(q.add(vec3(e, 0.0, 0.0)));
  const qx0 = valueNoise3DTsl(q.sub(vec3(e, 0.0, 0.0)));
  const qz1 = valueNoise3DTsl(q.add(vec3(0.0, 0.0, e)));
  const qz0 = valueNoise3DTsl(q.sub(vec3(0.0, 0.0, e)));

  const r = p.add(vec3(41.1, 13.7, 29.3));
  const rx1 = valueNoise3DTsl(r.add(vec3(e, 0.0, 0.0)));
  const rx0 = valueNoise3DTsl(r.sub(vec3(e, 0.0, 0.0)));
  const ry1 = valueNoise3DTsl(r.add(vec3(0.0, e, 0.0)));
  const ry0 = valueNoise3DTsl(r.sub(vec3(0.0, e, 0.0)));

  const inv2e = float(1.0).div(e.mul(2.0));
  const cx = ry1.sub(ry0).sub(qz1.sub(qz0)).mul(inv2e);
  const cy = pz1.sub(pz0).sub(rx1.sub(rx0)).mul(inv2e);
  const cz = qx1.sub(qx0).sub(py1.sub(py0)).mul(inv2e);
  return vec3(cx, cy, cz);
});

/**
 * Evaluates Lamb-Oseen toroidal vortex ring circulation + crisp core confinement
 * so rings hold a sleek, flattened swirling smoke-ring halo silhouette as they rise.
 */
const evalToroidalRingVelocity = Fn(
  ([pos_in, centerRadius_in, axisCirc_in, coreAge_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const circ = vec4(axisCirc_in).w;
    const center = vec4(centerRadius_in).xyz;
    const majorR = vec4(centerRadius_in).w;
    const axis = normalize(vec4(axisCirc_in).xyz.add(vec3(0.0, 1e-5, 0.0)));
    const coreA = max(float(0.048), vec4(coreAge_in).x);
    const ringAge = vec4(coreAge_in).y;
    const swirlPhase = vec4(coreAge_in).w;

    const d = vec3(pos_in).sub(center);
    const h = dot(d, axis);
    const radVec = d.sub(axis.mul(h));
    const r = length(radVec);
    const radDir = radVec.div(max(r, float(1e-4)));

    // Gentle organic azimuthal Widnall wave as the ring ages
    const azim = atan(radVec.z, radVec.x);
    const instabilityAmp = smoothstep(0.65, 3.8, ringAge).mul(0.085);
    const widnall = sin(azim.mul(2.0).add(swirlPhase.mul(0.42)))
      .mul(0.65)
      .add(cos(azim.mul(3.0).sub(swirlPhase.mul(0.30))).mul(0.35));
    const effR = majorR.mul(float(1.0).add(widnall.mul(instabilityAmp)));
    // Vertically flattened toroidal tube metric (reads cleanly like Concept A's sculpted halo!)
    const effH = h.mul(1.38).sub(sin(azim.mul(2.0).sub(swirlPhase.mul(0.26))).mul(instabilityAmp).mul(0.14));

    const dr = r.sub(effR);
    const coreDistSq = dr.mul(dr).add(effH.mul(effH));
    const a2 = coreA.mul(coreA);

    // Poloidal Biot-Savart circulation: UP through hole, OUT over top, DOWN outside, IN underneath
    const polRaw = radDir.mul(effH).sub(axis.mul(dr));
    const polLen = max(length(polRaw), float(1e-4));
    const polDir = polRaw.div(polLen);

    const lambFactor = float(1.0).sub(exp(coreDistSq.negate().div(a2.mul(1.1))));
    const speedMag = circ.mul(lambFactor).div(coreDistSq.add(a2.mul(0.95))).mul(0.11);
    const farEnv = exp(coreDistSq.negate().div(a2.mul(7.5).add(0.08)));

    // Azimuthal helical ribbon flow around the ring circumference
    const tangDir = normalize(cross(axis, radDir).add(vec3(1e-5, 0.0, 0.0)));
    const azimSwirl = tangDir.mul(circ.mul(0.16).mul(exp(coreDistSq.negate().div(a2.mul(3.4)))));

    // Restorative core-sheet attraction keeps filaments wrapped crisply around the vortex halo
    const cohesionGate = float(1.0).sub(smoothstep(2.8, 6.8, ringAge));
    const corePull = radDir.mul(dr.negate()).add(axis.mul(h.negate().mul(1.55)))
      .mul(cohesionGate.mul(1.05).mul(exp(coreDistSq.negate().div(a2.mul(5.2).add(0.05)))));

    const activeGate = smoothstep(0.001, 0.02, circ);
    return polDir.mul(speedMag).add(azimSwirl).add(corePull).mul(farEnv).mul(activeGate);
  }
);

export interface SporeComputeSystem {
  sprite: THREE.Sprite;
  particleCount: number;
  runInit: (renderer: THREE.WebGPURenderer) => void;
  update: (
    renderer: THREE.WebGPURenderer,
    physics: EarthstarBellowsPhysics,
    stepResult: BellowsStepResult,
    dtSec: number,
    elapsedSec: number,
    pointerWorld: THREE.Vector3,
    pointerVelocity: THREE.Vector3,
    pointerSpeed: number,
    pointerIsDown: boolean
  ) => void;
  dispose: () => void;
}

/**
 * True WebGPU TSL-Compute Toroidal Vortex-Ring & Buoyant Curl-Noise Spore Plume
 * (`49,152` GPU-simulated particles rendered in 1 draw call via `SpriteNodeMaterial`).
 *
 * Dual-Population Optical Craft (Zero Giant Bokeh Polka-Dots):
 *   - Class 1 (58%): Tight micro-powder smoke filaments (`0.020..0.044`) with organic
 *     directional wisp breakup forming silky continuous golden toroidal ribbons.
 *   - Class 2 (42%): Razor-crisp `1.5-3.5 px` mineral sulfur pollen grains sparkling
 *     along the outer swirling rim and delicate upward curl trails.
 */
export function createSporeComputeSystem(): SporeComputeSystem {
  const N = SPORE_PARTICLE_COUNT;
  const fN = float(N);

  const posLifeBuf = instancedArray(N, 'vec4');
  const velAgeBuf = instancedArray(N, 'vec4');
  const metaBuf = instancedArray(N, 'vec4');

  const uDt = uniform(0.016);
  const uTime = uniform(0);
  const uNozzlePos = uniform(new THREE.Vector3(0, PERISTOME_NOZZLE_Y, 0));
  const uNozzleAxis = uniform(new THREE.Vector3(0, 1, 0));

  const uBurstWindow = uniform(new THREE.Vector4(0, 0, 0, 0));
  const uBurstAxis = uniform(new THREE.Vector3(0, 1, 0));
  const uStemWindow = uniform(new THREE.Vector4(0, 0, 0, 0));

  const uPtrPos = uniform(new THREE.Vector3(0, 0.72, 0));
  const uPtrVel = uniform(new THREE.Vector3(0, 0, 0));
  const uPtrMeta = uniform(new THREE.Vector4(0, 0, 0, 0));

  const ringCenterRadiusUniforms = Array.from({ length: MAX_VORTEX_RINGS }, () =>
    uniform(new THREE.Vector4(0, 0.78, 0, 0.16))
  );
  const ringAxisCircUniforms = Array.from({ length: MAX_VORTEX_RINGS }, () =>
    uniform(new THREE.Vector4(0, 1, 0, 0))
  );
  const ringCoreAgeUniforms = Array.from({ length: MAX_VORTEX_RINGS }, () =>
    uniform(new THREE.Vector4(0.08, 0, 0, 0))
  );

  // -------------------------------------------------------------------------
  // 1. One-Shot Boot Initialization (`initCompute`)
  //    Sculpts Concept A's exact hero composition: a sleek horizontal golden
  //    toroidal vortex ring hovering over a billowing S-curved central core
  //    plus delicate wispy curl-ribbons rising smoothly to the upper right!
  // -------------------------------------------------------------------------
  const initCompute = Fn(() => {
    const idx = instanceIndex;
    const fIdx = float(idx);

    const h0 = hash(idx.add(11));
    const h1 = hash(idx.add(317));
    const h2 = hash(idx.add(919));
    const h3 = hash(idx.add(2027));
    const h4 = hash(idx.add(4093));
    const h5 = hash(idx.add(7919));

    const pos = vec3(0.0, PERISTOME_NOZZLE_Y, 0.0).toVar();
    const vel = vec3(0.0, 0.0, 0.0).toVar();
    const life = float(0.0).toVar();
    const age = float(0.0).toVar();
    const maxLife = float(5.5).add(h4.mul(4.0)).toVar();

    // Population A (0..14999): Sleek Flattened Horizontal Toroidal Vortex Halo at y = 0.68, R = 0.36
    If(fIdx.lessThan(15000.0), () => {
      const azim = h0.mul(6.2831853);
      const polAngle = h1.mul(6.2831853);
      // Tight, creamy core tube with subtle granular halo fringe on outer edge
      const coreSpread = mix(float(0.022), float(0.082), h2.mul(h2));
      const ringWobble = sin(azim.mul(2.0).add(0.9)).mul(0.028).add(cos(azim.mul(3.0).sub(0.5)).mul(0.015));
      const majorR = float(0.36).add(ringWobble);

      // Slightly wider powdery tail dispersing along the outer left rim (matching Concept A!)
      const leftScatter = smoothstep(0.6, 1.0, cos(azim.sub(3.05)).mul(0.5).add(0.5)).mul(h3).mul(0.12);
      const radialDist = majorR.add(cos(polAngle).mul(coreSpread)).add(leftScatter);
      // Flattened vertical profile (`0.52`) for crisp horizontal smoke-ring silhouette
      const heightOff = sin(polAngle).mul(coreSpread).mul(0.52).add(sin(azim.mul(2.0).sub(0.4)).mul(0.022));

      pos.assign(
        vec3(
          cos(azim).mul(radialDist).add(0.01),
          float(0.68).add(heightOff),
          sin(azim).mul(radialDist).add(0.008)
        )
      );

      const radDir = vec3(cos(azim), 0.0, sin(azim));
      const tangDir = vec3(sin(azim).negate(), 0.0, cos(azim));
      const upDir = vec3(0.0, 1.0, 0.0);
      const polVel = radDir.mul(sin(polAngle)).sub(upDir.mul(cos(polAngle))).mul(0.20);
      vel.assign(polVel.add(tangDir.mul(0.18)).add(vec3(0.0, 0.09, 0.0)));
      age.assign(float(1.05).add(h4.mul(0.95)));
      life.assign(clamp(float(0.94).sub(h3.mul(0.20)), 0.48, 0.98));
    });

    // Population B (15000..22499): Billowing, Sculpted Central S-Curve Puff Core rising out of peristome into ring eye
    If(fIdx.greaterThanEqual(15000.0).and(fIdx.lessThan(22500.0)), () => {
      const sH = h0; // 0 at peristome mouth -> 1 crowning inside/slightly above ring eye
      const yStem = float(PERISTOME_NOZZLE_Y).add(sH.mul(0.54));
      const swirlA = h1.mul(6.2831853).add(sH.mul(5.2));
      // Plump billowing golden core (0.075 at mouth -> full 0.19 billowing crown inside the ring!)
      const bulge = float(1.0).add(sin(sH.mul(3.14159)).mul(0.48));
      const stemRad = mix(float(0.072), float(0.185), smoothstep(0.0, 0.80, sH))
        .mul(bulge)
        .mul(float(0.22).add(sqrt(h2).mul(0.78)));

      // Organic billowing S-curve twist
      const twistX = sin(sH.mul(5.2).add(0.6)).mul(0.068).mul(smoothstep(0.05, 0.70, sH));
      const twistZ = cos(sH.mul(4.5).sub(0.2)).mul(0.058).mul(smoothstep(0.05, 0.70, sH));

      pos.assign(vec3(cos(swirlA).mul(stemRad).add(twistX), yStem, sin(swirlA).mul(stemRad).add(twistZ)));
      vel.assign(vec3(cos(swirlA).mul(0.12), float(0.24).sub(sH.mul(0.07)), sin(swirlA).mul(0.12)));
      age.assign(sH.mul(1.15));
      life.assign(clamp(float(0.95).sub(sH.mul(0.16)), 0.56, 0.98));
    });

    // Population C (22500..25499): Graceful Curling Upper Ribbon Wisps peeling upward off the vortex ring
    If(fIdx.greaterThanEqual(22500.0).and(fIdx.lessThan(25500.0)), () => {
      const wT = h0; // 0 at ring top -> 1 fading near y = 1.38
      const yHigh = float(0.76).add(wT.mul(0.60));
      const ribbonId = selectScalar(h5.lessThan(0.55), float(1.0), float(-1.0));
      const pathX = float(0.06).add(wT.mul(0.22)).add(sin(wT.mul(7.2).add(ribbonId.mul(1.2))).mul(0.10));
      const pathZ = cos(wT.mul(6.1).sub(ribbonId.mul(0.9))).mul(0.11);
      const spreadA = h1.mul(6.2831853);
      const spreadR = sqrt(h2).mul(mix(float(0.032), float(0.078), wT));

      pos.assign(
        vec3(
          pathX.add(cos(spreadA).mul(spreadR)),
          yHigh.add((h3.sub(0.5)).mul(0.038)),
          pathZ.add(sin(spreadA).mul(spreadR))
        )
      );
      vel.assign(vec3(0.03, 0.08, 0.02));
      age.assign(float(2.0).add(wT.mul(2.2)));
      life.assign(clamp(float(0.70).sub(wT.mul(0.46)), 0.16, 0.74));
    });

    posLifeBuf.element(idx).assign(vec4(pos, life));
    velAgeBuf.element(idx).assign(vec4(vel, age));
    metaBuf.element(idx).assign(vec4(h3, h5, maxLife, h1));
  })().compute(N);

  // -------------------------------------------------------------------------
  // 2. Per-Frame Physics & Vortex-Ring Advection (`updateCompute`)
  // -------------------------------------------------------------------------
  const updateCompute = Fn(() => {
    const idx = instanceIndex;
    const fIdx = float(idx);

    const pNode = posLifeBuf.element(idx);
    const vNode = velAgeBuf.element(idx);
    const mNode = metaBuf.element(idx);

    const pos = pNode.xyz.toVar();
    const life = pNode.w.toVar();
    const vel = vNode.xyz.toVar();
    const age = vNode.w.toVar();
    const meta = mNode.toVar();

    const maxLife = max(float(3.5), meta.z);
    const spinSeed = meta.w;

    // A. Burst Spawn Window (Tight Toroidal Slug Ejection from Fluted Peristome)
    const burstDist = mod(fIdx.sub(uBurstWindow.x).add(fN), fN);
    const inBurst = uBurstWindow.y.greaterThan(0.5).and(burstDist.lessThan(uBurstWindow.y));

    If(inBurst, () => {
      const bSeedU = uint(abs(uBurstWindow.w).mul(9973.0));
      const h0 = hash(idx.add(bSeedU));
      const h1 = hash(idx.add(bSeedU.mul(3)).add(19));
      const h2 = hash(idx.add(bSeedU.mul(7)).add(73));
      const h3 = hash(idx.add(bSeedU.mul(13)).add(137));

      const bStr = uBurstWindow.z;
      const bAxis = normalize(uBurstAxis.add(vec3(0.0, 1e-5, 0.0)));
      const refVec = vec3(1.0, 0.0, 0.0);
      const t1 = normalize(cross(bAxis, refVec));
      const t2 = normalize(cross(bAxis, t1));

      const azim = h0.mul(6.2831853);
      const radDir = t1.mul(cos(azim)).add(t2.mul(sin(azim)));

      const isRingCore = h3.lessThan(0.76);
      const rRing = selectScalar(isRingCore, float(0.08).add(sqrt(h1).mul(0.055)), sqrt(h1).mul(0.068));
      const axialSpread = selectScalar(isRingCore, float(0.035).add(h2.mul(0.055)), h2.mul(0.05));

      pos.assign(uNozzlePos.add(radDir.mul(rRing)).add(bAxis.mul(axialSpread)));

      const jetSpeed = float(0.68).add(bStr.mul(0.54)).mul(float(0.78).add(h2.mul(0.38)));
      const lipCurlOut = selectScalar(isRingCore, float(0.24).add(h1.mul(0.20)), h1.mul(0.11));
      vel.assign(bAxis.mul(jetSpeed).add(radDir.mul(lipCurlOut)));

      age.assign(float(0.0));
      life.assign(float(1.0));
      meta.assign(vec4(h1, h2, float(5.4).add(h3.mul(3.6)), h0));
    });

    // B. Continuous Billowing Chimney Column Wisp Stream
    const stemDist = mod(fIdx.sub(uStemWindow.x).add(fN), fN);
    const inStem = inBurst.not().and(uStemWindow.y.greaterThan(0.5)).and(stemDist.lessThan(uStemWindow.y));

    If(inStem, () => {
      const sSeedU = uint(abs(uStemWindow.w).mul(7919.0));
      const h0 = hash(idx.add(sSeedU).add(5));
      const h1 = hash(idx.add(sSeedU.mul(5)).add(41));
      const h2 = hash(idx.add(sSeedU.mul(11)).add(97));
      const h3 = hash(idx.add(sSeedU.mul(17)).add(163));

      const azim = h0.mul(6.2831853);
      const rNoz = sqrt(h1).mul(PERISTOME_RIM_RADIUS * 0.92);
      const press = uStemWindow.z;

      pos.assign(
        uNozzlePos.add(vec3(cos(azim).mul(rNoz), h2.mul(0.04), sin(azim).mul(rNoz)))
      );

      // Fuller billowing puff out of the mouth rather than a thin pinched needle
      const stemSpeed = float(0.24).add(press.mul(0.42)).mul(float(0.75).add(h2.mul(0.40)));
      const swirlOut = float(0.19).add(press.mul(0.10));
      const swirlTan = vec3(sin(azim).negate(), 0.0, cos(azim)).mul(0.12);
      vel.assign(
        uNozzleAxis.mul(stemSpeed).add(vec3(cos(azim).mul(swirlOut), 0.0, sin(azim).mul(swirlOut))).add(swirlTan)
      );

      age.assign(float(0.0));
      life.assign(float(0.92));
      meta.assign(vec4(h3, h1, float(3.8).add(h2.mul(2.8)), h0));
    });

    // C. Advect Active Particles (Toroidal Rings + Restrained Curl Ribbons + Pointer Wake)
    If(life.greaterThan(0.001), () => {
      const dt = uDt;

      const ringVel = vec3(0.0, 0.0, 0.0).toVar();
      for (let k = 0; k < MAX_VORTEX_RINGS; k++) {
        ringVel.addAssign(
          evalToroidalRingVelocity(
            pos,
            ringCenterRadiusUniforms[k],
            ringAxisCircUniforms[k],
            ringCoreAgeUniforms[k]
          )
        );
      }

      // Smoothly ramping divergence-free curl ribbons (tight near ring, wispy as cloud rises!)
      const curlAmp = mix(float(0.045), float(0.22), smoothstep(0.6, 3.8, age));
      const curlVel = sampleCurlNoise3D(pos, uTime).mul(curlAmp);

      // Gentle buoyant thermal lift tapering cleanly near top of frame
      const buoyY = float(0.075).mul(float(1.0).sub(smoothstep(1.15, 1.75, pos.y).mul(0.80)));

      // Interactive 3D Pointer Aerodynamic Wake (swirling curl vortices + parting drag)
      const toPtr = pos.sub(uPtrPos);
      const ptrDistSq = dot(toPtr, toPtr);
      const wakeFalloff = exp(ptrDistSq.negate().mul(3.6));
      const ptrWakeSpeed = clamp(uPtrMeta.x, 0.0, 6.0);
      const dragForce = uPtrVel.mul(wakeFalloff).mul(0.62);
      const swirlForce = cross(uPtrVel, toPtr).mul(wakeFalloff).mul(2.15);
      const partPush = normalize(toPtr.add(vec3(0.0, 1e-4, 0.0)))
        .mul(wakeFalloff)
        .mul(ptrWakeSpeed.mul(0.18).add(uPtrMeta.y.mul(0.24)));
      const ptrForce = dragForce.add(swirlForce).add(partPush);

      const airTarget = ringVel.add(curlVel).add(vec3(0.0, buoyY, 0.0));
      const dragRate = float(2.45).add(spinSeed.mul(0.85));
      const relax = float(1.0).sub(exp(dt.negate().mul(dragRate)));

      vel.assign(mix(vel, airTarget, relax).add(ptrForce.mul(dt.mul(3.0))));

      const vLen = length(vel);
      const maxSpd = float(2.8);
      If(vLen.greaterThan(maxSpd), () => {
        vel.assign(vel.mul(maxSpd.div(vLen)));
      });

      pos.addAssign(vel.mul(dt));

      // Keep swirling spores cleanly outside the spore sac silhouette
      const sacCenter = vec3(0.0, SAC_CENTER_Y, 0.0);
      const fromSac = pos.sub(sacCenter);
      const sacDist = length(fromSac);
      const minSacR = float(0.56);
      If(sacDist.lessThan(minSacR).and(pos.y.lessThan(PERISTOME_NOZZLE_Y - 0.03)), () => {
        const pushDir = fromSac.div(max(sacDist, float(1e-4)));
        pos.assign(sacCenter.add(pushDir.mul(minSacR)));
        vel.addAssign(pushDir.mul(0.22));
      });

      age.addAssign(dt);
      // Accelerate graceful dissipation if a stray particle drifts past upper headroom
      const ceilingDrain = smoothstep(1.38, 1.85, pos.y).mul(0.45);
      life.assign(max(float(0.0), life.sub(dt.mul(float(1.0).div(maxLife).add(ceilingDrain)))));
    });

    posLifeBuf.element(idx).assign(vec4(pos, life));
    velAgeBuf.element(idx).assign(vec4(vel, age));
    metaBuf.element(idx).assign(meta);
  })().compute(N);

  // -------------------------------------------------------------------------
  // 3. Sculptural Powder-Veil + Pinpoint Mineral Pollen Sprite Material
  //    ZERO giant bokeh balls! Fine silky smoke wisps + crisp 1.5-3.5 px grains.
  // -------------------------------------------------------------------------
  const sporeMaterial = new THREE.SpriteNodeMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.NormalBlending,
  });

  const posLifeAttr = posLifeBuf.toAttribute();
  const velAgeAttr = velAgeBuf.toAttribute();
  const metaAttr = metaBuf.toAttribute();

  const pWorld = posLifeAttr.xyz;
  const pLife = posLifeAttr.w;
  const pAge = velAgeAttr.w;
  const classSeed = metaAttr.x;
  const toneSeed = metaAttr.y;
  const spinSeed = metaAttr.w;

  const vLife = pLife.toVarying();
  const vAge = pAge.toVarying();
  const vClass = classSeed.toVarying();
  const vTone = toneSeed.toVarying();
  const vSpin = spinSeed.toVarying();
  const vWorldPos = pWorld.toVarying();

  // 62% creamy continuous micro-powder smoke clouds (`0.030..0.064`, gentle soft oval 1.15:0.92),
  // 38% razor-crisp pinpoint mineral pollen grains (`0.0032..0.0068`)!
  const isVeil = smoothstep(0.64, 0.60, classSeed);
  const lifeFadeIn = smoothstep(0.0, 0.14, pAge);
  const aliveGate = smoothstep(0.002, 0.012, pLife);

  const veilBase = mix(float(0.032), float(0.064), smoothstep(0.0, 3.0, pAge))
    .mul(float(0.76).add(toneSeed.mul(0.48)));
  const grainBase = mix(float(0.0032), float(0.0068), toneSeed);

  const scaleX = mix(grainBase, veilBase.mul(1.16), isVeil)
    .mul(aliveGate)
    .mul(float(0.55).add(lifeFadeIn.mul(0.45)));
  const scaleY = mix(grainBase, veilBase.mul(0.90), isVeil)
    .mul(aliveGate)
    .mul(float(0.55).add(lifeFadeIn.mul(0.45)));

  sporeMaterial.positionNode = pWorld;
  sporeMaterial.scaleNode = vec2(scaleX, scaleY);
  sporeMaterial.rotationNode = spinSeed.mul(6.2831853).add(pAge.mul(float(0.42).add(toneSeed.mul(0.45))));

  // Physical mineral palette constants (strictly from src/core/palette.ts)
  const colSulfur = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);
  const colOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const colKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const colUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const colAmber = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);

  const quadUv = uv().sub(0.5).mul(2.0);
  const r2 = dot(quadUv, quadUv);
  const rDist = sqrt(max(r2, float(1e-6)));

  // Silky feather-soft Gaussian powder puff (seamless overlap, zero dashes or hard circles)
  const wispNoise = valueNoise3DTsl(vec3(quadUv.mul(1.8), vSpin.mul(19.0)));
  const veilProfile = exp(r2.negate().mul(2.9))
    .mul(float(1.0).sub(smoothstep(0.52, 0.98, rDist)))
    .mul(float(0.76).add(wispNoise.mul(0.36)));
  const grainProfile = float(1.0).sub(smoothstep(0.22, 0.86, rDist));

  // Studio 3-point softbox + back-top calcite rim lighting for 3D volumetric cloud modeling
  const keyDir = normalize(vec3(2.8, 4.2, 3.2));
  const rimDir = normalize(vec3(-1.5, 3.2, -3.6));

  const macroNormal = normalize(
    vec3(vWorldPos.x.mul(1.25), vWorldPos.y.sub(0.68).mul(1.05), vWorldPos.z.mul(1.25))
      .add(vec3(quadUv.x.mul(0.32), quadUv.y.mul(0.32), 0.24))
  );
  const keyLit = clamp(dot(macroNormal, keyDir).mul(0.5).add(0.5), 0.0, 1.0);
  const rimLit = clamp(dot(macroNormal, rimDir).mul(0.5).add(0.5), 0.0, 1.0);

  const viewToPart = normalize(vWorldPos.sub(cameraPosition));
  const forwardScatter = powSafe(clamp(dot(viewToPart, rimDir.negate()).mul(0.5).add(0.5), 0.0, 1.0), float(2.2));
  const heightLight = smoothstep(0.22, 1.25, vWorldPos.y);

  // Rich golden Spore Sulfur crests over warm Yellow Ochre / Raw Umber core shadows
  const coreTone = mix(colOchre, colSulfur, vTone.mul(0.65).add(0.30));
  const warmShadow = mix(colUmber.mul(1.1), colAmber.mul(0.72), 0.36);
  const shadedBody = mix(warmShadow, coreTone, clamp(keyLit.mul(0.74).add(heightLight.mul(0.34)), 0.0, 1.0));

  const fragIsVeil = smoothstep(0.64, 0.60, vClass);
  const glint = smoothstep(0.84, 0.99, sin(vAge.mul(5.2).add(vSpin.mul(41.0))).mul(0.5).add(0.5))
    .mul(float(1.0).sub(fragIsVeil));

  const litColor = mix(
    shadedBody,
    mix(colSulfur, colKaolin, 0.56),
    clamp(rimLit.mul(0.46).add(forwardScatter.mul(0.24)).add(glint.mul(0.58)), 0.0, 0.90)
  ).mul(float(1.16).add(glint.mul(0.48)));

  const fragFadeIn = smoothstep(0.0, 0.16, vAge);
  const fragFadeOut = smoothstep(0.0, 0.32, vLife);
  // Graceful viewport headroom dissipation so clouds never clip sharply against screen edges
  const headroomFade = smoothstep(1.68, 1.20, vWorldPos.y)
    .mul(smoothstep(1.32, 0.78, length(vWorldPos.xz)));

  const veilAlpha = veilProfile.mul(0.048).mul(float(0.65).add(vLife.mul(0.42)));
  const grainAlpha = grainProfile.mul(float(0.62).add(vTone.mul(0.36))).mul(float(0.78).add(glint.mul(0.22)));
  const finalAlpha = mix(grainAlpha, veilAlpha, fragIsVeil).mul(fragFadeIn).mul(fragFadeOut).mul(headroomFade);

  sporeMaterial.colorNode = litColor;
  sporeMaterial.opacityNode = clamp(finalAlpha, 0.0, 0.96);

  const sprite = new THREE.Sprite(sporeMaterial);
  sprite.count = N;
  sprite.frustumCulled = false;
  sprite.renderOrder = 4;

  let spawnCursor = 25500;
  let initialized = false;

  const runInit = (renderer: THREE.WebGPURenderer) => {
    if (initialized) return;
    initialized = true;
    renderer.compute(initCompute);
  };

  const update = (
    renderer: THREE.WebGPURenderer,
    physics: EarthstarBellowsPhysics,
    stepResult: BellowsStepResult,
    dtSec: number,
    elapsedSec: number,
    pointerWorld: THREE.Vector3,
    pointerVelocity: THREE.Vector3,
    pointerSpeed: number,
    pointerIsDown: boolean
  ) => {
    if (!initialized) {
      runInit(renderer);
    }

    uDt.value = Math.min(0.05, Math.max(0.001, dtSec));
    uTime.value = elapsedSec;
    uNozzlePos.value.set(stepResult.nozzlePos[0], stepResult.nozzlePos[1], stepResult.nozzlePos[2]);
    uNozzleAxis.value.set(stepResult.nozzleAxis[0], stepResult.nozzleAxis[1], stepResult.nozzleAxis[2]);

    uPtrPos.value.copy(pointerWorld);
    uPtrVel.value.copy(pointerVelocity);
    uPtrMeta.value.set(pointerSpeed, pointerIsDown ? 1.0 : 0.0, 0.0, 0.0);

    for (let k = 0; k < MAX_VORTEX_RINGS; k++) {
      const r = physics.rings[k];
      ringCenterRadiusUniforms[k].value.set(r.center[0], r.center[1], r.center[2], r.majorRadius);
      ringAxisCircUniforms[k].value.set(r.axis[0], r.axis[1], r.axis[2], r.active ? r.circulation : 0.0);
      ringCoreAgeUniforms[k].value.set(r.coreRadius, r.ageSec, r.active ? 1.0 : 0.0, r.swirlPhase);
    }

    if (stepResult.puffEvents.length > 0) {
      const ev = stepResult.puffEvents[stepResult.puffEvents.length - 1];
      const count = Math.min(9600, ev.particleCount);
      uBurstWindow.value.set(spawnCursor, count, ev.burstStrength, ev.seed + elapsedSec * 0.13);
      uBurstAxis.value.set(ev.axis[0], ev.axis[1], ev.axis[2]);
      spawnCursor = (spawnCursor + count) % N;
    } else {
      uBurstWindow.value.set(0, 0, 0, 0);
    }

    const stemCount = Math.min(280, Math.max(0, stepResult.continuousEmitCount));
    uStemWindow.value.set(spawnCursor, stemCount, physics.pressure, elapsedSec * 1.73 + 0.37);
    spawnCursor = (spawnCursor + stemCount) % N;

    renderer.compute(updateCompute);
  };

  const dispose = () => {
    sporeMaterial.dispose();
  };

  return {
    sprite,
    particleCount: N,
    runInit,
    update,
    dispose,
  };
}

function selectScalar(cond: TslNode, trueVal: TslNode, falseVal: TslNode): TslNode {
  return mix(falseVal, trueVal, cond.toFloat());
}

function powSafe(base: TslNode, expNode: TslNode): TslNode {
  const safeBase = max(abs(base), float(1e-5));
  return exp(safeBase.log().mul(expNode));
}
