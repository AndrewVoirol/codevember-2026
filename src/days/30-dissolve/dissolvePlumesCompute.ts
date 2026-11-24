import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  abs,
  atan,
  cameraPosition,
  clamp,
  cos,
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
  select,
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
  BEAKER_INNER_FLOOR_Y,
  DISSOLVE_PIGMENT_CYCLE,
  MAX_ACTIVE_CUBES,
  WATER_RADIUS,
  WATER_REST_Y,
  evalCrystalErosionMetrics,
  type BeakerDissolveSim,
} from './dissolveSim';

// Exact multiple of 64 for Apple Silicon Metal SIMD occupancy (> 12,000+ requirement)
export const PLUME_PARTICLE_COUNT = 16384;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Divergence-free 3D curl-noise velocity field for convective mineral marbling inside the beaker.
 */
const sampleBeakerCurl3D = Fn(([pos_in, time_in]: [TslNode, TslNode]) => {
  const p = vec3(pos_in).mul(3.1).add(vec3(0.0, float(time_in).mul(-0.18), float(time_in).mul(0.09)));
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
  return vec3(
    ry1.sub(ry0).sub(qz1.sub(qz0)).mul(inv2e),
    pz1.sub(pz0).sub(rx1.sub(rx0)).mul(inv2e),
    qx1.sub(qx0).sub(py1.sub(py0)).mul(inv2e)
  );
});

export interface DissolvePlumesSystem {
  sprite: THREE.Sprite;
  particleCount: number;
  runInit: (renderer: THREE.WebGPURenderer) => void;
  reset: (renderer: THREE.WebGPURenderer, starterPigmentRgb: readonly [number, number, number]) => void;
  update: (renderer: THREE.WebGPURenderer, sim: BeakerDissolveSim, dtSec: number) => void;
  dispose: () => void;
}

/**
 * 16,384-particle WebGPU TSL-Compute descending refractive schlieren & swirling
 * marbled pigment plume system contained strictly inside the borosilicate water column.
 */
export function createDissolvePlumesSystem(): DissolvePlumesSystem {
  const N = PLUME_PARTICLE_COUNT;
  const fN = float(N);

  const posLifeBuf = instancedArray(N, 'vec4');
  const velAgeBuf = instancedArray(N, 'vec4');
  const colorSeedBuf = instancedArray(N, 'vec4');

  const uDt = uniform(0.016);
  const uTime = uniform(0);
  const uSwirlMeta = uniform(new THREE.Vector4(0, 0, 0, 1)); // x=omega, y=ekman, z=conc, w=activeCount
  const uRodPosDip = uniform(new THREE.Vector4(0.24, WATER_REST_Y, 0.14, 0));
  const uRodVel = uniform(new THREE.Vector3(0, 0, 0));
  const uSpawnWindow = uniform(new THREE.Vector4(0, 0, 0, 0)); // x=cursor, y=count, z=rateScale, w=frameSeed
  const uStarterColor = uniform(
    new THREE.Vector3(...MINERAL_PALETTE[DISSOLVE_PIGMENT_CYCLE[0]].rInfLinear)
  );

  // Per-cube emitter uniform slots (4 concurrent dissolving mineral cubes)
  const uCubePosRad = Array.from({ length: MAX_ACTIVE_CUBES }, () =>
    uniform(new THREE.Vector4(0, BEAKER_INNER_FLOOR_Y + 0.2, 0, 0))
  );
  const uCubeColorRate = Array.from({ length: MAX_ACTIVE_CUBES }, () =>
    uniform(new THREE.Vector4(0.5, 0.6, 0.55, 0))
  );

  const FLOOR_LIMIT = BEAKER_INNER_FLOOR_Y + 0.016;
  const TOP_LIMIT = WATER_REST_Y - 0.022;
  const RAD_LIMIT = WATER_RADIUS - 0.022;

  // -------------------------------------------------------------------------
  // 1. Boot & Flush Initialization Compute Pass
  //    Seeds delicate descending schlieren streams & pooled bottom wisps around
  //    the freshly dropped starter crystal so Frame 1 reads immediately.
  // -------------------------------------------------------------------------
  const initCompute = Fn(() => {
    const idx = instanceIndex;
    const fIdx = float(idx);

    const h0 = hash(idx.add(17));
    const h1 = hash(idx.add(311));
    const h2 = hash(idx.add(907));
    const h3 = hash(idx.add(2029));
    const h4 = hash(idx.add(4099));

    const pos = vec3(0.0, FLOOR_LIMIT + 0.05, 0.0).toVar();
    const vel = vec3(0.0, 0.0, 0.0).toVar();
    const life = float(0.0).toVar();
    const age = float(0.0).toVar();

    // Seed 3,600 dense initial solute streamers peeling straight off the plunging crystal & pooling on the floor
    If(fIdx.lessThan(3600.0), () => {
      const azim = h0.mul(6.2831853);
      const isFloorPool = h1.lessThan(0.46);

      const descFrac = h2;
      const poolR = float(0.06).add(sqrt(h3).mul(0.38));
      const streamR = float(0.05).add(h3.mul(0.16)).add(descFrac.mul(0.10));
      const r = select(isFloorPool, poolR, streamR);

      const yPool = float(FLOOR_LIMIT).add(h2.mul(h2).mul(0.13));
      const yStream = float(BEAKER_INNER_FLOOR_Y + 0.45).sub(descFrac.mul(0.42));
      const y = clamp(select(isFloorPool, yPool, yStream), float(FLOOR_LIMIT), float(TOP_LIMIT));

      const swirlShift = sin(y.mul(10.5).add(azim)).mul(0.032);
      pos.assign(
        vec3(
          cos(azim).mul(r).add(0.04).add(swirlShift),
          y,
          sin(azim).mul(r).add(0.03).sub(swirlShift)
        )
      );

      vel.assign(
        vec3(
          cos(azim).mul(0.055).add(sin(azim).mul(0.085)),
          select(isFloorPool, float(0.012), float(-0.15)),
          sin(azim).mul(0.055).sub(cos(azim).mul(0.085))
        )
      );

      age.assign(select(isFloorPool, float(1.1).add(h4.mul(2.4)), h2.mul(1.2)));
      life.assign(clamp(float(0.95).sub(h4.mul(0.42)), 0.34, 0.96));
    });

    posLifeBuf.element(idx).assign(vec4(pos, life));
    velAgeBuf.element(idx).assign(vec4(vel, age));
    colorSeedBuf.element(idx).assign(vec4(uStarterColor, h4));
  })().compute(N);

  // -------------------------------------------------------------------------
  // 2. Per-Frame Heavy Solute Descent, Floor Pooling, Stirring & Spiral Marbling
  // -------------------------------------------------------------------------
  const updateCompute = Fn(() => {
    const idx = instanceIndex;
    const fIdx = float(idx);

    const pNode = posLifeBuf.element(idx);
    const vNode = velAgeBuf.element(idx);
    const cNode = colorSeedBuf.element(idx);

    const pos = pNode.xyz.toVar();
    const life = pNode.w.toVar();
    const vel = vNode.xyz.toVar();
    const age = vNode.w.toVar();
    const pigment = cNode.xyz.toVar();
    const seed = cNode.w.toVar();

    // A. Continuous Solute Peeling Spawn Window from Active Submerged Cubes
    const spawnDist = mod(fIdx.sub(uSpawnWindow.x).add(fN), fN);
    const inSpawn = uSpawnWindow.y
      .greaterThan(0.5)
      .and(spawnDist.lessThan(uSpawnWindow.y))
      .and(uSwirlMeta.w.greaterThan(0.5));

    If(inSpawn, () => {
      const fSeedU = uint(abs(uSpawnWindow.w).mul(9973.0));
      const h0 = hash(idx.add(fSeedU));
      const h1 = hash(idx.add(fSeedU.mul(3)).add(37));
      const h2 = hash(idx.add(fSeedU.mul(7)).add(101));
      const h3 = hash(idx.add(fSeedU.mul(13)).add(223));

      // Distribute spawned solute particles evenly across active dissolving cubes
      const activeCount = max(float(1.0), uSwirlMeta.w);
      const pickRank = mod(spawnDist, activeCount);

      const chosenPosRad = vec4(uCubePosRad[0]).toVar();
      const chosenColRate = vec4(uCubeColorRate[0]).toVar();
      const seenActive = float(0.0).toVar();

      for (let k = 0; k < MAX_ACTIVE_CUBES; k++) {
        const cPos = uCubePosRad[k];
        const cCol = uCubeColorRate[k];
        const isLive = cCol.w.greaterThan(0.001).and(cPos.w.greaterThan(0.004));
        If(isLive, () => {
          const isMatch = abs(seenActive.sub(pickRank)).lessThan(0.5);
          If(isMatch, () => {
            chosenPosRad.assign(cPos);
            chosenColRate.assign(cCol);
          });
          seenActive.addAssign(1.0);
        });
      }

      // Spawn tightly onto the actively eroding crystal crust, biased toward corners & lower flanks where heavy brine pours off
      const azim = h0.mul(6.2831853);
      const elev = h1.mul(2.0).sub(1.22);
      const rCos = sqrt(max(float(0.04), float(1.0).sub(clamp(elev.mul(elev), 0.0, 0.96))));
      const shellDir = vec3(cos(azim).mul(rCos), clamp(elev, -0.96, 0.96), sin(azim).mul(rCos));
      const spawnRad = chosenPosRad.w.mul(float(0.72).add(h2.mul(0.30)));

      pos.assign(chosenPosRad.xyz.add(shellDir.mul(spawnRad)));
      // Clamp safely inside liquid bounds
      pos.y.assign(clamp(pos.y, float(FLOOR_LIMIT), float(TOP_LIMIT)));

      // Heavy dissolved pigment brine peels off the eroding notch and streams decisively down to the glass floor
      const sinkSpeed = float(0.15).add(h2.mul(0.16)).add(chosenColRate.w.mul(0.95));
      const peelOut = float(0.048).add(h3.mul(0.062));
      vel.assign(vec3(shellDir.x.mul(peelOut), sinkSpeed.negate(), shellDir.z.mul(peelOut)));

      pigment.assign(chosenColRate.xyz);
      seed.assign(h3);
      age.assign(float(0.0));
      life.assign(float(1.0));
    });

    // B. Advect Live Solute Ribbons (Gravity Current + Floor Pool + Curl + Vortex Spiral + Rod Wake)
    If(life.greaterThan(0.001), () => {
      const dt = uDt;
      const omega = uSwirlMeta.x;
      const ekman = uSwirlMeta.y;

      const rHoriz = length(pos.xz);
      const rNorm = clamp(rHoriz.div(WATER_RADIUS), 0.0, 1.0);
      const r2 = rNorm.mul(rNorm);
      const wallNoSlip = max(float(0.0), float(1.0).sub(r2.mul(r2)));
      const omegaLocal = omega.mul(float(0.26).add(wallNoSlip.mul(0.74)));

      // Right-handed +Y vortex tangential velocity: omega x r = (+omega * z, 0, -omega * x)
      const swirlVel = vec3(omegaLocal.mul(pos.z), float(0.0), omegaLocal.mul(pos.x).negate());

      // Height above borosilicate glass floor [0..1]
      const heightAboveFloor = max(float(0.0), pos.y.sub(FLOOR_LIMIT));
      const floorLayer = exp(heightAboveFloor.negate().mul(12.0));

      // Heavy brine gravity sinking when freshly peeled, relaxing into convective lift as it diffuses
      const heavySink = float(-0.22).mul(exp(age.negate().mul(0.62))).mul(float(1.0).sub(floorLayer.mul(0.85)));
      const convectiveLift = smoothstep(1.2, 4.8, age).mul(float(0.052).add(seed.mul(0.042)));

      // Gravity current radial spread across the beaker bottom when heavy solute hits the floor
      const radDir = pos.xz.div(max(rHoriz, float(1e-4)));
      const gravitySpread = radDir.mul(floorLayer.mul(exp(age.negate().mul(0.45))).mul(0.16));

      // Secondary Ekman tea-leaf spiral circulation when stirred:
      // pulls floor-pooled pigment inward & lifts tight marbled ribbons up the center axis!
      const ekmanInward = radDir.mul(floorLayer.mul(ekman).mul(-0.34));
      const ekmanUpCore = exp(r2.negate().mul(5.2)).mul(ekman).mul(0.32);
      const ekmanDownWall = smoothstep(0.55, 0.96, rNorm).mul(ekman).mul(-0.14);

      // 3D divergence-free convective curl marbling (grows smoothly with age & swirl energy)
      const curlGain = mix(float(0.055), float(0.18), smoothstep(0.5, 4.2, age)).add(ekman.mul(0.10));
      const curlVel = sampleBeakerCurl3D(pos, uTime).mul(curlGain);

      // Direct local hydrodynamic shear wake around the dipped borosilicate glass stirring rod
      const rodForce = vec3(0.0, 0.0, 0.0).toVar();
      If(uRodPosDip.w.greaterThan(0.15), () => {
        const dRodXZ = pos.xz.sub(uRodPosDip.xz);
        const dRodSq = dot(dRodXZ, dRodXZ);
        const rodImmersedMask = smoothstep(uRodPosDip.y.sub(0.08), uRodPosDip.y.add(0.06), pos.y);
        const rodFalloff = exp(dRodSq.negate().mul(26.0)).mul(uRodPosDip.w).mul(rodImmersedMask);
        const shedVortex = vec3(dRodXZ.y.negate(), float(0.0), dRodXZ.x).mul(length(uRodVel).mul(1.8));
        rodForce.assign(uRodVel.mul(1.15).add(shedVortex).mul(rodFalloff));
      });

      const targetVel = swirlVel
        .add(curlVel)
        .add(vec3(gravitySpread.x.add(ekmanInward.x), heavySink.add(convectiveLift).add(ekmanUpCore).add(ekmanDownWall), gravitySpread.y.add(ekmanInward.y)))
        .add(rodForce);

      const dragRate = float(2.8).add(seed.mul(1.1));
      const relax = float(1.0).sub(exp(dt.negate().mul(dragRate)));
      vel.assign(mix(vel, targetVel, relax));

      // Speed safety clamp
      const spd = length(vel);
      const maxSpd = float(3.2);
      If(spd.greaterThan(maxSpd), () => {
        vel.assign(vel.mul(maxSpd.div(spd)));
      });

      pos.addAssign(vel.mul(dt));

      // Strict containment inside the borosilicate water column (curved wall, floor, meniscus)
      const newR = length(pos.xz);
      If(newR.greaterThan(RAD_LIMIT), () => {
        const nXZ = pos.xz.div(max(newR, float(1e-4)));
        pos.x.assign(nXZ.x.mul(RAD_LIMIT));
        pos.z.assign(nXZ.y.mul(RAD_LIMIT));
        // Deflect radial velocity cleanly into vertical & azimuthal flow along glass wall
        const vRad = dot(vel.xz, nXZ);
        If(vRad.greaterThan(0.0), () => {
          vel.x.subAssign(nXZ.x.mul(vRad));
          vel.z.subAssign(nXZ.y.mul(vRad));
        });
      });

      If(pos.y.lessThan(FLOOR_LIMIT), () => {
        pos.y.assign(FLOOR_LIMIT);
        If(vel.y.lessThan(0.0), () => {
          vel.y.mulAssign(-0.15);
        });
      });

      const vortexTopCeil = float(TOP_LIMIT).add(
        minScalar(float(0.045), omega.mul(omega).mul(0.0012)).mul(r2.mul(2.0).sub(1.0))
      );
      If(pos.y.greaterThan(vortexTopCeil), () => {
        pos.y.assign(vortexTopCeil);
        If(vel.y.greaterThan(0.0), () => {
          vel.y.mulAssign(-0.2);
        });
      });

      age.addAssign(dt);
      const maxLife = float(9.0).add(seed.mul(7.0));
      life.assign(max(float(0.0), life.sub(dt.div(maxLife))));
    });

    posLifeBuf.element(idx).assign(vec4(pos, life));
    velAgeBuf.element(idx).assign(vec4(vel, age));
    colorSeedBuf.element(idx).assign(vec4(pigment, seed));
  })().compute(N);

  // -------------------------------------------------------------------------
  // 3. Refractive Schlieren Wisp & Suspended Mineral Crystal Sprite Material
  // -------------------------------------------------------------------------
  const plumeMaterial = new THREE.SpriteNodeMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.NormalBlending,
  });

  const posLifeAttr = posLifeBuf.toAttribute();
  const velAgeAttr = velAgeBuf.toAttribute();
  const colorSeedAttr = colorSeedBuf.toAttribute();

  const pWorld = posLifeAttr.xyz;
  const pLife = posLifeAttr.w;
  const pVel = velAgeAttr.xyz;
  const pAge = velAgeAttr.w;
  const pPigment = colorSeedAttr.xyz;
  const pSeed = colorSeedAttr.w;

  const vLife = pLife.toVarying();
  const vAge = pAge.toVarying();
  const vPigment = pPigment.toVarying();
  const vSeed = pSeed.toVarying();
  const vWorldPos = pWorld.toVarying();
  const vSpeed = length(pVel).toVarying();

  // 76% silky elongated refractive schlieren ribbons, 24% pinpoint suspended micro-crystal flecks
  const isWisp = smoothstep(0.78, 0.74, pSeed);
  const aliveGate = smoothstep(0.003, 0.018, pLife);
  const spawnRamp = smoothstep(0.0, 0.25, pAge);

  const wispSize = mix(float(0.020), float(0.078), smoothstep(0.0, 5.2, pAge))
    .mul(float(0.72).add(fractSimple(pSeed.mul(7.31)).mul(0.55)));
  const stretchFactor = clamp(float(1.18).add(length(pVel).mul(1.45)), 1.15, 2.75);
  const grainSize = mix(float(0.0042), float(0.0095), fractSimple(pSeed.mul(13.7)));

  const scaleX = mix(grainSize, wispSize.mul(stretchFactor), isWisp)
    .mul(aliveGate)
    .mul(float(0.50).add(spawnRamp.mul(0.50)));
  const scaleY = mix(grainSize, wispSize.div(sqrt(stretchFactor)), isWisp)
    .mul(aliveGate)
    .mul(float(0.50).add(spawnRamp.mul(0.50)));

  plumeMaterial.positionNode = pWorld;
  plumeMaterial.scaleNode = vec2(scaleX, scaleY);
  // Orient stretched schlieren ribbons along flow angle + gentle tumble
  const velAngle = atan(pVel.y, pVel.x.add(1e-4));
  plumeMaterial.rotationNode = select(
    pSeed.lessThan(0.76),
    velAngle.add(pSeed.sub(0.38).mul(0.45)),
    pSeed.mul(6.2831853).add(pAge.mul(1.2))
  );

  // Fragment shading: mineral pigment + refractive schlieren edge shimmer + studio lighting
  const quadUv = uv().sub(0.5).mul(2.0);
  const r2 = dot(quadUv, quadUv);
  const rDist = sqrt(max(r2, float(1e-6)));

  const wispDetail = valueNoise3DTsl(vec3(quadUv.mul(2.1), vSeed.mul(23.0).add(vAge.mul(0.18))));
  const softBell = exp(r2.negate().mul(2.75))
    .mul(float(1.0).sub(smoothstep(0.50, 0.98, rDist)))
    .mul(float(0.72).add(wispDetail.mul(0.42)));

  // Thin bright refractive schlieren caustic halo on fresh concentrated solute tendrils!
  const schlierenRing = exp(abs(rDist.sub(0.42)).negate().mul(9.5))
    .mul(smoothstep(3.2, 0.15, vAge))
    .mul(float(0.35).add(vSpeed.mul(0.45)));

  const grainMask = float(1.0).sub(smoothstep(0.24, 0.86, rDist));

  // Studio 3-point softbox illumination across the cylindrical water column
  const colCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const keyDir = normalize(vec3(2.8, 4.2, 3.2));
  const rimDir = normalize(vec3(-1.5, 3.2, -3.6));
  const macroNorm = normalize(
    vec3(vWorldPos.x.mul(1.3), vWorldPos.y.add(0.15).mul(0.9), vWorldPos.z.mul(1.3))
      .add(vec3(quadUv.x.mul(0.28), quadUv.y.mul(0.28), 0.25))
  );
  const keyLit = clamp(dot(macroNorm, keyDir).mul(0.5).add(0.5), 0.0, 1.0);
  const rimLit = clamp(dot(macroNorm, rimDir).mul(0.5).add(0.5), 0.0, 1.0);

  const viewDir = normalize(cameraPosition.sub(vWorldPos));
  const backScatter = clamp(dot(viewDir.negate(), rimDir).mul(0.5).add(0.5), 0.0, 1.0);

  // Rich saturated mineral pigment ribbons (deep dense core when freshly peeled, luminous studio edge)
  const freshRichness = smoothstep(3.6, 0.1, vAge);
  const corePigment = mix(vPigment, vPigment.mul(vPigment.mul(0.45).add(0.68)), freshRichness.mul(0.55));
  const shadedPigment = corePigment
    .mul(float(0.62).add(keyLit.mul(0.64)).add(backScatter.mul(0.28)))
    .add(colCalcite.mul(rimLit.mul(0.11).add(schlierenRing.mul(0.22))));

  const fragIsWisp = smoothstep(0.78, 0.74, vSeed);
  const grainSparkle = smoothstep(
    0.80,
    0.99,
    sin(vAge.mul(6.8).add(vSeed.mul(53.0))).mul(0.5).add(0.5)
  );
  const grainCol = mix(vPigment.mul(1.18), colCalcite, float(0.26).add(grainSparkle.mul(0.42)));

  const finalColor = mix(grainCol, shadedPigment, fragIsWisp);

  const fadeIn = smoothstep(0.0, 0.14, vAge);
  const fadeOut = smoothstep(0.0, 0.34, vLife);
  // Freshly peeled ink-like descending pigment curtains read clearly while keeping late-stage eroded cores legible
  const concentrationAlpha = mix(float(0.215), float(0.042), smoothstep(0.35, 5.5, vAge));
  const wispAlpha = softBell.mul(concentrationAlpha).add(schlierenRing.mul(0.042));
  const grainAlpha = grainMask.mul(float(0.68).add(grainSparkle.mul(0.28)));

  plumeMaterial.colorNode = finalColor;
  plumeMaterial.opacityNode = clamp(
    mix(grainAlpha, wispAlpha, fragIsWisp).mul(fadeIn).mul(fadeOut),
    0.0,
    0.92
  );

  const sprite = new THREE.Sprite(plumeMaterial);
  sprite.name = 'SchlierenPigmentPlumes';
  sprite.count = N;
  sprite.frustumCulled = false;
  sprite.renderOrder = 3;

  let spawnCursor = 3600;
  let spawnAccum = 0;
  let initialized = false;

  const runInit = (renderer: THREE.WebGPURenderer) => {
    if (initialized) return;
    initialized = true;
    renderer.compute(initCompute);
  };

  const reset = (
    renderer: THREE.WebGPURenderer,
    starterPigmentRgb: readonly [number, number, number]
  ) => {
    uStarterColor.value.set(starterPigmentRgb[0], starterPigmentRgb[1], starterPigmentRgb[2]);
    spawnCursor = 3600;
    spawnAccum = 0;
    initialized = true;
    renderer.compute(initCompute);
  };

  const update = (renderer: THREE.WebGPURenderer, sim: BeakerDissolveSim, dtSec: number) => {
    if (!initialized) {
      runInit(renderer);
    }

    const dt = Math.min(0.04, Math.max(0.001, dtSec));
    uDt.value = dt;
    uTime.value = sim.simTime;

    let activeSubmerged = 0;
    let totalDissolveRate = 0;

    for (let k = 0; k < MAX_ACTIVE_CUBES; k++) {
      const c = sim.cubes[k];
      if (c.active && c.submerged && c.mass > 0.002) {
        const metrics = evalCrystalErosionMetrics(c.mass);
        const pal = MINERAL_PALETTE[c.pigmentKey];
        uCubePosRad[k].value.set(c.pos.x, c.pos.y, c.pos.z, metrics.halfExtent);
        uCubeColorRate[k].value.set(
          pal.rInfLinear[0],
          pal.rInfLinear[1],
          pal.rInfLinear[2],
          Math.max(0.01, c.lastDissolveRate)
        );
        activeSubmerged++;
        totalDissolveRate += Math.max(0.01, c.lastDissolveRate);
      } else {
        uCubePosRad[k].value.set(0, BEAKER_INNER_FLOOR_Y, 0, 0);
        uCubeColorRate[k].value.set(0, 0, 0, 0);
      }
    }

    uSwirlMeta.value.set(
      sim.swirlAngularVel,
      sim.ekmanStrength,
      sim.bathSoluteMass,
      activeSubmerged
    );

    // Rod immersed tip height inside water column
    const rodBottomY = WATER_REST_Y + 0.45 - sim.rod.dip * (WATER_REST_Y - BEAKER_INNER_FLOOR_Y + 0.30);
    uRodPosDip.value.set(sim.rod.x, rodBottomY, sim.rod.z, sim.rod.dip);
    uRodVel.value.set(sim.rod.vx, 0, sim.rod.vz);

    if (activeSubmerged > 0) {
      // Rich continuous ink-like solute ribbon shedding (~1,450..3,400 particles/sec across active cubes)
      const emitPerSec = 1450.0 + totalDissolveRate * 22500.0;
      spawnAccum += emitPerSec * dt;
      const count = Math.min(340, Math.floor(spawnAccum));
      spawnAccum -= count;
      uSpawnWindow.value.set(spawnCursor, count, totalDissolveRate, sim.simTime * 1.37 + 0.51);
      spawnCursor = (spawnCursor + count) % N;
    } else {
      uSpawnWindow.value.set(0, 0, 0, 0);
    }

    renderer.compute(updateCompute);
  };

  const dispose = () => {
    plumeMaterial.dispose();
  };

  return {
    sprite,
    particleCount: N,
    runInit,
    reset,
    update,
    dispose,
  };
}

function minScalar(a: TslNode, b: TslNode): TslNode {
  return select(a.lessThan(b), a, b);
}

function fractSimple(x: TslNode): TslNode {
  return x.sub(x.floor());
}
