import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { vec2 } from 'three/tsl';
import {
  computeSignedVolume,
  createBeveledCGeometry,
  createSprayCGeometry,
} from '../src/side/sea-of-c/glyphGeometry';
import {
  applySeaStateToUniforms,
  blendSeaStates,
  createWaveFieldUniforms,
  evaluateWaveCPU,
  evaluateWaveTSL,
  getHeroSolitonStateCPU,
  HERO_SOLITONS,
  sampleOceanHeightAndCurl,
  SEA_STATE_PRESETS,
  WAKE_SLOT_COUNT,
  type SeaStateConfig,
  type SeaStateId,
  type WaveSampleCPU,
} from '../src/side/sea-of-c/waveField';

function cloneSeaState(src: SeaStateConfig): SeaStateConfig {
  return {
    ...src,
    sunColor: [...src.sunColor],
    skyZenith: [...src.skyZenith],
    skyHorizon: [...src.skyHorizon],
    waterDeep: [...src.waterDeep],
    waterShallow: [...src.waterShallow],
    glintColor: [...src.glintColor],
    foamColor: [...src.foamColor],
  };
}

function getTriangleCount(geom: THREE.BufferGeometry): number {
  const index = geom.getIndex();
  const pos = geom.getAttribute('position');
  return index ? Math.floor(index.count / 3) : Math.floor(pos.count / 3);
}

describe('Sea of C — 3D Beveled Glyph Geometry', () => {
  it('builds hero beveled C strictly outward-wound, centered, unit-scaled, with smooth unit normals', () => {
    const geom = createBeveledCGeometry();
    const vol = computeSignedVolume(geom);
    const tris = getTriangleCount(geom);

    expect(vol).toBeGreaterThan(0.04);
    expect(tris).toBeGreaterThanOrEqual(140);
    expect(tris).toBeLessThanOrEqual(240);

    geom.computeBoundingBox();
    expect(geom.boundingBox).not.toBeNull();
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    geom.boundingBox!.getCenter(center);
    geom.boundingBox!.getSize(size);

    expect(Math.abs(center.x)).toBeLessThan(1e-4);
    expect(Math.abs(center.y)).toBeLessThan(1e-4);
    expect(Math.abs(center.z)).toBeLessThan(1e-4);

    const majorDiameter = Math.max(size.x, size.y);
    expect(majorDiameter).toBeCloseTo(1.0, 3);
    expect(size.z).toBeGreaterThan(0.15);
    expect(size.z).toBeLessThan(0.40);

    const normal = geom.getAttribute('normal');
    expect(normal).toBeDefined();
    for (let i = 0; i < normal.count; i++) {
      const nx = normal.getX(i);
      const ny = normal.getY(i);
      const nz = normal.getZ(i);
      const len = Math.hypot(nx, ny, nz);
      expect(Number.isFinite(len)).toBe(true);
      expect(len).toBeCloseTo(1.0, 3);
    }

    geom.dispose();
  });

  it('builds lightweight spray C strictly outward-wound, centered, and leaner than hero C', () => {
    const hero = createBeveledCGeometry();
    const spray = createSprayCGeometry();

    const heroTris = getTriangleCount(hero);
    const sprayTris = getTriangleCount(spray);
    const sprayVol = computeSignedVolume(spray);

    expect(sprayVol).toBeGreaterThan(0.03);
    expect(sprayTris).toBeLessThan(heroTris);
    expect(sprayTris).toBeGreaterThanOrEqual(80);
    expect(sprayTris).toBeLessThanOrEqual(160);

    spray.computeBoundingBox();
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    spray.boundingBox!.getCenter(center);
    spray.boundingBox!.getSize(size);

    expect(Math.abs(center.x)).toBeLessThan(1e-4);
    expect(Math.abs(center.y)).toBeLessThan(1e-4);
    expect(Math.abs(center.z)).toBeLessThan(1e-4);
    expect(Math.max(size.x, size.y)).toBeCloseTo(1.0, 3);

    hero.dispose();
    spray.dispose();
  });

  it('supports custom BeveledCOptions while preserving outward winding', () => {
    const custom = createBeveledCGeometry({
      outerRadius: 0.6,
      innerRadius: 0.34,
      openingDeg: 90,
      depth: 0.22,
      bevelThickness: 0.04,
      bevelSize: 0.03,
      bevelSegments: 2,
      curveSegments: 14,
    });
    expect(computeSignedVolume(custom)).toBeGreaterThan(0.03);
    custom.dispose();
  });
});

describe('Sea of C — Unified Wave Field Physics & Sea-State Blending', () => {
  it('evaluates finite values and unit-length normals across all 3 SEA_STATE_PRESETS', () => {
    const stateIds: SeaStateId[] = ['golden', 'storm', 'biolum'];
    const points: Array<[number, number]> = [
      [0, 0],
      [4.2, 5.0],
      [-6.5, -4.4],
      [14.3, -18.7],
      [-22.5, 9.1],
      [0.1, -60.0],
    ];
    const times = [0.0, 0.5, 2.3, 8.7];
    const sample: WaveSampleCPU = {
      dx: 0,
      dy: 0,
      dz: 0,
      nx: 0,
      ny: 1,
      nz: 0,
      crestEnergy: 0,
      curlFactor: 0,
      wakeEnergy: 0,
      orbitalPhase: 0,
    };

    for (const id of stateIds) {
      const u = createWaveFieldUniforms(id);
      for (const t of times) {
        u.uTime.value = t;
        for (const [x, z] of points) {
          evaluateWaveCPU(x, z, u, sample);
          expect(Number.isFinite(sample.dx)).toBe(true);
          expect(Number.isFinite(sample.dy)).toBe(true);
          expect(Number.isFinite(sample.dz)).toBe(true);
          expect(Number.isFinite(sample.nx)).toBe(true);
          expect(Number.isFinite(sample.ny)).toBe(true);
          expect(Number.isFinite(sample.nz)).toBe(true);
          expect(Number.isFinite(sample.crestEnergy)).toBe(true);
          expect(Number.isFinite(sample.curlFactor)).toBe(true);
          expect(Number.isFinite(sample.wakeEnergy)).toBe(true);
          expect(Number.isFinite(sample.orbitalPhase)).toBe(true);

          const nLen = Math.hypot(sample.nx, sample.ny, sample.nz);
          expect(nLen).toBeCloseTo(1.0, 4);
          expect(sample.crestEnergy).toBeGreaterThanOrEqual(0);
          expect(sample.crestEnergy).toBeLessThanOrEqual(1);
          expect(sample.curlFactor).toBeGreaterThanOrEqual(0);
          expect(sample.curlFactor).toBeLessThanOrEqual(1);
          expect(sample.wakeEnergy).toBeGreaterThanOrEqual(0);
          expect(sample.wakeEnergy).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('analytic parametric normals match numerical finite-difference tangent cross product', () => {
    const u = createWaveFieldUniforms('golden');
    u.uTime.value = 0.65;
    // Activate a wake + vortex + rogue wave to test every partial derivative term together
    u.wakeData[0].set(-3.2, 2.4, 0.1, 1.1);
    u.uVortex.value.set(1.5, -2.0, 0.85, 1.4);
    u.uRogueWave.value.set(-1.0, -5.0, 0.15, 0.9);

    const eps = 1e-4;
    const testCoords: Array<[number, number]> = [
      [4.3, 4.9], // foreground-right hero breaker lip
      [-6.3, -4.2], // midground-left breaker
      [-2.1, 1.8], // near active wake
      [3.6, -1.3], // near vortex ring wall
      [-1.4, -2.8], // near rogue breaker
    ];

    for (const [x, z] of testCoords) {
      const center = evaluateWaveCPU(x, z, u);
      const px = evaluateWaveCPU(x + eps, z, u);
      const mx = evaluateWaveCPU(x - eps, z, u);
      const pz = evaluateWaveCPU(x, z + eps, u);
      const mz = evaluateWaveCPU(x, z - eps, u);

      const txX = 1.0 + (px.dx - mx.dx) / (2 * eps);
      const txY = (px.dy - mx.dy) / (2 * eps);
      const txZ = (px.dz - mx.dz) / (2 * eps);

      const tzX = (pz.dx - mz.dx) / (2 * eps);
      const tzY = (pz.dy - mz.dy) / (2 * eps);
      const tzZ = 1.0 + (pz.dz - mz.dz) / (2 * eps);

      const nx = tzY * txZ - tzZ * txY;
      const ny = tzZ * txX - tzX * txZ;
      const nz = tzX * txY - tzY * txX;
      const len = Math.hypot(nx, ny, nz);

      const dotProd = (nx / len) * center.nx + (ny / len) * center.ny + (nz / len) * center.nz;
      expect(dotProd).toBeGreaterThan(0.992);
    }
  });

  it('produces immediate dramatic curling foreground-right and midground-left hero breakers at t = 0.5s', () => {
    const u = createWaveFieldUniforms('golden');
    u.uTime.value = 0.5;

    // Foreground-right hero breaker (soliton 0)
    const fgState = getHeroSolitonStateCPU(0, 0.5);
    const fgSample = evaluateWaveCPU(fgState.crestX, fgState.crestZ, u);
    expect(fgState.riseEnv).toBeGreaterThan(0.85);
    expect(fgState.curlLife).toBeGreaterThan(0.85);
    expect(fgSample.dy).toBeGreaterThan(2.0);
    expect(fgSample.crestEnergy).toBeGreaterThan(0.65);
    expect(fgSample.curlFactor).toBeGreaterThan(0.65);
    // Curls leftward (-X) across the foreground right third
    expect(fgSample.dx).toBeLessThan(-1.2);

    // Midground-left secondary breaker (soliton 1)
    const mgState = getHeroSolitonStateCPU(1, 0.5);
    const mgSample = evaluateWaveCPU(mgState.crestX, mgState.crestZ, u);
    expect(mgState.riseEnv).toBeGreaterThan(0.85);
    expect(mgSample.dy).toBeGreaterThan(1.8);
    expect(mgSample.crestEnergy).toBeGreaterThan(0.60);
    expect(mgSample.curlFactor).toBeGreaterThan(0.60);
    // Curls rightward (+X)
    expect(mgSample.dx).toBeGreaterThan(1.0);
    expect(HERO_SOLITONS.length).toBe(3);

    // Center golden sun-glitter channel stays unobstructed at t = 0.5s
    const centerSamples: Array<[number, number]> = [
      [0.0, 2.0],
      [-1.5, -2.0],
      [1.2, -5.0],
      [-2.5, 4.0],
    ];
    for (const [cx, cz] of centerSamples) {
      const cSample = evaluateWaveCPU(cx, cz, u);
      expect(cSample.curlFactor).toBeLessThan(0.15);
    }
  });

  it('generates origin-relative expanding wakes with localized wakeEnergy', () => {
    const u = createWaveFieldUniforms('golden');
    expect(u.wakeData.length).toBe(WAKE_SLOT_COUNT);

    // Spawn wake off-center at (-5.4, 3.8) at birthTime = 0.2s, sample at t = 0.6s (age = 0.4s -> frontR ≈ 1.24m)
    u.uTime.value = 0.6;
    const originX = -5.4;
    const originZ = 3.8;
    const baselineNear = evaluateWaveCPU(originX + 1.2, originZ, u);
    expect(baselineNear.wakeEnergy).toBe(0);

    u.wakeData[3].set(originX, originZ, 0.2, 1.2);
    const activeNear = evaluateWaveCPU(originX + 1.2, originZ, u);
    const activeFar = evaluateWaveCPU(originX + 24.0, originZ - 20.0, u);
    const atWorldOrigin = evaluateWaveCPU(0.0, 0.0, u);

    expect(activeNear.wakeEnergy).toBeGreaterThan(0.45);
    expect(Math.abs(activeNear.dy - baselineNear.dy)).toBeGreaterThan(0.15);
    expect(activeFar.wakeEnergy).toBeLessThan(0.01);
    expect(atWorldOrigin.wakeEnergy).toBeLessThan(0.02);
  });

  it('modulates vortex center eye depression, swirling turquoise ring wall, and rogue breaker surge', () => {
    const u = createWaveFieldUniforms('golden');
    u.uTime.value = 1.2;

    const vx = -12.0;
    const vz = 5.0;
    const baseEye = evaluateWaveCPU(vx, vz, u);
    const baseRim = evaluateWaveCPU(vx + 2.6, vz, u);

    u.uVortex.value.set(vx, vz, 1.1, 2.0);
    const vortEye = evaluateWaveCPU(vx, vz, u);
    const vortRim = evaluateWaveCPU(vx + 2.6, vz, u);

    // Center eye dips down while surrounding ring crest rises sharply
    expect(vortEye.dy).toBeLessThan(baseEye.dy - 0.6);
    expect(vortRim.dy).toBeGreaterThan(baseRim.dy + 1.0);
    expect(vortRim.crestEnergy).toBeGreaterThan(baseRim.crestEnergy + 0.4);
    expect(vortRim.curlFactor).toBeGreaterThan(baseRim.curlFactor + 0.4);
    expect(vortRim.wakeEnergy).toBeGreaterThan(0.6);
    // Tangential counter-clockwise swirl at +X rim pushes along +Z
    expect(vortRim.dz).toBeGreaterThan(baseRim.dz + 0.6);

    // Reset vortex and verify rogue breaker surge
    u.uVortex.value.set(0, 0, 0, 0);
    const rogueOriginX = 1.0;
    const rogueOriginZ = -4.0;
    const baseAtSurge = evaluateWaveCPU(rogueOriginX - 1.1, rogueOriginZ + 3.2, u);
    u.uRogueWave.value.set(rogueOriginX, rogueOriginZ, 0.45, 1.25);
    const activeSurge = evaluateWaveCPU(rogueOriginX - 1.1, rogueOriginZ + 3.2, u);
    expect(activeSurge.dy).toBeGreaterThan(baseAtSurge.dy + 1.2);
    expect(activeSurge.curlFactor).toBeGreaterThan(baseAtSurge.curlFactor + 0.4);
  });

  it('blends sea states smoothly and applies uniforms with zero NaNs', () => {
    const u = createWaveFieldUniforms('golden');
    const out = cloneSeaState(SEA_STATE_PRESETS.golden);

    blendSeaStates(SEA_STATE_PRESETS.golden, SEA_STATE_PRESETS.storm, 0.5, out);
    expect(out.waveAmpScale).toBeCloseTo(
      (SEA_STATE_PRESETS.golden.waveAmpScale + SEA_STATE_PRESETS.storm.waveAmpScale) * 0.5,
      5
    );
    expect(out.breakerHeight).toBeCloseTo(
      (SEA_STATE_PRESETS.golden.breakerHeight + SEA_STATE_PRESETS.storm.breakerHeight) * 0.5,
      5
    );
    expect(out.skyHorizon[0]).toBeCloseTo(
      (SEA_STATE_PRESETS.golden.skyHorizon[0] + SEA_STATE_PRESETS.storm.skyHorizon[0]) * 0.5,
      5
    );

    // Out-of-bounds and NaN t must clamp cleanly
    blendSeaStates(SEA_STATE_PRESETS.storm, SEA_STATE_PRESETS.biolum, 1.8, out);
    expect(out.id).toBe('biolum');
    expect(out.biolumGain).toBeCloseTo(1.0, 5);

    blendSeaStates(SEA_STATE_PRESETS.golden, SEA_STATE_PRESETS.biolum, Number.NaN, out);
    expect(out.id).toBe('golden');
    expect(Number.isFinite(out.waveAmpScale)).toBe(true);

    applySeaStateToUniforms(u, SEA_STATE_PRESETS.biolum);
    expect(u.uBiolumGain.value).toBe(1.0);
    expect(u.uSunDir.value.length()).toBeCloseTo(1.0, 5);
    expect(Number.isFinite(u.uWaterShallow.value.x)).toBe(true);
    expect(Number.isFinite(u.uWaterShallow.value.y)).toBe(true);
    expect(Number.isFinite(u.uWaterShallow.value.z)).toBe(true);
    expect(u.uWaterShallow.value.z).toBeGreaterThan(0.7);
  });

  it('constructs TSL wave evaluation graphs cleanly without runtime errors', () => {
    const u = createWaveFieldUniforms('golden');
    const sample = evaluateWaveTSL(vec2(1.5, -2.5), u);
    expect(sample.displacement).toBeDefined();
    expect(sample.normal).toBeDefined();
    expect(sample.crestEnergy).toBeDefined();
    expect(sample.curlFactor).toBeDefined();
    expect(sample.wakeEnergy).toBeDefined();
    expect(sample.orbitalPhase).toBeDefined();

    const helper = sampleOceanHeightAndCurl(vec2(4.2, 4.8), u);
    expect(helper.height).toBeDefined();
    expect(helper.curlFactor).toBeDefined();
  });
});
