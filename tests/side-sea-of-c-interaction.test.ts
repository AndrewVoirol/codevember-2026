import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  WAKE_SLOT_COUNT,
  SEA_STATE_PRESETS,
  createWaveFieldUniforms,
  evaluateWaveCPU,
} from '../src/side/sea-of-c/waveField';
import {
  CAMERA_WAVE_CLEARANCE,
  HERO_CAMERA_POSE,
  HERO_CAMERA_TARGET,
  MAX_CAMERA_DISTANCE,
  MAX_CAMERA_ELEVATION,
  MAX_SUN_ELEVATION,
  MIN_CAMERA_DISTANCE,
  MIN_CAMERA_ELEVATION,
  MIN_SUN_ELEVATION,
  SEA_STATE_TRANSITION_SEC,
  SUN_AZIMUTH_LIMIT,
  SeaOfCController,
  type OceanRayHit,
} from '../src/side/sea-of-c/interaction';

describe('Sea of C — SeaOfCController Interaction & Camera Rig', () => {
  it('initializes at the default Hero marine framing with clean zeroed interactive fields', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');
    const snap = ctrl.getSnapshot();

    expect(snap.seaState).toBe('golden');
    expect(snap.tool).toBe('wake');
    expect(snap.azimuth).toBeCloseTo(HERO_CAMERA_POSE.azimuth, 5);
    expect(snap.elevation).toBeCloseTo(HERO_CAMERA_POSE.elevation, 5);
    expect(snap.distance).toBeCloseTo(HERO_CAMERA_POSE.distance, 5);

    expect(ctrl.cameraTarget.x).toBeCloseTo(HERO_CAMERA_TARGET.x, 5);
    expect(ctrl.cameraTarget.y).toBeCloseTo(HERO_CAMERA_TARGET.y, 5);
    expect(ctrl.cameraTarget.z).toBeCloseTo(HERO_CAMERA_TARGET.z, 5);

    expect(snap.cameraX).toBeCloseTo(0, 3);
    expect(snap.cameraY).toBeGreaterThanOrEqual(3.8);
    expect(snap.cameraZ).toBeCloseTo(13.8, 1);

    expect(snap.activeWakeCount).toBe(0);
    expect(snap.vortexStrength).toBe(0);
    expect(snap.flipP).toBeGreaterThanOrEqual(0);
    expect(snap.flipP).toBeLessThanOrEqual(1);
  });

  it('clamps camera elevation & distance and enforces dynamic wave-surface clearance', () => {
    const u = createWaveFieldUniforms('storm');
    const ctrl = new SeaOfCController(u, 'storm');

    // Attempt extreme below-minimum elevation and distance
    ctrl.setCameraPose(0.25, -0.5, 0.5, true);
    let snap = ctrl.getSnapshot();
    expect(snap.elevation).toBeCloseTo(MIN_CAMERA_ELEVATION, 5);
    expect(snap.distance).toBeCloseTo(MIN_CAMERA_DISTANCE, 5);

    // Attempt extreme above-maximum elevation and distance
    ctrl.setCameraPose(1.2, 2.5, 99.0, true);
    snap = ctrl.getSnapshot();
    expect(snap.elevation).toBeCloseTo(MAX_CAMERA_ELEVATION, 5);
    expect(snap.distance).toBeCloseTo(MAX_CAMERA_DISTANCE, 5);

    // Skim right at minimum elevation across a stormy ocean and check wave clearance on every frame
    ctrl.setCameraPose(0.15, MIN_CAMERA_ELEVATION, MIN_CAMERA_DISTANCE, true);
    for (let i = 0; i < 90; i++) {
      const s = ctrl.step(1 / 30);
      const waveAtCam = evaluateWaveCPU(s.cameraX, s.cameraZ, u);
      expect(Number.isFinite(s.cameraY)).toBe(true);
      expect(s.cameraY).toBeGreaterThanOrEqual(
        waveAtCam.dy + CAMERA_WAVE_CLEARANCE - 1e-4
      );
    }
  });

  it('allocates wake ring-buffer slots with distance (>=0.55m) and interval (>=45ms) throttling', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');

    // First water move always spawns into slot 0
    expect(ctrl.handleWaterMove(1.0, 2.0, false)).toBe(true);
    expect(u.wakeData[0].x).toBeCloseTo(1.0, 5);
    expect(u.wakeData[0].y).toBeCloseTo(2.0, 5);
    expect(u.wakeData[0].w).toBeGreaterThan(0.5);
    expect(ctrl.getActiveWakeCount()).toBe(1);

    // Immediate move far away (dt = 0 < 45ms) must be throttled
    expect(ctrl.handleWaterMove(4.0, 2.0, false)).toBe(false);

    // Advance 60ms, move only 0.2m (< 0.55m) — stationary hover must NOT stamp ribs
    ctrl.step(0.06);
    expect(ctrl.handleWaterMove(1.2, 2.0, false)).toBe(false);
    expect(ctrl.getActiveWakeCount()).toBe(1);

    // Move >= 0.55m after >= 45ms — spawns slot 1
    expect(ctrl.handleWaterMove(2.0, 2.0, false)).toBe(true);
    expect(u.wakeData[1].x).toBeCloseTo(2.0, 5);
    expect(ctrl.getActiveWakeCount()).toBe(2);

    // Wrap cleanly around all 8 ring-buffer slots without allocating extra slots
    for (let i = 0; i < WAKE_SLOT_COUNT + 2; i++) {
      ctrl.step(0.08);
      const spawned = ctrl.handleWaterMove(3.0 + i * 1.1, 2.0, false);
      expect(spawned).toBe(true);
    }
    expect(u.wakeData.length).toBe(WAKE_SLOT_COUNT);
    expect(ctrl.getActiveWakeCount()).toBe(WAKE_SLOT_COUNT);
  });

  it('ramps vortex strength during water drag hold and decays smoothly after release', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');

    ctrl.beginWaterDrag(2.5, -1.5);
    for (let i = 0; i < 30; i++) {
      ctrl.updateWaterDrag(2.5 + i * 0.08, -1.5);
      ctrl.step(1 / 30);
    }

    const peakSnap = ctrl.getSnapshot();
    expect(peakSnap.vortexStrength).toBeGreaterThan(0.8);
    expect(u.uVortex.value.z).toBeCloseTo(peakSnap.vortexStrength, 4);
    expect(u.uVortex.value.w).toBeGreaterThan(2.0); // spinPhase advanced

    ctrl.endWaterDrag();
    for (let i = 0; i < 120; i++) {
      ctrl.step(1 / 30);
    }
    const decayedSnap = ctrl.getSnapshot();
    expect(decayedSnap.vortexStrength).toBeLessThan(0.02);
    expect(u.uVortex.value.z).toBeLessThan(0.02);
  });

  it('triggers rogue breaker wave at target coordinates and updates uRogueWave uniform', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');

    ctrl.step(0.5);
    const tBefore = ctrl.simTime;
    ctrl.triggerRogueWave(3.2, -2.4, 1.25);

    expect(u.uRogueWave.value.x).toBeCloseTo(3.2, 5);
    expect(u.uRogueWave.value.y).toBeCloseTo(-2.4, 5);
    expect(u.uRogueWave.value.z).toBeCloseTo(tBefore, 5);
    expect(u.uRogueWave.value.w).toBeCloseTo(1.25, 5);
    expect(ctrl.getActiveWakeCount()).toBeGreaterThanOrEqual(1);
  });

  it('clamps sky-drag sun steering and classifies sky vs water rays accurately', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');

    ctrl.steerSunBy(4.0, 2.0);
    for (let i = 0; i < 60; i++) ctrl.step(1 / 30);
    let snap = ctrl.getSnapshot();
    expect(snap.sunAzimuth).toBeLessThanOrEqual(SUN_AZIMUTH_LIMIT + 1e-4);
    expect(snap.sunElevation).toBeLessThanOrEqual(MAX_SUN_ELEVATION + 1e-4);

    ctrl.steerSunBy(-8.0, -5.0);
    for (let i = 0; i < 90; i++) ctrl.step(1 / 30);
    snap = ctrl.getSnapshot();
    expect(snap.sunAzimuth).toBeGreaterThanOrEqual(-SUN_AZIMUTH_LIMIT - 1e-4);
    expect(snap.sunElevation).toBeGreaterThanOrEqual(MIN_SUN_ELEVATION - 1e-4);

    const hit: OceanRayHit = { region: 'sky', x: 0, z: 0 };
    ctrl.intersectRayWithOcean(
      new THREE.Vector3(0, 4, 14),
      new THREE.Vector3(0, 0.2, -1).normalize(),
      hit
    );
    expect(hit.region).toBe('sky');

    ctrl.intersectRayWithOcean(
      new THREE.Vector3(0, 4, 14),
      new THREE.Vector3(0, -0.35, -1).normalize(),
      hit
    );
    expect(hit.region).toBe('water');
    expect(Number.isFinite(hit.x)).toBe(true);
    expect(Number.isFinite(hit.z)).toBe(true);
  });

  it('crossfades sea states smoothly (golden -> storm -> biolum) over ~1.1s with zero NaNs', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');

    const goldenAmp = SEA_STATE_PRESETS.golden.waveAmpScale;
    const stormAmp = SEA_STATE_PRESETS.storm.waveAmpScale;
    const biolumGainTarget = SEA_STATE_PRESETS.biolum.biolumGain;

    ctrl.setSeaState('storm', false);
    // Halfway through 1.1s transition
    for (let i = 0; i < 16; i++) {
      ctrl.step(SEA_STATE_TRANSITION_SEC / 32);
    }
    const midAmp = u.uWaveAmpScale.value;
    const loAmp = Math.min(goldenAmp, stormAmp);
    const hiAmp = Math.max(goldenAmp, stormAmp);
    expect(midAmp).toBeGreaterThanOrEqual(loAmp - 1e-4);
    expect(midAmp).toBeLessThanOrEqual(hiAmp + 1e-4);
    expect(ctrl.getSnapshot().transitionProgress).toBeGreaterThan(0.35);
    expect(ctrl.getSnapshot().transitionProgress).toBeLessThan(0.75);

    // Complete transition to storm
    for (let i = 0; i < 24; i++) {
      ctrl.step(1 / 30);
    }
    expect(u.uWaveAmpScale.value).toBeCloseTo(stormAmp, 4);
    expect(ctrl.getSnapshot().transitionProgress).toBeCloseTo(1, 5);

    // Crossfade storm -> biolum
    ctrl.setSeaState('biolum', false);
    for (let i = 0; i < 45; i++) {
      const s = ctrl.step(1 / 30);
      expect(Number.isFinite(s.cameraY)).toBe(true);
      expect(Number.isFinite(s.flipP)).toBe(true);
      expect(Number.isFinite(u.uBiolumGain.value)).toBe(true);
    }
    expect(ctrl.seaState).toBe('biolum');
    expect(u.uBiolumGain.value).toBeCloseTo(biolumGainTarget, 3);
  });

  it('keeps pun slot-flip flipP strictly in [0, 1] with zero NaNs and reacts to macro zoom & vortex', () => {
    const u = createWaveFieldUniforms('golden');
    const ctrl = new SeaOfCController(u, 'golden');

    let sawLow = false;
    let sawHigh = false;
    for (let i = 0; i < 240; i++) {
      const s = ctrl.step(1 / 30);
      expect(Number.isFinite(s.flipP)).toBe(true);
      expect(s.flipP).toBeGreaterThanOrEqual(0);
      expect(s.flipP).toBeLessThanOrEqual(1);
      if (s.flipP < 0.15) sawLow = true;
      if (s.flipP > 0.85) sawHigh = true;
    }
    expect(sawLow).toBe(true);
    expect(sawHigh).toBe(true);

    // Macro close-up zoom should smoothly bring flipP toward 1 ("C")
    ctrl.setCameraPose(0, 0.16, 6.0, true);
    for (let i = 0; i < 45; i++) {
      ctrl.step(1 / 30);
    }
    expect(ctrl.flipP).toBeGreaterThan(0.9);
  });
});
