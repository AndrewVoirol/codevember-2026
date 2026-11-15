import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { RubberBandBallSim } from './bandPhysics';
import { createRubberBallVisuals } from './rubberOptics';
import type { RubberBallDiagnostics } from './types';

declare global {
  interface Window {
    __TENSION__?: {
      getState: () => RubberBallDiagnostics;
    };
  }
}

interface SnapImpulseShape {
  snapId: number;
  bandId: number;
  hitPos: readonly [number, number, number];
  pullDir: readonly [number, number, number];
  normal: readonly [number, number, number];
  intensity: number;
}

interface SimWithSnapInfo {
  getLastSnapInfo?: () => SnapImpulseShape | null;
}

// Pre-allocated scratch objects for zero-GC pointer interaction & torque math
const _ndc = new THREE.Vector2();
const _raycaster = new THREE.Raycaster();
const _localRay = new THREE.Ray();
const _invGroupMat = new THREE.Matrix4();
const _rayOriginTuple: [number, number, number] = [0, 0, 0];
const _rayDirTuple: [number, number, number] = [0, 0, 0];
const _pullDirTuple: [number, number, number] = [0, 0, 0];
const _deltaTuple: [number, number, number] = [0, 0, 0];
const _dragPlane = new THREE.Plane();
const _camDir = new THREE.Vector3();
const _camDirLocal = new THREE.Vector3();
const _camOutLocal = new THREE.Vector3();
const _hitPoint = new THREE.Vector3();
const _hitNormal = new THREE.Vector3();
const _startPlaneHit = new THREE.Vector3();
const _currPlaneHit = new THREE.Vector3();
const _cursorDrag = new THREE.Vector3();
const _tangentDrag = new THREE.Vector3();
const _targetPullVec = new THREE.Vector3();
const _sentPullVec = new THREE.Vector3();
const _deltaVec = new THREE.Vector3();
const _lastPullHitPos = new THREE.Vector3(0, 0, 0.9);
const _lastPullDir = new THREE.Vector3(0, 0, 1);
const _snapPosScratch = new THREE.Vector3();
const _snapDirScratch = new THREE.Vector3();
const _snapNormScratch = new THREE.Vector3();
const _torqueVec = new THREE.Vector3();

const NICK_TRAVEL_THRESHOLD_PX = 6;

interface RubberAudioSynth {
  unlock: () => void;
  playSnap: (intensity: number) => void;
  playTwang: (pullAmount: number) => void;
  dispose: () => void;
}

/**
 * Zero-chrome procedural WebAudio synthesizer for dry latex snap cracks,
 * elastic whip twangs, and dense rubber ball core thumps.
 * Lazily unlocks on first pointer/key interaction and dispatches queued first-gesture snaps.
 */
function createRubberAudio(): RubberAudioSynth {
  let ctx: AudioContext | null = null;
  let masterOut: AudioNode | null = null;
  let noiseBuf: AudioBuffer | null = null;

  function ensureContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    const AudioCtor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtor) return null;
    if (!ctx) {
      try {
        ctx = new AudioCtor();
        const comp = ctx.createDynamicsCompressor();
        comp.threshold.setValueAtTime(-14, ctx.currentTime);
        comp.knee.setValueAtTime(12, ctx.currentTime);
        comp.ratio.setValueAtTime(8, ctx.currentTime);
        comp.attack.setValueAtTime(0.001, ctx.currentTime);
        comp.release.setValueAtTime(0.08, ctx.currentTime);
        comp.connect(ctx.destination);
        masterOut = comp;

        const len = Math.floor(ctx.sampleRate * 0.08);
        noiseBuf = ctx.createBuffer(1, len, ctx.sampleRate);
        const data = noiseBuf.getChannelData(0);
        for (let i = 0; i < len; i++) {
          data[i] = Math.random() * 2 - 1;
        }
      } catch {
        ctx = null;
        masterOut = null;
        return null;
      }
    }
    if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => {});
    }
    return ctx;
  }

  function scheduleSnap(ac: AudioContext, intensity: number): void {
    const out = masterOut ?? ac.destination;
    const now = ac.currentTime;
    const gainScale = Math.min(1.15, Math.max(0.45, intensity));

    // 1. Sharp dry latex whip-crack transient (high click + crisp snap body)
    if (noiseBuf) {
      const highSrc = ac.createBufferSource();
      highSrc.buffer = noiseBuf;
      const highBp = ac.createBiquadFilter();
      highBp.type = 'bandpass';
      highBp.frequency.setValueAtTime(3400, now);
      highBp.frequency.exponentialRampToValueAtTime(1350, now + 0.014);
      highBp.Q.value = 1.4;
      const highGain = ac.createGain();
      highGain.gain.setValueAtTime(0.42 * gainScale, now);
      highGain.gain.exponentialRampToValueAtTime(0.001, now + 0.016);
      highSrc.connect(highBp);
      highBp.connect(highGain);
      highGain.connect(out);
      highSrc.start(now);
      highSrc.stop(now + 0.02);

      const bodySrc = ac.createBufferSource();
      bodySrc.buffer = noiseBuf;
      const bodyBp = ac.createBiquadFilter();
      bodyBp.type = 'bandpass';
      bodyBp.frequency.setValueAtTime(1700, now);
      bodyBp.frequency.exponentialRampToValueAtTime(560, now + 0.028);
      bodyBp.Q.value = 2.0;
      const bodyGain = ac.createGain();
      bodyGain.gain.setValueAtTime(0.32 * gainScale, now);
      bodyGain.gain.exponentialRampToValueAtTime(0.001, now + 0.032);
      bodySrc.connect(bodyBp);
      bodyBp.connect(bodyGain);
      bodyGain.connect(out);
      bodySrc.start(now);
      bodySrc.stop(now + 0.036);
    }

    // 2. Elastic rubber whip twang (pitch-rebounding fundamental + overtone)
    const lp = ac.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(2600, now);
    lp.frequency.exponentialRampToValueAtTime(520, now + 0.11);
    lp.Q.value = 1.8;

    const twangGain = ac.createGain();
    twangGain.gain.setValueAtTime(0.001, now);
    twangGain.gain.linearRampToValueAtTime(0.2 * gainScale, now + 0.003);
    twangGain.gain.exponentialRampToValueAtTime(0.001, now + 0.115);

    const osc = ac.createOscillator();
    osc.type = 'triangle';
    const startHz = 420 + gainScale * 170;
    osc.frequency.setValueAtTime(startHz, now);
    osc.frequency.exponentialRampToValueAtTime(140, now + 0.045);
    osc.frequency.exponentialRampToValueAtTime(182, now + 0.095);

    const overOsc = ac.createOscillator();
    overOsc.type = 'sine';
    overOsc.frequency.setValueAtTime(startHz * 2.08, now);
    overOsc.frequency.exponentialRampToValueAtTime(290, now + 0.06);

    const overGain = ac.createGain();
    overGain.gain.setValueAtTime(0.35, now);

    osc.connect(lp);
    overOsc.connect(overGain);
    overGain.connect(lp);
    lp.connect(twangGain);
    twangGain.connect(out);

    osc.start(now);
    osc.stop(now + 0.12);
    overOsc.start(now);
    overOsc.stop(now + 0.07);

    // 3. Dense rubber core thump
    const thump = ac.createOscillator();
    thump.type = 'sine';
    thump.frequency.setValueAtTime(138, now);
    thump.frequency.exponentialRampToValueAtTime(46, now + 0.055);

    const thumpGain = ac.createGain();
    thumpGain.gain.setValueAtTime(0.24 * gainScale, now);
    thumpGain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

    thump.connect(thumpGain);
    thumpGain.connect(out);
    thump.start(now);
    thump.stop(now + 0.065);
  }

  function scheduleTwang(ac: AudioContext, pullAmount: number): void {
    const out = masterOut ?? ac.destination;
    const now = ac.currentTime;
    const amp = Math.min(0.22, 0.05 + pullAmount * 0.32);
    const baseHz = 165 + Math.min(240, pullAmount * 300);

    const osc = ac.createOscillator();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(baseHz, now);
    osc.frequency.exponentialRampToValueAtTime(baseHz * 0.72, now + 0.13);

    const harmonic = ac.createOscillator();
    harmonic.type = 'sine';
    harmonic.frequency.setValueAtTime(baseHz * 1.48, now);
    harmonic.frequency.exponentialRampToValueAtTime(baseHz * 1.05, now + 0.09);

    const harmGain = ac.createGain();
    harmGain.gain.setValueAtTime(0.3, now);

    const g = ac.createGain();
    g.gain.setValueAtTime(0.001, now);
    g.gain.linearRampToValueAtTime(amp, now + 0.004);
    g.gain.exponentialRampToValueAtTime(0.001, now + 0.145);

    osc.connect(g);
    harmonic.connect(harmGain);
    harmGain.connect(g);
    g.connect(out);

    osc.start(now);
    osc.stop(now + 0.15);
    harmonic.start(now);
    harmonic.stop(now + 0.1);
  }

  return {
    unlock: () => {
      ensureContext();
    },
    playSnap: (intensity: number) => {
      const ac = ensureContext();
      if (!ac) return;
      if (ac.state === 'running') {
        scheduleSnap(ac, intensity);
      } else {
        const queuedAt = performance.now();
        void ac.resume().then(() => {
          if (ac.state === 'running' && performance.now() - queuedAt < 90) {
            scheduleSnap(ac, intensity);
          }
        }).catch(() => {});
      }
    },
    playTwang: (pullAmount: number) => {
      if (pullAmount < 0.025) return;
      const ac = ensureContext();
      if (!ac) return;
      if (ac.state === 'running') {
        scheduleTwang(ac, pullAmount);
      } else {
        const queuedAt = performance.now();
        void ac.resume().then(() => {
          if (ac.state === 'running' && performance.now() - queuedAt < 90) {
            scheduleTwang(ac, pullAmount);
          }
        }).catch(() => {});
      }
    },
    dispose: () => {
      if (ctx) {
        void ctx.close().catch(() => {});
        ctx = null;
        masterOut = null;
        noiseBuf = null;
      }
    },
  };
}

export async function bootDay14Tension(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(14);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.15, 3.6],
    fov: 36,
    enableOrbit: true,
    enableShadows: true,
  });

  const sim = new RubberBandBallSim();
  const visuals = createRubberBallVisuals(sim.getBandSpecs());
  stage.specimenGroup.add(visuals.group);
  const rubberAudio = createRubberAudio();

  const specs = sim.getBandSpecs();
  const coreRadius =
    specs.length > 0
      ? Math.max(0.28, specs[0].baseRadius - specs[0].thickness * 0.5 - 0.008)
      : 0.36;
  const _coreSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), coreRadius + 0.02);

  const canvas = stage.shell.canvas;

  let hoveredBandId: number | null = null;
  let activeBandId: number | null = null;
  let activeUHit = 0;
  let activePointerId: number | null = null;
  let grabLockPointerId: number | null = null;
  let isOrbiting = false;
  let dragStartClientX = 0;
  let dragStartClientY = 0;
  let dragMoved = false;
  let prevSnapCount = 0;
  let lastPullBandId = -1;
  let lastPullStrength = 0.65;

  // Damped angular spring for crisp directional torque recoil when bands snap
  let rotX = 0;
  let rotY = 0;
  let rotZ = 0;
  let rotVelX = 0;
  let rotVelY = 0;
  let rotVelZ = 0;

  function computeLocalRay(clientX: number, clientY: number): boolean {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    _ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    _raycaster.setFromCamera(_ndc, stage.camera);

    visuals.group.updateMatrixWorld();
    _invGroupMat.copy(visuals.group.matrixWorld).invert();
    _localRay.copy(_raycaster.ray).applyMatrix4(_invGroupMat);

    _rayOriginTuple[0] = _localRay.origin.x;
    _rayOriginTuple[1] = _localRay.origin.y;
    _rayOriginTuple[2] = _localRay.origin.z;
    _rayDirTuple[0] = _localRay.direction.x;
    _rayDirTuple[1] = _localRay.direction.y;
    _rayDirTuple[2] = _localRay.direction.z;
    return true;
  }

  function clearPullState(keepOrbitLocked: boolean, pointerId?: number): void {
    activeBandId = null;
    activePointerId = null;
    if (!keepOrbitLocked) {
      grabLockPointerId = null;
      stage.controls.enabled = true;
      if (pointerId !== undefined) {
        try {
          if (canvas.hasPointerCapture(pointerId)) {
            canvas.releasePointerCapture(pointerId);
          }
        } catch {
          // Ignore if pointer capture already released
        }
      }
    }
    canvas.style.cursor = hoveredBandId !== null ? 'grab' : 'default';
  }

  const onPointerMove = (e: PointerEvent): void => {
    if (activeBandId !== null && e.pointerId === activePointerId) {
      const travel = Math.hypot(e.clientX - dragStartClientX, e.clientY - dragStartClientY);
      if (travel >= NICK_TRAVEL_THRESHOLD_PX) {
        dragMoved = true;
      }

      if (!computeLocalRay(e.clientX, e.clientY)) return;
      if (_localRay.intersectPlane(_dragPlane, _currPlaneHit)) {
        _cursorDrag.subVectors(_currPlaneHit, _startPlaneHit);
        const dragDist = _cursorDrag.length();

        if (dragDist > 1e-5) {
          const normalComp = _cursorDrag.dot(_hitNormal);
          _tangentDrag.copy(_cursorDrag).addScaledVector(_hitNormal, -normalComp);

          if (normalComp >= 0) {
            // Outward or tangent-plane pull: track cursor direction while ensuring positive surface lift
            const liftNormal = Math.max(normalComp, dragDist * 0.24);
            _targetPullVec.copy(_tangentDrag).addScaledVector(_hitNormal, liftNormal);
          } else {
            // Dragging inward across the ball center from a limb: redirect into camera-facing
            // front hemisphere + tangential swipe and clamp away from tunneling into the ball
            _targetPullVec.copy(_tangentDrag).addScaledVector(_camOutLocal, -normalComp * 0.55);
            const postDot = _targetPullVec.dot(_hitNormal);
            if (postDot < 0) {
              _targetPullVec.addScaledVector(_hitNormal, -postDot);
            }
            _targetPullVec.addScaledVector(_hitNormal, dragDist * 0.34);
            const targetLen = _targetPullVec.length();
            if (targetLen > 1e-5) {
              _targetPullVec.multiplyScalar((dragDist * 0.96) / targetLen);
            }
          }
        } else {
          _targetPullVec.set(0, 0, 0);
        }

        _deltaVec.subVectors(_targetPullVec, _sentPullVec);
        _sentPullVec.copy(_targetPullVec);

        _deltaTuple[0] = _deltaVec.x;
        _deltaTuple[1] = _deltaVec.y;
        _deltaTuple[2] = _deltaVec.z;

        const rt = sim.getBandRuntimes()[activeBandId];
        if (rt) {
          _lastPullDir.set(rt.pullDir[0], rt.pullDir[1], rt.pullDir[2]);
          lastPullStrength = 0.65 + Math.min(0.85, rt.pullAmount * 1.4);
        }

        const snapped = sim.updatePull(_deltaTuple);
        if (snapped) {
          hoveredBandId = null;
          // Keep OrbitControls disabled until pointerup/pointercancel so residual drag never orbits
          clearPullState(true, e.pointerId);
        }
      }
      return;
    }

    if (grabLockPointerId !== null || isOrbiting) return;
    if (!computeLocalRay(e.clientX, e.clientY)) return;

    const hit = sim.raycastBand(_rayOriginTuple, _rayDirTuple);
    hoveredBandId = hit ? hit.bandId : null;
    canvas.style.cursor = hoveredBandId !== null ? 'grab' : 'default';
  };

  const onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0) return;
    rubberAudio.unlock();
    if (!computeLocalRay(e.clientX, e.clientY)) return;

    const hit = sim.raycastBand(_rayOriginTuple, _rayDirTuple);
    if (hit) {
      // Intercept in capture phase before OrbitControls enters STATE.ROTATE
      grabLockPointerId = e.pointerId;
      stage.controls.enabled = false;
      e.stopImmediatePropagation();
      e.preventDefault();

      activeBandId = hit.bandId;
      activeUHit = hit.uHit;
      activePointerId = e.pointerId;
      dragStartClientX = e.clientX;
      dragStartClientY = e.clientY;
      dragMoved = false;

      _hitPoint.set(hit.hitPoint[0], hit.hitPoint[1], hit.hitPoint[2]);
      _hitNormal.copy(_hitPoint).normalize();
      if (_hitNormal.lengthSq() < 0.5) _hitNormal.set(0, 0, 1);

      lastPullBandId = hit.bandId;
      _lastPullHitPos.copy(_hitPoint);
      _lastPullDir.copy(_hitNormal);
      lastPullStrength = 0.62;

      _pullDirTuple[0] = _hitNormal.x;
      _pullDirTuple[1] = _hitNormal.y;
      _pullDirTuple[2] = _hitNormal.z;

      stage.camera.getWorldDirection(_camDir);
      _camDirLocal.copy(_camDir).transformDirection(_invGroupMat).normalize();
      _camOutLocal.copy(_camDirLocal).negate();

      _dragPlane.setFromNormalAndCoplanarPoint(_camDirLocal, _hitPoint);
      _startPlaneHit.copy(_hitPoint);
      _currPlaneHit.copy(_hitPoint);
      _sentPullVec.set(0, 0, 0);

      sim.startPull(hit.bandId, hit.uHit, _pullDirTuple);
      canvas.style.cursor = 'grabbing';
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Ignore synthetic events without active pointer
      }
      return;
    }

    // If all bands are gone and user clicks the exposed starter core, re-wrap a fresh ball
    if (sim.getDiagnostics().intactCount === 0 && _localRay.intersectsSphere(_coreSphere)) {
      grabLockPointerId = e.pointerId;
      stage.controls.enabled = false;
      e.stopImmediatePropagation();
      e.preventDefault();
      sim.reset();
      hoveredBandId = null;
      canvas.style.cursor = 'default';
      return;
    }

    if (grabLockPointerId === null) {
      isOrbiting = true;
    }
  };

  const onPointerUp = (e: PointerEvent): void => {
    rubberAudio.unlock();
    if (activeBandId !== null && e.pointerId === activePointerId) {
      if (e.type === 'pointerup' && !dragMoved) {
        lastPullBandId = activeBandId;
        lastPullStrength = 0.58;
        sim.nickBand(activeBandId, activeUHit);
        hoveredBandId = null;
      } else {
        const pullAmt = sim.getBandRuntimes()[activeBandId]?.pullAmount ?? 0;
        sim.releasePull();
        rubberAudio.playTwang(pullAmt);
      }
      clearPullState(false, e.pointerId);
      return;
    }
    if (grabLockPointerId !== null && e.pointerId === grabLockPointerId) {
      clearPullState(false, e.pointerId);
      return;
    }
    isOrbiting = false;
  };

  const onDblClick = (): void => {
    hoveredBandId = null;
    if (activeBandId !== null) {
      sim.releasePull();
    }
    clearPullState(false, grabLockPointerId ?? activePointerId ?? undefined);
    isOrbiting = false;
    sim.reset();
  };

  const onWindowBlur = (): void => {
    if (activeBandId !== null) {
      sim.releasePull();
    }
    if (activePointerId !== null || grabLockPointerId !== null) {
      clearPullState(false, grabLockPointerId ?? activePointerId ?? undefined);
    }
    isOrbiting = false;
  };

  const onKeyUnlock = (): void => {
    rubberAudio.unlock();
  };

  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerdown', onPointerDown, { capture: true });
  window.addEventListener('pointerup', onPointerUp, { capture: true });
  window.addEventListener('pointercancel', onPointerUp, { capture: true });
  window.addEventListener('blur', onWindowBlur);
  canvas.addEventListener('dblclick', onDblClick);
  window.addEventListener('keydown', onKeyUnlock, { passive: true });

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerdown', onPointerDown, { capture: true });
    window.removeEventListener('pointerup', onPointerUp, { capture: true });
    window.removeEventListener('pointercancel', onPointerUp, { capture: true });
    window.removeEventListener('blur', onWindowBlur);
    canvas.removeEventListener('dblclick', onDblClick);
    window.removeEventListener('keydown', onKeyUnlock);
    stage.controls.enabled = true;
    rubberAudio.dispose();
    visuals.dispose();
  });

  stage.onTick(({ dtSec }) => {
    sim.step(dtSec);
    const diag = sim.getDiagnostics();
    if (diag.snapCount > prevSnapCount) {
      const snapsDelta = Math.min(3, diag.snapCount - prevSnapCount);
      const snapInfo = (sim as unknown as SimWithSnapInfo).getLastSnapInfo?.() ?? null;

      let intensity = Math.max(0.68, lastPullStrength);
      let snapBandId = lastPullBandId;

      if (snapInfo) {
        _snapPosScratch.set(snapInfo.hitPos[0], snapInfo.hitPos[1], snapInfo.hitPos[2]);
        _snapDirScratch.set(snapInfo.pullDir[0], snapInfo.pullDir[1], snapInfo.pullDir[2]);
        _snapNormScratch.set(snapInfo.normal[0], snapInfo.normal[1], snapInfo.normal[2]);
        intensity = snapInfo.intensity;
        snapBandId = snapInfo.bandId;
      } else {
        _snapPosScratch.copy(_lastPullHitPos);
        _snapDirScratch.copy(_lastPullDir);
        const fallbackSpec = snapBandId >= 0 ? specs[snapBandId] : undefined;
        if (fallbackSpec) {
          _snapNormScratch.set(
            fallbackSpec.normal[0],
            fallbackSpec.normal[1],
            fallbackSpec.normal[2]
          );
        } else {
          _snapNormScratch.set(0, 1, 0);
        }
      }

      rubberAudio.playSnap(Math.min(1.15, intensity + (snapsDelta - 1) * 0.12));

      // Physical rotational torque jolt: hitPos x (-pullDir) kicks the ball away from the snap
      _torqueVec.crossVectors(_snapPosScratch, _snapDirScratch).negate();
      const crossMag = _torqueVec.length();
      if (crossMag < 0.25) {
        // Pure radial nicks twist cleanly along the severed band's wrap plane
        const sign = snapBandId % 2 === 0 ? 1 : -1;
        _torqueVec.addScaledVector(_snapNormScratch, 0.42 * sign);
      }

      const tLen = _torqueVec.length();
      if (tLen > 1e-5) {
        _torqueVec.divideScalar(tLen);
        const kick = (1.45 + intensity * 1.15) * (1 + (snapsDelta - 1) * 0.28);
        rotVelX += _torqueVec.x * kick;
        rotVelY += _torqueVec.y * kick;
        rotVelZ += _torqueVec.z * kick;
      }
    } else if (diag.snapCount < prevSnapCount) {
      // Reset occurred
      rotX = 0;
      rotY = 0;
      rotZ = 0;
      rotVelX = 0;
      rotVelY = 0;
      rotVelZ = 0;
    }
    prevSnapCount = diag.snapCount;

    const dt = Math.min(dtSec, 0.05);
    const rotK = 165.0;
    const rotDamp = 9.5;
    rotVelX += (-rotK * rotX - rotDamp * rotVelX) * dt;
    rotVelY += (-rotK * rotY - rotDamp * rotVelY) * dt;
    rotVelZ += (-rotK * rotZ - rotDamp * rotVelZ) * dt;
    rotX = Math.max(-0.36, Math.min(0.36, rotX + rotVelX * dt));
    rotY = Math.max(-0.36, Math.min(0.36, rotY + rotVelY * dt));
    rotZ = Math.max(-0.36, Math.min(0.36, rotZ + rotVelZ * dt));

    const recoil = sim.getBallRecoil();
    visuals.group.position.set(recoil[0], recoil[1], recoil[2]);
    visuals.group.rotation.set(rotX, rotY, rotZ);
    visuals.syncFromSim(
      specs,
      sim.getBandRuntimes(),
      sim.getDustPuffs(),
      sim,
      hoveredBandId,
      dtSec,
      stage.camera.quaternion
    );
  });

  window.__TENSION__ = {
    getState: () => sim.getDiagnostics(),
  };

  const baseGetState = window.__SPECIMEN_STAGE__?.getSceneState;
  if (window.__SPECIMEN_STAGE__ && baseGetState) {
    window.__SPECIMEN_STAGE__.getSceneState = () => ({
      ...baseGetState(),
      ...sim.getDiagnostics(),
    });
  }

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay14Tension();
}
