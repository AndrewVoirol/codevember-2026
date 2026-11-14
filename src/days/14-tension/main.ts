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

// Pre-allocated scratch objects for zero-GC pointer interaction
const _ndc = new THREE.Vector2();
const _raycaster = new THREE.Raycaster();
const _rayOriginTuple: [number, number, number] = [0, 0, 0];
const _rayDirTuple: [number, number, number] = [0, 0, 0];
const _pullDirTuple: [number, number, number] = [0, 0, 0];
const _deltaTuple: [number, number, number] = [0, 0, 0];
const _dragPlane = new THREE.Plane();
const _camDir = new THREE.Vector3();
const _hitPoint = new THREE.Vector3();
const _hitNormal = new THREE.Vector3();
const _startPlaneHit = new THREE.Vector3();
const _prevPlaneHit = new THREE.Vector3();
const _currPlaneHit = new THREE.Vector3();
const _planeDelta = new THREE.Vector3();
const _coreSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 0.3);

const NICK_TRAVEL_THRESHOLD_PX = 6;

interface RubberAudioSynth {
  unlock: () => void;
  playSnap: (intensity: number) => void;
  playTwang: (pullAmount: number) => void;
  dispose: () => void;
}

/**
 * Zero-chrome procedural WebAudio synthesizer for rubber-band snaps and twangs.
 * Lazily unlocks on first pointer gesture and degrades silently if AudioContext is unavailable.
 */
function createRubberAudio(): RubberAudioSynth {
  let ctx: AudioContext | null = null;
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
        const len = Math.floor(ctx.sampleRate * 0.06);
        noiseBuf = ctx.createBuffer(1, len, ctx.sampleRate);
        const data = noiseBuf.getChannelData(0);
        for (let i = 0; i < len; i++) {
          data[i] = Math.random() * 2 - 1;
        }
      } catch {
        ctx = null;
        return null;
      }
    }
    if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => {});
    }
    return ctx;
  }

  return {
    unlock: () => {
      ensureContext();
    },
    playSnap: (intensity: number) => {
      const ac = ensureContext();
      if (!ac || ac.state !== 'running') return;
      const now = ac.currentTime;
      const gainScale = Math.min(1.0, Math.max(0.35, intensity));

      // 1. Crisp latex crack transient (band-passed noise)
      if (noiseBuf) {
        const src = ac.createBufferSource();
        src.buffer = noiseBuf;
        const bp = ac.createBiquadFilter();
        bp.type = 'bandpass';
        bp.frequency.setValueAtTime(2400, now);
        bp.frequency.exponentialRampToValueAtTime(750, now + 0.025);
        bp.Q.value = 2.2;

        const g = ac.createGain();
        g.gain.setValueAtTime(0.28 * gainScale, now);
        g.gain.exponentialRampToValueAtTime(0.001, now + 0.028);

        src.connect(bp);
        bp.connect(g);
        g.connect(ac.destination);
        src.start(now);
        src.stop(now + 0.032);
      }

      // 2. Elastic whip twang
      const osc = ac.createOscillator();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(360 + gainScale * 140, now);
      osc.frequency.exponentialRampToValueAtTime(115, now + 0.11);

      const twangGain = ac.createGain();
      twangGain.gain.setValueAtTime(0.16 * gainScale, now);
      twangGain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

      osc.connect(twangGain);
      twangGain.connect(ac.destination);
      osc.start(now);
      osc.stop(now + 0.125);

      // 3. Low ball core thump
      const thump = ac.createOscillator();
      thump.type = 'sine';
      thump.frequency.setValueAtTime(105, now);
      thump.frequency.exponentialRampToValueAtTime(46, now + 0.06);

      const thumpGain = ac.createGain();
      thumpGain.gain.setValueAtTime(0.18 * gainScale, now);
      thumpGain.gain.exponentialRampToValueAtTime(0.001, now + 0.065);

      thump.connect(thumpGain);
      thumpGain.connect(ac.destination);
      thump.start(now);
      thump.stop(now + 0.07);
    },
    playTwang: (pullAmount: number) => {
      if (pullAmount < 0.025) return;
      const ac = ensureContext();
      if (!ac || ac.state !== 'running') return;
      const now = ac.currentTime;
      const amp = Math.min(0.18, pullAmount * 0.25);

      const osc = ac.createOscillator();
      osc.type = 'sine';
      const baseHz = 175 + Math.min(220, pullAmount * 260);
      osc.frequency.setValueAtTime(baseHz, now);
      osc.frequency.exponentialRampToValueAtTime(baseHz * 0.72, now + 0.14);

      const g = ac.createGain();
      g.gain.setValueAtTime(amp, now);
      g.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

      osc.connect(g);
      g.connect(ac.destination);
      osc.start(now);
      osc.stop(now + 0.155);
    },
    dispose: () => {
      if (ctx) {
        void ctx.close().catch(() => {});
        ctx = null;
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

  const canvas = stage.shell.canvas;

  let hoveredBandId: number | null = null;
  let activeBandId: number | null = null;
  let activeUHit = 0;
  let activePointerId: number | null = null;
  let isOrbiting = false;
  let dragStartClientX = 0;
  let dragStartClientY = 0;
  let dragMoved = false;
  let prevSnapCount = 0;

  // Damped angular spring for subtle rotational jolt when bands snap
  let rotX = 0;
  let rotY = 0;
  let rotZ = 0;
  let rotVelX = 0;
  let rotVelY = 0;
  let rotVelZ = 0;

  function computeRay(clientX: number, clientY: number): boolean {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    _ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    _raycaster.setFromCamera(_ndc, stage.camera);
    const o = _raycaster.ray.origin;
    const d = _raycaster.ray.direction;
    _rayOriginTuple[0] = o.x;
    _rayOriginTuple[1] = o.y;
    _rayOriginTuple[2] = o.z;
    _rayDirTuple[0] = d.x;
    _rayDirTuple[1] = d.y;
    _rayDirTuple[2] = d.z;
    return true;
  }

  function clearPullState(pointerId?: number): void {
    activeBandId = null;
    activePointerId = null;
    stage.controls.enabled = true;
    canvas.style.cursor = hoveredBandId !== null ? 'grab' : 'default';
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

  const onPointerMove = (e: PointerEvent): void => {
    if (activeBandId !== null && e.pointerId === activePointerId) {
      const travel = Math.hypot(e.clientX - dragStartClientX, e.clientY - dragStartClientY);
      if (travel >= NICK_TRAVEL_THRESHOLD_PX) {
        dragMoved = true;
      }

      if (!computeRay(e.clientX, e.clientY)) return;
      if (_raycaster.ray.intersectPlane(_dragPlane, _currPlaneHit)) {
        _planeDelta.subVectors(_currPlaneHit, _prevPlaneHit);
        const curDist = _currPlaneHit.distanceTo(_startPlaneHit);
        const prevDist = _prevPlaneHit.distanceTo(_startPlaneHit);
        _prevPlaneHit.copy(_currPlaneHit);

        // Combine direct radial projection (strong at limb) with camera-plane pull distance
        // (essential on front-facing center hits where _hitNormal is nearly parallel to _camDir).
        const radialProj = _planeDelta.dot(_hitNormal);
        const facing = Math.abs(_hitNormal.dot(_camDir));
        const desiredPullDelta = radialProj + facing * facing * (curDist - prevDist);

        _deltaTuple[0] = _hitNormal.x * desiredPullDelta;
        _deltaTuple[1] = _hitNormal.y * desiredPullDelta;
        _deltaTuple[2] = _hitNormal.z * desiredPullDelta;

        const snapped = sim.updatePull(_deltaTuple);
        if (snapped) {
          hoveredBandId = null;
          clearPullState(e.pointerId);
        }
      }
      return;
    }

    if (isOrbiting) return;
    if (!computeRay(e.clientX, e.clientY)) return;

    const hit = sim.raycastBand(_rayOriginTuple, _rayDirTuple);
    hoveredBandId = hit ? hit.bandId : null;
    canvas.style.cursor = hoveredBandId !== null ? 'grab' : 'default';
  };

  const onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0) return;
    rubberAudio.unlock();
    if (!computeRay(e.clientX, e.clientY)) return;

    const hit = sim.raycastBand(_rayOriginTuple, _rayDirTuple);
    if (hit) {
      activeBandId = hit.bandId;
      activeUHit = hit.uHit;
      activePointerId = e.pointerId;
      dragStartClientX = e.clientX;
      dragStartClientY = e.clientY;
      dragMoved = false;

      _hitPoint.set(hit.hitPoint[0], hit.hitPoint[1], hit.hitPoint[2]);
      _hitNormal.copy(_hitPoint).normalize();
      if (_hitNormal.lengthSq() < 0.5) _hitNormal.set(0, 0, 1);
      _pullDirTuple[0] = _hitNormal.x;
      _pullDirTuple[1] = _hitNormal.y;
      _pullDirTuple[2] = _hitNormal.z;

      stage.camera.getWorldDirection(_camDir);
      _dragPlane.setFromNormalAndCoplanarPoint(_camDir, _hitPoint);
      _startPlaneHit.copy(_hitPoint);
      _prevPlaneHit.copy(_hitPoint);

      sim.startPull(hit.bandId, hit.uHit, _pullDirTuple);
      stage.controls.enabled = false;
      canvas.style.cursor = 'grabbing';
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Ignore synthetic events without active pointer
      }
      return;
    }

    // If all bands are gone and user clicks the exposed starter core, re-wrap a fresh ball
    if (sim.getDiagnostics().intactCount === 0 && _raycaster.ray.intersectsSphere(_coreSphere)) {
      sim.reset();
      hoveredBandId = null;
      canvas.style.cursor = 'default';
      return;
    }

    isOrbiting = true;
  };

  const onPointerUp = (e: PointerEvent): void => {
    if (activeBandId !== null && e.pointerId === activePointerId) {
      if (e.type === 'pointerup' && !dragMoved) {
        sim.nickBand(activeBandId, activeUHit);
        hoveredBandId = null;
      } else {
        const pullAmt = sim.getBandRuntimes()[activeBandId]?.pullAmount ?? 0;
        sim.releasePull();
        rubberAudio.playTwang(pullAmt);
      }
      clearPullState(e.pointerId);
      return;
    }
    isOrbiting = false;
  };

  const onDblClick = (): void => {
    hoveredBandId = null;
    clearPullState();
    sim.reset();
  };

  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('dblclick', onDblClick);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerUp);
    canvas.removeEventListener('dblclick', onDblClick);
    rubberAudio.dispose();
    visuals.dispose();
  });

  stage.onTick(({ dtSec }) => {
    sim.step(dtSec);
    const diag = sim.getDiagnostics();
    if (diag.snapCount > prevSnapCount) {
      const snapsDelta = Math.min(3, diag.snapCount - prevSnapCount);
      rubberAudio.playSnap(0.65 + snapsDelta * 0.15);
      // Crisp rotational torque jolt on each snap
      const angleSeed = diag.snapCount * 2.39996;
      const kick = 0.55 * snapsDelta;
      rotVelX += Math.cos(angleSeed) * kick;
      rotVelY += Math.sin(angleSeed * 1.3) * kick;
      rotVelZ += Math.sin(angleSeed * 0.7) * kick * 0.6;
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
    const rotK = 210.0;
    const rotDamp = 16.0;
    rotVelX += (-rotK * rotX - rotDamp * rotVelX) * dt;
    rotVelY += (-rotK * rotY - rotDamp * rotVelY) * dt;
    rotVelZ += (-rotK * rotZ - rotDamp * rotVelZ) * dt;
    rotX += rotVelX * dt;
    rotY += rotVelY * dt;
    rotZ += rotVelZ * dt;

    const recoil = sim.getBallRecoil();
    visuals.group.position.set(recoil[0], recoil[1], recoil[2]);
    visuals.group.rotation.set(rotX, rotY, rotZ);
    visuals.syncFromSim(
      sim.getBandSpecs(),
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
