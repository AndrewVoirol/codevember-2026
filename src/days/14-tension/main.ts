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

  const canvas = stage.shell.canvas;

  let hoveredBandId: number | null = null;
  let activeBandId: number | null = null;
  let activeUHit = 0;
  let activePointerId: number | null = null;
  let isOrbiting = false;
  let dragStartClientX = 0;
  let dragStartClientY = 0;
  let dragMoved = false;

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
        sim.releasePull();
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
    visuals.dispose();
  });

  stage.onTick(({ dtSec }) => {
    sim.step(dtSec);
    visuals.syncFromSim(
      sim.getBandSpecs(),
      sim.getBandRuntimes(),
      sim.getDustPuffs(),
      sim,
      hoveredBandId,
      dtSec
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
