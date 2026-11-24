import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { createBoltStudioEnvironment } from './environment';
import {
  buildDeadboltAssembly,
  buildForendAssembly,
  buildKeyAssembly,
  buildLeverPackAssembly,
  buildLockCaseAssembly,
  buildOakSpecimenBlock,
  buildSpringPackAssembly,
} from './geometry';
import {
  BOLT_TRAVEL,
  KEY_X,
  KEY_Y,
  LEVER_COUNT,
  PIVOT_X,
  PIVOT_Y,
  LockMechanism,
  getLeverZ,
  type LockDiagnosticsState,
} from './lockMechanism';
import { createBoltMaterials } from './materials';

declare global {
  interface Window {
    __DAY21__?: {
      getState: () => LockDiagnosticsState;
    };
  }
}

type HitKind = 'none' | 'key' | 'bolt' | 'lever';

interface RaycastResult {
  kind: HitKind;
  leverIndex: number;
  point: THREE.Vector3 | null;
}

// Pre-allocated scratch math objects (strictly zero per-frame GC allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _plane = new THREE.Plane();
const _planeNormal = new THREE.Vector3();
const _planePoint = new THREE.Vector3();
const _planeHit = new THREE.Vector3();
const _localHit = new THREE.Vector3();
const _keyWorldCenter = new THREE.Vector3();
const _keyNdc = new THREE.Vector3();
const _camForward = new THREE.Vector3();
const _lockForward = new THREE.Vector3();
const _orbitOffset = new THREE.Vector3();
const _orbitSpherical = new THREE.Spherical();

function wrapDeltaPi(delta: number): number {
  let d = delta;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

async function mountDay21Bolt(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 21,
    cameraPosition: [1.52, 0.44, 3.74],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0.10, 0.01, 0.03);
  stage.controls.minDistance = 1.65;
  stage.controls.maxDistance = 7.8;
  stage.controls.maxPolarAngle = Math.PI * 0.88;
  stage.controls.minPolarAngle = 0.14;
  stage.controls.update();

  // Calibrate Horologist's Studio Softbox PMREM Environment for brushed brass, blued steel & bronze
  const studioEnv = createBoltStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize directional studio rig & upgrade shadow map to 2048x2048 tightly fitted to lock specimen
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(-2.3, 4.4, 4.6);
        child.intensity = 1.48;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.95;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.0;
        child.shadow.bias = -0.00025;
        child.shadow.normalBias = 0.01;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        child.color.setHex(0xdfe6ee);
        child.position.set(3.9, -0.7, 3.2);
        child.intensity = 0.72;
      } else if (dirIdx === 2) {
        child.position.set(-1.1, 3.4, -3.8);
        child.intensity = 1.38;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.42;
    }
  }

  // Build deterministic 5-lever Chubb mortise deadbolt physics & TSL materials
  const lock = new LockMechanism(true); // Boots locked so massive blued-steel deadbolt is proudly extended
  const materials = createBoltMaterials();

  // Root assembly allows crisp tactile recoil impulse without fighting OrbitControls
  const lockRoot = new THREE.Group();
  lockRoot.name = 'ChubbDeadboltSpecimenRoot';
  stage.specimenGroup.add(lockRoot);

  // Assemble all 7 precision mechanical sub-assemblies
  const oak = buildOakSpecimenBlock(materials);
  lockRoot.add(oak.group);

  const forend = buildForendAssembly(materials);
  lockRoot.add(forend.group);

  const lockCase = buildLockCaseAssembly(materials);
  lockRoot.add(lockCase.group);

  const deadbolt = buildDeadboltAssembly(materials);
  lockRoot.add(deadbolt.group);

  const leverPack = buildLeverPackAssembly(materials);
  lockRoot.add(leverPack.group);

  const springPack = buildSpringPackAssembly(materials);
  lockRoot.add(springPack.group);

  const key = buildKeyAssembly(materials);
  lockRoot.add(key.group);

  // Sync initial visual transforms immediately before first frame
  deadbolt.group.position.x = lock.boltThrow01 * BOLT_TRAVEL;
  key.group.rotation.z = -lock.keyAngleRad;
  springPack.update(lock.leverAnglesRad);

  // Expose read-only verification hook
  window.__DAY21__ = {
    getState: () => lock.getState(),
  };

  // Interactive pointer ergonomics (key circular drag/click/scroll, individual lever picking, bolt nudge, orbit)
  const canvas = stage.shell.canvas;
  let hoveredKind: HitKind = 'none';
  let hoveredLeverIndex = -1;

  let activeDrag: HitKind | 'orbit-fallback' = 'none';
  let activeLeverIndex = -1;
  let leverDragPhase: 'none' | 'pending' | 'picking' = 'none';
  let boltDragPhase: 'none' | 'pending' | 'sliding' = 'none';
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTime = 0;
  let prevOrbitX = 0;
  let prevOrbitY = 0;

  let prevKeyPointerAngle = 0;
  let keyScreenSign = 1;

  let leverDragStartPointerAngle = 0;
  let leverDragStartAngle = 0;

  let boltDragStartLocalX = 0;
  let boltDragStartThrow01 = 0;

  function applyManualOrbitStep(deltaPxX: number, deltaPxY: number): void {
    const h = Math.max(1, canvas.clientHeight || window.innerHeight);
    const rotScale = (2 * Math.PI) / h;
    _orbitOffset.copy(stage.camera.position).sub(stage.controls.target);
    _orbitSpherical.setFromVector3(_orbitOffset);
    _orbitSpherical.theta -= deltaPxX * rotScale * 0.92;
    _orbitSpherical.phi = Math.max(
      stage.controls.minPolarAngle,
      Math.min(stage.controls.maxPolarAngle, _orbitSpherical.phi - deltaPxY * rotScale * 0.92)
    );
    _orbitOffset.setFromSpherical(_orbitSpherical);
    stage.camera.position.copy(stage.controls.target).add(_orbitOffset);
    stage.camera.lookAt(stage.controls.target);
  }

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastSpecimen(clientX: number, clientY: number): RaycastResult {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    const keyHits = _raycaster.intersectObjects(key.hitMeshes, false);
    if (keyHits.length > 0) {
      return { kind: 'key', leverIndex: -1, point: keyHits[0].point };
    }

    const leverHits = _raycaster.intersectObjects(leverPack.leverMeshes, false);
    if (leverHits.length > 0) {
      const idx = (leverHits[0].object.userData.leverIndex as number | undefined) ?? -1;
      return { kind: 'lever', leverIndex: idx, point: leverHits[0].point };
    }

    const boltHits = _raycaster.intersectObjects(deadbolt.hitMeshes, false);
    if (boltHits.length > 0) {
      return { kind: 'bolt', leverIndex: -1, point: boltHits[0].point };
    }

    return { kind: 'none', leverIndex: -1, point: null };
  }

  function intersectLockLocalPlane(
    clientX: number,
    clientY: number,
    localZ: number,
    outLocal: THREE.Vector3
  ): boolean {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);
    lockRoot.updateMatrixWorld();

    _planeNormal.set(0, 0, 1).transformDirection(lockRoot.matrixWorld).normalize();
    _planePoint.set(0, 0, localZ).applyMatrix4(lockRoot.matrixWorld);
    _plane.setFromNormalAndCoplanarPoint(_planeNormal, _planePoint);

    if (!_raycaster.ray.intersectPlane(_plane, _planeHit)) {
      return false;
    }
    outLocal.copy(_planeHit);
    lockRoot.worldToLocal(outLocal);
    return true;
  }

  function computeKeyScreenAngle(clientX: number, clientY: number): number {
    const rect = canvas.getBoundingClientRect();
    lockRoot.updateMatrixWorld();
    _keyWorldCenter.set(KEY_X, KEY_Y, 0.55).applyMatrix4(lockRoot.matrixWorld);
    _keyNdc.copy(_keyWorldCenter).project(stage.camera);
    const sx = rect.left + (_keyNdc.x * 0.5 + 0.5) * rect.width;
    const sy = rect.top + (-_keyNdc.y * 0.5 + 0.5) * rect.height;
    return Math.atan2(clientY - sy, clientX - sx);
  }

  function computeKeyViewSign(): number {
    lockRoot.updateMatrixWorld();
    _lockForward.set(0, 0, 1).transformDirection(lockRoot.matrixWorld);
    stage.camera.getWorldDirection(_camForward);
    // Viewing front face (+Z) means camera looks along -Z -> dot < 0 -> screen CW == lock CW (+1)
    return _lockForward.dot(_camForward) <= 0 ? 1 : -1;
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    prevOrbitX = ev.clientX;
    prevOrbitY = ev.clientY;
    pointerDownTime = performance.now();

    const hit = raycastSpecimen(ev.clientX, ev.clientY);
    activeDrag = hit.kind;
    activeLeverIndex = hit.leverIndex;
    leverDragPhase = 'none';
    boltDragPhase = 'none';

    if (hit.kind === 'key') {
      stage.controls.enabled = false;
      lock.beginKeyDrag();
      prevKeyPointerAngle = computeKeyScreenAngle(ev.clientX, ev.clientY);
      keyScreenSign = computeKeyViewSign();
      canvas.style.cursor = 'grabbing';
      canvas.setPointerCapture(ev.pointerId);
    } else if (hit.kind === 'lever' && hit.leverIndex >= 0) {
      stage.controls.enabled = false;
      const zPlane = getLeverZ(hit.leverIndex);
      if (intersectLockLocalPlane(ev.clientX, ev.clientY, zPlane, _localHit)) {
        leverDragStartPointerAngle = Math.atan2(_localHit.y - PIVOT_Y, _localHit.x - PIVOT_X);
        leverDragStartAngle = lock.leverAnglesRad[hit.leverIndex];
      } else {
        leverDragStartAngle = lock.leverAnglesRad[hit.leverIndex];
      }
      // If lever is already lifted, pick directly; if resting at 0 deg, wait for directional intent
      // so dragging downward/diagonally seamlessly orbits instead of dead-stopping against the 0-deg pin!
      if (leverDragStartAngle > 0.015) {
        leverDragPhase = 'picking';
        lock.beginLeverDrag(hit.leverIndex, leverDragStartAngle);
      } else {
        leverDragPhase = 'pending';
      }
      canvas.style.cursor = 'grabbing';
      canvas.setPointerCapture(ev.pointerId);
    } else if (hit.kind === 'bolt') {
      stage.controls.enabled = false;
      if (intersectLockLocalPlane(ev.clientX, ev.clientY, 0, _localHit)) {
        boltDragStartLocalX = _localHit.x;
        boltDragStartThrow01 = lock.boltThrow01;
      } else {
        boltDragStartThrow01 = lock.boltThrow01;
      }
      boltDragPhase = 'pending';
      canvas.style.cursor = 'grabbing';
      canvas.setPointerCapture(ev.pointerId);
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (activeDrag === 'orbit-fallback') {
      const stepX = ev.clientX - prevOrbitX;
      const stepY = ev.clientY - prevOrbitY;
      prevOrbitX = ev.clientX;
      prevOrbitY = ev.clientY;
      applyManualOrbitStep(stepX, stepY);
      return;
    }

    if (activeDrag === 'key') {
      const curAngle = computeKeyScreenAngle(ev.clientX, ev.clientY);
      const delta = wrapDeltaPi(curAngle - prevKeyPointerAngle);
      prevKeyPointerAngle = curAngle;
      // Avoid tiny radius jump near exact keyhole pixel center by clamping extreme single-event delta
      const clampedDelta = Math.max(-0.55, Math.min(0.55, delta)) * keyScreenSign;
      lock.applyKeyDragDelta(clampedDelta);
      return;
    }

    if (activeDrag === 'lever' && activeLeverIndex >= 0) {
      if (leverDragPhase === 'pending') {
        const dx = ev.clientX - pointerDownX;
        const dy = ev.clientY - pointerDownY;
        const dist = Math.hypot(dx, dy);
        if (dist < 5) return;

        // Upward drag (dy < 0 and predominantly vertical) lifts that exact brass lever;
        // downward or sideways sweep across a resting lever smoothly falls back to 3D orbit!
        if (dy < -3 && Math.abs(dy) >= Math.abs(dx) * 0.65) {
          leverDragPhase = 'picking';
          lock.beginLeverDrag(activeLeverIndex, leverDragStartAngle);
        } else {
          activeDrag = 'orbit-fallback';
          leverDragPhase = 'none';
          activeLeverIndex = -1;
          prevOrbitX = ev.clientX;
          prevOrbitY = ev.clientY;
          applyManualOrbitStep(dx, dy);
          return;
        }
      }

      const zPlane = getLeverZ(activeLeverIndex);
      if (intersectLockLocalPlane(ev.clientX, ev.clientY, zPlane, _localHit)) {
        const curPointerAng = Math.atan2(_localHit.y - PIVOT_Y, _localHit.x - PIVOT_X);
        const relDelta = wrapDeltaPi(curPointerAng - leverDragStartPointerAngle);
        lock.setLeverDragTarget(leverDragStartAngle + relDelta);
      }
      return;
    }

    if (activeDrag === 'bolt') {
      if (boltDragPhase === 'pending') {
        const dx = ev.clientX - pointerDownX;
        const dy = ev.clientY - pointerDownY;
        const dist = Math.hypot(dx, dy);
        if (dist < 5) return;

        // Pushing further outward when already fully locked (or predominantly vertical drag) orbits smoothly;
        // pushing/pulling toward active travel slides or tests bolt-stump lever blocking!
        const pushingPastLockStop = lock.boltThrow01 >= 0.99 && dx > 4;
        const pushingPastUnlockStop = lock.boltThrow01 <= 0.01 && dx < -4;
        const predominantlyVertical = Math.abs(dy) > Math.abs(dx) * 1.25;
        if (pushingPastLockStop || pushingPastUnlockStop || predominantlyVertical) {
          activeDrag = 'orbit-fallback';
          boltDragPhase = 'none';
          prevOrbitX = ev.clientX;
          prevOrbitY = ev.clientY;
          applyManualOrbitStep(dx, dy);
          return;
        }
        boltDragPhase = 'sliding';
        lock.beginBoltDrag();
      }

      if (intersectLockLocalPlane(ev.clientX, ev.clientY, 0, _localHit)) {
        const dx = _localHit.x - boltDragStartLocalX;
        lock.setBoltDragTarget(boltDragStartThrow01 + dx / BOLT_TRAVEL);
      }
      return;
    }

    if (ev.buttons !== 0) return;

    const hit = raycastSpecimen(ev.clientX, ev.clientY);
    hoveredKind = hit.kind;
    hoveredLeverIndex = hit.leverIndex;

    if (hit.kind === 'key') {
      canvas.style.cursor = 'grab';
    } else if (hit.kind === 'lever') {
      canvas.style.cursor = 'ns-resize';
    } else if (hit.kind === 'bolt') {
      canvas.style.cursor = 'ew-resize';
    } else {
      canvas.style.cursor = 'default';
    }
  };

  const releaseCaptureSafe = (ev: PointerEvent) => {
    try {
      canvas.releasePointerCapture(ev.pointerId);
    } catch {
      // Ignore if pointer capture was already released
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    const moveDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTime;
    const isCleanClick = moveDist < 7 && heldMs < 400;

    if (activeDrag === 'key') {
      lock.endKeyDrag();
      stage.controls.enabled = true;
      releaseCaptureSafe(ev);
      if (isCleanClick) {
        lock.triggerToggleTurn();
      }
      canvas.style.cursor = 'grab';
    } else if (activeDrag === 'lever') {
      if (leverDragPhase === 'picking') {
        lock.endLeverDrag();
      } else if (isCleanClick && activeLeverIndex >= 0) {
        // Tactile quick spring-pluck impulse when clicking a lever directly
        lock.leverAngVels[activeLeverIndex] = Math.max(lock.leverAngVels[activeLeverIndex], 2.4);
      }
      stage.controls.enabled = true;
      releaseCaptureSafe(ev);
      canvas.style.cursor = 'ns-resize';
    } else if (activeDrag === 'bolt') {
      if (boltDragPhase === 'sliding') {
        lock.endBoltDrag();
      }
      stage.controls.enabled = true;
      releaseCaptureSafe(ev);
      canvas.style.cursor = 'ew-resize';
    } else if (activeDrag === 'orbit-fallback') {
      stage.controls.enabled = true;
      stage.controls.update();
      releaseCaptureSafe(ev);
      canvas.style.cursor = 'default';
    }

    activeDrag = 'none';
    activeLeverIndex = -1;
    leverDragPhase = 'none';
    boltDragPhase = 'none';
  };

  const onPointerCancel = (ev: PointerEvent) => {
    if (activeDrag === 'key') lock.endKeyDrag();
    if (activeDrag === 'lever' && leverDragPhase === 'picking') lock.endLeverDrag();
    if (activeDrag === 'bolt' && boltDragPhase === 'sliding') lock.endBoltDrag();
    activeDrag = 'none';
    activeLeverIndex = -1;
    leverDragPhase = 'none';
    boltDragPhase = 'none';
    stage.controls.enabled = true;
    stage.controls.update();
    releaseCaptureSafe(ev);
    canvas.style.cursor = 'default';
  };

  const onWheel = (ev: WheelEvent) => {
    if (hoveredKind === 'key' && activeDrag === 'none') {
      ev.preventDefault();
      ev.stopPropagation();
      const sign = computeKeyViewSign();
      const delta = Math.max(-0.42, Math.min(0.42, ev.deltaY * 0.0045)) * sign;
      lock.nudgeKeyAngle(delta);
    }
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      lock.triggerToggleTurn();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerCancel);
  canvas.addEventListener('wheel', onWheel, { passive: false, capture: true });
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerCancel);
    canvas.removeEventListener('wheel', onWheel, { capture: true });
    window.removeEventListener('keydown', onKeyDown);

    stage.scene.environment = null;
    studioEnv.dispose();
    oak.dispose();
    forend.dispose();
    lockCase.dispose();
    deadbolt.dispose();
    leverPack.dispose();
    springPack.dispose();
    key.dispose();
    materials.dispose();
  });

  // Main animation tick (strictly zero per-frame allocations)
  stage.onTick(({ dtSec }) => {
    lock.step(dtSec);

    // 1. Subtle physical recoil tremor when heavy deadbolt strikes end stop or levers snap home
    lockRoot.position.x = lock.caseRecoilX;

    // 2. Drive key rotation (-keyAngleRad maps phi=0 down -> CW around +Z view)
    key.group.rotation.z = -lock.keyAngleRad;

    // 3. Drive heavy blued-steel deadbolt translation through brass forend slot
    deadbolt.group.position.x = lock.boltThrow01 * BOLT_TRAVEL;

    // 4. Drive all 5 independent brass lever pivots & their flexing phosphor-bronze leaf springs
    for (let i = 0; i < LEVER_COUNT; i++) {
      leverPack.leverPivots[i].rotation.z = lock.leverAnglesRad[i];

      const hoverGoal =
        (activeDrag === 'lever' && activeLeverIndex === i) ||
        (activeDrag === 'none' && hoveredKind === 'lever' && hoveredLeverIndex === i)
          ? 1
          : 0;
      const curH = materials.uLeverHovers[i].value;
      materials.uLeverHovers[i].value = curH + (hoverGoal - curH) * Math.min(1, dtSec * 14);
      materials.uLeverBinds[i].value = lock.leverBinding01[i];
    }
    springPack.update(lock.leverAnglesRad);

    // 5. Smooth hover burnish lifts for key bow & blued-steel deadbolt
    const keyHoverGoal = activeDrag === 'key' || (activeDrag === 'none' && hoveredKind === 'key') ? 1 : 0;
    materials.uKeyHover.value += (keyHoverGoal - materials.uKeyHover.value) * Math.min(1, dtSec * 14);

    const boltHoverGoal = activeDrag === 'bolt' || (activeDrag === 'none' && hoveredKind === 'bolt') ? 1 : 0;
    materials.uBoltHover.value += (boltHoverGoal - materials.uBoltHover.value) * Math.min(1, dtSec * 14);
  });
}

mountDay21Bolt().catch((err) => {
  console.error('[Day 21 - Bolt] Failed to initialize WebGPU specimen:', err);
});
