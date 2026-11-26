import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  AnamorphController,
  INITIAL_CAMERA_POS,
  STAGE_FOV_DEG,
  VANTAGE_TARGET,
  computeSightRingPose,
  type Day26Diagnostics,
  type Vec3Tuple,
} from './anamorphMath';
import {
  buildArchitecturalCourtyardGeometry,
  buildSurveyorSightRing,
} from './architectureGeometry';
import { createAnamorphStudioEnvironment } from './environment';
import { createAnamorphMaterials } from './materials';

declare global {
  interface Window {
    __DAY26__?: {
      getState: () => Day26Diagnostics;
    };
  }
}

// Pre-allocated scratch math objects (strictly zero per-frame allocations in onTick)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _ringHitSphere = new THREE.Sphere(new THREE.Vector3(), 0.165);
const _ringLookTarget = new THREE.Vector3();
const _liveCamTuple: Vec3Tuple = [0, 0, 0];
const _stepCamOut: Vec3Tuple = [0, 0, 0];
const _pointerHitTarget = new THREE.Vector3();

async function mountDay26Anamorph(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 26,
    cameraPosition: [...INITIAL_CAMERA_POS],
    fov: STAGE_FOV_DEG,
    enableOrbit: true,
    enableShadows: true,
  });

  // Smooth 360-degree architectural turntable orbit + close-up macro inspection inside courtyard
  stage.camera.near = 0.04;
  stage.camera.updateProjectionMatrix();
  stage.controls.target.set(VANTAGE_TARGET[0], VANTAGE_TARGET[1], VANTAGE_TARGET[2]);
  stage.controls.minDistance = 1.05;
  stage.controls.maxDistance = 8.8;
  stage.controls.minPolarAngle = 0.12;
  stage.controls.maxPolarAngle = Math.PI * 0.88;
  stage.controls.dampingFactor = 0.075;
  stage.controls.update();

  // Calibrated museum-gallery softbox PMREM environment for satin brass, honed travertine & paint sheen
  const studioEnv = createAnamorphStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 0.90;

  // Tune studio 3-point raking rig & tighten 4096x4096 soft PCF shadow frustum around the architectural model
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.color.setHex(0xf9f3e7);
        child.position.set(3.7, 4.5, 2.7);
        child.intensity = 2.28;
        child.shadow.mapSize.set(2048, 2048);
        child.shadow.radius = 2.0;
        const d = 1.68;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.5;
        child.shadow.camera.far = 12.5;
        child.shadow.bias = -0.00018;
        child.shadow.normalBias = 0.006;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        child.color.setHex(0xc4d4e4);
        child.position.set(-3.9, 1.3, 2.5);
        child.intensity = 0.56;
      } else if (dirIdx === 2) {
        child.color.setHex(0xf3efe6);
        child.position.set(-1.6, 3.6, -4.1);
        child.intensity = 1.38;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.44;
    }
  }

  // Build pure projective controller, TSL materials, brutalist courtyard, and brass sight-ring
  const controller = new AnamorphController();
  const materials = createAnamorphMaterials();
  materials.syncVantageBasis(controller.vantageBasis);
  materials.syncStencil(controller.activeStencilIndex);

  const courtyardGeo = buildArchitecturalCourtyardGeometry();
  const courtyardMesh = new THREE.Mesh(courtyardGeo, materials.courtyardMaterial);
  courtyardMesh.name = 'BrutalistCourtyardMaquette';
  courtyardMesh.castShadow = true;
  courtyardMesh.receiveShadow = true;
  stage.specimenGroup.add(courtyardMesh);

  const sightRing = buildSurveyorSightRing(
    materials.brassBezelMaterial,
    materials.brassTicksMaterial
  );
  stage.specimenGroup.add(sightRing.group);

  function syncSightRingTransform(): void {
    const pose = computeSightRingPose(controller.vantageBasis);
    sightRing.group.position.set(pose.position[0], pose.position[1], pose.position[2]);
    _ringLookTarget.set(pose.lookTarget[0], pose.lookTarget[1], pose.lookTarget[2]);
    sightRing.group.lookAt(_ringLookTarget);
    _ringHitSphere.center.copy(sightRing.group.position);
  }
  syncSightRingTransform();

  // Expose read-only verification hook matching Day 26 contract
  window.__DAY26__ = {
    getState: () => controller.getState(),
  };

  if (window.__SPECIMEN_STAGE__) {
    const baseGetState = window.__SPECIMEN_STAGE__.getSceneState;
    window.__SPECIMEN_STAGE__.getSceneState = () => {
      const diag = controller.getState();
      return {
        ...baseGetState(),
        alignmentErrorDeg: diag.alignmentErrorDeg,
        isLockedIn: diag.isLockedIn,
        activeStencilIndex: diag.activeStencilIndex,
        projectionCount: diag.projectionCount,
        vantagePos: diag.vantagePos,
      };
    };
  }

  // Interactive controls: opens dead-locked on the optical illusion, immediately reveals
  // depth fracture & stone-bonded paint on intentional micro-oblique hover, drag, or zoom;
  // click building (or Space) to project fresh stencil from any angle; click brass ring /
  // double-click / R to glide smoothly back into exact lock-in.
  const canvas = stage.shell.canvas;
  const bootTimeMs = performance.now();
  let hasIntentionalHover = false;
  let hoverAnchorX: number | null = null;
  let hoverAnchorY: number | null = null;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownMs = 0;
  let isPointerDown = false;
  let hoveredTarget: 'none' | 'building' | 'ring' = 'none';
  let hasHoverHit = false;
  let lockGlowSmoothed = 1.0;

  function rearmHoverGuard(): void {
    hasIntentionalHover = false;
    hoverAnchorX = null;
    hoverAnchorY = null;
    controller.clearHover();
  }

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastTarget(clientX: number, clientY: number): 'none' | 'building' | 'ring' {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    // Check visible brass optical sight-ring first
    if (sightRing.group.visible && controller.sightRingAlpha > 0.15) {
      if (_raycaster.ray.intersectsSphere(_ringHitSphere)) {
        return 'ring';
      }
    }

    const hits = _raycaster.intersectObject(courtyardMesh, false);
    if (hits.length > 0) {
      _pointerHitTarget.copy(hits[0].point);
      return 'building';
    }
    return 'none';
  }

  function performLiveProjection(): void {
    const cam = stage.camera.position;
    controller.projectFromViewpoint([cam.x, cam.y, cam.z], performance.now());
    materials.syncVantageBasis(controller.vantageBasis);
    materials.syncStencil(controller.activeStencilIndex);
    syncSightRingTransform();
    rearmHoverGuard();
  }

  function performLockInGlide(): void {
    const cam = stage.camera.position;
    controller.triggerLockInGlide([cam.x, cam.y, cam.z]);
    rearmHoverGuard();
  }

  const onControlsStart = () => {
    controller.notifyDragStart();
  };
  const onControlsEnd = () => {
    controller.notifyDragEnd();
  };
  stage.controls.addEventListener('start', onControlsStart);
  stage.controls.addEventListener('end', onControlsEnd);

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    isPointerDown = true;
    hasIntentionalHover = true;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    pointerDownMs = performance.now();
    controller.notifyDragStart();
    canvas.style.cursor = 'grabbing';
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (isPointerDown) {
      canvas.style.cursor = 'grabbing';
      return;
    }
    // Initial-load cursor & window-activation parallax guard (AGENTS.md §3.6):
    // keep Frame 0 locked at 0.000 deg until >= 8 px of intentional pointer travel after 350 ms settle.
    if (!hasIntentionalHover) {
      if (performance.now() - bootTimeMs < 350) {
        hoverAnchorX = ev.clientX;
        hoverAnchorY = ev.clientY;
      } else if (hoverAnchorX === null || hoverAnchorY === null) {
        hoverAnchorX = ev.clientX;
        hoverAnchorY = ev.clientY;
      } else if (Math.hypot(ev.clientX - hoverAnchorX, ev.clientY - hoverAnchorY) >= 8) {
        hasIntentionalHover = true;
      }
    }

    const hit = raycastTarget(ev.clientX, ev.clientY);
    if (hasIntentionalHover) {
      controller.setHoverNdc(_ndc.x, _ndc.y);
    }
    hoveredTarget = hit;
    hasHoverHit = hit === 'building';
    if (hit === 'ring') {
      canvas.style.cursor = 'pointer';
    } else if (hit === 'building') {
      canvas.style.cursor = 'crosshair';
    } else {
      canvas.style.cursor = 'grab';
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (!isPointerDown) return;
    isPointerDown = false;
    controller.notifyDragEnd();

    const moveDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownMs;
    const isCleanClick = moveDist < 7 && heldMs < 400;

    const hit = raycastTarget(ev.clientX, ev.clientY);
    hoveredTarget = hit;
    hasHoverHit = hit === 'building';

    if (isCleanClick) {
      if (hit === 'ring') {
        performLockInGlide();
      } else if (hit === 'building') {
        performLiveProjection();
      }
    }

    canvas.style.cursor =
      hit === 'ring' ? 'pointer' : hit === 'building' ? 'crosshair' : 'grab';
  };

  const onPointerLeave = () => {
    hoveredTarget = 'none';
    hasHoverHit = false;
    controller.clearHover();
  };

  const onDblClick = () => {
    // If the user double-clicked directly on the building, roll back the speculative
    // first-click re-projection before gliding smoothly into exact optical lock-in!
    const rolledBack = controller.rollbackRecentProjectionIfWithin(performance.now(), 380);
    if (rolledBack) {
      materials.syncVantageBasis(controller.vantageBasis);
      materials.syncStencil(controller.activeStencilIndex);
      syncSightRingTransform();
    }
    performLockInGlide();
  };

  const onWheel = () => {
    hasIntentionalHover = true;
    controller.notifyWheel();
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      performLiveProjection();
    } else if (ev.key === 'r' || ev.key === 'R') {
      ev.preventDefault();
      performLockInGlide();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('dblclick', onDblClick);
  canvas.addEventListener('wheel', onWheel, { passive: true });
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    stage.controls.removeEventListener('start', onControlsStart);
    stage.controls.removeEventListener('end', onControlsEnd);
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    canvas.removeEventListener('dblclick', onDblClick);
    canvas.removeEventListener('wheel', onWheel);
    window.removeEventListener('keydown', onKeyDown);

    stage.scene.environment = null;
    studioEnv.dispose();
    courtyardGeo.dispose();
    sightRing.dispose();
    materials.dispose();
  });

  // Main zero-GC animation tick
  stage.onTick(({ dtSec, elapsedSec }) => {
    _liveCamTuple[0] = stage.camera.position.x;
    _liveCamTuple[1] = stage.camera.position.y;
    _liveCamTuple[2] = stage.camera.position.z;

    const camModified = controller.step(dtSec, _liveCamTuple, _stepCamOut);
    if (camModified) {
      stage.camera.position.set(_stepCamOut[0], _stepCamOut[1], _stepCamOut[2]);
      stage.camera.lookAt(stage.controls.target);
    }

    // Update projection scan-flash & subtle optical lock-in resonance
    materials.uFlashProgress.value = controller.flashProgress;
    const lockGoal = controller.isLockedIn ? 1.0 : 0.0;
    lockGlowSmoothed += (lockGoal - lockGlowSmoothed) * Math.min(1, dtSec * 8.0);
    materials.uLockGlow.value = lockGlowSmoothed;

    // Smooth pointer grazing warmth on bare concrete
    const ptrActiveGoal = hasHoverHit && !isPointerDown ? 1.0 : 0.0;
    materials.uPointerActive.value +=
      (ptrActiveGoal - materials.uPointerActive.value) * Math.min(1, dtSec * 10.0);
    if (hasHoverHit) {
      materials.uPointerWorld.value.lerp(_pointerHitTarget, Math.min(1, dtSec * 18.0));
    }

    // Machined brass optical surveyor's sight-ring: subtle hover burnish & smooth dissolve near lock-in
    const ringHoverGoal = hoveredTarget === 'ring' && !isPointerDown ? 1.0 : 0.0;
    materials.uRingHover.value +=
      (ringHoverGoal - materials.uRingHover.value) * Math.min(1, dtSec * 12.0);

    const alpha = controller.sightRingAlpha;
    materials.uRingAlpha.value = alpha;
    sightRing.group.visible = alpha > 0.015;
    if (sightRing.group.visible) {
      // Gentle precision optical scale dilation as viewer approaches the brass eyepiece
      const approach = 1.0 - alpha;
      const hoverPulse = 1.0 + materials.uRingHover.value * 0.07 + Math.sin(elapsedSec * 2.4) * 0.012;
      sightRing.group.scale.setScalar((1.0 + approach * 0.45) * hoverPulse);
      const writeDepth = alpha > 0.65;
      materials.brassBezelMaterial.depthWrite = writeDepth;
      materials.brassTicksMaterial.depthWrite = writeDepth;
    }
  });
}

mountDay26Anamorph().catch((err) => {
  console.error('[Day 26 - Anamorph] Failed to initialize WebGPU specimen:', err);
});
