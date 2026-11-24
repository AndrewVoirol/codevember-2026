import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { ECHO_BOUNCE_COUNT, MAX_MIRROR_TILT_RAD } from './optics';
import { EchoSimulation } from './physics';
import { buildEchoCubeAssembly } from './geometry';
import { createEchoMaterials, createEchoStudioEnvironment } from './materials';

declare global {
  interface Window {
    __DAY22__?: {
      getState: () => {
        mirrorTiltDeg: [number, number];
        corePos: [number, number, number];
        coreVel: [number, number, number];
        bounceCount: number;
        presetIndex: number;
      };
    };
  }
}

// Pre-allocated scratch math objects (strictly zero per-frame heap allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _camRight = new THREE.Vector3();
const _camUp = new THREE.Vector3();
const _camForward = new THREE.Vector3();
const _localCam = new THREE.Vector3();
const _invCubeMatrix = new THREE.Matrix4();

/** Micrometer drag sensitivity (radians of mirror plane tilt per CSS pixel dragged). */
const MIRROR_TILT_DRAG_SENSITIVITY = 0.00024;
/** Micrometer axial lead-screw travel amplitude at full tilt (±3.2°). */
const LEAD_SCREW_TRAVEL = 0.008;

async function mountDay22Echo(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 22,
    cameraPosition: [2.38, 1.46, 3.68],
    fov: 34,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, -0.07, 0);
  stage.controls.minDistance = 2.55;
  stage.controls.maxDistance = 8.4;
  stage.controls.minPolarAngle = 0.06;
  stage.controls.maxPolarAngle = Math.PI - 0.06;
  stage.controls.update();

  // Studio Softbox PMREM Environment for crisp machined dark-bronze, satin-brass & knurling IBL
  const studioEnv = createEchoStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize 3-point studio lighting & tighten shadow frustum around the 1.45-unit brass cube
  let dirLightIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirLightIdx === 0) {
        child.position.set(-2.6, 4.6, 3.8);
        child.intensity = 1.28;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.45;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.0;
        child.shadow.bias = -0.00025;
        child.shadow.normalBias = 0.01;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirLightIdx === 1) {
        child.position.set(3.8, 1.5, 2.2);
        child.intensity = 0.42;
      } else if (dirLightIdx === 2) {
        child.position.set(-1.4, 3.4, -4.0);
        child.intensity = 1.32;
      }
      dirLightIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.42;
    }
  }

  const sim = new EchoSimulation(0);
  const materials = createEchoMaterials();
  const assembly = buildEchoCubeAssembly(materials);
  stage.specimenGroup.add(assembly.group);

  // Tactile pointer state (empty space -> 360° orbit; cube/thumbscrew/Shift drag -> mirror tilt;
  // click glass -> pluck suspended core; double-click -> snap/cycle canonical mirror presets)
  const canvas = stage.shell.canvas;
  let hoveredTarget: 'none' | 'thumbscrew' | 'glass' | 'frame' = 'none';
  let pendingCubeGesture = false;
  let isDraggingMirrorTilt = false;
  let downHitType: 'none' | 'thumbscrew' | 'glass' | 'frame' | 'shift' = 'none';
  let pointerDownX = 0;
  let pointerDownY = 0;
  let prevDragX = 0;
  let prevDragY = 0;
  let pointerDownTime = 0;
  let lastClickTime = 0;
  let spacePluckToggle = 1;
  const downHitPoint = new THREE.Vector3();

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastCube(clientX: number, clientY: number): {
    type: 'none' | 'thumbscrew' | 'glass' | 'frame';
    point: THREE.Vector3 | null;
  } {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    const thumbHits = _raycaster.intersectObjects(assembly.thumbscrewHitMeshes, false);
    if (thumbHits.length > 0) {
      return { type: 'thumbscrew', point: thumbHits[0].point };
    }

    const frameHits = _raycaster.intersectObjects(assembly.frameHitMeshes, false);
    const glassHits = _raycaster.intersectObject(assembly.glassMesh, false);

    if (glassHits.length > 0 && frameHits.length > 0) {
      if (glassHits[0].distance <= frameHits[0].distance) {
        return { type: 'glass', point: glassHits[0].point };
      }
      return { type: 'frame', point: frameHits[0].point };
    }
    if (glassHits.length > 0) {
      return { type: 'glass', point: glassHits[0].point };
    }
    if (frameHits.length > 0) {
      return { type: 'frame', point: frameHits[0].point };
    }
    return { type: 'none', point: null };
  }

  function pluckFromScreenHit(hitPoint: THREE.Vector3 | null): void {
    stage.camera.getWorldDirection(_camForward);
    _camRight.crossVectors(_camForward, stage.camera.up).normalize();
    _camUp.crossVectors(_camRight, _camForward).normalize();

    let ix = 0.68 * spacePluckToggle;
    let iz = -0.52 * spacePluckToggle;
    spacePluckToggle = -spacePluckToggle;

    if (hitPoint) {
      // Kick pendulum laterally away from click point projected onto horizontal cavity plane
      const rx = hitPoint.x - sim.corePos.x;
      const rz = hitPoint.z - sim.corePos.z;
      const horizLen = Math.hypot(rx, rz);
      if (horizLen > 0.08) {
        ix = (-rx / horizLen) * 0.78 + _camRight.x * 0.28;
        iz = (-rz / horizLen) * 0.78 + _camRight.z * 0.28;
      } else {
        ix = _camRight.x * 0.72 + _camForward.x * 0.34;
        iz = _camRight.z * 0.72 + _camForward.z * 0.34;
      }
    }
    sim.pluckCore(ix, iz);
  }

  const releaseCaptureSafe = (pointerId: number) => {
    try {
      canvas.releasePointerCapture(pointerId);
    } catch {
      // Ignore if pointer capture was already released
    }
  };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    prevDragX = ev.clientX;
    prevDragY = ev.clientY;
    pointerDownTime = performance.now();

    const hit = raycastCube(ev.clientX, ev.clientY);
    if (hit.point) downHitPoint.copy(hit.point);

    if (ev.shiftKey || hit.type !== 'none') {
      downHitType = hit.type !== 'none' ? hit.type : 'shift';
      pendingCubeGesture = true;
      isDraggingMirrorTilt = false;
      stage.controls.enabled = false;
      canvas.setPointerCapture(ev.pointerId);
      if (downHitType === 'thumbscrew' || downHitType === 'shift') {
        canvas.style.cursor = 'grabbing';
      }
    } else {
      downHitType = 'none';
      pendingCubeGesture = false;
      isDraggingMirrorTilt = false;
      stage.controls.enabled = true;
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (pendingCubeGesture) {
      const totalDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
      if (!isDraggingMirrorTilt && totalDist >= 6) {
        isDraggingMirrorTilt = true;
        canvas.style.cursor = 'grabbing';
      }
      if (isDraggingMirrorTilt) {
        const dx = ev.clientX - prevDragX;
        const dy = ev.clientY - prevDragY;
        // Continuous monotonic mirror micro-tilt (horizontal drag sweeps tiltY, vertical drag sweeps tiltX)
        sim.adjustMirrorTilt(
          dy * MIRROR_TILT_DRAG_SENSITIVITY,
          dx * MIRROR_TILT_DRAG_SENSITIVITY
        );
      }
      prevDragX = ev.clientX;
      prevDragY = ev.clientY;
      return;
    }

    if (ev.buttons !== 0) return;

    const hit = raycastCube(ev.clientX, ev.clientY);
    hoveredTarget = hit.type;
    if (hit.type === 'thumbscrew' || hit.type === 'frame') {
      canvas.style.cursor = 'grab';
    } else if (hit.type === 'glass') {
      canvas.style.cursor = 'pointer';
    } else {
      canvas.style.cursor = 'default';
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    const moveDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTime;

    if (pendingCubeGesture) {
      releaseCaptureSafe(ev.pointerId);
      stage.controls.enabled = true;
      pendingCubeGesture = false;

      if (isDraggingMirrorTilt) {
        isDraggingMirrorTilt = false;
        downHitType = 'none';
        canvas.style.cursor = hoveredTarget === 'glass' ? 'pointer' : 'grab';
        return;
      }
    }

    const isCleanClick = moveDist < 7 && heldMs < 400;
    if (isCleanClick) {
      const now = performance.now();
      const isDoubleClick = now - lastClickTime < 340;
      lastClickTime = now;

      if (isDoubleClick) {
        sim.cyclePreset();
        sim.pluckCore(0.22 * spacePluckToggle, -0.18 * spacePluckToggle);
        spacePluckToggle = -spacePluckToggle;
      } else if (downHitType === 'glass' || downHitType === 'frame' || downHitType === 'thumbscrew') {
        pluckFromScreenHit(downHitPoint);
      }
    }

    downHitType = 'none';
    canvas.style.cursor =
      hoveredTarget === 'thumbscrew' || hoveredTarget === 'frame'
        ? 'grab'
        : hoveredTarget === 'glass'
          ? 'pointer'
          : 'default';
  };

  const onPointerCancel = (ev: PointerEvent) => {
    pendingCubeGesture = false;
    isDraggingMirrorTilt = false;
    downHitType = 'none';
    stage.controls.enabled = true;
    releaseCaptureSafe(ev.pointerId);
    canvas.style.cursor = 'default';
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      sim.pluckCore(0.84 * spacePluckToggle, -0.62 * spacePluckToggle);
      spacePluckToggle = -spacePluckToggle;
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerCancel);
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerCancel);
    window.removeEventListener('keydown', onKeyDown);
    stage.scene.environment = null;
    studioEnv.dispose();
    assembly.dispose();
    materials.dispose();
  });

  // Read-only verification diagnostic hook
  window.__DAY22__ = {
    getState: () => {
      const [txDeg, tyDeg] = sim.getTiltDegrees();
      return {
        mirrorTiltDeg: [Number(txDeg.toFixed(3)), Number(tyDeg.toFixed(3))],
        corePos: [
          Number(sim.corePos.x.toFixed(4)),
          Number(sim.corePos.y.toFixed(4)),
          Number(sim.corePos.z.toFixed(4)),
        ],
        coreVel: [
          Number(sim.coreVel.x.toFixed(4)),
          Number(sim.coreVel.y.toFixed(4)),
          Number(sim.coreVel.z.toFixed(4)),
        ],
        bounceCount: ECHO_BOUNCE_COUNT,
        presetIndex: sim.presetIndex,
      };
    },
  };

  // Main animation tick (strictly zero per-frame allocations)
  stage.onTick(({ dtSec, elapsedSec, camera }) => {
    sim.step(dtSec);

    // Transform active camera into cube local space for exact 6-face raytracing
    assembly.group.updateMatrixWorld();
    _invCubeMatrix.copy(assembly.group.matrixWorld).invert();
    _localCam.copy(camera.position).applyMatrix4(_invCubeMatrix);
    materials.uCamLocal.value.copy(_localCam);

    // Sync the 6 tilted interior mirror plane normals
    for (let k = 0; k < 6; k++) {
      materials.uMirrorNormals[k].value.copy(sim.mirrorNormals[k]);
    }

    // Sync swinging Realgar-Amber filament / Bone-Calcite prism core uniforms
    materials.uCorePos.value.copy(sim.corePos);
    materials.uCoreAxis.value.copy(sim.coreAxis);
    materials.uCoreSpin.value = sim.coreSpinAngle;
    materials.uPluckFlash.value = sim.pluckFlash;
    materials.uTime.value = elapsedSec;

    // Smooth hover highlights for knurled thumbscrews & glass bezel edge
    const thumbHoverGoal = hoveredTarget === 'thumbscrew' || isDraggingMirrorTilt ? 1 : 0;
    const glassHoverGoal = hoveredTarget === 'glass' ? 1 : 0;
    const hoverLerp = Math.min(1, dtSec * 14);
    materials.uThumbHover.value += (thumbHoverGoal - materials.uThumbHover.value) * hoverLerp;
    materials.uGlassHover.value += (glassHoverGoal - materials.uGlassHover.value) * hoverLerp;

    // Drive physical rotation & lead-screw travel on all 6 knurled brass micrometer thumbscrews
    const normTiltX = sim.tiltXRad / MAX_MIRROR_TILT_RAD;
    const normTiltY = sim.tiltYRad / MAX_MIRROR_TILT_RAD;
    for (let i = 0; i < assembly.thumbscrews.length; i++) {
      const ts = assembly.thumbscrews[i];
      const angle = ts.axis === 'x' ? sim.screwAngleX : sim.screwAngleY;
      const travel = (ts.axis === 'x' ? normTiltX : normTiltY) * LEAD_SCREW_TRAVEL;
      ts.rotorGroup.rotation.y = angle * ts.sign;
      ts.rotorGroup.position.y = travel * ts.sign;
    }
  });
}

mountDay22Echo().catch((err) => {
  console.error('[Day 22 - Echo] Failed to initialize WebGPU specimen:', err);
});
