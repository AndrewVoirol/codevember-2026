import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  ARCH_HALF_WIDTH,
  ARCH_SILL_Y,
  ARCH_CROWN_Y,
  DEFAULT_CAMERA_DISTANCE,
  INITIAL_ORBIT_AZIMUTH_DEG,
  INITIAL_ORBIT_ELEVATION_DEG,
  MAX_CAMERA_DISTANCE,
  MAX_ELEVATION_DEG,
  MIN_CAMERA_DISTANCE,
  MIN_ELEVATION_DEG,
  PORTAL_FACES,
  PORTAL_FACE_ORDER,
  clampCameraDistance,
  classifyPortalView,
  computeFaceZoomGoal,
  computeNextQuarterTurnAzimuth,
  shortestAngleDeltaDeg,
  sphericalOrbitToPosition,
  type PortalFaceId,
} from './portalWorlds';
import { buildPortalCubeAssembly } from './geometry';
import { createPortalMaterials, createPortalStudioEnvironment } from './materials';

export interface Day29DiagnosticsState {
  dominantFace: PortalFaceId;
  visibleFaces: PortalFaceId[];
  cameraDistance: number;
  zoomedFace: PortalFaceId | null;
  orbitAzimuthDeg: number;
  orbitElevationDeg: number;
}

declare global {
  interface Window {
    __DAY29__?: {
      getState: () => Day29DiagnosticsState;
    };
  }
}

// Pre-allocated scratch math objects (strictly zero per-frame heap allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _invCubeMatrix = new THREE.Matrix4();
const _localCam = new THREE.Vector3();
const _localHit = new THREE.Vector3();
const _animPos = new THREE.Vector3();
const _camTuple: [number, number, number] = [0, 0, 0];

interface CameraTransitionState {
  active: boolean;
  elapsedSec: number;
  durationSec: number;
  startAzDeg: number;
  deltaAzDeg: number;
  startElDeg: number;
  targetElDeg: number;
  startDist: number;
  targetDist: number;
}

function smoothCubicInOut(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) * 0.5;
}

async function mountDay29Portal(): Promise<void> {
  const initCam = sphericalOrbitToPosition(
    INITIAL_ORBIT_AZIMUTH_DEG,
    INITIAL_ORBIT_ELEVATION_DEG,
    DEFAULT_CAMERA_DISTANCE
  );

  const stage = await SpecimenStage.create({
    day: 29,
    cameraPosition: [initCam.x, initCam.y, initCam.z],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  // Tighten near plane cleanly so zoomed threshold views never clip basalt or brass bezels
  stage.camera.near = 0.08;
  stage.camera.updateProjectionMatrix();

  stage.controls.target.set(0, 0, 0);
  stage.controls.minDistance = MIN_CAMERA_DISTANCE;
  stage.controls.maxDistance = MAX_CAMERA_DISTANCE;
  stage.controls.minPolarAngle = ((90 - MAX_ELEVATION_DEG) * Math.PI) / 180;
  stage.controls.maxPolarAngle = ((90 - MIN_ELEVATION_DEG) * Math.PI) / 180;
  stage.controls.dampingFactor = 0.07;
  stage.controls.update();

  // Studio Softbox PMREM Environment for matte basalt & brushed satin brass
  const studioEnv = createPortalStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize 3-point studio lighting & tighten shadow frustum cleanly around the 1.52-unit cube
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(-2.5, 4.5, 3.8);
        child.intensity = 1.42;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.55;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.0;
        child.shadow.bias = -0.00025;
        child.shadow.normalBias = 0.012;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        child.position.set(3.8, 1.4, 2.4);
        child.intensity = 0.56;
      } else if (dirIdx === 2) {
        child.position.set(-1.6, 3.3, -4.1);
        child.intensity = 1.38;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.48;
    }
  }

  const materials = createPortalMaterials();
  const assembly = buildPortalCubeAssembly(materials);
  stage.specimenGroup.add(assembly.group);

  // Interactive camera rig & aperture hover state
  const canvas = stage.shell.canvas;
  let hoveredFace: PortalFaceId | null = null;
  let zoomedFace: PortalFaceId | null = null;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTime = 0;
  let lastClickTime = 0;
  let isUserInteracting = false;

  const hoverTargetUv: Record<PortalFaceId, [number, number]> = {
    north: [0, 0],
    east: [0, 0],
    south: [0, 0],
    west: [0, 0],
  };

  const cameraAnim: CameraTransitionState = {
    active: false,
    elapsedSec: 0,
    durationSec: 0.85,
    startAzDeg: INITIAL_ORBIT_AZIMUTH_DEG,
    deltaAzDeg: 0,
    startElDeg: INITIAL_ORBIT_ELEVATION_DEG,
    targetElDeg: INITIAL_ORBIT_ELEVATION_DEG,
    startDist: DEFAULT_CAMERA_DISTANCE,
    targetDist: DEFAULT_CAMERA_DISTANCE,
  };

  function startCameraTransition(
    targetAzDeg: number,
    targetElDeg: number,
    targetDist: number,
    durationSec = 0.82
  ): void {
    const pos = stage.camera.position;
    _camTuple[0] = pos.x;
    _camTuple[1] = pos.y;
    _camTuple[2] = pos.z;
    const cur = classifyPortalView(_camTuple);

    cameraAnim.active = true;
    cameraAnim.elapsedSec = 0;
    cameraAnim.durationSec = durationSec;
    cameraAnim.startAzDeg = cur.orbitAzimuthDeg;
    cameraAnim.deltaAzDeg = shortestAngleDeltaDeg(cur.orbitAzimuthDeg, targetAzDeg);
    cameraAnim.startElDeg = cur.orbitElevationDeg;
    cameraAnim.targetElDeg = Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, targetElDeg));
    cameraAnim.startDist = cur.cameraDistance;
    cameraAnim.targetDist = clampCameraDistance(targetDist);
  }

  function triggerQuarterTurnShowcase(): void {
    const pos = stage.camera.position;
    _camTuple[0] = pos.x;
    _camTuple[1] = pos.y;
    _camTuple[2] = pos.z;
    const cur = classifyPortalView(_camTuple);
    const nextAz = computeNextQuarterTurnAzimuth(cur.orbitAzimuthDeg);

    if (zoomedFace !== null) {
      // Advance directly to the next cardinal world while keeping threshold zoom!
      const idx = PORTAL_FACE_ORDER.indexOf(zoomedFace);
      const nextFace = PORTAL_FACE_ORDER[(idx + 1) % PORTAL_FACE_ORDER.length];
      zoomedFace = nextFace;
      const goal = computeFaceZoomGoal(nextFace);
      startCameraTransition(goal.azimuthDeg, goal.elevationDeg, goal.distance, 0.92);
      return;
    }

    // Step 90° around the cube (preserving crisp 45° split-corner dual-world viewpoints!)
    const targetEl = Math.max(-8, Math.min(28, cur.orbitElevationDeg < 4 ? 16 : cur.orbitElevationDeg));
    const targetDist = cur.cameraDistance < 3.4 ? DEFAULT_CAMERA_DISTANCE : cur.cameraDistance;
    startCameraTransition(nextAz, targetEl, targetDist, 0.88);
  }

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastPortalAperture(
    clientX: number,
    clientY: number
  ): { face: PortalFaceId | null; uNorm: number; vNorm: number } {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    const portalHits = _raycaster.intersectObjects(assembly.portalHitMeshes, false);
    if (portalHits.length === 0) {
      return { face: null, uNorm: 0, vNorm: 0 };
    }

    const nearestPortal = portalHits[0];
    const solidHits = _raycaster.intersectObjects(assembly.exteriorHitMeshes, false);
    if (solidHits.length > 0 && solidHits[0].distance < nearestPortal.distance - 0.002) {
      return { face: null, uNorm: 0, vNorm: 0 };
    }

    const face = (nearestPortal.object.userData.portalFace as PortalFaceId | undefined) ?? null;
    if (!face) return { face: null, uNorm: 0, vNorm: 0 };

    assembly.group.updateMatrixWorld();
    _invCubeMatrix.copy(assembly.group.matrixWorld).invert();
    _localHit.copy(nearestPortal.point).applyMatrix4(_invCubeMatrix);

    const spec = PORTAL_FACES[face];
    const u =
      _localHit.x * spec.right[0] + _localHit.y * spec.right[1] + _localHit.z * spec.right[2];
    const v = _localHit.x * spec.up[0] + _localHit.y * spec.up[1] + _localHit.z * spec.up[2];

    const uNorm = Math.max(-1, Math.min(1, u / ARCH_HALF_WIDTH));
    const vMid = (ARCH_SILL_Y + ARCH_CROWN_Y) * 0.5;
    const vHalf = (ARCH_CROWN_Y - ARCH_SILL_Y) * 0.5;
    const vNorm = Math.max(-1, Math.min(1, (v - vMid) / vHalf));

    return { face, uNorm, vNorm };
  }

  const onControlsStart = () => {
    isUserInteracting = true;
    cameraAnim.active = false;
  };
  const onControlsEnd = () => {
    isUserInteracting = false;
  };
  stage.controls.addEventListener('start', onControlsStart);
  stage.controls.addEventListener('end', onControlsEnd);

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    pointerDownTime = performance.now();
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (ev.buttons !== 0) {
      canvas.style.cursor = 'grabbing';
      return;
    }

    const hit = raycastPortalAperture(ev.clientX, ev.clientY);
    hoveredFace = hit.face;
    if (hit.face) {
      hoverTargetUv[hit.face][0] = hit.uNorm;
      hoverTargetUv[hit.face][1] = hit.vNorm;
      canvas.style.cursor = 'pointer';
    } else {
      canvas.style.cursor = 'grab';
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    const moveDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTime;
    const isCleanClick = moveDist < 7 && heldMs < 380;

    if (isCleanClick) {
      const now = performance.now();
      const isDoubleClick = now - lastClickTime < 320;
      lastClickTime = now;

      if (isDoubleClick) {
        triggerQuarterTurnShowcase();
      } else {
        const hit = raycastPortalAperture(ev.clientX, ev.clientY);
        if (hit.face) {
          if (zoomedFace === hit.face) {
            // Clicking the zoomed portal again pulls smoothly back to the suspended cube
            zoomedFace = null;
            const spec = PORTAL_FACES[hit.face];
            startCameraTransition(spec.azimuthDeg + 42, INITIAL_ORBIT_ELEVATION_DEG, DEFAULT_CAMERA_DISTANCE, 0.85);
          } else {
            // Dolly right up to the chosen arched stone threshold
            zoomedFace = hit.face;
            const goal = computeFaceZoomGoal(hit.face);
            startCameraTransition(goal.azimuthDeg, goal.elevationDeg, goal.distance, 0.85);
          }
        } else if (zoomedFace !== null) {
          // Clicking empty slate space pulls cleanly back to the suspended cube
          const prevSpec = PORTAL_FACES[zoomedFace];
          zoomedFace = null;
          startCameraTransition(prevSpec.azimuthDeg + 42, INITIAL_ORBIT_ELEVATION_DEG, DEFAULT_CAMERA_DISTANCE, 0.85);
        }
      }
    }

    canvas.style.cursor = hoveredFace ? 'pointer' : 'grab';
  };

  const onPointerLeave = () => {
    hoveredFace = null;
    canvas.style.cursor = 'default';
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      triggerQuarterTurnShowcase();
    }
  };

  canvas.style.cursor = 'grab';
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerLeave);
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    stage.controls.removeEventListener('start', onControlsStart);
    stage.controls.removeEventListener('end', onControlsEnd);
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    window.removeEventListener('keydown', onKeyDown);

    stage.scene.environment = null;
    studioEnv.dispose();
    assembly.dispose();
    materials.dispose();
  });

  // Read-only verification hook
  window.__DAY29__ = {
    getState: () => {
      const pos = stage.camera.position;
      const view = classifyPortalView([pos.x, pos.y, pos.z]);
      return {
        dominantFace: view.dominantFace,
        visibleFaces: view.visibleFaces,
        cameraDistance: Number(view.cameraDistance.toFixed(4)),
        zoomedFace,
        orbitAzimuthDeg: Number(view.orbitAzimuthDeg.toFixed(2)),
        orbitElevationDeg: Number(view.orbitElevationDeg.toFixed(2)),
      };
    },
  };

  // Main animation tick (strictly zero per-frame allocations)
  stage.onTick(({ dtSec, elapsedSec, camera }) => {
    // 1. Advance smooth camera transition when active (cancelled immediately on manual orbit/scroll)
    if (cameraAnim.active && !isUserInteracting) {
      cameraAnim.elapsedSec += dtSec;
      const u = Math.min(1, cameraAnim.elapsedSec / Math.max(0.05, cameraAnim.durationSec));
      const ease = smoothCubicInOut(u);
      const curAz = cameraAnim.startAzDeg + cameraAnim.deltaAzDeg * ease;
      const curEl = cameraAnim.startElDeg + (cameraAnim.targetElDeg - cameraAnim.startElDeg) * ease;
      const curDist = cameraAnim.startDist + (cameraAnim.targetDist - cameraAnim.startDist) * ease;

      sphericalOrbitToPosition(curAz, curEl, curDist, _animPos);
      camera.position.copy(_animPos);
      camera.lookAt(stage.controls.target);

      if (u >= 1) {
        cameraAnim.active = false;
      }
    }

    // 2. If user manually scrolled back out from a zoomed threshold, clear zoomedFace cleanly
    if (zoomedFace !== null && !cameraAnim.active) {
      const dist = camera.position.length();
      if (dist > 3.35) {
        zoomedFace = null;
      }
    }

    // 3. Transform active camera into cube-local coordinates for exact 4-face raymarching
    assembly.group.updateMatrixWorld();
    _invCubeMatrix.copy(assembly.group.matrixWorld).invert();
    _localCam.copy(camera.position).applyMatrix4(_invCubeMatrix);
    materials.uCamLocal.value.copy(_localCam);
    materials.uTime.value = elapsedSec;

    // 4. Smoothly interpolate per-face hover weight & aperture pointer coordinates (zero allocations)
    const hoverLerp = Math.min(1, dtSec * 10);
    const uvLerp = Math.min(1, dtSec * 12);
    for (let i = 0; i < PORTAL_FACE_ORDER.length; i++) {
      const f = PORTAL_FACE_ORDER[i];
      const goalW = hoveredFace === f ? 1 : 0;
      materials.uHoverWeights[f].value += (goalW - materials.uHoverWeights[f].value) * hoverLerp;

      const ptrU = materials.uPointerLocal[f].value;
      const targetX = hoveredFace === f ? hoverTargetUv[f][0] : 0;
      const targetY = hoveredFace === f ? hoverTargetUv[f][1] : 0;
      ptrU.x += (targetX - ptrU.x) * uvLerp;
      ptrU.y += (targetY - ptrU.y) * uvLerp;
      ptrU.z = stage.pointer.impulse;
    }
  });
}

mountDay29Portal().catch((err) => {
  console.error('[Day 29 - Portal] Failed to initialize WebGPU specimen:', err);
});
