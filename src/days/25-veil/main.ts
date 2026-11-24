import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { FluidVeilSolver } from './fluidVeil';
import { createVeilMotesSystem } from './motesCompute';
import {
  buildGrazingCrossSectionGeometry,
  buildSlitLampAssembly,
  buildVolumetricVeilGeometry,
} from './slitLampGeometry';
import { createVeilMaterials, createVeilStudioEnvironment } from './veilMaterials';
import {
  ApertureModeController,
  intersectVeilRay,
  type VeilApertureMode,
  type VeilRayHit,
} from './veilSurface';

declare global {
  interface Window {
    __DAY25__?: {
      getState: () => {
        apertureMode: VeilApertureMode;
        maxVorticity: number;
        meanDensity: number;
        tearEnergy: number;
        activeMotes: number;
      };
    };
  }
}

// Pre-allocated zero-GC scratch objects for pointer raycasting & cursor-focused zoom
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _lampHitPt = new THREE.Vector3();
const _desiredTarget = new THREE.Vector3(0, 0.04, 0);
const _defaultTarget = new THREE.Vector3(0, 0.04, 0);
const _veilHit: VeilRayHit = {
  u: 0.5,
  v: 0.5,
  distance: 0,
  point: new THREE.Vector3(),
};

/**
 * Day 25 — Veil:
 * "Studio Slit-Lamp Volumetric Light-and-Haze Curtain"
 *
 * Interaction Contract:
 *   - Hover / sweep pointer across the suspended light veil -> injects incompressible
 *     vorticity momentum along the exact 3D ray-curtain intersection, shedding twin
 *     counter-rotating Kelvin-Helmholtz smoke curls that billow and slowly heal.
 *   - Hold & drag across the curtain -> carves a deep dark turbulent channel with sheared
 *     golden caustic rims and entrained swirling dust motes.
 *   - Click the machined overhead brass & iron slit-lamp housing (or press Space) -> seamlessly
 *     cycles aperture geometry between Flat Planar Curtain, Graceful Curved S-Veil, and
 *     Hollow Elliptical Light Cone while physically articulating the brass slit jaws & vernier drums.
 *   - Scroll to zoom right up to the lamp hardware (cursor-focused when hovering the projector)
 *     and drag empty Chalky Slate space to orbit 360° (right-drag pans; double-click resets view).
 */
export async function bootDay25Veil(): Promise<SpecimenStage> {
  const stage = await SpecimenStage.create({
    day: 25,
    cameraPosition: [1.45, 0.24, 5.25],
    fov: 36,
    enableOrbit: true,
    enableShadows: true,
  });

  // Allow crisp millimeter-range macro zoom-in on the slit-lamp metalwork & stone inlays
  stage.camera.near = 0.04;
  stage.camera.updateProjectionMatrix();

  stage.controls.target.copy(_defaultTarget);
  stage.controls.enablePan = true;
  stage.controls.screenSpacePanning = true;
  stage.controls.minDistance = 0.72;
  stage.controls.maxDistance = 8.8;
  stage.controls.minPolarAngle = 0.06;
  stage.controls.maxPolarAngle = Math.PI - 0.06;
  stage.controls.update();

  // Enveloping studio cyclorama + 7-scrim PMREM IBL so gunmetal iron, brushed & engraved brass,
  // honed dark slate, glazed steatite ceramic, and blued steel read crisply at every zoom level
  const studioEnv = createVeilStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.24;

  // Harmonize studio lights & sharpen 2048x2048 shadow map tightly around the overhead slit-lamp
  let dirLightIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirLightIdx === 0) {
        child.position.set(-2.4, 4.5, 3.6);
        child.intensity = 1.72;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.45;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 0.8;
        child.shadow.camera.far = 10.5;
        child.shadow.bias = -0.00022;
        child.shadow.normalBias = 0.008;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirLightIdx === 1) {
        child.position.set(3.6, 1.5, 2.5);
        child.intensity = 0.68;
      } else if (dirLightIdx === 2) {
        child.position.set(-1.2, 3.4, -3.8);
        child.intensity = 1.42;
      }
      dirLightIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.56;
    }
  }

  // Core pure-TS simulation + aperture mode controller
  const fluid = new FluidVeilSolver();
  const aperture = new ApertureModeController('planar');
  const materials = createVeilMaterials(fluid.dataTexture);

  // 1. Machined Overhead Brass, Iron & Slate Optical Slit-Lamp Housing
  const slitLamp = buildSlitLampAssembly(materials);
  slitLamp.group.renderOrder = 0;
  stage.specimenGroup.add(slitLamp.group);

  // 2. Multi-Stratum Volumetric Light-and-Haze Curtain (1 draw call)
  const veilGeo = buildVolumetricVeilGeometry(84, 64, 5);
  const veilMesh = new THREE.Mesh(veilGeo, materials.veilMaterial);
  veilMesh.name = 'VolumetricLightVeil';
  veilMesh.frustumCulled = false;
  veilMesh.renderOrder = 2;
  stage.specimenGroup.add(veilMesh);

  // 3. Grazing Cross-Section Edge Ribbon (preserves beam cross-section at 90 deg side profile)
  const grazingGeo = buildGrazingCrossSectionGeometry(48);
  const grazingMesh = new THREE.Mesh(grazingGeo, materials.grazingMaterial);
  grazingMesh.name = 'GrazingBeamCrossSection';
  grazingMesh.frustumCulled = false;
  grazingMesh.renderOrder = 3;
  stage.specimenGroup.add(grazingMesh);

  // 4. WebGPU TSL-Compute Sunlit Dust Motes (`8,192` motes gated strictly to the beam plane)
  const motes = createVeilMotesSystem(fluid.dataTexture);
  stage.specimenGroup.add(motes.sprite);
  motes.runInit(stage.renderer);

  // Pointer, zoom-focus & keyboard interaction state
  const canvas = stage.shell.canvas;
  const bootTimeMs = performance.now();
  let hoverUnlocked = false;
  let firstPointerX = NaN;
  let firstPointerY = NaN;

  let hoveredTarget: 'none' | 'lamp' | 'veil' = 'none';
  let hasLampHit = false;
  let downTarget: 'none' | 'lamp' | 'veil' | 'space' = 'none';
  let isSlicingVeil = false;
  let activePointerId: number | null = null;
  let downX = 0;
  let downY = 0;
  let downTimeMs = 0;

  // Smooth zoom-focus target glide (only active during wheel zoom toward lamp or double-click reset)
  let focusGlideStrength = 0;

  let hasPrevVeilHit = false;
  let prevVeilU = 0.5;
  let prevVeilV = 0.5;
  let prevMoveTimeMs = 0;
  let activeMotes = 2000;

  const updateNdc = (clientX: number, clientY: number): boolean => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    _ndc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    return true;
  };

  const raycastLamp = (): boolean => {
    _raycaster.setFromCamera(_ndc, stage.camera);
    const hits = _raycaster.intersectObjects(slitLamp.hitMeshes, false);
    if (hits.length > 0) {
      _lampHitPt.copy(hits[0].point);
      hasLampHit = true;
      return true;
    }
    hasLampHit = false;
    return false;
  };

  const raycastVeil = (): boolean => {
    _raycaster.setFromCamera(_ndc, stage.camera);
    return intersectVeilRay(_raycaster.ray, aperture.weights, _veilHit);
  };

  const triggerModeCycle = (): void => {
    aperture.cycleMode();
    fluid.injectAperturePulse();
    hasPrevVeilHit = false;
  };

  const injectStrokeAtHit = (isDrag: boolean, nowMs: number): void => {
    const dtMove = Math.max(0.008, Math.min(0.06, (nowMs - prevMoveTimeMs) * 0.001));
    if (hasPrevVeilHit) {
      let du = _veilHit.u - prevVeilU;
      if (fluid.periodicX) {
        if (du > 0.5) du -= 1;
        else if (du < -0.5) du += 1;
      }
      const dv = _veilHit.v - prevVeilV;
      const dist = Math.hypot(du, dv);
      if (dist > 0.0012 || isDrag) {
        const velU = du / dtMove;
        const velV = dv / dtMove;
        fluid.injectPointerSweep(
          prevVeilU,
          prevVeilV,
          _veilHit.u,
          _veilHit.v,
          velU,
          velV,
          isDrag
        );
      }
    } else if (isDrag) {
      fluid.injectPointerSweep(
        _veilHit.u,
        _veilHit.v,
        _veilHit.u,
        _veilHit.v,
        0.55,
        0.28,
        true
      );
    }

    hasPrevVeilHit = true;
    prevVeilU = _veilHit.u;
    prevVeilV = _veilHit.v;
    prevMoveTimeMs = nowMs;
  };

  const releaseCaptureSafe = (pointerId: number): void => {
    try {
      if (canvas.hasPointerCapture(pointerId)) {
        canvas.releasePointerCapture(pointerId);
      }
    } catch {
      // Synthetic Playwright events may not hold native pointer capture
    }
  };

  const onPointerDown = (ev: PointerEvent): void => {
    if (ev.button !== 0) {
      // Right-click / middle-click pan overrides any automatic target glide
      focusGlideStrength = 0;
      return;
    }
    hoverUnlocked = true;
    downX = ev.clientX;
    downY = ev.clientY;
    downTimeMs = performance.now();

    if (!updateNdc(ev.clientX, ev.clientY)) return;

    if (raycastLamp()) {
      downTarget = 'lamp';
      isSlicingVeil = false;
      stage.controls.enabled = true;
      canvas.style.cursor = 'pointer';
      return;
    }

    if (raycastVeil()) {
      downTarget = 'veil';
      isSlicingVeil = true;
      activePointerId = ev.pointerId;
      stage.controls.enabled = false;
      canvas.style.cursor = 'crosshair';
      hasPrevVeilHit = false;
      injectStrokeAtHit(true, downTimeMs);
      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Ignore capture failure on synthetic pointer events
      }
      return;
    }

    downTarget = 'space';
    isSlicingVeil = false;
    stage.controls.enabled = true;
    hasPrevVeilHit = false;
    canvas.style.cursor = 'grabbing';
  };

  const onPointerMove = (ev: PointerEvent): void => {
    const nowMs = performance.now();
    if (!hoverUnlocked) {
      if (Number.isNaN(firstPointerX)) {
        firstPointerX = ev.clientX;
        firstPointerY = ev.clientY;
        return;
      }
      const moved = Math.hypot(ev.clientX - firstPointerX, ev.clientY - firstPointerY);
      if (moved < 6 && nowMs - bootTimeMs < 240) return;
      hoverUnlocked = true;
    }

    if (!updateNdc(ev.clientX, ev.clientY)) return;

    if (isSlicingVeil) {
      if (raycastVeil()) {
        injectStrokeAtHit(true, nowMs);
      } else {
        hasPrevVeilHit = false;
      }
      return;
    }

    if (ev.buttons !== 0) {
      hasPrevVeilHit = false;
      return;
    }

    if (raycastLamp()) {
      hoveredTarget = 'lamp';
      hasPrevVeilHit = false;
      canvas.style.cursor = 'pointer';
      return;
    }

    if (raycastVeil()) {
      hoveredTarget = 'veil';
      canvas.style.cursor = 'crosshair';
      injectStrokeAtHit(false, nowMs);
      return;
    }

    hoveredTarget = 'none';
    hasPrevVeilHit = false;
    canvas.style.cursor = 'grab';
  };

  const onPointerUp = (ev: PointerEvent): void => {
    const moveDist = Math.hypot(ev.clientX - downX, ev.clientY - downY);
    const heldMs = performance.now() - downTimeMs;

    if (isSlicingVeil && (activePointerId === null || ev.pointerId === activePointerId)) {
      isSlicingVeil = false;
      activePointerId = null;
      stage.controls.enabled = true;
      releaseCaptureSafe(ev.pointerId);
    }

    if (downTarget === 'lamp' && moveDist < 8 && heldMs < 420) {
      triggerModeCycle();
    }

    downTarget = 'none';
    hasPrevVeilHit = false;
    canvas.style.cursor =
      hoveredTarget === 'lamp' ? 'pointer' : hoveredTarget === 'veil' ? 'crosshair' : 'grab';
  };

  const onPointerCancel = (ev: PointerEvent): void => {
    isSlicingVeil = false;
    activePointerId = null;
    downTarget = 'none';
    hasPrevVeilHit = false;
    stage.controls.enabled = true;
    releaseCaptureSafe(ev.pointerId);
    canvas.style.cursor = 'grab';
  };

  const onPointerLeave = (): void => {
    hoveredTarget = 'none';
    hasPrevVeilHit = false;
  };

  /**
   * Cursor-aware zoom focus:
   *  - Scrolling IN (`deltaY < 0`) while hovering the slit-lamp (or upper studio half) smoothly
   *    steers the orbit target onto the hovered lamp point so zooming in frames the brass, iron,
   *    slate & vernier hardware directly instead of clipping the projector off the top edge!
   *  - Scrolling OUT (`deltaY > 0`) gently returns focus toward the full-specimen center `(0, 0.04, 0)`.
   */
  const onWheel = (ev: WheelEvent): void => {
    if (!updateNdc(ev.clientX, ev.clientY)) return;
    if (ev.deltaY < 0) {
      if (raycastLamp() && hasLampHit) {
        _desiredTarget.set(
          Math.max(-0.98, Math.min(0.98, _lampHitPt.x)),
          Math.max(1.14, Math.min(1.52, _lampHitPt.y)),
          Math.max(-0.22, Math.min(0.22, _lampHitPt.z))
        );
        focusGlideStrength = 1.0;
      } else if (_ndc.y > 0.24) {
        // Upper-third zoom gesture aims naturally toward the overhead projector centerpiece
        _desiredTarget.set(Math.max(-0.85, Math.min(0.85, _ndc.x * 1.1)), 1.34, 0);
        focusGlideStrength = 0.85;
      }
    } else if (ev.deltaY > 0) {
      const dist = stage.camera.position.distanceTo(stage.controls.target);
      if (dist > 3.2) {
        _desiredTarget.copy(_defaultTarget);
        focusGlideStrength = 0.85;
      }
    }
  };

  const onDoubleClick = (ev: MouseEvent): void => {
    if (!updateNdc(ev.clientX, ev.clientY)) return;
    if (raycastLamp()) return; // Do not hijack rapid clicks on the lamp mode switcher
    _desiredTarget.copy(_defaultTarget);
    focusGlideStrength = 1.0;
  };

  const onKeyDown = (ev: KeyboardEvent): void => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      triggerModeCycle();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown, true);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });
  window.addEventListener('pointercancel', onPointerCancel, { passive: true });
  canvas.addEventListener('pointerleave', onPointerLeave, { passive: true });
  canvas.addEventListener('wheel', onWheel, { passive: true });
  canvas.addEventListener('dblclick', onDoubleClick);
  window.addEventListener('keydown', onKeyDown);

  // Read-only verification telemetry hook
  window.__DAY25__ = {
    getState: () => ({
      apertureMode: aperture.mode,
      maxVorticity: Number(fluid.maxVorticity.toFixed(3)),
      meanDensity: Number(fluid.meanDensity.toFixed(4)),
      tearEnergy: Number(fluid.tearEnergy.toFixed(4)),
      activeMotes,
    }),
  };

  // Main per-frame simulation & WebGPU compute tick (zero per-frame allocations)
  stage.onTick(({ dtSec, elapsedSec, renderer }) => {
    const dt = Math.min(0.05, Math.max(0.001, dtSec));

    // Smooth cursor-focused zoom target glide + bounded specimen pan envelope
    if (focusGlideStrength > 0.001) {
      const alpha = 1 - Math.exp(-dt * 5.8);
      stage.controls.target.lerp(_desiredTarget, alpha * focusGlideStrength);
      focusGlideStrength *= Math.exp(-dt * 2.4);
      if (focusGlideStrength < 0.015) focusGlideStrength = 0;
    }
    stage.controls.target.x = Math.max(-1.05, Math.min(1.05, stage.controls.target.x));
    stage.controls.target.y = Math.max(-0.65, Math.min(1.56, stage.controls.target.y));
    stage.controls.target.z = Math.max(-0.45, Math.min(0.45, stage.controls.target.z));

    // 1. Advance aperture morph weights & brass slit-lamp mechanical articulation
    aperture.step(dt);
    fluid.periodicX = aperture.weights.cone > 0.5;
    slitLamp.updateArticulation(aperture.weights, aperture.drumAngleRad);

    // 2. Step 128x128 incompressible Navier-Stokes + vorticity confinement fluid solver
    fluid.step(dt, true);

    // 3. Sync TSL shader uniforms (zero allocation)
    materials.uTime.value = elapsedSec;
    materials.uModeWeights.value.set(
      aperture.weights.planar,
      aperture.weights.curvedS,
      aperture.weights.cone
    );
    materials.uTransitionPulse.value = aperture.transitionPulse;
    const hoverGoal = hoveredTarget === 'lamp' ? 1 : 0;
    materials.uLampHover.value += (hoverGoal - materials.uLampHover.value) * Math.min(1, dt * 12);

    // 4. Advance GPU compute sunlit dust motes directly coupled to the live fluid field
    activeMotes = motes.update(
      renderer,
      dt,
      elapsedSec,
      aperture.weights,
      fluid.meanDensity,
      fluid.tearEnergy
    );
  });

  // Full cleanup on page dispose
  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown, true);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerCancel);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('dblclick', onDoubleClick);
    window.removeEventListener('keydown', onKeyDown);
    stage.scene.environment = null;
    studioEnv.dispose();
    slitLamp.dispose();
    veilGeo.dispose();
    grazingGeo.dispose();
    materials.dispose();
    motes.dispose();
    fluid.dispose();
    delete window.__DAY25__;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay25Veil().catch((err) => {
    console.error('[Day 25 - Veil] Failed to initialize WebGPU specimen:', err);
  });
}
