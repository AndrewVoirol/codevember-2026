import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  CORE_COUNT,
  buildCoreAssembly,
} from './coreAssemblies';
import {
  createCocoonMaterials,
  createCocoonStudioEnvironment,
} from './cocoonMaterials';
import { ShrinkFilmSimulation } from './shrinkPhysics';
import {
  DEFAULT_COCOON_CONFIG,
  type BuiltCoreAssembly,
  type Day20TelemetryState,
} from './types';

declare global {
  interface Window {
    __DAY20__?: {
      getState: () => Day20TelemetryState;
    };
  }
}

// Pre-allocated scratch math objects (strictly zero per-frame allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _dragPlane = new THREE.Plane();
const _camDir = new THREE.Vector3();
const _hitWorld = new THREE.Vector3();
const _hitLocal = new THREE.Vector3();
const _camLocal = new THREE.Vector3();
const _downHitLocal = new THREE.Vector3();

export async function bootDay20Cocoon(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(20);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.42, 0.56, 7.25],
    fov: 36,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, 0, 0);
  stage.controls.minDistance = 2.6;
  stage.controls.maxDistance = 12.5;
  stage.controls.enableDamping = true;
  stage.controls.dampingFactor = 0.065;
  stage.controls.update();

  // Studio softbox PMREM environment map for brass bevels, basalt facets & glossy shrink-film
  const studioEnv = createCocoonStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.15;

  // Tighten shadow frustum onto the suspended cocoon assembly
  let dirLightIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirLightIdx === 0) {
        child.position.set(2.9, 4.4, 3.4);
        child.intensity = 2.05;
        child.shadow.mapSize.set(2048, 2048);
        const d = 2.1;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 0.8;
        child.shadow.camera.far = 12.5;
        child.shadow.bias = -0.0003;
        child.shadow.normalBias = 0.01;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirLightIdx === 1) {
        child.intensity = 0.72;
      } else if (dirLightIdx === 2) {
        child.intensity = 1.48;
      }
      dirLightIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.52;
    }
  }

  // Root container holding both inner sculptural core and shrink-film sleeve in lockstep local space
  const specimenRoot = new THREE.Group();
  specimenRoot.name = 'CocoonSpecimenRoot';
  stage.specimenGroup.add(specimenRoot);

  const materials = createCocoonMaterials();

  // Build all 3 hidden sculptural inner assemblies upfront (instant zero-GC cycling)
  const cores: BuiltCoreAssembly[] = [];
  for (let i = 0; i < CORE_COUNT; i++) {
    const built = buildCoreAssembly(i, materials.coreMaterials);
    built.group.visible = i === 0;
    specimenRoot.add(built.group);
    cores.push(built);
  }
  let activeCoreIndex = 0;

  // Subtle internal warm Realgar-Amber cavity light illuminating brass & basalt crevices from within
  const cavityLight = new THREE.PointLight(0xe57e36, 2.1, 4.2, 1.85);
  cavityLight.position.set(0, 0, 0);
  specimenRoot.add(cavityLight);

  // XPBD Heat-Shrink / Vacuum-Form Film Simulation
  const sim = new ShrinkFilmSimulation(cores[activeCoreIndex].descriptor, DEFAULT_COCOON_CONFIG);

  // Back-then-front dual pass over the shared geometry eliminates alpha self-sorting pops at every angle
  const filmBackMesh = new THREE.Mesh(sim.geometry, materials.filmBackMat);
  filmBackMesh.name = 'ShrinkFilmBackPass';
  filmBackMesh.renderOrder = 2;
  filmBackMesh.frustumCulled = false;
  specimenRoot.add(filmBackMesh);

  const filmFrontMesh = new THREE.Mesh(sim.geometry, materials.filmFrontMat);
  filmFrontMesh.name = 'ShrinkFilmFrontPass';
  filmFrontMesh.renderOrder = 3;
  filmFrontMesh.frustumCulled = false;
  specimenRoot.add(filmFrontMesh);

  // Delicate stretched fibrous silk tether strands spanning freshly opened tear gaps
  const silkTetherMesh = new THREE.Mesh(sim.tetherGeometry, materials.tetherMat);
  silkTetherMesh.name = 'SilkTetherStrands';
  silkTetherMesh.renderOrder = 4;
  silkTetherMesh.frustumCulled = false;
  specimenRoot.add(silkTetherMesh);

  // Cached raycast list per core (film front + back + active core so rays stop on exposed metal)
  const corePickLists: THREE.Object3D[][] = cores.map((c) => {
    const list: THREE.Object3D[] = [filmFrontMesh, filmBackMesh];
    c.group.traverse((obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh) list.push(obj);
    });
    return list;
  });

  // Interaction state
  const canvas = stage.shell.canvas;
  let isHoveringSleeve = false;
  let isHoldingHeat = false;
  let downOnSleeve = false;
  let downOnEmpty = false;
  let downShiftKey = false;
  let downX = 0;
  let downY = 0;
  let downTimeMs = 0;
  let lastEmptyClickMs = 0;
  let lastSleeveClickMs = 0;
  let heatVisualBlend = 0;
  let currentHeatWorldValid = false;

  // Boot-settle cursor guard (prevents parked OS cursor from melting resting Frame 0)
  const bootTimeMs = performance.now();
  let hoverUnlocked = false;
  let firstHoverX: number | null = null;
  let firstHoverY: number | null = null;

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function pickSleeveOrCore(clientX: number, clientY: number): {
    kind: 'sleeve' | 'core' | 'empty';
    point: THREE.Vector3 | null;
  } {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);
    const hits = _raycaster.intersectObjects(corePickLists[activeCoreIndex], false);
    if (hits.length === 0) {
      return { kind: 'empty', point: null };
    }
    const first = hits[0];
    if (first.object === filmFrontMesh || first.object === filmBackMesh) {
      return { kind: 'sleeve', point: first.point };
    }
    return { kind: 'core', point: first.point };
  }

  function cycleCoreAndRewrap(): void {
    cores[activeCoreIndex].group.visible = false;
    activeCoreIndex = (activeCoreIndex + 1) % CORE_COUNT;
    cores[activeCoreIndex].group.visible = true;
    sim.reset(cores[activeCoreIndex].descriptor, true);
    isHoveringSleeve = false;
    isHoldingHeat = false;
    currentHeatWorldValid = false;
    canvas.style.cursor = 'default';
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    hoverUnlocked = true;
    downX = ev.clientX;
    downY = ev.clientY;
    downTimeMs = performance.now();
    downShiftKey = ev.shiftKey;

    const pick = pickSleeveOrCore(ev.clientX, ev.clientY);
    if (pick.kind === 'sleeve' && pick.point) {
      downOnSleeve = true;
      downOnEmpty = false;
      isHoldingHeat = true;
      isHoveringSleeve = true;
      stage.controls.enabled = false;

      _hitWorld.copy(pick.point);
      currentHeatWorldValid = true;
      specimenRoot.updateMatrixWorld(true);
      _downHitLocal.copy(pick.point);
      specimenRoot.worldToLocal(_downHitLocal);

      stage.camera.getWorldDirection(_camDir);
      _dragPlane.setFromNormalAndCoplanarPoint(_camDir, pick.point);
      canvas.style.cursor = 'crosshair';
      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Ignore if pointer capture unavailable
      }
    } else {
      downOnSleeve = false;
      downOnEmpty = pick.kind === 'empty';
      isHoldingHeat = false;
      stage.controls.enabled = true;
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (isHoldingHeat) {
      const pick = pickSleeveOrCore(ev.clientX, ev.clientY);
      if (pick.kind === 'sleeve' && pick.point) {
        _hitWorld.copy(pick.point);
        currentHeatWorldValid = true;
        stage.camera.getWorldDirection(_camDir);
        _dragPlane.setFromNormalAndCoplanarPoint(_camDir, pick.point);
      } else {
        updateNdc(ev.clientX, ev.clientY);
        _raycaster.setFromCamera(_ndc, stage.camera);
        if (_raycaster.ray.intersectPlane(_dragPlane, _hitWorld)) {
          currentHeatWorldValid = true;
        }
      }
      return;
    }

    // Ignore passive hover while OrbitControls is dragging
    if (ev.buttons !== 0) {
      isHoveringSleeve = false;
      return;
    }

    // Boot-settle guard against parked OS cursor on initial load
    if (!hoverUnlocked) {
      if (performance.now() - bootTimeMs < 350) return;
      if (firstHoverX === null || firstHoverY === null) {
        firstHoverX = ev.clientX;
        firstHoverY = ev.clientY;
        return;
      }
      if (Math.hypot(ev.clientX - firstHoverX, ev.clientY - firstHoverY) < 8) return;
      hoverUnlocked = true;
    }

    const pick = pickSleeveOrCore(ev.clientX, ev.clientY);
    if (pick.kind === 'sleeve' && pick.point) {
      isHoveringSleeve = true;
      _hitWorld.copy(pick.point);
      currentHeatWorldValid = true;

      specimenRoot.updateMatrixWorld(true);
      _hitLocal.copy(pick.point);
      specimenRoot.worldToLocal(_hitLocal);
      const near = sim.findNearestVertex(_hitLocal.x, _hitLocal.y, _hitLocal.z);
      const readyToTear = near.localTension >= 0.10 || near.localShrink >= 0.22;
      canvas.style.cursor = readyToTear ? 'pointer' : 'crosshair';
    } else {
      isHoveringSleeve = false;
      canvas.style.cursor = 'default';
    }
  };

  const releaseCapture = (ev: PointerEvent) => {
    try {
      canvas.releasePointerCapture(ev.pointerId);
    } catch {
      // Ignore if capture was already released
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    const moveDist = Math.hypot(ev.clientX - downX, ev.clientY - downY);
    const heldMs = performance.now() - downTimeMs;
    const isClick = moveDist < 9 && heldMs < 320;

    if (downOnSleeve) {
      releaseCapture(ev);
      stage.controls.enabled = true;
      isHoldingHeat = false;

      if (isClick) {
        const now = performance.now();
        const isSleeveDoubleClick = now - lastSleeveClickMs < 360;
        lastSleeveClickMs = now;

        const forceRupture = downShiftKey || isSleeveDoubleClick;
        const ruptured = sim.triggerRuptureAtPoint(
          _downHitLocal.x,
          _downHitLocal.y,
          _downHitLocal.z,
          forceRupture
        );
        if (!ruptured) {
          // Gentle warm poke impulse on unheated soft film so clicks always feel tactile
          sim.applySoftPoke(_downHitLocal.x, _downHitLocal.y, _downHitLocal.z, 0.72);
        }
      }
      downOnSleeve = false;
      downOnEmpty = false;
      return;
    }

    if (downOnEmpty && isClick) {
      const now = performance.now();
      if (now - lastEmptyClickMs < 360) {
        lastEmptyClickMs = 0;
        cycleCoreAndRewrap();
      } else {
        lastEmptyClickMs = now;
      }
    }

    downOnSleeve = false;
    downOnEmpty = false;
  };

  const onPointerCancel = (ev: PointerEvent) => {
    releaseCapture(ev);
    isHoldingHeat = false;
    isHoveringSleeve = false;
    downOnSleeve = false;
    downOnEmpty = false;
    stage.controls.enabled = true;
    canvas.style.cursor = 'default';
  };

  const onPointerLeave = () => {
    if (!isHoldingHeat) {
      isHoveringSleeve = false;
    }
  };

  const onWheel = () => {
    hoverUnlocked = true;
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      cycleCoreAndRewrap();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('wheel', onWheel, { passive: true });
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerCancel);
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    canvas.removeEventListener('wheel', onWheel);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerCancel);
    window.removeEventListener('keydown', onKeyDown);
    stage.scene.environment = null;
    studioEnv.dispose();
    for (let i = 0; i < cores.length; i++) {
      cores[i].dispose();
    }
    sim.dispose();
    materials.dispose();
  });

  // Read-only diagnostic hook
  window.__DAY20__ = {
    getState: () => ({
      meanShrink: Number(sim.meanShrink.toFixed(4)),
      maxTension: Number(sim.maxTension.toFixed(4)),
      tearFraction: Number(sim.tearFraction.toFixed(4)),
      coreIndex: activeCoreIndex,
      vertexCount: sim.activeVertexCount,
      activeTetherCount: sim.activeTetherCount,
    }),
  };

  // Main animation tick (strictly zero allocations)
  stage.onTick(({ elapsedSec, dtSec, camera }) => {
    // Gentle unified cabinet-of-curiosities suspension float (core + cocoon stay 100% locked!)
    specimenRoot.position.y = Math.sin(elapsedSec * 0.85) * 0.018;
    specimenRoot.updateMatrixWorld(true);

    // Apply active heat beam when hovering or hold-dragging across the sleeve
    const heatingNow = (isHoveringSleeve || isHoldingHeat) && currentHeatWorldValid;
    if (heatingNow) {
      _hitLocal.copy(_hitWorld);
      specimenRoot.worldToLocal(_hitLocal);
      _camLocal.copy(camera.position);
      specimenRoot.worldToLocal(_camLocal);

      const beamRadius = isHoldingHeat ? 0.86 : 0.66;
      const beamPower = isHoldingHeat ? 2.85 : 1.35;
      sim.applyHeatBeam(
        _hitLocal.x,
        _hitLocal.y,
        _hitLocal.z,
        _camLocal.x,
        _camLocal.y,
        _camLocal.z,
        beamRadius,
        beamPower,
        dtSec
      );
      materials.uniforms.uHeatOrigin.value.copy(_hitWorld);
    }

    const targetHeatVisual = heatingNow ? (isHoldingHeat ? 1.0 : 0.62) : 0.0;
    heatVisualBlend += (targetHeatVisual - heatVisualBlend) * Math.min(1, dtSec * 12);

    // Step XPBD shrink-film physics, analytical core SDF projection, and seam propagation
    sim.step(dtSec);

    // Warm internal amber cavity breath + subtle heat response
    const cavityPulse = 0.92 + 0.10 * Math.sin(elapsedSec * 1.6) + heatVisualBlend * 0.25;
    materials.uniforms.uTime.value = elapsedSec;
    materials.uniforms.uAmberGlow.value = cavityPulse;
    materials.uniforms.uHeatActive.value = heatVisualBlend;
    cavityLight.intensity = 1.95 * cavityPulse;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  bootDay20Cocoon().catch((err) => {
    console.error('[Day 20 - Cocoon] Failed to initialize WebGPU specimen:', err);
  });
}
