import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  CAVITY_H,
  CAVITY_HALF_H,
  CAVITY_HALF_W,
  CAVITY_W,
  GRID_H,
  GRID_W,
} from './constants';
import { SandSimulation, type SandSimMetrics } from './sandSim';
import { buildTerrariumAssembly } from './geometry';
import { createBurrowMaterials, createBurrowStudioEnvironment } from './materials';

declare global {
  interface Window {
    __DAY27__?: {
      getState: () => SandSimMetrics & {
        ants: Array<{
          id: number;
          role: string;
          state: string;
          x: number;
          y: number;
          carryingPigment: number;
        }>;
      };
    };
  }
}

// Pre-allocated scratch math objects (strictly zero per-frame allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _localPt = new THREE.Vector3();
const _localCam = new THREE.Vector3();
const _invGroupMat = new THREE.Matrix4();

const FIXED_SIM_DT = 1 / 60;
const EXCAVATE_RADIUS_CELLS = 5.8;

async function mountDay27Burrow(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 27,
    cameraPosition: [0.34, 0.14, 3.02],
    fov: 33,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, -0.02, 0);
  stage.controls.minDistance = 0.88; // Allow true macro close-up inspection of worker ants & sand grains!
  stage.controls.maxDistance = 7.4;
  stage.controls.minPolarAngle = 0.08;
  stage.controls.maxPolarAngle = Math.PI - 0.08;
  stage.controls.update();

  // Calibrated Studio Softbox PMREM Environment for satin brass, oiled walnut, chitin & beveled glass
  const studioEnv = createBurrowStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.16;

  // Harmonize 3-point studio lighting & tighten 2048 shadow frustum to the narrow glass slice
  let dirLightIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirLightIdx === 0) {
        child.position.set(-2.3, 4.1, 3.8);
        child.intensity = 1.38;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.38;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 0.8;
        child.shadow.camera.far = 12.0;
        child.shadow.bias = -0.00024;
        child.shadow.normalBias = 0.01;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirLightIdx === 1) {
        child.position.set(3.4, 1.2, 2.4);
        child.intensity = 0.54;
      } else if (dirLightIdx === 2) {
        // Warm rear studio key/rim so orbiting 180 deg behind the terrarium stays crisp & luminous
        child.position.set(-1.6, 3.2, -3.9);
        child.intensity = 1.32;
      }
      dirLightIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.48;
    }
  }

  // Pure granular sand simulation + stigmergic 12-worker-ant colony + live RGBA8 WebGPU DataTextures
  const sim = new SandSimulation(1);

  const sandColorTex = new THREE.DataTexture(
    sim.colorTexData,
    GRID_W,
    GRID_H,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  sandColorTex.colorSpace = THREE.SRGBColorSpace;
  sandColorTex.flipY = false;
  sandColorTex.wrapS = THREE.ClampToEdgeWrapping;
  sandColorTex.wrapT = THREE.ClampToEdgeWrapping;
  sandColorTex.magFilter = THREE.LinearFilter;
  sandColorTex.minFilter = THREE.LinearFilter;
  sandColorTex.generateMipmaps = false;
  sandColorTex.needsUpdate = true;

  const sandFieldTex = new THREE.DataTexture(
    sim.fieldTexData,
    GRID_W,
    GRID_H,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  sandFieldTex.colorSpace = THREE.NoColorSpace;
  sandFieldTex.flipY = false;
  sandFieldTex.wrapS = THREE.ClampToEdgeWrapping;
  sandFieldTex.wrapT = THREE.ClampToEdgeWrapping;
  sandFieldTex.magFilter = THREE.LinearFilter;
  sandFieldTex.minFilter = THREE.LinearFilter;
  sandFieldTex.generateMipmaps = false;
  sandFieldTex.needsUpdate = true;

  const materials = createBurrowMaterials(sandColorTex, sandFieldTex);
  materials.syncPebbles(sim.pebbles);
  const assembly = buildTerrariumAssembly(materials);
  stage.specimenGroup.add(assembly.group);

  // Sync Frame 0 sand crest, all 12 sculpted 3D worker ants & particle buffers immediately!
  assembly.updateSandCrest(sim.surfaceY, sim.surfaceRgb);
  assembly.updateWorkerAnts(sim.diggers, 0);
  assembly.updateParticles3D(sim.grains3D, sim.dust3D, sim.pigmentRgbLinear);

  // Tactile interaction state:
  // - Drag across glass sand bed -> excavate custom branching tunnels & test arching avalanches
  // - Click & hold open top airspace -> pour cohesive hourglass stream of layered mineral sand
  // - Drag outside frame (empty space) -> smooth 360-deg 3D orbit (scroll to macro-zoom onto ants!)
  // - Double-click empty space (or press Space) -> reset fresh stratified ant farm bed
  const canvas = stage.shell.canvas;
  let activeMode: 'none' | 'excavate' | 'pour' = 'none';
  let hoverActive = false;
  let hoverIsPour = false;
  let hoverU = 0.5;
  let hoverV = 0.5;
  let activeGridX = GRID_W * 0.5;
  let activeGridY = GRID_H * 0.5;
  let prevGridX = activeGridX;
  let prevGridY = activeGridY;

  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTime = 0;
  let lastEmptyClickTime = 0;
  let downOnEmpty = false;

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastCavity(clientX: number, clientY: number): {
    hit: 'none' | 'frame' | 'sand' | 'air';
    u: number;
    v: number;
    gx: number;
    gy: number;
  } {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    const frameHits = _raycaster.intersectObjects(assembly.frameHitMeshes, false);
    const cavityHits = _raycaster.intersectObject(assembly.cavityHitMesh, false);

    const frameDist = frameHits.length > 0 ? frameHits[0].distance : Infinity;
    const cavDist = cavityHits.length > 0 ? cavityHits[0].distance : Infinity;

    if (cavityHits.length > 0 && cavDist <= frameDist + 0.09) {
      assembly.group.updateMatrixWorld();
      _localPt.copy(cavityHits[0].point);
      assembly.group.worldToLocal(_localPt);

      const u = Math.max(0.002, Math.min(0.998, (_localPt.x + CAVITY_HALF_W) / CAVITY_W));
      const v = Math.max(0.002, Math.min(0.998, (_localPt.y + CAVITY_HALF_H) / CAVITY_H));
      const gx = u * (GRID_W - 1);
      const gy = v * (GRID_H - 1);
      const inAir = sim.isInAirPourZone(gx, gy);
      return { hit: inAir ? 'air' : 'sand', u, v, gx, gy };
    }

    if (frameHits.length > 0) {
      return { hit: 'frame', u: 0.5, v: 0.5, gx: 0, gy: 0 };
    }
    return { hit: 'none', u: 0.5, v: 0.5, gx: 0, gy: 0 };
  }

  const releaseCaptureSafe = (pointerId: number) => {
    try {
      canvas.releasePointerCapture(pointerId);
    } catch {
      // Ignore if capture was already released
    }
  };

  const resetSandBed = () => {
    sim.reset(sim.seed + 1, true);
    materials.syncPebbles(sim.pebbles);
    sandColorTex.needsUpdate = true;
    sandFieldTex.needsUpdate = true;
    assembly.updateSandCrest(sim.surfaceY, sim.surfaceRgb);
    assembly.updateWorkerAnts(sim.diggers, 0);
    assembly.updateParticles3D(sim.grains3D, sim.dust3D, sim.pigmentRgbLinear);
  };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    pointerDownTime = performance.now();

    const res = raycastCavity(ev.clientX, ev.clientY);
    if (res.hit === 'sand' || res.hit === 'air') {
      downOnEmpty = false;
      stage.controls.enabled = false;
      canvas.setPointerCapture(ev.pointerId);

      hoverActive = true;
      hoverU = res.u;
      hoverV = res.v;
      activeGridX = res.gx;
      activeGridY = res.gy;
      prevGridX = res.gx;
      prevGridY = res.gy;

      if (res.hit === 'air') {
        activeMode = 'pour';
        hoverIsPour = true;
        canvas.style.cursor = 'crosshair';
        sim.pourSandAt(activeGridX, activeGridY, 10);
      } else {
        activeMode = 'excavate';
        hoverIsPour = false;
        canvas.style.cursor = 'crosshair';
        sim.excavateCapsule(
          activeGridX,
          activeGridY,
          activeGridX,
          activeGridY,
          EXCAVATE_RADIUS_CELLS,
          true
        );
      }
      sim.recomputeSurfaceAndOpticalFields();
      sim.packTextures();
      sandColorTex.needsUpdate = true;
      sandFieldTex.needsUpdate = true;
      assembly.updateSandCrest(sim.surfaceY, sim.surfaceRgb);
    } else {
      activeMode = 'none';
      downOnEmpty = res.hit === 'none';
      stage.controls.enabled = true;
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (activeMode === 'excavate' || activeMode === 'pour') {
      const res = raycastCavity(ev.clientX, ev.clientY);
      if (res.hit === 'sand' || res.hit === 'air') {
        hoverActive = true;
        hoverU = res.u;
        hoverV = res.v;
        activeGridX = res.gx;
        activeGridY = res.gy;

        if (activeMode === 'excavate') {
          sim.excavateCapsule(
            prevGridX,
            prevGridY,
            activeGridX,
            activeGridY,
            EXCAVATE_RADIUS_CELLS,
            true
          );
          prevGridX = activeGridX;
          prevGridY = activeGridY;
        }
      }
      return;
    }

    if (ev.buttons !== 0) return;

    const res = raycastCavity(ev.clientX, ev.clientY);
    if (res.hit === 'sand') {
      hoverActive = true;
      hoverIsPour = false;
      hoverU = res.u;
      hoverV = res.v;
      canvas.style.cursor = 'crosshair';
    } else if (res.hit === 'air') {
      hoverActive = true;
      hoverIsPour = true;
      hoverU = res.u;
      hoverV = res.v;
      canvas.style.cursor = 'cell';
    } else {
      hoverActive = false;
      canvas.style.cursor = 'default';
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (activeMode !== 'none') {
      releaseCaptureSafe(ev.pointerId);
      activeMode = 'none';
      stage.controls.enabled = true;
      canvas.style.cursor = hoverActive ? (hoverIsPour ? 'cell' : 'crosshair') : 'default';
      return;
    }

    const moveDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTime;
    if (downOnEmpty && moveDist < 7 && heldMs < 420) {
      const now = performance.now();
      if (now - lastEmptyClickTime < 360) {
        resetSandBed();
        lastEmptyClickTime = 0;
      } else {
        lastEmptyClickTime = now;
      }
    }
    downOnEmpty = false;
  };

  const onPointerCancel = (ev: PointerEvent) => {
    if (activeMode !== 'none') {
      releaseCaptureSafe(ev.pointerId);
    }
    activeMode = 'none';
    downOnEmpty = false;
    hoverActive = false;
    stage.controls.enabled = true;
    canvas.style.cursor = 'default';
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      resetSandBed();
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
    sandColorTex.dispose();
    sandFieldTex.dispose();
  });

  // Read-only live verification hook
  window.__DAY27__ = {
    getState: () => ({
      ...sim.getMetrics(),
      ants: sim.diggers.map((a) => ({
        id: a.id,
        role: a.role,
        state: a.state,
        x: Math.round(a.x * 10) / 10,
        y: Math.round(a.y * 10) / 10,
        carryingPigment: a.carryingPigment,
      })),
    }),
  };

  // Fixed-timestep physics accumulator & instanced worker-ant/granule sync (strictly zero allocations!)
  let simAccum = 0;
  let hoverAlphaSmooth = 0;

  stage.onTick(({ dtSec, elapsedSec, camera }) => {
    simAccum = Math.min(FIXED_SIM_DT * 3.5, simAccum + dtSec);

    while (simAccum >= FIXED_SIM_DT) {
      if (activeMode === 'pour') {
        sim.pourSandAt(activeGridX, activeGridY, 10);
      }
      sim.step();
      simAccum -= FIXED_SIM_DT;
    }

    if (sim.texturesDirty) {
      sim.syncOpticalTextures();
      sandColorTex.needsUpdate = true;
      sandFieldTex.needsUpdate = true;
      assembly.updateSandCrest(sim.surfaceY, sim.surfaceRgb);
    }

    // Update terrarium-local camera coordinate for 3D parallax interior tunnel walls & glass Fresnel
    assembly.group.updateMatrixWorld();
    _invGroupMat.copy(assembly.group.matrixWorld).invert();
    _localCam.copy(camera.position).applyMatrix4(_invGroupMat);
    materials.uCamLocal.value.copy(_localCam);
    materials.uTime.value = elapsedSec;

    // Smooth hover cursor cue uniform
    const targetHover = hoverActive ? 1 : 0;
    hoverAlphaSmooth += (targetHover - hoverAlphaSmooth) * Math.min(1, dtSec * 16);
    materials.uHoverUv.value.set(hoverU, hoverV, hoverIsPour ? 1 : 0, hoverAlphaSmooth);

    // Sync all 12 articulated 3D worker ants + live 3D tumbling mineral granules & avalanche dust!
    assembly.updateWorkerAnts(sim.diggers, elapsedSec);
    assembly.updateParticles3D(sim.grains3D, sim.dust3D, sim.pigmentRgbLinear);
  });
}

mountDay27Burrow().catch((err) => {
  console.error('[Day 27 - Burrow] Failed to initialize WebGPU specimen:', err);
});
