import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { MINERAL_PALETTE } from '../../core/palette';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  DeformableWaterCylinder,
  buildBaseCausticDiscGeometry,
  buildBorosilicateBeakerGeometry,
  buildGraduatedMarkingsGeometry,
  buildStirringRodGeometry,
  buildSubdividedUnitCubeGeometry,
  createGraduatedMarkingsTexture,
} from './beakerGeometry';
import {
  createBeakerStudioEnvironment,
  createDissolveMaterials,
} from './dissolveMaterials';
import { createDissolvePlumesSystem } from './dissolvePlumesCompute';
import {
  BEAKER_BOTTOM_OUTER_Y,
  BEAKER_INNER_FLOOR_Y,
  BEAKER_OUTER_RADIUS,
  BEAKER_TOP_Y,
  BUBBLE_COUNT,
  MAX_ACTIVE_CUBES,
  WATER_RADIUS,
  WATER_REST_Y,
  BeakerDissolveSim,
  evalCrystalErosionMetrics,
  type Day30TelemetryState,
} from './dissolveSim';

declare global {
  interface Window {
    __DAY30__?: {
      getState: () => Day30TelemetryState;
    };
  }
}

// Pre-allocated module-scope scratch objects (strictly zero per-frame GC allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _stirPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
const _meniscusPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -WATER_REST_Y);
const _planeHit = new THREE.Vector3();
const _localHit = new THREE.Vector3();
const _bubbleMat = new THREE.Matrix4();
const _bubblePos = new THREE.Vector3();
const _bubbleScale = new THREE.Vector3();
const _bubbleQuat = new THREE.Quaternion();

/**
 * Day 30 — Dissolve:
 * "Effervescent Mineral Cubes in a Borosilicate Glass Beaker"
 *
 * Interactive Contract:
 *  - Click over the beaker (or tap Space): Drops a fresh crisp crystalline mineral cube
 *    (cycling celadonGlauconite -> realgarAmber -> azuritePrussian -> cinnabarVermilion -> yellowOchre,
 *    up to 4 concurrent dissolving cubes mixing colors in the bath) with an immediate splash ripple.
 *  - Drag inside the beaker: Smoothly dips and sweeps the slender borosilicate glass stirring rod
 *    to whip up a central vortex swirling cubes, silver bubbles, and marbled schlieren pigment ribbons.
 *  - Drag empty space outside the beaker: Smooth 360-degree 3D orbit around the suspended specimen.
 *  - Double-click empty space: Flushes the beaker clean with clear water and drops one crisp starter cube.
 */
export async function bootDay30Dissolve(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(30);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.72, 4.38],
    fov: 34,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0.0, 0.02, 0.0);
  stage.controls.minDistance = 2.05;
  stage.controls.maxDistance = 7.6;
  stage.controls.minPolarAngle = 0.16;
  stage.controls.maxPolarAngle = Math.PI * 0.88;
  stage.controls.rotateSpeed = 0.62;
  stage.controls.update();

  // Glass-lab photographer's PMREM studio softbox reflections (curved strip windows + calcite rim)
  const studioEnv = createBeakerStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize key/rim shadows tightly around the suspended laboratory beaker
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(2.8, 4.4, 3.3);
        child.intensity = 2.45;
        child.shadow.mapSize.set(1024, 1024);
        const d = 1.5;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 11.5;
        child.shadow.bias = -0.00035;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        child.intensity = 0.92;
      } else if (dirIdx === 2) {
        child.intensity = 1.65;
      }
      dirIdx++;
    }
  }

  // Subtle warm studio uplift beneath the thick solid borosilicate base puck
  const baseBounce = new THREE.PointLight(0xeae3d2, 0.95, 5.2, 1.8);
  baseBounce.position.set(0.25, -1.75, 0.45);
  stage.specimenGroup.add(baseBounce);

  // -------------------------------------------------------------------------
  // Core Physical Simulation, Procedural Watertight Meshes & Studio Materials
  // -------------------------------------------------------------------------
  const sim = new BeakerDissolveSim(true);
  const markingsTex = createGraduatedMarkingsTexture();
  const mats = createDissolveMaterials(sim, markingsTex);

  const beakerGroup = new THREE.Group();
  beakerGroup.name = 'BorosilicateDissolveApparatus';
  stage.specimenGroup.add(beakerGroup);

  // 1. Four Pre-Allocated Dissolving Crystalline Mineral Cube Slots (`renderOrder = 1`)
  const unitCubeGeom = buildSubdividedUnitCubeGeometry(18);
  const cubeMeshes: THREE.Mesh[] = [];
  const lastSlotPigment: Array<string> = new Array(MAX_ACTIVE_CUBES).fill('');

  for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
    const mesh = new THREE.Mesh(unitCubeGeom, mats.cubeHandles[i].material);
    mesh.name = `DissolvingMineralCube_${i}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.renderOrder = 1;
    mesh.frustumCulled = false;
    mesh.visible = false;
    beakerGroup.add(mesh);
    cubeMeshes.push(mesh);
  }

  // 2. Thick-Walled Borosilicate Glass Beaker (Back-Face Depth Pass = 2, Front-Face Reflections = 8)
  const beakerGeom = buildBorosilicateBeakerGeometry(96);

  const beakerBackMesh = new THREE.Mesh(beakerGeom, mats.beakerBackMat);
  beakerBackMesh.name = 'BorosilicateBeakerBackShell';
  beakerBackMesh.renderOrder = 2;
  beakerGroup.add(beakerBackMesh);

  // 3. 16,384-Particle WebGPU TSL-Compute Descending Schlieren & Marbled Pigment Plumes (`renderOrder = 3`)
  const plumes = createDissolvePlumesSystem();
  beakerGroup.add(plumes.sprite);
  plumes.runInit(stage.renderer);

  // 4. 2,000-Instance Rising Effervescent Silver Micro-Bubble Streams (`renderOrder = 4`)
  const bubbleSphereGeom = new THREE.SphereGeometry(1.0, 10, 8);
  const bubbleInstancedMesh = new THREE.InstancedMesh(
    bubbleSphereGeom,
    mats.bubbleMat,
    BUBBLE_COUNT
  );
  bubbleInstancedMesh.name = 'EffervescentSilverBubbles';
  bubbleInstancedMesh.renderOrder = 4;
  bubbleInstancedMesh.frustumCulled = false;
  beakerGroup.add(bubbleInstancedMesh);

  // 5. Slender Tactile Borosilicate Glass Stirring Rod (`renderOrder = 5`)
  const rodGeom = buildStirringRodGeometry(0.024, 1.92);
  const rodMesh = new THREE.Mesh(rodGeom, mats.rodMat);
  rodMesh.name = 'BorosilicateStirringRod';
  rodMesh.renderOrder = 5;
  rodMesh.visible = false;
  beakerGroup.add(rodMesh);

  // 6. Watertight Deformable Water Volume (Shared Perimeter Vertices, `renderOrder = 6`)
  const waterCylinder = new DeformableWaterCylinder(mats.waterMat);
  waterCylinder.mesh.renderOrder = 6;
  waterCylinder.updateFromSim(sim);
  beakerGroup.add(waterCylinder.mesh);

  // 7. Thick Glass Base Caustic & Solute Pool Shimmer (`renderOrder = 7`)
  const causticDiscGeom = buildBaseCausticDiscGeometry();
  const causticDiscMesh = new THREE.Mesh(causticDiscGeom, mats.causticDiscMat);
  causticDiscMesh.name = 'BeakerBaseCausticDisc';
  causticDiscMesh.renderOrder = 7;
  beakerGroup.add(causticDiscMesh);

  // 8. Front-Face Thick Borosilicate Glass Wall & Flared Pouring Spout (`renderOrder = 8`)
  const beakerFrontMesh = new THREE.Mesh(beakerGeom, mats.beakerFrontMat);
  beakerFrontMesh.name = 'BorosilicateBeakerFrontShell';
  beakerFrontMesh.renderOrder = 8;
  beakerGroup.add(beakerFrontMesh);

  // 9. Graduated Etched Eggshell Volumetric Markings (`renderOrder = 9`)
  const markingsGeom = buildGraduatedMarkingsGeometry();
  const markingsMesh = new THREE.Mesh(markingsGeom, mats.markingsMat);
  markingsMesh.name = 'GraduatedVolumetricMarkings';
  markingsMesh.renderOrder = 9;
  beakerGroup.add(markingsMesh);

  // 10. Invisible Cylinder Collider exclusively for crisp Pointer Raycasting over the Beaker
  const hitHeight = BEAKER_TOP_Y - BEAKER_BOTTOM_OUTER_Y + 0.16;
  const hitCylGeom = new THREE.CylinderGeometry(
    BEAKER_OUTER_RADIUS + 0.07,
    BEAKER_OUTER_RADIUS + 0.07,
    hitHeight,
    32
  );
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const beakerHitMesh = new THREE.Mesh(hitCylGeom, hitMat);
  beakerHitMesh.position.y = (BEAKER_TOP_Y + BEAKER_BOTTOM_OUTER_Y) * 0.5 + 0.06;
  beakerGroup.add(beakerHitMesh);

  // -------------------------------------------------------------------------
  // Tactile Interaction: Click to Drop Cube | Drag Inside to Stir | Orbit Outside | DblClick to Flush
  // -------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  let isPointerDownOnBeaker = false;
  let isStirring = false;
  let activePointerId: number | null = null;
  let downClientX = 0;
  let downClientY = 0;
  let downTimeMs = 0;
  let stirTargetX = 0.24;
  let stirTargetZ = 0.14;
  let dropTargetX = 0;
  let dropTargetZ = 0;

  const updateNdc = (clientX: number, clientY: number): boolean => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    _ndc.set(
      ((clientX - rect.left) / rect.width) * 2.0 - 1.0,
      -((clientY - rect.top) / rect.height) * 2.0 + 1.0
    );
    return true;
  };

  const raycastBeaker = (clientX: number, clientY: number): boolean => {
    if (!updateNdc(clientX, clientY)) return false;
    _raycaster.setFromCamera(_ndc, stage.camera);
    const hits = _raycaster.intersectObject(beakerHitMesh, false);
    return hits.length > 0;
  };

  const projectPointerToBeakerXZ = (
    clientX: number,
    clientY: number,
    useMeniscusPlane: boolean,
    outTarget: { x: number; z: number }
  ): boolean => {
    if (!updateNdc(clientX, clientY)) return false;
    _raycaster.setFromCamera(_ndc, stage.camera);
    const plane = useMeniscusPlane ? _meniscusPlane : _stirPlane;
    if (_raycaster.ray.intersectPlane(plane, _planeHit) !== null) {
      _localHit.copy(_planeHit);
      beakerGroup.worldToLocal(_localHit);
      const maxR = WATER_RADIUS - 0.06;
      const r = Math.hypot(_localHit.x, _localHit.z);
      if (r > maxR && r > 1e-4) {
        outTarget.x = (_localHit.x / r) * maxR;
        outTarget.z = (_localHit.z / r) * maxR;
      } else {
        outTarget.x = _localHit.x;
        outTarget.z = _localHit.z;
      }
      return true;
    }
    return false;
  };

  const _projScratch = { x: 0, z: 0 };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    downClientX = ev.clientX;
    downClientY = ev.clientY;
    downTimeMs = performance.now();

    if (raycastBeaker(ev.clientX, ev.clientY)) {
      isPointerDownOnBeaker = true;
      isStirring = false;
      activePointerId = ev.pointerId;
      stage.controls.enabled = false;

      if (projectPointerToBeakerXZ(ev.clientX, ev.clientY, true, _projScratch)) {
        dropTargetX = _projScratch.x;
        dropTargetZ = _projScratch.z;
      }
      if (projectPointerToBeakerXZ(ev.clientX, ev.clientY, false, _projScratch)) {
        stirTargetX = _projScratch.x;
        stirTargetZ = _projScratch.z;
        sim.rod.x = stirTargetX;
        sim.rod.z = stirTargetZ;
      }

      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Synthetic Playwright events may not carry pointer capture handles
      }
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (isPointerDownOnBeaker) {
      const movedPx = Math.hypot(ev.clientX - downClientX, ev.clientY - downClientY);
      if (!isStirring && movedPx >= 6) {
        isStirring = true;
        canvas.style.cursor = 'grabbing';
      }
      if (isStirring && projectPointerToBeakerXZ(ev.clientX, ev.clientY, false, _projScratch)) {
        stirTargetX = _projScratch.x;
        stirTargetZ = _projScratch.z;
      }
      return;
    }

    if (ev.buttons !== 0) return;
    const overBeaker = raycastBeaker(ev.clientX, ev.clientY);
    canvas.style.cursor = overBeaker ? 'pointer' : 'grab';
  };

  const finishBeakerPointer = (ev: PointerEvent, allowDropClick: boolean) => {
    if (!isPointerDownOnBeaker) return;
    if (activePointerId !== null && ev.pointerId !== activePointerId) return;

    const movedPx = Math.hypot(ev.clientX - downClientX, ev.clientY - downClientY);
    const heldMs = performance.now() - downTimeMs;
    const wasCleanClick = allowDropClick && !isStirring && movedPx < 8 && heldMs < 480;

    isPointerDownOnBeaker = false;
    isStirring = false;
    activePointerId = null;
    stage.controls.enabled = true;
    sim.rod.active = false;

    try {
      if (canvas.hasPointerCapture(ev.pointerId)) {
        canvas.releasePointerCapture(ev.pointerId);
      }
    } catch {
      // Ignore capture release errors on synthetic events
    }

    if (wasCleanClick) {
      sim.dropCube(dropTargetX * 0.72, dropTargetZ * 0.72);
    }

    const overBeaker = raycastBeaker(ev.clientX, ev.clientY);
    canvas.style.cursor = overBeaker ? 'pointer' : 'grab';
  };

  const onPointerUp = (ev: PointerEvent) => finishBeakerPointer(ev, true);
  const onPointerCancel = (ev: PointerEvent) => finishBeakerPointer(ev, false);

  const onDoubleClick = (ev: MouseEvent) => {
    // Double-click empty space outside the beaker -> flush clean & drop one crisp starter crystal
    if (!raycastBeaker(ev.clientX, ev.clientY)) {
      sim.flushBeaker(true);
      const starter = sim.cubes[0];
      const starterRgb = MINERAL_PALETTE[starter.pigmentKey].rInfLinear;
      plumes.reset(stage.renderer, starterRgb);
    }
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      sim.dropCube();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown, true);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });
  window.addEventListener('pointercancel', onPointerCancel, { passive: true });
  canvas.addEventListener('dblclick', onDoubleClick);
  window.addEventListener('keydown', onKeyDown);

  // Expose read-only verification telemetry (`window.__DAY30__.getState()`)
  window.__DAY30__ = {
    getState: () => sim.getTelemetry(),
  };

  // Helper to sync all 2,000 bubble instances without heap allocations
  const syncBubbleInstances = () => {
    for (let i = 0; i < BUBBLE_COUNT; i++) {
      if (sim.bubbleAlive[i] === 0) {
        _bubblePos.set(0, -10, 0);
        _bubbleScale.set(0, 0, 0);
      } else {
        const r = sim.bubbleRadius[i];
        // Slight hydrodynamic oblate flattening on larger rising bubbles
        const oblate = 1.0 - Math.min(0.22, r * 12.0);
        _bubblePos.set(sim.bubblePosX[i], sim.bubblePosY[i], sim.bubblePosZ[i]);
        _bubbleScale.set(r, r * oblate, r);
      }
      _bubbleMat.compose(_bubblePos, _bubbleQuat, _bubbleScale);
      bubbleInstancedMesh.setMatrixAt(i, _bubbleMat);
    }
    bubbleInstancedMesh.instanceMatrix.needsUpdate = true;
  };

  // -------------------------------------------------------------------------
  // Main Animation Tick (Zero Per-Frame Allocations)
  // -------------------------------------------------------------------------
  stage.onTick(({ dtSec, elapsedSec, renderer }) => {
    const dt = Math.min(0.04, Math.max(0.001, dtSec));

    // Gentle laboratory buoyancy hover on the suspended beaker plinth
    beakerGroup.position.y = Math.sin(elapsedSec * 0.72) * 0.012;

    // 1. Update stirring rod interaction & advance pure physical dissolution simulation
    sim.setStirringRodTarget(isStirring, stirTargetX, stirTargetZ, dt);
    sim.step(dt);

    // 2. Sync dissolving mineral cubes (GPU curvature-weighted corner rounding & pitting uniforms)
    for (let i = 0; i < MAX_ACTIVE_CUBES; i++) {
      const c = sim.cubes[i];
      const mesh = cubeMeshes[i];
      const handle = mats.cubeHandles[i];

      if (!c.active || c.mass <= 1e-5) {
        mesh.visible = false;
        continue;
      }

      mesh.visible = true;
      mesh.position.copy(c.pos);
      mesh.quaternion.copy(c.quat);

      if (lastSlotPigment[i] !== c.pigmentKey) {
        lastSlotPigment[i] = c.pigmentKey;
        handle.setPigment(c.pigmentKey);
      }

      const m = evalCrystalErosionMetrics(c.mass);
      handle.uShapeParams.value.set(m.halfExtent, m.coreHalf, m.filletRadius, m.pitAmp);
      handle.uAspectSeed.value.set(m.scaleXZ, m.scaleY, c.seed, m.progress);
    }

    // 3. Sync tactile borosilicate glass stirring rod immersion & tilt
    if (sim.rod.dip > 0.015) {
      rodMesh.visible = true;
      const rodBottomY =
        BEAKER_TOP_Y + 0.35 - sim.rod.dip * (BEAKER_TOP_Y - BEAKER_INNER_FLOOR_Y + 0.24);
      rodMesh.position.set(sim.rod.x, rodBottomY, sim.rod.z);
      rodMesh.rotation.set(sim.rod.tiltX, 0, sim.rod.tiltZ);
    } else {
      rodMesh.visible = false;
    }

    // 4. Update watertight deformable water column mesh from 2D wave PDE + vortex funnel
    waterCylinder.updateFromSim(sim);

    // 5. Sync 2,000 rising effervescent silver micro-bubbles & shared optical uniforms
    syncBubbleInstances();
    mats.syncFromSim(sim);

    // 6. Dispatch 16,384-particle WebGPU TSL-Compute descending schlieren & swirling pigment plumes
    plumes.update(renderer, sim, dt);
  });

  // Full resource cleanup on page unload / dispose
  stage.shell.registerDispose(() => {
    canvas.style.cursor = '';
    canvas.removeEventListener('pointerdown', onPointerDown, true);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerCancel);
    canvas.removeEventListener('dblclick', onDoubleClick);
    window.removeEventListener('keydown', onKeyDown);

    stage.scene.environment = null;
    studioEnv.dispose();
    plumes.dispose();
    sim.dispose();
    mats.dispose();
    markingsTex.dispose();
    unitCubeGeom.dispose();
    beakerGeom.dispose();
    bubbleSphereGeom.dispose();
    rodGeom.dispose();
    waterCylinder.dispose();
    causticDiscGeom.dispose();
    markingsGeom.dispose();
    hitCylGeom.dispose();
    hitMat.dispose();
    delete window.__DAY30__;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay30Dissolve();
}
