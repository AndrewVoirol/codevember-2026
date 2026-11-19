import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { EarthstarBellowsPhysics } from './bellowsPhysics';
import {
  buildExoperidiumPetalsGeometry,
  buildMycelialRootletsGeometry,
  buildSporeSacGeometry,
} from './earthstarGeometry';
import { createEarthstarMaterials } from './earthstarMaterials';
import { SporeAcoustics } from './sporeAudio';
import { createSporeComputeSystem } from './sporeCompute';

// Zero-GC module-scope scratch objects for pointer raycasting & world conversions
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _localHit = new THREE.Vector3();
const _prevHit = new THREE.Vector3();
const _plumePlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const _camDir = new THREE.Vector3();
const _plumeOrigin = new THREE.Vector3(0, 0.65, 0);
const _plumePointerWorld = new THREE.Vector3(0, 0.65, 0);
const _prevPlumePointerWorld = new THREE.Vector3(0, 0.65, 0);
const _plumePointerVel = new THREE.Vector3(0, 0, 0);
const _deltaScratch = new THREE.Vector3();

/**
 * Day 16 — Spore:
 * "Earthstar Puffball (Geastrum triplex) Pneumatic Bellows & Toroidal Vortex-Ring Plume"
 *
 * Interaction Contract:
 *   - Click / tap the papery spore sac -> tactile elastic thumb-dimple indentation +
 *     crisp pneumatic toroidal vortex smoke ring erupting from the fluted peristome.
 *   - Press, hold, or knead across the spore sac -> compresses the bellows deeply
 *     and launches a fuller secondary rebound vortex ring on release; rapid taps
 *     launch multiple mutually leapfrogging vortex rings.
 *   - Sweep cursor across the airborne golden spore plume -> parts and stirs real-time
 *     3D curl-noise eddies in the 49,152-particle WebGPU TSL-compute cloud.
 *   - Drag on open Chalky Slate space -> orbits freely around the suspended specimen;
 *     scroll/pinch zooms cleanly in to inspect the cracked exoperidium & fluted chimney.
 */
export async function bootDay16Spore(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(16);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.56, 5.65],
    fov: 34,
    enableOrbit: true,
    enableShadows: true,
  });

  // Aim studio camera slightly upward so the suspended Earthstar sits in the lower-center
  // third and leaves generous Chalky Slate headroom above the peristome for vortex rings.
  stage.controls.target.set(0, 0.16, 0);
  stage.controls.minDistance = 2.2;
  stage.controls.maxDistance = 8.8;
  stage.controls.maxPolarAngle = Math.PI * 0.86;
  stage.controls.minPolarAngle = Math.PI * 0.12;
  stage.controls.update();

  // Build Earthstar specimen geometry & physical mineral-pigment TSL materials
  const matRig = createEarthstarMaterials();
  const sacGeom = buildSporeSacGeometry();
  const petalsGeom = buildExoperidiumPetalsGeometry();
  const rootsGeom = buildMycelialRootletsGeometry();

  const earthstarGroup = new THREE.Group();
  earthstarGroup.name = 'GeastrumTriplexSpecimen';
  stage.specimenGroup.add(earthstarGroup);

  const sacMesh = new THREE.Mesh(sacGeom, matRig.sacMaterial);
  sacMesh.name = 'EndoperidiumSporeSac';
  sacMesh.castShadow = true;
  sacMesh.receiveShadow = true;
  earthstarGroup.add(sacMesh);

  const petalsMesh = new THREE.Mesh(petalsGeom, matRig.petalMaterial);
  petalsMesh.name = 'ExoperidiumStarPetals';
  petalsMesh.castShadow = true;
  petalsMesh.receiveShadow = true;
  earthstarGroup.add(petalsMesh);

  const rootsMesh = new THREE.Mesh(rootsGeom, matRig.rootMaterial);
  rootsMesh.name = 'MycelialRootletsAndMoss';
  rootsMesh.castShadow = true;
  rootsMesh.receiveShadow = false;
  earthstarGroup.add(rootsMesh);

  // Initialize pure-TS pneumatic bellows & multi-ring Biot-Savart physics + WebGPU TSL compute
  const physics = new EarthstarBellowsPhysics();
  physics.seedInitialHeroRing();

  const sporeCompute = createSporeComputeSystem();
  stage.specimenGroup.add(sporeCompute.sprite);
  sporeCompute.runInit(stage.renderer);

  const acoustics = new SporeAcoustics();

  // Pointer interaction state (clean separation between sac squeeze vs. space orbit)
  const canvas = stage.shell.canvas;
  let isHoveringSac = false;
  let isSqueezingSac = false;
  let activePointerId: number | null = null;
  let plumePointerInitialized = false;
  let plumePointerSpeed = 0;

  const updateNdcFromClient = (clientX: number, clientY: number): boolean => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    _ndc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    return true;
  };

  const raycastEarthstar = (includePetals: boolean): THREE.Intersection | null => {
    _raycaster.setFromCamera(_ndc, stage.camera);
    const hits = includePetals
      ? _raycaster.intersectObjects([sacMesh, petalsMesh], false)
      : _raycaster.intersectObject(sacMesh, false);
    return hits.length > 0 ? hits[0] : null;
  };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    acoustics.unlock();
    if (!updateNdcFromClient(ev.clientX, ev.clientY)) return;

    const hit = raycastEarthstar(true);
    if (hit) {
      // User pressed directly on the puffball specimen -> engage pneumatic squeeze & disable OrbitControls
      isSqueezingSac = true;
      activePointerId = ev.pointerId;
      stage.controls.enabled = false;
      canvas.style.cursor = 'grabbing';

      _localHit.copy(hit.point);
      earthstarGroup.worldToLocal(_localHit);
      _prevHit.copy(_localHit);
      physics.beginSqueeze(_localHit.x, _localHit.y, _localHit.z);
      acoustics.playPuff(0.82);

      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Synthetic Playwright events may not carry an active capture handle
      }
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (!updateNdcFromClient(ev.clientX, ev.clientY)) return;

    if (isSqueezingSac) {
      const hit = raycastEarthstar(true);
      if (hit) {
        _localHit.copy(hit.point);
        earthstarGroup.worldToLocal(_localHit);
        const dragDelta = _localHit.distanceTo(_prevHit);
        _prevHit.copy(_localHit);
        physics.updateSqueezePoint(_localHit.x, _localHit.y, _localHit.z, dragDelta);
      }
      return;
    }

    const hoverHit = raycastEarthstar(true);
    isHoveringSac = hoverHit !== null;
    canvas.style.cursor = isHoveringSac ? 'pointer' : 'grab';
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (!isSqueezingSac) return;
    if (activePointerId !== null && ev.pointerId !== activePointerId) return;

    const wasLongHold = physics.squeezeHoldSec > 0.24;
    isSqueezingSac = false;
    activePointerId = null;
    stage.controls.enabled = true;
    physics.endSqueeze();
    if (wasLongHold) {
      acoustics.playPuff(1.0);
    }

    canvas.style.cursor = isHoveringSac ? 'pointer' : 'grab';
    try {
      if (canvas.hasPointerCapture(ev.pointerId)) {
        canvas.releasePointerCapture(ev.pointerId);
      }
    } catch {
      // Ignore capture release errors on synthetic events
    }
  };

  const onDoubleClick = (ev: MouseEvent) => {
    acoustics.unlock();
    if (!updateNdcFromClient(ev.clientX, ev.clientY)) return;
    const hit = raycastEarthstar(true);
    if (!hit) {
      // Double-clicking open space launches a full showcase double-ring puff without moving camera
      physics.triggerPuff(1.05, 0, 0);
      acoustics.playPuff(1.0);
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown, true);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });
  window.addEventListener('pointercancel', onPointerUp, { passive: true });
  canvas.addEventListener('dblclick', onDoubleClick);

  // Expose read-only QA & playtest diagnostic telemetry (`window.__SPORE__`)
  window.__SPORE__ = {
    getState: () => {
      const snap = physics.getSnapshot();
      const ringHeights: number[] = [];
      const ringRadii: number[] = [];
      for (let i = 0; i < snap.rings.length; i++) {
        if (snap.rings[i].active) {
          ringHeights.push(Number(snap.rings[i].center[1].toFixed(3)));
          ringRadii.push(Number(snap.rings[i].majorRadius.toFixed(3)));
        }
      }
      return {
        totalPuffs: snap.totalPuffs,
        activeRings: snap.activeRingCount,
        compression: Number(snap.compression.toFixed(4)),
        pressure: Number(snap.pressure.toFixed(4)),
        peristomeFlare: Number(snap.peristomeFlare.toFixed(4)),
        petalRecoil: Number(snap.petalRecoil.toFixed(4)),
        particleCount: sporeCompute.particleCount,
        ringHeights,
        ringRadii,
      };
    },
    triggerPuff: (strength = 0.95, pokeX = 0, pokeZ = 0) => {
      physics.triggerPuff(strength, pokeX, pokeZ);
    },
  };

  // Main per-frame simulation & WebGPU compute tick (zero per-frame allocations)
  stage.onTick(({ dtSec, elapsedSec, pointer, camera, renderer }) => {
    const dt = Math.min(0.05, Math.max(0.001, dtSec));

    // Gentle suspended studio micro-bob (never fights raycasting or feels mechanical)
    earthstarGroup.position.y = Math.sin(elapsedSec * 0.85) * 0.014;
    earthstarGroup.rotation.y = Math.sin(elapsedSec * 0.24) * 0.035;

    // Project pointer NDC onto the camera-facing plane intersecting the airborne plume center
    camera.getWorldDirection(_camDir).negate();
    _plumePlane.setFromNormalAndCoplanarPoint(_camDir, _plumeOrigin);
    _raycaster.setFromCamera(pointer.ndc, camera);
    const hitPlume = _raycaster.ray.intersectPlane(_plumePlane, _plumePointerWorld);
    if (hitPlume) {
      if (!plumePointerInitialized || !pointer.isActive) {
        _prevPlumePointerWorld.copy(_plumePointerWorld);
        plumePointerInitialized = pointer.isActive;
      }
      _deltaScratch.subVectors(_plumePointerWorld, _prevPlumePointerWorld).divideScalar(dt);
      const alpha = 1 - Math.exp(-dt * 16.0);
      _plumePointerVel.lerp(_deltaScratch, alpha);
      plumePointerSpeed = _plumePointerVel.length();
      _prevPlumePointerWorld.copy(_plumePointerWorld);
    }

    // Step pneumatic bellows & multi-ring Biot-Savart physics
    const stepResult = physics.step(dt, isHoveringSac);

    // Sync TSL material deformation uniforms
    matRig.uCompression.value = physics.compression;
    matRig.uIndentStrength.value = physics.indentStrength;
    matRig.uIndentDir.value.set(physics.indentDir[0], physics.indentDir[1], physics.indentDir[2]);
    matRig.uPeristomeFlare.value = physics.peristomeFlare;
    matRig.uPetalRecoil.value = physics.petalRecoil;
    matRig.uHoverWarmth.value = physics.hoverWarmth;
    matRig.uTime.value = elapsedSec;

    // Dispatch WebGPU TSL compute particle advection & spawning
    sporeCompute.update(
      renderer,
      physics,
      stepResult,
      dt,
      elapsedSec,
      _plumePointerWorld,
      _plumePointerVel,
      pointer.isActive ? plumePointerSpeed : 0,
      pointer.isDown && !isSqueezingSac
    );
  });

  // Register full cleanup on page dispose
  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown, true);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerUp);
    canvas.removeEventListener('dblclick', onDoubleClick);
    acoustics.dispose();
    sporeCompute.dispose();
    sacGeom.dispose();
    petalsGeom.dispose();
    rootsGeom.dispose();
    matRig.sacMaterial.dispose();
    matRig.petalMaterial.dispose();
    matRig.rootMaterial.dispose();
    delete window.__SPORE__;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay16Spore();
}
