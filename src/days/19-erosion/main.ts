import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { buildAmmoniteGeometry } from './ammonite';
import { ErosionField, type ErosionRaycastHit } from './erosionField';
import { buildNoduleShellGeometry, buildNozzleAssembly } from './noduleGeometry';
import { NozzleController, type NozzlePoseInput } from './nozzleKinematics';
import { AbrasionParticleSystem } from './particles';
import { createErosionMaterials, createStudioEnvironment } from './materials';

declare global {
  interface Window {
    __DAY19__?: {
      getState: () => {
        exposedFraction: number;
        totalCarvedVolume: number;
        isBlasting: boolean;
        activeParticles: number;
        cameraPos: [number, number, number];
      };
    };
  }
}

// Pre-allocated scratch math objects (strictly zero per-frame allocations inside onTick)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _rayHit: ErosionRaycastHit = {
  position: new THREE.Vector3(),
  normal: new THREE.Vector3(0, 0, 1),
  direction: new THREE.Vector3(0, 0, 1),
  distance: 0,
  isFossil: false,
};
const _lastHitPos = new THREE.Vector3(0.42, 0.25, 0.44);
const _lastHitNormal = new THREE.Vector3(0.38, 0.22, 0.90).normalize();
const _lastBlastDir = new THREE.Vector3();
const _interpDir = new THREE.Vector3();
const _camUp = new THREE.Vector3(0, 1, 0);
const _poseInput: NozzlePoseInput = {
  isHoveringRock: false,
  isBlasting: false,
  isOrbiting: false,
  hitPos: _lastHitPos,
  hitNormal: _lastHitNormal,
  cameraPos: new THREE.Vector3(),
  cameraUp: _camUp,
  elapsedSec: 0,
};

async function mountDay19Erosion(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 19,
    cameraPosition: [0.0, 0.14, 4.35],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, 0, 0);
  stage.controls.enableDamping = true;
  stage.controls.dampingFactor = 0.068;
  stage.controls.minDistance = 2.05;
  stage.controls.maxDistance = 7.6;
  stage.controls.minPolarAngle = 0.06;
  stage.controls.maxPolarAngle = Math.PI - 0.06;
  stage.controls.update();

  // Studio Softbox PMREM Environment so pyritized ammonite ribs & satin brass gleam against #212529
  const studioEnv = createStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize 3-point studio lighting & upgrade shadow map to 2048x2048 tightly fitted to the nodule
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(-2.6, 4.5, 3.8);
        child.intensity = 2.15;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.75;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.0;
        child.shadow.bias = -0.00025;
        child.shadow.normalBias = 0.015;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        child.position.set(3.4, -0.8, 2.5);
        child.intensity = 0.65;
      } else if (dirIdx === 2) {
        child.position.set(-1.4, 3.2, -4.0);
        child.intensity = 1.42;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.52;
    }
  }

  // 1. Embedded Watertight 3D Pyritized Ammonite Relic (hard differential-hardness core)
  const ammoniteBuilt = buildAmmoniteGeometry(280, 36);
  const erosionField = new ErosionField(ammoniteBuilt.geometry, 1);

  // 2. TSL Materials & Shared Uniforms
  const materials = createErosionMaterials(erosionField.dataTexture);
  materials.uStrataAxis.value.copy(erosionField.strataAxis);
  materials.uStrataSeed.value = erosionField.strataSeed;

  const ammoniteMesh = new THREE.Mesh(ammoniteBuilt.geometry, materials.ammonite);
  ammoniteMesh.name = 'PyritizedAmmoniteRelic';
  ammoniteMesh.castShadow = true;
  ammoniteMesh.receiveShadow = true;
  stage.specimenGroup.add(ammoniteMesh);

  // 3. Carvable Stratified Limestone Matrix Nodule (40,962-vertex welded geodesic shell)
  const noduleGeo = buildNoduleShellGeometry(6);
  const noduleMesh = new THREE.Mesh(noduleGeo, materials.limestone);
  noduleMesh.name = 'StratifiedLimestoneNodule';
  noduleMesh.castShadow = true;
  noduleMesh.receiveShadow = true;
  stage.specimenGroup.add(noduleMesh);

  // 4. Articulated Brass & Dark-Iron Pneumatic Air-Abrasion Micro-Nozzle
  const nozzleAssembly = buildNozzleAssembly({
    satinBrass: materials.satinBrass,
    darkIron: materials.darkIron,
    carbideTip: materials.carbideTip,
    hoseRubber: materials.hoseRubber,
    abrasiveJet: materials.abrasiveJet,
  });
  stage.scene.add(nozzleAssembly.group);
  const nozzleCtrl = new NozzleController();

  // 5. Zero-GC Instanced Rebounding Dolomite Grit & Billowing Calcite Dust Plumes
  const particles = new AbrasionParticleSystem(materials.gritChips, materials.dustPlume);
  stage.scene.add(particles.group);

  // Tactile Interaction State
  const canvas = stage.shell.canvas;
  let pointerClientX = 0;
  let pointerClientY = 0;
  let hasPointerPos = false;
  let isHoveringRock = false;
  let isBlasting = false;
  let isOrbiting = false;
  let hasPrevBlastDir = false;
  let clickBurstTimer = 0;
  let lastDownTimeMs = 0;
  let lastDownX = 0;
  let lastDownY = 0;

  function raycastFromClient(clientX: number, clientY: number): boolean {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
    _raycaster.setFromCamera(_ndc, stage.camera);
    return erosionField.raycast(_raycaster.ray.origin, _raycaster.ray.direction, _rayHit);
  }

  function triggerReEncrust(): void {
    erosionField.reset(erosionField.strataSeed + 1);
    materials.uStrataAxis.value.copy(erosionField.strataAxis);
    materials.uStrataSeed.value = erosionField.strataSeed;
    hasPrevBlastDir = false;
    clickBurstTimer = 0;
    particles.emitResetVeil();
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    pointerClientX = ev.clientX;
    pointerClientY = ev.clientY;
    hasPointerPos = true;

    const nowMs = performance.now();
    const isDoubleTap =
      nowMs - lastDownTimeMs < 340 && Math.hypot(ev.clientX - lastDownX, ev.clientY - lastDownY) < 10;
    lastDownTimeMs = nowMs;
    lastDownX = ev.clientX;
    lastDownY = ev.clientY;

    if (isDoubleTap) {
      isBlasting = false;
      isOrbiting = false;
      stage.controls.enabled = true;
      triggerReEncrust();
      return;
    }

    const hit = raycastFromClient(ev.clientX, ev.clientY);
    if (hit) {
      isHoveringRock = true;
      isBlasting = true;
      isOrbiting = false;
      hasPrevBlastDir = false;
      clickBurstTimer = 0.085;
      _lastHitPos.copy(_rayHit.position);
      _lastHitNormal.copy(_rayHit.normal);
      stage.controls.enabled = false;
      canvas.style.cursor = 'crosshair';
      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Synthetic Playwright events may not carry an active capture pointerId
      }
    } else {
      isHoveringRock = false;
      isBlasting = false;
      isOrbiting = true;
      stage.controls.enabled = true;
      canvas.style.cursor = 'grabbing';
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    pointerClientX = ev.clientX;
    pointerClientY = ev.clientY;
    hasPointerPos = true;

    if (isOrbiting) return;

    const hit = raycastFromClient(ev.clientX, ev.clientY);
    isHoveringRock = hit;
    if (hit) {
      _lastHitPos.copy(_rayHit.position);
      _lastHitNormal.copy(_rayHit.normal);
      canvas.style.cursor = 'crosshair';
    } else if (!isBlasting) {
      canvas.style.cursor = 'grab';
    }
  };

  const finishPointer = (ev: PointerEvent) => {
    if (isBlasting) {
      try {
        canvas.releasePointerCapture(ev.pointerId);
      } catch {
        // Ignore if not captured
      }
    }
    isBlasting = false;
    isOrbiting = false;
    hasPrevBlastDir = false;
    stage.controls.enabled = true;
    canvas.style.cursor = isHoveringRock ? 'crosshair' : 'grab';
  };

  const onPointerLeave = () => {
    if (!isBlasting && !isOrbiting) {
      isHoveringRock = false;
      canvas.style.cursor = 'default';
    }
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      triggerReEncrust();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', finishPointer);
  canvas.addEventListener('pointercancel', finishPointer);
  canvas.addEventListener('pointerleave', onPointerLeave);
  window.addEventListener('keydown', onKeyDown);

  // Read-only verification & telemetry hook
  window.__DAY19__ = {
    getState: () => ({
      exposedFraction: Number(erosionField.exposedFraction.toFixed(4)),
      totalCarvedVolume: Number(erosionField.totalCarvedVolume.toFixed(5)),
      isBlasting: isBlasting || clickBurstTimer > 0,
      activeParticles: particles.activeParticles,
      cameraPos: [
        Number(stage.camera.position.x.toFixed(4)),
        Number(stage.camera.position.y.toFixed(4)),
        Number(stage.camera.position.z.toFixed(4)),
      ],
    }),
  };

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', finishPointer);
    canvas.removeEventListener('pointercancel', finishPointer);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    window.removeEventListener('keydown', onKeyDown);
    stage.scene.environment = null;
    studioEnv.dispose();
    ammoniteBuilt.geometry.dispose();
    noduleGeo.dispose();
    nozzleAssembly.dispose();
    particles.dispose();
    erosionField.dispose();
    materials.dispose();
  });

  // Main animation tick (strictly zero per-frame allocations)
  stage.onTick(({ elapsedSec, dtSec, camera }) => {
    materials.uTime.value = elapsedSec;

    // Re-evaluate live raycast against carved surface while hovering or blasting
    let hitRockNow = false;
    if (hasPointerPos && !isOrbiting) {
      hitRockNow = raycastFromClient(pointerClientX, pointerClientY);
      isHoveringRock = hitRockNow;
      if (hitRockNow) {
        _lastHitPos.copy(_rayHit.position);
        _lastHitNormal.copy(_rayHit.normal);
      }
    }

    const activeBlastNow = (isBlasting || clickBurstTimer > 0) && hitRockNow && !isOrbiting;
    if (clickBurstTimer > 0) {
      clickBurstTimer = Math.max(0, clickBurstTimer - dtSec);
    }

    // 1. Update articulated brass pneumatic micro-nozzle pose & focused grit jet cone
    _poseInput.isHoveringRock = hitRockNow;
    _poseInput.isBlasting = activeBlastNow;
    _poseInput.isOrbiting = isOrbiting;
    _poseInput.hitPos.copy(_lastHitPos);
    _poseInput.hitNormal.copy(_lastHitNormal);
    _poseInput.cameraPos.copy(camera.position);
    _poseInput.cameraUp.copy(camera.up);
    _poseInput.elapsedSec = elapsedSec;

    nozzleCtrl.update(_poseInput, dtSec);
    nozzleAssembly.group.position.copy(nozzleCtrl.tipPos);
    nozzleAssembly.group.quaternion.copy(nozzleCtrl.orientation);

    materials.uJetAlpha.value = nozzleCtrl.jetActiveAlpha;
    if (nozzleCtrl.jetActiveAlpha > 0.015) {
      nozzleAssembly.jetMesh.visible = true;
      nozzleAssembly.jetMesh.scale.set(1, 1, Math.max(0.06, nozzleCtrl.jetLength));
    } else {
      nozzleAssembly.jetMesh.visible = false;
    }

    // 2. Active Air-Abrasion Carving with Smooth Stroke Sub-Stepping (zero dotted caterpillar pits!)
    if (activeBlastNow) {
      let avgFossilRatio = _rayHit.isFossil ? 1.0 : 0.0;

      if (hasPrevBlastDir) {
        const angStep = Math.acos(Math.max(-1, Math.min(1, _lastBlastDir.dot(_rayHit.direction))));
        const subSteps = Math.max(1, Math.min(6, Math.ceil(angStep / 0.032)));
        const subDt = dtSec / subSteps;
        let fossilSum = 0;

        for (let s = 1; s <= subSteps; s++) {
          const frac = s / subSteps;
          _interpDir
            .copy(_lastBlastDir)
            .lerp(_rayHit.direction, frac)
            .normalize();
          const res = erosionField.carveAt(_interpDir.x, _interpDir.y, _interpDir.z, subDt, 1.0);
          fossilSum += res.hitFossilRatio;
        }
        avgFossilRatio = fossilSum / subSteps;
      } else {
        const res = erosionField.carveAt(
          _rayHit.direction.x,
          _rayHit.direction.y,
          _rayHit.direction.z,
          dtSec,
          1.15
        );
        avgFossilRatio = res.hitFossilRatio;
      }

      _lastBlastDir.copy(_rayHit.direction);
      hasPrevBlastDir = true;

      materials.uBlastSpotWorld.value.set(
        _rayHit.position.x,
        _rayHit.position.y,
        _rayHit.position.z,
        nozzleCtrl.jetActiveAlpha
      );

      // Re-sample exact newly carved impact point so rebounding grit & chalk plumes erupt from live floor
      particles.emitBlast(
        _rayHit.position,
        _rayHit.normal,
        nozzleCtrl.aimDir,
        avgFossilRatio,
        dtSec,
        1.0
      );
    } else {
      hasPrevBlastDir = false;
      materials.uBlastSpotWorld.value.w *= Math.exp(-dtSec * 16.0);
    }

    // 3. Flush dirty heightfield & chalk decay to GPU RGBA16F DataTexture (0 bytes uploaded when idle)
    erosionField.stepAndUpload(dtSec);

    // 4. Advance bouncing dolomite grit chips & swirling pale calcite dust plumes
    particles.step(dtSec, camera.quaternion, erosionField);
  });
}

mountDay19Erosion().catch((err) => {
  console.error('[Day 19 - Erosion] Failed to initialize WebGPU specimen:', err);
});
