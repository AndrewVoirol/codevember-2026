import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  buildDowelGeometry,
  buildPebbleEggGeometry,
  buildPlinthBrassInnerBezelGeometry,
  buildPlinthBrassOuterLipGeometry,
  buildPlinthLimestoneGeometry,
} from './nestGeometry';
import { createNestMaterials, createNestStudioEnvironment } from './nestMaterials';
import { ReciprocalNestSim, type Day24TelemetryState } from './nestSimulation';
import { generateReciprocalDomeLayout } from './reciprocalDome';

declare global {
  interface Window {
    __DAY24__?: {
      getState: () => Day24TelemetryState;
    };
  }
}

interface TierInstancedGroup {
  readonly tier: number;
  readonly startRodId: number;
  readonly count: number;
  readonly mesh: THREE.InstancedMesh;
  readonly geometry: THREE.BufferGeometry;
  readonly metaArray: Float32Array;
  readonly metaAttr: THREE.InstancedBufferAttribute;
}

export async function bootDay24Nest(): Promise<SpecimenStage> {
  await RAPIER.init();

  const dayEntry = getDayByNumber(24);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 1.82, 3.42],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  // Hero museum 3/4 framing centered on the suspended ring + woven timber dome + ovoid celadon eggs
  stage.controls.target.set(0, 0.28, 0);
  stage.controls.minDistance = 0.85;
  stage.controls.maxDistance = 7.8;
  stage.controls.minPolarAngle = 0.08;
  stage.controls.maxPolarAngle = Math.PI * 0.94;
  stage.controls.update();

  // Studio softbox PMREM environment map for dark-brass, oiled walnut & satin celadon reflections
  const studioEnv = createNestStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 0.95;

  // Subtle warm studio uplift beneath the hollow plinth ring so underside weaving reads crisply
  const undersideBounce = new THREE.PointLight(0xeae3d2, 1.05, 6.5, 1.6);
  undersideBounce.position.set(0.3, -1.65, 0.4);
  stage.specimenGroup.add(undersideBounce);

  // Build exact analytical zero-overlap 56-rod reciprocal dome + 3 ovoid Celadon river-pebble eggs
  const layout = generateReciprocalDomeLayout();
  const sim = new ReciprocalNestSim(RAPIER, layout);
  const mats = createNestMaterials(layout.eggs);

  const nestGroup = new THREE.Group();
  nestGroup.name = 'ReciprocalTimberNestSpecimen';
  stage.specimenGroup.add(nestGroup);

  // ---------------------------------------------------------------------------
  // 1. Suspended Honed-Limestone & Dark-Brass Circular Plinth Ring (Hollow Center)
  // ---------------------------------------------------------------------------
  const plinthLimestoneGeom = buildPlinthLimestoneGeometry(layout.plinth);
  const plinthLimestoneMesh = new THREE.Mesh(plinthLimestoneGeom, mats.limestoneMat);
  plinthLimestoneMesh.name = 'PlinthLimestoneRing';
  plinthLimestoneMesh.castShadow = true;
  plinthLimestoneMesh.receiveShadow = true;
  nestGroup.add(plinthLimestoneMesh);

  const plinthBrassLipGeom = buildPlinthBrassOuterLipGeometry(layout.plinth);
  const plinthBrassLipMesh = new THREE.Mesh(plinthBrassLipGeom, mats.brassMat);
  plinthBrassLipMesh.name = 'PlinthDarkBrassOuterLip';
  plinthBrassLipMesh.castShadow = true;
  plinthBrassLipMesh.receiveShadow = true;
  nestGroup.add(plinthBrassLipMesh);

  const plinthBrassInnerGeom = buildPlinthBrassInnerBezelGeometry(layout.plinth);
  const plinthBrassInnerMesh = new THREE.Mesh(plinthBrassInnerGeom, mats.brassMat);
  plinthBrassInnerMesh.name = 'PlinthDarkBrassInnerBezel';
  plinthBrassInnerMesh.castShadow = true;
  plinthBrassInnerMesh.receiveShadow = true;
  nestGroup.add(plinthBrassInnerMesh);

  // ---------------------------------------------------------------------------
  // 2. Five Per-Tier InstancedMesh Groups (56 Chamfered Flat-Ended Timber Dowels)
  // ---------------------------------------------------------------------------
  const tierGroups: TierInstancedGroup[] = [];
  const tierMeshes: THREE.InstancedMesh[] = [];
  const meshToTierGroup = new Map<THREE.Object3D, TierInstancedGroup>();

  let rodCursor = 0;
  for (let t = 0; t < layout.tiers.length; t++) {
    const tierMeta = layout.tiers[t];
    const firstRod = layout.rods[rodCursor];
    const geom = buildDowelGeometry(
      tierMeta.halfHeight,
      tierMeta.radius,
      firstRod.notchParams,
      24
    );

    const metaArray = new Float32Array(tierMeta.count * 4);
    for (let i = 0; i < tierMeta.count; i++) {
      const r = layout.rods[rodCursor + i];
      metaArray[i * 4 + 0] = r.woodSpecies;
      metaArray[i * 4 + 1] = r.grainSeed;
      metaArray[i * 4 + 2] = r.id;
      metaArray[i * 4 + 3] = 0.0;
    }
    const metaAttr = new THREE.InstancedBufferAttribute(metaArray, 4);
    metaAttr.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('aRodMeta', metaAttr);

    const instMesh = new THREE.InstancedMesh(geom, mats.timberMat, tierMeta.count);
    instMesh.name = `ReciprocalTimberTier${t}`;
    instMesh.castShadow = true;
    instMesh.receiveShadow = true;
    instMesh.frustumCulled = false;
    nestGroup.add(instMesh);

    const group: TierInstancedGroup = {
      tier: t,
      startRodId: rodCursor,
      count: tierMeta.count,
      mesh: instMesh,
      geometry: geom,
      metaArray,
      metaAttr,
    };
    tierGroups.push(group);
    tierMeshes.push(instMesh);
    meshToTierGroup.set(instMesh, group);
    rodCursor += tierMeta.count;
  }

  // ---------------------------------------------------------------------------
  // 3. Three Speckled Celadon & Bone-Calcite River-Pebble Ovoid "Eggs"
  // ---------------------------------------------------------------------------
  const eggGeoms: THREE.BufferGeometry[] = [];
  const eggMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < layout.eggs.length; i++) {
    const eggSpec = layout.eggs[i];
    const geom = buildPebbleEggGeometry(eggSpec.radius, eggSpec.visualScale, eggSpec.seed, 4);
    const mesh = new THREE.Mesh(geom, mats.eggMats[i]);
    mesh.name = `CeladonPebbleEgg${i}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    nestGroup.add(mesh);
    eggGeoms.push(geom);
    eggMeshes.push(mesh);
  }

  // ---------------------------------------------------------------------------
  // 4. Pre-Allocated Scratch Objects (Strictly Zero Per-Frame Allocations)
  // ---------------------------------------------------------------------------
  const scratchPos = new THREE.Vector3();
  const scratchQuat = new THREE.Quaternion();
  const scratchQuatInv = new THREE.Quaternion();
  const scratchScale = new THREE.Vector3(1, 1, 1);
  const scratchZeroScale = new THREE.Vector3(0, 0, 0);
  const scratchMat4 = new THREE.Matrix4();

  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  const camForward = new THREE.Vector3();
  const camRight = new THREE.Vector3();
  const camUp = new THREE.Vector3();
  const dragPlane = new THREE.Plane();
  const dragTargetVec = new THREE.Vector3();
  const localHitVec = new THREE.Vector3();
  const pokeDirVec = new THREE.Vector3();
  const sweepDirVec = new THREE.Vector3();
  const hitWorldVec = new THREE.Vector3();

  const rodFlash = new Float32Array(layout.rods.length);

  // Sync all 56 timber instances + 3 ovoid egg meshes directly from Rapier world transforms
  const syncMeshesFromPhysics = () => {
    for (let t = 0; t < tierGroups.length; t++) {
      const grp = tierGroups[t];
      let metaDirty = false;
      for (let i = 0; i < grp.count; i++) {
        const rodId = grp.startRodId + i;
        const rec = sim.rods[rodId];
        if (rec.removed) {
          scratchPos.set(0, -100, 0);
          scratchQuat.identity();
          scratchMat4.compose(scratchPos, scratchQuat, scratchZeroScale);
        } else {
          const tr = rec.body.translation();
          const rq = rec.body.rotation();
          scratchPos.set(tr.x, tr.y, tr.z);
          scratchQuat.set(rq.x, rq.y, rq.z, rq.w);
          scratchMat4.compose(scratchPos, scratchQuat, scratchScale);
        }
        grp.mesh.setMatrixAt(i, scratchMat4);

        const prevFlash = grp.metaArray[i * 4 + 3];
        const nextFlash = rodFlash[rodId];
        if (Math.abs(prevFlash - nextFlash) > 1e-4) {
          grp.metaArray[i * 4 + 3] = nextFlash;
          metaDirty = true;
        }
      }
      grp.mesh.instanceMatrix.needsUpdate = true;
      if (metaDirty) {
        grp.metaAttr.needsUpdate = true;
      }
    }

    for (let i = 0; i < sim.eggs.length; i++) {
      const eggRec = sim.eggs[i];
      const tr = eggRec.body.translation();
      const rq = eggRec.body.rotation();
      const mesh = eggMeshes[i];
      mesh.position.set(tr.x, tr.y, tr.z);
      mesh.quaternion.set(rq.x, rq.y, rq.z, rq.w);
    }
  };

  syncMeshesFromPhysics();

  // ---------------------------------------------------------------------------
  // 5. Tactile Jenga Interaction: Warm Subtle Hover Rim, 3D Grab-Spring Pull,
  //    Quick Tap-Test, 360° Empty-Space Orbit, and Double-Click / Space Reset
  // ---------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  let hoveredRodId = -1;
  let hoverLiftCurrent = 0;
  let grabbingRodId = -1;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastClientX = 0;
  let lastClientY = 0;
  let dragTravelPx = 0;
  let lastSweepNudgeMs = 0;

  // Boot-settle cursor guard (ignores stationary parked OS cursor on initial load)
  const bootTimeMs = performance.now();
  let hoverUnlocked = false;
  let firstHoverX: number | null = null;
  let firstHoverY: number | null = null;

  const updatePointerNdc = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    pointerNdc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    pointerNdc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  };

  const pickRodHit = (
    clientX: number,
    clientY: number
  ): { rodId: number; point: THREE.Vector3 } | null => {
    updatePointerNdc(clientX, clientY);
    raycaster.setFromCamera(pointerNdc, stage.camera);
    // Refresh instanced bounding spheres right before picking so displaced/pulled rods remain 100% pickable
    for (let t = 0; t < tierMeshes.length; t++) {
      tierMeshes[t].computeBoundingSphere();
    }
    const hits = raycaster.intersectObjects(tierMeshes, false);
    for (let i = 0; i < hits.length; i++) {
      const h = hits[i];
      if (h.instanceId === undefined) continue;
      const grp = meshToTierGroup.get(h.object);
      if (!grp) continue;
      const rodId = grp.startRodId + h.instanceId;
      const rec = sim.rods[rodId];
      if (!rec || rec.removed) continue;
      hitWorldVec.copy(h.point);
      return { rodId, point: hitWorldVec };
    }
    return null;
  };

  const resetSpecimen = () => {
    grabbingRodId = -1;
    hoveredRodId = -1;
    hoverLiftCurrent = 0;
    rodFlash.fill(0);
    mats.uHoveredRodId.value = -1;
    mats.uGrabbedRodId.value = -1;
    mats.uHoverLift.value = 0;
    stage.controls.enabled = true;
    canvas.style.cursor = 'default';
    sim.reset();
    syncMeshesFromPhysics();
  };

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    hoverUnlocked = true;
    const hit = pickRodHit(e.clientX, e.clientY);
    if (!hit) return;

    const rec = sim.rods[hit.rodId];
    if (!rec || rec.removed) return;

    grabbingRodId = hit.rodId;
    hoveredRodId = hit.rodId;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    dragTravelPx = 0;

    stage.controls.enabled = false;
    canvas.style.cursor = 'grabbing';
    mats.uGrabbedRodId.value = hit.rodId;
    mats.uHoveredRodId.value = hit.rodId;

    // Transform hit point into the rod rigid body's local frame
    const tr = rec.body.translation();
    const rq = rec.body.rotation();
    scratchQuatInv.set(rq.x, rq.y, rq.z, rq.w).invert();
    localHitVec.set(hit.point.x - tr.x, hit.point.y - tr.y, hit.point.z - tr.z);
    localHitVec.applyQuaternion(scratchQuatInv);

    // Anchor camera-facing 3D drag plane through the exact picked surface point
    stage.camera.getWorldDirection(camForward);
    dragPlane.setFromNormalAndCoplanarPoint(camForward, hit.point);
    dragTargetVec.copy(hit.point);

    sim.startGrab(
      hit.rodId,
      [localHitVec.x, localHitVec.y, localHitVec.z],
      [dragTargetVec.x, dragTargetVec.y, dragTargetVec.z]
    );
  };

  const onPointerMove = (e: PointerEvent) => {
    if (grabbingRodId !== -1) {
      const dx = e.clientX - lastClientX;
      const dy = e.clientY - lastClientY;
      dragTravelPx += Math.hypot(dx, dy);
      lastClientX = e.clientX;
      lastClientY = e.clientY;

      updatePointerNdc(e.clientX, e.clientY);
      raycaster.setFromCamera(pointerNdc, stage.camera);
      if (raycaster.ray.intersectPlane(dragPlane, dragTargetVec)) {
        sim.updateGrabTarget([dragTargetVec.x, dragTargetVec.y, dragTargetVec.z]);
      }
      return;
    }

    // Boot-settle guard against stationary OS cursors on initial page load
    if (!hoverUnlocked) {
      if (performance.now() - bootTimeMs < 350) return;
      if (firstHoverX === null || firstHoverY === null) {
        firstHoverX = e.clientX;
        firstHoverY = e.clientY;
        return;
      }
      if (Math.hypot(e.clientX - firstHoverX, e.clientY - firstHoverY) < 8) return;
      hoverUnlocked = true;
    }

    const hit = pickRodHit(e.clientX, e.clientY);
    if (hit) {
      hoveredRodId = hit.rodId;
      mats.uHoveredRodId.value = hit.rodId;
      canvas.style.cursor = 'grab';

      // Faint tactile micro-shift brush on brisk pointer sweeps across timber dowels
      const nowMs = performance.now();
      if (stage.pointer.speed > 0.45 && nowMs - lastSweepNudgeMs > 90) {
        lastSweepNudgeMs = nowMs;
        camRight.set(1, 0, 0).applyQuaternion(stage.camera.quaternion);
        camUp.set(0, 1, 0).applyQuaternion(stage.camera.quaternion);
        sweepDirVec
          .copy(camRight)
          .multiplyScalar(stage.pointer.velocity.x)
          .addScaledVector(camUp, stage.pointer.velocity.y);
        if (sweepDirVec.lengthSq() > 1e-5) {
          sweepDirVec.normalize();
          sim.pokeRod(
            hit.rodId,
            [hit.point.x, hit.point.y, hit.point.z],
            [sweepDirVec.x, sweepDirVec.y, sweepDirVec.z],
            0.00065
          );
          rodFlash[hit.rodId] = Math.max(rodFlash[hit.rodId], 0.16);
        }
      }
    } else {
      if (hoveredRodId !== -1) {
        hoveredRodId = -1;
        canvas.style.cursor = 'default';
      }
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    if (grabbingRodId === -1) return;
    const releasedRodId = grabbingRodId;
    grabbingRodId = -1;
    sim.endGrab();
    mats.uGrabbedRodId.value = -1;
    stage.controls.enabled = true;
    canvas.style.cursor = hoveredRodId !== -1 ? 'grab' : 'default';

    // Quick click on a dowel: crisp lateral tap/poke impulse to test whether it's loose or load-bearing
    const totalDist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
    if (dragTravelPx < 8 && totalDist < 8) {
      stage.camera.getWorldDirection(camForward);
      camRight.set(1, 0, 0).applyQuaternion(stage.camera.quaternion);
      pokeDirVec.copy(camForward).multiplyScalar(0.82).addScaledVector(camRight, 0.38).normalize();
      sim.pokeRod(
        releasedRodId,
        [dragTargetVec.x, dragTargetVec.y, dragTargetVec.z],
        [pokeDirVec.x, pokeDirVec.y, pokeDirVec.z],
        0.0028
      );
      rodFlash[releasedRodId] = 0.60;
    }
  };

  const onDoubleClick = (e: MouseEvent) => {
    // Double-clicking empty space cleanly re-weaves and resets the intact reciprocal dome
    const hit = pickRodHit(e.clientX, e.clientY);
    if (!hit) {
      resetSpecimen();
    }
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.code === 'Space' && !e.repeat) {
      e.preventDefault();
      resetSpecimen();
    }
  };

  const onWheel = () => {
    hoverUnlocked = true;
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('dblclick', onDoubleClick);
  canvas.addEventListener('wheel', onWheel, { passive: true });
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('dblclick', onDoubleClick);
    canvas.removeEventListener('wheel', onWheel);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerUp);
    window.removeEventListener('keydown', onKeyDown);

    stage.scene.environment = null;
    studioEnv.dispose();
    plinthLimestoneGeom.dispose();
    plinthBrassLipGeom.dispose();
    plinthBrassInnerGeom.dispose();
    for (const grp of tierGroups) grp.geometry.dispose();
    for (const g of eggGeoms) g.dispose();
    mats.dispose();
    sim.dispose();
  });

  // Expose canonical read-only Day 24 telemetry hook
  window.__DAY24__ = {
    getState: () => sim.getState(),
  };

  // Main render & Rapier 0.21 physics loop (strictly zero per-frame allocations)
  stage.onTick(({ dtSec }) => {
    sim.step(dtSec);

    // Smooth hover rim fade & tactile poke flash decay (strict zero tint when unhovered)
    const targetHover = hoveredRodId !== -1 ? 1.0 : 0.0;
    const hoverLerp = Math.min(1, dtSec * 14.0);
    hoverLiftCurrent += (targetHover - hoverLiftCurrent) * hoverLerp;
    if (hoveredRodId === -1 && hoverLiftCurrent < 0.002) {
      hoverLiftCurrent = 0;
      mats.uHoveredRodId.value = -1;
    }
    mats.uHoverLift.value = hoverLiftCurrent;

    const flashDecay = Math.exp(-dtSec * 6.5);
    for (let i = 0; i < rodFlash.length; i++) {
      if (rodFlash[i] > 0.002) {
        rodFlash[i] *= flashDecay;
      } else if (rodFlash[i] !== 0) {
        rodFlash[i] = 0;
      }
    }

    syncMeshesFromPhysics();
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay24Nest();
}
