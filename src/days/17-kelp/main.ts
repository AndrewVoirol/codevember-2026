import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  buildBrassHardwareGeometry,
  buildBulbGlassGeometry,
  buildFilamentGeometry,
  buildKelpFrillsGeometry,
  buildKelpPanelsGeometry,
  buildLacingGeometry,
  buildWalnutRibsGeometry,
  createDynamicHempCord,
} from './lanternGeometry';
import { createKelpLanternMaterials } from './kelpMaterials';
import {
  applyLanternImpulse,
  createLanternPhysicsState,
  cycleFilamentMode,
  pluckParchment,
  setParchmentPoke,
  stepLanternPhysics,
} from './lanternPhysics';
import { DEFAULT_LANTERN_CONFIG } from './types';

declare global {
  interface Window {
    __KELP_LANTERN__?: {
      getState: () => Record<string, unknown>;
    };
  }
}

export async function bootDay17Kelp(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(17);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.22, 5.15],
    fov: 35,
  });

  stage.controls.target.set(0, 0.14, 0);
  stage.controls.minDistance = 2.2;
  stage.controls.maxDistance = 7.6;
  stage.controls.maxPolarAngle = Math.PI * 0.86;
  stage.controls.minPolarAngle = Math.PI * 0.14;
  stage.controls.update();

  const config = DEFAULT_LANTERN_CONFIG;
  const physics = createLanternPhysicsState();
  const mats = createKelpLanternMaterials();

  // Pivot group anchored at the ceiling above the frame so the pendant swings physically
  const ceilingPivot = new THREE.Group();
  ceilingPivot.name = 'KelpLanternCeilingPivot';
  ceilingPivot.position.set(0, config.pivotY, 0);
  stage.specimenGroup.add(ceilingPivot);

  // Lantern body suspended at -pendulumLength below the ceiling pivot
  const lanternGroup = new THREE.Group();
  lanternGroup.name = 'KelpLanternBody';
  lanternGroup.position.set(0, -config.pendulumLength, 0);
  ceilingPivot.add(lanternGroup);

  // 1. Stretched Cured Kelp-Leather & Alginate Parchment Panels
  const panelsGeom = buildKelpPanelsGeometry(config);
  const panelsMesh = new THREE.Mesh(panelsGeom, mats.parchmentMat);
  panelsMesh.name = 'KelpParchmentPanels';
  panelsMesh.castShadow = true;
  panelsMesh.receiveShadow = true;
  lanternGroup.add(panelsMesh);

  // 2. Dark Cured Ruffled Kelp Outer Frills
  const frillsGeom = buildKelpFrillsGeometry(config);
  const frillsMesh = new THREE.Mesh(frillsGeom, mats.frillMat);
  frillsMesh.name = 'KelpOuterFrills';
  frillsMesh.castShadow = true;
  frillsMesh.receiveShadow = true;
  lanternGroup.add(frillsMesh);

  // 3. Bent Dark-Walnut Meridian Ribs
  const ribsGeom = buildWalnutRibsGeometry(config);
  const ribsMesh = new THREE.Mesh(ribsGeom, mats.walnutMat);
  ribsMesh.name = 'WalnutMeridianRibs';
  ribsMesh.castShadow = true;
  ribsMesh.receiveShadow = true;
  lanternGroup.add(ribsMesh);

  // 4. Cross-Laced Whip-Stitching Down All 6 Bays
  const lacingGeom = buildLacingGeometry(config);
  const lacingMesh = new THREE.Mesh(lacingGeom, mats.lacingMat);
  lacingMesh.name = 'KelpSeamLacing';
  lacingMesh.castShadow = true;
  lanternGroup.add(lacingMesh);

  // 5. Aged Patinated Brass Socket, Knurled Collar, Flange & Rivets
  const brassGeom = buildBrassHardwareGeometry(config);
  const brassMesh = new THREE.Mesh(brassGeom, mats.brassMat);
  brassMesh.name = 'AgedBrassCanopy';
  brassMesh.castShadow = true;
  brassMesh.receiveShadow = true;
  lanternGroup.add(brassMesh);

  // 6. Internal Coupled-Pendulum Edison Bulb & Warm Filament Core
  const bulbPivot = new THREE.Group();
  bulbPivot.name = 'InternalBulbPivot';
  bulbPivot.position.set(0, config.halfHeight - 0.14, 0);
  lanternGroup.add(bulbPivot);

  const bulbHolder = new THREE.Group();
  bulbHolder.position.set(0, -0.44, 0);
  bulbPivot.add(bulbHolder);

  const filamentGeom = buildFilamentGeometry();
  const filamentMesh = new THREE.Mesh(filamentGeom, mats.filamentMat);
  filamentMesh.name = 'EdisonFilament';
  bulbHolder.add(filamentMesh);

  const bulbGlassGeom = buildBulbGlassGeometry();
  const bulbGlassMesh = new THREE.Mesh(bulbGlassGeom, mats.bulbGlassMat);
  bulbGlassMesh.name = 'EdisonBulbGlass';
  bulbHolder.add(bulbGlassMesh);

  // Real warm internal point light illuminating ribs, brass collar, lacing & open underside
  const internalLight = new THREE.PointLight(0xffb454, 4.2, 7.0, 1.75);
  internalLight.position.set(0, 0.02, 0);
  bulbHolder.add(internalLight);

  // 7. Dynamic Braided Hemp Pendant Cord (anchored in specimenGroup space)
  const hempCord = createDynamicHempCord(28, 8);
  const hempCordMesh = new THREE.Mesh(hempCord.geometry, mats.hempCordMat);
  hempCordMesh.name = 'BraidedHempCord';
  hempCordMesh.castShadow = true;
  stage.specimenGroup.add(hempCordMesh);

  // Raycast targets for tactile lantern interaction
  const interactiveMeshes: THREE.Object3D[] = [panelsMesh, frillsMesh, ribsMesh, brassMesh];

  // Pre-allocated scratch vectors (zero per-frame allocations)
  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  const localHit = new THREE.Vector3();
  const socketLocalAnchor = new THREE.Vector3(0, config.halfHeight + 0.54, 0);
  const socketWorldPos = new THREE.Vector3();
  const bulbWorldPos = new THREE.Vector3();
  const bulbLocalPos = new THREE.Vector3();

  let isDraggingLantern = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastClientX = 0;
  let lastClientY = 0;
  let dragTravelPx = 0;
  let hoveredLantern = false;

  const canvas = stage.shell.canvas;

  const updatePointerNdc = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    pointerNdc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    pointerNdc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  };

  const pickLanternHit = (clientX: number, clientY: number): THREE.Intersection | null => {
    updatePointerNdc(clientX, clientY);
    raycaster.setFromCamera(pointerNdc, stage.camera);
    const hits = raycaster.intersectObjects(interactiveMeshes, false);
    return hits.length > 0 ? hits[0] : null;
  };

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    const hit = pickLanternHit(e.clientX, e.clientY);
    if (!hit) return;

    isDraggingLantern = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    dragTravelPx = 0;
    stage.controls.enabled = false;
    canvas.style.cursor = 'grabbing';

    localHit.copy(hit.point);
    lanternGroup.worldToLocal(localHit);
    setParchmentPoke(physics, localHit.x, localHit.y, localHit.z, 0.135);
  };

  const onPointerMove = (e: PointerEvent) => {
    if (isDraggingLantern) {
      const dx = e.clientX - lastClientX;
      const dy = e.clientY - lastClientY;
      dragTravelPx += Math.hypot(dx, dy);
      lastClientX = e.clientX;
      lastClientY = e.clientY;

      const normDx = dx / Math.max(240, window.innerWidth * 0.45);
      const normDy = -dy / Math.max(240, window.innerHeight * 0.45);
      applyLanternImpulse(physics, normDx * 2.8, normDy * 1.9, normDx * 3.1);

      const hit = pickLanternHit(e.clientX, e.clientY);
      if (hit) {
        localHit.copy(hit.point);
        lanternGroup.worldToLocal(localHit);
        setParchmentPoke(physics, localHit.x, localHit.y, localHit.z, 0.14);
      }
      return;
    }

    // Gentle hover flex & cursor affordance over the cured kelp parchment
    const hit = pickLanternHit(e.clientX, e.clientY);
    if (hit) {
      hoveredLantern = true;
      canvas.style.cursor = 'grab';
      localHit.copy(hit.point);
      lanternGroup.worldToLocal(localHit);
      setParchmentPoke(physics, localHit.x, localHit.y, localHit.z, 0.062);

      // Subtle aerodynamic wake nudge on brisk pointer sweeps across the lantern
      if (stage.pointer.speed > 0.35) {
        applyLanternImpulse(
          physics,
          stage.pointer.velocity.x * 0.014,
          stage.pointer.velocity.y * 0.008,
          stage.pointer.velocity.x * 0.018
        );
      }
    } else {
      if (hoveredLantern) {
        hoveredLantern = false;
        canvas.style.cursor = 'default';
      }
      physics.pokeTargetDepth = 0;
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    if (!isDraggingLantern) return;
    isDraggingLantern = false;
    stage.controls.enabled = true;
    canvas.style.cursor = hoveredLantern ? 'grab' : 'default';
    physics.pokeTargetDepth = hoveredLantern ? 0.05 : 0;

    // Treat short press-and-release as a crisp parchment pluck + filament stoke toggle
    const totalDist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
    if (dragTravelPx < 9 && totalDist < 9) {
      pluckParchment(
        physics,
        physics.pokeLocal[0],
        physics.pokeLocal[1],
        physics.pokeLocal[2],
        1.15
      );
      cycleFilamentMode(physics);
      // Gentle physical tap recoil
      applyLanternImpulse(
        physics,
        -physics.pokeLocal[0] * 0.22,
        -0.08,
        physics.pokeLocal[0] * 0.35
      );
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerUp);
    panelsGeom.dispose();
    frillsGeom.dispose();
    ribsGeom.dispose();
    lacingGeom.dispose();
    brassGeom.dispose();
    filamentGeom.dispose();
    bulbGlassGeom.dispose();
    hempCord.geometry.dispose();
    mats.dispose();
  });

  // Expose read-only state diagnostics for verification & live inspection
  window.__KELP_LANTERN__ = {
    getState: () => ({
      swingX: Number(physics.swingX.toFixed(4)),
      swingZ: Number(physics.swingZ.toFixed(4)),
      yaw: Number(physics.yaw.toFixed(4)),
      filamentMode: physics.filamentMode,
      filamentGlow: Number(physics.filamentGlow.toFixed(3)),
      pokeDepth: Number(physics.pokeDepth.toFixed(4)),
      rippleAmp: Number(physics.rippleAmp.toFixed(4)),
    }),
  };

  // Initial cord positioning before first frame
  ceilingPivot.updateMatrixWorld(true);
  socketWorldPos.copy(socketLocalAnchor).applyMatrix4(lanternGroup.matrixWorld);
  hempCord.update(config.pivotY, socketWorldPos);

  stage.onTick(({ dtSec }) => {
    stepLanternPhysics(physics, dtSec, config);

    // Apply physical pendulum tilt & torsional yaw to ceiling pivot
    ceilingPivot.rotation.set(physics.swingX, physics.yaw, physics.swingZ);
    bulbPivot.rotation.set(physics.bulbSwingX, 0, physics.bulbSwingZ);

    // Refresh world matrices so cord and bulb-local-to-lantern coordinates stay exact
    ceilingPivot.updateMatrixWorld(true);

    // Update braided hemp cord cleanly between ceiling anchor and swinging brass ferrule
    socketWorldPos.copy(socketLocalAnchor).applyMatrix4(lanternGroup.matrixWorld);
    hempCord.update(config.pivotY, socketWorldPos);

    // Compute internal filament center in lantern-local space for Beer-Lambert backlighting
    bulbWorldPos.set(0, 0.04, 0).applyMatrix4(bulbHolder.matrixWorld);
    bulbLocalPos.copy(bulbWorldPos);
    lanternGroup.worldToLocal(bulbLocalPos);

    // Sync TSL uniforms without allocating
    mats.uniforms.uPokeLocal.value.set(
      physics.pokeLocal[0],
      physics.pokeLocal[1],
      physics.pokeLocal[2]
    );
    mats.uniforms.uPokeDepth.value = physics.pokeDepth;
    mats.uniforms.uRippleAmp.value = physics.rippleAmp;
    mats.uniforms.uRipplePhase.value = physics.ripplePhase;
    mats.uniforms.uBulbLocal.value.copy(bulbLocalPos);
    mats.uniforms.uGlowIntensity.value = physics.filamentGlow;

    internalLight.intensity = 4.2 * physics.filamentGlow;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay17Kelp();
}
