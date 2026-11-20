import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  buildBrassHardwareGeometry,
  buildBulbGlassGeometry,
  buildBulbHardwareGeometry,
  buildCeramicSocketGeometry,
  buildFilamentAuraGeometry,
  buildFilamentGeometry,
  buildKelpFrillsGeometry,
  buildKelpPanelsGeometry,
  buildLacingGeometry,
  buildWalnutRibsGeometry,
  createDynamicHempCord,
} from './lanternGeometry';
import {
  createKelpLanternMaterials,
  createKelpStudioEnvironment,
} from './kelpMaterials';
import {
  applyWorldLanternImpulse,
  createLanternPhysicsState,
  cycleFilamentMode,
  pluckParchment,
  setLanternGrabTorque,
  setParchmentPoke,
  stepLanternPhysics,
} from './lanternPhysics';
import { DEFAULT_LANTERN_CONFIG } from './types';

declare global {
  interface Window {
    __KELP_LANTERN__?: {
      getState: () => Record<string, unknown>;
      setCamera?: (pos: [number, number, number], target?: [number, number, number]) => void;
    };
  }
}

export async function bootDay17Kelp(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(17);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.24, 5.42],
    fov: 35,
  });

  stage.controls.target.set(0, 0.18, 0);
  stage.controls.minDistance = 1.9;
  stage.controls.maxDistance = 7.8;
  stage.controls.maxPolarAngle = Math.PI * 0.94;
  stage.controls.minPolarAngle = Math.PI * 0.08;
  stage.controls.update();

  // Studio softbox PMREM environment map for authentic aged-brass, walnut & blown-glass reflections
  const studioEnv = createKelpStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.05;

  const config = DEFAULT_LANTERN_CONFIG;
  const physics = createLanternPhysicsState();
  const mats = createKelpLanternMaterials();

  // 1. Ceiling pivot anchored above viewport: applies symmetric world-space pendulum tilt ONLY
  const ceilingPivot = new THREE.Group();
  ceilingPivot.name = 'KelpLanternCeilingPivot';
  ceilingPivot.position.set(0, config.pivotY, 0);
  stage.specimenGroup.add(ceilingPivot);

  // 2. Lantern body suspended below ceiling pivot: applies independent torsional hemp-cord yaw ONLY
  const lanternGroup = new THREE.Group();
  lanternGroup.name = 'KelpLanternBody';
  lanternGroup.position.set(0, -config.pendulumLength, 0);
  ceilingPivot.add(lanternGroup);

  // Stretched Cured Kelp-Leather & Alginate Parchment Panels
  const panelsGeom = buildKelpPanelsGeometry(config);
  const panelsMesh = new THREE.Mesh(panelsGeom, mats.parchmentMat);
  panelsMesh.name = 'KelpParchmentPanels';
  panelsMesh.castShadow = true;
  panelsMesh.receiveShadow = true;
  lanternGroup.add(panelsMesh);

  // Dark Cured Dried-Kombu Ruffled Outer Selvage Flanges
  const frillsGeom = buildKelpFrillsGeometry(config);
  const frillsMesh = new THREE.Mesh(frillsGeom, mats.frillMat);
  frillsMesh.name = 'KelpOuterFrills';
  frillsMesh.castShadow = true;
  frillsMesh.receiveShadow = true;
  lanternGroup.add(frillsMesh);

  // Steam-Bent Dark-Walnut Meridian Ribs (solid, outward-facing, capped)
  const ribsGeom = buildWalnutRibsGeometry(config);
  const ribsMesh = new THREE.Mesh(ribsGeom, mats.walnutMat);
  ribsMesh.name = 'WalnutMeridianRibs';
  ribsMesh.castShadow = true;
  ribsMesh.receiveShadow = true;
  lanternGroup.add(ribsMesh);

  // Hand-Laced Waxed-Hemp Whip-Stitching & Punched Eyelets (GPU-coupled to kelp skin flex)
  const lacingGeom = buildLacingGeometry(config);
  const lacingMesh = new THREE.Mesh(lacingGeom, mats.lacingMat);
  lacingMesh.name = 'KelpSeamLacing';
  lacingMesh.castShadow = true;
  lanternGroup.add(lacingMesh);

  // Heirloom Aged Patinated Brass Canopy, Knurled Ring, Clamping Flange & Cap Screws
  const brassGeom = buildBrassHardwareGeometry(config);
  const brassMesh = new THREE.Mesh(brassGeom, mats.brassMat);
  brassMesh.name = 'AgedBrassCanopy';
  brassMesh.castShadow = true;
  brassMesh.receiveShadow = true;
  lanternGroup.add(brassMesh);

  // Interior Glazed Dark-Ceramic E26 Socket Insulator Cup
  const ceramicGeom = buildCeramicSocketGeometry(config);
  const ceramicMesh = new THREE.Mesh(ceramicGeom, mats.ceramicSocketMat);
  ceramicMesh.name = 'CeramicSocketInsulator';
  lanternGroup.add(ceramicMesh);

  // Coupled-Pendulum Internal Vintage Tubular Edison Bulb & Tungsten Squirrel-Cage Filament
  const bulbPivot = new THREE.Group();
  bulbPivot.name = 'InternalBulbPivot';
  bulbPivot.position.set(0, config.halfHeight - 0.14, 0);
  lanternGroup.add(bulbPivot);

  const bulbHolder = new THREE.Group();
  // Position bulb dead-center in the glowing lantern belly (y ~ +0.12) matching Concept F
  // and cleanly visible when orbiting underneath to inspect the interior!
  bulbHolder.position.set(0, -0.70, 0);
  bulbPivot.add(bulbHolder);

  const bulbHardwareGeom = buildBulbHardwareGeometry();
  const bulbHardwareMesh = new THREE.Mesh(bulbHardwareGeom, mats.bulbHardwareMat);
  bulbHardwareMesh.name = 'EdisonBulbHardware';
  bulbHolder.add(bulbHardwareMesh);

  const filamentGeom = buildFilamentGeometry();
  const filamentMesh = new THREE.Mesh(filamentGeom, mats.filamentMat);
  filamentMesh.name = 'EdisonFilament';
  bulbHolder.add(filamentMesh);

  const filamentAuraGeom = buildFilamentAuraGeometry();
  const filamentAuraMesh = new THREE.Mesh(filamentAuraGeom, mats.filamentAuraMat);
  filamentAuraMesh.name = 'EdisonFilamentAura';
  filamentAuraMesh.renderOrder = 2;
  bulbHolder.add(filamentAuraMesh);

  const bulbGlassGeom = buildBulbGlassGeometry();
  const bulbGlassMesh = new THREE.Mesh(bulbGlassGeom, mats.bulbGlassMat);
  bulbGlassMesh.name = 'EdisonBulbGlass';
  bulbGlassMesh.renderOrder = 3;
  bulbHolder.add(bulbGlassMesh);

  // Warm internal tungsten point light illuminating ribs, ceramic socket, lacing & open interior
  const internalLight = new THREE.PointLight(0xffb454, 4.4, 7.6, 1.72);
  internalLight.position.set(0, 0.02, 0);
  bulbHolder.add(internalLight);

  // Dynamic 3-Ply Braided Hemp Pendant Cord (anchored in specimenGroup world space)
  const hempCord = createDynamicHempCord(32, 12);
  const hempCordMesh = new THREE.Mesh(hempCord.geometry, mats.hempCordMat);
  hempCordMesh.name = 'BraidedHempCord';
  hempCordMesh.castShadow = true;
  stage.specimenGroup.add(hempCordMesh);

  const interactiveMeshes: THREE.Object3D[] = [panelsMesh, frillsMesh, ribsMesh, brassMesh];

  // Pre-allocated scratch math objects (strictly zero per-frame allocations)
  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  const localHit = new THREE.Vector3();
  const socketLocalAnchor = new THREE.Vector3(0, config.halfHeight + 0.525, 0);
  const socketWorldPos = new THREE.Vector3();
  const bulbWorldPos = new THREE.Vector3();
  const bulbLocalPos = new THREE.Vector3();
  const swingAxis = new THREE.Vector3();
  const ceilingWorldPos = new THREE.Vector3(0, config.pivotY, 0);
  const lanternCenterWorld = new THREE.Vector3();
  const cordAxisWorld = new THREE.Vector3();
  const camForward = new THREE.Vector3();
  const camRight = new THREE.Vector3();
  const camUp = new THREE.Vector3();
  const dragPlane = new THREE.Plane();
  const grabLocalPt = new THREE.Vector3();
  const grabCurrentWorld = new THREE.Vector3();
  const grabTargetWorld = new THREE.Vector3();
  const pullVec = new THREE.Vector3();
  const leverFromCeiling = new THREE.Vector3();
  const radialArm = new THREE.Vector3();
  const torqueVec = new THREE.Vector3();
  const yawCross = new THREE.Vector3();
  const tapImpulseVec = new THREE.Vector3();
  const hoverVelWorld = new THREE.Vector3();

  let isDraggingLantern = false;
  let grabHitWasParchment = false;
  let grabBayIdx = -1;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastClientX = 0;
  let lastClientY = 0;
  let dragTravelPx = 0;
  let hoveredLantern = false;

  // Boot-settle cursor guard (prevents parked OS cursor from denting resting hero frame 0)
  const bootTimeMs = performance.now();
  let hoverUnlocked = false;
  let firstHoverX: number | null = null;
  let firstHoverY: number | null = null;

  const canvas = stage.shell.canvas;
  const bayMetaAttr = panelsGeom.getAttribute('aBayMeta') as THREE.BufferAttribute;

  const extractBayIndex = (hit: THREE.Intersection): number => {
    if (hit.object !== panelsMesh || !hit.face) return -1;
    return Math.round(bayMetaAttr.getZ(hit.face.a));
  };

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
    hoverUnlocked = true;
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

    // Anchor spring-tether grab point on the exact clicked spot + camera-facing drag plane
    grabLocalPt.copy(hit.point);
    lanternGroup.worldToLocal(grabLocalPt);
    grabTargetWorld.copy(hit.point);

    stage.camera.getWorldDirection(camForward);
    dragPlane.setFromNormalAndCoplanarPoint(camForward, hit.point);

    grabHitWasParchment = hit.object === panelsMesh;
    grabBayIdx = extractBayIndex(hit);
    if (grabHitWasParchment) {
      setParchmentPoke(physics, grabLocalPt.x, grabLocalPt.y, grabLocalPt.z, 0.125, grabBayIdx);
    } else {
      physics.pokeTargetDepth = 0;
    }
  };

  const onPointerMove = (e: PointerEvent) => {
    if (isDraggingLantern) {
      const dx = e.clientX - lastClientX;
      const dy = e.clientY - lastClientY;
      dragTravelPx += Math.hypot(dx, dy);
      lastClientX = e.clientX;
      lastClientY = e.clientY;

      updatePointerNdc(e.clientX, e.clientY);
      raycaster.setFromCamera(pointerNdc, stage.camera);
      raycaster.ray.intersectPlane(dragPlane, grabTargetWorld);

      if (grabHitWasParchment) {
        setParchmentPoke(physics, grabLocalPt.x, grabLocalPt.y, grabLocalPt.z, 0.13, grabBayIdx);
      }
      return;
    }

    // Boot-settle guard against parked OS cursors on initial load
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

    const hit = pickLanternHit(e.clientX, e.clientY);
    if (hit) {
      hoveredLantern = true;
      canvas.style.cursor = 'grab';

      // Flex kelp skin ONLY when hovering actual cured kelp parchment panels (not rigid brass/wood!)
      if (hit.object === panelsMesh) {
        const bayIdx = extractBayIndex(hit);
        localHit.copy(hit.point);
        lanternGroup.worldToLocal(localHit);
        setParchmentPoke(physics, localHit.x, localHit.y, localHit.z, 0.072, bayIdx);
      } else {
        physics.pokeTargetDepth = 0;
      }

      // Subtle camera-relative aerodynamic brush wake on brisk pointer sweeps across the lantern
      if (stage.pointer.speed > 0.35) {
        camRight.set(1, 0, 0).applyQuaternion(stage.camera.quaternion);
        camUp.set(0, 1, 0).applyQuaternion(stage.camera.quaternion);
        hoverVelWorld
          .copy(camRight)
          .multiplyScalar(stage.pointer.velocity.x)
          .addScaledVector(camUp, stage.pointer.velocity.y);
        lanternGroup.getWorldPosition(lanternCenterWorld);
        radialArm.subVectors(hit.point, lanternCenterWorld);
        yawCross.crossVectors(radialArm, hoverVelWorld);
        applyWorldLanternImpulse(
          physics,
          hoverVelWorld.x * 0.015,
          hoverVelWorld.z * 0.015,
          yawCross.y * 0.022
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
    setLanternGrabTorque(physics, 0, 0, 0, false);
    stage.controls.enabled = true;
    canvas.style.cursor = hoveredLantern ? 'grab' : 'default';
    physics.pokeTargetDepth = hoveredLantern && grabHitWasParchment ? 0.055 : 0;

    // Short press-and-release: crisp parchment drumhead pluck + filament warmth toggle + camera-aligned tap recoil
    const totalDist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
    if (dragTravelPx < 9 && totalDist < 9) {
      if (grabHitWasParchment) {
        pluckParchment(
          physics,
          grabLocalPt.x,
          grabLocalPt.y,
          grabLocalPt.z,
          1.2,
          grabBayIdx
        );
      }
      cycleFilamentMode(physics);

      // Push gently along the camera's forward view ray so tapping always nudges away from the viewer's hand
      stage.camera.getWorldDirection(camForward);
      grabCurrentWorld.copy(grabLocalPt).applyMatrix4(lanternGroup.matrixWorld);
      lanternGroup.getWorldPosition(lanternCenterWorld);
      radialArm.subVectors(grabCurrentWorld, lanternCenterWorld);
      tapImpulseVec.copy(camForward).multiplyScalar(0.32);
      yawCross.crossVectors(radialArm, tapImpulseVec);
      applyWorldLanternImpulse(physics, tapImpulseVec.x, tapImpulseVec.z, yawCross.y * 0.9);
    }
  };

  const onWheel = () => {
    hoverUnlocked = true;
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('wheel', onWheel, { passive: true });
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('wheel', onWheel);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerUp);
    stage.scene.environment = null;
    studioEnv.dispose();
    panelsGeom.dispose();
    frillsGeom.dispose();
    ribsGeom.dispose();
    lacingGeom.dispose();
    brassGeom.dispose();
    ceramicGeom.dispose();
    bulbHardwareGeom.dispose();
    filamentGeom.dispose();
    filamentAuraGeom.dispose();
    bulbGlassGeom.dispose();
    hempCord.geometry.dispose();
    mats.dispose();
  });

  window.__KELP_LANTERN__ = {
    getState: () => ({
      swingX: Number(physics.swingX.toFixed(4)),
      swingZ: Number(physics.swingZ.toFixed(4)),
      yaw: Number(physics.yaw.toFixed(4)),
      filamentMode: physics.filamentMode,
      filamentGlow: Number(physics.filamentGlow.toFixed(3)),
      pokeBay: physics.pokeBay,
      pokeDepth: Number(physics.pokeDepth.toFixed(4)),
      rippleAmp: Number(physics.rippleAmp.toFixed(4)),
      grabActive: physics.grabActive,
    }),
    setCamera: (pos, target) => {
      stage.camera.position.set(pos[0], pos[1], pos[2]);
      if (target) stage.controls.target.set(target[0], target[1], target[2]);
      stage.controls.update();
    },
  };

  // Sync initial world matrices and taut braided hemp cord before first render
  ceilingPivot.updateMatrixWorld(true);
  socketWorldPos.copy(socketLocalAnchor).applyMatrix4(lanternGroup.matrixWorld);
  hempCord.update(config.pivotY, socketWorldPos, physics.yaw);

  stage.onTick(({ dtSec }) => {
    // Update continuous camera-relative spring-tether grab torques while holding the lantern
    if (isDraggingLantern) {
      ceilingPivot.updateMatrixWorld(true);
      grabCurrentWorld.copy(grabLocalPt).applyMatrix4(lanternGroup.matrixWorld);
      lanternGroup.getWorldPosition(lanternCenterWorld);

      pullVec.subVectors(grabTargetWorld, grabCurrentWorld);
      const pullLen = pullVec.length();
      if (pullLen > 1.4) {
        pullVec.multiplyScalar(1.4 / pullLen);
      }

      // True 3D pendulum torque around ceiling anchor: tau = r_ceiling x (k * pullVec)
      leverFromCeiling.subVectors(grabCurrentWorld, ceilingWorldPos);
      torqueVec.crossVectors(leverFromCeiling, pullVec).multiplyScalar(5.2);

      // True 3D torsional yaw torque around hanging cord axis: tau_y = (r_radial x pullVec) . u_cord
      radialArm.subVectors(grabCurrentWorld, lanternCenterWorld);
      cordAxisWorld.set(0, 1, 0).applyQuaternion(ceilingPivot.quaternion);
      yawCross.crossVectors(radialArm, pullVec);
      const yawTorque = yawCross.dot(cordAxisWorld) * 10.5;

      setLanternGrabTorque(physics, torqueVec.x, torqueVec.z, yawTorque, true);
    }

    stepLanternPhysics(physics, dtSec, config);

    // Apply isotropic world-space pendulum tilt to ceilingPivot (zero Euler X/Z/Y order gimbal fighting!)
    const swingAngle = Math.hypot(physics.swingX, physics.swingZ);
    if (swingAngle > 1e-6) {
      swingAxis.set(physics.swingX / swingAngle, 0, physics.swingZ / swingAngle);
      ceilingPivot.quaternion.setFromAxisAngle(swingAxis, swingAngle);
    } else {
      ceilingPivot.quaternion.identity();
    }

    // Apply independent torsional hemp-cord spin around local hanging axis
    lanternGroup.rotation.set(0, physics.yaw, 0);
    bulbPivot.rotation.set(physics.bulbSwingX, 0, physics.bulbSwingZ);

    // Refresh world matrices so cord and bulb-local-to-lantern coordinates stay exact
    ceilingPivot.updateMatrixWorld(true);

    // Update 3-ply braided hemp cord cleanly between ceiling anchor and swinging brass ferrule
    socketWorldPos.copy(socketLocalAnchor).applyMatrix4(lanternGroup.matrixWorld);
    hempCord.update(config.pivotY, socketWorldPos, physics.yaw);

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
    mats.uniforms.uPokeBay.value = physics.pokeBay;
    mats.uniforms.uPokeDepth.value = physics.pokeDepth;
    mats.uniforms.uRippleAmp.value = physics.rippleAmp;
    mats.uniforms.uRipplePhase.value = physics.ripplePhase;
    mats.uniforms.uBulbLocal.value.copy(bulbLocalPos);
    mats.uniforms.uGlowIntensity.value = physics.filamentGlow;

    internalLight.intensity = 4.4 * physics.filamentGlow;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay17Kelp();
}
