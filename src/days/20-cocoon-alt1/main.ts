import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  buildEmberCoreGeometry,
  buildEmberQuadGeometry,
  buildOrigamiButterflyWingGeometry,
  buildOvoidBambooFrameGeometry,
  buildOvoidPaperDomeGeometry,
  buildSilkWrapBandsGeometry,
  buildUnspoolingSilkLoopsGeometry,
} from './lanternGeometry';
import {
  createLanternStudioEnvironment,
  createSilkLanternMaterials,
} from './lanternMaterials';
import { SilkLanternSimulation } from './lanternSimulation';
import {
  DEFAULT_SILK_CONFIG,
  type ChrysalisStage,
  type Day20Alt1Telemetry,
} from './types';

declare global {
  interface Window {
    __COCOON_ALT1__?: {
      getState: () => Day20Alt1Telemetry;
      setPhase: (stage: ChrysalisStage) => void;
      resetToBound: () => void;
      stokeBurst: (worldX?: number, worldY?: number, worldZ?: number) => number;
      setOrbit: (azimuth: number, pitch: number, distance: number, targetY?: number) => void;
    };
  }
}

const PENDULUM_PIVOT_Y = 2.0;
const BUTTERFLY_BASE_SCALE = 0.115;

// Pre-allocated scratch math objects (strictly zero per-frame heap allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _planeHit = new THREE.Vector3();
const _prevPlaneHit = new THREE.Vector3();
const _pointerVel = new THREE.Vector3();
const _dragPlane = new THREE.Plane();
const _camDir = new THREE.Vector3();
const _flameWorldPos = new THREE.Vector3();
const _camOffset = new THREE.Vector3();
const _tiltAxis = new THREE.Vector3();
const _dummyPos = new THREE.Vector3();
const _dummyScale = new THREE.Vector3();
const _dummyMatrix = new THREE.Matrix4();

export async function bootDay20CocoonAlt1(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(20);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.04, 4.55],
    fov: 35,
    enableOrbit: true,
    enableShadows: false,
  });

  stage.camera.near = 0.05;
  stage.camera.updateProjectionMatrix();

  stage.controls.target.set(0, 0.02, 0);
  stage.controls.minDistance = 1.15;
  stage.controls.maxDistance = 8.5;
  stage.controls.minPolarAngle = 0.22;
  stage.controls.maxPolarAngle = Math.PI - 0.14;
  stage.controls.zoomSpeed = 0.85;
  stage.controls.enableDamping = true;
  stage.controls.dampingFactor = 0.065;
  stage.controls.update();

  // 7-softbox studio PMREM environment for oiled bamboo, raw silk, and kozo paper
  const studioEnv = createLanternStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 0.86;

  // Calibrate studio 3-point softbox lighting to match Concept A Hero & Macro renders
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(3.2, 4.4, 3.8);
        child.intensity = 1.65;
      } else if (dirIdx === 1) {
        child.position.set(-3.6, 0.8, 2.6);
        child.intensity = 0.68;
      } else if (dirIdx === 2) {
        child.position.set(-1.0, 3.8, -3.6);
        child.intensity = 1.48;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.56;
    }
  }

  const specimenRoot = new THREE.Group();
  specimenRoot.name = 'SilkLanternChrysalisRoot';
  stage.specimenGroup.add(specimenRoot);

  const sim = new SilkLanternSimulation(DEFAULT_SILK_CONFIG);
  const mats = createSilkLanternMaterials();

  // -------------------------------------------------------------------------
  // 1. Suspended Lantern Hierarchy (Decoupled Pendulum Pivot + Torsional Yaw)
  // -------------------------------------------------------------------------
  const pendulumPivot = new THREE.Group();
  pendulumPivot.name = 'SuspensionCordPivot';
  pendulumPivot.position.set(0, PENDULUM_PIVOT_Y, 0);
  specimenRoot.add(pendulumPivot);

  const lanternYawGroup = new THREE.Group();
  lanternYawGroup.name = 'OvoidLanternYawGroup';
  lanternYawGroup.position.set(0, -PENDULUM_PIVOT_Y, 0);
  pendulumPivot.add(lanternYawGroup);

  const paperDomeGeo = buildOvoidPaperDomeGeometry(12, 28);
  const paperDomeMesh = new THREE.Mesh(paperDomeGeo, mats.ricePaper);
  paperDomeMesh.name = 'OvoidKozoPaperCanopy';
  paperDomeMesh.renderOrder = 2;
  lanternYawGroup.add(paperDomeMesh);

  const bambooFrameGeo = buildOvoidBambooFrameGeometry(12);
  const bambooFrameMesh = new THREE.Mesh(bambooFrameGeo, mats.bambooFrame);
  bambooFrameMesh.name = 'OvoidBambooFrameAndCord';
  bambooFrameMesh.renderOrder = 3;
  lanternYawGroup.add(bambooFrameMesh);

  const silkWrapGeo = buildSilkWrapBandsGeometry(140);
  const silkWrapMesh = new THREE.Mesh(silkWrapGeo, mats.silkWrap);
  silkWrapMesh.name = 'RawSilkWrapBands';
  silkWrapMesh.renderOrder = 4;
  lanternYawGroup.add(silkWrapMesh);

  const silkLoopsGeo = buildUnspoolingSilkLoopsGeometry(sim.config.silkLoopCount);
  const silkLoopsMesh = new THREE.Mesh(silkLoopsGeo, mats.silkLoops);
  silkLoopsMesh.name = 'UnspoolingSilkLoops';
  silkLoopsMesh.renderOrder = 5;
  lanternYawGroup.add(silkLoopsMesh);

  const emberCoreGeo = buildEmberCoreGeometry();
  const emberCoreMesh = new THREE.Mesh(emberCoreGeo, mats.emberCore);
  emberCoreMesh.name = 'LanternEmberCore';
  emberCoreMesh.position.set(0, -0.34, 0);
  emberCoreMesh.renderOrder = 18;
  lanternYawGroup.add(emberCoreMesh);

  // Warm interior PointLight illuminating silk strands, bamboo ribs, and emerging butterflies
  const lanternLight = new THREE.PointLight(0xff9c42, 3.0, 5.4, 1.4);
  lanternLight.position.set(0, -0.36, 0);
  lanternYawGroup.add(lanternLight);

  // -------------------------------------------------------------------------
  // 2. 14 Articulated Faceted Origami Butterflies
  // -------------------------------------------------------------------------
  const leftWingGeo = buildOrigamiButterflyWingGeometry(-1);
  const rightWingGeo = buildOrigamiButterflyWingGeometry(1);
  const butterflyGroups: THREE.Group[] = [];
  const butterflyLeftWings: THREE.Mesh[] = [];
  const butterflyRightWings: THREE.Mesh[] = [];

  for (let i = 0; i < sim.butterflies.length; i++) {
    const bg = new THREE.Group();
    bg.name = `OrigamiButterfly_${i}`;
    bg.rotation.order = 'YXZ';
    const lw = new THREE.Mesh(leftWingGeo, mats.origamiButterfly);
    const rw = new THREE.Mesh(rightWingGeo, mats.origamiButterfly);
    lw.renderOrder = 26;
    rw.renderOrder = 26;
    bg.add(lw, rw);
    specimenRoot.add(bg);
    butterflyGroups.push(bg);
    butterflyLeftWings.push(lw);
    butterflyRightWings.push(rw);
  }

  // -------------------------------------------------------------------------
  // 3. 160 Instanced Backlit Ember Motes (gated off when stage === 'bound')
  // -------------------------------------------------------------------------
  const emberQuadGeo = buildEmberQuadGeometry();
  const emberMesh = new THREE.InstancedMesh(emberQuadGeo, mats.emberMotes, sim.embers.length);
  emberMesh.name = 'EmberMotes';
  emberMesh.renderOrder = 30;
  emberMesh.frustumCulled = false;
  specimenRoot.add(emberMesh);

  const hitMeshes: THREE.Object3D[] = [
    paperDomeMesh,
    bambooFrameMesh,
    silkWrapMesh,
    emberCoreMesh,
  ];

  // -------------------------------------------------------------------------
  // Telemetry & Programmatic Auditioning API (window.__COCOON_ALT1__)
  // -------------------------------------------------------------------------
  const getTelemetry = (): Day20Alt1Telemetry => {
    const s = sim.getState();
    _camOffset.copy(stage.camera.position).sub(stage.controls.target);
    const dist = Math.max(0.001, _camOffset.length());
    const pitch = Math.asin(Math.max(-1, Math.min(1, _camOffset.y / dist)));
    const azimuth = Math.atan2(_camOffset.x, _camOffset.z);

    return {
      stage: s.stage,
      unravel: s.unravel,
      breathExpansion: s.breathExpansion,
      ignition: s.ignition,
      emberPulse: s.emberPulse,
      lanternSwingX: s.lanternSwingX,
      lanternSwingZ: s.lanternSwingZ,
      lanternYaw: s.lanternYaw,
      lanternLiftY: s.lanternLiftY,
      butterfliesEmerged: s.butterfliesEmerged,
      activeButterflies: s.activeButterflies,
      time: s.time,
      cameraAzimuth: azimuth,
      cameraPitch: pitch,
      cameraDistance: dist,
    };
  };

  const setOrbit = (azimuth: number, pitch: number, distance: number, targetY?: number): void => {
    const clampedDist = Math.max(stage.controls.minDistance, Math.min(stage.controls.maxDistance, distance));
    const clampedPitch = Math.max(-Math.PI * 0.44, Math.min(Math.PI * 0.44, pitch));
    if (typeof targetY === 'number' && Number.isFinite(targetY)) {
      stage.controls.target.y = targetY;
    }
    const cosP = Math.cos(clampedPitch);
    stage.camera.position.set(
      stage.controls.target.x + Math.sin(azimuth) * cosP * clampedDist,
      stage.controls.target.y + Math.sin(clampedPitch) * clampedDist,
      stage.controls.target.z + Math.cos(azimuth) * cosP * clampedDist,
    );
    stage.camera.lookAt(stage.controls.target);
    stage.controls.update();
  };

  window.__COCOON_ALT1__ = {
    getState: getTelemetry,
    setPhase: (phase: ChrysalisStage) => sim.setPhase(phase),
    resetToBound: () => sim.resetToBound(),
    stokeBurst: (wx = 0, wy = -0.35, wz = 0) => sim.stokeBurst(wx, wy, wz),
    setOrbit,
  };

  // -------------------------------------------------------------------------
  // Continuous Pointer Swirl, Click Stoke Burst, Orbit & Keyboard Shortcuts
  // -------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  const bootTime = performance.now();
  let pointerTravelPx = 0;
  let lastClientX = Number.NaN;
  let lastClientY = Number.NaN;
  let hoverUnlocked = false;

  let isPointerDown = false;
  let downX = 0;
  let downY = 0;
  let hasPrevPlaneHit = false;
  let prevMoveTimeMs = performance.now();

  const updateNdcFromClient = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  };

  const projectOntoSpecimenPlane = (clientX: number, clientY: number, out: THREE.Vector3): boolean => {
    updateNdcFromClient(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);
    stage.camera.getWorldDirection(_camDir);
    _dragPlane.setFromNormalAndCoplanarPoint(_camDir.negate(), specimenRoot.position);
    if (_raycaster.ray.intersectPlane(_dragPlane, out)) {
      specimenRoot.worldToLocal(out);
      return true;
    }
    return false;
  };

  const hitsSpecimen = (clientX: number, clientY: number): boolean => {
    updateNdcFromClient(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);
    return _raycaster.intersectObjects(hitMeshes, false).length > 0;
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (!hoverUnlocked) {
      if (Number.isFinite(lastClientX)) {
        pointerTravelPx += Math.hypot(ev.clientX - lastClientX, ev.clientY - lastClientY);
      }
      lastClientX = ev.clientX;
      lastClientY = ev.clientY;
      if (performance.now() - bootTime > 360 && pointerTravelPx >= 8) {
        hoverUnlocked = true;
      } else if (!isPointerDown) {
        return;
      }
    }

    const nowMs = performance.now();
    const dtMove = Math.max(0.008, (nowMs - prevMoveTimeMs) / 1000);
    prevMoveTimeMs = nowMs;

    if (projectOntoSpecimenPlane(ev.clientX, ev.clientY, _planeHit)) {
      let vx = 0;
      let vy = 0;
      let vz = 0;
      if (hasPrevPlaneHit) {
        vx = (_planeHit.x - _prevPlaneHit.x) / dtMove;
        vy = (_planeHit.y - _prevPlaneHit.y) / dtMove;
        vz = (_planeHit.z - _prevPlaneHit.z) / dtMove;
      }
      sim.applyPointer(_planeHit.x, _planeHit.y, _planeHit.z, vx, vy, vz, isPointerDown, dtMove);
      mats.uniforms.uPointerWorld.value.copy(_planeHit);
      _pointerVel.set(
        Math.max(-4, Math.min(4, vx)),
        Math.max(-4, Math.min(4, vy)),
        Math.max(-4, Math.min(4, vz)),
      );
      _prevPlaneHit.copy(_planeHit);
      hasPrevPlaneHit = true;
    }

    if (!isPointerDown) {
      canvas.style.cursor = hitsSpecimen(ev.clientX, ev.clientY) ? 'pointer' : 'grab';
    }

    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
  };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    hoverUnlocked = true;
    isPointerDown = true;
    downX = ev.clientX;
    downY = ev.clientY;
    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
    prevMoveTimeMs = performance.now();
    canvas.style.cursor = 'grabbing';

    if (projectOntoSpecimenPlane(ev.clientX, ev.clientY, _planeHit)) {
      _prevPlaneHit.copy(_planeHit);
      hasPrevPlaneHit = true;
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (!isPointerDown) return;
    isPointerDown = false;
    const movedPx = Math.hypot(ev.clientX - downX, ev.clientY - downY);

    if (movedPx < 6 && hitsSpecimen(ev.clientX, ev.clientY)) {
      if (projectOntoSpecimenPlane(ev.clientX, ev.clientY, _planeHit)) {
        sim.stokeBurst(_planeHit.x, _planeHit.y, _planeHit.z);
      } else {
        sim.stokeBurst(0, -0.35, 0);
      }
    }

    canvas.style.cursor = hitsSpecimen(ev.clientX, ev.clientY) ? 'pointer' : 'grab';
  };

  const onDoubleClick = () => {
    sim.resetToBound();
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      sim.resetToBound();
    } else if (ev.code === 'Digit1') {
      sim.setPhase('bound');
    } else if (ev.code === 'Digit2') {
      sim.setPhase('unwinding');
    } else if (ev.code === 'Digit3') {
      sim.setPhase('emerged');
    } else if (ev.code === 'Digit4') {
      sim.setPhase('buoyant');
    }
  };

  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('dblclick', onDoubleClick);
  window.addEventListener('keydown', onKeyDown);

  // -------------------------------------------------------------------------
  // Animation Loop (Strictly Zero Per-Frame Heap Allocations)
  // -------------------------------------------------------------------------
  const syncSceneToSim = (elapsedSec: number, dtSec: number, camera: THREE.Camera) => {
    const s = sim.state;

    // 1. Isotropic world-space pendulum tilt on top cord pivot + decoupled torsional yaw on child
    const tiltAngle = Math.hypot(s.lanternSwingX, s.lanternSwingZ);
    if (tiltAngle > 1e-6) {
      _tiltAxis.set(s.lanternSwingX / tiltAngle, 0, s.lanternSwingZ / tiltAngle);
      pendulumPivot.quaternion.setFromAxisAngle(_tiltAxis, tiltAngle);
    } else {
      pendulumPivot.quaternion.identity();
    }

    lanternYawGroup.position.set(0, -PENDULUM_PIVOT_Y + s.lanternLiftY, 0);
    lanternYawGroup.rotation.set(0, s.lanternYaw, 0);

    const coreScale = 0.72 + 0.32 * s.ignition + 0.16 * s.emberPulse;
    emberCoreMesh.scale.set(coreScale, coreScale * 1.15, coreScale);

    specimenRoot.updateMatrixWorld(true);

    // 2. Smoothly decay pointer velocity wind uniform
    const velDecay = Math.exp(-Math.max(0.001, dtSec) * 6.0);
    _pointerVel.multiplyScalar(velDecay);
    mats.uniforms.uPointerVel.value.copy(_pointerVel);

    // 3. Synchronize TSL material uniforms & interior candle PointLight
    const breathPulse = 1.0 + s.emberPulse * 0.54 + Math.sin(elapsedSec * 2.3) * 0.06;
    mats.uniforms.uTime.value = elapsedSec;
    mats.uniforms.uUnravel.value = s.unravel;
    mats.uniforms.uBreath.value = s.breathExpansion;
    mats.uniforms.uIgnition.value = s.ignition;
    mats.uniforms.uEmberPulse.value = breathPulse;

    emberCoreMesh.getWorldPosition(_flameWorldPos);
    mats.uniforms.uFlamePos.value.copy(_flameWorldPos);

    lanternLight.intensity = (0.55 + s.ignition * 3.1) * breathPulse;

    // 4. Update 14 articulated origami butterflies
    for (let i = 0; i < sim.butterflies.length; i++) {
      const bf = sim.butterflies[i];
      const bg = butterflyGroups[i];
      bg.visible = bf.active;
      if (!bf.active) continue;

      bg.position.set(bf.pos[0], bf.pos[1], bf.pos[2]);
      bg.rotation.set(bf.pitch, bf.yaw, bf.roll);
      bg.scale.setScalar(bf.scale * BUTTERFLY_BASE_SCALE);

      const flap = Math.sin(bf.wingPhase) * 0.68 + 0.14;
      butterflyLeftWings[i].rotation.z = -flap;
      butterflyRightWings[i].rotation.z = flap;
    }

    // 5. Update 160 instanced ember motes (strictly hidden when tightly bound)
    const emberVisible = s.stage !== 'bound' && s.unravel > 0.06;
    emberMesh.visible = emberVisible;
    if (emberVisible) {
      const emberGate = Math.min(1, (s.unravel - 0.06) * 3.0);
      for (let i = 0; i < sim.embers.length; i++) {
        const mote = sim.embers[i];
        const lifeFrac = mote.maxLife > 0 ? Math.max(0, Math.min(1, mote.life / mote.maxLife)) : 0;
        const env = Math.sin(lifeFrac * Math.PI);
        const sz = env > 0.02 ? mote.size * (0.45 + 0.75 * env) * emberGate : 0.0001;
        _dummyPos.set(mote.pos[0], mote.pos[1], mote.pos[2]);
        _dummyScale.setScalar(sz);
        _dummyMatrix.compose(_dummyPos, camera.quaternion, _dummyScale);
        emberMesh.setMatrixAt(i, _dummyMatrix);
      }
      emberMesh.instanceMatrix.needsUpdate = true;
    }
  };

  // Sync frame 0 immediately so the initial render is cleanly posed
  syncSceneToSim(0, 1 / 60, stage.camera);

  stage.onTick(({ elapsedSec, dtSec, camera }) => {
    sim.step(dtSec);
    syncSceneToSim(elapsedSec, dtSec, camera);
  });

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('dblclick', onDoubleClick);
    window.removeEventListener('keydown', onKeyDown);
    paperDomeGeo.dispose();
    bambooFrameGeo.dispose();
    silkWrapGeo.dispose();
    silkLoopsGeo.dispose();
    emberCoreGeo.dispose();
    leftWingGeo.dispose();
    rightWingGeo.dispose();
    emberQuadGeo.dispose();
    sim.dispose();
    mats.dispose();
    studioEnv.dispose();
  });

  return stage;
}

void bootDay20CocoonAlt1();
