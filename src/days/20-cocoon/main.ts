import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { MINERAL_PALETTE } from '../../core/palette';
import { KozoCocoonSimulation } from './cocoonSimulation';
import {
  buildCalyxBaseCapGeometry,
  buildEmberPearlGeometry,
  buildOrigamiWingGeometry,
  buildPollenBillboardQuadGeometry,
  buildStamenCrownGeometry,
  PETAL_BASE_Y,
} from './kozoGeometry';
import {
  createCocoonStudioEnvironment,
  createKozoMaterials,
} from './kozoMaterials';
import { DEFAULT_KOZO_CONFIG, type Day20TelemetryState } from './types';

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
const _camLocal = new THREE.Vector3();
const _planeHit = new THREE.Vector3();
const _prevPlaneHit = new THREE.Vector3();
const _dragPlane = new THREE.Plane();
const _camDir = new THREE.Vector3();
const _radialDir = new THREE.Vector3();
const _dragVec = new THREE.Vector3();
const _dummyMatrix = new THREE.Matrix4();
const _dummyScale = new THREE.Vector3();
const _dummyPos = new THREE.Vector3();
const _colorA = new THREE.Color(MINERAL_PALETTE.boneCalcite.hex);
const _colorB = new THREE.Color(MINERAL_PALETTE.yellowOchre.hex);
const _colorC = new THREE.Color(MINERAL_PALETTE.realgarAmber.hex);
const _tmpCol = new THREE.Color();

export async function bootDay20Cocoon(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(20);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.18, 2.38, 5.75],
    fov: 35,
    enableOrbit: true,
    enableShadows: false,
  });

  stage.camera.near = 0.05;
  stage.camera.updateProjectionMatrix();

  stage.controls.target.set(0, 0.06, 0);
  stage.controls.minDistance = 2.0;
  stage.controls.maxDistance = 10.8;
  stage.controls.zoomSpeed = 0.85;
  stage.controls.enableDamping = true;
  stage.controls.dampingFactor = 0.065;
  stage.controls.update();

  const studioEnv = createCocoonStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 0.72;

  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(2.8, 4.2, 3.6);
        child.intensity = 1.72;
      } else if (dirIdx === 1) {
        child.position.set(-3.4, -1.0, 2.4);
        child.intensity = 0.65;
      } else if (dirIdx === 2) {
        child.position.set(-1.2, 3.6, -3.8);
        child.intensity = 1.65;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.62;
    }
  }

  const specimenRoot = new THREE.Group();
  specimenRoot.name = 'TornRicePaperCocoonRoot';
  stage.specimenGroup.add(specimenRoot);

  const sim = new KozoCocoonSimulation(DEFAULT_KOZO_CONFIG);
  sim.bloom = 0.36;
  sim.setTargetBloom(DEFAULT_KOZO_CONFIG.initialBloom);

  const mats = createKozoMaterials();

  // Internal honey-amber candle light cradled in the floral heart
  const heartLight = new THREE.PointLight(0xea8b38, 2.2, 4.8, 1.5);
  heartLight.position.set(0, PETAL_BASE_Y + 0.36, 0.04);
  specimenRoot.add(heartLight);

  // Calyx base cup sealing petal roots
  const calyxGeo = buildCalyxBaseCapGeometry();
  const calyxMesh = new THREE.Mesh(calyxGeo, mats.bloomPetalMat);
  calyxMesh.name = 'RicePaperCalyxCap';
  calyxMesh.renderOrder = 1;
  specimenRoot.add(calyxMesh);

  // 18 poised torn rice-paper petals across 3 staggered whorls
  const petalMeshes: THREE.Mesh[] = [];
  const hitMeshes: THREE.Object3D[] = [calyxMesh];
  for (let p = 0; p < sim.petals.length; p++) {
    const m = new THREE.Mesh(sim.petals[p].geometry, mats.bloomPetalMat);
    m.name = `BloomPetal_W${sim.petals[p].spec.whorl}_${p}`;
    specimenRoot.add(m);
    petalMeshes.push(m);
    hitMeshes.push(m);
  }

  // Delicate curved botanical stamen crown + soft nectar pearl (strictly between rear & front petals)
  const emberGeo = buildEmberPearlGeometry();
  const emberMesh = new THREE.Mesh(emberGeo, mats.emberMat);
  emberMesh.name = 'FloralNectarPearl';
  emberMesh.position.set(0, PETAL_BASE_Y + 0.14, 0);
  emberMesh.renderOrder = 19;
  specimenRoot.add(emberMesh);
  hitMeshes.push(emberMesh);

  const stamenGeo = buildStamenCrownGeometry(sim.config.stamenCount);
  const stamenMesh = new THREE.Mesh(stamenGeo, mats.stamenMat);
  stamenMesh.name = 'BotanicalStamenCrown';
  stamenMesh.renderOrder = 20;
  specimenRoot.add(stamenMesh);

  // 12 buoyant curl-gliding faceted Realgar-Amber origami moths
  const leftWingGeo = buildOrigamiWingGeometry(-1);
  const rightWingGeo = buildOrigamiWingGeometry(1);
  const mothGroups: THREE.Group[] = [];
  const mothLeftWings: THREE.Mesh[] = [];
  const mothRightWings: THREE.Mesh[] = [];

  for (let i = 0; i < sim.moths.length; i++) {
    const mg = new THREE.Group();
    mg.name = `OrigamiMoth_${i}`;
    const lw = new THREE.Mesh(leftWingGeo, mats.mothMat);
    const rw = new THREE.Mesh(rightWingGeo, mats.mothMat);
    lw.renderOrder = 42;
    rw.renderOrder = 42;
    mg.add(lw, rw);
    specimenRoot.add(mg);
    mothGroups.push(mg);
    mothLeftWings.push(lw);
    mothRightWings.push(rw);
  }

  // 280 soft radial-glow backlit pollen & silk motes
  const pollenGeo = buildPollenBillboardQuadGeometry();
  const pollenMesh = new THREE.InstancedMesh(pollenGeo, mats.pollenMat, sim.config.pollenCount);
  pollenMesh.name = 'LuminousPollenMotes';
  pollenMesh.renderOrder = 44;
  pollenMesh.frustumCulled = false;
  for (let i = 0; i < sim.config.pollenCount; i++) {
    const w = sim.pollenWarmth[i];
    if (w < 0.45) {
      _tmpCol.lerpColors(_colorA, _colorB, w / 0.45);
    } else {
      _tmpCol.lerpColors(_colorB, _colorC, (w - 0.45) / 0.55);
    }
    pollenMesh.setColorAt(i, _tmpCol);
  }
  if (pollenMesh.instanceColor) pollenMesh.instanceColor.needsUpdate = true;
  specimenRoot.add(pollenMesh);

  window.__DAY20__ = {
    getState: () => sim.getTelemetry(),
  };

  // =========================================================================
  // Natural Air-Wake, Direct Petal-Pull & Orbit Controls
  // =========================================================================
  const canvas = stage.shell.canvas;
  let bootTime = performance.now();
  let pointerTravelPx = 0;
  let lastClientX = Number.NaN;
  let lastClientY = Number.NaN;
  let hoverUnlocked = false;

  let isDraggingPetal = false;
  let grabbedPetalIdx = -1;
  let activePointerId = -1;
  let downX = 0;
  let downY = 0;
  let hasMovedDuringDrag = false;
  let hasPrevPlaneHit = false;
  let prevMoveTimeMs = performance.now();

  const updateNdcFromClient = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  };

  const raycastPetal = (clientX: number, clientY: number): number => {
    updateNdcFromClient(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);
    const hits = _raycaster.intersectObjects(hitMeshes, false);
    if (hits.length === 0) return -2;
    const obj = hits[0].object;
    return petalMeshes.indexOf(obj as THREE.Mesh);
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

  const onPointerMove = (ev: PointerEvent) => {
    if (!hoverUnlocked) {
      if (Number.isFinite(lastClientX)) {
        pointerTravelPx += Math.hypot(ev.clientX - lastClientX, ev.clientY - lastClientY);
      }
      lastClientX = ev.clientX;
      lastClientY = ev.clientY;
      if (performance.now() - bootTime > 360 && pointerTravelPx >= 8) {
        hoverUnlocked = true;
      } else if (!isDraggingPetal) {
        return;
      }
    }

    const nowMs = performance.now();
    const dtMove = Math.max(0.008, (nowMs - prevMoveTimeMs) / 1000);
    prevMoveTimeMs = nowMs;

    const planeValid = projectOntoSpecimenPlane(ev.clientX, ev.clientY, _planeHit);
    let vx = 0;
    let vy = 0;
    let vz = 0;
    if (planeValid && hasPrevPlaneHit) {
      vx = (_planeHit.x - _prevPlaneHit.x) / dtMove;
      vy = (_planeHit.y - _prevPlaneHit.y) / dtMove;
      vz = (_planeHit.z - _prevPlaneHit.z) / dtMove;
    }

    if (isDraggingPetal && ev.pointerId === activePointerId && planeValid) {
      const movedPx = Math.hypot(ev.clientX - downX, ev.clientY - downY);
      if (movedPx > 4) hasMovedDuringDrag = true;

      if (hasPrevPlaneHit) {
        _dragVec.copy(_planeHit).sub(_prevPlaneHit);
        if (grabbedPetalIdx >= 0) {
          const az = sim.petals[grabbedPetalIdx].spec.azimuth;
          _radialDir.set(Math.sin(az), -0.32, Math.cos(az)).normalize();
          const outwardDelta = _dragVec.dot(_radialDir);
          const lateralDelta = _dragVec.x * Math.cos(az) - _dragVec.z * Math.sin(az);
          sim.pullPetal(grabbedPetalIdx, outwardDelta * 0.95, lateralDelta * 0.85);
        } else {
          const prevDist = Math.hypot(_prevPlaneHit.x, _prevPlaneHit.y * 0.7);
          const curDist = Math.hypot(_planeHit.x, _planeHit.y * 0.7);
          sim.pullPetal(-1, (curDist - prevDist) * 0.95, 0);
        }
      }
      sim.setAirWake(_planeHit.x, _planeHit.y, _planeHit.z, vx, vy, vz, true);
    } else if (planeValid) {
      sim.setAirWake(_planeHit.x, _planeHit.y, _planeHit.z, vx, vy, vz, true);
      const hitIdx = raycastPetal(ev.clientX, ev.clientY);
      canvas.style.cursor = hitIdx !== -2 ? 'grab' : 'default';
    }

    if (planeValid) {
      _prevPlaneHit.copy(_planeHit);
      hasPrevPlaneHit = true;
    }
    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
  };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    hoverUnlocked = true;
    downX = ev.clientX;
    downY = ev.clientY;
    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
    hasMovedDuringDrag = false;
    prevMoveTimeMs = performance.now();

    if (projectOntoSpecimenPlane(ev.clientX, ev.clientY, _planeHit)) {
      _prevPlaneHit.copy(_planeHit);
      hasPrevPlaneHit = true;
    }

    const hitIdx = raycastPetal(ev.clientX, ev.clientY);
    if (hitIdx !== -2) {
      isDraggingPetal = true;
      grabbedPetalIdx = hitIdx;
      activePointerId = ev.pointerId;
      stage.controls.enabled = false;
      canvas.style.cursor = 'grabbing';
      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // synthetic pointer events may lack active capture
      }
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (isDraggingPetal && ev.pointerId === activePointerId) {
      isDraggingPetal = false;
      grabbedPetalIdx = -1;
      activePointerId = -1;
      stage.controls.enabled = true;
      canvas.style.cursor = 'grab';
      try {
        if (canvas.hasPointerCapture(ev.pointerId)) {
          canvas.releasePointerCapture(ev.pointerId);
        }
      } catch {
        // ignore
      }

      if (!hasMovedDuringDrag) {
        sim.pulseBloomBurst();
      }
    }
  };

  const onDoubleClick = (ev: MouseEvent) => {
    const hitIdx = raycastPetal(ev.clientX, ev.clientY);
    if (hitIdx === -2) {
      sim.cycleBloomPreset();
    } else {
      sim.pulseBloomBurst();
    }
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      sim.cycleBloomPreset();
    } else if (ev.code === 'KeyR') {
      sim.setTargetBloom(DEFAULT_KOZO_CONFIG.initialBloom);
      bootTime = performance.now();
      pointerTravelPx = 0;
      hoverUnlocked = false;
    }
  };

  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('dblclick', onDoubleClick);
  window.addEventListener('keydown', onKeyDown);

  // =========================================================================
  // Animation Loop (Zero Per-Frame Heap Allocations)
  // =========================================================================
  stage.onTick(({ elapsedSec, dtSec, camera }) => {
    sim.step(dtSec);

    specimenRoot.position.y = Math.sin(elapsedSec * 0.72) * 0.024;
    specimenRoot.rotation.y = Math.sin(elapsedSec * 0.30) * 0.055;
    specimenRoot.updateMatrixWorld();

    mats.uniforms.uTime.value = elapsedSec;
    mats.uniforms.uBloom.value = sim.bloom;
    mats.uniforms.uEmberGlow.value = sim.emberGlow;
    mats.uniforms.uCandleOffset.value.copy(sim.candleOffset);
    mats.uniforms.uClosedShadow.value = Math.max(0, Math.min(1, 1.0 - sim.bloom / 0.32));
    mats.uniforms.uHoverLocal.value.copy(sim.airWakeLocal);
    mats.uniforms.uHoverStrength.value = sim.airWakeStrength;

    const heartPulse = 1.0 + Math.sin(elapsedSec * 2.1) * 0.06;
    heartLight.intensity = (0.95 + sim.bloom * 1.75) * sim.emberGlow * heartPulse;
    heartLight.position.set(
      sim.candleOffset.x * 0.06,
      PETAL_BASE_Y + 0.36 + sim.candleOffset.y * 0.05,
      0.04
    );
    emberMesh.scale.setScalar(0.84 + sim.bloom * 0.18 + (sim.emberGlow - 1.0) * 0.14);

    _camLocal.copy(camera.position);
    specimenRoot.worldToLocal(_camLocal);
    sim.applyBackToFrontRenderOrder(_camLocal, petalMeshes);

    for (let i = 0; i < sim.moths.length; i++) {
      const m = sim.moths[i];
      const mg = mothGroups[i];
      mg.visible = m.active;
      if (!m.active) continue;

      mg.position.copy(m.pos);
      mg.rotation.set(m.pitch, m.heading, m.roll);
      mg.scale.setScalar(m.scale * (0.45 + 0.55 * m.liftProgress));

      const flap = Math.sin(m.wingPhase) * 0.66 + 0.16;
      mothLeftWings[i].rotation.z = -flap;
      mothRightWings[i].rotation.z = flap;
    }

    for (let i = 0; i < sim.config.pollenCount; i++) {
      const i3 = i * 3;
      const alpha = sim.pollenAlpha[i];
      const sz = alpha > 0.02 ? sim.pollenSize[i] * (0.45 + 0.75 * alpha) : 0.0001;
      _dummyPos.set(sim.pollenPos[i3], sim.pollenPos[i3 + 1], sim.pollenPos[i3 + 2]);
      _dummyScale.setScalar(sz);
      _dummyMatrix.compose(_dummyPos, camera.quaternion, _dummyScale);
      pollenMesh.setMatrixAt(i, _dummyMatrix);
    }
    pollenMesh.instanceMatrix.needsUpdate = true;
  });

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('dblclick', onDoubleClick);
    window.removeEventListener('keydown', onKeyDown);
    calyxGeo.dispose();
    stamenGeo.dispose();
    emberGeo.dispose();
    leftWingGeo.dispose();
    rightWingGeo.dispose();
    pollenGeo.dispose();
    sim.dispose();
    mats.dispose();
    studioEnv.dispose();
  });

  return stage;
}

void bootDay20Cocoon();
