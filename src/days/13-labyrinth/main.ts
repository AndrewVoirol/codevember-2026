import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { createLabyrinthTopology } from './labyrinthTopology';
import { LabyrinthBeadSimulator } from './labyrinthPhysics';
import {
  buildVaultOctantGeometries,
  createLabyrinthGeometries,
  hydrateVaultOctantGeometries,
  type VaultOctantsRawPayload,
  type VaultOctantsResult,
} from './labyrinthGeometry';
import { createLabyrinthOptics } from './labyrinthOptics';
import { createLabyrinthCenterpiece } from './labyrinthCenterpiece';

const DEFAULT_CAM_POS: readonly [number, number, number] = [0, 0.12, 4.35];
const MIN_CAM_DIST = 2.45;
const MAX_CAM_DIST = 6.8;
const MAX_HOVER_BIAS_RAD = 0.004;
const TRACKBALL_RADIUS = 0.82;

// Pre-allocated module-scope math scratch objects (strictly zero per-frame allocations)
const _xAxis = new THREE.Vector3(1, 0, 0);
const _yAxis = new THREE.Vector3(0, 1, 0);
const _zAxis = new THREE.Vector3(0, 0, 1);
const _pitchQuat = new THREE.Quaternion();
const _yawQuat = new THREE.Quaternion();
const _rollQuat = new THREE.Quaternion();
const _deltaQuat = new THREE.Quaternion();
const _inertiaQuat = new THREE.Quaternion();
const _inertiaAxis = new THREE.Vector3();
const _hoverQ1 = new THREE.Quaternion();
const _hoverQ2 = new THREE.Quaternion();
const _hoverQuat = new THREE.Quaternion();
const _invMonolithQuat = new THREE.Quaternion();
const _camRight = new THREE.Vector3();
const _camUp = new THREE.Vector3();
const _camBack = new THREE.Vector3();
const _prevTrackball = new THREE.Vector3();
const _curTrackball = new THREE.Vector3();
const _axisCam = new THREE.Vector3();
const _axisWorld = new THREE.Vector3();
const _instOmega = new THREE.Vector3();
const _angVelWorld = new THREE.Vector3();
const _prevMonolithQuat = new THREE.Quaternion();
const _rotFrameDeltaQuat = new THREE.Quaternion();
const _rotFrameVec = new THREE.Vector3();
const _rotFrameLocalVelVec = new THREE.Vector3();
const _rotFramePrevLocalVelVec = new THREE.Vector3();
const _rotFrameAccelVec = new THREE.Vector3();
const _worldGravity = new THREE.Vector3(0, -9.81, 0);
const _localGravityVec = new THREE.Vector3();
const _localGravityArr: [number, number, number] = [0, -9.81, 0];
const _localAngVelArr: [number, number, number] = [0, 0, 0];
const _localAngAccelArr: [number, number, number] = [0, 0, 0];
const _stepGravArr: [number, number, number] = [0, -9.81, 0];
const _beadVelScratch: [number, number, number] = [0, 0, 0];
const _clickRaycaster = new THREE.Raycaster();
const _clickNdc = new THREE.Vector2();
const _sanctumHits: THREE.Intersection[] = [];
const _octTiltAxis = new THREE.Vector3();
const _orbitTargetPos = new THREE.Vector3();
const _orbitTargetVel = new THREE.Vector3();
const _orbitNormal = new THREE.Vector3();
const _descentStartPos = new THREE.Vector3();

export type LabyrinthVictoryPhase = 'playing' | 'ignition' | 'opening' | 'revealed';

export interface LabyrinthTelemetryState {
  edgeId: number;
  t: number;
  speed: number;
  localPos: [number, number, number];
  localNormal: [number, number, number];
  localTangent: [number, number, number];
  impactImpulse: number;
  solved: boolean;
  solveProgress01: number;
  checkpointsUnlocked: number;
  dowsingHint01: number;
  victoryPhase: LabyrinthVictoryPhase;
  vaultOpen01: number;
  startNodeId: number;
  goalNodeId: number;
  nearestNodeId: number;
  solutionEdgeCount: number;
  wetTrailTexels: number;
  verdigrisTrailTexels: number;
  quaternion: [number, number, number, number];
  cameraDistance: number;
}

declare global {
  interface Window {
    __LABYRINTH__?: {
      getState: () => LabyrinthTelemetryState;
      stepTowardSolution: (steps?: number) => LabyrinthTelemetryState;
      reset: () => LabyrinthTelemetryState;
    };
  }
}

function buildDefaultIsoQuaternion(out: THREE.Quaternion): THREE.Quaternion {
  // Orient +Y Top, +Z Front-Left, +X Front-Right with a gentle clockwise roll so
  // the mercury bead visibly rolls across the sunlit front-left upper channel on boot
  _yawQuat.setFromAxisAngle(_yAxis, -0.62);
  _pitchQuat.setFromAxisAngle(_xAxis, 0.70);
  _rollQuat.setFromAxisAngle(_zAxis, -0.52);
  return out.copy(_rollQuat).multiply(_pitchQuat).multiply(_yawQuat).normalize();
}

function buildVictoryWindowQuaternion(out: THREE.Quaternion): THREE.Quaternion {
  // Look into the front crux aperture so all 8 parting stone octants frame the astrolabe unobstructed
  _yawQuat.setFromAxisAngle(_yAxis, -0.20);
  _pitchQuat.setFromAxisAngle(_xAxis, 0.20);
  _rollQuat.setFromAxisAngle(_zAxis, -0.32);
  return out.copy(_rollQuat).multiply(_pitchQuat).multiply(_yawQuat).normalize();
}

function projectToTrackball(nx: number, ny: number, out: THREE.Vector3): THREE.Vector3 {
  const r2 = TRACKBALL_RADIUS * TRACKBALL_RADIUS;
  const d2 = nx * nx + ny * ny;
  const z =
    d2 <= r2 * 0.5
      ? Math.sqrt(r2 - d2)
      : (r2 * 0.5) / Math.sqrt(Math.max(1e-6, d2));
  return out.set(nx, ny, z).normalize();
}

export async function bootDay13Labyrinth(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(13);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [...DEFAULT_CAM_POS],
    fov: 34,
    enableOrbit: false,
    enableShadows: true,
  });

  stage.controls.target.set(0, 0, 0);
  stage.camera.lookAt(0, 0, 0);
  stage.camera.updateMatrixWorld();

  // Tighten shadow frustum, shift key light toward +Z/+Y, and boost +Z/-X fill light so bead & trail read brightly
  let keyDirLight: THREE.DirectionalLight | null = null;
  let shadowHalfSpan = 1.48;
  const applyShadowHalfSpan = (span: number): void => {
    if (!keyDirLight) return;
    shadowHalfSpan = span;
    keyDirLight.shadow.camera.left = -span;
    keyDirLight.shadow.camera.right = span;
    keyDirLight.shadow.camera.top = span;
    keyDirLight.shadow.camera.bottom = -span;
    keyDirLight.shadow.camera.updateProjectionMatrix();
    keyDirLight.shadow.needsUpdate = true;
  };

  stage.scene.traverse((obj) => {
    if (obj instanceof THREE.DirectionalLight) {
      if (obj.castShadow) {
        keyDirLight = obj;
        obj.position.set(0.6, 4.4, 4.2);
        obj.shadow.mapSize.set(1024, 1024);
        obj.shadow.bias = -0.0003;
        obj.shadow.normalBias = 0.006;
        applyShadowHalfSpan(1.48);
      } else if (obj.position.z > 0) {
        obj.intensity = 1.15;
      }
    }
  });

  const topology = createLabyrinthTopology();
  const simulator = new LabyrinthBeadSimulator(topology);
  const {
    monolithGeometry,
    sanctumInlayGeometry,
    startSpringGeometry,
    waystoneInlayGeometry,
    waystoneOctantSpecs,
    beadGeometry,
  } = createLabyrinthGeometries(topology);
  const optics = createLabyrinthOptics(
    stage,
    simulator.wetAtlasData,
    topology.beadRadius,
    topology.nodes[topology.goalNodeId].localPos
  );

  const monolithGroup = new THREE.Group();
  monolithGroup.name = 'MonolithicLabyrinthCube';
  stage.specimenGroup.add(monolithGroup);

  const monolithMesh = new THREE.Mesh(monolithGeometry, optics.stoneMaterial);
  monolithMesh.name = 'CarvedLabyrinthMonolith';
  monolithMesh.castShadow = true;
  monolithMesh.receiveShadow = true;
  monolithGroup.add(monolithMesh);

  const sanctumMesh = new THREE.Mesh(sanctumInlayGeometry, optics.sanctumMaterial);
  sanctumMesh.name = 'CinnabarGoalSanctum';
  sanctumMesh.castShadow = false;
  sanctumMesh.receiveShadow = true;
  monolithGroup.add(sanctumMesh);

  const startSpringMesh = new THREE.Mesh(startSpringGeometry, optics.startSpringMaterial);
  startSpringMesh.name = 'AmberStartSpring';
  startSpringMesh.castShadow = false;
  startSpringMesh.receiveShadow = true;
  monolithGroup.add(startSpringMesh);

  const waystoneMesh = new THREE.Mesh(waystoneInlayGeometry, optics.waystoneMaterial);
  waystoneMesh.name = 'VerdigrisWaystones';
  waystoneMesh.castShadow = false;
  waystoneMesh.receiveShadow = true;
  monolithGroup.add(waystoneMesh);

  const centerpiece = createLabyrinthCenterpiece(stage, optics.beadMaterial);
  centerpiece.group.visible = false;
  monolithGroup.add(centerpiece.group);

  const vaultGroup = new THREE.Group();
  vaultGroup.name = 'VaultCleavageOctants';
  vaultGroup.visible = false;
  monolithGroup.add(vaultGroup);

  const octantGroups: THREE.Group[] = [];
  for (let oi = 0; oi < 8; oi++) {
    const octG = new THREE.Group();
    octG.name = `VaultOctant_${oi}`;
    vaultGroup.add(octG);
    octantGroups.push(octG);
  }

  const beadMesh = new THREE.Mesh(beadGeometry, optics.beadMaterial);
  beadMesh.name = 'QuicksilverBead';
  beadMesh.castShadow = true;
  beadMesh.receiveShadow = true;
  monolithGroup.add(beadMesh);

  // Camera zoom & SO(3) trackball state
  const baseCamDir = new THREE.Vector3(...DEFAULT_CAM_POS).normalize();
  const defaultCamDist = Math.hypot(DEFAULT_CAM_POS[0], DEFAULT_CAM_POS[1], DEFAULT_CAM_POS[2]);
  let camDist = defaultCamDist;
  let targetCamDist = defaultCamDist;

  const defaultIsoQuat = buildDefaultIsoQuaternion(new THREE.Quaternion());
  const victoryWindowQuat = buildVictoryWindowQuaternion(new THREE.Quaternion());
  const baseQuat = defaultIsoQuat.clone();
  let isDragging = false;
  let activePointerId: number | null = null;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let lastMoveMs = 0;
  let hoverYaw = 0;
  let hoverPitch = 0;
  let hoverAnchorNdcX = 0;
  let hoverAnchorNdcY = 0;

  let victoryPhase: LabyrinthVictoryPhase = 'playing';
  let victoryElapsedSec = 0;
  let vaultOpen01 = 0;
  let waveCompleted = false;
  let camEasedOnOpen = false;
  let userDraggedDuringVictory = false;
  let disposed = false;
  let octantsBuilt = false;
  let octantsResult: VaultOctantsResult | null = null;
  let octantWorker: Worker | null = null;
  let workerTimerId: ReturnType<typeof setTimeout> | null = null;

  const sanctumClickTargets: THREE.Object3D[] = [sanctumMesh, beadMesh];

  const mountVaultOctants = (result: VaultOctantsResult): void => {
    if (disposed || octantsBuilt) return;
    octantsBuilt = true;
    octantsResult = result;
    for (let oi = 0; oi < result.octantGeometries.length; oi++) {
      const m = new THREE.Mesh(result.octantGeometries[oi], optics.stoneMaterial);
      m.castShadow = true;
      m.receiveShadow = true;
      octantGroups[oi].add(m);
    }

    const sanctumHalfPosZMesh = new THREE.Mesh(result.sanctumHalfPosZGeometry, optics.sanctumMaterial);
    sanctumHalfPosZMesh.castShadow = false;
    sanctumHalfPosZMesh.receiveShadow = true;
    octantGroups[7].add(sanctumHalfPosZMesh);

    const sanctumHalfNegZMesh = new THREE.Mesh(result.sanctumHalfNegZGeometry, optics.sanctumMaterial);
    sanctumHalfNegZMesh.castShadow = false;
    sanctumHalfNegZMesh.receiveShadow = true;
    octantGroups[6].add(sanctumHalfNegZMesh);

    const startSpringVaultMesh = new THREE.Mesh(startSpringGeometry, optics.startSpringMaterial);
    startSpringVaultMesh.castShadow = false;
    startSpringVaultMesh.receiveShadow = true;
    octantGroups[3].add(startSpringVaultMesh);

    for (let wi = 0; wi < waystoneOctantSpecs.length; wi++) {
      const spec = waystoneOctantSpecs[wi];
      const wayVaultMesh = new THREE.Mesh(spec.geometry, optics.waystoneMaterial);
      wayVaultMesh.castShadow = false;
      wayVaultMesh.receiveShadow = true;
      octantGroups[spec.octantIndex].add(wayVaultMesh);
    }

    sanctumClickTargets.push(sanctumHalfPosZMesh, sanctumHalfNegZMesh, ...centerpiece.clickTargets);
  };

  const ensureVaultOctantsBuiltSync = (): void => {
    if (!octantsBuilt && !disposed) {
      mountVaultOctants(buildVaultOctantGeometries(topology));
    }
  };

  workerTimerId = setTimeout(() => {
    workerTimerId = null;
    if (disposed || octantsBuilt) return;
    if (typeof Worker !== 'undefined') {
      try {
        octantWorker = new Worker(new URL('./vaultOctants.worker.ts', import.meta.url), {
          type: 'module',
        });
        octantWorker.onmessage = (
          ev: MessageEvent<{ type: string; payload: VaultOctantsRawPayload }>
        ) => {
          if (ev.data?.type === 'ready') {
            if (!disposed && !octantsBuilt) {
              mountVaultOctants(hydrateVaultOctantGeometries(ev.data.payload));
            }
            octantWorker?.terminate();
            octantWorker = null;
          }
        };
        octantWorker.onerror = () => {
          octantWorker?.terminate();
          octantWorker = null;
        };
        octantWorker.postMessage({ type: 'build' });
      } catch {
        octantWorker = null;
      }
    }
  }, 60);

  monolithGroup.quaternion.copy(baseQuat);
  _prevMonolithQuat.copy(monolithGroup.quaternion);
  _rotFrameLocalVelVec.set(0, 0, 0);
  _rotFramePrevLocalVelVec.set(0, 0, 0);
  _rotFrameAccelVec.set(0, 0, 0);

  const syncVisualFromSimulator = (dtSec: number) => {
    const st = simulator.state;
    beadMesh.position.set(st.localPos[0], st.localPos[1], st.localPos[2]);
    _beadVelScratch[0] =
      st.localTangent[0] * st.speed + st.localBinormal[0] * st.latSpeed;
    _beadVelScratch[1] =
      st.localTangent[1] * st.speed + st.localBinormal[1] * st.latSpeed;
    _beadVelScratch[2] =
      st.localTangent[2] * st.speed + st.localBinormal[2] * st.latSpeed;
    optics.updateBeadUniforms(
      _beadVelScratch,
      st.localNormal,
      st.impactImpulse,
      st.solveProgress01,
      dtSec,
      st.rollQuat
    );
    optics.setProgressUniforms(st.checkpointsUnlocked, st.dowsingHint01);
    if (simulator.wetAtlasDirty) {
      optics.wetTrailTexture.needsUpdate = true;
      simulator.wetAtlasDirty = false;
    }
  };

  const resetAll = () => {
    simulator.reset();
    victoryPhase = 'playing';
    victoryElapsedSec = 0;
    vaultOpen01 = 0;
    waveCompleted = false;
    camEasedOnOpen = false;
    userDraggedDuringVictory = false;
    optics.setProgressUniforms(0, 0);
    optics.setVictoryUniforms(0, 0, 0);
    optics.wetTrailTexture.needsUpdate = true;
    simulator.wetAtlasDirty = false;

    monolithGroup.position.set(0, 0, 0);
    monolithMesh.visible = true;
    sanctumMesh.visible = true;
    startSpringMesh.visible = true;
    waystoneMesh.visible = true;
    vaultGroup.visible = false;
    centerpiece.group.visible = false;
    centerpiece.update(0, 0, 0);
    for (let oi = 0; oi < 8; oi++) {
      octantGroups[oi].position.set(0, 0, 0);
      octantGroups[oi].quaternion.identity();
    }
    applyShadowHalfSpan(1.48);

    baseQuat.copy(defaultIsoQuat);
    _angVelWorld.set(0, 0, 0);
    _hoverQuat.identity();
    hoverYaw = 0;
    hoverPitch = 0;
    hoverAnchorNdcX = 0;
    hoverAnchorNdcY = 0;
    targetCamDist = defaultCamDist;
    camDist = defaultCamDist;
    stage.camera.position.copy(baseCamDir).multiplyScalar(camDist);
    stage.camera.lookAt(0, 0, 0);
    stage.camera.updateMatrixWorld();
    monolithGroup.quaternion.copy(baseQuat);
    _prevMonolithQuat.copy(monolithGroup.quaternion);
    _rotFrameLocalVelVec.set(0, 0, 0);
    _rotFramePrevLocalVelVec.set(0, 0, 0);
    _rotFrameAccelVec.set(0, 0, 0);
    _localAngVelArr[0] = 0;
    _localAngVelArr[1] = 0;
    _localAngVelArr[2] = 0;
    _localAngAccelArr[0] = 0;
    _localAngAccelArr[1] = 0;
    _localAngAccelArr[2] = 0;
    if (stage.shell.canvas.style.cursor !== '') {
      stage.shell.canvas.style.cursor = '';
    }
    optics.updateBeadUniforms([0, 0, 0], simulator.state.localNormal, 0, 0, 0, simulator.state.rollQuat);
    syncVisualFromSimulator(1 / 60);
  };

  syncVisualFromSimulator(1 / 60);

  // Pointer NDC & raycast helpers for trackball drag and solved sanctum click-to-reset
  const canvas = stage.shell.canvas;
  const setCanvasCursor = (cursor: string): void => {
    if (canvas.style.cursor !== cursor) {
      canvas.style.cursor = cursor;
    }
  };
  const isPointerOverSanctumOrBead = (clientX: number, clientY: number): boolean => {
    const rect = canvas.getBoundingClientRect();
    if (
      rect.width <= 0 ||
      rect.height <= 0 ||
      clientX < rect.left ||
      clientX > rect.right ||
      clientY < rect.top ||
      clientY > rect.bottom
    ) {
      return false;
    }
    _clickNdc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    _clickRaycaster.setFromCamera(_clickNdc, stage.camera);
    _sanctumHits.length = 0;
    _clickRaycaster.intersectObjects(sanctumClickTargets, true, _sanctumHits);
    return _sanctumHits.length > 0;
  };
  const clientToNdc = (clientX: number, clientY: number, out: [number, number]): void => {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    const nx = ((clientX - rect.left) / w) * 2 - 1;
    const ny = -((clientY - rect.top) / h) * 2 + 1;
    out[0] = Math.max(-1.25, Math.min(1.25, nx));
    out[1] = Math.max(-1.25, Math.min(1.25, ny));
  };
  const _ndcPair: [number, number] = [0, 0];

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    // Fold active hover bias seamlessly into baseQuat so grab produces zero orientation snap
    if (Math.abs(hoverYaw) > 1e-6 || Math.abs(hoverPitch) > 1e-6) {
      baseQuat.premultiply(_hoverQuat).normalize();
      _hoverQuat.identity();
      hoverYaw = 0;
      hoverPitch = 0;
    }
    isDragging = true;
    if (simulator.state.solved) {
      userDraggedDuringVictory = true;
    }
    activePointerId = ev.pointerId;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    setCanvasCursor('');
    _angVelWorld.set(0, 0, 0);
    lastMoveMs = performance.now();
    clientToNdc(ev.clientX, ev.clientY, _ndcPair);
    projectToTrackball(_ndcPair[0], _ndcPair[1], _prevTrackball);
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // Synthetic pointer events in headless harnesses may omit pointer capture
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (!isDragging) {
      if (simulator.state.solved && isPointerOverSanctumOrBead(ev.clientX, ev.clientY)) {
        setCanvasCursor('pointer');
      } else {
        setCanvasCursor('');
      }
      return;
    }
    if (activePointerId !== null && ev.pointerId !== activePointerId) return;

    clientToNdc(ev.clientX, ev.clientY, _ndcPair);
    projectToTrackball(_ndcPair[0], _ndcPair[1], _curTrackball);

    _axisCam.crossVectors(_prevTrackball, _curTrackball);
    const sinAngle = _axisCam.length();
    if (sinAngle > 1e-6) {
      _axisCam.divideScalar(sinAngle);
      const angle = 2.25 * Math.asin(Math.min(1, sinAngle));
      _axisWorld.copy(_axisCam).transformDirection(stage.camera.matrixWorld).normalize();

      _deltaQuat.setFromAxisAngle(_axisWorld, angle);
      baseQuat.premultiply(_deltaQuat).normalize();

      const nowMs = performance.now();
      const dtMove = Math.max(0.005, Math.min(0.05, (nowMs - lastMoveMs) * 0.001));
      lastMoveMs = nowMs;

      _instOmega.copy(_axisWorld).multiplyScalar(angle / dtMove);
      _angVelWorld.lerp(_instOmega, 0.45);
      const speed = _angVelWorld.length();
      if (speed > 10.5) {
        _angVelWorld.multiplyScalar(10.5 / speed);
      }
    }

    _prevTrackball.copy(_curTrackball);
  };

  const endDrag = (ev: PointerEvent) => {
    if (!isDragging) return;
    if (activePointerId !== null && ev.pointerId !== activePointerId) return;
    if (performance.now() - lastMoveMs > 75) {
      _angVelWorld.set(0, 0, 0);
    }
    clientToNdc(ev.clientX, ev.clientY, _ndcPair);
    hoverAnchorNdcX = _ndcPair[0];
    hoverAnchorNdcY = _ndcPair[1];
    isDragging = false;
    activePointerId = null;

    if (
      ev.type === 'pointerup' &&
      ev.button === 0 &&
      simulator.state.solved &&
      Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY) < 6 &&
      isPointerOverSanctumOrBead(ev.clientX, ev.clientY)
    ) {
      resetAll();
      setCanvasCursor('');
    }
  };

  const onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    const zoomScale = Math.exp(ev.deltaY * 0.0011);
    targetCamDist = Math.max(MIN_CAM_DIST, Math.min(MAX_CAM_DIST, targetCamDist * zoomScale));
  };

  const onDoubleClick = () => {
    resetAll();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', endDrag, { passive: true });
  window.addEventListener('pointercancel', endDrag, { passive: true });
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDoubleClick);

  stage.shell.registerDispose(() => {
    disposed = true;
    if (workerTimerId !== null) {
      clearTimeout(workerTimerId);
      workerTimerId = null;
    }
    if (octantWorker) {
      octantWorker.terminate();
      octantWorker = null;
    }
    setCanvasCursor('');
    _sanctumHits.length = 0;
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', endDrag);
    window.removeEventListener('pointercancel', endDrag);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('dblclick', onDoubleClick);
    monolithGeometry.dispose();
    sanctumInlayGeometry.dispose();
    startSpringGeometry.dispose();
    waystoneInlayGeometry.dispose();
    for (let wi = 0; wi < waystoneOctantSpecs.length; wi++) {
      waystoneOctantSpecs[wi].geometry.dispose();
    }
    beadGeometry.dispose();
    if (octantsResult) {
      for (let oi = 0; oi < octantsResult.octantGeometries.length; oi++) {
        octantsResult.octantGeometries[oi].dispose();
      }
      octantsResult.sanctumHalfPosZGeometry.dispose();
      octantsResult.sanctumHalfNegZGeometry.dispose();
      octantsResult = null;
    }
    centerpiece.dispose();
    optics.dispose();
  });

  // Helper for diagnostic telemetry (never called inside onTick)
  const buildTelemetryState = (): LabyrinthTelemetryState => {
    const st = simulator.state;
    const px = st.localPos[0];
    const py = st.localPos[1];
    const pz = st.localPos[2];

    let nearestNodeId = topology.startNodeId;
    let bestD2 = Infinity;
    for (let i = 0; i < topology.nodes.length; i++) {
      const n = topology.nodes[i];
      const dx = n.localPos[0] - px;
      const dy = n.localPos[1] - py;
      const dz = n.localPos[2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD2) {
        bestD2 = d2;
        nearestNodeId = n.id;
      }
    }

    let wetTrailTexels = 0;
    let verdigrisTrailTexels = 0;
    const atlas = simulator.wetAtlasData;
    for (let i = 0; i < atlas.length; i += 4) {
      if (atlas[i] > 4 || atlas[i + 2] > 4) {
        wetTrailTexels++;
      }
      if (atlas[i + 3] > 96) {
        verdigrisTrailTexels++;
      }
    }

    return {
      edgeId: st.edgeId,
      t: Number(st.t.toFixed(4)),
      speed: Number(st.speed.toFixed(4)),
      localPos: [
        Number(px.toFixed(5)),
        Number(py.toFixed(5)),
        Number(pz.toFixed(5)),
      ],
      localNormal: [
        Number(st.localNormal[0].toFixed(5)),
        Number(st.localNormal[1].toFixed(5)),
        Number(st.localNormal[2].toFixed(5)),
      ],
      localTangent: [
        Number(st.localTangent[0].toFixed(5)),
        Number(st.localTangent[1].toFixed(5)),
        Number(st.localTangent[2].toFixed(5)),
      ],
      impactImpulse: Number(st.impactImpulse.toFixed(4)),
      solved: st.solved,
      solveProgress01: Number(st.solveProgress01.toFixed(4)),
      checkpointsUnlocked: st.checkpointsUnlocked,
      dowsingHint01: Number(st.dowsingHint01.toFixed(4)),
      victoryPhase,
      vaultOpen01: Number(vaultOpen01.toFixed(4)),
      startNodeId: topology.startNodeId,
      goalNodeId: topology.goalNodeId,
      nearestNodeId,
      solutionEdgeCount: topology.solutionEdgePath.length,
      wetTrailTexels,
      verdigrisTrailTexels,
      quaternion: [
        Number(monolithGroup.quaternion.x.toFixed(5)),
        Number(monolithGroup.quaternion.y.toFixed(5)),
        Number(monolithGroup.quaternion.z.toFixed(5)),
        Number(monolithGroup.quaternion.w.toFixed(5)),
      ],
      cameraDistance: Number(camDist.toFixed(4)),
    };
  };

  window.__LABYRINTH__ = {
    getState: () => buildTelemetryState(),
    stepTowardSolution: (steps = 600) => {
      const count = Math.max(0, Math.floor(steps));
      for (let i = 0; i < count && !simulator.state.solved; i++) {
        simulator.computeDownhillGravityForNextStep(_stepGravArr);
        simulator.step(1 / 60, _stepGravArr);
      }
      syncVisualFromSimulator(1 / 60);
      return buildTelemetryState();
    },
    reset: () => {
      resetAll();
      return buildTelemetryState();
    },
  };

  // Main 60-120 Hz Animation & Bead Simulation Loop (Zero Heap Allocations)
  stage.onTick(({ dtSec, pointer }) => {
    const dt = Math.min(0.033, Math.max(0.001, dtSec));

    // 1. Smooth wheel zoom
    if (Math.abs(targetCamDist - camDist) > 1e-4) {
      camDist += (targetCamDist - camDist) * (1 - Math.exp(-dt * 12));
      stage.camera.position.copy(baseCamDir).multiplyScalar(camDist);
      stage.camera.lookAt(0, 0, 0);
      stage.camera.updateMatrixWorld();
    }

    // 2. Rotational SO(3) inertia carry when released + auto-frame victory crux until user drags
    if (!isDragging) {
      if (simulator.state.solved && victoryElapsedSec >= 0.65 && !userDraggedDuringVictory) {
        _angVelWorld.multiplyScalar(Math.exp(-dt * 10));
        baseQuat.slerp(victoryWindowQuat, 1 - Math.exp(-dt * 3.6));
      }
      const omegaMag = _angVelWorld.length();
      if (omegaMag > 1e-4) {
        _inertiaAxis.copy(_angVelWorld).divideScalar(omegaMag);
        _inertiaQuat.setFromAxisAngle(_inertiaAxis, omegaMag * dt);
        baseQuat.premultiply(_inertiaQuat).normalize();
        _angVelWorld.multiplyScalar(Math.exp(-dt * 4.2));
      } else {
        _angVelWorld.set(0, 0, 0);
      }
    }

    // 3. Micro ±0.23° camera-aligned hover luster bias with 0.25 NDC release deadzone (never dislodges settled bead)
    let hoverDx = pointer.ndc.x - hoverAnchorNdcX;
    let hoverDy = pointer.ndc.y - hoverAnchorNdcY;
    hoverDx = Math.abs(hoverDx) <= 0.25 ? 0 : hoverDx - Math.sign(hoverDx) * 0.25;
    hoverDy = Math.abs(hoverDy) <= 0.25 ? 0 : hoverDy - Math.sign(hoverDy) * 0.25;
    const targetHoverX =
      !isDragging && pointer.isActive ? Math.tanh(hoverDx * 1.2) * MAX_HOVER_BIAS_RAD : 0;
    const targetHoverY =
      !isDragging && pointer.isActive ? Math.tanh(hoverDy * 1.2) * MAX_HOVER_BIAS_RAD : 0;
    const hoverAlpha = 1 - Math.exp(-dt * 7.5);
    hoverYaw += (targetHoverX - hoverYaw) * hoverAlpha;
    hoverPitch += (targetHoverY - hoverPitch) * hoverAlpha;

    stage.camera.matrixWorld.extractBasis(_camRight, _camUp, _camBack);
    _hoverQ1.setFromAxisAngle(_camUp, hoverYaw);
    _hoverQ2.setFromAxisAngle(_camRight, -hoverPitch);
    _hoverQuat.copy(_hoverQ1).multiply(_hoverQ2);

    monolithGroup.quaternion.copy(_hoverQuat).multiply(baseQuat).normalize();

    // 4. Compute cube-local rotational angular velocity & acceleration, then step physics
    _invMonolithQuat.copy(_prevMonolithQuat).invert();
    _rotFrameDeltaQuat.copy(_invMonolithQuat).multiply(monolithGroup.quaternion).normalize();
    if (_rotFrameDeltaQuat.w < 0) {
      _rotFrameDeltaQuat.set(
        -_rotFrameDeltaQuat.x,
        -_rotFrameDeltaQuat.y,
        -_rotFrameDeltaQuat.z,
        -_rotFrameDeltaQuat.w
      );
    }
    const sinHalfRot = Math.hypot(
      _rotFrameDeltaQuat.x,
      _rotFrameDeltaQuat.y,
      _rotFrameDeltaQuat.z
    );
    if (sinHalfRot > 1e-6) {
      const rotAngle = 2 * Math.atan2(sinHalfRot, _rotFrameDeltaQuat.w);
      const angSpeed = Math.min(14.0, rotAngle / dt);
      const invSinScale = angSpeed / sinHalfRot;
      _rotFrameLocalVelVec.set(
        _rotFrameDeltaQuat.x * invSinScale,
        _rotFrameDeltaQuat.y * invSinScale,
        _rotFrameDeltaQuat.z * invSinScale
      );
    } else {
      _rotFrameLocalVelVec.set(0, 0, 0);
    }
    const accelAlpha = 1 - Math.exp(-dt * 22);
    _rotFrameVec
      .copy(_rotFrameLocalVelVec)
      .sub(_rotFramePrevLocalVelVec)
      .divideScalar(dt)
      .clampLength(0, 48);
    _rotFrameAccelVec.lerp(_rotFrameVec, accelAlpha);
    _rotFramePrevLocalVelVec.copy(_rotFrameLocalVelVec);
    _prevMonolithQuat.copy(monolithGroup.quaternion);

    _localAngVelArr[0] = _rotFrameLocalVelVec.x;
    _localAngVelArr[1] = _rotFrameLocalVelVec.y;
    _localAngVelArr[2] = _rotFrameLocalVelVec.z;
    _localAngAccelArr[0] = _rotFrameAccelVec.x;
    _localAngAccelArr[1] = _rotFrameAccelVec.y;
    _localAngAccelArr[2] = _rotFrameAccelVec.z;

    _invMonolithQuat.copy(monolithGroup.quaternion).invert();
    _localGravityVec.copy(_worldGravity).applyQuaternion(_invMonolithQuat);
    _localGravityArr[0] = _localGravityVec.x;
    _localGravityArr[1] = _localGravityVec.y;
    _localGravityArr[2] = _localGravityVec.z;

    if (!simulator.state.solved) {
      victoryPhase = 'playing';
      vaultOpen01 = 0;
      simulator.step(dt, _localGravityArr, _localAngVelArr, _localAngAccelArr);
      syncVisualFromSimulator(dt);
    } else {
      victoryElapsedSec += dt;
      const t = victoryElapsedSec;
      optics.setProgressUniforms(3, 0);

      if (t < 0.75) {
        victoryPhase = 'ignition';
        vaultOpen01 = 0;
        simulator.igniteSolutionPathWave(Math.min(1, t / 0.72));
        if (simulator.wetAtlasDirty) {
          optics.wetTrailTexture.needsUpdate = true;
          simulator.wetAtlasDirty = false;
        }
        const igniteU = Math.min(1, t / 0.55);
        optics.setVictoryUniforms(1, igniteU, igniteU);

        // Pre-unlatch mechanical tremor right before octants part
        if (t > 0.58) {
          const frac = (t - 0.58) / 0.17;
          const amp = 0.0045 * frac;
          monolithGroup.position.set(
            amp * Math.sin(t * 78.0),
            amp * Math.cos(t * 93.0),
            0
          );
        }
        beadMesh.position.set(
          simulator.state.localPos[0],
          simulator.state.localPos[1],
          simulator.state.localPos[2]
        );
        optics.updateBeadUniforms([0, 0, 0], simulator.state.localNormal, 0, 1, dt);
      } else {
        monolithGroup.position.set(0, 0, 0);
        if (!waveCompleted) {
          waveCompleted = true;
          simulator.igniteSolutionPathWave(1);
          if (simulator.wetAtlasDirty) {
            optics.wetTrailTexture.needsUpdate = true;
            simulator.wetAtlasDirty = false;
          }
        }
        const postIgnite = t < 1.25 ? 1.0 : Math.max(0.35, 1.0 - (t - 1.25) * 0.55);
        optics.setVictoryUniforms(1, postIgnite, 1.0);

        ensureVaultOctantsBuiltSync();
        monolithMesh.visible = false;
        sanctumMesh.visible = false;
        startSpringMesh.visible = false;
        waystoneMesh.visible = false;
        vaultGroup.visible = true;
        centerpiece.group.visible = true;

        if (!camEasedOnOpen) {
          camEasedOnOpen = true;
          targetCamDist = Math.max(targetCamDist, 5.20);
        }

        // Gravitational power acceleration into damped mechanical pin-stop rebound (zero smoothstep braking)
        const tOpen = t - 0.75;
        let openVal: number;
        if (tOpen < 0.78) {
          openVal = Math.pow(tOpen / 0.78, 1.55);
        } else {
          const s = tOpen - 0.78;
          openVal = 1.0 + 0.048 * Math.exp(-6.2 * s) * Math.sin(14.5 * s);
        }
        vaultOpen01 = Math.max(0, openVal);
        victoryPhase = t < 2.45 ? 'opening' : 'revealed';

        const invSqrt3 = 1 / Math.sqrt(3);
        const radialStep = 0.56 * invSqrt3 * openVal;
        const tiltBase = Math.max(0, (openVal - 0.18) / 0.82) * 0.105;
        for (let oi = 0; oi < 8; oi++) {
          const sx = (oi & 4) !== 0 ? 1 : -1;
          const sy = (oi & 2) !== 0 ? 1 : -1;
          const sz = (oi & 1) !== 0 ? 1 : -1;
          octantGroups[oi].position.set(sx * radialStep, sy * radialStep, sz * radialStep);
          _octTiltAxis.set(sz * Math.SQRT1_2, 0, -sx * Math.SQRT1_2);
          octantGroups[oi].quaternion.setFromAxisAngle(_octTiltAxis, tiltBase * sy);
        }

        centerpiece.update(dt, t, openVal);

        // Smoothly carry quicksilver bead from parted Sanctum down into inner astrolabe orbit
        const goalPos = topology.nodes[topology.goalNodeId].localPos;
        if (t < 0.85) {
          beadMesh.position.set(goalPos[0] + radialStep, goalPos[1] + radialStep, goalPos[2]);
          optics.updateBeadUniforms([0, 0, 0], simulator.state.localNormal, 0, 1, dt);
        } else {
          const startStep = 0.56 * invSqrt3 * Math.pow(0.10 / 0.78, 1.55);
          _descentStartPos.set(goalPos[0] + startStep, goalPos[1] + startStep, goalPos[2]);
          const descend = Math.min(1, Math.pow((t - 0.85) / 0.95, 1.45));
          centerpiece.getBeadOrbitTarget(t, _orbitTargetPos, _orbitTargetVel);
          beadMesh.position.copy(_descentStartPos).lerp(_orbitTargetPos, descend);

          _beadVelScratch[0] = _orbitTargetVel.x * descend;
          _beadVelScratch[1] = _orbitTargetVel.y * descend;
          _beadVelScratch[2] = _orbitTargetVel.z * descend;
          _orbitNormal.copy(beadMesh.position);
          if (_orbitNormal.lengthSq() > 1e-6) {
            _orbitNormal.normalize();
          } else {
            _orbitNormal.set(0, 1, 0);
          }
          optics.updateBeadUniforms(_beadVelScratch, _orbitNormal, 0, 1, dt);
        }
      }
    }

    // 5. Smoothly expand directional shadow frustum when cleavage vault opens
    const targetHalfSpan = victoryPhase === 'opening' || victoryPhase === 'revealed' ? 2.28 : 1.48;
    const spanDiff = targetHalfSpan - shadowHalfSpan;
    if (Math.abs(spanDiff) > 1e-4) {
      const nextSpan =
        Math.abs(spanDiff) < 0.002
          ? targetHalfSpan
          : shadowHalfSpan + spanDiff * (1 - Math.exp(-dt * 8));
      applyShadowHalfSpan(nextSpan);
    }
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay13Labyrinth();
}
