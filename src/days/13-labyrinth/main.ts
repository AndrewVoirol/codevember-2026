import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { createLabyrinthTopology } from './labyrinthTopology';
import { LabyrinthBeadSimulator } from './labyrinthPhysics';
import { createLabyrinthGeometries } from './labyrinthGeometry';
import { createLabyrinthOptics } from './labyrinthOptics';

const DEFAULT_CAM_POS: readonly [number, number, number] = [0, 0.12, 4.35];
const MIN_CAM_DIST = 2.45;
const MAX_CAM_DIST = 6.8;
const MAX_HOVER_BIAS_RAD = (6 * Math.PI) / 180;
const TRACKBALL_RADIUS = 0.82;

// Pre-allocated module-scope math scratch objects (strictly zero per-frame allocations)
const _xAxis = new THREE.Vector3(1, 0, 0);
const _yAxis = new THREE.Vector3(0, 1, 0);
const _pitchQuat = new THREE.Quaternion();
const _yawQuat = new THREE.Quaternion();
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
const _worldGravity = new THREE.Vector3(0, -9.81, 0);
const _localGravityVec = new THREE.Vector3();
const _localGravityArr: [number, number, number] = [0, -9.81, 0];
const _stepGravArr: [number, number, number] = [0, -9.81, 0];
const _beadVelScratch: [number, number, number] = [0, 0, 0];

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
  startNodeId: number;
  goalNodeId: number;
  nearestNodeId: number;
  solutionEdgeCount: number;
  wetTrailTexels: number;
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
  // Orient +Y Top, +Z Front-Left, +X Front-Right symmetrically toward camera
  _yawQuat.setFromAxisAngle(_yAxis, -Math.PI * 0.25);
  _pitchQuat.setFromAxisAngle(_xAxis, 0.56);
  return out.copy(_pitchQuat).multiply(_yawQuat).normalize();
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

  // Tighten shadow frustum around the 1.76m carved stone monolith for crisp trench relief
  stage.scene.traverse((obj) => {
    if (obj instanceof THREE.DirectionalLight && obj.castShadow) {
      obj.shadow.mapSize.set(2048, 2048);
      const halfSpan = 1.48;
      obj.shadow.camera.left = -halfSpan;
      obj.shadow.camera.right = halfSpan;
      obj.shadow.camera.top = halfSpan;
      obj.shadow.camera.bottom = -halfSpan;
      obj.shadow.bias = -0.0003;
      obj.shadow.normalBias = 0.006;
      obj.shadow.camera.updateProjectionMatrix();
      obj.shadow.needsUpdate = true;
    }
  });

  const topology = createLabyrinthTopology();
  const simulator = new LabyrinthBeadSimulator(topology);
  const {
    monolithGeometry,
    sanctumInlayGeometry,
    startSpringGeometry,
    beadGeometry,
  } = createLabyrinthGeometries(topology);
  const optics = createLabyrinthOptics(stage, simulator.wetAtlasData, topology.beadRadius);

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
  const baseQuat = defaultIsoQuat.clone();
  let isDragging = false;
  let activePointerId: number | null = null;
  let lastMoveMs = 0;
  let hoverYaw = 0;
  let hoverPitch = 0;

  monolithGroup.quaternion.copy(baseQuat);

  const syncVisualFromSimulator = (dtSec: number) => {
    const st = simulator.state;
    beadMesh.position.set(st.localPos[0], st.localPos[1], st.localPos[2]);
    _beadVelScratch[0] = st.localTangent[0] * st.speed;
    _beadVelScratch[1] = st.localTangent[1] * st.speed;
    _beadVelScratch[2] = st.localTangent[2] * st.speed;
    optics.updateBeadUniforms(
      _beadVelScratch,
      st.localNormal,
      st.impactImpulse,
      st.solveProgress01,
      dtSec
    );
    if (simulator.wetAtlasDirty) {
      optics.wetTrailTexture.needsUpdate = true;
      simulator.wetAtlasDirty = false;
    }
  };

  const resetAll = () => {
    simulator.reset();
    optics.wetTrailTexture.needsUpdate = true;
    simulator.wetAtlasDirty = false;
    baseQuat.copy(defaultIsoQuat);
    _angVelWorld.set(0, 0, 0);
    hoverYaw = 0;
    hoverPitch = 0;
    targetCamDist = defaultCamDist;
    camDist = defaultCamDist;
    stage.camera.position.copy(baseCamDir).multiplyScalar(camDist);
    stage.camera.lookAt(0, 0, 0);
    stage.camera.updateMatrixWorld();
    monolithGroup.quaternion.copy(baseQuat);
    syncVisualFromSimulator(1 / 60);
  };

  syncVisualFromSimulator(1 / 60);

  // Pointer NDC helper for trackball drag
  const canvas = stage.shell.canvas;
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
    isDragging = true;
    activePointerId = ev.pointerId;
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
    if (!isDragging) return;
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
    isDragging = false;
    activePointerId = null;
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
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', endDrag);
    window.removeEventListener('pointercancel', endDrag);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('dblclick', onDoubleClick);
    monolithGeometry.dispose();
    sanctumInlayGeometry.dispose();
    startSpringGeometry.dispose();
    beadGeometry.dispose();
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
    const atlas = simulator.wetAtlasData;
    for (let i = 0; i < atlas.length; i += 4) {
      if (atlas[i] > 4 || atlas[i + 2] > 4) {
        wetTrailTexels++;
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
      startNodeId: topology.startNodeId,
      goalNodeId: topology.goalNodeId,
      nearestNodeId,
      solutionEdgeCount: topology.solutionEdgePath.length,
      wetTrailTexels,
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

    // 2. Rotational SO(3) inertia carry when released
    if (!isDragging) {
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

    // 3. Subtle ±6° camera-aligned hover gravity-tip bias (monotonic tanh, zero quadrant jumps)
    const targetHoverX =
      !isDragging && pointer.isActive
        ? Math.tanh(pointer.ndc.x * 1.45) * MAX_HOVER_BIAS_RAD
        : 0;
    const targetHoverY =
      !isDragging && pointer.isActive
        ? Math.tanh(pointer.ndc.y * 1.45) * MAX_HOVER_BIAS_RAD
        : 0;
    const hoverAlpha = 1 - Math.exp(-dt * 7.5);
    hoverYaw += (targetHoverX - hoverYaw) * hoverAlpha;
    hoverPitch += (targetHoverY - hoverPitch) * hoverAlpha;

    stage.camera.matrixWorld.extractBasis(_camRight, _camUp, _camBack);
    _hoverQ1.setFromAxisAngle(_camUp, hoverYaw);
    _hoverQ2.setFromAxisAngle(_camRight, -hoverPitch);
    _hoverQuat.copy(_hoverQ1).multiply(_hoverQ2);

    monolithGroup.quaternion.copy(_hoverQuat).multiply(baseQuat).normalize();

    // 4. Transform world gravity into cube-local frame & advance bead physics
    _invMonolithQuat.copy(monolithGroup.quaternion).invert();
    _localGravityVec.copy(_worldGravity).applyQuaternion(_invMonolithQuat);
    _localGravityArr[0] = _localGravityVec.x;
    _localGravityArr[1] = _localGravityVec.y;
    _localGravityArr[2] = _localGravityVec.z;

    simulator.step(dt, _localGravityArr);
    syncVisualFromSimulator(dt);
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay13Labyrinth();
}
