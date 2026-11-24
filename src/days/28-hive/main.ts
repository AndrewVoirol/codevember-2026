import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  CURTAIN_PIVOT_Y,
  HiveShimmerSimulation,
  type HiveTelemetry,
} from './shimmerGraph';
import {
  buildBeeAbdomenGeometry,
  buildBeeThoraxGeometry,
  buildBeeWingsGeometry,
  buildBranchGeometry,
  buildCurtainShellGeometry,
} from './hiveGeometry';
import {
  createHiveMaterials,
  createHiveStudioEnvironment,
} from './hiveMaterials';

/**
 * Day 28 — Hive: Shimmering Defense-Wave Cluster (*Apis dorsata*)
 *
 * Sculptural natural-history specimen suspended in the Chalky Slate (#212529) studio:
 * - Gnarled dark-walnut / charred-oak limb with broken side forks & bark micro-relief.
 * - ~2,640 shingled, articulated giant honeybees wrapping both sides and bottom crest
 *   of the hanging open-air teardrop bivouac (sculpted head+thorax, 5-tergite hinged
 *   abdomen, and paired veined translucent gossamer forewings).
 * - True 3D surface k-NN excitable-media defense-wave solver (Resting -> Firing ->
 *   Refractory -> Resting) driving crisp 65-75 deg abdomen flips, wing flicks,
 *   clean head-on wave annihilation, spiral eddies, and physical pendulum recoil.
 *
 * Tactile Gestures:
 * - Hover / sweep over or close to the cluster: triggers fast traveling golden-ochre waves.
 * - Click directly on the cluster: fires a high-amplitude synchronized radial shock-shimmer.
 * - Drag across the cluster: sweeps continuous predator triggers & colliding spiral fronts.
 * - Drag empty space: smooth 360-deg 3D orbit around the branch and both faces of the curtain.
 * - Double-click or Space: quiets active waves to hushed rest (or launches twin opposing-pole
 *   annihilation waves when already quiet).
 */

declare global {
  interface Window {
    __DAY28__?: {
      getState: () => HiveTelemetry;
    };
  }
}

// Pre-allocated scratch math objects (strictly zero allocations inside onTick / events)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _localHit = new THREE.Vector3();
const _lastTriggerLocal = new THREE.Vector3(1e6, 1e6, 1e6);
const _downHitLocal = new THREE.Vector3();

/**
 * Writes an orthonormal (T, U, N) frame rotated around local +X (T) by `-theta`
 * (so local -Y abdomen tips flip OUTWARD along +N and UPWARD toward +U)
 * directly into a Three.js InstancedMesh column-major Matrix4 Float32Array.
 */
function writeHingedInstanceMatrix(
  arr: Float32Array,
  offset: number,
  px: number,
  py: number,
  pz: number,
  tx: number,
  ty: number,
  tz: number,
  ux: number,
  uy: number,
  uz: number,
  nx: number,
  ny: number,
  nz: number,
  thetaRad: number,
  scaleX: number,
  scaleY: number,
  scaleZ: number
): void {
  const c = Math.cos(thetaRad);
  const s = Math.sin(thetaRad);

  // X' = T
  // Y' = c * U - s * N
  // Z' = s * U + c * N
  const yx = c * ux - s * nx;
  const yy = c * uy - s * ny;
  const yz = c * uz - s * nz;

  const zx = s * ux + c * nx;
  const zy = s * uy + c * ny;
  const zz = s * uz + c * nz;

  arr[offset] = tx * scaleX;
  arr[offset + 1] = ty * scaleX;
  arr[offset + 2] = tz * scaleX;
  arr[offset + 3] = 0;

  arr[offset + 4] = yx * scaleY;
  arr[offset + 5] = yy * scaleY;
  arr[offset + 6] = yz * scaleY;
  arr[offset + 7] = 0;

  arr[offset + 8] = zx * scaleZ;
  arr[offset + 9] = zy * scaleZ;
  arr[offset + 10] = zz * scaleZ;
  arr[offset + 11] = 0;

  arr[offset + 12] = px;
  arr[offset + 13] = py;
  arr[offset + 14] = pz;
  arr[offset + 15] = 1;
}

export async function bootDay28Hive(): Promise<SpecimenStage> {
  const stage = await SpecimenStage.create({
    day: 28,
    cameraPosition: [0.0, 0.14, 4.78],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0.0, 0.10, 0.0);
  stage.controls.minDistance = 1.55;
  stage.controls.maxDistance = 8.4;
  stage.controls.minPolarAngle = 0.22;
  stage.controls.maxPolarAngle = Math.PI * 0.85;
  stage.controls.update();

  // Studio softbox PMREM IBL so flipped ochre tergites, gossamer wings & dark-walnut bark read crisply
  const studioEnv = createHiveStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Calibrate SpecimenStage studio lights & tight 2048x2048 PCF shadow frustum around the suspended specimen
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        // Warm overhead-front-left calcite softbox key light
        child.position.set(-2.7, 4.7, 3.8);
        child.color.setHex(0xf7efe0);
        child.intensity = 2.18;
        child.shadow.mapSize.set(2048, 2048);
        const d = 2.15;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.5;
        child.shadow.bias = -0.00028;
        child.shadow.normalBias = 0.012;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        // Gentle opposing fill light sculpting the curved teardrop right flank
        child.position.set(3.6, 0.9, 2.8);
        child.color.setHex(0xd8cfc2);
        child.intensity = 0.72;
      } else if (dirIdx === 2) {
        // Crisp grazing back-top calcite rim light separating branch & wing silhouettes from #212529
        child.position.set(-1.3, 3.6, -4.1);
        child.color.setHex(0xf3efe6);
        child.intensity = 1.58;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.48;
    }
  }

  // ---------------------------------------------------------------------------
  // 1. Excitable-Media Simulation, Geometries & TSL Mineral Materials
  // ---------------------------------------------------------------------------
  const sim = new HiveShimmerSimulation();
  const graph = sim.graph;
  const unitCount = graph.unitCount;

  const materials = createHiveMaterials();

  const branchGeo = buildBranchGeometry();
  const innerCurtainGeo = buildCurtainShellGeometry(0.945, 34, 52);
  const raycastHullGeo = buildCurtainShellGeometry(1.075, 24, 36);
  const thoraxGeo = buildBeeThoraxGeometry();
  const abdomenGeo = buildBeeAbdomenGeometry();
  const wingsGeo = buildBeeWingsGeometry();

  // Shared per-instance GPU attribute: vec4(flip01, flashDrive, wingSpread, seed)
  const beeStateArr = new Float32Array(unitCount * 4);
  for (let i = 0; i < unitCount; i++) {
    beeStateArr[i * 4 + 3] = graph.seeds[i];
  }
  const beeStateAttr = new THREE.InstancedBufferAttribute(beeStateArr, 4);
  beeStateAttr.setUsage(THREE.DynamicDrawUsage);

  thoraxGeo.setAttribute('aBeeState', beeStateAttr);
  abdomenGeo.setAttribute('aBeeState', beeStateAttr);
  wingsGeo.setAttribute('aBeeState', beeStateAttr);

  // ---------------------------------------------------------------------------
  // 2. Suspended Specimen Hierarchy (Branch + Articulated Hanging Bivouac)
  // ---------------------------------------------------------------------------
  const branchGroup = new THREE.Group();
  branchGroup.name = 'GnarledWalnutBranchGroup';
  stage.specimenGroup.add(branchGroup);

  const branchMesh = new THREE.Mesh(branchGeo, materials.branchMat);
  branchMesh.castShadow = true;
  branchMesh.receiveShadow = true;
  branchMesh.name = 'GnarledWalnutBranch';
  branchGroup.add(branchMesh);

  const curtainPivotGroup = new THREE.Group();
  curtainPivotGroup.name = 'ApisDorsataCurtainPivot';
  curtainPivotGroup.position.set(0, CURTAIN_PIVOT_Y, 0);
  stage.specimenGroup.add(curtainPivotGroup);

  // Dark recessed organic brood core beneath the shingled bee layer
  const innerCurtainMesh = new THREE.Mesh(innerCurtainGeo, materials.innerCurtainMat);
  innerCurtainMesh.castShadow = true;
  innerCurtainMesh.receiveShadow = true;
  innerCurtainMesh.name = 'InnerBroodCore';
  curtainPivotGroup.add(innerCurtainMesh);

  // Invisible smooth envelope for responsive hover/click predator-approach raycasting
  const raycastHullMat = new THREE.MeshBasicMaterial({ visible: false });
  const raycastHullMesh = new THREE.Mesh(raycastHullGeo, raycastHullMat);
  raycastHullMesh.name = 'PredatorRaycastHull';
  curtainPivotGroup.add(raycastHullMesh);

  // Layer A: Static anchored fuzzy Head + Mesothorax (~2,640 instances, uploaded once)
  const thoraxMesh = new THREE.InstancedMesh(thoraxGeo, materials.thoraxMat, unitCount);
  thoraxMesh.castShadow = true;
  thoraxMesh.receiveShadow = true;
  thoraxMesh.frustumCulled = false;
  thoraxMesh.name = 'BeeThoraces';
  curtainPivotGroup.add(thoraxMesh);

  // Layer B: Hinged 5-tergite Segmented Abdomens (~2,640 instances, snapped outward 65-75 deg on wave!)
  const abdomenMesh = new THREE.InstancedMesh(abdomenGeo, materials.abdomenMat, unitCount);
  abdomenMesh.castShadow = true;
  abdomenMesh.receiveShadow = true;
  abdomenMesh.frustumCulled = false;
  abdomenMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  abdomenMesh.name = 'BeeAbdomens';
  curtainPivotGroup.add(abdomenMesh);

  // Layer C: Paired Translucent Veined Gossamer Forewings (~2,640 instances, sunlit flicker on flip!)
  const wingMesh = new THREE.InstancedMesh(wingsGeo, materials.wingMat, unitCount);
  wingMesh.castShadow = false;
  wingMesh.receiveShadow = false;
  wingMesh.frustumCulled = false;
  wingMesh.renderOrder = 2;
  wingMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  wingMesh.name = 'BeeGossamerWings';
  curtainPivotGroup.add(wingMesh);

  // Populate static thorax matrices & resting abdomen/wing poses once at init
  const thoraxArr = thoraxMesh.instanceMatrix.array as Float32Array;
  const abdomenArr = abdomenMesh.instanceMatrix.array as Float32Array;
  const wingArr = wingMesh.instanceMatrix.array as Float32Array;

  const pos = graph.positions;
  const norms = graph.normals;
  const ups = graph.upDirs;
  const tangs = graph.tangents;
  const scales = graph.scales;
  const restTilts = graph.restTilts;
  const seeds = graph.seeds;

  for (let i = 0; i < unitCount; i++) {
    const i3 = i * 3;
    const o = i * 16;
    const px = pos[i3];
    const py = pos[i3 + 1];
    const pz = pos[i3 + 2];
    const tx = tangs[i3];
    const ty = tangs[i3 + 1];
    const tz = tangs[i3 + 2];
    const ux = ups[i3];
    const uy = ups[i3 + 1];
    const uz = ups[i3 + 2];
    const nx = norms[i3];
    const ny = norms[i3 + 1];
    const nz = norms[i3 + 2];
    const s = scales[i];

    // Anchored thorax sits flush on the curtain with minimal natural posture tilt
    writeHingedInstanceMatrix(
      thoraxArr,
      o,
      px,
      py,
      pz,
      tx,
      ty,
      tz,
      ux,
      uy,
      uz,
      nx,
      ny,
      nz,
      0.04,
      s,
      s,
      s
    );
    writeHingedInstanceMatrix(
      abdomenArr,
      o,
      px,
      py,
      pz,
      tx,
      ty,
      tz,
      ux,
      uy,
      uz,
      nx,
      ny,
      nz,
      restTilts[i],
      s,
      s,
      s
    );
    const wpx = px + (0.015 * ux + 0.017 * nx) * s;
    const wpy = py + (0.015 * uy + 0.017 * ny) * s;
    const wpz = pz + (0.015 * uz + 0.017 * nz) * s;
    writeHingedInstanceMatrix(
      wingArr,
      o,
      wpx,
      wpy,
      wpz,
      tx,
      ty,
      tz,
      ux,
      uy,
      uz,
      nx,
      ny,
      nz,
      restTilts[i] * 0.85 + 0.05,
      s,
      s,
      s
    );
  }
  thoraxMesh.instanceMatrix.needsUpdate = true;
  abdomenMesh.instanceMatrix.needsUpdate = true;
  wingMesh.instanceMatrix.needsUpdate = true;

  // ---------------------------------------------------------------------------
  // 3. Tactile Predator-Approach Hover, Click Shock-Shimmer & Orbit Routing
  // ---------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  let isDraggingCluster = false;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTimeMs = 0;
  let downHitTarget: 'none' | 'cluster' | 'branch' = 'none';
  let hasUserInteracted = false;
  let hasTriggeredBootWave = false;
  let isHoveringCluster = false;
  let hoverDwellSec = 0;
  let lastMoveClientX = -1e6;
  let lastMoveClientY = -1e6;

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    _ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  }

  function raycastTarget(clientX: number, clientY: number): 'none' | 'cluster' | 'branch' {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    curtainPivotGroup.updateMatrixWorld();
    const clusterHits = _raycaster.intersectObject(raycastHullMesh, false);
    if (clusterHits.length > 0) {
      _localHit.copy(clusterHits[0].point);
      curtainPivotGroup.worldToLocal(_localHit);
      return 'cluster';
    }

    const branchHits = _raycaster.intersectObject(branchMesh, false);
    if (branchHits.length > 0) {
      _localHit.copy(branchHits[0].point);
      return 'branch';
    }

    return 'none';
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    hasUserInteracted = true;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    pointerDownTimeMs = performance.now();

    const hit = raycastTarget(ev.clientX, ev.clientY);
    downHitTarget = hit;

    if (hit === 'cluster') {
      isDraggingCluster = true;
      _downHitLocal.copy(_localHit);
      _lastTriggerLocal.copy(_localHit);
      stage.controls.enabled = false;
      canvas.style.cursor = 'crosshair';
      // Immediate crisp ignition right where the user touches the living curtain
      sim.exciteAtPoint(_localHit.x, _localHit.y, _localHit.z, 0.165, 1.18, 0.14);
      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // ignore
      }
    } else if (hit === 'branch') {
      stage.controls.enabled = false;
      canvas.style.cursor = 'pointer';
    } else {
      isDraggingCluster = false;
      stage.controls.enabled = true;
      canvas.style.cursor = 'grabbing';
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    const movePx = Math.hypot(ev.clientX - lastMoveClientX, ev.clientY - lastMoveClientY);
    lastMoveClientX = ev.clientX;
    lastMoveClientY = ev.clientY;

    // If orbiting empty space, let OrbitControls run smoothly without triggering waves
    if (ev.buttons !== 0 && !isDraggingCluster) {
      return;
    }

    const hit = raycastTarget(ev.clientX, ev.clientY);

    if (hit === 'cluster') {
      hasUserInteracted = true;
      canvas.style.cursor = 'crosshair';
      const enteredFresh = !isHoveringCluster;
      isHoveringCluster = true;

      const distMoved = _localHit.distanceTo(_lastTriggerLocal);
      if (enteredFresh || distMoved >= 0.055 || (isDraggingCluster && distMoved >= 0.04)) {
        const radius = isDraggingCluster ? 0.165 : 0.145;
        const boost = isDraggingCluster ? 1.22 : 1.05;
        const recoil = isDraggingCluster ? 0.12 : 0.08;
        sim.exciteAtPoint(_localHit.x, _localHit.y, _localHit.z, radius, boost, recoil);
        _lastTriggerLocal.copy(_localHit);
        hoverDwellSec = 0;
      } else if (movePx > 2.5) {
        hoverDwellSec += 0.06;
      }
    } else {
      isHoveringCluster = false;
      hoverDwellSec = 0;
      _lastTriggerLocal.set(1e6, 1e6, 1e6);
      if (!isDraggingCluster) {
        canvas.style.cursor = hit === 'branch' ? 'pointer' : 'grab';
      }
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    const travelPx = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTimeMs;
    const isCleanClick = travelPx < 8 && heldMs < 420;

    if (isDraggingCluster) {
      try {
        canvas.releasePointerCapture(ev.pointerId);
      } catch {
        // ignore
      }
    }

    if (isCleanClick) {
      if (downHitTarget === 'cluster') {
        // High-amplitude synchronized radial shock-shimmer with physical pendulum recoil
        sim.triggerShockWave(_downHitLocal.x, _downHitLocal.y, _downHitLocal.z);
      } else if (downHitTarget === 'branch') {
        // Tapping the gnarled branch shakes the upper collar & sends a cascading downward ripple
        sim.swayVelX += 0.14;
        sim.swayVelZ += 0.11;
        sim.exciteAtPoint(0.0, -0.06, 0.24, 0.28, 1.15, 0.16);
      }
    }

    isDraggingCluster = false;
    downHitTarget = 'none';
    stage.controls.enabled = true;
    canvas.style.cursor = isHoveringCluster ? 'crosshair' : 'grab';
  };

  const onPointerCancel = (ev: PointerEvent) => {
    try {
      canvas.releasePointerCapture(ev.pointerId);
    } catch {
      // ignore
    }
    isDraggingCluster = false;
    downHitTarget = 'none';
    stage.controls.enabled = true;
    canvas.style.cursor = 'grab';
  };

  const onPointerLeave = () => {
    isHoveringCluster = false;
    hoverDwellSec = 0;
    _lastTriggerLocal.set(1e6, 1e6, 1e6);
  };

  const triggerQuietOrTwinDemo = () => {
    hasUserInteracted = true;
    if (sim.activeFiringCount + sim.refractoryCount > 40) {
      sim.quietAll();
    } else {
      sim.triggerTwinCollision();
    }
  };

  const onDoubleClick = () => {
    triggerQuietOrTwinDemo();
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      triggerQuietOrTwinDemo();
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerCancel);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('dblclick', onDoubleClick);
  window.addEventListener('keydown', onKeyDown);

  // Read-only diagnostic telemetry hook matching verification contract
  window.__DAY28__ = {
    getState: () => sim.getTelemetry(),
  };

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerCancel);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    canvas.removeEventListener('dblclick', onDoubleClick);
    window.removeEventListener('keydown', onKeyDown);
    delete window.__DAY28__;

    stage.scene.environment = null;
    studioEnv.dispose();
    branchGeo.dispose();
    innerCurtainGeo.dispose();
    raycastHullGeo.dispose();
    raycastHullMat.dispose();
    thoraxGeo.dispose();
    abdomenGeo.dispose();
    wingsGeo.dispose();
    materials.dispose();
  });

  // ---------------------------------------------------------------------------
  // 4. Main Zero-GC Animation Loop (Excitable-Media Step + Instanced Articulation)
  // ---------------------------------------------------------------------------
  const peakFlipRef = HiveShimmerSimulation.PEAK_FLIP_RAD;
  const flipAngles = sim.flipAngle;
  const wingSpreads = sim.wingSpread;
  const flashInt = sim.flashIntensity;

  stage.onTick(({ dtSec, elapsedSec }) => {
    // Launch one subtle initial natural defense ripple on landing so the living curtain
    // immediately demonstrates its golden shimmering wave before settling to calm rest
    if (!hasTriggeredBootWave && !hasUserInteracted && elapsedSec >= 0.24) {
      hasTriggeredBootWave = true;
      sim.exciteAtPoint(-0.62, -0.32, 0.34, 0.19, 1.15, 0.14);
    }

    // Subtle predator-dwell warning pulse when pointer holds poised right in front of the cluster
    if (isHoveringCluster && !isDraggingCluster) {
      hoverDwellSec += dtSec;
      if (hoverDwellSec >= 0.85 && _lastTriggerLocal.x < 1e4) {
        hoverDwellSec = 0;
        sim.exciteAtPoint(_lastTriggerLocal.x, _lastTriggerLocal.y, _lastTriggerLocal.z, 0.14, 1.0, 0.06);
      }
    }

    sim.step(dtSec);

    // Physical pendulum recoil sway of the heavy hanging bivouac & slight timber flex
    curtainPivotGroup.rotation.set(sim.swayX, 0, sim.swayZ);
    branchGroup.rotation.z = sim.swayZ * 0.075;
    branchGroup.rotation.x = sim.swayX * 0.055;

    // Update all ~2,640 articulated abdomens, gossamer wings & per-instance shader states (0 heap allocs)
    const invPeak = 1.0 / peakFlipRef;
    for (let i = 0; i < unitCount; i++) {
      const i3 = i * 3;
      const o = i * 16;
      const o4 = i * 4;

      const px = pos[i3];
      const py = pos[i3 + 1];
      const pz = pos[i3 + 2];
      const tx = tangs[i3];
      const ty = tangs[i3 + 1];
      const tz = tangs[i3 + 2];
      const ux = ups[i3];
      const uy = ups[i3 + 1];
      const uz = ups[i3 + 2];
      const nx = norms[i3];
      const ny = norms[i3 + 1];
      const nz = norms[i3 + 2];
      const s = scales[i];
      const seed = seeds[i];

      // Subtle quiet living breathing rhythm while hushed at rest
      const breathe = Math.sin(elapsedSec * 1.45 + seed * 6.28318 + py * 2.15) * 0.022;
      const fAng = flipAngles[i];
      const wSpr = wingSpreads[i];
      const fl = flashInt[i];

      const abdTheta = restTilts[i] + fAng + breathe;
      writeHingedInstanceMatrix(
        abdomenArr,
        o,
        px,
        py,
        pz,
        tx,
        ty,
        tz,
        ux,
        uy,
        uz,
        nx,
        ny,
        nz,
        abdTheta,
        s,
        s,
        s
      );

      // Gossamer wings hinge from dorsal mesothorax, pitching outward & flaring laterally during snap
      const wingBuzz = Math.sin(elapsedSec * 46.0 + seed * 23.7) * 0.055 * wSpr;
      const wingTheta = restTilts[i] * 0.8 + 0.05 + fAng * 0.54 + wSpr * 0.26 + wingBuzz + breathe * 0.7;
      const wingSpanX = s * (1.0 + 0.38 * wSpr);
      const wpx = px + (0.015 * ux + 0.017 * nx) * s;
      const wpy = py + (0.015 * uy + 0.017 * ny) * s;
      const wpz = pz + (0.015 * uz + 0.017 * nz) * s;

      writeHingedInstanceMatrix(
        wingArr,
        o,
        wpx,
        wpy,
        wpz,
        tx,
        ty,
        tz,
        ux,
        uy,
        uz,
        nx,
        ny,
        nz,
        wingTheta,
        wingSpanX,
        s,
        s
      );

      const flipNorm = fAng * invPeak;
      beeStateArr[o4] = flipNorm > 1 ? 1 : flipNorm < 0 ? 0 : flipNorm;
      beeStateArr[o4 + 1] = fl;
      beeStateArr[o4 + 2] = wSpr;
    }

    abdomenMesh.instanceMatrix.needsUpdate = true;
    wingMesh.instanceMatrix.needsUpdate = true;
    beeStateAttr.needsUpdate = true;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay28Hive().catch((err) => {
    console.error('[Day 28 - Hive] Failed to initialize WebGPU specimen:', err);
  });
}
