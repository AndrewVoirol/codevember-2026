import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  BLADE_COUNT,
  BLADE_FURLED_ANGLES_RAD,
  BLADE_FURLED_SLIDES,
  BLADE_LENGTHS,
  BLADE_OPEN_ANGLES_RAD,
  PIVOT_Y,
  Z_LAYER_STEP,
  createBoneFerruleGeometry,
  createBronzeGuardLeafGeometry,
  createFeatherBarbsGeometry,
  createPivotPinGeometry,
  createRachisGeometry,
} from './featherGeometry';
import {
  type BladeOpticsUniforms,
  type SharedFeatherOpticsUniforms,
  createBoneFerruleMaterial,
  createBronzeGuardMaterial,
  createFeatherBarbMaterial,
  createFeatherPigmentSet,
  createGoniometricBackdropNode,
  createRachisMaterial,
} from './featherOptics';

/**
 * Day 12 — Feather: "Goniometric Peacock Fan"
 *
 * Seven layered, curved 3D Indian peacock (*Pavo cristatus*) train feathers anchored
 * into a stepped 7-leaf dark-bronze & Bone Calcite pivot clasp in the Chalky Slate
 * (#212529) studio:
 * - Real 3D anatomy: curved tapered rachises + 2,492 discrete curved/cambered 3D barb
 *   strands carrying local ocellus coordinates (zero flat textured sprites).
 * - Photonic-crystal structural iridescence: true angle-dependent thin-film shifts across
 *   lapisUltramarine, azuritePrussian, verdigrisMalachite, yellowOchre, and realgarAmber
 *   with inter-feather cast PCF shadows.
 * - Tactile gestures:
 *   * Hover / sweep pointer: steers the goniometer studio beam & parts/flexes barbs in wake.
 *   * Drag clasp or feather blades: continuously folds/opens the 7-blade shingled fan.
 *   * Click blade: plucks individual cantilever flex + structural flash ripple.
 *   * Click clasp (or double-click): smoothly toggles Open Display <-> Furled Stack.
 *   * Drag background: full 3D OrbitControls to inspect stepped clasp & z-layer depth.
 */

// Pre-allocated scratch math objects (zero allocations inside rAF)
const _raycaster = new THREE.Raycaster();
const _ndcScratch = new THREE.Vector2();
const _hitPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const _planeNormal = new THREE.Vector3();
const _dragCurWorld = new THREE.Vector3();
const _dragStartWorld = new THREE.Vector3();
const _invBladeMat = new THREE.Matrix4();
const _localPtrScratch = new THREE.Vector3();
const _localVelScratch = new THREE.Vector3();
const _bladeAxisLocal = new THREE.Vector3(0, 1, 0);
const _bladeAxisWorld = new THREE.Vector3();
const _worldOrigin = new THREE.Vector3(0, 0, 0);

interface BladeRig {
  index: number;
  pivotGroup: THREE.Group;
  slideGroup: THREE.Group;
  rachisMesh: THREE.Mesh;
  barbsMesh: THREE.Mesh;
  claspParts: THREE.Mesh[];
  uniforms: BladeOpticsUniforms;
  openPos: number;
  openVel: number;
  prevAngleRad: number;
  prevSlideY: number;
  flexZ: number;
  flexVelZ: number;
  twist: number;
  twistVel: number;
  ripplePhase: number;
  rippleAmp: number;
  prevLocalPtrValid: boolean;
  prevLocalPtr: THREE.Vector3;
}

declare global {
  interface Window {
    __FEATHER_FAN__?: {
      setFanOpen: (v: number) => void;
      setBeamAngles: (azRad: number, elRad: number) => void;
      pluckBlade: (idx: number, impulse?: number) => void;
      getFanState: () => {
        fanOpen: number;
        fanOpenTarget: number;
        beamDir: [number, number, number];
        flexZ: number[];
      };
    };
  }
}

export async function bootDay12Feather(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(12);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.24, 4.95],
    fov: 34,
    enableShadows: true,
  });

  stage.controls.target.set(0.0, 0.24, 0.0);
  stage.controls.minDistance = 1.75;
  stage.controls.maxDistance = 8.2;
  stage.controls.minPolarAngle = 0.28;
  stage.controls.maxPolarAngle = Math.PI * 0.78;
  stage.controls.update();

  // Locate SpecimenStage's shadow-casting studio key light so it steers with the goniometer beam
  let studioKeyLight: THREE.DirectionalLight | null = null;
  stage.scene.traverse((obj) => {
    if (!studioKeyLight && obj instanceof THREE.DirectionalLight && obj.castShadow) {
      studioKeyLight = obj;
    }
  });
  if (studioKeyLight) {
    const kl = studioKeyLight as THREE.DirectionalLight;
    kl.intensity = 2.48;
    kl.color.setHex(0xf7ecd8);
    kl.shadow.mapSize.set(2048, 2048);
    kl.shadow.camera.left = -2.25;
    kl.shadow.camera.right = 2.25;
    kl.shadow.camera.top = 2.25;
    kl.shadow.camera.bottom = -2.25;
    kl.shadow.bias = -0.0004;
    kl.shadow.normalBias = 0.008;
    kl.shadow.camera.updateProjectionMatrix();
  }

  // Shared optics uniforms & physical mineral pigments
  const pigments = createFeatherPigmentSet();
  const initialBeamDir = new THREE.Vector3(0.76, 0.48, 0.44).normalize();
  const uGonioBeamDir = uniform(initialBeamDir.clone());
  const uGonioBeamScreen = uniform(new THREE.Vector2(0.76, 0.56));
  const uBeamIntensity = uniform(1.0);

  const sharedUniforms: SharedFeatherOpticsUniforms = {
    uTime: stage.uTime,
    uPointerMeta: stage.uPointerMeta,
    uGonioBeamDir,
    uGonioBeamScreen,
    uBeamIntensity,
  };

  // Subtle Chalky Slate studio backdrop with directional beam wash tied to steered light
  stage.scene.backgroundNode = createGoniometricBackdropNode(uGonioBeamScreen);

  // Root articulated fan group
  const fanRoot = new THREE.Group();
  fanRoot.name = 'GoniometricPeacockFan';
  fanRoot.position.set(0, PIVOT_Y, 0);
  stage.specimenGroup.add(fanRoot);

  // Shared clasp materials & pivot pin
  const bronzeMat = createBronzeGuardMaterial(pigments);
  const boneMat = createBoneFerruleMaterial(pigments);

  const claspSpanZ = (BLADE_COUNT - 1) * Z_LAYER_STEP;
  const pivotPinMesh = new THREE.Mesh(createPivotPinGeometry(claspSpanZ), boneMat);
  pivotPinMesh.castShadow = true;
  pivotPinMesh.receiveShadow = false;
  pivotPinMesh.name = 'ClaspPivotPin';
  fanRoot.add(pivotPinMesh);

  const outerGuardGeom = createBronzeGuardLeafGeometry(true);
  const innerGuardGeom = createBronzeGuardLeafGeometry(false);
  const outerFerruleGeom = createBoneFerruleGeometry(0.017, 0.024, 0.105);
  const innerFerruleGeom = createBoneFerruleGeometry(0.0155, 0.022, 0.088);

  const blades: BladeRig[] = [];
  const interactiveMeshes: THREE.Object3D[] = [pivotPinMesh];
  const claspHitSet = new Set<THREE.Object3D>([pivotPinMesh]);
  const meshToBladeIdx = new Map<THREE.Object3D, number>();

  for (let i = 0; i < BLADE_COUNT; i++) {
    const featherLen = BLADE_LENGTHS[i];
    const zOffset = (i - (BLADE_COUNT - 1) * 0.5) * Z_LAYER_STEP;

    const pivotGroup = new THREE.Group();
    pivotGroup.position.set(0, 0, zOffset);
    pivotGroup.rotation.z = -BLADE_OPEN_ANGLES_RAD[i];
    fanRoot.add(pivotGroup);

    const slideGroup = new THREE.Group();
    pivotGroup.add(slideGroup);

    const bladeUniforms: BladeOpticsUniforms = {
      uBladeFlex: uniform(new THREE.Vector4(0, 0, 0, 0)),
      uPointerLocal: uniform(new THREE.Vector3(0, -10, 0)),
      uPointerVelLocal: uniform(new THREE.Vector3(0, 0, 0)),
      uFeatherLen: uniform(featherLen),
      uSlideY: uniform(0),
      uBladeAxisWorld: uniform(new THREE.Vector3(0, 1, 0)),
    };

    // Bronze stepped leaf guard + bone ferrule sleeve
    const isOuter = i === 0 || i === BLADE_COUNT - 1;
    const guardMesh = new THREE.Mesh(isOuter ? outerGuardGeom : innerGuardGeom, bronzeMat);
    guardMesh.castShadow = true;
    guardMesh.receiveShadow = true;
    pivotGroup.add(guardMesh);

    const ferruleMesh = new THREE.Mesh(isOuter ? outerFerruleGeom : innerFerruleGeom, boneMat);
    ferruleMesh.position.set(0, isOuter ? 0.224 : 0.168, 0);
    ferruleMesh.castShadow = true;
    ferruleMesh.receiveShadow = false;
    pivotGroup.add(ferruleMesh);

    // Tapered 3D Rachis & 356 discrete 3D curved barb strands
    const rachisMesh = new THREE.Mesh(
      createRachisGeometry(i, featherLen),
      createRachisMaterial(pigments, bladeUniforms, sharedUniforms)
    );
    rachisMesh.castShadow = true;
    rachisMesh.receiveShadow = false;
    slideGroup.add(rachisMesh);

    const barbsMesh = new THREE.Mesh(
      createFeatherBarbsGeometry(i, featherLen),
      createFeatherBarbMaterial(i, pigments, bladeUniforms, sharedUniforms)
    );
    barbsMesh.castShadow = true;
    barbsMesh.receiveShadow = true;
    slideGroup.add(barbsMesh);

    const claspParts = [guardMesh, ferruleMesh];
    for (const cp of claspParts) {
      claspHitSet.add(cp);
      interactiveMeshes.push(cp);
      meshToBladeIdx.set(cp, i);
    }
    interactiveMeshes.push(rachisMesh, barbsMesh);
    meshToBladeIdx.set(rachisMesh, i);
    meshToBladeIdx.set(barbsMesh, i);

    blades.push({
      index: i,
      pivotGroup,
      slideGroup,
      rachisMesh,
      barbsMesh,
      claspParts,
      uniforms: bladeUniforms,
      openPos: 1.0,
      openVel: 0.0,
      prevAngleRad: BLADE_OPEN_ANGLES_RAD[i],
      prevSlideY: 0.0,
      flexZ: 0,
      flexVelZ: 0,
      twist: 0,
      twistVel: 0,
      ripplePhase: i * 0.65,
      rippleAmp: 0,
      prevLocalPtrValid: false,
      prevLocalPtr: new THREE.Vector3(0, -10, 0),
    });
  }

  // -------------------------------------------------------------------------
  // Interactive State & Smooth Ergonomic Gesture Routing
  // -------------------------------------------------------------------------
  let fanOpen = 1.0;
  let fanOpenTarget = 1.0;

  // Goniometer incident beam spherical coordinates (azimuth from +Z toward +X, elevation above XZ)
  let beamAz = 0.98; // Upper-right default matching Shot 01
  let beamEl = 0.50;
  let targetBeamAz = 0.98;
  let targetBeamEl = 0.50;
  let pointerSteerActive = false;

  let dragActive = false;
  let dragIsClaspClick = false;
  let dragStartFanOpen = 1.0;
  let dragSignX = 1.0;
  let dragStartClientX = 0;
  let dragStartClientY = 0;
  let dragMaxTravelPx = 0;
  let dragHitBladeIdx = -1;
  let lastClaspClickMs = -1000;

  const canvas = stage.shell.canvas;

  const updateNdcFromClient = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    _ndcScratch.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
  };

  const projectToFocalPlane = (outVec: THREE.Vector3) => {
    stage.camera.getWorldDirection(_planeNormal).negate();
    _hitPlane.setFromNormalAndCoplanarPoint(_planeNormal, _worldOrigin);
    _raycaster.setFromCamera(_ndcScratch, stage.camera);
    _raycaster.ray.intersectPlane(_hitPlane, outVec);
  };

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    updateNdcFromClient(ev.clientX, ev.clientY);
    _raycaster.setFromCamera(_ndcScratch, stage.camera);

    const hits = _raycaster.intersectObjects(interactiveMeshes, false);
    if (hits.length === 0) {
      // Empty space -> allow full 3D OrbitControls (Shot 04)
      dragActive = false;
      stage.controls.enabled = true;
      return;
    }

    const hitObj = hits[0].object;
    dragActive = true;
    dragIsClaspClick = claspHitSet.has(hitObj);
    dragHitBladeIdx = meshToBladeIdx.get(hitObj) ?? -1;
    dragStartFanOpen = fanOpenTarget;
    dragStartClientX = ev.clientX;
    dragStartClientY = ev.clientY;
    dragMaxTravelPx = 0;

    projectToFocalPlane(_dragStartWorld);
    // Lock continuous drag polarity once at pointerdown (never flips across centerline!)
    dragSignX = Math.tanh(_dragStartWorld.x * 2.6);

    // Disable OrbitControls cleanly during specimen blade/clasp drag
    stage.controls.enabled = false;
  };

  const onPointerMove = (ev: PointerEvent) => {
    updateNdcFromClient(ev.clientX, ev.clientY);
    pointerSteerActive = true;

    // Steer goniometer beam smoothly across [-1.18, +1.18] rad azimuth & [-0.32, +0.86] rad elevation
    targetBeamAz = _ndcScratch.x * 1.18;
    targetBeamEl = _ndcScratch.y * 0.54 + 0.24;

    if (!dragActive) return;

    const travel = Math.hypot(ev.clientX - dragStartClientX, ev.clientY - dragStartClientY);
    if (travel > dragMaxTravelPx) dragMaxTravelPx = travel;

    projectToFocalPlane(_dragCurWorld);
    const dx = _dragCurWorld.x - _dragStartWorld.x;
    const dy = _dragCurWorld.y - _dragStartWorld.y;

    // Continuous radial + lift projection: drag outward opens, drag inward furls smoothly
    const dragDelta = (dx * dragSignX + dy * 0.55) * 0.72;
    fanOpenTarget = Math.max(0.0, Math.min(1.0, dragStartFanOpen + dragDelta));
  };

  const triggerToggleKineticWave = (opening: boolean) => {
    const dir = opening ? 1.0 : -1.0;
    for (let i = 0; i < BLADE_COUNT; i++) {
      const b = blades[i];
      const offsetFromCenter = i - 3;
      const distNorm = Math.abs(offsetFromCenter) / 3.0;
      const sideSign = offsetFromCenter === 0 ? 0.35 : Math.sign(offsetFromCenter);
      // Instant elastic release (mild lead velocity without slamming end-stops) + crisp feather wave
      b.openVel += dir * (0.28 + (i / (BLADE_COUNT - 1)) * 0.48);
      b.flexVelZ += dir * (-0.58 - distNorm * 0.55) + (i % 2 === 0 ? 0.24 : -0.18);
      b.twistVel += dir * sideSign * 1.35;
      b.rippleAmp = Math.min(0.068, b.rippleAmp + 0.024 + distNorm * 0.016);
    }
  };

  const onPointerUp = () => {
    if (dragActive) {
      if (dragMaxTravelPx < 6.0) {
        if (dragIsClaspClick) {
          // Clicking the clasp toggles Open <-> Furled with kinetic cascade wave
          lastClaspClickMs = performance.now();
          const nextOpen = fanOpenTarget > 0.5 ? 0.0 : 1.0;
          fanOpenTarget = nextOpen;
          triggerToggleKineticWave(nextOpen > 0.5);
        } else if (dragHitBladeIdx >= 0 && dragHitBladeIdx < BLADE_COUNT) {
          // Clicking an individual blade plucks a crisp cantilever bow & copper-emerald flash (Shot 05)
          const b = blades[dragHitBladeIdx];
          b.flexVelZ += 2.85;
          b.twistVel += (dragHitBladeIdx >= 3 ? -1 : 1) * 2.1;
          b.rippleAmp = Math.min(0.068, b.rippleAmp + 0.048);
        }
      } else {
        // Gentle magnetic snap only near endpoints; otherwise honor release position
        if (fanOpenTarget < 0.06) fanOpenTarget = 0.0;
        else if (fanOpenTarget > 0.94) fanOpenTarget = 1.0;
      }
    }
    dragActive = false;
    stage.controls.enabled = true;
  };

  const onDoubleClick = () => {
    if (performance.now() - lastClaspClickMs < 480) return;
    const nextOpen = fanOpenTarget > 0.5 ? 0.0 : 1.0;
    fanOpenTarget = nextOpen;
    triggerToggleKineticWave(nextOpen > 0.5);
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });
  canvas.addEventListener('dblclick', onDoubleClick);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('dblclick', onDoubleClick);
  });

  // Expose deterministic inspection API for storyboard / bench verification
  window.__FEATHER_FAN__ = {
    setFanOpen: (v: number) => {
      fanOpenTarget = Math.max(0.0, Math.min(1.0, v));
      fanOpen = fanOpenTarget;
      for (let i = 0; i < BLADE_COUNT; i++) {
        const b = blades[i];
        b.openPos = fanOpenTarget;
        b.openVel = 0;
        b.prevAngleRad =
          BLADE_FURLED_ANGLES_RAD[i] +
          (BLADE_OPEN_ANGLES_RAD[i] - BLADE_FURLED_ANGLES_RAD[i]) * fanOpenTarget;
        b.prevSlideY = BLADE_FURLED_SLIDES[i] * (1.0 - fanOpenTarget);
      }
    },
    setBeamAngles: (azRad: number, elRad: number) => {
      pointerSteerActive = false;
      targetBeamAz = azRad;
      targetBeamEl = elRad;
      beamAz = azRad;
      beamEl = elRad;
    },
    pluckBlade: (idx: number, impulse = 2.8) => {
      const b = blades[Math.max(0, Math.min(BLADE_COUNT - 1, idx))];
      b.flexVelZ += impulse;
      b.twistVel += (idx >= 3 ? -1 : 1) * 1.9;
      b.rippleAmp = 0.055;
    },
    getFanState: () => ({
      fanOpen: Number(fanOpen.toFixed(4)),
      fanOpenTarget: Number(fanOpenTarget.toFixed(4)),
      beamDir: [
        Number(uGonioBeamDir.value.x.toFixed(4)),
        Number(uGonioBeamDir.value.y.toFixed(4)),
        Number(uGonioBeamDir.value.z.toFixed(4)),
      ],
      flexZ: blades.map((b) => Number(b.flexZ.toFixed(4))),
    }),
  };

  // Seed a subtle initial wake ripple on outer right blades on boot so frame 1 looks alive
  blades[5].flexZ = 0.015;
  blades[6].flexZ = 0.01;

  // -------------------------------------------------------------------------
  // Main 60-120 Hz Animation & Spring Physics Loop (Zero Heap Allocations)
  // -------------------------------------------------------------------------
  stage.onTick(({ dtSec, elapsedSec, pointer }) => {
    const dt = Math.min(0.033, Math.max(0.001, dtSec));

    // 1. Steer goniometer light direction & backdrop shaft
    if (!pointerSteerActive) {
      // Gentle idle optical breathing if pointer hasn't entered canvas yet
      targetBeamAz = 0.98 + Math.sin(elapsedSec * 0.42) * 0.06;
      targetBeamEl = 0.50 + Math.cos(elapsedSec * 0.31) * 0.04;
    }
    const beamEma = 1.0 - Math.exp(-dt * 8.5);
    beamAz += (targetBeamAz - beamAz) * beamEma;
    beamEl += (targetBeamEl - beamEl) * beamEma;

    const cosEl = Math.cos(beamEl);
    const bx = Math.sin(beamAz) * cosEl;
    const by = Math.sin(beamEl);
    const bz = Math.max(0.28, Math.cos(beamAz) * cosEl);
    uGonioBeamDir.value.set(bx, by, bz).normalize();
    uGonioBeamScreen.value.set(bx, by);

    if (studioKeyLight) {
      const kl = studioKeyLight as THREE.DirectionalLight;
      kl.position.copy(uGonioBeamDir.value).multiplyScalar(5.8);
    }

    // 2. Per-blade staggered fan spring, elastic end-stop rebound & aerodynamic whip (zero stall!)
    let openSum = 0.0;
    for (let i = 0; i < BLADE_COUNT; i++) {
      const b = blades[i];
      const offsetNorm = Math.abs(i - 3) / 3.0;
      const cardOrder = (BLADE_COUNT - 1 - i) / (BLADE_COUNT - 1); // front blade 6 leads, back blade 0 follows smoothly

      // During active drag follow cursor responsively; on release let underdamped cascade springs oscillate freely
      const bladeK = dragActive ? 145.0 - cardOrder * 18.0 : 92.0 - cardOrder * 26.0;
      const bladeDamp = dragActive ? 16.5 : 6.2 + offsetNorm * 0.45;
      const bladeAcc = (fanOpenTarget - b.openPos) * bladeK - b.openVel * bladeDamp;
      b.openVel += bladeAcc * dt;
      b.openPos += b.openVel * dt;

      // Soft cushioned clasp stop at 0 + unconstrained open aerodynamic flare (safety cap 1.18)
      if (b.openPos < -0.018) {
        b.openPos = -0.018;
        if (b.openVel < 0) {
          const impact = -b.openVel;
          b.openVel = impact * 0.20;
          // Channel clasp-closing momentum into out-of-plane feather bow & barb shiver
          b.flexVelZ += Math.min(2.6, impact * 0.58);
          b.rippleAmp = Math.min(0.068, b.rippleAmp + Math.min(0.028, impact * 0.007));
        }
      } else if (b.openPos > 1.18) {
        b.openPos = 1.18;
        if (b.openVel > 0) {
          const impact = b.openVel;
          b.openVel = -impact * 0.35;
          b.flexVelZ -= Math.min(2.0, impact * 0.42);
          b.rippleAmp = Math.min(0.068, b.rippleAmp + Math.min(0.024, impact * 0.005));
        }
      }
      openSum += b.openPos;

      const targetAngle =
        BLADE_FURLED_ANGLES_RAD[i] +
        (BLADE_OPEN_ANGLES_RAD[i] - BLADE_FURLED_ANGLES_RAD[i]) * b.openPos;
      b.pivotGroup.rotation.z = -targetAngle;

      // Shingled axial staircase slide when furled (Shot 02)
      const clampedSlideFrac = Math.max(0.0, Math.min(1.03, 1.0 - b.openPos));
      const slideY = BLADE_FURLED_SLIDES[i] * clampedSlideFrac;
      b.slideGroup.position.y = slideY;
      b.uniforms.uSlideY.value = slideY;

      // Continuous aerodynamic drag & whip from angular sweep + axial slide velocity
      const angVel = (targetAngle - b.prevAngleRad) / dt;
      const slideVel = (slideY - b.prevSlideY) / dt;
      b.prevAngleRad = targetAngle;
      b.prevSlideY = slideY;

      const sweepMag = Math.abs(angVel) + Math.abs(slideVel) * 0.65;
      if (sweepMag > 0.03) {
        const dirSign = i === 3 ? 0.35 : Math.sign(i - 3);
        const leverage = 0.42 + 0.58 * offsetNorm;
        b.flexVelZ += (-angVel * dirSign * leverage * 0.58 - slideVel * 0.28) * dt * 9.0;
        b.twistVel += (-angVel * leverage * 0.72) * dt * 8.0;
        b.rippleAmp = Math.min(0.068, b.rippleAmp + Math.min(0.012, sweepMag * 0.0038) * dt * 15.0);
      }
    }

    // Derive root vertical centering continuously from live blade positions (zero out-of-sync clamp pauses)
    fanOpen = openSum / BLADE_COUNT;
    fanRoot.position.y = PIVOT_Y - 0.14 * (1.0 - Math.max(0.0, Math.min(1.0, fanOpen)));
    fanRoot.updateMatrixWorld(true);

    // 3. Per-blade pointer wake coupling & progressive-stiffness keratin spring physics
    for (let i = 0; i < BLADE_COUNT; i++) {
      const b = blades[i];
      b.slideGroup.updateMatrixWorld(true);

      // Transform focal-plane pointer into this blade's local coordinate system
      _invBladeMat.copy(b.slideGroup.matrixWorld).invert();
      _localPtrScratch.copy(pointer.worldPos).applyMatrix4(_invBladeMat);

      if (pointer.isActive) {
        if (!b.prevLocalPtrValid) {
          b.prevLocalPtr.copy(_localPtrScratch);
          b.prevLocalPtrValid = true;
        }
        _localVelScratch.subVectors(_localPtrScratch, b.prevLocalPtr).divideScalar(dt);
        b.prevLocalPtr.copy(_localPtrScratch);

        // Inject subtle aerodynamic breeze impulse into blade spring when cursor sweeps across blade
        const inBladeSpanY = _localPtrScratch.y > 0.25 && _localPtrScratch.y < BLADE_LENGTHS[i] + 0.22;
        const distFromShaftX = Math.abs(_localPtrScratch.x);
        if (inBladeSpanY && distFromShaftX < 0.34 && pointer.speed > 0.25) {
          const prox = Math.exp(-distFromShaftX * distFromShaftX * 18.0);
          const push = Math.min(2.4, pointer.speed * 0.42) * prox * dt * 16.0;
          b.flexVelZ += push;
          b.twistVel += -Math.sign(_localVelScratch.x || 1) * push * 0.45;
          b.rippleAmp = Math.min(0.055, b.rippleAmp + push * 0.012);
        }
      } else {
        _localPtrScratch.set(0, -10, 0);
        _localVelScratch.set(0, 0, 0);
      }

      // Advance progressive-stiffness keratin cantilever spring + torsional twist (no flat wall clamp!)
      const kFlex = 58.0 * (1.0 + 9.0 * b.flexZ * b.flexZ);
      const cFlex = 3.5;
      b.flexVelZ += (-b.flexZ * kFlex - b.flexVelZ * cFlex) * dt;
      b.flexZ += b.flexVelZ * dt;
      if (b.flexZ > 0.44) {
        b.flexZ = 0.44;
        if (b.flexVelZ > 0) b.flexVelZ = -b.flexVelZ * 0.5;
      } else if (b.flexZ < -0.44) {
        b.flexZ = -0.44;
        if (b.flexVelZ < 0) b.flexVelZ = -b.flexVelZ * 0.5;
      }

      b.twistVel += (-b.twist * 60.0 - b.twistVel * 3.8) * dt;
      b.twist = Math.max(-0.56, Math.min(0.56, b.twist + b.twistVel * dt));

      b.ripplePhase += dt * 15.0;
      b.rippleAmp *= Math.exp(-dt * 2.2);
      if (b.rippleAmp < 1e-4) b.rippleAmp = 0;

      // Update pre-allocated TSL blade uniforms
      b.uniforms.uBladeFlex.value.set(b.flexZ, b.twist, b.ripplePhase, b.rippleAmp);
      b.uniforms.uPointerLocal.value.copy(_localPtrScratch);
      b.uniforms.uPointerVelLocal.value.copy(_localVelScratch);

      // Update world-space shaft axis vector for physical goniometer angle evaluation
      _bladeAxisWorld
        .copy(_bladeAxisLocal)
        .transformDirection(b.slideGroup.matrixWorld)
        .normalize();
      b.uniforms.uBladeAxisWorld.value.copy(_bladeAxisWorld);
    }
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay12Feather();
}
