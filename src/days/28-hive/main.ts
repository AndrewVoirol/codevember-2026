import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  MagneticDipoleLattice,
  NEEDLE_PIVOT_Y,
  buildHexLatticeLayout,
} from './magneticLattice';
import {
  BEZEL_BOLT_RADIUS,
  BEZEL_INNER_CIRCUM_R,
  BEZEL_TOP_Y,
  SLAB_OUTER_CIRCUM_R,
  buildBezelBoltGeometry,
  buildBrassInnerTrimGeometry,
  buildBronzeBezelGeometry,
  buildHoneycombMatrixGeometry,
  buildLimestoneSlabGeometry,
  buildLodestoneStylusGeometries,
  buildNeedleHalfGeometry,
  buildPivotPinGeometry,
  buildUndersideMedallionGeometry,
} from './hiveGeometry';
import {
  createHiveMaterials,
  createHiveStudioEnvironment,
} from './hiveMaterials';

declare global {
  interface Window {
    __HIVE__?: {
      getState: () => Record<string, unknown>;
      triggerPulse: (x?: number, z?: number) => void;
      triggerThermalScatter: (intensity?: number) => void;
      flipPolarity: () => 1 | -1;
      seedTwinVortex: () => void;
      seedMacroVortex: (handedness?: 1 | -1) => void;
      setCameraPreset: (preset: 'hero' | 'top' | 'macro' | 'underside' | 'rear') => void;
    };
  }
}

// Pre-allocated zero-GC scratch objects
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _deckPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -NEEDLE_PIVOT_Y);
const _hitPoint = new THREE.Vector3();
const _dummyMat = new THREE.Matrix4();
const _dummyPos = new THREE.Vector3();
const _dummyQuat = new THREE.Quaternion();
const _dummyEuler = new THREE.Euler(0, 0, 0, 'YXZ');
const _dummyScale = new THREE.Vector3(1, 1, 1);

const IDLE_STYLUS_X = 0.78;
const IDLE_STYLUS_Z = -0.68;
const IDLE_STYLUS_Y = 0.21;

async function mountDay28Hive(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 28,
    cameraPosition: [0.0, 3.68, 3.92],
    fov: 34,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, -0.04, 0.02);
  stage.controls.minDistance = 1.65;
  stage.controls.maxDistance = 8.4;
  stage.controls.minPolarAngle = 0.08;
  stage.controls.maxPolarAngle = Math.PI - 0.12;
  stage.controls.update();

  // Studio softbox PMREM environment for satin brass, dark bronze & blued steel
  const studioEnv = createHiveStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize directional studio lights & upgrade shadow map to 2048x2048 tightly fitted to slab
  let dirLightIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirLightIdx === 0) {
        child.position.set(-2.6, 5.4, 3.1);
        child.intensity = 1.48;
        child.shadow.mapSize.set(2048, 2048);
        const d = 2.05;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.0;
        child.shadow.bias = -0.00025;
        child.shadow.normalBias = 0.011;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirLightIdx === 1) {
        child.position.set(3.8, 2.2, 2.4);
        child.intensity = 0.42;
      } else if (dirLightIdx === 2) {
        child.position.set(0.4, 3.6, -4.0);
        child.intensity = 1.22;
      }
      dirLightIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.44;
    }
  }

  // Build physics lattice, geometries & materials
  const layout = buildHexLatticeLayout();
  const lattice = new MagneticDipoleLattice(layout, 202628);
  const materials = createHiveMaterials();

  // Root instrument assembly
  const instrumentGroup = new THREE.Group();
  instrumentGroup.name = 'MagneticHoneycombInstrument';
  stage.specimenGroup.add(instrumentGroup);

  // 1. Chamfered Honed Bone-Calcite Limestone Base Slab
  const limestoneGeo = buildLimestoneSlabGeometry();
  const limestoneMesh = new THREE.Mesh(limestoneGeo, materials.limestone);
  limestoneMesh.castShadow = true;
  limestoneMesh.receiveShadow = true;
  instrumentGroup.add(limestoneMesh);

  // 2. Darkened-Bronze Bolted Bezel Frame
  const bezelGeo = buildBronzeBezelGeometry();
  const bezelMesh = new THREE.Mesh(bezelGeo, materials.darkBronze);
  bezelMesh.castShadow = true;
  bezelMesh.receiveShadow = true;
  instrumentGroup.add(bezelMesh);

  // 3. Satin-Brass Inner Trim Pinstripe
  const innerTrimGeo = buildBrassInnerTrimGeometry();
  const innerTrimMesh = new THREE.Mesh(innerTrimGeo, materials.satinBrass);
  innerTrimMesh.castShadow = true;
  innerTrimMesh.receiveShadow = true;
  instrumentGroup.add(innerTrimMesh);

  // 4. Six Machined Satin-Brass Bezel Corner Rivet Bolts (Instanced)
  const boltGeo = buildBezelBoltGeometry();
  const boltsMesh = new THREE.InstancedMesh(boltGeo, materials.satinBrass, 6);
  boltsMesh.castShadow = true;
  boltsMesh.receiveShadow = true;
  for (let k = 0; k < 6; k++) {
    const ang = -k * (Math.PI / 3);
    _dummyPos.set(
      Math.cos(ang) * BEZEL_BOLT_RADIUS,
      BEZEL_TOP_Y - 0.002,
      Math.sin(ang) * BEZEL_BOLT_RADIUS
    );
    _dummyQuat.identity();
    _dummyScale.set(1, 1, 1);
    _dummyMat.compose(_dummyPos, _dummyQuat, _dummyScale);
    boltsMesh.setMatrixAt(k, _dummyMat);
  }
  boltsMesh.instanceMatrix.needsUpdate = true;
  instrumentGroup.add(boltsMesh);

  // 5. Watertight 91-Cell Satin-Brass Honeycomb Matrix
  const honeycombGeo = buildHoneycombMatrixGeometry(layout);
  const honeycombMesh = new THREE.Mesh(honeycombGeo, materials.satinBrass);
  honeycombMesh.castShadow = true;
  honeycombMesh.receiveShadow = true;
  instrumentGroup.add(honeycombMesh);

  // 6. 91 Turned Brass Pivot Pins & Crowning Jewel Caps (Static Instanced)
  const pivotGeo = buildPivotPinGeometry();
  const pivotsMesh = new THREE.InstancedMesh(pivotGeo, materials.satinBrass, layout.cellCount);
  pivotsMesh.castShadow = true;
  pivotsMesh.receiveShadow = true;
  for (let i = 0; i < layout.cellCount; i++) {
    const cell = layout.cells[i];
    _dummyPos.set(cell.x, 0, cell.z);
    _dummyQuat.identity();
    _dummyScale.set(1, 1, 1);
    _dummyMat.compose(_dummyPos, _dummyQuat, _dummyScale);
    pivotsMesh.setMatrixAt(i, _dummyMat);
  }
  pivotsMesh.instanceMatrix.needsUpdate = true;
  instrumentGroup.add(pivotsMesh);

  // 7. 91 Faceted Magnetic Compass Needles (Synchronized North Cinnabar + South Prussian InstancedMeshes)
  const northGeo = buildNeedleHalfGeometry(1);
  const southGeo = buildNeedleHalfGeometry(-1);
  const northMesh = new THREE.InstancedMesh(
    northGeo,
    materials.needleNorthCinnabar,
    layout.cellCount
  );
  const southMesh = new THREE.InstancedMesh(
    southGeo,
    materials.needleSouthPrussian,
    layout.cellCount
  );
  northMesh.castShadow = true;
  northMesh.receiveShadow = true;
  southMesh.castShadow = true;
  southMesh.receiveShadow = true;
  instrumentGroup.add(northMesh);
  instrumentGroup.add(southMesh);

  // 8. Underside Machined Dark-Bronze Medallion & Brass Compass Ring
  const undersideGeos = buildUndersideMedallionGeometry();
  const undersidePlateMesh = new THREE.Mesh(undersideGeos.bronzePlate, materials.darkBronze);
  const undersideRingMesh = new THREE.Mesh(undersideGeos.brassRing, materials.satinBrass);
  undersidePlateMesh.receiveShadow = true;
  undersideRingMesh.receiveShadow = true;
  instrumentGroup.add(undersidePlateMesh);
  instrumentGroup.add(undersideRingMesh);

  // 9. Hovering Turned Satin-Brass & Dark Magnetite Lodestone Stylus Wand
  const stylusGeos = buildLodestoneStylusGeometries();
  const stylusGroup = new THREE.Group();
  stylusGroup.name = 'LodestoneStylusWand';
  const tipMesh = new THREE.Mesh(stylusGeos.lodestoneTip, materials.lodestone);
  const bandMesh = new THREE.Mesh(stylusGeos.polarityBand, materials.polarityBand);
  const bodyMesh = new THREE.Mesh(stylusGeos.brassBody, materials.satinBrass);
  const gripMesh = new THREE.Mesh(stylusGeos.darkGrip, materials.darkGrip);
  const pommelMesh = new THREE.Mesh(stylusGeos.brassPommel, materials.satinBrass);
  for (const m of [tipMesh, bandMesh, bodyMesh, gripMesh, pommelMesh]) {
    m.castShadow = true;
    m.receiveShadow = true;
    stylusGroup.add(m);
  }
  stylusGroup.position.set(IDLE_STYLUS_X, IDLE_STYLUS_Y, IDLE_STYLUS_Z);
  stylusGroup.rotation.order = 'YXZ';
  stylusGroup.rotation.set(-0.34, 0.25, -0.68);
  instrumentGroup.add(stylusGroup);

  // Helper: sync all 91 compass needle instance matrices from lattice state (zero allocations)
  const syncNeedles = () => {
    const cells = layout.cells;
    const angles = lattice.angles;
    const wp = lattice.wobblePitch;
    const wr = lattice.wobbleRoll;
    _dummyScale.set(1, 1, 1);

    for (let i = 0; i < layout.cellCount; i++) {
      const c = cells[i];
      _dummyPos.set(c.x, NEEDLE_PIVOT_Y, c.z);
      _dummyEuler.set(wr[i], angles[i], wp[i], 'YXZ');
      _dummyQuat.setFromEuler(_dummyEuler);
      _dummyMat.compose(_dummyPos, _dummyQuat, _dummyScale);
      northMesh.setMatrixAt(i, _dummyMat);
      southMesh.setMatrixAt(i, _dummyMat);
    }
    northMesh.instanceMatrix.needsUpdate = true;
    southMesh.instanceMatrix.needsUpdate = true;
  };
  syncNeedles();

  // ---------------------------------------------------------------------------
  // Interactive Pointer, Lodestone Drag & Orbit Ergonomics
  // ---------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  const bootTimeMs = performance.now();
  let hoverArmed = false;
  let lastRawClientX = -1;
  let lastRawClientY = -1;
  let accumulatedMovePx = 0;

  let pointerInsideHoneycomb = false;
  let pointerDraggingLattice = false;
  let pointerDownOnRim = false;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTimeMs = 0;
  let lastClickTimeMs = 0;

  // Smooth stylus kinematics
  let stylusTargetX = IDLE_STYLUS_X;
  let stylusTargetZ = IDLE_STYLUS_Z;
  let stylusTargetY = IDLE_STYLUS_Y;
  let stylusCurX = IDLE_STYLUS_X;
  let stylusCurZ = IDLE_STYLUS_Z;
  let stylusCurY = IDLE_STYLUS_Y;
  let stylusVelX = 0;
  let stylusVelZ = 0;
  let pulseDip = 0;
  let polarityFlipSpin = 0;
  let polarityBlendCur = 0;
  let engagedCur = 0;

  // Compute flat-top hexagon inradius test at local XZ
  const isInsideFlatTopHex = (x: number, z: number, circumR: number): boolean => {
    const inR = circumR * (Math.sqrt(3) * 0.5);
    const ax = Math.abs(x);
    const az = Math.abs(z);
    if (az > inR) return false;
    // Flat-top hex 6-plane boundary check
    return az <= inR && ax * 0.8660254 + az * 0.5 <= inR;
  };

  const raycastDeckPlane = (clientX: number, clientY: number): boolean => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    _ndc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -(((clientY - rect.top) / rect.height) * 2 - 1)
    );
    _raycaster.setFromCamera(_ndc, stage.camera);
    return _raycaster.ray.intersectPlane(_deckPlane, _hitPoint) !== null;
  };

  const onPointerMove = (e: PointerEvent) => {
    // Boot-settle cursor guard: prevent stationary parked OS cursors or macOS activation bursts
    // from hijacking the opening twin-vortex equilibrium before intentional travel
    if (!hoverArmed) {
      if (lastRawClientX >= 0) {
        accumulatedMovePx += Math.hypot(e.clientX - lastRawClientX, e.clientY - lastRawClientY);
      }
      lastRawClientX = e.clientX;
      lastRawClientY = e.clientY;
      if (performance.now() - bootTimeMs >= 320 && accumulatedMovePx >= 8) {
        hoverArmed = true;
      } else if (!pointerDraggingLattice) {
        return;
      }
    }

    if (!raycastDeckPlane(e.clientX, e.clientY)) {
      if (!pointerDraggingLattice) {
        pointerInsideHoneycomb = false;
        canvas.style.cursor = 'grab';
      }
      return;
    }

    const inComb = isInsideFlatTopHex(_hitPoint.x, _hitPoint.z, BEZEL_INNER_CIRCUM_R - 0.02);
    const inSlab = isInsideFlatTopHex(_hitPoint.x, _hitPoint.z, SLAB_OUTER_CIRCUM_R);

    if (pointerDraggingLattice) {
      // Clamp stylus smoothly inside the bronze bezel perimeter while sculpting
      const r = Math.hypot(_hitPoint.x, _hitPoint.z);
      const maxR = BEZEL_INNER_CIRCUM_R * 0.84;
      const clampScale = r > maxR ? maxR / r : 1.0;
      stylusTargetX = _hitPoint.x * clampScale;
      stylusTargetZ = _hitPoint.z * clampScale;
      stylusTargetY = 0.135;
      pointerInsideHoneycomb = true;
      canvas.style.cursor = 'crosshair';
    } else if (inComb) {
      stylusTargetX = _hitPoint.x;
      stylusTargetZ = _hitPoint.z;
      stylusTargetY = 0.22;
      pointerInsideHoneycomb = true;
      canvas.style.cursor = 'crosshair';
    } else {
      pointerInsideHoneycomb = false;
      stylusTargetX = IDLE_STYLUS_X;
      stylusTargetZ = IDLE_STYLUS_Z;
      stylusTargetY = IDLE_STYLUS_Y;
      canvas.style.cursor = inSlab ? 'pointer' : 'grab';
    }
  };

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    hoverArmed = true;
    pointerDownX = e.clientX;
    pointerDownY = e.clientY;
    pointerDownTimeMs = performance.now();
    pointerDownOnRim = false;

    if (!raycastDeckPlane(e.clientX, e.clientY)) return;

    const inComb = isInsideFlatTopHex(_hitPoint.x, _hitPoint.z, BEZEL_INNER_CIRCUM_R - 0.02);
    const inSlab = isInsideFlatTopHex(_hitPoint.x, _hitPoint.z, SLAB_OUTER_CIRCUM_R);

    if (inComb) {
      pointerDraggingLattice = true;
      pointerInsideHoneycomb = true;
      stage.controls.enabled = false;
      stylusTargetX = _hitPoint.x;
      stylusTargetZ = _hitPoint.z;
      stylusTargetY = 0.132;
      canvas.style.cursor = 'crosshair';
    } else if (inSlab) {
      pointerDownOnRim = true;
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    const moveDist = Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTimeMs;
    const isQuickClick = moveDist < 9 && heldMs < 360;

    if (pointerDraggingLattice) {
      pointerDraggingLattice = false;
      stage.controls.enabled = true;
      stylusTargetY = pointerInsideHoneycomb ? 0.22 : IDLE_STYLUS_Y;

      if (isQuickClick) {
        const now = performance.now();
        if (now - lastClickTimeMs < 330) {
          // Double-click inside comb -> demagnetizing thermal scatter so hive re-organizes freshly
          lattice.triggerThermalScatter(1.0);
          pulseDip = 1.0;
          lastClickTimeMs = 0;
        } else {
          // Single quick click -> radial swirling shock pulse at clicked cell
          lattice.triggerPulse(stylusTargetX, stylusTargetZ, lattice.stylus.polarity, 1.05, 0.65);
          pulseDip = 0.85;
          lastClickTimeMs = now;
        }
      }
    } else if (pointerDownOnRim && isQuickClick) {
      // Quick click on bronze bezel or limestone rim flips stylus polarity with baton flourish
      lattice.flipStylusPolarity();
      polarityFlipSpin = Math.PI * 2;
      pointerDownOnRim = false;
    } else {
      pointerDownOnRim = false;
      // Double-click empty space -> thermal scatter fresh self-organization
      if (isQuickClick) {
        const now = performance.now();
        if (now - lastClickTimeMs < 330) {
          lattice.triggerThermalScatter(1.0);
          lastClickTimeMs = 0;
        } else {
          lastClickTimeMs = now;
        }
      }
    }
  };

  const onPointerLeave = () => {
    if (!pointerDraggingLattice) {
      pointerInsideHoneycomb = false;
      stylusTargetX = IDLE_STYLUS_X;
      stylusTargetZ = IDLE_STYLUS_Z;
      stylusTargetY = IDLE_STYLUS_Y;
    }
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.code === 'Space') {
      e.preventDefault();
      lattice.triggerThermalScatter(1.0);
      pulseDip = 0.9;
    } else if (e.key === 'f' || e.key === 'F') {
      lattice.flipStylusPolarity();
      polarityFlipSpin = Math.PI * 2;
    } else if (e.key === 'r' || e.key === 'R') {
      lattice.seedTwinVortexEquilibrium();
    }
  };

  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerLeave);
  window.addEventListener('keydown', onKeyDown);

  // Expose non-invasive live diagnostic & camera preset hook
  window.__HIVE__ = {
    getState: () => ({
      ...lattice.getTelemetry(),
      stylusPos: [
        Number(stylusCurX.toFixed(3)),
        Number(stylusCurY.toFixed(3)),
        Number(stylusCurZ.toFixed(3)),
      ],
      stylusActive: lattice.stylus.active,
      draggingLattice: pointerDraggingLattice,
    }),
    triggerPulse: (x = 0, z = 0) => {
      lattice.triggerPulse(x, z, lattice.stylus.polarity, 1.1, 0.65);
      pulseDip = 0.9;
    },
    triggerThermalScatter: (intensity = 1.0) => {
      lattice.triggerThermalScatter(intensity);
    },
    flipPolarity: () => {
      polarityFlipSpin = Math.PI * 2;
      return lattice.flipStylusPolarity();
    },
    seedTwinVortex: () => {
      lattice.seedTwinVortexEquilibrium();
    },
    seedMacroVortex: (handedness = 1) => {
      lattice.seedMacroVortex(handedness);
    },
    setCameraPreset: (preset) => {
      if (preset === 'hero') {
        stage.camera.position.set(0.0, 3.68, 3.92);
        stage.controls.target.set(0, -0.04, 0.02);
      } else if (preset === 'top') {
        stage.camera.position.set(0.0, 4.45, 0.35);
        stage.controls.target.set(0, 0, 0);
      } else if (preset === 'macro') {
        stage.camera.position.set(0.55, 1.12, 1.18);
        stage.controls.target.set(0.12, 0.04, -0.04);
      } else if (preset === 'underside') {
        stage.camera.position.set(0.85, -2.55, 2.35);
        stage.controls.target.set(0, -0.15, 0);
      } else if (preset === 'rear') {
        stage.camera.position.set(-0.35, 2.75, -3.55);
        stage.controls.target.set(0, -0.04, 0);
      }
      stage.controls.update();
    },
  };

  // ---------------------------------------------------------------------------
  // Main Animation Loop (Strictly Zero Per-Frame Allocations)
  // ---------------------------------------------------------------------------
  const unsubTick = stage.onTick(({ elapsedSec, dtSec }) => {
    const smoothK = 1 - Math.exp(-15.0 * dtSec);
    const prevX = stylusCurX;
    const prevZ = stylusCurZ;

    stylusCurX += (stylusTargetX - stylusCurX) * smoothK;
    stylusCurZ += (stylusTargetZ - stylusCurZ) * smoothK;
    stylusCurY += (stylusTargetY - stylusCurY) * smoothK;

    const rawVx = (stylusCurX - prevX) / Math.max(0.004, dtSec);
    const rawVz = (stylusCurZ - prevZ) / Math.max(0.004, dtSec);
    const velSmooth = 1 - Math.exp(-12.0 * dtSec);
    stylusVelX += (rawVx - stylusVelX) * velSmooth;
    stylusVelZ += (rawVz - stylusVelZ) * velSmooth;

    // Decay click dip & polarity baton flourish
    pulseDip *= Math.exp(-9.5 * dtSec);
    polarityFlipSpin *= Math.exp(-7.5 * dtSec);

    const targetPolarityBlend = lattice.stylus.polarity === 1 ? 0.0 : 1.0;
    polarityBlendCur += (targetPolarityBlend - polarityBlendCur) * (1 - Math.exp(-10.0 * dtSec));
    materials.uPolarityBlend.value = polarityBlendCur;

    const targetEngaged = pointerDraggingLattice ? 1.0 : pointerInsideHoneycomb ? 0.48 : 0.08;
    engagedCur += (targetEngaged - engagedCur) * (1 - Math.exp(-10.0 * dtSec));
    materials.uStylusEngaged.value = engagedCur;

    // Gentle idle hover float when parked at upper-right rest position
    const idleFloat = pointerInsideHoneycomb || pointerDraggingLattice ? 0.0 : Math.sin(elapsedSec * 1.8) * 0.014;
    const dipOffset = pulseDip * 0.095;
    const tipY = Math.max(NEEDLE_PIVOT_Y + 0.065, stylusCurY + idleFloat - dipOffset);

    stylusGroup.position.set(stylusCurX, tipY, stylusCurZ);
    // Natural handheld 38° slant leaning out toward upper-right + subtle inertia lag during sweep
    const tiltLagX = Math.max(-0.22, Math.min(0.22, -stylusVelZ * 0.055));
    const tiltLagZ = Math.max(-0.22, Math.min(0.22, stylusVelX * 0.055));
    stylusGroup.rotation.set(
      -0.28 + tiltLagX + polarityFlipSpin,
      0.22,
      -0.78 + tiltLagZ
    );

    materials.uStylusLocalPos.value.set(stylusCurX, tipY, stylusCurZ);

    // Feed live 3D wand kinematics into coupled magnetic dipole physics
    // At idle parked flank, keep gentle localized magnetic influence active so nearby corner
    // cells subtly acknowledge the hovering lodestone just like the concept render!
    lattice.stylus.active = true;
    lattice.stylus.x = stylusCurX;
    lattice.stylus.z = stylusCurZ;
    lattice.stylus.height = Math.max(0.12, tipY - NEEDLE_PIVOT_Y);
    lattice.stylus.strength = pointerDraggingLattice
      ? 2.85
      : pointerInsideHoneycomb
        ? 1.15
        : 0.34;
    lattice.stylus.vx = pointerDraggingLattice ? stylusVelX * 1.45 : pointerInsideHoneycomb ? stylusVelX * 0.55 : 0;
    lattice.stylus.vz = pointerDraggingLattice ? stylusVelZ * 1.45 : pointerInsideHoneycomb ? stylusVelZ * 0.55 : 0;

    // Step coupled 91-dipole physics & sync instanced needle matrices
    lattice.step(dtSec);
    syncNeedles();
  });

  // Clean disposal on page unload
  stage.shell.registerDispose(() => {
    unsubTick();
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    window.removeEventListener('keydown', onKeyDown);

    limestoneGeo.dispose();
    bezelGeo.dispose();
    innerTrimGeo.dispose();
    boltGeo.dispose();
    honeycombGeo.dispose();
    pivotGeo.dispose();
    northGeo.dispose();
    southGeo.dispose();
    undersideGeos.bronzePlate.dispose();
    undersideGeos.brassRing.dispose();
    stylusGeos.dispose();
    materials.dispose();
    studioEnv.dispose();
  });
}

void mountDay28Hive();
