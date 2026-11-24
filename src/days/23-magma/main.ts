import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  CHAMBER_BOTTOM_Y,
  CHAMBER_ROOF_Y,
  CORE_HALF_D,
  DIKE_SEAM_X,
  MAX_LOBES,
  OVERBURDEN_TOP_Y,
  SILL_BED_Y,
  SLAB_COLUMNS,
  TectonicMagmaSolver,
  type Day23Telemetry,
} from './tectonicSolver';
import { createMagmaStudioEnvironment } from './studioEnv';
import { createMagmaMaterials } from './materials';
import {
  EMBER_COUNT,
  buildHydraulicPistonAssembly,
  buildLateralClampAssemblies,
  buildMagmaSystemAssembly,
  buildPressFrameAssembly,
  buildRockCoreAssembly,
} from './geometry';

declare global {
  interface Window {
    __DAY23__?: {
      getState: () => Day23Telemetry;
    };
  }
}

type HitKind = 'none' | 'piston' | 'chamber' | 'core' | 'leftClamp' | 'rightClamp';

interface EmberParticle {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  age: number;
  life: number;
  heat: number;
  scale: number;
  active: boolean;
}

// Pre-allocated scratch math objects (strictly zero per-frame allocations inside onTick)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _pos = new THREE.Vector3();
const _scale = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _euler = new THREE.Euler(0, 0, 0, 'YXZ');
const _mat4 = new THREE.Matrix4();
const _worldOriginNdc = new THREE.Vector3();
const _worldPlusXNdc = new THREE.Vector3();

const XENOLITH_BASE: ReadonlyArray<readonly [number, number, number, number]> = [
  [-0.36, CHAMBER_ROOF_Y - 0.14, 0.24, 0.7],
  [0.28, CHAMBER_ROOF_Y - 0.18, 0.18, 1.9],
  [-0.08, CHAMBER_ROOF_Y - 0.21, -0.22, 3.1],
  [0.42, CHAMBER_ROOF_Y - 0.13, -0.18, 4.4],
];

async function mountDay23Magma(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 23,
    cameraPosition: [2.16, 1.30, 4.74],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, -0.14, 0);
  stage.controls.minDistance = 2.15;
  stage.controls.maxDistance = 9.0;
  stage.controls.minPolarAngle = 0.20;
  stage.controls.maxPolarAngle = Math.PI * 0.86;
  stage.controls.update();

  // Photographer's studio softbox PMREM IBL for dark machined steel, satin brass, quartz & obsidian
  const studioEnv = createMagmaStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.18;

  // Harmonize SpecimenStage's 3-point studio rig + tight 2048x2048 shadow frustum
  let dirIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirIdx === 0) {
        child.position.set(-2.6, 5.0, 3.6);
        child.intensity = 1.38;
        child.shadow.mapSize.set(2048, 2048);
        const d = 2.45;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 0.8;
        child.shadow.camera.far = 13.0;
        child.shadow.bias = -0.00024;
        child.shadow.normalBias = 0.012;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirIdx === 1) {
        child.position.set(3.4, 1.6, 2.2);
        child.intensity = 0.42;
      } else if (dirIdx === 2) {
        child.position.set(-1.2, 3.6, -4.0);
        child.intensity = 1.35;
      }
      dirIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.40;
    }
  }

  // Two restrained warm volcanic point lights (interior chamber glow + active intrusion crest glow)
  const chamberGlowLight = new THREE.PointLight(0xe06d2b, 1.35, 3.6, 1.8);
  chamberGlowLight.position.set(0, -0.42, 0.72);
  stage.specimenGroup.add(chamberGlowLight);

  const crestGlowLight = new THREE.PointLight(0xc93b2b, 0.75, 2.8, 2.0);
  crestGlowLight.position.set(0, 0.48, 0.45);
  stage.specimenGroup.add(crestGlowLight);

  // Core geomechanics solver & GPU scene assemblies
  const sim = new TectonicMagmaSolver({
    initialPressure: 0.58,
    initialCompression: -0.34,
    preWarmSec: 2.6,
  });
  const materials = createMagmaMaterials();

  const pressFrame = buildPressFrameAssembly(materials);
  const piston = buildHydraulicPistonAssembly(materials);
  const lateralClamps = buildLateralClampAssemblies(materials);
  const rockCore = buildRockCoreAssembly(materials);
  const magmaSystem = buildMagmaSystemAssembly(materials);

  stage.specimenGroup.add(pressFrame.group);
  stage.specimenGroup.add(piston.group);
  stage.specimenGroup.add(lateralClamps.group);
  stage.specimenGroup.add(rockCore.group);
  stage.specimenGroup.add(magmaSystem.group);

  // Pre-allocated restrained volcanic ember pool for active crest eruptions
  const embers: EmberParticle[] = [];
  for (let i = 0; i < EMBER_COUNT; i++) {
    embers.push({
      x: 0,
      y: OVERBURDEN_TOP_Y,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      age: 0,
      life: 1,
      heat: 0,
      scale: 0,
      active: false,
    });
  }
  let emberCursor = 0;
  let emberSpawnAcc = 0;

  // -------------------------------------------------------------------------
  // Tactile Pointer & Keyboard Interaction Controller
  // -------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  let activeDrag: HitKind = 'none';
  let hoveredPart: HitKind = 'none';
  let downClientX = 0;
  let downClientY = 0;
  let prevClientX = 0;
  let prevClientY = 0;
  let hasDraggedBeyondDeadzone = false;
  let lastPointerDownMs = 0;
  let holdPumpTimerId: number | null = null;

  const hitTargets: Array<{ mesh: THREE.Object3D; kind: HitKind }> = [
    { mesh: piston.hitMesh, kind: 'piston' },
    { mesh: lateralClamps.leftHitMesh, kind: 'leftClamp' },
    { mesh: lateralClamps.rightHitMesh, kind: 'rightClamp' },
    { mesh: pressFrame.chamberHitMesh, kind: 'chamber' },
    { mesh: rockCore.coreHitMesh, kind: 'core' },
  ];

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastSpecimen(clientX: number, clientY: number): HitKind {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    let bestDist = Infinity;
    let bestKind: HitKind = 'none';

    for (let i = 0; i < hitTargets.length; i++) {
      const target = hitTargets[i];
      const hits = _raycaster.intersectObject(target.mesh, false);
      if (hits.length > 0 && hits[0].distance < bestDist) {
        bestDist = hits[0].distance;
        bestKind = target.kind;
      }
    }
    return bestKind;
  }

  /**
   * Computes camera-relative screen-space direction of world +X so dragging either
   * lateral clamp inward toward the core ALWAYS increases compression from any 360° orbit angle.
   */
  function getScreenXAxisSign(): number {
    _worldOriginNdc.set(0, 0.2, 0).project(stage.camera);
    _worldPlusXNdc.set(1, 0.2, 0).project(stage.camera);
    const dx = _worldPlusXNdc.x - _worldOriginNdc.x;
    return dx >= 0 ? 1 : -1;
  }

  function clearHoldTimer(): void {
    if (holdPumpTimerId !== null) {
      window.clearTimeout(holdPumpTimerId);
      holdPumpTimerId = null;
    }
  }

  function releasePointerCaptureSafe(pointerId: number): void {
    try {
      canvas.releasePointerCapture(pointerId);
    } catch {
      // Ignore if already released
    }
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;

    const nowMs = performance.now();
    const isDoubleClick = nowMs - lastPointerDownMs < 320;
    lastPointerDownMs = nowMs;

    if (isDoubleClick) {
      clearHoldTimer();
      sim.quenchAndReset(false);
      return;
    }

    const hit = raycastSpecimen(ev.clientX, ev.clientY);
    activeDrag = hit;
    downClientX = ev.clientX;
    downClientY = ev.clientY;
    prevClientX = ev.clientX;
    prevClientY = ev.clientY;
    hasDraggedBeyondDeadzone = false;

    if (hit !== 'none') {
      stage.controls.enabled = false;
      canvas.setPointerCapture(ev.pointerId);

      if (hit === 'piston' || hit === 'chamber' || hit === 'core') {
        canvas.style.cursor = 'grabbing';
        // Holding click on the chamber/piston pumps hydraulic magma pressure upward
        clearHoldTimer();
        holdPumpTimerId = window.setTimeout(() => {
          if (
            !hasDraggedBeyondDeadzone &&
            (activeDrag === 'piston' || activeDrag === 'chamber' || activeDrag === 'core')
          ) {
            sim.setHoldPump(true, 0.44);
          }
        }, 120);
      } else {
        canvas.style.cursor = 'ew-resize';
      }
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (activeDrag !== 'none') {
      const dx = ev.clientX - prevClientX;
      const dy = ev.clientY - prevClientY;
      prevClientX = ev.clientX;
      prevClientY = ev.clientY;

      const totalMove = Math.hypot(ev.clientX - downClientX, ev.clientY - downClientY);
      if (!hasDraggedBeyondDeadzone && totalMove >= 7) {
        hasDraggedBeyondDeadzone = true;
        clearHoldTimer();
        sim.setHoldPump(false);
      }

      if (activeDrag === 'piston' || activeDrag === 'chamber' || activeDrag === 'core') {
        // Dragging vertically upward pumps hydraulic magma pressure upward; downward relieves
        sim.nudgePressure(-dy / 220);
      } else if (activeDrag === 'leftClamp' || activeDrag === 'rightClamp') {
        // Dragging inward toward the center increases compression; outward opens extensional tension
        const camSign = getScreenXAxisSign();
        const sideSign = activeDrag === 'leftClamp' ? 1 : -1;
        const deltaCompression = (dx * camSign * sideSign) / 170;
        sim.nudgeCompression(deltaCompression);
      }
      return;
    }

    if (ev.buttons !== 0) return;

    const hit = raycastSpecimen(ev.clientX, ev.clientY);
    hoveredPart = hit;
    if (hit === 'piston' || hit === 'chamber' || hit === 'core') {
      canvas.style.cursor = 'ns-resize';
    } else if (hit === 'leftClamp' || hit === 'rightClamp') {
      canvas.style.cursor = 'ew-resize';
    } else {
      canvas.style.cursor = 'grab';
    }
  };

  const finishDrag = (ev: PointerEvent) => {
    clearHoldTimer();
    sim.setHoldPump(false);
    if (activeDrag !== 'none') {
      releasePointerCaptureSafe(ev.pointerId);
      stage.controls.enabled = true;
      activeDrag = 'none';
      canvas.style.cursor = 'grab';
    }
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      sim.quenchAndReset(false);
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', finishDrag);
  canvas.addEventListener('pointercancel', finishDrag);
  window.addEventListener('keydown', onKeyDown);

  // Read-only QA & verification telemetry hook
  window.__DAY23__ = {
    getState: () => sim.getState(),
  };

  // Dispose everything cleanly on route navigation / pagehide
  stage.shell.registerDispose(() => {
    clearHoldTimer();
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', finishDrag);
    canvas.removeEventListener('pointercancel', finishDrag);
    window.removeEventListener('keydown', onKeyDown);
    stage.scene.environment = null;
    studioEnv.dispose();
    pressFrame.dispose();
    piston.dispose();
    lateralClamps.dispose();
    rockCore.dispose();
    magmaSystem.dispose();
    materials.dispose();
    delete window.__DAY23__;
  });

  // -------------------------------------------------------------------------
  // Main Real-Time Simulation & GPU Sync Loop (Strictly Zero Allocations)
  // -------------------------------------------------------------------------
  stage.onTick(({ dtSec, elapsedSec }) => {
    sim.step(dtSec);

    // 1. Update Hydraulic Piston Vertical Stroke & Chamber Melt Volume
    piston.ramGroup.position.y = sim.pistonTravel;

    const plungerTopY = CHAMBER_BOTTOM_Y + 0.08 + sim.pistonTravel;
    const meltBottomY = Math.min(CHAMBER_ROOF_Y - 0.08, plungerTopY - 0.015);
    const meltHeight = Math.max(0.08, CHAMBER_ROOF_Y - meltBottomY);
    const baseMeltH = CHAMBER_ROOF_Y - CHAMBER_BOTTOM_Y - 0.025;
    magmaSystem.chamberMeltMesh.scale.y = meltHeight / baseMeltH;
    magmaSystem.chamberMeltMesh.position.y = meltBottomY + meltHeight * 0.5;

    // 2. Update Drifting Basalt Xenoliths Suspended in Lower Chamber
    for (let i = 0; i < magmaSystem.xenolithMeshes.length; i++) {
      const xm = magmaSystem.xenolithMeshes[i];
      const base = XENOLITH_BASE[i];
      const phase = base[3];
      const bob = Math.sin(elapsedSec * (0.55 + 0.15 * i) + phase) * 0.024;
      const driftX = Math.cos(elapsedSec * 0.32 + phase) * 0.032;
      const minY = plungerTopY + 0.065;
      const maxY = CHAMBER_ROOF_Y - 0.065;
      const clampedY = Math.max(minY, Math.min(maxY, base[1] + sim.pistonTravel * 0.35 + bob));
      xm.position.set(base[0] + driftX, clampedY, base[2]);
      xm.rotation.set(
        elapsedSec * 0.14 + phase,
        elapsedSec * 0.18 + phase * 1.3,
        elapsedSec * 0.10
      );
    }

    // 3. Update Left & Right Lateral Threaded Tectonic Screw-Clamps & Capstan Handwheels
    lateralClamps.leftPlatenGroup.position.x = -sim.platenOffset;
    lateralClamps.leftScrewWheelGroup.position.x = -sim.platenOffset;
    lateralClamps.leftScrewWheelGroup.rotation.x = -sim.clampScrewAngle;

    lateralClamps.rightPlatenGroup.position.x = sim.platenOffset;
    lateralClamps.rightScrewWheelGroup.position.x = sim.platenOffset;
    lateralClamps.rightScrewWheelGroup.rotation.x = sim.clampScrewAngle;

    // 4. Update Diegetic Upper Brass Heave Gauge Rods
    for (let i = 0; i < pressFrame.heaveGaugeRods.length; i++) {
      pressFrame.heaveGaugeRods[i].position.y = 1.04 + sim.heaveDisplacement * 0.55;
    }

    // 5. Update 12 Interlocking Stratified Basalt & Ochre-Tuff Rock Slabs
    let tier0AvgDy = 0;
    let tier1AvgDy = 0;
    let tier2AvgDy = 0;

    for (let i = 0; i < sim.slabs.length; i++) {
      const slab = sim.slabs[i];
      const mesh = rockCore.slabMeshes[i];
      mesh.position.set(
        slab.restCenterX + slab.dx,
        slab.restCenterY + slab.dy,
        slab.dz
      );
      mesh.rotation.set(slab.rotX, 0, slab.rotZ);

      if (slab.tier === 0) tier0AvgDy += slab.dy;
      else if (slab.tier === 1) tier1AvgDy += slab.dy;
      else tier2AvgDy += slab.dy;
    }
    tier0AvgDy /= SLAB_COLUMNS;
    tier1AvgDy /= SLAB_COLUMNS;
    tier2AvgDy /= SLAB_COLUMNS;

    // 6. Update 3 Dynamic Vertical Magma Dike Blades (positioned directly inside opening seams)
    for (let i = 0; i < sim.dikes.length; i++) {
      const d = sim.dikes[i];
      const mesh = magmaSystem.dikeMeshes[i];
      const visibleH = d.height;
      const veinThickness = Math.max(d.aperture * 1.18, d.quenchedObsidian * 0.018);

      if (visibleH > 0.015 && veinThickness > 0.004) {
        mesh.visible = true;
        const leftCol = i === 0 ? 1 : i === 1 ? 0 : 2;
        const rightCol = leftCol + 1;
        const sL = sim.slabs[SLAB_COLUMNS + leftCol];
        const sR = sim.slabs[SLAB_COLUMNS + rightCol];
        const seamMidDx = 0.5 * (sL.dx + sR.dx);
        const crestLift = 0.5 * (sim.slabs[2 * SLAB_COLUMNS + leftCol].dy + sim.slabs[2 * SLAB_COLUMNS + rightCol].dy);

        mesh.position.set(DIKE_SEAM_X[i] + seamMidDx, CHAMBER_ROOF_Y - 0.02, 0);
        mesh.scale.set(
          Math.max(0.012, veinThickness),
          visibleH + crestLift * 0.72 + 0.025,
          1.0
        );
      } else {
        mesh.visible = false;
      }
    }

    // 7. Update 2 Dynamic Horizontal Laccolith Sill Lenses (riding flush inside bedding gaps)
    for (let i = 0; i < sim.sills.length; i++) {
      const s = sim.sills[i];
      const mesh = magmaSystem.sillMeshes[i];
      const sillThick = Math.max(s.thickness * 1.16, s.quenchedObsidian * 0.014);

      if (s.halfWidth > 0.03 && sillThick > 0.004) {
        mesh.visible = true;
        const bedDy = i === 0 ? 0.5 * (tier0AvgDy + tier1AvgDy) : 0.5 * (tier1AvgDy + tier2AvgDy);
        mesh.position.set(0, SILL_BED_Y[i] + bedDy, 0);
        mesh.scale.set(s.halfWidth, sillThick, 1.0);
      } else {
        mesh.visible = false;
      }
    }

    // 8. Update 24 Instanced Viscoplastic Pahoehoe Magma Lobes (1 draw call, zero alloc)
    const thermalArr = magmaSystem.lobeThermalAttr.array as Float32Array;
    for (let i = 0; i < MAX_LOBES; i++) {
      const lobe = sim.lobes[i];
      if (lobe.active && lobe.growth > 0.01) {
        const g = lobe.growth;
        const swell = 0.35 + 0.65 * g;
        _pos.set(lobe.posX, lobe.posY, lobe.posZ);
        _euler.set(lobe.pitch, lobe.yaw, 0, 'YXZ');
        _quat.setFromEuler(_euler);
        _scale.set(lobe.radiusX * swell, lobe.radiusY * swell, lobe.radiusZ * swell);
        _mat4.compose(_pos, _quat, _scale);
        magmaSystem.lobeInstancedMesh.setMatrixAt(i, _mat4);

        const base = i * 4;
        thermalArr[base] = lobe.coreTemp;
        thermalArr[base + 1] = lobe.crustTemp;
        thermalArr[base + 2] = lobe.crackIntensity;
        thermalArr[base + 3] = lobe.seed;
      } else {
        _scale.set(0, 0, 0);
        _pos.set(0, -10, 0);
        _quat.identity();
        _mat4.compose(_pos, _quat, _scale);
        magmaSystem.lobeInstancedMesh.setMatrixAt(i, _mat4);
        const base = i * 4;
        thermalArr[base] = 0;
        thermalArr[base + 1] = 0;
        thermalArr[base + 2] = 0;
        thermalArr[base + 3] = 0;
      }
    }
    magmaSystem.lobeInstancedMesh.instanceMatrix.needsUpdate = true;
    magmaSystem.lobeThermalAttr.needsUpdate = true;

    // 9. Update Restrained Rising Volcanic Ash / Spatter Ember Motes During Active Eruptions
    const crestErupting = sim.dikes[0].breachedCrest || sim.dikes[1].breachedCrest || sim.dikes[2].breachedCrest;
    if (crestErupting && sim.chamberPressure > 0.42) {
      emberSpawnAcc += dtSec * (7.0 + sim.chamberPressure * 10.0);
      while (emberSpawnAcc >= 1.0) {
        emberSpawnAcc -= 1.0;
        const ep = embers[emberCursor];
        emberCursor = (emberCursor + 1) % EMBER_COUNT;
        const dikeIdx = sim.dikes[0].breachedCrest ? 0 : sim.dikes[1].breachedCrest ? 1 : 2;
        const h1 = Math.sin(elapsedSec * 73.1 + emberCursor * 19.3) * 43758.5453;
        const r1 = h1 - Math.floor(h1);
        const h2 = Math.sin(elapsedSec * 41.7 + emberCursor * 11.7) * 29183.173;
        const r2 = h2 - Math.floor(h2);
        const h3 = Math.sin(elapsedSec * 91.3 + emberCursor * 5.3) * 18421.719;
        const r3 = h3 - Math.floor(h3);

        ep.active = true;
        ep.age = 0;
        ep.life = 0.85 + r1 * 0.95;
        ep.x = DIKE_SEAM_X[dikeIdx] + (r1 - 0.5) * 0.14;
        ep.y = OVERBURDEN_TOP_Y + sim.heaveDisplacement + 0.03;
        ep.z = (r2 - 0.5) * (CORE_HALF_D * 1.45);
        ep.vx = (r1 - 0.5) * 0.24;
        ep.vy = 0.34 + r3 * 0.38;
        ep.vz = (r2 - 0.5) * 0.22;
        ep.heat = 0.72 + r3 * 0.26;
        ep.scale = 0.55 + r1 * 0.85;
      }
    }

    const emberHeatArr = magmaSystem.emberHeatAttr.array as Float32Array;
    for (let i = 0; i < EMBER_COUNT; i++) {
      const ep = embers[i];
      if (ep.active) {
        ep.age += dtSec;
        if (ep.age >= ep.life) {
          ep.active = false;
          ep.heat = 0;
        } else {
          ep.vy -= 0.16 * dtSec;
          ep.x += ep.vx * dtSec;
          ep.y += ep.vy * dtSec;
          ep.z += ep.vz * dtSec;
          const u = ep.age / ep.life;
          const curHeat = ep.heat * Math.max(0, 1 - u * 1.15);
          const sc = ep.scale * (1 - u * 0.55);
          _pos.set(ep.x, ep.y, ep.z);
          _euler.set(ep.age * 3.1, ep.age * 4.2, 0, 'YXZ');
          _quat.setFromEuler(_euler);
          _scale.set(sc, sc, sc);
          _mat4.compose(_pos, _quat, _scale);
          magmaSystem.emberInstancedMesh.setMatrixAt(i, _mat4);
          emberHeatArr[i] = curHeat;
          continue;
        }
      }
      _pos.set(0, -10, 0);
      _quat.identity();
      _scale.set(0, 0, 0);
      _mat4.compose(_pos, _quat, _scale);
      magmaSystem.emberInstancedMesh.setMatrixAt(i, _mat4);
      emberHeatArr[i] = 0;
    }
    magmaSystem.emberInstancedMesh.instanceMatrix.needsUpdate = true;
    magmaSystem.emberHeatAttr.needsUpdate = true;

    // 10. Sync Shared TSL Uniforms & Volcanic Point Lights
    const d0 = sim.dikes[0];
    const d1 = sim.dikes[1];
    const d2 = sim.dikes[2];
    const s0 = sim.sills[0];
    const s1 = sim.sills[1];

    const maxDikeTemp = Math.max(d0.temperature, d1.temperature, d2.temperature);
    const maxSillTemp = Math.max(s0.temperature, s1.temperature);
    const maxHeat = Math.max(maxDikeTemp, maxSillTemp, sim.chamberPressure * 0.72);

    materials.uTime.value = elapsedSec;
    materials.uChamberState.value.set(
      sim.chamberPressure,
      sim.tectonicCompression,
      maxHeat,
      sim.pressureSurge
    );
    materials.uDikeHeights.value.set(d0.height, d1.height, d2.height, maxDikeTemp);
    materials.uDikeThermal.value.set(
      Math.max(d0.coreTemp, d1.coreTemp, d2.coreTemp),
      maxDikeTemp,
      Math.max(d0.reCrackPulse, d1.reCrackPulse, d2.reCrackPulse),
      Math.max(d0.quenchedObsidian, d1.quenchedObsidian, d2.quenchedObsidian)
    );
    materials.uSillState.value.set(
      s0.halfWidth,
      s1.halfWidth,
      maxSillTemp,
      Math.max(s0.reCrackPulse, s1.reCrackPulse)
    );

    const hAlpha = Math.min(1, dtSec * 14);
    const pistonHoverGoal = hoveredPart === 'piston' || activeDrag === 'piston' ? 1 : 0;
    const leftHoverGoal = hoveredPart === 'leftClamp' || activeDrag === 'leftClamp' ? 1 : 0;
    const rightHoverGoal = hoveredPart === 'rightClamp' || activeDrag === 'rightClamp' ? 1 : 0;
    const coreHoverGoal = hoveredPart === 'core' || hoveredPart === 'chamber' ? 1 : 0;

    materials.uHoverState.value.x += (pistonHoverGoal - materials.uHoverState.value.x) * hAlpha;
    materials.uHoverState.value.y += (leftHoverGoal - materials.uHoverState.value.y) * hAlpha;
    materials.uHoverState.value.z += (rightHoverGoal - materials.uHoverState.value.z) * hAlpha;
    materials.uHoverState.value.w += (coreHoverGoal - materials.uHoverState.value.w) * hAlpha;

    chamberGlowLight.intensity = 0.75 + sim.chamberPressure * 1.45;
    crestGlowLight.position.y = Math.min(0.88, CHAMBER_ROOF_Y + sim.activeDikeHeight * 0.85 + 0.14);
    crestGlowLight.intensity = 0.25 + maxHeat * 1.25;
  });
}

mountDay23Magma().catch((err) => {
  console.error('[Day 23 - Magma] Failed to initialize WebGPU specimen:', err);
});
