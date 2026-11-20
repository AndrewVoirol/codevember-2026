import * as THREE from 'three/webgpu';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  DOME_CENTER,
  DOME_RADIUS,
  MAX_STONES,
  STONE_RADIUS,
  STONE_SEAT_OFFSET,
  DomePendulumSimulation,
  type MagneticStone,
  type PulleyKinematicsFrame,
} from './physics';
import {
  createHammeredIronSurfaceMaps,
  createStudioEnvironment,
} from './domeSurface';
import { MAX_IMPACT_RIPPLES, createPendulumMaterials } from './materials';
import {
  OrbitalWakeRibbon,
  TautPlumbWire,
  buildDomeAssembly,
  buildGimbalPulleyAssembly,
  buildPendulumBob,
  createStoneGeometryVariants,
} from './geometry';

declare global {
  interface Window {
    __PENDULUM__?: {
      getState: () => Record<string, unknown>;
      tossStone: (x: number, y: number, z: number) => void;
      castHandful: (count?: number) => void;
      flipStone: (id: number) => boolean;
    };
  }
}

interface StoneVisualSlot {
  mesh: THREE.Mesh;
  stoneId: number;
  variant: number;
}

// Pre-allocated scratch objects (strictly zero per-frame allocations)
const _raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _grabPlane = new THREE.Plane();
const _grabHit = new THREE.Vector3();
const _camDir = new THREE.Vector3();
const _launchOrigin = new THREE.Vector3();
const _stoneNormal = new THREE.Vector3();
const _pulleyFrame: PulleyKinematicsFrame = {
  azimuth: 0,
  wheelAngle: 0,
  bobTangentWorld: new THREE.Vector3(),
  bobColletWorld: new THREE.Vector3(),
  cwTangentWorld: new THREE.Vector3(),
  cwEyeletWorld: new THREE.Vector3(),
  departAngleLocal: 0,
  wireLength: 1.5,
  pluckPhase: 0,
  pluckAmp: 0,
};

async function mountDay18Pendulum(): Promise<void> {
  const stage = await SpecimenStage.create({
    day: 18,
    cameraPosition: [0, 0.52, 4.92],
    fov: 34,
    enableOrbit: true,
    enableShadows: true,
  });

  stage.controls.target.set(0, 0.28, 0);
  stage.controls.minDistance = 1.65;
  stage.controls.maxDistance = 8.0;
  stage.controls.maxPolarAngle = Math.PI * 0.84;
  stage.controls.minPolarAngle = 0.22;
  stage.controls.update();

  // Calibrate broadly diffused Studio Softbox PMREM Environment for Day 18 forged iron, brass & bronze
  const studioEnv = createStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 1.15;

  // Harmonize Day 18's directional rig & upgrade shadow map to 2048x2048 tightly fitted to specimen
  let dirLightIdx = 0;
  for (const child of stage.scene.children) {
    if (child instanceof THREE.DirectionalLight) {
      if (dirLightIdx === 0) {
        child.position.set(-2.9, 5.2, 3.6);
        child.intensity = 1.12;
        child.shadow.mapSize.set(2048, 2048);
        const d = 1.9;
        child.shadow.camera.left = -d;
        child.shadow.camera.right = d;
        child.shadow.camera.top = d;
        child.shadow.camera.bottom = -d;
        child.shadow.camera.near = 1.0;
        child.shadow.camera.far = 12.5;
        child.shadow.bias = -0.00022;
        child.shadow.normalBias = 0.012;
        child.shadow.camera.updateProjectionMatrix();
      } else if (dirLightIdx === 1) {
        child.position.set(-3.6, 1.4, 2.6);
        child.intensity = 0.22;
      } else if (dirLightIdx === 2) {
        child.intensity = 1.24;
      }
      dirLightIdx++;
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.36;
    }
  }

  const sim = new DomePendulumSimulation();
  const surfaceMaps = createHammeredIronSurfaceMaps();
  const materials = createPendulumMaterials(surfaceMaps);

  // 1. Suspended Hand-Planished Hammered-Iron Dome & Slim Bone Calcite Collar
  const dome = buildDomeAssembly(materials);
  stage.specimenGroup.add(dome.group);

  // 2. Overhead Horological Gimbal Horseshoe, Swivel Differential Pulley & Brass Counterweight
  const gimbalPulley = buildGimbalPulleyAssembly(materials);
  gimbalPulley.counterweightGroup.rotation.order = 'YXZ';
  stage.specimenGroup.add(gimbalPulley.group);
  stage.specimenGroup.add(gimbalPulley.counterweightGroup);

  // 3. Elongated 8-Facet Bone-Calcite & Satin-Brass Pendulum Bob
  const bob = buildPendulumBob(materials);
  stage.specimenGroup.add(bob.group);

  // 4. Razor-Taut 3-Segment Drawn Brass Plumb Wire (pulley groove + bob drop + counterweight drop)
  const plumbWire = new TautPlumbWire(materials.plumbWire);
  stage.specimenGroup.add(plumbWire.group);

  // 5. Ephemeral Per-Vertex Alpha-Faded Orbital Phase Filament
  const wake = new OrbitalWakeRibbon(materials.trailRibbon);
  stage.specimenGroup.add(wake.mesh);

  // 6. Pre-allocated Pool of MAX_STONES Watertight Cleaved Magnetite Crystal Meshes
  const stoneGeos = createStoneGeometryVariants();
  const stonePool: StoneVisualSlot[] = [];
  const stoneHitMeshes: THREE.Mesh[] = [];
  const stonesGroup = new THREE.Group();
  stonesGroup.name = 'MagneticStonesGroup';
  stage.specimenGroup.add(stonesGroup);

  for (let i = 0; i < MAX_STONES; i++) {
    const variant = i % stoneGeos.variants.length;
    const mesh = new THREE.Mesh(stoneGeos.variants[variant], materials.cinnabarStone);
    mesh.visible = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.poolIndex = i;
    stonesGroup.add(mesh);

    stonePool.push({ mesh, stoneId: -1, variant });
    stoneHitMeshes.push(mesh);
  }

  // Acoustic flexural impact ripple ring-buffer cursor
  let rippleCursor = 0;

  function pushImpactRipple(normal: THREE.Vector3, strength: number): void {
    const slot = rippleCursor;
    rippleCursor = (rippleCursor + 1) % MAX_IMPACT_RIPPLES;
    materials.impactOriginsAge[slot].set(normal.x, normal.y, normal.z, 0.005);
    const amps = materials.uImpactAmps.value;
    if (slot === 0) amps.x = strength;
    else if (slot === 1) amps.y = strength;
    else if (slot === 2) amps.z = strength;
    else amps.w = strength;
  }

  // Interaction state
  const canvas = stage.shell.canvas;
  let hoveredStoneId = -1;
  let isHoveringBob = false;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownTime = 0;
  let lastClickTime = 0;
  let downHitType: 'none' | 'bob' | 'stone' | 'dome' | 'rim' = 'none';
  let downStoneId = -1;
  const downDomePoint = new THREE.Vector3();

  function updateNdc(clientX: number, clientY: number): void {
    const rect = canvas.getBoundingClientRect();
    _ndc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  function raycastScene(clientX: number, clientY: number): {
    type: 'none' | 'bob' | 'stone' | 'dome' | 'rim';
    stoneId: number;
    point: THREE.Vector3 | null;
  } {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);

    const bobHits = _raycaster.intersectObject(bob.hitMesh, false);
    if (bobHits.length > 0) {
      return { type: 'bob', stoneId: -1, point: bobHits[0].point };
    }

    const activeStoneHits = _raycaster.intersectObjects(stoneHitMeshes, false);
    for (let i = 0; i < activeStoneHits.length; i++) {
      const idx = activeStoneHits[i].object.userData.poolIndex as number | undefined;
      if (idx !== undefined && stonePool[idx].mesh.visible && stonePool[idx].stoneId !== -1) {
        return {
          type: 'stone',
          stoneId: stonePool[idx].stoneId,
          point: activeStoneHits[i].point,
        };
      }
    }

    const domeHits = _raycaster.intersectObject(dome.domeHitMesh, false);
    if (domeHits.length > 0) {
      return { type: 'dome', stoneId: -1, point: domeHits[0].point };
    }

    const rimHits = _raycaster.intersectObject(dome.rimMesh, false);
    if (rimHits.length > 0) {
      return { type: 'rim', stoneId: -1, point: rimHits[0].point };
    }

    return { type: 'none', stoneId: -1, point: null };
  }

  function updateGrabTarget(clientX: number, clientY: number): void {
    updateNdc(clientX, clientY);
    _raycaster.setFromCamera(_ndc, stage.camera);
    stage.camera.getWorldDirection(_camDir);
    _grabPlane.setFromNormalAndCoplanarPoint(_camDir, sim.pendulum.pos);
    if (_raycaster.ray.intersectPlane(_grabPlane, _grabHit)) {
      sim.pendulum.grabTarget.copy(_grabHit);
    }
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    pointerDownX = ev.clientX;
    pointerDownY = ev.clientY;
    pointerDownTime = performance.now();

    const hit = raycastScene(ev.clientX, ev.clientY);
    downHitType = hit.type;
    downStoneId = hit.stoneId;
    if (hit.point) downDomePoint.copy(hit.point);

    if (hit.type === 'bob') {
      sim.pendulum.isGrabbed = true;
      stage.controls.enabled = false;
      updateGrabTarget(ev.clientX, ev.clientY);
      canvas.style.cursor = 'grabbing';
      canvas.setPointerCapture(ev.pointerId);
    }
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (sim.pendulum.isGrabbed) {
      updateGrabTarget(ev.clientX, ev.clientY);
      return;
    }

    if (ev.buttons !== 0) return;

    const hit = raycastScene(ev.clientX, ev.clientY);
    isHoveringBob = hit.type === 'bob';
    hoveredStoneId = hit.type === 'stone' ? hit.stoneId : -1;

    if (hit.type === 'bob') {
      canvas.style.cursor = 'grab';
    } else if (hit.type === 'stone' || hit.type === 'rim') {
      canvas.style.cursor = 'pointer';
    } else if (hit.type === 'dome') {
      canvas.style.cursor = 'crosshair';
    } else {
      canvas.style.cursor = 'default';
    }
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (sim.pendulum.isGrabbed) {
      sim.pendulum.isGrabbed = false;
      sim.pendulum.wirePluckAmp = Math.min(1.0, sim.pendulum.wirePluckAmp + 0.45);
      stage.controls.enabled = true;
      canvas.style.cursor = 'grab';
      try {
        canvas.releasePointerCapture(ev.pointerId);
      } catch {
        // Ignore if capture was already released
      }
      downHitType = 'none';
      return;
    }

    const moveDist = Math.hypot(ev.clientX - pointerDownX, ev.clientY - pointerDownY);
    const heldMs = performance.now() - pointerDownTime;
    const isCleanClick = moveDist < 7 && heldMs < 420;

    if (isCleanClick) {
      const now = performance.now();
      const isDoubleClick = now - lastClickTime < 340;
      lastClickTime = now;

      if (
        downHitType === 'rim' ||
        (isDoubleClick && (downHitType === 'dome' || downHitType === 'stone'))
      ) {
        sim.castHandful(9);
      } else if (downHitType === 'stone' && downStoneId !== -1) {
        sim.flipStone(downStoneId);
      } else if (downHitType === 'dome') {
        _launchOrigin
          .copy(stage.camera.position)
          .normalize()
          .multiplyScalar(1.65)
          .add(DOME_CENTER);
        _launchOrigin.y = Math.max(1.05, _launchOrigin.y + 0.85);
        sim.tossStoneToward(downDomePoint, _launchOrigin);
      }
    }

    downHitType = 'none';
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      sim.castHandful(9);
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('keydown', onKeyDown);
    stage.scene.environment = null;
    studioEnv.dispose();
    dome.dispose();
    gimbalPulley.dispose();
    bob.dispose();
    plumbWire.dispose();
    wake.dispose();
    stoneGeos.dispose();
    materials.dispose();
    surfaceMaps.dispose();
  });

  // Expose read-only + QA diagnostic hook
  window.__PENDULUM__ = {
    getState: () => ({
      simTime: Number(sim.simTime.toFixed(3)),
      stoneCount: sim.stones.length,
      clungCount: sim.stones.filter((s) => s.phase === 'clung').length,
      bobPos: sim.pendulum.pos.toArray().map((n) => Number(n.toFixed(4))),
      bobTipPos: sim.pendulum.tipPos.toArray().map((n) => Number(n.toFixed(4))),
      bobSpeed: Number(sim.pendulum.vel.length().toFixed(4)),
      bobClearance: Number(
        (sim.pendulum.pos.distanceTo(DOME_CENTER) - DOME_RADIUS).toFixed(4)
      ),
      tipClearance: Number(
        (sim.pendulum.tipPos.distanceTo(DOME_CENTER) - DOME_RADIUS).toFixed(4)
      ),
      wireLength: Number(sim.pendulum.wireLength.toFixed(4)),
      pulleyWheelAngle: Number(sim.pendulum.pulleyWheelAngle.toFixed(3)),
      counterweightY: Number(sim.pendulum.counterweightY.toFixed(4)),
      magneticForceMag: Number(sim.pendulum.magneticForceMag.toFixed(4)),
      activeScuffEnergy: surfaceMaps.getActiveScuffEnergy(),
      isGrabbed: sim.pendulum.isGrabbed,
      stones: sim.stones.map((s: MagneticStone) => ({
        id: s.id,
        polarity: s.polarity,
        phase: s.phase,
        pos: s.pos.toArray().map((n) => Number(n.toFixed(3))),
        tugMag: Number(s.tugTilt.length().toFixed(3)),
      })),
    }),
    tossStone: (x: number, y: number, z: number) => {
      sim.tossStoneToward(new THREE.Vector3(x, y, z));
    },
    castHandful: (count = 9) => {
      sim.castHandful(count);
    },
    flipStone: (id: number) => sim.flipStone(id),
  };

  // Main animation tick (strictly zero allocations)
  stage.onTick(({ dtSec, camera }) => {
    sim.step(dtSec);

    // 1. Drain impact acoustic ripples and stone scuff events onto dome surface
    for (let i = 0; i < sim.impactEvents.length; i++) {
      const ev = sim.impactEvents[i];
      pushImpactRipple(ev.normal, ev.strength);
    }
    for (let i = 0; i < sim.scuffEvents.length; i++) {
      const ev = sim.scuffEvents[i];
      surfaceMaps.stampScuff(ev.normal, ev.polarity, ev.intensity, ev.isImpact);
    }
    sim.clearSurfaceEvents();

    for (let k = 0; k < MAX_IMPACT_RIPPLES; k++) {
      const slot = materials.impactOriginsAge[k];
      if (slot.w >= 0) {
        slot.w += dtSec;
        if (slot.w > 1.05) {
          slot.w = -1.0;
        }
      }
    }

    const tipGap = sim.pendulum.tipPos.distanceTo(DOME_CENTER) - DOME_RADIUS;
    const wakeProx = Math.max(0, Math.min(1, 1.0 - (tipGap - 0.04) / 0.16));
    surfaceMaps.stampBobWake(sim.pendulum.dir, wakeProx, sim.pendulum.vel.length(), dtSec);
    surfaceMaps.decayAndUpload(dtSec);

    // 2. Subtle dome vertical recoil from stone impacts
    dome.group.position.set(
      DOME_CENTER.x,
      DOME_CENTER.y + sim.pendulum.domeRecoilY,
      DOME_CENTER.z
    );

    // 3. Update bob transform & shared shader uniforms
    bob.group.position.copy(sim.pendulum.pos);
    bob.group.quaternion.copy(sim.pendulum.quat);

    materials.uBobWorldPos.value.copy(sim.pendulum.pos);
    materials.uBobTipPos.value.copy(sim.pendulum.tipPos);
    const hoverTarget = isHoveringBob || sim.pendulum.isGrabbed ? 1 : 0;
    materials.uBobHover.value +=
      (hoverTarget - materials.uBobHover.value) * Math.min(1, dtSec * 14);

    // 4. Update swivel differential pulley, rising/falling brass counterweight & razor-taut plumb wire
    sim.computePulleyKinematics(_pulleyFrame);
    gimbalPulley.swivelGroup.rotation.y = _pulleyFrame.azimuth;
    gimbalPulley.wheelGroup.rotation.x = _pulleyFrame.wheelAngle;
    gimbalPulley.counterweightGroup.position.copy(_pulleyFrame.cwEyeletWorld);
    gimbalPulley.counterweightGroup.rotation.y = _pulleyFrame.azimuth;
    gimbalPulley.counterweightGroup.rotation.x = -sim.pendulum.cwSwayAngle;
    plumbWire.update(_pulleyFrame);

    // 5. Update ephemeral orbital phase filament
    wake.update(
      sim.trailPositions,
      sim.trailSpeeds,
      sim.trailCharge,
      sim.trailHead,
      sim.trailCount,
      camera.position
    );

    // 6. Sync cleaved lodestone pool visuals & real-time 18-stone dome contact AO uniforms
    const stoneCount = sim.stones.length;
    const seatR = DOME_RADIUS + STONE_SEAT_OFFSET;
    for (let i = 0; i < MAX_STONES; i++) {
      const slot = stonePool[i];
      const aoVec = materials.stoneAOData[i];

      if (i < stoneCount) {
        const st = sim.stones[i];
        const hoverGoal = st.id === hoveredStoneId ? 1 : 0;
        st.hoverPoise += (hoverGoal - st.hoverPoise) * Math.min(1, dtSec * 16);

        if (slot.variant !== st.variant) {
          slot.variant = st.variant;
          slot.mesh.geometry = stoneGeos.variants[st.variant];
        }

        slot.mesh.visible = true;
        slot.stoneId = st.id;
        slot.mesh.position.set(
          st.pos.x,
          st.pos.y + sim.pendulum.domeRecoilY,
          st.pos.z
        );
        slot.mesh.quaternion.copy(st.quat);
        slot.mesh.scale.setScalar(st.scale);
        slot.mesh.material =
          st.polarity === 1 ? materials.cinnabarStone : materials.verdigrisStone;

        _stoneNormal.copy(st.pos).sub(DOME_CENTER);
        const distFromCenter = _stoneNormal.length();
        const heightAboveIron = Math.max(0, distFromCenter - seatR);
        const seatTightness = Math.exp(-heightAboveIron / 0.085);
        if (distFromCenter > 1e-4) {
          _stoneNormal.divideScalar(distFromCenter);
          const angRadius = (STONE_RADIUS * st.scale) / DOME_RADIUS;
          aoVec.set(
            _stoneNormal.x,
            _stoneNormal.y,
            _stoneNormal.z,
            angRadius * seatTightness
          );
        } else {
          aoVec.set(0, 1, 0, 0);
        }
      } else {
        slot.mesh.visible = false;
        slot.stoneId = -1;
        aoVec.set(0, 1, 0, 0);
      }
    }
  });
}

mountDay18Pendulum().catch((err) => {
  console.error('[Day 18 - Pendulum] Failed to initialize WebGPU specimen:', err);
});
