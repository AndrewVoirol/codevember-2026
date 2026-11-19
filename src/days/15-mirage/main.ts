import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { buildFigureRig } from './figureSculpt';
import { createMirageAudio } from './mirageAudio';
import {
  ARCH_LAYERS,
  CAM0_FOV,
  CAM0_POS,
  FLAME_FLOOR_3D,
  HANDS_HOLLOW_3D,
  MOTH_SPECS,
  MirageController,
} from './mirageRig';
import {
  createArchSheetMaterial,
  createFaceFeaturesMaterial,
  createFigureMaterial,
  createFleckMaterial,
  createLanternGlowMaterial,
  createMirageUniforms,
  createMothWingMaterial,
  createSwirlMaterial,
  createTeardropFlameMaterial,
  type MothShaderControl,
} from './sketchMaterials';
import {
  buildAllArchGeometries,
  buildBackSanctumGeometry,
  buildPageBlockSheavesGeometry,
} from './sketchbookGeometry';
import type { MirageDiagnostics } from './types';

declare global {
  interface Window {
    __MIRAGE__?: {
      getState: () => MirageDiagnostics;
    };
  }
}

// Pre-allocated scratch math objects (zero allocations inside rAF tick)
const _mothForward = new THREE.Vector3();
const _mothUp = new THREE.Vector3(0, 1, 0);
const _mothRight = new THREE.Vector3();
const _mothTrueUp = new THREE.Vector3();
const _mothRotMat = new THREE.Matrix4();
const _mothRollQuat = new THREE.Quaternion();
const _fleckPos = new THREE.Vector3();
const _fleckQuat = new THREE.Quaternion();
const _fleckScale = new THREE.Vector3();
const _fleckMat = new THREE.Matrix4();
const _zAxis = new THREE.Vector3(0, 0, 1);

export async function bootDay15Mirage(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(15);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [...CAM0_POS],
    fov: CAM0_FOV,
    enableOrbit: false,
    enableShadows: false,
  });

  // Tighten camera near plane so smooth dollying through outer arches never clips
  stage.camera.near = 0.04;
  stage.camera.far = 25;
  stage.camera.updateProjectionMatrix();

  const controller = new MirageController();
  const uniforms = createMirageUniforms();
  const audio = createMirageAudio();

  // 1. Build 7 Carved Torn-Paper Arch Layers + Curved Back Sanctum + Stacked Page Block
  const frontMat = createArchSheetMaterial(uniforms, true);
  const innerArchMat = createArchSheetMaterial(uniforms, false);

  const archGeoms = buildAllArchGeometries();
  let frontSpreadMesh: THREE.Mesh | null = null;
  for (let k = 0; k < archGeoms.length; k++) {
    const isFront = k === 0;
    const mesh = new THREE.Mesh(archGeoms[k], isFront ? frontMat : innerArchMat);
    mesh.name = `ArchLayer_${k}`;
    mesh.renderOrder = isFront ? 2 : 0;
    stage.specimenGroup.add(mesh);
    if (isFront) frontSpreadMesh = mesh;
  }

  const backMesh = new THREE.Mesh(buildBackSanctumGeometry(), innerArchMat);
  backMesh.name = 'BackSanctumSheet';
  stage.specimenGroup.add(backMesh);

  const pageBlockMesh = new THREE.Mesh(buildPageBlockSheavesGeometry(), innerArchMat);
  pageBlockMesh.name = 'StackedPageBlock';
  stage.specimenGroup.add(pageBlockMesh);

  // 2. Sculpted Curled Young Woman, Cupped Hands, 9 Ray-Locked Origami Moths, Swirl Curls & Sparks
  const bodyMat = createFigureMaterial(uniforms, 'body');
  const hairMat = createFigureMaterial(uniforms, 'hair');
  const featuresMat = createFaceFeaturesMaterial(uniforms);
  const strokesMat = createFigureMaterial(uniforms, 'strokes');
  const swirlMat = createSwirlMaterial(uniforms);
  const fleckMat = createFleckMaterial(uniforms);

  const mothControls: MothShaderControl[] = [];
  const figureRig = buildFigureRig(
    bodyMat,
    hairMat,
    featuresMat,
    strokesMat,
    (id) => {
      const ctrl = createMothWingMaterial(uniforms, MOTH_SPECS[id].pigmentBias);
      mothControls.push(ctrl);
      return ctrl.material;
    },
    swirlMat,
    fleckMat
  );

  stage.specimenGroup.add(figureRig.root);
  for (let i = 0; i < figureRig.mothGroups.length; i++) {
    stage.specimenGroup.add(figureRig.mothGroups[i]);
  }
  stage.specimenGroup.add(figureRig.emberFlecksMesh);

  // 3. Tiny Floor Crease Ember Flame & Rising Cupped-Hands Lantern Halo
  const emberGlowMat = createLanternGlowMaterial(uniforms, 'creaseEmber');
  const emberFlameMat = createTeardropFlameMaterial(uniforms);
  const palmGlowMat = createLanternGlowMaterial(uniforms, 'palmBowl');

  const emberGlowPlane = new THREE.Mesh(new THREE.PlaneGeometry(0.14, 0.14), emberGlowMat);
  emberGlowPlane.position.set(FLAME_FLOOR_3D[0], FLAME_FLOOR_3D[1] + 0.008, FLAME_FLOOR_3D[2] + 0.014);
  emberGlowPlane.renderOrder = 4;
  stage.specimenGroup.add(emberGlowPlane);

  const emberFlamePlane = new THREE.Mesh(new THREE.PlaneGeometry(0.028, 0.044), emberFlameMat);
  emberFlamePlane.position.set(FLAME_FLOOR_3D[0], FLAME_FLOOR_3D[1] + 0.012, FLAME_FLOOR_3D[2] + 0.018);
  emberFlamePlane.renderOrder = 5;
  stage.specimenGroup.add(emberFlamePlane);

  const palmGlowPlane = new THREE.Mesh(new THREE.PlaneGeometry(0.24, 0.24), palmGlowMat);
  palmGlowPlane.position.set(HANDS_HOLLOW_3D[0], HANDS_HOLLOW_3D[1] + 0.012, HANDS_HOLLOW_3D[2] + 0.01);
  palmGlowPlane.renderOrder = 4;
  stage.specimenGroup.add(palmGlowPlane);

  // 4. Direct Interaction Listeners (Wheel Dolly/Orbit, Click Wake, Drag Scrub, Double-Click Reset)
  const canvas = stage.shell.canvas;

  const toNdc = (ev: PointerEvent): [number, number] => {
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, r.width);
    const h = Math.max(1, r.height);
    const nx = ((ev.clientX - r.left) / w) * 2 - 1;
    const ny = -((ev.clientY - r.top) / h) * 2 + 1;
    return [nx, ny];
  };

  // Guard initial dead-center 2D trompe-l'oeil against stationary/activation OS cursor events on load.
  // Parallax engages smoothly on the first intentional pointer motion (>= 8 px after boot settle), click, or scroll.
  const bootTimeMs = performance.now();
  let hasActivePointer = false;
  let initialMoveX: number | null = null;
  let initialMoveY: number | null = null;

  const onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    hasActivePointer = true;
    audio.unlock();
    controller.onWheel(ev.deltaY);
    audio.playPaperShift(Math.min(1, Math.abs(ev.deltaY) * 0.012));
  };

  const onPointerDown = (ev: PointerEvent) => {
    hasActivePointer = true;
    audio.unlock();
    const [nx, ny] = toNdc(ev);
    controller.onPointerDown(nx, ny, ev.clientX, ev.clientY);
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (!hasActivePointer) {
      if (performance.now() - bootTimeMs < 350) {
        initialMoveX = ev.clientX;
        initialMoveY = ev.clientY;
      } else if (initialMoveX === null || initialMoveY === null) {
        initialMoveX = ev.clientX;
        initialMoveY = ev.clientY;
      } else if (Math.hypot(ev.clientX - initialMoveX, ev.clientY - initialMoveY) >= 8) {
        hasActivePointer = true;
      }
    }
    const [nx, ny] = toNdc(ev);
    controller.onPointerMove(nx, ny, ev.clientX, ev.clientY);
  };

  const onPointerUp = () => {
    const wasClick = controller.onPointerUp();
    if (wasClick) {
      audio.playEmberSpark(0.85);
    }
  };

  const onDblClick = () => {
    hasActivePointer = false;
    initialMoveX = null;
    initialMoveY = null;
    controller.resetToRest();
    audio.playPaperShift(0.6);
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.key === 'ArrowUp' || ev.key === 'w') {
      hasActivePointer = true;
      controller.onWheel(-120);
    } else if (ev.key === 'ArrowDown' || ev.key === 's') {
      hasActivePointer = true;
      controller.onWheel(120);
    } else if (ev.key === 'r' || ev.key === 'R') {
      hasActivePointer = false;
      initialMoveX = null;
      initialMoveY = null;
      controller.resetToRest();
    }
  };

  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('pointerdown', onPointerDown, { passive: true });
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerup', onPointerUp, { passive: true });
  canvas.addEventListener('dblclick', onDblClick, { passive: true });
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('dblclick', onDblClick);
    window.removeEventListener('keydown', onKeyDown);
    audio.dispose();
  });

  // Expose live diagnostics without allocations
  window.__MIRAGE__ = {
    getState: () => controller.getDiagnostics(),
  };

  if (window.__SPECIMEN_STAGE__) {
    const baseGetState = window.__SPECIMEN_STAGE__.getSceneState;
    window.__SPECIMEN_STAGE__.getSceneState = () => {
      const diag = controller.getDiagnostics();
      return {
        ...baseGetState(),
        uReveal: Number(diag.uReveal.toFixed(4)),
        uAwake: Number(diag.uAwake.toFixed(4)),
        effectiveAwake: Number(diag.effectiveAwake.toFixed(4)),
        activeMothCount: diag.activeMothCount,
        arcAngleDeg: Number(diag.arcAngleDeg.toFixed(2)),
        faceDotLight: Number(diag.faceDotLight.toFixed(3)),
        handSeparationPxRatio: Number(diag.handSeparationPxRatio.toFixed(3)),
        maxRayLockErrorAtRest: Number(diag.maxRayLockErrorAtRest.toFixed(7)),
      };
    };
  }

  let lastFov = CAM0_FOV;

  // 5. Unpaused Zero-GC Animation Loop
  stage.onTick(({ dtSec, elapsedSec, pointer }) => {
    const activePtrX = hasActivePointer ? pointer.ndc.x : 0;
    const activePtrY = hasActivePointer ? pointer.ndc.y : 0;
    controller.step(
      dtSec,
      elapsedSec,
      activePtrX,
      activePtrY,
      hasActivePointer ? pointer.speed : 0,
      hasActivePointer ? pointer.dwellTimeSec : 0
    );

    const camState = controller.cameraState;
    stage.camera.position.set(camState.pos[0], camState.pos[1], camState.pos[2]);
    stage.camera.lookAt(camState.target[0], camState.target[1], camState.target[2]);
    if (Math.abs(camState.fovDeg - lastFov) > 1e-4) {
      stage.camera.fov = camState.fovDeg;
      stage.camera.updateProjectionMatrix();
      lastFov = camState.fovDeg;
    }

    const flicker =
      0.88 +
      0.12 * (0.62 * Math.sin(elapsedSec * 7.4) + 0.38 * Math.sin(elapsedSec * 16.9 + 0.8));

    uniforms.uTime.value = elapsedSec;
    uniforms.uReveal.value = controller.uReveal;
    uniforms.uAwake.value = controller.effectiveAwake;
    uniforms.uLightLift.value = controller.lightLift;
    uniforms.uSparkPulse.value = controller.sparkPulse;
    uniforms.uFlicker.value = flicker;
    uniforms.uLightPos.value.set(
      controller.lightPos[0],
      controller.lightPos[1],
      controller.lightPos[2]
    );
    uniforms.uEmberPos.value.set(
      controller.emberPos[0],
      controller.emberPos[1],
      controller.emberPos[2]
    );

    // Smoothly dissolve Layer 0 open front spread as camera passes inside the carved tunnel
    if (frontSpreadMesh) {
      const distToFront = stage.camera.position.z - ARCH_LAYERS[0].z;
      const frontFade = Math.max(0, Math.min(1, (distToFront - 0.08) / 0.14));
      uniforms.uFrontFade.value = frontFade;
      frontSpreadMesh.visible = frontFade > 0.01;
    }

    // Billboard soft ember & cupped-hands glow planes toward active camera
    emberGlowPlane.quaternion.copy(stage.camera.quaternion);
    emberFlamePlane.quaternion.copy(stage.camera.quaternion);
    palmGlowPlane.quaternion.copy(stage.camera.quaternion);
    const emberS = flicker * (0.85 + controller.sparkPulse * 0.4);
    emberFlamePlane.scale.set(emberS, flicker * (0.95 + controller.sparkPulse * 0.3), 1);

    // Sync 9 articulated 3D origami moths (zero allocations)
    for (let i = 0; i < figureRig.mothGroups.length; i++) {
      const m = controller.moths[i];
      const spec = MOTH_SPECS[i];
      const mg = figureRig.mothGroups[i];
      mg.position.set(m.pos[0], m.pos[1], m.pos[2]);
      mg.scale.setScalar(m.scale);

      // Orient moth along flight velocity when awake, or flat along silhouette contour when perched
      if (m.awake < 0.01) {
        mg.quaternion.identity();
        _mothRollQuat.setFromAxisAngle(_zAxis, spec.perchRoll);
        mg.quaternion.copy(_mothRollQuat);
        const tuck = 0.06 + m.wingFlapRad * 0.25;
        figureRig.mothRightWings[i].rotation.z = tuck;
        figureRig.mothLeftWings[i].rotation.z = -tuck;
      } else {
        _mothForward.set(m.tangent[0], m.tangent[1], m.tangent[2]).normalize();
        _mothRight.crossVectors(_mothUp, _mothForward);
        if (_mothRight.lengthSq() < 1e-6) _mothRight.set(1, 0, 0);
        _mothRight.normalize();
        _mothTrueUp.crossVectors(_mothForward, _mothRight).normalize();
        _mothRotMat.makeBasis(_mothRight, _mothTrueUp, _mothForward);
        mg.quaternion.setFromRotationMatrix(_mothRotMat);

        _mothRollQuat.setFromAxisAngle(_zAxis, spec.perchRoll * (1 - m.awake) + spec.wingTilt * m.awake);
        mg.quaternion.multiply(_mothRollQuat);

        const tuckRest = 0.06;
        const flightFlap = m.wingFlapRad - 0.32;
        const wingAngle = tuckRest * (1 - m.awake) + flightFlap * m.awake;
        figureRig.mothRightWings[i].rotation.z = wingAngle;
        figureRig.mothLeftWings[i].rotation.z = -wingAngle;
      }

      mothControls[i].uAwake.value = m.awake;
      mothControls[i].uHoverGlow.value = m.hoverGlow;
    }

    // Sync 28 rising amber ember flecks above cupped palms (zero-scaled when dormant at 0%)
    const fleckGate = Math.max(0, Math.min(1, (controller.effectiveAwake - 0.08) / 0.55));
    for (let j = 0; j < 28; j++) {
      const cycle = ((elapsedSec * (0.28 + (j % 5) * 0.06) + j * 0.37) % 1.0);
      const swirlA = cycle * 7.5 + j * 1.41;
      const rad = (0.018 + (j % 7) * 0.008) * (0.5 + cycle);
      _fleckPos.set(
        HANDS_HOLLOW_3D[0] - 0.025 * cycle + Math.cos(swirlA) * rad,
        HANDS_HOLLOW_3D[1] + 0.015 + cycle * 0.21,
        HANDS_HOLLOW_3D[2] + Math.sin(swirlA) * rad * 0.7
      );
      _fleckQuat.copy(stage.camera.quaternion);
      const pulse = Math.sin(cycle * Math.PI) * fleckGate;
      const s = (0.45 + (j % 3) * 0.22) * pulse;
      _fleckScale.set(s, s, s);
      _fleckMat.compose(_fleckPos, _fleckQuat, _fleckScale);
      figureRig.emberFlecksMesh.setMatrixAt(j, _fleckMat);
    }
    figureRig.emberFlecksMesh.instanceMatrix.needsUpdate = true;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay15Mirage();
}
