import * as THREE from 'three/webgpu';
import {
  abs,
  clamp,
  dot,
  exp,
  float,
  mix,
  normalLocal,
  positionLocal,
  positionWorld,
  smoothstep,
  uniform,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

const SLAT_COUNT = 10;
const HINGE_COUNT = SLAT_COUNT - 1;
const PANEL_WIDTH = 1.62;
const SLAT_LENGTH = 0.214;
const SLAT_PITCH = 0.222; // Includes 8mm articulated hinge clearance
const SLAT_THICKNESS = 0.032;
const STACK_PITCH = 0.0345; // Exact layer-to-layer thickness offset when folded 180°
const CASCADE_OVERLAP = 1.28; // Momentum handoff overlap so hinges never stall between slabs

/**
 * Physical gravity tip-and-fall curve for a rigid hinged slab:
 * Accelerates smoothly under gravity into the 180° contact stop (no pre-impact braking).
 */
function gravityHingeCurve(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return Math.pow(c, 1.52);
}

export async function bootDay01Fold(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(1);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.12, 4.65],
    fov: 35,
    enableOrbit: false, // Direct click & drag controls the diagonal panel's hinges & corner alignment
  });

  // Physical mineral pigments from src/core/palette.ts
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const rawUmber = createPigmentUniforms('rawUmber');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const realgarAmber = createPigmentUniforms('realgarAmber');

  // Uniforms driving shader highlights on active hinge seam & mechanical slap
  const uActiveHingeWorld = uniform(new THREE.Vector3(0, 0, 0));
  const uHingeStress = uniform(0);

  // Root articulated assembly suspended in center of #212529 Chalky Slate space
  const panelRig = new THREE.Group();
  panelRig.name = 'DiagonalRigidFoldPanel';
  stage.specimenGroup.add(panelRig);

  // Shared two-sided mineral lamina NodeMaterial:
  // - Recto (+Z local face): Bone Calcite & Warm Kaolin geological slab
  // - Verso (-Z local face): Raw Umber & Yellow Ochre underside (revealed on 180° fold-over)
  // - Chamfer walls (|normalLocal.z| < 0.5): Burnished Raw Umber + Realgar Amber hinge edge
  const slatMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.44,
    metalness: 0.08,
  });

  const pLocal = positionLocal;
  const nLocal = normalLocal;
  const pWorld = positionWorld;

  const toPointer = pWorld.sub(stage.uPointerWorld);
  const pointerProx = exp(dot(toPointer, toPointer).negate().mul(1.65));

  const toActiveHinge = pWorld.sub(uActiveHingeWorld);
  const hingeProx = exp(dot(toActiveHinge, toActiveHinge).negate().mul(3.2));

  const tooth = mineralToothFbm(pLocal.mul(vec3(1.8, 6.5, 4.0)), float(2.8));
  const glazeThickness = clamp(
    float(0.48).add(tooth.mul(0.32)).sub(pointerProx.mul(0.14)),
    0.08,
    1.15
  );

  // Recto face: Bone Calcite over Warm Kaolin
  const rectoColor = kubelkaMunkGlaze(
    boneCalcite.K,
    boneCalcite.S,
    glazeThickness,
    warmKaolin.rInf
  );

  // Verso face (revealed when a rigid slat hinges 180° flat over onto the next slat):
  // Warm Yellow Ochre mineral veining glazed over deep Raw Umber
  const versoGlazeThickness = clamp(float(0.34).add(tooth.mul(0.38)), 0.06, 0.95);
  const versoBase = kubelkaMunkGlaze(
    yellowOchre.K,
    yellowOchre.S,
    versoGlazeThickness,
    rawUmber.rInf
  );
  const versoColor = mix(rawUmber.rInf, versoBase, float(0.64));

  // Distinguish recto (+Z), verso (-Z), and rigid slab thickness walls
  const isRecto = smoothstep(-0.2, 0.2, nLocal.z);
  const faceColor = mix(versoColor, rectoColor, isRecto);

  const isEdgeWall = float(1.0).sub(smoothstep(0.45, 0.75, abs(nLocal.z)));
  const edgeTrimColor = mix(rawUmber.rInf, realgarAmber.rInf, float(0.28).add(tooth.mul(0.25)));
  const slabBaseRgb = mix(faceColor, edgeTrimColor, isEdgeWall.mul(0.88));

  // Subtle recessed border groove near each rigid slat's perimeter
  const edgeU = abs(pLocal.x).div(PANEL_WIDTH * 0.5);
  const edgeV = abs(pLocal.y).div(SLAT_LENGTH * 0.5);
  const bevelMask = smoothstep(0.86, 0.99, edgeU).add(smoothstep(0.78, 0.98, edgeV));
  const beveledRgb = mix(slabBaseRgb, rawUmber.rInf, clamp(bevelMask, 0.0, 1.0).mul(0.32));

  // Warm Realgar Amber resonance along the currently hinging fold line & pointer touch
  const stressGlow = clamp(
    hingeProx.mul(uHingeStress.mul(0.55)).add(pointerProx.mul(0.24)),
    0.0,
    0.65
  );
  slatMat.colorNode = mix(beveledRgb, realgarAmber.rInf, stressGlow);
  slatMat.roughnessNode = clamp(
    float(0.46).sub(tooth.mul(0.18)).sub(isEdgeWall.mul(0.12)),
    0.2,
    0.76
  );

  // Create the 10 rigid rectangular slats
  const slatGeom = new THREE.BoxGeometry(PANEL_WIDTH, SLAT_LENGTH, SLAT_THICKNESS, 1, 1, 1);
  const slatMeshes: THREE.Mesh[] = [];

  for (let i = 0; i < SLAT_COUNT; i++) {
    const mesh = new THREE.Mesh(slatGeom, slatMat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    panelRig.add(mesh);
    slatMeshes.push(mesh);
  }

  // Create precision Raw Umber & Ochre cylindrical hinge pins along the 9 crease joints
  const pinGeom = new THREE.CylinderGeometry(0.0085, 0.0085, PANEL_WIDTH * 1.032, 12);
  pinGeom.rotateZ(Math.PI * 0.5);
  const pinMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.32,
    metalness: 0.52,
  });
  pinMat.colorNode = mix(rawUmber.rInf, yellowOchre.rInf, float(0.38));

  const hingePinMeshes: THREE.Mesh[] = [];
  for (let j = 0; j < HINGE_COUNT; j++) {
    const pin = new THREE.Mesh(pinGeom, pinMat);
    pin.castShadow = true;
    panelRig.add(pin);
    hingePinMeshes.push(pin);
  }

  // Pre-allocated scratch buffers for zero-GC per-frame kinematics
  const slatV = new Float32Array(SLAT_COUNT);
  const slatW = new Float32Array(SLAT_COUNT);
  const slatAngle = new Float32Array(SLAT_COUNT);
  const baseV = new Float32Array(SLAT_COUNT);
  const hingeV = new Float32Array(HINGE_COUNT);
  const hingeW = new Float32Array(HINGE_COUNT);

  for (let i = 0; i < SLAT_COUNT; i++) {
    // Slat 0 is at the upper end (+v), Slat N-1 is at the lower end (-v)
    baseV[i] = ((SLAT_COUNT - 1) * 0.5 - i) * SLAT_PITCH;
  }

  // Kinematic state:
  // In Three.js right-handed coordinates, rotation.z = -PI/4 tilts local +Y (+v) to Top-Right (+X, +Y),
  // rotation.z = 0 keeps local +Y (+v) vertical at Top-Center,
  // and rotation.z = +PI/4 tilts local +Y (+v) to Top-Left (-X, +Y).
  let panelAngle = -Math.PI * 0.25;
  let targetPanelAngle = -Math.PI * 0.25;
  const panelOffset = new THREE.Vector2(0, 0);
  const targetPanelOffset = new THREE.Vector2(0, 0);

  // foldProgress in [0, HINGE_COUNT]:
  //   0           = flat planar mode
  //   HINGE_COUNT = all rigid slats folded 180° over on themselves into a stacked packet
  let foldProgress = 0.22;
  let foldVelocity = 0.0;
  let targetMode: 'poised' | 'fold' | 'unfold' = 'poised';
  let topAnchoredUnfold = false; // True when positioned at top so slats unfold downwards from Slat 0
  let lastIntHinge = 0;
  let hingeSlapImpulse = 0;

  // Pointer gesture tracking
  let wasPointerDown = false;
  let isDraggingGesture = false;
  const pointerDownWorld = new THREE.Vector2(0, 0);
  let pointerDownTime = 0;
  let dragStartFoldProgress = 0;
  let dragStartProj = 0;

  const activeHingeLocal = new THREE.Vector3();

  /**
   * Switches between bottom-stacked (anchorTop = false) and top-stacked (anchorTop = true)
   * without any visual jump when the panel is either completely planar (p ≈ 0)
   * or completely stacked (p ≈ HINGE_COUNT).
   */
  function setAnchorTopSeamless(nextAnchorTop: boolean): void {
    if (topAnchoredUnfold === nextAnchorTop) return;
    if (foldProgress > HINGE_COUNT - 0.15) {
      // Shift panelOffset along the panel's longitudinal axis so the folded stack stays put in world space
      const span = baseV[0] - baseV[SLAT_COUNT - 1];
      const dir = nextAnchorTop ? -1 : 1;
      const upX = -Math.sin(panelAngle);
      const upY = Math.cos(panelAngle);
      panelOffset.x += upX * span * dir;
      panelOffset.y += upY * span * dir;
    }
    topAnchoredUnfold = nextAnchorTop;
  }

  /**
   * Evaluates sequential rigid-packet folding kinematics with gravitational
   * momentum handoff (CASCADE_OVERLAP) and accelerating 180° hinge slaps.
   */
  function updateRigidFoldGeometry(p: number, anchorTop: boolean): void {
    for (let i = 0; i < SLAT_COUNT; i++) {
      slatV[i] = baseV[i];
      slatW[i] = 0;
      slatAngle[i] = 0;
    }
    for (let j = 0; j < HINGE_COUNT; j++) {
      hingeV[j] = baseV[j] - SLAT_PITCH * 0.5;
      hingeW[j] = 0;
    }

    const normP = Math.max(0, Math.min(1, p / HINGE_COUNT));
    const waveFront = normP * (HINGE_COUNT - 1 + CASCADE_OVERLAP);

    if (!anchorTop) {
      // Standard gravity fold-down: Upper slats (0..j) hinge 180° over onto lower slats toward the bottom edge
      for (let j = 0; j < HINGE_COUNT; j++) {
        const rawF = Math.max(0, Math.min(1, (waveFront - j) / CASCADE_OVERLAP));
        if (rawF <= 0) break;

        const theta = gravityHingeCurve(rawF) * Math.PI;
        const vPiv = baseV[j] - SLAT_PITCH * 0.5;
        const wPiv = ((j + 1) * STACK_PITCH) * 0.5;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        for (let i = 0; i <= j; i++) {
          const dv = slatV[i] - vPiv;
          const dw = slatW[i] - wPiv;
          slatV[i] = vPiv + dv * cosT - dw * sinT;
          slatW[i] = wPiv + dv * sinT + dw * cosT;
          slatAngle[i] += theta;
        }

        hingeV[j] = vPiv;
        hingeW[j] = wPiv;
        for (let k = 0; k < j; k++) {
          const dv = hingeV[k] - vPiv;
          const dw = hingeW[k] - wPiv;
          hingeV[k] = vPiv + dv * cosT - dw * sinT;
          hingeW[k] = wPiv + dv * sinT + dw * cosT;
        }
      }
    } else {
      // Top-anchored mode: Slat 0 stays anchored at the top; lower slats unfold downwards beneath it
      for (let j = HINGE_COUNT - 1; j >= 0; j--) {
        const stepIdx = HINGE_COUNT - 1 - j;
        const rawF = Math.max(0, Math.min(1, (waveFront - stepIdx) / CASCADE_OVERLAP));
        if (rawF <= 0) break;

        const theta = gravityHingeCurve(rawF) * Math.PI;
        const vPiv = baseV[j] - SLAT_PITCH * 0.5;
        const wPiv = ((stepIdx + 1) * STACK_PITCH) * 0.5;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        for (let i = j + 1; i < SLAT_COUNT; i++) {
          const dv = slatV[i] - vPiv;
          const dw = slatW[i] - wPiv;
          slatV[i] = vPiv + dv * cosT + dw * sinT;
          slatW[i] = wPiv - dv * sinT + dw * cosT;
          slatAngle[i] -= theta;
        }

        hingeV[j] = vPiv;
        hingeW[j] = wPiv;
        for (let k = j + 1; k < HINGE_COUNT; k++) {
          const dv = hingeV[k] - vPiv;
          const dw = hingeW[k] - wPiv;
          hingeV[k] = vPiv + dv * cosT + dw * sinT;
          hingeW[k] = wPiv - dv * sinT + dw * cosT;
        }
      }
    }

    for (let i = 0; i < SLAT_COUNT; i++) {
      const m = slatMeshes[i];
      m.position.set(0, slatV[i], slatW[i]);
      m.rotation.set(slatAngle[i], 0, 0);
    }

    for (let j = 0; j < HINGE_COUNT; j++) {
      hingePinMeshes[j].position.set(0, hingeV[j], hingeW[j]);
    }

    const activeIdx = Math.min(HINGE_COUNT - 1, Math.max(0, Math.floor(p)));
    activeHingeLocal.set(0, hingeV[activeIdx], hingeW[activeIdx]);
    panelRig.localToWorld(activeHingeLocal);
    uActiveHingeWorld.value.copy(activeHingeLocal);
  }

  // Initialize geometry on frame 0
  panelRig.rotation.z = panelAngle;
  updateRigidFoldGeometry(foldProgress, topAnchoredUnfold);

  stage.onTick(({ elapsedSec, dtSec, pointer }) => {
    const px = pointer.worldPos.x;
    const py = pointer.worldPos.y;

    // 1. Pointer press, smooth corner/top drag, and click release
    if (pointer.isDown && !wasPointerDown) {
      wasPointerDown = true;
      isDraggingGesture = false;
      pointerDownWorld.set(px, py);
      pointerDownTime = elapsedSec;
      dragStartFoldProgress = foldProgress;

      const upX = -Math.sin(panelAngle);
      const upY = Math.cos(panelAngle);
      dragStartProj = (px - panelOffset.x) * upX + (py - panelOffset.y) * upY;

      // Pressing while poised immediately releases the upper gravity latch
      if (targetMode === 'poised') {
        targetMode = 'fold';
        foldVelocity = Math.max(foldVelocity, 2.8);
      }
    } else if (pointer.isDown && wasPointerDown) {
      const dragDx = px - pointerDownWorld.x;
      const dragDy = py - pointerDownWorld.y;
      const dragDist = Math.hypot(dragDx, dragDy);

      if (dragDist > 0.065) {
        isDraggingGesture = true;
      }

      if (isDraggingGesture) {
        // Continuous, snap-free angle steering across Top-Left (+45°), Top-Center (0°), and Top-Right (-45°):
        // Never jumps across quadrant lines. When near Top-Center (|px| < 0.25, py > 0.25), smoothly aligns to 0° vertical.
        const topCenterWeight = clampScalar((py - 0.18) * 2.5, 0, 1) * Math.exp(-(px * px) * 9.0);
        const diagonalTarget = -Math.tanh(px * 2.6) * (Math.PI * 0.25);
        targetPanelAngle = diagonalTarget * (1 - topCenterWeight);

        // Let the panel smoothly follow the hand toward the corner or top edge without fighting
        targetPanelOffset.set(
          clampScalar(px * 0.28, -0.42, 0.42),
          clampScalar(py * 0.28, -0.35, 0.38)
        );

        // Relative delta along the panel axis drives smooth unrolling (pulling up) or folding (pushing down)
        const upX = -Math.sin(panelAngle);
        const upY = Math.cos(panelAngle);
        const curProj = (px - panelOffset.x) * upX + (py - panelOffset.y) * upY;
        const deltaProj = curProj - dragStartProj;

        // Pulling upward (+deltaProj or +dragDy) unrolls toward 0 (planar mode);
        // pushing downward (-deltaProj) folds toward HINGE_COUNT.
        const combinedPull = 0.65 * deltaProj + 0.35 * dragDy;
        const directTargetP = clampScalar(
          dragStartFoldProgress - (combinedPull / 1.15) * HINGE_COUNT,
          0,
          HINGE_COUNT
        );

        const prevP = foldProgress;
        foldProgress += (directTargetP - foldProgress) * Math.min(1, dtSec * 12.0);
        foldVelocity = (foldProgress - prevP) / Math.max(0.001, dtSec);
      }
    } else if (!pointer.isDown && wasPointerDown) {
      wasPointerDown = false;
      const heldSec = elapsedSec - pointerDownTime;
      const dragDy = py - pointerDownWorld.y;

      if (!isDraggingGesture && heldSec < 0.45) {
        // Quick Click: Toggle between natural gravity fold-down and unfold!
        targetPanelOffset.set(0, 0);
        if (foldProgress < HINGE_COUNT * 0.5) {
          setAnchorTopSeamless(false);
          targetMode = 'fold';
          foldVelocity = Math.max(foldVelocity, 2.4);
        } else {
          // If positioned near the top, unfold downwards from the top; otherwise unroll back into planar mode
          setAnchorTopSeamless(Math.abs(panelAngle) < 0.22);
          targetMode = 'unfold';
          foldVelocity = Math.min(foldVelocity, -2.6);
        }
        hingeSlapImpulse = 0.75;
      } else {
        // Released after dragging:
        // 1. If positioned at the top center -> anchor at top and let gravity unfold downwards!
        if (py > 0.24 && Math.abs(px) < 0.38) {
          setAnchorTopSeamless(true);
          targetPanelAngle = 0;
          targetPanelOffset.set(0, 0.14);
          targetMode = 'unfold';
          foldVelocity = Math.min(foldVelocity, -2.4);
        } else if (dragDy > 0.04 || py > 0.12 || foldProgress < HINGE_COUNT * 0.55) {
          // 2. Moved up toward an upper corner -> finish unrolling into planar mode along that corner's diagonal!
          setAnchorTopSeamless(false);
          targetPanelAngle = px >= 0 ? -Math.PI * 0.25 : Math.PI * 0.25;
          targetPanelOffset.set(0, 0);
          targetMode = 'unfold';
          foldVelocity = Math.min(foldVelocity, -2.2);
        } else {
          // 3. Dragged downward -> let gravity finish folding the rigid slats down over themselves
          setAnchorTopSeamless(false);
          targetPanelOffset.set(0, 0);
          targetMode = 'fold';
          foldVelocity = Math.max(2.0, foldVelocity);
        }
      }
      isDraggingGesture = false;
    }

    // 2. Natural gravity-driven rigid cascade when not actively dragged
    if (!pointer.isDown || !isDraggingGesture) {
      // Gravity pulls harder as more slats join the falling packet
      const slopeFactor = 0.78 + 0.22 * Math.abs(Math.cos(panelAngle));

      if (targetMode === 'poised') {
        // Subtle gravity tension on the top slat before the first click
        foldProgress = 0.22 + Math.sin(elapsedSec * 1.9) * 0.05;
        foldVelocity = 0;
      } else if (targetMode === 'fold') {
        const packetWeightBoost = 1.0 + 0.09 * foldProgress;
        const gravityAccel = 11.2 * slopeFactor * packetWeightBoost;
        foldVelocity += gravityAccel * dtSec;
        foldVelocity *= Math.exp(-dtSec * 0.85);
        foldProgress += foldVelocity * dtSec;

        if (foldProgress >= HINGE_COUNT) {
          foldProgress = HINGE_COUNT;
          if (Math.abs(foldVelocity) > 0.8) {
            hingeSlapImpulse = Math.min(1.0, Math.abs(foldVelocity) * 0.14);
            foldVelocity = -foldVelocity * 0.16; // Crisp rigid stack contact settle
          } else {
            foldVelocity = 0;
          }
        }
      } else {
        // Unfolding back into flat planar mode (downwards under gravity when top-anchored, or unrolling into corner)
        const unfoldBoost = topAnchoredUnfold
          ? 1.0 + 0.08 * (HINGE_COUNT - foldProgress)
          : 1.05;
        const unfoldAccel = -11.8 * slopeFactor * unfoldBoost;
        foldVelocity += unfoldAccel * dtSec;
        foldVelocity *= Math.exp(-dtSec * 1.05);
        foldProgress += foldVelocity * dtSec;

        if (foldProgress <= 0) {
          foldProgress = 0;
          // Once fully unfolded to planar mode, reset anchorTop seamlessly so the next click folds top-to-bottom
          topAnchoredUnfold = false;
          if (Math.abs(foldVelocity) > 0.8) {
            hingeSlapImpulse = Math.min(1.0, Math.abs(foldVelocity) * 0.14);
            foldVelocity = -foldVelocity * 0.14;
          } else {
            foldVelocity = 0;
          }
        }
      }
    }

    // Trigger tactile slap impulse as each rigid slat hits 180° on the stack
    const curIntHinge = Math.floor(clampScalar(foldProgress, 0, HINGE_COUNT - 0.001));
    if (curIntHinge !== lastIntHinge) {
      lastIntHinge = curIntHinge;
      hingeSlapImpulse = Math.min(1.0, hingeSlapImpulse + 0.38);
    }

    hingeSlapImpulse *= Math.exp(-dtSec * 7.0);
    uHingeStress.value = Math.min(
      1.0,
      hingeSlapImpulse + Math.min(1.0, Math.abs(foldVelocity) * 0.12)
    );

    // Smoothly damp panel diagonal angle & offset
    const angleSmoothing = 1 - Math.exp(-dtSec * 8.5);
    panelAngle += (targetPanelAngle - panelAngle) * angleSmoothing;
    panelOffset.x += (targetPanelOffset.x - panelOffset.x) * angleSmoothing;
    panelOffset.y += (targetPanelOffset.y - panelOffset.y) * angleSmoothing;

    // Subtle 3D studio presentation tilt so rigid slab thickness & stacked packet depth catch the key & rim lights
    const hoverTiltX = -0.24 + pointer.ndc.y * 0.08 + Math.sin(elapsedSec * 0.7) * 0.02;
    const hoverTiltY = 0.28 + pointer.ndc.x * 0.1 + Math.cos(elapsedSec * 0.55) * 0.025;

    panelRig.position.set(
      panelOffset.x,
      panelOffset.y + Math.sin(elapsedSec * 1.1) * 0.012 - hingeSlapImpulse * 0.01,
      0
    );
    panelRig.rotation.set(hoverTiltX, hoverTiltY, panelAngle);

    updateRigidFoldGeometry(foldProgress, topAnchoredUnfold);
  });

  return stage;
}

function clampScalar(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay01Fold();
}
