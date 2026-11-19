import { describe, expect, it } from 'vitest';
import {
  CAM0_POS,
  HAND_L_WORLD_3D,
  HAND_R_WORLD_3D,
  MOTH_SPECS,
  MirageController,
  computeRayLockAngularError,
} from '../mirageRig';

describe('Day 15 Mirage — Optical Ray-Lock, 42° Arc & Reversibility Rig', () => {
  it('ray-locks all 9 dormant origami moths onto the curled figure silhouette from CAM0', () => {
    expect(MOTH_SPECS).toHaveLength(9);

    for (const moth of MOTH_SPECS) {
      const errRad = computeRayLockAngularError(moth.perchPos, moth.pSil, CAM0_POS);
      expect(errRad).toBeLessThan(1e-6);
      // Staggered across distinct Z planes in front of the figure (z = -0.45)
      expect(moth.perchZ).toBeGreaterThan(-0.45);
      expect(moth.perchZ).toBeLessThan(0.15);
    }
  });

  it('produces immediate parallax shift on 0% hover and awakens moths on click', () => {
    const ctrl = new MirageController();
    ctrl.step(0.016, 0.5, 0.0, 0.0, 0.0, 0.0);
    const camRestX = ctrl.cameraState.pos[0];

    ctrl.step(0.12, 0.62, 0.45, 0.15, 0.8, 0.2);
    expect(Math.abs(ctrl.cameraState.pos[0] - camRestX)).toBeGreaterThan(0.04);

    // Click wake (< 6 px travel)
    ctrl.onPointerDown(0.0, 0.0, 400, 300);
    ctrl.onPointerMove(0.002, 0.002, 402, 301);
    const wasClick = ctrl.onPointerUp();
    expect(wasClick).toBe(true);

    for (let i = 0; i < 60; i++) {
      ctrl.step(0.016, 1.0 + i * 0.016, 0.0, 0.0, 0.0, 0.0);
    }
    const diag = ctrl.getDiagnostics();
    expect(diag.uAwake).toBeGreaterThan(0.25);
    expect(diag.activeMothCount).toBeGreaterThanOrEqual(2);
  });

  it('achieves ~42° left arc, >1.7x hand separation, and >115° face light swing at 100% reveal', () => {
    const ctrl = new MirageController();
    ctrl.step(0.016, 0.0, 0.0, 0.0, 0.0, 0.0);
    const restDiag = ctrl.getDiagnostics();

    // At 0% rest: contra-jour negative face lighting & tight eclipsed hand separation
    expect(restDiag.faceDotLight).toBeLessThan(0);
    expect(restDiag.maxRayLockErrorAtRest).toBeLessThan(1e-6);
    expect(restDiag.handSeparationPxRatio).toBeCloseTo(1.0, 4);

    // Scroll fully into 100% reveal
    for (let i = 0; i < 12; i++) {
      ctrl.onWheel(120);
    }
    for (let i = 0; i < 120; i++) {
      ctrl.step(0.016, 2.0 + i * 0.016, 0.0, 0.0, 0.0, 0.0);
    }

    const revealDiag = ctrl.getDiagnostics();
    expect(revealDiag.uReveal).toBeGreaterThan(0.98);
    expect(revealDiag.arcAngleDeg).toBeGreaterThanOrEqual(38);
    expect(revealDiag.arcAngleDeg).toBeLessThanOrEqual(46);
    expect(revealDiag.handSeparationPxRatio).toBeGreaterThan(1.7);
    expect(revealDiag.faceDotLight).toBeGreaterThan(0.75);
    expect(revealDiag.faceLightSwingDeg).toBeGreaterThan(115);
    expect(revealDiag.activeMothCount).toBe(9);
    expect(HAND_L_WORLD_3D[2]).toBeGreaterThan(HAND_R_WORLD_3D[2]);
  });

  it('honors continuous drag direction without quadrant jumps and cleanly reverses to 0%', () => {
    const ctrl = new MirageController();
    // Drag down-right continuously across multiple quadrants (zero jump inversions)
    ctrl.onPointerDown(-0.3, 0.3, 300, 200);
    ctrl.onPointerMove(-0.1, 0.1, 380, 260);
    const midReveal = ctrl.targetReveal;
    ctrl.onPointerMove(0.2, -0.2, 500, 380);
    const endReveal = ctrl.targetReveal;
    const wasClick = ctrl.onPointerUp();

    expect(wasClick).toBe(false);
    expect(midReveal).toBeGreaterThan(0.1);
    expect(endReveal).toBeGreaterThan(midReveal);
    expect(ctrl.targetReveal).toBe(endReveal); // Preserved on release

    // Scroll back to 0% -> full reversibility
    for (let i = 0; i < 15; i++) {
      ctrl.onWheel(-120);
    }
    for (let i = 0; i < 160; i++) {
      ctrl.step(0.016, 4.0 + i * 0.016, 0.0, 0.0, 0.0, 0.0);
    }
    const backDiag = ctrl.getDiagnostics();
    expect(backDiag.uReveal).toBeLessThan(0.01);
    expect(backDiag.uAwake).toBeLessThan(0.01);
    expect(backDiag.activeMothCount).toBe(0);
    expect(backDiag.faceDotLight).toBeLessThan(0);
  });
});
