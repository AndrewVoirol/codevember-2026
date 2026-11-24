import * as THREE from 'three/webgpu';

export const NOZZLE_HOVER_STANDOFF = 0.29;
export const NOZZLE_BLAST_STANDOFF = 0.21;
export const NOZZLE_PARK_RADIUS = 1.62;

export interface NozzlePoseInput {
  isHoveringRock: boolean;
  isBlasting: boolean;
  isOrbiting: boolean;
  hitPos: THREE.Vector3;
  hitNormal: THREE.Vector3;
  cameraPos: THREE.Vector3;
  cameraUp: THREE.Vector3;
  elapsedSec: number;
}

// Pre-allocated scratch vectors/matrices (strictly zero per-frame allocations)
const _toCam = new THREE.Vector3();
const _camRight = new THREE.Vector3();
const _camTrueUp = new THREE.Vector3();
const _offsetDir = new THREE.Vector3();
const _aimDir = new THREE.Vector3();
const _targetTip = new THREE.Vector3();
const _targetQuat = new THREE.Quaternion();
const _basisX = new THREE.Vector3();
const _basisY = new THREE.Vector3();
const _basisZ = new THREE.Vector3();
const _rotMat = new THREE.Matrix4();
const _parkAim = new THREE.Vector3();

/**
 * Computes an exact right-handed quaternion whose local `-Z` axis points strictly along
 * `forwardDir` (the carbide jet aim vector), keeping the stylus upright relative to `upHint`.
 */
export function computeAimQuaternion(
  forwardDir: THREE.Vector3,
  upHint: THREE.Vector3,
  outQuat: THREE.Quaternion
): THREE.Quaternion {
  // Local +Z points backward along the stylus handle = -forwardDir
  _basisZ.copy(forwardDir).normalize().negate();

  _basisX.crossVectors(upHint, _basisZ);
  if (_basisX.lengthSq() < 1e-6) {
    _basisX.set(1, 0, 0).cross(_basisZ);
  }
  _basisX.normalize();
  _basisY.crossVectors(_basisZ, _basisX).normalize();

  _rotMat.makeBasis(_basisX, _basisY, _basisZ);
  return outQuat.setFromRotationMatrix(_rotMat);
}

/**
 * Pure, unit-testable kinematic controller for the articulated Brass Air-Abrasion Micro-Nozzle.
 * Positions the carbide orifice cleanly outside the limestone surface on the camera-facing
 * oblique working angle, aims dead-on into the raycast impact point, applies subtle pneumatic
 * micro-vibration while blasting, and glides gracefully aside while orbiting empty space.
 */
export class NozzleController {
  readonly tipPos = new THREE.Vector3(0.96, 0.56, 1.18);
  readonly orientation = new THREE.Quaternion();
  readonly aimDir = new THREE.Vector3(-0.5, -0.3, -0.8).normalize();
  jetLength = NOZZLE_HOVER_STANDOFF;
  jetActiveAlpha = 0;

  private initialized = false;

  /**
   * Computes the instantaneous target tip position and orientation for a given surface hit
   * and camera vantage (exported for deterministic unit-testing of standoff & aiming math).
   */
  computeTargetPose(
    input: NozzlePoseInput,
    outTip: THREE.Vector3,
    outQuat: THREE.Quaternion,
    outAimDir: THREE.Vector3
  ): number {
    // Camera-aligned studio frame
    _toCam.subVectors(input.cameraPos, input.isHoveringRock ? input.hitPos : _parkAim.set(0, 0, 0));
    if (_toCam.lengthSq() < 1e-6) _toCam.set(0, 0, 1);
    _toCam.normalize();

    _camRight.crossVectors(input.cameraUp, _toCam);
    if (_camRight.lengthSq() < 1e-6) _camRight.set(1, 0, 0);
    _camRight.normalize();
    _camTrueUp.crossVectors(_toCam, _camRight).normalize();

    if (input.isHoveringRock && !input.isOrbiting) {
      const standoff = input.isBlasting ? NOZZLE_BLAST_STANDOFF : NOZZLE_HOVER_STANDOFF;

      // Oblique ~38-deg laboratory handpiece working posture:
      // Combines outward surface normal with gentle right-hand & upper-camera tilt so the
      // machined brass stylus silhouette is always readable and NEVER clips into the rock.
      _offsetDir
        .copy(input.hitNormal)
        .multiplyScalar(0.68)
        .addScaledVector(_camRight, 0.44)
        .addScaledVector(_camTrueUp, 0.24)
        .addScaledVector(_toCam, 0.34);

      // Guarantee strictly positive outward clearance from local surface tangent plane
      const dotNormal = _offsetDir.dot(input.hitNormal);
      if (dotNormal < 0.38) {
        _offsetDir.addScaledVector(input.hitNormal, 0.38 - dotNormal);
      }
      // Guarantee camera-facing side clearance
      const dotCam = _offsetDir.dot(_toCam);
      if (dotCam < 0.22) {
        _offsetDir.addScaledVector(_toCam, 0.22 - dotCam);
      }
      _offsetDir.normalize();

      outTip.copy(input.hitPos).addScaledVector(_offsetDir, standoff);

      // Subtle tactile pneumatic micro-vibration when firing pressurized dolomite grit
      if (input.isBlasting) {
        const vib = Math.sin(input.elapsedSec * 92.0) * 0.0022;
        const vib2 = Math.cos(input.elapsedSec * 127.0) * 0.0018;
        outTip.addScaledVector(_camRight, vib).addScaledVector(_camTrueUp, vib2);
      }

      // Aim strictly from the carbide tip dead-on into the raycast impact point
      outAimDir.subVectors(input.hitPos, outTip).normalize();
      computeAimQuaternion(outAimDir, _camTrueUp, outQuat);
      return outTip.distanceTo(input.hitPos);
    }

    // Poised resting park off the upper-right margin of the specimen (retracts slightly further while orbiting)
    const parkDist = input.isOrbiting ? NOZZLE_PARK_RADIUS + 0.36 : NOZZLE_PARK_RADIUS;
    const breath = Math.sin(input.elapsedSec * 1.4) * 0.025;
    outTip
      .copy(_camRight)
      .multiplyScalar(0.92 * (parkDist / NOZZLE_PARK_RADIUS))
      .addScaledVector(_camTrueUp, 0.58 + breath)
      .addScaledVector(_toCam, 1.05 * (parkDist / NOZZLE_PARK_RADIUS));

    // Gently angled toward the specimen's upper-right fracture window
    _parkAim.copy(_camRight).multiplyScalar(0.24).addScaledVector(_camTrueUp, 0.14);
    outAimDir.subVectors(_parkAim, outTip).normalize();
    computeAimQuaternion(outAimDir, _camTrueUp, outQuat);
    return NOZZLE_HOVER_STANDOFF;
  }

  /**
   * Advances damped nozzle position, orientation, and jet plume opacity without allocating.
   */
  update(input: NozzlePoseInput, dtSec: number): void {
    const targetJetLen = this.computeTargetPose(input, _targetTip, _targetQuat, _aimDir);

    if (!this.initialized) {
      this.tipPos.copy(_targetTip);
      this.orientation.copy(_targetQuat);
      this.aimDir.copy(_aimDir);
      this.jetLength = targetJetLen;
      this.initialized = true;
    } else {
      const rate = input.isBlasting ? 26.0 : input.isHoveringRock ? 18.0 : 9.0;
      const alpha = 1.0 - Math.exp(-rate * Math.max(0.001, dtSec));
      this.tipPos.lerp(_targetTip, alpha);
      this.orientation.slerp(_targetQuat, alpha);
      this.jetLength += (targetJetLen - this.jetLength) * alpha;

      // Keep aimDir synced to current orientation (-Z local axis)
      this.aimDir.set(0, 0, -1).applyQuaternion(this.orientation).normalize();
    }

    const jetGoal = input.isBlasting && input.isHoveringRock && !input.isOrbiting ? 1.0 : 0.0;
    const jetRate = jetGoal > this.jetActiveAlpha ? 32.0 : 18.0;
    this.jetActiveAlpha += (jetGoal - this.jetActiveAlpha) * (1.0 - Math.exp(-jetRate * dtSec));
  }
}
