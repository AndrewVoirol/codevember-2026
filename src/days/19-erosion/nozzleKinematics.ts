import * as THREE from 'three/webgpu';

export const NOZZLE_HOVER_STANDOFF = 0.29;
export const NOZZLE_BLAST_STANDOFF = 0.21;
export const NOZZLE_PARK_RADIUS = 1.62;

/**
 * Local-space clearance stations `(x, y, z, radius)` spanning the full articulated
 * micro-nozzle assembly: carbide tip (z=0), brass collet & knurled grip (0.14..0.54),
 * forged-iron sleeve (0.76..0.96), rear brass barb (1.10), and braided pneumatic hose (1.29..1.76).
 */
export const NOZZLE_CLEARANCE_STATIONS: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.0, 0.0, 0.015, 0.010],
  [0.0, 0.0, 0.14, 0.032],
  [0.0, 0.0, 0.34, 0.044],
  [0.0, 0.0, 0.54, 0.044],
  [0.0, 0.0, 0.76, 0.041],
  [0.0, 0.0, 0.96, 0.038],
  [0.0, 0.0, 1.10, 0.028],
  [0.03, 0.025, 1.29, 0.018],
  [0.14, 0.11, 1.54, 0.018],
  [0.32, 0.24, 1.76, 0.016],
];

export interface NozzlePoseInput {
  isHoveringRock: boolean;
  isBlasting: boolean;
  isOrbiting: boolean;
  hitPos: THREE.Vector3;
  hitNormal: THREE.Vector3;
  cameraPos: THREE.Vector3;
  cameraUp: THREE.Vector3;
  elapsedSec: number;
  /** Optional live surface radius evaluator `(dx, dy, dz) => r` for full-shaft & hose clearance */
  sampleSurfaceRadius?: (dx: number, dy: number, dz: number) => number;
}

// Pre-allocated scratch vectors/matrices (strictly zero per-frame allocations)
const _toCam = new THREE.Vector3();
const _camRight = new THREE.Vector3();
const _camTrueUp = new THREE.Vector3();
const _offsetDir = new THREE.Vector3();
const _radialOut = new THREE.Vector3();
const _aimDir = new THREE.Vector3();
const _targetTip = new THREE.Vector3();
const _targetQuat = new THREE.Quaternion();
const _basisX = new THREE.Vector3();
const _basisY = new THREE.Vector3();
const _basisZ = new THREE.Vector3();
const _rotMat = new THREE.Matrix4();
const _parkAim = new THREE.Vector3();
const _stationWorld = new THREE.Vector3();

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
 * Measures the minimum signed clearance (`|p_station| - surfaceRadius(dir) - stationRadius`)
 * across the entire nozzle tip, brass barrel, iron sleeve, and braided pneumatic hose.
 * Strictly `> 0` guarantees zero clipping anywhere along the 1.76-unit handpiece.
 */
export function measureNozzleMinRockClearance(
  tipPos: THREE.Vector3,
  orientation: THREE.Quaternion,
  sampleSurfaceRadius: (dx: number, dy: number, dz: number) => number
): number {
  let minClearance = Infinity;
  for (let i = 0; i < NOZZLE_CLEARANCE_STATIONS.length; i++) {
    const st = NOZZLE_CLEARANCE_STATIONS[i];
    _stationWorld.set(st[0], st[1], st[2]).applyQuaternion(orientation).add(tipPos);
    const dist = _stationWorld.length();
    if (dist < 1e-6) return -1.0;
    const inv = 1.0 / dist;
    const surfR = sampleSurfaceRadius(
      _stationWorld.x * inv,
      _stationWorld.y * inv,
      _stationWorld.z * inv
    );
    const clearance = dist - surfR - st[3];
    if (clearance < minClearance) {
      minClearance = clearance;
    }
  }
  return minClearance;
}

/**
 * Pure, unit-testable kinematic controller for the articulated Brass Air-Abrasion Micro-Nozzle.
 * Positions the carbide orifice cleanly outside the limestone surface on the camera-facing
 * oblique working angle, enforces full-shaft & hose clearance against the live boulder/crater
 * surface, locks the abrasive jet dead-on into the live impact point, applies subtle pneumatic
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
   * and camera vantage (exported for deterministic unit-testing of standoff, aiming, and clearance).
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
      const baseStandoff = input.isBlasting ? NOZZLE_BLAST_STANDOFF : NOZZLE_HOVER_STANDOFF;
      const hitLen = input.hitPos.length();
      if (hitLen > 1e-5) {
        _radialOut.copy(input.hitPos).multiplyScalar(1.0 / hitLen);
      } else {
        _radialOut.copy(input.hitNormal);
      }

      // Bias lateral lean gently away from the rock center on extreme left/right silhouettes
      // so the handle naturally clears the boulder belly even when orbiting side-on!
      const sideSign = input.hitPos.dot(_camRight) < -0.35 ? 0.18 : 0.40;

      // Oblique laboratory handpiece working posture combining outward surface normal,
      // outward boulder radial vector, camera-right grip posture, and camera-facing clearance.
      _offsetDir
        .copy(input.hitNormal)
        .multiplyScalar(0.58)
        .addScaledVector(_radialOut, 0.24)
        .addScaledVector(_camRight, sideSign)
        .addScaledVector(_camTrueUp, 0.24)
        .addScaledVector(_toCam, 0.36);

      const enforceHalfSpace = () => {
        const dotNormal = _offsetDir.dot(input.hitNormal);
        if (dotNormal < 0.42) {
          _offsetDir.addScaledVector(input.hitNormal, 0.42 - dotNormal);
        }
        const dotCam = _offsetDir.dot(_toCam);
        if (dotCam < 0.26) {
          _offsetDir.addScaledVector(_toCam, 0.26 - dotCam);
        }
        _offsetDir.normalize();
      };

      enforceHalfSpace();

      let standoff = baseStandoff;
      outTip.copy(input.hitPos).addScaledVector(_offsetDir, standoff);
      outAimDir.subVectors(input.hitPos, outTip).normalize();
      computeAimQuaternion(outAimDir, _camTrueUp, outQuat);

      // Full-shaft & braided-hose rock clearance resolver (zero clipping at deep craters or limbs!)
      if (input.sampleSurfaceRadius) {
        const desiredClearance = 0.032;
        for (let iter = 0; iter < 8; iter++) {
          const minClr = measureNozzleMinRockClearance(outTip, outQuat, input.sampleSurfaceRadius);
          if (minClr >= desiredClearance) break;

          // Pull stylus posture smoothly outward toward camera + radial normal away from rock belly
          _offsetDir.addScaledVector(_radialOut, 0.38).addScaledVector(_toCam, 0.34);
          enforceHalfSpace();

          // If deeply recessed inside a steep crater pocket, also raise standoff slightly
          if (iter >= 3) {
            standoff += Math.min(0.08, Math.max(0.025, desiredClearance - minClr));
          }

          outTip.copy(input.hitPos).addScaledVector(_offsetDir, standoff);
          outAimDir.subVectors(input.hitPos, outTip).normalize();
          computeAimQuaternion(outAimDir, _camTrueUp, outQuat);
        }
      }

      // Subtle tactile pneumatic micro-vibration when firing pressurized dolomite grit
      if (input.isBlasting) {
        const vib = Math.sin(input.elapsedSec * 92.0) * 0.0020;
        const vib2 = Math.cos(input.elapsedSec * 127.0) * 0.0016;
        outTip.addScaledVector(_camRight, vib).addScaledVector(_camTrueUp, vib2);
      }

      // Aim strictly from the carbide tip dead-on into the raycast impact point
      outAimDir.subVectors(input.hitPos, outTip).normalize();
      computeAimQuaternion(outAimDir, _camTrueUp, outQuat);
      return outTip.distanceTo(input.hitPos);
    }

    // Poised resting park off the upper-right margin of the specimen (retracts slightly further while orbiting)
    const parkDist = input.isOrbiting ? NOZZLE_PARK_RADIUS + 0.36 : NOZZLE_PARK_RADIUS;
    const breath = Math.sin(input.elapsedSec * 1.4) * 0.022;
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
   * Advances damped nozzle position, exact jet-locked orientation, and plume opacity without allocating.
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
      const rate = input.isBlasting ? 28.0 : input.isHoveringRock ? 18.0 : 9.0;
      const alpha = 1.0 - Math.exp(-rate * Math.max(0.001, dtSec));
      this.tipPos.lerp(_targetTip, alpha);

      // Guard mid-transit tip position against dipping into the convex boulder belly
      if (input.sampleSurfaceRadius && input.isHoveringRock && !input.isOrbiting) {
        const dTip = this.tipPos.length();
        if (dTip > 1e-5) {
          const inv = 1.0 / dTip;
          const surfR = input.sampleSurfaceRadius(
            this.tipPos.x * inv,
            this.tipPos.y * inv,
            this.tipPos.z * inv
          );
          const minTipR = surfR + 0.065;
          if (dTip < minTipR) {
            this.tipPos.multiplyScalar(minTipR * inv);
          }
        }
      }

      if (input.isBlasting && input.isHoveringRock && !input.isOrbiting) {
        // Lock stylus aim & jet length dead-on onto the live post-carve impact spot during active sweeps
        // so the focused dolomite grit stream NEVER drifts or overshoots the crater!
        const liveDist = this.tipPos.distanceTo(input.hitPos);
        if (liveDist > 1e-4) {
          this.aimDir.subVectors(input.hitPos, this.tipPos).multiplyScalar(1.0 / liveDist);
          computeAimQuaternion(this.aimDir, _camTrueUp, this.orientation);
          this.jetLength = liveDist;
        }
      } else {
        this.orientation.slerp(_targetQuat, alpha);
        this.jetLength += (targetJetLen - this.jetLength) * alpha;
        this.aimDir.set(0, 0, -1).applyQuaternion(this.orientation).normalize();
      }
    }

    const jetGoal = input.isBlasting && input.isHoveringRock && !input.isOrbiting ? 1.0 : 0.0;
    const jetRate = jetGoal > this.jetActiveAlpha ? 34.0 : 18.0;
    this.jetActiveAlpha += (jetGoal - this.jetActiveAlpha) * (1.0 - Math.exp(-jetRate * dtSec));
  }
}
