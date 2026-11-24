import * as THREE from 'three/webgpu';
import { MINERAL_PALETTE } from '../../core/palette';

/**
 * Day 22 - Echo: Shared Optical Cavity Constants & Analytic Multi-Bounce Raytracer
 *
 * Single source of truth consumed by the physics controller, procedural brass frame,
 * 40-bounce TSL fragment raytracer, and Vitest verification suite.
 */

/** Outer half-extent of the machined dark-bronze skeleton cube (~1.45 units wide). */
export const CUBE_OUTER_HALF = 0.725;

/** Interior optical mirror cavity half-extent (6 convex mirror planes at distance 0.668 from origin). */
export const CAVITY_HALF = 0.668;

/** Outer glass window box half-extent recessed inside the stepped satin-brass bezel. */
export const GLASS_PANE_HALF = 0.598;

/** Clear inner sight aperture framed by the machined satin-brass retaining bezels. */
export const APERTURE_HALF = 0.58;

/** Total interior reflection bounces evaluated per fragment inside the two-way mirror cube. */
export const ECHO_BOUNCE_COUNT = 40;

/**
 * Maximum micrometer mirror-plane micro-tilt in degrees.
 * At ±3.2°, every visible point on the GLASS_PANE_HALF panes satisfies dot(p, n_k) + CAVITY_HALF > 0
 * for all 6 tilted mirror planes simultaneously (100% strictly convex interior guarantee).
 */
export const MAX_MIRROR_TILT_DEG = 3.2;
export const MAX_MIRROR_TILT_RAD = (MAX_MIRROR_TILT_DEG * Math.PI) / 180;

/** Soda-lime mirror dielectric reflectivity at normal incidence per bounce. */
export const MIRROR_BASE_REFLECTIVITY = 0.945;

/** Effective double-pass soda-lime glass optical chord per reflection bounce. */
export const SODA_LIME_CHORD_PER_BOUNCE = 0.048;

/**
 * Soda-lime float glass ferrous-iron spectral extinction (per unit chord),
 * anchored directly in Verdigris Malachite & Celadon Glauconite mineral pigments.
 * Absorbs red strongest and green-celadon least -> authentic emerald progression across 40 echoes.
 */
export const SODA_LIME_EXTINCTION_RGB: readonly [number, number, number] = [
  MINERAL_PALETTE.celadonGlauconite.extinctionRgb[0] * 0.58 +
    MINERAL_PALETTE.verdigrisMalachite.extinctionRgb[0] * 0.42,
  MINERAL_PALETTE.celadonGlauconite.extinctionRgb[1] * 0.58 +
    MINERAL_PALETTE.verdigrisMalachite.extinctionRgb[1] * 0.42,
  MINERAL_PALETTE.celadonGlauconite.extinctionRgb[2] * 0.58 +
    MINERAL_PALETTE.verdigrisMalachite.extinctionRgb[2] * 0.42,
];

/** Canonical optical mirror geometries cycled by double-click. */
export interface MirrorPreset {
  readonly name: string;
  readonly tiltXDeg: number;
  readonly tiltYDeg: number;
}

export const MIRROR_PRESETS: readonly MirrorPreset[] = [
  { name: 'Sweeping Arc', tiltXDeg: 1.55, tiltYDeg: -1.1 },
  { name: 'Parallel Hall', tiltXDeg: 0.0, tiltYDeg: 0.0 },
  { name: 'Helical Vortex', tiltXDeg: 2.75, tiltYDeg: 2.15 },
  { name: 'Kaleidoscope Corner', tiltXDeg: -3.05, tiltYDeg: 2.55 },
];

/**
 * Populates the 6 inward-pointing unit normals of the slightly tilted interior mirror planes.
 *
 * Opposing mirror pairs form a symmetric slight wedge controlled by (tiltXRad, tiltYRad):
 * - When tiltX = tiltY = 0, opposing faces are strictly antiparallel (infinite orthogonal lattice).
 * - When tiltX or tiltY != 0, opposing walls converge slightly (n_pos + n_neg != 0),
 *   rotating reflected ray directions by ~2*tilt per bounce and curving echo corridors
 *   into sweeping catenary arches, helical vortices, and recursive kaleidoscopic rings.
 */
export function computeMirrorPlaneNormals(
  tiltXRad: number,
  tiltYRad: number,
  outNormals: readonly THREE.Vector3[]
): void {
  const tx = Math.max(-MAX_MIRROR_TILT_RAD, Math.min(MAX_MIRROR_TILT_RAD, tiltXRad));
  const ty = Math.max(-MAX_MIRROR_TILT_RAD, Math.min(MAX_MIRROR_TILT_RAD, tiltYRad));

  const sx = Math.tan(tx);
  const sy = Math.tan(ty);
  const crossX = 0.45 * sx;
  const crossY = 0.45 * sy;

  // 0: +X wall (inward -X), 1: -X wall (inward +X)
  outNormals[0].set(-1, crossX, sy).normalize();
  outNormals[1].set(1, crossX, sy).normalize();

  // 2: +Y wall (inward -Y), 3: -Y wall (inward +Y)
  outNormals[2].set(crossY, -1, sx).normalize();
  outNormals[3].set(crossY, 1, sx).normalize();

  // 4: +Z wall (inward -Z), 5: -Z wall (inward +Z)
  outNormals[4].set(-sy, -sx, -1).normalize();
  outNormals[5].set(-sy, -sx, 1).normalize();
}

/**
 * Evaluates cumulative soda-lime Beer-Lambert + dielectric mirror spectral transmittance
 * after `bounceIndex` interior mirror reflections (`bounceIndex >= 0`).
 *
 * Per-bounce attenuation:
 *   T_step(rgb) = R_eff * exp(-SODA_LIME_EXTINCTION_RGB * SODA_LIME_CHORD_PER_BOUNCE)
 */
export function evalBounceTransmittanceRgb(
  bounceIndex: number,
  reflectivity = MIRROR_BASE_REFLECTIVITY,
  chordPerBounce = SODA_LIME_CHORD_PER_BOUNCE
): [number, number, number] {
  const k = Math.max(0, bounceIndex);
  const stepR = reflectivity * Math.exp(-SODA_LIME_EXTINCTION_RGB[0] * chordPerBounce);
  const stepG = reflectivity * Math.exp(-SODA_LIME_EXTINCTION_RGB[1] * chordPerBounce);
  const stepB = reflectivity * Math.exp(-SODA_LIME_EXTINCTION_RGB[2] * chordPerBounce);
  return [Math.pow(stepR, k), Math.pow(stepG, k), Math.pow(stepB, k)];
}

export interface TracedBounceStep {
  bounce: number;
  wallIndex: number;
  hitPoint: THREE.Vector3;
  reflectedDir: THREE.Vector3;
  segmentLength: number;
  cumulativePath: number;
}

/**
 * CPU reference multi-bounce raytracer matching the exact 6-plane reflection geometry
 * of the TSL fragment shader (used for mathematical verification and unit testing).
 */
export function traceCavityRayCpu(
  origin: THREE.Vector3,
  direction: THREE.Vector3,
  tiltXRad: number,
  tiltYRad: number,
  maxBounces = ECHO_BOUNCE_COUNT
): TracedBounceStep[] {
  const normals = [
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
  ];
  computeMirrorPlaneNormals(tiltXRad, tiltYRad, normals);

  const ro = origin.clone();
  const rd = direction.clone().normalize();
  const steps: TracedBounceStep[] = [];
  let cumulativePath = 0;

  for (let b = 0; b < maxBounces; b++) {
    let bestT = 1e6;
    let bestWall = -1;

    for (let k = 0; k < 6; k++) {
      const n = normals[k];
      const denom = rd.dot(n);
      if (denom < -1e-5) {
        const distToPlane = ro.dot(n) + CAVITY_HALF;
        const t = -distToPlane / denom;
        if (t > 1e-4 && t < bestT) {
          bestT = t;
          bestWall = k;
        }
      }
    }

    if (bestWall === -1 || bestT >= 1e5) {
      break;
    }

    const hitPoint = ro.clone().addScaledVector(rd, bestT);
    const hitNormal = normals[bestWall];
    const reflectedDir = rd.clone().reflect(hitNormal).normalize();
    cumulativePath += bestT;

    steps.push({
      bounce: b,
      wallIndex: bestWall,
      hitPoint,
      reflectedDir,
      segmentLength: bestT,
      cumulativePath,
    });

    // Advance origin slightly inward along hit normal to stay strictly inside the convex cavity
    ro.copy(hitPoint).addScaledVector(hitNormal, 0.0005);
    rd.copy(reflectedDir);
  }

  return steps;
}
