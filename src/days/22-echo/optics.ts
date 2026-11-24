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

/**
 * Shared horological geometry constants for the suspended Edison prism lamp
 * (single source of truth ensuring exact Bounce-0 real-3D to Bounce-1..40 raytracer parity).
 */
export const CEILING_ROSETTE_TOP_Y = GLASS_PANE_HALF - 0.0005; // 0.5975 (flush against underside of +Y glass)
export const CEILING_ROSETTE_BOT_Y = 0.566;
export const CEILING_ROSETTE_RADIUS = 0.092;
export const CORD_GRIP_TOP_Y = 0.572;
export const CORD_GRIP_BOT_Y = 0.528;
export const GLAND_TOP_Y = 0.531;
export const GLAND_BOT_Y = 0.508;
export const GROMMET_TOP_Y = 0.511;
export const GROMMET_BOT_Y = 0.488;

/** Braided textile pendant cord radius & local socket attachment elevation (in lamp head local +Y). */
export const CORD_RADIUS = 0.0105;
export const CORD_SOCKET_ATTACH_Y = 0.304;

/** Vintage Edison lamp socket & prism gallery cage dimensions (in lamp head local +Y, 0 = prism center). */
export const SOCKET_TOP_Y = 0.316;
export const SOCKET_CAP_BOT_Y = 0.252;
export const SOCKET_VENT_BOT_Y = 0.224;
export const SOCKET_KNURL_BOT_Y = 0.198;
export const SOCKET_SKIRT_BOT_Y = 0.168;
export const SOCKET_INSULATOR_BOT_Y = 0.152;
export const E27_BASE_TOP_Y = 0.16;
export const E27_BASE_BOT_Y = 0.114;

/** Truncated 8-facet optical crystal prism & brass drop finial dimensions. */
export const PRISM_EQUATOR_R = 0.132;
export const PRISM_APEX_Y = 0.194;
export const PRISM_TRUNC_Y = 0.154;
export const PRISM_SLAB_Y_RATIO = PRISM_EQUATOR_R / PRISM_APEX_Y;
export const PRISM_SLAB_NORM_INV = 1 / Math.sqrt(2 + PRISM_SLAB_Y_RATIO * PRISM_SLAB_Y_RATIO);
export const PRISM_HALF_D = PRISM_EQUATOR_R * PRISM_SLAB_NORM_INV;
export const FINIAL_TOP_Y = -0.148;
export const FINIAL_TIP_Y = -0.242;

/** Hand-blown ST64 Edison bulb envelope & 6-loop squirrel-cage filament bounds (local +Y). */
export const BULB_TOP_Y = 0.118;
export const BULB_BOT_Y = -0.116;
export const BULB_MAX_R = 0.056;
export const FILAMENT_TOP_Y = 0.058;
export const FILAMENT_BOT_Y = -0.066;
export const FILAMENT_TOP_R = 0.026;
export const FILAMENT_BOT_R = 0.035;
export const FILAMENT_LOOPS = 6;
/** Bounding sphere radius around `corePos` enclosing socket, cord-grip nut, prism cage, and finial. */
export const LAMP_BOUND_RADIUS = 0.34;

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

export interface TiltedPlaneEdgeDistances {
  /** Signed distance to the nearest mirror plane (0 on a wall hit). */
  wallDist: number;
  /** Exact perpendicular distance to the nearest adjacent tilted mirror edge. */
  edgeDist: number;
  /** Exact perpendicular distance to the nearest tilted cube corner vertex. */
  cornerDist: number;
}

/**
 * Evaluates the exact signed interior distances from `point` to all 6 tilted mirror planes
 * `d_k = dot(point, n_k) + CAVITY_HALF` and returns the 3 smallest values `(wallDist, edgeDist, cornerDist)`.
 *
 * Matches the TSL fragment raytracer exactly so interior frame rails, corner gussets, and
 * optical chamfer strips remain strictly seam-locked across all mirror tilts up to ±3.2°.
 */
export function computeTiltedPlaneEdgeDistances(
  point: THREE.Vector3,
  normals: readonly THREE.Vector3[]
): TiltedPlaneEdgeDistances {
  let d1 = 1e6;
  let d2 = 1e6;
  let d3 = 1e6;

  for (let k = 0; k < 6; k++) {
    const dk = point.dot(normals[k]) + CAVITY_HALF;
    if (dk < d1) {
      d3 = d2;
      d2 = d1;
      d1 = dk;
    } else if (dk < d2) {
      d3 = d2;
      d2 = dk;
    } else if (dk < d3) {
      d3 = dk;
    }
  }

  return { wallDist: d1, edgeDist: d2, cornerDist: d3 };
}

/**
 * Maps screen-space pointer drag deltas `(dx, dy)` (CSS pixels, `+dx` right, `+dy` down)
 * smoothly into cube-local mirror micro-tilt increments `(dTiltXRad, dTiltYRad)` aligned
 * with the active camera orbit orientation across all 6 faces with zero quadrant jumps.
 */
export function projectScreenDragToMirrorTilt(
  dx: number,
  dy: number,
  camRight: THREE.Vector3,
  camUp: THREE.Vector3,
  camForward: THREE.Vector3,
  sensitivity: number
): [number, number] {
  const wx = camForward.x * camForward.x;
  const wy = camForward.y * camForward.y;
  const wz = camForward.z * camForward.z;

  // Exact 1st-bounce apparent echo-shift sensitivity vectors in cube-local space:
  //   J_tx = (0, 2*wz - 0.9*wx, -2*wy)
  //   J_ty = (2*wz - 0.9*wy, 0, -2*wx)
  const jtxY = 2.0 * wz - 0.9 * wx;
  const jtxZ = -2.0 * wy;
  const jtyX = 2.0 * wz - 0.9 * wy;
  const jtyZ = -2.0 * wx;

  const sXRaw = jtxY * camUp.y + jtxZ * camUp.z;
  const sYRaw = jtyX * camRight.x + jtyZ * camRight.z;

  // Smooth C^infinity regularization prevents dead spots at the 45° saddle azimuth while preserving
  // exact visual corridor steering polarity on all 6 faces (+X, -X, +Y, -Y, +Z, -Z).
  const sXEff = sXRaw + 0.32 * camUp.y * (1.0 - Math.min(1.0, Math.abs(sXRaw)));
  const sYEff = sYRaw + 0.32 * camRight.x * (1.0 - Math.min(1.0, Math.abs(sYRaw)));

  const gainX = sXEff / Math.sqrt(sXEff * sXEff + 0.35);
  const gainY = sYEff / Math.sqrt(sYEff * sYEff + 0.35);

  // Upward screen drag (-dy > 0) bends the visible echo corridor upward;
  // rightward screen drag (+dx > 0) bends the visible echo corridor rightward.
  const dTiltXRad = -dy * gainX * sensitivity;
  const dTiltYRad = dx * gainY * sensitivity;
  return [dTiltXRad, dTiltYRad];
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
