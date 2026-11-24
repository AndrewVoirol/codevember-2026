import * as THREE from 'three/webgpu';

/**
 * Day 29 - Portal — Non-Euclidean Four-World Architectural Math & Reference SDFs
 *
 * Pure, testable module defining:
 * - Architectural cube & recessed arched portal aperture dimensions
 * - Orthonormal cardinal face bases (+Z North, +X East, -Z South, -X West)
 * - Exact camera-parallax ray transforms from cube-local space into each portal world
 * - Visible-face & dominant-face classification (including 45° split-corner dual-world detection)
 * - Safe face-zoom dolly bounds & smooth orbit/zoom camera rig state machine
 * - CPU reference SDF evaluators & sphere-tracers mirroring all 4 GPU TSL worlds
 */

export type PortalFaceId = 'north' | 'east' | 'south' | 'west';

export const PORTAL_FACE_ORDER: readonly PortalFaceId[] = [
  'north',
  'east',
  'south',
  'west',
] as const;

// ---------------------------------------------------------------------------
// Architectural Cube & Recessed Arched Portal Geometry Constants
// ---------------------------------------------------------------------------

/** Outer half-width of the suspended basalt & brass architectural cube (~1.52 units wide). */
export const CUBE_HALF = 0.76;

/** Depth of the recessed chamfered stone & brass arched window reveal. */
export const APERTURE_REVEAL_DEPTH = 0.115;

/** Distance from cube center to each recessed arched portal aperture plane. */
export const APERTURE_PLANE_OFFSET = CUBE_HALF - APERTURE_REVEAL_DEPTH; // 0.645

/** Half-width of the clear arched portal window opening. */
export const ARCH_HALF_WIDTH = 0.435;

/** Bottom stone threshold sill height (cube-local Y). */
export const ARCH_SILL_Y = -0.51;

/** Semicircular arch springline height (cube-local Y). */
export const ARCH_SPRING_Y = 0.085;

/** Radius of the semicircular arch crown (matches ARCH_HALF_WIDTH -> crown at Y = +0.52). */
export const ARCH_RADIUS = ARCH_HALF_WIDTH;

/** Crown height of the arched opening (+0.52). */
export const ARCH_CROWN_Y = ARCH_SPRING_Y + ARCH_RADIUS;

/**
 * Uniform architectural scale factor mapping the ~0.87-unit cube aperture
 * into a monumental ~2.96 m wide portal threshold inside each non-Euclidean world.
 */
export const PORTAL_WORLD_SCALE = 3.4;

// ---------------------------------------------------------------------------
// Camera Orbit & Face-Zoom Dolly Bounds
// ---------------------------------------------------------------------------

/** Outer corner radius of the cube (~1.316). */
export const CUBE_BOUNDING_RADIUS = Math.hypot(CUBE_HALF, CUBE_HALF, CUBE_HALF);

/**
 * Minimum allowed camera distance from cube center.
 * Strictly outside the outer cube corner radius (1.316) + near plane (0.08)
 * and > 1.0 unit clear of every recessed aperture plane (0.645).
 */
export const MIN_CAMERA_DISTANCE = 1.65;

/** Default suspended-cube framing distance (leaves generous #212529 Chalky Slate margin around all 8 corners across 360° orbits). */
export const DEFAULT_CAMERA_DISTANCE = 4.92;

/** Face-zoom dolly target distance (frames full arched stone & brass threshold cleanly inside viewport). */
export const ZOOM_CAMERA_DISTANCE = 2.65;

/** Maximum allowed camera distance. */
export const MAX_CAMERA_DISTANCE = 8.4;

/** Boot camera spherical orientation: crisp ~42° NE corner showing North + East simultaneously. */
export const INITIAL_ORBIT_AZIMUTH_DEG = 42;
export const INITIAL_ORBIT_ELEVATION_DEG = 17;

/** Maximum zoom elevation / orbit polar limits (allowing steep views down stairwell or up vaults). */
export const MIN_ELEVATION_DEG = -72;
export const MAX_ELEVATION_DEG = 74;

// ---------------------------------------------------------------------------
// Cardinal Face Basis Descriptors
// ---------------------------------------------------------------------------

export interface PortalFaceSpec {
  readonly id: PortalFaceId;
  readonly label: string;
  readonly worldTitle: string;
  readonly normal: readonly [number, number, number];
  readonly inward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  /** Cardinal azimuth angle in degrees (0°=+Z North, 90°=+X East, 180°=-Z South, 270°=-X West). */
  readonly azimuthDeg: number;
  /** Ideal elevation angle when zooming into this portal face. */
  readonly zoomElevationDeg: number;
}

export const PORTAL_FACES: Readonly<Record<PortalFaceId, PortalFaceSpec>> = {
  north: {
    id: 'north',
    label: 'North (+Z)',
    worldTitle: 'Sunlit Travertine Colonnade',
    normal: [0, 0, 1],
    inward: [0, 0, -1],
    right: [1, 0, 0],
    up: [0, 1, 0],
    azimuthDeg: 0,
    zoomElevationDeg: 4,
  },
  east: {
    id: 'east',
    label: 'East (+X)',
    worldTitle: 'Plunging Lapis Spiral Stairwell',
    normal: [1, 0, 0],
    inward: [-1, 0, 0],
    right: [0, 0, -1],
    up: [0, 1, 0],
    azimuthDeg: 90,
    zoomElevationDeg: 10,
  },
  south: {
    id: 'south',
    label: 'South (-Z)',
    worldTitle: 'Amber Lantern Over Dark Still Water',
    normal: [0, 0, -1],
    inward: [0, 0, 1],
    right: [-1, 0, 0],
    up: [0, 1, 0],
    azimuthDeg: 180,
    zoomElevationDeg: 6,
  },
  west: {
    id: 'west',
    label: 'West (-X)',
    worldTitle: 'Warm Ochre Desert Slot-Canyon Arch',
    normal: [-1, 0, 0],
    inward: [1, 0, 0],
    right: [0, 0, 1],
    up: [0, 1, 0],
    azimuthDeg: 270,
    zoomElevationDeg: 4,
  },
};

// ---------------------------------------------------------------------------
// Portal Ray Transform & Visibility Classification
// ---------------------------------------------------------------------------

export interface PortalLocalRay {
  /** Entry point in portal-local meters (z = 0 at threshold, +z goes deep into world). */
  ro: [number, number, number];
  /** Normalized ray direction in portal-local coordinates (rd[2] > 0 looking inward). */
  rd: [number, number, number];
}

/**
 * Tests whether a 2D point `(u, v)` on a face lies inside the arched portal aperture.
 */
export function isPointInsideArchAperture(u: number, v: number, margin = 0): boolean {
  const hw = ARCH_HALF_WIDTH + margin;
  const sill = ARCH_SILL_Y - margin;
  if (Math.abs(u) > hw || v < sill) return false;
  if (v <= ARCH_SPRING_Y) return true;
  const dv = v - ARCH_SPRING_Y;
  return u * u + dv * dv <= hw * hw;
}

/**
 * Maps a 2D aperture coordinate `(u, v)` on a cardinal face into cube-local 3D position
 * on that face's recessed aperture plane (`APERTURE_PLANE_OFFSET`).
 */
export function apertureUvToCubeLocal(
  faceId: PortalFaceId,
  u: number,
  v: number,
  planeOffset = APERTURE_PLANE_OFFSET
): [number, number, number] {
  const f = PORTAL_FACES[faceId];
  return [
    f.normal[0] * planeOffset + f.right[0] * u + f.up[0] * v,
    f.normal[1] * planeOffset + f.right[1] * u + f.up[1] * v,
    f.normal[2] * planeOffset + f.right[2] * u + f.up[2] * v,
  ];
}

/**
 * Exact camera-parallax ray transform from cube-local fragment & camera positions
 * into a portal face's canonical local world frame (`+x = right`, `+y = up`, `+z = inward depth`).
 */
export function transformCameraRayToPortalLocal(
  faceId: PortalFaceId,
  fragCubeLocal: readonly [number, number, number],
  camCubeLocal: readonly [number, number, number]
): PortalLocalRay {
  const f = PORTAL_FACES[faceId];

  const dx = fragCubeLocal[0] - camCubeLocal[0];
  const dy = fragCubeLocal[1] - camCubeLocal[1];
  const dz = fragCubeLocal[2] - camCubeLocal[2];
  const invLen = 1 / Math.max(1e-8, Math.hypot(dx, dy, dz));
  const rx = dx * invLen;
  const ry = dy * invLen;
  const rz = dz * invLen;

  const u = fragCubeLocal[0] * f.right[0] + fragCubeLocal[1] * f.right[1] + fragCubeLocal[2] * f.right[2];
  const v = fragCubeLocal[0] * f.up[0] + fragCubeLocal[1] * f.up[1] + fragCubeLocal[2] * f.up[2];
  const nCoord =
    fragCubeLocal[0] * f.normal[0] + fragCubeLocal[1] * f.normal[1] + fragCubeLocal[2] * f.normal[2];
  const w = APERTURE_PLANE_OFFSET - nCoord;

  const rdU = rx * f.right[0] + ry * f.right[1] + rz * f.right[2];
  const rdV = rx * f.up[0] + ry * f.up[1] + rz * f.up[2];
  const rdW = rx * f.inward[0] + ry * f.inward[1] + rz * f.inward[2];

  return {
    ro: [u * PORTAL_WORLD_SCALE, v * PORTAL_WORLD_SCALE, w * PORTAL_WORLD_SCALE],
    rd: [rdU, rdV, rdW],
  };
}

export interface PortalViewClassification {
  dominantFace: PortalFaceId;
  visibleFaces: PortalFaceId[];
  cameraDistance: number;
  orbitAzimuthDeg: number;
  orbitElevationDeg: number;
}

/**
 * Classifies which cardinal portal apertures are visible from `camPos` (relative to cube center),
 * which face is dominant, and spherical orbit azimuth/elevation angles in degrees.
 *
 * A face's recessed aperture is considered visible when the camera is strictly in front of the
 * recessed plane and clears the `0.115`-deep stone reveal grazing angle (`dotNorm > 0.14`).
 */
export function classifyPortalView(
  camPos: readonly [number, number, number]
): PortalViewClassification {
  const [cx, cy, cz] = camPos;
  const cameraDistance = Math.hypot(cx, cy, cz);
  const horiz = Math.hypot(cx, cz);

  let azDeg = (Math.atan2(cx, cz) * 180) / Math.PI;
  if (azDeg < 0) azDeg += 360;
  const elDeg = (Math.atan2(cy, Math.max(1e-8, horiz)) * 180) / Math.PI;

  const invDist = 1 / Math.max(1e-8, cameraDistance);
  const nx = cx * invDist;
  const ny = cy * invDist;
  const nz = cz * invDist;

  let dominantFace: PortalFaceId = 'north';
  let bestDot = -Infinity;
  const visibleScored: Array<{ id: PortalFaceId; score: number }> = [];

  for (let i = 0; i < PORTAL_FACE_ORDER.length; i++) {
    const id = PORTAL_FACE_ORDER[i];
    const f = PORTAL_FACES[id];
    const planeDot = cx * f.normal[0] + cy * f.normal[1] + cz * f.normal[2];
    const dirDot = nx * f.normal[0] + ny * f.normal[1] + nz * f.normal[2];

    if (planeDot > bestDot) {
      bestDot = planeDot;
      dominantFace = id;
    }

    // Must clear the recessed aperture plane + stone jamb reveal grazing angle (~8°)
    if (planeDot > APERTURE_PLANE_OFFSET + 0.04 && dirDot > 0.14) {
      visibleScored.push({ id, score: dirDot });
    }
  }

  // Sort visible faces most-prominent first for stable diagnostic reporting
  visibleScored.sort((a, b) => b.score - a.score);
  const visibleFaces = visibleScored.map((s) => s.id);

  return {
    dominantFace,
    visibleFaces,
    cameraDistance,
    orbitAzimuthDeg: azDeg,
    orbitElevationDeg: elDeg,
  };
}

/**
 * Clamps a camera distance strictly within safe non-clipping bounds.
 */
export function clampCameraDistance(dist: number): number {
  return Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, dist));
}

/**
 * Converts spherical orbit coordinates (`azimuthDeg`, `elevationDeg`, `distance`)
 * to a 3D Cartesian camera position around the cube origin.
 */
export function sphericalOrbitToPosition(
  azimuthDeg: number,
  elevationDeg: number,
  distance: number,
  out?: THREE.Vector3
): THREE.Vector3 {
  const d = clampCameraDistance(distance);
  const az = (azimuthDeg * Math.PI) / 180;
  const el =
    (Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, elevationDeg)) * Math.PI) / 180;
  const cosEl = Math.cos(el);
  const x = d * cosEl * Math.sin(az);
  const y = d * Math.sin(el);
  const z = d * cosEl * Math.cos(az);
  if (out) return out.set(x, y, z);
  return new THREE.Vector3(x, y, z);
}

/**
 * Computes the safe face-zoom camera goal for a chosen portal face,
 * guaranteeing the camera stays well in front of both the outer basalt face (0.76)
 * and the recessed aperture plane (0.645).
 */
export function computeFaceZoomGoal(faceId: PortalFaceId): {
  azimuthDeg: number;
  elevationDeg: number;
  distance: number;
  position: [number, number, number];
} {
  const spec = PORTAL_FACES[faceId];
  const distance = clampCameraDistance(ZOOM_CAMERA_DISTANCE);
  const pos = sphericalOrbitToPosition(spec.azimuthDeg, spec.zoomElevationDeg, distance);
  return {
    azimuthDeg: spec.azimuthDeg,
    elevationDeg: spec.zoomElevationDeg,
    distance,
    position: [pos.x, pos.y, pos.z],
  };
}

/**
 * Shortest signed angular delta in degrees wrapped to `(-180, 180]`.
 */
export function shortestAngleDeltaDeg(fromDeg: number, toDeg: number): number {
  let d = ((toDeg - fromDeg) % 360 + 540) % 360 - 180;
  if (d <= -180) d += 360;
  return d;
}

/**
 * Computes the next 90° stepped orbit azimuth from the current azimuth.
 * Preserves whether the viewer is inspecting cardinal faces (0/90/180/270)
 * or 45° split-corner dual-world views (45/135/225/315), snapping cleanly to the nearest track.
 */
export function computeNextQuarterTurnAzimuth(currentAzimuthDeg: number): number {
  const norm = ((currentAzimuthDeg % 360) + 360) % 360;
  // Determine whether we are closer to a cardinal (0/90/180/270) or 45° split corner
  const mod90 = norm % 90;
  const distToCorner = Math.abs(mod90 - 45);
  const distToCardinal = Math.min(mod90, 90 - mod90);
  if (distToCorner <= distToCardinal) {
    const cornerIndex = Math.floor(norm / 90);
    return ((cornerIndex + 1) % 4) * 90 + 45;
  }
  const cardinalIndex = Math.round(norm / 90) % 4;
  return ((cardinalIndex + 1) % 4) * 90;
}

// ---------------------------------------------------------------------------
// CPU Reference SDF Evaluators (Exact Mathematical Counterparts to TSL Worlds)
// ---------------------------------------------------------------------------

function sdBox3(
  px: number,
  py: number,
  pz: number,
  bx: number,
  by: number,
  bz: number
): number {
  const qx = Math.abs(px) - bx;
  const qy = Math.abs(py) - by;
  const qz = Math.abs(pz) - bz;
  const outLen = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
  const inDist = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
  return outLen + inDist;
}

function smin(a: number, b: number, k: number): number {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * World 0 — North (`+Z`): Sunlit Travertine Colonnade (`> 20 m` deep).
 * Returns signed distance in portal-local meters.
 */
export function evalNorthColonnadeSdfCpu(
  px: number,
  py: number,
  pz: number
): number {
  // Seamless checkered travertine floor seated just below portal sill (-1.734)
  const dFloor = py + 1.76;

  // Repeating fluted classical columns along x = ±1.76, every 2.70 m along +z
  const colSpacing = 2.7;
  const colZ = Math.max(
    1.85,
    Math.min(20.75, Math.round((pz - 1.85) / colSpacing) * colSpacing + 1.85)
  );
  const lx = Math.abs(px) - 1.76;
  const lz = pz - colZ;
  const phi = Math.atan2(lz, lx === 0 && lz === 0 ? 1e-5 : lx);
  const flute = 0.015 * Math.abs(Math.sin(6.0 * phi));
  const rad = Math.hypot(lx, lz);
  const shaftR = 0.285 - flute;
  const dShaftCyl = Math.max(rad - shaftR, Math.abs(py - 0.04) - 1.54);
  const dPlinth = sdBox3(lx, py + 1.56, lz, 0.37, 0.2, 0.37) - 0.015;
  const dCapital = sdBox3(lx, py - 1.72, lz, 0.4, 0.14, 0.4) - 0.015;
  const dColumn = Math.min(dShaftCyl, Math.min(dPlinth, dCapital));

  // Longitudinal stone entablature beams overhead on both sides
  const dBeamSide = sdBox3(Math.abs(px) - 1.76, py - 2.16, pz - 11.4, 0.36, 0.3, 11.6);

  // Transverse stone ceiling rafters leaving open sunlit skylight bays between columns
  const dCrossBeam = sdBox3(px, py - 2.22, lz, 1.86, 0.24, 0.26);

  // Outer left warm limestone courtyard wall (right flank opens to warm sunlit sky)
  const dLeftWall = px + 4.5;

  // Distant monumental triumphal arch terminus at z = 22.8
  const dBackSlab = sdBox3(px, py - 0.42, pz - 22.8, 4.0, 2.9, 0.48);
  const dArchHole = Math.hypot(px, Math.max(0, py - 0.12)) - 1.28;
  const dBackArch = Math.max(dBackSlab, -dArchHole);

  return Math.min(
    dFloor,
    Math.min(dColumn, Math.min(dBeamSide, Math.min(dCrossBeam, Math.min(dLeftWall, dBackArch))))
  );
}

/**
 * World 1 — East (`+X`): Plunging Lapis Spiral Stairwell.
 * Deep vertical cylindrical tiled shaft with cantilevered helical stone stairs & brass handrail.
 */
export function evalEastStairwellSdfCpu(
  px: number,
  py: number,
  pz: number
): number {
  const shaftCz = 2.4;
  const shaftR = 2.3;
  const qx = px;
  const qz = pz - shaftCz;
  const r = Math.hypot(qx, qz);

  // Outer cylindrical glazed lapis masonry shaft wall breached cleanly by arched entry alcove
  const dCylWall = shaftR - r;
  const dAlcoveCut = sdBox3(px, py - 0.02, pz - 0.28, 1.56, 1.84, 0.72);
  const dWall = Math.max(dCylWall, -dAlcoveCut);

  // Narrow stone balcony threshold sill right at the portal entry (clears immediately into plunging abyss)
  const dLanding = sdBox3(px, py + 1.98, pz - 0.08, 1.62, 0.22, 0.36);

  // Cantilevered spiral stone stairs (16 discrete treads per 4.8 m full revolution, wide 2.21 m open oculus)
  // Descends smoothly rightward & downward from the portal threshold into subterranean depths
  const hTurn = 4.8;
  const nSteps = 16;
  const dTheta = (Math.PI * 2) / nSteps;
  const theta = Math.atan2(qz, qx === 0 && qz === 0 ? 1e-5 : qx);
  const thetaPhase = theta + Math.PI * 0.5;
  const yPhase = py + 1.72 + (thetaPhase / (Math.PI * 2)) * hTurn;
  const kWrap = Math.round(yPhase / hTurn);
  const baseIdx = Math.floor(thetaPhase / dTheta);

  let dStairs = 1e4;
  for (let offset = -1; offset <= 1; offset++) {
    const idx = baseIdx + offset;
    const stepThetaPhase = (idx + 0.5) * dTheta;
    let dAng = thetaPhase - stepThetaPhase;
    dAng = dAng - Math.round(dAng / (Math.PI * 2)) * (Math.PI * 2);
    const stepY = -1.72 - ((idx + 0.5) / nSteps) * hTurn + kWrap * hTurn;
    const yDiff = py - stepY;
    const yDiffWrapped = yDiff - Math.round(yDiff / hTurn) * hTurn;

    const rClamped = Math.max(1.1, r);
    const dRad = Math.abs(r - 1.7) - 0.58; // spans r in [1.12, 2.28], clear of z=0 aperture
    const dVert = Math.abs(yDiffWrapped) - 0.11;
    const dArc = Math.abs(dAng * rClamped) - (0.54 * dTheta * rClamped + 0.03);
    const dStep =
      Math.hypot(Math.max(dRad, 0), Math.max(dVert, 0), Math.max(dArc, 0)) +
      Math.min(Math.max(dRad, Math.max(dVert, dArc)), 0) -
      0.015;
    if (dStep < dStairs) dStairs = dStep;
  }

  // Sleek helical satin-brass inner handrail spiraling along r = 1.16
  const yRailRel = yPhase - kWrap * hTurn - 0.78;
  const yRailWrapped = yRailRel - Math.round(yRailRel / hTurn) * hTurn;
  const dRail = Math.hypot(r - 1.16, yRailWrapped) - 0.034;

  // Deep subterranean stone basin far below the cube's floor (-21 m)
  const dAbyssFloor = py + 21.0;

  return Math.min(dWall, Math.min(dLanding, Math.min(dStairs, Math.min(dRail, dAbyssFloor))));
}

/**
 * World 2 — South (`-Z`): Amber Lantern Over Dark Still Water.
 * Vaulted verdigris & raw-umber stone cistern, submerged arcade piers, dark water plane, and swaying lantern.
 */
export function evalSouthCisternSdfCpu(
  px: number,
  py: number,
  pz: number,
  timeSec = 0
): number {
  // Calm dark mirror cistern water plane seated just below portal sill (-1.734)
  const dWater = py + 1.76;

  // Barrel-vaulted cistern ceiling + side walls (springline y = 0.45, vault radius 2.60)
  const wallHalfW = 2.6;
  const dSideWalls = wallHalfW - Math.abs(px);
  const dVaultCeiling = 2.6 - Math.hypot(px, Math.max(0, py - 0.45));
  const dChamber = Math.min(dSideWalls, dVaultCeiling);

  // Repeating transverse stone vault ribs & submerged arcade piers every 3.2 m along +z
  const baySpacing = 3.2;
  const bayZ = Math.max(
    1.85,
    Math.min(21.05, Math.round((pz - 1.85) / baySpacing) * baySpacing + 1.85)
  );
  const lz = pz - bayZ;
  const dPier = sdBox3(Math.abs(px) - 1.94, py + 0.55, lz, 0.28, 1.65, 0.28) - 0.02;
  const archR = Math.hypot(px, py - 0.45);
  const dRib = Math.max(Math.abs(archR - 2.4) - 0.16, Math.abs(lz) - 0.22);
  const dRibUpper = Math.max(dRib, 0.4 - py);

  // Back cistern vaulted wall at z = 23.0
  const dBackWall = 23.0 - pz;

  // Gently swaying suspended Realgar-Amber brass lantern at z = 3.45
  const swayX = Math.sin(timeSec * 1.15) * 0.11;
  const swayZ = Math.cos(timeSec * 0.85) * 0.05;
  const lx = px - swayX;
  const ly = py - 0.22;
  const lzLantern = pz - (3.45 + swayZ);
  const dLanternBody = Math.max(Math.hypot(lx, lzLantern) - 0.24, Math.abs(ly) - 0.34) - 0.02;
  const dChain = Math.max(Math.hypot(lx, lzLantern) - 0.018, 0.32 - ly);

  return Math.min(
    dWater,
    Math.min(
      dChamber,
      Math.min(dPier, Math.min(dRibUpper, Math.min(dBackWall, Math.min(dLanternBody, dChain))))
    )
  );
}

/**
 * World 3 — West (`-X`): Warm Ochre Desert Slot-Canyon Arch.
 * Sculpted layered sandstone canyon walls framing a sunlit natural rock arch.
 */
export function evalWestCanyonSdfCpu(
  px: number,
  py: number,
  pz: number
): number {
  const depthRamp = smoothstepCpu(0.0, 2.2, pz);

  // Undulating wind-rippled ochre sandstone floor (strictly below portal sill at z = 0)
  const floorH =
    -1.78 +
    depthRamp * (0.14 * Math.sin(pz * 0.42) + 0.07 * Math.cos(px * 0.95));
  const dFloor = py - floorH;

  // Sinuous sculpted slot-canyon left & right sandstone walls (open mouth at z = 0, open sky slot above)
  const centerLineX = depthRamp * 0.52 * Math.sin(pz * 0.23 - 0.25);
  const relX = px - centerLineX;
  const strata = 0.075 * Math.sin(py * 5.0 + 0.55 * Math.sin(pz * 0.48));
  const billow =
    depthRamp *
    (0.42 * Math.cos(pz * 0.29) +
      0.42 * Math.sin(py * 0.68 + pz * 0.19) +
      0.18 * Math.cos(py * 1.32 - pz * 0.26));
  const mouthOpen = 0.72 * Math.exp(-Math.max(0, pz) * 0.55);
  const skyOpen = Math.max(0, py - 1.65) * 0.55;
  const canyonHalfW = 1.72 + mouthOpen + billow + strata + skyOpen;
  // Softly terminate canyon walls at rim height so pale Warm Kaolin sky crowns the arch
  const dWalls = Math.max(canyonHalfW - Math.abs(relX), py - 4.4);

  // Monumental natural sandstone rock arch spanning the slot canyon at z = 6.3
  const az = pz - 6.3;
  const archSpanX = px - 0.06;
  const archSpanY = py - 1.35;
  const bridgeCurveY = archSpanY + 0.27 * archSpanX * archSpanX - 0.07 * Math.sin(archSpanX * 2.1);
  const dBridge =
    sdBox3(archSpanX, bridgeCurveY, az, 2.95, 0.44 + 0.06 * Math.cos(archSpanX * 1.7), 0.64) -
    0.16;
  // Wide elliptical arch opening underneath framing sunlit distant canyon mesa & pale Kaolin sky
  const archVoid = Math.hypot((px - 0.06) / 1.5, (py + 0.35) / 1.58) - 1.0;
  const dArch = Math.max(dBridge, -archVoid * 1.2);

  // Distant sunlit sandstone mesa cliff beyond the natural arch (z = 21.8)
  const mesaRidge = 21.8 + 1.8 * Math.sin(px * 0.32) - Math.max(0, py - 1.2) * 0.45;
  const dMesa = Math.max(mesaRidge - pz, py - 4.8);

  const dRock = smin(dWalls, dArch, 0.45);
  return Math.min(dFloor, Math.min(dRock, dMesa));
}

/**
 * Evaluates the CPU reference SDF for any of the 4 cardinal portal worlds.
 */
export function evalPortalWorldSdfCpu(
  faceId: PortalFaceId,
  px: number,
  py: number,
  pz: number,
  timeSec = 0
): number {
  switch (faceId) {
    case 'north':
      return evalNorthColonnadeSdfCpu(px, py, pz);
    case 'east':
      return evalEastStairwellSdfCpu(px, py, pz);
    case 'south':
      return evalSouthCisternSdfCpu(px, py, pz, timeSec);
    case 'west':
      return evalWestCanyonSdfCpu(px, py, pz);
  }
}

export interface CpuRaymarchHit {
  hit: boolean;
  distance: number;
  steps: number;
  hitPos: [number, number, number];
  minSdf: number;
}

/**
 * CPU reference sphere-tracer matching the GPU TSL raymarcher's step budget and thresholds.
 */
export function raymarchPortalWorldCpu(
  faceId: PortalFaceId,
  ro: readonly [number, number, number],
  rd: readonly [number, number, number],
  timeSec = 0,
  maxSteps = 96,
  maxDist = 36
): CpuRaymarchHit {
  let t = 0.02;
  let minSdf = Infinity;

  for (let i = 0; i < maxSteps; i++) {
    const px = ro[0] + rd[0] * t;
    const py = ro[1] + rd[1] * t;
    const pz = ro[2] + rd[2] * t;

    const d = evalPortalWorldSdfCpu(faceId, px, py, pz, timeSec);
    if (d < minSdf) minSdf = d;

    if (d < 0.004) {
      return {
        hit: true,
        distance: t,
        steps: i + 1,
        hitPos: [px, py, pz],
        minSdf,
      };
    }

    t += Math.max(0.004, d * 0.78);
    if (t > maxDist) {
      return {
        hit: false,
        distance: maxDist,
        steps: i + 1,
        hitPos: [ro[0] + rd[0] * maxDist, ro[1] + rd[1] * maxDist, ro[2] + rd[2] * maxDist],
        minSdf,
      };
    }
  }

  return {
    hit: false,
    distance: t,
    steps: maxSteps,
    hitPos: [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t],
    minSdf,
  };
}
