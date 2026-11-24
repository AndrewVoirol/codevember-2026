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
 * World 0 — North (`+Z`): Sunlit Travertine Colonnade (`> 26 m` deep).
 * Classical fluted columns with Attic bases & molded capitals, side arcade arches,
 * coffered overhead beams, low right courtyard parapet, triumphal arch, and distant sunlit terrace.
 * Returns signed distance in portal-local meters.
 */
export function evalNorthColonnadeSdfCpu(
  px: number,
  py: number,
  pz: number
): number {
  // Seamless travertine gallery floor seated just below portal sill (-1.734)
  const dFloor = py + 1.76;

  // Low stepped stone threshold lip just inside the portal doorway (strictly behind z=0 & below sill)
  const dThresholdStep = sdBox3(px, py + 1.77, pz - 0.26, 1.52, 0.024, 0.17) - 0.006;

  // Repeating fluted classical columns along x = ±1.76, every 2.70 m along +z
  const colSpacing = 2.7;
  const colIdx = Math.round((pz - 1.85) / colSpacing);
  const colZ = Math.max(1.85, Math.min(20.75, colIdx * colSpacing + 1.85));
  const lx = Math.abs(px) - 1.76;
  const lz = pz - colZ;
  const phi = Math.atan2(lz, lx === 0 && lz === 0 ? 1e-5 : lx);
  const flute = 0.017 * Math.abs(Math.sin(6.0 * phi));
  const rad = Math.hypot(lx, lz);
  // Subtle classical entasis taper along the fluted shaft
  const shaftT = Math.max(-1, Math.min(1, (py - 0.04) / 1.54));
  const shaftR = 0.275 - 0.024 * (shaftT * 0.5 + 0.5) - flute;
  const dShaftCyl = Math.max(rad - shaftR, Math.abs(py - 0.04) - 1.52);

  // Molded Attic column base (square plinth + lower & upper rounded torus rings)
  const dPlinth = sdBox3(lx, py + 1.62, lz, 0.37, 0.14, 0.37) - 0.012;
  const dBaseTorus = Math.hypot(rad - 0.315, py + 1.43) - 0.055;

  // Classical Doric/Tuscan capital (necking astragal + flared echinus + square abacus)
  const dNeckRing = Math.hypot(rad - 0.275, py - 1.51) - 0.032;
  const dEchinus = Math.hypot(rad - 0.325, py - 1.60) - 0.068;
  const dAbacus = sdBox3(lx, py - 1.72, lz, 0.40, 0.085, 0.40) - 0.012;
  const dColumn = Math.min(
    dShaftCyl,
    Math.min(dPlinth, Math.min(dBaseTorus, Math.min(dNeckRing, Math.min(dEchinus, dAbacus))))
  );

  // Side colonnade arcade spandrels (carved semicircular arches spanning between every column pair!)
  const bayCenterZ = Math.max(
    3.2,
    Math.min(19.4, Math.round((pz - 3.2) / colSpacing) * colSpacing + 3.2)
  );
  const bLz = pz - bayCenterZ;
  const dArcadeWall = sdBox3(Math.abs(px) - 1.76, py - 2.12, pz - 11.4, 0.30, 0.44, 11.4);
  const dArcadeCut = Math.hypot(bLz, Math.max(0, py - 1.48)) - 0.96;
  const dSideArcade = Math.max(dArcadeWall, -dArcadeCut);

  // Molded cornice ledge & transverse coffered overhead stone beams leaving sunlit skylight bays
  const dCornice = sdBox3(Math.abs(px) - 1.76, py - 2.58, pz - 11.4, 0.42, 0.08, 11.55) - 0.01;
  const dCrossBeam = sdBox3(px, py - 2.34, lz, 1.88, 0.22, 0.24) - 0.01;

  // Left classical gallery wall with recessed vertical wall niches aligned with bays
  const dLeftWallRaw = px + 4.35;
  const dNicheCut = sdBox3(px + 4.35, py - 0.05, bLz, 0.38, 1.25, 0.72);
  const dLeftWall = Math.max(dLeftWallRaw, -dNicheCut);

  // Low sunlit right stone balustrade parapet opening outward to bright Mediterranean sky!
  const dRightParapet = sdBox3(px - 2.95, py + 1.28, pz - 11.4, 0.16, 0.48, 11.2) - 0.012;

  // Distant monumental triumphal arch terminus at z = 22.8 with stepped cornice
  const dBackSlab = sdBox3(px, py - 0.48, pz - 22.8, 4.3, 3.05, 0.50);
  const dArchHole = Math.hypot(px, Math.max(0, py - 0.26)) - 1.38;
  const dBackArch = Math.max(dBackSlab, -dArchHole);

  // Inviting sunlit terrace steps & freestanding classical rotunda pavilion visible THROUGH the triumphal arch!
  const dTerraceStep1 = sdBox3(px, py + 1.56, pz - 26.2, 5.5, 0.20, 2.2);
  const dTerraceStep2 = sdBox3(px, py + 1.32, pz - 28.4, 5.5, 0.20, 2.0);
  const dRotundaDome = Math.hypot(px, py - 0.68, pz - 30.2) - 1.05;
  const dRotundaDrum = Math.max(Math.hypot(px, pz - 30.2) - 0.96, Math.abs(py + 0.16) - 0.88);
  const dVista = Math.min(dTerraceStep1, Math.min(dTerraceStep2, Math.min(dRotundaDome, dRotundaDrum)));

  return Math.min(
    dFloor,
    Math.min(
      dThresholdStep,
      Math.min(
        dColumn,
        Math.min(
          dSideArcade,
          Math.min(
            dCornice,
            Math.min(dCrossBeam, Math.min(dLeftWall, Math.min(dRightParapet, Math.min(dBackArch, dVista))))
          )
        )
      )
    )
  );
}

/**
 * World 1 — East (`+X`): Plunging Lapis Spiral Stairwell.
 * Deep vertical cylindrical cobalt-glazed observatory shaft with 8 vertical pilasters,
 * cantilevered sculpted marble treads, vertical satin-brass balusters, helical handrail,
 * and threshold balcony landing.
 */
export function evalEastStairwellSdfCpu(
  px: number,
  py: number,
  pz: number
): number {
  const shaftCz = 2.4;
  const shaftR = 2.34;
  const qx = px;
  const qz = pz - shaftCz;
  const r = Math.hypot(qx, qz);
  const theta = Math.atan2(qz, qx === 0 && qz === 0 ? 1e-5 : qx);

  // Outer cylindrical glazed lapis shaft wall with 8 subtle vertical architectural pilasters
  const pilasterSector = (Math.PI * 2) / 8;
  const pilAng = theta - Math.round(theta / pilasterSector) * pilasterSector;
  const pilArc = Math.abs(pilAng) * r;
  const dPilaster = Math.max(r - (shaftR + 0.06), Math.max((shaftR - 0.095) - r, pilArc - 0.12));

  // Horizontal carved Bone-Calcite cornice ring ledges every full 4.8 m spiral revolution
  const hTurn = 4.8;
  const ringY = ((py + 0.72) % hTurn + hTurn * 1.5) % hTurn - hTurn * 0.5;
  const dCorniceRing = Math.max((shaftR - 0.085) - r, Math.abs(ringY) - 0.075);

  const dCylWall = Math.min(shaftR - r, Math.min(dPilaster, dCorniceRing));
  const dAlcoveCut = sdBox3(px, py - 0.02, pz - 0.28, 1.58, 1.88, 0.76);
  const dWall = Math.max(dCylWall, -dAlcoveCut);

  // Inviting cantilevered stone balcony landing at the portal threshold (clears z=0 & sill cleanly)
  const dLanding = sdBox3(px, py + 1.97, pz - 0.34, 1.56, 0.21, 0.27) - 0.012;

  // Cantilevered spiral stone stairs (16 discrete treads per 4.8 m full revolution, wide open central oculus)
  const nSteps = 16;
  const dTheta = (Math.PI * 2) / nSteps;
  const thetaPhase = theta + Math.PI * 0.5;
  const yPhase = py + 1.72 + (thetaPhase / (Math.PI * 2)) * hTurn;
  const kWrap = Math.round(yPhase / hTurn);
  const baseIdx = Math.floor(thetaPhase / dTheta);
  const rClamped = Math.max(1.12, r);

  let dStairs = 1e4;
  let dBalusters = 1e4;
  for (let offset = -1; offset <= 1; offset++) {
    const idx = baseIdx + offset;
    const stepThetaPhase = (idx + 0.5) * dTheta;
    let dAng = thetaPhase - stepThetaPhase;
    dAng = dAng - Math.round(dAng / (Math.PI * 2)) * (Math.PI * 2);
    const stepY = -1.72 - ((idx + 0.5) / nSteps) * hTurn + kWrap * hTurn;
    const yDiff = py - stepY;
    const yDiffWrapped = yDiff - Math.round(yDiff / hTurn) * hTurn;

    const dRad = Math.abs(r - 1.73) - 0.59; // spans r in [1.14, 2.32]
    const dVert = Math.abs(yDiffWrapped) - 0.088;
    const dArc = Math.abs(dAng * rClamped) - (0.45 * dTheta * rClamped + 0.015);
    const dStep =
      Math.hypot(Math.max(dRad, 0), Math.max(dVert, 0), Math.max(dArc, 0)) +
      Math.min(Math.max(dRad, Math.max(dVert, dArc)), 0) -
      0.012;
    if (dStep < dStairs) dStairs = dStep;

    // Slender vertical satin-brass baluster post rising from each tread nose up to the helical handrail
    const balArc = dAng * 1.16;
    const balRad = r - 1.16;
    const balH = Math.abs(yDiffWrapped - 0.44) - 0.36;
    const dBal = Math.max(Math.hypot(balRad, balArc) - 0.018, balH);
    if (dBal < dBalusters) dBalusters = dBal;
  }

  // Sleek helical satin-brass inner handrail spiraling along r = 1.16
  const yRailRel = yPhase - kWrap * hTurn - 0.78;
  const yRailWrapped = yRailRel - Math.round(yRailRel / hTurn) * hTurn;
  const dRail = Math.hypot(r - 1.16, yRailWrapped) - 0.034;

  // Deep subterranean illuminated mineral basin far below the cube's floor (-22 m)
  const dAbyssFloor = py + 22.0;

  return Math.min(
    dWall,
    Math.min(dLanding, Math.min(dStairs, Math.min(dBalusters, Math.min(dRail, dAbyssFloor))))
  );
}

/**
 * World 2 — South (`-Z`): Vaulted Verdigris Basilica Cistern & Receding Amber Lanterns.
 * Flooded vaulted Romanesque cistern with sculpted piers & capitals, transverse vault ribs,
 * foreground wet limestone threshold quay, mirror water plane, and three receding brass lanterns.
 */
export function evalSouthCisternSdfCpu(
  px: number,
  py: number,
  pz: number,
  timeSec = 0
): number {
  // Calm dark mirror cistern water plane seated just below portal sill (-1.734)
  const dWater = py + 1.76;

  // Two-step wet carved limestone threshold quay descending cleanly into the dark mirror water
  const dQuayTop = sdBox3(px, py + 1.855, pz - 0.44, 1.42, 0.105, 0.32) - 0.012;
  const dQuayLow = sdBox3(px, py + 1.915, pz - 0.78, 1.24, 0.145, 0.24) - 0.012;
  const dQuay = Math.min(dQuayTop, dQuayLow);

  // Barrel-vaulted cistern ceiling + side walls (springline y = 0.45, vault radius 2.62)
  const wallHalfW = 2.62;
  const dSideWalls = wallHalfW - Math.abs(px);
  const dVaultCeiling = 2.62 - Math.hypot(px, Math.max(0, py - 0.45));
  const dChamber = Math.min(dSideWalls, dVaultCeiling);

  // Repeating Romanesque arcade piers (plinth + shaft + flared capital) & transverse vault ribs every 3.2 m
  const baySpacing = 3.2;
  const bayZ = Math.max(
    1.85,
    Math.min(21.05, Math.round((pz - 1.85) / baySpacing) * baySpacing + 1.85)
  );
  const lz = pz - bayZ;
  const pierX = Math.abs(px) - 1.88;
  const dPierShaft = sdBox3(pierX, py + 0.62, lz, 0.25, 1.56, 0.25) - 0.018;
  const dPierBase = sdBox3(pierX, py + 1.72, lz, 0.34, 0.22, 0.34) - 0.015;
  const dPierCap = sdBox3(pierX, py - 0.88, lz, 0.35, 0.12, 0.35) - 0.015;
  const dPier = Math.min(dPierShaft, Math.min(dPierBase, dPierCap));

  // Transverse stone vault ribs springing cleanly from pier capitals across the vaulted nave
  const archR = Math.hypot(px, py - 0.45);
  const dRib = Math.max(Math.abs(archR - 2.38) - 0.15, Math.abs(lz) - 0.20);
  const dRibUpper = Math.max(dRib, 0.36 - py);

  // Longitudinal side-aisle arches connecting consecutive piers along x = ±1.88
  const midBayZ = Math.max(
    3.45,
    Math.min(19.45, Math.round((pz - 3.45) / baySpacing) * baySpacing + 3.45)
  );
  const dSideBeam = sdBox3(pierX, py - 1.24, pz - 11.4, 0.24, 0.42, 10.6);
  const dSideArchVoid = Math.hypot(pz - midBayZ, Math.max(0, py - 0.78)) - 1.12;
  const dSideArcade = Math.max(dSideBeam, -dSideArchVoid);

  // Back vaulted sanctuary apse wall at z = 23.0 with recessed arched niche
  const dBackWall = 23.0 - pz;

  // Three receding suspended Realgar-Amber filigree brass lanterns along the vaulted nave
  const evalLantern = (cx: number, cy: number, cz: number) => {
    const lx = px - cx;
    const ly = py - cy;
    const lzL = pz - cz;
    const rL = Math.hypot(lx, lzL);
    const dGlass = Math.max(rL - 0.195, Math.abs(ly) - 0.27) - 0.015;
    const dCap = Math.max(rL - 0.285, Math.abs(ly - 0.31) - 0.055) - 0.014;
    const dDome = Math.hypot(lx, (ly - 0.39) * 1.35, lzL) - 0.17;
    const dBase = Math.max(rL - 0.255, Math.abs(ly + 0.30) - 0.042) - 0.012;
    const dFinial = Math.hypot(lx, (ly + 0.40) * 0.9, lzL) - 0.062;
    const dChain = Math.max(rL - 0.016, 0.44 - ly);
    return Math.min(dGlass, Math.min(dCap, Math.min(dDome, Math.min(dBase, Math.min(dFinial, dChain)))));
  };

  const swayX1 = Math.sin(timeSec * 1.15) * 0.10;
  const swayZ1 = Math.cos(timeSec * 0.85) * 0.045;
  const dLantern1 = evalLantern(swayX1, 0.15, 3.45 + swayZ1);
  const dLantern2 = evalLantern(-0.24 + Math.sin(timeSec * 0.82 + 1.7) * 0.06, 0.46, 9.65);
  const dLantern3 = evalLantern(0.22 + Math.sin(timeSec * 0.65 + 3.1) * 0.045, 0.72, 15.85);

  return Math.min(
    dWater,
    Math.min(
      dQuay,
      Math.min(
        dChamber,
        Math.min(
          dPier,
          Math.min(
            dRibUpper,
            Math.min(
              dSideArcade,
              Math.min(dBackWall, Math.min(dLantern1, Math.min(dLantern2, dLantern3)))
            )
          )
        )
      )
    )
  );
}

/**
 * World 3 — West (`-X`): Crimson Sandstone Slot Canyon & Golden-Hour Natural Arch.
 * Lipschitz-safe sculpted Navajo/Antelope slot-canyon sandstone waves, clean parabolic
 * natural rock bridge (strictly clearing floor & zero raymarch tears), and terraced desert buttes.
 */
export function evalWestCanyonSdfCpu(
  px: number,
  py: number,
  pz: number
): number {
  const depthRamp = smoothstepCpu(0.0, 2.2, pz);

  // Winding wind-rippled desert sand floor (strictly below portal sill at z = 0)
  const floorH =
    -1.78 +
    depthRamp * (0.11 * Math.sin(pz * 0.36) + 0.055 * Math.cos(px * 0.82 - pz * 0.2));
  const dFloor = (py - floorH) * 0.94;

  // Sweeping water-carved Antelope/Navajo slot-canyon walls flaring smoothly open toward the sky
  const centerLineX = depthRamp * (0.54 * Math.sin(pz * 0.21 - 0.35));
  const relX = px - centerLineX;
  const strataLedge = 0.042 * Math.sin(py * 4.0 + 0.42 * Math.sin(pz * 0.40));
  const sweepWave =
    depthRamp *
    (0.32 * Math.cos(pz * 0.26) +
      0.30 * Math.sin(py * 0.68 + pz * 0.20) +
      0.13 * Math.cos(py * 1.12 - pz * 0.22));
  const mouthOpen = 0.66 * Math.exp(-Math.max(0, pz) * 0.56);
  const rimT = Math.max(0, py - 1.35);
  const skyFlare = rimT * 0.48 + rimT * rimT * 0.28;
  const basinOpen = smoothstepCpu(8.2, 15.0, pz) * 2.65;
  const canyonHalfW = 1.66 + mouthOpen + sweepWave + strataLedge + skyFlare + basinOpen;
  const dWalls = (canyonHalfW - Math.abs(relX)) * 0.66;

  // Monumental natural sandstone rock arch spanning cleanly over the canyon at z = 6.45
  const ax = px - 0.06;
  const az = pz - 6.45;
  const archCrownY = 1.44 - 0.27 * ax * ax + 0.05 * Math.sin(ax * 1.8);
  const archHalfThickY = 0.36 + 0.065 * Math.cos(ax * 1.4) + 0.025 * Math.sin(az * 2.0);
  const archHalfThickZ = 0.58 + 0.07 * Math.cos(ax * 1.1);
  const dArchCore = sdBox3(ax, py - archCrownY, az, 3.35, archHalfThickY, archHalfThickZ) - 0.14;
  const dArch = dArchCore * 0.66;

  // Terraced golden-hour sandstone monument buttes & distant mesa ridge framing the vista beyond the arch
  const butteLeft = sdBox3(px + 2.55, py + 0.25, pz - 14.8, 1.35, 2.35, 1.65) - 0.32;
  const butteRight = sdBox3(px - 2.95, py - 0.05, pz - 16.4, 1.45, 2.75, 1.85) - 0.35;
  const mesaRidgeY = 2.65 + 0.65 * Math.sin(px * 0.36 + 0.4) + 0.28 * Math.cos(px * 0.92);
  const dMesaBack = sdBox3(px, py - (mesaRidgeY * 0.5 - 0.9), pz - 21.8, 12.0, mesaRidgeY * 0.5 + 1.1, 1.8) - 0.35;
  const dMesas = Math.min(dMesaBack, Math.min(butteLeft, butteRight));

  const dRock = smin(dWalls, dArch, 0.38);
  return Math.min(dFloor, Math.min(dRock, dMesas));
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
  const stepScale = faceId === 'west' ? 0.92 : 0.84;

  for (let i = 0; i < maxSteps; i++) {
    const px = ro[0] + rd[0] * t;
    const py = ro[1] + rd[1] * t;
    const pz = ro[2] + rd[2] * t;

    const d = evalPortalWorldSdfCpu(faceId, px, py, pz, timeSec);
    if (d < minSdf) minSdf = d;

    const hitEps = faceId === 'west' ? Math.max(0.004, t * 0.0016) : 0.004;
    if (d < hitEps) {
      return {
        hit: true,
        distance: t,
        steps: i + 1,
        hitPos: [px, py, pz],
        minSdf,
      };
    }

    t += Math.max(hitEps, d * stepScale);
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
