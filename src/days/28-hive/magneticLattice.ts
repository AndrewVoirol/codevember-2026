/**
 * Day 28 - Hive: Coupled 2D Magnetic Dipole Honeycomb Lattice ("Hive Mind")
 *
 * Pure zero-GC physics simulation of N = 91 (R = 5 axial hex radius) jewel-pivoted
 * planar magnetic dipoles interacting via exact all-pairs long-range 1/r^3
 * magnetostatic dipole-dipole coupling plus a movable 3D external lodestone stylus.
 *
 * Convention (matches Three.js right-handed +Y rotation):
 *   Needle orientation `theta` -> rotation.y = theta
 *   Unit dipole vector in XZ plane (South -> North):
 *     m = (cos(theta), -sin(theta))
 *   Potential energy U = -m . B = -(Bx * cos(theta) - Bz * sin(theta))
 *   Restoring torque tau = -dU/dtheta = -Bx * sin(theta) - Bz * cos(theta)
 */

export const HEX_RADIUS = 5;
export const CELL_CIRCUMRADIUS = 0.128;
export const CELL_INRADIUS = CELL_CIRCUMRADIUS * (Math.sqrt(3) * 0.5); // ~0.11085
export const CELL_PITCH = CELL_CIRCUMRADIUS * Math.sqrt(3); // ~0.22170
export const WALL_HALF_THICKNESS = 0.0095;
export const DECK_FLOOR_Y = 0.0;
export const NEEDLE_PIVOT_Y = 0.060;

export interface HexCell {
  readonly id: number;
  readonly q: number;
  readonly r: number;
  readonly x: number;
  readonly z: number;
  /** Ring distance from center: 0..HEX_RADIUS */
  readonly ring: number;
  /** Neighbor cell indices (length 3..6), -1 where absent in direction d ∈ [0..5] */
  readonly neighborByDir: readonly [number, number, number, number, number, number];
}

export interface HexLatticeLayout {
  readonly radius: number;
  readonly cellCount: number;
  readonly cellCircumradius: number;
  readonly cellInradius: number;
  readonly cellPitch: number;
  readonly cells: readonly HexCell[];
  readonly maxVertexRadius: number;
  readonly maxFlatExtent: number;
}

/**
 * Six pointy-top axial neighbor directions ordered to align with the 6 cell edges.
 * Edge k connects corner k to corner (k + 1) % 6, where corner k is at angle (30° + 60°*k)
 * in standard (x, z) coordinates: (cos(a), sin(a)).
 *
 * Edge midpoints sit at angles:
 *   k=0:  60° -> (+0.5, +sqrt(3)/2) -> (dq= 0, dr=+1)
 *   k=1: 120° -> (-0.5, +sqrt(3)/2) -> (dq=-1, dr=+1)
 *   k=2: 180° -> (-1.0,  0.0)       -> (dq=-1, dr= 0)
 *   k=3: 240° -> (-0.5, -sqrt(3)/2) -> (dq= 0, dr=-1)
 *   k=4: 300° -> (+0.5, -sqrt(3)/2) -> (dq=+1, dr=-1)
 *   k=5:   0° -> (+1.0,  0.0)       -> (dq=+1, dr= 0)
 */
export const AXIAL_EDGE_DIRS: readonly (readonly [number, number])[] = [
  [0, 1],
  [-1, 1],
  [-1, 0],
  [0, -1],
  [1, -1],
  [1, 0],
];

/**
 * Builds a deterministic pointy-top axial hex lattice of radius R (N = 1 + 3*R*(R+1)).
 * A pointy-top cell lattice naturally packs into a flat-topped regular hexagonal outline.
 */
export function buildHexLatticeLayout(
  radius: number = HEX_RADIUS,
  cellCircumradius: number = CELL_CIRCUMRADIUS
): HexLatticeLayout {
  const sqrt3 = Math.sqrt(3);
  const cellInradius = cellCircumradius * (sqrt3 * 0.5);
  const cellPitch = cellCircumradius * sqrt3;

  const coordToId = new Map<string, number>();
  const rawCells: Array<{ id: number; q: number; r: number; x: number; z: number; ring: number }> = [];

  let nextId = 0;
  for (let r = -radius; r <= radius; r++) {
    const qMin = Math.max(-radius, -r - radius);
    const qMax = Math.min(radius, -r + radius);
    for (let q = qMin; q <= qMax; q++) {
      const x = cellCircumradius * sqrt3 * (q + r * 0.5);
      const z = cellCircumradius * 1.5 * r;
      const ring = Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r));
      const id = nextId++;
      coordToId.set(`${q},${r}`, id);
      rawCells.push({ id, q, r, x, z, ring });
    }
  }

  let maxVertexRadius = 0;
  let maxFlatExtent = 0;

  const cells: HexCell[] = rawCells.map((c) => {
    const nb: [number, number, number, number, number, number] = [-1, -1, -1, -1, -1, -1];
    for (let d = 0; d < 6; d++) {
      const nq = c.q + AXIAL_EDGE_DIRS[d][0];
      const nr = c.r + AXIAL_EDGE_DIRS[d][1];
      const found = coordToId.get(`${nq},${nr}`);
      nb[d] = found !== undefined ? found : -1;

      // Measure outer extents from pointy-top vertices
      const ang = (Math.PI / 6) + d * (Math.PI / 3);
      const vx = c.x + Math.cos(ang) * cellCircumradius;
      const vz = c.z + Math.sin(ang) * cellCircumradius;
      const rad = Math.hypot(vx, vz);
      if (rad > maxVertexRadius) maxVertexRadius = rad;
      if (Math.abs(vz) > maxFlatExtent) maxFlatExtent = Math.abs(vz);
    }
    return {
      id: c.id,
      q: c.q,
      r: c.r,
      x: c.x,
      z: c.z,
      ring: c.ring,
      neighborByDir: nb,
    };
  });

  return {
    radius,
    cellCount: cells.length,
    cellCircumradius,
    cellInradius,
    cellPitch,
    cells,
    maxVertexRadius,
    maxFlatExtent,
  };
}

export interface StylusFieldState {
  active: boolean;
  /** Local XZ position above the honeycomb deck */
  x: number;
  z: number;
  /** Height of stylus magnetic tip above needle pivot plane */
  height: number;
  /** Magnetic polarity: +1 attracts North (Cinnabar) tip, -1 repels North / attracts South */
  polarity: 1 | -1;
  /** Field strength multiplier (1.0 hover, ~2.8 press-drag) */
  strength: number;
  /** Pointer XZ velocity for tangential induction swirl */
  vx: number;
  vz: number;
}

export interface LatticeTelemetry {
  cellCount: number;
  /** Global ferromagnetic order parameter R in [0, 1] */
  globalOrder: number;
  /** Mean local neighbor alignment in [-1, 1] */
  localCoherence: number;
  /** Mean absolute angular velocity across all needles (rad/s) */
  meanSpeed: number;
  /** Peak angular velocity across all needles (rad/s) */
  peakSpeed: number;
  /** Total magnetostatic dipole interaction energy (dimensionless) */
  dipoleEnergy: number;
  /** Count of detected topological flux vortices (|winding| == 1) on interior hex plaquettes */
  vortexCount: number;
  /** Count of settled static-friction locked needles */
  settledCount: number;
  stylusPolarity: 1 | -1;
}

/**
 * Wraps an angle in radians into [-PI, PI).
 */
export function wrapAngle(a: number): number {
  let r = (a + Math.PI) % (Math.PI * 2);
  if (r < 0) r += Math.PI * 2;
  return r - Math.PI;
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Symplectic coupled magnetic dipole simulator over all 91 honeycomb cells.
 * Precomputes the exact symmetric pairwise 1/r^3 dipole tensor T_xx, T_xz, T_zz once at init
 * so every 240 Hz substep performs zero heap allocations and zero Math.sqrt calls.
 */
export class MagneticDipoleLattice {
  readonly layout: HexLatticeLayout;
  readonly count: number;

  // Per-cell dynamic state
  readonly angles: Float32Array;
  readonly omegas: Float32Array;
  readonly torques: Float32Array;
  /** Jewel-pivot 3D micro-nutation wobble angles (rad) */
  readonly wobblePitch: Float32Array;
  readonly wobbleRoll: Float32Array;
  readonly wobblePitchVel: Float32Array;
  readonly wobbleRollVel: Float32Array;
  /** Instantaneous dipole unit vector components */
  readonly mx: Float32Array;
  readonly mz: Float32Array;
  /** Total field at each cell */
  readonly bx: Float32Array;
  readonly bz: Float32Array;
  /** Per-cell dipole alignment score [0..1] for subtle optical/diagnostic telemetry */
  readonly localAlignment: Float32Array;
  /** Static-friction jewel-bearing rest latch (0 = moving, 1 = seated at rest) */
  readonly locked: Uint8Array;

  // Precomputed upper-triangle pairwise 1/r^3 dipole tensor (length = N*(N-1)/2)
  private readonly pairCount: number;
  private readonly pairI: Uint16Array;
  private readonly pairJ: Uint16Array;
  private readonly tensorXX: Float32Array;
  private readonly tensorXZ: Float32Array;
  private readonly tensorZZ: Float32Array;

  // Precomputed 6-cell hex plaquettes for exact topological vortex winding detection
  private readonly plaquetteRings: Int16Array;
  private readonly plaquetteCount: number;

  // Precomputed mild per-cell magnetic moment & pivot-bearing dispersion (physical craft realism)
  private readonly momentScale: Float32Array;
  private readonly dampingScale: Float32Array;

  // Physical tuning constants
  /** Normalized dipole-dipole coupling constant (scaled by pitch^3 so nearest-neighbor coupling ~ K_DIPOLE) */
  readonly dipoleCoupling = 26.0;
  /** Viscous jewel-pivot + eddy-current brass-wall damping (rad/s^-1; zeta ~ 0.14 for crisp spin waves) */
  readonly baseDamping = 3.15;
  /** Coulomb jewel-pivot static friction torque threshold (~0.42° deadband at B ~ 130, zero idle chatter) */
  readonly staticFrictionTorque = 0.95;
  readonly staticSpeedThreshold = 0.045;
  /** Subtle hexagonal crystal-field anisotropy from the 6 brass chamber walls (very weak) */
  readonly hexWallAnisotropy = 0.16;

  readonly stylus: StylusFieldState = {
    active: false,
    x: 0.72,
    z: -0.68,
    height: 0.34,
    polarity: 1,
    strength: 1.0,
    vx: 0,
    vz: 0,
  };

  // Internal impulse pulse ring (click shockwave)
  private pulseActive = false;
  private pulseX = 0;
  private pulseZ = 0;
  private pulseAge = 0;
  private pulseStrength = 0;
  private pulsePolarity: 1 | -1 = 1;
  private pulseSwirl = 0;

  private rng: () => number;
  private simTime = 0;

  constructor(layout: HexLatticeLayout = buildHexLatticeLayout(), seed = 202628) {
    this.layout = layout;
    this.count = layout.cellCount;
    this.rng = mulberry32(seed);

    this.angles = new Float32Array(this.count);
    this.omegas = new Float32Array(this.count);
    this.torques = new Float32Array(this.count);
    this.wobblePitch = new Float32Array(this.count);
    this.wobbleRoll = new Float32Array(this.count);
    this.wobblePitchVel = new Float32Array(this.count);
    this.wobbleRollVel = new Float32Array(this.count);
    this.mx = new Float32Array(this.count);
    this.mz = new Float32Array(this.count);
    this.bx = new Float32Array(this.count);
    this.bz = new Float32Array(this.count);
    this.localAlignment = new Float32Array(this.count);
    this.locked = new Uint8Array(this.count);
    this.momentScale = new Float32Array(this.count);
    this.dampingScale = new Float32Array(this.count);

    for (let i = 0; i < this.count; i++) {
      this.momentScale[i] = 0.94 + this.rng() * 0.12;
      this.dampingScale[i] = 0.90 + this.rng() * 0.20;
    }

    // Precompute exact all-pairs dipole interaction tensor T(i, j) scaled by pitch^3
    const n = this.count;
    this.pairCount = (n * (n - 1)) / 2;
    this.pairI = new Uint16Array(this.pairCount);
    this.pairJ = new Uint16Array(this.pairCount);
    this.tensorXX = new Float32Array(this.pairCount);
    this.tensorXZ = new Float32Array(this.pairCount);
    this.tensorZZ = new Float32Array(this.pairCount);

    const pitch3 = Math.pow(layout.cellPitch, 3);
    let pIdx = 0;
    for (let i = 0; i < n; i++) {
      const ci = layout.cells[i];
      for (let j = i + 1; j < n; j++) {
        const cj = layout.cells[j];
        const dx = ci.x - cj.x;
        const dz = ci.z - cj.z;
        const d2 = dx * dx + dz * dz;
        const invD = 1 / Math.sqrt(d2);
        const ux = dx * invD;
        const uz = dz * invD;
        const invD3Scaled = pitch3 * invD * invD * invD;

        this.pairI[pIdx] = i;
        this.pairJ[pIdx] = j;
        this.tensorXX[pIdx] = (3 * ux * ux - 1) * invD3Scaled;
        this.tensorXZ[pIdx] = (3 * ux * uz) * invD3Scaled;
        this.tensorZZ[pIdx] = (3 * uz * uz - 1) * invD3Scaled;
        pIdx++;
      }
    }

    // Collect every interior cell that has all 6 neighbors (forms a closed 6-cell ring)
    const rings: number[] = [];
    for (let i = 0; i < n; i++) {
      const nb = layout.cells[i].neighborByDir;
      let full = true;
      for (let d = 0; d < 6; d++) {
        if (nb[d] < 0) {
          full = false;
          break;
        }
      }
      if (full) {
        for (let d = 0; d < 6; d++) rings.push(nb[d]);
      }
    }
    this.plaquetteCount = rings.length / 6;
    this.plaquetteRings = new Int16Array(rings);

    // Seed with the signature twin-vortex flux-closure state (matching locked hero concept)
    this.seedTwinVortexEquilibrium();
  }

  /**
   * Initializes a rich twin-vortex closed-loop flux configuration (counter-rotating left/right
   * magnetic vortices separated by a central vertical flux river) and relaxes it slightly under
   * real dipole-dipole dynamics so the needles open in a physically self-consistent state.
   */
  seedTwinVortexEquilibrium(): void {
    const v1x = -0.46;
    const v1z = 0.04;
    const v2x = 0.46;
    const v2z = -0.04;

    for (let i = 0; i < this.count; i++) {
      const c = this.layout.cells[i];

      // Superpose two counter-rotating tangential vortex fields (CW left, CCW right)
      const dx1 = c.x - v1x;
      const dz1 = c.z - v1z;
      const r1Sq = dx1 * dx1 + dz1 * dz1 + 0.085;
      const fx1 = -dz1 / r1Sq;
      const fz1 = dx1 / r1Sq;

      const dx2 = c.x - v2x;
      const dz2 = c.z - v2z;
      const r2Sq = dx2 * dx2 + dz2 * dz2 + 0.085;
      const fx2 = dz2 / r2Sq;
      const fz2 = -dx2 / r2Sq;

      const fx = fx1 + fx2;
      const fz = fz1 + fz2;
      // Convert vector (fx, fz) to Three.js rotation.y angle where m = (cos(theta), -sin(theta))
      const theta = Math.atan2(-fz, fx);
      this.angles[i] = wrapAngle(theta + (this.rng() - 0.5) * 0.14);
      this.omegas[i] = (this.rng() - 0.5) * 0.6;
      this.wobblePitch[i] = 0;
      this.wobbleRoll[i] = 0;
      this.wobblePitchVel[i] = 0;
      this.wobbleRollVel[i] = 0;
      this.locked[i] = 0;
    }

    // Relax for 0.65 s with real dipole coupling (stylus inactive) to settle into a genuine
    // self-organized twin-vortex domain state with gentle opening shimmer still alive
    const prevActive = this.stylus.active;
    this.stylus.active = false;
    this.pulseActive = false;
    for (let k = 0; k < 40; k++) {
      this.step(1 / 60);
    }
    this.stylus.active = prevActive;
  }

  /**
   * Seeds a single macroscopic swirling flux vortex across the entire 91-cell hexagon.
   */
  seedMacroVortex(handedness: 1 | -1 = 1): void {
    for (let i = 0; i < this.count; i++) {
      const c = this.layout.cells[i];
      const fx = -c.z * handedness;
      const fz = c.x * handedness;
      const theta = Math.hypot(fx, fz) > 1e-4 ? Math.atan2(-fz, fx) : 0;
      this.angles[i] = wrapAngle(theta + (this.rng() - 0.5) * 0.08);
      this.omegas[i] = handedness * 0.9;
      this.locked[i] = 0;
    }
  }

  /**
   * Thermal agitation / demagnetizing scatter: imparts above-barrier angular & phase kicks so
   * all 91 needles break out of the current domain basin and self-organize into fresh vortices.
   */
  triggerThermalScatter(intensity = 1.0): void {
    const scale = Math.max(0.25, intensity);
    const kick = 29.0 * scale;
    const phaseKick = 2.1 * Math.min(1.5, scale);
    for (let i = 0; i < this.count; i++) {
      this.angles[i] = wrapAngle(this.angles[i] + (this.rng() - 0.5) * 2.0 * phaseKick);
      this.omegas[i] += (this.rng() - 0.5) * 2.0 * kick;
      this.wobblePitchVel[i] += (this.rng() - 0.5) * 4.5;
      this.wobbleRollVel[i] += (this.rng() - 0.5) * 4.5;
      this.locked[i] = 0;
    }
  }

  /**
   * Fires a localized radial magnetic shock pulse from (x, z) that ripples outward
   * through dipole-dipole coupling across the honeycomb.
   */
  triggerPulse(
    x: number,
    z: number,
    polarity: 1 | -1 = this.stylus.polarity,
    strength = 1.0,
    swirl = 0.55
  ): void {
    this.pulseActive = true;
    this.pulseX = x;
    this.pulseZ = z;
    this.pulseAge = 0;
    this.pulseStrength = strength;
    this.pulsePolarity = polarity;
    this.pulseSwirl = swirl;

    // Immediate localized angular + wobble kick at impact cells so response is crisp on frame 0
    for (let i = 0; i < this.count; i++) {
      const c = this.layout.cells[i];
      const dx = c.x - x;
      const dz = c.z - z;
      const dist = Math.hypot(dx, dz);
      if (dist < 0.58) {
        const falloff = Math.exp(-(dist * dist) / (2 * 0.24 * 0.24));
        const swirlKick = swirl * polarity * 11.5 * strength * falloff;
        this.omegas[i] += swirlKick;
        this.wobblePitchVel[i] += (dz / Math.max(0.06, dist)) * 3.2 * falloff * strength;
        this.wobbleRollVel[i] -= (dx / Math.max(0.06, dist)) * 3.2 * falloff * strength;
        this.locked[i] = 0;
      }
    }
  }

  /**
   * Flips the external lodestone stylus polarity (+1 <-> -1).
   */
  flipStylusPolarity(): 1 | -1 {
    this.stylus.polarity = this.stylus.polarity === 1 ? -1 : 1;
    // Wake nearby needles so polarity flip has immediate tactile feedback
    for (let i = 0; i < this.count; i++) {
      this.locked[i] = 0;
    }
    return this.stylus.polarity;
  }

  /**
   * Finds the nearest hexagonal cell index to a local XZ point, or -1 if outside tolerance.
   */
  findNearestCell(x: number, z: number, maxDist = this.layout.cellCircumradius * 1.15): number {
    let bestId = -1;
    let bestD2 = maxDist * maxDist;
    for (let i = 0; i < this.count; i++) {
      const c = this.layout.cells[i];
      const dx = c.x - x;
      const dz = c.z - z;
      const d2 = dx * dx + dz * dz;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestId = i;
      }
    }
    return bestId;
  }

  /**
   * Evaluates total magnetic field (all-pairs dipole + external stylus + active shock pulse)
   * and restoring torque for every needle without heap allocations.
   */
  private computeFieldsAndTorques(): void {
    const n = this.count;
    const angles = this.angles;
    const mx = this.mx;
    const mz = this.mz;
    const bx = this.bx;
    const bz = this.bz;
    const mScale = this.momentScale;

    for (let i = 0; i < n; i++) {
      const th = angles[i];
      mx[i] = Math.cos(th);
      mz[i] = -Math.sin(th);
      bx[i] = 0;
      bz[i] = 0;
    }

    // 1. Exact symmetric all-pairs 1/r^3 dipole-dipole field accumulation
    const pCount = this.pairCount;
    const pairI = this.pairI;
    const pairJ = this.pairJ;
    const tXX = this.tensorXX;
    const tXZ = this.tensorXZ;
    const tZZ = this.tensorZZ;

    for (let p = 0; p < pCount; p++) {
      const i = pairI[p];
      const j = pairJ[p];
      const xx = tXX[p];
      const xz = tXZ[p];
      const zz = tZZ[p];

      const mjx = mx[j] * mScale[j];
      const mjz = mz[j] * mScale[j];
      bx[i] += xx * mjx + xz * mjz;
      bz[i] += xz * mjx + zz * mjz;

      const mix = mx[i] * mScale[i];
      const miz = mz[i] * mScale[i];
      bx[j] += xx * mix + xz * miz;
      bz[j] += xz * mix + zz * miz;
    }

    // Scale dipole field by coupling constant
    const kDip = this.dipoleCoupling;
    for (let i = 0; i < n; i++) {
      bx[i] *= kDip;
      bz[i] *= kDip;
    }

    // 2. External hovering 3D lodestone stylus field (magnetic pole + drag-velocity swirl induction)
    if (this.stylus.active) {
      const sx = this.stylus.x;
      const sz = this.stylus.z;
      const sh = Math.max(0.14, this.stylus.height);
      const sh2 = sh * sh;
      const pol = this.stylus.polarity;
      const amp = 44.0 * this.stylus.strength;
      const vx = this.stylus.vx;
      const vz = this.stylus.vz;
      const speed = Math.hypot(vx, vz);

      for (let i = 0; i < n; i++) {
        const c = this.layout.cells[i];
        // Vector from needle to stylus pole (so polarity = +1 attracts North/Cinnabar tip toward wand)
        const rx = sx - c.x;
        const rz = sz - c.z;
        const r2 = rx * rx + rz * rz + sh2;
        const invR = 1 / Math.sqrt(r2);
        const invR3 = invR * invR * invR;

        // Radial monopole-like horizontal component of vertical lodestone tip
        const radGain = amp * pol * invR3 * 0.24;
        bx[i] += rx * radGain;
        bz[i] += rz * radGain;

        // Velocity-coupled inductive drag field: sweeping the wand combs needles smoothly
        // in the direction of motion, while circular dragging winds macroscopic vortices!
        if (speed > 0.02) {
          const velGain = Math.min(28.0, speed * 9.5) * this.stylus.strength * Math.exp(-(rx * rx + rz * rz) / 0.42);
          bx[i] += (vx / speed) * velGain;
          bz[i] += (vz / speed) * velGain;
        }
      }
    }

    // 3. Expanding magnetic shock pulse ring (from click / shock excitation)
    if (this.pulseActive) {
      const ringRadius = this.pulseAge * 2.4;
      const ringWidth = 0.28;
      const env = Math.exp(-this.pulseAge * 3.4) * this.pulseStrength;
      const px = this.pulseX;
      const pz = this.pulseZ;
      const pol = this.pulsePolarity;
      const sw = this.pulseSwirl;

      for (let i = 0; i < n; i++) {
        const c = this.layout.cells[i];
        const rx = c.x - px;
        const rz = c.z - pz;
        const d = Math.hypot(rx, rz);
        const dr = d - ringRadius;
        const band = Math.exp(-(dr * dr) / (2 * ringWidth * ringWidth)) * env;
        if (band > 1e-4) {
          const invD = 1 / Math.max(0.06, d);
          const ux = rx * invD;
          const uz = rz * invD;
          // Tangential swirl + radial kick
          const fx = (ux * 0.65 - uz * sw) * pol * 95.0 * band;
          const fz = (uz * 0.65 + ux * sw) * pol * 95.0 * band;
          bx[i] += fx;
          bz[i] += fz;
        }
      }
    }

    // 4. Torque = m x B + mild 6-fold brass-chamber anisotropy
    const aniso = this.hexWallAnisotropy;
    for (let i = 0; i < n; i++) {
      const th = angles[i];
      const sinTh = -mz[i]; // since mz = -sin(theta)
      const cosTh = mx[i];
      // tau = -Bx * sin(theta) - Bz * cos(theta)
      const magTorque = (-bx[i] * sinTh - bz[i] * cosTh) * mScale[i];
      // Subtle 6-fold hex-chamber crystal field favoring alignment toward cell corners or flats
      const wallTorque = -aniso * Math.sin(6 * th);
      this.torques[i] = magTorque + wallTorque;
    }
  }

  /**
   * Advances the coupled 91-dipole system by `dtSec` using fixed 240 Hz symplectic substeps
   * with Coulomb static-friction locking and 3D jewel-pivot nutation wobble.
   */
  step(dtSec: number): void {
    const clampedDt = Math.min(0.05, Math.max(0.0, dtSec));
    if (clampedDt <= 0) return;

    const substeps = Math.max(1, Math.min(12, Math.ceil(clampedDt / (1 / 240))));
    const h = clampedDt / substeps;
    const n = this.count;
    const baseDamp = this.baseDamping;
    const tauStatic = this.staticFrictionTorque;
    const wEps = this.staticSpeedThreshold;

    for (let s = 0; s < substeps; s++) {
      this.simTime += h;
      if (this.pulseActive) {
        this.pulseAge += h;
        if (this.pulseAge > 1.35) {
          this.pulseActive = false;
        }
      }

      this.computeFieldsAndTorques();

      for (let i = 0; i < n; i++) {
        const tau = this.torques[i];
        let w = this.omegas[i];

        // Coulomb jewel-bearing static friction threshold with exact zero-crossing capture
        if (Math.abs(tau) < tauStatic && Math.abs(w) < wEps) {
          this.omegas[i] = 0;
          this.locked[i] = 1;
        } else {
          this.locked[i] = 0;
          const kineticSign = w > 1e-4 ? 1 : w < -1e-4 ? -1 : 0;
          const netTau = tau - kineticSign * (tauStatic * 0.55) - baseDamp * this.dampingScale[i] * w;
          let wNext = w + netTau * h;
          // Prevent kinetic friction from overshooting zero and chattering across sign flip
          if (kineticSign !== 0 && wNext * w <= 0 && Math.abs(tau) < tauStatic) {
            wNext = 0;
            this.locked[i] = 1;
          }
          if (wNext > 42.0) wNext = 42.0;
          else if (wNext < -42.0) wNext = -42.0;
          this.omegas[i] = wNext;
          this.angles[i] = wrapAngle(this.angles[i] + wNext * h);
        }

        // Coupled 3D jewel-pivot micro-nutation wobble (spring-damped harmonic tilt driven by torque)
        const drivePitch = Math.max(-0.18, Math.min(0.18, tau * 0.0022));
        const driveRoll = Math.max(-0.18, Math.min(0.18, w * 0.0045));
        const wpAcc = 120.0 * (drivePitch - this.wobblePitch[i]) - 11.5 * this.wobblePitchVel[i];
        const wrAcc = 120.0 * (driveRoll - this.wobbleRoll[i]) - 11.5 * this.wobbleRollVel[i];
        this.wobblePitchVel[i] += wpAcc * h;
        this.wobbleRollVel[i] += wrAcc * h;
        this.wobblePitch[i] = Math.max(-0.11, Math.min(0.11, this.wobblePitch[i] + this.wobblePitchVel[i] * h));
        this.wobbleRoll[i] = Math.max(-0.11, Math.min(0.11, this.wobbleRoll[i] + this.wobbleRollVel[i] * h));
      }
    }

    // Refresh instantaneous alignment metric for telemetry/shading
    this.updateLocalAlignment();
  }

  private updateLocalAlignment(): void {
    const n = this.count;
    for (let i = 0; i < n; i++) {
      const th = this.angles[i];
      const c = Math.cos(th);
      const s = -Math.sin(th);
      this.mx[i] = c;
      this.mz[i] = s;
    }

    for (let i = 0; i < n; i++) {
      const nb = this.layout.cells[i].neighborByDir;
      let sum = 0;
      let cnt = 0;
      const mix = this.mx[i];
      const miz = this.mz[i];
      for (let d = 0; d < 6; d++) {
        const j = nb[d];
        if (j >= 0) {
          sum += mix * this.mx[j] + miz * this.mz[j];
          cnt++;
        }
      }
      this.localAlignment[i] = cnt > 0 ? (sum / cnt) * 0.5 + 0.5 : 0.5;
    }
  }

  /**
   * Computes exact topological winding counts, global order parameter, dipole energy,
   * and settling telemetry.
   */
  getTelemetry(): LatticeTelemetry {
    const n = this.count;
    let sumMx = 0;
    let sumMz = 0;
    let sumSpeed = 0;
    let peakSpeed = 0;
    let settledCount = 0;
    let cohSum = 0;

    for (let i = 0; i < n; i++) {
      const th = this.angles[i];
      this.mx[i] = Math.cos(th);
      this.mz[i] = -Math.sin(th);
      sumMx += this.mx[i];
      sumMz += this.mz[i];
      const sp = Math.abs(this.omegas[i]);
      sumSpeed += sp;
      if (sp > peakSpeed) peakSpeed = sp;
      if (this.locked[i] === 1) settledCount++;
      cohSum += this.localAlignment[i] * 2 - 1;
    }

    // Exact dipole-dipole interaction energy: U = -0.5 * sum_{i != j} m_i . B_{j->i}
    let energy = 0;
    const pCount = this.pairCount;
    for (let p = 0; p < pCount; p++) {
      const i = this.pairI[p];
      const j = this.pairJ[p];
      const mix = this.mx[i] * this.momentScale[i];
      const miz = this.mz[i] * this.momentScale[i];
      const mjx = this.mx[j] * this.momentScale[j];
      const mjz = this.mz[j] * this.momentScale[j];
      const bix = this.tensorXX[p] * mjx + this.tensorXZ[p] * mjz;
      const biz = this.tensorXZ[p] * mjx + this.tensorZZ[p] * mjz;
      energy -= (mix * bix + miz * biz);
    }

    // Count discrete topological vortices (|winding number| == 1) on interior 6-cell hex rings
    let vortexCount = 0;
    const rings = this.plaquetteRings;
    for (let r = 0; r < this.plaquetteCount; r++) {
      const base = r * 6;
      let dThetaSum = 0;
      for (let k = 0; k < 6; k++) {
        const a = this.angles[rings[base + k]];
        const b = this.angles[rings[base + ((k + 1) % 6)]];
        dThetaSum += wrapAngle(b - a);
      }
      const winding = Math.round(dThetaSum / (Math.PI * 2));
      if (winding !== 0) vortexCount++;
    }

    return {
      cellCount: n,
      globalOrder: Number((Math.hypot(sumMx, sumMz) / n).toFixed(4)),
      localCoherence: Number((cohSum / n).toFixed(4)),
      meanSpeed: Number((sumSpeed / n).toFixed(4)),
      peakSpeed: Number(peakSpeed.toFixed(4)),
      dipoleEnergy: Number(energy.toFixed(3)),
      vortexCount,
      settledCount,
      stylusPolarity: this.stylus.polarity,
    };
  }
}
