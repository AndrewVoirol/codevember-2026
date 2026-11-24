import { MINERAL_PALETTE } from '../../core/palette';

/**
 * Day 23 - Magma: Pure Geomechanics, Hydraulic Fracture & Viscoplastic Thermal Solver
 *
 * Models a pressurized magma chamber sealed beneath a stratified basalt & ochre-tuff
 * rock core clamped inside a triaxial rock-mechanics steel press:
 *  - Andersonian stress-regime coupling:
 *      Extensional (tectonicCompression < 0) -> vertical magma dikes wedge rock columns apart
 *      and erupt viscous pahoehoe magma lobes across the upper crest.
 *      Compressional (tectonicCompression > 0) -> rising magma deflects laterally into glowing
 *      horizontal sills along sedimentary bedding planes, physically heaving and doming
 *      the upper rock slabs into an anticlinal laccolith bulge.
 *  - Physical blackbody cooling & viscoplastic re-cracking:
 *      1180°C pale-gold core -> realgar amber -> cinnabar vermilion -> dark blistering crust
 *      -> glossy black volcanic obsidian (roughness ~ 0.14) that fractures back open with
 *      glowing cinnabar seams when fresh chamber pressure surges.
 */

// ---------------------------------------------------------------------------
// Shared Geometric Dimensions (meters / stage units)
// ---------------------------------------------------------------------------

export const CORE_WIDTH = 1.60;
export const CORE_HALF_W = CORE_WIDTH * 0.5; // 0.80
export const CORE_HEIGHT = 1.30;
export const CORE_HALF_H = CORE_HEIGHT * 0.5; // 0.65
export const CORE_DEPTH = 1.10;
export const CORE_HALF_D = CORE_DEPTH * 0.5; // 0.55

/** Magma chamber spans y in [-0.65, -0.22]; stratified overburden spans [-0.22, +0.65]. */
export const CHAMBER_BOTTOM_Y = -CORE_HALF_H; // -0.65
export const CHAMBER_ROOF_Y = -0.22;
export const CHAMBER_HEIGHT = CHAMBER_ROOF_Y - CHAMBER_BOTTOM_Y; // 0.43

export const OVERBURDEN_BOTTOM_Y = CHAMBER_ROOF_Y; // -0.22
export const OVERBURDEN_TOP_Y = CORE_HALF_H; // +0.65
export const OVERBURDEN_HEIGHT = OVERBURDEN_TOP_Y - OVERBURDEN_BOTTOM_Y; // 0.87

export const STRATA_TIERS = 3;
export const SLAB_COLUMNS = 4;
export const SLAB_COUNT = STRATA_TIERS * SLAB_COLUMNS; // 12 interlocking slabs

/** Bedding plane elevations between the 3 strata horizons */
export const TIER_BOUNDARIES_Y: readonly number[] = [
  OVERBURDEN_BOTTOM_Y, // -0.22 (base of Lower Vesicular Basalt)
  0.07, // Lower bedding plane (Basalt -> Banded Ochre Tuff)
  0.36, // Upper bedding plane (Ochre Tuff -> Upper Columnar Basalt Cap)
  OVERBURDEN_TOP_Y, // +0.65 (top crest)
];

/** Rest X boundaries of the 4 interlocking columns (3 vertical dike seams between them) */
export const COLUMN_BOUNDARIES_X: readonly number[] = [-0.80, -0.35, 0.0, 0.35, 0.80];

/** Vertical dike seam rest X positions: center (primary), left flank, right flank */
export const DIKE_SEAM_X: readonly [number, number, number] = [0.0, -0.35, 0.35];

/** Horizontal sill bedding elevations: lower sill (Basalt/Tuff), upper sill (Tuff/Cap) */
export const SILL_BED_Y: readonly [number, number] = [0.07, 0.36];

/** Strict press-frame physical clearance limits */
export const MAX_HEAVE_DISPLACEMENT = 0.24; // Keeps top crest safely below upper tie-beam (y = 0.94)
export const MAX_PLATEN_EXTENSION = 0.125; // Max outward clamp travel at compression = -1 (tightly tracks expanding rock)
export const MAX_PLATEN_COMPRESSION = 0.016; // Max inward elastic clamp preload at compression = +1 (realistic rock strain)
export const MAX_PISTON_TRAVEL = 0.155; // Vertical hydraulic piston stroke into lower chamber (keeps upper half blazing with melt)
export const ACME_THREAD_PITCH = 0.048; // Exact helical lead pitch (4.8 cm axial travel per revolution)
export const MAX_LOBES = 24;

export type StressRegime = 'extensional' | 'transitional' | 'compressional';

export interface DikeState {
  index: number;
  baseX: number;
  /** Intrusion ascent height above CHAMBER_ROOF_Y in meters [0, OVERBURDEN_HEIGHT + 0.06] */
  height: number;
  /** Physical lateral opening aperture in meters [0, 0.105] */
  aperture: number;
  /** Normalized surface thermal state [0 = obsidian (<380C), 1 = 1180C incandescent] */
  temperature: number;
  /** Normalized molten core temperature [0, 1] */
  coreTemp: number;
  /** Amount of solid quenched obsidian remaining in the fissure [0, 1] */
  quenchedObsidian: number;
  /** Bright re-cracking seam pulse when fresh pressure breaks cooled crust [0, 1] */
  reCrackPulse: number;
  /** Whether the dike has breached the upper rock crest */
  breachedCrest: boolean;
}

export interface SillState {
  index: number;
  baseY: number;
  /** Lateral half-width of the glowing sill lens from center [0, CORE_HALF_W - 0.01] */
  halfWidth: number;
  /** Vertical opening thickness of the sill lens in meters [0, 0.125] */
  thickness: number;
  /** Surface temperature [0, 1] */
  temperature: number;
  /** Core melt temperature [0, 1] */
  coreTemp: number;
  /** Cooled obsidian persistence [0, 1] */
  quenchedObsidian: number;
  /** Re-cracking pulse [0, 1] */
  reCrackPulse: number;
}

export interface RockSlabState {
  index: number;
  tier: number; // 0 = lower basalt, 1 = ochre tuff, 2 = upper basalt cap
  col: number; // 0..3 left-to-right
  restCenterX: number;
  restCenterY: number;
  dx: number;
  dy: number;
  dz: number;
  velX: number;
  velY: number;
  /** Roll rotation around Z (radians) from dome arch curvature / dike wedging */
  rotZ: number;
  velRotZ: number;
  /** Subtle pitch/yaw micro-jostle during hydraulic fracturing */
  rotX: number;
  /** Thermal glow absorbed along fractured walls [0, 1] */
  seamHeat: number;
}

export interface MagmaLobeState {
  index: number;
  active: boolean;
  ventType: 'crest-dike' | 'flank-sill';
  ventIndex: number;
  /** Rest anchor relative to its parent vent before slab heave/offset */
  localOffsetX: number;
  localOffsetY: number;
  localOffsetZ: number;
  /** World-space position synced to moving rock fissure mouth */
  posX: number;
  posY: number;
  posZ: number;
  /** Sluggish gravitational ooze travel vector */
  flowX: number;
  flowY: number;
  flowZ: number;
  /** Extruded growth envelope [0, 1] */
  growth: number;
  /** Viscous pillow dimensions (meters) */
  radiusX: number;
  radiusY: number;
  radiusZ: number;
  /** Yaw orientation along flow */
  yaw: number;
  /** Pitch droop along crest/flank */
  pitch: number;
  /** Internal molten core temperature [0, 1] */
  coreTemp: number;
  /** Outer skin temperature [0, 1] (cools faster -> glossy black obsidian) */
  crustTemp: number;
  /** Re-cracking seam intensity [0, 1] */
  crackIntensity: number;
  /** Accumulated ropy pahoehoe wrinkle factor [0, 1] */
  wrinkle: number;
  /** Age in simulation seconds */
  ageSec: number;
  /** Deterministic per-lobe shader seed */
  seed: number;
}

export interface Day23Telemetry {
  chamberPressure: number;
  tectonicCompression: number;
  activeDikeHeight: number;
  sillFraction: number;
  heaveDisplacement: number;
  extrudedLobes: number;
  stressRegime: StressRegime;
  pistonTravel: number;
  platenOffset: number;
  maxSillThickness: number;
  avgLobeCrustTemp: number;
  quenchedObsidianFraction: number;
  simTimeSec: number;
}

// ---------------------------------------------------------------------------
// Physical Blackbody Thermal Scale (Anchored in Mineral Palette, Zero Neons)
// ---------------------------------------------------------------------------

const OBSIDIAN_RGB: readonly [number, number, number] = [0.011, 0.012, 0.015];
const HEMATITE_RGB = MINERAL_PALETTE.hematiteRust.rInfLinear;
const CINNABAR_RGB = MINERAL_PALETTE.cinnabarVermilion.rInfLinear;
const REALGAR_RGB = MINERAL_PALETTE.realgarAmber.rInfLinear;
const OCHRE_RGB = MINERAL_PALETTE.yellowOchre.rInfLinear;
const CALCITE_RGB = MINERAL_PALETTE.boneCalcite.rInfLinear;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / Math.max(1e-6, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Evaluates physically motivated magma blackbody cooling color + emissive radiance
 * and obsidian glass roughness for normalized thermal state `temp01` in [0, 1]:
 *   0.00 -> <380°C quenched glossy volcanic obsidian (roughness ~ 0.14, emission = 0)
 *   0.20 -> ~580°C blistering dark hematite-rust crust (roughness ~ 0.50)
 *   0.45 -> ~780°C glowing cinnabar vermilion melt
 *   0.72 -> ~980°C bright realgar amber melt
 *   1.00 -> ~1180°C incandescent pale-gold core
 */
export function evalMagmaBlackbody(temp01: number): {
  albedoLinear: [number, number, number];
  emissiveLinear: [number, number, number];
  roughness: number;
  luminance: number;
  temperatureC: number;
} {
  const t = clamp(temp01, 0, 1);
  const temperatureC = 320 + t * 860; // 320°C (quenched obsidian) .. 1180°C (basaltic melt)

  // Piecewise smooth mineral blackbody ramp (matching TSL shader contract)
  let r: number;
  let g: number;
  let b: number;

  if (t < 0.22) {
    const u = smoothstep(0.0, 0.22, t);
    r = lerp(OBSIDIAN_RGB[0], HEMATITE_RGB[0] * 0.55, u);
    g = lerp(OBSIDIAN_RGB[1], HEMATITE_RGB[1] * 0.28, u);
    b = lerp(OBSIDIAN_RGB[2], HEMATITE_RGB[2] * 0.22, u);
  } else if (t < 0.48) {
    const u = smoothstep(0.22, 0.48, t);
    r = lerp(HEMATITE_RGB[0] * 0.55, CINNABAR_RGB[0] * 1.12, u);
    g = lerp(HEMATITE_RGB[1] * 0.28, CINNABAR_RGB[1] * 0.72, u);
    b = lerp(HEMATITE_RGB[2] * 0.22, CINNABAR_RGB[2] * 0.25, u);
  } else if (t < 0.76) {
    const u = smoothstep(0.48, 0.76, t);
    r = lerp(CINNABAR_RGB[0] * 1.12, REALGAR_RGB[0] * 1.28, u);
    g = lerp(CINNABAR_RGB[1] * 0.72, REALGAR_RGB[1] * 1.08, u);
    b = lerp(CINNABAR_RGB[2] * 0.25, REALGAR_RGB[2] * 0.40, u);
  } else {
    const u = smoothstep(0.76, 1.0, t);
    const paleGoldR = lerp(OCHRE_RGB[0], CALCITE_RGB[0], 0.52);
    const paleGoldG = lerp(OCHRE_RGB[1], CALCITE_RGB[1], 0.52);
    const paleGoldB = lerp(OCHRE_RGB[2], CALCITE_RGB[2], 0.46);
    r = lerp(REALGAR_RGB[0] * 1.28, paleGoldR * 1.32, u);
    g = lerp(REALGAR_RGB[1] * 1.08, paleGoldG * 1.24, u);
    b = lerp(REALGAR_RGB[2] * 0.40, paleGoldB * 1.02, u);
  }

  // Stefan-Boltzmann-inspired thermal emission envelope: strictly zero on cooled obsidian,
  // smoothly monotonic up to warm HDR core radiance (no digital clip)
  const emissiveGain = Math.pow(smoothstep(0.06, 1.0, t), 2.35) * 2.35;
  const er = r * emissiveGain;
  const eg = g * emissiveGain;
  const eb = b * emissiveGain;

  // Glossy black volcanic obsidian glass when cooled (roughness ~ 0.14), rough blistering
  // clinker transition around t ~ 0.25 (roughness ~ 0.52), smooth viscous melt at high t (~ 0.24)
  const blisterBump = Math.exp(-Math.pow((t - 0.26) / 0.14, 2)) * 0.38;
  const meltLift = smoothstep(0.45, 0.95, t) * 0.10;
  const roughness = clamp(0.14 + blisterBump + meltLift, 0.13, 0.58);

  const totalR = r + er;
  const totalG = g + eg;
  const totalB = b + eb;
  const luminance = 0.2126 * totalR + 0.7152 * totalG + 0.0722 * totalB;

  return {
    albedoLinear: [r, g, b],
    emissiveLinear: [er, eg, eb],
    roughness,
    luminance,
    temperatureC,
  };
}

/**
 * Computes the Andersonian stress-regime sill dominance fraction in [0.04, 0.96]
 * strictly from tectonicCompression in [-1, +1].
 */
export function computeSillFraction(tectonicCompression: number): number {
  const c = clamp(tectonicCompression, -1, 1);
  const logistic = 1 / (1 + Math.exp(-4.8 * (c - 0.04)));
  return clamp(logistic, 0.04, 0.96);
}

/**
 * Deterministic fracture boundary profile shared by geometry builder & solver
 * so neighboring rock slabs interlock cleanly along matching jagged joints.
 */
export function evalColumnSeamX(seamIdx: number, y: number, z: number): number {
  if (seamIdx <= 0) return COLUMN_BOUNDARIES_X[0];
  if (seamIdx >= SLAB_COLUMNS) return COLUMN_BOUNDARIES_X[SLAB_COLUMNS];
  const base = COLUMN_BOUNDARIES_X[seamIdx];
  const phase = seamIdx * 2.17;
  // Natural stepped rock-cleavage zig-zag along height and depth (bounded +/-0.032m)
  const yNorm = (y - OVERBURDEN_BOTTOM_Y) / OVERBURDEN_HEIGHT;
  const jagY =
    Math.sin(yNorm * Math.PI * 3.0 + phase) * 0.022 +
    Math.sin(yNorm * Math.PI * 7.0 + phase * 1.9) * 0.008;
  const jagZ = Math.sin(z * 4.3 + phase * 0.7) * 0.012;
  return base + jagY + jagZ;
}

/**
 * Deterministic undulating sedimentary bedding surface shared between stacked tiers.
 */
export function evalBeddingPlaneY(tierBoundaryIdx: number, x: number, z: number): number {
  const base = TIER_BOUNDARIES_Y[tierBoundaryIdx];
  if (tierBoundaryIdx === 0) return base; // Flat seat onto steel chamber roof collar
  const amp = tierBoundaryIdx === STRATA_TIERS ? 0.016 : 0.012;
  const phase = tierBoundaryIdx * 1.63;
  return (
    base +
    Math.sin(x * 3.1 + phase) * amp * 0.65 +
    Math.cos(z * 3.7 - phase * 0.8) * amp * 0.45
  );
}

// ---------------------------------------------------------------------------
// Deterministic Seeded PRNG (Zero Math.random jitter across tests)
// ---------------------------------------------------------------------------

function hashUnit(seed: number): number {
  const s = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

// ---------------------------------------------------------------------------
// Main Tectonic Magma-Intrusion Solver
// ---------------------------------------------------------------------------

export interface TectonicSolverOptions {
  initialPressure?: number;
  initialCompression?: number;
  preWarmSec?: number;
}

export class TectonicMagmaSolver {
  chamberPressure: number;
  targetPressure: number;
  tectonicCompression: number;
  targetCompression: number;

  /** Computed Andersonian sill-vs-dike ratio in [0.04, 0.96] */
  sillFraction: number;
  /** Peak vertical dike ascent height above chamber roof (meters) */
  activeDikeHeight = 0;
  /** Peak vertical overburden dome heave (meters) */
  heaveDisplacement = 0;
  /** Hydraulic piston vertical travel into chamber (meters) */
  pistonTravel = 0;
  /** Signed outward offset of lateral clamp platens (positive = open tension, negative = squeezed) */
  platenOffset = 0;
  /** Mechanical screw rotation angle (radians) coupled strictly to platenOffset */
  clampScrewAngle = 0;
  /** Instantaneous pressure rate dP/dt used to trigger live obsidian re-cracking */
  pressureSurge = 0;
  /** Elapsed simulation time (seconds) */
  simTimeSec = 0;

  readonly dikes: DikeState[];
  readonly sills: SillState[];
  readonly slabs: RockSlabState[];
  readonly lobes: MagmaLobeState[];

  private prevPressure: number;
  private lobeSpawnCursor = 0;
  private lobeSpawnCooldown = 0;
  private isPumpingHold = false;
  private holdPumpRate = 0;

  constructor(options: TectonicSolverOptions = {}) {
    this.chamberPressure = clamp(options.initialPressure ?? 0.56, 0, 1);
    this.targetPressure = this.chamberPressure;
    this.prevPressure = this.chamberPressure;
    this.tectonicCompression = clamp(options.initialCompression ?? -0.32, -1, 1);
    this.targetCompression = this.tectonicCompression;
    this.sillFraction = computeSillFraction(this.tectonicCompression);

    // 3 vertical dikes: central primary (0), left flank (1), right flank (2)
    this.dikes = DIKE_SEAM_X.map((baseX, index) => ({
      index,
      baseX,
      height: 0,
      aperture: 0,
      temperature: 0,
      coreTemp: 0,
      quenchedObsidian: 0,
      reCrackPulse: 0,
      breachedCrest: false,
    }));

    // 2 horizontal sills: lower bedding plane (0) and upper bedding plane (1)
    this.sills = SILL_BED_Y.map((baseY, index) => ({
      index,
      baseY,
      halfWidth: 0,
      thickness: 0,
      temperature: 0,
      coreTemp: 0,
      quenchedObsidian: 0,
      reCrackPulse: 0,
    }));

    // 12 interlocking stratified rock slabs (3 tiers x 4 columns)
    this.slabs = [];
    for (let tier = 0; tier < STRATA_TIERS; tier++) {
      const yMid = 0.5 * (TIER_BOUNDARIES_Y[tier] + TIER_BOUNDARIES_Y[tier + 1]);
      for (let col = 0; col < SLAB_COLUMNS; col++) {
        const xMid = 0.5 * (COLUMN_BOUNDARIES_X[col] + COLUMN_BOUNDARIES_X[col + 1]);
        this.slabs.push({
          index: tier * SLAB_COLUMNS + col,
          tier,
          col,
          restCenterX: xMid,
          restCenterY: yMid,
          dx: 0,
          dy: 0,
          dz: 0,
          velX: 0,
          velY: 0,
          rotZ: 0,
          velRotZ: 0,
          rotX: 0,
          seamHeat: 0,
        });
      }
    }

    // Pre-allocated pool of MAX_LOBES viscoplastic pahoehoe lobes
    this.lobes = [];
    for (let i = 0; i < MAX_LOBES; i++) {
      this.lobes.push({
        index: i,
        active: false,
        ventType: 'crest-dike',
        ventIndex: 0,
        localOffsetX: 0,
        localOffsetY: 0,
        localOffsetZ: 0,
        posX: 0,
        posY: OVERBURDEN_TOP_Y,
        posZ: 0,
        flowX: 0,
        flowY: 0,
        flowZ: 0,
        growth: 0,
        radiusX: 0.06,
        radiusY: 0.035,
        radiusZ: 0.06,
        yaw: 0,
        pitch: 0,
        coreTemp: 0,
        crustTemp: 0,
        crackIntensity: 0,
        wrinkle: 0,
        ageSec: 0,
        seed: (i + 1) * 1.371,
      });
    }

    // Optional deterministic pre-warm so Frame 0 opens as an active, sculpted specimen
    const preWarm = options.preWarmSec ?? 2.4;
    if (preWarm > 0) {
      const steps = Math.ceil(preWarm * 60);
      for (let s = 0; s < steps; s++) {
        this.step(1 / 60);
      }
    }
  }

  /** Directly commands target chamber pressure in [0, 1]. */
  setTargetPressure(pressure01: number): void {
    this.targetPressure = clamp(pressure01, 0, 1);
  }

  /** Adds a relative delta to target chamber pressure (used by vertical piston drag). */
  nudgePressure(delta: number): void {
    this.targetPressure = clamp(this.targetPressure + delta, 0, 1);
  }

  /** Enables or disables continuous hold-click hydraulic chamber pumping. */
  setHoldPump(active: boolean, ratePerSec = 0.42): void {
    this.isPumpingHold = active;
    this.holdPumpRate = ratePerSec;
  }

  /** Directly commands horizontal tectonic stress in [-1 (tension), +1 (compression)]. */
  setTargetCompression(compression: number): void {
    this.targetCompression = clamp(compression, -1, 1);
  }

  /** Adds a relative delta to horizontal tectonic stress (used by lateral clamp drag). */
  nudgeCompression(delta: number): void {
    this.targetCompression = clamp(this.targetCompression + delta, -1, 1);
  }

  /**
   * Releases chamber pressure, rapidly quenches active dikes/sills/lobes toward
   * glossy black obsidian veins, and relaxes overburden plates cleanly for a new regime.
   */
  quenchAndReset(clearLobes = false): void {
    this.isPumpingHold = false;
    this.targetPressure = 0;
    this.chamberPressure *= 0.25;
    this.pressureSurge = 0;

    for (let i = 0; i < this.dikes.length; i++) {
      const d = this.dikes[i];
      if (d.height > 0.08) {
        d.quenchedObsidian = Math.max(d.quenchedObsidian, 0.85);
      }
      d.temperature *= 0.18;
      d.coreTemp *= 0.25;
      d.reCrackPulse = 0;
      d.breachedCrest = false;
    }

    for (let i = 0; i < this.sills.length; i++) {
      const s = this.sills[i];
      if (s.halfWidth > 0.08) {
        s.quenchedObsidian = Math.max(s.quenchedObsidian, 0.85);
      }
      s.temperature *= 0.18;
      s.coreTemp *= 0.25;
      s.reCrackPulse = 0;
    }

    for (let i = 0; i < this.lobes.length; i++) {
      const l = this.lobes[i];
      if (clearLobes) {
        l.active = false;
        l.growth = 0;
      } else if (l.active) {
        // Quench extruded pahoehoe tongues straight toward glossy obsidian crust
        l.crustTemp = Math.min(l.crustTemp, 0.08);
        l.coreTemp = Math.min(l.coreTemp, 0.24);
        l.crackIntensity = 0;
        l.wrinkle = Math.min(1, l.wrinkle + 0.35);
      }
    }
  }

  /** Advances the coupled geomechanics, fracture, slab heave & thermal cooling model. */
  step(dtRaw: number): void {
    const dt = clamp(dtRaw, 0.0005, 0.05);
    this.simTimeSec += dt;

    if (this.isPumpingHold) {
      this.targetPressure = clamp(this.targetPressure + this.holdPumpRate * dt, 0, 1);
    }

    // 1. Smooth hydraulic pressure & mechanical screw-clamp response
    const pAlpha = 1 - Math.exp(-dt * 7.5);
    this.prevPressure = this.chamberPressure;
    this.chamberPressure = clamp(
      this.chamberPressure + (this.targetPressure - this.chamberPressure) * pAlpha,
      0,
      1
    );
    const dPdt = (this.chamberPressure - this.prevPressure) / dt;
    if (dPdt > 0.05) {
      this.pressureSurge = Math.min(1, this.pressureSurge + dPdt * dt * 2.8);
    } else {
      this.pressureSurge *= Math.exp(-dt * 2.4);
    }

    const cAlpha = 1 - Math.exp(-dt * 8.5);
    this.tectonicCompression = clamp(
      this.tectonicCompression + (this.targetCompression - this.tectonicCompression) * cAlpha,
      -1,
      1
    );

    // 2. Mechanical press kinematics (piston stroke + lateral screw-clamp travel & thread pitch)
    this.pistonTravel = this.chamberPressure * MAX_PISTON_TRAVEL;
    if (this.tectonicCompression <= 0) {
      this.platenOffset = -this.tectonicCompression * MAX_PLATEN_EXTENSION;
    } else {
      this.platenOffset = -this.tectonicCompression * MAX_PLATEN_COMPRESSION;
    }
    // Exact ACME thread lead coupling (pitch = 0.048 m/rev -> 1 full turn per 4.8 cm travel)
    this.clampScrewAngle = (-this.platenOffset / ACME_THREAD_PITCH) * Math.PI * 2;

    // 3. Andersonian stress regime partition (extensional dikes vs. compressional sills)
    this.sillFraction = computeSillFraction(this.tectonicCompression);
    const dikeFraction = 1 - this.sillFraction;
    const P = this.chamberPressure;
    const c = this.tectonicCompression;

    // 4. Vertical Dike Propagation, Aperture & Thermal Cooling
    let maxDikeH = 0;
    for (let i = 0; i < this.dikes.length; i++) {
      const d = this.dikes[i];
      const isCentral = i === 0;
      const flankPenalty = isCentral ? 0.0 : 0.10;
      const pCrit = 0.12 + flankPenalty + Math.max(0, c + 0.25) * 0.26;
      const overP = Math.max(0, P - pCrit) / Math.max(0.18, 1 - pCrit);

      // Under strong compression (high sillFraction), vertical dikes arrest near lower/mid bedding
      // and feed sideways into horizontal sills; under tension they shoot through the crest!
      const ceilingFrac = isCentral
        ? clamp(0.32 + dikeFraction * 0.74, 0.30, 1.06)
        : clamp(0.18 + dikeFraction * 0.84, 0.16, 1.02);
      const targetHeight = overP > 0 ? OVERBURDEN_HEIGHT * ceilingFrac * Math.pow(overP, 0.72) : 0;

      // Dikes climb rapidly when pressurized, stall/freeze in place as quenched obsidian veins
      // when pressure falls (gradually retracting only partially unless reset)
      if (targetHeight > d.height) {
        const climbRate = (1.05 + 1.45 * dikeFraction) * dt;
        d.height = Math.min(targetHeight, d.height + (targetHeight - d.height) * Math.min(1, climbRate * 3.2) + 0.08 * dt * overP);
      } else if (P < 0.08) {
        // Complete depressurization contracts molten column slowly, leaving chilled vein
        d.height *= Math.exp(-dt * 0.55);
        if (d.height < 0.005) d.height = 0;
      } else {
        d.height += (targetHeight - d.height) * (1 - Math.exp(-dt * 1.4));
      }

      // Aperture widens sharply in extensional tension and gets squeezed tight under compression
      const lateralAllowance = clamp(0.35 + (1 - c) * 0.52, 0.10, 1.15);
      const targetAperture =
        (isCentral ? 0.086 : 0.056) *
        Math.pow(overP, 0.85) *
        (0.20 + 0.80 * dikeFraction) *
        lateralAllowance;
      const apertureRate = targetAperture > d.aperture ? 4.8 : 2.6;
      d.aperture += (targetAperture - d.aperture) * (1 - Math.exp(-dt * apertureRate));
      d.aperture = clamp(d.aperture, 0, 0.105);

      // Thermal state & live obsidian re-cracking
      const activeFlowHeat = clamp(overP * (0.55 + 0.55 * dikeFraction), 0, 1);
      if (activeFlowHeat > d.coreTemp) {
        // Fresh hot magma surging into a cooled dike cracks its obsidian skin open!
        const reCrackBoost = d.quenchedObsidian * clamp((activeFlowHeat - d.temperature) * 2.5, 0, 1);
        d.reCrackPulse = Math.min(1, d.reCrackPulse + reCrackBoost * dt * 7.5 + this.pressureSurge * 0.85 * dt);
        d.coreTemp += (activeFlowHeat - d.coreTemp) * (1 - Math.exp(-dt * 4.5));
      } else {
        // Radiative cooling toward dark obsidian
        const coolK = 0.24 + 0.42 * (1 - P);
        d.coreTemp = Math.max(0, d.coreTemp - (0.09 + 0.32 * d.coreTemp * d.coreTemp) * coolK * dt);
      }

      const targetSkin = clamp(d.coreTemp * (0.76 + 0.24 * overP) + d.reCrackPulse * 0.35, 0, 1);
      if (targetSkin > d.temperature) {
        d.temperature += (targetSkin - d.temperature) * (1 - Math.exp(-dt * 5.0));
      } else {
        d.temperature = Math.max(0, d.temperature - (0.14 + 0.50 * d.temperature * d.temperature) * dt);
      }
      d.reCrackPulse *= Math.exp(-dt * 1.45);

      if (d.height > 0.10 && d.temperature < 0.35) {
        d.quenchedObsidian = Math.min(1, d.quenchedObsidian + dt * 0.35);
      } else if (d.temperature > 0.72) {
        d.quenchedObsidian = Math.max(0, d.quenchedObsidian - dt * 0.45);
      }

      d.breachedCrest = d.height >= OVERBURDEN_HEIGHT * 0.86 && d.aperture > 0.016 && P > 0.32;
      if (d.height > maxDikeH) maxDikeH = d.height;
    }
    this.activeDikeHeight = clamp(maxDikeH, 0, OVERBURDEN_HEIGHT + 0.06);

    // 5. Horizontal Sill Intrusion along Sedimentary Bedding Planes
    for (let i = 0; i < this.sills.length; i++) {
      const s = this.sills[i];
      const isLower = i === 0;
      const pCrit = (isLower ? 0.14 : 0.24) + Math.max(0, 0.25 - c) * 0.24;
      const overP = Math.max(0, P - pCrit) / Math.max(0.18, 1 - pCrit);

      // Horizontal sills grow strongly under compressional clamping (high sillFraction)
      const sillAuthority = isLower
        ? clamp(0.16 + this.sillFraction * 0.94, 0.16, 1.0)
        : clamp(0.06 + this.sillFraction * 0.96, 0.06, 1.0);

      const targetHalfW =
        overP > 0 ? (CORE_HALF_W - 0.01) * sillAuthority * Math.pow(overP, 0.68) : 0;
      const targetThick =
        (isLower ? 0.118 : 0.086) * Math.pow(overP, 0.88) * Math.pow(this.sillFraction, 0.95);

      if (targetHalfW > s.halfWidth) {
        s.halfWidth += (targetHalfW - s.halfWidth) * (1 - Math.exp(-dt * 2.9));
      } else if (P < 0.08) {
        s.halfWidth *= Math.exp(-dt * 0.55);
        if (s.halfWidth < 0.005) s.halfWidth = 0;
      } else {
        s.halfWidth += (targetHalfW - s.halfWidth) * (1 - Math.exp(-dt * 1.3));
      }

      const thickRate = targetThick > s.thickness ? 4.2 : 2.6;
      s.thickness += (targetThick - s.thickness) * (1 - Math.exp(-dt * thickRate));
      s.thickness = clamp(s.thickness, 0, isLower ? 0.125 : 0.095);

      const activeHeat = clamp(overP * (0.40 + 0.65 * this.sillFraction), 0, 1);
      if (activeHeat > s.coreTemp) {
        const reCrackBoost = s.quenchedObsidian * clamp((activeHeat - s.temperature) * 2.6, 0, 1);
        s.reCrackPulse = Math.min(1, s.reCrackPulse + reCrackBoost * dt * 7.0 + this.pressureSurge * 0.8 * dt);
        s.coreTemp += (activeHeat - s.coreTemp) * (1 - Math.exp(-dt * 4.4));
      } else {
        const coolK = 0.22 + 0.42 * (1 - P);
        s.coreTemp = Math.max(0, s.coreTemp - (0.09 + 0.30 * s.coreTemp * s.coreTemp) * coolK * dt);
      }

      const targetSkin = clamp(s.coreTemp * (0.78 + 0.22 * overP) + s.reCrackPulse * 0.35, 0, 1);
      if (targetSkin > s.temperature) {
        s.temperature += (targetSkin - s.temperature) * (1 - Math.exp(-dt * 4.8));
      } else {
        s.temperature = Math.max(0, s.temperature - (0.14 + 0.48 * s.temperature * s.temperature) * dt);
      }
      s.reCrackPulse *= Math.exp(-dt * 1.4);

      if (s.halfWidth > 0.20 && s.thickness > 0.018 && s.temperature < 0.35) {
        s.quenchedObsidian = Math.min(1, s.quenchedObsidian + dt * 0.35);
      } else if (s.temperature > 0.72 || s.thickness < 0.008) {
        s.quenchedObsidian = Math.max(0, s.quenchedObsidian - dt * 0.55);
      }
    }

    // 6. Interlocking 12-Slab Overburden Kinematics (Wedging, Laccolith Dome Heave & Platen Bounds)
    const centerAp = this.dikes[0].aperture;
    const leftAp = this.dikes[1].aperture;
    const rightAp = this.dikes[2].aperture;
    const lowerSillT = this.sills[0].thickness;
    const upperSillT = this.sills[1].thickness;
    const lowerSillFrac = this.sills[0].halfWidth / CORE_HALF_W;
    const upperSillFrac = this.sills[1].halfWidth / CORE_HALF_W;

    // Basal chamber pressure cushion adds subtle elastic dome lift across Tier 0 under high compression
    const basalLift = P * (0.012 + 0.022 * this.sillFraction);
    // Strict lateral containment inside the left/right forged-iron platens (inner platen face at ±(CORE_HALF_W + platenOffset + 0.004))
    const maxOuterEdgeX = CORE_HALF_W + this.platenOffset;

    let maxHeave = 0;

    for (let i = 0; i < this.slabs.length; i++) {
      const slab = this.slabs[i];
      const tier = slab.tier;
      const col = slab.col;

      const colLeftRest = COLUMN_BOUNDARIES_X[col];
      const colRightRest = COLUMN_BOUNDARIES_X[col + 1];
      const halfW = 0.5 * (colRightRest - colLeftRest);
      const halfH = 0.5 * (TIER_BOUNDARIES_Y[tier + 1] - TIER_BOUNDARIES_Y[tier]);
      const effHalfH = halfH + 0.009; // includes bedding undulation allowance

      // Height-gated dike reach for this tier (so upper slabs wedge only once dikes reach them)
      const tierBaseH = TIER_BOUNDARIES_Y[tier] - OVERBURDEN_BOTTOM_Y;
      const centerReach = smoothstep(tierBaseH - 0.08, tierBaseH + 0.22, this.dikes[0].height);
      const leftReach = smoothstep(tierBaseH - 0.08, tierBaseH + 0.22, this.dikes[1].height);
      const rightReach = smoothstep(tierBaseH - 0.08, tierBaseH + 0.22, this.dikes[2].height);

      // Dome radial profile: center columns (1, 2) dome highest; outer columns (0, 3) shoulder lower
      const radialNorm = Math.abs(slab.restCenterX) / CORE_HALF_W; // ~0.22 inner, ~0.72 outer
      const domeProfile = 1.0 - 0.44 * radialNorm * radialNorm;
      const lowerSillSupport = smoothstep(radialNorm * 0.65, radialNorm * 0.65 + 0.28, lowerSillFrac + 0.15);
      const upperSillSupport = smoothstep(radialNorm * 0.65, radialNorm * 0.65 + 0.28, upperSillFrac + 0.12);

      // Anticlinal dome outward tilt (left slabs tilt CCW +rotZ, right slabs tilt CW -rotZ)
      const sideSign = col < 2 ? 1 : -1;
      const isOuter = col === 0 || col === 3;
      const domeTiltMag =
        (lowerSillT * (tier >= 1 ? 0.92 : 0.18) + upperSillT * (tier >= 2 ? 0.78 : 0.12)) *
        (isOuter ? 0.88 : 0.56);
      const wedgeTiltMag =
        (centerAp * centerReach * (isOuter ? 0.34 : 0.46) +
          (col === 0 ? leftAp * leftReach : col === 3 ? rightAp * rightReach : 0) * 0.52) *
        (0.42 + 0.28 * tier);
      // Outer slabs held by steel platens stay within physically realistic tilt (< 6.5 deg)
      const maxTilt = isOuter ? 0.112 : 0.135;
      const targetRotZ = clamp(sideSign * (domeTiltMag + wedgeTiltMag), -maxTilt, maxTilt);

      // Target lateral displacement from dike wedging + slight inward elastic compression
      const elasticSqueeze = Math.max(0, c) * MAX_PLATEN_COMPRESSION * (slab.restCenterX / CORE_HALF_W);
      let targetDx = -elasticSqueeze;
      if (col === 0) {
        targetDx -= centerAp * 0.50 * centerReach + leftAp * 0.78 * leftReach;
      } else if (col === 1) {
        targetDx -= centerAp * 0.52 * centerReach;
      } else if (col === 2) {
        targetDx += centerAp * 0.52 * centerReach;
      } else {
        targetDx += centerAp * 0.50 * centerReach + rightAp * 0.78 * rightReach;
      }

      // Compute rotated outer corner horizontal reach so tilted slabs NEVER clip into lateral platens
      const targetRotReachX = halfW * Math.cos(targetRotZ) + effHalfH * Math.abs(Math.sin(targetRotZ));
      if (col === 0) {
        const minAllowedDx = -maxOuterEdgeX + targetRotReachX - slab.restCenterX;
        if (targetDx < minAllowedDx) targetDx = minAllowedDx;
      } else if (col === 3) {
        const maxAllowedDx = maxOuterEdgeX - targetRotReachX - slab.restCenterX;
        if (targetDx > maxAllowedDx) targetDx = maxAllowedDx;
      }
      targetDx = clamp(targetDx, -0.14, 0.14);

      let targetDy = basalLift * domeProfile;
      if (tier >= 1) {
        targetDy += lowerSillT * domeProfile * (0.45 + 0.55 * lowerSillSupport);
      }
      if (tier >= 2) {
        targetDy += upperSillT * domeProfile * (0.45 + 0.55 * upperSillSupport);
      }
      // Mild upward shoulder lift during vigorous extensional dike wedging
      targetDy += centerAp * 0.22 * centerReach * (tier === 2 ? 1.0 : tier === 1 ? 0.55 : 0.2);
      // Ensure Tier 0 lowest rotated corner rests cleanly atop the steel chamber roof (never dips below -0.22)
      if (tier === 0) {
        const cornerLift = halfW * Math.abs(Math.sin(targetRotZ)) + halfH * (Math.cos(targetRotZ) - 1);
        targetDy = Math.max(targetDy, Math.max(0, cornerLift));
      }
      targetDy = clamp(targetDy, 0, MAX_HEAVE_DISPLACEMENT);

      // Overdamped spring-damper integration (rock mass inertia, zero oscillation jitter)
      const stiffness = 34.0;
      const damping = 11.6;
      const accX = (targetDx - slab.dx) * stiffness - slab.velX * damping;
      const accY = (targetDy - slab.dy) * stiffness - slab.velY * damping;
      const accR = (targetRotZ - slab.rotZ) * (stiffness * 0.85) - slab.velRotZ * damping;

      slab.velX += accX * dt;
      slab.velY += accY * dt;
      slab.velRotZ += accR * dt;

      slab.dx += slab.velX * dt;
      slab.dy = clamp(slab.dy + slab.velY * dt, 0, MAX_HEAVE_DISPLACEMENT);
      slab.rotZ = clamp(slab.rotZ + slab.velRotZ * dt, -maxTilt, maxTilt);

      // Hard physical clamp against steel lateral platens accounting for rotated slab corners!
      const rotReachX = halfW * Math.cos(slab.rotZ) + effHalfH * Math.abs(Math.sin(slab.rotZ));
      if (col === 0) {
        const minAllowedDx = Math.max(-maxOuterEdgeX - colLeftRest, -maxOuterEdgeX + rotReachX - slab.restCenterX);
        if (slab.dx < minAllowedDx) {
          slab.dx = minAllowedDx;
          if (slab.velX < 0) slab.velX = 0;
        }
      }
      if (col === 3) {
        const maxAllowedDx = Math.min(maxOuterEdgeX - colRightRest, maxOuterEdgeX - rotReachX - slab.restCenterX);
        if (slab.dx > maxAllowedDx) {
          slab.dx = maxAllowedDx;
          if (slab.velX > 0) slab.velX = 0;
        }
      }
      // Hard floor clamp on Tier 0 so rotated bottom corners never sink below CHAMBER_ROOF_Y into casing walls
      if (tier === 0) {
        const minRoofDy = Math.max(0, halfW * Math.abs(Math.sin(slab.rotZ)) + halfH * (Math.cos(slab.rotZ) - 1));
        if (slab.dy < minRoofDy) {
          slab.dy = minRoofDy;
          if (slab.velY < 0) slab.velY = 0;
        }
      }

      // Subtle Z-breathing & pitch jostle on high pressure
      slab.dz = (col === 1 || col === 2 ? 0.005 : -0.003) * P * Math.sin(tier * 1.7 + col);
      slab.rotX = ((col % 2 === 0 ? 1 : -1) * 0.012 * P * (tier + 1)) / 3;

      // Radiant fracture-wall seam heat absorbed by this slab
      const adjDikeHeat = Math.max(
        (col === 1 || col === 2 ? this.dikes[0].temperature * centerReach : 0),
        (col === 0 || col === 1 ? this.dikes[1].temperature * leftReach : 0),
        (col === 2 || col === 3 ? this.dikes[2].temperature * rightReach : 0)
      );
      const adjSillHeat = Math.max(
        tier <= 1 ? this.sills[0].temperature * lowerSillSupport : 0,
        tier >= 1 ? this.sills[1].temperature * upperSillSupport : 0
      );
      const targetHeat = clamp(Math.max(adjDikeHeat, adjSillHeat) * 0.95 + P * 0.14, 0, 1);
      slab.seamHeat += (targetHeat - slab.seamHeat) * (1 - Math.exp(-dt * 4.0));

      if (slab.dy > maxHeave) maxHeave = slab.dy;
    }
    this.heaveDisplacement = clamp(maxHeave, 0, MAX_HEAVE_DISPLACEMENT);

    // 7. Extruding, Cooling & Re-Cracking Viscoplastic Pahoehoe Magma Lobes
    this.updatePahoehoeLobes(dt);
  }

  private updatePahoehoeLobes(dt: number): void {
    const P = this.chamberPressure;
    this.lobeSpawnCooldown = Math.max(0, this.lobeSpawnCooldown - dt);

    // Spawn fresh magma tongues when active dikes breach the top crest OR pressurized
    // compressional sills break out along the front/back bedding-plane fissures
    if (this.lobeSpawnCooldown <= 0 && P > 0.34) {
      let spawned = false;

      // Priority 1: Extensional crest dike eruptions
      for (let i = 0; i < this.dikes.length; i++) {
        const d = this.dikes[i];
        if (d.breachedCrest && d.temperature > 0.42) {
          this.spawnLobe('crest-dike', i);
          spawned = true;
          break;
        }
      }

      // Priority 2: Compressional bedding-plane sill side-ooze blisters (organic sparse distribution, max 6 active)
      if (!spawned) {
        let activeFlankLobes = 0;
        for (let k = 0; k < this.lobes.length; k++) {
          if (this.lobes[k].active && this.lobes[k].ventType === 'flank-sill') {
            activeFlankLobes++;
          }
        }
        if (activeFlankLobes < 6) {
          for (let i = 0; i < this.sills.length; i++) {
            const s = this.sills[i];
            if (s.thickness > 0.032 && s.halfWidth > 0.44 && s.temperature > 0.44) {
              this.spawnLobe('flank-sill', i);
              spawned = true;
              break;
            }
          }
        }
      }

      if (spawned) {
        this.lobeSpawnCooldown = clamp(0.68 - P * 0.34, 0.28, 0.72);
      }
    }

    // Update all active extruded lobes: anchor tracking, viscous spread, blackbody cooling & re-cracking
    for (let i = 0; i < this.lobes.length; i++) {
      const lobe = this.lobes[i];
      if (!lobe.active) continue;

      lobe.ageSec += dt;

      // Determine whether this lobe's feeder vent is currently supplying hot pressurized magma
      let ventOverpressure = 0;
      let ventTemp = 0;
      let parentSlabDy = 0;
      let parentSlabDx = 0;
      let parentRotSlopeY = 0;

      if (lobe.ventType === 'crest-dike') {
        const d = this.dikes[lobe.ventIndex];
        ventOverpressure = d.breachedCrest ? clamp((P - 0.30) * 1.4, 0, 1) : 0;
        ventTemp = d.temperature;
        // Anchor directly onto the upper basalt cap slab directly beneath this lobe
        const col = lobe.localOffsetX < -0.35 ? 0 : lobe.localOffsetX < 0 ? 1 : lobe.localOffsetX < 0.35 ? 2 : 3;
        const slab = this.slabs[2 * SLAB_COLUMNS + col];
        parentSlabDy = slab.dy;
        parentSlabDx = slab.dx;
        // Account for slab anticlinal roll slope so lobe stays seated flush on tilted basalt crest
        parentRotSlopeY = Math.sin(slab.rotZ) * (lobe.localOffsetX - slab.restCenterX);
      } else {
        const s = this.sills[lobe.ventIndex];
        ventOverpressure = s.thickness > 0.022 ? clamp((P - 0.26) * 1.35 * this.sillFraction, 0, 1) : 0;
        ventTemp = s.temperature;
        const tierBelow = lobe.ventIndex;
        const tierAbove = lobe.ventIndex + 1;
        const col = lobe.localOffsetX < -0.35 ? 0 : lobe.localOffsetX < 0 ? 1 : lobe.localOffsetX < 0.35 ? 2 : 3;
        const slabBelow = this.slabs[tierBelow * SLAB_COLUMNS + col];
        const slabAbove = this.slabs[tierAbove * SLAB_COLUMNS + col];
        parentSlabDy = 0.5 * (slabBelow.dy + slabAbove.dy);
        parentSlabDx = 0.5 * (slabBelow.dx + slabAbove.dx);
        parentRotSlopeY = Math.sin(0.5 * (slabBelow.rotZ + slabAbove.rotZ)) * (lobe.localOffsetX - slabBelow.restCenterX);
      }

      // Viscous inflation & sluggish gravitational pahoehoe creep while hot and fed
      const fluidity = smoothstep(0.18, 0.85, lobe.coreTemp);
      if (ventOverpressure > 0.05 && lobe.growth < 1.0) {
        const growStep = (0.38 + 0.52 * ventOverpressure) * (0.35 + 0.65 * fluidity) * dt;
        lobe.growth = Math.min(1.0, lobe.growth + growStep);
        // Subtle viscous flow creep outward along the rock crest or face (stays grounded in seam)
        lobe.localOffsetX += lobe.flowX * growStep * 0.042;
        lobe.localOffsetY += lobe.flowY * growStep * 0.014;
        lobe.localOffsetZ += lobe.flowZ * growStep * 0.022;
      }

      // Sync world coordinates strictly to the moving overburden slab so lobes never float
      if (lobe.ventType === 'crest-dike') {
        const baseY = evalBeddingPlaneY(STRATA_TIERS, lobe.localOffsetX, lobe.localOffsetZ);
        lobe.posX = clamp(lobe.localOffsetX + parentSlabDx, -CORE_HALF_W + 0.05, CORE_HALF_W - 0.05);
        lobe.posY = baseY + parentSlabDy + parentRotSlopeY + lobe.localOffsetY;
        lobe.posZ = clamp(lobe.localOffsetZ, -CORE_HALF_D + 0.06, CORE_HALF_D - 0.06);
      } else {
        const bedY = evalBeddingPlaneY(lobe.ventIndex + 1, lobe.localOffsetX, lobe.localOffsetZ);
        lobe.posX = clamp(lobe.localOffsetX + parentSlabDx, -CORE_HALF_W + 0.08, CORE_HALF_W - 0.08);
        lobe.posY = bedY + parentSlabDy + parentRotSlopeY + lobe.localOffsetY;
        lobe.posZ = clamp(lobe.localOffsetZ, -(CORE_HALF_D + 0.018), CORE_HALF_D + 0.018);
      }

      // Live Cooling & Obsidian Re-Cracking Dynamics:
      // Outer skin cools quickly via radiative + air cooling into glossy jet-black obsidian (~5-7s)
      // while core stays warmer longer; a fresh pressure surge fractures the dark obsidian skin!
      const freshFeed = ventOverpressure * ventTemp;
      const surgeCrack =
        (this.pressureSurge * 1.45 + Math.max(0, freshFeed - lobe.crustTemp) * 0.95) *
        smoothstep(0.02, 0.48, 0.55 - lobe.crustTemp);

      if (surgeCrack > 0.04 && P > 0.35) {
        lobe.crackIntensity = Math.min(1.0, lobe.crackIntensity + surgeCrack * dt * 3.6);
        lobe.coreTemp = Math.min(0.96, lobe.coreTemp + surgeCrack * dt * 0.85);
        lobe.wrinkle = Math.min(1.0, lobe.wrinkle + dt * 0.28);
      } else {
        lobe.crackIntensity *= Math.exp(-dt * 0.72);
      }

      // Outer crust cools strongly into glossy volcanic obsidian over ~4-6s when unpressurized,
      // while staying smoldering cinnabar/blistered while hot pressurized magma actively feeds the vent
      const crustFloor = clamp(freshFeed * (0.50 - lobe.growth * 0.16) + 0.02, 0.02, 0.92);
      const crustCoolRate =
        (0.115 + 0.24 * lobe.crustTemp * lobe.crustTemp) * (1.0 - 0.35 * freshFeed);
      if (freshFeed > lobe.crustTemp && lobe.growth < 0.72) {
        lobe.crustTemp += (freshFeed * 0.92 - lobe.crustTemp) * (1 - Math.exp(-dt * 2.2));
      } else if (lobe.crustTemp < crustFloor) {
        lobe.crustTemp += (crustFloor - lobe.crustTemp) * (1 - Math.exp(-dt * 1.8));
      } else {
        lobe.crustTemp = Math.max(crustFloor, lobe.crustTemp - crustCoolRate * dt);
      }

      // Core cools more slowly than crust so fresh cracks reveal glowing cinnabar-amber interior
      const coreFloor = clamp(freshFeed * 0.72 + 0.03, 0.03, 0.98);
      const coreCoolRate = (0.062 + 0.14 * lobe.coreTemp * lobe.coreTemp) * (1.0 - 0.65 * freshFeed);
      if (freshFeed > lobe.coreTemp && lobe.growth < 0.85) {
        lobe.coreTemp += (freshFeed - lobe.coreTemp) * (1 - Math.exp(-dt * 1.8));
      } else if (lobe.coreTemp < coreFloor) {
        lobe.coreTemp += (coreFloor - lobe.coreTemp) * (1 - Math.exp(-dt * 1.6));
      } else {
        lobe.coreTemp = Math.max(coreFloor, lobe.coreTemp - coreCoolRate * dt);
      }

      // Accumulate ropy pahoehoe skin wrinkles as viscous melt cools
      if (lobe.crustTemp > 0.08 && lobe.crustTemp < 0.75) {
        lobe.wrinkle = Math.min(1.0, lobe.wrinkle + dt * 0.12);
      }
    }
  }

  private spawnLobe(ventType: 'crest-dike' | 'flank-sill', ventIndex: number): void {
    // Prefer an unused slot; if full, recycle the oldest fully quenched lobe
    let slotIdx = -1;
    for (let i = 0; i < this.lobes.length; i++) {
      if (!this.lobes[i].active) {
        slotIdx = i;
        break;
      }
    }
    if (slotIdx === -1) {
      let oldestAge = -1;
      for (let i = 0; i < this.lobes.length; i++) {
        if (this.lobes[i].ageSec > oldestAge) {
          oldestAge = this.lobes[i].ageSec;
          slotIdx = i;
        }
      }
    }

    const lobe = this.lobes[slotIdx];
    this.lobeSpawnCursor++;
    const s1 = hashUnit(this.lobeSpawnCursor * 3.17 + ventIndex * 11.3);
    const s2 = hashUnit(this.lobeSpawnCursor * 7.41 + slotIdx * 5.9);
    const s3 = hashUnit(this.lobeSpawnCursor * 13.7 + ventIndex * 2.3);

    lobe.active = true;
    lobe.ventType = ventType;
    lobe.ventIndex = ventIndex;
    lobe.ageSec = 0;
    lobe.growth = 0.18;
    lobe.coreTemp = clamp(0.84 + 0.15 * this.chamberPressure, 0.75, 0.99);
    lobe.crustTemp = clamp(0.68 + 0.22 * this.chamberPressure, 0.58, 0.94);
    lobe.crackIntensity = 0.65;
    lobe.wrinkle = 0.15 + s3 * 0.22;

    if (ventType === 'crest-dike') {
      const seamX = DIKE_SEAM_X[ventIndex];
      const zPos = (s1 - 0.5) * (CORE_DEPTH - 0.24);
      const side = s2 > 0.5 ? 1 : -1;
      lobe.localOffsetX = seamX + side * (0.015 + s3 * 0.042);
      // Slightly seated into the crest seam so growing pillows never float above basalt
      lobe.localOffsetY = -0.004 + s2 * 0.008;
      lobe.localOffsetZ = zPos;
      lobe.flowX = side * (0.65 + s3 * 0.45);
      lobe.flowY = -0.05;
      lobe.flowZ = (s2 - 0.5) * 0.5;
      lobe.radiusX = 0.072 + s1 * 0.064;
      lobe.radiusY = 0.032 + s2 * 0.022;
      lobe.radiusZ = 0.092 + s3 * 0.072;
      lobe.yaw = Math.atan2(lobe.flowX, lobe.flowZ);
      lobe.pitch = -0.08 - s3 * 0.12;
    } else {
      const sill = this.sills[ventIndex];
      const frontOrBack = s2 > 0.34 ? 1 : -1;
      // Spread irregularly across central-to-mid sill span, alternating lower/upper sills cleanly
      const xPos = (s1 - 0.5) * Math.min(CORE_WIDTH - 0.36, sill.halfWidth * 1.55);
      lobe.localOffsetX = xPos;
      lobe.localOffsetY = -0.008 + (s3 - 0.5) * 0.010;
      // Tucked into the open bedding lip so molten tongues ooze out of the fissure instead of floating on the face
      lobe.localOffsetZ = frontOrBack * (CORE_HALF_D - 0.008);
      lobe.flowX = (s1 - 0.5) * 0.28;
      lobe.flowY = -0.58;
      lobe.flowZ = frontOrBack * 0.42;
      lobe.radiusX = 0.078 + s1 * 0.068;
      lobe.radiusY = 0.026 + s2 * 0.018;
      lobe.radiusZ = 0.038 + s3 * 0.026;
      lobe.yaw = (s1 - 0.5) * 0.32;
      lobe.pitch = frontOrBack * (0.24 + s3 * 0.22);
    }

    lobe.posX = lobe.localOffsetX;
    lobe.posY = ventType === 'crest-dike' ? OVERBURDEN_TOP_Y + lobe.localOffsetY : SILL_BED_Y[ventIndex];
    lobe.posZ = lobe.localOffsetZ;
  }

  /** Returns the current geological stress regime classification. */
  getStressRegime(): StressRegime {
    if (this.tectonicCompression < -0.22) return 'extensional';
    if (this.tectonicCompression > 0.22) return 'compressional';
    return 'transitional';
  }

  /** Returns the active extruded pahoehoe lobe count. */
  getActiveLobeCount(): number {
    let count = 0;
    for (let i = 0; i < this.lobes.length; i++) {
      if (this.lobes[i].active) count++;
    }
    return count;
  }

  /** Returns read-only telemetry snapshot matching window.__DAY23__.getState(). */
  getState(): Day23Telemetry {
    let lobeCount = 0;
    let sumCrustTemp = 0;
    let quenchedCount = 0;

    for (let i = 0; i < this.lobes.length; i++) {
      const l = this.lobes[i];
      if (!l.active) continue;
      lobeCount++;
      sumCrustTemp += l.crustTemp;
      if (l.crustTemp < 0.20) quenchedCount++;
    }

    const maxSillThick = Math.max(this.sills[0].thickness, this.sills[1].thickness);

    return {
      chamberPressure: Number(this.chamberPressure.toFixed(4)),
      tectonicCompression: Number(this.tectonicCompression.toFixed(4)),
      activeDikeHeight: Number(this.activeDikeHeight.toFixed(4)),
      sillFraction: Number(this.sillFraction.toFixed(4)),
      heaveDisplacement: Number(this.heaveDisplacement.toFixed(4)),
      extrudedLobes: lobeCount,
      stressRegime: this.getStressRegime(),
      pistonTravel: Number(this.pistonTravel.toFixed(4)),
      platenOffset: Number(this.platenOffset.toFixed(4)),
      maxSillThickness: Number(maxSillThick.toFixed(4)),
      avgLobeCrustTemp: lobeCount > 0 ? Number((sumCrustTemp / lobeCount).toFixed(4)) : 0,
      quenchedObsidianFraction: lobeCount > 0 ? Number((quenchedCount / lobeCount).toFixed(4)) : 0,
      simTimeSec: Number(this.simTimeSec.toFixed(3)),
    };
  }
}
