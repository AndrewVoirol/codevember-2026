import {
  AIR_CAVITY,
  AIR_SKY,
  AIR_SOLID,
  ARCH_SPAN_LIMIT,
  CELL_EMPTY,
  CELL_LOOSE_SAND,
  CELL_PACKED_SAND,
  CELL_PEBBLE,
  GRID_CELLS,
  GRID_H,
  GRID_W,
  MAX_3D_GRAINS,
  MAX_CANTILEVER_SPAN,
  MAX_DUST_MOTES,
  MIN_FULL_ARCH_ROOF,
  PEBBLE_ABUTMENT_BONUS,
  SAND_PIGMENTS,
  SURFACE_FRAC,
  linearToSrgbByte,
} from './constants';
import {
  AntColony,
  type ColonyGridHost,
  type ColonyStats,
  type DiggerAgent,
  type DiggerState,
} from './antColony';

export type { DiggerAgent, DiggerState, ColonyStats };

export interface SandSimMetrics {
  excavatedCells: number;
  activeFallingGrains: number;
  collapseEvents: number;
  diggerCount: number;
  tailingsDeposited: number;
  stepCount: number;
  active3DGrains: number;
  activeDustMotes: number;
  colony: ColonyStats;
}

export interface SandStepOptions {
  /** Allow autonomous worker ants to move, excavate, and haul grains (default true) */
  enableDiggers?: boolean;
  /** Run granular arch stability & shear-fault check (default true) */
  enableArchCheck?: boolean;
}

export interface PebbleSpec {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle: number;
  pigmentId: number;
  /** Smooth organic river-stone contour coefficients (shared 1:1 with TSL shader!) */
  asym1: number;
  asym2: number;
}

/** Live 3D tumbling mineral grain synchronized to active pouring, carving & avalanche cascades */
export interface Grain3D {
  active: boolean;
  x: number;
  y: number;
  z: number; // Normalized [-1..+1] inside glass cavity depth
  vx: number;
  vy: number;
  vz: number;
  rotX: number;
  rotY: number;
  rotZ: number;
  vRot: number;
  scale: number;
  pigmentId: number;
  shade: number;
  age: number;
  life: number;
}

/** Subtle warm mineral dust mote bloomed on heavy avalanche impacts & brisk carving */
export interface DustMote3D {
  active: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  size: number;
  age: number;
  life: number;
  pigmentId: number;
  alpha: number;
}

const GRAVITY_ACCEL = 0.34;
const MAX_FALL_SPEED = 4.85;
/** Deep confined consolidation only (>= 7s still AND >= 18 cells overburden): surface dunes stay permanently loose! */
const RECOMPACT_TICKS = 420;
const RECOMPACT_MIN_DEPTH = 18;
const SKY_MIN_OPEN_SPAN = 22;
/** Critical unbuttressed vertical packed-sand wall height before active Rankine wedge shear */
const CRITICAL_CLIFF_HEIGHT = 17;
/** Dry surface-horizon cut depth that ravels sharp 90-deg lips into natural ~46-deg funnels */
const DRY_RIM_FUNNEL_DROP = 7;

/** Deterministic Mulberry32 PRNG (zero allocations) */
class SeededRng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0 || 1;
  }
  reseed(seed: number): void {
    this.s = seed >>> 0 || 1;
  }
  next(): number {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

/**
 * Pure CPU Granular Sand Terrarium Simulation (`300 x 220` cells):
 * - Stratified geological mineral bands with differential compaction draping around river pebbles
 * - Terzaghi granular arching stability vs. arched Mohr-Coulomb conjugate shear-fault cave-ins
 * - Momentum-coupled 2D granular physics (`velX`, `velY`) with impact-triggered slope avalanches,
 *   pebble-crown ricochet fanning, and exact 1:1 grain mass conservation
 * - Stigmergic 12-worker-ant colony (`AntColony`) with strict open-corridor BFS navigation,
 *   rescue-digging after cave-ins, antennae-tap greetings, and crater-rim tailings mounds
 * - Live zero-GC 3D tumbling mineral granules & soft avalanche dust mote pools
 */
export class SandSimulation implements ColonyGridHost {
  readonly width = GRID_W;
  readonly height = GRID_H;
  readonly cellCount = GRID_CELLS;

  // Core per-cell simulation state (SoA, pre-allocated once)
  readonly cellType = new Uint8Array(GRID_CELLS);
  readonly pigmentId = new Uint8Array(GRID_CELLS);
  readonly grainShade = new Uint8Array(GRID_CELLS);
  readonly bgPigmentId = new Uint8Array(GRID_CELLS);
  readonly bgShade = new Uint8Array(GRID_CELLS);
  readonly bondHealth = new Uint8Array(GRID_CELLS);
  readonly faultDelay = new Uint8Array(GRID_CELLS);
  readonly velX = new Float32Array(GRID_CELLS);
  readonly velY = new Float32Array(GRID_CELLS);
  readonly fracX = new Float32Array(GRID_CELLS);
  readonly fracY = new Float32Array(GRID_CELLS);
  readonly restTicks = new Uint16Array(GRID_CELLS);
  readonly movedStamp = new Uint32Array(GRID_CELLS);

  // Derived optical & structural fields
  readonly airClass = new Uint8Array(GRID_CELLS);
  readonly recessField = new Uint8Array(GRID_CELLS);
  readonly wallDistField = new Uint8Array(GRID_CELLS);
  readonly stressField = new Uint8Array(GRID_CELLS);

  // Per-column surface tracking (for 3D top sand crest mesh & air/pour queries)
  readonly topSolidY = new Int16Array(GRID_W);
  readonly overburdenTopY = new Int16Array(GRID_W);
  readonly surfaceY = new Float32Array(GRID_W);
  readonly surfaceRgb = new Float32Array(GRID_W * 3);

  // Packed RGBA8 texture buffers uploaded directly to Three.js DataTextures
  /** RGBA8 SRGB: RGB = smooth grain/strata color, A = airborne falling loose-sand mask */
  readonly colorTexData = new Uint8Array(GRID_CELLS * 4);
  /** RGBA8 Linear: R = 3x3 smoothed cohesive solid mask, G = cavity recess depth, B = solid wall depth, A = packed arch stress */
  readonly fieldTexData = new Uint8Array(GRID_CELLS * 4);

  // Pre-allocated scratch buffers for zero-GC arch/distance/kinematic passes
  private readonly rowOpenSpan = new Int16Array(GRID_W);
  private readonly colHeight = new Int16Array(GRID_W);
  private readonly smoothHorizon = new Float32Array(GRID_W);
  private readonly skyFloorY = new Int16Array(GRID_W);
  private readonly collapseCooldown = new Uint16Array(GRID_W);
  private readonly startCellType = new Uint8Array(GRID_CELLS);
  private readonly bedSupportMask = new Uint8Array(GRID_CELLS);
  private readonly rawCohesiveMask = new Uint8Array(GRID_CELLS);
  private readonly distBufA = new Uint8Array(GRID_CELLS);
  private readonly distBufB = new Uint8Array(GRID_CELLS);

  // Embedded river pebbles for current seed
  readonly pebbles: PebbleSpec[] = [];

  // Stigmergic worker ant colony (strictly open-corridor navigation + awareness)
  readonly colony = new AntColony();

  // Zero-GC 3D tumbling sand granules & subtle avalanche dust pools
  readonly grains3D: Grain3D[] = [];
  readonly dust3D: DustMote3D[] = [];
  active3DGrains = 0;
  activeDustMotes = 0;
  private grain3DCursor = 0;
  private dust3DCursor = 0;

  // Pre-baked linear RGB + sRGB byte LUT per (pigmentId, shadeByte) for zero-pow O(1) packing
  readonly pigmentRgbLinear: Float32Array;
  private readonly pigmentContrast: Float32Array;
  private readonly srgbByteLut: Uint8Array;

  // Simulation metrics & state
  seed = 1;
  stepCount = 0;
  excavatedCells = 0;
  activeFallingGrains = 0;
  collapseEvents = 0;
  tailingsDeposited = 0;
  texturesDirty = true;

  private readonly rng = new SeededRng(1);
  private moveTickId = 1;
  private pourPigmentPhase = 0;
  private starterMouthX = 114;
  private activeFaultCells = 0;

  constructor(initialSeed = 1) {
    this.pigmentRgbLinear = new Float32Array(SAND_PIGMENTS.length * 3);
    this.pigmentContrast = new Float32Array(SAND_PIGMENTS.length);
    this.srgbByteLut = new Uint8Array(SAND_PIGMENTS.length * 256 * 3);
    for (let i = 0; i < SAND_PIGMENTS.length; i++) {
      const p = SAND_PIGMENTS[i];
      this.pigmentRgbLinear[i * 3] = p.rgbLinear[0];
      this.pigmentRgbLinear[i * 3 + 1] = p.rgbLinear[1];
      this.pigmentRgbLinear[i * 3 + 2] = p.rgbLinear[2];
      this.pigmentContrast[i] = p.grainContrast;

      for (let s = 0; s < 256; s++) {
        const shadeMul = 1.0 + ((s - 128) / 128) * p.grainContrast;
        const lutOff = ((i << 8) | s) * 3;
        this.srgbByteLut[lutOff] = linearToSrgbByte(p.rgbLinear[0] * shadeMul);
        this.srgbByteLut[lutOff + 1] = linearToSrgbByte(p.rgbLinear[1] * (shadeMul * 0.992));
        this.srgbByteLut[lutOff + 2] = linearToSrgbByte(p.rgbLinear[2] * (shadeMul * 0.978));
      }
    }

    for (let i = 0; i < MAX_3D_GRAINS; i++) {
      this.grains3D.push({
        active: false,
        x: 0,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        rotX: 0,
        rotY: 0,
        rotZ: 0,
        vRot: 0,
        scale: 1,
        pigmentId: 2,
        shade: 128,
        age: 0,
        life: 30,
      });
    }

    for (let i = 0; i < MAX_DUST_MOTES; i++) {
      this.dust3D.push({
        active: false,
        x: 0,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        size: 1,
        age: 0,
        life: 45,
        pigmentId: 1,
        alpha: 0,
      });
    }

    this.reset(initialSeed);
  }

  /** Backward-compatible getter returning the 12 autonomous worker ants */
  get diggers(): DiggerAgent[] {
    return this.colony.ants;
  }

  nextRandom(): number {
    return this.rng.next();
  }

  /** Cell linear index (y = 0 bottom, y = height - 1 top) */
  idx(x: number, y: number): number {
    return y * GRID_W + x;
  }

  /**
   * Resets to a freshly stratified geological sand bed with compacted pebble-draping strata,
   * embedded quartz/basalt river pebbles, an organic starter ant-farm burrow network
   * with bulbous chambers & layered volcanic crater rim, and 12 worker ants.
   */
  reset(newSeed = this.seed + 1, seedStarterBurrow = true): void {
    this.seed = newSeed;
    this.rng.reseed(newSeed * 7919 + 104729);
    this.stepCount = 0;
    this.excavatedCells = 0;
    this.activeFallingGrains = 0;
    this.collapseEvents = 0;
    this.tailingsDeposited = 0;
    this.moveTickId = 1;
    this.pourPigmentPhase = 0;
    this.activeFaultCells = 0;
    this.active3DGrains = 0;
    this.activeDustMotes = 0;
    this.grain3DCursor = 0;
    this.dust3DCursor = 0;

    this.cellType.fill(CELL_EMPTY);
    this.pigmentId.fill(0);
    this.grainShade.fill(128);
    this.bgPigmentId.fill(0);
    this.bgShade.fill(128);
    this.bondHealth.fill(0);
    this.faultDelay.fill(0);
    this.velX.fill(0);
    this.velY.fill(0);
    this.fracX.fill(0);
    this.fracY.fill(0);
    this.restTicks.fill(0);
    this.movedStamp.fill(0);
    this.stressField.fill(0);
    this.collapseCooldown.fill(0);
    this.bedSupportMask.fill(0);
    this.pebbles.length = 0;

    for (let i = 0; i < MAX_3D_GRAINS; i++) this.grains3D[i].active = false;
    for (let i = 0; i < MAX_DUST_MOTES; i++) this.dust3D[i].active = false;

    const baseSurface = Math.round(GRID_H * SURFACE_FRAC); // ~176

    // 1. Choose river pebble geometry specs FIRST so surrounding geological strata
    //    realistically drape & compact over and underneath each stone!
    this.generateRiverPebbleSpecs(baseSurface);

    // Geological wave phases for organic wavy bedding planes (irrational frequencies)
    const ph1 = this.rng.next() * Math.PI * 2;
    const ph2 = this.rng.next() * Math.PI * 2;
    const ph3 = this.rng.next() * Math.PI * 2;
    const ph4 = this.rng.next() * Math.PI * 2;

    for (let x = 0; x < GRID_W; x++) {
      const nx = x / GRID_W;
      // Gentle natural dune crest topography across the top horizon
      const crestWave =
        Math.sin(nx * Math.PI * 2.15 + ph1) * 5.0 +
        Math.sin(nx * Math.PI * 4.73 + ph2) * 2.3 +
        Math.cos(nx * Math.PI * 8.31 + ph3) * 1.05;
      const colTop = Math.max(
        150,
        Math.min(GRID_H - 28, Math.round(baseSurface + crestWave))
      );

      for (let y = 0; y < GRID_H; y++) {
        const i = y * GRID_W + x;
        const ny = y / GRID_H;
        let strataWarp =
          Math.sin(nx * Math.PI * 2.35 + ph1 + ny * 1.4) * 9.2 +
          Math.sin(nx * Math.PI * 5.17 + ph2 - ny * 2.2) * 4.0 +
          Math.cos(nx * Math.PI * 11.4 + ph4 + ny * 3.1) * 1.5;

        // Subtle geological differential compaction draping around each embedded river stone
        for (let k = 0; k < this.pebbles.length; k++) {
          const p = this.pebbles[k];
          const dx = x - p.cx;
          const dy = y - p.cy;
          const dist2 = dx * dx + dy * dy;
          const inflR = Math.max(p.rx, p.ry) * 2.3;
          if (dist2 < inflR * inflR) {
            const w = Math.exp(-dist2 / (p.rx * p.rx * 1.65));
            // Strata bow slightly up above pebble crown and tuck gently beneath its base
            strataWarp -= Math.sign(dy + 0.5) * w * 3.4;
          }
        }

        const depthFromTop = colTop - y;
        const geoY = y + strataWarp;

        const stratum = this.classifyStratum(geoY, depthFromTop, x, y);
        const grainJitter = Math.floor(
          128 +
            (this.rng.next() - 0.5) * 124 +
            Math.sin(geoY * 1.75) * 18 +
            Math.cos(x * 0.27 + geoY * 0.65) * 10
        );
        const clampedShade = Math.max(12, Math.min(244, grainJitter));

        // Always store undisturbed geological stratum in bgPigmentId/bgShade for tunnel back-walls
        this.bgPigmentId[i] = stratum;
        this.bgShade[i] = clampedShade;

        if (y <= colTop) {
          this.cellType[i] = CELL_PACKED_SAND;
          this.pigmentId[i] = stratum;
          this.grainShade[i] = clampedShade;
          this.bondHealth[i] = 255;
        }
      }
    }

    // 2. Rasterize the 8 hard river pebbles into the cell grid (immovable structural lintels!)
    this.rasterizeRiverPebbles();

    // 3. Carve organic ant-farm starter burrow network (curved galleries, pebble arches,
    //    rounded oval chambers & a layered volcanic crater cone!)
    this.starterMouthX = 112 + Math.floor(this.rng.next() * 20);
    if (seedStarterBurrow) {
      this.carveStarterBurrow(this.starterMouthX);
    }

    // 4. Compute surface horizons & AIR_SKY / AIR_CAVITY BEFORE placing worker ants
    //    so every ant snaps cleanly to verified walkable open cells!
    this.recomputeSurfaceAndOpticalFields();

    // 5. Place 12 worker ants strictly inside open walkable cavities / surface crest
    this.colony.reset(this, this.starterMouthX, seedStarterBurrow);

    this.packTextures();
  }

  private classifyStratum(geoY: number, depthFromTop: number, x: number, y: number): number {
    // Micro-laminated mineral ripple noise (zero polar harmonics)
    const microRipple = Math.sin(x * 0.095 + y * 0.14) * 1.35 + Math.cos(x * 0.21 - y * 0.08) * 0.7;
    const gy = geoY + microRipple;

    // Delicate mossy-umber topsoil skin near the very crest
    if (depthFromTop >= 0 && depthFromTop <= 5.5 + microRipple * 0.6) {
      return depthFromTop < 2.8 ? 6 : 4; // Moss Umbrous / Raw Umber crust
    }
    // Upper pale Bone Calcite & Warm Kaolin calcareous dune bed (high contrast for hauled sienna tailings!)
    if (gy >= 134) {
      return gy >= 152 || ((Math.floor(gy * 0.35) & 1) === 0) ? 0 : 1;
    }
    // Thin Hematite Rust ironstone parting
    if (gy >= 127 && gy < 134) {
      return 5;
    }
    // Deep Burnt Sienna terracotta band
    if (gy >= 104 && gy < 127) {
      return gy > 113 && gy < 117 ? 4 : 3;
    }
    // Luminous Yellow Ochre golden sandstone bed
    if (gy >= 72 && gy < 104) {
      return gy > 86 && gy < 90 ? 1 : 2;
    }
    // Crisp pale Warm Kaolin / Bone Calcite deep marker horizon (dramatic during fault slumps!)
    if (gy >= 55 && gy < 72) {
      return gy > 61 && gy < 67 ? 0 : 1;
    }
    // Lower Burnt Sienna + Hematite Rust ironstone beds
    if (gy >= 27 && gy < 55) {
      return gy > 39 && gy < 44 ? 5 : 3;
    }
    // Rare thin Verdigris Malachite glauconite clay vein deep near bedrock
    if (gy >= 17 && gy < 22) {
      return 7;
    }
    // Basal Raw Umber silt bedrock
    return 4;
  }

  private generateRiverPebbleSpecs(baseSurface: number): void {
    // Carefully distributed hard quartz (8) and basalt (9) pebbles serving as natural tunnel lintels
    const specs: Array<[number, number, number, number, number]> = [
      [52, 128, 11.5, 6.8, 8],   // upper-left quartz pebble (left gallery arcs cleanly beneath!)
      [164, 141, 12.2, 6.8, 9],  // central-upper basalt keystone lintel arching ABOVE the main chamber!
      [236, 132, 10.5, 6.4, 8],  // upper-right quartz pebble
      [82, 72, 12.8, 7.6, 9],    // mid-left basalt boulder
      [216, 86, 11.2, 6.8, 8],   // mid-right quartz pebble
      [258, 58, 10.0, 6.2, 9],   // lower-right basalt pebble
      [136, 38, 12.2, 7.2, 8],   // deep central quartz pebble
      [42, 38, 10.2, 6.3, 9],    // deep left basalt pebble
    ];

    for (let k = 0; k < specs.length; k++) {
      const [bx, by, brx, bry, pig] = specs[k];
      const cx = Math.max(24, Math.min(GRID_W - 24, Math.round(bx + (this.rng.next() - 0.5) * 12)));
      const cy = Math.max(20, Math.min(baseSurface - 30, Math.round(by + (this.rng.next() - 0.5) * 9)));
      const rx = brx * (0.9 + this.rng.next() * 0.22);
      const ry = bry * (0.88 + this.rng.next() * 0.22);
      const angle = (this.rng.next() - 0.5) * 0.52;
      // Smooth low-order asymmetric river-cobble taper (strictly organic, zero star harmonics)
      const asym1 = (this.rng.next() - 0.5) * 0.26; // slight egg-taper along major axis
      const asym2 = (this.rng.next() - 0.5) * 0.16; // gentle flat belly / water-worn facet
      this.pebbles.push({ cx, cy, rx, ry, angle, pigmentId: pig, asym1, asym2 });
    }
  }

  private rasterizeRiverPebbles(): void {
    for (let k = 0; k < this.pebbles.length; k++) {
      const p = this.pebbles[k];
      const cosA = Math.cos(p.angle);
      const sinA = Math.sin(p.angle);
      const pad = Math.ceil(Math.max(p.rx, p.ry)) + 3;
      const x0 = Math.max(1, p.cx - pad);
      const x1 = Math.min(GRID_W - 2, p.cx + pad);
      const y0 = Math.max(1, p.cy - pad);
      const y1 = Math.min(GRID_H - 2, p.cy + pad);

      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x - p.cx;
          const dy = y - p.cy;
          const lx = dx * cosA + dy * sinA;
          const ly = -dx * sinA + dy * cosA;
          const ux = lx / p.rx;
          const uy = ly / p.ry;
          // Exact same organic cobble contour evaluated sub-pixel in `materials.ts`!
          const taperedUy = uy / (1 + p.asym1 * ux);
          const normR2 = ux * ux + taperedUy * taperedUy + p.asym2 * uy * (1 - ux * ux * 0.5);
          if (normR2 <= 1.0) {
            const idx = y * GRID_W + x;
            this.cellType[idx] = CELL_PEBBLE;
            this.pigmentId[idx] = p.pigmentId;
            this.bgPigmentId[idx] = p.pigmentId;
            this.bondHealth[idx] = 255;
            const edgeFade = Math.max(0, 1 - Math.max(0, normR2));
            const speckle = (this.rng.next() - 0.5) * 34;
            const shade = Math.max(
              24,
              Math.min(240, Math.round(108 + edgeFade * 58 + speckle))
            );
            this.grainShade[idx] = shade;
            this.bgShade[idx] = shade;
          }
        }
      }
    }
  }

  /**
   * Carves a naturalistic, lived-in starter ant farm network:
   * - Gently sinuous arched main entrance shaft + sculpted upper junction vestibule
   * - Left foraging gallery curving cleanly beneath the upper-left quartz pebble into a rounded chamber
   * - Right arched main highway tucked under the central basalt lintel with a domed mid-chamber
   * - Deep lower fork probing the golden Ochre / Kaolin / Burnt Sienna strata
   * - Rich layered volcanic-cone tailings mound flanking the entrance mouth!
   */
  private carveStarterBurrow(mouthX: number): void {
    let mouthY = GRID_H - 1;
    while (mouthY > 20 && this.cellType[mouthY * GRID_W + mouthX] === CELL_EMPTY) {
      mouthY--;
    }

    // 1. Sinuous main entrance shaft curving smoothly down into upper vestibule & right highway
    //    passing cleanly BENEATH the central-upper basalt keystone lintel!
    this.carveOrganicGallery(
      [
        [mouthX, mouthY + 3],
        [mouthX + 2, mouthY - 14],
        [mouthX + 7, mouthY - 28],
        [mouthX + 16, mouthY - 43],
        [mouthX + 30, mouthY - 56],
        [mouthX + 46, mouthY - 62], // arches right under the basalt lintel with full clearance!
        [mouthX + 64, mouthY - 72],
        [mouthX + 78, mouthY - 86],
        [mouthX + 86, mouthY - 94],
      ],
      5.1
    );

    // 2. Left branching vaulted gallery meandering beneath upper-left quartz pebble into a bulbous chamber
    this.carveOrganicGallery(
      [
        [mouthX + 14, mouthY - 41],
        [mouthX - 2, mouthY - 47],
        [mouthX - 18, mouthY - 55],
        [mouthX - 36, mouthY - 64],
        [mouthX - 50, mouthY - 71],
      ],
      4.8
    );

    // 3. Lower central chamber fork dipping toward the deep ochre & terracotta beds
    this.carveOrganicGallery(
      [
        [mouthX + 28, mouthY - 55],
        [mouthX + 20, mouthY - 72],
        [mouthX + 16, mouthY - 87],
        [mouthX + 25, mouthY - 101],
      ],
      4.7
    );

    // 4. Sculpt three cozy rounded ant chambers (width ~13-14 cells -> safely arched under ARCH_SPAN_LIMIT=18!)
    this.carveOvalChamber(mouthX + 16, mouthY - 43, 7.6, 5.6); // Upper junction vestibule
    this.carveOvalChamber(mouthX + 46, mouthY - 62, 7.8, 5.8); // Basalt-lintel brood/gathering chamber
    this.carveOvalChamber(mouthX - 48, mouthY - 70, 7.4, 5.5); // Left granary chamber

    // 4b. Gently bevel the starter entrance funnel collar (~48 deg) so the ant nest has a natural
    //     weathered throat and deposited crater-rim grains never roll back down the shaft!
    for (let dy = 0; dy <= 7; dy++) {
      const y = mouthY - 5 + dy;
      if (y < 20 || y >= GRID_H - 2) continue;
      const halfFlare = 5 + Math.ceil(dy * 0.85);
      for (let x = Math.max(4, mouthX - halfFlare); x <= Math.min(GRID_W - 5, mouthX + 1 + halfFlare); x++) {
        const idx = y * GRID_W + x;
        if (this.cellType[idx] === CELL_PACKED_SAND || this.cellType[idx] === CELL_LOOSE_SAND) {
          this.cellType[idx] = CELL_EMPTY;
          this.bondHealth[idx] = 0;
          this.excavatedCells++;
        }
      }
    }

    // 5. Deposit a sculpted layered volcanic-cone crater mound of excavated Sienna & Ochre tailings!
    for (let k = 0; k < 240; k++) {
      const side = k % 2 === 0 ? -1 : 1;
      // Triangular distribution peaks cleanly on the outer crater rims (dist 11..25)
      const dist = 11 + Math.floor((this.rng.next() + this.rng.next()) * 8.5);
      const tx = Math.max(5, Math.min(GRID_W - 6, mouthX + side * dist));
      let ty = GRID_H - 3;
      while (ty > 4 && this.cellType[(ty - 1) * GRID_W + tx] === CELL_EMPTY) {
        ty--;
      }
      if (ty < GRID_H - 4 && this.cellType[ty * GRID_W + tx] === CELL_EMPTY) {
        const idx = ty * GRID_W + tx;
        this.cellType[idx] = CELL_LOOSE_SAND;
        // Coherent stratified deposit bands (Ochre base -> Burnt Sienna crest -> Hematite accents)
        const layerT = k / 240;
        this.pigmentId[idx] = layerT < 0.36 ? 2 : layerT < 0.84 ? 3 : 5;
        this.grainShade[idx] = Math.max(
          36,
          Math.min(232, Math.round(122 + (this.rng.next() - 0.5) * 68))
        );
        this.bondHealth[idx] = 0;
        this.velX[idx] = side * 0.22;
        this.velY[idx] = 0.18;
        this.fracX[idx] = 0;
        this.fracY[idx] = 0;
      }
    }

    // Settle starter crater mound gently to natural repose angle
    for (let s = 0; s < 36; s++) {
      this.stepLooseSandGravity();
    }
  }

  private carveOrganicGallery(pts: Array<[number, number]>, baseRadius: number): void {
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[i + 1];
      const segs = 3;
      for (let s = 0; s < segs; s++) {
        const t0 = s / segs;
        const t1 = (s + 1) / segs;
        const sx0 = x0 + (x1 - x0) * t0;
        const sy0 = y0 + (y1 - y0) * t0;
        const sx1 = x0 + (x1 - x0) * t1;
        const sy1 = y0 + (y1 - y0) * t1;
        // Subtle natural gallery pinch-and-swell (never rigid uniform tubes!)
        const swell = baseRadius * (0.92 + 0.14 * Math.sin((i * segs + s) * 1.45));
        this.excavateCapsule(sx0, sy0, sx1, sy1, swell, false);
      }
    }
  }

  private carveOvalChamber(cx: number, cy: number, rx: number, ry: number): void {
    const x0 = Math.max(4, Math.floor(cx - rx - 1));
    const x1 = Math.min(GRID_W - 5, Math.ceil(cx + rx + 1));
    const y0 = Math.max(6, Math.floor(cy - ry - 1));
    const y1 = Math.min(GRID_H - 12, Math.ceil(cy + ry + 1));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const ux = (x - cx) / rx;
        // Slightly flatter chamber floor, gently vaulted ceiling arch!
        const uy = (y - cy) / (y < cy ? ry * 0.82 : ry * 1.06);
        if (ux * ux + uy * uy <= 1.0) {
          const idx = y * GRID_W + x;
          if (this.cellType[idx] === CELL_PACKED_SAND || this.cellType[idx] === CELL_LOOSE_SAND) {
            this.cellType[idx] = CELL_EMPTY;
            this.bondHealth[idx] = 0;
            this.faultDelay[idx] = 0;
            this.velX[idx] = 0;
            this.velY[idx] = 0;
            this.restTicks[idx] = 0;
            this.excavatedCells++;
          }
        }
      }
    }
  }

  /**
   * Excavates a smooth capsule (tunnel stroke) between grid coordinates `(x0, y0)` and `(x1, y1)`.
   * Never digs through immovable river pebbles (`CELL_PEBBLE`).
   * Optionally dislodges rim grains & spawns tactile 3D sparkling granules + faint mineral dust.
   */
  excavateCapsule(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    radius = 6.2,
    scatterLooseRim = true
  ): number {
    const minX = Math.max(2, Math.floor(Math.min(x0, x1) - radius - 2));
    const maxX = Math.min(GRID_W - 3, Math.ceil(Math.max(x0, x1) + radius + 2));
    const minY = Math.max(2, Math.floor(Math.min(y0, y1) - radius - 2));
    const maxY = Math.min(GRID_H - 3, Math.ceil(Math.max(y0, y1) + radius + 2));

    const dx = x1 - x0;
    const dy = y1 - y0;
    const lenSq = dx * dx + dy * dy;
    const r2 = radius * radius;
    const rimR2 = (radius + 1.65) * (radius + 1.65);

    let removed = 0;
    let dustPigment = 2;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        let t = 0;
        if (lenSq > 1e-5) {
          t = ((x - x0) * dx + (y - y0) * dy) / lenSq;
          if (t < 0) t = 0;
          else if (t > 1) t = 1;
        }
        const px = x0 + dx * t;
        const py = y0 + dy * t;
        const ddx = x - px;
        const ddy = y - py;
        const dist2 = ddx * ddx + ddy * ddy;

        const idx = y * GRID_W + x;
        const ct = this.cellType[idx];
        if (ct === CELL_PEBBLE || ct === CELL_EMPTY) continue;

        if (dist2 <= r2) {
          const pig = this.pigmentId[idx];
          const shd = this.grainShade[idx];
          dustPigment = pig;
          this.cellType[idx] = CELL_EMPTY;
          this.bondHealth[idx] = 0;
          this.faultDelay[idx] = 0;
          this.velX[idx] = 0;
          this.velY[idx] = 0;
          this.fracX[idx] = 0;
          this.fracY[idx] = 0;
          this.restTicks[idx] = 0;
          removed++;

          // Emit crisp 3D tumbling mineral grains across the glass slice when actively carving!
          if (scatterLooseRim && this.rng.next() < 0.18) {
            this.spawnExcavatedGrain3D(
              x,
              y,
              (this.rng.next() - 0.5) * 1.15,
              -0.18 - this.rng.next() * 0.72,
              pig,
              shd
            );
          }
        } else if (
          scatterLooseRim &&
          dist2 <= rimR2 &&
          ct === CELL_PACKED_SAND
        ) {
          // Organic rim & steep-shoulder raveling:
          // - Dry upper dune horizon (`depth <= 16`) loosens generously around both sides & ceiling so cuts funnel!
          // - Deep moist burrows loosen subtle ceiling & steep shoulder grains (`ddy > -radius * 0.25`)
          const colTop = this.topSolidY[x] || Math.round(GRID_H * SURFACE_FRAC);
          const isDryUpperDune = colTop - y <= 16;
          const ravelChance = isDryUpperDune
            ? ddy > -radius * 0.35
              ? 0.42
              : 0.14
            : ddy > radius * 0.2
              ? 0.22
              : ddy > -radius * 0.15 && Math.abs(ddx) > radius * 0.65
                ? 0.09
                : 0;
          if (ravelChance > 0 && this.rng.next() < ravelChance) {
            this.cellType[idx] = CELL_LOOSE_SAND;
            this.bondHealth[idx] = 0;
            const inwardSign = ddx > 0 ? -1 : 1;
            this.velX[idx] = inwardSign * (0.18 + this.rng.next() * 0.32);
            this.velY[idx] = 0.22 + this.rng.next() * 0.34;
            this.fracX[idx] = this.rng.next() * 0.5;
            this.fracY[idx] = this.rng.next() * 0.7;
            this.restTicks[idx] = 0;
          }
        }
      }
    }

    if (removed > 0) {
      this.excavatedCells += removed;
      this.texturesDirty = true;
      this.colony.markTopologyDirty();
      if (scatterLooseRim) {
        this.colony.recordDisturbance((x0 + x1) * 0.5, (y0 + y1) * 0.5, radius + 8, 1.3);
        if (removed >= 16) {
          this.spawnDustMote((x0 + x1) * 0.5, (y0 + y1) * 0.5, dustPigment, 0.14);
        }
      }
    }
    return removed;
  }

  /**
   * Excavates an axis-aligned rectangular chamber (useful for deterministic flat-span tests
   * or cleanly hollowing chambers without touching pebbles).
   */
  excavateBox(xMin: number, yMin: number, xMax: number, yMax: number): number {
    const x0 = Math.max(2, Math.floor(xMin));
    const x1 = Math.min(GRID_W - 3, Math.ceil(xMax));
    const y0 = Math.max(2, Math.floor(yMin));
    const y1 = Math.min(GRID_H - 3, Math.ceil(yMax));
    let removed = 0;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const idx = y * GRID_W + x;
        const ct = this.cellType[idx];
        if (ct === CELL_PACKED_SAND || ct === CELL_LOOSE_SAND) {
          this.cellType[idx] = CELL_EMPTY;
          this.bondHealth[idx] = 0;
          this.faultDelay[idx] = 0;
          this.velX[idx] = 0;
          this.velY[idx] = 0;
          this.restTicks[idx] = 0;
          removed++;
        }
      }
    }
    if (removed > 0) {
      this.excavatedCells += removed;
      this.texturesDirty = true;
      this.colony.markTopologyDirty();
      this.colony.recordDisturbance((x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0) * 0.5, 1.4);
    }
    return removed;
  }

  /**
   * Pours a cohesive hourglass-funnel stream of layered mineral sand at grid coordinate `(gx, gy)`.
   * Cycles gently across natural mineral strata so poured piles build cross-bedded talus dunes
   * and trickle down open entrance shafts!
   */
  pourSandAt(gx: number, gy: number, grainBudget = 11): number {
    this.pourPigmentPhase += 0.016;
    const pourPalette = [2, 3, 0, 1, 4, 2, 5];
    const basePigment = pourPalette[Math.floor(this.pourPigmentPhase) % pourPalette.length];

    const cx = Math.max(4, Math.min(GRID_W - 5, Math.round(gx)));
    const cy = Math.max(8, Math.min(GRID_H - 3, Math.round(gy)));

    let spawned = 0;
    for (let k = 0; k < grainBudget; k++) {
      // Cohesive hourglass-nozzle stream core with natural aerodynamic dispersion
      const ox = Math.round(((this.rng.next() + this.rng.next()) - 1.0) * 2.6);
      const oy = Math.round((this.rng.next() - 0.5) * 3.2);
      const x = Math.max(3, Math.min(GRID_W - 4, cx + ox));
      const y = Math.max(4, Math.min(GRID_H - 2, cy + oy));
      const idx = y * GRID_W + x;
      if (this.cellType[idx] === CELL_EMPTY) {
        const pig =
          this.rng.next() < 0.08
            ? pourPalette[(Math.floor(this.pourPigmentPhase) + 1) % pourPalette.length]
            : basePigment;
        const shd = Math.max(
          28,
          Math.min(238, Math.round(128 + (this.rng.next() - 0.5) * 88))
        );
        this.cellType[idx] = CELL_LOOSE_SAND;
        this.pigmentId[idx] = pig;
        this.grainShade[idx] = shd;
        this.bondHealth[idx] = 0;
        this.faultDelay[idx] = 0;
        const vx = ox * 0.055 + (this.rng.next() - 0.5) * 0.26;
        const vy = 0.72 + this.rng.next() * 0.68;
        this.velX[idx] = vx;
        this.velY[idx] = vy;
        // Staggered sub-cell phase eliminates scanline banding in poured streams!
        this.fracX[idx] = (this.rng.next() - 0.5) * 0.8;
        this.fracY[idx] = this.rng.next() * 0.92;
        this.restTicks[idx] = 0;
        spawned++;

        // Dense 1:1-feel 3D tumbling mineral stream inside the glass slice cavity!
        if (this.rng.next() < 0.82) {
          this.spawnExcavatedGrain3D(x, y, vx, -vy, pig, shd);
        }
      }
    }
    if (spawned > 0) {
      this.texturesDirty = true;
      this.colony.markTopologyDirty();
      this.colony.recordDisturbance(cx, Math.max(20, this.topSolidY[cx] || cy), 10, 0.45);
    }
    return spawned;
  }

  /** Returns whether `(gx, gy)` is in the open air pour zone above the local sand crest. */
  isInAirPourZone(gx: number, gy: number): boolean {
    const x = Math.max(0, Math.min(GRID_W - 1, Math.round(gx)));
    const y = Math.max(0, Math.min(GRID_H - 1, Math.round(gy)));
    const idx = y * GRID_W + x;
    return this.airClass[idx] === AIR_SKY || y > this.topSolidY[x] + 2;
  }

  /**
   * Advances granular physics, Terzaghi arching collapse, shear-fault release,
   * 3D tumbling granules/dust, and the 12-worker-ant colony by one fixed simulation step (~1/60s).
   */
  step(options?: SandStepOptions): void {
    const enableDiggers = options?.enableDiggers ?? true;
    const enableArchCheck = options?.enableArchCheck ?? true;

    this.stepCount++;

    // Decay column collapse debounce cooldowns
    for (let x = 0; x < GRID_W; x++) {
      if (this.collapseCooldown[x] > 0) this.collapseCooldown[x]--;
    }

    // 1. Arch stability & Mohr-Coulomb shear-fault propagation (every 2 steps, immediate on step 1)
    if (enableArchCheck && (this.stepCount & 1) === 1) {
      this.evaluateArchStabilityAndFaults();
    }

    // 2. Release queued shear-fault cells layer-by-layer (progressive cracking cascade!)
    this.advanceFaultWave();

    // 3. Exact mass-conserving momentum-coupled loose sand gravity & kinetic avalanching
    const movedGrains = this.stepLooseSandGravity();

    // 4. Advance live 3D tumbling mineral granules & subtle avalanche dust motes
    this.stepParticles3D();

    // 5. Fast O(W) column surface update before ant colony step
    if (movedGrains > 0 || this.texturesDirty) {
      this.updateTopSolidColumns();
      if (movedGrains > 8) this.colony.markTopologyDirty();
      this.texturesDirty = true;
    }

    // 6. Stigmergic worker ant colony (strict open-corridor BFS, rescue digging, antennae greetings)
    if (enableDiggers) {
      this.colony.step(this);
    }
  }

  /**
   * Recomputes full optical fields (SKY vs CAVITY, Chamfer recess/wall distances) and packs
   * `colorTexData` + `fieldTexData` in < 0.6 ms via pre-baked sRGB byte LUT. Call once per render frame!
   */
  syncOpticalTextures(): void {
    this.recomputeSurfaceAndOpticalFields();
    this.packTextures();
  }

  /** Spawns a 3D tumbling mineral sand granule inside the glass slice cavity (zero GC). */
  spawnExcavatedGrain3D(
    gx: number,
    gy: number,
    vx: number,
    vy: number,
    pigmentId: number,
    shade: number
  ): void {
    const g = this.grains3D[this.grain3DCursor];
    this.grain3DCursor = (this.grain3DCursor + 1) % MAX_3D_GRAINS;
    g.active = true;
    g.x = gx + (this.rng.next() - 0.5) * 0.78;
    g.y = gy + (this.rng.next() - 0.5) * 0.78;
    g.z = (this.rng.next() - 0.5) * 1.64; // Full front-to-back glass cavity distribution
    g.vx = vx;
    g.vy = vy;
    g.vz = (this.rng.next() - 0.5) * 0.045;
    g.rotX = this.rng.next() * Math.PI * 2;
    g.rotY = this.rng.next() * Math.PI * 2;
    g.rotZ = this.rng.next() * Math.PI * 2;
    g.vRot = (this.rng.next() < 0.5 ? -1 : 1) * (0.16 + this.rng.next() * 0.34);
    g.scale = 0.68 + this.rng.next() * 0.62;
    g.pigmentId = pigmentId;
    g.shade = shade;
    g.age = 0;
    // Long safety ceiling so zero grains ever blink out mid-air; grains retire strictly on bed settle!
    g.life = 220;
  }

  /** Spawns a delicate, subtle warm mineral dust wisp inside a collapsing cavity (zero GC). */
  private spawnDustMote(gx: number, gy: number, pigmentId: number, alpha = 0.14): void {
    const m = this.dust3D[this.dust3DCursor];
    this.dust3DCursor = (this.dust3DCursor + 1) % MAX_DUST_MOTES;
    m.active = true;
    m.x = gx + (this.rng.next() - 0.5) * 6;
    m.y = gy + (this.rng.next() - 0.5) * 4;
    m.z = (this.rng.next() - 0.5) * 1.1;
    m.vx = (this.rng.next() - 0.5) * 0.22;
    m.vy = 0.04 + this.rng.next() * 0.1;
    m.size = 0.42 + this.rng.next() * 0.48;
    m.age = 0;
    m.life = 36 + Math.floor(this.rng.next() * 28);
    m.pigmentId = pigmentId;
    m.alpha = Math.min(0.18, alpha);
  }

  private stepParticles3D(): void {
    const ct = this.cellType;
    let activeG = 0;

    for (let i = 0; i < MAX_3D_GRAINS; i++) {
      const g = this.grains3D[i];
      if (!g.active) continue;
      g.age++;
      if (g.age >= g.life) {
        g.active = false;
        continue;
      }

      // Exact gravitational parity with the 2D granular solver (g.vy < 0 = downward in grid rows/tick)
      const massMul = 0.91 + (g.shade * (0.18 / 255));
      g.vy = Math.max(-MAX_FALL_SPEED * massMul, g.vy - GRAVITY_ACCEL * massMul);
      g.vx *= 0.985;
      const nx = g.x + g.vx;
      const ny = g.y + g.vy;
      g.z += g.vz;
      if (g.z > 0.86 || g.z < -0.86) {
        g.z = Math.max(-0.86, Math.min(0.86, g.z));
        g.vz *= -0.42;
      }

      const ix = Math.round(nx);
      const iy = Math.round(ny);
      if (ix < 2 || ix >= GRID_W - 2 || iy < 2 || iy >= GRID_H - 2) {
        g.active = false;
        continue;
      }

      const cellIdx = iy * GRID_W + ix;
      const cell = ct[cellIdx];
      // Collide against static solid surfaces or slow/settled sand slopes (never kill mid-stream!)
      const hitSolid =
        cell === CELL_PACKED_SAND ||
        cell === CELL_PEBBLE ||
        (cell === CELL_LOOSE_SAND && this.velY[cellIdx] < 0.45);

      if (hitSolid) {
        const impactSpeed = Math.hypot(g.vx, g.vy);
        if (cell === CELL_PEBBLE && impactSpeed > 0.42) {
          // Crisp lively river-stone ricochet & flank roll!
          const curX = Math.round(g.x);
          const pebbleSlopeSign =
            Math.abs(g.vx) > 0.16 ? Math.sign(g.vx) : ix >= curX ? (this.rng.next() < 0.5 ? -1 : 1) : -1;
          g.vy = Math.min(1.6, Math.abs(g.vy) * 0.38 + 0.14);
          g.vx = g.vx * 0.55 + pebbleSlopeSign * (0.42 + Math.min(1.2, Math.abs(g.vy) * 0.55));
          g.y = Math.max(g.y, iy + 0.65);
          g.vRot = -pebbleSlopeSign * (0.28 + this.rng.next() * 0.25);
        } else {
          // Check if an open surface slope chute exists to left or right so 3D grains saltate & roll down dunes!
          const openL =
            ix > 3 &&
            ct[iy * GRID_W + (ix - 1)] === CELL_EMPTY &&
            ct[(iy + 1) * GRID_W + (ix - 1)] === CELL_EMPTY;
          const openR =
            ix < GRID_W - 4 &&
            ct[iy * GRID_W + (ix + 1)] === CELL_EMPTY &&
            ct[(iy + 1) * GRID_W + (ix + 1)] === CELL_EMPTY;

          let rollDir = 0;
          if (openL && openR) {
            rollDir = Math.abs(g.vx) > 0.12 ? Math.sign(g.vx) : this.rng.next() < 0.5 ? -1 : 1;
          } else if (openL) {
            rollDir = -1;
          } else if (openR) {
            rollDir = 1;
          }

          if (rollDir !== 0 && impactSpeed > 0.34 && g.age < g.life - 12) {
            // Saltate / tumble tangentially along the inclined sand slipface!
            const tangSpeed = Math.min(2.1, impactSpeed * 0.68);
            g.vx = rollDir * Math.max(0.36, tangSpeed * 0.82);
            g.vy = Math.min(0.45, Math.abs(g.vy) * 0.18 - 0.16);
            g.x = ix + rollDir * 0.65;
            g.y = iy + 0.55;
            g.vRot = -rollDir * (0.24 + tangSpeed * 0.18);
          } else if (Math.abs(g.vy) > 1.55 && ct[(iy + 1) * GRID_W + ix] === CELL_EMPTY && g.age < g.life - 18) {
            // Energetic low-angle impact splash grain on flat floor
            g.vy = Math.min(0.85, Math.abs(g.vy) * 0.22);
            g.vx = (Math.abs(g.vx) > 0.15 ? Math.sign(g.vx) : this.rng.next() < 0.5 ? -1 : 1) * (0.38 + this.rng.next() * 0.52);
            g.y = iy + 0.72;
            g.age += 18;
          } else {
            // Cleanly absorbed into the settled granular bed!
            g.active = false;
            continue;
          }
        }
      } else {
        g.x = nx;
        g.y = ny;
      }

      g.rotX += g.vRot;
      g.rotY += g.vRot * 0.72;
      g.rotZ += g.vRot * 0.45;
      activeG++;
    }
    this.active3DGrains = activeG;

    let activeD = 0;
    for (let i = 0; i < MAX_DUST_MOTES; i++) {
      const m = this.dust3D[i];
      if (!m.active) continue;
      m.age++;
      if (m.age >= m.life) {
        m.active = false;
        continue;
      }
      m.vx *= 0.94;
      m.vy = m.vy * 0.93 - 0.009; // Gently drifts upward on impact then settles
      m.x = Math.max(3, Math.min(GRID_W - 4, m.x + m.vx));
      m.y = Math.max(3, Math.min(GRID_H - 4, m.y + m.vy));
      m.size += 0.018;
      activeD++;
    }
    this.activeDustMotes = activeD;
  }

  /** Lightweight per-step top solid surface tracker for ants & air-zone detection. */
  private updateTopSolidColumns(): void {
    const ct = this.cellType;
    const vy = this.velY;
    for (let x = 0; x < GRID_W; x++) {
      let topY = 0;
      for (let y = GRID_H - 1; y >= 0; y--) {
        const idx = y * GRID_W + x;
        const c = ct[idx];
        if (c === CELL_PACKED_SAND || c === CELL_PEBBLE || (c === CELL_LOOSE_SAND && vy[idx] < 0.35)) {
          topY = y;
          break;
        }
      }
      this.topSolidY[x] = topY;
    }
  }

  /**
   * Terzaghi Trapdoor Arching & Conjugate Shear-Fault Stability Evaluator:
   * - Narrow vaulted arches (span <= ARCH_SPAN_LIMIT, boosted near pebbles/walls) stay intact!
   * - Over-wide flat chambers exceed local arch span -> nucleate an arched Mohr-Coulomb
   *   conjugate slip wedge (`~64 deg`) propagating upward in cascading faulted strata bands!
   * - Detached zero-abutment floating islands immediately convert to loose falling sand.
   */
  private evaluateArchStabilityAndFaults(): void {
    let stressChanged = false;
    for (let i = 0; i < GRID_CELLS; i++) {
      const st = this.stressField[i];
      if (st > 0) {
        this.stressField[i] = st > 18 ? st - 18 : 0;
        stressChanged = true;
      }
    }
    if (stressChanged) this.texturesDirty = true;

    const ct = this.cellType;
    const fd = this.faultDelay;

    for (let y = 2; y < GRID_H - 2; y++) {
      const rowOff = y * GRID_W;
      const belowOff = (y - 1) * GRID_W;
      let x = 1;

      while (x < GRID_W - 1) {
        const i = rowOff + x;
        const belowType = ct[belowOff + x];
        const isUnsupported =
          ct[i] === CELL_PACKED_SAND &&
          fd[i] === 0 &&
          (belowType === CELL_EMPTY || (belowType === CELL_LOOSE_SAND && this.velY[belowOff + x] > 0.65));

        if (!isUnsupported) {
          x++;
          continue;
        }

        const spanL = x;
        while (x < GRID_W - 1) {
          const xi = rowOff + x;
          const bType = ct[belowOff + x];
          const unsup =
            ct[xi] === CELL_PACKED_SAND &&
            fd[xi] === 0 &&
            (bType === CELL_EMPTY || (bType === CELL_LOOSE_SAND && this.velY[belowOff + x] > 0.65));
          if (!unsup) break;
          x++;
        }
        const spanR = x - 1;
        const span = spanR - spanL + 1;

        const leftIdx = rowOff + (spanL - 1);
        const rightIdx = rowOff + (spanR + 1);
        const leftType = spanL <= 1 ? CELL_PEBBLE : ct[leftIdx];
        const rightType = spanR >= GRID_W - 2 ? CELL_PEBBLE : ct[rightIdx];

        const leftSupported =
          spanL <= 1 ||
          leftType === CELL_PEBBLE ||
          (leftType === CELL_PACKED_SAND && ct[belowOff + (spanL - 1)] !== CELL_EMPTY);
        const rightSupported =
          spanR >= GRID_W - 2 ||
          rightType === CELL_PEBBLE ||
          (rightType === CELL_PACKED_SAND && ct[belowOff + (spanR + 1)] !== CELL_EMPTY);

        const leftIsHard = spanL <= 1 || leftType === CELL_PEBBLE;
        const rightIsHard = spanR >= GRID_W - 2 || rightType === CELL_PEBBLE;

        // 1-3 cell hanging tooth stalactite (empty on both left and right) -> self-rounds naturally!
        if (!leftSupported && !rightSupported) {
          if (span <= 3) {
            for (let cx = spanL; cx <= spanR; cx++) {
              ct[rowOff + cx] = CELL_LOOSE_SAND;
              this.bondHealth[rowOff + cx] = 0;
              this.velX[rowOff + cx] = (this.rng.next() - 0.5) * 0.25;
              this.velY[rowOff + cx] = 0.28;
            }
            this.texturesDirty = true;
          } else {
            this.triggerShearWedgeCollapse(spanL, spanR, y, true);
          }
          continue;
        }

        // Single-sided cantilever overhang ledge
        if (!leftSupported || !rightSupported) {
          if (span > MAX_CANTILEVER_SPAN) {
            const dropStart = !leftSupported ? spanL : spanL + MAX_CANTILEVER_SPAN;
            const dropEnd = !leftSupported ? spanR - MAX_CANTILEVER_SPAN : spanR;
            const dropSpan = dropEnd - dropStart + 1;
            if (dropSpan >= 8) {
              this.triggerShearWedgeCollapse(dropStart, dropEnd, y, false);
            } else {
              for (let cx = dropStart; cx <= dropEnd; cx++) {
                for (let dy = 0; dy < 4 && y + dy < GRID_H - 1; dy++) {
                  const ti = (y + dy) * GRID_W + cx;
                  if (ct[ti] === CELL_PACKED_SAND) {
                    if (fd[ti] === 0) this.activeFaultCells++;
                    fd[ti] = Math.max(1, dy + 1);
                    this.stressField[ti] = 220;
                  } else {
                    break;
                  }
                }
              }
              this.texturesDirty = true;
            }
          }
          continue;
        }

        // Double-abutted arch span! Measure overburden roof thickness at center & quarter points
        const midX = (spanL + spanR) >> 1;
        const roofMid = this.measureRoofThickness(midX, y);
        const roofQ1 = this.measureRoofThickness((spanL + midX) >> 1, y);
        const roofQ3 = this.measureRoofThickness((midX + spanR) >> 1, y);
        const minRoof = Math.min(roofMid, roofQ1, roofQ3);

        let maxSpan = ARCH_SPAN_LIMIT;
        if (leftIsHard) maxSpan += PEBBLE_ABUTMENT_BONUS;
        if (rightIsHard) maxSpan += PEBBLE_ABUTMENT_BONUS;

        let hasOverheadPebbleLintel = false;
        for (let dy = 1; dy <= 5 && y + dy < GRID_H - 1; dy++) {
          if (ct[(y + dy) * GRID_W + midX] === CELL_PEBBLE) {
            hasOverheadPebbleLintel = true;
            break;
          }
        }
        if (hasOverheadPebbleLintel) maxSpan += PEBBLE_ABUTMENT_BONUS;

        if (minRoof < MIN_FULL_ARCH_ROOF) {
          const crustScale = Math.max(0.28, minRoof / MIN_FULL_ARCH_ROOF);
          maxSpan = Math.max(4, Math.floor(maxSpan * crustScale));
        }

        if (span > maxSpan) {
          this.triggerShearWedgeCollapse(spanL, spanR, y, false);
        } else if (span >= maxSpan - 4) {
          const stressNorm = Math.round(((span - (maxSpan - 5)) / 5) * 195);
          for (let cx = spanL; cx <= spanR; cx++) {
            const idx = rowOff + cx;
            if (stressNorm > this.stressField[idx]) {
              this.stressField[idx] = stressNorm;
              this.texturesDirty = true;
            }
          }
        }
      }
    }

    // Active Rankine Steep-Cliff & Dry-Surface-Rim Shear Failure (every 4 ticks + step 1):
    // Prevents unnatural 90-deg vertical concrete cliffs on tall trenches or surface cuts while
    // preserving normal vaulted subterranean ant tunnels (`faceH < CRITICAL_CLIFF_HEIGHT`)!
    if ((this.stepCount & 3) === 1) {
      this.evaluateSteepCliffAndRimSlump();
    }
  }

  private evaluateSteepCliffAndRimSlump(): void {
    const ct = this.cellType;
    const fd = this.faultDelay;

    for (let x = 3; x < GRID_W - 3; x++) {
      const colTop = this.topSolidY[x];
      if (colTop < 12) continue;

      for (let side = 0; side < 2; side++) {
        const outDir = side === 0 ? -1 : 1; // Direction of open void adjacent to column `x`
        const adjX = x + outDir;
        let y = 3;

        while (y <= colTop && y < GRID_H - 2) {
          const idx = y * GRID_W + x;
          const isExposedWall =
            ct[idx] === CELL_PACKED_SAND &&
            fd[idx] === 0 &&
            ct[y * GRID_W + adjX] === CELL_EMPTY;

          if (!isExposedWall) {
            y++;
            continue;
          }

          const yStart = y;
          while (
            y <= colTop &&
            y < GRID_H - 2 &&
            ct[y * GRID_W + x] === CELL_PACKED_SAND &&
            fd[y * GRID_W + x] === 0 &&
            ct[y * GRID_W + adjX] === CELL_EMPTY
          ) {
            y++;
          }
          const yEnd = y - 1;
          const faceH = yEnd - yStart + 1;

          // Check if `yEnd` reaches within 6 cells of the column top (even if capped by loose mound!)
          let packedAbove = false;
          for (let checkY = yEnd + 1; checkY <= Math.min(GRID_H - 2, yEnd + 6); checkY++) {
            if (ct[checkY * GRID_W + x] === CELL_PACKED_SAND || ct[checkY * GRID_W + x] === CELL_PEBBLE) {
              packedAbove = true;
              break;
            }
          }
          const reachesDryRim = !packedAbove && yEnd >= colTop - 8;
          const exceedsCriticalCliff = faceH >= CRITICAL_CLIFF_HEIGHT;
          const exceedsDryRimDrop = reachesDryRim && faceH >= DRY_RIM_FUNNEL_DROP;

          if (!exceedsCriticalCliff && !exceedsDryRimDrop) continue;

          // True uncapped Rankine slip plane (~59 deg for tall cliffs, ~48 deg for dry top dune rims):
          // Never clamp `backW` low (`11`), which previously carved a brand-new vertical cliff behind the cut!
          const slipToeY = exceedsCriticalCliff ? yStart + 3 : yStart + 2;
          const invTanTheta = exceedsCriticalCliff ? 0.58 : 0.78;
          const maxBackCut = exceedsCriticalCliff ? 34 : 18;
          let queuedWedge = 0;

          for (let wy = slipToeY; wy <= yEnd + 4 && wy < GRID_H - 2; wy++) {
            const dyAboveToe = wy - slipToeY;
            const jitter = Math.sin(wy * 0.37 + x * 0.23) * 0.65;
            const backW = Math.min(maxBackCut, Math.max(1, Math.floor(dyAboveToe * invTanTheta + jitter)));

            for (let stepIn = 0; stepIn < backW; stepIn++) {
              const tx = x - outDir * stepIn;
              if (tx < 2 || tx >= GRID_W - 2) break;
              const ti = wy * GRID_W + tx;
              if (ct[ti] === CELL_PEBBLE) break; // River cobbles buttress cliffs behind them!
              if (ct[ti] !== CELL_PACKED_SAND) continue;

              const delay = Math.max(
                1,
                Math.min(36, 1 + stepIn * 2 + Math.floor(dyAboveToe * 0.14))
              );
              if (fd[ti] === 0) {
                fd[ti] = delay;
                this.activeFaultCells++;
              } else if (delay < fd[ti]) {
                fd[ti] = delay;
              }
              // Pre-bias velX outward toward the open trench so released Rankine slices peel cleanly!
              this.velX[ti] = outDir * (0.48 + Math.min(0.45, dyAboveToe * 0.018));
              this.stressField[ti] = 230;
              queuedWedge++;
            }
          }

          if (queuedWedge >= 6 && this.collapseCooldown[x] === 0) {
            this.collapseEvents++;
            for (let cx = Math.max(0, x - 8); cx <= Math.min(GRID_W - 1, x + 8); cx++) {
              this.collapseCooldown[cx] = 22;
            }
            this.colony.recordDisturbance(x, (yStart + yEnd) * 0.5, 14, 1.5);
            this.spawnDustMote(adjX, (yStart + yEnd) * 0.5, this.pigmentId[yEnd * GRID_W + x] || 2, 0.15);
            this.texturesDirty = true;
          }
        }
      }
    }
  }

  private measureRoofThickness(x: number, startY: number): number {
    let thick = 0;
    const ct = this.cellType;
    for (let y = startY; y < GRID_H && thick < 20; y++) {
      const c = ct[y * GRID_W + x];
      if (c === CELL_PACKED_SAND || c === CELL_PEBBLE) {
        thick++;
      } else {
        break;
      }
    }
    return thick;
  }

  /**
   * Nucleates a conjugate curved Mohr-Coulomb shear-fault slip arch upward from an over-wide
   * unsupported span `[spanL..spanR]` at row `baseY`.
   * Staggers `faultDelay` progressively upward and outward from the keystone so strata crack,
   * delaminate in intact bands, and cascade down with dust puffs & colony disturbance!
   */
  private triggerShearWedgeCollapse(
    spanL: number,
    spanR: number,
    baseY: number,
    isFloatingIsland: boolean
  ): void {
    const ct = this.cellType;
    const fd = this.faultDelay;
    const midX = (spanL + spanR) >> 1;

    let inCooldown = false;
    const coolL = Math.max(0, midX - 10);
    const coolR = Math.min(GRID_W - 1, midX + 10);
    for (let cx = coolL; cx <= coolR; cx++) {
      if (this.collapseCooldown[cx] > 0) {
        inCooldown = true;
        break;
      }
    }

    let queuedCells = 0;
    const span = spanR - spanL + 1;
    const halfSpan = span * 0.5;
    const baseSlope = isFloatingIsland ? 0.14 : 0.35;
    const maxWedgeH = isFloatingIsland
      ? 44
      : Math.min(GRID_H - 2 - baseY, Math.ceil(span * 1.42) + 10);

    for (let dy = 0; dy <= maxWedgeH; dy++) {
      const y = baseY + dy;
      if (y >= GRID_H - 1) break;

      // Organic parabolic Mohr-Coulomb vault convergence with irrational multi-scale fracture contour
      const hNorm = dy / Math.max(1, maxWedgeH);
      const archCurve = dy * baseSlope * (0.85 + 0.32 * hNorm);
      const jitterL =
        Math.sin(y * 0.29 + spanL * 0.19) * 0.95 + Math.cos(y * 0.71 - spanL * 0.11) * 0.55;
      const jitterR =
        Math.cos(y * 0.31 + spanR * 0.17) * 0.95 + Math.sin(y * 0.67 - spanR * 0.13) * 0.55;
      const rowL = Math.max(1, Math.floor(spanL + archCurve + jitterL));
      const rowR = Math.min(GRID_W - 2, Math.ceil(spanR - archCurve + jitterR));
      if (rowL > rowR) break;

      const rowOff = y * GRID_W;
      let anySandInRow = false;

      for (let x = rowL; x <= rowR; x++) {
        const idx = rowOff + x;
        if (ct[idx] === CELL_PEBBLE) continue;
        if (ct[idx] !== CELL_PACKED_SAND) continue;

        // Embedded river stones shelter sand resting directly on top of them
        let shieldedByPebble = false;
        for (let py = 1; py <= 4 && y - py >= baseY; py++) {
          if (ct[(y - py) * GRID_W + x] === CELL_PEBBLE) {
            shieldedByPebble = true;
            break;
          }
        }
        if (shieldedByPebble) continue;

        anySandInRow = true;
        // Cohesive strata-preserving wave: center keystone releases first, shoulders follow smoothly
        // without random per-pixel delay static so mineral bands slump intact before fanning!
        const signedNorm = (x - midX) / Math.max(1, halfSpan);
        const centerDistNorm = Math.abs(signedNorm);
        const shoulderLag = Math.floor(centerDistNorm * 2.6);
        const delay = Math.max(1, Math.min(46, 1 + Math.floor(dy * 0.38) + shoulderLag));
        if (fd[idx] === 0) {
          fd[idx] = delay;
          this.activeFaultCells++;
        } else if (delay < fd[idx]) {
          fd[idx] = delay;
        }
        // Smooth radial-arch release velocity preserves layered geological bands!
        this.velX[idx] = signedNorm * 0.18;
        this.stressField[idx] = 255;
        queuedCells++;
      }

      if (!anySandInRow && dy > 2) break;
    }

    if (queuedCells >= 6 && !inCooldown) {
      this.collapseEvents++;
      for (let cx = Math.max(0, spanL); cx <= Math.min(GRID_W - 1, spanR); cx++) {
        this.collapseCooldown[cx] = 24;
      }
      this.colony.recordDisturbance(midX, baseY, Math.min(28, span * 0.7), 2.0);
      // Spawn subtle mineral dust wisps right along the cracking fault line!
      const dustPig = this.pigmentId[baseY * GRID_W + midX] || 2;
      for (let d = 0; d < 2; d++) {
        const dx = spanL + this.rng.next() * span;
        this.spawnDustMote(dx, baseY + 2, dustPig, 0.15);
      }
      this.texturesDirty = true;
    }
  }

  /** Converts queued shear-fault cells from `PACKED_SAND` -> `LOOSE_SAND` (exact mass conservation). */
  private advanceFaultWave(): void {
    if (this.activeFaultCells <= 0) return;
    const ct = this.cellType;
    const fd = this.faultDelay;
    let remaining = 0;
    for (let i = 0; i < GRID_CELLS; i++) {
      const d = fd[i];
      if (d === 0) continue;
      if (d > 1) {
        fd[i] = d - 1;
        this.stressField[i] = 235;
        remaining++;
      } else {
        fd[i] = 0;
        if (ct[i] === CELL_PACKED_SAND) {
          ct[i] = CELL_LOOSE_SAND;
          this.bondHealth[i] = 0;
          // Preserve pre-seeded coherent fault/Rankine outward velocity (zero salt-and-pepper scrambling!)
          const preVx = this.velX[i];
          const vx = Math.abs(preVx) > 0.02 ? preVx : (this.rng.next() - 0.5) * 0.08;
          const vy = 0.46 + (this.grainShade[i] / 255) * 0.22;
          this.velX[i] = vx;
          this.velY[i] = vy;
          this.fracX[i] = 0;
          this.fracY[i] = (this.grainShade[i] / 255) * 0.65;
          this.restTicks[i] = 0;
          this.texturesDirty = true;

          // Rich 3D tumbling mineral curtain synchronized to delaminating strata!
          if (this.active3DGrains < MAX_3D_GRAINS - 24 && this.rng.next() < 0.28) {
            const x = i % GRID_W;
            const y = (i - x) / GRID_W;
            this.spawnExcavatedGrain3D(x, y, vx, -vy, this.pigmentId[i], this.grainShade[i]);
          }
        }
      }
    }
    this.activeFaultCells = remaining;
  }

  /**
   * Moves `LOOSE_SAND` grains under continuous sub-cell kinematics (`fracX`, `fracY`),
   * dilatant free-fall, active cliff-face outward peeling (zero Pez-dispenser columns!),
   * multi-step Bagnold surface-layer cascades, down-slope collisional momentum transfer,
   * and static/dynamic angle-of-repose hysteresis (~37 deg start -> ~31.5 deg settle).
   * Strictly 1-to-1 cell swaps (`src -> EMPTY`, `EMPTY -> dst`): ZERO grains vanish or duplicate!
   */
  private stepLooseSandGravity(): number {
    this.moveTickId++;
    const stamp = this.moveTickId;
    const ct = this.cellType;
    const startCt = this.startCellType;
    startCt.set(ct);

    const vxArr = this.velX;
    const vyArr = this.velY;
    const fxArr = this.fracX;
    const fyArr = this.fracY;
    const colH = this.colHeight;

    // Fast O(W) column support horizon for laminar slope-gradient sorting:
    // Scans near `topSolidY[x]` first instead of sweeping full 220 rows from top every tick!
    for (let x = 0; x < GRID_W; x++) {
      let top = 0;
      const startScanY = Math.min(GRID_H - 1, (this.topSolidY[x] || GRID_H - 1) + 12);
      for (let y = startScanY; y >= 0; y--) {
        const c = startCt[y * GRID_W + x];
        if (c === CELL_PACKED_SAND || c === CELL_PEBBLE || (c === CELL_LOOSE_SAND && vyArr[y * GRID_W + x] < 0.55)) {
          top = y;
          break;
        }
      }
      colH[x] = top;
    }

    const leftToRight = (this.stepCount & 1) === 0;
    let fallingCount = 0;
    let movedCount = 0;

    for (let y = 1; y < GRID_H; y++) {
      const rowOff = y * GRID_W;
      for (let sx = 0; sx < GRID_W; sx++) {
        const x = leftToRight ? sx : GRID_W - 1 - sx;
        const idx = rowOff + x;
        if (ct[idx] !== CELL_LOOSE_SAND || this.movedStamp[idx] === stamp) continue;

        const belowIdx = idx - GRID_W;
        const belowNow = ct[belowIdx];
        const belowStart = startCt[belowIdx];

        // 0. Active Cliff-Face Outward Peel (Eliminates vertical Pez-dispenser column pistons!):
        //    If the grain underneath was present at tick-start and just moved away, AND this grain
        //    sits on an unconfined lateral cliff face, lateral granular pressure pushes it OUTWARD
        //    into the open void rather than dropping straight down its own vertical column!
        if (belowNow === CELL_EMPTY && belowStart !== CELL_EMPTY && x > 2 && x < GRID_W - 3) {
          const openL = startCt[idx - 1] === CELL_EMPTY && ct[belowIdx - 1] === CELL_EMPTY && ct[idx - 1] === CELL_EMPTY;
          const openR = startCt[idx + 1] === CELL_EMPTY && ct[belowIdx + 1] === CELL_EMPTY && ct[idx + 1] === CELL_EMPTY;
          const backedL = startCt[idx + 1] !== CELL_EMPTY;
          const backedR = startCt[idx - 1] !== CELL_EMPTY;
          const peelDir = openL && backedL ? -1 : openR && backedR ? 1 : 0;

          if (peelDir !== 0) {
            const nx = x + peelDir;
            const diagIdx = (y - 1) * GRID_W + nx;
            let destIdx = diagIdx;
            if (y >= 2 && ct[(y - 2) * GRID_W + nx] === CELL_EMPTY && vyArr[idx] > 0.45) {
              destIdx = (y - 2) * GRID_W + nx;
            }
            const nextVx = peelDir * Math.min(1.65, Math.abs(vxArr[idx]) * 0.7 + 0.54);
            const nextVy = Math.min(MAX_FALL_SPEED, Math.max(0.58, vyArr[idx] + GRAVITY_ACCEL));
            this.moveGrain(idx, destIdx, nextVx, nextVy, 0, fyArr[idx] * 0.5, stamp);
            if (this.active3DGrains < MAX_3D_GRAINS - 24 && this.rng.next() < 0.22) {
              this.spawnExcavatedGrain3D(nx, y - 1, nextVx, -nextVy, this.pigmentId[destIdx], this.grainShade[destIdx]);
            }
            fallingCount++;
            movedCount++;
            continue;
          }
        }

        // 1. Sub-Cell Continuous Ballistic Free-Fall & Aerodynamic Dilatancy (`belowNow === CELL_EMPTY`)
        if (belowNow === CELL_EMPTY) {
          const massMul = 0.91 + this.grainShade[idx] * (0.18 / 255);
          // Trailing grains inside a newly vacated column accelerate slightly slower than the leading
          // free-air edge (`0.76x`), naturally opening dilatant air gaps through falling curtains!
          const wakeMul = belowStart === CELL_EMPTY ? 1.0 : 0.76;
          let vy = vyArr[idx] + GRAVITY_ACCEL * massMul * wakeMul;
          const maxV = MAX_FALL_SPEED * massMul;
          if (vy > maxV) vy = maxV;
          let vx = vxArr[idx] * 0.985;

          const totalY = fyArr[idx] + vy;
          const stepsY = Math.floor(totalY);
          let remY = totalY - stepsY;

          const totalX = fxArr[idx] + vx;
          const stepsX = totalX >= 0 ? Math.floor(totalX) : Math.ceil(totalX);
          const absStepsX = Math.abs(stepsX);
          const dirX = stepsX > 0 ? 1 : -1;

          // Smooth sub-cell accumulation frame before crossing cell boundary
          if (stepsY === 0 && absStepsX === 0) {
            vyArr[idx] = vy;
            vxArr[idx] = vx;
            fyArr[idx] = remY;
            fxArr[idx] = totalX;
            this.restTicks[idx] = 0;
            this.movedStamp[idx] = stamp;
            if (vy > 0.12) fallingCount++;
            continue;
          }

          let destX = x;
          let destY = y;
          let usedX = 0;
          let hitGroundMidStep = false;
          const totalMicro = Math.max(stepsY, absStepsX);

          for (let s = 1; s <= totalMicro; s++) {
            const wantVert = s <= stepsY;
            const wantHoriz =
              absStepsX > 0 &&
              Math.floor((s * absStepsX) / totalMicro) > Math.floor(((s - 1) * absStepsX) / totalMicro);

            let ny = wantVert ? destY - 1 : destY;
            let nx = destX;

            if (wantHoriz) {
              const tryX = destX + dirX;
              if (
                tryX >= 1 &&
                tryX < GRID_W - 1 &&
                ct[destY * GRID_W + tryX] === CELL_EMPTY &&
                (ny >= 0 ? ct[ny * GRID_W + tryX] === CELL_EMPTY : false)
              ) {
                nx = tryX;
                usedX += dirX;
              }
            }

            if (ny < 0 || ct[ny * GRID_W + nx] !== CELL_EMPTY) {
              // If vertical step blocked, check if pure diagonal roll on impact is open right here!
              if (wantVert && ny >= 0) {
                hitGroundMidStep = true;
              }
              break;
            }
            destX = nx;
            destY = ny;
          }

          const remX = Math.max(-0.99, Math.min(0.99, totalX - usedX));
          if (hitGroundMidStep) {
            remY = 0;
            // Pre-convert part of mid-step landing speed into slope-directed lateral velocity!
            const hL = colH[Math.max(0, destX - 2)];
            const hR = colH[Math.min(GRID_W - 1, destX + 2)];
            const slopeSign =
              Math.abs(hL - hR) >= 2
                ? hL > hR
                  ? 1
                  : -1
                : Math.abs(vx) > 0.15
                  ? Math.sign(vx)
                  : ((destX + this.stepCount) & 1) === 0
                    ? 1
                    : -1;
            vx = vx * 0.65 + slopeSign * Math.min(1.5, vy * 0.42);
            vy = Math.max(0.42, vy * 0.68);
          }

          const destIdx = destY * GRID_W + destX;
          if (destIdx !== idx) {
            this.moveGrain(idx, destIdx, vx, vy, remX, remY, stamp);
            if (
              vy > 1.15 &&
              this.active3DGrains < MAX_3D_GRAINS - 24 &&
              this.rng.next() < 0.085
            ) {
              this.spawnExcavatedGrain3D(
                destX,
                destY,
                vx + (this.rng.next() - 0.5) * 0.22,
                -vy * 0.88,
                this.pigmentId[destIdx],
                this.grainShade[destIdx]
              );
            }
            movedCount++;
          } else {
            vxArr[idx] = vx;
            vyArr[idx] = vy;
            fxArr[idx] = remX;
            fyArr[idx] = remY;
            this.movedStamp[idx] = stamp;
          }
          fallingCount++;
          continue;
        }

        // 2. Impact Conversion, Slope-Aligned Laminar Sorting & Multi-Step Bagnold Surface Cascade
        let curVx = vxArr[idx];
        let curVy = vyArr[idx];

        const hL = colH[Math.max(0, x - 2)];
        const hR = colH[Math.min(GRID_W - 1, x + 2)];
        const slopeDiff = hL - hR; // > 0 means bed slopes down toward +X (right)

        if (curVy > 0.68) {
          // Slope-aligned laminar deflection (preserves crisp cross-bedded strata tongues!)
          const scatterSign =
            Math.abs(slopeDiff) >= 2
              ? slopeDiff > 0
                ? 1
                : -1
              : Math.abs(curVx) > 0.16
                ? Math.sign(curVx)
                : ((x + (y >> 1) + (this.stepCount >> 2)) & 1) === 0
                  ? 1
                  : -1;
          const boost = belowNow === CELL_PEBBLE ? 0.92 : 0.62;
          curVx = curVx * 0.55 + scatterSign * Math.min(2.1, curVy * boost);

          if (curVy > 2.0 && this.rng.next() < 0.022) {
            this.spawnDustMote(x, y + 1, this.pigmentId[idx], 0.19);
          }

          // Collisional momentum transfer into adjacent surface grains to nucleate avalanche waves!
          const pushMag = Math.min(0.85, curVy * 0.34);
          const fwdX = x + scatterSign;
          if (fwdX > 1 && fwdX < GRID_W - 2) {
            const adjIdx = rowOff + fwdX;
            if (ct[adjIdx] === CELL_LOOSE_SAND && vyArr[adjIdx] < 0.35) {
              vxArr[adjIdx] += scatterSign * pushMag;
              vyArr[adjIdx] = Math.max(vyArr[adjIdx], 0.38);
            }
            const adjDiag = (y - 1) * GRID_W + fwdX;
            if (ct[adjDiag] === CELL_LOOSE_SAND && vyArr[adjDiag] < 0.35) {
              vxArr[adjDiag] += scatterSign * pushMag * 0.85;
              vyArr[adjDiag] = Math.max(vyArr[adjDiag], 0.34);
            }
          }
        }

        // Preferred slide direction strictly honors slope gradient & existing tangential momentum
        const dirFirst =
          Math.abs(curVx) > 0.22
            ? curVx > 0
              ? 1
              : -1
            : Math.abs(slopeDiff) >= 1
              ? slopeDiff > 0
                ? 1
                : -1
              : leftToRight
                ? 1
                : -1;

        const initSpeed = Math.hypot(curVx, curVy);
        const isAlreadyFlowing = initSpeed > 0.24 || belowNow === CELL_PEBBLE;

        let curX = x;
        let curY = y;
        let slopeSpeed = Math.max(0.42, initSpeed);
        let slideDirChosen = 0;
        let glideStepsTaken = 0;

        const glideTotal = fyArr[idx] + Math.min(2.85, slopeSpeed);
        const maxGlideSteps = isAlreadyFlowing ? Math.max(1, Math.min(3, Math.floor(glideTotal))) : 1;

        for (let gStep = 0; gStep < maxGlideSteps; gStep++) {
          let steppedThisIter = false;
          const cRowOff = curY * GRID_W;
          const supportBelow = ct[(curY - 1) * GRID_W + curX];

          // If glide carried us off a lip into open mid-air, stop surface-gliding here and enter ballistic fall!
          if (gStep > 0 && supportBelow === CELL_EMPTY) {
            slopeSpeed = Math.min(MAX_FALL_SPEED, slopeSpeed + GRAVITY_ACCEL * 0.5);
            break;
          }

          for (let attempt = 0; attempt < 2; attempt++) {
            if (gStep > 0 && attempt > 0) break; // Keep multi-step cascade coherent along one flank!
            const dir = gStep > 0 ? slideDirChosen : attempt === 0 ? dirFirst : -dirFirst;
            const nx = curX + dir;
            if (nx < 1 || nx >= GRID_W - 1 || curY < 1) continue;

            const sideIdx = cRowOff + nx;
            const diagIdx = (curY - 1) * GRID_W + nx;

            // A. Steep >= 45-deg chute (`sin 45 - mu cos 45 > 0` -> Coulomb acceleration!)
            if (ct[sideIdx] === CELL_EMPTY && ct[diagIdx] === CELL_EMPTY) {
              curX = nx;
              curY -= 1;
              slideDirChosen = dir;
              slopeSpeed = Math.min(2.9, slopeSpeed + 0.22);
              steppedThisIter = true;
              glideStepsTaken++;
              break;
            }

            // B. Bagnold 2:1 / 3:2 repose slipface with true Static (~37 deg) vs Dynamic (~31.5 deg) Hysteresis!
            const nx2 = curX + dir * 2;
            if (nx2 >= 1 && nx2 < GRID_W - 1 && ct[sideIdx] === CELL_EMPTY) {
              const side2Idx = cRowOff + nx2;
              const diag2Idx = (curY - 1) * GRID_W + nx2;
              if (ct[side2Idx] === CELL_EMPTY && ct[diag2Idx] === CELL_EMPTY) {
                // Static resting grains only nucleate a fresh slide when slope > ~37 deg
                // (either deeper drop at nx2, or backed by higher grain upslope, or perched on slick pebble);
                // already-flowing avalanche grains glide smoothly down to ~31 deg before Coulomb friction stops them!
                const deepDrop2 = curY >= 2 && ct[(curY - 2) * GRID_W + nx2] === CELL_EMPTY;
                const upslopeX = curX - dir;
                const hasUpslopeHead =
                  upslopeX >= 1 &&
                  upslopeX < GRID_W - 1 &&
                  curY + 1 < GRID_H &&
                  ct[(curY + 1) * GRID_W + upslopeX] !== CELL_EMPTY;
                const staticFailure = supportBelow === CELL_PEBBLE || deepDrop2 || hasUpslopeHead;

                if (isAlreadyFlowing || staticFailure) {
                  curX = nx2;
                  curY -= 1;
                  slideDirChosen = dir;
                  // Gentle Coulomb basal deceleration on 26.6-deg step (`mu_d ~ 0.61`)
                  slopeSpeed =
                    supportBelow === CELL_PEBBLE
                      ? Math.min(2.4, slopeSpeed + 0.14)
                      : Math.max(0.26, slopeSpeed * 0.84 - 0.05);
                  steppedThisIter = true;
                  glideStepsTaken++;

                  // Retrogressive avalanche coupling: mobilize resting surface grain directly upslope!
                  if (gStep === 0 && hasUpslopeHead) {
                    const upIdx = (curY + 2) * GRID_W + upslopeX;
                    if (ct[upIdx] === CELL_LOOSE_SAND && vyArr[upIdx] < 0.28) {
                      vxArr[upIdx] = dir * 0.48;
                      vyArr[upIdx] = 0.38;
                    }
                  }
                  break;
                }
              }
            }

            // C. High-energy horizontal skitter across flat chamber floor or smooth river pebble crown
            if (slopeSpeed > 0.72 && (Math.abs(curVx) > 0.58 || supportBelow === CELL_PEBBLE) && ct[sideIdx] === CELL_EMPTY) {
              const sideBelow = (curY - 1) * GRID_W + nx;
              if (ct[sideBelow] !== CELL_EMPTY || supportBelow === CELL_PEBBLE) {
                curX = nx;
                slideDirChosen = dir;
                slopeSpeed *= supportBelow === CELL_PEBBLE ? 0.82 : 0.62;
                steppedThisIter = true;
                glideStepsTaken++;
                break;
              }
            }
          }

          if (!steppedThisIter) break;
        }

        if (glideStepsTaken > 0) {
          const destIdx = curY * GRID_W + curX;
          const finalVx = slideDirChosen * Math.min(2.2, Math.max(0.28, slopeSpeed * 0.78));
          const finalVy = Math.min(2.2, Math.max(0.26, slopeSpeed * 0.58));
          const nextRemY = Math.max(0, Math.min(0.95, glideTotal - glideStepsTaken));
          this.moveGrain(idx, destIdx, finalVx, finalVy, 0, nextRemY, stamp);

          // Spawn surface-saltating 3D mineral granules along brisk active slipfaces!
          if (
            slopeSpeed > 0.95 &&
            this.active3DGrains < MAX_3D_GRAINS - 24 &&
            this.rng.next() < 0.065
          ) {
            this.spawnExcavatedGrain3D(
              curX,
              curY + 0.35,
              finalVx * 0.92,
              -finalVy * 0.55,
              this.pigmentId[destIdx],
              this.grainShade[destIdx]
            );
          }

          fallingCount++;
          movedCount++;
        } else {
          // Blocked on slope/pile: transfer collisional momentum forward into down-slope loose neighbor
          // before damping, so traffic never kills an active avalanche wave!
          if (initSpeed > 0.32) {
            const pushX = x + dirFirst;
            if (pushX >= 1 && pushX < GRID_W - 1) {
              const blockSide = rowOff + pushX;
              const blockDiag = (y - 1) * GRID_W + pushX;
              const targetIdx =
                ct[blockDiag] === CELL_LOOSE_SAND
                  ? blockDiag
                  : ct[blockSide] === CELL_LOOSE_SAND
                    ? blockSide
                    : -1;
              if (targetIdx !== -1) {
                vxArr[targetIdx] = vxArr[targetIdx] * 0.5 + dirFirst * Math.min(1.4, initSpeed * 0.52);
                vyArr[targetIdx] = Math.max(vyArr[targetIdx], Math.min(1.1, initSpeed * 0.42));
              }
            }
          }

          vxArr[idx] = curVx * 0.38;
          if (Math.abs(vxArr[idx]) < 0.05) vxArr[idx] = 0;
          vyArr[idx] = curVy * 0.38;
          if (vyArr[idx] < 0.06) vyArr[idx] = 0;
          fxArr[idx] = 0;
          fyArr[idx] = 0;

          const rt = this.restTicks[idx] + 1;
          this.restTicks[idx] = rt > 65000 ? 65000 : rt;

          // Deep-overburden + laterally-confined consolidation ONLY (`>= 18` cells deep, `>= 7s` still):
          // Keeps surface dunes, crater cones, and talus slopes permanently loose & cascade-ready!
          if (
            rt >= RECOMPACT_TICKS &&
            y + RECOMPACT_MIN_DEPTH < colH[x] &&
            x >= 5 &&
            x < GRID_W - 5
          ) {
            const confined =
              ct[(y + 8) * GRID_W + x] !== CELL_EMPTY &&
              ct[(y + RECOMPACT_MIN_DEPTH) * GRID_W + x] !== CELL_EMPTY &&
              ct[rowOff + x - 4] !== CELL_EMPTY &&
              ct[rowOff + x + 4] !== CELL_EMPTY;
            if (confined) {
              ct[idx] = CELL_PACKED_SAND;
              this.bondHealth[idx] = 220;
              this.restTicks[idx] = 0;
            }
          }
        }
      }
    }

    this.activeFallingGrains = fallingCount;
    if (movedCount > 0) this.texturesDirty = true;
    return movedCount;
  }

  private moveGrain(
    srcIdx: number,
    dstIdx: number,
    newVelX: number,
    newVelY: number,
    newFracX: number,
    newFracY: number,
    stamp: number
  ): void {
    this.cellType[dstIdx] = CELL_LOOSE_SAND;
    this.pigmentId[dstIdx] = this.pigmentId[srcIdx];
    this.grainShade[dstIdx] = this.grainShade[srcIdx];
    this.bondHealth[dstIdx] = 0;
    this.faultDelay[dstIdx] = 0;
    this.velX[dstIdx] = newVelX;
    this.velY[dstIdx] = newVelY;
    this.fracX[dstIdx] = newFracX;
    this.fracY[dstIdx] = newFracY;
    this.restTicks[dstIdx] = 0;
    this.movedStamp[dstIdx] = stamp;

    this.cellType[srcIdx] = CELL_EMPTY;
    this.bondHealth[srcIdx] = 0;
    this.faultDelay[srcIdx] = 0;
    this.velX[srcIdx] = 0;
    this.velY[srcIdx] = 0;
    this.fracX[srcIdx] = 0;
    this.fracY[srcIdx] = 0;
    this.restTicks[srcIdx] = 0;
  }

  /**
   * Computes grounded bed support (`bedSupportMask`), column top surfaces (strictly ignoring
   * airborne pour streams!), `AIR_SKY` vs `AIR_CAVITY`, and 2-pass Chamfer distance fields.
   * Zero allocations.
   */
  private updateBedSupportMask(): void {
    const ct = this.cellType;
    const sup = this.bedSupportMask;

    for (let x = 0; x < GRID_W; x++) {
      sup[x] = ct[x] !== CELL_EMPTY ? 1 : 0;
    }

    for (let y = 1; y < GRID_H; y++) {
      const rowOff = y * GRID_W;
      const belOff = rowOff - GRID_W;
      for (let x = 0; x < GRID_W; x++) {
        const i = rowOff + x;
        const c = ct[i];
        if (c === CELL_EMPTY) {
          sup[i] = 0;
        } else if (c === CELL_PACKED_SAND || c === CELL_PEBBLE) {
          sup[i] = 1;
        } else {
          // CELL_LOOSE_SAND is part of the grounded sand bed (including sliding surface avalanches!)
          // iff supported directly or diagonally by grounded bed below; mid-air streams stay `0`!
          const sBel =
            sup[belOff + x] |
            (x > 0 ? sup[belOff + x - 1] : 0) |
            (x + 1 < GRID_W ? sup[belOff + x + 1] : 0);
          sup[i] = sBel;
        }
      }
    }
  }

  recomputeSurfaceAndOpticalFields(): void {
    this.updateBedSupportMask();
    const ct = this.cellType;
    const sup = this.bedSupportMask;
    const air = this.airClass;

    for (let x = 0; x < GRID_W; x++) {
      let topRestY = 0;
      let topOverY = 0;
      let topPigment = 0;
      let topShade = 128;

      for (let y = GRID_H - 1; y >= 0; y--) {
        const idx = y * GRID_W + x;
        if (sup[idx] === 0) continue; // Strictly ignore airborne poured/falling grains!

        if (topOverY === 0) {
          topOverY = y;
        }
        if (topRestY === 0) {
          topRestY = y;
          topPigment = this.pigmentId[idx];
          topShade = this.grainShade[idx];
          break;
        }
      }
      if (topOverY < topRestY) topOverY = topRestY;
      this.topSolidY[x] = topRestY;
      this.overburdenTopY[x] = topOverY;

      const pOff = topPigment * 3;
      const contrast = this.pigmentContrast[topPigment];
      const shadeMul = 1 + ((topShade - 128) / 128) * contrast;
      this.surfaceRgb[x * 3] = Math.min(1, this.pigmentRgbLinear[pOff] * shadeMul);
      this.surfaceRgb[x * 3 + 1] = Math.min(1, this.pigmentRgbLinear[pOff + 1] * shadeMul);
      this.surfaceRgb[x * 3 + 2] = Math.min(1, this.pigmentRgbLinear[pOff + 2] * shadeMul);
    }

    const hWin = 18;
    for (let x = 0; x < GRID_W; x++) {
      let sum = 0;
      let cnt = 0;
      const x0 = Math.max(0, x - hWin);
      const x1 = Math.min(GRID_W - 1, x + hWin);
      for (let k = x0; k <= x1; k++) {
        sum += this.overburdenTopY[k];
        cnt++;
      }
      this.smoothHorizon[x] = cnt > 0 ? sum / cnt : this.overburdenTopY[x];
    }

    this.skyFloorY.fill(0);
    for (let y = GRID_H - 1; y >= 0; y--) {
      const rowOff = y * GRID_W;
      let x = 0;
      while (x < GRID_W) {
        if (sup[rowOff + x] !== 0) {
          this.rowOpenSpan[x] = 0;
          x++;
          continue;
        }
        const runStart = x;
        while (x < GRID_W && sup[rowOff + x] === 0) {
          x++;
        }
        const runLen = x - runStart;
        for (let k = runStart; k < x; k++) {
          this.rowOpenSpan[k] = runLen;
        }
      }

      for (let cx = 0; cx < GRID_W; cx++) {
        const idx = rowOff + cx;
        if (ct[idx] !== CELL_EMPTY) {
          air[idx] = AIR_SOLID;
          // Only grounded bed cells raise `skyFloorY` — airborne pour streams NEVER tent the 3D crest!
          if (sup[idx] !== 0 && y > this.skyFloorY[cx]) this.skyFloorY[cx] = y;
          continue;
        }
        if (y > this.overburdenTopY[cx]) {
          const horiz = this.smoothHorizon[cx];
          if (y >= horiz - 2 || (y >= horiz - 14 && this.rowOpenSpan[cx] >= SKY_MIN_OPEN_SPAN)) {
            air[idx] = AIR_SKY;
          } else {
            air[idx] = AIR_CAVITY;
            if (y > this.skyFloorY[cx]) this.skyFloorY[cx] = y;
          }
        } else {
          air[idx] = AIR_CAVITY;
          if (y > this.skyFloorY[cx]) this.skyFloorY[cx] = y;
        }
      }
    }

    // Tighter, slope-preserving 5-tap crest filter (`x - 2 .. x + 2`) so surface rims stay crisp
    for (let x = 0; x < GRID_W; x++) {
      const centerH = this.skyFloorY[x];
      let sum = 0;
      let wSum = 0;
      const x0 = Math.max(0, x - 2);
      const x1 = Math.min(GRID_W - 1, x + 2);
      for (let k = x0; k <= x1; k++) {
        const nh = this.skyFloorY[k];
        // Bilateral clamp prevents stretching a wide ramp across a freshly carved trench mouth
        const clampedNh = Math.abs(nh - centerH) > 6 ? centerH : nh;
        const w = 3 - Math.abs(k - x);
        sum += clampedNh * w;
        wSum += w;
      }
      this.surfaceY[x] = wSum > 0 ? sum / wSum : this.topSolidY[x];
    }

    // Fast 2-pass Chamfer Distance Transforms for cavity recess & solid thin-wall translucency
    const dCav = this.distBufA;
    const dSol = this.distBufB;
    const INF = 28;

    for (let i = 0; i < GRID_CELLS; i++) {
      dCav[i] = air[i] === AIR_CAVITY ? INF : 0;
      dSol[i] = ct[i] !== CELL_EMPTY ? INF : 0;
    }

    for (let y = 1; y < GRID_H - 1; y++) {
      const rowOff = y * GRID_W;
      const prevOff = (y - 1) * GRID_W;
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = rowOff + x;
        let mc = dCav[i];
        if (mc > 0) {
          let t = dCav[i - 1] + 2;
          if (t < mc) mc = t;
          t = dCav[prevOff + x] + 2;
          if (t < mc) mc = t;
          t = dCav[prevOff + x - 1] + 3;
          if (t < mc) mc = t;
          dCav[i] = mc;
        }
        let ms = dSol[i];
        if (ms > 0) {
          let t = dSol[i - 1] + 2;
          if (t < ms) ms = t;
          t = dSol[prevOff + x] + 2;
          if (t < ms) ms = t;
          t = dSol[prevOff + x - 1] + 3;
          if (t < ms) ms = t;
          dSol[i] = ms;
        }
      }
    }

    for (let y = GRID_H - 2; y >= 1; y--) {
      const rowOff = y * GRID_W;
      const nextOff = (y + 1) * GRID_W;
      for (let x = GRID_W - 2; x >= 1; x--) {
        const i = rowOff + x;
        let mc = dCav[i];
        if (mc > 0) {
          let t = dCav[i + 1] + 2;
          if (t < mc) mc = t;
          t = dCav[nextOff + x] + 2;
          if (t < mc) mc = t;
          t = dCav[nextOff + x + 1] + 3;
          if (t < mc) mc = t;
          dCav[i] = mc;
        }
        let ms = dSol[i];
        if (ms > 0) {
          let t = dSol[i + 1] + 2;
          if (t < ms) ms = t;
          t = dSol[nextOff + x] + 2;
          if (t < ms) ms = t;
          t = dSol[nextOff + x + 1] + 3;
          if (t < ms) ms = t;
          dSol[i] = ms;
        }
      }
    }

    for (let i = 0; i < GRID_CELLS; i++) {
      if (air[i] === AIR_CAVITY) {
        const rc = 52 + dCav[i] * 11;
        this.recessField[i] = rc > 255 ? 255 : rc;
      } else {
        this.recessField[i] = 0;
      }
      const ws = dSol[i] * 16;
      this.wallDistField[i] = ws > 255 ? 255 : ws;
    }
  }

  /**
   * Packs `colorTexData` (RGBA8 SRGB, LinearFilter) and `fieldTexData` (RGBA8 Linear, LinearFilter).
   * Preserves exact channel contracts:
   * - `colorTex.rgb`: smooth grain/stratum pigment (undisturbed bgStrata in cavities -> zero fringe bleed!)
   * - `colorTex.a`: 255 strictly on airborne falling LOOSE_SAND (`velY > 0.12`), 0 elsewhere
   * - `fieldTex.r`: 3x3 Gaussian-smoothed cohesive solid coverage (0 on airborne falling grains)
   * - `fieldTex.g`: smooth cavity recess depth (`>= 52` in CAVITY, `0` in SKY/SOLID)
   * - `fieldTex.b`: solid thin-wall distance for warm backlit translucency
   * - `fieldTex.a`: compacted PACKED_SAND arch shear stress ONLY (`0` on LOOSE_SAND!)
   */
  packTextures(): void {
    this.updateBedSupportMask();
    const ct = this.cellType;
    const sup = this.bedSupportMask;
    const pig = this.pigmentId;
    const shd = this.grainShade;
    const bgP = this.bgPigmentId;
    const bgS = this.bgShade;
    const rec = this.recessField;
    const wal = this.wallDistField;
    const str = this.stressField;
    const vy = this.velY;
    const coh = this.rawCohesiveMask;
    const cOut = this.colorTexData;
    const fOut = this.fieldTexData;
    const lut = this.srgbByteLut;

    for (let i = 0; i < GRID_CELLS; i++) {
      const o = i << 2;
      const c = ct[i];
      const isActiveMotion = c === CELL_LOOSE_SAND && vy[i] > 0.12;
      // Grounded bed sand (including Bagnold surface grains gliding in contact with a dune slope!)
      // keeps cohesive silhouette continuity (`coh = 1 -> zero slope strobing`), while genuinely
      // airborne mid-air grains (`sup[i] === 0`) keep `coh = 0` (zero wall parallax extrusion)!
      const isCohesive =
        c === CELL_PACKED_SAND || c === CELL_PEBBLE || (c === CELL_LOOSE_SAND && sup[i] === 1);
      coh[i] = isCohesive ? 1 : 0;

      const pId = c === CELL_EMPTY ? bgP[i] : pig[i];
      const shadeByte = c === CELL_EMPTY ? bgS[i] : shd[i];
      const lutOff = ((pId << 8) | shadeByte) * 3;
      cOut[o] = lut[lutOff];
      cOut[o + 1] = lut[lutOff + 1];
      cOut[o + 2] = lut[lutOff + 2];
      cOut[o + 3] = isActiveMotion ? 255 : 0;

      fOut[o + 1] = rec[i];
      fOut[o + 2] = wal[i];
      fOut[o + 3] = c === CELL_PACKED_SAND ? str[i] : 0;
    }

    // 3x3 Gaussian-weighted cohesive solid coverage -> fieldTex.r (butter-smooth curved burrows!)
    for (let x = 0; x < GRID_W; x++) {
      fOut[x << 2] = coh[x] ? 255 : 0;
      const topIdx = (GRID_H - 1) * GRID_W + x;
      fOut[topIdx << 2] = coh[topIdx] ? 255 : 0;
    }
    for (let y = 1; y < GRID_H - 1; y++) {
      const rowOff = y * GRID_W;
      const upOff = rowOff + GRID_W;
      const dnOff = rowOff - GRID_W;
      fOut[rowOff << 2] = coh[rowOff] ? 255 : 0;
      fOut[(rowOff + GRID_W - 1) << 2] = coh[rowOff + GRID_W - 1] ? 255 : 0;

      for (let x = 1; x < GRID_W - 1; x++) {
        const i = rowOff + x;
        const c = coh[i];
        const sum16 =
          (c << 2) +
          ((coh[i - 1] + coh[i + 1] + coh[upOff + x] + coh[dnOff + x]) << 1) +
          coh[upOff + x - 1] +
          coh[upOff + x + 1] +
          coh[dnOff + x - 1] +
          coh[dnOff + x + 1];
        fOut[i << 2] = sum16 === 16 ? 255 : (sum16 * 16) | 0;
      }
    }

    this.texturesDirty = false;
  }

  /** Counts total solid sand cells currently in the grid (PACKED_SAND + LOOSE_SAND). */
  countGridSandGrains(): number {
    let n = 0;
    const ct = this.cellType;
    for (let i = 0; i < GRID_CELLS; i++) {
      if (ct[i] === CELL_PACKED_SAND || ct[i] === CELL_LOOSE_SAND) n++;
    }
    return n;
  }

  /** Counts total grains currently carried inside autonomous worker ant mandibles. */
  countCarriedGrains(): number {
    let n = 0;
    const ants = this.colony.ants;
    for (let i = 0; i < ants.length; i++) {
      if (ants[i].carryingPigment >= 0) n++;
    }
    return n;
  }

  /** Total conserved sand mass (grid sand + ant-carried sand). */
  countTotalSandMass(): number {
    return this.countGridSandGrains() + this.countCarriedGrains();
  }

  /** Counts immovable river pebble cells. */
  countPebbleCells(): number {
    let n = 0;
    const ct = this.cellType;
    for (let i = 0; i < GRID_CELLS; i++) {
      if (ct[i] === CELL_PEBBLE) n++;
    }
    return n;
  }

  /** Returns read-only state summary matching `window.__DAY27__.getState()`. */
  getMetrics(): SandSimMetrics {
    return {
      excavatedCells: this.excavatedCells,
      activeFallingGrains: this.activeFallingGrains,
      collapseEvents: this.collapseEvents,
      diggerCount: this.colony.ants.length,
      tailingsDeposited: this.tailingsDeposited,
      stepCount: this.stepCount,
      active3DGrains: this.active3DGrains,
      activeDustMotes: this.activeDustMotes,
      colony: this.colony.getStats(),
    };
  }
}
