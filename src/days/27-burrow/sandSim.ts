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

const GRAVITY_ACCEL = 0.29;
const MAX_FALL_SPEED = 3.35;
const RECOMPACT_TICKS = 115;
const SKY_MIN_OPEN_SPAN = 22;

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

  // Pre-allocated scratch buffers for zero-GC arch/distance passes
  private readonly rowOpenSpan = new Int16Array(GRID_W);
  private readonly smoothHorizon = new Float32Array(GRID_W);
  private readonly skyFloorY = new Int16Array(GRID_W);
  private readonly collapseCooldown = new Uint16Array(GRID_W);
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
    this.restTicks.fill(0);
    this.movedStamp.fill(0);
    this.stressField.fill(0);
    this.collapseCooldown.fill(0);
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

    // 5. Deposit a sculpted layered volcanic-cone crater mound of excavated Sienna & Ochre tailings!
    for (let k = 0; k < 210; k++) {
      const side = k % 2 === 0 ? -1 : 1;
      // Triangular distribution peaks right beside the shaft rim (dist 6..18)
      const dist = 6 + Math.floor((this.rng.next() + this.rng.next()) * 8.5);
      const tx = Math.max(5, Math.min(GRID_W - 6, mouthX + side * dist));
      let ty = GRID_H - 3;
      while (ty > 4 && this.cellType[(ty - 1) * GRID_W + tx] === CELL_EMPTY) {
        ty--;
      }
      if (ty < GRID_H - 4 && this.cellType[ty * GRID_W + tx] === CELL_EMPTY) {
        const idx = ty * GRID_W + tx;
        this.cellType[idx] = CELL_LOOSE_SAND;
        // Layered alternating Sienna, Ochre, and occasional Hematite excavated from below!
        const r = this.rng.next();
        this.pigmentId[idx] = r < 0.52 ? 3 : r < 0.86 ? 2 : 5;
        this.grainShade[idx] = Math.max(
          36,
          Math.min(232, Math.round(122 + (this.rng.next() - 0.5) * 76))
        );
        this.bondHealth[idx] = 0;
        this.velX[idx] = 0;
        this.velY[idx] = 0;
      }
    }

    // Settle starter crater mound gently to natural repose angle
    for (let s = 0; s < 26; s++) {
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
          this.restTicks[idx] = 0;
          removed++;

          // Emit a handful of tactile 3D sparkling dislodged sand chips when user actively carves
          if (scatterLooseRim && this.rng.next() < 0.045) {
            this.spawnExcavatedGrain3D(
              x,
              y,
              (this.rng.next() - 0.5) * 0.9,
              -0.1 - this.rng.next() * 0.5,
              pig,
              shd
            );
          }
        } else if (
          scatterLooseRim &&
          dist2 <= rimR2 &&
          ct === CELL_PACKED_SAND &&
          ddy > radius * 0.35
        ) {
          // Organic rim raveling: loosen a subtle veil of ceiling & shoulder grains for delightful live trickle!
          if (this.rng.next() < 0.13) {
            this.cellType[idx] = CELL_LOOSE_SAND;
            this.bondHealth[idx] = 0;
            this.velX[idx] = (this.rng.next() - 0.5) * 0.35;
            this.velY[idx] = 0.25 + this.rng.next() * 0.2;
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
      // Focused triangular/hourglass nozzle dispersion (not a scattered rectangular rain box!)
      const ox = Math.round(((this.rng.next() + this.rng.next()) - 1.0) * 3.1);
      const oy = Math.round((this.rng.next() - 0.5) * 2.4);
      const x = Math.max(3, Math.min(GRID_W - 4, cx + ox));
      const y = Math.max(4, Math.min(GRID_H - 2, cy + oy));
      const idx = y * GRID_W + x;
      if (this.cellType[idx] === CELL_EMPTY) {
        const pig =
          this.rng.next() < 0.14
            ? pourPalette[(Math.floor(this.pourPigmentPhase) + 1) % pourPalette.length]
            : basePigment;
        const shd = Math.max(
          28,
          Math.min(238, Math.round(128 + (this.rng.next() - 0.5) * 96))
        );
        this.cellType[idx] = CELL_LOOSE_SAND;
        this.pigmentId[idx] = pig;
        this.grainShade[idx] = shd;
        this.bondHealth[idx] = 0;
        this.faultDelay[idx] = 0;
        const vx = (this.rng.next() - 0.5) * 0.42;
        const vy = 0.52 + this.rng.next() * 0.48;
        this.velX[idx] = vx;
        this.velY[idx] = vy;
        this.restTicks[idx] = 0;
        spawned++;

        // Pair poured stream with glistening 3D tumbling granules inside the glass gap
        if (this.rng.next() < 0.45) {
          this.spawnExcavatedGrain3D(x, y, vx * 0.7, -vy * 0.75, pig, shd);
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
    g.x = gx + (this.rng.next() - 0.5) * 0.9;
    g.y = gy + (this.rng.next() - 0.5) * 0.9;
    g.z = (this.rng.next() - 0.5) * 1.48; // Distributed nicely across front-to-back glass thickness!
    g.vx = vx;
    g.vy = vy;
    g.vz = (this.rng.next() - 0.5) * 0.035;
    g.rotX = this.rng.next() * Math.PI * 2;
    g.rotY = this.rng.next() * Math.PI * 2;
    g.rotZ = this.rng.next() * Math.PI * 2;
    g.vRot = (this.rng.next() < 0.5 ? -1 : 1) * (0.14 + this.rng.next() * 0.28);
    g.scale = 0.72 + this.rng.next() * 0.65;
    g.pigmentId = pigmentId;
    g.shade = shade;
    g.age = 0;
    g.life = 28 + Math.floor(this.rng.next() * 34);
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
    m.life = 32 + Math.floor(this.rng.next() * 24);
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

      g.vy = Math.max(-3.2, g.vy - 0.16);
      g.vx *= 0.97;
      const nx = g.x + g.vx;
      const ny = g.y + g.vy;
      g.z += g.vz;
      if (g.z > 0.82 || g.z < -0.82) {
        g.z = Math.max(-0.82, Math.min(0.82, g.z));
        g.vz *= -0.45;
      }

      const ix = Math.round(nx);
      const iy = Math.round(ny);
      if (ix < 2 || ix >= GRID_W - 2 || iy < 2 || iy >= GRID_H - 2) {
        g.active = false;
        continue;
      }

      const cell = ct[iy * GRID_W + ix];
      // Collide only against static solid surfaces (packed sand, pebbles, or resting loose piles),
      // NEVER killing airborne granules against adjacent falling grains traveling in the same stream!
      const hitSolid =
        cell === CELL_PACKED_SAND ||
        cell === CELL_PEBBLE ||
        (cell === CELL_LOOSE_SAND && this.velY[iy * GRID_W + ix] < 0.25);
      if (hitSolid) {
        // Real lively pebble or slope bounce!
        if (cell === CELL_PEBBLE && Math.abs(g.vy) > 0.65 && g.age < g.life - 8) {
          g.vy = Math.abs(g.vy) * 0.36;
          g.vx += (g.vx >= 0 ? 1 : -1) * (0.35 + this.rng.next() * 0.45);
          g.vRot *= -1.1;
        } else if (Math.abs(g.vy) > 1.1 && g.age < g.life - 10) {
          // Tiny energetic sand-splash rebound on fast impact
          g.vy = Math.abs(g.vy) * 0.22;
          g.vx += (this.rng.next() - 0.5) * 0.8;
          g.age += 6;
        } else {
          g.active = false;
          continue;
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

      // Gently curved parabolic gothic-arch convergence (looks vastly more geological than straight lines!)
      const hNorm = dy / Math.max(1, maxWedgeH);
      const archCurve = dy * baseSlope * (0.85 + 0.32 * hNorm);
      const jitterL = Math.sin(y * 0.43 + spanL * 0.13) * 1.45;
      const jitterR = Math.cos(y * 0.49 + spanR * 0.17) * 1.45;
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
        // Center keystone breaks FIRST, then shoulders unzip outward + upward in staggered strata waves!
        const centerDistNorm = Math.abs(x - midX) / Math.max(1, halfSpan);
        const shoulderLag = Math.floor(centerDistNorm * 3.2);
        const delay = Math.max(
          1,
          Math.min(
            48,
            1 + Math.floor(dy * 0.44) + shoulderLag + (this.rng.next() < 0.35 ? 1 : 0)
          )
        );
        if (fd[idx] === 0) {
          fd[idx] = delay;
          this.activeFaultCells++;
        } else if (delay < fd[idx]) {
          fd[idx] = delay;
        }
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
          const vx = (this.rng.next() - 0.5) * 0.45;
          const vy = 0.44 + this.rng.next() * 0.36;
          this.velX[i] = vx;
          this.velY[i] = vy;
          this.restTicks[i] = 0;
          this.texturesDirty = true;

          // Sample a vibrant scatter of released fault grains into the 3D tumbling granule layer!
          if (this.rng.next() < 0.085) {
            const x = i % GRID_W;
            const y = (i - x) / GRID_W;
            this.spawnExcavatedGrain3D(x, y, vx * 0.9, -vy * 0.75, this.pigmentId[i], this.grainShade[i]);
          }
        }
      }
    }
    this.activeFaultCells = remaining;
  }

  /**
   * Moves `LOOSE_SAND` grains downward under gravity with true 2D momentum (`velX`, `velY`),
   * pebble-crown ricochet fanning, impact-coupled slope avalanches, and ~33-36 deg repose dunes.
   * Strictly 1-to-1 cell swaps (`src -> EMPTY`, `EMPTY -> dst`): ZERO grains vanish or duplicate!
   */
  private stepLooseSandGravity(): number {
    this.moveTickId++;
    const stamp = this.moveTickId;
    const ct = this.cellType;
    const vxArr = this.velX;
    const vyArr = this.velY;
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
        const belowType = ct[belowIdx];

        // 1. Direct free-fall with gravity acceleration & subtle lateral drift (`velX`)!
        if (belowType === CELL_EMPTY) {
          let vy = vyArr[idx] + GRAVITY_ACCEL;
          if (vy > MAX_FALL_SPEED) vy = MAX_FALL_SPEED;
          const vx = vxArr[idx] * 0.94;
          const maxSteps = Math.max(1, Math.floor(vy));

          let destX = x;
          let destY = y;
          for (let s = 1; s <= maxSteps; s++) {
            const ny = y - s;
            if (ny < 0) break;
            // Subtle lateral momentum drift during air fall (strictly checking open cells!)
            let candX = destX;
            if (Math.abs(vx) > 0.35 && s === 1) {
              const tryX = destX + (vx > 0 ? 1 : -1);
              if (
                tryX >= 1 &&
                tryX < GRID_W - 1 &&
                ct[destY * GRID_W + tryX] === CELL_EMPTY &&
                ct[ny * GRID_W + tryX] === CELL_EMPTY
              ) {
                candX = tryX;
              }
            }
            if (ct[ny * GRID_W + candX] !== CELL_EMPTY) break;
            destX = candX;
            destY = ny;
          }

          const destIdx = destY * GRID_W + destX;
          this.moveGrain(idx, destIdx, vx, vy, stamp);
          // Sample brisk avalanche grains into the sparkling 3D tumbling granule layer!
          if (vy > 1.35 && this.active3DGrains < MAX_3D_GRAINS - 20 && this.rng.next() < 0.032) {
            this.spawnExcavatedGrain3D(
              destX,
              destY,
              vx * 0.8 + (this.rng.next() - 0.5) * 0.35,
              -vy * 0.55,
              this.pigmentId[destIdx],
              this.grainShade[destIdx]
            );
          }
          fallingCount++;
          movedCount++;
          continue;
        }

        // 2. Impact energy conversion & slope avalanching!
        //    Fast-falling grains impacting solid ground or river pebbles convert downward kinetic
        //    energy into lively lateral tumbling velocity (`velX`) and wake adjacent steep loose slopes!
        let curVx = vxArr[idx];
        const curVy = vyArr[idx];

        if (curVy > 0.85) {
          // Convert portion of downward impact momentum into lateral roll velocity
          const scatterSign =
            Math.abs(curVx) > 0.1 ? Math.sign(curVx) : this.rng.next() < 0.5 ? -1 : 1;
          const boost = belowType === CELL_PEBBLE ? 0.85 : 0.52;
          curVx = curVx * 0.5 + scatterSign * Math.min(1.6, curVy * boost);

          // Spawn a crisp splash of dust on heavy multi-cell avalanche impact
          if (curVy > 2.1 && this.rng.next() < 0.014) {
            this.spawnDustMote(x, y + 1, this.pigmentId[idx], 0.22);
          }

          // Kinetic impact wake-up: jostle adjacent resting loose grains on steep shoulders
          if (x > 2 && ct[idx - 1] === CELL_LOOSE_SAND && vyArr[idx - 1] < 0.2) {
            vxArr[idx - 1] -= 0.45;
            vyArr[idx - 1] = Math.max(vyArr[idx - 1], 0.32);
          }
          if (x < GRID_W - 3 && ct[idx + 1] === CELL_LOOSE_SAND && vyArr[idx + 1] < 0.2) {
            vxArr[idx + 1] += 0.45;
            vyArr[idx + 1] = Math.max(vyArr[idx + 1], 0.32);
          }
        }

        // Preferred slide direction honors existing lateral momentum `velX`!
        const dirFirst =
          Math.abs(curVx) > 0.14 ? (curVx > 0 ? 1 : -1) : this.rng.next() < 0.5 ? -1 : 1;
        let slid = false;

        for (let attempt = 0; attempt < 2; attempt++) {
          const dir = attempt === 0 ? dirFirst : -dirFirst;
          const nx = x + dir;
          if (nx < 1 || nx >= GRID_W - 1) continue;

          const sideIdx = rowOff + nx;
          const diagIdx = (y - 1) * GRID_W + nx;

          // A. Primary 45-deg diagonal roll (never clips diagonally through solid corners!)
          if (ct[sideIdx] === CELL_EMPTY && ct[diagIdx] === CELL_EMPTY) {
            const nextVy = Math.max(0.38, curVy * 0.62);
            const nextVx = dir * Math.min(1.8, Math.max(0.35, Math.abs(curVx) * 0.78 + 0.24));
            this.moveGrain(idx, diagIdx, nextVx, nextVy, stamp);
            fallingCount++;
            movedCount++;
            slid = true;
            break;
          }

          // B. Kinetic 2:1 repose avalanche (~32-35 deg natural sand-dune slope!)
          const nx2 = x + dir * 2;
          if (nx2 >= 1 && nx2 < GRID_W - 1) {
            const kineticEnergy = curVy + Math.abs(curVx);
            const rollChance =
              belowType === CELL_PEBBLE ? 0.78 : kineticEnergy > 0.35 ? 0.64 : 0.22;
            if (this.rng.next() < rollChance) {
              const side2Idx = rowOff + nx2;
              const diag2Idx = (y - 1) * GRID_W + nx2;
              if (
                ct[sideIdx] === CELL_EMPTY &&
                ct[side2Idx] === CELL_EMPTY &&
                ct[diag2Idx] === CELL_EMPTY
              ) {
                const nextVy = Math.max(0.28, curVy * 0.48);
                const nextVx = dir * Math.min(1.6, Math.max(0.28, Math.abs(curVx) * 0.72 + 0.16));
                this.moveGrain(idx, diag2Idx, nextVx, nextVy, stamp);
                fallingCount++;
                movedCount++;
                slid = true;
                break;
              }
            }
          }

          // C. High-energy horizontal skitter along flat floor or pebble crown (prevents tall spikes!)
          if (Math.abs(curVx) > 0.55 && ct[sideIdx] === CELL_EMPTY) {
            const sideBelow = (y - 1) * GRID_W + nx;
            // Only skitter laterally if supported or dropping off a ledge (keeps natural dune flow!)
            if (ct[sideBelow] !== CELL_EMPTY || belowType === CELL_PEBBLE) {
              this.moveGrain(idx, sideIdx, curVx * 0.62, curVy * 0.35, stamp);
              fallingCount++;
              movedCount++;
              slid = true;
              break;
            }
          }
        }

        if (!slid) {
          // Grain settled on support
          vxArr[idx] = curVx * 0.25;
          if (Math.abs(vxArr[idx]) < 0.04) vxArr[idx] = 0;
          vyArr[idx] = curVy * 0.25;
          if (vyArr[idx] < 0.05) vyArr[idx] = 0;

          const rt = this.restTicks[idx] + 1;
          this.restTicks[idx] = rt > 65000 ? 65000 : rt;

          // Gently re-compact deeply buried settled loose grains back into PACKED_SAND
          // (only when buried >= 7 cells below open air and firmly supported underneath!)
          if (rt >= RECOMPACT_TICKS && y + 7 < GRID_H) {
            const aboveSolid =
              ct[(y + 1) * GRID_W + x] !== CELL_EMPTY &&
              ct[(y + 4) * GRID_W + x] !== CELL_EMPTY &&
              ct[(y + 7) * GRID_W + x] !== CELL_EMPTY;
            if (aboveSolid) {
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
    stamp: number
  ): void {
    this.cellType[dstIdx] = CELL_LOOSE_SAND;
    this.pigmentId[dstIdx] = this.pigmentId[srcIdx];
    this.grainShade[dstIdx] = this.grainShade[srcIdx];
    this.bondHealth[dstIdx] = 0;
    this.faultDelay[dstIdx] = 0;
    this.velX[dstIdx] = newVelX;
    this.velY[dstIdx] = newVelY;
    this.restTicks[dstIdx] = 0;
    this.movedStamp[dstIdx] = stamp;

    this.cellType[srcIdx] = CELL_EMPTY;
    this.bondHealth[srcIdx] = 0;
    this.faultDelay[srcIdx] = 0;
    this.velX[srcIdx] = 0;
    this.velY[srcIdx] = 0;
    this.restTicks[srcIdx] = 0;
  }

  /**
   * Computes column top surfaces, classifies `AIR_SKY` vs `AIR_CAVITY`, and computes
   * smooth signed-style 2-pass Manhattan distance fields for recessed 3D tunnel depth
   * and backlit thin-sand edge translucency. Zero allocations.
   */
  recomputeSurfaceAndOpticalFields(): void {
    const ct = this.cellType;
    const air = this.airClass;
    const vy = this.velY;

    for (let x = 0; x < GRID_W; x++) {
      let topRestY = 0;
      let topOverY = 0;
      let topPigment = 0;
      let topShade = 128;
      for (let y = GRID_H - 1; y >= 0; y--) {
        const idx = y * GRID_W + x;
        const c = ct[idx];
        if (c === CELL_EMPTY) continue;

        if (topOverY === 0) {
          const isDenseSlump =
            c !== CELL_LOOSE_SAND ||
            vy[idx] <= 0.35 ||
            (y >= 2 &&
              (ct[(y - 1) * GRID_W + x] !== CELL_EMPTY ||
                ct[(y - 2) * GRID_W + x] !== CELL_EMPTY));
          if (isDenseSlump) {
            topOverY = y;
          }
        }
        if (topRestY === 0 && (c === CELL_PACKED_SAND || c === CELL_PEBBLE || vy[idx] <= 0.25)) {
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
        if (ct[rowOff + x] !== CELL_EMPTY) {
          this.rowOpenSpan[x] = 0;
          x++;
          continue;
        }
        const runStart = x;
        while (x < GRID_W && ct[rowOff + x] === CELL_EMPTY) {
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
          if (y > this.skyFloorY[cx]) this.skyFloorY[cx] = y;
          continue;
        }
        if (y > this.overburdenTopY[cx]) {
          const horiz = this.smoothHorizon[cx];
          if (y >= horiz - 2 || (y >= horiz - 12 && this.rowOpenSpan[cx] >= SKY_MIN_OPEN_SPAN)) {
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

    for (let x = 0; x < GRID_W; x++) {
      let sum = 0;
      let wSum = 0;
      const x0 = Math.max(0, x - 4);
      const x1 = Math.min(GRID_W - 1, x + 4);
      for (let k = x0; k <= x1; k++) {
        const w = 5 - Math.abs(k - x);
        sum += this.skyFloorY[k] * w;
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
    const ct = this.cellType;
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
      const isFalling = c === CELL_LOOSE_SAND && vy[i] > 0.12;
      const isCohesive =
        c === CELL_PACKED_SAND || c === CELL_PEBBLE || (c === CELL_LOOSE_SAND && !isFalling);
      coh[i] = isCohesive ? 1 : 0;

      const pId = c === CELL_EMPTY ? bgP[i] : pig[i];
      const shadeByte = c === CELL_EMPTY ? bgS[i] : shd[i];
      const lutOff = ((pId << 8) | shadeByte) * 3;
      cOut[o] = lut[lutOff];
      cOut[o + 1] = lut[lutOff + 1];
      cOut[o + 2] = lut[lutOff + 2];
      cOut[o + 3] = isFalling ? 255 : 0;

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
