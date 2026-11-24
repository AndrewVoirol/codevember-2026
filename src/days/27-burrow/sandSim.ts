import {
  AIR_CAVITY,
  AIR_SKY,
  AIR_SOLID,
  ARCH_SPAN_LIMIT,
  CELL_EMPTY,
  CELL_LOOSE_SAND,
  CELL_PACKED_SAND,
  CELL_PEBBLE,
  DIGGER_COUNT,
  GRID_CELLS,
  GRID_H,
  GRID_W,
  MAX_CANTILEVER_SPAN,
  MIN_FULL_ARCH_ROOF,
  PEBBLE_ABUTMENT_BONUS,
  SAND_PIGMENTS,
  SURFACE_FRAC,
  linearToSrgbByte,
} from './constants';

export type DiggerState = 'patrol_down' | 'excavate' | 'haul_up' | 'deposit';

export interface DiggerAgent {
  id: number;
  /** Continuous grid X coordinate in [0, GRID_W) */
  x: number;
  /** Continuous grid Y coordinate in [0, GRID_H) */
  y: number;
  /** Slight depth lane in [-1, 1] inside the glass cavity thickness */
  zLane: number;
  /** Facing angle in XY grid plane (radians, 0 = +X, PI/2 = +Y) */
  heading: number;
  /** Continuous tripod-gait animation phase */
  walkPhase: number;
  /** Mandible excavation cycle phase */
  mandiblePhase: number;
  state: DiggerState;
  stateTimer: number;
  /** Pigment ID of carried sand grain (-1 if empty-handed) */
  carryingPigment: number;
  carryingShade: number;
  /** Current deep excavation destination */
  targetX: number;
  targetY: number;
  /** Preferred surface entrance shaft X */
  homeShaftX: number;
  /** Preferred gallery expansion direction (-1 left, +1 right) */
  branchDir: number;
  /** Last depth at which this digger excavated a grain */
  lastExcavatedY: number;
}

export interface SandSimMetrics {
  excavatedCells: number;
  activeFallingGrains: number;
  collapseEvents: number;
  diggerCount: number;
  tailingsDeposited: number;
  stepCount: number;
}

export interface SandStepOptions {
  /** Allow autonomous burrowers to move, excavate, and haul grains (default true) */
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
}

const GRAVITY_ACCEL = 0.28;
const MAX_FALL_SPEED = 3.2;
const RECOMPACT_TICKS = 110;
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
 * - Stratified geological mineral bands + immovable embedded quartz/basalt river pebbles
 * - Terzaghi granular arching stability vs. Rankine conjugate diagonal shear-fault cave-ins
 * - Exact 1:1 grain mass conservation across gravity settling, avalanches, and digger hauling
 * - Autonomous subterranean burrowing beetles hauling deep colored grains onto surface tailings cones
 * - Zero per-frame heap allocations; packs live RGBA8 DataTextures for WebGPU/TSL parallax rendering
 */
export class SandSimulation {
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

  // Autonomous subterranean digger agents
  readonly diggers: DiggerAgent[] = [];

  // Pre-baked linear RGB + sRGB byte LUT per (pigmentId, shadeByte) for zero-pow O(1) packing
  private readonly pigmentRgbLinear: Float32Array;
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

    for (let i = 0; i < DIGGER_COUNT; i++) {
      this.diggers.push({
        id: i,
        x: 114,
        y: 130,
        zLane: ((i % 2 === 0 ? -1 : 1) * (0.28 + 0.22 * (i >> 1))),
        heading: -Math.PI * 0.5,
        walkPhase: i * 1.7,
        mandiblePhase: i * 0.9,
        state: 'patrol_down',
        stateTimer: 0,
        carryingPigment: -1,
        carryingShade: 128,
        targetX: 114,
        targetY: 96,
        homeShaftX: 114,
        branchDir: i % 2 === 0 ? -1 : 1,
        lastExcavatedY: 96,
      });
    }

    this.reset(initialSeed);
  }

  /** Cell linear index (y = 0 bottom, y = height - 1 top) */
  idx(x: number, y: number): number {
    return y * GRID_W + x;
  }

  /**
   * Resets to a freshly stratified geological sand bed with embedded quartz/basalt
   * river pebbles, a realistic starter vaulted entrance burrow + tailings rim, and 4 diggers.
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

    this.cellType.fill(CELL_EMPTY);
    this.pigmentId.fill(0);
    this.grainShade.fill(128);
    this.bgPigmentId.fill(0);
    this.bgShade.fill(128);
    this.bondHealth.fill(0);
    this.faultDelay.fill(0);
    this.velY.fill(0);
    this.restTicks.fill(0);
    this.movedStamp.fill(0);
    this.stressField.fill(0);
    this.collapseCooldown.fill(0);
    this.pebbles.length = 0;

    // Geological wave phases for organic wavy bedding planes (irrational frequencies)
    const ph1 = this.rng.next() * Math.PI * 2;
    const ph2 = this.rng.next() * Math.PI * 2;
    const ph3 = this.rng.next() * Math.PI * 2;
    const ph4 = this.rng.next() * Math.PI * 2;

    const baseSurface = Math.round(GRID_H * SURFACE_FRAC); // ~176

    for (let x = 0; x < GRID_W; x++) {
      const nx = x / GRID_W;
      // Gentle natural dune crest topography across the top horizon
      const crestWave =
        Math.sin(nx * Math.PI * 2.15 + ph1) * 5.2 +
        Math.sin(nx * Math.PI * 4.73 + ph2) * 2.4 +
        Math.cos(nx * Math.PI * 8.31 + ph3) * 1.1;
      const colTop = Math.max(
        150,
        Math.min(GRID_H - 28, Math.round(baseSurface + crestWave))
      );

      for (let y = 0; y < GRID_H; y++) {
        const i = y * GRID_W + x;
        // Geological fold displacement varies gently with depth so lower strata dip realistically
        const ny = y / GRID_H;
        const strataWarp =
          Math.sin(nx * Math.PI * 2.35 + ph1 + ny * 1.4) * 9.5 +
          Math.sin(nx * Math.PI * 5.17 + ph2 - ny * 2.2) * 4.2 +
          Math.cos(nx * Math.PI * 11.4 + ph4 + ny * 3.1) * 1.6;
        const depthFromTop = colTop - y;
        const geoY = y + strataWarp;

        const stratum = this.classifyStratum(geoY, depthFromTop, x, y);
        const grainJitter = Math.floor(
          128 + (this.rng.next() - 0.5) * 118 + Math.sin(geoY * 1.65) * 18
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

    // Embed 8 natural rounded quartz & basalt river pebbles (immovable structural lintels!)
    this.embedRiverPebbles(baseSurface);

    // Carve starter arched entrance burrow + branching subterranean galleries + miniature tailings rim
    this.starterMouthX = 104 + Math.floor(this.rng.next() * 24);
    if (seedStarterBurrow) {
      this.carveStarterBurrow(this.starterMouthX);
    }

    // Place 4 autonomous burrowing beetles inside the starter network
    this.initializeDiggers(seedStarterBurrow);

    this.recomputeSurfaceAndOpticalFields();
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

  private embedRiverPebbles(baseSurface: number): void {
    // Carefully distributed hard quartz (8) and basalt (9) pebbles serving as natural tunnel lintels
    const specs: Array<[number, number, number, number, number]> = [
      [54, 124, 11, 7, 8],   // upper-left quartz pebble
      [148, 128, 12, 7.5, 9], // central-upper basalt lintel right above starter gallery!
      [234, 132, 10.5, 6.5, 8], // upper-right quartz pebble
      [82, 76, 13, 8, 9],    // mid-left basalt boulder
      [196, 86, 11.5, 7, 8], // mid-right quartz pebble
      [256, 62, 10, 6.5, 9], // lower-right basalt pebble
      [134, 42, 12.5, 7.5, 8], // deep central quartz pebble
      [42, 38, 10, 6.5, 9],  // deep left basalt pebble
    ];

    for (let k = 0; k < specs.length; k++) {
      const [bx, by, brx, bry, pig] = specs[k];
      const cx = Math.max(24, Math.min(GRID_W - 24, Math.round(bx + (this.rng.next() - 0.5) * 14)));
      const cy = Math.max(20, Math.min(baseSurface - 30, Math.round(by + (this.rng.next() - 0.5) * 10)));
      const rx = brx * (0.9 + this.rng.next() * 0.22);
      const ry = bry * (0.9 + this.rng.next() * 0.22);
      const angle = (this.rng.next() - 0.5) * 0.55;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      this.pebbles.push({ cx, cy, rx, ry, angle, pigmentId: pig });

      const pad = Math.ceil(Math.max(rx, ry)) + 2;
      const x0 = Math.max(1, cx - pad);
      const x1 = Math.min(GRID_W - 2, cx + pad);
      const y0 = Math.max(1, cy - pad);
      const y1 = Math.min(GRID_H - 2, cy + pad);

      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x - cx;
          const dy = y - cy;
          const lx = dx * cosA + dy * sinA;
          const ly = -dx * sinA + dy * cosA;
          // Smooth organically flattened river-stone contour (no polar star harmonics)
          const warp = 1 + 0.06 * Math.sin(lx * 0.28 + k * 1.7) * Math.cos(ly * 0.34 - k);
          const normR2 = (lx * lx) / (rx * rx) + (ly * ly) / (ry * ry);
          if (normR2 <= warp) {
            const idx = y * GRID_W + x;
            this.cellType[idx] = CELL_PEBBLE;
            this.pigmentId[idx] = pig;
            this.bgPigmentId[idx] = pig;
            this.bondHealth[idx] = 255;
            // Smooth concentric mineral shading inside pebble (brighter center, darker wet rim)
            const edgeFade = Math.max(0, 1 - normR2);
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

  private carveStarterBurrow(mouthX: number): void {
    // Find exact surface row at mouthX
    let mouthY = GRID_H - 1;
    while (mouthY > 20 && this.cellType[mouthY * GRID_W + mouthX] === CELL_EMPTY) {
      mouthY--;
    }

    // 1. Narrow arched main entrance shaft curving smoothly downward
    const rMain = 5.4;
    this.carvePolyline(
      [
        [mouthX, mouthY + 2],
        [mouthX + 2, mouthY - 18],
        [mouthX + 9, mouthY - 38],
        [mouthX + 22, mouthY - 56],
        [mouthX + 38, mouthY - 64], // passes cleanly under the central basalt lintel!
        [mouthX + 58, mouthY - 78],
        [mouthX + 74, mouthY - 94],
      ],
      rMain
    );

    // 2. Left branching vaulted foraging gallery dipping into the Yellow Ochre & Sienna beds
    this.carvePolyline(
      [
        [mouthX + 8, mouthY - 36],
        [mouthX - 12, mouthY - 48],
        [mouthX - 30, mouthY - 64],
        [mouthX - 44, mouthY - 76],
      ],
      4.9
    );

    // 3. Lower arched chamber fork curving beneath the central shaft
    this.carvePolyline(
      [
        [mouthX + 22, mouthY - 56],
        [mouthX + 12, mouthY - 78],
        [mouthX + 24, mouthY - 98],
      ],
      4.8
    );

    // 4. Deposit a natural starter crater rim of deep Burnt Sienna & Yellow Ochre tailings around mouthX
    for (let k = 0; k < 110; k++) {
      const side = k % 2 === 0 ? -1 : 1;
      const dist = 6 + Math.floor(this.rng.next() * 14);
      const tx = Math.max(4, Math.min(GRID_W - 5, mouthX + side * dist));
      let ty = GRID_H - 3;
      while (ty > 4 && this.cellType[(ty - 1) * GRID_W + tx] === CELL_EMPTY) {
        ty--;
      }
      if (ty < GRID_H - 4 && this.cellType[ty * GRID_W + tx] === CELL_EMPTY) {
        const idx = ty * GRID_W + tx;
        this.cellType[idx] = CELL_LOOSE_SAND;
        this.pigmentId[idx] = k % 3 === 0 ? 2 : 3; // Yellow Ochre & Burnt Sienna tailings!
        this.grainShade[idx] = Math.max(40, Math.min(230, Math.round(115 + (this.rng.next() - 0.5) * 70)));
        this.bondHealth[idx] = 0;
        this.velY[idx] = 0;
      }
    }

    // Settle the starter tailings gently so the crater cone starts crisp at natural repose
    for (let s = 0; s < 18; s++) {
      this.stepLooseSandGravity();
    }
  }

  private carvePolyline(pts: Array<[number, number]>, radius: number): void {
    for (let i = 0; i < pts.length - 1; i++) {
      this.excavateCapsule(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], radius, false);
    }
  }

  private initializeDiggers(hasStarterBurrow: boolean): void {
    const mx = this.starterMouthX;
    let mouthY = Math.round(GRID_H * SURFACE_FRAC);
    for (let y = GRID_H - 1; y >= 10; y--) {
      if (this.cellType[y * GRID_W + Math.min(GRID_W - 2, mx + 8)] !== CELL_EMPTY) {
        mouthY = y;
        break;
      }
    }

    const starts: Array<{
      x: number;
      y: number;
      state: DiggerState;
      carrying: number;
      tx: number;
      ty: number;
      dir: number;
    }> = hasStarterBurrow
      ? [
          // Digger 0: actively hauling a Burnt Sienna grain right up the entrance shaft!
          {
            x: mx + 4,
            y: mouthY - 22,
            state: 'haul_up',
            carrying: 3,
            tx: mx,
            ty: mouthY + 1,
            dir: -1,
          },
          // Digger 1: excavating the deep right Burnt Sienna / Yellow Ochre gallery tip
          {
            x: mx + 72,
            y: mouthY - 94,
            state: 'excavate',
            carrying: -1,
            tx: mx + 76,
            ty: mouthY - 96,
            dir: 1,
          },
          // Digger 2: excavating the left Yellow Ochre branch tip
          {
            x: mx - 42,
            y: mouthY - 75,
            state: 'patrol_down',
            carrying: -1,
            tx: mx - 47,
            ty: mouthY - 78,
            dir: -1,
          },
          // Digger 3: patrolling lower central chamber branch
          {
            x: mx + 22,
            y: mouthY - 95,
            state: 'haul_up',
            carrying: 2,
            tx: mx,
            ty: mouthY + 1,
            dir: 1,
          },
        ]
      : [
          { x: 100, y: mouthY - 30, state: 'patrol_down', carrying: -1, tx: 100, ty: mouthY - 60, dir: -1 },
          { x: 140, y: mouthY - 40, state: 'patrol_down', carrying: -1, tx: 150, ty: mouthY - 80, dir: 1 },
          { x: 180, y: mouthY - 35, state: 'patrol_down', carrying: -1, tx: 185, ty: mouthY - 70, dir: 1 },
          { x: 210, y: mouthY - 25, state: 'patrol_down', carrying: -1, tx: 205, ty: mouthY - 65, dir: -1 },
        ];

    for (let i = 0; i < this.diggers.length; i++) {
      const d = this.diggers[i];
      const st = starts[i % starts.length];
      d.x = Math.max(6, Math.min(GRID_W - 7, st.x));
      d.y = Math.max(8, Math.min(GRID_H - 8, st.y));
      d.state = st.state;
      d.stateTimer = 0;
      d.carryingPigment = st.carrying;
      d.carryingShade = 135;
      d.targetX = st.tx;
      d.targetY = st.ty;
      d.homeShaftX = mx;
      d.branchDir = st.dir;
      d.heading = st.state === 'haul_up' ? Math.PI * 0.5 : -Math.PI * 0.45;
      d.lastExcavatedY = st.ty;
    }
  }

  /**
   * Excavates a smooth capsule (tunnel stroke) between grid coordinates `(x0, y0)` and `(x1, y1)`.
   * Never digs through immovable river pebbles (`CELL_PEBBLE`).
   * Optionally dislodges a few stray roof grains along the carved ceiling lip for organic feel.
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
    const rimR2 = (radius + 1.45) * (radius + 1.45);

    let removed = 0;
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
          this.cellType[idx] = CELL_EMPTY;
          this.bondHealth[idx] = 0;
          this.faultDelay[idx] = 0;
          this.velY[idx] = 0;
          this.restTicks[idx] = 0;
          removed++;
        } else if (scatterLooseRim && dist2 <= rimR2 && ct === CELL_PACKED_SAND && ddy > radius * 0.5) {
          // Small chance to loosen a stray grain on the freshly dug ceiling for live trickle!
          if (this.rng.next() < 0.085) {
            this.cellType[idx] = CELL_LOOSE_SAND;
            this.bondHealth[idx] = 0;
            this.velY[idx] = 0.25;
          }
        }
      }
    }

    if (removed > 0) {
      this.excavatedCells += removed;
      this.texturesDirty = true;
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
          this.velY[idx] = 0;
          this.restTicks[idx] = 0;
          removed++;
        }
      }
    }
    if (removed > 0) {
      this.excavatedCells += removed;
      this.texturesDirty = true;
    }
    return removed;
  }

  /**
   * Pours a fresh trickling stream of colored mineral sand grains at grid coordinate `(gx, gy)`
   * (typically in the open air chamber above the sand crest).
   * Slowly cycles across natural mineral pigments so poured dunes form authentic cross-bedded bands!
   */
  pourSandAt(gx: number, gy: number, grainBudget = 11): number {
    this.pourPigmentPhase += 0.018;
    // Cycle smoothly through natural mineral sands (Yellow Ochre, Burnt Sienna, Bone Calcite, Raw Umber, Warm Kaolin)
    const pourPalette = [2, 3, 0, 1, 4, 2, 5];
    const basePigment = pourPalette[Math.floor(this.pourPigmentPhase) % pourPalette.length];

    const cx = Math.max(4, Math.min(GRID_W - 5, Math.round(gx)));
    const cy = Math.max(8, Math.min(GRID_H - 3, Math.round(gy)));

    let spawned = 0;
    for (let k = 0; k < grainBudget; k++) {
      const ox = Math.round((this.rng.next() - 0.5) * 6.0);
      const oy = Math.round((this.rng.next() - 0.5) * 3.0);
      const x = Math.max(3, Math.min(GRID_W - 4, cx + ox));
      const y = Math.max(4, Math.min(GRID_H - 2, cy + oy));
      const idx = y * GRID_W + x;
      if (this.cellType[idx] === CELL_EMPTY) {
        this.cellType[idx] = CELL_LOOSE_SAND;
        // Subtle 15% inter-grain mineral mixture for natural granular richness
        this.pigmentId[idx] =
          this.rng.next() < 0.14
            ? pourPalette[(Math.floor(this.pourPigmentPhase) + 1) % pourPalette.length]
            : basePigment;
        this.grainShade[idx] = Math.max(
          28,
          Math.min(238, Math.round(128 + (this.rng.next() - 0.5) * 96))
        );
        this.bondHealth[idx] = 0;
        this.faultDelay[idx] = 0;
        this.velY[idx] = 0.45 + this.rng.next() * 0.45;
        this.restTicks[idx] = 0;
        spawned++;
      }
    }
    if (spawned > 0) {
      this.texturesDirty = true;
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
   * and autonomous subterranean burrowers by one fixed simulation step (~1/60s).
   */
  step(options?: SandStepOptions): void {
    const enableDiggers = options?.enableDiggers ?? true;
    const enableArchCheck = options?.enableArchCheck ?? true;

    this.stepCount++;

    // Decay column collapse debounce cooldowns
    for (let x = 0; x < GRID_W; x++) {
      if (this.collapseCooldown[x] > 0) this.collapseCooldown[x]--;
    }

    // 1. Arch stability & Rankine shear-fault propagation (every 2 steps for crisp speed + immediate trigger on step 1)
    if (enableArchCheck && (this.stepCount & 1) === 1) {
      this.evaluateArchStabilityAndFaults();
    }

    // 2. Release queued shear-fault cells layer-by-layer (progressive cracking cascade!)
    this.advanceFaultWave();

    // 3. Exact mass-conserving loose sand gravity, free-fall acceleration & repose avalanching
    const movedGrains = this.stepLooseSandGravity();

    // 4. Autonomous subterranean burrowing beetles (patrol -> excavate -> haul up -> deposit tailings)
    if (enableDiggers) {
      this.stepDiggers();
    }

    // 5. Fast O(W) column surface update so diggers & pour queries stay accurate every tick
    if (movedGrains > 0 || this.texturesDirty) {
      this.updateTopSolidColumns();
      this.texturesDirty = true;
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

  /** Lightweight per-step top solid surface tracker for diggers & air-zone detection. */
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
   * - Over-wide flat chambers exceed local arch span -> nucleate a conjugate diagonal
   *   Mohr-Coulomb slip wedge (`~65 deg`) propagating upward in cascading faulted strata bands!
   * - Detached zero-abutment floating islands immediately convert to loose falling sand.
   */
  private evaluateArchStabilityAndFaults(): void {
    // Gently decay residual stress visual field
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

    // Scan horizontal soffit rows bottom-to-top (y = 2..GRID_H - 3)
    for (let y = 2; y < GRID_H - 2; y++) {
      const rowOff = y * GRID_W;
      const belowOff = (y - 1) * GRID_W;
      let x = 1;

      while (x < GRID_W - 1) {
        const i = rowOff + x;
        // Identify an unsupported PACKED_SAND ceiling cell (below is EMPTY or rapidly falling LOOSE_SAND)
        const belowType = ct[belowOff + x];
        const isUnsupported =
          ct[i] === CELL_PACKED_SAND &&
          fd[i] === 0 &&
          (belowType === CELL_EMPTY || (belowType === CELL_LOOSE_SAND && this.velY[belowOff + x] > 0.65));

        if (!isUnsupported) {
          x++;
          continue;
        }

        // Measure contiguous horizontal unsupported ceiling run [spanL..spanR]
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

        // Inspect left & right end conditions (abutment vs. open cantilever)
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

        // 1-cell hanging tooth stalactite (empty on both left and right) -> gently self-round!
        if (!leftSupported && !rightSupported) {
          if (span <= 3) {
            for (let cx = spanL; cx <= spanR; cx++) {
              ct[rowOff + cx] = CELL_LOOSE_SAND;
              this.bondHealth[rowOff + cx] = 0;
              this.velY[rowOff + cx] = 0.25;
            }
            this.texturesDirty = true;
          } else {
            // Detached floating sand island with zero lateral support -> drop whole island!
            this.triggerShearWedgeCollapse(spanL, spanR, y, true);
          }
          continue;
        }

        // Single-sided cantilever overhang ledge
        if (!leftSupported || !rightSupported) {
          if (span > MAX_CANTILEVER_SPAN) {
            // Shed outer cantilevered lip beyond MAX_CANTILEVER_SPAN
            const dropStart = !leftSupported ? spanL : spanL + MAX_CANTILEVER_SPAN;
            const dropEnd = !leftSupported ? spanR - MAX_CANTILEVER_SPAN : spanR;
            const dropSpan = dropEnd - dropStart + 1;
            if (dropSpan >= 8) {
              this.triggerShearWedgeCollapse(dropStart, dropEnd, y, false);
            } else {
              for (let cx = dropStart; cx <= dropEnd; cx++) {
                // Shed thin overhanging tip + up to 3 rows above
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

        // Compute effective Terzaghi arch span limit
        let maxSpan = ARCH_SPAN_LIMIT;
        if (leftIsHard) maxSpan += PEBBLE_ABUTMENT_BONUS;
        if (rightIsHard) maxSpan += PEBBLE_ABUTMENT_BONUS;

        // Also check if a hard pebble sits right inside the roof arch within 4 cells above center (lintel reinforcement!)
        let hasOverheadPebbleLintel = false;
        for (let dy = 1; dy <= 5 && y + dy < GRID_H - 1; dy++) {
          if (ct[(y + dy) * GRID_W + midX] === CELL_PEBBLE) {
            hasOverheadPebbleLintel = true;
            break;
          }
        }
        if (hasOverheadPebbleLintel) maxSpan += PEBBLE_ABUTMENT_BONUS;

        // Thin crust cannot sustain full compressive arch thrust
        if (minRoof < MIN_FULL_ARCH_ROOF) {
          const crustScale = Math.max(0.28, minRoof / MIN_FULL_ARCH_ROOF);
          maxSpan = Math.max(4, Math.floor(maxSpan * crustScale));
        }

        if (span > maxSpan) {
          // Exceeded granular arching stability -> initiate Rankine diagonal shear-fault collapse!
          this.triggerShearWedgeCollapse(spanL, spanR, y, false);
        } else if (span >= maxSpan - 4) {
          // Near arching limit: mark subtle compressive-shear stress lines along ceiling
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
   * Nucleates a realistic conjugate diagonal shear-fault slip wedge (`~64–68 deg` Mohr-Coulomb
   * fracture planes) upward from an over-wide unsupported span `[spanL..spanR]` at row `baseY`.
   * Staggers `faultDelay` upward by row so strata visibly crack and slump in cascading bands!
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

    // Count distinct collapse event (debounced across nearby columns for 24 ticks so one slump = 1 event)
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
    // Shear plane slope: dx/dy ~ 0.40 (converges upward into natural gothic arch OR breaks surface!)
    const slope = isFloatingIsland ? 0.14 : 0.38;
    const maxWedgeH = isFloatingIsland ? 42 : Math.min(GRID_H - 2 - baseY, Math.ceil(span * 1.35) + 8);

    // Track per-column pebble shelter (hard stones shield sand directly above them!)
    for (let dy = 0; dy <= maxWedgeH; dy++) {
      const y = baseY + dy;
      if (y >= GRID_H - 1) break;

      const jitterL = Math.sin(y * 0.47 + spanL * 0.13) * 1.25;
      const jitterR = Math.cos(y * 0.53 + spanR * 0.17) * 1.25;
      const rowL = Math.max(1, Math.floor(spanL + dy * slope + jitterL));
      const rowR = Math.min(GRID_W - 2, Math.ceil(spanR - dy * slope + jitterR));
      if (rowL > rowR) break;

      const rowOff = y * GRID_W;
      let anySandInRow = false;

      for (let x = rowL; x <= rowR; x++) {
        const idx = rowOff + x;
        if (ct[idx] === CELL_PEBBLE) continue;
        if (ct[idx] !== CELL_PACKED_SAND) continue;

        // Check if sheltered directly beneath by an embedded pebble within 4 cells
        let shieldedByPebble = false;
        for (let py = 1; py <= 4 && y - py >= baseY; py++) {
          if (ct[(y - py) * GRID_W + x] === CELL_PEBBLE) {
            shieldedByPebble = true;
            break;
          }
        }
        if (shieldedByPebble) continue;

        anySandInRow = true;
        // Progressive upward crack release: soffit breaks first (1..3 ticks), upper strata follow!
        const edgeLag = Math.min(x - rowL, rowR - x) < 2 ? 1 : 0;
        const delay = Math.max(
          1,
          Math.min(44, 1 + Math.floor(dy * 0.42) + edgeLag + (this.rng.next() < 0.35 ? 1 : 0))
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
          this.velY[i] = 0.42 + this.rng.next() * 0.35;
          this.restTicks[i] = 0;
          this.texturesDirty = true;
        }
      }
    }
    this.activeFaultCells = remaining;
  }

  /**
   * Moves `LOOSE_SAND` grains downward under gravity with acceleration & natural ~35 deg repose avalanching.
   * Strictly 1-to-1 cell swaps (`src -> EMPTY`, `EMPTY -> dst`): ZERO grains vanish or duplicate!
   */
  private stepLooseSandGravity(): number {
    this.moveTickId++;
    const stamp = this.moveTickId;
    const ct = this.cellType;
    const leftToRight = (this.stepCount & 1) === 0;
    let fallingCount = 0;
    let movedCount = 0;

    for (let y = 1; y < GRID_H; y++) {
      const rowOff = y * GRID_W;
      for (let sx = 0; sx < GRID_W; sx++) {
        const x = leftToRight ? sx : GRID_W - 1 - sx;
        const idx = rowOff + x;
        if (ct[idx] !== CELL_LOOSE_SAND || this.movedStamp[idx] === stamp) continue;

        // 1. Direct vertical free-fall (accelerated under gravity!)
        const belowIdx = idx - GRID_W;
        if (ct[belowIdx] === CELL_EMPTY) {
          let vy = this.velY[idx] + GRAVITY_ACCEL;
          if (vy > MAX_FALL_SPEED) vy = MAX_FALL_SPEED;
          const maxSteps = Math.max(1, Math.floor(vy));

          let destY = y;
          for (let s = 1; s <= maxSteps; s++) {
            const ny = y - s;
            if (ny < 0 || ct[ny * GRID_W + x] !== CELL_EMPTY) break;
            destY = ny;
          }

          const destIdx = destY * GRID_W + x;
          this.moveGrain(idx, destIdx, vy, stamp);
          fallingCount++;
          movedCount++;
          continue;
        }

        // 2. Diagonal repose slide (~45 deg primary + gentle ~35 deg 2:1 roll on loose crests)
        const dirFirst = this.rng.next() < 0.5 ? -1 : 1;
        let slid = false;

        for (let attempt = 0; attempt < 2; attempt++) {
          const dir = attempt === 0 ? dirFirst : -dirFirst;
          const nx = x + dir;
          if (nx < 1 || nx >= GRID_W - 1) continue;

          const sideIdx = rowOff + nx;
          const diagIdx = (y - 1) * GRID_W + nx;
          // Require side cell empty so sand never clips diagonally through solid rock/pebble barriers!
          if (ct[sideIdx] === CELL_EMPTY && ct[diagIdx] === CELL_EMPTY) {
            const nextVy = Math.max(0.35, this.velY[idx] * 0.55);
            this.moveGrain(idx, diagIdx, nextVy, stamp);
            fallingCount++;
            movedCount++;
            slid = true;
            break;
          }

          // Gentle 2-horizontal-to-1-vertical roll for authentic ~34-37 deg dry sand repose dunes!
          const nx2 = x + dir * 2;
          if (
            nx2 >= 1 &&
            nx2 < GRID_W - 1 &&
            this.velY[idx] > 0.15 &&
            this.rng.next() < 0.42
          ) {
            const side2Idx = rowOff + nx2;
            const diag2Idx = (y - 1) * GRID_W + nx2;
            if (
              ct[sideIdx] === CELL_EMPTY &&
              ct[side2Idx] === CELL_EMPTY &&
              ct[diag2Idx] === CELL_EMPTY
            ) {
              this.moveGrain(idx, diag2Idx, this.velY[idx] * 0.45, stamp);
              fallingCount++;
              movedCount++;
              slid = true;
              break;
            }
          }
        }

        if (!slid) {
          // Grain settled on support
          this.velY[idx] *= 0.25;
          if (this.velY[idx] < 0.05) this.velY[idx] = 0;
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

  private moveGrain(srcIdx: number, dstIdx: number, newVelY: number, stamp: number): void {
    this.cellType[dstIdx] = CELL_LOOSE_SAND;
    this.pigmentId[dstIdx] = this.pigmentId[srcIdx];
    this.grainShade[dstIdx] = this.grainShade[srcIdx];
    this.bondHealth[dstIdx] = 0;
    this.faultDelay[dstIdx] = 0;
    this.velY[dstIdx] = newVelY;
    this.restTicks[dstIdx] = 0;
    this.movedStamp[dstIdx] = stamp;

    this.cellType[srcIdx] = CELL_EMPTY;
    this.bondHealth[srcIdx] = 0;
    this.faultDelay[srcIdx] = 0;
    this.velY[srcIdx] = 0;
    this.restTicks[srcIdx] = 0;
  }

  /**
   * Autonomous Subterranean Burrowing Beetles (`DIGGER_COUNT = 4`):
   * Quietly patrol tunnels, bite individual deep colored sand grains (`PACKED_SAND`),
   * haul them up vertical shafts to the surface crest, and drop them onto miniature
   * cratered tailings mounds (exact grain mass conservation!).
   */
  private stepDiggers(): void {
    const ct = this.cellType;

    for (let i = 0; i < this.diggers.length; i++) {
      const d = this.diggers[i];
      d.walkPhase += 0.34;
      d.stateTimer++;

      if (d.state === 'patrol_down') {
        this.stepDiggerPatrolDown(d, ct);
      } else if (d.state === 'excavate') {
        this.stepDiggerExcavate(d, ct);
      } else if (d.state === 'haul_up') {
        this.stepDiggerHaulUp(d, ct);
      } else {
        this.stepDiggerDeposit(d, ct);
      }
    }
  }

  private stepDiggerPatrolDown(d: DiggerAgent, ct: Uint8Array): void {
    const dx = d.targetX - d.x;
    const dy = d.targetY - d.y;
    const dist = Math.hypot(dx, dy);

    if (dist < 4.5 || d.stateTimer > 260) {
      d.state = 'excavate';
      d.stateTimer = 0;
      return;
    }

    const speed = 0.68;
    const stepX = (dx / dist) * speed;
    const stepY = (dy / dist) * speed;
    const nextX = Math.max(8, Math.min(GRID_W - 9, d.x + stepX));
    const nextY = Math.max(14, Math.min(GRID_H - 12, d.y + stepY));
    const cellAhead = ct[Math.round(nextY) * GRID_W + Math.round(nextX)];

    if (cellAhead === CELL_PEBBLE) {
      // Skirt cleanly around hard river pebbles
      d.targetX = Math.max(18, Math.min(GRID_W - 19, d.targetX + d.branchDir * 18));
      d.targetY = Math.max(24, d.targetY - 8);
      return;
    }

    // If we hit solid deep sand while heading down toward our work face, start biting!
    const colTop = this.topSolidY[Math.round(nextX)] || Math.round(GRID_H * SURFACE_FRAC);
    if (cellAhead === CELL_PACKED_SAND && nextY < colTop - 22) {
      d.heading = Math.atan2(dy, dx);
      d.state = 'excavate';
      d.stateTimer = 0;
      return;
    }

    // Move forward through tunnel (or gentle wiggle if crossing loose tumbled sand)
    d.x = nextX;
    d.y = nextY;
    d.heading = this.lerpAngle(d.heading, Math.atan2(stepY, stepX), 0.25);
  }

  private stepDiggerExcavate(d: DiggerAgent, ct: Uint8Array): void {
    d.mandiblePhase += 0.55;
    if (d.stateTimer < 10) {
      // Subtle digging wiggle at the work face
      return;
    }

    // Search a small radius around the digger for a deep PACKED_SAND grain to bite cleanly
    const cx = Math.round(d.x);
    const cy = Math.round(d.y);
    let foundIdx = -1;
    let foundX = cx;
    let foundY = cy;

    // Prioritize digging slightly outward in branchDir & slightly downward (natural branching tunnel!)
    const maxR = 6;
    for (let r = 1; r <= maxR && foundIdx === -1; r++) {
      for (let a = 0; a < 12; a++) {
        const ang = (a / 12) * Math.PI * 2 + (d.branchDir > 0 ? -0.35 : Math.PI + 0.35);
        const sx = Math.max(4, Math.min(GRID_W - 5, Math.round(cx + Math.cos(ang) * r)));
        const sy = Math.max(12, Math.min(GRID_H - 35, Math.round(cy + Math.sin(ang) * r)));
        const idx = sy * GRID_W + sx;
        if (ct[idx] === CELL_PACKED_SAND && this.faultDelay[idx] === 0) {
          // Avoid over-widening ceilings directly above wide open spans (natural arch instinct!)
          const colTop = this.topSolidY[sx] || Math.round(GRID_H * SURFACE_FRAC);
          if (sy < colTop - 16) {
            foundIdx = idx;
            foundX = sx;
            foundY = sy;
            break;
          }
        }
      }
    }

    if (foundIdx !== -1) {
      // Bite exactly 1 grain into mandibles -> 100% exact mass conservation!
      d.carryingPigment = this.pigmentId[foundIdx];
      d.carryingShade = this.grainShade[foundIdx];
      ct[foundIdx] = CELL_EMPTY;
      this.bondHealth[foundIdx] = 0;
      this.velY[foundIdx] = 0;
      this.excavatedCells++;
      d.lastExcavatedY = foundY;
      d.x = d.x * 0.65 + foundX * 0.35;
      d.y = d.y * 0.65 + foundY * 0.35;
      d.heading = Math.atan2(foundY - cy, foundX - cx);
      d.state = 'haul_up';
      d.stateTimer = 0;
      this.texturesDirty = true;
    } else {
      // No solid grain within reach: pick a fresh deep target face in the gallery
      this.pickNextDeepTarget(d);
      d.state = 'patrol_down';
      d.stateTimer = 0;
    }
  }

  private stepDiggerHaulUp(d: DiggerAgent, ct: Uint8Array): void {
    // Head upward toward the open surface crest near homeShaftX
    const sx = Math.max(14, Math.min(GRID_W - 15, Math.round(d.homeShaftX)));
    const surfY = Math.max(110, (this.topSolidY[sx] || Math.round(GRID_H * SURFACE_FRAC)) + 2);
    const goalX = sx;
    const goalY = surfY;

    const dx = goalX - d.x;
    const dy = goalY - d.y;
    const dist = Math.hypot(dx, dy);

    if (d.y >= surfY - 2.5 || dist < 4.5) {
      // Reached surface mouth -> transition to deposit on the crater rim!
      const side = (d.id + this.tailingsDeposited) % 2 === 0 ? -1 : 1;
      const rimDist = 5 + ((d.id * 3 + this.tailingsDeposited * 2) % 10);
      d.targetX = Math.max(10, Math.min(GRID_W - 11, sx + side * rimDist));
      d.targetY = Math.min(GRID_H - 6, (this.topSolidY[Math.round(d.targetX)] || surfY) + 2);
      d.state = 'deposit';
      d.stateTimer = 0;
      return;
    }

    // Prefer climbing along open EMPTY tunnel cells if nearby, curving steadily upward toward the mouth
    let bestVx = dx / dist;
    let bestVy = dy / dist;
    const lookR = 3.2;
    let bestScore = -1e9;

    for (let k = -2; k <= 2; k++) {
      const candAng = Math.atan2(dy, dx) + k * 0.38;
      const vx = Math.cos(candAng);
      const vy = Math.sin(candAng);
      const px = Math.max(4, Math.min(GRID_W - 5, Math.round(d.x + vx * lookR)));
      const py = Math.max(4, Math.min(GRID_H - 4, Math.round(d.y + vy * lookR)));
      const cType = ct[py * GRID_W + px];
      if (cType === CELL_PEBBLE) continue;

      const alignScore = (vx * dx + vy * dy) / dist;
      const openBonus = cType === CELL_EMPTY ? 1.15 : cType === CELL_LOOSE_SAND ? 0.2 : -0.35;
      const score = alignScore + openBonus;
      if (score > bestScore) {
        bestScore = score;
        bestVx = vx;
        bestVy = vy;
      }
    }

    const climbSpeed = 0.72;
    d.x = Math.max(6, Math.min(GRID_W - 7, d.x + bestVx * climbSpeed));
    d.y = Math.max(8, Math.min(GRID_H - 6, d.y + bestVy * climbSpeed));
    d.heading = this.lerpAngle(d.heading, Math.atan2(bestVy, bestVx), 0.28);
  }

  private stepDiggerDeposit(d: DiggerAgent, ct: Uint8Array): void {
    const dx = d.targetX - d.x;
    const gx = Math.max(6, Math.min(GRID_W - 7, Math.round(d.x)));
    const localSurf = (this.topSolidY[gx] || Math.round(GRID_H * SURFACE_FRAC)) + 1.5;
    const dy = localSurf - d.y;

    if (Math.abs(dx) > 2.2 && d.stateTimer < 45) {
      const stepX = Math.sign(dx) * 0.62;
      d.x = Math.max(6, Math.min(GRID_W - 7, d.x + stepX));
      d.y = d.y * 0.65 + localSurf * 0.35;
      d.heading = this.lerpAngle(d.heading, Math.atan2(dy, stepX), 0.3);
      return;
    }

    // Drop the carried colored sand grain onto the surface crest (exact conservation!)
    if (d.carryingPigment >= 0) {
      const dropX = Math.max(4, Math.min(GRID_W - 5, Math.round(d.x)));
      let dropY = Math.min(GRID_H - 3, Math.max(4, Math.round(d.y) + 1));
      // Find first empty slot at or just above the surface crest
      while (dropY < GRID_H - 2 && ct[dropY * GRID_W + dropX] !== CELL_EMPTY) {
        dropY++;
      }
      if (ct[dropY * GRID_W + dropX] === CELL_EMPTY) {
        const idx = dropY * GRID_W + dropX;
        ct[idx] = CELL_LOOSE_SAND;
        this.pigmentId[idx] = d.carryingPigment;
        this.grainShade[idx] = d.carryingShade;
        this.bondHealth[idx] = 0;
        this.faultDelay[idx] = 0;
        this.velY[idx] = 0.25;
        this.restTicks[idx] = 0;
        d.carryingPigment = -1;
        this.tailingsDeposited++;
        this.texturesDirty = true;
      }
    }

    // Choose next deep subterranean excavation point & head back down!
    this.pickNextDeepTarget(d);
    d.state = 'patrol_down';
    d.stateTimer = 0;
  }

  private pickNextDeepTarget(d: DiggerAgent): void {
    // Pick a deep gallery face (y in [34..118] -> rich Burnt Sienna, Yellow Ochre & Raw Umber layers!)
    const spread = 18 + Math.floor(this.rng.next() * 78);
    let tx = d.homeShaftX + d.branchDir * spread;
    if (tx < 24 || tx > GRID_W - 25) {
      d.branchDir = -d.branchDir;
      tx = Math.max(24, Math.min(GRID_W - 25, d.homeShaftX + d.branchDir * 36));
    }
    const ty = 34 + Math.floor(this.rng.next() * 82);
    d.targetX = tx;
    d.targetY = ty;
    if (this.rng.next() < 0.28) d.branchDir = -d.branchDir;
  }

  private lerpAngle(cur: number, target: number, alpha: number): number {
    let diff = target - cur;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    return cur + diff * alpha;
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

    // 1. Find highest resting solid cell (topSolidY) AND highest total overburden including
    //    active slumping LOOSE_SAND (overburdenTopY) so mid-bed collapses never expose AIR_SKY!
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
          // Ignore single isolated high-air poured droplets far above neighboring horizon;
          // count packed sand, pebbles, resting loose sand, or dense slumping clusters (solid neighbor below)
          const isDenseSlump =
            c !== CELL_LOOSE_SAND ||
            vy[idx] <= 0.35 ||
            (y >= 2 && (ct[(y - 1) * GRID_W + x] !== CELL_EMPTY || ct[(y - 2) * GRID_W + x] !== CELL_EMPTY));
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

      // Record top grain linear RGB for the 3D top sand-crest ribbon mesh
      const pOff = topPigment * 3;
      const contrast = this.pigmentContrast[topPigment];
      const shadeMul = 1 + ((topShade - 128) / 128) * contrast;
      this.surfaceRgb[x * 3] = Math.min(1, this.pigmentRgbLinear[pOff] * shadeMul);
      this.surfaceRgb[x * 3 + 1] = Math.min(1, this.pigmentRgbLinear[pOff + 1] * shadeMul);
      this.surfaceRgb[x * 3 + 2] = Math.min(1, this.pigmentRgbLinear[pOff + 2] * shadeMul);
    }

    // 2. Smooth neighboring bedding horizon (rolling mean over ±18 cols -> keeps narrow shafts
    //    & active mid-bed collapse zones firmly classified as shadowed AIR_CAVITY!)
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

    // 3. Classify every cell: SOLID vs SKY (open air above smooth crest) vs CAVITY (subterranean burrow!)
    //    Count ALL non-empty cells (including slumping LOOSE_SAND!) as blocking horizontal open sky!
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
        // Must sit strictly above all solid/slumping overburden in column cx
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

    // Smooth 3D top sand crest height from the exact continuous sky-floor horizon (zero sheer cliffs!)
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

    // 4. Fast 2-pass Chamfer Distance Transforms:
    //    (a) distBufA = distance inside CAVITY away from nearest SOLID wall (recessed tunnel depth!)
    //    (b) distBufB = distance inside SOLID away from nearest EMPTY cell (thin-wall backlight glow!)
    const dCav = this.distBufA;
    const dSol = this.distBufB;
    const INF = 28;

    for (let i = 0; i < GRID_CELLS; i++) {
      dCav[i] = air[i] === AIR_CAVITY ? INF : 0;
      dSol[i] = ct[i] !== CELL_EMPTY ? INF : 0;
    }

    // Forward pass (bottom-left -> top-right, branchless integer min)
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

    // Backward pass (top-right -> bottom-left, branchless integer min)
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
      // Floor AIR_CAVITY recess at >= 52 so bilinear sandBackwall cleanly separates CAVITY from AIR_SKY (0)
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
   * Packs `colorTexData` (RGBA8 SRGB, LinearFilter) and `fieldTexData` (RGBA8 Linear, LinearFilter):
   * - `colorTex.rgb`: smooth grain/stratum pigment (undisturbed bgStrata in cavities -> zero fringe bleed!)
   * - `colorTex.a`: 255 strictly on airborne falling LOOSE_SAND (velY > 0.12), 0 elsewhere
   * - `fieldTex.r`: 3x3 Gaussian-smoothed cohesive solid coverage (PACKED + PEBBLE + resting LOOSE_SAND;
   *   0 on airborne falling grains -> zero parallax smudge extrusion + butter-smooth tunnel curves!)
   * - `fieldTex.g`: smooth cavity recess depth (>= 52 in CAVITY, 0 in SKY/SOLID)
   * - `fieldTex.b`: solid thin-wall distance for warm backlit translucency
   * - `fieldTex.a`: compacted PACKED_SAND arch shear stress ONLY (strictly 0 on LOOSE_SAND!)
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

    // Pass 1: build binary cohesive solid mask (excluding airborne falling LOOSE_SAND!) + pack color/G/B/A
    for (let i = 0; i < GRID_CELLS; i++) {
      const o = i << 2;
      const c = ct[i];
      const isFalling = c === CELL_LOOSE_SAND && vy[i] > 0.12;
      const isCohesive = c === CELL_PACKED_SAND || c === CELL_PEBBLE || (c === CELL_LOOSE_SAND && !isFalling);
      coh[i] = isCohesive ? 1 : 0;

      const pId = c === CELL_EMPTY ? bgP[i] : pig[i];
      const shadeByte = c === CELL_EMPTY ? bgS[i] : shd[i];
      const lutOff = ((pId << 8) | shadeByte) * 3;
      cOut[o] = lut[lutOff];
      cOut[o + 1] = lut[lutOff + 1];
      cOut[o + 2] = lut[lutOff + 2];
      // Color alpha = airborne falling sand grain mask ONLY (crisp bright tumbling grains!)
      cOut[o + 3] = isFalling ? 255 : 0;

      // Field G/B/A (strictly zero arch stress on LOOSE_SAND so falling grains never darken!)
      fOut[o + 1] = rec[i];
      fOut[o + 2] = wal[i];
      fOut[o + 3] = c === CELL_PACKED_SAND ? str[i] : 0;
    }

    // Pass 2: 3x3 Gaussian-weighted cohesive solid coverage -> fieldTex.r (eliminates 300x220 staircases!)
    // Border cells take raw binary mask; interior cells take (4*c + 2*(ortho) + 1*(diag)) * 255 / 16
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
        // Map [0..16] -> [0..255] cleanly
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

  /** Counts total grains currently carried inside autonomous digger mandibles. */
  countCarriedGrains(): number {
    let n = 0;
    for (let i = 0; i < this.diggers.length; i++) {
      if (this.diggers[i].carryingPigment >= 0) n++;
    }
    return n;
  }

  /** Total conserved sand mass (grid sand + digger-carried sand). */
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
      diggerCount: this.diggers.length,
      tailingsDeposited: this.tailingsDeposited,
      stepCount: this.stepCount,
    };
  }
}
