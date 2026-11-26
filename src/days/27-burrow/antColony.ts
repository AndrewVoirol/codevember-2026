import {
  AIR_CAVITY,
  AIR_SKY,
  ARCH_SPAN_LIMIT,
  CELL_EMPTY,
  CELL_LOOSE_SAND,
  CELL_PACKED_SAND,
  CELL_PEBBLE,
  COARSE_CELLS,
  COARSE_H,
  COARSE_W,
  DIGGER_COUNT,
  GRID_CELLS,
  GRID_H,
  GRID_W,
  SURFACE_FRAC,
} from './constants';

export type DiggerState =
  | 'patrol_down'
  | 'excavate'
  | 'haul_up'
  | 'deposit'
  | 'rescue_dig'
  | 'backfill'
  | 'greet'
  | 'groom'
  | 'falling'
  | 'buried';

export type WorkerAntRole = 'excavator' | 'hauler' | 'scout' | 'tender';

export interface DiggerAgent {
  id: number;
  role: WorkerAntRole;
  /** Continuous grid X coordinate in [0, GRID_W) */
  x: number;
  /** Continuous grid Y coordinate in [0, GRID_H) */
  y: number;
  /** Slight depth lane in [-0.78, 0.78] inside the glass cavity thickness */
  zLane: number;
  targetZLane: number;
  /** Velocity components for smooth momentum & gravity falls */
  vx: number;
  vy: number;
  /** Facing angle in XY grid plane (radians, 0 = +X, PI/2 = +Y) */
  heading: number;
  /** Smooth dorsal surface-normal vector (points away from floor/wall into tunnel cavity!) */
  upX: number;
  upY: number;
  /** Preferred side wall bias (+1 / -1) to stay cleanly attached inside narrow vertical shafts */
  wallSide: number;
  /** Continuous tripod-gait animation phase */
  walkPhase: number;
  /** Mandible excavation cycle phase */
  mandiblePhase: number;
  /** Independent antennae sensing & tapping phase */
  antennaPhase: number;
  /** Individual scale & speed cadence for natural colony variety */
  bodyScale: number;
  speedMul: number;
  chitinWarmth: number;
  state: DiggerState;
  prevState: DiggerState;
  stateTimer: number;
  greetCooldown: number;
  greetPartnerId: number;
  /** Pigment ID of carried sand grain (-1 if empty-handed) */
  carryingPigment: number;
  carryingShade: number;
  /** Current excavation or crater destination */
  targetX: number;
  targetY: number;
  /** Remembered gallery work coordinates for nestmate recruitment */
  lastWorkX: number;
  lastWorkY: number;
  /** Preferred surface entrance shaft X */
  homeShaftX: number;
  /** Preferred gallery expansion direction (-1 left, +1 right) */
  branchDir: number;
  /** Preferred organic tunnel inclination angle */
  digAngle: number;
  /** Last depth at which this ant excavated a grain */
  lastExcavatedY: number;
  /** Running count of grains excavated by this worker */
  grainsDug: number;
  /** Consecutive ticks without forward displacement during walking states (anti-stall guard) */
  stuckTicks: number;
  /** Cooldown preventing immediate re-entry into excavate after skipping a protected arch ceiling */
  digCooldown: number;
}

/** Minimal interface to the granular world needed by AntColony (avoids circular imports). */
export interface ColonyGridHost {
  readonly cellType: Uint8Array;
  readonly pigmentId: Uint8Array;
  readonly grainShade: Uint8Array;
  readonly bondHealth: Uint8Array;
  readonly faultDelay: Uint8Array;
  readonly velX: Float32Array;
  readonly velY: Float32Array;
  readonly restTicks: Uint16Array;
  readonly airClass: Uint8Array;
  readonly topSolidY: Int16Array;
  readonly overburdenTopY: Int16Array;
  excavatedCells: number;
  tailingsDeposited: number;
  texturesDirty: boolean;
  nextRandom: () => number;
  spawnExcavatedGrain3D?: (
    gx: number,
    gy: number,
    vx: number,
    vy: number,
    pigmentId: number,
    shade: number
  ) => void;
}

export interface ColonyStats {
  antCount: number;
  trappedCount: number;
  greetingCount: number;
  fallingCount: number;
  carryingCount: number;
  solidWallPenetrations: number;
  rescuesTriggered: number;
  nestmateGreetings: number;
}

const INF_DIST = 30000;
/**
 * Maximum chamfer distance from solid sand/pebble for valid surface-contact walking inside cavities.
 * Chamfer units: orthogonal = 2, diagonal = 3 -> 16 ≈ 8 cells from wall/floor (covers the entire
 * interior of every normal tunnel/chamber up to ~16 cells wide without centerline dead zones,
 * while AIR_SKY strictly clamps to the surface crest walkband `<= 6`!).
 */
const MAX_CAVITY_WALK_CHAMFER = 16;
const MAX_SKY_WALK_CHAMFER = 6;
/** Maximum chamfer distance for immediate physical footing before gravity pull triggers */
const MAX_FOOTING_CHAMFER = 14;

// 8-connected neighbor offsets (orthogonal first, then diagonals)
const DIR_X = [1, -1, 0, 0, 1, -1, 1, -1];
const DIR_Y = [0, 0, 1, -1, 1, 1, -1, -1];
const DIR_COST = [2, 2, 2, 2, 3, 3, 3, 3];

/**
 * Stigmergic Worker Ant Colony & Open-Corridor Navigation Engine (Zero-GC):
 * - Computes exact open-path BFS flow fields (`distToSurface`, `distToWork`, `distToRubble`)
 *   strictly through traversable `CELL_EMPTY` footing cells (ZERO wall-phasing through solid sand!).
 * - Detects trapped underground chambers after cave-ins (`distToSurface >= INF_DIST`) and
 *   coordinates rescue digging from inside (with floor backfilling) + rubble clearing from above.
 * - Maintains invisible half-res stigmergic pheromone & user-carve curiosity maps so ants
 *   naturally recruit nestmates via antennae-tap greetings and explore newly carved user tunnels.
 * - Drives surface-hugging 3D orthonormal footing frames (`heading`, `upX/upY`) plus real
 *   gravity tumbling if the floor collapses beneath an ant.
 */
export class AntColony {
  readonly ants: DiggerAgent[] = [];

  // Zero-GC spatial awareness & BFS navigation fields
  readonly emptyWallDist = new Uint8Array(GRID_CELLS);
  readonly distToSurface = new Int16Array(GRID_CELLS);
  readonly distToWork = new Int16Array(GRID_CELLS);
  readonly distToRubble = new Int16Array(GRID_CELLS);

  // Invisible half-resolution stigmergic pheromone & disturbance curiosity maps
  readonly pheromoneDig = new Float32Array(COARSE_CELLS);
  readonly disturbanceMap = new Float32Array(COARSE_CELLS);

  // Reusable flat BFS queue (zero per-step allocations)
  private readonly bfsQueue = new Int32Array(GRID_CELLS);

  private navDirty = true;
  private ticksSinceNavRebuild = 999;

  // Live telemetry counters (for verification tests & subtle non-invasive diagnostics)
  solidWallPenetrations = 0;
  rescuesTriggered = 0;
  nestmateGreetings = 0;

  constructor() {
    const roles: WorkerAntRole[] = [
      'hauler',
      'excavator',
      'excavator',
      'hauler',
      'scout',
      'excavator',
      'tender',
      'hauler',
      'excavator',
      'scout',
      'tender',
      'excavator',
    ];

    for (let i = 0; i < DIGGER_COUNT; i++) {
      const zLane = ((i % 5) - 2) * 0.28 + (i % 2 === 0 ? -0.06 : 0.06);
      this.ants.push({
        id: i,
        role: roles[i % roles.length],
        x: 114,
        y: 130,
        zLane,
        targetZLane: zLane,
        vx: 0,
        vy: 0,
        heading: -Math.PI * 0.5,
        upX: 0,
        upY: 1,
        wallSide: i % 2 === 0 ? -1 : 1,
        walkPhase: i * 1.37,
        mandiblePhase: i * 0.83,
        antennaPhase: i * 2.11,
        bodyScale: 0.91 + (i % 5) * 0.048,
        speedMul: 0.9 + ((i * 7) % 11) * 0.022,
        chitinWarmth: ((i * 37) % 100) / 100,
        state: 'patrol_down',
        prevState: 'patrol_down',
        stateTimer: 0,
        greetCooldown: 30 + i * 17,
        greetPartnerId: -1,
        carryingPigment: -1,
        carryingShade: 128,
        targetX: 114,
        targetY: 96,
        lastWorkX: 114,
        lastWorkY: 96,
        homeShaftX: 114,
        branchDir: i % 2 === 0 ? -1 : 1,
        digAngle: i % 2 === 0 ? -Math.PI * 0.72 : -Math.PI * 0.28,
        lastExcavatedY: 96,
        grainsDug: 0,
        stuckTicks: 0,
        digCooldown: 0,
      });
    }
  }

  /** Marks topology dirty so open-path BFS flow fields rebuild cleanly. */
  markTopologyDirty(): void {
    this.navDirty = true;
  }

  /**
   * Injects subtle localized curiosity/disturbance when the user carves a tunnel,
   * pours sand, or triggers a cave-in — drawing nearby scouts & rubble-clearers organically.
   */
  recordDisturbance(gx: number, gy: number, radiusCells: number, strength = 1.0): void {
    const cx = Math.max(0, Math.min(COARSE_W - 1, Math.round(gx) >> 1));
    const cy = Math.max(0, Math.min(COARSE_H - 1, Math.round(gy) >> 1));
    const r = Math.max(1, Math.ceil(radiusCells * 0.5) + 1);
    const r2 = r * r;
    const x0 = Math.max(0, cx - r);
    const x1 = Math.min(COARSE_W - 1, cx + r);
    const y0 = Math.max(0, cy - r);
    const y1 = Math.min(COARSE_H - 1, cy + r);
    for (let y = y0; y <= y1; y++) {
      const rowOff = y * COARSE_W;
      for (let x = x0; x <= x1; x++) {
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (d2 <= r2) {
          const w = (1 - Math.sqrt(d2) / (r + 0.5)) * strength;
          const idx = rowOff + x;
          const next = this.disturbanceMap[idx] + w;
          this.disturbanceMap[idx] = next > 2.5 ? 2.5 : next;
        }
      }
    }
    this.navDirty = true;
  }

  /** Resets colony fields and seats all 12 worker ants strictly inside open walkable cells. */
  reset(host: ColonyGridHost, starterMouthX: number, hasStarterBurrow: boolean): void {
    this.pheromoneDig.fill(0);
    this.disturbanceMap.fill(0);
    this.solidWallPenetrations = 0;
    this.rescuesTriggered = 0;
    this.nestmateGreetings = 0;
    this.navDirty = true;
    this.ticksSinceNavRebuild = 999;

    // Immediately compute exact open-path distance fields on the fresh grid
    this.rebuildNavigationFields(host);

    const mx = starterMouthX;
    const mouthY = this.findSurfaceY(host, mx + 7);

    // Anchor candidate positions along the organic starter burrow network (snapped strictly to CELL_EMPTY!)
    const starterAnchors: Array<{
      x: number;
      y: number;
      state: DiggerState;
      carrying: number;
      dir: number;
    }> = hasStarterBurrow
      ? [
          // 0: Hauling rich Burnt Sienna grain up the upper main entrance shaft
          { x: mx + 3, y: mouthY - 14, state: 'haul_up', carrying: 3, dir: -1 },
          // 1: Actively excavating the deep right Yellow Ochre / Kaolin gallery tip
          { x: mx + 82, y: mouthY - 90, state: 'excavate', carrying: -1, dir: 1 },
          // 2: Actively excavating the left pebble-lintel granary chamber
          { x: mx - 48, y: mouthY - 70, state: 'excavate', carrying: -1, dir: -1 },
          // 3: Hauling Yellow Ochre grain up from lower central fork
          { x: mx + 20, y: mouthY - 72, state: 'haul_up', carrying: 2, dir: 1 },
          // 4: Walking along the surface crater rim to deposit a Burnt Sienna grain
          { x: mx - 8, y: mouthY + 2, state: 'deposit', carrying: 3, dir: -1 },
          // 5: Descending the upper vestibule toward the left gallery
          { x: mx - 6, y: mouthY - 48, state: 'patrol_down', carrying: -1, dir: -1 },
          // 6: Tending inside the basalt-lintel rounded gathering chamber
          { x: mx + 46, y: mouthY - 62, state: 'patrol_down', carrying: -1, dir: 1 },
          // 7: Climbing up the main junction vestibule carrying deep Burnt Sienna sand
          { x: mx + 18, y: mouthY - 44, state: 'haul_up', carrying: 3, dir: 1 },
          // 8: Deep central fork excavator heading toward deep mineral strata
          { x: mx + 24, y: mouthY - 98, state: 'excavate', carrying: -1, dir: 1 },
          // 9: Surface scout patrolling the right crater flank & open sand dune
          { x: mx + 15, y: this.findSurfaceY(host, mx + 15) + 2, state: 'patrol_down', carrying: -1, dir: 1 },
          // 10: Mid-left gallery worker heading toward left granary chamber
          { x: mx - 28, y: mouthY - 60, state: 'patrol_down', carrying: -1, dir: -1 },
          // 11: Mid-right gallery worker heading past the basalt keystone chamber
          { x: mx + 64, y: mouthY - 72, state: 'patrol_down', carrying: -1, dir: 1 },
        ]
      : Array.from({ length: DIGGER_COUNT }, (_, idx) => {
          const sx = Math.max(24, Math.min(GRID_W - 25, Math.round(52 + (idx / DIGGER_COUNT) * 196)));
          const sy = this.findSurfaceY(host, sx) + 2;
          return {
            x: sx,
            y: sy,
            state: 'patrol_down' as DiggerState,
            carrying: -1,
            dir: idx % 2 === 0 ? -1 : 1,
          };
        });

    for (let i = 0; i < this.ants.length; i++) {
      const a = this.ants[i];
      const st = starterAnchors[i % starterAnchors.length];
      const snapped = this.findNearestWalkableOpenCell(host, st.x, st.y, 18);

      a.x = snapped.x;
      a.y = snapped.y;
      a.vx = 0;
      a.vy = 0;
      a.state = st.state;
      a.prevState = st.state;
      a.stateTimer = i * 3;
      a.greetCooldown = 25 + i * 19;
      a.greetPartnerId = -1;
      a.carryingPigment = st.carrying;
      a.carryingShade = 132 + ((i * 23) % 44);
      a.homeShaftX = mx;
      a.branchDir = st.dir;
      a.digAngle = st.dir > 0 ? -0.45 - (i % 3) * 0.22 : -Math.PI + 0.45 + (i % 3) * 0.22;
      a.lastExcavatedY = snapped.y;
      a.lastWorkX = snapped.x;
      a.lastWorkY = snapped.y;
      a.targetX = st.state === 'deposit' ? Math.max(14, mx - 12) : snapped.x + st.dir * 16;
      a.targetY = st.state === 'deposit' ? mouthY + 2 : Math.max(28, snapped.y - 16);
      a.grainsDug = 0;
      a.stuckTicks = 0;
      a.digCooldown = 0;

      // Initialize crisp surface-hugging dorsal normal right at spawn
      const norm = this.sampleSurfaceNormal(host, a.x, a.y, a.wallSide);
      a.upX = norm.nx;
      a.upY = norm.ny;
      a.heading =
        st.state === 'haul_up'
          ? Math.PI * 0.5
          : st.state === 'deposit'
            ? Math.PI
            : st.dir > 0
              ? -0.35
              : Math.PI + 0.35;

      // Seed subtle starter pheromone trail along existing galleries so traffic feels cohesive immediately
      if (hasStarterBurrow) {
        this.depositPheromone(a.x, a.y, 0.35);
      }
    }
  }

  /**
   * Advances the entire worker ant colony by 1 fixed simulation tick (~1/60s).
   * Strictly preserves sand mass conservation and zero solid-wall penetration.
   */
  step(host: ColonyGridHost): void {
    this.ticksSinceNavRebuild++;

    // Evaporate invisible pheromone & disturbance curiosity maps gently
    for (let i = 0; i < COARSE_CELLS; i++) {
      const p = this.pheromoneDig[i];
      if (p > 0.002) this.pheromoneDig[i] = p * 0.9965;
      else if (p > 0) this.pheromoneDig[i] = 0;

      const d = this.disturbanceMap[i];
      if (d > 0.002) this.disturbanceMap[i] = d * 0.994;
      else if (d > 0) this.disturbanceMap[i] = 0;
    }

    // Rebuild exact open-path BFS flow fields whenever topology changed (throttled cleanly to >= 6 ticks)
    if (this.navDirty && this.ticksSinceNavRebuild >= 6) {
      this.rebuildNavigationFields(host);
    }

    const ct = host.cellType;
    const ants = this.ants;

    // Step each worker ant
    for (let i = 0; i < ants.length; i++) {
      const a = ants[i];
      a.stateTimer++;
      if (a.greetCooldown > 0) a.greetCooldown--;
      if (a.digCooldown > 0) a.digCooldown--;

      // Smooth zLane separation (lets passing ants slip cleanly past each other in 3D!)
      a.zLane += (a.targetZLane - a.zLane) * 0.14;

      // Check current cell occupancy (detect if an avalanche or poured sand just buried this ant!)
      const gx = this.clampGridX(Math.round(a.x));
      const gy = this.clampGridY(Math.round(a.y));
      const curIdx = gy * GRID_W + gx;
      const curCell = ct[curIdx];

      if (curCell !== CELL_EMPTY) {
        if (a.state !== 'buried') {
          a.prevState = a.state;
          a.state = 'buried';
          a.stateTimer = 0;
        }
        this.stepAntBuried(a, host, gx, gy);
        this.updateAntSurfaceFrame(a, host);
        continue;
      }

      // Check footing loss (if floor/wall vanished beneath ant, tumble physically under gravity!)
      const wallD = this.emptyWallDist[curIdx];
      if (a.state !== 'falling' && wallD > MAX_FOOTING_CHAMFER) {
        a.prevState = a.state;
        a.state = 'falling';
        a.stateTimer = 0;
        a.vy = -0.15;
      }

      // Check for natural head-to-head nestmate antennae-tap encounters in open corridors
      if (
        a.greetCooldown === 0 &&
        (a.state === 'patrol_down' || a.state === 'haul_up') &&
        a.stateTimer > 12
      ) {
        this.tryStartNestmateGreeting(a, i);
      }

      switch (a.state) {
        case 'falling':
          this.stepAntFalling(a, host);
          break;
        case 'greet':
          this.stepAntGreet(a);
          break;
        case 'groom':
          this.stepAntGroom(a);
          break;
        case 'excavate':
          this.stepAntExcavate(a, host);
          break;
        case 'haul_up':
          this.stepAntHaulUp(a, host);
          break;
        case 'deposit':
          this.stepAntDeposit(a, host);
          break;
        case 'rescue_dig':
          this.stepAntRescueDig(a, host);
          break;
        case 'backfill':
          this.stepAntBackfill(a, host);
          break;
        case 'buried':
          // Escaped back to CELL_EMPTY!
          a.state = a.carryingPigment >= 0 ? 'haul_up' : 'patrol_down';
          a.stateTimer = 0;
          break;
        case 'patrol_down':
        default:
          this.stepAntPatrol(a, host);
          break;
      }

      // Verify strict zero solid-wall penetration invariant after non-buried movement
      const postX = this.clampGridX(Math.round(a.x));
      const postY = this.clampGridY(Math.round(a.y));
      const postCell = ct[postY * GRID_W + postX];
      if (postCell === CELL_PACKED_SAND || postCell === CELL_PEBBLE) {
        this.solidWallPenetrations++;
        const safe = this.findNearestWalkableOpenCell(host, a.x, a.y, 8);
        a.x = safe.x;
        a.y = safe.y;
      }

      this.updateAntSurfaceFrame(a, host);
    }
  }

  /**
   * Rebuilds:
   * 1. `emptyWallDist` (exact 2-pass Chamfer distance from every CELL_EMPTY to nearest solid surface)
   * 2. `distToSurface` (open-corridor BFS strictly through walkable CELL_EMPTY cells from surface crest)
   * 3. `distToRubble` (open-corridor BFS from subterranean collapsed LOOSE_SAND rubble)
   * 4. `distToWork` (open-corridor BFS from active excavation frontiers, rubble & fresh user carvings)
   */
  rebuildNavigationFields(host: ColonyGridHost): void {
    const ct = host.cellType;
    const air = host.airClass;
    const wDist = this.emptyWallDist;
    const INF_CH = 40;

    // 1. 2-pass Chamfer distance from every EMPTY cell to nearest SOLID cell (PACKED, LOOSE, PEBBLE)
    for (let i = 0; i < GRID_CELLS; i++) {
      wDist[i] = ct[i] === CELL_EMPTY ? INF_CH : 0;
    }
    // Forward pass (bottom-left -> top-right)
    for (let y = 1; y < GRID_H - 1; y++) {
      const row = y * GRID_W;
      const prev = row - GRID_W;
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = row + x;
        let m = wDist[i];
        if (m === 0) continue;
        let t = wDist[i - 1] + 2;
        if (t < m) m = t;
        t = wDist[prev + x] + 2;
        if (t < m) m = t;
        t = wDist[prev + x - 1] + 3;
        if (t < m) m = t;
        t = wDist[prev + x + 1] + 3;
        if (t < m) m = t;
        wDist[i] = m;
      }
    }
    // Backward pass (top-right -> bottom-left)
    for (let y = GRID_H - 2; y >= 1; y--) {
      const row = y * GRID_W;
      const next = row + GRID_W;
      for (let x = GRID_W - 2; x >= 1; x--) {
        const i = row + x;
        let m = wDist[i];
        if (m === 0) continue;
        let t = wDist[i + 1] + 2;
        if (t < m) m = t;
        t = wDist[next + x] + 2;
        if (t < m) m = t;
        t = wDist[next + x + 1] + 3;
        if (t < m) m = t;
        t = wDist[next + x - 1] + 3;
        if (t < m) m = t;
        wDist[i] = m;
      }
    }

    // 2. BFS #1: `distToSurface` strictly through walkable CELL_EMPTY cells (wallDist <= MAX_WALK_CHAMFER)
    const dSurf = this.distToSurface;
    dSurf.fill(INF_DIST);
    const q = this.bfsQueue;
    let qHead = 0;
    let qTail = 0;

    // Seed surface walkband: open AIR_SKY cells directly hugging the top sand crest (wDist <= 6)
    for (let x = 4; x < GRID_W - 4; x++) {
      const topY = host.topSolidY[x];
      const yMin = Math.max(12, topY + 1);
      const yMax = Math.min(GRID_H - 4, topY + 4);
      for (let y = yMin; y <= yMax; y++) {
        const idx = y * GRID_W + x;
        if (ct[idx] === CELL_EMPTY && air[idx] === AIR_SKY && wDist[idx] <= 6) {
          if (dSurf[idx] === INF_DIST) {
            dSurf[idx] = 0;
            q[qTail++] = idx;
          }
        }
      }
    }

    // Expand strictly through walkable open tunnel/floor/wall cells (NEVER through solid sand!)
    while (qHead < qTail) {
      const cur = q[qHead++];
      const curD = dSurf[cur];
      const cx = cur % GRID_W;
      const cy = (cur - cx) / GRID_W;

      for (let k = 0; k < 8; k++) {
        const dx = DIR_X[k];
        const dy = DIR_Y[k];
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 2 || nx >= GRID_W - 2 || ny < 2 || ny >= GRID_H - 3) continue;
        const nIdx = ny * GRID_W + nx;
        if (ct[nIdx] !== CELL_EMPTY) continue;
        const maxWalk = air[nIdx] === AIR_SKY ? MAX_SKY_WALK_CHAMFER : MAX_CAVITY_WALK_CHAMFER;
        if (wDist[nIdx] > maxWalk) continue;

        // Prevent squeezing diagonally between two solid sand corners
        if (k >= 4) {
          if (ct[cy * GRID_W + nx] !== CELL_EMPTY || ct[ny * GRID_W + cx] !== CELL_EMPTY) continue;
        }

        const nextD = curD + DIR_COST[k];
        if (nextD < dSurf[nIdx]) {
          dSurf[nIdx] = nextD;
          q[qTail++] = nIdx;
        }
      }
    }

    // 3. BFS #2: `distToRubble` (loose tumbled sand inside subterranean cavities needing clearing!)
    const dRub = this.distToRubble;
    dRub.fill(INF_DIST);
    qHead = 0;
    qTail = 0;

    // 4. BFS #3: `distToWork` (rubble + user-carved disturbances + deep subterranean dig frontiers!)
    const dWork = this.distToWork;
    dWork.fill(INF_DIST);
    let wHead = 0;
    let wTail = 0;

    // Scan cavity cells adjacent to solid sand for work & rubble seeds
    for (let y = 14; y < GRID_H - 18; y++) {
      const row = y * GRID_W;
      const cyCoarse = (y >> 1) * COARSE_W;
      for (let x = 6; x < GRID_W - 6; x++) {
        const idx = row + x;
        if (ct[idx] !== CELL_EMPTY || wDist[idx] > 4) continue;
        if (dSurf[idx] >= INF_DIST) continue; // Only seed surface-connected work sites here

        const isCavity = air[idx] === AIR_CAVITY;
        const topY = host.topSolidY[x] || Math.round(GRID_H * SURFACE_FRAC);
        const depth = topY - y;

        // Check immediate 4-neighbors for loose tumbled rubble or diggable deep packed sand
        let adjLoose = false;
        let adjPacked = false;
        const nL = ct[idx - 1];
        const nR = ct[idx + 1];
        const nD = ct[idx - GRID_W];
        const nU = ct[idx + GRID_W];

        if (
          nL === CELL_LOOSE_SAND ||
          nR === CELL_LOOSE_SAND ||
          nD === CELL_LOOSE_SAND ||
          nU === CELL_LOOSE_SAND
        ) {
          adjLoose = true;
        }
        if (
          nL === CELL_PACKED_SAND ||
          nR === CELL_PACKED_SAND ||
          nD === CELL_PACKED_SAND ||
          nU === CELL_PACKED_SAND
        ) {
          adjPacked = true;
        }

        // Priority A: Collapsed rubble inside subterranean tunnels
        if (isCavity && adjLoose && depth > 8) {
          dRub[idx] = 0;
          q[qTail++] = idx;
          dWork[idx] = 0;
        }

        // Priority B: Freshly carved user chambers or deep gallery frontiers
        const disturb = this.disturbanceMap[cyCoarse + (x >> 1)];
        if (adjPacked && ((isCavity && depth > 28) || (disturb > 0.25 && depth > 12))) {
          // Deeper frontier tips and newly disturbed zones start with lower seed distance (higher pull!)
          const seedBias =
            disturb > 0.25 ? 2 : depth > 72 ? 4 : depth > 46 ? 10 : 18;
          if (seedBias < dWork[idx]) {
            dWork[idx] = seedBias;
          }
        }
      }
    }

    // Run `distToRubble` BFS
    while (qHead < qTail) {
      const cur = q[qHead++];
      const curD = dRub[cur];
      if (curD > 260) continue;
      const cx = cur % GRID_W;
      const cy = (cur - cx) / GRID_W;

      for (let k = 0; k < 8; k++) {
        const nx = cx + DIR_X[k];
        const ny = cy + DIR_Y[k];
        if (nx < 2 || nx >= GRID_W - 2 || ny < 2 || ny >= GRID_H - 3) continue;
        const nIdx = ny * GRID_W + nx;
        const maxWalk = air[nIdx] === AIR_SKY ? MAX_SKY_WALK_CHAMFER : MAX_CAVITY_WALK_CHAMFER;
        if (ct[nIdx] !== CELL_EMPTY || wDist[nIdx] > maxWalk) continue;
        if (k >= 4 && (ct[cy * GRID_W + nx] !== CELL_EMPTY || ct[ny * GRID_W + cx] !== CELL_EMPTY)) {
          continue;
        }
        const nextD = curD + DIR_COST[k];
        if (nextD < dRub[nIdx]) {
          dRub[nIdx] = nextD;
          q[qTail++] = nIdx;
        }
      }
    }

    // Gather all seeded cells into queue for `distToWork` BFS
    for (let i = 0; i < GRID_CELLS; i++) {
      if (dWork[i] < INF_DIST) {
        q[wTail++] = i;
      }
    }

    // Fallback: if no underground cavity exists yet (clean undisturbed bed), seed surface pioneering spots!
    if (wTail === 0) {
      for (let x = 50; x < GRID_W - 50; x += 48) {
        const sy = Math.min(GRID_H - 4, (host.topSolidY[x] || 170) + 1);
        const idx = sy * GRID_W + x;
        if (ct[idx] === CELL_EMPTY) {
          dWork[idx] = 0;
          q[wTail++] = idx;
        }
      }
    }

    while (wHead < wTail) {
      const cur = q[wHead++];
      const curD = dWork[cur];
      const cx = cur % GRID_W;
      const cy = (cur - cx) / GRID_W;

      for (let k = 0; k < 8; k++) {
        const nx = cx + DIR_X[k];
        const ny = cy + DIR_Y[k];
        if (nx < 2 || nx >= GRID_W - 2 || ny < 2 || ny >= GRID_H - 3) continue;
        const nIdx = ny * GRID_W + nx;
        const maxWalk = air[nIdx] === AIR_SKY ? MAX_SKY_WALK_CHAMFER : MAX_CAVITY_WALK_CHAMFER;
        if (ct[nIdx] !== CELL_EMPTY || wDist[nIdx] > maxWalk) continue;
        if (k >= 4 && (ct[cy * GRID_W + nx] !== CELL_EMPTY || ct[ny * GRID_W + cx] !== CELL_EMPTY)) {
          continue;
        }
        const nextD = curD + DIR_COST[k];
        if (nextD < dWork[nIdx]) {
          dWork[nIdx] = nextD;
          q[wTail++] = nIdx;
        }
      }
    }

    this.navDirty = false;
    this.ticksSinceNavRebuild = 0;
  }

  /**
   * Patrols strictly along open corridors (`distToWork` + `distToRubble` + invisible `pheromoneDig`)
   * toward an active dig frontier, rubble clearing site, or freshly user-carved chamber.
   */
  private stepAntPatrol(a: DiggerAgent, host: ColonyGridHost): void {
    const gx = this.clampGridX(Math.round(a.x));
    const gy = this.clampGridY(Math.round(a.y));
    const curIdx = gy * GRID_W + gx;

    // Detect if a cave-in just sealed us off from the surface (throttled BFS check!)
    if (this.distToSurface[curIdx] >= INF_DIST) {
      if (this.navDirty && this.ticksSinceNavRebuild >= 6) {
        this.rebuildNavigationFields(host);
      }
      if (this.distToSurface[curIdx] >= INF_DIST) {
        a.state = 'rescue_dig';
        a.stateTimer = 0;
        this.rescuesTriggered++;
        return;
      }
    }

    // Occasional brief naturalistic grooming pause in quiet chambers
    if (
      a.stateTimer > 120 &&
      host.airClass[curIdx] === AIR_CAVITY &&
      host.nextRandom() < 0.0045
    ) {
      a.prevState = 'patrol_down';
      a.state = 'groom';
      a.stateTimer = 0;
      return;
    }

    const topY = host.topSolidY[gx] || Math.round(GRID_H * SURFACE_FRAC);
    const depth = topY - gy;

    // Check if we reached collapsed subterranean rubble -> prioritize clearing it immediately!
    if (host.airClass[curIdx] === AIR_CAVITY && depth > 6 && this.hasAdjacentCell(host, gx, gy, CELL_LOOSE_SAND)) {
      a.state = 'excavate';
      a.stateTimer = 0;
      return;
    }

    // Check if we arrived at a valid excavation frontier (either deep gallery tip or fresh user carve or virgin surface)
    const adjPacked = this.hasAdjacentCell(host, gx, gy, CELL_PACKED_SAND);
    const curWorkDist = this.distToWork[curIdx];
    const isFreshDisturb = this.sampleDisturbance(a.x, a.y) > 0.28;
    const isVirginSurfacePioneer = curWorkDist <= 2 && depth <= 4 && host.airClass[curIdx] === AIR_SKY;

    if (
      a.digCooldown === 0 &&
      adjPacked &&
      (isVirginSurfacePioneer ||
        (host.airClass[curIdx] === AIR_CAVITY &&
          (curWorkDist <= 8 || (depth > 30 && a.stateTimer > 55) || (isFreshDisturb && depth > 14))))
    ) {
      // Scouts sometimes wander a bit further down before digging so they explore full new chambers
      if (a.role !== 'scout' || a.stateTimer > 65 || curWorkDist <= 4) {
        a.state = 'excavate';
        a.stateTimer = 0;
        return;
      }
    }

    // Choose open-corridor steering direction strictly across walkable CELL_EMPTY neighbors!
    const followRubble = this.distToRubble[curIdx] < 160;
    const field = followRubble ? this.distToRubble : this.distToWork;
    const speed = (followRubble ? 0.62 : 0.54) * a.speedMul;

    const moved = this.stepAlongOpenField(
      a,
      host,
      field,
      speed,
      true, // blend subtle stigmergic dig pheromone + role directional preference
      false
    );

    if (!moved && adjPacked) {
      a.state = 'excavate';
      a.stateTimer = 0;
    }
  }

  /**
   * Braces at an adjacent solid face, animates mandibles biting, and cleanly excavates
   * exactly ONE directly adjacent mineral sand grain (`dist <= 2.3`, NEVER a river pebble!).
   */
  private stepAntExcavate(a: DiggerAgent, host: ColonyGridHost): void {
    a.mandiblePhase += 0.48;
    a.antennaPhase += 0.36;
    a.walkPhase += 0.14;

    if (a.stateTimer < 11) {
      // Brief bracing + mandible bite windup
      return;
    }

    const ct = host.cellType;
    const cx = this.clampGridX(Math.round(a.x));
    const cy = this.clampGridY(Math.round(a.y));
    const curIdx = cy * GRID_W + cx;

    // Check if trapped underground while digging
    const isTrapped = this.distToSurface[curIdx] >= INF_DIST;

    let bestIdx = -1;
    let bestX = cx;
    let bestY = cy;
    let bestScore = -1e9;

    // Evaluate ONLY directly adjacent sand cells within reach (r <= 2 -> NO telekinetic digging!)
    for (let dy = -2; dy <= 2; dy++) {
      const sy = cy + dy;
      if (sy < 10 || sy >= GRID_H - 12) continue;
      for (let dx = -2; dx <= 2; dx++) {
        if (dx === 0 && dy === 0) continue;
        const d2 = dx * dx + dy * dy;
        if (d2 > 5) continue; // Strictly within ~2.24 cells!
        const sx = cx + dx;
        if (sx < 4 || sx >= GRID_W - 4) continue;

        const idx = sy * GRID_W + sx;
        const c = ct[idx];
        if ((c !== CELL_PACKED_SAND && c !== CELL_LOOSE_SAND) || host.faultDelay[idx] !== 0) {
          continue;
        }

        const topY = host.topSolidY[sx] || Math.round(GRID_H * SURFACE_FRAC);
        const depth = topY - sy;

        let score = 0;

        // 1. Strongly prioritize clearing collapsed loose rubble inside cavities!
        if (c === CELL_LOOSE_SAND && host.airClass[curIdx] === AIR_CAVITY) {
          score += 5.5;
        }

        // 2. Architectural arch-preservation instinct:
        //    Avoid biting ceilings directly above an already wide open span (prevents reckless self-collapse!)
        if (dy > 0 && c === CELL_PACKED_SAND) {
          const spanBelow = this.measureLocalEmptySpan(ct, sx, sy - 1);
          if (spanBelow >= ARCH_SPAN_LIMIT - 5) {
            score -= 8.0; // Respect granular arch limits!
          }
        }

        // 3. Pebble-lintel instinct: love digging right underneath or curving around hard river pebbles!
        let nearPebbleAbove = false;
        for (let py = 1; py <= 4 && sy + py < GRID_H - 2; py++) {
          if (ct[(sy + py) * GRID_W + sx] === CELL_PEBBLE) {
            nearPebbleAbove = true;
            break;
          }
        }
        if (nearPebbleAbove) score += 2.2;

        // 4. Organic angled gallery progression (follow personal digAngle & branchDir)
        const ang = Math.atan2(dy, dx);
        const align = Math.cos(ang - a.digAngle);
        score += align * 1.9 + (dx * a.branchDir > 0 ? 0.65 : 0);

        // 5. Gently prefer digging deeper into rich lower Ochre/Sienna/Verdigris bands,
        //    and keep tunnel width cozy (~7-11 cells) unless near a pebble chamber!
        if (depth > 18) score += 0.9;
        if (dy < 0 && sy > 28) score += 0.55;
        if (sy < 26 && dy < 0) score -= 2.8; // Don't chew into wooden bottom frame

        // 6. Prefer immediate contact cell (d2 <= 2) over slightly further reach
        score -= Math.sqrt(d2) * 0.45;
        score += (host.nextRandom() - 0.5) * 0.7;

        if (score > bestScore) {
          bestScore = score;
          bestIdx = idx;
          bestX = sx;
          bestY = sy;
        }
      }
    }

    if (bestIdx !== -1 && bestScore > -4.5) {
      // Bite exactly 1 grain into mandibles -> 100% exact sand mass conservation!
      a.carryingPigment = host.pigmentId[bestIdx];
      a.carryingShade = host.grainShade[bestIdx];
      ct[bestIdx] = CELL_EMPTY;
      host.bondHealth[bestIdx] = 0;
      host.velX[bestIdx] = 0;
      host.velY[bestIdx] = 0;
      host.restTicks[bestIdx] = 0;
      host.excavatedCells++;
      a.grainsDug++;
      a.lastExcavatedY = bestY;
      a.lastWorkX = bestX;
      a.lastWorkY = bestY;
      a.heading = Math.atan2(bestY - a.y, bestX - a.x);

      // Slowly curve digAngle organically so ant-dug galleries meander naturally like real ant tunnels
      a.digAngle += (host.nextRandom() - 0.5) * 0.28;
      // Keep gallery angle gently sloped downward/sideways (avoid flat horizontal collapse ceilings!)
      const minAng = a.branchDir > 0 ? -1.25 : -Math.PI + 0.28;
      const maxAng = a.branchDir > 0 ? -0.28 : -Math.PI + 1.25;
      if (a.digAngle < minAng) a.digAngle = minAng;
      if (a.digAngle > maxAng) a.digAngle = maxAng;
      if (host.nextRandom() < 0.08) {
        a.branchDir = -a.branchDir;
        a.digAngle = a.branchDir > 0 ? -0.62 : -Math.PI + 0.62;
      }

      // Incremental local BFS update so freshly dug cell immediately inherits valid surface distance
      this.emptyWallDist[bestIdx] = 2;
      let minNeighborSurf = INF_DIST;
      for (let k = 0; k < 4; k++) {
        const nx = bestX + DIR_X[k];
        const ny = bestY + DIR_Y[k];
        if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
          const nd = this.distToSurface[ny * GRID_W + nx];
          if (nd + 2 < minNeighborSurf) minNeighborSurf = nd + 2;
        }
      }
      if (minNeighborSurf < INF_DIST) {
        this.distToSurface[bestIdx] = minNeighborSurf;
      }

      // Deposit faint invisible pheromone at active dig site to subtly recruit nearby nestmates
      this.depositPheromone(a.x, a.y, 0.9);
      host.texturesDirty = true;
      this.navDirty = true;

      if (isTrapped) {
        a.state = 'backfill';
        a.stateTimer = 0;
      } else {
        a.state = 'haul_up';
        a.stateTimer = 0;
      }
    } else {
      // Nothing safely diggable within immediate reach (e.g. protected arch ceiling or pebble) ->
      // walk along open tunnels for >= 38 ticks before probing a fresh work face!
      a.digCooldown = 38;
      a.branchDir = -a.branchDir;
      a.digAngle = a.branchDir > 0 ? -0.55 : -Math.PI + 0.55;
      a.state = isTrapped ? 'rescue_dig' : 'patrol_down';
      a.stateTimer = 0;
    }
  }

  /**
   * Hauls the carried mineral grain strictly along open walkable tunnels (`distToSurface` BFS!)
   * all the way to the nearest open surface entrance — ZERO solid-wall phasing ever!
   */
  private stepAntHaulUp(a: DiggerAgent, host: ColonyGridHost): void {
    const gx = this.clampGridX(Math.round(a.x));
    const gy = this.clampGridY(Math.round(a.y));
    const curIdx = gy * GRID_W + gx;
    const dSurf = this.distToSurface[curIdx];

    // If a cave-in just sealed off our escape shaft while we were hauling up -> verify throttled BFS!
    if (dSurf >= INF_DIST) {
      if (this.navDirty && this.ticksSinceNavRebuild >= 6) {
        this.rebuildNavigationFields(host);
      }
      if (this.distToSurface[curIdx] >= INF_DIST) {
        a.state = 'backfill'; // Drop our current grain on the chamber floor first, then dig out!
        a.stateTimer = 0;
        this.rescuesTriggered++;
        return;
      }
    }

    // Reached the open surface crest! Transition to crater-rim deposit
    if (dSurf <= 3 || (host.airClass[curIdx] === AIR_SKY && gy >= (host.topSolidY[gx] || 150))) {
      // Walk outward onto the outer crater flank (well clear of open shaft throat!) before dropping the grain
      const side = (a.id + host.tailingsDeposited) % 2 === 0 ? -1 : 1;
      const rimDist = 11 + ((a.id * 3 + host.tailingsDeposited * 2) % 14);
      a.homeShaftX = gx;
      a.targetX = Math.max(12, Math.min(GRID_W - 13, gx + side * rimDist));
      a.targetY = Math.min(GRID_H - 6, this.findSurfaceY(host, Math.round(a.targetX)) + 2);
      a.state = 'deposit';
      a.stateTimer = 0;
      return;
    }

    // Deposit faint invisible pheromone trace along open corridor so empty nestmates find our dig site
    if ((a.stateTimer & 3) === 0) {
      this.depositPheromone(a.x, a.y, 0.24);
    }

    // Follow exact open-corridor `distToSurface` gradient strictly through walkable CELL_EMPTY cells!
    const climbSpeed = 0.62 * a.speedMul;
    this.stepAlongOpenField(a, host, this.distToSurface, climbSpeed, false, true);
  }

  /**
   * Walks across the open surface sand crest to the outer flank of the volcanic crater rim
   * (avoiding dropping grains right back down the open shaft!) and releases the carried mineral grain.
   */
  private stepAntDeposit(a: DiggerAgent, host: ColonyGridHost): void {
    const ct = host.cellType;
    const dx = a.targetX - a.x;
    const gx = this.clampGridX(Math.round(a.x));
    const localTop = this.findSurfaceY(host, gx);
    const targetSurfY = Math.min(GRID_H - 5, localTop + 1.4);

    if (Math.abs(dx) > 2.2 && a.stateTimer < 60) {
      const stepX = Math.sign(dx) * 0.56 * a.speedMul;
      const nextX = this.clampGridX(a.x + stepX);
      const nextColTop = this.findSurfaceY(host, Math.round(nextX));
      const nextY = Math.min(GRID_H - 5, Math.max(a.y - 1.2, nextColTop + 1.4));
      const nIdx = this.clampGridY(Math.round(nextY)) * GRID_W + this.clampGridX(Math.round(nextX));

      if (ct[nIdx] === CELL_EMPTY) {
        const vx = nextX - a.x;
        const vy = nextY - a.y;
        a.x = nextX;
        a.y = nextY;
        a.vx = vx;
        a.vy = vy;
        a.heading = this.lerpAngle(a.heading, Math.atan2(vy, vx), 0.3);
        a.walkPhase += 0.34 * a.speedMul;
        a.antennaPhase += 0.22;
        return;
      }
    }

    // Verify we aren't dropping into or onto the inner lip of a vertical shaft mouth!
    let dropX = this.clampGridX(Math.round(a.x));
    if (Math.abs(dropX - a.homeShaftX) < 9) {
      const nudge = dropX >= a.homeShaftX ? 10 : -10;
      dropX = this.clampGridX(a.homeShaftX + nudge);
    }

    // Release the carried colored mineral sand grain onto the surface crater mound (100% mass conservation!)
    if (a.carryingPigment >= 0) {
      let dropY = Math.min(GRID_H - 3, Math.max(4, Math.round(targetSurfY)));
      while (dropY < GRID_H - 2 && ct[dropY * GRID_W + dropX] !== CELL_EMPTY) {
        dropY++;
      }
      if (ct[dropY * GRID_W + dropX] === CELL_EMPTY) {
        const idx = dropY * GRID_W + dropX;
        const outDir = dropX >= a.homeShaftX ? 1 : -1;
        ct[idx] = CELL_LOOSE_SAND;
        host.pigmentId[idx] = a.carryingPigment;
        host.grainShade[idx] = a.carryingShade;
        host.bondHealth[idx] = 0;
        host.faultDelay[idx] = 0;
        host.velX[idx] = outDir * 0.58;
        host.velY[idx] = 0.34;
        host.restTicks[idx] = 0;
        host.spawnExcavatedGrain3D?.(
          dropX,
          dropY,
          host.velX[idx],
          -0.36,
          a.carryingPigment,
          a.carryingShade
        );
        a.carryingPigment = -1;
        host.tailingsDeposited++;
        host.texturesDirty = true;
        this.navDirty = true;
        // Step this ant (and any nestmate standing on the exact same rim cell!) cleanly onto the newly dropped grain
        for (let j = 0; j < this.ants.length; j++) {
          const other = this.ants[j];
          if (Math.round(other.x) === dropX && Math.round(other.y) === dropY) {
            other.y = Math.min(GRID_H - 4, dropY + 1.25);
          }
        }
      }
    }

    a.state = 'patrol_down';
    a.stateTimer = 0;
  }

  /**
   * Trapped-Chamber Rescue Awareness:
   * When an avalanche seals off a burrow section (`distToSurface >= INF_DIST`), trapped ants
   * navigate strictly inside their open chamber toward the highest/closest solid plug cell
   * separating them from open surface-connected space, and excavate upward grain by grain!
   */
  private stepAntRescueDig(a: DiggerAgent, host: ColonyGridHost): void {
    const ct = host.cellType;
    const gx = this.clampGridX(Math.round(a.x));
    const gy = this.clampGridY(Math.round(a.y));
    const curIdx = gy * GRID_W + gx;

    // Did we (or a surface rescue crew) break through back to the surface?!
    if (this.distToSurface[curIdx] < INF_DIST) {
      a.state = a.carryingPigment >= 0 ? 'haul_up' : 'patrol_down';
      a.stateTimer = 0;
      return;
    }

    // If holding a grain while trapped, backfill it onto a lower floor pocket first
    if (a.carryingPigment >= 0) {
      a.state = 'backfill';
      a.stateTimer = 0;
      return;
    }

    // Look for an upward/sideways sand plug grain right next to us to dig upward toward freedom!
    const plugIdx = this.findBestRescuePlugGrain(host, gx, gy);
    if (plugIdx !== -1 && a.stateTimer >= 8) {
      const px = plugIdx % GRID_W;
      const py = (plugIdx - px) / GRID_W;
      a.carryingPigment = host.pigmentId[plugIdx];
      a.carryingShade = host.grainShade[plugIdx];
      ct[plugIdx] = CELL_EMPTY;
      host.bondHealth[plugIdx] = 0;
      host.velX[plugIdx] = 0;
      host.velY[plugIdx] = 0;
      host.restTicks[plugIdx] = 0;
      host.excavatedCells++;
      a.lastExcavatedY = py;
      a.heading = Math.atan2(py - a.y, px - a.x);
      a.mandiblePhase += 0.6;
      host.texturesDirty = true;
      this.navDirty = true;
      a.state = 'backfill';
      a.stateTimer = 0;
      return;
    }

    // Otherwise climb strictly through open empty cells inside the trapped cavity toward the highest point!
    let bestNx = gx;
    let bestNy = gy;
    let bestScore = -1e9;

    for (let k = 0; k < 8; k++) {
      const nx = gx + DIR_X[k];
      const ny = gy + DIR_Y[k];
      if (nx < 3 || nx >= GRID_W - 3 || ny < 3 || ny >= GRID_H - 4) continue;
      const nIdx = ny * GRID_W + nx;
      if (ct[nIdx] !== CELL_EMPTY || this.emptyWallDist[nIdx] > MAX_CAVITY_WALK_CHAMFER) continue;
      if (k >= 4 && (ct[gy * GRID_W + nx] !== CELL_EMPTY || ct[ny * GRID_W + gx] !== CELL_EMPTY)) {
        continue;
      }
      // Prefer climbing upward toward surface + steering slightly toward homeShaftX
      const towardShaft = -Math.abs(nx - a.homeShaftX) * 0.04;
      const score = ny * 1.4 + towardShaft + (host.nextRandom() - 0.5) * 0.4;
      if (score > bestScore) {
        bestScore = score;
        bestNx = nx;
        bestNy = ny;
      }
    }

    if (bestNx !== gx || bestNy !== gy) {
      this.moveAntSmoothlyTowardCell(a, host, bestNx, bestNy, 0.58 * a.speedMul);
    } else {
      a.mandiblePhase += 0.45;
      a.antennaPhase += 0.42;
    }
  }

  /**
   * Trapped ants carry excavated rescue-plug grains a few steps downward into a chamber pocket
   * and drop them cleanly on the floor (100% grain conservation) before resuming rescue digging.
   */
  private stepAntBackfill(a: DiggerAgent, host: ColonyGridHost): void {
    const ct = host.cellType;
    const gx = this.clampGridX(Math.round(a.x));
    const gy = this.clampGridY(Math.round(a.y));
    const curIdx = gy * GRID_W + gx;

    // If rescue broke through while carrying -> haul all the way up to the surface!
    if (this.distToSurface[curIdx] < INF_DIST) {
      a.state = 'haul_up';
      a.stateTimer = 0;
      return;
    }

    // Move downward/sideways ~12 ticks away from the rescue ceiling before setting grain down
    if (a.stateTimer < 14) {
      let bestNx = gx;
      let bestNy = gy;
      let bestScore = -1e9;
      for (let k = 0; k < 8; k++) {
        const nx = gx + DIR_X[k];
        const ny = gy + DIR_Y[k];
        if (nx < 3 || nx >= GRID_W - 3 || ny < 3 || ny >= GRID_H - 4) continue;
        const nIdx = ny * GRID_W + nx;
        if (ct[nIdx] !== CELL_EMPTY || this.emptyWallDist[nIdx] > MAX_CAVITY_WALK_CHAMFER) continue;
        const score = -ny * 1.2 + (host.nextRandom() - 0.5) * 0.8;
        if (score > bestScore) {
          bestScore = score;
          bestNx = nx;
          bestNy = ny;
        }
      }
      if (bestNx !== gx || bestNy !== gy) {
        this.moveAntSmoothlyTowardCell(a, host, bestNx, bestNy, 0.55 * a.speedMul);
        return;
      }
    }

    // Deposit grain onto an empty cell right beneath/beside us (NOT on top of our own cell!)
    if (a.carryingPigment >= 0) {
      const candidates: Array<[number, number]> = [
        [gx, gy - 1],
        [gx - 1, gy - 1],
        [gx + 1, gy - 1],
        [gx - 1, gy],
        [gx + 1, gy],
      ];
      for (let c = 0; c < candidates.length; c++) {
        const dx = this.clampGridX(candidates[c][0]);
        const dy = this.clampGridY(candidates[c][1]);
        const dIdx = dy * GRID_W + dx;
        if (ct[dIdx] === CELL_EMPTY) {
          ct[dIdx] = CELL_LOOSE_SAND;
          host.pigmentId[dIdx] = a.carryingPigment;
          host.grainShade[dIdx] = a.carryingShade;
          host.bondHealth[dIdx] = 0;
          host.faultDelay[dIdx] = 0;
          host.velX[dIdx] = 0;
          host.velY[dIdx] = 0.18;
          host.restTicks[dIdx] = 0;
          a.carryingPigment = -1;
          host.texturesDirty = true;
          this.navDirty = true;
          for (let j = 0; j < this.ants.length; j++) {
            const other = this.ants[j];
            if (Math.round(other.x) === dx && Math.round(other.y) === dy) {
              other.y = Math.min(GRID_H - 4, dy + 1.25);
            }
          }
          break;
        }
      }
    }

    a.state = 'rescue_dig';
    a.stateTimer = 0;
  }

  /** Brief head-to-head nestmate antennae-tap greeting (trophallaxis / recruitment). */
  private stepAntGreet(a: DiggerAgent): void {
    a.antennaPhase += 0.92; // Rapid animated antennae tapping!
    a.mandiblePhase += 0.25;
    a.walkPhase += 0.08;

    const partner =
      a.greetPartnerId >= 0 && a.greetPartnerId < this.ants.length
        ? this.ants[a.greetPartnerId]
        : null;
    if (partner) {
      const ang = Math.atan2(partner.y - a.y, partner.x - a.x);
      a.heading = this.lerpAngle(a.heading, ang, 0.25);
    }

    if (a.stateTimer >= 18) {
      a.greetPartnerId = -1;
      a.greetCooldown = 210 + (a.id * 29) % 90;
      a.state = a.prevState === 'greet' ? (a.carryingPigment >= 0 ? 'haul_up' : 'patrol_down') : a.prevState;
      a.stateTimer = 0;
    }
  }

  /** Occasional brief naturalistic pause to sweep antennae & groom at a junction. */
  private stepAntGroom(a: DiggerAgent): void {
    a.antennaPhase += 0.62;
    a.mandiblePhase += 0.18;
    a.walkPhase += 0.05;
    if (a.stateTimer >= 22) {
      a.state = a.carryingPigment >= 0 ? 'haul_up' : 'patrol_down';
      a.stateTimer = 0;
    }
  }

  /**
   * Physical gravity tumble when the sand floor/wall is carved or collapses away beneath an ant!
   * Scrambling legs until catching the next solid floor or wall.
   */
  private stepAntFalling(a: DiggerAgent, host: ColonyGridHost): void {
    const ct = host.cellType;
    a.walkPhase += 0.85; // Rapid scrambling legs!
    a.antennaPhase += 0.55;
    a.vy = Math.max(-2.4, a.vy - 0.14);
    a.vx *= 0.92;

    const steps = Math.max(1, Math.ceil(Math.abs(a.vy)));
    const stepX = a.vx / steps;
    const stepY = a.vy / steps;
    let landed = false;

    for (let s = 0; s < steps; s++) {
      const nx = Math.max(4, Math.min(GRID_W - 5, a.x + stepX));
      const ny = Math.max(4, Math.min(GRID_H - 5, a.y + stepY));
      const ix = Math.round(nx);
      const iy = Math.round(ny);
      const idx = iy * GRID_W + ix;

      if (ct[idx] !== CELL_EMPTY) {
        landed = true;
        break;
      }
      a.x = nx;
      a.y = ny;

      // Did we land right above a solid floor or catch a passing tunnel wall?
      if (iy <= 4 || ct[(iy - 1) * GRID_W + ix] !== CELL_EMPTY || this.emptyWallDist[idx] <= 4) {
        landed = true;
        break;
      }
    }

    if (landed) {
      a.vx = 0;
      a.vy = 0;
      a.state = a.carryingPigment >= 0 ? 'haul_up' : 'patrol_down';
      a.stateTimer = 0;
    }
  }

  /**
   * Buried escape: if an avalanche or poured sand lands directly on top of an ant,
   * it wriggles out into the nearest adjacent empty cell — or if completely entombed,
   * clears 1 loose grain into its mandibles (strictly conserving mass!) to open breathing room.
   */
  private stepAntBuried(a: DiggerAgent, host: ColonyGridHost, gx: number, gy: number): void {
    const ct = host.cellType;
    a.walkPhase += 0.65;
    a.mandiblePhase += 0.65;

    // 1. First look for any open empty cell within radius 4 to wriggle straight into!
    let bestX = -1;
    let bestY = -1;
    let bestD2 = 1e9;
    for (let dy = -4; dy <= 4; dy++) {
      const ny = gy + dy;
      if (ny < 3 || ny >= GRID_H - 3) continue;
      for (let dx = -4; dx <= 4; dx++) {
        const nx = gx + dx;
        if (nx < 3 || nx >= GRID_W - 3) continue;
        if (ct[ny * GRID_W + nx] === CELL_EMPTY) {
          // Prefer escaping upward into open cavity/air
          const d2 = dx * dx + dy * dy - dy * 0.6;
          if (d2 < bestD2) {
            bestD2 = d2;
            bestX = nx;
            bestY = ny;
          }
        }
      }
    }

    if (bestX !== -1) {
      a.x += (bestX - a.x) * 0.45;
      a.y += (bestY - a.y) * 0.45;
      if (Math.hypot(bestX - a.x, bestY - a.y) < 0.75) {
        a.x = bestX;
        a.y = bestY;
        a.state = a.carryingPigment >= 0 ? 'haul_up' : 'patrol_down';
        a.stateTimer = 0;
      }
      return;
    }

    // 2. Completely entombed inside loose/packed sand: bite 1 grain into mandibles (if empty-handed)
    //    or nudge upward toward top surface crest without destroying any grain!
    if (a.stateTimer >= 6) {
      const curIdx = gy * GRID_W + gx;
      if (a.carryingPigment < 0 && (ct[curIdx] === CELL_LOOSE_SAND || ct[curIdx] === CELL_PACKED_SAND)) {
        a.carryingPigment = host.pigmentId[curIdx];
        a.carryingShade = host.grainShade[curIdx];
        ct[curIdx] = CELL_EMPTY;
        host.bondHealth[curIdx] = 0;
        host.velX[curIdx] = 0;
        host.velY[curIdx] = 0;
        host.texturesDirty = true;
        this.navDirty = true;
        a.state = 'rescue_dig';
        a.stateTimer = 0;
      } else if (a.stateTimer > 18) {
        // Already carrying a grain while deeply entombed: emerge cleanly to nearest open cell
        const safe = this.findNearestWalkableOpenCell(host, a.x, a.y + 8, 28);
        a.x = safe.x;
        a.y = safe.y;
        a.state = 'haul_up';
        a.stateTimer = 0;
      }
    }
  }

  /**
   * Core strict open-corridor steering step: evaluates ONLY walkable `CELL_EMPTY` neighbors
   * around `(a.x, a.y)`, descending `distField` while hugging tunnel surfaces smoothly.
   */
  private stepAlongOpenField(
    a: DiggerAgent,
    host: ColonyGridHost,
    distField: Int16Array,
    speed: number,
    usePheromoneBias: boolean,
    isHauling: boolean
  ): boolean {
    const ct = host.cellType;
    const gx = this.clampGridX(Math.round(a.x));
    const gy = this.clampGridY(Math.round(a.y));
    const curIdx = gy * GRID_W + gx;
    const curDist = distField[curIdx];

    let bestScore = -1e9;
    let bestNx = gx;
    let bestNy = gy;
    let found = false;

    // Inspect a 2-ring neighborhood of strictly open reachable cells (prevents grid jitter!)
    for (let dy = -2; dy <= 2; dy++) {
      const ny = gy + dy;
      if (ny < 2 || ny >= GRID_H - 3) continue;
      for (let dx = -2; dx <= 2; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = gx + dx;
        if (nx < 2 || nx >= GRID_W - 2) continue;

        const nIdx = ny * GRID_W + nx;
        if (ct[nIdx] !== CELL_EMPTY) continue;
        const wD = this.emptyWallDist[nIdx];
        const maxWalk =
          host.airClass[nIdx] === AIR_SKY ? MAX_SKY_WALK_CHAMFER : MAX_CAVITY_WALK_CHAMFER;
        if (wD > maxWalk) continue;

        // Verify straight line of sight AND actual continuous substep from `(a.x, a.y)` land strictly in CELL_EMPTY!
        if (!this.isSegmentOpen(ct, gx, gy, nx, ny)) continue;
        const cDx = nx - a.x;
        const cDy = ny - a.y;
        const cDist = Math.hypot(cDx, cDy);
        if (cDist > 1e-4) {
          const stepLen = Math.min(speed, cDist);
          const probeX = this.clampGridX(Math.round(a.x + (cDx / cDist) * stepLen));
          const probeY = this.clampGridY(Math.round(a.y + (cDy / cDist) * stepLen));
          if (ct[probeY * GRID_W + probeX] !== CELL_EMPTY) continue;
        }

        const nDist = distField[nIdx];
        let score = 0;

        if (nDist < INF_DIST && curDist < INF_DIST) {
          // Strongest pull: actual open-corridor geodesic progress!
          score += (curDist - nDist) * 2.15;
        } else if (nDist < INF_DIST) {
          score += 24;
        } else {
          // Fallback exploration within open cavity
          score -= 6;
        }

        // Surface-hugging footing preference: ideal wall chamfer distance ~2..5 (1..2.5 cells from sand!)
        const idealFoot = wD >= 2 && wD <= 5 ? 0.55 : wD > 7 ? -0.85 : 0.15;
        score += idealFoot;

        // Smooth heading inertia so ants walk purposefully without nervously zig-zagging
        const candAng = Math.atan2(dy, dx);
        const align = Math.cos(candAng - a.heading);
        score += align * 0.85;

        // Invisible stigmergic pheromone recruitment when foraging down to work
        if (usePheromoneBias) {
          const ph = this.samplePheromone(nx, ny);
          score += ph * 0.65;
          // Slight individual branch preference so the 12 ants naturally split across left/right galleries!
          score += dx * a.branchDir * 0.16;
          if (!isHauling && dy < 0) score += 0.12;
        }

        // Gentle neighbor spacing (avoids stacking multiple ants on the exact same pixel)
        for (let j = 0; j < this.ants.length; j++) {
          if (j === a.id) continue;
          const other = this.ants[j];
          const od2 = (other.x - nx) * (other.x - nx) + (other.y - ny) * (other.y - ny);
          if (od2 < 12) {
            score -= (12 - od2) * 0.08;
          }
        }

        // Tiny organic jitter to break symmetry cleanly
        score += (host.nextRandom() - 0.5) * 0.18;

        if (score > bestScore) {
          bestScore = score;
          bestNx = nx;
          bestNy = ny;
          found = true;
        }
      }
    }

    if (!found) {
      // Corner pocket fallback: gently nudge directly toward the best immediately open 4-neighbor!
      a.stuckTicks++;
      for (let k = 0; k < 4; k++) {
        const nx = gx + DIR_X[k];
        const ny = gy + DIR_Y[k];
        if (nx < 3 || nx >= GRID_W - 3 || ny < 3 || ny >= GRID_H - 4) continue;
        if (ct[ny * GRID_W + nx] === CELL_EMPTY) {
          a.x = a.x * 0.5 + nx * 0.5;
          a.y = a.y * 0.5 + ny * 0.5;
          a.walkPhase += 0.25;
          return true;
        }
      }
      return false;
    }

    const moved = this.moveAntSmoothlyTowardCell(a, host, bestNx, bestNy, speed);
    if (moved) {
      a.stuckTicks = 0;
      return true;
    }
    a.stuckTicks++;
    if (a.stuckTicks >= 8 && ct[bestNy * GRID_W + bestNx] === CELL_EMPTY) {
      // Cleanly center onto the verified open neighbor cell so no ant ever stalls in a sub-cell notch!
      a.x = bestNx;
      a.y = bestNy;
      a.stuckTicks = 0;
      return true;
    }
    return false;
  }

  /** Moves ant smoothly toward an open cell `(tx, ty)` while verifying EVERY substep stays in `CELL_EMPTY`. */
  private moveAntSmoothlyTowardCell(
    a: DiggerAgent,
    host: ColonyGridHost,
    tx: number,
    ty: number,
    speed: number
  ): boolean {
    const ct = host.cellType;
    const dx = tx - a.x;
    const dy = ty - a.y;
    const dist = Math.hypot(dx, dy);
    if (dist < 1e-4) return false;

    const step = Math.min(speed, dist);
    const nx = Math.max(3, Math.min(GRID_W - 4, a.x + (dx / dist) * step));
    const ny = Math.max(3, Math.min(GRID_H - 4, a.y + (dy / dist) * step));
    const ix = Math.round(nx);
    const iy = Math.round(ny);

    if (ct[iy * GRID_W + ix] !== CELL_EMPTY) {
      return false;
    }

    a.vx = nx - a.x;
    a.vy = ny - a.y;
    a.x = nx;
    a.y = ny;

    const moveAng = Math.atan2(a.vy, a.vx);
    a.heading = this.lerpAngle(a.heading, moveAng, 0.24);
    a.walkPhase += (0.28 + step * 0.26) * a.speedMul;
    a.antennaPhase += 0.22;
    return true;
  }

  /**
   * Computes smooth local dorsal normal `(upX, upY)` pointing AWAY from the nearest
   * tunnel floor, wall, pebble, or ceiling so each 3D worker ant plants all 6 legs
   * firmly onto the actual sand surface!
   */
  private updateAntSurfaceFrame(a: DiggerAgent, host: ColonyGridHost): void {
    const norm = this.sampleSurfaceNormal(host, a.x, a.y, a.wallSide);
    // Smoothly damp dorsal up vector to prevent any abrupt flips at corners or junctions
    let ux = a.upX * 0.8 + norm.nx * 0.2;
    let uy = a.upY * 0.8 + norm.ny * 0.2;
    const len = Math.hypot(ux, uy);
    if (len > 1e-4) {
      ux /= len;
      uy /= len;
    } else {
      ux = 0;
      uy = 1;
    }
    a.upX = ux;
    a.upY = uy;
  }

  private sampleSurfaceNormal(
    host: ColonyGridHost,
    gxFloat: number,
    gyFloat: number,
    wallSideBias: number
  ): { nx: number; ny: number } {
    const ct = host.cellType;
    const cx = this.clampGridX(Math.round(gxFloat));
    const cy = this.clampGridY(Math.round(gyFloat));
    let sumX = 0;
    let sumY = 0;
    const R = 5;

    for (let dy = -R; dy <= R; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= GRID_H) continue;
      const row = y * GRID_W;
      for (let dx = -R; dx <= R; dx++) {
        if (dx === 0 && dy === 0) continue;
        const x = cx + dx;
        if (x < 0 || x >= GRID_W) continue;
        if (ct[row + x] !== CELL_EMPTY) {
          const d2 = dx * dx + dy * dy;
          if (d2 <= R * R) {
            const w = 1 / (d2 + 0.5);
            // Vector pointing AWAY from solid cell into open cavity
            sumX -= dx * w;
            sumY -= dy * w;
          }
        }
      }
    }

    // Subtle gravity floor bias + personal wall-side tiebreaker inside symmetric vertical shafts
    sumY += 0.16;
    sumX += wallSideBias * 0.05;

    const len = Math.hypot(sumX, sumY);
    if (len > 1e-4) {
      return { nx: sumX / len, ny: sumY / len };
    }
    return { nx: 0, ny: 1 };
  }

  /** Initiates brief antennae-tap greetings when two workers meet in a corridor. */
  private tryStartNestmateGreeting(a: DiggerAgent, idxA: number): void {
    const ants = this.ants;
    for (let j = idxA + 1; j < ants.length; j++) {
      const b = ants[j];
      if (b.greetCooldown > 0) continue;
      if (b.state !== 'patrol_down' && b.state !== 'haul_up') continue;

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > 36 || d2 < 2.5) continue; // Within ~6 cells

      // Offset zLanes slightly so they step cleanly past each other in 3D after greeting
      a.targetZLane = a.id % 2 === 0 ? 0.34 : -0.34;
      b.targetZLane = -a.targetZLane;

      // Only pause for antennae-tap greeting if heading roughly toward one another
      const dotHeading = Math.cos(a.heading) * Math.cos(b.heading) + Math.sin(a.heading) * Math.sin(b.heading);
      if (dotHeading < -0.15) {
        // Recruitment transfer: loaded hauler shares its active dig direction/site with empty worker!
        if (a.carryingPigment >= 0 && b.carryingPigment < 0) {
          b.branchDir = a.branchDir;
          b.lastWorkX = a.lastWorkX;
          b.lastWorkY = a.lastWorkY;
          this.depositPheromone(a.x, a.y, 0.55);
        } else if (b.carryingPigment >= 0 && a.carryingPigment < 0) {
          a.branchDir = b.branchDir;
          a.lastWorkX = b.lastWorkX;
          a.lastWorkY = b.lastWorkY;
          this.depositPheromone(b.x, b.y, 0.55);
        }

        a.prevState = a.state;
        b.prevState = b.state;
        a.state = 'greet';
        b.state = 'greet';
        a.stateTimer = 0;
        b.stateTimer = 0;
        a.greetPartnerId = b.id;
        b.greetPartnerId = a.id;
        this.nestmateGreetings++;
        return;
      }
    }
  }

  private findBestRescuePlugGrain(host: ColonyGridHost, gx: number, gy: number): number {
    const ct = host.cellType;
    let bestIdx = -1;
    let bestScore = -1e9;
    for (let dy = -1; dy <= 2; dy++) {
      const sy = gy + dy;
      if (sy < 10 || sy >= GRID_H - 6) continue;
      for (let dx = -2; dx <= 2; dx++) {
        if (dx === 0 && dy === 0) continue;
        if (dx * dx + dy * dy > 5) continue;
        const sx = gx + dx;
        if (sx < 4 || sx >= GRID_W - 4) continue;
        const idx = sy * GRID_W + sx;
        const c = ct[idx];
        if (c !== CELL_LOOSE_SAND && c !== CELL_PACKED_SAND) continue;

        // Strongly prefer digging upward & through loose collapse rubble toward surface/open tunnel!
        let score = dy * 2.6 + (c === CELL_LOOSE_SAND ? 3.2 : 0.4);
        // Bonus if open surface-connected tunnel lies just 2..6 cells past this plug!
        const probeX = this.clampGridX(sx + dx * 3);
        const probeY = this.clampGridY(sy + dy * 3);
        if (this.distToSurface[probeY * GRID_W + probeX] < INF_DIST) {
          score += 8.0;
        }
        if (score > bestScore) {
          bestScore = score;
          bestIdx = idx;
        }
      }
    }
    return bestIdx;
  }

  private measureLocalEmptySpan(ct: Uint8Array, sx: number, y: number): number {
    if (y < 2 || y >= GRID_H - 2) return 0;
    const row = y * GRID_W;
    if (ct[row + sx] !== CELL_EMPTY) return 0;
    let span = 1;
    for (let x = sx - 1; x >= 1 && span < 28; x--) {
      if (ct[row + x] === CELL_EMPTY) span++;
      else break;
    }
    for (let x = sx + 1; x < GRID_W - 1 && span < 28; x++) {
      if (ct[row + x] === CELL_EMPTY) span++;
      else break;
    }
    return span;
  }

  private hasAdjacentCell(host: ColonyGridHost, gx: number, gy: number, targetType: number): boolean {
    const ct = host.cellType;
    for (let dy = -2; dy <= 2; dy++) {
      const y = gy + dy;
      if (y < 2 || y >= GRID_H - 2) continue;
      const row = y * GRID_W;
      for (let dx = -2; dx <= 2; dx++) {
        if (dx * dx + dy * dy > 4) continue;
        const x = gx + dx;
        if (x < 2 || x >= GRID_W - 2) continue;
        if (ct[row + x] === targetType) return true;
      }
    }
    return false;
  }

  private isSegmentOpen(ct: Uint8Array, x0: number, y0: number, x1: number, y1: number): boolean {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const steps = Math.max(Math.abs(dx), Math.abs(dy));
    if (steps <= 1) {
      if (dx !== 0 && dy !== 0) {
        return ct[y0 * GRID_W + x1] === CELL_EMPTY && ct[y1 * GRID_W + x0] === CELL_EMPTY;
      }
      return ct[y1 * GRID_W + x1] === CELL_EMPTY;
    }
    for (let s = 1; s <= steps; s++) {
      const t = s / steps;
      const mx = Math.round(x0 + dx * t);
      const my = Math.round(y0 + dy * t);
      if (ct[my * GRID_W + mx] !== CELL_EMPTY) return false;
    }
    return true;
  }

  private findNearestWalkableOpenCell(
    host: ColonyGridHost,
    startX: number,
    startY: number,
    maxR: number
  ): { x: number; y: number } {
    const ct = host.cellType;
    const sx = this.clampGridX(Math.round(startX));
    const sy = this.clampGridY(Math.round(startY));
    if (
      ct[sy * GRID_W + sx] === CELL_EMPTY &&
      this.emptyWallDist[sy * GRID_W + sx] <=
        (host.airClass[sy * GRID_W + sx] === AIR_SKY
          ? MAX_SKY_WALK_CHAMFER
          : MAX_CAVITY_WALK_CHAMFER)
    ) {
      return { x: sx, y: sy };
    }

    let bestX = sx;
    let bestY = sy;
    let bestScore = 1e9;
    for (let r = 1; r <= maxR; r++) {
      for (let dy = -r; dy <= r; dy++) {
        const y = sy + dy;
        if (y < 4 || y >= GRID_H - 4) continue;
        for (let dx = -r; dx <= r; dx++) {
          if (Math.abs(dx) !== r && Math.abs(dy) !== r) continue;
          const x = sx + dx;
          if (x < 4 || x >= GRID_W - 4) continue;
          const idx = y * GRID_W + x;
          const maxW =
            host.airClass[idx] === AIR_SKY ? MAX_SKY_WALK_CHAMFER : MAX_CAVITY_WALK_CHAMFER;
          if (ct[idx] === CELL_EMPTY && this.emptyWallDist[idx] <= maxW) {
            const sc = dx * dx + dy * dy + (this.distToSurface[idx] < INF_DIST ? 0 : 12);
            if (sc < bestScore) {
              bestScore = sc;
              bestX = x;
              bestY = y;
            }
          }
        }
      }
      if (bestScore < 1e8) return { x: bestX, y: bestY };
    }

    // Ultimate fallback: top surface crest above sx
    const surfY = Math.min(GRID_H - 4, this.findSurfaceY(host, sx) + 2);
    return { x: sx, y: surfY };
  }

  private findSurfaceY(host: ColonyGridHost, x: number): number {
    const cx = this.clampGridX(x);
    const t = host.topSolidY[cx];
    if (t > 10) return t;
    const ct = host.cellType;
    for (let y = GRID_H - 2; y >= 10; y--) {
      if (ct[y * GRID_W + cx] !== CELL_EMPTY) return y;
    }
    return Math.round(GRID_H * SURFACE_FRAC);
  }

  private depositPheromone(gx: number, gy: number, amount: number): void {
    const cx = Math.max(0, Math.min(COARSE_W - 1, Math.round(gx) >> 1));
    const cy = Math.max(0, Math.min(COARSE_H - 1, Math.round(gy) >> 1));
    const idx = cy * COARSE_W + cx;
    const next = this.pheromoneDig[idx] + amount;
    this.pheromoneDig[idx] = next > 3.0 ? 3.0 : next;
  }

  private samplePheromone(gx: number, gy: number): number {
    const cx = Math.max(0, Math.min(COARSE_W - 1, Math.round(gx) >> 1));
    const cy = Math.max(0, Math.min(COARSE_H - 1, Math.round(gy) >> 1));
    return this.pheromoneDig[cy * COARSE_W + cx];
  }

  private sampleDisturbance(gx: number, gy: number): number {
    const cx = Math.max(0, Math.min(COARSE_W - 1, Math.round(gx) >> 1));
    const cy = Math.max(0, Math.min(COARSE_H - 1, Math.round(gy) >> 1));
    return this.disturbanceMap[cy * COARSE_W + cx];
  }

  private clampGridX(x: number): number {
    return x < 3 ? 3 : x > GRID_W - 4 ? GRID_W - 4 : x;
  }

  private clampGridY(y: number): number {
    return y < 3 ? 3 : y > GRID_H - 4 ? GRID_H - 4 : y;
  }

  private lerpAngle(cur: number, target: number, alpha: number): number {
    let diff = target - cur;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    return cur + diff * alpha;
  }

  /** Summarizes colony live state (read-only) for verification tests & `window.__DAY27__`. */
  getStats(): ColonyStats {
    let trappedCount = 0;
    let greetingCount = 0;
    let fallingCount = 0;
    let carryingCount = 0;
    for (let i = 0; i < this.ants.length; i++) {
      const a = this.ants[i];
      if (a.state === 'rescue_dig' || a.state === 'backfill') trappedCount++;
      if (a.state === 'greet') greetingCount++;
      if (a.state === 'falling') fallingCount++;
      if (a.carryingPigment >= 0) carryingCount++;
    }
    return {
      antCount: this.ants.length,
      trappedCount,
      greetingCount,
      fallingCount,
      carryingCount,
      solidWallPenetrations: this.solidWallPenetrations,
      rescuesTriggered: this.rescuesTriggered,
      nestmateGreetings: this.nestmateGreetings,
    };
  }
}
