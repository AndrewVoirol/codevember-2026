import {
  DEFAULT_SILK_CONFIG,
  type ButterflyState,
  type ChrysalisStage,
  type EmberMoteState,
  type SilkLanternConfig,
  type SilkLanternSimState,
  type SilkLoopState,
} from './types';

/** Vertical bounds of the ovoid silk-spun lantern in lantern-local coordinates. */
export const LANTERN_BOTTOM_Y = -0.78;
export const LANTERN_TOP_Y = 0.84;
export const LANTERN_HEIGHT = LANTERN_TOP_Y - LANTERN_BOTTOM_Y;
export const LANTERN_MOUTH_Y = -0.68;
export const LANTERN_MOUTH_RADIUS = 0.26;
export const LANTERN_MAX_RADIUS = 0.62;

const BF_MODE_DORMANT = 0;
const BF_MODE_EMERGING = 1;
const BF_MODE_SPIRALING = 2;
const BF_MODE_RECIRCULATING = 3;

const _curlScratch = new Float32Array(3);

/**
 * Pure, zero-GC, headless-testable physics and metamorphosis simulation
 * for Concept A ("Silk-Spun Lantern Chrysalis"):
 * `bound -> unwinding -> emerged -> buoyant`.
 */
export class SilkLanternSimulation {
  readonly config: SilkLanternConfig;
  readonly state: SilkLanternSimState;
  readonly butterflies: ButterflyState[];
  readonly silkLoops: SilkLoopState[];
  readonly embers: EmberMoteState[];

  private manualHoldTimer = 0;

  // Hanging cord pendulum, yaw, and thermal buoyancy springs
  private swingXVel = 0;
  private swingZVel = 0;
  private yawVel = 0;
  private liftYVel = 0;
  private ignitionThermalKick = 0;

  // Pointer air-wake state (zero allocations)
  private wakeX = 0;
  private wakeY = 0;
  private wakeZ = 0;
  private wakeVelX = 0;
  private wakeVelY = 0;
  private wakeVelZ = 0;
  private wakeStrength = 0;

  // Pre-allocated per-butterfly internal arrays
  private readonly bfMode: Uint8Array;
  private readonly bfTimer: Float32Array;
  private readonly bfOrbitAngle: Float32Array;
  private readonly bfOrbitDir: Float32Array;
  private readonly bfOrbitRadius: Float32Array;
  private readonly bfTargetY: Float32Array;
  private readonly bfBaseScale: Float32Array;
  private readonly bfBaseWingFreq: Float32Array;
  private readonly bfStartle: Float32Array;
  /** 0 = bottom bamboo mouth, 1 = waist silk seam */
  private readonly bfExitKind: Uint8Array;

  // Pre-allocated per-silk-loop sway velocities
  private readonly loopSwayVelX: Float32Array;
  private readonly loopSwayVelY: Float32Array;
  private readonly loopSwayVelZ: Float32Array;
  private readonly loopUnwindOrder: Float32Array;

  // Pre-allocated per-ember origin channel (0 = bottom mouth, 1 = waist silk seam)
  private readonly emberOriginKind: Uint8Array;
  private readonly emberBurstBoost: Float32Array;

  private isDisposed = false;

  constructor(config: Partial<SilkLanternConfig> = {}) {
    this.config = {
      ...DEFAULT_SILK_CONFIG,
      ...config,
    };

    const bfCount = Math.max(1, this.config.butterflyCount);
    const loopCount = Math.max(1, this.config.silkLoopCount);
    const emberCount = Math.max(1, this.config.emberCount);

    this.state = {
      stage: 'emerged',
      unravel: 0.62,
      breathExpansion: 0.68,
      ignition: 0.78,
      emberPulse: 0.32,
      lanternSwingX: 0,
      lanternSwingZ: 0,
      lanternYaw: 0,
      lanternLiftY: 0.04,
      butterfliesEmerged: 0,
      activeButterflies: 0,
      time: 0,
    };

    // 1. Pre-allocate silk loops (28 gossamer loops anchored around waist & lower rim)
    this.loopSwayVelX = new Float32Array(loopCount);
    this.loopSwayVelY = new Float32Array(loopCount);
    this.loopSwayVelZ = new Float32Array(loopCount);
    this.loopUnwindOrder = new Float32Array(loopCount);
    this.silkLoops = new Array<SilkLoopState>(loopCount);

    for (let i = 0; i < loopCount; i++) {
      const h1 = hash01(i * 13.71 + 2.3);
      const h2 = hash01(i * 37.19 + 6.1);
      // Distribute across left flank, right flank, and lower rim (matches Hero & Macro)
      const band = i % 3;
      const anchorAngle =
        band === 0
          ? Math.PI + (h1 - 0.5) * 1.15 // left flank
          : band === 1
            ? (h1 - 0.5) * 1.15 // right flank
            : (i / loopCount) * Math.PI * 2; // around lower rim
      const anchorV = band === 2 ? 0.05 + h2 * 0.16 : 0.16 + h2 * 0.38;

      this.loopUnwindOrder[i] = (i / Math.max(1, loopCount - 1)) * 0.65;
      this.silkLoops[i] = {
        index: i,
        anchorAngle,
        anchorV,
        peelAmount: 0,
        swayX: 0,
        swayY: 0,
        swayZ: 0,
        phase: i * 1.17 + h1 * 3.14,
      };
    }

    // 2. Pre-allocate origami butterflies (14)
    this.bfMode = new Uint8Array(bfCount);
    this.bfTimer = new Float32Array(bfCount);
    this.bfOrbitAngle = new Float32Array(bfCount);
    this.bfOrbitDir = new Float32Array(bfCount);
    this.bfOrbitRadius = new Float32Array(bfCount);
    this.bfTargetY = new Float32Array(bfCount);
    this.bfBaseScale = new Float32Array(bfCount);
    this.bfBaseWingFreq = new Float32Array(bfCount);
    this.bfStartle = new Float32Array(bfCount);
    this.bfExitKind = new Uint8Array(bfCount);

    this.butterflies = new Array<ButterflyState>(bfCount);
    for (let i = 0; i < bfCount; i++) {
      const h1 = hash01(i * 19.37 + 1.1);
      const h2 = hash01(i * 47.23 + 4.9);
      const h3 = hash01(i * 79.11 + 8.7);

      const baseScale = 0.82 + h1 * 0.38;
      const baseFreq = 9.0 + h2 * 3.6;

      this.bfBaseScale[i] = baseScale;
      this.bfBaseWingFreq[i] = baseFreq;
      this.bfOrbitDir[i] = i % 3 === 0 ? -1 : 1;
      this.bfOrbitAngle[i] = (i / bfCount) * Math.PI * 2 + h3 * 0.6;
      this.bfOrbitRadius[i] = 0.78 + h2 * 0.38;
      this.bfTargetY[i] = -0.55 + (i / Math.max(1, bfCount - 1)) * 1.65;
      this.bfExitKind[i] = i % 5 < 2 ? 0 : 1;

      this.butterflies[i] = {
        index: i,
        pos: [0, LANTERN_MOUTH_Y, 0],
        vel: [0, 0, 0],
        yaw: 0,
        pitch: 0.2,
        roll: 0,
        wingPhase: i * 1.43,
        wingFreq: baseFreq,
        scale: baseScale,
        warmTint: 0.35 + h2 * 0.55,
        emerged: false,
        orbitSeed: h1,
        active: false,
      };
    }

    // 3. Pre-allocate thermal ember motes (160)
    this.emberOriginKind = new Uint8Array(emberCount);
    this.emberBurstBoost = new Float32Array(emberCount);
    this.embers = new Array<EmberMoteState>(emberCount);
    for (let i = 0; i < emberCount; i++) {
      const seed = hash01(i * 31.17 + 2.9);
      this.emberOriginKind[i] = i % 2 === 0 ? 0 : 1;
      this.embers[i] = {
        pos: [0, LANTERN_MOUTH_Y, 0],
        vel: [0, 0.18, 0],
        life: 0,
        maxLife: 2.8,
        size: 0.014 + seed * 0.024,
        seed,
      };
    }

    // Boot directly into the rich 'emerged' Hero composition on frame 1
    this.applyStageSnapshot('emerged', true);
  }

  /** Read-only view of the live simulation state. */
  getState(): Readonly<SilkLanternSimState> {
    return this.state;
  }

  /**
   * Resets cleanly to the tightly bound silk chrysalis (`stage = 'bound'`, `unravel = 0.02`)
   * so the user can watch or drive the full bound -> unwinding -> emerged -> buoyant lifecycle.
   */
  resetToBound(): void {
    this.manualHoldTimer = 0;
    this.swingXVel = 0;
    this.swingZVel = 0;
    this.yawVel = 0;
    this.liftYVel = 0;
    this.ignitionThermalKick = 0;
    this.wakeStrength = 0;
    this.wakeVelX = 0;
    this.wakeVelY = 0;
    this.wakeVelZ = 0;

    this.state.stage = 'bound';
    this.state.unravel = 0.02;
    this.state.breathExpansion = 0.08;
    this.state.ignition = 0.32;
    this.state.emberPulse = 0.04;
    this.state.lanternSwingX = 0;
    this.state.lanternSwingZ = 0;
    this.state.lanternYaw = 0;
    this.state.lanternLiftY = 0;
    this.state.butterfliesEmerged = 0;
    this.state.activeButterflies = 0;

    this.syncSilkLoopsSnapshot();
    this.seedButterfliesDormant();
    this.seedEmbersForCurrentState();
  }

  /**
   * Deterministically snaps to any of the 4 stages (`bound`, `unwinding`, `emerged`, `buoyant`)
   * for interactive auditioning or optical-bench verification.
   */
  setPhase(stage: ChrysalisStage): void {
    this.applyStageSnapshot(stage, false);
  }

  /**
   * Couples pointer motion into lantern pendulum torque, unspooling silk loop sway,
   * butterfly curl-swirl, ember wakes, and waist-brush silk unspooling.
   */
  applyPointer(
    wx: number,
    wy: number,
    wz: number,
    vx: number,
    vy: number,
    vz: number,
    isDown: boolean,
    dt: number
  ): void {
    if (this.isDisposed) return;
    const safeDt = clamp(dt, 0.0005, 0.05);
    const cvx = clamp(vx, -8, 8);
    const cvy = clamp(vy, -8, 8);
    const cvz = clamp(vz, -8, 8);
    const speed = Math.hypot(cvx, cvy, cvz);

    this.wakeX = wx;
    this.wakeY = wy;
    this.wakeZ = wz;

    const blend = 1 - Math.exp(-safeDt * 18);
    this.wakeVelX += (cvx - this.wakeVelX) * blend;
    this.wakeVelY += (cvy - this.wakeVelY) * blend;
    this.wakeVelZ += (cvz - this.wakeVelZ) * blend;

    if (speed > 0.025 || isDown) {
      this.wakeStrength = clamp(
        this.wakeStrength + (isDown ? 0.42 : 0.24) * (0.35 + Math.min(1.6, speed * 0.45)),
        0,
        1.5
      );
    }

    // Hanging lantern pendulum torque & torsional yaw from pointer wind
    const dyLantern = wy - this.state.lanternLiftY;
    const lanternDistSq = wx * wx + dyLantern * dyLantern + wz * wz;
    const lanternProx = Math.exp(-lanternDistSq * 0.65);

    if (speed > 0.02) {
      const windGain = lanternProx * safeDt * (isDown ? 2.5 : 1.55);
      // Horizontal sweep swings the hanging lantern in the direction of airflow
      this.swingZVel += clamp(cvx * windGain * 0.45, -0.85, 0.85);
      this.swingXVel += clamp(-cvz * windGain * 0.42 + cvy * windGain * 0.12, -0.85, 0.85);
      const torqueY = (wz * cvx - wx * cvz) * windGain * 0.42;
      this.yawVel += clamp(torqueY, -0.8, 0.8);

      // Sweeping across the silk-wrapped waist gradually unwinds more silk!
      const radialXZ = Math.hypot(wx, wz);
      if (wy >= -0.68 && wy <= 0.28 && radialXZ < 1.15 && speed > 0.22) {
        const unwindBoost = speed * safeDt * (isDown ? 0.09 : 0.028) * lanternProx;
        this.state.unravel = clamp(this.state.unravel + unwindBoost, 0.02, 1.0);
        this.state.emberPulse = Math.min(1.6, this.state.emberPulse + unwindBoost * 1.4);
      }
    }

    // Direct pointer swirl on active butterflies
    if (speed > 0.04) {
      const bfCount = this.butterflies.length;
      for (let i = 0; i < bfCount; i++) {
        const bf = this.butterflies[i];
        if (!bf.active) continue;
        const dx = bf.pos[0] - wx;
        const dy = bf.pos[1] - wy;
        const dz = bf.pos[2] - wz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 2.4) {
          const inf = Math.exp(-d2 * 1.35) * safeDt * (isDown ? 3.0 : 1.9);
          bf.vel[0] += (cvx * 1.2 - dz * 0.8) * inf;
          bf.vel[1] += (cvy * 1.1 + 0.32 * Math.min(2.0, speed)) * inf;
          bf.vel[2] += (cvz * 1.2 + dx * 0.8) * inf;
          this.bfStartle[i] = Math.min(1.5, this.bfStartle[i] + inf * 2.2);
        }
      }
    }
  }

  /**
   * Click/tap stoke burst: nudges `unravel` higher (+0.14), stokes `emberPulse`,
   * imparts a gentle pendulum swing impulse, and launches 3-4 butterflies from the mouth/waist seams.
   */
  stokeBurst(wx: number, wy: number, wz: number): number {
    if (this.isDisposed) return 0;

    this.state.unravel = clamp(this.state.unravel + 0.14, 0.02, 1.0);
    this.state.emberPulse = Math.min(1.85, this.state.emberPulse + 0.85);
    this.ignitionThermalKick = Math.min(0.45, this.ignitionThermalKick + 0.22);
    this.liftYVel += 0.14 * (0.35 + 0.65 * this.state.unravel);

    // Gentle pendulum swing impulse from click point
    this.swingZVel += clamp(-wx * 0.42 + (Math.abs(wx) < 0.05 ? 0.18 : 0), -0.55, 0.55);
    this.swingXVel += clamp(wz * 0.42, -0.55, 0.55);
    this.yawVel += clamp((wz * 0.25 - wx * 0.25), -0.45, 0.45);

    // Kick silk loops outward in a brief warm gust
    const loopCount = this.silkLoops.length;
    for (let i = 0; i < loopCount; i++) {
      const loop = this.silkLoops[i];
      const ca = Math.cos(loop.anchorAngle);
      const sa = Math.sin(loop.anchorAngle);
      this.loopSwayVelX[i] += ca * 0.45;
      this.loopSwayVelY[i] += 0.28;
      this.loopSwayVelZ[i] += sa * 0.45;
    }

    // Launch 3-4 butterflies out of the bottom mouth & waist seams, and energize active ones
    const bfCount = this.butterflies.length;
    let launched = 0;
    let energized = 0;
    const launchQuota = 4;

    for (let i = 0; i < bfCount; i++) {
      const bf = this.butterflies[i];
      const mode = this.bfMode[i];

      if ((mode === BF_MODE_DORMANT || mode === BF_MODE_RECIRCULATING) && launched < launchQuota) {
        this.launchButterflyFromSeam(i, true);
        launched++;
        energized++;
      } else if (bf.active) {
        const dx = bf.pos[0] - wx;
        const dy = bf.pos[1] - wy;
        const dz = bf.pos[2] - wz;
        const d2 = dx * dx + dy * dy + dz * dz;
        const prox = Math.exp(-d2 * 0.7);
        bf.vel[0] += (dx * 0.35 + Math.cos(this.bfOrbitAngle[i]) * 0.2) * (0.4 + prox);
        bf.vel[1] += 0.24 + 0.26 * prox;
        bf.vel[2] += (dz * 0.35 + Math.sin(this.bfOrbitAngle[i]) * 0.2) * (0.4 + prox);
        this.bfStartle[i] = Math.min(1.5, this.bfStartle[i] + 0.95);
        energized++;
      }
    }

    // Kick ember motes upward
    const emberCount = this.embers.length;
    for (let i = 0; i < emberCount; i++) {
      const mote = this.embers[i];
      const dx = mote.pos[0] - wx;
      const dy = mote.pos[1] - wy;
      const dz = mote.pos[2] - wz;
      const d2 = dx * dx + dy * dy + dz * dz;
      const prox = Math.exp(-d2 * 0.65);
      if (prox > 0.12 || i % 3 === 0) {
        mote.vel[0] += dx * 0.25 * prox;
        mote.vel[1] += 0.26 + 0.38 * prox;
        mote.vel[2] += dz * 0.25 * prox;
        this.emberBurstBoost[i] = Math.min(1.5, this.emberBurstBoost[i] + 0.8 * (0.4 + prox));
      }
    }

    return energized;
  }

  /**
   * Continuous drag-to-unspool / re-bind control across `[0.02, 1.0]`.
   */
  nudgeUnravel(delta: number): void {
    if (this.isDisposed || !Number.isFinite(delta) || delta === 0) return;
    this.state.unravel = clamp(this.state.unravel + delta, 0.02, 1.0);
    this.manualHoldTimer = 2.2;
    if (delta > 0) {
      this.state.emberPulse = Math.min(1.6, this.state.emberPulse + delta * 0.9);
      this.liftYVel += delta * 0.18;
    }
  }

  /** Advances physics, silk unspooling, butterfly emergence, and ember convection by `dtSec`. */
  step(dtSec: number): void {
    if (this.isDisposed) return;
    const dt = clamp(dtSec, 0.0005, 0.05);
    this.state.time += dt;
    const t = this.state.time;

    if (this.manualHoldTimer > 0) {
      this.manualHoldTimer = Math.max(0, this.manualHoldTimer - dt);
    }

    // 1. Continuous unravel progression when autoAdvance is active
    if (this.config.autoAdvance && this.manualHoldTimer <= 0) {
      const u = this.state.unravel;
      if (u < 0.86) {
        this.state.unravel = Math.min(1.0, u + 0.095 * dt);
      } else {
        // Gentle asymptotic approach toward airy buoyant float (~0.96)
        this.state.unravel += (0.96 - u) * (1 - Math.exp(-0.42 * dt));
      }
    }

    const u = this.state.unravel;
    this.syncStageLabel(u);

    // 2. Coupled lantern breath expansion, ignition, ember pulse, and buoyant lift
    const breathWave = Math.sin(t * 1.15) * 0.018 * (0.2 + 0.8 * u);
    const targetBreath = clamp(0.10 + 0.90 * Math.pow(u, 0.86) + breathWave, 0.06, 1.0);
    this.state.breathExpansion += (targetBreath - this.state.breathExpansion) * (1 - Math.exp(-dt * 4.2));

    this.ignitionThermalKick *= Math.exp(-dt * 0.85);
    const targetIgnition = clamp(0.32 + 0.68 * Math.pow(u, 0.82) + this.ignitionThermalKick, 0.25, 1.0);
    this.state.ignition += (targetIgnition - this.state.ignition) * (1 - Math.exp(-dt * 3.6));

    this.state.emberPulse *= Math.exp(-dt * 1.5);

    const thermalBob = (Math.sin(t * 0.92) * 0.012 + Math.sin(t * 0.51 + 1.4) * 0.008) * u;
    const targetLiftY = clamp(
      0.14 * Math.pow(clamp((u - 0.16) / 0.78, 0, 1), 1.2) * (0.45 + 0.55 * this.state.ignition) + thermalBob,
      0,
      0.18
    );
    const liftForce = (targetLiftY - this.state.lanternLiftY) * 9.0;
    this.liftYVel = (this.liftYVel + liftForce * dt) * Math.exp(-4.4 * dt);
    this.state.lanternLiftY = clamp(this.state.lanternLiftY + this.liftYVel * dt, 0, 0.2);

    // 3. Hanging cord damped pendulum sway & torsional yaw
    const ambientSwingX = Math.sin(t * 0.78 + 0.5) * 0.022 * (0.35 + 0.65 * u);
    const ambientSwingZ = Math.cos(t * 0.64 + 1.8) * 0.026 * (0.35 + 0.65 * u);
    const fSwingX = (ambientSwingX - this.state.lanternSwingX) * 11.5;
    const fSwingZ = (ambientSwingZ - this.state.lanternSwingZ) * 11.5;
    this.swingXVel = (this.swingXVel + fSwingX * dt) * Math.exp(-2.8 * dt);
    this.swingZVel = (this.swingZVel + fSwingZ * dt) * Math.exp(-2.8 * dt);
    this.state.lanternSwingX = clamp(this.state.lanternSwingX + this.swingXVel * dt, -0.38, 0.38);
    this.state.lanternSwingZ = clamp(this.state.lanternSwingZ + this.swingZVel * dt, -0.38, 0.38);

    const naturalYawRate = 0.065 * (0.25 + 0.75 * u);
    this.yawVel += (naturalYawRate - this.yawVel) * (1 - Math.exp(-dt * 2.4));
    this.state.lanternYaw += this.yawVel * dt;

    // Decay pointer air-wake
    this.wakeStrength = Math.max(0, this.wakeStrength - dt * 1.35);
    const wakeDecay = Math.exp(-dt * 5.0);
    this.wakeVelX *= wakeDecay;
    this.wakeVelY *= wakeDecay;
    this.wakeVelZ *= wakeDecay;

    // 4. Step unspooling silk loops, emerging butterflies, and thermal embers
    this.stepSilkLoops(dt);
    this.stepButterflies(dt);
    this.stepEmbers(dt);
  }

  dispose(): void {
    this.isDisposed = true;
  }

  // ---------------------------------------------------------------------------
  // Stage snapshotting & seeding
  // ---------------------------------------------------------------------------

  private syncStageLabel(u: number): void {
    if (u < 0.22) {
      this.state.stage = 'bound';
    } else if (u < 0.54) {
      this.state.stage = 'unwinding';
    } else if (u < 0.84) {
      this.state.stage = 'emerged';
    } else {
      this.state.stage = 'buoyant';
    }
  }

  private applyStageSnapshot(stage: ChrysalisStage, isBoot: boolean): void {
    this.manualHoldTimer = 0;
    this.swingXVel = 0;
    this.swingZVel = 0;
    this.yawVel = 0;
    this.liftYVel = 0;
    this.wakeStrength = 0;

    if (stage === 'bound') {
      this.state.stage = 'bound';
      this.state.unravel = 0.05;
      this.state.breathExpansion = 0.12;
      this.state.ignition = 0.35;
      this.state.lanternLiftY = 0;
      this.state.emberPulse = 0.05;
      this.state.butterfliesEmerged = 0;
      this.ignitionThermalKick = 0;
      this.seedButterfliesDormant();
    } else if (stage === 'unwinding') {
      this.state.stage = 'unwinding';
      this.state.unravel = 0.38;
      this.state.breathExpansion = 0.42;
      this.state.ignition = 0.58;
      this.state.lanternLiftY = 0.015;
      this.state.emberPulse = 0.22;
      this.state.butterfliesEmerged = 5;
      this.ignitionThermalKick = 0.06;
      this.seedButterfliesForStage(5, 0.55);
    } else if (stage === 'emerged') {
      this.state.stage = 'emerged';
      this.state.unravel = isBoot ? 0.62 : 0.68;
      this.state.breathExpansion = isBoot ? 0.68 : 0.74;
      this.state.ignition = isBoot ? 0.78 : 0.82;
      this.state.lanternLiftY = isBoot ? 0.04 : 0.05;
      this.state.emberPulse = 0.32;
      this.state.butterfliesEmerged = Math.max(this.state.butterfliesEmerged, 11);
      this.ignitionThermalKick = 0.12;
      this.seedButterfliesForStage(11, 1.0);
    } else {
      // 'buoyant'
      this.state.stage = 'buoyant';
      this.state.unravel = 0.96;
      this.state.breathExpansion = 1.0;
      this.state.ignition = 1.0;
      this.state.lanternLiftY = 0.14;
      this.state.emberPulse = 0.48;
      this.state.butterfliesEmerged = Math.max(this.state.butterfliesEmerged, this.butterflies.length);
      this.ignitionThermalKick = 0.18;
      this.seedButterfliesForStage(this.butterflies.length, 1.15);
    }

    this.state.lanternSwingX = 0;
    this.state.lanternSwingZ = 0;
    this.syncSilkLoopsSnapshot();
    this.seedEmbersForCurrentState();
  }

  private syncSilkLoopsSnapshot(): void {
    const u = this.state.unravel;
    const count = this.silkLoops.length;
    for (let i = 0; i < count; i++) {
      const loop = this.silkLoops[i];
      const order = this.loopUnwindOrder[i];
      loop.peelAmount = clamp((u - order * 0.45) / 0.55, 0, 1);
      loop.swayX = 0;
      loop.swayY = 0;
      loop.swayZ = 0;
      this.loopSwayVelX[i] = 0;
      this.loopSwayVelY[i] = 0;
      this.loopSwayVelZ[i] = 0;
    }
  }

  private seedButterfliesDormant(): void {
    const count = this.butterflies.length;
    for (let i = 0; i < count; i++) {
      const bf = this.butterflies[i];
      this.bfMode[i] = BF_MODE_DORMANT;
      this.bfTimer[i] = 0.12 + i * 0.24;
      this.bfStartle[i] = 0;
      bf.pos[0] = Math.cos(this.bfOrbitAngle[i]) * 0.08;
      bf.pos[1] = LANTERN_MOUTH_Y + 0.14;
      bf.pos[2] = Math.sin(this.bfOrbitAngle[i]) * 0.08;
      bf.vel[0] = 0;
      bf.vel[1] = 0;
      bf.vel[2] = 0;
      bf.yaw = this.bfOrbitAngle[i];
      bf.pitch = 0.18;
      bf.roll = 0;
      bf.scale = this.bfBaseScale[i] * 0.65;
      bf.warmTint = 0.35 + bf.orbitSeed * 0.45;
      bf.emerged = false;
      bf.active = false;
    }
    this.state.activeButterflies = 0;
  }

  /**
   * Places active butterflies in the iconic Concept A Hero silhouette:
   * 2-3 emerging near the bottom bamboo mouth, 2-3 near the waist silk loops,
   * and 5-6 spiraling gracefully up the right & upper flanks.
   */
  private seedButterfliesForStage(activeQuota: number, spreadScale: number): void {
    const count = this.butterflies.length;
    const quota = clamp(activeQuota, 0, count);
    const liftY = this.state.lanternLiftY;
    let activeCount = 0;

    for (let i = 0; i < count; i++) {
      const bf = this.butterflies[i];
      this.bfStartle[i] = 0;

      if (i >= quota) {
        this.bfMode[i] = BF_MODE_DORMANT;
        this.bfTimer[i] = 0.25 + (i - quota) * 0.35;
        bf.pos[0] = Math.cos(this.bfOrbitAngle[i]) * 0.08;
        bf.pos[1] = LANTERN_MOUTH_Y + 0.12 + liftY;
        bf.pos[2] = Math.sin(this.bfOrbitAngle[i]) * 0.08;
        bf.vel[0] = 0;
        bf.vel[1] = 0;
        bf.vel[2] = 0;
        bf.emerged = false;
        bf.active = false;
        continue;
      }

      this.bfMode[i] = BF_MODE_SPIRALING;
      this.bfTimer[i] = 3.2 + bf.orbitSeed * 3.8;

      let ang: number;
      let y: number;
      let rad: number;

      if (i < 3) {
        // Near bottom bamboo mouth hoop
        ang = -0.55 + i * 0.68;
        y = LANTERN_BOTTOM_Y - 0.06 - i * 0.09 + liftY;
        rad = 0.18 + i * 0.13 * spreadScale;
      } else if (i < 6) {
        // Emerging from left & right waist silk loops
        const isLeft = i === 3;
        ang = isLeft ? Math.PI + 0.12 : -0.18 + (i - 4) * 0.55;
        y = -0.34 + (i - 3) * 0.14 + liftY;
        rad = (ovoidShellRadiusAtY(y - liftY) + 0.18 + bf.orbitSeed * 0.14) * spreadScale;
      } else {
        // Spiraling up the right & upper flanks toward the crown (matches Hero render!)
        const slot = (i - 6) / Math.max(1, quota - 6);
        ang = -0.42 + slot * 1.15 + (i % 2 === 0 ? 0.12 : -0.18);
        y = -0.08 + slot * 1.05 + liftY;
        const shellR = ovoidShellRadiusAtY(y - liftY);
        rad = Math.max(0.55, shellR + 0.24 + slot * 0.22) * spreadScale;
      }

      this.bfOrbitAngle[i] = ang;
      this.bfOrbitRadius[i] = Math.max(0.68, rad);
      this.bfTargetY[i] = y;

      bf.pos[0] = Math.cos(ang) * rad;
      bf.pos[1] = y;
      bf.pos[2] = Math.sin(ang) * rad;
      bf.vel[0] = -Math.sin(ang) * 0.22 * this.bfOrbitDir[i];
      bf.vel[1] = 0.16 + bf.orbitSeed * 0.12;
      bf.vel[2] = Math.cos(ang) * 0.22 * this.bfOrbitDir[i];
      bf.yaw = Math.atan2(bf.vel[0], bf.vel[2]);
      bf.pitch = 0.26;
      bf.roll = 0.12 * this.bfOrbitDir[i];
      bf.scale = this.bfBaseScale[i];
      bf.warmTint = 0.35 + bf.orbitSeed * 0.62;
      bf.emerged = true;
      bf.active = true;
      activeCount++;
    }

    this.state.activeButterflies = activeCount;
  }

  private launchButterflyFromSeam(i: number, boosted: boolean): void {
    const bf = this.butterflies[i];
    const liftY = this.state.lanternLiftY;
    const isMouth = this.bfExitKind[i] === 0;
    const ang = this.bfOrbitAngle[i] + this.state.time * 0.5;

    this.bfMode[i] = BF_MODE_EMERGING;
    this.bfTimer[i] = 0.85 + bf.orbitSeed * 0.35;

    if (isMouth) {
      const r = 0.08 + bf.orbitSeed * 0.14;
      bf.pos[0] = Math.cos(ang) * r;
      bf.pos[1] = LANTERN_BOTTOM_Y - 0.02 + liftY;
      bf.pos[2] = Math.sin(ang) * r;
      bf.vel[0] = Math.cos(ang) * (boosted ? 0.52 : 0.32);
      bf.vel[1] = boosted ? -0.18 : -0.12;
      bf.vel[2] = Math.sin(ang) * (boosted ? 0.52 : 0.32);
    } else {
      const waistY = -0.44 + bf.orbitSeed * 0.34;
      const r = ovoidShellRadiusAtY(waistY) + 0.02;
      bf.pos[0] = Math.cos(ang) * r;
      bf.pos[1] = waistY + liftY;
      bf.pos[2] = Math.sin(ang) * r;
      bf.vel[0] = Math.cos(ang) * (boosted ? 0.58 : 0.36);
      bf.vel[1] = boosted ? 0.26 : 0.14;
      bf.vel[2] = Math.sin(ang) * (boosted ? 0.58 : 0.36);
    }

    bf.yaw = Math.atan2(bf.vel[0], bf.vel[2]);
    bf.pitch = 0.28;
    bf.roll = 0;
    bf.scale = this.bfBaseScale[i] * 0.55;
    bf.warmTint = 0.42 + bf.orbitSeed * 0.55;
    bf.emerged = true;
    bf.active = true;
    this.bfStartle[i] = boosted ? 1.2 : 0.45;
    this.state.butterfliesEmerged += 1;
  }

  // ---------------------------------------------------------------------------
  // Per-frame silk loops, butterflies, and embers
  // ---------------------------------------------------------------------------

  private stepSilkLoops(dt: number): void {
    const count = this.silkLoops.length;
    const u = this.state.unravel;
    const t = this.state.time;
    const liftY = this.state.lanternLiftY;
    const damp = Math.exp(-3.8 * dt);

    for (let i = 0; i < count; i++) {
      const loop = this.silkLoops[i];
      const targetPeel = clamp((u - this.loopUnwindOrder[i] * 0.45) / 0.55, 0, 1);
      loop.peelAmount += (targetPeel - loop.peelAmount) * (1 - Math.exp(-dt * 4.5));

      const ca = Math.cos(loop.anchorAngle);
      const sa = Math.sin(loop.anchorAngle);
      const yAnchor = LANTERN_BOTTOM_Y + loop.anchorV * LANTERN_HEIGHT + liftY;
      const rAnchor = ovoidShellRadiusAtY(yAnchor - liftY) + 0.08 + loop.peelAmount * 0.28;
      const ax = ca * rAnchor;
      const az = sa * rAnchor;

      // Ambient thermal breeze + lantern pendulum coupling
      const breeze = Math.sin(t * 1.45 + loop.phase) * 0.045 * loop.peelAmount;
      const targetX = -sa * breeze - this.state.lanternSwingZ * 0.22 * loop.peelAmount;
      const targetY = (Math.cos(t * 1.15 + loop.phase * 1.3) * 0.032 + 0.025 * this.state.ignition) * loop.peelAmount;
      const targetZ = ca * breeze + this.state.lanternSwingX * 0.22 * loop.peelAmount;

      let fx = (targetX - loop.swayX) * 14.0;
      let fy = (targetY - loop.swayY) * 14.0;
      let fz = (targetZ - loop.swayZ) * 14.0;

      if (this.wakeStrength > 0.03 && loop.peelAmount > 0.05) {
        const dx = ax - this.wakeX;
        const dy = yAnchor - this.wakeY;
        const dz = az - this.wakeZ;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 2.25) {
          const inf = Math.exp(-d2 * 1.5) * this.wakeStrength * loop.peelAmount;
          fx += (this.wakeVelX * 1.6 - dz * 0.9) * inf;
          fy += (this.wakeVelY * 1.4 + 0.35) * inf;
          fz += (this.wakeVelZ * 1.6 + dx * 0.9) * inf;
        }
      }

      this.loopSwayVelX[i] = (this.loopSwayVelX[i] + fx * dt) * damp;
      this.loopSwayVelY[i] = (this.loopSwayVelY[i] + fy * dt) * damp;
      this.loopSwayVelZ[i] = (this.loopSwayVelZ[i] + fz * dt) * damp;

      loop.swayX = clamp(loop.swayX + this.loopSwayVelX[i] * dt, -0.35, 0.35);
      loop.swayY = clamp(loop.swayY + this.loopSwayVelY[i] * dt, -0.3, 0.35);
      loop.swayZ = clamp(loop.swayZ + this.loopSwayVelZ[i] * dt, -0.35, 0.35);
    }
  }

  private stepButterflies(dt: number): void {
    const count = this.butterflies.length;
    const u = this.state.unravel;
    const t = this.state.time;
    const liftY = this.state.lanternLiftY;
    const canEmerge = u >= 0.20;
    const allowedActive = canEmerge
      ? Math.min(count, Math.max(1, Math.round(1 + ((u - 0.20) / 0.72) * (count - 1))))
      : 0;

    let activeCount = 0;

    for (let i = 0; i < count; i++) {
      const bf = this.butterflies[i];
      const mode = this.bfMode[i];

      if (this.bfStartle[i] > 0) {
        this.bfStartle[i] = Math.max(0, this.bfStartle[i] - dt * 1.4);
      }

      // If cocoon is re-bound, draw butterflies cleanly back into the cavity
      if (!canEmerge) {
        if (bf.active) {
          const pull = 1 - Math.exp(-dt * 6.5);
          bf.pos[0] += (0 - bf.pos[0]) * pull;
          bf.pos[1] += (LANTERN_MOUTH_Y + 0.14 - bf.pos[1]) * pull;
          bf.pos[2] += (0 - bf.pos[2]) * pull;
          bf.scale *= Math.exp(-dt * 5.2);
          if (Math.hypot(bf.pos[0], bf.pos[2]) < 0.14) {
            this.bfMode[i] = BF_MODE_DORMANT;
            this.bfTimer[i] = 0.18 + i * 0.2;
            bf.active = false;
            bf.emerged = false;
          } else {
            activeCount++;
          }
        }
        continue;
      }

      if (mode === BF_MODE_DORMANT) {
        if (activeCount < allowedActive) {
          this.bfTimer[i] -= dt * (0.7 + 0.9 * u);
          if (this.bfTimer[i] <= 0) {
            this.launchButterflyFromSeam(i, false);
            activeCount++;
          }
        }
        continue;
      }

      if (mode === BF_MODE_EMERGING) {
        this.bfTimer[i] -= dt;
        const radXZ = Math.hypot(bf.pos[0], bf.pos[2]);
        const dirX = radXZ > 1e-4 ? bf.pos[0] / radXZ : Math.cos(this.bfOrbitAngle[i]);
        const dirZ = radXZ > 1e-4 ? bf.pos[2] / radXZ : Math.sin(this.bfOrbitAngle[i]);

        const shellR = ovoidShellRadiusAtY(bf.pos[1] - liftY);
        const clearR = shellR + 0.24;

        const ax = dirX * 1.85 - dirZ * 0.65 * this.bfOrbitDir[i];
        const ay = 0.52 + 0.25 * this.state.ignition;
        const az = dirZ * 1.85 + dirX * 0.65 * this.bfOrbitDir[i];

        const damp = Math.exp(-3.4 * dt);
        bf.vel[0] = (bf.vel[0] + ax * dt) * damp;
        bf.vel[1] = (bf.vel[1] + ay * dt) * damp;
        bf.vel[2] = (bf.vel[2] + az * dt) * damp;

        this.applyWakeToButterfly(bf, i, dt);

        bf.pos[0] += bf.vel[0] * dt;
        bf.pos[1] += bf.vel[1] * dt;
        bf.pos[2] += bf.vel[2] * dt;

        bf.scale += (this.bfBaseScale[i] - bf.scale) * (1 - Math.exp(-dt * 6.0));

        if (radXZ >= clearR || this.bfTimer[i] <= 0) {
          this.bfMode[i] = BF_MODE_SPIRALING;
          this.bfTimer[i] = 4.8 + bf.orbitSeed * 4.2;
          this.bfOrbitAngle[i] = Math.atan2(bf.pos[2], bf.pos[0]);
          this.bfTargetY[i] = bf.pos[1];
        }
      } else if (mode === BF_MODE_SPIRALING) {
        this.bfTimer[i] -= dt;
        this.bfOrbitAngle[i] += this.bfOrbitDir[i] * (0.68 + 0.42 * bf.orbitSeed) * dt;
        this.bfTargetY[i] += (0.14 + 0.12 * bf.orbitSeed + 0.08 * this.state.ignition) * dt;

        const targetY = this.bfTargetY[i] + Math.sin(t * 1.4 + i * 1.7) * 0.08;
        const shellR = ovoidShellRadiusAtY(targetY - liftY);
        const targetR = Math.max(shellR + 0.22, this.bfOrbitRadius[i] * (0.78 + 0.26 * u)) +
          Math.sin(t * 0.9 + i * 2.1) * 0.08;

        const tx = Math.cos(this.bfOrbitAngle[i]) * targetR;
        const tz = Math.sin(this.bfOrbitAngle[i]) * targetR;

        evalCurlFlow(bf.pos[0] * 1.35, bf.pos[1] * 1.35, bf.pos[2] * 1.35, t * 0.65 + i * 0.37, _curlScratch);

        let ax = (tx - bf.pos[0]) * 4.2 + _curlScratch[0] * 0.36;
        let ay = (targetY - bf.pos[1]) * 3.8 + _curlScratch[1] * 0.28;
        let az = (tz - bf.pos[2]) * 4.2 + _curlScratch[2] * 0.36;

        // Smooth radial clearance repulsion so butterflies never clip through the ovoid lantern
        const localY = bf.pos[1] - liftY;
        if (localY > LANTERN_BOTTOM_Y - 0.06 && localY < LANTERN_TOP_Y + 0.08) {
          const rNow = Math.hypot(bf.pos[0], bf.pos[2]);
          const minClear = ovoidShellRadiusAtY(localY) + 0.14;
          if (rNow < minClear && rNow > 1e-4) {
            const push = (minClear - rNow) * 14.0;
            ax += (bf.pos[0] / rNow) * push;
            az += (bf.pos[2] / rNow) * push;
          }
        }

        const damp = Math.exp(-3.1 * dt);
        bf.vel[0] = (bf.vel[0] + ax * dt) * damp;
        bf.vel[1] = (bf.vel[1] + ay * dt) * damp;
        bf.vel[2] = (bf.vel[2] + az * dt) * damp;

        this.applyWakeToButterfly(bf, i, dt);

        bf.pos[0] += bf.vel[0] * dt;
        bf.pos[1] += bf.vel[1] * dt;
        bf.pos[2] += bf.vel[2] * dt;

        bf.scale += (this.bfBaseScale[i] - bf.scale) * (1 - Math.exp(-dt * 5.0));

        if (bf.pos[1] > LANTERN_TOP_Y + 0.42 + liftY || this.bfTimer[i] <= 0) {
          this.bfMode[i] = BF_MODE_RECIRCULATING;
          this.bfTimer[i] = 1.25;
        }
      } else {
        // BF_MODE_RECIRCULATING: graceful upward fade-out before re-emerging from the lantern
        this.bfTimer[i] -= dt;
        bf.vel[1] = (bf.vel[1] + 0.28 * dt) * Math.exp(-2.2 * dt);
        bf.vel[0] *= Math.exp(-2.4 * dt);
        bf.vel[2] *= Math.exp(-2.4 * dt);

        bf.pos[0] += bf.vel[0] * dt;
        bf.pos[1] += bf.vel[1] * dt;
        bf.pos[2] += bf.vel[2] * dt;

        const fade = clamp(this.bfTimer[i] / 0.75, 0, 1);
        bf.scale = this.bfBaseScale[i] * fade;

        if (this.bfTimer[i] <= 0) {
          this.bfMode[i] = BF_MODE_DORMANT;
          this.bfTimer[i] = 0.2 + bf.orbitSeed * 0.45;
          bf.active = false;
          continue;
        }
      }

      // Smooth flight orientation & climb-coupled wingbeat frequency
      const horizSpeed = Math.hypot(bf.vel[0], bf.vel[2]);
      if (horizSpeed > 0.015) {
        const desiredYaw = Math.atan2(bf.vel[0], bf.vel[2]);
        bf.yaw += angleDiff(desiredYaw, bf.yaw) * (1 - Math.exp(-dt * 7.2));
      }
      const desiredPitch = clamp(0.18 + bf.vel[1] * 0.56, -0.45, 0.76);
      bf.pitch += (desiredPitch - bf.pitch) * (1 - Math.exp(-dt * 6.0));
      const desiredRoll = clamp(-bf.vel[0] * 0.48 + this.bfOrbitDir[i] * 0.16, -0.65, 0.65);
      bf.roll += (desiredRoll - bf.roll) * (1 - Math.exp(-dt * 6.0));

      const wingBoost = Math.max(0, bf.vel[1]) * 7.4 + horizSpeed * 3.2 + this.bfStartle[i] * 6.8;
      bf.wingFreq = this.bfBaseWingFreq[i] + Math.min(11.5, wingBoost);
      bf.wingPhase += dt * bf.wingFreq;

      activeCount++;
    }

    this.state.activeButterflies = activeCount;
  }

  private applyWakeToButterfly(bf: ButterflyState, index: number, dt: number): void {
    if (this.wakeStrength <= 0.03) return;
    const dx = bf.pos[0] - this.wakeX;
    const dy = bf.pos[1] - this.wakeY;
    const dz = bf.pos[2] - this.wakeZ;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < 2.4) {
      const inf = Math.exp(-d2 * 1.4) * this.wakeStrength * dt;
      bf.vel[0] += (this.wakeVelX * 1.15 - dz * 0.75) * inf;
      bf.vel[1] += (this.wakeVelY * 1.05 + 0.24) * inf;
      bf.vel[2] += (this.wakeVelZ * 1.15 + dx * 0.75) * inf;
      this.bfStartle[index] = Math.min(1.5, this.bfStartle[index] + inf * 1.8);
    }
  }

  private seedEmbersForCurrentState(): void {
    const n = this.embers.length;
    for (let i = 0; i < n; i++) {
      const frac = hash01(i * 59.17 + 7.3);
      this.respawnEmber(i, frac);
    }
  }

  private respawnEmber(i: number, initialLifeFrac: number): void {
    const mote = this.embers[i];
    const h1 = hash01(i * 23.17 + this.state.time * 3.1 + 1.9);
    const h2 = hash01(i * 57.41 + this.state.time * 2.3 + 5.7);
    const h3 = hash01(i * 83.29 + this.state.time * 1.7 + 3.3);
    const liftY = this.state.lanternLiftY;
    const ang = h1 * Math.PI * 2;
    const maxLife = 2.2 + h1 * 2.6;

    if (this.state.unravel < 0.22) {
      // Bound chrysalis: keep motes strictly inside the warm lantern cavity
      const r = 0.03 + Math.sqrt(h2) * 0.16;
      mote.pos[0] = Math.cos(ang) * r;
      mote.pos[1] = LANTERN_MOUTH_Y + 0.12 + initialLifeFrac * 0.72 + liftY;
      mote.pos[2] = Math.sin(ang) * r;
      mote.vel[0] = Math.cos(ang) * 0.02;
      mote.vel[1] = 0.12 + h3 * 0.12;
      mote.vel[2] = Math.sin(ang) * 0.02;
    } else {
      const isMouth = this.emberOriginKind[i] === 0;
      const baseY = isMouth
        ? LANTERN_BOTTOM_Y - 0.04 + h3 * 0.12 + liftY
        : -0.45 + h3 * 0.42 + liftY;
      const baseR = isMouth
        ? 0.04 + Math.sqrt(h2) * (LANTERN_MOUTH_RADIUS * 0.95)
        : ovoidShellRadiusAtY(baseY - liftY) + 0.04 + h2 * 0.14;
      const riseSpan = 0.85 + h2 * 0.85;
      const spread = 1 + initialLifeFrac * 0.65;

      mote.pos[0] = Math.cos(ang) * baseR * spread;
      mote.pos[1] = baseY + initialLifeFrac * riseSpan;
      mote.pos[2] = Math.sin(ang) * baseR * spread;
      mote.vel[0] = Math.cos(ang) * (0.04 + h3 * 0.05);
      mote.vel[1] = 0.16 + h2 * 0.22 + 0.14 * this.state.ignition;
      mote.vel[2] = Math.sin(ang) * (0.04 + h3 * 0.05);
    }

    mote.maxLife = maxLife;
    mote.life = initialLifeFrac * maxLife;
    mote.size = 0.013 + Math.pow(h3, 1.4) * 0.026;
    this.emberBurstBoost[i] = 0;
  }

  private stepEmbers(dt: number): void {
    const n = this.embers.length;
    const t = this.state.time;
    const liftY = this.state.lanternLiftY;
    const isBound = this.state.unravel < 0.22;
    const ceilingY = (isBound ? 0.35 : LANTERN_TOP_Y + 0.72) + liftY;
    const thermalDrive = 0.10 + 0.28 * this.state.ignition + 0.22 * this.state.emberPulse;

    for (let i = 0; i < n; i++) {
      const mote = this.embers[i];
      mote.life += dt;

      if (mote.life >= mote.maxLife || mote.pos[1] > ceilingY) {
        this.respawnEmber(i, 0);
        continue;
      }

      if (this.emberBurstBoost[i] > 0) {
        this.emberBurstBoost[i] = Math.max(0, this.emberBurstBoost[i] - dt * 1.6);
      }

      const px = mote.pos[0];
      const py = mote.pos[1];
      const pz = mote.pos[2];

      evalCurlFlow(px * 1.5, py * 1.4, pz * 1.5, t * 0.6 + mote.seed * 6.28, _curlScratch);

      let vx = mote.vel[0] + (_curlScratch[0] * (isBound ? 0.1 : 0.3) + (isBound ? -px * 0.6 : px * 0.04)) * dt;
      let vy = mote.vel[1] + (_curlScratch[1] * 0.18 + thermalDrive + this.emberBurstBoost[i] * 0.28) * dt;
      let vz = mote.vel[2] + (_curlScratch[2] * (isBound ? 0.1 : 0.3) + (isBound ? -pz * 0.6 : pz * 0.04)) * dt;

      if (!isBound && this.wakeStrength > 0.03) {
        const dx = px - this.wakeX;
        const dy = py - this.wakeY;
        const dz = pz - this.wakeZ;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 2.4) {
          const inf = Math.exp(-d2 * 1.5) * this.wakeStrength * dt;
          vx += (this.wakeVelX * 1.4 - dz * 0.9) * inf;
          vy += (this.wakeVelY * 1.3 + 0.24) * inf;
          vz += (this.wakeVelZ * 1.4 + dx * 0.9) * inf;
        }
      }

      const hDamp = Math.exp(-1.2 * dt);
      vx *= hDamp;
      vz *= hDamp;
      vy = clamp(vy * Math.exp(-0.45 * dt), 0.04, 1.05);

      mote.vel[0] = vx;
      mote.vel[1] = vy;
      mote.vel[2] = vz;

      mote.pos[0] = px + vx * dt;
      mote.pos[1] = py + vy * dt;
      mote.pos[2] = pz + vz * dt;
    }
  }
}

/** Smooth ovoid shell radius approximation at lantern-local Y (for collision-free butterfly flight). */
export function ovoidShellRadiusAtY(localY: number): number {
  const v = clamp((localY - LANTERN_BOTTOM_Y) / LANTERN_HEIGHT, 0, 1);
  const bell = Math.sin(Math.pow(v, 0.88) * Math.PI);
  const endR = LANTERN_MOUTH_RADIUS * (1 - v) + 0.20 * v;
  return endR + (LANTERN_MAX_RADIUS - 0.23) * bell;
}

function evalCurlFlow(x: number, y: number, z: number, t: number, out: Float32Array): void {
  const sx = Math.sin(y * 2.1 + t * 1.05) - Math.cos(z * 1.9 - t * 0.92);
  const sy = Math.sin(z * 2.0 + t * 0.85) - Math.cos(x * 2.2 + t * 1.08);
  const sz = Math.sin(x * 1.85 - t * 0.95) - Math.cos(y * 2.25 + t * 0.78);
  out[0] = sx * 0.45;
  out[1] = sy * 0.32 + 0.12;
  out[2] = sz * 0.45;
}

function hash01(n: number): number {
  const s = Math.sin(n * 12.9898 + 78.233) * 43758.5453;
  return s - Math.floor(s);
}

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

function angleDiff(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}
