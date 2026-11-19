import type {
  BellowsSnapshot,
  BellowsStepResult,
  PuffSpawnEvent,
  Vec3Tuple,
  VortexRingState,
} from './types';

export const MAX_VORTEX_RINGS = 6;
export const SAC_CENTER_Y = -0.28;
export const SAC_EQUATOR_RADIUS = 0.52;
export const PERISTOME_NOZZLE_Y = 0.28;
export const PERISTOME_RIM_RADIUS = 0.095;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function normalize3(x: number, y: number, z: number): Vec3Tuple {
  const len = Math.hypot(x, y, z);
  if (len < 1e-6) return [0, 1, 0];
  return [x / len, y / len, z / len];
}

function deterministicHash(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

/**
 * Pure-TS Pneumatic Bellows & Multi-Ring Toroidal Vortex Physics Engine
 * for Day 16 — Spore (Earthstar Puffball / Geastrum triplex).
 *
 * Zero DOM/WebGPU dependencies so every kinematic invariant is unit-testable in Vitest.
 */
export class EarthstarBellowsPhysics {
  compression = 0;
  compressionVel = 0;
  indentStrength = 0;
  indentVel = 0;
  indentDir: Vec3Tuple = [0, 0.35, 0.936];
  pressure = 0;
  peristomeFlare = 0;
  petalRecoil = 0;
  petalRecoilVel = 0;
  hoverWarmth = 0;

  isSqueezing = false;
  squeezeHoldSec = 0;
  squeezeDepthTarget = 0;
  timeSinceLastPuffSec = 0;
  elapsedSec = 0;
  totalPuffs = 0;

  readonly rings: VortexRingState[];
  private nextRingSlot = 0;
  private generationCounter = 0;

  // Scratch event array to avoid per-frame allocations
  private readonly pendingEvents: PuffSpawnEvent[] = [];
  private readonly nozzlePosScratch: Vec3Tuple = [0, PERISTOME_NOZZLE_Y, 0];
  private readonly nozzleAxisScratch: Vec3Tuple = [0, 1, 0];

  constructor() {
    this.rings = Array.from({ length: MAX_VORTEX_RINGS }, (_, slot) => ({
      slot,
      active: false,
      generation: 0,
      ageSec: 0,
      maxAgeSec: 8.5,
      center: [0, PERISTOME_NOZZLE_Y, 0],
      axis: [0, 1, 0],
      speed: 0,
      majorRadius: 0.14,
      coreRadius: 0.08,
      circulation: 0,
      swirlPhase: 0,
      burstStrength: 0,
      seed: deterministicHash(slot + 1),
    }));
  }

  /**
   * Seeds one mature rising toroidal vortex ring right above the peristome
   * on boot so Frame 0 displays the complete Earthstar plume silhouette.
   */
  seedInitialHeroRing(): VortexRingState {
    const r = this.rings[0];
    this.generationCounter += 1;
    this.nextRingSlot = 1 % MAX_VORTEX_RINGS;

    r.active = true;
    r.generation = this.generationCounter;
    r.ageSec = 1.35;
    r.maxAgeSec = 10.5;
    r.center[0] = 0.01;
    r.center[1] = 0.68;
    r.center[2] = 0.008;
    r.axis[0] = 0.028;
    r.axis[1] = 0.999;
    r.axis[2] = -0.02;
    r.speed = 0.095;
    r.majorRadius = 0.36;
    r.coreRadius = 0.092;
    r.circulation = 2.05;
    r.swirlPhase = 2.8;
    r.burstStrength = 0.92;
    r.seed = 0.371;

    this.pressure = 0.28;
    this.peristomeFlare = 0.22;
    this.timeSinceLastPuffSec = 0.5;
    return r;
  }

  /**
   * Begins a tactile press/squeeze at a normalized direction on the papery spore sac.
   * Immediately delivers an initial crisp puff if tapped quickly, or builds
   * elastic compression + deeper pneumatic overpressure while held.
   */
  beginSqueeze(localHitX: number, localHitY: number, localHitZ: number): void {
    this.isSqueezing = true;
    this.squeezeHoldSec = 0;
    this.squeezeDepthTarget = 0.58;

    const dir = normalize3(localHitX, localHitY - SAC_CENTER_Y, localHitZ);
    this.indentDir[0] = dir[0];
    this.indentDir[1] = dir[1];
    this.indentDir[2] = dir[2];

    // Immediate tactile crisp compression kick
    this.compressionVel += 3.8;
    this.indentVel += 5.2;
    this.petalRecoilVel += 1.9;
    this.pressure = clamp(this.pressure + 0.52, 0, 1.65);

    // Fire an immediate responsive starter puff on impact so clicks feel snappy
    this.launchRing(0.76, dir);
  }

  /**
   * Updates the active squeeze point while the user kneads/drags across the papery sac.
   */
  updateSqueezePoint(localHitX: number, localHitY: number, localHitZ: number, dragDelta = 0): void {
    if (!this.isSqueezing) return;
    const dir = normalize3(localHitX, localHitY - SAC_CENTER_Y, localHitZ);
    // Smoothly track kneading across the surface
    this.indentDir[0] += (dir[0] - this.indentDir[0]) * 0.35;
    this.indentDir[1] += (dir[1] - this.indentDir[1]) * 0.35;
    this.indentDir[2] += (dir[2] - this.indentDir[2]) * 0.35;
    const n = normalize3(this.indentDir[0], this.indentDir[1], this.indentDir[2]);
    this.indentDir[0] = n[0];
    this.indentDir[1] = n[1];
    this.indentDir[2] = n[2];

    // Kneading adds small incremental pressure bursts
    if (dragDelta > 0.04) {
      this.pressure = clamp(this.pressure + dragDelta * 0.85, 0, 1.65);
      this.squeezeDepthTarget = clamp(this.squeezeDepthTarget + dragDelta * 0.3, 0.55, 1.0);
    }
  }

  /**
   * Releases the squeeze. If held long enough to build extra pneumatic charge,
   * launches a fuller secondary rebound vortex ring as the papery bellows snaps back.
   */
  endSqueeze(): void {
    if (!this.isSqueezing) return;
    const heldSec = this.squeezeHoldSec;
    this.isSqueezing = false;
    this.squeezeDepthTarget = 0;

    // Sustained squeeze (> 0.24s) launches a rich charged rebound ring on release
    if (heldSec > 0.24) {
      const chargeStrength = clamp(0.68 + heldSec * 0.55, 0.72, 1.25);
      this.launchRing(chargeStrength, this.indentDir);
      this.petalRecoilVel -= 1.6;
    }
  }

  /**
   * Triggers a single discrete puffball tap/burst (used by quick taps, QA API, or subtle idle breath).
   */
  triggerPuff(strength = 0.9, pokeX = 0, pokeZ = 0): VortexRingState {
    const s = clamp(strength, 0.25, 1.35);
    const dir = normalize3(pokeX, 0.45, pokeZ === 0 && pokeX === 0 ? 0.89 : pokeZ);
    this.indentDir[0] = dir[0];
    this.indentDir[1] = dir[1];
    this.indentDir[2] = dir[2];
    this.compressionVel += 4.2 * s;
    this.indentVel += 5.0 * s;
    this.petalRecoilVel += 2.1 * s;
    this.pressure = clamp(this.pressure + 0.65 * s, 0, 1.65);
    return this.launchRing(s, dir);
  }

  private launchRing(strength: number, pokeDir: Vec3Tuple): VortexRingState {
    const slot = this.nextRingSlot;
    this.nextRingSlot = (this.nextRingSlot + 1) % MAX_VORTEX_RINGS;
    this.generationCounter += 1;
    this.totalPuffs += 1;
    this.timeSinceLastPuffSec = 0;

    const ring = this.rings[slot];
    // Off-center squeeze tilts the puff away from the poked flank (like a real bellows!)
    const tiltX = clamp(-pokeDir[0] * 0.22, -0.22, 0.22);
    const tiltZ = clamp(-pokeDir[2] * 0.14, -0.14, 0.14);
    const jitterAngle = deterministicHash(this.generationCounter * 1.73) * Math.PI * 2;
    const axis = normalize3(
      tiltX + Math.cos(jitterAngle) * 0.035,
      1.0,
      tiltZ + Math.sin(jitterAngle) * 0.035
    );

    ring.active = true;
    ring.generation = this.generationCounter;
    ring.ageSec = 0;
    ring.maxAgeSec = 7.4 + strength * 2.0;
    ring.center[0] = axis[0] * 0.04;
    ring.center[1] = PERISTOME_NOZZLE_Y + 0.06;
    ring.center[2] = axis[2] * 0.04;
    ring.axis[0] = axis[0];
    ring.axis[1] = axis[1];
    ring.axis[2] = axis[2];
    ring.speed = 0.56 + strength * 0.36;
    ring.majorRadius = 0.14 + strength * 0.035;
    ring.coreRadius = 0.056 + strength * 0.014;
    ring.circulation = 2.85 + strength * 1.35;
    ring.swirlPhase = 0;
    ring.burstStrength = strength;
    ring.seed = deterministicHash(this.generationCounter * 3.17 + 0.41);

    const particleCount = Math.round(clamp(3200 + strength * 4800, 2600, 9200));
    this.pendingEvents.push({
      ringSlot: slot,
      particleCount,
      burstStrength: strength,
      nozzlePos: [0, PERISTOME_NOZZLE_Y, 0],
      axis: [axis[0], axis[1], axis[2]],
      ringRadius: ring.majorRadius,
      seed: ring.seed,
    });

    return ring;
  }

  /**
   * Advances pneumatic bellows spring-mass physics and multi-ring Biot-Savart
   * kinematics by physical `dtSec`.
   */
  step(dtSec: number, isHoveringSac = false): BellowsStepResult {
    const dt = clamp(dtSec, 0.001, 0.05);
    this.elapsedSec += dt;
    this.timeSinceLastPuffSec += dt;

    // Hover readiness warmth
    const hoverTarget = isHoveringSac || this.isSqueezing ? 1.0 : 0.0;
    this.hoverWarmth += (hoverTarget - this.hoverWarmth) * (1 - Math.exp(-dt * 8.5));

    // Hold-squeeze pneumatic charge
    if (this.isSqueezing) {
      this.squeezeHoldSec += dt;
      this.squeezeDepthTarget = clamp(0.58 + this.squeezeHoldSec * 0.55, 0.58, 0.96);
      this.pressure = clamp(this.pressure + dt * 0.42, 0, 1.65);
    }

    // Damped spring-mass for global papery bellows compression (volume-conserving elastic recoil)
    const kComp = 44.0;
    const cComp = 8.4;
    const compForce = (this.squeezeDepthTarget - this.compression) * kComp - this.compressionVel * cComp;
    this.compressionVel += compForce * dt;
    this.compression = clamp(this.compression + this.compressionVel * dt, -0.14, 1.05);

    // Localized thumb indentation dimple spring (slightly crisper recovery)
    const indentTarget = this.isSqueezing ? this.squeezeDepthTarget : 0;
    const kInd = 56.0;
    const cInd = 9.2;
    const indForce = (indentTarget - this.indentStrength) * kInd - this.indentVel * cInd;
    this.indentVel += indForce * dt;
    this.indentStrength = clamp(this.indentStrength + this.indentVel * dt, -0.1, 1.05);

    // Sympathetic exoperidium petal brace & elastic spring-back
    const petalTarget = this.compression * 0.34;
    const kPet = 28.0;
    const cPet = 6.2;
    const petalForce = (petalTarget - this.petalRecoil) * kPet - this.petalRecoilVel * cPet;
    this.petalRecoilVel += petalForce * dt;
    this.petalRecoil = clamp(this.petalRecoil + this.petalRecoilVel * dt, -0.22, 0.45);

    // Internal pneumatic cavity pressure venting through the pleated peristome
    const ventRate = this.isSqueezing ? 0.95 : 1.65;
    this.pressure *= Math.exp(-dt * ventRate);
    if (this.pressure < 1e-4) this.pressure = 0;

    // Fluted peristome rim opens cleanly with cavity overpressure
    const flareTarget = clamp(this.pressure * 0.85, 0, 1.0);
    this.peristomeFlare += (flareTarget - this.peristomeFlare) * (1 - Math.exp(-dt * 16.0));

    // Gentle autonomous breathing whisper if idle for > 6.8s so the specimen never looks dead
    if (!this.isSqueezing && this.timeSinceLastPuffSec > 6.8) {
      this.triggerPuff(0.48, Math.sin(this.elapsedSec * 0.7) * 0.15, Math.cos(this.elapsedSec * 0.5) * 0.15);
    }

    // Advance all active toroidal vortex rings + mutual ring-ring leapfrogging induction
    for (let i = 0; i < MAX_VORTEX_RINGS; i++) {
      const r = this.rings[i];
      if (!r.active) continue;

      r.ageSec += dt;
      if (r.ageSec >= r.maxAgeSec) {
        r.active = false;
        r.circulation = 0;
        continue;
      }

      const lifeFrac = r.ageSec / r.maxAgeSec;

      // Mutual induction from other active rings (Biot-Savart leapfrogging approximation)
      let axialBoost = 0;
      let radialRateMod = 0;
      for (let j = 0; j < MAX_VORTEX_RINGS; j++) {
        if (i === j) continue;
        const other = this.rings[j];
        if (!other.active) continue;

        const dx = r.center[0] - other.center[0];
        const dy = r.center[1] - other.center[1];
        const dz = r.center[2] - other.center[2];
        const distSq = dx * dx + dy * dy + dz * dz + 0.08;
        const axialSep = dy; // positive if r is above other

        // A ring directly behind (below) a leading ring accelerates through its center & contracts;
        // the leading ring slows slightly and expands.
        const horizDistSq = dx * dx + dz * dz;
        if (horizDistSq < 0.45) {
          const proximity = Math.exp(-distSq * 2.8);
          if (axialSep < 0) {
            // r is chasing other from below -> accelerate and slightly pinch radius
            axialBoost += other.circulation * 0.085 * proximity;
            radialRateMod -= 0.075 * proximity;
          } else {
            // r is leading -> slow slightly and expand
            axialBoost -= other.circulation * 0.032 * proximity;
            radialRateMod += 0.065 * proximity;
          }
        }
      }

      // Viscous entrainment drag on self-propagation speed + gentle studio buoyancy
      const drag = Math.exp(-dt * (0.72 + lifeFrac * 0.52));
      r.speed = Math.max(0.055, r.speed * drag + axialBoost * dt);

      // Gentle buoyant meander as circulation ages
      const meander = lifeFrac * lifeFrac * 0.035;
      const vx = r.axis[0] * r.speed + Math.sin(this.elapsedSec * 0.75 + r.seed * 19.0) * meander;
      const vy = r.axis[1] * r.speed + 0.036 * (1.0 - lifeFrac * 0.45);
      const vz = r.axis[2] * r.speed + Math.cos(this.elapsedSec * 0.63 + r.seed * 23.0) * meander;

      r.center[0] += vx * dt;
      r.center[1] += vy * dt;
      r.center[2] += vz * dt;

      // Entrainment radial expansion R(t) and restrained core diffusion a(t) so ring stays sculptural
      const expandRate = (0.16 / (1.0 + r.ageSec * 1.05)) + radialRateMod;
      r.majorRadius = clamp(r.majorRadius + expandRate * dt, 0.12, 0.64);
      r.coreRadius = clamp(r.coreRadius + (0.028 / (1.0 + r.ageSec * 0.85)) * dt, 0.052, 0.185);

      // Circulation decays smoothly as laminar ring breaks into buoyant curl wisps
      const circFade = Math.max(0, 1.0 - Math.pow(lifeFrac, 1.65));
      const baseCirc = (2.85 + r.burstStrength * 1.35) * Math.exp(-r.ageSec * 0.22);
      r.circulation = baseCirc * circFade;
      r.swirlPhase += dt * (1.8 + r.circulation * 0.65);
    }

    // Drain pending events into a snapshot array for this step
    const puffEvents = this.pendingEvents.splice(0, this.pendingEvents.length);

    // Continuous trailing stem wisps venting out of the peristome chimney
    const pressureEmit = Math.round(this.pressure * 220);
    const ambientEmit = 48; // steady delicate filament rising out of the golden vent
    const continuousEmitCount = pressureEmit + ambientEmit;

    this.nozzlePosScratch[0] = 0;
    this.nozzlePosScratch[1] = PERISTOME_NOZZLE_Y - this.compression * 0.085;
    this.nozzlePosScratch[2] = 0;

    // Slight nozzle sway coupled to poke direction
    const ax = normalize3(-this.indentDir[0] * this.indentStrength * 0.16, 1.0, -this.indentDir[2] * this.indentStrength * 0.12);
    this.nozzleAxisScratch[0] = ax[0];
    this.nozzleAxisScratch[1] = ax[1];
    this.nozzleAxisScratch[2] = ax[2];

    return {
      puffEvents,
      continuousEmitCount,
      nozzlePos: this.nozzlePosScratch,
      nozzleAxis: this.nozzleAxisScratch,
    };
  }

  getActiveRingCount(): number {
    let count = 0;
    for (let i = 0; i < MAX_VORTEX_RINGS; i++) {
      if (this.rings[i].active) count++;
    }
    return count;
  }

  getSnapshot(): BellowsSnapshot {
    return {
      compression: this.compression,
      indentStrength: this.indentStrength,
      indentDir: [this.indentDir[0], this.indentDir[1], this.indentDir[2]],
      pressure: this.pressure,
      peristomeFlare: this.peristomeFlare,
      petalRecoil: this.petalRecoil,
      isSqueezing: this.isSqueezing,
      totalPuffs: this.totalPuffs,
      activeRingCount: this.getActiveRingCount(),
      rings: this.rings,
    };
  }
}
