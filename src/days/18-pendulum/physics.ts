import * as THREE from 'three/webgpu';

export const DOME_RADIUS = 0.94;
export const DOME_MAX_PHI = 1.31; // ~75.1 deg wide architectural hemisphere cap
export const DOME_HEIGHT = DOME_RADIUS * (1 - Math.cos(DOME_MAX_PHI));
export const DOME_CENTER_Y = -0.82;
export const DOME_CENTER = new THREE.Vector3(0, DOME_CENTER_Y, 0);
export const DOME_RIM_Y = DOME_CENTER_Y + DOME_RADIUS * Math.cos(DOME_MAX_PHI);

// Overhead horological gimbal arch & differential pulley constants
export const PIVOT_POS = new THREE.Vector3(0, 1.48, 0);
export const PULLEY_AXLE_Y = 1.40;
export const PULLEY_OUTER_RADIUS = 0.060; // outer V-groove wire track radius (bob plumb line)
export const PULLEY_INNER_RADIUS = 0.019; // coaxial inner drum radius (counterweight take-up, 3.16:1 ratio)
export const PULLEY_OUTER_OFFSET_X = 0.022; // lateral groove track offset in swivel frame
export const PULLEY_INNER_OFFSET_X = -0.024; // lateral inner drum offset in swivel frame
export const WIRE_REF_LENGTH = 1.85; // reference max pay-out length where counterweight sits highest
export const COUNTERWEIGHT_TOP_Y = 1.33; // counterweight top eyelet Y at WIRE_REF_LENGTH
export const COUNTERWEIGHT_LENGTH = 0.112; // physical height of turned brass counterweight cylinder

// Bob geometry dimensions & safe grazing shell
export const BOB_TOP_COLLET_OFFSET = 0.172;
export const BOB_TIP_OFFSET = 0.168;
export const BOB_CLEARANCE = 0.268; // bob center above dome surface -> tip clears iron & stone crowns
export const BOB_SHELL_RADIUS = DOME_RADIUS + BOB_CLEARANCE;

export const MAX_STONES = 18;
export const STONE_RADIUS = 0.068;
export const STONE_CROWN_HEIGHT = 0.076;
export const STONE_SEAT_OFFSET = 0.002; // flush seating on curved iron (base skirt embeds ~4mm)
export const TRAIL_LENGTH = 96; // ~1.6 s continuous arc wake filament

export type StonePolarity = 1 | -1; // +1 = Cinnabar (repel bob), -1 = Verdigris (attract bob)
export type StonePhase = 'flying' | 'sliding' | 'clung';

export interface DomeImpactEvent {
  normal: THREE.Vector3;
  strength: number;
  polarity: StonePolarity;
}

export interface DomeScuffEvent {
  normal: THREE.Vector3;
  polarity: StonePolarity;
  intensity: number;
  isImpact: boolean;
}

export interface PulleyKinematicsFrame {
  azimuth: number;
  wheelAngle: number;
  bobTangentWorld: THREE.Vector3;
  bobColletWorld: THREE.Vector3;
  cwTangentWorld: THREE.Vector3;
  cwEyeletWorld: THREE.Vector3;
  departAngleLocal: number; // angle on outer sheave where bob wire departs tangentially
  wireLength: number;
  pluckPhase: number;
  pluckAmp: number;
}

export interface MagneticStone {
  id: number;
  variant: number;
  polarity: StonePolarity;
  prevPolarity: StonePolarity;
  phase: StonePhase;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  normal: THREE.Vector3;
  quat: THREE.Quaternion;
  yaw: number;
  spinAxis: THREE.Vector3;
  spinSpeed: number;
  scale: number;
  seed: number;
  age: number;
  bouncesLeft: number;
  flipTimer: number;
  flipDuration: number;
  flipSwapped: boolean;
  flipAxis: THREE.Vector3;
  flipStartQuat: THREE.Quaternion;
  hoverPoise: number;
  pulseIntensity: number;
  tugTilt: THREE.Vector3;
  tugTiltVel: THREE.Vector3;
}

export function getStoneVisualPolarity(stone: MagneticStone): StonePolarity {
  return stone.flipTimer > stone.flipDuration * 0.5 ? stone.prevPolarity : stone.polarity;
}

export interface PendulumState {
  dir: THREE.Vector3;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  radialOffset: number;
  radialVel: number;
  spinAngle: number;
  spinVel: number;
  quat: THREE.Quaternion;
  tipPos: THREE.Vector3;
  magneticForceMag: number;
  isGrabbed: boolean;
  grabTarget: THREE.Vector3;
  dragVelEma: THREE.Vector3;
  draggedStoneId: number;
  draggedStoneTarget: THREE.Vector3;
  domeRecoilY: number;
  domeRecoilVel: number;
  domeTiltX: number;
  domeTiltZ: number;
  domeTiltVelX: number;
  domeTiltVelZ: number;
  archSwayX: number;
  archSwayZ: number;
  archSwayVelX: number;
  archSwayVelZ: number;
  magTipCant: THREE.Vector3;
  magTipCantVel: THREE.Vector3;
  // Differential horological pulley & counterweight state
  pulleyAzimuth: number;
  pulleySwivelVel: number;
  pulleyWheelAngle: number;
  wireLength: number;
  counterweightY: number;
  cwSwayAngle: number;
  cwSwayVel: number;
  wirePluckAmp: number;
  wirePluckPhase: number;
}

// Pre-allocated scratch vectors & quaternions (strictly zero per-frame allocations)
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3();
const _v5 = new THREE.Vector3();
const _v6 = new THREE.Vector3();
const _polePos = new THREE.Vector3();
const _crownPos = new THREE.Vector3();
const _wireVec = new THREE.Vector3();
const _wireDir = new THREE.Vector3();
const _fTotal = new THREE.Vector3();
const _fPar = new THREE.Vector3();
const _fPerp = new THREE.Vector3();
const _fTipCant = new THREE.Vector3();
const _bobUp = new THREE.Vector3(0, 1, 0);
const _up = new THREE.Vector3(0, 1, 0);
const _tangentForce = new THREE.Vector3();
const _magForce = new THREE.Vector3();
const _q0 = new THREE.Quaternion();
const _q1 = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _qFlip = new THREE.Quaternion();
const _prevVel = new THREE.Vector3();

function seededRand(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function computeClungQuaternion(
  normal: THREE.Vector3,
  yawRad: number,
  outQuat: THREE.Quaternion
): THREE.Quaternion {
  _q0.setFromUnitVectors(_up, normal);
  _q1.setFromAxisAngle(normal, yawRad);
  outQuat.multiplyQuaternions(_q1, _q0);
  return outQuat;
}

export class DomePendulumSimulation {
  readonly stones: MagneticStone[] = [];
  readonly pendulum: PendulumState;
  readonly trailPositions: Float32Array;
  readonly trailSpeeds: Float32Array;
  readonly trailCharge: Float32Array;
  trailHead = 0;
  trailCount = 0;
  simTime = 0;

  // Bounded zero-allocation event queues drained each frame by main.ts
  readonly impactEvents: DomeImpactEvent[] = [];
  readonly scuffEvents: DomeScuffEvent[] = [];
  private impactPoolCursor = 0;
  private scuffPoolCursor = 0;
  private readonly impactPool: DomeImpactEvent[] = [];
  private readonly scuffPool: DomeScuffEvent[] = [];

  private nextStoneId = 1;
  private rngCounter = 29;
  private trailAccum = 0;
  private lastSignedInfluence = 0;
  private nearestVerdigrisId = -1;
  private singleStoneDwellTimer = 0;
  private lastBobCollet = new THREE.Vector3();
  private lastDepartWorld = new THREE.Vector3(0, PULLEY_AXLE_Y, 0);

  constructor() {
    for (let i = 0; i < 12; i++) {
      this.impactPool.push({ normal: new THREE.Vector3(0, 1, 0), strength: 0, polarity: 1 });
    }
    for (let i = 0; i < 48; i++) {
      this.scuffPool.push({
        normal: new THREE.Vector3(0, 1, 0),
        polarity: 1,
        intensity: 0,
        isImpact: false,
      });
    }

    const initDir = new THREE.Vector3(0.68, 0.66, 0.32).normalize();
    const initAz = Math.atan2(initDir.x, initDir.z);

    this.pendulum = {
      dir: initDir,
      pos: new THREE.Vector3(),
      vel: new THREE.Vector3(-0.55, 0.20, 1.38),
      radialOffset: 0.025,
      radialVel: 0,
      spinAngle: 0,
      spinVel: 0.35,
      quat: new THREE.Quaternion(),
      tipPos: new THREE.Vector3(),
      magneticForceMag: 0,
      isGrabbed: false,
      grabTarget: new THREE.Vector3(),
      dragVelEma: new THREE.Vector3(),
      draggedStoneId: -1,
      draggedStoneTarget: new THREE.Vector3(),
      domeRecoilY: 0,
      domeRecoilVel: 0,
      domeTiltX: 0,
      domeTiltZ: 0,
      domeTiltVelX: 0,
      domeTiltVelZ: 0,
      archSwayX: 0,
      archSwayZ: 0,
      archSwayVelX: 0,
      archSwayVelZ: 0,
      magTipCant: new THREE.Vector3(),
      magTipCantVel: new THREE.Vector3(),
      pulleyAzimuth: initAz,
      pulleySwivelVel: 0,
      pulleyWheelAngle: 0,
      wireLength: 1.5,
      counterweightY: 1.12,
      cwSwayAngle: 0,
      cwSwayVel: 0,
      wirePluckAmp: 0,
      wirePluckPhase: 0,
    };

    this.trailPositions = new Float32Array(TRAIL_LENGTH * 3);
    this.trailSpeeds = new Float32Array(TRAIL_LENGTH);
    this.trailCharge = new Float32Array(TRAIL_LENGTH);

    this.syncPendulumPosFromDir();
    this.updatePulleyState(0.016, true);
    this.updateBobOrientation(1.0);
    this.updateBobTipPos();
    this.seedInitialConstellation();
    this.resetTrail();
  }

  private nextRand(): number {
    this.rngCounter += 1;
    return seededRand(this.rngCounter * 1.6180339);
  }

  private emitImpact(normal: THREE.Vector3, strength: number, polarity: StonePolarity): void {
    const clampedStrength = Math.min(1.0, Math.max(0.05, strength));
    // Excite weighty gyroscopic 2-axis dome rock from off-center stone impacts
    this.pendulum.domeTiltVelX += normal.z * clampedStrength * 0.115;
    this.pendulum.domeTiltVelZ -= normal.x * clampedStrength * 0.115;
    if (this.impactPoolCursor >= this.impactPool.length) return;
    const ev = this.impactPool[this.impactPoolCursor++];
    ev.normal.copy(normal);
    ev.strength = clampedStrength;
    ev.polarity = polarity;
    this.impactEvents.push(ev);
  }

  private emitScuff(
    normal: THREE.Vector3,
    polarity: StonePolarity,
    intensity: number,
    isImpact: boolean
  ): void {
    if (this.scuffPoolCursor >= this.scuffPool.length) return;
    const ev = this.scuffPool[this.scuffPoolCursor++];
    ev.normal.copy(normal);
    ev.polarity = polarity;
    ev.intensity = Math.min(1.0, Math.max(0.02, intensity));
    ev.isImpact = isImpact;
    this.scuffEvents.push(ev);
  }

  clearSurfaceEvents(): void {
    this.impactEvents.length = 0;
    this.scuffEvents.length = 0;
    this.impactPoolCursor = 0;
    this.scuffPoolCursor = 0;
  }

  private syncPendulumPosFromDir(): void {
    const r = BOB_SHELL_RADIUS + this.pendulum.radialOffset;
    this.pendulum.pos.copy(this.pendulum.dir).multiplyScalar(r).add(DOME_CENTER);
  }

  private updateBobTipPos(): void {
    _v0.set(0, -BOB_TIP_OFFSET, 0).applyQuaternion(this.pendulum.quat);
    this.pendulum.tipPos.copy(this.pendulum.pos).add(_v0);
  }

  /**
   * Computes minimum safe radial offset so the bob's sharp brass plumb tip
   * strictly clears both bare iron and every seated magnetite crystal crown.
   */
  private computeMinSafeRadialOffset(dir: THREE.Vector3): number {
    const minOverIron = DOME_RADIUS + BOB_TIP_OFFSET + 0.044 - BOB_SHELL_RADIUS;
    let minOffset = minOverIron;

    for (let i = 0; i < this.stones.length; i++) {
      const st = this.stones[i];
      if (st.phase === 'flying') continue;
      const cosAng = Math.max(-1, Math.min(1, dir.dot(st.normal)));
      const ang = Math.acos(cosAng);
      const stoneAngRadius = (STONE_RADIUS * st.scale + 0.095) / DOME_RADIUS;
      if (ang < stoneAngRadius * 1.45) {
        const proximity = Math.max(0, 1 - ang / (stoneAngRadius * 1.45));
        const smoothProx = proximity * proximity * (3 - 2 * proximity);
        const stoneCrownR = DOME_RADIUS + STONE_SEAT_OFFSET + STONE_CROWN_HEIGHT * st.scale;
        const neededR = stoneCrownR + BOB_TIP_OFFSET + 0.034;
        const neededOffset =
          (neededR - BOB_SHELL_RADIUS) * smoothProx + minOverIron * (1 - smoothProx);
        if (neededOffset > minOffset) {
          minOffset = neededOffset;
        }
      }
    }

    return minOffset;
  }

  seedInitialConstellation(): void {
    this.stones.length = 0;
    const presets: Array<{ phi: number; theta: number; polarity: StonePolarity; scale: number }> = [
      { phi: 0.34, theta: 0.14, polarity: -1, scale: 1.04 },
      { phi: 0.72, theta: 0.62, polarity: 1, scale: 1.02 },
      { phi: 0.86, theta: -0.56, polarity: -1, scale: 1.00 },
      { phi: 0.62, theta: -1.16, polarity: 1, scale: 0.94 },
      { phi: 0.56, theta: 1.38, polarity: -1, scale: 0.96 },
      { phi: 0.96, theta: -1.68, polarity: 1, scale: 0.98 },
      { phi: 0.50, theta: -2.28, polarity: 1, scale: 0.95 },
      { phi: 0.66, theta: 2.48, polarity: -1, scale: 1.01 },
      { phi: 0.94, theta: 0.08, polarity: 1, scale: 0.92 },
    ];

    for (let i = 0; i < presets.length; i++) {
      const p = presets[i];
      const normal = new THREE.Vector3(
        Math.sin(p.phi) * Math.sin(p.theta),
        Math.cos(p.phi),
        Math.sin(p.phi) * Math.cos(p.theta)
      ).normalize();
      const pos = normal.clone().multiplyScalar(DOME_RADIUS + STONE_SEAT_OFFSET).add(DOME_CENTER);
      const yaw = this.nextRand() * Math.PI * 2;
      const quat = new THREE.Quaternion();
      computeClungQuaternion(normal, yaw, quat);

      this.stones.push({
        id: this.nextStoneId++,
        variant: i % 4,
        polarity: p.polarity,
        prevPolarity: p.polarity,
        phase: 'clung',
        pos,
        vel: new THREE.Vector3(),
        normal,
        quat,
        yaw,
        spinAxis: new THREE.Vector3(1, 0, 0),
        spinSpeed: 0,
        scale: p.scale,
        seed: this.nextRand() * 100,
        age: 2.0,
        bouncesLeft: 0,
        flipTimer: 0,
        flipDuration: 0.34,
        flipSwapped: false,
        flipAxis: new THREE.Vector3(1, 0, 0),
        flipStartQuat: quat.clone(),
        hoverPoise: 0,
        pulseIntensity: 0,
        tugTilt: new THREE.Vector3(),
        tugTiltVel: new THREE.Vector3(),
      });
    }
  }

  tossStoneToward(
    targetOnDome: THREE.Vector3,
    launchOrigin?: THREE.Vector3,
    forcedPolarity?: StonePolarity
  ): MagneticStone {
    if (this.stones.length >= MAX_STONES) {
      const removed = this.stones.shift();
      if (removed && removed.id === this.pendulum.draggedStoneId) {
        this.pendulum.draggedStoneId = -1;
      }
    }

    let cinnabarCount = 0;
    let verdigrisCount = 0;
    for (const s of this.stones) {
      if (s.polarity === 1) cinnabarCount++;
      else verdigrisCount++;
    }
    const polarity: StonePolarity =
      forcedPolarity ??
      (cinnabarCount > verdigrisCount + 1
        ? -1
        : verdigrisCount > cinnabarCount + 1
          ? 1
          : this.nextRand() > 0.5
            ? 1
            : -1);

    const targetNormal = _v0.copy(targetOnDome).sub(DOME_CENTER).normalize();
    const minCos = Math.cos(DOME_MAX_PHI - 0.12);
    if (targetNormal.y < minCos) {
      const horiz = Math.hypot(targetNormal.x, targetNormal.z);
      const targetHoriz = Math.sqrt(Math.max(0.001, 1 - minCos * minCos));
      if (horiz > 1e-4) {
        targetNormal.x = (targetNormal.x / horiz) * targetHoriz;
        targetNormal.z = (targetNormal.z / horiz) * targetHoriz;
      } else {
        targetNormal.x = targetHoriz;
        targetNormal.z = 0;
      }
      targetNormal.y = minCos;
      targetNormal.normalize();
    }

    const landingPos = targetNormal
      .clone()
      .multiplyScalar(DOME_RADIUS + STONE_SEAT_OFFSET)
      .add(DOME_CENTER);

    const origin = launchOrigin
      ? launchOrigin.clone()
      : new THREE.Vector3(
          landingPos.x * 1.32 + (this.nextRand() - 0.5) * 0.4,
          landingPos.y + 0.94 + this.nextRand() * 0.24,
          landingPos.z * 1.32 + 0.82
        );

    const flightTime = 0.28 + this.nextRand() * 0.1;
    const gravity = 10.4;
    const vel = new THREE.Vector3(
      (landingPos.x - origin.x) / flightTime,
      (landingPos.y - origin.y + 0.5 * gravity * flightTime * flightTime) / flightTime,
      (landingPos.z - origin.z) / flightTime
    );

    const spinAxis = new THREE.Vector3(
      this.nextRand() - 0.5,
      this.nextRand() - 0.5,
      this.nextRand() - 0.5
    ).normalize();

    const yaw = this.nextRand() * Math.PI * 2;
    const id = this.nextStoneId++;
    const stone: MagneticStone = {
      id,
      variant: id % 4,
      polarity,
      prevPolarity: polarity,
      phase: 'flying',
      pos: origin,
      vel,
      normal: targetNormal.clone(),
      quat: new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          this.nextRand() * Math.PI * 2,
          this.nextRand() * Math.PI * 2,
          this.nextRand() * Math.PI * 2
        )
      ),
      yaw,
      spinAxis,
      spinSpeed: 8.5 + this.nextRand() * 7.0,
      scale: 0.90 + this.nextRand() * 0.20,
      seed: this.nextRand() * 100,
      age: 0,
      bouncesLeft: 1,
      flipTimer: 0,
      flipDuration: 0.34,
      flipSwapped: false,
      flipAxis: new THREE.Vector3(1, 0, 0),
      flipStartQuat: new THREE.Quaternion(),
      hoverPoise: 0,
      pulseIntensity: 0.65,
      tugTilt: new THREE.Vector3(),
      tugTiltVel: new THREE.Vector3(),
    };

    this.stones.push(stone);

    const speed = this.pendulum.vel.length();
    if (speed < 0.72 && !this.pendulum.isGrabbed) {
      _v1.crossVectors(this.pendulum.dir, _up);
      if (_v1.lengthSq() < 1e-4) _v1.set(1, 0, 0);
      _v1.normalize().multiplyScalar(0.82);
      this.pendulum.vel.add(_v1);
    }

    return stone;
  }

  castHandful(count = 9): void {
    this.stones.length = 0;
    this.pendulum.draggedStoneId = -1;
    const baseAngle = this.nextRand() * Math.PI * 2;
    for (let i = 0; i < count; i++) {
      const theta = baseAngle + (i / count) * Math.PI * 2 + (this.nextRand() - 0.5) * 0.58;
      const phi = 0.24 + this.nextRand() * (DOME_MAX_PHI - 0.38);
      const target = new THREE.Vector3(
        Math.sin(phi) * Math.sin(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.cos(theta)
      )
        .multiplyScalar(DOME_RADIUS)
        .add(DOME_CENTER);

      const origin = new THREE.Vector3(
        Math.sin(theta) * (0.98 + this.nextRand() * 0.42),
        1.02 + i * 0.1 + this.nextRand() * 0.18,
        Math.cos(theta) * (0.98 + this.nextRand() * 0.42)
      );
      const polarity: StonePolarity = i % 2 === 0 ? 1 : -1;
      this.tossStoneToward(target, origin, polarity);
    }

    if (!this.pendulum.isGrabbed) {
      const kickAngle = baseAngle + 0.85;
      this.pendulum.vel.set(Math.cos(kickAngle) * 1.48, 0.24, Math.sin(kickAngle) * 1.48);
    }
  }

  flipStone(stoneId: number): boolean {
    const stone = this.stones.find((s) => s.id === stoneId);
    if (!stone || stone.phase === 'flying') return false;

    stone.prevPolarity = getStoneVisualPolarity(stone);
    stone.polarity = stone.polarity === 1 ? -1 : 1;
    stone.flipDuration = 0.34;
    stone.flipTimer = stone.flipDuration;
    stone.flipSwapped = true;
    stone.flipStartQuat.copy(stone.quat);

    _v0.crossVectors(stone.normal, _up);
    if (_v0.lengthSq() < 1e-4) _v0.set(1, 0, 0);
    stone.flipAxis.copy(_v0).normalize();

    stone.pulseIntensity = 1.0;
    stone.phase = 'sliding';

    for (const other of this.stones) {
      if (other.id !== stone.id && other.phase === 'clung') {
        if (other.pos.distanceTo(stone.pos) < 0.65) {
          other.phase = 'sliding';
        }
      }
    }

    const distToBob = stone.pos.distanceTo(this.pendulum.pos);
    if (distToBob < 1.05 && !this.pendulum.isGrabbed) {
      _v0.copy(this.pendulum.pos).sub(stone.pos).normalize();
      this.pendulum.vel.addScaledVector(_v0, stone.polarity * 0.88);
      this.pendulum.wirePluckAmp = Math.min(1.0, this.pendulum.wirePluckAmp + 0.55);
    }

    this.emitImpact(stone.normal, 0.55, stone.polarity);
    return true;
  }

  setDraggedStone(stoneId: number, targetOnDome: THREE.Vector3): boolean {
    const stone = this.stones.find((s) => s.id === stoneId);
    if (!stone || stone.phase === 'flying') {
      this.pendulum.draggedStoneId = -1;
      return false;
    }

    _v0.copy(targetOnDome).sub(DOME_CENTER);
    if (_v0.lengthSq() < 1e-5) {
      _v0.copy(stone.normal);
    } else {
      _v0.normalize();
    }

    const minCos = Math.cos(DOME_MAX_PHI - 0.12);
    if (_v0.y < minCos) {
      const horiz = Math.hypot(_v0.x, _v0.z);
      const tHoriz = Math.sqrt(Math.max(0.001, 1 - minCos * minCos));
      if (horiz > 1e-4) {
        _v0.x = (_v0.x / horiz) * tHoriz;
        _v0.z = (_v0.z / horiz) * tHoriz;
      } else {
        _v0.x = tHoriz;
        _v0.z = 0;
      }
      _v0.y = minCos;
      _v0.normalize();
    }

    this.pendulum.draggedStoneId = stoneId;
    this.pendulum.draggedStoneTarget
      .copy(_v0)
      .multiplyScalar(DOME_RADIUS + STONE_SEAT_OFFSET)
      .add(DOME_CENTER);
    stone.phase = 'sliding';
    return true;
  }

  releaseDraggedStone(): void {
    this.pendulum.draggedStoneId = -1;
  }

  resetTrail(): void {
    this.trailHead = 0;
    this.trailCount = 0;
    this.trailAccum = 0;
    for (let i = 0; i < TRAIL_LENGTH; i++) {
      this.trailPositions[i * 3] = this.pendulum.tipPos.x;
      this.trailPositions[i * 3 + 1] = this.pendulum.tipPos.y;
      this.trailPositions[i * 3 + 2] = this.pendulum.tipPos.z;
      this.trailSpeeds[i] = 0;
      this.trailCharge[i] = 0;
    }
  }

  step(dtSec: number): void {
    const clampedDt = Math.min(0.05, Math.max(0.001, dtSec));
    const substeps = 4;
    const h = clampedDt / substeps;

    for (let s = 0; s < substeps; s++) {
      this.simTime += h;
      this.stepStones(h);
      this.stepPendulum(h);
      this.stepDomeRecoil(h);
    }

    this.trailAccum += clampedDt;
    if (this.trailAccum >= 1 / 60) {
      this.trailAccum = 0;
      const idx = this.trailHead;
      _v0.copy(this.pendulum.tipPos).sub(DOME_CENTER);
      const rTip = _v0.length();
      const minTrailR = DOME_RADIUS + 0.036;
      if (rTip < minTrailR && rTip > 1e-5) {
        _v0.multiplyScalar(minTrailR / rTip);
      }
      _v0.add(DOME_CENTER);

      this.trailPositions[idx * 3] = _v0.x;
      this.trailPositions[idx * 3 + 1] = _v0.y;
      this.trailPositions[idx * 3 + 2] = _v0.z;
      this.trailSpeeds[idx] = this.pendulum.vel.length();
      this.trailCharge[idx] = this.lastSignedInfluence;
      this.trailHead = (this.trailHead + 1) % TRAIL_LENGTH;
      if (this.trailCount < TRAIL_LENGTH) this.trailCount++;
    }
  }

  private stepDomeRecoil(h: number): void {
    const p = this.pendulum;
    const k = 190.0;
    const c = 17.5;
    const acc = -k * p.domeRecoilY - c * p.domeRecoilVel;
    p.domeRecoilVel += acc * h;
    p.domeRecoilY += p.domeRecoilVel * h;

    // Weighty gyroscopic 2-axis dome rock from off-center stone impacts
    const kTilt = 145.0;
    const cTilt = 14.0;
    const accTX = -kTilt * p.domeTiltX - cTilt * p.domeTiltVelX;
    const accTZ = -kTilt * p.domeTiltZ - cTilt * p.domeTiltVelZ;
    p.domeTiltVelX += accTX * h;
    p.domeTiltVelZ += accTZ * h;
    p.domeTiltX = Math.max(-0.032, Math.min(0.032, p.domeTiltX + p.domeTiltVelX * h));
    p.domeTiltZ = Math.max(-0.032, Math.min(0.032, p.domeTiltZ + p.domeTiltVelZ * h));

    // Overhead forged bronze arch subtle reaction sway driven by horizontal wire pull
    _wireVec.copy(p.pos).sub(this.lastDepartWorld);
    const wLen = Math.max(0.2, _wireVec.length());
    const targetArchX = -(_wireVec.z / wLen) * 0.012;
    const targetArchZ = (_wireVec.x / wLen) * 0.012;
    const kArch = 78.0;
    const cArch = 11.5;
    const accAX = kArch * (targetArchX - p.archSwayX) - cArch * p.archSwayVelX;
    const accAZ = kArch * (targetArchZ - p.archSwayZ) - cArch * p.archSwayVelZ;
    p.archSwayVelX += accAX * h;
    p.archSwayVelZ += accAZ * h;
    p.archSwayX = Math.max(-0.014, Math.min(0.014, p.archSwayX + p.archSwayVelX * h));
    p.archSwayZ = Math.max(-0.014, Math.min(0.014, p.archSwayZ + p.archSwayVelZ * h));
  }

  private stepStones(h: number): void {
    const gravity = 10.4;
    const stoneSurfaceR = DOME_RADIUS + STONE_SEAT_OFFSET;
    const minCos = Math.cos(DOME_MAX_PHI - 0.12);

    // 1. Flight integration + near-dome ferromagnetic snap-capture
    for (let i = 0; i < this.stones.length; i++) {
      const st = this.stones[i];
      st.age += h;
      st.pulseIntensity = Math.max(0, st.pulseIntensity - h * 2.4);

      if (st.flipTimer > 0) {
        const prevT = st.flipTimer;
        st.flipTimer = Math.max(0, st.flipTimer - h);
        if (prevT > 0 && st.flipTimer === 0) {
          this.pendulum.domeRecoilVel -= 0.085;
          this.emitImpact(st.normal, 0.72, st.polarity);
          this.emitScuff(st.normal, st.polarity, 0.65, true);
        }
      }

      if (st.phase === 'flying') {
        st.vel.y -= gravity * h;

        _v0.copy(st.pos).sub(DOME_CENTER);
        const distBefore = _v0.length();
        const surfGap = distBefore - stoneSurfaceR;
        if (surfGap > 0 && surfGap < 0.24 && _v0.y > 0) {
          const pull = 0.045 / Math.pow(surfGap + 0.065, 2.2);
          _v1.copy(_v0).divideScalar(Math.max(1e-5, distBefore));
          st.vel.addScaledVector(_v1, -pull * h);
        }

        st.pos.addScaledVector(st.vel, h);
        _q0.setFromAxisAngle(st.spinAxis, st.spinSpeed * h);
        st.quat.premultiply(_q0).normalize();

        _v0.copy(st.pos).sub(DOME_CENTER);
        const dist = _v0.length();
        if (dist <= stoneSurfaceR && _v0.y > -0.1) {
          st.normal.copy(_v0).divideScalar(Math.max(1e-5, dist));
          if (st.normal.y < minCos) {
            const horiz = Math.hypot(st.normal.x, st.normal.z);
            const tHoriz = Math.sqrt(Math.max(0.001, 1 - minCos * minCos));
            if (horiz > 1e-4) {
              st.normal.x = (st.normal.x / horiz) * tHoriz;
              st.normal.z = (st.normal.z / horiz) * tHoriz;
            }
            st.normal.y = minCos;
            st.normal.normalize();
          }

          st.pos.copy(st.normal).multiplyScalar(stoneSurfaceR).add(DOME_CENTER);
          const vn = st.vel.dot(st.normal);
          const impactMag = Math.min(1.0, Math.abs(vn) / 4.2);
          if (vn < 0) {
            this.pendulum.domeRecoilVel -= Math.min(0.24, Math.abs(vn) * 0.024);
            this.emitImpact(st.normal, Math.max(0.28, impactMag), st.polarity);
            this.emitScuff(st.normal, st.polarity, Math.max(0.35, impactMag * 0.85), true);
          }

          if (st.bouncesLeft > 0 && vn < -1.05) {
            st.bouncesLeft--;
            _v1.copy(st.vel).addScaledVector(st.normal, -vn);
            st.vel.copy(_v1).multiplyScalar(0.68).addScaledVector(st.normal, -vn * 0.24);
            st.spinSpeed *= 0.42;
            st.pulseIntensity = 0.85;
          } else {
            st.vel.addScaledVector(st.normal, -vn);
            st.vel.multiplyScalar(0.58);
            st.phase = 'sliding';
            st.pulseIntensity = 0.92;
            st.yaw = Math.atan2(st.vel.x, st.vel.z) + st.seed * 0.25;
            computeClungQuaternion(st.normal, st.yaw, st.quat);

            for (let j = 0; j < this.stones.length; j++) {
              if (j !== i && this.stones[j].phase === 'clung') {
                if (this.stones[j].pos.distanceTo(st.pos) < 0.58) {
                  this.stones[j].phase = 'sliding';
                }
              }
            }
          }
        } else if (st.pos.y < DOME_RIM_Y - 0.45) {
          st.normal.copy(_v0).normalize();
          st.normal.y = Math.max(minCos, Math.abs(st.normal.y));
          st.normal.normalize();
          st.pos.copy(st.normal).multiplyScalar(stoneSurfaceR).add(DOME_CENTER);
          st.vel.set(0, 0, 0);
          st.phase = 'sliding';
          computeClungQuaternion(st.normal, st.yaw, st.quat);
        }
      }
    }

    // 2. Rigid stone-stone clack separation & tangential impulse exchange
    for (let i = 0; i < this.stones.length; i++) {
      const a = this.stones[i];
      if (a.phase === 'flying') continue;
      for (let j = i + 1; j < this.stones.length; j++) {
        const b = this.stones[j];
        if (b.phase === 'flying') continue;

        _v1.copy(a.pos).sub(b.pos);
        const dist = _v1.length();
        const minSep = STONE_RADIUS * (a.scale + b.scale) * 0.96;
        if (dist < minSep && dist > 1e-5) {
          const overlap = minSep - dist;
          _v2.copy(_v1).divideScalar(dist);

          const aProj = _v2.dot(a.normal);
          _v3.copy(_v2).addScaledVector(a.normal, -aProj);
          if (_v3.lengthSq() > 1e-6) {
            _v3.normalize();
            a.pos.addScaledVector(_v3, overlap * 0.52);
            a.normal.copy(a.pos).sub(DOME_CENTER).normalize();
            a.pos.copy(a.normal).multiplyScalar(stoneSurfaceR).add(DOME_CENTER);
          }

          const bProj = _v2.dot(b.normal);
          _v4.copy(_v2).addScaledVector(b.normal, -bProj);
          if (_v4.lengthSq() > 1e-6) {
            _v4.normalize();
            b.pos.addScaledVector(_v4, -overlap * 0.52);
            b.normal.copy(b.pos).sub(DOME_CENTER).normalize();
            b.pos.copy(b.normal).multiplyScalar(stoneSurfaceR).add(DOME_CENTER);
          }

          const relVn = _v0.copy(a.vel).sub(b.vel).dot(_v2);
          if (relVn < 0) {
            const jImp = -0.68 * relVn;
            a.vel.addScaledVector(_v2, jImp);
            b.vel.addScaledVector(_v2, -jImp);
            a.vel.addScaledVector(a.normal, -a.vel.dot(a.normal));
            b.vel.addScaledVector(b.normal, -b.vel.dot(b.normal));
            if (Math.abs(relVn) > 0.25) {
              this.emitImpact(a.normal, Math.min(0.45, Math.abs(relVn) * 0.25), a.polarity);
            }
          }

          if (a.phase === 'clung') a.phase = 'sliding';
          if (b.phase === 'clung') b.phase = 'sliding';
        }
      }
    }

    // 3. Pairwise magnetic forces, bob-to-stone coupling, Coulomb static friction & poise
    const bobPos = this.pendulum.pos;
    const bobTipPos = this.pendulum.tipPos;
    const bobGrabBoost = this.pendulum.isGrabbed ? 2.4 : 1.0;

    for (let i = 0; i < this.stones.length; i++) {
      const a = this.stones[i];
      if (a.phase === 'flying') continue;
      const isDragged = a.id === this.pendulum.draggedStoneId;

      _tangentForce.set(0, 0, 0);

      _v0.set(0, -0.78, 0);
      const gn = _v0.dot(a.normal);
      _v0.addScaledVector(a.normal, -gn);
      _tangentForce.add(_v0);

      for (let j = 0; j < this.stones.length; j++) {
        if (i === j) continue;
        const b = this.stones[j];
        if (b.phase === 'flying') continue;

        _v1.copy(a.pos).sub(b.pos);
        const dist = _v1.length();
        if (dist < 1e-4) continue;
        const minSep = STONE_RADIUS * (a.scale + b.scale) * 1.04;

        if (dist < 0.72) {
          const effDist = Math.max(dist, minSep);
          const sameSign = a.polarity === b.polarity ? 1.0 : -1.0;
          const magForce = (sameSign * 0.165) / Math.pow(effDist + 0.062, 3.0);
          _v2.copy(_v1).divideScalar(dist);
          _v2.addScaledVector(a.normal, -_v2.dot(a.normal));
          if (_v2.lengthSq() > 1e-6) {
            _v2.normalize();
            _tangentForce.addScaledVector(_v2, magForce);
            if (dist < minSep) {
              const push = (minSep - dist) * 2800.0;
              const relApproach = Math.max(0, -_v0.copy(a.vel).sub(b.vel).dot(_v2));
              _tangentForce.addScaledVector(_v2, push + relApproach * 18.0);
            }
          }
        }

        if (isDragged && dist < 0.62 && b.phase === 'clung') {
          b.phase = 'sliding';
        }
      }

      if (isDragged) {
        _v5.copy(this.pendulum.draggedStoneTarget).sub(a.pos);
        _v5.addScaledVector(a.normal, -_v5.dot(a.normal));
        _tangentForce.addScaledVector(_v5, 160.0);
        _tangentForce.addScaledVector(a.vel, -22.0);
        a.phase = 'sliding';
      }

      _crownPos.copy(a.pos).addScaledVector(a.normal, 0.055);
      _v1.copy(bobTipPos).sub(_crownPos);
      const distTip = _v1.length();
      _v3.set(0, 0, 0);
      if (distTip < 0.68 && distTip > 1e-4) {
        const prox = 1.0 - distTip / 0.68;
        const smoothProx = prox * Math.sqrt(prox);
        _v2.copy(_v1).divideScalar(distTip);
        _v2.addScaledVector(a.normal, -_v2.dot(a.normal));
        if (_v2.lengthSq() > 1e-6) {
          _v2.normalize();
          _v3
            .copy(_v2)
            .multiplyScalar(-a.polarity * smoothProx * (this.pendulum.isGrabbed ? 1.85 : 1.35));
        }
      }

      _v1.copy(bobPos).sub(a.pos);
      const distBob = _v1.length();
      if (distBob < 0.45 && distBob > 1e-4) {
        _v2.copy(_v1).divideScalar(distBob);
        _v2.addScaledVector(a.normal, -_v2.dot(a.normal));
        if (_v2.lengthSq() > 1e-6) {
          _v2.normalize();
          const closeForce = (-a.polarity * 0.21 * bobGrabBoost) / Math.pow(distBob + 0.09, 2.5);
          _tangentForce.addScaledVector(_v2, closeForce);
          if (Math.abs(closeForce) > 3.6 && a.phase === 'clung') {
            a.phase = 'sliding';
          }
        }
      }

      // Critically damped spring for smooth magnetic stone tilt (active in BOTH sliding and clung!)
      const tiltK = 92.0;
      const tiltC = 17.0;
      _v4
        .copy(_v3)
        .sub(a.tugTilt)
        .multiplyScalar(tiltK)
        .addScaledVector(a.tugTiltVel, -tiltC);
      a.tugTiltVel.addScaledVector(_v4, h);
      a.tugTilt.addScaledVector(a.tugTiltVel, h);
      const tiltLen = a.tugTilt.length();
      if (tiltLen > 1.25) {
        a.tugTilt.multiplyScalar(1.25 / tiltLen);
      }

      const forceMag = _tangentForce.length();
      const staticFriction = 3.25;

      if (a.phase === 'clung' && forceMag > staticFriction + 0.25) {
        a.phase = 'sliding';
      }

      if (a.phase === 'sliding') {
        const speedSq = a.vel.lengthSq();
        if (!isDragged && forceMag < staticFriction && speedSq < 0.0025 && a.flipTimer <= 0) {
          a.vel.set(0, 0, 0);
          a.phase = 'clung';
          continue;
        }

        a.vel.addScaledVector(_tangentForce, h);
        const speed = a.vel.length();
        if (speed > 1e-5 && !isDragged) {
          const decel = Math.min(speed, (2.45 + 7.2 * speed) * h);
          a.vel.multiplyScalar((speed - decel) / speed);
        }
        a.pos.addScaledVector(a.vel, h);
        a.yaw += (a.vel.x - a.vel.z) * 0.42 * h;

        a.normal.copy(a.pos).sub(DOME_CENTER).normalize();
        if (a.normal.y < minCos) {
          const horiz = Math.hypot(a.normal.x, a.normal.z);
          const tHoriz = Math.sqrt(Math.max(0.001, 1 - minCos * minCos));
          if (horiz > 1e-4) {
            a.normal.x = (a.normal.x / horiz) * tHoriz;
            a.normal.z = (a.normal.z / horiz) * tHoriz;
          }
          a.normal.y = minCos;
          a.normal.normalize();
          a.vel.multiplyScalar(-0.22);
        }

        if (speed > 0.05 && a.flipTimer <= 0) {
          this.emitScuff(a.normal, a.polarity, Math.min(0.85, speed * 0.45), false);
        }

        let hopOffset = 0;
        const tiltLift = Math.min(0.014, a.tugTilt.length() * 0.010);
        if (a.flipTimer > 0) {
          const u = 1.0 - a.flipTimer / a.flipDuration;
          hopOffset = Math.sin(u * Math.PI) * 0.088;
          computeClungQuaternion(a.normal, a.yaw, _q2);
          const smoothU = u * u * (3 - 2 * u);
          const angle = smoothU * Math.PI * 2;
          _qFlip.setFromAxisAngle(a.flipAxis, angle);
          a.quat.copy(a.flipStartQuat).slerp(_q2, smoothU);
          a.quat.premultiply(_qFlip).normalize();
        } else {
          _v0.copy(a.normal).addScaledVector(a.tugTilt, 0.36);
          if (a.hoverPoise > 0.001) {
            _v0.addScaledVector(_up, a.hoverPoise * 0.055);
          }
          _v0.normalize();
          computeClungQuaternion(_v0, a.yaw, _q1);
          a.quat.slerp(_q1, Math.min(1, h * 22));
        }

        a.pos
          .copy(a.normal)
          .multiplyScalar(stoneSurfaceR + hopOffset + (a.flipTimer > 0 ? 0 : tiltLift))
          .add(DOME_CENTER);

        const vn = a.vel.dot(a.normal);
        a.vel.addScaledVector(a.normal, -vn);

        if (
          !isDragged &&
          a.vel.lengthSq() < 0.0012 &&
          forceMag < staticFriction + 0.1 &&
          a.flipTimer <= 0
        ) {
          a.vel.set(0, 0, 0);
          a.phase = 'clung';
        }
      } else {
        const tiltLift = Math.min(0.014, a.tugTilt.length() * 0.010);
        a.pos.copy(a.normal).multiplyScalar(stoneSurfaceR + tiltLift).add(DOME_CENTER);

        _v0.copy(a.normal).addScaledVector(a.tugTilt, 0.36);
        if (a.hoverPoise > 0.001) {
          _v0.addScaledVector(_up, a.hoverPoise * 0.055);
        }
        _v0.normalize();
        computeClungQuaternion(_v0, a.yaw, _q1);
        a.quat.slerp(_q1, Math.min(1, h * 20));
      }
    }
  }

  private stepPendulum(h: number): void {
    const p = this.pendulum;
    _prevVel.copy(p.vel);
    _bobUp.set(0, 1, 0).applyQuaternion(p.quat).normalize();
    _polePos.copy(p.pos).addScaledVector(_bobUp, -0.105);

    if (p.isGrabbed) {
      _v0.copy(p.grabTarget).sub(DOME_CENTER);
      const rawDist = _v0.length();
      if (rawDist < 1e-4) _v0.set(0, 1, 0);
      else _v0.divideScalar(rawDist);

      const maxPhi = DOME_MAX_PHI + 0.02;
      const minCos = Math.cos(maxPhi);
      if (_v0.y < minCos) {
        const horiz = Math.hypot(_v0.x, _v0.z);
        const tHoriz = Math.sqrt(Math.max(0.001, 1 - minCos * minCos));
        if (horiz > 1e-4) {
          _v0.x = (_v0.x / horiz) * tHoriz;
          _v0.z = (_v0.z / horiz) * tHoriz;
        }
        _v0.y = minCos;
        _v0.normalize();
      }

      const minSafeR = BOB_SHELL_RADIUS + this.computeMinSafeRadialOffset(_v0);
      const targetR = Math.max(minSafeR + 0.02, Math.min(BOB_SHELL_RADIUS + 0.45, rawDist));
      _v1.copy(_v0).multiplyScalar(targetR).add(DOME_CENTER);

      // Live tactile magnetic feel while dragging the bob across the lodestone constellation
      _magForce.set(0, 0, 0);
      _fTipCant.set(0, 0, 0);
      for (let i = 0; i < this.stones.length; i++) {
        const st = this.stones[i];
        if (st.phase === 'flying') continue;
        _crownPos.copy(st.pos).addScaledVector(st.normal, 0.055);
        _v2.copy(_polePos).sub(_crownPos);
        const dist = _v2.length();
        const softSq = dist * dist + 0.115;
        const invPow = 1.0 / (softSq * Math.sqrt(softSq));
        const strength = st.polarity === 1 ? 0.48 : -0.42;
        _v2.divideScalar(Math.max(1e-5, dist));
        _magForce.addScaledVector(_v2, strength * invPow);
        if (dist < 0.65) {
          const prox = 1.0 - dist / 0.65;
          st.pulseIntensity = Math.max(st.pulseIntensity, prox);
          _fTipCant.addScaledVector(_v2, -strength * invPow * 0.018);
        }
      }
      p.magneticForceMag = _magForce.length();

      _v5.copy(p.pos);
      _v3.copy(_v1).sub(p.pos).multiplyScalar(135.0).addScaledVector(p.vel, -21.0).addScaledVector(_magForce, 0.22);
      p.vel.addScaledVector(_v3, h);
      p.pos.addScaledVector(p.vel, h);

      _v4.copy(p.pos).sub(DOME_CENTER);
      let newR = _v4.length();
      if (newR < 1e-4) _v4.set(0, 1, 0), (newR = 1);
      else _v4.divideScalar(newR);

      const postMinR = BOB_SHELL_RADIUS + this.computeMinSafeRadialOffset(_v4);
      if (newR < postMinR) {
        newR = postMinR;
        p.pos.copy(_v4).multiplyScalar(newR).add(DOME_CENTER);
      }

      p.dir.copy(_v4);
      p.radialOffset = newR - BOB_SHELL_RADIUS;
      p.radialVel = 0;

      // Smooth drag velocity EMA so steady holds drop cleanly while flicks launch crisp orbits
      _v6.copy(p.pos).sub(_v5).divideScalar(Math.max(h, 0.001));
      const instSpd = _v6.length();
      if (instSpd > 4.2) _v6.multiplyScalar(4.2 / instSpd);
      p.dragVelEma.lerp(_v6, Math.min(1, h * 18.0));
      p.vel.copy(p.dragVelEma);

      // Update magnetic tip cant spring while grabbed
      _fTipCant.addScaledVector(_bobUp, -_fTipCant.dot(_bobUp));
      _v2.copy(_fTipCant).sub(p.magTipCant).multiplyScalar(110.0).addScaledVector(p.magTipCantVel, -18.0);
      p.magTipCantVel.addScaledVector(_v2, h);
      p.magTipCant.addScaledVector(p.magTipCantVel, h);
      const cantLen = p.magTipCant.length();
      if (cantLen > 0.092) p.magTipCant.multiplyScalar(0.092 / cantLen);

      this.updatePulleyState(h, false);
      this.updateBobOrientation(h);
      this.updateBobTipPos();
      return;
    }

    p.dragVelEma.set(0, 0, 0);

    // 1. True Wire-Coupled Gravity + Counterweight/Fusee Wire Tension
    _wireVec.copy(p.pos).sub(this.lastDepartWorld);
    const wireLen = Math.max(0.25, _wireVec.length());
    _wireDir.copy(_wireVec).divideScalar(wireLen);
    const dLdt = p.vel.dot(_wireDir);

    _fTotal.set(0, -7.4, 0);
    const wireTension = Math.max(0.5, 7.75 + 2.65 * (wireLen - 1.0) + 0.36 * dLdt);
    _fTotal.addScaledVector(_wireDir, -wireTension);

    // 2. Calibrated 3D Magnetic Dipole Forces acting on the bob's lower magnetic pole
    _magForce.set(0, 0, 0);
    _fTipCant.set(0, 0, 0);
    let torqueKick = 0;
    let signedInfluence = 0;
    let closeVerdigrisId = -1;
    let closeVerdigrisDist = 0.42;

    for (let i = 0; i < this.stones.length; i++) {
      const st = this.stones[i];
      if (st.phase === 'flying') continue;

      _crownPos.copy(st.pos).addScaledVector(st.normal, 0.055);
      _v1.copy(_polePos).sub(_crownPos);
      const dist = _v1.length();

      const softSq = dist * dist + 0.115;
      const invPow = 1.0 / (softSq * Math.sqrt(softSq));

      const strength = st.polarity === 1 ? 0.48 : -0.42;
      const fMag = strength * invPow;

      _v2.copy(_v1).divideScalar(Math.max(1e-5, dist));
      _magForce.addScaledVector(_v2, fMag);

      if (dist < 0.65) {
        const proximity = 1.0 - dist / 0.65;
        st.pulseIntensity = Math.max(st.pulseIntensity, proximity);
        signedInfluence += st.polarity * proximity;

        _v3.crossVectors(_v2, p.vel);
        torqueKick += st.polarity * proximity * _v3.dot(p.dir) * 3.4;

        // Lateral magnetic lean of bob tip toward Verdigris / away from Cinnabar
        _fTipCant.addScaledVector(_v2, -fMag * 0.018);
      }

      if (st.polarity === -1 && dist < closeVerdigrisDist) {
        closeVerdigrisDist = dist;
        closeVerdigrisId = st.id;
      }
    }

    this.lastSignedInfluence = Math.max(-1, Math.min(1, signedInfluence));
    p.magneticForceMag = _magForce.length();

    // Scale radial dipole boost cleanly: buoyant upward vaults over Cinnabar, gentle dip over Verdigris
    const magRad = _magForce.dot(p.dir);
    _fPerp.copy(_magForce).addScaledVector(p.dir, -magRad);
    const radScale = magRad > 0 ? 0.85 : 0.45;
    _fTotal.add(_fPerp).addScaledVector(p.dir, magRad * radScale);

    // 3. Ferromagnetic Iron Dome Grazing Boundary Layer + Cushion
    _v4.copy(p.pos).sub(DOME_CENTER);
    let r = Math.max(0.2, _v4.length());
    p.dir.copy(_v4).divideScalar(r);

    const minSafeOffset = this.computeMinSafeRadialOffset(p.dir);
    const minSafeR = BOB_SHELL_RADIUS + minSafeOffset;
    const targetR = minSafeR + 0.026;
    const vRad = p.vel.dot(p.dir);
    const kSurf = 32.0 / (1.0 + Math.max(0.0, r - targetR) * 4.2);
    const fRadialDome = -kSurf * (r - targetR) - 5.2 * vRad;
    _fTotal.addScaledVector(p.dir, fRadialDome);

    // 4. Anisotropic Pulley + Counterweight Effective Inertia Along the Wire
    const fParMag = _fTotal.dot(_wireDir);
    _fPar.copy(_wireDir).multiplyScalar(fParMag);
    _fPerp.copy(_fTotal).sub(_fPar);
    _fTotal.copy(_fPerp).addScaledVector(_fPar, 1.0 / 1.42);

    // 5. Anti-Trap Verdigris Dwell Release + Energy-Aware Horological Escapement Sustain
    if (closeVerdigrisId !== -1 && closeVerdigrisId === this.nearestVerdigrisId) {
      this.singleStoneDwellTimer += h;
    } else {
      this.nearestVerdigrisId = closeVerdigrisId;
      this.singleStoneDwellTimer = 0;
    }

    if (this.singleStoneDwellTimer > 1.35) {
      const trappedStone = this.stones.find((s) => s.id === this.nearestVerdigrisId);
      if (trappedStone) {
        _v1.copy(p.pos).sub(trappedStone.pos);
        _v1.addScaledVector(p.dir, -_v1.dot(p.dir));
        if (_v1.lengthSq() < 1e-4) {
          _v1.crossVectors(p.dir, _up);
        }
        if (_v1.lengthSq() < 1e-4) _v1.set(1, 0, 0);
        _v1.normalize();
        // Smooth crown-crossing + tangential escape impulse
        _v2.copy(_up).addScaledVector(p.dir, -p.dir.y).normalize();
        _fTotal.addScaledVector(_v1, 1.65).addScaledVector(_v2, 1.45);
      }
    }

    const speed = p.vel.length();
    const swingEnergy = 0.5 * speed * speed + 3.2 * (1.0 - p.dir.y);
    const targetEnergy = 2.15;
    if (swingEnergy < targetEnergy) {
      _v1.copy(p.vel).addScaledVector(p.dir, -p.vel.dot(p.dir));
      if (_v1.lengthSq() < 1e-4) {
        _v1.crossVectors(p.dir, _up);
      }
      if (_v1.lengthSq() < 1e-4) _v1.set(1, 0, 0);
      _v1.normalize();
      const sustainMag = Math.min(1.15, (targetEnergy - swingEnergy) * 0.90);
      _fTotal.addScaledVector(_v1, sustainMag);
    }

    // Integrate velocity & aerodynamic/bearing drag
    p.vel.addScaledVector(_fTotal, h);
    const damping = 0.042 + 0.020 * Math.min(4.0, speed);
    p.vel.multiplyScalar(Math.exp(-damping * h));

    const newSpeed = p.vel.length();
    if (newSpeed > 5.2) {
      p.vel.multiplyScalar(5.2 / newSpeed);
    }

    // Integrate 3D position & enforce smooth dome rim + radial clearance envelope
    p.pos.addScaledVector(p.vel, h);
    _v4.copy(p.pos).sub(DOME_CENTER);
    r = Math.max(0.2, _v4.length());
    p.dir.copy(_v4).divideScalar(r);

    const maxBobPhi = DOME_MAX_PHI - 0.14;
    const minBobCos = Math.cos(maxBobPhi);
    if (p.dir.y < minBobCos) {
      const horiz = Math.hypot(p.dir.x, p.dir.z);
      const tHoriz = Math.sqrt(Math.max(0.001, 1 - minBobCos * minBobCos));
      if (horiz > 1e-4) {
        p.dir.x = (p.dir.x / horiz) * tHoriz;
        p.dir.z = (p.dir.z / horiz) * tHoriz;
      }
      p.dir.y = minBobCos;
      p.dir.normalize();
      if (p.vel.y < 0) p.vel.y *= -0.58;
    }

    const postMinSafeR = BOB_SHELL_RADIUS + this.computeMinSafeRadialOffset(p.dir);
    const maxAllowedR = BOB_SHELL_RADIUS + 0.30;
    if (r < postMinSafeR) {
      r = postMinSafeR;
      const vn = p.vel.dot(p.dir);
      if (vn < 0) p.vel.addScaledVector(p.dir, -1.30 * vn);
    } else if (r > maxAllowedR) {
      r = maxAllowedR;
      const vn = p.vel.dot(p.dir);
      if (vn > 0) p.vel.addScaledVector(p.dir, -1.20 * vn);
    }

    p.radialOffset = r - BOB_SHELL_RADIUS;
    p.radialVel = p.vel.dot(p.dir);
    p.pos.copy(p.dir).multiplyScalar(r).add(DOME_CENTER);

    // Update magnetic tip cant spring
    _fTipCant.addScaledVector(_bobUp, -_fTipCant.dot(_bobUp));
    _v2.copy(_fTipCant).sub(p.magTipCant).multiplyScalar(110.0).addScaledVector(p.magTipCantVel, -18.0);
    p.magTipCantVel.addScaledVector(_v2, h);
    p.magTipCant.addScaledVector(p.magTipCantVel, h);
    const cantLen = p.magTipCant.length();
    if (cantLen > 0.092) p.magTipCant.multiplyScalar(0.092 / cantLen);

    p.spinVel += (torqueKick - p.spinVel * 1.45) * h;
    p.spinAngle += p.spinVel * h;

    // Trigger subtle high-tension wire pluck micro-shimmer on sharp magnetic impulse kicks
    const jerk = _v0.copy(p.vel).sub(_prevVel).length() / Math.max(h, 0.001);
    if (jerk > 12.0) {
      p.wirePluckAmp = Math.min(1.0, p.wirePluckAmp + (jerk - 12.0) * 0.012);
    }

    this.updatePulleyState(h, false);
    this.updateBobOrientation(h);
    this.updateBobTipPos();
  }

  /**
   * Updates swivel azimuth with smooth rotational inertia (zero ±180° apex snap),
   * exact tangent departure point on the outer pulley sheave, pay-out wire length,
   * differential wheel rotation angle, and unclamped counterweight elevation.
   */
  private updatePulleyState(h: number, snapInstant: boolean): void {
    const p = this.pendulum;

    const horizDist = Math.hypot(p.pos.x, p.pos.z);
    if (snapInstant) {
      if (horizDist > 0.02) {
        p.pulleyAzimuth = Math.atan2(p.pos.x, p.pos.z);
      }
      p.pulleySwivelVel = 0;
    } else if (horizDist > 1e-4) {
      const sinAzCur = Math.sin(p.pulleyAzimuth);
      const cosAzCur = Math.cos(p.pulleyAzimuth);
      const hx = p.pos.x / horizDist;
      const hz = p.pos.z / horizDist;
      const crossY = cosAzCur * hx - sinAzCur * hz;
      const dotXZ = sinAzCur * hx + cosAzCur * hz;
      const delta = Math.atan2(crossY, dotXZ);
      const horizWeight = horizDist / (horizDist + 0.12);
      const swivelAcc = 48.0 * delta * horizWeight - 12.5 * p.pulleySwivelVel;
      p.pulleySwivelVel += swivelAcc * h;
      p.pulleyAzimuth += p.pulleySwivelVel * h;
    }

    // Approximate collet top world point along current line from pulley
    const sinAz = Math.sin(p.pulleyAzimuth);
    const cosAz = Math.cos(p.pulleyAzimuth);
    const rightX = cosAz;
    const rightZ = -sinAz;
    const fwdX = sinAz;
    const fwdZ = cosAz;

    // Outer pulley axle center in world space
    const axleX = rightX * PULLEY_OUTER_OFFSET_X;
    const axleY = PULLEY_AXLE_Y;
    const axleZ = rightZ * PULLEY_OUTER_OFFSET_X;

    // Bob collet anchor (aligned toward pulley axle so wire enters collet dead-straight)
    _v0.set(axleX - p.pos.x, axleY - p.pos.y, axleZ - p.pos.z);
    const distToAxle = _v0.length();
    if (distToAxle > 1e-5) {
      _v0.divideScalar(distToAxle);
    } else {
      _v0.set(0, 1, 0);
    }
    this.lastBobCollet.copy(p.pos).addScaledVector(_v0, BOB_TOP_COLLET_OFFSET);

    // Project collet vector into pulley wheel plane (fwd = +Z_local, up = +Y_local)
    const relX = this.lastBobCollet.x - axleX;
    const relY = this.lastBobCollet.y - axleY;
    const relZ = this.lastBobCollet.z - axleZ;
    const dzLocal = relX * fwdX + relZ * fwdZ;
    const dyLocal = relY;
    const D = Math.max(PULLEY_OUTER_RADIUS * 1.15, Math.hypot(dzLocal, dyLocal));
    const departAngle =
      Math.atan2(dyLocal, dzLocal) + Math.acos(Math.min(0.98, PULLEY_OUTER_RADIUS / D));

    const tanZ = Math.cos(departAngle) * PULLEY_OUTER_RADIUS;
    const tanY = Math.sin(departAngle) * PULLEY_OUTER_RADIUS;
    this.lastDepartWorld.set(axleX + fwdX * tanZ, axleY + tanY, axleZ + fwdZ * tanZ);

    const prevWireLen = p.wireLength;
    const newWireLen = this.lastDepartWorld.distanceTo(this.lastBobCollet);
    p.wireLength = newWireLen;

    // Exact differential pulley kinematics:
    // Wheel spins through (newWireLen - WIRE_REF_LENGTH) / PULLEY_OUTER_RADIUS;
    // Inner drum (PULLEY_INNER_RADIUS) lifts/lowers counterweight proportionally (2.73:1 ratio!)
    const payoutDelta = newWireLen - WIRE_REF_LENGTH;
    p.pulleyWheelAngle = payoutDelta / PULLEY_OUTER_RADIUS;

    const gearRatio = PULLEY_INNER_RADIUS / PULLEY_OUTER_RADIUS;
    const targetCwY = COUNTERWEIGHT_TOP_Y + payoutDelta * gearRatio;
    p.counterweightY = Math.max(0.78, Math.min(1.36, targetCwY));

    // Counterweight micro-sway induced by wire pay-out acceleration and swivel rotation
    const wireSpeed = snapInstant ? 0 : (newWireLen - prevWireLen) / Math.max(h, 0.001);
    const swayAcc =
      -36.0 * p.cwSwayAngle - 4.6 * p.cwSwayVel + wireSpeed * 0.095 + p.pulleySwivelVel * 0.018;
    p.cwSwayVel += swayAcc * h;
    p.cwSwayAngle = Math.max(-0.135, Math.min(0.135, p.cwSwayAngle + p.cwSwayVel * h));

    // High-tension metal wire micro-pluck decay
    p.wirePluckAmp *= Math.exp(-10.5 * h);
    p.wirePluckPhase += 72.0 * h;
  }

  private updateBobOrientation(h: number): void {
    const p = this.pendulum;
    // Bob upper brass collet aligns coaxially with the taut plumb wire + subtle magnetic tip cant
    _v0.copy(this.lastDepartWorld).sub(p.pos);
    if (_v0.lengthSq() > 1e-6) {
      _v0.normalize();
    } else {
      _v0.set(0, 1, 0);
    }

    _v0.addScaledVector(p.magTipCant, -1.0).normalize();

    _q0.setFromUnitVectors(_up, _v0);
    _q1.setFromAxisAngle(_up, p.spinAngle);
    _q0.multiply(_q1);
    p.quat.slerp(_q0, Math.min(1, h * 46));
  }

  /**
   * Populates exact world-space endpoints & pulley angles for rendering the swivel clevis,
   * spoked brass differential sheave, counterweight cylinder, and 3-segment taut plumb wire.
   */
  computePulleyKinematics(out: PulleyKinematicsFrame): void {
    const p = this.pendulum;
    const sinAz = Math.sin(p.pulleyAzimuth);
    const cosAz = Math.cos(p.pulleyAzimuth);
    const rightX = cosAz;
    const rightZ = -sinAz;
    const fwdX = sinAz;
    const fwdZ = cosAz;

    // Exact bob upper collet anchor along bob's live quaternion +Y axis
    _v0.set(0, 1, 0).applyQuaternion(p.quat).normalize();
    out.bobColletWorld.copy(p.pos).addScaledVector(_v0, BOB_TOP_COLLET_OFFSET);

    // Outer pulley axle in world space
    const outerAxleX = rightX * PULLEY_OUTER_OFFSET_X;
    const outerAxleY = PULLEY_AXLE_Y;
    const outerAxleZ = rightZ * PULLEY_OUTER_OFFSET_X;

    // Analytical tangent departure point on the outer V-groove sheave toward bob collet
    const relX = out.bobColletWorld.x - outerAxleX;
    const relY = out.bobColletWorld.y - outerAxleY;
    const relZ = out.bobColletWorld.z - outerAxleZ;
    const dzLocal = relX * fwdX + relZ * fwdZ;
    const dyLocal = relY;
    const D = Math.max(PULLEY_OUTER_RADIUS * 1.15, Math.hypot(dzLocal, dyLocal));
    const departAngle =
      Math.atan2(dyLocal, dzLocal) + Math.acos(Math.min(0.98, PULLEY_OUTER_RADIUS / D));

    const tanZ = Math.cos(departAngle) * PULLEY_OUTER_RADIUS;
    const tanY = Math.sin(departAngle) * PULLEY_OUTER_RADIUS;
    out.bobTangentWorld.set(outerAxleX + fwdX * tanZ, outerAxleY + tanY, outerAxleZ + fwdZ * tanZ);

    // Inner coaxial drum rear tangent point -> drops vertically to the brass counterweight eyelet
    const innerAxleX = rightX * PULLEY_INNER_OFFSET_X;
    const innerAxleZ = rightZ * PULLEY_INNER_OFFSET_X;
    out.cwTangentWorld.set(
      innerAxleX - fwdX * PULLEY_INNER_RADIUS,
      PULLEY_AXLE_Y,
      innerAxleZ - fwdZ * PULLEY_INNER_RADIUS
    );

    const dropLen = Math.max(0.06, PULLEY_AXLE_Y - p.counterweightY);
    const swayOffset = Math.sin(p.cwSwayAngle) * dropLen;
    out.cwEyeletWorld.set(
      out.cwTangentWorld.x + fwdX * swayOffset,
      PULLEY_AXLE_Y - Math.cos(p.cwSwayAngle) * dropLen,
      out.cwTangentWorld.z + fwdZ * swayOffset
    );

    out.azimuth = p.pulleyAzimuth;
    out.wheelAngle = p.pulleyWheelAngle;
    out.departAngleLocal = departAngle;
    out.wireLength = out.bobTangentWorld.distanceTo(out.bobColletWorld);
    out.pluckPhase = p.wirePluckPhase;
    out.pluckAmp = p.wirePluckAmp;
  }
}
