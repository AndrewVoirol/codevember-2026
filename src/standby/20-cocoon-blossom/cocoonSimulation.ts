import * as THREE from 'three/webgpu';
import {
  BLOOM_PETAL_SPECS,
  buildBloomPetalBaseData,
  PETAL_BASE_Y,
  type BloomPetalBaseData,
} from './kozoGeometry';
import {
  DEFAULT_KOZO_CONFIG,
  type CocoonSimConfig,
  type Day20TelemetryState,
  type OrigamiMothState,
} from './types';

export class KozoCocoonSimulation {
  readonly config: CocoonSimConfig;
  readonly petals: BloomPetalBaseData[];
  readonly moths: OrigamiMothState[];

  /** Pre-allocated 280 curl-advected luminous pollen & silk motes */
  readonly pollenPos: Float32Array;
  readonly pollenVel: Float32Array;
  readonly pollenAge: Float32Array;
  readonly pollenMaxAge: Float32Array;
  readonly pollenSize: Float32Array;
  readonly pollenWarmth: Float32Array;
  readonly pollenAlpha: Float32Array;

  bloom: number;
  targetBloom: number;
  bloomVel = 0;

  /** Independent 2-DOF damped springs per petal (pitch radial + tangential sway) */
  readonly petalPitchOffset: Float32Array;
  readonly petalPitchVel: Float32Array;
  readonly petalSwayOffset: Float32Array;
  readonly petalSwayVel: Float32Array;
  readonly petalLocalBloom: Float32Array;

  emberGlow = 1.0;
  emberTarget = 1.0;

  /** Airflow wake driven continuously by cursor motion across air OR petals */
  readonly airWakeLocal = new THREE.Vector3(0, 0.2, 0.5);
  readonly airWakeVel = new THREE.Vector3(0, 0, 0);
  airWakeStrength = 0;
  readonly candleOffset = new THREE.Vector2(0, 0);

  elapsedSec = 0;
  private isDisposed = false;
  private readonly sortDistances: Float32Array;

  constructor(config: CocoonSimConfig = DEFAULT_KOZO_CONFIG) {
    this.config = config;
    this.bloom = config.initialBloom;
    this.targetBloom = config.initialBloom;

    this.petals = BLOOM_PETAL_SPECS.map((spec) => buildBloomPetalBaseData(spec));
    const nPetals = this.petals.length;

    this.petalPitchOffset = new Float32Array(nPetals);
    this.petalPitchVel = new Float32Array(nPetals);
    this.petalSwayOffset = new Float32Array(nPetals);
    this.petalSwayVel = new Float32Array(nPetals);
    this.petalLocalBloom = new Float32Array(nPetals);
    this.sortDistances = new Float32Array(nPetals);

    // 12 buoyant curl-gliding origami moths
    this.moths = [];
    for (let i = 0; i < config.mothCount; i++) {
      const tier = i < 4 ? 0 : i < 8 ? 1 : 2;
      const orbitRadius = tier === 0 ? 0.22 + (i % 4) * 0.08 : tier === 1 ? 0.52 + (i % 4) * 0.11 : 0.85 + (i % 4) * 0.12;
      const orbitHeight = tier === 0 ? 0.06 + (i % 4) * 0.16 : tier === 1 ? 0.44 + (i % 4) * 0.16 : 0.84 + (i % 4) * 0.15;
      const orbitPhase = (i / config.mothCount) * Math.PI * 2 + tier * 0.7;
      const startX = Math.cos(orbitPhase) * orbitRadius * 0.6;
      const startZ = Math.sin(orbitPhase) * orbitRadius * 0.6;

      this.moths.push({
        id: i,
        pos: new THREE.Vector3(startX, orbitHeight, startZ),
        vel: new THREE.Vector3(0, 0.05, 0),
        orbitRadius,
        orbitHeight,
        orbitPhase,
        orbitSpeed: (i % 2 === 0 ? 1 : -1) * (0.48 + (i % 5) * 0.11),
        heading: orbitPhase + Math.PI * 0.5,
        pitch: 0.22,
        roll: 0,
        wingPhase: i * 1.47,
        wingSpeed: 9.5 + (i % 4) * 1.1,
        liftProgress: this.bloom > 0.18 ? 1 : 0,
        scale: tier === 0 ? 0.095 : tier === 1 ? 0.082 : 0.072,
        burstTimer: 0,
        active: this.bloom > 0.18,
      });
    }

    // 280 curl-advected luminous pollen & silk motes
    const nPol = config.pollenCount;
    this.pollenPos = new Float32Array(nPol * 3);
    this.pollenVel = new Float32Array(nPol * 3);
    this.pollenAge = new Float32Array(nPol);
    this.pollenMaxAge = new Float32Array(nPol);
    this.pollenSize = new Float32Array(nPol);
    this.pollenWarmth = new Float32Array(nPol);
    this.pollenAlpha = new Float32Array(nPol);

    for (let i = 0; i < nPol; i++) {
      this.respawnPollen(i, (i / nPol) * 0.95);
    }

    this.updatePetals(0.016);
  }

  private respawnPollen(i: number, initialAgeFrac = 0): void {
    const i3 = i * 3;
    const h1 = fract(Math.sin(i * 12.9898 + 78.233) * 43758.5453);
    const h2 = fract(Math.sin(i * 93.9898 + 19.123) * 28471.3127);
    const h3 = fract(Math.sin(i * 41.1234 + 63.719) * 19384.6192);

    const ang = h1 * Math.PI * 2;
    const rad = 0.03 + Math.sqrt(h2) * 0.42;
    const y0 = PETAL_BASE_Y + 0.16 + h3 * 0.26 + initialAgeFrac * 1.55;

    this.pollenPos[i3] = Math.cos(ang) * rad * (1 + initialAgeFrac * 1.4);
    this.pollenPos[i3 + 1] = y0;
    this.pollenPos[i3 + 2] = Math.sin(ang) * rad * (1 + initialAgeFrac * 1.4);

    this.pollenVel[i3] = Math.cos(ang) * 0.06;
    this.pollenVel[i3 + 1] = 0.18 + h2 * 0.16;
    this.pollenVel[i3 + 2] = Math.sin(ang) * 0.06;

    const maxAge = 3.8 + h1 * 3.4;
    this.pollenMaxAge[i] = maxAge;
    this.pollenAge[i] = initialAgeFrac * maxAge;
    this.pollenSize[i] = 0.022 + Math.pow(h3, 1.4) * 0.044;
    this.pollenWarmth[i] = h2;
    this.pollenAlpha[i] = initialAgeFrac > 0 ? Math.sin(initialAgeFrac * Math.PI) : 0;
  }

  setTargetBloom(target: number): void {
    this.targetBloom = Math.max(0, Math.min(1, target));
  }

  pullPetal(petalIndex: number, outwardDelta: number, lateralDelta: number): void {
    if (petalIndex >= 0 && petalIndex < this.petals.length) {
      this.petalPitchVel[petalIndex] += outwardDelta * 3.8;
      this.petalSwayVel[petalIndex] += lateralDelta * 3.2;
      const n = this.petals.length;
      const prev = (petalIndex - 1 + n) % n;
      const next = (petalIndex + 1) % n;
      this.petalPitchVel[prev] += outwardDelta * 1.2;
      this.petalPitchVel[next] += outwardDelta * 1.2;
    }
    this.setTargetBloom(this.targetBloom + outwardDelta * 0.85);
  }

  setAirWake(
    localX: number,
    localY: number,
    localZ: number,
    velX: number,
    velY: number,
    velZ: number,
    active: boolean
  ): void {
    if (active) {
      this.airWakeLocal.set(localX, localY, localZ);
      this.airWakeVel.set(
        clamp(velX, -6, 6),
        clamp(velY, -6, 6),
        clamp(velZ, -6, 6)
      );
      this.airWakeStrength = Math.min(1, this.airWakeStrength + 0.28);
      this.candleOffset.lerp(_tmpVec2.set(clamp(localX * 0.9, -1, 1), clamp(localY * 0.9, -1, 1)), 0.14);

      const speed = Math.hypot(velX, velY, velZ);
      if (speed > 0.08) {
        const pAng = Math.atan2(localX, localZ);
        const radialPush = clamp(-velY * 0.03 + speed * 0.015, -0.18, 0.22);
        const tangentPush = clamp((velX * Math.cos(pAng) - velZ * Math.sin(pAng)) * 0.035, -0.22, 0.22);

        for (let i = 0; i < this.petals.length; i++) {
          const dAng = angleDiff(pAng, this.petals[i].spec.azimuth);
          const prox = Math.exp(-dAng * dAng * 2.6);
          this.petalPitchVel[i] += prox * radialPush * 0.65;
          this.petalSwayVel[i] += prox * tangentPush * 0.65;
        }
      }
    } else {
      this.airWakeStrength = Math.max(0, this.airWakeStrength - 0.08);
    }
  }

  pulseBloomBurst(): number {
    this.emberTarget = Math.min(1.75, this.emberTarget + 0.55);
    if (this.targetBloom < 0.45) {
      this.setTargetBloom(0.84);
    } else if (this.targetBloom < 0.88) {
      this.setTargetBloom(1.0);
    }

    for (let i = 0; i < this.petals.length; i++) {
      this.petalPitchVel[i] += 0.42 - this.petals[i].spec.whorl * 0.08;
    }

    let boosted = 0;
    for (let i = 0; i < this.moths.length; i++) {
      const m = this.moths[i];
      m.active = true;
      m.burstTimer = 3.2 + (i % 4) * 0.7;
      m.vel.y += 0.45 + (i % 3) * 0.14;
      m.vel.x += Math.cos(m.orbitPhase) * 0.25;
      m.vel.z += Math.sin(m.orbitPhase) * 0.25;
      boosted++;
    }

    const nPol = this.config.pollenCount;
    for (let i = 0; i < nPol; i += 3) {
      const i3 = i * 3;
      this.pollenVel[i3 + 1] += 0.35;
    }
    return boosted;
  }

  cycleBloomPreset(): number {
    if (this.targetBloom < 0.28) {
      this.setTargetBloom(0.96);
      this.pulseBloomBurst();
    } else {
      this.setTargetBloom(0.0);
    }
    return this.targetBloom;
  }

  step(dtSec: number): void {
    if (this.isDisposed) return;
    const dt = Math.min(0.05, Math.max(0.0005, dtSec));
    this.elapsedSec += dt;

    const force = (this.targetBloom - this.bloom) * this.config.bloomStiffness;
    this.bloomVel = (this.bloomVel + force * dt) * Math.exp(-this.config.bloomDamping * dt);
    this.bloom = Math.max(0, Math.min(1, this.bloom + this.bloomVel * dt));

    const nPetals = this.petals.length;
    for (let i = 0; i < nPetals; i++) {
      const fP = -this.petalPitchOffset[i] * 26.0;
      const fS = -this.petalSwayOffset[i] * 28.0;
      this.petalPitchVel[i] = (this.petalPitchVel[i] + fP * dt) * Math.exp(-5.4 * dt);
      this.petalSwayVel[i] = (this.petalSwayVel[i] + fS * dt) * Math.exp(-5.8 * dt);
      this.petalPitchOffset[i] = clamp(this.petalPitchOffset[i] + this.petalPitchVel[i] * dt, -0.45, 0.55);
      this.petalSwayOffset[i] = clamp(this.petalSwayOffset[i] + this.petalSwayVel[i] * dt, -0.35, 0.35);
    }

    for (let w = 0; w < 3; w++) {
      const base = w * 6;
      for (let k = 0; k < 6; k++) {
        const idx = base + k;
        const nIdx = base + ((k + 1) % 6);
        const diffP = this.petalPitchOffset[nIdx] - this.petalPitchOffset[idx];
        const diffS = this.petalSwayOffset[nIdx] - this.petalSwayOffset[idx];
        this.petalPitchVel[idx] += diffP * 3.8 * dt;
        this.petalPitchVel[nIdx] -= diffP * 3.8 * dt;
        this.petalSwayVel[idx] += diffS * 3.2 * dt;
        this.petalSwayVel[nIdx] -= diffS * 3.2 * dt;
      }
    }

    this.emberTarget += (1.0 - this.emberTarget) * (1.0 - Math.exp(-dt * 1.4));
    this.emberGlow += (this.emberTarget - this.emberGlow) * (1.0 - Math.exp(-dt * 5.8));
    this.airWakeStrength = Math.max(0, this.airWakeStrength - dt * 1.15);
    this.airWakeVel.multiplyScalar(Math.exp(-dt * 5.0));
    if (this.airWakeStrength <= 0.01) {
      this.candleOffset.multiplyScalar(Math.exp(-dt * 2.0));
    }

    this.updatePetals(dt);
    this.updateMoths(dt);
    this.updatePollen(dt);
  }

  private updatePetals(_dt: number): void {
    const t = this.elapsedSec;

    for (let p = 0; p < this.petals.length; p++) {
      const pet = this.petals[p];
      const spec = pet.spec;

      const endWin = 1.0 - (0.38 - spec.openDelay) * 0.32;
      const localBloom = smoothstepCpu(spec.openDelay, Math.max(spec.openDelay + 0.35, endWin), this.bloom);
      this.petalLocalBloom[p] = localBloom;

      const posArr = pet.posAttr.array as Float32Array;
      const normArr = pet.normAttr.array as Float32Array;
      const cosAz = Math.cos(spec.azimuth);
      const sinAz = Math.sin(spec.azimuth);

      const baseOpenAng =
        spec.closedTilt +
        (spec.openTilt - spec.closedTilt) * Math.pow(localBloom, 0.85) +
        this.petalPitchOffset[p];
      const sway = this.petalSwayOffset[p];
      const closedWeight = 1.0 - localBloom;

      for (let i = 0; i < pet.vertexCount; i++) {
        const lx = pet.restLx[i];
        const ly = pet.restLy[i];
        const cupZ = pet.restCupZ[i];
        const v = pet.restV[i];

        const breezeWave =
          Math.sin(t * 1.55 - v * 2.7 + spec.azimuth * 2.0 + spec.whorl * 0.9) *
          Math.pow(v, 1.45) *
          (0.015 + 0.048 * localBloom);

        const teardropSwell =
          Math.sin(Math.pow(v, 0.72) * Math.PI) *
          (0.78 - spec.whorl * 0.12) *
          closedWeight;

        // Poised upward-and-outward chalice curve (graceful horizontal-upward lotus cup, zero droop)
        const spineAngle =
          baseOpenAng * (0.58 + 0.42 * Math.pow(v, 0.85)) +
          spec.tipRecurve * Math.pow(localBloom, 1.1) * Math.pow(Math.max(0, v - 0.35), 1.6) +
          breezeWave;

        const rOpen = spec.baseRadius + ly * Math.sin(spineAngle);
        const yOpen = ly * Math.cos(spineAngle);

        const rClosed = spec.baseRadius + teardropSwell;
        const yClosed = ly * 0.96;

        const rCenter = rClosed * closedWeight + rOpen * localBloom;
        const yPos = PETAL_BASE_Y + (yClosed * closedWeight + yOpen * localBloom);

        const rLocal = rCenter + cupZ * Math.cos(spineAngle * 0.6);
        const yLocal = yPos - cupZ * Math.sin(spineAngle * 0.6) * 0.35;
        const lxSwayed = lx + sway * ly * 0.45;

        const wx = lxSwayed * cosAz + rLocal * sinAz;
        const wz = -lxSwayed * sinAz + rLocal * cosAz;

        const i3 = i * 3;
        posArr[i3] = wx;
        posArr[i3 + 1] = yLocal;
        posArr[i3 + 2] = wz;
      }

      recomputeMeshNormalsInPlace(pet.geometry, posArr, normArr);
      pet.posAttr.needsUpdate = true;
      pet.normAttr.needsUpdate = true;
    }
  }

  private updateMoths(dt: number): void {
    const openVis = smoothstepCpu(0.14, 0.48, this.bloom);
    const t = this.elapsedSec;
    const wakeX = this.airWakeLocal.x;
    const wakeY = this.airWakeLocal.y;
    const wakeZ = this.airWakeLocal.z;

    for (let i = 0; i < this.moths.length; i++) {
      const m = this.moths[i];
      m.liftProgress += (openVis - m.liftProgress) * (1.0 - Math.exp(-dt * 4.2));
      m.active = m.liftProgress > 0.04;

      if (m.burstTimer > 0) {
        m.burstTimer = Math.max(0, m.burstTimer - dt);
      }

      if (m.liftProgress < 0.08) {
        const restX = Math.cos(m.orbitPhase) * 0.08;
        const restZ = Math.sin(m.orbitPhase) * 0.08;
        m.pos.lerp(_tmpVec3.set(restX, PETAL_BASE_Y + 0.36, restZ), 1.0 - Math.exp(-dt * 5.0));
        m.wingPhase += dt * 4.0 * m.liftProgress;
        continue;
      }

      m.orbitPhase += m.orbitSpeed * dt;
      const burstLift = m.burstTimer > 0 ? Math.sin(Math.min(Math.PI, (m.burstTimer / 3.5) * Math.PI)) * 0.52 : 0;
      const spreadScale = (0.35 + 0.65 * this.bloom) * m.liftProgress;

      const targetR = m.orbitRadius * spreadScale + Math.sin(t * 0.9 + i * 1.7) * 0.12;
      const targetX = Math.cos(m.orbitPhase) * targetR + Math.sin(t * 1.3 + i) * 0.08;
      const targetZ = Math.sin(m.orbitPhase) * targetR + Math.cos(t * 1.1 + i * 2.1) * 0.08;
      const targetY =
        (PETAL_BASE_Y + 0.42) * (1 - m.liftProgress) +
        (m.orbitHeight * (0.65 + 0.45 * this.bloom) + Math.sin(t * 1.45 + i * 1.9) * 0.14 + burstLift) *
          m.liftProgress;

      evalCurlFlow(m.pos.x, m.pos.y, m.pos.z, t * 0.65, _curlOut);
      const ax = (targetX - m.pos.x) * 4.8 + _curlOut.x * 0.42;
      const ay = (targetY - m.pos.y) * 4.2 + _curlOut.y * 0.35;
      const az = (targetZ - m.pos.z) * 4.8 + _curlOut.z * 0.42;

      m.vel.x = (m.vel.x + ax * dt) * Math.exp(-3.2 * dt);
      m.vel.y = (m.vel.y + ay * dt) * Math.exp(-3.2 * dt);
      m.vel.z = (m.vel.z + az * dt) * Math.exp(-3.2 * dt);

      if (this.airWakeStrength > 0.04) {
        const dx = m.pos.x - wakeX;
        const dy = m.pos.y - wakeY;
        const dz = m.pos.z - wakeZ;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 2.2) {
          const wInf = Math.exp(-d2 * 1.6) * this.airWakeStrength * dt;
          m.vel.x += this.airWakeVel.x * wInf * 1.6 + dx * wInf * 0.9;
          m.vel.y += this.airWakeVel.y * wInf * 1.6 + 0.35 * wInf;
          m.vel.z += this.airWakeVel.z * wInf * 1.6 + dz * wInf * 0.9;
        }
      }

      m.pos.addScaledVector(m.vel, dt);

      const horizSpeed = Math.hypot(m.vel.x, m.vel.z);
      if (horizSpeed > 0.02) {
        const desiredHeading = Math.atan2(m.vel.x, m.vel.z);
        m.heading += angleDiff(desiredHeading, m.heading) * (1.0 - Math.exp(-dt * 6.5));
      }
      m.pitch = clamp(0.22 + m.vel.y * 0.42, -0.35, 0.72);
      m.roll = clamp(-m.vel.x * 0.55 + m.orbitSpeed * 0.25, -0.65, 0.65);

      const dynSpeed = m.wingSpeed + Math.min(6.0, (horizSpeed + Math.max(0, m.vel.y)) * 4.2);
      m.wingPhase += dt * dynSpeed;
    }
  }

  private updatePollen(dt: number): void {
    const openVis = smoothstepCpu(0.10, 0.50, this.bloom);
    const nPol = this.config.pollenCount;
    const t = this.elapsedSec;
    const wakeX = this.airWakeLocal.x;
    const wakeY = this.airWakeLocal.y;
    const wakeZ = this.airWakeLocal.z;

    for (let i = 0; i < nPol; i++) {
      this.pollenAge[i] += dt;
      if (this.pollenAge[i] >= this.pollenMaxAge[i] || this.pollenPos[i * 3 + 1] > 1.85) {
        this.respawnPollen(i, 0);
      }

      const i3 = i * 3;
      const px = this.pollenPos[i3];
      const py = this.pollenPos[i3 + 1];
      const pz = this.pollenPos[i3 + 2];

      evalCurlFlow(px * 1.25, py * 1.25, pz * 1.25, t * 0.55 + i * 0.03, _curlOut);

      let vx = this.pollenVel[i3] + (_curlOut.x * 0.28 + px * 0.04) * dt;
      let vy = this.pollenVel[i3 + 1] + (_curlOut.y * 0.22 + 0.05 * this.emberGlow) * dt;
      let vz = this.pollenVel[i3 + 2] + (_curlOut.z * 0.28 + pz * 0.04) * dt;

      if (this.airWakeStrength > 0.03) {
        const dx = px - wakeX;
        const dy = py - wakeY;
        const dz = pz - wakeZ;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 2.4) {
          const inf = Math.exp(-d2 * 1.7) * this.airWakeStrength * dt;
          vx += this.airWakeVel.x * inf * 1.45 - dz * inf * 0.85;
          vy += this.airWakeVel.y * inf * 1.45 + 0.22 * inf;
          vz += this.airWakeVel.z * inf * 1.45 + dx * inf * 0.85;
        }
      }

      vx *= Math.exp(-1.1 * dt);
      vy = clamp(vy * Math.exp(-0.4 * dt), 0.06, 0.85);
      vz *= Math.exp(-1.1 * dt);

      this.pollenVel[i3] = vx;
      this.pollenVel[i3 + 1] = vy;
      this.pollenVel[i3 + 2] = vz;

      this.pollenPos[i3] = px + vx * dt;
      this.pollenPos[i3 + 1] = py + vy * dt;
      this.pollenPos[i3 + 2] = pz + vz * dt;

      const lifeFrac = this.pollenAge[i] / Math.max(0.1, this.pollenMaxAge[i]);
      const fade = smoothstepCpu(0.0, 0.18, lifeFrac) * (1.0 - smoothstepCpu(0.68, 1.0, lifeFrac));
      this.pollenAlpha[i] = fade * openVis;
    }
  }

  /**
   * Orders rear petals (2..18) behind the stamen crown (20) and front sheer petals (22..38) in front
   * so translucent rice paper naturally diffuses the glowing stamens behind it.
   */
  applyBackToFrontRenderOrder(
    camPosLocal: THREE.Vector3,
    petalMeshes: readonly THREE.Mesh[]
  ): void {
    const cx = camPosLocal.x;
    const cy = camPosLocal.y;
    const cz = camPosLocal.z;
    const camCenterDist = Math.hypot(cx, cy, cz);

    for (let p = 0; p < this.petals.length; p++) {
      const pet = this.petals[p];
      const midIdx = Math.floor(pet.vertexCount * 0.45) * 3;
      const pos = pet.posAttr.array as Float32Array;
      const mx = pos[midIdx];
      const my = pos[midIdx + 1];
      const mz = pos[midIdx + 2];
      this.sortDistances[p] = Math.hypot(cx - mx, cy - my, cz - mz) - pet.spec.whorl * 0.02;
    }

    for (let p = 0; p < this.petals.length; p++) {
      let rank = 0;
      const d = this.sortDistances[p];
      for (let q = 0; q < this.petals.length; q++) {
        if (this.sortDistances[q] > d || (this.sortDistances[q] === d && q < p)) rank++;
      }
      const isBehindCenter = d > camCenterDist;
      petalMeshes[p].renderOrder = isBehindCenter ? 2 + rank : 22 + rank;
    }
  }

  getTelemetry(): Day20TelemetryState {
    let activeMothCount = 0;
    for (let i = 0; i < this.moths.length; i++) {
      if (this.moths[i].active) activeMothCount++;
    }
    let activePollenCount = 0;
    for (let i = 0; i < this.config.pollenCount; i++) {
      if (this.pollenAlpha[i] > 0.04) activePollenCount++;
    }
    return {
      bloom: Number(this.bloom.toFixed(4)),
      targetBloom: Number(this.targetBloom.toFixed(4)),
      petalCount: this.petals.length,
      activeMothCount,
      activePollenCount,
      emberGlow: Number(this.emberGlow.toFixed(4)),
    };
  }

  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    for (const p of this.petals) p.geometry.dispose();
  }
}

const _tmpVec2 = new THREE.Vector2();
const _tmpVec3 = new THREE.Vector3();
const _curlOut = new THREE.Vector3();

function evalCurlFlow(x: number, y: number, z: number, t: number, out: THREE.Vector3): void {
  const sx = Math.sin(y * 2.1 + t * 1.1) - Math.cos(z * 1.9 - t * 0.9);
  const sy = Math.sin(z * 2.0 + t * 0.85) - Math.cos(x * 2.2 + t * 1.05);
  const sz = Math.sin(x * 1.8 - t * 0.95) - Math.cos(y * 2.3 + t * 0.8);
  out.set(sx * 0.45, sy * 0.35 + 0.15, sz * 0.45);
}

function fract(x: number): number {
  return x - Math.floor(x);
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function angleDiff(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

function recomputeMeshNormalsInPlace(
  geometry: THREE.BufferGeometry,
  pos: Float32Array,
  norm: Float32Array
): void {
  norm.fill(0);
  const index = geometry.getIndex();
  if (!index) return;
  const idx = index.array;
  const nIdx = idx.length;

  for (let i = 0; i < nIdx; i += 3) {
    const a = idx[i] * 3;
    const b = idx[i + 1] * 3;
    const c = idx[i + 2] * 3;

    const ax = pos[a];
    const ay = pos[a + 1];
    const az = pos[a + 2];

    const e1x = pos[b] - ax;
    const e1y = pos[b + 1] - ay;
    const e1z = pos[b + 2] - az;

    const e2x = pos[c] - ax;
    const e2y = pos[c + 1] - ay;
    const e2z = pos[c + 2] - az;

    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;

    norm[a] += nx;
    norm[a + 1] += ny;
    norm[a + 2] += nz;

    norm[b] += nx;
    norm[b + 1] += ny;
    norm[b + 2] += nz;

    norm[c] += nx;
    norm[c + 1] += ny;
    norm[c + 2] += nz;
  }

  for (let i = 0; i < norm.length; i += 3) {
    const nx = norm[i];
    const ny = norm[i + 1];
    const nz = norm[i + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-8) {
      const inv = 1.0 / len;
      norm[i] = nx * inv;
      norm[i + 1] = ny * inv;
      norm[i + 2] = nz * inv;
    } else {
      norm[i] = 0;
      norm[i + 1] = 1;
      norm[i + 2] = 0;
    }
  }
}
