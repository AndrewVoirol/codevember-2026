import * as THREE from 'three/webgpu';
import {
  WAKE_SLOT_COUNT,
  SEA_STATE_PRESETS,
  applySeaStateToUniforms,
  blendSeaStates,
  evaluateWaveCPU,
  type SeaStateConfig,
  type SeaStateId,
  type WaveFieldUniforms,
  type WaveSampleCPU,
} from './waveField';

export type InteractionTool = 'wake' | 'orbit';

export interface CameraPose {
  azimuth: number;
  elevation: number;
  distance: number;
}

export interface OceanRayHit {
  region: 'water' | 'sky';
  x: number;
  z: number;
}

export interface SeaOfCSnapshot {
  seaState: SeaStateId;
  tool: InteractionTool;
  azimuth: number;
  elevation: number;
  distance: number;
  cameraX: number;
  cameraY: number;
  cameraZ: number;
  sunAzimuth: number;
  sunElevation: number;
  flipP: number;
  punWarm: boolean;
  activeWakeCount: number;
  vortexStrength: number;
  vortexX: number;
  vortexZ: number;
  rogueWaveAge: number;
  simTime: number;
  wallTime: number;
  transitionProgress: number;
}

export const HERO_CAMERA_TARGET = {
  x: 0,
  y: 1.1,
  z: -6.0,
} as const;

// Low-skimming marine hero pose: pos ≈ (0, 3.9, 13.8) looking toward (0, 1.1, -6.0)
export const HERO_CAMERA_POSE: Readonly<CameraPose> = {
  azimuth: 0,
  elevation: 0.14046,
  distance: 20.0,
};

export const MIN_CAMERA_ELEVATION = 0.06;
export const MAX_CAMERA_ELEVATION = 0.92;
export const MIN_CAMERA_DISTANCE = 4.5;
export const MAX_CAMERA_DISTANCE = 34.0;
export const CAMERA_WAVE_CLEARANCE = 1.35;

export const SUN_AZIMUTH_LIMIT = 0.65;
export const MIN_SUN_ELEVATION = 0.05;
export const MAX_SUN_ELEVATION = 0.34;

export const WAKE_MIN_DIST_M = 0.55;
export const WAKE_MIN_INTERVAL_SEC = 0.045;
export const WAKE_LIFETIME_SEC = 5.5;
export const WAKE_MAX_RADIUS_M = 95.0;

export const SEA_STATE_TRANSITION_SEC = 1.1;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function cloneSeaStateConfig(src: SeaStateConfig): SeaStateConfig {
  return {
    id: src.id,
    label: src.label,
    waveAmpScale: src.waveAmpScale,
    chopScale: src.chopScale,
    breakerHeight: src.breakerHeight,
    sunAzimuth: src.sunAzimuth,
    sunElevation: src.sunElevation,
    sunColor: [src.sunColor[0], src.sunColor[1], src.sunColor[2]],
    skyZenith: [src.skyZenith[0], src.skyZenith[1], src.skyZenith[2]],
    skyHorizon: [src.skyHorizon[0], src.skyHorizon[1], src.skyHorizon[2]],
    waterDeep: [src.waterDeep[0], src.waterDeep[1], src.waterDeep[2]],
    waterShallow: [src.waterShallow[0], src.waterShallow[1], src.waterShallow[2]],
    glintColor: [src.glintColor[0], src.glintColor[1], src.glintColor[2]],
    foamColor: [src.foamColor[0], src.foamColor[1], src.foamColor[2]],
    biolumGain: src.biolumGain,
    fogDensity: src.fogDensity,
    speedScale: src.speedScale,
  };
}

function copySeaStateConfig(src: SeaStateConfig, dst: SeaStateConfig): void {
  dst.id = src.id;
  dst.label = src.label;
  dst.waveAmpScale = src.waveAmpScale;
  dst.chopScale = src.chopScale;
  dst.breakerHeight = src.breakerHeight;
  dst.sunAzimuth = src.sunAzimuth;
  dst.sunElevation = src.sunElevation;
  dst.sunColor[0] = src.sunColor[0];
  dst.sunColor[1] = src.sunColor[1];
  dst.sunColor[2] = src.sunColor[2];
  dst.skyZenith[0] = src.skyZenith[0];
  dst.skyZenith[1] = src.skyZenith[1];
  dst.skyZenith[2] = src.skyZenith[2];
  dst.skyHorizon[0] = src.skyHorizon[0];
  dst.skyHorizon[1] = src.skyHorizon[1];
  dst.skyHorizon[2] = src.skyHorizon[2];
  dst.waterDeep[0] = src.waterDeep[0];
  dst.waterDeep[1] = src.waterDeep[1];
  dst.waterDeep[2] = src.waterDeep[2];
  dst.waterShallow[0] = src.waterShallow[0];
  dst.waterShallow[1] = src.waterShallow[1];
  dst.waterShallow[2] = src.waterShallow[2];
  dst.glintColor[0] = src.glintColor[0];
  dst.glintColor[1] = src.glintColor[1];
  dst.glintColor[2] = src.glintColor[2];
  dst.foamColor[0] = src.foamColor[0];
  dst.foamColor[1] = src.foamColor[1];
  dst.foamColor[2] = src.foamColor[2];
  dst.biolumGain = src.biolumGain;
  dst.fogDensity = src.fogDensity;
  dst.speedScale = src.speedScale;
}

export class SeaOfCController {
  public readonly uniforms: WaveFieldUniforms;
  public readonly cameraPosition = new THREE.Vector3();
  public readonly cameraTarget = new THREE.Vector3(
    HERO_CAMERA_TARGET.x,
    HERO_CAMERA_TARGET.y,
    HERO_CAMERA_TARGET.z
  );

  private _seaState: SeaStateId;
  private _tool: InteractionTool = 'wake';

  // Smooth sea-state crossfade state (zero allocations during step)
  private readonly _fromConfig: SeaStateConfig;
  private readonly _blendedConfig: SeaStateConfig;
  private _transitionProgress = 1;

  // Critically damped spherical orbit camera
  private _azimuth = HERO_CAMERA_POSE.azimuth;
  private _elevation = HERO_CAMERA_POSE.elevation;
  private _distance = HERO_CAMERA_POSE.distance;
  private _targetAzimuth = HERO_CAMERA_POSE.azimuth;
  private _targetElevation = HERO_CAMERA_POSE.elevation;
  private _targetDistance = HERO_CAMERA_POSE.distance;
  private _azVel = 0;
  private _elVel = 0;
  private _distVel = 0;

  // Interactive sky-drag sun steering
  private _curSunAz: number;
  private _curSunEl: number;
  private _targetSunAz: number;
  private _targetSunEl: number;
  private _hasManualSun = false;

  // 8-slot wake ring buffer state
  private _nextWakeSlot = 0;
  private _hasLastWakePos = false;
  private _lastWakeX = 0;
  private _lastWakeZ = 0;
  private _lastWakeWallTime = -10;

  // Interactive maelstrom vortex state
  private _isVortexHeld = false;
  private _vortexAutoHoldSec = 0;
  private _vortexX = 0;
  private _vortexZ = 0;
  private _vortexTargetX = 0;
  private _vortexTargetZ = 0;
  private _vortexStrength = 0;
  private _vortexTargetStrength = 1;
  private _vortexSpinPhase = 0;

  // Rogue breaker state
  private _lastRogueBirthTime = -100;

  // Pun slot-flip choreography ("sea" <-> "C")
  private _flipP = 0;
  private _punWarm = false;

  // Clocks
  private _simTime = 0;
  private _wallTime = 0;

  // Pre-allocated CPU wave sampling scratch buffers (zero GC)
  private readonly _waveSample: WaveSampleCPU = {
    dx: 0,
    dy: 0,
    dz: 0,
    nx: 0,
    ny: 1,
    nz: 0,
    crestEnergy: 0,
    curlFactor: 0,
    wakeEnergy: 0,
    orbitalPhase: 0,
  };
  private readonly _probeSample: WaveSampleCPU = {
    dx: 0,
    dy: 0,
    dz: 0,
    nx: 0,
    ny: 1,
    nz: 0,
    crestEnergy: 0,
    curlFactor: 0,
    wakeEnergy: 0,
    orbitalPhase: 0,
  };
  private readonly _snapshot: SeaOfCSnapshot;

  constructor(uniforms: WaveFieldUniforms, initialState: SeaStateId = 'golden') {
    this.uniforms = uniforms;
    this._seaState = initialState;

    const base = SEA_STATE_PRESETS[initialState];
    this._fromConfig = cloneSeaStateConfig(base);
    this._blendedConfig = cloneSeaStateConfig(base);
    this._curSunAz = base.sunAzimuth;
    this._curSunEl = base.sunElevation;
    this._targetSunAz = base.sunAzimuth;
    this._targetSunEl = base.sunElevation;

    // Ensure inactive wake slots start cleanly expired
    for (let i = 0; i < WAKE_SLOT_COUNT; i++) {
      if (this.uniforms.wakeData[i]) {
        this.uniforms.wakeData[i].set(0, 0, -100, 0);
      }
    }
    this.uniforms.uVortex.value.set(0, 0, 0, 0);
    this.uniforms.uRogueWave.value.set(0, 0, -100, 0);
    this.uniforms.uTime.value = 0;

    applySeaStateToUniforms(this.uniforms, this._blendedConfig);
    this._updateCameraPosition();

    this._snapshot = {
      seaState: this._seaState,
      tool: this._tool,
      azimuth: this._azimuth,
      elevation: this._elevation,
      distance: this._distance,
      cameraX: this.cameraPosition.x,
      cameraY: this.cameraPosition.y,
      cameraZ: this.cameraPosition.z,
      sunAzimuth: this._curSunAz,
      sunElevation: this._curSunEl,
      flipP: this._flipP,
      punWarm: this._punWarm,
      activeWakeCount: 0,
      vortexStrength: 0,
      vortexX: 0,
      vortexZ: 0,
      rogueWaveAge: 100,
      simTime: 0,
      wallTime: 0,
      transitionProgress: 1,
    };
  }

  public get seaState(): SeaStateId {
    return this._seaState;
  }

  public get tool(): InteractionTool {
    return this._tool;
  }

  public get simTime(): number {
    return this._simTime;
  }

  public get wallTime(): number {
    return this._wallTime;
  }

  public get flipP(): number {
    return this._flipP;
  }

  public get vortexStrength(): number {
    return this._vortexStrength;
  }

  public get blendedConfig(): Readonly<SeaStateConfig> {
    return this._blendedConfig;
  }

  public setTool(tool: InteractionTool): void {
    this._tool = tool;
  }

  public toggleTool(): InteractionTool {
    this._tool = this._tool === 'wake' ? 'orbit' : 'wake';
    return this._tool;
  }

  public setSeaState(state: SeaStateId, immediate = false): void {
    if (!(state in SEA_STATE_PRESETS)) return;
    copySeaStateConfig(this._blendedConfig, this._fromConfig);
    this._seaState = state;
    this._hasManualSun = false;

    if (immediate) {
      this._transitionProgress = 1;
      const target = SEA_STATE_PRESETS[state];
      copySeaStateConfig(target, this._blendedConfig);
      this._curSunAz = target.sunAzimuth;
      this._curSunEl = target.sunElevation;
      this._targetSunAz = target.sunAzimuth;
      this._targetSunEl = target.sunElevation;
      applySeaStateToUniforms(this.uniforms, this._blendedConfig);
      this._updateCameraPosition();
    } else {
      this._transitionProgress = 0;
    }
  }

  public setCameraPose(
    azimuth: number,
    elevation: number,
    distance: number,
    immediate = false
  ): void {
    const safeAz = Number.isFinite(azimuth) ? azimuth : HERO_CAMERA_POSE.azimuth;
    const safeEl = clamp(
      Number.isFinite(elevation) ? elevation : HERO_CAMERA_POSE.elevation,
      MIN_CAMERA_ELEVATION,
      MAX_CAMERA_ELEVATION
    );
    const safeDist = clamp(
      Number.isFinite(distance) ? distance : HERO_CAMERA_POSE.distance,
      MIN_CAMERA_DISTANCE,
      MAX_CAMERA_DISTANCE
    );

    this._targetAzimuth = safeAz;
    this._targetElevation = safeEl;
    this._targetDistance = safeDist;

    if (immediate) {
      this._azimuth = safeAz;
      this._elevation = safeEl;
      this._distance = safeDist;
      this._azVel = 0;
      this._elVel = 0;
      this._distVel = 0;
      this._updateCameraPosition();
    }
  }

  public resetToHero(immediate = false): void {
    this._hasManualSun = false;
    this.setCameraPose(
      HERO_CAMERA_POSE.azimuth,
      HERO_CAMERA_POSE.elevation,
      HERO_CAMERA_POSE.distance,
      immediate
    );
    const target = SEA_STATE_PRESETS[this._seaState];
    this._targetSunAz = target.sunAzimuth;
    this._targetSunEl = target.sunElevation;
    if (immediate) {
      this._curSunAz = target.sunAzimuth;
      this._curSunEl = target.sunElevation;
      this._blendedConfig.sunAzimuth = this._curSunAz;
      this._blendedConfig.sunElevation = this._curSunEl;
      applySeaStateToUniforms(this.uniforms, this._blendedConfig);
    }
  }

  public orbitBy(deltaAzimuth: number, deltaElevation: number): void {
    if (!Number.isFinite(deltaAzimuth) || !Number.isFinite(deltaElevation)) return;
    this._targetAzimuth += deltaAzimuth;
    this._targetElevation = clamp(
      this._targetElevation + deltaElevation,
      MIN_CAMERA_ELEVATION,
      MAX_CAMERA_ELEVATION
    );
  }

  public zoomBy(deltaDistance: number): void {
    if (!Number.isFinite(deltaDistance)) return;
    this._targetDistance = clamp(
      this._targetDistance + deltaDistance,
      MIN_CAMERA_DISTANCE,
      MAX_CAMERA_DISTANCE
    );
  }

  public steerSunBy(deltaAzimuth: number, deltaElevation: number): void {
    if (!Number.isFinite(deltaAzimuth) || !Number.isFinite(deltaElevation)) return;
    if (!this._hasManualSun) {
      this._targetSunAz = this._curSunAz;
      this._targetSunEl = this._curSunEl;
      this._hasManualSun = true;
    }
    this._targetSunAz = clamp(
      this._targetSunAz + deltaAzimuth,
      -SUN_AZIMUTH_LIMIT,
      SUN_AZIMUTH_LIMIT
    );
    this._targetSunEl = clamp(
      this._targetSunEl + deltaElevation,
      MIN_SUN_ELEVATION,
      MAX_SUN_ELEVATION
    );
  }

  /**
   * Intersects a world-space ray with the displaced ocean surface (1-step wave refinement).
   * Rays at or above the horizon classify as 'sky' so dragging the sky steers the sun.
   */
  public intersectRayWithOcean(
    rayOrigin: THREE.Vector3,
    rayDir: THREE.Vector3,
    out: OceanRayHit
  ): OceanRayHit {
    if (rayDir.y >= -0.015) {
      out.region = 'sky';
      out.x = 0;
      out.z = 0;
      return out;
    }

    const t0 = -rayOrigin.y / rayDir.y;
    if (!Number.isFinite(t0) || t0 <= 0.1 || t0 > 135) {
      out.region = 'sky';
      out.x = 0;
      out.z = 0;
      return out;
    }

    const x0 = rayOrigin.x + rayDir.x * t0;
    const z0 = rayOrigin.z + rayDir.z * t0;
    evaluateWaveCPU(x0, z0, this.uniforms, this._probeSample);

    const t1 = clamp((this._probeSample.dy - rayOrigin.y) / rayDir.y, 0.2, 135);
    const x1 = rayOrigin.x + rayDir.x * t1;
    const z1 = rayOrigin.z + rayDir.z * t1;

    out.region = 'water';
    out.x = x1;
    out.z = z1;
    return out;
  }

  /**
   * Throttled hover/drag wake emitter: spawns into the 8-slot uWakeSlots ring buffer
   * only when pointer travel >= 0.55m AND elapsed time >= 45ms.
   */
  public handleWaterMove(worldX: number, worldZ: number, isPressed = false): boolean {
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return false;
    if (Math.hypot(worldX, worldZ) > WAKE_MAX_RADIUS_M) return false;

    const dtSinceLast = this._wallTime - this._lastWakeWallTime;
    if (this._hasLastWakePos) {
      const dist = Math.hypot(worldX - this._lastWakeX, worldZ - this._lastWakeZ);
      if (dist < WAKE_MIN_DIST_M || dtSinceLast < WAKE_MIN_INTERVAL_SEC) {
        return false;
      }
      const speedBoost = clamp((dist / Math.max(0.045, dtSinceLast)) * 0.045, 0, 0.35);
      const strength = clamp((isPressed ? 1.05 : 0.72) + speedBoost, 0.55, 1.35);
      this._emitWakeSlot(worldX, worldZ, strength);
      return true;
    }

    this._emitWakeSlot(worldX, worldZ, isPressed ? 1.05 : 0.75);
    return true;
  }

  /** Immediate unthrottled wake spawn (for pointerdown / rogue bursts / QA API). */
  public triggerWake(worldX: number, worldZ: number, strength = 1.0): void {
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return;
    const clampedStrength = clamp(Number.isFinite(strength) ? strength : 1.0, 0.1, 2.0);
    this._emitWakeSlot(worldX, worldZ, clampedStrength);
  }

  public beginWaterDrag(worldX: number, worldZ: number): void {
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return;
    this._isVortexHeld = true;
    this._vortexAutoHoldSec = 0;
    this._vortexTargetX = worldX;
    this._vortexTargetZ = worldZ;
    this._vortexTargetStrength = 1.0;

    if (this._vortexStrength < 0.08) {
      this._vortexX = worldX;
      this._vortexZ = worldZ;
    }
    this._emitWakeSlot(worldX, worldZ, 1.15);
  }

  public updateWaterDrag(worldX: number, worldZ: number): void {
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return;
    this._vortexTargetX = worldX;
    this._vortexTargetZ = worldZ;
    this.handleWaterMove(worldX, worldZ, true);
  }

  public endWaterDrag(): void {
    this._isVortexHeld = false;
  }

  public triggerVortex(worldX: number, worldZ: number, strength = 1.0): void {
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return;
    const s = clamp(Number.isFinite(strength) ? strength : 1.0, 0.1, 1.8);
    this._vortexX = worldX;
    this._vortexZ = worldZ;
    this._vortexTargetX = worldX;
    this._vortexTargetZ = worldZ;
    this._vortexTargetStrength = s;
    this._vortexStrength = Math.max(this._vortexStrength, s * 0.75);
    this._vortexAutoHoldSec = 1.8;
    this.uniforms.uVortex.value.set(
      this._vortexX,
      this._vortexZ,
      this._vortexStrength,
      this._vortexSpinPhase
    );
  }

  public triggerRogueWave(worldX: number, worldZ: number, amplitude = 1.0): void {
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return;
    const amp = clamp(Number.isFinite(amplitude) ? amplitude : 1.0, 0.25, 2.0);
    this._lastRogueBirthTime = this._simTime;
    this.uniforms.uRogueWave.value.set(worldX, worldZ, this._simTime, amp);
    this._emitWakeSlot(worldX, worldZ, Math.min(1.4, 0.95 * amp));
  }

  public triggerRogueWaveAhead(amplitude = 1.15): void {
    const fwdX = -Math.sin(this._azimuth);
    const fwdZ = -Math.cos(this._azimuth);
    const x = this.cameraTarget.x + fwdX * 1.5;
    const z = this.cameraTarget.z + fwdZ * 1.5;
    this.triggerRogueWave(x, z, amplitude);
  }

  public getActiveWakeCount(): number {
    let count = 0;
    for (let i = 0; i < WAKE_SLOT_COUNT; i++) {
      const slot = this.uniforms.wakeData[i];
      if (!slot) continue;
      const age = this._simTime - slot.z;
      if (slot.w > 0.01 && age >= 0 && age < WAKE_LIFETIME_SEC) {
        count++;
      }
    }
    return count;
  }

  public step(rawDtSec: number): SeaOfCSnapshot {
    const dt = clamp(Number.isFinite(rawDtSec) ? rawDtSec : 0.016, 0.0001, 0.1);
    this._wallTime += dt;

    // 1. Smooth sea-state crossfade (~1.1s cubic smoothstep, zero shader rebuilds)
    const targetPreset = SEA_STATE_PRESETS[this._seaState];
    if (this._transitionProgress < 1) {
      this._transitionProgress = Math.min(
        1,
        this._transitionProgress + dt / SEA_STATE_TRANSITION_SEC
      );
      const t = this._transitionProgress;
      const eased = t * t * (3 - 2 * t);
      blendSeaStates(this._fromConfig, targetPreset, eased, this._blendedConfig);
    } else {
      copySeaStateConfig(targetPreset, this._blendedConfig);
    }

    // 2. Smooth sun steering (follows preset unless user dragged sky)
    if (!this._hasManualSun) {
      this._targetSunAz = this._blendedConfig.sunAzimuth;
      this._targetSunEl = this._blendedConfig.sunElevation;
    }
    const sunK = 1 - Math.exp(-dt * 8.0);
    this._curSunAz += (this._targetSunAz - this._curSunAz) * sunK;
    this._curSunEl += (this._targetSunEl - this._curSunEl) * sunK;
    this._blendedConfig.sunAzimuth = this._curSunAz;
    this._blendedConfig.sunElevation = this._curSunEl;

    // Advance ocean simulation clock scaled by blended sea-state tempo
    this._simTime += dt * this._blendedConfig.speedScale;
    this.uniforms.uTime.value = this._simTime;
    applySeaStateToUniforms(this.uniforms, this._blendedConfig);

    // 3. Critically damped spherical camera orbit & zoom
    const omega = 9.5;
    const exp = Math.exp(-omega * dt);

    const azDiff = this._azimuth - this._targetAzimuth;
    const azTemp = (this._azVel + omega * azDiff) * dt;
    this._azVel = (this._azVel - omega * azTemp) * exp;
    this._azimuth = this._targetAzimuth + (azDiff + azTemp) * exp;

    const elDiff = this._elevation - this._targetElevation;
    const elTemp = (this._elVel + omega * elDiff) * dt;
    this._elVel = (this._elVel - omega * elTemp) * exp;
    this._elevation = clamp(
      this._targetElevation + (elDiff + elTemp) * exp,
      MIN_CAMERA_ELEVATION,
      MAX_CAMERA_ELEVATION
    );

    const distDiff = this._distance - this._targetDistance;
    const distTemp = (this._distVel + omega * distDiff) * dt;
    this._distVel = (this._distVel - omega * distTemp) * exp;
    this._distance = clamp(
      this._targetDistance + (distDiff + distTemp) * exp,
      MIN_CAMERA_DISTANCE,
      MAX_CAMERA_DISTANCE
    );

    // 4. Interactive maelstrom vortex ramp / release decay
    if (this._vortexAutoHoldSec > 0) {
      this._vortexAutoHoldSec = Math.max(0, this._vortexAutoHoldSec - dt);
    }
    const vortexDriven = this._isVortexHeld || this._vortexAutoHoldSec > 0;
    if (vortexDriven) {
      const rampK = 1 - Math.exp(-dt * 4.6);
      this._vortexStrength += (this._vortexTargetStrength - this._vortexStrength) * rampK;
      const trackK = 1 - Math.exp(-dt * 12.0);
      this._vortexX += (this._vortexTargetX - this._vortexX) * trackK;
      this._vortexZ += (this._vortexTargetZ - this._vortexZ) * trackK;
      this._vortexSpinPhase += dt * (2.6 + 2.4 * this._vortexStrength);
    } else if (this._vortexStrength > 0) {
      this._vortexStrength *= Math.exp(-dt * 1.55);
      this._vortexSpinPhase += dt * (1.2 + 2.0 * this._vortexStrength);
      if (this._vortexStrength < 0.002) {
        this._vortexStrength = 0;
      }
    }
    this.uniforms.uVortex.value.set(
      this._vortexX,
      this._vortexZ,
      this._vortexStrength,
      this._vortexSpinPhase
    );

    // 5. Update camera world position with guaranteed wave-surface clearance
    this._updateCameraPosition();

    // 6. Pun slot-flip choreography ("sea" <-> "C") synced to fg-right breaker crest, vortex, rogue surge, or macro zoom
    evaluateWaveCPU(6.8, 4.1, this.uniforms, this._probeSample);
    const breakerPeak = this._probeSample.curlFactor > 0.88 && this._wallTime > 1.2;
    const rogueAge = this._simTime - this._lastRogueBirthTime;
    const cyclePhase = (this._wallTime % 6.8) / 6.8;
    const cycleWantsC = cyclePhase >= 0.28 && cyclePhase <= 0.72;
    const macroClose = this._distance < 10.5;
    const interactivePulse = this._vortexStrength > 0.22 || (rogueAge >= 0 && rogueAge < 3.0);

    const targetFlip = cycleWantsC || breakerPeak || interactivePulse || macroClose ? 1 : 0;
    const flipK = 1 - Math.exp(-dt * 5.0);
    this._flipP = clamp(this._flipP + (targetFlip - this._flipP) * flipK, 0, 1);
    if (this._flipP < 0.002 && targetFlip === 0) this._flipP = 0;
    if (this._flipP > 0.998 && targetFlip === 1) this._flipP = 1;
    this._punWarm = targetFlip === 1 || this._flipP > 0.12 || this._vortexStrength > 0.1;

    return this.getSnapshot();
  }

  public getSnapshot(): SeaOfCSnapshot {
    const s = this._snapshot;
    s.seaState = this._seaState;
    s.tool = this._tool;
    s.azimuth = this._azimuth;
    s.elevation = this._elevation;
    s.distance = this._distance;
    s.cameraX = this.cameraPosition.x;
    s.cameraY = this.cameraPosition.y;
    s.cameraZ = this.cameraPosition.z;
    s.sunAzimuth = this._curSunAz;
    s.sunElevation = this._curSunEl;
    s.flipP = this._flipP;
    s.punWarm = this._punWarm;
    s.activeWakeCount = this.getActiveWakeCount();
    s.vortexStrength = this._vortexStrength;
    s.vortexX = this._vortexX;
    s.vortexZ = this._vortexZ;
    s.rogueWaveAge = Math.max(0, this._simTime - this._lastRogueBirthTime);
    s.simTime = this._simTime;
    s.wallTime = this._wallTime;
    s.transitionProgress = this._transitionProgress;
    return s;
  }

  private _emitWakeSlot(worldX: number, worldZ: number, strength: number): void {
    const slot = this.uniforms.wakeData[this._nextWakeSlot];
    if (slot) {
      slot.set(worldX, worldZ, this._simTime, strength);
    }
    this._nextWakeSlot = (this._nextWakeSlot + 1) % WAKE_SLOT_COUNT;
    this._hasLastWakePos = true;
    this._lastWakeX = worldX;
    this._lastWakeZ = worldZ;
    this._lastWakeWallTime = this._wallTime;
  }

  private _updateCameraPosition(): void {
    const cosEl = Math.cos(this._elevation);
    const sinEl = Math.sin(this._elevation);
    const sinAz = Math.sin(this._azimuth);
    const cosAz = Math.cos(this._azimuth);

    const camX = this.cameraTarget.x + this._distance * cosEl * sinAz;
    const rawCamY = this.cameraTarget.y + this._distance * sinEl;
    const camZ = this.cameraTarget.z + this._distance * cosEl * cosAz;

    evaluateWaveCPU(camX, camZ, this.uniforms, this._waveSample);
    const minCamY = this._waveSample.dy + CAMERA_WAVE_CLEARANCE;
    const safeCamY = rawCamY < minCamY ? minCamY : rawCamY;

    this.cameraPosition.set(camX, safeCamY, camZ);
  }
}
