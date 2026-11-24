import type RAPIER_NS from '@dimforge/rapier3d-compat';
import {
  DOWEL_CHAMFER_FRAC,
  buildOvoidPebbleHullPoints,
  generateReciprocalDomeLayout,
  type CeladonEggSpec,
  type ReciprocalDomeLayout,
  type ReciprocalRodSpec,
} from './reciprocalDome';

export type RapierModule = typeof RAPIER_NS;

export interface NestSimOptions {
  /** Wood-on-wood Coulomb friction coefficient (default 0.68) */
  woodFriction?: number;
  /** Plinth stone/brass friction coefficient (default 0.76) */
  plinthFriction?: number;
  /** Egg-on-wood friction coefficient (default 0.68) */
  eggFriction?: number;
  /** Timber density (default 0.85) */
  woodDensity?: number;
  /** Celadon stone egg density (default 0.90) */
  eggDensity?: number;
  /** Substeps per 60 Hz frame (default 3 -> 180 Hz internal integration) */
  substepsPer60Hz?: number;
  /** Rapier PGS solver iterations (default 28 for crisp sub-mm hardwood contact) */
  solverIterations?: number;
  /** Rapier internal PGS iterations (default 4) */
  internalPgsIterations?: number;
  /** Hardwood contact natural frequency in Hz (default 120 Hz -> stiff timber, zero squash) */
  contactNaturalFrequency?: number;
  /** Hardwood contact penetration allowance in meters (default 0.0001 m) */
  allowedLinearError?: number;
  /** Mild aerodynamic linear damping (default 0.18) */
  linearDamping?: number;
  /** Mild aerodynamic angular damping (default 0.25) */
  angularDamping?: number;
  /** Rod indices to omit on spawn (for adversarial keystone-removal tests) */
  removedRodIds?: readonly number[];
}

export interface Day24TelemetryState {
  intactRodCount: number;
  fallenRodCount: number;
  collapsed: boolean;
  maxRodDrift: number;
  activeGrabId: number | null;
  simTime: number;
}

export interface RodBodyRecord {
  readonly spec: ReciprocalRodSpec;
  readonly body: RAPIER_NS.RigidBody;
  readonly collider: RAPIER_NS.Collider;
  removed: boolean;
  fallen: boolean;
  drift: number;
  drop: number;
}

export interface EggBodyRecord {
  readonly spec: CeladonEggSpec;
  readonly body: RAPIER_NS.RigidBody;
  readonly collider: RAPIER_NS.Collider;
  drop: number;
}

const GRAVITY_Y = -9.81;
const FALLEN_DROP_THRESHOLD = 0.16;
const FALLEN_DRIFT_THRESHOLD = 0.26;
const FALLEN_Y_FLOOR = -0.18;
const COLLAPSED_ROD_THRESHOLD = 6;

const DEFAULT_SIM_OPTIONS: Required<Omit<NestSimOptions, 'removedRodIds'>> = {
  woodFriction: 0.68,
  plinthFriction: 0.76,
  eggFriction: 0.68,
  woodDensity: 0.85,
  eggDensity: 0.90,
  substepsPer60Hz: 3,
  solverIterations: 28,
  internalPgsIterations: 4,
  contactNaturalFrequency: 120,
  allowedLinearError: 0.0001,
  linearDamping: 0.18,
  angularDamping: 0.25,
};

/**
 * Pure quaternion-vector rotation without allocating: writes q * v * q^-1 into `out`.
 */
function rotateVecByQuatOut(
  vx: number,
  vy: number,
  vz: number,
  qx: number,
  qy: number,
  qz: number,
  qw: number,
  out: [number, number, number]
): void {
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
}

/**
 * True 3D rigid-body simulation of the 56-dowel reciprocal timber dome + 3 ovoid
 * Celadon & Bone-Calcite river-pebble eggs on a hollow circular plinth ring using
 * `@dimforge/rapier3d-compat` 0.21 (`roundCylinder` dowels + `convexHull` ovoid eggs).
 *
 * Strictly zero artificial weld joints, zero fake freeze latches, zero velocity clamps.
 */
export class ReciprocalNestSim {
  readonly rapier: RapierModule;
  readonly layout: ReciprocalDomeLayout;
  readonly options: Required<Omit<NestSimOptions, 'removedRodIds'>>;

  world!: RAPIER_NS.World;
  rods: RodBodyRecord[] = [];
  eggs: EggBodyRecord[] = [];

  private simTimeSec = 0;
  private accumulatorSec = 0;
  private fixedDt = 1 / 180;
  private isDisposed = false;

  // Active 3D grab-spring state (zero per-frame allocations)
  private activeGrabRodId: number | null = null;
  private readonly grabLocalPt: [number, number, number] = [0, 0, 0];
  private readonly grabTargetWorld: [number, number, number] = [0, 0, 0];
  private readonly scratchRot: [number, number, number] = [0, 0, 0];
  private readonly scratchWorldPt: [number, number, number] = [0, 0, 0];
  private readonly scratchArm: [number, number, number] = [0, 0, 0];
  private readonly scratchPtVel: [number, number, number] = [0, 0, 0];
  private readonly scratchImpulse: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };
  private readonly scratchPoint: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };

  // Cached telemetry updated after every step
  private intactRodCount = 0;
  private fallenRodCount = 0;
  private collapsed = false;
  private maxRodDrift = 0;
  private maxRodDrop = 0;

  constructor(
    rapier: RapierModule,
    layout: ReciprocalDomeLayout = generateReciprocalDomeLayout(),
    options: NestSimOptions = {}
  ) {
    this.rapier = rapier;
    this.layout = layout;
    this.options = {
      ...DEFAULT_SIM_OPTIONS,
      ...options,
    };
    this.fixedDt = 1 / (60 * Math.max(1, this.options.substepsPer60Hz));
    this.buildWorld(options.removedRodIds ?? []);
  }

  private buildWorld(removedRodIds: readonly number[]): void {
    const R = this.rapier;
    const o = this.options;
    const world = new R.World({ x: 0, y: GRAVITY_Y, z: 0 });
    world.timestep = this.fixedDt;
    world.integrationParameters.numSolverIterations = o.solverIterations;
    world.integrationParameters.numInternalPgsIterations = o.internalPgsIterations;
    world.integrationParameters.normalizedAllowedLinearError = o.allowedLinearError;
    world.integrationParameters.normalizedPredictionDistance = Math.max(0.0008, o.allowedLinearError * 6);
    // Calibrate Rapier contact stiffness for hardwood-on-hardwood & stone, scaled safely to the
    // active substep rate so single-substep (60 Hz) ablated runs stay strictly within Nyquist stability
    const ip = world.integrationParameters as RAPIER_NS.IntegrationParameters & {
      contact_natural_frequency?: number;
    };
    if (typeof ip.contact_natural_frequency !== 'undefined' || 'contact_natural_frequency' in Object.getPrototypeOf(ip)) {
      const safeMaxFreq = o.substepsPer60Hz <= 1 ? 30 : Math.round(o.substepsPer60Hz * 40);
      ip.contact_natural_frequency = Math.min(o.contactNaturalFrequency, safeMaxFreq);
    }

    // 1. Hollow circular Honed-Limestone & Dark-Brass plinth ring collider (48 harmonic segments)
    //    Center bore `r < plinth.innerRadius (1.12 m)` is 100% open void so collapsing rods & eggs plummet cleanly!
    const plinth = this.layout.plinth;
    const M = plinth.colliderSegments;
    const shelfOuterR = plinth.lipOuterRadius;
    const shelfMidR = 0.5 * (plinth.innerRadius + shelfOuterR);
    const shelfRadWidth = shelfOuterR - plinth.innerRadius;
    const shelfThickness = Math.abs(plinth.baseBottomY);
    const segArcLen = ((2 * Math.PI * shelfOuterR) / M) * 1.06;

    const plinthBody = world.createRigidBody(R.RigidBodyDesc.fixed());
    const lipThick = Math.max(0.055, plinth.lipOuterRadius - plinth.seatOuterRadius);
    const lipMidR = plinth.seatOuterRadius + lipThick * 0.5;
    const lipHalfH = (plinth.lipTopY + 0.01) * 0.5;
    const lipCenterY = plinth.lipTopY - lipHalfH;

    for (let m = 0; m < M; m++) {
      const th = (m / M) * Math.PI * 2;
      const cs = Math.cos(th);
      const sn = Math.sin(th);
      // Map local +X to radial (cos(th), 0, sin(th)) via rotation by -th around +Y
      const q = { x: 0, y: Math.sin(-th * 0.5), z: 0, w: Math.cos(-th * 0.5) };

      const shelfDesc = R.ColliderDesc.cuboid(
        shelfRadWidth * 0.5,
        shelfThickness * 0.5,
        segArcLen * 0.5
      )
        .setTranslation(shelfMidR * cs, -shelfThickness * 0.5, shelfMidR * sn)
        .setRotation(q)
        .setFriction(o.plinthFriction)
        .setRestitution(0.02);
      world.createCollider(shelfDesc, plinthBody);

      const lipDesc = R.ColliderDesc.cuboid(lipThick * 0.5, lipHalfH, segArcLen * 0.55)
        .setTranslation(lipMidR * cs, lipCenterY, lipMidR * sn)
        .setRotation(q)
        .setFriction(o.plinthFriction)
        .setRestitution(0.02);
      world.createCollider(lipDesc, plinthBody);
    }

    // 2. Dynamic Chamfered RoundCylinder Rigid Bodies for all 56 Timber Dowels
    //    Matches `buildDowelGeometry` flat chamfered end-grain profile 1:1 (0.0 mm gap/clip)
    const removedSet = new Set(removedRodIds);
    const rods: RodBodyRecord[] = [];
    for (let i = 0; i < this.layout.rods.length; i++) {
      const spec = this.layout.rods[i];
      const isRemoved = removedSet.has(spec.id);
      const bodyDesc = R.RigidBodyDesc.dynamic()
        .setTranslation(
          isRemoved ? 0 : spec.center[0],
          isRemoved ? -50 - i : spec.center[1],
          isRemoved ? 0 : spec.center[2]
        )
        .setRotation(spec.quat)
        .setLinearDamping(o.linearDamping)
        .setAngularDamping(o.angularDamping);
      const body = world.createRigidBody(bodyDesc);
      if (isRemoved) {
        body.setEnabled(false);
      }

      const bevelR = spec.radius * DOWEL_CHAMFER_FRAC;
      const colliderDesc = R.ColliderDesc.roundCylinder(
        spec.halfHeight,
        spec.radius - bevelR,
        bevelR
      )
        .setDensity(o.woodDensity)
        .setFriction(o.woodFriction)
        .setFrictionCombineRule(R.CoefficientCombineRule.Average)
        .setRestitution(0.02);
      const collider = world.createCollider(colliderDesc, body);

      rods.push({
        spec,
        body,
        collider,
        removed: isRemoved,
        fallen: false,
        drift: 0,
        drop: 0,
      });
    }

    // 3. Dynamic Convex-Hull Ovoid Rigid Bodies for the 3 Celadon & Bone-Calcite River-Pebble Eggs
    //    Built from exact `buildOvoidPebbleHullPoints` matching `buildPebbleEggGeometry` to 0.0 mm
    const eggs: EggBodyRecord[] = [];
    for (let i = 0; i < this.layout.eggs.length; i++) {
      const spec = this.layout.eggs[i];
      const bodyDesc = R.RigidBodyDesc.dynamic()
        .setTranslation(spec.pos[0], spec.pos[1], spec.pos[2])
        .setRotation(spec.rot)
        .setLinearDamping(o.linearDamping)
        .setAngularDamping(o.angularDamping);
      const body = world.createRigidBody(bodyDesc);

      const hullPoints = buildOvoidPebbleHullPoints(spec.radius, spec.visualScale, spec.seed);
      const hullDesc = R.ColliderDesc.convexHull(hullPoints);
      const colliderDesc = (hullDesc ?? R.ColliderDesc.ball(spec.radius))
        .setDensity(o.eggDensity)
        .setFriction(o.eggFriction)
        .setRestitution(0.02);
      const collider = world.createCollider(colliderDesc, body);

      eggs.push({ spec, body, collider, drop: 0 });
    }

    this.world = world;
    this.rods = rods;
    this.eggs = eggs;
    this.simTimeSec = 0;
    this.accumulatorSec = 0;
    this.activeGrabRodId = null;
    this.refreshTelemetry();
  }

  /**
   * Starts a 3D spring-tether grab on rod `rodId` anchored at its local-space point `localPt`.
   */
  startGrab(
    rodId: number,
    localPt: readonly [number, number, number],
    targetWorld: readonly [number, number, number]
  ): boolean {
    const rec = this.rods[rodId];
    if (!rec || rec.removed) return false;
    this.activeGrabRodId = rodId;
    this.grabLocalPt[0] = localPt[0];
    this.grabLocalPt[1] = localPt[1];
    this.grabLocalPt[2] = localPt[2];
    this.grabTargetWorld[0] = targetWorld[0];
    this.grabTargetWorld[1] = targetWorld[1];
    this.grabTargetWorld[2] = targetWorld[2];
    rec.body.wakeUp();
    return true;
  }

  updateGrabTarget(targetWorld: readonly [number, number, number]): void {
    this.grabTargetWorld[0] = targetWorld[0];
    this.grabTargetWorld[1] = targetWorld[1];
    this.grabTargetWorld[2] = targetWorld[2];
  }

  endGrab(): void {
    this.activeGrabRodId = null;
  }

  /**
   * Applies a crisp tap/poke impulse at a world point on a rod so the user can test
   * whether a timber dowel is loose or tightly locked under structural compression.
   */
  pokeRod(
    rodId: number,
    worldPt: readonly [number, number, number],
    dir: readonly [number, number, number],
    strength = 0.0085
  ): boolean {
    const rec = this.rods[rodId];
    if (!rec || rec.removed) return false;
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const mass = rec.body.mass();
    const scale = (strength * Math.max(0.001, mass)) / 0.003;
    this.scratchImpulse.x = (dir[0] / len) * scale;
    this.scratchImpulse.y = (dir[1] / len) * scale;
    this.scratchImpulse.z = (dir[2] / len) * scale;
    this.scratchPoint.x = worldPt[0];
    this.scratchPoint.y = worldPt[1];
    this.scratchPoint.z = worldPt[2];
    rec.body.wakeUp();
    rec.body.applyImpulseAtPoint(this.scratchImpulse, this.scratchPoint, true);
    return true;
  }

  /**
   * Removes specified rod IDs cleanly from the simulation (e.g., adversarial keystone extraction).
   */
  removeRods(rodIds: readonly number[]): void {
    for (let i = 0; i < rodIds.length; i++) {
      const id = rodIds[i];
      const rec = this.rods[id];
      if (!rec || rec.removed) continue;
      rec.removed = true;
      rec.body.setEnabled(false);
      if (this.activeGrabRodId === id) {
        this.activeGrabRodId = null;
      }
    }
    // Wake all remaining bodies so support loss propagates immediately with zero stale sleepers
    for (let i = 0; i < this.rods.length; i++) {
      if (!this.rods[i].removed) this.rods[i].body.wakeUp();
    }
    for (let i = 0; i < this.eggs.length; i++) {
      this.eggs[i].body.wakeUp();
    }
    this.refreshTelemetry();
  }

  /**
   * Applies the active 3D damped spring-tether impulse to the currently grabbed dowel.
   */
  private applyGrabSpringSubstep(dt: number): void {
    if (this.activeGrabRodId === null) return;
    const rec = this.rods[this.activeGrabRodId];
    if (!rec || rec.removed) {
      this.activeGrabRodId = null;
      return;
    }

    const body = rec.body;
    const tr = body.translation();
    const q = body.rotation();
    const lv = body.linvel();
    const av = body.angvel();

    rotateVecByQuatOut(
      this.grabLocalPt[0],
      this.grabLocalPt[1],
      this.grabLocalPt[2],
      q.x,
      q.y,
      q.z,
      q.w,
      this.scratchRot
    );

    const wx = tr.x + this.scratchRot[0];
    const wy = tr.y + this.scratchRot[1];
    const wz = tr.z + this.scratchRot[2];
    this.scratchWorldPt[0] = wx;
    this.scratchWorldPt[1] = wy;
    this.scratchWorldPt[2] = wz;

    // Velocity at the world grab point: v_pt = v_cm + omega x r
    this.scratchArm[0] = this.scratchRot[0];
    this.scratchArm[1] = this.scratchRot[1];
    this.scratchArm[2] = this.scratchRot[2];
    this.scratchPtVel[0] = lv.x + (av.y * this.scratchArm[2] - av.z * this.scratchArm[1]);
    this.scratchPtVel[1] = lv.y + (av.z * this.scratchArm[0] - av.x * this.scratchArm[2]);
    this.scratchPtVel[2] = lv.z + (av.x * this.scratchArm[1] - av.y * this.scratchArm[0]);

    let dx = this.grabTargetWorld[0] - wx;
    let dy = this.grabTargetWorld[1] - wy;
    let dz = this.grabTargetWorld[2] - wz;

    // Smoothly cap tether extension so sudden cursor flicks feel weighted rather than explosive
    const dist = Math.hypot(dx, dy, dz);
    const maxStretch = 0.85;
    if (dist > maxStretch) {
      const s = maxStretch / dist;
      dx *= s;
      dy *= s;
      dz *= s;
    }

    // Gentle natural fingertip lift when pulling a Tier-0 rod radially outward over the 4.6 cm brass lip
    if (rec.spec.tier === 0 && wy < 0.22) {
      const rLen = Math.hypot(tr.x, tr.z) || 1;
      const outwardPull = (dx * tr.x + dz * tr.z) / rLen;
      if (outwardPull > 0.02) {
        dy += outwardPull * 0.24;
      }
    }

    // Mass-normalized PD spring-tether (stiff enough to slide a loaded Jenga dowel cleanly, damped to avoid buzz)
    const mass = Math.max(0.0005, body.mass());
    const stiffness = 240.0 * mass;
    const damping = 22.0 * mass;

    const fx = dx * stiffness - this.scratchPtVel[0] * damping;
    const fy = dy * stiffness - this.scratchPtVel[1] * damping + mass * 9.81 * 0.45;
    const fz = dz * stiffness - this.scratchPtVel[2] * damping;

    // Clamp max force to keep neighboring structural contacts responsive and natural
    const fMag = Math.hypot(fx, fy, fz);
    const maxForce = 185.0 * mass;
    const fScale = fMag > maxForce ? maxForce / fMag : 1.0;

    this.scratchImpulse.x = fx * fScale * dt;
    this.scratchImpulse.y = fy * fScale * dt;
    this.scratchImpulse.z = fz * fScale * dt;
    this.scratchPoint.x = wx;
    this.scratchPoint.y = wy;
    this.scratchPoint.z = wz;

    body.wakeUp();
    body.applyImpulseAtPoint(this.scratchImpulse, this.scratchPoint, true);
  }

  /**
   * Advances physics by exactly one 60 Hz frame (`substepsPer60Hz` fixed substeps).
   */
  stepFixedFrame(): void {
    if (this.isDisposed) return;
    const substeps = Math.max(1, this.options.substepsPer60Hz);
    for (let s = 0; s < substeps; s++) {
      this.applyGrabSpringSubstep(this.fixedDt);
      this.world.step();
      this.simTimeSec += this.fixedDt;
    }
    this.refreshTelemetry();
  }

  /**
   * Advances physics using a fixed-timestep accumulator suitable for variable-refresh displays.
   */
  step(dtSec: number): void {
    if (this.isDisposed) return;
    const clampedDt = Math.min(0.05, Math.max(0, dtSec));
    this.accumulatorSec += clampedDt;
    const maxSubsteps = Math.max(6, this.options.substepsPer60Hz * 3);
    let stepsRun = 0;
    while (this.accumulatorSec >= this.fixedDt && stepsRun < maxSubsteps) {
      this.applyGrabSpringSubstep(this.fixedDt);
      this.world.step();
      this.simTimeSec += this.fixedDt;
      this.accumulatorSec -= this.fixedDt;
      stepsRun++;
    }
    if (stepsRun > 0) {
      this.refreshTelemetry();
    }
  }

  private refreshTelemetry(): void {
    let intact = 0;
    let fallen = 0;
    let maxDrift = 0;
    let maxDrop = 0;

    for (let i = 0; i < this.rods.length; i++) {
      const rec = this.rods[i];
      if (rec.removed) continue;
      const tr = rec.body.translation();
      const dx = tr.x - rec.spec.center[0];
      const dy = tr.y - rec.spec.center[1];
      const dz = tr.z - rec.spec.center[2];
      const drift = Math.hypot(dx, dy, dz);
      const drop = rec.spec.center[1] - tr.y;
      rec.drift = drift;
      rec.drop = drop;

      if (drift > maxDrift) maxDrift = drift;
      if (drop > maxDrop) maxDrop = drop;

      const isFallen =
        drop > FALLEN_DROP_THRESHOLD ||
        drift > FALLEN_DRIFT_THRESHOLD ||
        tr.y < FALLEN_Y_FLOOR;
      rec.fallen = isFallen;
      if (isFallen) fallen++;
      else intact++;
    }

    for (let i = 0; i < this.eggs.length; i++) {
      const egg = this.eggs[i];
      const tr = egg.body.translation();
      egg.drop = egg.spec.pos[1] - tr.y;
    }

    this.intactRodCount = intact;
    this.fallenRodCount = fallen;
    this.collapsed = fallen >= COLLAPSED_ROD_THRESHOLD;
    this.maxRodDrift = maxDrift;
    this.maxRodDrop = maxDrop;
  }

  /**
   * Re-weaves and resets the intact reciprocal dome + ovoid Celadon river-pebble eggs.
   */
  reset(): void {
    if (this.isDisposed) return;
    this.world.free();
    this.buildWorld([]);
  }

  getMaxRodDrop(): number {
    return this.maxRodDrop;
  }

  getEggDrops(): number[] {
    return this.eggs.map((e) => e.drop);
  }

  getEggPositionsY(): number[] {
    return this.eggs.map((e) => e.body.translation().y);
  }

  /**
   * Canonical read-only telemetry matching `window.__DAY24__.getState()`.
   */
  getState(): Day24TelemetryState {
    return {
      intactRodCount: this.intactRodCount,
      fallenRodCount: this.fallenRodCount,
      collapsed: this.collapsed,
      maxRodDrift: Number(this.maxRodDrift.toFixed(4)),
      activeGrabId: this.activeGrabRodId,
      simTime: Number(this.simTimeSec.toFixed(3)),
    };
  }

  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    this.activeGrabRodId = null;
    this.world.free();
  }
}
