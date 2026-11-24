import * as THREE from 'three/webgpu';
import { MINERAL_PALETTE } from '../../core/palette';
import { type ErosionField } from './erosionField';
import { buildCleavedChipGeometry } from './noduleGeometry';

export const MAX_GRIT_CHIPS = 420;
export const MAX_DUST_PUFFS = 180;

interface GritParticle {
  active: boolean;
  px: number;
  py: number;
  pz: number;
  vx: number;
  vy: number;
  vz: number;
  rx: number;
  ry: number;
  rz: number;
  vrx: number;
  vry: number;
  vrz: number;
  size: number;
  age: number;
  life: number;
  r: number;
  g: number;
  b: number;
  metallic: number;
  bounces: number;
}

interface DustPuff {
  active: boolean;
  px: number;
  py: number;
  pz: number;
  vx: number;
  vy: number;
  vz: number;
  startSize: number;
  endSize: number;
  age: number;
  life: number;
  seed: number;
  density: number;
  warmth: number;
}

// Pre-allocated scratch math objects (strictly zero per-frame allocations)
const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _euler = new THREE.Euler();
const _scale = new THREE.Vector3();
const _reflectDir = new THREE.Vector3();
const _bounceNormal = new THREE.Vector3();
const _t1 = new THREE.Vector3();
const _t2 = new THREE.Vector3();

// Pre-cached linear RGB values from canonical mineral palette
const PAL_CALCITE = MINERAL_PALETTE.boneCalcite.rInfLinear;
const PAL_KAOLIN = MINERAL_PALETTE.warmKaolin.rInfLinear;
const PAL_OCHRE = MINERAL_PALETTE.yellowOchre.rInfLinear;
const PAL_SIENNA = MINERAL_PALETTE.burntSienna.rInfLinear;
const PAL_AMBER = MINERAL_PALETTE.realgarAmber.rInfLinear;

/**
 * Deterministic fast PRNG (Mulberry32-style LCG) so particle emission never allocates
 * and tests remain reproducible.
 */
class FastRng {
  private state: number;

  constructor(seed = 19019) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 4294967296;
  }

  signed(): number {
    return this.next() * 2.0 - 1.0;
  }
}

/**
 * Zero-GC Instanced Particle Engine driving:
 *  1. High-velocity rebounding dolomite abrasive grit & cleaved limestone rock chips
 *     (with metallic Realgar-Amber / Yellow-Ochre ricochets off exposed pyritized ribs
 *      and secondary bounces against the live carved nodule surface).
 *  2. Soft billowing pale Bone-Calcite & Warm-Kaolin chalk dust plumes erupting along
 *     the exact raycast impact normal.
 */
export class AbrasionParticleSystem {
  readonly group: THREE.Group;
  readonly gritMesh: THREE.InstancedMesh;
  readonly dustMesh: THREE.InstancedMesh;

  readonly gritMetaArr: Float32Array;
  readonly gritMetaAttr: THREE.InstancedBufferAttribute;
  readonly dustMetaArr: Float32Array;
  readonly dustMetaAttr: THREE.InstancedBufferAttribute;

  private readonly gritPool: GritParticle[] = [];
  private readonly dustPool: DustPuff[] = [];
  private gritCursor = 0;
  private dustCursor = 0;
  private activeGritCount = 0;
  private activeDustCount = 0;
  private emitChipCarry = 0;
  private emitDustCarry = 0;
  private readonly rng = new FastRng(202619);
  private readonly geometries: THREE.BufferGeometry[] = [];

  constructor(gritMaterial: THREE.Material, dustMaterial: THREE.Material) {
    this.group = new THREE.Group();
    this.group.name = 'AirAbrasionParticlesGroup';

    // 1. Cleaved Mineral Grit & Limestone Chip Pool
    const chipGeo = buildCleavedChipGeometry();
    // aChipMeta: [r, g, b, metallicSpark]
    this.gritMetaArr = new Float32Array(MAX_GRIT_CHIPS * 4);
    this.gritMetaAttr = new THREE.InstancedBufferAttribute(this.gritMetaArr, 4);
    chipGeo.setAttribute('aChipMeta', this.gritMetaAttr);
    this.geometries.push(chipGeo);

    this.gritMesh = new THREE.InstancedMesh(chipGeo, gritMaterial, MAX_GRIT_CHIPS);
    this.gritMesh.name = 'ReboundingGritChips';
    this.gritMesh.count = 0;
    this.gritMesh.frustumCulled = false;
    this.gritMesh.castShadow = false;
    this.gritMesh.receiveShadow = false;
    this.group.add(this.gritMesh);

    // 2. Swirling Calcite Dust Plume Billboards
    const dustGeo = new THREE.PlaneGeometry(2.0, 2.0);
    // aDustMeta: [alphaEnvelope, seed, warmth, ageNorm]
    this.dustMetaArr = new Float32Array(MAX_DUST_PUFFS * 4);
    this.dustMetaAttr = new THREE.InstancedBufferAttribute(this.dustMetaArr, 4);
    dustGeo.setAttribute('aDustMeta', this.dustMetaAttr);
    this.geometries.push(dustGeo);

    this.dustMesh = new THREE.InstancedMesh(dustGeo, dustMaterial, MAX_DUST_PUFFS);
    this.dustMesh.name = 'CalciteDustPlumes';
    this.dustMesh.count = 0;
    this.dustMesh.frustumCulled = false;
    this.dustMesh.castShadow = false;
    this.dustMesh.receiveShadow = false;
    this.dustMesh.renderOrder = 4;
    this.group.add(this.dustMesh);

    for (let i = 0; i < MAX_GRIT_CHIPS; i++) {
      this.gritPool.push({
        active: false,
        px: 0,
        py: 0,
        pz: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        rx: 0,
        ry: 0,
        rz: 0,
        vrx: 0,
        vry: 0,
        vrz: 0,
        size: 0.01,
        age: 0,
        life: 0.5,
        r: PAL_CALCITE[0],
        g: PAL_CALCITE[1],
        b: PAL_CALCITE[2],
        metallic: 0,
        bounces: 0,
      });
    }

    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      this.dustPool.push({
        active: false,
        px: 0,
        py: 0,
        pz: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        startSize: 0.04,
        endSize: 0.22,
        age: 0,
        life: 0.9,
        seed: 0,
        density: 0.5,
        warmth: 0.2,
      });
    }
  }

  get activeParticles(): number {
    return this.activeGritCount + this.activeDustCount;
  }

  /**
   * Emits bouncing dolomite grit grains, spalled limestone chips, and billowing calcite dust
   * from the exact raycast impact point along the reflected jet & surface normal.
   */
  emitBlast(
    hitPos: THREE.Vector3,
    hitNormal: THREE.Vector3,
    jetDir: THREE.Vector3,
    fossilRatio: number,
    dtSec: number,
    burstScale = 1.0
  ): void {
    // Build orthonormal tangent frame around impact normal
    if (Math.abs(hitNormal.y) < 0.9) {
      _t1.set(hitNormal.z, 0, -hitNormal.x).normalize();
    } else {
      _t1.set(0, -hitNormal.z, hitNormal.y).normalize();
    }
    _t2.crossVectors(hitNormal, _t1).normalize();

    // Specularly reflected abrasive rebound vector: R = J - 2*(J.N)*N
    const jDotN = jetDir.dot(hitNormal);
    _reflectDir
      .copy(jetDir)
      .addScaledVector(hitNormal, -2.0 * Math.min(0, jDotN))
      .normalize();

    // Blend between specular rebound and outward normal blast cone
    const chipRate = (260 + 140 * fossilRatio) * burstScale;
    this.emitChipCarry += chipRate * Math.max(0.004, dtSec);
    const chipCount = Math.min(24, Math.floor(this.emitChipCarry));
    this.emitChipCarry -= chipCount;

    for (let k = 0; k < chipCount; k++) {
      const slot = this.gritPool[this.gritCursor];
      this.gritCursor = (this.gritCursor + 1) % MAX_GRIT_CHIPS;

      const isSparkRicochet = this.rng.next() < 0.03 + 0.72 * fossilRatio;
      const spreadR = 0.012 + this.rng.next() * 0.028;
      const spreadAng = this.rng.next() * Math.PI * 2;
      const ca = Math.cos(spreadAng);
      const sa = Math.sin(spreadAng);

      slot.active = true;
      slot.px = hitPos.x + hitNormal.x * 0.008 + (_t1.x * ca + _t2.x * sa) * spreadR;
      slot.py = hitPos.y + hitNormal.y * 0.008 + (_t1.y * ca + _t2.y * sa) * spreadR;
      slot.pz = hitPos.z + hitNormal.z * 0.008 + (_t1.z * ca + _t2.z * sa) * spreadR;

      const speed = isSparkRicochet
        ? 1.85 + this.rng.next() * 1.65
        : 0.90 + this.rng.next() * 1.15;
      const specBias = isSparkRicochet ? 0.68 : 0.36;
      const coneSpread = isSparkRicochet ? 0.50 : 0.86;

      const dirX =
        _reflectDir.x * specBias +
        hitNormal.x * (1.0 - specBias) +
        (_t1.x * this.rng.signed() + _t2.x * this.rng.signed()) * coneSpread;
      const dirY =
        _reflectDir.y * specBias +
        hitNormal.y * (1.0 - specBias) +
        (_t1.y * this.rng.signed() + _t2.y * this.rng.signed()) * coneSpread;
      const dirZ =
        _reflectDir.z * specBias +
        hitNormal.z * (1.0 - specBias) +
        (_t1.z * this.rng.signed() + _t2.z * this.rng.signed()) * coneSpread;
      const invD = 1.0 / Math.max(1e-5, Math.hypot(dirX, dirY, dirZ));

      slot.vx = dirX * invD * speed;
      slot.vy = dirY * invD * speed;
      slot.vz = dirZ * invD * speed;

      slot.rx = this.rng.next() * Math.PI * 2;
      slot.ry = this.rng.next() * Math.PI * 2;
      slot.rz = this.rng.next() * Math.PI * 2;
      slot.vrx = this.rng.signed() * 18.0;
      slot.vry = this.rng.signed() * 18.0;
      slot.vrz = this.rng.signed() * 18.0;

      slot.size = isSparkRicochet
        ? 0.0042 + this.rng.next() * 0.0042
        : 0.0075 + this.rng.next() * 0.0105;
      slot.age = 0;
      slot.life = isSparkRicochet
        ? 0.26 + this.rng.next() * 0.26
        : 0.42 + this.rng.next() * 0.46;
      slot.bounces = 0;

      if (isSparkRicochet) {
        const pick = this.rng.next();
        if (pick < 0.55) {
          // Burnished golden pyrite FeS2 micro-spark
          slot.r = PAL_OCHRE[0] * 1.18;
          slot.g = PAL_OCHRE[1] * 1.12;
          slot.b = PAL_OCHRE[2] * 0.96;
        } else if (pick < 0.85) {
          // Pale hot calcite-pyrite flash
          slot.r = PAL_CALCITE[0] * 1.08;
          slot.g = PAL_CALCITE[1] * 1.02;
          slot.b = PAL_OCHRE[2] * 1.15;
        } else {
          // Warm realgar-amber mineral spark
          slot.r = PAL_AMBER[0] * 1.08;
          slot.g = PAL_AMBER[1] * 1.04;
          slot.b = PAL_AMBER[2] * 0.94;
        }
        slot.metallic = 0.92;
      } else {
        const pick = this.rng.next();
        if (pick < 0.52) {
          slot.r = PAL_CALCITE[0];
          slot.g = PAL_CALCITE[1];
          slot.b = PAL_CALCITE[2];
        } else if (pick < 0.84) {
          slot.r = PAL_KAOLIN[0];
          slot.g = PAL_KAOLIN[1];
          slot.b = PAL_KAOLIN[2];
        } else if (pick < 0.96) {
          slot.r = PAL_OCHRE[0] * 0.88;
          slot.g = PAL_OCHRE[1] * 0.88;
          slot.b = PAL_OCHRE[2] * 0.88;
        } else {
          slot.r = PAL_SIENNA[0] * 0.82;
          slot.g = PAL_SIENNA[1] * 0.82;
          slot.b = PAL_SIENNA[2] * 0.82;
        }
        slot.metallic = 0.05;
      }
    }

    // Billowing pale Bone-Calcite chalk dust puffs (denser on soft limestone, lighter mist on clean pyrite)
    const dustRate = (82 * (1.0 - 0.45 * fossilRatio) + 22) * burstScale;
    this.emitDustCarry += dustRate * Math.max(0.004, dtSec);
    const dustCount = Math.min(8, Math.floor(this.emitDustCarry));
    this.emitDustCarry -= dustCount;

    for (let k = 0; k < dustCount; k++) {
      const puff = this.dustPool[this.dustCursor];
      this.dustCursor = (this.dustCursor + 1) % MAX_DUST_PUFFS;

      const spreadR = 0.012 + this.rng.next() * 0.038;
      const ang = this.rng.next() * Math.PI * 2;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);

      puff.active = true;
      puff.px = hitPos.x + hitNormal.x * 0.018 + (_t1.x * ca + _t2.x * sa) * spreadR;
      puff.py = hitPos.y + hitNormal.y * 0.018 + (_t1.y * ca + _t2.y * sa) * spreadR;
      puff.pz = hitPos.z + hitNormal.z * 0.018 + (_t1.z * ca + _t2.z * sa) * spreadR;

      const puffSpeed = 0.46 + this.rng.next() * 0.62;
      const vx =
        hitNormal.x * 0.72 +
        _reflectDir.x * 0.28 +
        (_t1.x * this.rng.signed() + _t2.x * this.rng.signed()) * 0.65;
      const vy =
        hitNormal.y * 0.72 +
        _reflectDir.y * 0.28 +
        (_t1.y * this.rng.signed() + _t2.y * this.rng.signed()) * 0.65;
      const vz =
        hitNormal.z * 0.72 +
        _reflectDir.z * 0.28 +
        (_t1.z * this.rng.signed() + _t2.z * this.rng.signed()) * 0.65;
      const inv = 1.0 / Math.max(1e-5, Math.hypot(vx, vy, vz));

      puff.vx = vx * inv * puffSpeed;
      puff.vy = vy * inv * puffSpeed;
      puff.vz = vz * inv * puffSpeed;

      puff.startSize = 0.036 + this.rng.next() * 0.025;
      puff.endSize = 0.19 + this.rng.next() * 0.14;
      puff.age = 0;
      puff.life = 0.65 + this.rng.next() * 0.56;
      puff.seed = this.rng.next();
      puff.density = (0.38 + this.rng.next() * 0.24) * (1.0 - 0.30 * fossilRatio);
      puff.warmth = 0.10 + this.rng.next() * 0.26 + fossilRatio * 0.24;
    }
  }

  /**
   * Emits a brief soft settling mineral powder halo around the nodule on double-click / Space re-encrust.
   */
  emitResetVeil(): void {
    for (let k = 0; k < 28; k++) {
      const puff = this.dustPool[this.dustCursor];
      this.dustCursor = (this.dustCursor + 1) % MAX_DUST_PUFFS;

      const u = this.rng.next() * Math.PI * 2;
      const v = this.rng.signed() * 0.82;
      const rad = Math.sqrt(Math.max(0, 1 - v * v));
      const nx = Math.cos(u) * rad;
      const ny = v;
      const nz = Math.sin(u) * rad;

      puff.active = true;
      puff.px = nx * 1.08;
      puff.py = ny * 0.98;
      puff.pz = nz * 0.54;
      puff.vx = nx * 0.28;
      puff.vy = ny * 0.24 + 0.14;
      puff.vz = nz * 0.28;
      puff.startSize = 0.08;
      puff.endSize = 0.30;
      puff.age = 0;
      puff.life = 0.72 + this.rng.next() * 0.36;
      puff.seed = this.rng.next();
      puff.density = 0.26;
      puff.warmth = 0.18;
    }
  }

  /**
   * Steps all active bouncing grit chips and swirling calcite dust plumes and packs GPU instance matrices.
   * Strictly zero per-frame object allocations.
   */
  step(
    dtSec: number,
    cameraQuat: THREE.Quaternion,
    erosionField: ErosionField | null = null
  ): void {
    const dt = Math.min(0.05, Math.max(0.001, dtSec));
    const chipDrag = Math.exp(-dt * 1.45);
    const dustDrag = Math.exp(-dt * 2.65);

    // 1. Update rebounding grit chips
    let writeGrit = 0;
    for (let i = 0; i < MAX_GRIT_CHIPS; i++) {
      const p = this.gritPool[i];
      if (!p.active) continue;

      p.age += dt;
      if (p.age >= p.life) {
        p.active = false;
        continue;
      }

      // Gravity + pneumatic drag
      p.vy -= 3.85 * dt;
      p.vx *= chipDrag;
      p.vy *= chipDrag;
      p.vz *= chipDrag;

      p.px += p.vx * dt;
      p.py += p.vy * dt;
      p.pz += p.vz * dt;

      // Secondary physical rebound using exact oblate/carved heightfield surface normal
      if (erosionField && p.bounces < 2) {
        const dist = Math.hypot(p.px, p.py, p.pz);
        if (dist > 0.12 && dist < 1.32) {
          const inv = 1.0 / dist;
          const dx = p.px * inv;
          const dy = p.py * inv;
          const dz = p.pz * inv;
          const surfR = erosionField.sampleCombinedRadius(dx, dy, dz);
          if (dist < surfR + p.size * 0.8) {
            erosionField.sampleCombinedNormal(dx, dy, dz, _bounceNormal);
            const nx = _bounceNormal.x;
            const ny = _bounceNormal.y;
            const nz = _bounceNormal.z;
            const vDotN = p.vx * nx + p.vy * ny + p.vz * nz;
            if (vDotN < 0) {
              const restitution = p.metallic > 0.5 ? 0.58 : 0.36;
              p.vx -= (1 + restitution) * vDotN * nx;
              p.vy -= (1 + restitution) * vDotN * ny;
              p.vz -= (1 + restitution) * vDotN * nz;
              const pushR = surfR + p.size * 0.95;
              p.px = dx * pushR;
              p.py = dy * pushR;
              p.pz = dz * pushR;
              p.bounces++;
            }
          }
        }
      }

      p.rx += p.vrx * dt;
      p.ry += p.vry * dt;
      p.rz += p.vrz * dt;

      const u = p.age / p.life;
      const shrink = 1.0 - u * u;
      const curSize = Math.max(0.0005, p.size * shrink);

      _pos.set(p.px, p.py, p.pz);
      _euler.set(p.rx, p.ry, p.rz);
      _quat.setFromEuler(_euler);
      _scale.setScalar(curSize);
      _mat4.compose(_pos, _quat, _scale);
      this.gritMesh.setMatrixAt(writeGrit, _mat4);

      const m4 = writeGrit * 4;
      this.gritMetaArr[m4] = p.r;
      this.gritMetaArr[m4 + 1] = p.g;
      this.gritMetaArr[m4 + 2] = p.b;
      this.gritMetaArr[m4 + 3] = p.metallic;
      writeGrit++;
    }

    this.activeGritCount = writeGrit;
    this.gritMesh.count = writeGrit;
    if (writeGrit > 0) {
      this.gritMesh.instanceMatrix.needsUpdate = true;
      this.gritMetaAttr.needsUpdate = true;
    }

    // 2. Update billowing calcite dust puffs
    let writeDust = 0;
    for (let i = 0; i < MAX_DUST_PUFFS; i++) {
      const d = this.dustPool[i];
      if (!d.active) continue;

      d.age += dt;
      if (d.age >= d.life) {
        d.active = false;
        continue;
      }

      // Gentle buoyant swirl + air drag
      const swirlPhase = d.age * 4.2 + d.seed * 18.0;
      d.vx = d.vx * dustDrag + Math.sin(swirlPhase) * 0.24 * dt;
      d.vy = d.vy * dustDrag + (0.28 + Math.cos(swirlPhase * 0.83) * 0.16) * dt;
      d.vz = d.vz * dustDrag + Math.cos(swirlPhase * 1.17) * 0.22 * dt;

      d.px += d.vx * dt;
      d.py += d.vy * dt;
      d.pz += d.vz * dt;

      const u = d.age / d.life;
      const easeOut = 1.0 - (1.0 - u) * (1.0 - u);
      const curSize = d.startSize + (d.endSize - d.startSize) * easeOut;

      // Smooth fade-in then long airy dissipation
      const fadeIn = Math.min(1.0, u / 0.14);
      const fadeOut = Math.pow(1.0 - u, 1.55);
      const alphaEnv = d.density * fadeIn * fadeOut;

      _pos.set(d.px, d.py, d.pz);
      _scale.setScalar(curSize);
      _mat4.compose(_pos, cameraQuat, _scale);
      this.dustMesh.setMatrixAt(writeDust, _mat4);

      const m4 = writeDust * 4;
      this.dustMetaArr[m4] = alphaEnv;
      this.dustMetaArr[m4 + 1] = d.seed;
      this.dustMetaArr[m4 + 2] = d.warmth;
      this.dustMetaArr[m4 + 3] = u;
      writeDust++;
    }

    this.activeDustCount = writeDust;
    this.dustMesh.count = writeDust;
    if (writeDust > 0) {
      this.dustMesh.instanceMatrix.needsUpdate = true;
      this.dustMetaAttr.needsUpdate = true;
    }
  }

  clear(): void {
    for (let i = 0; i < MAX_GRIT_CHIPS; i++) this.gritPool[i].active = false;
    for (let i = 0; i < MAX_DUST_PUFFS; i++) this.dustPool[i].active = false;
    this.activeGritCount = 0;
    this.activeDustCount = 0;
    this.gritMesh.count = 0;
    this.dustMesh.count = 0;
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
  }
}
