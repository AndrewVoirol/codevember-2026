import * as THREE from 'three/webgpu';

/**
 * Day 10 — Caustic: Physical Constants, Meniscus Wave PDE & Venetian Goblet Geometry
 *
 * Zero per-frame heap allocations. All physics arrays and scratch vectors are
 * pre-allocated at module/instance scope.
 */

export const SLAB_TOP_Y = -0.52;
export const SLAB_THICKNESS = 0.14;
export const SLAB_HALF_X = 1.68;
export const SLAB_HALF_Z = 1.38;
export const SLAB_CHAMFER = 0.36;
/** World half-extent mapped into [0..1] UV space of the 1024x1024 caustic RenderTarget */
export const CAUSTIC_MAP_EXTENT = 1.75;

export const GOBLET_POS_X = -0.44;
export const GOBLET_POS_Z = -0.16;

/** Ovoid crystal bowl parameters (centered above GOBLET_POS_X, GOBLET_POS_Z) */
export const BOWL_CENTER_Y = SLAB_TOP_Y + 0.88; // +0.36
export const BOWL_RADIUS_XZ = 0.43;
export const BOWL_RADIUS_Y = 0.45;
export const BOWL_WALL_THICKNESS = 0.018;
export const BOWL_LIP_Y = BOWL_CENTER_Y + 0.34; // +0.70
/** Equilibrium fill level (~half-filled crystal bowl, slightly below equator for strong lens curvature) */
export const LIQUID_FILL_Y = BOWL_CENTER_Y - 0.03; // +0.33

// ---------------------------------------------------------------------------
// 2D Gravity-Capillary Meniscus Wave PDE + Coupled Slosh State (64x64 HalfFloat)
// ---------------------------------------------------------------------------
export const WAVE_RES = 64;
const WAVE_CELLS = WAVE_RES * WAVE_RES;

export class LiquidMeniscusSim {
  readonly heightCurr = new Float32Array(WAVE_CELLS);
  readonly heightPrev = new Float32Array(WAVE_CELLS);
  readonly heightNext = new Float32Array(WAVE_CELLS);
  readonly insideMask = new Float32Array(WAVE_CELLS);

  /** RGBA HalfFloat DataTexture: R = waveHeight, G = dH/dx, B = dH/dz, A = radialMask */
  readonly textureData = new Uint16Array(WAVE_CELLS * 4);
  readonly dataTexture: THREE.DataTexture;

  /** Coupled 2D bulk pendulum slosh angles (dz/dx = sloshX, dy/dz = sloshZ) */
  sloshX = 0.035;
  sloshZ = -0.022;
  sloshVelX = 0.0;
  sloshVelZ = 0.0;

  private ambientTimer = 0;

  constructor() {
    for (let iy = 0; iy < WAVE_RES; iy++) {
      const nz = ((iy + 0.5) / WAVE_RES) * 2.0 - 1.0;
      for (let ix = 0; ix < WAVE_RES; ix++) {
        const nx = ((ix + 0.5) / WAVE_RES) * 2.0 - 1.0;
        const r2 = nx * nx + nz * nz;
        const idx = iy * WAVE_RES + ix;
        this.insideMask[idx] = r2 < 0.96 ? 1.0 : 0.0;
      }
    }

    this.dataTexture = new THREE.DataTexture(
      this.textureData,
      WAVE_RES,
      WAVE_RES,
      THREE.RGBAFormat,
      THREE.HalfFloatType
    );
    this.dataTexture.minFilter = THREE.LinearFilter;
    this.dataTexture.magFilter = THREE.LinearFilter;
    this.dataTexture.wrapS = THREE.ClampToEdgeWrapping;
    this.dataTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.dataTexture.generateMipmaps = false;

    // Seed an initial gentle off-center capillary wavelet so Frame 1 shows rich optical life
    this.injectDropImpulse(0.18, -0.14, 0.012, 0.22);
    this.injectDropImpulse(-0.22, 0.19, -0.008, 0.26);
    this.packTexture();
  }

  /**
   * Injects a smooth Gaussian momentum/displacement dimple in normalized bowl coordinates [-1..1].
   * Strictly zero cos(m*theta) polar harmonics or (0,0)-locked bullseyes.
   */
  injectDropImpulse(normX: number, normZ: number, strength: number, radiusNorm = 0.18): void {
    const invSigma2 = 1.0 / Math.max(0.004, radiusNorm * radiusNorm);
    for (let iy = 0; iy < WAVE_RES; iy++) {
      const gz = ((iy + 0.5) / WAVE_RES) * 2.0 - 1.0;
      const dz = gz - normZ;
      for (let ix = 0; ix < WAVE_RES; ix++) {
        const idx = iy * WAVE_RES + ix;
        if (this.insideMask[idx] < 0.5) continue;
        const gx = ((ix + 0.5) / WAVE_RES) * 2.0 - 1.0;
        const dx = gx - normX;
        const d2 = dx * dx + dz * dz;
        if (d2 > 6.0 * radiusNorm * radiusNorm) continue;
        // Zero-mean capillary dimple + collar so liquid volume is conserved
        const g = Math.exp(-d2 * invSigma2) * (1.0 - 0.45 * d2 * invSigma2);
        this.heightCurr[idx] = Math.max(-0.045, Math.min(0.045, this.heightCurr[idx] + strength * g));
      }
    }
  }

  addSloshImpulse(impX: number, impZ: number): void {
    this.sloshVelX = Math.max(-1.8, Math.min(1.8, this.sloshVelX + impX));
    this.sloshVelZ = Math.max(-1.8, Math.min(1.8, this.sloshVelZ + impZ));
  }

  step(dtSec: number, elapsedSec: number): void {
    const dt = Math.min(0.033, Math.max(0.002, dtSec));

    // Gentle breathing micro-agitation so caustics shimmer organically even when idle
    this.ambientTimer += dt;
    if (this.ambientTimer > 1.35) {
      this.ambientTimer = 0;
      const ang = elapsedSec * 2.399963; // golden-angle non-repeating phase
      const r = 0.22 + 0.34 * (0.5 + 0.5 * Math.sin(elapsedSec * 0.73));
      this.injectDropImpulse(Math.cos(ang) * r, Math.sin(ang) * r, 0.0042, 0.21);
    }

    // Coupled 2D gravity pendulum slosh (natural frequency ~ 4.4 rad/s in a 9cm goblet)
    const omega2 = 19.5;
    const damping = 1.65;
    const microDriveX = Math.sin(elapsedSec * 1.15) * 0.09;
    const microDriveZ = Math.cos(elapsedSec * 0.92 + 0.7) * 0.08;

    this.sloshVelX += (-omega2 * this.sloshX - damping * this.sloshVelX + microDriveX) * dt;
    this.sloshVelZ += (-omega2 * this.sloshZ - damping * this.sloshVelZ + microDriveZ) * dt;
    this.sloshX = Math.max(-0.26, Math.min(0.26, this.sloshX + this.sloshVelX * dt));
    this.sloshZ = Math.max(-0.26, Math.min(0.26, this.sloshZ + this.sloshVelZ * dt));

    // 2 substeps of 2D damped wave equation with Neumann glass-wall reflection
    const substeps = 2;
    const c2 = 0.21; // CFL-safe Courant number squared (< 0.5)
    const atten = Math.exp(-dt * 1.35);

    for (let s = 0; s < substeps; s++) {
      for (let iy = 0; iy < WAVE_RES; iy++) {
        const row = iy * WAVE_RES;
        const yUp = Math.max(0, iy - 1) * WAVE_RES;
        const yDn = Math.min(WAVE_RES - 1, iy + 1) * WAVE_RES;

        for (let ix = 0; ix < WAVE_RES; ix++) {
          const idx = row + ix;
          if (this.insideMask[idx] < 0.5) {
            this.heightNext[idx] = 0.0;
            continue;
          }
          const hC = this.heightCurr[idx];
          const xL = Math.max(0, ix - 1);
          const xR = Math.min(WAVE_RES - 1, ix + 1);

          // Neumann dH/dn = 0 reflection at circular crystal bowl wall
          const hL = this.insideMask[row + xL] > 0.5 ? this.heightCurr[row + xL] : hC;
          const hR = this.insideMask[row + xR] > 0.5 ? this.heightCurr[row + xR] : hC;
          const hU = this.insideMask[yUp + ix] > 0.5 ? this.heightCurr[yUp + ix] : hC;
          const hD = this.insideMask[yDn + ix] > 0.5 ? this.heightCurr[yDn + ix] : hC;

          const lap = hL + hR + hU + hD - 4.0 * hC;
          const vel = (hC - this.heightPrev[idx]) * atten;
          // Weak restoring gravity-capillary dispersion term (-0.012 * hC) prevents DC drift
          const next = hC + vel + c2 * lap - 0.008 * hC;
          this.heightNext[idx] = Math.max(-0.05, Math.min(0.05, next));
        }
      }

      this.heightPrev.set(this.heightCurr);
      this.heightCurr.set(this.heightNext);
    }

    this.packTexture();
  }

  private packTexture(): void {
    const worldCellSize = (2.0 * BOWL_RADIUS_XZ) / WAVE_RES;
    const invTwoDx = 1.0 / (2.0 * worldCellSize);
    const toHalf = THREE.DataUtils.toHalfFloat;

    for (let iy = 0; iy < WAVE_RES; iy++) {
      const row = iy * WAVE_RES;
      const yUp = Math.max(0, iy - 1) * WAVE_RES;
      const yDn = Math.min(WAVE_RES - 1, iy + 1) * WAVE_RES;

      for (let ix = 0; ix < WAVE_RES; ix++) {
        const idx = row + ix;
        const mask = this.insideMask[idx];
        const hC = this.heightCurr[idx] * mask;

        const xL = Math.max(0, ix - 1);
        const xR = Math.min(WAVE_RES - 1, ix + 1);
        const hL = this.insideMask[row + xL] > 0.5 ? this.heightCurr[row + xL] : hC;
        const hR = this.insideMask[row + xR] > 0.5 ? this.heightCurr[row + xR] : hC;
        const hU = this.insideMask[yUp + ix] > 0.5 ? this.heightCurr[yUp + ix] : hC;
        const hD = this.insideMask[yDn + ix] > 0.5 ? this.heightCurr[yDn + ix] : hC;

        const dHdx = (hR - hL) * invTwoDx * mask;
        const dHdz = (hD - hU) * invTwoDx * mask;

        const base = idx * 4;
        this.textureData[base] = toHalf(hC);
        this.textureData[base + 1] = toHalf(dHdx);
        this.textureData[base + 2] = toHalf(dHdz);
        this.textureData[base + 3] = toHalf(mask);
      }
    }
    this.dataTexture.needsUpdate = true;
  }
}

/**
 * Evaluates horizontal inner bowl radius at local height `dy = y - BOWL_CENTER_Y`.
 */
export function innerBowlRadiusAtDy(dy: number): number {
  const innerRx = BOWL_RADIUS_XZ - BOWL_WALL_THICKNESS;
  const innerRy = BOWL_RADIUS_Y - BOWL_WALL_THICKNESS;
  const u = Math.max(-0.998, Math.min(0.998, dy / innerRy));
  return innerRx * Math.sqrt(Math.max(0.0, 1.0 - u * u));
}

// ---------------------------------------------------------------------------
// Sculptural Geometry Builders (Slab, Blown Venetian Bowl, Stem, Dynamic Liquid)
// ---------------------------------------------------------------------------

/**
 * Builds a beveled octagonal Bone Calcite stone plinth slab centered at (0, SLAB_TOP_Y - SLAB_THICKNESS/2, 0)
 * with top face strictly at y = SLAB_TOP_Y.
 */
export function createBeveledCalciteSlabGeometry(): THREE.BufferGeometry {
  const hx = SLAB_HALF_X;
  const hz = SLAB_HALF_Z;
  const ch = SLAB_CHAMFER;
  const bevel = 0.026;

  // 8-sided chamfered perimeter in XZ (CCW when viewed from +Y)
  const basePts: Array<[number, number]> = [
    [hx - ch, -hz],
    [hx, -hz + ch],
    [hx, hz - ch],
    [hx - ch, hz],
    [-hx + ch, hz],
    [-hx, hz - ch],
    [-hx, -hz + ch],
    [-hx + ch, -hz],
  ];

  const positions: number[] = [];
  const normals: number[] = [];

  const pushTri = (
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    cx: number,
    cy: number,
    cz: number
  ) => {
    const ux = bx - ax;
    const uy = by - ay;
    const uz = bz - az;
    const vx = cx - ax;
    const vy = cy - ay;
    const vz = cz - az;
    // CCW around +Y: cross(B - A, C - A) points outward (+Y on top cap)
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    positions.push(ax, ay, az, bx, by, bz, cx, cy, cz);
    normals.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
  };

  const pushQuad = (
    p0: [number, number, number],
    p1: [number, number, number],
    p2: [number, number, number],
    p3: [number, number, number]
  ) => {
    pushTri(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    pushTri(p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]);
  };

  const yTop = SLAB_TOP_Y;
  const yBevel = SLAB_TOP_Y - bevel;
  const yBotBevel = SLAB_TOP_Y - SLAB_THICKNESS + bevel * 0.65;
  const yBot = SLAB_TOP_Y - SLAB_THICKNESS;

  const shrinkPt = (pt: [number, number], inset: number): [number, number] => {
    const sx = Math.sign(pt[0]);
    const sz = Math.sign(pt[1]);
    return [pt[0] - sx * inset, pt[1] - sz * inset];
  };

  const topInner = basePts.map((p) => shrinkPt(p, bevel));
  const botInner = basePts.map((p) => shrinkPt(p, bevel * 0.65));

  // Subdivided top face fan (clean radial segments so vertex interpolation is crisp)
  for (let i = 0; i < 8; i++) {
    const next = (i + 1) % 8;
    // Note: basePts goes clockwise or CCW? Let's check:
    // p0 = (1.32, -1.38), p1 = (1.68, -1.02):
    // cross(p0, p1) in (x, 0, z): ux=1.32, uz=-1.38; vx=1.68, vz=-1.02
    // ny = uz*vx - ux*vz = (-1.38)*(1.68) - (1.32)*(-1.02) = -2.318 + 1.346 = -0.972 (< 0!)
    // So reversing order (0, next, i) gives ny = +1.0 (+Y upward normal)!
    pushTri(
      0,
      yTop,
      0,
      topInner[next][0],
      yTop,
      topInner[next][1],
      topInner[i][0],
      yTop,
      topInner[i][1]
    );

    // Upper 45-degree chamfer bevel
    pushQuad(
      [topInner[i][0], yTop, topInner[i][1]],
      [topInner[next][0], yTop, topInner[next][1]],
      [basePts[next][0], yBevel, basePts[next][1]],
      [basePts[i][0], yBevel, basePts[i][1]]
    );

    // Vertical mineral edge band
    pushQuad(
      [basePts[i][0], yBevel, basePts[i][1]],
      [basePts[next][0], yBevel, basePts[next][1]],
      [basePts[next][0], yBotBevel, basePts[next][1]],
      [basePts[i][0], yBotBevel, basePts[i][1]]
    );

    // Lower shadow chamfer
    pushQuad(
      [basePts[i][0], yBotBevel, basePts[i][1]],
      [basePts[next][0], yBotBevel, basePts[next][1]],
      [botInner[next][0], yBot, botInner[next][1]],
      [botInner[i][0], yBot, botInner[i][1]]
    );

    // Bottom cap (-Y normal)
    pushTri(
      0,
      yBot,
      0,
      botInner[i][0],
      yBot,
      botInner[i][1],
      botInner[next][0],
      yBot,
      botInner[next][1]
    );
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geom;
}

/**
 * Builds the thin-walled hand-blown Venetian crystal bowl + flared foot profile
 * (local origin at (GOBLET_POS_X, SLAB_TOP_Y, GOBLET_POS_Z)).
 */
export function createVenetianCrystalBowlAndFoot(): {
  bowlGeom: THREE.LatheGeometry;
  footGeom: THREE.LatheGeometry;
  stemGeom: THREE.LatheGeometry;
  lipRingGeom: THREE.TorusGeometry;
} {
  const bowlCenterLocalY = BOWL_CENTER_Y - SLAB_TOP_Y; // 0.88
  const lipLocalY = BOWL_LIP_Y - SLAB_TOP_Y; // 1.22

  // 1. Thin-walled double-sided crystal bowl profile (outer curve bottom->lip, rounded lip, inner curve lip->bottom)
  const bowlPts: THREE.Vector2[] = [];
  const outerSteps = 44;
  const dyMin = -BOWL_RADIUS_Y;
  const dyMax = lipLocalY - bowlCenterLocalY; // +0.34

  for (let i = 0; i <= outerSteps; i++) {
    const t = i / outerSteps;
    const dy = dyMin + (dyMax - dyMin) * t;
    const u = Math.max(-1.0, Math.min(1.0, dy / BOWL_RADIUS_Y));
    const r = i === 0 ? 0.0 : BOWL_RADIUS_XZ * Math.sqrt(Math.max(0.0, 1.0 - u * u));
    bowlPts.push(new THREE.Vector2(r, bowlCenterLocalY + dy));
  }

  // Smooth fire-polished rounded lip turnaround (outer -> inner)
  const rOuterLip = bowlPts[bowlPts.length - 1].x;
  const innerRx = BOWL_RADIUS_XZ - BOWL_WALL_THICKNESS;
  const innerRy = BOWL_RADIUS_Y - BOWL_WALL_THICKNESS;
  const uInnerLip = Math.max(-0.999, Math.min(0.999, dyMax / innerRy));
  const rInnerLip = innerRx * Math.sqrt(Math.max(0.0, 1.0 - uInnerLip * uInnerLip));
  const rMidLip = 0.5 * (rOuterLip + rInnerLip);
  const halfWall = 0.5 * (rOuterLip - rInnerLip);

  for (let k = 1; k <= 5; k++) {
    const ang = (k / 6) * Math.PI;
    bowlPts.push(
      new THREE.Vector2(
        rMidLip + Math.cos(ang) * halfWall,
        bowlCenterLocalY + dyMax + Math.sin(ang) * halfWall * 0.9
      )
    );
  }

  // Inner surface descending from lip down to inner bowl base
  for (let i = outerSteps; i >= 0; i--) {
    const t = i / outerSteps;
    const dy = -innerRy + (dyMax + innerRy) * t;
    const u = Math.max(-1.0, Math.min(1.0, dy / innerRy));
    const r = i === 0 ? 0.0 : innerRx * Math.sqrt(Math.max(0.0, 1.0 - u * u));
    bowlPts.push(new THREE.Vector2(r, bowlCenterLocalY + dy));
  }

  const bowlGeom = new THREE.LatheGeometry(bowlPts, 88);

  // 2. Venetian Baluster & Hollow-Knop Stem (y = 0.048 .. 0.445)
  const stemPts: THREE.Vector2[] = [];
  const stemSteps = 56;
  const stemY0 = 0.045;
  const stemY1 = bowlCenterLocalY - BOWL_RADIUS_Y + 0.015; // meets bowl base cleanly (~0.445)
  for (let i = 0; i <= stemSteps; i++) {
    const t = i / stemSteps;
    const y = stemY0 + (stemY1 - stemY0) * t;
    // Classic Murano profile: lower merese collar (t=0.18), central blown knop (t=0.52), upper merese collar (t=0.84)
    const baseNeck = 0.036;
    const knop = 0.058 * Math.exp(-Math.pow((t - 0.52) / 0.17, 2));
    const collarLow = 0.024 * Math.exp(-Math.pow((t - 0.17) / 0.045, 2));
    const collarHigh = 0.026 * Math.exp(-Math.pow((t - 0.85) / 0.045, 2));
    const flareTop = 0.048 * Math.pow(Math.max(0, (t - 0.86) / 0.14), 2);
    const flareBot = 0.032 * Math.pow(Math.max(0, (0.14 - t) / 0.14), 2);
    const r = baseNeck + knop + collarLow + collarHigh + flareTop + flareBot;
    stemPts.push(new THREE.Vector2(r, y));
  }
  const stemGeom = new THREE.LatheGeometry(stemPts, 64);

  // 3. Flared Venetian Foot with rolled glass rim (y = 0.002 .. 0.052)
  const footPts: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.008),
    new THREE.Vector2(0.18, 0.006),
    new THREE.Vector2(0.32, 0.002),
    new THREE.Vector2(0.355, 0.003),
    new THREE.Vector2(0.365, 0.011),
    new THREE.Vector2(0.352, 0.020),
    new THREE.Vector2(0.28, 0.024),
    new THREE.Vector2(0.15, 0.035),
    new THREE.Vector2(0.068, 0.050),
    new THREE.Vector2(0.0, 0.052),
  ];
  const footGeom = new THREE.LatheGeometry(footPts, 72);

  // 4. Delicate Realgar-Amber fire-polished lip thread torus
  const lipRingGeom = new THREE.TorusGeometry(rMidLip, halfWall * 0.42, 14, 88);
  lipRingGeom.rotateX(Math.PI * 0.5);
  lipRingGeom.translate(0, bowlCenterLocalY + dyMax + halfWall * 0.62, 0);

  return { bowlGeom, footGeom, stemGeom, lipRingGeom };
}

/**
 * Dynamic 3D Liquid Volume Mesh (Top Rippling Meniscus + Wetted Side/Bottom Wall)
 * Updated in place every frame with zero heap allocations.
 */
export class DynamicLiquidMesh {
  readonly mesh: THREE.Mesh;
  readonly geometry: THREE.BufferGeometry;

  private readonly radialSegs = 64;
  private readonly ringSegs = 22;
  private readonly sideSegs = 20;

  private readonly positions: Float32Array;
  private readonly normals: Float32Array;
  /** x: optical chord proxy [0..1], y: meniscus rim factor [0..1], z: wave height */
  private readonly opticalAttr: Float32Array;

  private readonly topVertOffset = 0;
  private readonly sideVertOffset: number;

  constructor(material: THREE.Material) {
    // Top cap: center vertex + ringSegs * (radialSegs + 1)
    const topVerts = 1 + this.ringSegs * (this.radialSegs + 1);
    // Wetted bowl shell: (sideSegs + 1) * (radialSegs + 1)
    const sideVerts = (this.sideSegs + 1) * (this.radialSegs + 1);
    this.sideVertOffset = topVerts;

    const totalVerts = topVerts + sideVerts;
    this.positions = new Float32Array(totalVerts * 3);
    this.normals = new Float32Array(totalVerts * 3);
    this.opticalAttr = new Float32Array(totalVerts * 3);

    const indices: number[] = [];
    const stride = this.radialSegs + 1;

    // Top meniscus indices (+Y upward normal: center, nextTheta, currTheta)
    for (let ir = 0; ir < this.radialSegs; ir++) {
      indices.push(0, 1 + ir + 1, 1 + ir);
    }
    for (let ring = 0; ring < this.ringSegs - 1; ring++) {
      const r0 = 1 + ring * stride;
      const r1 = 1 + (ring + 1) * stride;
      for (let ir = 0; ir < this.radialSegs; ir++) {
        const a = r0 + ir;
        const b = r0 + ir + 1;
        const c = r1 + ir;
        const d = r1 + ir + 1;
        indices.push(a, d, c, a, b, d);
      }
    }

    // Wetted bowl outer-facing liquid shell (from bottom pole ring=0 up to meniscus contact ring=sideSegs)
    for (let iv = 0; iv < this.sideSegs; iv++) {
      const r0 = this.sideVertOffset + iv * stride;
      const r1 = this.sideVertOffset + (iv + 1) * stride;
      for (let ir = 0; ir < this.radialSegs; ir++) {
        const a = r0 + ir;
        const b = r0 + ir + 1;
        const c = r1 + ir;
        const d = r1 + ir + 1;
        indices.push(a, b, d, a, d, c);
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3));
    this.geometry.setAttribute('aLiquidOptics', new THREE.BufferAttribute(this.opticalAttr, 3));
    this.geometry.setIndex(indices);

    this.mesh = new THREE.Mesh(this.geometry, material);
    // Render liquid before outer crystal goblet wall so transparent goblet Fresnel wraps cleanly over it
    this.mesh.renderOrder = 1;
  }

  updateFromSim(sim: LiquidMeniscusSim): void {
    const bowlCenterLocalY = BOWL_CENTER_Y - SLAB_TOP_Y;
    const baseFillLocalY = LIQUID_FILL_Y - SLAB_TOP_Y;
    const innerRx = (BOWL_RADIUS_XZ - BOWL_WALL_THICKNESS) * 0.992;
    const innerRy = (BOWL_RADIUS_Y - BOWL_WALL_THICKNESS) * 0.992;
    const bottomLocalY = bowlCenterLocalY - innerRy;

    const sampleWave = (nx: number, nz: number): number => {
      const fx = Math.max(0, Math.min(WAVE_RES - 1, (nx * 0.5 + 0.5) * (WAVE_RES - 1)));
      const fz = Math.max(0, Math.min(WAVE_RES - 1, (nz * 0.5 + 0.5) * (WAVE_RES - 1)));
      const x0 = Math.floor(fx);
      const z0 = Math.floor(fz);
      const x1 = Math.min(WAVE_RES - 1, x0 + 1);
      const z1 = Math.min(WAVE_RES - 1, z0 + 1);
      const tx = fx - x0;
      const tz = fz - z0;
      const h00 = sim.heightCurr[z0 * WAVE_RES + x0];
      const h10 = sim.heightCurr[z0 * WAVE_RES + x1];
      const h01 = sim.heightCurr[z1 * WAVE_RES + x0];
      const h11 = sim.heightCurr[z1 * WAVE_RES + x1];
      return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
    };

    // 1. Center vertex of meniscus
    const hCenterWave = sampleWave(0, 0);
    const yCenter = baseFillLocalY + hCenterWave;
    this.positions[0] = 0;
    this.positions[1] = yCenter;
    this.positions[2] = 0;
    this.normals[0] = -sim.sloshX;
    this.normals[1] = 1.0;
    this.normals[2] = -sim.sloshZ;
    this.opticalAttr[0] = 0.95;
    this.opticalAttr[1] = 0.0;
    this.opticalAttr[2] = hCenterWave;

    const stride = this.radialSegs + 1;

    // Precompute contact-line height & radius around azimuth ir = 0..radialSegs
    for (let ir = 0; ir <= this.radialSegs; ir++) {
      const theta = (ir / this.radialSegs) * Math.PI * 2.0;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);

      // Iterative 2-step contact intersection with curved inner bowl wall + capillary wetting climb
      let rWall = innerBowlRadiusAtDy(LIQUID_FILL_Y - BOWL_CENTER_Y) * 0.992;
      let yContact =
        baseFillLocalY +
        (sim.sloshX * cosT + sim.sloshZ * sinT) * rWall +
        0.018 +
        sampleWave(cosT * 0.92, sinT * 0.92);
      yContact = Math.max(bottomLocalY + 0.12, Math.min(BOWL_LIP_Y - SLAB_TOP_Y - 0.06, yContact));
      rWall = innerBowlRadiusAtDy(yContact - bowlCenterLocalY) * 0.992;

      // Populate top meniscus rings along this azimuth ray
      for (let ring = 0; ring < this.ringSegs; ring++) {
        const frac = (ring + 1) / this.ringSegs;
        const r = rWall * frac;
        const x = cosT * r;
        const z = sinT * r;
        const wH = sampleWave(cosT * frac * 0.94, sinT * frac * 0.94);
        const meniscusClimb = 0.018 * frac * frac * frac * frac;
        const y = baseFillLocalY + sim.sloshX * x + sim.sloshZ * z + meniscusClimb + wH;

        const vIdx = (this.topVertOffset + 1 + ring * stride + ir) * 3;
        this.positions[vIdx] = x;
        this.positions[vIdx + 1] = y;
        this.positions[vIdx + 2] = z;

        // Finite-difference wave slope + capillary rim climb slope
        const eps = 0.04;
        const dWaveDx = (sampleWave(cosT * frac * 0.94 + eps, sinT * frac * 0.94) - wH) / (eps * innerRx);
        const dWaveDz = (sampleWave(cosT * frac * 0.94, sinT * frac * 0.94 + eps) - wH) / (eps * innerRx);
        const climbSlope = 0.072 * frac * frac * frac;
        let nx = -(sim.sloshX + dWaveDx + cosT * climbSlope);
        let ny = 1.0;
        let nz = -(sim.sloshZ + dWaveDz + sinT * climbSlope);
        const nLen = Math.hypot(nx, ny, nz) || 1;
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;

        this.normals[vIdx] = nx;
        this.normals[vIdx + 1] = ny;
        this.normals[vIdx + 2] = nz;

        // Depth chord below this meniscus point to bowl floor
        const uRad = Math.min(0.999, r / innerRx);
        const yBowlFloor = bowlCenterLocalY - innerRy * Math.sqrt(Math.max(0, 1 - uRad * uRad));
        const depthChord = Math.max(0.02, y - yBowlFloor);
        this.opticalAttr[vIdx] = Math.min(1.4, depthChord * 2.1);
        this.opticalAttr[vIdx + 1] = frac * frac;
        this.opticalAttr[vIdx + 2] = wH;
      }

      // Populate wetted side/bottom liquid shell from bottomLocalY up to yContact
      for (let iv = 0; iv <= this.sideSegs; iv++) {
        const vFrac = iv / this.sideSegs;
        const y = bottomLocalY + (yContact - bottomLocalY) * vFrac;
        const dy = y - bowlCenterLocalY;
        const u = Math.max(-1.0, Math.min(0.999, dy / innerRy));
        const r = iv === 0 ? 0.0 : innerRx * Math.sqrt(Math.max(0.0, 1.0 - u * u));
        const x = cosT * r;
        const z = sinT * r;

        const sIdx = (this.sideVertOffset + iv * stride + ir) * 3;
        this.positions[sIdx] = x;
        this.positions[sIdx + 1] = y;
        this.positions[sIdx + 2] = z;

        // Outward ellipsoid normal
        let nx = iv === 0 ? 0 : x / (innerRx * innerRx);
        let ny = dy / (innerRy * innerRy);
        let nz = iv === 0 ? 0 : z / (innerRx * innerRx);
        const nLen = Math.hypot(nx, ny, nz) || 1;
        this.normals[sIdx] = nx / nLen;
        this.normals[sIdx + 1] = ny / nLen;
        this.normals[sIdx + 2] = nz / nLen;

        // Chord thickness across bowl at height y (~ 2 * r, tapering near top meniscus)
        const chord = Math.min(1.35, 2.0 * r * 1.65 * (1.0 - 0.35 * Math.pow(vFrac, 4)));
        this.opticalAttr[sIdx] = chord;
        this.opticalAttr[sIdx + 1] = Math.pow(vFrac, 3);
        this.opticalAttr[sIdx + 2] = 0.0;
      }
    }

    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.normal as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.aLiquidOptics as THREE.BufferAttribute).needsUpdate = true;
  }
}
