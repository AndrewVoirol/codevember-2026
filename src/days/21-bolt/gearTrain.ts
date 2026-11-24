import * as THREE from 'three/webgpu';
import {
  ARBOR_B2_X,
  ARBOR_B2_Y,
  ARBOR_B3_X,
  ARBOR_B3_Y,
  ARBOR_B4_X,
  ARBOR_B4_Y,
  ARBOR_U1_X,
  ARBOR_U1_Y,
  ARBOR_U2_X,
  ARBOR_U2_Y,
  GEAR_B1_CURTAIN,
  GEAR_B2_IDLER,
  GEAR_B3F_STEP,
  GEAR_B3R_PINION,
  GEAR_B4A_KEYHOME,
  GEAR_U1A_PINION,
  GEAR_U1B_SPUR,
  GEAR_U2A_DRIVEN,
  KEY_X,
  KEY_Y,
  LOWER_KEYHOME_CAM_SPEC,
  UPPER_DETENT_CAM_SPEC,
  buildInvoluteGearPolygon2D,
  evalDetentCamRadius,
  type DetentCamSpec,
  type SpurGearSpec,
} from './lockMechanism';
import {
  applyMachinedNormals,
  buildSlottedBrassScrew,
  ensureCCW,
  ensureCW,
  makeCircleHole,
  makeClosedLatheZ,
  offsetClosedPolygonCCW,
} from './geometry';
import type { BoltMaterials } from './materials';

const GEAR_BEVEL = 0.0011;
const DETENT_SPRING_SEGMENTS = 22;
const DETENT_SPRING_HALF_Z = 0.0105;
const DETENT_SPRING_HALF_N = 0.0068;
const PAWL_SPRING_PAD_LOCAL_X = 0.086;
const PAWL_SPRING_PAD_LOCAL_Y = -0.020;

interface GearBuilderOptions {
  boreRadius: number;
  spokeCount?: number;
  drillHoles?: number;
  keyholeCurtainSlot?: boolean;
}

/**
 * Builds an outward-wound, machined-normal 3D involute spur gear strictly from `SpurGearSpec`.
 * Outer contour is inset by `GEAR_BEVEL` before extrusion so the beveled outer metal skin lands on
 * exact analytical pitch/involute tooth coordinates (zero tooth clipping across 360 degrees!).
 */
export function createInvoluteGearGeometry(
  spec: SpurGearSpec,
  opts: GearBuilderOptions
): THREE.BufferGeometry {
  const rawPts = buildInvoluteGearPolygon2D(spec, 4).map((p) => new THREE.Vector2(p.x, p.y));
  const insetPts = offsetClosedPolygonCCW(ensureCCW(rawPts), -GEAR_BEVEL);
  const shape = new THREE.Shape(ensureCCW(insetPts));

  if (opts.keyholeCurtainSlot) {
    // Unified slotted keyhole-curtain bore: central key-pipe journal + radial -Y keyway slot
    // Rotates 1:1 with the key to seal the lock interior against pick wires as soon as key leaves 6 o'clock!
    const barrelR = opts.boreRadius + GEAR_BEVEL;
    const slotHalfW = 0.023 + GEAR_BEVEL;
    const slotBottomY = -0.095 - GEAR_BEVEL;
    const slotAng = Math.asin(Math.min(0.88, slotHalfW / barrelR));
    const khPts: THREE.Vector2[] = [];
    const arcSegs = 28;
    for (let s = 0; s <= arcSegs; s++) {
      const a = -Math.PI * 0.5 + slotAng + (Math.PI * 2 - 2 * slotAng) * (s / arcSegs);
      khPts.push(new THREE.Vector2(Math.cos(a) * barrelR, Math.sin(a) * barrelR));
    }
    khPts.push(new THREE.Vector2(-slotHalfW, slotBottomY));
    khPts.push(new THREE.Vector2(slotHalfW, slotBottomY));
    shape.holes.push(new THREE.Path(ensureCW(khPts)));

    // Two upper crescent inspection windows (solid lower shutter seals keyhole completely when rotated!)
    const rIn = opts.boreRadius + 0.020 + GEAR_BEVEL;
    const rOut = spec.rootRadius - 0.014 - GEAR_BEVEL;
    if (rOut - rIn > 0.014) {
      for (const [a0Deg, a1Deg] of [
        [24, 76],
        [104, 156],
      ] as Array<[number, number]>) {
        const a0 = (a0Deg * Math.PI) / 180;
        const a1 = (a1Deg * Math.PI) / 180;
        const winPts: THREE.Vector2[] = [];
        const arcN = 10;
        for (let s = 0; s <= arcN; s++) {
          const a = a0 + (a1 - a0) * (s / arcN);
          winPts.push(new THREE.Vector2(Math.cos(a) * rOut, Math.sin(a) * rOut));
        }
        for (let s = arcN; s >= 0; s--) {
          const a = a0 + (a1 - a0) * (s / arcN);
          winPts.push(new THREE.Vector2(Math.cos(a) * rIn, Math.sin(a) * rIn));
        }
        shape.holes.push(new THREE.Path(ensureCW(winPts)));
      }
    }
  } else {
    shape.holes.push(makeCircleHole(0, 0, opts.boreRadius + GEAR_BEVEL, 26));

    if (opts.spokeCount && opts.spokeCount >= 3) {
      const nSpokes = opts.spokeCount;
      const rIn = opts.boreRadius + 0.018 + GEAR_BEVEL;
      const rOut = spec.rootRadius - 0.015 - GEAR_BEVEL;
      if (rOut - rIn > 0.015) {
        const step = (Math.PI * 2) / nSpokes;
        const spokeHalfW = 0.0085 + GEAR_BEVEL;
        for (let k = 0; k < nSpokes; k++) {
          const baseA = k * step;
          const aIn0 = baseA + Math.asin(Math.min(0.45, spokeHalfW / rIn));
          const aIn1 = baseA + step - Math.asin(Math.min(0.45, spokeHalfW / rIn));
          const aOut0 = baseA + Math.asin(Math.min(0.45, spokeHalfW / rOut));
          const aOut1 = baseA + step - Math.asin(Math.min(0.45, spokeHalfW / rOut));
          if (aIn1 <= aIn0 + 0.08 || aOut1 <= aOut0 + 0.08) continue;

          const winPts: THREE.Vector2[] = [];
          const arcN = 8;
          for (let s = 0; s <= arcN; s++) {
            const a = aOut0 + (aOut1 - aOut0) * (s / arcN);
            winPts.push(new THREE.Vector2(Math.cos(a) * rOut, Math.sin(a) * rOut));
          }
          for (let s = arcN; s >= 0; s--) {
            const a = aIn0 + (aIn1 - aIn0) * (s / arcN);
            winPts.push(new THREE.Vector2(Math.cos(a) * rIn, Math.sin(a) * rIn));
          }
          shape.holes.push(new THREE.Path(ensureCW(winPts)));
        }
      }
    } else if (opts.drillHoles && opts.drillHoles >= 3) {
      const rRing = (opts.boreRadius + spec.rootRadius) * 0.5;
      const holeR = Math.min(0.012, (spec.rootRadius - opts.boreRadius) * 0.24);
      for (let k = 0; k < opts.drillHoles; k++) {
        const a = (k / opts.drillHoles) * Math.PI * 2;
        shape.holes.push(makeCircleHole(Math.cos(a) * rRing, Math.sin(a) * rRing, holeR + GEAR_BEVEL, 18));
      }
    }
  }

  const coreDepth = Math.max(0.004, spec.zFront - spec.zBack - GEAR_BEVEL * 2);
  const rawGeo = new THREE.ExtrudeGeometry(shape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: GEAR_BEVEL,
    bevelSize: GEAR_BEVEL,
    bevelSegments: 2,
    curveSegments: 20,
  });
  rawGeo.translate(0, 0, -coreDepth * 0.5);
  return applyMachinedNormals(rawGeo);
}

/**
 * Builds an outward-wound precision blued-steel detent escapement cam wheel strictly matching
 * `evalDetentCamRadius(spec, angle)` with 4 horological lightening cutouts.
 */
export function createDetentCamGeometry(spec: DetentCamSpec, boreRadius: number): THREE.BufferGeometry {
  const camBevel = 0.0014;
  const segs = 144;
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i < segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    const r = evalDetentCamRadius(spec, a);
    pts.push(new THREE.Vector2(Math.cos(a) * r, Math.sin(a) * r));
  }
  const inset = offsetClosedPolygonCCW(ensureCCW(pts), -camBevel);
  const shape = new THREE.Shape(ensureCCW(inset));

  shape.holes.push(makeCircleHole(0, 0, boreRadius + camBevel, 26));

  // Four precision drilled horological cutouts cleanly inside the deepest V-notch root radius
  const rRing = (boreRadius + spec.rootRadius) * 0.50;
  const holeR = Math.min(0.011, (spec.rootRadius - boreRadius) * 0.24);
  for (let k = 0; k < 4; k++) {
    const a = ((k + 0.5) / 4) * Math.PI * 2;
    shape.holes.push(makeCircleHole(Math.cos(a) * rRing, Math.sin(a) * rRing, holeR + camBevel, 18));
  }

  const coreDepth = Math.max(0.006, spec.zFront - spec.zBack - camBevel * 2);
  const rawGeo = new THREE.ExtrudeGeometry(shape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: camBevel,
    bevelSize: camBevel,
    bevelSegments: 2,
    curveSegments: 24,
  });
  rawGeo.translate(0, 0, -coreDepth * 0.5);
  return applyMachinedNormals(rawGeo);
}

/**
 * Builds an outward-wound curved hardened-steel detent escapement pawl in pawl-local XY:
 *  - Local (0, 0): pivot post center
 *  - Local (+pawlArmLength, 0): exact center of the rounded contact beak (radius = spec.beakRadius)
 *  - Local (-Y): upper-right back spine where the phosphor-bronze detent leaf spring presses flush
 *  - Local (+Y): lower-left concave throat clearing the cam rim
 */
export function createDetentPawlGeometry(spec: DetentCamSpec): THREE.BufferGeometry {
  const pawlBevel = 0.0014;
  const L = spec.pawlArmLength;
  const rBeak = spec.beakRadius;
  const rPivot = 0.028;

  const nominal: THREE.Vector2[] = [];

  // Semicircular pivot ear from +PI/2 around PI to -PI/2
  const earSegs = 12;
  for (let s = 0; s <= earSegs; s++) {
    const a = Math.PI * 0.5 + Math.PI * (s / earSegs);
    nominal.push(new THREE.Vector2(Math.cos(a) * rPivot, Math.sin(a) * rPivot));
  }

  // Gently tapered upper-right spine (-Y side) with flat spring-contact landing shoulder
  nominal.push(new THREE.Vector2(0.042, -0.024));
  nominal.push(new THREE.Vector2(PAWL_SPRING_PAD_LOCAL_X - 0.016, PAWL_SPRING_PAD_LOCAL_Y));
  nominal.push(new THREE.Vector2(PAWL_SPRING_PAD_LOCAL_X + 0.016, PAWL_SPRING_PAD_LOCAL_Y));
  nominal.push(new THREE.Vector2(L - 0.012, -0.013));

  // Rounded steel escapement beak centered strictly at (L, 0) with exact radius `rBeak`
  const beakSegs = 12;
  for (let s = 0; s <= beakSegs; s++) {
    const a = -Math.PI * 0.45 + (Math.PI * 1.15 * s) / beakSegs;
    nominal.push(new THREE.Vector2(L + Math.cos(a) * rBeak, Math.sin(a) * rBeak));
  }

  // Concave lower-left inner throat giving crisp watchmaker beak clearance past V-notch shoulders
  nominal.push(new THREE.Vector2(L - 0.018, -0.004));
  nominal.push(new THREE.Vector2(0.068, -0.007));
  nominal.push(new THREE.Vector2(0.036, 0.008));

  const inset = offsetClosedPolygonCCW(ensureCCW(nominal), -pawlBevel);
  const shape = new THREE.Shape(ensureCCW(inset));
  shape.holes.push(makeCircleHole(0, 0, 0.0165 + pawlBevel, 22));

  const coreDepth = Math.max(0.008, (spec.zFront - spec.zBack) * 0.88 - pawlBevel * 2);
  const rawGeo = new THREE.ExtrudeGeometry(shape, {
    depth: coreDepth,
    bevelEnabled: true,
    bevelThickness: pawlBevel,
    bevelSize: pawlBevel,
    bevelSegments: 2,
    curveSegments: 20,
  });
  rawGeo.translate(0, 0, -coreDepth * 0.5);
  return applyMachinedNormals(rawGeo);
}

/**
 * Zero-allocation dynamically flexing curved phosphor-bronze detent leaf spring.
 * Anchors cleanly off its brass spring pin and arrives strictly tangent & flush onto the pawl back pad!
 */
export class DetentSpringMesh {
  readonly spec: DetentCamSpec;
  readonly mesh: THREE.Mesh;
  readonly geometry: THREE.BufferGeometry;

  private readonly positions: Float32Array;
  private readonly normals: Float32Array;
  private readonly cx = new Float32Array(DETENT_SPRING_SEGMENTS + 1);
  private readonly cy = new Float32Array(DETENT_SPRING_SEGMENTS + 1);
  private readonly nx = new Float32Array(DETENT_SPRING_SEGMENTS + 1);
  private readonly ny = new Float32Array(DETENT_SPRING_SEGMENTS + 1);
  private lastPawlAngle = -999;

  constructor(name: string, spec: DetentCamSpec, material: THREE.Material) {
    this.spec = spec;
    this.geometry = new THREE.BufferGeometry();

    const totalTris = DETENT_SPRING_SEGMENTS * 8 + 4;
    this.positions = new Float32Array(totalTris * 9);
    this.normals = new Float32Array(totalTris * 9);
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3));

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.name = name;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
  }

  update(pawlWorldAngleRad: number): void {
    if (Math.abs(pawlWorldAngleRad - this.lastPawlAngle) < 1e-5) return;
    this.lastPawlAngle = pawlWorldAngleRad;

    const zMid = (this.spec.zBack + this.spec.zFront) * 0.5;
    const zBack = zMid - DETENT_SPRING_HALF_Z;
    const zFront = zMid + DETENT_SPRING_HALF_Z;

    const cosP = Math.cos(pawlWorldAngleRad);
    const sinP = Math.sin(pawlWorldAngleRad);

    // Contact pad on pawl's -Y back shoulder + outward normal pointing up-right (+sinP, -cosP)
    const padX =
      this.spec.pivotX + PAWL_SPRING_PAD_LOCAL_X * cosP - PAWL_SPRING_PAD_LOCAL_Y * sinP;
    const padY =
      this.spec.pivotY + PAWL_SPRING_PAD_LOCAL_X * sinP + PAWL_SPRING_PAD_LOCAL_Y * cosP;
    const padNx = sinP;
    const padNy = -cosP;
    const padTx = cosP;
    const padTy = sinP;

    // Anchor cleanly on upper-left shoulder of bronze spring pin collar for a single monotonic watchmaker arc
    const p0x = this.spec.springPinX - 0.010;
    const p0y = this.spec.springPinY + 0.017;

    // Tip centerline offset by half thickness so bottom face kisses pawl back shoulder flush!
    const hEnd = DETENT_SPRING_HALF_N * 0.86;
    const p3x = padX + padNx * hEnd;
    const p3y = padY + padNy * hEnd;

    // Single smooth convex watchmaker C-arch with zero S-kink, pressing cleanly downward onto pawl spine
    const p1x = p0x - 0.038;
    const p1y = p0y + 0.003;
    const p2x = p3x + padNx * 0.022 - padTx * 0.014;
    const p2y = p3y + padNy * 0.022 - padTy * 0.014;

    for (let s = 0; s <= DETENT_SPRING_SEGMENTS; s++) {
      const t = s / DETENT_SPRING_SEGMENTS;
      const it = 1 - t;
      const b0 = it * it * it;
      const b1 = 3 * it * it * t;
      const b2 = 3 * it * t * t;
      const b3 = t * t * t;

      this.cx[s] = b0 * p0x + b1 * p1x + b2 * p2x + b3 * p3x;
      this.cy[s] = b0 * p0y + b1 * p1y + b2 * p2y + b3 * p3y;

      const tx = 3 * it * it * (p1x - p0x) + 6 * it * t * (p2x - p1x) + 3 * t * t * (p3x - p2x);
      const ty = 3 * it * it * (p1y - p0y) + 6 * it * t * (p2y - p1y) + 3 * t * t * (p3y - p2y);
      const len = Math.max(1e-5, Math.hypot(tx, ty));
      this.nx[s] = -ty / len;
      this.ny[s] = tx / len;
    }

    let ptr = 0;
    const writeQuad = (
      ax: number, ay: number, az: number, nax: number, nay: number, naz: number,
      bx: number, by: number, bz: number, nbx: number, nby: number, nbz: number,
      cx: number, cy: number, cz: number, ncx: number, ncy: number, ncz: number,
      dx: number, dy: number, dz: number, ndx: number, ndy: number, ndz: number
    ) => {
      this.positions[ptr] = ax; this.positions[ptr + 1] = ay; this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = bx; this.positions[ptr + 4] = by; this.positions[ptr + 5] = bz;
      this.positions[ptr + 6] = cx; this.positions[ptr + 7] = cy; this.positions[ptr + 8] = cz;
      this.normals[ptr] = nax; this.normals[ptr + 1] = nay; this.normals[ptr + 2] = naz;
      this.normals[ptr + 3] = nbx; this.normals[ptr + 4] = nby; this.normals[ptr + 5] = nbz;
      this.normals[ptr + 6] = ncx; this.normals[ptr + 7] = ncy; this.normals[ptr + 8] = ncz;
      ptr += 9;

      this.positions[ptr] = ax; this.positions[ptr + 1] = ay; this.positions[ptr + 2] = az;
      this.positions[ptr + 3] = cx; this.positions[ptr + 4] = cy; this.positions[ptr + 5] = cz;
      this.positions[ptr + 6] = dx; this.positions[ptr + 7] = dy; this.positions[ptr + 8] = dz;
      this.normals[ptr] = nax; this.normals[ptr + 1] = nay; this.normals[ptr + 2] = naz;
      this.normals[ptr + 3] = ncx; this.normals[ptr + 4] = ncy; this.normals[ptr + 5] = ncz;
      this.normals[ptr + 6] = ndx; this.normals[ptr + 7] = ndy; this.normals[ptr + 8] = ndz;
      ptr += 9;
    };

    for (let s = 0; s < DETENT_SPRING_SEGMENTS; s++) {
      const h0 = DETENT_SPRING_HALF_N * (1.12 - 0.26 * (s / DETENT_SPRING_SEGMENTS));
      const h1 = DETENT_SPRING_HALF_N * (1.12 - 0.26 * ((s + 1) / DETENT_SPRING_SEGMENTS));

      const top0x = this.cx[s] + this.nx[s] * h0;
      const top0y = this.cy[s] + this.ny[s] * h0;
      const bot0x = this.cx[s] - this.nx[s] * h0;
      const bot0y = this.cy[s] - this.ny[s] * h0;

      const top1x = this.cx[s + 1] + this.nx[s + 1] * h1;
      const top1y = this.cy[s + 1] + this.ny[s + 1] * h1;
      const bot1x = this.cx[s + 1] - this.nx[s + 1] * h1;
      const bot1y = this.cy[s + 1] - this.ny[s + 1] * h1;

      writeQuad(
        bot0x, bot0y, zFront, 0, 0, 1,
        bot1x, bot1y, zFront, 0, 0, 1,
        top1x, top1y, zFront, 0, 0, 1,
        top0x, top0y, zFront, 0, 0, 1
      );
      writeQuad(
        bot1x, bot1y, zBack, 0, 0, -1,
        bot0x, bot0y, zBack, 0, 0, -1,
        top0x, top0y, zBack, 0, 0, -1,
        top1x, top1y, zBack, 0, 0, -1
      );
      writeQuad(
        top0x, top0y, zFront, this.nx[s], this.ny[s], 0,
        top1x, top1y, zFront, this.nx[s + 1], this.ny[s + 1], 0,
        top1x, top1y, zBack, this.nx[s + 1], this.ny[s + 1], 0,
        top0x, top0y, zBack, this.nx[s], this.ny[s], 0
      );
      writeQuad(
        bot0x, bot0y, zBack, -this.nx[s], -this.ny[s], 0,
        bot1x, bot1y, zBack, -this.nx[s + 1], -this.ny[s + 1], 0,
        bot1x, bot1y, zFront, -this.nx[s + 1], -this.ny[s + 1], 0,
        bot0x, bot0y, zFront, -this.nx[s], -this.ny[s], 0
      );
    }

    const hStart = DETENT_SPRING_HALF_N * 1.12;
    writeQuad(
      this.cx[0] - this.nx[0] * hStart, this.cy[0] - this.ny[0] * hStart, zBack, -this.ny[0], this.nx[0], 0,
      this.cx[0] - this.nx[0] * hStart, this.cy[0] - this.ny[0] * hStart, zFront, -this.ny[0], this.nx[0], 0,
      this.cx[0] + this.nx[0] * hStart, this.cy[0] + this.ny[0] * hStart, zFront, -this.ny[0], this.nx[0], 0,
      this.cx[0] + this.nx[0] * hStart, this.cy[0] + this.ny[0] * hStart, zBack, -this.ny[0], this.nx[0], 0
    );

    const e = DETENT_SPRING_SEGMENTS;
    writeQuad(
      this.cx[e] - this.nx[e] * hEnd, this.cy[e] - this.ny[e] * hEnd, zFront, this.ny[e], -this.nx[e], 0,
      this.cx[e] - this.nx[e] * hEnd, this.cy[e] - this.ny[e] * hEnd, zBack, this.ny[e], -this.nx[e], 0,
      this.cx[e] + this.nx[e] * hEnd, this.cy[e] + this.ny[e] * hEnd, zBack, this.ny[e], -this.nx[e], 0,
      this.cx[e] + this.nx[e] * hEnd, this.cy[e] + this.ny[e] * hEnd, zFront, this.ny[e], -this.nx[e], 0
    );

    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.normal as THREE.BufferAttribute).needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

export interface GearTrainAssembly {
  group: THREE.Group;
  upperArborU1Group: THREE.Group;
  upperArborU2Group: THREE.Group;
  upperPawlGroup: THREE.Group;
  upperDetentSpring: DetentSpringMesh;
  curtainB1Group: THREE.Group;
  curtainIdlerB2Group: THREE.Group;
  curtainCompoundB3Group: THREE.Group;
  keyHomeB4Group: THREE.Group;
  lowerPawlGroup: THREE.Group;
  lowerDetentSpring: DetentSpringMesh;
  update: (
    u1Rad: number,
    u2Rad: number,
    upperPawlRad: number,
    b1Rad: number,
    b2Rad: number,
    b3Rad: number,
    b4Rad: number,
    lowerPawlRad: number
  ) => void;
  dispose: () => void;
}

/**
 * Helper to create a turned brass oil-sink bearing collar centered around (x, y) spanning z0..z1.
 */
function addBearingCollar(
  parent: THREE.Object3D,
  name: string,
  x: number,
  y: number,
  z0: number,
  z1: number,
  rInner: number,
  rOuter: number,
  mat: THREE.Material,
  disposables: THREE.BufferGeometry[]
): void {
  const chamfer = Math.min(0.003, (z1 - z0) * 0.25, (rOuter - rInner) * 0.25);
  const geo = makeClosedLatheZ(
    [
      [z0, rInner],
      [z0, rOuter],
      [z1 - chamfer, rOuter],
      [z1, rOuter - chamfer],
      [z1, rInner],
    ],
    26,
    false
  );
  geo.translate(x, y, 0);
  disposables.push(geo);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
}

/**
 * Constructs the complete, structurally anchored, zero-decorative Horological Geared Deadbolt Drive Train:
 *  - Train A (Upper-Right): DeadboltUpperRack (in DeadboltAssembly) -> Arbor U1 (12T steel rack pinion + 16T spoked brass spur)
 *    -> Arbor U2 (18T spoked brass wheel + foreground blued-steel dual-notch detent cam + eccentric throw stud)
 *    -> Spring-Loaded Hardened-Steel Upper Detent Pawl + flexing phosphor-bronze detent blade spring.
 *  - Train B (Key Hub -> Lower-Right): Slotted 24T brass Keyhole Anti-Pick Curtain Wheel (B1)
 *    -> 20T brass Curtain Transfer Idler (B2) on bolted satin-brass Front Horological Bridge
 *    -> Full-depth Compound Arbor B3 (18T front step gear + 15T rear phosphor-bronze drilled pinion)
 *    -> Full-depth Arbor B4 (20T grand 5-spoked brass wheel + foreground blued-steel Key-Home V-notch cam)
 *    -> Spring-Loaded Lower Key-Home Detent Pawl + flexing phosphor-bronze detent blade spring.
 */
export function buildGearTrainAssembly(materials: BoltMaterials): GearTrainAssembly {
  const group = new THREE.Group();
  group.name = 'HorologicalGearTrainAssembly';
  const disposables: THREE.BufferGeometry[] = [];

  // ===========================================================================
  // PART 1: STATIONARY BACKPLATE ARBORS, OIL-SINK BEARINGS & FRONT BRIDGE
  // ===========================================================================
  const staticHardware = new THREE.Group();
  staticHardware.name = 'GearTrainStaticArborsAndBridge';
  group.add(staticHardware);

  // 1A. Arbor U1 Steel Post (anchored in backplate -0.182 .. -0.055) + Rear Oil-Sink Bushing
  const arborU1Geo = makeClosedLatheZ(
    [
      [-0.182, 0.023],
      [-0.064, 0.023],
      [-0.060, 0.032],
      [-0.054, 0.026],
    ],
    26
  );
  arborU1Geo.translate(ARBOR_U1_X, ARBOR_U1_Y, 0);
  disposables.push(arborU1Geo);
  const arborU1Mesh = new THREE.Mesh(arborU1Geo, materials.hardenedSteel);
  arborU1Mesh.name = 'ArborPost_U1';
  arborU1Mesh.castShadow = true;
  arborU1Mesh.receiveShadow = true;
  staticHardware.add(arborU1Mesh);
  addBearingCollar(
    staticHardware,
    'ArborRearBushing_U1',
    ARBOR_U1_X,
    ARBOR_U1_Y,
    -0.180,
    -0.158,
    0.024,
    0.044,
    materials.forendBrass,
    disposables
  );

  // 1B. Arbor U2 Full-Depth Steel Post (anchored in backplate -0.182 .. +0.112) + Oil-Sink Bushings
  const arborU2Geo = makeClosedLatheZ(
    [
      [-0.182, 0.025],
      [0.098, 0.025],
      [0.102, 0.035],
      [0.112, 0.028],
      [0.116, 0.018],
    ],
    28
  );
  arborU2Geo.translate(ARBOR_U2_X, ARBOR_U2_Y, 0);
  disposables.push(arborU2Geo);
  const arborU2Mesh = new THREE.Mesh(arborU2Geo, materials.hardenedSteel);
  arborU2Mesh.name = 'ArborPost_U2';
  arborU2Mesh.castShadow = true;
  arborU2Mesh.receiveShadow = true;
  staticHardware.add(arborU2Mesh);
  addBearingCollar(
    staticHardware,
    'ArborRearBushing_U2',
    ARBOR_U2_X,
    ARBOR_U2_Y,
    -0.180,
    -0.108,
    0.026,
    0.046,
    materials.forendBrass,
    disposables
  );

  // 1C. Upper Detent Pawl Pivot Post & Spring Pin (anchored solidly into backplate -0.182)
  const uPawlPostGeo = makeClosedLatheZ(
    [
      [-0.182, 0.0155],
      [0.092, 0.0155],
      [0.096, 0.023],
      [0.105, 0.018],
    ],
    22
  );
  uPawlPostGeo.translate(UPPER_DETENT_CAM_SPEC.pivotX, UPPER_DETENT_CAM_SPEC.pivotY, 0);
  disposables.push(uPawlPostGeo);
  const uPawlPostMesh = new THREE.Mesh(uPawlPostGeo, materials.hardenedSteel);
  uPawlPostMesh.name = 'UpperPawlPivotPost';
  uPawlPostMesh.castShadow = true;
  staticHardware.add(uPawlPostMesh);
  addBearingCollar(
    staticHardware,
    'UpperPawlStandoff',
    UPPER_DETENT_CAM_SPEC.pivotX,
    UPPER_DETENT_CAM_SPEC.pivotY,
    -0.180,
    UPPER_DETENT_CAM_SPEC.zBack,
    0.016,
    0.028,
    materials.forendBrass,
    disposables
  );

  const uSpringPinGeo = makeClosedLatheZ(
    [
      [-0.182, 0.0145],
      [0.082, 0.0145],
      [0.087, 0.021],
      [0.094, 0.016],
    ],
    20
  );
  uSpringPinGeo.translate(UPPER_DETENT_CAM_SPEC.springPinX, UPPER_DETENT_CAM_SPEC.springPinY, 0);
  disposables.push(uSpringPinGeo);
  const uSpringPinMesh = new THREE.Mesh(uSpringPinGeo, materials.forendBrass);
  uSpringPinMesh.name = 'UpperDetentSpringPin';
  uSpringPinMesh.castShadow = true;
  staticHardware.add(uSpringPinMesh);
  addBearingCollar(
    staticHardware,
    'UpperDetentSpringCollar',
    UPPER_DETENT_CAM_SPEC.springPinX,
    UPPER_DETENT_CAM_SPEC.springPinY,
    (UPPER_DETENT_CAM_SPEC.zBack + UPPER_DETENT_CAM_SPEC.zFront) * 0.5 - DETENT_SPRING_HALF_Z,
    (UPPER_DETENT_CAM_SPEC.zBack + UPPER_DETENT_CAM_SPEC.zFront) * 0.5 + DETENT_SPRING_HALF_Z,
    0.015,
    0.024,
    materials.phosphorBronze,
    disposables
  );

  // 1D. Bolted Satin-Brass Horological Front Curtain Bridge carrying B2's front bearing stud
  // Seated proudly FLUSH on the front face of CutawayFrontRim pedestal (+0.118, -0.488) -> B2 (+0.2106, -0.3758)
  // at z: +0.218..+0.236 with stepped brass spacer reaching back to B2 (+0.184) -> ZERO rim overlap or lever clip!
  {
    const ax = 0.118;
    const ay = -0.488;
    const bx = ARBOR_B2_X;
    const by = ARBOR_B2_Y;
    const vx = bx - ax;
    const vy = by - ay;
    const len = Math.hypot(vx, vy);
    const ux = vx / len;
    const uy = vy / len;
    const px = -uy;
    const py = ux;

    const rFoot = 0.030;
    const rBoss = 0.033;
    const bridgePts: THREE.Vector2[] = [];
    const baseAng = Math.atan2(vy, vx);
    for (let s = 0; s <= 10; s++) {
      const a = baseAng + Math.PI * 0.5 + Math.PI * (s / 10);
      bridgePts.push(new THREE.Vector2(ax + Math.cos(a) * rFoot, ay + Math.sin(a) * rFoot));
    }
    bridgePts.push(new THREE.Vector2(ax + vx * 0.48 - px * 0.020, ay + vy * 0.48 - py * 0.020));
    for (let s = 0; s <= 10; s++) {
      const a = baseAng - Math.PI * 0.5 + Math.PI * (s / 10);
      bridgePts.push(new THREE.Vector2(bx + Math.cos(a) * rBoss, by + Math.sin(a) * rBoss));
    }
    bridgePts.push(new THREE.Vector2(ax + vx * 0.48 + px * 0.020, ay + vy * 0.48 + py * 0.020));

    const bBevel = 0.0022;
    const bridgeShape = new THREE.Shape(ensureCCW(offsetClosedPolygonCCW(ensureCCW(bridgePts), -bBevel)));
    bridgeShape.holes.push(makeCircleHole(ax, ay, 0.017 + bBevel, 20));
    bridgeShape.holes.push(makeCircleHole(bx, by, 0.020 + bBevel, 22));

    const bridgeRaw = new THREE.ExtrudeGeometry(bridgeShape, {
      depth: 0.014,
      bevelEnabled: true,
      bevelThickness: bBevel,
      bevelSize: bBevel,
      bevelSegments: 2,
    });
    bridgeRaw.translate(0, 0, 0.220);
    const bridgeGeo = applyMachinedNormals(bridgeRaw);
    disposables.push(bridgeGeo);
    const bridgeMesh = new THREE.Mesh(bridgeGeo, materials.machinedGearBrass);
    bridgeMesh.name = 'CurtainIdlerFrontBridge';
    bridgeMesh.castShadow = true;
    bridgeMesh.receiveShadow = true;
    staticHardware.add(bridgeMesh);

    // Turned brass oil-sink stand-off sleeve bridging cleanly from front bridge (+0.218) back to B2 (+0.182)
    addBearingCollar(
      staticHardware,
      'CurtainBridgeStandoff_B2',
      ARBOR_B2_X,
      ARBOR_B2_Y,
      0.182,
      0.218,
      0.021,
      0.032,
      materials.forendBrass,
      disposables
    );

    // Countersunk slotted hardened-steel bridge anchor screw seated proudly on front face of bridge!
    const bridgeScrew = buildSlottedBrassScrew(
      0.0165,
      0.014,
      0.0055,
      0.006,
      materials.hardenedSteel,
      disposables
    );
    bridgeScrew.rotation.z = 0.72;
    bridgeScrew.position.set(ax, ay, 0.237);
    staticHardware.add(bridgeScrew);

    // Front turned hardened-steel journal stud carrying B2 strictly in Z ∈ [+0.152, +0.244] (clears Lever 4!)
    const b2StudGeo = makeClosedLatheZ(
      [
        [0.152, 0.015],
        [0.155, 0.021],
        [0.234, 0.021],
        [0.238, 0.027],
        [0.244, 0.019],
      ],
      24
    );
    b2StudGeo.translate(ARBOR_B2_X, ARBOR_B2_Y, 0);
    disposables.push(b2StudGeo);
    const b2StudMesh = new THREE.Mesh(b2StudGeo, materials.hardenedSteel);
    b2StudMesh.name = 'ArborStud_B2';
    b2StudMesh.castShadow = true;
    staticHardware.add(b2StudMesh);
  }

  // 1E. Full-Depth Steel Arbor B3 (anchored in backplate -0.182 .. +0.196, clears 360 key sweep!)
  const arborB3Geo = makeClosedLatheZ(
    [
      [-0.182, 0.021],
      [0.182, 0.021],
      [0.186, 0.030],
      [0.194, 0.024],
      [0.198, 0.015],
    ],
    26
  );
  arborB3Geo.translate(ARBOR_B3_X, ARBOR_B3_Y, 0);
  disposables.push(arborB3Geo);
  const arborB3Mesh = new THREE.Mesh(arborB3Geo, materials.hardenedSteel);
  arborB3Mesh.name = 'ArborPost_B3';
  arborB3Mesh.castShadow = true;
  arborB3Mesh.receiveShadow = true;
  staticHardware.add(arborB3Mesh);
  addBearingCollar(
    staticHardware,
    'ArborRearBushing_B3',
    ARBOR_B3_X,
    ARBOR_B3_Y,
    -0.180,
    0.046,
    0.022,
    0.038,
    materials.forendBrass,
    disposables
  );

  // 1F. Full-Depth Steel Arbor B4 (anchored in backplate -0.182 .. +0.182)
  const arborB4Geo = makeClosedLatheZ(
    [
      [-0.182, 0.025],
      [0.164, 0.025],
      [0.168, 0.036],
      [0.178, 0.028],
      [0.182, 0.018],
    ],
    28
  );
  arborB4Geo.translate(ARBOR_B4_X, ARBOR_B4_Y, 0);
  disposables.push(arborB4Geo);
  const arborB4Mesh = new THREE.Mesh(arborB4Geo, materials.hardenedSteel);
  arborB4Mesh.name = 'ArborPost_B4';
  arborB4Mesh.castShadow = true;
  arborB4Mesh.receiveShadow = true;
  staticHardware.add(arborB4Mesh);
  addBearingCollar(
    staticHardware,
    'ArborRearBushing_B4',
    ARBOR_B4_X,
    ARBOR_B4_Y,
    -0.180,
    0.046,
    0.026,
    0.046,
    materials.forendBrass,
    disposables
  );

  // 1G. Lower Key-Home Pawl Pivot Post & Spring Pin (anchored in backplate -0.182)
  const lPawlPostGeo = makeClosedLatheZ(
    [
      [-0.182, 0.0155],
      [0.164, 0.0155],
      [0.168, 0.023],
      [0.176, 0.018],
    ],
    22
  );
  lPawlPostGeo.translate(LOWER_KEYHOME_CAM_SPEC.pivotX, LOWER_KEYHOME_CAM_SPEC.pivotY, 0);
  disposables.push(lPawlPostGeo);
  const lPawlPostMesh = new THREE.Mesh(lPawlPostGeo, materials.hardenedSteel);
  lPawlPostMesh.name = 'LowerPawlPivotPost';
  lPawlPostMesh.castShadow = true;
  staticHardware.add(lPawlPostMesh);
  addBearingCollar(
    staticHardware,
    'LowerPawlStandoff',
    LOWER_KEYHOME_CAM_SPEC.pivotX,
    LOWER_KEYHOME_CAM_SPEC.pivotY,
    -0.180,
    LOWER_KEYHOME_CAM_SPEC.zBack,
    0.016,
    0.028,
    materials.forendBrass,
    disposables
  );

  const lSpringPinGeo = makeClosedLatheZ(
    [
      [-0.182, 0.0145],
      [0.156, 0.0145],
      [0.160, 0.021],
      [0.168, 0.016],
    ],
    20
  );
  lSpringPinGeo.translate(LOWER_KEYHOME_CAM_SPEC.springPinX, LOWER_KEYHOME_CAM_SPEC.springPinY, 0);
  disposables.push(lSpringPinGeo);
  const lSpringPinMesh = new THREE.Mesh(lSpringPinGeo, materials.forendBrass);
  lSpringPinMesh.name = 'LowerDetentSpringPin';
  lSpringPinMesh.castShadow = true;
  staticHardware.add(lSpringPinMesh);
  addBearingCollar(
    staticHardware,
    'LowerDetentSpringCollar',
    LOWER_KEYHOME_CAM_SPEC.springPinX,
    LOWER_KEYHOME_CAM_SPEC.springPinY,
    (LOWER_KEYHOME_CAM_SPEC.zBack + LOWER_KEYHOME_CAM_SPEC.zFront) * 0.5 - DETENT_SPRING_HALF_Z,
    (LOWER_KEYHOME_CAM_SPEC.zBack + LOWER_KEYHOME_CAM_SPEC.zFront) * 0.5 + DETENT_SPRING_HALF_Z,
    0.015,
    0.024,
    materials.phosphorBronze,
    disposables
  );

  // ===========================================================================
  // PART 2: ROTATING TRAIN A (UPPER RACK-AND-PINION + COMPOUND DETENT ESCAPEMENT)
  // ===========================================================================

  // 2A. Rotating Compound Group on Arbor U1 (12T Hardened-Steel Rack Pinion + 16T Spoked Brass Spur)
  const upperArborU1Group = new THREE.Group();
  upperArborU1Group.name = 'UpperCompoundArbor_U1';
  upperArborU1Group.position.set(ARBOR_U1_X, ARBOR_U1_Y, 0);
  group.add(upperArborU1Group);

  const u1aGeo = createInvoluteGearGeometry(GEAR_U1A_PINION, { boreRadius: 0.024 });
  u1aGeo.translate(0, 0, (GEAR_U1A_PINION.zBack + GEAR_U1A_PINION.zFront) * 0.5);
  disposables.push(u1aGeo);
  const u1aMesh = new THREE.Mesh(u1aGeo, materials.hardenedSteel);
  u1aMesh.name = GEAR_U1A_PINION.name;
  u1aMesh.castShadow = true;
  u1aMesh.receiveShadow = true;
  upperArborU1Group.add(u1aMesh);

  const u1bGeo = createInvoluteGearGeometry(GEAR_U1B_SPUR, { boreRadius: 0.024, spokeCount: 5 });
  u1bGeo.translate(0, 0, (GEAR_U1B_SPUR.zBack + GEAR_U1B_SPUR.zFront) * 0.5);
  disposables.push(u1bGeo);
  const u1bMesh = new THREE.Mesh(u1bGeo, materials.machinedGearBrass);
  u1bMesh.name = GEAR_U1B_SPUR.name;
  u1bMesh.castShadow = true;
  u1bMesh.receiveShadow = true;
  upperArborU1Group.add(u1bMesh);

  // Coaxial coupling sleeve joining U1A and U1B rigidly on Arbor U1
  addBearingCollar(
    upperArborU1Group,
    'CouplingSleeve_U1',
    0,
    0,
    GEAR_U1A_PINION.zFront,
    GEAR_U1B_SPUR.zBack,
    0.024,
    0.036,
    materials.machinedBronzeRack,
    disposables
  );

  // 2B. Rotating Detent Wheel Group on Arbor U2 (18T Spoked Brass Wheel + Foreground Dual-Notch Blued-Steel Cam)
  const upperArborU2Group = new THREE.Group();
  upperArborU2Group.name = 'UpperDetentArbor_U2';
  upperArborU2Group.position.set(ARBOR_U2_X, ARBOR_U2_Y, 0);
  group.add(upperArborU2Group);

  const u2aGeo = createInvoluteGearGeometry(GEAR_U2A_DRIVEN, { boreRadius: 0.026, spokeCount: 5 });
  u2aGeo.translate(0, 0, (GEAR_U2A_DRIVEN.zBack + GEAR_U2A_DRIVEN.zFront) * 0.5);
  disposables.push(u2aGeo);
  const u2aMesh = new THREE.Mesh(u2aGeo, materials.machinedGearBrass);
  u2aMesh.name = GEAR_U2A_DRIVEN.name;
  u2aMesh.castShadow = true;
  u2aMesh.receiveShadow = true;
  upperArborU2Group.add(u2aMesh);

  // Turned brass coaxial drive sleeve transferring torque from rear U2A forward to foreground U2B Detent Cam
  addBearingCollar(
    upperArborU2Group,
    'DriveSleeve_U2',
    0,
    0,
    GEAR_U2A_DRIVEN.zFront,
    UPPER_DETENT_CAM_SPEC.zBack,
    0.026,
    0.040,
    materials.machinedGearBrass,
    disposables
  );

  const u2bGeo = createDetentCamGeometry(UPPER_DETENT_CAM_SPEC, 0.026);
  u2bGeo.translate(0, 0, (UPPER_DETENT_CAM_SPEC.zBack + UPPER_DETENT_CAM_SPEC.zFront) * 0.5);
  disposables.push(u2bGeo);
  const u2bMesh = new THREE.Mesh(u2bGeo, materials.bluedSteelBolt);
  u2bMesh.name = 'UpperDetentEscapementCam_U2B';
  u2bMesh.castShadow = true;
  u2bMesh.receiveShadow = true;
  upperArborU2Group.add(u2bMesh);

  // Front chamfered brass oil-sink hub ring & eccentric turned steel throw-index indicator stud on U2B
  addBearingCollar(
    upperArborU2Group,
    'HubOilSink_U2B',
    0,
    0,
    UPPER_DETENT_CAM_SPEC.zFront,
    UPPER_DETENT_CAM_SPEC.zFront + 0.010,
    0.026,
    0.040,
    materials.machinedGearBrass,
    disposables
  );
  const u2StudGeo = makeClosedLatheZ(
    [
      [UPPER_DETENT_CAM_SPEC.zFront - 0.002, 0.011],
      [UPPER_DETENT_CAM_SPEC.zFront + 0.012, 0.011],
      [UPPER_DETENT_CAM_SPEC.zFront + 0.018, 0.007],
    ],
    18
  );
  // Placed cleanly on solid spoke at local angle 0 (r = 0.056) so user visibly tracks 206-deg throw sweep!
  u2StudGeo.translate(0.056, 0, 0);
  disposables.push(u2StudGeo);
  const u2StudMesh = new THREE.Mesh(u2StudGeo, materials.machinedGearBrass);
  u2StudMesh.name = 'ThrowIndexPin_U2B';
  u2StudMesh.castShadow = true;
  upperArborU2Group.add(u2StudMesh);

  // 2C. Upper Pivoting Detent Escapement Pawl + Flexing Phosphor-Bronze Blade Spring
  const upperPawlGroup = new THREE.Group();
  upperPawlGroup.name = 'UpperDetentPawlGroup';
  upperPawlGroup.position.set(
    UPPER_DETENT_CAM_SPEC.pivotX,
    UPPER_DETENT_CAM_SPEC.pivotY,
    (UPPER_DETENT_CAM_SPEC.zBack + UPPER_DETENT_CAM_SPEC.zFront) * 0.5
  );
  group.add(upperPawlGroup);

  const upperPawlGeo = createDetentPawlGeometry(UPPER_DETENT_CAM_SPEC);
  disposables.push(upperPawlGeo);
  const upperPawlMesh = new THREE.Mesh(upperPawlGeo, materials.hardenedSteel);
  upperPawlMesh.name = 'UpperDetentPawl';
  upperPawlMesh.castShadow = true;
  upperPawlMesh.receiveShadow = true;
  upperPawlGroup.add(upperPawlMesh);

  const upperDetentSpring = new DetentSpringMesh(
    'UpperDetentLeafSpring',
    UPPER_DETENT_CAM_SPEC,
    materials.phosphorBronze
  );
  group.add(upperDetentSpring.mesh);

  // ===========================================================================
  // PART 3: ROTATING TRAIN B (KEYHOLE ANTI-PICK CURTAIN -> COMPOUND KEY-HOME WHEEL)
  // ===========================================================================

  // 3A. Slotted Anti-Pick Keyhole Curtain Gear (B1) centered on Key Axis
  const curtainB1Group = new THREE.Group();
  curtainB1Group.name = 'KeyholeCurtainGearGroup_B1';
  curtainB1Group.position.set(KEY_X, KEY_Y, 0);
  group.add(curtainB1Group);

  const b1Geo = createInvoluteGearGeometry(GEAR_B1_CURTAIN, {
    boreRadius: 0.049,
    keyholeCurtainSlot: true,
  });
  b1Geo.translate(0, 0, (GEAR_B1_CURTAIN.zBack + GEAR_B1_CURTAIN.zFront) * 0.5);
  disposables.push(b1Geo);
  const b1Mesh = new THREE.Mesh(b1Geo, materials.machinedGearBrass);
  b1Mesh.name = GEAR_B1_CURTAIN.name;
  b1Mesh.castShadow = true;
  b1Mesh.receiveShadow = true;
  curtainB1Group.add(b1Mesh);

  // 3B. Curtain Transfer Idler Gear (B2) carried on the bolted Front Horological Bridge
  const curtainIdlerB2Group = new THREE.Group();
  curtainIdlerB2Group.name = 'CurtainIdlerGroup_B2';
  curtainIdlerB2Group.position.set(ARBOR_B2_X, ARBOR_B2_Y, 0);
  group.add(curtainIdlerB2Group);

  const b2Geo = createInvoluteGearGeometry(GEAR_B2_IDLER, { boreRadius: 0.022, spokeCount: 5 });
  b2Geo.translate(0, 0, (GEAR_B2_IDLER.zBack + GEAR_B2_IDLER.zFront) * 0.5);
  disposables.push(b2Geo);
  const b2Mesh = new THREE.Mesh(b2Geo, materials.machinedGearBrass);
  b2Mesh.name = GEAR_B2_IDLER.name;
  b2Mesh.castShadow = true;
  b2Mesh.receiveShadow = true;
  curtainIdlerB2Group.add(b2Mesh);

  // 3C. Compound Curtain Step Arbor Group (B3: 18T Front Brass Step Gear + 15T Rear Drilled Bronze Pinion)
  const curtainCompoundB3Group = new THREE.Group();
  curtainCompoundB3Group.name = 'CurtainCompoundArbor_B3';
  curtainCompoundB3Group.position.set(ARBOR_B3_X, ARBOR_B3_Y, 0);
  group.add(curtainCompoundB3Group);

  const b3fGeo = createInvoluteGearGeometry(GEAR_B3F_STEP, { boreRadius: 0.022, spokeCount: 5 });
  b3fGeo.translate(0, 0, (GEAR_B3F_STEP.zBack + GEAR_B3F_STEP.zFront) * 0.5);
  disposables.push(b3fGeo);
  const b3fMesh = new THREE.Mesh(b3fGeo, materials.machinedGearBrass);
  b3fMesh.name = GEAR_B3F_STEP.name;
  b3fMesh.castShadow = true;
  b3fMesh.receiveShadow = true;
  curtainCompoundB3Group.add(b3fMesh);

  const b3rGeo = createInvoluteGearGeometry(GEAR_B3R_PINION, { boreRadius: 0.022, drillHoles: 5 });
  b3rGeo.translate(0, 0, (GEAR_B3R_PINION.zBack + GEAR_B3R_PINION.zFront) * 0.5);
  disposables.push(b3rGeo);
  const b3rMesh = new THREE.Mesh(b3rGeo, materials.machinedBronzeRack);
  b3rMesh.name = GEAR_B3R_PINION.name;
  b3rMesh.castShadow = true;
  b3rMesh.receiveShadow = true;
  curtainCompoundB3Group.add(b3rMesh);

  addBearingCollar(
    curtainCompoundB3Group,
    'CouplingSleeve_B3',
    0,
    0,
    GEAR_B3R_PINION.zFront,
    GEAR_B3F_STEP.zBack,
    0.022,
    0.034,
    materials.hardenedSteel,
    disposables
  );

  // 3D. Grand 5-Spoked Key-Home Synchronizer Wheel (B4A) + Foreground Blued-Steel Key-Home Cam (B4B)
  const keyHomeB4Group = new THREE.Group();
  keyHomeB4Group.name = 'KeyHomeSynchronizerArbor_B4';
  keyHomeB4Group.position.set(ARBOR_B4_X, ARBOR_B4_Y, 0);
  group.add(keyHomeB4Group);

  const b4aGeo = createInvoluteGearGeometry(GEAR_B4A_KEYHOME, { boreRadius: 0.026, spokeCount: 5 });
  b4aGeo.translate(0, 0, (GEAR_B4A_KEYHOME.zBack + GEAR_B4A_KEYHOME.zFront) * 0.5);
  disposables.push(b4aGeo);
  const b4aMesh = new THREE.Mesh(b4aGeo, materials.machinedGearBrass);
  b4aMesh.name = GEAR_B4A_KEYHOME.name;
  b4aMesh.castShadow = true;
  b4aMesh.receiveShadow = true;
  keyHomeB4Group.add(b4aMesh);

  addBearingCollar(
    keyHomeB4Group,
    'DriveSpacer_B4',
    0,
    0,
    GEAR_B4A_KEYHOME.zFront,
    LOWER_KEYHOME_CAM_SPEC.zBack,
    0.026,
    0.042,
    materials.machinedBronzeRack,
    disposables
  );

  const b4bGeo = createDetentCamGeometry(LOWER_KEYHOME_CAM_SPEC, 0.026);
  b4bGeo.translate(0, 0, (LOWER_KEYHOME_CAM_SPEC.zBack + LOWER_KEYHOME_CAM_SPEC.zFront) * 0.5);
  disposables.push(b4bGeo);
  const b4bMesh = new THREE.Mesh(b4bGeo, materials.bluedSteelBolt);
  b4bMesh.name = 'LowerKeyHomeDetentCam_B4B';
  b4bMesh.castShadow = true;
  b4bMesh.receiveShadow = true;
  keyHomeB4Group.add(b4bMesh);

  // Turned brass oil-sink front collar & eccentric key-phase index stud on B4B
  addBearingCollar(
    keyHomeB4Group,
    'HubOilSink_B4B',
    0,
    0,
    LOWER_KEYHOME_CAM_SPEC.zFront,
    LOWER_KEYHOME_CAM_SPEC.zFront + 0.008,
    0.026,
    0.040,
    materials.machinedGearBrass,
    disposables
  );
  const b4StudGeo = makeClosedLatheZ(
    [
      [LOWER_KEYHOME_CAM_SPEC.zFront - 0.002, 0.011],
      [LOWER_KEYHOME_CAM_SPEC.zFront + 0.011, 0.011],
      [LOWER_KEYHOME_CAM_SPEC.zFront + 0.016, 0.007],
    ],
    18
  );
  b4StudGeo.translate(0.056, 0, 0);
  disposables.push(b4StudGeo);
  const b4StudMesh = new THREE.Mesh(b4StudGeo, materials.machinedGearBrass);
  b4StudMesh.name = 'KeyPhaseIndexPin_B4B';
  b4StudMesh.castShadow = true;
  keyHomeB4Group.add(b4StudMesh);

  // 3E. Lower Key-Home Pivoting Detent Pawl + Flexing Phosphor-Bronze Blade Spring
  const lowerPawlGroup = new THREE.Group();
  lowerPawlGroup.name = 'LowerKeyHomePawlGroup';
  lowerPawlGroup.position.set(
    LOWER_KEYHOME_CAM_SPEC.pivotX,
    LOWER_KEYHOME_CAM_SPEC.pivotY,
    (LOWER_KEYHOME_CAM_SPEC.zBack + LOWER_KEYHOME_CAM_SPEC.zFront) * 0.5
  );
  group.add(lowerPawlGroup);

  const lowerPawlGeo = createDetentPawlGeometry(LOWER_KEYHOME_CAM_SPEC);
  disposables.push(lowerPawlGeo);
  const lowerPawlMesh = new THREE.Mesh(lowerPawlGeo, materials.hardenedSteel);
  lowerPawlMesh.name = 'LowerKeyHomePawl';
  lowerPawlMesh.castShadow = true;
  lowerPawlMesh.receiveShadow = true;
  lowerPawlGroup.add(lowerPawlMesh);

  const lowerDetentSpring = new DetentSpringMesh(
    'LowerKeyHomeLeafSpring',
    LOWER_KEYHOME_CAM_SPEC,
    materials.phosphorBronze
  );
  group.add(lowerDetentSpring.mesh);

  return {
    group,
    upperArborU1Group,
    upperArborU2Group,
    upperPawlGroup,
    upperDetentSpring,
    curtainB1Group,
    curtainIdlerB2Group,
    curtainCompoundB3Group,
    keyHomeB4Group,
    lowerPawlGroup,
    lowerDetentSpring,
    update: (u1Rad, u2Rad, upperPawlRad, b1Rad, b2Rad, b3Rad, b4Rad, lowerPawlRad) => {
      upperArborU1Group.rotation.z = u1Rad;
      upperArborU2Group.rotation.z = u2Rad;
      upperPawlGroup.rotation.z = upperPawlRad;
      upperDetentSpring.update(upperPawlRad);

      curtainB1Group.rotation.z = b1Rad;
      curtainIdlerB2Group.rotation.z = b2Rad;
      curtainCompoundB3Group.rotation.z = b3Rad;
      keyHomeB4Group.rotation.z = b4Rad;
      lowerPawlGroup.rotation.z = lowerPawlRad;
      lowerDetentSpring.update(lowerPawlRad);
    },
    dispose: () => {
      upperDetentSpring.dispose();
      lowerDetentSpring.dispose();
      for (const g of disposables) g.dispose();
    },
  };
}
