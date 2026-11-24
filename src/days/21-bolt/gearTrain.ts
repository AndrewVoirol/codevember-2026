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
  GEAR_B1_KNOB_DRIVE,
  GEAR_B2A_PINION,
  GEAR_B2B_STEP,
  GEAR_B3F_TRANSFER,
  GEAR_B3R_PINION,
  GEAR_B4A_DETENT,
  GEAR_C1_CAM_DRIVEN,
  GEAR_U1A_PINION,
  GEAR_U1B_SPUR,
  GEAR_U2A_DRIVEN,
  KEY_X,
  KEY_Y,
  LOWER_DETENT_CAM_SPEC,
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
const DETENT_SPRING_SEGMENTS = 24;
const DETENT_SPRING_HALF_Z = 0.0098;
const DETENT_SPRING_HALF_N = 0.0062;
const PAWL_SPRING_PAD_LOCAL_X = 0.078;
const PAWL_SPRING_PAD_LOCAL_Y = -0.022;

interface GearBuilderOptions {
  boreRadius: number;
  spokeCount?: number;
  drillHoles?: number;
}

/**
 * Builds an outward-wound, machined-normal 3D involute spur gear strictly from `SpurGearSpec`.
 * Outer contour is inset by `GEAR_BEVEL` before extrusion so the beveled outer metal skin lands on
 * exact analytical pitch/involute tooth coordinates (zero tooth clipping across full travel!).
 */
export function createInvoluteGearGeometry(
  spec: SpurGearSpec,
  opts: GearBuilderOptions
): THREE.BufferGeometry {
  const rawPts = buildInvoluteGearPolygon2D(spec, 4).map((p) => new THREE.Vector2(p.x, p.y));
  const insetPts = offsetClosedPolygonCCW(ensureCCW(rawPts), -GEAR_BEVEL);
  const shape = new THREE.Shape(ensureCCW(insetPts));

  shape.holes.push(makeCircleHole(0, 0, opts.boreRadius + GEAR_BEVEL, 26));

  if (opts.spokeCount && opts.spokeCount >= 3) {
    const nSpokes = opts.spokeCount;
    const rIn = opts.boreRadius + 0.017 + GEAR_BEVEL;
    const rOut = spec.rootRadius - 0.014 - GEAR_BEVEL;
    if (rOut - rIn > 0.014) {
      const step = (Math.PI * 2) / nSpokes;
      const spokeHalfW = 0.0082 + GEAR_BEVEL;
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
    const holeR = Math.min(0.0115, (spec.rootRadius - opts.boreRadius) * 0.24);
    for (let k = 0; k < opts.drillHoles; k++) {
      const a = (k / opts.drillHoles) * Math.PI * 2;
      shape.holes.push(makeCircleHole(Math.cos(a) * rRing, Math.sin(a) * rRing, holeR + GEAR_BEVEL, 18));
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

  const rRing = (boreRadius + spec.rootRadius) * 0.50;
  const holeR = Math.min(0.010, (spec.rootRadius - boreRadius) * 0.24);
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
 * Builds an outward-wound sculpted watchmaker hardened-steel detent escapement pawl in pawl-local XY:
 *  - Local (0, 0): pivot post center
 *  - Local (+pawlArmLength, 0): exact center of rounded contact beak (radius = spec.beakRadius)
 *  - Local (-Y): substantial flat-machined back shoulder where the phosphor-bronze detent click spring kisses flush
 *  - Local (+Y): gently hollowed throat clearing the cam rim cleanly across full V-notch drop
 */
export function createDetentPawlGeometry(spec: DetentCamSpec): THREE.BufferGeometry {
  const pawlBevel = 0.0014;
  const L = spec.pawlArmLength;
  const rBeak = spec.beakRadius;
  const rPivot = 0.027;

  const nominal: THREE.Vector2[] = [];

  // Semicircular pivot ear from +PI/2 around PI to -PI/2
  const earSegs = 12;
  for (let s = 0; s <= earSegs; s++) {
    const a = Math.PI * 0.5 + Math.PI * (s / earSegs);
    nominal.push(new THREE.Vector2(Math.cos(a) * rPivot, Math.sin(a) * rPivot));
  }

  // Substantial tapered watchmaker back spine (-Y side) with flat spring-landing shoulder
  nominal.push(new THREE.Vector2(0.036, -0.025));
  nominal.push(new THREE.Vector2(PAWL_SPRING_PAD_LOCAL_X - 0.018, PAWL_SPRING_PAD_LOCAL_Y));
  nominal.push(new THREE.Vector2(PAWL_SPRING_PAD_LOCAL_X + 0.018, PAWL_SPRING_PAD_LOCAL_Y));
  nominal.push(new THREE.Vector2(L - 0.011, -0.015));

  // Rounded steel escapement beak centered strictly at (L, 0) with exact radius `rBeak`
  const beakSegs = 12;
  for (let s = 0; s <= beakSegs; s++) {
    const a = -Math.PI * 0.45 + (Math.PI * 1.15 * s) / beakSegs;
    nominal.push(new THREE.Vector2(L + Math.cos(a) * rBeak, Math.sin(a) * rBeak));
  }

  // Clean hollowed inner throat clearing cam perimeter cleanly with zero fragility
  nominal.push(new THREE.Vector2(L - 0.019, 0.001));
  nominal.push(new THREE.Vector2(0.068, -0.002));
  nominal.push(new THREE.Vector2(0.034, 0.011));

  const inset = offsetClosedPolygonCCW(ensureCCW(nominal), -pawlBevel);
  const shape = new THREE.Shape(ensureCCW(inset));
  shape.holes.push(makeCircleHole(0, 0, 0.016 + pawlBevel, 22));

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

export interface DetentSpringDiagnostics {
  tipContactGap: number;
  minDistFromCam: number;
  maxY: number;
  minY: number;
}

/**
 * Zero-allocation dynamically flexing curved phosphor-bronze watchmaker click spring.
 * Parameterized strictly from the pawl back pad (`s = 0`, tangent along `-padT` so outward normal
 * `(-ty, +tx)` aligns strictly with `+padN` -> exact flush bottom contact & positive outward winding!)
 * curving gracefully up into the anchored spring collar & reaction backup pin (`s = N`).
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
  private lastPadX = 0;
  private lastPadY = 0;

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

  inspectAtPawlAngle(pawlWorldAngleRad: number): DetentSpringDiagnostics {
    this.update(pawlWorldAngleRad);
    const hTip = DETENT_SPRING_HALF_N * 0.85;
    const tipBotX = this.cx[0] - this.nx[0] * hTip;
    const tipBotY = this.cy[0] - this.ny[0] * hTip;
    const tipContactGap = Math.hypot(tipBotX - this.lastPadX, tipBotY - this.lastPadY);

    let minDistFromCam = Infinity;
    let maxY = -Infinity;
    let minY = Infinity;
    for (let s = 0; s <= DETENT_SPRING_SEGMENTS; s++) {
      const h = DETENT_SPRING_HALF_N * (0.85 + 0.25 * (s / DETENT_SPRING_SEGMENTS));
      for (const sign of [-1, 1]) {
        const px = this.cx[s] + this.nx[s] * h * sign;
        const py = this.cy[s] + this.ny[s] * h * sign;
        const d = Math.hypot(px - this.spec.cx, py - this.spec.cy);
        if (d < minDistFromCam) minDistFromCam = d;
        if (py > maxY) maxY = py;
        if (py < minY) minY = py;
      }
    }
    return { tipContactGap, minDistFromCam, maxY, minY };
  }

  update(pawlWorldAngleRad: number): void {
    if (Math.abs(pawlWorldAngleRad - this.lastPawlAngle) < 1e-5) return;
    this.lastPawlAngle = pawlWorldAngleRad;

    const zMid = (this.spec.zBack + this.spec.zFront) * 0.5;
    const zBack = zMid - DETENT_SPRING_HALF_Z;
    const zFront = zMid + DETENT_SPRING_HALF_Z;

    const cosP = Math.cos(pawlWorldAngleRad);
    const sinP = Math.sin(pawlWorldAngleRad);

    // Pawl contact shoulder + outward normal (+sinP, -cosP) pointing up-right away from pawl
    const padX =
      this.spec.pivotX + PAWL_SPRING_PAD_LOCAL_X * cosP - PAWL_SPRING_PAD_LOCAL_Y * sinP;
    const padY =
      this.spec.pivotY + PAWL_SPRING_PAD_LOCAL_X * sinP + PAWL_SPRING_PAD_LOCAL_Y * cosP;
    this.lastPadX = padX;
    this.lastPadY = padY;

    const padNx = sinP;
    const padNy = -cosP;
    const padTx = cosP;
    const padTy = sinP;

    // Start at s = 0 strictly flush on the pawl spine (`bot0 = pad`, tangent along `-padT` -> `(-ty, +tx) = +padN`!)
    const hTip = DETENT_SPRING_HALF_N * 0.85;
    const p0x = padX + padNx * hTip;
    const p0y = padY + padNy * hTip;

    // Depart strictly tangent along the pawl back shoulder (tipContactGap < 0.0005),
    // then arch smoothly upward into the upper-left flank of the bronze spring collar (zero U-dip!)
    const p1x = p0x - padTx * 0.018 + padNx * 0.0012;
    const p1y = p0y - padTy * 0.018 + padNy * 0.0012;

    const p3x = this.spec.springPinX - 0.014;
    const p3y = this.spec.springPinY + 0.008;
    const p2x = p3x - 0.032;
    const p2y = p3y + 0.004;

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
      const h0 = DETENT_SPRING_HALF_N * (0.85 + 0.25 * (s / DETENT_SPRING_SEGMENTS));
      const h1 = DETENT_SPRING_HALF_N * (0.85 + 0.25 * ((s + 1) / DETENT_SPRING_SEGMENTS));

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

    const hStart = DETENT_SPRING_HALF_N * 0.85;
    writeQuad(
      this.cx[0] - this.nx[0] * hStart, this.cy[0] - this.ny[0] * hStart, zBack, -this.ny[0], this.nx[0], 0,
      this.cx[0] - this.nx[0] * hStart, this.cy[0] - this.ny[0] * hStart, zFront, -this.ny[0], this.nx[0], 0,
      this.cx[0] + this.nx[0] * hStart, this.cy[0] + this.ny[0] * hStart, zFront, -this.ny[0], this.nx[0], 0,
      this.cx[0] + this.nx[0] * hStart, this.cy[0] + this.ny[0] * hStart, zBack, -this.ny[0], this.nx[0], 0
    );

    const hEnd = DETENT_SPRING_HALF_N * 1.10;
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
  knobDriveB1Group: THREE.Group;
  stepUpB2Group: THREE.Group;
  camDrivenC1Group: THREE.Group;
  transferB3Group: THREE.Group;
  lowerDetentB4Group: THREE.Group;
  lowerPawlGroup: THREE.Group;
  lowerDetentSpring: DetentSpringMesh;
  update: (
    u1Rad: number,
    u2Rad: number,
    upperPawlRad: number,
    b1Rad: number,
    b2Rad: number,
    c1Rad: number,
    b3Rad: number,
    b4Rad: number,
    lowerPawlRad: number
  ) => void;
  dispose: () => void;
}

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
 * Constructs the complete, structurally anchored, load-bearing Horological Geared Deadbolt Drive Train
 * completely sealed behind the museum crystal inspection pane (`max.z <= 0.198 < GLASS_Z_BACK = 0.204`):
 *  - Train A (Upper-Right): DeadboltUpperRack -> Arbor U1 (12T steel rack pinion + 16T spoked brass spur)
 *    -> Arbor U2 (18T spoked brass wheel + foreground blued-steel dual-notch cam) + Upper Detent Pawl & Click Spring.
 *  - Train B (Load-Bearing 3.000:1 Coaxial Thumb-Turn Transmission + Lower Dual-Notch Detent):
 *    -> B1 (28T Spoked Front Brass Knob Drive Wheel on outer sleeve, z: 0.162..0.178)
 *    -> Compound Countershaft B2 (B2a 14T hardened-steel input pinion + B2b 24T spoked brass step wheel, carried on
 *       an internal skeletonized satin-brass Horological Gear Bridge spanning backplate anchor -> B2 -> B3!)
 *    -> C1 (16T Inner Cam-Barrel Driven Pinion at exact 3.000:1 same-direction speed-up!)
 *    -> Compound Detent Transfer Arbor B3 (B3f 16T front brass gear + B3r 14T rear bronze pinion)
 *    -> Grand 5-Spoked Lower Detent Wheel B4 (B4a 20T brass wheel + B4B dual-notch Locked/Unlocked blued-steel cam)
 *    -> Lower Detent Pawl & Watchmaker Click Spring.
 */
export function buildGearTrainAssembly(materials: BoltMaterials): GearTrainAssembly {
  const group = new THREE.Group();
  group.name = 'HorologicalGearTrainAssembly';
  const disposables: THREE.BufferGeometry[] = [];

  // ===========================================================================
  // PART 1: STATIONARY BACKPLATE ARBORS, OIL-SINK BEARINGS & INTERNAL BRIDGE
  // ===========================================================================
  const staticHardware = new THREE.Group();
  staticHardware.name = 'GearTrainStaticArborsAndBridge';
  group.add(staticHardware);

  // 1A. Arbor U1 Steel Post (-0.182 .. -0.058) + Rear Bushing
  const arborU1Geo = makeClosedLatheZ(
    [
      [-0.182, 0.022],
      [-0.068, 0.022],
      [-0.064, 0.030],
      [-0.058, 0.024],
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
    0.023,
    0.042,
    materials.forendBrass,
    disposables
  );

  // 1B. Arbor U2 Full-Depth Steel Post (-0.182 .. +0.112) + Rear Bushing
  const arborU2Geo = makeClosedLatheZ(
    [
      [-0.182, 0.024],
      [0.096, 0.024],
      [0.100, 0.034],
      [0.108, 0.027],
      [0.112, 0.017],
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
    -0.110,
    0.025,
    0.044,
    materials.forendBrass,
    disposables
  );

  // 1C. Upper Detent Pawl Pivot Post, Spring Pin & Anti-Rotation Backup Pin
  const uPawlPostGeo = makeClosedLatheZ(
    [
      [-0.182, 0.015],
      [0.090, 0.015],
      [0.094, 0.022],
      [0.102, 0.017],
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
    0.027,
    materials.forendBrass,
    disposables
  );

  const uSpringPinGeo = makeClosedLatheZ(
    [
      [-0.182, 0.014],
      [0.080, 0.014],
      [0.085, 0.020],
      [0.092, 0.015],
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
    0.0145,
    0.023,
    materials.phosphorBronze,
    disposables
  );

  // Steel reaction pin bracing the upper click spring collar against unwinding
  const uBackPinGeo = makeClosedLatheZ(
    [
      [-0.182, 0.008],
      [0.078, 0.008],
      [0.084, 0.005],
    ],
    16
  );
  uBackPinGeo.translate(
    UPPER_DETENT_CAM_SPEC.springPinX + 0.024,
    UPPER_DETENT_CAM_SPEC.springPinY - 0.016,
    0
  );
  disposables.push(uBackPinGeo);
  const uBackPinMesh = new THREE.Mesh(uBackPinGeo, materials.hardenedSteel);
  uBackPinMesh.name = 'UpperDetentReactionPin';
  staticHardware.add(uBackPinMesh);

  // 1D. Internal Skeletonized Satin-Brass Horological Gear Bridge (100% INSIDE the sealed glass cavity!)
  // Spans solidly from a full-depth lower-left backplate pillar P_A (-0.140, -0.468, safely clear of cam sweep!)
  // through Countershaft B2 (0.201, -0.373) to Full-Depth Backplate Arbor B3 (0.419, -0.422) at z: +0.182..+0.194!
  {
    const ax = -0.142;
    const ay = -0.466;
    const bx = ARBOR_B2_X;
    const by = ARBOR_B2_Y;
    const cx = ARBOR_B3_X;
    const cy = ARBOR_B3_Y;

    // Backplate anchor pillar at P_A in dark gunmetal steel (eliminates double-bar illusion behind brass bridge!)
    const pillarGeo = makeClosedLatheZ(
      [
        [-0.182, 0.024],
        [0.172, 0.024],
        [0.175, 0.020],
      ],
      24
    );
    pillarGeo.translate(ax, ay, 0);
    disposables.push(pillarGeo);
    const pillarMesh = new THREE.Mesh(pillarGeo, materials.gunmetalCase);
    pillarMesh.name = 'LowerBridgeAnchorPillar';
    pillarMesh.castShadow = true;
    pillarMesh.receiveShadow = true;
    staticHardware.add(pillarMesh);

    addBearingCollar(
      staticHardware,
      'LowerBridgeAnchorSeat',
      ax,
      ay,
      0.172,
      0.182,
      0.016,
      0.027,
      materials.forendBrass,
      disposables
    );

    // Sculpted 3-boss watchmaker bridge plate (z: +0.182 .. +0.194 -> clears glass +0.204 by 0.010!)
    const rA = 0.030;
    const rB = 0.036;
    const rC = 0.031;
    const angAB = Math.atan2(by - ay, bx - ax);
    const angBC = Math.atan2(cy - by, cx - bx);
    const pABx = -Math.sin(angAB);
    const pABy = Math.cos(angAB);
    const pBCx = -Math.sin(angBC);
    const pBCy = Math.cos(angBC);

    const bridgePts: THREE.Vector2[] = [];
    // Left boss A outer arc (+PI/2 -> +3PI/2)
    for (let s = 0; s <= 10; s++) {
      const a = angAB + Math.PI * 0.5 + Math.PI * (s / 10);
      bridgePts.push(new THREE.Vector2(ax + Math.cos(a) * rA, ay + Math.sin(a) * rA));
    }
    // Lower waist A -> B
    bridgePts.push(
      new THREE.Vector2((ax + bx) * 0.5 - pABx * 0.016, (ay + by) * 0.5 - pABy * 0.016)
    );
    bridgePts.push(new THREE.Vector2(bx - pABx * (rB * 0.88), by - pABy * (rB * 0.88)));
    // Lower waist B -> C
    bridgePts.push(
      new THREE.Vector2((bx + cx) * 0.5 - pBCx * 0.016, (by + cy) * 0.5 - pBCy * 0.016)
    );
    // Right boss C outer arc (-PI/2 -> +PI/2)
    for (let s = 0; s <= 10; s++) {
      const a = angBC - Math.PI * 0.5 + Math.PI * (s / 10);
      bridgePts.push(new THREE.Vector2(cx + Math.cos(a) * rC, cy + Math.sin(a) * rC));
    }
    // Upper waist C -> B
    bridgePts.push(
      new THREE.Vector2((bx + cx) * 0.5 + pBCx * 0.016, (by + cy) * 0.5 + pBCy * 0.016)
    );
    // Center boss B upper arc
    bridgePts.push(new THREE.Vector2(bx + pBCx * (rB * 0.92), by + pBCy * (rB * 0.92)));
    bridgePts.push(new THREE.Vector2(bx + pABx * (rB * 0.92), by + pABy * (rB * 0.92)));
    // Upper waist B -> A
    bridgePts.push(
      new THREE.Vector2((ax + bx) * 0.5 + pABx * 0.016, (ay + by) * 0.5 + pABy * 0.016)
    );

    const bBevel = 0.0018;
    const bridgeShape = new THREE.Shape(
      ensureCCW(offsetClosedPolygonCCW(ensureCCW(bridgePts), -bBevel))
    );
    bridgeShape.holes.push(makeCircleHole(ax, ay, 0.015 + bBevel, 20));
    bridgeShape.holes.push(makeCircleHole(bx, by, 0.020 + bBevel, 22));
    bridgeShape.holes.push(makeCircleHole(cx, cy, 0.020 + bBevel, 22));

    const bridgeRaw = new THREE.ExtrudeGeometry(bridgeShape, {
      depth: 0.0084,
      bevelEnabled: true,
      bevelThickness: bBevel,
      bevelSize: bBevel,
      bevelSegments: 2,
    });
    bridgeRaw.translate(0, 0, 0.1838);
    const bridgeGeo = applyMachinedNormals(bridgeRaw);
    disposables.push(bridgeGeo);
    const bridgeMesh = new THREE.Mesh(bridgeGeo, materials.machinedGearBrass);
    bridgeMesh.name = 'LowerHorologicalGearBridge';
    bridgeMesh.castShadow = true;
    bridgeMesh.receiveShadow = true;
    staticHardware.add(bridgeMesh);

    // Countersunk slotted blued-steel anchor screw at Pillar A (zMax = 0.197 < glass 0.204!)
    const anchorScrew = buildSlottedBrassScrew(
      0.0155,
      0.012,
      0.0052,
      0.0055,
      materials.bluedSteelBolt,
      disposables
    );
    anchorScrew.rotation.z = 0.65;
    anchorScrew.position.set(ax, ay, 0.196);
    staticHardware.add(anchorScrew);

    // Turned hardened-steel Countershaft Journal Stud carrying B2 strictly inside Z in [+0.132, +0.197]
    const b2StudGeo = makeClosedLatheZ(
      [
        [0.132, 0.015],
        [0.135, 0.020],
        [0.190, 0.020],
        [0.194, 0.026],
        [0.198, 0.018],
      ],
      24
    );
    b2StudGeo.translate(ARBOR_B2_X, ARBOR_B2_Y, 0);
    disposables.push(b2StudGeo);
    const b2StudMesh = new THREE.Mesh(b2StudGeo, materials.hardenedSteel);
    b2StudMesh.name = 'ArborStud_B2';
    b2StudMesh.castShadow = true;
    staticHardware.add(b2StudMesh);

    // Brass oil-sink bearing collar between B2a and bridge
    addBearingCollar(
      staticHardware,
      'BridgeBushing_B2',
      ARBOR_B2_X,
      ARBOR_B2_Y,
      0.178,
      0.183,
      0.0205,
      0.031,
      materials.forendBrass,
      disposables
    );

    // 1D-bis. Skeletonized Satin-Brass Thumb-Turn Spindle Bearing Cock rising from Pillar A -> (KEY_X, KEY_Y)
    // Physically anchors the stationary brass thumb-turn bearing & 0/90 index pips straight into the case backplate
    // while leaving 100% of the B1<->B2A and B2B<->C1 involute gear mesh unobstructed!
    {
      const kx = KEY_X;
      const ky = KEY_Y;
      const angAK = Math.atan2(ky - ay, kx - ax);
      const pAKx = -Math.sin(angAK);
      const pAKy = Math.cos(angAK);
      const rFootK = 0.029;
      const rHubK = 0.088;

      const cockPts: THREE.Vector2[] = [];
      for (let s = 0; s <= 10; s++) {
        const a = angAK + Math.PI * 0.5 + Math.PI * (s / 10);
        cockPts.push(new THREE.Vector2(ax + Math.cos(a) * rFootK, ay + Math.sin(a) * rFootK));
      }
      cockPts.push(
        new THREE.Vector2((ax + kx) * 0.5 - pAKx * 0.017, (ay + ky) * 0.5 - pAKy * 0.017)
      );
      for (let s = 0; s <= 16; s++) {
        const a = angAK - Math.PI * 0.5 + Math.PI * (s / 16);
        cockPts.push(new THREE.Vector2(kx + Math.cos(a) * rHubK, ky + Math.sin(a) * rHubK));
      }
      cockPts.push(
        new THREE.Vector2((ax + kx) * 0.5 + pAKx * 0.017, (ay + ky) * 0.5 + pAKy * 0.017)
      );

      const cBevel = 0.0018;
      const cockShape = new THREE.Shape(
        ensureCCW(offsetClosedPolygonCCW(ensureCCW(cockPts), -cBevel))
      );
      cockShape.holes.push(makeCircleHole(ax, ay, 0.015 + cBevel, 20));
      cockShape.holes.push(makeCircleHole(kx, ky, 0.050 + cBevel, 32));

      const cockRaw = new THREE.ExtrudeGeometry(cockShape, {
        depth: 0.0084,
        bevelEnabled: true,
        bevelThickness: cBevel,
        bevelSize: cBevel,
        bevelSegments: 2,
      });
      cockRaw.translate(0, 0, 0.1938);
      const cockGeo = applyMachinedNormals(cockRaw);
      disposables.push(cockGeo);
      const cockMesh = new THREE.Mesh(cockGeo, materials.machinedGearBrass);
      cockMesh.name = 'ThumbTurnSpindleBridgeCock';
      cockMesh.castShadow = true;
      cockMesh.receiveShadow = true;
      staticHardware.add(cockMesh);

      // Re-seat Pillar A's blued-steel anchor screw proud of the stacked bridge + cock feet
      anchorScrew.position.set(ax, ay, 0.206);

      // Stepped satin-brass bearing bezel collar on front of the spindle cock carrying the 0/90 index pips
      addBearingCollar(
        staticHardware,
        'ThumbTurnBearingCollar',
        kx,
        ky,
        0.180,
        0.205,
        0.0495,
        0.084,
        materials.forendBrass,
        disposables
      );

      // Two inlaid blued-steel 90-degree quarter-turn index pips at 12 o'clock (Unlocked) & 3 o'clock (Locked)
      for (const [label, ang] of [
        ['BridgeUnlockPip', Math.PI * 0.5],
        ['BridgeLockPip', 0],
      ] as const) {
        const rPip = 0.068;
        const pipGeo = makeClosedLatheZ(
          [
            [0.202, 0.0085],
            [0.209, 0.0085],
            [0.212, 0.005],
            [0.213, 0.0],
          ],
          18
        );
        pipGeo.translate(kx + Math.cos(ang) * rPip, ky + Math.sin(ang) * rPip, 0);
        disposables.push(pipGeo);
        const pipMesh = new THREE.Mesh(pipGeo, materials.bluedSteelBolt);
        pipMesh.name = label;
        pipMesh.castShadow = true;
        staticHardware.add(pipMesh);
      }
    }
  }

  // 1E. Full-Depth Steel Arbor B3 (anchored in backplate -0.182 .. +0.198, supports right end of bridge!)
  const arborB3Geo = makeClosedLatheZ(
    [
      [-0.182, 0.020],
      [0.188, 0.020],
      [0.192, 0.028],
      [0.197, 0.022],
      [0.199, 0.014],
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
    0.021,
    0.036,
    materials.forendBrass,
    disposables
  );
  addBearingCollar(
    staticHardware,
    'BridgeStandoff_B3',
    ARBOR_B3_X,
    ARBOR_B3_Y,
    0.156,
    0.182,
    0.021,
    0.032,
    materials.forendBrass,
    disposables
  );

  // 1F. Full-Depth Steel Arbor B4 (anchored in backplate -0.182 .. +0.174)
  const arborB4Geo = makeClosedLatheZ(
    [
      [-0.182, 0.024],
      [0.156, 0.024],
      [0.160, 0.034],
      [0.170, 0.026],
      [0.174, 0.017],
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
    0.025,
    0.044,
    materials.forendBrass,
    disposables
  );

  // 1G. Lower Detent Pawl Pivot Post, Spring Pin & Anti-Rotation Backup Pin
  const lPawlPostGeo = makeClosedLatheZ(
    [
      [-0.182, 0.015],
      [0.154, 0.015],
      [0.158, 0.022],
      [0.166, 0.017],
    ],
    22
  );
  lPawlPostGeo.translate(LOWER_DETENT_CAM_SPEC.pivotX, LOWER_DETENT_CAM_SPEC.pivotY, 0);
  disposables.push(lPawlPostGeo);
  const lPawlPostMesh = new THREE.Mesh(lPawlPostGeo, materials.hardenedSteel);
  lPawlPostMesh.name = 'LowerPawlPivotPost';
  lPawlPostMesh.castShadow = true;
  staticHardware.add(lPawlPostMesh);
  addBearingCollar(
    staticHardware,
    'LowerPawlStandoff',
    LOWER_DETENT_CAM_SPEC.pivotX,
    LOWER_DETENT_CAM_SPEC.pivotY,
    -0.180,
    LOWER_DETENT_CAM_SPEC.zBack,
    0.016,
    0.027,
    materials.forendBrass,
    disposables
  );

  const lSpringPinGeo = makeClosedLatheZ(
    [
      [-0.182, 0.014],
      [0.146, 0.014],
      [0.151, 0.020],
      [0.158, 0.015],
    ],
    20
  );
  lSpringPinGeo.translate(LOWER_DETENT_CAM_SPEC.springPinX, LOWER_DETENT_CAM_SPEC.springPinY, 0);
  disposables.push(lSpringPinGeo);
  const lSpringPinMesh = new THREE.Mesh(lSpringPinGeo, materials.forendBrass);
  lSpringPinMesh.name = 'LowerDetentSpringPin';
  lSpringPinMesh.castShadow = true;
  staticHardware.add(lSpringPinMesh);
  addBearingCollar(
    staticHardware,
    'LowerDetentSpringCollar',
    LOWER_DETENT_CAM_SPEC.springPinX,
    LOWER_DETENT_CAM_SPEC.springPinY,
    (LOWER_DETENT_CAM_SPEC.zBack + LOWER_DETENT_CAM_SPEC.zFront) * 0.5 - DETENT_SPRING_HALF_Z,
    (LOWER_DETENT_CAM_SPEC.zBack + LOWER_DETENT_CAM_SPEC.zFront) * 0.5 + DETENT_SPRING_HALF_Z,
    0.0145,
    0.023,
    materials.phosphorBronze,
    disposables
  );

  const lBackPinGeo = makeClosedLatheZ(
    [
      [-0.182, 0.008],
      [0.144, 0.008],
      [0.150, 0.005],
    ],
    16
  );
  lBackPinGeo.translate(
    LOWER_DETENT_CAM_SPEC.springPinX + 0.024,
    LOWER_DETENT_CAM_SPEC.springPinY - 0.016,
    0
  );
  disposables.push(lBackPinGeo);
  const lBackPinMesh = new THREE.Mesh(lBackPinGeo, materials.hardenedSteel);
  lBackPinMesh.name = 'LowerDetentReactionPin';
  staticHardware.add(lBackPinMesh);

  // ===========================================================================
  // PART 2: ROTATING TRAIN A (UPPER RACK-AND-PINION + COMPOUND DETENT ESCAPEMENT)
  // ===========================================================================

  const upperArborU1Group = new THREE.Group();
  upperArborU1Group.name = 'UpperCompoundArbor_U1';
  upperArborU1Group.position.set(ARBOR_U1_X, ARBOR_U1_Y, 0);
  group.add(upperArborU1Group);

  const u1aGeo = createInvoluteGearGeometry(GEAR_U1A_PINION, { boreRadius: 0.023 });
  u1aGeo.translate(0, 0, (GEAR_U1A_PINION.zBack + GEAR_U1A_PINION.zFront) * 0.5);
  disposables.push(u1aGeo);
  const u1aMesh = new THREE.Mesh(u1aGeo, materials.hardenedSteel);
  u1aMesh.name = GEAR_U1A_PINION.name;
  u1aMesh.castShadow = true;
  u1aMesh.receiveShadow = true;
  upperArborU1Group.add(u1aMesh);

  const u1bGeo = createInvoluteGearGeometry(GEAR_U1B_SPUR, { boreRadius: 0.023, spokeCount: 5 });
  u1bGeo.translate(0, 0, (GEAR_U1B_SPUR.zBack + GEAR_U1B_SPUR.zFront) * 0.5);
  disposables.push(u1bGeo);
  const u1bMesh = new THREE.Mesh(u1bGeo, materials.machinedGearBrass);
  u1bMesh.name = GEAR_U1B_SPUR.name;
  u1bMesh.castShadow = true;
  u1bMesh.receiveShadow = true;
  upperArborU1Group.add(u1bMesh);

  addBearingCollar(
    upperArborU1Group,
    'CouplingSleeve_U1',
    0,
    0,
    GEAR_U1A_PINION.zFront,
    GEAR_U1B_SPUR.zBack,
    0.023,
    0.035,
    materials.machinedBronzeRack,
    disposables
  );

  const upperArborU2Group = new THREE.Group();
  upperArborU2Group.name = 'UpperDetentArbor_U2';
  upperArborU2Group.position.set(ARBOR_U2_X, ARBOR_U2_Y, 0);
  group.add(upperArborU2Group);

  const u2aGeo = createInvoluteGearGeometry(GEAR_U2A_DRIVEN, { boreRadius: 0.025, spokeCount: 5 });
  u2aGeo.translate(0, 0, (GEAR_U2A_DRIVEN.zBack + GEAR_U2A_DRIVEN.zFront) * 0.5);
  disposables.push(u2aGeo);
  const u2aMesh = new THREE.Mesh(u2aGeo, materials.machinedGearBrass);
  u2aMesh.name = GEAR_U2A_DRIVEN.name;
  u2aMesh.castShadow = true;
  u2aMesh.receiveShadow = true;
  upperArborU2Group.add(u2aMesh);

  addBearingCollar(
    upperArborU2Group,
    'DriveSleeve_U2',
    0,
    0,
    GEAR_U2A_DRIVEN.zFront,
    UPPER_DETENT_CAM_SPEC.zBack,
    0.025,
    0.038,
    materials.machinedGearBrass,
    disposables
  );

  const u2bGeo = createDetentCamGeometry(UPPER_DETENT_CAM_SPEC, 0.025);
  u2bGeo.translate(0, 0, (UPPER_DETENT_CAM_SPEC.zBack + UPPER_DETENT_CAM_SPEC.zFront) * 0.5);
  disposables.push(u2bGeo);
  const u2bMesh = new THREE.Mesh(u2bGeo, materials.bluedSteelBolt);
  u2bMesh.name = 'UpperDetentEscapementCam_U2B';
  u2bMesh.castShadow = true;
  u2bMesh.receiveShadow = true;
  upperArborU2Group.add(u2bMesh);

  addBearingCollar(
    upperArborU2Group,
    'HubOilSink_U2B',
    0,
    0,
    UPPER_DETENT_CAM_SPEC.zFront,
    UPPER_DETENT_CAM_SPEC.zFront + 0.009,
    0.025,
    0.038,
    materials.machinedGearBrass,
    disposables
  );
  const u2StudGeo = makeClosedLatheZ(
    [
      [UPPER_DETENT_CAM_SPEC.zFront - 0.002, 0.010],
      [UPPER_DETENT_CAM_SPEC.zFront + 0.011, 0.010],
      [UPPER_DETENT_CAM_SPEC.zFront + 0.016, 0.0065],
    ],
    18
  );
  u2StudGeo.translate(0.052, 0, 0);
  disposables.push(u2StudGeo);
  const u2StudMesh = new THREE.Mesh(u2StudGeo, materials.machinedGearBrass);
  u2StudMesh.name = 'ThrowIndexPin_U2B';
  u2StudMesh.castShadow = true;
  upperArborU2Group.add(u2StudMesh);

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
  // PART 3: LOAD-BEARING 3:1 COAXIAL THUMB-TURN TRANSMISSION & LOWER DETENT
  // ===========================================================================

  // 3A. Primary 28T Spoked Brass Knob Drive Wheel (B1, front Deck 2: z 0.162..0.178)
  // Open 5-spoke design lets viewers see C1 spinning 3x faster right behind it!
  const knobDriveB1Group = new THREE.Group();
  knobDriveB1Group.name = 'ThumbTurnDriveWheelGroup_B1';
  knobDriveB1Group.position.set(KEY_X, KEY_Y, 0);
  group.add(knobDriveB1Group);

  const b1Geo = createInvoluteGearGeometry(GEAR_B1_KNOB_DRIVE, {
    boreRadius: 0.048,
    spokeCount: 5,
  });
  b1Geo.translate(0, 0, (GEAR_B1_KNOB_DRIVE.zBack + GEAR_B1_KNOB_DRIVE.zFront) * 0.5);
  disposables.push(b1Geo);
  const b1Mesh = new THREE.Mesh(b1Geo, materials.machinedGearBrass);
  b1Mesh.name = GEAR_B1_KNOB_DRIVE.name;
  b1Mesh.castShadow = true;
  b1Mesh.receiveShadow = true;
  knobDriveB1Group.add(b1Mesh);

  // 3B. Inner Coaxial 16T Hardened-Bronze Cam-Barrel Driven Pinion (C1, Deck 1: z 0.138..0.155, turns 3x faster!)
  const camDrivenC1Group = new THREE.Group();
  camDrivenC1Group.name = 'CamBarrelDrivenPinionGroup_C1';
  camDrivenC1Group.position.set(KEY_X, KEY_Y, 0);
  group.add(camDrivenC1Group);

  const c1Geo = createInvoluteGearGeometry(GEAR_C1_CAM_DRIVEN, {
    boreRadius: 0.032,
    drillHoles: 5,
  });
  c1Geo.translate(0, 0, (GEAR_C1_CAM_DRIVEN.zBack + GEAR_C1_CAM_DRIVEN.zFront) * 0.5);
  disposables.push(c1Geo);
  const c1Mesh = new THREE.Mesh(c1Geo, materials.machinedBronzeRack);
  c1Mesh.name = GEAR_C1_CAM_DRIVEN.name;
  c1Mesh.castShadow = true;
  c1Mesh.receiveShadow = true;
  camDrivenC1Group.add(c1Mesh);

  // 3C. Compound 3:1 Step-Up Countershaft B2 (B2a 14T steel pinion on Deck 2 + B2b 24T spoked brass wheel on Deck 1)
  const stepUpB2Group = new THREE.Group();
  stepUpB2Group.name = 'StepUpCountershaftGroup_B2';
  stepUpB2Group.position.set(ARBOR_B2_X, ARBOR_B2_Y, 0);
  group.add(stepUpB2Group);

  const b2aGeo = createInvoluteGearGeometry(GEAR_B2A_PINION, { boreRadius: 0.021 });
  b2aGeo.translate(0, 0, (GEAR_B2A_PINION.zBack + GEAR_B2A_PINION.zFront) * 0.5);
  disposables.push(b2aGeo);
  const b2aMesh = new THREE.Mesh(b2aGeo, materials.hardenedSteel);
  b2aMesh.name = GEAR_B2A_PINION.name;
  b2aMesh.castShadow = true;
  b2aMesh.receiveShadow = true;
  stepUpB2Group.add(b2aMesh);

  const b2bGeo = createInvoluteGearGeometry(GEAR_B2B_STEP, { boreRadius: 0.021, spokeCount: 5 });
  b2bGeo.translate(0, 0, (GEAR_B2B_STEP.zBack + GEAR_B2B_STEP.zFront) * 0.5);
  disposables.push(b2bGeo);
  const b2bMesh = new THREE.Mesh(b2bGeo, materials.machinedGearBrass);
  b2bMesh.name = GEAR_B2B_STEP.name;
  b2bMesh.castShadow = true;
  b2bMesh.receiveShadow = true;
  stepUpB2Group.add(b2bMesh);

  addBearingCollar(
    stepUpB2Group,
    'CouplingSleeve_B2',
    0,
    0,
    GEAR_B2B_STEP.zFront,
    GEAR_B2A_PINION.zBack,
    0.021,
    0.032,
    materials.machinedBronzeRack,
    disposables
  );

  // 3D. Compound Detent Transfer Arbor B3 (B3f 16T front brass gear + B3r 14T rear drilled bronze pinion)
  const transferB3Group = new THREE.Group();
  transferB3Group.name = 'DetentTransferArbor_B3';
  transferB3Group.position.set(ARBOR_B3_X, ARBOR_B3_Y, 0);
  group.add(transferB3Group);

  const b3fGeo = createInvoluteGearGeometry(GEAR_B3F_TRANSFER, { boreRadius: 0.021, drillHoles: 5 });
  b3fGeo.translate(0, 0, (GEAR_B3F_TRANSFER.zBack + GEAR_B3F_TRANSFER.zFront) * 0.5);
  disposables.push(b3fGeo);
  const b3fMesh = new THREE.Mesh(b3fGeo, materials.machinedGearBrass);
  b3fMesh.name = GEAR_B3F_TRANSFER.name;
  b3fMesh.castShadow = true;
  b3fMesh.receiveShadow = true;
  transferB3Group.add(b3fMesh);

  const b3rGeo = createInvoluteGearGeometry(GEAR_B3R_PINION, { boreRadius: 0.021, drillHoles: 4 });
  b3rGeo.translate(0, 0, (GEAR_B3R_PINION.zBack + GEAR_B3R_PINION.zFront) * 0.5);
  disposables.push(b3rGeo);
  const b3rMesh = new THREE.Mesh(b3rGeo, materials.machinedBronzeRack);
  b3rMesh.name = GEAR_B3R_PINION.name;
  b3rMesh.castShadow = true;
  b3rMesh.receiveShadow = true;
  transferB3Group.add(b3rMesh);

  addBearingCollar(
    transferB3Group,
    'CouplingSleeve_B3',
    0,
    0,
    GEAR_B3R_PINION.zFront,
    GEAR_B3F_TRANSFER.zBack,
    0.021,
    0.033,
    materials.hardenedSteel,
    disposables
  );

  // 3E. Grand 5-Spoked Lower Dual-Notch Detent Wheel (B4A + Blued-Steel Cam B4B)
  const lowerDetentB4Group = new THREE.Group();
  lowerDetentB4Group.name = 'LowerDetentArbor_B4';
  lowerDetentB4Group.position.set(ARBOR_B4_X, ARBOR_B4_Y, 0);
  group.add(lowerDetentB4Group);

  const b4aGeo = createInvoluteGearGeometry(GEAR_B4A_DETENT, { boreRadius: 0.025, spokeCount: 5 });
  b4aGeo.translate(0, 0, (GEAR_B4A_DETENT.zBack + GEAR_B4A_DETENT.zFront) * 0.5);
  disposables.push(b4aGeo);
  const b4aMesh = new THREE.Mesh(b4aGeo, materials.machinedGearBrass);
  b4aMesh.name = GEAR_B4A_DETENT.name;
  b4aMesh.castShadow = true;
  b4aMesh.receiveShadow = true;
  lowerDetentB4Group.add(b4aMesh);

  addBearingCollar(
    lowerDetentB4Group,
    'DriveSpacer_B4',
    0,
    0,
    GEAR_B4A_DETENT.zFront,
    LOWER_DETENT_CAM_SPEC.zBack,
    0.025,
    0.040,
    materials.machinedBronzeRack,
    disposables
  );

  const b4bGeo = createDetentCamGeometry(LOWER_DETENT_CAM_SPEC, 0.025);
  b4bGeo.translate(0, 0, (LOWER_DETENT_CAM_SPEC.zBack + LOWER_DETENT_CAM_SPEC.zFront) * 0.5);
  disposables.push(b4bGeo);
  const b4bMesh = new THREE.Mesh(b4bGeo, materials.bluedSteelBolt);
  b4bMesh.name = 'LowerDetentEscapementCam_B4B';
  b4bMesh.castShadow = true;
  b4bMesh.receiveShadow = true;
  lowerDetentB4Group.add(b4bMesh);

  addBearingCollar(
    lowerDetentB4Group,
    'HubOilSink_B4B',
    0,
    0,
    LOWER_DETENT_CAM_SPEC.zFront,
    LOWER_DETENT_CAM_SPEC.zFront + 0.008,
    0.025,
    0.038,
    materials.machinedGearBrass,
    disposables
  );
  const b4StudGeo = makeClosedLatheZ(
    [
      [LOWER_DETENT_CAM_SPEC.zFront - 0.002, 0.010],
      [LOWER_DETENT_CAM_SPEC.zFront + 0.010, 0.010],
      [LOWER_DETENT_CAM_SPEC.zFront + 0.015, 0.0065],
    ],
    18
  );
  b4StudGeo.translate(0.052, 0, 0);
  disposables.push(b4StudGeo);
  const b4StudMesh = new THREE.Mesh(b4StudGeo, materials.machinedGearBrass);
  b4StudMesh.name = 'LatchPhaseIndexPin_B4B';
  b4StudMesh.castShadow = true;
  lowerDetentB4Group.add(b4StudMesh);

  // 3F. Lower Dual-Notch Pivoting Detent Pawl + Flexing Phosphor-Bronze Click Spring
  const lowerPawlGroup = new THREE.Group();
  lowerPawlGroup.name = 'LowerDetentPawlGroup';
  lowerPawlGroup.position.set(
    LOWER_DETENT_CAM_SPEC.pivotX,
    LOWER_DETENT_CAM_SPEC.pivotY,
    (LOWER_DETENT_CAM_SPEC.zBack + LOWER_DETENT_CAM_SPEC.zFront) * 0.5
  );
  group.add(lowerPawlGroup);

  const lowerPawlGeo = createDetentPawlGeometry(LOWER_DETENT_CAM_SPEC);
  disposables.push(lowerPawlGeo);
  const lowerPawlMesh = new THREE.Mesh(lowerPawlGeo, materials.hardenedSteel);
  lowerPawlMesh.name = 'LowerDetentPawl';
  lowerPawlMesh.castShadow = true;
  lowerPawlMesh.receiveShadow = true;
  lowerPawlGroup.add(lowerPawlMesh);

  const lowerDetentSpring = new DetentSpringMesh(
    'LowerDetentLeafSpring',
    LOWER_DETENT_CAM_SPEC,
    materials.phosphorBronze
  );
  group.add(lowerDetentSpring.mesh);

  return {
    group,
    upperArborU1Group,
    upperArborU2Group,
    upperPawlGroup,
    upperDetentSpring,
    knobDriveB1Group,
    stepUpB2Group,
    camDrivenC1Group,
    transferB3Group,
    lowerDetentB4Group,
    lowerPawlGroup,
    lowerDetentSpring,
    update: (u1Rad, u2Rad, upperPawlRad, b1Rad, b2Rad, c1Rad, b3Rad, b4Rad, lowerPawlRad) => {
      upperArborU1Group.rotation.z = u1Rad;
      upperArborU2Group.rotation.z = u2Rad;
      upperPawlGroup.rotation.z = upperPawlRad;
      upperDetentSpring.update(upperPawlRad);

      knobDriveB1Group.rotation.z = b1Rad;
      stepUpB2Group.rotation.z = b2Rad;
      camDrivenC1Group.rotation.z = c1Rad;
      transferB3Group.rotation.z = b3Rad;
      lowerDetentB4Group.rotation.z = b4Rad;
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
