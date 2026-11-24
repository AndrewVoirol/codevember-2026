import * as THREE from 'three/webgpu';
import {
  BEAKER_BOTTOM_OUTER_Y,
  BEAKER_INNER_FLOOR_Y,
  BEAKER_INNER_RADIUS,
  BEAKER_OUTER_RADIUS,
  BEAKER_TOP_Y,
  WATER_RADIUS,
  WATER_REST_Y,
  type BeakerDissolveSim,
} from './dissolveSim';

/**
 * Day 30 — Dissolve:
 * Procedural Watertight Laboratory Glassware, Deformable Water Volume, Etched Markings,
 * High-Subdivision Crystal Cube, and Borosilicate Stirring Rod Geometries.
 *
 * All closed solids guarantee `signedVolume > 0` and outward exterior winding (`inwardTri === 0`).
 */

export const SPOUT_AZIMUTH_RAD = 0.56;

// ---------------------------------------------------------------------------
// 1. Thick-Walled Borosilicate Laboratory Beaker with Flared Pouring Spout
// ---------------------------------------------------------------------------
interface BeakerProfileNode {
  r: number;
  y: number;
  isExterior: number;
  rimBead: number;
  basePuck: number;
  thicknessNorm: number;
}

function buildBeakerProfile(): BeakerProfileNode[] {
  const pts: BeakerProfileNode[] = [];
  const rOut = BEAKER_OUTER_RADIUS;
  const rIn = BEAKER_INNER_RADIUS;
  const yBot = BEAKER_BOTTOM_OUTER_Y;
  const yFloor = BEAKER_INNER_FLOOR_Y;
  const yTop = BEAKER_TOP_Y;

  // 1. Exterior underside punt & thick base puck (strictly bottom-to-top -> outward winding!)
  pts.push({ r: 0.0, y: yBot + 0.011, isExterior: 1, rimBead: 0, basePuck: 1, thicknessNorm: 1 });
  pts.push({ r: 0.24, y: yBot + 0.009, isExterior: 1, rimBead: 0, basePuck: 1, thicknessNorm: 1 });
  pts.push({ r: 0.50, y: yBot + 0.004, isExterior: 1, rimBead: 0, basePuck: 1, thicknessNorm: 1 });
  pts.push({ r: rOut - 0.068, y: yBot, isExterior: 1, rimBead: 0, basePuck: 1, thicknessNorm: 1 });

  // 2. Smooth quarter-round outer bottom heel bevel
  const heelR = 0.068;
  const heelSegs = 6;
  for (let i = 1; i <= heelSegs; i++) {
    const a = (i / heelSegs) * (Math.PI * 0.5);
    const r = rOut - heelR + Math.sin(a) * heelR;
    const y = yBot + heelR - Math.cos(a) * heelR;
    pts.push({
      r,
      y,
      isExterior: 1,
      rimBead: 0,
      basePuck: 1.0 - (i / heelSegs) * 0.25,
      thicknessNorm: 0.95 - (i / heelSegs) * 0.35,
    });
  }

  // 3. Crisp vertical cylindrical outer borosilicate wall
  const wallY0 = yBot + heelR;
  const wallY1 = yTop - 0.085;
  const wallSegs = 18;
  for (let i = 1; i <= wallSegs; i++) {
    const t = i / wallSegs;
    const y = wallY0 + (wallY1 - wallY0) * t;
    const baseBlend = Math.max(0, 1.0 - (y - yBot) / (yFloor - yBot + 0.075));
    pts.push({
      r: rOut,
      y,
      isExterior: 1,
      rimBead: Math.max(0, (t - 0.85) / 0.15) * 0.25,
      basePuck: baseBlend * baseBlend,
      thicknessNorm: 0.32 + baseBlend * 0.68,
    });
  }

  // 4. Gently flared upper collar & fire-polished rolled laboratory rim bead
  const flareSteps = 5;
  const rRimOut = rOut + 0.036;
  const rRimIn = rIn + 0.022;
  const lipCenterR = 0.5 * (rRimOut + rRimIn);
  const lipHalfW = 0.5 * (rRimOut - rRimIn); // ~0.028

  for (let i = 1; i <= flareSteps; i++) {
    const t = i / flareSteps;
    const smooth = t * t * (3.0 - 2.0 * t);
    const y = wallY1 + (yTop - lipHalfW - wallY1) * t;
    const r = rOut + (rRimOut - rOut) * smooth;
    pts.push({
      r,
      y,
      isExterior: 1,
      rimBead: 0.25 + 0.65 * smooth,
      basePuck: 0,
      thicknessNorm: 0.35 + 0.45 * smooth,
    });
  }

  // 180-degree rounded fire-polished rolled lip turnaround (outer -> crest -> inner)
  const lipSegs = 8;
  for (let i = 1; i < lipSegs; i++) {
    const a = (i / lipSegs) * Math.PI;
    const r = lipCenterR + Math.cos(a) * lipHalfW;
    const y = yTop - lipHalfW + Math.sin(a) * (lipHalfW * 1.12);
    pts.push({
      r,
      y,
      isExterior: a <= Math.PI * 0.55 ? 1 : 0,
      rimBead: 1.0,
      basePuck: 0,
      thicknessNorm: 0.92,
    });
  }

  // 5. Inner upper flare returning smoothly into vertical inner wall (top-to-bottom)
  for (let i = flareSteps; i >= 1; i--) {
    const t = i / flareSteps;
    const smooth = t * t * (3.0 - 2.0 * t);
    const y = wallY1 + (yTop - lipHalfW - wallY1) * t;
    const r = rIn + (rRimIn - rIn) * smooth;
    pts.push({
      r,
      y,
      isExterior: 0,
      rimBead: 0.25 + 0.65 * smooth,
      basePuck: 0,
      thicknessNorm: 0.35 + 0.45 * smooth,
    });
  }

  // 6. Inner cylindrical wall descending top-to-bottom
  const innerFilletR = 0.048;
  const innerWallY0 = yFloor + innerFilletR;
  for (let i = wallSegs; i >= 0; i--) {
    const t = i / wallSegs;
    const y = innerWallY0 + (wallY1 - innerWallY0) * t;
    const baseBlend = Math.max(0, 1.0 - (y - yFloor) / 0.14);
    pts.push({
      r: rIn,
      y,
      isExterior: 0,
      rimBead: 0,
      basePuck: baseBlend * baseBlend,
      thicknessNorm: 0.32 + baseBlend * 0.68,
    });
  }

  // 7. Smooth inner floor corner fillet descending into inner glass floor
  const inFilletSegs = 5;
  for (let i = 1; i <= inFilletSegs; i++) {
    const a = (i / inFilletSegs) * (Math.PI * 0.5);
    const r = rIn - innerFilletR + Math.cos(a) * innerFilletR;
    const y = yFloor + innerFilletR - Math.sin(a) * innerFilletR;
    pts.push({
      r,
      y,
      isExterior: 0,
      rimBead: 0,
      basePuck: 1.0,
      thicknessNorm: 0.96,
    });
  }

  // 8. Flat thick borosilicate interior floor closing at axis
  pts.push({ r: rIn * 0.52, y: yFloor, isExterior: 0, rimBead: 0, basePuck: 1, thicknessNorm: 1 });
  pts.push({ r: 0.0, y: yFloor, isExterior: 0, rimBead: 0, basePuck: 1, thicknessNorm: 1 });

  return pts;
}

/**
 * Builds the thick-walled borosilicate glass beaker with a flared pouring spout lip,
 * thick refracting base puck, and seam-welded outward normals (`signedVolume > 0`).
 *
 * Vertex attribute `aGlassMeta` (`vec4`):
 *   x = isExterior [0..1]
 *   y = rimBeadMask [0..1]
 *   z = basePuckMask [0..1]
 *   w = thicknessNorm [0..1]
 */
export function buildBorosilicateBeakerGeometry(radialSegs = 96): THREE.BufferGeometry {
  const profile = buildBeakerProfile();
  const lathePts = profile.map((p) => new THREE.Vector2(p.r, p.y));
  const geom = new THREE.LatheGeometry(lathePts, radialSegs);

  const posAttr = geom.attributes.position as THREE.BufferAttribute;
  const vertCount = posAttr.count;
  const meta = new Float32Array(vertCount * 4);
  const pLen = profile.length;

  // Apply smooth C2 localized pouring spout flare at the upper rim around SPOUT_AZIMUTH_RAD
  // preserving uniform glass wall thickness and strict manifold continuity across phi = 0 / 2*PI.
  for (let iSeg = 0; iSeg <= radialSegs; iSeg++) {
    const phi = (iSeg / radialSegs) * Math.PI * 2.0;
    // Shortest signed angular distance to pouring spout centerline
    let dAng = phi - SPOUT_AZIMUTH_RAD;
    while (dAng > Math.PI) dAng -= Math.PI * 2.0;
    while (dAng < -Math.PI) dAng += Math.PI * 2.0;

    const spoutEnv = Math.exp(-(dAng * dAng) / (0.24 * 0.24));
    const spoutTipPinch = Math.exp(-(dAng * dAng) / (0.11 * 0.11));

    for (let j = 0; j < pLen; j++) {
      const vIdx = iSeg * pLen + j;
      const node = profile[j];

      const x0 = posAttr.getX(vIdx);
      const y0 = posAttr.getY(vIdx);
      const z0 = posAttr.getZ(vIdx);
      const horizR = Math.hypot(x0, z0);

      // Spout activates strictly on the upper rim collar (y > BEAKER_TOP_Y - 0.25)
      const yT = Math.max(0, Math.min(1, (y0 - (BEAKER_TOP_Y - 0.25)) / 0.25));
      const ySmooth = yT * yT * (3.0 - 2.0 * yT);

      if (horizR > 1e-4 && ySmooth > 1e-5 && spoutEnv > 1e-5) {
        const radialPush = 0.095 * ySmooth * ySmooth * spoutEnv;
        // Subtle dipped V-channel nose at pouring tip framed by lifted lip cheeks
        const lipLift = (0.022 * spoutEnv - 0.028 * spoutTipPinch) * ySmooth * ySmooth;
        const scale = (horizR + radialPush) / horizR;
        posAttr.setXYZ(vIdx, x0 * scale, y0 + lipLift, z0 * scale);
      }

      const mBase = vIdx * 4;
      meta[mBase] = node.isExterior;
      meta[mBase + 1] = Math.min(1.0, node.rimBead + spoutEnv * ySmooth * 0.35);
      meta[mBase + 2] = node.basePuck;
      meta[mBase + 3] = node.thicknessNorm;
    }
  }

  posAttr.needsUpdate = true;
  geom.setAttribute('aGlassMeta', new THREE.BufferAttribute(meta, 4));
  geom.computeVertexNormals();

  // Weld phi = 0 and phi = 2*PI seam normals so glass reflections wrap seamlessly 360 degrees
  const normAttr = geom.attributes.normal as THREE.BufferAttribute;
  for (let j = 0; j < pLen; j++) {
    const iStart = j;
    const iEnd = radialSegs * pLen + j;
    const nx = normAttr.getX(iStart) + normAttr.getX(iEnd);
    const ny = normAttr.getY(iStart) + normAttr.getY(iEnd);
    const nz = normAttr.getZ(iStart) + normAttr.getZ(iEnd);
    const len = Math.hypot(nx, ny, nz) || 1;
    normAttr.setXYZ(iStart, nx / len, ny / len, nz / len);
    normAttr.setXYZ(iEnd, nx / len, ny / len, nz / len);
  }
  normAttr.needsUpdate = true;

  return geom;
}

// ---------------------------------------------------------------------------
// 2. Watertight Deformable Cylindrical Water Volume (Shared Perimeter Vertices)
// ---------------------------------------------------------------------------
/**
 * Closed, airtight manifold cylindrical water volume whose top rippling meniscus
 * and vertical cylindrical wall literally share the exact perimeter waterline vertices
 * (zero detached floating disc, `signedVolume > 0`, `inwardTri === 0`).
 *
 * Vertex attribute `aWaterMeta` (`vec4`):
 *   x = isTopSurface [0..1]
 *   y = radialFrac [0..1]
 *   z = heightFrac [0 bottom .. 1 meniscus]
 *   w = capillaryRim [0..1]
 */
export class DeformableWaterCylinder {
  readonly geometry: THREE.BufferGeometry;
  readonly mesh: THREE.Mesh;

  private readonly radialSegs = 72;
  private readonly topRings = 16;
  private readonly sideRings = 12;

  private readonly positions: Float32Array;
  private readonly normals: Float32Array;
  private readonly waterMeta: Float32Array;
  private readonly cosTable: Float32Array;
  private readonly sinTable: Float32Array;
  private readonly rimHeights: Float32Array;

  private readonly totalRings: number;
  private readonly bottomCenterIdx: number;

  constructor(material: THREE.Material) {
    // Ring layout (every edge shared by exactly 2 triangles; closed genus-0 manifold!):
    //   idx 0                          : top center
    //   rings 0 .. topRings-1          : top meniscus concentric rings (ring topRings-1 == perimeter waterline!)
    //   rings topRings .. totalRings-1 : vertical wall + bottom floor fillet rings (starting directly from perimeter ring!)
    //   bottomCenterIdx                : bottom cap center
    this.totalRings = this.topRings + this.sideRings;
    const vertCount = 1 + this.totalRings * this.radialSegs + 1;
    this.bottomCenterIdx = vertCount - 1;

    this.positions = new Float32Array(vertCount * 3);
    this.normals = new Float32Array(vertCount * 3);
    this.waterMeta = new Float32Array(vertCount * 4);
    this.cosTable = new Float32Array(this.radialSegs);
    this.sinTable = new Float32Array(this.radialSegs);
    this.rimHeights = new Float32Array(this.radialSegs);

    for (let ir = 0; ir < this.radialSegs; ir++) {
      const theta = (ir / this.radialSegs) * Math.PI * 2.0;
      this.cosTable[ir] = Math.cos(theta);
      this.sinTable[ir] = Math.sin(theta);
    }

    const indices: number[] = [];
    const N = this.radialSegs;

    // Top pole fan (+Y outward normal: center, nextTheta, currTheta)
    for (let ir = 0; ir < N; ir++) {
      const next = (ir + 1) % N;
      indices.push(0, 1 + next, 1 + ir);
    }

    // Continuous quad strip from top meniscus ring 0 -> shared perimeter waterline -> vertical side wall -> bottom fillet
    for (let ring = 0; ring < this.totalRings - 1; ring++) {
      const r0 = 1 + ring * N;
      const r1 = 1 + (ring + 1) * N;
      for (let ir = 0; ir < N; ir++) {
        const next = (ir + 1) % N;
        const a = r0 + ir;
        const b = r0 + next;
        const c = r1 + ir;
        const d = r1 + next;
        // Outward winding (+Y on top cap, +radial on cylindrical wall, -Y on bottom bevel)
        indices.push(a, b, c, b, d, c);
      }
    }

    // Bottom pole fan (-Y outward normal: currTheta, nextTheta, bottomCenter)
    const lastRingBase = 1 + (this.totalRings - 1) * N;
    for (let ir = 0; ir < N; ir++) {
      const next = (ir + 1) % N;
      indices.push(lastRingBase + ir, lastRingBase + next, this.bottomCenterIdx);
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3));
    this.geometry.setAttribute('aWaterMeta', new THREE.BufferAttribute(this.waterMeta, 4));
    this.geometry.setIndex(indices);

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.name = 'WatertightWaterColumn';
    this.initStaticMetaAndRestPose();
  }

  private initStaticMetaAndRestPose(): void {
    const N = this.radialSegs;
    const botY = BEAKER_INNER_FLOOR_Y + 0.004;

    // Top center
    this.waterMeta[0] = 1.0;
    this.waterMeta[1] = 0.0;
    this.waterMeta[2] = 1.0;
    this.waterMeta[3] = 0.0;

    // Top meniscus rings (ring = 0 .. topRings - 1)
    for (let ring = 0; ring < this.topRings; ring++) {
      const frac = (ring + 1) / this.topRings;
      const capRim = Math.pow(frac, 6.0);
      for (let ir = 0; ir < N; ir++) {
        const vIdx = 1 + ring * N + ir;
        const m = vIdx * 4;
        this.waterMeta[m] = 1.0;
        this.waterMeta[m + 1] = frac;
        this.waterMeta[m + 2] = 1.0;
        this.waterMeta[m + 3] = capRim;
      }
    }

    // Side cylindrical wall + bottom fillet rings (ring = topRings .. totalRings - 1)
    for (let s = 0; s < this.sideRings; s++) {
      const ring = this.topRings + s;
      // s = 0 is a tight 6mm capillary meniscus shoulder right below the perimeter waterline
      const tDown = s === 0 ? 0.008 : s / (this.sideRings - 1);
      const heightFrac = Math.max(0.0, 1.0 - tDown);
      const capRim = s === 0 ? 0.88 : Math.pow(heightFrac, 14.0);
      const radialFrac = s === this.sideRings - 1 ? 0.94 : 1.0;
      for (let ir = 0; ir < N; ir++) {
        const vIdx = 1 + ring * N + ir;
        const m = vIdx * 4;
        this.waterMeta[m] = 0.0;
        this.waterMeta[m + 1] = radialFrac;
        this.waterMeta[m + 2] = heightFrac;
        this.waterMeta[m + 3] = capRim;
      }
    }

    // Bottom center
    const bPos = this.bottomCenterIdx * 3;
    this.positions[bPos] = 0;
    this.positions[bPos + 1] = botY;
    this.positions[bPos + 2] = 0;
    this.normals[bPos] = 0;
    this.normals[bPos + 1] = -1;
    this.normals[bPos + 2] = 0;
    const bMeta = this.bottomCenterIdx * 4;
    this.waterMeta[bMeta] = 0.0;
    this.waterMeta[bMeta + 1] = 0.0;
    this.waterMeta[bMeta + 2] = 0.0;
    this.waterMeta[bMeta + 3] = 0.0;

    // Seed flat resting pose so geometry is valid immediately even before first updateFromSim
    this.updateGeometryPositions((_x, _z) => 0);
  }

  updateFromSim(sim: BeakerDissolveSim): void {
    this.updateGeometryPositions((x, z) => sim.sampleMeniscusOffsetY(x, z));
  }

  private updateGeometryPositions(sampleOffsetY: (x: number, z: number) => number): void {
    const N = this.radialSegs;
    const R = WATER_RADIUS;
    const botY = BEAKER_INNER_FLOOR_Y + 0.004;
    const filletR = 0.042;
    const eps = 0.025;

    // 1. Top center vertex
    const hCenter = sampleOffsetY(0, 0);
    const dHx0 = (sampleOffsetY(eps, 0) - sampleOffsetY(-eps, 0)) / (2.0 * eps);
    const dHz0 = (sampleOffsetY(0, eps) - sampleOffsetY(0, -eps)) / (2.0 * eps);
    let nx0 = -dHx0;
    let ny0 = 1.0;
    let nz0 = -dHz0;
    const nLen0 = Math.hypot(nx0, ny0, nz0) || 1;
    this.positions[0] = 0;
    this.positions[1] = WATER_REST_Y + hCenter;
    this.positions[2] = 0;
    this.normals[0] = nx0 / nLen0;
    this.normals[1] = ny0 / nLen0;
    this.normals[2] = nz0 / nLen0;

    // 2. Top concentric meniscus rings (with concave capillary wall-wetting rise at glass wall)
    for (let ring = 0; ring < this.topRings; ring++) {
      const frac = (ring + 1) / this.topRings;
      const r = R * frac;
      const meniscusClimb = 0.024 * Math.pow(frac, 5.0);
      const climbSlope = (0.12 * Math.pow(frac, 4.0)) / R;
      const isPerimeter = ring === this.topRings - 1;

      for (let ir = 0; ir < N; ir++) {
        const c = this.cosTable[ir];
        const s = this.sinTable[ir];
        const x = c * r;
        const z = s * r;
        const hW = sampleOffsetY(x, z);
        const y = WATER_REST_Y + hW + meniscusClimb;

        if (isPerimeter) {
          this.rimHeights[ir] = y;
        }

        const vIdx = 1 + ring * N + ir;
        const p = vIdx * 3;
        this.positions[p] = x;
        this.positions[p + 1] = y;
        this.positions[p + 2] = z;

        const dHx = (sampleOffsetY(x + eps, z) - sampleOffsetY(x - eps, z)) / (2.0 * eps);
        const dHz = (sampleOffsetY(x, z + eps) - sampleOffsetY(x, z - eps)) / (2.0 * eps);
        let nx = -(dHx + c * climbSlope);
        let ny = 1.0;
        let nz = -(dHz + s * climbSlope);
        const nLen = Math.hypot(nx, ny, nz) || 1;
        this.normals[p] = nx / nLen;
        this.normals[p + 1] = ny / nLen;
        this.normals[p + 2] = nz / nLen;
      }
    }

    // 3. Cylindrical wetted wall + bottom fillet descending directly from shared perimeter ring
    for (let sIdx = 0; sIdx < this.sideRings; sIdx++) {
      const ring = this.topRings + sIdx;
      // First side ring sits just 6mm below the shared waterline to create a crisp capillary corner
      const isCornerRing = sIdx === 0;
      const isBottomFillet = sIdx >= this.sideRings - 2;

      for (let ir = 0; ir < N; ir++) {
        const c = this.cosTable[ir];
        const s = this.sinTable[ir];
        const yRim = this.rimHeights[ir];
        let r = R;
        let y: number;
        let nx = c;
        let ny = 0;
        let nz = s;

        if (isCornerRing) {
          y = yRim - 0.007;
          // Slightly upward-angled normal right at the capillary shoulder
          nx = c * 0.97;
          ny = 0.24;
          nz = s * 0.97;
        } else if (isBottomFillet) {
          const fT = sIdx === this.sideRings - 2 ? 0.5 : 1.0;
          const ang = fT * Math.PI * 0.5;
          r = R - filletR + Math.cos(ang) * filletR;
          y = botY + filletR - Math.sin(ang) * filletR;
          nx = c * Math.cos(ang);
          ny = -Math.sin(ang);
          nz = s * Math.cos(ang);
        } else {
          const wallFrac = (sIdx - 1) / Math.max(1, this.sideRings - 3);
          const yTopWall = yRim - 0.025;
          const yBotWall = botY + filletR;
          y = yTopWall + (yBotWall - yTopWall) * wallFrac;
        }

        const vIdx = 1 + ring * N + ir;
        const p = vIdx * 3;
        this.positions[p] = c * r;
        this.positions[p + 1] = y;
        this.positions[p + 2] = s * r;
        this.normals[p] = nx;
        this.normals[p + 1] = ny;
        this.normals[p + 2] = nz;
      }
    }

    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.normal as THREE.BufferAttribute).needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

// ---------------------------------------------------------------------------
// 3. Graduated Etched Eggshell Volumetric Scale Markings (500 mL Lab Beaker)
// ---------------------------------------------------------------------------
const DIGIT_GLYPHS: Readonly<Record<string, readonly string[]>> = {
  '0': ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  '1': ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  '2': ['01110', '10001', '00001', '00110', '01000', '10000', '11111'],
  '3': ['01110', '10001', '00001', '00110', '00001', '10001', '01110'],
  '4': ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  '5': ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  m: ['00000', '00000', '11010', '10101', '10101', '10101', '10001'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
  B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  '.': ['00000', '00000', '00000', '00000', '00000', '01100', '01100'],
  ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
};

/**
 * Generates a high-resolution 1024x1024 etched fired-enamel volumetric graduation texture
 * (50..500 mL major/minor ticks, fine vertical spine, Swiss laboratory sans-serif numerals,
 * "500 mL", "APPROX. VOL." and "BORO 3.3" mark). Uses crisp anti-aliased Canvas2D typography
 * in browsers with a deterministic DataTexture bitmap fallback in headless Node/Vitest.
 */
export function createGraduatedMarkingsTexture(): THREE.DataTexture {
  const W = 1024;
  const H = 1024;
  const data = new Uint8Array(W * H * 4);

  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#f3efe6';
      ctx.strokeStyle = '#f3efe6';
      ctx.textBaseline = 'middle';

      // Canvas Y=0 is top (corresponds to v=1 after flipY copy below)
      const toCanvasY = (texY: number) => H - texY;
      const spineX = 532;

      // Crisp thin laboratory calibration spine
      ctx.globalAlpha = 0.72;
      ctx.fillRect(spineX - 2, toCanvasY(888), 4, 772);

      // Volumetric ticks: 50 mL to 500 mL in 25 mL increments
      const yBase = 120;
      const ySpan = 764;
      ctx.font = '500 50px "Helvetica Neue", "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'left';

      for (let ml = 50; ml <= 500; ml += 25) {
        const frac = (ml - 50) / 450;
        const ty = yBase + frac * ySpan;
        const cy = toCanvasY(ty);
        const isMajor = ml % 100 === 0;
        const isMid = ml % 50 === 0 && !isMajor;

        const leftW = isMajor ? 112 : isMid ? 68 : 38;
        const rightW = isMajor ? 104 : isMid ? 56 : 32;
        const thick = isMajor ? 5.0 : isMid ? 3.4 : 2.4;

        ctx.globalAlpha = isMajor ? 0.96 : isMid ? 0.80 : 0.64;
        ctx.fillRect(spineX - leftW, cy - thick * 0.5, leftW + rightW, thick);

        if (isMajor) {
          ctx.globalAlpha = 0.94;
          ctx.fillText(String(ml), spineX + rightW + 26, cy + 2);
        }
      }

      // Authentic Pyrex / borosilicate 3.3 laboratory markings on left flank
      ctx.textAlign = 'left';
      ctx.globalAlpha = 0.92;
      ctx.font = '600 66px "Helvetica Neue", "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText('500 mL', 86, toCanvasY(844));

      ctx.globalAlpha = 0.74;
      ctx.font = '500 32px "Helvetica Neue", "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText('APPROX. VOL.', 88, toCanvasY(784));

      ctx.globalAlpha = 0.78;
      ctx.font = '600 34px "Helvetica Neue", "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText('BORO 3.3', 88, toCanvasY(734));

      // Authentic delicate Pyrex hairline writing-patch frame with subtle ruled lines (zero solid grey block)
      const patchX = 92;
      const patchY = toCanvasY(600);
      const patchW = 204;
      const patchH = 148;
      ctx.globalAlpha = 0.032;
      ctx.fillRect(patchX, patchY, patchW, patchH);
      ctx.globalAlpha = 0.34;
      ctx.lineWidth = 2.6;
      ctx.strokeRect(patchX, patchY, patchW, patchH);
      ctx.globalAlpha = 0.16;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(patchX + 18, patchY + patchH * 0.36);
      ctx.lineTo(patchX + patchW - 18, patchY + patchH * 0.36);
      ctx.moveTo(patchX + 18, patchY + patchH * 0.68);
      ctx.lineTo(patchX + patchW - 18, patchY + patchH * 0.68);
      ctx.stroke();

      const img = ctx.getImageData(0, 0, W, H).data;
      // Copy vertically flipped so v=0 maps to bottom of beaker (flipY = false)
      for (let row = 0; row < H; row++) {
        const srcRow = (H - 1 - row) * W * 4;
        const dstRow = row * W * 4;
        for (let col = 0; col < W * 4; col += 4) {
          const a = img[srcRow + col + 3];
          data[dstRow + col] = 243;
          data[dstRow + col + 1] = 239;
          data[dstRow + col + 2] = 230;
          data[dstRow + col + 3] = a;
        }
      }

      const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.flipY = false;
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.generateMipmaps = true;
      tex.needsUpdate = true;
      return tex;
    }
  }

  // Deterministic Node/Vitest fallback writing directly into the 1024x1024 DataTexture buffer
  const stampAlpha = (x: number, y: number, a: number) => {
    const ix = Math.round(x);
    const iy = Math.round(y);
    if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
    const idx = (iy * W + ix) * 4;
    const nextA = Math.min(255, Math.max(data[idx + 3], Math.round(a * 255)));
    data[idx] = 243;
    data[idx + 1] = 239;
    data[idx + 2] = 230;
    data[idx + 3] = nextA;
  };

  const drawRect = (x0: number, y0: number, x1: number, y1: number, alpha = 1.0) => {
    const minX = Math.max(0, Math.floor(Math.min(x0, x1)));
    const maxX = Math.min(W - 1, Math.ceil(Math.max(x0, x1)));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1)));
    const maxY = Math.min(H - 1, Math.ceil(Math.max(y0, y1)));
    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const dx = Math.min(px - x0, x1 - px);
        const dy = Math.min(py - y0, y1 - py);
        const edge = Math.max(0, Math.min(1, Math.min(dx, dy) + 0.65));
        stampAlpha(px, py, alpha * edge);
      }
    }
  };

  const drawText = (text: string, xStart: number, yBottom: number, scale: number, alpha = 0.92) => {
    let cursorX = xStart;
    for (let ci = 0; ci < text.length; ci++) {
      const glyph = DIGIT_GLYPHS[text[ci]];
      if (glyph) {
        for (let row = 0; row < 7; row++) {
          const bits = glyph[row];
          for (let col = 0; col < 5; col++) {
            if (bits[col] === '1') {
              const gy = yBottom + (6 - row) * scale;
              const gx = cursorX + col * scale;
              drawRect(gx, gy, gx + scale * 0.92, gy + scale * 0.92, alpha);
            }
          }
        }
      }
      cursorX += 6.4 * scale;
    }
  };

  const spineX = 532;
  drawRect(spineX - 2.0, 116, spineX + 2.0, 888, 0.72);
  const yBase = 120;
  const ySpan = 764;
  for (let ml = 50; ml <= 500; ml += 25) {
    const frac = (ml - 50) / 450;
    const y = yBase + frac * ySpan;
    const isMajor = ml % 100 === 0;
    const isMid = ml % 50 === 0 && !isMajor;
    const halfW = isMajor ? 108 : isMid ? 64 : 36;
    const halfH = isMajor ? 2.6 : 1.7;
    drawRect(spineX - halfW, y - halfH, spineX + halfW, y + halfH, isMajor ? 0.94 : 0.72);
    if (isMajor) {
      drawText(String(ml), spineX + halfW + 24, y - 16, 5.4, 0.90);
    }
  }
  drawText('500 mL', 112, 824, 5.8, 0.86);
  drawText('BORO 3.3', 112, 748, 4.0, 0.65);

  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Builds the curved outward-facing cylindrical decal strip carrying the etched volumetric markings.
 */
export function buildGraduatedMarkingsGeometry(): THREE.BufferGeometry {
  const radius = BEAKER_OUTER_RADIUS + 0.0024;
  const yMin = BEAKER_INNER_FLOOR_Y + 0.11;
  const yMax = BEAKER_TOP_Y - 0.15;
  // Centered cleanly on the front-left viewing face (-0.52 rad from +Z axis)
  const thetaCenter = -0.38;
  const thetaSpan = 1.05;
  const segsU = 36;
  const segsV = 8;

  const vertCount = (segsU + 1) * (segsV + 1);
  const pos = new Float32Array(vertCount * 3);
  const norm = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices: number[] = [];

  for (let iv = 0; iv <= segsV; iv++) {
    const v = iv / segsV;
    const y = yMin + (yMax - yMin) * v;
    for (let iu = 0; iu <= segsU; iu++) {
      const u = iu / segsU;
      // theta increases left-to-right across the front +Z hemisphere
      const theta = thetaCenter + (u - 0.5) * thetaSpan;
      const sx = Math.sin(theta);
      const cz = Math.cos(theta);
      const idx = iv * (segsU + 1) + iu;
      pos[idx * 3] = sx * radius;
      pos[idx * 3 + 1] = y;
      pos[idx * 3 + 2] = cz * radius;
      norm[idx * 3] = sx;
      norm[idx * 3 + 1] = 0;
      norm[idx * 3 + 2] = cz;
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  for (let iv = 0; iv < segsV; iv++) {
    for (let iu = 0; iu < segsU; iu++) {
      const a = iv * (segsU + 1) + iu;
      const b = a + 1;
      const c = a + (segsU + 1);
      const d = c + 1;
      // At front (+Z), +u is +X and +v is +Y -> (b-a) x (c-a) = +X x +Y = +Z (OUTWARD!)
      indices.push(a, b, d, a, d, c);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(norm, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  return geom;
}

// ---------------------------------------------------------------------------
// 4. Welded Edge-Clustered High-Subdivision Unit Cube (`[-1, 1]^3`)
// ---------------------------------------------------------------------------
/**
 * Builds a welded, watertight `[-1, 1]^3` subdivided cube with smooth edge clustering so
 * TSL curvature-weighted corner rounding & dissolution pitting deform with zero edge seams.
 */
export function buildSubdividedUnitCubeGeometry(segs = 22): THREE.BufferGeometry {
  const keyToIdx = new Map<string, number>();
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  // Gentle edge clustering concentrates subdivisions around corners & bevel fillets
  const cluster = (t01: number): number => {
    const s = t01 * 2.0 - 1.0;
    return s * (0.76 + 0.24 * s * s);
  };

  const getOrAddVertex = (x: number, y: number, z: number): number => {
    const key = `${Math.round(x * 10000)},${Math.round(y * 10000)},${Math.round(z * 10000)}`;
    const existing = keyToIdx.get(key);
    if (existing !== undefined) return existing;
    const idx = positions.length / 3;
    keyToIdx.set(key, idx);
    positions.push(x, y, z);
    const len = Math.hypot(x, y, z) || 1;
    normals.push(x / len, y / len, z / len);
    return idx;
  };

  // 6 faces defined by (origin normal, +u tangent, +v tangent) such that cross(u, v) === normal
  const faces: ReadonlyArray<{
    n: [number, number, number];
    u: [number, number, number];
    v: [number, number, number];
  }> = [
    { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
  ];

  const grid = new Int32Array((segs + 1) * (segs + 1));

  for (let f = 0; f < faces.length; f++) {
    const { n, u, v } = faces[f];
    for (let iv = 0; iv <= segs; iv++) {
      const cv = cluster(iv / segs);
      for (let iu = 0; iu <= segs; iu++) {
        const cu = cluster(iu / segs);
        const px = n[0] + u[0] * cu + v[0] * cv;
        const py = n[1] + u[1] * cu + v[1] * cv;
        const pz = n[2] + u[2] * cu + v[2] * cv;
        grid[iv * (segs + 1) + iu] = getOrAddVertex(px, py, pz);
      }
    }

    for (let iv = 0; iv < segs; iv++) {
      for (let iu = 0; iu < segs; iu++) {
        const a = grid[iv * (segs + 1) + iu];
        const b = grid[iv * (segs + 1) + iu + 1];
        const c = grid[(iv + 1) * (segs + 1) + iu];
        const d = grid[(iv + 1) * (segs + 1) + iu + 1];
        // cross(b - a, d - a) = cross(u, v) = +n (OUTWARD!)
        indices.push(a, b, d, a, d, c);
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geom.setIndex(indices);
  return geom;
}

// ---------------------------------------------------------------------------
// 5. Slender Borosilicate Glass Stirring Rod (Fire-Polished Domed Tips)
// ---------------------------------------------------------------------------
export function buildStirringRodGeometry(radius = 0.024, length = 1.92): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [];
  const domeSegs = 8;

  // Bottom fire-polished hemispherical tip (y = 0 .. radius)
  for (let i = 0; i <= domeSegs; i++) {
    const a = (i / domeSegs) * (Math.PI * 0.5);
    pts.push(new THREE.Vector2(Math.sin(a) * radius, radius - Math.cos(a) * radius));
  }

  // Slender straight borosilicate glass shaft
  const shaftSegs = 12;
  for (let i = 1; i < shaftSegs; i++) {
    const t = i / shaftSegs;
    pts.push(new THREE.Vector2(radius, radius + (length - 2.0 * radius) * t));
  }

  // Top fire-polished hemispherical tip (y = length - radius .. length)
  for (let i = 0; i <= domeSegs; i++) {
    const a = (i / domeSegs) * (Math.PI * 0.5);
    pts.push(
      new THREE.Vector2(Math.cos(a) * radius, length - radius + Math.sin(a) * radius)
    );
  }

  return new THREE.LatheGeometry(pts, 28);
}

// ---------------------------------------------------------------------------
// 6. Inner Glass Base Optical Caustic Disc (`+Y` Normal)
// ---------------------------------------------------------------------------
export function buildBaseCausticDiscGeometry(): THREE.BufferGeometry {
  const geom = new THREE.CircleGeometry(WATER_RADIUS * 0.985, 64);
  geom.rotateX(-Math.PI * 0.5);
  geom.translate(0, BEAKER_INNER_FLOOR_Y + 0.002, 0);
  return geom;
}

// ---------------------------------------------------------------------------
// 7. Faceted Cleaved Mineral Crumb / Crystal Chip Geometry (Outward Winding)
// ---------------------------------------------------------------------------
/**
 * Builds an angular, asymmetrically cleaved crystalline mineral grain (convex icosahedron
 * with subtle cleaved mineral facet stretching, `signedVolume > 0`, `inwardTri === 0`) for
 * solid crumbs detaching off the eroding brick boundary.
 */
export function buildMineralCrumbGeometry(): THREE.BufferGeometry {
  const raw = new THREE.IcosahedronGeometry(1.0, 0);
  const pos = raw.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    // Crisp crystalline cleavage facet irregularity (preserving strict convexity & outward winding)
    const cleavage = 0.84 + 0.24 * Math.abs(Math.sin(x * 3.7 + y * 2.1 - z * 2.9));
    pos.setXYZ(i, x * cleavage * 1.12, y * cleavage * 0.82, z * cleavage * 0.96);
  }
  pos.needsUpdate = true;
  raw.computeVertexNormals();
  return raw;
}
