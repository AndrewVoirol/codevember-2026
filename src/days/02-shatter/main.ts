import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three/webgpu';
import {
  attribute,
  clamp,
  dot,
  exp,
  float,
  max,
  mix,
  positionLocal,
  positionWorld,
  smoothstep,
  uint,
  uniform,
  uniformArray,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

const GEODE_RADIUS = 0.72;
const GEODE_HEIGHT = 1.48;
const HALF_HEIGHT = GEODE_HEIGHT * 0.5;
const SHARD_COUNT = 14;
const SUB_SHARDS_PER_SHARD = 4;
const STRAND_SEGMENTS = 5;
const SPALL_CHIP_COUNT = 48;

// Tighter initial fit so the specimen reads as a cohesive Kintsugi-veined core sample on boot
const INITIAL_SPREAD = 0.032;
// Longitudinal stress-wave propagation speed across the monolith (world units / sec)
const FRACTURE_WAVE_SPEED = 3.4;

interface VoronoiSite {
  seed: THREE.Vector3;
  radialDir: THREE.Vector3;
  tangentDir: THREE.Vector3;
}

interface SubShardRecord {
  parentIdx: number;
  subIdx: number;
  globalSubIdx: number;
  mesh: THREE.Mesh;
  body: RAPIER.RigidBody;
  localOffset: THREE.Vector3; // Offset from parent shard centroid in monolith space
  outwardDir: THREE.Vector3; // Secondary cleavage separation vector
  rotAxis: THREE.Vector3;
  restPos: THREE.Vector3;
  restQuat: THREE.Quaternion;
}

interface ShardCluster {
  index: number;
  homePos: THREE.Vector3;
  radialDir: THREE.Vector3;
  tangentDir: THREE.Vector3;
  isFrontWindow: boolean;
  subShards: SubShardRecord[];
  /** Individual Tier-1 opening state in [0..1] (driven by finite-speed fracture wavefront) */
  openAmount: number;
  targetOpen: number;
  waveDelaySec: number;
  pendingWaveTarget: number | null;
  /** Individual Tier-2 secondary shatter state in [0..1] */
  subShatter: number;
  targetSubShatter: number;
  /** Transient triboluminescent cleavage flash in [0..1] on fresh fracture */
  stressFlash: number;
  /** Subtle subsurface fault-line pre-glow when hovered in open state */
  hoverFaultGlow: number;
}

interface KintsugiBridge {
  subA: SubShardRecord;
  subB: SubShardRecord;
  localA: THREE.Vector3;
  localB: THREE.Vector3;
  sagDir: THREE.Vector3;
  restSpan: number;
  tier: 1 | 2;
  parentShardIdx: number;
}

interface SpallChip {
  parentShardIdx: number;
  localOrigin: THREE.Vector3;
  ejectDir: THREE.Vector3;
  rotAxis: THREE.Vector3;
  scale: THREE.Vector3;
  phase: number;
}

/**
 * Deterministic low-discrepancy Fibonacci + jittered mineral sites inside the solid cylinder.
 */
function generateMonolithSites(count: number): VoronoiSite[] {
  const sites: VoronoiSite[] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < count; i++) {
    const t = (i + 0.5) / count;
    const hash1 = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
    const j1 = hash1 - Math.floor(hash1) - 0.5;
    const hash2 = Math.sin(i * 45.164 + 19.13) * 19543.21;
    const j2 = hash2 - Math.floor(hash2) - 0.5;

    const theta = i * goldenAngle + j1 * 0.36;
    const y = (t * 2 - 1) * (HALF_HEIGHT * 0.76) + j2 * 0.12;
    const r = GEODE_RADIUS * (0.44 + Math.abs(j1) * 0.28);

    const sx = Math.cos(theta) * r;
    const sz = Math.sin(theta) * r;
    const seed = new THREE.Vector3(sx, y, sz);
    const radialDir = new THREE.Vector3(sx, y * 0.32, sz).normalize();
    const tangentDir = new THREE.Vector3(-Math.sin(theta), 0.18 * j2, Math.cos(theta)).normalize();

    sites.push({ seed, radialDir, tangentDir });
  }
  return sites;
}

/**
 * Clips a convex polyhedron against `dot(n, p) <= d` using 3D Sutherland-Hodgman
 * and caps the cleaved plane.
 */
function clipPolyhedronByPlane(
  faces: THREE.Vector3[][],
  planeNormal: THREE.Vector3,
  planeOffset: number
): THREE.Vector3[][] {
  const eps = 1e-5;
  const outFaces: THREE.Vector3[][] = [];
  const cutEdgePoints: THREE.Vector3[] = [];

  for (let f = 0; f < faces.length; f++) {
    const poly = faces[f];
    const clipped: THREE.Vector3[] = [];
    const n = poly.length;

    for (let i = 0; i < n; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % n];
      const da = a.dot(planeNormal) - planeOffset;
      const db = b.dot(planeNormal) - planeOffset;

      const aIn = da <= eps;
      const bIn = db <= eps;

      if (aIn && bIn) {
        clipped.push(b.clone());
      } else if (aIn && !bIn) {
        const t = da / (da - db);
        const inter = a.clone().lerp(b, t);
        clipped.push(inter);
        cutEdgePoints.push(inter.clone());
      } else if (!aIn && bIn) {
        const t = da / (da - db);
        const inter = a.clone().lerp(b, t);
        clipped.push(inter);
        cutEdgePoints.push(inter.clone());
        clipped.push(b.clone());
      }
    }

    if (clipped.length >= 3) {
      outFaces.push(clipped);
    }
  }

  const uniqueCut: THREE.Vector3[] = [];
  for (let i = 0; i < cutEdgePoints.length; i++) {
    const p = cutEdgePoints[i];
    let dup = false;
    for (let j = 0; j < uniqueCut.length; j++) {
      if (uniqueCut[j].distanceToSquared(p) < 1e-6) {
        dup = true;
        break;
      }
    }
    if (!dup) uniqueCut.push(p);
  }

  if (uniqueCut.length >= 3) {
    const center = new THREE.Vector3();
    for (let i = 0; i < uniqueCut.length; i++) center.add(uniqueCut[i]);
    center.divideScalar(uniqueCut.length);

    const upRef = Math.abs(planeNormal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const uAxis = new THREE.Vector3().crossVectors(upRef, planeNormal).normalize();
    const vAxis = new THREE.Vector3().crossVectors(planeNormal, uAxis).normalize();

    uniqueCut.sort((pa, pb) => {
      const da = pa.clone().sub(center);
      const db = pb.clone().sub(center);
      return Math.atan2(da.dot(vAxis), da.dot(uAxis)) - Math.atan2(db.dot(vAxis), db.dot(uAxis));
    });

    outFaces.push(uniqueCut);
  }

  return outFaces;
}

function computePolyhedronCentroid(faces: THREE.Vector3[][]): THREE.Vector3 {
  const centroid = new THREE.Vector3();
  let count = 0;
  for (let f = 0; f < faces.length; f++) {
    for (let i = 0; i < faces[f].length; i++) {
      centroid.add(faces[f][i]);
      count++;
    }
  }
  if (count > 0) centroid.divideScalar(count);
  return centroid;
}

/**
 * Triangulates a clipped polyhedron centered at `subCentroid` and bakes `surfaceMeta` (`vec4`):
 *   - `.x`: faceType (`0.0` = outer Azurite rind, `0.5` = Tier-1 primary cleavage wall, `1.0` = Tier-2 secondary internal cleavage wall)
 *   - `.y`: edgeDistance (`0.0` at cleavage boundary -> `1.0` at face center, driving Kintsugi gold lacquer rims)
 *   - `.z`: radialDepth (`0.0` at cylinder core axis -> `1.0` at outer barrel)
 *   - `.w`: parentShardIdx (`0..13`, indexing `uShardState` uniformArray on the GPU)
 * Also extracts scaled convex-hull vertices for Rapier `ColliderDesc.convexHull`.
 */
function buildSubShardGeometryAndHull(
  faces: THREE.Vector3[][],
  subCentroid: THREE.Vector3,
  parentShardIdx: number,
  tier2Planes: Array<{ normal: THREE.Vector3; offset: number }>
): { geometry: THREE.BufferGeometry; hullVertices: Float32Array } {
  const positions: number[] = [];
  const normals: number[] = [];
  const geodeCoords: number[] = [];
  const surfaceMeta: number[] = [];
  const hullPts: number[] = [];

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNorm = new THREE.Vector3();
  const faceCenter = new THREE.Vector3();
  const radialCheck = new THREE.Vector3();

  for (let f = 0; f < faces.length; f++) {
    const poly = faces[f];
    if (poly.length < 3) continue;

    faceCenter.set(0, 0, 0);
    for (let i = 0; i < poly.length; i++) faceCenter.add(poly[i]);
    faceCenter.divideScalar(poly.length);

    ab.subVectors(poly[1], poly[0]);
    ac.subVectors(poly[2], poly[0]);
    faceNorm.crossVectors(ab, ac).normalize();

    if (faceNorm.dot(faceCenter.clone().sub(subCentroid)) < 0) {
      faceNorm.negate();
      poly.reverse();
    }

    radialCheck.set(faceCenter.x, 0, faceCenter.z).normalize();
    const radialAlign = faceNorm.x * radialCheck.x + faceNorm.z * radialCheck.z;
    const rCenter = Math.hypot(faceCenter.x, faceCenter.z);

    let isTier2Cut = false;
    for (let pIdx = 0; pIdx < tier2Planes.length; pIdx++) {
      const tp = tier2Planes[pIdx];
      const align = Math.abs(faceNorm.dot(tp.normal));
      const distToPlane = Math.abs(faceCenter.dot(tp.normal) - tp.offset);
      if (align > 0.96 && distToPlane < 0.015) {
        isTier2Cut = true;
        break;
      }
    }

    let faceType = 0.5; // Tier-1 primary Voronoi cleavage face
    if ((radialAlign > 0.78 && rCenter > GEODE_RADIUS * 0.88) || Math.abs(faceNorm.y) > 0.92) {
      faceType = 0.0; // Outer Azurite-slate core barrel or cut end cap
    } else if (isTier2Cut) {
      faceType = 1.0; // Tier-2 secondary internal Realgar Amber & Cinnabar cleavage face
    }

    const rNormCenter = clampScalar(rCenter / GEODE_RADIUS, 0, 1);

    for (let i = 0; i < poly.length; i++) {
      const p0 = poly[i];
      const p1 = poly[(i + 1) % poly.length];

      // Collect inset convex hull vertex (`0.84` scale prevents resting self-collision while preserving exact facet collision on contact)
      hullPts.push(
        (p0.x - subCentroid.x) * 0.84,
        (p0.y - subCentroid.y) * 0.84,
        (p0.z - subCentroid.z) * 0.84
      );

      const r0 = clampScalar(Math.hypot(p0.x, p0.z) / GEODE_RADIUS, 0, 1);
      const r1 = clampScalar(Math.hypot(p1.x, p1.z) / GEODE_RADIUS, 0, 1);

      positions.push(
        faceCenter.x - subCentroid.x,
        faceCenter.y - subCentroid.y,
        faceCenter.z - subCentroid.z,
        p0.x - subCentroid.x,
        p0.y - subCentroid.y,
        p0.z - subCentroid.z,
        p1.x - subCentroid.x,
        p1.y - centroidComponent(subCentroid.y),
        p1.z - subCentroid.z
      );

      // Continuous monolith-space coordinates so 3D FBM mineral grain is 100% seamless across sub-shard boundaries
      geodeCoords.push(
        faceCenter.x,
        faceCenter.y,
        faceCenter.z,
        p0.x,
        p0.y,
        p0.z,
        p1.x,
        p1.y,
        p1.z
      );

      for (let k = 0; k < 3; k++) {
        normals.push(faceNorm.x, faceNorm.y, faceNorm.z);
      }

      surfaceMeta.push(
        faceType,
        1.0,
        rNormCenter,
        parentShardIdx,
        faceType,
        0.0,
        r0,
        parentShardIdx,
        faceType,
        0.0,
        r1,
        parentShardIdx
      );
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('geodePos', new THREE.Float32BufferAttribute(geodeCoords, 3));
  geometry.setAttribute('surfaceMeta', new THREE.Float32BufferAttribute(surfaceMeta, 4));
  geometry.computeBoundingSphere();

  return { geometry, hullVertices: new Float32Array(hullPts) };
}

function centroidComponent(v: number): number {
  return v;
}

/**
 * Builds a Tier-1 primary shard polyhedron AND recursively cleaves it into 4 Tier-2
 * convex-hull sub-shards along two oblique internal mineral cleavage planes with `0.0`
 * resting gap (so intact blocks show zero artificial X-seams until hovered or shattered).
 */
function buildTwoTierShardHierarchy(sites: VoronoiSite[], siteIdx: number): {
  parentCentroid: THREE.Vector3;
  subGeometries: Array<{
    geometry: THREE.BufferGeometry;
    hullVertices: Float32Array;
    localOffset: THREE.Vector3;
    outwardDir: THREE.Vector3;
    rotAxis: THREE.Vector3;
  }>;
} {
  const s0 = GEODE_RADIUS * 1.15;
  const h0 = HALF_HEIGHT;

  const v = [
    new THREE.Vector3(-s0, -h0, -s0),
    new THREE.Vector3(s0, -h0, -s0),
    new THREE.Vector3(s0, h0, -s0),
    new THREE.Vector3(-s0, h0, -s0),
    new THREE.Vector3(-s0, -h0, s0),
    new THREE.Vector3(s0, -h0, s0),
    new THREE.Vector3(s0, h0, s0),
    new THREE.Vector3(-s0, h0, s0),
  ];

  let primaryFaces: THREE.Vector3[][] = [
    [v[0], v[3], v[2], v[1]],
    [v[4], v[5], v[6], v[7]],
    [v[0], v[1], v[5], v[4]],
    [v[2], v[3], v[7], v[6]],
    [v[0], v[4], v[7], v[3]],
    [v[1], v[2], v[6], v[5]],
  ];

  const site = sites[siteIdx];

  // 1. 36-sided faceted mineral core barrel
  const barrelSides = 36;
  for (let k = 0; k < barrelSides; k++) {
    const ang = (k / barrelSides) * Math.PI * 2;
    const ripple = 1.0 + 0.013 * Math.sin(k * 7.0 + siteIdx * 0.5) + 0.007 * Math.cos(k * 11.0);
    const n = new THREE.Vector3(Math.cos(ang), 0, Math.sin(ang));
    primaryFaces = clipPolyhedronByPlane(primaryFaces, n, GEODE_RADIUS * ripple);
  }

  // 2. Top & bottom core-sample end caps
  primaryFaces = clipPolyhedronByPlane(primaryFaces, new THREE.Vector3(0, 1, 0), HALF_HEIGHT);
  primaryFaces = clipPolyhedronByPlane(primaryFaces, new THREE.Vector3(0, -1, 0), HALF_HEIGHT);

  // 3. Tier-1 Voronoi bisector planes with all neighboring sites
  const tier1SeamGap = 0.0055;
  const mid = new THREE.Vector3();
  const normal = new THREE.Vector3();

  for (let j = 0; j < sites.length; j++) {
    if (j === siteIdx) continue;
    mid.addVectors(site.seed, sites[j].seed).multiplyScalar(0.5);
    normal.subVectors(sites[j].seed, site.seed).normalize();
    const offset = normal.dot(mid) - tier1SeamGap;
    primaryFaces = clipPolyhedronByPlane(primaryFaces, normal, offset);
  }

  const parentCentroid = computePolyhedronCentroid(primaryFaces);

  // 4. Cleave this Tier-1 shard into 4 Tier-2 sub-shards with ZERO initial gap (`tier2SeamGap = 0.0`)
  // so intact shards look 100% solid until hovered or secondarily shattered!
  const nA = new THREE.Vector3(
    site.tangentDir.x * 0.65 + site.radialDir.x * 0.25,
    0.72 + 0.18 * Math.sin(siteIdx * 2.7),
    site.tangentDir.z * 0.65 + site.radialDir.z * 0.25
  ).normalize();

  const nB = new THREE.Vector3()
    .crossVectors(nA, site.radialDir)
    .addScaledVector(new THREE.Vector3(0, 1, 0), 0.28 * Math.cos(siteIdx * 1.9))
    .normalize();

  const dA = nA.dot(parentCentroid);
  const dB = nB.dot(parentCentroid);

  const signs: Array<[number, number]> = [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ];

  const subGeometries: Array<{
    geometry: THREE.BufferGeometry;
    hullVertices: Float32Array;
    localOffset: THREE.Vector3;
    outwardDir: THREE.Vector3;
    rotAxis: THREE.Vector3;
  }> = [];

  for (let sIdx = 0; sIdx < SUB_SHARDS_PER_SHARD; sIdx++) {
    const [sA, sB] = signs[sIdx];
    const pA = nA.clone().multiplyScalar(sA);
    const offA = sA * dA;
    const pB = nB.clone().multiplyScalar(sB);
    const offB = sB * dB;

    let subFaces = clipPolyhedronByPlane(primaryFaces, pA, offA);
    subFaces = clipPolyhedronByPlane(subFaces, pB, offB);

    if (subFaces.length < 4) {
      subFaces = primaryFaces;
    }

    const subCentroid = computePolyhedronCentroid(subFaces);
    const localOffset = subCentroid.clone().sub(parentCentroid);
    const outwardDir =
      localOffset.lengthSq() > 1e-5
        ? localOffset.clone().normalize()
        : new THREE.Vector3(sA * 0.7, sB * 0.7, 0.2).normalize();
    const rotAxis = new THREE.Vector3().crossVectors(outwardDir, site.radialDir).normalize();
    if (rotAxis.lengthSq() < 1e-4) rotAxis.set(1, 0, 0);

    const { geometry, hullVertices } = buildSubShardGeometryAndHull(
      subFaces,
      subCentroid,
      siteIdx,
      [
        { normal: pA, offset: offA },
        { normal: pB, offset: offB },
      ]
    );

    subGeometries.push({
      geometry,
      hullVertices,
      localOffset,
      outwardDir,
      rotAxis,
    });
  }

  return { parentCentroid, subGeometries };
}

export async function bootDay02Shatter(): Promise<SpecimenStage> {
  await RAPIER.init();

  const dayEntry = getDayByNumber(2);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.22, 4.45],
    fov: 35,
    enableOrbit: false, // Direct click, flick & drag controls wavefront fracture, Tier-2 sub-shatter & Kintsugi tension
  });

  // Zero-gravity suspended studio physics world in Rapier 0.21
  const world = new RAPIER.World({ x: 0.0, y: 0.0, z: 0.0 });
  stage.shell.registerDispose(() => world.free());

  // Physical mineral pigments from src/core/palette.ts
  const azuritePrussian = createPigmentUniforms('azuritePrussian');
  const lapisUltramarine = createPigmentUniforms('lapisUltramarine');
  const rawUmber = createPigmentUniforms('rawUmber');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const boneCalcite = createPigmentUniforms('boneCalcite');

  // Per-Shard GPU State Array (`vec4` per Tier-1 shard):
  //   x: openAmount [0..1]
  //   y: subShatter [0..1]
  //   z: stressFlash [0..1] (triboluminescent cleavage flash as wavefront hits or Tier-2 cracks)
  //   w: hoverFaultGlow [0..1] (subsurface fault-line pre-glow when hovering an intact shard)
  const shardStateVectors: THREE.Vector4[] = [];
  for (let i = 0; i < SHARD_COUNT; i++) {
    shardStateVectors.push(new THREE.Vector4(INITIAL_SPREAD, 0.0, 0.0, 0.0));
  }
  const uShardState = uniformArray<'vec4'>(shardStateVectors, 'vec4');
  const uGlobalOpen = uniform(INITIAL_SPREAD);
  const uShockImpulse = uniform(0.0);

  // Root specimen group suspended in Chalky Slate (#212529) studio space
  const geodeRig = new THREE.Group();
  geodeRig.name = 'TwoTierKintsugiMonolithSpecimen';
  stage.specimenGroup.add(geodeRig);

  // Warm interior point light illuminating opened cleavage fissures from within
  const coreLight = new THREE.PointLight(0xf09238, 2.0, 4.4, 1.35);
  coreLight.position.set(0, 0.04, 0.08);
  geodeRig.add(coreLight);

  // ---------------------------------------------------------------------------
  // 1. TSL NodeMaterial Driven by Per-Shard GPU Telemetry (`uShardState`)
  // ---------------------------------------------------------------------------
  const shardMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.44,
    metalness: 0.1,
  });

  const pGeode = attribute('geodePos', 'vec3');
  const pWorld = positionWorld;
  const meta = attribute('surfaceMeta', 'vec4');
  const faceType = meta.x; // 0.0 = outer rind, 0.5 = Tier-1 primary cleavage wall, 1.0 = Tier-2 internal cleavage wall
  const edgeDist = meta.y; // 0.0 at cleavage seam rim -> 1.0 at face interior
  const radialDepth = meta.z; // 0.0 at cylinder core axis -> 1.0 at outer barrel
  const shardIdxNode = uint(meta.w);

  // Read this specific shard's live physical state from the GPU uniformArray
  const shardState = uShardState.element(shardIdxNode);
  const sOpen = float(shardState.x);
  const sSubShatter = float(shardState.y);
  const sFlash = float(shardState.z);
  const sHoverFault = float(shardState.w);

  const toPointer = pWorld.sub(stage.uPointerWorld);
  const pointerProx = exp(dot(toPointer, toPointer).negate().mul(1.85));

  // Evaluate 3D FBM mineral grain in continuous monolith coordinates (`pGeode`) so
  // intact Tier-1 blocks have 100% seamless geological strata across their latent sub-shards
  const tooth = mineralToothFbm(pGeode.mul(3.8), float(3.1));
  const fineGrain = mineralToothFbm(pGeode.mul(8.5), float(4.2));

  // Role masks
  const isOuterRind = float(1.0).sub(smoothstep(0.15, 0.35, faceType));
  const isTier2Wall = smoothstep(0.65, 0.85, faceType);
  const isTier1Wall = float(1.0).sub(max(isOuterRind, isTier2Wall));

  // Gate Tier-2 rim & wall visibility so intact shards (`sSubShatter == 0`, `sHoverFault == 0`)
  // show zero artificial X-seams on the outer cylinder, then ignite when hovered or shattered!
  const tier2Gate = clamp(sSubShatter.mul(1.85).add(sHoverFault.mul(0.85)), 0.0, 1.0);

  // A. Outer Core-Sample Rind: Deep Azurite-Prussian mineral slate
  const rindGlazeD = clamp(float(0.64).add(tooth.mul(0.38)).sub(pointerProx.mul(0.12)), 0.16, 1.35);
  const outerRindRgb = kubelkaMunkGlaze(
    azuritePrussian.K,
    azuritePrussian.S,
    rindGlazeD,
    rawUmber.rInf
  );

  // B. Tier-1 Primary Voronoi Cleavage Walls: Stratified Lapis Ultramarine & Bone Calcite agate banding
  const bandWave = smoothstep(0.28, 0.72, fineGrain.add(radialDepth.mul(0.45)));
  const cleavageAgate = kubelkaMunkGlaze(
    lapisUltramarine.K,
    lapisUltramarine.S,
    clamp(float(0.28).add(bandWave.mul(0.52)), 0.08, 0.94),
    boneCalcite.rInf
  );
  const innerWarmth = smoothstep(0.68, 0.08, radialDepth);
  const tier1WallRgb = mix(cleavageAgate, realgarAmber.rInf, innerWarmth.mul(0.45));

  // C. Tier-2 Secondary Internal Fracture Walls: Incandescent Realgar Amber & Cinnabar Vermilion
  const tier2Glaze = kubelkaMunkGlaze(
    realgarAmber.K,
    realgarAmber.S,
    clamp(float(0.35).add(fineGrain.mul(0.45)), 0.08, 0.95),
    boneCalcite.rInf
  );
  const tier2WallRgb = mix(tier2Glaze, cinnabarVermilion.rInf, bandWave.mul(0.42));

  // D. Kintsugi Gold-Ochre Lacquer Rims (Tier-1 always traced in gold; Tier-2 gated by `tier2Gate`)
  const rimWidth = mix(float(0.17), float(0.22), isTier2Wall);
  const rimBand = float(1.0).sub(smoothstep(0.0, rimWidth.add(tooth.mul(0.07)), edgeDist));
  const kintsugiRimMask = isTier1Wall
    .add(isTier2Wall.mul(tier2Gate))
    .mul(rimBand);

  const kintsugiGoldRgb = mix(
    yellowOchre.rInf,
    boneCalcite.rInf,
    clamp(float(0.28).add(sFlash.mul(0.52)).add(pointerProx.mul(0.25)), 0.0, 0.92)
  );

  const wallColor = mix(
    mix(tier1WallRgb, outerRindRgb, isOuterRind),
    tier2WallRgb,
    isTier2Wall.mul(tier2Gate)
  );
  const rimmedColor = mix(wallColor, kintsugiGoldRgb, clamp(kintsugiRimMask.mul(0.94), 0.0, 1.0));

  // Per-shard triboluminescent cleavage flash & subsurface resonance
  const flashGlowRgb = mix(realgarAmber.rInf, boneCalcite.rInf, sFlash.mul(0.45));
  const interiorGlow = clamp(
    isTier1Wall
      .mul(innerWarmth.mul(sOpen.mul(0.42)).add(sFlash.mul(0.58)))
      .add(isTier2Wall.mul(tier2Gate).mul(float(0.48).add(sFlash.mul(0.5))))
      .add(kintsugiRimMask.mul(sFlash.mul(0.65)))
      .add(pointerProx.mul(0.22)),
    0.0,
    0.85
  );

  shardMat.colorNode = mix(rimmedColor, flashGlowRgb, interiorGlow);
  shardMat.roughnessNode = clamp(
    mix(float(0.54), float(0.18), kintsugiRimMask.add(isTier2Wall.mul(tier2Gate).mul(0.45))).sub(
      tooth.mul(0.12)
    ),
    0.14,
    0.76
  );
  shardMat.metalnessNode = clamp(
    kintsugiRimMask.mul(0.8).add(isOuterRind.mul(0.06)).add(0.05),
    0.04,
    0.86
  );

  // ---------------------------------------------------------------------------
  // 2. Construct 56 True 6-DOF Rapier Convex-Hull Rigid Bodies (14 Clusters × 4 Sub-Shards)
  // ---------------------------------------------------------------------------
  const sites = generateMonolithSites(SHARD_COUNT);
  const clusters: ShardCluster[] = [];
  const allSubShards: SubShardRecord[] = [];
  const allSubMeshes: THREE.Mesh[] = [];
  const meshToSubShard = new Map<THREE.Object3D, SubShardRecord>();

  for (let i = 0; i < SHARD_COUNT; i++) {
    const { parentCentroid, subGeometries } = buildTwoTierShardHierarchy(sites, i);
    const radialDir = parentCentroid.clone().setY(parentCentroid.y * 0.34).normalize();
    const isFrontWindow = radialDir.z > 0.35 && Math.abs(parentCentroid.y) < HALF_HEIGHT * 0.68;

    const subShards: SubShardRecord[] = [];

    for (let sIdx = 0; sIdx < subGeometries.length; sIdx++) {
      const sg = subGeometries[sIdx];
      const globalSubIdx = allSubShards.length;
      const restPos = parentCentroid.clone().add(sg.localOffset);

      const mesh = new THREE.Mesh(sg.geometry, shardMat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.position.copy(restPos);
      geodeRig.add(mesh);

      // True 6-DOF Rapier dynamic rigid body with actual 3D convex-hull cleavage geometry
      const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(restPos.x, restPos.y, restPos.z)
        .setLinearDamping(11.0)
        .setAngularDamping(12.0)
        .setCanSleep(false);
      const body = world.createRigidBody(bodyDesc);

      const hullDesc =
        RAPIER.ColliderDesc.convexHull(sg.hullVertices) ?? RAPIER.ColliderDesc.ball(0.08);
      hullDesc.setDensity(2.4).setRestitution(0.06).setFriction(0.82);
      world.createCollider(hullDesc, body);

      const record: SubShardRecord = {
        parentIdx: i,
        subIdx: sIdx,
        globalSubIdx,
        mesh,
        body,
        localOffset: sg.localOffset,
        outwardDir: sg.outwardDir,
        rotAxis: sg.rotAxis,
        restPos,
        restQuat: new THREE.Quaternion(),
      };

      subShards.push(record);
      allSubShards.push(record);
      allSubMeshes.push(mesh);
      meshToSubShard.set(mesh, record);
    }

    clusters.push({
      index: i,
      homePos: parentCentroid.clone(),
      radialDir,
      tangentDir: sites[i].tangentDir.clone(),
      isFrontWindow,
      subShards,
      openAmount: INITIAL_SPREAD,
      targetOpen: INITIAL_SPREAD,
      waveDelaySec: 0.0,
      pendingWaveTarget: null,
      subShatter: 0.0,
      targetSubShatter: 0.0,
      stressFlash: 0.0,
      hoverFaultGlow: 0.0,
    });
  }

  // ---------------------------------------------------------------------------
  // 3. Coupled Viscoelastic Kintsugi Bridges (`applyImpulseAtPoint` + Per-Strand Strain Attribute)
  // ---------------------------------------------------------------------------
  const bridges: KintsugiBridge[] = [];

  // A. Tier-1 Bridges connecting nearest sub-shards of neighboring primary clusters
  for (let i = 0; i < SHARD_COUNT; i++) {
    for (let j = i + 1; j < SHARD_COUNT; j++) {
      const dist = clusters[i].homePos.distanceTo(clusters[j].homePos);
      if (dist < 0.78) {
        // Find the 2 closest pairs of sub-shards across the shared Voronoi boundary
        const pairs: Array<{ subA: SubShardRecord; subB: SubShardRecord; d: number }> = [];
        for (const sA of clusters[i].subShards) {
          for (const sB of clusters[j].subShards) {
            pairs.push({ subA: sA, subB: sB, d: sA.restPos.distanceTo(sB.restPos) });
          }
        }
        pairs.sort((a, b) => a.d - b.d);

        for (let k = 0; k < 2; k++) {
          const { subA, subB } = pairs[k];
          const midWorld = subA.restPos.clone().add(subB.restPos).multiplyScalar(0.5);
          const spanVec = subB.restPos.clone().sub(subA.restPos).multiplyScalar(0.22);
          const anchorWorldA = midWorld.clone().sub(spanVec);
          const anchorWorldB = midWorld.clone().add(spanVec);

          const localA = anchorWorldA.sub(subA.restPos);
          const localB = anchorWorldB.sub(subB.restPos);
          const sagDir = midWorld.clone().negate().normalize();

          bridges.push({
            subA,
            subB,
            localA,
            localB,
            sagDir,
            restSpan: spanVec.length() * 2.0,
            tier: 1,
            parentShardIdx: i,
          });
        }
      }
    }
  }

  // B. Tier-2 Bridges connecting the 4 sibling sub-shards inside each primary cluster
  const subPairs: Array<[number, number]> = [
    [0, 1],
    [2, 3],
    [0, 2],
    [1, 3],
    [0, 3],
  ];
  for (let i = 0; i < SHARD_COUNT; i++) {
    const c = clusters[i];
    for (let p = 0; p < subPairs.length; p++) {
      const [aIdx, bIdx] = subPairs[p];
      const subA = c.subShards[aIdx];
      const subB = c.subShards[bIdx];
      const midWorld = subA.restPos.clone().add(subB.restPos).multiplyScalar(0.5);
      const spanVec = subB.restPos.clone().sub(subA.restPos).multiplyScalar(0.24);

      const localA = midWorld.clone().sub(spanVec).sub(subA.restPos);
      const localB = midWorld.clone().add(spanVec).sub(subB.restPos);
      const sagDir = c.radialDir.clone().negate();

      bridges.push({
        subA,
        subB,
        localA,
        localB,
        sagDir,
        restSpan: spanVec.length() * 2.0,
        tier: 2,
        parentShardIdx: i,
      });
    }
  }

  const totalInstances = bridges.length * STRAND_SEGMENTS;
  const threadGeom = new THREE.CylinderGeometry(0.008, 0.008, 1.0, 8, 1);

  // Per-instance strain & shock telemetry attribute (`vec2`: `[normalizedStrain, flashGlow]`)
  const tensionData = new Float32Array(totalInstances * 2);
  const tensionAttr = new THREE.InstancedBufferAttribute(tensionData, 2);
  tensionAttr.setUsage(THREE.DynamicDrawUsage);
  threadGeom.setAttribute('instanceTension', tensionAttr);

  const threadMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.22,
    metalness: 0.78,
  });

  const instTension = attribute('instanceTension', 'vec2');
  const strandStrain = instTension.x;
  const strandFlash = instTension.y;

  const tTooth = mineralToothFbm(positionLocal.mul(7.0), float(2.2));
  const goldResin = kubelkaMunkGlaze(
    yellowOchre.K,
    yellowOchre.S,
    clamp(float(0.68).add(tTooth.mul(0.28)).sub(strandStrain.mul(0.22)), 0.2, 1.15),
    realgarAmber.rInf
  );
  // High-strain Kintsugi strands shift optically toward incandescent Realgar Amber & Bone Calcite gold
  const strainedGold = mix(goldResin, realgarAmber.rInf, clamp(strandStrain.mul(0.55), 0.0, 0.65));
  threadMat.colorNode = mix(
    strainedGold,
    boneCalcite.rInf,
    clamp(strandStrain.mul(0.42).add(strandFlash.mul(0.55)), 0.0, 0.88)
  );
  threadMat.roughnessNode = clamp(float(0.28).sub(strandStrain.mul(0.14)), 0.1, 0.35);

  const threadInstanced = new THREE.InstancedMesh(threadGeom, threadMat, totalInstances);
  threadInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  geodeRig.add(threadInstanced);

  // ---------------------------------------------------------------------------
  // 4. Spalled Micro-Cleavage Mineral Chips (`InstancedMesh` in Slow-Time Suspension)
  // ---------------------------------------------------------------------------
  const spallGeom = new THREE.OctahedronGeometry(0.013, 0);
  const spallMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.25,
    metalness: 0.62,
  });
  spallMat.colorNode = mix(
    yellowOchre.rInf,
    realgarAmber.rInf,
    clamp(uGlobalOpen.mul(0.65).add(uShockImpulse.mul(0.35)), 0.0, 0.85)
  );

  const spallInstanced = new THREE.InstancedMesh(spallGeom, spallMat, SPALL_CHIP_COUNT);
  spallInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  geodeRig.add(spallInstanced);

  const spallChips: SpallChip[] = [];
  for (let k = 0; k < SPALL_CHIP_COUNT; k++) {
    const parentShardIdx = k % SHARD_COUNT;
    const host = clusters[parentShardIdx];
    const h1 = Math.sin(k * 41.3 + 7.1) * 43758.5453;
    const u1 = h1 - Math.floor(h1) - 0.5;
    const h2 = Math.sin(k * 83.9 + 19.4) * 19483.21;
    const u2 = h2 - Math.floor(h2) - 0.5;

    const localOrigin = host.homePos
      .clone()
      .multiplyScalar(0.88)
      .addScaledVector(host.tangentDir, u1 * 0.22);
    localOrigin.y = clampScalar(localOrigin.y + u2 * 0.2, -HALF_HEIGHT * 0.8, HALF_HEIGHT * 0.8);

    const ejectDir = host.radialDir
      .clone()
      .addScaledVector(host.tangentDir, u1 * 0.65)
      .add(new THREE.Vector3(0, u2 * 0.45, 0))
      .normalize();

    const rotAxis = new THREE.Vector3(u2, 1.0, u1).normalize();
    const s = 0.65 + Math.abs(u1) * 1.1;

    spallChips.push({
      parentShardIdx,
      localOrigin,
      ejectDir,
      rotAxis,
      scale: new THREE.Vector3(s * 1.3, s * 0.55, s * 0.9),
      phase: k * 1.9,
    });
  }

  // ---------------------------------------------------------------------------
  // 5. Wavefront Fracture Propagation, Direct Shard Tractor & Rapier Step Loop
  // ---------------------------------------------------------------------------
  let isMonolithOpen = false;
  let globalSpread = INITIAL_SPREAD;
  let shockPulse = 0.0;

  let wasPointerDown = false;
  let isDragging = false;
  const pointerDownPos = new THREE.Vector2(0, 0);
  let pointerDownTime = 0;
  let dragStartSpread = INITIAL_SPREAD;
  let dragStartRadius = 0;
  let activeDragSubShard: SubShardRecord | null = null;

  // Pre-allocated scratch math & Rapier vector structs (strictly zero GC inside onTick)
  const shardRaycaster = new THREE.Raycaster();
  const rayHits: THREE.Intersection[] = [];
  const clusterPos = new THREE.Vector3();
  const clusterQuat = new THREE.Quaternion();
  const subRotQuat = new THREE.Quaternion();
  const targetPos = new THREE.Vector3();
  const targetQuat = new THREE.Quaternion();
  const curPos = new THREE.Vector3();
  const forceVec = new THREE.Vector3();
  const torqueVec = new THREE.Vector3();
  const curQuat = new THREE.Quaternion();
  const deltaQuat = new THREE.Quaternion();
  const rotAxis = new THREE.Vector3();
  const ptrLocal = new THREE.Vector3();
  const strikeLocal = new THREE.Vector3();
  const toShard = new THREE.Vector3();
  const worldA = new THREE.Vector3();
  const worldB = new THREE.Vector3();
  const segStart = new THREE.Vector3();
  const segEnd = new THREE.Vector3();
  const midThread = new THREE.Vector3();
  const dirThread = new THREE.Vector3();
  const threadQuat = new THREE.Quaternion();
  const threadScale = new THREE.Vector3();
  const threadMatrix = new THREE.Matrix4();
  const upVec = new THREE.Vector3(0, 1, 0);

  const rapierVec = { x: 0, y: 0, z: 0 };
  const rapierVecNeg = { x: 0, y: 0, z: 0 };
  const rapierPtA = { x: 0, y: 0, z: 0 };
  const rapierPtB = { x: 0, y: 0, z: 0 };

  // Initialize all 56 convex-hull rigid bodies at compact `INITIAL_SPREAD` on frame 0
  for (let i = 0; i < SHARD_COUNT; i++) {
    const c = clusters[i];
    const windowBoost = c.isFrontWindow ? 1.2 : 1.0;
    const openDist = INITIAL_SPREAD * 0.42 * windowBoost;
    clusterPos.copy(c.homePos).addScaledVector(c.radialDir, openDist);
    const tiltAngle = INITIAL_SPREAD * 0.2 * windowBoost * (i % 2 === 0 ? 1 : -0.85);
    clusterQuat.setFromAxisAngle(c.tangentDir, tiltAngle);

    for (let sIdx = 0; sIdx < c.subShards.length; sIdx++) {
      const sub = c.subShards[sIdx];
      targetPos.copy(sub.localOffset).applyQuaternion(clusterQuat).add(clusterPos);
      sub.body.setTranslation({ x: targetPos.x, y: targetPos.y, z: targetPos.z }, true);
      sub.body.setRotation(
        { x: clusterQuat.x, y: clusterQuat.y, z: clusterQuat.z, w: clusterQuat.w },
        true
      );
      sub.mesh.position.copy(targetPos);
      sub.mesh.quaternion.copy(clusterQuat);
    }
  }

  function findHoveredSubShard(camera: THREE.Camera, ndc: THREE.Vector2): SubShardRecord | null {
    shardRaycaster.setFromCamera(ndc, camera);
    rayHits.length = 0;
    shardRaycaster.intersectObjects(allSubMeshes, false, rayHits);
    if (rayHits.length > 0) {
      return meshToSubShard.get(rayHits[0].object) ?? null;
    }
    return null;
  }

  /**
   * Launches a causal longitudinal stress wave from `epicenter` across the 14 Voronoi blocks.
   * Each block cracks open when the wavefront reaches its centroid (`delay = dist / FRACTURE_WAVE_SPEED`).
   */
  function triggerWavefrontCascade(
    epicenter: THREE.Vector3,
    nextTargetOpen: number,
    impulseStrength: number,
    focusedClusterIdx = -1
  ): void {
    shockPulse = Math.min(1.0, shockPulse + 0.85 * impulseStrength);
    isMonolithOpen = nextTargetOpen > 0.3;

    for (let i = 0; i < SHARD_COUNT; i++) {
      const c = clusters[i];
      const dist = c.homePos.distanceTo(epicenter);
      c.waveDelaySec = i === focusedClusterIdx ? 0.0 : dist / FRACTURE_WAVE_SPEED;
      c.pendingWaveTarget = nextTargetOpen;
      if (c.waveDelaySec <= 0.015) {
        c.targetOpen = nextTargetOpen;
        c.pendingWaveTarget = null;
        c.stressFlash = Math.min(1.0, c.stressFlash + 0.95);
      }

      const focusBoost = i === focusedClusterIdx ? 1.75 : 1.0;
      const falloff = focusBoost / (1.0 + dist * dist * 1.4);
      const dirSign = nextTargetOpen > c.openAmount ? 1.0 : -0.65;

      for (let sIdx = 0; sIdx < c.subShards.length; sIdx++) {
        const sub = c.subShards[sIdx];
        const tr = sub.body.translation();
        curPos.set(tr.x, tr.y, tr.z);
        toShard.subVectors(curPos, epicenter);
        if (toShard.lengthSq() > 1e-5) toShard.normalize();

        forceVec
          .copy(c.radialDir)
          .multiplyScalar(0.8 * dirSign)
          .addScaledVector(toShard, 0.35 * dirSign)
          .multiplyScalar(impulseStrength * falloff * 0.095);

        rapierVec.x = forceVec.x;
        rapierVec.y = forceVec.y;
        rapierVec.z = forceVec.z;
        sub.body.applyImpulse(rapierVec, true);

        torqueVec
          .copy(c.tangentDir)
          .multiplyScalar(impulseStrength * falloff * 0.005 * (sIdx % 2 === 0 ? 1 : -1));
        rapierVec.x = torqueVec.x;
        rapierVec.y = torqueVec.y;
        rapierVec.z = torqueVec.z;
        sub.body.applyTorqueImpulse(rapierVec, true);
      }
    }
  }

  /**
   * Triggers a Tier-2 secondary shatter on cluster `clusterIdx`, breaking the internal
   * weld between its 4 convex-hull sub-shards and ejecting them along their cleavage planes.
   */
  function triggerClusterSecondaryShatter(clusterIdx: number, nextState: number): void {
    const c = clusters[clusterIdx];
    c.targetSubShatter = nextState;
    c.stressFlash = 1.0;
    shockPulse = Math.min(1.0, shockPulse + 0.75);

    const dirSign = nextState > c.subShatter ? 1.0 : -0.6;
    for (let sIdx = 0; sIdx < c.subShards.length; sIdx++) {
      const sub = c.subShards[sIdx];
      forceVec
        .copy(sub.outwardDir)
        .addScaledVector(c.radialDir, 0.35)
        .multiplyScalar(0.11 * dirSign);

      rapierVec.x = forceVec.x;
      rapierVec.y = forceVec.y;
      rapierVec.z = forceVec.z;
      sub.body.applyImpulse(rapierVec, true);

      torqueVec.copy(sub.rotAxis).multiplyScalar(0.0065 * dirSign * (sIdx % 2 === 0 ? 1 : -1));
      rapierVec.x = torqueVec.x;
      rapierVec.y = torqueVec.y;
      rapierVec.z = torqueVec.z;
      sub.body.applyTorqueImpulse(rapierVec, true);
    }
  }

  stage.onTick(({ elapsedSec, dtSec, pointer, camera }) => {
    const px = pointer.worldPos.x;
    const py = pointer.worldPos.y;
    const ptrRadius = Math.hypot(px, py);

    ptrLocal.copy(pointer.worldPos);
    geodeRig.worldToLocal(ptrLocal);

    const hoveredSub = findHoveredSubShard(camera, pointer.ndc);
    const hoveredClusterIdx = hoveredSub ? hoveredSub.parentIdx : -1;

    // 1. Pointer Press, Wavefront Drag, & Two-Tier Click Logic
    if (pointer.isDown && !wasPointerDown) {
      wasPointerDown = true;
      isDragging = false;
      pointerDownPos.set(px, py);
      pointerDownTime = elapsedSec;
      dragStartSpread = globalSpread;
      dragStartRadius = ptrRadius;
      activeDragSubShard = hoveredSub;

      // Pressing on the closed monolith immediately nucleates a cleavage wave at the strike point
      if (globalSpread < 0.16) {
        strikeLocal.copy(ptrLocal);
        const focusIdx = activeDragSubShard ? activeDragSubShard.parentIdx : -1;
        if (focusIdx >= 0) {
          clusters[focusIdx].stressFlash = 1.0;
        }
        triggerWavefrontCascade(strikeLocal, 0.28, 0.65, focusIdx);
      }
    } else if (pointer.isDown && wasPointerDown) {
      const moveDist = Math.hypot(px - pointerDownPos.x, py - pointerDownPos.y);
      if (moveDist > 0.045) {
        isDragging = true;
      }

      if (isDragging) {
        const radialDelta = ptrRadius - dragStartRadius;
        const directionalDelta = (px - pointerDownPos.x) * 0.72 - (py - pointerDownPos.y) * 0.52;
        const combinedPull = radialDelta * 0.55 + directionalDelta * 0.95;
        const nextSpread = clampScalar(dragStartSpread + combinedPull * 1.25, 0.02, 0.92);
        isMonolithOpen = nextSpread > 0.3;

        // Distance-weighted drag peeling: shards closest to the pointer peel open furthest first!
        for (let i = 0; i < SHARD_COUNT; i++) {
          const c = clusters[i];
          const distToPtr = c.homePos.distanceTo(ptrLocal);
          const localPeelBoost = 1.0 + 0.28 * Math.exp(-distToPtr * distToPtr * 1.6);
          c.pendingWaveTarget = null;
          c.targetOpen = clampScalar(nextSpread * localPeelBoost, 0.02, 0.95);
        }

        // If dragging outward past primary shatter, progressively cleave Tier-2 sub-shards
        // on the grabbed shard AND any shard swept across during the drag
        if (nextSpread > 0.38) {
          const subTarget = clampScalar((nextSpread - 0.32) * 1.75, 0, 1);
          if (activeDragSubShard) {
            const dragCluster = clusters[activeDragSubShard.parentIdx];
            if (subTarget > dragCluster.targetSubShatter + 0.1) {
              dragCluster.stressFlash = Math.max(dragCluster.stressFlash, 0.75);
            }
            dragCluster.targetSubShatter = Math.max(dragCluster.targetSubShatter, subTarget);
          }
          if (hoveredClusterIdx >= 0) {
            const hovCluster = clusters[hoveredClusterIdx];
            if (subTarget > hovCluster.targetSubShatter + 0.1) {
              hovCluster.stressFlash = Math.max(hovCluster.stressFlash, 0.75);
            }
            hovCluster.targetSubShatter = Math.max(hovCluster.targetSubShatter, subTarget * 0.9);
          }
        } else if (combinedPull < -0.14) {
          // Dragging inward mends Tier-2 sub-shards back together
          for (let i = 0; i < SHARD_COUNT; i++) {
            clusters[i].targetSubShatter = Math.max(
              0,
              clusters[i].targetSubShatter + combinedPull * 0.85
            );
          }
        }
      }
    } else if (!pointer.isDown && wasPointerDown) {
      wasPointerDown = false;
      const heldSec = elapsedSec - pointerDownTime;

      if (!isDragging && heldSec < 0.45) {
        if (!isMonolithOpen || globalSpread < 0.32) {
          // Tier-1 Wavefront Shatter: Propagate fracture wave outward from the exact click location!
          strikeLocal.copy(
            hoveredClusterIdx >= 0 ? clusters[hoveredClusterIdx].homePos : ptrLocal
          );
          triggerWavefrontCascade(strikeLocal, 0.76, 1.15, hoveredClusterIdx);
        } else if (hoveredClusterIdx >= 0) {
          // Tier-2 Secondary Shatter: Cleave the clicked primary shard into its 4 convex-hull sub-shards!
          const hitCluster = clusters[hoveredClusterIdx];
          if (hitCluster.targetSubShatter < 0.5) {
            triggerClusterSecondaryShatter(hoveredClusterIdx, 1.0);
          } else {
            // Clicking an already sub-shattered shard cascades secondary fracture to its 2 nearest intact neighbors
            let cascaded = 0;
            for (let j = 0; j < SHARD_COUNT; j++) {
              if (
                j !== hoveredClusterIdx &&
                clusters[j].targetSubShatter < 0.5 &&
                clusters[j].homePos.distanceTo(hitCluster.homePos) < 0.78
              ) {
                triggerClusterSecondaryShatter(j, 1.0);
                cascaded++;
                if (cascaded >= 2) break;
              }
            }
            if (cascaded === 0) {
              triggerClusterSecondaryShatter(hoveredClusterIdx, 0.0);
            }
          }
        } else {
          // Clicked empty Chalky Slate space while open -> mend all Tier-2 & Tier-1 seams via reverse wave
          for (let i = 0; i < SHARD_COUNT; i++) {
            clusters[i].targetSubShatter = 0.0;
          }
          strikeLocal.set(0, 0, 0);
          triggerWavefrontCascade(strikeLocal, INITIAL_SPREAD, 0.65, -1);
        }
      }
      isDragging = false;
      activeDragSubShard = null;
    }

    // 2. Advance Per-Cluster Wavefront Timers, GPU Telemetry & Overdamped Sub-Shard Targets
    const openAlpha = 1 - Math.exp(-dtSec * 5.4);
    const subAlpha = 1 - Math.exp(-dtSec * 6.2);
    const hoverAlpha = 1 - Math.exp(-dtSec * 10.0);
    shockPulse *= Math.exp(-dtSec * 3.8);

    const springK = isDragging ? 22.0 : 16.0;
    const rotSpringK = 1.5;

    let sumOpen = 0;
    for (let i = 0; i < SHARD_COUNT; i++) {
      const c = clusters[i];

      // Check if a propagating fracture wavefront just arrived at this shard
      if (c.pendingWaveTarget !== null) {
        c.waveDelaySec -= dtSec;
        if (c.waveDelaySec <= 0) {
          c.targetOpen = c.pendingWaveTarget;
          c.pendingWaveTarget = null;
          c.stressFlash = Math.min(1.0, c.stressFlash + 0.9);
        }
      }

      c.openAmount += (c.targetOpen - c.openAmount) * openAlpha;
      c.subShatter += (c.targetSubShatter - c.subShatter) * subAlpha;
      c.stressFlash *= Math.exp(-dtSec * 3.6);

      // Subsurface fault-line pre-glow when hovering over an open shard before Tier-2 click
      const targetHoverGlow =
        i === hoveredClusterIdx && c.openAmount > 0.25 && c.subShatter < 0.5 ? 0.38 : 0.0;
      c.hoverFaultGlow += (targetHoverGlow - c.hoverFaultGlow) * hoverAlpha;

      // Sync per-shard GPU uniformArray element in place (zero GC)
      shardStateVectors[i].set(c.openAmount, c.subShatter, c.stressFlash, c.hoverFaultGlow);
      sumOpen += c.openAmount;

      // Compute parent cluster equilibrium frame
      const windowBoost = c.isFrontWindow ? 1.32 : 1.0;
      const secondaryPush = c.subShatter * 0.14;
      const drift = Math.sin(elapsedSec * 0.85 + i * 0.7) * 0.004 * (0.2 + c.openAmount);
      const radialOffset = (c.openAmount * 0.42 + secondaryPush + drift) * windowBoost;

      clusterPos
        .copy(c.homePos)
        .addScaledVector(c.radialDir, radialOffset)
        .addScaledVector(c.tangentDir, c.openAmount * 0.05 * Math.sin(i * 2.1));

      const tiltAngle = c.openAmount * 0.22 * windowBoost * (i % 2 === 0 ? 1 : -0.85);
      clusterQuat.setFromAxisAngle(c.tangentDir, tiltAngle);

      // Apply individual 6-DOF target forces to each of the 4 convex-hull sub-shards
      const subSpreadDist = c.subShatter * 0.2;
      const subTilt = c.subShatter * 0.28;

      for (let sIdx = 0; sIdx < c.subShards.length; sIdx++) {
        const sub = c.subShards[sIdx];
        const tr = sub.body.translation();
        curPos.set(tr.x, tr.y, tr.z);

        // Local sub-shard offset + secondary cleavage separation rotated by cluster frame
        targetPos
          .copy(sub.localOffset)
          .addScaledVector(sub.outwardDir, subSpreadDist)
          .applyQuaternion(clusterQuat)
          .add(clusterPos);

        // Localized pointer wedge & direct shard tractor pull
        toShard.subVectors(curPos, ptrLocal);
        const ptrDistSq = toShard.lengthSq();
        const ptrWedge = Math.exp(-ptrDistSq * 2.3) * (0.045 + Math.min(0.14, pointer.speed * 0.035));
        targetPos.addScaledVector(c.radialDir, ptrWedge);

        if (isDragging && activeDragSubShard === sub) {
          // Direct tactile tractor: pulling a grabbed sub-shard tugs it toward the pointer in 3D
          targetPos.lerp(ptrLocal, 0.22);
        }

        forceVec.subVectors(targetPos, curPos).multiplyScalar(springK * dtSec);

        if (pointer.speed > 0.08) {
          const shearWeight = Math.exp(-ptrDistSq * 2.6) * dtSec * 0.26;
          forceVec.addScaledVector(pointer.velocity, shearWeight);
        }

        rapierVec.x = forceVec.x;
        rapierVec.y = forceVec.y;
        rapierVec.z = forceVec.z;
        sub.body.applyImpulse(rapierVec, true);

        // 6-DOF rotational spring-damper
        subRotQuat.setFromAxisAngle(sub.rotAxis, subTilt * (sIdx % 2 === 0 ? 1 : -0.8));
        targetQuat.copy(clusterQuat).multiply(subRotQuat);

        const rot = sub.body.rotation();
        curQuat.set(rot.x, rot.y, rot.z, rot.w);

        deltaQuat.copy(targetQuat).multiply(curQuat.invert());
        if (deltaQuat.w < 0) {
          deltaQuat.x = -deltaQuat.x;
          deltaQuat.y = -deltaQuat.y;
          deltaQuat.z = -deltaQuat.z;
          deltaQuat.w = -deltaQuat.w;
        }
        const sinHalf = Math.hypot(deltaQuat.x, deltaQuat.y, deltaQuat.z);
        if (sinHalf > 1e-4) {
          const angle = 2 * Math.atan2(sinHalf, deltaQuat.w);
          rotAxis.set(deltaQuat.x / sinHalf, deltaQuat.y / sinHalf, deltaQuat.z / sinHalf);
          torqueVec.copy(rotAxis).multiplyScalar(angle * rotSpringK * dtSec);
          rapierVec.x = torqueVec.x;
          rapierVec.y = torqueVec.y;
          rapierVec.z = torqueVec.z;
          sub.body.applyTorqueImpulse(rapierVec, true);
        }
      }
    }

    globalSpread = sumOpen / SHARD_COUNT;
    uGlobalOpen.value = globalSpread;
    uShockImpulse.value = shockPulse;
    coreLight.intensity = 1.6 + globalSpread * 3.8 + shockPulse * 2.4;

    // 3. Coupled Anchor-Point Kintsugi Bridge Forces (`applyImpulseAtPoint`)
    //    Pulling or perturbing any shard physically tugs & torques its neighbors through the golden resin network!
    const bridgeStiffness = 1.85 * dtSec;
    for (let bIdx = 0; bIdx < bridges.length; bIdx++) {
      const bridge = bridges[bIdx];
      const subA = bridge.subA;
      const subB = bridge.subB;

      worldA.copy(bridge.localA).applyQuaternion(subA.mesh.quaternion).add(subA.mesh.position);
      worldB.copy(bridge.localB).applyQuaternion(subB.mesh.quaternion).add(subB.mesh.position);

      dirThread.subVectors(worldB, worldA);
      const span = dirThread.length();
      const parentCluster = clusters[bridge.parentShardIdx];
      const eqSpan =
        bridge.tier === 1
          ? bridge.restSpan + parentCluster.openAmount * 0.28
          : bridge.restSpan + parentCluster.subShatter * 0.16;

      if (span > eqSpan + 0.01 && span > 1e-4) {
        const stretch = Math.min(0.25, span - eqSpan);
        dirThread.divideScalar(span).multiplyScalar(stretch * bridgeStiffness);

        rapierVec.x = dirThread.x;
        rapierVec.y = dirThread.y;
        rapierVec.z = dirThread.z;
        rapierVecNeg.x = -dirThread.x;
        rapierVecNeg.y = -dirThread.y;
        rapierVecNeg.z = -dirThread.z;

        rapierPtA.x = worldA.x;
        rapierPtA.y = worldA.y;
        rapierPtA.z = worldA.z;
        rapierPtB.x = worldB.x;
        rapierPtB.y = worldB.y;
        rapierPtB.z = worldB.z;

        subA.body.applyImpulseAtPoint(rapierVec, rapierPtA, true);
        subB.body.applyImpulseAtPoint(rapierVecNeg, rapierPtB, true);
      }
    }

    // Step Rapier 3D convex-hull physics world
    world.timestep = Math.min(0.033, dtSec);
    world.step();

    // Sync all 56 sub-shard meshes from their Rapier rigid bodies
    for (let k = 0; k < allSubShards.length; k++) {
      const sub = allSubShards[k];
      const tr = sub.body.translation();
      const rot = sub.body.rotation();
      sub.mesh.position.set(tr.x, tr.y, tr.z);
      sub.mesh.quaternion.set(rot.x, rot.y, rot.z, rot.w);
    }

    // 4. Update Kintsugi Resin Strands + Per-Strand Strain-Optic Telemetry (`instanceTension`)
    let instIdx = 0;
    for (let bIdx = 0; bIdx < bridges.length; bIdx++) {
      const bridge = bridges[bIdx];
      const parentCluster = clusters[bridge.parentShardIdx];

      // Hide Tier-2 internal resin threads until their parent shard begins to sub-shatter
      if (bridge.tier === 2 && parentCluster.subShatter < 0.015) {
        threadScale.set(0, 0, 0);
        threadMatrix.compose(geodeRig.position, threadQuat.identity(), threadScale);
        for (let s = 0; s < STRAND_SEGMENTS; s++) {
          tensionAttr.setXY(instIdx, 0, 0);
          threadInstanced.setMatrixAt(instIdx++, threadMatrix);
        }
        continue;
      }

      const subA = bridge.subA;
      const subB = bridge.subB;
      worldA.copy(bridge.localA).applyQuaternion(subA.mesh.quaternion).add(subA.mesh.position);
      worldB.copy(bridge.localB).applyQuaternion(subB.mesh.quaternion).add(subB.mesh.position);

      const totalSpan = worldA.distanceTo(worldB);
      const sagAmp = Math.min(0.075, totalSpan * 0.14);
      const normStrain = clampScalar((totalSpan - bridge.restSpan) / 0.36, 0.0, 1.0);
      const flashVal = Math.max(shockPulse, parentCluster.stressFlash);

      for (let s = 0; s < STRAND_SEGMENTS; s++) {
        const u0 = s / STRAND_SEGMENTS;
        const u1 = (s + 1) / STRAND_SEGMENTS;
        const uMid = (u0 + u1) * 0.5;

        const arch0 = 4.0 * u0 * (1.0 - u0);
        const arch1 = 4.0 * u1 * (1.0 - u1);
        const archMid = 4.0 * uMid * (1.0 - uMid);

        segStart.lerpVectors(worldA, worldB, u0).addScaledVector(bridge.sagDir, arch0 * sagAmp);
        segEnd.lerpVectors(worldA, worldB, u1).addScaledVector(bridge.sagDir, arch1 * sagAmp);

        midThread.addVectors(segStart, segEnd).multiplyScalar(0.5);
        dirThread.subVectors(segEnd, segStart);
        const segLen = dirThread.length();

        if (segLen > 1e-4) {
          dirThread.divideScalar(segLen);
          threadQuat.setFromUnitVectors(upVec, dirThread);
        } else {
          threadQuat.identity();
        }

        const neckProfile = (1.35 - 0.6 * archMid) * (bridge.tier === 2 ? 0.82 : 1.0);
        const stretchThinning = 1.0 / Math.sqrt(1.0 + totalSpan * 1.15);
        const radialScale = clampScalar(neckProfile * stretchThinning, 0.32, 1.6);

        // Strain concentrates highest at the necked mid-span (`archMid -> 1`)
        const segStrain = clampScalar(normStrain * (0.65 + 0.55 * archMid), 0.0, 1.0);
        tensionAttr.setXY(instIdx, segStrain, flashVal);

        threadScale.set(radialScale, Math.max(0.003, segLen * 1.06), radialScale);
        threadMatrix.compose(midThread, threadQuat, threadScale);
        threadInstanced.setMatrixAt(instIdx++, threadMatrix);
      }
    }
    tensionAttr.needsUpdate = true;
    threadInstanced.instanceMatrix.needsUpdate = true;

    // 5. Update Spalled Micro-Cleavage Mineral Chips in Slow-Time Suspension
    for (let k = 0; k < SPALL_CHIP_COUNT; k++) {
      const chip = spallChips[k];
      const host = clusters[chip.parentShardIdx];
      const chipActivity = clampScalar(
        (host.openAmount - 0.12) * 1.35 + host.subShatter * 0.65,
        0.0,
        1.0
      );

      if (chipActivity <= 0.01) {
        threadScale.set(0, 0, 0);
        threadMatrix.compose(geodeRig.position, threadQuat.identity(), threadScale);
        spallInstanced.setMatrixAt(k, threadMatrix);
        continue;
      }

      const floatDist = chipActivity * (0.22 + 0.06 * Math.sin(elapsedSec * 1.4 + chip.phase));
      targetPos
        .copy(chip.localOrigin)
        .addScaledVector(chip.ejectDir, floatDist)
        .addScaledVector(
          host.tangentDir,
          Math.cos(elapsedSec * 1.1 + chip.phase) * 0.04 * chipActivity
        );

      // Subtle pointer vortex swirl on suspended mineral chips
      toShard.subVectors(targetPos, ptrLocal);
      const dSq = toShard.lengthSq();
      if (dSq < 0.8) {
        targetPos.addScaledVector(toShard, Math.exp(-dSq * 3.0) * 0.06 * chipActivity);
      }

      threadQuat.setFromAxisAngle(chip.rotAxis, elapsedSec * 0.9 + chip.phase + chipActivity * 2.2);
      threadScale.copy(chip.scale).multiplyScalar(chipActivity);
      threadMatrix.compose(targetPos, threadQuat, threadScale);
      spallInstanced.setMatrixAt(k, threadMatrix);
    }
    spallInstanced.instanceMatrix.needsUpdate = true;

    // 6. Slow suspended studio presentation rotation + tactile pointer parallax
    const rigRotY = elapsedSec * 0.13 + pointer.ndc.x * 0.25;
    const rigRotX = -0.15 + pointer.ndc.y * 0.14 + Math.sin(elapsedSec * 0.55) * 0.02;
    geodeRig.rotation.set(rigRotX, rigRotY, 0.12);
    geodeRig.position.set(0, Math.sin(elapsedSec * 0.85) * 0.014, 0);
  });

  return stage;
}

function clampScalar(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay02Shatter();
}
