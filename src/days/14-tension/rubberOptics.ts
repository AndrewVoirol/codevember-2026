import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  clamp,
  exp,
  float,
  length,
  mix,
  positionLocal,
  select,
  smoothstep,
  step,
  uniform,
  uv,
  vec3,
} from 'three/tsl';
import { CHALKY_SLATE_LINEAR, type PigmentKey } from '../../core/palette';
import {
  createPigmentUniforms,
  hash31Tsl,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import {
  SEGMENTS_PER_BAND,
  type BandRuntime,
  type BandSpec,
  type DustPuff,
} from './types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

const VERTS_PER_RING = 4;
const MAX_RINGS = SEGMENTS_PER_BAND + 1;
const MAX_VERTS_PER_BAND = MAX_RINGS * VERTS_PER_RING;
const MAX_INDICES_PER_BAND = SEGMENTS_PER_BAND * 4 * 6;
const MAX_DUST_PUFFS = 384;

// Pre-allocated scratch buffers for zero-GC syncFromSim
const _posScratch = new Float32Array(MAX_RINGS * 3);
const _strainScratch = new Float32Array(MAX_RINGS);
const _dummyMat = new THREE.Matrix4();
const _dummyPos = new THREE.Vector3();
const _dummyQuat = new THREE.Quaternion();
const _dummyScale = new THREE.Vector3();

// Corner offsets around the 4-vertex beveled strap ring:
// 0: top-left (+R, -B), 1: top-right (+R, +B), 2: bot-right (-R, +B), 3: bot-left (-R, -B)
const CORNER_R = [1, 1, -1, -1] as const;
const CORNER_B = [-1, 1, 1, -1] as const;
// Beveled outward normal weights so the 4-vert strap catches rounded edge highlights
const NORM_R = [0.86, 0.86, -0.86, -0.86] as const;
const NORM_B = [-0.51, 0.51, 0.51, -0.51] as const;

export interface RubberBallSimReader {
  sampleBandCenterline: (idx: number, pos: Float32Array, strain: Float32Array) => number;
}

export interface RubberBallVisuals {
  group: THREE.Group;
  syncFromSim: (
    specs: readonly BandSpec[],
    runtimes: readonly BandRuntime[],
    puffs: readonly DustPuff[],
    sim: RubberBallSimReader,
    hoveredBandId: number | null,
    dtSec: number,
    cameraQuat?: THREE.Quaternion
  ) => void;
  dispose: () => void;
}

/**
 * Builds the mineral-dyed natural rubber TSL material for a given pigment.
 * Includes Kubelka-Munk pigment depth, crepe rubber micro-grain,
 * strain-induced stress-whitening, talcum/sulfur bloom in crevices/edges,
 * dithered fade for tumbling scraps, and subtle hover highlight.
 */
function createRubberBandMaterial(
  pigmentKey: PigmentKey,
  uHoveredBandId: TslNode,
  uTime: TslNode
): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.68,
    metalness: 0.02,
    side: THREE.DoubleSide,
    alphaTest: 0.5,
  });

  const pigment = createPigmentUniforms(pigmentKey);
  const kaolin = createPigmentUniforms('warmKaolin');
  const calcite = createPigmentUniforms('boneCalcite');

  const pos0 = positionLocal;
  const aStrain = attribute('aStrain', 'float');
  const aCrossUv = attribute('aCrossUv', 'vec2'); // x: across [-1..1], y: along [0..1]
  const aBandMeta = attribute('aBandMeta', 'vec3'); // x: bandId, y: opacity, z: batchSeed

  const bandId = aBandMeta.x;
  const opacity = aBandMeta.y;
  const batchSeed = aBandMeta.z;

  // Evaluate all procedural 3D noise in uniform control flow
  const crepeGrain = mineralToothFbm(pos0.add(vec3(batchSeed.mul(13.7))), float(56.0));
  const talcumNoise = mineralToothFbm(pos0.add(vec3(batchSeed.mul(5.3))), float(14.5));
  const stretchFiber = mineralToothFbm(
    vec3(aCrossUv.y.mul(44.0), aCrossUv.x.mul(7.0), bandId.mul(3.17)),
    float(1.0)
  );
  const ditherVal = hash31Tsl(pos0.mul(92.0).add(vec3(uTime.mul(0.01))));

  // Kubelka-Munk mineral-dyed body color over chalky slate backing
  const chalkBacking = vec3(
    CHALKY_SLATE_LINEAR[0],
    CHALKY_SLATE_LINEAR[1],
    CHALKY_SLATE_LINEAR[2]
  );
  const batchThickness = float(0.72)
    .add(batchSeed.sub(0.5).mul(0.18))
    .add(crepeGrain.sub(0.5).mul(0.14));
  const baseBody = kubelkaMunkGlaze(pigment.K, pigment.S, batchThickness, chalkBacking);

  // Subtle ambient occlusion / darkening toward inner edge of strap
  const edgeProximity = smoothstep(0.55, 1.0, abs(aCrossUv.x));
  const grainedBody = baseBody.mul(float(0.92).add(crepeGrain.mul(0.16)));

  // Natural latex stress-whitening: paler & warmer with longitudinal stretch streaks
  const strainFactor = smoothstep(0.12, 0.88, aStrain);
  const stretchStreak = smoothstep(0.28, 0.74, stretchFiber);
  const whitenMix = clamp(
    strainFactor.mul(0.58).add(strainFactor.mul(stretchStreak).mul(0.28)),
    0.0,
    0.84
  );
  const strainColor = mix(grainedBody, kaolin.rInf.mul(0.94), whitenMix);

  // Talcum / sulfur bloom settling along beveled strap edges and micro-pockets
  const bloomMask = clamp(
    edgeProximity.mul(0.22).add(smoothstep(0.58, 0.86, talcumNoise).mul(0.18)),
    0.0,
    0.34
  );
  const finalColor = mix(strainColor, calcite.rInf.mul(0.88), bloomMask);

  // Hover affordance: gentle warm mineral lift on hovered band
  const isHovered = smoothstep(0.5, 0.0, abs(bandId.sub(uHoveredBandId)));
  const hoverLift = calcite.rInf.mul(0.14).mul(isHovered);

  mat.colorNode = finalColor;
  mat.emissiveNode = hoverLift;

  // Roughness: matte crepe at rest (0.72), tighter taut sheen under strain (0.42), chalky at talcum bloom
  const tautRoughness = mix(float(0.72), float(0.42), strainFactor);
  const grainRough = tautRoughness.add(crepeGrain.sub(0.5).mul(0.14));
  mat.roughnessNode = clamp(mix(grainRough, float(0.92), bloomMask), 0.32, 0.96);

  // Crisp opaque depth-buffered rendering for active bands; dithered dissolve only when fading
  mat.opacityNode = select(
    opacity.greaterThan(0.99),
    float(1.0),
    step(ditherVal, opacity)
  );

  return mat;
}

/**
 * Builds the knotted rawUmber & boneCalcite starter core at radius ~0.28
 * revealed when outer bands are excavated.
 */
function createStarterCore(): {
  coreGroup: THREE.Group;
  dispose: () => void;
} {
  const coreGroup = new THREE.Group();
  coreGroup.name = 'starter-core';

  const rawUmber = createPigmentUniforms('rawUmber');
  const boneCalcite = createPigmentUniforms('boneCalcite');

  // Faceted / crumpled rubber nub at r ~ 0.24
  const nubGeom = new THREE.IcosahedronGeometry(0.235, 3);
  const posAttr = nubGeom.getAttribute('position') as THREE.BufferAttribute;
  const arr = posAttr.array as Float32Array;
  for (let i = 0; i < posAttr.count; i++) {
    const x = arr[i * 3];
    const y = arr[i * 3 + 1];
    const z = arr[i * 3 + 2];
    const fold =
      Math.sin(x * 14.0 + y * 9.0) * 0.018 +
      Math.cos(z * 16.0 - x * 11.0) * 0.014;
    const scale = 1.0 + fold;
    arr[i * 3] = x * scale;
    arr[i * 3 + 1] = y * scale;
    arr[i * 3 + 2] = z * scale;
  }
  nubGeom.computeVertexNormals();

  const nubMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.78,
    metalness: 0.02,
  });
  const nubTooth = mineralToothFbm(positionLocal, float(24.0));
  const nubTalc = smoothstep(0.56, 0.84, mineralToothFbm(positionLocal, float(11.0)));
  const nubBase = kubelkaMunkGlaze(
    rawUmber.K,
    rawUmber.S,
    float(0.85),
    vec3(CHALKY_SLATE_LINEAR[0], CHALKY_SLATE_LINEAR[1], CHALKY_SLATE_LINEAR[2])
  );
  nubMat.colorNode = mix(nubBase.mul(float(0.88).add(nubTooth.mul(0.24))), boneCalcite.rInf.mul(0.82), nubTalc.mul(0.35));
  nubMat.roughnessNode = clamp(float(0.74).add(nubTooth.mul(0.16)).add(nubTalc.mul(0.15)), 0.5, 0.96);

  const nubMesh = new THREE.Mesh(nubGeom, nubMat);
  nubMesh.castShadow = true;
  nubMesh.receiveShadow = true;
  coreGroup.add(nubMesh);

  // Tightly knotted inner starter loops around the core
  const knotMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.72,
    metalness: 0.02,
  });
  const knotTooth = mineralToothFbm(positionLocal, float(38.0));
  knotMat.colorNode = mix(boneCalcite.rInf.mul(0.85), rawUmber.rInf, smoothstep(0.4, 0.75, knotTooth));

  const knotGeoms: THREE.TorusGeometry[] = [];
  const knotConfigs: Array<[number, number, number, number, number]> = [
    [0.245, 0.022, 0.4, 0.8, 0.1],
    [0.255, 0.02, -0.7, 0.3, 1.1],
    [0.262, 0.019, 1.1, -0.5, -0.6],
    [0.268, 0.018, -0.2, 1.3, 0.7],
  ];

  for (let i = 0; i < knotConfigs.length; i++) {
    const [r, tube, rx, ry, rz] = knotConfigs[i];
    const g = new THREE.TorusGeometry(r, tube, 8, 28);
    g.scale(1.0, 1.0, 0.55);
    knotGeoms.push(g);
    const m = new THREE.Mesh(g, i % 2 === 0 ? nubMat : knotMat);
    m.rotation.set(rx, ry, rz);
    m.castShadow = true;
    m.receiveShadow = true;
    coreGroup.add(m);
  }

  return {
    coreGroup,
    dispose: () => {
      nubGeom.dispose();
      nubMat.dispose();
      knotMat.dispose();
      for (let i = 0; i < knotGeoms.length; i++) {
        knotGeoms[i].dispose();
      }
    },
  };
}

/**
 * Builds the shared index buffer for a 4-vertex-per-ring ribbon up to SEGMENTS_PER_BAND.
 */
function createSharedRibbonIndices(): THREE.BufferAttribute {
  const indices = new Uint16Array(MAX_INDICES_PER_BAND);
  let ptr = 0;
  for (let s = 0; s < SEGMENTS_PER_BAND; s++) {
    const r0 = s * VERTS_PER_RING;
    const r1 = (s + 1) * VERTS_PER_RING;
    for (let k = 0; k < VERTS_PER_RING; k++) {
      const kNext = (k + 1) % VERTS_PER_RING;
      const v0 = r0 + k;
      const v1 = r0 + kNext;
      const v0Next = r1 + k;
      const v1Next = r1 + kNext;

      indices[ptr++] = v0;
      indices[ptr++] = v0Next;
      indices[ptr++] = v1;

      indices[ptr++] = v1;
      indices[ptr++] = v0Next;
      indices[ptr++] = v1Next;
    }
  }
  return new THREE.BufferAttribute(indices, 1);
}

/**
 * Creates the Day 14 mineral-dyed rubber band ball visual layer.
 */
export function createRubberBallVisuals(specs: readonly BandSpec[]): RubberBallVisuals {
  const group = new THREE.Group();
  group.name = 'rubber-band-ball';

  const uHoveredBandId = uniform(-1.0);
  const uTime = uniform(0.0);

  // 1. Starter core
  const core = createStarterCore();
  group.add(core.coreGroup);

  // 2. Materials grouped by PigmentKey
  const materialsByPigment = new Map<PigmentKey, THREE.MeshStandardNodeMaterial>();
  for (let i = 0; i < specs.length; i++) {
    const p = specs[i].pigment;
    if (!materialsByPigment.has(p)) {
      materialsByPigment.set(p, createRubberBandMaterial(p, uHoveredBandId, uTime));
    }
  }

  const sharedIndex = createSharedRibbonIndices();

  interface BandMeshRecord {
    mesh: THREE.Mesh;
    geom: THREE.BufferGeometry;
    posAttr: THREE.BufferAttribute;
    normAttr: THREE.BufferAttribute;
    strainAttr: THREE.BufferAttribute;
    crossUvAttr: THREE.BufferAttribute;
    metaAttr: THREE.BufferAttribute;
    posArr: Float32Array;
    normArr: Float32Array;
    strainArr: Float32Array;
    metaArr: Float32Array;
  }

  const bandRecords: BandMeshRecord[] = [];
  const bandsGroup = new THREE.Group();
  bandsGroup.name = 'rubber-bands';
  group.add(bandsGroup);

  for (let b = 0; b < specs.length; b++) {
    const spec = specs[b];
    const posArr = new Float32Array(MAX_VERTS_PER_BAND * 3);
    const normArr = new Float32Array(MAX_VERTS_PER_BAND * 3);
    const strainArr = new Float32Array(MAX_VERTS_PER_BAND);
    const crossUvArr = new Float32Array(MAX_VERTS_PER_BAND * 2);
    const metaArr = new Float32Array(MAX_VERTS_PER_BAND * 3);

    // Deterministic batch seed in [0, 1]
    const batchSeed = ((Math.sin(spec.id * 12.9898 + spec.layer * 78.233) * 43758.5453) % 1 + 1) % 1;

    for (let r = 0; r < MAX_RINGS; r++) {
      const uAlong = r / SEGMENTS_PER_BAND;
      for (let k = 0; k < VERTS_PER_RING; k++) {
        const vIdx = r * VERTS_PER_RING + k;
        crossUvArr[vIdx * 2] = CORNER_B[k];
        crossUvArr[vIdx * 2 + 1] = uAlong;

        metaArr[vIdx * 3] = spec.id;
        metaArr[vIdx * 3 + 1] = 1.0;
        metaArr[vIdx * 3 + 2] = batchSeed;
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setIndex(sharedIndex);

    const posAttr = new THREE.BufferAttribute(posArr, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    const normAttr = new THREE.BufferAttribute(normArr, 3);
    normAttr.setUsage(THREE.DynamicDrawUsage);
    const strainAttr = new THREE.BufferAttribute(strainArr, 1);
    strainAttr.setUsage(THREE.DynamicDrawUsage);
    const crossUvAttr = new THREE.BufferAttribute(crossUvArr, 2);
    const metaAttr = new THREE.BufferAttribute(metaArr, 3);
    metaAttr.setUsage(THREE.DynamicDrawUsage);

    geom.setAttribute('position', posAttr);
    geom.setAttribute('normal', normAttr);
    geom.setAttribute('aStrain', strainAttr);
    geom.setAttribute('aCrossUv', crossUvAttr);
    geom.setAttribute('aBandMeta', metaAttr);

    // Conservative bounding sphere so frustum culling never clips pulled bands
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.0);

    const mat = materialsByPigment.get(spec.pigment)!;
    const mesh = new THREE.Mesh(geom, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.userData.bandId = spec.id;
    bandsGroup.add(mesh);

    bandRecords.push({
      mesh,
      geom,
      posAttr,
      normAttr,
      strainAttr,
      crossUvAttr,
      metaAttr,
      posArr,
      normArr,
      strainArr,
      metaArr,
    });
  }

  // 3. Instanced billboarded talcum dust puffs (soft Gaussian + FBM wispy erosion)
  const puffGeom = new THREE.PlaneGeometry(2.0, 2.0);
  const puffLifeArr = new Float32Array(MAX_DUST_PUFFS);
  const puffLifeAttr = new THREE.InstancedBufferAttribute(puffLifeArr, 1);
  puffLifeAttr.setUsage(THREE.DynamicDrawUsage);
  puffGeom.setAttribute('aPuffLife', puffLifeAttr);

  const puffMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const aPuffLife = attribute('aPuffLife', 'float');
  const uvCentered = uv().sub(0.5).mul(2.0);
  const radialDist = length(uvCentered);
  const puffWisp = mineralToothFbm(vec3(uvCentered, float(0.35)), float(3.8));
  const softDisk = exp(radialDist.mul(radialDist).mul(-3.8)).mul(
    smoothstep(1.0, 0.25, radialDist.add(puffWisp.sub(0.5).mul(0.45)))
  );
  puffMat.colorNode = mix(kaolin.rInf, calcite.rInf, puffWisp);
  puffMat.opacityNode = softDisk.mul(aPuffLife).mul(aPuffLife).mul(0.42);

  const puffMesh = new THREE.InstancedMesh(puffGeom, puffMat, MAX_DUST_PUFFS);
  puffMesh.count = 0;
  puffMesh.frustumCulled = false;
  puffMesh.castShadow = false;
  puffMesh.receiveShadow = false;
  group.add(puffMesh);

  let elapsedSec = 0;

  function syncFromSim(
    syncSpecs: readonly BandSpec[],
    runtimes: readonly BandRuntime[],
    puffs: readonly DustPuff[],
    sim: RubberBallSimReader,
    hoveredBandId: number | null,
    dtSec: number,
    cameraQuat?: THREE.Quaternion
  ): void {
    elapsedSec += dtSec;
    uTime.value = elapsedSec;
    uHoveredBandId.value = hoveredBandId !== null ? hoveredBandId : -1.0;

    // Update each band ribbon in-place with zero allocations
    const count = Math.min(syncSpecs.length, bandRecords.length);
    for (let b = 0; b < count; b++) {
      const spec = syncSpecs[b];
      const rt = runtimes[b];
      const rec = bandRecords[b];

      if (!rt || rt.status === 'gone' || rt.opacity <= 0.005) {
        rec.mesh.visible = false;
        continue;
      }

      const activeSegments = sim.sampleBandCenterline(b, _posScratch, _strainScratch);
      if (activeSegments <= 0) {
        rec.mesh.visible = false;
        continue;
      }

      const clampedSegs = activeSegments > SEGMENTS_PER_BAND ? SEGMENTS_PER_BAND : activeSegments;
      const ringCount = clampedSegs + 1;

      rec.mesh.visible = true;
      rec.geom.setDrawRange(0, clampedSegs * 4 * 6);

      const fallX = rt.fallOffset[0];
      const fallY = rt.fallOffset[1];
      const fallZ = rt.fallOffset[2];
      const halfWBase = spec.width * 0.5;
      const halfTBase = spec.thickness * 0.5;
      const op = rt.opacity;

      const posArr = rec.posArr;
      const normArr = rec.normArr;
      const strainArr = rec.strainArr;
      const metaArr = rec.metaArr;

      for (let r = 0; r < ringCount; r++) {
        const pIdx = r * 3;
        const px = _posScratch[pIdx];
        const py = _posScratch[pIdx + 1];
        const pz = _posScratch[pIdx + 2];

        // Central-difference tangent T
        const prevIdx = (r > 0 ? r - 1 : 0) * 3;
        const nextIdx = (r < ringCount - 1 ? r + 1 : ringCount - 1) * 3;
        let tx = _posScratch[nextIdx] - _posScratch[prevIdx];
        let ty = _posScratch[nextIdx + 1] - _posScratch[prevIdx + 1];
        let tz = _posScratch[nextIdx + 2] - _posScratch[prevIdx + 2];
        const tLen = Math.hypot(tx, ty, tz);
        if (tLen > 1e-6) {
          tx /= tLen;
          ty /= tLen;
          tz /= tLen;
        } else {
          tx = spec.tangentU[0];
          ty = spec.tangentU[1];
          tz = spec.tangentU[2];
        }

        // Radial vector relative to band's falling center
        let rx = px - fallX;
        let ry = py - fallY;
        let rz = pz - fallZ;
        const rLen = Math.hypot(rx, ry, rz);
        if (rLen > 1e-5) {
          rx /= rLen;
          ry /= rLen;
          rz /= rLen;
        } else {
          rx = spec.tangentV[0];
          ry = spec.tangentV[1];
          rz = spec.tangentV[2];
        }

        // Lateral binormal B = R x T
        let bx = ry * tz - rz * ty;
        let by = rz * tx - rx * tz;
        let bz = rx * ty - ry * tx;
        let bLen = Math.hypot(bx, by, bz);
        if (bLen < 1e-4) {
          bx = spec.normal[0];
          by = spec.normal[1];
          bz = spec.normal[2];
          bLen = Math.hypot(bx, by, bz) || 1.0;
        }
        bx /= bLen;
        by /= bLen;
        bz /= bLen;

        // Orthonormalize radial normal R = T x B
        rx = ty * bz - tz * by;
        ry = tz * bx - tx * bz;
        rz = tx * by - ty * bx;
        const rOrthoLen = Math.hypot(rx, ry, rz) || 1.0;
        rx /= rOrthoLen;
        ry /= rOrthoLen;
        rz /= rOrthoLen;

        // Poisson necking: width and thickness thin noticeably as strain rises above rest
        const strain = _strainScratch[r];
        const extraStrain = strain > spec.restStrain ? strain - spec.restStrain : 0.0;
        const widthScale = Math.max(0.32, 1.0 / Math.sqrt(1.0 + extraStrain * 2.1));
        const thickScale = Math.max(0.45, 1.0 / Math.sqrt(1.0 + extraStrain * 1.2));
        const hw = halfWBase * widthScale;
        const ht = halfTBase * thickScale;

        for (let k = 0; k < VERTS_PER_RING; k++) {
          const vIdx = r * VERTS_PER_RING + k;
          const v3 = vIdx * 3;
          const cr = CORNER_R[k] * ht;
          const cb = CORNER_B[k] * hw;

          posArr[v3] = px + rx * cr + bx * cb;
          posArr[v3 + 1] = py + ry * cr + by * cb;
          posArr[v3 + 2] = pz + rz * cr + bz * cb;

          const nr = NORM_R[k];
          const nb = NORM_B[k];
          normArr[v3] = rx * nr + bx * nb;
          normArr[v3 + 1] = ry * nr + by * nb;
          normArr[v3 + 2] = rz * nr + bz * nb;

          strainArr[vIdx] = strain;
          metaArr[v3 + 1] = op;
        }
      }

      rec.posAttr.needsUpdate = true;
      rec.normAttr.needsUpdate = true;
      rec.strainAttr.needsUpdate = true;
      rec.metaAttr.needsUpdate = true;
    }

    // Update talcum dust puffs
    let activePuffs = 0;
    const puffCap = Math.min(puffs.length, MAX_DUST_PUFFS);
    for (let i = 0; i < puffCap; i++) {
      const p = puffs[i];
      if (!p.active || p.maxAgeSec <= 0) continue;
      const life = 1.0 - p.ageSec / p.maxAgeSec;
      if (life <= 0.005) continue;

      const s = p.size * (0.75 + (1.0 - life) * 2.1);
      _dummyPos.set(p.pos[0], p.pos[1], p.pos[2]);
      if (cameraQuat) {
        _dummyQuat.copy(cameraQuat);
      } else {
        _dummyQuat.identity();
      }
      _dummyScale.set(s, s, s);
      _dummyMat.compose(_dummyPos, _dummyQuat, _dummyScale);
      puffMesh.setMatrixAt(activePuffs, _dummyMat);
      puffLifeArr[activePuffs] = life;
      activePuffs++;
    }

    puffMesh.count = activePuffs;
    if (activePuffs > 0) {
      puffMesh.instanceMatrix.needsUpdate = true;
      puffLifeAttr.needsUpdate = true;
    }
  }

  function dispose(): void {
    core.dispose();
    for (let i = 0; i < bandRecords.length; i++) {
      bandRecords[i].geom.dispose();
    }
    materialsByPigment.forEach((mat) => mat.dispose());
    materialsByPigment.clear();
    puffGeom.dispose();
    puffMat.dispose();
  }

  return {
    group,
    syncFromSim,
    dispose,
  };
}
