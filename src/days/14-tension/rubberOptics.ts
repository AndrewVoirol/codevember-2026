import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  clamp,
  exp,
  float,
  length,
  max,
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

const VERTS_PER_RING = 8;
const MAX_RINGS = SEGMENTS_PER_BAND + 1;
const MAX_VERTS_PER_BAND = MAX_RINGS * VERTS_PER_RING;
const MAX_INDICES_PER_BAND = SEGMENTS_PER_BAND * VERTS_PER_RING * 6;
const MAX_DUST_PUFFS = 384;

// Pre-allocated scratch buffers for zero-GC syncFromSim
const _posScratch = new Float32Array(MAX_RINGS * 3);
const _strainScratch = new Float32Array(MAX_RINGS);
const _dummyMat = new THREE.Matrix4();
const _dummyPos = new THREE.Vector3();
const _dummyQuat = new THREE.Quaternion();
const _dummyScale = new THREE.Vector3();

// 8-vertex chamfered strap cross-section in (radial R, binormal B):
// Wide flat outer/inner faces with crisp beveled side shoulders for specular edge separation
const CORNER_R = [1.0, 1.0, 0.52, -0.52, -1.0, -1.0, -0.52, 0.52] as const;
const CORNER_B = [-0.78, 0.78, 1.0, 1.0, 0.78, -0.78, -1.0, -1.0] as const;
// Smooth outward normals around the chamfered profile
const NORM_R = [0.96, 0.96, 0.48, -0.48, -0.96, -0.96, -0.48, 0.48] as const;
const NORM_B = [-0.28, 0.28, 0.88, 0.88, 0.28, -0.28, -0.88, -0.88] as const;

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
 * contact/underside ambient occlusion for strand separation,
 * strain-induced stress-whitening & crazing, snap-tip flash,
 * talcum/sulfur bloom in crevices/edges, and subtle hover highlight.
 */
function createRubberBandMaterial(
  pigmentKey: PigmentKey,
  uHoveredBandId: TslNode,
  uTime: TslNode
): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.72,
    metalness: 0.02,
    side: THREE.DoubleSide,
    alphaTest: 0.5,
  });

  const pigment = createPigmentUniforms(pigmentKey);
  const kaolin = createPigmentUniforms('warmKaolin');
  const calcite = createPigmentUniforms('boneCalcite');

  const pos0 = positionLocal;
  const aStrain = attribute('aStrain', 'float');
  // x: across [-1..1], y: along [0..1], z: radialFace [-1..1] (+1 outer, -1 inner)
  const aCrossUv = attribute('aCrossUv', 'vec3');
  // x: bandId, y: opacity, z: batchSeed, w: snapTipFlash
  const aBandMeta = attribute('aBandMeta', 'vec4');

  const bandId = aBandMeta.x;
  const opacity = aBandMeta.y;
  const batchSeed = aBandMeta.z;
  const snapTipFlash = aBandMeta.w;

  // Evaluate all procedural 3D noise in uniform control flow
  const crepeGrain = mineralToothFbm(pos0.add(vec3(batchSeed.mul(13.7))), float(62.0));
  const talcumNoise = mineralToothFbm(pos0.add(vec3(batchSeed.mul(5.3))), float(15.5));
  const stretchFiber = mineralToothFbm(
    vec3(aCrossUv.y.mul(48.0), aCrossUv.x.mul(7.5), bandId.mul(3.17)),
    float(1.0)
  );
  const crazingNoise = mineralToothFbm(
    vec3(aCrossUv.y.mul(115.0), aCrossUv.x.mul(18.0), bandId.mul(5.41)),
    float(1.0)
  );
  const ditherVal = hash31Tsl(pos0.mul(92.0).add(vec3(uTime.mul(0.01))));

  // Kubelka-Munk mineral-dyed body color over chalky slate backing
  const chalkBacking = vec3(
    CHALKY_SLATE_LINEAR[0],
    CHALKY_SLATE_LINEAR[1],
    CHALKY_SLATE_LINEAR[2]
  );
  const batchThickness = float(0.74)
    .add(batchSeed.sub(0.5).mul(0.24))
    .add(crepeGrain.sub(0.5).mul(0.16));
  const baseBody = kubelkaMunkGlaze(pigment.K, pigment.S, batchThickness, chalkBacking);

  // Per-band subtle value variation + crepe micro-tooth for strand separation
  const batchTone = float(0.92).add(batchSeed.mul(0.16));
  const edgeProximity = smoothstep(0.55, 1.0, abs(aCrossUv.x));

  // Contact / crevice ambient occlusion: inner underside (-1) is deeply shaded,
  // side bevels (~0.52) get crisp edge-shadow separation, outer face (+1) stays clean
  const radialAo = mix(float(0.54), float(1.0), smoothstep(-0.85, 0.75, aCrossUv.z));
  const bevelShadow = mix(float(1.0), float(0.84), edgeProximity);
  const grainedBody = baseBody
    .mul(batchTone)
    .mul(float(0.91).add(crepeGrain.mul(0.18)))
    .mul(radialAo)
    .mul(bevelShadow);

  // Natural latex stress-whitening & high-strain micro-crazing
  const strainFactor = smoothstep(0.12, 0.88, aStrain);
  const highStrain = smoothstep(0.52, 0.96, aStrain);
  const stretchStreak = smoothstep(0.26, 0.74, stretchFiber);
  const crazingFissure = smoothstep(0.54, 0.82, crazingNoise).mul(highStrain);

  const whitenMix = clamp(
    strainFactor
      .mul(0.54)
      .add(strainFactor.mul(stretchStreak).mul(0.28))
      .add(crazingFissure.mul(0.36)),
    0.0,
    0.88
  );
  const crazedColor = mix(grainedBody, kaolin.rInf.mul(0.95), whitenMix);

  // Snap-tip break-point crazing flash (brief bright kaolin/calcite highlight at severed tips)
  const tipFlashColor = mix(crazedColor, calcite.rInf, clamp(snapTipFlash.mul(0.85), 0.0, 0.85));

  // Talcum / sulfur bloom settling along beveled strap edges and micro-pockets
  const bloomMask = clamp(
    edgeProximity.mul(0.24).add(smoothstep(0.58, 0.86, talcumNoise).mul(0.18)),
    0.0,
    0.36
  );
  const finalColor = mix(tipFlashColor, calcite.rInf.mul(0.9), bloomMask);

  // Hover affordance: gentle warm mineral lift on hovered band
  const isHovered = smoothstep(0.5, 0.0, abs(bandId.sub(uHoveredBandId)));
  const hoverLift = calcite.rInf.mul(0.14).mul(isHovered).add(calcite.rInf.mul(0.18).mul(snapTipFlash));

  mat.colorNode = finalColor;
  mat.emissiveNode = hoverLift;

  // Roughness: matte vulcanized crepe at rest (0.76), taut rubber sheen under strain (0.38),
  // chalky rough in crazed micro-fissures & talcum bloom
  const tautRoughness = mix(float(0.76), float(0.38), strainFactor);
  const grainRough = tautRoughness
    .add(crepeGrain.sub(0.5).mul(0.14))
    .add(crazingFissure.mul(0.22));
  mat.roughnessNode = clamp(mix(grainRough, float(0.93), bloomMask), 0.3, 0.96);

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
  nubMat.colorNode = mix(
    nubBase.mul(float(0.88).add(nubTooth.mul(0.24))),
    boneCalcite.rInf.mul(0.82),
    nubTalc.mul(0.35)
  );
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
 * Builds the shared index buffer for an 8-vertex-per-ring ribbon up to SEGMENTS_PER_BAND.
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
    const crossUvArr = new Float32Array(MAX_VERTS_PER_BAND * 3);
    const metaArr = new Float32Array(MAX_VERTS_PER_BAND * 4);

    // Deterministic batch seed in [0, 1]
    const batchSeed = (((Math.sin(spec.id * 12.9898 + spec.layer * 78.233) * 43758.5453) % 1) + 1) % 1;

    for (let r = 0; r < MAX_RINGS; r++) {
      const uAlong = r / SEGMENTS_PER_BAND;
      for (let k = 0; k < VERTS_PER_RING; k++) {
        const vIdx = r * VERTS_PER_RING + k;
        crossUvArr[vIdx * 3] = CORNER_B[k];
        crossUvArr[vIdx * 3 + 1] = uAlong;
        crossUvArr[vIdx * 3 + 2] = CORNER_R[k];

        metaArr[vIdx * 4] = spec.id;
        metaArr[vIdx * 4 + 1] = 1.0;
        metaArr[vIdx * 4 + 2] = batchSeed;
        metaArr[vIdx * 4 + 3] = 0.0;
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
    const crossUvAttr = new THREE.BufferAttribute(crossUvArr, 3);
    const metaAttr = new THREE.BufferAttribute(metaArr, 4);
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

  // 3. Instanced billboarded talcum dust puffs (crisp mineral specks + wispy billowing clouds)
  const puffGeom = new THREE.PlaneGeometry(2.0, 2.0);
  // x: life [0..1], y: isSpeck [0|1], z: per-instance seed [0..1]
  const puffMetaArr = new Float32Array(MAX_DUST_PUFFS * 3);
  const puffMetaAttr = new THREE.InstancedBufferAttribute(puffMetaArr, 3);
  puffMetaAttr.setUsage(THREE.DynamicDrawUsage);
  puffGeom.setAttribute('aPuffMeta', puffMetaAttr);

  const puffMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const calcite = createPigmentUniforms('boneCalcite');
  const kaolin = createPigmentUniforms('warmKaolin');
  const aPuffMeta = attribute('aPuffMeta', 'vec3');
  const puffLife = aPuffMeta.x;
  const isSpeck = aPuffMeta.y;
  const puffSeed = aPuffMeta.z;

  const uvCentered = uv().sub(0.5).mul(2.0);
  const radialDist = length(uvCentered);
  const puffWisp = mineralToothFbm(
    vec3(uvCentered, puffSeed.mul(17.3).add(0.35)),
    float(3.8)
  );

  // Billowing plume vs crisp talc granule/flake
  const plumeAlpha = exp(radialDist.mul(radialDist).mul(-2.7))
    .mul(smoothstep(1.0, 0.16, radialDist.add(puffWisp.sub(0.5).mul(0.58))))
    .mul(puffLife.mul(puffLife).mul(0.54));

  const speckAlpha = smoothstep(0.68, 0.2, radialDist.add(puffWisp.sub(0.5).mul(0.52)))
    .mul(puffLife)
    .mul(0.84);

  puffMat.colorNode = mix(
    mix(kaolin.rInf, calcite.rInf, puffWisp),
    calcite.rInf.mul(1.08),
    isSpeck
  );
  puffMat.opacityNode = max(float(0.0), mix(plumeAlpha, speckAlpha, isSpeck));

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
      const isSnapping = rt.status === 'snapping';
      const isClosedLoop = !isSnapping && clampedSegs === SEGMENTS_PER_BAND;

      rec.mesh.visible = true;
      rec.geom.setDrawRange(0, clampedSegs * VERTS_PER_RING * 6);

      const fallX = rt.fallOffset[0];
      const fallY = rt.fallOffset[1];
      const fallZ = rt.fallOffset[2];
      const halfWBase = spec.width * 0.5;
      const halfTBase = spec.thickness * 0.5;
      const op = rt.opacity;
      const snapFlash = isSnapping && rt.curlAmount < 0.28 ? 1.0 - rt.curlAmount / 0.28 : 0.0;

      const posArr = rec.posArr;
      const normArr = rec.normArr;
      const strainArr = rec.strainArr;
      const metaArr = rec.metaArr;

      for (let r = 0; r < ringCount; r++) {
        // For closed loops, copy ring 0 verbatim to final ring so the seam is bit-identical
        if (isClosedLoop && r === ringCount - 1) {
          const dstStart = r * VERTS_PER_RING;
          for (let k = 0; k < VERTS_PER_RING; k++) {
            const dstIdx = dstStart + k;
            const dst3 = dstIdx * 3;
            const src3 = k * 3;
            posArr[dst3] = posArr[src3];
            posArr[dst3 + 1] = posArr[src3 + 1];
            posArr[dst3 + 2] = posArr[src3 + 2];
            normArr[dst3] = normArr[src3];
            normArr[dst3 + 1] = normArr[src3 + 1];
            normArr[dst3 + 2] = normArr[src3 + 2];
            strainArr[dstIdx] = strainArr[k];
            const dst4 = dstIdx * 4;
            metaArr[dst4 + 1] = op;
            metaArr[dst4 + 3] = 0.0;
          }
          continue;
        }

        const pIdx = r * 3;
        const px = _posScratch[pIdx];
        const py = _posScratch[pIdx + 1];
        const pz = _posScratch[pIdx + 2];

        // Central-difference tangent T (wrapped for closed loops to avoid u=0 normal kink)
        const prevIdx = isClosedLoop
          ? (r > 0 ? r - 1 : ringCount - 2) * 3
          : (r > 0 ? r - 1 : 0) * 3;
        const nextIdx = isClosedLoop
          ? (r < ringCount - 1 ? r + 1 : 1) * 3
          : (r < ringCount - 1 ? r + 1 : ringCount - 1) * 3;

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
        let hw = halfWBase * widthScale;
        let ht = halfTBase * thickScale;

        let tipFlash = 0.0;
        if (isSnapping) {
          const frac = clampedSegs > 0 ? r / clampedSegs : 0.5;
          const endDist = Math.abs(frac - 0.5) * 2.0;
          if (endDist > 0.65) {
            tipFlash = snapFlash * ((endDist - 0.65) / 0.35);
          }
          // Pinch cut ends cleanly so snapping tips look like solid severed rubber
          if (r === 0 || r === ringCount - 1) {
            ht *= 0.22;
            hw *= 0.78;
          }
        }

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
          const v4 = vIdx * 4;
          metaArr[v4 + 1] = op;
          metaArr[v4 + 3] = tipFlash;
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

      const isSpeck = p.size < 0.012 ? 1.0 : 0.0;
      const expand = isSpeck > 0.5 ? 0.9 + (1.0 - life) * 0.6 : 0.8 + (1.0 - life) * 2.5;
      const s = p.size * expand;

      _dummyPos.set(p.pos[0], p.pos[1], p.pos[2]);
      if (cameraQuat) {
        _dummyQuat.copy(cameraQuat);
      } else {
        _dummyQuat.identity();
      }
      _dummyScale.set(s, s, s);
      _dummyMat.compose(_dummyPos, _dummyQuat, _dummyScale);
      puffMesh.setMatrixAt(activePuffs, _dummyMat);

      const m3 = activePuffs * 3;
      puffMetaArr[m3] = life;
      puffMetaArr[m3 + 1] = isSpeck;
      puffMetaArr[m3 + 2] = ((i * 0.61803398875) % 1 + 1) % 1;
      activePuffs++;
    }

    puffMesh.count = activePuffs;
    if (activePuffs > 0) {
      puffMesh.instanceMatrix.needsUpdate = true;
      puffMetaAttr.needsUpdate = true;
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
