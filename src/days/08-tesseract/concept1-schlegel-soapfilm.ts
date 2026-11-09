import * as THREE from 'three/webgpu';
import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  cos,
  dot,
  exp,
  float,
  length,
  max,
  mix,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  sin,
  smoothstep,
  uniform,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

/**
 * Day 08 — Tesseract (Concept 8A — Pure Stereographic S³ Soap-Bubble & Thin-Film Polychoron):
 * "Stereographic S³ Plateau Soap-Bubble & Iridescent Thin-Film Polychoron"
 *
 * Pure 4D soap-bubble foam & Airy thin-film interference optics:
 * - The 16 vertices, 32 Plateau border struts, and 24 minimal-surface soap-film
 *   membranes of the 4D hypercube are projected onto the 3-sphere S³, rotated in
 *   SO(4), and stereographically projected into R³.
 * - Pressurized Inner Bubble Cushion & Catenoid Laminae: Each of the 24 soap-film
 *   membranes (`14x14` grid) relaxes along a smooth Plateau minimal-surface
 *   curvature profile while bulging with subtle 4D cell pressure differentials
 *   so the inner cell reads as a taut, pressurized spherical soap bubble suspended
 *   by 24 catenoid liquid sheets.
 * - Authentic Soap-Bubble Optics (`soapFilmMat`):
 *   - Crystal-clear interior transmission at normal incidence (`opacity ~ 0.08`)
 *     transitioning to luminous grazing Fresnel reflections (`(1 - |N·V|)^2.4`) and
 *     liquid Plateau border menisci (`aFaceMeta.x -> 1`) where 3 films meet along
 *     each curved strut.
 *   - Multi-order Airy thin-film optical path interference (`2 * n * d * cos(theta)`)
 *     cycling across `lapisUltramarine`, `celadonGlauconite`, `sporeSulfur`,
 *     `realgarAmber`, `cinnabarVermilion`, and `boneCalcite`, complete with
 *     gravity-drained thinner apex zones and pointer-driven Marangoni vortex swirls.
 *   - Analytical curved studio softbox window reflections evaluated directly on the
 *     bubble surface normals so every membrane catches crisp wet specular highlights.
 */

// ---------------------------------------------------------------------------
// 4D Tesseract Topology Constants & Pre-Allocated Scratch Buffers (Zero GC)
// ---------------------------------------------------------------------------
const NUM_VERTICES = 16;
const NUM_EDGES = 32;
const NUM_FACES = 24;

const STRUT_AXIAL_SEGMENTS = 28;
const STRUT_RADIAL_SEGMENTS = 12;
const VERTS_PER_STRUT = (STRUT_AXIAL_SEGMENTS + 1) * (STRUT_RADIAL_SEGMENTS + 1);
const TOTAL_STRUT_VERTS = NUM_EDGES * VERTS_PER_STRUT;

const FACE_GRID_STEPS = 14;
const VERTS_PER_FACE = (FACE_GRID_STEPS + 1) * (FACE_GRID_STEPS + 1);
const TOTAL_FACE_VERTS = NUM_FACES * VERTS_PER_FACE;

// Base 16 vertices in [-1, 1]^4
const _baseVerts4D = new Float32Array(NUM_VERTICES * 4);
for (let i = 0; i < NUM_VERTICES; i++) {
  _baseVerts4D[i * 4 + 0] = i & 1 ? 1 : -1;
  _baseVerts4D[i * 4 + 1] = i & 2 ? 1 : -1;
  _baseVerts4D[i * 4 + 2] = i & 4 ? 1 : -1;
  _baseVerts4D[i * 4 + 3] = i & 8 ? 1 : -1;
}

// 32 edges (pairs of vertex indices differing in 1 bit)
const _edgePairs = new Uint8Array(NUM_EDGES * 2);
{
  let eIdx = 0;
  for (let i = 0; i < NUM_VERTICES; i++) {
    for (let bit = 0; bit < 4; bit++) {
      const j = i ^ (1 << bit);
      if (i < j) {
        _edgePairs[eIdx * 2 + 0] = i;
        _edgePairs[eIdx * 2 + 1] = j;
        eIdx++;
      }
    }
  }
}

// 24 square faces (4 vertex indices ordered cyclically around each square)
const _faceQuads = new Uint8Array(NUM_FACES * 4);
{
  let fIdx = 0;
  for (let d0 = 0; d0 < 4; d0++) {
    for (let d1 = d0 + 1; d1 < 4; d1++) {
      const freeAxes: number[] = [];
      for (let a = 0; a < 4; a++) {
        if (a !== d0 && a !== d1) freeAxes.push(a);
      }
      const uAxis = freeAxes[0];
      const vAxis = freeAxes[1];

      for (let s0 = 0; s0 < 2; s0++) {
        for (let s1 = 0; s1 < 2; s1++) {
          const baseMask = (s0 << d0) | (s1 << d1);
          const v00 = baseMask;
          const v10 = baseMask | (1 << uAxis);
          const v11 = baseMask | (1 << uAxis) | (1 << vAxis);
          const v01 = baseMask | (1 << vAxis);
          _faceQuads[fIdx * 4 + 0] = v00;
          _faceQuads[fIdx * 4 + 1] = v10;
          _faceQuads[fIdx * 4 + 2] = v11;
          _faceQuads[fIdx * 4 + 3] = v01;
          fIdx++;
        }
      }
    }
  }
}

// Pre-allocated 4D rotation & projection scratch arrays
const _rotMat4D = new Float32Array(16);
const _tmpMat4D = new Float32Array(16);

// Pre-allocated Strut curve spinal samples per edge
const _spinePoints = new Float32Array((STRUT_AXIAL_SEGMENTS + 1) * 3);
const _spineRadii = new Float32Array(STRUT_AXIAL_SEGMENTS + 1);
const _spineW = new Float32Array(STRUT_AXIAL_SEGMENTS + 1);
const _radialCos = new Float32Array(STRUT_RADIAL_SEGMENTS + 1);
const _radialSin = new Float32Array(STRUT_RADIAL_SEGMENTS + 1);
for (let r = 0; r <= STRUT_RADIAL_SEGMENTS; r++) {
  const theta = (r / STRUT_RADIAL_SEGMENTS) * Math.PI * 2;
  _radialCos[r] = Math.cos(theta);
  _radialSin[r] = Math.sin(theta);
}

// Pre-allocated Strut & Face geometry attribute buffers
const _strutPositions = new Float32Array(TOTAL_STRUT_VERTS * 3);
const _strutNormals = new Float32Array(TOTAL_STRUT_VERTS * 3);
const _strutMeta = new Float32Array(TOTAL_STRUT_VERTS * 3); // x: axial t [0..1], y: w4D, z: junctionFlare [0..1]

const _facePositions = new Float32Array(TOTAL_FACE_VERTS * 3);
const _faceNormals = new Float32Array(TOTAL_FACE_VERTS * 3);
const _faceMeta = new Float32Array(TOTAL_FACE_VERTS * 3); // x: meniscus edge factor [0=center..1=edge], y: w4D, z: uCoord

// Pre-allocated Vector3 scratch objects
const _tangent = new THREE.Vector3();
const _normal = new THREE.Vector3();
const _binormal = new THREE.Vector3();
const _upRef = new THREE.Vector3();
const _dU = new THREE.Vector3();
const _dV = new THREE.Vector3();
const _faceNorm = new THREE.Vector3();

// ---------------------------------------------------------------------------
// 4D Givens Matrix & Stereographic Projection Helpers (Zero Allocation)
// ---------------------------------------------------------------------------
function setIdentity4D(m: Float32Array): void {
  m.fill(0);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
}

function applyGivensRotation4D(m: Float32Array, axisA: number, axisB: number, angle: number): void {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  _tmpMat4D.set(m);
  for (let row = 0; row < 4; row++) {
    const va = _tmpMat4D[row * 4 + axisA];
    const vb = _tmpMat4D[row * 4 + axisB];
    m[row * 4 + axisA] = c * va - s * vb;
    m[row * 4 + axisB] = s * va + c * vb;
  }
}

const _out5 = new Float32Array(5);
function projectPoint4DTo3D(
  x: number,
  y: number,
  z: number,
  w: number,
  s3Curvature: number,
  wProjDist: number,
  stereoScale: number
): void {
  const len4 = Math.hypot(x, y, z, w) || 1;
  const euclidScale = 0.5;
  const sphereScale = 1.0 / len4;
  const normScale = euclidScale + (sphereScale - euclidScale) * s3Curvature;

  const nx = x * normScale;
  const ny = y * normScale;
  const nz = z * normScale;
  const nw = w * normScale;

  const rx = _rotMat4D[0] * nx + _rotMat4D[1] * ny + _rotMat4D[2] * nz + _rotMat4D[3] * nw;
  const ry = _rotMat4D[4] * nx + _rotMat4D[5] * ny + _rotMat4D[6] * nz + _rotMat4D[7] * nw;
  const rz = _rotMat4D[8] * nx + _rotMat4D[9] * ny + _rotMat4D[10] * nz + _rotMat4D[11] * nw;
  const rw = _rotMat4D[12] * nx + _rotMat4D[13] * ny + _rotMat4D[14] * nz + _rotMat4D[15] * nw;

  const denom = Math.max(0.40, wProjDist - rw);
  const conf = stereoScale / denom;

  _out5[0] = rx * conf;
  _out5[1] = ry * conf;
  _out5[2] = rz * conf;
  _out5[3] = rw;
  _out5[4] = conf;
}

// ---------------------------------------------------------------------------
// Main Concept 8A Boot Function
// ---------------------------------------------------------------------------
export async function bootTesseractConcept1(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(8);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0.0, 0.12, 4.10],
    fov: 34,
    enableOrbit: false,
    enableShadows: false,
  });

  stage.camera.lookAt(0, 0.0, 0);
  stage.controls.target.set(0, 0.0, 0);

  // -------------------------------------------------------------------------
  // Physical Mineral & Botanical Pigments (`src/core/palette.ts`)
  // -------------------------------------------------------------------------
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const sporeSulfur = createPigmentUniforms('sporeSulfur');
  const lapisUltramarine = createPigmentUniforms('lapisUltramarine');
  const celadonGlauconite = createPigmentUniforms('celadonGlauconite');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');

  const uEversionPulse = uniform(0.0);

  // -------------------------------------------------------------------------
  // TSL Node Materials
  // -------------------------------------------------------------------------

  // 1. Wet Liquid-Meniscus Plateau Border Strut Material
  // (Slender liquid borders where 3 soap films meet at 120°, glowing with thin-film highlights)
  const strutMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.08,
    metalness: 0.02,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
  });
  {
    const aMeta = attribute('aStrutMeta', 'vec3'); // x: t [0..1], y: w4D [-1..1], z: junctionFlare [0..1]
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const ndotv = abs(dot(nWorld, viewDir));
    const fresnel = clamp(float(1.0).sub(ndotv), 0.0, 1.0);

    // Liquid interference along the wet Plateau border tube
    const phase = aMeta.x
      .mul(12.0)
      .add(aMeta.y.mul(4.5))
      .add(ndotv.mul(3.5))
      .sub(stage.uTime.mul(0.8));
    const w1 = sin(phase).mul(0.5).add(0.5);
    const w2 = cos(phase.mul(0.75).add(1.1)).mul(0.5).add(0.5);

    const lapisTint = kubelkaMunkGlaze(
      lapisUltramarine.K,
      lapisUltramarine.S,
      float(0.35),
      boneCalcite.rInf
    );
    const celadonTint = kubelkaMunkGlaze(
      celadonGlauconite.K,
      celadonGlauconite.S,
      float(0.32),
      boneCalcite.rInf
    );
    const amberCinnabar = mix(realgarAmber.rInf, cinnabarVermilion.rInf, w2.mul(0.6));

    const liquidBorderColor = mix(
      mix(lapisTint, celadonTint, w1),
      amberCinnabar,
      w2.mul(0.55)
    );

    // Studio softbox specular reflection along the wet liquid border
    const keyDir = normalize(vec3(0.48, 0.78, 0.52));
    const halfKey = normalize(keyDir.add(viewDir));
    const wetSpec = max(float(0.0), dot(nWorld, halfKey)).pow(64.0);

    const baseStrut = mix(
      liquidBorderColor,
      boneCalcite.rInf,
      fresnel.pow(2.0).mul(0.65).add(aMeta.z.mul(0.35))
    );

    strutMat.colorNode = clamp(baseStrut, 0.05, 0.98);
    strutMat.emissiveNode = mix(liquidBorderColor, boneCalcite.rInf, float(0.5))
      .mul(fresnel.pow(2.2).mul(0.42).add(wetSpec.mul(0.85)).add(uEversionPulse.mul(0.22)));
  }

  // 2. Smooth Liquid Plateau Vertex Droplets (where 4 liquid borders meet at tetrahedral angles)
  const vertexNodeMat = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0.06,
    metalness: 0.02,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
  });
  {
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const fresnel = clamp(float(1.0).sub(abs(dot(nWorld, viewDir))), 0.0, 1.0);
    const radialDist = length(pWorld);
    const innerMask = float(1.0).sub(smoothstep(0.55, 1.35, radialDist));

    const dropColor = mix(
      mix(boneCalcite.rInf, celadonGlauconite.rInf, float(0.35)),
      mix(realgarAmber.rInf, cinnabarVermilion.rInf, float(0.45)),
      innerMask
    );
    vertexNodeMat.colorNode = dropColor;
    vertexNodeMat.emissiveNode = mix(dropColor, boneCalcite.rInf, fresnel).mul(
      float(0.32).add(fresnel.pow(2.5).mul(0.55)).add(uEversionPulse.mul(0.25))
    );
  }

  // 3. 24-Face Iridescent Soap-Bubble & Plateau Minimal-Surface Membrane Material
  const soapFilmMat = new THREE.MeshPhysicalNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    roughness: 0.04,
    metalness: 0.0,
    ior: 1.333, // Exact refractive index of aqueous soap film
    clearcoat: 1.0,
    clearcoatRoughness: 0.03,
  });
  {
    const aFaceMeta = attribute('aFaceMeta', 'vec3'); // x: borderMeniscus [0=center..1=edge], y: w4D, z: faceIdNorm
    const pLoc = positionLocal;
    const pWorld = positionWorld;
    const nWorld = normalize(normalWorld);
    const viewDir = normalize(cameraPosition.sub(pWorld));
    const ndotv = abs(dot(nWorld, viewDir));
    const fresnel = clamp(float(1.0).sub(ndotv), 0.0, 1.0);
    const fresnelPow = fresnel.pow(2.2);

    // Live pointer-driven Marangoni surface-tension swirl on the soap film
    const ptrDist = length(pWorld.sub(stage.uPointerWorld));
    const ptrSwirl = exp(ptrDist.mul(ptrDist).negate().mul(1.55));

    // Flowing Marangoni thickness convection field across each curved soap bubble sheet
    const flowCoord = pLoc
      .mul(1.85)
      .add(vec3(stage.uTime.mul(0.16), stage.uTime.mul(-0.12), aFaceMeta.z.mul(6.28)));
    const marangoniSwirl = mineralToothFbm(flowCoord, float(2.2));
    const secondarySwirl = sin(
      pLoc.x.mul(5.5).add(pLoc.y.mul(4.8)).add(marangoniSwirl.mul(4.2)).sub(stage.uTime.mul(0.55))
    ).mul(0.18);

    // Physical soap-film thickness `d(x)`:
    // - Thicker near the Plateau border meniscus (`aFaceMeta.x -> 1`) due to capillary suction
    // - Thinner near the upper bubble apex (`pWorld.y > 0.5`) due to gravitational drainage
    // - Modulated by 4D depth `w`, Marangoni convection swirls, and `cos(theta_r)` (`ndotv`)
    const borderThick = smoothstep(0.35, 1.0, aFaceMeta.x).mul(0.58);
    const gravityDrain = clamp(float(0.65).sub(pWorld.y.mul(0.22)), 0.25, 1.05);
    const filmOpticalPath = gravityDrain
      .add(borderThick)
      .add(marangoniSwirl.mul(0.52))
      .add(secondarySwirl)
      .add(aFaceMeta.y.mul(0.32))
      .add(ndotv.mul(0.55))
      .add(ptrSwirl.mul(0.48))
      .add(uEversionPulse.mul(0.35));

    // Multi-order Airy thin-film interference phase (`delta = 2 *pi * OPD / lambda`)
    // Producing crisp rainbow-like mineral interference bands (Lapis -> Celadon -> Sulfur/Amber -> Cinnabar -> Calcite)
    const phase = filmOpticalPath.mul(19.5);
    const s1 = sin(phase).mul(0.5).add(0.5);
    const s2 = sin(phase.add(2.094)).mul(0.5).add(0.5); // +120 deg
    const s3 = sin(phase.add(4.188)).mul(0.5).add(0.5); // +240 deg
    const highOrderring = cos(phase.mul(1.65)).mul(0.5).add(0.5);

    // Map interference orders onto high-luminance mineral pigments over Bone Calcite
    const bandLapis = kubelkaMunkGlaze(
      lapisUltramarine.K,
      lapisUltramarine.S,
      float(0.36).add(s1.mul(0.35)),
      boneCalcite.rInf
    );
    const bandCeladon = kubelkaMunkGlaze(
      celadonGlauconite.K,
      celadonGlauconite.S,
      float(0.34).add(s2.mul(0.32)),
      boneCalcite.rInf
    );
    const bandSulfurAmber = mix(
      sporeSulfur.rInf,
      realgarAmber.rInf,
      s3
    );
    const bandCinnabarRose = kubelkaMunkGlaze(
      cinnabarVermilion.K,
      cinnabarVermilion.S,
      float(0.32).add(highOrderring.mul(0.28)),
      boneCalcite.rInf
    );

    const iridA = mix(bandLapis, bandCeladon, s1);
    const iridB = mix(bandSulfurAmber, bandCinnabarRose, s2);
    const thinFilmSpectrum = mix(iridA, iridB, s3);

    // Crisp luminous Plateau border meniscus fillet along the 4 edges of each cell membrane
    const meniscusFillet = smoothstep(float(0.76), float(0.995), aFaceMeta.x);
    const meniscusColor = mix(
      thinFilmSpectrum,
      boneCalcite.rInf,
      float(0.42)
    );
    const surfaceIridescence = mix(thinFilmSpectrum, meniscusColor, meniscusFillet.mul(0.72));

    // Analytical Studio Softbox Window Reflections on the curved soap-bubble surface
    // Only on front-facing curved bubble normals (`dot(reflDir, keyBoxDir) > 0.92`, no `abs`!)
    // so stacked interior faces never wash out to white!
    const reflDir = viewDir.negate().sub(nWorld.mul(dot(viewDir.negate(), nWorld).mul(2.0)));
    const keyBoxDir = normalize(vec3(0.52, 0.74, 0.45));
    const rimBoxDir = normalize(vec3(-0.68, 0.42, 0.45));
    const softboxHit1 = smoothstep(0.925, 0.988, dot(reflDir, keyBoxDir));
    const softboxHit2 = smoothstep(0.945, 0.992, dot(reflDir, rimBoxDir)).mul(0.55);
    const bubbleWindowGlint = softboxHit1.add(softboxHit2);

    // Distinct干渉 (interference) ribbon contrast so bands read vividly across the bubble
    const ribbonContrast = smoothstep(0.25, 0.85, max(s1, s3));

    soapFilmMat.colorNode = clamp(
      mix(surfaceIridescence, boneCalcite.rInf, bubbleWindowGlint.mul(0.55)),
      0.04,
      0.98
    );

    // Balanced self-luminous thin-film interference: low in face centers (`0.08`) so 6 stacked
    // membranes stay crystal-clear, surging at grazing Fresnel rims & Plateau borders!
    soapFilmMat.emissiveNode = surfaceIridescence
      .mul(
        float(0.08)
          .add(ribbonContrast.mul(0.14))
          .add(fresnelPow.mul(0.52))
          .add(meniscusFillet.mul(0.45))
          .add(ptrSwirl.mul(0.26))
      )
      .add(boneCalcite.rInf.mul(bubbleWindowGlint.mul(0.55)));

    // Authentic Soap-Bubble Opacity Profile:
    // Ultra-clear in the face center (`~0.045`), rising sharply at grazing Fresnel angles,
    // Plateau borders (`meniscusFillet`), and softbox window glints!
    const bubbleOpacity = float(0.045)
      .add(ribbonContrast.mul(0.08))
      .add(fresnelPow.mul(0.48))
      .add(meniscusFillet.mul(0.48))
      .add(bubbleWindowGlint.mul(0.38))
      .add(ptrSwirl.mul(0.12));

    soapFilmMat.opacityNode = clamp(bubbleOpacity, 0.04, 0.82);
  }

  // -------------------------------------------------------------------------
  // Assemble Pure Stereographic Soap-Bubble Polychoron
  // -------------------------------------------------------------------------
  const specimenRoot = new THREE.Group();
  specimenRoot.name = 'StereographicSoapBubblePolychoron';
  stage.specimenGroup.add(specimenRoot);

  // --- A. Pre-Allocated 32 Slender Wet Plateau Border Struts ---
  const strutGeom = new THREE.BufferGeometry();
  const strutPosAttr = new THREE.BufferAttribute(_strutPositions, 3);
  strutPosAttr.setUsage(THREE.DynamicDrawUsage);
  const strutNormAttr = new THREE.BufferAttribute(_strutNormals, 3);
  strutNormAttr.setUsage(THREE.DynamicDrawUsage);
  const strutMetaAttr = new THREE.BufferAttribute(_strutMeta, 3);
  strutMetaAttr.setUsage(THREE.DynamicDrawUsage);

  strutGeom.setAttribute('position', strutPosAttr);
  strutGeom.setAttribute('normal', strutNormAttr);
  strutGeom.setAttribute('aStrutMeta', strutMetaAttr);

  const strutIndices: number[] = [];
  const ringStride = STRUT_RADIAL_SEGMENTS + 1;
  for (let e = 0; e < NUM_EDGES; e++) {
    const baseV = e * VERTS_PER_STRUT;
    for (let s = 0; s < STRUT_AXIAL_SEGMENTS; s++) {
      for (let r = 0; r < STRUT_RADIAL_SEGMENTS; r++) {
        const i00 = baseV + s * ringStride + r;
        const i10 = baseV + (s + 1) * ringStride + r;
        const i11 = baseV + (s + 1) * ringStride + (r + 1);
        const i01 = baseV + s * ringStride + (r + 1);
        strutIndices.push(i00, i10, i11, i00, i11, i01);
      }
    }
  }
  strutGeom.setIndex(strutIndices);

  const strutsMesh = new THREE.Mesh(strutGeom, strutMat);
  strutsMesh.frustumCulled = false;
  specimenRoot.add(strutsMesh);

  // --- B. 16 Liquid Plateau Vertex Junction Droplets ---
  const vertexNodeMeshes: THREE.Mesh[] = [];
  const vertexDropletGeom = new THREE.SphereGeometry(0.026, 20, 14);

  for (let v = 0; v < NUM_VERTICES; v++) {
    const vMesh = new THREE.Mesh(vertexDropletGeom, vertexNodeMat);
    specimenRoot.add(vMesh);
    vertexNodeMeshes.push(vMesh);
  }

  // --- C. Pre-Allocated 24 Curved Soap-Bubble Minimal-Surface Membranes (`14x14`) ---
  const faceGeom = new THREE.BufferGeometry();
  const facePosAttr = new THREE.BufferAttribute(_facePositions, 3);
  facePosAttr.setUsage(THREE.DynamicDrawUsage);
  const faceNormAttr = new THREE.BufferAttribute(_faceNormals, 3);
  faceNormAttr.setUsage(THREE.DynamicDrawUsage);
  const faceMetaAttr = new THREE.BufferAttribute(_faceMeta, 3);
  faceMetaAttr.setUsage(THREE.DynamicDrawUsage);

  faceGeom.setAttribute('position', facePosAttr);
  faceGeom.setAttribute('normal', faceNormAttr);
  faceGeom.setAttribute('aFaceMeta', faceMetaAttr);

  const faceIndices: number[] = [];
  const faceRowStride = FACE_GRID_STEPS + 1;
  for (let f = 0; f < NUM_FACES; f++) {
    const baseV = f * VERTS_PER_FACE;
    for (let iy = 0; iy < FACE_GRID_STEPS; iy++) {
      for (let ix = 0; ix < FACE_GRID_STEPS; ix++) {
        const i00 = baseV + iy * faceRowStride + ix;
        const i10 = baseV + iy * faceRowStride + (ix + 1);
        const i11 = baseV + (iy + 1) * faceRowStride + (ix + 1);
        const i01 = baseV + (iy + 1) * faceRowStride + ix;
        faceIndices.push(i00, i10, i11, i00, i11, i01);
      }
    }
  }
  faceGeom.setIndex(faceIndices);

  const soapFilmMesh = new THREE.Mesh(faceGeom, soapFilmMat);
  soapFilmMesh.renderOrder = 2;
  soapFilmMesh.frustumCulled = false;
  specimenRoot.add(soapFilmMesh);

  // -------------------------------------------------------------------------
  // In-Place 4D Soap-Bubble Polychoron Geometry Updater (Zero Allocations)
  // -------------------------------------------------------------------------
  function updatePolychoronGeometry(
    angleXW: number,
    angleYW: number,
    angleZW: number,
    angleXY: number,
    angleXZ: number,
    s3Curvature: number,
    wProjDist: number,
    stereoScale: number,
    elapsedSec: number
  ): void {
    setIdentity4D(_rotMat4D);
    applyGivensRotation4D(_rotMat4D, 0, 1, angleXY);
    applyGivensRotation4D(_rotMat4D, 0, 2, angleXZ);
    applyGivensRotation4D(_rotMat4D, 0, 3, angleXW);
    applyGivensRotation4D(_rotMat4D, 1, 3, angleYW);
    applyGivensRotation4D(_rotMat4D, 2, 3, angleZW);

    // 1. Project the 16 vertices & position the liquid Plateau vertex droplets
    for (let i = 0; i < NUM_VERTICES; i++) {
      const bx = _baseVerts4D[i * 4 + 0];
      const by = _baseVerts4D[i * 4 + 1];
      const bz = _baseVerts4D[i * 4 + 2];
      const bw = _baseVerts4D[i * 4 + 3];

      projectPoint4DTo3D(bx, by, bz, bw, s3Curvature, wProjDist, stereoScale);
      const px = _out5[0];
      const py = _out5[1];
      const pz = _out5[2];
      const conf = _out5[4];

      const vMesh = vertexNodeMeshes[i];
      vMesh.position.set(px, py, pz);
      const nodeScale = Math.max(0.50, Math.min(1.85, conf * 0.85));
      vMesh.scale.setScalar(nodeScale);
    }

    // 2. Recompute all 32 Curved Liquid Plateau Border Struts in-place
    for (let e = 0; e < NUM_EDGES; e++) {
      const i0 = _edgePairs[e * 2 + 0];
      const i1 = _edgePairs[e * 2 + 1];

      const ax = _baseVerts4D[i0 * 4 + 0];
      const ay = _baseVerts4D[i0 * 4 + 1];
      const az = _baseVerts4D[i0 * 4 + 2];
      const aw = _baseVerts4D[i0 * 4 + 3];

      const bx = _baseVerts4D[i1 * 4 + 0];
      const by = _baseVerts4D[i1 * 4 + 1];
      const bz = _baseVerts4D[i1 * 4 + 2];
      const bw = _baseVerts4D[i1 * 4 + 3];

      for (let s = 0; s <= STRUT_AXIAL_SEGMENTS; s++) {
        const t = s / STRUT_AXIAL_SEGMENTS;
        const ix = ax + (bx - ax) * t;
        const iy = ay + (by - ay) * t;
        const iz = az + (bz - az) * t;
        const iw = aw + (bw - aw) * t;

        projectPoint4DTo3D(ix, iy, iz, iw, s3Curvature, wProjDist, stereoScale);
        _spinePoints[s * 3 + 0] = _out5[0];
        _spinePoints[s * 3 + 1] = _out5[1];
        _spinePoints[s * 3 + 2] = _out5[2];
        _spineW[s] = _out5[3];

        // Slender liquid Plateau border profile: thin waist along the span, flaring smoothly into vertex droplets
        const endDist = Math.min(t, 1.0 - t);
        const plateauFlare = 1.0 + 0.65 * Math.exp(-endDist * 14.0);
        const midWaist = 1.0 - 0.28 * Math.sin(t * Math.PI);
        const baseRadius = 0.0155 * Math.max(0.52, Math.min(1.95, _out5[4] * 0.86));
        _spineRadii[s] = baseRadius * midWaist * plateauFlare;
      }

      const baseV = e * VERTS_PER_STRUT;
      for (let s = 0; s <= STRUT_AXIAL_SEGMENTS; s++) {
        const t = s / STRUT_AXIAL_SEGMENTS;
        const endDist = Math.min(t, 1.0 - t);
        const junctionFlare = Math.exp(-endDist * 11.0);

        const sx = _spinePoints[s * 3 + 0];
        const sy = _spinePoints[s * 3 + 1];
        const sz = _spinePoints[s * 3 + 2];
        const rad = _spineRadii[s];
        const wVal = _spineW[s];

        if (s === 0) {
          _tangent.set(
            _spinePoints[3] - sx,
            _spinePoints[4] - sy,
            _spinePoints[5] - sz
          );
        } else if (s === STRUT_AXIAL_SEGMENTS) {
          const prev = (s - 1) * 3;
          _tangent.set(
            sx - _spinePoints[prev],
            sy - _spinePoints[prev + 1],
            sz - _spinePoints[prev + 2]
          );
        } else {
          const prev = (s - 1) * 3;
          const next = (s + 1) * 3;
          _tangent.set(
            _spinePoints[next] - _spinePoints[prev],
            _spinePoints[next + 1] - _spinePoints[prev + 1],
            _spinePoints[next + 2] - _spinePoints[prev + 2]
          );
        }
        _tangent.normalize();

        if (Math.abs(_tangent.y) < 0.92) {
          _upRef.set(0, 1, 0);
        } else {
          _upRef.set(1, 0, 0);
        }
        _normal.crossVectors(_tangent, _upRef).normalize();
        _binormal.crossVectors(_tangent, _normal).normalize();

        const ringStart = (baseV + s * ringStride) * 3;
        for (let r = 0; r <= STRUT_RADIAL_SEGMENTS; r++) {
          const c = _radialCos[r];
          const sn = _radialSin[r];
          const nx = _normal.x * c + _binormal.x * sn;
          const ny = _normal.y * c + _binormal.y * sn;
          const nz = _normal.z * c + _binormal.z * sn;

          const idx3 = ringStart + r * 3;
          _strutPositions[idx3 + 0] = sx + nx * rad;
          _strutPositions[idx3 + 1] = sy + ny * rad;
          _strutPositions[idx3 + 2] = sz + nz * rad;

          _strutNormals[idx3 + 0] = nx;
          _strutNormals[idx3 + 1] = ny;
          _strutNormals[idx3 + 2] = nz;

          _strutMeta[idx3 + 0] = t;
          _strutMeta[idx3 + 1] = wVal;
          _strutMeta[idx3 + 2] = junctionFlare;
        }
      }
    }
    strutPosAttr.needsUpdate = true;
    strutNormAttr.needsUpdate = true;
    strutMetaAttr.needsUpdate = true;

    // 3. Recompute all 24 Curved Soap-Bubble Minimal-Surface Membranes in-place
    for (let f = 0; f < NUM_FACES; f++) {
      const iv00 = _faceQuads[f * 4 + 0] * 4;
      const iv10 = _faceQuads[f * 4 + 1] * 4;
      const iv11 = _faceQuads[f * 4 + 2] * 4;
      const iv01 = _faceQuads[f * 4 + 3] * 4;

      const baseV = f * VERTS_PER_FACE;
      const faceNormId = f / NUM_FACES;

      for (let iy = 0; iy <= FACE_GRID_STEPS; iy++) {
        const v = iy / FACE_GRID_STEPS;
        const vSym = v * 2 - 1;
        for (let ix = 0; ix <= FACE_GRID_STEPS; ix++) {
          const u = ix / FACE_GRID_STEPS;
          const uSym = u * 2 - 1;

          const w00 = (1 - u) * (1 - v);
          const w10 = u * (1 - v);
          const w11 = u * v;
          const w01 = (1 - u) * v;

          let qx =
            _baseVerts4D[iv00] * w00 +
            _baseVerts4D[iv10] * w10 +
            _baseVerts4D[iv11] * w11 +
            _baseVerts4D[iv01] * w01;
          let qy =
            _baseVerts4D[iv00 + 1] * w00 +
            _baseVerts4D[iv10 + 1] * w10 +
            _baseVerts4D[iv11 + 1] * w11 +
            _baseVerts4D[iv01 + 1] * w01;
          let qz =
            _baseVerts4D[iv00 + 2] * w00 +
            _baseVerts4D[iv10 + 2] * w10 +
            _baseVerts4D[iv11 + 2] * w11 +
            _baseVerts4D[iv01 + 2] * w01;
          let qw =
            _baseVerts4D[iv00 + 3] * w00 +
            _baseVerts4D[iv10 + 3] * w10 +
            _baseVerts4D[iv11 + 3] * w11 +
            _baseVerts4D[iv01 + 3] * w01;

          // Soap-bubble Plateau catenoid relaxation + dynamic acoustic surface-tension wave
          // Pinned strictly to the 4 curved boundary struts (`interiorBubble = 0` on edges)
          const interiorBubble = (1 - uSym * uSym) * (1 - vSym * vSym);
          const bubbleWave =
            Math.sin(elapsedSec * 2.6 + f * 0.65 + (uSym * uSym + vSym * vSym) * 2.5) * 0.065;
          const catenoidPinch = 1.0 - interiorBubble * (0.20 - bubbleWave);
          qx *= catenoidPinch;
          qy *= catenoidPinch;
          qz *= catenoidPinch;
          qw *= catenoidPinch;

          projectPoint4DTo3D(qx, qy, qz, qw, s3Curvature, wProjDist, stereoScale);

          const vIdx3 = (baseV + iy * faceRowStride + ix) * 3;
          _facePositions[vIdx3 + 0] = _out5[0];
          _facePositions[vIdx3 + 1] = _out5[1];
          _facePositions[vIdx3 + 2] = _out5[2];

          // Exact distance to the nearest of the 4 Plateau border struts (`1` on strut, `0` at center)
          const borderMeniscus = Math.max(Math.abs(uSym), Math.abs(vSym));
          _faceMeta[vIdx3 + 0] = borderMeniscus;
          _faceMeta[vIdx3 + 1] = _out5[3];
          _faceMeta[vIdx3 + 2] = faceNormId;
        }
      }

      for (let iy = 0; iy <= FACE_GRID_STEPS; iy++) {
        const iyPrev = Math.max(0, iy - 1);
        const iyNext = Math.min(FACE_GRID_STEPS, iy + 1);
        for (let ix = 0; ix <= FACE_GRID_STEPS; ix++) {
          const ixPrev = Math.max(0, ix - 1);
          const ixNext = Math.min(FACE_GRID_STEPS, ix + 1);

          const idxU0 = (baseV + iy * faceRowStride + ixPrev) * 3;
          const idxU1 = (baseV + iy * faceRowStride + ixNext) * 3;
          const idxV0 = (baseV + iyPrev * faceRowStride + ix) * 3;
          const idxV1 = (baseV + iyNext * faceRowStride + ix) * 3;

          _dU.set(
            _facePositions[idxU1] - _facePositions[idxU0],
            _facePositions[idxU1 + 1] - _facePositions[idxU0 + 1],
            _facePositions[idxU1 + 2] - _facePositions[idxU0 + 2]
          );
          _dV.set(
            _facePositions[idxV1] - _facePositions[idxV0],
            _facePositions[idxV1 + 1] - _facePositions[idxV0 + 1],
            _facePositions[idxV1 + 2] - _facePositions[idxV0 + 2]
          );
          _faceNorm.crossVectors(_dU, _dV).normalize();

          const vIdx3 = (baseV + iy * faceRowStride + ix) * 3;
          _faceNormals[vIdx3 + 0] = _faceNorm.x;
          _faceNormals[vIdx3 + 1] = _faceNorm.y;
          _faceNormals[vIdx3 + 2] = _faceNorm.z;
        }
      }
    }
    facePosAttr.needsUpdate = true;
    faceNormAttr.needsUpdate = true;
    faceMetaAttr.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Tactile 4D Givens Rotation & Stereographic Bubble Kinematics
  // -------------------------------------------------------------------------
  let angleXW = 0.48;
  let angleYW = 0.32;
  let angleZW = 0.18;
  let angleXY = 0.36;
  let angleXZ = -0.22;

  let velXW = 0.34;
  let velYW = 0.21;
  let s3Curvature = 0.94; // Strong S^3 spherical soap-bubble curvature
  let targetS3Curvature = 0.94;
  let eversionPulse = 0.0;

  let wasPointerDown = false;
  let prevNdcX = 0;
  let prevNdcY = 0;
  let downTimeSec = 0;
  let dragDist = 0;

  updatePolychoronGeometry(
    angleXW,
    angleYW,
    angleZW,
    angleXY,
    angleXZ,
    s3Curvature,
    1.52,
    1.22,
    0
  );

  const baseGetState = window.__SPECIMEN_STAGE__?.getSceneState;
  if (window.__SPECIMEN_STAGE__ && baseGetState) {
    window.__SPECIMEN_STAGE__.getSceneState = () => ({
      ...baseGetState(),
      concept: '8A-schlegel-soapfilm',
      angleXW: Number(angleXW.toFixed(3)),
      angleYW: Number(angleYW.toFixed(3)),
      s3Curvature: Number(s3Curvature.toFixed(3)),
    });
  }

  stage.onTick(({ elapsedSec, dtSec, pointer }) => {
    if (pointer.isDown) {
      if (!wasPointerDown) {
        wasPointerDown = true;
        prevNdcX = pointer.ndc.x;
        prevNdcY = pointer.ndc.y;
        downTimeSec = elapsedSec;
        dragDist = 0;
      }

      const dx = pointer.ndc.x - prevNdcX;
      const dy = pointer.ndc.y - prevNdcY;
      prevNdcX = pointer.ndc.x;
      prevNdcY = pointer.ndc.y;
      dragDist += Math.hypot(dx, dy);

      angleXW += dx * 2.85;
      angleYW += dy * 2.85;
      velXW = Math.max(-2.6, Math.min(2.6, velXW + dx * 6.5));
      velYW = Math.max(-2.6, Math.min(2.6, velYW + dy * 6.5));

      specimenRoot.rotation.y += dx * 0.25;
      specimenRoot.rotation.x = Math.max(
        -0.35,
        Math.min(0.35, specimenRoot.rotation.x - dy * 0.25)
      );
    } else {
      if (wasPointerDown) {
        wasPointerDown = false;
        if (elapsedSec - downTimeSec < 0.26 && dragDist < 0.06) {
          eversionPulse = 1.0;
          velXW += (velXW >= 0 ? 1 : -1) * 1.45;
          velYW += 0.85;
          targetS3Curvature = targetS3Curvature > 0.78 ? 0.58 : 0.98;
        }
      }
      specimenRoot.rotation.y +=
        (Math.tanh(pointer.ndc.x * 0.9) * 0.16 - specimenRoot.rotation.y) * dtSec * 3.2;
      specimenRoot.rotation.x +=
        (-Math.tanh(pointer.ndc.y * 0.9) * 0.14 - specimenRoot.rotation.x) * dtSec * 3.2;
    }

    const decay = Math.exp(-dtSec * 1.8);
    velXW = 0.34 + (velXW - 0.34) * decay;
    velYW = 0.21 + (velYW - 0.21) * decay;
    eversionPulse *= Math.exp(-dtSec * 2.4);
    uEversionPulse.value = eversionPulse;

    angleXW += velXW * dtSec;
    angleYW += velYW * dtSec;
    angleZW += (0.15 + eversionPulse * 0.45) * dtSec;
    angleXY += 0.11 * dtSec;
    angleXZ = -0.22 + Math.sin(elapsedSec * 0.35) * 0.16;

    s3Curvature += (targetS3Curvature - s3Curvature) * (1 - Math.exp(-dtSec * 4.5));
    const wProjDist = 1.52 - eversionPulse * 0.12 + Math.sin(elapsedSec * 0.9) * 0.04;
    const stereoScale = 1.22 + eversionPulse * 0.08;

    updatePolychoronGeometry(
      angleXW,
      angleYW,
      angleZW,
      angleXY,
      angleXZ,
      s3Curvature,
      wProjDist,
      stereoScale,
      elapsedSec
    );
  });

  return stage;
}
