import * as THREE from 'three/webgpu';
import { FIG_ORIGIN, FIG_YAW_DEG } from './mirageRig';

export interface FigureRigHandles {
  root: THREE.Group;
  bodyMesh: THREE.Mesh;
  hairMesh: THREE.Mesh;
  faceFeaturesMesh: THREE.Mesh;
  sketchStrokesMesh: THREE.Mesh;
  mothGroups: THREE.Group[];
  mothLeftWings: THREE.Mesh[];
  mothRightWings: THREE.Mesh[];
  swirlRibbonsMesh: THREE.Mesh;
  emberFlecksMesh: THREE.InstancedMesh;
}

interface MeshAccumulator {
  positions: number[];
  uvs: number[];
  indices: number[];
  features: number[];
}

function createAccum(): MeshAccumulator {
  return { positions: [], uvs: [], indices: [], features: [] };
}

function finalizeGeometry(acc: MeshAccumulator): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(acc.positions, 3));
  geom.setAttribute('uv', new THREE.Float32BufferAttribute(acc.uvs, 2));
  geom.setAttribute('aFeature', new THREE.Float32BufferAttribute(acc.features, 1));
  geom.setIndex(acc.indices);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Resamples sparse control stations along a smooth Catmull-Rom curve for crease-free organic limbs.
 */
function smoothStations(
  control: ReadonlyArray<{ center: [number, number, number]; rx: number; ry: number }>,
  subdivisions: number
): Array<{ center: [number, number, number]; rx: number; ry: number }> {
  if (control.length <= 2 || subdivisions <= 1) return [...control];
  const pts = control.map((s) => new THREE.Vector3(s.center[0], s.center[1], s.center[2]));
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
  const total = (control.length - 1) * subdivisions;
  const out: Array<{ center: [number, number, number]; rx: number; ry: number }> = [];
  const tmp = new THREE.Vector3();

  for (let i = 0; i <= total; i++) {
    const t = i / total;
    curve.getPoint(t, tmp);
    const fIdx = t * (control.length - 1);
    const i0 = Math.min(control.length - 1, Math.floor(fIdx));
    const i1 = Math.min(control.length - 1, i0 + 1);
    const frac = fIdx - i0;
    const rx = control[i0].rx * (1 - frac) + control[i1].rx * frac;
    const ry = control[i0].ry * (1 - frac) + control[i1].ry * frac;
    out.push({ center: [tmp.x, tmp.y, tmp.z], rx, ry });
  }
  return out;
}

/**
 * Lofts a closed organic tube using a Parallel-Transport (Bishop) frame so curved spines
 * and bent knees never experience 180-deg frame flips or waist pinch twists.
 */
function addLoftedTube(
  acc: MeshAccumulator,
  stations: ReadonlyArray<{
    center: [number, number, number];
    rx: number;
    ry: number;
  }>,
  radialSegments: number,
  featureId: number,
  capEnds = true
): void {
  const nStations = stations.length;
  if (nStations < 2) return;

  const baseVert = acc.positions.length / 3;
  const _t = new THREE.Vector3();
  const _tPrev = new THREE.Vector3();
  const _n = new THREE.Vector3();
  const _b = new THREE.Vector3();
  const _q = new THREE.Quaternion();

  // Compute tangents cleanly per station
  const tangents: THREE.Vector3[] = [];
  for (let i = 0; i < nStations; i++) {
    const prev = stations[Math.max(0, i - 1)].center;
    const next = stations[Math.min(nStations - 1, i + 1)].center;
    const t = new THREE.Vector3(next[0] - prev[0], next[1] - prev[1], next[2] - prev[2]);
    if (t.lengthSq() < 1e-10) t.set(0, 1, 0);
    t.normalize();
    tangents.push(t);
  }

  // Initial stable normal aligned toward world +X (anatomical lateral) or +Z
  _tPrev.copy(tangents[0]);
  const seedAxis = Math.abs(_tPrev.x) < 0.85 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
  _n.copy(seedAxis).addScaledVector(_tPrev, -seedAxis.dot(_tPrev)).normalize();
  _b.crossVectors(_tPrev, _n).normalize();

  for (let i = 0; i < nStations; i++) {
    _t.copy(tangents[i]);
    if (i > 0) {
      _q.setFromUnitVectors(_tPrev, _t);
      _n.applyQuaternion(_q).normalize();
      // Gram-Schmidt orthogonalize
      _n.addScaledVector(_t, -_n.dot(_t)).normalize();
      _b.crossVectors(_t, _n).normalize();
    }
    _tPrev.copy(_t);

    const st = stations[i];
    const vCoord = i / (nStations - 1);

    for (let j = 0; j <= radialSegments; j++) {
      const uCoord = j / radialSegments;
      const theta = uCoord * Math.PI * 2;
      const cx = Math.cos(theta) * st.rx;
      const cy = Math.sin(theta) * st.ry;
      const px = st.center[0] + _n.x * cx + _b.x * cy;
      const py = st.center[1] + _n.y * cx + _b.y * cy;
      const pz = st.center[2] + _n.z * cx + _b.z * cy;
      acc.positions.push(px, py, pz);
      acc.uvs.push(uCoord, vCoord);
      acc.features.push(featureId);
    }
  }

  const stride = radialSegments + 1;
  const midI = Math.floor((nStations - 1) / 2);
  const a0 = baseVert + midI * stride;
  const b0 = a0 + stride;
  const d0 = a0 + 1;
  const cMid = stations[midI].center;
  const pa = new THREE.Vector3(acc.positions[a0 * 3], acc.positions[a0 * 3 + 1], acc.positions[a0 * 3 + 2]);
  const pb = new THREE.Vector3(acc.positions[b0 * 3], acc.positions[b0 * 3 + 1], acc.positions[b0 * 3 + 2]);
  const pd = new THREE.Vector3(acc.positions[d0 * 3], acc.positions[d0 * 3 + 1], acc.positions[d0 * 3 + 2]);
  const triN = pb.clone().sub(pa).cross(pd.clone().sub(pa));
  const radialOut = pa.clone().set(pa.x - cMid[0], pa.y - cMid[1], pa.z - cMid[2]);
  const flipWinding = triN.dot(radialOut) < 0;

  for (let i = 0; i < nStations - 1; i++) {
    for (let j = 0; j < radialSegments; j++) {
      const a = baseVert + i * stride + j;
      const b = a + stride;
      const c = b + 1;
      const d = a + 1;
      if (flipWinding) {
        acc.indices.push(a, d, c, a, c, b);
      } else {
        acc.indices.push(a, b, c, a, c, d);
      }
    }
  }

  if (capEnds) {
    const startCenterIdx = acc.positions.length / 3;
    const s0 = stations[0].center;
    acc.positions.push(s0[0], s0[1], s0[2]);
    acc.uvs.push(0.5, 0.0);
    acc.features.push(featureId);
    for (let j = 0; j < radialSegments; j++) {
      const p1 = baseVert + j;
      const p2 = baseVert + j + 1;
      if (flipWinding) acc.indices.push(startCenterIdx, p1, p2);
      else acc.indices.push(startCenterIdx, p2, p1);
    }

    const endCenterIdx = acc.positions.length / 3;
    const sEnd = stations[nStations - 1].center;
    acc.positions.push(sEnd[0], sEnd[1], sEnd[2]);
    acc.uvs.push(0.5, 1.0);
    acc.features.push(featureId);
    const endRing = baseVert + (nStations - 1) * stride;
    for (let j = 0; j < radialSegments; j++) {
      const p1 = endRing + j;
      const p2 = endRing + j + 1;
      if (flipWinding) acc.indices.push(endCenterIdx, p2, p1);
      else acc.indices.push(endCenterIdx, p1, p2);
    }
  }
}

const gauss2 = (x: number, y: number, x0: number, y0: number, sx: number, sy: number) => {
  const dx = (x - x0) / sx;
  const dy = (y - y0) / sy;
  return Math.exp(-(dx * dx + dy * dy));
};

/**
 * Shared exact head surface evaluator so facial features (eyes, irises, lashes, brows, lips)
 * track the sculpted head surface with exact millimeter precision and zero sinking.
 */
function evalHeadSurfacePoint(nx: number, ny: number, nz: number, out: THREE.Vector3): THREE.Vector3 {
  const jawTaper = ny < 0 ? 1.0 - 0.24 * Math.pow(-ny, 1.35) : 1.0 - 0.05 * Math.pow(ny, 2.0);
  const craniumDepth = ny > 0.05 ? 1.0 + 0.14 * Math.exp(-Math.pow(ny - 0.25, 2) * 5.0) : 1.0;

  const rx = 0.044 * jawTaper;
  const ry = 0.057;
  const rz = 0.051 * craniumDepth * (ny < -0.25 ? 0.9 : 1.0);

  let nzMod = nz;
  if (nz > 0) {
    const frontW = Math.min(1, nz * 1.45);
    const brow = 0.003 * (gauss2(nx, ny, -0.26, 0.22, 0.2, 0.09) + gauss2(nx, ny, 0.26, 0.22, 0.2, 0.09));
    const socket = -0.0046 * (gauss2(nx, ny, -0.32, 0.07, 0.16, 0.11) + gauss2(nx, ny, 0.32, 0.07, 0.16, 0.11));
    const cheek = 0.004 * (gauss2(nx, ny, -0.36, -0.10, 0.20, 0.15) + gauss2(nx, ny, 0.36, -0.10, 0.20, 0.15));
    const noseBridge = 0.0072 * gauss2(nx, ny, 0.0, 0.05, 0.085, 0.21);
    const noseTip = 0.011 * gauss2(nx, ny, 0.0, -0.15, 0.09, 0.09);
    const nostrils = 0.0034 * (gauss2(nx, ny, -0.10, -0.19, 0.06, 0.055) + gauss2(nx, ny, 0.10, -0.19, 0.06, 0.055));
    const upperLip = 0.0038 * gauss2(nx, ny, 0.0, -0.36, 0.14, 0.05);
    const lipPart = -0.0022 * gauss2(nx, ny, 0.0, -0.42, 0.11, 0.025);
    const lowerLip = 0.0042 * gauss2(nx, ny, 0.0, -0.48, 0.12, 0.048);
    const chin = 0.005 * gauss2(nx, ny, 0.0, -0.74, 0.18, 0.13);

    const relief =
      (brow + socket + cheek + noseBridge + noseTip + nostrils + upperLip + lipPart + lowerLip + chin) * frontW;
    nzMod += relief / Math.max(0.02, rz);
  }

  return out.set(nx * rx, ny * ry, nzMod * rz);
}

/**
 * Evaluates a point directly ON the front face (+Z) of the head for a given unit (nx, ny)
 * plus an explicit surface Z offset (`zBias`), transformed into figure-local space via `headMat`.
 */
function sampleFaceFeaturePos(
  nx: number,
  ny: number,
  zBias: number,
  headMat: THREE.Matrix4,
  out: THREE.Vector3
): THREE.Vector3 {
  const nz = Math.sqrt(Math.max(0.01, 1.0 - nx * nx - ny * ny));
  evalHeadSurfacePoint(nx, ny, nz, out);
  out.z += zBias;
  return out.applyMatrix4(headMat);
}

function appendSculptedHead(
  bodyAcc: MeshAccumulator,
  headCenter: [number, number, number],
  headPitchRad: number
): THREE.Matrix4 {
  const mat = new THREE.Matrix4().makeTranslation(headCenter[0], headCenter[1], headCenter[2]);
  const rot = new THREE.Matrix4().makeRotationX(headPitchRad);
  mat.multiply(rot);

  const nLat = 56;
  const nLon = 72;
  const baseVert = bodyAcc.positions.length / 3;
  const vPos = new THREE.Vector3();

  for (let iy = 0; iy <= nLat; iy++) {
    const v = iy / nLat;
    const phi = v * Math.PI;
    const sinP = Math.sin(phi);
    const cosP = Math.cos(phi);

    for (let ix = 0; ix <= nLon; ix++) {
      const u = ix / nLon;
      const theta = (u - 0.5) * Math.PI * 2;
      const nx = sinP * Math.sin(theta);
      const ny = cosP;
      const nz = sinP * Math.cos(theta);

      evalHeadSurfacePoint(nx, ny, nz, vPos).applyMatrix4(mat);
      bodyAcc.positions.push(vPos.x, vPos.y, vPos.z);
      bodyAcc.uvs.push(u, v);
      bodyAcc.features.push(0);
    }
  }

  const stride = nLon + 1;
  for (let iy = 0; iy < nLat; iy++) {
    for (let ix = 0; ix < nLon; ix++) {
      const a = baseVert + iy * stride + ix;
      const b = a + 1;
      const c = a + stride + 1;
      const d = a + stride;
      bodyAcc.indices.push(a, d, c, a, c, b);
    }
  }

  return mat;
}

/**
 * Builds eyes, amber irises, pupils, catchlights, upper lashes, and lifted eyebrows
 * anchored directly on the sculpted face surface using `sampleFaceFeaturePos`.
 */
function buildFaceFeaturesGeometry(headMat: THREE.Matrix4): THREE.BufferGeometry {
  const acc = createAccum();
  const _p = new THREE.Vector3();
  const _center = new THREE.Vector3();

  const pushSurfaceEllipsoid = (
    nx0: number,
    ny0: number,
    zBias: number,
    rx: number,
    ry: number,
    rz: number,
    featureId: number,
    segs = 14
  ) => {
    sampleFaceFeaturePos(nx0, ny0, zBias, headMat, _center);
    const base = acc.positions.length / 3;
    for (let iy = 0; iy <= segs; iy++) {
      const v = iy / segs;
      const phi = v * Math.PI;
      for (let ix = 0; ix <= segs; ix++) {
        const u = ix / segs;
        const theta = u * Math.PI * 2;
        _p.set(
          Math.sin(phi) * Math.sin(theta) * rx,
          Math.cos(phi) * ry,
          Math.sin(phi) * Math.cos(theta) * rz
        );
        // Rotate local offset by head pitch only (extract rotation from headMat)
        _p.applyMatrix4(new THREE.Matrix4().extractRotation(headMat)).add(_center);
        acc.positions.push(_p.x, _p.y, _p.z);
        acc.uvs.push(u, v);
        acc.features.push(featureId);
      }
    }
    const stride = segs + 1;
    for (let iy = 0; iy < segs; iy++) {
      for (let ix = 0; ix < segs; ix++) {
        const a = base + iy * stride + ix;
        const b = a + 1;
        const c = a + stride + 1;
        const d = a + stride;
        acc.indices.push(a, d, c, a, c, b);
      }
    }
  };

  const pushFaceTube = (
    normCoords: Array<[number, number, number]>,
    radius: number,
    featureId: number
  ) => {
    const stations = normCoords.map(([nx, ny, zb]) => {
      sampleFaceFeaturePos(nx, ny, zb, headMat, _p);
      return { center: [_p.x, _p.y, _p.z] as [number, number, number], rx: radius, ry: radius };
    });
    addLoftedTube(acc, stations, 8, featureId, true);
  };

  for (const side of [-1, 1]) {
    const enx = side * 0.33;
    const eny = 0.075;

    // Almond sclera protruding cleanly out of the orbital socket (+0.0012 above skin surface)
    pushSurfaceEllipsoid(enx, eny, 0.0012, 0.0088, 0.0056, 0.0036, 3, 18);
    // Luminous Realgar-Amber iris gazing down-left toward cupped palms
    pushSurfaceEllipsoid(enx - 0.025, eny - 0.015, 0.0040, 0.0048, 0.0046, 0.0020, 4, 18);
    // Crisp dark pupil
    pushSurfaceEllipsoid(enx - 0.028, eny - 0.018, 0.0056, 0.0023, 0.0023, 0.0011, 5, 12);
    // Warm amber-ivory specular catchlight
    pushSurfaceEllipsoid(enx - 0.055, eny + 0.012, 0.0066, 0.0011, 0.0011, 0.0007, 6, 10);

    // Expressive dark upper eyelash rim arching cleanly across the top of the eye
    pushFaceTube(
      [
        [enx - side * 0.19, eny - 0.01, 0.0026],
        [enx - side * 0.08, eny + 0.072, 0.0038],
        [enx + side * 0.04, eny + 0.078, 0.0038],
        [enx + side * 0.17, eny + 0.018, 0.0024],
      ],
      0.00115,
      7
    );

    // Delicate lower lash hint
    pushFaceTube(
      [
        [enx - side * 0.14, eny - 0.048, 0.0020],
        [enx, eny - 0.076, 0.0030],
        [enx + side * 0.13, eny - 0.042, 0.0020],
      ],
      0.0006,
      7
    );

    // Gently lifted expressive eyebrow
    pushFaceTube(
      [
        [enx - side * 0.21, eny + 0.155, 0.0020],
        [enx - side * 0.05, eny + 0.225, 0.0026],
        [enx + side * 0.11, eny + 0.205, 0.0024],
        [enx + side * 0.23, eny + 0.130, 0.0016],
      ],
      0.00145,
      7
    );
  }

  // Delicate nose shadow edge + soft parted awe-struck lips
  pushFaceTube(
    [
      [-0.08, -0.18, 0.0012],
      [0.0, -0.20, 0.0015],
      [0.08, -0.18, 0.0012],
    ],
    0.00065,
    7
  );
  pushFaceTube(
    [
      [-0.125, -0.415, 0.0016],
      [-0.045, -0.405, 0.0020],
      [0.0, -0.402, 0.0021],
      [0.045, -0.405, 0.0020],
      [0.125, -0.415, 0.0016],
    ],
    0.00075,
    7
  );
  pushFaceTube(
    [
      [-0.075, -0.465, 0.0014],
      [0.0, -0.468, 0.0016],
      [0.075, -0.465, 0.0014],
    ],
    0.00055,
    7
  );

  return finalizeGeometry(acc);
}

/**
 * Builds tied-back hair crown + low nape bun + 12 loose graphite wisps framing temples & neck.
 */
function buildHairAndTendrilsGeometry(headMat: THREE.Matrix4): THREE.BufferGeometry {
  const acc = createAccum();
  const _p = new THREE.Vector3();

  const nLat = 34;
  const nLon = 48;
  const baseVert = acc.positions.length / 3;

  for (let iy = 0; iy <= nLat; iy++) {
    const v = iy / nLat;
    const phi = v * Math.PI * 0.78;
    const sinP = Math.sin(phi);
    const cosP = Math.cos(phi);

    for (let ix = 0; ix <= nLon; ix++) {
      const u = ix / nLon;
      const theta = (u - 0.5) * Math.PI * 2;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);

      const isFaceWindow = cosT > 0.24 && cosP < 0.78;
      let radialScale = 1.09;
      if (isFaceWindow) {
        const pullBack = Math.min(1, (cosT - 0.24) / 0.54) * Math.max(0, 0.78 - cosP);
        radialScale -= pullBack * 0.2;
      }
      const strandWave = 0.0022 * Math.sin(phi * 16.0 + Math.abs(sinT) * 6.0);
      const bunPull = cosT < -0.55 && cosP < 0.15 ? 0.012 * Math.exp(-Math.pow(cosP + 0.15, 2) * 18) : 0;

      const rx = 0.0475 * radialScale + strandWave;
      const ry = 0.0605 * (cosP > 0 ? 1.05 : 1.0) + strandWave * 0.6;
      const rz = 0.055 * radialScale + bunPull;

      const x = sinP * sinT * rx;
      const y = cosP * ry + 0.002;
      const z = sinP * cosT * rz - 0.004;

      const zClamped = cosT > 0.34 && cosP < 0.54 ? Math.min(z, 0.016 + Math.max(0, cosP) * 0.025) : z;

      _p.set(x, y, zClamped).applyMatrix4(headMat);
      acc.positions.push(_p.x, _p.y, _p.z);
      acc.uvs.push(u, v);
      acc.features.push(2);
    }
  }

  const stride = nLon + 1;
  for (let iy = 0; iy < nLat; iy++) {
    for (let ix = 0; ix < nLon; ix++) {
      const a = baseVert + iy * stride + ix;
      const b = a + 1;
      const c = a + stride + 1;
      const d = a + stride;
      acc.indices.push(a, d, c, a, c, b);
    }
  }

  // Tied-back low gathered hair bun at the nape
  const bunPts: Array<{ pt: [number, number, number]; rx: number; ry: number }> = [
    { pt: [0.0, -0.01, -0.046], rx: 0.022, ry: 0.02 },
    { pt: [0.0, -0.016, -0.062], rx: 0.024, ry: 0.022 },
    { pt: [0.0, -0.026, -0.074], rx: 0.018, ry: 0.016 },
    { pt: [0.003, -0.042, -0.082], rx: 0.009, ry: 0.009 },
  ];
  addLoftedTube(
    acc,
    bunPts.map((b) => {
      _p.set(b.pt[0], b.pt[1], b.pt[2]).applyMatrix4(headMat);
      return { center: [_p.x, _p.y, _p.z], rx: b.rx, ry: b.ry };
    }),
    18,
    2,
    true
  );

  // 12 Loose wispy graphite hair tendrils framing temples, cheeks, and neck
  const tendrilSpecs: Array<{ pts: Array<[number, number, number]>; rStart: number }> = [
    { pts: [[0.030, 0.042, 0.026], [0.044, 0.020, 0.024], [0.048, -0.006, 0.014], [0.044, -0.032, 0.002]], rStart: 0.003 },
    { pts: [[0.024, 0.046, 0.030], [0.038, 0.028, 0.032], [0.043, 0.008, 0.024], [0.040, -0.018, 0.014]], rStart: 0.0024 },
    { pts: [[0.042, 0.012, 0.002], [0.047, -0.012, -0.010], [0.042, -0.038, -0.022], [0.030, -0.056, -0.032]], rStart: 0.0034 },
    { pts: [[0.036, 0.035, -0.014], [0.045, 0.005, -0.028], [0.038, -0.024, -0.042], [0.022, -0.045, -0.055]], rStart: 0.0036 },
    { pts: [[-0.028, 0.042, 0.026], [-0.043, 0.020, 0.026], [-0.047, -0.006, 0.016], [-0.042, -0.028, 0.006]], rStart: 0.0028 },
    { pts: [[-0.022, 0.046, 0.032], [-0.037, 0.026, 0.033], [-0.042, 0.005, 0.026], [-0.038, -0.016, 0.018]], rStart: 0.0023 },
    { pts: [[-0.040, 0.010, 0.000], [-0.046, -0.014, -0.012], [-0.040, -0.038, -0.024], [-0.026, -0.055, -0.036]], rStart: 0.0032 },
    { pts: [[-0.034, 0.034, -0.016], [-0.043, 0.004, -0.030], [-0.036, -0.025, -0.044], [-0.020, -0.046, -0.056]], rStart: 0.0034 },
    { pts: [[0.006, 0.052, 0.032], [0.022, 0.038, 0.038], [0.035, 0.022, 0.036]], rStart: 0.0021 },
    { pts: [[-0.005, 0.052, 0.032], [-0.020, 0.038, 0.038], [-0.032, 0.022, 0.037]], rStart: 0.0021 },
    { pts: [[0.016, -0.020, -0.056], [0.022, -0.042, -0.058], [0.018, -0.064, -0.052]], rStart: 0.0028 },
    { pts: [[-0.015, -0.020, -0.056], [-0.020, -0.044, -0.058], [-0.016, -0.065, -0.050]], rStart: 0.0028 },
  ];

  for (const td of tendrilSpecs) {
    const stations = td.pts.map((pt, idx) => {
      const f = idx / (td.pts.length - 1);
      const r = td.rStart * (1.0 - 0.72 * f);
      _p.set(pt[0], pt[1], pt[2]).applyMatrix4(headMat);
      return {
        center: [_p.x, _p.y, _p.z] as [number, number, number],
        rx: r,
        ry: r * 0.72,
      };
    });
    addLoftedTube(acc, smoothStations(stations, 3), 8, 2, true);
  }

  return finalizeGeometry(acc);
}

/**
 * Anatomically sculpted 5-finger cupped hand anchored right over/beside the drawn-up knee
 * so at 0% side-profile it seamlessly reads as arms wrapped around knees in grief,
 * while separating laterally at 42° into an open glowing bowl.
 */
function appendCuppedHand(
  acc: MeshAccumulator,
  side: 1 | -1,
  wristPos: [number, number, number]
): void {
  const wx = wristPos[0];
  const wy = wristPos[1];
  const wz = wristPos[2];

  const palmStations = smoothStations(
    [
      { center: [wx, wy, wz], rx: 0.015, ry: 0.010 },
      { center: [wx - side * 0.004, wy + 0.003, wz + 0.012], rx: 0.020, ry: 0.0095 },
      { center: [wx - side * 0.009, wy + 0.008, wz + 0.024], rx: 0.021, ry: 0.008 },
      { center: [wx - side * 0.011, wy + 0.014, wz + 0.034], rx: 0.018, ry: 0.0065 },
    ],
    2
  );
  addLoftedTube(acc, palmStations, 16, 0, true);

  const fingerOffsets = [
    { lat: -0.012, len: 0.029, r: 0.0042, curl: 0.024 },
    { lat: -0.004, len: 0.032, r: 0.0043, curl: 0.026 },
    { lat: 0.005, len: 0.030, r: 0.0040, curl: 0.027 },
    { lat: 0.013, len: 0.028, r: 0.0037, curl: 0.028 },
  ];

  for (const f of fingerOffsets) {
    const fx0 = wx - side * (0.009 + f.lat);
    const fy0 = wy + 0.013 - Math.abs(f.lat) * 0.12;
    const fz0 = wz + 0.032;
    const fStations = smoothStations(
      [
        { center: [fx0, fy0, fz0], rx: f.r, ry: f.r },
        { center: [fx0 - side * 0.003, fy0 + f.curl * 0.45, fz0 + f.len * 0.32], rx: f.r * 0.92, ry: f.r * 0.92 },
        { center: [fx0 - side * 0.008, fy0 + f.curl * 0.92, fz0 + f.len * 0.62], rx: f.r * 0.80, ry: f.r * 0.80 },
        { center: [fx0 - side * 0.013, fy0 + f.curl * 1.25, fz0 + f.len * 0.82], rx: f.r * 0.65, ry: f.r * 0.65 },
      ],
      2
    );
    addLoftedTube(acc, fStations, 10, 0, true);
  }

  const thumbBaseX = wx + side * 0.011;
  const thumbBaseY = wy + 0.012;
  const thumbBaseZ = wz + 0.014;
  addLoftedTube(
    acc,
    smoothStations(
      [
        { center: [thumbBaseX, thumbBaseY, thumbBaseZ], rx: 0.0058, ry: 0.0052 },
        { center: [thumbBaseX + side * 0.004, thumbBaseY + 0.012, thumbBaseZ + 0.013], rx: 0.0050, ry: 0.0046 },
        { center: [thumbBaseX + side * 0.002, thumbBaseY + 0.022, thumbBaseZ + 0.024], rx: 0.0041, ry: 0.0038 },
      ],
      2
    ),
    10,
    0,
    true
  );
}

function buildSketchStrokesGeometry(): THREE.BufferGeometry {
  const acc = createAccum();

  addLoftedTube(
    acc,
    smoothStations(
      [
        { center: [-0.044, 0.254, 0.012], rx: 0.0014, ry: 0.0014 },
        { center: [-0.022, 0.242, 0.034], rx: 0.0015, ry: 0.0015 },
        { center: [0.0, 0.234, 0.044], rx: 0.0016, ry: 0.0016 },
        { center: [0.022, 0.242, 0.034], rx: 0.0015, ry: 0.0015 },
        { center: [0.044, 0.254, 0.012], rx: 0.0014, ry: 0.0014 },
      ],
      2
    ),
    8,
    7,
    true
  );

  addLoftedTube(
    acc,
    smoothStations(
      [
        { center: [-0.018, 0.232, 0.042], rx: 0.0011, ry: 0.0011 },
        { center: [-0.010, 0.190, 0.048], rx: 0.0013, ry: 0.0013 },
        { center: [0.004, 0.150, 0.042], rx: 0.0010, ry: 0.0010 },
      ],
      2
    ),
    6,
    7,
    true
  );
  addLoftedTube(
    acc,
    smoothStations(
      [
        { center: [0.022, 0.234, 0.038], rx: 0.0011, ry: 0.0011 },
        { center: [0.018, 0.192, 0.046], rx: 0.0012, ry: 0.0012 },
        { center: [0.008, 0.154, 0.040], rx: 0.0010, ry: 0.0010 },
      ],
      2
    ),
    6,
    7,
    true
  );

  for (const side of [-1, 1]) {
    addLoftedTube(
      acc,
      smoothStations(
        [
          { center: [side * 0.054, 0.188, 0.130], rx: 0.0013, ry: 0.0013 },
          { center: [side * 0.043, 0.200, 0.144], rx: 0.0014, ry: 0.0014 },
          { center: [side * 0.036, 0.192, 0.148], rx: 0.0012, ry: 0.0012 },
        ],
        2
      ),
      6,
      7,
      true
    );
  }

  return finalizeGeometry(acc);
}

function buildOrigamiWingGeometry(side: -1 | 1): THREE.BufferGeometry {
  const s = side;
  const verts = new Float32Array([
    0.0, 0.0, 0.36,
    0.0, 0.03, 0.02,
    0.0, -0.02, -0.44,
    s * 1.18, 0.24, 0.62,
    s * 0.84, -0.08, 0.16,
    s * 0.92, 0.10, -0.22,
    s * 0.44, -0.06, -0.66,
  ]);
  const uvs = new Float32Array([
    0.0, 0.8,
    0.0, 0.5,
    0.0, 0.1,
    1.0, 0.95,
    0.72, 0.52,
    0.82, 0.25,
    0.45, 0.02,
  ]);
  const indices =
    side === 1
      ? [0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 6]
      : [0, 4, 3, 0, 1, 4, 1, 5, 4, 1, 2, 5, 2, 6, 5];

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

function buildSwirlRibbonsGeometry(): THREE.BufferGeometry {
  const acc = createAccum();
  const ribbonSeeds = [
    { r0: 0.020, r1: 0.068, h: 0.165, turns: 1.55, phase: 0.2 },
    { r0: 0.016, r1: 0.058, h: 0.140, turns: 1.35, phase: 2.1 },
    { r0: 0.022, r1: 0.078, h: 0.185, turns: 1.75, phase: 3.9 },
    { r0: 0.014, r1: 0.046, h: 0.108, turns: 1.15, phase: 5.1 },
  ];

  const steps = 28;
  for (let c = 0; c < ribbonSeeds.length; c++) {
    const seed = ribbonSeeds[c];
    const stations: Array<{ center: [number, number, number]; rx: number; ry: number }> = [];
    for (let i = 0; i <= steps; i++) {
      const f = i / steps;
      const ang = seed.phase + f * Math.PI * 2 * seed.turns;
      const rad = seed.r0 + (seed.r1 - seed.r0) * f;
      const x = Math.sin(ang) * rad * 0.9 - 0.015 * f;
      const y = 0.215 + f * seed.h + Math.sin(f * Math.PI * 3 + seed.phase) * 0.008;
      const z = 0.185 + Math.cos(ang) * rad * 0.85;
      const w = 0.0014 * Math.sin(f * Math.PI) + 0.0004;
      stations.push({ center: [x, y, z], rx: w, ry: w });
    }
    addLoftedTube(acc, stations, 6, 7, true);
  }

  return finalizeGeometry(acc);
}

export function buildFigureRig(
  bodyMaterial: THREE.Material,
  hairMaterial: THREE.Material,
  featuresMaterial: THREE.Material,
  strokesMaterial: THREE.Material,
  mothWingMaterialFactory: (mothId: number) => THREE.Material,
  swirlMaterial: THREE.Material,
  flecksMaterial: THREE.Material
): FigureRigHandles {
  const root = new THREE.Group();
  root.name = 'CurledYoungWomanRig';
  root.position.set(FIG_ORIGIN[0], FIG_ORIGIN[1], FIG_ORIGIN[2]);
  root.rotation.y = -((90 - FIG_YAW_DEG) * Math.PI) / 180;

  const bodyAcc = createAccum();

  // 1. Smoothly Curved Seated Spine & Sweater Torso (parallel-transport Bishop frame)
  addLoftedTube(
    bodyAcc,
    smoothStations(
      [
        { center: [0.0, 0.018, -0.028], rx: 0.062, ry: 0.048 },
        { center: [0.0, 0.068, -0.068], rx: 0.064, ry: 0.048 },
        { center: [0.0, 0.132, -0.088], rx: 0.068, ry: 0.050 },
        { center: [0.0, 0.194, -0.068], rx: 0.070, ry: 0.048 },
        { center: [0.0, 0.242, -0.026], rx: 0.064, ry: 0.042 },
        { center: [0.0, 0.268, 0.010], rx: 0.036, ry: 0.028 },
      ],
      3
    ),
    28,
    1,
    true
  );

  // Ribbed Crewneck Collar Band
  addLoftedTube(
    bodyAcc,
    smoothStations(
      [
        { center: [0.0, 0.252, 0.002], rx: 0.035, ry: 0.031 },
        { center: [0.0, 0.264, 0.010], rx: 0.032, ry: 0.028 },
        { center: [0.0, 0.274, 0.018], rx: 0.029, ry: 0.026 },
      ],
      2
    ),
    24,
    1,
    true
  );

  // Smooth Neck rising forward into bowed head
  addLoftedTube(
    bodyAcc,
    smoothStations(
      [
        { center: [0.0, 0.254, 0.012], rx: 0.023, ry: 0.022 },
        { center: [0.0, 0.282, 0.028], rx: 0.021, ry: 0.020 },
        { center: [0.0, 0.304, 0.046], rx: 0.022, ry: 0.021 },
      ],
      3
    ),
    22,
    0,
    true
  );

  // 2. Sculpted Bowed Head & Face (tilted forward-down toward cupped hands)
  const headMat = appendSculptedHead(bodyAcc, [0.0, 0.312, 0.062], 0.44);

  // 3. Drawn-Up Knees, Shins & Bare Sketched Feet (knees drawn tight to chest/wrists for Beat 01!)
  for (const side of [-1, 1] as const) {
    const kLift = side === 1 ? 0.008 : -0.004;
    // Continuous leg sweep from hip -> raised knee -> sloped shin -> tucked bare foot!
    addLoftedTube(
      bodyAcc,
      smoothStations(
        [
          { center: [side * 0.036, 0.032, -0.024], rx: 0.042, ry: 0.036 },
          { center: [side * 0.042, 0.124 + kLift * 0.5, 0.072], rx: 0.034, ry: 0.032 },
          { center: [side * 0.044, 0.212 + kLift, 0.156], rx: 0.029, ry: 0.028 }, // high knee!
          { center: [side * 0.044, 0.186 + kLift, 0.168], rx: 0.025, ry: 0.025 }, // knee cap bend
          { center: [side * 0.043, 0.104 + kLift * 0.4, 0.194], rx: 0.022, ry: 0.022 }, // sloped shin
          { center: [side * 0.040, 0.026, 0.215], rx: 0.017, ry: 0.016 }, // ankle
          { center: [side * 0.038, 0.012, 0.244], rx: 0.015, ry: 0.011 }, // heel/arch
          { center: [side * 0.036, 0.008, 0.268], rx: 0.012, ry: 0.008 }, // toes
        ],
        3
      ),
      22,
      1,
      true
    );
  }

  // 4. Sweater Sleeves hugging outside of knees + Anatomically Cupped Hands tucked neatly beside knees
  for (const side of [-1, 1] as const) {
    addLoftedTube(
      bodyAcc,
      smoothStations(
        [
          { center: [side * 0.056, 0.236, -0.020], rx: 0.028, ry: 0.026 },
          { center: [side * 0.062, 0.194, 0.036], rx: 0.025, ry: 0.024 },
          { center: [side * 0.056, 0.174, 0.088], rx: 0.023, ry: 0.022 },
          { center: [side * 0.048, 0.186, 0.118], rx: 0.020, ry: 0.019 },
          { center: [side * 0.042, 0.192, 0.134], rx: 0.016, ry: 0.015 },
        ],
        3
      ),
      20,
      1,
      true
    );

    appendCuppedHand(bodyAcc, side, [side * 0.042, 0.194, 0.136]);
  }

  const bodyMesh = new THREE.Mesh(finalizeGeometry(bodyAcc), bodyMaterial);
  bodyMesh.name = 'YoungWomanBody';
  root.add(bodyMesh);

  const hairMesh = new THREE.Mesh(buildHairAndTendrilsGeometry(headMat), hairMaterial);
  hairMesh.name = 'YoungWomanHairAndTendrils';
  root.add(hairMesh);

  const faceFeaturesMesh = new THREE.Mesh(buildFaceFeaturesGeometry(headMat), featuresMaterial);
  faceFeaturesMesh.name = 'YoungWomanFaceFeatures';
  root.add(faceFeaturesMesh);

  const sketchStrokesMesh = new THREE.Mesh(buildSketchStrokesGeometry(), strokesMaterial);
  sketchStrokesMesh.name = 'YoungWomanSketchStrokes';
  root.add(sketchStrokesMesh);

  const swirlRibbonsMesh = new THREE.Mesh(buildSwirlRibbonsGeometry(), swirlMaterial);
  swirlRibbonsMesh.name = 'GraphiteSwirlRibbons';
  root.add(swirlRibbonsMesh);

  const leftWingGeom = buildOrigamiWingGeometry(-1);
  const rightWingGeom = buildOrigamiWingGeometry(1);
  const mothGroups: THREE.Group[] = [];
  const mothLeftWings: THREE.Mesh[] = [];
  const mothRightWings: THREE.Mesh[] = [];

  for (let i = 0; i < 9; i++) {
    const mg = new THREE.Group();
    mg.name = `OrigamiMoth_${i}`;
    const mMat = mothWingMaterialFactory(i);
    const lw = new THREE.Mesh(leftWingGeom, mMat);
    const rw = new THREE.Mesh(rightWingGeom, mMat);
    mg.add(lw);
    mg.add(rw);
    mothGroups.push(mg);
    mothLeftWings.push(lw);
    mothRightWings.push(rw);
  }

  const fleckGeom = new THREE.BufferGeometry();
  const fleckPos = new Float32Array([
    0.0, 0.0032, 0.0,
    0.0020, 0.0, 0.0006,
    0.0, -0.0032, 0.0,
    -0.0020, 0.0, -0.0006,
  ]);
  fleckGeom.setAttribute('position', new THREE.BufferAttribute(fleckPos, 3));
  fleckGeom.setIndex([0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]);
  fleckGeom.computeVertexNormals();

  const emberFlecksMesh = new THREE.InstancedMesh(fleckGeom, flecksMaterial, 28);
  emberFlecksMesh.name = 'RisingEmberFlecks';
  emberFlecksMesh.frustumCulled = false;

  return {
    root,
    bodyMesh,
    hairMesh,
    faceFeaturesMesh,
    sketchStrokesMesh,
    mothGroups,
    mothLeftWings,
    mothRightWings,
    swirlRibbonsMesh,
    emberFlecksMesh,
  };
}
