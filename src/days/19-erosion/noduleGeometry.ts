import * as THREE from 'three/webgpu';

export interface NozzleMaterials {
  satinBrass: THREE.Material;
  darkIron: THREE.Material;
  carbideTip: THREE.Material;
  hoseRubber: THREE.Material;
  abrasiveJet: THREE.Material;
}

export interface NozzleAssembly {
  group: THREE.Group;
  jetMesh: THREE.Mesh;
  dispose: () => void;
}

/**
 * Builds a watertight, strictly outward-wound welded geodesic icosphere on the unit sphere
 * (`detail = 6` -> 40,962 welded vertices, 81,920 triangles) for zero-seam, zero-pole-pinch
 * GPU octahedral heightfield displacement of the stratified limestone nodule.
 */
export function buildNoduleShellGeometry(detail = 6): THREE.BufferGeometry {
  const phi = (1 + Math.sqrt(5)) * 0.5;
  const rawVerts: number[] = [
    -1, phi, 0,
    1, phi, 0,
    -1, -phi, 0,
    1, -phi, 0,
    0, -1, phi,
    0, 1, phi,
    0, -1, -phi,
    0, 1, -phi,
    phi, 0, -1,
    phi, 0, 1,
    -phi, 0, -1,
    -phi, 0, 1,
  ];

  // Normalize initial 12 icosahedron vertices onto the unit sphere
  for (let i = 0; i < rawVerts.length; i += 3) {
    const l = Math.hypot(rawVerts[i], rawVerts[i + 1], rawVerts[i + 2]);
    rawVerts[i] /= l;
    rawVerts[i + 1] /= l;
    rawVerts[i + 2] /= l;
  }

  let faces: number[] = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];

  // Ensure every seed face is outward-wound before subdivision
  for (let f = 0; f < faces.length; f += 3) {
    const i0 = faces[f] * 3;
    const i1 = faces[f + 1] * 3;
    const i2 = faces[f + 2] * 3;
    const ax = rawVerts[i0];
    const ay = rawVerts[i0 + 1];
    const az = rawVerts[i0 + 2];
    const bx = rawVerts[i1] - ax;
    const by = rawVerts[i1 + 1] - ay;
    const bz = rawVerts[i1 + 2] - az;
    const cx = rawVerts[i2] - ax;
    const cy = rawVerts[i2 + 1] - ay;
    const cz = rawVerts[i2 + 2] - az;
    const nx = by * cz - bz * cy;
    const ny = bz * cx - bx * cz;
    const nz = bx * cy - by * cx;
    if (nx * ax + ny * ay + nz * az < 0) {
      const tmp = faces[f + 1];
      faces[f + 1] = faces[f + 2];
      faces[f + 2] = tmp;
    }
  }

  for (let step = 0; step < detail; step++) {
    const midpointCache = new Map<number, number>();
    const nextFaces: number[] = new Array(faces.length * 4);
    let nf = 0;

    const getMidpoint = (a: number, b: number): number => {
      const min = a < b ? a : b;
      const max = a < b ? b : a;
      const key = min * 65536 + max;
      const existing = midpointCache.get(key);
      if (existing !== undefined) return existing;

      const mx = 0.5 * (rawVerts[a * 3] + rawVerts[b * 3]);
      const my = 0.5 * (rawVerts[a * 3 + 1] + rawVerts[b * 3 + 1]);
      const mz = 0.5 * (rawVerts[a * 3 + 2] + rawVerts[b * 3 + 2]);
      const inv = 1.0 / Math.hypot(mx, my, mz);
      const newIdx = rawVerts.length / 3;
      rawVerts.push(mx * inv, my * inv, mz * inv);
      midpointCache.set(key, newIdx);
      return newIdx;
    };

    for (let f = 0; f < faces.length; f += 3) {
      const a = faces[f];
      const b = faces[f + 1];
      const c = faces[f + 2];
      const ab = getMidpoint(a, b);
      const bc = getMidpoint(b, c);
      const ca = getMidpoint(c, a);

      nextFaces[nf++] = a;
      nextFaces[nf++] = ab;
      nextFaces[nf++] = ca;

      nextFaces[nf++] = b;
      nextFaces[nf++] = bc;
      nextFaces[nf++] = ab;

      nextFaces[nf++] = c;
      nextFaces[nf++] = ca;
      nextFaces[nf++] = bc;

      nextFaces[nf++] = ab;
      nextFaces[nf++] = bc;
      nextFaces[nf++] = ca;
    }

    faces = nextFaces;
  }

  const positions = new Float32Array(rawVerts);
  // On unit sphere, vertex normals equal positions initially (TSL fragment computes exact carved normals)
  const normals = new Float32Array(rawVerts);
  const indices = new Uint32Array(faces);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1.45);
  geo.boundingBox = new THREE.Box3(
    new THREE.Vector3(-1.35, -1.35, -1.35),
    new THREE.Vector3(1.35, 1.35, 1.35)
  );
  return geo;
}

/**
 * Helper to create a strictly outward-wound closed LatheGeometry along the `+Z` axis.
 * Input `profile` MUST be ordered with `y` strictly non-decreasing (bottom-to-top) and
 * start/end closed at tiny positive radius `0.0002` so Three.js `LatheGeometry` produces
 * strictly outward normals (`signedVolume > 0`, `inwardTri === 0`).
 */
function buildClosedLatheAlongZ(
  profilePoints: Array<[number, number]>,
  segments = 36
): THREE.BufferGeometry {
  const vecs = profilePoints.map(([r, z]) => new THREE.Vector2(r, z));
  const geo = new THREE.LatheGeometry(vecs, segments);
  // Rigid +90 deg rotation around X maps +Y -> +Z while preserving det = +1 outward winding
  geo.rotateX(Math.PI * 0.5);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a watertight capped curved tube along a cubic spline with guaranteed outward winding.
 */
function buildCappedTubeAlongCurve(
  curve: THREE.Curve<THREE.Vector3>,
  tubularSegments: number,
  radiusStart: number,
  radiusEnd: number,
  radialSegments: number
): THREE.BufferGeometry {
  const frames = curve.computeFrenetFrames(tubularSegments, false);
  const ringCount = tubularSegments + 1;
  const totalVerts = ringCount * radialSegments + 2;
  const startCapIdx = ringCount * radialSegments;
  const endCapIdx = startCapIdx + 1;

  const positions = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const p = new THREE.Vector3();

  for (let i = 0; i <= tubularSegments; i++) {
    const u = i / tubularSegments;
    curve.getPointAt(u, p);
    const N = frames.normals[i];
    const B = frames.binormals[i];
    const rad = radiusStart + (radiusEnd - radiusStart) * u;

    for (let j = 0; j < radialSegments; j++) {
      const v = j / radialSegments;
      const ang = v * Math.PI * 2;
      const cosA = Math.cos(ang);
      const sinA = Math.sin(ang);
      const idx = i * radialSegments + j;

      positions[idx * 3] = p.x + rad * (cosA * N.x + sinA * B.x);
      positions[idx * 3 + 1] = p.y + rad * (cosA * N.y + sinA * B.y);
      positions[idx * 3 + 2] = p.z + rad * (cosA * N.z + sinA * B.z);
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  // Start & end cap centers
  curve.getPointAt(0, p);
  positions[startCapIdx * 3] = p.x - frames.tangents[0].x * 0.002;
  positions[startCapIdx * 3 + 1] = p.y - frames.tangents[0].y * 0.002;
  positions[startCapIdx * 3 + 2] = p.z - frames.tangents[0].z * 0.002;
  uvs[startCapIdx * 2] = 0;
  uvs[startCapIdx * 2 + 1] = 0.5;

  curve.getPointAt(1, p);
  positions[endCapIdx * 3] = p.x + frames.tangents[tubularSegments].x * 0.002;
  positions[endCapIdx * 3 + 1] = p.y + frames.tangents[tubularSegments].y * 0.002;
  positions[endCapIdx * 3 + 2] = p.z + frames.tangents[tubularSegments].z * 0.002;
  uvs[endCapIdx * 2] = 1;
  uvs[endCapIdx * 2 + 1] = 0.5;

  // Check Frenet frame handedness: T x N = B (right-handed -> (d/du) x (d/dv) = T x (-sin N + cos B) = -sin B - cos N = INWARD!)
  // Therefore outward quads use (a, c, b) and (a, d, c)! Let's verify:
  // (c - a) x (b - a) = (T + d/dv) x T = (d/dv) x T = (-sin N + cos B) x T = +sin B + cos N = OUTWARD!
  const triCount = tubularSegments * radialSegments * 2 + radialSegments * 2;
  const indices = new Uint32Array(triCount * 3);
  let ptr = 0;

  for (let i = 0; i < tubularSegments; i++) {
    const r0 = i * radialSegments;
    const r1 = (i + 1) * radialSegments;
    for (let j = 0; j < radialSegments; j++) {
      const jNext = (j + 1) % radialSegments;
      const a = r0 + j;
      const b = r1 + j;
      const c = r1 + jNext;
      const d = r0 + jNext;

      indices[ptr++] = a;
      indices[ptr++] = c;
      indices[ptr++] = b;

      indices[ptr++] = a;
      indices[ptr++] = d;
      indices[ptr++] = c;
    }
  }

  // Start cap (faces -T): (center, ring(0, jNext), ring(0, j)) -> (N x B = T, so B x N = -T)
  for (let j = 0; j < radialSegments; j++) {
    const jNext = (j + 1) % radialSegments;
    indices[ptr++] = startCapIdx;
    indices[ptr++] = jNext;
    indices[ptr++] = j;
  }

  // End cap (faces +T): (center, ring(last, j), ring(last, jNext)) -> N x B = +T
  const lastRow = tubularSegments * radialSegments;
  for (let j = 0; j < radialSegments; j++) {
    const jNext = (j + 1) % radialSegments;
    indices[ptr++] = endCapIdx;
    indices[ptr++] = lastRow + j;
    indices[ptr++] = lastRow + jNext;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds the machined Satin-Brass & Forged-Iron Pneumatic Air-Abrasion Fossil-Prep Stylus.
 * Orifice tip sits precisely at local `(0, 0, 0)`, body extends along `+Z`, and the focused
 * dolomite abrasive jet fires cleanly along local `-Z`.
 */
export function buildNozzleAssembly(materials: NozzleMaterials): NozzleAssembly {
  const group = new THREE.Group();
  group.name = 'AirAbrasionNozzleGroup';

  const geometries: THREE.BufferGeometry[] = [];

  // 1. Sintered Tungsten-Carbide Micro-Tip with crisp chamfered orifice lip (z = 0.000 -> 0.142)
  const tipGeo = buildClosedLatheAlongZ(
    [
      [0.0002, 0.0],
      [0.0048, 0.0004],
      [0.0072, 0.003],
      [0.0088, 0.012],
      [0.0114, 0.048],
      [0.0138, 0.068],
      [0.0156, 0.072],
      [0.0188, 0.114],
      [0.0195, 0.138],
      [0.0002, 0.142],
    ],
    32
  );
  geometries.push(tipGeo);
  const tipMesh = new THREE.Mesh(tipGeo, materials.carbideTip);
  tipMesh.name = 'CarbideMicroTip';
  tipMesh.castShadow = true;
  tipMesh.receiveShadow = true;
  group.add(tipMesh);

  // 2. Turned Satin-Brass Compression Collet & Ergonomic Diamond-Knurled Grip Barrel (z = 0.110 -> 0.568)
  const brassBodyGeo = buildClosedLatheAlongZ(
    [
      [0.0002, 0.11],
      [0.0225, 0.112],
      [0.0278, 0.118],
      [0.0308, 0.128],
      [0.0315, 0.164],
      [0.0295, 0.174],
      [0.0245, 0.184],
      [0.0245, 0.214],
      [0.0278, 0.222],
      [0.0365, 0.236],
      [0.0412, 0.252],
      [0.0418, 0.264],
      [0.0418, 0.514],
      [0.0408, 0.526],
      [0.0365, 0.544],
      [0.0335, 0.562],
      [0.0002, 0.568],
    ],
    44
  );
  geometries.push(brassBodyGeo);
  const brassBodyMesh = new THREE.Mesh(brassBodyGeo, materials.satinBrass);
  brassBodyMesh.name = 'BrassColletGrip';
  brassBodyMesh.castShadow = true;
  brassBodyMesh.receiveShadow = true;
  group.add(brassBodyMesh);

  // 3. Oil-Blackened Forged-Iron Pneumatic Handpiece Sleeve (z = 0.545 -> 0.966)
  const ironSleeveGeo = buildClosedLatheAlongZ(
    [
      [0.0002, 0.545],
      [0.0345, 0.549],
      [0.0372, 0.564],
      [0.0372, 0.614],
      [0.0355, 0.619],
      [0.0355, 0.646],
      [0.0372, 0.651],
      [0.037, 0.854],
      [0.0355, 0.859],
      [0.0355, 0.886],
      [0.037, 0.891],
      [0.0358, 0.928],
      [0.0325, 0.948],
      [0.0272, 0.962],
      [0.0002, 0.966],
    ],
    36
  );
  geometries.push(ironSleeveGeo);
  const ironSleeveMesh = new THREE.Mesh(ironSleeveGeo, materials.darkIron);
  ironSleeveMesh.name = 'ForgedIronSleeve';
  ironSleeveMesh.castShadow = true;
  ironSleeveMesh.receiveShadow = true;
  group.add(ironSleeveMesh);

  // 4. Twin Precision Chamfered Satin-Brass Accent Trim Rings & Stepped Rear Hose Barb Fitting
  const ring1Geo = buildClosedLatheAlongZ(
    [
      [0.0002, 0.617],
      [0.0375, 0.619],
      [0.0412, 0.625],
      [0.0412, 0.639],
      [0.0375, 0.645],
      [0.0002, 0.647],
    ],
    36
  );
  const ring2Geo = buildClosedLatheAlongZ(
    [
      [0.0002, 0.857],
      [0.0375, 0.859],
      [0.0412, 0.865],
      [0.0412, 0.879],
      [0.0375, 0.885],
      [0.0002, 0.887],
    ],
    36
  );
  const barbGeo = buildClosedLatheAlongZ(
    [
      [0.0002, 0.95],
      [0.0285, 0.954],
      [0.0315, 0.962],
      [0.0315, 0.99],
      [0.0285, 0.996],
      [0.0215, 1.006],
      [0.0245, 1.03],
      [0.0175, 1.05],
      [0.0205, 1.074],
      [0.0148, 1.106],
      [0.0002, 1.114],
    ],
    32
  );
  geometries.push(ring1Geo, ring2Geo, barbGeo);

  const ring1 = new THREE.Mesh(ring1Geo, materials.satinBrass);
  ring1.name = 'BrassTrimRingFore';
  const ring2 = new THREE.Mesh(ring2Geo, materials.satinBrass);
  ring2.name = 'BrassTrimRingAft';
  const barb = new THREE.Mesh(barbGeo, materials.satinBrass);
  barb.name = 'BrassHoseBarb';
  group.add(ring1, ring2, barb);

  // 5. Graceful Braided Pneumatic Supply Hose Tail curving gently back & away (+Y/+Z) off-frame
  const hoseCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0, 1.08),
    new THREE.Vector3(0.03, 0.025, 1.29),
    new THREE.Vector3(0.14, 0.11, 1.54),
    new THREE.Vector3(0.32, 0.24, 1.76),
  ]);
  const hoseGeo = buildCappedTubeAlongCurve(hoseCurve, 26, 0.0142, 0.0128, 16);
  geometries.push(hoseGeo);
  const hoseMesh = new THREE.Mesh(hoseGeo, materials.hoseRubber);
  hoseMesh.name = 'PneumaticAirHose';
  group.add(hoseMesh);

  // 6. Focused High-Velocity Dolomite Abrasive Jet Plume (local z = 0 -> -1, scaled to exact standoff)
  // Ordered bottom-to-top from z = -1.0 (impact flare radius 0.038) to z = -0.002 (orifice radius 0.0048)
  const jetProfile = [
    new THREE.Vector2(0.038, -1.0),
    new THREE.Vector2(0.021, -0.55),
    new THREE.Vector2(0.0095, -0.15),
    new THREE.Vector2(0.0048, -0.002),
  ];
  const jetGeo = new THREE.LatheGeometry(jetProfile, 24);
  jetGeo.rotateX(Math.PI * 0.5);
  jetGeo.computeVertexNormals();
  geometries.push(jetGeo);

  const jetMesh = new THREE.Mesh(jetGeo, materials.abrasiveJet);
  jetMesh.name = 'AbrasiveGritJet';
  jetMesh.visible = false;
  jetMesh.renderOrder = 3;
  group.add(jetMesh);

  return {
    group,
    jetMesh,
    dispose: () => {
      for (const g of geometries) g.dispose();
    },
  };
}

/**
 * Builds a cleaved irregular mineral polyhedron (faceted dolomite grit & limestone chip)
 * with strictly outward triangle winding for zero-GC InstancedMesh debris rendering.
 */
export function buildCleavedChipGeometry(): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(1.0, 0);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  // Slightly flatten & cleave asymmetrically into a natural rock spall flake
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const skew = 0.78 + 0.34 * Math.abs(Math.sin(x * 3.1 + y * 2.7 - z * 1.9));
    pos.setXYZ(i, x * 1.15 * skew, y * 0.58 * skew, z * 0.92 * skew);
  }
  geo.computeVertexNormals();
  return geo;
}
