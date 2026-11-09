// Procedural 3D CAD Geometry Library for the Grand Coaxial, Da Vinci & Antikythera Orrery
// Vertex format: 9 floats [x, y, z, nx, ny, nz, u, v, edgeMask] (36 bytes/vertex)

export function buildOrreryGeometry() {
  const verts = [];
  const indices = [];
  const meshes = {};
  let vertCount = 0;

  function pushVert(x, y, z, nx, ny, nz, u = 0, v = 0, edge = 0) {
    const len = Math.hypot(nx, ny, nz) || 1;
    verts.push(x, y, z, nx / len, ny / len, nz / len, u, v, edge);
    return vertCount++;
  }

  function pushQuad(i0, i1, i2, i3) {
    indices.push(i0, i1, i2, i0, i2, i3);
  }

  function pushTri(i0, i1, i2) {
    indices.push(i0, i1, i2);
  }

  function beginMesh() {
    return indices.length;
  }

  function endMesh(name, startIdx) {
    meshes[name] = {
      firstIndex: startIdx,
      indexCount: indices.length - startIdx,
    };
  }

  // Flat-shaded quad with automatic normal calculation and edge highlight mask
  function addQuadFace(p0, p1, p2, p3, u0 = 0, v0 = 0, u1 = 1, v1 = 1, edgeVal = 0.25) {
    const ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
    const vx = p3[0] - p0[0], vy = p3[1] - p0[1], vz = p3[2] - p0[2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const i0 = pushVert(p0[0], p0[1], p0[2], nx, ny, nz, u0, v0, edgeVal);
    const i1 = pushVert(p1[0], p1[1], p1[2], nx, ny, nz, u1, v0, edgeVal);
    const i2 = pushVert(p2[0], p2[1], p2[2], nx, ny, nz, u1, v1, edgeVal);
    const i3 = pushVert(p3[0], p3[1], p3[2], nx, ny, nz, u0, v1, edgeVal);
    pushQuad(i0, i1, i2, i3);
  }

  // Helper: add a 3D box from [x0,y0,z0] to [x1,y1,z1] inside a mesh
  function addSubBox(x0, y0, z0, x1, y1, z1, edgeVal = 0.72) {
    addQuadFace([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], 0, 0, 1, 1, edgeVal);
    addQuadFace([x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0], 0, 0, 1, 1, edgeVal * 0.7);
    addQuadFace([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], 0, 0, 1, 1, edgeVal);
    addQuadFace([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1], 0, 0, 1, 1, edgeVal);
    addQuadFace([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], 0, 0, 1, 1, edgeVal);
    addQuadFace([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], 0, 0, 1, 1, edgeVal);
  }

  // Helper: add a cylinder sub-part along Z axis inside a mesh
  function addSubCylinderZ(cx, cy, rOut, rIn, z0, z1, segs = 24, edgeVal = 0.55) {
    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const u0 = i / segs, u1 = (i + 1) / segs;

      const sideEdge0 = edgeVal >= 0.88 ? edgeVal * 0.85 : 0;
      const sideEdge1 = edgeVal >= 0.88 ? edgeVal : 0;
      const p00 = pushVert(cx + c0 * rOut, cy + s0 * rOut, z0, c0, s0, 0, u0, 0, sideEdge0);
      const p10 = pushVert(cx + c1 * rOut, cy + s1 * rOut, z0, c1, s1, 0, u1, 0, sideEdge0);
      const p11 = pushVert(cx + c1 * rOut, cy + s1 * rOut, z1, c1, s1, 0, u1, 1, sideEdge1);
      const p01 = pushVert(cx + c0 * rOut, cy + s0 * rOut, z1, c0, s0, 0, u0, 1, sideEdge1);
      pushQuad(p00, p10, p11, p01);

      if (rIn > 0.01) {
        const t00 = pushVert(cx + c0 * rIn, cy + s0 * rIn, z1, 0, 0, 1, u0, 0, edgeVal);
        const t01 = pushVert(cx + c0 * rOut, cy + s0 * rOut, z1, 0, 0, 1, u0, 1, edgeVal);
        const t11 = pushVert(cx + c1 * rOut, cy + s1 * rOut, z1, 0, 0, 1, u1, 1, edgeVal);
        const t10 = pushVert(cx + c1 * rIn, cy + s1 * rIn, z1, 0, 0, 1, u1, 0, edgeVal);
        pushQuad(t00, t01, t11, t10);

        const b00 = pushVert(cx + c1 * rIn, cy + s1 * rIn, z0, 0, 0, -1, u1, 0, edgeVal * 0.7);
        const b01 = pushVert(cx + c1 * rOut, cy + s1 * rOut, z0, 0, 0, -1, u1, 1, edgeVal * 0.7);
        const b11 = pushVert(cx + c0 * rOut, cy + s0 * rOut, z0, 0, 0, -1, u0, 1, edgeVal * 0.7);
        const b10 = pushVert(cx + c0 * rIn, cy + s0 * rIn, z0, 0, 0, -1, u0, 0, edgeVal * 0.7);
        pushQuad(b00, b01, b11, b10);

        const w00 = pushVert(cx + c1 * rIn, cy + s1 * rIn, z0, -c1, -s1, 0, u1, 0, 0);
        const w10 = pushVert(cx + c0 * rIn, cy + s0 * rIn, z0, -c0, -s0, 0, u0, 0, 0);
        const w11 = pushVert(cx + c0 * rIn, cy + s0 * rIn, z1, -c0, -s0, 0, u0, 1, 0);
        const w01 = pushVert(cx + c1 * rIn, cy + s1 * rIn, z1, -c1, -s1, 0, u1, 1, 0);
        pushQuad(w00, w10, w11, w01);
      } else {
        const tc = pushVert(cx, cy, z1, 0, 0, 1, 0.5, 0.5, 0);
        const t0 = pushVert(cx + c0 * rOut, cy + s0 * rOut, z1, 0, 0, 1, c0 * 0.5 + 0.5, s0 * 0.5 + 0.5, edgeVal);
        const t1 = pushVert(cx + c1 * rOut, cy + s1 * rOut, z1, 0, 0, 1, c1 * 0.5 + 0.5, s1 * 0.5 + 0.5, edgeVal);
        pushTri(tc, t0, t1);

        const bc = pushVert(cx, cy, z0, 0, 0, -1, 0.5, 0.5, 0);
        const b0 = pushVert(cx + c1 * rOut, cy + s1 * rOut, z0, 0, 0, -1, c1 * 0.5 + 0.5, s1 * 0.5 + 0.5, edgeVal * 0.7);
        const b1 = pushVert(cx + c0 * rOut, cy + s0 * rOut, z0, 0, 0, -1, c0 * 0.5 + 0.5, s0 * 0.5 + 0.5, edgeVal * 0.7);
        pushTri(bc, b0, b1);
      }
    }
  }

  // 1. Solid Cylinder / Hollow Tube with Beveled Rims
  function buildCylinder(name, rOut, rIn, z0, z1, segs = 48, bevel = 0.2) {
    const start = beginMesh();
    const h = z1 - z0;
    const b = Math.min(bevel, h * 0.25, (rOut - rIn) * 0.35);

    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const u0 = i / segs, u1 = (i + 1) / segs;

      const p00 = pushVert(c0 * rOut, s0 * rOut, z0 + b, c0, s0, 0, u0, 0, 0);
      const p10 = pushVert(c1 * rOut, s1 * rOut, z0 + b, c1, s1, 0, u1, 0, 0);
      const p11 = pushVert(c1 * rOut, s1 * rOut, z1 - b, c1, s1, 0, u1, 1, 0);
      const p01 = pushVert(c0 * rOut, s0 * rOut, z1 - b, c0, s0, 0, u0, 1, 0);
      pushQuad(p00, p10, p11, p01);

      if (b > 0.001) {
        const b00 = pushVert(c0 * rOut, s0 * rOut, z1 - b, c0 * 0.7, s0 * 0.7, 0.7, u0, 0, 0.85);
        const b10 = pushVert(c1 * rOut, s1 * rOut, z1 - b, c1 * 0.7, s1 * 0.7, 0.7, u1, 0, 0.85);
        const b11 = pushVert(c1 * (rOut - b), s1 * (rOut - b), z1, c1 * 0.7, s1 * 0.7, 0.7, u1, 1, 0.85);
        const b01 = pushVert(c0 * (rOut - b), s0 * (rOut - b), z1, c0 * 0.7, s0 * 0.7, 0.7, u0, 1, 0.85);
        pushQuad(b00, b10, b11, b01);

        const d00 = pushVert(c0 * (rOut - b), s0 * (rOut - b), z0, c0 * 0.7, s0 * 0.7, -0.7, u0, 0, 0.6);
        const d10 = pushVert(c1 * (rOut - b), s1 * (rOut - b), z0, c1 * 0.7, s1 * 0.7, -0.7, u1, 0, 0.6);
        const d11 = pushVert(c1 * rOut, s1 * rOut, z0 + b, c1 * 0.7, s1 * 0.7, -0.7, u1, 1, 0.6);
        const d01 = pushVert(c0 * rOut, s0 * rOut, z0 + b, c0 * 0.7, s0 * 0.7, -0.7, u0, 1, 0.6);
        pushQuad(d00, d10, d11, d01);
      }

      const rTopOut = rOut - b;
      const rTopIn = rIn > 0.01 ? rIn : 0;

      if (rTopIn > 0.01) {
        const t00 = pushVert(c0 * rTopIn, s0 * rTopIn, z1, 0, 0, 1, c0 * 0.5 + 0.5, s0 * 0.5 + 0.5, 0.65);
        const t01 = pushVert(c0 * rTopOut, s0 * rTopOut, z1, 0, 0, 1, c0 * 0.95 + 0.5, s0 * 0.95 + 0.5, 0.35);
        const t11 = pushVert(c1 * rTopOut, s1 * rTopOut, z1, 0, 0, 1, c1 * 0.95 + 0.5, s1 * 0.95 + 0.5, 0.35);
        const t10 = pushVert(c1 * rTopIn, s1 * rTopIn, z1, 0, 0, 1, c1 * 0.5 + 0.5, s1 * 0.5 + 0.5, 0.65);
        pushQuad(t00, t01, t11, t10);

        const m00 = pushVert(c1 * rTopIn, s1 * rTopIn, z0, 0, 0, -1, u1, 0, 0.4);
        const m01 = pushVert(c1 * rTopOut, s1 * rTopOut, z0, 0, 0, -1, u1, 1, 0.4);
        const m11 = pushVert(c0 * rTopOut, s0 * rTopOut, z0, 0, 0, -1, u0, 1, 0.4);
        const m10 = pushVert(c0 * rTopIn, s0 * rTopIn, z0, 0, 0, -1, u0, 0, 0.4);
        pushQuad(m00, m01, m11, m10);

        const w00 = pushVert(c1 * rTopIn, s1 * rTopIn, z0, -c1, -s1, 0, u1, 0, 0);
        const w10 = pushVert(c0 * rTopIn, s0 * rTopIn, z0, -c0, -s0, 0, u0, 0, 0);
        const w11 = pushVert(c0 * rTopIn, s0 * rTopIn, z1, -c0, -s0, 0, u0, 1, 0);
        const w01 = pushVert(c1 * rTopIn, s1 * rTopIn, z1, -c1, -s1, 0, u1, 1, 0);
        pushQuad(w00, w10, w11, w01);
      } else {
        const tc = pushVert(0, 0, z1, 0, 0, 1, 0.5, 0.5, 0);
        const t0 = pushVert(c0 * rTopOut, s0 * rTopOut, z1, 0, 0, 1, c0 * 0.5 + 0.5, s0 * 0.5 + 0.5, 0.35);
        const t1 = pushVert(c1 * rTopOut, s1 * rTopOut, z1, 0, 0, 1, c1 * 0.5 + 0.5, s1 * 0.5 + 0.5, 0.35);
        pushTri(tc, t0, t1);

        const bc = pushVert(0, 0, z0, 0, 0, -1, 0.5, 0.5, 0);
        const b0 = pushVert(c1 * rTopOut, s1 * rTopOut, z0, 0, 0, -1, c1 * 0.5 + 0.5, s1 * 0.5 + 0.5, 0.35);
        const b1 = pushVert(c0 * rTopOut, s0 * rTopOut, z0, 0, 0, -1, c0 * 0.5 + 0.5, s0 * 0.5 + 0.5, 0.35);
        pushTri(bc, b0, b1);
      }
    }
    endMesh(name, start);
  }

  // 2. Rounded Rectangular Box / Slab (centered at 0,0, z0..z1)
  function buildRoundedBox(name, w, d, z0, z1, rCorner = 3.0, bevel = 0.6, cornerSegs = 6) {
    const start = beginMesh();
    const hw = w * 0.5, hd = d * 0.5;
    const rc = Math.min(rCorner, hw * 0.45, hd * 0.45);
    const b = Math.min(bevel, (z1 - z0) * 0.25, rc * 0.4);

    const ring = [];
    const corners = [
      [hw - rc, hd - rc, 0],
      [-hw + rc, hd - rc, Math.PI * 0.5],
      [-hw + rc, -hd + rc, Math.PI],
      [hw - rc, -hd + rc, Math.PI * 1.5],
    ];
    for (const [cx, cy, aStart] of corners) {
      for (let i = 0; i <= cornerSegs; i++) {
        const a = aStart + (i / cornerSegs) * (Math.PI * 0.5);
        const ca = Math.cos(a), sa = Math.sin(a);
        ring.push({ x: cx + ca * rc, y: cy + sa * rc, nx: ca, ny: sa });
      }
    }
    const N = ring.length;
    for (let i = 0; i < N; i++) {
      const p0 = ring[i];
      const p1 = ring[(i + 1) % N];
      const u0 = i / N, u1 = (i + 1) / N;

      const s00 = pushVert(p0.x, p0.y, z0 + b, p0.nx, p0.ny, 0, u0, 0, 0);
      const s10 = pushVert(p1.x, p1.y, z0 + b, p1.nx, p1.ny, 0, u1, 0, 0);
      const s11 = pushVert(p1.x, p1.y, z1 - b, p1.nx, p1.ny, 0, u1, 1, 0);
      const s01 = pushVert(p0.x, p0.y, z1 - b, p0.nx, p0.ny, 0, u0, 1, 0);
      pushQuad(s00, s10, s11, s01);

      const p0InX = p0.x - p0.nx * b, p0InY = p0.y - p0.ny * b;
      const p1InX = p1.x - p1.nx * b, p1InY = p1.y - p1.ny * b;
      const tb00 = pushVert(p0.x, p0.y, z1 - b, p0.nx * 0.7, p0.ny * 0.7, 0.7, u0, 0, 0.85);
      const tb10 = pushVert(p1.x, p1.y, z1 - b, p1.nx * 0.7, p1.ny * 0.7, 0.7, u1, 0, 0.85);
      const tb11 = pushVert(p1InX, p1InY, z1, p1.nx * 0.7, p1.ny * 0.7, 0.7, u1, 1, 0.85);
      const tb01 = pushVert(p0InX, p0InY, z1, p0.nx * 0.7, p0.ny * 0.7, 0.7, u0, 1, 0.85);
      pushQuad(tb00, tb10, tb11, tb01);

      const tc = pushVert(0, 0, z1, 0, 0, 1, 0.5, 0.5, 0);
      const tf0 = pushVert(p0InX, p0InY, z1, 0, 0, 1, p0InX / w + 0.5, p0InY / d + 0.5, 0.35);
      const tf1 = pushVert(p1InX, p1InY, z1, 0, 0, 1, p1InX / w + 0.5, p1InY / d + 0.5, 0.35);
      pushTri(tc, tf0, tf1);

      const bc = pushVert(0, 0, z0, 0, 0, -1, 0.5, 0.5, 0);
      const bf0 = pushVert(p1.x, p1.y, z0, 0, 0, -1, 0, 0, 0.35);
      const bf1 = pushVert(p0.x, p0.y, z0, 0, 0, -1, 1, 0, 0.35);
      pushTri(bc, bf0, bf1);
    }
    endMesh(name, start);
  }

  // 3. True 3D Involute Spur Gear with Recessed Web, Tapered Spokes, Hub Boss & Exact Bore
  function buildSpurGear(name, rPitch, teeth, h, rHub, spokes = 4, rBoreOverride = null) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const mod = (2 * rPitch) / teeth;
    const rAdd = rPitch + mod * 0.42;
    const rDed = Math.max(rHub + 0.42, rPitch - mod * 0.50);
    const rRimIn = Math.max(rHub + 0.36, rDed - Math.min(1.02, rPitch * 0.086));
    const webZ = h * 0.24;

    const profile = [];
    for (let t = 0; t < teeth; t++) {
      const aBase = (t / teeth) * Math.PI * 2;
      const pitchA = (Math.PI * 2) / teeth;
      const pts = [
        [aBase + pitchA * 0.00, rDed],
        [aBase + pitchA * 0.16, rDed],
        [aBase + pitchA * 0.27, rPitch],
        [aBase + pitchA * 0.34, rAdd],
        [aBase + pitchA * 0.66, rAdd],
        [aBase + pitchA * 0.73, rPitch],
        [aBase + pitchA * 0.84, rDed],
      ];
      for (const [ang, r] of pts) {
        profile.push({ x: Math.cos(ang) * r, y: Math.sin(ang) * r, ang, r });
      }
    }

    const P = profile.length;
    for (let i = 0; i < P; i++) {
      const p0 = profile[i];
      const p1 = profile[(i + 1) % P];
      const dx = p1.x - p0.x, dy = p1.y - p0.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = dy / len, ny = -dx / len;
      const isTipOrCorner = Math.abs(p0.r - p1.r) > 0.05 ? 0.88 : 0.48;

      const v00 = pushVert(p0.x, p0.y, z0, nx, ny, 0, i / P, 0, isTipOrCorner);
      const v10 = pushVert(p1.x, p1.y, z0, nx, ny, 0, (i + 1) / P, 0, isTipOrCorner);
      const v11 = pushVert(p1.x, p1.y, z1, nx, ny, 0, (i + 1) / P, 1, isTipOrCorner);
      const v01 = pushVert(p0.x, p0.y, z1, nx, ny, 0, i / P, 1, isTipOrCorner);
      pushQuad(v00, v10, v11, v01);

      const c0 = Math.cos(p0.ang), s0 = Math.sin(p0.ang);
      const c1 = Math.cos(p1.ang), s1 = Math.sin(p1.ang);

      const t00 = pushVert(c0 * rRimIn, s0 * rRimIn, z1, 0, 0, 1, 0, i / P, 0.72);
      const t10 = pushVert(p0.x, p0.y, z1, 0, 0, 1, 1, i / P, 0.72);
      const t11 = pushVert(p1.x, p1.y, z1, 0, 0, 1, 1, (i + 1) / P, 0.72);
      const t01 = pushVert(c1 * rRimIn, s1 * rRimIn, z1, 0, 0, 1, 0, (i + 1) / P, 0.72);
      pushQuad(t00, t10, t11, t01);

      const b00 = pushVert(c1 * rRimIn, s1 * rRimIn, z0, 0, 0, -1, 0, (i + 1) / P, 0.5);
      const b10 = pushVert(p1.x, p1.y, z0, 0, 0, -1, 1, (i + 1) / P, 0.5);
      const b11 = pushVert(p0.x, p0.y, z0, 0, 0, -1, 1, i / P, 0.5);
      const b01 = pushVert(c0 * rRimIn, s0 * rRimIn, z0, 0, 0, -1, 0, i / P, 0.5);
      pushQuad(b00, b10, b11, b01);
    }

    const rimSegs = 48;
    for (let i = 0; i < rimSegs; i++) {
      const a0 = (i / rimSegs) * Math.PI * 2;
      const a1 = ((i + 1) / rimSegs) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      const w00 = pushVert(c1 * rRimIn, s1 * rRimIn, z0, -c1, -s1, 0, 0, 0, 0.68);
      const w10 = pushVert(c0 * rRimIn, s0 * rRimIn, z0, -c0, -s0, 0, 1, 0, 0.68);
      const w11 = pushVert(c0 * rRimIn, s0 * rRimIn, z1, -c0, -s0, 0, 1, 1, 0.68);
      const w01 = pushVert(c1 * rRimIn, s1 * rRimIn, z1, -c1, -s1, 0, 0, 1, 0.68);
      pushQuad(w00, w10, w11, w01);

      if (spokes === 0 || rRimIn - rHub < 0.52) {
        const f00 = pushVert(c0 * rHub, s0 * rHub, webZ, 0, 0, 1, 0.2, i / rimSegs, 0.45);
        const f10 = pushVert(c0 * rRimIn, s0 * rRimIn, webZ, 0, 0, 1, 0.9, i / rimSegs, 0.45);
        const f11 = pushVert(c1 * rRimIn, s1 * rRimIn, webZ, 0, 0, 1, 0.9, (i + 1) / rimSegs, 0.45);
        const f01 = pushVert(c1 * rHub, s1 * rHub, webZ, 0, 0, 1, 0.2, (i + 1) / rimSegs, 0.45);
        pushQuad(f00, f10, f11, f01);

        const fb00 = pushVert(c1 * rHub, s1 * rHub, -webZ, 0, 0, -1, 0.2, (i + 1) / rimSegs, 0.4);
        const fb10 = pushVert(c1 * rRimIn, s1 * rRimIn, -webZ, 0, 0, -1, 0.9, (i + 1) / rimSegs, 0.4);
        const fb11 = pushVert(c0 * rRimIn, s0 * rRimIn, -webZ, 0, 0, -1, 0.9, i / rimSegs, 0.4);
        const fb01 = pushVert(c0 * rHub, s0 * rHub, -webZ, 0, 0, -1, 0.2, i / rimSegs, 0.4);
        pushQuad(fb00, fb10, fb11, fb01);
      }
    }

    // Tapered Swiss horological crossed-out spokes with 45° polished anglage bevels
    if (spokes > 0 && rRimIn - rHub >= 0.52) {
      const swRoot = Math.max(0.32, Math.min(0.96, rPitch * 0.064));
      const swTip = swRoot * 0.62;
      const r0 = rHub * 0.88;
      const r1 = rRimIn * 1.03;
      const sSegs = 5;
      const bev = Math.min(0.14, webZ * 0.35);
      for (let s = 0; s < spokes; s++) {
        const ang = (s / spokes) * Math.PI * 2;
        const ux = Math.cos(ang), uy = Math.sin(ang);
        const vx = -uy, vy = ux;
        for (let k = 0; k < sSegs; k++) {
          const t0 = k / sSegs, t1 = (k + 1) / sSegs;
          const ra = r0 + (r1 - r0) * t0;
          const rb = r0 + (r1 - r0) * t1;
          // Gracefully waisted Swiss wheel spoke profile flaring at hub and rim roots
          const w0 = swRoot * (1 - t0) + swTip * t0 - 0.18 * swRoot * Math.sin(t0 * Math.PI);
          const w1 = swRoot * (1 - t1) + swTip * t1 - 0.18 * swRoot * Math.sin(t1 * Math.PI);
          const pA = [ux * ra - vx * w0, uy * ra - vy * w0];
          const pB = [ux * rb - vx * w1, uy * rb - vy * w1];
          const pC = [ux * rb + vx * w1, uy * rb + vy * w1];
          const pD = [ux * ra + vx * w0, uy * ra + vy * w0];
          const pAi = [ux * ra - vx * (w0 - bev), uy * ra - vy * (w0 - bev)];
          const pBi = [ux * rb - vx * (w1 - bev), uy * rb - vy * (w1 - bev)];
          const pCi = [ux * rb + vx * (w1 - bev), uy * rb + vy * (w1 - bev)];
          const pDi = [ux * ra + vx * (w0 - bev), uy * ra + vy * (w0 - bev)];
          // Top flat table + 45° anglage bevel strips
          addQuadFace([pAi[0], pAi[1], webZ], [pBi[0], pBi[1], webZ], [pCi[0], pCi[1], webZ], [pDi[0], pDi[1], webZ], t0, 0, t1, 1, 0.84);
          addQuadFace([pA[0], pA[1], webZ - bev], [pB[0], pB[1], webZ - bev], [pBi[0], pBi[1], webZ], [pAi[0], pAi[1], webZ], t0, 0, t1, 1, 0.96);
          addQuadFace([pCi[0], pCi[1], webZ], [pC[0], pC[1], webZ - bev], [pD[0], pD[1], webZ - bev], [pDi[0], pDi[1], webZ], t0, 0, t1, 1, 0.96);
          // Bottom soffit + side flanks
          addQuadFace([pD[0], pD[1], -webZ], [pC[0], pC[1], -webZ], [pB[0], pB[1], -webZ], [pA[0], pA[1], -webZ], t0, 0, t1, 1, 0.55);
          addQuadFace([pA[0], pA[1], -webZ], [pB[0], pB[1], -webZ], [pB[0], pB[1], webZ - bev], [pA[0], pA[1], webZ - bev], t0, 0, t1, 1, 0.76);
          addQuadFace([pC[0], pC[1], -webZ], [pD[0], pD[1], -webZ], [pD[0], pD[1], webZ - bev], [pC[0], pC[1], webZ - bev], t0, 0, t1, 1, 0.76);
        }
      }
    } else {
      // Compact horological pinions receive 6 countersunk circular anglage lightening bezels around the web
      const rMidWeb = (rHub + rRimIn) * 0.5;
      const rBezel = Math.max(0.14, Math.min(0.32, (rRimIn - rHub) * 0.36));
      for (let hIdx = 0; hIdx < 6; hIdx++) {
        const ha = (hIdx / 6) * Math.PI * 2;
        addSubCylinderZ(Math.cos(ha) * rMidWeb, Math.sin(ha) * rMidWeb, rBezel, rBezel * 0.48, -webZ * 1.15, webZ * 1.15, 12, 0.92);
      }
    }

    // Central Raised Hub Boss with stepped anglage collet and exact inner bore
    const hubZ1 = z1 + h * 0.16;
    const hubZ0 = z0 - h * 0.12;
    const rBore = rBoreOverride !== null ? rBoreOverride : Math.max(0.75, rHub * 0.65);
    addSubCylinderZ(0, 0, rHub, rBore, hubZ0, hubZ1, 32, 0.82);

    endMesh(name, start);
  }

  // 4. Da Vinci Crown Wheel ("Ruota a Corona" — contrate gear with vertical face teeth along +Z)
  function buildCrownGear(name, rPitch, teeth, hRim, toothH, rHub, spokes = 4) {
    const start = beginMesh();
    const rIn = rPitch - 1.25;
    const rOut = rPitch + 1.25;
    const z0 = -hRim * 0.5;
    const z1 = hRim * 0.5;

    // Base ring & hub
    addSubCylinderZ(0, 0, rOut, rIn, z0, z1, 48, 0.7);
    addSubCylinderZ(0, 0, rHub, rHub * 0.55, z0 - 0.5, z1 + 0.6, 28, 0.75);

    // Spokes connecting hub to crown ring
    if (spokes > 0) {
      const sw = 0.85;
      for (let s = 0; s < spokes; s++) {
        const ang = (s / spokes) * Math.PI * 2;
        const ux = Math.cos(ang), uy = Math.sin(ang);
        const vx = -uy * sw, vy = ux * sw;
        const r0 = rHub * 0.85, r1 = rIn * 1.03;
        addQuadFace([ux * r0 - vx, uy * r0 - vy, z1 * 0.6], [ux * r1 - vx, uy * r1 - vy, z1 * 0.6], [ux * r1 + vx, uy * r1 + vy, z1 * 0.6], [ux * r0 + vx, uy * r0 + vy, z1 * 0.6], 0, 0, 1, 1, 0.75);
        addQuadFace([ux * r0 + vx, uy * r0 + vy, z0 * 0.6], [ux * r1 + vx, uy * r1 + vy, z0 * 0.6], [ux * r1 - vx, uy * r1 - vy, z0 * 0.6], [ux * r0 - vx, uy * r0 - vy, z0 * 0.6], 0, 0, 1, 1, 0.55);
      }
    }

    // Crown teeth projecting in +Z from z1 to z1 + toothH
    const pitchA = (Math.PI * 2) / teeth;
    for (let t = 0; t < teeth; t++) {
      const a0 = t * pitchA;
      const aB0 = a0 + pitchA * 0.14;
      const aT0 = a0 + pitchA * 0.28;
      const aT1 = a0 + pitchA * 0.58;
      const aB1 = a0 + pitchA * 0.72;

      const pB0_in = [Math.cos(aB0) * rIn, Math.sin(aB0) * rIn, z1];
      const pB0_out = [Math.cos(aB0) * rOut, Math.sin(aB0) * rOut, z1];
      const pT0_in = [Math.cos(aT0) * rIn, Math.sin(aT0) * rIn, z1 + toothH];
      const pT0_out = [Math.cos(aT0) * rOut, Math.sin(aT0) * rOut, z1 + toothH];
      const pT1_in = [Math.cos(aT1) * rIn, Math.sin(aT1) * rIn, z1 + toothH];
      const pT1_out = [Math.cos(aT1) * rOut, Math.sin(aT1) * rOut, z1 + toothH];
      const pB1_in = [Math.cos(aB1) * rIn, Math.sin(aB1) * rIn, z1];
      const pB1_out = [Math.cos(aB1) * rOut, Math.sin(aB1) * rOut, z1];

      // Outer, inner, leading flank, trailing flank, and crest of crown tooth
      addQuadFace(pB0_out, pB1_out, pT1_out, pT0_out, 0, 0, 1, 1, 0.85);
      addQuadFace(pB1_in, pB0_in, pT0_in, pT1_in, 0, 0, 1, 1, 0.75);
      addQuadFace(pB0_in, pB0_out, pT0_out, pT0_in, 0, 0, 1, 1, 0.88);
      addQuadFace(pB1_out, pB1_in, pT1_in, pT1_out, 0, 0, 1, 1, 0.88);
      addQuadFace(pT0_in, pT0_out, pT1_out, pT1_in, 0, 0, 1, 1, 0.92);
    }

    endMesh(name, start);
  }

  // 5. Da Vinci Lantern Pinion ("Rocchetto a Lanterna" — two shroud discs connected by cylindrical trundle staves)
  function buildLanternPinion(name, rPitch, staves, length, rShroud, rStave, rBore = 1.1) {
    const start = beginMesh();
    const z0 = -length * 0.5;
    const z1 = length * 0.5;
    const shroudT = Math.min(1.1, length * 0.18);

    // Bottom and top shroud discs + central arbor hub
    addSubCylinderZ(0, 0, rShroud, rBore, z0, z0 + shroudT, 32, 0.82);
    addSubCylinderZ(0, 0, rShroud, rBore, z1 - shroudT, z1, 32, 0.82);
    addSubCylinderZ(0, 0, rBore * 1.45, rBore * 0.75, z0 - 0.4, z1 + 0.4, 20, 0.65);

    // Ring of cylindrical steel trundle staves
    for (let k = 0; k < staves; k++) {
      const ang = (k / staves) * Math.PI * 2;
      const sx = Math.cos(ang) * rPitch;
      const sy = Math.sin(ang) * rPitch;
      addSubCylinderZ(sx, sy, rStave, 0, z0 + shroudT, z1 - shroudT, 12, 0.75);
    }

    endMesh(name, start);
  }

  // 6. 3D Conical Miter / Bevel Gear (axis +Z, cone apex toward +Z)
  function buildBevelGear(name, rBase, rTop, teeth, h) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const mod = (2 * rBase) / teeth;
    for (let t = 0; t < teeth; t++) {
      const a0 = (t / teeth) * Math.PI * 2;
      const pitch = (Math.PI * 2) / teeth;
      const aL = a0 + pitch * 0.18;
      const aM1 = a0 + pitch * 0.34;
      const aM2 = a0 + pitch * 0.66;
      const aR = a0 + pitch * 0.82;
      const aN = a0 + pitch;

      const rB_ded = rBase - mod * 0.35, rB_add = rBase + mod * 0.38;
      const rT_ded = rTop - mod * 0.25, rT_add = rTop + mod * 0.28;

      const ringB = [
        [a0, rB_ded], [aL, rB_ded], [aM1, rB_add], [aM2, rB_add], [aR, rB_ded], [aN, rB_ded],
      ];
      const ringT = [
        [a0, rT_ded], [aL, rT_ded], [aM1, rT_add], [aM2, rT_add], [aR, rT_ded], [aN, rT_ded],
      ];

      for (let k = 0; k < ringB.length - 1; k++) {
        const [ab0, rb0] = ringB[k];
        const [ab1, rb1] = ringB[k + 1];
        const [at0, rt0] = ringT[k];
        const [at1, rt1] = ringT[k + 1];
        const p0 = [Math.cos(ab0) * rb0, Math.sin(ab0) * rb0, z0];
        const p1 = [Math.cos(ab1) * rb1, Math.sin(ab1) * rb1, z0];
        const p2 = [Math.cos(at1) * rt1, Math.sin(at1) * rt1, z1];
        const p3 = [Math.cos(at0) * rt0, Math.sin(at0) * rt0, z1];
        addQuadFace(p0, p1, p2, p3, 0, 0, 1, 1, 0.78);

        const tc = pushVert(0, 0, z1, 0, 0, 1, 0.5, 0.5, 0);
        const t0 = pushVert(p3[0], p3[1], z1, 0, 0, 1, 0, 0, 0.65);
        const t1 = pushVert(p2[0], p2[1], z1, 0, 0, 1, 1, 0, 0.65);
        pushTri(tc, t0, t1);

        const bc = pushVert(0, 0, z0, 0, 0, -1, 0.5, 0.5, 0);
        const b0 = pushVert(p1[0], p1[1], z0, 0, 0, -1, 0, 0, 0.5);
        const b1 = pushVert(p0[0], p0[1], z0, 0, 0, -1, 1, 0, 0.5);
        pushTri(bc, b0, b1);
      }
    }
    // Back hub collar
    addSubCylinderZ(0, 0, rBase * 0.52, rBase * 0.25, z0 - h * 0.35, z0, 24, 0.7);
    endMesh(name, start);
  }

  // 7. 3D Helical Worm Screw (vertical Z axis)
  function buildWormScrew(name, rRoot, rTip, length, turns = 4.5) {
    const start = beginMesh();
    const z0 = -length * 0.5, z1 = length * 0.5;
    addSubCylinderZ(0, 0, rRoot, 0, z0, z1, 32, 0.45);

    const stepsPerTurn = turns > 15 ? 24 : 32;
    const steps = Math.round(turns * stepsPerTurn);
    const pitch = length / (turns + 1);
    const wRoot = pitch * 0.34;
    const wTip = pitch * 0.18;
    const rBase = rRoot * 0.90;
    for (let i = 0; i < steps; i++) {
      const t0 = i / steps;
      const t1 = (i + 1) / steps;
      const a0 = t0 * turns * Math.PI * 2;
      const a1 = t1 * turns * Math.PI * 2;
      const zc0 = z0 + pitch * 0.5 + t0 * (length - pitch);
      const zc1 = z0 + pitch * 0.5 + t1 * (length - pitch);

      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      const p0B = [c0 * rBase, s0 * rBase, zc0 - wRoot];
      const p0T1 = [c0 * rTip, s0 * rTip, zc0 - wTip];
      const p0T2 = [c0 * rTip, s0 * rTip, zc0 + wTip];
      const p0U = [c0 * rBase, s0 * rBase, zc0 + wRoot];

      const p1B = [c1 * rBase, s1 * rBase, zc1 - wRoot];
      const p1T1 = [c1 * rTip, s1 * rTip, zc1 - wTip];
      const p1T2 = [c1 * rTip, s1 * rTip, zc1 + wTip];
      const p1U = [c1 * rBase, s1 * rBase, zc1 + wRoot];

      addQuadFace(p0B, p1B, p1T1, p0T1, 0, 0, 1, 1, 0.65);
      addQuadFace(p0T1, p1T1, p1T2, p0T2, 0, 0, 1, 1, 0.85);
      addQuadFace(p0T2, p1T2, p1U, p0U, 0, 0, 1, 1, 0.65);
      addQuadFace(p0U, p1U, p1B, p0B, 0, 0, 1, 1, 0.45);
      if (i === 0) {
        addQuadFace(p0B, p0T1, p0T2, p0U, 0, 0, 1, 1, 0.78);
      }
      if (i === steps - 1) {
        addQuadFace(p1U, p1T2, p1T1, p1B, 0, 0, 1, 1, 0.78);
      }
    }
    endMesh(name, start);
  }

  // 8. Da Vinci Conical Helical Fusee Drum ("Cono Elicoidale")
  function buildConicalFusee(name, rBot, rTop, height, turns = 6.0) {
    const start = beginMesh();
    const z0 = 0, z1 = height;
    const radAt = (t) => rBot * (1 - t) + rTop * t - 0.55 * Math.sin(t * Math.PI);

    // Hyperboloid/concave conical body with 16 axial bands
    const bands = 18, segs = 36;
    for (let b = 0; b < bands; b++) {
      const t0 = b / bands, t1 = (b + 1) / bands;
      const za = z0 + t0 * height, zb = z0 + t1 * height;
      const ra = radAt(t0), rb = radAt(t1);
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2;
        const a1 = ((i + 1) / segs) * Math.PI * 2;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const p00 = pushVert(c0 * ra, s0 * ra, za, c0, s0, 0.25, i / segs, t0, 0.15);
        const p10 = pushVert(c1 * ra, s1 * ra, za, c1, s1, 0.25, (i + 1) / segs, t0, 0.15);
        const p11 = pushVert(c1 * rb, s1 * rb, zb, c1, s1, 0.25, (i + 1) / segs, t1, 0.15);
        const p01 = pushVert(c0 * rb, s0 * rb, zb, c0, s0, 0.25, i / segs, t1, 0.15);
        pushQuad(p00, p10, p11, p01);
      }
    }
    // Top and bottom caps
    addSubCylinderZ(0, 0, rBot + 0.4, 0, -1.2, 0.2, 36, 0.78);
    addSubCylinderZ(0, 0, rTop + 0.35, 0, height - 0.2, height + 1.0, 32, 0.78);

    // Clockwise (-theta) helical chain groove & keeper ledge wound continuously from z = 0 to z = height
    // so a chain arriving from +X along the front (-Y) tangent continues smoothly in the -X direction around the cone without a 180° hairpin reversal!
    const steps = Math.round(turns * 36);
    for (let i = 0; i < steps; i++) {
      const t0 = i / steps, t1 = (i + 1) / steps;
      const a0 = -t0 * turns * Math.PI * 2;
      const a1 = -t1 * turns * Math.PI * 2;
      const za = t0 * height;
      const zb = t1 * height;
      const r0 = radAt(t0), r1 = radAt(t1);
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      // Lower supporting helical flange lip (z - 0.44 .. z - 0.14) so chain links sit on top of the ledge in the recessed channel
      const dr = 0.36, zLipBot = -0.44, zLipMid = -0.26, zLipTop = -0.14;
      addQuadFace(
        [c1 * r1, s1 * r1, zb + zLipBot],
        [c0 * r0, s0 * r0, za + zLipBot],
        [c0 * (r0 + dr), s0 * (r0 + dr), za + zLipMid],
        [c1 * (r1 + dr), s1 * (r1 + dr), zb + zLipMid],
        0, 0, 1, 1, 0.78
      );
      addQuadFace(
        [c1 * (r1 + dr), s1 * (r1 + dr), zb + zLipMid],
        [c0 * (r0 + dr), s0 * (r0 + dr), za + zLipMid],
        [c0 * r0, s0 * r0, za + zLipTop],
        [c1 * r1, s1 * r1, zb + zLipTop],
        0, 0, 1, 1, 0.88
      );
    }

    endMesh(name, start);
  }

  // 9. Da Vinci Verge & Foliot Balance Beam (with regulation notches and terminal scroll finials)
  function buildFoliotBeam(name, halfSpan = 14.5) {
    const start = beginMesh();
    // Central verge collar
    addSubCylinderZ(0, 0, 2.2, 0.75, -1.4, 1.4, 24, 0.8);
    // Horizontal balance arm from -halfSpan to +halfSpan
    addSubBox(-halfSpan, -0.65, -0.55, halfSpan, 0.65, 0.55, 0.82);

    // Top V-notches along both arms (poise weights are emitted as separate articulated `foliotWeight` instances)
    for (const sign of [-1, 1]) {
      for (let k = 1; k <= 5; k++) {
        const nx = sign * (2.6 + k * 1.85);
        addSubBox(nx - 0.28, -0.68, 0.55, nx + 0.28, 0.68, 0.98, 0.9);
      }
      // Terminal scroll finial
      addSubCylinderZ(sign * halfSpan, 0, 1.05, 0, -0.65, 0.65, 16, 0.85);
    }
    endMesh(name, start);
  }

  // 10. Greek Fluted Doric / Corinthian Column (Attic base + 16-flute shaft + capital)
  function buildFlutedColumn(name, rBase, rShaft, height, flutes = 16) {
    const start = beginMesh();
    // Stepped square plinth + Attic torus base
    const bw = rBase * 1.12;
    addSubBox(-bw, -bw, 0, bw, bw, 1.4, 0.82);
    addSubCylinderZ(0, 0, rBase, 0, 1.4, 2.6, 32, 0.78);
    addSubCylinderZ(0, 0, rBase * 0.84, 0, 2.6, 3.6, 32, 0.75);

    // Fluted column shaft from z = 3.6 to height - 3.4
    const zShaft0 = 3.6;
    const zShaft1 = height - 3.4;
    const segs = flutes * 4;
    const bands = 8;
    for (let b = 0; b < bands; b++) {
      const t0 = b / bands, t1 = (b + 1) / bands;
      const za = zShaft0 + t0 * (zShaft1 - zShaft0);
      const zb = zShaft0 + t1 * (zShaft1 - zShaft0);
      // Classical Greek entasis (slight tapering toward capital)
      const ent0 = 1.0 - 0.08 * Math.pow(t0, 1.4);
      const ent1 = 1.0 - 0.08 * Math.pow(t1, 1.4);

      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2;
        const a1 = ((i + 1) / segs) * Math.PI * 2;
        const f0 = Math.max(0, Math.sin(a0 * flutes));
        const f1 = Math.max(0, Math.sin(a1 * flutes));
        const r0a = (rShaft - 0.22 * f0) * ent0;
        const r1a = (rShaft - 0.22 * f1) * ent0;
        const r0b = (rShaft - 0.22 * f0) * ent1;
        const r1b = (rShaft - 0.22 * f1) * ent1;

        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const arris = f0 < 0.15 || f1 < 0.15 ? 0.55 : 0.1;

        const p00 = pushVert(c0 * r0a, s0 * r0a, za, c0, s0, 0, i / segs, t0, arris);
        const p10 = pushVert(c1 * r1a, s1 * r1a, za, c1, s1, 0, (i + 1) / segs, t0, arris);
        const p11 = pushVert(c1 * r1b, s1 * r1b, zb, c1, s1, 0, (i + 1) / segs, t1, arris);
        const p01 = pushVert(c0 * r0b, s0 * r0b, zb, c0, s0, 0, i / segs, t1, arris);
        pushQuad(p00, p10, p11, p01);
      }
    }

    // Necking astragal ring, echinus flare, and square abacus capital
    addSubCylinderZ(0, 0, rShaft * 1.15, 0, zShaft1, zShaft1 + 0.9, 32, 0.82);
    addSubCylinderZ(0, 0, rBase * 0.95, 0, zShaft1 + 0.9, height - 1.2, 32, 0.82);
    addSubBox(-bw * 0.98, -bw * 0.98, height - 1.2, bw * 0.98, bw * 0.98, height, 0.88);

    endMesh(name, start);
  }

  // 11. Cutaway Windowed Barrel (Hollow cylinder with an open inspection sector)
  function buildWindowedBarrel(name, rIn, rOut, z0, z1, winStartDeg, winEndDeg, segs = 48) {
    const start = beginMesh();
    const w0 = ((winStartDeg % 360) + 360) % 360;
    const w1 = ((winEndDeg % 360) + 360) % 360;
    const inWin = (deg) => (w0 < w1 ? deg >= w0 && deg < w1 : deg >= w0 || deg < w1);

    const zWin0 = z0 + (z1 - z0) * 0.16;
    const zWin1 = z1 - (z1 - z0) * 0.16;

    for (let i = 0; i < segs; i++) {
      const deg0 = (i / segs) * 360;
      const deg1 = ((i + 1) / segs) * 360;
      const a0 = (deg0 * Math.PI) / 180;
      const a1 = (deg1 * Math.PI) / 180;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const open = inWin((deg0 + deg1) * 0.5);

      const bands = open
        ? [
            [z0, zWin0],
            [zWin1, z1],
          ]
        : [[z0, z1]];

      for (const [za, zb] of bands) {
        const o00 = pushVert(c0 * rOut, s0 * rOut, za, c0, s0, 0, i / segs, 0, 0.25);
        const o10 = pushVert(c1 * rOut, s1 * rOut, za, c1, s1, 0, (i + 1) / segs, 0, 0.25);
        const o11 = pushVert(c1 * rOut, s1 * rOut, zb, c1, s1, 0, (i + 1) / segs, 1, 0.25);
        const o01 = pushVert(c0 * rOut, s0 * rOut, zb, c0, s0, 0, i / segs, 1, 0.25);
        pushQuad(o00, o10, o11, o01);

        const n00 = pushVert(c1 * rIn, s1 * rIn, za, -c1, -s1, 0, 0, 0, 0.2);
        const n10 = pushVert(c0 * rIn, s0 * rIn, za, -c0, -s0, 0, 1, 0, 0.2);
        const n11 = pushVert(c0 * rIn, s0 * rIn, zb, -c0, -s0, 0, 1, 1, 0.2);
        const n01 = pushVert(c1 * rIn, s1 * rIn, zb, -c1, -s1, 0, 0, 1, 0.2);
        pushQuad(n00, n10, n11, n01);
      }

      if (open) {
        addQuadFace(
          [c0 * rIn, s0 * rIn, zWin0],
          [c0 * rOut, s0 * rOut, zWin0],
          [c1 * rOut, s1 * rOut, zWin0],
          [c1 * rIn, s1 * rIn, zWin0],
          0, 0, 1, 1, 0.85
        );
        addQuadFace(
          [c1 * rIn, s1 * rIn, zWin1],
          [c1 * rOut, s1 * rOut, zWin1],
          [c0 * rOut, s0 * rOut, zWin1],
          [c0 * rIn, s0 * rIn, zWin1],
          0, 0, 1, 1, 0.85
        );
      }

      addQuadFace(
        [c0 * rIn, s0 * rIn, z1],
        [c0 * rOut, s0 * rOut, z1],
        [c1 * rOut, s1 * rOut, z1],
        [c1 * rIn, s1 * rIn, z1],
        0, 0, 1, 1, 0.65
      );
      addQuadFace(
        [c1 * rIn, s1 * rIn, z0],
        [c1 * rOut, s1 * rOut, z0],
        [c0 * rOut, s0 * rOut, z0],
        [c0 * rIn, s0 * rIn, z0],
        0, 0, 1, 1, 0.45
      );
    }
    endMesh(name, start);
  }

  // 12. Cutaway C-Shaped Sector Ring with 3D Graduated Degree Ticks & Zodiac Parapegma Moulding
  function buildEclipticRing(name, rIn, rOut, z0, z1, degStart, degEnd, segs = 80) {
    const start = beginMesh();
    const span = degEnd - degStart;
    const b = 0.48;
    const rMid = (rIn + rOut) * 0.5;

    for (let i = 0; i < segs; i++) {
      const a0 = ((degStart + (i / segs) * span) * Math.PI) / 180;
      const a1 = ((degStart + ((i + 1) / segs) * span) * Math.PI) / 180;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      addQuadFace(
        [c0 * rOut, s0 * rOut, z0],
        [c1 * rOut, s1 * rOut, z0],
        [c1 * rOut, s1 * rOut, z1],
        [c0 * rOut, s0 * rOut, z1],
        i / segs, 0, (i + 1) / segs, 1, 0.38
      );
      addQuadFace(
        [c1 * rIn, s1 * rIn, z0],
        [c0 * rIn, s0 * rIn, z0],
        [c0 * rIn, s0 * rIn, z1],
        [c1 * rIn, s1 * rIn, z1],
        i / segs, 0, (i + 1) / segs, 1, 0.38
      );

      const houseEdge = i % 8 === 0 ? 0.92 : 0.45;
      addQuadFace(
        [c0 * rIn, s0 * rIn, z1],
        [c0 * (rOut - b), s0 * (rOut - b), z1],
        [c1 * (rOut - b), s1 * (rOut - b), z1],
        [c1 * rIn, s1 * rIn, z1],
        0, 0, 1, 1, houseEdge
      );
      addQuadFace(
        [c0 * (rOut - b), s0 * (rOut - b), z1],
        [c0 * rOut, s0 * rOut, z1 - b],
        [c1 * rOut, s1 * rOut, z1 - b],
        [c1 * (rOut - b), s1 * (rOut - b), z1],
        0, 0, 1, 1, 0.85
      );
      addQuadFace(
        [c1 * rIn, s1 * rIn, z0],
        [c1 * rOut, s1 * rOut, z0],
        [c0 * rOut, s0 * rOut, z0],
        [c0 * rIn, s0 * rIn, z0],
        0, 0, 1, 1, 0.4
      );
    }

    for (const [deg, flip] of [
      [degStart, -1],
      [degEnd, 1],
    ]) {
      const a = (deg * Math.PI) / 180;
      const c = Math.cos(a), s = Math.sin(a);
      const p0 = [c * rIn, s * rIn, z0];
      const p1 = [c * rOut, s * rOut, z0];
      const p2 = [c * rOut, s * rOut, z1];
      const p3 = [c * rIn, s * rIn, z1];
      if (flip > 0) addQuadFace(p0, p1, p2, p3, 0, 0, 1, 1, 0.9);
      else addQuadFace(p1, p0, p3, p2, 0, 0, 1, 1, 0.9);
    }

    // Raised 3D graduation ticks and Zodiac house divider ribs along top face
    const tickCount = 46;
    for (let i = 0; i <= tickCount; i++) {
      const deg = degStart + 2.5 + ((span - 5) * i) / tickCount;
      const a = (deg * Math.PI) / 180;
      const isHouse = i % 6 === 0;
      const major = i % 3 === 0;
      const rT0 = isHouse ? rIn + 0.6 : rOut - (major ? 4.8 : 2.6);
      const rT1 = rOut - 0.7;
      const tw = isHouse ? 0.42 : major ? 0.30 : 0.18;
      const th = isHouse ? 0.42 : major ? 0.28 : 0.16;
      const ux = Math.cos(a), uy = Math.sin(a);
      const vx = -uy * tw, vy = ux * tw;
      const pA = [ux * rT0 - vx, uy * rT0 - vy, z1 + th];
      const pB = [ux * rT1 - vx, uy * rT1 - vy, z1 + th];
      const pC = [ux * rT1 + vx, uy * rT1 + vy, z1 + th];
      const pD = [ux * rT0 + vx, uy * rT0 + vy, z1 + th];
      addQuadFace(pA, pB, pC, pD, 0, 0, 1, 1, 0.95);
    }

    // Concentric Parapegma index bead ridge at rMid
    for (let i = 0; i < segs; i += 4) {
      const a = ((degStart + ((i + 0.5) / segs) * span) * Math.PI) / 180;
      addSubCylinderZ(Math.cos(a) * rMid, Math.sin(a) * rMid, 0.48, 0, z1, z1 + 0.35, 8, 0.9);
    }

    endMesh(name, start);
  }

  // 13. Archimedes Celestial Meridian / Solstitial Colure Omega Arch (anchored at r = 48 on Ecliptic Ring, flaring to r = 108.4 to clear all 6 planets & Saturn rings, crowning at Zenith z = 85.5)
  function buildArmillaryArch(name, radius, radialW, axialT, degStart = 0, degEnd = 180, segs = 64) {
    const start = beginMesh();
    const ht = axialT * 0.5;
    const span = degEnd - degStart;

    function archProfile(deg) {
      const a = (deg * Math.PI) / 180;
      const s = Math.sin(a), c = Math.cos(a);
      const sRise = 1 - Math.pow(1 - Math.min(1, s / 0.20), 3);
      const rBase = 48.0 + (108.5 - 48.0) * sRise;
      const cCap = c / Math.hypot(c, 0.16 * s + 1e-6);
      // Terminate at rZenith = 2.85 at deg = 90° so the arch rib joins the outer wall of the Open Zenith Polar Ring (r = 3.4)
      const x = 2.85 + (rBase - 2.85) * cCap;
      const z = 85.5 * s;
      return [x, z];
    }

    function archPointAndNormal(deg) {
      const [x, z] = archProfile(deg);
      const eps = 0.25;
      const [x0, z0] = archProfile(Math.max(0, deg - eps));
      const [x1, z1] = archProfile(Math.min(180, deg + eps));
      const tx = x1 - x0, tz = z1 - z0;
      const tLen = Math.hypot(tx, tz) || 1;
      // Outward normal in XZ plane for deg increasing from 0 (+X) through 90 (+Z) to 180 (-X)
      const nx = tz / tLen;
      const nz = -tx / tLen;
      return { x, z, nx, nz };
    }

    for (let i = 0; i < segs; i++) {
      const d0 = degStart + (i / segs) * span;
      const d1 = degStart + ((i + 1) / segs) * span;
      const p0 = archPointAndNormal(d0);
      const p1 = archPointAndNormal(d1);
      const hw = radialW * 0.5;

      const x0In = p0.x - p0.nx * hw, z0In = p0.z - p0.nz * hw;
      const x0Out = p0.x + p0.nx * hw, z0Out = p0.z + p0.nz * hw;
      const x1In = p1.x - p1.nx * hw, z1In = p1.z - p1.nz * hw;
      const x1Out = p1.x + p1.nx * hw, z1Out = p1.z + p1.nz * hw;
      const edge = i % 4 === 0 ? 0.88 : 0.42;

      // Outer crest face
      addQuadFace(
        [x0Out, -ht, z0Out],
        [x1Out, -ht, z1Out],
        [x1Out, ht, z1Out],
        [x0Out, ht, z0Out],
        0, 0, 1, 1, edge
      );
      // Inner soffit face
      addQuadFace(
        [x1In, -ht, z1In],
        [x0In, -ht, z0In],
        [x0In, ht, z0In],
        [x1In, ht, z1In],
        0, 0, 1, 1, 0.55
      );
      // Front (-Y) and Back (+Y) arch faces
      addQuadFace(
        [x0In, -ht, z0In],
        [x1In, -ht, z1In],
        [x1Out, -ht, z1Out],
        [x0Out, -ht, z0Out],
        0, 0, 1, 1, edge
      );
      addQuadFace(
        [x1In, ht, z1In],
        [x0In, ht, z0In],
        [x0Out, ht, z0Out],
        [x1Out, ht, z1Out],
        0, 0, 1, 1, edge
      );
    }

    // Integral turned mounting foot flange(s) at r = 48.0, z = 0 where the arch lands on the Ecliptic Parapegma Ring
    if (degStart <= 1) {
      addSubCylinderZ(48.0, 0, radialW * 0.95, 0, -0.6, 1.4, 20, 0.88);
    }
    if (degEnd >= 179) {
      addSubCylinderZ(-48.0, 0, radialW * 0.95, 0, -0.6, 1.4, 20, 0.88);
    }

    endMesh(name, start);
  }

  // 14. Sculpted Pierced-Truss Horological Planetary Carrier Arm (along +X from -tailLen to +armLen, with central & distal bosses and open bores)
  function buildCarrierBar(name, tailLen, armLen, w, h, rCenterBoss = 0, rEndBoss = 0, rCenterBore = 0, rEndBore = 0) {
    const start = beginMesh();
    const hw = w * 0.5, hh = h * 0.5;
    const rBore = rCenterBore > 0.1 ? rCenterBore : rCenterBoss * 0.62;
    const rSplit = Math.max(rBore + 0.15, rCenterBoss * 0.72);
    const xArmEnd = rEndBore > 0.1 ? armLen - Math.max(rEndBore + 0.15, rEndBoss * 0.72) : armLen;

    // Split bar into tail segment [-tailLen .. -rSplit] and main arm segment [+rSplit .. xArmEnd] so bores are 100% open!
    if (tailLen > rSplit + 0.2) {
      addSubBox(-tailLen, -hw * 0.84, -hh * 0.88, -rSplit, hw * 0.84, hh * 0.88, 0.82);
    }
    const span = xArmEnd - rSplit;
    if (span > 18.0 && rEndBore < 0.05) {
      // Pierced Horological Lattice / Truss Arm: twin outer side rails + cross-tie webs with open lightening windows!
      const railW = Math.max(0.58, hw * 0.34);
      addSubBox(rSplit, -hw, -hh, xArmEnd, -hw + railW, hh, 0.86);
      addSubBox(rSplit, hw - railW, -hh, xArmEnd, hw, hh, 0.86);
      const bayCount = Math.max(3, Math.min(8, Math.floor(span / 10.5)));
      for (let b = 0; b <= bayCount; b++) {
        const bx = rSplit + (b / bayCount) * (span - 1.2);
        addSubBox(bx, -hw + railW, -hh * 0.92, bx + 1.15, hw - railW, hh * 0.92, 0.88);
      }
      // Distal Vernier Zodiac Index Pointer beak extending +3.4 units beyond outer boss
      addSubBox(armLen + rEndBoss * 0.75, -hw * 0.24, -hh * 0.45, armLen + rEndBoss * 0.75 + 3.2, hw * 0.24, hh * 0.45, 0.94);
    } else if (xArmEnd > rSplit + 0.2) {
      addSubBox(rSplit, -hw, -hh, xArmEnd, hw, hh, 0.78);
    }

    if (rCenterBoss > 0.1) {
      addSubCylinderZ(0, 0, rCenterBoss, rBore, -hh * 1.18, hh * 1.18, 36, 0.85);
    }
    if (rEndBoss > 0.1) {
      addSubCylinderZ(armLen, 0, rEndBoss, rEndBore, -hh * 1.18, hh * 1.18, 32, 0.85);
      if (rEndBore < 0.05) {
        addSubCylinderZ(-tailLen, 0, rEndBoss * 0.85, 0, -hh * 1.15, hh * 1.15, 24, 0.84);
      }
    }
    endMesh(name, start);
  }

  // 15. Antikythera Pin-and-Slot Slotted Coupler Arm (open radial slot from slotR0 to slotR1)
  function buildSlottedAnomalyArm(name, rHub, slotR0, slotR1, slotHalfW, rOuter, h) {
    const start = beginMesh();
    const hh = h * 0.5;
    const railW = 0.85;
    // Central hub ring (bore = 1.55 > Earth shaft 1.42)
    addSubCylinderZ(0, 0, rHub, 1.55, -hh, hh, 24, 0.82);
    // Two parallel side rails flanking the radial slot from x = slotR0 to slotR1
    addSubBox(slotR0, -slotHalfW - railW, -hh, slotR1, -slotHalfW, hh, 0.88);
    addSubBox(slotR0, slotHalfW, -hh, slotR1, slotHalfW + railW, hh, 0.88);
    // Distal bridge closing the slot and reaching rOuter
    addSubBox(slotR1, -slotHalfW - railW, -hh, rOuter, slotHalfW + railW, hh, 0.88);
    // Counterweight tail on -X
    addSubBox(-rHub - 3.2, -railW, -hh * 0.85, -rHub * 0.95, railW, hh * 0.85, 0.78);
    endMesh(name, start);
  }

  // 16. Equation-of-Time Reniform (Kidney) Cam: r(theta) = rMean + a1*sin(theta - 0.049) + a2*sin(2*theta)
  function buildKidneyCam(name, rMean = 6.2, h = 1.8) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const segs = 72;
    const rAt = (ang) => rMean - 0.95 * Math.sin(ang - 0.049) + 1.25 * Math.sin(2 * ang);

    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const r0 = rAt(a0), r1 = rAt(a1);
      const x0 = Math.cos(a0) * r0, y0 = Math.sin(a0) * r0;
      const x1 = Math.cos(a1) * r1, y1 = Math.sin(a1) * r1;

      addQuadFace([x0, y0, z0], [x1, y1, z0], [x1, y1, z1], [x0, y0, z1], i / segs, 0, (i + 1) / segs, 1, 0.82);

      const tc = pushVert(0, 0, z1, 0, 0, 1, 0.5, 0.5, 0);
      const t0 = pushVert(x0, y0, z1, 0, 0, 1, 0, 0, 0.82);
      const t1 = pushVert(x1, y1, z1, 0, 0, 1, 1, 0, 0.82);
      pushTri(tc, t0, t1);

      const bc = pushVert(0, 0, z0, 0, 0, -1, 0.5, 0.5, 0);
      const b0 = pushVert(x1, y1, z0, 0, 0, -1, 0, 0, 0.6);
      const b1 = pushVert(x0, y0, z0, 0, 0, -1, 1, 0, 0.6);
      pushTri(bc, b0, b1);
    }
    addSubCylinderZ(0, 0, 1.8, 0.75, z0 - 0.6, z1 + 0.6, 24, 0.85);
    endMesh(name, start);
  }

  // 17. Toothed Linear Ephemeris Rack & Graduated Rule (length L along X, centered at 0,0)
  function buildToothedRack(name, length = 184, width = 8.5, height = 1.6, teeth = 56) {
    const start = beginMesh();
    const hw = length * 0.5, hd = width * 0.5;
    // Main base rule bar
    addSubBox(-hw, -hd, 0, hw, hd, height, 0.78);
    // Raised back rack rail with 3D involute rack teeth along y = +0.5..+hd-0.6
    const xStart = -hw + 8.0, xEnd = hw - 8.0;
    const span = xEnd - xStart;
    const pitch = span / teeth;
    const toothH = 0.75;
    for (let t = 0; t < teeth; t++) {
      const x0 = xStart + t * pitch;
      const xB0 = x0 + pitch * 0.15;
      const xT0 = x0 + pitch * 0.30;
      const xT1 = x0 + pitch * 0.65;
      const xB1 = x0 + pitch * 0.80;
      const y0 = 0.6, y1 = hd - 0.6;
      addQuadFace([xB0, y0, height], [xB1, y0, height], [xT1, y0, height + toothH], [xT0, y0, height + toothH], 0, 0, 1, 1, 0.88);
      addQuadFace([xB1, y1, height], [xB0, y1, height], [xT0, y1, height + toothH], [xT1, y1, height + toothH], 0, 0, 1, 1, 0.88);
      addQuadFace([xT0, y0, height + toothH], [xT1, y0, height + toothH], [xT1, y1, height + toothH], [xT0, y1, height + toothH], 0, 0, 1, 1, 0.92);
      addQuadFace([xB0, y1, height], [xB0, y0, height], [xT0, y0, height + toothH], [xT0, y1, height + toothH], 0, 0, 1, 1, 0.86);
      addQuadFace([xB1, y0, height], [xB1, y1, height], [xT1, y1, height + toothH], [xT1, y0, height + toothH], 0, 0, 1, 1, 0.86);
    }
    // Front graduation ticks along y = -hd .. -0.4
    const gradCount = 36;
    for (let g = 0; g <= gradCount; g++) {
      const gx = xStart + (g / gradCount) * span;
      const major = g % 3 === 0;
      const gy1 = major ? 0.1 : -1.2;
      addSubBox(gx - (major ? 0.25 : 0.14), -hd + 0.3, height, gx + (major ? 0.25 : 0.14), gy1, height + (major ? 0.32 : 0.18), 0.92);
    }
    endMesh(name, start);
  }

  // 18. Cast-Brass Horological Pillow Block Bearing Journal (centered at 0,0, base z = 0, shaft bore at zShaft)
  function buildPillowBlock(name, w = 5.8, d = 3.4, zShaft = 4.5, rHousing = 2.5) {
    const start = beginMesh();
    const hw = w * 0.5, hd = d * 0.5;
    // Base flange with mounting ears
    addSubBox(-hw, -hd, 0, hw, hd, 1.2, 0.82);
    addSubCylinderZ(-hw + 0.9, 0, 0.55, 0, 1.2, 1.65, 10, 0.88);
    addSubCylinderZ(hw - 0.9, 0, 0.55, 0, 1.2, 1.65, 10, 0.88);
    // Lower pedestal web rising to shaft journal
    addSubBox(-rHousing * 0.88, -hd * 0.78, 1.2, rHousing * 0.88, hd * 0.78, zShaft - rHousing * 0.85, 0.80);
    // Split bearing journal block enclosing z = zShaft (with side cap bolts)
    addSubBox(-rHousing * 1.05, -hd * 0.86, zShaft - rHousing * 0.95, rHousing * 1.05, hd * 0.86, zShaft + rHousing * 1.05, 0.86);
    // Top oil-cup finial
    addSubCylinderZ(0, 0, 0.85, 0.35, zShaft + rHousing * 1.05, zShaft + rHousing * 1.05 + 1.0, 14, 0.88);
    endMesh(name, start);
  }

  // 19. Sculpted Horological Clock Hands (Breguet Sun, Equation-of-Time Crescent, Antikythera Dragon Node)
  function buildDialHand(name, tailLen, tipLen, wStem, h, style = "sun") {
    const start = beginMesh();
    const hh = h * 0.5;
    const rBossOut = wStem * 1.95;
    const rBossIn = wStem * 0.58;
    // Central hub collet boss
    addSubCylinderZ(0, 0, rBossOut, rBossIn, -hh * 1.18, hh * 1.18, 28, 0.86);

    function addTaperedBladeSegment(x0, x1, w0, w1, zScale = 1.0) {
      const zB = -hh * zScale;
      const zT = hh * zScale;
      const zMid = hh * zScale * 1.18;
      const p0L = [x0, -w0 * 0.5, zB], p0R = [x0, w0 * 0.5, zB];
      const p1L = [x1, -w1 * 0.5, zB], p1R = [x1, w1 * 0.5, zB];
      const t0L = [x0, -w0 * 0.5, zT], t0R = [x0, w0 * 0.5, zT];
      const t1L = [x1, -w1 * 0.5, zT], t1R = [x1, w1 * 0.5, zT];
      const t0M = [x0, 0, zMid], t1M = [x1, 0, zMid];

      // Beveled Breguet/lance roof facets (+Z)
      addQuadFace(t0L, t1L, t1M, t0M, 0, 0, 1, 1, 0.90);
      addQuadFace(t0M, t1M, t1R, t0R, 0, 0, 1, 1, 0.90);
      // Flat bottom soffit (-Z) & side walls
      addQuadFace(p1L, p0L, p0R, p1R, 0, 0, 1, 1, 0.62);
      addQuadFace(p0L, p1L, t1L, t0L, 0, 0, 1, 1, 0.82);
      addQuadFace(p1R, p0R, t0R, t1R, 0, 0, 1, 1, 0.82);
      // End cap at x1 if sharp tip
      if (w1 > 0.01) {
        addQuadFace(p1L, p1R, t1R, t1L, 0, 0, 1, 1, 0.88);
      }
    }

    // Tapered counterweight tail segment [-tailLen .. -rBossOut * 0.85]
    addTaperedBladeSegment(-tailLen, -rBossOut * 0.85, wStem * 0.62, wStem * 0.92, 0.92);

    if (style === "sun") {
      // Open-center Breguet pomme ring at 62% of tipLen (blade interrupted across ring interior!)
      const xPomme = tipLen * 0.62;
      const rPommeOut = wStem * 1.85;
      const rPommeIn = wStem * 1.08;
      addTaperedBladeSegment(rBossOut * 0.85, xPomme - rPommeOut * 0.92, wStem * 0.95, wStem * 0.68, 0.96);
      addSubCylinderZ(xPomme, 0, rPommeOut, rPommeIn, -hh * 1.05, hh * 1.05, 28, 0.92);
      addTaperedBladeSegment(xPomme + rPommeOut * 0.92, tipLen, wStem * 0.56, wStem * 0.12, 0.86);
      // Counterweight poise ring on tail
      addSubCylinderZ(-tailLen, 0, wStem * 1.32, wStem * 0.52, -hh * 1.08, hh * 1.08, 22, 0.88);
    } else if (style === "dragon") {
      // Continuous tapered shaft + Hellenistic Dragon Head arrow at tip + open ring on tail
      addTaperedBladeSegment(rBossOut * 0.85, tipLen * 0.78, wStem * 0.92, wStem * 0.52, 0.95);
      addTaperedBladeSegment(tipLen * 0.78, tipLen, wStem * 1.45, wStem * 0.10, 0.92);
      addSubCylinderZ(-tailLen, 0, wStem * 1.45, wStem * 0.68, -hh, hh, 22, 0.88);
    } else {
      // Smoothly tapered Equation-of-Time lance blade with delicate mid-collar & counterweight poise ring
      addTaperedBladeSegment(rBossOut * 0.85, tipLen * 0.68, wStem * 0.92, wStem * 0.54, 0.95);
      addTaperedBladeSegment(tipLen * 0.68, tipLen, wStem * 0.54, wStem * 0.12, 0.86);
      addSubCylinderZ(-tailLen, 0, wStem * 1.18, wStem * 0.48, -hh * 1.04, hh * 1.04, 20, 0.86);
    }
    endMesh(name, start);
  }

  // 20. High-Tessellation UV Sphere (radius 1.0, scaled via Instance model matrix)
  function buildUnitSphere(name, latSteps = 32, lonSteps = 48) {
    const start = beginMesh();
    const grid = [];
    for (let i = 0; i <= latSteps; i++) {
      const v = i / latSteps;
      const theta = v * Math.PI;
      const st = Math.sin(theta), ct = Math.cos(theta);
      const row = [];
      for (let j = 0; j <= lonSteps; j++) {
        const u = j / lonSteps;
        const phi = u * Math.PI * 2;
        const nx = st * Math.cos(phi);
        const ny = st * Math.sin(phi);
        const nz = ct;
        row.push(pushVert(nx, ny, nz, nx, ny, nz, u, v, 0));
      }
      grid.push(row);
    }
    for (let i = 0; i < latSteps; i++) {
      for (let j = 0; j < lonSteps; j++) {
        pushQuad(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]);
      }
    }
    endMesh(name, start);
  }

  // 21. Torus Ring (radius R, tube radius r, in XY plane)
  function buildTorus(name, R, r, majSegs = 48, minSegs = 16) {
    const start = beginMesh();
    const grid = [];
    for (let i = 0; i <= majSegs; i++) {
      const u = i / majSegs;
      const a = u * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const row = [];
      for (let j = 0; j <= minSegs; j++) {
        const v = j / minSegs;
        const b = v * Math.PI * 2;
        const cb = Math.cos(b), sb = Math.sin(b);
        const rad = R + r * cb;
        const x = ca * rad, y = sa * rad, z = r * sb;
        const nx = ca * cb, ny = sa * cb, nz = sb;
        row.push(pushVert(x, y, z, nx, ny, nz, u, v, 0.2));
      }
      grid.push(row);
    }
    for (let i = 0; i < majSegs; i++) {
      for (let j = 0; j < minSegs; j++) {
        pushQuad(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]);
      }
    }
    endMesh(name, start);
  }

  // 22. Scotch-Yoke Slotted Frame + Plunger Stem + Vernier Pointer
  function buildScotchYokePlunger(name) {
    const start = beginMesh();
    // Slotted rectangular frame at bottom (-4.6..+4.6 X, -1.2..+1.2 Y, -6.5..-1.5 Z; slot center z = -4.0)
    // Inner slot width [-3.85, +3.85] accommodates circular crank radius rCrank = 2.85 + ruby pin r = 0.82 = 3.67!
    addSubBox(-4.6, -1.2, -2.5, 4.6, 1.2, -1.5, 0.88);
    addSubBox(-4.6, -1.2, -6.5, 4.6, 1.2, -5.5, 0.88);
    addSubBox(-4.6, -1.2, -5.5, -3.85, 1.2, -2.5, 0.88);
    addSubBox(3.85, -1.2, -5.5, 4.6, 1.2, -2.5, 0.88);

    // Vertical polished stem from z = -1.5 to z = +16.5
    addSubCylinderZ(0, 0, 1.62, 0, -1.5, 16.5, 24, 0.68);
    addSubCylinderZ(0, 0, 2.15, 0, 16.0, 17.2, 24, 0.85);

    // Vernier index pointer arm reaching -X to flange-mounted scale bar (x = -5.4..-1.4, y = -1.3..-0.3, z = 12.4..13.4)
    addSubBox(-5.4, -1.3, 12.4, -1.4, -0.3, 13.4, 0.92);
    endMesh(name, start);
  }

  // 23. Maltese Cross (Geneva) 6-Slot Intermittent Wheel
  function buildGenevaWheel(name, rOuter = 6.24, slots = 6, h = 1.6) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const rHub = 2.45;
    const rBore = 0.85;
    addSubCylinderZ(0, 0, rHub, rBore, z0 * 1.25, z1 * 1.25, 28, 0.84);

    const slotHalfAng = 0.095;
    const lobeSegs = 10;
    for (let k = 0; k < slots; k++) {
      const aStart = (k * Math.PI * 2) / slots + slotHalfAng;
      const aEnd = ((k + 1) * Math.PI * 2) / slots - slotHalfAng;
      const span = aEnd - aStart;
      for (let i = 0; i < lobeSegs; i++) {
        const u0 = i / lobeSegs;
        const u1 = (i + 1) / lobeSegs;
        const a0 = aStart + u0 * span;
        const a1 = aStart + u1 * span;
        // Concave crescent locking arc indented into the outer lobe perimeter
        const rOut0 = rOuter - 0.95 * Math.sin(u0 * Math.PI);
        const rOut1 = rOuter - 0.95 * Math.sin(u1 * Math.PI);
        const rIn = rHub * 0.92;

        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);

        const p0In = [c0 * rIn, s0 * rIn], p0Out = [c0 * rOut0, s0 * rOut0];
        const p1In = [c1 * rIn, s1 * rIn], p1Out = [c1 * rOut1, s1 * rOut1];

        addQuadFace([p0Out[0], p0Out[1], z0], [p1Out[0], p1Out[1], z0], [p1Out[0], p1Out[1], z1], [p0Out[0], p0Out[1], z1], u0, 0, u1, 1, 0.86);
        addQuadFace([p0In[0], p0In[1], z1], [p0Out[0], p0Out[1], z1], [p1Out[0], p1Out[1], z1], [p1In[0], p1In[1], z1], 0, 0, 1, 1, 0.82);
        addQuadFace([p1In[0], p1In[1], z0], [p1Out[0], p1Out[1], z0], [p0Out[0], p0Out[1], z0], [p0In[0], p0In[1], z0], 0, 0, 1, 1, 0.62);

        if (i === 0) {
          addQuadFace([p0In[0], p0In[1], z0], [p0Out[0], p0Out[1], z0], [p0Out[0], p0Out[1], z1], [p0In[0], p0In[1], z1], 0, 0, 1, 1, 0.90);
        }
        if (i === lobeSegs - 1) {
          addQuadFace([p1Out[0], p1Out[1], z0], [p1In[0], p1In[1], z0], [p1In[0], p1In[1], z1], [p1Out[0], p1Out[1], z1], 0, 0, 1, 1, 0.90);
        }
      }
    }
    endMesh(name, start);
  }

  // 24. Continuous 1-Pin Geneva Driver with 300° Crescent Locking Cam Disc
  function buildGenevaDriver(name, rPitch = 4.4, rLock = 2.50, rPinOrbit = 3.60, rPin = 0.52, h = 1.6) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    // Lower driving disc & central bore
    addSubCylinderZ(0, 0, rPitch, 0.85, z0, 0.0, 36, 0.80);
    // Crank web extending along +X to carry the eccentric drive pin at rPinOrbit
    addSubBox(0.9, -1.05, z0, rPinOrbit + 0.85, 1.05, z1 * 0.35, 0.84);
    addSubCylinderZ(rPinOrbit, 0, rPin * 1.65, 0, z0, z1 * 0.45, 18, 0.84);
    addSubCylinderZ(rPinOrbit, 0, rPin, 0, z1 * 0.35, z1 * 1.35, 18, 0.92);

    // Raised 240° crescent locking cam disc spanning 60°..300° (open on -60°..+60° for tangent pin slot transit)
    const camSegs = 28;
    const aStart = Math.PI / 3;
    const aSpan = (Math.PI * 4) / 3;
    const rIn = 0.85;
    for (let i = 0; i < camSegs; i++) {
      const a0 = aStart + (i / camSegs) * aSpan;
      const a1 = aStart + ((i + 1) / camSegs) * aSpan;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      addQuadFace([c0 * rLock, s0 * rLock, 0], [c1 * rLock, s1 * rLock, 0], [c1 * rLock, s1 * rLock, z1], [c0 * rLock, s0 * rLock, z1], 0, 0, 1, 1, 0.85);
      addQuadFace([c0 * rIn, s0 * rIn, z1], [c0 * rLock, s0 * rLock, z1], [c1 * rLock, s1 * rLock, z1], [c1 * rIn, s1 * rIn, z1], 0, 0, 1, 1, 0.82);
      if (i === 0) {
        addQuadFace([c0 * rIn, s0 * rIn, 0], [c0 * rLock, s0 * rLock, 0], [c0 * rLock, s0 * rLock, z1], [c0 * rIn, s0 * rIn, z1], 0, 0, 1, 1, 0.88);
      }
      if (i === camSegs - 1) {
        addQuadFace([c1 * rLock, s1 * rLock, 0], [c1 * rIn, s1 * rIn, 0], [c1 * rIn, s1 * rIn, z1], [c1 * rLock, s1 * rLock, z1], 0, 0, 1, 1, 0.88);
      }
    }
    endMesh(name, start);
  }

  // 25. Da Vinci Asymmetric Sawtooth Ratchet Wheel ("Arresto a Dente", Codex Madrid I f. 16r)
  function buildRatchetWheel(name, rPitch = 6.5, teeth = 24, h = 1.4, rBore = 1.8) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const rRoot = rPitch - 0.55;
    const rTip = rPitch + 0.45;
    addSubCylinderZ(0, 0, rRoot, rBore, z0, z1, 36, 0.78);

    const pitchA = (Math.PI * 2) / teeth;
    for (let t = 0; t < teeth; t++) {
      const a0 = t * pitchA;
      const aPeak = a0 + pitchA * 0.14;
      const aEnd = a0 + pitchA * 0.94;
      const p0 = [Math.cos(a0) * rRoot, Math.sin(a0) * rRoot];
      const p1 = [Math.cos(aPeak) * rTip, Math.sin(aPeak) * rTip];
      const p2 = [Math.cos(aEnd) * rRoot, Math.sin(aEnd) * rRoot];

      // Steep locking flank (p0 -> p1) and shallow ramping back (p1 -> p2)
      addQuadFace([p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z1], [p0[0], p0[1], z1], 0, 0, 1, 1, 0.92);
      addQuadFace([p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1], 0, 0, 1, 1, 0.78);

      const t0 = pushVert(p0[0], p0[1], z1, 0, 0, 1, 0, 0, 0.82);
      const t1 = pushVert(p1[0], p1[1], z1, 0, 0, 1, 1, 0, 0.92);
      const t2 = pushVert(p2[0], p2[1], z1, 0, 0, 1, 0, 1, 0.82);
      pushTri(t0, t1, t2);

      const b0 = pushVert(p2[0], p2[1], z0, 0, 0, -1, 0, 0, 0.58);
      const b1 = pushVert(p1[0], p1[1], z0, 0, 0, -1, 1, 0, 0.58);
      const b2 = pushVert(p0[0], p0[1], z0, 0, 0, -1, 0, 1, 0.58);
      pushTri(b0, b1, b2);
    }
    endMesh(name, start);
  }

  // 26. Pivoted Blued-Steel Ratchet Click Pawl with Continuous Swept Preload Leaf Spring
  function buildClickPawl(name, length = 4.6, w = 1.1, h = 1.2) {
    const start = beginMesh();
    const hh = h * 0.5;
    // Pivot boss at (0, 0)
    addSubCylinderZ(0, 0, 1.35, 0.55, -hh * 1.15, hh * 1.15, 20, 0.85);
    // Rigid pawl arm reaching +X to locking tooth beak at x = length
    addSubBox(0.85, -w * 0.45, -hh * 0.85, length, w * 0.45, hh * 0.85, 0.86);
    addSubBox(length - 0.85, -w * 0.45, -hh * 0.85, length + 0.35, w * 0.95, hh * 0.85, 0.92);
    // Continuous 16-segment swept curved preload leaf-spring tail along -X (zero disconnected boxes!)
    const segs = 16;
    const zS0 = -hh * 0.55, zS1 = hh * 0.55;
    const ptSpring = (t) => {
      const x = -0.95 - t * 3.15;
      const y = -0.85 * t * t;
      const halfThick = 0.22 * (1.0 - 0.45 * t);
      return { x, y0: y - halfThick, y1: y + halfThick };
    };
    for (let i = 0; i < segs; i++) {
      const p0 = ptSpring(i / segs);
      const p1 = ptSpring((i + 1) / segs);
      addQuadFace([p1.x, p1.y0, zS1], [p0.x, p0.y0, zS1], [p0.x, p0.y1, zS1], [p1.x, p1.y1, zS1], 0, 0, 1, 1, 0.82);
      addQuadFace([p0.x, p0.y0, zS0], [p1.x, p1.y0, zS0], [p1.x, p1.y1, zS0], [p0.x, p0.y1, zS0], 0, 0, 1, 1, 0.58);
      addQuadFace([p1.x, p1.y0, zS0], [p0.x, p0.y0, zS0], [p0.x, p0.y0, zS1], [p1.x, p1.y0, zS1], 0, 0, 1, 1, 0.78);
      addQuadFace([p0.x, p0.y1, zS0], [p1.x, p1.y1, zS0], [p1.x, p1.y1, zS1], [p0.x, p0.y1, zS1], 0, 0, 1, 1, 0.78);
    }
    endMesh(name, start);
  }

  // 27. Coiled Archimedes Spiral Mainspring / Hairspring (7 concentric spiral turns with crisp radial spacing)
  function buildSpiralSpring(name, rInner = 1.12, rOuter = 4.48, turns = 7.0, h = 0.42) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    addSubCylinderZ(0, 0, rInner, rInner * 0.62, z0 * 1.3, z1 * 1.3, 20, 0.82);

    const steps = Math.round(turns * 30);
    const bladeT = 0.11;
    for (let i = 0; i < steps; i++) {
      const u0 = i / steps, u1 = (i + 1) / steps;
      const a0 = u0 * turns * Math.PI * 2;
      const a1 = u1 * turns * Math.PI * 2;
      const r0 = rInner + (rOuter - rInner) * u0;
      const r1 = rInner + (rOuter - rInner) * u1;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      const p0In = [c0 * (r0 - bladeT), s0 * (r0 - bladeT)];
      const p0Out = [c0 * (r0 + bladeT), s0 * (r0 + bladeT)];
      const p1In = [c1 * (r1 - bladeT), s1 * (r1 - bladeT)];
      const p1Out = [c1 * (r1 + bladeT), s1 * (r1 + bladeT)];

      addQuadFace([p0Out[0], p0Out[1], z0], [p1Out[0], p1Out[1], z0], [p1Out[0], p1Out[1], z1], [p0Out[0], p0Out[1], z1], u0, 0, u1, 1, 0.88);
      addQuadFace([p1In[0], p1In[1], z0], [p0In[0], p0In[1], z0], [p0In[0], p0In[1], z1], [p1In[0], p1In[1], z1], u0, 0, u1, 1, 0.78);
      addQuadFace([p0In[0], p0In[1], z1], [p0Out[0], p0Out[1], z1], [p1Out[0], p1Out[1], z1], [p1In[0], p1In[1], z1], u0, 0, u1, 1, 0.94);
      addQuadFace([p1In[0], p1In[1], z0], [p1Out[0], p1Out[1], z0], [p0Out[0], p0Out[1], z0], [p0In[0], p0In[1], z0], u0, 0, u1, 1, 0.64);
    }
    // Outer hairspring anchor stud
    const aEnd = turns * Math.PI * 2;
    addSubCylinderZ(Math.cos(aEnd) * rOuter, Math.sin(aEnd) * rOuter, 0.42, 0, z0 * 1.4, z1 * 1.4, 12, 0.90);
    endMesh(name, start);
  }

  // 28. Rigid Bell-Crank Equation-of-Time Lever (Roller Arm L = 13.0 along +X + Toothed Sector Arc)
  function buildSectorLever(name, armRoller = 13.0, armSector = 9.5, sectorSpanDeg = 38, teeth = 9, h = 1.4) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    // Central pivot boss at (0, 0)
    addSubCylinderZ(0, 0, 2.15, 0.95, z0 * 1.25, z1 * 1.25, 28, 0.86);
    // Rigid primary roller arm along +X to exact length armRoller = 13.000
    addSubBox(1.65, -0.68, z0 * 0.88, armRoller - 0.85, 0.68, z1 * 0.88, 0.86);
    addSubCylinderZ(armRoller, 0, 1.35, 0.52, z0 * 1.1, z1 * 1.1, 20, 0.88);

    // Secondary bell-crank sector arm centered around +120° (pointing toward Saros column differential)
    const secCenter = (120 * Math.PI) / 180;
    const halfSpan = ((sectorSpanDeg * 0.5) * Math.PI) / 180;
    const aStart = secCenter - halfSpan;
    const aEnd = secCenter + halfSpan;
    const rRimIn = armSector - 1.35;

    for (const strutAng of [aStart + 0.06, aEnd - 0.06]) {
      const ux = Math.cos(strutAng), uy = Math.sin(strutAng);
      const vx = -uy * 0.48, vy = ux * 0.48;
      const r0 = 1.75, r1 = rRimIn + 0.25;
      addQuadFace([ux * r0 - vx, uy * r0 - vy, z1 * 0.8], [ux * r1 - vx, uy * r1 - vy, z1 * 0.8], [ux * r1 + vx, uy * r1 + vy, z1 * 0.8], [ux * r0 + vx, uy * r0 + vy, z1 * 0.8], 0, 0, 1, 1, 0.82);
      addQuadFace([ux * r0 + vx, uy * r0 + vy, z0 * 0.8], [ux * r1 + vx, uy * r1 + vy, z0 * 0.8], [ux * r1 - vx, uy * r1 - vy, z0 * 0.8], [ux * r0 - vx, uy * r0 - vy, z0 * 0.8], 0, 0, 1, 1, 0.62);
      addQuadFace([ux * r0 - vx, uy * r0 - vy, z0 * 0.8], [ux * r1 - vx, uy * r1 - vy, z0 * 0.8], [ux * r1 - vx, uy * r1 - vy, z1 * 0.8], [ux * r0 - vx, uy * r0 - vy, z1 * 0.8], 0, 0, 1, 1, 0.82);
      addQuadFace([ux * r1 + vx, uy * r1 + vy, z0 * 0.8], [ux * r0 + vx, uy * r0 + vy, z0 * 0.8], [ux * r0 + vx, uy * r0 + vy, z1 * 0.8], [ux * r1 + vx, uy * r1 + vy, z1 * 0.8], 0, 0, 1, 1, 0.82);
    }

    // Toothed involute sector arc from aStart to aEnd with solid top (+Z) and bottom (-Z) tooth caps
    // Module matches mating 12T gear_gal_io (rPitch = 2.10 -> mod = 0.35, circular pitch = 1.100 u)
    const mod = 0.35;
    const rDed = armSector - mod * 0.50;
    const rAdd = armSector + mod * 0.42;
    const pitchA = (aEnd - aStart) / teeth;
    const profile = [];
    for (let t = 0; t < teeth; t++) {
      const aBase = aStart + t * pitchA;
      const pts = [
        [aBase + pitchA * 0.00, rDed],
        [aBase + pitchA * 0.16, rDed],
        [aBase + pitchA * 0.27, armSector],
        [aBase + pitchA * 0.34, rAdd],
        [aBase + pitchA * 0.66, rAdd],
        [aBase + pitchA * 0.73, armSector],
        [aBase + pitchA * 0.84, rDed],
        [aBase + pitchA * 1.00, rDed],
      ];
      for (let k = 0; k < pts.length; k++) {
        if (t > 0 && k === 0) continue;
        const [ang, r] = pts[k];
        profile.push({ x: Math.cos(ang) * r, y: Math.sin(ang) * r, ang, r });
      }
    }

    for (let i = 0; i < profile.length - 1; i++) {
      const p0 = profile[i];
      const p1 = profile[i + 1];
      const c0 = Math.cos(p0.ang), s0 = Math.sin(p0.ang);
      const c1 = Math.cos(p1.ang), s1 = Math.sin(p1.ang);
      const pIn0 = [c0 * rRimIn, s0 * rRimIn];
      const pIn1 = [c1 * rRimIn, s1 * rRimIn];
      const edge = Math.abs(p0.r - p1.r) > 0.04 ? 0.88 : 0.76;

      // Outer tooth flank/crest/root vertical quad
      addQuadFace([p0.x, p0.y, z0], [p1.x, p1.y, z0], [p1.x, p1.y, z1], [p0.x, p0.y, z1], 0, 0, 1, 1, edge);
      // Inner rim concave vertical quad
      addQuadFace([pIn1[0], pIn1[1], z0], [pIn0[0], pIn0[1], z0], [pIn0[0], pIn0[1], z1], [pIn1[0], pIn1[1], z1], 0, 0, 1, 1, 0.78);
      // Solid top (+Z) and bottom (-Z) radial strip quads capping both rim and full tooth profile!
      addQuadFace([pIn0[0], pIn0[1], z1], [p0.x, p0.y, z1], [p1.x, p1.y, z1], [pIn1[0], pIn1[1], z1], 0, 0, 1, 1, 0.84);
      addQuadFace([pIn1[0], pIn1[1], z0], [p1.x, p1.y, z0], [p0.x, p0.y, z0], [pIn0[0], pIn0[1], z0], 0, 0, 1, 1, 0.62);
    }
    // End-cap walls at aStart and aEnd closing the sector arc
    const cs0 = Math.cos(aStart), sn0 = Math.sin(aStart);
    const cs1 = Math.cos(aEnd), sn1 = Math.sin(aEnd);
    addQuadFace([cs0 * rRimIn, sn0 * rRimIn, z0], [cs0 * rDed, sn0 * rDed, z0], [cs0 * rDed, sn0 * rDed, z1], [cs0 * rRimIn, sn0 * rRimIn, z1], 0, 0, 1, 1, 0.86);
    addQuadFace([cs1 * rDed, sn1 * rDed, z0], [cs1 * rRimIn, sn1 * rRimIn, z0], [cs1 * rRimIn, sn1 * rRimIn, z1], [cs1 * rDed, sn1 * rDed, z1], 0, 0, 1, 1, 0.86);
    endMesh(name, start);
  }

  // 29. Hemispherical Bimetallic Lunar Synodic Phase Hood (covering +X dark hemisphere)
  function buildHemisphereCup(name, radius = 1.92, wallT = 0.16, latSteps = 16, lonSteps = 24) {
    const start = beginMesh();
    const rOut = radius;
    const rIn = radius - wallT;
    const outerGrid = [];
    const innerGrid = [];

    for (let i = 0; i <= latSteps; i++) {
      const v = i / latSteps;
      const theta = 0.08 * Math.PI + v * 0.84 * Math.PI;
      const st = Math.sin(theta), ct = Math.cos(theta);
      const rowOut = [];
      const rowIn = [];
      for (let j = 0; j <= lonSteps; j++) {
        const u = j / lonSteps;
        const phi = -Math.PI * 0.5 + u * Math.PI; // +X hemisphere
        const nx = st * Math.cos(phi);
        const ny = st * Math.sin(phi);
        const nz = ct;
        const edge = j === 0 || j === lonSteps ? 0.88 : 0.12;
        rowOut.push(pushVert(nx * rOut, ny * rOut, nz * rOut, nx, ny, nz, u, v, edge));
        rowIn.push(pushVert(nx * rIn, ny * rIn, nz * rIn, -nx, -ny, -nz, u, v, edge));
      }
      outerGrid.push(rowOut);
      innerGrid.push(rowIn);
    }

    for (let i = 0; i < latSteps; i++) {
      for (let j = 0; j < lonSteps; j++) {
        pushQuad(outerGrid[i][j], outerGrid[i + 1][j], outerGrid[i + 1][j + 1], outerGrid[i][j + 1]);
        pushQuad(innerGrid[i][j + 1], innerGrid[i + 1][j + 1], innerGrid[i + 1][j], innerGrid[i][j]);
      }
    }
    // Lower bevel phase drive collar
    addSubCylinderZ(0, 0, 0.98, 0.72, -radius - 0.35, -radius + 0.25, 20, 0.85);
    endMesh(name, start);
  }

  // 30. Skeletonized Horological Movement Cage ("Platine et Ponts Squelettes Ajurés")
  // Features open recessed front transmission galleries (so takeoff/input bevels & shafts are 100% unburied)
  // and openworked Swiss skeleton bridge arches & circular bearing islands revealing the recessed perlage baseplate below.
  function buildSkeletonCage(name, w, d, z0, z1) {
    const start = beginMesh();
    const hw = w * 0.5, hd = d * 0.5;
    const plateB = 2.2;
    const plateT = 2.2;
    const rimW = 4.6;
    const zTop0 = z1 - plateT;

    // 1) Lower perlage movement baseplate (z0 .. z0 + plateB) with chamfered perimeter & interior web so open skeleton windows reveal gilt perlage!
    addSubBox(-hw, -hd, z0, hw, hd, z0 + plateB, 0.86);

    // 2) Upper Swiss skeleton bridge deck (zTop0 .. z1) with open recessed transmission gallery and wide pierced skeleton windows:
    const frontRailW = 4.8;
    addSubBox(-hw, hd - rimW, zTop0, hw, hd, z1, 0.92);

    if (name === "driveBox") {
      // driveBox centered at [-113, 0] (hw = 34, hd = 32, z1 = 14)
      // Front perimeter rail stops at x = 23.0 so the front-right takeoff bevel gallery (x in [21.5, hw]) opens directly to the front!
      addSubBox(-hw, -hd, zTop0, 23.0, -hd + frontRailW, z1, 0.92);
      addSubBox(-hw, -hd + frontRailW, zTop0, -hw + rimW, hd - rimW, z1, 0.92);
      // Right perimeter rail across y in [-15.2, hd - rimW] (leaving y in [-hd, -15.2] open for the takeoff bevel gallery!)
      addSubBox(hw - rimW, -15.2, zTop0, hw, hd - rimW, z1, 0.92);
      // Open recessed front-right takeoff gallery shelf at z = 7.4 .. 8.5 under [-79.60, -21.80..-23.43] (local x in [21.5, hw], y in [-hd, -15.2])
      addSubBox(21.5, -hd, 7.4, hw, -15.2, 8.5, 0.90);
      addSubBox(21.5, -hd, 8.5, 23.0, -15.2, z1, 0.92);
      addSubBox(21.5, -16.6, 8.5, hw, -15.2, z1, 0.92);

      // Slender chamfered Swiss skeleton bridge arms spanning between circular bearing islands (leaving 4 large open skeleton cutouts!)
      addSubBox(-hw + rimW, -18.0, zTop0, 21.5, -12.6, z1, 0.94);
      addSubBox(-hw + rimW, -5.8, zTop0, 2.0, -1.2, z1, 0.94);
      addSubBox(-14.5, -hd + frontRailW, zTop0, -9.5, hd - rimW, z1, 0.94);
      addSubBox(6.5, -hd + frontRailW, zTop0, 11.5, hd - rimW, z1, 0.94);
      addSubBox(11.5, -13.5, zTop0, hw - rimW, -7.8, z1, 0.94);

      // Turned circular horological bearing bosses at every barrel, fusee, escapement & idler post (relative to driveBox center [-108, 0]):
      const driveIslands = [
        [-13.0, 11.0, 11.2, 4.2],   // Worm barrel annular bridge ring (world [-121, 11], open center!)
        [-17.0, -15.5, 7.4, 2.2],   // Conical Fusee & ratchet boss (world [-125, -15.5])
        [-17.0, -23.65, 2.6, 0.0],  // Click pawl boss (world [-125, -23.65])
        [-27.6, -15.5, 2.5, 0.0],   // Fusee bridge cock foot boss (world [-135.6, -15.5])
        [4.0, -15.5, 6.8, 2.4],     // Mainspring barrel annular bridge ring (world [-104, -15.5])
        [24.0, -11.0, 4.4, 0.0],    // Verge escapement footstep boss (world [-84, -11])
        [24.0, -4.5, 5.2, 0.0],     // Gallows suspension bracket & flanking balusters boss (world [-84, -4.5])
        [19.2, -11.0, 3.2, 0.0],    // Crown wheel & escapement drive arbor boss (world [-88.8, -11])
        [-8.6, -3.6, 2.8, 0.0],     // Outboard worm drop arbor boss (world [-116.6, -3.6])
        [-13.8, -6.8, 2.4, 0.0],    // Winding pinion boss (world [-121.8, -6.8])
        [-6.5, -21.8, 2.6, 0.0],    // Great-wheel idler boss (world [-114.5, -21.8])
        [14.02, -12.53, 2.6, 0.0],  // Escapement coupling idler boss (world [-93.98, -12.53])
        [22.4, -17.5, 2.8, 0.0],    // Takeoff idler boss (world [-85.6, -17.5])
      ];
      for (const [ix, iy, rO, rI] of driveIslands) {
        addSubCylinderZ(ix, iy, rO, rI, zTop0 + 0.03, z1 + 0.03, 28, 0.94);
      }
    } else {
      // ephemBox centered at [94, 0] (w = 74, d = 66 -> hw = 37, hd = 33, z1 = 13):
      // Open recessed front-left/center transmission gallery across local x in [-hw, 6.2] (world x in [57, 100.2]), y in [-hd, -18.2]
      // dropping the deck to z = 7.4 .. 8.5 so input bevels [64.0..65.4, -23.43], Branch A shaft, flangedCoupling [79.5, -23.43]
      // and Declination bevels [94.0..95.2, -23.43] sit 100% unburied and unoccluded in open air!
      addSubBox(4.8, -hd, zTop0, hw, -hd + frontRailW, z1, 0.92);
      addSubBox(-hw, -hd, 7.4, 6.2, -18.2, 8.5, 0.90);
      addSubBox(4.8, -hd, 8.5, 6.2, -18.2, z1, 0.92);
      addSubBox(-hw, -19.6, 8.5, 6.2, -18.2, z1, 0.92);
      // Left perimeter rail behind the transmission gallery (y in [-18.2, hd - rimW])
      addSubBox(-hw, -18.2, zTop0, -hw + rimW, hd - rimW, z1, 0.92);
      // Right perimeter rail (full span)
      addSubBox(hw - rimW, -hd + frontRailW, zTop0, hw, hd - rimW, z1, 0.92);

      // Slender chamfered Swiss skeleton bridge arms spanning between circular bearing islands (starting at y = -15.6 behind the gallery!)
      addSubBox(-hw + rimW, -15.6, zTop0, hw - rimW, -10.2, z1, 0.94);
      addSubBox(-hw + rimW, -5.2, zTop0, -14.0, -0.8, z1, 0.94);
      addSubBox(-hw + rimW, 11.2, zTop0, 12.0, 16.8, z1, 0.94);
      addSubBox(-22.5, -15.6, zTop0, -17.5, hd - rimW, z1, 0.94);
      addSubBox(-1.2, -15.6, zTop0, 3.8, hd - rimW, z1, 0.94);

      // Turned circular horological bearing bosses at every barrel, cam, lever, Geneva & Saros post:
      const ephemIslands = [
        [1.2, -11.8, 8.4, 3.8],     // Declination barrel annular bridge ring (world [95.2, -11.8], outer y = -20.2 clear of y = -23.43 bevels!)
        [-20.0, -3.0, 4.6, 0.0],    // EoT Kidney Cam boss (world [74.0, -3.0])
        [-27.85, -5.6, 2.4, 0.0],   // Cam bridge cock foot boss (world [66.15, -5.6])
        [-14.2, 10.0, 3.4, 0.0],    // EoT Sector Lever pivot boss (world [79.8, 10.0])
        [-15.46, 17.96, 2.4, 0.0],  // Sector bridge cock foot boss (world [78.54, 17.96])
        [-4.61, 16.53, 2.6, 0.0],   // Sector differential output pinion boss (world [89.39, 16.53])
        [-24.3, -13.215, 2.8, 0.0], // Branch B idler boss (world [69.7, -13.215])
        [-24.75, 6.22, 2.8, 0.0],   // Branch C idler boss (world [69.25, 6.22])
        [-28.5, 13.5, 3.6, 0.0],    // Geneva driver boss (world [65.5, 13.5])
        [-28.5, 20.7, 3.6, 0.0],    // Geneva wheel boss (world [65.5, 20.7])
        [-35.16, 17.28, 2.4, 0.0],  // Geneva bridge left outrigger pillar boss (world [58.84, 17.28])
        [-21.84, 17.28, 2.4, 0.0],  // Geneva bridge right outrigger pillar boss (world [72.16, 17.28])
        [-19.0, 26.8, 2.5, 0.0],    // Geneva sautoir jumperSpring foot boss (world [75.0, 26.8])
        [2.7, 14.26, 2.6, 0.0],     // Saros coupling idler boss (world [96.7, 14.26])
        [10.0, 12.0, 6.2, 2.2],     // Saros dial column annular boss (world [104.0, 12.0])
        [28.0, -12.5, 2.8, 0.0],    // Metonic coupling countershaft boss (world [122.0, -12.5])
        [21.5, -12.5, 4.2, 0.0],    // Metonic/Olympiad sub-turret boss (world [115.5, -12.5])
        [21.5, -20.56, 2.5, 0.0],   // Metonic bridge cock foot boss (world [115.5, -20.56])
      ];
      for (const [ix, iy, rO, rI] of ephemIslands) {
        addSubCylinderZ(ix, iy, rO, rI, zTop0 + 0.03, z1 + 0.03, 28, 0.94);
      }
    }

    // 3) 6 Turned Horological Baluster Pillars supporting upper bridge (or recessed gallery floor z = 7.4) from lower plate
    const pxOff = hw - 3.6, pyOff = hd - 3.6;
    const pillarCoords = [
      [-pxOff, -pyOff], [0, -pyOff], [pxOff, -pyOff],
      [-pxOff, pyOff],  [0, pyOff],  [pxOff, pyOff],
    ];
    const zPil0 = z0 + plateB;
    for (const [px, py] of pillarCoords) {
      const inGallery =
        (name === "driveBox" && px > 20 && py < -15) ||
        (name === "ephemBox" && px < 6 && py < -18);
      const zPil1 = inGallery ? 7.4 : zTop0;
      const pilH = zPil1 - zPil0;
      addSubCylinderZ(px, py, 2.35, 0, zPil0, zPil0 + pilH * 0.14, 18, 0.85);
      addSubCylinderZ(px, py, 1.45, 0, zPil0 + pilH * 0.14, zPil0 + pilH * 0.42, 18, 0.72);
      addSubCylinderZ(px, py, 2.15, 0, zPil0 + pilH * 0.42, zPil0 + pilH * 0.58, 18, 0.88);
      addSubCylinderZ(px, py, 1.45, 0, zPil0 + pilH * 0.58, zPil0 + pilH * 0.86, 18, 0.72);
      addSubCylinderZ(px, py, 2.35, 0, zPil0 + pilH * 0.86, zPil1, 18, 0.85);
    }
    endMesh(name, start);
  }

  // 31. Turned Brass Baluster Collar / Pillar (for planetary risers & movement bridges)
  function buildBalusterPillar(name, rBase = 2.2, rWaist = 1.35, rSwell = 2.05, height = 8.0, rBore = 0.0) {
    const start = beginMesh();
    const segs = 28;
    const prof = [
      [0.00, rBase],
      [0.14, rBase * 0.92],
      [0.28, rWaist],
      [0.44, rSwell * 0.96],
      [0.50, rSwell],
      [0.56, rSwell * 0.96],
      [0.72, rWaist],
      [0.86, rBase * 0.92],
      [1.00, rBase],
    ];
    for (let k = 0; k < prof.length - 1; k++) {
      const [t0, r0] = prof[k];
      const [t1, r1] = prof[k + 1];
      const z0 = (t0 - 0.5) * height;
      const z1 = (t1 - 0.5) * height;
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2;
        const a1 = ((i + 1) / segs) * Math.PI * 2;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const edge = k === 0 || k === 4 || k === prof.length - 2 ? 0.78 : 0.18;
        const p00 = pushVert(c0 * r0, s0 * r0, z0, c0, s0, 0.15, i / segs, t0, edge);
        const p10 = pushVert(c1 * r0, s1 * r0, z0, c1, s1, 0.15, (i + 1) / segs, t0, edge);
        const p11 = pushVert(c1 * r1, s1 * r1, z1, c1, s1, 0.15, (i + 1) / segs, t1, edge);
        const p01 = pushVert(c0 * r1, s0 * r1, z1, c0, s0, 0.15, i / segs, t1, edge);
        pushQuad(p00, p10, p11, p01);
      }
    }
    addSubCylinderZ(0, 0, rBase, rBore, -height * 0.5, -height * 0.44, segs, 0.85);
    addSubCylinderZ(0, 0, rBase, rBore, height * 0.44, height * 0.5, segs, 0.85);
    endMesh(name, start);
  }

  // 32. Saturn Concentric Multi-Ring System (C Ring, B Ring, Cassini Division, A Ring, Encke Gap)
  function buildSaturnRings(name, rIn = 5.6, rOut = 12.4, h = 0.22, segs = 72, radialSteps = 12) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    for (let rIdx = 0; rIdx < radialSteps; rIdx++) {
      const u0 = rIdx / radialSteps;
      const u1 = (rIdx + 1) / radialSteps;
      const ra = rIn + (rOut - rIn) * u0;
      const rb = rIn + (rOut - rIn) * u1;
      const edge = rIdx === 0 || rIdx === radialSteps - 1 || rIdx === 7 ? 0.82 : 0.12;
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2;
        const a1 = ((i + 1) / segs) * Math.PI * 2;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const v0 = i / segs, v1 = (i + 1) / segs;

        // Top face (+Z) with radial coordinate in uv.x
        const t00 = pushVert(c0 * ra, s0 * ra, z1, 0, 0, 1, u0, v0, edge);
        const t10 = pushVert(c0 * rb, s0 * rb, z1, 0, 0, 1, u1, v0, edge);
        const t11 = pushVert(c1 * rb, s1 * rb, z1, 0, 0, 1, u1, v1, edge);
        const t01 = pushVert(c1 * ra, s1 * ra, z1, 0, 0, 1, u0, v1, edge);
        pushQuad(t00, t10, t11, t01);

        // Bottom face (-Z)
        const b00 = pushVert(c1 * ra, s1 * ra, z0, 0, 0, -1, u0, v1, edge);
        const b10 = pushVert(c1 * rb, s1 * rb, z0, 0, 0, -1, u1, v1, edge);
        const b11 = pushVert(c0 * rb, s0 * rb, z0, 0, 0, -1, u1, v0, edge);
        const b01 = pushVert(c0 * ra, s0 * ra, z0, 0, 0, -1, u0, v0, edge);
        pushQuad(b00, b10, b11, b01);
      }
    }
    // Outer & inner beveled rims
    addSubCylinderZ(0, 0, rOut, rOut - 0.25, z0, z1, segs, 0.88);
    addSubCylinderZ(0, 0, rIn + 0.25, rIn, z0, z1, segs, 0.88);
    endMesh(name, start);
  }

  // 33. Da Vinci Articulated Flat-Plate Roller Chain Link ("Catena Articolata", Codex Madrid I f. 10r)
  function buildDaVinciChainLink(name, pitch = 1.45, w = 0.78, h = 0.52) {
    const start = beginMesh();
    const hp = pitch * 0.5;
    const hw = w * 0.5;
    const hh = h * 0.5;
    // Upper and lower waisted side-plates
    addSubBox(-hp, -hw * 0.78, hh * 0.42, hp, hw * 0.78, hh, 0.86);
    addSubBox(-hp, -hw * 0.78, -hh, hp, hw * 0.78, -hh * 0.42, 0.86);
    // End eye bosses around both rivet pins
    for (const sx of [-hp, hp]) {
      addSubCylinderZ(sx, 0, hw * 0.96, hw * 0.34, hh * 0.38, hh * 1.04, 14, 0.88);
      addSubCylinderZ(sx, 0, hw * 0.96, hw * 0.34, -hh * 1.04, -hh * 0.38, 14, 0.88);
      // Central hardened steel rivet pin & roller sleeve
      addSubCylinderZ(sx, 0, hw * 0.56, 0, -hh * 0.42, hh * 0.42, 12, 0.78);
      addSubCylinderZ(sx, 0, hw * 0.32, 0, -hh * 1.28, hh * 1.28, 10, 0.92);
    }
    endMesh(name, start);
  }

  // 34. Swiss High-Arch Pierced S-Scroll Horological Bridge Cock ("Coq d'Horlogerie Suisse")
  // Vertical neck rises immediately at the foot (t < 0.16) so the entire mid-span arch maintains zBot >= h - 0.45,
  // clearing all underlying kidney cams, sector levers, and balance hairsprings with a clean open air gap.
  function buildBridgeCock(name, span = 12.0, w = 3.2, h = 4.2) {
    const start = beginMesh();
    const hw = w * 0.5;
    // 1) Turned circular mounting foot boss & stepped vertical pillar neck at x = -span + 1.4 rising directly to h - 0.42
    const footX = -span + 1.4;
    addSubCylinderZ(footX, 0, hw * 1.18, hw * 0.38, 0, 1.35, 24, 0.86);
    addSubCylinderZ(footX, 0, hw * 0.92, hw * 0.38, 1.35, h - 0.42, 22, 0.82);
    addSubCylinderZ(footX, -hw * 0.72, 0.46, 0.0, 1.35, 1.68, 12, 0.90);
    addSubCylinderZ(footX, hw * 0.72, 0.46, 0.0, 1.35, 1.68, 12, 0.90);
    // Twin side volute curls flanking the upper foot-pillar capital
    addSubCylinderZ(footX + 1.15, -hw * 0.84, 0.62, 0.22, h - 0.65, h + 0.15, 14, 0.84);
    addSubCylinderZ(footX + 1.15, hw * 0.84, 0.62, 0.22, h - 0.65, h + 0.15, 14, 0.84);

    // 2) Continuous 36-segment high-clearance swept Swiss bridge arch from footX + 0.6 to -1.10
    const segs = 36;
    const xStart = footX + 0.6;
    const xEnd = -1.10;
    const thick = 0.82;
    const profileAt = (t) => {
      const x = xStart + (xEnd - xStart) * t;
      const archCrown = 0.48 * Math.sin(t * Math.PI);
      const zBot = (h - 0.42) + archCrown;
      const zTop = zBot + thick * (1.0 - 0.08 * Math.sin(t * Math.PI));
      // Graceful waisted S-scroll width with mid-span medallion swell
      const halfW = hw * (0.84 - 0.22 * Math.sin(t * Math.PI) + 0.10 * Math.sin(t * Math.PI * 3));
      return { x, zBot, zTop, halfW };
    };

    for (let i = 0; i < segs; i++) {
      const p0 = profileAt(i / segs);
      const p1 = profileAt((i + 1) / segs);
      const u0 = i / segs, u1 = (i + 1) / segs;

      // Top face (+Z/normal)
      addQuadFace(
        [p0.x, -p0.halfW, p0.zTop],
        [p1.x, -p1.halfW, p1.zTop],
        [p1.x, p1.halfW, p1.zTop],
        [p0.x, p0.halfW, p0.zTop],
        u0, 0, u1, 1, 0.88
      );
      // Raised central anglage spine Moulding Rib along the top of the S-scroll arch
      const ribW0 = p0.halfW * 0.34;
      const ribW1 = p1.halfW * 0.34;
      const ribH = 0.26;
      addQuadFace(
        [p0.x, -ribW0, p0.zTop + ribH],
        [p1.x, -ribW1, p1.zTop + ribH],
        [p1.x, ribW1, p1.zTop + ribH],
        [p0.x, ribW0, p0.zTop + ribH],
        u0, 0, u1, 1, 0.94
      );
      addQuadFace(
        [p0.x, -ribW0, p0.zTop],
        [p1.x, -ribW1, p1.zTop],
        [p1.x, -ribW1, p1.zTop + ribH],
        [p0.x, -ribW0, p0.zTop + ribH],
        u0, 0, u1, 1, 0.84
      );
      addQuadFace(
        [p1.x, ribW1, p1.zTop],
        [p0.x, ribW0, p0.zTop],
        [p0.x, ribW0, p0.zTop + ribH],
        [p1.x, ribW1, p1.zTop + ribH],
        u0, 0, u1, 1, 0.84
      );
      // Bottom face (-Z/normal)
      addQuadFace(
        [p0.x, p0.halfW, p0.zBot],
        [p1.x, p1.halfW, p1.zBot],
        [p1.x, -p1.halfW, p1.zBot],
        [p0.x, -p0.halfW, p0.zBot],
        u0, 0, u1, 1, 0.60
      );
      // Front side (-Y)
      addQuadFace(
        [p0.x, -p0.halfW, p0.zBot],
        [p1.x, -p1.halfW, p1.zBot],
        [p1.x, -p1.halfW, p1.zTop],
        [p0.x, -p0.halfW, p0.zTop],
        u0, 0, u1, 1, 0.84
      );
      // Back side (+Y)
      addQuadFace(
        [p1.x, p1.halfW, p1.zBot],
        [p0.x, p0.halfW, p0.zBot],
        [p0.x, p0.halfW, p0.zTop],
        [p1.x, p1.halfW, p1.zTop],
        u0, 0, u1, 1, 0.84
      );
    }

    // 3) Upper jewel bearing ring & beveled collet boss centered at (0, 0, h - 0.42 .. h + 0.56)
    addSubCylinderZ(0, 0, hw * 1.16, hw * 0.44, h - 0.42, h + 0.30, 28, 0.90);
    addSubCylinderZ(0, 0, hw * 0.86, hw * 0.38, h + 0.30, h + 0.56, 24, 0.94);
    // Decorative neck volute curls flanking the jewel boss
    addSubCylinderZ(-1.35, -hw * 0.78, 0.56, 0.20, h - 0.40, h + 0.26, 14, 0.86);
    addSubCylinderZ(-1.35, hw * 0.78, 0.56, 0.20, h - 0.40, h + 0.26, 14, 0.86);
    endMesh(name, start);
  }

  // 35. Da Vinci Forged Two-Ear Butterfly Tensioning Nut ("Dado a Farfalla", Codex Madrid I f. 58r)
  function buildWingNut(name, rHub = 1.15, wingSpan = 3.6, h = 1.4) {
    const start = beginMesh();
    addSubCylinderZ(0, 0, rHub, rHub * 0.45, 0, h * 0.78, 18, 0.86);
    const hw = rHub * 0.38;
    for (const sign of [-1, 1]) {
      // Flared butterfly wing ear
      addSubBox(sign * rHub * 0.65, -hw, h * 0.18, sign * wingSpan * 0.5, hw, h * 1.15, 0.88);
      addSubCylinderZ(sign * wingSpan * 0.48, 0, hw * 1.15, 0, h * 0.25, h * 1.22, 12, 0.90);
    }
    endMesh(name, start);
  }

  // 36. Turned Brass Gravity-Feed Horological Oil Cup Lubricator
  function buildOilCup(name, rCup = 1.05, h = 1.65) {
    const start = beginMesh();
    addSubCylinderZ(0, 0, rCup * 0.44, 0, 0, h * 0.30, 14, 0.80);
    addSubCylinderZ(0, 0, rCup * 0.96, 0, h * 0.30, h * 0.44, 18, 0.88);
    addSubCylinderZ(0, 0, rCup * 0.82, rCup * 0.45, h * 0.44, h * 0.82, 18, 0.78);
    addSubCylinderZ(0, 0, rCup * 1.04, 0, h * 0.82, h, 20, 0.90);
    addSubCylinderZ(0, 0, rCup * 0.28, 0, h, h * 1.18, 12, 0.92);
    endMesh(name, start);
  }

  // 37. Da Vinci Overhanging Gallows Suspension Bracket ("Forca di Sospensione", Codex Madrid I f. 27v)
  function buildGallowsBracket(name, reach = 6.5, height = 29.8) {
    const start = beginMesh();
    // Stepped base plinth
    addSubBox(-1.8, -3.2, 0, 1.8, 3.2, 1.8, 0.86);
    // Twin upright cheek standards at y = -2.1 and y = +2.1
    for (const sy of [-2.1, 2.1]) {
      addSubBox(-1.1, sy - 0.55, 1.8, 1.1, sy + 0.55, height, 0.84);
      // Curved diagonal knee brace from (0.8, sy, height - 7.5) to (reach * 0.75, sy, height - 1.1)
      for (let k = 0; k < 5; k++) {
        const t0 = k / 5, t1 = (k + 1) / 5;
        const x0 = 0.8 + (reach * 0.75 - 0.8) * t0;
        const x1 = 0.8 + (reach * 0.75 - 0.8) * t1;
        const z0 = height - 7.5 + 6.4 * Math.sin(t0 * Math.PI * 0.5);
        const z1 = height - 7.5 + 6.4 * Math.sin(t1 * Math.PI * 0.5);
        addSubBox(x0, sy - 0.38, z0, x1 + 0.2, sy + 0.38, z1 + 0.65, 0.82);
      }
      // Overhanging horizontal gallows beam at top
      addSubBox(-1.3, sy - 0.55, height - 1.25, reach + 1.4, sy + 0.55, height, 0.88);
    }
    // Cross-ties and terminal suspension windlass cradle at x = reach
    addSubBox(-1.1, -2.5, height * 0.48, 1.1, 2.5, height * 0.54, 0.85);
    addSubBox(reach - 1.2, -2.65, height - 1.35, reach + 1.4, 2.65, height + 0.35, 0.90);
    addSubCylinderZ(reach, 0, 1.75, 0.55, height - 1.4, height + 0.85, 22, 0.92);
    endMesh(name, start);
  }

  // 38. Renaissance Stepped Foliot Regulation Poise Weight ("Contrappeso del Bilanciere")
  function buildPoiseWeight(name, rBody = 1.85, hBody = 4.2) {
    const start = beginMesh();
    // Top suspension hook collar & conical shoulder
    addSubCylinderZ(0, 0, rBody * 0.38, 0, 0.0, 1.1, 14, 0.86);
    addSubCylinderZ(0, 0, rBody * 0.76, 0, -0.55, 0.0, 20, 0.88);
    // Main cylindrical poise mass with raised central knurled band
    addSubCylinderZ(0, 0, rBody, 0, -hBody, -0.55, 28, 0.84);
    addSubCylinderZ(0, 0, rBody * 1.14, 0, -hBody * 0.62, -hBody * 0.38, 28, 0.92);
    addSubCylinderZ(0, 0, rBody * 1.18, 0, -hBody - 0.45, -hBody, 28, 0.92);
    // Bottom plumb finial
    addSubCylinderZ(0, 0, rBody * 0.52, 0, -hBody - 1.05, -hBody - 0.45, 18, 0.88);
    endMesh(name, start);
  }

  // 39. Dovetail Verge Pallet Collet Clamp ("Colletto della Paletta")
  function buildVergePalletCollet(name, rCollet = 1.25, armLen = 2.2, h = 1.55) {
    const start = beginMesh();
    const hh = h * 0.5;
    // Split collar around verge spindle (rBore = 0.68)
    addSubCylinderZ(0, 0, rCollet, 0.68, -hh, hh, 22, 0.88);
    // Dovetail pallet holder jaw along -X
    addSubBox(-armLen, -0.58, -hh * 0.85, -0.65, 0.58, hh * 0.85, 0.90);
    // Pinch-clamp ears along +X
    addSubBox(0.65, -0.72, -hh * 0.75, rCollet + 0.95, -0.15, hh * 0.75, 0.86);
    addSubBox(0.65, 0.15, -hh * 0.75, rCollet + 0.95, 0.72, hh * 0.75, 0.86);
    endMesh(name, start);
  }

  // 40. Da Vinci Bolted Split-Flange Shaft Coupling ("Giunto a Flangia", Codex Madrid I f. 101r)
  function buildFlangedCoupling(name, rFlange = 2.45, rHub = 1.62, length = 2.8, bolts = 6) {
    const start = beginMesh();
    const hl = length * 0.5;
    const rBore = rHub * 0.62;
    // Left & right hub barrels along local Z
    addSubCylinderZ(0, 0, rHub, rBore, -hl, hl, 24, 0.84);
    // Twin mating central flanges with hairline center seam
    addSubCylinderZ(0, 0, rFlange, rBore, -0.58, -0.03, 28, 0.90);
    addSubCylinderZ(0, 0, rFlange, rBore, 0.03, 0.58, 28, 0.90);
    // Ring of axial coupling bolts piercing both flanges
    const rBolt = (rHub + rFlange) * 0.52;
    for (let b = 0; b < bolts; b++) {
      const ba = (b / bolts) * Math.PI * 2;
      addSubCylinderZ(Math.cos(ba) * rBolt, Math.sin(ba) * rBolt, 0.26, 0, -0.82, 0.82, 8, 0.92);
    }
    endMesh(name, start);
  }

  // 41. Open Architectural Da Vinci Right-Angle Lantern-Crown Gearbox Cage ("Gabbia a Lanterna", Codex Madrid I f. 12v)
  function buildLanternCageFrame(name, w = 12.2, d = 14.0, h = 8.4) {
    const start = beginMesh();
    const hw = w * 0.5, hd = d * 0.5;
    // Lower stepped plinth plate
    addSubBox(-hw, -hd, 0, hw, hd, 1.35, 0.86);
    // 4 Corner Turned Pillars leaving all 4 sides open to view the 90° lantern-to-crown mesh
    const px = hw - 1.35, py = hd - 1.35;
    for (const [cx, cy] of [[-px, -py], [px, -py], [-px, py], [px, py]]) {
      addSubCylinderZ(cx, cy, 1.05, 0, 1.35, h - 1.15, 14, 0.84);
      addSubCylinderZ(cx, cy, 1.32, 0, h * 0.44, h * 0.56, 14, 0.90);
    }
    // Upper pierced architrave perimeter rails
    const rim = 1.85;
    addSubBox(-hw, -hd, h - 1.15, hw, -hd + rim, h, 0.88);
    addSubBox(-hw, hd - rim, h - 1.15, hw, hd, h, 0.88);
    addSubBox(-hw, -hd + rim, h - 1.15, -hw + rim, hd - rim, h, 0.88);
    addSubBox(hw - rim, -hd + rim, h - 1.15, hw, hd - rim, h, 0.88);
    // Transverse top bearing bridge & central journal boss
    addSubBox(-1.6, -hd + rim, h - 1.15, 1.6, hd - rim, h, 0.88);
    addSubCylinderZ(0, 0, 2.05, 0.85, h, h + 0.65, 20, 0.92);
    endMesh(name, start);
  }

  // 42. Antikythera 4-Turn Archimedes Spiral Guide Rail on Saros Dial Face (r = 4.6 .. 11.0)
  function buildSarosSpiralTrack(name, rInner = 4.6, rOuter = 11.0, turns = 4.0, wallW = 0.24, wallH = 0.34) {
    const start = beginMesh();
    const segs = 192;
    const totalAng = turns * Math.PI * 2;
    for (let railSign of [-1, 1]) {
      const rOff = railSign * 0.62;
      let prevIL = 0, prevIR = 0, prevOL = 0, prevOR = 0;
      for (let i = 0; i <= segs; i++) {
        const u = i / segs;
        const theta = u * totalAng;
        const rCenter = rInner + u * (rOuter - rInner) + rOff;
        const rIn = rCenter - wallW * 0.5;
        const rOut = rCenter + wallW * 0.5;
        const cs = Math.cos(theta), sn = Math.sin(theta);

        const vIL = pushVert(cs * rIn, sn * rIn, 0.0, -cs, -sn, 0.2, u, 0.1, 0.72);
        const vIR = pushVert(cs * rIn, sn * rIn, wallH, -cs * 0.5, -sn * 0.5, 0.86, u, 0.35, 0.92);
        const vOR = pushVert(cs * rOut, sn * rOut, wallH, cs * 0.5, sn * 0.5, 0.86, u, 0.65, 0.92);
        const vOL = pushVert(cs * rOut, sn * rOut, 0.0, cs, sn, 0.2, u, 0.9, 0.72);

        if (i > 0) {
          pushQuad(prevIL, vIL, vIR, prevIR);
          pushQuad(prevIR, vIR, vOR, prevOR);
          pushQuad(prevOR, vOR, vOL, prevOL);
        }
        prevIL = vIL;
        prevIR = vIR;
        prevOL = vOL;
        prevOR = vOR;
      }
    }
    endMesh(name, start);
  }

  // ============================================================================
  // Instantiate All Precision CAD Meshes into Shared Buffer
  // ============================================================================
  buildRoundedBox("bedplate", 312, 184, -10, 0, 9.5, 1.5, 8);
  buildRoundedBox("bedInlay", 294, 166, 0, 0.8, 7.5, 0.3, 6);
  buildSkeletonCage("driveBox", 68, 64, 0, 14);
  buildSkeletonCage("ephemBox", 74, 66, 0, 13);
  buildRoundedBox("transferBox", 11.5, 13.5, 0, 7.8, 2.0, 0.45, 4);
  buildRoundedBox("makerPlaque", 74, 9.5, 0, 0.95, 1.2, 0.18, 3);
  buildRoundedBox("scaleBar", 2.6, 3.0, 0, 13.5, 0.55, 0.22, 3);
  buildSarosSpiralTrack("sarosSpiralTrack", 4.6, 8.2, 4.0, 0.20, 0.28);

  buildToothedRack("rulerStrip", 182, 8.6, 1.4, 54);
  buildPillowBlock("pillowBlock", 5.8, 3.4, 4.5, 2.5);
  buildBalusterPillar("balusterPillar", 2.1, 1.25, 1.95, 8.0, 0.0);

  buildCylinder("unitCyl", 1.0, 0.0, -0.5, 0.5, 36, 0.06);
  buildCylinder("unitTube", 1.0, 0.78, -0.5, 0.5, 40, 0.04);
  buildCylinder("plinthDisc", 56.0, 0.0, 0, 5.0, 72, 0.65);
  buildCylinder("plinthRing", 55.2, 46.5, 5.0, 6.2, 72, 0.28);
  buildCylinder("astrolabeDisc", 46.4, 7.8, 0.0, 0.45, 72, 0.08);
  buildCylinder("horizonRing", 52.0, 44.0, 0.0, 1.4, 72, 0.25);
  buildCylinder("handwheelRim", 14.2, 10.6, -2.4, 2.4, 48, 0.45);
  buildCylinder("dialBezel", 21.2, 18.8, 0, 4.6, 60, 0.42);
  buildCylinder("dialFace", 19.0, 0.0, 0, 0.8, 60, 0.05);

  // Windowed barrels with front inspection windows facing -Y (angles 205° to 335°)
  buildWindowedBarrel("wormBarrel", 7.8, 10.6, 0, 18.5, 205, 335, 48);
  buildWindowedBarrel("declBarrel", 5.8, 8.4, 0, 16.0, 200, 340, 44);

  // 6-Level True Coaxial Telescoping Keplerian Spur Gear Train (Center distance D = 24.00)
  // Ordered bottom-to-top with strict telescoping tube bores (r = 1.65, 2.25, 3.00, 3.95, 5.15, 6.70; wall rIn = 0.78*rOut):
  // Level 1 (z = 12.0): Mercury (62T : 16T) — innermost sleeve r = 1.65 (rIn = 1.29 > Sun 1.10)
  buildSpurGear("gear_lay_M", 19.08, 62, 1.85, 3.15, 6, 2.55);
  buildSpurGear("gear_cen_M", 4.92, 16, 1.85, 2.02, 5, 1.65);

  // Level 2 (z = 16.0): Venus (48T : 30T) — 2nd sleeve r = 2.25 (rIn = 1.76 > 1.65)
  buildSpurGear("gear_lay_V", 14.77, 48, 1.85, 3.15, 5, 2.55);
  buildSpurGear("gear_cen_V", 9.23, 30, 1.85, 2.72, 5, 2.25);

  // Level 3 (z = 20.0): Earth (40T : 40T) — 3rd sleeve r = 3.00 (rIn = 2.34 > 2.25)
  buildSpurGear("gear_lay_E", 12.00, 40, 1.85, 3.15, 5, 2.55);
  buildSpurGear("gear_cen_E", 12.00, 40, 1.85, 3.48, 5, 3.00);

  // Level 4 (z = 24.0): Mars (28T : 52T) — 4th sleeve r = 3.95 (rIn = 3.08 > 3.00)
  buildSpurGear("gear_lay_Mars", 8.40, 28, 1.85, 3.08, 5, 2.55);
  buildSpurGear("gear_cen_Mars", 15.60, 52, 1.85, 4.45, 6, 3.95);

  // Level 5 (z = 28.0): Jupiter (14T : 58T) — 5th sleeve r = 5.15 (rIn = 4.02 > 3.95)
  buildSpurGear("gear_lay_Jup", 4.66, 14, 1.85, 2.92, 5, 2.55);
  buildSpurGear("gear_cen_Jup", 19.34, 58, 1.85, 5.65, 6, 5.15);

  // Level 6 (z = 32.0): Saturn (10T : 64T) — 6th outermost sleeve r = 6.70 (rIn = 5.226 > 5.15; 3.24 + 20.76 = 24.00 exact!)
  buildSpurGear("gear_lay_Sat", 3.24, 10, 1.85, 2.75, 0, 2.55);
  buildSpurGear("gear_cen_Sat", 20.76, 64, 1.85, 7.25, 6, 6.70);

  // Tellurian 5-Gear Tooth-to-Tooth Parallel Orientation Train (24T, rPitch = 5.625, spacing = 11.25 -> 4*11.25 = 45.00 = R_E!)
  // Central stationary Sun gear (gear_tellurian_cen) has bore 3.15 around Earth's r = 3.00 sleeve!
  buildSpurGear("gear_tellurian_cen", 5.625, 24, 1.75, 3.65, 5, 3.15);
  buildSpurGear("gear_tellurian", 5.625, 24, 1.75, 1.85, 5, 1.25);

  // Antikythera Pin-and-Slot Lunar Anomaly Coupler Gears & Concentric Apsidal Carrier Gear (e5)
  // gear_lunarHub sits at eccentric offset 0.58 around Earth riser shaft (r = 1.42), so bore = 2.15 > 1.42 + 0.58 = 2.00!
  buildSpurGear("gear_lunarHub", 4.4, 24, 2.0, 2.65, 5, 2.15);
  buildSpurGear("gear_jupHub", 4.8, 26, 2.0, 2.22, 5, 1.62);
  buildSpurGear("gear_wormWheel", 6.8, 28, 2.6, 2.10, 5, 1.40);
  buildSpurGear("gear_rackPinion", 2.8, 14, 2.0, 1.35, 0, 0.90);

  // Tight-Bore 5-Spoke Swiss Horological Train Wheels & Pinions for Arbors in Groups 1, 8, 9, 11, 12, 13, 14, 15:
  buildSpurGear("gear_gal_io", 2.10, 12, 1.1, 1.02, 0, 0.68);
  buildSpurGear("gear_gal_eu", 2.80, 16, 1.1, 1.12, 5, 0.74);
  buildSpurGear("gear_gal_ga", 3.50, 20, 1.1, 1.20, 5, 0.76);
  buildSpurGear("gear_gal_ca", 4.20, 24, 1.1, 1.28, 5, 0.78);
  // Dedicated Coaxial Sleeve-Bore Galilean Spur Gears for Jupiter's 4-Tier Telescoping Sleeve Tower (rSleeve = [0.72, 0.98, 1.32, 1.78]):
  buildSpurGear("gear_jup_io", 2.10, 12, 1.1, 1.15, 0, 0.72);
  buildSpurGear("gear_jup_eu", 2.80, 16, 1.1, 1.45, 5, 0.98);
  buildSpurGear("gear_jup_ga", 3.50, 20, 1.1, 1.85, 5, 1.32);
  buildSpurGear("gear_jup_ca", 4.20, 24, 1.1, 2.35, 5, 1.78);
  buildCarrierBar("carrier_gal_io", 1.8, 6.8, 1.10, 0.85, 1.20, 0.85, 0.72);
  buildCarrierBar("carrier_gal_eu", 2.1, 8.2, 1.15, 0.85, 1.50, 0.85, 0.98);
  buildCarrierBar("carrier_gal_ga", 2.4, 9.8, 1.22, 0.90, 1.90, 0.90, 1.32);
  buildCarrierBar("carrier_gal_ca", 2.8, 11.4, 1.28, 0.95, 2.35, 0.95, 1.78);

  // Da Vinci Crown Wheels, Lantern Pinions, Conical Fusee, Ratchet Click & Foliot Beam
  buildCrownGear("daVinciCrown", 5.6, 20, 1.8, 1.25, 2.1, 4);
  buildLanternPinion("daVinciLantern", 2.8, 10, 4.8, 3.5, 0.42, 1.1);
  buildConicalFusee("daVinciFusee", 6.4, 3.2, 13.5, 6.0);
  buildFoliotBeam("daVinciFoliot", 13.5);
  buildRatchetWheel("ratchetWheel", 6.5, 24, 1.4, 1.8);
  buildClickPawl("clickPawl", 4.6, 1.1, 1.2);
  buildDaVinciChainLink("fuseeChainLink", 1.45, 0.78, 0.52);
  buildBridgeCock("horologicalCock", 12.0, 3.4, 4.2);
  buildWingNut("daVinciWingNut", 1.15, 3.6, 1.4);
  buildOilCup("oilCup", 1.05, 1.65);
  buildGallowsBracket("gallowsBracket", 6.5, 29.8);
  buildPoiseWeight("foliotWeight", 1.85, 4.2);
  buildVergePalletCollet("vergePalletCollet", 1.25, 2.2, 1.55);
  buildFlangedCoupling("flangedCoupling", 2.45, 1.62, 2.8, 6);
  buildLanternCageFrame("lanternCageFrame", 12.2, 14.0, 8.4);

  // Bevel gears, Helical worm screw & Front Ephemeris Leadscrew
  buildBevelGear("bevelGear", 3.8, 2.1, 18, 2.6);
  buildWormScrew("wormScrew", 1.8, 3.2, 13.5, 4.5);
  buildWormScrew("rackLeadscrew", 0.48, 0.68, 174.0, 44.0);

  // Greek Fluted Columns, Cutaway Ecliptic Ring (spanning -25° through +90° back to +205°), and Celestial Armillary Arches
  buildFlutedColumn("greekColumn", 3.4, 2.15, 33.0, 16);
  buildEclipticRing("eclipticRing", 42.5, 53.5, 0, 3.8, -25, 205, 84);
  buildArmillaryArch("armillaryArch", 82.0, 2.8, 1.8, 0, 90, 48);
  buildArmillaryArch("armillaryColure", 82.0, 2.2, 1.5, 0, 90, 48);

  // Horological Planetary Carrier Bars (each with open central bore clearing its sleeve and all inner sleeves!)
  buildCarrierBar("carrier_Sat", 19.5, 90.0, 4.2, 2.4, 8.4, 3.5, 6.70);
  buildCarrierBar("carrier_Jup", 17.5, 76.0, 3.8, 2.3, 6.8, 3.2, 5.15);
  buildCarrierBar("carrier_Mars", 15.5, 62.0, 3.4, 2.2, 5.5, 2.8, 3.95);
  buildCarrierBar("carrier_E", 14.0, 45.0, 4.2, 2.5, 4.5, 3.2, 3.00);
  buildCarrierBar("carrier_V", 11.5, 29.5, 3.2, 2.1, 3.5, 2.6, 2.25);
  buildCarrierBar("carrier_M", 9.0, 17.5, 2.8, 1.9, 2.6, 2.2, 1.65);
  buildCarrierBar("carrier_Moon", 2.8, 8.4, 1.7, 1.3, 2.2, 1.3, 1.42);
  buildCarrierBar("bridgeBar", 0.0, 24.0, 6.4, 3.2, 8.6, 4.6, 6.95, 2.60);
  buildSlottedAnomalyArm("antikytheraSlotArm", 2.2, 2.4, 7.6, 0.65, 8.6, 1.2);

  // Equation-of-Time Kidney Cam, Rigid Sector Lever, Spiral Hairspring, Geneva Indexer & Sculpted Dial Hands
  buildKidneyCam("eotKidneyCam", 5.8, 1.8);
  buildSectorLever("eotSectorLever", 13.0, 9.5, 39.79, 6, 1.4);
  buildSpiralSpring("hairspring", 1.12, 4.48, 7.0, 0.42);
  buildGenevaWheel("genevaWheel", 6.24, 6, 1.6);
  buildGenevaDriver("genevaDriver", 4.4, 2.50, 3.60, 0.52, 1.6);
  buildHemisphereCup("lunarPhaseCup", 1.92, 0.16);
  buildDialHand("hand_sun", 4.0, 16.8, 0.72, 0.55, "sun");
  buildDialHand("hand_eot", 3.4, 15.2, 0.58, 0.48, "eot");
  buildDialHand("hand_dragon", 4.2, 13.4, 0.52, 0.45, "dragon");

  // Spheres, Saturn Rings, Toruses & Scotch-Yoke Plunger
  buildUnitSphere("unitSphere", 32, 48);
  buildSaturnRings("saturnRings", 5.6, 12.4, 0.22, 72, 12);
  buildTorus("lunarNodalRing", 9.8, 0.55, 52, 14);
  buildTorus("earthMeridianRing", 4.95, 0.34, 44, 12);
  buildTorus("earthHorizonRing", 5.45, 0.38, 44, 12);
  buildTorus("jupMoonRing", 9.2, 0.32, 44, 10);
  buildScotchYokePlunger("scotchYokePlunger");

  // 42. Swiss Polished Slotted Countersunk Bridge Screw ("Vis Horlogère Fendue")
  function buildSlottedScrew(name, rHead = 1.0, hHead = 0.55, slotW = 0.22, slotD = 0.28) {
    const start = beginMesh();
    const z0 = 0.0, z1 = hHead;
    // Lower shank & chamfered shoulder
    addSubCylinderZ(0, 0, rHead * 0.52, 0, -hHead * 0.85, z0, 14, 0.72);
    addSubCylinderZ(0, 0, rHead, 0, z0, z1 - slotD, 20, 0.86);
    // Split left & right head halves leaving an open diametral screwdriver slot along X (-slotW..+slotW in Y)
    const segs = 10;
    for (const side of [-1, 1]) {
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 0.82 + (side > 0 ? 0.09 * Math.PI : 1.09 * Math.PI);
        const a1 = ((i + 1) / segs) * Math.PI * 0.82 + (side > 0 ? 0.09 * Math.PI : 1.09 * Math.PI);
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const yIn0 = side * slotW;
        const xIn0 = c0 * rHead * 0.92;
        const xIn1 = c1 * rHead * 0.92;
        addQuadFace(
          [c0 * rHead, s0 * rHead, z1 - slotD],
          [c1 * rHead, s1 * rHead, z1 - slotD],
          [c1 * (rHead * 0.92), s1 * (rHead * 0.92), z1],
          [c0 * (rHead * 0.92), s0 * (rHead * 0.92), z1],
          0, 0, 1, 1, 0.92
        );
        addQuadFace(
          [xIn0, yIn0, z1],
          [c0 * (rHead * 0.92), s0 * (rHead * 0.92), z1],
          [c1 * (rHead * 0.92), s1 * (rHead * 0.92), z1],
          [xIn1, yIn0, z1],
          0, 0, 1, 1, 0.90
        );
      }
      // Vertical walls of the screwdriver slot
      const xSlot = Math.sqrt(Math.max(0.05, rHead * rHead - slotW * slotW)) * 0.92;
      if (side > 0) {
        addQuadFace([-xSlot, slotW, z1 - slotD], [xSlot, slotW, z1 - slotD], [xSlot, slotW, z1], [-xSlot, slotW, z1], 0, 0, 1, 1, 0.95);
      } else {
        addQuadFace([xSlot, -slotW, z1 - slotD], [-xSlot, -slotW, z1 - slotD], [-xSlot, -slotW, z1], [xSlot, -slotW, z1], 0, 0, 1, 1, 0.95);
      }
    }
    endMesh(name, start);
  }

  // 43. Swiss Gold/Brass Beveled Ruby Jewel Chaton ("Chaton Empierré à Vis")
  function buildJewelChaton(name, rOut = 1.65, rBore = 0.72, h = 0.68) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const rMid = rBore + (rOut - rBore) * 0.52;
    // Outer beveled setting flange with 3 peripheral chaton clamping notches
    addSubCylinderZ(0, 0, rOut, rMid, z0, z1 * 0.55, 28, 0.88);
    addSubCylinderZ(0, 0, rOut * 0.90, rBore, z0, z1, 28, 0.94);
    for (let k = 0; k < 3; k++) {
      const a = (k / 3) * Math.PI * 2;
      addSubCylinderZ(Math.cos(a) * (rOut * 0.94), Math.sin(a) * (rOut * 0.94), 0.24, 0, z0, z1 * 0.85, 10, 0.92);
    }
    endMesh(name, start);
  }

  // 44. Sculpted Chamfered Swiss Train Bridge ("Pont de Rouage Suisse") with Anglage Bevels & Twin Outrigger Pillar Wings
  function buildSwissTrainBridge(name, span = 24.0, rCenterBoss = 8.6, rEndBoss = 4.6, rCenterBore = 6.95, rEndBore = 2.60, h = 3.2) {
    const start = beginMesh();
    const hh = h * 0.5;
    const bev = 0.42;
    // Central coaxial turret boss (open bore = 6.95 > Saturn sleeve 6.70!)
    addSubCylinderZ(0, 0, rCenterBoss, rCenterBore, -hh * 1.12, hh * 1.12, 40, 0.90);
    addSubCylinderZ(0, 0, rCenterBoss - bev, rCenterBore + 0.15, hh * 1.12, hh * 1.26, 40, 0.95);

    // Distal countershaft bearing boss at +X = span (open bore = 2.60 > layshaft 2.55!)
    addSubCylinderZ(span, 0, rEndBoss, rEndBore, -hh * 1.12, hh * 1.12, 32, 0.90);
    addSubCylinderZ(span, 0, rEndBoss - bev * 0.8, rEndBore + 0.12, hh * 1.12, hh * 1.26, 32, 0.95);

    // Sculpted waisted S-curve Swiss bridge arm with 45-degree anglage bevels and recessed interior coffer well
    const segs = 24;
    const x0 = rCenterBore + 0.85;
    const x1 = span - rEndBore - 0.65;
    for (let i = 0; i < segs; i++) {
      const t0 = i / segs, t1 = (i + 1) / segs;
      const xa = x0 + (x1 - x0) * t0;
      const xb = x0 + (x1 - x0) * t1;
      const w0 = 3.55 - 0.95 * Math.sin(t0 * Math.PI) + 0.55 * (1 - t0);
      const w1 = 3.55 - 0.95 * Math.sin(t1 * Math.PI) + 0.55 * (1 - t1);

      // Vertical flanks
      addQuadFace([xa, -w0, -hh], [xb, -w1, -hh], [xb, -w1, hh - bev], [xa, -w0, hh - bev], t0, 0, t1, 1, 0.82);
      addQuadFace([xb, w1, -hh], [xa, w0, -hh], [xa, w0, hh - bev], [xb, w1, hh - bev], t0, 0, t1, 1, 0.82);
      // 45° polished anglage bevel strips along both edges
      addQuadFace([xa, -w0, hh - bev], [xb, -w1, hh - bev], [xb, -(w1 - bev), hh], [xa, -(w0 - bev), hh], t0, 0, t1, 1, 0.96);
      addQuadFace([xb, w1, hh - bev], [xa, w0, hh - bev], [xa, w0 - bev, hh], [xb, w1 - bev, hh], t0, 0, t1, 1, 0.96);
      // Top flat bridge table & bottom soffit
      addQuadFace([xa, -(w0 - bev), hh], [xb, -(w1 - bev), hh], [xb, w1 - bev, hh], [xa, w0 - bev, hh], t0, 0, t1, 1, 0.88);
      addQuadFace([xb, -(w1 - bev), -hh], [xa, -(w0 - bev), -hh], [xa, w0 - bev, -hh], [xb, w1 - bev, -hh], t0, 0, t1, 1, 0.62);
    }

    // Twin swept Calatrava outrigger wings arching from bridge waist (x = 12.6, y = +-2.8) to outboard pillar bosses (x = 12.6, y = +-22.2)
    // so vertical support pillars sit strictly outside all 12 coaxial and countershaft gears (dist = 25.53 > 21.03 and 24.96 > 19.34!)
    const wingSegs = 10;
    for (const sy of [-1, 1]) {
      for (let k = 0; k < wingSegs; k++) {
        const t0 = k / wingSegs, t1 = (k + 1) / wingSegs;
        const ya = sy * (2.6 + (22.2 - 2.6) * t0);
        const yb = sy * (2.6 + (22.2 - 2.6) * t1);
        const xc0 = 12.6 - 0.85 * Math.sin(t0 * Math.PI);
        const xc1 = 12.6 - 0.85 * Math.sin(t1 * Math.PI);
        const hw0 = 1.65 * (1 - 0.32 * Math.sin(t0 * Math.PI));
        const hw1 = 1.65 * (1 - 0.32 * Math.sin(t1 * Math.PI));
        if (sy > 0) {
          addQuadFace([xc0 - hw0, ya, hh * 0.92], [xc0 + hw0, ya, hh * 0.92], [xc1 + hw1, yb, hh * 0.92], [xc1 - hw1, yb, hh * 0.92], 0, 0, 1, 1, 0.92);
          addQuadFace([xc1 - hw1, yb, -hh * 0.88], [xc1 + hw1, yb, -hh * 0.88], [xc0 + hw0, ya, -hh * 0.88], [xc0 - hw0, ya, -hh * 0.88], 0, 0, 1, 1, 0.65);
          addQuadFace([xc0 - hw0, ya, -hh * 0.88], [xc0 - hw0, ya, hh * 0.92], [xc1 - hw1, yb, hh * 0.92], [xc1 - hw1, yb, -hh * 0.88], 0, 0, 1, 1, 0.86);
          addQuadFace([xc1 + hw1, yb, -hh * 0.88], [xc1 + hw1, yb, hh * 0.92], [xc0 + hw0, ya, hh * 0.92], [xc0 + hw0, ya, -hh * 0.88], 0, 0, 1, 1, 0.86);
        } else {
          addQuadFace([xc1 - hw1, yb, hh * 0.92], [xc1 + hw1, yb, hh * 0.92], [xc0 + hw0, ya, hh * 0.92], [xc0 - hw0, ya, hh * 0.92], 0, 0, 1, 1, 0.92);
          addQuadFace([xc0 - hw0, ya, -hh * 0.88], [xc0 + hw0, ya, -hh * 0.88], [xc1 + hw1, yb, -hh * 0.88], [xc1 - hw1, yb, -hh * 0.88], 0, 0, 1, 1, 0.65);
          addQuadFace([xc1 - hw1, yb, -hh * 0.88], [xc1 - hw1, yb, hh * 0.92], [xc0 - hw0, ya, hh * 0.92], [xc0 - hw0, ya, -hh * 0.88], 0, 0, 1, 1, 0.86);
          addQuadFace([xc0 + hw0, ya, -hh * 0.88], [xc0 + hw0, ya, hh * 0.92], [xc1 + hw1, yb, hh * 0.92], [xc1 + hw1, yb, -hh * 0.88], 0, 0, 1, 1, 0.86);
        }
      }
      const ey = sy * 22.2;
      addSubCylinderZ(12.6, ey, 2.25, 0.68, -hh * 1.05, hh * 1.05, 22, 0.92);
      // Locating steady-pin boss beside each pillar ear
      addSubCylinderZ(12.6, sy * 19.4, 0.52, 0.0, -hh * 1.25, hh * 1.25, 12, 0.90);
    }
    endMesh(name, start);
  }

  // 45. Slotted Clamping Split-Collet with Tangential Pinch-Bolt Lugs ("Virole Fendue à Vis de Serrage")
  function buildSplitCollet(name, rOut = 1.0, rIn = 0.78, h = 1.0) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    const segs = 32;
    const gapRad = 0.14;
    // C-shaped split collar leaving an open radial clamping kerf at angle 0 (+X)
    for (let i = 0; i < segs; i++) {
      const a0 = gapRad + (i / segs) * (Math.PI * 2 - 2 * gapRad);
      const a1 = gapRad + ((i + 1) / segs) * (Math.PI * 2 - 2 * gapRad);
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      addQuadFace([c0 * rOut, s0 * rOut, z0], [c1 * rOut, s1 * rOut, z0], [c1 * rOut, s1 * rOut, z1], [c0 * rOut, s0 * rOut, z1], i / segs, 0, (i + 1) / segs, 1, 0.80);
      addQuadFace([c1 * rIn, s1 * rIn, z0], [c0 * rIn, s0 * rIn, z0], [c0 * rIn, s0 * rIn, z1], [c1 * rIn, s1 * rIn, z1], i / segs, 0, (i + 1) / segs, 1, 0.70);
      addQuadFace([c0 * rIn, s0 * rIn, z1], [c0 * rOut, s0 * rOut, z1], [c1 * rOut, s1 * rOut, z1], [c1 * rIn, s1 * rIn, z1], 0, 0, 1, 1, 0.88);
      addQuadFace([c1 * rIn, s1 * rIn, z0], [c1 * rOut, s1 * rOut, z0], [c0 * rOut, s0 * rOut, z0], [c0 * rIn, s0 * rIn, z0], 0, 0, 1, 1, 0.62);
      if (i === 0) {
        addQuadFace([c0 * rIn, s0 * rIn, z0], [c0 * rOut, s0 * rOut, z0], [c0 * rOut, s0 * rOut, z1], [c0 * rIn, s0 * rIn, z1], 0, 0, 1, 1, 0.90);
      }
      if (i === segs - 1) {
        addQuadFace([c1 * rOut, s1 * rOut, z0], [c1 * rIn, s1 * rIn, z0], [c1 * rIn, s1 * rIn, z1], [c1 * rOut, s1 * rOut, z1], 0, 0, 1, 1, 0.90);
      }
    }
    // Twin tangential clamping ears projecting along +X flanking the split kerf (y = -0.34..-0.09 and +0.09..+0.34)
    const earX0 = rIn * 0.94, earX1 = rOut + 0.28;
    addSubBox(earX0, -0.36, z0 * 0.88, earX1, -0.09, z1 * 0.88, 0.92);
    addSubBox(earX0, 0.09, z0 * 0.88, earX1, 0.36, z1 * 0.88, 0.92);
    endMesh(name, start);
  }

  // 46. Miniature Horological Roller-Thrust Bearing Cage ("Cage de Butée à Rouleaux")
  function buildRollerThrustCage(name, rOut = 1.0, rIn = 0.78, h = 0.36, rollers = 12) {
    const start = beginMesh();
    const z0 = -h * 0.5, z1 = h * 0.5;
    // Lower and upper perforated bronze raceway rings
    addSubCylinderZ(0, 0, rOut, rIn, z0, z0 + h * 0.24, 28, 0.86);
    addSubCylinderZ(0, 0, rOut, rIn, z1 - h * 0.24, z1, 28, 0.86);
    // Crown of 12 hardened steel bearing rollers/balls between the races
    const rMid = (rIn + rOut) * 0.5;
    const rBall = (rOut - rIn) * 0.36;
    for (let k = 0; k < rollers; k++) {
      const a = (k / rollers) * Math.PI * 2;
      addSubCylinderZ(Math.cos(a) * rMid, Math.sin(a) * rMid, rBall, 0, z0 + h * 0.15, z1 - h * 0.15, 10, 0.92);
    }
    endMesh(name, start);
  }

  // 47. Swiss Leaf-Spring Detent Jumper ("Sautoir de Croix de Malte")
  function buildJumperSpring(name, length = 8.4, h = 0.85) {
    const start = beginMesh();
    const hh = h * 0.5;
    // Anchor foot boss at (0, 0) with locating pin & screw boss
    addSubCylinderZ(0, 0, 1.35, 0.48, -hh * 1.15, hh * 1.15, 20, 0.88);
    addSubBox(-1.1, -0.85, -hh, 0.85, 0.85, hh, 0.86);
    // Swept flexible leaf-spring blade curving from x = 0.8 to x = length - 0.8
    const segs = 18;
    for (let i = 0; i < segs; i++) {
      const t0 = i / segs, t1 = (i + 1) / segs;
      const xa = 0.8 + (length - 1.4) * t0;
      const xb = 0.8 + (length - 1.4) * t1;
      const ya = 0.55 * Math.sin(t0 * Math.PI);
      const yb = 0.55 * Math.sin(t1 * Math.PI);
      const w0 = 0.28 * (1.0 - 0.35 * t0);
      const w1 = 0.28 * (1.0 - 0.35 * t1);
      addQuadFace([xa, ya - w0, hh * 0.8], [xb, yb - w1, hh * 0.8], [xb, yb + w1, hh * 0.8], [xa, ya + w0, hh * 0.8], 0, 0, 1, 1, 0.88);
      addQuadFace([xb, yb - w1, -hh * 0.8], [xa, ya - w0, -hh * 0.8], [xa, ya + w0, -hh * 0.8], [xb, yb + w1, -hh * 0.8], 0, 0, 1, 1, 0.65);
      addQuadFace([xa, ya - w0, -hh * 0.8], [xb, yb - w1, -hh * 0.8], [xb, yb - w1, hh * 0.8], [xa, ya - w0, hh * 0.8], 0, 0, 1, 1, 0.84);
      addQuadFace([xb, yb + w1, -hh * 0.8], [xa, ya + w0, -hh * 0.8], [xa, ya + w0, hh * 0.8], [xb, yb + w1, hh * 0.8], 0, 0, 1, 1, 0.84);
    }
    // Angled V-shaped jewel-beak tip at x = length engaging peripheral indexing notches
    const xTip0 = length - 0.75, xTip1 = length + 0.35;
    addQuadFace([xTip0, -0.45, -hh], [xTip1, 0.0, -hh], [xTip1, 0.0, hh], [xTip0, -0.45, hh], 0, 0, 1, 1, 0.94);
    addQuadFace([xTip1, 0.0, -hh], [xTip0, 0.45, -hh], [xTip0, 0.45, hh], [xTip1, 0.0, hh], 0, 0, 1, 1, 0.94);
    addQuadFace([xTip0, 0.45, -hh], [xTip0, -0.45, -hh], [xTip0, -0.45, hh], [xTip0, 0.45, hh], 0, 0, 1, 1, 0.84);
    const jt0 = pushVert(xTip0, -0.45, hh, 0, 0, 1, 0, 0, 0.92);
    const jt1 = pushVert(xTip1, 0.0, hh, 0, 0, 1, 1, 0.5, 0.92);
    const jt2 = pushVert(xTip0, 0.45, hh, 0, 0, 1, 0, 1, 0.92);
    pushTri(jt0, jt1, jt2);
    const jb0 = pushVert(xTip0, 0.45, -hh, 0, 0, -1, 0, 1, 0.70);
    const jb1 = pushVert(xTip1, 0.0, -hh, 0, 0, -1, 1, 0.5, 0.70);
    const jb2 = pushVert(xTip0, -0.45, -hh, 0, 0, -1, 0, 0, 0.70);
    pushTri(jb0, jb1, jb2);
    endMesh(name, start);
  }

  // 48. Swiss Incabloc Lyre-Shaped Shock-Absorber Spring Clip ("Ressort Lyre Incabloc")
  function buildLyreSpring(name, rRing = 1.6, h = 0.28) {
    const start = beginMesh();
    const hh = h * 0.5;
    const segs = 20;
    // Twin symmetrical S-curved lyre arms flanking the bearing bezel (rMin >= 0.78 * rRing so central sleeves never collide!)
    for (const side of [-1, 1]) {
      for (let i = 0; i < segs; i++) {
        const t0 = i / segs, t1 = (i + 1) / segs;
        const a0 = Math.PI - side * t0 * Math.PI * 0.92;
        const a1 = Math.PI - side * t1 * Math.PI * 0.92;
        const r0 = rRing * (0.94 + 0.08 * Math.sin(t0 * Math.PI * 2));
        const r1 = rRing * (0.94 + 0.08 * Math.sin(t1 * Math.PI * 2));
        const w = rRing * 0.08;
        const x0In = Math.cos(a0) * (r0 - w), y0In = Math.sin(a0) * (r0 - w);
        const x0Out = Math.cos(a0) * (r0 + w), y0Out = Math.sin(a0) * (r0 + w);
        const x1In = Math.cos(a1) * (r1 - w), y1In = Math.sin(a1) * (r1 - w);
        const x1Out = Math.cos(a1) * (r1 + w), y1Out = Math.sin(a1) * (r1 + w);
        if (side > 0) {
          addQuadFace([x1In, y1In, hh], [x1Out, y1Out, hh], [x0Out, y0Out, hh], [x0In, y0In, hh], 0, 0, 1, 1, 0.92);
          addQuadFace([x0In, y0In, -hh], [x0Out, y0Out, -hh], [x1Out, y1Out, -hh], [x1In, y1In, -hh], 0, 0, 1, 1, 0.65);
          addQuadFace([x1Out, y1Out, -hh], [x0Out, y0Out, -hh], [x0Out, y0Out, hh], [x1Out, y1Out, hh], 0, 0, 1, 1, 0.88);
          addQuadFace([x0In, y0In, -hh], [x1In, y1In, -hh], [x1In, y1In, hh], [x0In, y0In, hh], 0, 0, 1, 1, 0.84);
        } else {
          addQuadFace([x0In, y0In, hh], [x0Out, y0Out, hh], [x1Out, y1Out, hh], [x1In, y1In, hh], 0, 0, 1, 1, 0.92);
          addQuadFace([x1In, y1In, -hh], [x1Out, y1Out, -hh], [x0Out, y0Out, -hh], [x0In, y0In, -hh], 0, 0, 1, 1, 0.65);
          addQuadFace([x0Out, y0Out, -hh], [x1Out, y1Out, -hh], [x1Out, y1Out, hh], [x0Out, y0Out, hh], 0, 0, 1, 1, 0.88);
          addQuadFace([x1In, y1In, -hh], [x0In, y0In, -hh], [x0In, y0In, hh], [x1In, y1In, hh], 0, 0, 1, 1, 0.84);
        }
      }
    }
    // Hinge tab at -X (rIn >= 0.82 * rRing)
    addSubBox(-rRing * 1.14, -rRing * 0.24, -hh * 1.2, -rRing * 0.82, rRing * 0.24, hh * 1.2, 0.90);
    endMesh(name, start);
  }

  // 49. Pierced Swiss Spider Bridge Arm with Distal Eccentric Beat/Index Vernier ("Bras d'Araignée Ajuré")
  function buildSpiderBridgeArm(name, rIn = 8.4, rOut = 44.2, wRoot = 3.2, wTip = 2.0, h = 1.6) {
    const start = beginMesh();
    const hh = h * 0.5;
    const hw0 = wRoot * 0.5, hw1 = wTip * 0.5;
    const railW = 0.55;
    // Twin tapered chamfered outer rails from rIn to rOut with 4 open lightening bays
    const segs = 16;
    for (let i = 0; i < segs; i++) {
      const t0 = i / segs, t1 = (i + 1) / segs;
      const xa = rIn + (rOut - rIn) * t0;
      const xb = rIn + (rOut - rIn) * t1;
      const ya = hw0 + (hw1 - hw0) * t0;
      const yb = hw0 + (hw1 - hw0) * t1;
      for (const sgn of [-1, 1]) {
        const yOutA = sgn * ya, yInA = sgn * (ya - railW);
        const yOutB = sgn * yb, yInB = sgn * (yb - railW);
        if (sgn > 0) {
          addQuadFace([xa, yInA, hh], [xb, yInB, hh], [xb, yOutB, hh], [xa, yOutA, hh], 0, 0, 1, 1, 0.90);
          addQuadFace([xb, yInB, -hh], [xa, yInA, -hh], [xa, yOutA, -hh], [xb, yOutB, -hh], 0, 0, 1, 1, 0.65);
          addQuadFace([xa, yOutA, -hh], [xb, yOutB, -hh], [xb, yOutB, hh], [xa, yOutA, hh], 0, 0, 1, 1, 0.86);
          addQuadFace([xb, yInB, -hh], [xa, yInA, -hh], [xa, yInA, hh], [xb, yInB, hh], 0, 0, 1, 1, 0.84);
        } else {
          addQuadFace([xa, yOutA, hh], [xb, yOutB, hh], [xb, yInB, hh], [xa, yInA, hh], 0, 0, 1, 1, 0.90);
          addQuadFace([xb, yOutB, -hh], [xa, yOutA, -hh], [xa, yInA, -hh], [xb, yInB, -hh], 0, 0, 1, 1, 0.65);
          addQuadFace([xb, yOutB, -hh], [xa, yOutA, -hh], [xa, yOutA, hh], [xb, yOutB, hh], 0, 0, 1, 1, 0.86);
          addQuadFace([xa, yInA, -hh], [xb, yInB, -hh], [xb, yInB, hh], [xa, yInA, hh], 0, 0, 1, 1, 0.84);
        }
      }
    }
    // Cross-tie webs dividing the arm into 4 open horological truss bays
    for (let b = 0; b <= 4; b++) {
      const t = b / 4;
      const bx = rIn + (rOut - rIn - 1.2) * t;
      const by = (hw0 + (hw1 - hw0) * t) - railW * 0.8;
      addSubBox(bx, -by, -hh * 0.88, bx + 1.1, by, hh * 0.88, 0.88);
    }
    // Distal mounting foot & engraved Avance / Retard (A/R) beat vernier scale arc at x = rOut
    addSubCylinderZ(rOut, 0, 1.85, 0.58, -hh * 1.15, hh * 1.15, 24, 0.92);
    addSubBox(rOut - 1.6, -2.4, hh * 0.4, rOut + 1.6, 2.4, hh * 1.18, 0.90);
    for (let k = -2; k <= 2; k++) {
      addSubBox(rOut - 0.6, k * 0.85 - 0.10, hh * 1.18, rOut + 0.6, k * 0.85 + 0.10, hh * 1.42, 0.96);
    }
    endMesh(name, start);
  }

  buildSlottedScrew("slottedScrew", 1.0, 0.55, 0.22, 0.28);
  buildJewelChaton("jewelChaton", 1.65, 0.72, 0.68);
  buildSwissTrainBridge("swissTrainBridge", 24.0, 8.6, 4.6, 6.95, 2.60, 3.2);
  buildSplitCollet("splitCollet", 1.0, 0.78, 1.0);
  buildRollerThrustCage("rollerThrustCage", 1.0, 0.78, 0.36, 12);
  buildJumperSpring("jumperSpring", 8.4, 0.85);
  buildLyreSpring("lyreSpring", 1.6, 0.28);
  buildSpiderBridgeArm("spiderBridgeArm", 8.4, 44.2, 3.2, 2.0, 1.6);

  return {
    vertexData: new Float32Array(verts),
    indexData: new Uint32Array(indices),
    meshes,
  };
}
