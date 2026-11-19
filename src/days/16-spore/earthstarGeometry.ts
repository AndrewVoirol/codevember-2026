import * as THREE from 'three/webgpu';
import { PERISTOME_NOZZLE_Y, PERISTOME_RIM_RADIUS, SAC_CENTER_Y } from './bellowsPhysics';

// ---------------------------------------------------------------------------
// Deterministic Seeded PRNG & Smooth 3D Organic Noise (Zero Per-Frame GC)
// ---------------------------------------------------------------------------
function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453123;
  return s - Math.floor(s);
}

function hash2(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

function valueNoise3(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = hash3(ix, iy, iz);
  const n100 = hash3(ix + 1, iy, iz);
  const n010 = hash3(ix, iy + 1, iz);
  const n110 = hash3(ix + 1, iy + 1, iz);
  const n001 = hash3(ix, iy, iz + 1);
  const n101 = hash3(ix + 1, iy, iz + 1);
  const n011 = hash3(ix, iy + 1, iz + 1);
  const n111 = hash3(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

/**
 * Irrational-angle rotated 3-octave FBM (matches TSL `mineralToothFbm`).
 */
function fbm3(x: number, y: number, z: number): number {
  const v0 = valueNoise3(x, y, z);
  const c = 0.80901699;
  const s = 0.58778525;
  const x1 = (x * c - z * s) * 2.03;
  const y1 = (y * c + x * s) * 2.03;
  const z1 = (z * c + y * s) * 2.03;
  const v1 = valueNoise3(x1, y1, z1);

  const x2 = (y1 * c - x1 * s) * 2.07;
  const y2 = (z1 * c + y1 * s) * 2.07;
  const z2 = (x1 * c + z1 * s) * 2.07;
  const v2 = valueNoise3(x2, y2, z2);
  return v0 * 0.57 + v1 * 0.29 + v2 * 0.14;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * 2D Voronoi tectonic desiccation crack evaluator (synced with fragment-stage TSL Voronoi)
 * so vertex relief notches align 1:1 with razor-sharp per-pixel crack fissures.
 */
function evalDesiccationCracks(u: number, v: number, seedOffset: number): { edgeDist: number; cellId: number } {
  const wu = u + (fbm3(u * 1.35 + seedOffset, v * 1.35, 2.3) - 0.5) * 0.34;
  const wv = v + (fbm3(u * 1.35, v * 1.35 + seedOffset, 7.1) - 0.5) * 0.34;

  const iu = Math.floor(wu);
  const iv = Math.floor(wv);
  let f1 = 1e9;
  let f2 = 1e9;
  let bestId = 0;

  for (let dv = -1; dv <= 1; dv++) {
    for (let du = -1; du <= 1; du++) {
      const cu = iu + du;
      const cv = iv + dv;
      const rx = hash2(cu + seedOffset * 13.7, cv * 1.13 + 4.1);
      const ry = hash2(cu * 1.29 + 9.2, cv + seedOffset * 7.3);
      const px = cu + 0.14 + rx * 0.72;
      const py = cv + 0.14 + ry * 0.72;
      const d = Math.hypot(px - wu, py - wv);
      if (d < f1) {
        f2 = f1;
        f1 = d;
        bestId = hash2(cu * 3.7, cv * 5.3);
      } else if (d < f2) {
        f2 = d;
      }
    }
  }

  return { edgeDist: Math.max(0, f2 - f1), cellId: bestId };
}

/**
 * Verifies outward triangle winding against a reference outward vector.
 */
function pushOutwardQuad(
  indices: number[],
  positions: ArrayLike<number>,
  a: number,
  b: number,
  c: number,
  d: number,
  outX: number,
  outY: number,
  outZ: number
): void {
  const ax = positions[a * 3];
  const ay = positions[a * 3 + 1];
  const az = positions[a * 3 + 2];
  const bx = positions[b * 3];
  const by = positions[b * 3 + 1];
  const bz = positions[b * 3 + 2];
  const dx = positions[d * 3];
  const dy = positions[d * 3 + 1];
  const dz = positions[d * 3 + 2];

  const e1x = bx - ax;
  const e1y = by - ay;
  const e1z = bz - az;
  const e2x = dx - ax;
  const e2y = dy - ay;
  const e2z = dz - az;

  const nx = e1y * e2z - e1z * e2y;
  const ny = e1z * e2x - e1x * e2z;
  const nz = e1x * e2y - e1y * e2x;

  if (nx * outX + ny * outY + nz * outZ >= 0) {
    indices.push(a, b, c, a, c, d);
  } else {
    indices.push(a, c, b, a, d, c);
  }
}

// ---------------------------------------------------------------------------
// 1. Papery Endoperidium Spore Sac & Fluted Pleated Peristome Chimney
// ---------------------------------------------------------------------------
export function buildSporeSacGeometry(): THREE.BufferGeometry {
  const radialSegs = 108;
  const meridianSegs = 84;

  const vertCount = (meridianSegs + 1) * (radialSegs + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  // aSacMeta: x = peristomeMask, y = sulfurDustMask, z = creaseAO, w = throatMask
  const sacMeta = new Float32Array(vertCount * 4);
  const indices: number[] = [];

  for (let j = 0; j <= meridianSegs; j++) {
    const t = j / meridianSegs;

    let baseR = 0;
    let baseY = 0;
    let peristomeMask = 0;
    let throatMask = 0;

    if (t <= 0.67) {
      // Plump, taller papery fig/onion bulb matching Concept A's endoperidium
      const u = t / 0.67; // 0..1
      const phi = u * Math.PI * 0.865;
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);
      // Slightly fuller belly, gentle taper up to peristome neck
      const figProfile = 1.04 - 0.16 * Math.pow(u, 2.0) + 0.085 * Math.sin(u * Math.PI * 1.15);
      baseR = Math.max(0.004, sinP * 0.515 * figProfile);
      baseY = SAC_CENTER_Y - cosP * 0.41 + u * 0.055;
      peristomeMask = smoothstep(0.57, 0.67, t) * 0.35;
      throatMask = 0;
    } else if (t <= 0.84) {
      // Taller fluted pleated conical peristome chimney (+0.09 -> +0.28)
      const u = (t - 0.67) / 0.17;
      const shoulderR = 0.218;
      const rimR = PERISTOME_RIM_RADIUS;
      const curve = Math.pow(u, 0.80);
      baseR = shoulderR + (rimR - shoulderR) * curve;
      baseY = 0.095 + u * (PERISTOME_NOZZLE_Y - 0.095);
      peristomeMask = 0.35 + 0.65 * smoothstep(0.0, 0.52, u);
      throatMask = smoothstep(0.86, 1.0, u) * 0.25;
    } else if (t <= 0.88) {
      // Ragged papery rim rolling over into the dark interior spore chamber
      const u = (t - 0.84) / 0.04;
      const lipAngle = u * Math.PI;
      baseR = PERISTOME_RIM_RADIUS - 0.015 * (1 - Math.cos(lipAngle)) * 0.5 - u * 0.012;
      baseY = PERISTOME_NOZZLE_Y + Math.sin(lipAngle) * 0.014 - u * 0.022;
      peristomeMask = 1.0 - u * 0.18;
      throatMask = 0.25 + u * 0.48;
    } else {
      // Deep shadowed hollow spore throat
      const u = (t - 0.88) / 0.12;
      baseR = Math.max(0.004, (PERISTOME_RIM_RADIUS - 0.026) * (1 - u * 0.88));
      baseY = (PERISTOME_NOZZLE_Y - 0.022) - u * 0.34;
      peristomeMask = (1 - u) * 0.8;
      throatMask = 0.73 + u * 0.27;
    }

    for (let i = 0; i <= radialSegs; i++) {
      const s = i / radialSegs;
      const theta = s * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);

      const nx = cosT;
      const nz = sinT;
      const ny = (baseY - SAC_CENTER_Y) / 0.55;

      // Sculpted tactile parchment vein-creases & diagonal papery buckling folds
      const bodyWeight = smoothstep(0.06, 0.22, t) * (1 - smoothstep(0.62, 0.76, t));
      const warpX = nx * 2.35 + fbm3(nx * 2.1 + 3.1, ny * 2.1, nz * 2.1) * 0.92;
      const warpY = ny * 2.45 + fbm3(nx * 1.8, ny * 1.8 + 5.2, nz * 1.8) * 0.72;
      const warpZ = nz * 2.35 + fbm3(nx * 2.1, ny * 2.1, nz * 2.1 + 8.4) * 0.92;

      const n1 = fbm3(warpX * 1.35, warpY * 0.85, warpZ * 1.35);
      const n2 = fbm3(warpX * 2.85 + 4.1, warpY * 2.1, warpZ * 2.85 - 2.7);
      const ridge1 = 1.0 - Math.abs(n1 * 2.0 - 1.0);
      const ridge2 = 1.0 - Math.abs(n2 * 2.0 - 1.0);
      const creaseVein = Math.pow(ridge1, 2.2) * 0.65 + Math.pow(ridge2, 2.6) * 0.35;

      // Prominent natural tactile parchment indent on right-front cheek (matching Concept A!)
      const cheekDot = Math.max(0, nx * 0.42 + ny * 0.16 + nz * 0.89);
      const cheekIndent = Math.pow(cheekDot, 5.2) * (0.048 + 0.028 * Math.sin(ny * 7.5 + nx * 5.0));

      // Diagonal raised parchment vein running from upper shoulder down-left
      const diagCoord = nx * 2.1 + ny * 1.5 - nz * 0.8;
      const diagVein = Math.exp(-Math.pow((diagCoord - 0.18 + (n1 - 0.5) * 0.6) / 0.24, 2)) * 0.022;

      const macroWarp = (fbm3(nx * 1.2 + 1.4, ny * 1.2, nz * 1.2 + 2.7) - 0.5) * 0.056 * bodyWeight;

      // Crisp pleated/fimbriate Geastrum peristome folds (phase-warped irregular teeth)
      const pleatPhaseWarp = (fbm3(nx * 2.6 + 6.1, 1.3, nz * 2.6 - 3.2) - 0.5) * 1.85;
      const pleatWave = Math.cos(theta * 14.0 + pleatPhaseWarp);
      const pleatSharp = Math.sign(pleatWave) * Math.pow(Math.abs(pleatWave), 0.65);
      const pleatDepth = peristomeMask * (1 - throatMask * 0.5);
      const pleatRadial = pleatSharp * 0.021 * pleatDepth;

      const rimFactor = smoothstep(0.75, 0.84, t) * (1 - smoothstep(0.85, 0.91, t));
      const crownJagged =
        (pleatSharp * 0.028 + (fbm3(nx * 5.4, 2.1, nz * 5.4) - 0.5) * 0.028) * rimFactor;

      const rFinal = Math.max(
        0.002,
        baseR + macroWarp + (creaseVein - 0.35) * 0.044 * bodyWeight + diagVein * bodyWeight - cheekIndent * bodyWeight + pleatRadial
      );
      const yFinal = baseY + crownJagged;

      const idx = j * (radialSegs + 1) + i;
      positions[idx * 3] = cosT * rFinal;
      positions[idx * 3 + 1] = yFinal;
      positions[idx * 3 + 2] = sinT * rFinal;

      uvs[idx * 2] = s;
      uvs[idx * 2 + 1] = t;

      // Concentrated sulfur-gold pollen collar right around the pleated peristome neck
      const collarBand =
        Math.exp(-Math.pow((t - 0.688) / 0.048, 2)) *
        (0.72 + 0.55 * fbm3(nx * 6.5, ny * 6.5, nz * 6.5));
      const lipDust =
        smoothstep(0.79, 0.85, t) *
        (1 - smoothstep(0.87, 0.94, t)) *
        (0.58 + 0.42 * fbm3(nx * 8.0 + 1.2, 3.3, nz * 8.0));
      const sulfurDustMask = Math.min(1, collarBand * 1.25 + lipDust * 0.78);

      const valleyAO = Math.max(0, 0.45 - creaseVein) * 1.55 * bodyWeight + cheekIndent * 4.5 * bodyWeight;
      const pleatValleyAO = Math.max(0, -pleatSharp) * 0.68 * pleatDepth;
      const baseContactAO = (1 - smoothstep(0.05, 0.24, t)) * 0.68;
      const creaseAO = Math.min(1, valleyAO + pleatValleyAO + baseContactAO + throatMask * 0.94);

      sacMeta[idx * 4] = peristomeMask;
      sacMeta[idx * 4 + 1] = sulfurDustMask;
      sacMeta[idx * 4 + 2] = creaseAO;
      sacMeta[idx * 4 + 3] = throatMask;
    }
  }

  const stride = radialSegs + 1;
  for (let j = 0; j < meridianSegs; j++) {
    const tMid = (j + 0.5) / meridianSegs;
    const isThroat = tMid > 0.865;
    for (let i = 0; i < radialSegs; i++) {
      const a = j * stride + i;
      const b = a + 1;
      const c = (j + 1) * stride + (i + 1);
      const d = (j + 1) * stride + i;

      const midTheta = ((i + 0.5) / radialSegs) * Math.PI * 2;
      const outSign = isThroat ? -1 : 1;
      const outX = Math.cos(midTheta) * outSign;
      const outZ = Math.sin(midTheta) * outSign;
      const outY = tMid < 0.15 ? -0.8 : tMid > 0.83 && tMid <= 0.87 ? 0.9 : 0.12 * outSign;
      pushOutwardQuad(indices, positions, a, b, c, d, outX, outY, outZ);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setAttribute('aSacMeta', new THREE.BufferAttribute(sacMeta, 4));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

// ---------------------------------------------------------------------------
// 2. Seven Bold, Thick, Cracked-Crust Recurved Exoperidium Star Petals
// ---------------------------------------------------------------------------
interface PetalSpec {
  angle: number;
  lengthScale: number;
  widthScale: number;
  curlExtra: number;
  twist: number;
  seed: number;
}

export function buildExoperidiumPetalsGeometry(): THREE.BufferGeometry {
  const petalCount = 7;
  const specs: PetalSpec[] = [];
  for (let p = 0; p < petalCount; p++) {
    // Rotate slightly so front-left and front-right rays frame the rootlets cleanly like Concept A
    const baseAngle = (p / petalCount) * Math.PI * 2 + 0.42;
    const angleJitter = (hash2(p * 3.1, 1.7) - 0.5) * 0.13;
    specs.push({
      angle: baseAngle + angleJitter,
      lengthScale: 0.94 + hash2(p * 5.7, 2.3) * 0.15,
      widthScale: 0.93 + hash2(p * 7.3, 4.1) * 0.15,
      curlExtra: (hash2(p * 2.9, 6.7) - 0.5) * 0.22,
      twist: (hash2(p * 4.3, 9.1) - 0.5) * 0.14,
      seed: 2.3 + p * 4.71,
    });
  }

  const uSegs = 56;
  const ringSegs = 26;
  const positions: number[] = [];
  const uvs: number[] = [];
  // aPetalMeta: x = crustMask, y = fleshyEdgeMask, z = crackDepth, w = flexWeight
  const petalMeta: number[] = [];
  // aPetalCoord: x = u (0..1), y = sideShape (-1..1), z = petalSeed, w = rawUp (-1..1)
  const petalCoord: number[] = [];
  const indices: number[] = [];

  for (let p = 0; p < petalCount; p++) {
    const spec = specs[p];
    const cosA = Math.cos(spec.angle);
    const sinA = Math.sin(spec.angle);
    const rightX = -sinA;
    const rightZ = cosA;

    const baseVertexOffset = positions.length / 3;

    const spineR = new Float32Array(uSegs + 1);
    const spineY = new Float32Array(uSegs + 1);

    for (let iu = 0; iu <= uSegs; iu++) {
      const u = iu / uSegs;
      const L = spec.lengthScale;

      // Bold ~140° arched star-claw curve matching Concept A (NOT closed thin hoops!):
      // Emerges from dark basal cup tucked cleanly under the sac (r=0.22, y=-0.64),
      // rises outward to thick shoulder arch hugging the lower sac (r=0.66, y=-0.42),
      // bends firmly downward (r=1.08, y=-0.66), and hooks subtly inward at pointed claw tip (r=0.84, y=-0.92).
      let r: number;
      let y: number;
      if (u < 0.24) {
        const k = u / 0.24;
        const sk = k * k * (3 - 2 * k);
        r = 0.22 + sk * 0.44 * L;
        y = -0.64 + Math.sin(k * Math.PI * 0.5) * 0.20;
      } else {
        const k = (u - 0.24) / 0.76;
        // Sweeps ~142° down and gently back inward into sturdy leathery feet
        const arcRadR = (0.44 + 0.08 * Math.sin(k * Math.PI)) * L;
        const arcRadY = (0.37 + 0.05 * L);
        const thetaArc = Math.PI * 0.56 - k * (Math.PI * 0.92 + spec.curlExtra * 0.45);
        r = 0.66 * L + Math.cos(thetaArc) * arcRadR - k * k * 0.08 * L;
        y = -0.80 + Math.sin(thetaArc) * arcRadY - k * 0.06;
      }

      const undul = (fbm3(u * 2.6 + spec.seed, p * 1.3, 0.7) - 0.5) * 0.028 * u;
      spineR[iu] = r + undul;
      spineY[iu] = y + (fbm3(u * 2.1, spec.seed, 3.9) - 0.5) * 0.022 * u;
    }

    for (let iu = 0; iu <= uSegs; iu++) {
      const u = iu / uSegs;
      const prevIdx = Math.max(0, iu - 1);
      const nextIdx = Math.min(uSegs, iu + 1);
      const dr = spineR[nextIdx] - spineR[prevIdx];
      const dy = spineY[nextIdx] - spineY[prevIdx];
      const tLen = Math.hypot(dr, dy) || 1;
      const tr = dr / tLen;
      const ty = dy / tLen;
      const nr = -ty;
      const ny = tr;

      // Sturdy fleshy triangular-lanceolate petal width & stout slab thickness
      const baseShoulder = smoothstep(0.0, 0.16, u);
      const tipTaper = Math.pow(Math.max(0.015, 1.0 - Math.pow(u, 1.65)), 0.72);
      const halfW = (0.145 + 0.145 * baseShoulder) * tipTaper * spec.widthScale;
      const halfH = (0.086 - 0.044 * Math.pow(u, 1.15)) * (0.42 + 0.58 * tipTaper);

      const twistAngle = spec.twist * Math.sin(u * Math.PI);
      const cosTw = Math.cos(twistAngle);
      const sinTw = Math.sin(twistAngle);

      for (let ir = 0; ir <= ringSegs; ir++) {
        const s = ir / ringSegs;
        const phi = s * Math.PI * 2;
        const rawSide = Math.sin(phi);
        const rawUp = Math.cos(phi);

        // Chiseled slab cross-section: flat-topped crusty slab, thick beveled Burnt Sienna side rims, stout dark underside
        const sideShape = Math.sign(rawSide) * Math.pow(Math.abs(rawSide), 0.68);
        const upShape = rawUp >= 0
          ? Math.pow(rawUp, 0.48) * 0.94 + 0.08 * (1 - sideShape * sideShape)
          : -Math.pow(-rawUp, 0.78) * 0.95;

        const crustMask =
          smoothstep(0.14, 0.48, rawUp) *
          smoothstep(0.05, 0.15, u) *
          (1 - smoothstep(0.88, 0.98, u));
        const edgeBand = Math.exp(-Math.pow((rawUp - 0.02) / 0.34, 2));

        // Vertex-level crack notch matching the exact 1:1 isometric Voronoi domain evaluated per-pixel in TSL!
        const vor = evalDesiccationCracks(u * 4.8, sideShape * 0.88, spec.seed);
        const topOnlyMask = smoothstep(0.42, 0.72, rawUp) * crustMask;
        const crackDepth = (1.0 - smoothstep(0.02, 0.11, vor.edgeDist)) * topOnlyMask;

        const plateLift = (vor.cellId - 0.45) * 0.011 * topOnlyMask * (1 - crackDepth);
        const crackIndent = crackDepth * (0.015 * (0.45 + 0.55 * tipTaper));
        const microBark = (fbm3(u * 9.0 + spec.seed, sideShape * 4.2, rawUp * 3.0) - 0.5) * 0.011;

        const localSide = sideShape * halfW;
        const localUp = upShape * halfH + plateLift - crackIndent + microBark;

        const rotSide = localSide * cosTw - localUp * sinTw;
        const rotUp = localSide * sinTw + localUp * cosTw;

        const rPos = spineR[iu] + nr * rotUp;
        const yPos = spineY[iu] + ny * rotUp;
        const wx = cosA * rPos + rightX * rotSide;
        const wy = yPos;
        const wz = sinA * rPos + rightZ * rotSide;

        positions.push(wx, wy, wz);
        uvs.push(u, s);

        const fleshyEdgeMask = Math.min(1.0, edgeBand * 0.95 + crackDepth * 0.85);
        const flexWeight = Math.pow(u, 1.45);
        petalMeta.push(crustMask, fleshyEdgeMask, crackDepth, flexWeight);
        petalCoord.push(u, sideShape, spec.seed, rawUp);
      }
    }

    const stride = ringSegs + 1;
    for (let iu = 0; iu < uSegs; iu++) {
      const nextIdx = Math.min(uSegs, iu + 1);
      const prevIdx = Math.max(0, iu - 1);
      const dr = spineR[nextIdx] - spineR[prevIdx];
      const dy = spineY[nextIdx] - spineY[prevIdx];
      const tLen = Math.hypot(dr, dy) || 1;
      const nr = -dy / tLen;
      const ny = dr / tLen;

      for (let ir = 0; ir < ringSegs; ir++) {
        const a = baseVertexOffset + iu * stride + ir;
        const b = baseVertexOffset + (iu + 1) * stride + ir;
        const c = baseVertexOffset + (iu + 1) * stride + (ir + 1);
        const d = baseVertexOffset + iu * stride + (ir + 1);

        const phiMid = ((ir + 0.5) / ringSegs) * Math.PI * 2;
        const sOut = Math.sin(phiMid);
        const uOut = Math.cos(phiMid);
        const outX = cosA * (nr * uOut) + rightX * sOut;
        const outY = ny * uOut;
        const outZ = sinA * (nr * uOut) + rightZ * sOut;
        pushOutwardQuad(indices, positions, a, b, c, d, outX, outY, outZ);
      }
    }
  }

  // Snug dark basal cup tucked cleanly underneath the petal junction (zero protruding rim belt!)
  const cupBaseOffset = positions.length / 3;
  const cupRadSegs = 44;
  const cupRingSegs = 10;
  for (let j = 0; j <= cupRingSegs; j++) {
    const t = j / cupRingSegs;
    const phi = t * Math.PI * 0.56;
    const r = Math.max(0.005, Math.sin(phi) * 0.44);
    const y = -0.74 + (1 - Math.cos(phi)) * 0.22;
    for (let i = 0; i <= cupRadSegs; i++) {
      const s = i / cupRadSegs;
      const theta = s * Math.PI * 2;
      const rough = (fbm3(Math.cos(theta) * 3.5, y * 4.5, Math.sin(theta) * 3.5) - 0.5) * 0.015;
      positions.push(Math.cos(theta) * (r + rough), y + rough * 0.5, Math.sin(theta) * (r + rough));
      uvs.push(s, t);
      petalMeta.push(0.0, 0.12 * t, 0.25, 0.04 * t);
      petalCoord.push(0.0, 0.0, 1.0, -1.0);
    }
  }
  const cupStride = cupRadSegs + 1;
  for (let j = 0; j < cupRingSegs; j++) {
    for (let i = 0; i < cupRadSegs; i++) {
      const a = cupBaseOffset + j * cupStride + i;
      const b = a + 1;
      const c = cupBaseOffset + (j + 1) * cupStride + (i + 1);
      const d = cupBaseOffset + (j + 1) * cupStride + i;
      const thetaMid = ((i + 0.5) / cupRadSegs) * Math.PI * 2;
      pushOutwardQuad(indices, positions, a, b, c, d, Math.cos(thetaMid), -0.68, Math.sin(thetaMid));
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geom.setAttribute('aPetalMeta', new THREE.Float32BufferAttribute(petalMeta, 4));
  geom.setAttribute('aPetalCoord', new THREE.Float32BufferAttribute(petalCoord, 4));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

// ---------------------------------------------------------------------------
// 3. Gnarled Hanging Mycelial Rootlets & Subtle Shadowy Underside Moss Tuft
// ---------------------------------------------------------------------------
export function buildMycelialRootletsGeometry(): THREE.BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];
  const rootMeta: number[] = [];
  const indices: number[] = [];

  const appendTaperedTube = (
    pts: THREE.Vector3[],
    rStart: number,
    rEnd: number,
    radialSegs: number,
    seed: number,
    swayBase: number
  ) => {
    const baseOff = positions.length / 3;
    const lenSegs = pts.length - 1;
    const tangent = new THREE.Vector3();
    const normal = new THREE.Vector3();
    const binormal = new THREE.Vector3();
    const refUp = new THREE.Vector3(1, 0, 0);

    for (let j = 0; j <= lenSegs; j++) {
      const t = j / lenSegs;
      const p = pts[j];
      const pPrev = pts[Math.max(0, j - 1)];
      const pNext = pts[Math.min(lenSegs, j + 1)];
      tangent.subVectors(pNext, pPrev).normalize();
      if (Math.abs(tangent.dot(refUp)) > 0.9) refUp.set(0, 0, 1);
      else refUp.set(1, 0, 0);
      normal.crossVectors(tangent, refUp).normalize();
      binormal.crossVectors(tangent, normal).normalize();

      const radius = Math.max(0.0014, rStart + (rEnd - rStart) * Math.pow(t, 0.78));
      for (let i = 0; i <= radialSegs; i++) {
        const phi = (i / radialSegs) * Math.PI * 2;
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        const ox = normal.x * c + binormal.x * s;
        const oy = normal.y * c + binormal.y * s;
        const oz = normal.z * c + binormal.z * s;
        positions.push(p.x + ox * radius, p.y + oy * radius, p.z + oz * radius);
        uvs.push(i / radialSegs, t);
        rootMeta.push(t, 0.0, seed, swayBase + t * 0.85);
      }
    }

    const stride = radialSegs + 1;
    for (let j = 0; j < lenSegs; j++) {
      const pMid = pts[j];
      for (let i = 0; i < radialSegs; i++) {
        const a = baseOff + j * stride + i;
        const b = baseOff + (j + 1) * stride + i;
        const c = baseOff + (j + 1) * stride + (i + 1);
        const d = baseOff + j * stride + (i + 1);
        const outX = positions[a * 3] - pMid.x;
        const outY = positions[a * 3 + 1] - pMid.y;
        const outZ = positions[a * 3 + 2] - pMid.z;
        pushOutwardQuad(indices, positions, a, b, c, d, outX, outY, outZ);
      }
    }
  };

  // 24 gnarled, twisted Raw Umber root tendrils + fine curling side filaments
  const rootCount = 24;
  for (let r = 0; r < rootCount; r++) {
    const seed = hash2(r * 2.31, 7.17);
    const angle = (r / rootCount) * Math.PI * 2 + (seed - 0.5) * 0.45;
    const startR = 0.015 + hash2(r * 4.1, 1.9) * 0.11;
    const totalLen = 0.18 + hash2(r * 6.7, 3.3) * 0.24;
    const spread = 0.24 + hash2(r * 8.3, 5.1) * 0.30;
    const segs = 12;
    const pts: THREE.Vector3[] = [];

    let x = Math.cos(angle) * startR - 0.02;
    let y = -0.71;
    let z = Math.sin(angle) * startR;
    pts.push(new THREE.Vector3(x, y, z));

    for (let k = 1; k <= segs; k++) {
      const f = k / segs;
      const stepL = totalLen / segs;
      // Stronger organic root zig-zag gnarl (matching Concept A's fibrous tendrils!)
      const kinkX = (fbm3(r * 3.1 + f * 5.2, 1.2, 4.4) - 0.5) * 0.072;
      const kinkZ = (fbm3(4.4, r * 3.1 + f * 5.2, 1.2) - 0.5) * 0.072;
      x += Math.cos(angle + f * 0.8) * stepL * spread + kinkX;
      y -= stepL * (0.92 - f * 0.16);
      z += Math.sin(angle + f * 0.8) * stepL * spread + kinkZ;
      pts.push(new THREE.Vector3(x, y, z));
    }

    const rBase = 0.0068 + hash2(r * 1.9, 8.7) * 0.0065;
    appendTaperedTube(pts, rBase, 0.0014, 5, seed, 0.1);

    // Frequent curling side twigs
    if (r % 3 !== 0) {
      const forkIdx = 3 + (r % 5);
      const forkOrigin = pts[forkIdx].clone();
      const forkAngle = angle + (seed > 0.5 ? 0.85 : -0.85);
      const forkPts: THREE.Vector3[] = [forkOrigin];
      let fx = forkOrigin.x;
      let fy = forkOrigin.y;
      let fz = forkOrigin.z;
      for (let k = 1; k <= 6; k++) {
        fx += Math.cos(forkAngle) * 0.022 + (hash2(r + k * 1.7, 2.2) - 0.5) * 0.026;
        fy -= 0.018;
        fz += Math.sin(forkAngle) * 0.022 + (hash2(2.2, r + k * 1.7) - 0.5) * 0.026;
        forkPts.push(new THREE.Vector3(fx, fy, fz));
      }
      appendTaperedTube(forkPts, rBase * 0.5, 0.0011, 4, seed * 0.7, 0.35);
    }
  }

  // Subtle shadowy dark Moss Umbrous peat cluster nestled to the right side of the rootlets
  const mossClumpCount = 14;
  const icoBase = new THREE.IcosahedronGeometry(1, 2);
  const icoPos = icoBase.attributes.position;
  const icoIdx = icoBase.index;

  for (let m = 0; m < mossClumpCount; m++) {
    const seed = hash2(m * 5.13, 11.7);
    const angle = -0.3 + (m / mossClumpCount) * 1.45 + (seed - 0.5) * 0.35;
    const rad = 0.04 + hash2(m * 3.7, 2.9) * 0.10;
    const cx = Math.cos(angle) * rad + 0.06;
    const cy = -0.75 - hash2(m * 7.1, 1.3) * 0.13;
    const cz = Math.sin(angle) * rad + 0.01;
    const clumpR = 0.018 + hash2(m * 2.3, 6.1) * 0.022;

    const baseOff = positions.length / 3;
    for (let v = 0; v < icoPos.count; v++) {
      const vx = icoPos.getX(v);
      const vy = icoPos.getY(v);
      const vz = icoPos.getZ(v);
      const bump = 0.62 + fbm3(vx * 5.2 + m * 2.7, vy * 5.2, vz * 5.2) * 0.78;
      positions.push(cx + vx * clumpR * bump, cy + vy * clumpR * 1.1 * bump, cz + vz * clumpR * bump);
      uvs.push(0.5 + vx * 0.5, 0.5 + vz * 0.5);
      rootMeta.push(0.5, 1.0, seed, 0.15);
    }
    if (icoIdx) {
      for (let k = 0; k < icoIdx.count; k++) {
        indices.push(baseOff + icoIdx.getX(k));
      }
    } else {
      for (let k = 0; k < icoPos.count; k++) {
        indices.push(baseOff + k);
      }
    }
  }
  icoBase.dispose();

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geom.setAttribute('aRootMeta', new THREE.Float32BufferAttribute(rootMeta, 4));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}
