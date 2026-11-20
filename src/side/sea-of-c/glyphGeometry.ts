import * as THREE from 'three/webgpu';

export interface BeveledCOptions {
  outerRadius?: number;
  innerRadius?: number;
  /** Total mouth opening angle on the +X side in degrees (default 76 => ±38°). */
  openingDeg?: number;
  depth?: number;
  bevelThickness?: number;
  bevelSize?: number;
  bevelSegments?: number;
  curveSegments?: number;
}

const DEFAULT_HERO_C: Required<BeveledCOptions> = {
  outerRadius: 0.5,
  innerRadius: 0.275,
  openingDeg: 78,
  depth: 0.115,
  bevelThickness: 0.025,
  bevelSize: 0.024,
  bevelSegments: 1,
  curveSegments: 7,
};

/**
 * Welds coincident vertices within `1e-4` into an indexed BufferGeometry so
 * `computeVertexNormals()` produces continuous, glossy sea-glass bevel shading
 * instead of faceted per-triangle normals, while cutting vertex count ~4x.
 */
function weldToIndexedGeometry(raw: THREE.BufferGeometry): THREE.BufferGeometry {
  const posAttr = raw.getAttribute('position');
  const uvAttr = raw.getAttribute('uv');
  const srcIndex = raw.getIndex();
  const vertCount = srcIndex ? srcIndex.count : posAttr.count;

  const keyToNewIndex = new Map<string, number>();
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const quant = 10000;
  const mapVertex = (srcIdx: number): number => {
    const x = posAttr.getX(srcIdx);
    const y = posAttr.getY(srcIdx);
    const z = posAttr.getZ(srcIdx);
    const key = `${Math.round(x * quant)},${Math.round(y * quant)},${Math.round(z * quant)}`;
    const existing = keyToNewIndex.get(key);
    if (existing !== undefined) return existing;

    const newIdx = positions.length / 3;
    keyToNewIndex.set(key, newIdx);
    positions.push(x, y, z);
    if (uvAttr) {
      uvs.push(uvAttr.getX(srcIdx), uvAttr.getY(srcIdx));
    } else {
      uvs.push(x + 0.5, y + 0.5);
    }
    return newIdx;
  };

  for (let i = 0; i + 2 < vertCount; i += 3) {
    const a = mapVertex(srcIndex ? srcIndex.getX(i) : i);
    const b = mapVertex(srcIndex ? srcIndex.getX(i + 1) : i + 1);
    const c = mapVertex(srcIndex ? srcIndex.getX(i + 2) : i + 2);
    if (a !== b && b !== c && c !== a) {
      indices.push(a, b, c);
    }
  }

  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  out.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  out.setIndex(indices);
  return out;
}

/**
 * Computes signed volume of a closed triangle mesh (`> 0` proves strictly outward winding).
 */
export function computeSignedVolume(geom: THREE.BufferGeometry): number {
  const pos = geom.getAttribute('position');
  if (!pos) return 0;
  const index = geom.getIndex();
  const triCount = index ? Math.floor(index.count / 3) : Math.floor(pos.count / 3);
  let vol6 = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = index ? index.getX(t * 3) : t * 3;
    const i1 = index ? index.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = index ? index.getX(t * 3 + 2) : t * 3 + 2;

    const ax = pos.getX(i0);
    const ay = pos.getY(i0);
    const az = pos.getZ(i0);
    const bx = pos.getX(i1);
    const by = pos.getY(i1);
    const bz = pos.getZ(i1);
    const cx = pos.getX(i2);
    const cy = pos.getY(i2);
    const cz = pos.getZ(i2);

    vol6 +=
      ax * (by * cz - bz * cy) +
      ay * (bz * cx - bx * cz) +
      az * (bx * cy - by * cx);
  }

  return vol6 / 6;
}

/**
 * Builds a clean modern geometric sans-serif capital 'C' BufferGeometry with
 * chamfered/rounded bevels, centered strictly at (0, 0, 0), normalized to unit
 * diameter (~1.0), smooth welded vertex normals, and guaranteed outward winding.
 */
export function createBeveledCGeometry(options?: BeveledCOptions): THREE.BufferGeometry {
  const outerRadius = Math.max(0.15, options?.outerRadius ?? DEFAULT_HERO_C.outerRadius);
  const innerRadius = Math.max(
    0.04,
    Math.min(outerRadius * 0.82, options?.innerRadius ?? DEFAULT_HERO_C.innerRadius)
  );
  const openingDeg = Math.max(20, Math.min(140, options?.openingDeg ?? DEFAULT_HERO_C.openingDeg));
  const depth = Math.max(0.02, options?.depth ?? DEFAULT_HERO_C.depth);
  const strokeWidth = outerRadius - innerRadius;
  const bevelSize = Math.min(
    strokeWidth * 0.34,
    Math.max(0.004, options?.bevelSize ?? DEFAULT_HERO_C.bevelSize)
  );
  const bevelThickness = Math.max(0.004, options?.bevelThickness ?? DEFAULT_HERO_C.bevelThickness);
  const bevelSegments = Math.max(1, Math.round(options?.bevelSegments ?? DEFAULT_HERO_C.bevelSegments));
  const curveSegments = Math.max(4, Math.round(options?.curveSegments ?? DEFAULT_HERO_C.curveSegments));

  const halfGap = ((openingDeg * 0.5) * Math.PI) / 180;

  // Counter-clockwise outer arc -> radial lower terminal -> clockwise inner arc -> radial upper terminal
  const shape = new THREE.Shape();
  shape.moveTo(Math.cos(halfGap) * outerRadius, Math.sin(halfGap) * outerRadius);
  shape.absarc(0, 0, outerRadius, halfGap, Math.PI * 2 - halfGap, false);
  shape.lineTo(Math.cos(-halfGap) * innerRadius, Math.sin(-halfGap) * innerRadius);
  shape.absarc(0, 0, innerRadius, Math.PI * 2 - halfGap, halfGap, true);
  shape.closePath();

  const raw = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelThickness,
    bevelSize,
    bevelOffset: -bevelSize,
    bevelSegments,
    curveSegments,
  });

  const geom = weldToIndexedGeometry(raw);
  raw.dispose();

  // Center strictly at (0, 0, 0) and normalize to unit major diameter (1.0)
  geom.computeBoundingBox();
  if (geom.boundingBox) {
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    geom.boundingBox.getCenter(center);
    geom.boundingBox.getSize(size);
    geom.translate(-center.x, -center.y, -center.z);

    const majorDiameter = Math.max(size.x, size.y);
    if (majorDiameter > 1e-4 && Math.abs(majorDiameter - 1.0) > 1e-4) {
      const invScale = 1.0 / majorDiameter;
      geom.scale(invScale, invScale, invScale);
    }
  }

  // Verify outward winding; flip triangle winding defensively if ever inverted
  if (computeSignedVolume(geom) < 0) {
    const idx = geom.getIndex();
    if (idx) {
      for (let i = 0; i + 2 < idx.count; i += 3) {
        const b = idx.getX(i + 1);
        const c = idx.getX(i + 2);
        idx.setX(i + 1, c);
        idx.setX(i + 2, b);
      }
      idx.needsUpdate = true;
    }
  }

  geom.computeVertexNormals();
  geom.computeBoundingBox();
  geom.computeBoundingSphere();
  return geom;
}

/**
 * Lightweight chamfered 3D 'c' geometry optimized for thousands of airborne breaking-lip
 * and pointer-wake foam spray droplets (~110–180 tris, strictly outward-wound).
 */
export function createSprayCGeometry(): THREE.BufferGeometry {
  return createBeveledCGeometry({
    outerRadius: 0.5,
    innerRadius: 0.265,
    openingDeg: 82,
    depth: 0.13,
    bevelThickness: 0.025,
    bevelSize: 0.022,
    bevelSegments: 1,
    curveSegments: 4,
  });
}

/**
 * Builds a twilight marine PMREM environment texture for glossy beveled sea-glass 'C's:
 * - Moody deep slate-indigo overhead dome
 * - Restrained warm amber-gold horizon sunset strip (scaled ~0.62x so dark sapphire
 *   shadows remain deep and crisp while beveled rims catch clean specular highlights)
 * - Cool emerald-turquoise lower ocean bounce strip illuminating glass undersides
 * - Subtle flanking twilight rim fills for 360° chamfer catchlights
 */
export function createMarineEnvMap(renderer: THREE.WebGPURenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x060d18);

  const disposables: Array<THREE.BufferGeometry | THREE.Material> = [];
  const addPanel = (
    w: number,
    h: number,
    color: [number, number, number],
    pos: [number, number, number],
    lookTarget: [number, number, number] = [0, 0, 0]
  ) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      side: THREE.DoubleSide,
    });
    disposables.push(geo, mat);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.lookAt(lookTarget[0], lookTarget[1], lookTarget[2]);
    envScene.add(mesh);
  };

  // Moody twilight slate-indigo overhead sky softbox (0.62x calibrated)
  addPanel(24, 24, [0.10, 0.16, 0.30], [0, 9.5, 0], [0, 0, 0]);

  // Broad warm amber sunset band hugging the -Z horizon
  addPanel(28, 3.6, [0.86, 0.42, 0.14], [0, 1.0, -10.5], [0, 0.6, 0]);

  // Focused champagne-gold solar core reflection strip (capped <= 1.3x linear)
  addPanel(5.2, 2.2, [1.28, 0.97, 0.45], [0, 1.15, -10.0], [0, 0.4, 0]);

  // Restrained emerald-sapphire lower ocean bounce illuminating glass 'C' undersides without mint wash
  addPanel(26, 26, [0.012, 0.22, 0.26], [0, -7.2, 0], [0, 0, 0]);

  // Flanking twilight steel-cyan and warm dusk rim strips for 360° chamfer catchlights
  addPanel(14, 5.5, [0.14, 0.35, 0.48], [-10.0, 2.2, 1.5], [0, 0, 0]);
  addPanel(14, 5.5, [0.47, 0.29, 0.21], [10.0, 1.8, -1.5], [0, 0, 0]);
  addPanel(16, 4.5, [0.09, 0.19, 0.32], [0, 2.4, 10.5], [0, 0, 0]);

  const rt = pmrem.fromScene(envScene, 0.035);
  pmrem.dispose();
  for (const d of disposables) {
    d.dispose();
  }

  return rt.texture;
}
