import * as THREE from 'three/webgpu';

export interface StudioEnvironmentHandle {
  readonly texture: THREE.Texture;
  readonly dispose: () => void;
}

function createFeatheredSoftboxTexture(r: number, g: number, b: number): THREE.DataTexture {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = ((y + 0.5) / size) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = ((x + 0.5) / size) * 2 - 1;
      const d = Math.hypot(nx, ny);
      const falloff = Math.max(0, Math.min(1, 1 - d * d));
      const soft = falloff * falloff;
      const idx = (y * size + x) * 4;
      data[idx] = Math.min(255, Math.round(r * soft * 255));
      data[idx + 1] = Math.min(255, Math.round(g * soft * 255));
      data[idx + 2] = Math.min(255, Math.round(b * soft * 255));
      data[idx + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Calibrated architectural museum-gallery softbox PMREM environment cubemap.
 * Gives satin-brass optical sight-ring specular continuity and subtle gallery
 * ceiling bounce on honed travertine and board-formed concrete bevels without
 * collapsing high-metalness surfaces into the dark #212529 Chalky Slate void.
 */
export function createAnamorphStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x13171c);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: [number, number, number],
    lookAt: [number, number, number],
    r: number,
    g: number,
    b: number,
    gain: number
  ) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const tex = createFeatheredSoftboxTexture(r, g, b);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      color: new THREE.Color().setRGB(gain, gain, gain),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.lookAt(lookAt[0], lookAt[1], lookAt[2]);
    envScene.add(mesh);
    disposables.push(geo, tex, mat);
  };

  // Overhead warm museum skylight diffuser
  addSoftbox(12.0, 10.0, [2.2, 6.4, 3.8], [0, 0, 0], 1.0, 0.96, 0.90, 1.55);
  // Front-right warm gallery key bounce
  addSoftbox(8.5, 9.0, [6.2, 2.4, 5.2], [0, 0, 0], 0.99, 0.95, 0.89, 1.38);
  // Cool slate left-flank gallery fill (separates concrete shadow planes)
  addSoftbox(9.5, 8.5, [-6.6, 1.6, 2.4], [0, 0, 0], 0.84, 0.91, 1.0, 1.15);
  // Crisp back-top rim strip for architectural contour separation against #212529
  addSoftbox(12.5, 5.5, [-1.4, 5.2, -5.8], [0, 0, 0], 0.94, 0.97, 1.0, 1.32);
  // Soft rear-right orbit fill so 360-degree views stay balanced
  addSoftbox(8.5, 7.5, [5.6, 1.2, -5.4], [0, 0, 0], 0.96, 0.93, 0.89, 0.95);
  // Subtle warm plinth under-bounce
  addSoftbox(12.0, 12.0, [0, -6.4, 0.8], [0, 0, 0], 0.95, 0.91, 0.86, 0.36);

  const renderTarget = pmrem.fromScene(envScene, 0.065);

  for (const d of disposables) d.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}
