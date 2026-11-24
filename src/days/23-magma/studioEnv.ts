import * as THREE from 'three/webgpu';

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

/**
 * Generates a smooth radially feathered studio softbox scrim texture so metallic
 * reflections on machined iron, threaded steel, satin brass, quartz glass, and
 * glossy black obsidian roll smoothly without hard rectangular card artifacts.
 */
function createFeatheredSoftboxTexture(
  r: number,
  g: number,
  b: number
): THREE.DataTexture {
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
 * Builds a photographer's studio softbox PMREM environment map tailored for
 * Day 23's machined dark-iron press frame, turned steel threads, satin-brass
 * pressure collars, beveled quartz cutaway windows, and glossy cooled obsidian.
 */
export function createMagmaStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Calibrated slate-grey studio enclosure prevents forged iron & gunmetal bevels from collapsing black
  envScene.background = new THREE.Color(0x161b21);

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

  // 1. Primary overhead-left warm calcite studio softbox (catches press chamfers, brass handwheels & obsidian lobes)
  addSoftbox(11.5, 9.0, [-5.2, 5.6, 4.4], [0, 0, 0], 1.0, 0.96, 0.90, 2.05);

  // 2. Right tall side scrim (delineates helical ACME screw threads, brass Bourdon gauge & basalt facets)
  addSoftbox(8.8, 10.8, [6.2, 2.8, 3.4], [0, 0, 0], 0.98, 0.95, 0.91, 1.80);

  // 3. Front-high narrow studio strip light (crisp horizontal metallic catchlight along top crosshead & quartz bezel)
  addSoftbox(11.0, 2.4, [0.4, 4.2, 6.2], [0, 0, 0], 0.96, 0.97, 1.0, 1.45);

  // 4. Rear-top cool mineral rim scrim (separates dark press frame & basalt crown from #212529 Chalky Slate)
  addSoftbox(12.5, 5.2, [-0.6, 5.2, -5.8], [0, 0, 0], 0.90, 0.95, 1.0, 1.65);

  // 5. Overhead zenith soft fill (gives glossy cooled obsidian pillows silky zenith catchlights)
  addSoftbox(9.8, 9.8, [0.0, 7.4, 0.6], [0, 0, 0], 0.97, 0.95, 0.92, 1.25);

  // 6. Rear-right cool fill (keeps back quartz window, rear tie-rods & brass hardware legible on 360 orbit)
  addSoftbox(8.4, 6.8, [4.8, 1.8, -5.6], [0, 0, 0], 0.89, 0.94, 1.0, 1.22);

  // 7. Subtle warm studio floor bounce (lifts underside of hydraulic cylinder, base platen & brass T-handle)
  addSoftbox(12.5, 12.5, [0, -6.0, 0.8], [0, 0, 0], 0.98, 0.89, 0.80, 0.52);

  const renderTarget = pmrem.fromScene(envScene, 0.068);

  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}
