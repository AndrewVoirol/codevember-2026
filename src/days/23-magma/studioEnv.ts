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
  // Deep negative-fill studio enclosure preserves crisp dark-iron & obsidian contrast
  envScene.background = new THREE.Color(0x0a0d11);

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

  // 1. Primary overhead-left warm calcite studio softbox (catches brass handwheels, obsidian lobes & press beams)
  addSoftbox(11.0, 8.8, [-5.4, 5.8, 4.2], [0, 0, 0], 1.0, 0.96, 0.90, 1.85);

  // 2. Right tall side scrim (delineates right lateral screw threads, brass collar & basalt fracture steps)
  addSoftbox(8.5, 10.5, [6.4, 3.0, 3.2], [0, 0, 0], 0.98, 0.95, 0.90, 1.65);

  // 3. Rear-top cool mineral rim scrim (separates dark press frame & basalt crown from #212529 Chalky Slate)
  addSoftbox(12.5, 5.0, [-0.6, 5.2, -5.8], [0, 0, 0], 0.90, 0.95, 1.0, 1.50);

  // 4. Overhead zenith soft fill (gives glossy cooled obsidian pillows crisp zenith catchlights)
  addSoftbox(9.5, 9.5, [0.0, 7.4, 0.6], [0, 0, 0], 0.97, 0.95, 0.91, 1.05);

  // 5. Rear-right subtle cool fill (keeps back cutaway window & rear tie-rods legible on 360 orbit)
  addSoftbox(8.0, 6.5, [4.8, 1.8, -5.6], [0, 0, 0], 0.89, 0.94, 1.0, 1.08);

  // 6. Subtle warm studio floor bounce (lifts underside of hydraulic cylinder & brass piston T-handle)
  addSoftbox(12.0, 12.0, [0, -6.2, 0.5], [0, 0, 0], 0.98, 0.88, 0.78, 0.36);

  const renderTarget = pmrem.fromScene(envScene, 0.082);

  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}
