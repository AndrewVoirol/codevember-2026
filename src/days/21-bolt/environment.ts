import * as THREE from 'three/webgpu';

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

/**
 * Generates a radial-feathered softbox luminance map so specular highlights
 * wrap smoothly across brushed brass, blued steel, and phosphor bronze bevels
 * without harsh rectangular studio-card edges.
 */
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
 * Builds a calibrated horologist's studio softbox PMREM environment cubemap.
 * Ensures high-metalness satin brass, blued steel, phosphor bronze, and gunmetal
 * never collapse into the dark #212529 Chalky Slate void.
 */
export function createBoltStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  // Calibrated horological dark-slate negative fill preserves crisp metallic edge contrast without black collapse
  envScene.background = new THREE.Color(0x121720);

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

  // 1. Broad overhead-front-left warm calcite key softbox (sweeps silky reflections across lathe-turned gears, levers & oak)
  addSoftbox(11.0, 8.2, [-3.6, 5.4, 5.2], [0, 0, 0], 1.0, 0.96, 0.89, 2.02);

  // 2. Tall right-front warm-neutral strip scrim (illuminates forend faceplate, blued deadbolt & right gear train)
  addSoftbox(7.4, 11.2, [6.4, 2.0, 4.5], [0, 0, 0], 0.99, 0.96, 0.91, 1.92);

  // 3. Narrow vertical high-contrast horological highlight ribbon (makes concentric gear lathe turning & bevels sparkle!)
  addSoftbox(2.8, 10.5, [2.2, 4.2, 6.2], [0, 0, 0], 1.0, 0.98, 0.94, 2.15);

  // 4. Front-center soft fill ring card (keeps front faces of stacked brass levers, gears & escutcheon luminous)
  addSoftbox(9.2, 7.2, [0.4, 1.2, 7.8], [0, 0, 0], 0.97, 0.95, 0.92, 1.22);

  // 5. Cool mineral slate left flank fill (brings out Prussian blue oxide in thermally blued steel & gunmetal case)
  addSoftbox(8.5, 8.0, [-6.8, 1.2, 1.8], [0, 0, 0], 0.82, 0.90, 1.0, 1.44);

  // 6. Rear-top grazing rim softbox (carves crisp contour separation against #212529 Chalky Slate)
  addSoftbox(12.5, 5.2, [0.8, 5.8, -5.6], [0, 0, 0], 0.92, 0.96, 1.0, 1.54);

  // 7. Rear-right cool-warm bounce (lights rear of charred oak and forend bevel during 360 orbits)
  addSoftbox(8.0, 7.0, [5.4, 0.8, -5.8], [0, 0, 0], 0.95, 0.92, 0.88, 1.14);

  // 8. Rear-left cool calcite fill (illuminates rear oak bevels and left flank without dark collapse)
  addSoftbox(8.5, 7.2, [-5.6, 1.8, -5.2], [0, 0, 0], 0.90, 0.94, 0.99, 1.02);

  // 9. Warm workbench under-bounce (lifts key bow underside, lower gear train & forend bevel)
  addSoftbox(12.0, 12.0, [0, -6.2, 1.2], [0, 0, 0], 0.96, 0.91, 0.84, 0.54);

  // Sharper 0.048 museum pre-blur preserves crisp lathe-turned gear reflections and chamfered steel glints
  const renderTarget = pmrem.fromScene(envScene, 0.048);

  for (const d of disposables) d.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}
