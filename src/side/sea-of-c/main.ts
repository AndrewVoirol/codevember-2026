import './style.css';
import * as THREE from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createWaveFieldUniforms, type SeaStateId } from './waveField';
import { createMarineEnvMap } from './glyphGeometry';
import { createSkyDome } from './skyDome';
import { createOceanSurface } from './oceanSurface';
import { createGranularFluid } from './granularFluid';
import {
  SeaOfCController,
  type InteractionTool,
  type OceanRayHit,
} from './interaction';

interface SeaOfCQaApi {
  isReady: boolean;
  frameCount: number;
  avgFrameMs: number;
  p95FrameMs: number;
  setSeaState: (state: SeaStateId, immediate?: boolean) => void;
  setCameraPose: (
    az: number,
    el: number,
    dist: number,
    immediate?: boolean
  ) => void;
  setTool: (tool: InteractionTool) => void;
  triggerWake: (x: number, z: number, strength?: number) => void;
  triggerVortex: (x: number, z: number, strength?: number) => void;
  triggerRogueWave: (x: number, z: number, amplitude?: number) => void;
  getState: () => Record<string, unknown>;
}

declare global {
  interface Window {
    __SEA_OF_C__?: SeaOfCQaApi;
  }
}

type DragKind = 'none' | 'water' | 'sky' | 'orbit';

function computeAvgAndP95(
  ring: Float64Array,
  scratch: Float64Array,
  count: number,
  outStats: { avg: number; p95: number }
): void {
  if (count <= 0) {
    outStats.avg = 0;
    outStats.p95 = 0;
    return;
  }
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const v = ring[i];
    sum += v;
    // Insertion sort into scratch (zero allocation, tiny N <= 120)
    let j = i - 1;
    while (j >= 0 && scratch[j] > v) {
      scratch[j + 1] = scratch[j];
      j--;
    }
    scratch[j + 1] = v;
  }
  const p95Idx = Math.min(count - 1, Math.floor(count * 0.95));
  outStats.avg = Math.round((sum / count) * 100) / 100;
  outStats.p95 = Math.round(scratch[p95Idx] * 100) / 100;
}

async function boot(): Promise<void> {
  const search = new URLSearchParams(window.location.search);
  if (search.has('capture')) {
    document.body.classList.add('is-capture');
  }
  if (search.has('clean')) {
    document.body.classList.add('is-clean');
  }

  const canvas = document.getElementById('ocean-canvas') as HTMLCanvasElement;
  const punSlotEl = document.getElementById('pun-slot');
  const punSeaEl = document.getElementById('pun-sea');
  const gestureHintEl = document.getElementById('gesture-hint');
  const stateButtons = Array.from(
    document.querySelectorAll<HTMLButtonElement>('.dock-btn[data-state]')
  );
  const toolToggleBtn = document.getElementById(
    'tool-toggle'
  ) as HTMLButtonElement | null;
  const toolLabelEl = document.getElementById('tool-label');

  // ---------------------------------------------------------------------------
  // 1. Three.js r187 WebGPURenderer + Calibrated ACES Filmic Tone Mapping
  // ---------------------------------------------------------------------------
  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(new THREE.Color(0x081326), 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.96;

  await renderer.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x081326);

  const camera = new THREE.PerspectiveCamera(
    40,
    window.innerWidth / Math.max(1, window.innerHeight),
    0.18,
    560
  );

  // ---------------------------------------------------------------------------
  // 2. Unified Wave Field, Marine Environment, Sky, Halftone Sea & 3D C Fluid
  // ---------------------------------------------------------------------------
  const waveUniforms = createWaveFieldUniforms('golden');
  const envMap = createMarineEnvMap(renderer);
  const controller = new SeaOfCController(waveUniforms, 'golden');

  const skyDome = createSkyDome(waveUniforms);
  const oceanSurface = createOceanSurface(waveUniforms);
  const granularFluid = createGranularFluid(waveUniforms, envMap);

  scene.add(skyDome.mesh);
  scene.add(oceanSurface.mesh);
  scene.add(granularFluid.group);

  // ---------------------------------------------------------------------------
  // 3. Marine Studio Lighting Rig (restrained fill so shadowed C sides stay glossy sapphire)
  // ---------------------------------------------------------------------------
  const sunLight = new THREE.DirectionalLight(0xffcf82, 1.85);
  scene.add(sunLight);

  const hemiLight = new THREE.HemisphereLight(0x3a527a, 0x061624, 0.28);
  scene.add(hemiLight);

  // Restrained cyan-emerald ocean bounce light lifting translucent sea-glass bellies
  const bounceLight = new THREE.DirectionalLight(0x36d4c4, 0.36);
  bounceLight.position.set(-14, -5, 18);
  scene.add(bounceLight);

  // Subtle warm-champagne back-rim light separating foreground beveled C edges
  const rimLight = new THREE.DirectionalLight(0xffe0ad, 0.32);
  rimLight.position.set(10, 6, -32);
  scene.add(rimLight);

  // ---------------------------------------------------------------------------
  // 4. Selective HDR Bloom via RenderPipeline (0.5x res + threshold 1.45 keeps C edges crisp)
  // ---------------------------------------------------------------------------
  let renderPipeline: THREE.RenderPipeline | null = null;
  try {
    const rp = new THREE.RenderPipeline(renderer);
    const scenePass = pass(scene, camera);
    const sceneColor = scenePass.getTextureNode('output');
    const bloomPass = bloom(sceneColor, 0.16, 0.22, 1.45);
    bloomPass.setResolutionScale(0.5);
    rp.outputNode = sceneColor.add(bloomPass);
    renderPipeline = rp;
  } catch {
    renderPipeline = null;
  }

  // ---------------------------------------------------------------------------
  // 5. DOM Dock & Pun Synchronization
  // ---------------------------------------------------------------------------
  let hintDismissed = false;
  const dismissHint = (): void => {
    if (hintDismissed) return;
    hintDismissed = true;
    gestureHintEl?.classList.add('is-dismissed');
  };

  const syncStateButtons = (): void => {
    const active = controller.seaState;
    document.body.dataset.seaState = active;
    for (const btn of stateButtons) {
      const isMatch = btn.dataset.state === active;
      btn.classList.toggle('is-active', isMatch);
      btn.setAttribute('aria-pressed', isMatch ? 'true' : 'false');
    }
  };

  const syncToolButton = (): void => {
    const tool = controller.tool;
    const isOrbit = tool === 'orbit';
    document.body.classList.toggle('tool-orbit', isOrbit);
    if (toolToggleBtn) {
      toolToggleBtn.dataset.tool = tool;
      toolToggleBtn.classList.toggle('is-orbit', isOrbit);
      toolToggleBtn.setAttribute('aria-pressed', isOrbit ? 'true' : 'false');
    }
    if (toolLabelEl) {
      toolLabelEl.textContent = isOrbit ? 'Orbit' : 'Wake';
    }
  };

  for (const btn of stateButtons) {
    btn.addEventListener('click', () => {
      const nextState = btn.dataset.state as SeaStateId | undefined;
      if (!nextState) return;
      dismissHint();
      controller.setSeaState(nextState, false);
      syncStateButtons();
    });
  }

  toolToggleBtn?.addEventListener('click', () => {
    dismissHint();
    controller.toggleTool();
    syncToolButton();
  });

  syncStateButtons();
  syncToolButton();

  // ---------------------------------------------------------------------------
  // 6. Tactile Pointer, Wheel, Double-Click & Keyboard Coupling (Zero GC)
  // ---------------------------------------------------------------------------
  const _raycaster = new THREE.Raycaster();
  const _ndc = new THREE.Vector2();
  const _rayHit: OceanRayHit = { region: 'sky', x: 0, z: 0 };

  const castPointerToOcean = (clientX: number, clientY: number): OceanRayHit => {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    _ndc.x = ((clientX - rect.left) / w) * 2 - 1;
    _ndc.y = -((clientY - rect.top) / h) * 2 + 1;
    _raycaster.setFromCamera(_ndc, camera);
    return controller.intersectRayWithOcean(
      _raycaster.ray.origin,
      _raycaster.ray.direction,
      _rayHit
    );
  };

  let activeDrag: DragKind = 'none';
  let activePointerId = -1;
  let lastClientX = 0;
  let lastClientY = 0;

  // Guard hover wakes against OS cursor parked on initial window activation
  const bootMs = performance.now();
  let hoverInitialized = false;
  let hoverAnchorX = 0;
  let hoverAnchorY = 0;
  let hoverUnlocked = false;

  const onPointerDown = (ev: PointerEvent): void => {
    if (ev.button !== 0 && ev.button !== 2) return;
    dismissHint();
    hoverUnlocked = true;
    activePointerId = ev.pointerId;
    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
    canvas.setPointerCapture(ev.pointerId);

    const forceOrbit =
      ev.button === 2 ||
      ev.shiftKey ||
      ev.altKey ||
      ev.metaKey ||
      controller.tool === 'orbit';

    if (forceOrbit) {
      activeDrag = 'orbit';
      document.body.classList.add('is-orbiting');
      return;
    }

    const hit = castPointerToOcean(ev.clientX, ev.clientY);
    if (hit.region === 'sky') {
      activeDrag = 'sky';
    } else {
      activeDrag = 'water';
      controller.beginWaterDrag(hit.x, hit.z);
    }
  };

  const onPointerMove = (ev: PointerEvent): void => {
    if (activeDrag !== 'none' && ev.pointerId === activePointerId) {
      const dx = ev.clientX - lastClientX;
      const dy = ev.clientY - lastClientY;
      lastClientX = ev.clientX;
      lastClientY = ev.clientY;

      if (activeDrag === 'orbit') {
        controller.orbitBy(-dx * 0.0052, dy * 0.004);
      } else if (activeDrag === 'sky') {
        controller.steerSunBy(dx * 0.003, -dy * 0.0022);
      } else if (activeDrag === 'water') {
        const hit = castPointerToOcean(ev.clientX, ev.clientY);
        if (hit.region === 'water') {
          controller.updateWaterDrag(hit.x, hit.z);
        }
      }
      return;
    }

    // Passive hover over ocean spawns throttled expanding ripples
    if (!hoverUnlocked) {
      if (performance.now() - bootMs < 350) return;
      if (!hoverInitialized) {
        hoverInitialized = true;
        hoverAnchorX = ev.clientX;
        hoverAnchorY = ev.clientY;
        return;
      }
      if (Math.hypot(ev.clientX - hoverAnchorX, ev.clientY - hoverAnchorY) < 8) {
        return;
      }
      hoverUnlocked = true;
    }

    const hit = castPointerToOcean(ev.clientX, ev.clientY);
    if (hit.region === 'water') {
      controller.handleWaterMove(hit.x, hit.z, false);
    }
  };

  const endDrag = (ev: PointerEvent): void => {
    if (activeDrag === 'none' || ev.pointerId !== activePointerId) return;
    if (activeDrag === 'water') {
      controller.endWaterDrag();
    }
    if (activeDrag === 'orbit') {
      document.body.classList.remove('is-orbiting');
    }
    activeDrag = 'none';
    activePointerId = -1;
    if (canvas.hasPointerCapture(ev.pointerId)) {
      canvas.releasePointerCapture(ev.pointerId);
    }
  };

  const onDoubleClick = (ev: MouseEvent): void => {
    dismissHint();
    const hit = castPointerToOcean(ev.clientX, ev.clientY);
    if (hit.region === 'water') {
      controller.triggerRogueWave(hit.x, hit.z, 1.15);
    } else {
      controller.triggerRogueWaveAhead(1.15);
    }
  };

  const onWheel = (ev: WheelEvent): void => {
    ev.preventDefault();
    dismissHint();
    controller.zoomBy(ev.deltaY * 0.014);
  };

  const onContextMenu = (ev: MouseEvent): void => {
    ev.preventDefault();
  };

  const onKeyDown = (ev: KeyboardEvent): void => {
    if (ev.code === 'Space') {
      ev.preventDefault();
      dismissHint();
      controller.triggerRogueWaveAhead(1.2);
    } else if (ev.key === '1') {
      dismissHint();
      controller.setSeaState('golden', false);
      syncStateButtons();
    } else if (ev.key === '2') {
      dismissHint();
      controller.setSeaState('storm', false);
      syncStateButtons();
    } else if (ev.key === '3') {
      dismissHint();
      controller.setSeaState('biolum', false);
      syncStateButtons();
    } else if (ev.key === 'o' || ev.key === 'O') {
      dismissHint();
      controller.setTool('orbit');
      syncToolButton();
    } else if (ev.key === 'w' || ev.key === 'W') {
      dismissHint();
      controller.setTool('wake');
      syncToolButton();
    } else if (ev.key === 'r' || ev.key === 'R') {
      dismissHint();
      controller.resetToHero(false);
    }
  };

  const onResize = (): void => {
    const w = window.innerWidth;
    const h = Math.max(1, window.innerHeight);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('dblclick', onDoubleClick);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', onContextMenu);
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('resize', onResize);

  // ---------------------------------------------------------------------------
  // 7. Per-Frame Lighting, Camera & Caption Sync (Zero Per-Frame Allocations)
  // ---------------------------------------------------------------------------
  let lastWrittenFlip = -1;
  let lastPunWarm = false;

  const syncSceneToController = (): void => {
    // 1. Camera pose & matrices
    camera.position.copy(controller.cameraPosition);
    camera.lookAt(controller.cameraTarget);
    camera.updateMatrixWorld();

    // Keep sky dome centered on camera so horizon stays infinitely crisp at all orbits
    skyDome.mesh.position.copy(camera.position);

    // 2. Sync directional sun/moon & marine bounce lights with active blended config
    const cfg = controller.blendedConfig;
    const sunDir = waveUniforms.uSunDir.value;
    sunLight.position.set(sunDir.x * 64, Math.max(2.5, sunDir.y * 64), sunDir.z * 64);
    sunLight.color.setRGB(cfg.sunColor[0], cfg.sunColor[1], cfg.sunColor[2]);
    sunLight.intensity = 1.85 - cfg.biolumGain * 1.4;

    hemiLight.color.setRGB(
      Math.min(1, cfg.skyZenith[0] * 1.8 + 0.015),
      Math.min(1, cfg.skyZenith[1] * 1.8 + 0.022),
      Math.min(1, cfg.skyZenith[2] * 1.8 + 0.042)
    );
    hemiLight.groundColor.setRGB(
      cfg.waterDeep[0] * 0.75,
      cfg.waterDeep[1] * 0.85,
      cfg.waterDeep[2] * 0.95
    );
    hemiLight.intensity = 0.28 - cfg.biolumGain * 0.14;

    bounceLight.color.setRGB(
      cfg.waterShallow[0] * 0.85,
      cfg.waterShallow[1] * 0.95,
      cfg.waterShallow[2] * 0.95
    );
    bounceLight.intensity = 0.36 - cfg.biolumGain * 0.08;

    rimLight.position.set(sunDir.x * 36, Math.max(4, sunDir.y * 28 + 3), sunDir.z * 36);
    rimLight.color.setRGB(cfg.glintColor[0], cfg.glintColor[1], cfg.glintColor[2]);
    rimLight.intensity = 0.32 - cfg.biolumGain * 0.1;

    // 3. Sync frame-locked editorial pun slot ("sea" <-> "C")
    const snap = controller.getSnapshot();
    const roundedFlip = Math.round(snap.flipP * 1000) / 1000;
    if (punSlotEl && roundedFlip !== lastWrittenFlip) {
      lastWrittenFlip = roundedFlip;
      punSlotEl.style.setProperty('--flip-p', String(roundedFlip));
    }
    if (snap.punWarm !== lastPunWarm) {
      lastPunWarm = snap.punWarm;
      punSeaEl?.classList.toggle('is-warm', snap.punWarm);
    }
  };

  syncSceneToController();

  // ---------------------------------------------------------------------------
  // 8. Frame Timing & QA Diagnostics API (window.__SEA_OF_C__)
  // ---------------------------------------------------------------------------
  const timer = new THREE.Timer();
  timer.connect(document);

  let frameCount = 0;
  const frameRing = new Float64Array(120);
  const frameScratch = new Float64Array(120);
  let frameWriteIdx = 0;
  let frameSampleCount = 0;
  const frameStats = { avg: 0, p95: 0 };
  let lastWallMs = performance.now();

  const qaApi: SeaOfCQaApi = {
    isReady: true,
    frameCount: 0,
    avgFrameMs: 0,
    p95FrameMs: 0,
    setSeaState: (state, immediate = false) => {
      controller.setSeaState(state, immediate);
      syncStateButtons();
      syncSceneToController();
    },
    setCameraPose: (az, el, dist, immediate = true) => {
      controller.setCameraPose(az, el, dist, immediate);
      syncSceneToController();
    },
    setTool: (tool) => {
      controller.setTool(tool);
      syncToolButton();
    },
    triggerWake: (x, z, strength = 1.0) => {
      controller.triggerWake(x, z, strength);
    },
    triggerVortex: (x, z, strength = 1.0) => {
      controller.triggerVortex(x, z, strength);
    },
    triggerRogueWave: (x, z, amplitude = 1.15) => {
      controller.triggerRogueWave(x, z, amplitude);
    },
    getState: () => {
      const s = controller.getSnapshot();
      const sun = waveUniforms.uSunDir.value;
      return {
        seaState: s.seaState,
        tool: s.tool,
        cameraPos: [
          Number(s.cameraX.toFixed(3)),
          Number(s.cameraY.toFixed(3)),
          Number(s.cameraZ.toFixed(3)),
        ],
        azimuth: Number(s.azimuth.toFixed(4)),
        elevation: Number(s.elevation.toFixed(4)),
        distance: Number(s.distance.toFixed(3)),
        sunDir: [
          Number(sun.x.toFixed(4)),
          Number(sun.y.toFixed(4)),
          Number(sun.z.toFixed(4)),
        ],
        sunAzimuth: Number(s.sunAzimuth.toFixed(4)),
        sunElevation: Number(s.sunElevation.toFixed(4)),
        flipP: Number(s.flipP.toFixed(3)),
        activeWakeCount: s.activeWakeCount,
        vortexStrength: Number(s.vortexStrength.toFixed(3)),
        glassCount: granularFluid.glassCount,
        sprayCount: granularFluid.sprayCount,
        simTime: Number(s.simTime.toFixed(3)),
        avgFrameMs: qaApi.avgFrameMs,
        p95FrameMs: qaApi.p95FrameMs,
        frameCount: qaApi.frameCount,
      };
    },
  };
  window.__SEA_OF_C__ = qaApi;

  // ---------------------------------------------------------------------------
  // 9. Main Unpaused Animation Loop
  // ---------------------------------------------------------------------------
  renderer.setAnimationLoop((timestamp?: number) => {
    const nowMs = performance.now();
    const wallDeltaMs = Math.max(0.1, nowMs - lastWallMs);
    lastWallMs = nowMs;

    timer.update(timestamp);
    const dtSec = Math.min(0.1, Math.max(0.001, timer.getDelta()));
    controller.step(dtSec);
    syncSceneToController();

    if (renderPipeline) {
      try {
        renderPipeline.render();
      } catch {
        renderPipeline.dispose();
        renderPipeline = null;
        renderer.render(scene, camera);
      }
    } else {
      renderer.render(scene, camera);
    }

    frameCount++;
    frameRing[frameWriteIdx] = wallDeltaMs;
    frameWriteIdx = (frameWriteIdx + 1) % frameRing.length;
    if (frameSampleCount < frameRing.length) frameSampleCount++;

    computeAvgAndP95(frameRing, frameScratch, frameSampleCount, frameStats);
    qaApi.frameCount = frameCount;
    qaApi.avgFrameMs = frameStats.avg;
    qaApi.p95FrameMs = frameStats.p95;
  });

  // ---------------------------------------------------------------------------
  // 10. Clean GPU Resource Disposal on pagehide
  // ---------------------------------------------------------------------------
  window.addEventListener('pagehide', (ev) => {
    if (ev.persisted) return;
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', endDrag);
    canvas.removeEventListener('pointercancel', endDrag);
    canvas.removeEventListener('dblclick', onDoubleClick);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('contextmenu', onContextMenu);
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('resize', onResize);

    renderer.setAnimationLoop(null);
    timer.dispose();
    renderPipeline?.dispose();
    skyDome.dispose();
    oceanSurface.dispose();
    granularFluid.dispose();
    envMap.dispose();
    const gpuCtx = canvas.getContext('webgpu') as GPUCanvasContext | null;
    try {
      gpuCtx?.unconfigure?.();
    } catch {
      // Renderer teardown already released context
    }
    renderer.dispose();
  });
}

void boot();
