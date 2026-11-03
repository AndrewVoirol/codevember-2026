import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { type DayEntry } from '../manifest';
import { type MountedDayShell, mountDayShell } from '../DayShell';
import { CHALKY_SLATE_HEX, EGGSHELL_INK_HEX } from '../palette';
import { chalkySlateBackdropNode } from '../tslMaterials';
import { PointerField } from './PointerField';

export interface SpecimenStageOptions {
  day: number | DayEntry;
  cameraPosition?: [number, number, number];
  fov?: number;
  enableOrbit?: boolean;
  enableShadows?: boolean;
  autoRotateSpeed?: number;
}

export interface SpecimenTickContext {
  elapsedSec: number;
  dtSec: number;
  pointer: PointerField;
  specimenGroup: THREE.Group;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGPURenderer;
  scene: THREE.Scene;
}

declare global {
  interface Window {
    __SPECIMEN_STAGE__?: {
      day: DayEntry;
      isReady: boolean;
      frameCount: number;
      avgFrameMs: number;
      p95FrameMs: number;
      getSceneState: () => Record<string, unknown>;
    };
  }
}

export class SpecimenStage {
  readonly shell: MountedDayShell;
  readonly renderer: THREE.WebGPURenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly specimenGroup: THREE.Group;
  readonly pointer: PointerField;
  readonly timer: THREE.Timer;

  // Shared TSL Uniforms for shaders
  readonly uTime = uniform(0);
  readonly uDelta = uniform(0.016);
  readonly uPointerWorld = uniform(new THREE.Vector3(0, 0, 0));
  readonly uPointerVelocity = uniform(new THREE.Vector3(0, 0, 0));
  readonly uPointerMeta = uniform(new THREE.Vector4(0, 0, 0, 0));

  private readonly tickCallbacks: Array<(ctx: SpecimenTickContext) => void> = [];
  private readonly frameSamplesMs: number[] = [];
  private frameCount = 0;
  private isDisposed = false;
  private readonly onResizeBound: () => void;

  private constructor(
    shell: MountedDayShell,
    renderer: THREE.WebGPURenderer,
    options: SpecimenStageOptions
  ) {
    this.shell = shell;
    this.renderer = renderer;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(CHALKY_SLATE_HEX);
    this.scene.backgroundNode = chalkySlateBackdropNode();

    const fov = options.fov ?? 36;
    this.camera = new THREE.PerspectiveCamera(
      fov,
      window.innerWidth / Math.max(1, window.innerHeight),
      0.1,
      100
    );
    const camPos = options.cameraPosition ?? [0, 0.2, 4.85];
    this.camera.position.set(camPos[0], camPos[1], camPos[2]);

    this.controls = new OrbitControls(this.camera, shell.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.8;
    this.controls.maxDistance = 9.0;
    this.controls.enabled = options.enableOrbit ?? true;
    if (options.autoRotateSpeed !== undefined) {
      this.controls.autoRotate = options.autoRotateSpeed !== 0;
      this.controls.autoRotateSpeed = options.autoRotateSpeed;
    }

    this.specimenGroup = new THREE.Group();
    this.specimenGroup.name = 'SuspendedSpecimenPlinth';
    this.scene.add(this.specimenGroup);

    this.setupStudioLighting(options.enableShadows ?? true);

    this.pointer = new PointerField(shell.canvas);
    this.timer = new THREE.Timer();
    this.timer.connect(document);

    this.onResizeBound = () => this.handleResize();
    window.addEventListener('resize', this.onResizeBound);

    shell.registerDispose(() => this.dispose());
  }

  static async create(options: SpecimenStageOptions): Promise<SpecimenStage> {
    const shell = mountDayShell(options.day);
    const renderer = new THREE.WebGPURenderer({
      canvas: shell.canvas,
      antialias: true,
      powerPreference: 'high-performance',
      alpha: false,
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.setClearColor(new THREE.Color(CHALKY_SLATE_HEX), 1);

    if (options.enableShadows ?? true) {
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFShadowMap;
    }

    await renderer.init();

    const stage = new SpecimenStage(shell, renderer, options);
    stage.exposeQaHook();
    stage.startLoop();
    return stage;
  }

  /**
   * Shared 3-point Cabinet-of-Curiosities Studio Softbox & Grazing Rim Rig.
   */
  private setupStudioLighting(enableShadows: boolean): void {
    const ambient = new THREE.HemisphereLight(EGGSHELL_INK_HEX, CHALKY_SLATE_HEX, 0.72);
    this.scene.add(ambient);

    // Overhead warm calcite softbox key light
    const keyLight = new THREE.DirectionalLight(0xf6f0e4, 2.35);
    keyLight.position.set(2.8, 4.2, 3.2);
    if (enableShadows) {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.near = 0.5;
      keyLight.shadow.camera.far = 14;
      const d = 2.4;
      keyLight.shadow.camera.left = -d;
      keyLight.shadow.camera.right = d;
      keyLight.shadow.camera.top = d;
      keyLight.shadow.camera.bottom = -d;
      keyLight.shadow.bias = -0.0005;
    }
    this.scene.add(keyLight);

    // Cool mineral slate fill light from opposite flank
    const fillLight = new THREE.DirectionalLight(0x8fa3b0, 0.85);
    fillLight.position.set(-3.4, -1.2, 2.0);
    this.scene.add(fillLight);

    // Grazing back-top calcite rim light to separate suspended specimen from #212529 ground
    const rimLight = new THREE.DirectionalLight(0xf3efe6, 1.45);
    rimLight.position.set(-1.5, 3.2, -3.6);
    this.scene.add(rimLight);
  }

  onTick(fn: (ctx: SpecimenTickContext) => void): () => void {
    this.tickCallbacks.push(fn);
    return () => {
      const idx = this.tickCallbacks.indexOf(fn);
      if (idx !== -1) this.tickCallbacks.splice(idx, 1);
    };
  }

  private startLoop(): void {
    let lastWallMs = performance.now();

    this.renderer.setAnimationLoop((timestamp?: number) => {
      if (this.isDisposed) return;

      const nowMs = performance.now();
      const wallDeltaMs = Math.max(0.1, nowMs - lastWallMs);
      lastWallMs = nowMs;

      this.timer.update(timestamp);
      const dtSec = Math.min(0.1, Math.max(0.001, this.timer.getDelta()));
      const elapsedSec = this.timer.getElapsed();

      this.controls.update();
      this.pointer.update(this.camera, dtSec);

      // Sync shared TSL uniforms without allocating
      this.uTime.value = elapsedSec;
      this.uDelta.value = dtSec;
      this.uPointerWorld.value.copy(this.pointer.uWorldPos);
      this.uPointerVelocity.value.copy(this.pointer.uVelocity);
      this.uPointerMeta.value.copy(this.pointer.uMeta);

      const ctx: SpecimenTickContext = {
        elapsedSec,
        dtSec,
        pointer: this.pointer,
        specimenGroup: this.specimenGroup,
        camera: this.camera,
        renderer: this.renderer,
        scene: this.scene,
      };

      for (let i = 0; i < this.tickCallbacks.length; i++) {
        this.tickCallbacks[i](ctx);
      }

      this.renderer.render(this.scene, this.camera);

      this.frameCount++;
      if (this.frameSamplesMs.length >= 120) {
        this.frameSamplesMs.shift();
      }
      this.frameSamplesMs.push(wallDeltaMs);
      this.syncQaTelemetry();
    });
  }

  private handleResize(): void {
    if (this.isDisposed) return;
    const w = window.innerWidth;
    const h = Math.max(1, window.innerHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  private exposeQaHook(): void {
    window.__SPECIMEN_STAGE__ = {
      day: this.shell.day,
      isReady: true,
      frameCount: 0,
      avgFrameMs: 0,
      p95FrameMs: 0,
      getSceneState: () => ({
        day: this.shell.day.day,
        slug: this.shell.day.slug,
        headerLabel: this.shell.day.headerLabel,
        frameCount: this.frameCount,
        cameraPos: this.camera.position.toArray(),
        pointerWorld: this.pointer.worldPos.toArray(),
        pointerSpeed: Number(this.pointer.speed.toFixed(4)),
        pointerDwellSec: Number(this.pointer.dwellTimeSec.toFixed(4)),
        pointerImpulse: Number(this.pointer.impulse.toFixed(4)),
      }),
    };
  }

  private syncQaTelemetry(): void {
    if (!window.__SPECIMEN_STAGE__ || this.frameSamplesMs.length === 0) return;
    const n = this.frameSamplesMs.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += this.frameSamplesMs[i];
    const sorted = [...this.frameSamplesMs].sort((a, b) => a - b);
    const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))];

    window.__SPECIMEN_STAGE__.frameCount = this.frameCount;
    window.__SPECIMEN_STAGE__.avgFrameMs = Number((sum / n).toFixed(2));
    window.__SPECIMEN_STAGE__.p95FrameMs = Number(p95.toFixed(2));
  }

  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    window.removeEventListener('resize', this.onResizeBound);
    this.renderer.setAnimationLoop(null);
    this.pointer.dispose();
    this.controls.dispose();
    this.timer.dispose();
    this.renderer.dispose();
  }
}
