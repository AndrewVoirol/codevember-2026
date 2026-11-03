import { type DayEntry } from '../manifest';
import { type MountedDayShell, mountDayShell } from '../DayShell';
import { CHALKY_SLATE_LINEAR } from '../palette';

export interface RawWebGPUStageOptions {
  day: number | DayEntry;
  preferHdrExtended?: boolean;
}

export interface RawWebGPUTickContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  width: number;
  height: number;
  elapsedSec: number;
  dtSec: number;
  pointerNdc: readonly [number, number];
  pointerVelocity: readonly [number, number];
  pointerDown: boolean;
  pointerDwellSec: number;
  pointerImpulse: number;
}

const OPTIONAL_2026_FEATURES: GPUFeatureName[] = [
  'shader-f16',
  'subgroups',
  'float32-filterable',
  'float32-blendable',
  'dual-source-blending',
  'clip-distances',
  'timestamp-query',
];

/**
 * Synchronous Single-Encoder Float32 HDR Readback Helper (`rgba16float`)
 * Encodes and submits `copyTextureToBuffer` synchronously BEFORE yielding across `await`.
 */
export async function readRgba16FloatTextureAsFloat32(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number
): Promise<{ width: number; height: number; rgba: Float32Array; nanInfCount: number }> {
  const bytesPerPixel = 8; // rgba16float = 4 * 2 bytes
  const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
  const staging = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture, origin: { x: 0, y: 0, z: 0 } },
    { buffer: staging, bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 }
  );
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const f16View = new Float16Array(staging.getMappedRange());
  const rgba = new Float32Array(width * height * 4);
  let nanInfCount = 0;
  const rowShorts = bytesPerRow / 2;

  for (let y = 0; y < height; y++) {
    const srcRow = y * rowShorts;
    const dstRow = y * width * 4;
    for (let c = 0; c < width * 4; c++) {
      const v = f16View[srcRow + c];
      if (!Number.isFinite(v)) nanInfCount++;
      rgba[dstRow + c] = v;
    }
  }
  staging.unmap();
  staging.destroy();
  return { width, height, rgba, nanInfCount };
}

/**
 * Turnkey Raw WebGPU Stage (`src/core/stage/RawWebGPUStage.ts`)
 * For multi-pass compute, optical caustic, hydraulic erosion, and wave-PDE days.
 */
export class RawWebGPUStage {
  readonly shell: MountedDayShell;
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;

  /** Canonical Chalky Slate (#212529) clear value for render passes */
  readonly clearColor: GPUColorDict = {
    r: CHALKY_SLATE_LINEAR[0],
    g: CHALKY_SLATE_LINEAR[1],
    b: CHALKY_SLATE_LINEAR[2],
    a: 1.0,
  };

  private readonly tickCallbacks: Array<(ctx: RawWebGPUTickContext) => void> = [];
  private rafId = 0;
  private isDisposed = false;
  private pointerNdc: [number, number] = [0, 0];
  private prevPointerNdc: [number, number] = [0, 0];
  private pointerVel: [number, number] = [0, 0];
  private pointerDown = false;
  private pointerDwellSec = 0;
  private pointerImpulse = 0;
  private readonly onResizeBound: () => void;

  private constructor(
    shell: MountedDayShell,
    adapter: GPUAdapter,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat
  ) {
    this.shell = shell;
    this.adapter = adapter;
    this.device = device;
    this.context = context;
    this.format = format;

    this.onResizeBound = () => this.syncCanvasSize();
    window.addEventListener('resize', this.onResizeBound);
    this.syncCanvasSize();
    this.attachPointerListeners();

    shell.registerDispose(() => this.dispose());
  }

  static async create(options: RawWebGPUStageOptions): Promise<RawWebGPUStage> {
    const shell = mountDayShell(options.day);
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser.');
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('Failed to acquire WebGPU adapter.');
    }

    const requiredFeatures = OPTIONAL_2026_FEATURES.filter((feat) => adapter.features.has(feat));
    const device = await adapter.requestDevice({ requiredFeatures });

    const context = shell.canvas.getContext('webgpu');
    if (!context) {
      throw new Error('Failed to acquire WebGPU canvas context.');
    }

    const hdrMatches =
      Boolean(options.preferHdrExtended) &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(dynamic-range: high)').matches;

    const format: GPUTextureFormat = hdrMatches
      ? 'rgba16float'
      : navigator.gpu.getPreferredCanvasFormat();

    context.configure({
      device,
      format,
      alphaMode: 'premultiplied',
      toneMapping: { mode: hdrMatches ? 'extended' : 'standard' },
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const stage = new RawWebGPUStage(shell, adapter, device, context, format);
    stage.startLoop();
    return stage;
  }

  private attachPointerListeners(): void {
    const canvas = this.shell.canvas;
    canvas.addEventListener(
      'pointermove',
      (ev: PointerEvent) => {
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const nx = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        const ny = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        this.pointerNdc = [Math.max(-1, Math.min(1, nx)), Math.max(-1, Math.min(1, ny))];
      },
      { passive: true }
    );
    canvas.addEventListener(
      'pointerdown',
      () => {
        this.pointerDown = true;
        this.pointerImpulse = 1.0;
      },
      { passive: true }
    );
    window.addEventListener(
      'pointerup',
      () => {
        this.pointerDown = false;
      },
      { passive: true }
    );
  }

  private syncCanvasSize(): void {
    if (this.isDisposed) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(window.innerWidth * dpr));
    const h = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (this.shell.canvas.width !== w || this.shell.canvas.height !== h) {
      this.shell.canvas.width = w;
      this.shell.canvas.height = h;
    }
  }

  onTick(fn: (ctx: RawWebGPUTickContext) => void): () => void {
    this.tickCallbacks.push(fn);
    return () => {
      const idx = this.tickCallbacks.indexOf(fn);
      if (idx !== -1) this.tickCallbacks.splice(idx, 1);
    };
  }

  private startLoop(): void {
    const startMs = performance.now();
    let prevMs = startMs;

    const frame = (nowMs: number) => {
      if (this.isDisposed) return;
      const dtSec = Math.min(0.1, Math.max(0.001, (nowMs - prevMs) * 0.001));
      const elapsedSec = (nowMs - startMs) * 0.001;
      prevMs = nowMs;

      const vx = (this.pointerNdc[0] - this.prevPointerNdc[0]) / dtSec;
      const vy = (this.pointerNdc[1] - this.prevPointerNdc[1]) / dtSec;
      const alpha = 1 - Math.exp(-dtSec * 16);
      this.pointerVel[0] += (vx - this.pointerVel[0]) * alpha;
      this.pointerVel[1] += (vy - this.pointerVel[1]) * alpha;
      this.prevPointerNdc = [this.pointerNdc[0], this.pointerNdc[1]];

      if (this.pointerDown) {
        this.pointerDwellSec = Math.min(4.0, this.pointerDwellSec + dtSec * 1.4);
      } else {
        this.pointerDwellSec = Math.max(0, this.pointerDwellSec - dtSec * 1.8);
      }
      this.pointerImpulse *= Math.exp(-dtSec * 4.5);

      const ctx: RawWebGPUTickContext = {
        device: this.device,
        context: this.context,
        format: this.format,
        width: this.shell.canvas.width,
        height: this.shell.canvas.height,
        elapsedSec,
        dtSec,
        pointerNdc: this.pointerNdc,
        pointerVelocity: this.pointerVel,
        pointerDown: this.pointerDown,
        pointerDwellSec: this.pointerDwellSec,
        pointerImpulse: this.pointerImpulse,
      };

      for (let i = 0; i < this.tickCallbacks.length; i++) {
        this.tickCallbacks[i](ctx);
      }

      this.rafId = requestAnimationFrame(frame);
    };

    this.rafId = requestAnimationFrame(frame);
  }

  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    cancelAnimationFrame(this.rafId);
    window.removeEventListener('resize', this.onResizeBound);
    try {
      this.context.unconfigure();
      this.device.destroy();
    } catch {
      // Ignore teardown errors
    }
  }
}
