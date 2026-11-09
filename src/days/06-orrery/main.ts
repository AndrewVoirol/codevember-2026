import { type MountedDayShell, mountDayShell } from '../../core/DayShell';
import {
  type OrreryEngineState,
  type OrreryKinematics,
  type OrreryWebGPUEngine,
  evaluateKinematics,
  formatTelemetryForGroup,
  initOrreryWebGPU,
} from './engine-webgpu';
import { type OrreryGeometry, buildOrreryGeometry } from './geometry-3d';
import wgslSource from './shaders.wgsl?raw';
import './orrery.css';

declare global {
  interface Window {
    __ORRERY__?: {
      engine: OrreryWebGPUEngine;
      geometry: OrreryGeometry;
      getActiveDraws: () => number;
      getState: () => OrreryEngineState & { kinematics: OrreryKinematics };
      setPreset: (name: string) => void;
      setDay: (d: number) => void;
      setCutaway: (on: boolean) => void;
      setExplode: (on: boolean) => void;
      setHoveredGroup: (g: number) => void;
    };
  }
}

const SPEEDS = [6, 18, 45, 120] as const;

export interface BootedDay06Orrery {
  shell: MountedDayShell;
  engine: OrreryWebGPUEngine;
  geometry: OrreryGeometry;
}

export async function bootDay06Orrery(): Promise<BootedDay06Orrery> {
  const shell = mountDayShell(6);
  const appRoot = document.getElementById('app') ?? document.body;

  // Mount Translucent Floating Top & Bottom HUD Bars
  const topHud = document.createElement('div');
  topHud.className = 'or-hud-top';
  topHud.innerHTML = `
    <div class="or-btn-group" role="group" aria-label="Camera Presets">
      <span class="or-group-label">Camera</span>
      <button type="button" class="or-btn" data-preset="overview" data-active="1">1 · Overview</button>
      <button type="button" class="or-btn" data-preset="gearbox">2 · Gearbox</button>
      <button type="button" class="or-btn" data-preset="tellurian">3 · Tellurian</button>
      <button type="button" class="or-btn" data-preset="ephemeris">4 · Antikythera</button>
      <button type="button" class="or-btn" data-preset="davinci">5 · Da Vinci</button>
      <button type="button" class="or-btn" data-preset="ecliptic">6 · Plan View</button>
    </div>
    <div class="or-btn-group" role="group" aria-label="Simulation Modes">
      <button type="button" class="or-btn" data-action="cutaway" data-active="0" title="Section outer sleeves & housings (Key: C)">X-Ray Cutaway [C]</button>
      <button type="button" class="or-btn" data-action="explode" data-active="0" title="Vertically telescope coaxial sleeves & gear decks (Key: E)">Explode [E]</button>
      <button type="button" class="or-btn" data-action="play" data-active="1" title="Play / Pause orbital drive (Key: Space)">Playing [Space]</button>
      <button type="button" class="or-btn" data-action="speed" data-speed="1">18 d/s</button>
    </div>
  `;

  const bottomHud = document.createElement('div');
  bottomHud.className = 'or-hud-bottom';
  bottomHud.innerHTML = `
    <div class="or-scrubber-row">
      <span class="or-scrubber-label">EPOCH</span>
      <input type="range" class="or-scrubber" min="0" max="365.25" step="0.25" value="45" aria-label="Day of orbital year (0 to 365.25)" />
      <div class="or-epoch-group">
        <button type="button" class="or-btn" data-epoch="0">Vernal 0°</button>
        <button type="button" class="or-btn" data-epoch="91.3">Solstice 90°</button>
        <button type="button" class="or-btn" data-epoch="182.6">Autumnal 180°</button>
        <button type="button" class="or-btn" data-epoch="238.3">Retrograde 238d</button>
        <button type="button" class="or-btn" data-epoch="273.9">Hibernal 270°</button>
        <button type="button" class="or-btn" data-epoch="12.2">Syzygy Eclipse</button>
      </div>
      <span class="or-readout" data-readout>Vernal · Day 45 · λ 44° (Tau) · δ +16.4° · Δt +4.8m</span>
    </div>
    <div class="or-inspector" data-inspector>
      <span class="or-insp-tag" data-insp-code>GRAND ORRERY</span>
      <div>
        <span class="or-insp-title" data-insp-name>6-Planet Coaxial Train, Saturn Rings, Da Vinci Foliot & Antikythera Anomaly</span>
        <span class="or-insp-metrics" data-insp-metrics> · ω_M 4.15× · ω_V 1.63× · ω_E 1.00× · ω_Mars 0.53× · ω_Jup 0.24× · ω_Sat 0.16×</span>
      </div>
      <div class="or-insp-desc" data-insp-desc>Hover or click any 3D gear, coaxial sleeve, pierced truss carrier, Saturn ring, Da Vinci foliot, or Antikythera dial to inspect its live telemetry. Press [E] to explode coaxial decks or drag the left handwheel.</div>
    </div>
  `;

  appRoot.appendChild(topHud);
  appRoot.appendChild(bottomHud);

  const readout = bottomHud.querySelector<HTMLElement>('[data-readout]');
  const scrubber = bottomHud.querySelector<HTMLInputElement>('.or-scrubber');
  const inspCode = bottomHud.querySelector<HTMLElement>('[data-insp-code]');
  const inspName = bottomHud.querySelector<HTMLElement>('[data-insp-name]');
  const inspMetrics = bottomHud.querySelector<HTMLElement>('[data-insp-metrics]');
  const inspDesc = bottomHud.querySelector<HTMLElement>('[data-insp-desc]');
  const presetBtns = Array.from(topHud.querySelectorAll<HTMLButtonElement>('[data-preset]'));
  const epochBtns = Array.from(bottomHud.querySelectorAll<HTMLButtonElement>('[data-epoch]'));
  const cutawayBtn = topHud.querySelector<HTMLButtonElement>('[data-action="cutaway"]');
  const explodeBtn = topHud.querySelector<HTMLButtonElement>('[data-action="explode"]');
  const playBtn = topHud.querySelector<HTMLButtonElement>('[data-action="play"]');
  const speedBtn = topHud.querySelector<HTMLButtonElement>('[data-action="speed"]');

  let speedIdx = 1;

  function syncButtons(state: OrreryEngineState): void {
    for (const btn of presetBtns) {
      btn.dataset.active = btn.dataset.preset === state.preset ? '1' : '0';
    }
    if (cutawayBtn) {
      const on = state.goalCutaway > 0.5;
      cutawayBtn.dataset.active = on ? '1' : '0';
      cutawayBtn.textContent = on ? 'X-Ray Cutaway: ON [C]' : 'X-Ray Cutaway [C]';
    }
    if (explodeBtn) {
      const on = state.goalExplode > 0.5;
      explodeBtn.dataset.active = on ? '1' : '0';
      explodeBtn.textContent = on ? 'Explode: ON [E]' : 'Explode [E]';
    }
    if (playBtn) {
      playBtn.dataset.active = state.playing ? '1' : '0';
      playBtn.textContent = state.playing ? 'Playing [Space]' : 'Paused [Space]';
    }
  }

  function updateInspectorAndReadout(kin: OrreryKinematics, state: OrreryEngineState): void {
    const dayMod = Math.round(kin.dWrap);
    const declSign = kin.declDeg >= 0 ? '+' : '';
    const eotSign = kin.eotMin >= 0 ? '+' : '';
    const text = `${kin.season} · Day ${dayMod} · λ ${kin.lonDeg}° (${kin.zodiacSign}) · δ ${declSign}${kin.declDeg.toFixed(1)}° · Δt ${eotSign}${kin.eotMin.toFixed(1)}m`;
    if (readout && readout.textContent !== text) {
      readout.textContent = text;
    }
    if (scrubber && document.activeElement !== scrubber) {
      scrubber.value = String(kin.dWrap.toFixed(2));
    }

    const activeGroup = state.hoveredGroup || state.selectedGroup || 0;
    if (activeGroup > 0 && inspCode && inspName && inspMetrics && inspDesc) {
      const info = formatTelemetryForGroup(activeGroup, kin);
      if (info) {
        inspCode.textContent = info.code;
        inspName.textContent = info.name;
        inspMetrics.textContent = ' · ' + info.metrics;
        inspDesc.textContent = info.summary;
      }
    } else if (inspCode && inspName && inspMetrics && inspDesc) {
      inspCode.textContent =
        state.goalExplode > 0.5
          ? 'EXPLODED VIEW'
          : state.goalCutaway > 0.5
            ? 'X-RAY SECTION'
            : 'GRAND ORRERY';
      inspName.textContent =
        '6-Planet Coaxial Train, Saturn Rings, Da Vinci Foliot & Antikythera Anomaly';
      inspMetrics.textContent =
        ' · ω_M 4.15× · ω_V 1.63× · ω_E 1.00× · ω_Mars 0.53× · ω_Jup 0.24× · ω_Sat 0.16×';
      inspDesc.textContent =
        'Hover or click any 3D gear, coaxial sleeve, pierced truss carrier, Saturn ring, Da Vinci foliot, or Antikythera dial to inspect its live telemetry. Press [E] to explode coaxial decks or drag the left handwheel.';
    }
  }

  const geometry = buildOrreryGeometry();
  const engine = await initOrreryWebGPU({
    canvas: shell.canvas,
    shaderSource: wgslSource,
    geometry,
    isLightTheme: false,
    onTelemetryChange: updateInspectorAndReadout,
  });

  // Expose QA & Optical-Bench hooks
  const frameWindowMs = new Float32Array(60);
  let frameRingIdx = 0;
  let frameCount = 0;
  let p95FrameMs = 8.5;

  window.__ORRERY__ = {
    engine,
    geometry,
    getActiveDraws: () => engine.getActiveDraws(),
    getState: () => ({
      ...engine.state,
      kinematics: evaluateKinematics(engine.state.day),
    }),
    setPreset: (name: string) => {
      engine.state.hoveredGroup = 0;
      engine.setPreset(name);
      syncButtons(engine.state);
    },
    setDay: (d: number) => {
      engine.state.playing = false;
      engine.state.day = d;
      engine.state.goalDay = d;
      syncButtons(engine.state);
    },
    setCutaway: (on: boolean) => {
      engine.state.goalCutaway = on ? 1 : 0;
      engine.state.cutaway = on ? 1 : 0;
      syncButtons(engine.state);
    },
    setExplode: (on: boolean) => {
      engine.state.goalExplode = on ? 1 : 0;
      engine.state.explode = on ? 1 : 0;
      syncButtons(engine.state);
    },
    setHoveredGroup: (g: number) => {
      engine.state.hoveredGroup = g;
    },
  };

  window.__SPECIMEN_STAGE__ = {
    day: shell.day,
    isReady: true,
    get frameCount() {
      return frameCount;
    },
    get avgFrameMs() {
      return Number(engine.state.avgFrameMs.toFixed(2));
    },
    get p95FrameMs() {
      return Number(p95FrameMs.toFixed(2));
    },
    getSceneState: () => ({
      day: engine.state.day,
      preset: engine.state.preset,
      az: engine.state.az,
      el: engine.state.el,
      dist: engine.state.dist,
      cutaway: engine.state.cutaway,
      explode: engine.state.explode,
      activeDraws: engine.getActiveDraws(),
      hoveredGroup: engine.state.hoveredGroup,
      selectedGroup: engine.state.selectedGroup,
    }),
  };

  // Wire HUD Buttons
  for (const btn of presetBtns) {
    btn.addEventListener('click', () => {
      const preset = btn.dataset.preset;
      if (preset) {
        engine.state.hoveredGroup = 0;
        engine.setPreset(preset);
        syncButtons(engine.state);
      }
    });
  }
  for (const btn of epochBtns) {
    btn.addEventListener('click', () => {
      const d = Number.parseFloat(btn.dataset.epoch ?? '');
      if (Number.isFinite(d)) {
        engine.state.goalDay = d;
      }
    });
  }
  cutawayBtn?.addEventListener('click', () => {
    engine.state.goalCutaway = engine.state.goalCutaway > 0.5 ? 0 : 1;
    syncButtons(engine.state);
  });
  explodeBtn?.addEventListener('click', () => {
    engine.state.goalExplode = engine.state.goalExplode > 0.5 ? 0 : 1;
    syncButtons(engine.state);
  });
  playBtn?.addEventListener('click', () => {
    engine.state.playing = !engine.state.playing;
    syncButtons(engine.state);
  });
  speedBtn?.addEventListener('click', () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    engine.state.speedDaysPerSec = SPEEDS[speedIdx];
    speedBtn.textContent = `${SPEEDS[speedIdx]} d/s`;
  });
  scrubber?.addEventListener('input', () => {
    engine.state.playing = false;
    engine.state.goalDay = Number.parseFloat(scrubber.value) || 0;
    syncButtons(engine.state);
  });

  // 3D Pointer Orbit / Pan / Handwheel Crank / Sub-Assembly Picking
  const canvas = shell.canvas;
  let pointerDown = false;
  let dragMode: 'orbit' | 'pan' | 'crank' = 'orbit';
  let startX = 0;
  let startY = 0;
  let movedDist = 0;

  canvas.addEventListener('pointerdown', (ev: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const nx = (ev.clientX - rect.left) / rect.width;
    const ny = (ev.clientY - rect.top) / rect.height;
    const hitGroup = engine.pickAtNormalized(nx, ny);

    pointerDown = true;
    startX = ev.clientX;
    startY = ev.clientY;
    movedDist = 0;
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // Ignore synthetic pointer events without valid pointerId
    }

    if (ev.shiftKey || ev.button === 2) {
      dragMode = 'pan';
    } else if (hitGroup === 1) {
      dragMode = 'crank';
      engine.state.playing = false;
      syncButtons(engine.state);
    } else {
      dragMode = 'orbit';
    }
  });

  canvas.addEventListener('pointermove', (ev: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const nx = (ev.clientX - rect.left) / rect.width;
    const ny = (ev.clientY - rect.top) / rect.height;

    if (!pointerDown) {
      engine.state.hoveredGroup = engine.pickAtNormalized(nx, ny);
      return;
    }

    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    startX = ev.clientX;
    startY = ev.clientY;
    movedDist += Math.hypot(dx, dy);

    if (dragMode === 'crank') {
      engine.state.goalDay += (dx - dy * 0.5) * 0.45;
    } else if (dragMode === 'pan') {
      const azRad = (engine.state.az * Math.PI) / 180;
      const panScale = engine.state.dist * 0.0014;
      engine.state.preset = 'custom';
      engine.state.goalTarget[0] += Math.sin(azRad) * dx * panScale;
      engine.state.goalTarget[1] -= Math.cos(azRad) * dx * panScale;
      engine.state.goalTarget[2] += dy * panScale;
      syncButtons(engine.state);
    } else {
      engine.state.preset = 'custom';
      engine.state.goalAz -= dx * 0.42;
      engine.state.goalEl = Math.max(8, Math.min(85, engine.state.goalEl + dy * 0.35));
      syncButtons(engine.state);
    }
  });

  const endPointer = (ev: PointerEvent) => {
    if (!pointerDown) return;
    pointerDown = false;
    if (movedDist < 5) {
      const rect = canvas.getBoundingClientRect();
      const nx = (ev.clientX - rect.left) / rect.width;
      const ny = (ev.clientY - rect.top) / rect.height;
      const hit = engine.pickAtNormalized(nx, ny);
      engine.state.selectedGroup = engine.state.selectedGroup === hit ? 0 : hit;
    }
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('pointerleave', () => {
    if (!pointerDown) engine.state.hoveredGroup = 0;
  });

  canvas.addEventListener(
    'wheel',
    (ev: WheelEvent) => {
      ev.preventDefault();
      const factor = Math.exp(ev.deltaY * 0.0011);
      engine.state.goalDist = Math.max(28, Math.min(560, engine.state.goalDist * factor));
    },
    { passive: false }
  );

  // Keyboard Shortcuts
  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.target instanceof HTMLInputElement) return;
    if (shell.drawerBackdrop.classList.contains('is-open')) return;

    if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      engine.state.playing = false;
      engine.state.goalDay += ev.shiftKey ? 30 : 8;
      syncButtons(engine.state);
    } else if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      engine.state.playing = false;
      engine.state.goalDay -= ev.shiftKey ? 30 : 8;
      syncButtons(engine.state);
    } else if (ev.key === ' ') {
      ev.preventDefault();
      engine.state.playing = !engine.state.playing;
      syncButtons(engine.state);
    } else if (ev.key === 'c' || ev.key === 'C') {
      ev.preventDefault();
      engine.state.goalCutaway = engine.state.goalCutaway > 0.5 ? 0 : 1;
      syncButtons(engine.state);
    } else if (ev.key === 'e' || ev.key === 'E') {
      ev.preventDefault();
      engine.state.goalExplode = engine.state.goalExplode > 0.5 ? 0 : 1;
      syncButtons(engine.state);
    } else if (ev.key === '1') {
      engine.state.hoveredGroup = 0;
      engine.setPreset('overview');
      syncButtons(engine.state);
    } else if (ev.key === '2') {
      engine.state.hoveredGroup = 0;
      engine.setPreset('gearbox');
      syncButtons(engine.state);
    } else if (ev.key === '3') {
      engine.state.hoveredGroup = 0;
      engine.setPreset('tellurian');
      syncButtons(engine.state);
    } else if (ev.key === '4') {
      engine.state.hoveredGroup = 0;
      engine.setPreset('ephemeris');
      syncButtons(engine.state);
    } else if (ev.key === '5') {
      engine.state.hoveredGroup = 0;
      engine.setPreset('davinci');
      syncButtons(engine.state);
    } else if (ev.key === '6') {
      engine.state.hoveredGroup = 0;
      engine.setPreset('ecliptic');
      syncButtons(engine.state);
    } else if (ev.key === 'Escape') {
      engine.state.selectedGroup = 0;
      engine.state.hoveredGroup = 0;
      engine.state.goalCutaway = 0;
      engine.state.goalExplode = 0;
      engine.setPreset('overview');
      syncButtons(engine.state);
    }
  };
  window.addEventListener('keydown', onKeyDown);

  // Main Zero-Allocation rAF Loop
  let rafId = 0;
  let disposed = false;
  let lastNow = 0;
  const sortedScratch = new Float32Array(60);

  function loop(now: number): void {
    if (disposed) return;
    const dtMs = lastNow ? now - lastNow : 16.67;
    lastNow = now;
    const dtSec = Math.min(dtMs * 0.001, 0.05);
    engine.state.lastFrameMs = dtMs;
    engine.state.avgFrameMs = engine.state.avgFrameMs * 0.92 + dtMs * 0.08;

    frameWindowMs[frameRingIdx] = dtMs;
    frameRingIdx = (frameRingIdx + 1) % 60;
    frameCount++;
    if (frameCount % 15 === 0) {
      sortedScratch.set(frameWindowMs);
      sortedScratch.sort();
      p95FrameMs = sortedScratch[56];
    }

    engine.renderFrame(now, dtSec);
    rafId = requestAnimationFrame(loop);
  }
  rafId = requestAnimationFrame(loop);

  shell.registerDispose(() => {
    disposed = true;
    cancelAnimationFrame(rafId);
    window.removeEventListener('keydown', onKeyDown);
    topHud.remove();
    bottomHud.remove();
    engine.dispose();
  });

  return { shell, engine, geometry };
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay06Orrery();
}
