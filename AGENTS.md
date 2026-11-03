# Codevember 2026 (`code-vemeber-26`) — Engineering & Optical Playbook (`AGENTS.md`)

## 0. Fast Triage & Anti-Relearning Runbook

- **Daily Build Skill**: Read [`.agents/skills/build-day/SKILL.md`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/.agents/skills/build-day/SKILL.md) and [`SPECIMEN_CATALOG.md`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/SPECIMEN_CATALOG.md) before starting any day.
- **Scaffold a Day (`01`..`30`)**: `npm run day:new <XX> [-- --substrate=tsl|tsl-compute|rapier|wgsl]`
- **Symbol Map**: `npm run map:symbols -- <keyword>`
- **Blast-Radius Tests**: `npm run test:blast` (or `npm test` for full Vitest suite)
- **Static WGSL & Layout Audit**: `npm run lint:wgsl`
- **Dev Server**: `npm run dev` (runs on `http://localhost:5173`)
- **Production MPA Build**: `npm run build` (`tsc -b && vite build`)
- **Live Hardware Metal & Contact-Sheet Gate**: `npm run bench:optical -- --day <XX>`

---

## 1. Series Identity & Architectural Contract

**Codevember 2026 (`code-vemeber-26`)** is a 30-day series of interactive WebGPU, Three.js `r187` (`WebGPURenderer` + `TSL`), Rapier `0.21`, and WGSL pieces (`Day 01 - Fold` through `Day 30 - Dissolve`), unified by four strict invariants:

1. **Single Vite 8 Multi-Page App (MPA) Topology**:
   - Every day lives at its own standalone HTML route (`/<XX>-<slug>/index.html` → `/src/days/<XX>-<slug>/main.ts`), guaranteeing a fresh DOM and isolated WebGPU `GPUDevice` / `WebGPURenderer` context per day with zero cross-day VRAM or bind-group leaks.
   - Root `/` ([`index.html`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/index.html) → [`src/root.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/root.ts)) dynamically mounts `getLatestActiveDay()` from [`src/core/manifest.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/core/manifest.ts) with zero client-side redirect.
2. **Minimal Chalky Slate (`#212529`) Shell (`DayShell.ts`)**:
   - **Ground**: Strictly **Chalky Slate (`#212529`)** across all 30 days (rendered with subtle studio softbox falloff and mineral tooth dithering via `chalkySlateBackdropNode` in [`src/core/tslMaterials.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/core/tslMaterials.ts)—never pure `#000000` black).
   - **Header**: Top-left `"Day XX - [THEME]"` button in **Eggshell (`#f3efe6`)** via [`src/core/DayShell.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/core/DayShell.ts). Clicking it opens the portaled 30-day index drawer (`document.body`, `z-index: 1000`); `ArrowLeft` / `ArrowRight` steps across completed days.
   - **Zero UI Chrome**: Never add debug sliders, pill bars, FPS counters, or subtitle overlays unless explicitly requested by the user for a specific day.
3. **Suspended Specimen Framing ("Cabinet of Curiosities")**:
   - Stage every theme as an isolated physical specimen, core sample, or optical apparatus suspended in the center of the Chalky Slate space (`stage.specimenGroup`) under the shared 3-point studio softbox + calcite rim lighting rig ([`SpecimenStage.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/core/stage/SpecimenStage.ts) or [`RawWebGPUStage.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/core/stage/RawWebGPUStage.ts)).
   - Every piece must respond physically to pointer kinematics (`stage.pointer`: `worldPos`, `velocity`, `speed`, `dwellTimeSec`, `impulse`)—never ship a passive screensaver.
4. **Physical Mineral & Botanical Pigment Palette (`src/core/palette.ts`)**:
   - Strictly zero digital RGB neons (`#00ffff`, `#ff00ff`, etc.).
   - Pull all colors from the 14 natural pigments in [`src/core/palette.ts`](file:///Users/andrewvoirol/Antigravity/code-vemeber-26/src/core/palette.ts) (`boneCalcite`, `warmKaolin`, `yellowOchre`, `rawUmber`, `burntSienna`, `hematiteRust`, `cinnabarVermilion`, `realgarAmber`, `verdigrisMalachite`, `celadonGlauconite`, `azuritePrussian`, `lapisUltramarine`, `mossUmbrous`, `sporeSulfur`) using 2-flux Kubelka-Munk (`kubelkaMunkGlaze`) or Beer-Lambert transmittance (`beerLambertTransmittance`).

---

## 2. WebGPU, Three.js (`r187` / TSL) & WGSL Guardrails

1. **Canonical Three.js `r187` Imports & Lifecycle**:
   - Import `THREE` from `'three/webgpu'` and nodes from `'three/tsl'`.
   - Call `await renderer.init()` before starting `renderer.setAnimationLoop(...)` (handled automatically by `SpecimenStage.create()`).
   - Use `THREE.Timer` (`timer.connect(document)`) instead of deprecated `THREE.Clock`.
   - In TSL struct storage buffers, always index via `.element(i).get('field')` (`r187` array requirement).
2. **Uniform Control Flow & Layout Hygiene**:
   - Never place implicit-LOD `texture(...)` / `textureSample(...)`, `dpdx()`, `dpdy()`, `fwidth()`, or `subgroupAdd()` inside dynamic `if` / `If(...)` or `Loop(...)` branches.
   - In raw WGSL structs, avoid `vec3<f32>` alignment traps by packing `vec4<f32>` or running `npm run lint:wgsl` (`wgsl_reflect@1.6.0`).
3. **Zero Per-Frame Allocations**:
   - Pre-allocate all scratch `THREE.Vector3`, `THREE.Matrix4`, `THREE.Color`, and `Float32Array` buffers at module or class scope. Never allocate inside `onTick` or `setAnimationLoop`.
4. **Zero Synthetic Mathematical Tell-Tales**:
   - Never use polar star harmonics $\cos(m\theta)$ or $\sin(n\theta)$ ($m \ge 3$) on radial droplets, ripples, or blooms. Use irrational-angle rotated 3D FBM (`mineralToothFbm`) and physical wave/shear kinematics.
   - Never hardcode `(0, 0)` world-origin shading vectors for pointer interactions; always evaluate distances relative to `stage.uPointerWorld`.
5. **Hardware Metal Verification Before Marking Complete**:
   - Before setting a day's `status: 'complete'` in `src/core/manifest.ts`, run `npm run bench:optical -- --day <XX>` and visually inspect `screenshots/day-<XX>-<slug>-contact-sheet.png` to confirm `0` WebGPU/console errors, `avgFrameMs <= 16.6 ms`, and clear tactile pointer response.
