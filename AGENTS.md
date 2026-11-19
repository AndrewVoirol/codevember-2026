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
5. **Physical Delamination, Hover Affordance & Overhang Shadow Hygiene**:
   - For exfoliating or eroding materials (rust scale, bark, mineral crust, paint, ice), never use uniform small discs/triangles tumbling in midair ("confetti") or smooth Gaussian wear ovals clamped to rectangular UV bounds. Combine **surface-adhering irregular polygonal scabs** that visibly pry upward on mouseover (`cursor: 'crosshair'` / clear hover affordance) with multi-tier stepped chips + downward-settling grit, and gate substrate wear by **per-texel spall-plate ceilings** so scars fracture along irregular plate seams.
   - Disable `receiveShadow` on porous hanging drapery/stalactites directly beneath overhanging bars when directional shadow maps would stamp a hard horizontal cutoff shelf across the organic form.
6. **Hardware Metal Verification Before Marking Complete**:
   - Before running `npm run bench:optical -- --day <XX>`, navigate any open Chrome DevTools MCP WebGPU tab to `about:blank` so competing unpaused WebGPU tabs do not inflate `avgFrameMs`. Visually inspect `screenshots/day-<XX>-<slug>-contact-sheet.png` to confirm `0` WebGPU/console errors, `avgFrameMs <= 16.6 ms`, and clear tactile pointer response before setting `status: 'complete'` in `src/core/manifest.ts`.
7. **3D Carved Stone Relief, Signed Cube-Face Masks & Multi-Tier Atlas Isolation**:
   - Give carved stone/mineral monoliths authentic material weight via periodic **3D object-space petrography** (`Data3DTexture` / 3D triplanar clasts + normal-mapped ridge fracture) gated by a **surface-vs-groove relief mask**: rough chiseled outer blocks and wall faces paired with **honed, low-relief corridor/optical beds** so routes and wet glazes stay readable (never stretch 2D face-UV noise down vertical trench walls, and weld/overlap octant Dual-Contouring chunk boundaries so seams never appear on flat faces).
   - Never project per-face bridge/feature masks onto a cube with unsigned `Math.abs(dot(axis, normal))`, which bleeds masks onto the antipodal opposite face (`+X <-> -X`, `+Y <-> -Y`, `+Z <-> -Z`) and punches holes in floor trails; always enforce signed `featureFace === domFace`.
   - Whenever multi-tier geometry (e.g., `tier = 0` overpass bridge decks above `tier = 1` underpass tunnels) overlaps at the same `(face, u, v)`, split surface state/paint `DataTexture`s into **tier-isolated UV rows** (`tier = 0 -> v ∈ [0, 0.5]`, `tier = 1 -> [0.5, 1.0]`).
8. **Perspective-Ray-Projected Coplanar Trompe-l'Œil (Zero Z-Layer Seams at Rest)**:
   - When disguising stacked 3D `Z`-layers as a flat 2D drawing or surface at rest (`depthBreak = 0`), never drive the resting flat wash from discrete per-layer attributes (`aLayerIdx`) or raw world `wp.xy`, which leave visible concentric step seams. Project world coordinates along the resting camera sightline onto the front plane (`projS = (camZ - frontZ) / max(eps, camZ - wp.z); projXY = (wp.xy - camXY) * projS + camXY`), carry flat surface cues (e.g., page gutter crease) continuously across interior layers, and suppress 3D bevels, deckle rims, and tunnel-step AO until `depthBreak > 0`.

---

## 3. Tactile Physics Ergonomics & Zero-Stall Command Hygiene

1. **Continuous Pointer Drag & Release Ergonomics (Never Fight the Cursor)**:
   - **Zero Quadrant Angle Jumps**: Never use discrete `if / else if` quadrant branches (`px > 0`, `py > 0`) to snap target orientation angles or flip anchor ends during an active pointer drag—doing so inverts the projection axis mid-gesture and fights the user's hand. Use smooth, monotonic functions (`Math.tanh(px * k)`) and track relative drag deltas (`curProj - dragStartProj`) anchored at `pointerdown`.
   - **Honor Drag Direction on Release**: When the user drags a specimen toward an open/unfolded state and releases, complete the transition in the direction of their drag rather than snapping back to a collapsed default unless a narrow threshold is met.
2. **Gravitational Acceleration vs. `smoothstep` Braking**:
   - Never use `smoothstep` (`t²(3 - 2t)`) for gravity-driven falls, pendulums, or sequential hinge cascades—`smoothstep` brakes velocity to `0` right before contact (`t = 1`), creating robotic start-stop pauses.
   - Use gravitational acceleration curves ($\theta(t) = \pi \, t^\gamma$ with $\gamma \in [1.4, 1.7]$) that accelerate all the way into impact, paired with cascade overlap (`CASCADE_OVERLAP > 1.0`) and damped contact rebound so momentum transfers continuously across segments.
3. **2-DOF Groove Tribology, Orthogonal Junction Continuity & V-Pocket Corner Rest**:
   - Never fake perpendicular T-junction capture on a 1D spline with velocity-inverse score multipliers (which feel like magnetic/assisted steering) or apply global NDC pointer tilt while dragging. Simulate **2-DOF lateral groove play** (`latOffset`, `latSpeed`), wall-contact Coulomb drag, parabolic mouth capture vs. high-speed rim-skip ($w_{\text{cap}} = w_0 + k v^2$), rotating-frame inertial coupling, release-anchored hover bias, and no-slip `rollQuat` 3D surface luster spin.
   - When combining a 1D edge graph with lateral offset (`centerline + latOffset * localBinormal`): (a) seat low-speed contacts (`vIn` and `maxUnblockedDrive` below threshold) in a **universal gravitational V-pocket rest** across **all** node degrees (`t = 0` and `t = 1` symmetric hold + `else { newSpeed = 0; }` axial static stop) so corners/T-walls never chatter across `edgeId` at `60–180 Hz`, and (b) bridge `90°` orthogonal `localBinormal` switches (`b_in ⊥ b_out`) with a tangent-plane `cornerBlendOffset` decayed strictly by **rolled arc-length** ($\exp(-d / L)$, never wall-clock `dt`).
4. **Diegetic Puzzle Wayfinding & Non-Invasive Live Playtest Telemetry**:
   - Keep zero-chrome puzzle specimens free of artificial keyboard brakes (`Space` hold) or HUD hints. Prevent punishing multi-face rollback diegetically via **one-way physical catch-basins/waystones**, subtle pigment temperature shifts (cool Celadon proximity wake), and a multi-stage kinetic/optical victory payoff.
   - Expose read-only `window.__<SPECIMEN>__.getState()` diagnostics (exact node/edge, `localPos`, `quaternion`, velocities) and attach non-invasively via Chrome DevTools MCP `evaluate_script` (`0` reloads or state mutations) to observe live user bottlenecks directly when playtesting.
5. **Zero Runaway Background Tasks**:
   - Only the local Vite dev server (`npm run dev`, `IsDaemon: true`) should remain running in the background. If any synchronous command (`git`, `curl`, `grep`, `bench:optical`) exceeds `WaitMsBeforeAsync` and backgrounds unexpectedly, immediately inspect or `kill` it via `manage_task` on the next turn.
   - Always pass `--exclude-dir={node_modules,.git,dist}` to recursive `grep`, and verify HTTP/DNS endpoints with `curl -sI --max-time 5` or `dig +short` rather than interactive `npx vercel` inspect commands.
6. **Initial-Load Cursor & Window-Activation Parallax Guard**:
   - Whenever a specimen's opening illusion depends on exact dead-center camera alignment (`camOffAxis === 0`), guard hover parallax in `main.ts` against stationary OS cursors parked over the viewport or macOS window-activation `(0,0) -> (realX, realY)` synthetic `pointermove` pairs: ignore hover until `>= 8 px` of intentional pointer travel after a `~350 ms` boot-settle window (bypassed immediately by `pointerdown`/`wheel`, and reset on `resetToRest`), keeping pure physics/rig controllers unit-testable.
