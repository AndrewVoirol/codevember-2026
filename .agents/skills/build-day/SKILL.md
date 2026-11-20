---
name: build-day
description: >-
  Chat-first daily kickoff, inspiration ingestion, MPA scaffolding, WebGPU/TSL
  specimen implementation, and hardware Metal optical-bench verification workflow
  for Codevember 2026 (code-vemeber-26). Trigger whenever the user says "Day XX",
  names one of the 30 Codevember themes (Fold..Dissolve), or pastes an inspiration
  image/URL/prompt for the next daily piece.
---

# Codevember 2026 — Daily Specimen Build Workflow (`build-day`)

Always read `/Users/andrewvoirol/.gemini/config/skills/webgpu-interactive-simulation/SKILL.md` and the workspace [`AGENTS.md`](../../../AGENTS.md) on Turn 0 before generating or modifying shaders.

---

## Step 1: Ingest Inspiration & Align on the Specimen Brief

1. **Lookup Catalog Baseline**:
   - Read the target day's entry in [`SPECIMEN_CATALOG.md`](../../../SPECIMEN_CATALOG.md) and [`src/core/manifest.ts`](../../../src/core/manifest.ts) (`day`, `slug`, `title`, `substrate`, `primaryPigment`, `secondaryPigment`, `specimenSummary`, `pointerInteraction`).
   - Check if the user pasted an image, video, shader link, or custom concept in chat (or dropped files in `inspirations/XX-<slug>/`).
2. **Rapid 1-Question Alignment (Only If Needed)**:
   - If the user already specified what they want to build, proceed directly to **Step 2**.
   - If the user only said `"Let's do Day 01 - Fold"`, present the pre-seeded concept from `SPECIMEN_CATALOG.md` alongside 1–2 tactile variations using `ask_question` (unbiased framing, no `(Recommended)` prefix, max 3 options).
3. **Visual-First Concept Renders & Fused Mockup Lock (Before `/plan`, `/swarm`, or Rewrites)**:
   - When exploring multiple architectural directions, avoiding crowded tropes, or pivoting a build, spawn 3–5 parallel `image-generator` renders on the `#212529` Chalky Slate studio stage rather than pitching text-only descriptions or jumping straight into a 1,500-line 3D build.
   - Whenever the user fuses two concepts (`Concept A + Concept B`) or invokes `/plan` + `/swarm`, generate a dedicated **fused Hero + Macro Detail mockup pair** *first* and embed it at the top of `implementation_plan.md` + every `/swarm` worker brief so geometry, material relief, and verifier parity share one locked visual reference.

---

## Step 2: Scaffold the MPA Entrypoint (`npm run day:new`)

Run the idempotent generator for the target day (`01`..`30`):

```bash
npm run day:new <XX> -- --substrate=<tsl|tsl-compute|rapier|wgsl>
```

This creates:
- `<XX>-<slug>/index.html` (standalone MPA route served at `/<XX>-<slug>/`)
- `src/days/<XX>-<slug>/main.ts` (pre-wired to `SpecimenStage` or `RawWebGPUStage`, `DayShell`, and the day's mineral pigments from `src/core/palette.ts`)

---

## Step 3: Implement the Suspended Specimen & Tactile Physics

Enforce the three Codevember 2026 identity invariants:

1. **Minimal Chalky Slate Interface**:
   - Background is strictly **Chalky Slate (`#212529`)** via `SpecimenStage` (`chalkySlateBackdropNode`) or `RawWebGPUStage.clearColor`.
   - Top-left header is strictly `"Day XX - [THEME]"` in **Eggshell (`#f3efe6`)** via `mountDayShell`.
   - **Zero UI chrome** (no sliders, FPS counters, pill bars, or subtitles) unless the user explicitly asks for a specific control on that day.
2. **Suspended Specimen Framing**:
   - Stage the subject as a physical specimen, core sample, or optical apparatus suspended in the center of the `#212529` space (`stage.specimenGroup`), illuminated by the studio softbox + calcite rim rig.
3. **Physical Mineral & Botanical Pigments + Tactile Pointer Coupling**:
   - Pull all colors from [`src/core/palette.ts`](../../../src/core/palette.ts) using `createPigmentUniforms()`, `kubelkaMunkGlaze()`, or `beerLambertTransmittance()`.
   - Couple `stage.pointer` (`uPointerWorld`, `uPointerVelocity`, `uPointerMeta` `[speed, dwellSec, impulse, isDown]`) directly into the physical simulation or vertex/fragment deformation—never ship a passive screensaver.
   - Follow all WGSL/TSL guardrails: uniform control flow for derivatives/implicit-LOD textures, zero per-frame allocations inside `onTick`/`setAnimationLoop`, and zero polar star harmonics $\cos(m\theta)$ ($m \ge 3$).

---

## Step 4: Static Linting, Unit Tests & Live Metal Optical Bench

Before marking a day complete, run all four verification gates:

```bash
# 1. Static WGSL & struct layout verification
npm run lint:wgsl

# 2. Manifest & palette contract tests
npm test

# 3. Production MPA bundle verification
npm run build

# 4. Real Apple Silicon Metal Chromium verification + Contact Sheet capture
npm run bench:optical -- --day <XX>
```

Inspect `screenshots/day-<XX>-<slug>-initial.png` and `screenshots/day-<XX>-<slug>-contact-sheet.png` using `read_file` side-by-side against the locked concept render to verify: (a) relative object scale ratios and framing on `#212529`, (b) surface material/texture fidelity (no dark-void metal collapse or cloudy low-freq noise), (c) flush surface seating and mechanical joint alignment, and (d) unpaused pointer interaction response.
- **Additional Gates for Multi-Face / Puzzle / `/swarm` Builds**:
  1. **`1:1` Native-Pixel Material & Seam Crop**: Alongside `2×2` full-specimen contact sheets (which hide smooth/bitmap stone and octant DC seams), capture at least one `1:1` Retina micro-crop across carved ridges, vertical walls, and chamfers.
  2. **Bounded-Rate Playability vs. BFS Teleport**: Verify interactive puzzles under bounded angular-rate tilting (`<= 2.5 rad/s`), never solely via instantaneous per-substep gravity-teleport helpers (`stepTowardSolution`).
  3. **Multi-Orientation Corner/Junction Rest Settle**: Run `>= 90` unpaused frames with gravity parked into L-corners, T-junctions, and catch-basins asserting `edgeFlips === 0`, `maxPosDelta === 0`, and `speed === 0` after settling (`C^0` continuous position across all junction handoffs).
  4. **4-Angle Macro, Interior & Rear-Orbit Audit (3D Hollow / Rotatable Specimens)**: Alongside the default front `bench:optical` pass, inspect a stitched 4-view sheet covering top joinery macro, primary seam/texture macro, underside/interior cavity, and a live rear-orbit DOM drag so inside-out lathes, rear-panel UV seams, inverted orbit drags, and back-face shading defects are caught before promotion.

---

## Step 5: Promote Day Status to `'complete'` & Deploy

Once verified with `0` console/WebGPU errors and `avgFrameMs <= 16.6 ms`:
1. Update the day's `status` from `'upcoming'` (or `'in-progress'`) to `'complete'` in [`src/core/manifest.ts`](../../../src/core/manifest.ts) and update [`SPECIMEN_CATALOG.md`](../../../SPECIMEN_CATALOG.md) if the specimen concept evolved during design.
2. Because [`src/root.ts`](../../../src/root.ts) calls `getLatestActiveDay()` (and routes `codevember-2026-day-XX.andrewvoirol.com` directly to Day `XX`), promoting Day `XX` to `'complete'` automatically makes `/` boot Day `XX` and unlocks its link in the 30-day header drawer.
3. Commit and `git push` (all 30 `codevember-2026-day-XX.andrewvoirol.com` Cloudflare CNAMEs and Vercel domains are pre-provisioned; `git push` auto-deploys `https://codevember-2026.vercel.app`, `https://codevember-2026.andrewvoirol.com/<XX>-<slug>/`, and `https://codevember-2026-day-<XX>.andrewvoirol.com` with zero DNS steps).
4. **Concurrent Multi-Day Shared-Checkout Safety**:
   - Before running `bench:optical` or committing Day `<XX>`, check `git status -s` and `git diff src/core/manifest.ts SPECIMEN_CATALOG.md` for concurrent uncommitted sibling days.
   - Never run `git add -A` or stage another uncommitted day's `'complete'` hunk in `manifest.ts` (which breaks production `/` `getLatestActiveDay()` when its `src/days/` module is untracked, and bakes an unlocked future day into `day-<XX>-drawer.png`). Save sibling-day hunks to `<ArtifactDir>/scratch/` during `bench:optical` + explicit-path `git add` + `git commit`, then immediately re-apply the sibling day's working-tree patch.

---

## Step 6: Deterministic 60fps Lockstep Demo Video Capture (When Requested)

When recording a `1920×1080` `60 fps` demo `.mp4` (`exports/day-<XX>-<slug>-demo.mp4`) from real Apple Silicon Metal Chromium (`headless: false`, `deviceScaleFactor: 1`) including the `"Day XX - [THEME]"` DOM header:

1. **Lockstep `rAF` + `performance.now()` Harness (`page.addInitScript`)**:
   - Keep native `requestAnimationFrame` active during WebGPU boot until `window.__SPECIMEN_STAGE__.frameCount >= 12`, then engage lockstep (`__ENABLE_LOCKSTEP__`) and step exact `1000 / 60 ms` frames (`__STEP_FRAME__`), waiting one native `rAF` per step so Metal presents before capture.
   - Step `>= 60` silent settle frames before Frame 0 (after pressing reset / centering pointer) and assert exact resting camera coordinates so Frame 0 has zero residual exponential damper drift.
2. **OS-Cursor Isolation Gate (Prevent Real-Mouse Contamination)**:
   - Because `headless: false` opens a visible desktop window on macOS, install a capture-phase `window.addEventListener('pointermove', e => { if (!window.__ALLOW_POINTER__) e.stopImmediatePropagation(); }, true)` in `addInitScript` and toggle `window.__ALLOW_POINTER__ = true` strictly around script-driven `page.mouse.move(...)` calls.
3. **Fast Q98 MJPEG Pipe → Standard TV-Range BT.709 H.264**:
   - Pipe `page.screenshot({ type: 'jpeg', quality: 98 })` into `ffmpeg -f image2pipe -vcodec mjpeg -r 60 -i -` (`~8×` faster than PNG, `~40 s` for `960` frames, zero dropped frames) and normalize MJPEG's full-range `yuvj420p` to standard TV-range `yuv420p` BT.709 for QuickTime/Safari/X parity:
     ```bash
     -c:v libx264 -preset slow -crf 16 \
     -vf "scale=in_range=full:out_range=tv,format=yuv420p" \
     -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
     -x264-params colorprim=bt709:transfer=bt709:colormatrix=bt709 \
     -movflags +faststart
     ```
4. **Exact Frame-Indexed Visual Audit (Never `read_file` Multi-MB MP4s)**:
   - Verify stream tags via `ffprobe` (`pix_fmt=yuv420p`, `color_range=tv`, `bt709`, `60/1`, exact `nb_frames`), and inspect a `<= 200 KB` 4×3 timeline JPEG built with exact frame indices (`-vf "select='eq(n,0)+eq(n,90)+...',setpts=N/TB,scale=480:270,tile=4x3"`) rather than `fps=` rounding so Frame 0 and loop seams are audited cleanly.
