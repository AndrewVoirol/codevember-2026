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

Inspect the generated contact sheet at `screenshots/day-<XX>-<slug>-contact-sheet.png` using `read_file` to visually verify the specimen framing, mineral palette contrast against `#212529`, and unpaused pointer interaction response.

---

## Step 5: Promote Day Status to `'complete'` & Deploy

Once verified with `0` console/WebGPU errors and `avgFrameMs <= 16.6 ms`:
1. Update the day's `status` from `'upcoming'` (or `'in-progress'`) to `'complete'` in [`src/core/manifest.ts`](../../../src/core/manifest.ts) and update [`SPECIMEN_CATALOG.md`](../../../SPECIMEN_CATALOG.md) if the specimen concept evolved during design.
2. Because [`src/root.ts`](../../../src/root.ts) calls `getLatestActiveDay()` (and routes `codevember-2026-day-XX.andrewvoirol.com` directly to Day `XX`), promoting Day `XX` to `'complete'` automatically makes `/` boot Day `XX` and unlocks its link in the 30-day header drawer.
3. Commit and `git push` (all 30 `codevember-2026-day-XX.andrewvoirol.com` Cloudflare CNAMEs and Vercel domains are pre-provisioned; `git push` auto-deploys `https://codevember-2026.vercel.app`, `https://codevember-2026.andrewvoirol.com/<XX>-<slug>/`, and `https://codevember-2026-day-<XX>.andrewvoirol.com` with zero DNS steps).
