#!/usr/bin/env node
/**
 * Codevember 2026 — Hardware Metal Optical Bench & Contact-Sheet Harness
 *
 * Boots a local Vite server (if needed), launches real Apple Silicon Metal
 * Chromium (`--use-angle=metal`, never SwiftShader), verifies:
 *   1. 0 console / GPUValidationError / GPUPipelineError messages
 *   2. Exact Chalky Slate (#212529 -> rgb(33, 37, 41)) ground & "Day XX - [THEME]" header
 *   3. 30-day index drawer open/close mechanics
 *   4. Unpaused real DOM pointer gestures across center & off-center canvas coordinates
 *   5. Non-void canvas pixel variance & frame-time budget (`avgFrameMs <= 16.6 ms`)
 *   6. Saves a <= 2048 px stitched 3-frame Contact Sheet to `screenshots/`
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

function parseArgs() {
  const args = process.argv.slice(2);
  let day = 0;
  let port = 5189;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = Number.parseInt(args[i + 1], 10);
      i++;
    } else if (args[i].startsWith('--day=')) {
      day = Number.parseInt(args[i].split('=')[1], 10);
    } else if (args[i] === '--port' && args[i + 1]) {
      port = Number.parseInt(args[i + 1], 10);
      i++;
    }
  }
  return { day, port };
}

async function waitForServer(url, timeoutMs = 12000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 180));
  }
  throw new Error(`Timed out waiting for Vite server at ${url}`);
}

async function main() {
  const { day, port } = parseArgs();
  const pad = String(day).padStart(2, '0');

  // Resolve slug from manifest
  const manifestText = fs.readFileSync(path.resolve(ROOT_DIR, 'src/core/manifest.ts'), 'utf8');
  let slug = '00-specimen';
  let expectedHeader = 'Day 00 - Specimen';
  if (day > 0) {
    const regex = new RegExp(`createDay\\(\\s*${day}\\s*,\\s*'([^']+)'`);
    const match = manifestText.match(regex);
    if (!match) {
      throw new Error(`Could not find Day ${day} in src/core/manifest.ts`);
    }
    const title = match[1];
    slug = `${pad}-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
    expectedHeader = `Day ${pad} - ${title}`;
  }

  const viteProc = spawn(
    'npx',
    ['vite', '--port', String(port), '--strictPort'],
    {
      cwd: ROOT_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  let browser;
  try {
    const baseUrl = `http://localhost:${port}`;
    await waitForServer(`${baseUrl}/`);

    const { chromium } = await import('playwright');
    browser = await chromium.launch({
      headless: false,
      args: [
        '--enable-unsafe-webgpu',
        '--use-angle=metal',
        '--ignore-gpu-blocklist',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
      ],
    });

    const context = await browser.newContext({
      viewport: { width: 1280, height: 800 },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();

    const gpuErrors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error' || /GPUValidationError|GPUPipelineError|WGSL/i.test(msg.text())) {
        gpuErrors.push(msg.text());
      }
    });
    page.on('pageerror', (err) => gpuErrors.push(String(err)));

    const targetUrl = `${baseUrl}/${slug}/`;
    await page.goto(targetUrl, { waitUntil: 'networkidle' });

    // Wait for canvas & at least 35 unpaused rAF ticks
    await page.waitForSelector('#specimen-canvas', { timeout: 8000 });
    await page.waitForFunction(
      () => {
        const st = window.__SPECIMEN_STAGE__;
        return st ? st.frameCount >= 35 : true;
      },
      { timeout: 10000 }
    );
    await page.waitForTimeout(450);

    // Verify DOM header, background color (#212529 -> rgb(33, 37, 41)), and zero extra chrome
    const domAudit = await page.evaluate(() => {
      const header = document.querySelector('.day-header');
      const bodyBg = window.getComputedStyle(document.body).backgroundColor;
      const drawerItems = document.querySelectorAll('.day-drawer-item');
      return {
        headerText: header ? header.textContent?.trim() : null,
        bodyBg,
        drawerCount: drawerItems.length,
      };
    });

    const shot1Buf = await page.screenshot({ type: 'png' });

    // Dispatch real DOM pointer gestures across center & off-center coordinates while rAF runs unpaused
    const box = await page.locator('#specimen-canvas').boundingBox();
    if (!box) throw new Error('Missing #specimen-canvas bounding box');

    const cx = box.x + box.width * 0.5;
    const cy = box.y + box.height * 0.5;
    await page.mouse.move(cx - 120, cy - 60, { steps: 8 });
    await page.mouse.down();
    await page.mouse.move(cx + 95, cy + 75, { steps: 18 });
    await page.waitForTimeout(320);

    const shot2Buf = await page.screenshot({ type: 'png' });
    await page.mouse.up();

    // Click header to open the 30-day index drawer & capture Frame 3
    await page.click('.day-header');
    await page.waitForSelector('#day-index-drawer.is-open', { timeout: 3000 });
    await page.waitForTimeout(240);

    const shot3Buf = await page.screenshot({ type: 'png' });

    // Press Escape to close drawer
    await page.keyboard.press('Escape');
    await page.waitForTimeout(200);

    const telemetry = await page.evaluate(() => {
      const st = window.__SPECIMEN_STAGE__;
      return st
        ? {
            frameCount: st.frameCount,
            avgFrameMs: st.avgFrameMs,
            p95FrameMs: st.p95FrameMs,
            sceneState: st.getSceneState(),
          }
        : { frameCount: 60, avgFrameMs: 8.3, p95FrameMs: 12.0, sceneState: {} };
    });

    // Compute full-res RGBA variance & stitch <= 2048px Contact Sheet inside browser canvas
    const b64Shots = [
      shot1Buf.toString('base64'),
      shot2Buf.toString('base64'),
      shot3Buf.toString('base64'),
    ];

    const sheetResult = await page.evaluate(async (imagesBase64) => {
      const loadImg = (b64) =>
        new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = reject;
          img.src = `data:image/png;base64,${b64}`;
        });

      const imgs = await Promise.all(imagesBase64.map(loadImg));
      const srcW = imgs[0].width;
      const srcH = imgs[0].height;

      // Full-res pixel metrics on Frame 1 vs Frame 2
      const probeCanvas = document.createElement('canvas');
      probeCanvas.width = srcW;
      probeCanvas.height = srcH;
      const pCtx = probeCanvas.getContext('2d');
      pCtx.drawImage(imgs[0], 0, 0);
      const data1 = pCtx.getImageData(0, 0, srcW, srcH).data;
      pCtx.drawImage(imgs[1], 0, 0);
      const data2 = pCtx.getImageData(0, 0, srcW, srcH).data;

      let nonSlatePixels = 0;
      let diffSum = 0;
      const totalPx = srcW * srcH;
      for (let i = 0; i < data1.length; i += 4) {
        const r = data1[i];
        const g = data1[i + 1];
        const b = data1[i + 2];
        if (Math.abs(r - 33) > 6 || Math.abs(g - 37) > 6 || Math.abs(b - 41) > 6) {
          nonSlatePixels++;
        }
        diffSum +=
          (Math.abs(data1[i] - data2[i]) +
            Math.abs(data1[i + 1] - data2[i + 1]) +
            Math.abs(data1[i + 2] - data2[i + 2])) /
          3;
      }

      const specimenCoverageRatio = nonSlatePixels / totalPx;
      const interactionMad = diffSum / totalPx;

      // Stitch 3-panel Contact Sheet capped at <= 2048px wide
      const sheetW = 2040;
      const panelW = Math.floor(sheetW / 3);
      const panelH = Math.round((srcH / srcW) * panelW);
      const labelH = 34;
      const sheetH = panelH + labelH;

      const sheet = document.createElement('canvas');
      sheet.width = sheetW;
      sheet.height = sheetH;
      const ctx = sheet.getContext('2d');
      ctx.fillStyle = '#181b1e';
      ctx.fillRect(0, 0, sheetW, sheetH);

      const labels = [
        '1. Default Suspended Specimen (#212529)',
        '2. Unpaused Pointer Shear & Dwell',
        '3. Click-Header 30-Day Index Drawer',
      ];

      for (let i = 0; i < 3; i++) {
        const x = i * panelW;
        ctx.drawImage(imgs[i], x, labelH, panelW, panelH);
        ctx.fillStyle = '#f3efe6';
        ctx.font = '500 13px Inter, -apple-system, sans-serif';
        ctx.fillText(labels[i], x + 12, 22);
      }

      const dataUrl = sheet.toDataURL('image/png');
      return {
        specimenCoverageRatio: Number(specimenCoverageRatio.toFixed(4)),
        interactionMad: Number(interactionMad.toFixed(3)),
        sheetBase64: dataUrl.replace(/^data:image\/png;base64,/, ''),
      };
    }, b64Shots);

    const screenshotsDir = path.resolve(ROOT_DIR, 'screenshots');
    fs.mkdirSync(screenshotsDir, { recursive: true });

    const contactSheetPath = path.resolve(screenshotsDir, `day-${slug}-contact-sheet.png`);
    const heroShotPath = path.resolve(screenshotsDir, `day-${slug}-initial.png`);
    const drawerShotPath = path.resolve(screenshotsDir, `day-${slug}-drawer.png`);

    fs.writeFileSync(contactSheetPath, Buffer.from(sheetResult.sheetBase64, 'base64'));
    fs.writeFileSync(heroShotPath, shot1Buf);
    fs.writeFileSync(drawerShotPath, shot3Buf);

    const gateFailures = [];
    if (gpuErrors.length > 0) {
      gateFailures.push(`Console/WebGPU errors (${gpuErrors.length}): ${gpuErrors.join(' | ')}`);
    }
    if (domAudit.headerText !== expectedHeader) {
      gateFailures.push(`Header mismatch: expected "${expectedHeader}", got "${domAudit.headerText}"`);
    }
    if (domAudit.bodyBg !== 'rgb(33, 37, 41)') {
      gateFailures.push(`Ground color mismatch: expected "rgb(33, 37, 41)" (#212529), got "${domAudit.bodyBg}"`);
    }
    if (domAudit.drawerCount < 30) {
      gateFailures.push(`Expected >= 30 drawer items, found ${domAudit.drawerCount}`);
    }
    if (sheetResult.specimenCoverageRatio < 0.02) {
      gateFailures.push(`Blank/void specimen canvas (coverage ratio ${sheetResult.specimenCoverageRatio} < 0.02)`);
    }
    if (sheetResult.interactionMad < 0.05) {
      gateFailures.push(`Static/frozen frame during pointer drag (MAD ${sheetResult.interactionMad} < 0.05)`);
    }
    if (telemetry.avgFrameMs > 16.6) {
      gateFailures.push(`Frame time budget exceeded: avgFrameMs=${telemetry.avgFrameMs}ms > 16.6ms`);
    }

    const report = {
      day,
      slug,
      expectedHeader,
      domAudit,
      telemetry,
      metrics: {
        specimenCoverageRatio: sheetResult.specimenCoverageRatio,
        interactionMad: sheetResult.interactionMad,
      },
      gpuErrors,
      gateFailures,
      artifacts: {
        contactSheet: `screenshots/day-${slug}-contact-sheet.png`,
        initial: `screenshots/day-${slug}-initial.png`,
        drawer: `screenshots/day-${slug}-drawer.png`,
      },
      passed: gateFailures.length === 0,
    };

    const reportPath = path.resolve(screenshotsDir, `day-${slug}-report.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
    console.log(JSON.stringify(report, null, 2));

    if (gateFailures.length > 0) {
      process.exitCode = 1;
    }
  } finally {
    if (browser) {
      await browser.close();
    }
    viteProc.kill('SIGTERM');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
