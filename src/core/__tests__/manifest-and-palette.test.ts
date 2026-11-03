import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import {
  CALIBRATION_DAY,
  DAYS_MANIFEST,
  getAdjacentCompletedDays,
  getDayByNumber,
  getDayBySlug,
  getLatestActiveDay,
} from '../manifest';
import {
  CHALKY_SLATE_HEX,
  CHALKY_SLATE_LINEAR,
  EGGSHELL_INK_HEX,
  MINERAL_PALETTE,
  PIGMENT_KEYS,
  evalKubelkaMunkCpu,
} from '../palette';

const EXPECTED_THEMES = [
  'Fold',
  'Shatter',
  'Melt',
  'Tangle',
  'Moss',
  'Orrery',
  'Hollow',
  'Tesseract',
  'Membrane',
  'Caustic',
  'Rust',
  'Feather',
  'Labyrinth',
  'Tension',
  'Mirage',
  'Spore',
  'Kelp',
  'Pendulum',
  'Erosion',
  'Cocoon',
  'Bolt',
  'Echo',
  'Magma',
  'Nest',
  'Veil',
  'Anamorph',
  'Burrow',
  'Hive',
  'Portal',
  'Dissolve',
];

describe('Codevember 2026 Manifest & 30-Theme Contract', () => {
  it('contains all 30 daily themes in exact order with canonical "Day XX - [THEME]" headers', () => {
    expect(DAYS_MANIFEST).toHaveLength(30);

    for (let i = 0; i < 30; i++) {
      const entry = DAYS_MANIFEST[i];
      const expectedDay = i + 1;
      const expectedPad = String(expectedDay).padStart(2, '0');
      const expectedTitle = EXPECTED_THEMES[i];

      expect(entry.day).toBe(expectedDay);
      expect(entry.pad).toBe(expectedPad);
      expect(entry.title).toBe(expectedTitle);
      expect(entry.headerLabel).toBe(`Day ${expectedPad} - ${expectedTitle}`);
      expect(PIGMENT_KEYS).toContain(entry.primaryPigment);
      expect(PIGMENT_KEYS).toContain(entry.secondaryPigment);
      expect(getDayByNumber(expectedDay)).toBe(entry);
      expect(getDayBySlug(entry.slug)).toBe(entry);
    }
  });

  it('maintains 1:1 parity between DAYS_MANIFEST and SPECIMEN_CATALOG.md', () => {
    const catalogPath = path.resolve(process.cwd(), 'SPECIMEN_CATALOG.md');
    const catalogMd = fs.readFileSync(catalogPath, 'utf8');

    for (const entry of DAYS_MANIFEST) {
      expect(catalogMd).toContain(`\`${entry.slug}\``);
      expect(catalogMd).toContain(`**${entry.title}**`);
      expect(catalogMd).toContain(`\`${entry.primaryPigment}\``);
      expect(catalogMd).toContain(`\`${entry.secondaryPigment}\``);
    }
  });

  it('resolves getLatestActiveDay() and adjacent navigation cleanly', () => {
    const active = getLatestActiveDay();
    expect(active.day).toBeGreaterThanOrEqual(0);
    expect(CALIBRATION_DAY.headerLabel).toBe('Day 00 - Specimen');

    const adj = getAdjacentCompletedDays(0);
    expect(adj.prev).toBeNull();
  });
});

describe('Physical Mineral & Botanical Pigment Palette & Kubelka-Munk Optics', () => {
  it('locks Chalky Slate (#212529) and Eggshell (#f3efe6) surface constants', () => {
    expect(CHALKY_SLATE_HEX).toBe('#212529');
    expect(EGGSHELL_INK_HEX).toBe('#f3efe6');
  });

  it('produces finite, bounded Kubelka-Munk reflectance across all 14 pigments and film thicknesses', () => {
    const thicknesses = [0, 1e-6, 1e-4, 0.05, 0.25, 0.75, 2.5, 12.0];

    for (const key of PIGMENT_KEYS) {
      const p = MINERAL_PALETTE[key];
      expect(p.rInfLinear.every((v) => Number.isFinite(v) && v > 0 && v < 1)).toBe(true);
      expect(p.absorptionK.every((v) => Number.isFinite(v) && v > 0)).toBe(true);
      expect(p.scatteringS.every((v) => Number.isFinite(v) && v > 0)).toBe(true);

      // At zero thickness, Kubelka-Munk must return backing reflectance exactly
      const atZero = evalKubelkaMunkCpu(p.absorptionK, p.scatteringS, 0, CHALKY_SLATE_LINEAR);
      expect(atZero).toEqual([...CHALKY_SLATE_LINEAR]);

      for (const d of thicknesses) {
        const rgb = evalKubelkaMunkCpu(p.absorptionK, p.scatteringS, d, CHALKY_SLATE_LINEAR);
        for (const c of rgb) {
          expect(Number.isFinite(c)).toBe(true);
          expect(c).toBeGreaterThanOrEqual(0);
          expect(c).toBeLessThanOrEqual(1);
        }
      }

      // At thick mass-tone (d = 12), reflectance converges toward rInfLinear
      const atMassTone = evalKubelkaMunkCpu(p.absorptionK, p.scatteringS, 12.0, CHALKY_SLATE_LINEAR);
      for (let i = 0; i < 3; i++) {
        expect(Math.abs(atMassTone[i] - p.rInfLinear[i])).toBeLessThan(0.02);
      }
    }
  });
});

describe('CLI Scaffolder (scripts/new-day.mjs --dry-run)', () => {
  it('scaffolds TSL and WGSL day templates in dry-run mode without touching disk', () => {
    const out01 = JSON.parse(
      execFileSync('node', ['scripts/new-day.mjs', '01', '--dry-run'], {
        cwd: process.cwd(),
        encoding: 'utf8',
      })
    );
    expect(out01).toEqual({
      day: 1,
      slug: '01-fold',
      title: 'Fold',
      substrate: 'tsl',
      files: ['01-fold/index.html', 'src/days/01-fold/main.ts'],
    });

    const out10 = JSON.parse(
      execFileSync('node', ['scripts/new-day.mjs', '10', '--substrate=wgsl', '--dry-run'], {
        cwd: process.cwd(),
        encoding: 'utf8',
      })
    );
    expect(out10).toEqual({
      day: 10,
      slug: '10-caustic',
      title: 'Caustic',
      substrate: 'wgsl',
      files: ['10-caustic/index.html', 'src/days/10-caustic/main.ts'],
    });
  });
});
