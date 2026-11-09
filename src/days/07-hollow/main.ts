import { SpecimenStage } from '../../core/stage/SpecimenStage';
import { bootHollowConcept1 } from './concept1-geode';
import { bootHollowConcept2 } from './concept2-trabecular';
import { bootHollowConcept3 } from './concept3-nautilus';

/**
 * Day 07 — Hollow: Multi-Concept Router
 *
 * Supports 3 full interactive concepts selectable via URL query `?concept=1|2|3`
 * (or keyboard shortcuts `1`, `2`, `3`):
 * - `?concept=1` (Default): Cleaved Fortification Agate & Cinnabar-Quartz Druzy Geode Nodule
 * - `?concept=2`: Calcified Trabecular Osteon & Karst Cavern with Live Surgical Clip-Sphere
 * - `?concept=3`: Sagittal Nautilus Phragmocone & Siphuncle Resonance Chamber
 */
export function resolveHollowConcept(): '1' | '2' | '3' {
  if (typeof window === 'undefined') return '2';
  const raw = new URLSearchParams(window.location.search).get('concept')?.toLowerCase();
  if (raw === '1' || raw === 'a' || raw === 'geode') return '1';
  if (raw === '3' || raw === 'c' || raw === 'nautilus') return '3';
  return '2';
}

export async function bootDay07Hollow(): Promise<SpecimenStage> {
  const concept = resolveHollowConcept();

  if (typeof window !== 'undefined') {
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      if (ev.key === '1' || ev.key === '2' || ev.key === '3') {
        const url = new URL(window.location.href);
        if (url.searchParams.get('concept') !== ev.key) {
          url.searchParams.set('concept', ev.key);
          window.location.href = url.toString();
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
  }

  let stage: SpecimenStage;
  if (concept === '2') {
    stage = await bootHollowConcept2();
  } else if (concept === '3') {
    stage = await bootHollowConcept3();
  } else {
    stage = await bootHollowConcept1();
  }

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay07Hollow();
}
