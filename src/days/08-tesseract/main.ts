import { type SpecimenStage } from '../../core/stage/SpecimenStage';
import { bootTesseractConcept1 } from './concept1-schlegel-soapfilm';
import { bootTesseractConcept2 } from './concept2-unfolding-net';
import { bootTesseractConcept3 } from './concept3-slicer-tomograph';

export {
  bootTesseractConcept1,
  bootTesseractConcept2,
  bootTesseractConcept3,
};

function resolveSelectedConcept(): '1' | '2' | '3' {
  if (typeof window === 'undefined') return '3';
  const params = new URLSearchParams(window.location.search);
  const raw = (params.get('concept') ?? '3').trim().toLowerCase();
  if (raw === '1' || raw === 'a' || raw === '8a') return '1';
  if (raw === '2' || raw === 'b' || raw === '8b') return '2';
  return '3';
}

function registerConceptHotkeys(): void {
  if (typeof window === 'undefined') return;
  window.addEventListener('keydown', (ev: KeyboardEvent) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.key === '1' || ev.key === '2' || ev.key === '3') {
      const url = new URL(window.location.href);
      url.searchParams.set('concept', ev.key);
      window.location.href = url.toString();
    }
  });
}

export async function bootDay08Tesseract(): Promise<SpecimenStage> {
  registerConceptHotkeys();
  const concept = resolveSelectedConcept();
  if (concept === '2') {
    return bootTesseractConcept2();
  }
  if (concept === '3') {
    return bootTesseractConcept3();
  }
  return bootTesseractConcept1();
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay08Tesseract();
}
