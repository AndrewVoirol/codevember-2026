/**
 * Codevember 2026 — Physical Mineral & Botanical Pigment Palette
 *
 * Strictly zero digital RGB neons. Every pigment is anchored in physical
 * mass-tone reflectance (R_inf), Kubelka-Munk absorption/scattering (K, S),
 * and Beer-Lambert extinction coefficients against the canonical Chalky Slate
 * (#212529) room ground and Eggshell Calcite (#f3efe6) typography.
 */

export const CHALKY_SLATE_HEX = '#212529' as const;
export const CHALKY_SLATE_SRGB: readonly [number, number, number] = [0x21 / 255, 0x25 / 255, 0x29 / 255];
export const CHALKY_SLATE_LINEAR: readonly [number, number, number] = [0.015209, 0.0185, 0.022174];

export const EGGSHELL_INK_HEX = '#f3efe6' as const;
export const EGGSHELL_INK_SRGB: readonly [number, number, number] = [0xf3 / 255, 0xef / 255, 0xe6 / 255];
export const EGGSHELL_INK_LINEAR: readonly [number, number, number] = [0.896269, 0.863157, 0.791298];

export type PigmentKey =
  | 'boneCalcite'
  | 'warmKaolin'
  | 'yellowOchre'
  | 'rawUmber'
  | 'burntSienna'
  | 'hematiteRust'
  | 'cinnabarVermilion'
  | 'realgarAmber'
  | 'verdigrisMalachite'
  | 'celadonGlauconite'
  | 'azuritePrussian'
  | 'lapisUltramarine'
  | 'mossUmbrous'
  | 'sporeSulfur';

export interface MineralPigment {
  key: PigmentKey;
  name: string;
  origin: string;
  hex: string;
  /** Linear sRGB mass-tone reflectance R_infinity in [0.005, 0.96] */
  rInfLinear: readonly [number, number, number];
  /** Kubelka-Munk scattering coefficient S (per mm film thickness) */
  scatteringS: readonly [number, number, number];
  /** Kubelka-Munk absorption coefficient K = S * (1 - R_inf)^2 / (2 * R_inf) */
  absorptionK: readonly [number, number, number];
  /** Beer-Lambert RGB extinction sigma_t (per unit optical chord) */
  extinctionRgb: readonly [number, number, number];
}

export function srgbChannelToLinear(c: number): number {
  const clamped = Math.max(0, Math.min(1, c));
  return clamped <= 0.04045 ? clamped / 12.92 : Math.pow((clamped + 0.055) / 1.055, 2.4);
}

export function hexToLinearRgb(hex: string): [number, number, number] {
  const clean = hex.replace(/^#/, '');
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  return [
    Number(srgbChannelToLinear(r).toFixed(6)),
    Number(srgbChannelToLinear(g).toFixed(6)),
    Number(srgbChannelToLinear(b).toFixed(6)),
  ];
}

/**
 * Computes Kubelka-Munk absorption K from physical mass-tone reflectance R_inf
 * and scattering S using K/S = (1 - R_inf)^2 / (2 * R_inf).
 */
export function computeKubelkaMunkK(
  rInf: readonly [number, number, number],
  scatteringS: readonly [number, number, number]
): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const r = Math.max(0.005, Math.min(0.985, rInf[i]));
    const kOverS = ((1 - r) * (1 - r)) / (2 * r);
    out[i] = Number((kOverS * scatteringS[i]).toFixed(6));
  }
  return out;
}

/**
 * Evaluates 2-flux Kubelka-Munk reflectance of a pigment film of thickness `d`
 * over a backing reflectance `backingRgb`, matching the WGSL/TSL shader contract.
 */
export function evalKubelkaMunkCpu(
  K: readonly [number, number, number],
  S: readonly [number, number, number],
  d: number,
  backingRgb: readonly [number, number, number] = CHALKY_SLATE_LINEAR
): [number, number, number] {
  const thickness = Math.max(0, d);
  const out: [number, number, number] = [0, 0, 0];

  for (let i = 0; i < 3; i++) {
    const kEff = K[i] * thickness;
    const sEff = S[i] * thickness;
    const rg = backingRgb[i];
    if (kEff + sEff < 1e-4 || sEff < 1e-5) {
      out[i] = rg;
      continue;
    }
    const a = 1 + K[i] / Math.max(S[i], 1e-5);
    const b = Math.sqrt(Math.max(1e-8, a * a - 1));
    const u = b * sEff;
    const cothU = Math.abs(u) < 1e-3 ? 1 / Math.max(u, 1e-6) + u / 3 : Math.cosh(u) / Math.sinh(u);
    const num = 1 - rg * (a - b * cothU);
    const den = a - rg + b * cothU;
    out[i] = Math.max(0, Math.min(1, num / Math.max(den, 1e-6)));
  }

  return out;
}

function definePigment(
  key: PigmentKey,
  name: string,
  origin: string,
  hex: string,
  scatteringS: readonly [number, number, number],
  extinctionRgb: readonly [number, number, number]
): MineralPigment {
  const rInfLinear = hexToLinearRgb(hex);
  const absorptionK = computeKubelkaMunkK(rInfLinear, scatteringS);
  return {
    key,
    name,
    origin,
    hex,
    rInfLinear,
    scatteringS,
    absorptionK,
    extinctionRgb,
  };
}

export const MINERAL_PALETTE: Readonly<Record<PigmentKey, MineralPigment>> = {
  boneCalcite: definePigment(
    'boneCalcite',
    'Bone Calcite',
    'Calcined calcium carbonate & iceland spar',
    '#eae3d2',
    [3.8, 3.6, 3.2],
    [0.18, 0.24, 0.38]
  ),
  warmKaolin: definePigment(
    'warmKaolin',
    'Warm Kaolin',
    'Unbleached porcelain clay & gypsum',
    '#d8cebe',
    [3.2, 3.0, 2.7],
    [0.28, 0.36, 0.52]
  ),
  yellowOchre: definePigment(
    'yellowOchre',
    'Yellow Ochre',
    'Goethite hydrated iron oxide (FeOOH)',
    '#c8963e',
    [2.4, 2.1, 1.4],
    [0.42, 0.88, 2.65]
  ),
  rawUmber: definePigment(
    'rawUmber',
    'Raw Umber',
    'Manganese-rich Cypriot iron clay',
    '#5c4a38',
    [1.6, 1.5, 1.3],
    [1.85, 2.45, 3.4]
  ),
  burntSienna: definePigment(
    'burntSienna',
    'Burnt Sienna',
    'Calcined Tuscan limonite & iron oxide',
    '#9e472a',
    [2.1, 1.6, 1.2],
    [0.55, 2.1, 3.8]
  ),
  hematiteRust: definePigment(
    'hematiteRust',
    'Hematite Rust',
    'Anhydrous ferric oxide scale (Fe2O3)',
    '#7a2e22',
    [1.9, 1.3, 1.1],
    [0.78, 2.95, 4.2]
  ),
  cinnabarVermilion: definePigment(
    'cinnabarVermilion',
    'Cinnabar Vermilion',
    'Pulverized mineral mercuric sulfide (HgS)',
    '#c93b2b',
    [2.8, 1.5, 1.2],
    [0.32, 2.8, 4.1]
  ),
  realgarAmber: definePigment(
    'realgarAmber',
    'Realgar Amber',
    'Arsenic sulfide crystal & fossilized resin',
    '#e06d2b',
    [2.6, 1.9, 1.1],
    [0.24, 1.35, 3.9]
  ),
  verdigrisMalachite: definePigment(
    'verdigrisMalachite',
    'Verdigris Malachite',
    'Basic copper carbonate mineral patina',
    '#3b7a67',
    [1.4, 2.3, 2.0],
    [2.6, 0.72, 1.15]
  ),
  celadonGlauconite: definePigment(
    'celadonGlauconite',
    'Celadon Glauconite',
    'Green earth phyllosilicate mineral glaze',
    '#7fa38c',
    [2.2, 2.7, 2.4],
    [0.95, 0.48, 0.76]
  ),
  azuritePrussian: definePigment(
    'azuritePrussian',
    'Azurite Prussian',
    'Deep copper carbonate & ferric ferrocyanide',
    '#24476e',
    [1.1, 1.6, 2.4],
    [3.6, 1.75, 0.62]
  ),
  lapisUltramarine: definePigment(
    'lapisUltramarine',
    'Lapis Ultramarine',
    'Lazurite sodium aluminosilicate',
    '#2e3d78',
    [1.2, 1.5, 2.6],
    [3.1, 2.1, 0.54]
  ),
  mossUmbrous: definePigment(
    'mossUmbrous',
    'Moss Umbrous',
    'Bryophyte chlorophyll & woodland tannin',
    '#596e3d',
    [1.6, 2.2, 1.3],
    [1.65, 0.92, 2.8]
  ),
  sporeSulfur: definePigment(
    'sporeSulfur',
    'Spore Sulfur',
    'Lycopodium pollen & native volcanic sulfur',
    '#d4b84c',
    [2.9, 2.7, 1.5],
    [0.35, 0.54, 2.4]
  ),
};

export const PIGMENT_KEYS = Object.keys(MINERAL_PALETTE) as PigmentKey[];
