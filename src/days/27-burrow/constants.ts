import { MINERAL_PALETTE, type PigmentKey } from '../../core/palette';

/**
 * Day 27 - Burrow: Physical Dimensions, Granular Simulation & Colony Constants
 *
 * Narrow vertical museum ant-farm observation slice (~1.76 W x 1.36 H x 0.24 D)
 * framed in oiled dark raw-umber walnut, satin-brass corner brackets & slotted screws,
 * and beveled front/back glass panes.
 */

// Outer frame dimensions (world units)
export const FRAME_OUTER_W = 1.76;
export const FRAME_OUTER_H = 1.36;
export const FRAME_OUTER_D = 0.24;

// Inner sand cavity dimensions (between walnut stiles & front/back glass)
export const CAVITY_W = 1.56;
export const CAVITY_H = 1.16;
export const CAVITY_D = 0.136;

export const CAVITY_HALF_W = CAVITY_W * 0.5; // 0.78
export const CAVITY_HALF_H = CAVITY_H * 0.5; // 0.58
export const CAVITY_HALF_D = CAVITY_D * 0.5; // 0.068

// Glass observation panes seated cleanly inside satin-brass channels
export const GLASS_PANE_W = 1.584;
export const GLASS_PANE_H = 1.184;
export const GLASS_PANE_THICK = 0.012;
export const GLASS_CENTER_Z = CAVITY_HALF_D + GLASS_PANE_THICK * 0.5 + 0.002; // 0.076

// Recessed interior tunnel back-wall offset (physically behind ants at z ~ 0)
export const BACKWALL_Z = 0.056;

// Granular cellular grid resolution (x = 0..W-1 left->right, y = 0..H-1 bottom->top)
export const GRID_W = 300;
export const GRID_H = 220;
export const GRID_CELLS = GRID_W * GRID_H;

// Half-resolution stigmergic pheromone & disturbance field dimensions (150 x 110)
export const COARSE_W = GRID_W >> 1;
export const COARSE_H = GRID_H >> 1;
export const COARSE_CELLS = COARSE_W * COARSE_H;

// Cell occupancy types
export const CELL_EMPTY = 0;
export const CELL_PACKED_SAND = 1;
export const CELL_LOOSE_SAND = 2;
export const CELL_PEBBLE = 3;

// Air classification for optical transparency vs. shadowed subterranean cavities
export const AIR_SOLID = 0;
export const AIR_SKY = 1;
export const AIR_CAVITY = 2;

// Arching & shear-fault collapse physics parameters (grid cell units)
/** Maximum horizontal unsupported ceiling span (cells) that moist-compacted sand can arch cleanly */
export const ARCH_SPAN_LIMIT = 18;
/** Extra span support when an arch shoulder abuts a hard quartz/basalt pebble or frame wall */
export const PEBBLE_ABUTMENT_BONUS = 5;
/** Maximum unsupported single-side cantilever overhang (cells) */
export const MAX_CANTILEVER_SPAN = 5;
/** Minimum solid roof overburden thickness (cells) needed to sustain full arch compression */
export const MIN_FULL_ARCH_ROOF = 7;

// Default starter surface horizon (fraction of GRID_H, leaving top ~20% clear open air)
export const SURFACE_FRAC = 0.8;

/**
 * Physical mineral pigment table for sand strata & river pebbles (strictly zero digital neons).
 * Indices 0..7 are stratified mineral sands; 8..9 are embedded hard river pebbles.
 */
export interface SandPigmentDef {
  id: number;
  name: string;
  key: PigmentKey | 'quartzPebble' | 'basaltPebble';
  /** Linear sRGB base reflectance */
  rgbLinear: readonly [number, number, number];
  /** Relative grain roughness / contrast variation */
  grainContrast: number;
}

function scaleLinear(
  rgb: readonly [number, number, number],
  rMul: number,
  gMul = rMul,
  bMul = rMul
): readonly [number, number, number] {
  return [
    Math.min(0.96, Math.max(0.01, rgb[0] * rMul)),
    Math.min(0.96, Math.max(0.01, rgb[1] * gMul)),
    Math.min(0.96, Math.max(0.01, rgb[2] * bMul)),
  ];
}

export const SAND_PIGMENTS: readonly SandPigmentDef[] = [
  {
    id: 0,
    name: 'Bone Calcite Sand',
    key: 'boneCalcite',
    rgbLinear: scaleLinear(MINERAL_PALETTE.boneCalcite.rInfLinear, 0.92),
    grainContrast: 0.12,
  },
  {
    id: 1,
    name: 'Warm Kaolin Silt',
    key: 'warmKaolin',
    rgbLinear: scaleLinear(MINERAL_PALETTE.warmKaolin.rInfLinear, 0.94),
    grainContrast: 0.13,
  },
  {
    id: 2,
    name: 'Yellow Ochre Sand',
    key: 'yellowOchre',
    rgbLinear: scaleLinear(MINERAL_PALETTE.yellowOchre.rInfLinear, 1.02),
    grainContrast: 0.16,
  },
  {
    id: 3,
    name: 'Burnt Sienna Sand',
    key: 'burntSienna',
    rgbLinear: scaleLinear(MINERAL_PALETTE.burntSienna.rInfLinear, 1.08),
    grainContrast: 0.18,
  },
  {
    id: 4,
    name: 'Raw Umber Loam',
    key: 'rawUmber',
    rgbLinear: scaleLinear(MINERAL_PALETTE.rawUmber.rInfLinear, 1.05),
    grainContrast: 0.15,
  },
  {
    id: 5,
    name: 'Hematite Rust Seam',
    key: 'hematiteRust',
    rgbLinear: scaleLinear(MINERAL_PALETTE.hematiteRust.rInfLinear, 1.12),
    grainContrast: 0.17,
  },
  {
    id: 6,
    name: 'Moss Umbrous Humus',
    key: 'mossUmbrous',
    rgbLinear: scaleLinear(MINERAL_PALETTE.mossUmbrous.rInfLinear, 0.88),
    grainContrast: 0.15,
  },
  {
    id: 7,
    name: 'Verdigris Mineral Vein',
    key: 'verdigrisMalachite',
    rgbLinear: scaleLinear(MINERAL_PALETTE.verdigrisMalachite.rInfLinear, 0.78),
    grainContrast: 0.14,
  },
  {
    id: 8,
    name: 'Quartz River Pebble',
    key: 'quartzPebble',
    rgbLinear: scaleLinear(MINERAL_PALETTE.warmKaolin.rInfLinear, 0.84, 0.83, 0.82),
    grainContrast: 0.11,
  },
  {
    id: 9,
    name: 'Basalt River Pebble',
    key: 'basaltPebble',
    rgbLinear: scaleLinear(MINERAL_PALETTE.rawUmber.rInfLinear, 0.48, 0.50, 0.54),
    grainContrast: 0.09,
  },
];

/** Active worker ant colony population (kept exported as DIGGER_COUNT for manifest/test parity) */
export const DIGGER_COUNT = 12;
export const ANT_COUNT = DIGGER_COUNT;

/** Live 3D instanced tumbling mineral granule & avalanche dust pool capacities */
export const MAX_3D_GRAINS = 2200;
export const MAX_DUST_MOTES = 120;

/** Convert linear [0..1] channel to sRGB byte [0..255] for RGBA8 SRGB DataTexture */
export function linearToSrgbByte(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(s * 255)));
}
