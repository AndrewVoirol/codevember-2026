import { type PigmentKey } from './palette';

export type DaySubstrate = 'tsl' | 'tsl-compute' | 'rapier' | 'wgsl';
export type DayStatus = 'upcoming' | 'in-progress' | 'complete';

export interface DayEntry {
  /** 1..30 (0 for calibration specimen) */
  day: number;
  /** Zero-padded two-digit string ('00'..'30') */
  pad: string;
  /** URL & folder slug ('01-fold'..'30-dissolve') */
  slug: string;
  /** Theme title ('Fold'..'Dissolve') */
  title: string;
  /** Header text rendered in top-left ('Day 01 - Fold') */
  headerLabel: string;
  /** Current completion state */
  status: DayStatus;
  /** Recommended GPU rendering substrate */
  substrate: DaySubstrate;
  /** Primary physical mineral pigment */
  primaryPigment: PigmentKey;
  /** Secondary accent / shadow mineral pigment */
  secondaryPigment: PigmentKey;
  /** Suspended specimen concept summary */
  specimenSummary: string;
  /** Tactile pointer-field interaction summary */
  pointerInteraction: string;
}

function createDay(
  day: number,
  title: string,
  status: DayStatus,
  substrate: DaySubstrate,
  primaryPigment: PigmentKey,
  secondaryPigment: PigmentKey,
  specimenSummary: string,
  pointerInteraction: string
): DayEntry {
  const pad = String(day).padStart(2, '0');
  const slug = `${pad}-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
  return {
    day,
    pad,
    slug,
    title,
    headerLabel: `Day ${pad} - ${title}`,
    status,
    substrate,
    primaryPigment,
    secondaryPigment,
    specimenSummary,
    pointerInteraction,
  };
}

export const CALIBRATION_DAY: DayEntry = createDay(
  0,
  'Specimen',
  'complete',
  'tsl',
  'celadonGlauconite',
  'boneCalcite',
  'Suspended gyroid-foliated calcite & celadon mineral core on the studio plinth',
  'Pointer velocity shears foliation ridges; press-and-hold injects warm realgar amber resonance'
);

export const DAYS_MANIFEST: readonly DayEntry[] = [
  createDay(
    1,
    'Fold',
    'complete',
    'tsl',
    'boneCalcite',
    'rawUmber',
    'Diagonal articulated Bone Calcite & Raw Umber rigid-slat panel that hinges 180° over on itself under gravity',
    'Click to release the gravity fold cascade; drag toward any corner or the top edge to unroll back into planar mode'
  ),
  createDay(
    2,
    'Shatter',
    'complete',
    'rapier',
    'azuritePrussian',
    'realgarAmber',
    'Suspended Azurite-slate Kintsugi monolith fracturing along two-tier 3D Voronoi cleavage planes held in tension by golden-ochre resin threads',
    'Click to shatter the monolith into primary shards, then click any individual shard to trigger a second-tier fracture; drag inward or click empty space to mend'
  ),
  createDay(
    3,
    'Melt',
    'complete',
    'tsl',
    'realgarAmber',
    'burntSienna',
    'Diagonal-cut Warm Bone Calcite & Burnt Sienna sandwich block oozing and stretching a curtain of molten Realgar-Amber & Cinnabar strings across the 45° gap',
    'Drag to squeeze molten amber out the diagonal seam or pull the wedges apart to stretch and neck viscoelastic strings; click any taut strand to snap it'
  ),
  createDay(
    4,
    'Tangle',
    'complete',
    'tsl',
    'rawUmber',
    'realgarAmber',
    'Smoky polycarbonate MIXTAPE Compact Cassette specimen with a hexagonal yellow-ochre #2 pencil wedged in the right 6-toothed drive hub and a multi-loop bird’s-nest tangle of glossy ferric-oxide tape cascading down the left flank and bottom',
    'Grab and crank the hexagonal pencil around the right drive hub (or click) to wind the tangled tape back into the cassette; drag or sweep through the spilled ribbon loops to pull more slack and tangle it further'
  ),
  createDay(
    5,
    'Moss',
    'complete',
    'tsl',
    'mossUmbrous',
    'yellowOchre',
    'Seven-pillar stepped polished veined-marble and dark-rock cluster where poured translucent green liquid naturally pools and cascades, subtly shifts color into an olive-ochre mold biofilm, and blooms into high-precision bryophyte moss',
    'Click any pillar to pour translucent green liquid (click repeatedly or hold to pour more and cascade across lower tiers); drag to rotate and scroll/pinch to zoom in and explore the mossing'
  ),
  createDay(
    6,
    'Orrery',
    'upcoming',
    'tsl',
    'yellowOchre',
    'lapisUltramarine',
    'Precision oxidized-brass & lapis planetary escapement mechanism with eccentric epicyclic rings',
    'Pointer drag winds the mainspring differential, accelerating orbital harmonics and eclipse alignments'
  ),
  createDay(
    7,
    'Hollow',
    'upcoming',
    'tsl',
    'warmKaolin',
    'cinnabarVermilion',
    'Seven-shell carved Canton mineral-ivory puzzle ball (Guigongqiu) with concentric counter-rotating rose-window spheres enclosing a glowing cinnabar-and-amber filigree lantern core',
    'Drag to spin the counter-rotating concentric shell differential; click or dwell to align the pierced circular windows in syzygy down the sightline to the hollow core'
  ),
  createDay(
    8,
    'Tesseract',
    'complete',
    'tsl',
    'lapisUltramarine',
    'boneCalcite',
    '4D hyperplane crystallographic tomograph slicing a rotating tesseract into a multi-mineral Euclidean fortification-agate polyhedron pinned to 32 tubular 4D struts with cinnabar piercing calipers',
    'Vertical drag sweeps the w = w0 slicing hyperplane along the 4D struts; horizontal drag orbits and tilts the 4D crystal; click cycles canonical 4D symmetry axes'
  ),
  createDay(
    9,
    'Membrane',
    'upcoming',
    'tsl-compute',
    'celadonGlauconite',
    'cinnabarVermilion',
    'Bilayer lipid & soap-film minimal surface stretched across a warped wire loop with thin-film interference',
    'Pointer pressure indents the elastic membrane and advects Kubelka-Munk swirl thickness bands'
  ),
  createDay(
    10,
    'Caustic',
    'upcoming',
    'wgsl',
    'celadonGlauconite',
    'realgarAmber',
    'Suspended optical glass lens & water cuvette projecting spectral photon caustics onto a calcite receiver plate',
    'Pointer perturbs the free meniscus and tilts the glass element, focusing razor-sharp caustic cusps'
  ),
  createDay(
    11,
    'Rust',
    'upcoming',
    'tsl-compute',
    'hematiteRust',
    'verdigrisMalachite',
    'Galvanic iron-and-copper mechanical plate undergoing live reaction-diffusion oxidation and flaking scale',
    'Pointer scrubs away loose ferric scale or deposits electrolyte droplets that nucleate fresh rust blooms'
  ),
  createDay(
    12,
    'Feather',
    'upcoming',
    'tsl',
    'boneCalcite',
    'azuritePrussian',
    'Single macro avian flight feather specimen with interlocking barbules and structural interference sheen',
    'Pointer breeze parts and unzips adjacent barbs; smoothing along the rachis zips the vane back together'
  ),
  createDay(
    13,
    'Labyrinth',
    'upcoming',
    'tsl',
    'rawUmber',
    'yellowOchre',
    'Carved monolithic limestone maze core with liquid mercury-ochre streams finding shortest geodesic paths',
    'Tilting the specimen with the pointer tips the gravity vector, sloshing the stream through new corridors'
  ),
  createDay(
    14,
    'Tension',
    'upcoming',
    'tsl-compute',
    'cinnabarVermilion',
    'boneCalcite',
    'Tensegrity prism of taut cinnabar silk cables and floating calcite compression struts near buckling threshold',
    'Plucking individual tension cables excites standing transverse string harmonics across the whole rig'
  ),
  createDay(
    15,
    'Mirage',
    'upcoming',
    'wgsl',
    'realgarAmber',
    'lapisUltramarine',
    'Non-linear Fermat thermal boundary-layer slab bending rays into inverted inferior & Fata Morgana mirages',
    'Pointer injects rising thermal plumes that warp, invert, and stack refracted slices of the suspended artifact'
  ),
  createDay(
    16,
    'Spore',
    'upcoming',
    'tsl-compute',
    'sporeSulfur',
    'rawUmber',
    'Puffball gasteromycete fruiting body releasing turbulent buoyant clouds of sulfur-gold spores',
    'Tapping the peridium triggers a peristome burst; pointer velocity stirs curl-noise vortex rings'
  ),
  createDay(
    17,
    'Kelp',
    'upcoming',
    'tsl-compute',
    'mossUmbrous',
    'realgarAmber',
    'Submerged Macrocystis kelp blades and pneumatocyst bladders swaying in a turbid coastal surge column',
    'Pointer sweeps underwater current eddies, bending translucent amber-olive fronds with subsurface scatter'
  ),
  createDay(
    18,
    'Pendulum',
    'upcoming',
    'tsl',
    'boneCalcite',
    'burntSienna',
    'Coupled Blackburn double-pendulum harmonograph tracing decaying phase-space ribbons into a sand tray',
    'Drag and release the suspended brass bob to launch a new Lissajous orbital trajectory'
  ),
  createDay(
    19,
    'Erosion',
    'upcoming',
    'wgsl',
    'yellowOchre',
    'burntSienna',
    'Stratified sandstone mesa block carved in real time by hydraulic raindrop flow and sediment deposition',
    'Holding the pointer rains pressurized water onto the crest, incising braided canyons and alluvial fans'
  ),
  createDay(
    20,
    'Cocoon',
    'upcoming',
    'tsl',
    'warmKaolin',
    'realgarAmber',
    'Spun Bombyx silk lattice shell wrapped around a faintly pulsing amber chrysalis core',
    'Pointer winds additional translucent silk threads around the shell and warms the internal bioluminescence'
  ),
  createDay(
    21,
    'Bolt',
    'upcoming',
    'wgsl',
    'boneCalcite',
    'azuritePrussian',
    'Dielectric acrylic block capturing branching Lichtenberg breakdown trees and stepped-leader arcs',
    'Click to discharge high-voltage breakdown from the pointer electrode into the dielectric block'
  ),
  createDay(
    22,
    'Echo',
    'upcoming',
    'wgsl',
    'verdigrisMalachite',
    'boneCalcite',
    'Ellipsoidal acoustic whispering-gallery chamber visualizing 3D wave-equation wavefront reflections',
    'Click anywhere inside the cavity to emit a sharp acoustic pulse that focuses at the conjugate focus'
  ),
  createDay(
    23,
    'Magma',
    'upcoming',
    'wgsl',
    'cinnabarVermilion',
    'rawUmber',
    'Convecting pahoehoe basalt lava pillow with viscoplastic cooling crust cracking over incandescent fissures',
    'Pointer shears the dark basalt crust apart, exposing white-hot/cinnabar blackbody magma that cools back to rock'
  ),
  createDay(
    24,
    'Nest',
    'upcoming',
    'tsl',
    'rawUmber',
    'celadonGlauconite',
    'Interwoven birch-twig, mud-dauber, and lichen cup nest cradling speckled celadon mineral eggs',
    'Pointer gently parts the outer twig weave and rotates the specimen under soft studio rim light'
  ),
  createDay(
    25,
    'Veil',
    'upcoming',
    'tsl-compute',
    'boneCalcite',
    'lapisUltramarine',
    'Gossamer silk organza sheet settling in slow air over an unseen sculpted bust with anisotropic sheen',
    'Pointer lifts and billows the gossamer cloth via XPBD aerodynamic lift before letting it drape anew'
  ),
  createDay(
    26,
    'Anamorph',
    'upcoming',
    'tsl',
    'yellowOchre',
    'azuritePrussian',
    'Fragmented cloud of mineral shards and catoptric cylinder that resolves into a coherent form from one vantage',
    'Orbiting with the pointer moves between chaotic shard suspension and the exact optical lock-in angle'
  ),
  createDay(
    27,
    'Burrow',
    'upcoming',
    'tsl',
    'burntSienna',
    'boneCalcite',
    'Cast aluminum/calcite endocast of a subterranean ant & root gallery network suspended in cross-section',
    'Pointer illuminates and traces nutrient flow packets traveling down the branching subterranean shafts'
  ),
  createDay(
    28,
    'Hive',
    'upcoming',
    'tsl',
    'yellowOchre',
    'realgarAmber',
    'Natural wild comb of rhombic dodecahedral beeswax cells backlit through varied amber honey fill levels',
    'Pointer warms and ripples the uncapped honey menisci across hexagonal cells with Beer-Lambert depth'
  ),
  createDay(
    29,
    'Portal',
    'upcoming',
    'tsl',
    'lapisUltramarine',
    'verdigrisMalachite',
    'Suspended obsidian ring framing a non-Euclidean stencil/raymarched aperture into an inverted mineral vault',
    'Orbiting around the thin ring reveals an interior space larger than the exterior room'
  ),
  createDay(
    30,
    'Dissolve',
    'upcoming',
    'tsl-compute',
    'boneCalcite',
    'celadonGlauconite',
    'Crystalline calcite sculpture undergoing Liesegang-ring reaction-diffusion dissolution into buoyant dust',
    'Pointer sweeps a solvent wavefront across the specimen, sublimating solid geometry into drifting particles'
  ),
];

export function getDayByNumber(dayNumber: number): DayEntry {
  if (dayNumber === 0) return CALIBRATION_DAY;
  const found = DAYS_MANIFEST.find((d) => d.day === dayNumber);
  if (!found) {
    throw new Error(`Day ${dayNumber} not found in DAYS_MANIFEST (expected 0..30)`);
  }
  return found;
}

export function getDayBySlug(slug: string): DayEntry | undefined {
  if (slug === CALIBRATION_DAY.slug) return CALIBRATION_DAY;
  return DAYS_MANIFEST.find((d) => d.slug === slug);
}

/**
 * Returns the highest-numbered day whose status is 'complete'.
 * Falls back to the Day 00 calibration specimen before Day 01 is marked complete.
 */
export function getLatestActiveDay(): DayEntry {
  for (let i = DAYS_MANIFEST.length - 1; i >= 0; i--) {
    if (DAYS_MANIFEST[i].status === 'complete') {
      return DAYS_MANIFEST[i];
    }
  }
  return CALIBRATION_DAY;
}

/**
 * Returns the previous and next navigable (completed) days relative to `currentDay`.
 */
export function getAdjacentCompletedDays(currentDay: number): {
  prev: DayEntry | null;
  next: DayEntry | null;
} {
  const completed: DayEntry[] = [
    CALIBRATION_DAY,
    ...DAYS_MANIFEST.filter((d) => d.status === 'complete'),
  ];
  const idx = completed.findIndex((d) => d.day === currentDay);
  if (idx === -1) {
    return { prev: null, next: null };
  }
  return {
    prev: idx > 0 ? completed[idx - 1] : null,
    next: idx < completed.length - 1 ? completed[idx + 1] : null,
  };
}
