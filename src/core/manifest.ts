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
    'complete',
    'wgsl',
    'yellowOchre',
    'lapisUltramarine',
    'Grand 6-planet coaxial brass planetarium uniting an 18th-century stepped gear pyramid, Da Vinci Codex Madrid conical fusee & verge-foliot regulator, and Antikythera Hipparchus pin-and-slot lunar anomaly + 4-turn Saros dial',
    'Drag the left handwheel to crank manually or drag to orbit across 6 camera presets; hover/click any of 16 mechanical groups to inspect live telemetry or press [C] for X-Ray cutaway and [E] to explode coaxial decks'
  ),
  createDay(
    7,
    'Hollow',
    'complete',
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
    'complete',
    'tsl-compute',
    'boneCalcite',
    'yellowOchre',
    'Horizontal brass-and-walnut tension hoop stretching a translucent natural vellum/latex membrane where dropped polished mineral spheres bounce, deform deep gravity-well funnels, and coalesce into a sagging central pocket',
    'Click (or hold to charge size and flick laterally to launch orbits) to drop polished mineral marbles onto the membrane; drag any resting stone to pull its tension funnel across the skin, or double-click / press Space to snap the drumhead upward'
  ),
  createDay(
    10,
    'Caustic',
    'complete',
    'tsl',
    'celadonGlauconite',
    'realgarAmber',
    'Blown Venetian celadon-crystal goblet half-filled with botanical-mineral elixir casting forward-traced nephroid, cardioid, and rippling meniscus caustics onto a beveled Bone Calcite slab',
    'Drag the stone slab or brass condenser lamp to orbit and tilt the incident beam; sweep or tap the goblet to slosh the liquid and ripple the caustic web'
  ),
  createDay(
    11,
    'Rust',
    'complete',
    'tsl',
    'hematiteRust',
    'yellowOchre',
    'Hand-forged wrought-iron shipwreck spike suspended in a beveled soda-lime glass brine cube with porous goethite rusticle stalactites, gossamer ferric wisps, and delaminating oxide bark',
    'Hover across the submerged nail to pry up and scrape brittle rust scabs down to dark forged iron; click or drag to gouge heavier flake bursts and slosh the tank'
  ),
  createDay(
    12,
    'Feather',
    'complete',
    'tsl',
    'azuritePrussian',
    'realgarAmber',
    'Seven layered 3D peacock train feathers articulated in a stepped dark-bronze and Bone Calcite clasp with thin-film photonic-crystal barbs and concentric cordate ocelli',
    'Hover sweeps the goniometric studio beam and parts localized barb wakes; drag blades/clasp to fold or unfurl, click a blade to pluck cantilever flex, or drag space to orbit'
  ),
  createDay(
    13,
    'Labyrinth',
    'complete',
    'tsl',
    'rawUmber',
    'yellowOchre',
    'Suspended chamfered limestone and vesicular basalt 6-face labyrinth cube with verdigris copper-carbonate seams, vaulted stone underpasses, and a liquid quicksilver bead painting a yellow-ochre glaze trail into the cinnabar sanctum to unlatch an 8-octant brass astrolabe & lapis-realgar reliquary vault',
    'Drag to tilt and roll via gimbal-free SO(3) trackball rotation; hover imparts subtle specular micro-bias without dislodging resting marbles; scroll to zoom; click the opened core or double-click to reset'
  ),
  createDay(
    14,
    'Tension',
    'complete',
    'tsl',
    'cinnabarVermilion',
    'rawUmber',
    'Multi-layer mineral-dyed natural rubber band ball wrapped in 200 stratified great-circle straps around a knotted Raw Umber and Bone Calcite starter core, with stress-whitening stretch, compressive breathing, sympathetic cascade pops, and talcum bloom',
    'Drag any rubber band outward to stretch and whiten it until it snaps or release to twang back; click a band to nick and whip-curl it away in a burst of talcum dust; double-click or click the bare core to re-wrap'
  ),
  createDay(
    15,
    'Mirage',
    'complete',
    'tsl',
    'realgarAmber',
    'lapisUltramarine',
    'Suspended open cotton-rag sketchbook whose flat charcoal drawing breaks open in parallax through seven torn-paper mineral arches, re-signifying a contra-jour curled silhouette into cupped hands cradling glowing faceted Realgar-Amber origami moths',
    'Hover to shear the seven carved paper arches in parallax; click to stoke the crease ember and wake dormant charcoal facets into fluttering amber origami moths; scroll or drag to dolly and arc 42° inside the lantern'
  ),
  createDay(
    16,
    'Spore',
    'complete',
    'tsl-compute',
    'sporeSulfur',
    'rawUmber',
    'Suspended Earthstar puffball (Geastrum triplex) with a papery Warm Kaolin spore sac, fluted sulfur-dusted peristome chimney, seven recurved cracked Bone Calcite & Raw Umber exoperidium claws, hanging mycelial rootlets, and a 49,152-particle WebGPU TSL-compute toroidal vortex-ring spore plume',
    'Click or squeeze and knead the papery spore sac to indent its skin and launch leapfrogging toroidal sulfur vortex rings (off-center pokes tilt the plume); sweep across the drifting cloud to stir 3D curl-noise eddies, or drag open space to orbit'
  ),
  createDay(
    17,
    'Kelp',
    'complete',
    'tsl',
    'mossUmbrous',
    'realgarAmber',
    'Suspended cured kelp-leather and alginate parchment pendant lantern tensioned over six bent dark-walnut ribs with crisscross whip-stitched seams, crinkled olive-umber outer frills, aged-brass knurled canopy, and a warm glowing tubular Edison filament core',
    'Hover over any parchment panel to flex the cured kelp skin; drag the lantern to swing and spin it on its braided hemp cord; click to pluck a drumhead ripple and cycle filament warmth (warm, bright, ember), or drag open space to orbit'
  ),
  createDay(
    18,
    'Pendulum',
    'complete',
    'tsl',
    'boneCalcite',
    'cinnabarVermilion',
    'Suspended hand-planished hammered-iron magnetic dome ringed in honed Bone Calcite beneath a dark-bronze horseshoe gimbal, 5-spoke brass differential pulley, and rising counterweight plumb, with cleaved Cinnabar and Verdigris magnetite lodestones steering a sweeping 8-facet Bone-Calcite magnetic pendulum bob',
    'Grab and fling the Bone-Calcite bob to launch chaotic magnetic slingshots, drag any lodestone across the iron dome to sculpt the field, click a stone to flip its polarity mid-air, click bare iron to toss a new stone, or click the calcite rim (or press Space) to cast a fresh handful'
  ),
  createDay(
    19,
    'Erosion',
    'complete',
    'tsl',
    'boneCalcite',
    'yellowOchre',
    'Suspended stratified Bone-Calcite and Yellow-Ochre limestone fossil concretion enclosing a ribbed, chambered pyritized ammonite relic prepared in real time by an articulated satin-brass pneumatic air-abrasion micro-nozzle',
    'Hold and sweep across the limestone nodule to fire the pressurized dolomite-grit stream and strip soft sedimentary strata down to the gleaming pyritized ammonite ribs; drag empty space to orbit 360°; double-click or press Space to re-encrust a fresh nodule'
  ),
  createDay(
    20,
    'Cocoon',
    'complete',
    'tsl',
    'warmKaolin',
    'realgarAmber',
    'Suspended translucent Warm-Kaolin & Realgar-Amber heat-shrink polymer-parchment cocoon enveloping three cyclable hidden sculptural cores (brass-and-basalt astrolabe caliper reliquary, prismatic Bone-Calcite & Verdigris spire cluster, and tri-axis brass gyro-armature around a glowing amber cavity) with real XPBD rest-length contraction, analytical SDF vacuum-forming, optical contact-clarity, and propagating Mode-I seam ruptures',
    'Hover or sweep (hold-drag to intensify) a warm infrared thermal cone across the crinkled sleeve to shrink and vacuum-form the translucent skin taut over every ridge and spoke; click any high-tension span (or Shift-click / double-click) to rupture and peel the elastic flaps open; drag empty space to orbit; double-click empty space or press Space to wrap a fresh cocoon around the next hidden sculptural core'
  ),
  createDay(
    21,
    'Bolt',
    'complete',
    'tsl',
    'yellowOchre',
    'azuritePrussian',
    'Skeletonized 19th-century Chubb-style 5-lever mortise deadbolt cutaway set into an oiled Raw-Umber charred-oak timber block with a beveled satin-brass forend faceplate, slotted wood-screws, heavy Prussian-blued steel deadbolt with U-shaped talon and riveted square bolt stump, five stacked independently patinated brass lever tumblers separated by spacer washers with anti-pick serrations, five flexing phosphor-bronze leaf springs, and a stepped Victorian warded bit key',
    'Drag the Victorian key bow circularly (or click it, scroll over it, or press Space) to turn 360° and watch the 5 stepped bitting teeth lift all five brass levers against their flexing phosphor-bronze leaf springs to align their gates and throw the blued-steel deadbolt; drag any individual brass lever to pick it against spring torque; drag the deadbolt head or stump to feel misaligned lever blocking; drag empty space to orbit'
  ),
  createDay(
    22,
    'Echo',
    'complete',
    'tsl',
    'verdigrisMalachite',
    'realgarAmber',
    'Machined dark-bronze and satin-brass two-way infinity mirror cube with diamond-knurled micrometer thumbscrews and a 40-bounce TSL optical cavity enclosing a swinging Realgar-Amber Edison filament inside a faceted Bone-Calcite prism receding through Beer-Lambert soda-lime celadon and verdigris depths',
    'Drag empty space to orbit 360° around all six faces; drag the cube or knurled brass thumbscrews (or Shift+drag) to micro-tilt the interior mirrors and bend the 40-bounce echo corridor into curves and vortices; click the glass (or press Space) to pluck the suspended amber pendulum; double-click to snap parallel or cycle canonical mirror geometries'
  ),
  createDay(
    23,
    'Magma',
    'complete',
    'tsl',
    'cinnabarVermilion',
    'rawUmber',
    'Suspended heavy steel triaxial geology press clamping 12 interlocking stratified vesicular basalt and banded yellow-ochre tuff slabs above a quartz-windowed incandescent magma chamber, with a vertical hydraulic pressure piston, lateral ACME tectonic screw-clamps, stress-coupled vertical dikes and horizontal laccolith sills, and cooling glossy-obsidian pahoehoe lobes',
    'Hold-click the chamber or drag vertically on the bottom hydraulic piston or rock core to pump magma pressure upward; drag the left or right lateral clamp screws inward or outward to switch continuously between extensional vertical dikes and compressional horizontal sills with overburden dome heave; drag empty space to orbit; double-click or press Space to release pressure and quench veins into dark obsidian'
  ),
  createDay(
    24,
    'Nest',
    'complete',
    'rapier',
    'rawUmber',
    'celadonGlauconite',
    'Suspended honed-limestone and dark-brass hollow ring plinth supporting a 56-dowel self-supporting Leonardo/Zollinger reciprocal timber dome of counter-woven walnut, smoked-ash, ochre-cedar, and birch dowels cradling three smooth speckled Celadon and Bone-Calcite river-pebble eggs under live Rapier 0.21 rigid-body equilibrium',
    'Hover any timber dowel for warm ochre rim lift, click quickly to tap-test whether it is loose or load-bearing, grab and pull dowels outward via 3D spring-tether like structural Jenga to topple the weave and drop the Celadon stones through the hollow plinth, drag empty space to orbit 360°, or double-click empty space / press Space to re-weave'
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
    'complete',
    'tsl',
    'yellowOchre',
    'realgarAmber',
    'Suspended gnarled dark-walnut and charred-oak limb draped with a curved open-air giant honeybee (Apis dorsata) teardrop bivouac of 2,643 shingled articulated insects—each with sculpted head, velvety thorax, hinged 5-tergite banded abdomen, and veined translucent gossamer forewings—coupled on a 3D surface k-NN excitable-media graph driving crisp 65°–75° abdomen-flip defense waves, head-on wave annihilation, rotating spirals, and physical pendulum recoil',
    'Hover or sweep close to the hanging bivouac to ignite traveling golden-ochre defense waves and spiral eddies; click directly on the cluster for a radial shock-shimmer with pendulum recoil; drag across the curtain to launch colliding fronts; drag empty space to orbit 360°; double-click or press Space to quiet active waves or trigger a twin opposing-pole collision wave'
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
    'complete',
    'tsl-compute',
    'celadonGlauconite',
    'realgarAmber',
    'Suspended thick-walled borosilicate 500 mL laboratory beaker with flared pouring spout, etched eggshell volumetric markings, and a watertight rippling water column dissolving up to four crystalline mineral-pigment salt cubes through curvature-weighted corner rounding, rising silver effervescent micro-bubble streams, 16,384 GPU-advected descending refractive schlieren plumes, and a tactile glass stirring rod',
    'Click over the beaker (or press Space) to drop fresh crystalline mineral cubes cycling through Celadon, Realgar Amber, Azurite, Cinnabar, and Yellow Ochre; drag inside the beaker to dip and sweep the borosilicate stirring rod into a central vortex; drag empty space to orbit 360°; double-click empty space to flush clear water and drop a starter crystal'
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
