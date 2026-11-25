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
    'Suspended 3-whorl cascading cocoon unfurling into 18 razor-thin hand-torn rice-paper and parchment petals with varied pulp opacity and frayed deckle whiskers around a golden stamen crown, 12 curl-gliding origami moths, and 280 backlit pollen motes',
    'Sweep across the air or flower to swirl golden pollen motes, steer origami moths, and ripple petals in a wave; grab and pull any petal outward or inward to blossom or fold shut into the candled cocoon; click to stoke a buoyant moth & pollen burst; drag empty space to orbit 360°, or double-click / press Space to toggle closed and full bloom'
  ),
  createDay(
    21,
    'Bolt',
    'complete',
    'tsl',
    'yellowOchre',
    'azuritePrussian',
    'Skeletonized cutaway 19th-century geared 5-lever Chubb thumb-turn mortise deadbolt set into an oiled Raw-Umber charred-oak door stile with a beveled satin-brass cutaway bezel and forend faceplate, pairing a quarter-turn architectural satin-brass thumb-turn knob and 3:1 coaxial involute speed-up gear train on a horological brass bridge with five patinated brass lever tumblers, staggered phosphor-bronze cantilever leaf springs, a heavy Prussian-blued steel deadbolt with upper rack-and-pinion detent escapement, and dual spring-loaded escapement pawls',
    'Drag the brass thumb-turn knob circularly (or click it, scroll over it, or press Space) to throw 90° between Vertical Unlocked and Horizontal Locked and watch the 3:1 coaxial gear train drive the internal cam barrel 270°, lifting all five brass levers against their staggered phosphor-bronze leaf springs to throw the heavy blued-steel deadbolt; drag any brass lever upward to pick it against spring torque; drag the deadbolt head to feel misaligned lever blocking or back-drive the gear train when aligned; drag empty space to orbit 360°'
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
    'complete',
    'tsl-compute',
    'boneCalcite',
    'realgarAmber',
    'Suspended machined dark-iron and satin-brass studio optical slit-lamp collimator with milled cooling fins, diamond-knurled focus drums, and articulating brass aperture blades projecting a razor-thin volumetric light-and-mist veil driven by a live 128x128 incompressible Navier-Stokes vorticity solver, upstream Beer-Lambert light-shaft shadowing, and 8,192 WebGPU-compute sunlit dust motes illuminated exclusively while piercing the beam plane',
    'Hover or sweep across the suspended light veil to slice dark openings and shed counter-rotating Kelvin-Helmholtz smoke curls that billow and slowly heal; hold and drag across the curtain to carve a deep turbulent channel with bright sheared amber caustic rims; click the overhead brass slit-lamp housing (or press Space) to cycle aperture geometry seamlessly across Flat Planar Curtain, Graceful Curved S-Veil, and Hollow Elliptical Light Cone; drag empty space to orbit 360°'
  ),
  createDay(
    26,
    'Anamorph',
    'complete',
    'tsl',
    'cinnabarVermilion',
    'yellowOchre',
    'Suspended 30-slab brutalist board-formed concrete and honed Roman travertine courtyard maquette—with freestanding portal archway, six cantilevered stair treads, offset baffle wall, recessed colonnade bays, overhead concrete sky-beams, and a hovering machined satin-brass surveyor sight-ring—carrying a real-time Felice Varini projective Kubelka-Munk mineral-pigment stencil with exact 30-slab analytical line-of-sight occlusion',
    'Drag to orbit 360° around the brutalist courtyard maquette and align with the hovering brass surveyor sight-ring to cushion into razor-sharp optical lock-in where fractured pigment bands snap into a floating geometric figure; click the building (or press Space) from any angle to project and bake a fresh mineral-pigment stencil (Concentric Rings & Diamond, Impossible Penrose Tribar, Nested Isometric Cube, Architectural Squares) from your current camera viewpoint; click the brass sight-ring, double-click, or press R to glide smoothly into optical lock-in'
  ),
  createDay(
    27,
    'Burrow',
    'complete',
    'tsl',
    'burntSienna',
    'boneCalcite',
    'Suspended oiled dark-walnut and satin-brass observation ant-farm terrarium slice with slotted corner hardware, 45° beveled glass panes, and a live 300×220 stratified mineral sand bed with pebble-draped strata, momentum-coupled granular avalanches, Terzaghi arching vs. conjugate shear-fault cave-ins, 3D tumbling mineral granules, and a 12-worker-ant colony with articulated anatomy, open-corridor BFS navigation, cave-in rescue digging, and antennae-tap nestmate greetings',
    'Drag directly across the glass sand bed to carve branching burrows and test vaulted arch spans against cascading shear-fault cave-ins; click and hold in the clear open top airspace to pour a cohesive hourglass stream of layered mineral sand; drag empty space to orbit 360° or scroll to inspect the worker ants close-up; double-click empty space or press Space to reset'
  ),
  createDay(
    28,
    'Hive',
    'complete',
    'tsl',
    'yellowOchre',
    'cinnabarVermilion',
    'Suspended chamfered Bone-Calcite limestone and oil-rubbed bronze hexagonal honeycomb apparatus divided by beveled satin-brass walls into 91 chambers—each housing a jewel-pivoted faceted Cinnabar-Vermilion and Prussian-blued steel magnetic compass needle interacting via exact all-pairs 1/r³ dipole-dipole coupling to form self-organizing magnetic flux vortices, murmuration spin waves, and domain-wall avalanches beneath a hovering brass-and-magnetite lodestone stylus',
    'Hover over the honeycomb to glide the brass lodestone stylus and launch cascading magnetic spin waves; press and drag across cells to comb flux rivers or wind macroscopic vortices; quick-click a cell for a swirling radial shock pulse; click the outer bronze bezel (or press F) to flip stylus polarity; double-click or press Space for a thermal scatter that lets all 91 dipoles self-organize freshly; drag empty space to orbit 360°'
  ),
  createDay(
    29,
    'Portal',
    'complete',
    'tsl',
    'lapisUltramarine',
    'yellowOchre',
    'Suspended chamfered dark-basalt and machined satin-brass architectural cube with stacked ashlar piers, beveled arched stone spandrels, recessed brass window casings, corner joinery brackets, and cap/base brass compass-rose inlays enclosing four distinct camera-parallax WebGPU TSL SDF worlds sharing one 1.5-unit interior: a sunlit Bone-Calcite travertine colonnade (+Z), a plunging Lapis-Ultramarine cantilevered spiral stairwell (+X), a vaulted Verdigris cistern with a swaying Realgar-Amber lantern over rippling dark water (-Z), and a layered Burnt-Sienna desert slot-canyon rock arch (-X)',
    'Drag anywhere to orbit 360° (including full vertical pitch to peer down the Lapis spiral stairwell or up at vaulted stone arches) and cross-cut two incompatible full-scale worlds along every 45° basalt-and-brass corner; hover any arched aperture to shift sunlight beams, subterranean depth glow, or cistern water ripples; click any portal face or scroll to dolly right up to the arched stone threshold (click again or scroll out to pull back); double-click or press Space to step 90° around the four worlds'
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
