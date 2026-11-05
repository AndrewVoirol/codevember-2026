import * as THREE from 'three/webgpu';
import {
  abs,
  atan,
  attribute,
  clamp,
  cos,
  dot,
  float,
  max,
  min,
  mix,
  normalLocal,
  normalWorld,
  positionLocal,
  sin,
  smoothstep,
  step,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/**
 * Day 04 — Tangle: "The Ultimate Test of Patience — Eaten Mixtape & Pencil Rewind"
 *
 * A realistic Philips-standard Compact Cassette specimen inspired by the classic
 * tangled Maxell-UR mixtape & hexagonal #2 pencil rewind ritual:
 * - Translucent smoky charcoal-slate polycarbonate shell with visible internal
 *   reel-chamber containment rings, corner stiffening struts, sonic-weld bevels,
 *   5 countersunk Phillips (+) machine screws, and a sculpted 14°-tapered
 *   trapezoidal bottom head block ("the chin") with horizontal grip fluting and
 *   4 recessed capstan & alignment-pin apertures.
 * - Die-cut vintage cream & cinnabar-vermilion paper label featuring left/right
 *   "UR" red-orange accent blocks, horizontal red-orange rule stripes, lower
 *   brand bar, top-left Side "A" badge, and felt-tip marker "MIXTAPE" lettering.
 * - Central clear window bezel with 100|50|0 remaining-tape gauge ticks,
 *   slotted 6-toothed Delrin drive hubs, and asymmetric wound ferric-oxide tape
 *   pancakes with anisotropic radial bow-tie reflections.
 * - Hexagonal yellow-ochre #2 wooden pencil with a sharpened raw cedar cone,
 *   dark graphite tip wedged at an ergonomic angle into the right-hand 6-toothed
 *   spool hub, crimped brass & verdigris-striped metallic ferrule, and rounded
 *   terracotta-pink rubber eraser—serving as the tactile crank to wind the
 *   spilled tape back into the cassette.
 * - Dense, multi-loop 3D bird's-nest cascade of glossy 3.81 mm ferric-oxide
 *   Mylar magnetic tape spilling from the bottom head bay and draping across
 *   the bottom and up the entire left flank of the cassette with crisp hairpin
 *   kinks, torsional twists, specular glints, and progressive pencil-crank
 *   untangling and rewind physics.
 */

const NODE_COUNT = 440; // High-density physics nodes for 22+ overlapping 3D ribbon loops
const FREE_START = 16; // First free node exiting the left brass guide roller
const FREE_END = NODE_COUNT - 17; // Last free node entering the right brass guide roller
const FREE_COUNT = FREE_END - FREE_START;
const SUBDIV = 4; // Render rings per physics segment (1,757 rings total)
const RENDER_RINGS = (NODE_COUNT - 1) * SUBDIV + 1;
const TAPE_WIDTH = 0.082; // Realistic 3.81 mm tape width relative to cassette head bay
const TAPE_THICKNESS = 0.0052;
const SELF_COLLISION_DIST = 0.020; // Gentle non-adjacent crossing clearance
const FOLD_STRIDE = 3; // Capstan accordion bite-mark pleat spacing

// Cassette shell & guide-roller coordinates in local specimen space
const SPOOL_Y = 0.70;
const LEFT_SPOOL_X = -0.44;
const RIGHT_SPOOL_X = 0.44;
const LEFT_GUIDE_X = -0.52;
const RIGHT_GUIDE_X = 0.52;
const THROAT_EXIT_Y = 0.158;

// Pre-allocated module-scope scratch vectors & buffers (strictly 0 per-frame allocations)
const _s0 = new THREE.Vector3();
const _s1 = new THREE.Vector3();
const _s2 = new THREE.Vector3();
const _s3 = new THREE.Vector3();
const _tang = new THREE.Vector3();
const _smoothTang = new THREE.Vector3();
const _prevSmoothTang = new THREE.Vector3();
const _bishopU = new THREE.Vector3();
const _bishopV = new THREE.Vector3();
const _norm = new THREE.Vector3();
const _binorm = new THREE.Vector3();
const _localPtr = new THREE.Vector3();
const _localVel = new THREE.Vector3();
const _invRigQuat = new THREE.Quaternion();
const _samplePt = new THREE.Vector3();
const _pencilDir = new THREE.Vector3();
const _pencilUp = new THREE.Vector3(0, 1, 0);
const _pencilQuat = new THREE.Quaternion();

const ARC_SAMPLES = 960;
const _arcX = new Float32Array(ARC_SAMPLES);
const _arcY = new Float32Array(ARC_SAMPLES);
const _arcZ = new Float32Array(ARC_SAMPLES);
const _arcCumLen = new Float32Array(ARC_SAMPLES);

/**
 * 64 Curated 3D Bird's-Nest Tangle Waypoints `[x, y, z, sharpKink]` in local
 * cassette space. Traces the spilled tape from the Left Guide Roller `(-0.52, 0.158, 0)`,
 * routing behind/under the bottom-left corner, cascading in 20+ layered figure-8
 * and cloverleaf loops above the top-left crown (`y in [1.26, 1.46], z < -0.06`),
 * all down the left flank (`x in [-1.82, -0.96], y in [-0.88, 1.38]`), and pooling
 * across the bottom-left and bottom-right (`x in [-1.46, +0.94], y in [-0.95, 0.08]`)
 * before feeding cleanly back into the Right Guide Roller `(+0.52, 0.158, 0)`.
 */
const TANGLE_WAYPOINTS: readonly [number, number, number, number][] = [
  [-0.52, 0.158, 0.0, 0.0], // 0: Left Guide Roller exit
  [-0.68, -0.14, 0.03, 0.30],
  [-1.24, -0.26, 0.07, 0.50],
  [-1.56, 0.14, -0.04, 0.55],
  [-1.18, 0.52, -0.10, 0.50],
  // Zone 1: Top-left & upper-crown loops emerging above the top-left cassette edge
  [-1.56, 0.88, -0.06, 0.55],
  [-1.16, 1.28, -0.08, 0.50],
  [-0.68, 1.50, -0.02, 0.55], // High top-left arch 1 rising above top cassette edge
  [-0.32, 1.42, -0.05, 0.50], // Upper top-center-left arch peeking above top edge
  [-0.78, 1.29, -0.09, 0.45],
  [-1.18, 1.54, 0.02, 0.60], // High outer top-left arch 2 catching studio key light
  [-1.52, 1.24, 0.06, 0.55],
  [-1.06, 0.94, 0.08, 0.50],
  [-1.04, 1.34, -0.03, 0.55],
  [-1.42, 1.42, 0.03, 0.60], // Upper-left cloverleaf loop
  [-1.68, 1.04, 0.05, 0.55],
  [-1.22, 0.72, 0.11, 0.50],
  [-1.02, 1.00, -0.04, 0.45],
  // Zone 2: Dense multi-layered left-flank bird's-nest loops (strictly x <= -0.98)
  [-1.44, 0.96, -0.11, 0.55],
  [-1.78, 0.62, 0.05, 0.60], // Wide upper-mid left loop
  [-1.38, 0.34, 0.13, 0.50],
  [-0.98, 0.66, 0.06, 0.45],
  [-1.26, 0.82, -0.09, 0.55],
  [-1.68, 0.42, -0.12, 0.60],
  [-1.48, 0.06, 0.08, 0.55],
  [-0.98, 0.32, 0.12, 0.45],
  [-1.16, 0.56, -0.07, 0.50],
  [-1.62, 0.22, -0.04, 0.55],
  [-1.82, -0.18, 0.07, 0.65], // Far outer mid-lower left loop
  [-1.36, -0.36, 0.14, 0.55],
  [-0.97, -0.02, 0.05, 0.45],
  [-1.18, 0.24, -0.10, 0.50],
  [-1.58, -0.08, -0.12, 0.55],
  [-1.66, -0.52, 0.04, 0.60], // Lower-left cascading loop 1
  [-1.22, -0.64, 0.12, 0.55],
  [-0.96, -0.26, 0.02, 0.45],
  [-1.28, -0.06, -0.09, 0.50],
  [-1.72, -0.38, -0.06, 0.60],
  [-1.52, -0.82, 0.08, 0.65], // Deep lower-left loop 2
  [-1.04, -0.68, 0.14, 0.55],
  [-0.86, -0.22, -0.08, 0.45],
  [-1.26, -0.32, 0.06, 0.50],
  // Zone 3: Bottom-left & bottom-center intertwined loops (strictly y <= 0.10)
  [-1.38, -0.78, -0.09, 0.60],
  [-0.92, -0.94, 0.05, 0.65], // Bottom-left sweeping base loop
  [-0.48, -0.62, 0.13, 0.50],
  [-0.72, -0.14, 0.04, 0.45], // Tuck under left chin
  [-1.14, -0.44, -0.11, 0.55],
  [-0.78, -0.86, -0.04, 0.60],
  [-0.22, -0.76, 0.11, 0.55],
  [-0.36, -0.22, 0.07, 0.45],
  [-0.86, -0.34, -0.08, 0.50],
  [-0.52, -0.88, 0.04, 0.60],
  [0.06, -0.84, 0.12, 0.55], // Broad horizontal ribbon sweep across bottom center
  [0.18, -0.34, -0.06, 0.45],
  [-0.34, -0.18, -0.10, 0.45],
  [-0.24, -0.68, 0.06, 0.55],
  // Zone 4: Graceful sweeping bottom-right ribbon loops (zero sharp reversal knots!)
  [0.36, -0.78, 0.11, 0.55],
  [0.78, -0.58, 0.05, 0.60], // Bottom-right wide loop 1
  [0.54, -0.22, -0.07, 0.45],
  [0.04, -0.36, -0.09, 0.45],
  [0.28, -0.82, 0.04, 0.55],
  [0.88, -0.86, 0.09, 0.65], // Far bottom-right extended loop 2 (matching reference photo!)
  [0.92, -0.54, 0.02, 0.55],
  [0.44, -0.34, -0.05, 0.40],
  [0.48, -0.06, -0.02, 0.25],
  [0.52, 0.158, 0.0, 0.0], // 63: Right Guide Roller entry
];

/**
 * TSL 2D line-segment distance helper for crisp procedural felt-tip marker lettering.
 */
function sdSegment2DTsl(
  p: TslNode,
  ax: number,
  ay: number,
  bx: number,
  by: number
): TslNode {
  const a = vec2(ax, ay);
  const b = vec2(bx, by);
  const pa = p.sub(a);
  const ba = b.sub(a);
  const h = clamp(dot(pa, ba).div(max(float(1e-6), dot(ba, ba))), 0.0, 1.0);
  return pa.sub(ba.mul(h)).length();
}

/**
 * Helper: Creates a 2D rounded-rectangle `THREE.Shape` or `THREE.Path` centered at `(cx, cy)`.
 */
function createRoundedRectPath(
  cx: number,
  cy: number,
  w: number,
  h: number,
  r: number,
  isShape = true
): THREE.Shape | THREE.Path {
  const x0 = cx - w * 0.5;
  const y0 = cy - h * 0.5;
  const x1 = cx + w * 0.5;
  const y1 = cy + h * 0.5;
  const cr = Math.min(r, w * 0.45, h * 0.45);

  const p = isShape ? new THREE.Shape() : new THREE.Path();
  p.moveTo(x0 + cr, y0);
  p.lineTo(x1 - cr, y0);
  p.quadraticCurveTo(x1, y0, x1, y0 + cr);
  p.lineTo(x1, y1 - cr);
  p.quadraticCurveTo(x1, y1, x1 - cr, y1);
  p.lineTo(x0 + cr, y1);
  p.quadraticCurveTo(x0, y1, x0, y1 - cr);
  p.lineTo(x0, y0 + cr);
  p.quadraticCurveTo(x0, y0, x0 + cr, y0);
  return p;
}

/**
 * Builds the 8-vertex cross-section extruded ribbon buffer geometry for the
 * magnetic cassette tape:
 *   0..1: Top magnetic oxide face (+normal)
 *   2..3: Right slitted edge (+binormal)
 *   4..5: Bottom Mylar backing face (-normal)
 *   6..7: Left slitted edge (-binormal)
 */
function createTapeRibbonGeometry(rings: number): {
  geometry: THREE.BufferGeometry;
  posAttr: THREE.BufferAttribute;
  normAttr: THREE.BufferAttribute;
  metaAttr: THREE.BufferAttribute;
} {
  const vertsPerRing = 8;
  const vertCount = rings * vertsPerRing;
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const meta = new Float32Array(vertCount * 4);

  for (let r = 0; r < rings; r++) {
    const sNorm = r / (rings - 1);
    const base = r * vertsPerRing * 4;
    // 0: Top-Right (+N, +B)
    meta[base + 0] = sNorm; meta[base + 1] = 1.0; meta[base + 2] = 1.0; meta[base + 3] = 0.0;
    // 1: Top-Left (+N, -B)
    meta[base + 4] = sNorm; meta[base + 5] = -1.0; meta[base + 6] = 1.0; meta[base + 7] = 0.0;
    // 2: Right-Top (+B, +N)
    meta[base + 8] = sNorm; meta[base + 9] = 1.0; meta[base + 10] = 0.0; meta[base + 11] = 0.0;
    // 3: Right-Bottom (+B, -N)
    meta[base + 12] = sNorm; meta[base + 13] = 1.0; meta[base + 14] = 0.0; meta[base + 15] = 0.0;
    // 4: Bottom-Left (-N, -B)
    meta[base + 16] = sNorm; meta[base + 17] = -1.0; meta[base + 18] = -1.0; meta[base + 19] = 0.0;
    // 5: Bottom-Right (-N, +B)
    meta[base + 20] = sNorm; meta[base + 21] = 1.0; meta[base + 22] = -1.0; meta[base + 23] = 0.0;
    // 6: Left-Bottom (-B, -N)
    meta[base + 24] = sNorm; meta[base + 25] = -1.0; meta[base + 26] = 0.0; meta[base + 27] = 0.0;
    // 7: Left-Top (-B, +N)
    meta[base + 28] = sNorm; meta[base + 29] = -1.0; meta[base + 30] = 0.0; meta[base + 31] = 0.0;
  }

  const quadsPerSegment = 4;
  const indices = new Uint32Array((rings - 1) * quadsPerSegment * 6);
  const quadPairs = [
    [0, 1], // Top oxide face
    [2, 3], // Right edge
    [4, 5], // Bottom backing face
    [6, 7], // Left edge
  ];

  let idx = 0;
  for (let r = 0; r < rings - 1; r++) {
    const r0 = r * vertsPerRing;
    const r1 = (r + 1) * vertsPerRing;
    for (let q = 0; q < 4; q++) {
      const a = quadPairs[q][0];
      const b = quadPairs[q][1];
      indices[idx++] = r0 + a;
      indices[idx++] = r1 + a;
      indices[idx++] = r1 + b;
      indices[idx++] = r0 + a;
      indices[idx++] = r1 + b;
      indices[idx++] = r0 + b;
    }
  }

  const geometry = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  const normAttr = new THREE.BufferAttribute(normals, 3);
  normAttr.setUsage(THREE.DynamicDrawUsage);
  const metaAttr = new THREE.BufferAttribute(meta, 4);
  metaAttr.setUsage(THREE.DynamicDrawUsage);

  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('normal', normAttr);
  geometry.setAttribute('aRibbonMeta', metaAttr);
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  return { geometry, posAttr, normAttr, metaAttr };
}

async function bootDay04(): Promise<void> {
  const dayMeta = getDayByNumber(4);
  const stage = await SpecimenStage.create({
    day: dayMeta ?? 4,
    cameraPosition: [0, 0.02, 4.70],
    enableOrbit: false,
  });

  // Natural mineral pigments from src/core/palette.ts
  const boneCalcite = createPigmentUniforms('boneCalcite');
  const warmKaolin = createPigmentUniforms('warmKaolin');
  const yellowOchre = createPigmentUniforms('yellowOchre');
  const realgarAmber = createPigmentUniforms('realgarAmber');
  const cinnabarVermilion = createPigmentUniforms('cinnabarVermilion');
  const rawUmber = createPigmentUniforms('rawUmber');
  const burntSienna = createPigmentUniforms('burntSienna');
  const hematiteRust = createPigmentUniforms('hematiteRust');
  const verdigrisMalachite = createPigmentUniforms('verdigrisMalachite');
  const azuritePrussian = createPigmentUniforms('azuritePrussian');
  const sporeSulfur = createPigmentUniforms('sporeSulfur');

  const uRewindGlow = uniform(0.0);
  const uLeftPackRadius = uniform(0.32);
  const uRightPackRadius = uniform(0.22);

  // Root Cassette, Pencil & Spilled Tape Rig suspended in #212529 Chalky Slate space.
  // Positioned slightly right of center and tilted ~8.5 deg counter-clockwise to match
  // the iconic composition where the tangle cascades down the left flank and bottom
  // while the hexagonal pencil juts out from the right drive hub.
  const cassetteRig = new THREE.Group();
  cassetteRig.name = 'TangledCassetteSpecimen';
  cassetteRig.position.set(0.20, -0.18, 0.0);
  cassetteRig.rotation.set(-0.08, 0.11, 0.15);
  stage.specimenGroup.add(cassetteRig);

  // ===========================================================================
  // 1. SMOKY TRANSLUCENT POLYCARBONATE CASSETTE HOUSING, MAXELL-UR LABEL & REELS
  // ===========================================================================
  const shellGroup = new THREE.Group();
  shellGroup.name = 'CompactCassetteHousing';
  cassetteRig.add(shellGroup);

  // A. Smoky Translucent Dark-Charcoal Polycarbonate Shell Material (Maxell-UR style)
  //    Shows internal molded reel-chamber rings, corner diagonal struts, capstan boss
  //    rings, internal wound tape pancake silhouettes, and crisp beveled rim highlights.
  const shellMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.25,
    metalness: 0.14,
  });
  {
    const pLoc = positionLocal;
    const nLoc = normalLocal;
    const grain = mineralToothFbm(pLoc.mul(22.0), float(2.4));
    const macroMold = mineralToothFbm(pLoc.mul(6.0), float(1.7));

    // Distance to Left & Right Spool Centers for internal reel chamber architecture
    const dLeft = vec2(pLoc.x.sub(LEFT_SPOOL_X), pLoc.y.sub(SPOOL_Y)).length();
    const dRight = vec2(pLoc.x.sub(RIGHT_SPOOL_X), pLoc.y.sub(SPOOL_Y)).length();

    // Internal molded circular reel containment ribs (r = 0.415) visible inside smoky shell
    const leftRingRib = float(1.0).sub(smoothstep(0.005, 0.013, abs(dLeft.sub(0.415))));
    const rightRingRib = float(1.0).sub(smoothstep(0.005, 0.013, abs(dRight.sub(0.415))));

    // Internal diagonal corner stiffening struts & outer perimeter step rib
    const outerStepX = float(1.0).sub(smoothstep(0.006, 0.016, abs(abs(pLoc.x).sub(0.84))));
    const outerStepY = float(1.0).sub(
      smoothstep(0.006, 0.016, min(abs(pLoc.y.sub(1.17)), abs(pLoc.y.sub(0.21))))
    );
    const diagCornerRib = float(1.0)
      .sub(smoothstep(0.006, 0.016, abs(abs(pLoc.x).sub(0.54).sub(abs(pLoc.y.sub(0.69)).mul(0.55)))))
      .mul(step(float(0.65), abs(pLoc.x)));

    // Molded circular bosses around the 4 bottom capstan/pin holes (y = 0.278) & corner screws
    const dCap1 = vec2(abs(pLoc.x).sub(0.49), pLoc.y.sub(0.278)).length();
    const dCap2 = vec2(abs(pLoc.x).sub(0.30), pLoc.y.sub(0.278)).length();
    const capBossRing = float(1.0)
      .sub(smoothstep(0.005, 0.014, abs(dCap1.sub(0.048))))
      .add(float(1.0).sub(smoothstep(0.005, 0.014, abs(dCap2.sub(0.042)))));

    const dCornerScrew = vec2(
      abs(pLoc.x).sub(0.82),
      min(abs(pLoc.y.sub(1.15)), abs(pLoc.y.sub(0.29)))
    ).length();
    const screwBossRing = float(1.0).sub(smoothstep(0.006, 0.016, abs(dCornerScrew.sub(0.046))));

    // Twin lower-corner circular molded guide-roller/pin bosses (|x| = 0.725, y = 0.268) visible in smoky shell
    const dLowerCornerBoss = vec2(abs(pLoc.x).sub(0.725), pLoc.y.sub(0.268)).length();
    const lowerCornerBoss = float(1.0)
      .sub(smoothstep(0.005, 0.015, abs(dLowerCornerBoss.sub(0.064))))
      .add(float(1.0).sub(smoothstep(0.006, 0.018, dLowerCornerBoss.sub(0.022))).mul(0.65));

    // Horizontal molded step ridge across the top of the trapezoidal head shield (y = 0.325)
    const chinTopRidge = float(1.0)
      .sub(smoothstep(0.004, 0.012, abs(pLoc.y.sub(0.325))))
      .mul(float(1.0).sub(smoothstep(0.68, 0.72, abs(pLoc.x))));

    const internalRibMask = clamp(
      leftRingRib
        .add(rightRingRib)
        .add(outerStepX.mul(0.75))
        .add(outerStepY.mul(0.75))
        .add(diagCornerRib.mul(0.65))
        .add(capBossRing.mul(0.85))
        .add(screwBossRing.mul(0.90))
        .add(lowerCornerBoss.mul(0.88))
        .add(chinTopRidge.mul(0.80)),
      0.0,
      1.0
    );

    // Internal dark ferric tape pancake silhouette dimly transmitted through smoky shell
    const inLeftPack = float(1.0)
      .sub(smoothstep(uLeftPackRadius.sub(0.015), uLeftPackRadius.add(0.015), dLeft))
      .mul(smoothstep(0.14, 0.16, dLeft));
    const inRightPack = float(1.0)
      .sub(smoothstep(uRightPackRadius.sub(0.015), uRightPackRadius.add(0.015), dRight))
      .mul(smoothstep(0.14, 0.16, dRight));
    const internalPackShadow = clamp(inLeftPack.add(inRightPack), 0.0, 1.0);

    // Horizontal anti-slip grip fluting + angled side serrations on the raised trapezoidal head block
    const inGripZone = smoothstep(0.16, 0.18, pLoc.y)
      .mul(float(1.0).sub(smoothstep(0.31, 0.328, pLoc.y)))
      .mul(smoothstep(0.5, 0.85, abs(nLoc.z)))
      .mul(float(1.0).sub(smoothstep(0.62, 0.69, abs(pLoc.x))));
    const gripRibs = sin(pLoc.y.mul(420.0)).mul(0.5).add(0.5).mul(inGripZone);

    // Translucent smoky charcoal-slate polycarbonate base
    const smokySlateBase = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.82).add(grain.mul(0.12)).add(macroMold.mul(0.10)),
      azuritePrussian.rInf
    ).mul(0.54);

    const smokyTransmittedTint = mix(warmKaolin.rInf, boneCalcite.rInf, float(0.45)).mul(0.18);
    const ribHighlightCol = mix(warmKaolin.rInf, boneCalcite.rInf, float(0.55)).mul(0.36);

    // Beveled edge specular lift so the dark smoky cassette shell pops against #212529
    const bevelRimMask = float(1.0).sub(abs(nLoc.z)).mul(smoothstep(-0.02, 0.08, pLoc.z));
    const inSplitSeam = float(1.0)
      .sub(smoothstep(0.003, 0.011, abs(pLoc.z)))
      .mul(float(1.0).sub(abs(nLoc.z)));

    const withTranslucency = smokySlateBase
      .add(smokyTransmittedTint.mul(float(1.0).sub(internalPackShadow.mul(0.62))))
      .add(ribHighlightCol.mul(internalRibMask.mul(0.55)))
      .add(boneCalcite.rInf.mul(bevelRimMask.mul(0.14)))
      .sub(rawUmber.rInf.mul(gripRibs.mul(0.06)));

    const seamCol = rawUmber.rInf.mul(0.14);
    shellMat.colorNode = clamp(mix(withTranslucency, seamCol, inSplitSeam.mul(0.75)), 0.0, 1.0);
    shellMat.roughnessNode = clamp(
      float(0.24)
        .add(grain.mul(0.08))
        .sub(internalRibMask.mul(0.05))
        .add(gripRibs.mul(0.09)),
      0.14,
      0.44
    );
  }

  // B. Vintage Maxell-UR Style Cream & Cinnabar-Red "MIXTAPE" Paper Label Material
  const labelMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.58,
    metalness: 0.02,
  });
  {
    const pLoc = positionLocal;
    const paperFiber = mineralToothFbm(pLoc.mul(30.0), float(2.3));
    const wearPatina = mineralToothFbm(pLoc.mul(7.5), float(1.9));

    // Cream Bone Calcite & Warm Kaolin paper stock with subtle vintage handling patina
    const edgeWear = smoothstep(0.68, 0.80, abs(pLoc.x)).add(
      smoothstep(1.04, 1.13, pLoc.y).mul(0.6)
    );
    const paperCol = kubelkaMunkGlaze(
      boneCalcite.K,
      boneCalcite.S,
      float(0.22).add(paperFiber.mul(0.12)).sub(wearPatina.mul(0.07)),
      warmKaolin.rInf
    ).sub(rawUmber.rInf.mul(edgeWear.add(wearPatina.mul(0.35)).mul(0.06)));

    // 1. Left & Right Cinnabar-Vermilion "UR" Side Accent Blocks (|x| in [0.545, 0.778], y in [0.485, 0.775])
    const ax = abs(pLoc.x);
    const inRedBlockX = smoothstep(0.543, 0.548, ax).mul(
      float(1.0).sub(smoothstep(0.774, 0.779, ax))
    );
    const inRedBlockY = smoothstep(0.483, 0.488, pLoc.y).mul(
      float(1.0).sub(smoothstep(0.772, 0.777, pLoc.y))
    );
    const inRedBlock = inRedBlockX.mul(inRedBlockY);

    // Fine cream horizontal stripes inside the upper portion of the red-orange side blocks
    const inBlockStripesX = smoothstep(0.565, 0.570, ax).mul(
      float(1.0).sub(smoothstep(0.752, 0.757, ax))
    );
    const stripe1 = float(1.0).sub(smoothstep(0.003, 0.007, abs(pLoc.y.sub(0.735))));
    const stripe2 = float(1.0).sub(smoothstep(0.003, 0.007, abs(pLoc.y.sub(0.705))));
    const stripe3 = float(1.0).sub(smoothstep(0.003, 0.007, abs(pLoc.y.sub(0.675))));

    // Geometric "U R" block lettering inside the lower half of the red-orange side blocks
    // (Use non-mirrored ux so both left and right blocks read "UR" left-to-right!)
    const ux = mix(pLoc.x, pLoc.x.add(1.323), step(pLoc.x, float(0.0)));
    const urGlyphU = sdSegment2DTsl(vec2(ux, pLoc.y), 0.595, 0.625, 0.595, 0.545)
      .min(sdSegment2DTsl(vec2(ux, pLoc.y), 0.595, 0.545, 0.635, 0.545))
      .min(sdSegment2DTsl(vec2(ux, pLoc.y), 0.635, 0.545, 0.635, 0.625));
    const urGlyphR = sdSegment2DTsl(vec2(ux, pLoc.y), 0.665, 0.545, 0.665, 0.625)
      .min(sdSegment2DTsl(vec2(ux, pLoc.y), 0.665, 0.625, 0.708, 0.625))
      .min(sdSegment2DTsl(vec2(ux, pLoc.y), 0.708, 0.625, 0.708, 0.588))
      .min(sdSegment2DTsl(vec2(ux, pLoc.y), 0.708, 0.588, 0.665, 0.588))
      .min(sdSegment2DTsl(vec2(ux, pLoc.y), 0.685, 0.588, 0.712, 0.545));
    const urKnockout = float(1.0)
      .sub(smoothstep(0.005, 0.009, min(urGlyphU, urGlyphR)))
      .add(clamp(stripe1.add(stripe2).add(stripe3), 0.0, 1.0).mul(inBlockStripesX))
      .mul(inRedBlock);

    // 2. Full-width Cinnabar Horizontal Rule Lines above (y = 0.835) and below (y = 0.448) the window
    const inRuleSpanX = float(1.0).sub(smoothstep(0.770, 0.778, ax));
    const topRedRule = float(1.0)
      .sub(smoothstep(0.004, 0.008, abs(pLoc.y.sub(0.835))))
      .mul(inRuleSpanX);
    const botRedRule = float(1.0)
      .sub(smoothstep(0.005, 0.009, abs(pLoc.y.sub(0.448))))
      .mul(inRuleSpanX);

    // 3. Lower-center "maxell" red-orange typographic brand mark (x in [-0.15, +0.15], y in [0.380, 0.418])
    const bp = vec2(pLoc.x, pLoc.y);
    // m
    const bm = sdSegment2DTsl(bp, -0.145, 0.382, -0.145, 0.408)
      .min(sdSegment2DTsl(bp, -0.145, 0.408, -0.095, 0.408))
      .min(sdSegment2DTsl(bp, -0.120, 0.382, -0.120, 0.408))
      .min(sdSegment2DTsl(bp, -0.095, 0.382, -0.095, 0.408));
    // a
    const ba = sdSegment2DTsl(bp, -0.072, 0.382, -0.042, 0.382)
      .min(sdSegment2DTsl(bp, -0.072, 0.408, -0.042, 0.408))
      .min(sdSegment2DTsl(bp, -0.042, 0.382, -0.042, 0.408))
      .min(sdSegment2DTsl(bp, -0.072, 0.382, -0.072, 0.396))
      .min(sdSegment2DTsl(bp, -0.072, 0.396, -0.042, 0.396));
    // x
    const bx = sdSegment2DTsl(bp, -0.022, 0.382, 0.008, 0.408).min(
      sdSegment2DTsl(bp, -0.022, 0.408, 0.008, 0.382)
    );
    // e
    const be = sdSegment2DTsl(bp, 0.026, 0.382, 0.056, 0.382)
      .min(sdSegment2DTsl(bp, 0.026, 0.408, 0.056, 0.408))
      .min(sdSegment2DTsl(bp, 0.026, 0.395, 0.056, 0.395))
      .min(sdSegment2DTsl(bp, 0.026, 0.382, 0.026, 0.408))
      .min(sdSegment2DTsl(bp, 0.056, 0.395, 0.056, 0.408));
    // l l
    const bl1 = sdSegment2DTsl(bp, 0.078, 0.382, 0.078, 0.422);
    const bl2 = sdSegment2DTsl(bp, 0.102, 0.382, 0.102, 0.422);
    const dBrand = min(min(bm, ba), min(min(bx, be), min(bl1, bl2)));
    const inBrandLogo = float(1.0).sub(smoothstep(0.0045, 0.0075, dBrand));

    // 4. Subtle ruled pen lines & lower-left technical print ticks ("NORMAL POSITION IEC I")
    const ruleLineTop = float(1.0)
      .sub(smoothstep(0.002, 0.005, abs(pLoc.y.sub(0.955))))
      .mul(inRuleSpanX)
      .mul(0.28);
    const techSpecTicks = smoothstep(-0.74, -0.73, pLoc.x)
      .mul(float(1.0).sub(smoothstep(-0.42, -0.41, pLoc.x)))
      .mul(float(1.0).sub(smoothstep(0.004, 0.008, abs(pLoc.y.sub(0.392)))))
      .mul(step(float(-0.2), sin(pLoc.x.mul(160.0))))
      .mul(0.45);

    // 5. Top-Left Dark Charcoal Side "A" Square Badge (x in [-0.752, -0.662], y in [0.875, 0.968])
    const inBadgeX = smoothstep(-0.754, -0.749, pLoc.x).mul(
      float(1.0).sub(smoothstep(-0.664, -0.659, pLoc.x))
    );
    const inBadgeY = smoothstep(0.873, 0.878, pLoc.y).mul(
      float(1.0).sub(smoothstep(0.966, 0.971, pLoc.y))
    );
    const inSideBadge = inBadgeX.mul(inBadgeY);

    const aBadgeDist = sdSegment2DTsl(vec2(pLoc.x, pLoc.y), -0.726, 0.894, -0.707, 0.948)
      .min(sdSegment2DTsl(vec2(pLoc.x, pLoc.y), -0.707, 0.948, -0.688, 0.894))
      .min(sdSegment2DTsl(vec2(pLoc.x, pLoc.y), -0.720, 0.914, -0.694, 0.914));
    const inAGlyph = float(1.0)
      .sub(smoothstep(0.0035, 0.0065, aBadgeDist))
      .mul(inSideBadge);

    // 6. Handwritten Felt-Tip Marker "MIXTAPE" Title across the top cream strip
    //    Slanted slightly upward (+4.2 deg) to match the handwritten reference label!
    const mx = pLoc.x.add(0.08);
    const my = pLoc.y.sub(0.965).sub(mx.mul(0.075));
    const mp = vec2(mx, my);

    // M (around x = -0.31)
    const dM = sdSegment2DTsl(mp, -0.348, -0.044, -0.342, 0.046)
      .min(sdSegment2DTsl(mp, -0.342, 0.046, -0.312, -0.012))
      .min(sdSegment2DTsl(mp, -0.312, -0.012, -0.282, 0.048))
      .min(sdSegment2DTsl(mp, -0.282, 0.048, -0.275, -0.044));
    // I (around x = -0.22)
    const dI = sdSegment2DTsl(mp, -0.222, -0.042, -0.216, 0.045);
    // X (around x = -0.14)
    const dX = sdSegment2DTsl(mp, -0.172, -0.042, -0.114, 0.046).min(
      sdSegment2DTsl(mp, -0.168, 0.044, -0.112, -0.042)
    );
    // T (around x = -0.04)
    const dT = sdSegment2DTsl(mp, -0.045, -0.044, -0.038, 0.044).min(
      sdSegment2DTsl(mp, -0.076, 0.042, -0.004, 0.047)
    );
    // A (around x = +0.06)
    const dA = sdSegment2DTsl(mp, 0.024, -0.044, 0.058, 0.047)
      .min(sdSegment2DTsl(mp, 0.058, 0.047, 0.088, -0.042))
      .min(sdSegment2DTsl(mp, 0.034, -0.006, 0.078, -0.002));
    // P (around x = +0.16)
    const dP = sdSegment2DTsl(mp, 0.132, -0.044, 0.136, 0.046)
      .min(sdSegment2DTsl(mp, 0.136, 0.046, 0.180, 0.038))
      .min(sdSegment2DTsl(mp, 0.180, 0.038, 0.180, 0.006))
      .min(sdSegment2DTsl(mp, 0.180, 0.006, 0.134, 0.002));
    // E (around x = +0.26)
    const dE = sdSegment2DTsl(mp, 0.228, -0.042, 0.233, 0.046)
      .min(sdSegment2DTsl(mp, 0.233, 0.046, 0.280, 0.048))
      .min(sdSegment2DTsl(mp, 0.230, 0.004, 0.272, 0.007))
      .min(sdSegment2DTsl(mp, 0.228, -0.042, 0.282, -0.039));

    const dMixtape = min(min(min(dM, dI), min(dX, dT)), min(min(dA, dP), dE));
    const markerInkMask = float(1.0).sub(
      smoothstep(0.0042, 0.0082, dMixtape.add(paperFiber.sub(0.5).mul(0.0018)))
    );

    const cinnabarRedCol = kubelkaMunkGlaze(
      cinnabarVermilion.K,
      cinnabarVermilion.S,
      float(0.82),
      realgarAmber.rInf
    );
    const charcoalInkCol = rawUmber.rInf.mul(0.24);

    const redAccentMask = clamp(
      inRedBlock
        .sub(urKnockout.mul(0.92))
        .add(topRedRule)
        .add(botRedRule)
        .add(inBrandLogo),
      0.0,
      1.0
    );

    const darkInkTotal = clamp(
      ruleLineTop
        .add(techSpecTicks)
        .add(inSideBadge.sub(inAGlyph.mul(0.92)))
        .add(markerInkMask.mul(0.92)),
      0.0,
      1.0
    );

    const withRedAccents = mix(paperCol, cinnabarRedCol, redAccentMask);
    labelMat.colorNode = mix(withRedAccents, charcoalInkCol, darkInkTotal);
  }

  // C. Molded Window Bezel & Graphite Slip-Sheet Materials
  const darkBezelMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.32,
    metalness: 0.16,
  });
  darkBezelMat.colorNode = mix(rawUmber.rInf.mul(0.36), warmKaolin.rInf.mul(0.28), float(0.35));

  const graphiteSlipSheetMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.40,
    metalness: 0.24,
  });
  {
    const pLoc = positionLocal;
    const diagRibA = float(1.0).sub(
      smoothstep(0.006, 0.014, abs(abs(pLoc.x).sub(0.18).sub(abs(pLoc.y.sub(0.70)).mul(0.85))))
    );
    const horizRib = float(1.0).sub(smoothstep(0.005, 0.012, abs(pLoc.y.sub(0.70))));
    const baseGraphite = kubelkaMunkGlaze(
      azuritePrussian.K,
      azuritePrussian.S,
      float(1.30),
      rawUmber.rInf
    ).mul(0.30);
    graphiteSlipSheetMat.colorNode = baseGraphite.add(
      rawUmber.rInf.mul(diagRibA.add(horizRib).mul(0.06))
    );
  }

  // Smoked Acrylic Crystal Window Lens Material
  const smokedGlassMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  });
  {
    const pLoc = positionLocal;
    const band1 = float(1.0).sub(
      smoothstep(0.015, 0.085, abs(pLoc.x.mul(0.52).add(pLoc.y).sub(0.11)))
    );
    const band2 = float(1.0).sub(
      smoothstep(0.008, 0.035, abs(pLoc.x.mul(0.52).add(pLoc.y).add(0.02)))
    );
    const centerVerticalSeam = float(1.0).sub(smoothstep(0.006, 0.018, abs(pLoc.x)));
    const topBevelRim = smoothstep(0.18, 0.22, abs(pLoc.y));
    const glassAlpha = clamp(
      band1
        .mul(0.14)
        .add(band2.mul(0.08))
        .add(centerVerticalSeam.mul(0.12))
        .add(topBevelRim.mul(0.09)),
      0.0,
      0.26
    );
    smokedGlassMat.colorNode = boneCalcite.rInf;
    smokedGlassMat.opacityNode = glassAlpha;
  }

  const spindleHoleMat = new THREE.MeshBasicNodeMaterial();
  spindleHoleMat.colorNode = vec3(0.0155, 0.0185, 0.022);

  // Brass, Steel & Felt Hardware Materials
  const brassHardwareMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.22,
    metalness: 0.72,
  });
  brassHardwareMat.colorNode = kubelkaMunkGlaze(
    yellowOchre.K,
    yellowOchre.S,
    float(0.42),
    boneCalcite.rInf
  );

  const steelShieldMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.24,
    metalness: 0.80,
  });
  steelShieldMat.colorNode = mix(boneCalcite.rInf, rawUmber.rInf, float(0.45));

  const screwWellMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.55,
    metalness: 0.15,
  });
  screwWellMat.colorNode = rawUmber.rInf.mul(0.20);

  const feltPadMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.88,
    metalness: 0.04,
  });
  feltPadMat.colorNode = kubelkaMunkGlaze(
    burntSienna.K,
    burntSienna.S,
    float(0.65),
    realgarAmber.rInf
  );

  // ---------------------------------------------------------------------------
  // D. BUILD EXTRUDED BEVELED FRONT & BACK CASSETTE SHELL HALVES
  // ---------------------------------------------------------------------------
  const shellOuterShape = new THREE.Shape();
  {
    const x0 = -0.90;
    const x1 = 0.90;
    const y0 = 0.15;
    const y1 = 1.23;
    const cr = 0.045;

    shellOuterShape.moveTo(x0 + cr, y0);
    shellOuterShape.lineTo(-0.62, y0);
    shellOuterShape.lineTo(-0.62, 0.245);
    shellOuterShape.lineTo(0.62, 0.245);
    shellOuterShape.lineTo(0.62, y0);
    shellOuterShape.lineTo(x1 - cr, y0);
    shellOuterShape.quadraticCurveTo(x1, y0, x1, y0 + cr);
    shellOuterShape.lineTo(x1, y1 - cr);
    shellOuterShape.quadraticCurveTo(x1, y1, x1 - cr, y1);
    shellOuterShape.lineTo(x0 + cr, y1);
    shellOuterShape.quadraticCurveTo(x0, y1, x0, y1 - cr);
    shellOuterShape.lineTo(x0, y0 + cr);
    shellOuterShape.quadraticCurveTo(x0, y0, x0 + cr, y0);
  }
  // Central Maxell-UR style rounded window cutout hole (`w = 1.06, h = 0.34` centered at `(0, 0.64)`)
  const winHolePath = createRoundedRectPath(0, 0.64, 1.06, 0.36, 0.045, false) as THREE.Path;
  shellOuterShape.holes.push(winHolePath);

  const shellExtrudeOpts: THREE.ExtrudeGeometryOptions = {
    depth: 0.068,
    bevelEnabled: true,
    bevelThickness: 0.010,
    bevelSize: 0.010,
    bevelSegments: 3,
    curveSegments: 16,
  };

  const frontShellGeo = new THREE.ExtrudeGeometry(shellOuterShape, shellExtrudeOpts);
  frontShellGeo.translate(0, 0, 0.014);
  const frontShellMesh = new THREE.Mesh(frontShellGeo, shellMat);
  frontShellMesh.castShadow = true;
  frontShellMesh.receiveShadow = true;
  shellGroup.add(frontShellMesh);

  const backShellGeo = new THREE.ExtrudeGeometry(shellOuterShape, shellExtrudeOpts);
  backShellGeo.translate(0, 0, -0.082);
  const backShellMesh = new THREE.Mesh(backShellGeo, shellMat);
  backShellMesh.castShadow = true;
  backShellMesh.receiveShadow = true;
  shellGroup.add(backShellMesh);

  // Top Write-Protect Knockout Tab Cavities + Intact Record Tabs
  [-0.72, 0.72].forEach((tx) => {
    const notchGeo = new THREE.BoxGeometry(0.13, 0.045, 0.14);
    const notchMesh = new THREE.Mesh(notchGeo, screwWellMat);
    notchMesh.position.set(tx, 1.212, 0.0);
    shellGroup.add(notchMesh);

    const tabGeo = new THREE.BoxGeometry(0.055, 0.038, 0.11);
    const tabMesh = new THREE.Mesh(tabGeo, shellMat);
    tabMesh.position.set(tx - Math.sign(tx) * 0.03, 1.212, 0.0);
    shellGroup.add(tabMesh);
  });

  // Side Handling / Tape-Type Sensor Notches on Left & Right Outer Edges
  [-0.895, 0.895].forEach((sx) => {
    const sideNotchGeo = new THREE.BoxGeometry(0.032, 0.075, 0.14);
    const sideNotchMesh = new THREE.Mesh(sideNotchGeo, screwWellMat);
    sideNotchMesh.position.set(sx, 0.48, 0.0);
    shellGroup.add(sideNotchMesh);
  });

  // Internal Graphite Anti-Friction Slip-Sheet Liner Plate behind the reels
  const slipSheetGeo = new THREE.BoxGeometry(1.70, 0.96, 0.012);
  slipSheetGeo.translate(0, 0.70, -0.048);
  const slipSheetMesh = new THREE.Mesh(slipSheetGeo, graphiteSlipSheetMat);
  shellGroup.add(slipSheetMesh);

  // ---------------------------------------------------------------------------
  // E. DIE-CUT RECESSED MAXELL-UR PAPER LABEL & MOLDED WINDOW BEZEL
  // ---------------------------------------------------------------------------
  const labelShape = createRoundedRectPath(0, 0.74, 1.56, 0.76, 0.038, true) as THREE.Shape;
  const labelWinHole = createRoundedRectPath(0, 0.64, 1.09, 0.39, 0.048, false) as THREE.Path;
  labelShape.holes.push(labelWinHole);

  const labelGeo = new THREE.ExtrudeGeometry(labelShape, {
    depth: 0.004,
    bevelEnabled: true,
    bevelThickness: 0.0015,
    bevelSize: 0.0015,
    bevelSegments: 2,
    curveSegments: 14,
  });
  labelGeo.translate(0, 0, 0.091);
  const labelMesh = new THREE.Mesh(labelGeo, labelMat);
  labelMesh.receiveShadow = true;
  shellGroup.add(labelMesh);

  // Molded Window Bezel Frame surrounding the central stadium/rounded window
  const bezelOuterShape = createRoundedRectPath(0, 0.64, 1.10, 0.40, 0.048, true) as THREE.Shape;
  const bezelInnerHole = createRoundedRectPath(0, 0.64, 1.03, 0.33, 0.038, false) as THREE.Path;
  bezelOuterShape.holes.push(bezelInnerHole);

  const bezelGeo = new THREE.ExtrudeGeometry(bezelOuterShape, {
    depth: 0.016,
    bevelEnabled: true,
    bevelThickness: 0.004,
    bevelSize: 0.004,
    bevelSegments: 2,
    curveSegments: 14,
  });
  bezelGeo.translate(0, 0, 0.082);
  const bezelMesh = new THREE.Mesh(bezelGeo, darkBezelMat);
  shellGroup.add(bezelMesh);

  // Recessed Smoked-Glass Window Lens Pane (`z = +0.074`)
  const glassPaneGeo = new THREE.PlaneGeometry(1.04, 0.34);
  const glassPaneMesh = new THREE.Mesh(glassPaneGeo, smokedGlassMat);
  glassPaneMesh.position.set(0, 0.64, 0.074);
  shellGroup.add(glassPaneMesh);

  // Horizontal Window Guide Bars framing the central gauge scale
  const scaleRailGeo = new THREE.BoxGeometry(0.36, 0.008, 0.006);
  [0.705, 0.585].forEach((ry) => {
    const rail = new THREE.Mesh(scaleRailGeo, darkBezelMat);
    rail.position.set(0, ry, 0.075);
    shellGroup.add(rail);
  });

  // ---------------------------------------------------------------------------
  // F. SCULPTED 14°-TAPERED TRAPEZOIDAL HEAD SHIELD ("THE CHIN") & 5 OPEN BAYS
  // ---------------------------------------------------------------------------
  const chinShape = new THREE.Shape();
  const bayWindows: [number, number][] = [
    [-0.58, -0.45],
    [-0.39, -0.21],
    [-0.14, 0.14],
    [0.21, 0.39],
    [0.45, 0.58],
  ];
  chinShape.moveTo(-0.65, 0.14);
  bayWindows.forEach(([bx0, bx1]) => {
    chinShape.lineTo(bx0, 0.14);
    chinShape.lineTo(bx0, 0.232);
    chinShape.lineTo(bx1, 0.232);
    chinShape.lineTo(bx1, 0.14);
  });
  chinShape.lineTo(0.65, 0.14);
  chinShape.lineTo(0.71, 0.325);
  chinShape.lineTo(-0.71, 0.325);
  chinShape.closePath();

  const chinGeo = new THREE.ExtrudeGeometry(chinShape, {
    depth: 0.216,
    bevelEnabled: true,
    bevelThickness: 0.008,
    bevelSize: 0.006,
    bevelSegments: 3,
  });
  chinGeo.translate(0, 0, -0.108);
  const chinMesh = new THREE.Mesh(chinGeo, shellMat);
  chinMesh.castShadow = true;
  chinMesh.receiveShadow = true;
  shellGroup.add(chinMesh);

  // Dark interior head-bay ceiling & back wall inside the trapezoid
  const headBayRecessGeo = new THREE.BoxGeometry(1.24, 0.10, 0.15);
  headBayRecessGeo.translate(0, 0.195, -0.025);
  const headBayRecessMesh = new THREE.Mesh(headBayRecessGeo, screwWellMat);
  shellGroup.add(headBayRecessMesh);

  // 4 Recessed Capstan & Reference Alignment Pin Apertures with Molded Boss Collars
  [
    { x: -0.49, r: 0.034 },
    { x: -0.30, r: 0.028 },
    { x: 0.30, r: 0.028 },
    { x: 0.49, r: 0.034 },
  ].forEach(({ x, r }) => {
    const collarGeo = new THREE.RingGeometry(r * 0.95, r * 1.38, 22);
    const collarMesh = new THREE.Mesh(collarGeo, darkBezelMat);
    collarMesh.position.set(x, 0.278, 0.119);
    shellGroup.add(collarMesh);

    const holeGeo = new THREE.CylinderGeometry(r, r, 0.242, 18);
    holeGeo.rotateX(Math.PI * 0.5);
    const holeMesh = new THREE.Mesh(holeGeo, spindleHoleMat);
    holeMesh.position.set(x, 0.278, 0.0);
    shellGroup.add(holeMesh);
  });

  // ---------------------------------------------------------------------------
  // G. INTERNAL HEAD-BAY MECHANICS (MU-METAL SHIELD, SPRING, FELT PAD & ROLLERS)
  // ---------------------------------------------------------------------------
  const muShieldGeo = new THREE.BoxGeometry(0.26, 0.085, 0.12);
  const muShieldMesh = new THREE.Mesh(muShieldGeo, steelShieldMat);
  muShieldMesh.position.set(0, 0.205, -0.015);
  shellGroup.add(muShieldMesh);

  const leafSpringGeo = new THREE.BoxGeometry(0.56, 0.009, 0.085);
  const leafSpringMesh = new THREE.Mesh(leafSpringGeo, brassHardwareMat);
  leafSpringMesh.position.set(0, 0.202, 0.008);
  shellGroup.add(leafSpringMesh);

  const feltPadGeo = new THREE.BoxGeometry(0.115, 0.030, 0.095);
  const feltPadMesh = new THREE.Mesh(feltPadGeo, feltPadMat);
  feltPadMesh.position.set(0, 0.186, 0.012);
  shellGroup.add(feltPadMesh);

  const guidePinGeo = new THREE.CylinderGeometry(0.016, 0.016, 0.16, 14);
  guidePinGeo.rotateX(Math.PI * 0.5);
  [
    [-0.30, 0.195],
    [0.30, 0.195],
    [-0.56, 0.49],
    [0.56, 0.49],
  ].forEach(([px, py]) => {
    const pin = new THREE.Mesh(guidePinGeo, steelShieldMat);
    pin.position.set(px, py, 0.0);
    shellGroup.add(pin);
  });

  function createGroovedGuideRoller(xPos: number): THREE.Group {
    const g = new THREE.Group();
    g.position.set(xPos, THROAT_EXIT_Y + 0.022, 0.0);
    shellGroup.add(g);

    const barrelGeo = new THREE.CylinderGeometry(0.038, 0.038, 0.095, 20);
    barrelGeo.rotateX(Math.PI * 0.5);
    g.add(new THREE.Mesh(barrelGeo, brassHardwareMat));

    const flangeGeo = new THREE.CylinderGeometry(0.048, 0.048, 0.016, 20);
    flangeGeo.rotateX(Math.PI * 0.5);
    [-0.046, 0.046].forEach((fz) => {
      const fl = new THREE.Mesh(flangeGeo, brassHardwareMat);
      fl.position.z = fz;
      g.add(fl);
    });

    const axleGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.14, 12);
    axleGeo.rotateX(Math.PI * 0.5);
    g.add(new THREE.Mesh(axleGeo, steelShieldMat));

    return g;
  }

  createGroovedGuideRoller(LEFT_GUIDE_X);
  createGroovedGuideRoller(RIGHT_GUIDE_X);

  function createPhillipsScrew(sx: number, sy: number, sz: number): void {
    const screwGroup = new THREE.Group();
    screwGroup.position.set(sx, sy, sz);
    shellGroup.add(screwGroup);

    const wellGeo = new THREE.RingGeometry(0.022, 0.036, 18);
    const wellMesh = new THREE.Mesh(wellGeo, screwWellMat);
    wellMesh.position.z = 0.001;
    screwGroup.add(wellMesh);

    const headGeo = new THREE.CylinderGeometry(0.023, 0.025, 0.012, 16);
    headGeo.rotateX(Math.PI * 0.5);
    const headMesh = new THREE.Mesh(headGeo, steelShieldMat);
    screwGroup.add(headMesh);

    const slotHGeo = new THREE.BoxGeometry(0.028, 0.005, 0.006);
    const slotVGeo = new THREE.BoxGeometry(0.005, 0.028, 0.006);
    const slotH = new THREE.Mesh(slotHGeo, screwWellMat);
    const slotV = new THREE.Mesh(slotVGeo, screwWellMat);
    slotH.position.z = 0.005;
    slotV.position.z = 0.005;
    screwGroup.add(slotH, slotV);
  }

  createPhillipsScrew(-0.82, 1.15, 0.092);
  createPhillipsScrew(0.82, 1.15, 0.092);
  createPhillipsScrew(-0.82, 0.25, 0.092);
  createPhillipsScrew(0.82, 0.25, 0.092);
  createPhillipsScrew(0.0, 0.275, 0.118);

  // ---------------------------------------------------------------------------
  // H. TWIN 6-TOOTHED DELRIN REEL HUBS & ANISOTROPIC BOW-TIE TAPE PANCAKES
  // ---------------------------------------------------------------------------
  const tapePackMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.17,
    metalness: 0.34,
  });
  {
    const pLoc = positionLocal;
    const r = pLoc.x.mul(pLoc.x).add(pLoc.y.mul(pLoc.y)).sqrt();
    const angle = atan(pLoc.y, pLoc.x);

    const grooves = sin(r.mul(480.0)).mul(0.5).add(0.5);
    const bowTieHighlight = cos(angle.mul(2.0).sub(0.65)).mul(0.5).add(0.5);
    const bowTieLobe = bowTieHighlight.mul(bowTieHighlight).mul(0.42);

    const radialVignette = smoothstep(0.11, 0.135, r).mul(
      float(1.0).sub(smoothstep(0.31, 0.34, r).mul(0.35))
    );

    const packBase = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.88).add(grooves.mul(0.28)).sub(bowTieLobe.mul(0.25)),
      burntSienna.rInf
    );
    const sheenCol = mix(burntSienna.rInf, yellowOchre.rInf, float(0.48)).mul(bowTieLobe);

    tapePackMat.colorNode = packBase
      .add(sheenCol)
      .mul(float(0.62).add(grooves.mul(0.16)))
      .mul(float(0.75).add(radialVignette.mul(0.25)));
    tapePackMat.roughnessNode = clamp(
      float(0.19).sub(bowTieLobe.mul(0.08)).add(grooves.mul(0.04)),
      0.09,
      0.29
    );
  }

  const hubSprocketMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.28,
    metalness: 0.12,
  });
  hubSprocketMat.colorNode = mix(
    boneCalcite.rInf,
    sporeSulfur.rInf,
    uRewindGlow.mul(0.26)
  );

  const hubRecessMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.38,
    metalness: 0.10,
  });
  hubRecessMat.colorNode = mix(warmKaolin.rInf, rawUmber.rInf, float(0.28));

  const clampAccentMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.3,
    metalness: 0.2,
  });
  clampAccentMat.colorNode = cinnabarVermilion.rInf;

  // 7 Precision Window Gauge Scale Ticks ("100 | | 50 | | 0") between the two reels
  const tickGeo = new THREE.BoxGeometry(0.006, 0.034, 0.008);
  [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15].forEach((tx, idx) => {
    const tm = new THREE.Mesh(tickGeo, hubSprocketMat);
    tm.scale.y = idx % 3 === 0 ? 1.45 : 0.80;
    tm.position.set(tx, 0.645, 0.076);
    shellGroup.add(tm);
  });

  function createReelAssembly(xPos: number, yPos: number): {
    group: THREE.Group;
    packMesh: THREE.Mesh;
  } {
    const group = new THREE.Group();
    group.position.set(xPos, yPos, 0.0);
    shellGroup.add(group);

    // 1. Wound ferric-oxide magnetic tape pancake (base radius = 0.33, scaled dynamically)
    const packGeo = new THREE.CylinderGeometry(0.33, 0.33, TAPE_WIDTH, 52);
    packGeo.rotateX(Math.PI * 0.5);
    const packMesh = new THREE.Mesh(packGeo, tapePackMat);
    group.add(packMesh);

    // 2. 3D Sculpted Delrin Hub Flange with 6 Slotted Spokes & Circular Lightening Holes
    const hubFlangeShape = new THREE.Shape();
    hubFlangeShape.absarc(0, 0, 0.144, 0, Math.PI * 2, false);
    const spindleBoreHole = new THREE.Path();
    spindleBoreHole.absarc(0, 0, 0.098, 0, Math.PI * 2, true);
    hubFlangeShape.holes.push(spindleBoreHole);

    for (let h = 0; h < 6; h++) {
      const ha = ((h + 0.5) / 6) * Math.PI * 2;
      const hole = new THREE.Path();
      hole.absarc(Math.cos(ha) * 0.121, Math.sin(ha) * 0.121, 0.012, 0, Math.PI * 2, true);
      hubFlangeShape.holes.push(hole);
    }

    const hubFlangeGeo = new THREE.ExtrudeGeometry(hubFlangeShape, {
      depth: 0.092,
      bevelEnabled: true,
      bevelThickness: 0.004,
      bevelSize: 0.003,
      bevelSegments: 2,
      curveSegments: 28,
    });
    hubFlangeGeo.translate(0, 0, -0.046);
    const hubFlangeMesh = new THREE.Mesh(hubFlangeGeo, hubSprocketMat);
    group.add(hubFlangeMesh);

    const outerLipGeo = new THREE.RingGeometry(0.135, 0.146, 36);
    const outerLipMesh = new THREE.Mesh(outerLipGeo, hubRecessMat);
    outerLipMesh.position.z = 0.051;
    group.add(outerLipMesh);

    const boreGeo = new THREE.CircleGeometry(0.100, 32);
    const boreMesh = new THREE.Mesh(boreGeo, spindleHoleMat);
    boreMesh.position.z = 0.044;
    group.add(boreMesh);

    // 6 sculpted 3D inward-pointing cassette drive spline teeth
    const toothGeo = new THREE.BoxGeometry(0.022, 0.040, 0.098);
    for (let t = 0; t < 6; t++) {
      const a = (t / 6) * Math.PI * 2;
      const tm = new THREE.Mesh(toothGeo, hubSprocketMat);
      tm.position.set(Math.cos(a) * 0.083, Math.sin(a) * 0.083, 0.002);
      tm.rotation.z = a + Math.PI * 0.5;
      group.add(tm);
    }

    // Cinnabar Vermilion leader-tape C-clamp wedge slotted into the hub rim
    const clampGeo = new THREE.BoxGeometry(0.036, 0.022, 0.100);
    const clampMesh = new THREE.Mesh(clampGeo, clampAccentMat);
    clampMesh.position.set(0.122, 0, 0.002);
    group.add(clampMesh);

    return { group, packMesh };
  }

  const REEL_Y = 0.645;
  const leftReel = createReelAssembly(LEFT_SPOOL_X, REEL_Y);
  const rightReel = createReelAssembly(RIGHT_SPOOL_X, REEL_Y);

  // ===========================================================================
  // 2. HEXAGONAL #2 WOODEN PENCIL WEDGED IN THE RIGHT-HAND 6-TOOTHED SPOOL HUB
  // ===========================================================================
  // Seated through the right reel hub bore (RIGHT_SPOOL_X, REEL_Y, 0.015) so the
  // sharpened point passes inside the hub and the 6-sided yellow hexagonal shaft
  // wedges directly against the 6 white Delrin hub teeth—matching the reference
  // photograph and real-world cassette winding mechanics.
  const pencilSocketGroup = new THREE.Group();
  pencilSocketGroup.name = 'PencilRewindToolSocket';
  pencilSocketGroup.position.set(RIGHT_SPOOL_X, REEL_Y, 0.015);
  shellGroup.add(pencilSocketGroup);

  const pencilShaftGroup = new THREE.Group();
  pencilShaftGroup.name = 'HexagonalPencilBody';
  pencilSocketGroup.add(pencilShaftGroup);

  // A. Graphite Core Lead Tip (`y in [-0.18, -0.09]`) extending inside the back of the hub bore
  const graphiteLeadMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.24,
    metalness: 0.52,
  });
  graphiteLeadMat.colorNode = kubelkaMunkGlaze(
    azuritePrussian.K,
    azuritePrussian.S,
    float(1.45),
    rawUmber.rInf
  ).mul(0.32);

  const leadTipGeo = new THREE.CylinderGeometry(0.024, 0.006, 0.09, 18);
  leadTipGeo.translate(0, -0.135, 0);
  const leadTipMesh = new THREE.Mesh(leadTipGeo, graphiteLeadMat);
  leadTipMesh.castShadow = true;
  pencilShaftGroup.add(leadTipMesh);

  // B. Sharpened Raw Cedar Wood Cone (`y in [-0.09, +0.035]`) seated inside the 6 hub teeth
  const pencilWoodMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.58,
    metalness: 0.03,
  });
  {
    const pLoc = positionLocal;
    const woodFiber = mineralToothFbm(vec3(pLoc.x.mul(45.0), pLoc.y.mul(14.0), pLoc.z.mul(45.0)), float(2.4));
    const cedarCol = kubelkaMunkGlaze(
      burntSienna.K,
      burntSienna.S,
      float(0.32).add(woodFiber.mul(0.16)),
      warmKaolin.rInf
    );
    const ochreTint = mix(cedarCol, yellowOchre.rInf, float(0.30));
    pencilWoodMat.colorNode = ochreTint;
  }

  const woodConeGeo = new THREE.CylinderGeometry(0.076, 0.024, 0.125, 24);
  woodConeGeo.translate(0, -0.0275, 0);
  const woodConeMesh = new THREE.Mesh(woodConeGeo, pencilWoodMat);
  woodConeMesh.castShadow = true;
  woodConeMesh.receiveShadow = true;
  pencilShaftGroup.add(woodConeMesh);

  // C. 6-Sided Hexagonal Yellow-Ochre Lacquered Cedar Shaft (`y in [+0.035, +0.765]`)
  const pencilLacquerMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.22,
    metalness: 0.06,
  });
  {
    const pLoc = positionLocal;
    const nLoc = normalLocal;
    const nWorld = normalWorld;
    const grain = mineralToothFbm(vec3(pLoc.x.mul(18.0), pLoc.y.mul(8.0), pLoc.z.mul(18.0)), float(2.0));

    // Vibrant #2 Golden Yellow Ochre + Spore Sulfur + Realgar Amber lacquer
    const baseYellow = kubelkaMunkGlaze(
      yellowOchre.K,
      yellowOchre.S,
      float(0.56).add(grain.mul(0.10)),
      sporeSulfur.rInf
    );
    const warmAmberFacet = kubelkaMunkGlaze(
      realgarAmber.K,
      realgarAmber.S,
      float(0.34),
      sporeSulfur.rInf
    );

    const hexAngle = atan(pLoc.z, pLoc.x);
    const hexRidge = cos(hexAngle.mul(6.0)).mul(0.5).add(0.5);
    const ridgeEdgeMask = smoothstep(0.84, 0.98, hexRidge);

    // Subtle dark stamped foil branding ticks along the front facet ("No. 2 / HB")
    const onFrontFacet = smoothstep(0.75, 0.92, nLoc.z);
    const inStampSpan = smoothstep(0.26, 0.28, pLoc.y).mul(
      float(1.0).sub(smoothstep(0.54, 0.56, pLoc.y))
    );
    const stampTicks = step(float(0.25), sin(pLoc.y.mul(190.0)))
      .mul(float(1.0).sub(smoothstep(0.010, 0.016, abs(pLoc.x))))
      .mul(onFrontFacet)
      .mul(inStampSpan);

    // Studio softbox facet highlight so the 6 hexagonal faces read crisply
    const facetSun = smoothstep(0.15, 0.85, nWorld.z.mul(0.55).add(nWorld.y.mul(0.45)).sub(nWorld.x.mul(0.25)));
    const facetedYellow = mix(baseYellow, warmAmberFacet, float(0.26).add(ridgeEdgeMask.mul(0.22)))
      .add(sporeSulfur.rInf.mul(facetSun.mul(0.20)));
    const withRidgeHighlight = mix(facetedYellow, boneCalcite.rInf, ridgeEdgeMask.mul(0.24));
    pencilLacquerMat.colorNode = clamp(
      mix(withRidgeHighlight, rawUmber.rInf.mul(0.25), stampTicks.mul(0.62)),
      0.0,
      1.0
    );
    pencilLacquerMat.roughnessNode = clamp(
      float(0.20).add(grain.mul(0.07)).sub(ridgeEdgeMask.mul(0.06)),
      0.13,
      0.32
    );
  }

  // Non-indexed hexagonal cylinder so each of the 6 flat facets has crisp flat normals!
  const hexShaftGeo = new THREE.CylinderGeometry(0.078, 0.078, 0.73, 6, 1, false).toNonIndexed();
  hexShaftGeo.computeVertexNormals();
  hexShaftGeo.translate(0, 0.40, 0);
  const hexShaftMesh = new THREE.Mesh(hexShaftGeo, pencilLacquerMat);
  hexShaftMesh.castShadow = true;
  hexShaftMesh.receiveShadow = true;
  pencilShaftGroup.add(hexShaftMesh);

  // D. Crimped Brass & Verdigris-Banded Metallic Ferrule (`y in [+0.765, +0.905]`)
  const ferruleMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.22,
    metalness: 0.74,
  });
  {
    const pLoc = positionLocal;
    // Circumferential crimp corrugation waves along the ferrule (`y in [0.765, 0.905]`)
    const crimpWave = sin(pLoc.y.mul(270.0)).mul(0.5).add(0.5);
    // Classic twin dark verdigris/umber painted stripes around the center of the ferrule
    const band1 = float(1.0).sub(smoothstep(0.011, 0.017, abs(pLoc.y.sub(0.816))));
    const band2 = float(1.0).sub(smoothstep(0.011, 0.017, abs(pLoc.y.sub(0.854))));
    const inGreenBand = clamp(band1.add(band2), 0.0, 1.0);

    const brassCol = kubelkaMunkGlaze(
      yellowOchre.K,
      yellowOchre.S,
      float(0.42).add(crimpWave.mul(0.18)),
      boneCalcite.rInf
    );
    const darkBandCol = kubelkaMunkGlaze(
      verdigrisMalachite.K,
      verdigrisMalachite.S,
      float(1.15),
      rawUmber.rInf
    ).mul(0.34);

    ferruleMat.colorNode = mix(
      brassCol.mul(float(0.86).add(crimpWave.mul(0.22))),
      darkBandCol,
      inGreenBand.mul(0.90)
    );
    ferruleMat.roughnessNode = clamp(
      float(0.20).add(inGreenBand.mul(0.14)).sub(crimpWave.mul(0.05)),
      0.12,
      0.42
    );
  }

  const ferruleGeo = new THREE.CylinderGeometry(0.080, 0.080, 0.14, 32);
  ferruleGeo.translate(0, 0.835, 0);
  const ferruleMesh = new THREE.Mesh(ferruleGeo, ferruleMat);
  ferruleMesh.castShadow = true;
  ferruleMesh.receiveShadow = true;
  pencilShaftGroup.add(ferruleMesh);

  // E. Rounded Terracotta-Cinnabar Rubber Eraser (`y in [+0.905, +1.025]`)
  const eraserMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.78,
    metalness: 0.02,
  });
  {
    const pLoc = positionLocal;
    const rubberTooth = mineralToothFbm(pLoc.mul(36.0), float(2.2));
    const pinkTerracotta = kubelkaMunkGlaze(
      cinnabarVermilion.K,
      cinnabarVermilion.S,
      float(0.28).add(rubberTooth.mul(0.09)),
      warmKaolin.rInf
    );
    eraserMat.colorNode = mix(pinkTerracotta, boneCalcite.rInf, float(0.28));
  }

  const eraserBodyGeo = new THREE.CylinderGeometry(0.075, 0.076, 0.080, 28);
  eraserBodyGeo.translate(0, 0.945, 0);
  const eraserBodyMesh = new THREE.Mesh(eraserBodyGeo, eraserMat);
  eraserBodyMesh.castShadow = true;
  pencilShaftGroup.add(eraserBodyMesh);

  const eraserCapGeo = new THREE.SphereGeometry(0.075, 28, 16);
  eraserCapGeo.scale(1.0, 0.65, 1.0);
  eraserCapGeo.translate(0, 0.985, 0);
  const eraserCapMesh = new THREE.Mesh(eraserCapGeo, eraserMat);
  eraserCapMesh.castShadow = true;
  pencilShaftGroup.add(eraserCapMesh);

  // Pencil crank state:
  // Default resting azimuth ~0.94 rad (~54 deg up-and-right in cassette local XY)
  // and polar tilt ~0.66 rad (~38 deg off the +Z normal), matching the reference image!
  const PENCIL_REST_AZIMUTH = 0.94;
  const PENCIL_TILT_RAD = 0.66;
  let pencilAzimuth = PENCIL_REST_AZIMUTH;
  let pencilHoverTilt = 0.0;

  function updatePencilTransform(reelAngle: number): void {
    const tilt = PENCIL_TILT_RAD + pencilHoverTilt;
    const sinTilt = Math.sin(tilt);
    const cosTilt = Math.cos(tilt);
    _pencilDir.set(
      sinTilt * Math.cos(pencilAzimuth),
      sinTilt * Math.sin(pencilAzimuth),
      cosTilt
    ).normalize();

    _pencilQuat.setFromUnitVectors(_pencilUp, _pencilDir);
    pencilSocketGroup.quaternion.copy(_pencilQuat);
    // Spin the hexagonal shaft around its longitudinal Y axis in lockstep with the 6-toothed hub
    pencilShaftGroup.rotation.y = reelAngle;
  }

  updatePencilTransform(0);

  // ===========================================================================
  // 3. MAGNETIC TAPE RIBBON MATERIAL (ANISOTROPIC FERRIC OXIDE + MYLAR BACKING)
  // ===========================================================================
  const tapeMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.11,
    metalness: 0.38,
  });
  {
    const meta = attribute('aRibbonMeta', 'vec4');
    const sNorm = meta.x;
    const vEdge = meta.y;
    const faceSide = meta.z;
    const creaseSeverity = meta.w;

    const nWorld = normalWorld;

    const grain = mineralToothFbm(
      vec3(sNorm.mul(140.0), vEdge.mul(5.0), float(1.7)),
      float(2.0)
    );

    // Rich dark espresso/chestnut ferric-oxide coating (Raw Umber + Burnt Sienna)
    const oxideSide = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(0.96).add(grain.mul(0.14)),
      burntSienna.rInf
    ).mul(0.46);
    const backingSide = kubelkaMunkGlaze(
      rawUmber.K,
      rawUmber.S,
      float(1.25).add(grain.mul(0.12)),
      hematiteRust.rInf
    ).mul(0.34);

    const isOxide = smoothstep(-0.2, 0.2, faceSide);
    const isEdge = float(1.0).sub(abs(faceSide));
    const baseTapeCol = mix(backingSide, oxideSide, isOxide);

    // Crisp studio softbox specular reflection bands across curved glossy Mylar loops
    const studioReflect = smoothstep(
      0.48,
      0.88,
      nWorld.z.mul(0.64).add(abs(nWorld.y).mul(0.46)).add(abs(nWorld.x).mul(0.26))
    );
    const sharpGlint = smoothstep(0.75, 0.94, studioReflect);
    const oxideSheenCol = mix(burntSienna.rInf, yellowOchre.rInf, float(0.54))
      .mul(studioReflect)
      .mul(0.52)
      .add(boneCalcite.rInf.mul(sharpGlint.mul(0.58)));

    const grazing = float(1.0).sub(abs(nWorld.z));
    const creaseRidge = clamp(
      creaseSeverity.mul(float(0.48).add(grazing.mul(0.58))),
      0.0,
      1.0
    );
    const creaseGlintCol = mix(yellowOchre.rInf, boneCalcite.rInf, float(0.64));
    const edgeRimCol = mix(burntSienna.rInf, warmKaolin.rInf, float(0.42)).mul(0.78);

    const finalTapeCol = mix(
      mix(baseTapeCol.add(oxideSheenCol), creaseGlintCol, creaseRidge.mul(0.52)),
      edgeRimCol,
      clamp(isEdge.mul(0.62), 0.0, 1.0)
    );

    tapeMat.colorNode = clamp(finalTapeCol, 0.0, 1.0);
    tapeMat.roughnessNode = clamp(
      float(0.09)
        .add(grain.mul(0.05))
        .add(creaseRidge.mul(0.07))
        .sub(isOxide.mul(0.02)),
      0.05,
      0.26
    );
  }

  const {
    geometry: tapeGeo,
    posAttr: tapePosAttr,
    normAttr: tapeNormAttr,
    metaAttr: tapeMetaAttr,
  } = createTapeRibbonGeometry(RENDER_RINGS);

  const tapeMesh = new THREE.Mesh(tapeGeo, tapeMat);
  tapeMesh.name = 'SpilledMagneticTapeRibbon';
  tapeMesh.castShadow = true;
  tapeMesh.receiveShadow = true;
  tapeMesh.frustumCulled = false;
  cassetteRig.add(tapeMesh);

  // ===========================================================================
  // 4. MULTI-LOOP BIRD'S-NEST MYLAR TANGLE & PENCIL-CRANK REWIND PHYSICS ENGINE
  // ===========================================================================
  const nodePos = new Float32Array(NODE_COUNT * 3);
  const nodePrev = new Float32Array(NODE_COUNT * 3);
  const restTarget = new Float32Array(NODE_COUNT * 3);
  const plasticDisp = new Float32Array(NODE_COUNT * 3);
  const scratchPlastic = new Float32Array(NODE_COUNT * 3);
  const nodeCrease = new Float32Array(NODE_COUNT);
  const restTwist = new Float32Array(NODE_COUNT);
  const nodeTwist = new Float32Array(NODE_COUNT);

  const smoothCenters = new Float32Array(RENDER_RINGS * 3);
  const smoothTangents = new Float32Array(RENDER_RINGS * 3);
  const subCenters = new Float32Array(RENDER_RINGS * 3);
  const subTangents = new Float32Array(RENDER_RINGS * 3);
  const subNormals = new Float32Array(RENDER_RINGS * 3);
  const subBinormals = new Float32Array(RENDER_RINGS * 3);
  const subCreases = new Float32Array(RENDER_RINGS);

  let unspoolRatio = 0.92;
  let knotSeverity = 0.90;
  let isRewinding = false;
  let isJammed = false;
  let clearReliefTimer = 0.0;
  let leftReelAngle = 0.0;
  let rightReelAngle = 0.0;
  let totalCreaseEnergy = 0.45;
  let rigTorqueRockZ = 0.0;
  let rigPullOffsetY = 0.0;

  let grabbedNodeIdx = -1;
  let isCrankingHub = false;
  let prevCrankAngle = 0.0;
  let prevCrankPtrX = 0.0;
  let prevCrankPtrY = 0.0;

  /**
   * Shortest 2D distance from `(px, py)` in local cassette coordinates to the
   * active 3D hexagonal pencil shaft projecting from `(RIGHT_SPOOL_X, REEL_Y)`.
   */
  function distToPencilSegment(px: number, py: number): number {
    const ax = RIGHT_SPOOL_X;
    const ay = REEL_Y;
    const bx = RIGHT_SPOOL_X + Math.cos(pencilAzimuth) * 0.72;
    const by = REEL_Y + Math.sin(pencilAzimuth) * 0.72;
    const abx = bx - ax;
    const aby = by - ay;
    const apx = px - ax;
    const apy = py - ay;
    const t = Math.max(0.0, Math.min(1.0, (apx * abx + apy * aby) / Math.max(1e-6, abx * abx + aby * aby)));
    return Math.hypot(px - (ax + abx * t), py - (ay + aby * t));
  }

  /**
   * Push any free tape point `(x, y, z)` outside the cassette housing's 3D volume
   * (`x in [-0.945, +0.945], y in [0.132, 1.250]`) so the ribbon NEVER clips across
   * the front face (`z > -0.108`) of the cassette shell or paper label.
   * Upper-crown loops (`y > 0.88`) tuck cleanly behind the back shell (`z <= -0.108`)
   * or outside the left/top rim, while side/bottom loops route around `x <= -0.945`
   * or `y <= 0.132`.
   */
  function enforceCassetteShellExclusion(
    x: number,
    y: number,
    z: number,
    uNorm: number,
    out: THREE.Vector3
  ): void {
    out.set(x, y, z);

    // Near the left and right guide roller throats (u < 0.03 or u > 0.97), keep below throat exit
    if (uNorm < 0.032 || uNorm > 0.968) {
      const maxThroat = THROAT_EXIT_Y - 0.008 * Math.sin(uNorm * Math.PI);
      if (out.y > maxThroat) out.y = maxThroat;
      return;
    }

    const boxMinX = -0.955;
    const boxMaxX = 0.955;
    const boxMinY = 0.130;
    const boxMaxY = 1.242;

    if (
      out.x > boxMinX &&
      out.x < boxMaxX &&
      out.y > boxMinY &&
      out.y < boxMaxY
    ) {
      // Push outward strictly through top (y = boxMaxY), left flank (x = boxMinX),
      // or bottom edge (y = boxMinY)—NEVER onto the front face (z = +0.115) and
      // without discontinuous Z-step jumps!
      const dLeft = out.x - boxMinX;
      const dBottom = out.y - boxMinY;
      const dTop = boxMaxY - out.y;
      if (out.y > 0.92 && dTop <= dLeft) {
        out.y = boxMaxY;
      } else if (dLeft <= dBottom) {
        out.x = boxMinX;
      } else {
        out.y = boxMinY;
      }
    }
  }

  /**
   * Evaluates the 64-waypoint 3D bird's-nest Mylar tangle spline for `u in [0, 1]`
   * between Left Guide Roller (`u = 0`) and Right Guide Roller (`u = 1`).
   * As `slack` decreases from `1.0` to `0.0` (while winding with the pencil),
   * upper-left loops slide down along the left flank around the bottom-left corner
   * before pulling into the bottom cassette throat so the ribbon never cuts across
   * the front of the cassette housing.
   */
  function evaluateCarrierTangleCurve(
    u: number,
    slack: number,
    knot: number,
    out: THREE.Vector3
  ): void {
    const env = Math.sin(u * Math.PI);
    const tautX = LEFT_GUIDE_X + (RIGHT_GUIDE_X - LEFT_GUIDE_X) * u;
    const tautY = THROAT_EXIT_Y - 0.004 * env;
    const tautZ = 0.006 * env;

    if (slack <= 1e-4) {
      out.set(tautX, tautY, tautZ);
      return;
    }

    const s = Math.max(0.0, Math.min(1.0, slack));
    const k = Math.max(0.0, Math.min(1.0, knot));
    const envWindow = Math.pow(env, 0.50);

    // Evaluate Catmull-Rom spline across TANGLE_WAYPOINTS
    const nWp = TANGLE_WAYPOINTS.length;
    const fWp = u * (nWp - 1);
    const idx1 = Math.max(0, Math.min(nWp - 2, Math.floor(fWp)));
    const idx0 = Math.max(0, idx1 - 1);
    const idx2 = Math.min(nWp - 1, idx1 + 1);
    const idx3 = Math.min(nWp - 1, idx1 + 2);
    const tRaw = fWp - idx1;

    // Gentle Mylar fold turn shaping without collapsing loop arches
    const kinkStrength =
      (TANGLE_WAYPOINTS[idx1][3] * (1.0 - tRaw) + TANGLE_WAYPOINTS[idx2][3] * tRaw) * k;
    const tSmooth = tRaw * tRaw * (3.0 - 2.0 * tRaw);
    const t = tRaw + (tSmooth - tRaw) * (0.18 * kinkStrength);
    const t2 = t * t;
    const t3 = t2 * t;

    const p0 = TANGLE_WAYPOINTS[idx0];
    const p1 = TANGLE_WAYPOINTS[idx1];
    const p2 = TANGLE_WAYPOINTS[idx2];
    const p3 = TANGLE_WAYPOINTS[idx3];

    let wpX =
      0.5 *
      (2 * p1[0] +
        (-p0[0] + p2[0]) * t +
        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
    let wpY =
      0.5 *
      (2 * p1[1] +
        (-p0[1] + p2[1]) * t +
        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
    const wpZ =
      0.5 *
      (2 * p1[2] +
        (-p0[2] + p2[2]) * t +
        (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 +
        (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3);

    // Corner-wrapping retraction during rewind:
    // Upper-left flank loops (wpY > 0.05) slide downward along the left flank (x <= -0.98)
    // toward the bottom-left corner (-0.98, -0.10) as s decreases, so they never cut
    // diagonally across the cassette face!
    if (wpY > 0.05) {
      const cornerGuideWeight = 1.0 - smoothstepScalar(0.22, 0.85, s);
      const targetCornerX = Math.min(wpX, -0.98);
      wpX = wpX + (targetCornerX - wpX) * cornerGuideWeight;
      wpY = wpY + (-0.10 - wpY) * cornerGuideWeight * 0.85;
    }

    const sFull = Math.min(1.0, s / 0.90);
    const kFull = Math.min(1.0, k / 0.88);
    const loopActivation = Math.pow(sFull, 0.82 + 0.42 * u);
    const knotBlend = 0.38 + 0.62 * kFull;
    const nestWeight = loopActivation * knotBlend;

    // Simple drooped catenary under the bottom head bay when slack is small
    const simpleDroopX = tautX - 0.24 * s * Math.sin(u * Math.PI * 2.0);
    const simpleDroopY = THROAT_EXIT_Y - 0.62 * s * Math.pow(env, 0.75);
    const simpleDroopZ = 0.06 * s * Math.sin(u * Math.PI * 2.0 - 0.3);

    const baseX = tautX + (simpleDroopX - tautX) * s;
    const baseY = tautY + (simpleDroopY - tautY) * s;
    const baseZ = tautZ + (simpleDroopZ - tautZ) * s;

    const fIdx = FREE_START + u * FREE_COUNT;
    const i0 = Math.max(FREE_START, Math.min(FREE_END - 1, Math.floor(fIdx)));
    const frac = fIdx - i0;
    const b0 = i0 * 3;
    const b1 = (i0 + 1) * 3;
    const px = plasticDisp[b0] * (1.0 - frac) + plasticDisp[b1] * frac;
    const py = plasticDisp[b0 + 1] * (1.0 - frac) + plasticDisp[b1 + 1] * frac;
    const pz = plasticDisp[b0 + 2] * (1.0 - frac) + plasticDisp[b1 + 2] * frac;

    const rawX = baseX + (wpX - baseX) * nestWeight + px * envWindow * s;
    const rawY = baseY + (wpY - baseY) * nestWeight + py * envWindow * s;
    const rawZ = baseZ + (wpZ - baseZ) * nestWeight + pz * envWindow * s;

    enforceCassetteShellExclusion(rawX, rawY, rawZ, u, out);
  }

  /**
   * Arc-length reparameterizes `evaluateCarrierTangleCurve` across `FREE_START..FREE_END`
   * and routes internal cassette strands (`0..FREE_START` and `FREE_END..NODE_COUNT-1`)
   * around the internal window guide pins so the tape visibly feeds off the reels.
   */
  function rebuildArcLengthConsistentTargets(
    slack: number,
    knot: number,
    leftPackScale: number,
    rightPackScale: number
  ): number {
    // 1. Internal Left Cassette Path: Left Spool Bottom Tangent -> Guide Pin -> Left Brass Roller
    const leftRadius = 0.31 * leftPackScale;
    const leftStartX = LEFT_SPOOL_X - leftRadius * 0.68;
    const leftStartY = REEL_Y - leftRadius * 0.65;
    for (let i = 0; i <= FREE_START; i++) {
      const f = i / FREE_START;
      const i3 = i * 3;
      const pinDeflect = Math.sin(f * Math.PI);
      restTarget[i3] = leftStartX + (LEFT_GUIDE_X - leftStartX) * f - 0.035 * pinDeflect;
      restTarget[i3 + 1] = leftStartY + (THROAT_EXIT_Y - leftStartY) * f;
      restTarget[i3 + 2] = 0.0;
    }

    // 2. Internal Right Cassette Path: Right Brass Roller -> Guide Pin -> Right Spool Bottom Tangent
    const rightRadius = 0.31 * rightPackScale;
    const rightEndX = RIGHT_SPOOL_X + rightRadius * 0.68;
    const rightEndY = REEL_Y - rightRadius * 0.65;
    const rightSpan = NODE_COUNT - 1 - FREE_END;
    for (let i = FREE_END; i < NODE_COUNT; i++) {
      const f = (i - FREE_END) / rightSpan;
      const i3 = i * 3;
      const pinDeflect = Math.sin(f * Math.PI);
      restTarget[i3] = RIGHT_GUIDE_X + (rightEndX - RIGHT_GUIDE_X) * f + 0.035 * pinDeflect;
      restTarget[i3 + 1] = THROAT_EXIT_Y + (rightEndY - THROAT_EXIT_Y) * f;
      restTarget[i3 + 2] = 0.0;
    }

    // 3. Dense sampling + cumulative arc length of the spilled loop between FREE_START and FREE_END
    let cumLen = 0.0;
    for (let s = 0; s < ARC_SAMPLES; s++) {
      const u = s / (ARC_SAMPLES - 1);
      evaluateCarrierTangleCurve(u, slack, knot, _samplePt);
      _arcX[s] = _samplePt.x;
      _arcY[s] = _samplePt.y;
      _arcZ[s] = _samplePt.z;
      if (s > 0) {
        cumLen += Math.hypot(
          _arcX[s] - _arcX[s - 1],
          _arcY[s] - _arcY[s - 1],
          _arcZ[s] - _arcZ[s - 1]
        );
      }
      _arcCumLen[s] = cumLen;
    }

    const totalLen = Math.max(0.01, cumLen);
    const uniformSegLen = totalLen / FREE_COUNT;

    let searchIdx = 0;
    for (let i = FREE_START + 1; i < FREE_END; i++) {
      const targetS = (i - FREE_START) * uniformSegLen;
      while (searchIdx < ARC_SAMPLES - 2 && _arcCumLen[searchIdx + 1] < targetS) {
        searchIdx++;
      }
      const s0 = _arcCumLen[searchIdx];
      const s1 = _arcCumLen[searchIdx + 1];
      const t = s1 > s0 ? (targetS - s0) / (s1 - s0) : 0.0;
      const i3 = i * 3;
      restTarget[i3] = _arcX[searchIdx] + (_arcX[searchIdx + 1] - _arcX[searchIdx]) * t;
      restTarget[i3 + 1] = _arcY[searchIdx] + (_arcY[searchIdx + 1] - _arcY[searchIdx]) * t;
      restTarget[i3 + 2] = _arcZ[searchIdx] + (_arcZ[searchIdx + 1] - _arcZ[searchIdx]) * t;
    }

    return uniformSegLen;
  }

  function seedTangledTapeLoop(initialSlack = 0.92, initialKnot = 0.90): void {
    unspoolRatio = initialSlack;
    knotSeverity = initialKnot;
    isJammed = false;
    clearReliefTimer = 0.0;

    const nWp = TANGLE_WAYPOINTS.length;
    for (let i = 0; i < NODE_COUNT; i++) {
      const i3 = i * 3;
      plasticDisp[i3] = 0;
      plasticDisp[i3 + 1] = 0;
      plasticDisp[i3 + 2] = 0;

      if (i <= FREE_START || i >= FREE_END) {
        nodeCrease[i] = 0;
        restTwist[i] = 0;
        nodeTwist[i] = 0;
      } else {
        const u = (i - FREE_START) / FREE_COUNT;
        const env = Math.sin(u * Math.PI);
        const fWp = u * (nWp - 1);
        const wIdx = Math.min(nWp - 1, Math.round(fWp));
        const wpKink = TANGLE_WAYPOINTS[wIdx][3];

        // Capstan pinch creases + loop turn creases
        const capstanBite = Math.exp(-Math.pow((u - 0.78) / 0.06, 2.0));
        const flankPinch = Math.exp(-Math.pow((u - 0.36) / 0.06, 2.0));
        nodeCrease[i] = Math.min(
          1.0,
          env * (0.50 * wpKink + 0.55 * capstanBite + 0.45 * flankPinch)
        );

        // Persistent multi-loop torsional ribbon twist so flat Mylar faces flip & glint at loop turns
        const tw =
          env *
          (1.42 * Math.sin(u * Math.PI * 15.0 + 0.35) +
            0.92 * Math.cos(u * Math.PI * 23.0 - 0.50));
        restTwist[i] = tw;
        nodeTwist[i] = tw;
      }
    }

    const leftPackScale = 0.80 + (1.0 - unspoolRatio) * 0.16;
    const rightPackScale = 0.56 + (1.0 - unspoolRatio) * 0.42;
    rebuildArcLengthConsistentTargets(unspoolRatio, knotSeverity, leftPackScale, rightPackScale);

    for (let i = 0; i < NODE_COUNT * 3; i++) {
      nodePos[i] = restTarget[i];
      nodePrev[i] = restTarget[i];
    }
  }

  seedTangledTapeLoop(0.92, 0.90);

  function applyTapePointerInteraction(
    ptrPos: THREE.Vector3,
    ptrVel: THREE.Vector3,
    ptrSpeed: number,
    isPointerDown: boolean,
    dt: number
  ): void {
    // Subtle tactile hover response on the pencil when the cursor nears the right hub / pencil shaft
    const dPencilSeg = distToPencilSegment(ptrPos.x, ptrPos.y);
    const targetHoverTilt = dPencilSeg < 0.48 ? -0.065 * Math.exp(-dPencilSeg * dPencilSeg * 9.0) : 0.0;
    pencilHoverTilt += (targetHoverTilt - pencilHoverTilt) * Math.min(1.0, dt * 10.0);

    if (isCrankingHub) return;

    if (isPointerDown) {
      if (grabbedNodeIdx < 0) {
        let bestDistSq = 0.52 * 0.52;
        let bestIdx = -1;
        for (let i = FREE_START + 2; i <= FREE_END - 2; i++) {
          const i3 = i * 3;
          const dx = nodePos[i3] - ptrPos.x;
          const dy = nodePos[i3 + 1] - ptrPos.y;
          const dz = (nodePos[i3 + 2] - ptrPos.z) * 0.35;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < bestDistSq) {
            bestDistSq = d2;
            bestIdx = i;
          }
        }
        if (bestIdx >= 0) {
          grabbedNodeIdx = bestIdx;
          isRewinding = false;
        }
      }

      if (grabbedNodeIdx >= FREE_START + 2 && grabbedNodeIdx <= FREE_END - 2) {
        const g3 = grabbedNodeIdx * 3;
        const clampedPtrX = Math.max(-1.85, Math.min(1.25, ptrPos.x));
        const clampedPtrY = Math.max(-1.15, Math.min(1.42, ptrPos.y));

        const moveDx = Math.max(-0.09, Math.min(0.09, ptrVel.x * dt));
        const moveDy = Math.max(-0.09, Math.min(0.09, ptrVel.y * dt));

        const toCursorX = (clampedPtrX - nodePos[g3]) * Math.min(1.0, ptrSpeed * 0.35) * dt * 8.0;
        const toCursorY = (clampedPtrY - nodePos[g3 + 1]) * Math.min(1.0, ptrSpeed * 0.35) * dt * 8.0;

        const totalDx = moveDx * 1.25 + toCursorX;
        const totalDy = moveDy * 1.25 + toCursorY;

        const span = 34;
        for (
          let i = Math.max(FREE_START + 1, grabbedNodeIdx - span);
          i <= Math.min(FREE_END - 1, grabbedNodeIdx + span);
          i++
        ) {
          const distNodes = (i - grabbedNodeIdx) / span;
          const w = Math.exp(-distNodes * distNodes * 2.5);
          const i3 = i * 3;
          nodePos[i3] += totalDx * w;
          nodePos[i3 + 1] += totalDy * w;
          plasticDisp[i3] = Math.max(
            -0.55,
            Math.min(0.55, plasticDisp[i3] + totalDx * w * 0.85)
          );
          plasticDisp[i3 + 1] = Math.max(
            -0.55,
            Math.min(0.45, plasticDisp[i3 + 1] + totalDy * w * 0.85)
          );
        }

        rigTorqueRockZ += (ptrVel.x * 0.015 - rigTorqueRockZ) * Math.min(1.0, dt * 10.0);
        rigPullOffsetY += (Math.min(0.0, ptrVel.y * 0.015) - rigPullOffsetY) * Math.min(1.0, dt * 10.0);

        // Pulling/dragging the tape unspools more slack and stirs the tangle
        const dragPull = Math.min(0.06, Math.hypot(ptrVel.x, ptrVel.y) * dt * 0.45);
        if (dragPull > 0.001) {
          const deltaSlack = Math.min(1.0 - unspoolRatio, dragPull * 0.55);
          unspoolRatio += deltaSlack;
          knotSeverity = Math.min(1.0, knotSeverity + dragPull * 0.45);
          leftReelAngle -= deltaSlack * 22.0;
          rightReelAngle += deltaSlack * 14.0;
        }
        return;
      }
    } else {
      grabbedNodeIdx = -1;
    }

    if (ptrSpeed < 0.02) return;
    const brushRadiusSq = 0.42 * 0.42;
    let brushedAny = false;

    for (let i = FREE_START + 2; i <= FREE_END - 2; i++) {
      const i3 = i * 3;
      const dx = nodePos[i3] - ptrPos.x;
      const dy = nodePos[i3 + 1] - ptrPos.y;
      const dz = (nodePos[i3 + 2] - ptrPos.z) * 0.45;
      const d2 = dx * dx + dy * dy + dz * dz;

      if (d2 < brushRadiusSq) {
        brushedAny = true;
        const w = Math.exp(-d2 * 12.0);
        const pushScale = Math.min(1.3, ptrSpeed * 0.52) * w * dt * 3.8;
        nodePos[i3] += ptrVel.x * pushScale;
        nodePos[i3 + 1] += ptrVel.y * pushScale;
        nodePos[i3 + 2] += (dx * ptrVel.y - dy * ptrVel.x) * 0.35 * pushScale;

        if (ptrSpeed > 0.35) {
          plasticDisp[i3] = Math.max(-0.48, Math.min(0.48, plasticDisp[i3] + ptrVel.x * w * dt * 0.35));
          plasticDisp[i3 + 1] = Math.max(-0.48, Math.min(0.38, plasticDisp[i3 + 1] + ptrVel.y * w * dt * 0.30));
          nodeCrease[i] = Math.min(1.0, nodeCrease[i] + w * ptrSpeed * dt * 0.55);
          knotSeverity = Math.min(1.0, knotSeverity + w * ptrSpeed * dt * 0.08);
        }
      }
    }

    if (brushedAny && ptrSpeed > 0.55) {
      const extraSlack = Math.min(0.18, ptrSpeed * 0.04) * dt;
      unspoolRatio = Math.min(1.0, unspoolRatio + extraSlack);
      leftReelAngle -= extraSlack * 18.0;
    }
  }

  function stepTapePhysics(dt: number, elapsedSec: number): void {
    // Gentle spatial smoothing on user-induced plasticDisp only when actively dragging/rewinding
    if (grabbedNodeIdx >= 0 || isRewinding || isCrankingHub) {
      for (let i = FREE_START + 1; i < FREE_END; i++) {
        const p3 = (i - 1) * 3;
        const c3 = i * 3;
        const n3 = (i + 1) * 3;
        scratchPlastic[c3] = 0.12 * plasticDisp[p3] + 0.76 * plasticDisp[c3] + 0.12 * plasticDisp[n3];
        scratchPlastic[c3 + 1] = 0.12 * plasticDisp[p3 + 1] + 0.76 * plasticDisp[c3 + 1] + 0.12 * plasticDisp[n3 + 1];
        scratchPlastic[c3 + 2] = 0.12 * plasticDisp[p3 + 2] + 0.76 * plasticDisp[c3 + 2] + 0.12 * plasticDisp[n3 + 2];
      }
      for (let i = FREE_START + 1; i < FREE_END; i++) {
        const c3 = i * 3;
        plasticDisp[c3] = scratchPlastic[c3];
        plasticDisp[c3 + 1] = scratchPlastic[c3 + 1];
        plasticDisp[c3 + 2] = scratchPlastic[c3 + 2];
      }
    }

    if (clearReliefTimer > 0.0) {
      clearReliefTimer -= dt;
      knotSeverity = Math.max(0.0, knotSeverity - 2.6 * dt);
      for (let i = FREE_START + 1; i < FREE_END; i++) {
        const i3 = i * 3;
        plasticDisp[i3] *= Math.exp(-dt * 6.0);
        plasticDisp[i3 + 1] *= Math.exp(-dt * 6.0);
        plasticDisp[i3 + 2] *= Math.exp(-dt * 6.0);
      }
      if (clearReliefTimer <= 0.0) {
        isJammed = false;
        isRewinding = true;
      }
    } else if (isRewinding) {
      // Active pencil-crank rewind cycle: the pencil orbits in a conical crank motion
      // around the right drive hub while reeling the tangled tape steadily back in!
      if (unspoolRatio > 0.004) {
        isJammed = false;
        const rewindRate = 0.38 + (1.0 - unspoolRatio) * 0.24;
        const prevRatio = unspoolRatio;
        unspoolRatio = Math.max(0.0, unspoolRatio - rewindRate * dt);
        const woundDelta = prevRatio - unspoolRatio;

        knotSeverity = Math.max(0.0, knotSeverity - woundDelta * 0.95);
        rightReelAngle -= woundDelta * 28.0;
        leftReelAngle -= woundDelta * 16.0;
        pencilAzimuth -= woundDelta * 28.0;
        rigTorqueRockZ = Math.sin(pencilAzimuth) * 0.018;

        const advectBlend = Math.min(0.35, woundDelta * 8.5);
        for (let i = FREE_END - 1; i > FREE_START; i--) {
          const i3 = i * 3;
          const next3 = (i + 1) * 3;
          nodePos[i3] += (nodePos[next3] - nodePos[i3]) * advectBlend;
          nodePos[i3 + 1] += (nodePos[next3 + 1] - nodePos[i3 + 1]) * advectBlend;
          nodePos[i3 + 2] += (nodePos[next3 + 2] - nodePos[i3 + 2]) * advectBlend;
        }

        const ironOut = Math.exp(-dt * (1.6 + (1.0 - unspoolRatio) * 3.4));
        for (let i = FREE_START + 1; i < FREE_END; i++) {
          nodeCrease[i] *= ironOut;
          plasticDisp[i * 3] *= ironOut;
          plasticDisp[i * 3 + 1] *= ironOut;
          plasticDisp[i * 3 + 2] *= ironOut;
        }

        if (unspoolRatio <= 0.004) {
          unspoolRatio = 0.0;
          knotSeverity = 0.0;
          isRewinding = false;
          isJammed = false;
        }
      } else {
        unspoolRatio = 0.0;
        knotSeverity = 0.0;
        isRewinding = false;
        isJammed = false;
      }
    } else if (!isCrankingHub) {
      // Gently relax pencil azimuth back toward the iconic upper-right resting angle when idle
      const angleDiff =
        ((PENCIL_REST_AZIMUTH - pencilAzimuth + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      pencilAzimuth += angleDiff * Math.min(1.0, dt * 3.2);
    }

    rigTorqueRockZ *= Math.exp(-dt * 5.5);
    rigPullOffsetY *= Math.exp(-dt * 5.5);

    // Anchor nodeTwist to restTwist scaled by slack so ribbon faces keep their 3D twists indefinitely
    const twistScale = Math.min(1.0, unspoolRatio * 1.25);
    for (let i = FREE_START + 1; i < FREE_END; i++) {
      const targetTw = restTwist[i] * twistScale;
      nodeTwist[i] += (targetTw - nodeTwist[i]) * Math.min(1.0, dt * 8.0);
    }

    // Asymmetric tape pancake scales: right reel visibly fills up as the pencil winds the tape in!
    const leftPackScale = 0.80 + (1.0 - unspoolRatio) * 0.16;
    const rightPackScale = 0.56 + (1.0 - unspoolRatio) * 0.42;
    leftReel.packMesh.scale.set(leftPackScale, leftPackScale, 1.0);
    rightReel.packMesh.scale.set(rightPackScale, rightPackScale, 1.0);
    leftReel.group.rotation.z = leftReelAngle;
    rightReel.group.rotation.z = rightReelAngle;

    uLeftPackRadius.value = 0.33 * leftPackScale;
    uRightPackRadius.value = 0.33 * rightPackScale;
    uRewindGlow.value = isRewinding || isCrankingHub ? 1.0 : 0.0;

    updatePencilTransform(rightReelAngle);

    const segRestLen = rebuildArcLengthConsistentTargets(
      unspoolRatio,
      knotSeverity,
      leftPackScale,
      rightPackScale
    );

    const substeps = 4;
    const velDamping = isRewinding || isCrankingHub ? 0.78 : 0.80;
    const coilStiffness =
      grabbedNodeIdx >= 0 ? 0.06 : isRewinding || isCrankingHub ? 0.26 : 0.20;

    for (let step = 0; step < substeps; step++) {
      for (let i = FREE_START + 1; i < FREE_END; i++) {
        const i3 = i * 3;
        const vx = (nodePos[i3] - nodePrev[i3]) * velDamping;
        const vy = (nodePos[i3 + 1] - nodePrev[i3 + 1]) * velDamping;
        const vz = (nodePos[i3 + 2] - nodePrev[i3 + 2]) * velDamping;

        nodePrev[i3] = nodePos[i3];
        nodePrev[i3 + 1] = nodePos[i3 + 1];
        nodePrev[i3 + 2] = nodePos[i3 + 2];

        const airSwayX = Math.sin(elapsedSec * 1.1 + i * 0.04) * 0.00008 * unspoolRatio;
        const airSwayZ = Math.cos(elapsedSec * 0.85 + i * 0.05) * 0.00008 * unspoolRatio;

        nodePos[i3] += vx + (restTarget[i3] - nodePos[i3]) * coilStiffness + airSwayX;
        nodePos[i3 + 1] += vy + (restTarget[i3 + 1] - nodePos[i3 + 1]) * coilStiffness;
        nodePos[i3 + 2] += vz + (restTarget[i3 + 2] - nodePos[i3 + 2]) * coilStiffness + airSwayZ;
      }

      for (let i = 0; i <= FREE_START; i++) {
        const i3 = i * 3;
        nodePos[i3] = restTarget[i3];
        nodePos[i3 + 1] = restTarget[i3 + 1];
        nodePos[i3 + 2] = restTarget[i3 + 2];
        nodePrev[i3] = restTarget[i3];
        nodePrev[i3 + 1] = restTarget[i3 + 1];
        nodePrev[i3 + 2] = restTarget[i3 + 2];
      }
      for (let i = FREE_END; i < NODE_COUNT; i++) {
        const i3 = i * 3;
        nodePos[i3] = restTarget[i3];
        nodePos[i3 + 1] = restTarget[i3 + 1];
        nodePos[i3 + 2] = restTarget[i3 + 2];
        nodePrev[i3] = restTarget[i3];
        nodePrev[i3 + 1] = restTarget[i3 + 1];
        nodePrev[i3 + 2] = restTarget[i3 + 2];
      }

      for (let pass = 0; pass < 4; pass++) {
        for (let i = FREE_START; i < FREE_END; i++) {
          const a3 = i * 3;
          const b3 = (i + 1) * 3;
          const dx = nodePos[b3] - nodePos[a3];
          const dy = nodePos[b3 + 1] - nodePos[a3 + 1];
          const dz = nodePos[b3 + 2] - nodePos[a3 + 2];
          const dist = Math.hypot(dx, dy, dz) || 1e-6;
          const corr = (dist - segRestLen) / dist;

          const wA = i === FREE_START ? 0.0 : 0.5;
          const wB = i + 1 === FREE_END ? 0.0 : 0.5;
          const wSum = wA + wB || 1.0;

          nodePos[a3] += dx * corr * (wA / wSum);
          nodePos[a3 + 1] += dy * corr * (wA / wSum);
          nodePos[a3 + 2] += dz * corr * (wA / wSum);

          nodePos[b3] -= dx * corr * (wB / wSum);
          nodePos[b3 + 1] -= dy * corr * (wB / wSum);
          nodePos[b3 + 2] -= dz * corr * (wB / wSum);
        }
      }

      const minDist = SELF_COLLISION_DIST;
      const minDistSq = minDist * minDist;
      for (let i = FREE_START + 2; i < FREE_END - 24; i++) {
        const i3 = i * 3;
        const xi = nodePos[i3];
        const yi = nodePos[i3 + 1];
        const zi = nodePos[i3 + 2];

        for (let j = i + 22; j < FREE_END - 2; j++) {
          const j3 = j * 3;
          const dx = xi - nodePos[j3];
          const dy = yi - nodePos[j3 + 1];
          const dz = zi - nodePos[j3 + 2];
          const d2 = dx * dx + dy * dy + dz * dz;

          if (d2 < minDistSq && d2 > 1e-9) {
            const d = Math.sqrt(d2);
            const push = (0.16 * (minDist - d)) / d;
            const px = dx * push;
            const py = dy * push;
            const pz = dz * push;

            nodePos[i3] += px;
            nodePos[i3 + 1] += py;
            nodePos[i3 + 2] += pz;

            nodePos[j3] -= px;
            nodePos[j3 + 1] -= py;
            nodePos[j3 + 2] -= pz;
          }
        }
      }

      // Enforce 3D cassette shell exclusion & outer studio bounds
      for (let i = FREE_START + 1; i < FREE_END; i++) {
        const i3 = i * 3;
        const uNorm = (i - FREE_START) / FREE_COUNT;
        enforceCassetteShellExclusion(
          nodePos[i3],
          nodePos[i3 + 1],
          nodePos[i3 + 2],
          uNorm,
          _samplePt
        );
        nodePos[i3] = Math.max(-1.88, Math.min(1.25, _samplePt.x));
        nodePos[i3 + 1] = Math.max(-1.08, Math.min(1.64, _samplePt.y));
        nodePos[i3 + 2] = Math.max(-0.42, Math.min(0.45, _samplePt.z));
      }
    }

    let cSum = 0;
    for (let i = FREE_START + 1; i < FREE_END; i++) cSum += nodeCrease[i];
    totalCreaseEnergy = cSum / Math.max(1, FREE_COUNT - 1);
  }

  function updateTapeRibbonMesh(): void {
    for (let seg = 0; seg < NODE_COUNT - 1; seg++) {
      const i0 = Math.max(0, seg - 1) * 3;
      const i1 = seg * 3;
      const i2 = (seg + 1) * 3;
      const i3 = Math.min(NODE_COUNT - 1, seg + 2) * 3;

      _s0.set(nodePos[i0], nodePos[i0 + 1], nodePos[i0 + 2]);
      _s1.set(nodePos[i1], nodePos[i1 + 1], nodePos[i1 + 2]);
      _s2.set(nodePos[i2], nodePos[i2 + 1], nodePos[i2 + 2]);
      _s3.set(nodePos[i3], nodePos[i3 + 1], nodePos[i3 + 2]);

      const c1 = nodeCrease[seg];
      const c2 = nodeCrease[seg + 1];

      const subCount = seg === NODE_COUNT - 2 ? SUBDIV + 1 : SUBDIV;
      for (let s = 0; s < subCount; s++) {
        const t = s / SUBDIV;
        const t2 = t * t;
        const t3 = t2 * t;

        const crX =
          0.5 *
          (2 * _s1.x +
            (-_s0.x + _s2.x) * t +
            (2 * _s0.x - 5 * _s1.x + 4 * _s2.x - _s3.x) * t2 +
            (-_s0.x + 3 * _s1.x - 3 * _s2.x + _s3.x) * t3);
        const crY =
          0.5 *
          (2 * _s1.y +
            (-_s0.y + _s2.y) * t +
            (2 * _s0.y - 5 * _s1.y + 4 * _s2.y - _s3.y) * t2 +
            (-_s0.y + 3 * _s1.y - 3 * _s2.y + _s3.y) * t3);
        const crZ =
          0.5 *
          (2 * _s1.z +
            (-_s0.z + _s2.z) * t +
            (2 * _s0.z - 5 * _s1.z + 4 * _s2.z - _s3.z) * t2 +
            (-_s0.z + 3 * _s1.z - 3 * _s2.z + _s3.z) * t3);

        const rIdx = seg * SUBDIV + s;
        const r3 = rIdx * 3;
        smoothCenters[r3] = crX;
        smoothCenters[r3 + 1] = crY;
        smoothCenters[r3 + 2] = crZ;
        subCreases[rIdx] = c1 * (1 - t) + c2 * t;
      }
    }

    for (let r = 0; r < RENDER_RINGS; r++) {
      const rPrev = Math.max(0, r - 1) * 3;
      const rNext = Math.min(RENDER_RINGS - 1, r + 1) * 3;
      const tx = smoothCenters[rNext] - smoothCenters[rPrev];
      const ty = smoothCenters[rNext + 1] - smoothCenters[rPrev + 1];
      const tz = smoothCenters[rNext + 2] - smoothCenters[rPrev + 2];
      const tLen = Math.hypot(tx, ty, tz) || 1.0;

      const r3 = r * 3;
      smoothTangents[r3] = tx / tLen;
      smoothTangents[r3 + 1] = ty / tLen;
      smoothTangents[r3 + 2] = tz / tLen;
    }

    _prevSmoothTang.set(smoothTangents[0], smoothTangents[1], smoothTangents[2]);
    _bishopV.set(0, 0, 1).addScaledVector(_prevSmoothTang, -_prevSmoothTang.z);
    if (_bishopV.lengthSq() < 1e-5) _bishopV.set(1, 0, 0);
    _bishopV.normalize();

    const pleatSlackGate = Math.min(1.0, unspoolRatio * 1.8);

    for (let r = 0; r < RENDER_RINGS; r++) {
      const r3 = r * 3;
      _smoothTang.set(smoothTangents[r3], smoothTangents[r3 + 1], smoothTangents[r3 + 2]);

      const sumX = _prevSmoothTang.x + _smoothTang.x;
      const sumY = _prevSmoothTang.y + _smoothTang.y;
      const sumZ = _prevSmoothTang.z + _smoothTang.z;
      const sumLenSq = sumX * sumX + sumY * sumY + sumZ * sumZ;
      if (sumLenSq > 1e-6) {
        const dotVS = (_bishopV.x * sumX + _bishopV.y * sumY + _bishopV.z * sumZ) * (2.0 / sumLenSq);
        _bishopV.x -= dotVS * sumX;
        _bishopV.y -= dotVS * sumY;
        _bishopV.z -= dotVS * sumZ;
      }

      const sNorm = r / (RENDER_RINGS - 1);
      const rollerAnchorWeight = sNorm < 0.05 || sNorm > 0.95 ? 0.45 : 0.10;
      _bishopV.z += (1.0 - _bishopV.z) * rollerAnchorWeight;
      _bishopV.addScaledVector(_smoothTang, -_smoothTang.dot(_bishopV));
      if (_bishopV.lengthSq() < 1e-5) {
        _bishopV.set(0, 0, 1).addScaledVector(_smoothTang, -_smoothTang.z);
      }
      _bishopV.normalize();

      _bishopU.crossVectors(_bishopV, _smoothTang).normalize();

      const fNode = sNorm * (NODE_COUNT - 1);
      const nIdx = Math.min(NODE_COUNT - 2, Math.floor(fNode));
      const frac = fNode - nIdx;
      const rawTwist = nodeTwist[nIdx] * (1 - frac) + nodeTwist[nIdx + 1] * frac;

      const inFreeLoop =
        smoothstepScalar(0.03, 0.08, sNorm) *
        (1.0 - smoothstepScalar(0.92, 0.97, sNorm)) *
        Math.min(1.0, unspoolRatio * 2.5);
      const twist = (0.72 + rawTwist * 0.68) * inFreeLoop;

      const cosTw = Math.cos(twist);
      const sinTw = Math.sin(twist);

      _norm.copy(_bishopU).multiplyScalar(cosTw).addScaledVector(_bishopV, sinTw).normalize();
      _binorm.copy(_bishopV).multiplyScalar(cosTw).addScaledVector(_bishopU, -sinTw).normalize();

      subBinormals[r3] = _binorm.x;
      subBinormals[r3 + 1] = _binorm.y;
      subBinormals[r3 + 2] = _binorm.z;

      const crease = subCreases[r];
      const pleatPeriod = FOLD_STRIDE * SUBDIV;
      const phase = (r % (pleatPeriod * 2)) / pleatPeriod;
      const triWave = phase <= 1.0 ? phase * 2.0 - 1.0 : 3.0 - phase * 2.0;
      const pleatDisp = crease * 0.0020 * pleatSlackGate * triWave;

      subCenters[r3] = smoothCenters[r3] + _norm.x * pleatDisp;
      subCenters[r3 + 1] = smoothCenters[r3 + 1] + _norm.y * pleatDisp;
      subCenters[r3 + 2] = smoothCenters[r3 + 2] + _norm.z * pleatDisp;

      _prevSmoothTang.copy(_smoothTang);
    }

    for (let r = 0; r < RENDER_RINGS; r++) {
      const rPrev = Math.max(0, r - 1) * 3;
      const rNext = Math.min(RENDER_RINGS - 1, r + 1) * 3;
      const tx = subCenters[rNext] - subCenters[rPrev];
      const ty = subCenters[rNext + 1] - subCenters[rPrev + 1];
      const tz = subCenters[rNext + 2] - subCenters[rPrev + 2];
      const tLen = Math.hypot(tx, ty, tz) || 1.0;

      const r3 = r * 3;
      _tang.set(tx / tLen, ty / tLen, tz / tLen);
      _binorm.set(subBinormals[r3], subBinormals[r3 + 1], subBinormals[r3 + 2]);

      _binorm.addScaledVector(_tang, -_tang.dot(_binorm)).normalize();
      _norm.crossVectors(_binorm, _tang).normalize();

      subTangents[r3] = _tang.x;
      subTangents[r3 + 1] = _tang.y;
      subTangents[r3 + 2] = _tang.z;

      subNormals[r3] = _norm.x;
      subNormals[r3 + 1] = _norm.y;
      subNormals[r3 + 2] = _norm.z;

      subBinormals[r3] = _binorm.x;
      subBinormals[r3 + 1] = _binorm.y;
      subBinormals[r3 + 2] = _binorm.z;
    }

    const posArr = tapePosAttr.array as Float32Array;
    const normArr = tapeNormAttr.array as Float32Array;
    const metaArr = tapeMetaAttr.array as Float32Array;
    const halfH = TAPE_THICKNESS * 0.5;
    const halfW = TAPE_WIDTH * 0.5;

    for (let r = 0; r < RENDER_RINGS; r++) {
      const r3 = r * 3;
      const cx = subCenters[r3];
      const cy = subCenters[r3 + 1];
      const cz = subCenters[r3 + 2];

      const nx = subNormals[r3];
      const ny = subNormals[r3 + 1];
      const nz = subNormals[r3 + 2];

      const bx = subBinormals[r3];
      const by = subBinormals[r3 + 1];
      const bz = subBinormals[r3 + 2];

      const crease = subCreases[r];

      const tox = cx + nx * halfH + bx * halfW;
      const toy = cy + ny * halfH + by * halfW;
      const toz = cz + nz * halfH + bz * halfW;

      const box = cx + nx * halfH - bx * halfW;
      const boy = cy + ny * halfH - by * halfW;
      const boz = cz + nz * halfH - bz * halfW;

      const bix = cx - nx * halfH - bx * halfW;
      const biy = cy - ny * halfH - by * halfW;
      const biz = cz - nz * halfH - bz * halfW;

      const tix = cx - nx * halfH + bx * halfW;
      const tiy = cy - ny * halfH + by * halfW;
      const tiz = cz - nz * halfH + bz * halfW;

      const vBase = r * 8 * 3;
      posArr[vBase + 0] = tox; posArr[vBase + 1] = toy; posArr[vBase + 2] = toz;
      posArr[vBase + 3] = box; posArr[vBase + 4] = boy; posArr[vBase + 5] = boz;
      posArr[vBase + 6] = tox; posArr[vBase + 7] = toy; posArr[vBase + 8] = toz;
      posArr[vBase + 9] = tix; posArr[vBase + 10] = tiy; posArr[vBase + 11] = tiz;
      posArr[vBase + 12] = bix; posArr[vBase + 13] = biy; posArr[vBase + 14] = biz;
      posArr[vBase + 15] = tix; posArr[vBase + 16] = tiy; posArr[vBase + 17] = tiz;
      posArr[vBase + 18] = box; posArr[vBase + 19] = boy; posArr[vBase + 20] = boz;
      posArr[vBase + 21] = bix; posArr[vBase + 22] = biy; posArr[vBase + 23] = biz;

      normArr[vBase + 0] = nx; normArr[vBase + 1] = ny; normArr[vBase + 2] = nz;
      normArr[vBase + 3] = nx; normArr[vBase + 4] = ny; normArr[vBase + 5] = nz;

      normArr[vBase + 6] = bx; normArr[vBase + 7] = by; normArr[vBase + 8] = bz;
      normArr[vBase + 9] = bx; normArr[vBase + 10] = by; normArr[vBase + 11] = bz;

      normArr[vBase + 12] = -nx; normArr[vBase + 13] = -ny; normArr[vBase + 14] = -nz;
      normArr[vBase + 15] = -nx; normArr[vBase + 16] = -ny; normArr[vBase + 17] = -nz;

      normArr[vBase + 18] = -bx; normArr[vBase + 19] = -by; normArr[vBase + 20] = -bz;
      normArr[vBase + 21] = -bx; normArr[vBase + 22] = -by; normArr[vBase + 23] = -bz;

      const mBase = r * 8 * 4;
      for (let k = 0; k < 8; k++) {
        metaArr[mBase + k * 4 + 3] = crease;
      }
    }

    tapePosAttr.needsUpdate = true;
    tapeNormAttr.needsUpdate = true;
    tapeMetaAttr.needsUpdate = true;
  }

  function smoothstepScalar(edge0: number, edge1: number, x: number): number {
    const t = Math.max(0.0, Math.min(1.0, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0 - 2.0 * t);
  }

  // ===========================================================================
  // 5. INTERACTIVE GESTURE HANDLERS (PENCIL-CRANK REWIND & TAPE TANGLE DRAG)
  // ===========================================================================
  let downClientX = 0;
  let downClientY = 0;
  let dragDistancePx = 0;

  window.addEventListener(
    'pointerdown',
    (ev: PointerEvent) => {
      const target = ev.target as HTMLElement | null;
      if (target && target.closest('.day-header, #day-index-drawer')) return;
      downClientX = ev.clientX;
      downClientY = ev.clientY;
      dragDistancePx = 0;

      _invRigQuat.copy(cassetteRig.quaternion).invert();
      _localPtr.copy(stage.pointer.worldPos).sub(cassetteRig.position).applyQuaternion(_invRigQuat);

      // Grabbing the pencil shaft/eraser, right hub, or cassette body engages Pencil-Crank Winding;
      // clicking/dragging on the left-flank or bottom tape loops grabs the ribbon directly!
      const distToRightHub = Math.hypot(_localPtr.x - RIGHT_SPOOL_X, _localPtr.y - REEL_Y);
      const dPencilSeg = distToPencilSegment(_localPtr.x, _localPtr.y);
      if (
        distToRightHub < 0.44 ||
        dPencilSeg < 0.34 ||
        (_localPtr.y > 0.22 && _localPtr.y < 1.24 && _localPtr.x > -0.58 && _localPtr.x < 1.25)
      ) {
        isCrankingHub = true;
        prevCrankAngle = Math.atan2(_localPtr.y - REEL_Y, _localPtr.x - RIGHT_SPOOL_X);
        prevCrankPtrX = _localPtr.x;
        prevCrankPtrY = _localPtr.y;
      } else {
        isCrankingHub = false;
      }
    },
    { passive: true }
  );

  window.addEventListener(
    'pointermove',
    (ev: PointerEvent) => {
      if (ev.buttons > 0) {
        dragDistancePx = Math.hypot(
          ev.clientX - downClientX,
          ev.clientY - downClientY
        );

        if (isCrankingHub && dragDistancePx > 4) {
          isRewinding = false;
          _invRigQuat.copy(cassetteRig.quaternion).invert();
          _localPtr.copy(stage.pointer.worldPos).sub(cassetteRig.position).applyQuaternion(_invRigQuat);

          const curAngle = Math.atan2(_localPtr.y - REEL_Y, _localPtr.x - RIGHT_SPOOL_X);
          let dAngle = curAngle - prevCrankAngle;
          if (dAngle > Math.PI) dAngle -= Math.PI * 2;
          if (dAngle < -Math.PI) dAngle += Math.PI * 2;
          prevCrankAngle = curAngle;

          const moveStep = Math.hypot(
            _localPtr.x - prevCrankPtrX,
            _localPtr.y - prevCrankPtrY
          );
          prevCrankPtrX = _localPtr.x;
          prevCrankPtrY = _localPtr.y;

          // Update the pencil's 3D crank azimuth directly with the user's cursor sweep
          const crankSweep = Math.abs(dAngle) > 0.01 ? dAngle : -moveStep * 1.4;
          pencilAzimuth += crankSweep;

          // Cranking the pencil reels the tangled tape steadily back into the cassette
          const windDelta = Math.min(
            unspoolRatio,
            Math.abs(dAngle) * 0.11 + moveStep * 0.075
          );
          if (windDelta > 0.0001) {
            unspoolRatio = Math.max(0.0, unspoolRatio - windDelta);
            knotSeverity = Math.max(0.0, knotSeverity - windDelta * 0.95);
            rightReelAngle += crankSweep * 1.35;
            leftReelAngle += crankSweep * 0.82;
            rigTorqueRockZ = Math.sin(pencilAzimuth) * 0.016;
            isJammed = false;

            const ironOut = Math.exp(-windDelta * 2.8);
            for (let i = FREE_START + 1; i < FREE_END; i++) {
              nodeCrease[i] *= ironOut;
              plasticDisp[i * 3] *= ironOut;
              plasticDisp[i * 3 + 1] *= ironOut;
              plasticDisp[i * 3 + 2] *= ironOut;
            }
          }
        }
      }
    },
    { passive: true }
  );

  window.addEventListener(
    'pointerup',
    (ev: PointerEvent) => {
      const target = ev.target as HTMLElement | null;
      if (target && target.closest('.day-header, #day-index-drawer')) return;

      isCrankingHub = false;
      grabbedNodeIdx = -1;

      if (dragDistancePx < 10) {
        if (unspoolRatio <= 0.03) {
          isRewinding = false;
          seedTangledTapeLoop(0.92, 0.90);
        } else if (isJammed) {
          clearReliefTimer = 0.42;
        } else {
          isRewinding = !isRewinding;
        }
      }
    },
    { passive: true }
  );

  for (let w = 0; w < 18; w++) {
    stepTapePhysics(1 / 60, 0);
  }
  updateTapeRibbonMesh();

  if (window.__SPECIMEN_STAGE__) {
    const baseGetState = window.__SPECIMEN_STAGE__.getSceneState;
    window.__SPECIMEN_STAGE__.getSceneState = () => {
      const posArr = tapePosAttr.array as Float32Array;
      const normArr = tapeNormAttr.array as Float32Array;
      let finiteGeometry = true;
      for (let i = 0; i < posArr.length; i++) {
        if (!Number.isFinite(posArr[i]) || !Number.isFinite(normArr[i])) {
          finiteGeometry = false;
          break;
        }
      }
      let frontFaceClipCount = 0;
      for (let i = FREE_START + 2; i < FREE_END - 2; i++) {
        const i3 = i * 3;
        const nx = nodePos[i3];
        const ny = nodePos[i3 + 1];
        const nz = nodePos[i3 + 2];
        if (nx > -0.92 && nx < 0.92 && ny > 0.16 && ny < 1.23 && nz > -0.08) {
          frontFaceClipCount++;
        }
      }
      return {
        ...baseGetState(),
        specimen: 'TangledCassetteAndPencil',
        unspoolRatio: Number(unspoolRatio.toFixed(3)),
        knotSeverity: Number(knotSeverity.toFixed(3)),
        isRewinding,
        isJammed,
        pencilAzimuth: Number(pencilAzimuth.toFixed(3)),
        totalCreaseEnergy: Number(totalCreaseEnergy.toFixed(3)),
        leftReelAngle: Number(leftReelAngle.toFixed(3)),
        rightReelAngle: Number(rightReelAngle.toFixed(3)),
        finiteGeometry,
        frontFaceClipCount,
      };
    };
  }

  stage.onTick(({ dtSec, elapsedSec }) => {
    const dt = Math.min(0.033, dtSec);

    const ptrX = stage.pointer.worldPos.x;
    const ptrY = stage.pointer.worldPos.y;
    const targetRotY =
      0.11 + Math.sin(elapsedSec * 0.36) * 0.032 + Math.tanh(ptrX * 0.40) * 0.11;
    const targetRotX =
      -0.08 + Math.cos(elapsedSec * 0.28) * 0.024 - Math.tanh(ptrY * 0.40) * 0.08;
    const targetRotZ = 0.15 + rigTorqueRockZ;

    cassetteRig.rotation.y += (targetRotY - cassetteRig.rotation.y) * Math.min(1.0, dt * 4.5);
    cassetteRig.rotation.x += (targetRotX - cassetteRig.rotation.x) * Math.min(1.0, dt * 4.5);
    cassetteRig.rotation.z += (targetRotZ - cassetteRig.rotation.z) * Math.min(1.0, dt * 8.0);
    cassetteRig.position.y +=
      (-0.18 + rigPullOffsetY - cassetteRig.position.y) * Math.min(1.0, dt * 8.0);

    _invRigQuat.copy(cassetteRig.quaternion).invert();
    _localPtr.copy(stage.pointer.worldPos).sub(cassetteRig.position).applyQuaternion(_invRigQuat);
    _localVel.copy(stage.pointer.velocity).applyQuaternion(_invRigQuat);

    applyTapePointerInteraction(
      _localPtr,
      _localVel,
      stage.pointer.speed,
      stage.pointer.isDown,
      dt
    );

    stepTapePhysics(dt, elapsedSec);
    updateTapeRibbonMesh();
  });
}

bootDay04().catch((err) => {
  console.error('Failed to boot Day 04 (Tangle):', err);
});
