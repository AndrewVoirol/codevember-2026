import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  atan,
  clamp,
  cos,
  dot,
  exp,
  float,
  floor,
  fract,
  length,
  max,
  min,
  mix,
  normalLocal,
  normalize,
  positionLocal,
  pow,
  reflect,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';
import { MINERAL_PALETTE } from '../../core/palette';
import { mineralToothFbm } from '../../core/tslMaterials';
import {
  APERTURE_PLANE_OFFSET,
  ARCH_HALF_WIDTH,
  ARCH_RADIUS,
  ARCH_SILL_Y,
  ARCH_SPRING_Y,
  PORTAL_FACES,
  PORTAL_WORLD_SCALE,
  type PortalFaceId,
} from './portalWorlds';
import { type PortalMaterialsContract } from './geometry';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface StudioEnvironmentHandle {
  texture: THREE.Texture;
  dispose: () => void;
}

export interface PortalMaterials extends PortalMaterialsContract {
  matteBasalt: THREE.MeshStandardNodeMaterial;
  basaltAccent: THREE.MeshStandardNodeMaterial;
  satinBrass: THREE.MeshStandardNodeMaterial;
  darkBrassFastener: THREE.MeshStandardNodeMaterial;
  portalMaterials: Record<PortalFaceId, THREE.MeshBasicNodeMaterial>;
  uCamLocal: { value: THREE.Vector3 };
  uTime: { value: number };
  uHoverWeights: Record<PortalFaceId, { value: number }>;
  /** Per-face hovered aperture local UV in [-1, 1] x [-1, 1] + impulse in .z */
  uPointerLocal: Record<PortalFaceId, { value: THREE.Vector3 }>;
  dispose: () => void;
}

/**
 * Generates a radial-feathered softbox luminance map so PMREM specular reflections roll
 * cleanly across chamfered basalt ashlar blocks, brass arch casings, and compass inlays.
 */
function createFeatheredSoftboxTexture(r: number, g: number, b: number): THREE.DataTexture {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ny = ((y + 0.5) / size) * 2 - 1;
    for (let x = 0; x < size; x++) {
      const nx = ((x + 0.5) / size) * 2 - 1;
      const d = Math.hypot(nx, ny);
      const falloff = Math.max(0, Math.min(1, 1 - d * d));
      const soft = falloff * falloff;
      const idx = (y * size + x) * 4;
      data[idx] = Math.min(255, Math.round(r * soft * 255));
      data[idx + 1] = Math.min(255, Math.round(g * soft * 255));
      data[idx + 2] = Math.min(255, Math.round(b * soft * 255));
      data[idx + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Builds a studio softbox PMREM environment cubemap tuned for honed volcanic basalt
 * and brushed satin brass against the canonical #212529 Chalky Slate gallery space.
 */
export function createPortalStudioEnvironment(
  renderer: THREE.WebGPURenderer
): StudioEnvironmentHandle {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0c1015);

  const disposables: Array<{ dispose: () => void }> = [];

  const addSoftbox = (
    w: number,
    h: number,
    pos: [number, number, number],
    r: number,
    g: number,
    b: number,
    gain: number
  ) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const tex = createFeatheredSoftboxTexture(r, g, b);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      color: new THREE.Color().setRGB(gain, gain, gain),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.lookAt(0, 0, 0);
    envScene.add(mesh);
    disposables.push(geo, tex, mat);
  };

  // Overhead-left warm calcite key softbox
  addSoftbox(11.5, 9.0, [-4.8, 5.8, 4.4], 1.0, 0.96, 0.89, 1.85);
  // Right tall architectural studio scrim (highlights vertical brass arch bezels & corner piers)
  addSoftbox(8.5, 11.5, [6.2, 2.4, 3.2], 0.99, 0.95, 0.9, 1.72);
  // Rear-top cool mineral rim softbox (separates dark honed basalt cleanly off #212529)
  addSoftbox(13.0, 5.6, [-1.2, 5.4, -5.8], 0.9, 0.95, 1.0, 1.58);
  // Overhead diffused zenith scrim (catches top brass compass-rose inlay)
  addSoftbox(9.5, 9.5, [0.4, 7.2, 0.4], 0.97, 0.95, 0.91, 1.08);
  // Opposite cool-slate flank fill (ensures South & West brass brackets gleam across 360° orbits)
  addSoftbox(9.0, 7.5, [-5.6, 1.4, -4.8], 0.86, 0.93, 0.99, 1.24);
  addSoftbox(8.5, 7.0, [5.0, 1.2, -5.4], 0.95, 0.92, 0.88, 1.15);
  // Subtle warm gallery bounce from below (illuminates underside compass plate & plinth bevels)
  addSoftbox(12.0, 12.0, [0.0, -6.4, 0.0], 0.96, 0.92, 0.86, 0.42);

  const renderTarget = pmrem.fromScene(envScene, 0.072);
  for (const item of disposables) item.dispose();
  pmrem.dispose();

  return {
    texture: renderTarget.texture,
    dispose: () => {
      renderTarget.dispose();
    },
  };
}

/**
 * Scale-normalized 4-tap tetrahedral 3D FBM micro-normal perturbation in object space.
 * Produces honed volcanic stone grain or fine brushed brass micro-relief without macro waviness.
 */
const perturbNormal3DFbm = Fn(
  ([baseNorm_in, pos_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(baseNorm_in));
    const p = vec3(pos_in);
    const sc = float(scale_in);
    const eps = float(0.16).div(max(float(1.0), sc));

    const k1 = vec3(1.0, -1.0, -1.0);
    const k2 = vec3(-1.0, -1.0, 1.0);
    const k3 = vec3(-1.0, 1.0, -1.0);
    const k4 = vec3(1.0, 1.0, 1.0);

    const f1 = mineralToothFbm(p.add(k1.mul(eps)), sc);
    const f2 = mineralToothFbm(p.add(k2.mul(eps)), sc);
    const f3 = mineralToothFbm(p.add(k3.mul(eps)), sc);
    const f4 = mineralToothFbm(p.add(k4.mul(eps)), sc);

    const grad = k1.mul(f1).add(k2.mul(f2)).add(k3.mul(f3)).add(k4.mul(f4)).mul(1.5625);
    const tanGrad = grad.sub(n.mul(dot(grad, n)));
    return transformNormalToView(normalize(n.sub(tanGrad.mul(float(strength_in)))));
  }
);

// ---------------------------------------------------------------------------
// Shared Pure Analytical TSL SDF Helpers (Zero Textures / Zero Derivatives)
// ---------------------------------------------------------------------------

const sdBox3Tsl = Fn(([p_in, b_in]: [TslNode, TslNode]) => {
  const q = abs(vec3(p_in)).sub(vec3(b_in));
  return length(max(q, vec3(0.0))).add(min(max(q.x, max(q.y, q.z)), float(0.0)));
});

const sminTsl = Fn(([a_in, b_in, k_in]: [TslNode, TslNode, TslNode]) => {
  const a = float(a_in);
  const b = float(b_in);
  const k = float(k_in);
  const h = max(k.sub(abs(a.sub(b))), float(0.0)).div(k);
  return min(a, b).sub(h.mul(h).mul(k).mul(0.25));
});

/** Safe atan2 avoiding atan(0, 0) singularity on Apple Silicon / Dawn fast-math. */
const safeAtan2 = Fn(([y_in, x_in]: [TslNode, TslNode]) => {
  const y = float(y_in);
  const x = float(x_in);
  const safeX = x.add(x.lessThan(0.0).select(float(-1e-5), float(1e-5)));
  return atan(y, safeX);
});

/**
 * WGSL-safe descending smoothstep: returns 1 when x <= lo, 0 when x >= hi.
 * Strictly enforces lo < hi because WGSL smoothstep(low, high, x) rejects low >= high.
 */
function invSmoothstep(lo: number, hi: number, x: TslNode): TslNode {
  return float(1.0).sub(smoothstep(float(lo), float(hi), x));
}

/** Subtle architectural reveal shadow right at the perimeter lip of the arched window aperture. */
const computeArchApertureEdgeAo = Fn(([u_in, v_in]: [TslNode, TslNode]) => {
  const u = float(u_in);
  const v = float(v_in);
  const dSide = float(ARCH_HALF_WIDTH).sub(abs(u));
  const dSill = v.sub(float(ARCH_SILL_Y));
  const dv = max(float(0.0), v.sub(float(ARCH_SPRING_Y)));
  const dCrown = float(ARCH_RADIUS).sub(length(vec2(u, dv)));
  const dEdge = min(dSide, min(dSill, dCrown));
  return float(0.84).add(smoothstep(float(0.0), float(0.036), dEdge).mul(0.16));
});

/** Unified mineral-gamut filmic highlight shoulder keeping HDR shafts & lanterns natural. */
const toneMapMineral = Fn(([raw_in]: [TslNode]) => {
  const raw = max(vec3(0.0), vec3(raw_in));
  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const lum = dot(raw, vec3(0.2126, 0.7152, 0.0722));
  const compressed = raw.div(vec3(1.0).add(raw.mul(0.38)));
  const warmRoll = smoothstep(float(1.75), float(5.4), lum).mul(0.24);
  return mix(compressed, cCalcite.mul(0.97), warmRoll);
});

/**
 * Projects cube-local fragment position & camera into the chosen cardinal portal's
 * canonical world coordinates (`x = right`, `y = up`, `z = inward depth >= 0`).
 */
function buildPortalLocalRayNodes(faceId: PortalFaceId, uCamLocal: TslNode) {
  const spec = PORTAL_FACES[faceId];
  const fRight = vec3(spec.right[0], spec.right[1], spec.right[2]);
  const fUp = vec3(spec.up[0], spec.up[1], spec.up[2]);
  const fNormal = vec3(spec.normal[0], spec.normal[1], spec.normal[2]);
  const fInward = vec3(spec.inward[0], spec.inward[1], spec.inward[2]);

  const pFrag = vec3(positionLocal);
  const rdCube = normalize(pFrag.sub(uCamLocal));

  const uAper = dot(pFrag, fRight);
  const vAper = dot(pFrag, fUp);
  const wAper = float(APERTURE_PLANE_OFFSET).sub(dot(pFrag, fNormal));

  const ro = vec3(uAper, vAper, wAper).mul(float(PORTAL_WORLD_SCALE));
  const rd = normalize(vec3(dot(rdCube, fRight), dot(rdCube, fUp), dot(rdCube, fInward)));
  const edgeAo = computeArchApertureEdgeAo(uAper, vAper);

  return { ro, rd, uAper, vAper, edgeAo };
}

// ---------------------------------------------------------------------------
// World 0 — North (`+Z`): Sunlit Travertine Colonnade
// ---------------------------------------------------------------------------

const northColonnadeSdf = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const dFloor = p.y.add(1.76);

  const colSpacing = float(2.7);
  const colIdx = floor(p.z.sub(1.85).div(colSpacing).add(0.5));
  const colZ = clamp(colIdx.mul(colSpacing).add(1.85), float(1.85), float(20.75));
  const lx = abs(p.x).sub(1.76);
  const lz = p.z.sub(colZ);

  const phi = safeAtan2(lz, lx);
  const flute = abs(sin(phi.mul(6.0))).mul(0.015);
  const rad = length(vec2(lx, lz));
  const dShaftCyl = max(rad.sub(float(0.285).sub(flute)), abs(p.y.sub(0.04)).sub(1.54));
  const dPlinth = sdBox3Tsl(vec3(lx, p.y.add(1.56), lz), vec3(0.37, 0.2, 0.37)).sub(0.015);
  const dCapital = sdBox3Tsl(vec3(lx, p.y.sub(1.72), lz), vec3(0.4, 0.14, 0.4)).sub(0.015);
  const dColumn = min(dShaftCyl, min(dPlinth, dCapital));

  const dBeamSide = sdBox3Tsl(
    vec3(abs(p.x).sub(1.76), p.y.sub(2.16), p.z.sub(11.4)),
    vec3(0.36, 0.3, 11.6)
  );
  const dCrossBeam = sdBox3Tsl(vec3(p.x, p.y.sub(2.22), lz), vec3(1.86, 0.24, 0.26));
  const dLeftWall = p.x.add(4.5);

  const dBackSlab = sdBox3Tsl(vec3(p.x, p.y.sub(0.42), p.z.sub(22.8)), vec3(4.0, 2.9, 0.48));
  const dArchHole = length(vec2(p.x, max(float(0.0), p.y.sub(0.12)))).sub(1.28);
  const dBackArch = max(dBackSlab, dArchHole.negate());

  return min(dFloor, min(dColumn, min(dBeamSide, min(dCrossBeam, min(dLeftWall, dBackArch)))));
});

const northColonnadeNormal = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0035);
  const k1 = vec3(1.0, -1.0, -1.0);
  const k2 = vec3(-1.0, -1.0, 1.0);
  const k3 = vec3(-1.0, 1.0, -1.0);
  const k4 = vec3(1.0, 1.0, 1.0);
  return normalize(
    k1
      .mul(northColonnadeSdf(p.add(k1.mul(e))))
      .add(k2.mul(northColonnadeSdf(p.add(k2.mul(e)))))
      .add(k3.mul(northColonnadeSdf(p.add(k3.mul(e)))))
      .add(k4.mul(northColonnadeSdf(p.add(k4.mul(e)))))
  );
});

// ---------------------------------------------------------------------------
// World 1 — East (`+X`): Plunging Lapis Spiral Stairwell
// ---------------------------------------------------------------------------

const eastStairwellSdf = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const shaftCz = float(2.4);
  const shaftR = float(2.3);
  const qx = p.x;
  const qz = p.z.sub(shaftCz);
  const r = length(vec2(qx, qz));

  const dCylWall = shaftR.sub(r);
  const dAlcoveCut = sdBox3Tsl(vec3(p.x, p.y.sub(0.02), p.z.sub(0.28)), vec3(1.56, 1.84, 0.72));
  const dWall = max(dCylWall, dAlcoveCut.negate());

  const dLanding = sdBox3Tsl(vec3(p.x, p.y.add(1.98), p.z.sub(0.08)), vec3(1.62, 0.22, 0.36));

  const hTurn = float(4.8);
  const twoPi = float(Math.PI * 2);
  const nSteps = float(16.0);
  const dTheta = twoPi.div(nSteps);

  const theta = safeAtan2(qz, qx);
  const thetaPhase = theta.add(float(Math.PI * 0.5));
  const yPhase = p.y.add(1.72).add(thetaPhase.div(twoPi).mul(hTurn));
  const kWrap = floor(yPhase.div(hTurn).add(0.5));
  const baseIdx = floor(thetaPhase.div(dTheta));
  const rClamped = max(float(1.1), r);

  // Evaluate nearest 3 angular stair sectors for exact Lipschitz-1 step edges
  const evalStep = (offsetVal: number) => {
    const idx = baseIdx.add(float(offsetVal));
    const stepTheta = idx.add(0.5).mul(dTheta);
    const dAngRaw = thetaPhase.sub(stepTheta);
    const dAng = dAngRaw.sub(floor(dAngRaw.div(twoPi).add(0.5)).mul(twoPi));

    const stepY = float(-1.72).sub(idx.add(0.5).div(nSteps).mul(hTurn)).add(kWrap.mul(hTurn));
    const yDiffRaw = p.y.sub(stepY);
    const yDiff = yDiffRaw.sub(floor(yDiffRaw.div(hTurn).add(0.5)).mul(hTurn));

    const dRad = abs(r.sub(1.7)).sub(0.58);
    const dVert = abs(yDiff).sub(0.11);
    const dArc = abs(dAng.mul(rClamped)).sub(dTheta.mul(rClamped).mul(0.54).add(0.03));
    const q = vec3(dRad, dVert, dArc);
    return length(max(q, vec3(0.0)))
      .add(min(max(q.x, max(q.y, q.z)), float(0.0)))
      .sub(0.015);
  };

  const dStairs = min(evalStep(0), min(evalStep(-1), evalStep(1)));

  const yRailRaw = yPhase.sub(kWrap.mul(hTurn)).sub(0.78);
  const yRail = yRailRaw.sub(floor(yRailRaw.div(hTurn).add(0.5)).mul(hTurn));
  const dRail = length(vec2(r.sub(1.16), yRail)).sub(0.034);

  const dAbyssFloor = p.y.add(21.0);

  return min(dWall, min(dLanding, min(dStairs, min(dRail, dAbyssFloor))));
});

const eastStairwellNormal = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0038);
  const k1 = vec3(1.0, -1.0, -1.0);
  const k2 = vec3(-1.0, -1.0, 1.0);
  const k3 = vec3(-1.0, 1.0, -1.0);
  const k4 = vec3(1.0, 1.0, 1.0);
  return normalize(
    k1
      .mul(eastStairwellSdf(p.add(k1.mul(e))))
      .add(k2.mul(eastStairwellSdf(p.add(k2.mul(e)))))
      .add(k3.mul(eastStairwellSdf(p.add(k3.mul(e)))))
      .add(k4.mul(eastStairwellSdf(p.add(k4.mul(e)))))
  );
});

// ---------------------------------------------------------------------------
// World 2 — South (`-Z`): Amber Lantern Over Dark Still Water
// ---------------------------------------------------------------------------

/** Solid vaulted stone cistern + brass lantern cage SDF (water plane traced analytically). */
const southCisternSolidSdf = Fn(([p_in, lanternPos_in]: [TslNode, TslNode]) => {
  const p = vec3(p_in);
  const lPos = vec3(lanternPos_in);

  const dSideWalls = float(2.6).sub(abs(p.x));
  const dVaultCeiling = float(2.6).sub(length(vec2(p.x, max(float(0.0), p.y.sub(0.45)))));
  const dChamber = min(dSideWalls, dVaultCeiling);

  const baySpacing = float(3.2);
  const bayIdx = floor(p.z.sub(1.85).div(baySpacing).add(0.5));
  const bayZ = clamp(bayIdx.mul(baySpacing).add(1.85), float(1.85), float(21.05));
  const lz = p.z.sub(bayZ);

  const dPier = sdBox3Tsl(vec3(abs(p.x).sub(1.94), p.y.add(0.55), lz), vec3(0.28, 1.65, 0.28)).sub(
    0.02
  );
  const archR = length(vec2(p.x, p.y.sub(0.45)));
  const dRib = max(abs(archR.sub(2.4)).sub(0.16), abs(lz).sub(0.22));
  const dRibUpper = max(dRib, float(0.4).sub(p.y));

  const dBackWall = float(23.0).sub(p.z);

  // Foreground swaying Realgar-Amber brass lantern + suspension chain
  const qL = p.sub(lPos);
  const rLantern = length(vec2(qL.x, qL.z));
  const dLanternGlass = max(rLantern.sub(0.21), abs(qL.y).sub(0.28)).sub(0.02);
  const dLanternCap = max(rLantern.sub(0.27), abs(qL.y.sub(0.32)).sub(0.06)).sub(0.015);
  const dLanternBase = max(rLantern.sub(0.25), abs(qL.y.add(0.32)).sub(0.045)).sub(0.015);
  const dLantern = min(dLanternGlass, min(dLanternCap, dLanternBase));
  const dChain = max(rLantern.sub(0.018), float(0.34).sub(qL.y));

  return min(dChamber, min(dPier, min(dRibUpper, min(dBackWall, min(dLantern, dChain)))));
});

const southCisternNormal = Fn(([p_in, lanternPos_in]: [TslNode, TslNode]) => {
  const p = vec3(p_in);
  const lPos = vec3(lanternPos_in);
  const e = float(0.004);
  const k1 = vec3(1.0, -1.0, -1.0);
  const k2 = vec3(-1.0, -1.0, 1.0);
  const k3 = vec3(-1.0, 1.0, -1.0);
  const k4 = vec3(1.0, 1.0, 1.0);
  return normalize(
    k1
      .mul(southCisternSolidSdf(p.add(k1.mul(e)), lPos))
      .add(k2.mul(southCisternSolidSdf(p.add(k2.mul(e)), lPos)))
      .add(k3.mul(southCisternSolidSdf(p.add(k3.mul(e)), lPos)))
      .add(k4.mul(southCisternSolidSdf(p.add(k4.mul(e)), lPos)))
  );
});

const shadeSouthCisternSolid = Fn(
  ([pHit_in, rayD_in, distVal_in, lPos_in, farPos_in, pulse_in]: [
    TslNode,
    TslNode,
    TslNode,
    TslNode,
    TslNode,
    TslNode,
  ]) => {
    const pHit = vec3(pHit_in);
    const rayD = vec3(rayD_in);
    const distVal = float(distVal_in);
    const lPos = vec3(lPos_in);
    const fPos = vec3(farPos_in);
    const pulse = float(pulse_in);

    const cBoneCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
    const cYellowOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
    const cRawUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
    const cRealgarAmber = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
    const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
    const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
    const cSporeSulfur = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);

    const n = southCisternNormal(pHit, lPos);
    const grain = mineralToothFbm(pHit, float(2.6));

    // Detect whether hit is on the swaying Realgar-Amber lantern glass / brass cage
    const qL = pHit.sub(lPos);
    const rL = length(vec2(qL.x, qL.z));
    const isLanternGlass = invSmoothstep(0.215, 0.24, rL).mul(
      invSmoothstep(0.25, 0.29, abs(qL.y))
    );
    const isLanternMetal = invSmoothstep(0.26, 0.32, rL)
      .mul(invSmoothstep(0.36, 0.42, abs(qL.y)))
      .mul(float(1.0).sub(isLanternGlass));

    // Damp masonry stonework: Verdigris-Malachite copper-stained mortar & Raw-Umber ashlar blocks
    const dampBand = invSmoothstep(-1.72, -0.2, pHit.y);
    const stoneBase = mix(
      cRawUmber.mul(0.54),
      cVerdigris.mul(0.44),
      dampBand.mul(0.62).add(grain.mul(0.38))
    );
    const mossHighlight = mix(
      stoneBase,
      cCeladon.mul(0.42),
      smoothstep(float(0.62), float(0.88), grain).mul(0.42)
    );
    const albedo = mix(mossHighlight, cYellowOchre.mul(0.75), isLanternMetal);

    // Foreground swaying lantern point-light illumination (rich warm Realgar-Amber + Yellow-Ochre)
    const toL1 = lPos.sub(pHit);
    const d1Sq = max(float(0.18), dot(toL1, toL1));
    const lDir1 = normalize(toL1);
    const atten1 = float(1.0).div(float(0.42).add(d1Sq.mul(0.18)));
    const nDotL1 = max(float(0.0), dot(n, lDir1));
    const h1 = normalize(lDir1.sub(rayD));
    const spec1 = pow(max(float(0.0), dot(n, h1)), 22.0).mul(float(0.18).add(dampBand.mul(0.26)));

    // Distant background cistern lantern illumination (reveals deep vaulted bays!)
    const toL2 = fPos.sub(pHit);
    const d2Sq = max(float(0.4), dot(toL2, toL2));
    const lDir2 = normalize(toL2);
    const atten2 = float(1.0).div(float(0.8).add(d2Sq.mul(0.22)));
    const nDotL2 = max(float(0.0), dot(n, lDir2));

    const ao = clamp(southCisternSolidSdf(pHit.add(n.mul(0.24)), lPos).div(0.24), 0.25, 1.0);
    const warmLightCol = mix(cRealgarAmber, cYellowOchre, float(0.36)).mul(2.55).mul(pulse);
    const farLightCol = mix(cRealgarAmber, cYellowOchre, float(0.28)).mul(1.75);
    const dampAmbient = mix(cVerdigris.mul(0.085), cRawUmber.mul(0.075), float(0.45)).mul(ao);

    const litStone = albedo
      .mul(
        warmLightCol
          .mul(nDotL1.mul(atten1))
          .add(farLightCol.mul(nDotL2.mul(atten2)))
          .add(dampAmbient)
      )
      .add(cRealgarAmber.mul(spec1.mul(atten1).mul(1.4)));

    // Glowing Realgar-Amber + Spore-Sulfur inner lantern glass with dark vertical brass cage ribs
    const cagePhi = safeAtan2(qL.z, qL.x);
    const ribMask = smoothstep(float(0.86), float(0.96), abs(cos(cagePhi.mul(2.0))));
    const coreGlow = mix(
      cRealgarAmber.mul(2.6),
      mix(cSporeSulfur.mul(3.4), cBoneCalcite.mul(3.8), invSmoothstep(0.0, 0.18, abs(qL.y))),
      float(0.58)
    ).mul(pulse);
    const lanternSurface = mix(coreGlow, cRawUmber.mul(0.22), ribMask.mul(0.82));

    const combined = mix(litStone, lanternSurface, isLanternGlass);

    // Deep cistern atmospheric mist (hushed Verdigris-Umber darkness)
    const fogT = float(1.0).sub(exp(distVal.mul(-0.072)));
    const fogCol = mix(cRawUmber.mul(0.065), cVerdigris.mul(0.095), float(0.52));
    return mix(combined, fogCol, clamp(fogT, 0.0, 0.92));
  }
);

// ---------------------------------------------------------------------------
// World 3 — West (`-X`): Warm Ochre Desert Slot-Canyon Arch
// ---------------------------------------------------------------------------

const westCanyonSdf = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const depthRamp = smoothstep(float(0.0), float(2.2), p.z);

  const floorH = float(-1.78).add(
    depthRamp.mul(sin(p.z.mul(0.42)).mul(0.14).add(cos(p.x.mul(0.95)).mul(0.07)))
  );
  const dFloor = p.y.sub(floorH);

  const centerLineX = depthRamp.mul(sin(p.z.mul(0.23).sub(0.25)).mul(0.52));
  const relX = p.x.sub(centerLineX);
  const strata = sin(p.y.mul(5.0).add(sin(p.z.mul(0.48)).mul(0.55))).mul(0.075);
  const billow = depthRamp.mul(
    cos(p.z.mul(0.29))
      .mul(0.42)
      .add(sin(p.y.mul(0.68).add(p.z.mul(0.19))).mul(0.42))
      .add(cos(p.y.mul(1.32).sub(p.z.mul(0.26))).mul(0.18))
  );
  const mouthOpen = exp(max(float(0.0), p.z).mul(-0.55)).mul(0.72);
  const skyOpen = max(float(0.0), p.y.sub(1.65)).mul(0.55);
  const canyonHalfW = float(1.72).add(mouthOpen).add(billow).add(strata).add(skyOpen);
  const dWalls = max(canyonHalfW.sub(abs(relX)), p.y.sub(4.4));

  // Monumental natural sandstone rock arch spanning the canyon at z = 6.3
  const az = p.z.sub(6.3);
  const archSpanX = p.x.sub(0.06);
  const archSpanY = p.y.sub(1.35);
  const bridgeCurveY = archSpanY
    .add(archSpanX.mul(archSpanX).mul(0.27))
    .sub(sin(archSpanX.mul(2.1)).mul(0.07));
  const bridgeThickY = float(0.44).add(cos(archSpanX.mul(1.7)).mul(0.06));
  const dBridge = sdBox3Tsl(
    vec3(archSpanX, bridgeCurveY, az),
    vec3(2.95, bridgeThickY, 0.64)
  ).sub(0.16);
  const archVoid = length(vec2(archSpanX.div(1.5), p.y.add(0.35).div(1.58))).sub(1.0);
  const dArch = max(dBridge, archVoid.mul(-1.2));

  // Distant sunlit sandstone mesa cliff beyond the natural arch
  const mesaRidge = float(21.8)
    .add(sin(p.x.mul(0.32)).mul(1.8))
    .sub(max(float(0.0), p.y.sub(1.2)).mul(0.45));
  const dMesa = max(mesaRidge.sub(p.z), p.y.sub(4.8));

  const dRock = sminTsl(dWalls, dArch, float(0.45));
  return min(dFloor, min(dRock, dMesa));
});

const westCanyonNormal = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0045);
  const k1 = vec3(1.0, -1.0, -1.0);
  const k2 = vec3(-1.0, -1.0, 1.0);
  const k3 = vec3(-1.0, 1.0, -1.0);
  const k4 = vec3(1.0, 1.0, 1.0);
  return normalize(
    k1
      .mul(westCanyonSdf(p.add(k1.mul(e))))
      .add(k2.mul(westCanyonSdf(p.add(k2.mul(e)))))
      .add(k3.mul(westCanyonSdf(p.add(k3.mul(e)))))
      .add(k4.mul(westCanyonSdf(p.add(k4.mul(e)))))
  );
});

// ---------------------------------------------------------------------------
// Public Material Factory
// ---------------------------------------------------------------------------

export function createPortalMaterials(): PortalMaterials {
  const uCamLocal = uniform(new THREE.Vector3(2.45, 1.12, 2.72));
  const uTime = uniform(0);

  const uHoverWeights = {
    north: uniform(0),
    east: uniform(0),
    south: uniform(0),
    west: uniform(0),
  };

  const uPointerLocal = {
    north: uniform(new THREE.Vector3(0, 0, 0)),
    east: uniform(new THREE.Vector3(0, 0, 0)),
    south: uniform(new THREE.Vector3(0, 0, 0)),
    west: uniform(new THREE.Vector3(0, 0, 0)),
  };

  // Canonical Mineral Pigment Linear RGB Constants (strictly zero digital neons)
  const cBoneCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const cWarmKaolin = vec3(...MINERAL_PALETTE.warmKaolin.rInfLinear);
  const cYellowOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
  const cRawUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
  const cBurntSienna = vec3(...MINERAL_PALETTE.burntSienna.rInfLinear);
  const cHematiteRust = vec3(...MINERAL_PALETTE.hematiteRust.rInfLinear);
  const cRealgarAmber = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
  const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const cAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const cLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);
  const cSporeSulfur = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Honed Matte Volcanic Basalt Exterior Masonry
  // ---------------------------------------------------------------------------
  const matteBasalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.08,
    roughness: 0.68,
    envMapIntensity: 1.05,
  });
  {
    const mineralGrain = mineralToothFbm(positionLocal, float(46.0));
    const macroVeining = mineralToothFbm(positionLocal.add(vec3(3.7, 1.9, 5.2)), float(14.0));
    const microFleck = mineralToothFbm(positionLocal.add(vec3(9.1, 4.3, 2.8)), float(115.0));
    const basaltDeep = vec3(0.032, 0.036, 0.042);
    const basaltHoned = vec3(0.062, 0.067, 0.075);
    const baseStone = mix(
      basaltDeep,
      basaltHoned,
      smoothstep(float(0.28), float(0.76), mineralGrain.mul(0.62).add(macroVeining.mul(0.38)))
    );
    const feldsparGlint = smoothstep(float(0.78), float(0.95), microFleck).mul(0.016);
    matteBasalt.colorNode = baseStone.add(cWarmKaolin.mul(feldsparGlint));
    matteBasalt.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(56.0), float(0.026));
    matteBasalt.roughnessNode = clamp(
      float(0.66).add(mineralGrain.sub(0.5).mul(0.14)).sub(feldsparGlint.mul(4.0)),
      0.52,
      0.82
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Darker Honed Basalt Accent Reveals (Inner Jambs & Cornice Shadow Undercuts)
  // ---------------------------------------------------------------------------
  const basaltAccent = new THREE.MeshStandardNodeMaterial({
    metalness: 0.12,
    roughness: 0.56,
    envMapIntensity: 1.12,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(64.0));
    basaltAccent.colorNode = mix(
      vec3(0.02, 0.024, 0.029),
      vec3(0.042, 0.047, 0.055),
      smoothstep(float(0.25), float(0.78), grain)
    );
    basaltAccent.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(72.0), float(0.018));
    basaltAccent.roughnessNode = clamp(float(0.54).add(grain.sub(0.5).mul(0.12)), 0.44, 0.68);
  }

  // ---------------------------------------------------------------------------
  // 3. Machined Brushed Satin Brass (Arched Bezels, Corner Brackets, Keystones, Compass Rose)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
    envMapIntensity: 1.32,
  });
  {
    const fineBrush = mineralToothFbm(positionLocal, float(92.0));
    const microSatin = mineralToothFbm(positionLocal.add(vec3(4.3, 2.1, 6.7)), float(150.0));
    const brassAged = vec3(0.56, 0.43, 0.20);
    const brassBright = vec3(0.76, 0.61, 0.30);
    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.26), float(0.78), fineBrush.mul(0.6).add(microSatin.mul(0.4)))
    );
    satinBrass.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(115.0), float(0.004));
    satinBrass.roughnessNode = clamp(
      float(0.21).add(fineBrush.sub(0.5).mul(0.06)).add(microSatin.sub(0.5).mul(0.03)),
      0.16,
      0.30
    );
  }

  // ---------------------------------------------------------------------------
  // 4. Patinated Dark-Brass Fasteners & Compass Pin
  // ---------------------------------------------------------------------------
  const darkBrassFastener = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.32,
    envMapIntensity: 1.18,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(74.0));
    darkBrassFastener.colorNode = mix(
      vec3(0.24, 0.18, 0.09),
      vec3(0.42, 0.32, 0.15),
      smoothstep(float(0.25), float(0.75), grain)
    );
    darkBrassFastener.roughnessNode = clamp(float(0.32).add(grain.sub(0.5).mul(0.08)), 0.24, 0.42);
  }

  // ===========================================================================
  // PORTAL WORLD 0 — NORTH (+Z): SUNLIT TRAVERTINE COLONNADE
  // ===========================================================================
  const portalNorth = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalNorth.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('north', uCamLocal);
    const hoverW = uHoverWeights.north;
    const ptr = uPointerLocal.north;

    // Warm Mediterranean sun slanting diagonally right-to-left across the fluted colonnade
    const sunShiftX = ptr.x.mul(hoverW).mul(0.22).add(sin(uTime.mul(0.25)).mul(0.03));
    const sunShiftY = ptr.y.mul(hoverW).mul(0.12);
    const sunDir = normalize(vec3(float(0.64).add(sunShiftX), float(0.66).add(sunShiftY), float(-0.34)));

    const t = float(0.02).toVar();
    const hit = float(0.0).toVar();
    const maxDist = float(28.0);

    Loop(58, () => {
      const p = ro.add(rd.mul(t));
      const d = northColonnadeSdf(p);
      If(d.lessThan(0.0045), () => {
        hit.assign(1.0);
        Break();
      });
      t.addAssign(max(float(0.0045), d.mul(0.84)));
      If(t.greaterThan(maxDist), () => {
        Break();
      });
    });

    // Warm luminous sunlit Mediterranean sky visible between open right bays & skylight slits
    const skyElev = clamp(rd.y.mul(0.65).add(0.42), 0.0, 1.0);
    const sunAlign = max(float(0.0), dot(rd, sunDir));
    const skyCol = mix(cYellowOchre.mul(0.95), cBoneCalcite.mul(1.22), skyElev)
      .add(cBoneCalcite.mul(pow(sunAlign, 10.0).mul(0.55)));

    const finalCol = skyCol.toVar();

    If(hit.greaterThan(0.5), () => {
      const pHit = ro.add(rd.mul(t));
      const n = northColonnadeNormal(pHit);

      // Single-point procedural mineral grain on hit point (zero FBM inside raymarch loop)
      const grain = mineralToothFbm(pHit, float(2.8));
      const finePore = mineralToothFbm(pHit.add(vec3(4.1, 7.3, 2.2)), float(11.5));

      // Checkered inlaid travertine floor vs fluted Bone-Calcite classical columns & architrave
      const isFloor = invSmoothstep(-1.755, -1.7, pHit.y);
      const tileU = floor(pHit.x.mul(0.78).add(1024.0));
      const tileV = floor(pHit.z.mul(0.78).add(1024.0));
      const checker = fract(tileU.add(tileV).mul(0.5).add(1024.0)).mul(2.0);
      const fU = abs(fract(pHit.x.mul(0.78).add(1024.0)).sub(0.5));
      const fV = abs(fract(pHit.z.mul(0.78).add(1024.0)).sub(0.5));
      const grout = smoothstep(float(0.46), float(0.495), max(fU, fV));

      const lightTile = mix(cBoneCalcite.mul(0.96), cWarmKaolin.mul(0.92), grain.mul(0.6));
      const ochreTile = mix(cYellowOchre.mul(0.78), cWarmKaolin.mul(0.72), grain.mul(0.55));
      const floorAlbedo = mix(
        mix(lightTile, ochreTile, smoothstep(float(0.25), float(0.75), checker)),
        cRawUmber.mul(0.42),
        grout.mul(0.85)
      );

      // Fluted travertine stone columns & entablature with warm sedimentary travertine banding
      const band = sin(pHit.y.mul(8.5).add(grain.mul(2.6))).mul(0.5).add(0.5);
      const stoneAlbedo = mix(
        cWarmKaolin.mul(0.84),
        cBoneCalcite.mul(1.02),
        band.mul(0.55).add(finePore.mul(0.45))
      );
      const wallTint = mix(
        stoneAlbedo,
        mix(cYellowOchre.mul(0.74), cWarmKaolin.mul(0.82), float(0.55)),
        invSmoothstep(-4.48, -4.2, pHit.x)
      );
      const albedo = mix(wallTint, floorAlbedo, isFloor);

      // 14-step crisp geometric soft shadow ray toward warm angled sunlight
      const sh = float(1.0).toVar();
      const ts = float(0.06).toVar();
      Loop(14, () => {
        const hs = northColonnadeSdf(pHit.add(n.mul(0.015)).add(sunDir.mul(ts)));
        sh.assign(min(sh, clamp(hs.mul(7.5).div(ts), float(0.0), float(1.0))));
        ts.addAssign(clamp(hs, float(0.08), float(0.85)));
        If(sh.lessThan(0.015).or(ts.greaterThan(11.0)), () => {
          Break();
        });
      });

      // Cheap 2-tap analytical ambient occlusion
      const ao = clamp(
        northColonnadeSdf(pHit.add(n.mul(0.18)))
          .div(0.18)
          .mul(0.55)
          .add(northColonnadeSdf(pHit.add(n.mul(0.52))).div(0.52).mul(0.45)),
        0.22,
        1.0
      );

      const nDotL = max(float(0.0), dot(n, sunDir));
      const sunCol = mix(cBoneCalcite, cYellowOchre, float(0.26)).mul(2.15);
      // Warm bounced travertine GI + subtle cool sky skylight fill inside shadow troughs
      const warmBounce = mix(cYellowOchre, cWarmKaolin, float(0.45))
        .mul(float(0.24).add(max(float(0.0), n.y.negate()).mul(0.14)));
      const coolSkyFill = mix(cWarmKaolin, cAzurite.mul(0.85), float(0.32))
        .mul(max(float(0.0), n.y.mul(0.5).add(0.5)).mul(0.22));

      // Subtle honed travertine specular glint on floor tiles
      const halfV = normalize(sunDir.sub(rd));
      const spec = pow(max(float(0.0), dot(n, halfV)), 28.0).mul(isFloor.mul(0.28).add(0.06)).mul(sh);

      const lit = albedo
        .mul(sunCol.mul(nDotL.mul(sh)).add(warmBounce.add(coolSkyFill).mul(ao)))
        .add(cBoneCalcite.mul(spec));

      // Warm sunlit atmospheric distance fog (> 16 m perspective depth)
      const fogT = float(1.0).sub(exp(t.mul(-0.048)));
      const fogCol = mix(cWarmKaolin.mul(0.88), cYellowOchre.mul(0.95), float(0.34));
      finalCol.assign(mix(lit, fogCol, clamp(fogT, 0.0, 0.88)));
    });

    // Lightweight 5-tap golden volumetric sunlight shafts slanting between columns
    const shaftAccum = float(0.0).toVar();
    const maxShaftT = min(t, float(15.5));
    for (let s = 0; s < 5; s++) {
      const fracS = (s + 0.45) / 5.0;
      const pBeam = ro.add(rd.mul(maxShaftT.mul(fracS)));
      // Analytical sun-ray alignment through inter-column gaps & skylight bays
      const zProj = pBeam.z.sub(pBeam.x.mul(sunDir.z.div(max(float(0.15), sunDir.x))));
      const bayWave = sin(zProj.sub(0.5).mul(float((Math.PI * 2) / 2.7))).mul(0.5).add(0.5);
      const inGallery = invSmoothstep(0.4, 2.2, abs(pBeam.x))
        .mul(smoothstep(float(-1.8), float(0.5), pBeam.y));
      const sunClear = smoothstep(float(0.18), float(0.75), northColonnadeSdf(pBeam.add(sunDir.mul(1.8))));
      shaftAccum.addAssign(bayWave.mul(inGallery).mul(sunClear).mul(0.036));
    }
    const shaftWarmth = mix(cYellowOchre, cBoneCalcite, float(0.52)).mul(
      shaftAccum.mul(float(1.0).add(hoverW.mul(0.25)))
    );

    return toneMapMineral(finalCol.add(shaftWarmth).mul(edgeAo));
  })();

  // ===========================================================================
  // PORTAL WORLD 1 — EAST (+X): PLUNGING LAPIS SPIRAL STAIRWELL
  // ===========================================================================
  const portalEast = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalEast.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('east', uCamLocal);
    const hoverW = uHoverWeights.east;
    const ptr = uPointerLocal.east;

    const t = float(0.02).toVar();
    const hit = float(0.0).toVar();
    const maxDist = float(30.0);

    Loop(64, () => {
      const p = ro.add(rd.mul(t));
      const d = eastStairwellSdf(p);
      If(d.lessThan(0.0045), () => {
        hit.assign(1.0);
        Break();
      });
      t.addAssign(max(float(0.0045), d.mul(0.8)));
      If(t.greaterThan(maxDist), () => {
        Break();
      });
    });

    // Deep subterranean abyssal lapis mist base (or luminous zenith oculus looking up)
    const abyssFog = mix(cLapis.mul(0.22), cAzurite.mul(0.32), float(0.5).add(hoverW.mul(0.2)));
    const oculusGlow = mix(cWarmKaolin, cBoneCalcite, float(0.65))
      .mul(smoothstep(float(0.35), float(0.95), rd.y).mul(1.15));
    const finalCol = abyssFog.add(oculusGlow).toVar();

    If(hit.greaterThan(0.5), () => {
      const pHit = ro.add(rd.mul(t));
      const n = eastStairwellNormal(pHit);
      const qx = pHit.x;
      const qz = pHit.z.sub(2.4);
      const r = length(vec2(qx, qz));
      const theta = safeAtan2(qz, qx);

      const grain = mineralToothFbm(pHit, float(3.4));

      // Identify material regions cleanly by cylindrical radius & height
      const isBrassRail = invSmoothstep(1.185, 1.22, r).mul(
        smoothstep(float(1.1), float(1.135), r)
      );
      const isOuterWall = smoothstep(float(2.24), float(2.285), r);
      const isLanding = invSmoothstep(0.32, 0.48, pHit.z).mul(
        invSmoothstep(-1.73, -1.62, pHit.y)
      );

      // Glazed Lapis-Ultramarine & Azurite-Prussian curved ashlar tiles on cylindrical shaft wall
      const tileAng = theta.mul(16.0 / (Math.PI * 2)).add(1024.0);
      const tileRow = pHit.y.mul(2.1).add(1024.0);
      const rowShift = fract(floor(tileRow).mul(0.5).add(1024.0));
      const uTile = fract(tileAng.add(rowShift).add(1024.0));
      const vTile = fract(tileRow);
      const mortar = smoothstep(float(0.45), float(0.492), max(abs(uTile.sub(0.5)), abs(vTile.sub(0.5))));
      // Horizontal Bone-Calcite string-course molding bands every 4.8 m
      const bandPhase = abs(fract(pHit.y.div(4.8).add(0.15).add(1024.0)).sub(0.5));
      const isStringCourse = invSmoothstep(0.022, 0.036, bandPhase);

      const tileVar = fract(abs(sin(floor(tileAng).mul(12.9898).add(floor(tileRow).mul(78.233))).mul(43758.5453)));
      const lapisTile = mix(
        cLapis.mul(0.95),
        cAzurite.mul(1.08),
        tileVar.mul(0.55).add(grain.mul(0.45))
      );
      const wallAlbedo = mix(
        mix(lapisTile, cBoneCalcite.mul(0.68), mortar.mul(0.42)),
        cBoneCalcite.mul(0.96),
        isStringCourse
      );

      // Crisp cantilevered Bone-Calcite stone stair treads with warm Yellow-Ochre brass nosing edge
      const treadStone = mix(cWarmKaolin.mul(0.86), cBoneCalcite.mul(1.06), grain.mul(0.65).add(max(float(0.0), n.y).mul(0.35)));
      const innerNosing = invSmoothstep(1.12, 1.22, r).mul(max(float(0.0), n.y));
      const stairAlbedo = mix(treadStone, cYellowOchre.mul(1.08), innerNosing.mul(0.58));

      // Satin-brass helical handrail
      const brassAlbedo = mix(cYellowOchre.mul(1.05), cBoneCalcite.mul(0.95), float(0.28));

      const stoneOrWall = mix(stairAlbedo, wallAlbedo, isOuterWall.mul(float(1.0).sub(isLanding)));
      const albedo = mix(stoneOrWall, brassAlbedo, isBrassRail);

      // Overhead warm-cool zenith oculus key light pouring down the cylindrical tower shaft
      const oculusPos = vec3(ptr.x.mul(hoverW).mul(0.45), float(9.5), float(2.4).add(ptr.y.mul(hoverW).mul(0.45)));
      const toOculus = oculusPos.sub(pHit);
      const oculusDist = max(float(1.0), length(toOculus));
      const lightDir = toOculus.div(oculusDist);

      // 10-step soft shadow march toward zenith oculus (makes each cantilevered step cast crisp shadows below!)
      const sh = float(1.0).toVar();
      const ts = float(0.08).toVar();
      Loop(10, () => {
        const hs = eastStairwellSdf(pHit.add(n.mul(0.02)).add(lightDir.mul(ts)));
        sh.assign(min(sh, clamp(hs.mul(5.2).div(ts), float(0.0), float(1.0))));
        ts.addAssign(clamp(hs, float(0.1), float(0.95)));
        If(sh.lessThan(0.02).or(ts.greaterThan(min(oculusDist, float(10.5)))), () => {
          Break();
        });
      });

      const ao = clamp(
        eastStairwellSdf(pHit.add(n.mul(0.16)))
          .div(0.16)
          .mul(0.6)
          .add(eastStairwellSdf(pHit.add(n.mul(0.46))).div(0.46).mul(0.4)),
        0.2,
        1.0
      );

      // Elevation-based zenith falloff + luminous cool subterranean azure/celadon up-glow from abyss!
      const zenithAtten = exp(max(float(0.0), float(1.5).sub(pHit.y)).mul(-0.075));
      const nDotL = max(float(0.0), dot(n, lightDir));
      const oculusCol = mix(cBoneCalcite, cWarmKaolin, float(0.35)).mul(2.05).mul(zenithAtten);

      // Subterranean mineral depth glow rising softly from the deep cistern floor below
      const depthLift = invSmoothstep(-13.0, 2.0, pHit.y);
      const deepGlowStrength = float(0.38).add(hoverW.mul(0.28)).add(sin(uTime.mul(0.8)).mul(0.04));
      const deepGlowCol = mix(cAzurite, cCeladon, float(0.42).add(hoverW.mul(0.16)))
        .mul(deepGlowStrength)
        .mul(float(0.45).add(max(float(0.0), n.y.negate()).mul(0.55)))
        .mul(float(0.35).add(depthLift.mul(0.95)));

      // Crisp specular highlight on glazed lapis tiles and helical brass handrail
      const halfV = normalize(lightDir.sub(rd));
      const specExp = mix(float(26.0), float(48.0), isBrassRail);
      const specGain = mix(isOuterWall.mul(0.34).add(0.08), float(1.25), isBrassRail);
      const specCol = mix(cBoneCalcite, cYellowOchre, isBrassRail).mul(
        pow(max(float(0.0), dot(n, halfV)), specExp).mul(specGain).mul(float(0.3).add(sh.mul(0.7)))
      );

      const ambientLapis = mix(cLapis.mul(0.26), cWarmKaolin.mul(0.16), float(0.35)).mul(ao);
      const lit = albedo
        .mul(oculusCol.mul(nDotL.mul(sh)).add(ambientLapis).add(deepGlowCol))
        .add(specCol);

      // Exponential depth + vertical plunge haze (sells impossible subterranean drop below cube floor!)
      const plungeDepth = max(float(0.0), float(-1.8).sub(pHit.y));
      const fogAmt = float(1.0).sub(exp(t.mul(-0.052).sub(plungeDepth.mul(0.058))));
      const fogCol = mix(
        cLapis.mul(0.28),
        mix(cAzurite.mul(0.42), cCeladon.mul(0.36), hoverW.mul(0.35)),
        clamp(plungeDepth.div(14.0), 0.0, 1.0)
      );
      finalCol.assign(mix(lit, fogCol, clamp(fogAmt, 0.0, 0.92)));
    });

    return toneMapMineral(finalCol.mul(edgeAo));
  })();

  // ===========================================================================
  // PORTAL WORLD 2 — SOUTH (-Z): AMBER LANTERN OVER DARK STILL WATER
  // ===========================================================================
  const portalSouth = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalSouth.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('south', uCamLocal);
    const hoverW = uHoverWeights.south;
    const ptr = uPointerLocal.south;

    // Gently swaying Realgar-Amber brass lantern pendulum (subtly nudged on pointer hover)
    const swayX = sin(uTime.mul(1.15)).mul(0.11).add(ptr.x.mul(hoverW).mul(0.16));
    const swayY = cos(uTime.mul(2.3)).mul(0.015);
    const swayZ = cos(uTime.mul(0.85)).mul(0.05);
    const lanternPos = vec3(swayX, float(0.22).add(swayY), float(3.45).add(swayZ));
    const farLanternPos = vec3(sin(uTime.mul(0.7).add(1.4)).mul(0.06), float(0.38), float(13.05));

    // Flickering organic oil-lantern flame pulse
    const flamePulse = float(1.0)
      .add(sin(uTime.mul(6.4)).mul(0.045))
      .add(cos(uTime.mul(11.3)).mul(0.025));

    // Analytical water plane intersection (y = -1.76, seated strictly below portal sill)
    const waterY = float(-1.76);
    const tWaterRaw = waterY.sub(ro.y).div(rd.y.lessThan(-1e-4).select(rd.y, float(-1.0)));
    const hitsWaterPlane = rd.y.lessThan(-1e-4).and(tWaterRaw.greaterThan(0.005)).and(tWaterRaw.lessThan(25.0));
    const tWater = hitsWaterPlane.select(tWaterRaw, float(100.0));

    // Primary SDF march for vaulted stone cistern, arcade piers, vault ribs & lantern cage
    const tSolid = float(0.02).toVar();
    const hitSolid = float(0.0).toVar();
    const maxDist = min(float(25.5), tWater.add(0.08));

    Loop(52, () => {
      const p = ro.add(rd.mul(tSolid));
      const d = southCisternSolidSdf(p, lanternPos);
      If(d.lessThan(0.005), () => {
        hitSolid.assign(1.0);
        Break();
      });
      tSolid.addAssign(max(float(0.005), d.mul(0.84)));
      If(tSolid.greaterThan(maxDist), () => {
        Break();
      });
    });

    const fogFloor = mix(cRawUmber.mul(0.06), cVerdigris.mul(0.085), float(0.5));
    const finalCol = fogFloor.toVar();
    const tVisual = min(tSolid, tWater).toVar();

    const waterIsNearest = hitsWaterPlane.and(tWater.lessThan(tSolid));

    If(waterIsNearest, () => {
      const pW = ro.add(rd.mul(tWater));

      // Gentle dark cistern micro-swells + physical concentric ripple rings driven by pointer hover!
      const rippleCenter = vec2(ptr.x.mul(1.35), float(3.8).add(ptr.y.mul(2.1)));
      const dRipVec = vec2(pW.x, pW.z).sub(rippleCenter);
      const dRip = length(dRipVec);
      const ripEnv = exp(dRip.mul(-0.62)).mul(float(0.14).add(hoverW.mul(0.86)));
      const ripGrad = cos(dRip.mul(7.5).sub(uTime.mul(4.8))).mul(ripEnv).mul(0.048);
      const ripDir = dRipVec.div(max(float(0.08), dRip));

      const ambGx = cos(pW.x.mul(2.6).add(uTime.mul(1.1)))
        .mul(0.011)
        .add(sin(pW.z.mul(4.1).sub(uTime.mul(1.4))).mul(0.008));
      const ambGz = sin(pW.z.mul(3.2).add(uTime.mul(0.95)))
        .mul(0.014)
        .add(cos(pW.x.mul(3.8).add(pW.z.mul(2.2)).sub(uTime.mul(1.3))).mul(0.008));

      const nWater = normalize(
        vec3(ambGx.add(ripDir.x.mul(ripGrad)).negate(), float(1.0), ambGz.add(ripDir.y.mul(ripGrad)).negate())
      );

      // Dielectric water Fresnel reflection of vaulted cistern & glowing swaying amber lantern
      const ndotv = clamp(dot(nWater, rd.negate()), 0.0, 1.0);
      const fresnel = float(0.055).add(float(0.9).mul(pow(max(float(0.0), float(1.0).sub(ndotv)), 3.8)));

      const rdRefl = normalize(reflect(rd, nWater));
      const tR = float(0.06).toVar();
      const hitR = float(0.0).toVar();
      Loop(24, () => {
        const pR = pW.add(rdRefl.mul(tR));
        const dR = southCisternSolidSdf(pR, lanternPos);
        If(dR.lessThan(0.018), () => {
          hitR.assign(1.0);
          Break();
        });
        tR.addAssign(max(float(0.025), dR.mul(0.92)));
        If(tR.greaterThan(18.0), () => {
          Break();
        });
      });

      const reflCol = fogFloor.toVar();
      If(hitR.greaterThan(0.5), () => {
        reflCol.assign(
          shadeSouthCisternSolid(
            pW.add(rdRefl.mul(tR)),
            rdRefl,
            tR.add(tWater.mul(0.45)),
            lanternPos,
            farLanternPos,
            flamePulse
          )
        );
      });

      // Shimmering specular reflection streak from the swaying Realgar-Amber lantern across ripples
      const toLanternW = normalize(lanternPos.sub(pW));
      const hW = normalize(toLanternW.sub(rd));
      const lanternGlint = pow(max(float(0.0), dot(nWater, hW)), 76.0).mul(2.8);
      const lanternBroad = pow(max(float(0.0), dot(nWater, hW)), 14.0).mul(0.42);
      const lDistW = length(lanternPos.sub(pW));
      const glintAtten = float(1.0).div(float(0.55).add(lDistW.mul(lDistW).mul(0.09)));

      const toFarW = normalize(farLanternPos.sub(pW));
      const hFarW = normalize(toFarW.sub(rd));
      const farGlint = pow(max(float(0.0), dot(nWater, hFarW)), 56.0).mul(0.95);

      const shimmerCol = mix(cRealgarAmber, cSporeSulfur, float(0.35))
        .mul(lanternGlint.add(lanternBroad).mul(glintAtten).add(farGlint))
        .mul(flamePulse);

      // Deep dark submerged Beer-Lambert verdigris/umber water body + faint caustic tint under lantern
      const underWater = mix(cRawUmber.mul(0.04), cVerdigris.mul(0.09), float(0.62))
        .add(cRealgarAmber.mul(0.075).mul(glintAtten));

      const waterFog = float(1.0).sub(exp(tWater.mul(-0.068)));
      const waterShaded = mix(underWater, reflCol, clamp(fresnel.mul(1.15), 0.12, 0.96)).add(shimmerCol);
      finalCol.assign(mix(waterShaded, fogFloor, clamp(waterFog, 0.0, 0.9)));
    }).Else(() => {
      If(hitSolid.greaterThan(0.5), () => {
        finalCol.assign(
          shadeSouthCisternSolid(ro.add(rd.mul(tSolid)), rd, tSolid, lanternPos, farLanternPos, flamePulse)
        );
      });
    });

    // Soft atmospheric Realgar-Amber optical glow around foreground & distant lanterns
    const toL1Ray = lanternPos.sub(ro);
    const tClose1 = clamp(dot(toL1Ray, rd), float(0.0), tVisual);
    const dClose1Sq = dot(ro.add(rd.mul(tClose1)).sub(lanternPos), ro.add(rd.mul(tClose1)).sub(lanternPos));
    const halo1 = exp(dClose1Sq.mul(-3.2)).mul(0.34).add(exp(dClose1Sq.mul(-0.55)).mul(0.065));

    const toL2Ray = farLanternPos.sub(ro);
    const tClose2 = clamp(dot(toL2Ray, rd), float(0.0), tVisual);
    const dClose2Sq = dot(ro.add(rd.mul(tClose2)).sub(farLanternPos), ro.add(rd.mul(tClose2)).sub(farLanternPos));
    const halo2 = exp(dClose2Sq.mul(-2.2)).mul(0.24);

    const lanternHalo = mix(cRealgarAmber, cYellowOchre, float(0.32))
      .mul(halo1.add(halo2).mul(flamePulse));

    return toneMapMineral(finalCol.add(lanternHalo).mul(edgeAo));
  })();

  // ===========================================================================
  // PORTAL WORLD 3 — WEST (-X): WARM OCHRE DESERT SLOT-CANYON ARCH
  // ===========================================================================
  const portalWest = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalWest.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('west', uCamLocal);
    const hoverW = uHoverWeights.west;
    const ptr = uPointerLocal.west;

    // High warm desert sun pouring down through the canyon slot & backlighting the natural rock arch
    const sunShiftX = ptr.x.mul(hoverW).mul(0.24).add(sin(uTime.mul(0.22)).mul(0.03));
    const sunShiftY = ptr.y.mul(hoverW).mul(0.1);
    const sunDir = normalize(vec3(float(-0.4).add(sunShiftX), float(0.79).add(sunShiftY), float(0.46)));

    const t = float(0.02).toVar();
    const hit = float(0.0).toVar();
    const maxDist = float(29.0);

    Loop(60, () => {
      const p = ro.add(rd.mul(t));
      const d = westCanyonSdf(p);
      If(d.lessThan(0.005), () => {
        hit.assign(1.0);
        Break();
      });
      t.addAssign(max(float(0.005), d.mul(0.76)));
      If(t.greaterThan(maxDist), () => {
        Break();
      });
    });

    // Warm pale Kaolin & Bone-Calcite desert sky crowning the slot canyon & shining through the arch
    const skyT = clamp(rd.y.mul(0.75).add(0.38), 0.0, 1.0);
    const sunAlign = max(float(0.0), dot(rd, sunDir));
    const desertSky = mix(
      mix(cWarmKaolin.mul(1.02), cYellowOchre.mul(0.94), float(0.24)),
      cBoneCalcite.mul(1.24),
      skyT
    ).add(cBoneCalcite.mul(pow(sunAlign, 12.0).mul(0.5)));

    const finalCol = desertSky.toVar();

    If(hit.greaterThan(0.5), () => {
      const pHit = ro.add(rd.mul(t));
      const n = westCanyonNormal(pHit);
      const grain = mineralToothFbm(pHit, float(2.2));
      const fineSand = mineralToothFbm(pHit.add(vec3(5.2, 1.7, 8.4)), float(9.5));

      // Sculpted layered Navajo/Petra sandstone geological strata in Burnt Sienna, Yellow Ochre & Hematite
      const strataCoord = pHit.y
        .mul(3.4)
        .add(sin(pHit.z.mul(0.45).add(pHit.x.mul(0.35))).mul(0.65))
        .add(grain.mul(0.85));
      const layerA = sin(strataCoord).mul(0.5).add(0.5);
      const layerB = cos(strataCoord.mul(2.3).add(1.1)).mul(0.5).add(0.5);
      const thinVein = smoothstep(float(0.78), float(0.95), sin(strataCoord.mul(4.8)).mul(0.5).add(0.5));

      const rockBase = mix(cBurntSienna.mul(0.88), cYellowOchre.mul(0.94), layerA);
      const rockRich = mix(rockBase, cHematiteRust.mul(0.78), layerB.mul(0.34));
      const rockStrata = mix(rockRich, cWarmKaolin.mul(0.86), thinVein.mul(0.38).add(fineSand.mul(0.1)));

      // Soft wind-rippled ochre sand drift on the flat canyon floor
      const isSandFloor = smoothstep(float(0.68), float(0.9), n.y).mul(
        invSmoothstep(-1.5, -0.8, pHit.y)
      );
      const sandRipple = sin(pHit.z.mul(9.5).add(pHit.x.mul(2.4)).add(grain.mul(2.0))).mul(0.5).add(0.5);
      const sandAlbedo = mix(cYellowOchre.mul(0.82), cWarmKaolin.mul(0.9), sandRipple.mul(0.32).add(0.28));
      const albedo = mix(rockStrata, sandAlbedo, isSandFloor);

      // 12-step soft desert sun shadow march (carves crisp natural rock-arch shadow across floor & walls)
      const sh = float(1.0).toVar();
      const ts = float(0.08).toVar();
      Loop(12, () => {
        const hs = westCanyonSdf(pHit.add(n.mul(0.025)).add(sunDir.mul(ts)));
        sh.assign(min(sh, clamp(hs.mul(6.2).div(ts), float(0.0), float(1.0))));
        ts.addAssign(clamp(hs, float(0.12), float(0.95)));
        If(sh.lessThan(0.02).or(ts.greaterThan(11.5)), () => {
          Break();
        });
      });

      const ao = clamp(
        westCanyonSdf(pHit.add(n.mul(0.22)))
          .div(0.22)
          .mul(0.55)
          .add(westCanyonSdf(pHit.add(n.mul(0.62))).div(0.62).mul(0.45)),
        0.24,
        1.0
      );

      const nDotL = max(float(0.0), dot(n, sunDir));
      const desertSunCol = mix(cBoneCalcite, cYellowOchre, float(0.28)).mul(2.15);
      // Warm bounced terracotta radiosity glowing inside shaded slot-canyon curves!
      const canyonBounce = mix(cBurntSienna, cRealgarAmber, float(0.42))
        .mul(float(0.32).add(abs(n.x).mul(0.22)))
        .mul(ao);
      const kaolinSkyFill = cWarmKaolin
        .mul(max(float(0.0), n.y.mul(0.5).add(0.5)).mul(0.26))
        .mul(ao);

      const lit = albedo.mul(
        desertSunCol.mul(nDotL.mul(sh)).add(canyonBounce).add(kaolinSkyFill)
      );

      // Warm golden desert atmospheric perspective haze beyond the natural rock arch
      const fogT = float(1.0).sub(exp(t.mul(-0.044)));
      const hazeCol = mix(cWarmKaolin.mul(0.96), cYellowOchre.mul(0.88), float(0.36));
      finalCol.assign(mix(lit, hazeCol, clamp(fogT, 0.0, 0.85)));
    });

    return toneMapMineral(finalCol.mul(edgeAo));
  })();

  const portalMaterials: Record<PortalFaceId, THREE.MeshBasicNodeMaterial> = {
    north: portalNorth,
    east: portalEast,
    south: portalSouth,
    west: portalWest,
  };

  return {
    matteBasalt,
    basaltAccent,
    satinBrass,
    darkBrassFastener,
    portalMaterials,
    uCamLocal,
    uTime,
    uHoverWeights,
    uPointerLocal,
    dispose: () => {
      matteBasalt.dispose();
      basaltAccent.dispose();
      satinBrass.dispose();
      darkBrassFastener.dispose();
      portalNorth.dispose();
      portalEast.dispose();
      portalSouth.dispose();
      portalWest.dispose();
    },
  };
}
