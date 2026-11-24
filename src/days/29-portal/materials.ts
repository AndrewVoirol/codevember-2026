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
  CUBE_HALF,
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
  addSoftbox(11.5, 9.0, [-4.8, 5.8, 4.4], 1.0, 0.96, 0.89, 1.92);
  // Right tall architectural studio scrim (highlights vertical brass arch bezels & corner piers)
  addSoftbox(8.5, 11.5, [6.2, 2.4, 3.2], 0.99, 0.95, 0.9, 1.78);
  // Rear-top cool mineral rim softbox (separates dark honed basalt cleanly off #212529)
  addSoftbox(13.0, 5.6, [-1.2, 5.4, -5.8], 0.9, 0.95, 1.0, 1.64);
  // Overhead diffused zenith scrim (catches top brass compass-rose inlay)
  addSoftbox(9.5, 9.5, [0.4, 7.2, 0.4], 0.97, 0.95, 0.91, 1.15);
  // Opposite cool-slate flank fill (ensures South & West brass brackets gleam across 360° orbits)
  addSoftbox(9.0, 7.5, [-5.6, 1.4, -4.8], 0.86, 0.93, 0.99, 1.28);
  addSoftbox(8.5, 7.0, [5.0, 1.2, -5.4], 0.95, 0.92, 0.88, 1.22);
  // Subtle warm gallery bounce from below (illuminates underside compass plate & plinth bevels)
  addSoftbox(12.0, 12.0, [0.0, -6.4, 0.0], 0.96, 0.92, 0.86, 0.45);

  const renderTarget = pmrem.fromScene(envScene, 0.068);
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
 * Produces crisp honed volcanic stone grain or fine brushed brass micro-relief without macro waviness.
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

/**
 * Subtle physically grounded portal-world bounce light kissing the recessed stone jambs
 * and inner satin-brass window bore walls (`d_face` between aperture plane and outer facade).
 * Visually unites the physical cube frame with each portal's distinct atmosphere.
 */
const computeRecessedPortalSpill = Fn(([p_in, n_in, uTime_in, gain_in]: [TslNode, TslNode, TslNode, TslNode]) => {
  const p = vec3(p_in);
  const n = normalize(vec3(n_in));
  const t = float(uTime_in);

  const cNorthBounce = mix(
    vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear),
    vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear),
    float(0.22)
  );
  const cEastBounce = mix(
    vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear),
    vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear),
    float(0.38)
  );
  const flameFlicker = float(1.0).add(sin(t.mul(6.4)).mul(0.06)).add(cos(t.mul(11.3)).mul(0.035));
  const cSouthBounce = mix(
    vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear),
    vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear),
    float(0.25)
  ).mul(flameFlicker);
  const cWestBounce = mix(
    vec3(...MINERAL_PALETTE.burntSienna.rInfLinear),
    vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear),
    float(0.42)
  );

  // Compute per-face recessed aperture proximity (strictly inside each face's window reveal)
  const evalFaceWeight = (depthCoord: TslNode, uCoord: TslNode, vCoord: TslNode, faceNorm: TslNode) => {
    const inDepth = smoothstep(float(APERTURE_PLANE_OFFSET - 0.015), float(APERTURE_PLANE_OFFSET + 0.008), depthCoord).mul(
      float(1.0).sub(smoothstep(float(APERTURE_PLANE_OFFSET + 0.025), float(CUBE_HALF - 0.008), depthCoord))
    );
    const inWidth = float(1.0).sub(smoothstep(float(ARCH_HALF_WIDTH - 0.03), float(ARCH_HALF_WIDTH + 0.065), abs(uCoord)));
    const inVert = smoothstep(float(ARCH_SILL_Y - 0.045), float(ARCH_SILL_Y + 0.015), vCoord).mul(
      float(1.0).sub(smoothstep(float(ARCH_SPRING_Y + ARCH_RADIUS - 0.02), float(ARCH_SPRING_Y + ARCH_RADIUS + 0.06), vCoord))
    );
    // Favor inward-facing bore/jamb surfaces & upper sill faces (not outer facade facing outward!)
    const notOuterFacade = float(1.0).sub(smoothstep(float(0.65), float(0.95), dot(n, faceNorm)));
    return inDepth.mul(inWidth).mul(inVert).mul(notOuterFacade);
  };

  const wNorth = evalFaceWeight(p.z, p.x, p.y, vec3(0, 0, 1));
  const wEast = evalFaceWeight(p.x, p.z.negate(), p.y, vec3(1, 0, 0));
  const wSouth = evalFaceWeight(p.z.negate(), p.x.negate(), p.y, vec3(0, 0, -1));
  const wWest = evalFaceWeight(p.x.negate(), p.z, p.y, vec3(-1, 0, 0));

  return cNorthBounce
    .mul(wNorth.mul(0.85))
    .add(cEastBounce.mul(wEast.mul(1.05)))
    .add(cSouthBounce.mul(wSouth.mul(1.25)))
    .add(cWestBounce.mul(wWest.mul(1.05)))
    .mul(float(gain_in));
});

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

/**
 * Subtle architectural reveal shadow right at the perimeter lip of the arched window aperture.
 * Weighted by near-threshold proximity in each portal shader so distant sky stays razor-clean.
 */
const computeArchApertureEdgeAo = Fn(([u_in, v_in]: [TslNode, TslNode]) => {
  const u = float(u_in);
  const v = float(v_in);
  const dSide = float(ARCH_HALF_WIDTH).sub(abs(u));
  const dSill = v.sub(float(ARCH_SILL_Y));
  const dv = max(float(0.0), v.sub(float(ARCH_SPRING_Y)));
  const dCrown = float(ARCH_RADIUS).sub(length(vec2(u, dv)));
  const dEdge = min(dSide, min(dSill, dCrown));
  return float(0.80).add(smoothstep(float(0.0), float(0.042), dEdge).mul(0.20));
});

/** Unified mineral-gamut filmic highlight shoulder keeping HDR shafts, skies & lanterns natural. */
const toneMapMineral = Fn(([raw_in]: [TslNode]) => {
  const raw = max(vec3(0.0), vec3(raw_in));
  const cCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
  const lum = dot(raw, vec3(0.2126, 0.7152, 0.0722));
  const compressed = raw.div(vec3(1.0).add(raw.mul(0.36)));
  const warmRoll = smoothstep(float(1.85), float(5.8), lum).mul(0.22);
  return mix(compressed, cCalcite.mul(0.98), warmRoll);
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

/**
 * Lightweight 3-tap finite-difference bump normal in portal-local space (used once at raymarch hit point).
 */
const applyPortalMicroNormal = Fn(
  ([n_in, p_in, scale_in, strength_in]: [TslNode, TslNode, TslNode, TslNode]) => {
    const n = normalize(vec3(n_in));
    const p = vec3(p_in);
    const sc = float(scale_in);
    const eps = float(0.14).div(max(float(1.0), sc));

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
    return normalize(n.sub(tanGrad.mul(float(strength_in))));
  }
);

// ---------------------------------------------------------------------------
// World 0 — North (`+Z`): Sunlit Mediterranean Travertine Colonnade
// ---------------------------------------------------------------------------

const northColonnadeSdf = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const dFloor = p.y.add(1.76);
  const dThresholdStep = sdBox3Tsl(vec3(p.x, p.y.add(1.77), p.z.sub(0.26)), vec3(1.52, 0.024, 0.17)).sub(0.006);

  const colSpacing = float(2.7);
  const colIdx = floor(p.z.sub(1.85).div(colSpacing).add(0.5));
  const colZ = clamp(colIdx.mul(colSpacing).add(1.85), float(1.85), float(20.75));
  const lx = abs(p.x).sub(1.76);
  const lz = p.z.sub(colZ);

  const phi = safeAtan2(lz, lx);
  const flute = abs(sin(phi.mul(6.0))).mul(0.017);
  const rad = length(vec2(lx, lz));
  const shaftT = clamp(p.y.sub(0.04).div(1.54), float(-1.0), float(1.0));
  const shaftR = float(0.275).sub(shaftT.mul(0.5).add(0.5).mul(0.024)).sub(flute);
  const dShaftCyl = max(rad.sub(shaftR), abs(p.y.sub(0.04)).sub(1.52));

  // Attic base + Doric/Tuscan capital moldings
  const dPlinth = sdBox3Tsl(vec3(lx, p.y.add(1.62), lz), vec3(0.37, 0.14, 0.37)).sub(0.012);
  const dBaseTorus = length(vec2(rad.sub(0.315), p.y.add(1.43))).sub(0.055);
  const dNeckRing = length(vec2(rad.sub(0.275), p.y.sub(1.51))).sub(0.032);
  const dEchinus = length(vec2(rad.sub(0.325), p.y.sub(1.60))).sub(0.068);
  const dAbacus = sdBox3Tsl(vec3(lx, p.y.sub(1.72), lz), vec3(0.40, 0.085, 0.40)).sub(0.012);
  const dColumn = min(
    dShaftCyl,
    min(dPlinth, min(dBaseTorus, min(dNeckRing, min(dEchinus, dAbacus))))
  );

  // Semicircular side colonnade arcade arches spanning between columns
  const bayIdx = floor(p.z.sub(3.2).div(colSpacing).add(0.5));
  const bayZ = clamp(bayIdx.mul(colSpacing).add(3.2), float(3.2), float(19.4));
  const bLz = p.z.sub(bayZ);
  const dArcadeWall = sdBox3Tsl(vec3(abs(p.x).sub(1.76), p.y.sub(2.12), p.z.sub(11.4)), vec3(0.30, 0.44, 11.4));
  const dArcadeCut = length(vec2(bLz, max(float(0.0), p.y.sub(1.48)))).sub(0.96);
  const dSideArcade = max(dArcadeWall, dArcadeCut.negate());

  // Molded cornice ledge & transverse coffered beams
  const dCornice = sdBox3Tsl(vec3(abs(p.x).sub(1.76), p.y.sub(2.58), p.z.sub(11.4)), vec3(0.42, 0.08, 11.55)).sub(0.01);
  const dCrossBeam = sdBox3Tsl(vec3(p.x, p.y.sub(2.34), lz), vec3(1.88, 0.22, 0.24)).sub(0.01);

  // Left classical gallery wall with recessed vertical wall niches aligned with bays
  const dLeftWallRaw = p.x.add(4.35);
  const dNicheCut = sdBox3Tsl(vec3(p.x.add(4.35), p.y.sub(0.05), bLz), vec3(0.38, 1.25, 0.72));
  const dLeftWall = max(dLeftWallRaw, dNicheCut.negate());

  // Low sunlit right stone parapet opening outward to bright Mediterranean sky
  const dRightParapet = sdBox3Tsl(vec3(p.x.sub(2.95), p.y.add(1.28), p.z.sub(11.4)), vec3(0.16, 0.48, 11.2)).sub(0.012);

  // Monumental triumphal arch terminus at z = 22.8
  const dBackSlab = sdBox3Tsl(vec3(p.x, p.y.sub(0.48), p.z.sub(22.8)), vec3(4.3, 3.05, 0.50));
  const dArchHole = length(vec2(p.x, max(float(0.0), p.y.sub(0.26)))).sub(1.38);
  const dBackArch = max(dBackSlab, dArchHole.negate());

  // Inviting sunlit terrace steps & freestanding classical rotunda pavilion framed by sky through the triumphal arch
  const dTerraceStep1 = sdBox3Tsl(vec3(p.x, p.y.add(1.56), p.z.sub(26.2)), vec3(5.5, 0.20, 2.2));
  const dTerraceStep2 = sdBox3Tsl(vec3(p.x, p.y.add(1.32), p.z.sub(28.4)), vec3(5.5, 0.20, 2.0));
  const dRotundaDome = length(vec3(p.x, p.y.sub(0.68), p.z.sub(30.2))).sub(1.05);
  const dRotundaDrum = max(length(vec2(p.x, p.z.sub(30.2))).sub(0.96), abs(p.y.add(0.16)).sub(0.88));
  const dVista = min(dTerraceStep1, min(dTerraceStep2, min(dRotundaDome, dRotundaDrum)));

  return min(
    dFloor,
    min(
      dThresholdStep,
      min(
        dColumn,
        min(
          dSideArcade,
          min(
            dCornice,
            min(dCrossBeam, min(dLeftWall, min(dRightParapet, min(dBackArch, dVista))))
          )
        )
      )
    )
  );
});

const northColonnadeNormal = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0032);
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
// World 1 — East (`+X`): Plunging Midnight Lapis Spiral Stairwell
// ---------------------------------------------------------------------------

/**
 * Evaluates East stairwell sub-SDF components as vec4(dTotal, dWall, dStairsOrLanding, dBrassMetal)
 * so surface shading discriminates exact materials cleanly with zero radius color-bleed.
 */
const eastStairwellParts = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const shaftCz = float(2.4);
  const shaftR = float(2.34);
  const qx = p.x;
  const qz = p.z.sub(shaftCz);
  const r = length(vec2(qx, qz));
  const theta = safeAtan2(qz, qx);

  const twoPi = float(Math.PI * 2);
  const hTurn = float(4.8);
  const nSteps = float(16.0);
  const dTheta = twoPi.div(nSteps);

  // 8 vertical architectural pilasters around the cylindrical shaft wall
  const pilSector = twoPi.div(8.0);
  const pilAng = theta.sub(floor(theta.div(pilSector).add(0.5)).mul(pilSector));
  const pilArc = abs(pilAng).mul(r);
  const dPilaster = max(r.sub(shaftR.add(0.06)), max(shaftR.sub(0.095).sub(r), pilArc.sub(0.12)));

  // Horizontal carved Bone-Calcite cornice ring ledges every 4.8 m full turn
  const ringRaw = p.y.add(0.72);
  const ringY = ringRaw.sub(floor(ringRaw.div(hTurn).add(0.5)).mul(hTurn));
  const dCorniceRing = max(shaftR.sub(0.085).sub(r), abs(ringY).sub(0.075));

  const dCylWall = min(shaftR.sub(r), min(dPilaster, dCorniceRing));
  const dAlcoveCut = sdBox3Tsl(vec3(p.x, p.y.sub(0.02), p.z.sub(0.28)), vec3(1.58, 1.88, 0.76));
  const dWall = max(dCylWall, dAlcoveCut.negate());

  // Threshold balcony landing (strictly below sill and clear of z = 0)
  const dLanding = sdBox3Tsl(vec3(p.x, p.y.add(1.97), p.z.sub(0.34)), vec3(1.56, 0.21, 0.27)).sub(0.012);

  // Cantilevered helical treads + slender vertical brass balusters
  const thetaPhase = theta.add(float(Math.PI * 0.5));
  const yPhase = p.y.add(1.72).add(thetaPhase.div(twoPi).mul(hTurn));
  const kWrap = floor(yPhase.div(hTurn).add(0.5));
  const baseIdx = floor(thetaPhase.div(dTheta));
  const rClamped = max(float(1.12), r);

  const evalStepAndBaluster = (offsetVal: number) => {
    const idx = baseIdx.add(float(offsetVal));
    const stepTheta = idx.add(0.5).mul(dTheta);
    const dAngRaw = thetaPhase.sub(stepTheta);
    const dAng = dAngRaw.sub(floor(dAngRaw.div(twoPi).add(0.5)).mul(twoPi));

    const stepY = float(-1.72).sub(idx.add(0.5).div(nSteps).mul(hTurn)).add(kWrap.mul(hTurn));
    const yDiffRaw = p.y.sub(stepY);
    const yDiff = yDiffRaw.sub(floor(yDiffRaw.div(hTurn).add(0.5)).mul(hTurn));

    const dRad = abs(r.sub(1.73)).sub(0.59);
    const dVert = abs(yDiff).sub(0.088);
    const dArc = abs(dAng.mul(rClamped)).sub(dTheta.mul(rClamped).mul(0.45).add(0.015));
    const q = vec3(dRad, dVert, dArc);
    const dStep = length(max(q, vec3(0.0)))
      .add(min(max(q.x, max(q.y, q.z)), float(0.0)))
      .sub(0.012);

    // Slender vertical brass baluster rising from inner tread corner up to handrail
    const balArc = dAng.mul(1.16);
    const balRad = r.sub(1.16);
    const balH = abs(yDiff.sub(0.44)).sub(0.36);
    const dBal = max(length(vec2(balRad, balArc)).sub(0.018), balH);

    return vec2(dStep, dBal);
  };

  const sb0 = evalStepAndBaluster(0);
  const sbM = evalStepAndBaluster(-1);
  const sbP = evalStepAndBaluster(1);

  const dStairs = min(sb0.x, min(sbM.x, sbP.x));
  const dBalusters = min(sb0.y, min(sbM.y, sbP.y));

  // Helical satin-brass inner handrail spiraling along r = 1.16
  const yRailRaw = yPhase.sub(kWrap.mul(hTurn)).sub(0.78);
  const yRail = yRailRaw.sub(floor(yRailRaw.div(hTurn).add(0.5)).mul(hTurn));
  const dRail = length(vec2(r.sub(1.16), yRail)).sub(0.034);

  const dAbyssFloor = p.y.add(22.0);

  const dStoneTreads = min(dStairs, dLanding);
  const dBrass = min(dRail, dBalusters);
  const dTotal = min(dWall, min(dStoneTreads, min(dBrass, dAbyssFloor)));

  return vec3(dTotal, dStoneTreads, dBrass);
});

const eastStairwellSdf = Fn(([p_in]: [TslNode]) => {
  return eastStairwellParts(p_in).x;
});

const eastStairwellNormal = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0034);
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
// World 2 — South (`-Z`): Vaulted Verdigris Basilica Cistern & Receding Lanterns
// ---------------------------------------------------------------------------

/** Evaluates one sculpted hanging filigree brass lantern: returns vec2(dGlass, dBrassAndChain). */
const southLanternEval = Fn(([p_in, c_in]: [TslNode, TslNode]) => {
  const qL = vec3(p_in).sub(vec3(c_in));
  const rL = length(vec2(qL.x, qL.z));
  const dGlass = max(rL.sub(0.195), abs(qL.y).sub(0.27)).sub(0.015);
  const dCap = max(rL.sub(0.285), abs(qL.y.sub(0.31)).sub(0.055)).sub(0.014);
  const dDome = length(vec3(qL.x, qL.y.sub(0.39).mul(1.35), qL.z)).sub(0.17);
  const dBase = max(rL.sub(0.255), abs(qL.y.add(0.30)).sub(0.042)).sub(0.012);
  const dFinial = length(vec3(qL.x, qL.y.add(0.40).mul(0.9), qL.z)).sub(0.062);
  const dChain = max(rL.sub(0.016), float(0.44).sub(qL.y));
  const dMetal = min(dCap, min(dDome, min(dBase, min(dFinial, dChain))));
  return vec2(dGlass, dMetal);
});

/**
 * Solid vaulted basilica cistern + threshold quay + 3 receding brass filigree lanterns.
 * Returns vec3(dTotal, dLanternGlassMin, dLanternMetalMin).
 */
const southCisternParts = Fn(([p_in, l1_in, l2_in, l3_in]: [TslNode, TslNode, TslNode, TslNode]) => {
  const p = vec3(p_in);

  // Two-step wet carved limestone threshold quay descending cleanly into the dark mirror water
  const dQuayTop = sdBox3Tsl(vec3(p.x, p.y.add(1.855), p.z.sub(0.44)), vec3(1.42, 0.105, 0.32)).sub(0.012);
  const dQuayLow = sdBox3Tsl(vec3(p.x, p.y.add(1.915), p.z.sub(0.78)), vec3(1.24, 0.145, 0.24)).sub(0.012);
  const dQuay = min(dQuayTop, dQuayLow);

  // Barrel-vaulted basilica cistern chamber
  const dSideWalls = float(2.62).sub(abs(p.x));
  const dVaultCeiling = float(2.62).sub(length(vec2(p.x, max(float(0.0), p.y.sub(0.45)))));
  const dChamber = min(dSideWalls, dVaultCeiling);

  // Romanesque arcade piers (plinth + shaft + flared capital) & transverse vault ribs
  const baySpacing = float(3.2);
  const bayIdx = floor(p.z.sub(1.85).div(baySpacing).add(0.5));
  const bayZ = clamp(bayIdx.mul(baySpacing).add(1.85), float(1.85), float(21.05));
  const lz = p.z.sub(bayZ);
  const pierX = abs(p.x).sub(1.88);
  const dPierShaft = sdBox3Tsl(vec3(pierX, p.y.add(0.62), lz), vec3(0.25, 1.56, 0.25)).sub(0.018);
  const dPierBase = sdBox3Tsl(vec3(pierX, p.y.add(1.72), lz), vec3(0.34, 0.22, 0.34)).sub(0.015);
  const dPierCap = sdBox3Tsl(vec3(pierX, p.y.sub(0.88), lz), vec3(0.35, 0.12, 0.35)).sub(0.015);
  const dPier = min(dPierShaft, min(dPierBase, dPierCap));

  const archR = length(vec2(p.x, p.y.sub(0.45)));
  const dRib = max(abs(archR.sub(2.38)).sub(0.15), abs(lz).sub(0.20));
  const dRibUpper = max(dRib, float(0.36).sub(p.y));

  // Longitudinal side-aisle arches connecting consecutive piers along x = ±1.88
  const midIdx = floor(p.z.sub(3.45).div(baySpacing).add(0.5));
  const midBayZ = clamp(midIdx.mul(baySpacing).add(3.45), float(3.45), float(19.45));
  const dSideBeam = sdBox3Tsl(vec3(pierX, p.y.sub(1.24), p.z.sub(11.4)), vec3(0.24, 0.42, 10.6));
  const dSideArchVoid = length(vec2(p.z.sub(midBayZ), max(float(0.0), p.y.sub(0.78)))).sub(1.12);
  const dSideArcade = max(dSideBeam, dSideArchVoid.negate());

  const dBackWall = float(23.0).sub(p.z);

  const l1 = southLanternEval(p, l1_in);
  const l2 = southLanternEval(p, l2_in);
  const l3 = southLanternEval(p, l3_in);

  const dGlass = min(l1.x, min(l2.x, l3.x));
  const dMetal = min(l1.y, min(l2.y, l3.y));

  const dStone = min(
    dQuay,
    min(dChamber, min(dPier, min(dRibUpper, min(dSideArcade, dBackWall))))
  );
  const dTotal = min(dStone, min(dGlass, dMetal));

  return vec3(dTotal, dGlass, dMetal);
});

const southCisternSolidSdf = Fn(([p_in, l1_in, l2_in, l3_in]: [TslNode, TslNode, TslNode, TslNode]) => {
  return southCisternParts(p_in, l1_in, l2_in, l3_in).x;
});

const southCisternNormal = Fn(([p_in, l1_in, l2_in, l3_in]: [TslNode, TslNode, TslNode, TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0038);
  const k1 = vec3(1.0, -1.0, -1.0);
  const k2 = vec3(-1.0, -1.0, 1.0);
  const k3 = vec3(-1.0, 1.0, -1.0);
  const k4 = vec3(1.0, 1.0, 1.0);
  return normalize(
    k1
      .mul(southCisternSolidSdf(p.add(k1.mul(e)), l1_in, l2_in, l3_in))
      .add(k2.mul(southCisternSolidSdf(p.add(k2.mul(e)), l1_in, l2_in, l3_in)))
      .add(k3.mul(southCisternSolidSdf(p.add(k3.mul(e)), l1_in, l2_in, l3_in)))
      .add(k4.mul(southCisternSolidSdf(p.add(k4.mul(e)), l1_in, l2_in, l3_in)))
  );
});

const shadeSouthCisternSolid = Fn(
  ([pHit_in, rayD_in, distVal_in, l1_in, l2_in, l3_in, pulse_in]: [
    TslNode,
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
    const l1 = vec3(l1_in);
    const l2 = vec3(l2_in);
    const l3 = vec3(l3_in);
    const pulse = float(pulse_in);

    const cBoneCalcite = vec3(...MINERAL_PALETTE.boneCalcite.rInfLinear);
    const cYellowOchre = vec3(...MINERAL_PALETTE.yellowOchre.rInfLinear);
    const cRawUmber = vec3(...MINERAL_PALETTE.rawUmber.rInfLinear);
    const cRealgarAmber = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
    const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
    const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
    const cSporeSulfur = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);
    const cAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);

    const parts = southCisternParts(pHit, l1, l2, l3);
    const isLanternGlass = invSmoothstep(0.012, 0.028, parts.y).mul(
      parts.y.lessThan(parts.z.add(0.005)).select(float(1.0), float(0.0))
    );
    const isLanternMetal = invSmoothstep(0.012, 0.028, parts.z).mul(float(1.0).sub(isLanternGlass));

    const nGeom = southCisternNormal(pHit, l1, l2, l3);

    // Dressed Romanesque ashlar stonework mortar grooves & damp mineral patina
    const blockU = pHit.z.mul(1.15).add(pHit.x.mul(1.15)).add(1024.0);
    const blockV = pHit.y.mul(2.15).add(1024.0);
    const rowShift = fract(floor(blockV).mul(0.5).add(1024.0));
    const uBlk = fract(blockU.add(rowShift).add(1024.0));
    const vBlk = fract(blockV);
    const mortarEdgeX = smoothstep(float(0.455), float(0.492), abs(uBlk.sub(0.5)));
    const mortarEdgeY = smoothstep(float(0.445), float(0.490), abs(vBlk.sub(0.5)));
    const mortar = max(mortarEdgeX, mortarEdgeY);

    // Subtle stone normal relief (ashlar chisel tooth + recessed horizontal course lip)
    const nStone = applyPortalMicroNormal(nGeom, pHit, float(5.5), float(0.11));
    const grooveBump = vec3(float(0.0), sin(vBlk.mul(float(Math.PI * 2.0))).mul(mortarEdgeY).mul(0.28), float(0.0));
    const n = normalize(mix(nStone.add(grooveBump), nGeom, max(isLanternGlass, isLanternMetal)));

    const grain = mineralToothFbm(pHit, float(2.9));
    const mossNoise = mineralToothFbm(pHit.add(vec3(7.3, 2.1, 4.9)), float(1.4));

    // Distinct wet capillary waterline band rising ~0.42 m above the mirror water plane (-1.76)
    const wetBand = invSmoothstep(-1.74, -1.30, pHit.y);
    const dampMid = invSmoothstep(-1.55, 0.45, pHit.y);

    // Deep Basilica Cistern palette: Verdigris-Malachite copper staining + dark weathered stone + Celadon moss
    const blockVar = fract(abs(sin(floor(blockU).mul(19.19).add(floor(blockV).mul(73.17))).mul(43758.5453)));
    const dryStone = mix(cRawUmber.mul(0.52), cVerdigris.mul(0.56), float(0.42).add(blockVar.mul(0.24)));
    const patinaStone = mix(dryStone, cVerdigris.mul(0.68), dampMid.mul(0.55).add(grain.mul(0.25)));
    const mossyStone = mix(
      patinaStone,
      cCeladon.mul(0.58),
      smoothstep(float(0.54), float(0.84), mossNoise).mul(0.48).mul(dampMid.mul(0.6).add(0.4))
    );
    // Wet capillary darkening & recessed dark mortar lines
    const mortaredStone = mix(mossyStone, cRawUmber.mul(0.14), mortar.mul(0.75));
    const stoneAlbedo = mortaredStone.mul(float(1.0).sub(wetBand.mul(0.34)));
    const brassAlbedo = mix(cYellowOchre.mul(0.82), cRawUmber.mul(0.55), float(0.22));
    const albedo = mix(stoneAlbedo, brassAlbedo, isLanternMetal);

    // Three localized warm Realgar-Amber / Spore-Sulfur lantern point lights receding down the vaulted hall
    const evalPointLight = (lPos: TslNode, intensity: TslNode) => {
      const toL = vec3(lPos).sub(pHit);
      const dSq = max(float(0.16), dot(toL, toL));
      const lDir = normalize(toL);
      const atten = float(intensity).div(float(0.38).add(dSq.mul(0.19)));
      const nDotL = max(float(0.0), dot(n, lDir));
      const h = normalize(lDir.sub(rayD));
      const specExp = mix(mix(float(22.0), float(64.0), wetBand), float(46.0), isLanternMetal);
      const specScale = mix(float(0.14).add(wetBand.mul(0.55)), float(0.92), isLanternMetal);
      const spec = pow(max(float(0.0), dot(n, h)), specExp).mul(specScale);
      return vec2(nDotL.mul(atten), spec.mul(atten));
    };

    const lt1 = evalPointLight(l1, float(2.95).mul(pulse));
    const lt2 = evalPointLight(l2, float(2.45));
    const lt3 = evalPointLight(l3, float(2.15));

    const ao = clamp(
      southCisternSolidSdf(pHit.add(nGeom.mul(0.22)), l1, l2, l3)
        .div(0.22)
        .mul(0.6)
        .add(southCisternSolidSdf(pHit.add(nGeom.mul(0.55)), l1, l2, l3).div(0.55).mul(0.4)),
      0.24,
      1.0
    );

    const warmLanternCol = mix(cRealgarAmber, cYellowOchre, float(0.38));
    const totalDiff = lt1.x.add(lt2.x).add(lt3.x);
    const totalSpec = lt1.y.add(lt2.y).add(lt3.y);

    // Distinctly cool deep Verdigris-Azurite subterranean ambient fill so shadows read emerald-teal, NOT brown soup!
    const coolCisternAmb = mix(cVerdigris.mul(0.20), cAzurite.mul(0.16), float(0.36)).mul(ao);
    // Subtle up-bounced water shimmer onto vault undersides & arch intrados
    const waterBounce = mix(cVerdigris, cCeladon, float(0.45))
      .mul(max(float(0.0), n.y.negate()).mul(0.16))
      .mul(ao);

    const litStone = albedo
      .mul(warmLanternCol.mul(totalDiff).add(coolCisternAmb).add(waterBounce))
      .add(mix(cRealgarAmber, cBoneCalcite, float(0.32)).mul(totalSpec));

    // Incandescent Realgar-Amber + Spore-Sulfur lantern glass with crisp 6-rib brass filigree cage
    // Determine nearest lantern center for local cage coordinates
    const d1Len = length(pHit.sub(l1));
    const d2Len = length(pHit.sub(l2));
    const d3Len = length(pHit.sub(l3));
    const near12 = d1Len.lessThan(d2Len).select(l1, l2);
    const nearLantern = min(d1Len, d2Len).lessThan(d3Len).select(near12, l3);
    const qNear = pHit.sub(nearLantern);
    const cagePhi = safeAtan2(qNear.z, qNear.x);
    const ribMask = smoothstep(float(0.88), float(0.965), abs(cos(cagePhi.mul(3.0))));
    const horizBand = smoothstep(float(0.84), float(0.95), abs(sin(qNear.y.mul(14.0))));
    const cageFiligree = max(ribMask, horizBand.mul(0.65));
    const flameHeart = invSmoothstep(0.0, 0.16, abs(qNear.y.add(0.02)));
    const coreGlow = mix(
      cRealgarAmber.mul(2.85),
      mix(cSporeSulfur.mul(3.6), cBoneCalcite.mul(4.5), flameHeart),
      float(0.55).add(flameHeart.mul(0.35))
    ).mul(pulse);
    const lanternSurface = mix(coreGlow, cRawUmber.mul(0.16), cageFiligree.mul(0.88));

    const combined = mix(litStone, lanternSurface, isLanternGlass);

    // Crisp restrained Verdigris-slate atmospheric depth veil (keeps all 3 receding bays razor-legible!)
    const fogT = float(1.0).sub(exp(distVal.mul(-0.044)));
    const fogCol = mix(cVerdigris.mul(0.115), cAzurite.mul(0.095), float(0.42));
    return mix(combined, fogCol, clamp(fogT, 0.0, 0.86));
  }
);

// ---------------------------------------------------------------------------
// World 3 — West (`-X`): Crimson Sandstone Slot Canyon & Golden-Hour Natural Arch
// ---------------------------------------------------------------------------

/**
 * Exact Lipschitz-normalized SDF matching `evalWestCanyonSdfCpu` — smooth organic flared rims, zero tears!
 */
const westCanyonSdf = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const depthRamp = smoothstep(float(0.0), float(2.2), p.z);

  const floorH = float(-1.78).add(
    depthRamp.mul(
      sin(p.z.mul(0.36)).mul(0.11).add(cos(p.x.mul(0.82).sub(p.z.mul(0.20))).mul(0.055))
    )
  );
  const dFloor = p.y.sub(floorH).mul(0.94);

  // Sweeping water-carved Antelope/Navajo slot-canyon walls flaring smoothly open to the sunset sky
  const centerLineX = depthRamp.mul(sin(p.z.mul(0.21).sub(0.35)).mul(0.54));
  const relX = p.x.sub(centerLineX);
  const strataLedge = sin(p.y.mul(4.0).add(sin(p.z.mul(0.40)).mul(0.42))).mul(0.042);
  const sweepWave = depthRamp.mul(
    cos(p.z.mul(0.26))
      .mul(0.32)
      .add(sin(p.y.mul(0.68).add(p.z.mul(0.20))).mul(0.30))
      .add(cos(p.y.mul(1.12).sub(p.z.mul(0.22))).mul(0.13))
  );
  const mouthOpen = exp(max(float(0.0), p.z).mul(-0.56)).mul(0.66);
  const rimT = max(float(0.0), p.y.sub(1.35));
  const skyFlare = rimT.mul(0.48).add(rimT.mul(rimT).mul(0.28));
  const basinOpen = smoothstep(float(8.2), float(15.0), p.z).mul(2.65);
  const canyonHalfW = float(1.66)
    .add(mouthOpen)
    .add(sweepWave)
    .add(strataLedge)
    .add(skyFlare)
    .add(basinOpen);
  const dWalls = canyonHalfW.sub(abs(relX)).mul(0.66);

  // Monumental natural sandstone rock arch spanning cleanly over the canyon at z = 6.45
  const ax = p.x.sub(0.06);
  const az = p.z.sub(6.45);
  const archCrownY = float(1.44).sub(ax.mul(ax).mul(0.27)).add(sin(ax.mul(1.8)).mul(0.05));
  const archHalfThickY = float(0.36)
    .add(cos(ax.mul(1.4)).mul(0.065))
    .add(sin(az.mul(2.0)).mul(0.025));
  const archHalfThickZ = float(0.58).add(cos(ax.mul(1.1)).mul(0.07));
  const dArchCore = sdBox3Tsl(
    vec3(ax, p.y.sub(archCrownY), az),
    vec3(3.35, archHalfThickY, archHalfThickZ)
  ).sub(0.14);
  const dArch = dArchCore.mul(0.66);

  // Terraced golden-hour sandstone monument buttes & distant mesa ridge framing the vista beyond the arch
  const butteLeft = sdBox3Tsl(vec3(p.x.add(2.55), p.y.add(0.25), p.z.sub(14.8)), vec3(1.35, 2.35, 1.65)).sub(0.32);
  const butteRight = sdBox3Tsl(vec3(p.x.sub(2.95), p.y.sub(0.05), p.z.sub(16.4)), vec3(1.45, 2.75, 1.85)).sub(0.35);
  const mesaRidgeY = float(2.65)
    .add(sin(p.x.mul(0.36).add(0.4)).mul(0.65))
    .add(cos(p.x.mul(0.92)).mul(0.28));
  const dMesaBack = sdBox3Tsl(
    vec3(p.x, p.y.sub(mesaRidgeY.mul(0.5).sub(0.9)), p.z.sub(21.8)),
    vec3(float(12.0), mesaRidgeY.mul(0.5).add(1.1), float(1.8))
  ).sub(0.35);
  const dMesas = min(dMesaBack, min(butteLeft, butteRight));

  const dRock = sminTsl(dWalls, dArch, float(0.38));
  return min(dFloor, min(dRock, dMesas));
});

const westCanyonNormal = Fn(([p_in]: [TslNode]) => {
  const p = vec3(p_in);
  const e = float(0.0042);
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
  const cCinnabar = vec3(...MINERAL_PALETTE.cinnabarVermilion.rInfLinear);
  const cRealgarAmber = vec3(...MINERAL_PALETTE.realgarAmber.rInfLinear);
  const cVerdigris = vec3(...MINERAL_PALETTE.verdigrisMalachite.rInfLinear);
  const cCeladon = vec3(...MINERAL_PALETTE.celadonGlauconite.rInfLinear);
  const cAzurite = vec3(...MINERAL_PALETTE.azuritePrussian.rInfLinear);
  const cLapis = vec3(...MINERAL_PALETTE.lapisUltramarine.rInfLinear);
  const cSporeSulfur = vec3(...MINERAL_PALETTE.sporeSulfur.rInfLinear);

  // ---------------------------------------------------------------------------
  // 1. Honed Volcanic Basalt Exterior Masonry (Tight Value Range + Fine Vesicular Relief)
  // ---------------------------------------------------------------------------
  const matteBasalt = new THREE.MeshStandardNodeMaterial({
    metalness: 0.08,
    roughness: 0.66,
    envMapIntensity: 1.08,
  });
  {
    const fineGrain = mineralToothFbm(positionLocal, float(68.0));
    const mineralTooth = mineralToothFbm(positionLocal.add(vec3(3.1, 5.7, 2.4)), float(142.0));
    const microPitNoise = mineralToothFbm(positionLocal.add(vec3(8.4, 1.9, 6.3)), float(96.0));
    const subtleBed = mineralToothFbm(
      vec3(positionLocal.x.mul(16.0), positionLocal.y.mul(32.0), positionLocal.z.mul(16.0)),
      float(1.0)
    );

    // Tight honed-basalt value range eliminates cloudy low-frequency blotches
    const basaltDeep = vec3(0.036, 0.040, 0.047);
    const basaltHoned = vec3(0.054, 0.058, 0.066);
    const pitMask = invSmoothstep(0.18, 0.30, microPitNoise);
    const feldsparGlint = smoothstep(float(0.80), float(0.96), mineralTooth).mul(0.015);

    const stoneTone = mix(
      basaltDeep,
      basaltHoned,
      smoothstep(float(0.28), float(0.76), fineGrain.mul(0.65).add(subtleBed.mul(0.35)))
    );
    matteBasalt.colorNode = stoneTone
      .mul(float(1.0).sub(pitMask.mul(0.28)))
      .add(cWarmKaolin.mul(feldsparGlint));
    matteBasalt.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(76.0), float(0.036));
    matteBasalt.roughnessNode = clamp(
      float(0.64)
        .add(fineGrain.sub(0.5).mul(0.10))
        .add(pitMask.mul(0.16))
        .sub(feldsparGlint.mul(4.5)),
      0.52,
      0.84
    );
  }

  // ---------------------------------------------------------------------------
  // 2. Darker Honed Basalt Accent Reveals (Recessed Jambs + Compass Bed + Portal Spill Light)
  // ---------------------------------------------------------------------------
  const basaltAccent = new THREE.MeshStandardNodeMaterial({
    metalness: 0.12,
    roughness: 0.54,
    envMapIntensity: 1.15,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(82.0));
    basaltAccent.colorNode = mix(
      vec3(0.022, 0.026, 0.032),
      vec3(0.038, 0.043, 0.050),
      smoothstep(float(0.25), float(0.78), grain)
    );
    basaltAccent.normalNode = perturbNormal3DFbm(normalLocal, positionLocal, float(88.0), float(0.022));
    basaltAccent.roughnessNode = clamp(float(0.52).add(grain.sub(0.5).mul(0.09)), 0.42, 0.65);
    // Subtle physically-grounded portal-world bounce onto recessed stone window cheeks
    basaltAccent.emissiveNode = computeRecessedPortalSpill(positionLocal, normalLocal, uTime, float(0.068));
  }

  // ---------------------------------------------------------------------------
  // 3. Precision-Machined Brushed Satin Brass (Directional Micro-Brushing + Inner Rim Spill)
  // ---------------------------------------------------------------------------
  const satinBrass = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.21,
    envMapIntensity: 1.38,
  });
  {
    // Fine anisotropic-feeling machined brushing streaks (tight reflectance -> zero cloudy spots)
    const brushCoord = vec3(
      positionLocal.x.mul(180.0),
      positionLocal.y.mul(42.0),
      positionLocal.z.mul(180.0)
    );
    const fineBrush = mineralToothFbm(brushCoord, float(1.0));
    const microSatin = mineralToothFbm(positionLocal.add(vec3(4.3, 2.1, 6.7)), float(155.0));
    const brassAged = vec3(0.62, 0.48, 0.22);
    const brassBright = vec3(0.80, 0.65, 0.32);

    satinBrass.colorNode = mix(
      brassAged,
      brassBright,
      smoothstep(float(0.26), float(0.76), fineBrush.mul(0.55).add(microSatin.mul(0.45)))
    );
    satinBrass.normalNode = perturbNormal3DFbm(normalLocal, brushCoord.mul(0.45), float(2.4), float(0.0045));
    satinBrass.roughnessNode = clamp(
      float(0.20).add(fineBrush.sub(0.5).mul(0.05)).add(microSatin.sub(0.5).mul(0.025)),
      0.16,
      0.27
    );
    // Subtle warm/cool portal light kissing only the inner recessed brass window bore
    satinBrass.emissiveNode = computeRecessedPortalSpill(positionLocal, normalLocal, uTime, float(0.085));
  }

  // ---------------------------------------------------------------------------
  // 4. Patinated Dark-Brass Fasteners & Compass Center Pin
  // ---------------------------------------------------------------------------
  const darkBrassFastener = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.30,
    envMapIntensity: 1.22,
  });
  {
    const grain = mineralToothFbm(positionLocal, float(92.0));
    darkBrassFastener.colorNode = mix(
      vec3(0.26, 0.20, 0.10),
      vec3(0.44, 0.34, 0.16),
      smoothstep(float(0.25), float(0.75), grain)
    );
    darkBrassFastener.roughnessNode = clamp(float(0.30).add(grain.sub(0.5).mul(0.06)), 0.24, 0.38);
  }

  // ===========================================================================
  // PORTAL WORLD 0 — NORTH (+Z): SUNLIT MEDITERRANEAN TRAVERTINE COLONNADE
  // ===========================================================================
  const portalNorth = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalNorth.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('north', uCamLocal);
    const hoverW = uHoverWeights.north;
    const ptr = uPointerLocal.north;

    // Crisp bright Mediterranean morning sun slanting cleanly from upper-right across the arcade
    const sunShiftX = ptr.x.mul(hoverW).mul(0.24).add(sin(uTime.mul(0.24)).mul(0.025));
    const sunShiftY = ptr.y.mul(hoverW).mul(0.12);
    const sunDir = normalize(vec3(float(0.68).add(sunShiftX), float(0.64).add(sunShiftY), float(-0.30)));

    // Analytical floor plane guard prevents grazing-ray step exhaustion across deep inlaid colonnade floor!
    const floorY = float(-1.76);
    const tFloorRaw = floorY.sub(ro.y).div(rd.y.lessThan(-1e-4).select(rd.y, float(-1.0)));
    const hitsFloorPlane = rd.y
      .lessThan(-1e-4)
      .and(tFloorRaw.greaterThan(0.01))
      .and(tFloorRaw.lessThan(34.0));
    const tFloor = hitsFloorPlane.select(tFloorRaw, float(100.0));

    const t = float(0.02).toVar();
    const hit = float(0.0).toVar();
    const maxDist = min(float(34.0), tFloor.add(0.02));

    Loop(72, () => {
      const p = ro.add(rd.mul(t));
      const d = northColonnadeSdf(p);
      If(d.lessThan(0.004), () => {
        hit.assign(1.0);
        Break();
      });
      t.addAssign(max(float(0.004), d.mul(0.88)));
      If(t.greaterThan(maxDist), () => {
        Break();
      });
    });

    // If a grazing floor ray reached the analytical floor plane without intersecting a column first, lock cleanly onto the floor!
    If(hit.lessThan(0.5).and(hitsFloorPlane).and(t.greaterThan(tFloor.sub(0.35))), () => {
      t.assign(tFloor);
      hit.assign(1.0);
    });

    // Crisp Mediterranean morning sky: radiant ivory-white horizon rising to airy pale cerulean-Azurite zenith
    // (Strong daylight cool-sky contrast vs the crimson-amber sunset of West canyon!)
    const skyElev = clamp(rd.y.mul(1.25).add(0.18), 0.0, 1.0);
    const sunAlign = max(float(0.0), dot(rd, sunDir));
    const paleCerulean = mix(cBoneCalcite.mul(0.92), cAzurite.mul(2.15), float(0.46));
    const horizonBright = mix(cBoneCalcite.mul(1.22), cWarmKaolin.mul(1.06), float(0.16));
    const skyCol = mix(horizonBright, paleCerulean, pow(skyElev, 0.72))
      .add(cBoneCalcite.mul(pow(sunAlign, 12.0).mul(0.78)));

    const finalCol = skyCol.toVar();

    If(hit.greaterThan(0.5), () => {
      const pHit = ro.add(rd.mul(t));
      const nGeom = northColonnadeNormal(pHit);

      const grain = mineralToothFbm(pHit, float(3.2));
      const finePore = mineralToothFbm(pHit.add(vec3(4.1, 7.3, 2.2)), float(13.5));

      const isFloor = invSmoothstep(-1.755, -1.70, pHit.y).mul(
        smoothstep(float(0.72), float(0.95), nGeom.y)
      );
      const isThresholdStep = invSmoothstep(0.45, 0.52, pHit.z).mul(
        invSmoothstep(-1.71, -1.66, pHit.y)
      );

      // Architectural inlaid travertine courtyard floor:
      // Wide central Bone-Calcite flagstone slabs framed by dark Raw-Umber runner borders & Ochre medallions
      const absX = abs(pHit.x);
      const borderBand = smoothstep(float(1.10), float(1.14), absX).mul(
        invSmoothstep(1.24, 1.28, absX)
      );
      const tileScale = float(0.74);
      const uT = fract(pHit.x.mul(tileScale).add(1024.0));
      const vT = fract(pHit.z.mul(tileScale).add(1024.0));
      const tileU = floor(pHit.x.mul(tileScale).add(1024.0));
      const tileV = floor(pHit.z.mul(tileScale).add(1024.0));
      const checker = fract(tileU.add(tileV).mul(0.5).add(1024.0)).mul(2.0);
      const fU = abs(uT.sub(0.5));
      const fV = abs(vT.sub(0.5));
      const grout = smoothstep(float(0.468), float(0.495), max(fU, fV));
      // Small inlaid Yellow-Ochre diamond cabochons at slab corners
      const cornerDist = length(vec2(float(0.5).sub(fU), float(0.5).sub(fV)));
      const diamondInlay = invSmoothstep(0.085, 0.115, cornerDist);

      const brightMarble = mix(cBoneCalcite.mul(1.04), cWarmKaolin.mul(0.94), grain.mul(0.45));
      const ivoryMarble = mix(cWarmKaolin.mul(0.88), cBoneCalcite.mul(0.96), finePore.mul(0.5));
      const slabBase = mix(brightMarble, ivoryMarble, smoothstep(float(0.3), float(0.7), checker));
      const withDiamond = mix(slabBase, cYellowOchre.mul(0.86), diamondInlay.mul(0.82));
      const withRunner = mix(withDiamond, cRawUmber.mul(0.48), borderBand.mul(0.75));
      const floorAlbedo = mix(withRunner, cRawUmber.mul(0.36), grout.mul(0.82));

      // Honed Roman travertine columns, arcades, niches & triumphal arch with sedimentary banding
      const veinCoord = pHit.y.mul(7.2).add(grain.mul(2.4)).add(sin(pHit.z.mul(0.8)).mul(0.35));
      const band = sin(veinCoord).mul(0.5).add(0.5);
      const ashlarY = abs(fract(pHit.y.mul(1.45).add(1024.0)).sub(0.5));
      const stoneJoint = smoothstep(float(0.475), float(0.496), ashlarY).mul(
        smoothstep(float(1.65), float(1.9), abs(pHit.x))
      );
      const travertineBase = mix(
        cWarmKaolin.mul(0.92),
        cBoneCalcite.mul(1.08),
        band.mul(0.48).add(finePore.mul(0.52))
      );
      const leftWallTint = mix(
        travertineBase,
        mix(cWarmKaolin.mul(0.88), cYellowOchre.mul(0.82), float(0.28)),
        invSmoothstep(-4.42, -4.05, pHit.x)
      );
      const wallAlbedo = mix(leftWallTint, cRawUmber.mul(0.45), stoneJoint.mul(0.45));

      const thresholdAlbedo = mix(cBoneCalcite.mul(0.98), cWarmKaolin.mul(0.90), grain.mul(0.5));
      const albedo = mix(mix(wallAlbedo, floorAlbedo, isFloor), thresholdAlbedo, isThresholdStep);

      // Micro-normal travertine surface tooth + crisp inlaid floor grout bevels
      const nMicro = applyPortalMicroNormal(nGeom, pHit, float(6.5), float(0.055));
      const groutSlope = vec3(
        sin(uT.mul(float(Math.PI * 2.0))).mul(smoothstep(float(0.44), float(0.495), fU)).mul(0.18),
        float(0.0),
        sin(vT.mul(float(Math.PI * 2.0))).mul(smoothstep(float(0.44), float(0.495), fV)).mul(0.18)
      );
      const n = normalize(nMicro.add(groutSlope.mul(isFloor)));

      // 16-step crisp geometric soft shadows from Mediterranean sun through arcade bays
      const sh = float(1.0).toVar();
      const ts = float(0.05).toVar();
      Loop(16, () => {
        const hs = northColonnadeSdf(pHit.add(nGeom.mul(0.014)).add(sunDir.mul(ts)));
        sh.assign(min(sh, clamp(hs.mul(9.5).div(ts), float(0.0), float(1.0))));
        ts.addAssign(clamp(hs, float(0.06), float(0.75)));
        If(sh.lessThan(0.012).or(ts.greaterThan(12.5)), () => {
          Break();
        });
      });

      // 2-tap analytical ambient occlusion carving crisp column flute & cornice creases
      const ao = clamp(
        northColonnadeSdf(pHit.add(nGeom.mul(0.16)))
          .div(0.16)
          .mul(0.58)
          .add(northColonnadeSdf(pHit.add(nGeom.mul(0.48))).div(0.48).mul(0.42)),
        0.22,
        1.0
      );

      const nDotL = max(float(0.0), dot(n, sunDir));
      const sunCol = mix(cBoneCalcite, cYellowOchre, float(0.16)).mul(2.35);

      // Distinctively airy Mediterranean cool-sky shadow fill + clean sunlit marble bounce
      const coolSkyFill = mix(cBoneCalcite.mul(0.55), cAzurite.mul(0.95), float(0.52))
        .mul(float(0.24).add(max(float(0.0), n.y.mul(0.5).add(0.5)).mul(0.22)));
      const warmMarbleBounce = mix(cBoneCalcite, cYellowOchre, float(0.32))
        .mul(float(0.16).add(max(float(0.0), n.y.negate()).mul(0.16)));

      // Honed marble specular reflection on floor slabs
      const halfV = normalize(sunDir.sub(rd));
      const spec = pow(max(float(0.0), dot(n, halfV)), 38.0)
        .mul(isFloor.mul(0.38).add(0.07))
        .mul(sh);

      const lit = albedo
        .mul(sunCol.mul(nDotL.mul(sh)).add(coolSkyFill.add(warmMarbleBounce).mul(ao)))
        .add(cBoneCalcite.mul(spec));

      // Light, luminous airy Mediterranean perspective haze (keeps triumphal arch & distant sunlit rotunda crisp!)
      const fogT = float(1.0).sub(exp(t.mul(-0.026)));
      const hazeCol = mix(horizonBright.mul(0.94), paleCerulean.mul(0.96), float(0.38));
      finalCol.assign(mix(lit, hazeCol, clamp(fogT, 0.0, 0.72)));
    });

    // 6-tap volumetric morning sunbeams slanting cleanly through the side arcade arches
    const shaftAccum = float(0.0).toVar();
    const maxShaftT = min(t, float(18.0));
    for (let s = 0; s < 6; s++) {
      const fracS = (s + 0.42) / 6.0;
      const pBeam = ro.add(rd.mul(maxShaftT.mul(fracS)));
      const zProj = pBeam.z.sub(pBeam.x.mul(sunDir.z.div(max(float(0.18), sunDir.x))));
      const bayWave = sin(zProj.sub(0.5).mul(float((Math.PI * 2) / 2.7))).mul(0.5).add(0.5);
      const inGallery = invSmoothstep(0.35, 2.35, abs(pBeam.x)).mul(
        smoothstep(float(-1.75), float(0.6), pBeam.y)
      );
      const sunClear = smoothstep(
        float(0.16),
        float(0.72),
        northColonnadeSdf(pBeam.add(sunDir.mul(1.85)))
      );
      // Subtle floating sunlit dust mote sparkle inside active sunbeams
      const moteNoise = mineralToothFbm(
        pBeam.add(vec3(0.0, uTime.mul(-0.08), uTime.mul(0.04))),
        float(2.4)
      );
      const moteLift = float(0.85).add(smoothstep(float(0.62), float(0.92), moteNoise).mul(0.55));
      shaftAccum.addAssign(bayWave.mul(inGallery).mul(sunClear).mul(moteLift).mul(0.032));
    }
    const shaftWarmth = mix(cBoneCalcite, cYellowOchre, float(0.32)).mul(
      shaftAccum.mul(float(1.0).add(hoverW.mul(0.28)))
    );

    // Distance-gated reveal shadow: kisses foreground floor/step right inside frame without darkening distant sky
    const nearFrameAo = mix(float(1.0), edgeAo, exp(t.mul(-0.85)));
    return toneMapMineral(finalCol.add(shaftWarmth).mul(nearFrameAo));
  })();

  // ===========================================================================
  // PORTAL WORLD 1 — EAST (+X): PLUNGING MIDNIGHT LAPIS SPIRAL STAIRWELL
  // ===========================================================================
  const portalEast = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalEast.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('east', uCamLocal);
    const hoverW = uHoverWeights.east;
    const ptr = uPointerLocal.east;

    const t = float(0.02).toVar();
    const hit = float(0.0).toVar();
    const maxDist = float(32.0);

    Loop(68, () => {
      const p = ro.add(rd.mul(t));
      const d = eastStairwellSdf(p);
      If(d.lessThan(0.004), () => {
        hit.assign(1.0);
        Break();
      });
      t.addAssign(max(float(0.004), d.mul(0.78)));
      If(t.greaterThan(maxDist), () => {
        Break();
      });
    });

    // Deep midnight-lapis abyss mist below + cool luminous moonlight/daylight oculus overhead
    const abyssFog = mix(cLapis.mul(0.18), cAzurite.mul(0.28), float(0.45).add(hoverW.mul(0.22)));
    const oculusGlow = mix(cBoneCalcite, cCeladon.mul(1.15), float(0.24))
      .mul(smoothstep(float(0.32), float(0.94), rd.y).mul(1.35));
    const finalCol = abyssFog.add(oculusGlow).toVar();

    If(hit.greaterThan(0.5), () => {
      const pHit = ro.add(rd.mul(t));
      const nGeom = eastStairwellNormal(pHit);
      const qx = pHit.x;
      const qz = pHit.z.sub(2.4);
      const r = length(vec2(qx, qz));
      const theta = safeAtan2(qz, qx);

      // Exact SDF component discrimination (eliminates blue color-bleed onto step ends & balusters!)
      const parts = eastStairwellParts(pHit);
      const dStone = parts.y;
      const dBrass = parts.z;
      const isBrass = invSmoothstep(0.008, 0.020, dBrass).mul(
        dBrass.lessThan(dStone.add(0.004)).select(float(1.0), float(0.0))
      );
      const isStairOrLanding = invSmoothstep(0.008, 0.022, dStone).mul(float(1.0).sub(isBrass));
      const isOuterWall = float(1.0).sub(max(isBrass, isStairOrLanding));

      const grain = mineralToothFbm(pHit, float(3.8));
      const fineGrain = mineralToothFbm(pHit.add(vec3(3.4, 8.1, 1.9)), float(12.5));

      // Glazed Lapis-Ultramarine & Azurite-Prussian curved ceramic masonry tiles on cylindrical shaft
      const tileAng = theta.mul(16.0 / (Math.PI * 2)).add(1024.0);
      const tileRow = pHit.y.mul(2.15).add(1024.0);
      const rowShift = fract(floor(tileRow).mul(0.5).add(1024.0));
      const uTile = fract(tileAng.add(rowShift).add(1024.0));
      const vTile = fract(tileRow);
      const mU = smoothstep(float(0.452), float(0.492), abs(uTile.sub(0.5)));
      const mV = smoothstep(float(0.445), float(0.490), abs(vTile.sub(0.5)));
      const mortar = max(mU, mV);

      // Detect carved Bone-Calcite cornice rings & vertical pilasters on the cylindrical shaft
      const hTurn = float(4.8);
      const ringRaw = pHit.y.add(0.72);
      const ringY = abs(ringRaw.sub(floor(ringRaw.div(hTurn).add(0.5)).mul(hTurn)));
      const isCorniceRing = invSmoothstep(0.068, 0.095, ringY).mul(invSmoothstep(2.26, 2.33, r));

      const tileVar = fract(
        abs(sin(floor(tileAng).mul(12.9898).add(floor(tileRow).mul(78.233))).mul(43758.5453))
      );
      const kilnLapis = mix(
        cLapis.mul(0.88),
        cAzurite.mul(1.16),
        tileVar.mul(0.58).add(grain.mul(0.42))
      );
      // Subtle Celadon glaze shimmer on select fired ceramic tiles
      const glazedTile = mix(
        kilnLapis,
        cCeladon.mul(0.55),
        smoothstep(float(0.72), float(0.95), tileVar).mul(0.28)
      );
      const wallAlbedo = mix(
        mix(glazedTile, cBoneCalcite.mul(0.52), mortar.mul(0.48)),
        mix(cBoneCalcite.mul(0.98), cWarmKaolin.mul(0.90), grain.mul(0.45)),
        isCorniceRing
      );

      // Crisp honed Bone-Calcite cantilevered stair treads with recessed satin-brass nosing strip
      const treadStone = mix(
        cWarmKaolin.mul(0.88),
        cBoneCalcite.mul(1.06),
        fineGrain.mul(0.45).add(max(float(0.0), nGeom.y).mul(0.45))
      );
      const brassNosing = invSmoothstep(1.14, 1.24, r).mul(smoothstep(float(0.65), float(0.92), nGeom.y));
      const stairAlbedo = mix(treadStone, cYellowOchre.mul(1.12), brassNosing.mul(0.72));

      // Brushed satin-brass helical handrail & balusters
      const brassAlbedo = mix(cYellowOchre.mul(1.14), cBoneCalcite.mul(0.96), float(0.26));

      const albedo = mix(mix(wallAlbedo, stairAlbedo, isStairOrLanding), brassAlbedo, isBrass);

      // Crisp normal relief: recessed curved ceramic tile mortar grooves (keep stone treads clean & honed matte!)
      const tanDir = normalize(vec3(qz.negate(), float(0.0), qx));
      const tileGrooveN = tanDir
        .mul(sin(uTile.mul(float(Math.PI * 2.0))).mul(mU).mul(0.24))
        .add(vec3(float(0.0), sin(vTile.mul(float(Math.PI * 2.0))).mul(mV).mul(0.26), float(0.0)));
      const nMicro = applyPortalMicroNormal(nGeom, pHit, float(8.5), mix(float(0.042), float(0.016), isStairOrLanding));
      const n = normalize(
        mix(nMicro.add(tileGrooveN.mul(isOuterWall.mul(float(1.0).sub(isCorniceRing)))), nGeom, isBrass)
      );

      // Overhead cool-silver zenith oculus skylight pouring down the central cylindrical open well
      const oculusPos = vec3(
        ptr.x.mul(hoverW).mul(0.55),
        float(9.8),
        float(2.4).add(ptr.y.mul(hoverW).mul(0.55))
      );
      const toOculus = oculusPos.sub(pHit);
      const oculusDist = max(float(1.0), length(toOculus));
      const lightDir = toOculus.div(oculusDist);

      // 12-step soft shadow march toward zenith oculus (every cantilevered tread & baluster casts crisp shadows!)
      const sh = float(1.0).toVar();
      const ts = float(0.07).toVar();
      Loop(12, () => {
        const hs = eastStairwellSdf(pHit.add(nGeom.mul(0.018)).add(lightDir.mul(ts)));
        sh.assign(min(sh, clamp(hs.mul(6.2).div(ts), float(0.0), float(1.0))));
        ts.addAssign(clamp(hs, float(0.08), float(0.88)));
        If(sh.lessThan(0.015).or(ts.greaterThan(min(oculusDist, float(11.0)))), () => {
          Break();
        });
      });

      const ao = clamp(
        eastStairwellSdf(pHit.add(nGeom.mul(0.15)))
          .div(0.15)
          .mul(0.60)
          .add(eastStairwellSdf(pHit.add(nGeom.mul(0.42))).div(0.42).mul(0.40)),
        0.20,
        1.0
      );

      // Crisp zenith skylight falloff + cool luminous subterranean sapphire-celadon sanctuary glow rising below!
      const zenithAtten = exp(max(float(0.0), float(1.8).sub(pHit.y)).mul(-0.072));
      const nDotL = max(float(0.0), dot(n, lightDir));
      const oculusCol = mix(cBoneCalcite, cCeladon.mul(1.18), float(0.18)).mul(2.35).mul(zenithAtten);

      // Deep subterranean mineral pool up-glow rising from the abyss below (draws the eye downward!)
      const depthLift = invSmoothstep(-15.0, 1.5, pHit.y);
      const deepPulse = float(0.46).add(hoverW.mul(0.32)).add(sin(uTime.mul(0.85)).mul(0.045));
      const deepGlowCol = mix(cAzurite, cCeladon.mul(1.18), float(0.48).add(hoverW.mul(0.18)))
        .mul(deepPulse)
        .mul(float(0.42).add(max(float(0.0), n.y.negate()).mul(0.58)))
        .mul(float(0.35).add(depthLift.mul(1.05)));

      // Delicate warm recessed wall-sconce glow spiraling down deeper turns (invites curiosity downward!)
      const sconcePhase = abs(fract(pHit.y.div(4.8).add(0.48).add(1024.0)).sub(0.5));
      const sconceAngDist = abs(sin(theta.mul(2.0).add(0.6)));
      const sconceProx = invSmoothstep(0.02, 0.14, sconcePhase).mul(invSmoothstep(0.05, 0.35, sconceAngDist));
      const sconceWarmth = mix(cRealgarAmber, cYellowOchre, float(0.4))
        .mul(sconceProx.mul(0.62).mul(invSmoothstep(-12.0, -0.5, pHit.y)));

      // Crisp specular highlight on glazed cobalt tiles, subtle honed limestone treads, and gleaming brass handrail
      const halfV = normalize(lightDir.sub(rd));
      const specExp = mix(mix(float(40.0), float(34.0), isStairOrLanding), float(64.0), isBrass);
      const specGain = mix(
        mix(float(0.54).mul(float(1.0).sub(mortar)), float(0.08), isStairOrLanding),
        float(1.55),
        isBrass
      );
      const specCol = mix(cBoneCalcite, cYellowOchre, isBrass).mul(
        pow(max(float(0.0), dot(n, halfV)), specExp).mul(specGain).mul(float(0.35).add(sh.mul(0.65)))
      );

      const ambientLapis = mix(cLapis.mul(0.24), cAzurite.mul(0.20), float(0.42)).mul(ao);
      const lit = albedo
        .mul(oculusCol.mul(nDotL.mul(sh)).add(ambientLapis).add(deepGlowCol).add(sconceWarmth))
        .add(specCol);

      // Vertical plunge depth haze selling the impossible subterranean shaft below the cube
      const plungeDepth = max(float(0.0), float(-1.8).sub(pHit.y));
      const fogAmt = float(1.0).sub(exp(t.mul(-0.042).sub(plungeDepth.mul(0.052))));
      const fogCol = mix(
        cLapis.mul(0.24),
        mix(cAzurite.mul(0.38), cCeladon.mul(0.44), float(0.36).add(hoverW.mul(0.25))),
        clamp(plungeDepth.div(13.0), 0.0, 1.0)
      );
      finalCol.assign(mix(lit, fogCol, clamp(fogAmt, 0.0, 0.90)));
    });

    // Subtle cool central oculus skylight beam cutting down through the hollow spiral center
    const beamAccum = float(0.0).toVar();
    const maxBeamT = min(t, float(10.0));
    for (let s = 0; s < 4; s++) {
      const fS = (s + 0.4) / 4.0;
      const pB = ro.add(rd.mul(maxBeamT.mul(fS)));
      const rB = length(vec2(pB.x, pB.z.sub(2.4)));
      const inOculus = invSmoothstep(0.15, 1.15, rB);
      const heightFade = smoothstep(float(-9.0), float(2.5), pB.y);
      beamAccum.addAssign(inOculus.mul(heightFade).mul(0.026));
    }
    const oculusBeam = mix(cBoneCalcite, cCeladon, float(0.38)).mul(
      beamAccum.mul(float(1.0).add(hoverW.mul(0.3)))
    );

    const nearFrameAo = mix(float(1.0), edgeAo, exp(t.mul(-0.85)));
    return toneMapMineral(finalCol.add(oculusBeam).mul(nearFrameAo));
  })();

  // ===========================================================================
  // PORTAL WORLD 2 — SOUTH (-Z): VERDIGRIS BASILICA CISTERN & RECEDING LANTERNS
  // ===========================================================================
  const portalSouth = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalSouth.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('south', uCamLocal);
    const hoverW = uHoverWeights.south;
    const ptr = uPointerLocal.south;

    // Three staggered swaying Realgar-Amber filigree brass lanterns receding cleanly in perspective down the nave
    const swayX1 = sin(uTime.mul(1.15)).mul(0.10).add(ptr.x.mul(hoverW).mul(0.16));
    const swayY1 = cos(uTime.mul(2.3)).mul(0.012);
    const swayZ1 = cos(uTime.mul(0.85)).mul(0.045);
    const l1Pos = vec3(swayX1, float(0.15).add(swayY1), float(3.45).add(swayZ1));
    const l2Pos = vec3(float(-0.24).add(sin(uTime.mul(0.82).add(1.7)).mul(0.06)), float(0.46), float(9.65));
    const l3Pos = vec3(float(0.22).add(sin(uTime.mul(0.65).add(3.1)).mul(0.045)), float(0.72), float(15.85));

    // Organic flickering oil-lantern flame cadence
    const flamePulse = float(1.0)
      .add(sin(uTime.mul(6.4)).mul(0.048))
      .add(cos(uTime.mul(11.3)).mul(0.026));

    // Analytical mirror water plane (y = -1.76, seated cleanly below portal sill)
    const waterY = float(-1.76);
    const tWaterRaw = waterY.sub(ro.y).div(rd.y.lessThan(-1e-4).select(rd.y, float(-1.0)));
    const hitsWaterPlane = rd.y
      .lessThan(-1e-4)
      .and(tWaterRaw.greaterThan(0.005))
      .and(tWaterRaw.lessThan(25.0));
    const tWater = hitsWaterPlane.select(tWaterRaw, float(100.0));

    // Primary SDF march for vaulted stone cistern, arcade piers, vault ribs, stone quay & 3 lanterns
    const tSolid = float(0.02).toVar();
    const hitSolid = float(0.0).toVar();
    const maxDist = min(float(25.5), tWater.add(0.06));

    Loop(60, () => {
      const p = ro.add(rd.mul(tSolid));
      const d = southCisternSolidSdf(p, l1Pos, l2Pos, l3Pos);
      If(d.lessThan(0.0045), () => {
        hitSolid.assign(1.0);
        Break();
      });
      tSolid.addAssign(max(float(0.0045), d.mul(0.84)));
      If(tSolid.greaterThan(maxDist), () => {
        Break();
      });
    });

    const fogFloor = mix(cVerdigris.mul(0.11), cAzurite.mul(0.09), float(0.42));
    const finalCol = fogFloor.toVar();
    const tVisual = min(tSolid, tWater).toVar();

    const waterIsNearest = hitsWaterPlane.and(tWater.lessThan(tSolid));

    If(waterIsNearest, () => {
      const pW = ro.add(rd.mul(tWater));

      // Fine crisp capillary water ripples + smooth concentric pointer-hover ripples
      // Restrained slope amplitude keeps upside-down vaulted arches & lanterns razor-legible!
      const rippleCenter = vec2(ptr.x.mul(1.35), float(3.8).add(ptr.y.mul(2.1)));
      const dRipVec = vec2(pW.x, pW.z).sub(rippleCenter);
      const dRip = length(dRipVec);
      const ripEnv = exp(dRip.mul(-0.58)).mul(float(0.10).add(hoverW.mul(0.85)));
      const ripGrad = cos(dRip.mul(8.8).sub(uTime.mul(4.6))).mul(ripEnv).mul(0.030);
      const ripDir = dRipVec.div(max(float(0.08), dRip));

      const ambGx = cos(pW.x.mul(3.4).add(pW.z.mul(1.3)).add(uTime.mul(1.1)))
        .mul(0.0065)
        .add(sin(pW.z.mul(5.8).sub(pW.x.mul(2.1)).sub(uTime.mul(1.5))).mul(0.0045));
      const ambGz = sin(pW.z.mul(4.2).sub(pW.x.mul(1.6)).add(uTime.mul(0.95)))
        .mul(0.0075)
        .add(cos(pW.x.mul(5.1).add(pW.z.mul(3.4)).sub(uTime.mul(1.35))).mul(0.0045));

      const nWater = normalize(
        vec3(
          ambGx.add(ripDir.x.mul(ripGrad)).negate(),
          float(1.0),
          ambGz.add(ripDir.y.mul(ripGrad)).negate()
        )
      );

      // Physical dielectric Fresnel reflection of vaulted basilica nave & all 3 glowing lanterns
      const ndotv = clamp(dot(nWater, rd.negate()), 0.0, 1.0);
      const fresnel = float(0.045).add(
        float(0.92).mul(pow(max(float(0.0), float(1.0).sub(ndotv)), 4.2))
      );

      const rdRefl = normalize(reflect(rd, nWater));
      const tR = float(0.05).toVar();
      const hitR = float(0.0).toVar();
      Loop(28, () => {
        const pR = pW.add(rdRefl.mul(tR));
        const dR = southCisternSolidSdf(pR, l1Pos, l2Pos, l3Pos);
        If(dR.lessThan(0.014), () => {
          hitR.assign(1.0);
          Break();
        });
        tR.addAssign(max(float(0.020), dR.mul(0.90)));
        If(tR.greaterThan(20.0), () => {
          Break();
        });
      });

      const reflCol = fogFloor.toVar();
      If(hitR.greaterThan(0.5), () => {
        reflCol.assign(
          shadeSouthCisternSolid(
            pW.add(rdRefl.mul(tR)),
            rdRefl,
            tR.add(tWater.mul(0.35)),
            l1Pos,
            l2Pos,
            l3Pos,
            flamePulse
          )
        );
      });

      // Crisp broken lantern glints dancing across fine water ripples (NO broad muddy orange blob!)
      const evalWaterGlint = (lPos: TslNode, sharpPow: TslNode, gain: TslNode) => {
        const toL = vec3(lPos).sub(pW);
        const lDist = length(toL);
        const hW = normalize(normalize(toL).sub(rd));
        const atten = float(1.0).div(float(0.6).add(lDist.mul(lDist).mul(0.11)));
        return pow(max(float(0.0), dot(nWater, hW)), sharpPow).mul(gain).mul(atten);
      };

      const g1 = evalWaterGlint(l1Pos, float(130.0), float(2.85));
      const g2 = evalWaterGlint(l2Pos, float(110.0), float(1.65));
      const g3 = evalWaterGlint(l3Pos, float(95.0), float(1.15));
      const shimmerCol = mix(cRealgarAmber, cSporeSulfur, float(0.42))
        .mul(g1.add(g2).add(g3))
        .mul(flamePulse);

      // Deep dark emerald-verdigris cistern water body + delicate flickering underwater caustics near front quay
      const cWave = sin(pW.x.mul(5.2).add(uTime.mul(1.4))).mul(cos(pW.z.mul(4.8).sub(uTime.mul(1.1))));
      const causticNet = pow(max(float(0.0), float(1.0).sub(abs(cWave))), 5.0);
      const nearQuayFade = exp(tWater.mul(-0.28));
      const underWater = mix(cAzurite.mul(0.045), cVerdigris.mul(0.145), float(0.66))
        .add(mix(cCeladon, cRealgarAmber, float(0.45)).mul(causticNet.mul(0.11).mul(nearQuayFade)));

      const waterFog = float(1.0).sub(exp(tWater.mul(-0.042)));
      const waterShaded = mix(underWater, reflCol, clamp(fresnel.mul(1.18), 0.16, 0.96)).add(shimmerCol);
      finalCol.assign(mix(waterShaded, fogFloor, clamp(waterFog, 0.0, 0.85)));
    }).Else(() => {
      If(hitSolid.greaterThan(0.5), () => {
        finalCol.assign(
          shadeSouthCisternSolid(
            ro.add(rd.mul(tSolid)),
            rd,
            tSolid,
            l1Pos,
            l2Pos,
            l3Pos,
            flamePulse
          )
        );
      });
    });

    // Tight atmospheric candlelight halos around all three suspended lanterns (never washes out vaults!)
    const evalHalo = (lPos: TslNode, sharpK: TslNode, gain: TslNode) => {
      const toRay = vec3(lPos).sub(ro);
      const tClose = clamp(dot(toRay, rd), float(0.0), tVisual);
      const diff = ro.add(rd.mul(tClose)).sub(vec3(lPos));
      const dSq = dot(diff, diff);
      return exp(dSq.mul(sharpK.negate())).mul(gain);
    };

    const haloSum = evalHalo(l1Pos, float(4.2), float(0.32))
      .add(evalHalo(l1Pos, float(0.85), float(0.055)))
      .add(evalHalo(l2Pos, float(3.2), float(0.24)))
      .add(evalHalo(l3Pos, float(2.6), float(0.18)));

    const lanternHalo = mix(cRealgarAmber, cYellowOchre, float(0.34)).mul(haloSum.mul(flamePulse));
    const nearFrameAo = mix(float(1.0), edgeAo, exp(tVisual.mul(-0.85)));
    return toneMapMineral(finalCol.add(lanternHalo).mul(nearFrameAo));
  })();

  // ===========================================================================
  // PORTAL WORLD 3 — WEST (-X): CRIMSON SANDSTONE SLOT CANYON & SUNSET ARCH
  // ===========================================================================
  const portalWest = new THREE.MeshBasicNodeMaterial({ side: THREE.FrontSide });
  portalWest.colorNode = Fn(() => {
    const { ro, rd, edgeAo } = buildPortalLocalRayNodes('west', uCamLocal);
    const hoverW = uHoverWeights.west;
    const ptr = uPointerLocal.west;

    // Dramatic golden-hour desert sun pouring down through the organic flared canyon slot & across the arch!
    const sunShiftX = ptr.x.mul(hoverW).mul(0.24).add(sin(uTime.mul(0.22)).mul(0.025));
    const sunShiftY = ptr.y.mul(hoverW).mul(0.10);
    const sunDir = normalize(vec3(float(-0.32).add(sunShiftX), float(0.80).add(sunShiftY), float(0.50)));

    const t = float(0.02).toVar();
    const hit = float(0.0).toVar();
    const maxDist = float(28.0);

    // Distance-proportional cone epsilon eliminates 1px grazing-silhouette step exhaustion cleanly!
    Loop(96, () => {
      const p = ro.add(rd.mul(t));
      const d = westCanyonSdf(p);
      const hitEps = max(float(0.004), t.mul(0.0016));
      If(d.lessThan(hitEps), () => {
        hit.assign(1.0);
        Break();
      });
      t.addAssign(max(hitEps, d.mul(0.92)));
      If(t.greaterThan(maxDist), () => {
        Break();
      });
    });

    // Dramatic Golden-Hour / Desert Sunset Sky:
    // Blazing Realgar-Amber & Yellow-Ochre horizon rising into rich twilight Prussian-Azurite above the arch!
    const skyT = clamp(rd.y.mul(2.15).add(0.06), 0.0, 1.0);
    const sunAlign = max(float(0.0), dot(rd, sunDir));
    const sunsetHorizon = mix(cRealgarAmber.mul(1.22), cYellowOchre.mul(1.20), float(0.45));
    const sunsetMid = mix(cCinnabar.mul(0.80), cWarmKaolin.mul(0.84), float(0.36));
    const twilightZenith = mix(cAzurite.mul(0.76), cLapis.mul(0.68), float(0.48));
    const lowerSky = mix(sunsetHorizon, sunsetMid, smoothstep(float(0.0), float(0.40), skyT));
    const desertSky = mix(lowerSky, twilightZenith, smoothstep(float(0.28), float(0.84), skyT))
      .add(mix(cSporeSulfur, cBoneCalcite, float(0.58)).mul(pow(sunAlign, 12.0).mul(0.92)));

    const finalCol = desertSky.toVar();

    If(hit.greaterThan(0.5), () => {
      const pHit = ro.add(rd.mul(t));
      const nGeom = westCanyonNormal(pHit);
      const grain = mineralToothFbm(pHit, float(2.6));
      const fineSand = mineralToothFbm(pHit.add(vec3(5.2, 1.7, 8.4)), float(11.5));

      // Rich sculpted Navajo/Antelope Canyon cross-bedded sandstone strata:
      // Burnt Sienna, Cinnabar-Vermilion, Hematite Rust, Yellow Ochre & dark Raw-Umber desert-varnish streaks
      const strataCoord = pHit.y
        .mul(4.2)
        .add(sin(pHit.z.mul(0.42).add(pHit.x.mul(0.35))).mul(0.72))
        .add(grain.mul(0.95));
      const layerA = sin(strataCoord).mul(0.5).add(0.5);
      const layerB = cos(strataCoord.mul(2.15).add(1.2)).mul(0.5).add(0.5);
      const sharpBed = smoothstep(float(0.72), float(0.94), sin(strataCoord.mul(5.2)).mul(0.5).add(0.5));
      // Vertical manganese desert-varnish streaks down steep rock overhangs
      const varnishNoise = mineralToothFbm(vec3(pHit.x.mul(2.4), pHit.y.mul(0.35), pHit.z.mul(2.4)), float(1.0));
      const desertVarnish = smoothstep(float(0.64), float(0.88), varnishNoise).mul(
        float(1.0).sub(smoothstep(float(0.45), float(0.8), abs(nGeom.y)))
      );

      const rockBase = mix(cBurntSienna.mul(0.94), cCinnabar.mul(0.82), layerA.mul(0.65));
      const rockOchre = mix(rockBase, cYellowOchre.mul(0.92), layerB.mul(0.38));
      const rockIron = mix(rockOchre, cHematiteRust.mul(0.74), sharpBed.mul(0.52));
      const rockStrata = mix(
        rockIron.add(cWarmKaolin.mul(fineSand.mul(0.06))),
        cRawUmber.mul(0.34),
        desertVarnish.mul(0.48)
      );

      // Crisp wind-rippled desert sand floor winding beneath the natural rock arch
      const isSandFloor = smoothstep(float(0.66), float(0.90), nGeom.y).mul(
        invSmoothstep(-1.45, -0.75, pHit.y)
      );
      const ripPhase = pHit.z.mul(10.5).add(sin(pHit.x.mul(1.8)).mul(1.2)).add(grain.mul(1.8));
      const sandRipple = sin(ripPhase).mul(0.5).add(0.5);
      const sandAlbedo = mix(
        cBurntSienna.mul(0.76),
        mix(cYellowOchre.mul(0.92), cWarmKaolin.mul(0.88), float(0.38)),
        sandRipple.mul(0.36).add(0.34)
      );
      const albedo = mix(rockStrata, sandAlbedo, isSandFloor);

      // High-definition sandstone strata ledge relief & crisp wind-rippled floor sand normal perturbation
      const nMicro = applyPortalMicroNormal(nGeom, pHit, float(5.8), float(0.085));
      const strataSlope = vec3(float(0.0), cos(strataCoord.mul(2.15)).mul(0.18), float(0.0));
      const sandSlope = vec3(float(0.0), float(0.0), cos(ripPhase).mul(0.22));
      const n = normalize(nMicro.add(mix(strataSlope, sandSlope, isSandFloor)));

      // 14-step crisp golden-hour shadow march (carves dramatic natural rock-arch shadow across canyon floor!)
      const sh = float(1.0).toVar();
      const ts = float(0.07).toVar();
      Loop(14, () => {
        const hs = westCanyonSdf(pHit.add(nGeom.mul(0.022)).add(sunDir.mul(ts)));
        sh.assign(min(sh, clamp(hs.mul(7.5).div(ts), float(0.0), float(1.0))));
        ts.addAssign(clamp(hs, float(0.09), float(0.85)));
        If(sh.lessThan(0.015).or(ts.greaterThan(13.5)), () => {
          Break();
        });
      });

      const ao = clamp(
        westCanyonSdf(pHit.add(nGeom.mul(0.22)))
          .div(0.22)
          .mul(0.56)
          .add(westCanyonSdf(pHit.add(nGeom.mul(0.65))).div(0.65).mul(0.44)),
        0.22,
        1.0
      );

      const nDotL = max(float(0.0), dot(n, sunDir));
      const goldenSunCol = mix(cRealgarAmber, cBoneCalcite, float(0.44)).mul(2.45);

      // Signature Antelope Canyon warm glowing terracotta/amber bounced radiosity in shaded wall curves!
      const canyonBounce = mix(cBurntSienna, cRealgarAmber, float(0.55))
        .mul(float(0.28).add(abs(n.x).mul(0.28)).add(max(float(0.0), n.y.negate()).mul(0.18)))
        .mul(ao);
      // Cool twilight Prussian-Azurite zenith skylight filling deep shadow crevices for vivid warm/cool contrast
      const twilightFill = mix(cAzurite, cWarmKaolin, float(0.32))
        .mul(max(float(0.0), n.y.mul(0.5).add(0.5)).mul(0.24))
        .mul(ao);

      const lit = albedo.mul(
        goldenSunCol.mul(nDotL.mul(sh)).add(canyonBounce).add(twilightFill)
      );

      // Golden-hour desert dust perspective haze framing the distant terraced buttes beyond the rock arch
      const fogT = float(1.0).sub(exp(t.mul(-0.034)));
      const hazeCol = mix(sunsetHorizon.mul(0.76), sunsetMid.mul(0.84), float(0.45));
      finalCol.assign(mix(lit, hazeCol, clamp(fogT, 0.0, 0.78)));
    });

    // Subtle 4-tap golden-hour canyon sun-shafts slanting through the narrow sky slot & arch
    const dustShaft = float(0.0).toVar();
    const maxDustT = min(t, float(14.5));
    for (let s = 0; s < 4; s++) {
      const fS = (s + 0.45) / 4.0;
      const pD = ro.add(rd.mul(maxDustT.mul(fS)));
      const sunClear = smoothstep(
        float(0.18),
        float(0.82),
        westCanyonSdf(pD.add(sunDir.mul(2.1)))
      );
      const inCanyon = smoothstep(float(1.2), float(6.5), pD.z).mul(
        smoothstep(float(-1.7), float(0.9), pD.y)
      );
      dustShaft.addAssign(sunClear.mul(inCanyon).mul(0.032));
    }
    const canyonGlow = mix(cRealgarAmber, cYellowOchre, float(0.45)).mul(
      dustShaft.mul(float(1.0).add(hoverW.mul(0.30)))
    );

    const nearFrameAo = mix(float(1.0), edgeAo, exp(t.mul(-0.85)));
    return toneMapMineral(finalCol.add(canyonGlow).mul(nearFrameAo));
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
