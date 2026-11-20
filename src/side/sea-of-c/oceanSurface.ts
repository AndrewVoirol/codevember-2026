import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  floor,
  fwidth,
  length,
  max,
  min,
  mix,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  reflect,
  select,
  sin,
  smoothstep,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { evaluateWaveTSL, type WaveFieldUniforms } from './waveField';
import { evaluateHorizonHazeTSL, evaluateSkyReflectionTSL } from './skyDome';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

export interface OceanSurfaceRig {
  mesh: THREE.Mesh;
  uGlyphDensity: { value: number };
  uHalftoneSharpness: { value: number };
  dispose: () => void;
}

const OCEAN_SEGMENTS = 208;
const OCEAN_INNER_RADIUS = 34.0;
const OCEAN_OUTER_RADIUS = 220.0;
const OCEAN_RADIAL_POWER = 3.4;
const OCEAN_CENTER_Z = -4.0;

/**
 * Builds a high-resolution, pole-free radial circular ocean disc concentrated
 * densely across the hero foreground/midground (~0.32 m vertex spacing near
 * breaking waves) and flaring smoothly to r = 220 m with zero square-corner
 * horizon cutoffs from any orbit angle.
 */
function createConcentratedOceanGeometry(): THREE.BufferGeometry {
  const n = OCEAN_SEGMENTS;
  const stride = n + 1;
  const vertexCount = stride * stride;

  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);

  let pIdx = 0;
  let uvIdx = 0;

  for (let iz = 0; iz <= n; iz++) {
    const vf = iz / n;
    const v = vf * 2.0 - 1.0;
    const v2 = v * v;

    for (let ix = 0; ix <= n; ix++) {
      const uf = ix / n;
      const u = uf * 2.0 - 1.0;
      const u2 = u * u;

      // Elliptical squircle mapping: maps [-1, 1]^2 smoothly onto the unit disc
      // with isotropic near-square quads in foreground and zero polar pinch.
      const dx = u * Math.sqrt(Math.max(0.0, 1.0 - 0.5 * v2));
      const dz = v * Math.sqrt(Math.max(0.0, 1.0 - 0.5 * u2));
      const r = Math.hypot(dx, dz);

      let wx = 0.0;
      let wz = OCEAN_CENTER_Z;
      if (r > 1e-6) {
        const rWorld =
          OCEAN_INNER_RADIUS * r +
          (OCEAN_OUTER_RADIUS - OCEAN_INNER_RADIUS) *
            Math.pow(r, OCEAN_RADIAL_POWER);
        const scale = rWorld / r;
        wx = dx * scale;
        wz = dz * scale + OCEAN_CENTER_Z;
      }

      positions[pIdx] = wx;
      positions[pIdx + 1] = 0.0;
      positions[pIdx + 2] = wz;

      normals[pIdx] = 0.0;
      normals[pIdx + 1] = 1.0;
      normals[pIdx + 2] = 0.0;
      pIdx += 3;

      uvs[uvIdx] = uf;
      uvs[uvIdx + 1] = vf;
      uvIdx += 2;
    }
  }

  const indexCount = n * n * 6;
  const indices = new Uint32Array(indexCount);
  let iIdx = 0;

  // Outward +Y winding: (c - a) x (b - a) = (0, 0, +1) x (+1, 0, 0) = (0, +1, 0)
  for (let iz = 0; iz < n; iz++) {
    const rowA = iz * stride;
    const rowB = (iz + 1) * stride;
    for (let ix = 0; ix < n; ix++) {
      const a = rowA + ix;
      const b = a + 1;
      const c = rowB + ix;
      const d = c + 1;

      indices[iIdx++] = a;
      indices[iIdx++] = c;
      indices[iIdx++] = b;

      indices[iIdx++] = b;
      indices[iIdx++] = c;
      indices[iIdx++] = d;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  geom.boundingSphere = new THREE.Sphere(
    new THREE.Vector3(0, 0, OCEAN_CENTER_Z),
    OCEAN_OUTER_RADIUS + 20.0
  );
  return geom;
}

/**
 * Negative-safe fractional part (`x - floor(x)`) avoiding Apple Silicon Metal
 * fast-math `fract(x)` truncation bugs on negative world coordinates.
 */
const safeFract2 = Fn(([v_in]: [TslNode]) => {
  const v = vec2(v_in);
  return v.sub(floor(v));
});

const safeFract1 = Fn(([x_in]: [TslNode]) => {
  const x = float(x_in);
  return x.sub(floor(x));
});

/**
 * Fast 2D -> 1D hash in [0, 1) for per-cell micro-facet variation and capillary noise.
 */
const hash21 = Fn(([p_in]: [TslNode]) => {
  const p = safeFract2(vec2(p_in).mul(vec2(123.34, 456.21)));
  const d = dot(p, p.add(45.32));
  return safeFract1(p.x.add(d).mul(p.y.add(d)));
});

/**
 * Smooth quintic 2D value noise (`C^2` continuous).
 */
const valueNoise2D = Fn(([p_in]: [TslNode]) => {
  const p = vec2(p_in);
  const i = floor(p);
  const f = p.sub(i);
  const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0));

  const n00 = hash21(i.add(vec2(0.0, 0.0)));
  const n10 = hash21(i.add(vec2(1.0, 0.0)));
  const n01 = hash21(i.add(vec2(0.0, 1.0)));
  const n11 = hash21(i.add(vec2(1.0, 1.0)));

  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
});

/**
 * Dynamic capillary micro-ripple height via irrational-angle rotated 3-octave FBM.
 * Strictly zero polar star harmonics and zero grid alignment.
 */
const capillaryHeight = Fn(([xz_in, time_in]: [TslNode, TslNode]) => {
  const p = vec2(xz_in);
  const t = float(time_in);

  // Golden-angle rotation matrix between octaves
  const c1 = float(0.80901699);
  const s1 = float(0.58778525);
  const c2 = float(0.30901699);
  const s2 = float(0.95105652);

  const p0 = p.mul(1.35).add(vec2(t.mul(0.42), t.mul(-0.31)));
  const v0 = valueNoise2D(p0);

  const p1Base = vec2(
    p.x.mul(c1).sub(p.y.mul(s1)),
    p.x.mul(s1).add(p.y.mul(c1))
  );
  const p1 = p1Base.mul(2.85).add(vec2(t.mul(-0.58), t.mul(-0.44)));
  const v1 = valueNoise2D(p1);

  const p2Base = vec2(
    p.x.mul(c2).sub(p.y.mul(s2)),
    p.x.mul(s2).add(p.y.mul(c2))
  );
  const p2 = p2Base.mul(5.9).add(vec2(t.mul(0.83), t.mul(0.51)));
  const v2 = valueNoise2D(p2);

  return v0.mul(0.54).add(v1.mul(0.30)).add(v2.mul(0.16));
});

/**
 * Smooth asymptotic HDR highlight shoulder strictly above 1.0:
 * Leaves diffuse/troughs <= 1.0 untouched while capping peak sun-glitter cores
 * cleanly at <= 1.78 so selective bloom (threshold 1.45) softly lights only the
 * hottest golden sun-path `C`s and bioluminescent crests.
 */
const hdrSoftShoulder = Fn(([col_in]: [TslNode]) => {
  const col = max(vec3(col_in), vec3(0.0));
  const base = min(col, vec3(1.0));
  const excess = max(col.sub(1.0), vec3(0.0));
  const compressed = excess.div(vec3(1.0).add(excess.mul(1.12))).mul(0.86);
  return base.add(compressed);
});

/**
 * Creates the Infinite Sun-Glint Typographic 'C'-Halftone Ocean Surface.
 * Shares `evaluateWaveTSL` verbatim with `granularFluid.ts` and `evaluateHorizonHazeTSL`
 * verbatim with `skyDome.ts` for seamless 3D wave crests and invisible horizon fusion.
 */
export function createOceanSurface(u: WaveFieldUniforms): OceanSurfaceRig {
  const geometry = createConcentratedOceanGeometry();

  // Tuned for 40 deg FOV hero framing: bold unmistakable capital 'C' glyphs
  // across foreground & midground swells, fading smoothly into silky horizon shimmer.
  const uGlyphDensity = uniform(2.05);
  const uHalftoneSharpness = uniform(1.0);

  const material = new THREE.MeshBasicNodeMaterial({
    side: THREE.DoubleSide,
    fog: false,
    depthWrite: true,
    depthTest: true,
  });

  // ---------------------------------------------------------------------------
  // 1. Vertex Stage: Unified Gerstner + Curling Breaker + Wake/Vortex Displacement
  // ---------------------------------------------------------------------------
  const restXZ = positionLocal.xz;
  const wave = evaluateWaveTSL(restXZ, u);

  // Taper displacement cleanly near far horizon (r = 115..210 m) to prevent
  // distant silhouette aliasing while preserving 100% amplitude across foreground/midground.
  const rFromCenter = length(restXZ.sub(vec2(0.0, OCEAN_CENTER_Z)));
  const horizonTaper = float(1.0).sub(
    smoothstep(float(115.0), float(210.0), rFromCenter)
  );

  const waveDisp = vec3(wave.displacement).mul(horizonTaper);
  const displacedPos = vec3(
    positionLocal.x.add(waveDisp.x),
    positionLocal.y.add(waveDisp.y),
    positionLocal.z.add(waveDisp.z)
  );
  material.positionNode = displacedPos;

  // Compact vertex-to-fragment varyings (evaluated strictly once per vertex)
  const vWaveNormal = normalize(
    mix(vec3(0.0, 1.0, 0.0), vec3(wave.normal), horizonTaper)
  ).toVarying('vOceanWaveNormal');

  const vWaveMetrics = vec4(
    float(wave.crestEnergy).mul(horizonTaper),
    float(wave.curlFactor).mul(horizonTaper),
    float(wave.wakeEnergy).mul(horizonTaper),
    float(wave.orbitalPhase)
  ).toVarying('vOceanWaveMetrics');

  const vSurfaceXZ = restXZ.toVarying('vOceanSurfaceXZ');

  // ---------------------------------------------------------------------------
  // 2. Fragment Stage: Analytical Embossed 'C'-Halftone Sea + Physical Optics
  // ---------------------------------------------------------------------------
  const oceanColorFn = Fn(() => {
    const time = float(u.uTime);
    const sunDir = normalize(vec3(u.uSunDir));
    const sunColor = u.uSunColor.rgb;
    const waterDeep = u.uWaterDeep.rgb;
    const waterShallow = u.uWaterShallow.rgb;
    const glintColor = u.uGlintColor.rgb;
    const foamColor = u.uFoamColor.rgb;
    const biolumGain = float(u.uBiolumGain);
    const fogDensity = float(u.uFogDensity);

    // Interactive maelstrom swirl gently twists the halftone 'C' lattice into the vortex
    const vortex = vec4(u.uVortex);
    const vRel = vSurfaceXZ.sub(vortex.xy);
    const vFalloff = exp(dot(vRel, vRel).negate().mul(0.032));
    const swirlAng = vortex.z.mul(vFalloff).mul(0.82);
    const cosSw = cos(swirlAng);
    const sinSw = sin(swirlAng);
    const swirledXZ = vortex.xy.add(
      vec2(
        vRel.x.mul(cosSw).sub(vRel.y.mul(sinSw)),
        vRel.x.mul(sinSw).add(vRel.y.mul(cosSw))
      )
    );

    const cellCoord = swirledXZ.mul(uGlyphDensity);

    // STRICT UNIFORM CONTROL FLOW: evaluate screen-space derivatives unconditionally
    // at the top of the fragment shader before any conditional selection.
    const fwCell = fwidth(cellCoord);
    const filterW = max(fwCell.x, fwCell.y);

    const cellId = floor(cellCoord);
    const cellFrac = cellCoord.sub(cellId);

    // Local glyph space in [-0.5, 0.5]^2:
    // +localP.x = +X (screen right), +localP.y = -Z (screen up toward horizon)
    // so capital 'C' reads upright and opens cleanly to the right (+X).
    const localP = vec2(cellFrac.x.sub(0.5), float(0.5).sub(cellFrac.y));

    const crestEnergy = clamp(vWaveMetrics.x, float(0.0), float(1.4));
    const curlFactor = clamp(vWaveMetrics.y, float(0.0), float(1.4));
    const wakeEnergy = clamp(vWaveMetrics.z, float(0.0), float(1.4));

    const worldPos = positionWorld;
    const toCam = cameraPosition.sub(worldPos);
    const camDist = max(length(toCam), float(1e-4));
    const V = toCam.div(camDist);

    // Wave normal + safe backface orientation for curling wave overhangs
    const rawN = normalize(vWaveNormal);
    const N_base = select(
      dot(rawN, V).lessThan(float(0.0)),
      rawN.negate(),
      rawN
    );

    // Orthonormal wave tangent frame aligned with localP (+X right, -Z up),
    // guarded against near-vertical breaking wave walls where |N_base.x| -> 1.
    const refAxis = select(
      abs(N_base.x).greaterThan(float(0.92)),
      vec3(0.0, 0.0, -1.0),
      vec3(1.0, 0.0, 0.0)
    );
    const T_wave = normalize(
      refAxis.sub(N_base.mul(dot(refAxis, N_base))).add(vec3(1e-5, 0.0, 0.0))
    );
    const B_up = normalize(cross(N_base, T_wave));

    // Dynamic irrational-angle capillary micro-ripples (faded cleanly at distance)
    const capFade = exp(camDist.negate().mul(0.018));
    const capEps = float(0.075);
    const capC = capillaryHeight(vSurfaceXZ, time);
    const capX = capillaryHeight(vSurfaceXZ.add(vec2(capEps, 0.0)), time);
    const capZ = capillaryHeight(vSurfaceXZ.add(vec2(0.0, capEps.negate())), time);
    const capSlopeScale = float(0.15).mul(capFade);
    const N_water = normalize(
      N_base.add(T_wave.mul(capC.sub(capX).div(capEps).mul(capSlopeScale)))
        .add(B_up.mul(capC.sub(capZ).div(capEps).mul(capSlopeScale)))
    );

    // Physical Fresnel & analytical twilight sky reflection
    const NdotV = max(dot(N_water, V), float(0.002));
    const fresnel = float(0.025).add(
      float(0.975).mul(pow(max(float(1.0).sub(NdotV), float(1e-5)), float(5.0)))
    );
    const rawRefl = reflect(V.negate(), N_water);
    const reflDir = normalize(
      vec3(rawRefl.x, max(rawRefl.y, float(0.015)), rawRefl.z)
    );
    const skyRefl = evaluateSkyReflectionTSL(reflDir, u);

    // Physical sun-glitter driver along sun reflection path
    const H = normalize(sunDir.add(V));
    const sunVis = smoothstep(float(-0.10), float(0.03), sunDir.y);
    const NdotH_water = max(dot(N_water, H), float(1e-5));
    const specWide = pow(NdotH_water, float(16.0)).mul(sunVis);
    const specMid = pow(NdotH_water, float(64.0)).mul(sunVis);

    // Mode-aware solar glint scale: blazing warm gold in 'golden',
    // restrained cool silver in 'storm' and moonlight in 'biolum'.
    const sunWarmth = clamp(sunColor.r.sub(sunColor.b).mul(2.4), float(0.0), float(1.0));
    const nonBiolum = float(1.0).sub(clamp(biolumGain, float(0.0), float(1.0)));
    const goldenMask = sunWarmth.mul(nonBiolum);
    const glintModeScale = mix(
      mix(float(0.66), float(1.24), sunWarmth),
      float(0.55),
      clamp(biolumGain, float(0.0), float(1.0))
    );

    // Narrow sun-glitter lane mask separates blazing champagne-gold central path
    // from cool royal cobalt-sapphire flanks (eliminates warm brown tint off-axis).
    const glintLaneMask = clamp(
      specWide.mul(1.85).add(specMid.mul(1.4)),
      float(0.0),
      float(1.0)
    );

    // Backlit translucent jewel emerald-turquoise subsurface scattering restricted to
    // steep wave bellies & breaking curls so calm foreground troughs remain deep sapphire.
    const forwardVec = normalize(sunDir.negate().add(N_water.mul(0.55)));
    const sunForward = pow(
      clamp(dot(V, forwardVec), float(1e-5), float(1.0)),
      float(3.0)
    );
    const steepCrest = smoothstep(float(0.32), float(0.96), crestEnergy);
    const curlTight = pow(clamp(curlFactor, float(1e-5), float(1.2)), float(1.75));
    const crestThin = clamp(
      steepCrest.mul(0.28).add(curlTight.mul(0.76)),
      float(0.0),
      float(0.92)
    );
    const jewelEmerald = mix(waterShallow, vec3(0.005, 0.24, 0.26), float(0.45));
    const sssRadiance = jewelEmerald.mul(
      crestThin.mul(float(0.08).add(sunForward.mul(0.54).mul(sunVis)))
    );
    // Warm amber-gold horizon light transmitting through thin backlit breaking curls
    const warmCurlTransmit = glintColor.mul(
      curlTight.mul(sunForward).mul(sunVis).mul(goldenMask).mul(0.34)
    );

    // Dinoflagellate bioluminescence on breaking crests & interactive wakes
    const biolumExcite = clamp(
      wakeEnergy
        .mul(1.45)
        .add(curlFactor.mul(0.98))
        .add(smoothstep(float(0.28), float(0.92), crestEnergy).mul(0.75)),
      float(0.0),
      float(1.5)
    );
    const biolumDrive = biolumExcite.mul(biolumGain);
    const biolumCol = vec3(0.04, 0.88, 1.15);

    // -------------------------------------------------------------------------
    // 3. Analytical SDF Capital 'C' Halftone Lattice + Embossed Bevel Normal
    // -------------------------------------------------------------------------
    const cellRand = hash21(cellId);

    // Modulate 'C' scale & stroke boldness smoothly from physical light + wave energy:
    // slender deep sapphire 'C's in troughs -> bold luminous golden 'C's along sun path
    const opticalEnergy = clamp(
      specWide
        .mul(0.95)
        .add(specMid.mul(0.90))
        .mul(glintModeScale)
        .add(steepCrest.mul(0.34))
        .add(curlFactor.mul(0.45))
        .add(wakeEnergy.mul(0.65))
        .add(biolumDrive.mul(0.55)),
      float(0.0),
      float(1.0)
    );

    const glyphRadius = mix(float(0.306), float(0.330), opticalEnergy);
    const halfWidth = mix(float(0.076), float(0.114), opticalEnergy).mul(
      float(0.96).add(cellRand.mul(0.08))
    );
    const cornerR = halfWidth.mul(0.26);

    // Exact C^1 continuous SDF for a geometric sans-serif capital 'C'
    // opening cleanly to +X with ±40 deg radial-cut chamfered terminals.
    const sy = select(localP.y.lessThan(float(0.0)), float(-1.0), float(1.0));
    const q = vec2(localP.x, abs(localP.y));
    const qLen = max(length(q), float(1e-5));

    const uRad = vec2(0.766044, 0.642788);
    const uTan = vec2(0.642788, -0.766044);
    const qRad = dot(q, uRad);
    const qTan = dot(q, uTan);

    const dArc = abs(qLen.sub(glyphRadius)).sub(halfWidth);

    const ax = abs(qRad.sub(glyphRadius)).sub(halfWidth.sub(cornerR));
    const ay = qTan.add(cornerR);
    const dTerm = length(vec2(max(ax, float(0.0)), max(ay, float(0.0))))
      .add(min(max(ax, ay), float(0.0)))
      .sub(cornerR);

    const isTerminal = qTan.greaterThan(cornerR.negate());
    const sdf = select(isTerminal, dTerm, dArc);

    // Exact analytical outward 2D bevel gradient (zero dFdx/dFdy foil spikes)
    const sArc = select(qLen.lessThan(glyphRadius), float(-1.0), float(1.0));
    const gArcQ = q.div(qLen).mul(sArc);

    const sTermRad = select(
      qRad.lessThan(glyphRadius),
      float(-1.0),
      float(1.0)
    );
    const wRad = max(ax.add(cornerR), float(0.0));
    const wTan = max(ay.add(cornerR), float(0.0));
    const gTermQ = normalize(
      uRad.mul(sTermRad.mul(wRad)).add(uTan.mul(wTan)).add(vec2(1e-5, 0.0))
    );

    const gQ = select(isTerminal, gTermQ, gArcQ);
    const bevelGrad2D = vec2(gQ.x, gQ.y.mul(sy));

    // Distance Anti-Aliasing (Zero Horizon Moire!):
    // Blend razor-sharp AA 'C' strokes into exact analytical cell area coverage
    // as cell footprint approaches Nyquist limit toward the distant horizon.
    const sharpScale = float(0.84).div(max(uHalftoneSharpness, float(0.25)));
    const aaEdge = clamp(filterW.mul(sharpScale), float(0.0035), float(0.24));
    const glyphCrisp = float(1.0).sub(smoothstep(aaEdge.negate(), aaEdge, sdf));

    // Analytical area fraction of 280° (4.88692 rad) circular arc stroke inside 1x1 cell
    const areaFill = float(4.88692).mul(glyphRadius).mul(halfWidth.mul(2.0));
    const nyquistBlend = smoothstep(float(0.18), float(0.68), filterW);

    // Steep breaking wave wall fade: gently transitions out stretched 2D glyphs ONLY on
    // near-vertical curling faces so overhead views stay seamlessly tiled in sapphire 'C's.
    const steepWallFade = smoothstep(float(0.42), float(0.78), N_base.y).mul(
      float(1.0).sub(smoothstep(float(0.38), float(0.82), curlFactor))
    );

    const glyphMask = mix(glyphCrisp, areaFill, nyquistBlend).mul(steepWallFade);

    // Glossy chamfer bevel profile along outer & inner edges of each 'C' tile
    const bevelOuter = smoothstep(
      halfWidth.mul(-0.88),
      halfWidth.mul(-0.12),
      sdf
    );
    const bevelClip = float(1.0).sub(smoothstep(float(-0.008), float(0.014), sdf));
    const bevelBand = bevelOuter.mul(bevelClip);
    const bevelAmt = bevelBand
      .mul(float(1.0).sub(nyquistBlend))
      .mul(steepWallFade)
      .mul(0.52);

    // Subtle per-cell normal micro-cant gives organic sparkling tile facets along sun path
    const cellJitterX = cellRand.sub(0.5).mul(0.055);
    const cellJitterY = hash21(cellId.add(vec2(17.3, 31.7))).sub(0.5).mul(0.055);
    const facetTilt = float(1.0).sub(nyquistBlend).mul(steepWallFade);

    const N_glyph = normalize(
      N_water.add(
        T_wave.mul(bevelGrad2D.x.mul(bevelAmt).add(cellJitterX.mul(facetTilt)))
      ).add(
        B_up.mul(bevelGrad2D.y.mul(bevelAmt).add(cellJitterY.mul(facetTilt)))
      )
    );

    // -------------------------------------------------------------------------
    // 4. Dual-Material Composite: Deep Sapphire Channels + Embossed 'C' Tiles
    // -------------------------------------------------------------------------
    // Cool reflection tint on off-axis water prevents warm brown-bronze washout
    // while keeping warm golden reflection active inside the sun-glitter corridor.
    const coolSkyRefl = mix(
      vec3(
        skyRefl.x.mul(0.55),
        skyRefl.y.mul(0.86),
        skyRefl.z.mul(1.26)
      ),
      skyRefl,
      glintLaneMask
    );

    // Liquid body between 'C' tiles & inside steep breaking soliton walls:
    // stays DEEP abyssal sapphire-indigo (never pastel mint wallpaper!) so crevices between
    // stacked 3D sea-glass 'C's read with rich dark liquid depth & jewel contrast.
    const wallLiquidLift = float(1.0).sub(steepWallFade);
    const wallBellyTint = mix(
      waterDeep.mul(0.65),
      vec3(0.001, 0.014, 0.028).mul(nonBiolum).add(vec3(0.001, 0.028, 0.055).mul(biolumGain)),
      clamp(curlTight.mul(0.75), float(0.0), float(1.0))
    );
    const channelBody = mix(
      waterDeep.mul(float(0.20).add(steepCrest.mul(0.06))),
      wallBellyTint,
      wallLiquidLift.mul(0.88)
    )
      .add(sssRadiance.mul(float(0.18).add(wallLiquidLift.mul(0.12))))
      .add(warmCurlTransmit.mul(float(0.24).add(wallLiquidLift.mul(0.18))));

    const channelCol = mix(
      channelBody,
      coolSkyRefl.mul(0.22),
      fresnel.mul(0.32)
    )
      .add(glintColor.mul(specMid.mul(0.16).mul(glintModeScale)))
      .add(biolumCol.mul(biolumDrive.mul(float(0.24).add(wallLiquidLift.mul(0.36)))));

    // Embossed glossy 'C' glyph tiles
    const NdotV_g = max(dot(N_glyph, V), float(0.002));
    const fresnelG = float(0.04).add(
      float(0.96).mul(pow(max(float(1.0).sub(NdotV_g), float(1e-5)), float(4.2)))
    );
    const rawReflG = reflect(V.negate(), N_glyph);
    const reflDirG = normalize(
      vec3(rawReflG.x, max(rawReflG.y, float(0.025)), rawReflG.z)
    );
    const skyReflG_raw = evaluateSkyReflectionTSL(reflDirG, u);
    const skyReflG = mix(
      vec3(
        skyReflG_raw.x.mul(0.54),
        skyReflG_raw.y.mul(0.86),
        skyReflG_raw.z.mul(1.28)
      ),
      skyReflG_raw,
      glintLaneMask
    );

    // Rich cool royal sapphire-cerulean tiles away from sun path (matching fused_hero_sea_of_c
    // cobalt flanks), moody slate-teal in Storm, deep midnight navy in Bioluminescent mode.
    const slopeFacingSky = clamp(N_glyph.y.mul(0.65).add(0.35), float(0.0), float(1.0));
    const royalCobaltLift = waterDeep
      .mul(3.8)
      .add(waterShallow.mul(0.05))
      .add(
        vec3(0.009, 0.036, 0.095)
          .mul(goldenMask)
          .add(vec3(0.008, 0.024, 0.040).mul(nonBiolum.sub(goldenMask)))
          .add(vec3(0.002, 0.010, 0.026).mul(biolumGain))
      );
    const troughSapphire = mix(
      waterDeep.mul(1.38),
      royalCobaltLift,
      float(0.28).add(slopeFacingSky.mul(0.26)).add(fresnelG.mul(0.28))
    );

    // Blend cooled twilight reflection onto raised tile plateau & bevel edges
    let glyphCol = mix(
      troughSapphire,
      skyReflG.mul(0.82),
      clamp(fresnelG.mul(0.54).add(bevelBand.mul(0.16)), float(0.0), float(0.72))
    );

    // Backlit translucent emerald-turquoise glow on rolling wave bellies
    glyphCol = glyphCol
      .add(sssRadiance.mul(1.05))
      .add(warmCurlTransmit.mul(0.95));

    // Blazing champagne-gold & warm ivory sun-glitter highway across embossed 'C' tiles
    const NdotH_g = max(dot(N_glyph, H), float(1e-5));
    const specBroadG = pow(NdotH_g, float(16.0)).mul(sunVis);
    const specMidG = pow(NdotH_g, float(68.0)).mul(sunVis);
    const specBevelSpark = pow(NdotH_g, float(200.0))
      .mul(sunVis)
      .mul(float(0.52).add(bevelBand.mul(1.48)));

    const glintWeight = specBroadG
      .mul(0.96)
      .add(specMidG.mul(1.32))
      .add(specBevelSpark.mul(1.08))
      .mul(glintModeScale);
    const hotGlintTint = mix(
      mix(glintColor, vec3(1.0, 0.86, 0.52), goldenMask.mul(0.35)),
      mix(sunColor, vec3(1.0, 0.97, 0.88), float(0.48)),
      clamp(specMidG.add(specBevelSpark.mul(0.65)), float(0.0), float(1.0))
    );
    glyphCol = glyphCol.add(hotGlintTint.mul(glintWeight));

    // Crisp foam lip highlights on breaking crests & live pointer wake rings
    const foamDrive = clamp(
      smoothstep(float(0.56), float(0.98), crestEnergy.add(curlFactor.mul(0.48)))
        .mul(0.58)
        .add(wakeEnergy.mul(0.62)),
      float(0.0),
      float(1.0)
    );
    glyphCol = mix(glyphCol, foamColor.mul(0.92), foamDrive.mul(0.56));

    // Crisp electric cyan-turquoise dinoflagellate ignition in Bioluminescent mode
    glyphCol = glyphCol.add(
      biolumCol.mul(biolumDrive.mul(float(0.94).add(bevelBand.mul(0.45))))
    );

    // Calibrated HDR shoulder (<= 1.78 peak) so selective bloom (threshold 1.45) lights
    // only the hottest sun-path 'C' glints & biolum peaks without hazing foreground glyphs
    const glyphCapped = hdrSoftShoulder(glyphCol);
    const channelCapped = hdrSoftShoulder(channelCol);
    const surfaceCol = mix(channelCapped, glyphCapped, glyphMask);

    // -------------------------------------------------------------------------
    // 5. Seamless Atmospheric Horizon Fog (Locked to skyDome.ts Horizon Haze)
    // -------------------------------------------------------------------------
    const viewRay = normalize(worldPos.sub(cameraPosition));
    const horizonFogCol = evaluateHorizonHazeTSL(viewRay, u);

    // Keep foreground & near-midground cobalt flanks crisp (0..18 m clear zone), then
    // melt smoothly into horizon haze so sea & twilight sky fuse invisibly at distance.
    const hazeDist = max(camDist.sub(18.0), float(0.0));
    const distFog = float(1.0).sub(
      exp(hazeDist.negate().mul(max(fogDensity, float(0.001))).mul(1.26))
    );
    const radialOut = length(vSurfaceXZ.sub(vec2(0.0, OCEAN_CENTER_Z)));
    const edgeMelt = smoothstep(float(138.0), float(212.0), radialOut);
    const fogFactor = clamp(max(distFog, edgeMelt), float(0.0), float(1.0));

    return mix(surfaceCol, horizonFogCol, fogFactor);
  });

  material.colorNode = oceanColorFn();

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'SeaOfC_OceanSurface';
  mesh.frustumCulled = false;

  return {
    mesh,
    uGlyphDensity,
    uHalftoneSharpness,
    dispose: () => {
      geometry.dispose();
      material.dispose();
    },
  };
}
