struct SceneUniforms {
  viewProj: mat4x4<f32>,
  lightViewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,        // .xyz = camera world pos, .w = time (seconds)
  sunPosAndRadius: vec4<f32>,  // .xyz = sun center world pos, .w = sun radius
  earthPosAndRadius: vec4<f32>,// .xyz = earth center world pos, .w = earth radius
  moonPosAndRadius: vec4<f32>, // .xyz = moon center world pos, .w = moon radius
  keyLightDir: vec4<f32>,      // .xyz = directional studio light dir, .w = isLightTheme (0 or 1)
  params: vec4<f32>,           // .x = hoveredGroup, .y = selectedGroup, .z = cutawayAlpha, .w = eclipseStrength
  viewport: vec4<f32>,         // .xy = canvas size px, .zw = 1.0 / canvas size
};

struct InstanceData {
  model: mat4x4<f32>,
  normalMat: mat4x4<f32>,
  colorAndMat: vec4<f32>,      // .rgb = base color, .w = materialId
  partMeta: vec4<f32>,         // .x = groupId, .y = roughness, .z = metallic, .w = cutawayFlag
};

@group(0) @binding(0) var<uniform> u_scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> u_instances: array<InstanceData>;
@group(0) @binding(2) var u_shadowMap: texture_depth_2d;
@group(0) @binding(3) var u_shadowSampler: sampler_comparison;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) edgeMask: f32,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) localPos: vec3<f32>,
  @location(3) uv: vec2<f32>,
  @location(4) colorAndMat: vec4<f32>,
  @location(5) partMeta: vec4<f32>,
  @location(6) shadowPos: vec3<f32>,
  @location(7) edgeMask: f32,
};

@vertex
fn vs_shadow(
  in: VertexInput,
  @builtin(instance_index) instIdx: u32
) -> @builtin(position) vec4<f32> {
  let inst = u_instances[instIdx];
  let world = inst.model * vec4<f32>(in.position, 1.0);
  if (inst.partMeta.w > 1.5) {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
  }
  return u_scene.lightViewProj * world;
}

@vertex
fn vs_main(
  in: VertexInput,
  @builtin(instance_index) instIdx: u32
) -> VertexOutput {
  let inst = u_instances[instIdx];
  let world4 = inst.model * vec4<f32>(in.position, 1.0);
  let worldNormal = normalize((inst.normalMat * vec4<f32>(in.normal, 0.0)).xyz);
  let lightClip = u_scene.lightViewProj * world4;
  let shadowNdc = lightClip.xyz / max(lightClip.w, 0.0001);
  let shadowUv = vec2<f32>(shadowNdc.x * 0.5 + 0.5, 0.5 - shadowNdc.y * 0.5);

  var out: VertexOutput;
  out.clipPos = u_scene.viewProj * world4;
  out.worldPos = world4.xyz;
  out.worldNormal = worldNormal;
  out.localPos = in.position;
  out.uv = in.uv;
  out.colorAndMat = inst.colorAndMat;
  out.partMeta = inst.partMeta;
  out.shadowPos = vec3<f32>(shadowUv, shadowNdc.z);
  out.edgeMask = in.edgeMask;
  return out;
}

fn hash31(p: vec3<f32>) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn valueNoise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(
      mix(hash31(i + vec3<f32>(0.0, 0.0, 0.0)), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
      mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
      u.y
    ),
    mix(
      mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
      mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
      u.y
    ),
    u.z
  );
}

fn sampleShadowPCF(shadowPos: vec3<f32>, nDotL: f32) -> f32 {
  let inBounds = all(shadowPos.xy >= vec2<f32>(0.002)) &&
                 all(shadowPos.xy <= vec2<f32>(0.998)) &&
                 shadowPos.z >= 0.0 && shadowPos.z <= 1.0;
  let bias = max(0.0018 * (1.0 - nDotL), 0.0006);
  let depthRef = shadowPos.z - bias;
  let texel = 1.0 / 2048.0;
  var sum = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      let offset = vec2<f32>(f32(x), f32(y)) * texel;
      sum += textureSampleCompareLevel(u_shadowMap, u_shadowSampler, shadowPos.xy + offset, depthRef);
    }
  }
  return select(1.0, sum / 9.0, inBounds);
}

fn sphereShadowOcclusion(p: vec3<f32>, lightPos: vec3<f32>, lightR: f32, occPos: vec3<f32>, occR: f32) -> f32 {
  let toLight = lightPos - p;
  let distLight = length(toLight);
  let L = toLight / max(distLight, 0.0001);
  let toOcc = occPos - p;
  let proj = dot(toOcc, L);
  if (proj <= 0.0 || proj >= distLight) {
    return 1.0;
  }
  let closest = toOcc - L * proj;
  let d = length(closest);
  let penumbra = occR + lightR * (proj / max(distLight, 1.0)) * 0.65;
  let umbra = max(occR * 0.55, occR - lightR * (proj / max(distLight, 1.0)) * 0.35);
  return smoothstep(umbra, penumbra, d);
}

fn evaluateStudioEnvironment(R: vec3<f32>, isLightTheme: bool) -> vec3<f32> {
  let domeT = clamp(R.z * 0.5 + 0.5, 0.0, 1.0);
  let domeTop = select(vec3<f32>(0.24, 0.26, 0.32), vec3<f32>(0.95, 0.93, 0.88), isLightTheme);
  let domeBot = select(vec3<f32>(0.05, 0.05, 0.07), vec3<f32>(0.42, 0.40, 0.36), isLightTheme);
  var env = mix(domeBot, domeTop, domeT);

  // Dark workshop horizon band for high-contrast horological metal reflections
  let horizonDark = 1.0 - 0.52 * exp(-R.z * R.z * 18.0);
  env *= horizonDark;

  // 3-panel horological studio softboxes (overhead key panel, warm brass rim strip, cool silver fill)
  let boxKey = pow(max(dot(R, normalize(vec3<f32>(-0.34, -0.42, 0.84))), 0.0), 14.0);
  let boxWarmRim = pow(max(dot(R, normalize(vec3<f32>(0.76, -0.28, 0.58))), 0.0), 10.0);
  let boxCoolFill = pow(max(dot(R, normalize(vec3<f32>(-0.62, 0.56, 0.55))), 0.0), 8.0);

  env += vec3<f32>(1.35, 1.28, 1.16) * boxKey;
  env += vec3<f32>(1.15, 0.82, 0.42) * boxWarmRim * 0.75;
  env += vec3<f32>(0.62, 0.76, 0.96) * boxCoolFill * 0.55;
  return env;
}

fn acesFilm(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  // 1. Evaluate all derivatives in uniform control flow at top of fs_main
  let dNdx = dpdx(in.worldNormal);
  let dNdy = dpdy(in.worldNormal);
  let dPdx = dpdx(in.worldPos);
  let dPdy = dpdy(in.worldPos);
  let uvGrad = fwidth(in.uv);
  let edgeFwidth = fwidth(in.edgeMask);

  var N = normalize(in.worldNormal);
  let V = normalize(u_scene.cameraPos.xyz - in.worldPos);
  var nDotV = max(dot(N, V), 0.001);

  let matId = i32(round(in.colorAndMat.w));
  let groupId = in.partMeta.x;
  let roughness = clamp(in.partMeta.y, 0.06, 0.95);
  let metallic = clamp(in.partMeta.z, 0.0, 1.0);
  let cutawayFlag = in.partMeta.w;
  let isLightTheme = u_scene.keyLightDir.w > 0.5;

  // 2. Shadow map sampling (uniform control flow)
  let keyL = normalize(u_scene.keyLightDir.xyz);
  var nDotKey = max(dot(N, keyL), 0.0);
  let keyShadow = sampleShadowPCF(in.shadowPos, nDotKey);

  // 3. Cutaway sectioning for outer coaxial sleeves, upper gears & barrels when X-Ray Cutaway [C] is active
  let cutawayActive = u_scene.params.z;
  var cutEdgeMask = 0.0;
  if (cutawayActive > 0.25 && cutawayFlag > 0.5 && cutawayFlag < 1.5) {
    var cutPivot = vec2<f32>(-8.0, 0.0);
    if (in.worldPos.x < -60.0) {
      cutPivot = vec2<f32>(-121.0, 11.0);
    } else if (in.worldPos.x > 45.0) {
      cutPivot = vec2<f32>(95.2, -11.8);
    }
    let relXY = in.worldPos.xy - cutPivot;
    let ang = atan2(relXY.y, relXY.x);
    if (ang > -2.75 && ang < -0.35) {
      discard;
    }
    let dEdge = min(abs(ang + 2.75), abs(ang + 0.35));
    cutEdgeMask = smoothstep(0.11, 0.015, dEdge) * cutawayActive;
  }

  let sunVec = u_scene.sunPosAndRadius.xyz - in.worldPos;
  let sunDist = length(sunVec);
  let sunL = sunVec / max(sunDist, 0.0001);
  var nDotSun = max(dot(N, sunL), 0.0);

  // 4. Base Albedo, Micro-Relief Normal Perturbation & Procedural Horological Surface Detail by Material ID
  var baseColor = in.colorAndMat.rgb;
  var rough = roughness;
  var metal = metallic;
  var emissive = vec3<f32>(0.0);
  if (cutEdgeMask > 0.01) {
    baseColor = mix(baseColor, vec3<f32>(1.0, 0.76, 0.36), cutEdgeMask * 0.72);
    emissive = emissive + vec3<f32>(0.34, 0.18, 0.05) * cutEdgeMask;
  }

  let radialR = length(in.localPos.xy);
  let localAng = atan2(in.localPos.y, in.localPos.x);
  let turnGrain = sin(radialR * 38.0 + in.uv.x * 160.0 + in.localPos.z * 16.0) * 0.5 + 0.5;
  let cerclage = sin(radialR * 58.0) * 0.5 + 0.5;
  // Swiss radial Soleillage (sunburst grinding rays) + circular Cerclage graining on flat gear webs, barrel caps & bridge tables
  let topFlatMask = smoothstep(0.78, 0.96, abs(N.z));
  let anglageBevelMask = smoothstep(0.84, 0.95, in.edgeMask);
  let soleillage = (sin(localAng * 64.0) * 0.46 + sin(localAng * 144.0 + radialR * 1.8) * 0.26 + cerclage * 0.28) * 0.5 + 0.5;
  if (matId == 0) {
    // Turned & Swiss Soleillage / Cerclage Horological Brass + Object-Locked Patek-style Hand-Engraved Arabesque Scrollwork (Gravure) + 45° Anglage Bevels
    let isBridgeGroup = (abs(groupId - 1.0) < 0.2 || abs(groupId - 3.0) < 0.2 || abs(groupId - 10.0) < 0.2 || abs(groupId - 12.0) < 0.2 || abs(groupId - 14.0) < 0.2);
    let innerWebMask = topFlatMask * (1.0 - anglageBevelMask) * smoothstep(0.08, 0.18, in.uv.x) * (1.0 - smoothstep(0.82, 0.92, in.uv.x));
    let scrollP = in.localPos.xy * 2.85;
    let scrollWaveA = sin(scrollP.x * 1.8 + sin(scrollP.y * 2.1) * 1.65);
    let scrollWaveB = cos(scrollP.y * 1.9 - sin(scrollP.x * 2.0) * 1.55);
    let arabesqueField = scrollWaveA * 0.55 + scrollWaveB * 0.45;
    let gravureGroove = (1.0 - smoothstep(0.04, 0.16, abs(arabesqueField))) * select(0.0, 1.0, isBridgeGroup) * innerWebMask;
    let finish = mix(turnGrain, mix(soleillage, cerclage, 0.38), topFlatMask * 0.78);
    baseColor = mix(baseColor * 0.88, baseColor * 1.14, finish * 0.36);
    baseColor = mix(baseColor, baseColor * 0.72, gravureGroove * 0.24);
    baseColor = mix(baseColor, vec3<f32>(1.0, 0.92, 0.66), anglageBevelMask * 0.44);
    rough = mix(clamp(rough + (finish - 0.5) * 0.05 + gravureGroove * 0.08, 0.09, 0.44), 0.04, anglageBevelMask);
    N = normalize(N + vec3<f32>(cos(scrollP.x * 1.8) * gravureGroove * 0.06, -sin(scrollP.y * 1.9) * gravureGroove * 0.06, 0.0));
  } else if (matId == 1 || matId == 2) {
    // Straight-Grained & Radial Soleillage Swiss Steel / Brushed Rhodium with Mirror Anglage Chamfers
    let linearGrain = sin(in.localPos.x * 42.0 + in.localPos.y * 10.0) * 0.5 + 0.5;
    let finish = mix(turnGrain, mix(soleillage, linearGrain, 0.48), topFlatMask * 0.72);
    baseColor = mix(baseColor * 0.89, baseColor * 1.12, finish * 0.32);
    baseColor = mix(baseColor, vec3<f32>(0.98, 0.99, 1.0), anglageBevelMask * 0.38);
    rough = mix(clamp(rough + (finish - 0.5) * 0.05, 0.08, 0.38), 0.04, anglageBevelMask);
  } else if (matId == 3) {
    // Phosphor Bronze / 18k Rose Gold Chaton & Wheel Alloy with Radial Soleillage + Circular Cerclage + Diamond Anglage
    let finish = mix(turnGrain, mix(soleillage, cerclage, 0.42), topFlatMask * 0.76);
    baseColor = mix(baseColor * 0.86, baseColor * 1.15, finish * 0.38);
    baseColor = mix(baseColor, vec3<f32>(1.0, 0.86, 0.70), anglageBevelMask * 0.40);
    rough = mix(clamp(rough + (finish - 0.5) * 0.05, 0.10, 0.44), 0.04, anglageBevelMask);
  } else if (matId == 4) {
    // Cast Bedplate & Skeleton Movement Cage: Subtle dark satin on bedplate/plinth, Rhodium Côtes de Genève on raised skeleton bridges, Warm Gilt Rose-Engine Perlage in recessed movement wells
    let genevaCoord = (in.worldPos.x + in.worldPos.y) * 1.35;
    let genevaRib = abs(sin(genevaCoord));
    let genevaBrush = sin((in.worldPos.x - in.worldPos.y) * 32.0 + sin(genevaCoord * 2.0) * 1.5) * 0.5 + 0.5;
    let geneva = genevaRib * 0.78 + genevaBrush * 0.22;
    let pCellA = fract(in.worldPos.xy * 0.48 + vec2<f32>(10.0)) - vec2<f32>(0.5);
    let pCellB = fract(in.worldPos.xy * 0.48 + vec2<f32>(10.5)) - vec2<f32>(0.5);
    let rA = length(pCellA);
    let rB = length(pCellB);
    let angA = atan2(pCellA.y, pCellA.x);
    let angB = atan2(pCellB.y, pCellB.x);
    let swirlA = (sin(rA * 38.0 + angA * 2.0) * 0.5 + 0.5) * smoothstep(0.52, 0.06, rA);
    let swirlB = (sin(rB * 38.0 - angB * 2.0) * 0.5 + 0.5) * smoothstep(0.52, 0.06, rB);
    let perlage = max(swirlA, swirlB);
    let isRecessedWell = (1.0 - smoothstep(6.8, 7.3, in.worldPos.z)) * smoothstep(0.9, 1.5, in.worldPos.z);
    let isRaisedBridge = smoothstep(7.2, 8.2, in.worldPos.z);
    let platePattern = mix(geneva, perlage, isRecessedWell * 0.90);
    let topMask = smoothstep(0.75, 0.95, abs(N.z));
    let bridgeOrWell = max(isRecessedWell, isRaisedBridge);
    let surfaceTint = mix(
      baseColor,
      mix(baseColor * 1.18, vec3<f32>(0.48, 0.37, 0.20), isRecessedWell * 0.68),
      bridgeOrWell
    );
    let patternAmp = mix(0.08, 0.36, bridgeOrWell);
    baseColor = mix(baseColor, surfaceTint * ((1.0 - patternAmp * 0.45) + patternAmp * platePattern), topMask * 0.88);
    baseColor = mix(baseColor, vec3<f32>(0.88, 0.90, 0.94), anglageBevelMask * mix(0.18, 0.44, bridgeOrWell));
    rough = mix(rough, clamp(rough - mix(0.03, 0.14, bridgeOrWell) * platePattern, 0.11, 0.48), topMask * 0.80);
    rough = mix(rough, 0.05, anglageBevelMask * bridgeOrWell);
    let genevaSlope = cos(genevaCoord) * mix(0.018, 0.11, isRaisedBridge) * (1.0 - isRecessedWell) * topMask;
    let perlageSlope = (swirlA - swirlB) * isRecessedWell * topMask * 0.12;
    N = normalize(N + vec3<f32>(genevaSlope + perlageSlope, genevaSlope - perlageSlope, 0.0));
  } else if (matId == 5) {
    // Central Sun (Emissive Heliocentric Sphere + Armillary Equator/Colure Filigree)
    let lat = abs(in.localPos.z);
    let lon = atan2(in.localPos.y, in.localPos.x);
    let gran = valueNoise3(in.localPos * 10.5 + vec3<f32>(u_scene.cameraPos.w * 0.4, 0.0, 0.0)) * 0.65 +
               valueNoise3(in.localPos * 22.0 - vec3<f32>(0.0, u_scene.cameraPos.w * 0.6, 0.0)) * 0.35;
    let limb = pow(nDotV, 0.48);
    let eqBand = smoothstep(0.055, 0.018, lat);
    let meridian = smoothstep(0.036, 0.010, abs(sin(lon * 6.0)));
    let filigree = max(eqBand, meridian * 0.55);
    let sunCore = mix(vec3<f32>(1.0, 0.46, 0.08), vec3<f32>(1.0, 0.96, 0.76), limb * 0.72 + gran * 0.28);
    baseColor = mix(sunCore, vec3<f32>(0.92, 0.72, 0.30), filigree * 0.60);
    emissive = sunCore * (1.75 - filigree * 0.55);
    rough = 0.18;
    metal = 0.25;
  } else if (matId == 6) {
    // Earth Tellurian Globe (Continents, Oceans, Polar Caps, Brass Graticule & Rayleigh Blue Atmosphere)
    let pNorm = normalize(in.localPos);
    let lat = asin(clamp(pNorm.z, -1.0, 1.0));
    let lon = atan2(pNorm.y, pNorm.x);
    let nCont = valueNoise3(pNorm * 3.4 + vec3<f32>(1.7, 2.3, 0.5)) * 0.65 +
                valueNoise3(pNorm * 7.8 + vec3<f32>(4.1, 1.2, 3.3)) * 0.35;
    let isLand = smoothstep(0.49, 0.54, nCont);
    let isIce = smoothstep(1.12, 1.26, abs(lat));
    let oceanCol = select(vec3<f32>(0.09, 0.32, 0.62), vec3<f32>(0.14, 0.40, 0.70), isLightTheme);
    let landCol = select(vec3<f32>(0.28, 0.54, 0.30), vec3<f32>(0.36, 0.58, 0.34), isLightTheme);
    var globeCol = mix(oceanCol, landCol, isLand);
    globeCol = mix(globeCol, vec3<f32>(0.94, 0.96, 0.99), isIce);

    let eqLine = smoothstep(0.030, 0.009, abs(lat));
    let tropicLine = smoothstep(0.024, 0.007, abs(abs(lat) - 0.4091));
    let merLine = smoothstep(0.028, 0.008, abs(sin(lon * 6.0)) * cos(lat));
    let grat = max(eqLine, max(tropicLine * 0.8, merLine * 0.6));
    baseColor = mix(globeCol, vec3<f32>(0.92, 0.76, 0.38), grat * 0.78);
    rough = mix(0.20, 0.52, isLand);
    metal = grat * 0.72;
    let atmoRim = pow(1.0 - nDotV, 2.6);
    emissive = emissive + vec3<f32>(0.16, 0.48, 0.96) * atmoRim * (0.28 + 0.72 * nDotSun);
  } else if (matId == 7) {
    // Mars Globe (Ochre/Rust Albedo + Polar Ice + Graticule + Dusty Limb Rim)
    let pNorm = normalize(in.localPos);
    let lat = asin(clamp(pNorm.z, -1.0, 1.0));
    let nM = valueNoise3(pNorm * 4.5 + vec3<f32>(2.1, 0.8, 5.2));
    let ice = smoothstep(1.25, 1.38, abs(lat));
    let eqLine = smoothstep(0.032, 0.010, abs(lat));
    baseColor = mix(vec3<f32>(0.70, 0.30, 0.16), vec3<f32>(0.88, 0.50, 0.28), nM);
    baseColor = mix(baseColor, vec3<f32>(0.95, 0.93, 0.89), ice);
    baseColor = mix(baseColor, vec3<f32>(0.92, 0.76, 0.40), eqLine * 0.68);
    let marsRim = pow(1.0 - nDotV, 2.8);
    emissive = emissive + vec3<f32>(0.82, 0.38, 0.18) * marsRim * (0.22 + 0.55 * nDotSun);
  } else if (matId == 8) {
    // Venus Banded Sulfuric Atmosphere + Golden Haze Rim
    let pNorm = normalize(in.localPos);
    let lat = asin(clamp(pNorm.z, -1.0, 1.0));
    let bands = sin(lat * 14.0 + valueNoise3(pNorm * 4.0) * 2.5) * 0.5 + 0.5;
    let eqLine = smoothstep(0.032, 0.010, abs(lat));
    baseColor = mix(vec3<f32>(0.80, 0.68, 0.42), vec3<f32>(0.95, 0.86, 0.64), bands);
    baseColor = mix(baseColor, vec3<f32>(0.58, 0.46, 0.22), eqLine * 0.5);
    let venusRim = pow(1.0 - nDotV, 2.5);
    emissive = emissive + vec3<f32>(0.92, 0.78, 0.44) * venusRim * (0.25 + 0.55 * nDotSun);
  } else if (matId == 9 || matId == 10) {
    // Mercury (9), Moon & Satellites (10)
    let pNorm = normalize(in.localPos);
    let lat = asin(clamp(pNorm.z, -1.0, 1.0));
    let crater = valueNoise3(pNorm * 6.5 + vec3<f32>(f32(matId) * 1.3));
    let eqLine = smoothstep(0.038, 0.012, abs(lat));
    baseColor = mix(baseColor * 0.80, baseColor * 1.16, crater);
    baseColor = mix(baseColor, vec3<f32>(0.86, 0.70, 0.36), eqLine * 0.55);
  } else if (matId == 11) {
    // Antikythera Spiral Saros, Exeligmos/Callippic Sub-Dials & Parapegma Champlevé Engraved Dial Face with Physical Micro-Bevel Normal Relief
    let rawR = length(in.localPos.xy);
    let r = rawR / select(1.0, 18.8, rawR > 1.2);
    let ang = atan2(in.localPos.y, in.localPos.x);
    let isMainSarosDial = (length(in.worldPos.xy - vec2<f32>(104.0, 12.0)) < 20.0) && (in.worldPos.z > 46.0);

    // Subsidiary Exeligmos (3-sector at [-8.0, -8.0]) and Callippic (4-sector at [+8.0, -8.0]) sub-dial engravings (ONLY on main Saros dial!)
    let subExVec = in.localPos.xy - vec2<f32>(-8.0, -8.0);
    let subCaVec = in.localPos.xy - vec2<f32>(8.0, -8.0);
    let rEx = length(subExVec);
    let rCa = length(subCaVec);
    let angEx = atan2(subExVec.y, subExVec.x);
    let angCa = atan2(subCaVec.y, subCaVec.x);
    let subCutout = select(0.0, max(smoothstep(2.70, 2.45, rEx), smoothstep(2.70, 2.45, rCa)), isMainSarosDial);
    let mainMask = 1.0 - subCutout;

    let subGuilloche = sin(min(rEx, rCa) * 48.0) * 0.5 + 0.5;
    let mainGuilloche = (sin(r * 180.0 + sin(ang * 24.0) * 0.8) * 0.55 + cos(ang * 72.0) * 0.45) * 0.5 + 0.5;
    let dialGuilloche = mix(mainGuilloche, subGuilloche, subCutout);

    let outerTick = smoothstep(0.76, 0.92, r) * (1.0 - smoothstep(0.95, 0.98, r)) * mainMask;
    let tick36 = smoothstep(0.08, 0.02, abs(sin(ang * 18.0)));
    let tick12 = smoothstep(0.05, 0.015, abs(sin(ang * 6.0))) * smoothstep(0.75, 0.96, r) * mainMask;
    let ring1 = smoothstep(0.018, 0.005, abs(r - 0.95));
    let ring2 = smoothstep(0.018, 0.005, abs(r - 0.76)) * mainMask;
    let ring3 = smoothstep(0.014, 0.004, abs(r - select(0.62, 0.46, isMainSarosDial))) * mainMask;
    let spiralMaxR = select(0.60, 0.45, isMainSarosDial);
    let spiralFreq = select(11.5, 19.8, isMainSarosDial);
    let spiralPhase = fract(r * spiralFreq - ang / 6.283185 + 1.0);
    let spiralLine = smoothstep(0.09, 0.022, abs(spiralPhase - 0.5)) *
                     smoothstep(0.22, 0.25, r) * (1.0 - smoothstep(spiralMaxR - 0.02, spiralMaxR, r)) * mainMask;
    // 223-month Saros cell divisions + Σ/Η eclipse glyph marks inside the 4-turn spiral track
    let spiralCells = smoothstep(0.055, 0.014, abs(sin(ang * 28.0))) *
                      smoothstep(0.23, 0.26, r) * (1.0 - smoothstep(spiralMaxR - 0.02, spiralMaxR, r)) * 0.52 * mainMask;
    let glyphDots = smoothstep(0.18, 0.05, abs(spiralPhase - 0.25)) *
                    smoothstep(0.08, 0.02, abs(sin(ang * 7.0 + r * 12.0))) *
                    smoothstep(0.24, 0.27, r) * (1.0 - smoothstep(spiralMaxR - 0.03, spiralMaxR - 0.01, r)) * 0.75 * mainMask;

    let exTicks = smoothstep(0.09, 0.02, abs(sin(angEx * 6.0))) * smoothstep(1.90, 2.08, rEx) * (1.0 - smoothstep(2.26, 2.35, rEx)) * 0.55;
    let caTicks = smoothstep(0.09, 0.02, abs(sin(angCa * 8.0))) * smoothstep(1.90, 2.08, rCa) * (1.0 - smoothstep(2.26, 2.35, rCa)) * 0.55;
    let exSectors = (smoothstep(0.11, 0.028, abs(rEx - 2.32)) + smoothstep(0.09, 0.022, abs(rEx - 0.52)) + exTicks + smoothstep(0.045, 0.012, abs(sin(angEx * 1.5))) * smoothstep(2.30, 2.15, rEx) * smoothstep(0.50, 0.64, rEx)) * smoothstep(2.55, 2.38, rEx);
    let caSectors = (smoothstep(0.11, 0.028, abs(rCa - 2.32)) + smoothstep(0.09, 0.022, abs(rCa - 0.52)) + caTicks + smoothstep(0.045, 0.012, abs(sin(angCa * 2.0))) * smoothstep(2.30, 2.15, rCa) * smoothstep(0.50, 0.64, rCa)) * smoothstep(2.55, 2.38, rCa);
    let subInk = select(0.0, (exSectors + caSectors) * 0.88, isMainSarosDial);

    let inkMask = clamp(
      outerTick * tick36 * 0.65 + tick12 * 0.9 + ring1 + ring2 + ring3 * 0.75 + spiralLine * 0.85 + spiralCells + glyphDots + subInk,
      0.0,
      1.0
    );
    let inkCol = select(vec3<f32>(0.95, 0.81, 0.48), vec3<f32>(0.15, 0.13, 0.10), isLightTheme);
    let bgDial = baseColor * (0.94 + 0.12 * dialGuilloche);
    baseColor = mix(bgDial, inkCol, inkMask * 0.94);
    rough = mix(clamp(rough - 0.04 * dialGuilloche, 0.12, 0.36), 0.10, inkMask);
    metal = mix(metal, 0.90, inkMask);
    let grooveSlopeR = (cos(r * 47.12 - ang) * spiralLine * 0.16 + sin(r * 180.0) * (1.0 - inkMask) * 0.04) * mainMask;
    let grooveSlopeA = cos(ang * 28.0) * (spiralCells + outerTick * tick36) * 0.14 * mainMask;
    N = normalize(N + vec3<f32>(cos(ang) * grooveSlopeR - sin(ang) * grooveSlopeA, sin(ang) * grooveSlopeR + cos(ang) * grooveSlopeA, 0.0) * topFlatMask);
    emissive = emissive + vec3<f32>(1.0, 0.68, 0.22) * glyphDots * (0.18 + 0.85 * u_scene.params.w);
  } else if (matId == 12) {
    // Horological Ruby Jewel Bearing (Subsurface Crimson Glow)
    let rimRuby = pow(1.0 - nDotV, 1.8);
    emissive = emissive + vec3<f32>(0.55, 0.05, 0.10) * rimRuby;
  } else if (matId == 13) {
    // Thermally Blued Horological Steel (Iridescent Cobalt-Indigo Specular Shift)
    baseColor = mix(vec3<f32>(0.09, 0.22, 0.56), vec3<f32>(0.24, 0.46, 0.84), pow(1.0 - nDotV, 1.5));
  } else if (matId == 14) {
    // Hellenistic Patinated Corinthian Bronze with Incised Greek Key / Degree Flute Relief
    let pat = valueNoise3(in.localPos * 1.8);
    let fluteWave = sin(localAng * 36.0) * 0.5 + 0.5;
    let meanderBand = smoothstep(0.08, 0.02, abs(fract(in.localPos.z * 0.55) - 0.5)) * smoothstep(0.85, 0.25, abs(N.z));
    let verdigris = (1.0 - nDotV) * 0.22 + pat * 0.14;
    baseColor = mix(baseColor * (0.86 + 0.12 * fluteWave), vec3<f32>(0.34, 0.49, 0.43), verdigris);
    baseColor = mix(baseColor, vec3<f32>(0.90, 0.74, 0.38), meanderBand * 0.48 + anglageBevelMask * 0.35);
    rough = clamp(rough + pat * 0.08 - meanderBand * 0.08, 0.14, 0.52);
    N = normalize(N + vec3<f32>(-sin(localAng) * cos(localAng * 36.0) * 0.09, cos(localAng) * cos(localAng * 36.0) * 0.09, sin(in.localPos.z * 3.45) * meanderBand * 0.10));
  } else if (matId == 15) {
    // Jupiter Banded Gas Giant + Great Red Spot + Brass Equator
    let pNorm = normalize(in.localPos);
    let lat = asin(clamp(pNorm.z, -1.0, 1.0));
    let lon = atan2(pNorm.y, pNorm.x);
    let turb = valueNoise3(pNorm * 6.0 + vec3<f32>(0.4, 1.2, 2.8)) * 0.18;
    let zone = sin((lat + turb) * 18.0) * 0.5 + 0.5;
    let beltCol = vec3<f32>(0.68, 0.42, 0.26);
    let zoneCol = vec3<f32>(0.94, 0.86, 0.70);
    var jupCol = mix(beltCol, zoneCol, zone);
    let dLat = (lat + 0.38) * 3.8;
    let dLon = sin(lon - 0.8) * 2.1;
    let grs = 1.0 - smoothstep(0.28, 0.65, length(vec2<f32>(dLon, dLat)));
    jupCol = mix(jupCol, vec3<f32>(0.86, 0.26, 0.14), grs * 0.88);
    let eqLine = smoothstep(0.030, 0.009, abs(lat));
    baseColor = mix(jupCol, vec3<f32>(0.90, 0.74, 0.38), eqLine * 0.68);
    rough = 0.34;
    metal = eqLine * 0.60;
  } else if (matId == 16) {
    // Saturn Banded Gas Giant + Polar Hexagon + Brass Equatorial Graticule
    let pNorm = normalize(in.localPos);
    let lat = asin(clamp(pNorm.z, -1.0, 1.0));
    let turb = valueNoise3(pNorm * 5.2 + vec3<f32>(2.4, 0.7, 1.9)) * 0.12;
    let bands = sin((lat + turb) * 22.0) * 0.5 + 0.5;
    let sBelt = vec3<f32>(0.76, 0.62, 0.38);
    let sZone = vec3<f32>(0.95, 0.88, 0.68);
    var satCol = mix(sBelt, sZone, bands);
    let polarCap = smoothstep(1.10, 1.35, abs(lat));
    satCol = mix(satCol, vec3<f32>(0.54, 0.58, 0.52), polarCap * 0.55);
    let eqLine = smoothstep(0.030, 0.009, abs(lat));
    baseColor = mix(satCol, vec3<f32>(0.92, 0.76, 0.40), eqLine * 0.68);
    rough = 0.32;
    metal = eqLine * 0.60;
    let satRim = pow(1.0 - nDotV, 2.8);
    emissive = emissive + vec3<f32>(0.88, 0.76, 0.46) * satRim * (0.20 + 0.45 * nDotSun);
  } else if (matId == 17) {
    // Saturn Concentric Ring System (C Ring, B Ring, Cassini Division, A Ring, Encke Gap)
    let u = clamp(in.uv.x, 0.0, 1.0);
    // Open Cassini Division gap (u in [0.575, 0.618]) on horizontal faces
    if (u > 0.575 && u < 0.618 && abs(N.z) > 0.4) {
      discard;
    }
    let ringlets = sin(u * 220.0) * 0.5 + 0.5;
    let fineRings = sin(u * 540.0) * 0.5 + 0.5;
    let cRingMask = 1.0 - smoothstep(0.18, 0.24, u);
    let bRingMask = smoothstep(0.20, 0.26, u) * (1.0 - smoothstep(0.54, 0.575, u));
    let aRingMask = smoothstep(0.618, 0.65, u);
    let enckeGap = 1.0 - 0.65 * (1.0 - smoothstep(0.004, 0.016, abs(u - 0.86)));

    let cCol = vec3<f32>(0.58, 0.52, 0.42);
    let bCol = mix(vec3<f32>(0.86, 0.76, 0.56), vec3<f32>(0.98, 0.92, 0.76), ringlets * 0.65 + fineRings * 0.35);
    let aCol = mix(vec3<f32>(0.76, 0.68, 0.52), vec3<f32>(0.90, 0.82, 0.66), ringlets * 0.50) * enckeGap;
    // Engraved brass boundary bezels at inner, Cassini, and outer edges
    let brassEdge = smoothstep(0.025, 0.005, u) +
                    smoothstep(0.025, 0.005, 1.0 - u) +
                    smoothstep(0.018, 0.004, abs(u - 0.568)) +
                    smoothstep(0.018, 0.004, abs(u - 0.625));
    var ringCol = cCol * cRingMask + bCol * bRingMask + aCol * aRingMask;
    ringCol = mix(ringCol, vec3<f32>(0.92, 0.76, 0.38), clamp(brassEdge, 0.0, 1.0));
    baseColor = ringCol;
    rough = mix(0.28, 0.16, clamp(brassEdge, 0.0, 1.0));
    metal = mix(0.28, 0.88, clamp(brassEdge, 0.0, 1.0));
    emissive = emissive + ringCol * (0.28 + 0.24 * abs(dot(N, sunL)));
  } else if (matId == 18) {
    // Engraved Brass & Guilloché Enamel Planisphere / Astrolabe Floor Plate (r = 7.8 .. 46.4)
    let r = length(in.localPos.xy);
    let ang = atan2(in.localPos.y, in.localPos.x);
    let rNorm = clamp((r - 7.8) / (46.4 - 7.8), 0.0, 1.0);
    // Concentric altitude almucantar circles + Tropic of Cancer / Celestial Equator / Tropic of Capricorn + 24-hour meridians
    let almucantars = smoothstep(0.026, 0.007, abs(fract(rNorm * 8.0) - 0.5));
    let tropicRings = smoothstep(0.015, 0.004, abs(rNorm - 0.36)) +
                      smoothstep(0.016, 0.004, abs(rNorm - 0.58)) +
                      smoothstep(0.015, 0.004, abs(rNorm - 0.84));
    let meridians24 = smoothstep(0.022, 0.005, abs(sin(ang * 12.0))) * smoothstep(0.08, 0.15, rNorm);
    let cardinalCross = (smoothstep(0.32, 0.07, abs(in.localPos.x)) + smoothstep(0.32, 0.07, abs(in.localPos.y))) * smoothstep(0.04, 0.10, rNorm);
    let degTicks72 = smoothstep(0.048, 0.012, abs(sin(ang * 36.0))) * smoothstep(0.88, 0.93, rNorm);
    // Stereographic Horizon Arc & Eccentric Zodiac Rete Circle offset toward +X (+5.5) with 12 zodiac house radial divisions
    let horizonArcDist = abs(length(in.localPos.xy - vec2<f32>(-11.0, 0.0)) - 34.5);
    let horizonArc = smoothstep(0.48, 0.12, horizonArcDist) * smoothstep(-2.0, 4.0, in.localPos.x);
    let reteVec = in.localPos.xy - vec2<f32>(5.5, 0.0);
    let reteDist = abs(length(reteVec) - 28.0);
    let reteAng = atan2(reteVec.y, reteVec.x);
    let reteRing = smoothstep(0.55, 0.14, reteDist) +
                   smoothstep(0.042, 0.011, abs(sin(reteAng * 6.0))) * smoothstep(1.6, 0.4, reteDist);
    // Fine rose-engine guilloché basketweave in the dark enamel planisphere field
    let guilloche = (sin(r * 24.0 + sin(ang * 36.0) * 1.45) * 0.65 + cos(r * 38.0 - cos(ang * 48.0) * 1.15) * 0.35) * 0.5 + 0.5;
    let lineMask = clamp(
      almucantars * 0.68 + tropicRings * 0.94 + meridians24 * 0.60 + cardinalCross * 0.85 + degTicks72 * 0.88 + horizonArc * 0.78 + reteRing * 0.96,
      0.0,
      1.0
    );
    let baseEnamel = select(vec3<f32>(0.09, 0.11, 0.15), vec3<f32>(0.88, 0.85, 0.78), isLightTheme);
    let plateBg = baseEnamel * (0.92 + 0.16 * guilloche);
    let brassInk = select(vec3<f32>(0.94, 0.77, 0.38), vec3<f32>(0.56, 0.42, 0.16), isLightTheme);
    baseColor = mix(plateBg, brassInk, lineMask);
    rough = mix(0.32 - 0.07 * guilloche, 0.10, lineMask);
    metal = mix(0.38, 0.94, lineMask);
    let astroSlopeR = cos(rNorm * 50.26) * almucantars * 0.12 + cos(r * 24.0) * (1.0 - lineMask) * 0.06;
    let astroSlopeA = cos(ang * 24.0) * meridians24 * 0.12;
    N = normalize(N + vec3<f32>(cos(ang) * astroSlopeR - sin(ang) * astroSlopeA, sin(ang) * astroSlopeR + cos(ang) * astroSlopeA, 0.0) * topFlatMask);
  }

  nDotV = max(dot(N, V), 0.001);
  nDotKey = max(dot(N, keyL), 0.0);
  nDotSun = max(dot(N, sunL), 0.0);

  // 5. Dual Lighting: Heliocentric Point Light (from Sun) + Studio Key/Fill Rig + Multi-Light Studio Reflections
  var sunOcclusion = 1.0;
  let sunR = u_scene.sunPosAndRadius.w;
  if (matId == 6) {
    sunOcclusion = sphereShadowOcclusion(
      in.worldPos,
      u_scene.sunPosAndRadius.xyz,
      sunR,
      u_scene.moonPosAndRadius.xyz,
      u_scene.moonPosAndRadius.w * 1.35
    );
  } else if (matId == 10) {
    sunOcclusion = sphereShadowOcclusion(
      in.worldPos,
      u_scene.sunPosAndRadius.xyz,
      sunR,
      u_scene.earthPosAndRadius.xyz,
      u_scene.earthPosAndRadius.w * 1.15
    );
  }

  let isPlanet = (matId >= 6 && matId <= 10) || (matId == 15) || (matId == 16);
  let sunAtten = 1.0 / (1.0 + 0.000045 * sunDist * sunDist);

  // Cook-Torrance GGX Specular + Schlick Fresnel
  let f0 = mix(vec3<f32>(0.04), baseColor, metal);
  let H_key = normalize(keyL + V);
  let H_sun = normalize(sunL + V);
  let nDotH_key = max(dot(N, H_key), 0.0);
  let nDotH_sun = max(dot(N, H_sun), 0.0);

  let a2 = rough * rough * rough * rough;
  let denomKey = (nDotH_key * nDotH_key * (a2 - 1.0) + 1.0);
  let D_key = a2 / max(3.14159 * denomKey * denomKey, 0.0001);
  let F_key = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - max(dot(H_key, V), 0.0), 5.0);
  let specKey = D_key * F_key * (0.28 / max(nDotV, 0.12));

  let denomSun = (nDotH_sun * nDotH_sun * (a2 - 1.0) + 1.0);
  let D_sun = a2 / max(3.14159 * denomSun * denomSun, 0.0001);
  let F_sun = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - max(dot(H_sun, V), 0.0), 5.0);
  let specSun = D_sun * F_sun * (0.30 / max(nDotV, 0.12));

  let kD = (vec3<f32>(1.0) - F_key) * (1.0 - metal);

  let upFactor = N.z * 0.5 + 0.5;
  let ambTop = select(vec3<f32>(0.24, 0.25, 0.29), vec3<f32>(0.92, 0.90, 0.86), isLightTheme);
  let ambBot = select(vec3<f32>(0.08, 0.08, 0.10), vec3<f32>(0.58, 0.56, 0.52), isLightTheme);
  let ambientEnv = mix(ambBot, ambTop, upFactor);
  let R = reflect(-V, N);
  let studioRefl = evaluateStudioEnvironment(R, isLightTheme);
  let envSpec = F_key * studioRefl * (1.0 - rough * 0.52);

  let shadowFactor = mix(0.32, 1.0, keyShadow);
  let keyWeight = select(0.92, 0.26, isPlanet);
  let sunWeight = select(0.68, 1.85, isPlanet);

  let keyLightCol = select(vec3<f32>(1.0, 0.97, 0.92), vec3<f32>(1.05, 1.02, 0.98), isLightTheme);
  let sunLightCol = vec3<f32>(1.0, 0.88, 0.60) * sunAtten * sunOcclusion;

  // Translucent backscatter on Saturn's rings (matId == 17) so both top & bottom faces glow warmly in sunlight
  let ringBacklight = select(0.0, abs(dot(N, sunL)) * 0.65, matId == 17);

  var radiance =
    (kD * baseColor / 3.14159 + specKey) * keyLightCol * nDotKey * shadowFactor * keyWeight +
    (kD * baseColor / 3.14159 + specSun) * sunLightCol * max(nDotSun, ringBacklight) * sunWeight +
    baseColor * ambientEnv * (1.0 - metal * 0.62) * select(0.68, 0.24, isPlanet) +
    envSpec * select(0.82, 0.95, isLightTheme) +
    emissive;

  if (isPlanet && sunOcclusion < 0.95 && nDotSun > 0.0) {
    let umbraDepth = 1.0 - sunOcclusion;
    radiance = mix(radiance, baseColor * vec3<f32>(0.55, 0.16, 0.08), umbraDepth * 0.65);
  }

  // 6. Micro-Cavity Crease Occlusion + Swiss 45° Anglage Mirror-Bevel Specular Catch-Lights
  let normalJump = length(dNdx) + length(dNdy);
  let crease = smoothstep(0.34, 0.76, normalJump);
  let meshEdge = smoothstep(0.45 - edgeFwidth, 0.45 + edgeFwidth, in.edgeMask);
  let anglageEdge = smoothstep(0.86, 0.96, in.edgeMask);
  let silhouette = smoothstep(0.22, 0.05, nDotV);

  // Darken interior creases for crisp gear-tooth and engraving relief
  radiance *= mix(1.0, select(0.46, 0.52, isLightTheme), crease * 0.68);
  // Bright metallic catch-light along chamfered bridge anglage bevels & gear tips
  let anglageGlint = pow(max(nDotH_key, nDotH_sun), 18.0) * anglageEdge * metal;
  let bevelCatch = (meshEdge * (0.35 + 0.65 * max(nDotKey, nDotSun)) + anglageGlint * 1.15) * (0.35 + 0.65 * metal);
  radiance += mix(vec3<f32>(1.0, 0.95, 0.82), baseColor * 1.45, 0.55) * bevelCatch * 0.38;
  // Subtle technical silhouette ink
  let silInk = select(radiance * 0.45, vec3<f32>(0.14, 0.13, 0.12), isLightTheme);
  radiance = mix(radiance, silInk, silhouette * select(0.28, 0.45, isLightTheme));

  // 7. Surgical Rim-Light & Bevel Contour Highlight on Hover / Selection (Preserves 100% of Interior Material Colors!)
  let isHovered = abs(groupId - u_scene.params.x) < 0.25 && u_scene.params.x > 0.5;
  let isSelected = abs(groupId - u_scene.params.y) < 0.25 && u_scene.params.y > 0.5;
  if (isHovered || isSelected) {
    let rim = pow(1.0 - nDotV, 2.6);
    let contour = clamp(rim * 0.85 + meshEdge * 0.55 + crease * 0.35, 0.0, 1.0);
    let accent = select(vec3<f32>(1.0, 0.80, 0.34), vec3<f32>(0.88, 0.50, 0.08), isLightTheme);
    let strength = select(0.42, 0.65, isHovered);
    radiance = radiance * 1.06 + accent * contour * strength;
  }

  let mapped = acesFilm(radiance * 1.12);
  let finalRgb = pow(mapped, vec3<f32>(0.90));
  return vec4<f32>(finalRgb, 1.0);
}

// ============================================================================
// PASS 3: Forward Transparent Overlays (Solar Corona, Eclipse Cone, Orbit & Retrograde Ribbons)
// ============================================================================
struct OverlayInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
  @location(3) uv: vec2<f32>,
};

struct OverlayOutput {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) color: vec4<f32>,
  @location(3) uv: vec2<f32>,
};

@vertex
fn vs_overlay(in: OverlayInput) -> OverlayOutput {
  var out: OverlayOutput;
  out.clipPos = u_scene.viewProj * vec4<f32>(in.position, 1.0);
  out.worldPos = in.position;
  out.worldNormal = in.normal;
  out.color = in.color;
  out.uv = in.uv;
  return out;
}

@fragment
fn fs_overlay(in: OverlayOutput) -> @location(0) vec4<f32> {
  let V = normalize(u_scene.cameraPos.xyz - in.worldPos);
  let N = normalize(in.worldNormal);
  let kind = in.uv.y;
  var alpha = in.color.a;
  var rgb = in.color.rgb;

  if (kind < 0.5) {
    // Orbit ring or velocity ribbon with bright core spine
    let edge = 1.0 - abs(in.uv.x * 2.0 - 1.0);
    let core = pow(clamp(edge, 0.0, 1.0), 2.2);
    alpha *= smoothstep(0.0, 0.42, edge);
    rgb = mix(rgb, vec3<f32>(1.0, 0.97, 0.88), core * 0.35);
  } else if (kind < 1.5) {
    // Volumetric Solar Corona + 16-Point Diffraction Rays (uv.x in [0, 1], uv.y in [0.8, 1.2])
    let dUv = vec2<f32>(in.uv.x - 0.5, (in.uv.y - 1.0) * 2.5);
    let r = length(dUv) * 2.0;
    let ang = atan2(dUv.y, dUv.x);
    let coreGlow = exp(-r * 2.4) * (1.0 - smoothstep(0.75, 1.0, r));
    let rays8 = pow(abs(cos(ang * 4.0 + u_scene.cameraPos.w * 0.18)), 12.0) * exp(-r * 1.8) * 0.58;
    let rays16 = pow(abs(cos(ang * 8.0 - u_scene.cameraPos.w * 0.12)), 18.0) * exp(-r * 2.2) * 0.36;
    let totalGlow = clamp((coreGlow + rays8 + rays16) * (1.0 - smoothstep(0.84, 1.0, r)), 0.0, 1.0);
    alpha *= totalGlow;
    rgb = mix(vec3<f32>(1.0, 0.72, 0.26), vec3<f32>(1.0, 0.98, 0.88), clamp(coreGlow * 1.4 + rays8 * 0.5, 0.0, 1.0));
  } else {
    // 3D Eclipse Shadow Cones
    let graze = smoothstep(0.02, 0.28, abs(dot(N, V)));
    let axialFade = sin(clamp(in.uv.x, 0.0, 1.0) * 3.14159);
    alpha *= graze * axialFade;
  }

  return vec4<f32>(rgb * alpha, alpha);
}
